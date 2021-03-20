import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
#from scipy.misc import imread, imresize
import imageio
from skimage.transform import resize
#from matplotlib.pyplot import imread
#from skimage.transform import resize
#from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def caption_greedy(encoder, decoder, image_path, word_map, max_length):
    vocab_size = len(word_map)

    # Read image and process
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    #print(image)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    print(encoder_out.shape)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    #print(encoder_out)
    num_pixels = encoder_out.size(1)

    mean_encoder_out = encoder_out.mean(dim=1)
    h = decoder.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
    c = decoder.init_c(mean_encoder_out)

    # keys of start and end tokens
    start_key = next((k for k in rev_word_map if rev_word_map[k] == '<start>'), None)
    end_key = next((k for k in rev_word_map if rev_word_map[k] == '<end>'), None)
    start_key = torch.tensor(start_key).to(device)

    seq = []
    seq_alphas = []
    #input_embedding = decoder.embedding(idx_start_token)

    input_embedding = decoder.embedding(start_key)
    input_embedding = torch.unsqueeze(input_embedding, 0).to(device)

    for t in range(max_length):
        
        attention_weighted_encoding, alpha = decoder.attention(encoder_out, # image information
                                                            h) # image + language info
        
        seq_alphas.append(alpha)

        # gate decides what information in the hidden state is important
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)

        # multiplies with image attention
        attention_weighted_encoding = gate * attention_weighted_encoding
        #print('attention_weighted_encoding', attention_weighted_encoding.shape)
        
        # print(input_embedding.shape)
        # feeds embedding of current word and attention
        h, c = decoder.decode_step(
            torch.cat([input_embedding, attention_weighted_encoding], dim=1),
            (h, c))  # (batch_size_t, decoder_dim)
        
        scores = decoder.fc(h)  # (s, vocab_size) # fully connected layer maps to vocab size (logits)
        scores = F.log_softmax(scores, dim=1) # turn logits into probabilities
        # highest is the index of the biggest score
        # make sure highest is an INDEX!!!
        #print(scores)
        highest = torch.argmax(scores) # return highest probability
        #print(highest)
        word_string = rev_word_map[highest.item()] # get word from word map
        
        seq.append(word_string)

        input_embedding = decoder.embedding(highest).unsqueeze(0)

        #print(word_string)
        if word_string == '<end>':
            break
            
    return seq, seq_alphas
    
def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # BEAM SEARCH!

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        print('kprev', type(k_prev_words))
        embeddings = decoder.embedding(k_prev_words).squeeze(1) # (s, embed_dim)
        print('embeddings', type(embeddings))

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size) # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = (gate * awe) # awe = attention weight encoder

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size) # fully connected layer maps to vocab size (logits)
        scores = F.log_softmax(scores, dim=1) # probabilities

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        prev_word_inds = prev_word_inds.type(torch.LongTensor)

        next_word_inds = top_k_words % vocab_size  # (s)
        next_word_inds = next_word_inds.type(torch.LongTensor)

        print('HELLOOOOOOOOO')

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds].cpu(), next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

        # change to dtype=long tensor
        seqs = seqs.type(torch.LongTensor)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        print('after', type(k_prev_words))

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, seq_alphas, rev_word_map, smooth=True):
    # subplot settings
    num_col = 4
    num_row = len(seq_alphas) // num_col + 2
    subplot_size = 4

    fig = plt.figure(dpi=100)
    fig.set_size_inches(subplot_size * num_col, subplot_size * (num_row + 1))
    fig.set_facecolor('white')

    # generate caption results
    print("Visualizing results...")
    plt.subplot2grid((num_row, num_col), (0, 0))
    image = imageio.imread(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot2grid((num_row, num_col), (0, 1), colspan=num_col-1)
    plt.text(0, 0.5, seq, fontsize=16)
    plt.axis('off')

    # visualize attention weights
    print("Visualizing attention weights...\n")
    for i in range(len(seq_alphas)):
        plt.subplot2grid((num_row, num_col), (i // num_col + 1, i % num_col))
        #attention = seq_alphas[i][2].data.cpu().numpy()
        attention = seq_alphas[i].data.cpu().numpy()
        attention = skimage.transform.pyramid_expand(attention.reshape(14, 14), upscale=32, sigma=20)
        #attention = skimage.transform.pyramid_expand(attention, upscale=32, sigma=20)
        plt.imshow(image)
        plt.imshow(attention, alpha=0.8)
        #plt.text(0, 0, seq_alphas[i], fontsize=16, color='black', backgroundcolor='white')
        #plt.set_cmap(cm.Greys_r)
        #plt.axis('off')

    # fig.tight_layout()
    fig_output_dir = '/home/gusviloca@GU.GU.SE/aics-vizwiz/figs'
    image_file = image_path.split('/')[-1]
    image_file = image_file[:-4]
    plt.savefig('/home/gusviloca@GU.GU.SE/aics-vizwiz/figs/{}.png'.format(image_file), bbox_inches="tight")
    #plt.savefig("outputs/vis/{}/{}.png".format(dir_name, outname), bbox_inches="tight")
    fig.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    # parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    #parser.add_argument('--max_len', '-max', help='maximum caption length')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    max_len = 50

    # NEED TO EDIT FOR GREEDY
    # Encode, decode with attention and beam search
    seq, seq_alphas = caption_greedy(encoder, decoder, args.img, word_map, max_len)
    print('Generated Caption:', seq)
    #alphas = torch.FloatTensor(alphas)
    
    #decoded_seq = []
    #for item in seq:
    #    decoded_seq.append(rev_word_map[item])
    #print(decoded_seq)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, seq_alphas, rev_word_map, args.smooth)

    """
    Command

    python caption.py --img /home/gusviloca@GU.GU.SE/aics-vizwiz/data/vizwiz_images/VizWiz_train_00008121.jpg --model /home/gusviloca@GU.GU.SE/aics-vizwiz/code/image-captioning/BEST_checkpoint_vizwiz_5_cap_per_img_5_min_word_freq.pth.tar --word_map /home/gusviloca@GU.GU.SE/aics-vizwiz/out_data/WORDMAP_vizwiz_5_cap_per_img_5_min_word_freq.json
    """
