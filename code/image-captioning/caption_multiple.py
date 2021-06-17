# caption a bunch of images

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
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = '/home/gusviloca@GU.GU.SE/aics-vizwiz/code/image-captioning/BEST_checkpoint_vizwiz_5_cap_per_img_5_min_word_freq.pth.tar' 
images_dir = '/home/gusviloca@GU.GU.SE/aics-vizwiz/data/vizwiz_images'
word_map = '/home/gusviloca@GU.GU.SE/aics-vizwiz/out_data/WORDMAP_vizwiz_5_cap_per_img_5_min_word_freq.json'

def caption_greedy(encoder, decoder, image_path, word_map, max_length):
    
    vocab_size = len(word_map)

    image_file = image_path.split('/')[-1]
    image_file = image_file[:-4]

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
            
    return image_file, seq, seq_alphas


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

    # Load model
    checkpoint = torch.load(model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    max_len = 50
    image_paths = glob.glob('/home/gusviloca@GU.GU.SE/aics-vizwiz/data/vizwiz_images/*.jpg')
    print('len image paths', len(image_paths))
    image_paths = image_paths[500:600]
    print('new len image paths', len(image_paths))

    for every_pic_idx in range(len(image_paths)):
        image_file, seq, seq_alphas = caption_greedy(encoder, decoder, image_paths[every_pic_idx], word_map, max_len)
        print('image', image_file)
        print('Caption:', seq)

    print('Finished captioning.')

    # NEED TO EDIT FOR GREEDY
    # Encode, decode with attention and greedy
    #seq, seq_alphas = caption_greedy(encoder, decoder, img, word_map, max_len)
    #print('Generated Caption:', seq)
    #alphas = torch.FloatTensor(alphas)
    
    #decoded_seq = []
    #for item in seq:
    #    decoded_seq.append(rev_word_map[item])
    #print(decoded_seq)

    # Visualize caption and attention of best sequence
    #visualize_att(args.img, seq, seq_alphas, rev_word_map, args.smooth)


