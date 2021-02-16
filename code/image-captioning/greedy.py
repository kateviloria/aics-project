# GREEDY CAPTION - takes top probability for each to make caption

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
import imageio
from skimage.transform import resize

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('device', device)

max_len = 50

def get_greedy_caption(encoder, decoder, image_path, word_map):
    """
    Takes most likely word to create greedy caption.
    """

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

    # Start decoding 
    h, c = decoder.init_hidden_state(encoder_out)
    # hidden state of image?
    # hidden state is important
    # initialise 
    # this should be in the loop no?

    #print('h')
    #print(h)
    #print('c')
    #print(c)

    # Tensor to store predicted word at each step; now they're just <start>
    # start = torch.LongTensor([[word_map['<start>' * k ]]]).to(device)  # (k, 1)

    start_token = '<start>'
    idx_start_token = word_map[start_token]
    inputs = idx_start_token # rename to get previous index
    seq = ['<start>']
    
    for previous_word_idx in range(max_len):
        
        print(decoder.embedding)
        print(idx_start_token)
        # unless pred word is <end>
        embeddings = decoder.embedding(torch.LongTensor(inputs).cuda()).squeeze(1) # (s, embed_dim)
        # FROM CAPTION.PY
        print(idx_start_token)
        print(embeddings.shape)
        
        h, c = decoder.decode_step(embeddings, (h, c))  # (s, decoder_dim)

        #sequence = [word_map[s] for s in in_text.split(" ") if s in word_map]
        #print('sequence', sequence)
        scores = decoder.fc(h)  # (s, vocab_size) # fully connected layer maps to vocab size (logits)
        print('scores')
        print(scores)
        scores = F.log_softmax(scores, dim=1) # turn logits into probabilities
        print('log to probabilities')
        print(scores)
        # highest is the index of the biggest score
        # make sure highest is an INDEX!!!
        highest = torch.argmax(scores) # return highest probability
        print('highest score')
        print(highest)

        word = word_map[highest] # get word from word map
        print('word mapped to')
        print(word)
        seq.append(word)
        
        inputs = highest
        
        print('in_text update', in_text)
        if word == '<end>':
            break
            
    final = in_text.split()
    print('final split')
    print(final)
    final = final[1:-1]
    print('final')
    print(final)

    return final



    # torch max
    # torch argmax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    #parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    #parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    # Load model
    #checkpoint = torch.load(args.model, map_location=str(device))
    checkpoint = torch.load(args.model, map_location='cuda:2')
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

    # Encode, decode with attention and beam search
    wee = get_greedy_caption(encoder, decoder, args.img, word_map)
    #seq, alphas = get_greedy_caption(encoder, decoder, args.img, word_map, args.beam_size)
    #alphas = torch.FloatTensor(alphas)
    
    """
    decoded_seq = []
    for item in seq:
        decoded_seq.append(rev_word_map[item])
    print(decoded_seq)
    """
    # Visualize caption and attention of best sequence
    #visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)


"""
COMMAND
python greedy.py --img /home/gusviloca@GU.GU.SE/aics-vizwiz/data/vizwiz_images/VizWiz_train_00008121.jpg --model /home/gusviloca@GU.GU.SE/aics-vizwiz/code/image-captioning/BEST_checkpoint_vizwiz_5_cap_per_img_5_min_word_freq.pth.tar --word_map /home/gusviloca@GU.GU.SE/aics-vizwiz/out_data/WORDMAP_vizwiz_5_cap_per_img_5_min_word_freq.json
"""