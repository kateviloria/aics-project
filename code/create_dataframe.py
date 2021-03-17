# v2
import glob
from IPython.display import Image
import random
import json
import pandas as pd
from nltk.tokenize import WhitespaceTokenizer
import re

"""
Create CSV file for text data from JSON file provided by VizWiz.
- format can be found on VizWiz website (https://vizwiz.org/tasks-and-datasets/image-captioning/)
Use CSV file to create JSON file in the same format as Karpathy's for flickr8k data
"""
print('Writing CSV file...')

whitespacetokenizer = WhitespaceTokenizer()

# load train json file as dict
# CHANGE PATH if necessary

# **REMINDER**
# different path structure for mltgpu and github repo

with open('/home/gusviloca@GU.GU.SE/aics-vizwiz/data/annotations/train.json') as f:
    anns_dict = json.load(f)

# make images dict into pandas df
images_df = pd.json_normalize(anns_dict, record_path=['images'])

# annotations dict to df
anns_df = pd.json_normalize(anns_dict, record_path=['annotations'])

# check how many are rejected (spam) and precanned
reject_captions = anns_df.loc[anns_df.is_rejected == True, 'is_rejected'].count()
print('no. of rejected captions', reject_captions)

precanned_captions = anns_df.loc[anns_df.is_precanned == True, 'is_precanned'].count()
print('no. of precanned captions', precanned_captions)

# give id of rejected captions and precanned
reject_ids = anns_df.loc[anns_df['is_rejected'] == True, 'image_id'].tolist()
print('no. of reject ids', len(reject_ids))

precanned_ids = anns_df.loc[anns_df['is_precanned'] == True, 'image_id'].tolist()
print('no. of precanned ids', len(precanned_ids))

# all images that must be dropped
reject_ids_unique = list(set(reject_ids))
print('unique no. of reject ids aka images', len(reject_ids_unique))

precanned_ids_unique = list(set(precanned_ids))
print('unique no. of precanned ids aka images', len(precanned_ids_unique))

all_rejects = reject_ids_unique + precanned_ids_unique

# delete rows where caption was rejected or precanned
anns_filtered_rejected = anns_df[~anns_df['image_id'].isin(reject_ids_unique)]

anns_filtered_precanned = anns_filtered_rejected[~anns_filtered_rejected['image_id'].isin(precanned_ids_unique)]

# unique list of image_ids in anns
non_rejects = list(anns_filtered_precanned.image_id.unique())
print('len non rejects', len(non_rejects))

# delete images that had captions that were rejected
images_filtered = images_df[~images_df['id'].isin(all_rejects)]
print(images_filtered.shape)

#HERE
# check if both filtered dfs have the same number of image_ids
image_ids = list(images_filtered.id.unique())
assert len(non_rejects)==len(image_ids)

# make dictionary from images_filtered {image_id: filename}
filename_to_id = dict(zip(images_filtered.id, images_filtered.file_name))

# use filename_to_id to fill in column in anns_filtered to add filename
anns_filtered_precanned['file_name'] = anns_filtered_precanned['image_id'].map(filename_to_id) 

new_df = anns_filtered_precanned

# reorganise columns and rename id to caption id to decrease confusion between image_id
new_df = new_df.rename(columns={'id': 'caption_id'})
new_df = new_df[['file_name', 'image_id','caption_id','caption','is_rejected', 'is_precanned', 'text_detected']]

# add column where caption is tokenized
def tokenize_caption(string):
    lower = string.lower()
    no_punct = re.sub('[^A-Za-z0-9]+', ' ', lower)
    tokens = whitespacetokenizer.tokenize(no_punct)
    return tokens

new_df['tokens'] = new_df['caption'].apply(tokenize_caption)

# assign train, validate, test
train_size = 60/100
val_size = 20/100
test_size = 20/100

unique_images = list(new_df.image_id.unique())
n_images = len(unique_images)

# make list that (based on given weights) assigns train/validate/test for same length as images
assign_split = random.choices(population=['train','val','test'], weights = [train_size, val_size, test_size], k = n_images)
image_id_to_split = dict(zip(unique_images, assign_split))

# fill in split column through mapping of image_id_to_split
new_df['split'] = new_df['image_id'].map(image_id_to_split)

new_df.to_csv(r'/home/gusviloca@GU.GU.SE/aics-vizwiz/data/vizwiz_dataframe.csv', index = False, header=True)

print('CSV file created.')


# make nested dictionary and export to json file

def make_nested(dataframe):
    main_dict ={}
    main_dict['dataset'] = 'vizwiz'
    #print(main_dict)
    images_val = []
    
    for every_image in range(len(unique_images)):
        image_dict = {}
        
        img_id = int(unique_images[every_image])
        image_dict['imgid'] = img_id
        #print('img_id', img_id, 'type', type(img_id))
        
        # create smaller df of one image
        image_df = dataframe.loc[dataframe['image_id'] == img_id]
        #print(image_df)
        
        # add split
        split = image_df.split.unique()
        split = split[0]
        #print('split type', split, type(split))
        image_dict['split'] = split

        # add sent ids
        sent_ids = image_df['caption_id'].tolist()
        #print('sent_ids', sent_ids, 'type', type(sent_ids))
        image_dict['sentids'] = sent_ids
        
        # add filename
        filename = image_df.file_name.unique()
        filename = filename[0]
        #print('filename', filename, 'type', type(filename))
        image_dict['filename'] = filename
        
        # make separate dicts for each caption as a value for sentences
        sentence_dicts =[]  
        for every_caption in range(len(sent_ids)):
            caption_dict ={}
            caption_id = sent_ids[every_caption]
            #print('caption id', caption_id, 'type', type(caption_id))
            caption_dict['sentid'] = caption_id
            sent_df = image_df.loc[image_df['caption_id'] == caption_id]
            
            # add tokens
            tokens = sent_df.iloc[0]['tokens']
            #print('tokens', tokens, 'type', type(tokens))
            caption_dict['tokens'] = tokens
            
            # add raw/caption
            caption = sent_df.iloc[0]['caption']
            #print('caption', caption, 'type', type(caption))
            caption_dict['raw'] = caption
            
            # add image id and sentence id
            caption_dict['imgid'] = img_id
            
            #print('caption dict')
            #print(caption_dict)
            
            sentence_dicts.append(caption_dict)
            
        # sentence dict as value in image_dict
        image_dict['sentences'] = sentence_dicts
        
        #print('HELELEOELEOELELELLEASDLKFJASLDFJS')
        #print('image dict', image_dict)
        images_val.append(image_dict)
        #print(images_val)
        
    # images as value for main dict
    main_dict['images'] = images_val

    print('Writing JSON file...')
    
    with open('/home/gusviloca@GU.GU.SE/aics-vizwiz/data/vizwiz_annotations.json', 'w') as fp:
        json.dump(main_dict, fp)
    
    print('JSON file written!')

print('Reformatting as nested dictionary...')
make_nested(new_df)