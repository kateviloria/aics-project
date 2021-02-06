import glob
from IPython.display import Image
import random
import json
import pandas as pd

"""
Create CSV file for text data from JSON file provided by VizWiz.
- format can be found on VizWiz website (https://vizwiz.org/tasks-and-datasets/image-captioning/)
"""
print('Writing CSV file...')

# load train json file as dict
# CHANGE PATH if necessary
with open('/home/gusviloca@GU.GU.SE/aics-vizwiz/annotations/train.json') as f:
    anns_dict = json.load(f)

# make images dict into pandas df
images_df = pd.json_normalize(anns_dict, record_path=['images'])

# annotations dict to df
anns_df = pd.json_normalize(anns_dict, record_path=['annotations'])

# check how many are rejected (spam)
reject_captions = anns_df.loc[anns_df.is_rejected == True, 'is_rejected'].count()
#print('no. of rejected captions', reject_captions)

# give id of rejected captions
reject_ids = anns_df.loc[anns_df['is_rejected'] == True, 'image_id'].tolist()
#print('no. of reject ids', len(reject_ids))

# all images that must be dropped
reject_ids_unique = set(reject_ids)
#print('unique no. of reject ids aka images', len(reject_ids_unique))

# delete rows where caption was rejected
anns_filtered = anns_df[~anns_df['image_id'].isin(reject_ids_unique)]

# unique list of image_ids in anns
non_rejects = list(anns_filtered.image_id.unique())
#print('len non rejects', len(non_rejects))

# delete images that had captions that were rejected
images_filtered = images_df[~images_df['id'].isin(reject_ids_unique)]
#print(images_filtered.shape)

# check if both filtered dfs have the same number of image_ids
image_ids = list(images_filtered.id.unique())
assert len(non_rejects)==len(image_ids)

# make dictionary from images_filtered {image_id: filename}
filename_to_id = dict(zip(images_filtered.id, images_filtered.file_name))

# use filename_to_id to fill in column in anns_filtered to add filename
anns_filtered['file_name'] = anns_filtered['image_id'].map(filename_to_id) 

final_df = anns_filtered

# reorganise columns and rename id to caption id to decrease confusion between image_id
new_df = final_df.rename(columns={'id': 'caption_id'})
new_df = new_df[['file_name', 'image_id','caption_id','caption','is_rejected', 'is_precanned', 'text_detected']]

final_df.to_csv(r'/home/gusviloca@GU.GU.SE/aics-vizwiz/vizwiz_dataframe.csv', index = False, header=True)

print('CSV file created.')