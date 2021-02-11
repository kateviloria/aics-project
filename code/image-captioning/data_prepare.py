from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='vizwiz',
                       karpathy_json_path='/home/gusviloca@GU.GU.SE/aics-vizwiz/data/vizwiz_annotations.json',
                       image_folder='/home/gusviloca@GU.GU.SE/aics-vizwiz/data/vizwiz_images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/home/gusviloca@GU.GU.SE/aics-vizwiz/out_data',
                       max_len=50)
