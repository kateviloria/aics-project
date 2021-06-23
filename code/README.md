# Code  

Use this folder for the code related to your project.  

###### Commands  
```bash
python create_dataframe.py
python data_prepare.py
python train.py
python test.py
```   
##### create_dataframe.py  
- takes JSON files provided by VizWiz and uses pandas to create a dataframe  
- exports dataframe to a .csv file  
- takes .csv file to make a nested dictionary in the same format as Karpathy mentioned in image caption tutorial (format shown in data/README)
- nested dictionary saved as JSON file

##### data_prepare.py  
- creates and prepares input files by splits (training, validation, and test) using vizwiz_annotations.json
- creates word map and saves as JSON file