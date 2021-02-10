# Data

Use this folder for datasets that you may have created and the generated tables of results.

#### vizwiz_dataframe.csv  
- __file_name__  
- __image_id__
- __caption_id__
- __caption__
- __is_rejected__: spam captions (should be all False since it was filtered when .csv file was being written)
- __is_precanned__: “Quality issues are too severe to recognize visual content.”
- __text_detected__: true for at least three of the five crowdsourced results and false otherwise

#### vizwiz_annotations.json
##### FORMAT

```
{ images: 
        [{ "sentids": [0, 1, 2, 3, 4], 
            "imgid": 0, 
            "sentences":
                        [{ "tokens": 
                                    ["a", "black", "dog", "is",    "running", "after", "a", "white", "dog", "in", "the", "snow"], 
                            "raw": "A black dog is running after a white dog in the snow .", 
                            "imgid": 0, 
                            "sentid": 0 
                        }]
            "split": "train", 
            "filename": "2513260012_03d33305cf.jpg"
  filename : 'vizwiz'}
```