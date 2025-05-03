
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:27:15 2025

@author: mikey
"""
"""
/////////////DOCUMENTATION/////////////////
PILLOW: https://pillow.readthedocs.io/en/stable/reference/Image.html
"""
#keras
import numpy as np
import pandas as pd
#image processing
import numpy as np

#png of the I am dataset is in folder we need to get to the path
def get_picture_path(line):
    #png of the I am dataset is in folder we need to get to the path
    substrings=line.split()
    target = substrings[-1]
    png_name = substrings[0]
    substrings=png_name.split('-')
    parent = substrings[0]
    child = substrings[0]+'-'+substrings[1]
    package = [parent,child,png_name, target]
    print(package)
    return package


#//////////////////Make Dataframe with input image datapath and target//////////
image_path_target_df = pd.DataFrame({"image_path": [], "target": []})


#get file path and target
with open('words.txt', 'r') as file:
    # Loop through each line in the file
    for line in file:
        # Process each line (you can print or manipulate it)
        important_info = line.strip()  # .strip() removes the trailing newline
        if(important_info[0]=='#'):
            None
        else:
            #[parent,child,png_name, target]
            parts = get_picture_path(important_info)
            
            #values to add
            image_path=parts[0]+'/'+parts[1]+'/'+parts[2]+".png"
            target_value = parts[3]
            new_row = [image_path, target_value]
            
            image_path_target_df.loc[len(image_path_target_df)] = new_row
    file.close()

#store in csv
image_path_target_df.to_csv('word_dataset_info.csv',columns=["image_path", "target"], sep=',')