
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
import csv

import csv


lines_txt_path = r"C:\Users\mikey\OneDrive\Documents\USCB\Data Mining\Project\ascii\lines.txt"

forms_data = {}

with open(lines_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.strip().split(' ')
        if parts[1] != 'ok':
            continue

        form_id = '-'.join(parts[0].split('-')[:2])
        text = ' '.join(parts[8:])
        
        #get rid of | and put in spaces
        text = text.replace('|', ' ').strip()

        forms_data.setdefault(form_id, []).append(text)

#keep only forms starting with 'j'
forms_data = {fid: lines for fid, lines in forms_data.items() if fid.startswith('j')}

#prepare the data to write to CSV (one big line per form)
csv_data = []
for form_id, lines in forms_data.items():
    #join all lines into one single string (normal readable sentence)
    big_line = ' '.join(lines)
    csv_data.append([form_id, big_line])

#write the data to a csv file
csv_output_path = r"C:\Users\mikey\OneDrive\Documents\USCB\Data Mining\Project\forms_data.csv"
with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Form ID', 'Form Data'])  #write header
    w.writerows(csv_data)  #write all the rows
