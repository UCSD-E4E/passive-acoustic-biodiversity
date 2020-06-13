from microfaune.detection import RNNDetector
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import csv

src_str = "/Users/fyy0194/Documents/cse237D/microfaune-master/microfaune_package/unknown_grabador16"

label_dict = dict()

threshold= 0.5
with open('/Users/fyy0194/Documents/cse237D/grabador2019_1/output/specs/labels.csv','r',newline = '') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        label_dict[row['file']] = row['label']
print('dict complete\n')

with open(src_str+'/result.csv','w',newline = '') as rstcsv:
    fieldnames = ['file','result']
    writer = csv.DictWriter(rstcsv,fieldnames=fieldnames)
    writer.writeheader()
    detector = RNNDetector()
    for file in os.listdir(src_str):
        #total = total +1
        audio_name = os.fsdecode(file)
        if audio_name.endswith(".wav"):
            global_score, local_score = detector.predict_on_wav(src_str+'/'+audio_name)
            if global_score>=threshold:
                writer.writerow({'file':audio_name[:-4],'result':str(1)})
            else:
                writer.writerow({'file':audio_name[:-4],'result':str(0)})
        
            
