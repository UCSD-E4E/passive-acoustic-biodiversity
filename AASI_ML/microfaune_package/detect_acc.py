import sys
import numpy as np
import librosa
import os
import csv
from microfaune.detection import RNNDetector

folder_dir = sys.argv[1]
detector = RNNDetector()
pred_dict = dict()
with open(folder_dir+'/prediction.csv','w', newline='') as csvfile:
    fieldnames = ['file','pred']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()
    for file in os.listdir(folder_dir):
        audio_name = os.fsdecode(file)
        if audio_name.endswith(".wav"):
            global_score,local_score = detector.predict_on_wav(folder_dir + '/'+audio_name)
            if (global_score>=0.5):
                writer.writerow({'file':audio_name[:-4],'pred':str(1)})
                pred_dict[audio_name[:-4]] = str(1)
            else:
                writer.writerow({'file':audio_name[:-4],'pred':str(0)})
                pred_dict[audio_name[:-4]] = str(0)

#csv_preds = open(folder_dir+'/prediction.csv',newline='')
#reader1 = csv.DictReader(csv_preds)
#for row in reader1:
#    pred_dict[row['file']] = row['pred']
#csv_preds.close()

csv_labels = open(folder_dir+'/labels.csv', newline = '')
label_dict = dict()
reader = csv.DictReader(csv_labels)
total = 0
correct = 0
for row in reader:
    f_name = row['file']
    if (pred_dict[f_name] == row['label']):
        correct = correct +1
    total = total +1
    
print("total: "+ str(total) + " correct: " + str(correct) + " accuracy: "+ str(correct/total))
csv_labels.close()
