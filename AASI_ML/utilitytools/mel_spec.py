import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import csv

folder_str = sys.argv[1]
tar_str = sys.argv[2]
#folder_str ="/Users/fyy0194/Documents/cse237D/grabador2019_1/output/trimed"
#tar_str = "/Users/fyy0194/Documents/cse237D/grabador2019_1/output/specs"
with open(tar_str+'/labels.csv','w', newline='') as csvfile:
    fieldnames = ['file', 'label']
    writer = csv.DictWriter(csvfile,fieldnames = fieldnames)
    writer.writeheader()
    
    for file in os.listdir(folder_str):
        audio_name = os.fsdecode(file)
        if audio_name.endswith(".wav"):
            writer.writerow({'file':audio_name[:-4],'label':str(0)})
            y, sr = librosa.load(folder_str+'/'+audio_name)
            spec = np.abs(librosa.stft(y,hop_length=512))
            spec = librosa.amplitude_to_db(spec, ref = np.max)
            librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('spectrogram')
            #plt.show()
            plt.savefig(tar_str + '/'+ audio_name[:-4] + "_spec")
            plt.close()

