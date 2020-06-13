import sys
from pydub import AudioSegment

import librosa
import os

folder_str = sys.argv[1]
tar_str = sys.argv[2]
#folder_str = "/Users/fyy0194/Documents/cse237D/grabador2019_1"
#tar_str = "/Users/fyy0194/Documents/cse237D/grabador2019_1/output/"
for file in os.listdir(folder_str):
    audio_name = os.fsdecode(file)
    if audio_name.endswith(".WAV"):
        print("abs Path: "+ folder_str + '/' + audio_name)
        y,s = librosa.load(folder_str+ '/' + audio_name, sr = 44100)
        librosa.output.write_wav(tar_str + audio_name, y, s)
    
    
