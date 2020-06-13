from microfaune.detection import RNNDetector
import matplotlib.pyplot as plt
import sys
import numpy as np
from pydub import AudioSegment
import librosa
import os

if len(sys.argv)!= 3:
    print("usage: -D [path] for sample rate over 44100kHz file; -L [path] for 44100kHz files")
    exit()
flag = sys.argv[1]
if not (flag == "-D" or flag == "-L"):
    print("invalid flag. Usage: -D [path] for sample rate over 44100kHz file; -L [path] for 44100kHz files")
    exit()
elif flag == "-D":
    detector = RNNDetector()
    audio_str = sys.argv[2]
    y,s = librosa.load(audio_str, sr = 44100)
    ds_str=audio_str[:-4]+"_ds.wav"
    librosa.output.write_wav(ds_str, y, s)
    loop = AudioSegment.from_wav(ds_str)
    length = len(loop)
    clips = []
    for i in range(0,(int)(length/10000)):
        newAudio = loop[i*10000 : i*10000+10000]
        clips.append(ds_str[:-4]+"_"+str(i)+".wav")
        newAudio.export(ds_str[:-4]+"_"+str(i)+".wav",format = "wav")
    
    for i in range(0,len(clips)):
        global_score, local_score = detector.predict_on_wav(clips[i])
        print("processing "+str(i) +" of " + str(len(clips)))
        if global_score>=0.5:
            print("Bird vocalisation detected in "+ clips[i])
        
    
else:
    detector = RNNDetector()
    audio_str = sys.argv[2]
    loop = AudioSegment.from_wav(audio_str)
    length = len(loop)
    clips = []
    for i in range(0,(int)(length/10000)):
        newAudio = loop[i*10000 : i*10000+10000]
        clips.append(audio_str[:-4]+"_"+str(i)+".wav")
        newAudio.export(audio_str[:-4]+"_"+str(i)+".wav",format = "wav")
    
    for i in range(0,len(clips)):
        global_score, local_score = detector.predict_on_wav(clips[i])
        print("process "+str(i) +" of " + str(len(clips)))
        if global_score>=0.5:
            print("Bird vocalisation detected in "+ clips[i])
    



