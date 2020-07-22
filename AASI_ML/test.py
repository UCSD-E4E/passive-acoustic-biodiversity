from microfaune_package.microfaune.detection import RNNDetector
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pdb

AM_NAME = "AM16"
audio_dir = AM_NAME + "_Clips/"
detector = RNNDetector()
local_scores = []
global_scores = []

for wav_file in os.listdir(audio_dir):
    try:
        global_score, local_score = detector.predict_on_wav(audio_dir + wav_file)
        print("Loaded", wav_file)
    except:
        print("Error on file", wav_file)
        global_score = [0]
        local_score = [0]*22501
        continue

    local_scores.append(local_score)
    global_scores.extend(global_score)
    
with open("local_scores_"+AM_NAME+".txt", "w") as f:
    for arr in local_scores:
        for el in arr:
            f.write(str(el) + ",")
        f.write("\n")

with open("global_scores_"+AM_NAME+".txt", "w") as f:
    for sc in global_scores:
        f.write(str(sc) + '\n')
