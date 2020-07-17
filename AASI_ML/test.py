from microfaune_package.microfaune.detection import RNNDetector
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import datetime

audio_dir = sys.argv[1]
detector = RNNDetector()

for wav_file in os.listdir(audio_dir):
    try:
        global_score, local_score = detector.predict_on_wav(wav_file)
    except:
        print("Error on file", wav_file)

    print(wav_file)
    print("Global score:", global_score)
    print("Local score:", local_score)

dates = ["06/17/19", "06/18/19", "06/19/19", "06/20/19", "06/21/19"]
x_vals = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in dates]
ax = plt.gca()

formatter = mdates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(formatter)

locator = mdates.DayLocator()
ax.xaxis.set_major_locator(locator)

plt.figure(figsize=(15, 8))
plt.plot(x_vals, local_score, lw=1)
plt.title("Bird Vocalizations from 00:00 6/17 - 23:50 6/21")
plt.xlabel("Time")
plt.ylabel("Prediction scores")
plt.ylim(0,1.0)

plt.savefig("local_score_AM1.png")
