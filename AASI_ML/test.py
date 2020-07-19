from microfaune_package.microfaune.detection import RNNDetector
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import datetime
import matplotlib.dates as mdates
import pdb

audio_dir = sys.argv[1]
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
    
with open("local_scores_AM15.txt", "w") as f:
    for arr in local_scores:
        for el in arr:
            f.write(str(el) + ",")
        f.write("\n")

with open("global_scores_AM15.txt", "w") as f:
    for sc in global_scores:
        f.write(str(sc) + '\n')

# dates = ["06/17/2019", "06/18/2019", "06/19/2019", "06/20/2019", "06/21/2019"]
# x_vals = [datetime.datetime.strptime(d,"%m/%d/%Y").date() for d in dates]
# ax = plt.gca()

# formatter = mdates.DateFormatter("%Y-%m-%d")
# ax.xaxis.set_major_formatter(formatter)

# locator = mdates.DayLocator()
# ax.xaxis.set_major_locator(locator)

# y_vals = global_scores
# y_vals = local_scores
# x_vals = np.arange(0, len(y_vals))

# plt.figure(figsize=(15, 8))
# plt.plot(x_vals, y_vals)
# plt.title("Bird Vocalizations from 00:00 6/17 - 23:50 6/21")
# plt.xlabel("Time")
# plt.ylabel("Prediction scores")
# plt.ylim(0,1.0)

# plt.savefig("local_score_AM15_2.png")
