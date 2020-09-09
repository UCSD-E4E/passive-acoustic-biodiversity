from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
import pdb
import os
import argparse
from scipy import signal
from scipy.io import wavfile
import scipy.signal as scipy_signal
from pathlib import Path

AM_NAME = "AM16"
FIRST_STAMP = "00:00:00 06/13/2019"
NUM_DAYS_AVERAGED = 0


"""
Parse input arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Detect birds")
    parser.add_argument('--global', dest='do_global_graphs',
                        help='whether creating global graphs or not',
                        action='store_true')
    args = parser.parse_args()
    
    return args


def global_line_graph(global_scores):
    # create range of datetime objects
    base = datetime.datetime.strptime(FIRST_STAMP, "%H:%M:%S %m/%d/%Y")
    time_stamps = [base + datetime.timedelta(minutes=x*10) for x in range(len(global_scores))]

    # graph
    plt.figure(figsize=(18, 8))
    plt.plot(time_stamps, global_scores)
    plt.plot_date(time_stamps, global_scores)
    plt.title("Bird Vocalizations from 00:00 6/13/19 - 23:50 6/17/19 - " + AM_NAME)
    plt.xlabel("Time")
    plt.ylabel("Prediction scores")
    plt.grid(which='major', linestyle='-')
    plt.ylim(0,1.0)
    plt.savefig("global_scores_line_"+AM_NAME+".png")


def diurnal_line_graph(global_scores):
    # average all days
    mean_global_scores = [np.mean(global_scores[i::144]) for i in range(144)]

    # create range of datetime objects
    base = datetime.datetime.strptime("00:00:00", "%H:%M:%S")
    time_stamps = [base + datetime.timedelta(minutes=x*10) for x in range(144)]

    # graph
    plt.figure(figsize=(18, 8))
    plt.plot(time_stamps, mean_global_scores)
    plt.plot_date(time_stamps, mean_global_scores)
    plt.title("Average Bird Vocalizations from 6/13/19 to 6/30/19 - " + AM_NAME)
    plt.xlabel("Time")
    plt.ylabel("Prediction scores")
    plt.grid(which='major', linestyle='-')
    plt.ylim(0,1.0)
    plt.savefig("diurnal_line_"+AM_NAME+".png")


def box_plot(global_scores):
    day_scores = []
    night_scores = []
    # box and whisker for 6:00-17:50 and 18:00-5:50
    for t in range(len(time_stamps)):
        # if time is day: 6am to 5:50pm
        if 6 <= time_stamps[t].hour < 18:
            day_scores.append(global_scores[t])
        else:
            night_scores.append(global_scores[t])

    print('Median day score:   {:.4f}'.format(np.median(day_scores)))
    print('Median night score: {:.4f}'.format(np.median(night_scores)))
    print('Mean day score:     {:.4f}'.format(np.average(day_scores)))
    print('Mean night score:   {:.4f}'.format(np.average(night_scores)))

    scores = [day_scores, night_scores]

    fig, ax = plt.subplots()
    ax.set_title("Day vs Night Bird Vocalizations - " + AM_NAME)
    ax.boxplot(scores)
    plt.savefig("global_scores_box_"+AM_NAME+".png")


def local_line_graph(local_scores, clip_name, scores_dir):
    # load signal to get spectrogram
    home = str(Path.home())
    wav_path = os.path.join(
            home, "../../media/e4e/New Volume/AudiomothData/AM15_16_Birds/")
    # remove file extension and _LS label from end of file
    wav_name = wav_path + clip_name[:-7] + ".WAV"
    # load file
    sample_rate, samples = wavfile.read(wav_name)

    # calculate time stamps - x axis
    # takes first two lines
    duration = local_scores.pop(0)
    num_scores = local_scores.pop(0)

    step = duration / num_scores
    time_stamps = np.arange(0, duration, step)

    if len(time_stamps) > len(local_scores):
        time_stamps = time_stamps[:-1]

    # general graph features
    fig, axs = plt.subplots(2)
    fig.set_figwidth(22)
    fig.set_figheight(10)
    fig.suptitle("Spectrogram and Local Scores for "+clip_name)
    # score line plot - top plot
    axs[0].plot(time_stamps, local_scores)
    axs[0].set_xlim(0,duration)
    axs[0].set_ylim(0,1)
    axs[0].grid(which='major', linestyle='-')
    # spectrogram - bottom plot
    Pxx, freqs, bins, im = axs[1].specgram(samples, Fs=sample_rate,
            NFFT=4096, noverlap=2048,
            window=np.hanning(4096), cmap="ocean")
    axs[1].set_xlim(0,duration)
    axs[1].set_ylim(0,22050)
    axs[1].grid(which='major', linestyle='-')

    # save graph
    plt.savefig(scores_dir+clip_name[:-4]+".png")


if __name__ == "__main__":
    scores_dir = "score_files/AM15_16/"
    args = parse_args()
    
    # Do global graphs
    if args.do_global_graphs:
        # collect data
        global_scores = []
        with open("global_scores_"+AM_NAME+".txt", "r") as f:
            for line in f:
                global_scores.append(float(line.strip()))

        # create graphs
        global_line_graph(global_scores)
        diurnal_line_graph(global_scores)
        box_plot(global_scores)
    # Do local graphs
    else:
        # make graph for each score file in scores_dir
        for scores_file in os.listdir(scores_dir):
            # skip if not text file
            if not scores_file.endswith(".txt"): continue

            # convert txt file into array of floats
            local_scores = []
            with open(scores_dir+scores_file, "r") as f:
                for line in f:
                    local_scores.append(float(line.strip()))

            # create graph
            local_line_graph(local_scores, scores_file, scores_dir)
