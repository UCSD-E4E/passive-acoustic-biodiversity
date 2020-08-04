from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
import pdb
import os
import argparse

AM_NAME = "AM16"
FIRST_STAMP = "00:00:00 06/13/2019"
NUM_DAYS_AVERAGED = 0
SCORES_DIR = "test_dir/outputs/"


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
    duration = local_scores.pop(0)
    num_scores = local_scores.pop(0)
    step = duration / num_scores
    time_stamps = np.arange(0, duration, step)

    # graph
    plt.figure(figsize=(18, 8))
    plt.plot(time_stamps, local_scores)
    plt.title("Local Scores for "+clip_name)
    plt.xlabel("Time")
    plt.ylabel("Prediction scores")
    plt.grid(which='major', linestyle='-')
    plt.ylim(0,1.0)
    plt.savefig(scores_dir+clip_name[:-4]+".png")


if __name__ == "__main__":
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
        # collect data
        for scores_file in os.listdir(SCORES_DIR):
            local_scores = []
            if not scores_file.endswith(".txt"): continue
            
            with open(SCORES_DIR+scores_file, "r") as f:
                for line in f:
                    local_scores.append(float(line.strip()))

            # create graph
            local_line_graph(local_scores, scores_file, SCORES_DIR)
