import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.dates as mdates
import pdb

AM_NAME = "AM16"
FIRST_STAMP = "00:00:00 06/13/2019"

def line_graph(time_stamps, global_scores):
    plt.figure(figsize=(15, 8))
    plt.plot(time_stamps, global_scores)
    plt.plot_date(time_stamps, global_scores)
    plt.title("Bird Vocalizations from 00:00 6/13/19 - 23:50 6/17/19 - " + AM_NAME)
    plt.xlabel("Time")
    plt.ylabel("Prediction scores")
    plt.grid(which='major', linestyle='-')
    plt.ylim(0,1.0)
    plt.savefig("global_scores_line_"+AM_NAME+".png")

def box_plot(time_stamps, global_scores):
    day_scores = []
    night_scores = []
    # box and whisker for 6:00-17:50 and 18:00-5:50
    for t in range(len(time_stamps)):
        # if time is day: 6am to 5:50pm
        if 6 <= time_stamps[t].hour < 18:
            day_scores.append(global_scores[t])
        else:
            night_scores.append(global_scores[t])

    scores = [day_scores, night_scores]

    fig, ax = plt.subplots()
    ax.set_title("Day vs Night Bird Vocalizations - " + AM_NAME)
    ax.boxplot(scores)
    plt.savefig("global_scores_box_"+AM_NAME+".png")


if __name__ == "__main__":
    # collect data
    global_scores = []
    with open("global_scores_"+AM_NAME+".txt", "r") as f:
        for line in f:
            global_scores.append(float(line.strip()))

    # create range of datetime objects
    base = datetime.datetime.strptime(FIRST_STAMP, "%H:%M:%S %m/%d/%Y")
    time_stamps = [base + datetime.timedelta(minutes=x*10) for x in range(len(global_scores))]

    # line_graph(time_stamps, global_scores)
    box_plot(time_stamps, global_scores)