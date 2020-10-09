import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import glob
import scipy.signal as scipy_signal
import time
from playsound import playsound
#import _thread as thread
#import multiprocessing
import csv
import os

def spectrogram(Signal,rate_ratio):
    plt.figure();
    plt.specgram(scipy_signal.resample(SIGNAL,int(len(SIGNAL)*rate_ratio)),Fs=44100,NFFT=1024,noverlap=512,window=np.hanning(1024))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (hz)")
    plt.show()


def play_clip(clip_path):
    playsound(clip_path)


numFiles = len(next(os.walk("./AM4_samples_split/"))[2])


ndx = 1
for file in glob.glob("./AM4_samples_split/" + "*.WAV"):
    print(file)
    print(str(ndx) + "/" + str(numFiles) + " Files Labeled")
    ndx += 1
    pathList = file.split('/')
    SAMPLE_RATE,SIGNAL = wavfile.read(file)
#    rate_ratio = 44100/SAMPLE_RATE
    playsound(file)
    while True:
        userInput = input("Did you here a bird? y/n/repeat: ")
        if userInput == "y" or userInput == "n":
            #print(userInput)
            with open("Stratified_Random_Sample_Bird_Labels.csv",mode='a') as testCSV:
                csvWriter = csv.writer(testCSV,delimiter=",")
                csvWriter.writerow([pathList[len(pathList)-2],pathList[len(pathList)-1],userInput])
            break
        elif userInput == "repeat":
            playsound(file)
            continue
#    plt.figure();
#    plt.specgram(scipy_signal.resample(SIGNAL,int(len(SIGNAL)*rate_ratio)),Fs=44100,NFFT=1024,noverlap=512,window=np.hanning(1024))
#    plt.xlabel("Time (s)")
#    plt.ylabel("Frequency (hz)")
#    plt.show()
    #time.sleep(3)
#    plt.close()

#for file in glob.glob("./test_split/" + "*.WAV"):
#    SAMPLE_RATE,SIGNAL = wavfile.read(file)
#    rate_ratio = 44100/SAMPLE_RATE
#    thread.start_new_thread(play_clip,(file,))
#    thread.start_new_thread(spectrogram,(SIGNAL, rate_ratio,))
#    audio = multiprocessing.Process(target = play_clip, args=(file,))
#    display = multiprocessing.Process(target = spectrogram, args=(SIGNAL,rate_ratio,))
#    audio.start()
#    display.start()
#    audio.join()

#for file in glob.glob("./test_split/"+".WAV"):
#    SAMPLE_RATE,SIGNAL = wavfile.read(file)
#    print(file)
#    play_clip(file)
#    while True:
#        userInput = input("Did you here a bird? y/n/r: ")
#        if userInput == "y" or userInput == "n":
#            print(userInput)
#            break
#        elif userInput == "r":
#            play_clip(file)

