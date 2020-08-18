from __future__ import division

from microfaune_package.microfaune.detection import RNNDetector
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import numpy as np
import pdb
import csv
import argparse
from scipy.io import wavfile
import scipy.signal as scipy_signal


"""
Parse input arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Detect birds")
    parser.add_argument('--global', dest='do_global_scores',
                        help='whether creating global scores or not',
                        action='store_true')
    parser.add_argument('--test', dest='test_net',
                        help='whether evaluate performance of network or not',
                        action='store_true')
    args = parser.parse_args()
    
    return args


"""
This function creates a global score file
and evaluates performance of network using labelled input data
"""
def calc_global_scores_test(audio_dir, label):
    detector = RNNDetector()
    global_scores = []
    raw_error = {'correct':0, 'total':0}

    for audio_file in os.listdir(audio_dir):
        try:
            # for wavs
            if audio_file.lower().endswith('.wav'):
                global_score, _ = detector.predict_on_wav(audio_dir + audio_file)
                print("Loaded", audio_file)
            # for mp3s
            elif audio_file.lower().endswith('.mp3'):
                global_score, _ = detector.predict(audio_dir + audio_file)
                print("Loaded", audio_file)
            else:
                print("Invalid file extension, skipping", audio_file)
                continue
        except:
            print("Error in file, skipping", audio_file)
            global_score = [0]
            continue
        
        global_scores.append((audio_file, global_score))
        # clips have birds
        if label == True:
            if global_score > 0.5:
                raw_error['correct']+=1
        else:
            if global_score <= 0.5:
                raw_error['correct']+=1
        raw_error['total']+=1

    with open("global_scores_"+str(label)+".txt", "w") as f:
        for sc in global_scores:
            f.write(str(sc) + '\n')
    
    return raw_error


"""
This function creates a global score file
"""
def calc_global_scores(audio_dir):
    detector = RNNDetector()
    global_scores = []

    for audio_file in os.listdir(audio_dir):
        try:
            # for wavs
            if audio_file.lower().endswith('.wav'):
                global_score, _ = detector.predict_on_wav(audio_dir + audio_file)
                print("Loaded", audio_file)
            # for mp3s
            elif audio_file.lower().endswith('.mp3'):
                global_score, _ = detector.predict(audio_dir + audio_file)
                print("Loaded", audio_file)
            else:
                print("Invalid file extension, skipping", audio_file)
                continue
        except:
            print("Error in file, skipping", audio_file)
            print(sys.exc_info()[0])
            global_score = [0]
            continue
        
        global_scores.append((global_score, audio_file))

    with open("score_files/XCSelection_Subset_GS.txt", "w") as f:
        for sc in global_scores:
            f.write(str(sc) + '\n')


"""
This function creates a local score file

--- Description of file it creates --- 
line 1: Duration of audio clip in seconds
line 2: Amount of local scores were created
line 3: one local score each line
...
line 3+len(local_scores)
"""
def calc_local_scores(audio_dir):
    # TODO optimize detector.predict to not have to take in real file, just numpy arr
    # init detector
    detector = RNNDetector()

    # generate local scores for every file in chosen directory
    for audio_file in os.listdir(audio_dir):
        # skip directories
        if os.path.isdir(audio_dir+audio_file): continue
        
        # read file
        raw_sample_rate, raw_samples = wavfile.read(audio_dir + audio_file)
        
        # downsample the sample if > 44.1 kHz
        if raw_sample_rate > 44100:
            rate_ratio = 44100 / raw_sample_rate
            samples = scipy_signal.resample(
                    raw_samples, int(len(raw_samples)*rate_ratio))
            sample_rate = 44100
            # resample produces unreadable float32 array so convert back
            samples = np.asarray(samples, dtype=np.int16)
            
            # add DS to end of downsampled file
            new_filename = audio_file[:-4] + "_DS" + audio_file[-4:]
            audio_file = new_filename
            
            # write downsampled file
            wavfile.write(audio_dir + new_filename, sample_rate, samples)
        else:
            sample_rate = raw_sample_rate
            samples = raw_samples

        # detection
        try:
            # for wavs
            if audio_file.lower().endswith('.wav'):
                _, local_score = detector.predict_on_wav(audio_dir + audio_file)
                print("Loaded", audio_file)
            # for mp3s
            elif audio_file.lower().endswith('.mp3'):
                _, local_score = detector.predict(audio_dir + audio_file)
                print("Loaded", audio_file)
            else:
                print("Invalid file extension, skipping", audio_file)
                continue
        except:
            print("Error in file, skipping", audio_file)
            continue
        
        # get duration of clip
        duration = len(samples) / sample_rate
        
        # write local score file in chosen directory
        with open("score_files/XCSubset/" + audio_file[:-4]+"_LS.txt", "w") as f:
            f.write(str(duration) + "\n")
            f.write(str(len(local_score))+"\n")
            for sc in local_score:
                f.write(str(sc) + '\n')
        
        # isolate bird sounds in the clip by eliminating dead noise
        isolate(local_score, samples, sample_rate, duration, audio_dir, audio_file)


def isolate(scores, samples, sample_rate, duration, audio_dir, filename):
    # how many samples does one score represent
    scale = len(samples) // len(scores)
    isolated_samples = np.empty(0, dtype=np.int16)
    
    # METHOD 1
    # isolate samples that produce a score above thresh
    # thresh = 0.1
    # indices = []
    # for i in range(len(scores)):
    #     if scores[i] >= thresh:
    #         indices.append(i)
    #         isolated_samples = np.append( isolated_samples, samples[i*scale:(i+1)*scale] )
    
    # METHOD 2
    # indices, props = scipy_signal.find_peaks(scores)
    # for i in indices:
    #     isolated_samples = np.append(isolated_samples, samples[i*scale:(i+1)*scale])

    # METHOD 3
    indices, props = scipy_signal.find_peaks(samples)
    for i in indices:
        isolated_samples = np.append(isolated_samples, samples[i])

    # METHOD 4
    # indices, props = scipy_signal.find_peaks(samples)
    # for i in indices:
    #     lo, hi = max(0, i-scale), min(len(samples), i+scale)
    #     isolated_samples = np.append(isolated_samples, samples[lo:hi])

    # calculate new duration
    new_duration = len(isolated_samples) / sample_rate
    percent_reduced = 1 - (new_duration / duration)
    print('Reduced {} from {:.2f}s to {:.2f}s. {:.2%} reduced.'.format( \
            filename, duration, new_duration, percent_reduced))

    # write file
    new_filename = filename[:-4] + "_RED" + filename[-4:]
    wavfile.write(audio_dir + new_filename, sample_rate, isolated_samples)


if __name__ == '__main__':
    args = parse_args()

    home = str(Path.home())
    peru_dir = os.path.join(
            home, "../../media/e4e/New Volume/AudiomothData/AM3_Subset/")
    present_dir = os.path.join(
            home, "../../media/e4e/New Volume/XCSelection/")
    absent_dir = os.path.join(
            home, "../../media/e4e/New Volume/audioset_nonbird/")
    global_dir = os.path.join(
            home, "../../media/e4e/New Volume/XCSelection/Subset/")
    local_dir = os.path.join(
            home, "../../media/e4e/New Volume/XCSelection/Subset/")

    # Calculate global scores
    if args.do_global_scores:

        # evalutate performace, output metrics
        if args.test_net:
            # Second parameter is True if birds present, False if absent
            present_error = calc_global_scores_test(present_dir, True)
            absent_error = calc_global_scores_test(absent_dir, False)

            raw_error = {'tp': present_error['correct'],
                         'fp': absent_error['total']-absent_error['correct'],
                         'tn': absent_error['correct'],
                         'fn': present_error['total']-present_error['correct']}
            
            # calculate relative error
            rel_error = {'prec':0, 'recall':0, 'f1':0}
            # precision = tp / (tp+fp)
            rel_error['prec'] = raw_error['tp'] / (raw_error['tp']+raw_error['fp'])
            # recall = tp / (tp+fn)
            rel_error['recall'] = raw_error['tp'] / (raw_error['tp']+raw_error['fn'])
            # f1 = 2 * [(prec*rec)/(prec+rec)]
            rel_error['f1'] = 2 * \
                    ((rel_error['prec'] * rel_error['recall']) / \
                    (rel_error['prec'] + rel_error['recall']))

            with open("error_report.csv","w", newline='') as f:
                writer = csv.writer(f)

                for key in raw_error.keys():
                    writer.writerow([key] + [raw_error[key]])
                writer.writerow(['----'])
                for key in rel_error.keys():
                    writer.writerow([key] + [rel_error[key]])
        # do not evaluate performance
        else:
            calc_global_scores(global_dir)

    # Calculate local scores
    else:
        calc_local_scores(local_dir)
