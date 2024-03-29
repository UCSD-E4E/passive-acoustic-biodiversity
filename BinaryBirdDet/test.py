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
def calc_local_scores(bird_dir, nonbird_dir):
    # TODO optimize detector.predict to not have to take in real file, just numpy arr
    # init detector
    detector = RNNDetector()
    # init labels dict
    annotations = []

    # generate local scores for every bird file in chosen directory
    for audio_file in os.listdir(bird_dir):
        # skip directories
        if os.path.isdir(bird_dir+audio_file): continue
        
        # read file
        raw_sample_rate, raw_samples = wavfile.read(bird_dir + audio_file)
        
        # downsample the sample if > 22.05 kHz
        if raw_sample_rate > 22050:
            rate_ratio = 22050 / raw_sample_rate
            samples = scipy_signal.resample(
                    raw_samples, int(len(raw_samples)*rate_ratio))
            sample_rate = 22050
            # resample produces unreadable float32 array so convert back
            samples = np.asarray(samples, dtype=np.int16)
            
            # add DS to end of downsampled file
            new_filename = audio_file[:-4] + "_DS" + audio_file[-4:]
            audio_file = new_filename
            
            # write downsampled file
            wavfile.write(bird_dir + new_filename, sample_rate, samples)
        else:
            sample_rate = raw_sample_rate
            samples = raw_samples
        
        # convert mono to stereo if needed
        if len(samples.shape) == 2:
            samples = samples.sum(axis=1) / 2

        # detection
        try:
            # for wavs
            if audio_file.lower().endswith('.wav'):
                _, local_score = detector.predict_on_wav(bird_dir + audio_file)
                print("Loaded", audio_file)
            # for mp3s
            elif audio_file.lower().endswith('.mp3'):
                _, local_score = detector.predict(bird_dir + audio_file)
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
        # not needed for csv output to opensoundscape
        # needed for matplotlib graph of local scores
        with open("score_files/XCSubset/" + audio_file[:-4]+"_LS.txt", "w") as f:
            f.write(str(duration) + "\n")
            f.write(str(len(local_score))+"\n")
            for sc in local_score:
                f.write(str(sc) + '\n')
        
        # isolate bird sounds in the clip by eliminating dead noise
        new_entry = isolate(local_score, samples, sample_rate, bird_dir, audio_file)
        annotations.append(new_entry)

    # generate local scores for every nonbird file in chosen directory
    for audio_file in os.listdir(nonbird_dir):
        # skip directories
        if os.path.isdir(nonbird_dir+audio_file): continue
    
        # read file
        raw_sample_rate, raw_samples = wavfile.read(nonbird_dir + audio_file)
        
        # downsample the sample if > 44.1 kHz
        if raw_sample_rate > 22050:
            rate_ratio = 22050 / raw_sample_rate
            samples = scipy_signal.resample(
                    raw_samples, int(len(raw_samples)*rate_ratio))
            sample_rate = 22050
            # resample produces unreadable float32 array so convert back
            samples = np.asarray(samples, dtype=np.int16)
            
            # add DS to end of downsampled file
            new_filename = audio_file[:-4] + "_DS" + audio_file[-4:]
            audio_file = new_filename
            
            # write downsampled file
            wavfile.write(nonbird_dir + new_filename, sample_rate, samples)
        else:
            sample_rate = raw_sample_rate
            samples = raw_samples
        
        # convert mono to stereo if needed
        if len(samples.shape) == 2:
            samples = samples.sum(axis=1) / 2

        # get duration of clip
        duration = len(samples) / sample_rate

        # create entry for audio clip
        new_entry = {'folder'  : nonbird_dir,
                     'file'    : audio_file,
                     'channel' : 0,
                     'duration': duration,
                     'stamps'  : [[0,duration]],
                     'labels'  : [0]}
        annotations.append(new_entry)

    # write csv with time stamps and labels
    header = ["FOLDER","IN FILE","CHANNEL","OFFSET","DURATION","MANUAL ID"]
    with open("annotations.csv", "w") as f:
        writer = csv.writer(f)
        # write titles of columns
        writer.writerow(header)
        
        for el in annotations:
            for i in range(len(el['stamps'])):
                clip_duration = el['stamps'][i][1] - el['stamps'][i][0]
                writer.writerow([el['folder'],
                                 el['file'], 
                                 el['channel'],
                                 el['stamps'][i][0], # start time or offset 
                                 clip_duration, # end - start
                                 el['labels'][i]])


def isolate(scores, samples, sample_rate, audio_dir, filename):
    # calculate original duration
    old_duration = len(samples) / sample_rate

    # create entry for audio clip
    entry = {'folder'  : audio_dir,
             'file'    : filename,
             'channel' : 0,
             'duration': old_duration,
             'stamps'  : [],
             'labels'  : []}

    # treshold is 'thresh_mult' times above median score value
    thresh_mult = 2
    thresh = np.median(scores) * thresh_mult

    # how many samples one score represents
    samples_per_score = len(samples) // len(scores)
    
    # isolate samples that produce a score above thresh
    isolated_samples = np.empty(0, dtype=np.int16)
    prev_cap = 0        # sample idx of previously captured
    for i in range(len(scores)):
        # if a score hits or surpasses thresh, capture 1s on both sides of it
        if scores[i] >= thresh:
            # score_pos is the sample index that the score corresponds to
            score_pos = i * samples_per_score
 
            # upper and lower bound of captured call
            # sample rate is # of samples in 1 second: +-1 second
            lo_idx = max(0, score_pos - sample_rate)
            hi_idx = min(len(samples), score_pos + sample_rate)
            lo_time = lo_idx / sample_rate
            hi_time = hi_idx / sample_rate
            
            # calculate start and end stamps
            # create new sample if not overlapping or if first stamp
            if prev_cap < lo_idx or prev_cap == 0:
                new_stamp = [lo_time, hi_time]
                entry['stamps'].append(new_stamp)
                entry['labels'].append(1)
            # extend same stamp if still overlapping
            else:
                entry['stamps'][-1][1] = hi_time

            # mark previously captured to prevent overlap collection
            lo_idx = max(prev_cap, lo_idx)
            prev_cap = hi_idx

            # add to isolated samples
            # sub-clip numpy array
            isolated_samples = np.append(isolated_samples,samples[lo_idx:hi_idx])

    # calculate new duration
    new_duration = len(isolated_samples) / sample_rate
    percent_reduced = 1 - (new_duration / old_duration)
    print('Reduced {} from {:.2f}s to {:.2f}s. {:.2%} reduced.'.format( \
            filename, old_duration, new_duration, percent_reduced))
        
    # write file
    # new_filename = filename[:-4] + "_RED" + filename[-4:]
    # wavfile.write(audio_dir + new_filename, sample_rate, isolated_samples)
    
    return entry
    

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
    # local_dir = os.path.join(
    #         home, "../../media/e4e/New Volume/XCSelection/Subset/")
    local_bird_dir = os.path.join(
            home, "../../media/e4e/Rainforest_Data1/XCSelectionTrain/")
    local_nonbird_dir = os.path.join(
            home, "../../media/e4e/Rainforest_Data1/audioset_nonbird_train/")

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
        calc_local_scores(local_bird_dir, local_nonbird_dir)
