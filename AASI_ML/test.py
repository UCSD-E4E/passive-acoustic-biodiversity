from microfaune_package.microfaune.detection import RNNDetector
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import numpy as np
import pdb
import csv
import argparse
from pydub import AudioSegment


"""
Parse input arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Detect birds")
    parser.add_argument('--global', dest='do_global_scores',
                        help='whether creating global scores or not',
                        action='store_true')
    args = parser.parse_args()
    
    return args


"""
This function creates a global score file
"""
def calc_global_scores(audio_dir, label):
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
This function creates a local score file

\/ Description of file it creates \/
line 1: Duration of audio clip in seconds
line 2: Amount of local scores were created
line 3: one local score each line
...
line 3+len(local_scores)
"""
def calc_local_scores(audio_dir):
    detector = RNNDetector()
    local_scores = []

    for audio_file in os.listdir(audio_dir):
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
        
        audio = AudioSegment.from_file(audio_dir + audio_file)
        duration = str(audio.duration_seconds)
        
        with open(audio_dir + audio_file[:-4]+"_LS.txt", "w") as f:
            f.write(duration)
            f.write(str(len(local_score)))
            for sc in local_score:
                f.write(str(sc) + '\n')


if __name__ == '__main__':
    args = parse_args()

    home = str(Path.home())
    present_dir = os.path.join(home, "../../media/e4e/New Volume/XCSelection/")
    absent_dir = os.path.join(home, "../../media/e4e/New Volume/audioset_nonbird/")
    test_dir = "test_dir/audio/"

    # Calculate global scores
    if args.do_global_scores:
        # Second parameter is True if birds present, False if absent
        present_error = calc_global_scores(present_dir, True)
        absent_error = calc_global_scores(absent_dir, False)

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

    # Calculate local scores
    else:
        calc_local_scores(test_dir)
