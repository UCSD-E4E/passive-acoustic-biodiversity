import os
import sys
import csv
import math
import pickle
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader
from network import TweetyNet
import librosa
from librosa import display
from microfaune.audio import wav2spc, create_spec, load_wav
from glob import glob
import random
import scipy.signal as scipy_signal

from torch.utils.data import Dataset
from CustomAudioDataset import CustomAudioDataset
from TweetyNetModel import TweetyNetModel

def get_frames(x, frame_size, hop_length):
    return ((x) / hop_length) + 1#(x - frame_size)/hop_length + 1
def frames2seconds(x, sr):
    return x/sr
def find_tags(data_path, folder):
    fnames = os.listdir(os.path.join(data_path, "temporal_annotations_nips4b"))
    csvs = []
    for f in fnames:
        csvs.append(pd.read_csv(os.path.join(data_path, "temporal_annotations_nips4b", f), index_col=False, names=["start", "duration", "tag"]))
    return csvs

def create_tags(data_path, folder):
    csvs = find_tags(data_path, folder)
    tags = [csv["tag"] for csv in csvs]
    tag = []
    for t in tags:
        for a in t:
            tag.append(a)
    tag = set(tag)
    tags = {"None": 0}
    for i, t in enumerate(sorted(tag)):
        tags[t] = i + 1
    return tags


def load_dataset(data_path, folder, SR, n_mels, frame_size, hop_length, nonBird_labels, found, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_{}_bin_mel_dataset.pkl".format(folder))
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = compute_feature(data_path, folder, SR, n_mels, frame_size, hop_length, nonBird_labels, found)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 216]
    X = np.array([dataset["X"][i].transpose() for i in inds]).astype(np.float32)/255
    Y = np.array([dataset["Y"][i] for i in inds])
    uids = np.array([dataset["uids"][i] for i in inds])
    return X, Y, uids


def compute_feature(data_path, folder, SR, n_mels, frame_size, hop_length, nonBird_labels, found):
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    features = {"uids": [], "X": [], "Y": []}
    filenames = os.listdir(os.path.join(data_path, folder))
    "Recordings in the format of nips4b_birds_{folder}filexxx.wav"
    "annotations in the format annotation_{folder}xxx.csv"
    tags = create_tags(data_path, folder)
    for f in filenames:
		#signal, SR = downsampled_mono_audio(signal, sample_rate, SR)
        spc = wav2spc(os.path.join(data_path, folder, f), fs=SR, n_mels=n_mels)
        Y = compute_Y(f, spc, tags, data_path, folder, SR, frame_size, hop_length, nonBird_labels, found)
        features["uids"].append(f)
        features["X"].append(spc)
        features["Y"].append(Y)
    return features


def compute_Y(f, spc, tags, data_path, folder, SR, frame_size, hop_length, nonBird_labels, found):
    file_num = f.split("file")[-1][:3]
    fpath = os.path.join(data_path, "temporal_annotations_nips4b", "".join(["annotation_", folder, file_num, ".csv"]))
    if os.path.isfile(fpath):
        x, sr = librosa.load(os.path.join(data_path, folder, f), sr=SR)
        annotation = pd.read_csv(fpath, index_col=False, names=["start", "duration", "tag"])
        y = calc_Y(x, sr, spc, annotation, tags, frame_size, hop_length, nonBird_labels, found)
        return np.array(y)
    else:
        print("file does not exist: ", f)
    return [0] * spc.shape[1]


def calc_Y(x, sr, spc, annotation, tags, frame_size, hop_length, nonBird_labels, found):
    y = [0] * spc.shape[1]
    for i in range(len(annotation)):
        start = get_frames(annotation.loc[i, "start"] * sr, frame_size, hop_length)
        end = get_frames((annotation.loc[i, "start"] + annotation.loc[i, "duration"]) * sr, frame_size, hop_length)
        #print(annotation["tag"], len(annotation["tag"]))
        if annotation["tag"][i] not in nonBird_labels:
            for j in range(math.floor(start), math.floor(end)):
                y[j] = 1 # For binary use. add if statement later tags[annotation.loc[i, "tag"]]
        else: 
            #print(str(annotation["tag"][i]))
            found[str(annotation["tag"][i])] += 1
    return y


def split_dataset(X, Y, test_size=0.2, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test

def get_pos_total(Y):
    pos, total = 0,0
    for y in Y:
        pos += sum(y)
        total += len(y)
    #print(pos, total, pos/total, len(Y))
    return pos, total

def random_split_to_fifty(X, Y, uids):
    pos, total = get_pos_total(Y)
    j = 0
    while (pos/total < .50):
        idx = random.randint(0, len(Y)-1)
        if (sum(Y[idx])/Y.shape[1] < .5):
            #print(uids[idx],(sum(Y[idx])/Y.shape[1]))
            X = np.delete(X, idx, axis=0)
            Y = np.delete(Y, idx, axis=0)
            uids = np.delete(uids, idx, axis=0)
            #print(j, pos/total)
            j += 1

        pos, total = get_pos_total(Y)
    return X, Y, uids



def main():
    train = True
    fineTuning = False
    SR=44100
    HOP_LENGTH = 1024
    FRAME_SIZE = 2048
    #needs at least 80 for mel spectrograms ## may be able to do a little less, but must be greater than 60
    n_mels=72 # The closest we can get tmeporally is 72 with an output of 432 : i think it depends on whats good
    #this number should be proportional to the length of the videos.
    datasets_dir = "/Users/mugetronblue/E4E/AcousticSpecies/passive-acoustic-biodiversity/TweetyNET/data/NIPS4BPlus"
    #datasets_dir = "/home/e4e/e4e_nas_aid/nips4bplus/NIPS4BPlus"
    nonBird_labels = ["Plasab_song", "Unknown", "Tibtom_song", "Lyrple_song", "Plaaff_song", "Pelgra_call", "Cicatr_song", "Cicorn_song", "Tetpyg_song", "Ptehey_song"]
    found = {"Plasab_song": 0, "Unknown": 0, "Tibtom_song": 0, "Lyrple_song": 0, "Plaaff_song": 0, "Pelgra_call": 0, "Cicatr_song": 0, "Cicorn_song": 0, "Tetpyg_song": 0, "Ptehey_song": 0}
    #keep track of how many occurences we find.
    #get a sample so there is even distribution = 50-50%
    folder = "train"
    X, Y, uids = load_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, nonBird_labels, found, use_dump=True)
    test_dataset = CustomAudioDataset(X, Y, uids)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    all_tags = [0,1]
    tweetynet = TweetyNetModel(len(Counter(all_tags)), (1, n_mels, 216), device, binary=False)
    #***Specify your own weights***
    test_out = tweetynet.test_load_step(test_dataset, model_weights="OnlyBirds/model_weights-20210821_145528.h5")
    test_out.to_csv("Evaluation_on_nips.csv")

main()
