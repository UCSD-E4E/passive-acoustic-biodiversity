import os
import sys
import csv
import pickle
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch import nn
from torch.utils.data import DataLoader
from network import TweetyNet
import librosa
from librosa import display
from microfaune.audio import wav2spc, create_spec, load_wav
from glob import glob

from torch.utils.data import Dataset
from CustomAudioDataset import CustomAudioDataset

import matplotlib.pyplot as plt
import scipy

from TweetyNetModel import TweetyNetModel  

def load_dataset(data_path, n_mels, use_dump=True):
    mel_dump_file = os.path.join(data_path, "mel_dataset.pkl")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = compute_feature(data_path, n_mels)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 431]
    X = np.array([dataset["X"][i].transpose() for i in inds])
    Y = np.array([int(dataset["Y"][i]) for i in inds])
    uids = [dataset["uids"][i] for i in inds]
    return X, Y, uids

def compute_feature(data_path, n_mels):
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    labels_file = os.path.join(data_path, "labels.csv")
    print(labels_file)
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            reader = csv.reader(f, delimiter=',')
            labels = {}
            next(reader)  # pass fields names
            for name, _, y in reader:
                labels[name] = y
    else:
        print("Warning: no label file detected.")
        wav_files = glob(os.path.join(data_path, "wav/*.wav"))
        labels = {os.path.basename(f)[:-4]: None for f in wav_files}
    i = 1
    X = []
    Y = []
    uids = []
    for file_id, y in labels.items():
        print(f"{i:04d}/{len(labels)}: {file_id:20s}", end="\r")
        spc = wav2spc(os.path.join(data_path, "wav", f"{file_id}.wav"), n_mels=n_mels)
        X.append(spc)
        Y.append(y)
        uids.append(file_id)
        i += 1
    return {"uids": uids, "X": X, "Y": Y}

def split_dataset(X, Y, test_size=0.2, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test

def stratified_subset_dataset(X, Y, uids):
    _, indices = split_dataset(X, Y, test_size=0.01)
    return X[indices, :, :, np.newaxis], Y[indices], uids[indices] 

def main():
    train = True
    fineTuning = False
    #needs at least 80 for mel spectrograms ## may be able to do a little less, but must be greater than 60
    n_mels=72 # The closest we can get tmeporally is 72 with an output of 432 : i think it depends on whats good
    #this number should be proportional to the length of the videos. 
    datasets_dir = "/Users/mugetronblue/E4E/AcousticSpecies/passive-acoustic-biodiversity/TweetyNET/data"
    #datasets_dir = "/home/e4e/e4e_nas_aid/"

    X0, Y0, uids0 = load_dataset(os.path.join(datasets_dir, "ff1010bird_wav"), n_mels)
    X1, Y1, uids1 = load_dataset(os.path.join(datasets_dir, "warblrb10k_public_wav"), n_mels)

    X = np.concatenate([X0, X1]).astype(np.float32)/255
    Y = np.concatenate([Y0, Y1])
    uids = np.concatenate([uids0, uids1])
    del X0, Y0, uids0, X1, Y1, uids1 

    X, Y, uids = stratified_subset_dataset(X, Y, uids)
    ind_train_val, ind_test = split_dataset(X, Y)
    ind_train, ind_val = split_dataset(X[ind_train_val, :, :, np.newaxis], Y[ind_train_val], test_size=0.1)
    X_train, X_test, X_val = X[ind_train, :, :, np.newaxis], X[ind_test, :, :, np.newaxis], X[ind_val, :, :, np.newaxis]
    Y_train, Y_test, Y_val = Y[ind_train], Y[ind_test], Y[ind_val]
    uids_train, uids_test, uids_val = uids[ind_train], uids[ind_test], uids[ind_val]
    del X, Y, uids
    print("Training set :", Counter(Y_train))
    print("Testing set :", Counter(Y_test))
    print("Val set :", Counter(Y_val))

    train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
    test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
    val_dataset = CustomAudioDataset(X_val, Y_val, uids_val)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    #tweetynet = TweetyNetModel(len(Counter(Y_train)), (1, n_mels, 431), device, binary=True)

    #history, test_out, start_time, end_time = tweetynet.train_pipeline(train_dataset, val_dataset, test_dataset, 
    #                                                                   lr=.005, batch_size=6,epochs=100, save_me=True,
    #                                                                   fine_tuning=False, finetune_path=None)
    #print(end_time - start_time)
    #with open('icml_history.pkl', 'wb') as f:
    #    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    
    #test_out.to_csv("icml_test_predictions.csv")

main()
