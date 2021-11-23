import os
import sys
import csv
import math
import pickle
from collections import Counter
from datetime import datetime
import wave

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader
from network import TweetyNet
import librosa
from librosa import display
from audio import wav2spc, create_spec, load_wav
from glob import glob

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


def load_dataset(data_path, folder, SR, n_mels, frame_size, hop_length, use_dump=True):
    mel_dump_file = os.path.join(data_path, "{}_bin_mel_dataset.pkl".format(folder))
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = compute_feature(data_path, folder, SR, n_mels, frame_size, hop_length)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 216]
    X = np.array([dataset["X"][i].transpose() for i in inds]).astype(np.float32)/255
    Y = np.array([dataset["Y"][i] for i in inds])
    uids = np.array([dataset["uids"][i] for i in inds])
    return X, Y, uids


def compute_feature(data_path, folder, SR, n_mels, frame_size, hop_length):
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    features = {"uids": [], "X": [], "Y": []}
    filenames = os.listdir(os.path.join(data_path, folder))
    "Recordings in the format of nips4b_birds_{folder}filexxx.wav"
    "annotations in the format annotation_{folder}xxx.csv"
    tags = create_tags(data_path, folder)
    for f in filenames:
        spc = wav2spc(os.path.join(data_path, folder, f), fs=SR, n_mels=n_mels, downsample=True)
        Y = compute_Y(f, spc, tags, data_path, folder, SR, frame_size, hop_length)
        features["uids"].append(f)
        features["X"].append(spc)
        features["Y"].append(Y)
    return features


def compute_Y(f, spc, tags, data_path, folder, SR, frame_size, hop_length):
    file_num = f.split("file")[-1][:3]
    fpath = os.path.join(data_path, "temporal_annotations_nips4b", "".join(["annotation_", folder, file_num, ".csv"]))
    if os.path.isfile(fpath):
        x, sr = librosa.load(os.path.join(data_path, folder, f), sr=SR)
        annotation = pd.read_csv(fpath, index_col=False, names=["start", "duration", "tag"])
        y = calc_Y(x, sr, spc, annotation, tags, frame_size, hop_length)
        return np.array(y)
    else:
        print("file does not exist: ", f)
    return [0] * spc.shape[1]


def calc_Y(x, sr, spc, annotation, tags, frame_size, hop_length):
    y = [0] * spc.shape[1]
    for i in range(len(annotation)):
        start = get_frames(annotation.loc[i, "start"] * sr, frame_size, hop_length)
        end = get_frames((annotation.loc[i, "start"] + annotation.loc[i, "duration"]) * sr, frame_size, hop_length)
        for j in range(math.floor(start), math.floor(end)):
            y[j] = 1 # For binary use. add if statement later tags[annotation.loc[i, "tag"]]
    return y


def split_dataset(X, Y, test_size=0.2, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test

def old_load_dataset(data_path, n_mels, use_dump=True):
    mel_dump_file = os.path.join(data_path, "NIPS_bin_mel_dataset.pkl")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = old_compute_feature(data_path, n_mels)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    #segmentation will be needed
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 431]
    X = np.array([dataset["X"][i].transpose() for i in inds])
    Y = np.array([dataset["Y"][i] for i in inds])
    uids = np.array([dataset["uids"][i] for i in inds])
    return X, Y, uids

def old_compute_feature(data_path, n_mels):
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
        spc = wav2spc(os.path.join(data_path, "wav", f"{file_id}.wav"), n_mels=n_mels, downsample=True)
        X.append(spc)
        Y.append(np.array([y]*(spc.shape[1]+1)))
        uids.append(file_id)
        i += 1
    return {"uids": uids, "X": X, "Y": Y}

def newest_load_dataset(data_path, the_label, n_mels, use_dump=True):
    mel_dump_file = os.path.join(data_path, "downsampled_NIPS_bin_mel_dataset.pkl")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = newest_compute_feature(data_path, the_label, n_mels)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    print("basic segmentation", data_path)
    return basic_segmentation(dataset, 216)

def basic_segmentation(dataset, segment_size=216):
    X = []
    Y = []
    uids = []
    for i in range(len(dataset["X"])):
        start_pos = 0
        end_pos = segment_size
        spec = dataset["X"][i]
        j = 0
        if spec.shape[1] == segment_size:
            X.append(spec)
            Y.append(dataset["Y"][i])
            uids.append("_".join([str(j), str(start_pos), str(end_pos), dataset["uids"][i]]))

        elif spec.shape[1] > end_pos:
            while spec.shape[1] >= end_pos:
                X.append(spec[:, start_pos:end_pos])
                Y.append(dataset["Y"][i][start_pos:end_pos])
                uids.append("_".join([str(j), str(start_pos), str(end_pos), dataset["uids"][i]]))
                start_pos += segment_size
                end_pos += segment_size
                j+=1

        #let's just truncate the last bit
        if spec.shape[1] < end_pos:
            if spec.shape[1] - (segment_size) < 0:
                print("skipping: ", dataset["uids"][i])
            """
            else:
                X.append(spec[:, spec.shape[1]-segment_size:spec.shape[1]])
                Y.append(dataset["Y"][i][spec.shape[1]-segment_size:spec.shape[1]])
                uids.append("_".join([str(j), str(start_pos), str(end_pos), dataset["uids"][i]]))
            """

    X = np.array(X)
    Y = np.array(Y)
    uids = np.array(uids)
    return X, Y, uids
    

def newest_compute_feature(data_path, the_label, n_mels):
    print(f"Compute features for dataset {os.path.basename(data_path)}")
    file_names = os.listdir(data_path)
    i = 1
    X = []
    Y = []
    uids = []
    for file_id in file_names:
        if file_id[-4:].lower() == ".wav":
            print(f"{i:04d}/{len(file_names)}: {file_id:20s}", end="\r")
            #must add segmenting
            print(os.path.join(data_path, file_id))
            #x, sr  = librosa.load(os.path.join(data_path, file_id))
            sr = get_sr(os.path.join(data_path, file_id))
            spc = wav2spc(os.path.join(data_path, file_id), fs=sr, n_mels=n_mels, downsample=True)
            X.append(spc)
            Y.append(np.array([the_label]*(spc.shape[1])))
            uids.append(file_id)
            i += 1
    return {"uids": uids, "X": X, "Y": Y}

def get_sr(file_path):
     with wave.open(file_path, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        print(frame_rate)
        return frame_rate


def main():
    train = True
    fineTuning = False
    SR=44100
    HOP_LENGTH = 1024
    FRAME_SIZE = 2048
    #needs at least 80 for mel spectrograms ## may be able to do a little less, but must be greater than 60
    n_mels=72 # The closest we can get tmeporally is 72 with an output of 432 : i think it depends on whats good
    #this number should be proportional to the length of the videos.
    #datasets_dir = "/Users/mugetronblue/E4E/AcousticSpecies/passive-acoustic-biodiversity/TweetyNET/data/"#NIPS4BPlus"
    datasets_dir = "/Users/mugetronblue/E4E/AcousticSpecies/passive-acoustic-biodiversity/TweetyNET/data/ICML_DATASET/ICML_Test_Datasets/Xeno-canto-Google_AudioSet"
    #datasets_dir = "/home/e4e/e4e_nas_aid/nips4bplus/NIPS4BPlus"
    folder = "train"
    #X, Y, uids = load_dataset(datasets_dir, folder, SR, n_mels, FRAME_SIZE, HOP_LENGTH, use_dump=True)
    X0, Y0, uids0 = newest_load_dataset(os.path.join(datasets_dir, "mono", "Bird", "XCSelection"), 1, n_mels)
    X1, Y1, uids1 = newest_load_dataset(os.path.join(datasets_dir, "mono", "Bird", "Xeno_Canto_High_Priority"), 1, n_mels)
    X2, Y2, uids2 = newest_load_dataset(os.path.join(datasets_dir, "mono", "NonBird"), 0, n_mels)
    print("Finished_loading data")
    print(X0.shape, X1.shape, X2.shape)
    #means there are no 431 frame audios in the other two folders
    X = np.concatenate([X0, X1, X2]).astype(np.float32)/255
    Y = np.concatenate([Y0, Y1, Y2]).astype(int)
    uids = np.concatenate([uids0, uids1, uids2])
    print(X.shape, Y.shape, uids.shape)
    print(X[0].shape, Y[0].shape, uids[0].shape)
    print(type(X[0]), type(Y[0]), type(uids[0]))
    print(sum(Y.flatten()), len(Y.flatten()), sum(Y.flatten())/len(Y.flatten()))
    #all_tags = create_tags(datasets_dir, folder)
    all_tags = [0,1]
    test_dataset = CustomAudioDataset(X, Y, uids)
    print(test_dataset[0])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    tweetynet = TweetyNetModel(len(Counter(all_tags)), (1, n_mels, 216), device, binary=False)
    test_out = tweetynet.test_load_step(test_dataset, model_weights="OnlyBirds/model_weights-20210821_145528.h5")
    test_out.to_csv("50_OnlyBirdsi_google_Predictions.csv")

main()
