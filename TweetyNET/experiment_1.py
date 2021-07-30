import os
import sys
import random
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

def load_dataset(data_path, bird_name, use_dump=True):
    mel_dump_file = os.path.join(data_path, bird_name, "mel_dataset.pkl")
    dataset = None
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("FAILED")
        return 0
        """
        dataset = compute_bird_features(os.path.join(data_path, bird_name))
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
        """
    if dataset == None:
       print("BIG_FAIL")
    inds = [i for i, x in enumerate(dataset["X"])]
    X = np.array([dataset["X"][i].transpose() for i in inds])
    Y = np.array([dataset["Y"][i] for i in inds])
    uids = np.array([dataset["uids"][i] for i in inds])
    return X, Y, uids

def split_dataset(X, Y, test_size, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test

def get_time(path=None, data=None, sr=None):
    if path != None:
        y, _sr = librosa.load(path)
    elif data != None:
        y = data
        _sr = sr
    else:
        print("Usage: specify a path or the data.")
        return 0
    frames = [len(y)]
    return librosa.frames_to_time(frames, _sr, 1)
def get_bird_time(data_dir, bird_num, use_dump):
    # let's store all this info in a csv and push it to the server.
    the_path = os.path.join(data_dir, bird_num, "Wave")
    all_time = 0
    times = {"file": [], "time" : []}
    mel_dump_file = os.path.join(data_dir, bird_num, "time_dataset.pkl")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            times = pickle.load(f)
            all_time = times["time"][-1]
    else:
        files = os.listdir(the_path)
        for f in files:
            tim = get_time(os.path.join(the_path, f))
            all_time += tim
            times['file'].append(f)
            times['time'].append(tim)
        times['file'].append(bird_num)
        times['time'].append(all_time)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(times, f)
    return all_time, times


def split_data(X, Y, uids, time_dataset, train_size=900, test_size=400):
    test = {'X': [], 'Y': [], 'uids': []}
    train = {'X': [], 'Y': [], 'uids': []}
    val = {'X': [], 'Y': [], 'uids': []}
    # actually split it up now. then look at class distribution among classes.
    # labels = tag_videos(Y, uids, time_dataset)
    train_names, test_names, val_names = split_time(time_dataset, train_size, test_size)
    for i, f in enumerate(uids):
        if f in train_names:
            train["X"].append(X[i])
            train["Y"].append(Y[i])
            train["uids"].append(uids[i])

        elif f in test_names:
            test["X"].append(X[i])
            test["Y"].append(Y[i])
            test["uids"].append(uids[i])

        if f in val_names:
            val["X"].append(X[i])
            val["Y"].append(Y[i])
            val["uids"].append(uids[i])

    return train, test, val

def split_time(time_dataset, train_size, test_size):
    train = set()
    val = set()
    test = set()
    seen = set()
    train_amt = 0
    test_amt = 0
    val_amt = 0
    val_seen = False
    dataset = {'file' : time_dataset['file'][:-1], 'time' : time_dataset['time'][:-1]}
    #print(time_dataset['time'][-1])
    while ((train_amt + test_amt + val_amt < time_dataset['time'][-1]) and len(seen) != len(dataset['time'])):
        idx = random.randint(0, len(dataset['file'])-1)
        if idx not in seen:
            #print(train_amt + test_amt + val_amt)
            seen.add(idx)
            if (test_amt < test_size):
                test.add(dataset["file"][idx])
                test_amt += dataset["time"][idx]
            elif (train_amt < train_size):
                train.add(dataset["file"][idx])
                train_amt += dataset["time"][idx]
            else:
                val_seen  = True
                val.add(dataset["file"][idx])
                val_amt += dataset["time"][idx]
    if not val_seen :
        val.add(dataset["file"][0])
        val_amt += dataset["time"][0]
        
    print("Train Time: ", train_amt, "Number of files: ", len(train))
    print("Test Time: ", test_amt, "Number of files: ", len(test))
    print("Val Time: ", val_amt, "Number of files: ", len(val))
    return train, test, val

def get_window_time(X, fileidx, time_dataset, window_size):
    time_file = time_dataset["time"][fileidx][0]
    frames_file = len(X)
    return window_size * time_file / frames_file


def window_spectrograms(data, time_dataset, window_size=88, size=900):
    X = []
    Y = []
    uids = []
    seen = set()
    seconds = 0
    while (seconds < size):
        fileidx = 0
        if len(data["uids"]) > 1:
            fileidx = random.randint(0, len(data["uids"]) - 1)
        idx = random.randint(0, data["X"][fileidx].shape[0] - window_size - 1)
        timeidx = [x for x in range(len(time_dataset["file"])) if time_dataset["file"][x] == data["uids"][fileidx]][0]
        if (fileidx, idx) not in seen:
            seen.add((fileidx, idx))
            # print(fileidx, timeidx)
            # print(data["uids"][fileidx], time_dataset["file"][timeidx], idx)
            # frequency bins need to be capped or 0 padded.
            X.append(data["X"][fileidx][idx:idx + window_size, :])
            Y.append(data["Y"][fileidx][idx:idx + window_size])
            uids.append(data["uids"][fileidx] + "_" + str(idx))
            seconds += get_window_time(data["X"][fileidx], timeidx, time_dataset, window_size)
    return {"X": np.array(X), "Y": np.array(Y), "uids": np.array(uids)}

def transpose_spectrograms(dataset):
    X = []
    for i in range(len(dataset["X"])):
        X.append(dataset["X"][i].transpose())
    dataset["X"] = np.array(X)

def create_dataset(data_dir, bird_folder, train_size, use_dump):
    X, Y, uids = load_dataset(data_dir, bird_folder, use_dump=True)
    X = np.array([x.astype(np.float32) / 255 for x in X])
    uids = uids
    print(X.shape, Y.shape, uids.shape)
    all_labels = get_labels(Y)
    time_dataset = get_bird_time(data_dir, bird_folder, use_dump)[-1]
    train, test, val = split_data(X, Y, uids, time_dataset, train_size=train_size, test_size=400)
    train_set = window_spectrograms(train, time_dataset, window_size=88, size=train_size)
    test_set = window_spectrograms(test, time_dataset, window_size=88, size=400)
    val_set = window_spectrograms(val, time_dataset, window_size=88, size=100)
    transpose_spectrograms(train_set)
    transpose_spectrograms(test_set)
    transpose_spectrograms(val_set)
    train_dataset = CustomAudioDataset(train_set["X"], train_set["Y"], train_set["uids"])
    test_dataset = CustomAudioDataset(test_set["X"], test_set["Y"], test_set["uids"])
    val_dataset = CustomAudioDataset(val_set["X"], val_set["Y"], val_set["uids"])
    return train_dataset, test_dataset, val_dataset, all_labels
def get_labels(Y):
    all_labels = set()
    for lab in Y:
        for idx in lab:
            all_labels.add(idx)
    return all_labels

def print_results(history, bird_folder, train_size):
    plt.figure(figsize=(9, 6))
    plt.title("Loss")
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.savefig(bird_folder+"_"+str(train_size)+'loss.png')

    plt.figure(figsize=(9, 6))
    plt.title("Accuracy")
    plt.plot(history["acc"])
    plt.plot(history["val_acc"])
    plt.legend(["acc", "val_acc"])
    plt.savefig(bird_folder+"_"+str(train_size)+'acc.png')

def frame_error(pred, actual):
    if len(pred) != len(actual):
        print("Incorrect Lengths: ", len(pred), len(actual))
        return 0
    match_up = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            match_up += 1
    return match_up#/len(pred)
# This only matters if things are in the correct order. Can I split by video file?

def syllable_edit_distance(pred, actual):
    if len(pred) != len(actual):
        print("Incorrect Lengths: ", len(pred), len(actual))
        return 0
    distances = range(len(pred) + 1)
    for i2, c2 in enumerate(actual):
        distances_ = [i2+1]
        for i1, c1 in enumerate(pred):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]#/len(pred)

def CreateTweetyNet(device, num_classes, input_shape):
    network = TweetyNet(num_classes = num_classes,
             input_shape = input_shape,
             padding='same',
             conv1_filters=32,
             conv1_kernel_size=(5, 5),
             conv2_filters=64,
             conv2_kernel_size=(5, 5),
             pool1_size=(8, 1),
             pool1_stride=(8, 1),
             pool2_size=(8, 1),
             pool2_stride=(8, 1),
             hidden_size=None,
             rnn_dropout=0.,
             num_layers=1
        )
    model = network.to(device)
    print(model)
    return model

def reset_weights(the_model):
    for name, module in the_model.named_children():
        if hasattr(module, 'reset_parameters'):
            print('resetting ', name)
            module.reset_parameters()


def training_step(model, device, train_loader, val_loader, window_size, criterion, optimizer, scheduler, epoch):
    history = {"loss": [],
               "val_loss": [],
               "acc": [],
               "val_acc": [],
               "edit_distance" : [],
               "val_edit_distance" : []
               }
    for e in range(epoch):  # loop over the dataset multiple times
        print("Start of epoch:", e)
        model.train(True)
        running_loss = 0.0
        correct = 0.0
        edit_distance = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data  # , input_lengths, label_lengths = data
            #labels = labels.reshape(1, labels.shape[0], labels.shape[1])
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape, labels.shape)
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs, inputs.shape[0], labels.shape[0])


            #print(output.shape)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # get statistics
            running_loss += loss.item()
            output = torch.argmax(output, dim=1)
            #print(output.shape)
            correct += (output == labels).float().sum()
            for j in range(len(labels)):
                edit_distance += syllable_edit_distance(output[j], labels[j])
            # print update
            if i % 10 == 9:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %(e + 1, i + 1, running_loss / 2000))
        history["loss"].append(running_loss)
        history["acc"].append(100 * correct / (len(train_loader.dataset) * window_size))
        history["edit_distance"].append(edit_distance / (len(train_loader.dataset) * window_size))
        validation_step(model, device, val_loader, window_size, criterion, history)
    print('Finished Training')
    return history

def validation_step(model, device, val_loader, window_size, criterion, history):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0.0
        val_edit_distance = 0.0
        for i, data in enumerate(val_loader):
            inputs, labels, _ = data  # , input_lengths, label_lengths = data
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape, labels.shape)
            # forward + backward + optimize
            output = model(inputs, inputs.shape[0], labels.shape[0])

            loss = criterion(output, labels)
            # get statistics
            val_loss += loss.item()
            output = torch.argmax(output, dim=1)
            #print(output.shape)
            #print((output == labels).shape, (output == labels).float().sum())
            val_correct += (output == labels).float().sum()
            for j in range(len(labels)):
                val_edit_distance += syllable_edit_distance(output[j], labels[j])
            #print(val_edit_distance)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(100 * val_correct / (len(val_loader.dataset) * window_size))
        history["val_edit_distance"].append(val_edit_distance / (len(val_loader.dataset) * window_size))

def testing_step(model, device, window_size, test_loader):
    predictions = pd.DataFrame()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, uids = data  # , input_lengths, label_lengths = data
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape, labels.shape)
            # forward + backward + optimize
            output = model(inputs, inputs.shape[0], labels.shape[0])
            output = torch.argmax(output, dim=1)
            # get statistics
            temp_uids = []
            for u in uids:
                for j in range(window_size):
                    temp_uids.append(u + "_" + str(j))
            uids = np.array(temp_uids)
            d = {"uid": uids.flatten(), "pred": output.flatten(), "label": labels.flatten()}
            new_preds = pd.DataFrame(d)
            predictions = predictions.append(new_preds)
    print('Finished Testing')
    return predictions


def train_pipeline(model, device, train_dataset, val_dataset, test_dataset, bird_folder, train_size, 
                   train=True, fineTuning=False, save_me=True, finetune_path=None):
    batch_size = 64
    lr = .005
    epochs = 80

    if train:
        if fineTuning:
            model.load_weights(finetune_path)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.Adam(params=model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=lr,
                                                        steps_per_epoch=int(len(train_data_loader)),
                                                        epochs=epochs,
                                                        anneal_strategy='linear')
        start_time = datetime.now()
        history = training_step(model, device, train_data_loader, val_data_loader, 88, criterion, optimizer, scheduler,
                                epochs)
        end_time = datetime.now()

        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_out = testing_step(model, device, 88, test_data_loader)

        if save_me:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), (bird_folder+"_"+str(train_size)+f"model_weights-{date_str}.h5"))
        print_results(history, bird_folder, train_size)
        return history, test_out, start_time, end_time


def main(argv):
    train = True
    fineTuning = False
    use_dump = True
    bird_folder = argv[0]
    train_size = int(argv[1])
    print(bird_folder)
    data_dir = "/Users/mugetronblue/E4E/AcousticSpecies/passive-acoustic-biodiversity/TweetyNET/data/BirdSong_Recognition/"
    #data_dir = "/home/e4e/e4e_nas_aid/BirdSong_Recognition/"
    train_dataset, test_dataset, val_dataset, all_labels = create_dataset(data_dir, bird_folder, train_size, use_dump)
    print(all_labels)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = CreateTweetyNet(device, len(all_labels), input_shape = (1, 1025, 88))
    hist, test_out, start_time, end_time = train_pipeline(model, device, train_dataset, val_dataset, test_dataset, bird_folder, train_size)
    print("Time elapsed:", end_time - start_time)
    np.save("bird_folder" + str(train_size) +'acc_loss.npy', hist)
    test_out.to_csv(bird_folder+"_predictions.csv")

if __name__ == "__main__":
   main(sys.argv[1:])
