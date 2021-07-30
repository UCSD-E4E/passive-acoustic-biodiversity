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
from microfaune.audio import wav2spc
from glob import glob

from torch.utils.data import Dataset
from CustomAudioDataset import CustomAudioDataset

def load_dataset(data_path, use_dump=True):
    mel_dump_file = os.path.join(data_path, "mel_dataset.pkl")
    if os.path.exists(mel_dump_file) and use_dump:
        with open(mel_dump_file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = compute_feature(data_path)
        with open(mel_dump_file, "wb") as f:
            pickle.dump(dataset, f)
    inds = [i for i, x in enumerate(dataset["X"]) if x.shape[1] == 431]
    X = np.array([dataset["X"][i].transpose() for i in inds])
    Y = np.array([int(dataset["Y"][i]) for i in inds])
    uids = [dataset["uids"][i] for i in inds]
    return X, Y, uids

def compute_feature(data_path):
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
        spc = wav2spc(os.path.join(data_path, "wav", f"{file_id}.wav"))
        X.append(spc)
        Y.append(y)
        uids.append(file_id)
        i += 1
    return {"uids": uids, "X": X, "Y": Y}

def split_dataset(X, Y, random_state=0):
    split_generator = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    ind_train, ind_test = next(split_generator.split(X, Y))
    X_train, X_test = X[ind_train, :, :], X[ind_test, :, :]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    return ind_train, ind_test

def create_dataset(datasets_dir):
    X0, Y0, uids0 = load_dataset(os.path.join(datasets_dir, "ff1010bird_wav"))
    X1, Y1, uids1 = load_dataset(os.path.join(datasets_dir, "warblrb10k_public_wav"))
    print(X0.shape, X1.shape)
    print(Y0.shape, Y1.shape)
    print(len(uids0), len(uids1))

    X = np.concatenate([X0, X1]).astype(np.float32) / 255
    Y = np.concatenate([Y0, Y1])
    uids = np.concatenate([uids0, uids1])
    print(X.shape, Y.shape, uids.shape)
    del X0, X1, Y0, Y1

    ind_train, ind_test = split_dataset(X, Y)
    X_train, X_test = X[ind_train, :, :, np.newaxis], X[ind_test, :, :, np.newaxis]
    Y_train, Y_test = Y[ind_train], Y[ind_test]
    uids_train, uids_test = uids[ind_train], uids[ind_test]
    del X, Y
    print("Training set: ", Counter(Y_train))
    print("Test set: ", Counter(Y_test))
    train_dataset = CustomAudioDataset(X_train, Y_train, uids_train)
    test_dataset = CustomAudioDataset(X_test, Y_test, uids_test)
    return train_dataset, test_dataset

def CreateTweetyNet(device):
    network = TweetyNet(num_classes = 2,
             input_shape=(431, 40, 1),
             padding='same',
             conv1_filters=32,
             conv1_kernel_size=(5, 5),
             conv2_filters=64,
             conv2_kernel_size=(5, 5),
             pool1_size=(8, 1),
             pool1_stride=(8, 1),
             pool2_size=(4, 1),
             pool2_stride=(4, 1),
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


def training_step(model, device, train_loader, val_loader, criterion, optimizer, scheduler, epoch):
    history = {"loss": [],
               "val_loss": [],
               "acc": [],
               "val_acc": []
               }
    for e in range(epoch):  # loop over the dataset multiple times
        model.train(True)
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(train_loader, 0):
            scheduler.step()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data  # , input_lengths, label_lengths = data
            labels = labels.reshape(labels.shape[0], 1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs, len(inputs), len(labels))
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # get statistics
            running_loss += loss.item()
            output = torch.argmax(output, dim=1)
            correct += (output == labels).float().sum()

            # print update
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (e + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        history["loss"].append(running_loss)
        history["acc"].append(100 * correct / len(train_loader.dataset))
        validation_step(model, device, val_loader, criterion, history)
    print('Finished Training')
    return history

def validation_step(model, device, val_loader, criterion, history):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_correct = 0.0
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data  # , input_lengths, label_lengths = data
            labels = labels.reshape(labels.shape[0], 1)

            output = model(inputs, len(inputs), len(labels))
            loss = criterion(output, labels)

            # get statistics
            val_loss += loss.item()
            output = torch.argmax(output, dim=1)
            val_correct += (output == labels).float().sum()
        history["val_loss"].append(val_loss)
        history["val_acc"].append(100 * val_correct / len(val_loader.dataset))

def testing_step(model, device, test_loader):
    predictions = pd.DataFrame()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels, uids = data
            labels = labels.reshape(labels.shape[0], 1)
            uids = np.array([u for u in uids])
            output = model(inputs, len(inputs), len(labels))
            output = torch.argmax(output, dim=1)
            d = {"uid": uids.flatten(), "pred": output.flatten(), "label": labels.flatten()}
            new_preds = pd.DataFrame(d)
            predictions = predictions.append(new_preds)
    print('Finished Testing')
    return predictions

def train_pipeline(model, device, train_dataset, test_dataset,
                   train = True, fineTuning = False, save_me = False, finetune_path = None):
    batch_size = 64
    lr = .001
    epochs = 5

    if train:
        if fineTuning:
            model.load_weights(finetune_path)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.Adam(params=model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=lr,
                                                        steps_per_epoch=int(len(train_data_loader)),
                                                        epochs=epochs,
                                                        anneal_strategy='linear')

        # Can implement when we actually start training the model
        # es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        # mc = keras.callbacks.ModelCheckpoint(filepath='best_model_weights.h5', save_weights_only=True, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                              patience=5, min_lr=1e-5)
        history = training_step(model, device, train_data_loader, val_data_loader, criterion, optimizer, scheduler,
                                    epochs)
        if save_me:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), f"model_weights-{date_str}.h5")
        print_results(history)
        return history
def print_results(history):
    plt.figure(figsize=(9, 6))
    plt.title("Loss")
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.savefig('loss.png')

    plt.figure(figsize=(9, 6))
    plt.title("Accuracy")
    plt.plot(history["acc"])
    plt.plot(history["val_acc"])
    plt.legend(["acc", "val_acc"])
    plt.savefig('acc.png')



def main(argv):
    train = True
    fineTuning = False
    datasets_dir = "/Users/mugetronblue/E4E/AcousticSpecies/passive-acoustic-biodiversity/TweetyNET/data"
    train_dataset, test_dataset = create_dataset(datasets_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = CreateTweetyNet(device)
    history = train_pipeline(model, device, train_dataset, test_dataset)
    np.save('acc_loss.npy', history)

if __name__ == "__main__":
   main(sys.argv[1:])
