import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch

from torch.utils.data import DataLoader
from network import TweetyNet
from EvaluationFunctions import frame_error, syllable_edit_distance


# could turn this into a wrapper class and add flexibility
class TweetyNetModel:
    # Creates a tweetynet instance with training and evaluation functions.
    # input: num_classes = number of classes TweetyNet needs to classify
    #       input_shape = the shape of the spectrograms when fed to the model.
    #       ex: (1, 1025, 88) where (# channels, # of frequency bins/mel bands, # of frames)
    #       device: "cuda" or "cpu" to specify if machine will run on gpu or cpu.
    # output: None
    def __init__(self, num_classes, input_shape, device, epochs = 1, binary=False, criterion=None, optimizer=None):
        self.model = TweetyNet(num_classes=num_classes,
                               input_shape=input_shape,
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
        self.device = device
        self.model.to(device)
        self.binary = binary
        self.window_size = input_shape[-1]
        self.runtime = 0
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(params=self.model.parameters())
        self.epochs = epochs
        self.batchsize = 32
        self.n_train_examples = self.batchsize *30 
        self.n_valid_examples = self.batchsize *10 

"""
Function:
Input:
output:
purpose:
"""
    #let's not change the model just yet.
    def define_model(trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []

        in_features = 28 * 28
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, CLASSES))
        layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)
        # can add history here
        #print(self.model)


    @staticmethod
    def print_results(history, show_plots=False, save_plots=True):
        plt.figure(figsize=(9, 6))
        plt.title("Loss")
        plt.plot(history["loss"])
        plt.plot(history["val_loss"])
        plt.legend(["loss", "val_loss"])
        if save_plots:
            plt.savefig('loss.png')
        if show_plots:
            plt.show()

        plt.figure(figsize=(9, 6))
        plt.title("Accuracy")
        plt.plot(history["acc"])
        plt.plot(history["val_acc"])
        plt.legend(["acc", "val_acc"])
        if save_plots:
            plt.savefig('acc.png')
        if show_plots:
            plt.show()

        plt.figure(figsize=(9, 6))
        plt.title("Edit Distance")
        plt.plot(history["edit_distance"])
        plt.plot(history["val_edit_distance"])
        plt.legend(["edit_distance", "val_edit_distance"])
        if save_plots:
            plt.savefig('edit_distance.png')
        if show_plots:
            plt.show()

    def reset_weights(self):
        for name, module in self.model.named_children():
            if hasattr(module, 'reset_parameters'):
                print('resetting ', name)
                module.reset_parameters()

    def training_step(self, train_loader, val_loader, scheduler, epochs):
        history = {"loss": [],
                   "val_loss": [],
                   "acc": [],
                   "val_acc": [],
                   "edit_distance": [],
                   "val_edit_distance": [],
                   "best_weights" : 0
                   }
        #add in early stopping criteria and svaing best weights at each epoch

        for e in range(epochs):  # loop over the dataset multiple times
            print("Start of epoch:", e)
            self.model.train(True)
            running_loss = 0.0
            correct = 0.0
            edit_distance = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels, _ = data
                inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(inputs, inputs.shape[0], labels.shape[0])
                if self.binary:
                    labels = torch.from_numpy((np.array([[x] * output.shape[-1] for x in labels])))
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                # get statistics
                running_loss += loss.item()
                output = torch.argmax(output, dim=1)
                correct += (output == labels).float().sum()
                for j in range(len(labels)):
                    edit_distance += syllable_edit_distance(output[j], labels[j])

                # print update Improve this to make it better Maybe a global counter
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_loss ))
            history["loss"].append(running_loss)
            history["acc"].append(100 * correct / (len(train_loader.dataset) * self.window_size))
            history["edit_distance"].append(edit_distance / (len(train_loader.dataset) * self.window_size))
            if val_loader != None:
                self.validation_step(val_loader, history)
        print('Finished Training')
        return history

    def validation_step(self, val_loader, history):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0.0
            val_edit_distance = 0.0
            for i, data in enumerate(val_loader):
                inputs, labels, _ = data  # , input_lengths, label_lengths = data
                inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                output = self.model(inputs, inputs.shape[0], labels.shape[0])
                if self.binary:
                    labels = torch.from_numpy((np.array([[x] * output.shape[-1] for x in labels])))
                loss = self.criterion(output, labels)
                # get statistics
                val_loss += loss.item()
                output = torch.argmax(output, dim=1)

                val_correct += (output == labels).float().sum()
                for j in range(len(labels)):
                    val_edit_distance += syllable_edit_distance(output[j], labels[j])
                # print(val_edit_distance)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(100 * val_correct / (len(val_loader.dataset) * self.window_size))
            history["val_edit_distance"].append(val_edit_distance / (len(val_loader.dataset) * self.window_size))
            if history["val_acc"][-1] > history["best_weights"]:
                torch.save(self.model.state_dict(), "best_model_weights.h5")
                history["best_weights"] = history["val_acc"][-1]



    def testing_step(self, test_loader):
        predictions = pd.DataFrame()
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels, uids = data  # , input_lengths, label_lengths = data
                inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                output = self.model(inputs, inputs.shape[0], labels.shape[0])
                #output = torch.argmax(output, dim=1)
                #print(output.shape, labels.shape)
                # may need to convert the output accordingly rather than doing this.
                temp_uids = []
                if self.binary:
                    labels = torch.from_numpy((np.array([[x] * output.shape[-1] for x in labels])))
                    temp_uids = np.array([[x] * output.shape[-1] for x in uids])
                # get statistics
                else:
                    for u in uids:
                        #for j in range(self.window_size):
                        #    temp_uids.append(u + "_" + str(j))
                        for j in range(output.shape[-1]):
                             temp_uids.append(str(j) + "_" + u)
                    temp_uids = np.array(temp_uids)
                #print(temp_uids.flatten().shape, output.flatten().shape, labels.flatten().shape)
                #print(len(uids), output.shape, labels.shape)
                #I may not liek this format
                # not great approach
                #uids = 64x1 -> 64 x 216 x 1
                #pred = 64 x 216 x 2 -> 64 x 216 x 1, 64 x 216 x 1
                #label = 64 x 216 x 1
                zero_pred = output[:, 0, :]
                one_pred = output[:, 1, :]
                pred = torch.argmax(output, dim=1)
                #print(temp_uids.shape, zero_pred.shape, one_pred.shape, labels.shape)
                d = {"uid": temp_uids.flatten(), "zero_pred": zero_pred.flatten(), "one_pred": one_pred.flatten(), "pred": pred.flatten(), "label": labels.flatten()}
                new_preds = pd.DataFrame(d)
                predictions = predictions.append(new_preds)
        print('Finished Testing')
        return predictions

    def test_load_step(self, test_dataset, batch_size=64, model_weights=None):
        self.model.load_state_dict(torch.load(model_weights))
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_out = self.testing_step(test_data_loader)
        return test_out

    def train_pipeline(self, train_dataset, val_dataset=None, test_dataset=None, lr=.005, batch_size=64,
                       epochs=100, save_me=True, fine_tuning=False, finetune_path=None):
        if fine_tuning:
            self.model.load_weights(finetune_path)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data_loader = None
        if val_dataset != None:
            val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                        max_lr=lr,
                                                        steps_per_epoch=int(len(train_data_loader)),
                                                        epochs=epochs,
                                                        anneal_strategy='linear')
        start_time = datetime.now()
        history = self.training_step(train_data_loader, val_data_loader, scheduler, epochs)
        end_time = datetime.now()
        self.runtime = end_time - start_time
        test_out = []
        if test_dataset != None:
            test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            test_out = self.testing_step(test_data_loader)

        if save_me:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(self.model.state_dict(), f"model_weights-{date_str}.h5")
        self.print_results(history)
        return history, test_out, start_time, end_time



