import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset



def create_directory(directory_path):
    print('Directory doesnt exist')
    if not os.path.isdir(directory_path):
        print('Directory doesnt exist')
        os.makedirs(directory_path)


def load_data(file_name):

    # folder_path = "/home/jabdullayev/Codes/UCRArchive_2018/"
    folder_path = "/home/jabdullayev/phd/datasets/UCRArchive_2018/"
    folder_path += file_name + "/"

    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"

    if os.path.exists(test_path) <= 0:
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest


def znormalisation(x):

    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


def encode_labels(y):

    labenc = LabelEncoder()

    return labenc.fit_transform(y)


def preprocess_data(data, target, mini_batch_size=64, shuffle=True,):

    data = znormalisation(data)
    data = np.expand_dims(data, axis=1)

    target = encode_labels(target)

    data, target = torch.from_numpy(data), torch.from_numpy(target)

    torch.manual_seed(42)    
    dataloader = DataLoader(
        TensorDataset(data, target),
        batch_size=mini_batch_size,
        shuffle=shuffle,
    )

    return dataloader

def plot_loss_and_acc_curves(training_losses, val_losses, training_accuracies, val_accuracies, out_dir):
    plt.plot(training_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.savefig(out_dir + 'losses.png')
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(training_accuracies, label='train_acc')
    plt.plot(val_accuracies, label='val_acc')
    plt.savefig(out_dir + 'accuracies.png')
    plt.clf()
    plt.cla()
    plt.close()


# Plotly Codes
def update_fig_layout(fig):

    plot_bg_color='rgba(255, 255,255, 0.8)'

    fig.update_layout(xaxis_title="",)
    fig.update_layout(xaxis_title="Dim1",  yaxis_title="Dim2",plot_bgcolor=plot_bg_color,)
    fig.update_yaxes( showline=True, linewidth=2, linecolor='white', mirror=True, showgrid=True, gridwidth=0.1, gridcolor='white', tickprefix='', ticksuffix=' ') 

    fig.update_xaxes(visible=True, showticklabels=True)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgb(230, 230, 230)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgb(230, 230, 230)')

    fig.update_layout(autosize=False, width=1000, height=500)

    return fig