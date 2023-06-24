from neurokit2 import ecg_peaks
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import pickle
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import tensor, nn
from torch.utils.data import Dataset

from src.visualization.visualize import MyConfMatrix

def BandPassFilter(signal: np.ndarray, fs: float, cutoff_low: float = 0.67, \
        cutoff_high: float = 40, order: int = 5, axis:int = -1): 
    """
    Bandpasses the signal for the given cutoffs

    Parameters
    ----------
    signal: data to be filetered

    fs: the sampling rate

    cutoff_low: the low cutoff frequency

    cutoff_high: the high cutoff frequency

    order: the order of the bandpass filter

    Returns
    ----------
    filtered array
    """

    nyquist_freq = 0.5 * fs
    sos = butter(N=order, Wn=[cutoff_low/nyquist_freq, \
        cutoff_high/nyquist_freq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x=signal, axis=axis)


def PickleDump(obj, filepath):
    """
    Pickles and object at a given filepath
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def PickleLoad(filepath):
    """
    Loads an object from a pickle filepath 
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.expand_dims(X, axis=1)
        MyEncoder = OneHotEncoder(sparse_output=False, dtype=int)
        self.y = MyEncoder.fit_transform(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return tensor(X, dtype=torch.float32), tensor(y, dtype=torch.float32)
    
def ScoreModel(y_true: np.ndarray, y_pred: np.ndarray, 
               class_labels: list=['C1', 'C2', 'C3', 'C4', 'C5'],
               plot_cm: bool=True, save_plot: bool=True,
               save_dir:str='', title: str='CM'):
    """
    Scores the predicitions from the model

    Parameters
    ----------
    y_true: true label array

    y_pred: predicted label array

    class_labels: labels for each of the classes (optional)

    plot_cm: boolean to decide if confusion matrix is generated

    save_plot: boolean to decide if confusion matrix is saved (ignored if plot_cm=False)
    
    save_dir: Location for saving plot (ignored if plot_cm=False)
    
    title: plot title for saving (ignored if plot_cm=False)
    """
    scores = classification_report(y_true, y_pred, target_names=class_labels, zero_division=0, output_dict=True)
    if plot_cm:
        MyConfMatrix(y_true, y_pred, class_labels, save_plot, save_dir, title)

    return scores



