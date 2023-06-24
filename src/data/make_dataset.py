import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.signal import resample


def LoadData(raw_path:str='C:\Desktop'):
    """
    Loads the mitbih data from the data folder within the repo

    Parameters
    ----------
    raw_path: Path to the raw MITBIH data

    Returns
    ----------
    X: Raw MITBIH data
    
    X_imp: Imputed ecg data (based on raw data)
    
    y: ECG class labels
    
    fs: sampling rate
    """
    trainData = pd.read_csv(os.path.join(raw_path, 'mitbih_train.csv'), header=None)
    
    testData = pd.read_csv(os.path.join(raw_path, 'mitbih_test.csv'), header=None)
    
    data = pd.concat([trainData, testData], ignore_index=True)

    X = data.iloc[:,:-2].to_numpy()
    y = data.iloc[:,-1:].to_numpy().astype(np.int16)

    fs = 125

    X_imp = _ImputeData(X)

    return (X, X_imp, y, fs)


def AugmentData(X: np.ndarray, y: np.ndarray, aug_detials: dict):
    """
    Augments data for a specific class

    Parameters
    ----------
    X: ecg array of size (n,m) that needs to be augmented
    
    y: ecg class labels of size (n,)
    
    aug_details: dictionary explaining classes to be augmented and amount of augmented signals to generate
    
    {class_label: n_augmented_signals}, if n_augmented_signals=-1 will double the sample population

    Returns
    ----------
    X_aug: Augmented data array
    y_aug: Label array for augmented data
    """

    for ii, class_label in enumerate(aug_detials.keys()):
        class_idx = np.where(y==class_label)[0]

        X_class = X[class_idx]

        if aug_detials[class_label] > X_class.shape[0]:
            print('Requesting too many augmented samples, at most can double current class sample population')
            return
        
        if aug_detials[class_label] == -1:
            window_samples = X_class[:]
        else:
            window_samples = X_class[np.random.randint(X_class.shape[0], size=aug_detials[class_label])]

        if window_samples.shape[0] == 0:
            continue

        class_aug = resample(window_samples, int(window_samples.shape[1]*.5), axis=-1)
        class_aug = resample(class_aug, int(class_aug.shape[1]*2), axis=-1)
        class_aug = np.array([(signal - np.min(signal)) / (np.max(signal) - np.min(signal))
                     for signal in class_aug])

        aug_labels = np.empty(class_aug.shape[0], dtype=np.int16)
        aug_labels[:] = class_label

        try:
            X_aug = np.vstack((X_aug, class_aug))
            y_aug = np.append(y_aug, aug_labels)
        except NameError:
            X_aug = class_aug[:]
            y_aug = aug_labels[:]

    return (X_aug, y_aug)


def _ImputeData(X: np.ndarray):
    """
    Imputation techniqe for data

    Parameters
    ----------
    X: ecg array of size nxm that needs to be imputed

    Returns
    ----------
    X_imp: Imputed array
    """

    X_imp = X.copy()

    # Locate main R peak
    R_peak_idx = np.argmax(X_imp[:,25:], axis=-1) + 25 #Remove the first 25 samples to avoid cropped peak

    for ii, window in enumerate(X_imp):
        pad_idx = np.where(window == 0)[0]
        pad_idx = pad_idx[pad_idx > R_peak_idx[ii]] 
        if pad_idx.shape[0] == 0:
            continue
        X_imp[ii, pad_idx] = np.mean(X_imp[ii, :pad_idx[0]]) + np.random.normal(scale = 0.005, size=pad_idx.shape[0])

    return X_imp

if __name__ == '__main__':
    LoadData()
