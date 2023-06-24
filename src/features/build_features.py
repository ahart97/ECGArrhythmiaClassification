import numpy as np
import neurokit2 as nk
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from pywt import wavedec
from mrmr import mrmr_classif

from src.utils.signal_utils import BandPassFilter


class FeatureExtractor:
    def __init__(self):
        """
        Feature extraction object for arrythmia detection in single beat ECG signals
        """
        pass

    def ExtractFeatures(self, X: np.ndarray, fs: float) -> pd.DataFrame:
        """
        Extracts all the necessary features from the raw data array

        Parameters
        ----------
        X: (n_samples, n_windowSize) array
        
        fs: sampling rate of the signals

        Returns
        -------
        features: pd.DataFrame describing the features extracted for each window, columns are feature names
        """

        self.X = BandPassFilter(X, fs, axis=-1)
        self.fs = fs

        #self._ProcessECG()

        features = pd.DataFrame()

        # Time Domain Features
        features['RMSMagnitude'] = self._ExtractRMSVoltage()
        features['Skewness'] = self._ExtractVoltageSkew()
        features['Kurtosis'] = self._ExtractVoltageKurtosis()
        features['RPeakOnset'], features['RPeakMagnitude'] = self._ExtractRPeak()

        # Frequency and Wavelet Domain Features
        features['MPF'] = self._ExtractMPF()
        self._WaveletTransform()
        coeff_labels = ['cA4', 'cD4', 'cD3', 'cD2', 'cD1']
        for ii, coeff in enumerate(coeff_labels):
            features['MaxCoeff_{}'.format(coeff)] = self._ExtractMaxCoeff(ii)
            features['MinCoeff_{}'.format(coeff)] = self._ExtractMinCoeff(ii)
            features['MeanCoeff_{}'.format(coeff)] = self._ExtractMeanCoeff(ii)
            features['SDCoeff_{}'.format(coeff)] = self._ExtractSDCoeff(ii)

        return features


    def GradeFeatures(self, features: pd.DataFrame, y: np.ndarray):
        """
        Parameters
        ----------
        features: feature dataframe for grading (nxm)

        y: (n,) labels for grading features

        Returns
        -------
        feature_order: list of feature names in order based on mrmr
        """

        feature_order = mrmr_classif(X=features, y=pd.Series(y), K=features.shape[1])

        return feature_order


    def _ExtractRMSVoltage(self):
        return np.sqrt(np.mean(self.X**2, axis=-1))
    
    def _ExtractVoltageSkew(self):
        return skew(self.X, axis=-1)
    
    def _ExtractVoltageKurtosis(self):
        return kurtosis(self.X, axis=-1)
    
    def _ExtractMPF(self):
        f, PSD = welch(self.X, self.fs, nperseg = self.X.shape[-1], axis=-1)
        return np.sum(f*PSD, axis=-1)/np.sum(PSD, axis=-1)
    
    def _ExtractRPeak(self):
        return ((np.argmax(self.X[:,25:], axis=-1) + 25)/self.fs, np.max(self.X[:,25:], axis=-1)) #Have to ignore the cropped peak at the beginning
    
    def _ExtractMaxCoeff(self, ii):
        return np.max(self.wavelet_coeffs[ii], axis=-1)
    
    def _ExtractMinCoeff(self, ii):
        return np.min(self.wavelet_coeffs[ii], axis=-1)
    
    def _ExtractMeanCoeff(self, ii):
        return np.mean(self.wavelet_coeffs[ii], axis=-1)
    
    def _ExtractSDCoeff(self, ii):
        return np.std(self.wavelet_coeffs[ii], axis=-1)
    
    def _WaveletTransform(self):
        self.wavelet_coeffs = wavedec(self.X, wavelet='db2', level=4, axis=-1) #cA4, cD4, cD3, cD2, cD1


if __name__ == '__main__':
    print('Running')

    test_signal = nk.ecg_simulate(duration=10, sampling_rate=125)
    test = FeatureExtractor()
    print(test.ExtractFeatures(np.array([test_signal, test_signal]), 125))