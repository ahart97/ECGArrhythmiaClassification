import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def ClassDistribution(y: np.ndarray, save_plot:bool=True,
                      save_dir:str='',
                      title:str='ClassDistribution'):
    """
    Function for plotting the distribution of the labels

    Parameters
    ----------
    y: (n,) array of class labels

    save_plot: boolean to save plot

    save_dir: Location for saving plot
    
    title: plot title for saving (optional)
    """
    # Define class labels
    class_labels = ['N', 'S', 'V', 'F', 'Q']

    # Count the occurrences of each class
    if y.ndim > 1:
        y = y.flatten()
    class_counts = np.bincount(y)

    # Generate x-axis values for the classes
    classes = np.arange(len(class_labels))

    # Plot the distribution
    plt.bar(classes, class_counts, align='center', alpha=0.5)

    # Set x-axis tick labels
    plt.xticks(classes, class_labels)

    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Count')
    # plt.title('Distribution of Classes')

    # Add percentages above each bar
    total = sum(class_counts)
    for i, count in enumerate(class_counts):
        percentage = count / total * 100
        plt.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')

    # Save Figure
    if save_plot:
        plt.savefig(os.path.join(save_dir, '{}.png'.format(title)))

    # Show the plot
    plt.show()


def FeatureDistributions(y: np.ndarray, X: pd.DataFrame, 
                         save_plot:bool=True,
                         save_dir:str=''):
    """
    Function for plotting the distribution of the features

    Parameters
    ----------
    y: (n,) array of class labels

    X: pd.DataFrame for feature array

    save_dir: Location for saving plot

    save_plot: boolean to save plot
    """

    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    num_features = X.shape[1]
    
    # Define class labels
    class_labels = ['N', 'S', 'V', 'F', 'Q']

    # Define colors for each class
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    # Set up subplots
    fig, axs = plt.subplots(np.ceil(num_features/2).astype(int), 2, figsize=(8, 2*num_features))
    
    for ii, feature in enumerate(X.columns):

        feature_array = X[feature].to_numpy()
        # feature_array = (feature_array - np.min(feature_array))/(np.max(feature_array)-np.min(feature_array))

        ax = axs[np.floor(ii/2).astype(int), -1*((ii % 2) + 1)] if num_features > 2 else axs[ii]

        # Set plot title for each subplot
        ax.set_title(f'{feature}')

        for jj, cls in enumerate(unique_classes):
            class_data = feature_array[y == cls]
            mu, sigma = np.mean(class_data), np.std(class_data)
            
            # Generate x-axis values for the Gaussian distribution
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            
            # Generate y-axis values for the Gaussian distribution
            cls_y = norm.pdf(x, mu, sigma)
            
            # Plot the Gaussian line distribution
            ax.plot(x, cls_y, color=colors(jj), linewidth=2, label = class_labels[jj], alpha=0.75)
        
        if ii==1:
            ax.legend(loc='upper center', bbox_to_anchor=(1.1, 1.25),
                      ncol=5, fancybox=True)
    
    # Adjust vertical spacing between subplots
    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    # Save Figure
    if save_plot:
        plt.savefig(os.path.join(save_dir, 'FeatureDistribution.png'))
    
    # Show the plot
    plt.show()

def SignalSamplePlot(X: np.ndarray, fs: float, save_plot:bool=True,
                     save_dir:str='', title: str='SignalSample'):
    """
    Function for plotting the distribution of the features

    Parameters
    ----------
    X: (n,) array of raw ECG data

    fs: sampling rate

    save_dir: Location for saving plot

    save_plot: boolean to save plot

    title: plot title for saving (optional)
    """
    fig, ax = plt.subplots(1)
    t = np.linspace(0, X.shape[0]/fs, X.shape[0])
    ax.plot(t, X, alpha=0.75)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [mV]')
    ax.set_title('ECG Sample')

    # Save Figure
    if save_plot:
        project_dir = Path(__file__).resolve().parents[2]
        plt.savefig(os.path.join(project_dir, 'reports', 'figures', '{}.png'.format(title)))

    plt.show()


def MyConfMatrix(y_true: np.ndarray, y_pred: np.ndarray,
                 class_labels: list, save_plot:bool=True,
                 save_dir:str='', title: str='CM'):
    """
    Plots a confusion matrix based on the predictions

    Parameters
    ----------
    y_true: true labels ((n,) array)

    y_true: predicted labels ((n,) array)

    save_dir: Location for saving plot

    save_plot: boolean to save plot

    class_labels: labels to display on CM

    title: plot title for saving (optional)
    """
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    disp.plot(values_format='.2f', cmap='BuGn')

    if save_plot:
        plt.savefig(os.path.join(save_dir, '{}.png'.format(title)))

