from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna
import numpy as np
import os
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader
import torch.optim as optim

from src.utils.signal_utils import PickleLoad, PickleDump, CustomDataset, ScoreModel

class EarlyStopping:
    def __init__(self, tolerance:int=5, min_delta:float=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss: float, validation_loss: float):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        else:
            self.counter = 0

class CNNArch(nn.Module):
    def __init__(self, num_classes=5, dropout=0.0):
        super(CNNArch, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(1472, 16)
        self.relu3 = nn.LeakyReLU()
        
        self.fc2 = nn.Linear(16, num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.dropout(self.pool1(self.relu1(self.conv1(x))))
        x = self.dropout(self.pool2(self.relu2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) 
        
        return x

class CNNModel():
    def __init__(self, model_dir: str) -> None:
        """
        CNN Classifier

        Parameters
        ----------
        model_dir: directory of model and parameters pickle files 
        (these files can be download from the repo https://github.com/ahart97/SickKidsTest/tree/main/models)
        """
        self.model = CNNArch()
        self.model_tuned = False
        self.model_fit = False
        self.model_dir = model_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def TuneHyperparameters(self, X:np.ndarray, y:np.ndarray, overwrite:bool=False):
        """
        Tunes the hyperparameters of the RF classifier and sets them to the current model

        Parameters
        ----------
        X: (n,m) array for training

        y: (n,1) array for training labels

        overwrite: boolean variable to overwrite current hyperparameter tuning
        """

        if os.path.exists(os.path.join(self.model_dir, 'CNNParmas.pkl')) and not overwrite:
            self.params = PickleLoad(os.path.join(self.model_dir, 'CNNParmas.pkl'))

        else:
            def _objective(trial):

                dropout = trial.suggest_float('dropout', 0.0, 0.8, step=0.1)
                optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'SGD'])
                lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
                momentum = trial.suggest_float('momentum', 0.0, 0.3)
                num_epochs = trial.suggest_int('num_epochs', 50, 120)
                batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

                kf = StratifiedKFold(n_splits=3, shuffle=True)

                score = []

                for ii, (train_idx, test_idx) in enumerate(kf.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    dataset = CustomDataset(X_train, y_train)
                    loader = DataLoader(dataset, shuffle=True, batch_size = batch_size, num_workers=0, drop_last=False)
                    dataset_test = CustomDataset(X_test, y_test)
                    loader_test = DataLoader(dataset_test, shuffle=True, batch_size = X_test.shape[0], num_workers=0, drop_last=False)

                    model = CNNArch(dropout=dropout)
                    model.to(device=self.device)

                    loss_fn = nn.CrossEntropyLoss().to(self.device)

                    if optimizer_name == 'SGD':
                        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, momentum=momentum)
                    else:
                        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

                    for epoch in range(0, num_epochs, 1): 
                        for jj, data in enumerate(loader):
                            inputs, targets = data
                            inputs, targets = inputs.to(self.device), targets.to(self.device)

                            #Resets the optimizer to zero grad
                            optimizer.zero_grad()

                            output = model(inputs)
                            loss = loss_fn(output, targets)

                            #Back propagate based on the loss
                            loss.backward()

                            #Update coefficients based on the back prop
                            optimizer.step()

                    with torch.no_grad():
                        for jj, data in enumerate(loader_test):
                            inputs, targets = data

                            output = model(inputs)

                            scores = ScoreModel(np.argmax(targets.detach().cpu().numpy(), axis=-1),
                                                np.argmax(output.detach().cpu().numpy(), axis=-1),
                                                plot_cm=False)
                            
                            score.append(scores['macro avg']['f1-score'])

                return np.mean(score)
            
            study = optuna.create_study(study_name='CNN_tuner', direction='maximize', load_if_exists=True)
            study.optimize(_objective, n_trials=50, gc_after_trial=True)

            PickleDump(study.best_params, os.path.join(self.model_dir, 'CNNParmas.pkl'))

        self.model_tuned = True

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Fits model to data

        Parameters
        ----------
        X: (n,m) array for training

        y: (n,1) array for training labels
        """
        if not self.model_tuned and os.path.exists(os.path.join(self.model_dir, 'CNNParmas.pkl')):
            self.params = PickleLoad(os.path.join(self.model_dir, 'CNNParmas.pkl'))
        elif not self.model_tuned:
            print('Model not tuned, please call "TuneHyperparameters" first')
            return 
        
        early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)
        
        X_train, X_valdiation, y_train, y_validation = train_test_split(X, y, test_size=0.18, stratify=y)

        self.model = CNNArch(dropout=self.params['dropout'])
        self.model.to(device=self.device)

        num_epochs = self.params['num_epochs']
        loss_fn = nn.CrossEntropyLoss().to(self.device)

        if self.params['optimizer_name'] == 'SGD':
            optimizer = getattr(optim, self.params['optimizer_name'])(self.model.parameters(), lr=self.params['lr'], 
                                                                 momentum=self.params['momentum'])
        else:
            optimizer = getattr(optim, self.params['optimizer_name'])(self.model.parameters(), lr=self.params['lr'])

        dataset = CustomDataset(X_train, y_train)
        loader = DataLoader(dataset, shuffle=True, batch_size = self.params['batch_size'], num_workers=0, drop_last=False)
        dataset_val = CustomDataset(X_valdiation, y_validation)
        loader_val = DataLoader(dataset_val, shuffle=True, batch_size = X_valdiation.shape[0], num_workers=0, drop_last=False)

        self.train_losses = []
        self.val_losses = []

        for epoch in range(0, num_epochs, 1):
            train_epoch = 0
            for jj, data in enumerate(loader):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                #Resets the optimizer to zero grad
                optimizer.zero_grad()

                output = self.model(inputs)
                loss = loss_fn(output, targets)

                train_epoch += loss.detach().cpu().numpy()

                #Back propagate based on the loss
                loss.backward()

                #Update coefficients based on the back prop
                optimizer.step()
            
            self.train_losses.append(train_epoch/jj)

            #Compute the validation losses
            with torch.no_grad():
                val_epoch = 0
                for jj, data in enumerate(loader_val):
                    inputs, targets = data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    output = self.model(inputs)
                    loss = loss_fn(output, targets)

                    val_epoch += loss.detach().cpu().numpy()

                self.val_losses.append(val_epoch)

            early_stopping(train_epoch/jj, val_epoch)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
        
        PickleDump(self.model, os.path.join(self.model_dir, 'CNNModel.pkl'))
        self.model_fit = True

    def predict(self, X:np.ndarray):
        """
        Use fitted model to predict labels

        Parameters
        ----------
        X: (n,m) array for prediction

        Returns
        ----------
        y_pred: prected labels
        """
        if not self.model_fit and os.path.exists(os.path.join(self.model_dir, 'CNNModel.pkl')):
            self.model = PickleLoad(os.path.join(self.model_dir, 'CNNModel.pkl'))
        elif not self.model_fit:
            print('No model fit, please call "fit" first')
            return 
        
        X = np.expand_dims(X, axis=1)
        
        X = tensor(X, dtype=torch.float32).to(self.device)
        self.model.to(self.device)
        
        y_pred = self.model(X)
        
        return np.argmax(y_pred.detach().cpu().numpy(), axis=-1)
    
    def score(self, y_true:np.ndarray, X:np.ndarray, 
              save_dir:str = '', title:str = 'CM'):
        """
        Score the fitted model

        Parameters
        ----------
        X: (n,m) array for testing

        y_true: (n,) array for testing labels

        save_dir: Location for saving plot 
        
        title: plot title for saving

        Returns
        ----------
        scores: scores pd.DataFrame, score name is represented in column names
        """
        y_pred = self.predict(X)
        class_labels = ['N', 'S', 'V', 'F', 'Q']

        return ScoreModel(y_true, y_pred, class_labels=class_labels,
                          save_dir=save_dir, title=title)


