from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import optuna
import numpy as np
import os
from joblib import dump, load

from src.utils.signal_utils import PickleLoad, PickleDump, ScoreModel


class RFModel():
    def __init__(self, model_dir: str) -> None:
        """
        Random Forest Classifier

        This model builds on the RandomForestClassifier in sklearn by adding in hyperparameter tunning and custom scoring.

        Parameters
        ----------
        model_dir: directory of model and parameters pickle files 
        (these files can be download from the repo https://github.com/ahart97/SickKidsTest/tree/main/models)
        """
        self.model = RandomForestClassifier(n_jobs=-1)
        self.model_tuned = False
        self.model_fit = False
        self.model_dir = model_dir

    def TuneHyperparameters(self, X:np.ndarray, y:np.ndarray, overwrite:bool=False):
        """
        Tunes the hyperparameters of the RF classifier and sets them to the current model

        Parameters
        ----------
        X: (n,m) array for training

        y: (n,) array for training labels

        overwrite: boolean variable to overwrite current hyperparameter tuning
        """

        if os.path.exists(os.path.join(self.model_dir, 'RFParams.pkl')) and not overwrite:
            self.params = PickleLoad(os.path.join(self.model_dir, 'RFParams.pkl'))
            self.model.set_params(**self.params)

        else:
            def _objective(trial):

                max_depth = trial.suggest_int('max_depth', 2, 40)
                n_estimators = trial.suggest_int('n_estimators', 25, 200, step=25)
                criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

                rf_obj = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                                criterion=criterion, max_features=max_features,
                                                n_jobs=-1)
                
                kf = StratifiedKFold(n_splits=3, shuffle=True)

                score = []

                for ii, (train_idx, test_idx) in enumerate(kf.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    rf_obj.fit(X_train, y_train)

                    scores = ScoreModel(y_test,
                                        rf_obj.predict(X_test),
                                        plot_cm=False)
                    
                    score.append(scores['macro avg']['f1-score'])

                return np.mean(score)
            
            study = optuna.create_study(study_name='RF_tuner', direction='maximize', load_if_exists=True)
            study.optimize(_objective, n_trials=50, gc_after_trial=True)
            self.model.set_params(**study.best_params)
            PickleDump(study.best_params, os.path.join(self.model_dir, 'RFParams.pkl'))

        self.model_tuned = True

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Fits model to data

        Parameters
        ----------
        X: (n,m) array for training

        y: (n,) array for training labels
        """
        if not self.model_tuned and os.path.exists(os.path.join(self.model_dir, 'RFParams.pkl')):
            self.params = PickleLoad(os.path.join(self.model_dir, 'RFParams.pkl'))
            self.model.set_params(**self.model.set_params(**self.params))
        elif not self.model_tuned:
            print('Model not tuned, please call "TuneHyperparameters" first')
            return 
        
        self.model.fit(X, y)
        #Had to switch to joblib for RF model due to size constraints
        dump(self.model, os.path.join(self.model_dir, 'RFModel.joblib'), compress=3)
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
        if not self.model_fit and os.path.exists(os.path.join(self.model_dir, 'RFModel.pkl')):
            self.model = load(os.path.join(self.model_dir, 'RFModel.joblib'))
        elif not self.model_fit:
            print('No model fit, please call "fit" first')
            return 
        
        y_pred = self.model.predict(X)
        
        return y_pred
    
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

