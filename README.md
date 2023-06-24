ECG Arrhythmia Classification
==============================

Details
------------

Models for classification of arrhythmias in ECG signals.

Data used in this repo can be found here: [MIT-BIH Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?resource=download&select=mitbih_train.csv). 

Refer to the orgnaization below for where to save this data. Can load the Kaggle dataset with the code below.

```Python
from src.data.make_dataset import LoadData

X_raw, X, y, fs = LoadData(raw_path='Path\to\data\dir')
```

Environment Setup
------------

1. Install [Python 3.9](https://www.python.org/downloads/release/python-3913/)
2. Clone the repository to a local lcoation (Note: this requires git to be installed, an alternative is to download the repo as a zip and unzip the folder in the desired location)

    ```
    $ git clone https://github.com/ahart97/SickKidsTest.git$
    ```
    
    1. (Optional) Create a [virtual environment](https://docs.python.org/3/library/venv.html)

        ```
        $ "Path\to\Python39\python.exe" -m venv "Path\to\venv\location"
        ```

    2. (Optional) The [virtual environment](https://docs.python.org/3/library/venv.html) can be activated with the following

        ```
        $ "Path\to\venv\location\Scripts\activate"
        ```

3. Install the repo with pip install

    ```
    $ pip install -e "Path\to\SickKidsTest"
    ```

4. Depending on the use of the package, may be required to install all requirements via:

    ```
    $ pip install -r "Path\to\SickKidsTest\requirements.txt"
    ```

Example
------------

Important Note: All predefined model parameters and weights can be found in .\models\

Random forest classifier

```Python
from src.features.build_features import FeatureExtractor
from src.models.RF_model import RFModel

'''
Load in your data of n samples
X = raw data array (n,186)
y = labels (n,)
fs = sampling rate
'''

# Prepare features
featureExtractor = FeatureExtractor()
features = featureExtractor.ExtractFeatures(X, fs)

# Define model
rfModel = RFModel(model_dir='Path\to\model\dir')

# Tune model hyperparameters (optional - can used already tuned parameters RFParams.pkl)
rfModel.TuneHyperparameters(features.values, y)

# Train model (optional - can used already trained model RFModel.pkl)
rfModel.fit(features.values, y)

# Predict with model
y_pred = rfModel.fit(features.values)

# Evaluate model
scores = rfModel.score(y, X, save_dir='Path\to\save\dir', title='TitleOfCMPlot')

```

CNN classifier

```Python
from src.models.RF_model import CNNModel

'''
Load in your data of n samples
X = raw data array (n,186)
y = labels (n,1)
fs = sampling rate
'''

# Define model
cnnModel = CNNModel(model_dir='Path\to\model\dir')

# Tune model hyperparameters (optional - can used already tuned parameters CNNParmas.pkl)
cnnModel.TuneHyperparameters(X, y)

# Train model (optional - can used already trained model CNNModel.pkl)
cnnModel.fit(X, y)

# Predict with model
y_pred = cnnModel.fit(X)

# Evaluate model
scores = cnnModel.score(y.flatten(), X, save_dir='Path\to\save\dir', title='TitleOfCMPlot')

```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`.
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The processed data (augmentation, imputation, feature extraction).
    │   └── raw            <- The original, immutable data dump from Kaggle.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks for experiments.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── documents      <- Documents for reporting the design and evalution procedure.
    │   ├── CSVs           <- CSVs for reporting.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported.
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   ├── data           <- Scripts to download or generate data.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
