from setuptools import find_packages, setup


setup(
    name='src',
    packages=find_packages(),
    install_requires=['numpy',
                      'pandas',
                      'scikit-learn',
                      'torch',
                      'torchaudio',
                      'torchvision',
                      'matplotlib',
                      'neurokit2',
                      'optuna',
                      'scipy',
                      'mrmr_selection'],
    version='0.1.0',
    description='Take-home technical test for SickKids ML specialist application.',
    author='Andrew Hart',
    license='MIT',
)
