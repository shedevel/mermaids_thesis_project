Thesis Project: Cross-Dataset Music Emotion Recognition
by Sabina Hult & Line Bay Kreiberg

---------------------------------------

This folder includes the main codebase for the conducted experiments presented in our thesis. Code is written in Python 3.7 using the PyCharm IDE.

The project is not meant for execution via command line or without altering the code files (such as commenting/out-commenting lines), but serves as transparency regarding our conducted experiments.

---------------------------------------

The data folder contains different versions of each dataset, i.e. csv-files of extracted 
features and emotion annotations, both separately and combined. Audio files are excluded.
The src folder contains the main code files listed beneath

FeatureExtraction.py
- contains the applied code for feature extration from audio files applying 
the LibROSA library

DataHelper.py
- contains the applied code for reading in and manipulating datasets in order 
to create different versions of each dataset

Experiments.py
- contains the applied code for conducting experiments, both preliminary, baseline and
cross-dataset experiments

ParameterSearch.py
- not part of submitted code, but a work in progress to find optimal parameters for some selected algorithms to further investigation

---------------------------------------