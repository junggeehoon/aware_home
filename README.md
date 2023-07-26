# aware_home

## Overview
This project forms a part of [The Aware Home](https://gvu.gatech.edu/research/labs/aware-home-research-initiative) project, carried out during the [2023 IPAT Summer Internship Program](https://research.gatech.edu/research-interns-selected-summer-2023). Detailed documentation regarding this project can be found on this [Notion](https://geehoon.notion.site/Machine-Learning-Approaches-for-Indoor-Location-Fingerprinting-Using-Apple-Watch-RSSI-fc6cbe1d59e44aa1a66004446faf7bb2?pvs=4) page.
The system scans Bluetooth signals emitted by an Apple Watch to determine its location. Various prediction models, including Random Forest, Support Vector Machine, and K-Nearest Neighbors, were utilized.

## Installation
1. Clone this repository:

    `git clone https://github.com/junggeehoon/aware_home.git`
    

2. Navigate to `aware_home/` directory
3. Run the following command to install all requirements for this project:

    `pip install -r requirements.txt`

## Project Directory
1. **src**: Contains all python files needed for this project.
2. **figures**: Contains all plotted images.
3. **models**: Contains trained machine learning models.
4. **vectors**: Contains raw RSSI measurements.
5. **data**: Contains filtered RSSI measurements.
6. **result**: Contains RMSE results of the model.
7. **test**: Contains training data.
8. **train**: Contains testing data.
9. **diagram**: Contains diagram from project description.


## Setup

### Data collection
Measure and collect RSSI values by running `./src/bluetooth_scanner.py`. It will collect RSSI vectors and save it to csv file.

### Data preparation
1. **Data cleaning**: Apply 3-sigma cutoff and replace RSSI less than -100 dBm with -100 by running `./src/data_preprocessing.py`
2. **Sampling**: Randomly select 1000 datapoints for each reference point by running `./src/select_random_data.py`
3. **Train Test Split**: Split data with training and testing by running `./src/train_test_split.py`


### Training
1. **Random Forest**: Fits random forest classifier by running `./src/models/rf.py`
2. **K-Nearest Neighbors**: Fits K-Nearest Neighbors classifier by running `./src/models/knn.py`
3. **Support Vector Machine**: sFits Support Vector Machine classifier by running `./src/models/svm.py`

### Usage
For realtime prediction, run `./src/realtime_prediction.py`. It will use previously generated files for prediction.

### Evaluation
Run `./src/evaluate_offline.py` to calculate Root Mean Squared Error (RSME).