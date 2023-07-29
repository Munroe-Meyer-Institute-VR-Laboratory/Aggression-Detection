import glob
import os

import pandas as pd
import pathlib


models = [
    "Transformer",
    "ViViT",
    "Bidirectional GRU",
    "Custom"
]

fes = [
    "DenseNet",
    "Xception",
    "Inception ResNet V2",
    "NASNET Large",
    "Efficient Net V2L",
    "CONVNEXTXLARGE",
    "RESNET152V2",
    "VGG16",
    "128x128x1"
]
weights_pattern = r'*.hdf5'
root_filepath = r"E:\Thesis Results\Keypoint-LSTM\MLHC\All"
data_files = [f for f in glob.glob(f'{root_filepath}/**_None/**/results.csv')]

table = pd.DataFrame()

for file in data_files:
    parts = pathlib.Path(file).parts
    approach = parts[4].split('_')[0]
    model = int(parts[4].split('_')[1])
    fe = int(parts[4].split('_')[2])
    subject = parts[4].split('_')[5]
    result = pd.read_csv(file)
    weights_dir = pathlib.Path(os.path.join(*pathlib.Path(file).parts[:-2], 'checkpoints'))
    convergence_epoch = str(max(weights_dir.glob(weights_pattern), key=lambda f: f.stat().st_ctime)).split('.')[1].split('-')[0]
    result['Approach'] = approach if approach == "Tracklet" else "Full Frame"
    result['Model'] = models[model]
    result['Feature Extractor'] = fes[fe]
    result['Subject'] = subject if subject != 'None' else '80/20 All'
    result['Convergence Epoch'] = convergence_epoch
    table = pd.concat([table, result], axis=0)

table.to_excel('mlhc_binary_results.xlsx')

root_filepath = r"C:\GitHub\Keypoint-LSTM\experiment_results"
data_files = [f for f in glob.glob(f'{root_filepath}/**_mc_**/**/results.csv')]

table = pd.DataFrame()

for file in data_files:
    parts = pathlib.Path(file).parts
    approach = parts[4].split('_')[0]
    model = int(parts[4].split('_')[1])
    fe = int(parts[4].split('_')[2])
    subject = parts[4].split('_')[5]
    result = pd.read_csv(file)
    weights_dir = pathlib.Path(os.path.join(*pathlib.Path(file).parts[:-2], 'checkpoints'))
    convergence_epoch = str(max(weights_dir.glob(weights_pattern), key=lambda f: f.stat().st_ctime)).split('.')[1].split('-')[0]
    result['Approach'] = approach if approach == "Tracklet" else "Full Frame"
    result['Model'] = models[model]
    result['Feature Extractor'] = fes[fe]
    result['Subject'] = subject if subject != 'None' else '80/20 All'
    result['Convergence Epoch'] = convergence_epoch
    table = pd.concat([table, result], axis=0)

table.to_excel('8mc_results.xlsx')
