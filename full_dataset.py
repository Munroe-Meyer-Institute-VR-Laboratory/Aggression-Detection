import copy
import glob
import json
import math
import os
import pathlib
import random
import traceback
import warnings
from json import JSONDecodeError

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tensorflow import keras
from keras import layers, optimizers, losses, applications
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models.video import MC3_18_Weights, mc3_18
from torchvision.models.video import MViT_V1_B_Weights, mvit_v1_b
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
from torchvision.models.video import R3D_18_Weights, r3d_18
from torchvision.models.video import S3D_Weights, s3d
from torchvision.models.video import Swin3D_B_Weights, swin3d_b
from torchvision.models.video import Swin3D_S_Weights, swin3d_s
from torchvision.models.video import Swin3D_T_Weights, swin3d_t
from torchvision.models.video import VideoResNet, MViT, S3D, SwinTransformer3d
from torchvision.transforms.functional import crop
from torchvision.transforms import Resize

from experiment_setup import FeatureExtractorSpec
from cnn_transformer import PositionalEmbedding, TransformerEncoder

warnings.filterwarnings("error")
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def PositionalEmbeddingTransformer(seq_len, num_features, num_classes, image_size, spec):
    sequence_length = seq_len
    embed_dim = num_features
    dense_dim = 64
    num_heads = 16

    if spec == FeatureExtractorSpec.DENSENET121:
        feature_extractor = applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
    elif spec == FeatureExtractorSpec.DENSENET169:
        feature_extractor = applications.DenseNet169(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
    elif spec == FeatureExtractorSpec.DENSENET201:
        feature_extractor = applications.DenseNet201(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_size, image_size, 3),
        )
    else:
        raise ValueError("Invalid DenseNet spec")
    preprocess_input = keras.applications.densenet.preprocess_input

    inputs = keras.Input((image_size, image_size, 3))
    x = preprocess_input(inputs)
    x = feature_extractor(x)
    # inputs = keras.Input(shape=(None, None))
    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(x)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy(),
        metrics=["accuracy"])
    return model


def get_classification_table(y_true, y_score):
    classes = list(set(y_true))
    class_table = [[0] * (len(classes) + 1) for _ in range(len(classes))]
    for y, y_s in zip(y_true, y_score):
        class_table[int(y)][0] += 1
        class_table[int(y)][y_s + 1] += 1
    rows = [f"Class {i}" for i in classes]
    cols = [" "] + rows
    return pd.DataFrame(class_table, index=rows, columns=cols)


def calculate_auprc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


def calculate_tpr_10_per(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    tpr_at_1_fpr = tpr[np.abs(fpr - 0.1).argmin()]
    return tpr_at_1_fpr


def calculate_tpr_5_per(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    tpr_at_5_fpr = tpr[np.abs(fpr - 0.05).argmin()]
    return tpr_at_5_fpr


def calculate_eer(y_true, y_score):
    """
    Returns the equal error rate for a binary classifier output.
    https://github.com/scikit-learn/scikit-learn/issues/15247#issuecomment-542138349
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def get_keystroke_info(keystroke_json, event_history):
    freq_bindings = {}
    dur_bindings = {}
    key_freq = {}
    key_dur = {}
    for key in keystroke_json:
        if key == "Frequency":
            for bindings in keystroke_json[key]:
                freq_bindings[bindings[1]] = bindings[0]
                key_freq[bindings[1]] = 0
        if key == "Duration":
            for bindings in keystroke_json[key]:
                dur_bindings[bindings[1]] = bindings[0]
                key_dur[bindings[1]] = 0
    for session_info in event_history:
        session_param = session_info[0]
        try:
            if session_param in freq_bindings:
                key_freq[session_param] += 1
            elif session_param in dur_bindings:
                key_dur[session_param] += int(session_info[1][1]) - int(session_info[1][0])
        except Exception as e:
            print(f"\n\tERROR: Error encountered\n{str(e)}\n{traceback.print_exc()}")
            continue
    return key_freq, key_dur


def get_dataset_stats(dataset_files, name):
    dataset_df = pd.DataFrame()
    for file in dataset_files:
        with open(file, 'r') as f:
            try:
                json_file = json.load(f)
            except JSONDecodeError:
                print("\tFile was empty, continuing...")
                continue
        coder = pathlib.Path(file).parts[4]
        participant = pathlib.Path(file).parts[6]
        date = json_file['Session Date']
        length = json_file['Session Time']
        key_freq, key_dur = get_keystroke_info(json_file['KSF'], json_file['Event History'])
        subject_data = {"Coder": str(coder),
                        "Participant": str(participant),
                        "Date": date,
                        "Length": length}
        for c in key_freq.keys():
            subject_data.update({
                f"{c}": key_freq[c]
            })
        for c in key_dur.keys():
            subject_data.update({
                f"{c}": key_dur[c]
            })
        dataset_df = dataset_df.append(subject_data, ignore_index=True)
    mean_data = {"Coder": "Average"}
    for c in key_freq.keys():
        mean_data.update({
            f"{c}": dataset_df[c].sum()
        })
    for c in key_dur.keys():
        mean_data.update({
            f"{c}": dataset_df[c].sum()
        })
    dataset_df = dataset_df.append(mean_data, ignore_index=True)
    dataset_df.set_index("Coder", inplace=True)
    dataset_df.to_excel(os.path.join('dataset_stats', f"{name} Stats.xlsx"))


def get_video_dataset(dataset_files):
    video_files, history_files, ksf = [], [], []
    for file in dataset_files:
        video_files.append(file[:-4] + "mp4")
        with open(file, 'r') as f:
            try:
                json_file = json.load(f)
            except JSONDecodeError:
                print("\tFile was empty, continuing...")
                continue
        history_files.append(json_file['Event History'])
        ksf.append(json_file['KSF'])
    return video_files, history_files, ksf


def setup_finetune_model(model, feature_extracting, num_classes):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    if type(model) is VideoResNet:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    elif type(model) is MViT:
        model.head[-1] = nn.Linear(in_features=model.head[-1].in_features, out_features=num_classes)
    elif type(model) is S3D:
        model.classifier[-1] = nn.Conv3d(in_channels=model.classifier[-1].in_channels, out_channels=num_classes,
                                         kernel_size=model.classifier[-1].kernel_size,
                                         stride=model.classifier[-1].stride)
    elif type(model) is SwinTransformer3d:
        model.head = nn.Linear(in_features=model.head.in_features, out_features=num_classes)
    else:
        raise ValueError(f"Model selection is not supported! {type(model)}")


class TrainDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.labels[ind]
        return x, y


def run_experiment(dataset, test_dataset, classes, batch_size, epochs, output_dir):
    # Instantiate PyTorch fine tuning models
    models = [
        # (r3d_18(weights=R3D_18_Weights.KINETICS400_V1), R3D_18_Weights.KINETICS400_V1, 'r3d_18'),
        # (r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1), R2Plus1D_18_Weights.KINETICS400_V1,
        #  'r2plus1d_18'),
        # (mc3_18(weights=MC3_18_Weights.KINETICS400_V1), MC3_18_Weights.KINETICS400_V1, 'mc3_18'),
        # (mvit_v1_b(weights=MViT_V1_B_Weights.KINETICS400_V1), MViT_V1_B_Weights.KINETICS400_V1, 'mvit_v1_b'),
        # (mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1), MViT_V2_S_Weights.KINETICS400_V1, 'mvit_v2_s'),
        # (s3d(weights=S3D_Weights.KINETICS400_V1), S3D_Weights.KINETICS400_V1, 's3d'),
        # (swin3d_b(weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1),
        #  Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1, 'swin3d_b'),
        # (swin3d_s(weights=Swin3D_S_Weights.KINETICS400_V1), Swin3D_S_Weights.KINETICS400_V1, 'swin3d_s'),
        (swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1), Swin3D_T_Weights.KINETICS400_V1, 'swin3d_t')
    ]
    # Perform fine tuning using segmented dataset
    for model in models:
        setup_finetune_model(model[0], False, len(classes))
        model_dir = os.path.join(output_dir, model[2])
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
            finetune_pytorch_vision(model, dataset, batch_size, epochs, model_dir)


def evaluate_pytorch_vision():
    pass


def finetune_pytorch_vision(model_params, dataset, batch_size, epochs, output_dir):
    model = model_params[0]
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_set = TrainDataset(*dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    model.train()

    epoch_mse = []
    best_mse_loss = 1e10
    best_exp_model = None
    predictions = []
    for i in np.arange(epochs):
        train_mse_loss = 0
        for idx, (video_data, video_label) in enumerate(train_loader):
            video_data = video_data.to(device)
            video_label = video_label.to(device)
            optimizer.zero_grad()

            outputs = model(video_data)
            predictions.append(outputs.cpu().detach().numpy())
            loss = criterion(outputs, video_label)
            loss.backward()
            optimizer.step()
            train_mse_loss += loss.item()
            del loss
        train_mse_loss /= len(train_loader)
        epoch_mse.append(train_mse_loss)

        if best_mse_loss > train_mse_loss:
            best_mse_epoch = i
            best_mse_loss = train_mse_loss
            best_exp_model = copy.deepcopy(model.state_dict())
        print(f'\r\tEpoch: {i} | Epoch Loss: {train_mse_loss:.4f} | Best Loss: {best_mse_loss:.4f}', end='')
    model.load_state_dict(best_exp_model)
    return model


def construct_video_tensor(video_frames, cropped=True):
    frame_tensors = []
    resize = Resize(size=(500, 500))
    for anno in video_frames:
        if cropped:
            with open(anno) as f:
                annotations = f.readlines()
                annotation = None
                if len(annotations) > 0:
                    if len(annotations) == 1:
                        annotation = annotations[0].strip()
                    elif len([x for x in annotations if int(x.split(' ')[0]) == 0]) == 1:
                        annotation = [x for x in annotations if int(x.split(' ')[0]) == 0][0]
                    else:
                        continue
                if annotation:
                    if int(annotation.split(' ')[0]) == 0:
                        _, x, y, w, h = map(float, annotation.split(' '))
                        frame = read_image(anno[:-3] + 'png')
                        _, dh, dw = frame.shape
                        l = int((x - w / 2) * dw)
                        t = int((y - h / 2) * dh)
                        w = int(w * dw)
                        h = int(h * dh)
                        frame_tensors.append(resize(crop(frame, t, l, h, w)))
        else:
            frame_tensors.append(read_image(f[:-3] + 'png'))
    return torch.stack(frame_tensors)


def save_clips(vids, labs, parts, freq_classes, dur_classes, ksfs):
    for vid, lab, part, ksf in zip(vids, labs, parts, ksfs):
        if not os.path.exists(rf'E:\Thesis Results\Datasets\MMISB Frequency\{part}'):
            os.mkdir(rf'E:\Thesis Results\Datasets\MMISB Frequency\{part}')
        if not os.path.exists(rf'E:\Thesis Results\Datasets\MMISB Duration\{part}'):
            os.mkdir(rf'E:\Thesis Results\Datasets\MMISB Duration\{part}')
        with VideoFileClip(vid, audio=False, fps_source='fps') as full_video:
            for l in lab:
                clip_fps = full_video.fps
                if l[0] in freq_classes:
                    l[0] = l[0].replace('/', '-')
                    class_dir = rf'E:\Thesis Results\Datasets\MMISB Frequency\{part}\{l[0]}'
                    clip_start = float(l[2] - clip_fps) / float(clip_fps)
                    clip_end = float(l[2] + clip_fps) / float(clip_fps)
                elif l[0] in dur_classes:
                    l[0] = l[0].replace('/', '-')
                    class_dir = rf'E:\Thesis Results\Datasets\MMISB Duration\{part}\{l[0]}'
                    clip_end = float(l[2]) / float(clip_fps)
                    clip_start = float(l[2] - ((l[1][1] - l[1][0]) * clip_fps)) / float(clip_fps)
                    if clip_start == clip_end:
                        continue
                    clip_start = clip_start if clip_start >= 0.0 else 0.0
                else:
                    continue

                if not os.path.exists(class_dir):
                    os.mkdir(class_dir)
                print(f"Processing {l[0]} from {vid}")
                clip_name = f"{len(list(os.listdir(class_dir)))}.mp4"

                try:
                    full_video.subclip(t_start=clip_start, t_end=clip_end).write_videofile(
                        os.path.join(class_dir, clip_name), logger=None)
                except UserWarning:
                    print(f"{vid}: Can't export {clip_start} to {clip_end}")


def load_preliminary_dataset():
    files = [f for f in glob.glob(rf'C:\GitHub\Keypoint-LSTM\datasets\stills\**\*.txt', recursive=True) if
             'classes' not in f]
    videos = [[] for x in range(len(classes))]
    labels = [[] for x in range(len(classes))]
    participants = [[] for x in range(len(classes))]
    processed_files = []
    for f in files:
        if f not in processed_files:
            video_name = pathlib.Path(f).parts[-2]
            video = [x for x in files if pathlib.Path(x).parts[-2] == video_name in x]
            if len(video) != 16:
                print(video_name)
            label = pathlib.Path(f).parts[-3]
            participant = pathlib.Path(f).parts[-2].split('_')[0]
            if label in classes:
                processed_files.extend(video)
                videos[classes.index(label)].append(video)
                labels[classes.index(label)].append(label)
                participants[classes.index(label)].append(participant)
        else:
            continue
    return videos, labels, participants


freq_classes = [
    'hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking',
    'sib-head bang', 'sib-head hit', 'sib-self-hit', 'sib-biting', 'sib-eye poke', 'sib-body slam',
    'sib-hair pull', 'sib-choking', 'sib-pinch_scratch', 'throwing object', 'kick_hit object', 'flip furniture',
    'flop', 'no pbx'
]
dur_classes = [
    'st- rock', 'st-hand flap', 'st-touch/tap', 'st-head swing', 'stereovox'
]

classes = [
    *freq_classes,
    # *dur_classes
]

if __name__ == '__main__':
    # videos, labels, participants = load_preliminary_dataset()
    videos = [f for f in glob.glob(rf'C:\GitHub\Keypoint-LSTM\datasets\stills\MMISB Cropped\*/*/*/')
              if 'p003_2021-12-07 10111_1_0_9569_9585_805' not in f
              and 'p003_2021-12-07 10111_1_26_9567_9583_804' not in f]

    classes, subjects = [], []
    for v in videos:
        subjects.append(pathlib.Path(v).parts[6])
        classes.append(pathlib.Path(v).parts[7])
    classes = set(classes)
    subjects = list(set(subjects))
    event_dict = [dict() for s in subjects]
    for s in subjects:
        for c in classes:
            v_c = len([f for f in videos if c in f and s in f])
            event_dict[subjects.index(s)].update({
                f"{c}": f"{v_c}"
            })
    pd.DataFrame(event_dict, index=subjects).to_excel(os.path.join('dataset_stats', f"Final Dataset Stats.xlsx"))
    # get_dataset_stats(videos, "Preliminary")
    # filepath = r""
    # files = [f for f in glob.glob(f'{filepath}/**/**/**/**/**/Primary/*.json')
    #          if 'training' not in f
    #          and '1aMLMar102022.json' not in f
    #          and os.path.isfile(f[:-4] + "mp4")]
    #
    # root_filepath = r""
    # get_dataset_stats(files, "MMISB")
    #
    # video_files, labels, ksf = get_video_dataset(files)
    # save_clips(video_files, labels, freq_classes, dur_classes, ksf)
    # root_filepath = r""
    # full_videos = [f for f in glob.glob(f'{root_filepath}/**/*.mp4')]
    # full_labels = [pathlib.Path(x).parts[-2] for x in full_videos]

    # files = [f for f in glob.glob(rf'datasets\stills\**\*.txt', recursive=True) if
    #          'classes' not in f]
    # videos = [[] for x in range(len(classes))]
    # labels = [[] for x in range(len(classes))]
    # participants = [[] for x in range(len(classes))]
    # processed_files = []
    # for f in files:
    #     if f not in processed_files:
    #         video_name = pathlib.Path(f).parts[-2]
    #         video = [x for x in files if video_name in x]
    #         label = pathlib.Path(f).parts[-3]
    #         participant = pathlib.Path(f).parts[-2].split('_')[0]
    #         if label in classes:
    #             processed_files.extend(video)
    #             videos[classes.index(label)].append(video)
    #             labels[classes.index(label)].append(label)
    #             participants[classes.index(label)].append(participant)
    #     else:
    #         continue
    # videos, labels, participants = load_preliminary_dataset()
    #
    # train_videos, train_labels = [], []
    # test_videos, test_labels = [], []
    #
    # for v, l in zip(videos, labels):
    #     if v and l:
    #         if l[0] in classes:
    #             train_size = math.ceil(len(v) * 0.8)
    #             train_v = random.sample(v, train_size)
    #             train_l = [classes.index(l[0])] * train_size
    #             train_videos.extend(train_v)
    #             train_labels.extend(train_l)
    #             test_v = [x for x in v if x not in train_v]
    #             test_l = [classes.index(l[0])] * len(test_v)
    #             test_videos.extend(test_v)
    #             test_labels.extend(test_l)
    #
    # output_dir = 'results'
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    #
    # train_videos_tensors = []
    # train_labels_tensors = []
    # for v, l, p in zip(train_videos, train_labels, participants):
    #     if len(v) == 16:
    #         video_tensor = construct_video_tensor(v)
    #         if len(video_tensor) == 16:
    #             train_videos_tensors.append(video_tensor)
    #             train_labels_tensors.append(torch.tensor(l))
    # train_videos_tensors = torch.stack(train_videos_tensors)
    # train_videos_tensors = torch.swapaxes(train_videos_tensors, 1, 2)
    # train_videos_tensors = train_videos_tensors.float()
    #
    # test_videos_tensors = []
    # test_labels_tensors = []
    # for v, l, p in zip(test_videos, test_labels, participants):
    #     if len(v) == 16:
    #         video_tensor = construct_video_tensor(v)
    #         if len(video_tensor) == 16:
    #             test_videos_tensors.append(video_tensor)
    #             test_labels_tensors.append(torch.tensor(l))
    # test_videos_tensors = torch.stack(test_videos_tensors)
    # test_videos_tensors = torch.swapaxes(test_videos_tensors, 1, 2)
    # test_videos_tensors = test_videos_tensors.float()
    #
    # run_experiment([train_videos_tensors, train_labels_tensors],
    #                [test_videos_tensors, test_labels_tensors],
    #                classes, batch_size=1, epochs=80, output_dir=output_dir)
    # run_experiment([video_files, labels], classes, output_dir, files)
