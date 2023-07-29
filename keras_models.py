import csv
import datetime
import json
import pathlib
import pickle
import time
from os import path

from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold
import cv2
import imageio
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow import keras

from cnn_transformer import FeatureExtractorFeatures, build_feature_extractor
from dataloader_utils import create_pandas_dataset, create_tracklet_pandas_dataset
from experiment_setup import run_experiment, full_experiment_schedule
from tag_lut import all_classes

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
    "128x128x1",
    "HOG"
]


def convert_json_csv(json_files, existing_files, output_dir):
    # Get the current time to put into export files
    date = datetime.datetime.today().strftime("%B %d, %Y")
    time = datetime.datetime.now().strftime("%H:%M:%S")
    # Iterate through files
    for file in json_files:
        # If converted file exists already, skip it
        name = pathlib.Path(file).stem
        if name in existing_files:
            continue
        # Load session and split it up
        with open(file, 'r') as f:
            session = json.load(f)
        session_data = {k: v for k, v in session.items() if k in list(session.keys())[:15]}
        event_history = session["Event History"]
        # TODO: Export E4 data to CSV... somehow
        e4_data = session["E4 Data"]
        # Open output file and write session to it
        with open(path.join(output_dir, f"{name}.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, 'Generated on', date, time])
            # Write out the session fields
            writer.writerow(['Field', 'Value'])
            for key in session_data:
                writer.writerow([key, session[key]])
            # Write out the event history
            writer.writerow(['Tag', 'Onset', 'Offset', 'Frame', 'E4 Window'])
            for event in event_history:
                row = [event[0]]
                if type(event[1]) is list:
                    row.append(event[1][0])
                    row.append(event[1][1])
                else:
                    row.append(event[1])
                    row.append('')
                row.append(event[2])
                row.append(event[3])
                writer.writerow(row)


# Following method is modified from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def load_video(path, image_size, max_frames=0):
    try:
        frame_data = imageio.get_reader(path)
        frames = []
        last_image = None
        for frame in frame_data.iter_data():
            if last_image is not None:
                if (last_image == frame).all():
                    continue
            last_image = frame
            frame = cv2.resize(frame, (image_size, image_size))
            frame = frame[:, :, [0, 1, 2]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        pass
        # cap.release()
    return np.array(frames)


# def get_videos(train_df, test_df, output_dir, image_size, num_features, datagen, seq_len, exp_name, spec):
#     train_filepath = f'{output_dir}/full_frame/train_{image_size}_{num_features}_{exp_name}_{int(spec)}.pkl'
#     test_filepath = f'{output_dir}/full_frame/test_{image_size}_{num_features}_{exp_name}_{int(spec)}.pkl'
#     if os.path.exists(train_filepath) and os.path.exists(test_filepath):
#         with open(train_filepath, 'rb') as f:
#             train_d, train_l = pickle.load(f)
#         with open(test_filepath, 'rb') as f:
#             test_d, test_l = pickle.load(f)
#         return train_d, train_l, test_d, test_l
#     else:
#         train_d, train_l = prepare_all_videos(train_df, datagen, seq_len, num_features, image_size)
#         test_d, test_l = prepare_all_videos(test_df, datagen, seq_len, num_features, image_size)
#         with open(train_filepath, 'wb') as f:
#             pickle.dump((train_d, train_l), f)
#         with open(test_filepath, 'wb') as f:
#             pickle.dump((test_d, test_l), f)
#         return train_d, train_l, test_d, test_l


# def prepare_all_videos(df, datagen, seq_len, num_features, image_size):
#     num_samples = len(df)
#     video_paths = df["video_name"].values.tolist()
#     labels = df["tag"].values
#     # final_labels = []
#
#     # labels = label_processor(labels[..., None]).numpy()
#     # `frame_features` are what we will feed to our sequence model.
#     frame_features = np.zeros(
#         shape=(num_samples, seq_len, num_features), dtype="float32"
#     )
#     # For each video.
#     for idx, path in enumerate(video_paths):
#         print(f"Parsing {path}, {idx} of {len(video_paths) - 1}...")
#         # Gather all its frames and add a batch dimension.
#         frames = load_video(path, image_size)
#         # labels = [raw_labels[idx]] * sequence_len
#         frames = frames[None, ...]
#         # Initialize placeholder to store the features of the current video.
#         temp_frame_features = np.zeros(
#             shape=(1, seq_len, num_features), dtype="float32"
#         )
#         # Extract features from the frames of the current video.
#         for i, batch in enumerate(frames):
#             video_length = batch.shape[0]
#             length = min(seq_len, video_length)
#             for j in range(length):
#                 if np.mean(batch[j, :]) > 0.0:
#                     temp_frame_features[i, j, :] = feature_extractor.predict(
#                         batch[None, j, :]
#                     )
#                 else:
#                     temp_frame_features[i, j, :] = 0.0
#         frame_features[idx,] = temp_frame_features.squeeze()
#         # final_labels.append(labels)
#     return frame_features, labels


# def prepare_single_video(path, max_seq, num_features, image_size):
#     # Gather all its frames and add a batch dimension.
#     raw_frames = load_video(path, image_size, max_seq)
#     frames = raw_frames[None, ...]
#     # Initialize placeholder to store the features of the current video.
#     frame_features = np.zeros(
#         shape=(1, max_seq, num_features), dtype="float32"
#     )
#     # Extract features from the frames of the current video.
#     for i, batch in enumerate(frames):
#         video_length = batch.shape[0]
#         length = min(max_seq, video_length)
#         for j in range(length):
#             if np.mean(batch[j, :]) > 0.0:
#                 frame_features[i, j, :] = feature_extractor.predict(
#                     batch[None, j, :]
#                 )
#             else:
#                 frame_features[i, j, :] = 0.0
#     return raw_frames, frame_features


# def predict_action(model, path, seq_len, num_features, image_size):
#     # class_vocab = []
#     frames, frame_features = prepare_single_video(path, seq_len, num_features, image_size)
#     probabilities = model.predict(frame_features)[0]
#
#     for i in np.argsort(probabilities)[::-1]:
#         print(f"  {i}: {probabilities[i] * 100:5.2f}%")
#     return frames, probabilities


def to_gif(images, filepath):
    # This utility is for visualization.
    # Referenced from:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    converted_images = images.astype(np.uint8)
    imageio.mimsave(filepath, converted_images, fps=8)


# def extract_frame_features(frames, max_seq, num_features):
#     frames = frames[None, ...]
#
#     # Initialize placeholder to store the features of the current video.
#     frame_features = np.zeros(
#         shape=(1, max_seq, num_features), dtype="float32"
#     )
#
#     # Extract features from the frames of the current video.
#     for i, batch in enumerate(frames):
#         video_length = batch.shape[0]
#         length = min(max_seq, video_length)
#         for j in range(length):
#             if np.mean(batch[j, :]) > 0.0:
#                 frame_features[i, j, :] = feature_extractor.predict(
#                     batch[None, j, :]
#                 )
#
#             else:
#                 frame_features[i, j, :] = 0.0
#
#     return frame_features


def silent_predict_action(model, frame_features):
    probabilities = model.predict(frame_features).reshape(-1).tolist()
    return probabilities


# def inference_clips(model, path, image_size, max_frames, vocab, output, num_features, threshold):
#     window_size = 1
#     true_path = os.path.join(pathlib.Path(path).parent, pathlib.Path(path).stem + ".json")
#     with open(true_path) as f:
#         true_json = json.load(f)
#     ml_json = {k: v for k, v in true_json.items() if k in list(true_json.keys())[:15]}
#     true_event_h = true_json['Event History']
#     cap = cv2.VideoCapture(path)
#     frames = []
#     raw_frames = []
#     predictions = []
#     windows = []
#     frame_counter = 0
#     counter = 0
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     ml_file = os.path.join(output, f"{pathlib.Path(path).stem}_ML_clip_window_R.json")
#     event_history = []
#     y_pred, y_true = [], []
#     total_frames = cap.get(7)
#     event_count = 0
#     try:
#         for event in true_event_h:
#             event_count += 1
#             current_frame = int(event[2]) - 8
#             y_true.append(1)
#             cap.set(1, current_frame)
#             while True:
#                 ret, frame = cap.read()
#                 while not ret:
#                     ret, frame = cap.read()
#                 if not ret:
#                     break
#                 raw_frames.append(frame)
#                 frame = cv2.resize(frame, (image_size, image_size))
#                 frame = frame[:, :, [2, 1, 0]]
#                 frames.append(frame)
#                 current_frame += 1
#
#                 if len(frames) == max_frames:
#                     frames_seq = extract_frame_features(np.array(frames), max_frames, num_features)
#                     event_time = (current_frame - 8) / float(fps)
#                     event_pred = silent_predict_action(model, frames_seq)
#                     predictions.append(event_pred)
#                     windows.append(event_pred)
#                     if len(windows) == window_size:
#                         window_pred = 0.
#                         for window in windows:
#                             window_pred += window[0]
#                         window_pred /= float(window_size)
#                         print(f"{current_frame - 7}: {window_pred}")
#                         y_pred.append(window_pred)
#                         if window_pred > 0.5:
#                             frame_width = int(cap.get(3))
#                             frame_height = int(cap.get(4))
#                             out_file = f"{pathlib.Path(path).stem}_{event_pred[0]:.4f}_{int(event_time)}_{event_count}.mp4"
#                             out = cv2.VideoWriter(os.path.join(output, out_file),
#                                                   cv2.VideoWriter_fourcc(*'mp4v'), fps,
#                                                   (frame_width, frame_height))
#                             for frame in raw_frames:
#                                 out.write(frame)
#                             out.release()
#                             counter += 1
#                             event_history.append([vocab, int(event_time),
#                                                   current_frame - 7, None])
#                         windows = []
#                         raw_frames = []
#                     frames = []
#                     break
#     finally:
#         cap.release()
#         ml_json['Primary Data'] = "Reliability"
#         ml_json['Event History'] = event_history
#         ml_json['E4 Data'] = true_json['E4 Data']
#         ml_json['KSF'] = true_json['KSF']
#         with open(ml_file, 'w') as f:
#             json.dump(ml_json, f)
#         convert_json_csv([ml_file], [], pathlib.Path(ml_file).parent)
#         roc_file = os.path.join(output, f"{pathlib.Path(path).stem}_ML_clip_window_roc_R.csv")
#         with open(roc_file, mode='w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(y_pred)
#             writer.writerow(y_true)


# def inference_video_window(model, path, image_size, max_frames, vocab, output, num_features, threshold, window_size):
#     window_size = int(window_size / 2)
#     true_path = os.path.join(pathlib.Path(path).parent, pathlib.Path(path).stem + ".json")
#     with open(true_path) as f:
#         true_json = json.load(f)
#     ml_json = {k: v for k, v in true_json.items() if k in list(true_json.keys())[:15]}
#     true_event_h = true_json['Event History']
#     cap = cv2.VideoCapture(path)
#     frames = []
#     raw_frames = []
#     predictions = []
#     windows = []
#     frame_counter = 0
#     counter = 0
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     ml_file = os.path.join(output, f"{pathlib.Path(path).stem}_ML_{window_size}_window_R.json")
#     event_history = []
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             raw_frames.append(frame)
#             frame = cv2.resize(frame, (image_size, image_size))
#             frame = frame[:, :, [2, 1, 0]]
#             frames.append(frame)
#             frame_counter += 1
#
#             if len(frames) == max_frames:
#                 frames_seq = extract_frame_features(np.array(frames), max_frames, num_features)
#                 event_time = ((float(len(predictions)) * 16.0) + 8.0) / float(fps)
#                 event_pred = silent_predict_action(model, frames_seq)
#                 predictions.append(event_pred)
#                 windows.append(event_pred)
#                 if len(windows) == window_size:
#                     window_pred = 0.
#                     for window in windows:
#                         window_pred += window[0]
#                     window_pred /= float(window_size)
#                     print(f"{frame_counter - int((window_size * max_frames) / 2)}: {window_pred} | {windows}")
#                     if window_pred > threshold:
#                         # frame_width = int(cap.get(3))
#                         # frame_height = int(cap.get(4))
#                         # out_file = f"{pathlib.Path(path).stem}_{event_pred[0]:.4f}_{int(event_time)}.mp4"
#                         # out = cv2.VideoWriter(os.path.join(output, out_file),
#                         #                       cv2.VideoWriter_fourcc(*'mp4v'), fps,
#                         #                       (frame_width, frame_height))
#                         # for frame in raw_frames:
#                         #     out.write(frame)
#                         # out.release()
#                         counter += 1
#                         event_history.append([vocab, int(event_time),
#                                               frame_counter - int((window_size * max_frames) / 2), None])
#                     windows = []
#                     raw_frames = []
#                 frames = []
#     finally:
#         cap.release()
#         ml_json['Primary Data'] = "Reliability"
#         ml_json['Event History'] = event_history
#         ml_json['E4 Data'] = true_json['E4 Data']
#         ml_json['KSF'] = true_json['KSF']
#         with open(ml_file, 'w') as f:
#             json.dump(ml_json, f)
#         convert_json_csv([ml_file], [], pathlib.Path(ml_file).parent)


# def inference_video_sliding(model, path, image_size, max_frames, vocab, output, num_features, threshold, window_size):
#     window_size = window_size - 1
#     true_path = os.path.join(pathlib.Path(path).parent, pathlib.Path(path).stem + ".json")
#     with open(true_path) as f:
#         true_json = json.load(f)
#     ml_json = {k: v for k, v in true_json.items() if k in list(true_json.keys())[:15]}
#     cap = cv2.VideoCapture(path)
#     frames = []
#     raw_frames = []
#     predictions = []
#     windows = []
#     frame_counter = 0
#     counter = 0
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     ml_file = os.path.join(output, f"{pathlib.Path(path).stem}_ML_{window_size}_sliding_R.json")
#     event_history = []
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             raw_frames.append(frame)
#             frame = cv2.resize(frame, (image_size, image_size))
#             # cv2.imshow("crop", frame)
#             # cv2.waitKey(0)
#             frame = frame[:, :, [2, 1, 0]]
#             frames.append(frame)
#             frame_counter += 1
#
#             if len(frames) == max_frames:
#                 frames_seq = extract_frame_features(np.array(frames), max_frames, num_features)
#                 event_time = ((float(len(predictions)) * 16.0) + 8.0) / float(fps)
#                 event_pred = silent_predict_action(model, frames_seq)
#                 predictions.append(event_pred)
#                 windows.append(event_pred)
#                 if len(windows) == window_size:
#                     window_pred = 0.
#                     for window in windows:
#                         window_pred += window[0]
#                     window_pred /= float(window_size)
#                     print(f"{frame_counter - int((window_size * max_frames) / 2)}: {window_pred} | {windows}")
#                     if window_pred > 0.5:
#                         # frame_width = int(cap.get(3))
#                         # frame_height = int(cap.get(4))
#                         # out_file = f"{pathlib.Path(path).stem}_{event_pred[0]:.4f}_{int(event_time)}.mp4"
#                         # out = cv2.VideoWriter(os.path.join(output, out_file),
#                         #                       cv2.VideoWriter_fourcc(*'mp4v'), fps,
#                         #                       (frame_width, frame_height))
#                         # for frame in raw_frames:
#                         #     out.write(frame)
#                         # out.release()
#                         counter += 1
#                         event_history.append([vocab, int(event_time),
#                                               frame_counter - int((window_size * max_frames) / 2), None])
#                     windows = []
#                     raw_frames = []
#                 # Save the last second as the new first second
#                 frames = frames[int(max_frames / 2):]
#     finally:
#         cap.release()
#         ml_json['Primary Data'] = "Reliability"
#         ml_json['Event History'] = event_history
#         ml_json['E4 Data'] = true_json['E4 Data']
#         ml_json['KSF'] = true_json['KSF']
#         with open(ml_file, 'w') as f:
#             json.dump(ml_json, f)
#         convert_json_csv([ml_file], [], pathlib.Path(ml_file).parent)


# def inference_video_datapal(model, path, image_size, max_frames, vocab, output, num_features, threshold):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     raw_frames = []
#     predictions = []
#     counter = 0
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     dp_file = f"{pathlib.Path(path).stem}.MC"
#     # Write the standard header for our current KSF file
#     with open(os.path.join(output, dp_file), 'w') as f:
#         f.write(
#             '''Client: ML Coding
# Medical Record Number: 9999
# Date: 01.4.22
# Time: 2:13:36 PM
# Session Number: 12_12_13_10111_1
# Location:
# Assessment:
# Condition:
# Primary Therapist: Seth Walker
# Case Manager: Seth Walker
# Session Therapist:
# Data Recorder:
# Data Type:
#
# KEY DEFINITIONS
#  Listed as key, key description, active during PauseTime or Pause, Active during SessionTime
#     Frequency Keys
# a, hitting, no, yes
# s, kicking, no, yes
# d, pushing, no, yes
# f, grab scratch, no, yes
# g, head butting, no, yes
# j, hair pulling, no, yes
# h, biting, no, yes
# k, choking, no, yes
# l, sib-head banging, no, yes
# q, sib-head hit, no, yes
# w, sib-self hit, no, yes
# e, sib-biting, no, yes
# r, sib-eye poking, no, yes
# t, sib-body slam, no, yes
# y, sib-hair pulling, no, yes
# u, sib-choking, no, yes
# i, sib-pinch scratch, no, yes
# o, throwing obj, no, yes
# p, kick hit obj, no, yes
# z, flip furniture, no, yes
# n, flopping, no, yes
#     Duration Keys
# 1, st-rocking, no, yes
# 2, st-hand flap, no, yes
# 3, st-touch tap, no, yes
# 4, st-head swin, no, yes
# 5, stereo-vox, no, yes
# EVENT RECORDING START
# SessionTime, TI, SessionTime, 0
# '''
#         )
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             raw_frames.append(frame)
#             frame = cv2.resize(frame, (image_size, image_size))
#             frame = frame[:, :, [2, 1, 0]]
#             frames.append(frame)
#
#             if len(frames) == max_frames:
#                 frames_seq = extract_frame_features(np.array(frames), max_frames, num_features)
#                 event_time = ((float(len(predictions)) * 16.0) + 8.0) / float(fps)
#                 event_pred = silent_predict_action(model, frames_seq)
#                 predictions.append(event_pred)
#                 if event_pred[0] > threshold:
#                     frame_width = int(cap.get(3))
#                     frame_height = int(cap.get(4))
#                     out_file = f"{pathlib.Path(path).stem}_{event_pred[0]:.4f}_{counter}.mp4"
#                     out = cv2.VideoWriter(os.path.join(output, out_file),
#                                           cv2.VideoWriter_fourcc(*'mp4v'), fps,
#                                           (frame_width, frame_height))
#                     for frame in raw_frames:
#                         out.write(frame)
#                     out.release()
#                     counter += 1
#                     print(f"\nEvent Count: {counter}")
#                     print(f"  {vocab}: {event_pred[0] * 100:5.2f}%")
#                     with open(os.path.join(output, dp_file), 'a') as f:
#                         f.write(f'"Freq","a","hitting",{event_time:.1f}\n')
#                 frames = []
#                 raw_frames = []
#     finally:
#         cap.release()
#     with open(os.path.join(output, dp_file), 'a') as f:
#         f.write('End, Session State, Session Ended, 1200')


# def model_inference(model, inference_dir, video_paths, target_label, threshold, checkpoint_dir, image_size, seq_len,
#                     window_sizes, datapal_video_paths):
#     # Get latest weights and load into model
#     # weights_dir = pathlib.Path(checkpoint_dir)
#     # weights_pattern = r'*.hdf5'
#     # latest_weight = max(weights_dir.glob(weights_pattern), key=lambda f: f.stat().st_ctime)
#     # print(f"Loaded weights for testing: {latest_weight}")
#     # Test model using test data
#     # model = get_compiled_model_2(seq_len, num_features)
#     # model.load_weights(latest_weight)
#
#     for video_path in video_paths:
#         # # Perform an overlapping window inferencing
#         # for window_size in window_sizes:
#         #     output_path = os.path.join(inference_dir, pathlib.Path(video_path).stem + f"_WS{window_size}_OV")
#         #     if not os.path.exists(output_path):
#         #         os.mkdir(output_path)
#         #         inference_video_sliding(model, video_path, image_size, seq_len, target_label,
#         #                                 output_path, num_features, threshold, window_size)
#         # # Perform a non-overlapping window inferencing
#         # for window_size in window_sizes:
#         #     output_path = os.path.join(inference_dir, pathlib.Path(video_path).stem + f"_WS{window_size}_NOV")
#         #     # If the folder already exists, assume it's been inferenced
#         #     if not os.path.exists(output_path):
#         #         os.mkdir(output_path)
#         #         inference_video_window(model, video_path, image_size, seq_len, target_label,
#         #                                output_path, num_features, threshold, window_size)
#         output_path = os.path.join(inference_dir, pathlib.Path(video_path).stem + f"_CLIPS_NOV")
#         if not os.path.exists(output_path):
#             os.mkdir(output_path)
#             inference_video_sliding(model, video_path, image_size, seq_len, target_label,
#                                     output_path, num_features, threshold, 1)
#         print(f"Finished inferencing video {pathlib.Path(video_path).name}")
#     for video_path in datapal_video_paths:
#         output_path = os.path.join(inference_dir, pathlib.Path(video_path).stem + f"_NOV_DP")
#         if not os.path.exists(output_path):
#             os.mkdir(output_path)
#             inference_video_datapal(model, video_path, image_size, seq_len, target_label,
#                                     output_path, num_features, threshold)
#         print(f"Finished inferencing video {pathlib.Path(video_path).name}")


def get_pandas_dataset():
    # Parse dataset into train and test sets
    create_pandas_dataset(r'C:\UNMC Data\Problematic Behavior Recognition\OneDrive_2022-01-21\2. event folder')
    # Turn CSV datasets into Pandas Data Frames
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # Define label map
    label_map = {
        'hitting': 1,
        'kicking': 0,
        'pushing': 0,
        'grabbingscratching': 0,
        'head butting': 0,
        'hair pull': 0,
        'biting': 0,
        'choking': 0,
        'SIB headbang': 0,
        'SIB headhit': 0,
        'SIB self-hit': 0,
        'SIB biting': 0,
        'SIB eyepoke': 0,
        'SIB body slam': 0,
        'SIB hair pull': 0,
        'SIB choking': 0,
        'SIB pinch scratch': 0,
        'throw object': 0,
        'kick hit object': 0,
        'flip furniture': 0,
        'flopping': 0,
        'stereoypy rocking': 0,
        'stereoypy hand flap': 0,
        'no pbx': 0,
    }
    target_label = 'hitting'
    # Simplify to binary classification
    train_df['tag'] = train_df['tag'].map(label_map)
    train_df.dropna(inplace=True)
    # Simplify to binary classification
    test_df['tag'] = test_df['tag'].map(label_map)
    test_df.dropna(inplace=True)
    print(f"Total videos for training: {len(train_df)}")
    print(f"Total videos for testing: {len(test_df)}")


# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)
# for later versions:
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
K.set_session(sess)

results_headers = ['Class', 'Test Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC-AUC', 'mAP', 'EER',
                                   'TPR@1', 'TPR@5', 'TPR@10', 'AUPRC', 'Kappa', 'Model', 'Feature Extractor',
                                   'Convergence Epoch', 'Subject', 'Sensitivity', 'Specificity']
subject_split = True
balance = False
mc = False
if __name__ == '__main__':
    video_paths = [
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 1.avi",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 2.avi",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 3.mp4",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 4.mp4",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 5.mp4",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 6.mp4",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 7.mp4",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 8.mp4",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 9.mp4",
        r"C:\UNMC Data\Problematic Behavior Recognition\novel_inference\video 10.mp4"
    ]
    datapal_video_paths = [
        # r"C:\UNMC Data\Problematic Behavior Recognition\Test video set\Test video set\2021-12-03 10111 PTZ_3.avi",
        # r"C:\UNMC Data\Problematic Behavior Recognition\Test video set\Test video set\2021-12-03 10111_3.avi",
        # r"C:\UNMC Data\Problematic Behavior Recognition\Test video set\Test video set\2021-12-06 10111 PTZ_2.avi",
        # r"C:\UNMC Data\Problematic Behavior Recognition\Test video set\Test video set\2021-12-06 10111_2.avi",
        # r"C:\UNMC Data\Problematic Behavior Recognition\Test video set\Test video set\2021-12-17 10111 PTZ_5.avi",
        # r"C:\UNMC Data\Problematic Behavior Recognition\Test video set\Test video set\2021-12-17 10111_5.avi"
    ]
    final_results = pd.DataFrame()
    for experiment in full_experiment_schedule:
        subject_schedule = ['p001', 'p003', 'p005', 'p008'] if subject_split else [None]
        for target_subject in subject_schedule:
            # Get experiment variables
            spec = experiment["dense-net"]
            tracklet = experiment["tracklet"]
            model_choice = experiment["model_choice"]
            label_map = experiment["label-map"]  # binary_label_map if not mc else mc_label_map
            full_label_map = experiment["full-label-map"]  # binary_full_label_map if not mc else mc_full_label_map

            sequence_len = 16
            epochs = 200
            batch_size = 128
            experiment_name = f"{'Tracklet' if tracklet else 'FullFrame'}_{str(model_choice)}_{str(spec)}_{len(full_label_map)-1}_{'mc' if mc else 'hitting'}_{target_subject}"
            image_size = 256 if model_choice != 1 else 128
            root_dir = "./MLHC/Patients"

            if not os.path.exists(root_dir):
                os.mkdir(root_dir)
            # Dataset output dir allows sharing of dataset generation across experiments
            dataset_dir = os.path.join(root_dir, 'video datasets')
            if not os.path.exists(dataset_dir):
                os.mkdir(dataset_dir)
            # Create our data output dir and dataset output dir
            output_dir = os.path.join(root_dir, experiment_name)
            # Create output dir if it does not exist
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            # Create our inference dir so we can store our tested model results
            inference_dir = os.path.join(output_dir, 'inference')
            if not os.path.exists(inference_dir):
                os.mkdir(inference_dir)
            # Create the output directory for the model checkpoints
            checkpoint_dir = os.path.join(output_dir, f'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            start = time.time()
            if len(os.listdir(checkpoint_dir)) == 0:
                random.seed(seed_value)
                train_data, train_labels, test_data, test_labels = create_tracklet_pandas_dataset(all_classes,
                                                                                                  model_choice,
                                                                                                  image_size,
                                                                                                  sequence_len,
                                                                                                  dataset_dir,
                                                                                                  spec,
                                                                                                  tracklet,
                                                                                                  subject=target_subject)

                if mc:
                    label_map = list(set([x for x in test_labels if x != 'no pbx']))
                    full_label_map = list(set([x for x in test_labels]))
                train_labels = [1 if x in label_map else 0 for x in train_labels]
                test_labels = [1 if x in label_map else 0 for x in test_labels]

                if balance:
                    hitting_labels = np.where(np.array(train_labels) == 1)[0]
                    nopbx_labels = np.where(np.array(train_labels) == 0)[0]
                    random.seed(seed_value)
                    keep_nopbx = np.array(random.sample(list(nopbx_labels), len(hitting_labels)))
                    train_data = np.array(
                        list(np.array(train_data)[hitting_labels]) + list(np.array(train_data)[keep_nopbx]))
                    train_labels = np.array(
                        list(np.array(train_labels)[hitting_labels]) + list(np.array(train_labels)[keep_nopbx]))

                    hitting_labels = np.where(np.array(test_labels) == 1)[0]
                    nopbx_labels = np.where(np.array(test_labels) == 0)[0]
                    random.seed(seed_value)
                    keep_nopbx = np.array(random.sample(list(nopbx_labels), len(hitting_labels)))
                    test_data = np.array(
                        list(np.array(test_data)[hitting_labels]) + list(np.array(test_data)[keep_nopbx]))
                    test_labels = np.array(
                        list(np.array(test_labels)[hitting_labels]) + list(np.array(test_labels)[keep_nopbx]))

                # if (len(full_label_map) - 1) != 1:
                #     train_labels = keras.utils.to_categorical(train_labels)
                #     test_labels = keras.utils.to_categorical(test_labels)

                if subject_split:
                    train_labels = np.array(train_labels)
                    test_labels = np.array(test_labels)
                else:
                    train_data = np.vstack((train_data, test_data))
                    train_labels = np.hstack((train_labels, test_labels))

                fold_results = pd.DataFrame()

                rng = np.random.RandomState(0)
                if not subject_split:
                    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
                    train_set, val_set = list(k_fold.split(train_data, train_labels))[0]
                    val_data, val_labels = train_data[val_set], train_labels[val_set]
                else:
                    train_set = np.arange(len(train_data))
                    val_data, val_labels = test_data, test_labels
                k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)

                for i in range(len(full_label_map)):
                    print(
                        f"Baseline {full_label_map[i]} Train Accuracy: {(list(train_labels[train_set]).count(i) / len(train_labels[train_set])) * 100:.2f}%")
                    print(
                        f"Baseline {full_label_map[i]} Test Accuracy: {(list(val_labels).count(i) / len(val_labels)) * 100:.2f}%")

                fold = 1
                for train_ix, test_ix in k_fold.split(train_data[train_set], train_labels[train_set]):
                    train_x, x_test = train_data[train_ix], train_data[test_ix]
                    train_y, y_test = train_labels[train_ix], train_labels[test_ix]
                    # Run the experiment
                    trained_model, results, ce = run_experiment(train_x, train_y, x_test, y_test, len(full_label_map) - 1,
                                                                output_dir, epochs, image_size, sequence_len,
                                                                FeatureExtractorFeatures[int(spec)],
                                                                checkpoint_dir, fold,
                                                                model_choice=model_choice,
                                                                batch_size=batch_size,
                                                                label_map=full_label_map,
                                                                val_x=val_data, val_y=val_labels)
                    results = results[0]
                    results = results + [models[model_choice], fes[int(spec)], ce, target_subject]
                    fold_results = pd.concat((fold_results, pd.DataFrame(results, index=results_headers)), axis=1)
                    fold += 1
                fold_results = fold_results.transpose()
                folds_results = [
                    list(fold_results['Class'])[0],
                    f"{np.mean(fold_results[results_headers[1]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[1]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[2]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[2]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[3]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[3]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[4]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[4]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[5]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[5]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[6]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[6]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[7]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[7]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[8]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[8]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[9]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[9]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[10]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[10]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[11]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[11]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[12]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[12]]) * 100:.2f}",
                    list(fold_results['Model'])[0],
                    list(fold_results['Feature Extractor'])[0],
                    f"{np.mean(fold_results[results_headers[15]]):.2f} \u00B1 {np.std(fold_results[results_headers[15]]):.2f}",
                    target_subject
                ]
                fold_results.to_excel(os.path.join(output_dir, 'fold_results.xlsx'))
                final_results = pd.concat((final_results, pd.DataFrame(folds_results, index=results_headers)), axis=1)
            else:
                fold_results = pd.read_excel(os.path.join(output_dir, 'fold_results.xlsx'))
                sn, sp = [], []
                for i in range(1, 6):
                    pred_file = pd.read_csv(os.path.join(output_dir, 'history', f"{i}_pred.csv"))
                    pred_file = pred_file.set_index("Unnamed: 0").transpose()
                    tn, fp, fn, tp = confusion_matrix(pred_file['True'], pred_file['Predicted'] > 0.5).ravel()
                    sp.append(tn / (tn + fp))
                    sn.append(tp / (tp + fn))
                folds_results = [
                    list(fold_results['Class'])[0],
                    f"{np.mean(fold_results[results_headers[1]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[1]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[2]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[2]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[3]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[3]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[4]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[4]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[5]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[5]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[6]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[6]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[7]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[7]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[8]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[8]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[9]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[9]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[10]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[10]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[11]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[11]]) * 100:.2f}",
                    f"{np.mean(fold_results[results_headers[12]]) * 100:.2f} \u00B1 {np.std(fold_results[results_headers[12]]) * 100:.2f}",
                    list(fold_results['Model'])[0],
                    list(fold_results['Feature Extractor'])[0],
                    f"{np.mean(fold_results[results_headers[15]]):.2f} \u00B1 {np.std(fold_results[results_headers[15]]):.2f}",
                    target_subject,
                    f"{np.mean(sn) * 100:.2f} \u00B1 {np.std(sn) * 100:.2f}",
                    f"{np.mean(sp) * 100:.2f} \u00B1 {np.std(sp) * 100:.2f}",
                ]
                final_results = pd.concat((final_results, pd.DataFrame(folds_results, index=results_headers)), axis=1)
            end = time.time()
            print(f"Experiment {experiment_name} completed in {end - start}")
    print("All experiments completed, saving data...")
    final_results.transpose().to_excel(os.path.join(root_dir, "final_results.xlsx"))
