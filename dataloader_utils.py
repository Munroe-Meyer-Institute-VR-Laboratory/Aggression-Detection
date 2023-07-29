import csv
import glob
import math
import os
import pathlib
import pickle
import random

import cv2
import numpy as np
import pandas as pd
import torch

import project_utils
from cnn_transformer import build_feature_extractor, FeatureExtractorSpec, FeatureExtractorFeatures
from tag_lut import tag_count, freq_tag_lut, dur_tag_lut, all_tags, nb_tag, all_classes


def load_frame_tags(tag_dir):
    frames = {}
    _, _, files = next(os.walk(tag_dir))
    for file in files:
        with open(os.path.join(tag_dir, file), 'rb') as f:
            frames[pathlib.Path(file).stem] = pickle.load(f)
    return frames


def load_videos(video_dir):
    files = []
    _, _, video_files = next(os.walk(video_dir))
    for file in video_files:
        files.append(os.path.join(video_dir, file))
    return files


def load_videos_sorted_dir(top_dir):
    root, folders, _ = next(os.walk(top_dir))
    videos = []
    video_classes = []
    for folder in folders:
        video_class = folder.split('.')[0]
        video = []
        _, _, files = next(os.walk(os.path.join(root, folder)))
        for file in files:
            if pathlib.Path(file).suffix == ".mp4":
                video.append(os.path.join(root, folder, file))
        if video:
            video_classes.append(video_class)
            videos.append(video)
    return videos, video_classes

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def load_preliminary_dataset(classes):
    if os.path.exists('tracklet_videos.pkl'):
        with open('tracklet_videos.pkl', 'rb') as f:
            videos, labels, participants = pickle.load(f)
        return videos, labels, participants
    else:
        files = [f for f in glob.glob(rf'C:\GitHub\Keypoint-LSTM\datasets\stills\MMISB Cropped\**\*.txt', recursive=True) if
                 'classes' not in f]
        videos = [[] for x in range(len(classes))]
        labels = [[] for x in range(len(classes))]
        participants = [[] for x in range(len(classes))]
        processed_files = []
        for f in files:
            if f not in processed_files:
                video_name = str(pathlib.Path(f).parent)
                video = [x for x in files if str(pathlib.Path(x).parent) == video_name in x]
                sort_nicely(video)
                # if len(video) != 15:
                #     print(video_name)
                label = pathlib.Path(f).parts[-3]
                participant = pathlib.Path(f).parts[-4]
                if label in classes:
                    processed_files.extend(video)
                    videos[classes.index(label)].append(video)
                    labels[classes.index(label)].append(label)
                    participants[classes.index(label)].append(participant)
            else:
                continue
        with open('tracklet_videos.pkl', 'wb') as f:
            pickle.dump((videos, labels, participants), f)
        return videos, labels, participants


def create_video_sample(video, image_size, tracklet, img_read=cv2.IMREAD_UNCHANGED):
    frame_tensors = []
    # writer = imageio.get_writer(f'debug_video.mp4',
    #                             fps=8.0)
    # height, width = 0, 0
    # for still in self.accepted_c_stills:
    #     if still.shape[0] > width:
    #         width = still.shape[0]
    #     if still.shape[1] > height:
    #         height = still.shape[1]
    # for still in self.accepted_c_stills:
    #     writer.append_data(cv2.resize(still, (height, width)))
    # writer.close()
    for anno in video:
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
            if annotation and tracklet:
                if int(annotation.split(' ')[0]) == 0:
                    _, x, y, w, h = map(float, annotation.split(' '))
                    frame = cv2.imread(anno[:-3] + 'png', img_read)

                    dh, dw = frame.shape[0], frame.shape[1]
                    l = int((x - w / 2) * dw)
                    t = int((y - h / 2) * dh)
                    w = int(w * dw)
                    h = int(h * dh)

                    cropped_image = frame[t:t+h, l:l+w]
                    resized_crop = cv2.resize(cropped_image, (image_size, image_size))
                    # writer.append_data(resized_crop)
                    if img_read == cv2.IMREAD_UNCHANGED:
                        resized_frame = resized_crop[:, :, [0, 1, 2]]
                    else:
                        resized_frame = resized_crop
                    frame_tensors.append(resized_frame)
            else:
                # Just load the entire frame if no annotation found
                frame = cv2.imread(anno[:-3] + 'png', img_read)
                resized_crop = cv2.resize(frame, (image_size, image_size))
                # writer.append_data(resized_crop)
                if img_read == cv2.IMREAD_UNCHANGED:
                    resized_frame = resized_crop[:, :, [0, 1, 2]]
                else:
                    resized_frame = resized_crop
                frame_tensors.append(resized_frame)
    return np.array(frame_tensors)


def create_tracklet_pandas_dataset(classes, model_choice, image_size, seq_len,
                                   output_dir, spec, tracklet, channels=1, split=0.8, subject='p001'):
    num_features = FeatureExtractorFeatures[int(spec)]
    if tracklet:
        train_filepath = f'{output_dir}/tracklet/train_{image_size}_{num_features}_{int(spec)}.pkl'
    else:
        train_filepath = f'{output_dir}/full_frame/train_{image_size}_{num_features}_{int(spec)}.pkl'
    if os.path.exists(train_filepath):
        with open(train_filepath, 'rb') as f:
            v, l, p = pickle.load(f)
    else:
        # Build the feature extractor
        if model_choice != 1:
            feature_extractor = build_feature_extractor(spec, image_size)
            # num_features = feature_extractor.output_shape[1]
            print(f"Number of Features: {num_features}")
        else:
            feature_extractor = None
            num_features = None
            batch_size = 2
        videos, labels, participants = load_preliminary_dataset(classes)
        v, l, p = [], labels, participants
        for class_idx, vid in enumerate(videos):
            print(f"Processing {class_idx} / {len(videos)}...")
            if num_features:
                frame_v = np.zeros(
                    shape=(len(vid), seq_len, num_features), dtype="float32"
                )
            else:
                frame_v = np.zeros(
                    shape=(len(vid), seq_len, image_size, image_size, channels), dtype="float32"
                )

            for idx, video in enumerate(vid):
                # print(f"Processing {idx} / {len(vid)}...")
                frames = create_video_sample(video, image_size, tracklet, img_read=cv2.IMREAD_GRAYSCALE)
                frames = frames[None, ...]
                # Initialize placeholder to store the features of the current video.
                if num_features:
                    temp_frame_features = np.zeros(
                        shape=(1, seq_len, num_features), dtype="float32"
                    )
                else:
                    temp_frame_features = np.zeros(
                        shape=(1, seq_len, image_size, image_size, channels), dtype="float32"
                    )
                if feature_extractor:
                    # Extract features from the frames of the current video.
                    for i, batch in enumerate(frames):
                        video_length = batch.shape[0]
                        length = min(seq_len, video_length)
                        for j in range(length):
                            if np.mean(batch[j, :]) > 0.0:
                                temp_frame_features[i, j, :] = feature_extractor.predict(
                                    np.squeeze(batch[None, j, :]), verbose=0
                                )
                            else:
                                temp_frame_features[i, j, :] = 0.0
                    frame_v[idx, ] = temp_frame_features.squeeze()
                else:
                    for i, batch in enumerate(frames):
                        video_length = batch.shape[0]
                        length = min(seq_len, video_length)
                        for j in range(length):
                            temp_frame_features[i, j, :] = cv2.cvtColor(batch[j, :], cv2.COLOR_BGR2GRAY)[..., None]
                    frame_v[idx, ] = temp_frame_features
            v.append(frame_v)
        with open(train_filepath, 'wb') as f:
            pickle.dump((v, l, p), f)

    train_videos, train_labels, train_participants = [], [], []
    test_videos, test_labels, test_participants = [], [], []

    train_samples, test_samples = 0, 0

    for videos, labels, participants in zip(v, l, p):
        if videos.shape[0] and labels:
            if videos.shape[0] != 1:
                if labels[0] in classes:
                    if type(subject) is str:
                        training_samples = [i for i, x in enumerate(videos) if participants[i] != subject]
                        train_size = len(training_samples)
                    else:
                        train_size = math.floor(len(videos) * split)
                        training_samples = np.array(random.sample(list(np.arange(len(videos))), train_size))
                    train_v = videos[training_samples]
                    train_l = list(np.array(labels)[training_samples])
                    train_p = list(np.array(participants)[training_samples])
                    train_videos[train_samples:train_samples+train_size] = train_v
                    train_samples += train_size
                    train_labels.extend(train_l)
                    train_participants.extend(train_p)

                    testing_samples = [x for x in np.arange(len(videos)) if x not in training_samples]
                    test_v = videos[testing_samples]
                    test_l = list(np.array(labels)[testing_samples])
                    test_p = list(np.array(participants)[testing_samples])
                    test_videos[test_samples:test_samples+(len(videos)-train_size)] = test_v
                    test_samples += (len(videos) - train_size)
                    test_labels.extend(test_l)
                    test_participants.extend(test_p)

    return np.array(train_videos), np.array(train_labels), np.array(test_videos), np.array(test_labels)


def create_pandas_dataset(data_dir):
    videos, classes = load_videos_sorted_dir(data_dir)
    train_videos, eval_videos = [], []
    train_tags, eval_tags = [], []
    for video_class, classe in zip(videos, classes):
        # np.random.shuffle(video_class)
        if classe in all_classes:
            train_index = int(len(video_class) * 0.9)
            train_videos.extend(video_class[:train_index])
            train_tags.extend([all_classes.index(classe)] * train_index)
            eval_videos.extend(video_class[train_index:])
            eval_tags.extend([all_classes.index(classe)] * (len(video_class) - train_index))
    train = pd.DataFrame()
    test = pd.DataFrame()
    # for video in train_videos:
    #     train_tags.append(all_class_list[int(pathlib.Path(video).stem.split('_')[-3])])
    # for video in eval_videos:
    #     eval_tags.append(all_class_list[int(pathlib.Path(video).stem.split('_')[-3])])
    train['video_name'] = train_videos
    train['tag'] = train_tags
    train = train[:-1]
    train.head()
    test['video_name'] = eval_videos
    test['tag'] = eval_tags
    test = test[:-1]
    test.head()
    train_new = train.reset_index(drop=True)
    test_new = test.reset_index(drop=True)
    train_new.to_csv("train.csv", index=False)
    test_new.to_csv("test.csv", index=False)


def create_csv_dataset(data_dir):
    videos = load_videos(data_dir)
    train_index = int(len(videos) * 0.8)
    train_videos, eval_videos = videos[:train_index], videos[train_index:]
    with open(r'old_results/dataset\train_video_csv_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'tag'])
        for video in train_videos:
            tag = pathlib.Path(video).stem.split('_')[-3]
            writer.writerow([str(video), str(tag)])
    with open(r'old_results/dataset\val_video_csv_file.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'tag'])
        for video in eval_videos:
            tag = pathlib.Path(video).stem.split('_')[-3]
            writer.writerow([str(video), str(tag)])


def create_video_dataset(data_dir):
    output_file = r'/old_results/dataset\video_dataset.pkl'
    if os.path.exists(output_file):
        dataset = torch.load(output_file)
        return dataset
    videos = load_videos(data_dir)
    event_ticker = [0] * 3
    x_data = [[] for i in range(3)]
    y_data = [[] for i in range(3)]
    print("Parsing video dataset...")
    for video in videos:
        tag = pathlib.Path(video).stem.split('_')[-3]
        event_ticker[int(tag)] = event_ticker[int(tag)] + 1
        video_data = cv2.VideoCapture(video)
        clips = []
        while video_data.isOpened():
            ret, frame = video_data.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clips.append(project_utils.transform(image=frame)['image'])
            else:
                break
        input_frames = np.array(clips)
        # add an extra dimension
        input_frames = np.expand_dims(input_frames, axis=0)
        # transpose to get [1, 3, num_clips, height, width]
        input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
        # convert the frames to tensor
        input_frames = torch.tensor(input_frames, dtype=torch.float32)
        # tag = torch.tensor(frame_tag[1], dtype=torch.long)
        target = [int(tag)]
        # target = np.expand_dims(target, axis=0)
        target = torch.tensor(target, dtype=torch.int64)
        x_data[int(tag)].append(input_frames)
        y_data[int(tag)].append(target)
    print("Finished parsing video dataset...")
    x_train, y_train, x_eval, y_eval, x_test, y_test = [], [], [], [], [], []
    # Shuffle data in order

    # np.random.shuffle(data)
    for x, y in zip(x_data, y_data):
        data = list(zip(x, y))
        np.random.shuffle(data)
        x, y = zip(*data)
        # np.random.shuffle(data)
        train_index = int(len(x) * 0.7)
        test_index = int(len(x) * 0.2) + train_index
        x_train.extend(x[:train_index])
        y_train.extend(y[:train_index])
        x_eval.extend(x[train_index:test_index])
        y_eval.extend(y[train_index:test_index])
        x_test.extend(x[test_index:])
        y_test.extend(y[test_index:])
    dataloader_dict = {'train': (x_train, y_train),
                       'val': (x_eval, y_eval),
                       'test': (x_test, y_test)
                       }
    print("Writing dataset to file...")
    torch.save(dataloader_dict, output_file)
    print(event_ticker)
    return dataloader_dict


def create_dataset(video_dir, tag_dir, tag_width, split=[0.7, 0.2, 0.1], output_dir=None, tag_filter=None):
    video_output_dir = r"/old_results/dataset\video_datasets\event_videos"
    loaded = False
    if tag_filter:
        dataset = [[] for i in range(len(tag_filter))]
    else:
        dataset = [[] for i in range(tag_count)]
    if output_dir:
        if os.path.exists(os.path.join(output_dir, f"dataset_{tag_width}.pkl")):
            print("Dataset already exists, loading...")
            output_file = os.path.join(output_dir, f"dataset_{tag_width}.pkl")
            dataset = torch.load(output_file)
            loaded = True
            if tag_filter:
                if len(dataset) != len(tag_filter):
                    print("Dataset is not the same as requested, regenerating...")
                    dataset = [[] for i in range(len(tag_filter))]
                    loaded = False
            else:
                if len(dataset) != tag_count:
                    print("Dataset is not the same as requested, regenerating...")
                    dataset = [[] for i in range(len(tag_count))]
                    loaded = False
    if not loaded:
        print("Generating dataset...")
        frames = load_frame_tags(tag_dir)
        videos = load_videos(video_dir)
        event_count = 0
        for video in videos:
            end_frame = -1
            print(f"Parsing {video}...")
            frame_tags = frames[pathlib.Path(video).stem]
            video_data = cv2.VideoCapture(video)
            frame_width = int(video_data.get(3))
            frame_height = int(video_data.get(4))
            for frame_tag in frame_tags:
                if tag_filter:
                    if not frame_tag[1] in tag_filter:
                        continue
                    else:
                        tag = tag_filter.index(frame_tag[1])
                else:
                    tag = frame_tag[1]

                data = dict()
                raw_clips, clips = [], []
                start_frame = int(abs(frame_tag[2] - tag_width / 2))
                # # If we added a no-behavior, then filter out any that conflict with the last event
                # if start_frame < end_frame:
                #     continue
                end_frame = int(abs(frame_tag[2] + tag_width / 2))
                out = cv2.VideoWriter(os.path.join(video_output_dir, f"{pathlib.Path(video).stem}_{tag}_{start_frame}_{end_frame}_{event_count}.mp4"),
                                      cv2.VideoWriter_fourcc(*'mp4v'), 8,
                                      (frame_width, frame_height))
                event_count += 1
                for i in range(start_frame, end_frame):
                    video_data.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = video_data.read()
                    if ret:
                        out.write(frame)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        clips.append(project_utils.transform(image=frame)['image'])
                input_frames = np.array(clips)
                # add an extra dimension
                input_frames = np.expand_dims(input_frames, axis=0)
                # transpose to get [1, 3, num_clips, height, width]
                input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
                # convert the frames to tensor
                input_frames = torch.tensor(input_frames, dtype=torch.float32)
                # tag = torch.tensor(frame_tag[1], dtype=torch.long)
                data["Clip"] = input_frames
                data["Tag"] = tag
                data["Path"] = video
                dataset[tag].append(data)
        if output_dir:
            print("Writing dataset to file...")
            output_file = os.path.join(output_dir, f"dataset_{tag_width}.pkl")
            torch.save(dataset, output_file)
    train_dataset, eval_dataset, test_dataset = [], [], []
    for data in dataset:
        # np.random.shuffle(data)
        train_index = int(len(data) * split[0])
        test_index = int(len(data) * split[1]) + train_index
        train_dataset.extend(data[:train_index])
        eval_dataset.extend(data[train_index:test_index])
        test_dataset.extend(data[test_index:])
    # np.random.shuffle(train_dataset)
    # np.random.shuffle(eval_dataset)
    # np.random.shuffle(test_dataset)
    print("Finished generating dataset...")
    return train_dataset, eval_dataset, test_dataset


def generate_raw_data(tag_dir, video_dir, output_dir):
    output_files = []
    event_ticker = [0] * tag_count
    _, _, files = next(os.walk(tag_dir))
    _, _, video_files = next(os.walk(video_dir))
    total_tags = 0
    print("Parsing DataPal files to pickle format...")
    for file in files:
        start_parsing = False
        output_dict = {
            "Freq": [],
            "Dur": []
        }
        with open(os.path.join(tag_dir, file), 'r') as f:
            paused = False
            adjust_time = 0
            for line in f:
                split = line.split(":")
                if len(split) > 1:
                    output_dict[split[0]] = split[1].strip()
                if "EVENT RECORDING START" in line:
                    start_parsing = True
                elif start_parsing:
                    splits = line.split(',')
                    for i in range(0, len(splits)):
                        splits[i] = str(splits[i].strip("\""))
                    if splits[0] == "PauseTime":
                        end_time = float(splits[3])
                        paused = True
                    if splits[0] == "SessionTime" and paused:
                        start_time = float(splits[3])
                        adjust_time = adjust_time + (start_time - end_time)
                        paused = False
                    if splits[0] == "Freq":
                        output_dict[splits[0]].append((str(splits[2]), float(splits[3]) - adjust_time))
                    if splits[0] == "Dur":
                        output_dict[splits[0]].append((str(splits[2]), float(splits[3]) - adjust_time))
                    if splits[0] == "End":
                        output_dict["Session Length"] = float(splits[3]) - adjust_time
                        print("File: {0} | Adjusted Session Length: {1}".format(file, output_dict["Session Length"]))

        output_file = os.path.join(output_dir, pathlib.Path(file).stem + ".pkl")
        output_files.append(output_file)
        with open(output_file, 'wb') as f:
            pickle.dump(output_dict, f)
    print("Finished DataPal parsing to pickle format...\n")
    print("Adjusting event times for session length...")
    for video_file, output_file in zip(video_files, output_files):
        with open(output_file, 'rb') as f:
            tags = pickle.load(f)
        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        if not cap.isOpened():
            print('Error while trying to read video. Please check path again')
            continue
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        seconds = int(frames / fps)
        frame_tags = []
        cap.release()
        time_adj = abs(seconds - float(tags["Session Length"]))
        print("\nVideo Length (s):", seconds, "| Adjustment:", time_adj, "| Path:", video_file)
        for i in range(0, len(tags["Freq"])):
            tags["Freq"][i] = (tags["Freq"][i][0], tags["Freq"][i][1] + time_adj)
        for i in range(0, len(tags["Dur"])):
            tags["Dur"][i] = (tags["Dur"][i][0], tags["Dur"][i][1] + time_adj)
        freq_tags = tags["Freq"]
        dur_tags = tags["Dur"]
        nb_tags = []
        for tag in freq_tags:
            frame = int(tag[1] * fps)
            tag_value = freq_tag_lut[tag[0]]
            event_ticker[tag_value] = event_ticker[tag_value] + 1
            print(f"Video: {video_file} | Event: {tag[0]} | Event Code: {tag_value} | Frame Index: {frame} | Timestamp: {int(tag[1] / 60)}:{int(tag[1] % 60)}")
            frame_tags.append((tag[0], tag_value, frame, tag[1], fps))
            total_tags = total_tags + 1
        for tag in dur_tags:
            frame = int(tag[1] * fps)
            tag_value = dur_tag_lut[tag[0]]
            event_ticker[tag_value] = event_ticker[tag_value] + 1
            print(f"Video: {video_file} | Event: {tag[0]} | Event Code: {tag_value} | Frame Index: {frame} | Timestamp: {int(tag[1] / 60)}:{int(tag[1] % 60)}")
            frame_tags.append((tag[0], tag_value, frame, tag[1], fps))
            total_tags = total_tags + 1
        for i in range(1, len(frame_tags)):
            event_ticker[all_tags["no-behavior"]] = event_ticker[all_tags["no-behavior"]] + 1
            frame = int(frame_tags[i - 1][2] + ((frame_tags[i][2] - frame_tags[i - 1][2]) / 2))
            video_time = frame / fps
            nb_tags.append(('no-behavior', nb_tag, frame, video_time, fps))
        if len(frame_tags) % 2:
            event_ticker[all_tags["no-behavior"]] = event_ticker[all_tags["no-behavior"]] + 1
            frame = int(frame_tags[-1][2] + ((seconds - frame_tags[-1][2]) / 2))
            video_time = frame / fps
            nb_tags.append(('no-behavior', nb_tag, frame, video_time, fps))
        frame_tags.extend(nb_tags)
        # Put the events in order by frame
        frame_tags.sort(key=lambda y: y[2])
        with open(output_file, 'wb') as f:
            pickle.dump(frame_tags, f)
    print("Finished adjusting event times, found {0} tags...\n".format(total_tags))
    for tag in all_tags:
        print(f"{tag}: {event_ticker[all_tags[tag]]}")
