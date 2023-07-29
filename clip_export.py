import glob
import json
import os
import pathlib
import traceback
from json import JSONDecodeError

import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip


frequency_path = None
duration_path = None


def get_video_dataset(dataset_files):
    video_files, history_files, ksf, subjects = [], [], [], []
    for file in dataset_files:
        video_files.append(file[:-4] + "mp4")
        with open(file, 'r') as f:
            try:
                json_file = json.load(f)
            except JSONDecodeError:
                print("\tFile was empty, continuing...")
                continue
        subjects.append(pathlib.Path(file).parts[6])
        history_files.append(json_file['Event History'])
        ksf.append(json_file['KSF'])
    return video_files, history_files, ksf, subjects


def save_clips(vids, labs, parts, freq_classes, dur_classes, ksfs):
    no_pbx_videos = 0
    for vid, lab, part, ksf in zip(vids, labs, parts, ksfs):
        if not os.path.exists(rf'{frequency_path}{part}'):
            os.mkdir(rf'{frequency_path}{part}')
        if not os.path.exists(rf'{duration_path}{part}'):
            os.mkdir(rf'{duration_path}{part}')
        with VideoFileClip(vid, audio=False, fps_source='fps') as full_video:
            previous_l = None
            for l in lab:
                clip_fps = full_video.fps
                if l[0] in freq_classes:
                    l[0] = l[0].replace('/', '-')
                    class_dir = rf'{frequency_path}{part}\{l[0]}'
                    clip_start = float(l[2] - clip_fps) / float(clip_fps)
                    clip_end = float(l[2] + clip_fps) / float(clip_fps)
                elif l[0] in dur_classes:
                    l[0] = l[0].replace('/', '-')
                    class_dir = rf'{duration_path}{part}\{l[0]}'
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

                if previous_l is not None:
                    if l[2] - previous_l[2] > 160:
                        clip_middle = int((l[2] + previous_l[2]) / 2)
                        clip_start = float(clip_middle - clip_fps) / float(clip_fps)
                        clip_end = float(clip_middle + clip_fps) / float(clip_fps)
                        class_dir = rf'{frequency_path}{part}\no pbx'
                        if not os.path.exists(class_dir):
                            os.mkdir(class_dir)
                        clip_name = f"{len(list(os.listdir(class_dir)))}.mp4"
                        try:
                            full_video.subclip(t_start=clip_start, t_end=clip_end).write_videofile(
                                os.path.join(class_dir, clip_name), logger=None)
                            no_pbx_videos += 1
                        except UserWarning:
                            print(f"{vid}: Can't export {clip_start} to {clip_end}")
                previous_l = l
    print(f"Exported {no_pbx_videos} no pbx videos")


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


freq_classes = [
    'hitting', 'kicking', 'pushing', 'grab_scratch', 'head butting', 'hair pulling', 'biting', 'choking',
    'sib-head bang', 'sib-head hit', 'sib-self-hit', 'sib-biting', 'sib-eye poke', 'sib-body slam',
    'sib-hair pull', 'sib-choking', 'sib-pinch_scratch', 'throwing object', 'kick_hit object', 'flip furniture',
    'flop', 'no pbx'
]
dur_classes = [
    # 'st- rock', 'st-hand flap', 'st-touch/tap', 'st-head swing', 'stereovox'
]


if __name__ == '__main__':
    dataset_filepath = r""
    files = [f for f in glob.glob(f'{dataset_filepath}/**/**/**/**/**/Primary/*.json')
             if 'training' not in f
             and '1aMLMar102022.json' not in f
             and os.path.isfile(f[:-4] + "mp4")]

    cropped_filepath = r""
    # get_dataset_stats(files, "MMISB")

    video_files, labels, ksf, parts = get_video_dataset(files)
    save_clips(video_files, labels, parts, freq_classes, dur_classes, ksf)
