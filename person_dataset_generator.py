from tkinter import *
from tkinter import messagebox
import os.path
import imageio
import pandas as pd
import torch
import cv2
import numpy as np
import os
from dataloader_utils import create_pandas_dataset
import threading
from PIL import Image, ImageTk
import time
import pathlib


class VideoThread:
    def __init__(self, videos, labels, model, path_var, p_container, c_container, fps=8):
        self.fps = fps
        self.videos = videos
        self.model = model
        self.labels = labels
        self.path_var = path_var
        self.p_container = p_container
        self.c_container = c_container
        self.start_thread = False
        self.current_frame = None
        self.next_frame = False
        self.largest_bb_area = []
        self.p_selection = 0
        self.p_accepted = 0
        self.c_selection = 0
        self.c_accepted = 0
        self.largest_bb_img = []
        # self.bb = []
        if os.path.exists('datasets/stills'):
            _, _, f = next(os.walk(r'datasets/stills'))
            self.p_video_count = int(len(f) / 16 / 2)
            print(f"Starting video is {self.p_video_count}")
        else:
            self.p_video_count = 0
        if os.path.exists('datasets/clinician_videos'):
            _, _, f = next(os.walk(r'datasets/clinician_videos'))
            self.c_video_count = len(f)
        else:
            self.c_video_count = 0
        self.accepted_p_stills = []
        self.accepted_c_stills = []

    def video_thread(self):
        while True:
            if self.start_thread:
                self.video_cropping_thread()
            else:
                time.sleep(0.5)

    def start_videos(self):
        self.start_thread = True

    def get_next_frame(self):
        self.next_frame = True
        patient_accept_button.config(state='active')
        clinician_accept_button.config(state='active')

    def accept_p_selection(self):
        self.p_accepted += 1
        if self.p_selection != -1:
            # self.accepted_p_stills.append(self.largest_bb_img[self.p_selection][0])
            # imageio.imwrite(f'datasets/patient_stills/{self.p_video_count}_{len(self.accepted_p_stills)}_patient.png', self.last_image)
            # with open(f'datasets/patient_stills/{self.p_video_count}_{len(self.accepted_p_stills)}_patient.txt', 'w') as f:
            #     f.write(' '.join(['0'] + self.largest_bb_img[self.p_selection][1]))
            patient_accept_button.config(state='disabled')

    def accept_c_selection(self):
        self.c_accepted += 1
        if self.c_selection != -1:
            # self.accepted_c_stills.append(self.largest_bb_img[self.c_selection][0])
            # imageio.imwrite(f'datasets/clinician_stills/{self.c_video_count}_{len(self.accepted_c_stills)}_clinician.png',
            #                 self.last_image)
            # with open(f'datasets/clinician_stills/{self.c_video_count}_{len(self.accepted_c_stills)}_clinician.txt', 'w') as f:
            #     f.writelines(' '.join(['1'] + self.largest_bb_img[self.c_selection][1]))
            clinician_accept_button.config(state='disabled')

    def accept_selection(self):
        self.accept_c_selection()
        self.accept_p_selection()
        imageio.imwrite(f'datasets/stills/{self.p_video_count}_{self.p_accepted}.png', self.last_image)
        with open(f'datasets/stills/{self.p_video_count}_{self.c_accepted}.txt', 'w') as f:
            for bb in self.largest_bb_img:
                f.writelines(' '.join([bb[2]] + bb[1]))
                f.write('\n')
            # if self.p_selection != -1:
            #     f.writelines(' '.join(['1'] + self.largest_bb_img[self.c_selection][1]))

    def get_next_p_selection(self):
        print(f"Getting next patient selection {self.p_selection}")
        if self.p_selection == len(self.largest_bb_img) - 1:
            return
        self.p_selection += 1
        if self.p_selection == len(self.largest_bb_img):
            self.p_selection = 0
        self.load_p_selection()

    def get_next_c_selection(self):
        print(f"Getting next clinician selection {self.c_selection}")
        if self.c_selection == len(self.largest_bb_img) - 1:
            return
        self.c_selection += 1
        if self.c_selection == len(self.largest_bb_img):
            self.c_selection = 0
        self.load_c_selection()

    def get_prev_c_selection(self):
        print(f"Getting previous clinician selection {self.c_selection}")
        if self.c_selection == 0:
            return
        self.c_selection -= 1
        if self.c_selection == -1:
            self.c_selection = len(self.largest_bb_img) - 1
        self.load_c_selection()

    def get_prev_p_selection(self):
        print(f"Getting previous patient selection {self.p_selection}")
        if self.p_selection == 0:
            return
        self.p_selection -= 1
        if self.p_selection == -1:
            self.p_selection = len(self.largest_bb_img) - 1
        self.load_p_selection()

    def load_p_selection(self):
        if self.p_selection != -1:
            self.p_container.load_image(self.largest_bb_img[self.p_selection][0])

    def load_c_selection(self):
        if self.c_selection != -1:
            self.c_container.load_image(self.largest_bb_img[self.c_selection][0])

    def video_cropping_thread(self):
        for i in range(self.p_video_count, len(self.videos)):
            self.path_var.set(pathlib.Path(str(self.videos[i])).name)
            _ = self.load_video(self.videos[i])
            label = self.labels[i]
            self.p_selection = 0
            self.c_selection = 0
            if self.accepted_p_stills:
                writer = imageio.get_writer(f'datasets/patient_videos/{self.p_video_count}_{str(label)}_patient.mp4',
                                            fps=self.fps)
                height, width = 0, 0
                for still in self.accepted_p_stills:
                    if still.shape[0] > width:
                        width = still.shape[0]
                    if still.shape[1] > height:
                        height = still.shape[1]
                for still in self.accepted_p_stills:
                    writer.append_data(cv2.resize(still, (height, width)))
                writer.close()
            if self.accepted_c_stills:
                writer = imageio.get_writer(f'datasets/clinician_videos/{self.c_video_count}_clinician.mp4',
                                            fps=self.fps)
                height, width = 0, 0
                for still in self.accepted_c_stills:
                    if still.shape[0] > width:
                        width = still.shape[0]
                    if still.shape[1] > height:
                        height = still.shape[1]
                for still in self.accepted_c_stills:
                    writer.append_data(cv2.resize(still, (height, width)))
                writer.close()
            self.accepted_p_stills = []
            self.accepted_c_stills = []
            self.p_video_count += 1
            self.c_video_count += 1
            self.p_accepted = 0
            self.c_accepted = 0

    def load_video(self, path, max_frames=0):
        frames = []
        self.last_image = None
        frame_data = imageio.get_reader(path)
        for frame in frame_data.iter_data():
            try:
                if self.last_image is not None:
                    if (self.last_image == frame).all():
                        continue
                if os.path.exists(f'datasets/stills/{self.p_video_count}_{self.c_accepted+1}.txt'):
                    self.c_accepted += 1
                    self.p_accepted += 1
                    continue
                self.last_image = frame
                inf_frame = frame[:, :, [0, 1, 2]]
                img_result = self.model(inf_frame)
                if len(img_result.names) == 2:
                    custom = True
                    classes = [0, 1]
                else:
                    custom = False
                    classes = [0]
                p_select = -1
                c_select = -1
                for xyxy, xywh, preds in zip(img_result.xyxy[0], img_result.xywhn[0], img_result.pred[0]):
                    if preds[5] in classes:
                        x1 = int(xyxy[0].item())
                        y1 = int(xyxy[1].item())
                        x2 = int(xyxy[2].item())
                        y2 = int(xyxy[3].item())
                        area = (int(xyxy[2].item()) - int(xyxy[0].item())) * (int(xyxy[3].item()) - int(xyxy[1].item()))
                        cropped_img = frame[y1:y2, x1:x2]
                        self.largest_bb_area.append(area)
                        self.largest_bb_img.append((cropped_img, [str(float(xywh[0])), str(float(xywh[1])),
                                                                  str(float(xywh[2])), str(float(xywh[3]))],
                                                    str(int(preds[5].item()))))
                    # if custom:
                    #     if preds[5] == 0:
                    #         p_select = len(self.largest_bb_img) - 1
                    #     elif preds[5] == 1:
                    #         c_select = len(self.largest_bb_img) - 1
                frames.append(frame)
                if custom:
                    self.p_selection = p_select
                    self.c_selection = c_select
                else:
                    self.p_selection = 0
                    self.c_selection = 0
                self.load_p_selection()
                self.load_c_selection()

                if len(frames) == max_frames:
                    break
                while not self.next_frame:
                    time.sleep(0.25)
                self.accept_selection()
                self.next_frame = False
                self.largest_bb_area = []
                self.largest_bb_img = []
            except Exception as e:
                messagebox.showerror("Exception", f"Exception encountered:\n{str(e)}")
        return np.array(frames)


def get_pandas_dataset():
    # Parse dataset into train and test sets
    create_pandas_dataset(
        r'')
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
    return train_df, test_df


class ImageContainer:
    def __init__(self, label, init_image=None, image_size=(300, 500)):
        self.label = label
        self.size = image_size
        if init_image:
            self.load_image(cv2.imread(init_image))

    def load_image(self, img):
        self.image = (ImageTk.PhotoImage(Image.fromarray(img).resize(self.size)))
        self.label.config(image=self.image)
        self.label.image = self.image


def center(toplevel, y_offset=-20):
    toplevel.update_idletasks()

    # Tkinter way to find the screen resolution
    screen_width = toplevel.winfo_screenwidth()
    screen_height = toplevel.winfo_screenheight()

    size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
    x = screen_width / 2 - size[0] / 2
    y = screen_height / 2 - size[1] / 2
    y += y_offset
    toplevel.geometry("+%d+%d" % (x, y))


root = Tk()
root.geometry("700x700")
root.title("Patient Clinician Separator")

train_videos_df, test_videos_df = get_pandas_dataset()
train_videos = train_videos_df["video_name"].values.tolist()
train_labels = train_videos_df["tag"].values.tolist()
test_videos = test_videos_df["video_name"].values.tolist()
test_labels = test_videos_df["tag"].values.tolist()
all_videos = [*train_videos, *test_videos]
all_labels = [*train_labels, *test_labels]

video_label_var = StringVar(root, value="Press start button to load a video")
video_label = Label(root, textvariable=video_label_var, font=('Purisa', 12))
video_label.place(x=350, y=10, anchor=N)

question_mark_img = 'question_mark.png'
clinician_text = Label(root, text="Clinician", font=('Purisa', 12))
clinician_text.place(x=525, y=75, anchor=S)
clinician_label = Label(root)
clinician_label.place(x=525, y=75, anchor=N)
clinician_container = ImageContainer(clinician_label, question_mark_img)

patient_text = Label(root, text='Patient', font=('Purisa', 12))
patient_text.place(x=175, y=75, anchor=S)
patient_label = Label(root)
patient_label.place(x=175, y=75, anchor=N)
patient_container = ImageContainer(patient_label, question_mark_img)

detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True)

video_object = VideoThread(all_videos, all_labels, detection_model, video_label_var, patient_container,
                           clinician_container)

patient_next_button = Button(root, text="Next", font=('Purisa', 12), command=video_object.get_next_p_selection)
patient_next_button.place(x=325, y=575, anchor=NE, width=100, height=30)

patient_prev_button = Button(root, text="Prev", font=('Purisa', 12), command=video_object.get_prev_p_selection)
patient_prev_button.place(x=25, y=575, anchor=NW, width=100, height=30)

patient_accept_button = Button(root, text="Accept", font=('Purisa', 12), command=video_object.accept_p_selection)
patient_accept_button.place(x=175, y=575, anchor=N, width=100, height=30)

clinician_next_button = Button(root, text="Next", font=('Purisa', 12), command=video_object.get_next_c_selection)
clinician_next_button.place(x=675, y=575, anchor=NE, width=100, height=30)

clinician_prev_button = Button(root, text="Prev", font=('Purisa', 12), command=video_object.get_prev_c_selection)
clinician_prev_button.place(x=375, y=575, anchor=NW, width=100, height=30)

clinician_accept_button = Button(root, text="Accept", font=('Purisa', 12), command=video_object.accept_c_selection)
clinician_accept_button.place(x=525, y=575, anchor=N, width=100, height=30)

start_button = Button(root, text="Start", bg='green', font=('Purisa', 12), command=video_object.start_videos)
start_button.place(x=350, y=675, anchor=SE, width=100, height=30)

continue_button = Button(root, text="Continue", font=('Purisa', 12), command=video_object.get_next_frame)
continue_button.place(x=350, y=675, anchor=SW, width=100, height=30)

if not os.path.exists('datasets'):
    os.mkdir('datasets')

clinician_videos_dir = 'datasets/clinician_videos'
if not os.path.exists(clinician_videos_dir):
    os.mkdir(clinician_videos_dir)

patient_videos_dir = 'datasets/patient_videos'
if not os.path.exists(patient_videos_dir):
    os.mkdir(patient_videos_dir)

clinician_stills_dir = 'datasets/clinician_stills'
if not os.path.exists(clinician_stills_dir):
    os.mkdir(clinician_stills_dir)

patient_stills_dir = 'datasets/patient_stills'
if not os.path.exists(patient_stills_dir):
    os.mkdir(patient_stills_dir)

stills_dir = 'datasets/stills'
if not os.path.exists(stills_dir):
    os.mkdir(stills_dir)

crop_thread = threading.Thread(target=video_object.video_thread)
crop_thread.daemon = True
crop_thread.start()

center(root)
root.mainloop()
