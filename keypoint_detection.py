import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
import os
import pathlib
from keypoint_utils import filter_persons, draw_keypoints


def load_detectron():
    # Load Detectron2 model
    cfg = get_cfg()
    # load the pre trained model from Detectron2 model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    # set confidence threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # load model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    # create the predictor for pose estimation using the config
    predictor = DefaultPredictor(cfg)
    return predictor


def get_keypoints(video_path, predictor, write_video=False, write_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
        return []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if write_video:
        out = cv2.VideoWriter(os.path.join(write_path, pathlib.Path(video_path).stem + "_detectron.mp4"),
                              cv2.VideoWriter_fourcc(*'mp4v'), 8,
                              (frame_width, frame_height))
        out2 = cv2.VideoWriter(os.path.join(write_path, pathlib.Path(video_path).stem + "_detectron_skele.mp4"),
                               cv2.VideoWriter_fourcc(*'mp4v'), 8,
                               (frame_width, frame_height))
    frame_count = 0
    keypoints = []
    while cap.isOpened():
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            keypoint_tensor = torch.empty(0)
            image = frame.copy()
            outputs = predictor(image)
            persons, _ = filter_persons(outputs)
            for person in persons:
                if write_video:
                    image_blank = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    draw_keypoints(persons[person], image)
                    draw_keypoints(persons[person], image_blank)
                    out.write(image)
                    out2.write(image_blank)
                keypoint_tensor = torch.cat((keypoint_tensor, persons[person]))
            if len(persons) < 4:
                for i in range(0, 4 - len(persons)):
                    keypoint_tensor = torch.cat((keypoint_tensor, torch.zeros((17, 3))))
            frame_count += 1
            if not frame_count % 100:
                print("Frame:", frame_count)
            keypoints.append(torch.flatten(keypoint_tensor))
        else:
            break
    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
