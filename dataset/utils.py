import numpy as np
import random
import cv2 as cv
from mtcnn import MTCNN
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

import torch
from torch.utils import data


def resize(frames, dynamic_det, det_length,
           w, h, larger_box, crop_face, larger_box_size):
    """
    :param frames:
    :param dynamic_det: 是否动态检测
    :param det_length: the interval of dynamic detection
    :param w:
    :param h:
    :param larger_box: whether to enlarge the detected region.
    :param crop_face:  whether to crop the frames.
    :param larger_box_size:
    """
    if dynamic_det:
        det_num = ceil(len(frames) / det_length)  # 检测次数
    else:
        det_num = 1
    face_region = []
    # 获取人脸区域
    detector = MTCNN()
    for idx in range(det_num):
        if crop_face:
            face_region.append(facial_detection(detector, frames[det_length * idx],
                                                larger_box, larger_box_size))
        else:  # 不截取
            face_region.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region, dtype='int')
    # resize_frames = np.zeros((frames.shape[0], h, w, 3))  # T x H x W x 3
    resize_frames = []

    # 截取人脸并 resize
    for i in range(len(frames)):
        frame = frames[i]
        # 选定人脸区域
        if dynamic_det:
            reference_index = i // det_length
        else:
            reference_index = 0
        if crop_face:
            face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[3], frame.shape[0]),
                          max(face_region[0], 0):min(face_region[2], frame.shape[1])]
        if w > 0 and h > 0:
            resize_frames.append(cv.resize(frame, (w + 4, h + 4),
                                           interpolation=cv.INTER_CUBIC)[2: w + 2, 2: h + 2, :])
        else:
            resize_frames.append(frame)
    if w > 0 and h > 0:
        return np.asarray(resize_frames)
    else:  # list
        return resize_frames


def facial_detection(detector, frame, larger_box=False, larger_box_size=1.0):
    """
    利用 MTCNN 检测人脸区域
    :param detector:
    :param frame:
    :param larger_box: 是否放大 bbox, 处理运动情况
    :param larger_box_size:
    """
    face_zone = detector.detect_faces(frame)
    if len(face_zone) < 1:
        print("Warning: No Face Detected!")
        return [0, 0, frame.shape[0], frame.shape[1]]
    if len(face_zone) >= 2:
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    result = face_zone[0]['box']
    h = result[3]
    w = result[2]
    result[2] += result[0]
    result[3] += result[1]
    if larger_box:
        print("Larger Bounding Box")
        result[0] = round(max(0, result[0] + (1. - larger_box_size) / 2 * w))
        result[1] = round(max(0, result[1] + (1. - larger_box_size) / 2 * h))
        result[2] = round(max(0, result[0] + (1. + larger_box_size) / 2 * w))
        result[3] = round(max(0, result[1] + (1. + larger_box_size) / 2 * h))
    return result


def chunk(frames, gts, chunk_length, chunk_stride=-1):
    """Chunks the data into clips."""
    if chunk_stride < 0:
        chunk_stride = chunk_length
    # clip_num = (frames.shape[0] - chunk_length + chunk_stride) // chunk_stride
    frames_clips = [frames[i: i + chunk_length]
                    for i in range(0, len(frames) - chunk_length + 1, chunk_stride)]
    bvps_clips = [gts[i: i + chunk_length]
                  for i in range(0, len(gts) - chunk_length + 1, chunk_stride)]
    return np.array(frames_clips), np.array(bvps_clips)


def get_blocks(frame, h_num=5, w_num=5):
    h, w, _ = frame.shape  # 61, 59
    h_len = h // h_num  # 12
    w_len = w // w_num  # 11
    ret = []
    h_idx = [i * h_len for i in range(0, h_num)]  # 0, 12, 24, 36, 48
    w_idx = [i * w_len for i in range(0, w_num)]
    for i in h_idx:
        for j in w_idx:
            ret.append(frame[i: i + h_len, j: j + w_len, :])  # h_len x w_len x 3
    return ret


def get_STMap(frames, hrs, Fs, chunk_length=300, roi_num=25) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param frames: T x H x W x C or list[H x W x C], len = T
    :param hrs:
    :param Fs:
    :param chunk_length:
    :param roi_num: 划分块数, 5 * 5 = 25
    :return: clip_num x chunk_length (T1) x roi_num (25) x C (YUV, 3)
    """
    chunk_stride = round(Fs / 2)  # 0.5s
    clip_num = (len(frames) - chunk_length + chunk_stride) // chunk_stride
    STMaps = []
    average_hrs = []
    scaler = MinMaxScaler()
    for i in range(0, len(frames) - chunk_length + 1, chunk_stride):
        temp = np.zeros((chunk_length, roi_num, 3))  # T1 x 25 x 3
        for j, frame in enumerate(frames[i: i + chunk_length]):
            blocks = get_blocks(frame)
            for k, block in enumerate(blocks):
                # temp[j, k, :] = block.mean(axis=0).mean(axis=0)
                temp[j, k, 0] = block[:, :, 0].mean()
                temp[j, k, 1] = block[:, :, 1].mean()
                temp[j, k, 2] = block[:, :, 2].mean()
        # In order to make the best use of the HR signals,
        # a min-max normalization is applied to each temporal signal,
        # and the values of the temporal series are scaled into [0, 255]
        # 首先用 minmax_scaler 缩放至 [0, 1], 再 * 255; **在时间维进行**
        for j in range(roi_num):
            scaled_c0 = scaler.fit_transform(temp[:, j, 0].reshape(-1, 1))
            temp[:, j, 0] = (scaled_c0 * 255.).reshape(-1).astype(np.uint8)
            scaled_c1 = scaler.fit_transform(temp[:, j, 1].reshape(-1, 1))
            temp[:, j, 1] = (scaled_c1 * 255.).reshape(-1).astype(np.uint8)
            scaled_c2 = scaler.fit_transform(temp[:, j, 2].reshape(-1, 1))
            temp[:, j, 2] = (scaled_c2 * 255.).reshape(-1).astype(np.uint8)
        STMaps.append(temp)
        # 此段 STMap 对应时间段的平均 HR
        average_hrs.append(hrs[int(i // Fs): min(len(hrs) - 1, int((i + chunk_length) // Fs))].mean())
    assert len(STMaps) == clip_num, "Number of Clips Error, Please check your code!"
    STMaps = np.asarray(STMaps)
    average_hrs = np.asarray(average_hrs)
    return STMaps, average_hrs


def randomMask(x):
    """
    During the training phase, half of the generated spatial-temporal maps
    were randomly masked, and the mask length varies from 10
    frames to 30 frames.
    :param x: clip_num x chunk_length (T1) x roi_num (25) x C
    :return:
    """
    for stmap in x:
        if random.random() < 0.5:
            continue
        mask_len = random.randint(10, 30)
        idx = random.randint(0, len(stmap) - 30)
        stmap[idx: idx + mask_len, :, :] = 0
    return x


def normalize_frame(frame):
    """[0, 255] -> [-1, 1]"""
    return (frame - 127.5) / 128


def standardize(data):
    """
    :param data:
    :return: (x - \mu) / \sigma
    """
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data
