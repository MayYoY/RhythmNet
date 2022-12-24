"""
处理 VIPL-HR 数据集
计算 STMap 与 average HR, 保存至 cache_path; 对应 ground truth 保存至 gt_cache
"""

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import os
import glob
import re
import random

from tqdm.auto import tqdm
from scipy import io
from torch.utils import data

from . import utils


class STMapPreprocess:
    def __init__(self, config):
        self.config = config
        # [p1, p2, ..., pn]
        self.dirs = glob.glob(self.config.input_path + os.sep + "data" + os.sep + "*")
        self.folds = self.get_fold()

    def get_fold(self):
        ret = {}
        files = glob.glob(self.config.input_path + os.sep + "fold" + os.sep + "*.mat")
        for f in files:
            i = int(f[-5])
            temp = io.loadmat(f)[f"fold{i}"][0]  # subject_idx of fold(i + 1)
            for idx in temp:
                ret[idx] = i
        return ret

    def read_process(self):
        file_num = len(self.dirs)
        progress_bar = tqdm(list(range(file_num)))
        csv_info = {"input_files": [], "gt_files": [], "total_HR": [],
                    "Fs": [], "fold": [], "task": [], "source": []}
        for pi in self.dirs:  # i_th subject
            p_idx = re.findall("p(\d\d?\d?)", pi)[0]  # p1, p2, ..., p10, p100, p101
            tasks = glob.glob(pi + os.sep + "*")  # [v1(, v1-2), v2, ...]
            for ti in tasks:
                if not re.findall("v(\d-\d)", ti):
                    t_idx = re.findall("v(\d)", ti)[0]
                else:
                    t_idx = re.findall("v(\d-\d)", ti)[0]  # v1-2
                sources = glob.glob(ti + os.sep + "*")  # [source1, source2, ...]
                for si in sources:
                    s_idx = re.findall("source(\d)", si)[0]  # source_i
                    filename = f"p{p_idx}_v{t_idx}_source{s_idx}"  # 命名信息
                    # T x H x W x C
                    frames, Fs, end_time = self.read_video(si)
                    hrs, total_hr = self.read_hrs(len(frames), si, end_time)
                    if round(len(hrs) * Fs) < len(frames):  # 视频长度 > HR 序列长度, 需要截取
                        frames = frames[: round(len(hrs) * Fs)]
                    # 丢弃长度不足的视频
                    if len(frames) < self.config.CHUNK_LENGTH:
                        continue
                    # 计算 STMap 以及对应时间段的平均 HR
                    STMaps, average_hrs = utils.get_STMap(frames, hrs, Fs=Fs,
                                                          chunk_length=self.config.CHUNK_LENGTH)
                    if not len(STMaps):
                        continue  # 保险起见
                    # 保存数据
                    os.makedirs(self.config.cache_path, exist_ok=True)
                    input_file = self.config.cache_path + os.sep + filename + "_input.npy"
                    gt_file = self.config.cache_path + os.sep + filename + "_gt.npy"
                    np.save(input_file, STMaps)
                    np.save(gt_file, average_hrs)
                    # 视频的平均 HR
                    csv_info["input_files"].append(input_file)
                    csv_info["gt_files"].append(gt_file)
                    csv_info["total_HR"].append(total_hr)
                    csv_info["fold"].append(self.folds[int(p_idx)])
                    csv_info["Fs"].append(Fs)
                    csv_info["task"].append(int(t_idx[0]))
                    csv_info["source"].append(int(s_idx))
            progress_bar.update(1)

        csv_info = pd.DataFrame(csv_info)
        csv_info.to_csv(self.config.record_path, index=False)

    def read_video(self, data_path):
        """读取视频, 人脸检测, 保存帧; 返回帧下标, 帧率"""
        vid = cv.VideoCapture(data_path + os.sep + "video.avi")
        vid.set(cv.CAP_PROP_POS_MSEC, 0)  # 设置从 0 开始读取
        ret, frame = vid.read()
        frames = list()
        while ret:
            # BGR -> YUV, for STMap
            frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2YUV)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0
            frames.append(frame)
            ret, frame = vid.read()

        frames = np.asarray(frames)

        # 人脸检测并截取, list
        frames = utils.resize(frames, self.config.DYNAMIC_DETECTION,
                              self.config.DYNAMIC_DETECTION_FREQUENCY,
                              self.config.W, self.config.H,
                              self.config.LARGE_FACE_BOX,
                              self.config.CROP_FACE,
                              self.config.LARGE_BOX_COEF)
        # for return
        if data_path[-1] == "2":
            Fs = 30
            # 计算视频结束时间
            end_time = round(len(frames) / Fs)
        else:
            time_record = np.loadtxt(data_path + os.sep + "time.txt")
            bound = min(len(time_record) - 1, len(frames) - 1)
            Fs = len(frames) * 1000 / time_record[bound]
            end_time = round(time_record[bound] / 1000)
        return frames, Fs, end_time

    @staticmethod
    def read_hrs(T, data_path, end_time):
        # The HR and SpO2 of the subject is recorded every second
        # 根据结束时间截取 HR 序列
        hrs = pd.read_csv(data_path + os.sep + "gt_HR.csv")["HR"].values[: end_time]
        return hrs, hrs.mean()


class VIPL_HR(data.Dataset):
    def __init__(self, config):
        super(VIPL_HR, self).__init__()
        record = pd.read_csv(config.record)
        self.config = config
        self.input_files = []
        self.gt_files = []
        self.average_hrs = []
        self.Fs = []
        for i in range(len(record)):
            if self.isValid(record, i):
                self.input_files.append(record.loc[i, "input_files"])
                self.gt_files.append(record.loc[i, "gt_files"])
                self.average_hrs.append(record.loc[i, "total_HR"])
                self.Fs.append(record.loc[i, "Fs"])

    def isValid(self, record, idx):
        flag = True
        if self.config.folds:
            flag &= record.loc[idx, "fold"] in self.config.folds
        if self.config.tasks:
            flag &= record.loc[idx, "task"] in self.config.tasks
        if self.config.sources:
            flag &= record.loc[idx, "source"] in self.config.sources
        return flag

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # clip_num x chunk_length (T1) x roi_num (25) x C (YUV, 3)
        x_path = self.input_files[idx]
        x = torch.from_numpy(np.load(x_path))
        # 每段 clip 的平均 HR
        y_path = self.gt_files[idx]
        y = torch.from_numpy(np.load(y_path))  # T,
        # 整个视频的平均 HR
        average_hr = torch.tensor([self.average_hrs[idx]])
        # 帧率
        Fs = torch.tensor([self.Fs[idx]])
        # torchvision.transforms.RandomHorizontalFlip, ToTensor
        if self.config.mask:
            x = utils.randomMask(x)
        x = x.permute(0, 3, 1, 2)  # Nclip x T1 x Nr x C -> Nclip x C x T1 x Nr

        ret = {"stmaps": x.float(),
               "hrs": y.float(),
               "average_hr": average_hr.float(),
               "Fs": Fs.float()}
        return ret


def collate_fn(batch):
    return batch
