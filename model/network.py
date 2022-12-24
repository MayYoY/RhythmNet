import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models


class RhythmNet(nn.Module):
    def __init__(self):
        super(RhythmNet, self).__init__()

        # We choose ResNet-18 as the backbone convolutional layers
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]  # 剔除最后的 fc
        self.resnet18 = nn.Sequential(*modules)
        # The output of the network is a single HR value
        # regressed by a fully connected layer
        self.resnet_regress = nn.Linear(1000, 1)

        self.convert = nn.Linear(512, 1000)  # 用于连接 resnet, gru

        # the features extracted from the backbone CNN are fed to a one-layer GRU
        self.rnn = nn.GRU(input_size=1000, hidden_size=1000, num_layers=1)
        # The output of GRU is fed into a fully connected layer
        # to regress the HR values for individual video clips
        self.gru_regress = nn.Linear(1000, 1)

    def forward(self, stmaps, Fs):
        """
        :param stmaps: clip_num x C x T x roi_num
        :param Fs: frame rate, for normalize the prediction
        :return:
        """
        res_pred_per_clip = []
        gru_input_per_clip = []
        gru_pred_per_clip = []

        stmaps = stmaps.unsqueeze(0)  # 1 x clip_num x C x T x roi_num
        for t in range(stmaps.shape[1]):
            # 1 x C x T x roi_num -> 1 x hidden_size x 1 x 1
            res_hidden = self.resnet18(stmaps[:, t, :, :, :])
            res_hidden = res_hidden.view(1, -1)  # 1 x hidden_size
            # 1 x hidden_size1
            gru_input = self.convert(res_hidden)
            # Save CNN features per clip for the GRU
            gru_input_per_clip.append(gru_input.squeeze(0))

            # resnet 的预测结果
            single_pred = self.resnet_regress(gru_input)
            # TODO: check whether to normalize
            single_pred = single_pred * Fs
            res_pred_per_clip.append(single_pred.squeeze(0))  # 1,

        # resnet 预测结果, 需要计算 L1 loss
        resnet_pred = torch.stack(res_pred_per_clip, dim=0).flatten()  # T,

        # T x 1 x hidden_size1
        gru_input = torch.stack(gru_input_per_clip, dim=0).unsqueeze(1)
        gru_hidden, _ = self.rnn(gru_input)  # T x 1 x hidden_size1
        for i in range(gru_hidden.shape[0]):
            # 1 x hidden_size1 -> 1 x 1
            per_pred = self.gru_regress(gru_hidden[i, :, :])
            gru_pred_per_clip.append(per_pred.squeeze(0) * Fs)  # normalize !!!

        # gru 预测结果, 需要计算 L_smooth
        gru_pred = torch.stack(gru_pred_per_clip, dim=0).flatten()  # T,
        # For each face video, the average of all the predicted HRs for individual
        # video clips are computed as the final HR result.
        # TODO: resnet 平均 or gru 平均
        # resnet_pred 的平均作为预测
        average_pred = resnet_pred.mean().reshape(-1)
        # return resnet_pred, average_pred
        return resnet_pred, gru_pred, average_pred
