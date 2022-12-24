from typing import Any

import torch
import torch.nn as nn


class SmoothLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hr_t, hr_seq, T):
        """
        :param ctx: context object that can be used to stash information
                    for backward computation. You can cache arbitrary objects
                    for use in the backward pass using the ctx.save_for_backward method.
        :param hr_t:
        :param hr_seq:
        :param T:
        :return:
        """
        ctx.hr_seq = hr_seq
        ctx.hr_mean = hr_seq.mean()
        ctx.T = T
        ctx.save_for_backward(hr_t)
        # pdb.set_trace()
        # hr_t, hr_mean, T = input
        if hr_t > ctx.hr_mean:
            loss = hr_t - ctx.hr_mean
        else:
            loss = ctx.hr_mean - hr_t
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hr_t, = ctx.saved_tensors
        hr_seq = ctx.hr_seq
        output = torch.zeros(1).to(hr_seq.device)

        # create a list of hr_seq without hr_t

        for hr in hr_seq:
            if hr == hr_t:
                pass
            else:
                output = output + (1 / ctx.T) * torch.sign(ctx.hr_mean - hr)

        output = (1 / ctx.T - 1) * torch.sign(ctx.hr_mean - hr_t) + output

        return output, None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass


class TotalLoss(nn.Module):
    def __init__(self, lambda_val=100):
        super(TotalLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.lambda_val = lambda_val
        self.gru_outputs_considered = None
        self.smooth_loss = SmoothLoss()

    def forward(self, resnet_pred, gru_pred, average_pred, y, average_hr):
        # TODO: resnet or gru
        # resnet 平均
        l1_loss = self.l1_loss(resnet_pred, y)
        # 逐段计算 L_smooth
        smooth_loss = torch.zeros(1, device=y.device, requires_grad=True)
        # For the temporal relationship modelling,
        # **six adjacent** estimated HRs are used to compute the L_smooth
        T = int(gru_pred.shape[0] // 6)
        for i in range(T):
            pred_seq = gru_pred[i * 6: (i + 1) * 6].flatten()
            temp = torch.zeros(1, device=y.device, requires_grad=True)
            for hr_t in pred_seq:
                temp = temp + self.smooth_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                     pred_seq, 6)
            smooth_loss = smooth_loss + temp / 6
        # 仍有一段序列
        if gru_pred.shape[0] % 6:
            pred_seq = gru_pred[T * 6:].flatten()
            temp = torch.zeros(1, device=y.device, requires_grad=True)
            for hr_t in pred_seq:
                temp = temp + self.smooth_loss.apply(torch.autograd.Variable(hr_t, requires_grad=True),
                                                     pred_seq, len(pred_seq))
            smooth_loss = smooth_loss + temp / len(pred_seq)

        l1_loss = l1_loss + self.l1_loss(average_pred, average_hr)
        loss = l1_loss + self.lambda_val * smooth_loss
        return loss, l1_loss, smooth_loss
