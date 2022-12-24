from torch.utils import data
import numpy as np
import torch
import torch.nn as nn
import random
import os
from tqdm.auto import tqdm

from model import network, loss_function
from evaluate import metric
from dataset import vipl_hr
from configs import running


def xavier(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight)


def cross_validation(folds, path, train_config, test_config, methods=None):
    """
    受试者独立交叉验证
    :param folds:
    :param path: for saving models
    :param train_config:
    :param test_config:
    :param methods:
    :return:
    """
    if methods is None:
        methods = ["Mean", "Std", "MAE", "RMSE", "MAPE", "R"]
    result = metric.Accumulate(len(methods))
    for i in range(folds):
        print(f"================Fold{i + 1}================")
        train_config.folds = [j for j in range(1, 6) if j != i + 1]
        test_config.folds = [i + 1]
        train_set = vipl_hr.VIPL_HR(train_config)
        test_set = vipl_hr.VIPL_HR(test_config)
        train_iter = data.DataLoader(train_set, batch_size=train_config.batch_size,
                                     shuffle=True, collate_fn=vipl_hr.collate_fn)
        test_iter = data.DataLoader(test_set, batch_size=test_config.batch_size,
                                    shuffle=False, collate_fn=vipl_hr.collate_fn)
        # init and train
        net = network.RhythmNet()
        """net = nn.DataParallel(network.RhythmNet(), 
                              device_ids=train_config.device_ids)"""
        net.apply(xavier)
        net = net.to(train_config.device)
        # We use an Adam solver [37] with an initial learning rate of
        # 0.001, and set the maximum epoch number to 50.
        # TODO: 修改 lr
        lr = 0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        print("Training...")
        train(net, optimizer, scheduler, train_iter, train_config)
        torch.save(net.state_dict(), path + os.sep + f"avgResNet_lr{lr}_fold{i + 1}.pt")

        # test
        net = net.to(test_config.device)
        print(f"Evaluating...")
        # Mean, Std, MAE, RMSE, MAPE, R
        temp = test(net, test_iter, test_config)
        print(f"Mean: {temp[0]: .3f}\n"
              f"Std: {temp[1]: .3f}\n"
              f"MAE: {temp[2]: .3f}\n"
              f"RMSE: {temp[3]: .3f}\n"
              f"MAPE: {temp[4]: .3f}\n"
              f"R: {temp[5]: .3f}")
        result.update(val=temp, n=1)
    print(f"Cross Validation:\n"
          f"Mean: {result.acc[0] / result.cnt[0]: .3f}\n"
          f"Std: {result.acc[1] / result.cnt[1]: .3f}\n"
          f"MAE: {result.acc[2] / result.cnt[2]: .3f}\n"
          f"RMSE: {result.acc[3] / result.cnt[3]: .3f}\n"
          f"MAPE: {result.acc[4] / result.cnt[4]: .3f}\n"
          f"R: {result.acc[5] / result.cnt[5]: .3f}")


def train(net: nn.Module, optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          train_iter: data.DataLoader, train_config: running.TrainConfig):
    net.train()
    # TODO: 损失太大, 改小 lambda
    loss_fun = loss_function.TotalLoss(lambda_val=0.)
    train_loss = metric.Accumulate(3)  # for print
    progress_bar = tqdm(range(len(train_iter) * train_config.num_epochs))

    for epoch in range(train_config.num_epochs):
        train_loss.reset()
        print(f"Epoch {epoch + 1}...")
        for batch in train_iter:
            batch_loss = torch.zeros(1, requires_grad=True, device=train_config.device)
            batch_l1 = torch.zeros(1, device=train_config.device)
            batch_smooth = torch.zeros(1, device=train_config.device)
            for sample in batch:
                x = sample["stmaps"].to(train_config.device)  # T x C x chunk_len x roi_num
                y = sample["hrs"].to(train_config.device)  # T
                # average_hr = sample["average_hr"].to(train_config.device)  # 1
                Fs = sample["Fs"].to(train_config.device)
                # forward: T, T, 1
                resnet_pred, gru_pred, average_pred = net(x, Fs)
                loss, l1_loss, smooth_loss = loss_fun(resnet_pred, gru_pred,
                                                      average_pred, y, y.mean().reshape(-1))
                """optimizer.zero_grad()  # backward
                loss.backward()
                optimizer.step()"""
                # 累积
                batch_loss = batch_loss + loss
                batch_l1 = batch_l1 + l1_loss
                batch_smooth = batch_smooth + smooth_loss
            # backward
            batch_loss /= len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            # for print
            val = [batch_loss[0], batch_l1[0] / len(batch), batch_smooth[0] / len(batch)]
            train_loss.update(val=val, n=1)
            progress_bar.update(1)
        # scheduler.step()
        print(f"****************************************************\n"
              f"Epoch{epoch + 1}:\n"
              f"Total Loss: {train_loss.acc[0] / train_loss.cnt[0]: .3f}\n"
              f"L1 Loss: {train_loss.acc[1] / train_loss.cnt[1]: .3f}\n"
              f"Smooth Loss: {train_loss.acc[2] / train_loss.cnt[2]: .3f}\n"
              f"****************************************************")


def test(net: nn.Module, test_iter: data.DataLoader,
         test_config: running.TestConfig) -> list:
    net.eval()
    pred_phys = []
    label_phys = []
    progress_bar = tqdm(range(len(test_iter)))
    for batch in test_iter:
        for sample in batch:
            x = sample["stmaps"].to(test_config.device)
            y = sample["hrs"].to(test_config.device)
            # average_hr = sample["average_hr"].to(test_config.device)
            Fs = sample["Fs"].to(test_config.device)
            _, _, average_pred = net(x, Fs)
            # 预测心率与真实心率
            pred_phys.append(average_pred.cpu().detach().numpy().reshape(-1))
            label_phys.append(y.mean().cpu().detach().numpy().reshape(-1))
        progress_bar.update(1)
    pred_phys = np.asarray(pred_phys)
    label_phys = np.asarray(label_phys)

    return metric.cal_metric(pred_phys, label_phys)


def fixSeed(seed: int):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu
    # torch.backends.cudnn.deterministic = True  # 会大大降低速度
    torch.backends.cudnn.benchmark = True  # False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  # 增加运行效率，默认就是True
    torch.manual_seed(seed)


# TODO: 修改预测方式和损失函数, L1 损失的作用范围
#  目前修改了预测标签, 弃用 average_hr;
#  再修改了 average_pred, 直接对 resnet 输出取平均
fixSeed(42)
# folds, path, train_config, test_config
cross_validation(1, "./saved", running.TrainConfig,
                 running.TestConfig, running.TestConfig.methods)
