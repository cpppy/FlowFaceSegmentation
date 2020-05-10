import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets.seg import SegDataset
from model_design.deeplabv3plus_model import DeepLabV3Plus
from checkpoint_mgr.checkpoint_mgr import CheckpointMgr
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():

    dataset = SegDataset()
    data_loader = DataLoader(dataset=dataset,
                             batch_size=2,
                             shuffle=True,
                             num_workers=1,
                             pin_memory=False,
                             drop_last=True)

    model = DeepLabV3Plus()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)  # , weight_decay=5e-4)
    save_dir = '/data/output/deeplabv3plus_v1'
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model, warm_load=True)

    model = model.cuda()

    model.train()
    epochs = 100
    loss_list = []
    acc_list = []

    for epoch in range(epochs):
        for idx, batch_data in enumerate(data_loader):
            input_x, gt_masks = batch_data
            input_x = input_x.cuda()
            gt_masks = gt_masks.cuda()

            output = model(inputs=input_x, labels=gt_masks)
            loss = output['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            # # calc acc
            # pred = torch.max(scores, 1).indices
            # train_correct = (pred == gt_labels).sum().item()
            # train_acc = float(train_correct) / len(gt_labels)
            # acc_list.append(train_acc)

            steps = (idx + 1 + epoch * len(data_loader))
            lr = optimizer.param_groups[0]['lr']

            log_interval = 1
            if steps % log_interval == 0:
                print('epoch[{}][{}/{}] lr:{}, loss:{:.3f}'.format(epoch + 1,
                                                                   idx + 1,
                                                                   len(data_loader),
                                                                   lr,
                                                                   np.mean(loss_list),
                                                                   ))
                loss_list.clear()
                acc_list.clear()

            ckpt_interval = 100
            if steps % ckpt_interval == 0:
                checkpoint_op.save_checkpoint(model=model)





if __name__ == '__main__':

    main()

