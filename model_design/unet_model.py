import torch
import torch.nn as nn
import torch.nn.functional as F

from model_design.unet.unet_model import UNet

class UNetModel(nn.Module):

    def __init__(self):
        super(UNetModel, self).__init__()
        self.unet = UNet(n_channels=3, n_classes=1)
        # self.criteria = nn.CrossEntropyLoss()
        self.criteria = nn.BCEWithLogitsLoss()

    def forward_train(self, x, gt_masks):
        pred_masks = self.unet(x)
        # print(pred_masks.size())
        # print(gt_masks.size())
        seg_loss = self.loss(pred_masks=pred_masks, gt_masks=gt_masks)
        return dict(pred_masks=pred_masks, loss=seg_loss)

    def forward_test(self, x):
        out = self.unet(x)
        pred_masks = F.sigmoid(out)
        return dict(pred_masks=pred_masks)

    def forward(self, x, gt_masks=None):
        if self.training:
            return self.forward_train(x, gt_masks)
        else:
            return self.forward_test(x)

    def loss(self, pred_masks, gt_masks):
        seg_loss = self.criteria(input=pred_masks, target=gt_masks)
        return seg_loss




def main():

    model = UNetModel()
    model.train()
    # model.cuda()

    input = torch.randn(2, 3, 128, 128, dtype=torch.float32)
    # gt_masks = torch.randint(low=0, high=1, size=(2, 1, 128, 128), dtype=torch.long)
    gt_masks = torch.randn(2, 1, 128, 128, dtype=torch.float32)
    # print(gt_masks)
    output = model(x=input, gt_masks=gt_masks)
    print(output['pred_masks'].size())
    print(output['loss'])



if __name__=='__main__':

    main()

















