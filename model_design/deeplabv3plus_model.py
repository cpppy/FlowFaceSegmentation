import torch
import torch.nn as nn
import torch.nn.functional as F
from model_design.deeplab_v3plus.deepv3 import DeepV3Plus
from model_design.deeplab_v3plus.calc_loss import CrossEntropyLoss2d, BCEWithLogitsLoss



class DeepLabV3Plus(nn.Module):

    def __init__(self, n_cls=17):
        super(DeepLabV3Plus, self).__init__()
        self.seg_module = DeepV3Plus(num_classes=n_cls, trunk='seresnext-101')
        self.criterion = CrossEntropyLoss2d()
        # self.criterion = BCEWithLogitsLoss()

    def forward_train(self, inputs, labels):
        seg_map = self.seg_module(inputs)
        loss = self.criterion(seg_map, labels)
        return dict(seg_map=seg_map, loss=loss)

    def forward_test(self, inputs):
        seg_map = self.seg_module(inputs)
        return dict(seg_map=seg_map)

    def forward(self, inputs, labels=None):
        if self.training:
            return self.forward_train(inputs, labels)
        else:
            return self.foward_test(inputs)


if __name__=='__main__':

    model = DeepLabV3Plus()
    inputs = torch.randn(2, 3, 512, 768, dtype=torch.float32)
    targets = torch.randint(low=0, high=16, size=(2, 512, 768), dtype=torch.long)
    output = model(inputs, targets)
    print(output['loss'])



