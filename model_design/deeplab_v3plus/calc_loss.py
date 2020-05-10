import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(reduction='mean')
        # self.weight = weight

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


if __name__=='__main__':

    model = CrossEntropyLoss2d()
    inputs = torch.randn(2, 10, 512, 512, dtype=torch.float32)
    targets = torch.randint(low=0, high=9, size=(2, 512, 512), dtype=torch.long)
    output = model(inputs, targets)
    print(output)