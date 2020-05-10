import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cls_net import ResNeXt_CLS


class OptimizerMgr(object):
    
    def __init__(self):
        super(OptimizerMgr, self).__init__()
        
    def test_mgr(self, model, frozen_scope='Resnext101'):
        
        for name, child in model.named_children():
            print('-------------------- name: ', name)
            if frozen_scope in name:
            # if True:
                print('frozen [{}]'.format(name))
                for param in child.parameters():
                    param.requires_grad = False
                    print(param.size())
        
        model_dict = model.state_dict()
        for key, val in model_dict.items():
            print(key)
            print(val.requires_grad)
            param = model.__getattr__(name=key)
            print(param)
            
            
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=1e-3,
                                     amsgrad=True)
        
        
        
        
        


if __name__=='__main__':
    
    model = ResNeXt_CLS(n_cls=3)
    
    optim_mgr = OptimizerMgr()
    optim_mgr.test_mgr(model=model)
    

