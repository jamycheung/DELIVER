import torch
import math
from torch import nn
from semseg.models.backbones import *
from semseg.models.layers import trunc_normal_
from collections import OrderedDict

def load_dualpath_model(model, model_file):
    # load raw state_dict
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        #raw_state_dict = torch.load(model_file)
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            # patch_embedx, proj, weight = k.split('.')
            # state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
            # state_dict[new_k] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            # state_dict[k.replace('block', 'extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            # state_dict[k.replace('norm', 'extra_norm')] = v

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    del state_dict
    

class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19, modals: list = ['rgb', 'depth', 'event', 'lidar']) -> None:
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant, modals)
        # self.backbone = eval(backbone)(variant)
        self.modals = modals

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            if len(self.modals)>1:
                load_dualpath_model(self.backbone, pretrained)
            else:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                # if 'PoolFormer' in self.__class__.__name__:
                #     new_dict = OrderedDict()
                #     for k, v in checkpoint.items():
                #         if not 'backbone.' in k:
                #             new_dict['backbone.'+k] = v
                #         else:
                #             new_dict[k] = v
                #     checkpoint = new_dict
                if 'model' in checkpoint.keys(): # --- for HorNet
                    checkpoint = checkpoint['model']
                msg = self.backbone.load_state_dict(checkpoint, strict=False)
                print(msg)
