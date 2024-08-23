
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
import model.ResNet as CResNet
from model.ctran import ctranspath
from torchvision import transforms

import sys
sys.path.append('/data/CVizTool/tiatoolbox/tiatoolbox/')
from tiatoolbox.models.models_abc import (ModelABC)

def upsample2x(feat):
    return F.interpolate(
        feat, scale_factor=2, mode="bilinear", align_corners=False
    )


def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image
    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


class ResNetExt(ResNet):
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x0 = x = self.conv1(x)
        x0 = x = self.bn1(x)
        x0 = x = self.relu(x)
        x1 = x = self.maxpool(x)
        x1 = x = self.layer1(x)
        x2 = x = self.layer2(x)
        x3 = x = self.layer3(x)
        x4 = x = self.layer4(x)
        return x0, x1, x2, x3, x4

    @staticmethod
    def resnet50():
        return ResNetExt(ResNetBottleneck, [3, 4, 6, 3])


class UpSample2x(nn.Module):
    """Upsample input by a factor of 2.

    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2x, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


class CustomCNN(ModelABC):
    def __init__(self,
                repo='pytorch/vision:v0.10.0',
                model_name = 'shufflenet_v2_x1_0',
                pretrained=True,
                freeze_encoder=True
                ):
        self.repo = repo
        self.model_name = model_name
        self.pretrained = pretrained
        super(CustomCNN,self).__init__()
        self.is_freeze = freeze_encoder
        model = torch.hub.load(self.repo,
                                self.model_name, 
                                pretrained=self.pretrained
                                )
        # if the model is missing adaptive pooling layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        randImg = torch.rand((1,3,224,224))
        if len(model(randImg).shape)>2:
            self.feature_extract = torch.nn.Sequential(*list(model.children()),nn.AdaptiveAvgPool2d(1))
        else:
            # No need of adaptive pooling as it is already feature vector
            self.feature_extract = torch.nn.Sequential(*list(model.children()))
    
    def forward(self,imgs):
         # is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not self.is_freeze):
            return self.feature_extract(imgs)
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        
        #~~~~~~~ Normalizing Images as RetCCL paper
        preprocess =  transforms.Compose(
            [
                transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225))
            ]
            )
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model.eval()
        device = 'cuda' if on_gpu else 'cpu'
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        images = batch_data/255.0  # in NHWC
        images = images.to(device).type(torch.float32)
        images = images.permute(0, 3, 1, 2)  # to NCHW

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with torch.no_grad():
            # Preprocessing as per github code
            images = preprocess(images)
            return [model(images).reshape((images.shape[0],1024)).cpu().numpy()]

class RetCCL(ModelABC):
    def __init__(self,
                 num_classes=128,
                mlp=False,
                two_branch=False,
                normlinear=True,
                freeze_encoder=True,
                checkpoint_path = None):
        super(RetCCL,self).__init__()
        self.is_freeze = freeze_encoder
        model = CResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
        model.fc= nn.Identity()
        model.load_state_dict(torch.load(checkpoint_path), strict=True)
        self.feature_extract = model
    
    def forward(self,imgs):
         # is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not self.is_freeze):
            return self.feature_extract(imgs)
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        
        #~~~~~~~ Normalizing Images as RetCCL paper
        preprocess =  transforms.Compose(
            [
                transforms.Resize(256),
                transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225))
            ]
            )
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        model.eval()
        device = 'cuda' if on_gpu else 'cpu'
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        images = batch_data/255.0  # in NHWC
        images = images.to(device).type(torch.float32)
        images = images.permute(0, 3, 1, 2)  # to NCHW

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with torch.no_grad():
            # Preprocessing as per github code
            # import pdb;pdb.set_trace()
            images = preprocess(images)
            return [model(images).cpu().numpy()]
        

class TransformerEncoder(ModelABC):
    def __init__(self,
                 num_classes=0,
                freeze_encoder=True,
                pretrained=True,
                model_name='beitv2_base_patch16_224.in1k_ft_in22k'):
        super(TransformerEncoder,self).__init__()
        self.is_freeze = freeze_encoder
        import timm
        
        model = timm.create_model(
            'beitv2_base_patch16_224.in1k_ft_in22k',
            pretrained=True,
            num_classes=num_classes  # remove classifier nn.Linear
            )
        model = model.eval()
        self.feature_extract = model
    
    def forward(self,imgs):
         # is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not self.is_freeze):
            return self.feature_extract(imgs) #forward_features
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        
        #~~~~~~~ Normalizing Images as RetCCL paper
        preprocess =  transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225))
            ]
            )
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        device = 'cuda' if on_gpu else 'cpu'
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        images = batch_data/255.0  # in NHWC
        images = images.to(device).type(torch.float32)
        images = images.permute(0, 3, 1, 2)  # to NCHW

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with torch.no_grad():
            # Preprocessing as per github code
            images = preprocess(images)
            return [model(images).cpu().numpy()]

class FCN_Model(ModelABC):
    def __init__(self, 
        nr_output_ch=2,
        freeze_encoder=True
    ):
        super(FCN_Model, self).__init__()
        # Normalize over last dimension
        self.freeze_encoder = freeze_encoder

        self.backbone = ResNetExt.resnet50()
        img_list = torch.rand([1, 3, 256, 256])
        out_list = self.backbone(img_list)
        # orderd from lores hires
        down_ch_list = [v.shape[1] for v in out_list][::-1]

        self.conv1x1 = None
        if down_ch_list[0] != down_ch_list[1]:  # channel mapping for shortcut
            self.conv1x1 = nn.Conv2d(
                down_ch_list[0], down_ch_list[1], (1, 1), bias=False)

        self.uplist = nn.ModuleList()
        for ch_idx, ch in enumerate(down_ch_list[1:]):
            next_up_ch = ch
            if ch_idx + 2 < len(down_ch_list):
                next_up_ch = down_ch_list[ch_idx+2]
            self.uplist.append(
                nn.Sequential(
                    nn.BatchNorm2d(ch), nn.ReLU(),
                    nn.Conv2d(ch, next_up_ch, (3, 3), padding=1, bias=False),
                    nn.BatchNorm2d(next_up_ch), nn.ReLU(),
                    nn.Conv2d(next_up_ch, next_up_ch, (3, 3), padding=1, bias=False),
                )
            )

        self.clf = nn.Conv2d(next_up_ch, nr_output_ch, (1, 1), bias=True)
        self.upsample2x = UpSample2x()
        return

    def forward(self, img_list):
        img_list = img_list / 255.0  # scale to 0-1

        is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not is_freeze):
            # assume output is after each down-sample resolution
            en_list = self.backbone(img_list)

        if self.conv1x1 is not None:
            x = self.conv1x1(en_list[-1])

        en_list = en_list[:-1]
        for idx in range(1, len(en_list)+1):
            y = en_list[-idx]
            x = self.upsample2x(x) + y
            x = self.uplist[idx-1](x)
        output = self.clf(x)
        return output

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):

        ####
        model.eval()
        device = 'cuda' if on_gpu else 'cpu'

        ####
        img_list = batch_data
        img_list = img_list.to(device).type(torch.float32)
        img_list = img_list.permute(0, 3, 1, 2)  # to NCHW

        # --------------------------------------------------------------
        with torch.no_grad():
            logit_list = model(img_list)
            logit_list = logit_list.permute(0, 2, 3, 1)  # to NHWC
            prob_list = F.softmax(logit_list, -1)
            prob_list = prob_list.permute(0, 3, 1, 2)  # to NCHW
            prob_list = upsample2x(prob_list)
            prob_list = crop_op(prob_list, [512, 512])
            prob_list = prob_list.permute(0, 2, 3, 1)  # to NHWC

        prob_list = prob_list.cpu().numpy()
        return [prob_list]


#CTransPath model
class CTransPath(ModelABC):
    def __init__(self,
                 num_classes=0,
                freeze_encoder=True,
                pretrained=True,
                checkpoint_path = None):
        super(CTransPath,self).__init__()
        self.is_freeze = freeze_encoder
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(checkpoint_path)
        model.load_state_dict(td['model'], strict=True)
        model = model.eval()
        self.feature_extract = model
    
    def forward(self,imgs):
         # is_freeze = not self.training or self.freeze_encoder
        with torch.set_grad_enabled(not self.is_freeze):
            return self.feature_extract(imgs) #forward_features
    
    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        
        #~~~~~~~ Normalizing Images as RetCCL paper
        preprocess =  transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225))
            ]
            )
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        device = 'cuda' if on_gpu else 'cpu'
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        images = batch_data/255.0  # in NHWC
        images = images.to(device).type(torch.float32)
        images = images.permute(0, 3, 1, 2)  # to NCHW

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with torch.no_grad():
            # Preprocessing as per github code
            images = preprocess(images)
            return [model(images).cpu().numpy()]