import segmentation_models_pytorch as smp
import torch
from config import CFG
from u2net import U2NET, U2NETP
from unet.unet_model import UNet
from utils import criterion

import sys

sys.path.append('./SonarSAMsrc/')
sys.path.append('./cenet/')
from SonarSAMsrc.model.model_proxy_SAM import SonarSAM
from cenet.cenet import CE_Net_

############################### 网络搭建##############################
if CFG.model_type == 'smp':
    net = smp.DeepLabV3Plus(
        encoder_name="se_resnext50_32x4d",
        encoder_weights=None,
        in_channels=3,
        classes=1,

    )
elif CFG.model_type == 'unet':
    net = UNet(n_channels=3, n_classes=1)
elif CFG.model_type == 'u2net':
    net = U2NET(in_ch=3, out_ch=1)
elif CFG.model_type == 'u2net_p':
    net = U2NETP(in_ch=3, out_ch=1)
elif CFG.model_type == 'mobile_sam':  # num_classes仅对custom的head类型有效
    net = SonarSAM(model_name='mobile',
                   checkpoint=r'/home/zyj/demo/SonarSAMsrc/mobile_sam.pt',
                   num_classes=1,
                   is_finetune_image_encoder=True,
                   use_adaptation=False,
                   adaptation_type='LORA',
                   head_type='semantic_mask_decoder',
                   reduction=4, upsample_times=2, groups=4)
elif CFG.model_type == 'cenet':
    net = CE_Net_(num_classes=1, num_channels=3)

if __name__ == '__main__':
    print(net)
    x = torch.randn(1, 3, 1024, 1024)

    out = net(x)
    print(out.shape)
