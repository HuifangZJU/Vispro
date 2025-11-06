from ..models.ffc import DefaultInpaintingModule
from ..models.unet import Marker_Unet, Marker_Unet_Deep
from ..models.u2net import u2net
import torch
import torch.optim

__all__ = ["get_combined_Generator", "getLamaInpainter", "get_bg_model"]

def get_combined_Generator(device):
    generator = Marker_Unet_Deep()
    generator.load_state_dict((torch.load(
        './pretrained_models/g_1000.pth')))
    generator.eval()
    generator.to(device)
    return generator
def getLamaInpainter(device):
    model = DefaultInpaintingModule()
    state = torch.load('./pretrained_models/best.ckpt', map_location=device)
    model.load_state_dict(state['state_dict'],strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    return model

def get_bg_model(model_name,device):
    if model_name == "u2netp":
        net = u2net.U2NETP(3, 1)
        path = '/home/huifang/workspace/code/Vispro/pretrained_models/u2netp.pth'

    elif model_name == "u2net":
        net = u2net.U2NET(3, 1)
        path = '/home/huifang/workspace/code/Vispro/pretrained_models/u2net.pth'

    elif model_name == "u2net_human_seg":
        net = u2net.U2NET(3, 1)
        path = '/home/huifang/workspace/code/Vispro/pretrained_models/u2net_human_seg.pth'

    else:
        print("Choose between u2net, u2net_human_seg or u2netp")
    net.load_state_dict(torch.load(path))
    net.to(device)
    net.eval()
    return net