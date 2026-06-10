from networks.unet import UNet, UNet_FEA, UNet_RE
import argparse

def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
        # net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "unet_fea":
        net = UNet_FEA(in_chns=in_chns, class_num=class_num).cuda()
    
    # XNet based on unet, two inputs two outputs 
    elif net_type == "unet_re":
        net = UNet_RE(in_chns=in_chns, class_num=class_num).cuda()


    # XNet based on unet, two inputs two outputs 
    # elif net_type == "LNet":
    #     net = LNet(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
