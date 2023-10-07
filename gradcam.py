import json
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from model_v2 import MobileNetV2
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from sparseinst.sparseinst import SparseInst
from detectron2.config.config import get_cfg
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
import detectron2.utils.comm as comm
from sparseinst.config import add_sparse_inst_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "sparseinst" module
    # setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sparseinst")
    return cfg


def main():
    # ----载入自己的模型，按照自己训练集训练的
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    model = SparseInst(cfg)
    # model = MobileNetV2(num_classes=5)  # .to(device)
    # load model weights
    model_weight_path = "pretrained_models/sparse_inst_r50vd_dcn_giam_aug_67dc06.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))  # 加载权重
    model.eval()
    target_layers = [model.features[-1]]  # 拿到最后一个层结构

    # 载入模型，模型权重---载入的模型是按照ImageNet训练的（不是自己的模型是从torchvision中导入的）
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "grad-test/test.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    # 实例化，输出模型，要计算的层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # 感兴趣的label
    target_category = 0  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog
    # 计算cam图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)  # 实例化
    # 将只传入的一张图的cam图提取出来
    grayscale_cam = grayscale_cam[0, :]
    # 变成彩色热力图的形式
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  # 将原图缩放到[0,1]之间
                                      grayscale_cam,
                                      use_rgb=True)
    # 展示出来
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    plt.title(class_indict[str(target_category)])
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
