from typing import Optional, List

from torch import Tensor
import torch.distributed as dist

import os
import cv2
import numpy as np
import sys
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torchvision

import logging
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone

from detectron2.engine import default_argument_parser

import detectron2.utils.comm as comm

# from tools.myModel import SparseInst
#
# sys.path.append(".")
#
# logger = logging.getLogger("sparseinst")


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i]
                                            for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def nested_masks_from_list(tensor_list: List[Tensor], input_shape=None):
    if tensor_list[0].ndim == 3:
        dim_size = sum([img.shape[0] for img in tensor_list])
        if input_shape is None:
            max_size = _max_by_axis([list(img.shape[-2:]) for img in tensor_list])
        else:
            max_size = [input_shape[0], input_shape[1]]
        batch_shape = [dim_size] + max_size
        # b, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.zeros(batch_shape, dtype=torch.bool, device=device)
        idx = 0
        for img in tensor_list:
            c = img.shape[0]
            c_ = idx + c
            tensor[idx: c_, :img.shape[1], : img.shape[2]].copy_(img)
            mask[idx: c_, :img.shape[1], :img.shape[2]] = True
            idx = c_
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def aligned_bilinear(tensor, factor):
    # borrowed from Adelaidet: https://github1s.com/aim-uofa/AdelaiDet/blob/HEAD/adet/utils/comm.py
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


# def load_model(model_name, cfg):
#     try:
#         if '.pth' in model_name:  # for saved model (.pth)
#             if torch.typename(torch.load(model_name)) == 'dict':
#
#                 """
#                 if you want to use customized model that has a type 'OrderedDict',
#                 you shoud load model object as follows:
#
#                 from Net import Net()
#                 model=Net()
#                 """
#                 model = SparseInst(cfg)
#                 weights_dict = torch.load(model_name, map_location='cpu')
#                 model.load_state_dict(weights_dict, strict=False)  # 加载自己模型的权重
#                 print(model)
#             else:
#
#                 model = torch.load(model_name)
#
#         elif hasattr(models, model_name):  # for pretrained model (ImageNet)
#             model = getattr(models, model_name)(pretrained=True)
#
#         model.eval()
#         if cuda_available():
#             model.cuda()
#     except:
#         raise ValueError(f'Not unvalid model was loaded: {model_name}')
#
#     return model
#
#
# def cuda_available():
#     use_cuda = torch.cuda.is_available()
#     return use_cuda
#
#
# def load_image(path):
#     img = cv2.imread(path, 1)
#     img = cv2.resize(img, (224, 224))
#     img = np.float32(img) / 255
#
#     return img
#
#
# def preprocess_image(img):
#     means = [0.485, 0.456, 0.406]
#     stds = [0.229, 0.224, 0.225]
#
#     preprocessed_img = img.copy()[:, :, ::-1]
#     for i in range(3):
#         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
#         preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
#     preprocessed_img = \
#         np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
#
#     if cuda_available():
#         preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
#     else:
#         preprocessed_img_tensor = torch.from_numpy(preprocessed_img)
#
#     preprocessed_img_tensor.unsqueeze_(0)
#     return Variable(preprocessed_img_tensor, requires_grad=False)
#
#
# def save(mask, img, img_path, model_path):
#     mask = (mask - np.min(mask)) / np.max(mask)
#
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#
#     heatmap = np.float32(heatmap) / 255
#     gradcam = 1.0 * heatmap + img
#     gradcam = gradcam / np.max(gradcam)
#
#     index = img_path.find('/')
#     index2 = img_path.find('.')
#     path = 'result/' + img_path[index + 1:index2] + '/' + model_path
#     if not (os.path.isdir(path)):
#         os.makedirs(path)
#
#     gradcam_path = path + "/gradcam.png"
#     cv2.imwrite(gradcam_path, np.uint8(255 * gradcam))
#
#
# def is_int(v):
#     v = str(v).strip()
#     return v == '0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()
#
#
# def _exclude_layer(layer):
#     if isinstance(layer, nn.Sequential):
#         return True
#     if not 'torch.nn' in str(layer.__class__):
#         return True
#
#     return False
#
#
# def choose_tlayer(model):
#     name_to_num = {}
#     num_to_layer = {}
#     for idx, data in enumerate(model.named_modules()):
#         name, layer = data
#         if _exclude_layer(layer):
#             continue
#
#         name_to_num[name] = idx
#         num_to_layer[idx] = layer
#         print(f'[ Number: {idx},  Name: {name} ] -> Layer: {layer}\n')
#
#     print('\n<<-------------------------------------------------------------------->>')
#     print('\n<<      You sholud not select [classifier module], [fc layer] !!      >>')
#     print('\n<<-------------------------------------------------------------------->>\n')
#
#     a = input(f'Choose "Number" or "Name" of a target layer: ')
#
#     if a.isnumeric() == False:
#         a = name_to_num[a]
#     else:
#         a = int(a)
#
#     t_layer = num_to_layer[a]
#     return t_layer
#     # except IndexError:
#     #     raise NoIndexError('Selected index (number) is not allowed.')
#     # except KeyError:
#     #     raise NoSuchNameError('Selected name is not allowed.')
#
#
# if __name__ == '__main__':
#     args = default_argument_parser()
#     args.add_argument("--fp16", action="store_true",
#                       help="support fp16 for inference")
#     args = args.parse_args()
#     logger.info("Command Line Args:", args)
#     cfg = setup(args)
