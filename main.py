import argparse
import sys
from gradeCAM import GradCAM

from detectron2.utils.logger import setup_logger
import logging
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup

import detectron2.utils.comm as comm

sys.path.append(".")
from sparseinst import add_sparse_inst_config

logger = logging.getLogger("sparseinst")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#
# from exceptions import NoSuchNameError, NoIndexError


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
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sparseinst")
    return cfg


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument("--fp16", action="store_true",
                      help="support fp16 for inference")
    args.add_argument('--img_path', type=str, default="grad-test/test.jpg ", help='Image Path')

    # Available model list:{'alexnet', 'vgg19', 'resnet50', 'densenet169', 'mobilenet_v2' ,'wide_resnet50_2', ...}
    args.add_argument('--model_path', type=str, default="resnet50",
                      help='Choose a pretrained model or saved model (.pth)')
    args.add_argument('--select_t_layer', type=str2bool, default='False', help='Choose a target layer manually?')
    args = args.parse_args()
    logger.info("Command Line Args:", args)
    cfg = setup(args)

    gradcam_obj = GradCAM(cfg,img_path=args.img_path,
                          model_path=args.model_path,
                          select_t_layer=args.select_t_layer,class_index=968)
    gradcam_obj()
