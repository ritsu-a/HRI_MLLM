# region Import

import yaml
import torch
import argparse

import torch.distributed as dist

from easydict import EasyDict
from motion_gpt import MoGPT

# endregion


# region Args Parser
# cd ~/HRI_MLLM/HRI_mllm/model/motion_head
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 train_motion_gpt.py --config motion_gpt.yaml --train

parser = argparse.ArgumentParser(
    description="Pytorch implementation"
)

parser.add_argument('--config', type=str, default='')
parser.add_argument('--local-rank', default=-1, type=int,
                    help='node rank for distributed training')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true')
group.add_argument('--eval', action='store_true')

args = parser.parse_args()

# endregion


# region Distribution Setting

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

# endregion


def main(args):
    with open(args.config, "r") as f:
        config = EasyDict(yaml.safe_load(f))
    config['local_rank'] = args.local_rank

    agent = MoGPT(config)

    if args.train:
        agent.train()
    elif args.eval:
        agent.eval()
    else:
        raise ValueError


if __name__ == '__main__':
    main(args)