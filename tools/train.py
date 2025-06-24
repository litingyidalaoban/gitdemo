# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import os
import os.path as osp
from mmengine.logging import print_log
from mmengine.config import DictAction,Config
import logging

from mmengine.runner import Runner

from mmseg.registry import RUNNERS

#如果是两个横杠
def parse_args():
    #创建一个args的解析的对象，所有的args都会绑定到他上面去
    parser = argparse.ArgumentParser(description='Train a segmentor')
    #添加声明，比如config，help就是帮助信息
    #action=storetrue就是当这个参数有输入的时候就是True，不然模型只有输入True才行，一定要严格
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    #这个nargs就是如果你用了--cfg-options，你就得至少输入一个，然后变成列表
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    #这个就是只能从里面选一个，不然报错，或者默认就是none
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    #允许两种不同的名称，但是限定输入为整数类型，默认是0
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    #这里是把所有args都放在命名空间，可以用args.xxx来获取
    args = parse_args()
###############################################################
    #这里其实只用到了模型的配置文件的声明信息，就是unet.py
    cfg=Config.fromfile(args.config)
    #这里用的是pytorch环境
    cfg.launcher = args.launcher
    #这里的如果有其他需要更改的，就把args里的那些额外的配置，改到cfg里面去
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    #命令行的优先级大于配置文件大于里面自带的
    # work_dir is determined in this priority: CLI > segment in file > filename
    #如果需要更改工作目录，就把cfg里的工作目录改成args里的
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    #又如果配置文件里没有工作目录，那就是默认在根目录下创建一个config同名的这个目录
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
##########################################################
        #关于混合精度训练
        # enable automatic-mixed-precision training
        #如果需要混合精度训练，只有true和false两种情况，选择了参数但不填也是True
    if args.amp is True:
        #优化器就变成了配置文件中的优化器的类型
        optim_wrapper = cfg.optim_wrapper.type
        #如果他是自动混合精度的优化器，就打印你已经是混合精度了
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        #否则，如果开了，但是optimwrapper不是Optimwrapper，就会把optimwrapper变成ampoptimwrapper
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
############################################################
    #关于继续训练，看看args里面需不需要继续训练，如果要就覆盖上去，没有就是cfg里默认的
    cfg.resume = args.resume
    
##################################
    #从配置文件中构建runner，
        # build the runner from config
    #先看看是不是在cfg里面，没有的话就是默认用cfg里的配置实例化一个runner。
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    # start training
    runner.train()
    # print("over")

if __name__ == '__main__':
    main()