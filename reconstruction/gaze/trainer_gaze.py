import time

import numpy as np
np.bool = np.bool_
np.int  = int

from argparse import ArgumentParser

import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import timm
from datasets.dataset_gaze import GazeDataset, DataLoaderX
from models import GazeModel





def cli_main():
    pl.seed_everything(727)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--backbone', default='resnet34', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=16, type=int)
    parser.add_argument('--root', default='data/gaze_refine', type=str)
    parser.add_argument('--num-gpus', default=1, type=int)
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if not args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ------------
    # data
    # ------------
    train_set = GazeDataset(root_dir=args.root, is_train=True)
    val_set = GazeDataset(root_dir=args.root, is_train=False)
    print('train data size:', len(train_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ------------
    # model
    # ------------
    model = GazeModel(backbone=args.backbone, epoch=args.epoch)
    ckpt_path = f'../../logs/gaze_ffhq_dgaze_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if args.resume:
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict['state_dict'], strict=True)

    # ------------
    # training
    # ------------
    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=ckpt_path,
            filename='{epoch:02d}-{val_loss:.6f}',
            save_top_k=5,
            mode='min',
            )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus = args.num_gpus,
        accelerator="gpu",
        strategy="ddp",
        benchmark=True,
		logger=TensorBoardLogger(ckpt_path),
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=1,
        #progress_bar_refresh_rate=1,
        log_every_n_steps=1,
        max_epochs=args.epoch,
    )
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    cli_main()

