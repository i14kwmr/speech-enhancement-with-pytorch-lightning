import functools
import argparse
import pathlib
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.pldataset import VbdDataModule
from src.plmodel import VbdLitModel
from src.dataset import VbdTestDataset


# collate_fn (trainではpldataset.pyに記述している)
def collate_fn(batch, stft_hparams):
    # Batch size is assumed to be 1
    noisy, clean = batch[0]

    length = len(clean)
    npad = (
        (length - stft_hparams.nfft) // stft_hparams.nhop * stft_hparams.nhop
        + stft_hparams.nfft
        - length
    )
    noisy = F.pad(noisy, (0, npad))
    clean = F.pad(clean, (0, npad))
    return noisy[None, :], clean[None, :], length


def test(dataset, plmodel, cfg):
    # 再現性を保つための処理
    pl.seed_everything(0)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # dataloaderの定義 (trainではpldataset.pyに記述している)
    teset_data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=1,
        collate_fn=functools.partial(collate_fn, stft_hparams=cfg.stft),
    )

    # trainerの定義
    trainer = Trainer(gpus=1)

    # sisdrを出力, dataloaderを定義しないとvalidのデータで処理してしまう．
    result = trainer.test(model=plmodel, dataloaders=[teset_data_loader])
    print(result[0])


def main():
    # 準備
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("best_epoch", type=int)
    args = parser.parse_args()

    # load config (hydraのOmegaConf)
    cfg_path = pathlib.Path(args.cfg_path)
    cfg = OmegaConf.load(cfg_path.joinpath("config.yaml"))

    # Datasetの定義, Trainとは異なるDatasetを利用
    dataset = VbdTestDataset(args.data_dir)

    # Modelの定義 (保存したモデルの重みを読み込み)
    plmodel = VbdLitModel.load_from_checkpoint(
        cfg_path.joinpath(f"model_epoch={args.best_epoch:04}.ckpt")
    )

    test(dataset, plmodel, cfg)


if __name__ == "__main__":
    main()
