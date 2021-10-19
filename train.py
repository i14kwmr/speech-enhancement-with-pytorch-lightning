import argparse
import pathlib
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from src.pldataset import VbdDataModule
from src.plmodel import VbdLitModel
from src.utils.callbacks import PerEpochProgressBar


def train(dm, plmodel, cfg, save_path):
    # 再現性を保つための処理
    pl.seed_everything(0)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # callback関数（何かあったら戻す関数）

    # EarlyStoppingする関数
    # patience: patience回lossが改善しなかったらやめる．
    early_stop_callback = EarlyStopping(
        monitor="valid_loss", min_delta=0.00, patience=cfg.optim.patience
    )
    # Checkpointを出力する関数
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=save_path,
        filename="model_{epoch:04d}",
        save_top_k=1,  # lossが良いk番目までモデルを保存する (1だと1つ保存)
    )
    pbar_callback = PerEpochProgressBar()  # Epochごとのプログレスバーの書き出し処理

    # trainerの定義
    trainer = Trainer(
        max_epochs=cfg.optim.nepoch,  # データを全て使って1epoch
        gpus=1,
        precision=16,
        num_sanity_val_steps=-1,  # validの途中でGPUのメモリに乗らないかも，0の時やらない
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            pbar_callback,
        ],  # callbackをリストに登録
    )

    trainer.fit(plmodel, dm)  # 訓練
    return trainer


def main():
    # 準備
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str)
    args = parser.parse_args()

    # load config (hydraのOmegaConf)
    cfg_path = args.cfg_path  # model/config.yaml
    cfg = OmegaConf.load(pathlib.Path(cfg_path).joinpath("config.yaml"))

    # DataModuleの定義 (dataloaderをまとめた関数)
    dm = VbdDataModule(
        base_dir=cfg.data.base_dir,
        siglen=cfg.data.siglen,  # 最大の信号長
        batch_size=cfg.optim.batch_size,
        stft_hparams=cfg.stft,
    )

    # Modelの定義
    plmodel = VbdLitModel(cfg.model, cfg.optim, cfg.stft)

    # trainerの定義と訓練
    trainer = train(dm, plmodel, cfg, save_path=cfg_path)  # trainer

    # sisdrを出力, validでテストしている（testのデータセットで良いモデルを論文に書くの良くない）．
    result = trainer.test()
    print(result)


if __name__ == "__main__":
    main()
