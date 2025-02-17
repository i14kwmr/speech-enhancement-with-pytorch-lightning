import functools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from pytorch_lightning import LightningModule

from src.model import UNet
from src.utils.scheduler import CosineWarmupScheduler
from src.metrics import psa, sisdr


class VbdLitModel(LightningModule):  # モデル
    def __init__(self, model_hparams, optimizer_hparams, stft_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet()  # モデルの登録
        self.stft = functools.partial(
            torch.stft,
            n_fft=stft_hparams.nfft,
            hop_length=stft_hparams.nhop,
            return_complex=True,
        )
        self.istft = functools.partial(
            torch.istft,
            n_fft=stft_hparams.nfft,
            hop_length=stft_hparams.nhop,
            return_complex=False,
        )

    # 順伝搬．どこまで書くかを自分で決める（lossを取る対象によって戻り値を変更）
    def forward(self, wave):
        # STFTはGPU上でできるため，ここで処理
        complex_spec, normalized_magnitude = self.pre_process_on_gpu(wave)
        # UNetでマスクを推定
        mask = self.model(normalized_magnitude[:, None, :, :])
        return mask * complex_spec

    # 訓練ステップ，lossを必ず戻さないといけない．
    # pytorchlightningではlossを戻すだけで良い（pytorchではloss.backward等の処理が必要）
    def training_step(self, batch, batch_idx):
        noisy_wave, clean_wave = batch  # Tensors are on GPU.
        complex_spec_estimate = self.forward(noisy_wave)

        complex_spec_clean = self.stft(clean_wave)

        # psa: 2つの複素数の差の絶対値の平均 (metrics.py)
        loss = psa(complex_spec_clean, complex_spec_estimate)

        self.log("train_loss", loss)  # 持ってorderdictに記録
        return loss  # 戻り値はlossのみ

    # 検証ステップ，訓練ステップとの違いは戻り値（validation_epoch_end）
    def validation_step(self, batch, batch_idx):
        noisy_wave, clean_wave = batch  # Tensors are on GPU.
        complex_spec_estimate = self.forward(noisy_wave)

        complex_spec_clean = self.stft(clean_wave)
        valid_loss = psa(complex_spec_clean, complex_spec_estimate)

        self.log("valid_loss", valid_loss)
        return valid_loss.item()  # torch.Tensor -> 組み込み型

    # プログレスバーの設定．設定なしでもプログレスバーは出てくるがvalid_lossは出力されない．
    # outputsはvalidation_stepの出力valid_loss.item()のリスト
    def validation_epoch_end(self, outputs):
        self.log("valid_avg_loss", np.mean(outputs), prog_bar=True)

    # Tensors are on GPU. Batch size is assumed to be 1
    def test_step(self, batch, batch_idx):
        # lengthはどこから取得？
        noisy_wave, clean_wave, length = batch

        complex_spec_estimate = self.forward(noisy_wave)
        estimate_wave = self.istft(complex_spec_estimate)

        clean_wave = clean_wave[0, :length]
        noisy_wave = noisy_wave[0, :length]
        estimate_wave = estimate_wave[0, :length]

        # 評価指標 (SISDR) の計算
        sisdri = (
            sisdr(clean_wave, estimate_wave) - sisdr(clean_wave, noisy_wave)
        ).item()

        self.log("test_sisdri", sisdri, prog_bar=True)
        return sisdri

    # optimizerを定義
    def configure_optimizers(self):
        params = self.hparams.optimizer_hparams
        # AdamW: Adamの亜種
        optimizer = optim.AdamW(
            self.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
        )
        # スケジューラ: learning_rateを変動させる，learning_rateは最初から急に動くと良くない（？）
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup=params.warmup, max_iters=params.nepoch
        )
        scheduler = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        return [optimizer], [scheduler]

    # GPU上で行う処理（CPU上でやる場合はcollate_fnのイメージ）
    # 処理はバッチごとで行われる．ここでは，波形をスペクトログラムに変換
    def pre_process_on_gpu(self, wave, flooring=1e-4):
        complex_spec = self.stft(wave)
        magnitude = torch.abs(complex_spec)
        maxval = magnitude.reshape(-1).max(-1)[0]  # 正規化（要調査）
        log_magnitude = torch.log10(torch.clamp(magnitude, min=flooring * maxval))
        normalized_magnitude = log_magnitude - log_magnitude.mean(-1, keepdim=True)
        return complex_spec, normalized_magnitude
