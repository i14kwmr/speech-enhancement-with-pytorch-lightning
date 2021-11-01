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
        mask1, mask2 = self.model(normalized_magnitude[:, None, :, :])
        return mask1 * complex_spec, mask2 * complex_spec

    # 訓練ステップ，lossを必ず戻さないといけない．
    # pytorchlightningではlossを戻すだけで良い（pytorchではloss.backward等の処理が必要）
    def training_step(self, batch, batch_idx):
        mixture_wave, source1_wave, source2_wave = batch  # Tensors are on GPU.
        complex_spec_estimate1, complex_spec_estimate2 = self.forward(mixture_wave)

        complex_spec_source1 = self.stft(source1_wave)
        complex_spec_source2 = self.stft(source2_wave)

        # psa: 2つの複素数の差の絶対値の平均 (metrics.py)
        # pit: permutation invariant training
        loss1 = psa(complex_spec_source1, complex_spec_estimate1) + psa(
            complex_spec_source2, complex_spec_estimate2
        )
        loss2 = psa(complex_spec_source1, complex_spec_estimate2) + psa(
            complex_spec_source2, complex_spec_estimate1
        )

        loss = torch.max(loss1, loss2)

        self.log("train_loss", loss)  # 持ってorderdictに記録
        return loss  # 戻り値はlossのみ

    # 検証ステップ，訓練ステップとの違いは戻り値（validation_epoch_end）
    def validation_step(self, batch, batch_idx):
        mixture_wave, source1_wave, source2_wave = batch  # Tensors are on GPU.
        complex_spec_estimate1, complex_spec_estimate2 = self.forward(mixture_wave)

        complex_spec_source1 = self.stft(source1_wave)
        complex_spec_source2 = self.stft(source2_wave)

        valid_loss1 = psa(complex_spec_source1, complex_spec_estimate1) + psa(
            complex_spec_source2, complex_spec_estimate2
        )
        valid_loss2 = psa(complex_spec_source1, complex_spec_estimate2) + psa(
            complex_spec_source2, complex_spec_estimate1
        )

        valid_loss = torch.max(valid_loss1, valid_loss2)

        self.log("valid_loss", valid_loss)
        return valid_loss.item()  # torch.Tensor -> 組み込み型

    # プログレスバーの設定．設定なしでもプログレスバーは出てくるがvalid_lossは出力されない．
    # outputsはvalidation_stepの出力valid_loss.item()のリスト
    def validation_epoch_end(self, outputs):
        self.log("valid_avg_loss", np.mean(outputs), prog_bar=True)

    # Tensors are on GPU. Batch size is assumed to be 1
    def test_step(self, batch, batch_idx):
        # lengthはどこから取得？
        mixture_wave, source1_wave, source2_wave, length = batch

        complex_spec_estimate1, complex_spec_estimate2 = self.forward(mixture_wave)
        estimate_wave1 = self.istft(complex_spec_estimate1)
        estimate_wave2 = self.istft(complex_spec_estimate2)

        source1_wave = source1_wave[0, :length]
        source2_wave = source2_wave[0, :length]
        mixture_wave = mixture_wave[0, :length]
        estimate_wave1 = estimate_wave1[0, :length]
        estimate_wave2 = estimate_wave2[0, :length]

        # 評価指標 (SISDR) の計算
        sisdri1 = (
            (sisdr(source1_wave, estimate_wave1) - sisdr(source1_wave, mixture_wave))
            + (sisdr(source2_wave, estimate_wave2) - sisdr(source2_wave, mixture_wave))
        ) / 2
        sisdri2 = (
            (sisdr(source1_wave, estimate_wave2) - sisdr(source1_wave, mixture_wave))
            + (sisdr(source2_wave, estimate_wave1) - sisdr(source2_wave, mixture_wave))
        ) / 2

        sisdri = torch.max(sisdri1, sisdri2).item()

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
