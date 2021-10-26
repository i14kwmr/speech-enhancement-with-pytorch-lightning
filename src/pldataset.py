import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.dataset import VbdDataset, VbdTestDataset


# DataModule: daatloaderをまとめて記述（collate_fnも定義する）
# 今回は信号を渡すように設定している．（STFTはネットワーク側でやる）
class VbdDataModule(LightningDataModule):
    def __init__(
        self,
        base_dir,
        siglen=None,
        batch_size=1,
        batch_size_valid=None,
        stft_hparams=None,
        pin_memory=True,
        num_workers=4,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.siglen = siglen
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        if stft_hparams is None:
            self.nfft = 1024
            self.nhop = 256
        else:
            self.nfft = stft_hparams.nfft
            self.nhop = stft_hparams.nhop

    def setup(self, stage=None):  # 必須．setup関数
        # fit: 学習（？）
        if stage == "fit" or stage is None:
            self.train_set = VbdDataset(self.base_dir, mode="train")  # [base_dir]/train
            self.valid_set = VbdDataset(self.base_dir, mode="valid")  # [base_dir]/valid

        # validate: 検証（？）
        if stage == "validate" or stage is None:
            self.valid_set = VbdDataset(self.base_dir, mode="valid")

        # test: テスト（？），今回は [base_dir]/valid からロードしている．
        if stage == "test" or stage is None:
            # Testing with a "real" test set should only be run once. Here, we used the validation set.
            self.test_set = VbdDataset(self.base_dir, mode="valid")

    # dataloader: データをまとめたバッチを出力する．（必須）

    def train_dataloader(self):
        loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,  # バッチサイズ
            shuffle=True,  # バッチ作成時にシャッフル（？）
            drop_last=True,  # 最後のデータの扱い（True: 無視する．False: 無視しない）
            pin_memory=self.pin_memory,
            # Trueにすると処理が速くなる何かの危険性あり（？）
            # 致命的にやばくはないらしい．
            num_workers=self.num_workers,
            # スレッド数（バッチを出すのに時間かかる場合は変更）
            # 0: そのまま, 1: マルチスレッド（plでは，1だとワーニングが出る？）
            collate_fn=self.collate_fn_train,  # バッチ (list?) に対する処理
        )
        return loader  # 戻り値の名前を統一

    def val_dataloader(self):  # 名前はval_dataloaderにしないとダメ（関数を上書きするため）
        loader = DataLoader(
            self.valid_set,  #
            batch_size=self.batch_size,
            shuffle=False,  #
            drop_last=False,  #
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_valid,  #
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_set,  #
            batch_size=1,
            shuffle=False,  #
            drop_last=False,  #
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_test,  #
        )
        return loader

    # collate function: データセットから取り出した1つずつのデータをバッチにする際の前処理
    # 引数はバッチ（リスト？）

    def collate_fn_train(self, batch):  # 訓練ステップ，位置はランダムにしている
        # 長さを揃える
        siglen = self.siglen  # 統一する長さ
        noisy_list = []
        clean_list = []
        # バッチにあるnoisyとcleanの長さを揃える．
        for noisy, clean in batch:
            if len(clean) > siglen:  # クリップ（siglenより短い場合）
                start_idx = np.random.randint(0, len(clean) - siglen)  # ランダムな位置から切り出し
                # listにおける+はappendと同義
                noisy_list += [noisy[start_idx : start_idx + siglen]]
                clean_list += [clean[start_idx : start_idx + siglen]]

            else:  # 0埋め（siglenより長い場合）
                noisy_list += [F.pad(noisy, (0, siglen - len(clean)))]
                clean_list += [F.pad(clean, (0, siglen - len(clean)))]

        return torch.stack(noisy_list), torch.stack(clean_list)  # stackでtorchに書き換え

    def collate_fn_valid(self, batch):  # 検証ステップ，目安が変わると良くないため位置は固定
        siglen = self.siglen

        noisy_list = []
        clean_list = []
        for noisy, clean in batch:
            if len(clean) > siglen:
                start_idx = 0  # 固定
                noisy_list += [noisy[start_idx : start_idx + siglen]]  # 固定した位置から切り出し
                clean_list += [clean[start_idx : start_idx + siglen]]

            else:
                noisy_list += [F.pad(noisy, (0, siglen - len(clean)))]
                clean_list += [F.pad(clean, (0, siglen - len(clean)))]

        return torch.stack(noisy_list), torch.stack(clean_list)

    def collate_fn_test(self, batch):  # テストステップ，バッチは1つずつ読み出し
        # Batch size is assumed to be 1
        noisy, clean = batch[0]

        # STFTのシフト長の倍数になるよう0埋め（全てのデータを利用）
        length = len(clean)  # 評価する際，iSTFTしたものをクリップするために準備
        npad = (length - self.nfft) // self.nhop * self.nhop + self.nfft - length
        noisy = F.pad(noisy, (0, npad))
        clean = F.pad(clean, (0, npad))
        return noisy[None, :], clean[None, :], length
