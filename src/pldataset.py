import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.dataset import Wsj02mixDataset, VbdDataset, VbdTestDataset


# DataModule: dataloaderをまとめて記述（collate_fnも定義する）
class VbdDataModule(LightningDataModule):  # 名前をWsj02mixにしたい
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
            self.train_set = Wsj02mixDataset(self.base_dir, mode="train")  # [base_dir]/train
            self.valid_set = Wsj02mixDataset(self.base_dir, mode="valid")  # [base_dir]/valid

        # validate: 検証（？）
        if stage == "validate" or stage is None:
            self.valid_set = Wsj02mixDataset(self.base_dir, mode="valid")

        # test: テスト（？），今回は [base_dir]/valid からロードしている．
        if stage == "test" or stage is None:
            # Testing with a "real" test set should only be run once. Here, we used the validation set.
            self.test_set = Wsj02mixDataset(self.base_dir, mode="test")

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
            collate_fn=self.collate_fn_train,  # バッチに対する処理
        )
        return loader  # 戻り値の名前を統一

    def val_dataloader(self):  # 名前はval_dataloaderにしないとダメ（関数を上書きするため）
        loader = DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_valid,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_test,
        )
        return loader

    # collate function: データセットから取り出した1つずつのデータをバッチにする際の前処理
    # 引数はバッチ（リスト？）

    def collate_fn_train(self, batch):  # 訓練ステップ，位置はランダムにしている
        # 長さを揃える
        siglen = self.siglen  # 統一する長さ
        mixture_list = []
        source1_list = []
        source2_list = []
        # バッチにあるnoisyとcleanの長さを揃える．
        for mixture, source1, source2 in batch:
            if len(mixture) > siglen:  # クリップ（siglenより短い場合）
                start_idx = np.random.randint(0, len(mixture) - siglen)  # ランダムな位置から切り出し
                # listにおける+はappendと同義
                mixture_list += [mixture[start_idx : start_idx + siglen]]
                source1_list += [source1[start_idx : start_idx + siglen]]
                source2_list += [source2[start_idx : start_idx + siglen]]

            else:  # 0埋め（siglenより長い場合）
                mixture_list += [F.pad(mixture, (0, siglen - len(mixture)))]
                source1_list += [F.pad(source1, (0, siglen - len(source1)))]
                source2_list += [F.pad(source2, (0, siglen - len(source2)))]

        return (
            torch.stack(mixture_list),
            torch.stack(source1_list),
            torch.stack(source2_list),
        )  # stackでtorchに書き換え

    def collate_fn_valid(self, batch):  # 検証ステップ，目安が変わると良くないため位置は固定
        siglen = self.siglen

        mixture_list = []
        source1_list = []
        source2_list = []
        for mixture, source1, source2 in batch:
            if len(mixture) > siglen:
                start_idx = 0  # 固定
                mixture_list += [
                    mixture[start_idx : start_idx + siglen]
                ]  # 固定した位置から切り出し
                source1_list += [source1[start_idx : start_idx + siglen]]
                source2_list += [source2[start_idx : start_idx + siglen]]

            else:
                mixture_list += [F.pad(mixture, (0, siglen - len(mixture)))]
                source1_list += [F.pad(source1, (0, siglen - len(source1)))]
                source2_list += [F.pad(source2, (0, siglen - len(source2)))]

        return (
            torch.stack(mixture_list),
            torch.stack(source1_list),
            torch.stack(source2_list),
        )

    def collate_fn_test(self, batch):  # テストステップ，バッチは1つずつ読み出し
        # Batch size is assumed to be 1
        mixture, source1, source2 = batch[0]

        # STFTのシフト長の倍数になるよう0埋め
        length = len(mixture)
        npad = (length - self.nfft) // self.nhop * self.nhop + self.nfft - length
        mixture = F.pad(mixture, (0, npad))
        source1 = F.pad(source1, (0, npad))
        source2 = F.pad(source2, (0, npad))

        return mixture[None, :], source1[None, :], source2[None, :], length  #
