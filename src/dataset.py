import pathlib
import numpy as np
import torch
import torchaudio


class Wsj02mixDataset(torch.utils.data.Dataset):  # torch.utils.data.Datasetを継承
    def __init__(self, base_dir, mode="train"):
        # mode: train/valid/test
        path = pathlib.Path(base_dir).joinpath(mode)
        self.npy_names = np.sort(list(path.glob("*.npy")))  # 名前をsort

    def __len__(self):
        return len(self.npy_names)  # len()の戻り値

    def __getitem__(self, idx):
        with open(self.npy_names[idx], "rb") as f:
            # load file (.npy), rb: read binary
            mixture = np.load(f)
            source1 = np.load(f)
            source2 = np.load(f)

        # numpy -> torch
        mixture = torch.from_numpy(mixture)
        source1 = torch.from_numpy(source1)
        source2 = torch.from_numpy(source2)
        return mixture, source1, source2  # 参照の戻り値z


# seedについて固定・確認
class VbdDataset(torch.utils.data.Dataset):  # torch.utils.data.Datasetを継承
    def __init__(self, base_dir, mode="train"):
        # mode: train/valid/test
        path = pathlib.Path(base_dir).joinpath(mode)
        self.npy_names = np.sort(list(path.glob("*.npy")))  # 名前をsort

    def __len__(self):
        return len(self.npy_names)  # len()の戻り値

    def __getitem__(self, idx):
        with open(self.npy_names[idx], "rb") as f:
            # load file (.npy), rb: read binary
            noisy = np.load(f)
            clean = np.load(f)

        # numpy -> torch
        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)
        return noisy, clean  # 参照の戻り値


class VbdTestDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir):
        self.base_dir = pathlib.Path(base_dir)
        path = self.base_dir.joinpath("clean_testset_wav2")
        # fname.parts: パスを構成要素に分割したもの（タプル）
        self.fnames = np.sort([fname.parts[-1] for fname in path.glob("*.wav")])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        noisy, _ = torchaudio.load(self.base_dir.joinpath("noisy_testset_wav2", fname))
        clean, _ = torchaudio.load(self.base_dir.joinpath("clean_testset_wav2", fname))
        return noisy[0, :], clean[0, :]
