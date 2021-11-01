import argparse
import pathlib
from tqdm import tqdm

import numpy as np
import soundfile as sf


# タスク依存なフェーズ，一番大変
# ここでシミュレート，リサンプリング
# オブションも書くと良い→データセット増えるがデバッグが楽になる．
# バッチ：3000ファイルをメモリにそもそも積めない，確率的に局所解を回避できる
# バッチサイズを大きくすると速くなるが，
# 3->1000
# 6->500 ただ，処理スピード1/2となるわけではない．
def make_dataset(base_path, fnames, save_path):
    for fname in tqdm(fnames):
        mixture, _ = sf.read(base_path.joinpath("mix", fname))
        source1, _ = sf.read(base_path.joinpath("s1", fname))
        source2, _ = sf.read(base_path.joinpath("s2", fname))

        npy_name = (
            pathlib.Path(save_path).joinpath(fname).with_suffix(".npy")
        )  # numpyのbinデータ
        with open(npy_name, "wb") as f:  #
            np.save(f, mixture.astype(np.float32))  # float64->float32
            np.save(f, source1.astype(np.float32))
            np.save(f, source2.astype(np.float32))


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("base_data_dir", type=str)
    parser.add_argument("base_save_dir", type=str)
    parser.add_argument("--num_train_ratio", type=float, default=0.8) # not use
    args = parser.parse_args()
    
    mode_dict = {"train":"tr", "valid":"cv", "test":"tt"}

    for mode in ["train", "valid", "test"]:
        # Prepare file names
        base_path = pathlib.Path(args.base_data_dir + "/" + mode_dict[mode])
        mixture_path = base_path.joinpath("mix")
        fnames = np.sort([fname.parts[-1] for fname in mixture_path.glob("*.wav")])

        # Run make_dataset
        make_dataset(
            base_path, fnames=fnames, save_path=args.base_save_dir + "/" + mode
        )


if __name__ == "__main__":
    main()
