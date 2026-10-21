"""splits.csv'deki görüntüleri sabit boyuta küçültüp bölme başına tek bir .npy dosyasına yazar.
Colab'de Drive'dan binlerce küçük dosya okumak yavaş olduğu için eğitim bu önbellekten yapılır.

    python -m src.cache_images --splits data/splits.csv --size 224 --out data/cache224
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image


def load(path, size):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="data/cache224")
    args = ap.parse_args()
    df = pd.read_csv(args.splits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        d = df[df.split == split]
        x = np.stack(Parallel(n_jobs=-1)(delayed(load)(p, args.size) for p in d.path))
        np.save(out / f"{split}_x.npy", x)
        np.save(out / f"{split}_y.npy", d.label.values.astype(np.int64))
        print(split, x.shape)


if __name__ == "__main__":
    main()
