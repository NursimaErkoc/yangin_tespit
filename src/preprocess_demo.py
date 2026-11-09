"""Ön işleme örnek çıktısı: orijinal -> 224x224 yeniden boyutlandırma -> veri artırma (eğitim) görseli.

    python -m src.preprocess_demo
Çıktı: results/eda/preprocess_example.png, results/eda/preprocess_stats.json
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from src.train import CachedDataset

CLASSES = ["yok", "yangin", "duman", "ikisi"]
OUT = Path("results/eda")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/splits.csv")
    train = df[df.split == "train"].reset_index(drop=True)
    cache_x = np.load("data/cache224/train_x.npy", mmap_mode="r")

    rows = [train[train.label == c].index[3] for c in range(4)]
    aug = v2.Compose([v2.RandomHorizontalFlip(p=1.0),
                      v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                      v2.ColorJitter(0.2, 0.2, 0.2, 0.02)])
    torch.manual_seed(0)

    fig, ax = plt.subplots(3, 4, figsize=(11, 8))
    for j, i in enumerate(rows):
        with Image.open(train.path[i]) as im:
            orig = im.convert("RGB")
            size = orig.size
        small = Image.fromarray(np.array(cache_x[i]))
        augmented = aug(small)
        for r, (img, t) in enumerate([(orig, f"{CLASSES[j]}: orijinal {size[0]}x{size[1]}"),
                                      (small, "224x224 yeniden boyutlandırılmış"),
                                      (augmented, "veri artırma uygulanmış")]):
            ax[r, j].imshow(img)
            ax[r, j].set_title(t, fontsize=9)
            ax[r, j].axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "preprocess_example.png", dpi=150)

    # normalizasyon sonrası istatistik (ImageNet ortalama/std)
    ds = CachedDataset("data/cache224", "val", train=False)
    xs = torch.stack([ds[i][0] for i in range(200)])
    raw = torch.from_numpy(np.array(np.load("data/cache224/val_x.npy", mmap_mode="r")[:200])).float() / 255
    stats = {
        "onbellek_dtype": str(cache_x.dtype), "onbellek_sekil_train": list(cache_x.shape),
        "ham_piksel_araligi": [float(raw.min()), float(raw.max())],
        "ham_kanal_ortalama": [round(float(m), 3) for m in raw.mean((0, 1, 2))],
        "normalize_sonrasi_ortalama": [round(float(m), 3) for m in xs.mean((0, 2, 3))],
        "normalize_sonrasi_std": [round(float(s), 3) for s in xs.std((0, 2, 3))],
        "normalize_sonrasi_aralik": [round(float(xs.min()), 2), round(float(xs.max()), 2)],
    }
    (OUT / "preprocess_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
