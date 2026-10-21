"""D-Fire YOLO etiketlerinden 4 sınıflı görüntü etiketi üretir, yinelenenleri eler,
train/val/test bölmesini yazar.

Beklenen yapı (D-Fire'ın orijinal düzeni):
    <dfire_dir>/train/images, <dfire_dir>/train/labels
    <dfire_dir>/test/images,  <dfire_dir>/test/labels
D-Fire sınıfları: 0 = smoke (duman), 1 = fire (yangın).

Kullanım:
    python -m src.prepare_data --dfire_dir data/D-Fire --out data/splits.csv
"""
import argparse
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CLASSES = ["yok", "yangin", "duman", "ikisi"]
SMOKE, FIRE = 0, 1
IMG_EXT = {".jpg", ".jpeg", ".png"}


def label_from_file(label_path: Path) -> int:
    has_smoke = has_fire = False
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            has_smoke |= cls == SMOKE
            has_fire |= cls == FIRE
    return CLASSES.index({(0, 0): "yok", (0, 1): "yangin",
                          (1, 0): "duman", (1, 1): "ikisi"}[(has_smoke, has_fire)])


def collect(dfire_dir: Path, part: str) -> pd.DataFrame:
    rows = []
    for img in sorted((dfire_dir / part / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        lbl = dfire_dir / part / "labels" / (img.stem + ".txt")
        rows.append({"path": str(img), "label": label_from_file(lbl), "orig_split": part})
    return pd.DataFrame(rows)


def add_hash(df: pd.DataFrame) -> pd.DataFrame:
    hashes = []
    for p in tqdm(df["path"], desc="phash"):
        with Image.open(p) as im:
            hashes.append(str(imagehash.phash(im.convert("RGB"))))
    df = df.copy()
    df["phash"] = hashes
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfire_dir", required=True)
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.dfire_dir)
    df = pd.concat([collect(root, "train"), collect(root, "test")], ignore_index=True)
    df = add_hash(df)

    # Aynı phash'e sahip görüntüler: tek örnek tut, ve test'te olan bir hash train'de varsa
    # train kopyasını at (sızıntıyı önlemek için test önceliklidir).
    df["is_test"] = df["orig_split"] == "test"
    df = df.sort_values("is_test", ascending=False).drop_duplicates("phash", keep="first")
    df = df.drop(columns="is_test").reset_index(drop=True)

    train = df[df["orig_split"] == "train"]
    tr, va = train_test_split(train, test_size=args.val_frac, stratify=train["label"],
                              random_state=args.seed)
    df.loc[tr.index, "split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[df["orig_split"] == "test", "split"] = "test"

    assert not set(df[df.split == "test"].phash) & set(df[df.split != "test"].phash)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = (df.groupby(["split", "label"]).size().unstack(fill_value=0)
                 .rename(columns=dict(enumerate(CLASSES))))
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/data_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
