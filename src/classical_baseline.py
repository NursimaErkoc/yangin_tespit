"""Klasik ML baseline: HSV renk histogramı + LBP doku özellikleri -> SVM / Random Forest.

Kullanım:
    python -m src.classical_baseline --splits data/splits.csv [--subset 300]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["yok", "yangin", "duman", "ikisi"]
SIZE = 128
LBP_P, LBP_R = 8, 1


def extract(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((SIZE, SIZE))
        hsv = np.asarray(im.convert("HSV"), dtype=np.uint8)
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    feats = []
    for ch, bins in zip(range(3), (16, 8, 8)):
        h, _ = np.histogram(hsv[..., ch], bins=bins, range=(0, 256), density=True)
        feats.append(h)
    feats.append(np.concatenate([hsv.reshape(-1, 3).mean(0), hsv.reshape(-1, 3).std(0)]) / 255.0)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    h, _ = np.histogram(lbp, bins=LBP_P + 2, range=(0, LBP_P + 2), density=True)
    feats.append(h)
    return np.concatenate(feats).astype(np.float32)


def features(paths, cache: Path):
    if cache.exists():
        return np.load(cache)["x"]
    x = np.stack(Parallel(n_jobs=-1)(delayed(extract)(p) for p in paths))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x)
    return x


def evaluate(name, model, xtr, ytr, xva, yva, xte, yte):
    t = time.time()
    model.fit(xtr, ytr)
    fit_s = time.time() - t
    out = {"model": name, "fit_seconds": round(fit_s, 1)}
    for split, x, y in (("val", xva, yva), ("test", xte, yte)):
        pred = model.predict(x)
        out[f"{split}_acc"] = round(accuracy_score(y, pred), 4)
        out[f"{split}_macro_f1"] = round(f1_score(y, pred, average="macro"), 4)
    pred = model.predict(xte)
    recall = np.diag(confusion_matrix(yte, pred, labels=range(4), normalize="true"))
    out["test_per_class_recall"] = {c: round(float(r), 4) for c, r in zip(CLASSES, recall)}
    print(name, json.dumps(out, ensure_ascii=False))
    print(classification_report(yte, pred, target_names=CLASSES, digits=3, zero_division=0))
    return out, confusion_matrix(yte, pred, labels=range(4)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--subset", type=int, default=0, help="her bölmeden en fazla N örnek (hızlı test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.splits)
    if args.subset:
        df = pd.concat([g.sample(min(len(g), args.subset), random_state=args.seed)
                        for _, g in df.groupby("split")])
    tag = f"subset{args.subset}" if args.subset else "full"
    data = {}
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        data[s] = (features(d.path.tolist(), Path(f"data/features_{tag}_{s}.npz")), d.label.values)
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]

    sc = StandardScaler().fit(xtr)
    xtr_s, xva_s, xte_s = sc.transform(xtr), sc.transform(xva), sc.transform(xte)

    results, cms = [], {}
    for name, model, xs in (
        ("SVM-RBF", SVC(C=10, kernel="rbf", class_weight="balanced", random_state=args.seed),
         (xtr_s, xva_s, xte_s)),
        ("RandomForest", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                n_jobs=-1, random_state=args.seed),
         (xtr, xva, xte)),
    ):
        r, cm = evaluate(name, model, xs[0], ytr, xs[1], yva, xs[2], yte)
        results.append(r)
        cms[name] = cm
    Path("results").mkdir(exist_ok=True)
    Path(f"results/classical_{tag}.json").write_text(
        json.dumps({"results": results, "confusion_matrices": cms}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
