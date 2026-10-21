"""Eğitim + test değerlendirmesi.

    python -m src.train --model mobilenetv3 --seed 0 --cache data/cache224 --epochs 20
    python -m src.train --model cnn --subset 200 --epochs 1 --no_pretrained   # hızlı duman testi
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from src.models import build_model

CLASSES = ["yok", "yangin", "duman", "ikisi"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CachedDataset(Dataset):
    def __init__(self, cache, split, train, subset=0, seed=0):
        self.x = np.load(Path(cache) / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(Path(cache) / f"{split}_y.npy")
        self.idx = np.arange(len(self.y))
        if subset:
            self.idx = np.random.RandomState(seed).permutation(len(self.y))[:subset]
        norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
        aug = [v2.RandomHorizontalFlip(),
               v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
               v2.ColorJitter(0.2, 0.2, 0.2, 0.02)] if train else []
        self.tf = v2.Compose(aug + norm)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = torch.from_numpy(np.array(self.x[j])).permute(2, 0, 1)
        return self.tf(img), int(self.y[j])

    @property
    def labels(self):
        return self.y[self.idx]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def metrics(logits, y):
    pred = logits.argmax(1).numpy()
    y = y.numpy()
    cm = confusion_matrix(y, pred, labels=range(4))
    recall = np.diag(confusion_matrix(y, pred, labels=range(4), normalize="true"))
    return {"acc": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")),
            "recall": {c: float(r) for c, r in zip(CLASSES, recall)}, "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnn", "resnet50", "mobilenetv3"])
    ap.add_argument("--cache", default="data/cache224")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = f"{args.model}_seed{args.seed}"
    ckpt = Path(args.ckpt_dir) / f"{name}.pt"
    best_path = Path(args.ckpt_dir) / f"{name}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    ds = {s: CachedDataset(args.cache, s, s == "train", args.subset, args.seed)
          for s in ("train", "val", "test")}
    dl = {s: DataLoader(d, args.batch_size, shuffle=(s == "train"), num_workers=args.workers,
                        pin_memory=device == "cuda") for s, d in ds.items()}

    counts = np.bincount(ds["train"].labels, minlength=4).astype(np.float32)
    weights = torch.tensor(counts.sum() / (4 * np.maximum(counts, 1)), device=device)
    model = build_model(args.model, not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    start, best, history = 0, -1.0, []
    if args.resume and ckpt.exists():
        st = torch.load(ckpt, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, best, history = st["epoch"] + 1, st["best"], st["history"]
        print(f"resume epoch {start}")

    for ep in range(start, args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        vm = metrics(*predict(model, dl["val"], device))
        history.append({"epoch": ep, "train_loss": tot / len(ds["train"]), "val_acc": vm["acc"],
                        "val_macro_f1": vm["macro_f1"], "sec": time.time() - t0})
        print(json.dumps(history[-1]))
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "history": history}, ckpt)

    model.load_state_dict(torch.load(best_path, map_location=device))
    logits, y = predict(model, dl["test"], device)
    res = {"model": args.model, "seed": args.seed, "params": sum(p.numel() for p in model.parameters()),
           "test": metrics(logits, y), "best_val_macro_f1": best, "history": history, "device": device}

    # çıkarım süresi (batch=1, ısınma sonrası)
    x1 = ds["test"][0][0].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x1)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(30):
            model(x1)
        if device == "cuda":
            torch.cuda.synchronize()
    res["ms_per_image"] = (time.time() - t) / 30 * 1000

    torch.save(logits, Path(args.out) / f"{name}_test_logits.pt")
    Path(args.out, f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("model", "seed", "params", "ms_per_image")}),
          "test acc", res["test"]["acc"], "macro_f1", res["test"]["macro_f1"])


if __name__ == "__main__":
    main()
