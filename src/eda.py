"""Veri seti keşifsel analizi (Faz 1): sınıf dağılımı, kutu istatistikleri, görüntü özellikleri,
eksik veri kontrolü, örnek görüntüler, D-Fire / AI For Mankind karşılaştırması.

    python -m src.eda --splits data/splits.csv --dfire_dir data/D-Fire \
        --afm_dir data/AFM_v1/extracted/annotated_bounding_box_hpwren
Çıktılar: results/eda/*.png, results/eda/*.csv, results/eda/summary.json
"""
import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

CLASSES = ["yok", "yangin", "duman", "ikisi"]
COLORS = {"yok": "#7f8c8d", "yangin": "#e74c3c", "duman": "#3498db", "ikisi": "#8e44ad"}
OUT = Path("results/eda")


def img_info(path):
    with Image.open(path) as im:
        return im.size[0], im.size[1], im.mode


def small_stats(path, size=64):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    gray = a.mean(2)
    return a.reshape(-1, 3).mean(0).tolist() + [float(gray.mean()), float(gray.std())]


def dfire_boxes(label_path):
    rows = []
    if Path(label_path).exists():
        for line in Path(label_path).read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, _, _, w, h = int(float(p[0])), *map(float, p[1:5])
                rows.append(("duman" if c == 0 else "yangin", w * h))
    return rows


def label_path_of(img_path):
    p = Path(img_path)
    return p.parent.parent / "labels" / (p.stem + ".txt")


def afm_records(afm_dir):
    recs = []
    for x in sorted((Path(afm_dir) / "xmls").glob("*.xml")):
        r = ET.parse(x).getroot()
        w, h, d = (int(r.find("size").find(k).text) for k in ("width", "height", "depth"))
        boxes = []
        for o in r.findall("object"):
            b = o.find("bndbox")
            bw = int(b.find("xmax").text) - int(b.find("xmin").text)
            bh = int(b.find("ymax").text) - int(b.find("ymin").text)
            boxes.append((o.find("name").text, bw * bh / (w * h)))
        recs.append({"file": r.find("filename").text, "w": w, "h": h, "depth": d, "boxes": boxes})
    return recs


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150)
    plt.close()


def grid(paths, titles, name, cols=4, size=3):
    rows = int(np.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 0.8))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, p, t in zip(np.ravel(axes), paths, titles):
        with Image.open(p) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(t, fontsize=9)
    savefig(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--dfire_dir", default="data/D-Fire")
    ap.add_argument("--afm_dir", default="data/AFM_v1/extracted/annotated_bounding_box_hpwren")
    ap.add_argument("--n_sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    df = pd.read_csv(args.splits)
    df["sinif"] = df.label.map(dict(enumerate(CLASSES)))

    # 1) Sınıf dağılımı (bölme x sınıf)
    tab = df.groupby(["split", "sinif"]).size().unstack(fill_value=0)[CLASSES].loc[["train", "val", "test"]]
    tab.to_csv(OUT / "class_distribution.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tab.plot(kind="bar", stacked=False, ax=ax[0], color=[COLORS[c] for c in CLASSES], rot=0)
    ax[0].set_title("Bölmelere göre sınıf sayıları"); ax[0].set_ylabel("Görüntü sayısı")
    (tab.div(tab.sum(1), axis=0) * 100).plot(kind="bar", stacked=True, ax=ax[1],
                                              color=[COLORS[c] for c in CLASSES], rot=0)
    ax[1].set_title("Sınıf oranları (%)"); ax[1].set_ylabel("%")
    ax[1].legend(loc="center left", bbox_to_anchor=(1, .5), title="sinif")
    savefig("class_distribution.png")
    total = df.sinif.value_counts()[CLASSES]
    summary["n_images"] = int(len(df))
    summary["class_counts"] = {c: int(total[c]) for c in CLASSES}
    summary["imbalance_ratio_max_min"] = round(float(total.max() / total.min()), 2)

    # 2) Eksik veri / etiket kontrolü + yinelenen eleme etkisi
    missing_label = sum(not label_path_of(p).exists() for p in df.path)
    empty_label = sum(label_path_of(p).exists() and label_path_of(p).stat().st_size == 0 for p in df.path)
    raw = {s: len(list((Path(args.dfire_dir) / s / "images").iterdir())) for s in ("train", "test")}
    kept = df.orig_split.value_counts().to_dict()
    summary["dedup"] = {s: {"ham": raw[s], "kalan": int(kept[s]), "elenen": raw[s] - int(kept[s])} for s in raw}
    summary["missing_label_files"] = int(missing_label)
    summary["empty_label_files_(yok_sinifi)"] = int(empty_label)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["train", "test"], [raw["train"], raw["test"]], color="#bdc3c7", label="ham")
    ax.bar(["train", "test"], [kept["train"], kept["test"]], color="#27ae60", label="yinelenenler elendikten sonra")
    ax.legend(); ax.set_title("D-Fire: yinelenen görüntü eleme etkisi"); ax.set_ylabel("Görüntü sayısı")
    savefig("dedup_effect.png")

    # 3) Kutu istatistikleri (D-Fire)
    boxes = Parallel(n_jobs=-1)(delayed(dfire_boxes)(label_path_of(p)) for p in df.path)
    flat = [(c, a) for bs in boxes for c, a in bs]
    bdf = pd.DataFrame(flat, columns=["sinif", "alan_orani"])
    summary["dfire_boxes"] = {"toplam": int(len(bdf)), "yangin": int((bdf.sinif == "yangin").sum()),
                              "duman": int((bdf.sinif == "duman").sum()),
                              "goruntu_basina_ortalama_kutu": round(len(bdf) / len(df), 2)}
    summary["dfire_box_area_median_pct"] = {c: round(float(bdf[bdf.sinif == c].alan_orani.median() * 100), 2)
                                            for c in ("yangin", "duman")}
    summary["dfire_small_box_lt1pct_ratio"] = {c: round(float((bdf[bdf.sinif == c].alan_orani < 0.01).mean()), 3)
                                               for c in ("yangin", "duman")}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for c in ("yangin", "duman"):
        ax[0].hist(bdf[bdf.sinif == c].alan_orani * 100, bins=np.logspace(-3, 2, 40), alpha=.6,
                   color=COLORS[c], label=c)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Kutu alanı / görüntü alanı (%)"); ax[0].legend()
    ax[0].set_title("D-Fire kutu boyutu dağılımı (küçük nesne sorunu)")
    nbox = pd.Series([len(b) for b in boxes]).clip(upper=10).value_counts().sort_index()
    ax[1].bar(nbox.index.astype(str), nbox.values, color="#34495e")
    ax[1].set_xlabel("Görüntüdeki kutu sayısı (10+ birleştirildi)"); ax[1].set_title("Görüntü başına kutu sayısı")
    savefig("dfire_box_stats.png")

    # 4) Görüntü özellikleri (boyut, mod/kanal): D-Fire örneklem, AFM tamamı
    samp = df.sample(min(args.n_sample, len(df)), random_state=args.seed)
    info = Parallel(n_jobs=-1)(delayed(img_info)(p) for p in samp.path)
    idf = pd.DataFrame(info, columns=["w", "h", "mod"])
    afm = afm_records(args.afm_dir)
    summary["dfire_image_props_sample"] = {
        "ornek": int(len(idf)), "modlar": idf["mod"].value_counts().to_dict(),
        "genislik_min_medyan_max": [int(idf.w.min()), int(idf.w.median()), int(idf.w.max())],
        "yukseklik_min_medyan_max": [int(idf.h.min()), int(idf.h.median()), int(idf.h.max())],
        "en_yaygin_boyutlar": {f"{k[0]}x{k[1]}": int(v) for k, v in
                               Counter(zip(idf.w, idf.h)).most_common(5)}}
    adf = pd.DataFrame(afm)
    summary["afm"] = {"goruntu": int(len(adf)), "kanal_sayisi": adf.depth.value_counts().to_dict(),
                      "boyutlar": {f"{k[0]}x{k[1]}": int(v) for k, v in Counter(zip(adf.w, adf.h)).most_common(5)},
                      "kutu_toplam": int(sum(len(b) for b in adf.boxes)),
                      "sadece_duman": all(n == "smoke" for bs in adf.boxes for n, _ in bs),
                      "kutusuz_goruntu": int(sum(len(b) == 0 for b in adf.boxes)),
                      "kutu_alani_medyan_pct": round(float(np.median([a for bs in adf.boxes for _, a in bs]) * 100), 2)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(idf.w, idf.h, s=6, alpha=.3, color="#2c3e50")
    ax[0].set_title(f"D-Fire görüntü boyutları (n={len(idf)})"); ax[0].set_xlabel("genişlik"); ax[0].set_ylabel("yükseklik")
    ax[1].scatter(adf.w, adf.h, s=6, alpha=.3, color="#c0392b")
    ax[1].set_title(f"AI For Mankind boyutları (n={len(adf)})"); ax[1].set_xlabel("genişlik")
    savefig("image_sizes.png")

    # 5) Renk/parlaklık: kaynaklar arası alan farkı (domain shift)
    dsamp = df.sample(min(744, len(df)), random_state=args.seed)
    afm_files = sorted((Path(args.afm_dir) / "images").iterdir())
    ds = np.array(Parallel(n_jobs=-1)(delayed(small_stats)(p) for p in dsamp.path))
    as_ = np.array(Parallel(n_jobs=-1)(delayed(small_stats)(p) for p in afm_files))
    summary["brightness_mean"] = {"dfire": round(float(ds[:, 3].mean()), 3), "afm": round(float(as_[:, 3].mean()), 3)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(ds[:, 3], bins=30, alpha=.6, label="D-Fire", color="#2980b9")
    ax[0].hist(as_[:, 3], bins=30, alpha=.6, label="AI For Mankind", color="#c0392b")
    ax[0].set_title("Ortalama parlaklık dağılımı"); ax[0].legend()
    for arr, lab, col in ((ds, "D-Fire", "#2980b9"), (as_, "AI For Mankind", "#c0392b")):
        ax[1].bar([x + (0 if lab == "D-Fire" else .38) for x in range(3)], arr[:, :3].mean(0), width=.38,
                  label=lab, color=col)
    ax[1].set_xticks([.19, 1.19, 2.19]); ax[1].set_xticklabels(["R", "G", "B"]); ax[1].set_title("Ortalama kanal değerleri")
    ax[1].legend()
    savefig("source_comparison.png")

    # 6) Örnek görüntüler
    rng = np.random.RandomState(args.seed)
    paths, titles = [], []
    for c in CLASSES:
        sub = df[df.sinif == c]
        for p in sub.sample(4, random_state=args.seed).path:
            paths.append(p); titles.append(c)
    grid(paths, titles, "samples_dfire.png")
    pick = rng.choice(len(afm_files), 8, replace=False)
    grid([afm_files[i] for i in pick], ["duman (AFM)"] * 8, "samples_afm.png")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
