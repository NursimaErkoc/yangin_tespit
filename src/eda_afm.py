"""AI For Mankind (HPWREN, sürüm 1.0) için ayrıntılı analiz: kutu boyutu ve konumu.

    python -m src.eda_afm --afm_dir data/AFM_v1/extracted/annotated_bounding_box_hpwren
Çıktılar: results/eda/afm_box_stats.png, results/eda/afm_summary.json
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/eda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afm_dir", default="data/AFM_v1/extracted/annotated_bounding_box_hpwren")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    area, cx, cy, per_img, names = [], [], [], [], []
    for x in sorted((Path(args.afm_dir) / "xmls").glob("*.xml")):
        r = ET.parse(x).getroot()
        w, h = (int(r.find("size").find(k).text) for k in ("width", "height"))
        objs = r.findall("object")
        per_img.append(len(objs))
        for o in objs:
            b = o.find("bndbox")
            x0, y0, x1, y1 = (int(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            names.append(o.find("name").text)
            area.append((x1 - x0) * (y1 - y0) / (w * h) * 100)
            cx.append((x0 + x1) / 2 / w)
            cy.append((y0 + y1) / 2 / h)
    area, cx, cy = map(np.array, (area, cx, cy))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(area, bins=np.logspace(-2, 2, 35), color="#5C7A8C")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Kutu alanı / görüntü alanı (%)")
    ax[0].set_ylabel("Kutu sayısı")
    ax[0].set_title("AI For Mankind: duman kutusu boyutu")
    sc = ax[1].hist2d(cx, 1 - cy, bins=[16, 12], range=[[0, 1], [0, 1]], cmap="Oranges")
    plt.colorbar(sc[3], ax=ax[1], label="Kutu sayısı")
    ax[1].set_xlabel("Yatay konum (0 = sol, 1 = sağ)")
    ax[1].set_ylabel("Dikey konum (0 = alt, 1 = üst)")
    ax[1].set_title("Duman kutusu merkezlerinin konumu")
    plt.tight_layout()
    plt.savefig(OUT / "afm_box_stats.png", dpi=150)

    summary = {
        "goruntu": len(per_img), "kutu": int(len(area)), "siniflar": sorted(set(names)),
        "goruntu_basina_kutu": {"min": int(min(per_img)), "max": int(max(per_img)),
                                "ortalama": round(float(np.mean(per_img)), 2)},
        "kutu_alani_pct": {"min": round(float(area.min()), 3), "medyan": round(float(np.median(area)), 2),
                           "ortalama": round(float(area.mean()), 2), "max": round(float(area.max()), 2)},
        "kutu_lt_1pct": round(float((area < 1).mean()), 3),
        "kutu_lt_5pct": round(float((area < 5).mean()), 3),
        "dikey_konum_medyan_ustten": round(float(np.median(cy)), 2),
    }
    (OUT / "afm_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
