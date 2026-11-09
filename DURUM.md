# Proje Durumu

*Son güncelleme: 2026-11-09*

**Proje:** D-Fire üzerinde 4 sınıflı (yok / yangın / duman / ikisi) görüntü sınıflandırma. AI For Mankind bağımsız duman testi için kullanılacak.
**Kararlar:** FLAME ve Corsican projeden çıkarıldı. Görev sınıflandırma; YOLO isteğe bağlı ve en sonda.

## Aşamalar

| # | Aşama | Durum |
|---|---|---|
| 1 | Repo iskeleti | ✅ Bitti |
| 2 | Veri hazırlama (D-Fire etiket, dedup, bölme) | ✅ Bitti |
| 2b | Veri analizi (EDA, grafikler) — Faz 1 şartı | ✅ Bitti (`results/eda/`) |
| 2c | Faz 1 raporu ve sunumu | ✅ Hazır (`rapor/`) |
| 3 | Klasik ML baseline (HSV+LBP, SVM/RF) | ✅ Bitti |
| 4 | Derin öğrenme eğitimi (CNN, ResNet50, MobileNetV3) | 🟡 Kod hazır, tam eğitim yapılmadı |
| 5 | Değerlendirme (eşik, Grad-CAM, AI For Mankind testi) | ⬜ Başlamadı |
| 6 | Rapor güncellemesi (md/pdf/docx) | ⬜ Başlamadı |

## Sonuçlar

**Veri (yinelenenler elendikten sonra, 16.619 görüntü)**

| Bölme | yok | yangın | duman | ikisi |
|---|---|---|---|---|
| train | 4.420 | 594 | 2.947 | 2.914 |
| val | 780 | 105 | 520 | 515 |
| test | 1.699 | 200 | 1.059 | 866 |

21.527 görüntüden 4.908'i (%23) birebir aynı hash nedeniyle elendi.

**Klasik ML (test seti)**

| Model | Doğruluk | Makro-F1 | Yangın duyarlılığı |
|---|---|---|---|
| SVM (RBF) | 0.752 | 0.707 | 0.78 |
| Random Forest | 0.809 | 0.779 | 0.63 |

**Derin modeller:** henüz gerçek sonuç yok. Küçük veriyle 1 epoch hızlı test geçti (parametre: CNN 1,2M, MobileNetV3 4,2M, ResNet50 23,5M).

## Faz 1 çıktıları
- Rapor: `rapor/Faz1_Raporu.docx` ve `.pdf`; sunum: `rapor/Faz1_Sunum.pptx`
- Veri analizi: `src/eda.py` → `results/eda/` (grafikler, `summary.json`)
- AI For Mankind v1.0 (744 görüntü) indirildi; yalnızca duman, negatif örnek yok
- Önemli bulgular: sınıf dengesizliği 7,67×; yangın kutularının %63'ü görüntünün %1'inden küçük; D-Fire'dan %23 yinelenen elendi; iki kaynak arasında parlaklık farkı

## Dengesizlik kararı (2026-11-09)
Eğitim çok etiketli yapıldı (yangın var/yok + duman var/yok, 4 sınıf türetilir). Dış veri (FLAME yangın görüntüleri) yalnızca ilk eğitimde yangın duyarlılığı zayıf çıkarsa, sadece eğitime eklenip ablation olarak raporlanacak. Kod 60 görüntülük alt kümeyle hata vermediği doğrulandı; gerçek eğitim henüz yapılmadı.

## Sıradaki adımlar
1. `python -m src.cache_images` ile 224px önbellek üret (`data/cache224`).
2. Colab not defterini yaz; projeyi ve önbelleği Google Drive'a yükle.
3. Colab'de 3 model × 3 seed eğit, `results/` altındaki JSON'ları repoya ekle.
4. Değerlendirme: eşik seçimi, Grad-CAM, hata analizi, AI For Mankind bağımsız test, çıkarım süresi.
5. Raporu yeni veri/sonuçlara göre güncelle.

## Bilinen sınırlılıklar ve notlar
- "Yangın" sınıfı küçük (test 200, val 105); metrikleri oynak olabilir, makro-F1 ve sınıf başı duyarlılık esas alınacak.
- Sadece birebir aynı hash elendi; çok benzer kareler kalmış olabilir (klasik ML'in %81'e ulaşması bunun bir işareti olabilir). Raporda sınırlılık olarak yazılacak.
- D-Fire ağırlıklı yer kamerası görüntüsü; rapordaki "drone" vurgusu yumuşatılmalı.
- AI For Mankind yalnızca duman içerir; bağımsız testte sadece duman var/yok metriği hesaplanacak.
- Yangın sınıfı zayıf çıkarsa FLAME yangın görüntüleri yalnızca eğitime eklenip ablation olarak karşılaştırılabilir.
- Rapor (`yangin_tespiti_on_rapor.*`) hâlâ eski 4 veri setli hâliyle duruyor; güncellenecek.

## Nasıl çalıştırılır
```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.prepare_data --dfire_dir data/D-Fire
.venv\Scripts\python -m src.classical_baseline
.venv\Scripts\python -m src.cache_images
.venv\Scripts\python -m src.train --model mobilenetv3 --seed 0
```
