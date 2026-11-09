# Derin Öğrenme Tabanlı Orman Yangını ve Duman Tespiti

Uygulamalı Yapay Zeka Dersi — Dönem Projesi (bireysel) · Öğrenci: Nursima Erkoç

Anlık ilerleme için bkz. [DURUM.md](DURUM.md). Bu dosya projenin **ne yapmak istediğini, neyi neden seçtiğimizi ve planı** anlatır.

## 1. Amaç

Açık alan görüntülerinden orman yangınını ve dumanını otomatik tespit etmek. Erken uyarı için kritik olan şey, yangının büyümeden, ince bir duman hâlindeyken yakalanmasıdır.

Her görüntü şu 4 sınıftan birine atanır:

| Sınıf | Anlamı |
|---|---|
| yok | yangın da duman da yok |
| yangın | yalnızca alev var |
| duman | yalnızca duman var |
| ikisi | alev ve duman birlikte |

## 2. Kesinleşen kararlar

| Konu | Karar | Neden |
|---|---|---|
| Görev | 4 sınıflı görüntü sınıflandırma | Üç mimariyi adil karşılaştırmak kolay; nesne tespitine göre daha az riskli |
| Eğitim verisi | **D-Fire** (sadece) | Hem yangın hem duman için kutu etiketi var, 4 sınıf eksiksiz üretilebiliyor |
| Bağımsız test | **AI For Mankind Wildfire Smoke** | Eğitimde hiç görülmeyen kaynak; genelleme ölçülür (yalnızca duman var/yok) |
| Çıkarılan veriler | FLAME, Corsican | Duman etiketi yok, alan farkı (drone / yer kamerası) ve etiket uyumsuzluğu yaratıyor |
| Eğitim ortamı | Google Colab (GPU) | 
| Çatı | PyTorch, scikit-learn | |
| Esas metrik | Makro-F1 ve sınıf başına duyarlılık | Sınıflar dengesiz; kaçırılan yangın en pahalı hata |
| YOLO | İsteğe bağlı, en sonda | Önce sınıflandırma tamamlanacak |

**Dengesizlik sırası:** (1) çok etiketli çıktı, (2) ağırlıklı kayıp + veri artırma + eşik seçimi, (3) en son çare dış veri. **Koşullu karar:** yangın duyarlılığı bu önlemlerle de zayıf çıkarsa, FLAME'in yangın görüntüleri **yalnızca eğitime** eklenir. Doğrulama ve test D-Fire'da kalır ve ek verinin etkisi ablation olarak raporlanır. Zayıf çıkmazsa eklenmez.

## Öneri raporuyla fark

Öneri aşamasında teslim edilen rapor (`yangin_tespiti_on_rapor.pdf`, `.md`) dört veri seti (D-Fire, FLAME, Corsican, AI For Mankind) içerir. Faz 1'de FLAME ve Corsican kapsam dışı bırakıldı (duman etiketi yok ve alan farkı nedeniyle dört sınıflı etiket şemasıyla uyumsuz). Güncel kapsam yukarıdaki "Kesinleşen kararlar" tablosundadır; Faz 1 çıktıları `rapor/` klasöründedir.

## 3. Yöntem

**Veri hazırlama:** D-Fire'ın YOLO kutu etiketlerinden görüntü düzeyi etiket üretilir (kutu yok → yok; yalnızca yangın; yalnızca duman; ikisi). Birebir aynı görüntüler hash ile elenir (test'te olanın eğitim kopyası atılır, böylece sızıntı olmaz). D-Fire'ın kendi test bölmesi korunur, eğitimden %15 doğrulama ayrılır.

**Modeller** (hepsi aynı bölme, aynı veri artırma ve aynı eğitim ayarıyla):
1. Klasik ML baseline: HSV renk histogramı + LBP doku özellikleri → SVM, Random Forest
2. Temel CNN (sıfırdan)
3. ResNet50 (transfer öğrenme)
4. MobileNetV3-Large (drone/kenar cihaz için hafif model)

**Eğitim (çok etiketli):** derin modeller iki bağımsız çıktı üretir: *yangın var/yok* ve *duman var/yok*; 4 sınıf bunlardan türetilir (karışıklık matrisi ve makro-F1 yine 4 sınıf üzerinden raporlanır). Böylece sınıf dengesizliği azalır: yangın içeren 5.194 / içermeyen 11.425 (2,2:1), duman içeren 8.821 / içermeyen 7.798 (1,1:1); tek başına "sadece yangın" sınıfı ise %5,4'tür. Kayıp: `pos_weight`'li BCE; AdamW, cosine öğrenme hızı, epoch başı checkpoint, her model için 3 seed. Eşikler doğrulamada seçilir. Klasik ML baseline 4 sınıflı kalır.

## 4. Değerlendirme planı

- Doğruluk, makro-F1, sınıf başına duyarlılık, karışıklık matrisi, ROC/PR eğrileri
- Doğrulamada yangın duyarlılığı hedefli eşik seçimi
- Parametre sayısı ve çıkarım süresi (hız–doğruluk dengesi)
- Grad-CAM ile modelin nereye baktığı
- En kötü yanlış alarm ve kaçırma örnekleri (bulut, sis, gün batımı vb.)
- AI For Mankind üzerinde bağımsız genelleme testi
- Literatür karşılaştırması: her makalenin veri/görev/metriği belirtilerek; birebir kıyas iddia edilmeden

## 5. Plan ve durum

| # | Aşama | Durum |
|---|---|---|
| 1 | Repo iskeleti | ✅ |
| 2 | Veri hazırlama (etiket, dedup, bölme) | ✅ |
| 3 | Klasik ML baseline | ✅ |
| 4 | Derin model eğitimi (3 model × 3 seed, Colab) | 🟡 kod hazır, tam eğitim bekliyor |
| 5 | Değerlendirme (eşik, Grad-CAM, hata analizi, bağımsız test) | ⬜ |
| 6 | Rapor güncellemesi | ⬜ |
| 7 | İsteğe bağlı: FLAME ablation, YOLOv8n | ⬜ |

## 6. Sonuçlar (şu ana kadar)

D-Fire, yinelenenler elendikten sonra 16.619 görüntü (train 10.875 / val 1.920 / test 3.824). Test seti:

| Model | Doğruluk | Makro-F1 | Yangın duyarlılığı |
|---|---|---|---|
| SVM (RBF) | 0.752 | 0.707 | 0.78 |
| Random Forest | 0.809 | 0.779 | 0.63 |
| CNN / ResNet50 / MobileNetV3 | henüz eğitilmedi | | |

## 7. Bilinen sınırlılıklar

- "Yangın" sınıfı küçük (test 200, val 105); metrikleri oynak olabilir.
- Yalnızca birebir aynı hash elendi; çok benzer kareler kalmış olabilir ve sonuçları şişirebilir.
- D-Fire ağırlıklı yer kamerası/internet görüntüsüdür; drone senaryosuna doğrudan genellenmeyebilir.
- AI For Mankind yalnızca duman içerdiğinden yangın sınıfı bağımsız testte ölçülemez.

## 8. Depo yapısı

```
src/prepare_data.py       D-Fire etiket dönüştürme, dedup, bölme
src/classical_baseline.py HSV+LBP, SVM / Random Forest
src/cache_images.py       Görüntüleri 224px .npy önbelleğe yazar
src/models.py             CNN, ResNet50, MobileNetV3
src/train.py              Eğitim + test değerlendirmesi
configs/                  Ayarlar
results/                  Sonuç JSON/CSV dosyaları
DURUM.md                  Güncel ilerleme
```

## 9. Çalıştırma

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
1. D-Fire'ı [buradan](https://github.com/gaia-solutions-on-demand/DFireDataset) indirip `data/D-Fire/` altına açın (`train/images`, `train/labels`, `test/images`, `test/labels`).
2. `python -m src.prepare_data --dfire_dir data/D-Fire`
3. `python -m src.classical_baseline`
4. `python -m src.cache_images` → `python -m src.train --model mobilenetv3 --seed 0`

## 10. Kaynaklar

- de Venâncio, P.V.A.B., Lisboa, A.C., Barbosa, A.V. (2022). A dataset for fire and smoke object detection. *Multimedia Tools and Applications*. [10.1007/s11042-022-13580-x](https://doi.org/10.1007/s11042-022-13580-x)
- Mamadaliev, D. vd. (2024). ESFD-YOLOv8n: Early Smoke and Fire Detection Method Based on an Improved YOLOv8n Model. *Fire*, 7(9), 303. [10.3390/fire7090303](https://doi.org/10.3390/fire7090303)
- Kumar, A. vd. (2024). Wildfire and smoke early detection for drone applications: A light-weight deep learning approach. *Engineering Applications of Artificial Intelligence*, 136, 108977. [10.1016/j.engappai.2024.108977](https://doi.org/10.1016/j.engappai.2024.108977)
- Shamsoshoara, A. vd. (2021). Aerial imagery pile burn detection using deep learning: The FLAME dataset. *Computer Networks*, 193, 108001. [10.1016/j.comnet.2021.108001](https://doi.org/10.1016/j.comnet.2021.108001) (koşullu ablation için)
- Joshi, D.D. vd. (2024). Deep learning with ensemble approach for early pile fire detection using aerial images. *Frontiers in Environmental Science*, 12. [10.3389/fenvs.2024.1440396](https://doi.org/10.3389/fenvs.2024.1440396)
- Toulouse, T. vd. (2017). Computer vision for wildfire research: An evolving image dataset for processing and analysis. *Fire Safety Journal*, 92, 188–194. [10.1016/j.firesaf.2017.06.012](https://doi.org/10.1016/j.firesaf.2017.06.012)

Veri setleri: [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) · [AI For Mankind](https://github.com/aiformankind/wildfire-smoke-dataset)
