# Literatür: İncelenen Makaleler

Bu klasörde projenin literatür taramasında incelenen makalelerin PDF'leri ve özetleri bulunur. Aşağıdaki sayılar makalelerin tam metninden alınmıştır. Tam metne erişilemeyen tek çalışma (D-Fire makalesi) ayrıca belirtilmiştir.

## Genel bakış

| # | Çalışma | Veri | Görev | Yöntem | Öne çıkan sonuç | PDF |
|---|---|---|---|---|---|---|
| 1 | de Venâncio vd. (2022) | D-Fire | Nesne tespiti (yangın, duman) | YOLOv4, YOLOv4-tiny, filtre budama | Düşük güçlü cihazda (Raspberry Pi 4) çalışan dedektör | Klasörde yok (ücretli) |
| 2 | Mamadaliev vd. (2024) | D-Fire | Nesne tespiti | ESFD-YOLOv8n: WIoUv3, ResBlock, C2fGELAN | mAP@0.5 = %79,4 (YOLOv8n: %77,4) | [fire-07-00303-v2.pdf](fire-07-00303-v2.pdf) |
| 3 | Kumar vd. (2024) | Corsican, FLAME, SMOKE5K, AI For Mankind | Piksel segmentasyonu (yangın ve duman ayrı ayrı) | DeepLabv3+ ile MobileViT / MobileNetV3 omurgaları | ResNet-50'ye yakın IoU, yaklaşık 1/6 parametreyle | [1-s2.0-S0952197624011357-main.pdf](1-s2.0-S0952197624011357-main.pdf) |
| 4 | Shamsoshoara vd. (2021) | FLAME | İkili sınıflandırma ve segmentasyon | Xception, U-Net | Sınıflandırma testi %76,23; segmentasyon F1 %87,75 | [2012.14036v1.pdf](2012.14036v1.pdf) |
| 5 | Joshi vd. (2024) | FLAME (4 palet) | İkili sınıflandırma | RF, SVM, 5 CNN, VGG16 + oylama | Termal paletlerde %99; normal RGB'de %77 (oylama ile %91) | [fenvs-12-1440396.pdf](fenvs-12-1440396.pdf) |
| 6 | Toulouse vd. (2017) | Corsican Fire Database | Yangın pikseli tespiti (veri seti makalesi) | Maskeli veri tabanı ve renk kuralı yöntemlerinin karşılaştırması | En iyi renk kuralı yöntemi F = 0,91 | [PrePrintFSJ.pdf](PrePrintFSJ.pdf) |
| 7 | Şafak ve Barışçı (2023) | 3 veri setinin birleşimi (43.355 görüntü) | İkili sınıflandırma (yangın var/yok) | Revize MobileNet, transfer öğrenme | Doğruluk %98,37 | [10.17341-gazimmfd.1041091-2148715.pdf](10.17341-gazimmfd.1041091-2148715.pdf) |

## Veri ve değerlendirme karşılaştırması

| Çalışma | Bölme yöntemi | Bağımsız test | Yinelenen eleme | Raporlanan metrik |
|---|---|---|---|---|
| de Venâncio vd. (2022) | D-Fire train/test | Yok | Belirtilmemiş | mAP |
| Mamadaliev vd. (2024) | Tüm D-Fire rastgele %70/20/10 | Yok | Yok | P, R, mAP@0.5 |
| Kumar vd. (2024) | Her veri setinde %70/15/15 | Kaynaklar arası nitel test | Belirtilmemiş | MPA, F1, IoU, MCC |
| Shamsoshoara vd. (2021) | Eğitim ve test farklı drone'lardan | Var (farklı kamera) | — | Doğruluk; segmentasyonda P, R, F1, IoU |
| Joshi vd. (2024) | Termalde aynı videodan; RGB'de farklı drone | Yalnızca RGB'de | Yok | Doğruluk, P, R, F1, AUC |
| Şafak ve Barışçı (2023) | Rastgele %80/20 | Yok | Yok | Doğruluk, P, R, F1 |
| **Bu proje** | D-Fire'ın kendi test bölmesi + eğitimden %15 doğrulama | **AI For Mankind** | **Algısal özet ile (%23 elendi)** | Makro-F1, sınıf başı duyarlılık, ROC/PR, ms/görüntü |

**Ortak gözlem:** Eğitimle aynı kaynaktan gelen test setinde başarı yüksek çıkıyor, farklı kaynakta belirgin biçimde düşüyor. FLAME'de doğrulama doğruluğu %94,31, farklı drone'la test doğruluğu %76,23. Joshi vd.'de tek videodan gelen termal testlerde başarı %99, farklı drone'la RGB testte %77. Bu proje yinelenenleri eleyerek ve bağımsız bir test seti ayırarak bu riski azaltmayı hedefler.

---

## Ayrıntılı özetler

### 1. de Venâncio, Lisboa, Barbosa (2022)
*An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices.* Neural Computing and Applications, 34, 15349–15368. DOI: [10.1007/s00521-022-07467-z](https://doi.org/10.1007/s00521-022-07467-z)

> Tam metin ücretli olduğu için okunamadı. Aşağıdaki veri istatistikleri [D-Fire GitHub sayfasından](https://github.com/gaia-solutions-on-demand/DFireDataset) doğrulanmıştır.

- **Amaç:** Düşük güçlü ve kaynakları sınırlı cihazlarda çalışabilen otomatik yangın ve duman tespiti.
- **Yöntem:** YOLOv4 ve YOLOv4-tiny ile iki sınıflı (yangın, duman) nesne tespiti. Hesap ve bellek maliyetini düşürmek için filtre budama (pruning) uygulanmış.
- **D-Fire veri seti:** Toplam 21.527 görüntü:

  | Kategori | Görüntü |
  |---|---|
  | Yalnızca yangın | 1.164 |
  | Yalnızca duman | 5.867 |
  | Yangın ve duman | 4.658 |
  | Hiçbiri | 9.838 |

  Kutu sayıları: 14.692 yangın, 11.865 duman. Etiketler YOLO formatında. Kaynaklar: güvenlik kameraları, internet, HPWREN, FESB ve FireNet.
- **Projeyle ilişkisi:** Projenin ana veri seti. Yinelenenler elendikten sonra 16.619 görüntü kullanılıyor.

### 2. Mamadaliev, Touko, Kim, Kim (2024)
*ESFD-YOLOv8n: Early Smoke and Fire Detection Method Based on an Improved YOLOv8n Model.* Fire, 7(9), 303. DOI: [10.3390/fire7090303](https://doi.org/10.3390/fire7090303)

- **Amaç:** Gerçek zamanlı çalışacak kadar hafif ama daha doğru bir yangın ve duman dedektörü.
- **Veri:** D-Fire'ın 21.527 görüntüsü rastgele bölünmüş: 15.069 eğitim (%70), 4.305 doğrulama (%20), 2.153 test (%10).
- **Mimari değişiklikleri:**
  1. Kutu kaybı CIoU yerine WIoUv3 kullanılmış. Bu kayıp, odaklanmayı dinamik olarak ayarlayarak düşük kaliteli etiketlerin etkisini azaltıyor.
  2. Omurgadaki (backbone) C2f modülleri yerine residual blok konmuş.
  3. Boyun (neck) kısmında C2f yerine C2fGELAN kullanılmış.
  4. Ön eğitimli ağırlıklarla (`yolov8n.pt`) transfer öğrenme yapılmış.
- **Eğitim:** 300 epoch, batch 32, 640 piksel girdi, öğrenme hızı 0,01, patience 50. Donanım: RTX 4060 Ti.
- **Ablation tablosu** (her satır öncekine eklenen değişikliği gösterir):

  | Adım | P (%) | R (%) | mAP@0.5 (%) | Süre (ms) | FLOPs (B) |
  |---|---|---|---|---|---|
  | YOLOv8n | 77,8 | 70,0 | 77,4 | 0,8 | 8,1 |
  | + WIoUv3 | 79,1 | 70,6 | 78,0 | 0,8 | 8,1 |
  | + ResBlock | 80,8 | 71,0 | 78,8 | 0,9 | 10,6 |
  | + C2fGELAN | 81,2 | 71,2 | 79,3 | 1,0 | 10,8 |
  | + Transfer öğrenme | 80,1 | 72,7 | **79,4** | 1,0 | 10,8 |

- **Karşılaştırma:** YOLOv5n/s, YOLOX-tiny, YOLOv6n, YOLOv7-tiny, GELAN, YOLOv9 ve YOLOv10n'den mAP@0.5'te 1–3 puan daha iyi.
- **Yazarların belirttiği sınırlılıklar:** Sınıf dengesizliği ve çok sayıda arka plan görüntüsü; uzak nesnelerde gereksiz kutular; bulut ve gün batımı gibi benzer görüntülerin karıştırılması.
- **Değerlendirme:** Yinelenen görüntüler elenmeden rastgele bölme yapılmış, bu yüzden test sonuçları iyimser olabilir. Metrik nesne tespiti mAP'i olduğu için görüntü sınıflandırma sonuçlarıyla doğrudan karşılaştırılamaz.

### 3. Kumar, Perrusquía, Al-Rubaye, Guo (2024)
*Wildfire and smoke early detection for drone applications: A light-weight deep learning approach.* Engineering Applications of Artificial Intelligence, 136, 108977. DOI: [10.1016/j.engappai.2024.108977](https://doi.org/10.1016/j.engappai.2024.108977)

- **Amaç:** Belleği sınırlı drone'larda çalışabilen hafif segmentasyon modelleri geliştirmek.
- **Görev:** Yangın ve duman için ayrı ayrı ikili piksel segmentasyonu.
- **Önerilen mimariler:** DeepLabv3+'ın ResNet-50 omurgası yerine iki hafif seçenek:
  1. **MobileViT-DeepLabv3+:** CNN ile vision transformer karışımı; S ve XS boyutları.
  2. **MobileNetV3-DeepLabv3+:** Decoder'daki düşük seviyeli özellikler kaldırılmış, ASPP bloğunda ayrılabilir (separable) evrişim kullanılmış.
- **Veri:**

  | Veri seti | Görüntü | Hedef piksel oranı |
  |---|---|---|
  | Corsican (büyük yangınlar) | 1.122 | %17,09 |
  | FLAME (küçük yangınlar) | 2.003 | %0,58 |
  | Birleşik (Corsican + FLAME) | 3.125 | %1,95 |
  | Duman (SMOKE5K gerçek görüntüler + AI For Mankind) | 2.848 | %12,44 |

  AI For Mankind görüntülerinin duman maskeleri LabelMe aracıyla elle çizilmiş. SMOKE5K'nın sentetik görüntüleri kullanılmamış.
- **Eğitim:**
  - Bölme %70 / %15 / %15, girdi 512×512.
  - Adam, öğrenme hızı 1e-4, batch 16.
  - 30 epoch (MobileNetV3 için 100).
  - Kayıp fonksiyonları: Dice, Tversky, Focal Tversky, BCE, Combo.
  - Donanım: 4 adet A100 GPU.
- **Sonuçlar (IoU):**

  | Omurga | Parametre | Corsican | FLAME | Birleşik | Duman |
  |---|---|---|---|---|---|
  | MobileViT_S | 6,6 M | 0,877 | 0,799 | 0,817 | 0,744 |
  | MobileViT_XS | 3,1 M | 0,872 | 0,801 | **0,870** | 0,742 |
  | MobileNetV3 | 3,5 M | 0,882 | 0,811 | 0,834 | 0,759 |
  | ResNet-50 | 17,9 M | 0,889 | 0,814 | 0,837 | **0,809** |

- **Önemli bulgu:** Yalnızca Corsican ile eğitilen model küçük ve kapalı (occluded) yangınları kaçırıyor. Yalnızca FLAME ile eğitilen model büyük yangınlarda zorlanıyor. Birleşik veri iki önyargıyı da dengeliyor.
- **Gelecek çalışma:** Yangın ve dumanı tek modelde birlikte segmente etmek; fizik bilgisini modele kısıt olarak eklemek.
- **Projeyle ilişkisi:** AI For Mankind'ın kullanımına ve bağımsız test fikrine dayanak. Küçük ve büyük yangın önyargısı, projedeki "yangın kutularının %63'ü görüntünün %1'inden küçük" bulgusuyla örtüşüyor.

### 4. Shamsoshoara, Afghah, Razi, Zheng, Fulé, Blasch (2021)
*Aerial imagery pile burn detection using deep learning: The FLAME dataset.* Computer Networks, 193, 108001. DOI: [10.1016/j.comnet.2021.108001](https://doi.org/10.1016/j.comnet.2021.108001)

- **Veri toplama:** 16 Ocak 2020'de Flagstaff (Arizona) yakınlarında yapılan kontrollü yığın yakma sırasında çekilmiş.
  - Eğitim görüntüleri: DJI Matrice 200 + Zenmuse X4S.
  - Test görüntüleri: DJI Phantom 3.
  - Termal görüntüler: FLIR Vue Pro R (Fusion, WhiteHot ve GreenHot paletleri).
- **Veri boyutu:**
  - Sınıflandırma eğitimi: 39.375 kare (25.018 yangın, 14.357 yangın yok).
  - Sınıflandırma testi: 8.617 kare (5.137 yangın, 3.480 yangın yok).
  - Segmentasyon: 2.003 maske.
- **Sınıflandırma:** Küçültülmüş Xception; 254×254 girdi, Adam, öğrenme hızı 0,001, 40 epoch, batch 32.

  | Küme | Doğruluk |
  |---|---|
  | Eğitim | %96,79 |
  | Doğrulama | %94,31 |
  | Test | **%76,23** |

  Yazarlar test başarısının düşmesini, test görüntülerinin farklı bir drone ve kamerayla çekilmiş olmasına bağlıyor.
- **Segmentasyon:** U-Net (ELU aktivasyonu ve dropout eklenmiş). Precision %91,99, recall %83,88, F1 %87,75, IoU %78,17.
- **Sınırlılıklar:** Yalnızca tek bir yakmadan alınmış veri; az sayıda maske. Makale kendi içinde de tutarsız: özette "ANN" diyor, metinde Xception kullanıyor.
- **Projeyle ilişkisi:** Duman etiketi olmadığı için kapsam dışı bırakıldı. Doğrulama ile test arasındaki fark, bağımsız test setinin gerekliliğini gösteriyor.

### 5. Joshi, Kumar, Patil, Kamat, Kolhar, Kotecha (2024)
*Deep learning with ensemble approach for early pile fire detection using aerial images.* Frontiers in Environmental Science, 12. DOI: [10.3389/fenvs.2024.1440396](https://doi.org/10.3389/fenvs.2024.1440396)

- **Görev:** FLAME'in 4 paletinde (normal RGB, termal Fusion, WhiteHot, GreenHot) ikili sınıflandırma.
- **Klasik ML:** Piksel değerleri, 4 Gabor filtresi ve Sobel kenar özellikleriyle RF ve SVM.

  | Palet | RF test doğruluğu | SVM test doğruluğu |
  |---|---|---|
  | Fusion | 0,995 | 0,981 |
  | WhiteHot | 0,996 | 0,983 |
  | GreenHot | 0,958 | 0,912 |

- **Transfer öğrenme:** AlexNet, InceptionV3, Inception-ResNetV2, VGG16 ve ResNet50V2; 227 piksel girdi, Adamax.

  | Palet | Test görüntüsü | En iyi model | Test doğruluğu |
  |---|---|---|---|
  | Fusion | 218 | ResNet50V2 | %99,08 |
  | WhiteHot | 214 | InceptionV3 | %99,86 |
  | GreenHot | 218 | InceptionV3 | %99,89 |
  | Normal RGB | yaklaşık 12.350 | VGG16 | %77,00 |

- **Topluluk (ensemble):** Normal RGB'de VGG16 özellikleri üzerine LR + RF + Naive Bayes oylaması kurulmuş; test doğruluğu %91.
- **Değerlendirme:**
  - Termal test setleri küçük ve her biri tek bir videodan geliyor; ardışık kareler eğitim ile test arasında sızıntıya yol açabilir.
  - RGB'de eğitim ve test farklı drone'lardan geldiği için başarı %77'ye düşüyor.
  - Bazı tablo değerleri tutarsız (örneğin bir satırda P = 0,78 ve R = 0,45 iken F1 = 1,00).
  - Özetteki "ResNet50V2 + InceptionV3 topluluğu" metinde yer almıyor.
- **Projeyle ilişkisi:** Klasik ML ile transfer öğrenme karşılaştırması tasarımına dayanak.

### 6. Toulouse, Rossi, Campana, Celik, Akhloufi (2017)
*Computer vision for wildfire research: An evolving image dataset for processing and analysis.* Fire Safety Journal, 92, 188–194. DOI: [10.1016/j.firesaf.2017.06.012](https://doi.org/10.1016/j.firesaf.2017.06.012)

- **Amaç:** Yangın pikseli tespit algoritmalarını karşılaştırmak için elle çizilmiş referans maskeleri (ground truth) olan açık bir veri tabanı oluşturmak.
- **İçerik:**
  - Dünyanın farklı yerlerinden toplanan 2.000'den fazla görüntüden seçilmiş **500 görünür görüntü**.
  - JAI AD-080GE kamerasıyla çekilmiş **100 çok kipli (görünür + yakın kızılötesi) görüntü**.
  - Yaklaşık 30'ar görüntülük **5 dizi**.
  - Her görüntü için uzman tarafından çizilmiş ikili maske.
- **Açıklayıcı etiketler (descriptor):** Yangın piksel oranı, baskın renk (kırmızı, turuncu, sarı-beyaz), duman örtüşmesi ve rengi, doku, mesafe, parlaklık, gece veya gündüz, bitki örtüsü, yanlış alarm yaratabilecek nesneler, bulut varlığı.
- **Örnek deney:** Üç renk kuralı yöntemi karşılaştırılmış. Phillips (2002) F = 0,91, Chen (2004) F = 0,88.
- **Not:** Veri tabanı zamanla büyümüştür. Sonraki çalışmalarda (ör. Kumar vd. 2024) "1.135 RGB ve 640 kızılötesi görüntü" ifadesi geçer. Bu sayılar 2017 makalesine değil, sonraki sürüme aittir.
- **Projeyle ilişkisi:** Duman etiketi bulunmadığı ve görüntü alanı farklı olduğu için kapsam dışı bırakıldı.

### 7. Şafak, Barışçı (2023)
*Derin öğrenme kullanılarak mobil cihazlar için gerçek zamanlı yangın ve duman tespiti.* Gazi Üniversitesi Mühendislik Mimarlık Fakültesi Dergisi, 38(4), 2179–2190. DOI: [10.17341/gazimmfd.1041091](https://doi.org/10.17341/gazimmfd.1041091)

- **Amaç:** Görüntüyü sunucuya göndermeden mobil cihazda gerçek zamanlı tespit yapmak.
- **Görev:** Başlıkta "yangın ve duman" geçse de ikili sınıflandırma (yangın / yangın yok).
- **Veri:** MIVIA Fire Detection, FIRESENSE ve CAIR Fire Detection Image veri setleri birleştirilmiş; toplam 43.355 görüntü.
  - Eğitim: 34.684 görüntü (16.773 yangın, 17.911 yangın yok).
  - Test: 8.671 görüntü (4.159 yangın, 4.512 yangın yok).
- **Yöntem:** ImageNet ağırlıklı MobileNet'in son 3 katmanı kaldırılmış; yerine düzleştirme katmanı ve 2 düğümlü yoğun katman eklenmiş. Parametre sayısı 4,25 M'den 3,23 M'ye düşmüş. Adam, öğrenme hızı 0,001, 9 epoch.
- **Sonuçlar:**

  | Model | Doğruluk | F1 | Parametre |
  |---|---|---|---|
  | Revize MobileNet | **%98,37** | %97,15 | 3,23 M |
  | EfficientNetB0 | %96,30 | %93,50 | 5,3 M |
  | PeleeNet | %95,71 | %95,88 | 2,8 M |
  | ShuffleNet | %95,40 | %94,91 | 5,2 M |
  | MobileNetV2 | %93,69 | %91,08 | 3,5 M |
  | MobileNet (orijinal) | %92,26 | %90,24 | 4,25 M |
  | NASNetMobile | %87,22 | %86,78 | 5,6 M |

  Çıkarım süresi 2,4 ms, model boyutu 17 MB.
- **Değerlendirme:** MIVIA ve FIRESENSE video veri setleridir. Kareler rastgele bölündüğü için aynı videonun kareleri hem eğitimde hem testte olabilir. Bağımsız bir test seti yok.
- **Projeyle ilişkisi:** Hafif ağların mobil ve kenar cihazlarda kullanılabileceğini gösteriyor; projedeki MobileNetV3 seçimini destekliyor.

---

## Kaynakça notu

Faz 1 raporunda D-Fire için verilen `10.1007/s11042-022-13580-x` DOI'si başka bir çalışmaya aittir: Wu vd., *A dataset for fire and smoke object detection*, Multimedia Tools and Applications. D-Fire'ın doğru kaynağı yukarıdaki 1 numaralı çalışmadır.
