[yangin_tespiti_on_rapor.md](https://github.com/user-attachments/files/32764008/yangin_tespiti_on_rapor.md)
# Derin Öğrenme Tabanlı Orman Yangını ve Duman Tespiti

*Uygulamalı Yapay Zeka Dersi — Dönem Projesi Önerisi*

**Proje türü:** Bireysel proje.

**Öğrenci:** Nursima Erkoç

---

## 1. Seçilen Problemin Detaylı Tanımı

Orman yangınları ilk dakikalarda küçük bir alanda başlarken müdahale edilirse kolayca söndürülebilir, ancak fark edilmeden geçen her dakika yangının yayılma hızını katlayarak artırır ve kontrolü neredeyse imkânsız hale getirir; bu yüzden erken tespit hem can kaybını hem de ekosistem zararını önlemede belirleyici rol oynar. Drone ve derin öğrenme tabanlı sistemler, insan gözlemine veya uydu taramasına göre çok daha hızlı ve sürekli izleme sağladığından, alevin henüz büyümeden, başlangıç aşamasında yani incecik bir duman halindeyken tespit edilmesini mümkün kılarak afet yönetiminde hayati bir erken uyarı katmanı oluşturur.

Bu projede, drone ve kamera ile elde edilen açık alan görüntülerinden yangın alevi / duman varlığının otomatik olarak tespit edilmeye çalışılacaktır. Problem, ikili veya çok sınıflı bir görüntü sınıflandırma/nesne tespiti problemi olarak tanımlanmaktadır. Verilen bir görüntüde:

- yangın alevi var mı?
- duman var mı?
- her ikisi de var mı?
- her ikisi de yok mu?

sorularına cevap verilecektir. Proje kapsamında klasik makine öğrenmesi yöntemleri ile en az üç farklı derin öğrenme mimarisi (ör. temel CNN, ResNet/EfficientNet gibi transfer öğrenme modelleri, MobileNet gibi hafif bir mimari) eğitilecek ve performansları literatürdeki sonuçlarla karşılaştırılacaktır.

---

## 2. Probleme Göre Seçilen Veri Setleri

Bu projede model eğitimi için 3 veri seti, bağımsız genelleme testi için ise ek bir 4. veri seti kullanılacaktır.

**FLAME (Fire Luminosity Airborne-based Machine learning Evaluation)** — Shamsoshoara, A., Afghah, F., Razi, A., Zheng, L., Fulé, P.Z., Blasch, E. (2021).
<https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs>

**D-Fire** — de Venâncio, P.V.A.B., Lisboa, A.C., Barbosa, A.V. (2022).
<https://github.com/gaia-solutions-on-demand/DFireDataset>

**Corsican Fire Dataset** — Toulouse, T., Rossi, L., Campana, A., Celik, T., Akhloufi, M.A. (2017).
<http://cfdb.univ-corse.fr>

**AI For Mankind Wildfire Smoke Dataset** (bağımsız genelleme test seti).
<https://github.com/aiformankind/wildfire-smoke-dataset>

---

## 3. Literatür Taraması / Kaynakça

Aynı problem için seçilen veri setlerinin kullanıldığı altı kaynak incelenmiştir:

- Shamsoshoara, A., Afghah, F., Razi, A., Zheng, L., Fulé, P.Z., Blasch, E. (2021). Aerial imagery pile burn detection using deep learning: The FLAME dataset. *Computer Networks*, 193, 108001.

**DOI/Link:** [10.1016/j.comnet.2021.108001](https://doi.org/10.1016/j.comnet.2021.108001)

**Kullanım:** FLAME veri setini oluşturan orijinal makale. Xception ile ikili sınıflandırma, U-Net ile piksel-düzeyi yangın segmentasyonu yapılmıştır.

- de Venâncio, P.V.A.B., Lisboa, A.C., Barbosa, A.V. (2022). A dataset for fire and smoke object detection. *Multimedia Tools and Applications*.

**DOI/Link:** [10.1007/s11042-022-13580-x](https://doi.org/10.1007/s11042-022-13580-x)

**Kullanım:** D-Fire veri setini oluşturan orijinal makale. YOLOv4 ile yangın/duman tespiti yapılmış, veri setinin sınıf dağılımı ve etiketleme protokolü detaylandırılmıştır.

- Mamadaliev, D., Touko, P.L.M., Kim, J., Kim, S. (2024). ESFD-YOLOv8n: Early Smoke and Fire Detection Method Based on an Improved YOLOv8n Model. *Fire*, 7(9), 303.

**DOI/Link:** [10.3390/fire7090303](https://doi.org/10.3390/fire7090303)

**Kullanım:** D-Fire üzerinde, C2f modüllerinin residual bloklarla ve WIoUv3 kayıp fonksiyonuyla geliştirildiği bir YOLOv8n varyantı önerilmiş, veri setinin sınıf dengesizliği tartışılmıştır.

- Kumar, A., Perrusquía, A., Al-Rubaye, S., Guo, W. (2024). Wildfire and smoke early detection for drone applications: A light-weight deep learning approach. *Engineering Applications of Artificial Intelligence*, 136, 108977.

**DOI/Link:** [10.1016/j.engappai.2024.108977](https://doi.org/10.1016/j.engappai.2024.108977)

**Kullanım:** Corsican ve FLAME, farklı yangın ölçeklerini dengelemek için birleştirilmiş; duman için SMOKE5K ve AI For Mankind kullanılmıştır. Tek veri setiyle eğitimin önyargı yarattığı, birleştirmenin bunu azalttığı gösterilmiştir.

- Joshi, D.D., Kumar, S., Patil, S., Kamat, P., Kolhar, S., Kotecha, K. (2024). Deep learning with ensemble approach for early pile fire detection using aerial images. *Frontiers in Environmental Science*, 12.

**DOI/Link:** [10.3389/fenvs.2024.1440396](https://doi.org/10.3389/fenvs.2024.1440396)

**Kullanım:** FLAME'in dört paleti üzerinde klasik ML (RF, SVM) ve transfer öğrenme modelleri (ResNet50V2, InceptionV3, AlexNet, VGG16) karşılaştırılmış; voting classifier ile doğruluk artırılmıştır.

- Toulouse, T., Rossi, L., Campana, A., Celik, T., Akhloufi, M.A. (2017). Computer vision for wildfire research: An evolving image dataset for processing and analysis. *Fire Safety Journal*, 92, 188–194.

**DOI/Link:** [10.1016/j.firesaf.2017.06.012](https://doi.org/10.1016/j.firesaf.2017.06.012)

**Kullanım:** Corsican veri setini oluşturan orijinal makale. 1122 RGB ve 640 kızılötesi görüntüden oluşan, piksel-düzeyi segmentasyon maskeli yangın veri setinin toplanma süreci anlatılmıştır.
