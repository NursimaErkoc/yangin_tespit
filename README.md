# Derin Öğrenme Tabanlı Orman Yangını ve Duman Tespiti

4 sınıflı görüntü sınıflandırma: yok / yangın / duman / ikisi.
Veri: D-Fire (eğitim/doğrulama/test), AI For Mankind Wildfire Smoke (bağımsız duman testi).

## Kurulum
    pip install -r requirements.txt

## Veri
1. D-Fire'ı https://github.com/gaia-solutions-on-demand/DFireDataset adresinden indirip `data/D-Fire/` altına açın
   (`train/images`, `train/labels`, `test/images`, `test/labels`).
2. `python -m src.prepare_data --dfire_dir data/D-Fire --out data/splits.csv`

Eğitim Colab'de yapılır (yerelde GPU yok).
