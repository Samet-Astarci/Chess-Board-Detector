
# Satranç Tahtası Tespit & Analiz Sistemi

Bu proje, bir satranç tahtasındaki taşların konumlarını gerçek zamanlı veya görsel üzerinden algılayan, tespit edilen durumu FEN formatına çevirerek **Stockfish** motoru ile en iyi hamleyi öneren bir analiz sistemidir.

## Özellikler

-  Görsel veya kamera üzerinden satranç tahtası tespiti
-  YOLOv8 ile satranç taşı algılama
-  Gerçek zamanlı perspektif düzeltme ve konumlandırma
-  FEN üretimi ve Stockfish ile hamle önerisi
-  Fazla/eksik taşlara karşı mantıksal uyarı sistemi
-  GUI arayüz (PyQt5) üzerinden rahat kullanım

---

## Klasör Yapısı

```
.
├── assets/                   # Satranç taşı ikonları (PNG)
│   ├── black_*.png
│   └── white_*.png
├── chessBoard1.jpg          # Örnek görseller
├── best4.pt                 # YOLOv8 model dosyası
├── chess_detector.py        # Ana uygulama (GUI + model entegrasyonu)
├── README.md                # Bu dosya
```

---

## Gerekli Kurulumlar

### Python Paketleri

```bash
pip install -r requirements.txt
```

İçeriği (örnek):
```txt
opencv-python
numpy
PyQt5
ultralytics
pillow
stockfish
pandas
```

### Ek Gereksinim

- [Stockfish](https://stockfishchess.org/download/) motoru: Windows için `.exe` yolunu `STOCKFISH_PATH` içinde belirtmeyi unutmayın.

```python
STOCKFISH_PATH = r"C:\Program Files\stockfish\stockfish-windows-x86-64-avx2.exe"
```

---

## Kullanım

### 1. Uygulamayı Başlat

```bash
python chess_detector.py
```

### 2. Görsel ile Analiz

- “Fotoğraf Seç” butonu ile bir satranç görseli yükleyin.
- 4 köşeyi sırayla tıklayın (Sol Üst → Sağ Üst → Sağ Alt → Sol Alt).
- “Analiz Et” butonu ile tespitleri başlatın.

### 3. Kamera ile Canlı Tarama

- “Kameradan Canlı” ile webcam veya IP kamerayı seçin.
- “Kare Al” butonuyla görüntüyü dondurun ve analiz yapın.

---

## Teknik Yapı

### Algılama

- `detect_pieces_ensemble()`: YOLO modelini birden fazla kez çalıştırarak kutuları gruplayıp ortalama sonucu üretir.

### Dönüşüm ve Konumlandırma

- `assign_pieces_dynamic()`: Perspesktif düzeltme sonrası taşları 8x8 tahtaya oturtur.
- `logical_piece_check()`: Mantıksal taş sınırları (örneğin 9 vezir olup olmadığını) denetler.

### Satranç Mantığı

- `board_to_fen()`: 8x8 tahtayı FEN formatına çevirir.
- `get_best_move_from_fen()`: Stockfish motoru ile en iyi hamleyi üretir.

---

## Örnek Çıktı

```
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1
En iyi hamle: e2e4
```


