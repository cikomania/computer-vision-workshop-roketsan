# LEVEL UP AI | ROKETSAN AI Hackathon

## Workshop İçeriği

### 1. Görüntü İşleme

#### Proje Oluşturulması ve Sanal Ortam

```bash
py -3.12 -m venv venv
.\venv\Scripts\activate
```

#### Kullanılan Kütüphaneler

```text
opencv-python
numpy
matplotlib
```

```bash
pip install -r requirements.txt
```

#### Proje Yapısı

```text
images/
outputs/
preprocessing.py
```

Görüntü okuma, yeniden boyutlandırma, crop, renk uzayı dönüşümleri, normalizasyon ve filtreleme işlemleri uygulanmıştır.

---

### 2. Transfer Learning ile Görüntü Sınıflandırma

4 farklı araç sınıfının sınıflandırılması üzerine bir görüntü sınıflandırma modeli geliştirilmiştir:

- Bus
- Car
- Motorcycle
- Truck
  
[Kaggle Vehicle Type Recognition](https://www.kaggle.com/datasets/kaggleashwin/vehicle-type-recognition) veri seti kullanılmıştır.

#### Kullanılan Kütüphaneler

```text
tensorflow
pillow
```

> TensorFlow kurulumu sırasında Python sürümüne dikkat edilmiştir. Çalışmada Python 3.12 kullanılmıştır.

#### Model

ImageNet üzerinde önceden eğitilmiş MobileNetV2 modeli kullanılarak transfer learning uygulanmıştır.

#### Proje Yapısı

```text
Dataset/
├── Bus/
├── Car/
├── Motorcycle/
└── Truck/
classification.py
```

---

### 3. YOLO

YOLO kullanılarak araçların görüntü ve video üzerinde tespit edilmesi ve takip edilmesi gerçekleştirilmiştir.

Veri seti: [VisDrone2019-MOT-val](https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view)

#### Kullanılan Kütüphane

```text
ultralytics
```

#### Model ve Takip

```text
YOLO11n
ByteTrack
```

#### Proje Yapısı
```text
video_sequences/
bytetrack.yaml
detection_tracking.py
```

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Matplotlib
- TensorFlow / Keras
- MobileNetV2
- YOLO
- Ultralytics
- ByteTrack
