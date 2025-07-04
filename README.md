> 🇬🇧 English version below  
> 🇵🇱 Polska wersja poniżej

# Detektor Emocji w Czasie Rzeczywistym na Podstawie Punktów Charakterystycznych Twarzy

**Cel projektu**: stworzenie lekkiego, szybkiego i prywatnego systemu rozpoznawania emocji w czasie rzeczywistym przy użyciu tylko 13 najważniejszych punktów twarzy (landmarków), bez potrzeby przetwarzania całego obrazu twarzy.

---

## Główna idea

- Wykorzystuje **MediaPipe FaceMesh** do wyodrębnienia 13 kluczowych punktów twarzy.
- Emocje są rozpoznawane na podstawie współrzędnych `(x, y, z)` tych punktów.
- Model to niewielka **sieć neuronowa** oparta na TensorFlow/Keras.
- Dane wejściowe: `39 wartości` (13 punktów × 3 współrzędne).
- Prywatność: **obrazy nie są zapisywane**, tylko liczby.

---

## Obsługiwane emocje

- 😄 Szczęśliwy (Happy)  
- 😢 Smutny (Sad)  
- 😠 Zły (Angry)  
- 😐 Neutralny (Neutral)

---

## Struktura projektu

| Plik / Folder                         | Opis |
|--------------------------------------|------|
| `collect_landmark_emotions_light.py` | Zbieranie danych z kamery (naciśnięcie klawiszy 1–4) |
| `lightweight_landmark_emotion_trainer.py` | Trenowanie modelu na zebranych plikach JSON |
| `live_landmark_emotion_detector.py`  | Wykrywanie emocji z kamery w czasie rzeczywistym |
| `example_landmark_data/`             | Przykładowe dane JSON |
| `requirements.txt`                   | Wymagane biblioteki |

---

## Jak uruchomić

### 1. Zainstaluj wymagane biblioteki

```bash
pip install -r requirements.txt
```

### 2. Zbieranie własnych danych

```bash
python collect_landmark_emotions_light.py
```

- Skrypt automatycznie utworzy folder `landmark_data/`
- Naciśnij:
  - `1` — szczęśliwy
  - `2` — smutny
  - `3` — zły
  - `4` — neutralny
- Zbierz **co najmniej 100 przykładów każdej emocji**

---

### 3. Trenowanie modelu

```bash
python lightweight_landmark_emotion_trainer.py
```

- Wygenerowane zostaną dwa pliki:
  - `landmark_model.h5` — zapisany model
  - `label_encoder.pkl` — zakodowane etykiety

---

### 4. Uruchomienie detektora emocji

```bash
python live_landmark_emotion_detector.py
```

- Uruchomi się kamera
- Model będzie wyświetlał emocję w czasie rzeczywistym

---

## 🛠 Wymagania

- Python 3.8+
- TensorFlow
- OpenCV
- MediaPipe
- scikit-learn

---

## Autor

- Artem Koshovyi  

(Uniwersytet WSB we Wrocławiu, 2025)

---

---

# Real-Time Emotion Detector with Face Landmarks

**Project goal**: Build a lightweight, fast, and privacy-focused real-time emotion recognition system using only 13 facial landmarks instead of the whole face image.

---

## Main idea

- Uses **MediaPipe FaceMesh** to extract 13 key facial points.
- Emotions are recognized based on `(x, y, z)` coordinates of those points.
- The model is a small **neural network** built with TensorFlow/Keras.
- Input data: `39 values` (13 points × 3 coordinates).
- Privacy: **no images are stored**, only numerical data.

---

## Supported emotions

- 😄 Happy  
- 😢 Sad  
- 😠 Angry  
- 😐 Neutral

---

## Project structure

| File / Folder                         | Description |
|--------------------------------------|-------------|
| `collect_landmark_emotions_light.py` | Collect data from webcam using keys 1–4 |
| `lightweight_landmark_emotion_trainer.py` | Train the model using collected JSON files |
| `live_landmark_emotion_detector.py`  | Real-time emotion detection via webcam |
| `example_landmark_data/`             | Sample labeled landmark data |
| `requirements.txt`                   | Required dependencies |

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect your own data

```bash
python collect_landmark_emotions_light.py
```

- The script will create a `landmark_data/` folder
- Press:
  - `1` — happy
  - `2` — sad
  - `3` — angry
  - `4` — neutral
- Collect **at least 100 samples per emotion**

---

### 3. Train the model

```bash
python lightweight_landmark_emotion_trainer.py
```

- Will generate two files:
  - `landmark_model.h5` — the trained model
  - `label_encoder.pkl` — saved label encoder

---

### 4. Run live emotion detection

```bash
python live_landmark_emotion_detector.py
```

- Opens your camera
- The model displays predicted emotion in real-time

---

## 🛠 Requirements

- Python 3.8+
- TensorFlow
- OpenCV
- MediaPipe
- scikit-learn

---

## Author

- Artem Koshovyi   

(Wrocław University WSB, 2025)
