# 🤟 BSL-Fingerspelling-Recognizer

A real-time British Sign Language (BSL) fingerspelling recognition system built using a lightweight deep learning model (MobileNetV2) and deployed via a Flask-based GUI. This solution is optimized for children with hearing impairments and designed to run efficiently on resource-constrained devices such as Raspberry Pi or smartphones.

---

## 📌 Project Overview

This project addresses the language deprivation problem in children with hearing impairments by providing an AI-based fingerspelling recognition system. The model takes live webcam input and predicts fingerspelled characters (A–Z) in real-time with high accuracy.

---

## 🗃️ Dataset

- **Source**: [Kaggle BSL Alphabet Dataset](https://www.kaggle.com/datasets/)
- **Classes**: 24 static BSL letters (2 missing from the full alphabet)
- **Structure**: Split into `train/` and `test/` directories, with class folders
- **Preprocessing**:
  - Resized all images to **224x224**
  - Normalized pixel values to **[0, 1]**
  - Applied **data augmentation**:
    - Rotation: ±15°
    - Zoom: ±20%
    - Shear: ±20%
    - Horizontal flips

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
```

---

## 🧠 Model Architecture

The model uses **MobileNetV2** as the feature extractor, fine-tuned for BSL classification. Additional dense layers are added to improve classification performance while avoiding overfitting.

| Layer                    | Output Shape    | Parameters  |
|-------------------------|------------------|-------------|
| MobileNetV2 (frozen)    | (None, 7, 7, 1280) | 2,257,984   |
| GlobalAveragePooling2D  | (None, 1280)     | 0           |
| Dropout (rate=0.3)      | (None, 1280)     | 0           |
| Dense (ReLU, 128 units) | (None, 128)      | 163,968     |
| Dropout (rate=0.3)      | (None, 128)      | 0           |
| Output Dense (Softmax)  | (None, 24)       | 3,096       |

**Total trainable parameters**: 167,064  
**Model size**: ~9.25 MB

### 📷 Model Summary Screenshot  
![Model Summary](images/Screenshot_2025-04-12_092230.png)

---

## 📈 Training & Evaluation

The model was trained for **10 epochs** with early stopping. Below is a summary of the training progress:

```plaintext
Epoch 1/10: accuracy=0.02 → val_accuracy=0.08
Epoch 5/10: accuracy=0.28 → val_accuracy=0.47
Epoch 9/10: accuracy=0.44 → val_accuracy=0.83
Epoch 10/10: accuracy=0.47 → val_accuracy=0.83
```

### ✅ Final Validation Accuracy: **83.4%**

### 📷 Training Logs Screenshot  
![Training Logs](images/Screenshot_2025-04-12_092256.png)

---

## 📊 Confusion Matrix & Metrics

This matrix illustrates how well each letter was recognized. Most confusion occurs with visually similar signs.

![Confusion Matrix] images/Screenshot_2025-04-12_092324.png

| Metric      | Value |
|-------------|-------|
| Accuracy    | 0.83  |
| Precision   | 0.87  |
| Recall      | 0.84  |
| F1-Score    | 0.83  |
| Classes     | 24    |
| Samples     | 236   |

---

## 🖥️ Flask GUI

A simple GUI built using **Flask** allows users to:
- Start/stop webcam stream
- View live predictions
- Annotate the video feed with the detected character

```bash
# To run the app locally
python app.py
```

Then visit: `http://localhost:5000`

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/BSL-Fingerspelling-Recognizer.git
cd BSL-Fingerspelling-Recognizer
pip install -r requirements.txt
python app.py
```

---

## 🚀 Future Improvements

- 🔁 Dynamic gesture & sentence prediction  
- 📱 TensorFlow Lite deployment for Android  
- 🌐 Multilingual sign language support  
- 🧠 Sequence modeling with RNNs for continuous signing  

---

## 📚 References

1. [Kaggle BSL Dataset](https://www.kaggle.com/datasets/)
2. Howard et al., *MobileNets: Efficient CNNs for Mobile Vision*, 2017  
3. [TensorFlow Lite](https://www.tensorflow.org/lite/guide)  
4. [Flask Documentation](https://flask.palletsprojects.com/)

---

## 🤝 Contributing

Feel free to fork this repo, submit PRs, or open issues for feature suggestions and bugs.

---

© 2025 Alif Sathar. All rights reserved.
Unauthorized use or distribution is strictly prohibited.
