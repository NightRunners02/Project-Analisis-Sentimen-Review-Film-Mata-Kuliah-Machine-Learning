# 🎬 **Analisis Sentimen Review Film – Text Classification**

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <br>
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge\&logo=python\&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?style=for-the-badge\&logo=tensorflow\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-red?style=for-the-badge\&logo=pytorch\&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)
![LoRA](https://img.shields.io/badge/PEFT-LoRA-green?style=for-the-badge)

<br>

**Analisis Sentimen Review Film** adalah proyek perbandingan metode NLP mulai dari Machine Learning, Deep Learning, hingga Transformer & LoRA.
Menggunakan dataset **IMDB Movie Reviews (50K data)** untuk eksperimen klasifikasi sentimen *positive/negative*.

📌 **Live Notebook / Colab**
*(Opsional: Masukkan link Colab jika ada)*

</div>

---

## ✨ **Fitur Utama**

* 📦 Dataset IMDB 50.000 review
* 🔧 Preprocessing lengkap & otomatis
* 🧠 Perbandingan 5 model:

  * Naive Bayes (TF-IDF)
  * SVM
  * LSTM
  * DistilBERT
  * LoRA Fine-Tuned (PEFT)
* 📊 Evaluasi lengkap:

  * Accuracy, Precision, Recall, F1
  * Confusion Matrix
  * Grafik perbandingan
* 🚀 Siap dijalankan di Google Colab
* 💾 Export hasil ke tabel `.csv` & gambar `.png`
* 🔥 LoRA sebagai model terbaik & efisien

---

## 📊 **Hasil Perbandingan Model**

| Model              | Pendekatan       | Accuracy   |
| ------------------ | ---------------- | ---------- |
| 🟦 Naive Bayes     | Machine Learning | **0.8405** |
| 🟪 SVM             | Machine Learning | **0.8872** |
| 🟩 LSTM            | Deep Learning    | **0.8442** |
| 🟨 DistilBERT      | Transformer      | **0.9184** |
| 🟧 LoRA Fine-Tuned | PEFT             | **0.9281** |

⚡ **LoRA menang:** akurasi tinggi + waktu training cepat + parameter sedikit.

---

## 📂 **Struktur Folder**

```
Text-Classification-Comparison/
│
├── Dataset Raw/
│   ├── imdb_raw_train.csv
│   ├── imdb_raw_test.csv
│
├── Models/
│
├── Notebooks/
│   ├── Text_Classification_Comparison.ipynb
│   ├── LSTM_Training.ipynb
│   ├── BERT_LoRA_Training.ipynb
│
├── Result/
│   ├── accuracy_comparison.png
│   ├── confusion_matrix_nb.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_lstm.png
│   ├── confusion_matrix_bert.png
│   ├── confusion_matrix_lora.png
│   ├── training_history_lstm.png
│   ├── performance_table.csv
│
├── LICENSE
└── README.md
```

---

## 🧠 **Ringkasan Metode**

### **1️⃣ Naive Bayes**

* Metode baseline
* Cepat, ringan
* Menggunakan **TF-IDF**

---

### **2️⃣ Linear SVM**

* TF-IDF + LinearSVC
* Paling stabil untuk model klasik
* Akurasi tinggi dan robust

---

### **3️⃣ LSTM**

* Embedding → Bidirectional LSTM
* Memahami konteks sekuens
* Cocok untuk teks panjang

---

### **4️⃣ DistilBERT**

* Pretrained Transformer
* Lebih efisien dari BERT
* Performa sangat kuat

---

### **5️⃣ LoRA Fine-Tuning**

* Parameter Efficient Fine-Tuning
* Melatih hanya *adapter layers*
* Hemat GPU, training cepat
* Hasil terbaik di eksperimen

---

## 🚀 **Cara Menjalankan Proyek (Google Colab)**

### **1. Clone Repo**

```bash
!git clone https://github.com/NightRunners02/Text-Classification-Comparison.git
%cd Text-Classification-Comparison
```

### **2. Install Dependency**

```bash
!pip install -r requirements.txt
```

### **3. Download Dataset IMDB**

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("imdb")

pd.DataFrame(dataset["train"]).to_csv("imdb_raw_train.csv", index=False)
pd.DataFrame(dataset["test"]).to_csv("imdb_raw_test.csv", index=False)
```

### **4. Jalankan Notebook**

* `Text_Classification_Comparison.ipynb`
* `LSTM_Training.ipynb`
* `BERT_LoRA_Training.ipynb`

---

## 🎥 **Demo Visualisasi Hasil**

<div align="center">

> Tambahkan file PNG di folder `/Result` lalu update path-nya.

### 📌 **Perbandingan Akurasi**

![Comparison Results](Result/Comparison%20Results/comparison_results.png)


### 📌 **Confusion Matrix Tiap Model**

NB – SVM – LSTM – DistilBERT – LoRA

</div>

---

## 🧩 **Contoh Kode – Naive Bayes**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nb = MultinomialNB()
nb.fit(X_train_tfidf, train_labels)

preds = nb.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(test_labels, preds))
print(classification_report(test_labels, preds))
```

---

## 🛠 **Teknologi yang Digunakan**

* Python
* Scikit-learn
* TensorFlow / Keras
* PyTorch
* HuggingFace Transformers
* PEFT (LoRA)
* Matplotlib / Seaborn

---

## 📄 **Lisensi**

MIT License - boleh dimodifikasi, digunakan, dan didistribusikan.

---

## 🤝 **Kontribusi**

1. Fork repo
2. Buat branch fitur
3. Commit → Push → Pull Request

---

<div align="center">

Dibuat dengan ❤️ oleh **Night (NightRunners02)**
Jika proyek ini bermanfaat, jangan lupa kasih ⭐ di repository!

</div>

---

<details> 
  <summary>
  <h2> ⛓️‍💥 Misc / Lain-lain</h2> 
  </summary>

<p>
<div align="center">
<h3>
  🗣️ Powered By:
</h3>
<img src="https://awesome-svg.vercel.app/card/card_2?name=NightRunners02&summary=Newbie%20Developer&style=nameColor:rgba(223,255,0,1);summaryColor:rgba(57,255,20,1);backgroundColor:rgba(0,0,0,1);" />

---
<h3>
  🌠 Starred:
</h3>
  
[![Stargazers repo roster for @NightRunners02/](https://reporoster.com/stars/NightRunners02/Project-Analisis-Sentimen-Review-Film-Mata-Kuliah-Machine-Learning)](https://github.com/NightRunners02/Project-Analisis-Sentimen-Review-Film-Mata-Kuliah-Machine-Learning/stargazers)

---
<h3>
  🪐 Forked:
</h3>

[![Forkers repo roster for @NightRunners02/](https://reporoster.com/forks/NightRunners02/Project-Analisis-Sentimen-Review-Film-Mata-Kuliah-Machine-Learning)](https://github.com/NightRunners02/Project-Analisis-Sentimen-Review-Film-Mata-Kuliah-Machine-Learning/network/members)

---
<h3>
  💫 Star History:
</h3>

[![Star History Chart](https://api.star-history.com/svg?repos=NightRunners02/Project-Analisis-Sentimen-Review-Film-Mata-Kuliah-Machine-Learning&type=Date)](https://star-history.com/#Project-Analisis-Sentimen-Review-Film-Mata-Kuliah-Machine-Learning/Portofolio_Simple&Date)

</p>
</div>
</details>
