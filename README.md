# Face Mask Detection (MobileNetV2 + OpenCV) 😷

**Ringkasan singkat**
Proyek ini mendeteksi wajah dan mengklasifikasikan apakah seseorang memakai masker atau tidak menggunakan MobileNetV2 (transfer learning) dan OpenCV Haar Cascade untuk deteksi wajah. Termasuk notebook pelatihan (`uas_viskom.ipynb`) dan aplikasi Streamlit (`streamlit_app.py`).

---

## 🔧 Persyaratan Sistem

- OS: Linux / macOS / Windows
- Python 3.8+ (disarankan 3.10–3.11)
- RAM & Storage sesuai dataset dan model (GPU optional untuk training)

Rekomendasi paket (tambahkan ke `requirements.txt`):

```
tensorflow>=2.11
streamlit
opencv-python
pillow
numpy
pandas
matplotlib
seaborn
scikit-learn
kaggle
```

> Tip: Gunakan virtual environment (venv / conda) untuk isolasi.

---

## 🛠️ Instalasi

1. Clone repository:

```bash
git clone <repo-url>
cd uas-visi-komputer
```

2. Buat dan aktifkan virtual env:

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate     # Windows (PowerShell)
```

3. Install requirements:

```bash
pip install -r requirements.txt
# Jika belum ada requirements.txt, install manual:
# pip install tensorflow streamlit opencv-python pillow numpy pandas matplotlib seaborn scikit-learn kaggle
```

---

## 🚀 Menjalankan Aplikasi (Streamlit)

Aplikasi Streamlit siap pakai di `streamlit_app.py`.

1. Pastikan model telah tersedia di `data/models/` (mis. `best_model.h5` atau `final_model.h5`). Jika belum, jalankan training di notebook.
2. Jalankan aplikasi:

```bash
streamlit run streamlit_app.py
```

3. Buka URL lokal yang ditampilkan (biasanya http://localhost:8501).

Fitur:

- Upload gambar → deteksi wajah & klasifikasi masker
- Upload video → proses per-frame (dengan opsi skip frame untuk kecepatan)
- Sidebar untuk mengatur threshold confidennce

> Jika mengalami masalah terkait CUDA/GPU, ada opsi di top file untuk memaksa CPU: set `FORCE_CPU = True` atau jalankan `export CUDA_VISIBLE_DEVICES=-1`.

---

## 🧪 Melatih dan Mengevaluasi Model

Pelatihan dilakukan via `uas_viskom.ipynb` (Jupyter Notebook).

Langkah singkat:

1. Jalankan notebook dari JupyterLab / Colab / VS Code Notebook.
2. Ikuti sel untuk: download dataset (opsional via Kaggle), membuat struktur folder, preprocessing, augmentasi, membangun model, dan memulai training.
3. Model terbaik otomatis disimpan ke `data/models/best_model.h5` dan model akhir ke `data/models/final_model.h5`.
4. Hasil evaluasi, grafik, confusion matrix, dan laporan klasifikasi disimpan di folder `results/`.

Parameter penting:

- IMG_SIZE = 224
- BATCH_SIZE = 32
- EPOCHS (default dalam notebook = 20)

---

## 📁 Struktur Project (singkat)

```
README.md
streamlit_app.py
uas_viskom.ipynb
data/
  └─ models/
      ├─ best_model.h5
      └─ final_model.h5
results/
```