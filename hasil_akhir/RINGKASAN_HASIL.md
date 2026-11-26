# 📂 HASIL AKHIR — Segmentasi & Klasifikasi Sel Darah Putih (WBC)

Topik 18 — Informatika Biomedik (Biomedical Image Processing)
Tanggal training: 27 November 2026 · Environment: Python 3.11, TensorFlow 2.13 (CPU)

---

## Struktur Folder

```
hasil_akhir/
├── RINGKASAN_HASIL.md          ← file ini
├── grafik/
│   ├── grafik_segmentasi.png   ← kurva Loss, IoU, Dice U-Net (15 epoch)
│   ├── grafik_klasifikasi.png  ← kurva Loss, Akurasi CNN (10 epoch)
│   ├── perbandingan_mask.png   ← citra asli vs ground truth vs prediksi
│   └── confusion_matrix.png    ← matriks prediksi 4 kelas
├── model/
│   ├── unet_wbc.keras          ← model U-Net terlatih (89 MB)
│   └── wbc_classifier.keras    ← model CNN klasifikasi (1,3 MB)
└── riwayat/
    └── riwayat_training.json   ← riwayat metrik per epoch (dipakai notebook)
```

---

## 1. Hasil Segmentasi (U-Net)

Konfigurasi: IMG_SIZE 128×128, batch 16, 15 epoch, loss BCE, optimizer Adam 1e-3,
data: 367 pasang gambar+mask BCCD (split 80/20).

| Metrik | Train (epoch 15) | Val terbaik |
|---|---|---|
| **IoU** | 0,870 | **0,894** (epoch 14) |
| **Dice** | 0,930 | **0,944** (epoch 14) |
| Val Loss | — | 0,0225 (epoch 14) |

- Naik cepat: epoch 1 Dice 0,016 → epoch 3 Dice 0,902 → stabil 0,91–0,94.
- **Target jurnal (Dice ≥ 0,85) TERCAPAI.**
- Tidak ada overfitting nyata (gap train/val < 0,02).

## 2. Hasil Klasifikasi (CNN ringan)

Konfigurasi: IMG_SIZE 128×128, batch 16, 10 epoch, steps_per_epoch 80,
data: 10.206 train + 2.550 val, 4 kelas (eosinophil, lymphocyte, monocyte, neutrophil).

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| eosinophil | 0,29 | 0,29 | 0,29 | 641 |
| lymphocyte | 0,41 | 0,80 | 0,55 | 626 |
| monocyte | 0,92 | 0,53 | 0,67 | 622 |
| neutrophil | 0,31 | 0,16 | 0,21 | 661 |
| **Akurasi total** | | | **0,44** | 2.550 |

### Diagnosis hasil (faktual, dari log training)
1. **Underfitting** — train acc 0,43 vs loss tersendat di ~1,19.
2. **Eksposur data rendah** — 10 epoch × 80 step = hanya ~10% dataset terlihat.
3. **Model terlalu ringan** — 3 conv block tanpa bobot pre-trained, lr konstan tanpa decay.

## 3. Rencana Perbaikan (urutan prioritas)
1. `EPOCHS_CLS=30`, `CLS_STEPS=200`, tambah `ReduceLROnPlateau` → ±15 menit CPU.
2. Ganti backbone dengan EfficientNetB0 pre-trained (butuh GPU/Colab).
3. Tambah BatchNorm + augmentasi lebih agresif.

---

## Cara Pakai Hasil Ini

```bash
# Load model di Python
import tensorflow as tf
unet  = tf.keras.models.load_model("hasil_akhir/model/unet_wbc.keras")
clf   = tf.keras.models.load_model("hasil_akhir/model/wbc_classifier.keras")
```

- **Notebook**: jalankan sel **9b** (grafik dari riwayat) dan **9c** (tampilkan PNG)
  di `training_wbc.ipynb` — membaca langsung dari `hasil_training/`.
- **Untuk jurnal**: pakai `grafik_segmentasi.png` sebagai kurva training utama,
  `perbandingan_mask.png` sebagai kualitatif hasil, `confusion_matrix.png` untuk evaluasi.

## Sumber Referensi
Daftar 16 jurnal terverifikasi (PubMed) + dataset: lihat `DATASET_DAN_REFERENSI.md`.