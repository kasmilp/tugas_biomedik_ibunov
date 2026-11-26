# 📦 KOLEKSI DATASET & REFERENSI JURNAL

# Topik: Segmentasi & Klasifikasi Sel Darah Putih (WBC)

# Menggunakan U-Net dan CNN

**Topik 18** — Informatika Biomedik (Biomedical Image Processing)

---

## A. DATASET

### A.1. Mendeley — Blood Cell Images (✅ TERVERIFIKASI AKTIF)

| Atribut                   | Detail                                                                                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nama**            | A dataset for microscopic peripheral blood cell images for development of automatic recognition systems                                               |
| **Penulis**         | Andrea Acevedo, Anna Merino, Santiago Alférez, Ángel Molina, Laura Boldú, José Rodellar (Hospital Clinic de Barcelona)                            |
| **DOI**             | 10.17632/snkd93bnjr.1                                                                                                                                 |
| **Link akses**      | https://data.mendeley.com/datasets/snkd93bnjr/1                                                                                                       |
| **Jumlah data**     | 17.092 citra individu sel normal                                                                                                                      |
| **Ukuran & format** | 360 × 363 px, format JPG                                                                                                                             |
| **Kelas**           | 8 kelompok: neutrofil, eosinofil, basofil, limfosit, monosit, granulosit imatur (promielosit, mielosit, metamielosit), eritroblas, trombosit/platelet |
| **Anotasi**         | Oleh ahli patologi klinis                                                                                                                             |
| **Lisensi**         | CC BY 4.0                                                                                                                                             |
| **Kegunaan**        | Utama untuk **tahap klasifikasi** (ada 5 jenis WBC utama; bisa pakai 5 kelas atau semua 8)                                                      |

**Status**: ✅ AKTIF (sudah diverifikasi lewat akses langsung). Bisa langsung diunduh.

---

### A.2. BCCD — Blood Cell Count Dataset (untuk Segmentasi)

| Atribut               | Detail                                                           |
| --------------------- | ---------------------------------------------------------------- |
| **Nama**        | Blood Cell Count Dataset (BCCD)                                  |
| **Sumber**      | Repositori GitHub `Shenggan/BCCD` + beragam mirror di Kaggle   |
| **Jumlah data** | ±364 citra dengan mask anotasi                                  |
| **Isi**         | Sel darah merah (RBC), sel darah putih (WBC), dan platelet       |
| **Format**      | Gambar + mask biner untuk segmentasi                             |
| **Kegunaan**    | Utama untuk **tahap segmentasi** (punya ground-truth mask) |

**Status**: ⚠️ Kaggle login-gated (anti-bot). Cari di Kaggle dengan kata kunci **"BCCD"**. Link rusak di tool otomatis karena butuh login, tetapi berfungsi di browser biasa.

---

### A.3. GitHub Mirror — BCCD Lengkap (✅ SUDAH DIUNDUH, Terverifikasi)

| Atribut                     | Detail                                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Nama**              | An Automatic Nucleus Segmentation and CNN Model based Classification Method of White Blood Cell                                      |
| **Pemilik**           | ParthaPratimBanik (GitHub)                                                                                                           |
| **Link**              | https://github.com/ParthaPratimBanik/An-Automatic-Nucleus-Segmentation-and-CNN-Model-based-Classification-Method-of-White-Blood-Cell |
| **Isi (aktual)**      | 15.216 gambar, ±386 MB                                                                                                              |
| **Segmentasi**        | BCCD Input (368 gambar) + Ground Truth (367 mask)                                                                                    |
| **Klasifikasi**       | TRAIN (10.206) + TEST (2.558): eosinofil, limfosit, monosit, neutrofil                                                               |
| **Dataset penunjang** | ALL-IDB2, CellaVision, JTSC (segmentasi + klasifikasi)                                                                               |

**Status**: ✅ **SUDAH DIUNDUH** ke folder `dataset/bccd_bankik/` dan dirapikan ke `dataset/bccd/` + `dataset/blood_cells/` melalui skrip `siapkan_dataset.py`.

> 💡 **Catatan**: Repo ini adalah kombinasi TEPAT sesuai topik kalian (segmentasi + klasifikasi WBC), jadi sangat cocok sebagai sumber utama. Bedanya dengan BCCD asli: kelas klasifikasi di repo ini **4 kelas** (tanpa basofil), sementara Kaggle 5 kelas menyediakan basofil.

---

### A.4. (Alternatif) Kaggle — Blood Cell Images (5 kelas WBC)

| Atribut               | Detail                                                        |
| --------------------- | ------------------------------------------------------------- |
| **Nama**        | Blood Cell Images (leukocyte classification)                  |
| **Jumlah data** | 12.500+ citra preprocessed                                    |
| **Kelas**       | 5 kelas WBC: neutrofil, limfosit, monosit, eosinofil, basofil |
| **Ukuran**      | 320 × 240 px                                                 |
| **Kegunaan**    | Alternatif dataset klasifikasi jika Mendeley kurang nyaman    |

**Status**: ⚠️ Kaggle login-gated. Cari dengan kata kunci **"Blood Cell Images"** di Kaggle.

---

### Ringkasan Strategi Dataset

| Tahap                        | Dataset Utama                    | Cadangan                      |
| ---------------------------- | -------------------------------- | ----------------------------- |
| **Segmentasi (U-Net)** | BCCD (punya mask)                | Mendeley (gambar, tanpa mask) |
| **Klasifikasi (CNN)**  | Mendeley (17.092 citra, 8 kelas) | Kaggle 5 kelas                |

> 💡 **Saran**: Mulai dari **Mendeley** yang sudah terbukti aktif (cukup untuk klasifikasi). Untuk segmentasi, unduh **BCCD** dari Kaggle lewat browser manual.

---

## B. REFERENSI JURNAL (Terverifikasi di PubMed)

Format: [No] Penulis (Tahun). Judul. *Jurnal*, Volume(Nomor):Halaman. DOI/PMID.

### B.1. Segmentasi WBC (U-Net)

1. Benaissa I, Zitouni A, Sbaa S, et al. (2025). Gated Attention-Augmented Double U-Net for White Blood Cell Segmentation. *J Imaging*, 11(11):386. PMID: 41295103.
2. Zhu M, Chen W, Sun Y, Li Z. (2023). Improved U-Net-based leukocyte segmentation method. *J Biomed Opt*, 28(4):045002. PMID: 37065646.
3. Fanous MJ, He S, Sengupta S, et al. (2022). White blood cell detection, classification and analysis using phase imaging with computational specificity (PICS). *Sci Rep*, 12(1):20043. PMID: 36414631.
4. Chen W, Zhu M. (2022). Leukocyte Segmentation Method Based on Adaptive Retinex Correction and U-Net. *Comput Math Methods Med*, 2022:9951582. PMID: 35832136.
5. Li D, Tang P, Zhang R, et al. (2021). Robust Blood Cell Image Segmentation Method Based on Neural Ordinary Differential Equations. *Comput Math Methods Med*, 2021:5590180. PMID: 34413897.
6. Li Z, Lu Y, Cao X, et al. (2022). White blood segmentation based on dual path and atrous spatial pyramid pooling. *Sheng Wu Yi Xue Gong Cheng Xue Za Zhi*, 39(3):471-479. PMID: 35788516.

### B.2. Klasifikasi WBC (CNN)

7. Cheuque C, Querales M, León R, et al. (2022). An Efficient Multi-Level Convolutional Neural Network Approach for White Blood Cells Classification. *Diagnostics*, 12(2):248. PMID: 35204339.
8. Islam O, Assaduzzaman M, Hasan MZ. (2024). An explainable AI-based blood cell classification using optimized convolutional neural network. *J Pathol Inform*, 15:100389. PMID: 39161471.
9. Ichim L, Popa AG, Popescu D. (2025). Optimal decision fusion-based systems of neural network ensembles for white blood cell classification. *Comput Biol Med*, 196:110942. PMID: 40834635.
10. Chen J, Fu L, Wei M, et al. (2024). Label-free white blood cells classification using a deep feature fusion neural network. *Heliyon*, 10(11):e31496. PMID: 38845979.
11. Almezhghwi K, Serte S. (2020). Improved Classification of White Blood Cells with the Generative Adversarial Network and Deep Convolutional Neural Network. *Comput Intell Neurosci*, 2020:6490479. PMID: 32695152.
12. Das SK, Islam KS, Neha TA, et al. (2021). Towards the Segmentation and Classification of White Blood Cell Cancer Using Hybrid Mask-Recurrent Neural Network and Transfer Learning. *Contrast Media Mol Imaging*, 2021:4954854. PMID: 34955694.
13. Jung C, Abuhamad M, Mohaisen D, et al. (2022). WBC image classification and generative models based on convolutional neural network. *BMC Med Imaging*, 22(1):94. PMID: 35596153.

### B.3. Referensi Fondasi (Wajib Dikutip)

14. Ronneberger O, Fischer P, Brox T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 234-241. (Paper asli U-Net)
15. He K, Zhang X, Ren S, Sun J. (2016). Deep Residual Learning for Image Recognition. *IEEE CVPR*, 770-778. (ResNet — dasar transfer learning)
16. Acevedo A, Merino A, Alférez S, et al. (2019). Recognition of peripheral blood cell images using convolutional neural networks. *Computer Methods and Programs in Biomedicine*, 180:105020. DOI: 10.1016/j.cmpb.2019.105020. (Paper sumber dataset Mendeley)

---

## C. CATATAN PENTING

- Semua referensi di bagian B.1 dan B.2 **diverifikasi langsung dari PubMed** (status: valid, dapat dicari via PMID).
- Referensi B.3 (fondasi) adalah paper klasik yang sudah mapan dan wajib ada di tinjauan pustaka.
- Dataset **Mendeley** sudah terverifikasi aktif; **BCCD & Kaggle** perlu login manual di browser.
- Format kutipan di atas sudah mengikuti gaya **Vancouver/AMA** yang umum dipakai jurnal kesehatan & komputer.
