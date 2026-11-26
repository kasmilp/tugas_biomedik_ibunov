"""
=============================================================================
 Skrip Persiapan Dataset WBC
---------------------------------------------------------------------------
 Mengubah struktur dataset dari repo GitHub (ParthaPratimBanik) menjadi
 struktur standar yang dibaca oleh wbc_segmentation_classification.py:

   dataset/
   ├── bccd/
   │   ├── train_images/       (gambar BCCD)
   │   └── train_masks/        (mask ground-truth BCCD)
   └── blood_cells/
       ├── neutrophil/  (atau nama kelas lainnya)
       ├── lymphocyte/
       └── ...
=============================================================================
"""
import os
import shutil

# Sumber (hasil klon repo)
SRC_SEG_INPUT = "dataset/bccd_bankik/wbc_nucleus_seg_localz/Datasets/Input/BCCD"
SRC_SEG_MASK  = "dataset/bccd_bankik/wbc_nucleus_seg_localz/Datasets/Ground Truth/BCCD"
SRC_CLS_TRAIN = "dataset/bccd_bankik/wbc_classify_cnn_model/bccd_wbc_seg/TRAIN"
SRC_CLS_TEST   = "dataset/bccd_bankik/wbc_classify_cnn_model/bccd_wbc_seg/TEST"

# Target (struktur standar)
DST_SEG_IMG = "dataset/bccd/train_images"
DST_SEG_MSK = "dataset/bccd/train_masks"
DST_CLS     = "dataset/blood_cells"


def copy_bccd_seg():
    """Salin gambar + mask BCCD, samakan nama file."""
    os.makedirs(DST_SEG_IMG, exist_ok=True)
    os.makedirs(DST_SEG_MSK, exist_ok=True)

    # Salin gambar
    for f in sorted(os.listdir(SRC_SEG_INPUT)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy2(os.path.join(SRC_SEG_INPUT, f), DST_SEG_IMG)

    # Salin mask: ubah nama BloodImage_Mask_XXXXX.png -> BloodImage_XXXXX.png
    for f in sorted(os.listdir(SRC_SEG_MASK)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            newname = f.replace("_Mask", "")
            shutil.copy2(os.path.join(SRC_SEG_MASK, f),
                         os.path.join(DST_SEG_MSK, newname))

    n_img = len(os.listdir(DST_SEG_IMG))
    n_msk = len(os.listdir(DST_SEG_MSK))
    print(f"[OK] BCCD segmentasi: {n_img} gambar, {n_msk} mask")


def copy_classification():
    """Salin dataset klasifikasi: gabung TRAIN + TEST per kelas."""
    for split in (SRC_CLS_TRAIN, SRC_CLS_TEST):
        for cls_name in sorted(os.listdir(split)):
            src_dir = os.path.join(split, cls_name)
            if not os.path.isdir(src_dir):
                continue
            # Nama folder kelas (lowercase)
            cls_key = cls_name.lower()
            dst_dir = os.path.join(DST_CLS, cls_key)
            os.makedirs(dst_dir, exist_ok=True)

            n = 0
            for f in sorted(os.listdir(src_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(os.path.join(src_dir, f),
                                 os.path.join(dst_dir, f))
                    n += 1
            print(f"  Kelas {cls_key}: +{n} gambar")

    print("[OK] Klasifikasi selesai. Struktur:")
    for cls in sorted(os.listdir(DST_CLS)):
        total = len(os.listdir(os.path.join(DST_CLS, cls)))
        print(f"  - {cls}: {total} gambar")


if __name__ == "__main__":
    print("Memproses dataset BCCD dan klasifikasi...\n")
    copy_bccd_seg()
    print()
    copy_classification()
    print("\nSelesai. Dataset siap digunakan pipeline training.")
