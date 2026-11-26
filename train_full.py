"""
=============================================================================
 Training Penuh + Grafik Hasil (versi CPU-friendly)
 File  : train_full.py
 Runtime: Python 3.10, TensorFlow 2.13 (CPU), Windows
 Catatan: CPU-only -> IMG_SIZE=128, batch=16, epoch dibatasi agar selesai.
 Output : kurva training PNG + confusion matrix + model .keras
=============================================================================
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# KONSTANTA (disesuaikan untuk CPU)
# -----------------------------------------------------------------------------
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS_SEG = 15
EPOCHS_CLS = 10
CLS_STEPS = 80          # batch per epoch klasifikasi (subset agar cepat)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SEG_IMG_DIR = "dataset/bccd/train_images"
SEG_MSK_DIR = "dataset/bccd/train_masks"
CLS_DATA_DIR = "dataset/blood_cells"
OUT_DIR = "hasil_training"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# METRIK
# -----------------------------------------------------------------------------
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + 1e-7) / (union + 1e-7)

def dice_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

# -----------------------------------------------------------------------------
# U-NET (ringkas: filter dikurangi agar muat di CPU)
# -----------------------------------------------------------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    return x

def build_unet(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(input_size)
    c1 = conv_block(inputs, 32);  p1 = layers.MaxPool2D(2)(c1)
    c2 = conv_block(p1, 64);      p2 = layers.MaxPool2D(2)(c2)
    c3 = conv_block(p2, 128);     p3 = layers.MaxPool2D(2)(c3)
    c4 = conv_block(p3, 256);     p4 = layers.MaxPool2D(2)(c4)
    c5 = conv_block(p4, 512)
    u6 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4]); c6 = conv_block(u6, 256)
    u7 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3]); c7 = conv_block(u7, 128)
    u8 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2]); c8 = conv_block(u8, 64)
    u9 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1]); c9 = conv_block(u9, 32)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs, outputs)

# -----------------------------------------------------------------------------
# LOAD DATA SEGMENTASI
# -----------------------------------------------------------------------------
def load_bccd():
    imgs, masks = [], []
    # Index mask berdasarkan stem nama file (tanpa ekstensi):
    # gambar = BloodImage_00000.jpg, mask = BloodImage_00000.png -> cocok via stem
    mask_index = {}
    for m in os.listdir(SEG_MSK_DIR):
        if m.lower().endswith((".jpg", ".png", ".jpeg")):
            mask_index[os.path.splitext(m)[0]] = os.path.join(SEG_MSK_DIR, m)

    files = sorted(f for f in os.listdir(SEG_IMG_DIR)
                   if f.lower().endswith((".jpg", ".png", ".jpeg")))
    for fname in files:
        img = cv2.imread(os.path.join(SEG_IMG_DIR, fname))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        msk_path = mask_index.get(os.path.splitext(fname)[0])
        if msk_path is None:
            continue
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            continue
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE))
        msk = (msk > 0).astype("float32")[..., None]
        imgs.append(img); masks.append(msk)
    return np.array(imgs), np.array(masks)

# -----------------------------------------------------------------------------
# CNN KLASIFIKASI RINGAN (pengganti ResNet50 untuk CPU)
# -----------------------------------------------------------------------------
def build_classifier(num_classes):
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# -----------------------------------------------------------------------------
# GRAFIK
# -----------------------------------------------------------------------------
def plot_seg_history(h):
    epochs = range(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, h['loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, h['val_loss'], 'r-o', label='Val Loss')
    axes[0].set_title('Segmentasi U-Net — Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, h['iou_metric'], 'b-o', label='Train IoU')
    axes[1].plot(epochs, h['val_iou_metric'], 'r-o', label='Val IoU')
    axes[1].set_title('Segmentasi U-Net — IoU'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, h['dice_metric'], 'b-o', label='Train Dice')
    axes[2].plot(epochs, h['val_dice_metric'], 'r-o', label='Val Dice')
    axes[2].set_title('Segmentasi U-Net — Dice Coefficient'); axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "grafik_segmentasi.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")

def plot_cls_history(h):
    epochs = range(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, h['loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, h['val_loss'], 'r-o', label='Val Loss')
    axes[0].set_title('Klasifikasi CNN — Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, h['accuracy'], 'b-o', label='Train Akurasi')
    axes[1].plot(epochs, h['val_accuracy'], 'r-o', label='Val Akurasi')
    axes[1].set_title('Klasifikasi CNN — Akurasi'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "grafik_klasifikasi.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")

def plot_mask_comparison(X, y, model, n=4):
    preds = model.predict(X[:n], verbose=0)
    fig, axes = plt.subplots(3, n, figsize=(3.2*n, 9))
    for i in range(n):
        axes[0, i].imshow(X[i]); axes[0, i].set_title('Citra Asli'); axes[0, i].axis('off')
        axes[1, i].imshow(y[i, :, :, 0], cmap='gray'); axes[1, i].set_title('Ground Truth'); axes[1, i].axis('off')
        pm = (preds[i, :, :, 0] > 0.5).astype(float)
        axes[2, i].imshow(pm, cmap='gray'); axes[2, i].set_title('Prediksi U-Net'); axes[2, i].axis('off')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "perbandingan_mask.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    results = {}

    # ===== TAHAP A: SEGMENTASI =====
    print("[A] Loading data segmentasi...")
    imgs, masks = load_bccd()
    print(f"    {len(imgs)} sampel, bentuk {imgs.shape}")
    Xtr, Xva, ytr, yva = train_test_split(imgs, masks, test_size=0.2, random_state=SEED)

    unet = build_unet()
    unet.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[iou_metric, dice_metric])
    print("[A] Training U-Net...")
    h_seg = unet.fit(Xtr, ytr, validation_data=(Xva, yva),
                     batch_size=BATCH_SIZE, epochs=EPOCHS_SEG, verbose=2)
    results['segmentasi'] = {k: [float(v) for v in vals] for k, vals in h_seg.history.items()}
    print("[A] Grafik segmentasi...")
    plot_seg_history(h_seg.history)
    plot_mask_comparison(Xva, yva, unet)
    unet.save(os.path.join(OUT_DIR, "unet_wbc.keras"))
    best_iou = max(results['segmentasi']['val_iou_metric'])
    best_dice = max(results['segmentasi']['val_dice_metric'])
    results['segmentasi_best'] = {'val_iou': round(best_iou, 4),
                                  'val_dice': round(best_dice, 4)}
    print(f"    Best Val IoU={best_iou:.4f}, Dice={best_dice:.4f}")

    # ===== TAHAP B: KLASIFIKASI =====
    print("[B] Menyiapkan generator klasifikasi...")
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=15,
                                 width_shift_range=0.1, height_shift_range=0.1,
                                 zoom_range=0.1, horizontal_flip=True,
                                 validation_split=0.2)
    train_gen = datagen.flow_from_directory(CLS_DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", subset="training",
        shuffle=True, seed=SEED)
    val_gen = datagen.flow_from_directory(CLS_DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", subset="validation",
        shuffle=False, seed=SEED)
    n_cls = len(train_gen.class_indices)
    results['kelas'] = train_gen.class_indices

    clf = build_classifier(n_cls)
    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss="categorical_crossentropy", metrics=["accuracy"])
    print("[B] Training CNN klasifikasi...")
    h_cls = clf.fit(train_gen, validation_data=val_gen,
                    epochs=EPOCHS_CLS, steps_per_epoch=CLS_STEPS,
                    validation_steps=max(1, val_gen.samples // BATCH_SIZE),
                    verbose=2)
    results['klasifikasi'] = {k: [float(v) for v in vals] for k, vals in h_cls.history.items()}
    print("[B] Grafik klasifikasi...")
    plot_cls_history(h_cls.history)

    # ===== EVALUASI + CONFUSION MATRIX =====
    print("[Evaluasi] Confusion matrix & report...")
    val_gen.reset()
    y_pred = clf.predict(val_gen, steps=max(1, val_gen.samples // BATCH_SIZE) + 1,
                         verbose=0)[:val_gen.samples]
    y_pred_lab = np.argmax(y_pred, axis=1)
    y_true_lab = val_gen.classes[:len(y_pred_lab)]
    class_names = list(val_gen.class_indices.keys())
    report = classification_report(y_true_lab, y_pred_lab, target_names=class_names, output_dict=True)
    results['classification_report'] = report
    print(classification_report(y_true_lab, y_pred_lab, target_names=class_names))

    cm = confusion_matrix(y_true_lab, y_pred_lab)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Klasifikasi WBC")
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  -> {path}")

    clf.save(os.path.join(OUT_DIR, "wbc_classifier.keras"))

    # ===== SIMPAN RIWAYAT JSON =====
    with open(os.path.join(OUT_DIR, "riwayat_training.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> {OUT_DIR}/riwayat_training.json")
    print("\nSELESAI. Semua hasil di folder:", OUT_DIR)

if __name__ == "__main__":
    main()