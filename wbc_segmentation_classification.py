"""
=============================================================================
 Klasifikasi Jenis Sel Darah Putih pada Citra Mikroskopi Digital
 Menggunakan Segmentasi U-Net dan Klasifikasi CNN
---------------------------------------------------------------------------
 Topik  : Informatika Biomedik (Biomedical Image Processing)
 Dataset : BCCD (segmentasi) + Mendeley Blood Cell Images (klasifikasi)
 Tools  : Python 3, TensorFlow/Keras, OpenCV, NumPy, Matplotlib
=============================================================================
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# 0. PENGATURAN & KONSTANTA
# -----------------------------------------------------------------------------
IMG_SIZE = 256        # ukuran input (persegi)
BATCH_SIZE = 8
EPOCHS_SEG = 50
EPOCHS_CLS = 30
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GANTI dengan path folder dataset kalian
SEG_DATA_DIR = "dataset/bccd/"        # berisi subfolder train_images, train_masks, dll.
CLS_DATA_DIR = "dataset/blood_cells/" # berisi subfolder per kelas (neutrophil, lymphocyte, dst)

# -----------------------------------------------------------------------------
# 1. METRIK KUSTOM (untuk segmentasi)
# -----------------------------------------------------------------------------
def iou_metric(y_true, y_pred):
    """Intersection over Union (Jaccard) untuk mask binari."""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def dice_metric(y_true, y_pred):
    """Sorensen-Dice coefficient untuk mask binari."""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + 1e-7) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7
    )

# -----------------------------------------------------------------------------
# 2. ARSITEKTUR U-NET
# -----------------------------------------------------------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    return x

def build_unet(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(input_size)

    # ENCODER (kontraksi)
    c1 = conv_block(inputs, 64);  p1 = layers.MaxPool2D(2)(c1)
    c2 = conv_block(p1, 128);     p2 = layers.MaxPool2D(2)(c2)
    c3 = conv_block(p2, 256);     p3 = layers.MaxPool2D(2)(c3)
    c4 = conv_block(p3, 512);     p4 = layers.MaxPool2D(2)(c4)

    # BOTTLENECK
    c5 = conv_block(p4, 1024)

    # DECODER (ekspansi)
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c4]); c6 = conv_block(u6, 512)

    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c3]); c7 = conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = layers.concatenate([u8, c2]); c8 = conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = layers.concatenate([u9, c1]); c9 = conv_block(u9, 64)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
    return Model(inputs, outputs)

# -----------------------------------------------------------------------------
# 3. PERSIAPAN DATA SEGMENTASI (BCCD)
# -----------------------------------------------------------------------------
def load_bccd(seg_dir):
    """
    Asumsi struktur:
      dataset/bccd/train_images/*.jpg
      dataset/bccd/train_masks/*.png
    Sesuaikan nama folder sesuai dataset kalian.
    """
    img_dir = os.path.join(seg_dir, "train_images")
    msk_dir = os.path.join(seg_dir, "train_masks")

    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan. Pastikan BCCD ada di: {seg_dir}"
        )

    imgs, masks = [], []
    # Index mask via stem nama file (gambar .jpg, mask .png)
    mask_index = {}
    for m in os.listdir(msk_dir):
        if m.lower().endswith((".jpg", ".png", ".jpeg")):
            mask_index[os.path.splitext(m)[0]] = os.path.join(msk_dir, m)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(img_dir, fname)
        msk_path = mask_index.get(os.path.splitext(fname)[0])

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0

        if msk_path is None:
            continue
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            continue
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE))
        # Mask WBC: semua pixel bukan background dianggap sel (1)
        msk = (msk > 0).astype("float32")
        msk = np.expand_dims(msk, axis=-1)

        imgs.append(img)
        masks.append(msk)

    imgs = np.array(imgs)
    masks = np.array(masks)
    return imgs, masks

# -----------------------------------------------------------------------------
# 4. PERSIAPAN DATA KLASIFIKASI (Mendeley / Kaggle 5 kelas)
# -----------------------------------------------------------------------------
def load_classification(CLS_DATA_DIR):
    """
    Dataset berisi subfolder per kelas (neutrophil, lymphocyte, monocyte,
    eosinophil, basophil). Gunakan ImageDataGenerator utk augmentasi + split.
    """
    if not os.path.isdir(CLS_DATA_DIR):
        raise FileNotFoundError(
            f"Folder dataset klasifikasi tidak ditemukan. Pastikan ada di: {CLS_DATA_DIR}"
        )

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=0.2,
    )
    train_gen = datagen.flow_from_directory(
        CLS_DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )
    val_gen = datagen.flow_from_directory(
        CLS_DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )
    return train_gen, val_gen

# -----------------------------------------------------------------------------
# 5. MODEL KLASIFIKASI (CNN / Transfer Learning)
# -----------------------------------------------------------------------------
def build_classifier(num_classes):
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = True  # fine-tuning; bisa di-set False utk transfer learning murni

    x = layers.Dropout(0.3)(base.output)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    return model

# -----------------------------------------------------------------------------
# 6. PIPELINE UTAMA
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" PIPELINE: Segmentasi WBC (U-Net) -> Klasifikasi WBC (CNN)")
    print("=" * 70)

    # ---------- TAHAP A: SEGMENTASI ----------
    print("\n[1/3] Memuat dataset segmentasi (BCCD)...")
    try:
        imgs, masks = load_bccd(SEG_DATA_DIR)
    except FileNotFoundError as e:
        print(f"  ! {e}")
        imgs, masks = None, None

    if imgs is not None and len(imgs) > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            imgs, masks, test_size=0.2, random_state=SEED
        )
        print(f"  Data segmentasi: train={len(X_train)}, val={len(X_val)}")

        unet = build_unet()
        unet.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=[iou_metric, dice_metric],
        )
        unet.summary()

        print("\n[2/3] Melatih U-Net...")
        history = unet.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_SEG,
        )

        # Simpan model
        unet.save("unet_wbc_segmentation.h5")
        print("  Model U-Net tersimpan: unet_wbc_segmentation.h5")

        # Plot loss & metrik
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Loss U-Net"); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history["dice_metric"], label="Train Dice")
        plt.plot(history.history["val_dice_metric"], label="Val Dice")
        plt.title("Dice U-Net"); plt.legend()
        plt.tight_layout(); plt.savefig("unet_training.png"); plt.show()
    else:
        print("  ! Dataset segmentasi dilewati (folder belum ada).")

    # ---------- TAHAP B: KLASIFIKASI ----------
    print("\n[3/3] Memuat dataset klasifikasi (Blood Cell Images)...")
    try:
        train_gen, val_gen = load_classification(CLS_DATA_DIR)
    except FileNotFoundError as e:
        print(f"  ! {e}")
        train_gen, val_gen = None, None

    if train_gen is not None:
        num_classes = len(train_gen.class_indices)
        print(f"  Kelas ditemukan: {train_gen.class_indices}")
        print(f"  Jumlah sampel train={train_gen.samples}, val={val_gen.samples}")

        classifier = build_classifier(num_classes)
        classifier.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        classifier.summary()

        hist_cls = classifier.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_CLS,
        )
        classifier.save("wbc_classifier.h5")
        print("  Model klasifikasi tersimpan: wbc_classifier.h5")

        # Evaluasi final
        print("\n--- Evaluasi Klasifikasi ---")
        val_loss, val_acc = classifier.evaluate(val_gen)
        print(f"  Validasi Loss: {val_loss:.4f}, Validasi Akurasi: {val_acc:.4f}")

        # Prediksi + confusion matrix
        val_gen.reset()
        y_pred = classifier.predict(val_gen)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = val_gen.classes[: len(y_pred_labels)]

        class_names = list(val_gen.class_indices.keys())
        print("\n--- Classification Report ---")
        print(classification_report(y_true_labels, y_pred_labels,
                                    target_names=class_names))

        cm = confusion_matrix(y_true_labels, y_pred_labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix Klasifikasi WBC")
        plt.savefig("confusion_matrix.png"); plt.show()
    else:
        print("  ! Dataset klasifikasi dilewati (folder belum ada).")

    print("\nSelesai. Pipeline siap dikembangkan untuk dataset lengkap.")

if __name__ == "__main__":
    main()
