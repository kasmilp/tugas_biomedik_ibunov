"""
=============================================================================
 Perbaikan Klasifikasi WBC — Versi 2 (underfitting fix)
 File  : train_cls_improved.py
 Runtime: Python 3.11, TensorFlow 2.13 (CPU), Windows
---------------------------------------------------------------------------
 Perubahan vs v1 (dari diagnosis log training):
   1. Epoch 10 -> 30
   2. steps_per_epoch 80 -> 150  (eksposur data ~12% -> ~44% per run)
   3. lr konstan 1e-3 -> ReduceLROnPlateau (factor 0.5, patience 3)
   4. CNN ringan -> CNN + BatchNorm + blok lebih dalam (32/64/128/256)
   5. Dropout 0.4 -> 0.5, Dense 256
Output langsung ke hasil_akhir/ (grafik, model, riwayat, report).
=============================================================================
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# KONSTANTA
# -----------------------------------------------------------------------------
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS_CLS = 30
CLS_STEPS = 150          # batch per epoch (638 batch total tersedia; 150 = ~44%/epoch)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLS_DATA_DIR = "dataset/blood_cells"
OUT = "hasil_akhir"
for sub in ("grafik", "model", "riwayat"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

# -----------------------------------------------------------------------------
# CNN V2: + BatchNorm, blok lebih dalam
# -----------------------------------------------------------------------------
def conv_bn(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_classifier_v2(num_classes):
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    # blok 1
    x = conv_bn(inputs, 32)
    x = conv_bn(x, 32)
    x = layers.MaxPool2D(2)(x)
    # blok 2
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    x = layers.MaxPool2D(2)(x)
    # blok 3
    x = conv_bn(x, 128)
    x = conv_bn(x, 128)
    x = layers.MaxPool2D(2)(x)
    # blok 4
    x = conv_bn(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# -----------------------------------------------------------------------------
# GRAFIK
# -----------------------------------------------------------------------------
def plot_history(h):
    ep = range(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ep, h['loss'], 'b-o', ms=4, label='Train Loss')
    axes[0].plot(ep, h['val_loss'], 'r-o', ms=4, label='Val Loss')
    axes[0].set_title('Klasifikasi CNN v2 — Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, h['accuracy'], 'b-o', ms=4, label='Train Akurasi')
    axes[1].plot(ep, h['val_accuracy'], 'r-o', ms=4, label='Val Akurasi')
    axes[1].set_title('Klasifikasi CNN v2 — Akurasi'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT, "grafik", "grafik_klasifikasi.png")
    plt.savefig(p, dpi=150); plt.close(fig)
    print("  ->", p)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("[Data] Loading generator...")
    datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.15, horizontal_flip=True,
        validation_split=0.2)
    train_gen = datagen.flow_from_directory(CLS_DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", subset="training",
        shuffle=True, seed=SEED)
    val_gen = datagen.flow_from_directory(CLS_DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", subset="validation",
        shuffle=False, seed=SEED)
    n_cls = len(train_gen.class_indices)
    val_steps = max(1, val_gen.samples // BATCH_SIZE)
    print(f"  train={train_gen.samples}, val={val_gen.samples}, kelas={n_cls}")

    model = build_classifier_v2(n_cls)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                         restore_best_weights=True, verbose=1),
    ]

    print(f"[Train] CNN v2: {EPOCHS_CLS} epoch x {CLS_STEPS} step...")
    h = model.fit(train_gen, validation_data=val_gen,
                  epochs=EPOCHS_CLS, steps_per_epoch=CLS_STEPS,
                  validation_steps=val_steps, callbacks=callbacks, verbose=2)

    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    print("[Grafik] menyimpan kurva...")
    plot_history(h.history)

    # ---- Evaluasi penuh di set validation ----
    print("[Evaluasi] confusion matrix & report...")
    val_gen.reset()
    preds = model.predict(val_gen, steps=val_steps + 1, verbose=0)[:val_gen.samples]
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes[:len(y_pred)]
    names = list(val_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=names))

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=names).plot(cmap="Blues")
    plt.title("Confusion Matrix — CNN v2")
    p = os.path.join(OUT, "grafik", "confusion_matrix.png")
    plt.savefig(p, dpi=150); plt.close()
    print("  ->", p)

    # ---- Simpan semua ke hasil_akhir ----
    model.save(os.path.join(OUT, "model", "wbc_classifier.keras"))
    print("  -> hasil_akhir/model/wbc_classifier.keras")

    riwayat = {"klasifikasi_v2": hist,
               "konfigurasi": {"IMG_SIZE": IMG_SIZE, "BATCH_SIZE": BATCH_SIZE,
                               "EPOCHS": EPOCHS_CLS, "STEPS_PER_EPOCH": CLS_STEPS,
                               "optimizer": "Adam 1e-3 + ReduceLROnPlateau"},
               "kelas": names,
               "classification_report": report}
    with open(os.path.join(OUT, "riwayat", "riwayat_klasifikasi.json"), "w") as f:
        json.dump(riwayat, f, indent=2)
    print("  -> hasil_akhir/riwayat/riwayat_klasifikasi.json")
    print(f"\nSELESAI. Akurasi val terbaik: {report['accuracy']:.4f}")

if __name__ == "__main__":
    main()