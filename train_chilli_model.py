"""
Chilli Leaf Disease Classification Model Training Pipeline
==========================================================
Uses MobileNetV2 Transfer Learning to classify 5 chilli disease/health categories
from the Chilli Leaf Disease Classification Dataset:
1. cercospora
2. healthy
3. murda_complex_mites_trips
4. nutritional
5. powdery_mildew

Source Repository:
https://github.com/ai-agriculture-circuits-and-systems/chilli_leaf_classification

Key Safety & Architectural Principles:
- Strict candidate separation: Never touches models/best_model.keras.
- Verifies production model SHA-256 hash before and after training.
- Image integrity validation with Pillow + SHA-256 deduplication.
- Stratified 70/15/15 Train/Validation/Test split.
- Data augmentation applied ONLY on training set.
- MobileNetV2 transfer learning with frozen backbone and fine-tuning.
- Comprehensive evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix.
- Saves all artifacts into models/candidates/.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainChilliModel")

# Canonical class ordering for chilli leaf disease classification
CHILLI_CLASSES = [
    "cercospora",
    "healthy",
    "murda_complex_mites_trips",
    "nutritional",
    "powdery_mildew",
]

PRODUCTION_MODEL_PATH = Path("models/best_model.keras")
EXPECTED_PROD_SHA256 = "08daa96f0e3adc676ce993af47b32534f69e4340d0bfd0e2ae4487a1bd425fd2"


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file for integrity and deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_production_model_protection():
    """Confirms production model exists and its hash is untouched."""
    if not PRODUCTION_MODEL_PATH.exists():
        logger.warning("Production model at %s not found.", PRODUCTION_MODEL_PATH)
        return
    current_hash = compute_sha256(PRODUCTION_MODEL_PATH)
    if current_hash != EXPECTED_PROD_SHA256:
        raise RuntimeError(
            f"SAFETY ALERT: Production model hash mismatch! Expected {EXPECTED_PROD_SHA256}, got {current_hash}."
        )
    logger.info("Production model integrity verified (SHA-256: %s...)", current_hash[:16])


def scan_and_validate_chilli_dataset(
    dataset_root: Path,
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Scans dataset/chilli directory, validates image decodability with Pillow,
    computes file hashes, and identifies exact duplicates.
    """
    valid_records = []
    corrupted_records = []
    seen_hashes = {}
    class_counts = {c: 0 for c in CHILLI_CLASSES}

    logger.info("Scanning chilli dataset at: %s", dataset_root)

    search_dirs = [dataset_root]
    if (dataset_root / "chillies").is_dir():
        search_dirs.append(dataset_root / "chillies")

    for category in CHILLI_CLASSES:
        found_cat_dir = None
        for sdir in search_dirs:
            candidate = sdir / category
            if candidate.is_dir():
                found_cat_dir = candidate
                break

        if not found_cat_dir:
            logger.warning("Category folder not found for: %s", category)
            continue

        # Look in images/ subfolder or directly in category folder
        img_dir = found_cat_dir / "images" if (found_cat_dir / "images").is_dir() else found_cat_dir

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
        logger.info("Found %d raw files in category: %s", len(files), category)

        for fpath in sorted(files):
            try:
                # Decodability check with Pillow
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    w, h = img.size
                    mode = img.mode

                file_hash = compute_sha256(fpath)
                file_size = fpath.stat().st_size

                is_duplicate = False
                orig_file = None
                if file_hash in seen_hashes:
                    is_duplicate = True
                    orig_file = seen_hashes[file_hash]
                else:
                    seen_hashes[file_hash] = str(fpath)

                record = {
                    "filepath": str(fpath.resolve()),
                    "filename": fpath.name,
                    "category": category,
                    "class_index": CHILLI_CLASSES.index(category),
                    "file_size": file_size,
                    "sha256": file_hash,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "is_duplicate": is_duplicate,
                    "original_filepath": orig_file,
                }

                if is_duplicate:
                    logger.debug("Duplicate image skipped: %s (matches %s)", fpath.name, orig_file)
                else:
                    valid_records.append(record)
                    class_counts[category] += 1

            except (UnidentifiedImageError, OSError, Exception) as err:
                logger.warning("Corrupted image skipped: %s (%s)", fpath.name, err)
                corrupted_records.append({
                    "filepath": str(fpath.resolve()),
                    "filename": fpath.name,
                    "category": category,
                    "error": str(err),
                })

    logger.info("Dataset scan complete:")
    logger.info("  Valid unique images: %d", len(valid_records))
    logger.info("  Corrupted images: %d", len(corrupted_records))
    for c, cnt in class_counts.items():
        logger.info("    - %-28s: %d", c, cnt)

    return valid_records, corrupted_records, class_counts


def create_stratified_splits(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Splits records into stratified train, val, and test subsets with verified distribution."""
    labels = [r["class_index"] for r in records]

    # First split: Train vs Temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_recs, temp_recs = train_test_split(
        records,
        test_size=temp_ratio,
        stratify=labels,
        random_state=random_seed,
    )

    # Second split: Val vs Test
    temp_labels = [r["class_index"] for r in temp_recs]
    relative_test_ratio = test_ratio / temp_ratio
    val_recs, test_recs = train_test_split(
        temp_recs,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=random_seed,
    )

    for r in train_recs:
        r["split"] = "train"
    for r in val_recs:
        r["split"] = "val"
    for r in test_recs:
        r["split"] = "test"

    logger.info(
        "Stratified Split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(train_recs),
        len(train_recs) / len(records) * 100,
        len(val_recs),
        len(val_recs) / len(records) * 100,
        len(test_recs),
        len(test_recs) / len(records) * 100,
    )

    return train_recs, val_recs, test_recs


def build_tf_dataset(
    records: List[Dict],
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    is_training: bool = False,
    num_classes: int = 5,
) -> tf.data.Dataset:
    """Builds a high-performance tf.data.Dataset pipeline with MobileNetV2 preprocessing."""
    filepaths = [r["filepath"] for r in records]
    labels = [r["class_index"] for r in records]

    def load_and_preprocess_img(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, image_size, method="bilinear")
        img = preprocess_input(img)  # MobileNetV2 expects [-1, 1]
        one_hot_label = tf.one_hot(label, depth=num_classes)
        return img, one_hot_label

    path_ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = path_ds.map(load_and_preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # Augmentation pipeline applied ONLY during training
        augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ], name="train_augmentation")

        ds = ds.shuffle(buffer_size=min(len(records), 1000), seed=42)
        ds = ds.batch(batch_size)
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def build_mobilenetv2_model(
    num_classes: int = 5,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.35,
) -> tf.keras.Model:
    """Instantiates MobileNetV2 pretrained backbone with custom classification head."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Initially frozen

    inputs = tf.keras.Input(shape=input_shape, name="input_tensor")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batchnorm")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Chilli_MobileNetV2")
    return model, base_model


def train_chilli_classifier(
    dataset_root: Path,
    output_dir: Path,
    epochs_frozen: int = 10,
    epochs_fine_tune: int = 6,
    batch_size: int = 16,
    lr_initial: float = 1e-3,
    lr_fine_tune: float = 1e-5,
) -> Dict:
    """Complete training and evaluation pipeline for chilli leaf disease classification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Safety verification
    verify_production_model_protection()

    # 2. Ingestion & Validation
    valid_recs, corrupt_recs, class_counts = scan_and_validate_chilli_dataset(dataset_root)
    if len(valid_recs) < 25:
        raise ValueError(f"Insufficient valid images ({len(valid_recs)}) in {dataset_root}.")

    # 3. Stratified Splits (70/15/15)
    train_recs, val_recs, test_recs = create_stratified_splits(valid_recs, random_seed=42)

    # Save dataset manifest
    manifest_path = output_dir / "chilli_dataset_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "category", "class_index", "split", "file_size", "sha256", "width", "height"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in train_recs + val_recs + test_recs:
            writer.writerow(r)
    logger.info("Saved dataset manifest to: %s", manifest_path)

    # 4. TensorFlow Datasets
    train_ds = build_tf_dataset(train_recs, batch_size=batch_size, is_training=True)
    val_ds = build_tf_dataset(val_recs, batch_size=batch_size, is_training=False)
    test_ds = build_tf_dataset(test_recs, batch_size=batch_size, is_training=False)

    # 5. Model Architecture
    model, base_model = build_mobilenetv2_model(num_classes=len(CHILLI_CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_initial),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    history_records = []

    # 6. Phase 1: Train classification head with frozen backbone
    logger.info("=== Phase 1: Training Classification Head (%d epochs) ===", epochs_frozen)
    t0 = time.time()
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_frozen,
        verbose=1,
    )
    for ep in range(epochs_frozen):
        history_records.append({
            "phase": "frozen",
            "epoch": ep + 1,
            "train_loss": float(h1.history["loss"][ep]),
            "train_acc": float(h1.history["accuracy"][ep]),
            "val_loss": float(h1.history["val_loss"][ep]),
            "val_acc": float(h1.history["val_accuracy"][ep]),
        })

    # 7. Phase 2: Fine-Tuning top backbone layers
    logger.info("=== Phase 2: Fine-Tuning Backbone (%d epochs) ===", epochs_fine_tune)
    base_model.trainable = True
    # Freeze all layers except last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_fine_tune),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = checkpoints_dir / "best_chilli_weights.weights.h5"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
    )

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_fine_tune,
        callbacks=[checkpoint_cb],
        verbose=1,
    )

    for ep in range(epochs_fine_tune):
        history_records.append({
            "phase": "fine_tune",
            "epoch": epochs_frozen + ep + 1,
            "train_loss": float(h2.history["loss"][ep]),
            "train_acc": float(h2.history["accuracy"][ep]),
            "val_loss": float(h2.history["val_loss"][ep]),
            "val_acc": float(h2.history["val_accuracy"][ep]),
        })

    # Restore best weights if available
    if ckpt_path.exists():
        logger.info("Restoring best validation weights from %s", ckpt_path)
        model.load_weights(str(ckpt_path))

    total_training_time = time.time() - t0
    logger.info("Training completed in %.1f seconds.", total_training_time)

    # 8. Unseen Test Set Evaluation
    logger.info("=== Evaluating on Unseen Test Set (%d images) ===", len(test_recs))
    y_true_indices = []
    y_pred_indices = []
    y_pred_probs = []

    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true_indices.extend(np.argmax(lbls.numpy(), axis=1))
        y_pred_indices.extend(np.argmax(preds, axis=1))
        y_pred_probs.extend(preds)

    y_true_indices = np.array(y_true_indices)
    y_pred_indices = np.array(y_pred_indices)
    y_pred_probs = np.array(y_pred_probs)

    test_accuracy = float(np.mean(y_true_indices == y_pred_indices))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average="weighted", zero_division=0
    )

    c_matrix = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(CHILLI_CLASSES)))
    clf_report = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=CHILLI_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    logger.info("TEST ACCURACY: %.4f (%.2f%%)", test_accuracy, test_accuracy * 100)
    logger.info("Macro F1: %.4f | Weighted F1: %.4f", f1_macro, f1_weighted)
    logger.info("\nConfusion Matrix:\n%s", c_matrix)

    # 9. Save Artifacts strictly into candidate location
    candidate_model_path = output_dir / "chilli_candidate_mobilenetv2.keras"
    model.save(str(candidate_model_path))
    model_sha256 = compute_sha256(candidate_model_path)
    logger.info("Saved candidate model to %s (SHA-256: %s)", candidate_model_path, model_sha256)

    # Class names JSON
    class_names_path = output_dir / "chilli_class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(CHILLI_CLASSES, f, indent=2)

    # Training history CSV
    history_csv_path = output_dir / "chilli_training_history.csv"
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history_records)

    # Full Evaluation Report
    eval_report = {
        "dataset_name": "chilli_leaf_classification",
        "dataset_source": "https://github.com/ai-agriculture-circuits-and-systems/chilli_leaf_classification",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": "MobileNetV2 (Transfer Learning)",
        "num_classes": len(CHILLI_CLASSES),
        "classes": CHILLI_CLASSES,
        "sample_counts": {
            "total_valid": len(valid_recs),
            "train": len(train_recs),
            "val": len(val_recs),
            "test": len(test_recs),
            "by_class": class_counts,
        },
        "training_time_seconds": round(total_training_time, 2),
        "test_metrics": {
            "accuracy": round(test_accuracy, 4),
            "macro_precision": round(float(precision_macro), 4),
            "macro_recall": round(float(recall_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_precision": round(float(precision_weighted), 4),
            "weighted_recall": round(float(recall_weighted), 4),
            "weighted_f1": round(float(f1_weighted), 4),
        },
        "per_class_report": clf_report,
        "confusion_matrix": c_matrix.tolist(),
        "candidate_model_file": str(candidate_model_path.name),
        "candidate_model_sha256": model_sha256,
        "production_model_protected": True,
    }

    eval_json_path = output_dir / "chilli_evaluation_report.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved evaluation report to: %s", eval_json_path)

    # 10. Re-verify production model protection
    verify_production_model_protection()

    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chilli Leaf Disease Classification Model")
    parser.add_argument("--dataset-dir", type=str, default="dataset/chilli", help="Path to chilli dataset")
    parser.add_argument("--output-dir", type=str, default="models/candidates", help="Output directory for candidate artifacts")
    parser.add_argument("--epochs-frozen", type=int, default=10, help="Epochs for frozen head training")
    parser.add_argument("--epochs-fine-tune", type=int, default=6, help="Epochs for backbone fine-tuning")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    report = train_chilli_classifier(
        dataset_root=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        epochs_frozen=args.epochs_frozen,
        epochs_fine_tune=args.epochs_fine_tune,
        batch_size=args.batch_size,
    )
    print("\nTraining and Evaluation Summary:")
    print(json.dumps(report["test_metrics"], indent=2))
