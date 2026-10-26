"""
Unified Multi-Crop Plant Disease Classification Model Training Pipeline
=======================================================================
Combines multiple real agricultural crop datasets into a single, comprehensive
MobileNetV2 classifier:
1. Blackgram (Vigna mungo): anthracnose, healthy, leaf_crinckle, powdery_mildew, yellow_mosaic
2. Chilli (Capsicum annuum): cercospora, healthy, murda_complex_mites_trips, nutritional, powdery_mildew
3. Tea (Camellia sinensis): algal_spot, brown_blight, gray_blight, healthy, helopeltis, red_spot
4. Citrus (Citrus spp.): black_spot, canker, greening, healthy, melanose

Key Features:
- Standardized class nomenclature: <Crop>___<Disease>
- Image integrity validation with Pillow and SHA-256 deduplication
- Detailed dataset inspection displaying image filenames and disease labels
- Stratified 70/15/15 Train / Validation / Test split across all 21 classes
- MobileNetV2 Transfer Learning with ImageNet weights and top-layer fine-tuning
- Comprehensive evaluation: Accuracy, Loss, Precision, Recall, F1, Confusion Matrix
- Strict verification of saved artifacts
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
logger = logging.getLogger("TrainUnifiedModel")

DEFAULT_DATASETS = {
    "Blackgram": Path("dataset/blackgram/blackgrams"),
    "Chilli": Path("dataset/chilli/chillies"),
    "Tea": Path("dataset/tea_extracted/tea_leaf_disease_classification-main/tea_leaves"),
    "Citrus": Path("dataset/citrus_extracted/citrus_leaves-main/leaves"),
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash for file integrity and deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_and_validate_multicrop_dataset(
    dataset_dirs: Dict[str, Path],
    max_per_class: int = 250,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Scans all crop directories, validates images, detects duplicates,
    and returns valid records, corrupted records, and unique class names.
    """
    valid_records = []
    corrupted_records = []
    seen_hashes = {}
    classes_set = set()

    print("\n" + "=" * 80)
    print("        DATASET INSPECTION: MULTI-CROP PLANT LEAF DISEASE DATASETS")
    print("=" * 80)

    for crop_name, crop_dir in dataset_dirs.items():
        crop_path = crop_dir.resolve()
        if not crop_path.exists():
            logger.warning("Dataset directory not found for crop '%s': %s", crop_name, crop_path)
            continue

        logger.info("Scanning crop: %s at %s", crop_name, crop_path)
        
        # Scan category subdirectories
        cat_dirs = sorted([d for d in crop_path.iterdir() if d.is_dir() and d.name not in ("sets", "annotations", "data", "docs", "fruits")])
        
        for cat_dir in cat_dirs:
            disease_name = cat_dir.name
            canonical_class = f"{crop_name}___{disease_name}"
            classes_set.add(canonical_class)

            img_dir = cat_dir / "images" if (cat_dir / "images").is_dir() else cat_dir
            img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])

            logger.info("Category '%s': found %d potential images", canonical_class, len(img_files))

            # Sample/cap per class for balanced distribution across crops
            selected_files = img_files[:max_per_class] if max_per_class > 0 else img_files

            for img_path in selected_files:
                try:
                    with Image.open(img_path) as img:
                        img.verify()

                    with Image.open(img_path) as img:
                        width, height = img.size
                        img_format = img.format or img_path.suffix.upper().replace(".", "")

                    sha = compute_sha256(img_path)

                    if sha in seen_hashes:
                        corrupted_records.append({
                            "path": img_path,
                            "filename": img_path.name,
                            "crop": crop_name,
                            "class_name": canonical_class,
                            "reason": f"Exact duplicate of {seen_hashes[sha]['path'].name}",
                            "sha256": sha,
                        })
                        print(f"[DATASET INSPECTION] Image: {img_path.name:<20} | Crop: {crop_name:<10} | Disease: {canonical_class:<35} | Status: DUPLICATE")
                        continue

                    record = {
                        "path": img_path,
                        "filename": img_path.name,
                        "crop": crop_name,
                        "disease": disease_name,
                        "class_name": canonical_class,
                        "sha256": sha,
                        "width": width,
                        "height": height,
                        "format": img_format,
                    }
                    seen_hashes[sha] = record
                    valid_records.append(record)
                    print(f"[DATASET INSPECTION] Image: {img_path.name:<20} | Crop: {crop_name:<10} | Disease: {canonical_class:<35} | Status: VALID ({width}x{height})")

                except (UnidentifiedImageError, OSError, Exception) as err:
                    corrupted_records.append({
                        "path": img_path,
                        "filename": img_path.name,
                        "crop": crop_name,
                        "class_name": canonical_class,
                        "reason": str(err),
                        "sha256": "N/A",
                    })
                    print(f"[DATASET INSPECTION] Image: {img_path.name:<20} | Crop: {crop_name:<10} | Disease: {canonical_class:<35} | Status: CORRUPTED ({err})")

    canonical_classes = sorted(list(classes_set))
    print("=" * 80)
    print(f"Inspection Summary: {len(valid_records)} valid images across {len(canonical_classes)} classes, {len(corrupted_records)} duplicates/corrupted.")
    print("=" * 80 + "\n")

    return valid_records, corrupted_records, canonical_classes


def create_stratified_splits(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Splits records into stratified train, val, test subsets."""
    labels = [r["class_name"] for r in records]
    train_recs, temp_recs, train_labels, temp_labels = train_test_split(
        records,
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state,
    )

    val_prop_of_temp = val_ratio / (val_ratio + test_ratio)
    val_recs, test_recs, _, _ = train_test_split(
        temp_recs,
        temp_labels,
        test_size=(1.0 - val_prop_of_temp),
        stratify=temp_labels,
        random_state=random_state,
    )

    for r in train_recs:
        r["split"] = "train"
    for r in val_recs:
        r["split"] = "val"
    for r in test_recs:
        r["split"] = "test"

    return train_recs, val_recs, test_recs


def load_and_preprocess_image(path_str: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Loads image, converts to RGB, resizes, and scales to [-1, 1] for MobileNetV2."""
    with Image.open(path_str) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        return preprocess_input(arr)


def records_to_numpy(
    records: List[Dict],
    class_to_idx: Dict[str, int],
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads all records into numpy arrays."""
    images = []
    labels = []
    for r in records:
        arr = load_and_preprocess_image(str(r["path"]), target_size=target_size)
        images.append(arr)
        labels.append(class_to_idx[r["class_name"]])
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def build_multicrop_model(num_classes: int, input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Constructs MobileNetV2 transfer learning model."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="leaf_image_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.3, name="dropout_regularization")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction_head")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Agrotech_Unified_MobileNetV2")
    return model, base_model


def train_and_evaluate_unified(
    output_dir: str = "models/candidates",
    epochs: int = 6,
    batch_size: int = 32,
    max_per_class: int = 200,
    seed: int = 42,
) -> Dict:
    """Executes the complete multi-crop dataset training and evaluation pipeline."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

    candidate_path = Path(output_dir).resolve()
    candidate_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Unified Multi-Crop Training Pipeline...")

    # 1. Ingestion & Validation
    valid_recs, corrupted_recs, class_names = scan_and_validate_multicrop_dataset(
        DEFAULT_DATASETS, max_per_class=max_per_class
    )

    if len(valid_recs) == 0:
        raise ValueError("No valid images found in provided datasets.")

    logger.info("Total classes: %d, Total valid images: %d", len(class_names), len(valid_recs))

    # 2. Stratified Splits
    train_recs, val_recs, test_recs = create_stratified_splits(valid_recs, random_state=seed)
    logger.info("Splits created: Train=%d, Val=%d, Test=%d", len(train_recs), len(val_recs), len(test_recs))

    # Save Manifest and Class Names
    manifest_file = candidate_path / "unified_dataset_manifest.csv"
    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "crop", "disease", "class_name", "split", "width", "height", "format", "sha256", "path"])
        for r in train_recs + val_recs + test_recs:
            writer.writerow([r["filename"], r["crop"], r["disease"], r["class_name"], r["split"], r["width"], r["height"], r["format"], r["sha256"], str(r["path"])])

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    # Save class_names.json to candidate dir, models dir, and project root
    for p in [candidate_path / "unified_class_names.json", Path("unified_class_names.json").resolve(), Path("models/candidates/class_names.json").resolve()]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)
    logger.info("Class names written to: %s and root unified_class_names.json", candidate_path / "unified_class_names.json")

    # 3. Load Numpy Arrays
    logger.info("Loading training images into memory...")
    X_train, y_train = records_to_numpy(train_recs, class_to_idx)
    logger.info("Loading validation images into memory...")
    X_val, y_val = records_to_numpy(val_recs, class_to_idx)
    logger.info("Loading held-out test images into memory...")
    X_test, y_test = records_to_numpy(test_recs, class_to_idx)

    # 4. Data Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
    ], name="data_augmentation")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train), seed=seed)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 5. Build and Compile Model
    model, base_model = build_multicrop_model(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_weights_path = candidate_path / "best_unified_weights.weights.h5"
    history_csv_path = candidate_path / "unified_training_history.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_csv_path)),
    ]

    # 6. Stage 1: Feature Extraction
    logger.info("--- Starting Stage 1: Frozen Backbone Training (%d epochs) ---", epochs)
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    stage1_time = time.time() - t0
    logger.info("Stage 1 completed in %.2f seconds.", stage1_time)

    # 7. Stage 2: Fine-Tuning Top 30 Layers
    logger.info("--- Starting Stage 2: Fine-Tuning Top 30 Layers (3 epochs) ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        callbacks=callbacks,
        verbose=1,
    )

    if best_weights_path.exists():
        logger.info("Restoring best weights from %s", best_weights_path)
        model.load_weights(str(best_weights_path))

    # 8. Save Model Artifacts
    primary_candidate_file = candidate_path / "unified_multidataset_model.keras"
    model.save(str(primary_candidate_file))
    logger.info("Primary candidate model saved to: %s", primary_candidate_file)

    root_model_path = Path("unified_plant_disease_model.keras").resolve()
    shutil.copyfile(primary_candidate_file, root_model_path)
    logger.info("Saved unified model to root: %s", root_model_path)

    candidate_hash = compute_sha256(primary_candidate_file)
    model_size_mb = primary_candidate_file.stat().st_size / (1024 * 1024)

    # 9. Evaluate on Held-Out Test Set
    logger.info("--- Evaluating Unified Model on Held-Out Test Set (%d images) ---", len(X_test))
    eval_t0 = time.time()
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_pred_probs = model.predict(test_ds, verbose=0)
    eval_elapsed = time.time() - eval_t0
    avg_latency_ms = (eval_elapsed / len(X_test)) * 1000

    y_pred = np.argmax(y_pred_probs, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    per_class_results = {}
    for i, cname in enumerate(class_names):
        per_class_results[cname] = {
            "precision": float(round(precision[i], 4)),
            "recall": float(round(recall[i], 4)),
            "f1_score": float(round(f1[i], 4)),
            "test_support": int(support[i]),
        }

    eval_report = {
        "candidate_model": str(primary_candidate_file.name),
        "candidate_sha256": candidate_hash,
        "model_size_mb": round(model_size_mb, 2),
        "dataset": "Multi-Crop Plant Disease (Blackgram, Chilli, Tea, Citrus)",
        "num_classes": len(class_names),
        "classes": class_names,
        "total_images_scanned": len(valid_recs) + len(corrupted_recs),
        "valid_images": len(valid_recs),
        "corrupted_or_duplicates": len(corrupted_recs),
        "split_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "test_metrics": {
            "test_loss": float(round(test_loss, 4)),
            "test_accuracy": float(round(test_acc, 4)),
            "macro_precision": float(round(macro_p, 4)),
            "macro_recall": float(round(macro_r, 4)),
            "macro_f1": float(round(macro_f1, 4)),
            "weighted_precision": float(round(weighted_p, 4)),
            "weighted_recall": float(round(weighted_r, 4)),
            "weighted_f1": float(round(weighted_f1, 4)),
            "avg_latency_ms": round(avg_latency_ms, 2),
        },
        "per_class_metrics": per_class_results,
        "confusion_matrix": cm.tolist(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report_path = candidate_path / "unified_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Evaluation report saved to: %s", report_path)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("           UNIFIED MULTI-CROP MODEL EVALUATION REPORT")
    print("=" * 80)
    print(f"Model File:        {root_model_path.name}")
    print(f"SHA-256:           {candidate_hash}")
    print(f"Model Size:        {model_size_mb:.2f} MB")
    print(f"Test Accuracy:     {test_acc * 100:.2f}%")
    print(f"Macro Precision:   {macro_p * 100:.2f}%")
    print(f"Macro Recall:      {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print(f"Avg Latency:       {avg_latency_ms:.2f} ms/image")
    print("-" * 80)
    print(f"{'Class Name':<38} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
    print("-" * 80)
    for cname, mets in per_class_results.items():
        print(f"{cname:<38} | {mets['precision']*100:>8.2f}% | {mets['recall']*100:>8.2f}% | {mets['f1_score']*100:>8.2f}% | {mets['test_support']}")
    print("=" * 80 + "\n")

    # 10. Verification of Artifacts
    logger.info("--- Verifying Saved Artifacts ---")
    assert root_model_path.exists() and root_model_path.stat().st_size > 0, "Root model missing"
    root_class_names_path = Path("unified_class_names.json").resolve()
    assert root_class_names_path.exists(), "Root class names missing"

    with open(root_class_names_path, "r", encoding="utf-8") as f:
        loaded_classes = json.load(f)
    print(f"[VERIFICATION] Saved class names file: {root_class_names_path} ({len(loaded_classes)} classes)")
    print(f"[VERIFICATION] Saved model file: {root_model_path} ({root_model_path.stat().st_size / (1024*1024):.2f} MB)")
    print("[VERIFICATION] Loading saved model to verify inference integrity...")
    test_load_model = tf.keras.models.load_model(str(root_model_path))
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    dummy_pred = test_load_model.predict(dummy_input, verbose=0)
    print(f"[VERIFICATION] Sanity inference passed! Shape: {dummy_pred.shape}, Probabilities sum: {dummy_pred.sum():.4f}")
    print("[VERIFICATION] Unified model training and verification completed successfully!\n")

    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unified Multi-Crop Plant Disease Classifier")
    parser.add_argument("--candidate_dir", type=str, default="models/candidates", help="Output directory for candidate artifacts")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs for Stage 1")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_per_class", type=int, default=200, help="Max images per class for balanced training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_and_evaluate_unified(
        output_dir=args.candidate_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
