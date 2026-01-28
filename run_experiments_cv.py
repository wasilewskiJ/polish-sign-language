"""
Cross-Validation Experiments for Polish Sign Language Recognition
Uses Stratified K-Fold CV as recommended by professor.
"""

import sys
import numpy as np
import cv2
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from translator.landmarks import extract_landmarks, compute_landmark_relationships
from translator.preprocess import augment_image

warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    category=UserWarning,
    module="sklearn.metrics",
)


# ============================================================================
# DATA LOADING & AUGMENTATION
# ============================================================================

def augment_and_extract_features(img, num_augmentations=5):
    """
    Apply augmentations to an image and extract landmark features from each.
    
    Args:
        img: Original image (numpy array)
        num_augmentations: Number of augmented versions to create
    
    Returns:
        List of feature vectors (each 78-dim), or empty list if augmentation fails
    """
    features_list = []
    
    for _ in range(num_augmentations):
        try:
            # Apply augmentation
            aug_img = augment_image(img)
            
            # Extract landmarks from augmented image
            detection_result = extract_landmarks(aug_img)
            
            if not detection_result or not detection_result.hand_landmarks:
                continue
            
            # Compute features
            relationships = compute_landmark_relationships(detection_result)
            hand_landmarks = detection_result.hand_landmarks[0]
            raw_coords = np.array([
                [lm.x, lm.y, lm.z] for lm in hand_landmarks
            ]).flatten()
            
            features = np.concatenate([raw_coords, relationships])
            features_list.append(features)
            
        except Exception:
            continue
    
    return features_list


def load_raw_data_and_extract_features(raw_dir="experiments/data/raw"):
    """
    Load all images from raw/ and extract landmark features.
    
    Args:
        raw_dir: Path to raw data directory with structure: raw/A/*.jpg, raw/B/*.jpg, etc.
    
    Returns:
        X: Features array (N, 78) - 63 raw coords + 15 relationships
        y: Labels array (N,) - letter strings
        image_paths: List of image paths corresponding to X, y
    """
    raw_path = Path(raw_dir)
    print(f"Loading data from {raw_path}...")
    
    X_list = []
    y_list = []
    image_paths = []
    skipped = {}
    
    for letter_dir in sorted(raw_path.iterdir()):
        if not letter_dir.is_dir() or letter_dir.name.startswith('.'):
            continue
            
        letter = letter_dir.name
        print(f"  Processing {letter}...", end=" ", flush=True)
        
        count = 0
        letter_skipped = 0
        
        for img_path in sorted(letter_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                letter_skipped += 1
                continue
            
            try:
                # Extract landmarks using MediaPipe
                detection_result = extract_landmarks(img)
                
                if not detection_result or not detection_result.hand_landmarks:
                    letter_skipped += 1
                    continue
                
                # Compute relationships (15 features)
                relationships = compute_landmark_relationships(detection_result)
                
                # Get raw coordinates (21 landmarks × 3 coords = 63 features)
                hand_landmarks = detection_result.hand_landmarks[0]
                raw_coords = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks
                ]).flatten()
                
                # Concatenate: 63 raw + 15 relationships = 78 features
                features = np.concatenate([raw_coords, relationships])
                
                X_list.append(features)
                y_list.append(letter)
                image_paths.append(img_path)
                count += 1
                
            except Exception as e:
                print(f"\n    Error processing {img_path.name}: {e}")
                letter_skipped += 1
                continue
        
        print(f"{count} samples loaded", end="")
        if letter_skipped > 0:
            print(f" ({letter_skipped} skipped)", end="")
            skipped[letter] = letter_skipped
        print()
    
    if not X_list:
        raise ValueError("No data loaded! Check if raw directory has images.")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n✅ Total: {len(X)} samples, {len(np.unique(y))} classes")
    
    if skipped:
        print(f"⚠️  Skipped samples per class: {skipped}")
    
    # Print class distribution
    class_counts = Counter(y)
    print("\nClass distribution:")
    for label in sorted(class_counts.keys()):
        print(f"  {label}: {class_counts[label]} samples")
    
    return X, y, image_paths


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def compute_weights(y_encoded):
    """Compute class weights for imbalanced datasets."""
    classes = np.unique(y_encoded)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_encoded)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def per_class_metrics(cm, class_labels):
    """Compute per-class precision, recall, F1, accuracy from confusion matrix."""
    metrics = {}
    for idx, label in enumerate(class_labels):
        tp = cm[idx, idx]
        support = cm[idx].sum()
        predicted = cm[:, idx].sum()
        precision = float(tp / predicted) if predicted else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics[label] = {
            "support": int(support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": recall,
        }
    return metrics


def save_confusion_matrix(cm, class_labels, title: str, path: Path):
    """Save confusion matrix as PNG image."""
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90)
    ax.set_yticklabels(class_labels)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value > 0:
                ax.text(
                    j,
                    i,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if value > cm.max() * 0.5 else "black",
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved confusion matrix: {path.name}")


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_stratified_kfold_cv(
    X, y, model, model_name, n_splits=5, seed=42, results_dir=None,
    use_augmentation=False, image_paths=None, num_augmentations=5, raw_dir=None
):
    """
    Run Stratified K-Fold Cross-Validation for a model.
    
    Args:
        X: Features (N, 78)
        y: Labels (N,) - string labels
        model: sklearn model instance
        model_name: Name for reporting
        n_splits: Number of folds (5 or 10)
        seed: Random state
        results_dir: Directory to save confusion matrices
        use_augmentation: If True, augment training data on-the-fly
        image_paths: List of image paths (required if use_augmentation=True)
        num_augmentations: Number of augmented samples per original image
        raw_dir: Raw data directory (required if use_augmentation=True)
    
    Returns:
        dict with results including mean ± std for metrics
    """
    print(f"\n{'='*70}")
    print(f"{model_name} - {n_splits}-Fold Stratified Cross-Validation")
    print(f"{'='*70}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_labels = le.classes_
    n_classes = len(class_labels)
    
    print(f"Classes: {n_classes} ({', '.join(class_labels)})")
    print(f"Total samples: {len(X)}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Storage for results across folds
    fold_results = {
        'accuracies': [],
        'balanced_accuracies': [],
        'f1_macros': [],
        'f1_weighteds': [],
        'precisions_macro': [],
        'recalls_macro': [],
        'confusion_matrices': [],
        'per_class_metrics': [],
    }
    
    # Run CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        # Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        print(f"Train: {len(X_train)} samples (original)")
        print(f"Val:   {len(X_val)} samples")
        
        # Apply augmentation if requested (only to training set!)
        if use_augmentation and image_paths is not None:
            print(f"Applying augmentation ({num_augmentations} per sample)...", end=" ", flush=True)
            
            X_train_aug_list = []
            y_train_aug_list = []
            
            # Keep original samples
            for i in range(len(X_train)):
                X_train_aug_list.append(X_train[i])
                y_train_aug_list.append(y_train[i])
            
            # Add augmented samples
            train_paths = [image_paths[idx] for idx in train_idx]
            for img_path, label in zip(train_paths, y_train):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                aug_features = augment_and_extract_features(img, num_augmentations)
                for feat in aug_features:
                    X_train_aug_list.append(feat)
                    y_train_aug_list.append(label)
            
            X_train = np.array(X_train_aug_list)
            y_train = np.array(y_train_aug_list)
            
            print(f"✅ Now {len(X_train)} samples (original + augmented)")
        
        # Check stratification
        train_counts = Counter(y_train)
        val_counts = Counter(y_val)
        print(f"Train classes: min={min(train_counts.values())}, max={max(train_counts.values())}")
        print(f"Val classes:   min={min(val_counts.values())}, max={max(val_counts.values())}")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Clone model for this fold (fresh start)
        model_clone = clone(model)
        
        # Train
        print("Training...", end=" ", flush=True)
        model_clone.fit(X_train_scaled, y_train)
        print("✅")
        
        # Predict
        y_pred = model_clone.predict(X_val_scaled)
        
        # Compute metrics
        label_ids = list(range(n_classes))
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', labels=label_ids, zero_division=0)
        prec_macro = precision_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        rec_macro = recall_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=label_ids)
        per_class = per_class_metrics(cm, class_labels)
        
        # Store results
        fold_results['accuracies'].append(acc)
        fold_results['balanced_accuracies'].append(bal_acc)
        fold_results['f1_macros'].append(f1_macro)
        fold_results['f1_weighteds'].append(f1_weighted)
        fold_results['precisions_macro'].append(prec_macro)
        fold_results['recalls_macro'].append(rec_macro)
        fold_results['confusion_matrices'].append(cm)
        fold_results['per_class_metrics'].append(per_class)
        
        # Print fold results
        print(f"Results:")
        print(f"  Accuracy:        {acc:.4f}")
        print(f"  Balanced Acc:    {bal_acc:.4f}")
        print(f"  F1-macro:        {f1_macro:.4f}")
        print(f"  F1-weighted:     {f1_weighted:.4f}")
        
        # Save confusion matrix for this fold
        if results_dir:
            cm_path = results_dir / f"confusion_{model_name.lower().replace(' ', '_')}_fold{fold}.png"
            save_confusion_matrix(
                cm, 
                class_labels, 
                f"{model_name} - Fold {fold}/{n_splits}",
                cm_path
            )
    
    # Aggregate results across folds
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS - {model_name}")
    print(f"{'='*70}")
    
    results = {
        'model': model_name,
        'n_splits': n_splits,
        'n_classes': n_classes,
        'class_labels': class_labels.tolist(),
        'accuracies': fold_results['accuracies'],
        'acc_mean': float(np.mean(fold_results['accuracies'])),
        'acc_std': float(np.std(fold_results['accuracies'])),
        'balanced_acc_mean': float(np.mean(fold_results['balanced_accuracies'])),
        'balanced_acc_std': float(np.std(fold_results['balanced_accuracies'])),
        'f1_macro_mean': float(np.mean(fold_results['f1_macros'])),
        'f1_macro_std': float(np.std(fold_results['f1_macros'])),
        'f1_weighted_mean': float(np.mean(fold_results['f1_weighteds'])),
        'f1_weighted_std': float(np.std(fold_results['f1_weighteds'])),
        'precision_macro_mean': float(np.mean(fold_results['precisions_macro'])),
        'precision_macro_std': float(np.std(fold_results['precisions_macro'])),
        'recall_macro_mean': float(np.mean(fold_results['recalls_macro'])),
        'recall_macro_std': float(np.std(fold_results['recalls_macro'])),
    }
    
    # Aggregate confusion matrix (sum across folds)
    aggregate_cm = np.sum(fold_results['confusion_matrices'], axis=0)
    results['aggregate_confusion_matrix'] = aggregate_cm
    
    # Aggregate per-class metrics (average across folds)
    aggregate_per_class = {}
    for label in class_labels:
        metrics_list = [fold_pc[label] for fold_pc in fold_results['per_class_metrics']]
        aggregate_per_class[label] = {
            'support': int(np.sum([m['support'] for m in metrics_list])),
            'precision': float(np.mean([m['precision'] for m in metrics_list])),
            'recall': float(np.mean([m['recall'] for m in metrics_list])),
            'f1': float(np.mean([m['f1'] for m in metrics_list])),
            'accuracy': float(np.mean([m['accuracy'] for m in metrics_list])),
        }
    results['per_class_metrics'] = aggregate_per_class
    
    # Save aggregate confusion matrix
    if results_dir:
        cm_agg_path = results_dir / f"confusion_{model_name.lower().replace(' ', '_')}_aggregate.png"
        save_confusion_matrix(
            aggregate_cm,
            class_labels,
            f"{model_name} - Aggregate ({n_splits}-fold)",
            cm_agg_path
        )
        results['confusion_matrix_path'] = str(cm_agg_path)
    
    # Print summary
    print(f"\nAccuracy:        {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
    print(f"Balanced Acc:    {results['balanced_acc_mean']:.4f} ± {results['balanced_acc_std']:.4f}")
    print(f"F1-macro:        {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
    print(f"F1-weighted:     {results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}")
    print(f"Precision-macro: {results['precision_macro_mean']:.4f} ± {results['precision_macro_std']:.4f}")
    print(f"Recall-macro:    {results['recall_macro_mean']:.4f} ± {results['recall_macro_std']:.4f}")
    
    return results


# ============================================================================
# KERAS MLP MODEL
# ============================================================================

def build_keras_model(input_shape, num_classes):
    """Build Keras MLP model for landmark features."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.005),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.005),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_keras_cv(X, y, n_splits=5, seed=42, results_dir=None,
                 use_augmentation=False, image_paths=None, num_augmentations=5):
    """
    Run CV for Keras MLP model.
    
    Note: Keras models need special handling as they can't be cloned like sklearn models.
    We rebuild the model for each fold.
    
    Args:
        use_augmentation: If True, augment training data on-the-fly
        image_paths: List of image paths (required if use_augmentation=True)
        num_augmentations: Number of augmented samples per original image
    """
    print(f"\n{'='*70}")
    print(f"Keras MLP - {n_splits}-Fold Stratified Cross-Validation")
    print(f"{'='*70}")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_labels = le.classes_
    n_classes = len(class_labels)
    
    print(f"Classes: {n_classes}")
    print(f"Total samples: {len(X)}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Storage
    fold_results = {
        'accuracies': [],
        'balanced_accuracies': [],
        'f1_macros': [],
        'f1_weighteds': [],
        'precisions_macro': [],
        'recalls_macro': [],
        'confusion_matrices': [],
        'per_class_metrics': [],
    }
    
    # Run CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        # Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        print(f"Train: {len(X_train)} samples (original)")
        print(f"Val:   {len(X_val)} samples")
        
        # Apply augmentation if requested (only to training set!)
        if use_augmentation and image_paths is not None:
            print(f"Applying augmentation ({num_augmentations} per sample)...", end=" ", flush=True)
            
            X_train_aug_list = []
            y_train_aug_list = []
            
            # Keep original samples
            for i in range(len(X_train)):
                X_train_aug_list.append(X_train[i])
                y_train_aug_list.append(y_train[i])
            
            # Add augmented samples
            train_paths = [image_paths[idx] for idx in train_idx]
            for img_path, label in zip(train_paths, y_train):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                aug_features = augment_and_extract_features(img, num_augmentations)
                for feat in aug_features:
                    X_train_aug_list.append(feat)
                    y_train_aug_list.append(label)
            
            X_train = np.array(X_train_aug_list)
            y_train = np.array(y_train_aug_list)
            
            print(f"✅ Now {len(X_train)} samples (original + augmented)")
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Build fresh model
        model = build_keras_model(input_shape=X_train_scaled.shape[1], num_classes=n_classes)
        
        # Compute class weights
        class_weights = compute_weights(y_train)
        
        # Train
        print("Training...", end=" ", flush=True)
        model.fit(
            X_train_scaled,
            y_train,
            epochs=50,
            batch_size=32,
            class_weight=class_weights,
            verbose=0,
        )
        print("✅")
        
        # Predict
        y_pred_proba = model.predict(X_val_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        label_ids = list(range(n_classes))
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', labels=label_ids, zero_division=0)
        prec_macro = precision_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        rec_macro = recall_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=label_ids)
        per_class = per_class_metrics(cm, class_labels)
        
        # Store
        fold_results['accuracies'].append(acc)
        fold_results['balanced_accuracies'].append(bal_acc)
        fold_results['f1_macros'].append(f1_macro)
        fold_results['f1_weighteds'].append(f1_weighted)
        fold_results['precisions_macro'].append(prec_macro)
        fold_results['recalls_macro'].append(rec_macro)
        fold_results['confusion_matrices'].append(cm)
        fold_results['per_class_metrics'].append(per_class)
        
        print(f"Results: Acc={acc:.4f}, F1-macro={f1_macro:.4f}")
        
        # Save confusion matrix
        if results_dir:
            cm_path = results_dir / f"confusion_keras_mlp_fold{fold}.png"
            save_confusion_matrix(cm, class_labels, f"Keras MLP - Fold {fold}/{n_splits}", cm_path)
    
    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS - Keras MLP")
    print(f"{'='*70}")
    
    results = {
        'model': 'Keras MLP',
        'n_splits': n_splits,
        'n_classes': n_classes,
        'class_labels': class_labels.tolist(),
        'accuracies': fold_results['accuracies'],
        'acc_mean': float(np.mean(fold_results['accuracies'])),
        'acc_std': float(np.std(fold_results['accuracies'])),
        'balanced_acc_mean': float(np.mean(fold_results['balanced_accuracies'])),
        'balanced_acc_std': float(np.std(fold_results['balanced_accuracies'])),
        'f1_macro_mean': float(np.mean(fold_results['f1_macros'])),
        'f1_macro_std': float(np.std(fold_results['f1_macros'])),
        'f1_weighted_mean': float(np.mean(fold_results['f1_weighteds'])),
        'f1_weighted_std': float(np.std(fold_results['f1_weighteds'])),
        'precision_macro_mean': float(np.mean(fold_results['precisions_macro'])),
        'precision_macro_std': float(np.std(fold_results['precisions_macro'])),
        'recall_macro_mean': float(np.mean(fold_results['recalls_macro'])),
        'recall_macro_std': float(np.std(fold_results['recalls_macro'])),
    }
    
    # Aggregate confusion matrix
    aggregate_cm = np.sum(fold_results['confusion_matrices'], axis=0)
    results['aggregate_confusion_matrix'] = aggregate_cm
    
    # Aggregate per-class metrics
    aggregate_per_class = {}
    for label in class_labels:
        metrics_list = [fold_pc[label] for fold_pc in fold_results['per_class_metrics']]
        aggregate_per_class[label] = {
            'support': int(np.sum([m['support'] for m in metrics_list])),
            'precision': float(np.mean([m['precision'] for m in metrics_list])),
            'recall': float(np.mean([m['recall'] for m in metrics_list])),
            'f1': float(np.mean([m['f1'] for m in metrics_list])),
            'accuracy': float(np.mean([m['accuracy'] for m in metrics_list])),
        }
    results['per_class_metrics'] = aggregate_per_class
    
    # Save aggregate confusion matrix
    if results_dir:
        cm_agg_path = results_dir / f"confusion_keras_mlp_aggregate.png"
        save_confusion_matrix(aggregate_cm, class_labels, f"Keras MLP - Aggregate ({n_splits}-fold)", cm_agg_path)
        results['confusion_matrix_path'] = str(cm_agg_path)
    
    print(f"\nAccuracy:        {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
    print(f"Balanced Acc:    {results['balanced_acc_mean']:.4f} ± {results['balanced_acc_std']:.4f}")
    print(f"F1-macro:        {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
    print(f"F1-weighted:     {results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}")
    
    return results


# ============================================================================
# CNN MODEL (on raw images)
# ============================================================================

def build_cnn_model(input_shape, num_classes):
    """Build lightweight CNN model for raw images."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_image_for_cnn(path, label, img_size=(128, 128)):
    """Load and preprocess image for CNN."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, label


def make_tf_dataset(files, labels, img_size=(128, 128), batch_size=32, shuffle=False, seed=42, use_augmentation=False):
    """
    Create tf.data.Dataset from file paths and labels.
    
    Args:
        use_augmentation: If True, apply random augmentations to training data
    """
    files = [str(p) for p in files]
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda f, l: load_image_for_cnn(f, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation if requested
    if use_augmentation:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.05),  # ±10 degrees
            tf.keras.layers.RandomTranslation(0.1, 0.1),  # ±10% translation
            tf.keras.layers.RandomZoom(0.1),  # ±10% zoom
            tf.keras.layers.RandomBrightness(0.2),  # ±20% brightness
        ])
        ds = ds.map(
            lambda img, lbl: (augmentation(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def run_cnn_cv(raw_dir, n_splits=5, seed=42, img_size=(128, 128), results_dir=None, use_augmentation=False):
    """
    Run CV for CNN model on raw images.
    
    Args:
        raw_dir: Path to raw images directory (experiments/data/raw)
        n_splits: Number of CV folds
        seed: Random seed
        img_size: Image size (height, width)
        results_dir: Directory to save results
        use_augmentation: If True, apply augmentation to training data
    """
    print(f"\n{'='*70}")
    print(f"CNN (Raw Images) - {n_splits}-Fold Stratified Cross-Validation")
    print(f"{'='*70}")
    
    # Load all raw images
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise ValueError(f"Raw directory not found: {raw_dir}")
    
    all_files = []
    all_labels = []
    
    for label_dir in sorted(raw_dir.iterdir()):
        if label_dir.is_dir():
            label = label_dir.name
            files = sorted(label_dir.glob("*.jpg"))
            all_files.extend(files)
            all_labels.extend([label] * len(files))
    
    if not all_files:
        raise ValueError(f"No images found in {raw_dir}")
    
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    class_labels = le.classes_
    n_classes = len(class_labels)
    
    print(f"Classes: {n_classes}")
    print(f"Total samples: {len(all_files)}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Storage
    fold_results = {
        'accuracies': [],
        'balanced_accuracies': [],
        'f1_macros': [],
        'f1_weighteds': [],
        'precisions_macro': [],
        'recalls_macro': [],
        'confusion_matrices': [],
        'per_class_metrics': [],
    }
    
    # Run CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, y_encoded), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        # Split
        files_train = all_files[train_idx]
        files_val = all_files[val_idx]
        y_train = y_encoded[train_idx]
        y_val = y_encoded[val_idx]
        
        print(f"Train: {len(files_train)} samples, Val: {len(files_val)} samples")
        
        # Create datasets
        train_ds = make_tf_dataset(files_train, y_train, img_size=img_size, batch_size=32, shuffle=True, seed=seed, use_augmentation=use_augmentation)
        val_ds = make_tf_dataset(files_val, y_val, img_size=img_size, batch_size=32, shuffle=False, use_augmentation=False)
        
        # Build fresh model
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        model = build_cnn_model((img_size[0], img_size[1], 3), n_classes)
        
        # Compute class weights
        class_weights = compute_weights(y_train)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
        ]
        
        # Train
        print("Training...", end=" ", flush=True)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            verbose=0,
            callbacks=callbacks,
            class_weight=class_weights,
        )
        print("✅")
        
        # Predict
        y_pred_proba = model.predict(val_ds, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        label_ids = list(range(n_classes))
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', labels=label_ids, zero_division=0)
        prec_macro = precision_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        rec_macro = recall_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=label_ids)
        per_class = per_class_metrics(cm, class_labels)
        
        # Store
        fold_results['accuracies'].append(acc)
        fold_results['balanced_accuracies'].append(bal_acc)
        fold_results['f1_macros'].append(f1_macro)
        fold_results['f1_weighteds'].append(f1_weighted)
        fold_results['precisions_macro'].append(prec_macro)
        fold_results['recalls_macro'].append(rec_macro)
        fold_results['confusion_matrices'].append(cm)
        fold_results['per_class_metrics'].append(per_class)
        
        print(f"Results: Acc={acc:.4f}, F1-macro={f1_macro:.4f}")
        
        # Save confusion matrix
        if results_dir:
            cm_path = results_dir / f"confusion_cnn_fold{fold}.png"
            save_confusion_matrix(cm, class_labels, f"CNN - Fold {fold}/{n_splits}", cm_path)
    
    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS - CNN")
    print(f"{'='*70}")
    
    results = {
        'model': 'CNN',
        'n_splits': n_splits,
        'n_classes': n_classes,
        'class_labels': class_labels.tolist(),
        'accuracies': fold_results['accuracies'],
        'acc_mean': float(np.mean(fold_results['accuracies'])),
        'acc_std': float(np.std(fold_results['accuracies'])),
        'balanced_acc_mean': float(np.mean(fold_results['balanced_accuracies'])),
        'balanced_acc_std': float(np.std(fold_results['balanced_accuracies'])),
        'f1_macro_mean': float(np.mean(fold_results['f1_macros'])),
        'f1_macro_std': float(np.std(fold_results['f1_macros'])),
        'f1_weighted_mean': float(np.mean(fold_results['f1_weighteds'])),
        'f1_weighted_std': float(np.std(fold_results['f1_weighteds'])),
        'precision_macro_mean': float(np.mean(fold_results['precisions_macro'])),
        'precision_macro_std': float(np.std(fold_results['precisions_macro'])),
        'recall_macro_mean': float(np.mean(fold_results['recalls_macro'])),
        'recall_macro_std': float(np.std(fold_results['recalls_macro'])),
    }
    
    # Aggregate confusion matrix
    aggregate_cm = np.sum(fold_results['confusion_matrices'], axis=0)
    results['aggregate_confusion_matrix'] = aggregate_cm
    
    # Aggregate per-class metrics
    aggregate_per_class = {}
    for label in class_labels:
        metrics_list = [fold_pc[label] for fold_pc in fold_results['per_class_metrics']]
        aggregate_per_class[label] = {
            'support': int(np.sum([m['support'] for m in metrics_list])),
            'precision': float(np.mean([m['precision'] for m in metrics_list])),
            'recall': float(np.mean([m['recall'] for m in metrics_list])),
            'f1': float(np.mean([m['f1'] for m in metrics_list])),
            'accuracy': float(np.mean([m['accuracy'] for m in metrics_list])),
        }
    results['per_class_metrics'] = aggregate_per_class
    
    # Save aggregate confusion matrix
    if results_dir:
        cm_agg_path = results_dir / f"confusion_cnn_aggregate.png"
        save_confusion_matrix(aggregate_cm, class_labels, f"CNN - Aggregate ({n_splits}-fold)", cm_agg_path)
        results['confusion_matrix_path'] = str(cm_agg_path)
    
    print(f"\nAccuracy:        {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
    print(f"Balanced Acc:    {results['balanced_acc_mean']:.4f} ± {results['balanced_acc_std']:.4f}")
    print(f"F1-macro:        {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
    print(f"F1-weighted:     {results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}")
    
    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def write_report(all_results, output_path: Path, n_splits=5, total_samples=0):
    """Generate markdown report for CV results."""
    lines = []
    lines.append("# Wyniki eksperymentow (Stratified K-Fold Cross-Validation)\n")
    lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"## Metoda: {n_splits}-Fold Stratified Cross-Validation\n\n")
    lines.append(f"Zgodnie z zaleceniami profesora, uzywamy stratyfikowanej walidacji krzyzowej.\n")
    lines.append(f"Kazda klasa ma rowna reprezentacje w kazdym foldzie.\n\n")
    lines.append(f"**Zbior danych:**\n")
    lines.append(f"- Liczba sampli: {total_samples}\n")
    lines.append(f"- Liczba klas: {all_results[0]['n_classes'] if all_results else 'N/A'}\n")
    lines.append(f"- Foldy: {n_splits}\n\n")
    
    lines.append("---\n\n")
    
def write_report(all_results, output_path: Path, n_splits=5, total_samples=0):
    """Generate markdown report for CV results."""
    lines = []
    lines.append("# Wyniki eksperymentow (Stratified K-Fold Cross-Validation)\n")
    lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"## Metoda: {n_splits}-Fold Stratified Cross-Validation\n\n")
    lines.append(f"Zgodnie z zaleceniami profesora, uzywamy stratyfikowanej walidacji krzyzowej.\n")
    lines.append(f"Kazda klasa ma rowna reprezentacje w kazdym foldzie.\n\n")
    lines.append(f"**Zbior danych:**\n")
    lines.append(f"- Liczba sampli: {total_samples}\n")
    lines.append(f"- Liczba klas: {all_results[0]['n_classes'] if all_results else 'N/A'}\n")
    lines.append(f"- Foldy: {n_splits}\n\n")
    
    lines.append("---\n\n")
    
    # Split results into no-aug and augmented
    no_aug_results = [r for r in all_results if 'no aug' in r['model']]
    aug_results = [r for r in all_results if 'augmented' in r['model']]
    
    # Part 1: Without augmentation
    if no_aug_results:
        lines.append("## CZĘŚĆ 1: Eksperymenty BEZ augmentacji\n\n")
        for result in no_aug_results:
            lines.append(f"### {result['model']}\n\n")
            
            lines.append(f"**Metryki ({n_splits}-fold CV, srednia ± odchylenie standardowe)**\n\n")
            lines.append(f"- **Accuracy**: {result['acc_mean']:.4f} ± {result['acc_std']:.4f}\n")
            lines.append(f"- **Balanced Accuracy**: {result['balanced_acc_mean']:.4f} ± {result['balanced_acc_std']:.4f}\n")
            lines.append(f"- **F1-macro**: {result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f}\n")
            lines.append(f"- **F1-weighted**: {result['f1_weighted_mean']:.4f} ± {result['f1_weighted_std']:.4f}\n")
            lines.append(f"- **Precision-macro**: {result['precision_macro_mean']:.4f} ± {result['precision_macro_std']:.4f}\n")
            lines.append(f"- **Recall-macro**: {result['recall_macro_mean']:.4f} ± {result['recall_macro_std']:.4f}\n\n")
            
            lines.append(f"**Wyniki dla kazdego foldu:**\n\n")
            for i, acc in enumerate(result['accuracies'], 1):
                lines.append(f"- Fold {i}: {acc:.4f}\n")
            lines.append("\n")
            
            if result.get('confusion_matrix_path'):
                lines.append(f"**Macierz pomylek (agregowana)**: `{Path(result['confusion_matrix_path']).name}`\n\n")
            
            lines.append("---\n\n")
    
    # Part 2: With augmentation
    if aug_results:
        lines.append("## CZĘŚĆ 2: Eksperymenty Z augmentacją\n\n")
        for result in aug_results:
            lines.append(f"### {result['model']}\n\n")
            
            lines.append(f"**Metryki ({n_splits}-fold CV, srednia ± odchylenie standardowe)**\n\n")
            lines.append(f"- **Accuracy**: {result['acc_mean']:.4f} ± {result['acc_std']:.4f}\n")
            lines.append(f"- **Balanced Accuracy**: {result['balanced_acc_mean']:.4f} ± {result['balanced_acc_std']:.4f}\n")
            lines.append(f"- **F1-macro**: {result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f}\n")
            lines.append(f"- **F1-weighted**: {result['f1_weighted_mean']:.4f} ± {result['f1_weighted_std']:.4f}\n")
            lines.append(f"- **Precision-macro**: {result['precision_macro_mean']:.4f} ± {result['precision_macro_std']:.4f}\n")
            lines.append(f"- **Recall-macro**: {result['recall_macro_mean']:.4f} ± {result['recall_macro_std']:.4f}\n\n")
            
            lines.append(f"**Wyniki dla kazdego foldu:**\n\n")
            for i, acc in enumerate(result['accuracies'], 1):
                lines.append(f"- Fold {i}: {acc:.4f}\n")
            lines.append("\n")
            
            if result.get('confusion_matrix_path'):
                lines.append(f"**Macierz pomylek (agregowana)**: `{Path(result['confusion_matrix_path']).name}`\n\n")
            
            lines.append("---\n\n")
    
    # Comparison table
    if len(all_results) > 1:
        lines.append("## PORÓWNANIE WSZYSTKICH MODELI\n\n")
        lines.append("| Model | Accuracy | F1-macro | F1-weighted |\n")
        lines.append("|-------|----------|----------|-------------|\n")
        
        for result in all_results:
            lines.append(
                f"| {result['model']} | "
                f"{result['acc_mean']:.4f} ± {result['acc_std']:.4f} | "
                f"{result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f} | "
                f"{result['f1_weighted_mean']:.4f} ± {result['f1_weighted_std']:.4f} |\n"
            )
        lines.append("\n")
        
        # Augmentation impact analysis
        no_aug_results = [r for r in all_results if 'no aug' in r['model']]
        aug_results = [r for r in all_results if 'augmented' in r['model']]
        
        if no_aug_results and aug_results:
            lines.append("## WPŁYW AUGMENTACJI NA WYNIKI\n\n")
            lines.append("Porównanie tego samego modelu przed i po augmentacji:\n\n")
            lines.append("| Model | No Aug Accuracy | Augmented Accuracy | Zmiana |\n")
            lines.append("|-------|-----------------|--------------------|---------|\n")
            
            # Match models by base name
            base_names = set()
            for r in no_aug_results:
                base_name = r['model'].replace(' (no aug)', '')
                base_names.add(base_name)
            
            for base_name in sorted(base_names):
                no_aug = next((r for r in no_aug_results if base_name in r['model']), None)
                aug = next((r for r in aug_results if base_name in r['model']), None)
                
                if no_aug and aug:
                    diff = aug['acc_mean'] - no_aug['acc_mean']
                    sign = "+" if diff > 0 else ""
                    lines.append(
                        f"| {base_name} | "
                        f"{no_aug['acc_mean']:.4f} ± {no_aug['acc_std']:.4f} | "
                        f"{aug['acc_mean']:.4f} ± {aug['acc_std']:.4f} | "
                        f"{sign}{diff:.4f} |\n"
                    )
            lines.append("\n")
    
    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"\n✅ Report saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main experiment runner with Stratified K-Fold CV."""
    print("="*70)
    print("POLISH SIGN LANGUAGE RECOGNITION - CROSS-VALIDATION EXPERIMENTS")
    print("="*70)
    print("\nMetoda: Stratified K-Fold Cross-Validation")
    print("Zgodnie z zaleceniami profesora\n")
    
    # Configuration
    n_splits = 5  # Can change to 10
    seed = 42
    num_augmentations = 5  # Number of augmented versions per original image
    
    # Get script directory and construct absolute paths
    script_dir = Path(__file__).parent.resolve()
    raw_dir = script_dir / "data" / "raw"
    results_dir = script_dir / "results_cv"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  K-folds: {n_splits}")
    print(f"  Random seed: {seed}")
    print(f"  Augmentations per sample: {num_augmentations}")
    print(f"  Raw data: {raw_dir}")
    print(f"  Results: {results_dir}")
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    X, y, image_paths = load_raw_data_and_extract_features(raw_dir)
    
    print(f"\nLoaded:")
    print(f"  X.shape: {X.shape}")
    print(f"  y.shape: {y.shape}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Models to test
    models = [
        (RandomForestClassifier(random_state=seed, n_estimators=100), "RandomForest"),
        (LogisticRegression(max_iter=1000, random_state=seed), "LogisticRegression"),
    ]
    
    all_results = []
    
    # ========================================================================
    # PART 1: Experiments WITHOUT augmentation
    # ========================================================================
    print("\n" + "="*70)
    print("PART 1: EXPERIMENTS WITHOUT AUGMENTATION")
    print("="*70)
    
    for model, model_name in models:
        result = run_stratified_kfold_cv(
            X, y,
            model=model,
            model_name=f"{model_name} (no aug)",
            n_splits=n_splits,
            seed=seed,
            results_dir=results_dir,
            use_augmentation=False
        )
        all_results.append(result)
    
    # Keras MLP without augmentation
    keras_result = run_keras_cv(
        X, y,
        n_splits=n_splits,
        seed=seed,
        results_dir=results_dir,
        use_augmentation=False
    )
    keras_result['model'] = 'Keras MLP (no aug)'
    all_results.append(keras_result)
    
    # CNN without augmentation
    cnn_result = run_cnn_cv(
        raw_dir=raw_dir,
        n_splits=n_splits,
        seed=seed,
        img_size=(128, 128),
        results_dir=results_dir,
        use_augmentation=False
    )
    cnn_result['model'] = 'CNN (no aug)'
    all_results.append(cnn_result)
    
    # ========================================================================
    # PART 2: Experiments WITH augmentation
    # ========================================================================
    print("\n" + "="*70)
    print("PART 2: EXPERIMENTS WITH AUGMENTATION")
    print("="*70)
    
    for model, model_name in models:
        result = run_stratified_kfold_cv(
            X, y,
            model=model,
            model_name=f"{model_name} (augmented)",
            n_splits=n_splits,
            seed=seed,
            results_dir=results_dir,
            use_augmentation=True,
            image_paths=image_paths,
            num_augmentations=num_augmentations,
            raw_dir=raw_dir
        )
        all_results.append(result)
    
    # Keras MLP with augmentation
    keras_result_aug = run_keras_cv(
        X, y,
        n_splits=n_splits,
        seed=seed,
        results_dir=results_dir,
        use_augmentation=True,
        image_paths=image_paths,
        num_augmentations=num_augmentations
    )
    keras_result_aug['model'] = 'Keras MLP (augmented)'
    all_results.append(keras_result_aug)
    
    # CNN with augmentation
    cnn_result_aug = run_cnn_cv(
        raw_dir=raw_dir,
        n_splits=n_splits,
        seed=seed,
        img_size=(128, 128),
        results_dir=results_dir,
        use_augmentation=True
    )
    cnn_result_aug['model'] = 'CNN (augmented)'
    all_results.append(cnn_result_aug)
    
    # Generate report
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    
    report_path = results_dir / "report_cv.md"
    write_report(
        all_results,
        report_path,
        n_splits=n_splits,
        total_samples=len(X)
    )
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY - COMPARISON OF ALL MODELS")
    print("="*70)
    print(f"\n{'Model':<30} {'Accuracy':<25} {'F1-macro':<25}")
    print("-"*80)
    
    for result in all_results:
        print(
            f"{result['model']:<30} "
            f"{result['acc_mean']:.4f} ± {result['acc_std']:.4f}        "
            f"{result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f}"
        )
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved in: {results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
