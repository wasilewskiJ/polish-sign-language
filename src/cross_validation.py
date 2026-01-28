"""
Cross-validation logic for different model types.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
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

from .config import Config
from .models import build_keras_mlp, build_cnn
from .evaluation import compute_weights, per_class_metrics, save_confusion_matrix
from .data_loader import get_augmented_data_for_fold


def run_sklearn_cv(X, y, model, model_name, n_splits=5, seed=666, results_dir=None, 
                   augmented_data=None, use_augmentation=False):

    print(f"\n{'='*70}")
    print(f"{model_name} - {n_splits}-Fold Stratified Cross-Validation")
    if use_augmentation and augmented_data:
        print("With data augmentation")
    print(f"{'='*70}")
    
    le = LabelEncoder()
    
    if augmented_data and use_augmentation:
        y_original_only = y
    
    y_encoded = le.fit_transform(y)
    class_labels = le.classes_
    n_classes = len(class_labels)
    
    print(f"Classes: {n_classes} ({', '.join(class_labels)})")
    print(f"Total samples: {len(X)}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
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
    
    fold_iterator = enumerate(skf.split(X, y_encoded), 1)
    
    for fold, (train_idx, val_idx) in fold_iterator:
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        if augmented_data and use_augmentation:
            X_train, y_train, X_val, y_val = get_augmented_data_for_fold(
                augmented_data, train_idx, val_idx, use_augmentation=True
            )
            y_train = le.transform(y_train)
            y_val = le.transform(y_val)
            print(f"Train: {len(X_train)} samples (with augmentation)")
            print(f"Val:   {len(X_val)} samples (original only)")
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            print(f"Train: {len(X_train)} samples")
            print(f"Val:   {len(X_val)} samples")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model_clone = clone(model)
        
        print("Training...", end=" ", flush=True)
        model_clone.fit(X_train_scaled, y_train)
        print("Done")
        
        y_pred = model_clone.predict(X_val_scaled)
        
        label_ids = list(range(n_classes))
        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', labels=label_ids, zero_division=0)
        prec_macro = precision_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        rec_macro = recall_score(y_val, y_pred, average='macro', labels=label_ids, zero_division=0)
        
        cm = confusion_matrix(y_val, y_pred, labels=label_ids)
        per_class = per_class_metrics(cm, class_labels)
        
        fold_results['accuracies'].append(acc)
        fold_results['balanced_accuracies'].append(bal_acc)
        fold_results['f1_macros'].append(f1_macro)
        fold_results['f1_weighteds'].append(f1_weighted)
        fold_results['precisions_macro'].append(prec_macro)
        fold_results['recalls_macro'].append(rec_macro)
        fold_results['confusion_matrices'].append(cm)
        fold_results['per_class_metrics'].append(per_class)
        
        print(f"Results: Acc={acc:.4f}, F1-macro={f1_macro:.4f}")
        
        if results_dir:
            cm_path = results_dir / f"confusion_{model_name.lower().replace(' ', '_')}_fold{fold}.png"
            save_confusion_matrix(cm, class_labels, f"{model_name} - Fold {fold}/{n_splits}", cm_path)
    
    return _aggregate_results(fold_results, class_labels, model_name, n_splits, results_dir)


def run_keras_mlp_cv(X, y, n_splits=5, seed=666, results_dir=None,
                     augmented_data=None, use_augmentation=False):
    model_name = "Keras MLP"
    print(f"\n{'='*70}")
    print(f"{model_name} - {n_splits}-Fold Stratified Cross-Validation")
    if use_augmentation and augmented_data:
        print("With data augmentation")
    print(f"{'='*70}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_labels = le.classes_
    n_classes = len(class_labels)
    
    print(f"Classes: {n_classes}")
    print(f"Total samples: {len(X)}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
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
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        if augmented_data and use_augmentation:
            X_train, y_train, X_val, y_val = get_augmented_data_for_fold(
                augmented_data, train_idx, val_idx, use_augmentation=True
            )
            y_train = le.transform(y_train)
            y_val = le.transform(y_val)
            print(f"Train: {len(X_train)} samples (with augmentation)")
            print(f"Val:   {len(X_val)} samples (original only)")
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = build_keras_mlp(input_shape=X_train_scaled.shape[1], num_classes=n_classes)
        
        class_weights = compute_weights(y_train)
        
        print("Training...", end=" ", flush=True)
        model.fit(
            X_train_scaled, y_train,
            epochs=Config.KERAS_EPOCHS,
            batch_size=Config.KERAS_BATCH_SIZE,
            class_weight=class_weights,
            verbose=0,
        )
        print("Done")
        
        y_pred_proba = model.predict(X_val_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Compute and store metrics
        _compute_and_store_metrics(y_val, y_pred, n_classes, class_labels, fold_results)
        
        print(f"Results: Acc={fold_results['accuracies'][-1]:.4f}, F1-macro={fold_results['f1_macros'][-1]:.4f}")
        
        if results_dir:
            cm_path = results_dir / f"confusion_keras_mlp_fold{fold}.png"
            save_confusion_matrix(
                fold_results['confusion_matrices'][-1], 
                class_labels, 
                f"{model_name} - Fold {fold}/{n_splits}", 
                cm_path
            )
    
    return _aggregate_results(fold_results, class_labels, model_name, n_splits, results_dir)


def run_cnn_cv(raw_dir, n_splits=5, seed=666, img_size=(128, 128), results_dir=None,
               augmented_img_dir=None, use_augmentation=False):
    model_name = "CNN"
    print(f"\n{'='*70}")
    print(f"{model_name} - {n_splits}-Fold Stratified Cross-Validation")
    if use_augmentation and augmented_img_dir:
        print("With data augmentation")
    print(f"{'='*70}")
    
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise ValueError(f"Raw directory not found: {raw_dir}")
    
    all_files = []
    all_labels = []
    
    for label_dir in sorted(raw_dir.iterdir()):
        if label_dir.is_dir() and not label_dir.name.startswith('.'):
            label = label_dir.name
            files = sorted(label_dir.glob("*.jpg"))
            all_files.extend(files)
            all_labels.extend([label] * len(files))
    
    if not all_files:
        raise ValueError(f"No images found in {raw_dir}")
    
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    class_labels = le.classes_
    n_classes = len(class_labels)
    
    print(f"Classes: {n_classes}")
    print(f"Total samples: {len(all_files)}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
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
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, y_encoded), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        # Validation: always use original images
        files_val = all_files[val_idx]
        y_val = y_encoded[val_idx]
        
        # Training: use augmented if available, otherwise original
        if use_augmentation and augmented_img_dir:
            augmented_dir = Path(augmented_img_dir)
            
            files_train = []
            y_train = []
            
            for idx in train_idx:
                orig_file = all_files[idx]
                label = all_labels[idx]
                
                # Add original image
                files_train.append(orig_file)
                y_train.append(label)
                
                # Add augmented versions (aug0, aug1, aug2, ...)
                letter_dir = augmented_dir / label
                img_stem = orig_file.stem  # e.g., "K21"
                
                # Pattern: K21_aug0.jpg, K21_aug1.jpg, ...
                pattern = f"{img_stem}_aug*.jpg"
                aug_files = sorted(letter_dir.glob(pattern))
                
                if aug_files:
                    files_train.extend(aug_files)
                    y_train.extend([label] * len(aug_files))
            
            files_train = np.array(files_train)
            y_train = le.transform(y_train)
            
            print(f"Train: {len(files_train)} samples (with augmentation)")
            print(f"Val:   {len(files_val)} samples (original only)")
        else:
            files_train = all_files[train_idx]
            y_train = y_encoded[train_idx]
            print(f"Train: {len(files_train)} samples")
            print(f"Val:   {len(files_val)} samples")
        
        train_ds = _make_tf_dataset(files_train, y_train, img_size=img_size, batch_size=Config.CNN_BATCH_SIZE, shuffle=True, seed=seed)
        val_ds = _make_tf_dataset(files_val, y_val, img_size=img_size, batch_size=Config.CNN_BATCH_SIZE, shuffle=False)
        
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        model = build_cnn((img_size[0], img_size[1], 3), n_classes)
        
        class_weights = compute_weights(y_train)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=Config.CNN_EARLY_STOPPING_PATIENCE, 
                restore_best_weights=True
            ),
        ]
        
        print("Training...", end=" ", flush=True)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=Config.CNN_EPOCHS,
            verbose=0,
            callbacks=callbacks,
            class_weight=class_weights,
        )
        print("Done")
        
        y_pred_proba = model.predict(val_ds, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Compute and store metrics
        _compute_and_store_metrics(y_val, y_pred, n_classes, class_labels, fold_results)
        
        print(f"Results: Acc={fold_results['accuracies'][-1]:.4f}, F1-macro={fold_results['f1_macros'][-1]:.4f}")
        
        if results_dir:
            cm_path = results_dir / f"confusion_cnn_fold{fold}.png"
            save_confusion_matrix(
                fold_results['confusion_matrices'][-1], 
                class_labels, 
                f"{model_name} - Fold {fold}/{n_splits}", 
                cm_path
            )
    
    return _aggregate_results(fold_results, class_labels, model_name, n_splits, results_dir)


# ============================================================================

# ============================================================================

def _compute_and_store_metrics(y_true, y_pred, n_classes, class_labels, fold_results):
    label_ids = list(range(n_classes))
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=label_ids, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=label_ids, zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average='macro', labels=label_ids, zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', labels=label_ids, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    per_class = per_class_metrics(cm, class_labels)
    
    fold_results['accuracies'].append(acc)
    fold_results['balanced_accuracies'].append(bal_acc)
    fold_results['f1_macros'].append(f1_macro)
    fold_results['f1_weighteds'].append(f1_weighted)
    fold_results['precisions_macro'].append(prec_macro)
    fold_results['recalls_macro'].append(rec_macro)
    fold_results['confusion_matrices'].append(cm)
    fold_results['per_class_metrics'].append(per_class)


def _aggregate_results(fold_results, class_labels, model_name, n_splits, results_dir):
    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS - {model_name}")
    print(f"{'='*70}")
    
    results = {
        'model': model_name,
        'n_splits': n_splits,
        'n_classes': len(class_labels),
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
    
    aggregate_cm = np.sum(fold_results['confusion_matrices'], axis=0)
    results['aggregate_confusion_matrix'] = aggregate_cm
    
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
    
    if results_dir:
        cm_agg_path = results_dir / f"confusion_{model_name.lower().replace(' ', '_')}_aggregate.png"
        save_confusion_matrix(
            aggregate_cm, 
            class_labels, 
            f"{model_name} - Aggregate ({n_splits}-fold)", 
            cm_agg_path
        )
        results['confusion_matrix_path'] = str(cm_agg_path)
    
    print(f"\nAccuracy:        {results['acc_mean']:.4f} ± {results['acc_std']:.4f}")
    print(f"Balanced Acc:    {results['balanced_acc_mean']:.4f} ± {results['balanced_acc_std']:.4f}")
    print(f"F1-macro:        {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}")
    print(f"F1-weighted:     {results['f1_weighted_mean']:.4f} ± {results['f1_weighted_std']:.4f}")
    
    return results


def _load_image_for_cnn(path, label, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, label


def _make_tf_dataset(files, labels, img_size, batch_size, shuffle=False, seed=666):
    files = [str(p) for p in files]
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda f, l: _load_image_for_cnn(f, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
