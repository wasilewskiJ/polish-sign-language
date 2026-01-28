"""
Report generation module.
"""

from pathlib import Path
from datetime import datetime


def write_report(all_results, output_path, n_splits=5, total_samples=0, comparison_mode=False):
    lines = []
    lines.append("# Wyniki eksperymentow (Stratified K-Fold Cross-Validation)\n")
    lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    if comparison_mode:
        lines.append("## 🔬 Eksperymenty PORÓWNAWCZE: Z i BEZ augmentacji\n\n")
        lines.append("Ten raport porównuje wydajność modeli:\n")
        lines.append("- **Baseline**: Modele trenowane na oryginalnych danych\n")
        lines.append("- **Augmented**: Modele trenowane z augmentacją danych\n\n")
    
    lines.append(f"## Metoda: {n_splits}-Fold Stratified Cross-Validation\n\n")
    lines.append(f"Kazda klasa ma rowna reprezentacje w kazdym foldzie.\n\n")
    
    if comparison_mode:
        lines.append("**Augmentacja:**\n")
        lines.append("- Zbior treningowy (augmented): oryginaly + augmentacje (~6× wiecej danych)\n")
        lines.append("- Zbior walidacyjny: ZAWSZE tylko oryginaly (no data leakage)\n\n")
    
    lines.append(f"**Zbior danych:**\n")
    lines.append(f"- Liczba sampli (oryginalne): {total_samples}\n")
    lines.append(f"- Liczba klas: {all_results[0]['n_classes'] if all_results else 'N/A'}\n")
    lines.append(f"- Foldy: {n_splits}\n\n")
    
    lines.append("---\n\n")
    
    if comparison_mode:
        baseline_results = [r for r in all_results if "(Augmented)" not in r['model']]
        augmented_results = [r for r in all_results if "(Augmented)" in r['model']]
        
        lines.append("## Podsumowanie porównawcze\n\n")
        lines.append("| Model | Baseline Acc | Augmented Acc | Δ Acc | Baseline F1 | Augmented F1 | Δ F1 |\n")
        lines.append("|-------|--------------|---------------|-------|-------------|--------------|------|\n")
        
        for r_base in baseline_results:
            base_name = r_base['model']
            r_aug = next((r for r in augmented_results if r['model'].replace(" (Augmented)", "") == base_name), None)
            
            if r_aug:
                acc_base = r_base['acc_mean']
                acc_aug = r_aug['acc_mean']
                acc_delta = acc_aug - acc_base
                acc_delta_pct = (acc_delta / acc_base * 100) if acc_base > 0 else 0
                
                f1_base = r_base['f1_macro_mean']
                f1_aug = r_aug['f1_macro_mean']
                f1_delta = f1_aug - f1_base
                f1_delta_pct = (f1_delta / f1_base * 100) if f1_base > 0 else 0
                
                acc_arrow = "↑" if acc_delta > 0.001 else "↓" if acc_delta < -0.001 else "→"
                f1_arrow = "↑" if f1_delta > 0.001 else "↓" if f1_delta < -0.001 else "→"
                
                lines.append(
                    f"| {base_name} | {acc_base:.4f} | {acc_aug:.4f} | "
                    f"{acc_arrow} {acc_delta:+.4f} ({acc_delta_pct:+.2f}%) | "
                    f"{f1_base:.4f} | {f1_aug:.4f} | "
                    f"{f1_arrow} {f1_delta:+.4f} ({f1_delta_pct:+.2f}%) |\n"
                )
        
        lines.append("\n---\n\n")
    
    for result in all_results:
        lines.append(f"## {result['model']}\n\n")
        
        lines.append(f"### Metryki ({n_splits}-fold CV, srednia ± odchylenie standardowe)\n\n")
        lines.append(f"- **Accuracy**: {result['acc_mean']:.4f} ± {result['acc_std']:.4f}\n")
        lines.append(f"- **Balanced Accuracy**: {result['balanced_acc_mean']:.4f} ± {result['balanced_acc_std']:.4f}\n")
        lines.append(f"- **F1-macro**: {result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f}\n")
        lines.append(f"- **F1-weighted**: {result['f1_weighted_mean']:.4f} ± {result['f1_weighted_std']:.4f}\n")
        lines.append(f"- **Precision-macro**: {result['precision_macro_mean']:.4f} ± {result['precision_macro_std']:.4f}\n")
        lines.append(f"- **Recall-macro**: {result['recall_macro_mean']:.4f} ± {result['recall_macro_std']:.4f}\n\n")
        
        lines.append(f"### Wyniki dla kazdego foldu\n\n")
        for i, acc in enumerate(result['accuracies'], 1):
            lines.append(f"- Fold {i}: {acc:.4f}\n")
        lines.append("\n")
        
        if result.get('confusion_matrix_path'):
            lines.append(f"**Macierz pomylek (agregowana)**: {result['confusion_matrix_path']}\n\n")
        
        if result.get('per_class_metrics'):
            lines.append("### Metryki per-class (srednia z foldow)\n\n")
            lines.append("| Klasa | Support | Precision | Recall | F1 | Accuracy |\n")
            lines.append("|-------|---------|-----------|--------|----|-----------|\n")
            
            per_class = result['per_class_metrics']
            for label in sorted(per_class.keys()):
                m = per_class[label]
                lines.append(
                    f"| {label} | {m['support']} | "
                    f"{m['precision']:.4f} | {m['recall']:.4f} | "
                    f"{m['f1']:.4f} | {m['accuracy']:.4f} |\n"
                )
            lines.append("\n")
        
        lines.append("---\n\n")
    
    if comparison_mode:
        baseline_results = [r for r in all_results if "(Augmented)" not in r['model']]
        augmented_results = [r for r in all_results if "(Augmented)" in r['model']]
        
        lines.append("## Wnioski z porównania\n\n")
        
        avg_acc_delta = 0
        avg_f1_delta = 0
        count = 0
        
        for r_base in baseline_results:
            base_name = r_base['model']
            r_aug = next((r for r in augmented_results if r['model'].replace(" (Augmented)", "") == base_name), None)
            if r_aug:
                avg_acc_delta += (r_aug['acc_mean'] - r_base['acc_mean'])
                avg_f1_delta += (r_aug['f1_macro_mean'] - r_base['f1_macro_mean'])
                count += 1
        
        if count > 0:
            avg_acc_delta /= count
            avg_f1_delta /= count
            
            lines.append(f"- **Średnia zmiana Accuracy**: {avg_acc_delta:+.4f} ({avg_acc_delta*100:+.2f}%)\n")
            lines.append(f"- **Średnia zmiana F1-macro**: {avg_f1_delta:+.4f} ({avg_f1_delta*100:+.2f}%)\n")
            lines.append(f"- **Liczba porównanych modeli**: {count}\n\n")
            
            if avg_acc_delta > 0.001:
                lines.append("**Augmentacja danych przyniosła poprawę wydajności modeli.**\n\n")
            elif avg_acc_delta < -0.001:
                lines.append("**Augmentacja danych nie przyniosła spodziewanej poprawy.**\n\n")
            else:
                lines.append("**Augmentacja danych nie miała znaczącego wpływu na wydajność.**\n\n")
        
        lines.append("**Uwaga**: Wszystkie wyniki walidacji są na oryginalnych danych, ")
        lines.append("więc porównanie jest uczciwe i nie ma data leakage.\n\n")
    elif len(all_results) > 1:
        lines.append("## Porownanie modeli\n\n")
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
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"\nReport saved: {output_path}")


def write_augmentation_comparison_report(results_no_aug, results_with_aug, output_path):

    lines = []
    lines.append("# Porownanie: Augmentacja vs Bez Augmentacji\n\n")
    lines.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    lines.append("## Cel\n\n")
    lines.append("Analiza wplywu augmentacji danych na wydajnosc modeli rozpoznawania PSL.\n\n")
    
    lines.append("## Metodologia\n\n")
    lines.append("1. **Baseline**: Eksperymenty na oryginalnych danych (~2400 sampli)\n")
    lines.append("2. **Z augmentacją**: Eksperymenty z augmentowanymi danymi (~14400 sampli treningowych)\n")
    lines.append("3. **Walidacja**: Zawsze na oryginalnych danych (no data leakage)\n\n")
    
    lines.append("---\n\n")
    
    lines.append("## Podsumowanie wynikow\n\n")
    lines.append("| Model | Bez Aug (Acc) | Z Aug (Acc) | Zmiana | Bez Aug (F1) | Z Aug (F1) | Zmiana |\n")
    lines.append("|-------|---------------|-------------|--------|--------------|------------|--------|\n")
    
    results_no_aug_dict = {r['model']: r for r in results_no_aug}
    results_with_aug_dict = {r['model']: r for r in results_with_aug}
    
    common_models = set(results_no_aug_dict.keys()) & set(results_with_aug_dict.keys())
    
    for model_name in sorted(common_models):
        r_no = results_no_aug_dict[model_name]
        r_aug = results_with_aug_dict[model_name]
        
        acc_no = r_no['acc_mean']
        acc_aug = r_aug['acc_mean']
        acc_change = acc_aug - acc_no
        acc_change_pct = (acc_change / acc_no) * 100 if acc_no > 0 else 0
        
        f1_no = r_no['f1_macro_mean']
        f1_aug = r_aug['f1_macro_mean']
        f1_change = f1_aug - f1_no
        f1_change_pct = (f1_change / f1_no) * 100 if f1_no > 0 else 0
        
        acc_arrow = "↑" if acc_change > 0 else "↓" if acc_change < 0 else "→"
        f1_arrow = "↑" if f1_change > 0 else "↓" if f1_change < 0 else "→"
        
        lines.append(
            f"| {model_name} | "
            f"{acc_no:.4f} | {acc_aug:.4f} | "
            f"{acc_arrow} {acc_change:+.4f} ({acc_change_pct:+.2f}%) | "
            f"{f1_no:.4f} | {f1_aug:.4f} | "
            f"{f1_arrow} {f1_change:+.4f} ({f1_change_pct:+.2f}%) |\n"
        )
    
    lines.append("\n")
    
    lines.append("## Szczegoly per model\n\n")
    
    for model_name in sorted(common_models):
        r_no = results_no_aug_dict[model_name]
        r_aug = results_with_aug_dict[model_name]
        
        lines.append(f"### {model_name}\n\n")
        
        lines.append("| Metryka | Bez Augmentacji | Z Augmentacją | Zmiana |\n")
        lines.append("|---------|-----------------|---------------|--------|\n")
        
        metrics = [
            ('Accuracy', 'acc_mean', 'acc_std'),
            ('Balanced Accuracy', 'balanced_acc_mean', 'balanced_acc_std'),
            ('F1-macro', 'f1_macro_mean', 'f1_macro_std'),
            ('F1-weighted', 'f1_weighted_mean', 'f1_weighted_std'),
            ('Precision-macro', 'precision_macro_mean', 'precision_macro_std'),
            ('Recall-macro', 'recall_macro_mean', 'recall_macro_std'),
        ]
        
        for metric_name, mean_key, std_key in metrics:
            val_no = r_no[mean_key]
            std_no = r_no[std_key]
            val_aug = r_aug[mean_key]
            std_aug = r_aug[std_key]
            change = val_aug - val_no
            change_pct = (change / val_no) * 100 if val_no > 0 else 0
            
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            
            lines.append(
                f"| {metric_name} | "
                f"{val_no:.4f} ± {std_no:.4f} | "
                f"{val_aug:.4f} ± {std_aug:.4f} | "
                f"{arrow} {change:+.4f} ({change_pct:+.2f}%) |\n"
            )
        
        lines.append("\n")
    
    lines.append("---\n\n")
    
    lines.append("## Wnioski\n\n")
    
    avg_acc_improvement = sum(
        results_with_aug_dict[m]['acc_mean'] - results_no_aug_dict[m]['acc_mean']
        for m in common_models
    ) / len(common_models) if common_models else 0
    
    avg_f1_improvement = sum(
        results_with_aug_dict[m]['f1_macro_mean'] - results_no_aug_dict[m]['f1_macro_mean']
        for m in common_models
    ) / len(common_models) if common_models else 0
    
    lines.append(f"- **Srednia poprawa Accuracy**: {avg_acc_improvement:+.4f} ({avg_acc_improvement*100:+.2f}%)\n")
    lines.append(f"- **Srednia poprawa F1-macro**: {avg_f1_improvement:+.4f} ({avg_f1_improvement*100:+.2f}%)\n")
    lines.append(f"- **Liczba porownanych modeli**: {len(common_models)}\n\n")
    
    if avg_acc_improvement > 0:
        lines.append("**Augmentacja danych przyniosla poprawe wydajnosci modeli.**\n\n")
    elif avg_acc_improvement < 0:
        lines.append("**Augmentacja danych nie przyniosla spodziewanej poprawy.**\n\n")
    else:
        lines.append("**Augmentacja danych nie miala wplywu na wydajnosc.**\n\n")
    
    lines.append("---\n\n")
    lines.append("**Uwaga**: Wyniki walidacji sa zawsze na oryginalnych danych, ")
    lines.append("wiec porownanie jest uczciwe i nie ma data leakage.\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"\nAugmentation comparison report saved: {output_path}")
