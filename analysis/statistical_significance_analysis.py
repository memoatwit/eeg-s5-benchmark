#!/usr/bin/env python3
"""
Statistical Significance Testing for LOSO Results

Performs McNemar's test and paired t-test on 60-fold LOSO cross-validation results
to assess statistical significance of performance differences between S5 and EEGTransformer.
"""

import numpy as np
import json
from scipy import stats
from scipy.stats import ttest_rel
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

def load_loso_results(json_path):
    """Load LOSO results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_fold_results(data, segment_length="32s"):
    """Extract fold-level results for statistical testing"""
    results = {}
    
    for model_name in data['results'][segment_length]:
        model_data = data['results'][segment_length][model_name]
        results[model_name] = {
            'accuracies': model_data['fold_results']['accuracies'],
            'f1_scores': model_data['fold_results']['f1_scores'],
            'successful_folds': model_data['successful_folds']
        }
    
    return results

def align_results_for_comparison(s5_results, eegt_results):
    """
    Align results for statistical comparison
    Since EEGTransformer only has 20 folds vs S5's 60, we'll use the first 20 folds for comparison
    """
    n_comparison_folds = min(len(s5_results['accuracies']), len(eegt_results['accuracies']))
    
    s5_acc = np.array(s5_results['accuracies'][:n_comparison_folds])
    eegt_acc = np.array(eegt_results['accuracies'][:n_comparison_folds])
    
    s5_f1 = np.array(s5_results['f1_scores'][:n_comparison_folds])
    eegt_f1 = np.array(eegt_results['f1_scores'][:n_comparison_folds])
    
    return {
        'n_folds': n_comparison_folds,
        's5_accuracies': s5_acc,
        'eegt_accuracies': eegt_acc,
        's5_f1': s5_f1,
        'eegt_f1': eegt_f1
    }

def create_contingency_table_from_accuracies(acc1, acc2, threshold=0.5):
    """
    Create contingency table for McNemar's test from accuracy scores
    We'll consider a fold as 'correct' if accuracy > threshold
    """
    correct1 = acc1 > threshold
    correct2 = acc2 > threshold
    
    # Create 2x2 contingency table
    # [S5_correct & EEGT_correct, S5_correct & EEGT_incorrect]
    # [S5_incorrect & EEGT_correct, S5_incorrect & EEGT_incorrect]
    both_correct = np.sum(correct1 & correct2)
    s5_only = np.sum(correct1 & ~correct2)
    eegt_only = np.sum(~correct1 & correct2)
    both_incorrect = np.sum(~correct1 & ~correct2)
    
    contingency = np.array([[both_correct, s5_only],
                           [eegt_only, both_incorrect]])
    
    return contingency

def paired_t_test(metric1, metric2, metric_name):
    """Perform paired t-test"""
    t_stat, p_value = ttest_rel(metric1, metric2)
    
    mean1 = np.mean(metric1)
    mean2 = np.mean(metric2)
    std1 = np.std(metric1, ddof=1)
    std2 = np.std(metric2, ddof=1)
    
    # Effect size (Cohen's d for paired samples)
    diff = metric1 - metric2
    cohen_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return {
        'metric': metric_name,
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_s5': mean1,
        'mean_eegt': mean2,
        'std_s5': std1,
        'std_eegt': std2,
        'mean_difference': mean1 - mean2,
        'cohens_d': cohen_d,
        'significant': p_value < 0.05
    }

def mcnemar_test(contingency, metric_name):
    """Perform McNemar's test"""
    try:
        result = mcnemar(contingency, exact=True)
        return {
            'metric': metric_name,
            'statistic': result.statistic,
            'p_value': result.pvalue,
            'contingency_table': contingency.tolist(),
            'significant': result.pvalue < 0.05
        }
    except Exception as e:
        print(f"McNemar test failed for {metric_name}: {e}")
        return None

def comprehensive_statistical_analysis():
    """Run comprehensive statistical analysis on LOSO results"""
    
    print("🔬 Statistical Significance Analysis of LOSO Results")
    print("=" * 60)
    
    # Load results
    results_path = "/home/ergezerm/eeg_25/extension_results_optimized/optimized_loso_final_20250720_180906.json"
    data = load_loso_results(results_path)
    
    # Extract fold results
    fold_results = extract_fold_results(data)
    
    print(f"📊 Loaded Results:")
    for model, results in fold_results.items():
        print(f"   • {model}: {results['successful_folds']} successful folds")
    
    # Align results for comparison
    s5_results = fold_results['S5']
    eegt_results = fold_results['EEGTransformer_Lightweight']
    
    aligned = align_results_for_comparison(s5_results, eegt_results)
    print(f"\\n🔄 Aligned {aligned['n_folds']} folds for statistical comparison")
    
    # Statistical Tests Results
    statistical_results = {
        'experiment_info': {
            'comparison_folds': aligned['n_folds'],
            'models_compared': ['S5', 'EEGTransformer_Lightweight'],
            'metrics_tested': ['Accuracy', 'F1-Score']
        }
    }
    
    print(f"\n📈 PAIRED T-TEST RESULTS")
    print("-" * 40)
    
    # Paired t-test for accuracy
    acc_ttest = paired_t_test(aligned['s5_accuracies'], aligned['eegt_accuracies'], 'Accuracy')
    statistical_results['paired_t_test_accuracy'] = acc_ttest
    
    print(f"\\n🎯 ACCURACY COMPARISON:")
    print(f"   S5 Mean ± SD: {acc_ttest['mean_s5']:.3f} ± {acc_ttest['std_s5']:.3f}")
    print(f"   EEGT Mean ± SD: {acc_ttest['mean_eegt']:.3f} ± {acc_ttest['std_eegt']:.3f}")
    print(f"   Mean Difference: {acc_ttest['mean_difference']:.3f}")
    print(f"   t-statistic: {acc_ttest['t_statistic']:.3f}")
    print(f"   p-value: {acc_ttest['p_value']:.6f}")
    print(f"   Cohen's d: {acc_ttest['cohens_d']:.3f}")
    print(f"   Significant: {'✅ YES' if acc_ttest['significant'] else '❌ NO'}")
    
    # Paired t-test for F1-score
    f1_ttest = paired_t_test(aligned['s5_f1'], aligned['eegt_f1'], 'F1-Score')
    statistical_results['paired_t_test_f1'] = f1_ttest
    
    print(f"\\n🎯 F1-SCORE COMPARISON:")
    print(f"   S5 Mean ± SD: {f1_ttest['mean_s5']:.3f} ± {f1_ttest['std_s5']:.3f}")
    print(f"   EEGT Mean ± SD: {f1_ttest['mean_eegt']:.3f} ± {f1_ttest['std_eegt']:.3f}")
    print(f"   Mean Difference: {f1_ttest['mean_difference']:.3f}")
    print(f"   t-statistic: {f1_ttest['t_statistic']:.3f}")
    print(f"   p-value: {f1_ttest['p_value']:.6f}")
    print(f"   Cohen's d: {f1_ttest['cohens_d']:.3f}")
    print(f"   Significant: {'✅ YES' if f1_ttest['significant'] else '❌ NO'}")
    
    print(f"\\n🔄 MCNEMAR'S TEST RESULTS")
    print("-" * 40)
    
    # McNemar's test for accuracy (threshold = 0.5)
    acc_contingency = create_contingency_table_from_accuracies(
        aligned['s5_accuracies'], aligned['eegt_accuracies'], threshold=0.5
    )
    acc_mcnemar = mcnemar_test(acc_contingency, 'Accuracy (>50%)')
    if acc_mcnemar:
        statistical_results['mcnemar_test_accuracy'] = acc_mcnemar
        
        print(f"\\n📊 ACCURACY CONTINGENCY TABLE (>50% threshold):")
        print(f"   [[S5_correct & EEGT_correct, S5_correct & EEGT_incorrect],")
        print(f"    [S5_incorrect & EEGT_correct, S5_incorrect & EEGT_incorrect]]")
        print(f"   {acc_mcnemar['contingency_table']}")
        print(f"   McNemar statistic: {acc_mcnemar['statistic']:.3f}")
        print(f"   p-value: {acc_mcnemar['p_value']:.6f}")
        print(f"   Significant: {'✅ YES' if acc_mcnemar['significant'] else '❌ NO'}")
    
    # Alternative threshold for accuracy
    acc_contingency_60 = create_contingency_table_from_accuracies(
        aligned['s5_accuracies'], aligned['eegt_accuracies'], threshold=0.6
    )
    acc_mcnemar_60 = mcnemar_test(acc_contingency_60, 'Accuracy (>60%)')
    if acc_mcnemar_60:
        statistical_results['mcnemar_test_accuracy_60'] = acc_mcnemar_60
        
        print(f"\\n📊 ACCURACY CONTINGENCY TABLE (>60% threshold):")
        print(f"   {acc_mcnemar_60['contingency_table']}")
        print(f"   McNemar statistic: {acc_mcnemar_60['statistic']:.3f}")
        print(f"   p-value: {acc_mcnemar_60['p_value']:.6f}")
        print(f"   Significant: {'✅ YES' if acc_mcnemar_60['significant'] else '❌ NO'}")
    
    # McNemar's test for F1-score (threshold = 0.4)
    f1_contingency = create_contingency_table_from_accuracies(
        aligned['s5_f1'], aligned['eegt_f1'], threshold=0.4
    )
    f1_mcnemar = mcnemar_test(f1_contingency, 'F1-Score (>40%)')
    if f1_mcnemar:
        statistical_results['mcnemar_test_f1'] = f1_mcnemar
        
        print(f"\\n📊 F1-SCORE CONTINGENCY TABLE (>40% threshold):")
        print(f"   {f1_mcnemar['contingency_table']}")
        print(f"   McNemar statistic: {f1_mcnemar['statistic']:.3f}")
        print(f"   p-value: {f1_mcnemar['p_value']:.6f}")
        print(f"   Significant: {'✅ YES' if f1_mcnemar['significant'] else '❌ NO'}")
    
    print(f"\\n📝 STATISTICAL SUMMARY")
    print("=" * 60)
    
    # Overall summary
    significant_tests = []
    if acc_ttest['significant']:
        significant_tests.append('Accuracy (paired t-test)')
    if f1_ttest['significant']:
        significant_tests.append('F1-score (paired t-test)')
    if acc_mcnemar and acc_mcnemar['significant']:
        significant_tests.append('Accuracy (McNemar test)')
    if f1_mcnemar and f1_mcnemar['significant']:
        significant_tests.append('F1-score (McNemar test)')
    
    print(f"✅ Statistically significant differences found in:")
    for test in significant_tests:
        print(f"   • {test}")
    
    if not significant_tests:
        print(f"❌ No statistically significant differences found at α = 0.05")
    
    # Effect size interpretation
    print(f"\\n📏 EFFECT SIZE INTERPRETATION (Cohen's d):")
    for metric, result in [('Accuracy', acc_ttest), ('F1-Score', f1_ttest)]:
        d = abs(result['cohens_d'])
        if d < 0.2:
            effect = "negligible"
        elif d < 0.5:
            effect = "small"
        elif d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        print(f"   • {metric}: {result['cohens_d']:.3f} ({effect} effect)")
    
    # Save results (convert numpy types for JSON serialization)
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    # Convert all values recursively
    def convert_dict_for_json(d):
        if isinstance(d, dict):
            return {k: convert_dict_for_json(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict_for_json(v) for v in d]
        else:
            return convert_for_json(d)
    
    json_safe_results = convert_dict_for_json(statistical_results)
    
    output_path = "/home/ergezerm/eeg_25/statistical_significance_results.json"
    with open(output_path, 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    
    print(f"\\n💾 Results saved to: {output_path}")
    
    return statistical_results

if __name__ == "__main__":
    results = comprehensive_statistical_analysis()
