#!/usr/bin/env python3
"""
Comprehensive analysis for publication results
Focuses on sequence length effects and model comparison
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_results():
    """Load all available results"""
    results_data = {}
    
    # Look for both old and new results
    result_patterns = [
        '/home/ergezerm/eeg_25/results/*s',
        '/home/ergezerm/eeg_25/results/*s_publication'
    ]
    
    for pattern in result_patterns:
        result_dirs = glob.glob(pattern)
        
        for result_dir in result_dirs:
            seq_name = os.path.basename(result_dir)
            seq_length = int(seq_name.replace('s_publication', '').replace('s', ''))
            
            # Find JSON results
            json_files = glob.glob(f"{result_dir}/*results*.json")
            
            if json_files:
                latest_json = max(json_files, key=os.path.getctime)
                
                try:
                    with open(latest_json, 'r') as f:
                        results = json.load(f)
                    
                    results_data[seq_length] = results
                    print(f"📊 Loaded {seq_name}: {len(results.get('model_results', {}))} models")
                    
                except Exception as e:
                    print(f"❌ Error loading {latest_json}: {e}")
    
    return results_data

def create_publication_plots(results_data):
    """Create publication-quality plots"""
    
    if not results_data:
        print("❌ No results data available")
        return
    
    # Prepare data for plotting
    plot_data = []
    
    for seq_length, results in results_data.items():
        if 'model_results' in results:
            for model, metrics in results['model_results'].items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    plot_data.append({
                        'Sequence Length (s)': seq_length,
                        'Model': model,
                        'Accuracy (%)': metrics['accuracy'],
                        'Parameters': metrics.get('parameters', 0),
                        'Training Time (s)': metrics.get('training_time', 0)
                    })
    
    if not plot_data:
        print("❌ No valid plot data")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ICASSP EEG Movie Classification: Long Sequence Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy vs Sequence Length
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model].sort_values('Sequence Length (s)')
        ax1.plot(model_data['Sequence Length (s)'], model_data['Accuracy (%)'], 
                marker='o', linewidth=3, markersize=8, label=model)
    
    ax1.set_xlabel('Sequence Length (seconds)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Performance vs Sequence Length', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Bar plot for latest results
    latest_results = df[df['Sequence Length (s)'] == df['Sequence Length (s)'].max()]
    bars = ax2.bar(latest_results['Model'], latest_results['Accuracy (%)'], 
                   color=sns.color_palette("husl", len(latest_results)), alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    max_seq = latest_results['Sequence Length (s)'].iloc[0]
    ax2.set_title(f'Model Comparison at {max_seq}s Sequences', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, latest_results['Accuracy (%)']):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Model efficiency (Accuracy vs Parameters)
    scatter = ax3.scatter(df['Parameters'], df['Accuracy (%)'], 
                         c=df['Sequence Length (s)'], s=100, alpha=0.7, cmap='viridis')
    
    for _, row in df.iterrows():
        ax3.annotate(f"{row['Model']}\n{row['Sequence Length (s)']}s", 
                    (row['Parameters'], row['Accuracy (%)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Model Parameters', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Model Efficiency: Accuracy vs Parameters', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Sequence Length (s)', fontsize=11)
    
    # Plot 4: Training time analysis
    if df['Training Time (s)'].sum() > 0:
        pivot_time = df.pivot(index='Sequence Length (s)', columns='Model', values='Training Time (s)')
        pivot_time.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_xlabel('Sequence Length (seconds)', fontsize=12)
        ax4.set_ylabel('Training Time (seconds)', fontsize=12)
        ax4.set_title('Training Time by Model and Sequence Length', fontsize=14, fontweight='bold')
        ax4.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=0)
    else:
        ax4.text(0.5, 0.5, 'Training time data\nnot available', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=14)
        ax4.set_title('Training Time Analysis', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f'/home/ergezerm/eeg_25/paper_figures/publication_long_sequence_analysis_{timestamp}.png'
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    print(f"📊 Publication plot saved: {plot_file}")
    plt.close()
    
    return plot_file

def generate_summary_report(results_data):
    """Generate text summary for publication"""
    
    if not results_data:
        return "No results available"
    
    report = []
    report.append("ICASSP EEG Movie Classification: Long Sequence Analysis")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Sort by sequence length
    sorted_lengths = sorted(results_data.keys())
    
    report.append("📊 RESULTS SUMMARY:")
    report.append("-" * 40)
    
    for seq_length in sorted_lengths:
        results = results_data[seq_length]
        report.append(f"\n🎬 {seq_length}s Sequence Results:")
        
        if 'model_results' in results:
            # Sort models by accuracy
            model_accs = [(model, metrics['accuracy']) 
                         for model, metrics in results['model_results'].items()
                         if isinstance(metrics, dict) and 'accuracy' in metrics]
            model_accs.sort(key=lambda x: x[1], reverse=True)
            
            for model, acc in model_accs:
                metrics = results['model_results'][model]
                params = metrics.get('parameters', 'N/A')
                time_taken = metrics.get('training_time', 'N/A')
                report.append(f"  {model:12s}: {acc:6.2f}% ({params:,} params)")
    
    # Trend analysis
    report.append("\n📈 TREND ANALYSIS:")
    report.append("-" * 40)
    
    if len(sorted_lengths) > 1:
        # Track each model's performance across lengths
        model_trends = {}
        
        for seq_length in sorted_lengths:
            results = results_data[seq_length]
            if 'model_results' in results:
                for model, metrics in results['model_results'].items():
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        if model not in model_trends:
                            model_trends[model] = []
                        model_trends[model].append((seq_length, metrics['accuracy']))
        
        for model, trend_data in model_trends.items():
            if len(trend_data) > 1:
                trend_data.sort()
                first_acc = trend_data[0][1]
                last_acc = trend_data[-1][1]
                change = last_acc - first_acc
                change_pct = (change / first_acc) * 100 if first_acc > 0 else 0
                
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                report.append(f"  {model:12s}: {direction} {change:+5.2f}% ({change_pct:+5.1f}%)")
    
    # Key findings
    report.append("\n🔍 KEY FINDINGS:")
    report.append("-" * 40)
    
    if len(sorted_lengths) >= 2:
        latest_length = max(sorted_lengths)
        latest_results = results_data[latest_length]['model_results']
        
        # Find best model at longest sequence
        best_model = max(latest_results.items(), 
                        key=lambda x: x[1]['accuracy'] if isinstance(x[1], dict) and 'accuracy' in x[1] else 0)
        
        report.append(f"• Best model at {latest_length}s: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")
        
        # Check if sequence models are catching up to CNN
        if 'CNN' in latest_results and 'S4' in latest_results:
            cnn_acc = latest_results['CNN']['accuracy']
            s4_acc = latest_results['S4']['accuracy']
            gap = cnn_acc - s4_acc
            
            if gap < 5:
                report.append(f"• S4 is competitive with CNN (gap: {gap:.2f}%)")
            elif gap > 10:
                report.append(f"• CNN still significantly outperforms S4 (gap: {gap:.2f}%)")
            else:
                report.append(f"• Moderate gap between CNN and S4 (gap: {gap:.2f}%)")
    
    report_text = "\n".join(report)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f'/home/ergezerm/eeg_25/paper_figures/publication_summary_{timestamp}.txt'
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"📄 Summary report saved: {report_file}")
    
    return report_text

def main():
    print("📊 ICASSP Publication Results Analysis")
    print("=" * 50)
    
    # Load all results
    results_data = load_all_results()
    
    if not results_data:
        print("❌ No results found to analyze")
        return
    
    print(f"\n📁 Found results for sequence lengths: {sorted(results_data.keys())}")
    
    # Generate plots
    plot_file = create_publication_plots(results_data)
    
    # Generate summary
    summary = generate_summary_report(results_data)
    print("\n" + summary)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Figures saved to: /home/ergezerm/eeg_25/paper_figures/")

if __name__ == "__main__":
    main()
