#!/usr/bin/env python3
"""
ICASSP Results Analysis and Visualization
========================================
Comprehensive analysis script for ablation study results.
Generates confusion matrices, performance tables, and insights for the paper.

Author: Embedded AI Research Team
Date: July 7, 2025
Purpose: ICASSP 2025 - S4 Analysis and Paper Figures
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def convert_segment_length_to_numeric(segment_length):
    """Convert segment length from string format (e.g., '32s') to numeric"""
    if isinstance(segment_length, str):
        return int(segment_length.replace('s', ''))
    return segment_length

def analyze_ablation_results(results_dir="results", plots_dir="plots", paper_dir="paper_figures"):
    """Analyze ablation study results and generate paper-ready figures"""
    
    print("🔬 ICASSP Results Analysis Starting...")
    
    # Create paper figures directory
    os.makedirs(paper_dir, exist_ok=True)
    
    # Find latest ablation results
    csv_files = list(Path(results_dir).glob("*ablation*.csv"))
    json_files = list(Path(results_dir).glob("*ablation*.json"))
    
    if not csv_files:
        print("❌ No ablation results found")
        return
    
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Analyzing results from: {latest_csv}")
    
    # Load results
    df = pd.read_csv(latest_csv)
    print(f"   Found {len(df)} result entries")
    
    # Clean and process data
    df = clean_results_dataframe(df)
    
    # Generate analysis
    create_performance_summary_table(df, paper_dir)
    create_segment_length_scaling_plot(df, paper_dir)
    create_model_efficiency_analysis(df, paper_dir)
    create_s4_advantage_heatmap(df, paper_dir)
    
    # Load detailed results if available
    if json_files:
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        with open(latest_json, 'r') as f:
            detailed_results = json.load(f)
        
        create_training_dynamics_plots(detailed_results, paper_dir)
    
    # Generate summary report
    generate_summary_report(df, paper_dir)
    
    print("✅ Analysis complete! Check the paper_figures/ directory")

def clean_results_dataframe(df):
    """Clean and standardize the results dataframe"""
    
    # Standardize column names
    column_mapping = {
        'Segment_Length_s': 'Segment_Length',
        'Segment_Length': 'Segment_Length',
        'Accuracy_Pct': 'Accuracy',
        'Training_Time_s': 'Training_Time',
        'Inference_Time_ms': 'Inference_Time',
        'Peak_Memory_MB': 'Peak_Memory'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Convert string percentages to float
    if 'Accuracy' in df.columns:
        df['Accuracy'] = df['Accuracy'].astype(str).str.replace('%', '').astype(float)
    
    # Convert string numbers to float
    for col in ['F1_Score', 'Training_Time', 'Inference_Time', 'Peak_Memory']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert parameters (remove commas)
    if 'Parameters' in df.columns:
        df['Parameters'] = df['Parameters'].astype(str).str.replace(',', '').astype(float)
    
    return df

def create_performance_summary_table(df, save_dir):
    """Create a publication-ready performance summary table"""
    
    print("📋 Creating performance summary table...")
    
    # Group by segment length and model
    if 'Segment_Length' not in df.columns:
        print("❌ No segment length data found")
        return
    
    # Create pivot table
    pivot_acc = df.pivot_table(values='Accuracy', index='Model', columns='Segment_Length', fill_value=np.nan)
    pivot_f1 = df.pivot_table(values='F1_Score', index='Model', columns='Segment_Length', fill_value=np.nan)
    
    # Create summary table
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy heatmap
    sns.heatmap(pivot_acc, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=ax1, cbar_kws={'label': 'Accuracy (%)'})
    ax1.set_title('Model Accuracy vs Segment Length', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Segment Length (seconds)', fontsize=12)
    ax1.set_ylabel('Model', fontsize=12)
    
    # F1-Score heatmap
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', 
                ax=ax2, cbar_kws={'label': 'F1-Score'})
    ax2.set_title('Model F1-Score vs Segment Length', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Segment Length (seconds)', fontsize=12)
    ax2.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_summary_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical table
    summary_table = pd.concat([
        pivot_acc.add_suffix('_Acc'),
        pivot_f1.add_suffix('_F1')
    ], axis=1)
    
    summary_table.to_csv(f'{save_dir}/performance_summary_table.csv')
    print(f"   Saved to {save_dir}/performance_summary_heatmaps.png")

def create_segment_length_scaling_plot(df, save_dir):
    """Create the main scaling plot showing S4's advantage on longer sequences"""
    
    print("📈 Creating segment length scaling plot...")
    
    if 'Segment_Length' not in df.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Define colors for models
    model_colors = {
        'S4': '#FF6B6B',      # Red
        'CNN': '#4ECDC4',     # Teal  
        'LSTM': '#45B7D1',    # Blue
        'Transformer': '#96CEB4'  # Green
    }
    
    # Plot 1: Accuracy vs Segment Length
    ax1 = axes[0, 0]
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model].copy()
        model_data['Segment_Length_Numeric'] = model_data['Segment_Length'].apply(convert_segment_length_to_numeric)
        model_data = model_data.sort_values('Segment_Length_Numeric')
        if not model_data.empty:
            ax1.plot(model_data['Segment_Length'], model_data['Accuracy'], 
                    marker='o', linewidth=3, markersize=8, 
                    label=model, color=model_colors.get(model, 'gray'))
    
    ax1.set_xlabel('EEG Segment Length (seconds)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax1.set_title('Model Performance vs Sequence Length', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim(left=0)
    
    # Plot 2: Training Time vs Segment Length
    ax2 = axes[0, 1]
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model].copy()
        model_data['Segment_Length_Numeric'] = model_data['Segment_Length'].apply(convert_segment_length_to_numeric)
        model_data = model_data.sort_values('Segment_Length_Numeric')
        if not model_data.empty and 'Training_Time' in df.columns:
            ax2.plot(model_data['Segment_Length'], model_data['Training_Time'], 
                    marker='s', linewidth=3, markersize=8,
                    label=model, color=model_colors.get(model, 'gray'))
    
    ax2.set_xlabel('EEG Segment Length (seconds)', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Efficiency vs Sequence Length', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_yscale('log')
    
    # Plot 3: Memory Usage vs Segment Length
    ax3 = axes[1, 0]
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model].copy()
        model_data['Segment_Length_Numeric'] = model_data['Segment_Length'].apply(convert_segment_length_to_numeric)
        model_data = model_data.sort_values('Segment_Length_Numeric')
        if not model_data.empty and 'Peak_Memory' in df.columns:
            ax3.plot(model_data['Segment_Length'], model_data['Peak_Memory'], 
                    marker='^', linewidth=3, markersize=8,
                    label=model, color=model_colors.get(model, 'gray'))
    
    ax3.set_xlabel('EEG Segment Length (seconds)', fontsize=12)
    ax3.set_ylabel('Peak Memory Usage (MB)', fontsize=12)
    ax3.set_title('Memory Efficiency vs Sequence Length', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    # Plot 4: S4 Advantage (Accuracy difference)
    ax4 = axes[1, 1]
    segment_lengths = sorted(df['Segment_Length'].unique())
    s4_advantages = []
    
    for seg_len in segment_lengths:
        subset = df[df['Segment_Length'] == seg_len]
        s4_acc = subset[subset['Model'] == 'S4']['Accuracy']
        other_acc = subset[subset['Model'] != 'S4']['Accuracy']
        
        if not s4_acc.empty and not other_acc.empty:
            s4_best = s4_acc.iloc[0] if len(s4_acc) > 0 else 0
            other_best = other_acc.max() if len(other_acc) > 0 else 0
            s4_advantages.append(s4_best - other_best)
        else:
            s4_advantages.append(0)
    
    bars = ax4.bar(segment_lengths, s4_advantages, alpha=0.7, 
                   color=['green' if x > 0 else 'red' for x in s4_advantages])
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('EEG Segment Length (seconds)', fontsize=12)
    ax4.set_ylabel('S4 Accuracy Advantage (%)', fontsize=12)
    ax4.set_title('S4 Advantage Over Best Competitor', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, advantage in zip(bars, s4_advantages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + np.sign(height) * 0.5,
                f'{advantage:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/segment_length_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to {save_dir}/segment_length_scaling_analysis.png")

def create_model_efficiency_analysis(df, save_dir):
    """Create parameter efficiency and computational cost analysis"""
    
    print("⚡ Creating efficiency analysis...")
    
    if 'Parameters' not in df.columns:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parameter efficiency (Accuracy per million parameters)
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        if not model_data.empty:
            efficiency = model_data['Accuracy'] / (model_data['Parameters'] / 1e6)
            ax1.scatter(model_data['Segment_Length'], efficiency, 
                       s=100, alpha=0.7, label=model)
    
    ax1.set_xlabel('Segment Length (seconds)', fontsize=12)
    ax1.set_ylabel('Accuracy per Million Parameters', fontsize=12)
    ax1.set_title('Parameter Efficiency Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Computational cost vs accuracy
    if 'Training_Time' in df.columns:
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            if not model_data.empty:
                ax2.scatter(model_data['Training_Time'], model_data['Accuracy'], 
                           s=100, alpha=0.7, label=model)
        
        ax2.set_xlabel('Training Time (seconds)', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Computational Cost vs Performance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to {save_dir}/efficiency_analysis.png")

def create_s4_advantage_heatmap(df, save_dir):
    """Create a detailed heatmap showing where S4 excels"""
    
    print("🔥 Creating S4 advantage heatmap...")
    
    if len(df['Model'].unique()) < 2:
        return
    
    # Create advantage matrix
    segment_lengths = sorted(df['Segment_Length'].unique())
    metrics = ['Accuracy', 'F1_Score']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
            
        advantage_data = []
        
        for seg_len in segment_lengths:
            subset = df[df['Segment_Length'] == seg_len]
            s4_score = subset[subset['Model'] == 'S4'][metric]
            other_scores = subset[subset['Model'] != 'S4'][metric]
            
            if not s4_score.empty and not other_scores.empty:
                s4_val = s4_score.iloc[0]
                other_max = other_scores.max()
                advantage = s4_val - other_max
            else:
                advantage = 0
            
            advantage_data.append(advantage)
        
        # Create heatmap data
        heatmap_data = np.array(advantage_data).reshape(1, -1)
        
        # Plot heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=[str(x) for x in segment_lengths],
                   yticklabels=['S4 Advantage'],
                   annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=axes[i],
                   cbar_kws={'label': f'{metric} Difference'})
        
        axes[i].set_title(f'S4 {metric} Advantage vs Best Competitor', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('EEG Segment Length', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/s4_advantage_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved to {save_dir}/s4_advantage_heatmap.png")

def create_training_dynamics_plots(detailed_results, save_dir):
    """Create training dynamics plots if detailed data is available"""
    
    print("📊 Creating training dynamics plots...")
    
    # This would use the detailed JSON data to show learning curves, etc.
    # Implementation depends on the structure of detailed_results
    pass

def generate_summary_report(df, save_dir):
    """Generate a text summary report for the paper"""
    
    print("📄 Generating summary report...")
    
    report = []
    report.append("ICASSP 2025: S4 vs Traditional Models on EEG Data")
    report.append("=" * 60)
    report.append("")
    
    # Overall statistics
    n_models = df['Model'].nunique()
    n_segment_lengths = df['Segment_Length'].nunique() if 'Segment_Length' in df.columns else 1
    
    report.append(f"Experimental Setup:")
    report.append(f"  - Models tested: {n_models}")
    report.append(f"  - Segment lengths: {n_segment_lengths}")
    report.append(f"  - Total experiments: {len(df)}")
    report.append("")
    
    # Best performance per segment length
    if 'Segment_Length' in df.columns:
        report.append("Best Model per Segment Length:")
        for seg_len in sorted(df['Segment_Length'].unique()):
            subset = df[df['Segment_Length'] == seg_len]
            if not subset.empty:
                best_row = subset.loc[subset['Accuracy'].idxmax()]
                seg_len_clean = str(seg_len).replace('s', '')
                report.append(f"  {seg_len_clean}s: {best_row['Model']} ({best_row['Accuracy']:.1f}%)")
        report.append("")
    
    # S4 performance summary
    s4_data = df[df['Model'] == 'S4']
    if not s4_data.empty:
        report.append("S4 Performance Summary:")
        report.append(f"  - Average accuracy: {s4_data['Accuracy'].mean():.1f}%")
        report.append(f"  - Best accuracy: {s4_data['Accuracy'].max():.1f}%")
        if 'Segment_Length' in df.columns:
            best_seg = s4_data.loc[s4_data['Accuracy'].idxmax(), 'Segment_Length']
            best_seg_clean = str(best_seg).replace('s', '')
            report.append(f"  - Best performance at: {best_seg_clean}s")
        report.append("")
    
    # Key insights
    report.append("Key Insights:")
    
    if 'Segment_Length' in df.columns and len(df['Segment_Length'].unique()) > 1:
        # Analyze scaling trends
        s4_trend = analyze_scaling_trend(df, 'S4')
        cnn_trend = analyze_scaling_trend(df, 'CNN')
        
        if s4_trend == 'improving':
            report.append("  ✅ S4 shows improving performance with longer sequences")
        elif s4_trend == 'stable':
            report.append("  ➖ S4 shows stable performance across sequence lengths")
        else:
            report.append("  ❌ S4 performance decreases with longer sequences")
        
        if cnn_trend == 'declining':
            report.append("  📉 CNN performance declines with longer sequences")
        
        # Long sequence advantage
        if 'Segment_Length' in df.columns:
            # Convert segment lengths to numeric for comparison
            df['Segment_Length_Numeric'] = df['Segment_Length'].apply(convert_segment_length_to_numeric)
            long_seqs = df[df['Segment_Length_Numeric'] >= 32]
        else:
            long_seqs = df
        if not long_seqs.empty:
            s4_long = long_seqs[long_seqs['Model'] == 'S4']['Accuracy']
            others_long = long_seqs[long_seqs['Model'] != 'S4']['Accuracy']
            
            if not s4_long.empty and not others_long.empty:
                if s4_long.mean() > others_long.mean():
                    report.append(f"  🎯 S4 excels on long sequences: {s4_long.mean():.1f}% vs {others_long.mean():.1f}%")
    
    report.append("")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save report
    with open(f'{save_dir}/summary_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print(f"   Saved to {save_dir}/summary_report.txt")
    
    # Print key findings
    print("\n🎯 KEY FINDINGS:")
    for line in report:
        if line.strip().startswith(('✅', '📉', '🎯', '❌', '➖')):
            print(f"   {line.strip()}")

def analyze_scaling_trend(df, model):
    """Analyze how a model's performance scales with sequence length"""
    
    model_data = df[df['Model'] == model].copy()
    if len(model_data) < 2:
        return 'insufficient_data'
    
    # Convert segment lengths to numeric for proper sorting
    model_data['Segment_Length_Numeric'] = model_data['Segment_Length'].apply(convert_segment_length_to_numeric)
    model_data = model_data.sort_values('Segment_Length_Numeric')
    
    # Simple trend analysis
    accuracies = model_data['Accuracy'].values
    
    # Calculate trend
    if accuracies[-1] > accuracies[0] + 2:  # Improving by >2%
        return 'improving'
    elif accuracies[-1] < accuracies[0] - 2:  # Declining by >2%
        return 'declining'
    else:
        return 'stable'

if __name__ == "__main__":
    print("🚀 Starting ICASSP Results Analysis...")
    try:
        analyze_ablation_results()
        print("✅ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
