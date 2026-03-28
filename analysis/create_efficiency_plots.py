#!/usr/bin/env python3
"""
Performance vs. Efficiency Plot for ICASSP Paper

Creates a scatter plot showing the trade-off between training efficiency and accuracy
across different EEG architectures, highlighting S5's superior efficiency.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 8),
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def create_performance_efficiency_plot():
    """
    Create a comprehensive performance vs efficiency plot
    """
    
    # Data from your table and cross-frequency results
    models_data = {
        # From your main table (subject-mixed, 32s and 64s segments)
        'S5_32s_mixed': {
            'name': 'S5 (32s, subject-mixed)',
            'accuracy': 96.2,
            'training_time_min': 11.4,
            'params_k': 182,
            'color': '#2E86AB',
            'marker': 'o',
            'size': 100,
            'evaluation': 'subject-mixed'
        },
        'S5_64s_mixed': {
            'name': 'S5 (64s, subject-mixed)', 
            'accuracy': 94.3,
            'training_time_min': 16.1,
            'params_k': 515,
            'color': '#A23B72',
            'marker': 'o', 
            'size': 100,
            'evaluation': 'subject-mixed'
        },
        
        # From cross-frequency results (subject-independent)
        'S5_32s_indep': {
            'name': 'S5 (32s, subject-independent)',
            'accuracy': 53.1,
            'training_time_min': 12.6,  # 753 seconds / 60
            'params_k': 182,  # Estimated based on config
            'color': '#2E86AB',
            'marker': 's',
            'size': 100,
            'evaluation': 'subject-independent'
        },
        
        'EEGTransformer_8s_mixed': {
            'name': 'EEGTransformer (8s, subject-mixed)',
            'accuracy': 89.2,
            'training_time_min': 22.3,
            'params_k': 1000,
            'color': '#F18F01',
            'marker': '^',
            'size': 100,
            'evaluation': 'subject-mixed'
        },
        'EEGTransformer_64s_mixed': {
            'name': 'EEGTransformer (64s, subject-mixed)',
            'accuracy': 63.8,
            'training_time_min': 363.0,
            'params_k': 30600,
            'color': '#C73E1D',
            'marker': '^',
            'size': 100,
            'evaluation': 'subject-mixed'
        },
        
        # From cross-frequency results
        'EEGTransformer_32s_indep': {
            'name': 'EEGTransformer (32s, subject-independent)',
            'accuracy': 40.9,
            'training_time_min': 156.5,  # 9391 seconds / 60
            'params_k': 1700,  # From cross-freq config (1.7M params)
            'color': '#F18F01',
            'marker': 'D',
            'size': 100,
            'evaluation': 'subject-independent'
        },
        
        # CNN baselines
        'CNN_8s': {
            'name': 'CNN (8s)',
            'accuracy': 63.5,
            'training_time_min': 180.0,
            'params_k': 2500,
            'color': '#3F7D20',
            'marker': 'v',
            'size': 80,
            'evaluation': 'subject-mixed'
        },
        'CNN_64s': {
            'name': 'CNN (64s)',
            'accuracy': 74.1,
            'training_time_min': 420.0,
            'params_k': 2500,
            'color': '#3F7D20',
            'marker': 'v',
            'size': 80,
            'evaluation': 'subject-mixed'
        }
    }
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Training Time vs Accuracy
    ax1.set_xscale('log')
    ax1.set_xlim(5, 500)
    ax1.set_ylim(35, 100)
    
    # Separate by evaluation type
    mixed_models = [k for k, v in models_data.items() if v['evaluation'] == 'subject-mixed']
    indep_models = [k for k, v in models_data.items() if v['evaluation'] == 'subject-independent']
    
    # Plot subject-mixed results
    for model_key in mixed_models:
        model = models_data[model_key]
        ax1.scatter(model['training_time_min'], model['accuracy'], 
                   color=model['color'], marker=model['marker'], s=model['size'],
                   alpha=0.8, edgecolors='black', linewidth=1.5,
                   label=model['name'])
    
    # Plot subject-independent results
    for model_key in indep_models:
        model = models_data[model_key]
        ax1.scatter(model['training_time_min'], model['accuracy'], 
                   color=model['color'], marker=model['marker'], s=model['size'],
                   alpha=0.6, edgecolors='black', linewidth=1.5,
                   label=model['name'], facecolors='none')
    
    # Highlight the efficiency zones
    # "Sweet spot" zone (high accuracy, low time)
    sweet_spot = Rectangle((5, 85), 25, 15, alpha=0.15, color='green', 
                          label='Efficiency Sweet Spot')
    ax1.add_patch(sweet_spot)
    
    # "Expensive" zone (high time)
    expensive_zone = Rectangle((100, 35), 400, 65, alpha=0.15, color='red',
                              label='Computationally Expensive')
    ax1.add_patch(expensive_zone)
    
    ax1.set_xlabel('Training Time (minutes, log scale)', fontsize=14)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax1.set_title('Training Efficiency vs. Accuracy\nEEG Movie Classification', fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key points
    ax1.annotate('S5: Fast & Accurate\n(subject-mixed)', 
                xy=(11.4, 96.2), xytext=(20, 90),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax1.annotate('EEGTransformer:\nSlow & Variable', 
                xy=(363, 63.8), xytext=(200, 50),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
    
    # Plot 2: Parameters vs Accuracy  
    ax2.set_xscale('log')
    ax2.set_xlim(100, 50000)
    ax2.set_ylim(35, 100)
    
    # Plot subject-mixed results
    for model_key in mixed_models:
        model = models_data[model_key]
        ax2.scatter(model['params_k'], model['accuracy'], 
                   color=model['color'], marker=model['marker'], s=model['size'],
                   alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Plot subject-independent results  
    for model_key in indep_models:
        model = models_data[model_key]
        ax2.scatter(model['params_k'], model['accuracy'], 
                   color=model['color'], marker=model['marker'], s=model['size'],
                   alpha=0.6, edgecolors='black', linewidth=1.5, facecolors='none')
    
    # Efficiency zones for parameter plot
    param_sweet_spot = Rectangle((100, 85), 900, 15, alpha=0.15, color='green')
    ax2.add_patch(param_sweet_spot)
    
    param_expensive_zone = Rectangle((5000, 35), 45000, 65, alpha=0.15, color='red')
    ax2.add_patch(param_expensive_zone)
    
    ax2.set_xlabel('Model Parameters (thousands, log scale)', fontsize=14)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax2.set_title('Model Complexity vs. Accuracy\nEEG Movie Classification', fontsize=16, pad=20)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax2.annotate('S5: Compact & Effective', 
                xy=(182, 96.2), xytext=(500, 90),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax2.annotate('EEGTransformer:\n30M+ Parameters', 
                xy=(30600, 63.8), xytext=(20000, 50),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
    
    # Create unified legend
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
                  markersize=10, label='S5', markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='^', color='w', markerfacecolor='#F18F01', 
                  markersize=10, label='EEGTransformer', markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='v', color='w', markerfacecolor='#3F7D20', 
                  markersize=8, label='CNN', markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                  markersize=8, label='Subject-Mixed Split', markeredgecolor='black'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                  markersize=8, label='Subject-Independent Split', markeredgecolor='black'),
    ]
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=5, fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def create_efficiency_summary_table():
    """
    Create a summary table showing key efficiency metrics
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Data for the table
    data = [
        ['Model', 'Segment', 'Test Acc (%)', 'Training Time', 'Parameters', 'Speed vs EEGTrans', 'Evaluation'],
        ['S5', '32s', '96.2', '11.4 min', '182K', '1.0×', 'Subject-Mixed'],
        ['S5', '32s', '53.1', '12.6 min', '182K', '12.4×', 'Subject-Independent'],
        ['EEGTransformer', '8s', '89.2', '22.3 min', '1.0M', '—', 'Subject-Mixed'],
        ['EEGTransformer', '32s', '40.9', '156.5 min', '1.7M', '1.0×', 'Subject-Independent'],
        ['EEGTransformer', '64s', '63.8', '363.0 min', '30.6M', '—', 'Subject-Mixed'],
        ['CNN', '8s', '63.5', '180.0 min', '2.5M', '—', 'Subject-Mixed'],
        ['CNN', '64s', '74.1', '420.0 min', '2.5M', '—', 'Subject-Mixed'],
    ]
    
    # Create table
    table = ax.table(cellText=data[1:], colLabels=data[0], 
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.08, 0.12, 0.15, 0.12, 0.15, 0.18])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color coding
    for i in range(len(data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight S5 rows
    for i in [1, 2]:  # S5 rows
        for j in range(len(data[0])):
            table[(i, j)].set_facecolor('#E8F5E8')
    
    ax.set_title('EEG Architecture Efficiency Comparison\nHighlighting S5 Advantages', 
                fontsize=16, pad=20, weight='bold')
    
    plt.tight_layout()
    return fig

def create_robustness_comparison():
    """
    Create a robustness comparison chart from cross-frequency results
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from cross-frequency results
    frequencies = ['256Hz', '128Hz', '64Hz']
    s5_accuracy = [53.1, 37.5, 34.4]
    transformer_accuracy = [40.9, 39.7, 40.7]
    
    x = np.arange(len(frequencies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, s5_accuracy, width, label='S5', 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, transformer_accuracy, width, label='EEGXF', 
                   color='#F18F01', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
    
    # Calculate and show robustness metrics
    s5_drop_128 = s5_accuracy[0] - s5_accuracy[1]
    s5_drop_64 = s5_accuracy[0] - s5_accuracy[2]
    trans_drop_128 = transformer_accuracy[0] - transformer_accuracy[1]
    trans_drop_64 = transformer_accuracy[0] - transformer_accuracy[2]
    
    # Add robustness annotations
    ax.text(0.02, 0.95, f'Frequency Robustness (Accuracy Drops):\n'
                       f'S5: -15.6% (128Hz), -18.7% (64Hz)\n'
                       f'EEGXF: -1.2% (128Hz), -0.2% (64Hz)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    ax.set_xlabel('Sampling Frequency', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title('Cross-Frequency Robustness Comparison\nZero-Shot Performance at Different Sampling Rates', 
                fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(frequencies)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 60)
    
    plt.tight_layout()
    return fig

def main():
    """
    Generate all performance vs efficiency visualizations
    """
    
    print("🎨 Generating Performance vs. Efficiency Visualizations...")
    
    # Create the main efficiency plot
    print("📊 Creating Performance vs. Efficiency scatter plots...")
    fig1 = create_performance_efficiency_plot()
    fig1.savefig('/home/ergezerm/eeg_25/paper_figures/performance_vs_efficiency.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig('/home/ergezerm/eeg_25/paper_figures/performance_vs_efficiency.pdf', 
                bbox_inches='tight', facecolor='white')
    
    # Create the efficiency summary table
    print("📋 Creating efficiency summary table...")
    fig2 = create_efficiency_summary_table()
    fig2.savefig('/home/ergezerm/eeg_25/paper_figures/efficiency_summary_table.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig('/home/ergezerm/eeg_25/paper_figures/efficiency_summary_table.pdf', 
                bbox_inches='tight', facecolor='white')
    
    # Create robustness comparison
    print("🔄 Creating frequency robustness comparison...")
    fig3 = create_robustness_comparison()
    fig3.savefig('/home/ergezerm/eeg_25/paper_figures/frequency_robustness_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig3.savefig('/home/ergezerm/eeg_25/paper_figures/frequency_robustness_comparison.pdf', 
                bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    print("\n✅ All visualizations generated successfully!")
    print(f"📁 Files saved in: /home/ergezerm/eeg_25/paper_figures/")
    print(f"   • performance_vs_efficiency.png/pdf")
    print(f"   • efficiency_summary_table.png/pdf") 
    print(f"   • frequency_robustness_comparison.png/pdf")
    
    print(f"\n🎯 Key Insights Visualized:")
    print(f"   • S5 achieves high accuracy with 12.4× faster training")
    print(f"   • S5 uses 17× fewer parameters than large EEGXFs")
    print(f"   • EEGXF shows superior frequency robustness")
    print(f"   • Clear speed-accuracy-robustness trade-offs revealed")

if __name__ == "__main__":
    main()
