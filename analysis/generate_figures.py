# generate_paper_figures.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
import glob

def parse_results(results_dir: Path) -> pd.DataFrame:
    """Parses all JSON result files into a single pandas DataFrame."""
    all_results = []

    # Find all JSON files in the directory
    json_files = glob.glob(str(results_dir / '*.json'))
    print(f"Found {len(json_files)} result files.")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Find the best EEGXF config for each segment length
            if "UltraSimplifiedEEGTransformer" in data['experiment_info']['model']:
                model_name = 'EEGXF'
            else:
                model_name = data['experiment_info']['model']

            result = {
                'model': model_name,
                'segment_length': data['experiment_info']['segment_length'],
                'test_accuracy': data['performance_metrics']['test_accuracy'] * 100, # Convert to %
                'training_time': data['computational_metrics']['training_time'] / 60, # Convert to minutes
                'parameters': data['model_info']['n_parameters'] / 1e6, # Convert to millions
            }
            all_results.append(result)
        except Exception as e:
            print(f"Could not parse {file_path}: {e}")

    return pd.DataFrame(all_results)

def get_best_eegxf(df: pd.DataFrame) -> pd.DataFrame:
    """Filters for the best performing EEGXF config at each segment length."""
    eegxf_df = df[df['model'] == 'EEGXF'].copy()
    # Find the index of the max accuracy for each segment length
    best_idx = eegxf_df.groupby('segment_length')['test_accuracy'].idxmax()
    return eegxf_df.loc[best_idx]

def create_final_plots(df: pd.DataFrame, save_dir: Path):
    """Generates and saves the final, publication-quality figures for the ICASSP paper."""
    
    # --- Prepare Data ---
    best_eegxf_df = get_best_eegxf(df)
    other_models_df = df[df['model'].isin(['S5', 'CNN', 'LSTM', 'S4'])]
    final_df = pd.concat([other_models_df, best_eegxf_df]).sort_values(by='segment_length')
    
    # --- Plotting Setup for IEEE ---
    # Enhanced color palette - our models get bold colors, baselines get muted colors
    ieee_colors = ['#003f7f', '#d62728', '#2ca02c', '#888888', '#cccccc']  # Bold blue/red for ours, muted for others
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })

    model_styles = {
        'S5': {'marker': 'o', 'linestyle': '-', 'label': 'S5 (Ours)', 'linewidth': 2.5, 'markersize': 7, 'alpha': 1.0},
        'EEGXF': {'marker': 's', 'linestyle': '--', 'label': 'EEGXF (Ours)', 'linewidth': 2.5, 'markersize': 7, 'alpha': 1.0},
        'CNN': {'marker': '^', 'linestyle': ':', 'label': 'CNN', 'linewidth': 1.5, 'markersize': 5, 'alpha': 0.7},
        'S4': {'marker': 'D', 'linestyle': '-.', 'label': 'S4', 'linewidth': 1.5, 'markersize': 5, 'alpha': 0.7},
        'LSTM': {'marker': 'P', 'linestyle': ':', 'label': 'LSTM', 'linewidth': 1.5, 'markersize': 5, 'alpha': 0.7}
    }
    model_order = list(model_styles.keys())

    # --- Figure 1: Accuracy vs. Segment Length ---
    fig1, ax1 = plt.subplots(figsize=(4, 2.5))
    for model_name, group in final_df.groupby('model'):
        model_name_str = str(model_name)
        style = model_styles.get(model_name_str)
        if style:
            color = ieee_colors[model_order.index(model_name_str)]
            ax1.plot(group['segment_length'], group['test_accuracy'], 
                     marker=style['marker'], linestyle=style['linestyle'], 
                     color=color, label=style['label'],
                     linewidth=style['linewidth'], markersize=style['markersize'],
                     alpha=style['alpha'], markeredgecolor='white', markeredgewidth=0.5)

    ax1.set_xlabel("Segment Length (seconds)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([8, 16, 32, 64, 128])
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    
    # Add performance annotations
    s5_data = final_df[final_df['model'] == 'S5']
    if not s5_data.empty:
        best_s5 = s5_data.loc[s5_data['test_accuracy'].idxmax()]
        ax1.annotate('Best efficiency', 
                    xy=(best_s5['segment_length'], best_s5['test_accuracy']),
                    xytext=(best_s5['segment_length']*0.6, best_s5['test_accuracy']+1),
                    arrowprops=dict(arrowstyle='->', color='#003f7f', alpha=0.7),
                    fontsize=8, color='#003f7f', weight='bold')
    
    ax1.legend(loc='lower right')
    
    fig1.tight_layout(pad=0.1)
    fig1_path = save_dir / "fig1_accuracy_vs_segment_length.pdf"
    fig1.savefig(fig1_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready Figure 1 saved to: {fig1_path}")

    # --- Figure 2: Performance vs. Efficiency (at 64s) ---
    df_64s = final_df[final_df['segment_length'] == 64].copy()
    df_64s['model'] = pd.Categorical(df_64s['model'], categories=model_order, ordered=True)
    df_64s = df_64s.sort_values('model')

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True)

    # Left plot: Training Time vs. Accuracy
    sns.scatterplot(data=df_64s, x='training_time', y='test_accuracy', hue='model', style='model', 
                    palette=ieee_colors, markers=[model_styles[m]['marker'] for m in model_order],
                    s=60, ax=ax2a, legend=False, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax2a.set_xlabel("Training Time (min, log scale)")
    ax2a.set_ylabel("Test Accuracy (%)")
    ax2a.set_xscale('log')
    
    # Right plot: Parameters vs. Accuracy
    sns.scatterplot(data=df_64s, x='parameters', y='test_accuracy', hue='model', style='model', 
                    palette=ieee_colors, markers=[model_styles[m]['marker'] for m in model_order],
                    s=60, ax=ax2b, legend=True, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax2b.set_xlabel("Parameters (M, log scale)")
    ax2b.set_xscale('log')
    
    handles, labels = ax2b.get_legend_handles_labels()
    ax2b.get_legend().remove()
    fig2.legend(handles, [model_styles[lbl]['label'] for lbl in labels], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)

    fig2.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig2_path = save_dir / "fig2_performance_vs_efficiency.pdf"
    fig2.savefig(fig2_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready Figure 2 saved to: {fig2_path}")

    # --- Figure 3: Accuracy vs. Parameters vs. Training Time (at 64s) ---
    fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
    
    # Use a size norm for better visual scaling of training time
    norm = mcolors.Normalize(df_64s['training_time'].min(), df_64s['training_time'].max())
    
    # Create the scatter plot without automatic legend
    sns.scatterplot(
        data=df_64s,
        x='parameters',
        y='test_accuracy',
        hue='model',
        style='model',
        size='training_time',
        sizes=(40, 400),  # Min and max marker sizes
        size_norm=norm,
        palette=ieee_colors,
        markers=[model_styles[m]['marker'] for m in model_order],
        ax=ax3,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.85,
        legend=False  # Turn off automatic legend
    )
    
    ax3.set_xlabel("Model Parameters (M, log scale)")
    ax3.set_ylabel("Test Accuracy (%)")
    ax3.set_xscale('log')
    
    # Add efficiency zone annotation
    s5_point = df_64s[df_64s['model'] == 'S5'].iloc[0]
    ax3.annotate('Efficiency\nSweet Spot', 
                xy=(s5_point['parameters'], s5_point['test_accuracy']),
                xytext=(s5_point['parameters']*2.5, s5_point['test_accuracy']-2.5),
                arrowprops=dict(arrowstyle='->', color='#003f7f', alpha=0.8, lw=1.5),
                fontsize=8, color='#003f7f', weight='bold', ha='center')
    
    ax3.set_xlabel("Model Parameters (M, log scale)")
    ax3.set_ylabel("Test Accuracy (%)")
    ax3.set_xscale('log')
    
    # Create custom legends
    # 1. Model Legend
    model_handles = []
    for i, (model, style) in enumerate(model_styles.items()):
        handle = mlines.Line2D([0], [0], marker=style['marker'], color=ieee_colors[i], 
                           linestyle='None', markersize=6, markeredgecolor='black')
        model_handles.append(handle)
    
    model_labels = [style['label'] for style in model_styles.values()]
    legend1 = ax3.legend(model_handles, model_labels, title='Model', loc='lower right', 
                        frameon=True, fontsize=8, title_fontsize=8)
    ax3.add_artist(legend1)
    
    # 2. Training Time Legend (Manual with properly scaled, readable markers)
    time_values = sorted(df_64s['training_time'].unique())
    # Select 3-4 representative values to avoid overcrowding
    if len(time_values) > 4:
        indices = [0, len(time_values)//3, 2*len(time_values)//3, -1]
        time_values = [time_values[i] for i in indices]
    
    time_handles = []
    time_labels = []
    # Create scaled markers for legend (much smaller than plot markers but still proportional)
    min_legend_size, max_legend_size = 15, 60  # Reasonable legend marker range
    min_time, max_time = min(time_values), max(time_values)
    
    for t in time_values:
        # Scale marker size proportionally but within legend-friendly range
        if max_time > min_time:
            scaled_size = min_legend_size + (max_legend_size - min_legend_size) * (t - min_time) / (max_time - min_time)
        else:
            scaled_size = min_legend_size
        
        handle = plt.scatter([], [], s=scaled_size, color='gray', edgecolor='black', alpha=0.7)
        time_handles.append(handle)
        time_labels.append(f"{t:.0f}")
    
    ax3.legend(time_handles, time_labels, title='Training Time (min)', 
                        loc='lower left', frameon=True, fontsize=8, title_fontsize=8,
                        scatterpoints=1, markerscale=1, labelspacing=1.2)
    
    fig3.tight_layout(pad=0.1)
    fig3_path = save_dir / "fig3_performance_tradeoff.pdf"
    fig3.savefig(fig3_path, bbox_inches='tight', dpi=300)
    print(f"✅ Publication-ready Figure 3 saved to: {fig3_path}")


if __name__ == "__main__":
    results_folder = Path('/Users/memo/Downloads/eeg_results_aug07') # Assumes JSON files are in a 'results' subfolder
    if not results_folder.exists():
        print(f"Error: The '{results_folder}' directory was not found.")
        print("Please place all your .json result files into a folder named 'results'.")
    else:
        # 1. Parse all data
        full_df = parse_results(results_folder)
        
        # 2. Generate and save the final plots
        create_final_plots(full_df, Path('.')) # Save plots in the current directory