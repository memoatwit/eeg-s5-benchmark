# EEG S5 Benchmark

**Temporal Context and Architecture: A Benchmark for Naturalistic EEG Decoding**
*Mehmet Ergezer, Wentworth Institute of Technology, ICASSP 2026*

## Overview

This repository contains the code, results, and paper for a systematic benchmark comparing five neural architectures for naturalistic EEG decoding using the [HBN movie-watching dataset](https://github.com/shirazi2024hbn):

| Model | Description |
|-------|-------------|
| CNN | Local convolutional baseline |
| LSTM | Recurrent baseline |
| EEGXF | Stabilized EEG Transformer (introduced here) |
| S4 | Structured State Space model |
| **S5** | Simplified S4: best parameter efficiency |

Models are evaluated across segment lengths from **8 s to 128 s** on a 4-class movie classification task, plus three generalization tests: zero-shot cross-task, cross-frequency, and leave-one-subject-out (LOSO).

### Key Result
At 64 s, **S5 reaches 98.7% ± 0.6** accuracy using ~20× fewer parameters than CNN. EEGXF offers greater robustness under distribution shift.

## Repository Structure

```
eeg-s5-benchmark/
├── train.py                     # Main training script (all 5 architectures)
├── models/
│   ├── __init__.py              # Unified import: from models import S5Classifier
│   ├── cnn.py                   # CNNClassifier
│   ├── lstm.py                  # LSTMClassifier
│   ├── transformer.py           # TransformerClassifier (standard)
│   ├── eeg_transformer.py       # EEGXF — stabilized EEG Transformer (introduced here)
│   ├── s4.py                    # OptimizedS4Classifier
│   └── s5.py                    # S5Classifier
├── ablation/
│   └── run_ablation.py          # Multi-GPU ablation study (3 seeds)
├── analysis/
│   ├── generate_figures.py      # Paper figures (Fig 1–3)
│   ├── create_efficiency_plots.py
│   ├── analyze_results.py
│   ├── analyze_publication_results.py
│   └── statistical_significance_analysis.py
└── results/                     # Experiment result JSONs
```

## Usage

### Training

```bash
python train.py --model s5 --segment_length 64 --dataset hbn
```

See `train.py` for the full list of arguments (model, segment length, dataset path, GPU settings).

### Running the Ablation Study

```bash
python ablation/run_ablation.py
```

Runs all model × segment-length combinations across 3 seeds on available GPUs.

### Generating Paper Figures

```bash
python analysis/generate_figures.py --results_dir results/
```

## Results

Pre-computed result JSONs are in `results/`. Each file is named `{Model}_{SegmentLength}s_gpu{N}_exp{N}.json`.

## Citation

```bibtex
@inproceedings{ergezer2026eegbenchmark,
  title     = {Temporal Context and Architecture: A Benchmark for Naturalistic {EEG} Decoding},
  author    = {Ergezer, Mehmet},
  booktitle = {Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```

## License

MIT
