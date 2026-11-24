# Transfer Learning vs. Training from Scratch under Limited Labeled Data

**DSAA 5020 Final Project (Topic 8)** — An empirical study on the 37-class **Oxford-IIIT Pet** fine-grained classification benchmark comparing random initialization (*Scratch*), frozen feature extraction (*Frozen*), partial fine-tuning (*Partial FT*), and full fine-tuning (*Full FT*) across **44 controlled NVIDIA L4 GPU runs** (`ResNet-18` primary + `MobileNetV3-Small` cross-backbone validation).

- **Final 6-Page ACM Report (PDF)**: [`report/main.pdf`](report/main.pdf)
- **LaTeX Source**: [`report/main.tex`](report/main.tex) | [`report/references.bib`](report/references.bib)
- **Standalone Colab Notebook (with saved outputs)**: [`colab_standalone_topic8.ipynb`](colab_standalone_topic8.ipynb)
- **Full Aggregated Results**: [`results/tables/summary_table.md`](results/tables/summary_table.md) | [`results/tables/all_runs.csv`](results/tables/all_runs.csv)

---

## Key Results (ResNet-18, 3 Seeds Mean ± Std, NVIDIA L4 GPU)

| Data % | Strategy | Trainable Params | Test Accuracy | Test Macro-F1 | Train–Val Gap | Total Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **10%** | From Scratch | 11.20M | 14.09 ± 1.09% | 13.74 ± 1.01% | +82.50% | 139.0 ± 3.0s |
| **10%** | **Frozen Feature Extraction** | **19.0K** | **83.65 ± 0.39%** | **83.37 ± 0.45%** | **+12.68%** | **51.0 ± 17.6s** |
| **10%** | Partial FT (`layer4` + `fc`) | 8.41M | 77.09 ± 0.73% | 76.65 ± 0.91% | +18.84% | 33.8 ± 3.0s |
| **10%** | Full Fine-Tuning | 11.20M | 77.81 ± 0.50% | 77.36 ± 0.65% | +18.37% | 46.1 ± 10.9s |
| **100%** | From Scratch | 11.20M | 55.95 ± 1.33% | 55.39 ± 1.48% | +37.06% | 1250.1 ± 8.0s |
| **100%** | Frozen Feature Extraction | 19.0K | 88.70 ± 0.18% | 88.58 ± 0.16% | +6.00% | 524.2 ± 105.2s |
| **100%** | Partial FT (`layer4` + `fc`) | 8.41M | 88.73 ± 0.23% | 88.57 ± 0.20% | +7.71% | 403.0 ± 48.4s |
| **100%** | **Full Fine-Tuning** | **11.20M** | **89.54 ± 0.30%** | **89.41 ± 0.31%** | +7.02% | 524.8 ± 134.2s |

### Main Findings
1. **Massive Low-Data Advantage**: On 10% labeled data (~8 imgs/class), *Frozen* updates only **0.17%** of parameters (19K) and reaches **83.65%** accuracy (+69.6 pp over *Scratch*, 2.7× faster).
2. **Low-Data Crossover**: Across 5%–25% data, *Frozen* beats *Full FT* by **+2.4 to +12.7 pp** by avoiding small-sample feature distortion; a crossover occurs at **50% data**, with *Full FT* peaking at 100% (**89.54%**).
3. **20× Label & 38× Compute Efficiency**: Even **5% *Frozen*** (81.55%, 32.6s) outperforms **100% *Scratch*** (55.95%, 1250s) by +25.6 pp.
4. **Source-Domain Asymmetry (Dog vs. Cat)**: At 100% data, dog breeds (heavily covered in ImageNet-1K) saturate under *Frozen* (91.36% vs. 91.27% *Full FT*), whereas virtually 100% of *Full FT*'s overall gain comes from under-represented cat breeds (+2.74 pp).

![Data Efficiency Curve](results/figures/fig3_data_efficiency_curve.png)
![Performance vs Compute Pareto Trade-off](results/figures/fig6_efficiency_tradeoff.png)

---

## Repository Structure

```text
.
├── src/                               # Modular PyTorch codebase
│   ├── dataset.py                     # Oxford-IIIT Pet download, leak-free stratified nested splits, EDA
│   ├── models.py                      # ResNet-18 / MobileNetV3-Small / EfficientNet-B0 + BN-safe freezing
│   ├── train.py                       # AMP training loop, synchronized GPU timing, embedding extraction
│   ├── run_experiments.py             # CLI orchestrator (core / extended / full_with_mobilenet)
│   ├── visualize.py                   # Summary tables (CSV/MD) & 8 publication-grade 300 DPI figures
│   └── utils.py                       # Seeding, hardware metadata, I/O helpers
├── results/                           # All 44 runs' logs, metrics, predictions, embeddings, tables & figures
│   ├── eda_summary.json
│   ├── tables/                        # all_runs.csv, summary_by_condition.csv, summary_table.md
│   ├── figures/                       # fig1 – fig8 (300 DPI PNGs)
│   ├── resnet18/                      # 36 run directories (5%–100%, multi-seed)
│   └── mobilenet_v3_small/            # 8 cross-backbone validation run directories
├── report/                            # Self-contained 6-page ACM SIGCONF LaTeX report
│   ├── main.pdf                       # Compiled 6-page final report
│   ├── main.tex                       # LaTeX source
│   ├── references.bib                 # 12 BibTeX references
│   └── figures/                       # High-res figures for LaTeX compilation
├── colab_standalone_topic8.ipynb      # Self-contained single-file Colab notebook (with full L4 outputs)
├── colab_transfer_learning_pet.ipynb  # Drive-backed modular Colab notebook
└── requirements.txt
```

---

## Reproducing the Experiments

### Option 1: Google Colab (One-Click)
Upload [`colab_standalone_topic8.ipynb`](colab_standalone_topic8.ipynb) to Google Colab, switch **Runtime → Change runtime type → T4 / L4 / A100 GPU**, and click **Run all**.

### Option 2: Command Line (Local / GPU Server)
```bash
pip install -r requirements.txt

# Run all 44 experiments (Core + Extended 2x4 + 5%-100% curve + MobileNetV3-Small)
python -m src.run_experiments --mode full_with_mobilenet --data_dir data --output_dir results

# Regenerate summary tables and all 8 figures
python -m src.visualize --output_dir results --data_dir data
```

### Recompiling the LaTeX Report
```bash
cd report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
