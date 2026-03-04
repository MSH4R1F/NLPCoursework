# Patronizing and Condescending Language Detection

Binary classification of news paragraphs as patronizing/condescending (PCL) or not, based on [SemEval 2022 Task 4](https://sites.google.com/view/pcl-detection-semeval2022) and the "Don't Patronize Me!" dataset.

## Approach

A **DeBERTa-v3-base** model fine-tuned with multi-task learning:

- **Primary task:** Binary PCL classification (PCL vs. No PCL)
- **Auxiliary task:** 7-category PCL subtype multi-label classification (unbalanced power relations, shallow solution, presupposition, authority voice, metaphor, compassion, "the poorer the merrier")

The dataset has a 9.6:1 class imbalance (No PCL : PCL), handled via class-weighted cross-entropy loss, 2.5x minority oversampling, and threshold tuning on the validation set.

**Dev set F1 (positive class): 0.5202** (vs. 0.48 baseline)

**Python Notebook model.ipynb actually works you might have to download it though** 
## Repository Structure

```
├── model.ipynb                       # Full pipeline: EDA, training, evaluation, prediction
├── train_semeval_parids-labels.csv   # Train split paragraph IDs and subcategory labels
├── dev_semeval_parids-labels.csv     # Dev split paragraph IDs and subcategory labels
├── dev.txt                           # Dev set predictions
├── test.txt                          # Test set predictions
├── scripts/
│   ├── error_analysis.py             # Confusion matrix, per-subcategory recall, FN/FP examples
│   └── error_analysis_writeup.txt    # Written analysis of error patterns
├── outputs/
│   ├── eda1_class_distribution.png   # Train set class distribution plot
│   ├── eda2_text_distribution.png    # Text length distribution plot
│   ├── error_cm.png                  # Confusion matrix heatmap
│   └── error_recall_by_category.png  # Recall breakdown by PCL subcategory
└── BestModel/
    ├── model.txt                     # Google Drive link to the trained model weights
    ├── tokenizer_config.json         # Tokenizer configuration
    └── tokenizer.json                # Tokenizer vocabulary
```

## Setup

### Requirements

- Python 3.8+
- GPU recommended (trained on A100)

### Dependencies

```bash
pip install torch transformers datasets accelerate scikit-learn pandas matplotlib scipy
```

### External Data

The following files are required but not included in this repository. They can be obtained from the [SemEval 2022 Task 4 organizers](https://sites.google.com/view/pcl-detection-semeval2022):

- `dontpatronizeme_pcl.tsv` — Full dataset with paragraphs and labels
- `task4_test.tsv` — Test set paragraphs

## Usage

Open `model.ipynb` and run the cells sequentially. The notebook covers:

1. **Data loading** — Reads the TSV dataset, maps the 0–4 label scale to binary (0–1 → 0, 2–4 → 1), and splits by paragraph ID using the provided CSV files
2. **Exploratory data analysis** — Class distribution and bigram analysis
3. **Model training** — Fine-tunes DeBERTa-v3-base with combined binary + subcategory loss
4. **Threshold tuning** — Selects the optimal classification threshold on the validation set
5. **Prediction** — Generates `dev.txt` and `test.txt` with one prediction per line

## Error Analysis

Run the error analysis script to generate metrics and plots:

```bash
source venv/bin/activate
python scripts/error_analysis.py
```

Key findings on the dev set (101 false negatives, 149 false positives):

- **Subcategory recall:** Ranges from 0.47 (Shallow Solution) to 0.64 (Poorer the Merrier). Unbalanced Power (73.3%) and Compassion (47.5%) dominate the false negatives.
- **PCL intensity matters:** Paragraphs with 1–2 active subcategories are recalled at ~0.43, while those with 3+ reach 0.60+, confirming that subtle, single-dimension PCL is hardest to detect.
- **False positives driven by lexical overlap:** Non-PCL texts containing words like "hopeless", "homeless", or "poor families" in neutral contexts get incorrectly flagged (e.g. par_id=8395, a football commentary).
- **False negatives involve implicit tone:** Strong PCL paragraphs (label 4) are missed when patronising language is embedded in advocacy or sports contexts without overt negative vocabulary (e.g. par_ids 1572, 3661).

See `scripts/error_analysis_writeup.txt` for the full qualitative analysis.
