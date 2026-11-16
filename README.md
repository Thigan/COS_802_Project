# Paraphrase Detection, Generation and Augmentation

COS 802 project

Do paraphrase models use meaning, or only shared words? It is run on a single laptop with a GPU, using open datasets and open pretrained models.

## Research questions

| | Question | Notebook |
|---|---|---|
| RQ1 | Does a fine tuned transformer (DistilBERT) outperform a TF-IDF and Logistic Regression baseline for paraphrase detection on QQP? | 01 |
| RQ2 | How well do both models separate meaning from word overlap on PAWS, where shared words are not a clue? | 01 |
| RQ3 | Does WordNet substitution, back translation or T5 give the best balance between preserving meaning and changing the wording? | 02 |
| RQ4 | Does the addition of quality filtered, automatically generated paraphrases to the training data improve detection? | 03 |

## Repository layout

```
COS802_Project/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_data_and_detection.ipynb        RQ1 and RQ2
│   ├── 02_paraphrase_generation.ipynb     RQ3
│   └── 03_augmentation_experiment.ipynb   RQ4
│
└── results/
    ├── qqp_train.csv
    ├── qqp_val.csv
    ├── qqp_test.csv
    ├── paws_test.csv
    ├── detection_results.csv
    ├── training_curve.csv
    ├── qqp_to_paws_change.csv
    ├── baseline_top_words.csv
    ├── confident_errors.csv
    ├── generated_paraphrases.csv
    ├── generation_results.csv
    ├── manual_review_sheet.csv
    ├── manual_review_results.csv
    ├── augmented_candidates.csv
    ├── augmentation_results.csv
    └── Figures
```

The `results/` folder has everything the notebooks produce, so you can look at the tables and figures without rerunning anything. If you do rerun them, set `PROJECT_DIR` in each notebook to the main project folder.

## Contents

| File | Purpose |
|---|---|
| `01_data_and_detection.ipynb` | Sets up the data, trains the TF-IDF baseline and DistilBERT, and tests both on QQP and PAWS. Also has the error analysis and the overlap check (RQ1 and RQ2). |
| `02_paraphrase_generation.ipynb` | Generates paraphrases for 120 sentences with the three methods and scores them. Also creates the sheet for manual rating (RQ3). |
| `03_augmentation_experiment.ipynb` | Back translates 800 training pairs, keeps the good ones, retrains DistilBERT and compares it with the original model (RQ4). |
| `results/manual_review_sheet.csv` | The 45 sentence pairs (15 per method) with my ratings for meaning, fluency and diversity. |
| `requirements.txt` | The Python packages the notebooks need. |

Run the notebooks in order, because notebooks 02 and 03 read files written by notebook 01.

## Data

| Dataset | Hugging Face name | Used for | Pairs |
|---|---|---|---|
| Quora Question Pairs (QQP) | `nyu-mll/glue`, config `qqp` | Training, validation and testing | 10,000 sampled with seed 42: 8,000 train, 1,000 validation, 1,000 test |
| PAWS Wikipedia | `google-research-datasets/paws`, config `labeled_final` | Testing only | 8,000 (44.2% paraphrases) |

PAWS QQP is not included in Hugging Face. Therefore PAWS Wikipedia is used.

## Setup

1. Install the packages with `pip install -r requirements.txt`. To use a GPU, install the matching PyTorch build from pytorch.org first.
2. Open each notebook and change `PROJECT_DIR` from the author's folder to a folder on your own computer. The notebooks save everything in a `results` folder inside it.
3. Run the notebooks in order: 01, then 02, then 03.
4. The first run downloads what it needs: the models and datasets from Hugging Face, and the WordNet files for notebook 02.

Output files are written to `results/`, and the notebooks use a GPU when one is available.

## Models and data sources

- DistilBERT: `distilbert-base-uncased`
- Back translation: `Helsinki-NLP/opus-mt-en-de` and `Helsinki-NLP/opus-mt-de-en`
- Paraphrase generation: `humarin/chatgpt_paraphraser_on_T5_base`
- Sentence embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- BERTScore: the default English model loaded by the `bert-score` package
- Datasets:
  - `nyu-mll/glue` (QQP): https://huggingface.co/datasets/nyu-mll/glue. QQP is part of the GLUE benchmark (Wang et al., 2018): https://arxiv.org/abs/1804.07461
  - `google-research-datasets/paws` (PAWS Wikipedia, `labeled_final`): https://huggingface.co/datasets/google-research-datasets/paws. Introduced by Zhang et al. (2019): https://arxiv.org/abs/1904.01130
