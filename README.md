# Active Learning for Race Classification from Electronic Health Records

> **ECE2500 Master of Engineering Project**  
> Department of Electrical & Computer Engineering, University of Toronto  
> Sepehr Ahmadi | February 2025

## Overview

This project implements a two-phase active learning pipeline for extracting race and immigration status information from unstructured clinical notes in Electronic Health Records (EHRs). The system addresses severe class imbalance (95% "absent" labels) and achieves a **13x improvement** in identifying race-present entries compared to random sampling.

### Key Results

| Metric | Random Sampling | Active Learning |
|--------|-----------------|-----------------|
| Race "Present" Prevalence | 5% (217/4,375) | 66% (910/1,386) |
| Annotation Efficiency | Baseline | **4x more "present" labels** |
| Data Required | 4,375 samples | **3x less data** |

## Motivation

Race and ethnicity are critical Social Determinants of Health (SDOHs) that influence health disparities and disease risk. However, race-related information in EHRs is often:
- Incomplete or missing-not-at-random
- Embedded in unstructured clinical notes
- Subject to misspellings, abbreviations, and lexical diversity

This pipeline enables automated extraction of race information to support healthcare equity research and address racial disparities in health outcomes.

## Architecture

### Two-Phase Active Learning Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE I: Initial Training                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Labeled Data │──▶│ Preprocessor │───▶│ Hierarchical BERT Model  │   │
│  │  (4,375)     │    │              │    │ (Binary + Multi-class)   │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE II: Active Learning Loop                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Unlabeled    │───▶│ Uncertainty  │──▶│ Human Annotator          │   │
│  │ Pool         │    │ Sampling     │    │ (CLI Tool)               │   │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│         ▲                                           │                   │
│         └───────────── Retrain ◀────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Hierarchical BERT Classifier

The classification model uses a two-level hierarchy to handle extreme class imbalance:

```
Input Text ──▶ [Level 1: Presence Model] ──▶ Present/Absent
                         │
                         ▼ (if Present)
              [Level 2: Category Model] ──▶ Race Category
```

**Level 1 (Binary):** Detects whether race information exists in the text  
**Level 2 (Multi-class):** Classifies into 9 racial categories when present

**Race Categories:** White, Black, East Asian, Southeast Asian, South Asian, Middle Eastern, Mixed Heritage, Latin American, Indigenous

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- BERT-base model weights

### Dependencies
```bash
pip install torch transformers scikit-learn pandas numpy tqdm modAL-python
```

### BERT Model Setup
```bash
# Download BERT-base and set path in notebooks:
BERT_VERSION_PATH = '/path/to/bertbase'
```

## Quick Start

### Race Classification Pipeline
```bash
jupyter notebook main_race.ipynb
```

### Immigration Status Classification
```bash
jupyter notebook main_Imm.ipynb
```

### Random Sampling Baseline (for comparison)
```bash
jupyter notebook main_race_predictLearningRandom.ipynb
```

## Project Structure

```
├── main_race.ipynb                      # Primary race active learning pipeline
├── main_Imm.ipynb                       # Immigration status pipeline
├── main_race_predictLearning.ipynb      # Prediction-guided sampling variant
├── main_race_predictLearningRandom.ipynb # Random sampling baseline
├── Race_Dataset_CLEANED.csv             # Initial labeled dataset
├── tools/
│   ├── HierarchicalBertClassifier.py    # Two-level BERT model
│   ├── BatchActiveLearner.py            # Active learning with uncertainty sampling
│   ├── DataLoader.py                    # Dataset management & merging
│   ├── DataPreprocessor.py              # Sampling, upsampling & train/test split
│   ├── LabelingTool.py                  # Interactive CLI annotation interface
│   ├── TextDataset.py                   # PyTorch dataset wrapper
│   └── README.md                        # Tools API documentation
└── data/
    ├── race/                            # Race annotations & outputs
    │   ├── filtered/                    # Exported filtered datasets
    │   └── newly_labeled_data.csv       # Active learning annotations
    └── citizenship/                     # Immigration annotations
```

## Usage

### Basic Pipeline

```python
from tools.DataLoader import DataLoader
from tools.DataPreprocessor import DataPreprocessor
from tools.HierarchicalBertClassifier import HierarchicalBertClassifier
from tools.BatchActiveLearner import BatchActiveLearner
from modAL.uncertainty import uncertainty_sampling

# 1. Load data
loader = DataLoader()
df = loader.load_race_data()
unlabeled = loader.load_unlabeled_data()
race_names = loader.get_label_names(df)

# 2. Preprocess with balanced sampling
preprocessor = DataPreprocessor()
train_df, test_df = preprocessor.sample_and_split_data(df)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(train_df, test_df)

# 3. Initialize classifier
classifier = HierarchicalBertClassifier(BERT_VERSION_PATH, len(race_names))

# 4. Create active learner
learner = BatchActiveLearner(
    estimator=classifier,
    X_training=X_train,
    y_training=y_train,
    query_strategy=uncertainty_sampling
)

# 5. Active learning loop
for _ in range(n_iterations):
    query_idx = learner.query(X_pool)
    # ... annotation and teaching
```

### Interactive Labeling

The `LabelingTool` provides a CLI interface for human annotation:

```
Text: "Racial or Ethnic Group-Black - Caribbean (Barbadian, Jamaican)..."
Predicted race: Black
Confidence: 0.847

Is race mentioned? (yes/no): yes
Available race types: 0: White, 1: Black, 2: East Asian...
Enter number: 1
Is race Assumed? (yes/no): no
```

## Configuration

### Key Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `BATCH_SIZE` | HierarchicalBertClassifier.py | 16 | Training batch size |
| `MAX_LENGTH` | HierarchicalBertClassifier.py | 512 | BERT token limit |
| `NUM_EPOCHS` | HierarchicalBertClassifier.py | 10 | Training epochs |
| `QUERY_BATCH_SIZE` | BatchActiveLearner.py | 64 | Pool batch size for querying |
| `RACE_SAMPLE_SIZE` | DataPreprocessor.py | 100 | Samples per race category |
| `RACE_ABSENT_MULTIPLIER` | DataPreprocessor.py | 9 | Absent class multiplier |
| `TEST_SIZE` | DataPreprocessor.py | 0.2 | Train/test split ratio |

## Data Format

### Input CSV Structure
```csv
patient_id,text,Race_Label
12345,"STRESS TEST: screening negative...",absent
67890,"Racial or Ethnic Group-Black...",Black
```

### Clinical Note Examples

| Type | Example | Label |
|------|---------|-------|
| Absent | "STRESS TEST: screening negative, Family: married with 3 sons, Occupation: Employed..." | absent |
| Present | "Racial or Ethnic Group-Black - Caribbean (Barbadian, Jamaican) Widowed..." | Black |

## Model Swapping

The architecture is modular—replace BERT with any PyTorch/scikit-learn model:

```python
# In HierarchicalBertClassifier.__init__():
self.presence_model = YourBinaryModel(...)    # Replace binary classifier
self.race_model = YourMultiClassModel(...)    # Replace category classifier
```

## Results Summary

The active learning approach significantly outperformed random sampling:

- **Race Present Prevalence:** Increased from 5% to 66% in annotated samples
- **Minority Class Coverage:** Substantially improved representation for underrepresented categories:
  - Middle Eastern: 84 (AL) vs 27 (RS)
  - Latin American: 39 (AL) vs 3 (RS)
  - Mixed Heritage: 59 (AL) vs 2 (RS)
- **Annotation Efficiency:** 4x more informative samples identified with 3x less data

## Citation

```bibtex
@mastersthesis{ahmadi2025active,
  title={Implementation of Active Learning Pipeline for Classification of 
         Race and Origin from Electronic Health Records},
  author={Ahmadi, Sepehr},
  year={2025},
  school={University of Toronto},
  department={Electrical \& Computer Engineering},
  type={Master of Engineering Project}
}
```

## Acknowledgments

This project was developed as part of the research initiative *Machine Learning Techniques for Phenotyping Social Determinants of Health Characteristics in Electronic Health Records*, using data from the University of Toronto Practice-Based Research Network (UTOPIAN), Department of Family and Community Medicine.

## License

Research project—contact for usage permissions.
