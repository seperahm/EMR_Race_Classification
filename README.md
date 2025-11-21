# Active Learning for Race Classification from EHRs

Two-phase active learning pipeline for extracting race and immigration status from electronic health records using hierarchical BERT classifiers.

## Key Features

- **Hierarchical Classification**: Two-level BERT model (presence detection → category classification)
- **Active Learning**: Uncertainty sampling to address severe class imbalance (95% "absent" labels)
- **Modular Architecture**: Swap BERT with any PyTorch/Scikit-learn model
- **Interactive Labeling**: CLI tool for human-in-the-loop annotation
- **4x Efficiency Gain**: Active learning identified 4x more "present" labels vs random sampling with 3x less data

## Installation
```bash
pip install torch transformers scikit-learn pandas numpy tqdm modal
```

Download BERT-base model:
```bash
# Set path in notebooks: BERT_VERSION_PATH = 'path/to/bertbase'
```

## Quick Start

### Race Classification
```bash
jupyter notebook main_race.ipynb
```

### Immigration Status Classification
```bash
jupyter notebook main_Imm.ipynb
```

### Baseline (Random Sampling)
```bash
jupyter notebook main_race_predictLearningRandom.ipynb
```

## Project Structure
```
├── main_race.ipynb                    # Race active learning pipeline
├── main_Imm.ipynb                     # Immigration status pipeline
├── main_race_predictLearning.ipynb    # Prediction-based variant
├── tools/
│   ├── HierarchicalBertClassifier.py  # Two-level BERT model
│   ├── BatchActiveLearner.py          # Active learning with uncertainty sampling
│   ├── DataLoader.py                  # Dataset management
│   ├── DataPreprocessor.py            # Sampling & train/test split
│   ├── LabelingTool.py                # Interactive annotation interface
│   └── TextDataset.py                 # PyTorch dataset wrapper
└── data/
    ├── race/                          # Race annotations
    └── citizenship/                   # Immigration annotations
```

## Data Requirements

**Input Format** (CSV):
- `patient_id`: Unique identifier
- `text`: Clinical note text
- `label`: Race category or 'absent'

**Race Categories**: White, Black, East Asian, Southeast Asian, South Asian, Middle Eastern, Mixed Heritage, Latin American, Indigenous

**Immigration Categories**: Custom citizenship labels

## Configuration

Edit constants in tool files:
```python
# DataPreprocessor.py
RACE_SAMPLE_SIZE = 100          # Samples per race category
RACE_ABSENT_MULTIPLIER = 9      # Absent label multiplier
TEST_SIZE = 0.2                 # Train/test split

# HierarchicalBertClassifier.py
BATCH_SIZE = 16
MAX_LENGTH = 512
NUM_EPOCHS = 10

# BatchActiveLearner.py
QUERY_BATCH_SIZE = 64           # Unlabeled batch size for querying
```

## Hierarchical BERT Architecture

**Level 1**: Binary classifier (race present/absent)  
**Level 2**: Multi-class classifier (9 race categories)

Trained independently to handle extreme class imbalance.

## Results

Active learning achieved **66% Race Present prevalence** vs **5% with random sampling** (13x improvement). See full report for detailed analysis.

## Citation
```bibtex
@mastersthesis{ahmadi2025active,
  title={Implementation of Active Learning Pipeline for Classification of Race and Origin from Electronic Health Records},
  author={Ahmadi, Sepehr},
  year={2025},
  school={University of Toronto},
  department={Electrical & Computer Engineering}
}
```

## License

Research project - contact for usage permissions.
