# Tools Module

Core utilities for the active learning pipeline. All modules are modular and can be used independently or as part of the full pipeline.

## Module Overview

### HierarchicalBertClassifier.py
Two-level BERT-based classifier for race/immigration classification.

**Architecture:**
- **Level 1**: Binary classifier (present/absent)
- **Level 2**: Multi-class classifier (specific categories)

**Key Methods:**
```python
classifier = HierarchicalBertClassifier(model_path, num_categories)
classifier.fit(X_train, y_train)           # Train both levels
predictions = classifier.predict(X_test)    # Hierarchical prediction
probabilities = classifier.predict_proba(X_test)  # Combined probabilities
```

**Configuration:**
```python
BATCH_SIZE = 16      # Training batch size
MAX_LENGTH = 512     # BERT token limit
NUM_EPOCHS = 10      # Training epochs
```

**Model Swapping:**
Replace BERT with any PyTorch/sklearn model by modifying initialization:
```python
self.presence_model = YourModel(...)  # Binary classifier
self.race_model = YourModel(...)      # Multi-class classifier
```

---

### BatchActiveLearner.py
Active learning wrapper extending modAL's `ActiveLearner` with batch processing.

**Query Strategies:**

1. **Uncertainty Sampling** (default):
```python
learner = BatchActiveLearner(
    estimator=classifier,
    X_training=X_train,
    y_training=y_train,
    query_strategy=uncertainty_sampling
)
query_idx = learner.query(X_pool)  # Returns most uncertain samples
```

2. **Race-Present Query** (custom):
```python
query_idx = learner.query_by_race_present(X_pool, n_instances=10)
# Returns instances predicted as 'present' sorted by confidence
```

**Configuration:**
```python
QUERY_BATCH_SIZE = 64  # Pool batch size for memory efficiency
```

**Integration:**
```python
learner.teach(X=new_samples, y=new_labels)  # Incremental learning
predictions = learner.predict(X_test)
```

---

### DataLoader.py
Manages datasets from multiple sources with automatic merging.

**Primary Methods:**

**Load Labeled Data:**
```python
dataloader = DataLoader()

# Race data with optional CSV export
df_race = dataloader.load_race_data(output_csv=True)
# Outputs: data/race/filtered/original_and_active_labels.csv
#          data/race/filtered/active_labels.csv

# Immigration data
df_imm = dataloader.load_Imm_data()

# Unlabeled pool
unlabeled = dataloader.load_unlabeled_data()
```

**Filter Unlabeled Data:**
```python
# Remove already-labeled patient IDs from pool
filtered = dataloader.filter_unlabeled_data(unlabeled, df_race)
```

**Utility Methods:**
```python
label_names = dataloader.get_label_names(df)  # Extracts unique labels
dataloader.scan_csv_files('./data')  # Recursively analyze CSVs
```

**Data Paths (configure in file):**
```python
CLEANED_DATA_PATH = 'Race_Dataset_CLEANED.csv'
NEW_RACE_LABELED_DATA_FOLDER_PATH = 'data/race'
NEW_IMM_LABELED_DATA_FOLDER_PATH = 'data/citizenship'
UTOPIAN_PATH_UNLABELED = "/path/to/UTOPIAN_Dataset.csv"
```

**Expected CSV Structure:**
- **Labeled data**: `patient_id`, `text`, `Race_Label`/`Citizenship_Label`, `Race_Status`, `Race_Assumed`, etc.
- **Unlabeled data**: `patient_id`, `text_orig`, `patient_age`, `site_id`, `provider_id`

---

### DataPreprocessor.py
Handles sampling, upsampling, and train/test splitting with stratification.

**Workflow:**
```python
preprocessor = DataPreprocessor()

# Sample and split in one step
train_df, test_df = preprocessor.sample_and_split_data(
    df, 
    sample_size=100,        # Samples per category
    absent_multiplier=9     # Multiplier for 'absent' class
)

# Prepare for BERT
X_train, X_test, y_train, y_test = preprocessor.prepare_data(train_df, test_df)
```

**Balanced Upsampling:**
Automatically handles class imbalance:
- Classes with < `sample_size` entries: repeated with remainder sampling
- Classes with ≥ `sample_size` entries: randomly sampled to `sample_size`
- 'absent' class: sampled to `sample_size * absent_multiplier`

**Configuration:**
```python
RACE_SAMPLE_SIZE = 100          # Per-category sample size
RACE_ABSENT_MULTIPLIER = 9      # Absent class multiplier
TEST_SIZE = 0.2                 # 80/20 train/test split
RANDOM_STATE = 42               # Reproducibility
```

---

### LabelingTool.py
Interactive CLI for human annotation with automatic CSV export.

**Usage:**
```python
labeler = LabelingTool()

# Assign labels with optional prediction display
labels = labeler.assign_labels(
    pool=X_pool,
    query_ids=query_indices,
    unlabeled_data=unlabeled_df,
    label_type='race',           # or 'citizenship'
    label_names=race_names,
    with_confidence=True         # Show predictions
)
```

**Interactive Prompts:**
```
Text: [clinical note excerpt]
Predicted race: Black
Confidence: 0.847

Is race mentioned? (yes/no): yes

Available race types:
0: White
1: Black
2: East Asian
...
Enter number: 1

Is race Assumed? (yes/no): no
```

**Auto-Export:**
Labels saved to `data/{label_type}/newly_labeled_data.csv` with columns:
- `patient_id`, `text`, `age`, `Race_Status`, `Race_Assumed`, `Race_Label`, `site_id`, `provider_id`

**Configuration:**
```python
NEW_LABELED_DATA_PARENT_FOLDER_PATH = 'data'
```

---

### TextDataset.py
PyTorch Dataset wrapper for BERT tokenization.

**Usage:**
```python
dataset = TextDataset(
    texts=X_train,
    labels=y_train,
    tokenizer=bert_tokenizer,
    max_length=512
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']
```

**Returns:**
- `input_ids`: Tokenized text (padded/truncated to `max_length`)
- `attention_mask`: Attention mask for padding tokens
- `label`: Encoded label as LongTensor

---

## Integration Example

Complete pipeline showing module interactions:
```python
from tools.DataLoader import DataLoader
from tools.DataPreprocessor import DataPreprocessor
from tools.HierarchicalBertClassifier import HierarchicalBertClassifier
from tools.BatchActiveLearner import BatchActiveLearner
from tools.LabelingTool import LabelingTool
from modAL.uncertainty import uncertainty_sampling

# 1. Load data
dataloader = DataLoader()
df = dataloader.load_race_data()
unlabeled_data = dataloader.load_unlabeled_data()
race_names = dataloader.get_label_names(df)

# 2. Preprocess
preprocessor = DataPreprocessor()
train_df, test_df = preprocessor.sample_and_split_data(df)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(train_df, test_df)

# 3. Initialize classifier
classifier = HierarchicalBertClassifier('path/to/bert', len(race_names))

# 4. Active learning loop
learner = BatchActiveLearner(
    estimator=classifier,
    X_training=X_train,
    y_training=y_train,
    query_strategy=uncertainty_sampling
)

labeler = LabelingTool()

for iteration in range(10):
    # Query uncertain samples
    X_pool = unlabeled_data['text'].values
    query_idx = learner.query(X_pool)
    
    # Human annotation
    new_labels = labeler.assign_labels(
        X_pool, query_idx, unlabeled_data, 'race', race_names
    )
    
    # Teach model
    query_instances = [X_pool[idx] for idx in query_idx]
    learner.teach(X=query_instances, y=new_labels)
    
    # Update pool
    unlabeled_data = unlabeled_data.drop(query_idx).reset_index(drop=True)

# 5. Final predictions
predictions = learner.predict(X_test)
```

---

## Dependencies
```python
# Core
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Deep Learning
torch>=1.10.0
transformers>=4.15.0

# Active Learning
modAL>=0.4.0

# Utils
tqdm>=4.62.0
```

---

## Design Patterns

**Modularity**: Each module is independent; swap implementations without breaking others.

**Sklearn Compatibility**: `HierarchicalBertClassifier` follows sklearn's API:
```python
fit(X, y)          # Training
predict(X)         # Inference
predict_proba(X)   # Probability estimates
```

**State Management**: `DataLoader` auto-merges newly labeled data from `data/` folders into training sets.

**Memory Efficiency**: `BatchActiveLearner` processes large unlabeled pools in configurable batches.

---

## File Outputs

**DataLoader:**
- `data/race/filtered/original_and_active_labels.csv`: Combined dataset
- `data/race/filtered/active_labels.csv`: Active learning labels only

**LabelingTool:**
- `data/race/newly_labeled_data.csv`: Race annotations
- `data/citizenship/newly_labeled_data.csv`: Immigration annotations

All CSVs are auto-merged on next `DataLoader` initialization.
