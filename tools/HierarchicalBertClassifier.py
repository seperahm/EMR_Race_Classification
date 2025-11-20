import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from tools.TextDataset import TextDataset
from tqdm import tqdm

BATCH_SIZE = 16
MAX_LENGTH = 512
NUM_EPOCHS = 10

class HierarchicalBertClassifier:
    """
    Hierchical BERT classifier is a text classifier consisted of two levels of a BERT model;
    First Level is a BERT model that detects if in an input text, race is 'present' or 'absent' (Binary Classification)
    Second Level is a BERT model that classifies the race type when race is 'present' (Multi-Class classification)
    """
    def __init__(self, model_name, num_races):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Binary classifier for presence/absence of race
        self.presence_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)

        # Multi-class classifier for specific races
        self.race_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_races).to(self.device)

        # Label encoders
        self.presence_encoder = LabelEncoder()
        self.race_encoder = LabelEncoder()

    def fit(self, X, y):
        # Prepare data for presence/absence classification
        y_presence = np.where(y == 'absent', 'absent', 'present')
        y_presence_encoded = self.presence_encoder.fit_transform(y_presence)

        # Prepare data for race classification (only for samples where race is present)
        X_race = X[y != 'absent']
        y_race = y[y != 'absent']
        y_race_encoded = self.race_encoder.fit_transform(y_race)

        # Train presence/absence model
        print("Training presence/absence model:")
        self._train_model(self.presence_model, X, y_presence_encoded)

        # Train race model
        print("\nTraining race model:")
        self._train_model(self.race_model, X_race, y_race_encoded)

    def _train_model(self, model, X, y):
        dataset = TextDataset(X, y, self.tokenizer, max_length=MAX_LENGTH)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        model.train()
        for epoch in range(NUM_EPOCHS):  # You may want to adjust the number of epochs
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
            for batch in progress_bar:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.shape[0]

                # Update progress bar
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1),
                                          'accuracy': correct_predictions / total_predictions})

            # Print epoch results
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss / len(train_loader):.4f}, "
                  f"Accuracy: {correct_predictions / total_predictions:.4f}")

    def predict(self, X):
        presence_proba = self._predict_proba(self.presence_model, X)
        presence_pred = np.argmax(presence_proba, axis=1)

        race_proba = self._predict_proba(self.race_model, X)
        race_pred = np.argmax(race_proba, axis=1)

        final_pred = np.where(presence_pred == 0, 
                              self.presence_encoder.inverse_transform([0])[0],
                              self.race_encoder.inverse_transform(race_pred))
        return final_pred

    def predict_proba(self, X):
        presence_proba = self._predict_proba(self.presence_model, X)
        race_proba = self._predict_proba(self.race_model, X)

        # Combine probabilities
        final_proba = np.zeros((len(X), len(self.race_encoder.classes_) + 1))  # +1 for 'absent'
        final_proba[:, 0] = presence_proba[:, 0]  # Probability of 'absent'
        final_proba[:, 1:] = presence_proba[:, 1:] * race_proba  # Probabilities of specific races

        return final_proba

    def _predict_proba(self, model, X):
        model.eval()
        dataset = TextDataset(X, [0]*len(X), self.tokenizer, max_length=MAX_LENGTH)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        probabilities = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return np.array(probabilities)
