import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from tqdm import tqdm

QUERY_BATCH_SIZE = 64#256 # Number of unlabeled data entries to query and find the most uncertain sample
class BatchActiveLearner(ActiveLearner):
    def query(self, X_pool, n_instances=1, **query_kwargs):
        if 'classifier' not in query_kwargs:
            query_kwargs['classifier'] = self

        n_batches = (len(X_pool) + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE
        top_uncertain_instances = []
        top_uncertain_indices = []

        with tqdm(total=n_batches, desc="Processing query batches") as pbar:
            for i in range(0, len(X_pool), QUERY_BATCH_SIZE):
                batch = X_pool[i:i+QUERY_BATCH_SIZE]
                uncertainties = uncertainty_sampling(X=batch, n_instances=1, **query_kwargs)
                top_uncertain_indices.append(uncertainties[0][0] + i)
                pbar.update(1)

        return np.array(top_uncertain_indices)

    def query_by_race_present(self, X_pool, n_instances=1, **query_kwargs):
        if 'classifier' not in query_kwargs:
            query_kwargs['classifier'] = self

        n_batches = (len(X_pool) + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE
        race_present_indices = []

        # Get predictions for all instances
        with tqdm(total=n_batches, desc="Processing query batches") as pbar:
            for i in range(0, len(X_pool), QUERY_BATCH_SIZE):
                batch = X_pool[i:i+QUERY_BATCH_SIZE]

                # Get predictions for the batch
                predictions = self.estimator.predict(batch)
                probabilities = self.estimator.predict_proba(batch)

                # Find instances where race is predicted as present
                for j, pred in enumerate(predictions):
                    if pred != 'absent':  # or however you denote 'race present' in your predictions
                        batch_index = i + j
                        race_present_indices.append({
                            'index': batch_index,
                            'text': X_pool[batch_index],
                            'predicted_race': pred,
                            'confidence': np.max(probabilities[j])
                        })
                pbar.update(1)

        # Sort by confidence (you might want to prioritize less confident predictions)
        race_present_indices.sort(key=lambda x: x['confidence'])

        # Print predictions for selected instances
        print("\nSelected instances for labeling:")
        for idx, instance in enumerate(race_present_indices[:n_instances]):
            print(f"\nInstance {idx + 1}:")
            print(f"Text: {instance['text']}")
            print(f"Predicted race: {instance['predicted_race']}")
            print(f"Confidence: {instance['confidence']:.3f}")

        # Return only the indices
        selected_indices = [instance['index'] for instance in race_present_indices[:n_instances]]
        return np.array(selected_indices)