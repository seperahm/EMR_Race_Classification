import os
import pandas as pd
import numpy as np

NEW_LABELED_DATA_PARENT_FOLDER_PATH = 'data'
# Function to get labels for queried instances
class LabelingTool:
    def get_label_for(self, query_instance, label_type, label_names, with_confidence = False):

        while True:
            print(f"\nText: {query_instance['text']}")
            if with_confidence:
                print(f"Predicted race: {query_instance['predicted_race']}")
                print(f"Confidence: {query_instance['confidence']:.3f}")
            
            presence = input(f"Is {label_type} mentioned? (yes/no): ").lower().strip()
    
            if presence not in ['yes', 'no']:
                print("Invalid input. Please enter 'yes' or 'no'.")
                continue
    
            if presence == 'no':
                return {label_type: 'absent', 'assumed': False}
    
            # If race is present, ask for specific race
            labels = list(label_names)
            while True:
                print(f"\nAvailable {label_type} types:")
                for i, race in enumerate(labels):
                    print(f"{i}: {race}")
    
                try:
                    choice = int(input(f"Enter the number corresponding to the {label_type}: "))
                    if 0 <= choice < len(labels):
                        # return races[choice]
                        label = labels[choice]
                        break
                    else:
                        print(f"Invalid choice. Please enter a number between 0 and {len(labels)-1}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
             
            while True:
                assumed = input(f"Is {label_type} Assumed? (yes/no): ").lower().strip()
    
                if assumed not in ['yes', 'no']:
                    print("Invalid input. Please enter 'yes' or 'no'.")
                    continue
        
                if assumed == 'no':
                    return {label_type: label, 'assumed': False}
                else:
                    return {label_type: label, 'assumed': True}
    
    # Function to save newly labeled data to CSV
    def save_to_csv(self, df, filename):
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
            
    # main function to create newly labeled dataframe
    def assign_labels(self, pool, query_ids, unlabeled_data, label_type, label_names, with_confidence = False):
        y = []
        # if with_confidence:
        #     # Get indices for labeling
        #     query_ids = [instance['index'] for instance in query_ids]
        for instance in query_ids:
            if with_confidence:
                query_instance = instance
            else:
                query_instance = {
                    'index': instance,
                    'text': pool[instance],
                    'predicted_race': 'NO PREDICTION',
                    'confidence': 0
                }
                
            idx = query_instance['index']
            label = self.get_label_for(query_instance, label_type, label_names, with_confidence)
            race, assumed = label[label_type], label['assumed']
            y.append(race)
    
            # Create a DataFrame to store newly labeled data
            new_labeled_data = pd.DataFrame(columns=['patient_id', 'text', 'age',
                                                    'Race_Status', 'Race_Assumed', 'Race_Label',
                                                    'Citizenship_Status', 'Citizenship_Assumed', 'Citizenship_Label',
                                                    'Table_Results', 'site_id', 'provider_id'])
            # Add newly labeled data to DataFrame
            if label_type == 'race':
                new_labeled_data = pd.DataFrame({
                    'patient_id': unlabeled_data.iloc[idx]['patient_id'],
                    'text': query_instance['text'],
                    'age': unlabeled_data.iloc[idx]['age'],
                    'Race_Status': 'absent' if race == 'absent' else 'present',
                    'Race_Assumed': assumed,
                    'Race_Label': race,
                    'Citizenship_Status': None,
                    'Citizenship_Assumed': None,
                    'Citizenship_Label': None,
                    'Table_Results': None,
                    'site_id': unlabeled_data.iloc[idx]['site_id'],
                    'provider_id': unlabeled_data.iloc[idx]['provider_id'],
                }, index = [0])
            elif label_type == 'citizenship':
                new_labeled_data = pd.DataFrame({
                    'patient_id': unlabeled_data.iloc[idx]['patient_id'],
                    'text': query_instance['text'],
                    'age': unlabeled_data.iloc[idx]['age'],
                    'Race_Status': None,
                    'Race_Assumed': None,
                    'Race_Label': None,
                    'Citizenship_Status': 'absent' if race == 'absent' else 'present',
                    'Citizenship_Assumed': assumed,
                    'Citizenship_Label': race,
                    'Table_Results': None,
                    'site_id': unlabeled_data.iloc[idx]['site_id'],
                    'provider_id': unlabeled_data.iloc[idx]['provider_id'],
                }, index = [0])
    
            # Save newly labeled data to CSV
            print('Saving...')
            self.save_to_csv(new_labeled_data, f'{NEW_LABELED_DATA_PARENT_FOLDER_PATH}/{label_type}/newly_labeled_data.csv')
        return(np.array(y))