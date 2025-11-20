import pandas as pd 
import glob
import os

CLEANED_DATA_PATH = 'Race_Dataset_CLEANED.csv'
NEW_RACE_LABELED_DATA_FOLDER_PATH = 'data/race'
OUTPUT_CSV_RACE_FOLDER_PATH = 'data/race/filtered'
NEW_IMM_LABELED_DATA_FOLDER_PATH = 'data/citizenship'
UTOPIAN_PATH_UNLABELED = "/home/saveuser/S/projects/rawan2_project/Cleaned_dataset/UTOPIAN_Dataset.csv"

class DataLoader:
    def load_cleaned_data(self, result_csv_file_path):
        """
        Rawan's orignial 'load_data' method for the CLEANED datastet
        """
        
        df = pd.read_csv(result_csv_file_path)
        infor_columns = df.columns[:15]
        # retrieve columns that belong to different category
        place_of_birth_related = [column for column in infor_columns if 'POB' in column]
        race_related = [column for column in infor_columns if 'Race' in column]
        citizenship_related = [column for column in infor_columns if 'Citizenship' in column]
        # append patient id and text
        patient_id_column = df['patient_id']
        text_column = df['text']
        # with characteristic Status and if it's Assumed
        place_of_birth_full = pd.concat([patient_id_column,text_column,df[place_of_birth_related]],axis=1)
        race_full = pd.concat([patient_id_column,text_column,df[race_related]],axis=1)
        citizenship_full = pd.concat([patient_id_column,text_column,df[citizenship_related]],axis=1)
        # only with characteristic labels
        race_label_column = df['Race_Label']
        pob_label_column = df['POB_Label']
        citizenship_label_column = df['Citizenship_Label']
        place_of_birth = pd.concat([patient_id_column,text_column,pob_label_column],axis=1)
        race = pd.concat([patient_id_column,text_column,race_label_column],axis=1)
        citizenship = pd.concat([patient_id_column,text_column,citizenship_label_column],axis=1)
        data = {"place_of_birth_full":place_of_birth_full,
                "place_of_birth": place_of_birth,
                "race_full": race_full,
                "race": race,
                "citizenship_full": citizenship_full,
                "citizenship": citizenship}
        return data

    def concat_csv_with_df(self, csv_file, df):
        """
        Loads data from a CSV file, concatenates it with an existing DataFrame, and
        returns a new DataFrame containing only the columns common to both.
        
        Args:
        csv_file: The path to the CSV file.
        df: The existing DataFrame.
        
        Returns:
        A new DataFrame containing the concatenated data with only common columns.
        """
        
        # Load the CSV data into a DataFrame
        csv_data = pd.read_csv(csv_file)
        
        # Find the common columns between the CSV data and the existing DataFrame
        common_cols = csv_data.columns.intersection(df.columns)
        
        # Concatenate the DataFrames, keeping only the common columns
        new_df = pd.concat([csv_data[common_cols], df[common_cols]], ignore_index=True)
        
        return new_df

    def concat_csv_files_in_folder_with_df(self, folder_path, df):
        """
        Loads all CSV files within a specified folder, concatenates them with an existing DataFrame, and
        returns a new DataFrame containing only the columns common to both.
        
        Args:
        folder_path: The path to the folder containing CSV files.
        df: The existing DataFrame.
        
        Returns:
        A new DataFrame containing the concatenated data with only common columns.
        """
        
        # Get a list of all CSV files in the folder
        csv_files = glob.glob(folder_path + "/*.csv")
        
        new_df = df
        if len(csv_files)>0:
            for csv_file in csv_files:
                new_df = self.concat_csv_with_df(csv_file, new_df)
        else:
            print('No newly labeled data found; Only using the CLEANED dataset')
        
        return new_df

    def concat_csv_files_in_folder_with_filter(self, folder_path, df, filter_column, filter_value):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        new_df = df.copy()
    
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            temp_df = pd.read_csv(file_path)
            new_df = pd.concat([new_df, temp_df[temp_df[filter_column] != filter_value]], ignore_index=True)
        return new_df
        
        
    def load_race_data(self, output_csv = False):
        dataloader = self.load_cleaned_data(CLEANED_DATA_PATH)
        # select dataframe
        df = dataloader['race_full']
        if output_csv:
            # create .csv files that Rawan requested
            filtered_df_1 = self.concat_csv_files_in_folder_with_filter(NEW_RACE_LABELED_DATA_FOLDER_PATH, df, 'Race_Label', 'absent')
            filtered_df_2 = self.concat_csv_files_in_folder_with_filter(NEW_RACE_LABELED_DATA_FOLDER_PATH, pd.DataFrame(), 'Race_Label', '')
            output_csv_file_path_1 = os.path.join(OUTPUT_CSV_RACE_FOLDER_PATH, 'original_and_active_labels.csv')
            output_csv_file_path_2 = os.path.join(OUTPUT_CSV_RACE_FOLDER_PATH, 'active_labels.csv')
            print(f"Number of rows added to the original DataFrame in \'{output_csv_file_path_1}\': {len(filtered_df_1) - len(df)}")
            print(f"Number active learning labels located in \'{output_csv_file_path_2}\': {len(filtered_df_2)}")
            filtered_df_1.to_csv(output_csv_file_path_1, index=False)
            filtered_df_2.to_csv(output_csv_file_path_2, index=False)
        # join the newly labeled dataframe(s) with the original one (Rawan's)
        df = self.concat_csv_files_in_folder_with_df(NEW_RACE_LABELED_DATA_FOLDER_PATH, df)
        # Rename the 'Race_Label' column to 'label'
        df = df.rename(columns={'Race_Label': 'label'})
        return df

    def load_Imm_data(self):
        dataloader = self.load_cleaned_data(CLEANED_DATA_PATH)
        # select dataframe
        df = dataloader['citizenship_full']
        # join the newly labeled dataframe(s) with the original one (Rawan's)
        df = self.concat_csv_files_in_folder_with_df(NEW_IMM_LABELED_DATA_FOLDER_PATH, df)
        # Rename the 'Race_Label' column to 'label'
        df = df.rename(columns={'Citizenship_Label': 'label'})
        return df

    import pandas as pd

    def filter_unlabeled_data(self, unlabeled_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows from unlabeled_data where patient_ids match those in df.
    
        Parameters:
        -----------
        unlabeled_data : pd.DataFrame
            DataFrame containing rows to be filtered
        df : pd.DataFrame
            DataFrame containing reference patient_ids to filter against
    
        Returns:
        --------
        pd.DataFrame
            Filtered version of unlabeled_data with matching patient_ids removed
        """
    
        # Get unique patient_ids from df
        existing_patient_ids = df['patient_id'].unique()
    
        # Filter unlabeled_data to keep only rows where patient_id is not in existing_patient_ids
        filtered_data = unlabeled_data[~unlabeled_data['patient_id'].isin(existing_patient_ids)]
    
        # Print some information about the filtering
        removed_count = len(unlabeled_data) - len(filtered_data)
        print(f"Removed {removed_count} rows from unlabeled_data")
        print(f"Original shape: {unlabeled_data.shape}")
        print(f"New shape: {filtered_data.shape}")
    
        return filtered_data

    def load_unlabeled_data(self):
        unlabeled_data = pd.read_csv(UTOPIAN_PATH_UNLABELED)
        # cleanup dataframe
        unlabeled_data = self.filter_and_rename_columns(unlabeled_data)
        return unlabeled_data
        

    def filter_and_rename_columns(self,df):
        """
        Drops all columns except 'patient_id' and 'text_orig', renames 'text_orig' to 'text'.
        
        Args:
          df (pandas.DataFrame): The DataFrame to modify.
        
        Returns:
          pandas.DataFrame: The modified DataFrame.
        """
        
        # Get the columns to keep
        columns_to_keep = ['patient_id', 'text_orig', 'patient_age', 'site_id', 'provider_id']
        # Filter the DataFrame to keep only the specified columns
        filtered_df = df[columns_to_keep]
        # Rename the 'text_orig' column to 'text'
        filtered_df = filtered_df.rename(columns={'text_orig': 'text', 'patient_age': 'age', })
        
        return filtered_df

    def get_label_names(self, df):
        label_names = set(df['label'].unique()) - {'absent'}
        return label_names

    def scan_csv_files(self, directory):
        """
        Recursively scan through a directory for CSV files and analyze their contents.
    
        Parameters:
        directory (str): The root directory to start scanning from
        """
        # Walk through the directory
        for root, dirs, files in os.walk(directory, topdown=True):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                # Check if the file is a CSV
                if file.lower().endswith('.csv'):
                    # Construct the full file path
                    full_path = os.path.join(root, file)
    
                    # Print the directory and filename
                    print(f"\nFound CSV file:")
                    print(f"Directory: {root}")
                    print(f"Filename: {file}")
    
                    try:
                        # Read the CSV file
                        df = pd.read_csv(full_path)
    
                        # Check for 'label' column
                        if 'label' in df.columns:
                            print("\nValue counts for 'label' column:")
                            print(df['label'].value_counts())
    
                        # Check for 'Race_Label' column
                        if 'Race_Label' in df.columns:
                            print("\nValue counts for 'Race_Label' column:")
                            print(df['Race_Label'].value_counts())
    
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

