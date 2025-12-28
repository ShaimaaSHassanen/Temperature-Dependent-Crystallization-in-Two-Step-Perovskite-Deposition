import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """Loads the dataset from Excel."""
        print(f"Loading data from {self.file_path}...")
        self.df = pd.read_excel(self.file_path)
        return self.df

    def clean_and_filter_outliers(self):
        """General cleaning applied to the whole dataset."""
        # 1. Filter outlier PCE based on your notebook
        if "JV_reverse_scan_PCE" in self.df.columns:
            self.df = self.df[self.df["JV_reverse_scan_PCE"] <= 27]
        
        # Drop columns usually not needed for training or that leak info
        drop_cols = ['Ref_ID', 'Outdoor_average_over_n_number_of_cells', 'Ref_DOI_number', 
                     'Ref_internal_sample_id']
        self.df = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns])
        
        # Handle missing values (simple fill for numerical)
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numerical_cols] = self.df[numerical_cols].fillna(0)

    def get_datasets_by_step(self):
        """
        Splits the data into 1-step and 2-step deposition subsets.
        Returns: A dictionary {'1_step': df1, '2_step': df2}
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        datasets = {}
        
        # split based on deposition steps
        if 'Perovskite_deposition_number_of_deposition_steps' in self.df.columns:
            df_1 = self.df[self.df['Perovskite_deposition_number_of_deposition_steps'] == 1.0].copy()
            df_2 = self.df[self.df['Perovskite_deposition_number_of_deposition_steps'] == 2.0].copy()
            
            datasets['1_step'] = df_1
            datasets['2_step'] = df_2
            
            print(f"Data Split: 1-step samples: {len(df_1)}, 2-step samples: {len(df_2)}")
        else:
            print("Warning: Deposition steps column not found. Returning full data as 'all_data'.")
            datasets['all_data'] = self.df

        return datasets

    def prepare_features_labels(self, data):
        """Separates Features (X) and Target (y) for a specific dataset."""
        target_col = 'JV_reverse_scan_PCE'
        
        # Output labels to exclude from X
        output_labels = [
            'JV_reverse_scan_PCE', 'JV_forward_scan_PCE', 'JV_forward_scan_FF', 'JV_forward_scan_Voc',
            'JV_reverse_scan_Voc', 'JV_reverse_scan_Jsc', 'JV_reverse_scan_FF', 'JV_forward_scan_Jsc'
        ]
        
        # Drop outputs from inputs
        X = data.drop(columns=[c for c in output_labels if c in data.columns])
        
        # Ensure we only use numeric features for the Decision Tree
        X = X.select_dtypes(include=['float64', 'int64'])
        y = data[target_col]
        
        return X, y
