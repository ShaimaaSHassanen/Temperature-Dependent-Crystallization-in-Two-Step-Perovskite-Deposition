import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.data = None
        
    def load_data(self):
        """Loads the dataset from Excel."""
        print(f"Loading data from {self.file_path}...")
        self.df = pd.read_excel(self.file_path)
        return self.df

    def preprocess_and_filter(self):
        """
        Applies filtering logic defined in the notebook:
        - Selects specific columns.
        - Filters for specific deposition steps (1.0).
        - Filters outliers (PCE <= 27).
        """
        # Feature selection based on notebook
        selected_columns = [
            'Ref_DOI_number','Ref_internal_sample_id', 'Perovskite_composition_b_ions', 
            'Perovskite_composition_b_ions_coefficients', 'Perovskite_composition_c_ions', 
            'Perovskite_composition_c_ions_coefficients', 'Perovskite_composition_long_form',
            'Perovskite_additives_compounds', 'Perovskite_additives_concentrations', 
            'Perovskite_thickness', 'Perovskite_deposition_procedure', 
            'Perovskite_deposition_solvents', 'Perovskite_deposition_solvents_mixing_ratios', 
            'Perovskite_deposition_quenching_induced_crystallisation', 
            'Perovskite_deposition_quenching_media', 
            'Perovskite_deposition_thermal_annealing_temperature', 
            'Perovskite_deposition_thermal_annealing_time', 
            'Perovskite_deposition_number_of_deposition_steps', # Needed for filtering
            'Outdoor_average_over_n_number_of_cells', # Drop later
            'JV_reverse_scan_Voc', 'JV_reverse_scan_Jsc', 'JV_reverse_scan_FF', 'JV_reverse_scan_PCE',
            'JV_forward_scan_Voc', 'JV_forward_scan_Jsc', 'JV_forward_scan_FF', 'JV_forward_scan_PCE'
        ]
        
        # Filter columns that exist in the dataframe
        cols_to_use = [c for c in selected_columns if c in self.df.columns]
        self.data = self.df[cols_to_use].copy()

        # Apply Filters from notebook
        # 1. Filter outlier PCE
        self.data = self.data[self.data["JV_reverse_scan_PCE"] <= 27]
        
        # 2. Filter for single deposition step (from notebook logic)
        if 'Perovskite_deposition_number_of_deposition_steps' in self.data.columns:
            self.data = self.data[self.data['Perovskite_deposition_number_of_deposition_steps'] == 1.0]

        print(f"Data shape after filtering: {self.data.shape}")
        return self.data

    def get_features_and_labels(self):
        """Separates features (X) and target (y)."""
        if self.data is None:
            raise ValueError("Data not loaded or processed.")

        # Define Output Labels (Targets)
        output_labels = [
            'JV_reverse_scan_PCE', 'JV_forward_scan_PCE', 'JV_forward_scan_FF', 'JV_forward_scan_Voc',
            'JV_reverse_scan_Voc', 'JV_reverse_scan_Jsc', 'JV_reverse_scan_FF', 'JV_forward_scan_Jsc'
        ]
        
        # Helper columns to drop if they exist
        drop_cols = ['Ref_ID', 'Perovskite_deposition_number_of_deposition_steps', 
                     'Outdoor_average_over_n_number_of_cells', 'Ref_DOI_number', 
                     'Ref_internal_sample_id']
        
        # Prepare Features
        features = self.data.drop(columns=[c for c in output_labels if c in self.data.columns])
        features = features.drop(columns=[c for c in drop_cols if c in features.columns])
        
        # Select only Numerical Features for the model (simplified for standard ML compatibility)
        numerical_features = features.select_dtypes(include=['float64', 'int64'])
        
        # Fill missing values for training (basic imputation)
        numerical_features = numerical_features.fillna(0)

        # Target Variable
        target = self.data['JV_reverse_scan_PCE']
        
        # Drop rows where target is NaN
        mask = target.notna()
        return numerical_features[mask], target[mask]

    def split_data(self, test_size=0.2, random_state=42):
        X, y = self.get_features_and_labels()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
