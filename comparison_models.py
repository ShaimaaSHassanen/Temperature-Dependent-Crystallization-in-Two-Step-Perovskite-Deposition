import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import sys

# --- 1. DATA LOADING ---
print(">>> [STEP 1/5] Loading Dataset...")
try:
    all_data = pd.read_excel('Perovskite_FAIR database_ORIGINAL_all_data.xlsx')
    print(f"    Data loaded successfully. Shape: {all_data.shape}")
except FileNotFoundError:
    print("    ERROR: Excel file not found. Please check the path.")
    sys.exit(1)

# Select relevant columns
data = all_data[['Ref_DOI_number','Ref_internal_sample_id', 'Perovskite_composition_b_ions', 
                 'Perovskite_composition_b_ions_coefficients', 'Perovskite_composition_c_ions', 
                 'Perovskite_composition_c_ions_coefficients', 'Perovskite_composition_long_form',
                 'Perovskite_additives_compounds', 'Perovskite_additives_concentrations', 
                 'Perovskite_thickness', 'Perovskite_deposition_procedure', 
                 'Perovskite_deposition_solvents', 'Perovskite_deposition_solvents_mixing_ratios', 
                 'Perovskite_deposition_quenching_induced_crystallisation', 
                 'Perovskite_deposition_quenching_media', 
                 'Perovskite_deposition_thermal_annealing_temperature', 
                 'Perovskite_deposition_thermal_annealing_time', 
                 'JV_reverse_scan_Voc', 'JV_reverse_scan_Jsc', 'JV_reverse_scan_FF', 
                 'JV_reverse_scan_PCE', 'JV_forward_scan_Voc', 'JV_forward_scan_Jsc', 
                 'JV_forward_scan_FF', 'JV_forward_scan_PCE']]

# Filter unlikely PCE values
data = data[data["JV_reverse_scan_PCE"] <= 27]

# Filter for 1-step deposition
data_1_step = all_data[all_data['Perovskite_deposition_number_of_deposition_steps'] == 1.0]

# Define Outputs
output_labels = [
    'JV_reverse_scan_PCE', 'JV_forward_scan_PCE', 'JV_forward_scan_FF', 'JV_forward_scan_Voc',
    'JV_reverse_scan_Voc', 'JV_reverse_scan_Jsc', 'JV_reverse_scan_FF', 'JV_forward_scan_Jsc',
]

# Separate Features and Labels
features = data_1_step.drop(columns=output_labels)
labels = data_1_step[output_labels]

# Drop ID/Metadata columns from features
if 'Ref_ID' in features.columns:
    features = features.drop(columns=['Ref_ID', 'Perovskite_deposition_number_of_deposition_steps', 'Outdoor_average_over_n_number_of_cells'])

# Remove metadata columns if they exist
cols_to_drop = ['Ref_DOI_number','Ref_journal', 'Perovskite_composition_short_form', 'Perovskite_pl_max']
features = features.drop(columns=[c for c in cols_to_drop if c in features.columns], errors='ignore')

# Reset Index globally
features = features.reset_index(drop=True)

print(f"    Initial Features Shape: {features.shape}")

# --- 2. PREPROCESSING & ENCODING ---
print(">>> [STEP 2/5] Preprocessing and One-Hot Encoding...")

# Separate Numerical and Categorical
numerical_features = features.select_dtypes(include=['float64', 'int64'])
categorical_features = features.drop(columns=numerical_features.columns)

# *** FIX: Drop completely empty numerical columns BEFORE imputation ***
# This prevents the shape mismatch error.
numerical_features = numerical_features.dropna(axis=1, how='all')

print(f"    Numerical Features (Cleaned): {numerical_features.shape[1]}")
print(f"    Categorical Features: {categorical_features.shape[1]}")

# Create a map to track which final columns belong to which original feature
feature_map = {} 
final_df_list = []

# --- Process Categorical Features ---
for col in categorical_features.columns:
    encoded = pd.get_dummies(categorical_features[col], prefix=col, prefix_sep='_', drop_first=False)
    feature_map[col] = list(encoded.columns)
    final_df_list.append(encoded)

# --- Process Numerical Features ---
print("    Imputing missing numerical values (Mean Imputation)...")
imputer = SimpleImputer(strategy='mean')
# Now the input shape and output shape will match because we already dropped the 100% empty cols
numerical_data_imputed = imputer.fit_transform(numerical_features)
numerical_features_clean = pd.DataFrame(numerical_data_imputed, columns=numerical_features.columns)

# Map numerical features (1-to-1)
for col in numerical_features_clean.columns:
    feature_map[col] = [col]

final_df_list.append(numerical_features_clean)

# Concatenate all features
X_full = pd.concat(final_df_list, axis=1)

# --- 3. ALIGNMENT WITH TARGET ---
print(">>> [STEP 3/5] Aligning Features with Target...")

target_col = 'JV_reverse_scan_PCE'

# Re-extract y using the index of the PRE-RESET features
original_indices = data_1_step.index
y_full_original = data_1_step.loc[original_indices, target_col]
y_full = y_full_original.reset_index(drop=True)

# Filter NaNs in Target
non_null_indices = y_full.notnull()

X_clean = X_full.loc[non_null_indices]
y_clean = y_full[non_null_indices]

# Handle remaining NaNs in X (if any categorical rows generated NaNs during merge, unlikely but safe)
if X_clean.isnull().sum().sum() > 0:
    print("    WARNING: NaNs still found in X_clean! Filling with 0...")
    X_clean = X_clean.fillna(0)

print(f"    Final X Shape: {X_clean.shape}")
print(f"    Final y Shape: {y_clean.shape}")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# --- 4. MODEL TRAINING ---
print(">>> [STEP 4/5] Training Models...")

models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

results = []

for name, model in models.items():
    print(f"    Training {name}...")
    
    # Fit and Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Get Raw Importances
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    else:
        # Linear Regression: Use absolute coefficients
        raw_importances = np.abs(model.coef_)

    # Aggregate Importances
    aggregated_importance = {}
    
    for original_feat, associated_cols in feature_map.items():
        total_imp = 0
        for col_name in associated_cols:
            if col_name in X_train.columns:
                idx = X_train.columns.get_loc(col_name)
                val = raw_importances[idx]
                
                if isinstance(val, (list, np.ndarray)):
                    total_imp += np.sum(val)
                else:
                    total_imp += val
        
        aggregated_importance[original_feat] = total_imp

    # Find Top Feature
    sorted_importance = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
    top_feature_name = sorted_importance[0][0]
    
    print(f"    --> {name} Finished. R2: {r2:.2f}, Top Feature: {top_feature_name}")

    results.append({
        "Model": name,
        "R2 (Test)": round(r2, 2),
        "RMSE": round(rmse, 2),
        "Primary Top Feature": top_feature_name
    })

# --- 5. OUTPUT ---
print(">>> [STEP 5/5] Saving Results...")

comparison_df = pd.DataFrame(results)

print("\n" + "="*40)
print("FINAL RESULTS SUMMARY")
print("="*40)
# FIX: Use to_string() instead of to_markdown() to avoid 'tabulate' error
print(comparison_df.to_string(index=False))
print("="*40 + "\n")

comparison_df.to_csv("models_results.csv", index=False)
print(">>> Script Complete.")
