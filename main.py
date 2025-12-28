from data_management import DataManager
from model_training import ModelTrainer
from visualization import Visualizer
from sklearn.model_selection import train_test_split

DATA_FILE = 'Perovskite_FAIR_database_ORIGINAL_all_data.xlsx'  # Update path relative to repo

def run_experiment(name, df):
    print(f"\n--- Running Experiment: {name} ---")
    
    # 1. Prepare Data
    dm = DataManager(None) # Helper instance
    X, y = dm.prepare_features_labels(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Decision Tree
    trainer = ModelTrainer(max_depth=10) # Limit depth to avoid extreme overfitting on small data
    trainer.train(X_train, y_train)
    
    # 3. Evaluate
    preds, r2, mse = trainer.evaluate(X_test, y_test)
    print(f"Results for {name}: R2={r2:.4f}, MSE={mse:.4f}")
    
    # 4. Save Artifacts
    trainer.save_model(f"model_{name}.pkl")
    
    # 5. Visualize
    Visualizer.plot_predictions(y_test, preds, title=f"{name} Deposition")
    Visualizer.plot_feature_importance(trainer.model, X.columns, title=f"{name} Importance")

def main():
    # Load and Preprocess Global Data
    dm = DataManager(DATA_FILE)
    try:
        dm.load_data()
        dm.clean_and_filter_outliers()
        
        # Get Dictionary of datasets {'1_step': df, '2_step': df}
        datasets = dm.get_datasets_by_step()
        
        # Loop through datasets and run training for each
        for name, data in datasets.items():
            if len(data) > 50: # Only run if sufficient data points exist
                run_experiment(name, data)
            else:
                print(f"Skipping {name}: Insufficient data ({len(data)} rows)")
                
    except FileNotFoundError:
        print(f"Error: Could not find {DATA_FILE}. Ensure the file is in the repo.")

if __name__ == "__main__":
    main()
