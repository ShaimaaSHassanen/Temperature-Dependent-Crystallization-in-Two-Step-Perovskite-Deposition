
### 1. Install Dependencies
pip install -r requirements.txt

### 2. Place Your Data
Ensure your dataset file (`Perovskite_FAIR_database_ORIGINAL_all_data.xlsx`) is in the root directory of this project.

### 3. Execute the Pipeline

Run the main script:

```bash
python main.py

```


## 📊 Outputs

After running the pipeline, the following files will be generated:

* **Models**: `model_1_step.pkl`, `model_2_step.pkl` (Saved using Joblib)
* **Figures**:
* `1_step_predictions.png`: Visualizes how well the model predicts PCE for 1-step deposition.
* `2_step_predictions.png`: Visualizes predictions for 2-step deposition.
* `*_feature_importance.png`: Bar charts showing which chemical/process features impacted efficiency the most.


