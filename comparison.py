import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- Your Data Split (Unchanged) ---
# Assuming 'one_hot_features' and 'y' are already defined
X_train, X_test, y_train, y_test = train_test_split(one_hot_features, y, test_size=0.2, random_state=42)

# --- Define Models ---
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

results = []

# --- Train and Evaluate Loop ---
for name, model in models.items():
    # 1. Fit the model
    model.fit(X_train, y_train)
    
    # 2. Predict on test set
    y_pred = model.predict(X_test)
    
    # 3. Calculate Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # 4. Extract Primary Top Feature
    # Tree models use .feature_importances_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argmax(importances)
        
    # Linear models use coefficients (.coef_)
    # We take the absolute value because a large negative coefficient 
    # is just as "important" as a large positive one.
    else:
        importances = np.abs(model.coef_)
        top_idx = np.argmax(importances)
    
    top_feature_name = X_train.columns[top_idx]
    
    # Append to results list
    results.append({
        "Model": name,
        "R2 (Test)": round(r2, 2),
        "RMSE": round(rmse, 2),
        "Primary Top Feature": top_feature_name
    })

# --- Display Comparison Table ---
comparison_df = pd.DataFrame(results)
print(comparison_df.to_markdown(index=False))
