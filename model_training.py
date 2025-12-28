import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, max_depth=None, random_state=42):
        # Using Decision Tree Regressor instead of Random Forest
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return predictions, r2, mse

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
