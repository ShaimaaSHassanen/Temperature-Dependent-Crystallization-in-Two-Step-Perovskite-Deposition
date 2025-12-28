import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_predictions(y_true, y_pred, title="Model Performance"):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, color='green')
        
        # Plot diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        plt.xlabel('Actual PCE')
        plt.ylabel('Predicted PCE')
        plt.title(f'{title}: Actual vs Predicted')
        
        # Save figure with title in filename
        clean_title = title.replace(" ", "_").lower()
        plt.savefig(f"{clean_title}_predictions.png")
        # plt.show() # Commented out for server/GitHub Actions execution

    @staticmethod
    def plot_feature_importance(model, feature_names, title="Feature Importance"):
        """Plots feature importance from the decision tree."""
        if not hasattr(model, 'feature_importances_'):
            return

        importance = model.feature_importances_
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
        df_imp = df_imp.sort_values(by='importance', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=df_imp, palette='viridis')
        plt.title(f'{title}: Top 10 Features')
        
        clean_title = title.replace(" ", "_").lower()
        plt.savefig(f"{clean_title}_feature_importance.png")
