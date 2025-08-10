import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class LinearRegression:
    """clean linear regression"""
    
    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.training_history: List[float] = []
        
    def train(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, 
              epochs: int = 1000, quiet: bool = False) -> None:
        """Train the model with optional real-time loss tracking"""
        n_samples, n_features = X.shape
        
        if not quiet:
            print(f"X dimensions: {n_samples} x {n_features}")
            print(f"y dimensions: {len(y)} x 1")
        
        # init weights 
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.training_history = []
        
        # training loop with loss tracking
        for epoch in range(epochs):
            predictions = X @ self.weights + self.bias
            
            loss = np.mean((y - predictions) ** 2)
            self.training_history.append(loss)
            
            errors = y - predictions
            weight_gradients = -(2/n_samples) * (X.T @ errors)
            bias_gradient = -(2/n_samples) * np.sum(errors)
            
            self.weights -= learning_rate * weight_gradients
            self.bias -= learning_rate * bias_gradient
            
            if not quiet and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        return X @ self.weights + self.bias
    
    def get_learning_curve(self) -> List[float]:
        """Return the training loss history"""
        return self.training_history


def normalize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features to have zero mean and unit variance"""
    means = np.mean(X, axis=0)
    std_devs = np.std(X, axis=0)
    # avoid division by zero
    std_devs = np.where(std_devs == 0, 1, std_devs)
    normalized_X = (X - means) / std_devs
    return normalized_X, means, std_devs


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Calculate MSE, MAE, and R²"""
    mse = np.mean((y_true - y_pred) ** 2)
    
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
    
    return mse, mae, r2


def plot_learning_curve(training_history: List[float], title: str = "Learning Curve") -> None:
    """Plot the learning curve in a minimal style"""
    plt.figure(figsize=(8, 5))
    plt.plot(training_history, color='#2E86AB', linewidth=2, alpha=0.8)
    plt.title(title, fontsize=14, color='#2D3748', pad=20)
    plt.xlabel('Epoch', color='#4A5568')
    plt.ylabel('Loss (MSE)', color='#4A5568')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#E2E8F0')
    plt.gca().spines['bottom'].set_color('#E2E8F0')
    
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    title: str = "Actual vs Predicted") -> None:
    """Plot actual vs predicted values"""
    plt.figure(figsize=(8, 6))
    
    # prediction line
    min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 
             '--', color='#E53E3E', alpha=0.7, linewidth=2, label='Perfect Prediction')
    
    plt.scatter(y_true, y_pred, color='#2E86AB', alpha=0.6, s=50, edgecolors='white', linewidth=1)
    
    plt.xlabel('Actual Values', color='#4A5568')
    plt.ylabel('Predicted Values', color='#4A5568')
    plt.title(title, fontsize=14, color='#2D3748', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # styling 
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#E2E8F0')
    plt.gca().spines['bottom'].set_color('#E2E8F0')
    
    plt.tight_layout()
    plt.show()