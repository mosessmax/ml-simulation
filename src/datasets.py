import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class DatasetPersonalities:
    """Different dataset personalities for testing the model"""
    
    @staticmethod
    def clean_linear(n_samples: int = 100, n_features: int = 3, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Clean linear relationship - the well-behaved dataset"""
        np.random.seed(42)  # For reproducibility
        
        X = np.random.uniform(-5, 5, (n_samples, n_features))
        # Simple linear relationship: y = sum of features + small noise
        true_weights = np.array([2.5, -1.8, 3.2][:n_features])
        y = X @ true_weights + np.random.normal(0, noise_level, n_samples)
        
        features = [f"Feature_{i+1}" for i in range(n_features)]
        
        info = {
            "name": "Clean Linear",
            "description": "Well-behaved linear relationship with minimal noise",
            "features": features,
            "true_weights": true_weights,
            "noise_level": noise_level,
            "difficulty": "Easy"
        }
        
        return X, y, info
    
    @staticmethod
    def noisy_sales(n_samples: int = 150) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Sales data with realistic noise - the realistic dataset"""
        np.random.seed(123)
        
        # Marketing spend, seasonality, competition
        marketing_spend = np.random.uniform(1000, 10000, n_samples)
        seasonality = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 500
        competition = np.random.uniform(0.5, 2.0, n_samples)
        
        X = np.column_stack([marketing_spend, seasonality, competition])
        
        # Sales = 0.8 * marketing + seasonality effect - competition penalty + noise
        y = (0.8 * marketing_spend + 
             seasonality * 1.2 - 
             competition * 800 + 
             np.random.normal(0, 300, n_samples))
        
        info = {
            "name": "Noisy Sales",
            "description": "Sales data with marketing spend, seasonality, and competition",
            "features": ["Marketing_Spend", "Seasonality", "Competition"],
            "noise_level": "Moderate",
            "difficulty": "Medium"
        }
        
        return X, y, info
    
    @staticmethod
    def housing_simple(n_samples: int = 120) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Simple housing price model - the practical dataset"""
        np.random.seed(456)
        
        sqft = np.random.uniform(800, 3000, n_samples)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05])
        age = np.random.uniform(0, 50, n_samples)
        
        X = np.column_stack([sqft, bedrooms, age])
        
        # Price = base + sqft effect + bedroom premium - age depreciation + noise
        y = (150000 + 
             sqft * 120 + 
             bedrooms * 8000 - 
             age * 1200 + 
             np.random.normal(0, 15000, n_samples))
        
        info = {
            "name": "Housing Simple",
            "description": "House prices based on size, bedrooms, and age",
            "features": ["Square_Feet", "Bedrooms", "Age_Years"],
            "noise_level": "Realistic",
            "difficulty": "Medium "
        }
        
        return X, y, info
    
    @staticmethod
    def polynomial_disguised(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Polynomial relationship disguised as linear - the tricky dataset"""
        np.random.seed(789)
        
        x1 = np.random.uniform(-3, 3, n_samples)
        x2 = np.random.uniform(-2, 2, n_samples)
        
        # a polynomial relationship that linear regression will struggle with
        X = np.column_stack([x1, x2])
        y = (2 * x1**2 + 1.5 * x2**2 + x1 * x2 + 
             np.random.normal(0, 0.5, n_samples))
        
        info = {
            "name": "Polynomial Disguised",
            "description": "Hidden polynomial relationship - linear regression will struggle",
            "features": ["X1", "X2"],
            "noise_level": "Low",
            "difficulty": "Hard (Non-linear!)"
        }
        
        return X, y, info
    
    @staticmethod
    def get_dataset_menu() -> Dict[str, callable]:
        """Return a menu of available datasets"""
        return {
            "1": DatasetPersonalities.clean_linear,
            "2": DatasetPersonalities.noisy_sales,
            "3": DatasetPersonalities.housing_simple,
            "4": DatasetPersonalities.polynomial_disguised
        }
    
    @staticmethod
    def show_dataset_menu():
        """Display the dataset menu"""
        print("\n📊 Dataset Personalities:")
        print("========================")
        print("1. Clean Linear      - Well-behaved, minimal noise")
        print("2. Noisy Sales       - Realistic business data")
        print("3. Housing Simple    - Practical real estate")
        print("4. Polynomial Disguised - Tricky non-linear")
        print("\nPress Enter for demo data (Clean Linear)")


def load_csv_data(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load data from CSV file - keeping the original functionality"""
    try:
        df = pd.read_csv(filename)
        
        # Assume last column is target, rest are features
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        features = df.columns[:-1].tolist()
        
        info = {
            "name": f"CSV: {filename}",
            "description": f"Loaded from {filename}",
            "features": features,
            "difficulty": "Unknown"
        }
        
        return X, y, info
        
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")


def generate_demo_data(n_samples: int = 100, n_features: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate demo data"""
    return DatasetPersonalities.clean_linear(n_samples, n_features, noise_level=0.5)