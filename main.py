import numpy as np
import pandas as pd

from src.core import LinearRegression, normalize_data, calculate_metrics, plot_learning_curve, plot_predictions
from src.datasets import DatasetPersonalities, generate_demo_data, load_csv_data

def print_welcome():
    """Calm welcome message"""
    print("\n" + "="*50)
    print("🧠 Machine Learning Simulation - Python Edition")
    print("="*50)
    print("Calm linear regression with learning curves")
    print()

def print_results_table(metrics: dict, model: LinearRegression, info: dict):
    """Print results in a clean table format"""
    print("\n Model Results:")
    print("="*40)
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Mean Squared Error':<25} {metrics['mse']:<15.4f}")
    print(f"{'Mean Absolute Error':<25} {metrics['mae']:<15.4f}")
    print(f"{'R-squared':<25} {metrics['r2']:<15.4f}")
    print(f"{'Final Training Loss':<25} {model.training_history[-1]:<15.4f}")
    
    print(f"\n🔧 Model Coefficients:")
    print("="*40)
    for i, (feature, weight) in enumerate(zip(info['features'], model.weights)):
        print(f"{feature:<20} {weight:>15.4f}")
    print(f"{'Bias':<20} {model.bias:>15.4f}")


def interactive_prediction_loop(model: LinearRegression, means: np.ndarray, 
                              std_devs: np.ndarray, features: list):
    """Interactive prediction loop with gentle UX"""
    print(f"\n🎯 Interactive Predictions")
    print("="*40)
    print(f"Enter {len(features)} values separated by commas")
    print("Features:", ", ".join(features))
    print("Type 'q' to quit\n")
    
    while True:
        try:
            user_input = input("📝 Enter values: ").strip()
            
            if user_input.lower() == 'q':
                break
                
            # Parse input
            values = [float(x.strip()) for x in user_input.split(',')]
            
            if len(values) != len(features):
                print(f"❌ Expected {len(features)} values, got {len(values)}")
                continue
            
            # Normalize and predict
            normalized_values = (np.array(values) - means) / std_devs
            prediction = model.predict(normalized_values.reshape(1, -1))[0]
            
            print(f"🎯 Predicted value: {prediction:.4f}\n")
            
        except ValueError as e:
            print(f"❌ Error parsing input: {e}")
        except KeyboardInterrupt:
            break


def main():
    """Main application flow"""
    print_welcome()
    
    # Step 1: Data Selection
    print("📂 Data Source Selection:")
    print("-" * 30)
    filename = input("Enter CSV filename (or press Enter for dataset menu): ").strip()
    
    if filename:
        try:
            X, y, info = load_csv_data(filename)
            print(f"✅ Loaded {filename} successfully")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            print("Falling back to demo data...")
            X, y, info = generate_demo_data()
    else:
        # Show dataset personalities
        DatasetPersonalities.show_dataset_menu()
        choice = input("\nChoose dataset (1-4): ").strip()
        
        datasets = DatasetPersonalities.get_dataset_menu()
        if choice in datasets:
            X, y, info = datasets[choice]()
        else:
            print("Using demo data...")
            X, y, info = generate_demo_data()
    
    print(f"\n🎭 Dataset: {info['name']}")
    print(f"📝 {info['description']}")
    print(f"🎚️  Difficulty: {info['difficulty']}")
    print(f"📊 Shape: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Step 2: Data Preprocessing
    print(f"\n🔄 Preprocessing data...")
    normalized_X, means, std_devs = normalize_data(X)
    
    # Step 3: Model Training
    print(f"\n🚀 Training model...")
    model = LinearRegression()
    model.train(normalized_X, y, learning_rate=0.01, epochs=1000)
    
    # Step 4: Evaluation
    predictions = model.predict(normalized_X)
    mse, mae, r2 = calculate_metrics(y, predictions)
    
    metrics = {'mse': mse, 'mae': mae, 'r2': r2}
    print_results_table(metrics, model, info)
    
    # Step 5: Visualizations
    print(f"\n📈 Generating visualizations...")
    plot_learning_curve(model.training_history, 
                       f"Learning Curve - {info['name']}")
    
    plot_predictions(y, predictions, 
                    f"Predictions - {info['name']}")
    
    # Step 6: Interactive Predictions
    interactive_prediction_loop(model, means, std_devs, info['features'])
    
    # Gentle goodbye
    print(f"\n✨ Thanks for using ML Simulation!")
    print("Stay curious! 🧠💙")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your data and try again.")