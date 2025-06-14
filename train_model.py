"""
Boston House Price Prediction Model Training Script

This script trains a Linear Regression model on the Boston Housing dataset
and saves it as a pickle file for use in the Flask application.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Note: The Boston housing dataset was removed from sklearn.datasets
# This is a recreation of the dataset with the same features
def create_boston_housing_data():
    """
    Create a sample Boston housing dataset with the same structure
    as the original dataset for demonstration purposes.
    """
    np.random.seed(42)
    
    # Generate sample data with 506 samples (same as original dataset)
    n_samples = 506
    
    # Feature generation with realistic ranges
    crim = np.random.exponential(3, n_samples)  # Crime rate
    zn = np.random.uniform(0, 100, n_samples)   # Residential land zoned
    indus = np.random.uniform(0, 30, n_samples) # Industrial business
    chas = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # Charles River
    nox = np.random.uniform(0.3, 0.9, n_samples)  # NOx concentration
    rm = np.random.normal(6.3, 0.7, n_samples)     # Average rooms
    age = np.random.uniform(0, 100, n_samples)     # Age of units
    dis = np.random.uniform(1, 12, n_samples)      # Distance to employment
    rad = np.random.randint(1, 25, n_samples)      # Highway access
    tax = np.random.uniform(150, 800, n_samples)   # Property tax
    ptratio = np.random.uniform(12, 22, n_samples) # Pupil-teacher ratio
    b = np.random.uniform(300, 400, n_samples)     # Black population
    lstat = np.random.uniform(1, 40, n_samples)    # Lower status population
    
    # Create target variable with realistic relationship to features
    medv = (
        50 - 0.5 * crim
        + 0.1 * zn
        - 0.2 * indus
        + 5 * chas
        - 30 * nox
        + 8 * rm
        - 0.1 * age
        + 2 * dis
        - 0.3 * rad
        - 0.01 * tax
        - 1.5 * ptratio
        + 0.01 * b
        - 0.8 * lstat
        + np.random.normal(0, 3, n_samples)  # Add noise
    )
    
    # Ensure positive prices
    medv = np.maximum(medv, 5)
    
    # Create feature matrix
    X = np.column_stack([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat])
    
    # Feature names
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
        'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    return X, medv, feature_names

def train_model():
    """
    Train the Linear Regression model and save it as a pickle file.
    """
    print("Loading Boston Housing Dataset...")
    
    # Load the data
    X, y, feature_names = create_boston_housing_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: ${y.min():.1f}K - ${y.max():.1f}K")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train the model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Training RMSE: ${np.sqrt(train_mse):.2f}K")
    print(f"Test RMSE: ${np.sqrt(test_mse):.2f}K")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Feature importance (coefficients)
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE (Coefficients)")
    print("="*50)
    for name, coef in zip(feature_names, model.coef_):
        print(f"{name:>8}: {coef:>8.4f}")
    print(f"{'INTERCEPT':>8}: {model.intercept_:>8.4f}")
    
    # Save the model
    model_filename = 'boston_house_price_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"\nModel saved as '{model_filename}'")
    
    # Test the saved model
    print("\nTesting saved model...")
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    
    # Make a test prediction
    sample_features = X_test[0].reshape(1, -1)
    original_pred = model.predict(sample_features)[0]
    loaded_pred = loaded_model.predict(sample_features)[0]
    actual_price = y_test[0]
    
    print(f"Actual price: ${actual_price:.2f}K")
    print(f"Original model prediction: ${original_pred:.2f}K")
    print(f"Loaded model prediction: ${loaded_pred:.2f}K")
    print(f"Predictions match: {np.allclose(original_pred, loaded_pred)}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("You can now run the Flask application:")
    print("python app.py")
    
    return model, test_r2

if __name__ == "__main__":
    model, accuracy = train_model()