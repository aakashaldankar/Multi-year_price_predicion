import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# # Ensure directories exist
# os.makedirs('models/saved_models', exist_ok=True)
# os.makedirs('predictions', exist_ok=True)

# Global feature columns
FEATURE_COLS = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Week',
                'Quarter', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
                'IsFebruary', 'DaysSinceStart', 'DayOfYear_Sin',
                'DayOfYear_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos']


def get_param_grids():
    """
    Returns focused parameter grids for hyperparameter tuning.
    Grid sizes: XGBoost=81, LightGBM=108, RandomForest=72 combinations.
    """
    param_grid_xgb = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0]
    }

    param_grid_lgbm = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31],
        'min_child_samples': [10, 20]
    }

    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    return {
        'XGBoost': param_grid_xgb,
        'LightGBM': param_grid_lgbm,
        'RandomForest': param_grid_rf
    }


def create_model_instances():
    """
    Creates and returns dictionary of model instances with default parameters.
    All models set with random_state=42 for reproducibility.
    """
    models = {
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
        'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1)
    }
    return models


def train_model_with_gridsearch(model, param_grid, X_train, y_train, cv_splits=3):
    """
    Trains a model using GridSearchCV with TimeSeriesSplit.

    Args:
        model: Scikit-learn compatible model instance
        param_grid: Dictionary of hyperparameters to search
        X_train: Training features
        y_train: Training target
        cv_splits: Number of time-series splits for cross-validation

    Returns:
        best_estimator: Best model found by GridSearchCV
        best_params: Best hyperparameters
        best_cv_score: Best cross-validation score (negative RMSE)
    """
    # Create TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',  # RMSE as primary metric
        cv=tscv,
        n_jobs=-1,  # Use all available CPU cores
        verbose=1,
        return_train_score=False  # Save memory
    )

    # Fit GridSearchCV
    print(f"Fitting {len(param_grid[list(param_grid.keys())[0]]) * np.prod([len(v) for v in param_grid.values()]) // len(param_grid[list(param_grid.keys())[0]])} combinations...")
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_val, y_val, model_name):
    """
    Evaluates model on validation set and returns metrics.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        model_name: Name of the model for display

    Returns:
        Dictionary containing RMSE, MAE, and R² metrics
    """
    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    # Print results
    print(f"\n{model_name} Validation Results:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  R²:   {r2:.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def save_model(model, metrics, model_name, save_dir='models/saved_models'):
    """
    Saves trained model and metadata to disk.

    Args:
        model: Trained model to save
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
        save_dir: Directory to save model files

    Returns:
        Path to saved model file
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename with metrics
    filename = f"{model_name.lower()}_rmse_{metrics['rmse']:.2f}_mae_{metrics['mae']:.2f}_r2_{metrics['r2']:.4f}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)

    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata_filename = filename.replace('.pkl', '_metadata.txt')
    metadata_filepath = os.path.join(save_dir, metadata_filename)

    with open(metadata_filepath, 'w') as f:
        f.write(f"Model Training Metadata\n")
        f.write(f"={'='*60}\n\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Training Timestamp: {timestamp}\n")
        f.write(f"Model File: {filename}\n\n")

        f.write(f"Evaluation Metrics (Validation Set)\n")
        f.write(f"{'-'*60}\n")
        f.write(f"RMSE: ${metrics['rmse']:.2f}\n")
        f.write(f"MAE:  ${metrics['mae']:.2f}\n")
        f.write(f"R²:   {metrics['r2']:.4f}\n\n")

        f.write(f"Model Hyperparameters\n")
        f.write(f"{'-'*60}\n")
        for param, value in model.get_params().items():
            f.write(f"{param}: {value}\n")

        f.write(f"\nFeatures Used ({len(FEATURE_COLS)} features)\n")
        f.write(f"{'-'*60}\n")
        for i, feature in enumerate(FEATURE_COLS, 1):
            f.write(f"{i:2d}. {feature}\n")

    return filepath


def main():
    """
    Main function to execute the model training pipeline.
    """
    print("="*80)
    print("TIME-SERIES PRICE PREDICTION MODEL TRAINING PIPELINE")
    print("="*80)
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv('data/train/train_data.csv')
    val_df = pd.read_csv('data/train/val_data.csv')
    feb_2020_df = pd.read_csv('data/test/feb_2020_data.csv')

    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples (Feb 2020): {len(feb_2020_df)}")

    # Prepare features and targets
    X_train = train_df[FEATURE_COLS]
    y_train = train_df['Price']
    X_val = val_df[FEATURE_COLS]
    y_val = val_df['Price']
    X_feb_2020 = feb_2020_df[FEATURE_COLS]

    print(f"  Features: {len(FEATURE_COLS)}")

    # -------------------------------------------------------------------------
    # 2. Initialize Models and Parameter Grids
    # -------------------------------------------------------------------------
    print("\n[2/5] Initializing models and parameter grids...")
    param_grids = get_param_grids()
    models = create_model_instances()

    for model_name in models.keys():
        grid = param_grids[model_name]
        n_combinations = np.prod([len(v) for v in grid.values()])
        print(f"  {model_name}: {n_combinations} combinations to search")

    # -------------------------------------------------------------------------
    # 3. Train Models with GridSearchCV
    # -------------------------------------------------------------------------
    print("\n[3/5] Training models with GridSearchCV (TimeSeriesSplit, 3 folds)...")
    print("="*80)

    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Training {model_name}...")
        print(f"{'='*80}")

        # Hyperparameter tuning with GridSearchCV
        best_model, best_params, best_cv_score = train_model_with_gridsearch(
            model=model,
            param_grid=param_grids[model_name],
            X_train=X_train,
            y_train=y_train,
            cv_splits=3
        )

        print(f"\nBest CV RMSE: ${-best_cv_score:.2f}")
        print(f"Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        # Evaluate on validation set
        metrics = evaluate_model(best_model, X_val, y_val, model_name)

        # Store results
        results[model_name] = {
            'model': best_model,
            'params': best_params,
            'cv_score': -best_cv_score,
            'metrics': metrics
        }

    # -------------------------------------------------------------------------
    # 4. Compare Models and Select Best
    # -------------------------------------------------------------------------
    print("\n[4/5] Comparing models...")
    print("\n" + "="*80)
    print("MODEL COMPARISON (Validation Set)")
    print("="*80)
    print(f"{'Model':<20} | {'RMSE':>10} | {'MAE':>10} | {'R²':>10}")
    print("-"*80)

    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:<20} | ${metrics['rmse']:>9.2f} | "
              f"${metrics['mae']:>9.2f} | {metrics['r2']:>10.4f}")

    # Select best model based on RMSE
    best_model_name = min(results.keys(),
                          key=lambda k: results[k]['metrics']['rmse'])
    best_result = results[best_model_name]

    print("\n" + "="*80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Validation RMSE: ${best_result['metrics']['rmse']:.2f}")
    print("="*80)

    # -------------------------------------------------------------------------
    # 5. Save All Models
    # -------------------------------------------------------------------------
    print("\n[5/5] Saving all trained models...")

    saved_paths = {}
    for model_name, result in results.items():
        saved_path = save_model(
            model=result['model'],
            metrics=result['metrics'],
            model_name=model_name
        )
        saved_paths[model_name] = saved_path

        marker = " [BEST]" if model_name == best_model_name else ""
        print(f"  {model_name}{marker}: {os.path.basename(saved_path)}")

    # -------------------------------------------------------------------------
    # 6. Generate February 2020 Predictions
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("GENERATING FEBRUARY 2020 PREDICTIONS")
    print("="*80)

    # Make predictions with best model
    feb_2020_predictions = best_result['model'].predict(X_feb_2020)

    # Add to dataframe
    feb_2020_df['Predicted_Price'] = feb_2020_predictions

    # Save predictions
    pred_filename = f'predictions/feb_2020_predictions_{best_model_name.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    feb_2020_df[['Date', 'Predicted_Price']].to_csv(pred_filename, index=False)

    print(f"\nPredictions saved to: {pred_filename}")
    print(f"\nSample predictions (first 10 days of February 2020):")
    print("-"*80)
    print(feb_2020_df[['Date', 'Predicted_Price']].head(10).to_string(index=False))

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  - Best Model: {best_model_name}")
    print(f"  - Validation RMSE: ${best_result['metrics']['rmse']:.2f}")
    print(f"  - Models saved: {len(saved_paths)}")
    print(f"  - Predictions generated: {len(feb_2020_predictions)} days")
    print("\n")


if __name__ == "__main__":
    main()
