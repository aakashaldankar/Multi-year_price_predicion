import gradio as gr
import pandas as pd
import numpy as np
import joblib
import json
import os
import tempfile
from datetime import datetime
from prophet.serialize import model_from_json
import warnings
warnings.filterwarnings('ignore')

# Feature columns in exact order as used in training
FEATURE_COLS = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Week',
                'Quarter', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
                'IsFebruary', 'DaysSinceStart', 'DayOfYear_Sin',
                'DayOfYear_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos']

# Reference start date for DaysSinceStart calculation
ORIGINAL_START_DATE = datetime(2012, 1, 1)

# SARIMA training end date (predictions start from next day)
SARIMA_TRAINING_END = datetime(2016, 1, 16)
SARIMA_FORECAST_START = datetime(2016, 1, 17)

# Model paths
MODEL_PATHS = {
    'xgboost': 'models/saved_models/xgboost_rmse_8.58_mae_6.28_r2_0.1512_20260111_011434.pkl',
    'lightgbm': 'models/saved_models/lightgbm_rmse_9.08_mae_6.42_r2_0.0502_20260111_011434.pkl',
    'randomforest': 'models/saved_models/randomforest_rmse_10.15_mae_7.90_r2_-0.1875_20260111_011434.pkl',
    'prophet': 'models/saved_models/prophet_model.json',
    'sarima_best': 'models/saved_models/sarima_best_model.joblib',
    'sarima_ensemble': 'models/saved_models/sarima_ensemble_models.joblib'
}


def create_features_for_date_range(start_date, end_date):
    """
    Create temporal features for a given date range.

    Args:
        start_date: Start date (datetime object or string)
        end_date: End date (datetime object or string)

    Returns:
        DataFrame with Date column and all temporal features
    """
    # Convert to datetime if strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'Date': date_range})

    # Basic temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter

    # Binary features
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['IsFebruary'] = (df['Month'] == 2).astype(int)

    # Days since start (from original training data start date)
    df['DaysSinceStart'] = (df['Date'] - ORIGINAL_START_DATE).dt.days

    # Cyclical encodings
    df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    return df


def load_models():
    """
    Load all trained models.

    Returns:
        Dictionary containing all loaded models
    """
    print("Loading models...")
    models = {}

    # Load ML models (XGBoost, LightGBM, RandomForest)
    models['xgboost'] = joblib.load(MODEL_PATHS['xgboost'])
    print("[OK] XGBoost loaded")

    models['lightgbm'] = joblib.load(MODEL_PATHS['lightgbm'])
    print("[OK] LightGBM loaded")

    models['randomforest'] = joblib.load(MODEL_PATHS['randomforest'])
    print("[OK] RandomForest loaded")

    # Load Prophet model
    with open(MODEL_PATHS['prophet'], 'r') as f:
        models['prophet'] = model_from_json(json.load(f))
    print("[OK] Prophet loaded")

    # Load SARIMA models
    models['sarima_best'] = joblib.load(MODEL_PATHS['sarima_best'])
    print("[OK] SARIMA Best Model loaded")

    models['sarima_ensemble'] = joblib.load(MODEL_PATHS['sarima_ensemble'])
    print("[OK] SARIMA Ensemble loaded")

    print("All models loaded successfully!\n")
    return models


# Load models at module level (once at startup)
MODELS = load_models()


def predict_prices(start_date_str, end_date_str):
    """
    Generate price predictions for a given date range using all models.

    Args:
        start_date_str: Start date as string (YYYY-MM-DD)
        end_date_str: End date as string (YYYY-MM-DD)

    Returns:
        DataFrame with predictions from all models
    """
    try:
        # Handle gr.DateTime component which sends Unix timestamps (seconds since epoch)
        if isinstance(start_date_str, (int, float)):
            # Convert from Unix timestamp using milliseconds (Gradio sends in ms)
            # Then get the date portion only
            dt = datetime.fromtimestamp(start_date_str)
            start_date = pd.Timestamp(dt.year, dt.month, dt.day)
        elif isinstance(start_date_str, datetime):
            start_date = pd.Timestamp(start_date_str.year, start_date_str.month, start_date_str.day)
        else:
            start_date = pd.to_datetime(start_date_str)

        if isinstance(end_date_str, (int, float)):
            # Convert from Unix timestamp using milliseconds (Gradio sends in ms)
            # Then get the date portion only
            dt = datetime.fromtimestamp(end_date_str)
            end_date = pd.Timestamp(dt.year, dt.month, dt.day)
        elif isinstance(end_date_str, datetime):
            end_date = pd.Timestamp(end_date_str.year, end_date_str.month, end_date_str.day)
        else:
            end_date = pd.to_datetime(end_date_str)

        # Validate dates
        if start_date > end_date:
            return pd.DataFrame({'Error': ['Start date must be before end date']})

        # Generate features
        features_df = create_features_for_date_range(start_date, end_date)

        # Initialize results DataFrame
        results = pd.DataFrame()
        results['Date'] = features_df['Date'].dt.strftime('%Y-%m-%d')

        # Get features for ML models (exclude Date column)
        X = features_df[FEATURE_COLS]

        # Predictions from ML models (round to 1 decimal)
        results['XGBoost'] = pd.Series(MODELS['xgboost'].predict(X)).round(1)
        results['LightGBM'] = pd.Series(MODELS['lightgbm'].predict(X)).round(1)
        results['RandomForest'] = pd.Series(MODELS['randomforest'].predict(X)).round(1)

        # Prophet predictions
        prophet_df = pd.DataFrame({'ds': features_df['Date']})
        prophet_forecast = MODELS['prophet'].predict(prophet_df)
        results['Prophet'] = pd.Series(prophet_forecast['yhat'].values).round(1)

        # SARIMA predictions
        if start_date < SARIMA_FORECAST_START:
            # Handle dates before SARIMA training end
            warning_msg = f"⚠️ Warning: SARIMA models cannot predict dates before {SARIMA_FORECAST_START.date()}. SARIMA columns will show NaN for those dates."

            # Create NaN arrays for SARIMA
            sarima_best_preds = np.full(len(features_df), np.nan)
            sarima_ensemble_preds = np.full(len(features_df), np.nan)

            # If some dates are after SARIMA_FORECAST_START, predict for those
            if end_date >= SARIMA_FORECAST_START:
                # Calculate indices for valid prediction range
                valid_start_idx = max(0, (SARIMA_FORECAST_START - start_date).days)
                steps_needed = (end_date - SARIMA_FORECAST_START).days + 1

                # SARIMA Best Model
                forecast_best = MODELS['sarima_best'].forecast(steps=steps_needed)
                sarima_best_preds[valid_start_idx:] = forecast_best.values

                # SARIMA Ensemble
                ensemble_forecasts = []
                for model_info in MODELS['sarima_ensemble']['models']:
                    pred = model_info['model'].forecast(steps=steps_needed)
                    ensemble_forecasts.append(pred.values * model_info['weight'])
                sarima_ensemble_preds[valid_start_idx:] = np.sum(ensemble_forecasts, axis=0)

            results['SARIMA_Best'] = pd.Series(sarima_best_preds).round(1)
            results['SARIMA_Ensemble'] = pd.Series(sarima_ensemble_preds).round(1)
            print(warning_msg)
        else:
            # All dates are after SARIMA training - normal prediction
            # Calculate steps and indices
            steps_needed = (end_date - SARIMA_FORECAST_START).days + 1
            start_idx = (start_date - SARIMA_FORECAST_START).days
            end_idx = (end_date - SARIMA_FORECAST_START).days + 1

            # SARIMA Best Model
            forecast_best = MODELS['sarima_best'].forecast(steps=steps_needed)
            results['SARIMA_Best'] = pd.Series(forecast_best[start_idx:end_idx].values).round(1)

            # SARIMA Ensemble
            ensemble_forecasts = []
            for model_info in MODELS['sarima_ensemble']['models']:
                pred = model_info['model'].forecast(steps=steps_needed)
                ensemble_forecasts.append(pred.values * model_info['weight'])

            final_ensemble_forecast = np.sum(ensemble_forecasts, axis=0)
            results['SARIMA_Ensemble'] = pd.Series(final_ensemble_forecast[start_idx:end_idx]).round(1)

        # Calculate Ensemble (simple average of all models)
        model_columns = ['XGBoost', 'LightGBM', 'RandomForest', 'Prophet', 'SARIMA_Best', 'SARIMA_Ensemble']
        results['Ensemble'] = results[model_columns].mean(axis=1).round(1)

        return results

    except Exception as e:
        return pd.DataFrame({'Error': [f'Prediction failed: {str(e)}']})


def predict_and_save(start_date_str, end_date_str):
    """
    Generate predictions and save to CSV file for download.

    Args:
        start_date_str: Start date as string or datetime
        end_date_str: End date as string or datetime

    Returns:
        Tuple of (dataframe, csv_file_path)
    """
    # Convert datetime objects/timestamps to strings if needed
    if start_date_str is None or end_date_str is None:
        return pd.DataFrame({'Error': ['Please select both start and end dates']}), None

    # Handle gr.DateTime component which sends Unix timestamps (seconds since epoch)
    if isinstance(start_date_str, (int, float)):
        # Convert Unix timestamp to local datetime and extract date only
        dt = datetime.fromtimestamp(start_date_str)
        start_date_str = f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    elif isinstance(start_date_str, datetime):
        start_date_str = start_date_str.strftime('%Y-%m-%d')
    elif hasattr(start_date_str, 'strftime'):
        start_date_str = start_date_str.strftime('%Y-%m-%d')

    if isinstance(end_date_str, (int, float)):
        # Convert Unix timestamp to local datetime and extract date only
        dt = datetime.fromtimestamp(end_date_str)
        end_date_str = f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    elif isinstance(end_date_str, datetime):
        end_date_str = end_date_str.strftime('%Y-%m-%d')
    elif hasattr(end_date_str, 'strftime'):
        end_date_str = end_date_str.strftime('%Y-%m-%d')

    # Get predictions
    results_df = predict_prices(start_date_str, end_date_str)

    # Save to temporary CSV file
    if 'Error' not in results_df.columns:
        # Create predictions directory if it doesn't exist
        os.makedirs('predictions', exist_ok=True)

        # Generate filename with date range
        filename = f"predictions_{start_date_str}_to_{end_date_str}.csv"
        filepath = os.path.join('predictions', filename)

        # Save DataFrame to CSV
        results_df.to_csv(filepath, index=False)

        return results_df, filepath
    else:
        return results_df, None


def create_gradio_interface():
    """
    Create and configure the Gradio interface.

    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Multi-Model Hotel Price Prediction") as demo:
        gr.Markdown("# Hotel Price Prediction App")
        gr.Markdown("Select a date range to get price predictions from multiple models (XGBoost, LightGBM, RandomForest, Prophet, SARIMA) with an ensemble average.")

        with gr.Row():
            start_date = gr.DateTime(
                label="Start Date",
                value=datetime(2020, 2, 1),
                include_time=False,
                info="Select start date"
            )
            end_date = gr.DateTime(
                label="End Date",
                value=datetime(2020, 2, 29),
                include_time=False,
                info="Select end date"
            )

        predict_button = gr.Button("Predict Prices", variant="primary", size="lg")

        gr.Markdown("### Predictions")
        output_df = gr.Dataframe(
            label="Price Predictions",
            interactive=False,
            wrap=True
        )

        with gr.Row():
            download_button = gr.DownloadButton(
                label="Download Predictions as CSV",
                variant="secondary",
                visible=True
            )

        gr.Markdown("""
        **Model Information:**
        - **XGBoost, LightGBM, RandomForest**: Tree-based ML models trained on temporal features
        - **Prophet**: Facebook's time series forecasting model
        - **SARIMA Best**: Best performing SARIMA model
        - **SARIMA Ensemble**: Weighted ensemble of multiple SARIMA models
        - **Ensemble**: Simple average of all 6 model predictions

        **Note:** SARIMA models can only predict dates from 2016-01-17 onwards (trained until 2016-01-16)
        """)

        # Connect button to prediction function
        predict_button.click(
            fn=predict_and_save,
            inputs=[start_date, end_date],
            outputs=[output_df, download_button]
        )

    return demo


if __name__ == "__main__":
    # Create and launch the Gradio app
    demo = create_gradio_interface()
    demo.launch(share=False, show_error=True)
