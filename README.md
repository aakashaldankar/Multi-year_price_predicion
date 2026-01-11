# Hotel Price Prediction - Multi-Year Forecasting

A comprehensive time series forecasting project that predicts hotel prices for February 2020 based on historical data from 2012-2016. The solution implements multiple modeling approaches including machine learning (XGBoost, LightGBM, Random Forest), statistical models (Prophet, SARIMA), and an ensemble method.

## Problem Statement

**Objective:** Based on hotel price data from 2012 to 2016, estimate hotel prices for each day in February 2020.

**Challenge:** This is a long-horizon forecasting problem (approximately 4 years ahead) that requires:
- Learning patterns from limited historical data (4 years of training)
- Extrapolating trends across a significant time gap
- Capturing seasonal patterns, particularly for February (historically the 2nd highest price month)

## Key Findings

- Strong upward trend: 32.6% growth from 2012 to 2015
- Clear seasonality: February averages ~$130 (2nd highest month)
- High data quality: 99.93% completeness (1,476 days of data)
- Strong autocorrelation: 0.87 at 365-day lag confirms yearly seasonality

## Project Structure

```
Multi-year_price_predicion/
│
├── input_data/                          # Original dataset
│   └── Multi-Year Price Data (Aggregate).csv
│
├── data/                                # Processed datasets
│   ├── train/
│   │   ├── train_data.csv              # Training set (2012-mid 2015)
│   │   └── val_data.csv                # Validation set (mid 2015-2016)
│   └── test/
│       └── feb_2020_data.csv           # February 2020 features
│
├── models/                              # Saved trained models
│   └── saved_models/
│       ├── xgboost_*.pkl               # XGBoost model
│       ├── lightgbm_*.pkl              # LightGBM model
│       ├── randomforest_*.pkl          # Random Forest model
│       ├── prophet_model.json          # Prophet model
│       ├── sarima_best_model.joblib    # Best SARIMA model
│       └── sarima_ensemble_models.joblib
│
├── predictions/                         # Output predictions
│   └── feb_2020_predictions_*.csv
│
├    
├── eda_price_prediction.ipynb      # Exploratory Data Analysis
├── feature_engineering.ipynb       # Feature creation
├── machine_learning_models.ipynb   # ML model training
├── prophet_model.ipynb             # Prophet implementation
└── sarima_forecasting.ipynb        # SARIMA implementation
│
├── main.py                              # Gradio web application
├── load_and_predict_models.py          # Model loading utility
├── EDA_report.md                        # Detailed EDA findings
└── README.md                            
```

## Dataset Description

**Source:** `input_data/Multi-Year Price Data (Aggregate).csv`

**Characteristics:**
- Date Range: January 1, 2012 to January 16, 2016
- Total Records: 1,476 daily observations
- Features: Date, Price
- Missing Values: 0
- Missing Dates: 1 (July 13, 2012)

**Price Statistics:**
- Mean: $114.10
- Median: $111.50
- Range: $81 - $281
- Standard Deviation: $20.48

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Data quality assessment
- Temporal pattern analysis (yearly, monthly, weekly)
- Autocorrelation analysis
- Seasonality detection
- Detailed findings in `EDA_report.md`

### 2. Feature Engineering
Created temporal features to capture patterns:
- **Temporal**: Year, Month, Day, DayOfWeek, DayOfYear, Week, Quarter
- **Binary**: IsWeekend, IsMonthStart, IsMonthEnd, IsFebruary
- **Derived**: DaysSinceStart (from 2012-01-01)
- **Cyclical**: DayOfYear_Sin, DayOfYear_Cos, DayOfWeek_Sin, DayOfWeek_Cos

### 3. Models Implemented

#### Machine Learning Models
1. **XGBoost** - Gradient boosting (RMSE: 8.58, MAE: 6.28, R²: 0.1512)
2. **LightGBM** - Light gradient boosting (RMSE: 9.08, MAE: 6.42, R²: 0.0502)
3. **Random Forest** - Ensemble of decision trees (RMSE: 10.15, MAE: 7.90, R²: -0.1875)

#### Statistical Models
4. **Prophet** - Facebook's time series forecasting model
5. **SARIMA** - Seasonal AutoRegressive Integrated Moving Average
   - Best single SARIMA model
   - Weighted ensemble of multiple SARIMA configurations

#### Ensemble
6. **Simple Average Ensemble** - Combines all 6 model predictions

## Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Install Dependencies

```bash
# Create virtual environment
python -m venv price_env

# Activate virtual environment
# On Windows:
price_env\Scripts\activate
# On Linux/Mac:
source price_env/bin/activate

# Install required packages
pip install pandas numpy scikit-learn xgboost lightgbm
pip install prophet statsmodels joblib gradio matplotlib seaborn
```

### Required Libraries
- pandas, numpy - Data manipulation
- scikit-learn - ML utilities and Random Forest
- xgboost - XGBoost model
- lightgbm - LightGBM model
- prophet - Time series forecasting
- statsmodels - SARIMA implementation
- joblib - Model serialization
- gradio - Web interface
- matplotlib, seaborn - Visualization

## Usage

### Option 1: Interactive Web Application (Recommended)

Launch the Gradio web interface for easy predictions:

```bash
python main.py
```

This will start a local web server. Open the provided URL in your browser to:
- Select any date range for predictions
- View predictions from all 6 models plus ensemble
- Download results as CSV

**Features:**
- Date picker interface
- Real-time predictions
- Automatic CSV export
- Model comparison table

### Option 2: Jupyter Notebooks

For detailed analysis and model training:

1. **EDA and Exploration:**
   ```bash
   jupyter notebook eda_price_prediction.ipynb
   ```

2. **Feature Engineering:**
   ```bash
   jupyter notebook feature_engineering.ipynb
   ```

3. **Train ML Models:**
   ```bash
   jupyter notebook machine_learning_models.ipynb
   ```

4. **Train Prophet Model:**
   ```bash
   jupyter notebook prophet_model.ipynb
   ```

5. **Train SARIMA Models:**
   ```bash
   jupyter notebook sarima_forecasting.ipynb
   ```

## Model Details

### XGBoost (Best Performer)
- **Type:** Gradient Boosted Trees
- **Strengths:** Handles non-linear relationships, captures complex feature interactions
- **Performance:** RMSE: 8.58, MAE: 6.28
- **Use Case:** Primary predictor for short to medium-term forecasts

### LightGBM
- **Type:** Gradient Boosted Trees (optimized)
- **Strengths:** Fast training, memory efficient
- **Performance:** RMSE: 9.08, MAE: 6.42
- **Use Case:** Alternative to XGBoost with similar accuracy

### Random Forest
- **Type:** Ensemble of Decision Trees
- **Strengths:** Robust to overfitting, good baseline
- **Performance:** RMSE: 10.15, MAE: 7.90
- **Use Case:** Ensemble diversification

### Prophet
- **Type:** Additive time series model
- **Strengths:** Captures trend and seasonality, handles missing data
- **Features:** Yearly seasonality, trend changepoints
- **Use Case:** Long-horizon forecasting with trend extrapolation

### SARIMA (Best & Ensemble)
- **Type:** Statistical time series model
- **Strengths:** Strong theoretical foundation, captures seasonality
- **Configuration:** Tuned for seasonal patterns
- **Limitation:** Can only predict from 2016-01-17 onwards (trained until 2016-01-16)
- **Use Case:** Statistical complement to ML models

### Ensemble
- **Method:** Simple average of all 6 models
- **Rationale:** Reduces individual model bias, increases robustness
- **Use Case:** Most reliable predictions for production

## Results & Predictions

### February 2020 Predictions
The models predict February 2020 daily prices using patterns learned from 2012-2016 data.

**Expected Range:** Approximately $180-$220 per day
**Methodology:** Combines trend extrapolation (32.6% growth over training period) with seasonal patterns (February is historically 2nd highest month)

### Model Performance on Validation Set
Evaluated on July 2015 - January 2016 data:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| XGBoost | 8.58 | 6.28 | 0.1512 |
| LightGBM | 9.08 | 6.42 | 0.0502 |
| Random Forest | 10.15 | 7.90 | -0.1875 |
| Prophet | 8.64 | 6.65 | 0.14 |
| SARIMA Best | 9.91 | 7.09 | 0.06 |


## File Descriptions

### Python Scripts
- **main.py**: Gradio web application with date picker interface and CSV download

### Notebooks
- **eda_price_prediction.ipynb**: Comprehensive exploratory data analysis with visualizations
- **feature_engineering.ipynb**: Feature creation and train/test split preparation
- **machine_learning_models.ipynb**: Training XGBoost, LightGBM, Random Forest models
- **prophet_model.ipynb**: Facebook Prophet model implementation
- **sarima_forecasting.ipynb**: SARIMA model selection, training, and ensemble creation

### Documentation
- **LICENSE**: Project license information

## Key Insights

### Trend Analysis
- 2012-2015 Growth: 32.6%
- Average Annual Growth: 10.87%
- February-specific growth: 12.7% per year (higher than overall)

### Seasonality
- Peak Months: March ($150), February ($130), April ($127)
- Low Months: September ($102), November ($103), December ($104)
- February Pattern: Consistently 2nd highest month across all years

### Feature Importance
1. **High Importance:** Year, Month, DayOfYear, Quarter
2. **Medium Importance:** Week, Cyclical encodings (sin/cos)
3. **Low Importance:** DayOfWeek, IsWeekend (only 3% variation)

## Challenges Addressed

1. **Long Forecast Horizon:** 4 years ahead with only 4 years of training data
2. **Missing Recent Data:** No 2017-2019 data available for pattern learning
3. **Trend Extrapolation:** Uncertainty in whether growth rate will continue
4. **Outlier Handling:** Preserved real price spikes (e.g., March 2015: $281)

## Future Improvements

1. **Additional Features:**
   - Holiday calendar integration
   - Local events/conferences
   - Economic indicators (inflation, tourism stats)

2. **Advanced Models:**
   - LSTM/GRU neural networks for sequence learning


## Technical Notes

### SARIMA Limitation
SARIMA models can only predict dates from **2016-01-17 onwards** because they were trained on data up to 2016-01-16. For dates before this, SARIMA predictions will show as NaN in the output.

### Feature Alignment
All models use the same 16 temporal features in exact order:
```python
['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Week',
 'Quarter', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
 'IsFebruary', 'DaysSinceStart', 'DayOfYear_Sin',
 'DayOfYear_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos']
```

### Date Reference
- Training Start: 2012-01-01 (used as reference for DaysSinceStart calculation)
- Training End: 2016-01-16
- Target: February 2020 (29 days)

## License

See LICENSE file for details.

---

