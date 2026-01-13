# Multi-Year Hotel Price Prediction

**Live Demo:** [Try it on HuggingFace Spaces](https://huggingface.co/spaces/aakashaldankar/Multi-year_price_predicion)

Predicting daily hotel prices 4+ years into the future using ensemble machine learning and time series forecasting.

---

## The Problem

Given 4 years of historical hotel prices (January 2012 - January 2016), can we predict what prices will be for February 2020?

This presents a challenging **4+ year forecasting horizon** with only temporal patterns to rely on - no external data like events, tourism statistics, or economic indicators. The dataset contains **1,477 daily price records**, and the task requires pure time-based feature engineering.

## Approach & Thought Process

### 1. Exploration
Initial analysis revealed critical insights:
- **Strong February seasonality**: February prices are 27.4% above annual average ($129.70 vs $114.10)
- **Upward trend**: Consistent 10.87% average annual growth from 2012-2015
- **High autocorrelation**: Strong yearly cycles (lag-365 correlation: 0.87)

### 2. Feature Engineering
Created 16 temporal features without using historical prices:
- Basic features: Year, Month, Day, DayOfWeek, DayOfYear
- Cyclical encodings: Sin/Cos transformations to preserve circular nature of time
- Binary indicators: IsFebruary, IsWeekend, IsMonthStart/End
- Trend proxy: DaysSinceStart (days from 2012-01-01)

### 3. Modeling Strategy
Combined diverse approaches to capture different patterns:
- **Machine Learning**: XGBoost, LightGBM, RandomForest (tree-based methods)
- **Time Series**: Prophet (Facebook's forecasting tool), SARIMA models (statistical approach)

### 4. Ensemble
Averaged predictions from all 6 models for robust, balanced forecasts that reduce individual model biases.

## What I Did

1. **Exploratory Data Analysis** - Identified seasonality, trends, and outliers
2. **Feature Engineering** - Built temporal and cyclical features for long-horizon forecasting
3. **Model Training** - Trained 6 diverse models with hyperparameter tuning via GridSearchCV
4. **Validation** - Used temporal train/validation splits (no data leakage)
5. **Deployment** - Built interactive Gradio web application
6. **Production** - Deployed on HuggingFace Spaces for public access

## Results

### Model Performance (Validation Set)

| Model | RMSE | MAE | R² | Status |
|-------|------|-----|-----|--------|
| **XGBoost** | **$6.85** | **$5.21** | **0.46** | **Best** |
| LightGBM | $7.88 | $5.93 | 0.28 | Strong |
| Prophet | $8.64 | $6.65 | 0.14 | Good |
| SARIMA | $9.36 | $7.08 | - | Moderate |

**Key Finding**: XGBoost achieved the best performance with RMSE of $6.85 and R² of 0.46, explaining 46% of price variance.

**Final Prediction**: Ensemble forecast for February 2020 averages **~$237/day** (range: $196-$284 across models)

### Key Insights
- February's 27.4% premium was successfully captured via the `IsFebruary` feature
- Cyclical encodings prevented artificial discontinuities (e.g., Dec 31 vs Jan 1)
- Ensemble approach balanced optimistic SARIMA predictions with conservative ML forecasts
- Moderate R² scores (0.14-0.46) reflect the inherent challenge of 4-year extrapolation without external regressors

## How to Use

### Try the Live Demo
Visit the interactive web app: **[HuggingFace Space](https://huggingface.co/spaces/aakashaldankar/Multi-year_price_predicion)**

- Select any date range
- View predictions from all 6 models + ensemble
- Download results as CSV

### Run Locally

```bash
# Clone the repository
git clone https://github.com/aakashaldankar/Multi-year_price_predicion
cd Multi-year_price_predicion

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio app
python main.py
```

The app will open at `http://localhost:7860`

### Explore the Analysis

Jupyter notebooks document the complete workflow:

1. **`eda_price_prediction.ipynb`** - Data exploration and pattern discovery
2. **`feature_engineering.ipynb`** - Feature creation and data splitting
3. **`machine_learning_models.ipynb`** - XGBoost, LightGBM, RandomForest training
4. **`prophet_model.ipynb`** - Facebook Prophet time series forecasting
5. **`sarima_forecasting.ipynb`** - SARIMA model selection and ensemble

## Tech Stack

- **ML Frameworks**: XGBoost, LightGBM, scikit-learn
- **Time Series**: Prophet, statsmodels (SARIMA)
- **Data Processing**: pandas, numpy
- **Web Interface**: Gradio
- **Deployment**: HuggingFace Spaces

## Documentation

For comprehensive analysis and methodology:
- **`project_report0.md`** - Detailed project report with full methodology, metrics, and insights

## Project Structure

```
Multi-year_price_predicion/
├── input_data/              # Raw historical data (2012-2016)
├── data/                    # Processed train/validation/test sets
├── models/saved_models/     # Trained model artifacts (.pkl, .json, .joblib)
├── predictions/             # Generated forecasts (CSV files)
├── *.ipynb                  # Analysis notebooks (5 notebooks)
├── main.py                  # Gradio web application
├── requirements.txt         # Python dependencies
├── project_report0.md       # Comprehensive project report
└── EDA_report.md           # EDA summary
```

## Author

**Aakash**

- [GitHub Repository](https://github.com/aakashaldankar/Multi-year_price_predicion)
- [Live Demo on HuggingFace](https://huggingface.co/spaces/aakashaldankar/Multi-year_price_predicion)

---

**License**: See `LICENSE` file for details
