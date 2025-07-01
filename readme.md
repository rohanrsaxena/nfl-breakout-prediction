# NFL Player Breakout Prediction Model

## Team Members
- **Rohan Saxena** - [@rohanrsaxena](https://github.com/rohanrsaxena) 
- **Kyle Ayisi** - [@kayisi1](https://github.com/kayisi1)
- **Jeremy Ampofo** - [@jtamps](https://github.com/jtamps)

## Overview

This project predicts which NFL players (RB, WR, TE) in their first 3 seasons will have "breakout" performances using machine learning. Built with 21 seasons of NFL data (2003-2024), the model combines traditional statistics with advanced efficiency metrics to identify emerging fantasy football stars before they become widely recognized.

## Project Highlights

 **Real-world applicable**: Generated specific predictions for 2025 season  
 **Comprehensive data**: 12,904 player-season records from 2003-2024  
 **Domain expertise**: Position-specific features and breakout definitions  
 **Production-ready**: Robust error handling, logging, and model persistence  

## Breakout Definition

A player achieves "breakout" status when they:
- Have ≤3 years NFL experience
- Reach position-specific performance thresholds:
  - **RB**: Top 24 fantasy finish (~RB2 or better)
  - **WR**: Top 36 fantasy finish (~WR3 or better) 
  - **TE**: Top 12 fantasy finish (~TE1)
- Show significant improvement from previous season
- Meet minimum games played (6+ games)

## Model Performance

### Final Test Results (2023 Season)
- **Accuracy**: 94.1%
- **Precision**: 26.7% 
- **Recall**: 18.2%
- **F1-Score**: 21.6%
- **ROC-AUC**: 86.3%

### Key Insights
- **7.9% overall breakout rate** with position variation:
  - RB: 11.0% breakout rate
  - WR: 8.1% breakout rate  
  - TE: 3.3% breakout rate
- High accuracy reflects realistic class imbalance
- ROC-AUC of 0.863 indicates strong discriminative ability

## Technical Implementation

### Data Pipeline
```python
# Core workflow
df = load_data(start_year=2003, end_year=2024)
df = clean_data(df)
df = create_breakout_targets(df)
df = engineer_features(df)

# Temporal validation (prevents data leakage)
train: 2003-2020
validation: 2021-2022  
test: 2023
```

### Feature Engineering (100+ features)
- **Efficiency metrics**: Yards per carry, catch rate, yards per target
- **Opportunity indicators**: Target share, snap percentage, touches per game
- **Experience factors**: Age, NFL tenure, rookie/sophomore flags
- **Historical performance**: Career trajectory, year-over-year improvement
- **Position-specific**: RB workload, WR target concentration, TE red zone usage

### Model Architecture
- **Algorithm**: Logistic Regression (selected via validation)
- **Class balancing**: SMOTE oversampling with 1:3 target ratio
- **Feature scaling**: RobustScaler for outlier resistance
- **Hyperparameter tuning**: GridSearchCV with stratified cross-validation
- **Calibration**: Isotonic regression for reliable probabilities

### Comparison Results
| Model | F1-Score | ROC-AUC |
|-------|----------|---------|
| **Logistic Regression** | **0.328** | **0.902** |
| Random Forest | 0.293 | 0.909 |
| Gradient Boosting | 0.310 | 0.893 |

## Repository Structure

```
├── nfl_prediction_fixed.py          # Enhanced production model
├── nfl-prediction.py                # Simplified reference implementation  
├── nfl_breakout_outputs_enhanced/   # Model outputs
│   ├── models/                      # Saved model artifacts
│   ├── plots/                       # Evaluation visualizations
│   └── breakout_predictions_2025.csv # 2025 season predictions
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 2025 Season Predictions

### Top Breakout Candidates by Position

**Running Backs**
- Bucky Irving (47.5% probability)
- Tank Bigsby (46.7% probability)  
- Braelon Allen (46.2% probability)

**Wide Receivers**
- Rome Odunze (47.5% probability)
- Alec Pierce (47.5% probability)
- Keon Coleman (47.5% probability)

**Tight Ends**
- Ja'Tavion Sanders (44.4% probability)
- Kyle Pitts (44.4% probability)
- Tucker Kraft (44.4% probability)

*Full predictions available in `breakout_predictions_2025.csv`*

## Installation & Usage

### Prerequisites
```bash
pip install nfl_data_py pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib
```

### Quick Start
```python
# Train model and generate predictions
from nfl_prediction_fixed import NFLBreakoutPredictor

predictor = NFLBreakoutPredictor()
predictor, metrics, predictions = main()
```

### Custom Predictions
```python
# Load saved model for new predictions
from nfl_prediction_fixed import NFLBreakoutPredictor

predictor = NFLBreakoutPredictor()
predictor.load_model("path/to/saved/model.pkl")
predictions = predictor.predict_breakouts(new_data, top_n=20)
```

## Methodology Strengths

**Temporal validation** prevents data leakage  
**Domain expertise** in feature engineering  
**Robust evaluation** with multiple metrics  
**Production considerations** (logging, error handling)  
**Reproducible results** with saved artifacts  

## Limitations & Future Work

### Current Limitations
- **Position scope**: Limited to RB/WR/TE (excludes QB, defense)
- **External factors**: No injury history, coaching changes, or team context
- **Static predictions**: Annual updates only (no mid-season adjustments)
- **Sample size**: Small positive class limits model performance

## Model Validation

This model will be validated throughout the 2025 NFL season by tracking predicted vs. actual breakout performances. Results will be used to iterate and improve the model for future seasons.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## Acknowledgments

- **nfl_data_py** library for comprehensive NFL statistics
- **scikit-learn** and **imbalanced-learn** for ML framework
- NFL data community for maintaining historical statistics

---

*Predicting the next generation of NFL stars through data science*
