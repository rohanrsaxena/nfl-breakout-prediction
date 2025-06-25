# NFL Player Breakout Prediction Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## Developers
- **Kyle Ayisi** - [@kayisi1](https://github.com/kayisi1)
- **Rohan Saxena** - [@rohanrsaxena](https://github.com/rohanrsaxena) 
- **Jeremy Ampofo** - [@jtamps](https://github.com/jtamps)

---

## Project Overview

This project develops a machine learning model to predict NFL "breakout" players using 21 seasons of historical data (2003-2024). The model identifies young players positioned for significant performance improvements in the following season.

### Key Results
- 91% accuracy on validation data
- 47% recall for identifying actual breakouts
- 0.897 ROC-AUC score
- Focus on RB, WR, and TE positions

## Breakout Player Definition

A breakout player meets all of the following criteria:
- 1-3 years of NFL experience
- Reaches top 15 (RB/WR) or top 10 (TE) in position-based fantasy points
- Demonstrates at least 25% improvement in key production metrics

## Dataset & Features

**Data Sources:**
- 21 seasons of NFL data (2003-2024)
- 70+ engineered features combining traditional and advanced metrics
- Player roster information and performance statistics

**Feature Categories:**
- Traditional Stats: Rushing/receiving yards, touchdowns, games played
- Efficiency Metrics: Yards per route, catch rate, rushing efficiency
- Opportunity Metrics: Targets per game, snap percentage, routes run
- Experience Factors: Years in NFL, age, sophomore/junior indicators

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/[your-username]/nfl-breakout-prediction.git
cd nfl-breakout-prediction
pip install -r requirements.txt
```

### Usage
```python
# Run the complete training pipeline
python MLProjFinal.py

# The model will:
# 1. Load and process NFL data (2003-2024)
# 2. Engineer position-specific features
# 3. Train Random Forest classifier
# 4. Generate 2025 breakout predictions
# 5. Save results to /FinalOutput/breakout_predictions_2025.csv
```

## Model Architecture

### Data Pipeline
1. **Data Loading**: NFL seasonal data + roster information
2. **Cleaning**: Handle missing values, standardize formats
3. **Feature Engineering**: Create 70+ predictive features
4. **Class Balancing**: SMOTE + undersampling for imbalanced data
5. **Model Training**: Random Forest with hyperparameter tuning

### Model Selection
Multiple approaches were evaluated:
- Logistic Regression: Baseline model with L1 regularization
- Random Forest: Best performing (200 trees, max depth 10)
- Gradient Boosting: Competitive but less interpretable

## Results & Evaluation

### Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 91.0% |
| Precision | Variable by threshold |
| Recall | 47.0% |
| F1-Score | Optimized via threshold tuning |
| ROC-AUC | 0.897 |

### Key Insights
- Efficiency metrics outperform volume statistics for prediction
- Years 2-3 show highest breakout probability
- Position-specific thresholds improve model performance

## Project Structure

```
nfl-breakout-prediction/
├── nfl-prediction.py              # Main training script
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── Project_Report.pdf      # Detailed technical report
├── FinalOutput/               # Model outputs
│   ├── breakout_predictions_2025.csv
│   └── randomforestbreakoutmodel.pkl
└── Plots/                     # Generated visualizations
    ├── roc_curve.png
    ├── confusion_matrix.png
    └── precision_recall_curve.png
```

## Applications

### NFL Teams & Scouts
- Identify undervalued players before market value increases
- Guide draft strategy and roster decisions
- Supplement traditional scouting with data-driven insights

### Fantasy Sports
- Draft high-upside players at low cost
- Gain competitive advantage in leagues
- Inform waiver wire pickups

### Sports Betting
- Adjust player prop betting lines
- Identify value bets on player performance
- Inform season-long award predictions

## Limitations & Future Work

### Current Limitations
- Scope limited to RB/WR/TE positions
- External factors not modeled (injuries, coaching changes)
- Modest dataset size by machine learning standards
- Subjective breakout definition criteria

### Future Enhancements
- Expand to quarterback and defensive players
- Integrate injury history and coaching data
- Implement time-series models for trajectory analysis
- Real-time model updates during season

## Acknowledgments

- NFL data provided by [nfl_data_py](https://github.com/cooperdff/nfl_data_py)

## Contact
For questions about this project, please contact any of the team members:
- Kyle Ayisi - [@kayisi1](https://github.com/kayisi1)
- Rohan Saxena - [@rohanrsaxena](https://github.com/rohanrsaxena)
- Jeremy Ampofo - [@jtamps](https://github.com/jtamps)
