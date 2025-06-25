# NFL Player Breakout Prediction Model

## Team
- **Rohan Saxena** - [@rohanrsaxena](https://github.com/rohanrsaxena) 
- **Kyle Ayisi** - [@kayisi1](https://github.com/kayisi1)
- **Jeremy Ampofo** - [@jtamps](https://github.com/jtamps)

## Overview

We built a machine learning model to predict which NFL players in their first three seasons will have breakout performances. Using data from 2003-2024, the model focuses on running backs, wide receivers, and tight ends—positions where young players commonly emerge as fantasy football stars.

## What We Define as a "Breakout"

A breakout player meets these criteria:
- 1-3 years of NFL experience
- Finishes in top 15 (RB/WR) or top 10 (TE) for fantasy points
- Shows at least 25% improvement in key stats from previous season

## The Data

**Source:** 21 seasons of NFL player statistics (2003-2024)
**Features:** 70+ engineered metrics including:
- Traditional stats (rushing yards, receptions, TDs)
- Efficiency metrics (yards per route, catch rate)
- Opportunity indicators (target share, snap percentage)
- Experience factors (age, NFL tenure)

## Model Performance

Our Random Forest model achieved:
- **91% accuracy** on test data
- **47% recall** (identifies nearly half of actual breakouts)
- **0.897 ROC-AUC** (strong predictive discrimination)

We used SMOTE for handling class imbalance and time-based validation to simulate real forecasting conditions.

## Key Findings

1. **Efficiency matters more than volume** - Players who do more with limited opportunities often break out
2. **Year 2-3 sweet spot** - Most breakouts happen in a player's second or third season
3. **Usage trends** - Rising snap counts and target shares are strong predictors

## Technical Implementation

**Languages/Tools:** Python, scikit-learn, pandas, NumPy
**Approach:** 
- Compared multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting)
- Selected Random Forest with 200 trees and optimized hyperparameters
- Used GridSearch for parameter tuning
- Implemented stratified cross-validation

**Handling Imbalanced Data:**
- Combined SMOTE oversampling with random undersampling
- Calibrated probabilities for reliable confidence scores

## Applications

**Fantasy Sports:** Identify sleeper picks before they become widely recognized

**NFL Teams:** Supplement scouting with data-driven insights on young talent

**Sports Analytics:** Framework applicable to other positions and sports

## Limitations & Future Work

**Current Limitations:**
- Limited to RB/WR/TE positions
- Doesn't account for injuries or coaching changes
- Static yearly predictions (no mid-season updates)

**Potential Improvements:**
- Expand to QB and defensive positions
- Incorporate injury history and team context
- Add real-time updates during season
- Experiment with time-series models for trajectory analysis

## 2025 Season Predictions

The model has generated predictions for the upcoming season, which will serve as a real-world test of its effectiveness. We'll track these predictions throughout the 2025 season to validate model performance.

## Results & Code

Model artifacts, evaluation metrics, and prediction code are available in the project repository. The pipeline is automated and ready for annual updates with new season data.

---

*Analysis based on 21 seasons of NFL data using production-ready machine learning practices.*
