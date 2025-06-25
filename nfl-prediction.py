"""
Machine Learning Final Project

NFL Player Breakout Prediction Model - Kyle, Rohan, Jeremy Group 35
------------------------------------------------------
A  model for predicting NFL player breakouts with:
1. Position-specific feature engineering
2. Efficient classification pipeline
3. Customizable model parameters
4. Comprehensive evaluation metrics

This model should output the breakout players for 2025, along with their probability of breaking out, and their names
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import joblib
import os
from datetime import datetime

# Suppress warnings and set display options
warnings.filterwarnings("ignore")
np.random.seed(42)
pd.set_option('display.max_columns', None)

# Define directories for plots and final outputs
PLOTS_DIR = '/Users/jeremy/Downloads/ML Project Outputs/Plots'
FINAL_OUTPUT_DIR = '/Users/jeremy/Downloads/ML Project Outputs/FinalOutput'

# Create the directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)



# Utility Functions

def save_plot(fig, filename, dpi=300):
    filepath = os.path.join(PLOTS_DIR, filename)
    print(f"Saving plot to: {filepath}")  # Debug: print file path
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return filepath

# Execute a plotting function and save the resulting figure.
def plot_with_save(plot_func, filename, *args, **kwargs):

    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6)))
    plot_func(ax, *args, **kwargs)
    plt.tight_layout()
    return save_plot(fig, filename)

# Plot and save a reliability (calibration) curve for a classifier.
def plot_reliability_curve(y_true, y_prob, model_name, n_bins=10):

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Reliability Curve - {model_name}")
    ax.legend(loc="best")
    return save_plot(fig, f"{model_name.lower().replace(' ', '_')}_reliability_curve.png")

# Plot and save a curve showing both precision and recall versus decision thresholds.
def plot_precision_recall_vs_threshold(y_true, y_prob, filename="precision_recall_vs_threshold.png"):

    from sklearn.metrics import precision_score, recall_score
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, precisions, marker="o", label="Precision")
    ax.plot(thresholds, recalls, marker="o", label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall vs Threshold")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.7)

    save_plot(fig, filename)
    plt.show()
# Find optimal classification threshold to maximize F1 score.
def optimize_threshold(y_true, y_prob, thresholds=None):

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    print(f"Optimized threshold: {best_threshold:.3f} with F1 score: {best_f1:.3f}")
    return best_threshold, best_f1

# Plot and save a confusion matrix.
# If normalize is True, each cell displays both the raw count and the percentage.
def plot_confusion_matrix(cm, classes, model_name, filename, normalize=True):


    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_norm[i, j]:.2%})'
        fmt = ''
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        annot = True
        fmt = 'd'
        title = f'Confusion Matrix - {model_name}'

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    return save_plot(fig, filename)

 # Plot F1 score as a function of decision thresholds and save the figure.
def plot_threshold_vs_f1(y_true, y_prob, thresholds=None):

    from sklearn.metrics import f1_score
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, f1_scores, marker='o')
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs. Decision Threshold")
    ax.grid(True)
    save_plot(fig, "threshold_vs_f1.png")
    plt.show()



# Data Processing Functions

# Load NFL data for the specified year range.
def load_data(start_year=2003, end_year=2024):

    print(f"Loading NFL data from {start_year} to {end_year}...")
    years = list(range(start_year, end_year + 1))
    df = nfl.import_seasonal_data(years)
    rosters = nfl.import_seasonal_rosters(years)
    df.columns = df.columns.str.lower()
    rosters.columns = rosters.columns.str.lower()
    df = df.rename(columns={'rushing_yds': 'rushing_yards', 'receiving_yds': 'receiving_yards'})
    rosters_slim = rosters[['player_id', 'position', 'player_name', 'height',
                             'weight', 'college', 'birth_date']].drop_duplicates()
    df = df.merge(rosters_slim, on='player_id', how='left')
    print(f"Loaded {df.shape[0]} player-season records")
    return df

# Clean and prepare data for analysis
def clean_data(df):

    default_columns = {
        'rush_td': 0, 'rec_td': 0, 'targets': 0, 'rec': 0,
        'routes_run': 0, 'games_started': 0, 'rushing_yards': 0, 'receiving_yards': 0
    }
    for col, default in default_columns.items():
        if col not in df.columns:
            df[col] = default
    if 'rec' in df.columns and 'receptions' not in df.columns:
        df['receptions'] = df['rec']
    df['rookie_year'] = df.groupby('player_id')['season'].transform('min')
    df['years_in_nfl'] = df['season'] - df['rookie_year']
    if 'birth_date' in df.columns:
        df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
        df['age'] = df.apply(
            lambda x: np.nan if pd.isna(x['birth_date']) else
            x['season'] - x['birth_date'].year -
            ((x['season'] % 100, x['birth_date'].month, x['birth_date'].day) < (9, 1, 1)),
            axis=1
        )
        df['age'] = df['age'].fillna(df.groupby('position')['age'].transform('mean'))
    if 'height' in df.columns:
        df['height_inches'] = df['height'].apply(
            lambda x: np.nan if pd.isna(x) else
            int(x.split('-')[0]) * 12 + int(x.split('-')[1])
            if isinstance(x, str) and '-' in x else np.nan
        )
        df['height_inches'] = df['height_inches'].fillna(
            df.groupby('position')['height_inches'].transform('mean')
        )
    df = df[df['position'].isin(['RB', 'WR', 'TE'])].copy()
    df = df[df['games'] > 0].copy()
    df['touchdowns'] = df['rush_td'].fillna(0) + df['rec_td'].fillna(0)
    df['fantasy_points'] = (df['rushing_yards'] / 10) + (df['receiving_yards'] / 10) + (6 * df['touchdowns'])
    df['season_rank'] = df.groupby(['season', 'position'])['fantasy_points'].rank(method='min', ascending=False)
    df.drop(columns=['rookie_year'], inplace=True, errors='ignore')
    return df

# Create breakout target variable based on specified parameters.
def create_breakout_target(df, rank_threshold_rb_wr=15, rank_threshold_te=10,
                           improvement_threshold=0.25, max_experience=3,
                           consecutive_seasons=True, target_column='breakout_next_year'):

    print(f"Creating breakout target with threshold: RB/WR={rank_threshold_rb_wr}, TE={rank_threshold_te}")
    def assign_breakout(group):
        group = group.sort_values('season').reset_index(drop=True)
        breakout_flags = [0] * len(group)
        pos = group.iloc[0]['position']
        rank_threshold = rank_threshold_rb_wr if pos in ['RB', 'WR'] else rank_threshold_te
        breakout_achieved = False
        for i in range(len(group) - 1):
            if not breakout_flags[i] and not breakout_achieved:
                curr_season = group.iloc[i]['season']
                curr_rank = group.iloc[i]['season_rank']
                curr_points = group.iloc[i]['fantasy_points']
                curr_experience = group.iloc[i]['years_in_nfl']
                next_season = group.iloc[i + 1]['season']
                next_rank = group.iloc[i + 1]['season_rank']
                next_points = group.iloc[i + 1]['fantasy_points']
                consecutive_season_check = not consecutive_seasons or (next_season - curr_season) == 1
                rank_condition = next_rank <= rank_threshold and curr_rank > rank_threshold
                points_condition = (curr_points > 0) and ((next_points - curr_points) / curr_points >= improvement_threshold)
                experience_condition = curr_experience <= max_experience
                if consecutive_season_check and (rank_condition or points_condition) and experience_condition:
                    breakout_flags[i] = 1
                    breakout_achieved = True
        group[target_column] = breakout_flags
        return group
    df = df.groupby('player_id').apply(assign_breakout).reset_index(drop=True)
    breakout_count = df[target_column].sum()
    breakout_pct = (breakout_count / len(df)) * 100
    print(f"Identified {breakout_count} breakout instances ({breakout_pct:.2f}% of all player-seasons)")
    return df

# Add position-specific and general features for modeling.
def engineer_features(df):

    print("Engineering features...")
    eps = 1e-6
    df['fantasy_point_change'] = df.groupby('player_id')['fantasy_points'].diff().fillna(0)
    df['fantasy_point_pct_change'] = df.groupby('player_id')['fantasy_points'].pct_change().fillna(0)
    df['touchdowns_per_game'] = df['touchdowns'] / (df['games'] + eps)
    df['sophomore_junior'] = ((df['years_in_nfl'] == 1) | (df['years_in_nfl'] == 2)).astype(int)
    df['experience_squared'] = df['years_in_nfl'] ** 2
    rb_mask = df['position'] == 'RB'
    df['rushing_yards_per_game'] = 0
    df.loc[rb_mask, 'rushing_yards_per_game'] = df.loc[rb_mask, 'rushing_yards'] / (df.loc[rb_mask, 'games'] + eps)
    if 'rush_att' in df.columns:
        df['rushing_efficiency'] = 0
        df.loc[rb_mask, 'rushing_efficiency'] = df.loc[rb_mask, 'rushing_yards'] / (df.loc[rb_mask, 'rush_att'] + eps)
    wr_mask = df['position'] == 'WR'
    df['receiving_yards_per_game'] = 0
    df.loc[wr_mask, 'receiving_yards_per_game'] = df.loc[wr_mask, 'receiving_yards'] / (df.loc[wr_mask, 'games'] + eps)
    df['targets_per_game'] = 0
    df.loc[wr_mask, 'targets_per_game'] = df.loc[wr_mask, 'targets'] / (df.loc[wr_mask, 'games'] + eps)
    df['catch_rate'] = 0
    df.loc[wr_mask, 'catch_rate'] = df.loc[wr_mask, 'receptions'] / (df.loc[wr_mask, 'targets'] + eps)
    te_mask = df['position'] == 'TE'
    df.loc[te_mask, 'receiving_yards_per_game'] = df.loc[te_mask, 'receiving_yards'] / (df.loc[te_mask, 'games'] + eps)
    df.loc[te_mask, 'targets_per_game'] = df.loc[te_mask, 'targets'] / (df.loc[te_mask, 'games'] + eps)
    df.loc[te_mask, 'catch_rate'] = df.loc[te_mask, 'receptions'] / (df.loc[te_mask, 'targets'] + eps)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = pd.get_dummies(df, columns=['position'], drop_first=False)
    print(f"Feature engineering complete. DataFrame now has {df.shape[1]} columns")
    return df

# Select relevant features for modeling, preserving key identifier columns.
def select_features(df, target_column, exclude_cols=None):

    if exclude_cols is None:
        exclude_cols = ['height', 'weight', 'college', 'birth_date',
                        'season_type', 'headshot_url']
    if target_column in exclude_cols:
        exclude_cols.remove(target_column)
    all_columns = df.columns.tolist()
    modeling_columns = [col for col in all_columns if col not in exclude_cols]
    print(f"Selected {len(modeling_columns)} columns for data")
    return df[modeling_columns]



# Model Training and Evaluation

# Split data into training, validation, and test sets based on seasons.
def split_train_test(df, target_col, test_season=None, val_season=None):

    print(f"Splitting data: validation season {val_season}, test season {test_season}")
    if test_season is None and val_season is not None:
        train_mask = df['season'] < val_season
        val_mask = df['season'] == val_season
        X_train = df[train_mask].drop(columns=[target_col])
        y_train = df[train_mask][target_col]
        X_val = df[val_mask].drop(columns=[target_col])
        y_val = df[val_mask][target_col]
        X_test = X_val.copy().iloc[0:0]
        y_test = y_val.copy().iloc[0:0]
        print(f"Training set: {len(X_train)} samples with {y_train.sum()} positive cases")
        print(f"Validation set: {len(X_val)} samples with {y_val.sum()} positive cases")
        print("Test set: Not used (using all data for training and validation)")
    elif test_season is not None and val_season is not None:
        train_mask = df['season'] < val_season
        val_mask = df['season'] == val_season
        test_mask = df['season'] == test_season
        X_train = df[train_mask].drop(columns=[target_col])
        y_train = df[train_mask][target_col]
        X_val = df[val_mask].drop(columns=[target_col])
        y_val = df[val_mask][target_col]
        X_test = df[test_mask].drop(columns=[target_col])
        y_test = df[test_mask][target_col]
        print(f"Training set: {len(X_train)} samples with {y_train.sum()} positive cases")
        print(f"Validation set: {len(X_val)} samples with {y_val.sum()} positive cases")
        print(f"Test set: {len(X_test)} samples with {y_test.sum()} positive cases")
    else:
        print("No specific seasons provided. Using random 70/15/15 split.")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=y if len(y.unique()) > 1 else None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42,
            stratify=y_temp if len(y_temp.unique()) > 1 else None
        )
        print(f"Training set: {len(X_train)} samples with {y_train.sum()} positive cases")
        print(f"Validation set: {len(X_val)} samples with {y_val.sum()} positive cases")
        print(f"Test set: {len(X_test)} samples with {y_test.sum()} positive cases")
    return X_train, y_train, X_val, y_val, X_test, y_test

# Scale features and  address class imbalances.
def prepare_modeling_data(X_train, y_train, X_val, X_test, balance_method='combined'):

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Scaling {len(numeric_cols)} numeric features")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_val_scaled = scaler.transform(X_val[numeric_cols]) if len(X_val) > 0 else np.empty((0, len(numeric_cols)))
    X_test_scaled = scaler.transform(X_test[numeric_cols]) if len(X_test) > 0 else np.empty((0, len(numeric_cols)))
    print(f"Balancing classes using {balance_method} method")
    if balance_method == 'smote':
        sampler = SMOTE(random_state=42)
        X_balanced, y_balanced = sampler.fit_resample(X_train_scaled, y_train)
    elif balance_method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = sampler.fit_resample(X_train_scaled, y_train)
    elif balance_method == 'combined':
        pipeline = ImbPipeline([
            ('undersample', RandomUnderSampler(random_state=42, sampling_strategy=0.5)),
            ('smote', SMOTE(random_state=42))
        ])
        X_balanced, y_balanced = pipeline.fit_resample(X_train_scaled, y_train)
    else:
        X_balanced, y_balanced = X_train_scaled, y_train
    class_counts = np.bincount(y_balanced)
    print(f"Balanced class distribution: {class_counts[0]} neg, {class_counts[1]} pos")
    return X_balanced, y_balanced, X_val_scaled, X_test_scaled, scaler, numeric_cols

# Train a classification model with grid search for hyperparameters.
def train_model(X_train, y_train, X_val, y_val, model_name='RandomForest'):

    print(f"Training {model_name} model...")
    if model_name == 'RandomForest':
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
    else:
        base_model = LogisticRegression(random_state=42, max_iter=5000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': ['balanced', None],
            'solver': ['liblinear', 'saga']
        }
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    calibrated_model = CalibratedClassifierCV(estimator=best_model, cv=3, method='isotonic')
    calibrated_model.fit(X_train, y_train)
    if len(np.unique(y_val)) > 1:
        y_val_prob = calibrated_model.predict_proba(X_val)[:, 1]
        opt_threshold, _ = optimize_threshold(y_val, y_val_prob)
    else:
        print("Warning: Validation set has only one class. Using default threshold of 0.5.")
        opt_threshold = 0.5
    return calibrated_model, opt_threshold

# Evaluate model performance and generate plots.
def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):

    print(f"Evaluating {model_name} with threshold {threshold:.3f}")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
        'threshold': threshold
    }
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    if len(np.unique(y_test)) > 1:
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        classes = ["No Breakout", "Breakout"]
        cm_filepath = plot_confusion_matrix(cm, classes, model_name,
                                            f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png',
                                            normalize=True)
        print(f"Normalized confusion matrix saved to: {cm_filepath}")

        # Plot ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.7)
        save_plot(fig, f'{model_name.lower().replace(" ", "_")}_roc_curve.png')

        # Plot Precision-Recall Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        ax.plot(recall_curve, precision_curve, label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.legend(loc="lower left")
        ax.grid(True, linestyle='--', alpha=0.7)
        save_plot(fig, f'{model_name.lower().replace(" ", "_")}_pr_curve.png')

        # Plot Reliability Curve
        plot_reliability_curve(y_test, y_prob, model_name)
    else:
        print("Warning: Cannot create evaluation plots - test set has only one class")
    return metrics, y_pred, y_prob

# Save trained model and associated metadata.
def save_model(model, scaler, feature_names, numeric_cols, model_name, threshold, metrics):

    print("Saving model artifacts...")
    model_info = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'numeric_cols': numeric_cols,
        'threshold': threshold,
        'metrics': metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    model_filename = os.path.join(FINAL_OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}.pkl")
    joblib.dump(model_info, model_filename)
    print(f"Model saved to {model_filename}")
    return model_filename

# Make breakout predictions for new data or a specific season.
def predict_breakouts(model_path, new_data=None, season=None):

    print(f"Loading model from {model_path}...")
    model_info = joblib.load(model_path)
    model = model_info['model']
    scaler = model_info['scaler']
    feature_names = model_info['feature_names']
    threshold = model_info['threshold']
    if new_data is None:
        if season is None:
            current_year = datetime.now().year
            season = current_year - 1 if datetime.now().month < 9 else current_year
        print(f"Loading data for season {season}...")
        df = load_data(start_year=season, end_year=season)
        df = clean_data(df)
        df = engineer_features(df)
    else:
        df = new_data.copy()
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features in data: {missing_features}")
        for feature in missing_features:
            df[feature] = 0
    numeric_cols = model_info['numeric_cols']
    X = df[numeric_cols].copy()
    X_scaled = scaler.transform(X)
    breakout_probs = model.predict_proba(X_scaled)[:, 1]
    breakout_preds = (breakout_probs >= threshold).astype(int)
    df['breakout_probability'] = breakout_probs
    df['breakout_prediction'] = breakout_preds
    df_sorted = df.sort_values('breakout_probability', ascending=False)
    desired_cols = ['player_id', 'player_name', 'position', 'team', 'breakout_probability', 'breakout_prediction']
    available_cols = [col for col in desired_cols if col in df_sorted.columns]
    if 'breakout_probability' not in available_cols:
        available_cols.append('breakout_probability')
    if 'breakout_prediction' not in available_cols:
        available_cols.append('breakout_prediction')
    print("\nTop Breakout Predictions:")
    print(df_sorted[available_cols].head(10))
    return df_sorted

 # Run complete training pipeline for breakout prediction.
def run_training_pipeline(start_year=2003, end_year=2024, test_season=None, val_season=2024):

    run_start = datetime.now()
    print("=" * 80)
    print("NFL PLAYER BREAKOUT PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Starting run at: {run_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Training on data from {start_year} to {end_year}")
    print(f"Using {val_season} as validation season")
    df = load_data(start_year, end_year)
    df = clean_data(df)
    target_col = 'breakout_next_year'
    df = create_breakout_target(df, target_column=target_col)
    df = engineer_features(df)
    modeling_df = select_features(df, target_col)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_test(
        modeling_df, target_col, test_season, val_season
    )
    if len(np.unique(y_val)) < 2:
        print("Warning: Validation set has only one class. Using cross-validation on training data instead.")
        X_train_combined = pd.concat([X_train, X_val])
        y_train_combined = pd.concat([y_train, y_val])
        cv_folds = min(5, np.bincount(y_train_combined)[1])
        cv_folds = max(cv_folds, 3)
        X_balanced, y_balanced, _, X_test_scaled, scaler, numeric_cols = prepare_modeling_data(
            X_train_combined, y_train_combined, X_train_combined.iloc[0:0], X_test, balance_method='combined'
        )
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        thresholds = []
        f1_scores = []
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        for train_idx, val_idx in cv.split(X_balanced, y_balanced):
            X_fold_train, X_fold_val = X_balanced[train_idx], X_balanced[val_idx]
            y_fold_train, y_fold_val = y_balanced[train_idx], y_balanced[val_idx]
            rf_model.fit(X_fold_train, y_fold_train)
            y_fold_prob = rf_model.predict_proba(X_fold_val)[:, 1]
            fold_threshold, fold_f1 = optimize_threshold(y_fold_val, y_fold_prob)
            thresholds.append(fold_threshold)
            f1_scores.append(fold_f1)
        rf_threshold = np.mean(thresholds)
        print(f"Average optimal threshold from CV: {rf_threshold:.3f}")
        rf_model.fit(X_balanced, y_balanced)
        rf_metrics = {
            'cv_f1_mean': np.mean(f1_scores),
            'cv_f1_std': np.std(f1_scores),
            'cv_threshold_mean': rf_threshold,
            'cv_threshold_std': np.std(thresholds)
        }
        print(f"Cross-validation F1 score: {rf_metrics['cv_f1_mean']:.4f} ± {rf_metrics['cv_f1_std']:.4f}")
    else:
        X_balanced, y_balanced, X_val_scaled, X_test_scaled, scaler, numeric_cols = prepare_modeling_data(
            X_train, y_train, X_val, X_test, balance_method='combined'
        )
        rf_model, rf_threshold = train_model(
            X_balanced, y_balanced, X_val_scaled, y_val, model_name='RandomForest'
        )
        print("Evaluating model on validation set")
        rf_metrics, _, _ = evaluate_model(
            rf_model, X_val_scaled, y_val, "Random Forest (Validation)", threshold=rf_threshold
        )
    model_path = save_model(
        rf_model, scaler, X_train.columns.tolist(), numeric_cols,
        "RandomForestBreakoutModel", rf_threshold, rf_metrics
    )
    run_end = datetime.now()
    print(f"\nTraining pipeline completed. Runtime: {run_end - run_start}")
    return model_path

# Main function to train model and predict breakouts for 2025.
def main():

    print("NFL Player Breakout Prediction Model")
    print("------------------------------------")

    # Use 2023 as the validation season so that the breakout targets are computed
    # (players in 2023 have their breakout target based on their 2024 performance)
    model_path = run_training_pipeline(
        start_year=2003,
        end_year=2024,
        test_season=None,
        val_season=2023
    )

    # To predict breakouts for 2025, we use season 2024 data as input.
    print(f"\nPredicting breakouts for the 2025 season based on 2024 data:")
    predictions = predict_breakouts(model_path, season=2024)
    prediction_file = os.path.join(FINAL_OUTPUT_DIR, "breakout_predictions_2025.csv")
    pred_cols = ['player_id', 'player_name', 'position', 'team', 'breakout_probability', 'breakout_prediction']
    available_pred_cols = [col for col in pred_cols if col in predictions.columns]
    predictions[available_pred_cols].to_csv(prediction_file, index=False)
    print(f"Predictions saved to {prediction_file}")
    return predictions


if __name__ == "__main__":
    main()