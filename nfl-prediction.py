"""
NFL Player Breakout Prediction Model 

A  model for predicting NFL player breakouts with proper temporal validation,
improved feature engineering, and better error handling.

Key Improvements:
1. Better data validation and error handling
2. More sophisticated breakout definitions
3. Enhanced feature engineering with domain expertise
4. Proper handling of edge cases
5. Better class imbalance handling
6. More evaluation metrics
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score,
                             classification_report)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
import joblib
import os
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
import logging

# Configuration
warnings.filterwarnings("ignore")
np.random.seed(42)
pd.set_option('display.max_columns', None)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants - Refined based on NFL expertise
POSITIONS = ['RB', 'WR', 'TE']
RANK_THRESHOLDS = {
    'RB': 24,    # Top 24 RBs (RB2 or better)
    'WR': 36,    # Top 36 WRs (WR3 or better)
    'TE': 12     # Top 12 TEs (TE1)
}
MIN_GAMES = 6  # At least 6 games for meaningful sample
FANTASY_BREAKOUT_THRESHOLDS = {
    'RB': 150,   # ~9.4 PPG for 16 games
    'WR': 140,   # ~8.75 PPG for 16 games
    'TE': 100    # ~6.25 PPG for 16 games
}
MAX_EXPERIENCE = 3  # Focus on years 0-3

class NFLBreakoutPredictor:
    """
    Enhanced NFL player breakout prediction model with robust error handling
    and domain-specific improvements.
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        self.models_dir = os.path.join(output_dir, "models")
        self.logs_dir = os.path.join(output_dir, "logs")
        
        # Create directories
        for directory in [self.output_dir, self.plots_dir, self.models_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.optimal_threshold = 0.5
        self.feature_importance = None
        
    def load_data(self, start_year: int = 2003, end_year: int = 2024) -> pd.DataFrame:
        """Load and merge NFL seasonal data with proper error handling."""
        logger.info(f"Loading NFL data from {start_year} to {end_year}...")
        
        try:
            years = list(range(start_year, end_year + 1))
            
            # Load seasonal stats
            df = nfl.import_seasonal_data(years)
            if df.empty:
                raise ValueError("No seasonal data loaded")
            
            # Load rosters
            rosters = nfl.import_seasonal_rosters(years)
            if rosters.empty:
                logger.warning("No roster data available, proceeding with limited features")
                rosters = pd.DataFrame()
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            if not rosters.empty:
                rosters.columns = rosters.columns.str.lower()
            
            # Fix common column name issues
            column_mappings = {
                'rushing_yds': 'rushing_yards',
                'receiving_yds': 'receiving_yards',
                'receiving_rec': 'receptions',
                'rec': 'receptions',
                'position_group': 'position'
            }
            
            for old_col, new_col in column_mappings.items():
                if old_col in df.columns and new_col not in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Merge with roster data if available
            if not rosters.empty:
                # Handle potential duplicates in roster data
                roster_cols = ['player_id', 'season', 'position', 'player_name', 
                              'height', 'weight', 'college', 'birth_date', 'years_exp']
                available_roster_cols = [col for col in roster_cols if col in rosters.columns]
                
                # Keep most recent position for each player-season
                rosters_dedup = rosters[available_roster_cols].sort_values(
                    ['player_id', 'season', 'years_exp']
                ).drop_duplicates(subset=['player_id', 'season'], keep='last')
                
                df = df.merge(rosters_dedup, on=['player_id', 'season'], how='left', suffixes=('', '_roster'))
                
                # Use roster position if stats position is missing
                if 'position_roster' in df.columns:
                    df['position'] = df['position'].fillna(df['position_roster'])
                    df = df.drop(columns=['position_roster'])
            
            logger.info(f"Loaded {df.shape[0]} player-season records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data with robust error handling."""
        logger.info("Cleaning data...")
        
        try:
            # Essential columns that must exist
            required_cols = ['player_id', 'season', 'games', 'position']
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")
            
            # Fill missing statistical columns with zeros
            stat_columns = {
                'rush_td': 0, 'rec_td': 0, 'targets': 0, 'receptions': 0,
                'rushing_yards': 0, 'receiving_yards': 0, 'rush_att': 0,
                'games_started': 0, 'fumbles': 0, 'fumbles_lost': 0
            }
            
            for col, default_val in stat_columns.items():
                if col not in df.columns:
                    df[col] = default_val
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
            
            # Calculate experience
            df['rookie_year'] = df.groupby('player_id')['season'].transform('min')
            df['years_in_nfl'] = df['season'] - df['rookie_year']
            
            # Calculate age if birth_date available
            if 'birth_date' in df.columns:
                df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
                df['age'] = df.apply(self._calculate_age, axis=1)
                df['age'] = df['age'].fillna(df.groupby('position')['age'].transform('median'))
            else:
                # Estimate age based on experience
                df['age'] = 22 + df['years_in_nfl']
            
            # Physical attributes
            if 'height' in df.columns:
                df['height_inches'] = df['height'].apply(self._height_to_inches)
                df['height_inches'] = df['height_inches'].fillna(
                    df.groupby('position')['height_inches'].transform('median')
                )
            
            if 'weight' in df.columns:
                df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
                df['weight'] = df['weight'].fillna(
                    df.groupby('position')['weight'].transform('median')
                )
            
            # Filter to relevant positions and active players
            df = df[df['position'].isin(POSITIONS)].copy()
            df = df[df['games'] > 0].copy()
            
            # Calculate fantasy metrics (standard scoring)
            df['total_tds'] = df['rush_td'] + df['rec_td']
            df['fantasy_points'] = (
                df['rushing_yards'] * 0.1 + 
                df['receiving_yards'] * 0.1 + 
                df['total_tds'] * 6 +
                df['receptions'] * 0  # Standard scoring, change to 0.5 or 1 for PPR
            )
            
            # Remove temporary columns
            df = df.drop(columns=['rookie_year'], errors='ignore')
            
            # Remove duplicate player-seasons
            df = df.drop_duplicates(subset=['player_id', 'season'])
            
            logger.info(f"Cleaned data: {df.shape[0]} records remaining")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def _calculate_age(self, row):
        """Calculate age at start of NFL season."""
        if pd.isna(row.get('birth_date')):
            return np.nan
        
        season_start = datetime(row['season'], 9, 1)
        age = (season_start - row['birth_date']).days / 365.25
        return age
    
    def _height_to_inches(self, height_str):
        """Convert height string to inches."""
        if pd.isna(height_str) or not isinstance(height_str, str):
            return np.nan
        
        try:
            if '-' in height_str:
                feet, inches = height_str.split('-')
                return int(feet) * 12 + int(inches)
            elif '\'' in height_str:  # Handle 6'2" format
                parts = height_str.replace('"', '').split('\'')
                if len(parts) == 2:
                    return int(parts[0]) * 12 + int(parts[1])
        except:
            pass
        return np.nan
    
    def create_breakout_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create breakout targets with sophisticated NFL-specific logic.
        """
        logger.info("Creating breakout targets...")
        
        def calculate_breakout_for_player(group):
            """Calculate breakout status for each player-season."""
            group = group.sort_values('season').reset_index(drop=True)
            group['is_breakout'] = 0
            
            position = group.iloc[0]['position']
            rank_threshold = RANK_THRESHOLDS[position]
            fantasy_threshold = FANTASY_BREAKOUT_THRESHOLDS[position]
            
            for i in range(len(group) - 1):
                current_row = group.iloc[i]
                next_row = group.iloc[i + 1]
                
                # Check consecutive seasons
                if next_row['season'] - current_row['season'] != 1:
                    continue
                
                # Experience requirement
                if current_row['years_in_nfl'] > MAX_EXPERIENCE:
                    continue
                
                # Performance metrics
                next_fantasy = next_row['fantasy_points']
                next_games = next_row['games']
                current_fantasy = current_row['fantasy_points']
                current_games = current_row['games']
                
                # Must play meaningful games
                if next_games < MIN_GAMES:
                    continue
                
                # Calculate position rank
                same_pos_next = df[
                    (df['season'] == next_row['season']) & 
                    (df['position'] == position) &
                    (df['games'] >= MIN_GAMES)
                ]['fantasy_points'].values
                
                if len(same_pos_next) < 20:  # Need sufficient sample
                    continue
                
                next_rank = (same_pos_next > next_fantasy).sum() + 1
                next_ppg = next_fantasy / next_games if next_games > 0 else 0
                
                # Multiple breakout criteria
                
                # 1. Elite performance breakout
                elite_breakout = (next_rank <= rank_threshold and 
                                 next_fantasy >= fantasy_threshold)
                
                # 2. Massive improvement breakout (50%+ improvement and top-tier)
                improvement_breakout = False
                if current_games >= 4 and current_fantasy > 20:
                    current_ppg = current_fantasy / current_games
                    improvement_pct = (next_ppg - current_ppg) / current_ppg if current_ppg > 0 else 0
                    improvement_breakout = (improvement_pct >= 0.5 and 
                                          next_rank <= rank_threshold * 1.5)
                
                # 3. Young emergence breakout (rookies/sophomores)
                emergence_breakout = (current_row['years_in_nfl'] <= 1 and 
                                    next_ppg >= fantasy_threshold / 16 and  # Per-game threshold
                                    next_rank <= rank_threshold * 1.25)
                
                # 4. Volume breakout (significant increase in opportunities)
                volume_breakout = False
                if position in ['RB', 'WR']:
                    current_touches = current_row.get('rush_att', 0) + current_row.get('targets', 0)
                    next_touches = next_row.get('rush_att', 0) + next_row.get('targets', 0)
                    if current_touches > 0:
                        touch_increase = (next_touches - current_touches) / current_touches
                        volume_breakout = (touch_increase >= 0.5 and next_rank <= rank_threshold)
                
                if elite_breakout or improvement_breakout or emergence_breakout or volume_breakout:
                    group.loc[group.index[i], 'is_breakout'] = 1
            
            return group
        
        # Apply breakout calculation
        df_with_targets = df.groupby('player_id').apply(
            calculate_breakout_for_player
        ).reset_index(drop=True)
        
        # Calculate and report statistics
        eligible_mask = df_with_targets['years_in_nfl'] <= MAX_EXPERIENCE
        breakout_count = df_with_targets.loc[eligible_mask, 'is_breakout'].sum()
        total_eligible = eligible_mask.sum()
        
        logger.info(f"Breakout instances: {breakout_count}")
        logger.info(f"Eligible player-seasons: {total_eligible}")
        logger.info(f"Breakout rate: {breakout_count/total_eligible*100:.1f}%")
        
        # Position-specific breakout rates
        for pos in POSITIONS:
            pos_mask = eligible_mask & (df_with_targets['position'] == pos)
            pos_breakouts = df_with_targets.loc[pos_mask, 'is_breakout'].sum()
            pos_total = pos_mask.sum()
            if pos_total > 0:
                logger.info(f"{pos} breakout rate: {pos_breakouts/pos_total*100:.1f}%")
        
        return df_with_targets
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with NFL domain expertise."""
        logger.info("Engineering features...")
        
        eps = 1e-6
        
        # Per-game metrics
        df['fantasy_points_per_game'] = df['fantasy_points'] / (df['games'] + eps)
        df['tds_per_game'] = df['total_tds'] / (df['games'] + eps)
        
        # Experience features
        df['is_rookie'] = (df['years_in_nfl'] == 0).astype(int)
        df['is_sophomore'] = (df['years_in_nfl'] == 1).astype(int)
        df['is_third_year'] = (df['years_in_nfl'] == 2).astype(int)
        
        # Age-related features
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['young_player'] = (df['age'] < 24).astype(int)
        
        # Rushing efficiency metrics
        df['rushing_yards_per_game'] = df['rushing_yards'] / (df['games'] + eps)
        df['yards_per_carry'] = df['rushing_yards'] / (df['rush_att'] + eps)
        df['rushing_td_rate'] = df['rush_td'] / (df['rush_att'] + eps)
        
        # Receiving efficiency metrics
        df['receiving_yards_per_game'] = df['receiving_yards'] / (df['games'] + eps)
        df['targets_per_game'] = df['targets'] / (df['games'] + eps)
        df['receptions_per_game'] = df['receptions'] / (df['games'] + eps)
        df['catch_rate'] = df['receptions'] / (df['targets'] + eps)
        df['yards_per_reception'] = df['receiving_yards'] / (df['receptions'] + eps)
        df['yards_per_target'] = df['receiving_yards'] / (df['targets'] + eps)
        df['receiving_td_rate'] = df['rec_td'] / (df['targets'] + eps)
        
        # Opportunity share metrics
        df['total_touches'] = df['rush_att'] + df['receptions']
        df['touches_per_game'] = df['total_touches'] / (df['games'] + eps)
        df['target_share'] = df['targets'] / (df['games'] * 30 + eps)  # Approx 30 targets per team game
        
        # Versatility metrics
        df['rushing_share'] = df['rushing_yards'] / (df['rushing_yards'] + df['receiving_yards'] + eps)
        df['receiving_share'] = 1 - df['rushing_share']
        df['versatility_score'] = 1 - abs(df['rushing_share'] - 0.5) * 2  # Higher when balanced
        
        # Efficiency composite scores
        df['yardage_efficiency'] = (df['yards_per_carry'] * df['rushing_share'] + 
                                   df['yards_per_reception'] * df['receiving_share'])
        
        # Physical attributes (if available)
        if 'height_inches' in df.columns and 'weight' in df.columns:
            df['bmi'] = (df['weight'] / (df['height_inches'] ** 2)) * 703
            df['size_score'] = df['height_inches'] * df['weight'] / 1000  # Simple size metric
        
        # Historical performance (without leakage)
        df = df.sort_values(['player_id', 'season'])
        player_groups = df.groupby('player_id')
        
        # Previous season stats
        lag_features = ['fantasy_points', 'games', 'total_touches', 'fantasy_points_per_game']
        for feature in lag_features:
            if feature in df.columns:
                df[f'prev_{feature}'] = player_groups[feature].shift(1).fillna(0)
        
        # Career trajectory
        df['career_games'] = player_groups['games'].cumsum() - df['games']  # Exclude current season
        df['career_fantasy_points'] = player_groups['fantasy_points'].cumsum() - df['fantasy_points']
        df['career_ppg'] = df['career_fantasy_points'] / (df['career_games'] + eps)
        
        # Improvement metrics
        df['ppg_improvement'] = df['fantasy_points_per_game'] - df['prev_fantasy_points_per_game']
        df['touches_improvement'] = df['touches_per_game'] - df.get('prev_touches_per_game', 0)
        
        # Team context features (simplified - would need team data for full implementation)
        df['games_started_pct'] = df.get('games_started', 0) / (df['games'] + eps)
        
        # Position-specific features
        df['is_rb'] = (df['position'] == 'RB').astype(int)
        df['is_wr'] = (df['position'] == 'WR').astype(int)
        df['is_te'] = (df['position'] == 'TE').astype(int)
        
        # RB-specific
        df['rb_workhorse'] = df['is_rb'] * (df['touches_per_game'] > 15).astype(int)
        df['rb_receiver'] = df['is_rb'] * (df['targets_per_game'] > 3).astype(int)
        
        # WR-specific
        df['wr_target_hog'] = df['is_wr'] * (df['target_share'] > 0.2).astype(int)
        df['wr_deep_threat'] = df['is_wr'] * (df['yards_per_reception'] > 15).astype(int)
        
        # TE-specific
        df['te_redzone'] = df['is_te'] * (df['rec_td'] / (df['games'] + eps) > 0.4).astype(int)
        
        # Clean up infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        logger.info(f"Feature engineering complete. Dataset shape: {df.shape}")
        return df
    
    def select_modeling_features(self, df: pd.DataFrame, target_col: str = 'is_breakout') -> pd.DataFrame:
        """Select features for modeling with careful exclusion of problematic columns."""
        
        # Columns to exclude
        exclude_cols = [
            # Identifiers
            'player_id', 'player_name', 'team',
            # Raw text/dates
            'birth_date', 'college', 'height',
            # Temporal (leakage risk)
            'season', 'season_type',
            # Raw stats (using engineered versions)
            'rush_att', 'rush_td', 'rec_td', 'rushing_yards', 'receiving_yards',
            'targets', 'receptions', 'fumbles', 'fumbles_lost',
            # Redundant
            'position', 'years_exp', 'games_started',
            # Other
            'headshot_url', 'status', 'roster_status'
        ]
        
        # Select columns
        modeling_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure target is included
        if target_col not in modeling_cols and target_col in df.columns:
            modeling_cols.append(target_col)
        
        logger.info(f"Selected {len(modeling_cols)} features for modeling")
        
        return df[modeling_cols]
    
    def create_temporal_splits(self, df: pd.DataFrame, target_col: str, 
                              train_end: int, val_years: List[int], test_years: List[int]) -> Tuple:
        """
        Create temporal train/validation/test splits with multiple years.
        """
        logger.info(f"Creating temporal splits:")
        logger.info(f"  Training: up to {train_end}")
        logger.info(f"  Validation: {val_years}")
        logger.info(f"  Test: {test_years}")
        
        # Add season back temporarily for splitting
        modeling_df = self.select_modeling_features(df, target_col)
        modeling_df['season'] = df['season']
        
        # Create splits
        train_mask = modeling_df['season'] <= train_end
        val_mask = modeling_df['season'].isin(val_years)
        test_mask = modeling_df['season'].isin(test_years)
        
        # Split data
        train_data = modeling_df[train_mask].drop(columns=['season'])
        val_data = modeling_df[val_mask].drop(columns=['season'])
        test_data = modeling_df[test_mask].drop(columns=['season'])
        
        # Separate features and target
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        
        X_val = val_data.drop(columns=[target_col])
        y_val = val_data[target_col]
        
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]
        
        logger.info(f"Train set: {len(X_train)} samples, {y_train.sum()} positives ({y_train.mean()*100:.1f}%)")
        logger.info(f"Val set: {len(X_val)} samples, {y_val.sum()} positives ({y_val.mean()*100:.1f}%)")
        logger.info(f"Test set: {len(X_test)} samples, {y_test.sum()} positives ({y_test.mean()*100:.1f}%)")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def prepare_data_for_modeling(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, X_test: pd.DataFrame,
                                balance_method: str = 'moderate_smote') -> Tuple:
        """Prepare data for modeling with appropriate balancing."""
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Handle class imbalance
        logger.info(f"Original class distribution: {np.bincount(y_train)}")
        
        if balance_method == 'moderate_smote':
            minority_count = np.sum(y_train == 1)
            majority_count = np.sum(y_train == 0)
            
            if minority_count < 5:
                logger.warning(f"Very few positive samples ({minority_count}), using class weights instead")
                X_balanced, y_balanced = X_train_scaled, y_train
            else:
                # Target 1:3 ratio
                target_ratio = 3
                k_neighbors = min(5, minority_count - 1)
                
                try:
                    sampler = SMOTE(random_state=42, k_neighbors=k_neighbors, 
                                   sampling_strategy=1/target_ratio)
                    X_balanced, y_balanced = sampler.fit_resample(X_train_scaled, y_train)
                except Exception as e:
                    logger.warning(f"SMOTE failed: {e}. Using original data.")
                    X_balanced, y_balanced = X_train_scaled, y_train
                    
        elif balance_method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42, sampling_strategy=1/3)
            X_balanced, y_balanced = undersampler.fit_resample(X_train_scaled, y_train)
        else:
            X_balanced, y_balanced = X_train_scaled, y_train
        
        logger.info(f"Balanced class distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced, X_val_scaled, X_test_scaled
    
    def train_and_evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train and compare multiple models."""
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, solver='saga'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42, n_iter_no_change=10
            )
        }
        
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1.0],
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced', None]
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            
            try:
                # Determine CV strategy based on data size
                n_positives = np.sum(y_train == 1)
                cv_folds = min(5, n_positives // 2) if n_positives >= 10 else 2
                
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                # Grid search
                grid_search = GridSearchCV(
                    model, param_grids[name],
                    cv=cv, scoring='f1', n_jobs=1, verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Calibrate if enough data
                if len(np.unique(y_val)) > 1 and np.sum(y_val) >= 5:
                    calibrated_model = CalibratedClassifierCV(
                        best_model, cv=2, method='isotonic'
                    )
                    calibrated_model.fit(X_train, y_train)
                else:
                    calibrated_model = best_model
                
                # Predictions
                y_val_prob = calibrated_model.predict_proba(X_val)[:, 1]
                
                # Find optimal threshold
                optimal_threshold = self._find_optimal_threshold(y_val, y_val_prob)
                y_val_pred = (y_val_prob >= optimal_threshold).astype(int)
                
                # Calculate metrics
                metrics = {
                    'model': calibrated_model,
                    'threshold': optimal_threshold,
                    'accuracy': accuracy_score(y_val, y_val_pred),
                    'precision': precision_score(y_val, y_val_pred, zero_division=0),
                    'recall': recall_score(y_val, y_val_pred, zero_division=0),
                    'f1': f1_score(y_val, y_val_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_val, y_val_prob) if len(np.unique(y_val)) > 1 else 0.5,
                    'best_params': grid_search.best_params_
                }
                
                results[name] = metrics
                logger.info(f"{name} - F1: {metrics['f1']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No models were successfully trained")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
        best_model_info = results[best_model_name]
        
        logger.info(f"\nBest model: {best_model_name}")
        logger.info(f"Best F1 score: {best_model_info['f1']:.3f}")
        
        self.model = best_model_info['model']
        self.optimal_threshold = best_model_info['threshold']
        
        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return results
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find threshold that maximizes F1 score."""
        thresholds = np.linspace(0.1, 0.9, 81)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]
    
    def evaluate_final_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate the final model on test set."""
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Predictions
        y_test_prob = self.model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= self.optimal_threshold).astype(int)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else 0.5,
            'avg_precision': average_precision_score(y_test, y_test_prob) if np.sum(y_test) > 0 else 0
        }
        
        logger.info("\nFinal Test Results:")
        logger.info("=" * 40)
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        # Classification report
        logger.info(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred, 
                                  target_names=['No Breakout', 'Breakout'],
                                  zero_division=0))
        
        # Create evaluation plots
        self._create_evaluation_plots(y_test, y_test_prob, y_test_pred)
        
        # Feature importance plot
        if self.feature_importance is not None:
            self._plot_feature_importance()
        
        return metrics
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray):
        """Create comprehensive evaluation plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        if np.sum(y_true) > 0:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
            axes[0, 1].plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
            axes[0, 1].axhline(y=y_true.mean(), color='k', linestyle='--', label='Baseline')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Probability Distribution
        axes[1, 1].hist(y_prob[y_true == 0], bins=30, alpha=0.5, label='No Breakout', density=True)
        axes[1, 1].hist(y_prob[y_true == 1], bins=30, alpha=0.5, label='Breakout', density=True)
        axes[1, 1].axvline(x=self.optimal_threshold, color='r', linestyle='--', label=f'Threshold = {self.optimal_threshold:.2f}')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Probability Distribution by Class')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {self.plots_dir}")
    
    def _plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance."""
        if self.feature_importance is None:
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {self.plots_dir}")
    
    def save_model(self, model_name: str = "nfl_breakout_model"):
        """Save the trained model and artifacts."""
        
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat(),
            'constants': {
                'POSITIONS': POSITIONS,
                'RANK_THRESHOLDS': RANK_THRESHOLDS,
                'FANTASY_BREAKOUT_THRESHOLDS': FANTASY_BREAKOUT_THRESHOLDS,
                'MAX_EXPERIENCE': MAX_EXPERIENCE
            }
        }
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        joblib.dump(model_artifacts, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def predict_breakouts(self, df: pd.DataFrame, top_n: int = 50, 
                         position_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Predict breakouts for current season players.
        """
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare data
        df_processed = df.copy()
        df_processed = self.engineer_features(df_processed)
        
        # Filter eligible players
        eligible_mask = df_processed['years_in_nfl'] <= MAX_EXPERIENCE
        if position_filter:
            eligible_mask &= df_processed['position'] == position_filter
        
        eligible_players = df_processed[eligible_mask].copy()
        
        if len(eligible_players) == 0:
            logger.warning("No eligible players found!")
            return pd.DataFrame()
        
        # Handle missing features
        missing_features = [f for f in self.feature_names if f not in eligible_players.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}, filling with zeros")
            for feature in missing_features:
                eligible_players[feature] = 0
        
        # Make predictions
        X = eligible_players[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        breakout_probs = self.model.predict_proba(X_scaled)[:, 1]
        breakout_preds = (breakout_probs >= self.optimal_threshold).astype(int)
        
        # Create results
        results = pd.DataFrame({
            'player_id': eligible_players['player_id'],
            'player_name': eligible_players.get('player_name', 'Unknown'),
            'position': eligible_players.get('position', 'Unknown'),
            'team': eligible_players.get('team', 'Unknown'),
            'years_in_nfl': eligible_players['years_in_nfl'],
            'age': eligible_players.get('age', 0),
            'games': eligible_players['games'],
            'fantasy_points': eligible_players['fantasy_points'],
            'fantasy_ppg': eligible_players['fantasy_points_per_game'],
            'touches_per_game': eligible_players.get('touches_per_game', 0),
            'breakout_probability': breakout_probs,
            'breakout_prediction': breakout_preds
        })
        
        # Sort by probability
        results = results.sort_values('breakout_probability', ascending=False)
        
        # Display top candidates
        logger.info(f"\nTop {min(top_n, len(results))} Breakout Candidates:")
        logger.info("=" * 100)
        display_cols = ['player_name', 'position', 'team', 'years_in_nfl', 
                       'fantasy_ppg', 'touches_per_game', 'breakout_probability']
        print(results[display_cols].head(top_n).to_string(index=False, float_format='%.3f'))
        
        return results.head(top_n)


def main():
    """Main training and prediction pipeline."""
    
    logger.info("NFL Player Breakout Prediction Model - Enhanced Version")
    logger.info("=" * 60)
    
    try:
        # Initialize predictor
        predictor = NFLBreakoutPredictor(output_dir="./nfl_breakout_outputs_enhanced")
        
        # Load and prepare data
        logger.info("\n1. Loading and preparing data...")
        df = predictor.load_data(start_year=2003, end_year=2024)
        df = predictor.clean_data(df)
        df = predictor.create_breakout_targets(df)
        df = predictor.engineer_features(df)
        
        # Create temporal splits with multiple years for robustness
        logger.info("\n2. Creating temporal splits...")
        X_train, y_train, X_val, y_val, X_test, y_test = predictor.create_temporal_splits(
            df, target_col='is_breakout', 
            train_end=2020,
            val_years=[2021, 2022],
            test_years=[2023]
        )
        
        # Prepare data for modeling
        logger.info("\n3. Preparing data for modeling...")
        X_train_balanced, y_train_balanced, X_val_scaled, X_test_scaled = predictor.prepare_data_for_modeling(
            X_train, y_train, X_val, X_test, balance_method='moderate_smote'
        )
        
        # Train and compare models
        logger.info("\n4. Training models...")
        model_results = predictor.train_and_evaluate_models(
            X_train_balanced, y_train_balanced, X_val_scaled, y_val
        )
        
        # Final evaluation
        logger.info("\n5. Final evaluation...")
        final_metrics = predictor.evaluate_final_model(X_test_scaled, y_test)
        
        # Save model
        logger.info("\n6. Saving model...")
        model_path = predictor.save_model("nfl_breakout_predictor_enhanced")
        
        # Make predictions for 2025
        logger.info("\n7. Making predictions for 2025 season...")
        
        # Load 2024 data
        df_2024 = predictor.load_data(start_year=2024, end_year=2024)
        df_2024 = predictor.clean_data(df_2024)
        
        # Predictions by position
        all_predictions = []
        for position in POSITIONS:
            logger.info(f"\n{position} Breakout Candidates:")
            pos_predictions = predictor.predict_breakouts(df_2024, top_n=15, position_filter=position)
            all_predictions.append(pos_predictions)
        
        # Combine and save
        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            predictions_file = os.path.join(predictor.output_dir, "breakout_predictions_2025.csv")
            combined_predictions.to_csv(predictions_file, index=False)
            logger.info(f"\nPredictions saved to {predictions_file}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Model files saved in: {predictor.models_dir}")
        logger.info(f"Plots saved in: {predictor.plots_dir}")
        logger.info(f"Predictions saved in: {predictor.output_dir}")
        
        return predictor, final_metrics, combined_predictions if all_predictions else None
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    predictor, metrics, predictions = main()
