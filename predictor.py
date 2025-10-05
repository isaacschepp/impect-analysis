"""
Machine Learning Model for Goalkeeper Performance Prediction
Uses historical data to identify high-potential goalkeepers
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
import logging
from typing import List, Tuple, Dict
from config import ML_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalkeeperPredictor:
    """
    Machine learning model to predict goalkeeper performance and identify targets
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = None
        self.trained = False
    
    def _prepare_features(self, data: pd.DataFrame, 
                         target_col: str = 'composite_score') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning
        
        Args:
            data: Input DataFrame
            target_col: Name of target column to predict
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Select numeric columns as features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and identifier columns
        exclude_cols = [
            target_col, 'playerId', 'iteration_id', 'year',
            'matchId', 'teamId'
        ]
        
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove normalized and score columns if we're using raw metrics
        feature_cols = [col for col in feature_cols 
                       if not col.endswith('_normalized') 
                       and not col.endswith('_score')
                       and col != target_col]
        
        # Handle missing values
        X = data[feature_cols].fillna(data[feature_cols].median())
        y = data[target_col]
        
        self.feature_names = feature_cols
        logger.info(f"Using {len(feature_cols)} features for modeling")
        
        return X, y
    
    def train(self, data: pd.DataFrame, target_col: str = 'composite_score',
             model_type: str = 'random_forest') -> Dict[str, float]:
        """
        Train the machine learning model
        
        Args:
            data: Training data with features and target
            target_col: Name of column to predict
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {model_type} model...")
        
        # Prepare data
        X, y = self._prepare_features(data, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=ML_PARAMS['test_size'],
            random_state=ML_PARAMS['random_state']
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=ML_PARAMS['random_state'],
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=ML_PARAMS['random_state']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=ML_PARAMS['cv_folds'],
            scoring='r2'
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.trained = True
        
        logger.info("Model training complete!")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"CV R² (mean ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for new goalkeeper data
        
        Args:
            data: DataFrame with goalkeeper features
            
        Returns:
            Array of predicted scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = data[self.feature_names].fillna(data[self.feature_names].median())
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def identify_targets(self, current_data: pd.DataFrame, 
                        n_targets: int = 20,
                        min_minutes: int = 450) -> pd.DataFrame:
        """
        Identify top goalkeeper targets for recruitment
        
        Args:
            current_data: Current goalkeeper data to evaluate
            n_targets: Number of targets to identify
            min_minutes: Minimum minutes played to qualify
            
        Returns:
            DataFrame with top targets and their predicted scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before identifying targets")
        
        # Filter by minimum minutes
        if 'minutes_played' in current_data.columns:
            qualified = current_data[current_data['minutes_played'] >= min_minutes].copy()
        else:
            qualified = current_data.copy()
        
        # Make predictions
        qualified['predicted_score'] = self.predict(qualified)
        
        # Sort by predicted score
        targets = qualified.nlargest(n_targets, 'predicted_score')
        
        # Select relevant columns for output
        output_cols = ['playerId', 'predicted_score']
        if 'playerName' in targets.columns:
            output_cols.insert(1, 'playerName')
        if 'teamName' in targets.columns:
            output_cols.insert(2, 'teamName')
        if 'age' in targets.columns:
            output_cols.append('age')
        if 'minutes_played' in targets.columns:
            output_cols.append('minutes_played')
        if 'composite_score' in targets.columns:
            output_cols.append('composite_score')
        
        output_cols = [col for col in output_cols if col in targets.columns]
        
        return targets[output_cols]
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get the most important features for the model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.trained = True
        
        logger.info(f"Model loaded from {filepath}")
