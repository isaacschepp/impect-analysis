"""
Mathematical Scoring System for Goalkeeper Evaluation
100% objective, data-driven evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from config import GOALKEEPER_METRICS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalkeeperScorer:
    """
    Calculates objective mathematical scores for goalkeepers
    Based purely on statistical metrics, no subjective evaluation
    """
    
    def __init__(self, metric_weights: Dict[str, float] = None):
        """
        Initialize the scorer
        
        Args:
            metric_weights: Dictionary mapping metric names to their weights
                          (defaults to config.GOALKEEPER_METRICS)
        """
        self.metric_weights = metric_weights or GOALKEEPER_METRICS
        self.available_metrics = []
        self.normalized_stats = None
    
    def normalize_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all metrics to 0-1 scale for fair comparison
        Uses min-max normalization
        
        Args:
            data: DataFrame containing goalkeeper metrics
            
        Returns:
            DataFrame with normalized metrics
        """
        normalized_data = data.copy()
        
        # Identify available metrics in the data
        self.available_metrics = [
            metric for metric in self.metric_weights.keys()
            if metric in data.columns
        ]
        
        logger.info(f"Found {len(self.available_metrics)} available metrics: {self.available_metrics}")
        
        # Normalize each metric
        for metric in self.available_metrics:
            values = data[metric].copy()
            
            # Handle missing values
            values = values.fillna(values.median())
            
            # Skip if all values are the same
            if values.std() == 0:
                normalized_data[f'{metric}_normalized'] = 0.5
                continue
            
            # For negative-weighted metrics (like goals conceded), invert the scale
            if self.metric_weights[metric] < 0:
                # Lower is better, so invert
                min_val = values.min()
                max_val = values.max()
                if max_val != min_val:
                    normalized_data[f'{metric}_normalized'] = 1 - (values - min_val) / (max_val - min_val)
                else:
                    normalized_data[f'{metric}_normalized'] = 0.5
            else:
                # Higher is better, standard normalization
                min_val = values.min()
                max_val = values.max()
                if max_val != min_val:
                    normalized_data[f'{metric}_normalized'] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_data[f'{metric}_normalized'] = 0.5
        
        self.normalized_stats = normalized_data
        return normalized_data
    
    def calculate_composite_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted composite score for each goalkeeper
        
        Args:
            data: DataFrame with normalized metrics
            
        Returns:
            DataFrame with composite scores added
        """
        scored_data = data.copy()
        
        # Calculate weighted score
        total_weight = 0
        weighted_sum = pd.Series(0, index=data.index)
        
        for metric in self.available_metrics:
            normalized_col = f'{metric}_normalized'
            if normalized_col in data.columns:
                weight = abs(self.metric_weights[metric])
                weighted_sum += data[normalized_col] * weight
                total_weight += weight
        
        # Normalize to 0-100 scale
        if total_weight > 0:
            scored_data['composite_score'] = (weighted_sum / total_weight) * 100
        else:
            scored_data['composite_score'] = 50
        
        return scored_data
    
    def calculate_category_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate scores for different goalkeeper skill categories
        
        Args:
            data: DataFrame with normalized metrics
            
        Returns:
            DataFrame with category scores
        """
        scored_data = data.copy()
        
        # Define metric categories
        categories = {
            'shot_stopping': [
                'saves', 'saves_per_90', 'save_percentage', 
                'goals_prevented', 'expected_goals_against'
            ],
            'distribution': [
                'passes_completed', 'pass_completion_percentage',
                'long_passes_completed', 'long_pass_completion_percentage',
                'passes_into_final_third', 'goal_kick_completion_percentage',
                'progressive_passes'
            ],
            'sweeping': [
                'defensive_actions_outside_penalty_area',
                'successful_sweeper_actions'
            ],
            'aerial': [
                'crosses_claimed', 'cross_claim_percentage',
                'high_ball_wins', 'punches'
            ],
            'reliability': [
                'clean_sheets', 'clean_sheet_percentage',
                'errors_leading_to_shot', 'errors_leading_to_goal'
            ]
        }
        
        # Calculate score for each category
        for category_name, category_metrics in categories.items():
            available_category_metrics = [
                m for m in category_metrics 
                if m in self.available_metrics and f'{m}_normalized' in data.columns
            ]
            
            if available_category_metrics:
                category_score = pd.Series(0, index=data.index)
                total_weight = 0
                
                for metric in available_category_metrics:
                    weight = abs(self.metric_weights.get(metric, 1.0))
                    category_score += data[f'{metric}_normalized'] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    scored_data[f'{category_name}_score'] = (category_score / total_weight) * 100
                else:
                    scored_data[f'{category_name}_score'] = 50
            else:
                scored_data[f'{category_name}_score'] = np.nan
        
        return scored_data
    
    def score_goalkeepers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to score all goalkeepers
        
        Args:
            data: Raw goalkeeper data
            
        Returns:
            DataFrame with all scores calculated
        """
        logger.info("Normalizing metrics...")
        normalized_data = self.normalize_metrics(data)
        
        logger.info("Calculating composite scores...")
        scored_data = self.calculate_composite_score(normalized_data)
        
        logger.info("Calculating category scores...")
        scored_data = self.calculate_category_scores(scored_data)
        
        logger.info("Scoring complete!")
        return scored_data
    
    def get_top_performers(self, scored_data: pd.DataFrame, n: int = 10, 
                          min_minutes: int = 450) -> pd.DataFrame:
        """
        Get top performing goalkeepers
        
        Args:
            scored_data: DataFrame with calculated scores
            n: Number of top performers to return
            min_minutes: Minimum minutes played to qualify
            
        Returns:
            DataFrame with top performers
        """
        # Filter by minimum minutes if available
        if 'minutes_played' in scored_data.columns:
            qualified = scored_data[scored_data['minutes_played'] >= min_minutes]
        else:
            qualified = scored_data
        
        # Sort by composite score
        top_performers = qualified.nlargest(n, 'composite_score')
        
        return top_performers
    
    def compare_goalkeepers(self, scored_data: pd.DataFrame, 
                           player_ids: List[int]) -> pd.DataFrame:
        """
        Compare specific goalkeepers side by side
        
        Args:
            scored_data: DataFrame with calculated scores
            player_ids: List of player IDs to compare
            
        Returns:
            DataFrame with comparison data
        """
        comparison = scored_data[scored_data['playerId'].isin(player_ids)]
        
        # Sort by composite score
        comparison = comparison.sort_values('composite_score', ascending=False)
        
        return comparison
