"""
Main analysis pipeline for USLC Goalkeeper Moneyball System
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List
from data_collector import GoalkeeperDataCollector
from scorer import GoalkeeperScorer
from predictor import GoalkeeperPredictor
from config import TRAINING_ITERATIONS, OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalkeeperMoneyball:
    """
    Main class orchestrating the goalkeeper moneyball analysis
    """
    
    def __init__(self, use_cache: bool = True, use_data_driven_weights: bool = True):
        """
        Initialize the moneyball system
        
        Args:
            use_cache: Whether to use cached data
            use_data_driven_weights: If True, use ML-derived feature importance as weights
                                    instead of manual subjective weights (recommended)
        """
        self.collector = GoalkeeperDataCollector(use_cache=use_cache)
        self.scorer = GoalkeeperScorer()
        self.predictor = GoalkeeperPredictor()
        self.training_data = None
        self.scored_data = None
        self.use_data_driven_weights = use_data_driven_weights
        self.ml_weights = None  # Will store ML-derived weights
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare training data from all USLC iterations
        
        Returns:
            Prepared DataFrame
        """
        logger.info("=" * 80)
        logger.info("LOADING TRAINING DATA")
        logger.info("=" * 80)
        
        # Collect data from all training iterations
        self.training_data = self.collector.collect_training_data()
        
        logger.info(f"Loaded {len(self.training_data)} goalkeeper records")
        logger.info(f"Years covered: {sorted(self.training_data['year'].unique().tolist())}")
        logger.info(f"Columns available: {self.training_data.columns.tolist()}")
        
        return self.training_data
    
    def score_goalkeepers(self, use_ml_weights: bool = False) -> pd.DataFrame:
        """
        Calculate mathematical scores for all goalkeepers
        
        Args:
            use_ml_weights: If True, use ML-derived weights; if False, use initial weights
        
        Returns:
            DataFrame with scores
        """
        logger.info("=" * 80)
        logger.info("SCORING GOALKEEPERS")
        logger.info("=" * 80)
        
        if self.training_data is None:
            self.load_and_prepare_data()
        
        if use_ml_weights and self.ml_weights is not None:
            logger.info("Using ML-derived data-driven weights")
            self.scorer = GoalkeeperScorer(metric_weights=self.ml_weights)
        else:
            logger.info("Using initial weights for ML training")
            self.scorer = GoalkeeperScorer(use_equal_weights=True)
        
        self.scored_data = self.scorer.score_goalkeepers(self.training_data)
        
        logger.info("Scoring complete!")
        logger.info(f"Score statistics:")
        logger.info(self.scored_data['composite_score'].describe())
        
        return self.scored_data
    
    def train_model(self, model_type: str = 'random_forest', 
                   target_metric: str = 'clean_sheet_percentage') -> Dict[str, float]:
        """
        Train the machine learning model to predict ACTUAL SUCCESS
        
        SUCCESS DEFINITION:
        Instead of predicting a circular composite_score (which is based on the same 
        metrics we use as features), we predict REAL GAME OUTCOMES:
        
        - clean_sheet_percentage (default): How often does the goalkeeper help keep 
          a clean sheet? This is a TRUE measure of success - did they prevent goals?
        
        - goals_prevented: How many goals did they save beyond expectation? 
          (Expected goals against minus actual goals conceded)
        
        - goals_conceded_per_90: Direct measure of defensive success (lower is better)
        
        This approach answers: "What goalkeeper behaviors actually lead to winning outcomes?"
        The ML model learns which metrics (saves, distribution, aerial ability, etc.) 
        correlate with these real-world results.
        
        Args:
            model_type: Type of model to train
            target_metric: The success metric to predict (should be a real outcome)
            
        Returns:
            Training metrics
        """
        logger.info("=" * 80)
        logger.info("TRAINING MACHINE LEARNING MODEL")
        logger.info(f"SUCCESS METRIC: {target_metric}")
        logger.info("This metric represents ACTUAL GAME OUTCOMES, not subjective scores")
        logger.info("=" * 80)
        
        if self.scored_data is None:
            self.score_goalkeepers()
        
        # Check if target metric exists in data
        if target_metric not in self.scored_data.columns:
            logger.warning(f"Target metric '{target_metric}' not found in data")
            logger.warning("Available metrics: " + ", ".join(self.scored_data.columns.tolist()[:20]))
            logger.info("Falling back to composite_score (not ideal - circular dependency)")
            target_metric = 'composite_score'
        
        metrics = self.predictor.train(self.scored_data, target_col=target_metric, 
                                      model_type=model_type)
        
        logger.info("Model training metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def analyze_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analyze which metrics are most important for goalkeeper performance
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        logger.info("=" * 80)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("=" * 80)
        
        importance = self.predictor.get_feature_importance(top_n)
        
        logger.info(f"\nTop {top_n} most important metrics:")
        for idx, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance
    
    def extract_ml_weights(self) -> Dict[str, float]:
        """
        Extract ML-derived feature importance as weights for scoring.
        This replaces subjective manual weights with data-driven weights.
        
        Returns:
            Dictionary mapping metric names to ML-derived importance weights
        """
        logger.info("=" * 80)
        logger.info("EXTRACTING DATA-DRIVEN WEIGHTS FROM ML MODEL")
        logger.info("=" * 80)
        
        if self.predictor.feature_importance is None:
            raise ValueError("Model must be trained first to extract weights")
        
        # Convert feature importance to weights dictionary
        ml_weights = {}
        for _, row in self.predictor.feature_importance.iterrows():
            feature = row['feature']
            importance = row['importance']
            ml_weights[feature] = float(importance)
        
        # Normalize weights to sum to a reasonable scale (similar to manual weights)
        # This maintains interpretability while being data-driven
        total_importance = sum(ml_weights.values())
        if total_importance > 0:
            # Scale to match the scale of manual weights (roughly 30-40 total)
            target_total = 35.0
            scale_factor = target_total / total_importance
            ml_weights = {k: v * scale_factor for k, v in ml_weights.items()}
        
        logger.info(f"Extracted {len(ml_weights)} data-driven weights")
        logger.info("Top 10 ML-derived weights:")
        sorted_weights = sorted(ml_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        for metric, weight in sorted_weights:
            logger.info(f"  {metric}: {weight:.4f}")
        
        self.ml_weights = ml_weights
        return ml_weights
    
    def identify_top_targets(self, target_year: int = 2025, 
                            n_targets: int = 20) -> pd.DataFrame:
        """
        Identify top goalkeeper targets for a specific year
        
        Args:
            target_year: Year to analyze
            n_targets: Number of targets to identify
            
        Returns:
            DataFrame with top targets
        """
        logger.info("=" * 80)
        logger.info(f"IDENTIFYING TOP {n_targets} GOALKEEPER TARGETS FOR {target_year}")
        logger.info("=" * 80)
        
        # Get data for target year
        if target_year in TRAINING_ITERATIONS:
            target_data = self.collector.collect_iteration_data(target_year)
            
            # Score the goalkeepers
            scored_target_data = self.scorer.score_goalkeepers(target_data)
            
            # Use model to identify targets
            targets = self.predictor.identify_targets(scored_target_data, n_targets)
            
            logger.info(f"\nTop {n_targets} targets identified:")
            logger.info(targets.to_string())
            
            return targets
        else:
            logger.error(f"Year {target_year} not in training iterations")
            return pd.DataFrame()
    
    def generate_report(self, success_metric: str = 'clean_sheet_percentage') -> Dict:
        """
        Generate comprehensive analysis report using data-driven methodology
        
        SUCCESS METRIC:
        By default, the system predicts 'clean_sheet_percentage' - a real outcome
        that shows how often a goalkeeper helps their team avoid conceding goals.
        This is NOT a circular metric based on our input features; it's an actual
        game result that answers: "Does doing X or Y actually help win games?"
        
        If use_data_driven_weights is True (recommended), this uses a two-phase approach:
        Phase 1: Score with initial weights and train ML model to learn feature importance
        Phase 2: Re-score using ML-derived weights for final results
        
        Args:
            success_metric: The outcome metric to predict (default: clean_sheet_percentage)
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("=" * 80)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info(f"SUCCESS METRIC: {success_metric}")
        if self.use_data_driven_weights:
            logger.info("Using DATA-DRIVEN methodology (ML-derived weights)")
        else:
            logger.info("Using MANUAL weights")
        logger.info("=" * 80)
        
        report = {
            'training_metrics': {},
            'feature_importance': None,
            'ml_weights': None,
            'top_performers_by_year': {},
            'targets_2025': None,
            'success_metric': success_metric
        }
        
        # Load data
        self.load_and_prepare_data()
        
        if self.use_data_driven_weights:
            # PHASE 1: Initial scoring with equal weights to train ML model
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1: Initial scoring to train ML model")
            logger.info("=" * 80)
            self.score_goalkeepers(use_ml_weights=False)
            
            # Train model to learn feature importance using REAL SUCCESS METRIC
            report['training_metrics'] = self.train_model(target_metric=success_metric)
            
            # Extract data-driven weights from ML model
            report['ml_weights'] = self.extract_ml_weights()
            
            # PHASE 2: Re-score using ML-derived weights
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2: Final scoring with data-driven weights")
            logger.info("=" * 80)
            self.score_goalkeepers(use_ml_weights=True)
            
            # Feature importance
            report['feature_importance'] = self.analyze_feature_importance()
        else:
            # Traditional approach with manual weights
            self.score_goalkeepers(use_ml_weights=False)
            report['training_metrics'] = self.train_model(target_metric=success_metric)
            report['feature_importance'] = self.analyze_feature_importance()
        
        # Top performers by year
        for year in sorted(self.scored_data['year'].unique()):
            year_data = self.scored_data[self.scored_data['year'] == year]
            top_performers = self.scorer.get_top_performers(year_data, n=10)
            report['top_performers_by_year'][year] = top_performers
            
            logger.info(f"\nTop 10 performers in {year}:")
            logger.info(top_performers[['composite_score']].describe())
        
        # Identify targets for 2025
        if 2025 in TRAINING_ITERATIONS:
            report['targets_2025'] = self.identify_top_targets(2025, 20)
        
        return report
    
    def export_results(self, report: Dict):
        """
        Export analysis results to files
        
        Args:
            report: Report dictionary from generate_report()
        """
        logger.info("=" * 80)
        logger.info("EXPORTING RESULTS")
        logger.info("=" * 80)
        
        # Export scored data
        if self.scored_data is not None:
            self.collector.export_to_csv(
                self.scored_data, 
                'goalkeeper_scores_all_years.csv'
            )
        
        # Export success metric info
        if report.get('success_metric'):
            success_info = pd.DataFrame([{
                'success_metric': report['success_metric'],
                'description': 'The target variable the ML model predicts - represents ACTUAL GAME OUTCOMES',
                'why_it_matters': 'This is not a circular composite score. It measures real success: clean sheets, goals prevented, etc.',
                'interpretation': 'The model learns which goalkeeper actions correlate with this real outcome'
            }])
            success_info.to_csv(
                os.path.join(OUTPUT_DIR, 'success_metric_explanation.csv'),
                index=False
            )
            logger.info(f"Success metric explanation exported")
        
        # Export ML-derived weights if available
        if report.get('ml_weights') is not None:
            ml_weights_df = pd.DataFrame([
                {'metric': k, 'ml_weight': v} 
                for k, v in sorted(report['ml_weights'].items(), 
                                  key=lambda x: x[1], reverse=True)
            ])
            ml_weights_df.to_csv(
                os.path.join(OUTPUT_DIR, 'ml_derived_weights.csv'),
                index=False
            )
            logger.info(f"ML-derived weights exported (data-driven approach)")
        
        # Export feature importance
        if report['feature_importance'] is not None:
            report['feature_importance'].to_csv(
                os.path.join(OUTPUT_DIR, 'feature_importance.csv'),
                index=False
            )
            logger.info(f"Feature importance exported")
        
        # Export top performers by year
        for year, performers in report['top_performers_by_year'].items():
            performers.to_csv(
                os.path.join(OUTPUT_DIR, f'top_performers_{year}.csv'),
                index=False
            )
            logger.info(f"Top performers for {year} exported")
        
        # Export 2025 targets
        if report['targets_2025'] is not None:
            report['targets_2025'].to_csv(
                os.path.join(OUTPUT_DIR, 'targets_2025.csv'),
                index=False
            )
            logger.info(f"2025 targets exported")
        
        # Save model
        self.predictor.save_model(os.path.join(OUTPUT_DIR, 'goalkeeper_model.pkl'))
        
        logger.info("All results exported successfully!")
    
    def run_full_analysis(self, success_metric: str = 'clean_sheet_percentage'):
        """
        Run the complete moneyball analysis pipeline
        
        SUCCESS DEFINITION:
        The system predicts real game outcomes (clean_sheet_percentage by default),
        NOT a circular composite score. This answers: "What actions actually lead to success?"
        
        Args:
            success_metric: The outcome to predict (default: clean_sheet_percentage)
        """
        logger.info("\n" + "=" * 80)
        logger.info("USLC GOALKEEPER MONEYBALL SYSTEM")
        logger.info("100% Mathematical, Data-Driven Analysis")
        logger.info(f"SUCCESS METRIC: {success_metric} (REAL GAME OUTCOME)")
        if self.use_data_driven_weights:
            logger.info("Using ML-Derived Weights (NO SUBJECTIVE WEIGHTS)")
        logger.info("Training on ALL historical data: " + ", ".join(map(str, sorted(TRAINING_ITERATIONS.keys()))))
        logger.info("=" * 80 + "\n")
        
        # Generate report
        report = self.generate_report(success_metric=success_metric)
        
        # Export results
        self.export_results(report)
        
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info(f"Model trained to predict: {success_metric}")
        if self.use_data_driven_weights:
            logger.info("Weights were determined by the data, not subjective choices")
        logger.info("=" * 80 + "\n")
        
        return report


if __name__ == "__main__":
    # Run the analysis with data-driven weights (recommended)
    # This uses ML feature importance instead of manual subjective weights
    moneyball = GoalkeeperMoneyball(use_cache=True, use_data_driven_weights=True)
    moneyball.run_full_analysis()
    
    # To use manual weights instead (not recommended):
    # moneyball = GoalkeeperMoneyball(use_cache=True, use_data_driven_weights=False)
    # moneyball.run_full_analysis()
