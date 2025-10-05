"""
Example Usage of USLC Goalkeeper Moneyball System

This script demonstrates how to use the system with sample data.
For production use with real Impect API data, see the comments.
"""

import pandas as pd
import os
import sys

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 30)


def example_1_generate_sample_data():
    """Example 1: Generate sample data for testing"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Generate Sample Data")
    print("=" * 80)
    
    from generate_sample_data import save_sample_data
    
    print("\nGenerating sample goalkeeper data for all USLC iterations...")
    data = save_sample_data()
    
    print(f"\nGenerated {len(data)} goalkeeper records")
    print(f"Years: {sorted(data['year'].unique())}")
    print(f"\nSample records:")
    print(data[['playerName', 'teamName', 'year', 'saves_per_90', 'save_percentage', 'clean_sheets']].head(10))


def example_2_score_goalkeepers():
    """Example 2: Score goalkeepers using mathematical metrics"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Score Goalkeepers")
    print("=" * 80)
    
    from data_collector import GoalkeeperDataCollector
    from scorer import GoalkeeperScorer
    
    # Load data
    collector = GoalkeeperDataCollector(use_cache=True)
    data = collector.collect_training_data()
    
    print(f"\nLoaded {len(data)} goalkeeper records")
    
    # Calculate scores
    scorer = GoalkeeperScorer()
    scored_data = scorer.score_goalkeepers(data)
    
    print("\nComposite Score Statistics:")
    print(scored_data['composite_score'].describe())
    
    # Get top performers
    print("\n" + "-" * 80)
    print("Top 10 Goalkeepers (All Years):")
    print("-" * 80)
    top_10 = scorer.get_top_performers(scored_data, n=10, min_minutes=900)
    
    display_cols = ['playerName', 'teamName', 'year', 'composite_score', 
                    'shot_stopping_score', 'distribution_score', 'reliability_score']
    display_cols = [col for col in display_cols if col in top_10.columns]
    
    print(top_10[display_cols].to_string(index=False))
    
    # Top performers by year
    for year in sorted(scored_data['year'].unique()):
        print(f"\n{'-' * 80}")
        print(f"Top 5 Goalkeepers in {year}:")
        print("-" * 80)
        year_data = scored_data[scored_data['year'] == year]
        top_5 = scorer.get_top_performers(year_data, n=5, min_minutes=900)
        print(top_5[display_cols].to_string(index=False))


def example_3_train_model():
    """Example 3: Train machine learning model"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Train Prediction Model")
    print("=" * 80)
    
    from data_collector import GoalkeeperDataCollector
    from scorer import GoalkeeperScorer
    from predictor import GoalkeeperPredictor
    
    # Load and score data
    collector = GoalkeeperDataCollector(use_cache=True)
    data = collector.collect_training_data()
    
    scorer = GoalkeeperScorer()
    scored_data = scorer.score_goalkeepers(data)
    
    # Train model
    print("\nTraining Random Forest model...")
    predictor = GoalkeeperPredictor()
    metrics = predictor.train(scored_data, model_type='random_forest')
    
    print("\nModel Performance Metrics:")
    print("-" * 80)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")
    
    # Feature importance
    print("\n" + "-" * 80)
    print("Top 15 Most Important Metrics:")
    print("-" * 80)
    importance = predictor.get_feature_importance(top_n=15)
    for idx, row in importance.iterrows():
        print(f"{row['feature']:40s}: {row['importance']:.4f}")


def example_4_identify_targets():
    """Example 4: Identify goalkeeper targets"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Identify Recruitment Targets")
    print("=" * 80)
    
    from data_collector import GoalkeeperDataCollector
    from scorer import GoalkeeperScorer
    from predictor import GoalkeeperPredictor
    
    # Load and score data
    collector = GoalkeeperDataCollector(use_cache=True)
    data = collector.collect_training_data()
    
    scorer = GoalkeeperScorer()
    scored_data = scorer.score_goalkeepers(data)
    
    # Train model
    predictor = GoalkeeperPredictor()
    predictor.train(scored_data, model_type='random_forest')
    
    # Identify targets for each year
    for year in sorted(scored_data['year'].unique()):
        print(f"\n{'-' * 80}")
        print(f"Top 10 Recruitment Targets for {year}:")
        print("-" * 80)
        
        year_data = scored_data[scored_data['year'] == year]
        targets = predictor.identify_targets(year_data, n_targets=10, min_minutes=900)
        
        display_cols = ['playerName', 'teamName', 'predicted_score', 'composite_score']
        if 'age' in targets.columns:
            display_cols.insert(2, 'age')
        display_cols = [col for col in display_cols if col in targets.columns]
        
        print(targets[display_cols].to_string(index=False))


def example_5_full_analysis():
    """Example 5: Run complete analysis pipeline"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Complete Moneyball Analysis")
    print("=" * 80)
    
    from moneyball import GoalkeeperMoneyball
    
    # Run full analysis
    moneyball = GoalkeeperMoneyball(use_cache=True)
    report = moneyball.run_full_analysis()
    
    print("\n" + "=" * 80)
    print("Analysis Results Summary")
    print("=" * 80)
    
    print("\nModel Performance:")
    for metric, value in report['training_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nTop 5 Most Important Metrics:")
    for idx, row in report['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nTop 5 ML-Derived Weights (Data-Driven):")
    if report.get('ml_weights'):
        sorted_weights = sorted(report['ml_weights'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for metric, weight in sorted_weights:
            print(f"  {metric}: {weight:.4f}")
    
    print(f"\nResults exported to 'output/' directory")
    print("  - goalkeeper_scores_all_years.csv")
    print("  - ml_derived_weights.csv (NEW - data-driven weights!)")
    print("  - feature_importance.csv")
    print("  - top_performers_YYYY.csv (for each year)")
    print("  - targets_2025.csv")
    print("  - goalkeeper_model.pkl")


def example_6_production_usage():
    """Example 6: Production usage with real Impect API"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Production Usage with Real Impect API")
    print("=" * 80)
    
    print("""
This example shows how to use the system with real Impect API data.

IMPORTANT: This requires network access to the Impect API.

NEW: The system now uses DATA-DRIVEN weights (ML feature importance)
     instead of manual subjective weights!

Code:
-----

from moneyball import GoalkeeperMoneyball

# Initialize with data-driven weights (recommended)
# This will:
# 1. Load ALL historical data (2022-2025)
# 2. Score with equal weights (no bias)
# 3. Train ML model to learn feature importance
# 4. Re-score using ML-derived weights
moneyball = GoalkeeperMoneyball(use_cache=False, use_data_driven_weights=True)

# Run complete analysis
report = moneyball.run_full_analysis()

# Access specific results
training_metrics = report['training_metrics']
feature_importance = report['feature_importance']
ml_weights = report['ml_weights']  # NEW: Data-driven weights!
top_performers = report['top_performers_by_year'][2025]
targets = report['targets_2025']

# Export results (includes ml_derived_weights.csv)
moneyball.export_results(report)

# The ml_derived_weights.csv shows which metrics matter most
# based on DATA, not human opinion!

Note: When using real API data, the system will:
1. Authenticate with Impect API using provided credentials
2. Fetch data from ALL training iterations (2022-2025)
3. Cache the data for faster subsequent runs
4. Train the model on ALL historical data
5. Generate predictions and rankings using data-driven weights
    """)


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("USLC GOALKEEPER MONEYBALL SYSTEM - EXAMPLES")
    print("=" * 80)
    
    examples = [
        ("Generate Sample Data", example_1_generate_sample_data),
        ("Score Goalkeepers", example_2_score_goalkeepers),
        ("Train Model", example_3_train_model),
        ("Identify Targets", example_4_identify_targets),
        ("Full Analysis", example_5_full_analysis),
        ("Production Usage", example_6_production_usage),
    ]
    
    if len(sys.argv) > 1:
        # Run specific example
        try:
            example_num = int(sys.argv[1])
            if 1 <= example_num <= len(examples):
                name, func = examples[example_num - 1]
                print(f"\nRunning: {name}")
                func()
            else:
                print(f"Error: Example number must be between 1 and {len(examples)}")
        except ValueError:
            print("Error: Please provide a valid example number")
    else:
        # Show menu
        print("\nAvailable examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"  {i}. {name}")
        print("\nUsage:")
        print("  python example.py [example_number]")
        print("\nOr run all examples:")
        print("  python example.py all")
        
        if len(sys.argv) == 1:
            print("\nRunning Example 1 by default...")
            example_1_generate_sample_data()


if __name__ == "__main__":
    main()
