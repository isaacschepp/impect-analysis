"""
Configuration file for the USLC Goalkeeper Moneyball System
"""

# Impect API Credentials
IMPECT_EMAIL = "isaac.schepp@gmail.com"
IMPECT_PASSWORD = "ZJnpgKNSQkm9A_G"

# USLC Iterations for training
TRAINING_ITERATIONS = {
    2025: 1236,
    2024: 893,
    2023: 642,
    2022: 510
}

# Goalkeeper position codes (adjust based on actual Impect data)
GOALKEEPER_POSITIONS = ['GK', 'Goalkeeper', 'G']

# Key goalkeeper metrics for mathematical analysis
# All metrics are objective and quantifiable
GOALKEEPER_METRICS = {
    # Shot stopping
    'saves': 1.0,
    'saves_per_90': 1.0,
    'save_percentage': 1.5,
    'shots_on_target_against': 0.5,
    'goals_conceded': -1.0,
    'goals_conceded_per_90': -1.2,
    'expected_goals_against': 0.8,
    'goals_prevented': 1.5,  # xG prevented (saves above expected)
    
    # Distribution and passing
    'passes_completed': 0.5,
    'pass_completion_percentage': 0.8,
    'long_passes_completed': 0.4,
    'long_pass_completion_percentage': 0.6,
    'passes_into_final_third': 0.7,
    'goal_kicks': 0.2,
    'goal_kick_completion_percentage': 0.5,
    
    # Sweeping and positioning
    'defensive_actions_outside_penalty_area': 1.0,
    'avg_distance_from_goal': 0.3,
    'successful_sweeper_actions': 1.2,
    
    # Claiming crosses and high balls
    'crosses_claimed': 0.9,
    'cross_claim_percentage': 1.1,
    'punches': 0.3,
    'high_ball_wins': 0.8,
    
    # Ball retention and build-up
    'touches': 0.2,
    'passes_received': 0.3,
    'progressive_passes': 0.9,
    
    # Error prevention (negative weights for mistakes)
    'errors_leading_to_shot': -2.0,
    'errors_leading_to_goal': -3.0,
    'penalties_conceded': -1.5,
    
    # Match impact
    'clean_sheets': 2.0,
    'clean_sheet_percentage': 1.8,
    'minutes_played': 0.1,
}

# Machine learning parameters
ML_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_top_features': 20
}

# Output configuration
OUTPUT_DIR = 'output'
DATA_CACHE_DIR = 'data/cache'
