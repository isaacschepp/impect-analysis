"""
Sample Data Generator for Testing
Generates realistic goalkeeper statistics for demonstration
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import os
from config import TRAINING_ITERATIONS, GOALKEEPER_METRICS, DATA_CACHE_DIR


def generate_goalkeeper_name(idx: int) -> str:
    """Generate realistic goalkeeper names"""
    first_names = [
        "Alex", "Ben", "Carlos", "David", "Eric", "Frank", "George", "Henry",
        "Ivan", "Jake", "Kyle", "Luis", "Matt", "Nick", "Oscar", "Paul",
        "Quinn", "Ryan", "Sam", "Tom", "Victor", "Will", "Xavier", "Yuri", "Zach"
    ]
    last_names = [
        "Anderson", "Brown", "Clark", "Davis", "Evans", "Foster", "Garcia",
        "Harris", "Johnson", "King", "Lewis", "Martinez", "Nelson", "O'Brien",
        "Parker", "Quinn", "Rodriguez", "Smith", "Taylor", "Valdez", "Wilson",
        "Young", "Zhang"
    ]
    
    first = first_names[idx % len(first_names)]
    last = last_names[(idx // len(first_names)) % len(last_names)]
    return f"{first} {last}"


def generate_team_name(idx: int) -> str:
    """Generate team names"""
    cities = [
        "Louisville", "Birmingham", "Tampa Bay", "Charleston", "Memphis",
        "Orange County", "Pittsburgh", "Hartford", "Phoenix", "San Antonio",
        "Monterey Bay", "Sacramento", "Oakland", "Detroit", "Indy Eleven",
        "Las Vegas", "Miami", "Colorado Springs", "El Paso"
    ]
    return f"{cities[idx % len(cities)]} FC"


def generate_sample_goalkeeper_data(
    iteration_id: int, 
    year: int, 
    num_goalkeepers: int = 30
) -> pd.DataFrame:
    """
    Generate sample goalkeeper data for an iteration
    
    Args:
        iteration_id: Iteration ID
        year: Year
        num_goalkeepers: Number of goalkeepers to generate
        
    Returns:
        DataFrame with sample goalkeeper data
    """
    np.random.seed(iteration_id)  # For reproducibility
    
    data = []
    
    for i in range(num_goalkeepers):
        # Base player info
        player_id = 10000 + i + (year - 2022) * 100
        player_name = generate_goalkeeper_name(i)
        team_id = 5000 + (i % 19)
        team_name = generate_team_name(i)
        age = np.random.randint(20, 36)
        
        # Generate correlated performance metrics
        # Base quality level for this goalkeeper
        quality_level = np.random.beta(2, 5)  # Skewed toward lower values (realistic)
        
        # Minutes played (some play more than others)
        matches_played = np.random.randint(10, 35)
        minutes_played = matches_played * np.random.uniform(80, 90)
        
        # Shot stopping metrics (correlated with quality)
        shots_against = matches_played * np.random.uniform(3, 7)
        save_percentage = 0.55 + quality_level * 0.25 + np.random.normal(0, 0.05)
        save_percentage = max(0.5, min(0.85, save_percentage))
        saves = shots_against * save_percentage
        saves_per_90 = saves / (minutes_played / 90)
        goals_conceded = shots_against - saves
        goals_conceded_per_90 = goals_conceded / (minutes_played / 90)
        
        # Expected goals metrics
        expected_goals_against = goals_conceded * np.random.uniform(0.9, 1.1)
        goals_prevented = expected_goals_against - goals_conceded
        
        # Distribution metrics
        passes_attempted = minutes_played * np.random.uniform(15, 35)
        pass_completion_pct = 0.60 + quality_level * 0.25 + np.random.normal(0, 0.05)
        pass_completion_pct = max(0.55, min(0.90, pass_completion_pct))
        passes_completed = passes_attempted * pass_completion_pct
        
        long_passes_attempted = passes_attempted * np.random.uniform(0.3, 0.5)
        long_pass_completion_pct = pass_completion_pct * np.random.uniform(0.6, 0.85)
        long_passes_completed = long_passes_attempted * long_pass_completion_pct
        
        passes_into_final_third = passes_completed * np.random.uniform(0.15, 0.30)
        progressive_passes = passes_completed * np.random.uniform(0.10, 0.20)
        
        goal_kicks = matches_played * np.random.uniform(8, 15)
        goal_kick_completion_pct = pass_completion_pct * np.random.uniform(0.7, 0.9)
        
        # Sweeping actions
        defensive_actions_outside_penalty_area = matches_played * np.random.uniform(1, 5)
        successful_sweeper_actions = defensive_actions_outside_penalty_area * np.random.uniform(0.7, 0.95)
        
        # Aerial and crosses
        crosses_faced = matches_played * np.random.uniform(3, 8)
        cross_claim_pct = 0.20 + quality_level * 0.30 + np.random.normal(0, 0.05)
        cross_claim_pct = max(0.15, min(0.60, cross_claim_pct))
        crosses_claimed = crosses_faced * cross_claim_pct
        punches = crosses_faced * np.random.uniform(0.1, 0.3)
        high_ball_wins = (crosses_claimed + punches) * np.random.uniform(0.8, 1.0)
        
        # Reliability metrics
        clean_sheets = matches_played * quality_level * np.random.uniform(0.15, 0.40)
        clean_sheets = int(clean_sheets)
        clean_sheet_percentage = clean_sheets / matches_played if matches_played > 0 else 0
        
        # Match results (wins/draws/losses) - correlated with quality and clean sheets
        # Better goalkeepers should have more wins
        win_probability = 0.30 + quality_level * 0.30  # Range: 30-60% win rate
        draw_probability = 0.25 + np.random.uniform(-0.05, 0.05)  # ~25% draw rate
        
        wins = 0
        draws = 0
        losses = 0
        
        for _ in range(int(matches_played)):
            rand = np.random.random()
            if rand < win_probability:
                wins += 1
            elif rand < win_probability + draw_probability:
                draws += 1
            else:
                losses += 1
        
        points_gained = wins * 3 + draws * 1
        points_per_match = points_gained / matches_played if matches_played > 0 else 0
        
        errors_leading_to_shot = np.random.poisson((1 - quality_level) * 3)
        errors_leading_to_goal = np.random.poisson((1 - quality_level) * 1.5)
        penalties_conceded = np.random.poisson((1 - quality_level) * 2)
        
        # Other metrics
        touches = passes_completed + saves + crosses_claimed + punches
        passes_received = touches * np.random.uniform(0.3, 0.5)
        
        goalkeeper_record = {
            'playerId': player_id,
            'playerName': player_name,
            'teamId': team_id,
            'teamName': team_name,
            'age': age,
            'position': 'GK',
            'iteration_id': iteration_id,
            'year': year,
            
            # Shot stopping
            'saves': saves,
            'saves_per_90': saves_per_90,
            'save_percentage': save_percentage,
            'shots_on_target_against': shots_against,
            'goals_conceded': goals_conceded,
            'goals_conceded_per_90': goals_conceded_per_90,
            'expected_goals_against': expected_goals_against,
            'goals_prevented': goals_prevented,
            
            # Distribution
            'passes_completed': passes_completed,
            'pass_completion_percentage': pass_completion_pct,
            'long_passes_completed': long_passes_completed,
            'long_pass_completion_percentage': long_pass_completion_pct,
            'passes_into_final_third': passes_into_final_third,
            'goal_kicks': goal_kicks,
            'goal_kick_completion_percentage': goal_kick_completion_pct,
            'progressive_passes': progressive_passes,
            
            # Sweeping
            'defensive_actions_outside_penalty_area': defensive_actions_outside_penalty_area,
            'successful_sweeper_actions': successful_sweeper_actions,
            
            # Aerial
            'crosses_claimed': crosses_claimed,
            'cross_claim_percentage': cross_claim_pct,
            'punches': punches,
            'high_ball_wins': high_ball_wins,
            
            # Reliability
            'clean_sheets': clean_sheets,
            'clean_sheet_percentage': clean_sheet_percentage,
            'errors_leading_to_shot': errors_leading_to_shot,
            'errors_leading_to_goal': errors_leading_to_goal,
            'penalties_conceded': penalties_conceded,
            
            # Match results (north star metric)
            'matches_played_with_result': matches_played,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'points_gained': points_gained,
            'points_per_match': points_per_match,
            
            # General
            'minutes_played': minutes_played,
            'touches': touches,
            'passes_received': passes_received,
        }
        
        data.append(goalkeeper_record)
    
    return pd.DataFrame(data)


def generate_all_sample_data() -> pd.DataFrame:
    """
    Generate sample data for all training iterations
    
    Returns:
        Combined DataFrame with all sample data
    """
    all_data = []
    
    for year, iteration_id in TRAINING_ITERATIONS.items():
        print(f"Generating sample data for {year} (iteration {iteration_id})...")
        year_data = generate_sample_goalkeeper_data(iteration_id, year, num_goalkeepers=30)
        all_data.append(year_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Generated {len(combined_data)} total goalkeeper records")
    
    return combined_data


def save_sample_data():
    """Generate and save sample data to cache"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    # Generate and save combined data
    data = generate_all_sample_data()
    cache_path = os.path.join(DATA_CACHE_DIR, 'training_data_all_iterations.pkl')
    
    import pickle
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Sample data saved to {cache_path}")
    
    # Also save individual iterations
    for year, iteration_id in TRAINING_ITERATIONS.items():
        year_data = data[data['year'] == year]
        cache_path = os.path.join(DATA_CACHE_DIR, f'iteration_{iteration_id}_{year}.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(year_data, f)
        print(f"Saved {year} data to {cache_path}")
    
    return data


if __name__ == "__main__":
    # Generate sample data
    data = save_sample_data()
    print("\nSample data generation complete!")
    print(f"\nData shape: {data.shape}")
    print(f"\nColumns: {data.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(data.head())
