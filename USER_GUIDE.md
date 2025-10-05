# USLC Goalkeeper Moneyball System - User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Using with Real Impect API](#using-with-real-impect-api)
4. [Understanding the Scoring System](#understanding-the-scoring-system)
5. [Interpreting Results](#interpreting-results)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/isaacschepp/impect-analysis.git
cd impect-analysis

# Install dependencies
pip install -r requirements.txt
```

### Run with Sample Data (for testing)

```bash
# Generate sample data
python generate_sample_data.py

# Run analysis
python moneyball.py
```

### Run with Real API Data

```bash
# Simply run the main script (it will fetch data from Impect API)
python moneyball.py
```

## System Architecture

### Components

1. **impect_client.py** - Handles authentication and data fetching from Impect API
2. **data_collector.py** - Manages data collection and caching
3. **scorer.py** - Calculates mathematical scores for goalkeepers
4. **predictor.py** - Machine learning model for predictions
5. **moneyball.py** - Main orchestration pipeline
6. **config.py** - Configuration and metric weights

### Data Flow

```
Impect API → Data Collection → Scoring → ML Training → Target Identification → Reports
     ↓              ↓              ↓           ↓                ↓                ↓
  Raw Data    Cached Data   Normalized   Predictions      Ranked List      CSV Files
```

## Using with Real Impect API

### Prerequisites

- Impect API credentials (configured in `config.py`)
- Network access to api.impect.com and login.impect.com
- Python 3.8 or higher

### Step-by-Step Guide

1. **Verify Credentials**

Edit `config.py` to ensure your credentials are correct:

```python
IMPECT_EMAIL = "isaac.schepp@gmail.com"
IMPECT_PASSWORD = "ZJnpgKNSQkm9A_G"
```

2. **Run Analysis**

```bash
python moneyball.py
```

The system will:
- Authenticate with Impect API
- Fetch data from all training iterations (2022-2025 USLC)
- Cache the data locally for faster subsequent runs
- Calculate scores for all goalkeepers
- Train machine learning model
- Identify top targets
- Export results to `output/` directory

3. **Use Cached Data**

After the first run, data is cached in `data/cache/`. To use cached data:

```python
from moneyball import GoalkeeperMoneyball

# Use cached data (fast)
moneyball = GoalkeeperMoneyball(use_cache=True)
moneyball.run_full_analysis()

# Fetch fresh data from API (slow)
moneyball = GoalkeeperMoneyball(use_cache=False)
moneyball.run_full_analysis()
```

### Working with Individual Components

```python
from impect_client import ImpectClient
from data_collector import GoalkeeperDataCollector
from scorer import GoalkeeperScorer
from predictor import GoalkeeperPredictor

# 1. Fetch data for a specific iteration
client = ImpectClient()
data_2025 = client.get_goalkeeper_data(iteration_id=1236)

# 2. Collect and cache data
collector = GoalkeeperDataCollector(use_cache=True)
all_data = collector.collect_training_data()

# 3. Score goalkeepers
scorer = GoalkeeperScorer()
scored_data = scorer.score_goalkeepers(all_data)
top_10 = scorer.get_top_performers(scored_data, n=10)

# 4. Train model and predict
predictor = GoalkeeperPredictor()
metrics = predictor.train(scored_data)
targets = predictor.identify_targets(scored_data, n_targets=20)

# 5. Save model for reuse
predictor.save_model('output/my_model.pkl')

# Later, load the model
predictor.load_model('output/my_model.pkl')
predictions = predictor.predict(new_data)
```

## Understanding the Scoring System

### Metric Categories

The system evaluates goalkeepers across 5 key categories:

#### 1. Shot Stopping (Weight: Very High)
- **save_percentage**: % of shots saved (1.5x weight)
- **saves_per_90**: Saves per 90 minutes (1.0x weight)
- **goals_prevented**: xG prevented (1.5x weight)
- **expected_goals_against**: Quality of shots faced (0.8x weight)

#### 2. Distribution (Weight: Medium-High)
- **pass_completion_percentage**: Overall passing accuracy (0.8x weight)
- **long_pass_completion_percentage**: Long ball accuracy (0.6x weight)
- **progressive_passes**: Forward-thinking passes (0.9x weight)
- **goal_kick_completion_percentage**: GK accuracy (0.5x weight)

#### 3. Sweeping (Weight: Medium)
- **defensive_actions_outside_penalty_area**: Sweeper keeper actions (1.0x weight)
- **successful_sweeper_actions**: Successful interventions (1.2x weight)

#### 4. Aerial Ability (Weight: Medium)
- **cross_claim_percentage**: % of crosses claimed (1.1x weight)
- **crosses_claimed**: Total crosses claimed (0.9x weight)
- **high_ball_wins**: Aerial duels won (0.8x weight)

#### 5. Reliability (Weight: Very High)
- **clean_sheet_percentage**: % of clean sheets (1.8x weight)
- **clean_sheets**: Total clean sheets (2.0x weight)
- **errors_leading_to_goal**: Mistakes (negative -3.0x weight)
- **errors_leading_to_shot**: Mistakes (negative -2.0x weight)

### Composite Score Calculation

1. **Normalization**: All metrics normalized to 0-1 scale
2. **Weighting**: Multiplied by importance weights
3. **Aggregation**: Weighted average across all metrics
4. **Scaling**: Converted to 0-100 score

Formula:
```
composite_score = (Σ(normalized_metric * weight) / Σ(weights)) * 100
```

## Interpreting Results

### Output Files

After running the analysis, check the `output/` directory:

#### goalkeeper_scores_all_years.csv
Complete dataset with calculated scores for all goalkeepers across all years.

**Key columns:**
- `composite_score`: Overall performance (0-100)
- `shot_stopping_score`: Shot stopping ability (0-100)
- `distribution_score`: Passing ability (0-100)
- `aerial_score`: Aerial ability (0-100)
- `reliability_score`: Consistency (0-100)

#### targets_2025.csv
Top 20 recruitment targets for 2025 USLC season.

**Key columns:**
- `predicted_score`: ML model prediction
- `composite_score`: Current performance score
- `age`: Player age
- `minutes_played`: Playing time

#### feature_importance.csv
Most important metrics for goalkeeper performance.

**Interpretation:**
- Higher importance = stronger predictor of success
- Use to focus scouting efforts on key metrics

#### goalkeeper_model.pkl
Trained machine learning model (can be loaded for predictions on new data)

### Score Interpretation

- **80-100**: Elite goalkeeper, top tier
- **70-79**: Excellent goalkeeper, high quality
- **60-69**: Good goalkeeper, solid performer
- **50-59**: Average goalkeeper, adequate
- **40-49**: Below average goalkeeper
- **Below 40**: Poor performer

### Model Performance Metrics

When training completes, you'll see:

- **Test R²**: Proportion of variance explained (higher is better, 0-1)
  - > 0.7 = Good model
  - > 0.8 = Excellent model
  
- **Test RMSE**: Root mean squared error (lower is better)
  - < 5 = Good predictions
  - < 3 = Excellent predictions

- **CV R² (mean ± std)**: Cross-validation score (consistency check)
  - Similar to Test R² = consistent model

## Customization

### Adjusting Metric Weights

Edit `config.py` to change metric importance:

```python
GOALKEEPER_METRICS = {
    'save_percentage': 2.0,  # Increase weight (more important)
    'pass_completion_percentage': 0.3,  # Decrease weight (less important)
    # ... other metrics
}
```

### Adding New Metrics

If the Impect API provides additional metrics:

1. Add to `GOALKEEPER_METRICS` in `config.py`
2. Assign appropriate weight (positive for "higher is better", negative for "lower is better")
3. Re-run the analysis

### Changing Training Data

Edit `config.py` to use different iterations:

```python
TRAINING_ITERATIONS = {
    2026: 1500,  # Add new season
    2025: 1236,
    2024: 893,
    # ... etc
}
```

### Model Selection

Choose between Random Forest and Gradient Boosting:

```python
from moneyball import GoalkeeperMoneyball

moneyball = GoalkeeperMoneyball()
moneyball.load_and_prepare_data()
moneyball.score_goalkeepers()

# Option 1: Random Forest (default, more robust)
moneyball.train_model(model_type='random_forest')

# Option 2: Gradient Boosting (potentially higher accuracy)
moneyball.train_model(model_type='gradient_boosting')
```

### Filtering Criteria

Adjust minimum playing time requirements:

```python
# Require more minutes for qualification
top_performers = scorer.get_top_performers(
    scored_data, 
    n=10, 
    min_minutes=1350  # ~15 full matches
)

# Identify targets with different criteria
targets = predictor.identify_targets(
    data, 
    n_targets=30, 
    min_minutes=900  # ~10 full matches
)
```

## Troubleshooting

### Common Issues

#### "Failed to authenticate with Impect API"
- Check credentials in `config.py`
- Verify network access to login.impect.com
- Check if password has expired

#### "No data retrieved from any iteration"
- Verify iteration IDs are correct
- Check API access permissions
- Ensure goalkeeper position data is available

#### "Model must be trained before making predictions"
- Call `predictor.train()` before `predictor.predict()`
- Or load a previously saved model with `predictor.load_model()`

#### "Module not found" errors
- Install all requirements: `pip install -r requirements.txt`
- Verify Python version is 3.8 or higher

#### "Permission denied" when writing files
- Check write permissions for `output/` and `data/cache/` directories
- Run with appropriate user permissions

### Performance Optimization

#### Slow Data Fetching
- Use cached data: `GoalkeeperMoneyball(use_cache=True)`
- Reduce number of iterations in `config.py`

#### Slow Model Training
- Reduce dataset size (filter by min_minutes)
- Use Random Forest instead of Gradient Boosting
- Reduce cross-validation folds in `config.py`

### Getting Help

If you encounter issues:

1. Check the log output for detailed error messages
2. Verify all dependencies are installed
3. Try with sample data first: `python generate_sample_data.py`
4. Review the example scripts: `python example.py`

## Advanced Usage

### Custom Analysis Pipeline

```python
from moneyball import GoalkeeperMoneyball
import pandas as pd

# Initialize
moneyball = GoalkeeperMoneyball()

# Custom workflow
data = moneyball.load_and_prepare_data()

# Filter to specific years
recent_data = data[data['year'].isin([2024, 2025])]

# Score only recent data
scored_recent = moneyball.scorer.score_goalkeepers(recent_data)

# Train on recent data
metrics = moneyball.predictor.train(scored_recent)

# Export custom results
scored_recent.to_csv('output/custom_analysis.csv', index=False)
```

### Comparing Specific Goalkeepers

```python
from scorer import GoalkeeperScorer

scorer = GoalkeeperScorer()
scored_data = scorer.score_goalkeepers(data)

# Compare specific players by ID
player_ids = [10300, 10328, 10309]  # Replace with actual IDs
comparison = scorer.compare_goalkeepers(scored_data, player_ids)

print(comparison[['playerName', 'composite_score', 
                  'shot_stopping_score', 'distribution_score']])
```

### Historical Player Analysis

```python
from data_collector import GoalkeeperDataCollector

collector = GoalkeeperDataCollector()

# Get player's history across all iterations
player_history = collector.get_player_history(player_id=10300)

print(f"Player appeared in {len(player_history)} iterations")
print(player_history[['year', 'teamName', 'composite_score']])
```

## Best Practices

1. **Start with cached data** for iterative analysis
2. **Refresh data monthly** during the season
3. **Re-train model** when new data becomes available
4. **Cross-validate** results with traditional scouting
5. **Track predictions** over time to validate model accuracy
6. **Adjust weights** based on team playing style
7. **Consider age** and contract status in final decisions
8. **Use multiple seasons** of data for more reliable predictions

## Contact & Support

For questions or issues, please open an issue on GitHub:
https://github.com/isaacschepp/impect-analysis/issues
