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

**The system uses a data-driven approach:**
- Trains on ALL historical data (2022-2025 USLC seasons)
- Uses machine learning to determine metric weights automatically
- NO manual/subjective weight choices required

```bash
python moneyball.py
```

The system will:
- Authenticate with Impect API
- Fetch data from all training iterations (2022-2025 USLC)
- Cache the data locally for faster subsequent runs
- **Phase 1**: Score with equal weights and train ML model
- **Phase 2**: Extract ML-derived weights from model
- **Phase 3**: Re-score using data-driven weights
- Identify top targets
- Export results to `output/` directory, including:
  - `ml_derived_weights.csv` - Data-driven weights (not subjective)
  - `feature_importance.csv` - Which metrics matter most
  - `goalkeeper_scores_all_years.csv` - Final scores
  - `targets_2025.csv` - Recommended targets

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

### Data-Driven Approach (NEW)

The system now uses **machine learning to determine metric weights automatically**:

1. **Phase 1**: Score all goalkeepers with equal weights (no bias)
2. **Phase 2**: Train ML model to learn which metrics predict performance
3. **Phase 3**: Extract feature importance as weights (data-driven)
4. **Phase 4**: Re-score using ML-derived weights

**Example ML-Derived Weights** (from actual training):
- `crosses_claimed`: 18.11 (very high importance - discovered by ML)
- `touches`: 2.48
- `progressive_passes`: 2.26
- `saves`: 1.08
- `high_ball_wins`: 1.06

**Key Advantage**: Weights reflect what actually predicts performance, not human opinion.

### Metric Categories

The system evaluates goalkeepers across 30+ metrics in 5 key categories:

#### 1. Shot Stopping
- save_percentage, saves_per_90, goals_prevented, expected_goals_against

#### 2. Distribution  
- pass_completion_percentage, long_pass_completion_percentage
- progressive_passes, goal_kick_completion_percentage

#### 3. Sweeping
- defensive_actions_outside_penalty_area, successful_sweeper_actions

#### 4. Aerial Ability
- cross_claim_percentage, crosses_claimed, high_ball_wins

#### 5. Reliability
- clean_sheet_percentage, clean_sheets
- errors_leading_to_goal, errors_leading_to_shot

**Note**: Specific weights for each metric are determined by the ML model based on historical data, not manually assigned.

### Composite Score Calculation

1. **Normalization**: All metrics normalized to 0-1 scale
2. **ML Weighting**: Multiplied by ML-derived importance weights
3. **Aggregation**: Weighted average across all metrics  
4. **Scaling**: Converted to 0-100 score

Formula:
```
composite_score = (Σ(normalized_metric * ml_weight) / Σ(ml_weights)) * 100
```

where `ml_weight` comes from Random Forest feature importance, not manual assignment.

## Interpreting Results

### Output Files

After running the analysis, check the `output/` directory:

#### ml_derived_weights.csv (NEW)
**Data-driven weights learned from ML model** - shows which metrics matter most.

**Columns:**
- `metric`: Name of the metric
- `ml_weight`: Importance weight determined by Random Forest (not manually chosen)

This file proves the analysis is data-driven, not based on subjective opinions.

#### feature_importance.csv
Feature importance from the ML model, showing top predictive metrics.

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

### Using Data-Driven vs Manual Weights

**Recommended (Default)**: Use data-driven weights
```python
from moneyball import GoalkeeperMoneyball

# Use ML-derived weights (recommended - no subjective bias)
moneyball = GoalkeeperMoneyball(use_cache=True, use_data_driven_weights=True)
moneyball.run_full_analysis()
```

**Optional**: Use manual weights (not recommended)
```python
# Disable data-driven weights to use manual weights from config.py
moneyball = GoalkeeperMoneyball(use_cache=True, use_data_driven_weights=False)
moneyball.run_full_analysis()
```

### Adjusting Metric Weights (Optional - Not Recommended)

If you choose to use manual weights (`use_data_driven_weights=False`), you can edit `config.py`:

```python
GOALKEEPER_METRICS = {
    'save_percentage': 2.0,  # Increase weight (more important)
    'pass_completion_percentage': 0.3,  # Decrease weight (less important)
    # ... other metrics
}
```

**However**, this approach is **not recommended** as it introduces subjective bias. The data-driven approach is more objective.

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
