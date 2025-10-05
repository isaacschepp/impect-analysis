# USLC Goalkeeper Moneyball System

A 100% mathematical, data-driven system for identifying high-potential goalkeepers in the United Soccer League Championship (USLC) using the Impect API.

## Overview

This system implements a "Moneyball" approach to goalkeeper analysis, using only objective, quantifiable metrics - no subjective evaluation. It uses a **data-driven, two-phase methodology** where machine learning determines metric weights automatically.

**Key Innovation**: Instead of manually choosing how much each metric matters (subjective), the system uses machine learning to learn which metrics predict performance from historical data (objective).

The system combines:

1. **Data Collection**: Fetches goalkeeper statistics from USLC iterations via the Impect API
2. **Initial Scoring**: Calculates scores with equal weights (no bias) to train ML model
3. **ML Weight Learning**: Trains model on ALL historical data (2022-2025) to learn feature importance
4. **Final Scoring**: Re-scores using ML-derived weights (data-driven, not subjective)
5. **Target Identification**: Ranks and recommends goalkeepers for recruitment

## Features

- **100% Data-Driven Analysis**: Metric weights determined by ML, not human opinion
- **No Subjective Weights**: The data tells us what matters, not manual choices
- **Trains on ALL Historical Data**: Uses complete data from 2022-2025 USLC seasons
- **Comprehensive Metrics**: Evaluates goalkeepers across multiple dimensions:
  - Shot stopping (saves, save percentage, goals prevented)
  - Distribution (passing accuracy, long balls, progressive passes)
  - Sweeping (defensive actions outside penalty area)
  - Aerial ability (crosses claimed, high ball wins)
  - Reliability (clean sheets, error prevention)
- **Machine Learning Models**: Uses Random Forest to learn feature importance
- **Two-Phase Approach**: Initial equal-weight scoring → ML training → Final ML-weighted scoring
- **Target Identification**: Generates ranked lists of recruitment targets

## Training Data

The system uses the following USLC iterations for training:
- 2025 USLC: Iteration 1236
- 2024 USLC: Iteration 893
- 2023 USLC: Iteration 642
- 2022 USLC: Iteration 510

## Installation

```bash
# Clone the repository
git clone https://github.com/isaacschepp/impect-analysis.git
cd impect-analysis

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to update:
- Impect API credentials
- Training iterations (currently 2022-2025)
- ML parameters

**Note**: Metric weights are now automatically determined by the ML model (data-driven approach). Manual weights in `config.py` are no longer used by default.

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
python moneyball.py
```

This will:
1. Fetch data from all training iterations
2. Calculate mathematical scores for all goalkeepers
3. Train the machine learning model
4. Identify top targets
5. Export results to the `output/` directory

### Using with Real Impect API Data

If you have access to the Impect API:

```python
from moneyball import GoalkeeperMoneyball

# Initialize the system
moneyball = GoalkeeperMoneyball(use_cache=False)

# Run full analysis
report = moneyball.run_full_analysis()
```

### Using with Sample Data (for testing)

Generate sample data first:

```bash
python generate_sample_data.py
```

Then run the analysis:

```bash
python moneyball.py
```

### Custom Analysis

You can also use individual components:

```python
from data_collector import GoalkeeperDataCollector
from scorer import GoalkeeperScorer
from predictor import GoalkeeperPredictor

# Collect data
collector = GoalkeeperDataCollector()
data = collector.collect_training_data()

# Score goalkeepers
scorer = GoalkeeperScorer()
scored_data = scorer.score_goalkeepers(data)

# Get top performers
top_10 = scorer.get_top_performers(scored_data, n=10)

# Train prediction model
predictor = GoalkeeperPredictor()
metrics = predictor.train(scored_data)

# Identify targets
targets = predictor.identify_targets(scored_data, n_targets=20)
```

## Output

The system generates several output files in the `output/` directory:

- `goalkeeper_scores_all_years.csv`: Scores for all goalkeepers across all years
- `feature_importance.csv`: Most important metrics for performance prediction
- `top_performers_YYYY.csv`: Top 10 performers for each year
- `targets_2025.csv`: Top 20 goalkeeper targets for recruitment
- `goalkeeper_model.pkl`: Trained machine learning model

## Metric Categories

### Shot Stopping (Weight: High)
- Saves per 90 minutes
- Save percentage
- Goals prevented (xG - actual goals)
- Expected goals against

### Distribution (Weight: Medium-High)
- Pass completion percentage
- Long pass accuracy
- Passes into final third
- Progressive passes
- Goal kick completion

### Sweeping (Weight: Medium)
- Defensive actions outside penalty area
- Successful sweeper actions
- Average distance from goal

### Aerial Ability (Weight: Medium)
- Crosses claimed
- Cross claim percentage
- High ball wins
- Punches

### Reliability (Weight: High)
- Clean sheets
- Clean sheet percentage
- Errors leading to shots/goals (negative)
- Penalty prevention

## Machine Learning Models

The system supports two model types:

1. **Random Forest** (default): Robust ensemble method, handles non-linear relationships
2. **Gradient Boosting**: Higher accuracy, better for complex patterns

Models are evaluated using:
- R² score (explained variance)
- RMSE (prediction error)
- Cross-validation (5-fold)
- Feature importance analysis

## API Reference

### GoalkeeperMoneyball

Main orchestration class.

```python
moneyball = GoalkeeperMoneyball(use_cache=True)
moneyball.load_and_prepare_data()
moneyball.score_goalkeepers()
moneyball.train_model(model_type='random_forest')
moneyball.identify_top_targets(target_year=2025, n_targets=20)
report = moneyball.generate_report()
```

### GoalkeeperDataCollector

Handles data fetching and caching.

```python
collector = GoalkeeperDataCollector(use_cache=True)
data = collector.collect_training_data()
year_data = collector.collect_iteration_data(year=2025)
```

### GoalkeeperScorer

Calculates mathematical scores.

```python
scorer = GoalkeeperScorer()
scored_data = scorer.score_goalkeepers(data)
top_10 = scorer.get_top_performers(scored_data, n=10)
```

### GoalkeeperPredictor

Machine learning predictions.

```python
predictor = GoalkeeperPredictor()
metrics = predictor.train(data, model_type='random_forest')
targets = predictor.identify_targets(data, n_targets=20)
importance = predictor.get_feature_importance(top_n=20)
```

## Requirements

- Python 3.8+
- impectPy 2.4.4+
- pandas 2.0.0+
- numpy 1.24.2+
- scikit-learn 1.3.0+
- matplotlib 3.7.0+ (for visualization)
- seaborn 0.12.0+ (for visualization)

## Credentials

The system requires Impect API credentials:
- Email: isaac.schepp@gmail.com
- Password: ZJnpgKNSQkm9A_G

These are configured in `config.py`.

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.