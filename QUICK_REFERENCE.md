# USLC Goalkeeper Moneyball System - Quick Reference

## Installation
```bash
pip install -r requirements.txt
```

## Quick Commands

### Generate Sample Data (for testing)
```bash
python generate_sample_data.py
```

### Run Complete Analysis
```bash
python moneyball.py
```

### Run Specific Examples
```bash
python example.py 1  # Generate sample data
python example.py 2  # Score goalkeepers
python example.py 3  # Train ML model
python example.py 4  # Identify targets
python example.py 5  # Full analysis
```

## Python Quick Start

### Basic Usage
```python
from moneyball import GoalkeeperMoneyball

# Run everything
moneyball = GoalkeeperMoneyball(use_cache=True)
report = moneyball.run_full_analysis()
```

### Step-by-Step
```python
from data_collector import GoalkeeperDataCollector
from scorer import GoalkeeperScorer
from predictor import GoalkeeperPredictor

# 1. Collect data
collector = GoalkeeperDataCollector()
data = collector.collect_training_data()

# 2. Score goalkeepers
scorer = GoalkeeperScorer()
scored_data = scorer.score_goalkeepers(data)

# 3. Train model
predictor = GoalkeeperPredictor()
metrics = predictor.train(scored_data)

# 4. Identify targets
targets = predictor.identify_targets(scored_data, n_targets=20)
```

## Key Metrics Explained

### Most Important (based on feature importance)
1. **crosses_claimed** - Number of crosses successfully claimed
2. **clean_sheets** - Number of games without conceding
3. **clean_sheet_percentage** - Percentage of clean sheets
4. **save_percentage** - Percentage of shots saved
5. **cross_claim_percentage** - Success rate claiming crosses

### Composite Score Categories
- **80-100**: Elite goalkeeper
- **70-79**: Excellent goalkeeper
- **60-69**: Good goalkeeper
- **50-59**: Average goalkeeper
- **Below 50**: Below average

## Output Files

| File | Description |
|------|-------------|
| `goalkeeper_scores_all_years.csv` | All goalkeeper scores |
| `targets_2025.csv` | Top 20 targets for 2025 |
| `feature_importance.csv` | Most important metrics |
| `top_performers_YYYY.csv` | Top 10 per year |
| `goalkeeper_model.pkl` | Trained ML model |

## Configuration

Edit `config.py` to customize:

```python
# Change metric weights
GOALKEEPER_METRICS = {
    'save_percentage': 1.5,  # Your custom weight
    # ... other metrics
}

# Add/modify training iterations
TRAINING_ITERATIONS = {
    2025: 1236,
    2024: 893,
    # ... add more
}
```

## Common Tasks

### Get Top 10 Goalkeepers
```python
scorer = GoalkeeperScorer()
scored_data = scorer.score_goalkeepers(data)
top_10 = scorer.get_top_performers(scored_data, n=10, min_minutes=900)
```

### Compare Specific Players
```python
player_ids = [10300, 10328, 10309]
comparison = scorer.compare_goalkeepers(scored_data, player_ids)
```

### Train Different Model
```python
# Random Forest (default)
predictor.train(data, model_type='random_forest')

# Gradient Boosting
predictor.train(data, model_type='gradient_boosting')
```

### Save and Load Model
```python
# Save
predictor.save_model('output/my_model.pkl')

# Load
predictor.load_model('output/my_model.pkl')
predictions = predictor.predict(new_data)
```

### Use Fresh API Data
```python
# Ignore cache, fetch new data
moneyball = GoalkeeperMoneyball(use_cache=False)
moneyball.run_full_analysis()
```

### Export Results
```python
# Export to CSV
collector.export_to_csv(scored_data, 'my_results.csv')
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Authentication failed | Check credentials in `config.py` |
| Module not found | Run `pip install -r requirements.txt` |
| No data retrieved | Verify iteration IDs and API access |
| Model not trained | Call `predictor.train()` first |
| Permission denied | Check write permissions for output/ |

## System Requirements

- Python 3.8+
- 2GB RAM minimum
- Internet access for API (or use sample data)
- ~100MB disk space

## Training Data

| Year | Iteration ID |
|------|--------------|
| 2025 | 1236 |
| 2024 | 893 |
| 2023 | 642 |
| 2022 | 510 |

## API Credentials

Configured in `config.py`:
- Email: isaac.schepp@gmail.com
- Password: ZJnpgKNSQkm9A_G

## Performance Tips

1. Use cached data: `use_cache=True`
2. Filter by minimum minutes to reduce dataset
3. Use Random Forest for faster training
4. Run analysis during off-peak hours

## Support

- Documentation: See `README.md` and `USER_GUIDE.md`
- Examples: Run `python example.py`
- Issues: https://github.com/isaacschepp/impect-analysis/issues
