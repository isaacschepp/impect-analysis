# Quick Start: Using the Comprehensive Metrics System

## Running the System

### With Real Impect API Data

```python
from moneyball import GoalkeeperMoneyball

# Initialize (will fetch fresh data from Impect API)
moneyball = GoalkeeperMoneyball(
    use_cache=False,  # Force fresh API fetch
    use_data_driven_weights=True  # Use ML-derived weights
)

# Run with points_per_match (north star metric)
report = moneyball.run_full_analysis(success_metric='points_per_match')
```

### With Cached/Sample Data

```python
from moneyball import GoalkeeperMoneyball

# Initialize (will use cached data)
moneyball = GoalkeeperMoneyball(
    use_cache=True,  # Use cache if available
    use_data_driven_weights=True
)

# Run analysis
report = moneyball.run_full_analysis(success_metric='points_per_match')
```

## Understanding the Output

### Key Output Files

```
output/
├── success_metric_explanation.csv    # What we're optimizing
├── goalkeeper_scores_all_years.csv   # All goalkeepers with 44 metrics
├── ml_derived_weights.csv            # What behaviors matter most
├── feature_importance.csv            # Top predictors of winning
├── targets_2025.csv                  # Top 20 recruitment targets
└── goalkeeper_model.pkl              # Trained model
```

### Interpreting Feature Importance

Open `output/feature_importance.csv` to see what predicts points per match:

```csv
feature,importance
losses,0.372        # 37.2% - Avoiding losses is critical
wins,0.356          # 35.6% - Winning games drives points
points_gained,0.087 # 8.7% - Direct correlation
clean_sheet_percentage,0.027  # 2.7% - Defense matters
long_pass_completion_percentage,0.010  # 1.0% - Build-up helps
```

**What this means:**
- The model validates that wins/losses predict points (as expected)
- But also discovers which **goalkeeper behaviors** correlate with winning
- Use this to identify goalkeepers with winning behaviors even if on poor teams

### Top Recruitment Targets

Open `output/targets_2025.csv` to see top prospects:

```csv
playerId,playerName,teamName,predicted_score,age,minutes_played,composite_score
10315,Paul Anderson,Las Vegas FC,1.85,35,2215.78,54.75
10307,Henry Anderson,Hartford FC,1.78,33,1790.54,41.63
...
```

- `predicted_score`: Model's prediction of their points_per_match
- `composite_score`: Overall quality score (ML-weighted)
- Sort by `predicted_score` for goalkeepers most likely to help team win

## Alternative Success Metrics

### Clean Sheet Percentage

```python
report = moneyball.run_full_analysis(success_metric='clean_sheet_percentage')
```

Focuses on defensive performance - how often the goalkeeper keeps a clean sheet.

### Goals Prevented

```python
report = moneyball.run_full_analysis(success_metric='goals_prevented')
```

Measures shot-stopping value - expected goals minus actual goals conceded.

### Goals Conceded Per 90

```python
report = moneyball.run_full_analysis(success_metric='goals_conceded_per_90')
```

Direct measure of defensive performance (lower is better).

## Comprehensive Metrics Breakdown

### The 44 Metrics Tracked

1. **Identity** (8): playerId, playerName, teamId, teamName, age, position, iteration_id, year

2. **Shot Stopping** (8): saves, saves_per_90, save_percentage, shots_on_target_against, goals_conceded, goals_conceded_per_90, expected_goals_against, goals_prevented

3. **Distribution** (8): passes_completed, pass_completion_percentage, long_passes_completed, long_pass_completion_percentage, passes_into_final_third, progressive_passes, goal_kicks, goal_kick_completion_percentage

4. **Sweeping** (2): defensive_actions_outside_penalty_area, successful_sweeper_actions

5. **Aerial** (4): crosses_claimed, cross_claim_percentage, punches, high_ball_wins

6. **Reliability** (5): clean_sheets, clean_sheet_percentage, errors_leading_to_shot, errors_leading_to_goal, penalties_conceded

7. **Match Results** (6) 🆕: matches_played_with_result, wins, draws, losses, points_gained, **points_per_match**

8. **General** (3): minutes_played, touches, passes_received

## Common Questions

### Q: How do I know we're testing all available metrics?

**A:** The system:
- Uses `getPlayerIterationScores()` which returns ALL metrics from Impect
- Evaluates ALL 44 metrics simultaneously with ML
- No cherry-picking - takes everything from the API
- Feature importance shows which metrics matter

### Q: What if Impect adds new metrics?

**A:** The system automatically:
- Fetches any new metrics from the API
- Includes them in ML training
- Evaluates their importance
- No code changes needed

### Q: Why is points_per_match the "north star"?

**A:** Because it:
- Directly measures winning (3 for win, 1 for draw, 0 for loss)
- Determines league standings
- Is what teams actually care about
- Can't be "gamed" - it's a pure game outcome

### Q: Can I use multiple success metrics?

**A:** Yes! Run the analysis multiple times:

```python
# Winning focus
report_winning = moneyball.run_full_analysis(success_metric='points_per_match')

# Defense focus
report_defense = moneyball.run_full_analysis(success_metric='clean_sheet_percentage')

# Shot-stopping focus
report_saves = moneyball.run_full_analysis(success_metric='goals_prevented')
```

Compare results to identify specialists vs all-around performers.

## Advanced Usage

### Custom Model Type

```python
# Use Gradient Boosting instead of Random Forest
from predictor import GoalkeeperPredictor

predictor = GoalkeeperPredictor()
metrics = predictor.train(data, model_type='gradient_boosting')
```

### Analyzing Specific Years

```python
from data_collector import GoalkeeperDataCollector

collector = GoalkeeperDataCollector(use_cache=True)
data_2024 = collector.collect_iteration_data(2024, comprehensive=True)

# Analyze just 2024
scorer = GoalkeeperScorer()
scored_2024 = scorer.score_goalkeepers(data_2024)
```

### Exporting for External Analysis

```python
# All data with comprehensive metrics
data = moneyball.training_data
data.to_csv('my_analysis/goalkeeper_data.csv', index=False)

# Just top performers
report = moneyball.generate_report()
targets = report['targets_2025']
targets.to_excel('my_analysis/recruitment_targets.xlsx', index=False)
```

## Next Steps

1. **Review the comprehensive guide**: See [COMPREHENSIVE_METRICS_GUIDE.md](COMPREHENSIVE_METRICS_GUIDE.md)
2. **Check implementation details**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
3. **Run the analysis**: `python moneyball.py`
4. **Review outputs**: Check `output/` directory
5. **Use for recruitment**: Analyze `targets_2025.csv`

## Support

For detailed information:
- **Comprehensive Metrics**: `COMPREHENSIVE_METRICS_GUIDE.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **System Overview**: `README.md`
- **Methodology**: `METHODOLOGY.md`
- **Issues & Solutions**: `ISSUES_ADDRESSED.md`

## Key Takeaway

The system answers the question: **"What goalkeeper behaviors actually help teams WIN games?"**

By using `points_per_match` as the north star metric and evaluating ALL 44 available metrics, the system discovers which behaviors correlate with winning - allowing you to identify high-potential goalkeepers based on data, not opinions.
