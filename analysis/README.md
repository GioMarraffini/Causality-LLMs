# Causal Analysis Script

This script performs comprehensive analysis of causal scenarios evaluation data from the `all_scenarios_evaluation.json` file.

## What the script does:

1. **Data Processing**: 
   - Loads the JSON data containing scenarios and their evaluation results
   - Processes ratings, automatically inverting the scale for `negative_causal_assessment` variation type (100 - rating)

2. **Statistical Analysis**:
   - Calculates mean, median, and standard deviation for each scenario
   - Calculates mean, median, and standard deviation for each variation type
   - Calculates overall statistics across all data

3. **Statistical Testing**:
   - Runs Wilcoxon signed-rank tests between all pairs of scenarios (NxN matrix)
   - Runs Wilcoxon signed-rank tests between all pairs of variation types (4x4 matrix)

4. **Visualizations**:
   - Creates bar plots showing mean, median, and std for variation types
   - Creates bar plots showing mean, median, and std for scenarios
   - Creates heatmaps for Wilcoxon test p-values (scenarios and variation types)

5. **Output Files**:
   All results are saved in the `results/` folder:
   
   **CSV Files:**
   - `scenario_statistics.csv` - Statistics per scenario
   - `variation_statistics.csv` - Statistics per variation type
   - `scenario_wilcoxon_tests.csv` - Wilcoxon test results for scenarios
   - `variation_wilcoxon_tests.csv` - Wilcoxon test results for variation types
   
   **JSON Files:**
   - `overall_statistics.json` - Overall statistics
   
   **PNG Files:**
   - `variation_statistics.png` - Bar plots for variation type statistics
   - `scenario_statistics.png` - Bar plots for scenario statistics (ordered numerically)
   - `scenario_wilcoxon_heatmap.png` - Heatmap for scenario Wilcoxon tests (hierarchically clustered)
   - `variation_wilcoxon_heatmap.png` - Heatmap for variation type Wilcoxon tests

## Usage:

```bash
python causal_analysis.py
```

## Requirements:

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- pathlib

## Data Structure Expected:

The script expects a JSON file with the following structure:
```json
{
  "scenarios": [
    {
      "id": "scenario_1",
      "results": [
        {
          "variation_type": "positive_causal_assessment",
          "rating": 95,
          ...
        },
        ...
      ]
    }
  ]
}
```

## Key Features:

- **Automatic Scale Inversion**: For `negative_causal_assessment` variation type, ratings are automatically converted using `100 - rating`
- **Comprehensive Statistics**: Calculates mean, median, and standard deviation at multiple levels
- **Statistical Testing**: Uses Wilcoxon signed-rank tests for non-parametric comparison
- **Professional Visualizations**: Creates publication-ready plots and heatmaps
- **Complete Documentation**: All results are saved in both CSV and JSON formats for further analysis 