#!/usr/bin/env python3
"""
Comprehensive analysis script for causal scenarios evaluation data.
Processes JSON data, calculates statistics, runs statistical tests, and generates visualizations.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class CausalAnalysis:
    """Main analysis class for causal scenarios evaluation."""
    
    def __init__(self, json_path: str):
        """Initialize with path to the JSON data file."""
        self.json_path = json_path
        self.data = None
        self.processed_data = []
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load and parse the JSON data."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded data with {len(self.data['scenarios'])} scenarios")
        
    def process_ratings(self):
        """Process ratings from the data, handling negative_causal_assessment scale inversion."""
        for scenario in self.data['scenarios']:
            scenario_id = scenario['id']
            
            for result in scenario['results']:
                rating = result['rating']
                variation_type = result['variation_type']
                
                # Invert scale for negative_causal_assessment
                if variation_type == 'negative_causal_assessment':
                    rating = 100 - rating
                
                self.processed_data.append({
                    'scenario_id': scenario_id,
                    'scenario_paper': scenario.get('paper', ''),
                    'causal_structure': scenario.get('causal_structure', ''),
                    'variation_type': variation_type,
                    'rating': rating,
                    'original_rating': result['rating']  # Keep original for reference
                })
        
        print(f"Processed {len(self.processed_data)} ratings")
        
    def calculate_scenario_statistics(self) -> pd.DataFrame:
        """Calculate mean, median, and std for each scenario."""
        df = pd.DataFrame(self.processed_data)
        
        scenario_stats = df.groupby('scenario_id').agg({
            'rating': ['mean', 'median', 'std'],
            'scenario_paper': 'first',
            'causal_structure': 'first'
        }).round(2)
        
        # Flatten column names
        scenario_stats.columns = ['mean', 'median', 'std', 'paper', 'causal_structure']
        scenario_stats = scenario_stats.reset_index()
        
        # Sort scenarios numerically by ID
        scenario_stats['scenario_num'] = scenario_stats['scenario_id'].str.extract(r'(\d+)').astype(int)
        scenario_stats = scenario_stats.sort_values('scenario_num').drop('scenario_num', axis=1)
        
        return scenario_stats
    
    def calculate_variation_statistics(self) -> pd.DataFrame:
        """Calculate mean, median, and std for each variation type."""
        df = pd.DataFrame(self.processed_data)
        
        variation_stats = df.groupby('variation_type').agg({
            'rating': ['mean', 'median', 'std']
        }).round(2)
        
        # Flatten column names
        variation_stats.columns = ['mean', 'median', 'std']
        variation_stats = variation_stats.reset_index()
        
        return variation_stats
    
    def calculate_overall_statistics(self) -> Dict[str, float]:
        """Calculate overall mean, median, and std across all data."""
        df = pd.DataFrame(self.processed_data)
        
        overall_stats = {
            'mean': df['rating'].mean(),
            'median': df['rating'].median(),
            'std': df['rating'].std()
        }
        
        return {k: round(v, 2) for k, v in overall_stats.items()}
    
    def run_scenario_wilcoxon_tests(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run Wilcoxon signed-rank tests between all pairs of scenarios."""
        df = pd.DataFrame(self.processed_data)
        scenarios = df['scenario_id'].unique()
        n_scenarios = len(scenarios)
        
        # Create matrix to store p-values
        p_matrix = np.zeros((n_scenarios, n_scenarios))
        stat_matrix = np.zeros((n_scenarios, n_scenarios))
        
        # Store detailed results
        test_results = []
        
        for i, scenario1 in enumerate(scenarios):
            for j, scenario2 in enumerate(scenarios):
                if i == j:
                    p_matrix[i, j] = 1.0
                    stat_matrix[i, j] = 0.0
                    continue
                
                # Get ratings for both scenarios
                ratings1 = df[df['scenario_id'] == scenario1]['rating'].values
                ratings2 = df[df['scenario_id'] == scenario2]['rating'].values
                
                # Run Wilcoxon test
                try:
                    stat, p_value = stats.wilcoxon(ratings1, ratings2, alternative='two-sided')
                    p_matrix[i, j] = p_value
                    stat_matrix[i, j] = stat
                    
                    test_results.append({
                        'scenario1': scenario1,
                        'scenario2': scenario2,
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                except ValueError:
                    # Handle cases where test cannot be performed
                    p_matrix[i, j] = np.nan
                    stat_matrix[i, j] = np.nan
        
        return pd.DataFrame(test_results), p_matrix
    
    def run_variation_wilcoxon_tests(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run Wilcoxon signed-rank tests between all pairs of variation types."""
        df = pd.DataFrame(self.processed_data)
        variations = df['variation_type'].unique()
        n_variations = len(variations)
        
        # Create matrix to store p-values
        p_matrix = np.zeros((n_variations, n_variations))
        stat_matrix = np.zeros((n_variations, n_variations))
        
        # Store detailed results
        test_results = []
        
        for i, var1 in enumerate(variations):
            for j, var2 in enumerate(variations):
                if i == j:
                    p_matrix[i, j] = 1.0
                    stat_matrix[i, j] = 0.0
                    continue
                
                # Get ratings for both variation types
                ratings1 = df[df['variation_type'] == var1]['rating'].values
                ratings2 = df[df['variation_type'] == var2]['rating'].values
                
                # Run Wilcoxon test
                try:
                    stat, p_value = stats.wilcoxon(ratings1, ratings2, alternative='two-sided')
                    p_matrix[i, j] = p_value
                    stat_matrix[i, j] = stat
                    
                    test_results.append({
                        'variation1': var1,
                        'variation2': var2,
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                except ValueError:
                    # Handle cases where test cannot be performed
                    p_matrix[i, j] = np.nan
                    stat_matrix[i, j] = np.nan
        
        return pd.DataFrame(test_results), p_matrix
    
    def create_variation_plots(self, variation_stats: pd.DataFrame):
        """Create plots for variation type statistics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Mean plot
        axes[0].bar(variation_stats['variation_type'], variation_stats['mean'], 
                   color='skyblue', alpha=0.7)
        axes[0].set_title('Mean Ratings by Variation Type', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Variation Type')
        axes[0].set_ylabel('Mean Rating')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0, 100)
        axes[0].grid(True, alpha=0.3)
        
        # Median plot
        axes[1].bar(variation_stats['variation_type'], variation_stats['median'], 
                   color='lightgreen', alpha=0.7)
        axes[1].set_title('Median Ratings by Variation Type', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Variation Type')
        axes[1].set_ylabel('Median Rating')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, alpha=0.3)
        
        # Std plot
        axes[2].bar(variation_stats['variation_type'], variation_stats['std'], 
                   color='salmon', alpha=0.7)
        axes[2].set_title('Standard Deviation by Variation Type', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Variation Type')
        axes[2].set_ylabel('Standard Deviation')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'variation_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_scenario_plots(self, scenario_stats: pd.DataFrame):
        """Create plots for scenario statistics."""
        # Sort scenarios numerically by ID
        scenario_stats_sorted = scenario_stats.copy()
        scenario_stats_sorted['scenario_num'] = scenario_stats_sorted['scenario_id'].str.extract(r'(\d+)').astype(int)
        scenario_stats_sorted = scenario_stats_sorted.sort_values('scenario_num')
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Mean plot
        axes[0].bar(range(len(scenario_stats_sorted)), scenario_stats_sorted['mean'], 
                   color='skyblue', alpha=0.7)
        axes[0].set_title('Mean Ratings by Scenario', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Scenario ID')
        axes[0].set_ylabel('Mean Rating')
        axes[0].set_xticks(range(len(scenario_stats_sorted)))
        axes[0].set_xticklabels(scenario_stats_sorted['scenario_id'], rotation=45)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # Median plot
        axes[1].bar(range(len(scenario_stats_sorted)), scenario_stats_sorted['median'], 
                   color='lightgreen', alpha=0.7)
        axes[1].set_title('Median Ratings by Scenario', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Scenario ID')
        axes[1].set_ylabel('Median Rating')
        axes[1].set_xticks(range(len(scenario_stats_sorted)))
        axes[1].set_xticklabels(scenario_stats_sorted['scenario_id'], rotation=45)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
    
        # Std plot
        axes[2].bar(range(len(scenario_stats_sorted)), scenario_stats_sorted['std'], 
                   color='salmon', alpha=0.7)
        axes[2].set_title('Standard Deviation by Scenario', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Scenario ID')
        axes[2].set_ylabel('Standard Deviation')
        axes[2].set_xticks(range(len(scenario_stats_sorted)))
        axes[2].set_xticklabels(scenario_stats_sorted['scenario_id'], rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'scenario_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_scenario_heatmap(self, p_matrix: np.ndarray, scenario_stats: pd.DataFrame):
        """Create heatmap for scenario Wilcoxon test results."""
        fig, ax = plt.subplots(figsize=(20, 15))
        
        # Create mask for diagonal (self-comparisons)
        mask = np.eye(p_matrix.shape[0], dtype=bool)
        
        # Use hierarchical clustering to reorder the matrix
        # Replace NaN values with 1.0 for clustering (they represent self-comparisons)
        p_matrix_cluster = p_matrix.copy()
        p_matrix_cluster[np.isnan(p_matrix_cluster)] = 1.0
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(p_matrix_cluster, method='ward')
        
        # Get the optimal leaf ordering
        from scipy.cluster.hierarchy import optimal_leaf_ordering
        optimal_linkage = optimal_leaf_ordering(linkage_matrix, p_matrix_cluster)
        
        # Get the order of leaves
        from scipy.cluster.hierarchy import dendrogram
        dendro = dendrogram(optimal_linkage, no_plot=True)
        order = dendro['leaves']
        
        # Reorder the matrix and labels
        p_matrix_reordered = p_matrix[order][:, order]
        scenario_ids_reordered = scenario_stats['scenario_id'].values[order]
        
        # Create heatmap with reordered data
        sns.heatmap(p_matrix_reordered, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0.05,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'p-value'})
        
        ax.set_title('Wilcoxon Test p-values: Scenarios (Hierarchically Clustered)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Scenario ID')
        ax.set_ylabel('Scenario ID')
        
        # Set tick labels with reordered scenario IDs
        ax.set_xticks(np.arange(len(scenario_ids_reordered)) + 0.5)
        ax.set_yticks(np.arange(len(scenario_ids_reordered)) + 0.5)
        ax.set_xticklabels(scenario_ids_reordered, rotation=45, ha='right')
        ax.set_yticklabels(scenario_ids_reordered, rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'scenario_wilcoxon_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_variation_heatmap(self, p_matrix: np.ndarray, variation_stats: pd.DataFrame):
        """Create heatmap for variation type Wilcoxon test results."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for diagonal (self-comparisons)
        mask = np.eye(p_matrix.shape[0], dtype=bool)
        
        # Create heatmap
        sns.heatmap(p_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r',
                   center=0.05,
                   square=True,
                   ax=ax,
                   cbar_kws={'label': 'p-value'})
        
        ax.set_title('Wilcoxon Test p-values: Variation Types', fontsize=16, fontweight='bold')
        ax.set_xlabel('Variation Type')
        ax.set_ylabel('Variation Type')
        
        # Set tick labels
        variation_types = variation_stats['variation_type'].values
        ax.set_xticks(np.arange(len(variation_types)) + 0.5)
        ax.set_yticks(np.arange(len(variation_types)) + 0.5)
        ax.set_xticklabels(variation_types, rotation=45, ha='right')
        ax.set_yticklabels(variation_types, rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'variation_wilcoxon_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, scenario_stats: pd.DataFrame, variation_stats: pd.DataFrame, 
                    overall_stats: Dict[str, float], scenario_tests: pd.DataFrame, 
                    variation_tests: pd.DataFrame):
        """Save all results to files."""
        
        # Save statistics to CSV
        scenario_stats.to_csv(self.results_dir / 'scenario_statistics.csv', index=False)
        variation_stats.to_csv(self.results_dir / 'variation_statistics.csv', index=False)
        
        # Save overall statistics
        with open(self.results_dir / 'overall_statistics.json', 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        # Save Wilcoxon test results (CSV only, no redundant JSON files)
        scenario_tests.to_csv(self.results_dir / 'scenario_wilcoxon_tests.csv', index=False)
        variation_tests.to_csv(self.results_dir / 'variation_wilcoxon_tests.csv', index=False)
        
        print(f"Results saved to {self.results_dir}")
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting causal analysis...")
        
        # Load and process data
        self.load_data()
        self.process_ratings()
        
        # Calculate statistics
        print("Calculating statistics...")
        scenario_stats = self.calculate_scenario_statistics()
        variation_stats = self.calculate_variation_statistics()
        overall_stats = self.calculate_overall_statistics()
        
        # Run statistical tests
        print("Running Wilcoxon tests...")
        scenario_tests, scenario_p_matrix = self.run_scenario_wilcoxon_tests()
        variation_tests, variation_p_matrix = self.run_variation_wilcoxon_tests()
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_variation_plots(variation_stats)
        self.create_scenario_plots(scenario_stats)
        self.create_scenario_heatmap(scenario_p_matrix, scenario_stats)
        self.create_variation_heatmap(variation_p_matrix, variation_stats)
        
        # Save results
        print("Saving results...")
        self.save_results(scenario_stats, variation_stats, overall_stats, 
                         scenario_tests, variation_tests)
        
        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total scenarios analyzed: {len(scenario_stats)}")
        print(f"Total variation types: {len(variation_stats)}")
        print(f"Overall mean rating: {overall_stats['mean']}")
        print(f"Overall median rating: {overall_stats['median']}")
        print(f"Overall standard deviation: {overall_stats['std']}")
        print(f"\nResults saved to: {self.results_dir}")


def main():
    """Main function to run the analysis."""
    # Path to the JSON data file
    json_path = "../results/all_scenarios_evaluation.json"
    
    # Create and run analysis
    analyzer = CausalAnalysis(json_path)
    analyzer.run_analysis()


if __name__ == "__main__":
    main() 