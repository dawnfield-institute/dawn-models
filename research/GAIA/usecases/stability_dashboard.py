#!/usr/bin/env python3
"""
GAIA Stability Analysis Dashboard

Visualizes test stability across trials to identify where recognition/optimization wobble most.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from datetime import datetime

class StabilityDashboard:
    """Visualize GAIA test stability patterns."""
    
    def __init__(self, results_path=None):
        """Initialize dashboard with results data."""
        if results_path is None:
            # Find most recent results
            results_dir = Path(__file__).parent / "results"
            if results_dir.exists():
                latest_dir = max(results_dir.glob("*"), key=lambda x: x.name)
                results_path = latest_dir / "test_results.json"
        
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.timestamp = self.results.get("timestamp", "unknown")
    
    def plot_stability_analysis(self):
        """Create comprehensive stability analysis plots."""
        tests = self.results.get("tests", {})
        
        # Filter tests with trial data
        stable_tests = []
        unstable_tests = []
        
        for name, data in tests.items():
            if "trials" in data and "cv" in data:
                if data.get("stable", False):
                    stable_tests.append((name, data))
                else:
                    unstable_tests.append((name, data))
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'GAIA Stability Analysis Dashboard - {self.timestamp}', fontsize=16, fontweight='bold')
        
        # Plot 1: Score variance across trials
        ax1 = plt.subplot(2, 3, 1)
        self._plot_score_variance(ax1, stable_tests, unstable_tests)
        
        # Plot 2: CV distribution
        ax2 = plt.subplot(2, 3, 2)
        self._plot_cv_distribution(ax2, stable_tests, unstable_tests)
        
        # Plot 3: Trial-by-trial trends
        ax3 = plt.subplot(2, 3, 3)
        self._plot_trial_trends(ax3, stable_tests, unstable_tests)
        
        # Plot 4: Score stability heatmap
        ax4 = plt.subplot(2, 3, 4)
        self._plot_stability_heatmap(ax4, tests)
        
        # Plot 5: Performance vs stability scatter
        ax5 = plt.subplot(2, 3, 5)
        self._plot_performance_stability(ax5, tests)
        
        # Plot 6: Wobble frequency analysis
        ax6 = plt.subplot(2, 3, 6)
        self._plot_wobble_analysis(ax6, unstable_tests)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = Path(__file__).parent / "results" / self.timestamp / "stability_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"Stability dashboard saved to: {dashboard_path}")
        
        plt.show()
    
    def _plot_score_variance(self, ax, stable_tests, unstable_tests):
        """Plot score variance across trials with error bars."""
        all_tests = stable_tests + unstable_tests
        names = [name.replace('_', ' ').title() for name, _ in all_tests]
        means = [data["mean_score"] for _, data in all_tests]
        stds = [data["std_score"] for _, data in all_tests]
        
        colors = ['green' if (name, data) in stable_tests else 'red' 
                 for name, data in all_tests]
        
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=5, 
                     color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Tests')
        ax.set_ylabel('Composite Score')
        ax.set_title('Score Variance Across Trials')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # Add CV labels on bars
        for i, (_, data) in enumerate(all_tests):
            cv = data["cv"] * 100
            ax.text(i, means[i] + stds[i] + 0.01, f'CV: {cv:.1f}%', 
                   ha='center', va='bottom', fontsize=8)
        
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Pass Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cv_distribution(self, ax, stable_tests, unstable_tests):
        """Plot coefficient of variation distribution."""
        stable_cvs = [data["cv"] * 100 for _, data in stable_tests]
        unstable_cvs = [data["cv"] * 100 for _, data in unstable_tests]
        
        ax.hist([stable_cvs, unstable_cvs], bins=10, alpha=0.7, 
               label=['Stable Tests', 'Unstable Tests'], 
               color=['green', 'red'])
        
        ax.axvline(x=20, color='orange', linestyle='--', label='Stability Threshold (20%)')
        ax.set_xlabel('Coefficient of Variation (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('CV Distribution: Stable vs Unstable')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trial_trends(self, ax, stable_tests, unstable_tests):
        """Plot trial-by-trial score trends."""
        all_tests = stable_tests + unstable_tests
        
        for name, data in all_tests:
            if "trials" in data:
                trials = data["trials"]
                scores = [t.get("composite_score", 0) for t in trials]
                
                color = 'green' if (name, data) in stable_tests else 'red'
                alpha = 0.8 if (name, data) in stable_tests else 0.6
                
                ax.plot(range(1, len(scores) + 1), scores, 
                       marker='o', label=name.replace('_', ' ').title(),
                       color=color, alpha=alpha)
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Composite Score')
        ax.set_title('Trial-by-Trial Score Trends')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_stability_heatmap(self, ax, tests):
        """Create stability heatmap showing CV vs mean score."""
        test_names = list(tests.keys())
        
        if not test_names:
            ax.text(0.5, 0.5, 'No trial data available', ha='center', va='center')
            ax.set_title('Stability Heatmap')
            return
        
        means = []
        cvs = []
        
        for name in test_names:
            data = tests[name]
            if "mean_score" in data and "cv" in data:
                means.append(data["mean_score"])
                cvs.append(data["cv"] * 100)
            else:
                means.append(0)
                cvs.append(100)  # High CV for failed tests
        
        # Create scatter plot with color coding
        colors = ['green' if cv < 20 and mean > 0.1 else 
                 'orange' if cv < 20 or mean > 0.1 else 'red' 
                 for cv, mean in zip(cvs, means)]
        
        scatter = ax.scatter(means, cvs, c=colors, s=100, alpha=0.7, edgecolor='black')
        
        # Add test name labels
        for i, name in enumerate(test_names):
            ax.annotate(name.replace('_', ' ').title(), 
                       (means[i], cvs[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='CV Threshold')
        ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, label='Score Threshold')
        
        ax.set_xlabel('Mean Composite Score')
        ax.set_ylabel('Coefficient of Variation (%)')
        ax.set_title('Stability Heatmap (Score vs CV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_stability(self, ax, tests):
        """Scatter plot of performance vs stability."""
        performance_scores = []
        stability_scores = []
        test_names = []
        
        for name, data in tests.items():
            if "mean_score" in data and "cv" in data:
                performance_scores.append(data["mean_score"])
                stability_scores.append(1 / (1 + data["cv"]))  # Inverse CV as stability
                test_names.append(name.replace('_', ' ').title())
        
        colors = ['green' if perf > 0.1 and stab > 0.5 else 'red' 
                 for perf, stab in zip(performance_scores, stability_scores)]
        
        ax.scatter(performance_scores, stability_scores, c=colors, s=100, alpha=0.7)
        
        for i, name in enumerate(test_names):
            ax.annotate(name, (performance_scores[i], stability_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Performance (Mean Score)')
        ax.set_ylabel('Stability (1/(1+CV))')
        ax.set_title('Performance vs Stability')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
    
    def _plot_wobble_analysis(self, ax, unstable_tests):
        """Analyze wobble patterns in unstable tests."""
        if not unstable_tests:
            ax.text(0.5, 0.5, 'No unstable tests to analyze', ha='center', va='center')
            ax.set_title('Wobble Analysis')
            return
        
        wobble_data = []
        names = []
        
        for name, data in unstable_tests:
            if "trials" in data:
                trials = data["trials"]
                scores = [t.get("composite_score", 0) for t in trials]
                
                if len(scores) > 1:
                    # Calculate wobble intensity (variance of differences)
                    diffs = np.diff(scores)
                    wobble_intensity = np.var(diffs) if len(diffs) > 0 else 0
                    wobble_data.append(wobble_intensity)
                    names.append(name.replace('_', ' ').title())
        
        if wobble_data:
            bars = ax.bar(range(len(names)), wobble_data, color='red', alpha=0.7)
            ax.set_xlabel('Unstable Tests')
            ax.set_ylabel('Wobble Intensity (Variance of Score Differences)')
            ax.set_title('Wobble Analysis: Where Tests Fluctuate Most')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            
            # Add values on bars
            for i, value in enumerate(wobble_data):
                ax.text(i, value + max(wobble_data) * 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3)
    
    def print_stability_report(self):
        """Print detailed stability analysis report."""
        tests = self.results.get("tests", {})
        summary = self.results.get("summary", {})
        
        print(f"\n🔍 GAIA Stability Analysis Report - {self.timestamp}")
        print("=" * 60)
        print(f"Overall Performance: {summary.get('success_rate', 0):.0%} ({summary.get('intelligence_level', 'UNKNOWN')})")
        print()
        
        stable_tests = []
        unstable_tests = []
        
        for name, data in tests.items():
            if "cv" in data:
                if data.get("stable", False):
                    stable_tests.append((name, data))
                else:
                    unstable_tests.append((name, data))
        
        print("🔒 STABLE TESTS:")
        for name, data in stable_tests:
            cv = data["cv"] * 100
            score = data["mean_score"]
            print(f"  ✅ {name.replace('_', ' ').title():25} Score: {score:.3f}±{data['std_score']:.3f} (CV: {cv:.1f}%)")
        
        print("\n🔀 UNSTABLE TESTS:")
        for name, data in unstable_tests:
            cv = data["cv"] * 100
            score = data.get("mean_score", 0)
            print(f"  ❌ {name.replace('_', ' ').title():25} Score: {score:.3f}±{data.get('std_score', 0):.3f} (CV: {cv:.1f}%)")
            
            # Analyze wobble pattern
            if "trials" in data:
                trials = data["trials"]
                scores = [t.get("composite_score", 0) for t in trials]
                if len(scores) > 1:
                    trend = "↗" if scores[-1] > scores[0] else "↘" if scores[-1] < scores[0] else "→"
                    range_val = max(scores) - min(scores)
                    print(f"     Trend: {trend} Range: {range_val:.3f} Trials: {scores}")
        
        print(f"\n📊 KEY INSIGHTS:")
        if unstable_tests:
            most_unstable = max(unstable_tests, key=lambda x: x[1]["cv"])
            print(f"  • Most unstable: {most_unstable[0].replace('_', ' ').title()} (CV: {most_unstable[1]['cv']*100:.1f}%)")
        
        if stable_tests:
            most_stable = min(stable_tests, key=lambda x: x[1]["cv"])
            print(f"  • Most stable: {most_stable[0].replace('_', ' ').title()} (CV: {most_stable[1]['cv']*100:.1f}%)")
        
        print(f"  • Stability rate: {len(stable_tests)}/{len(stable_tests + unstable_tests)} tests")


def main():
    """Run stability dashboard analysis."""
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        results_path = None
    
    dashboard = StabilityDashboard(results_path)
    dashboard.print_stability_report()
    dashboard.plot_stability_analysis()


if __name__ == "__main__":
    main()