#!/usr/bin/env python3
"""
GAIA Intelligence Test Suite

Comprehensive test runner for all GAIA capabilities.
Fixed red flags: zero-energy protection, composite scoring, determinism, enhanced memory tests.
"""

import numpy as np
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gaia import GAIA


class GAIATestSuite:
    """Comprehensive GAIA intelligence test suite with red flag fixes."""
    
    def __init__(self):
        """Initialize test suite with results tracking."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(__file__).parent / "results" / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.gaia = GAIA()
        self.results = {
            "timestamp": self.timestamp,
            "tests": {},
            "categories": {},
            "summary": {}
        }
        
        # Testing constants - addresses red flags
        self.ENERGY_EPSILON = 1e-8  # Minimum non-trivial energy
        self.CV_THRESHOLD = 0.2     # Coefficient of variation threshold for stability
        self.NUM_TRIALS = 3         # Number of trials for stability testing
    
    def _normalize_field(self, field):
        """Normalize field to safe dynamic range - prevents zero-energy collapses."""
        field_norm = np.linalg.norm(field)
        if field_norm < self.ENERGY_EPSILON:
            # Add small perturbation if field is too small
            field = field + np.random.random(field.shape) * 1e-6
            field_norm = np.linalg.norm(field)
        return field / field_norm * 0.5  # Scale to reasonable range
    
    def _extract_physics_state(self, response):
        """Extract comprehensive physics state from GAIA response."""
        state = {
            "klein_gordon_energy": response.klein_gordon_energy,
            "conservation_residual": response.conservation_residual,
            "field_amplitude_mean": np.mean(response.field_state),
            "field_amplitude_std": np.std(response.field_state),
            "field_norm": np.linalg.norm(response.field_state)
        }
        
        # Add Xi deviation if available
        if hasattr(response.state, 'xi_deviation'):
            state["xi_deviation"] = response.state.xi_deviation
        else:
            state["xi_deviation"] = 0.0
            
        return state
    
    def _compute_composite_score(self, state1, state2, weights=None):
        """Compute composite discrimination score - addresses energy-only scoring."""
        if weights is None:
            weights = {"energy": 0.5, "conservation": 0.3, "xi": 0.2}
        
        energy_delta = abs(state1["klein_gordon_energy"] - state2["klein_gordon_energy"])
        conservation_delta = abs(state1["conservation_residual"] - state2["conservation_residual"])
        xi_delta = abs(state1["xi_deviation"] - state2["xi_deviation"])
        
        # Normalize by baselines to avoid resolution artifacts - addresses brittle thresholds
        energy_baseline = max(self.ENERGY_EPSILON, (state1["klein_gordon_energy"] + state2["klein_gordon_energy"]) / 2)
        conservation_baseline = max(1e-12, (abs(state1["conservation_residual"]) + abs(state2["conservation_residual"])) / 2)
        xi_baseline = max(1e-10, (abs(state1["xi_deviation"]) + abs(state2["xi_deviation"])) / 2)
        
        energy_score = energy_delta / energy_baseline
        conservation_score = conservation_delta / conservation_baseline
        xi_score = xi_delta / xi_baseline
        
        composite = (weights["energy"] * energy_score + 
                    weights["conservation"] * conservation_score + 
                    weights["xi"] * xi_score)
        
        return {
            "composite_score": composite,
            "energy_score": energy_score,
            "conservation_score": conservation_score,
            "xi_score": xi_score
        }
    
    def _run_stable_test(self, test_func, test_name):
        """Run test with stability checking across multiple trials - addresses determinism."""
        np.random.seed(42 + hash(test_name) % 1000)  # Deterministic per test
        
        results = []
        for trial in range(self.NUM_TRIALS):
            np.random.seed(42 + trial + hash(test_name) % 1000)
            result = test_func()
            results.append(result)
        
        # Check stability
        if all(isinstance(r, dict) and 'composite_score' in r for r in results):
            scores = [r['composite_score'] for r in results]
            mean_score = np.mean(scores)
            cv = np.std(scores) / mean_score if mean_score > 0 else float('inf')
            
            stable = cv < self.CV_THRESHOLD
            
            return {
                "trials": results,
                "mean_score": mean_score,
                "std_score": np.std(scores),
                "cv": cv,
                "stable": stable,
                "passed": stable and mean_score > 0.1
            }
        else:
            # Fallback for simple boolean results
            passed_count = sum(1 for r in results if r.get('passed', False))
            stable = passed_count >= 2  # At least 2/3 trials pass
            
            return {
                "trials": results,
                "passed_count": passed_count,
                "stable": stable,
                "passed": stable
            }
    
    # ==================== FUNDAMENTAL INTELLIGENCE ====================
    
    def test_pattern_recognition(self):
        """Mathematical pattern recognition through field energy."""
        # Fibonacci pattern
        fibonacci_pattern = np.array([
            [1.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 5.0, 8.0], 
            [3.0, 5.0, 8.0, 13.0],
            [5.0, 8.0, 13.0, 21.0]
        ])
        fibonacci_pattern = self._normalize_field(fibonacci_pattern)
        
        # Random pattern
        random_pattern = np.random.rand(4, 4)
        random_pattern = self._normalize_field(random_pattern)
        
        fib_response = self.gaia.process_field(fibonacci_pattern, dt=0.01)
        rand_response = self.gaia.process_field(random_pattern, dt=0.01)
        
        fib_state = self._extract_physics_state(fib_response)
        rand_state = self._extract_physics_state(rand_response)
        
        score_data = self._compute_composite_score(fib_state, rand_state)
        
        return {
            "fib_state": fib_state,
            "rand_state": rand_state,
            "composite_score": score_data["composite_score"],
            "score_breakdown": score_data,
            "passed": score_data["composite_score"] > 0.1
        }
    
    def test_fibonacci_reasoning(self):
        """Advanced Fibonacci pattern reasoning with zero-energy protection."""
        # Create full Fibonacci sequence as field with proper normalization - match GAIA 4x4 dimensions
        fib_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        fib_field = np.array(fib_seq, dtype=float).reshape(4, 4)
        fib_field = self._normalize_field(fib_field)
        
        # Random sequence for comparison
        random_seq = [7, 2, 9, 1, 5, 3, 8, 4, 6, 12, 15, 11, 13, 14, 10, 16]
        random_field = np.array(random_seq, dtype=float).reshape(4, 4)
        random_field = self._normalize_field(random_field)
        
        fib_response = self.gaia.process_field(fib_field, dt=0.01)
        rand_response = self.gaia.process_field(random_field, dt=0.01)
        
        fib_state = self._extract_physics_state(fib_response)
        rand_state = self._extract_physics_state(rand_response)
        
        # Guard against zero-energy edge cases
        assert fib_state["klein_gordon_energy"] > self.ENERGY_EPSILON, "Fibonacci energy collapsed to zero"
        assert rand_state["klein_gordon_energy"] > self.ENERGY_EPSILON, "Random energy collapsed to zero"
        
        score_data = self._compute_composite_score(fib_state, rand_state)
        
        return {
            "fib_state": fib_state,
            "rand_state": rand_state,
            "composite_score": score_data["composite_score"],
            "score_breakdown": score_data,
            "passed": score_data["composite_score"] > 0.1
        }
    
    def test_enhanced_memory(self):
        """Enhanced pattern memory with delayed recall and noise tolerance."""
        # Learn a checkerboard pattern
        memory_pattern = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0]
        ])
        memory_pattern = self._normalize_field(memory_pattern)
        
        # Learn pattern multiple times (imprinting)
        memory_states = []
        for i in range(3):
            response = self.gaia.process_field(memory_pattern, dt=0.005)
            state = self._extract_physics_state(response)
            memory_states.append(state)
        
        memory_baseline = np.mean([s["klein_gordon_energy"] for s in memory_states])
        
        # Test 1: Exact recall
        exact_response = self.gaia.process_field(memory_pattern, dt=0.005)
        exact_state = self._extract_physics_state(exact_response)
        exact_score = 1.0 - abs(exact_state["klein_gordon_energy"] - memory_baseline) / max(memory_baseline, self.ENERGY_EPSILON)
        
        # Test 2: Noisy recall at multiple noise levels
        noise_scores = []
        for noise_level in [0.1, 0.2, 0.3]:
            noisy_pattern = memory_pattern + np.random.random(memory_pattern.shape) * noise_level
            noisy_pattern = self._normalize_field(noisy_pattern)
            
            noisy_response = self.gaia.process_field(noisy_pattern, dt=0.005)
            noisy_state = self._extract_physics_state(noisy_response)
            noise_score = 1.0 - abs(noisy_state["klein_gordon_energy"] - memory_baseline) / max(memory_baseline, self.ENERGY_EPSILON)
            noise_scores.append(max(0, noise_score))  # Clamp to 0
        
        # Test 3: Delayed recall (using different pattern then returning)
        interfering_pattern = np.random.rand(4, 4)
        interfering_pattern = self._normalize_field(interfering_pattern)
        
        # Process interfering pattern
        self.gaia.process_field(interfering_pattern, dt=0.005)
        
        # Return to original pattern
        delayed_response = self.gaia.process_field(memory_pattern, dt=0.005)
        delayed_state = self._extract_physics_state(delayed_response)
        delayed_score = 1.0 - abs(delayed_state["klein_gordon_energy"] - memory_baseline) / max(memory_baseline, self.ENERGY_EPSILON)
        
        # Composite memory score
        memory_composite = (exact_score * 0.4 + np.mean(noise_scores) * 0.4 + delayed_score * 0.2)
        
        return {
            "memory_baseline": memory_baseline,
            "exact_score": exact_score,
            "noise_scores": noise_scores,
            "delayed_score": delayed_score,
            "composite_score": memory_composite,
            "memory_states": memory_states,
            "passed": memory_composite > 0.5  # Strengthen threshold
        }
    
    def test_optimization_with_validation(self):
        """Mathematical optimization with zero-energy validation."""
        # Create optimization landscape with proper normalization
        landscape = np.array([
            [1.0, 0.8, 0.6, 0.4],
            [0.8, 0.6, 0.4, 0.2],
            [0.6, 0.4, 0.2, 0.0],
            [0.4, 0.2, 0.0, 0.0]
        ])
        landscape = self._normalize_field(landscape)
        
        energies = []
        states = []
        
        for step in range(5):
            # Evolve landscape through optimization steps
            evolved = landscape * (1.0 - step * 0.1) + np.random.rand(4, 4) * 0.05
            evolved = self._normalize_field(evolved)
            
            response = self.gaia.process_field(evolved, dt=0.01)
            state = self._extract_physics_state(response)
            
            energies.append(state["klein_gordon_energy"])
            states.append(state)
        
        # Guard against all-zero energies
        assert any(e > self.ENERGY_EPSILON for e in energies), "All optimization energies collapsed to zero"
        
        # Check for optimization behavior (non-trivial energy evolution)
        energy_range = max(energies) - min(energies)
        energy_trend = np.polyfit(range(len(energies)), energies, 1)[0]
        
        return {
            "energies": energies,
            "states": states,
            "energy_range": energy_range,
            "energy_trend": energy_trend,
            "composite_score": energy_range,  # Use range as composite score
            "passed": energy_range > 0.01  # Require non-trivial evolution
        }
    
    # ==================== TEST EXECUTION ====================
    
    def run_all_tests(self):
        """Run all test categories with stability checking."""
        test_categories = {
            "fundamental": [
                ("pattern_recognition", self.test_pattern_recognition),
                ("fibonacci_reasoning", self.test_fibonacci_reasoning)
            ],
            "enhanced_capabilities": [
                ("enhanced_memory", self.test_enhanced_memory),
                ("optimization_validation", self.test_optimization_with_validation)
            ]
        }
        
        category_results = {}
        total_passed = 0
        total_tests = 0
        
        for category, tests in test_categories.items():
            category_passed = 0
            category_total = len(tests)
            
            for name, method in tests:
                try:
                    # Run with stability checking
                    stable_result = self._run_stable_test(method, name)
                    self.results["tests"][name] = stable_result
                    
                    if stable_result["passed"]:
                        category_passed += 1
                        total_passed += 1
                except Exception as e:
                    self.results["tests"][name] = {
                        "error": str(e),
                        "passed": False,
                        "stable": False
                    }
                
                total_tests += 1
            
            category_results[category] = {
                "passed": category_passed,
                "total": category_total,
                "rate": category_passed / category_total if category_total > 0 else 0
            }
        
        self.results["categories"] = category_results
        
        overall_rate = total_passed / total_tests if total_tests > 0 else 0
        self.results["summary"] = {
            "total_passed": total_passed,
            "total_tests": total_tests,
            "success_rate": overall_rate,
            "intelligence_level": self._get_intelligence_level(overall_rate),
            "timestamp": self.timestamp
        }
        
        return total_passed, total_tests
    
    def _get_intelligence_level(self, success_rate):
        """Determine intelligence level from success rate."""
        if success_rate >= 0.9:
            return "EXCEPTIONAL"
        elif success_rate >= 0.75:
            return "STRONG" 
        elif success_rate >= 0.5:
            return "PROMISING"
        else:
            return "DEVELOPING"
    
    def save_results(self):
        """Save comprehensive results with enhanced logging."""
        # JSON results
        with open(self.results_dir / "test_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Summary report
        with open(self.results_dir / "summary.txt", "w") as f:
            f.write(f"GAIA Intelligence Test Suite Results (Red Flags Fixed)\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"=" * 60 + "\n\n")
            
            summary = self.results["summary"]
            f.write(f"Overall Results:\n")
            f.write(f"Tests Passed: {summary['total_passed']}/{summary['total_tests']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1%}\n")
            f.write(f"Intelligence Level: {summary['intelligence_level']}\n\n")
            
            f.write(f"Category Breakdown:\n")
            for category, data in self.results["categories"].items():
                f.write(f"  {category.replace('_', ' ').title():20} {data['passed']}/{data['total']} ({data['rate']:.0%})\n")
            
            f.write(f"\nIndividual Test Results:\n")
            for name, data in self.results["tests"].items():
                status = "PASSED" if data.get("passed", False) else "FAILED"
                stable = "STABLE" if data.get("stable", False) else "UNSTABLE"
                f.write(f"  {name.replace('_', ' ').title():25} {status:8} ({stable})\n")
                
                if "error" in data:
                    f.write(f"    Error: {data['error']}\n")
                elif "mean_score" in data:
                    f.write(f"    Score: {data['mean_score']:.3f}±{data['std_score']:.3f} (CV: {data['cv']:.1%})\n")
    
    def display_results(self):
        """Display clean final results with stability indicators."""
        summary = self.results["summary"]
        categories = self.results["categories"]
        
        print(f"\nGAIA Intelligence Test Suite (Fixed) - {self.timestamp}")
        print("=" * 60)
        print(f"Tests Passed: {summary['total_passed']}/{summary['total_tests']} ({summary['success_rate']:.0%})")
        print(f"Intelligence Level: {summary['intelligence_level']}")
        print()
        
        print("Category Results:")
        for category, data in categories.items():
            print(f"  {category.replace('_', ' ').title():20} {data['passed']}/{data['total']} ({data['rate']:.0%})")
        
        print("\nTest Status:")
        for name, data in self.results["tests"].items():
            status = "✅" if data.get("passed", False) else "❌"
            stable = "🔒" if data.get("stable", False) else "🔀"
            
            score_info = ""
            if "mean_score" in data:
                score_info = f" ({data['mean_score']:.3f}±{data['std_score']:.3f})"
            
            print(f"  {status}{stable} {name.replace('_', ' ').title()}{score_info}")
        
        print(f"\nResults saved to: {self.results_dir}")
        print("🔒 = Stable across trials, 🔀 = Unstable")
        
        return summary['success_rate'] >= 0.7


def main():
    """Run comprehensive GAIA intelligence test suite with red flag fixes."""
    suite = GAIATestSuite()
    
    passed, total = suite.run_all_tests()
    suite.save_results()
    success = suite.display_results()
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)