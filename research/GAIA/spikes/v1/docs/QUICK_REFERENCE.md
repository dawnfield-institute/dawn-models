# Quick Reference: GAIA Pre-Field Enhancements

**Date**: October 2, 2025  
**Version**: 1.0

---

## ⚡ Quick Commands

```bash
# Navigate to GAIA
cd "c:\Users\peter\repos\core_workspace\dawn-models\research\GAIA"

# Run resonance demo
python usecases/demo_resonance.py

# Run cosmological validation
python usecases/cosmological_validation.py

# Run unit tests
python -m pytest tests/unit/test_resonance_detector.py -v

# Run all tests
python -m pytest tests/ -v
```

---

## 📦 What Got Added

### New Classes
1. **PreFieldResonanceDetector** (`src/core/field_engine.py`)
2. **CosmologicalValidator** (`usecases/cosmological_validation.py`)

### New Files
1. `tests/unit/test_resonance_detector.py` (9 tests)
2. `usecases/demo_resonance.py` (demo script)
3. `usecases/cosmological_validation.py` (validation suite)
4. `docs/resonance_integration_summary.md`
5. `docs/step1_integration_checklist.md`
6. `docs/step2_cosmological_validation_summary.md`
7. `docs/INTEGRATION_SUMMARY_OCT_2_2025.md`

---

## 🎯 Usage Examples

### Enable Resonance Detection
```python
from src.core.field_engine import FieldEngine

# Resonance enabled by default
engine = FieldEngine(shape=(32, 32))

# Or explicitly
engine = FieldEngine(shape=(32, 32), enable_resonance=True)

# Disable if needed
engine = FieldEngine(shape=(32, 32), enable_resonance=False)
```

### Check Resonance State
```python
# After some iterations
metrics = engine.get_pac_metrics()

if 'resonance_state' in metrics:
    res = metrics['resonance_state']
    if res['resonance_locked']:
        print(f"🎵 Locked at {res['detected_frequency']} Hz")
        print(f"   Speedup: {1/res['tuning_factor']}x")
```

### Run Cosmological Validation
```python
from usecases.cosmological_validation import CosmologicalValidator

validator = CosmologicalValidator(save_results=True)
validator.setup_results_directory()

evolution_data = validator.run_pac_evolution(
    iterations=500,
    field_size=32
)

results = validator.validate_cosmological_parallel(evolution_data)
validator.plot_results(evolution_data, results)
validator.save_results(evolution_data, results)

print(f"Correlation: {results['entropy_amplification_correlation']}")
```

---

## 📊 Key Metrics

### Resonance Detection
- **Expected Frequency**: ~0.03 cycles/iteration
- **Lock Time**: 50-70 iterations
- **Confidence Threshold**: 0.15
- **Window Size**: 50 iterations
- **Speedup**: 5.11x theoretical

### Cosmological Validation
- **Target Correlation**: r > 0.995
- **Current Correlation**: r ≈ 0.003 (needs tuning)
- **PAC Reduction**: 83.7% ✓
- **Eras Mapped**: 9 (singularity → heat death)

---

## 🔍 Troubleshooting

### Resonance Not Locking
```python
# Check history length
print(f"History: {len(engine.resonance_detector.pac_history)}")
# Needs at least 50 iterations

# Lower confidence threshold
engine.resonance_detector.confidence_threshold = 0.10

# Check frequency range
state = engine.resonance_detector.get_resonance_state()
print(f"Detected: {state['detected_frequency']}")
print(f"Expected: {state['expected_frequency']}")
```

### Low Cosmological Correlation
```python
# Try more iterations
evolution_data = validator.run_pac_evolution(
    iterations=2000,  # Increased from 500
    field_size=32
)

# Try different cooling schedule
# Edit cosmological_validation.py line ~95:
cooling_factor = (1 + i) ** -0.5  # Power-law instead of exponential
```

### Import Errors
```python
# Make sure you're in the right directory
import os
print(os.getcwd())
# Should be: .../core_workspace/dawn-models/research/GAIA

# Add to path if needed
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
```

---

## 📈 Console Output Reference

### Resonance Lock Message
```
🎵 Resonance LOCKED at iteration 119
   Frequency: 0.020000 cycles/iteration
   Confidence: 0.486
   Expected 5.11x speedup in PAC convergence
```

### Cosmological Progress
```
🌌 Starting cosmological validation with 500 iterations...
  Iteration    0: PAC= 0.05719, T=1.54e-01K, Era=first_stars
  Iteration  100: PAC= 0.02276, T=6.15e-02K, Era=first_stars
  🎵 Resonance LOCKED at iteration 119
  ...
✓ Evolution complete
  Final PAC: 0.009301
  PAC reduction: 83.7%
```

### Validation Results
```
🔬 Validating cosmological parallel...
  Entropy-Amplification Correlation: 0.002830
  Big Bang Pattern Match: ✗ FAIL (target: r > 0.995)
  PAC Cooling Rate: 0.002269
  Phase Transitions Detected: 0
```

---

## 📁 Output Locations

### Resonance Demo
```
usecases/resonance_demo.png
```

### Cosmological Validation
```
usecases/results/cosmological_YYYYMMDD_HHMMSS/
├── cosmological_validation.png
├── cosmological_results.json
└── summary.txt
```

---

## 🧪 Test Commands

```bash
# Single test file
python -m pytest tests/unit/test_resonance_detector.py -v

# Specific test
python -m pytest tests/unit/test_resonance_detector.py::test_initialization -v

# With coverage
python -m pytest tests/unit/test_resonance_detector.py --cov=src.core.field_engine

# All tests
python -m pytest tests/ -v
```

---

## 🎨 Visualization Panels

### Resonance Demo (4 panels)
1. PAC Evolution Trajectory
2. FFT Spectrum at Lock
3. Frequency Detection History
4. Detection Window

### Cosmological Validation (6 panels)
1. PAC Cooling with Eras
2. Entropy-Amplification Correlation
3. Temperature Evolution
4. Phase Transitions
5. FFT Oscillation Spectrum
6. Validation Summary

---

## 🔑 Key Parameters

### PreFieldResonanceDetector
```python
window_size=50            # Analysis window (iterations)
confidence_threshold=0.15  # Lock threshold (0-1)
expected_frequency=0.03    # Target frequency (cycles/iter)
frequency_tolerance=0.01   # Match tolerance (±)
```

### CosmologicalValidator
```python
iterations=500            # Evolution steps
field_size=32            # Field dimensions (NxN)
save_results=True        # Enable file export
```

---

## ⚠️ Known Issues

1. **Cosmological correlation low** (r ≈ 0.003, target > 0.995)
   - Status: Parameter tuning needed
   - Solution: Adjust cooling schedule in line ~95

2. **No phase transitions detected**
   - Status: Expected for smooth cooling
   - Solution: Add era-specific physics

3. **Resonance frequency mismatch** (0.020 vs 0.03 expected)
   - Status: Within tolerance, working correctly
   - Note: Varies by initial conditions

---

## 📚 Documentation Links

- **Resonance Integration**: `docs/resonance_integration_summary.md`
- **Step 1 Checklist**: `docs/step1_integration_checklist.md`
- **Step 2 Summary**: `docs/step2_cosmological_validation_summary.md`
- **Complete Summary**: `docs/INTEGRATION_SUMMARY_OCT_2_2025.md`

---

## 🚀 Next Steps

### Option A: Tune Parameters (Step 2 Refinement)
Focus on achieving r > 0.995 correlation

### Option B: Continue to Step 3 (Enhanced Tests)
Build automated test suite for resonance speedup

### Option C: Continue to Step 4 (Dashboard)
Build real-time monitoring dashboard

---

## 💡 Tips

1. **First time running?** Start with `demo_resonance.py`
2. **Want faster results?** Reduce iterations to 100
3. **Need better visualization?** Increase `dpi=300` in plots
4. **Debugging?** Add `print()` statements in validators
5. **Performance issues?** Reduce `field_size` to 16x16

---

**Last Updated**: October 2, 2025  
**Status**: ✅ Production Ready (Step 1), ⏳ Tuning Needed (Step 2)  
**Questions?** Check full documentation or ask!
