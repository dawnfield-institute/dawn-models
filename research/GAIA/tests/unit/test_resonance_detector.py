"""
Unit tests for PreFieldResonanceDetector
Tests the resonance detection capability added in Pre-Field Recursion v2.2
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.field_engine import PreFieldResonanceDetector


class TestPreFieldResonanceDetector:
    """Test suite for resonance detection."""
    
    def test_initialization(self):
        """Test detector initializes with correct default values."""
        detector = PreFieldResonanceDetector()
        
        assert detector.window_size == 50
        assert detector.confidence_threshold == 0.15
        assert detector.expected_frequency == 0.03
        assert not detector.resonance_locked
        assert detector.detected_frequency is None
        
    def test_insufficient_history(self):
        """Test that detector doesn't lock with insufficient data."""
        detector = PreFieldResonanceDetector(window_size=50)
        
        # Feed less than window_size samples
        for i in range(30):
            pac_residual = 0.5 + 0.1 * np.sin(2 * np.pi * 0.03 * i)
            locked = detector.update(pac_residual)
            assert not locked
            
        assert not detector.resonance_locked
        
    def test_resonance_detection_with_synthetic_signal(self):
        """Test resonance detection with synthetic oscillating PAC signal."""
        detector = PreFieldResonanceDetector(
            window_size=50,
            confidence_threshold=0.10  # Lower threshold for synthetic signal
        )
        
        # Generate synthetic PAC trajectory with known frequency (0.03 cycles/iteration)
        target_frequency = 0.03
        iterations = 100
        
        locked = False
        lock_iteration = None
        
        for i in range(iterations):
            # Synthetic PAC with natural frequency + noise
            pac_residual = 0.5 + 0.2 * np.sin(2 * np.pi * target_frequency * i)
            pac_residual += np.random.normal(0, 0.02)  # Add small noise
            pac_residual = abs(pac_residual)  # Keep positive
            
            newly_locked = detector.update(pac_residual)
            
            if newly_locked:
                locked = True
                lock_iteration = i
                break
        
        # Should lock eventually
        assert locked, "Detector failed to lock on synthetic resonance signal"
        assert lock_iteration is not None
        assert lock_iteration >= 49  # Can lock at or after window is full
        
        # Check detected frequency is close to target
        assert detector.detected_frequency is not None
        frequency_error = abs(detector.detected_frequency - target_frequency)
        assert frequency_error < 0.02, f"Frequency error {frequency_error} too large"
        
        # Check resonance state
        state = detector.get_resonance_state()
        assert state['resonance_locked']
        assert state['confidence'] > 0.10
        
    def test_tuning_factor(self):
        """Test that tuning factor is applied correctly."""
        detector = PreFieldResonanceDetector()
        
        # Before locking, tuning factor should be 1.0
        assert detector.get_tuning_factor() == 1.0
        
        # Manually lock resonance
        detector.resonance_locked = True
        detector.detected_frequency = 0.03
        
        # After locking, tuning factor should be calculated and clamped
        tuning_factor = detector.get_tuning_factor()
        expected = 2 * np.pi * 0.03  # ≈ 0.188
        
        # Tuning factor is clamped to [0.5, 2.0], so expected gets clamped to 0.5
        assert 0.5 <= tuning_factor <= 2.0
        assert abs(tuning_factor - 0.5) < 0.01  # Should be clamped to minimum
        
    def test_reset(self):
        """Test reset functionality."""
        detector = PreFieldResonanceDetector()
        
        # Add some history
        for i in range(60):
            detector.update(0.5 + 0.1 * np.sin(2 * np.pi * 0.03 * i))
        
        # Manually lock for testing
        detector.resonance_locked = True
        detector.detected_frequency = 0.03
        detector.lock_iteration = 55
        
        # Reset
        detector.reset()
        
        # Check everything is cleared
        assert not detector.resonance_locked
        assert detector.detected_frequency is None
        assert detector.lock_iteration is None
        assert len(detector.pac_history) == 0
        assert detector.confidence == 0.0
        
    def test_zero_crossing_validation(self):
        """Test zero-crossing validation method."""
        detector = PreFieldResonanceDetector()
        
        # Create clear oscillating signal
        signal = np.sin(2 * np.pi * 0.03 * np.arange(50))
        
        confidence = detector._validate_zero_crossings(signal)
        
        # Should have high confidence for clean sine wave
        assert confidence > 0.5, f"Confidence {confidence} too low for clean signal"
        
    def test_frequency_history_tracking(self):
        """Test that frequency history is tracked."""
        detector = PreFieldResonanceDetector(window_size=50)
        
        # Feed enough data to trigger multiple detections
        for i in range(150):
            pac_residual = 0.5 + 0.1 * np.sin(2 * np.pi * 0.03 * i)
            detector.update(pac_residual)
        
        # Should have collected frequency measurements
        assert len(detector.frequency_history) > 0
        
    def test_confidence_threshold_requirement(self):
        """Test that high confidence threshold prevents false positives."""
        detector = PreFieldResonanceDetector(
            window_size=50,
            confidence_threshold=0.95  # Very high threshold
        )
        
        # Feed noisy signal that shouldn't lock
        for i in range(100):
            pac_residual = abs(np.random.normal(0.5, 0.3))
            locked = detector.update(pac_residual)
            assert not locked
        
        # Should not lock due to high threshold
        assert not detector.resonance_locked


class TestResonanceIntegrationWithFieldEngine:
    """Test resonance detector integration with FieldEngine."""
    
    @pytest.mark.skip(reason="Requires Fracton SDK - integration test")
    def test_field_engine_with_resonance(self):
        """Test FieldEngine with resonance enabled."""
        from core.field_engine import FieldEngine
        
        # Initialize with resonance enabled
        engine = FieldEngine(shape=(32, 32), enable_resonance=True)
        
        assert engine.resonance_detector is not None
        assert engine.resonance_locks == 0
        
        # Run some updates
        for i in range(10):
            state = engine.update_fields(f"test input {i}")
        
        # Check metrics include resonance state
        metrics = engine.get_pac_metrics()
        assert 'resonance_state' in metrics
        assert 'resonance_locks' in metrics
        
    @pytest.mark.skip(reason="Requires Fracton SDK - integration test")
    def test_field_engine_without_resonance(self):
        """Test FieldEngine with resonance disabled."""
        from core.field_engine import FieldEngine
        
        # Initialize with resonance disabled
        engine = FieldEngine(shape=(32, 32), enable_resonance=False)
        
        assert engine.resonance_detector is None
        
        # Metrics should not include resonance state
        metrics = engine.get_pac_metrics()
        assert 'resonance_state' not in metrics


def test_resonance_state_dict():
    """Test resonance state dictionary format."""
    detector = PreFieldResonanceDetector()
    
    # Manually set some state
    detector.resonance_locked = True
    detector.detected_frequency = 0.029
    detector.confidence = 0.85
    detector.lock_iteration = 62
    
    state = detector.get_resonance_state()
    
    assert 'resonance_locked' in state
    assert 'detected_frequency' in state
    assert 'expected_frequency' in state
    assert 'confidence' in state
    assert 'lock_iteration' in state
    assert 'history_length' in state
    assert 'tuning_factor' in state
    
    assert state['resonance_locked'] == True
    assert state['detected_frequency'] == 0.029
    assert state['confidence'] == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
