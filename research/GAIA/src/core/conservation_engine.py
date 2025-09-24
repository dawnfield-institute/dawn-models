"""
Conservation Engine for GAIA
Native implementation of thermodynamic conservation laws for cognitive processes.
Inspired by PAC conservation mathematics but tuned for GAIA's entropy dynamics.
"""

import numpy as np
import math
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ConservationMode(Enum):
    """Types of conservation validation."""
    ENERGY_ONLY = "energy_only"
    INFORMATION_ONLY = "information_only"
    ENERGY_INFORMATION = "energy_information"
    FULL_THERMODYNAMIC = "full_thermodynamic"


@dataclass
class ConservationResult:
    """Result of conservation validation."""
    is_valid: bool
    violation_magnitude: float
    conservation_mode: ConservationMode
    energy_balance: float
    information_balance: float
    entropy_change: float
    corrected_values: Optional[Dict[str, float]] = None


class ConservationEngine:
    """
    GAIA-native conservation engine for validating and correcting 
    thermodynamic violations in cognitive field dynamics.
    """
    
    def __init__(self, 
                 mode: ConservationMode = ConservationMode.ENERGY_INFORMATION,
                 tolerance: float = 0.1,
                 k_boltzmann: float = 1.380649e-23,
                 temperature: float = 300.0):
        self.mode = mode
        self.tolerance = tolerance
        self.k_boltzmann = k_boltzmann
        self.temperature = temperature
        self.conservation_history = []
        self.violation_count = 0
        self.total_validations = 0
    
    def validate_conservation(self, 
                            initial_energy: float,
                            final_energy: float,
                            initial_information: float,
                            final_information: float,
                            mode: ConservationMode = ConservationMode.ENERGY_INFORMATION) -> ConservationResult:
        """
        Validate conservation laws for cognitive field transitions.
        
        Args:
            initial_energy: Energy before cognitive operation
            final_energy: Energy after cognitive operation  
            initial_information: Information before operation
            final_information: Information after operation
            mode: Type of conservation to validate
            
        Returns:
            ConservationResult with validation outcome
        """
        self.total_validations += 1
        
        # Calculate energy and information changes
        energy_change = final_energy - initial_energy
        information_change = final_information - initial_information
        
        # Calculate conservation violations
        energy_violation = self._calculate_energy_violation(energy_change)
        info_violation = self._calculate_information_violation(information_change)
        
        # Determine overall violation based on mode
        if mode == ConservationMode.ENERGY_ONLY:
            total_violation = energy_violation
        elif mode == ConservationMode.INFORMATION_ONLY:
            total_violation = info_violation
        elif mode == ConservationMode.ENERGY_INFORMATION:
            total_violation = max(energy_violation, info_violation)
        else:  # FULL_THERMODYNAMIC
            total_violation = self._calculate_full_thermodynamic_violation(
                energy_change, information_change
            )
        
        # Check if within tolerance
        is_valid = total_violation <= self.tolerance
        
        if not is_valid:
            self.violation_count += 1
        
        # Calculate balances
        energy_balance = 1.0 - min(energy_violation / max(self.tolerance, 0.01), 1.0)
        info_balance = 1.0 - min(info_violation / max(self.tolerance, 0.01), 1.0)
        
        # Calculate entropy change (Landauer principle)
        entropy_change = self._calculate_entropy_change(energy_change, information_change)
        
        # Generate corrections if needed
        corrected_values = None
        if not is_valid:
            corrected_values = self._generate_corrections(
                initial_energy, final_energy, initial_information, final_information
            )
        
        result = ConservationResult(
            is_valid=is_valid,
            violation_magnitude=total_violation,
            conservation_mode=mode,
            energy_balance=energy_balance,
            information_balance=info_balance,
            entropy_change=entropy_change,
            corrected_values=corrected_values
        )
        
        # Record in history
        self.conservation_history.append({
            'timestamp': np.datetime64('now'),
            'result': result,
            'energy_change': energy_change,
            'information_change': information_change
        })
        
        # Keep history bounded
        if len(self.conservation_history) > 1000:
            self.conservation_history.pop(0)
        
        return result
    
    def _calculate_energy_violation(self, energy_change: float) -> float:
        """Calculate energy conservation violation magnitude."""
        # For cognitive systems, small energy fluctuations are expected
        # due to quantum effects and thermal noise
        thermal_noise = self.k_boltzmann * self.temperature
        expected_fluctuation = thermal_noise * 10  # Scale for cognitive systems
        
        violation = abs(energy_change) - expected_fluctuation
        return max(violation, 0.0) / expected_fluctuation
    
    def _calculate_information_violation(self, info_change: float) -> float:
        """Calculate information conservation violation magnitude."""
        # Information can increase (learning) but shouldn't decrease without
        # corresponding entropy increase (forgetting/compression)
        if info_change >= 0:
            return 0.0  # Information increase is allowed
        else:
            # Information decrease should be bounded by compression limits
            max_compression = 0.8  # Maximum 80% compression allowed
            violation = abs(info_change) - max_compression
            return max(violation, 0.0) / max_compression
    
    def _calculate_full_thermodynamic_violation(self, 
                                              energy_change: float, 
                                              info_change: float) -> float:
        """Calculate full thermodynamic violation including entropy constraints."""
        # Landauer's principle: erasing information costs energy
        min_energy_cost = abs(min(info_change, 0)) * self.k_boltzmann * self.temperature * math.log(2)
        
        if energy_change < min_energy_cost:
            return (min_energy_cost - energy_change) / min_energy_cost
        
        return 0.0
    
    def _calculate_entropy_change(self, energy_change: float, info_change: float) -> float:
        """Calculate entropy change from energy and information changes."""
        # ΔS = ΔE/T + k_B * ln(2) * ΔI_bits
        entropy_from_energy = energy_change / self.temperature
        entropy_from_info = self.k_boltzmann * math.log(2) * abs(info_change)
        
        return entropy_from_energy + entropy_from_info
    
    def _generate_corrections(self, 
                            initial_energy: float,
                            final_energy: float, 
                            initial_info: float,
                            final_info: float) -> Dict[str, float]:
        """Generate corrected values that satisfy conservation laws."""
        # Apply minimum energy correction for information processing
        info_change = final_info - initial_info
        min_energy_change = abs(min(info_change, 0)) * self.k_boltzmann * self.temperature * math.log(2)
        
        corrected_final_energy = initial_energy + min_energy_change
        
        # Apply maximum information compression
        if info_change < 0:
            max_compression = 0.8
            corrected_final_info = initial_info - max_compression
        else:
            corrected_final_info = final_info
        
        return {
            'corrected_final_energy': corrected_final_energy,
            'corrected_final_information': corrected_final_info,
            'energy_correction': corrected_final_energy - final_energy,
            'information_correction': corrected_final_info - final_info
        }
    
    def validate_state_transition(self, 
                                pre_state: Dict[str, float], 
                                post_state: Dict[str, float], 
                                operation_type: str) -> bool:
        """
        Validate a cognitive state transition for conservation compliance.
        
        Args:
            pre_state: State before transition (must contain 'energy' and optionally 'information')
            post_state: State after transition 
            operation_type: Type of cognitive operation
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Extract energy and information from states
        pre_energy = pre_state.get('energy', 0.0)
        post_energy = post_state.get('energy', 0.0)
        pre_info = pre_state.get('information', pre_energy)  # Default to energy if no info
        post_info = post_state.get('information', post_energy)
        
        # Use existing validation method
        result = self.validate_conservation(
            initial_energy=pre_energy,
            final_energy=post_energy,
            initial_information=pre_info,
            final_information=post_info,
            mode=self.mode
        )
        
        # Log transition for tracking
        self.conservation_history.append({
            'operation': operation_type,
            'pre_state': pre_state.copy(),
            'post_state': post_state.copy(),
            'result': result,
            'timestamp': time.time() if 'time' in globals() else 0.0
        })
        
        return result.is_valid
    
    def detect_violations(self, 
                         pre_state: Dict[str, float], 
                         post_state: Dict[str, float]) -> Dict[str, float]:
        """
        Detect and quantify conservation violations in state transition.
        
        Args:
            pre_state: State before transition
            post_state: State after transition
            
        Returns:
            Dictionary mapping violation types to magnitudes
        """
        violations = {}
        
        # Energy conservation violation
        energy_change = post_state.get('energy', 0.0) - pre_state.get('energy', 0.0)
        energy_violation = self._calculate_energy_violation(energy_change)
        if energy_violation > self.tolerance:
            violations['energy'] = energy_violation
        
        # Information conservation violation
        info_change = post_state.get('information', 0.0) - pre_state.get('information', 0.0)
        info_violation = self._calculate_information_violation(info_change)
        if info_violation > self.tolerance:
            violations['information'] = info_violation
        
        # Entropy violation (if present in states)
        if 'entropy' in pre_state and 'entropy' in post_state:
            entropy_change = post_state['entropy'] - pre_state['entropy']
            if entropy_change < 0:  # Entropy shouldn't decrease
                violations['entropy'] = abs(entropy_change)
        
        return violations
    
    def get_conservation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive conservation statistics."""
        violation_rate = self.violation_count / max(self.total_validations, 1)
        
        recent_history = self.conservation_history[-100:]  # Last 100 validations
        avg_energy_balance = np.mean([h['result'].energy_balance for h in recent_history]) if recent_history else 1.0
        avg_info_balance = np.mean([h['result'].information_balance for h in recent_history]) if recent_history else 1.0
        
        return {
            'total_validations': self.total_validations,
            'violation_count': self.violation_count,
            'violation_rate': violation_rate,
            'average_energy_balance': avg_energy_balance,
            'average_information_balance': avg_info_balance,
            'conservation_integrity': 1.0 - violation_rate,
            'tolerance': self.tolerance,
            'cognitive_temperature': self.temperature
        }
    
    def set_temperature(self, temperature: float):
        """Set cognitive temperature for thermodynamic calculations."""
        self.temperature = max(temperature, 1.0)  # Minimum 1K
    
    def reset_conservation_state(self):
        """Reset conservation engine state."""
        self.conservation_history.clear()
        self.violation_count = 0
        self.total_validations = 0
        print("Conservation engine state reset")
    
    def reset_statistics(self):
        """Reset conservation statistics."""
        self.conservation_history.clear()
        self.violation_count = 0
        self.total_validations = 0