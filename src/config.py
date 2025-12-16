"""
Configuration management for 3D N² Queens MCMC Solver.

This module provides a clean interface for loading and validating
configuration from YAML files.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Any
from pathlib import Path


@dataclass
class Config:
    """
    Configuration container for MCMC solver.
    
    Attributes:
        sizes: List of board sizes to run
        steps: Number of MCMC steps
        seed: Random seed
        method: MCMC method ('basic' or 'improved')
        cooling: Cooling schedule ('linear', 'geometric', 'adaptive')
        beta_min: Initial/minimum beta
        beta_max: Final/maximum beta
        simulated_annealing: Whether to use simulated annealing
        energy_treatment: Energy function type
        state_space: State space type ('full' or 'reduced')
        energy_reground_interval: Steps between energy recalculations
        mode: Execution mode ('single' or 'multiple')
        num_runs: Number of runs for multiple mode
        show: Whether to show plots
        save: Whether to save results
        output_dir: Directory to save results
    """
    
    # Board configuration
    sizes: List[int] = field(default_factory=lambda: [5])
    
    # Solver configuration
    steps: int = 1000000
    seed: int = 42
    method: str = 'basic'
    cooling: str = 'geometric'
    beta_min: float = 0.1
    beta_max: float = 25.0
    simulated_annealing: bool = True
    energy_treatment: str = 'linear'
    complexity: str = 'hash'
    state_space: str = 'full'
    energy_reground_interval: int = 0
    log_interval: int = 0
    
    # Execution configuration
    mode: str = 'single'
    num_runs: int = 1
    
    # Visualization and output
    show: bool = False
    save: bool = False
    output_dir: str = 'results'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Config instance
        """
        # Handle 'sizes' vs 'size' for backward compatibility
        sizes = data.get('sizes')
        if sizes is None:
            size = data.get('size')
            sizes = [size] if size is not None else [5]
        if not isinstance(sizes, list):
            sizes = [sizes]
        
        return cls(
            sizes=sizes,
            steps=data.get('steps', 1000000),
            seed=data.get('seed', 42),
            method=data.get('method', 'basic'),
            cooling=data.get('cooling', 'geometric'),
            beta_min=data.get('beta_min', 0.1),
            beta_max=data.get('beta_max', 25.0),
            simulated_annealing=data.get('simulated_annealing', True),
            energy_treatment=data.get('energy_treatment', 'linear'),
            complexity=data.get('complexity', 'hash'),
            state_space=data.get('state_space', 'full'),
            energy_reground_interval=data.get('energy_reground_interval', 0),
            log_interval=data.get('log_interval', 0),
            mode=data.get('mode', 'single'),
            num_runs=data.get('num_runs', 1),
            show=data.get('show', False),
            save=data.get('save', False),
            output_dir=data.get('output_dir', 'results'),
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'sizes': self.sizes,
            'steps': self.steps,
            'seed': self.seed,
            'method': self.method,
            'cooling': self.cooling,
            'beta_min': self.beta_min,
            'beta_max': self.beta_max,
            'simulated_annealing': self.simulated_annealing,
            'energy_treatment': self.energy_treatment,
            'complexity': self.complexity,
            'state_space': self.state_space,
            'energy_reground_interval': self.energy_reground_interval,
            'log_interval': self.log_interval,
            'mode': self.mode,
            'num_runs': self.num_runs,
            'show': self.show,
            'save': self.save,
            'output_dir': self.output_dir,
        }
    
    def validate(self) -> List[str]:
        """
        Validate configuration values.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate sizes
        if not self.sizes:
            errors.append("At least one board size must be specified")
        for size in self.sizes:
            if int(size) < 1:
                errors.append(f"Board size must be positive, got {size}")
        
        # Validate steps
        if self.steps < 1:
            errors.append(f"Steps must be positive, got {self.steps}")
        
        # Validate cooling
        valid_cooling = ['linear', 'geometric', 'adaptive']
        if self.cooling not in valid_cooling:
            errors.append(f"Invalid cooling '{self.cooling}', must be one of {valid_cooling}")
        
        # Validate beta
        if self.beta_min <= 0:
            errors.append(f"beta_min must be positive, got {self.beta_min}")
        if self.beta_max <= 0:
            errors.append(f"beta_max must be positive, got {self.beta_max}")
        
        # Validate energy treatment
        valid_treatments = ['linear', 'quadratic', 'log', 'log_quadratic']
        if self.energy_treatment not in valid_treatments:
            errors.append(f"Invalid energy_treatment '{self.energy_treatment}', must be one of {valid_treatments}")
        
        # Validate complexity
        valid_complexity = ['hash', 'iter', 'endangered', 'colored', 'weighted', 'colored_endangered', 'weighted_endangered']
        if self.complexity not in valid_complexity:
            errors.append(f"Invalid complexity '{self.complexity}', must be one of {valid_complexity}")
        
        # Validate state space
        valid_spaces = ['full', 'reduced']
        if self.state_space not in valid_spaces:
            errors.append(f"Invalid state_space '{self.state_space}', must be one of {valid_spaces}")
        
        # Validate mode
        valid_modes = ['single', 'multiple']
        if self.mode not in valid_modes:
            errors.append(f"Invalid mode '{self.mode}', must be one of {valid_modes}")
        
        # Validate num_runs
        if self.num_runs < 1:
            errors.append(f"num_runs must be positive, got {self.num_runs}")
        
        return errors
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        print(f"Board sizes: {self.sizes}")
        print(f"Steps: {self.steps:,}")
        print(f"Seed: {self.seed}")
        print(f"State space: {self.state_space}")
        print(f"Cooling: {self.cooling}")
        print(f"Beta: {self.beta_min} → {self.beta_max}" if self.simulated_annealing else f"Beta: {self.beta_min}")
        print(f"Energy treatment: {self.energy_treatment}")
        print(f"Complexity: {self.complexity}")
        if self.log_interval > 0:
            print(f"Log interval: {self.log_interval:,}")
        print(f"Mode: {self.mode}" + (f" ({self.num_runs} runs)" if self.mode == 'multiple' else ""))
        print(f"Save: {self.save}" + (f" → {self.output_dir}" if self.save else ""))
        print("=" * 60)
