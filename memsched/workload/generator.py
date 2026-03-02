"""
Workload Generator
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from ..core.request import Request


@dataclass
class WorkloadConfig:
    """Workload configuration"""
    num_requests: int = 100
    arrival_rate: float = 10.0  # requests per second
    
    # Prompt length distribution
    prompt_min: int = 50
    prompt_max: int = 500
    prompt_mean: int = 200
    prompt_std: int = 100
    
    # Output length distribution
    output_min: int = 20
    output_max: int = 300
    output_mean: int = 100
    output_std: int = 50
    
    # Max tokens (for request)
    max_output_tokens: int = 500
    
    # Priority distribution
    priority_weights: List[float] = None  # [low, medium, high]
    
    # Deadline (optional)
    deadline_ratio: float = 0.0  # fraction of requests with deadlines
    deadline_slack: float = 2.0  # deadline = expected_time * slack
    
    # Random seed
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.priority_weights is None:
            self.priority_weights = [0.7, 0.2, 0.1]


class WorkloadGenerator:
    """Generate synthetic workloads"""
    
    def __init__(self, config: WorkloadConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
    
    def generate(self) -> List[Request]:
        """Generate workload requests"""
        requests = []
        
        # Generate arrival times (Poisson process)
        inter_arrivals = self.rng.exponential(
            1.0 / self.config.arrival_rate,
            self.config.num_requests
        )
        arrival_times = np.cumsum(inter_arrivals) * 1000  # Convert to ms
        
        for i in range(self.config.num_requests):
            # Prompt length (truncated normal)
            prompt_tokens = int(self.rng.normal(
                self.config.prompt_mean,
                self.config.prompt_std
            ))
            prompt_tokens = np.clip(
                prompt_tokens,
                self.config.prompt_min,
                self.config.prompt_max
            )
            
            # Output length (truncated normal)
            output_tokens = int(self.rng.normal(
                self.config.output_mean,
                self.config.output_std
            ))
            output_tokens = np.clip(
                output_tokens,
                self.config.output_min,
                self.config.output_max
            )
            
            # Priority
            priority = self.rng.choice(
                [0, 1, 2],
                p=self.config.priority_weights
            )
            
            # Deadline
            deadline = None
            if self.rng.random() < self.config.deadline_ratio:
                expected_time = prompt_tokens * 0.5 + output_tokens * 10  # rough estimate
                deadline = arrival_times[i] + expected_time * self.config.deadline_slack
            
            # Predicted output (with some noise)
            prediction_error = self.rng.normal(0, 0.2)
            predicted_output = int(output_tokens * (1 + prediction_error))
            predicted_output = max(10, min(predicted_output, self.config.max_output_tokens))
            
            request = Request(
                id=i,
                arrival_time=arrival_times[i],
                prompt_tokens=prompt_tokens,
                max_output_tokens=self.config.max_output_tokens,
                priority=priority,
                deadline=deadline,
                predicted_output_tokens=predicted_output,
                actual_output_tokens=output_tokens,
            )
            requests.append(request)
        
        return requests
    
    def generate_bursty(
        self,
        burst_ratio: float = 0.3,
        burst_multiplier: float = 5.0,
    ) -> List[Request]:
        """Generate bursty workload"""
        requests = []
        current_time = 0.0
        request_id = 0
        
        while request_id < self.config.num_requests:
            # Decide if this is a burst period
            is_burst = self.rng.random() < burst_ratio
            
            if is_burst:
                # Burst: generate multiple requests quickly
                burst_size = min(
                    int(self.rng.exponential(5)) + 2,
                    self.config.num_requests - request_id
                )
                burst_interval = 1.0 / (self.config.arrival_rate * burst_multiplier)
                
                for _ in range(burst_size):
                    req = self._generate_single_request(request_id, current_time * 1000)
                    requests.append(req)
                    request_id += 1
                    current_time += self.rng.exponential(burst_interval)
            else:
                # Normal: single request
                req = self._generate_single_request(request_id, current_time * 1000)
                requests.append(req)
                request_id += 1
                current_time += self.rng.exponential(1.0 / self.config.arrival_rate)
        
        return requests
    
    def generate_bimodal(self) -> List[Request]:
        """Generate bimodal workload (short and long requests)"""
        requests = []
        
        inter_arrivals = self.rng.exponential(
            1.0 / self.config.arrival_rate,
            self.config.num_requests
        )
        arrival_times = np.cumsum(inter_arrivals) * 1000
        
        for i in range(self.config.num_requests):
            # 50% short, 50% long
            is_short = self.rng.random() < 0.5
            
            if is_short:
                prompt_tokens = self.rng.randint(50, 150)
                output_tokens = self.rng.randint(20, 80)
            else:
                prompt_tokens = self.rng.randint(300, 600)
                output_tokens = self.rng.randint(150, 400)
            
            predicted_output = int(output_tokens * (1 + self.rng.normal(0, 0.2)))
            predicted_output = max(10, min(predicted_output, self.config.max_output_tokens))
            
            request = Request(
                id=i,
                arrival_time=arrival_times[i],
                prompt_tokens=prompt_tokens,
                max_output_tokens=self.config.max_output_tokens,
                priority=self.rng.choice([0, 1, 2], p=self.config.priority_weights),
                predicted_output_tokens=predicted_output,
                actual_output_tokens=output_tokens,
            )
            requests.append(request)
        
        return requests
    
    def _generate_single_request(self, req_id: int, arrival_time: float) -> Request:
        """Generate a single request"""
        prompt_tokens = int(np.clip(
            self.rng.normal(self.config.prompt_mean, self.config.prompt_std),
            self.config.prompt_min,
            self.config.prompt_max
        ))
        
        output_tokens = int(np.clip(
            self.rng.normal(self.config.output_mean, self.config.output_std),
            self.config.output_min,
            self.config.output_max
        ))
        
        predicted_output = int(output_tokens * (1 + self.rng.normal(0, 0.2)))
        predicted_output = max(10, min(predicted_output, self.config.max_output_tokens))
        
        return Request(
            id=req_id,
            arrival_time=arrival_time,
            prompt_tokens=prompt_tokens,
            max_output_tokens=self.config.max_output_tokens,
            priority=self.rng.choice([0, 1, 2], p=self.config.priority_weights),
            predicted_output_tokens=predicted_output,
            actual_output_tokens=output_tokens,
        )


# Preset workload configurations
WORKLOAD_PRESETS: Dict[str, WorkloadConfig] = {
    "light": WorkloadConfig(
        num_requests=50,
        arrival_rate=5.0,
        prompt_mean=150,
        output_mean=80,
    ),
    "medium": WorkloadConfig(
        num_requests=100,
        arrival_rate=10.0,
        prompt_mean=200,
        output_mean=100,
    ),
    "heavy": WorkloadConfig(
        num_requests=200,
        arrival_rate=20.0,
        prompt_mean=250,
        output_mean=150,
    ),
    "long_context": WorkloadConfig(
        num_requests=100,
        arrival_rate=8.0,
        prompt_min=200,
        prompt_max=1000,
        prompt_mean=500,
        output_mean=200,
    ),
}


def get_workload_config(name: str) -> WorkloadConfig:
    if name not in WORKLOAD_PRESETS:
        raise ValueError(f"Unknown workload: {name}. Available: {list(WORKLOAD_PRESETS.keys())}")
    return WORKLOAD_PRESETS[name]
