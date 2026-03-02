"""
Request definition for LLM serving
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    COMPLETED = "completed"


@dataclass
class Request:
    """LLM inference request"""
    
    # Basic info
    id: int
    arrival_time: float
    prompt_tokens: int
    max_output_tokens: int
    
    # Metadata
    priority: int = 0
    deadline: Optional[float] = None
    
    # Prediction
    predicted_output_tokens: int = 0
    
    # Runtime state
    status: RequestStatus = RequestStatus.WAITING
    start_time: float = 0.0
    first_token_time: float = 0.0
    completion_time: float = 0.0
    generated_tokens: int = 0
    preempt_count: int = 0
    
    # Ground truth (known in simulation)
    actual_output_tokens: int = 0
    
    def __post_init__(self):
        if self.predicted_output_tokens == 0:
            self.predicted_output_tokens = self.max_output_tokens // 2
        if self.actual_output_tokens == 0:
            self.actual_output_tokens = self.predicted_output_tokens
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens
    
    @property
    def estimated_total_tokens(self) -> int:
        return self.prompt_tokens + self.predicted_output_tokens
    
    @property
    def remaining_tokens(self) -> int:
        return max(0, self.actual_output_tokens - self.generated_tokens)
    
    @property
    def progress(self) -> float:
        if self.actual_output_tokens == 0:
            return 1.0
        return self.generated_tokens / self.actual_output_tokens
    
    @property
    def is_completed(self) -> bool:
        return self.generated_tokens >= self.actual_output_tokens
    
    @property
    def latency(self) -> float:
        if self.completion_time > 0:
            return self.completion_time - self.arrival_time
        return 0.0
    
    @property
    def waiting_time(self) -> float:
        if self.start_time > 0:
            return self.start_time - self.arrival_time
        return 0.0
    
    def kv_cache_mb(self, bytes_per_token: float) -> float:
        return self.total_tokens * bytes_per_token / 1024 / 1024
    
    def estimated_kv_cache_mb(self, bytes_per_token: float) -> float:
        return self.estimated_total_tokens * bytes_per_token / 1024 / 1024
    
    def __repr__(self):
        return (f"Request(id={self.id}, status={self.status.value}, "
                f"tokens={self.generated_tokens}/{self.actual_output_tokens})")
