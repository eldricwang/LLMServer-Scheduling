"""
System state management
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .request import Request, RequestStatus
from .config import ModelConfig, SystemConfig


@dataclass
class SystemState:
    """System state for simulation"""
    
    model_config: ModelConfig
    system_config: SystemConfig
    
    # Request queues
    waiting_queue: List[Request] = field(default_factory=list)
    running_requests: List[Request] = field(default_factory=list)
    completed_requests: List[Request] = field(default_factory=list)
    
    # Current time
    current_time: float = 0.0
    
    # Statistics
    total_arrived: int = 0
    total_completed: int = 0
    total_preemptions: int = 0
    
    @property
    def kv_bytes_per_token(self) -> float:
        return self.model_config.kv_bytes_per_token
    
    @property
    def used_kv_memory_mb(self) -> float:
        total = 0.0
        for req in self.running_requests:
            total += req.kv_cache_mb(self.kv_bytes_per_token)
        return total
    
    @property
    def available_kv_memory_mb(self) -> float:
        return self.system_config.available_kv_memory_mb - self.used_kv_memory_mb
    
    @property
    def memory_utilization(self) -> float:
        total = self.system_config.available_kv_memory_mb
        if total <= 0:
            return 0.0
        return self.used_kv_memory_mb / total
    
    def can_admit(self, request: Request, use_predicted: bool = True) -> bool:
        if use_predicted:
            needed = request.estimated_kv_cache_mb(self.kv_bytes_per_token)
        else:
            needed = request.kv_cache_mb(self.kv_bytes_per_token)
        return needed <= self.available_kv_memory_mb
    
    def add_request(self, request: Request):
        request.status = RequestStatus.WAITING
        self.waiting_queue.append(request)
        self.total_arrived += 1
    
    def start_request(self, request: Request):
        if request in self.waiting_queue:
            self.waiting_queue.remove(request)
        request.status = RequestStatus.RUNNING
        if request.start_time == 0:
            request.start_time = self.current_time
        self.running_requests.append(request)
    
    def complete_request(self, request: Request):
        if request in self.running_requests:
            self.running_requests.remove(request)
        request.status = RequestStatus.COMPLETED
        request.completion_time = self.current_time
        self.completed_requests.append(request)
        self.total_completed += 1
    
    def preempt_request(self, request: Request):
        if request in self.running_requests:
            self.running_requests.remove(request)
        request.status = RequestStatus.PREEMPTED
        request.preempt_count += 1
        self.waiting_queue.append(request)
        self.total_preemptions += 1
    
    def get_request_by_id(self, req_id: int) -> Optional[Request]:
        for req in self.waiting_queue + self.running_requests:
            if req.id == req_id:
                return req
        return None
