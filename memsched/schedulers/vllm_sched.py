"""
vLLM-style Scheduler
"""

from typing import Optional, List
from .base import BaseScheduler
from ..core.request import Request
from ..core.state import SystemState


class VLLMScheduler(BaseScheduler):
    """
    vLLM-style scheduler with continuous batching
    
    Features:
    - FCFS for fairness
    - Memory-aware admission
    - Iteration-level scheduling
    """
    
    name = "vllm"
    
    def __init__(
        self,
        max_num_seqs: int = 256,
        max_tokens_per_batch: int = 8192,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_num_seqs = max_num_seqs
        self.max_tokens_per_batch = max_tokens_per_batch
    
    def select_next(self, state: SystemState) -> Optional[Request]:
        if not state.waiting_queue:
            return None
        
        # Check sequence limit
        if len(state.running_requests) >= self.max_num_seqs:
            return None
        
        # Check batch token limit
        current_batch_tokens = sum(
            req.total_tokens for req in state.running_requests
        )
        
        # FCFS order
        for req in state.waiting_queue:
            # Check memory
            if not state.can_admit(req):
                continue
            
            # Check batch token limit
            if current_batch_tokens + req.prompt_tokens > self.max_tokens_per_batch:
                continue
            
            return req
        
        return None
    
    def should_preempt(
        self, state: SystemState, new_request: Request
    ) -> Optional[Request]:
        """vLLM uses swapping, simplified here as preemption"""
        if not state.running_requests:
            return None
        
        # Only preempt under severe memory pressure
        if state.memory_utilization < 0.95:
            return None
        
        # Preempt the request with most remaining work
        candidates = [
            (req.remaining_tokens, req) 
            for req in state.running_requests
        ]
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
