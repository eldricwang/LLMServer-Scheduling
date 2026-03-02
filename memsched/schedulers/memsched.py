"""
MemSched: KV Cache Aware Scheduler (Our Method)
Optimized for best overall performance
"""

from typing import Optional, Dict, List, Tuple
from .base import BaseScheduler
from ..core.request import Request
from ..core.state import SystemState


class MemSchedScheduler(BaseScheduler):
    """
    MemSched: Memory-Aware Scheduler for LLM Serving
    
    Key innovations:
    1. Adaptive memory-aware admission
    2. Hybrid SJF + Fairness scoring
    3. Memory pressure-based strategy switching
    4. Efficient batch packing
    """
    
    name = "memsched"
    
    def __init__(
        self,
        memory_high_threshold: float = 0.85,
        memory_critical_threshold: float = 0.95,
        starvation_threshold_ms: float = 10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.memory_high_threshold = memory_high_threshold
        self.memory_critical_threshold = memory_critical_threshold
        self.starvation_threshold_ms = starvation_threshold_ms
    
    def reset(self):
        pass
    
    def select_next(self, state: SystemState) -> Optional[Request]:
        if not state.waiting_queue:
            return None
        
        # Get memory pressure level
        memory_util = state.memory_utilization
        available_mb = state.available_kv_memory_mb
        
        if available_mb <= 0:
            return None
        
        # Find admissible candidates
        candidates = []
        for req in state.waiting_queue:
            needed_mb = req.estimated_kv_cache_mb(state.kv_bytes_per_token)
            if needed_mb <= available_mb:
                candidates.append(req)
        
        if not candidates:
            return None
        
        # Check for starving requests first (fairness)
        starving = [
            req for req in candidates
            if (state.current_time - req.arrival_time) > self.starvation_threshold_ms
        ]
        
        if starving:
            # Prioritize oldest starving request that fits well
            starving.sort(key=lambda r: (
                -r.priority,  # Higher priority first
                r.arrival_time,  # Then oldest
                r.estimated_total_tokens  # Then shortest
            ))
            return starving[0]
        
        # Strategy based on memory pressure
        if memory_util > self.memory_critical_threshold:
            # Critical: only admit smallest requests
            return self._select_smallest(candidates, state)
        elif memory_util > self.memory_high_threshold:
            # High pressure: prefer short jobs with memory efficiency
            return self._select_memory_efficient(candidates, state)
        else:
            # Normal: balanced SJF with slight FCFS bias
            return self._select_balanced(candidates, state)
    
    def _select_smallest(
        self, candidates: List[Request], state: SystemState
    ) -> Request:
        """Select smallest request (memory-wise)"""
        return min(
            candidates,
            key=lambda r: r.estimated_kv_cache_mb(state.kv_bytes_per_token)
        )
    
    def _select_memory_efficient(
        self, candidates: List[Request], state: SystemState
    ) -> Request:
        """Select based on memory efficiency and job size"""
        def score(req):
            # Prefer shorter jobs
            size_score = req.estimated_total_tokens
            # Slight preference for older requests
            wait_penalty = (state.current_time - req.arrival_time) / 1000.0
            # Priority boost
            priority_bonus = req.priority * 100
            
            return size_score - wait_penalty * 10 - priority_bonus
        
        return min(candidates, key=score)
    
    def _select_balanced(
        self, candidates: List[Request], state: SystemState
    ) -> Request:
        """Balanced selection: SJF with fairness"""
        def score(req):
            # Base: estimated total tokens (SJF)
            size_score = req.estimated_total_tokens
            
            # Waiting time factor (fairness)
            wait_time = state.current_time - req.arrival_time
            wait_factor = wait_time / 500.0  # Every 500ms reduces score
            
            # Priority
            priority_bonus = req.priority * 50
            
            # Combined score (lower is better)
            return size_score - wait_factor - priority_bonus
        
        return min(candidates, key=score)
    
    def should_preempt(
        self, state: SystemState, new_request: Request
    ) -> Optional[Request]:
        """Preemption for high-priority requests"""
        if not state.running_requests:
            return None
        
        # Only preempt for highest priority
        if new_request.priority < 2:
            return None
        
        # Check if we actually need memory
        needed = new_request.estimated_kv_cache_mb(state.kv_bytes_per_token)
        if needed <= state.available_kv_memory_mb:
            return None
        
        # Find victim with lowest priority and least progress
        victims = [
            req for req in state.running_requests
            if req.priority < new_request.priority
        ]
        
        if not victims:
            return None
        
        # Select victim: low priority, low progress, high memory
        return min(victims, key=lambda r: (
            r.priority,
            r.progress,
            -r.kv_cache_mb(state.kv_bytes_per_token)
        ))
