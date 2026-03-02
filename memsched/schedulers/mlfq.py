"""
Multi-Level Feedback Queue Scheduler
"""

from typing import Optional, List, Dict
from collections import deque
from .base import BaseScheduler
from ..core.request import Request, RequestStatus
from ..core.state import SystemState


class MLFQScheduler(BaseScheduler):
    """
    Multi-Level Feedback Queue Scheduler
    
    - Multiple priority queues
    - Requests start at highest priority
    - Demote after using time quantum
    - Periodic boost to prevent starvation
    """
    
    name = "mlfq"
    
    def __init__(
        self,
        num_queues: int = 4,
        base_quantum_tokens: int = 50,  # Tokens before demotion
        boost_interval_ms: float = 5000.0,  # Boost every 5 seconds
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_queues = num_queues
        self.base_quantum_tokens = base_quantum_tokens
        self.boost_interval_ms = boost_interval_ms
        
        # Queue for each level (0 = highest priority)
        self.queues: List[deque] = [deque() for _ in range(num_queues)]
        
        # Track request queue level
        self.request_levels: Dict[int, int] = {}
        
        # Track tokens generated at current level
        self.tokens_at_level: Dict[int, int] = {}
        
        # Last boost time
        self.last_boost_time: float = 0.0
    
    def reset(self):
        self.queues = [deque() for _ in range(self.num_queues)]
        self.request_levels.clear()
        self.tokens_at_level.clear()
        self.last_boost_time = 0.0
    
    def on_request_arrival(self, state: SystemState, request: Request):
        """New requests start at highest priority queue"""
        # Higher priority requests start at higher queue
        initial_level = max(0, 1 - request.priority)  # Priority 2 -> level 0
        
        self.request_levels[request.id] = initial_level
        self.tokens_at_level[request.id] = 0
        self.queues[initial_level].append(request)
    
    def select_next(self, state: SystemState) -> Optional[Request]:
        # Periodic boost
        self._maybe_boost(state)
        
        # Rebuild queues from waiting_queue (sync with state)
        self._sync_queues(state)
        
        # Select from highest priority non-empty queue
        for level in range(self.num_queues):
            while self.queues[level]:
                req = self.queues[level][0]
                
                # Check if request is still waiting
                if req.status != RequestStatus.WAITING:
                    self.queues[level].popleft()
                    continue
                
                # Check if can admit
                if state.can_admit(req):
                    self.queues[level].popleft()
                    return req
                else:
                    # Can't admit, try next in queue
                    break
        
        return None
    
    def on_token_generated(self, state: SystemState, request: Request):
        """Track tokens and demote if needed"""
        if request.id not in self.tokens_at_level:
            return
        
        self.tokens_at_level[request.id] += 1
        current_level = self.request_levels.get(request.id, 0)
        
        # Time quantum for this level (higher levels have smaller quantum)
        quantum = self.base_quantum_tokens * (2 ** current_level)
        
        # Demote if exceeded quantum
        if self.tokens_at_level[request.id] >= quantum:
            new_level = min(current_level + 1, self.num_queues - 1)
            if new_level != current_level:
                self.request_levels[request.id] = new_level
                self.tokens_at_level[request.id] = 0
    
    def on_request_complete(self, state: SystemState, request: Request):
        """Clean up tracking"""
        self.request_levels.pop(request.id, None)
        self.tokens_at_level.pop(request.id, None)
    
    def _maybe_boost(self, state: SystemState):
        """Periodically boost all requests to prevent starvation"""
        if state.current_time - self.last_boost_time >= self.boost_interval_ms:
            self.last_boost_time = state.current_time
            
            # Move all requests to highest priority queue
            all_requests = []
            for level in range(1, self.num_queues):
                while self.queues[level]:
                    req = self.queues[level].popleft()
                    all_requests.append(req)
            
            for req in all_requests:
                self.request_levels[req.id] = 0
                self.tokens_at_level[req.id] = 0
                self.queues[0].append(req)
    
    def _sync_queues(self, state: SystemState):
        """Sync internal queues with state's waiting queue"""
        # Get all waiting request IDs
        waiting_ids = {req.id for req in state.waiting_queue}
        
        # Remove completed/running requests from queues
        for level in range(self.num_queues):
            self.queues[level] = deque(
                req for req in self.queues[level] 
                if req.id in waiting_ids
            )
        
        # Add any new requests not in queues
        queued_ids = set()
        for level in range(self.num_queues):
            for req in self.queues[level]:
                queued_ids.add(req.id)
        
        for req in state.waiting_queue:
            if req.id not in queued_ids:
                level = self.request_levels.get(req.id, 0)
                self.queues[level].append(req)
                if req.id not in self.request_levels:
                    self.request_levels[req.id] = level
                    self.tokens_at_level[req.id] = 0
