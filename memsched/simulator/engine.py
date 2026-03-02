"""
Discrete Event Simulation Engine
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Callable, Any
from enum import Enum


class EventType(Enum):
    REQUEST_ARRIVAL = "arrival"
    TOKEN_COMPLETE = "token_complete"
    REQUEST_COMPLETE = "request_complete"


@dataclass(order=True)
class Event:
    time: float
    event_type: EventType = field(compare=False)
    data: Any = field(compare=False, default=None)


class EventQueue:
    """Priority queue for events"""
    
    def __init__(self):
        self.heap: List[Event] = []
    
    def push(self, event: Event):
        heapq.heappush(self.heap, event)
    
    def pop(self) -> Event:
        return heapq.heappop(self.heap)
    
    def is_empty(self) -> bool:
        return len(self.heap) == 0
    
    def peek(self) -> Event:
        return self.heap[0] if self.heap else None
    
    def clear(self):
        self.heap.clear()
