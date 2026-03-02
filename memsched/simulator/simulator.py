"""
LLM Inference Simulator
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from tqdm import tqdm

from .engine import Event, EventType, EventQueue
from ..core.request import Request, RequestStatus
from ..core.config import ModelConfig, SystemConfig
from ..core.state import SystemState
from ..schedulers.base import BaseScheduler


@dataclass
class SimulationResult:
    """Simulation results"""
    completed_requests: List[Request]
    total_time_ms: float  # in milliseconds
    
    # Computed metrics
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p99_latency: float = 0.0
    avg_waiting_time: float = 0.0
    throughput: float = 0.0  # requests per second
    token_throughput: float = 0.0  # tokens per second
    preemption_count: int = 0
    avg_memory_utilization: float = 0.0
    
    def compute_metrics(self):
        if not self.completed_requests:
            return
        
        latencies = [r.latency for r in self.completed_requests]
        waiting_times = [r.waiting_time for r in self.completed_requests]
        total_tokens = sum(r.actual_output_tokens for r in self.completed_requests)
        
        latencies.sort()
        n = len(latencies)
        
        self.avg_latency = sum(latencies) / n
        self.p50_latency = latencies[n // 2]
        self.p99_latency = latencies[int(n * 0.99)] if n > 1 else latencies[-1]
        self.avg_waiting_time = sum(waiting_times) / n
        
        # Convert ms to seconds for throughput
        total_time_sec = self.total_time_ms / 1000.0
        if total_time_sec > 0:
            self.throughput = n / total_time_sec
            self.token_throughput = total_tokens / total_time_sec
        
        self.preemption_count = sum(r.preempt_count for r in self.completed_requests)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_requests": len(self.completed_requests),
            "total_time_ms": self.total_time_ms,
            "avg_latency": self.avg_latency,
            "p50_latency": self.p50_latency,
            "p99_latency": self.p99_latency,
            "avg_waiting_time": self.avg_waiting_time,
            "throughput": self.throughput,
            "token_throughput": self.token_throughput,
            "preemption_count": self.preemption_count,
            "avg_memory_utilization": self.avg_memory_utilization,
        }
    
    def __repr__(self):
        return (
            f"SimulationResult(\n"
            f"  requests={len(self.completed_requests)},\n"
            f"  avg_latency={self.avg_latency:.2f}ms,\n"
            f"  p99_latency={self.p99_latency:.2f}ms,\n"
            f"  throughput={self.throughput:.2f} req/s,\n"
            f"  token_throughput={self.token_throughput:.2f} tok/s\n"
            f")"
        )


class Simulator:
    """LLM Inference Simulator"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        system_config: SystemConfig,
        scheduler: BaseScheduler,
    ):
        self.model_config = model_config
        self.system_config = system_config
        self.scheduler = scheduler
        
        self.state: Optional[SystemState] = None
        self.event_queue = EventQueue()
        
        # Statistics tracking
        self.memory_samples: List[float] = []
    
    def reset(self):
        """Reset simulator state"""
        self.state = SystemState(
            model_config=self.model_config,
            system_config=self.system_config,
        )
        self.event_queue.clear()
        self.scheduler.reset()
        self.memory_samples.clear()
    
    def run(
        self,
        requests: List[Request],
        show_progress: bool = True,
        max_time: float = float('inf'),
    ) -> SimulationResult:
        """
        Run simulation with given requests
        """
        self.reset()
        
        # Schedule all arrivals
        for req in requests:
            self.event_queue.push(Event(
                time=req.arrival_time,
                event_type=EventType.REQUEST_ARRIVAL,
                data=req,
            ))
        
        # Progress bar
        pbar = None
        if show_progress:
            pbar = tqdm(total=len(requests), desc="Simulating")
        
        completed_before = 0
        
        # Main simulation loop
        while not self.event_queue.is_empty():
            event = self.event_queue.pop()
            
            if event.time > max_time:
                break
            
            self.state.current_time = event.time
            
            # Handle event
            if event.event_type == EventType.REQUEST_ARRIVAL:
                self._handle_arrival(event.data)
            elif event.event_type == EventType.TOKEN_COMPLETE:
                self._handle_token_complete(event.data)
            elif event.event_type == EventType.REQUEST_COMPLETE:
                self._handle_request_complete(event.data)
                if pbar:
                    new_completed = len(self.state.completed_requests)
                    pbar.update(new_completed - completed_before)
                    completed_before = new_completed
            
            # Try to schedule more requests
            self._try_schedule()
            
            # Sample memory utilization periodically
            if len(self.state.running_requests) > 0:
                self.memory_samples.append(self.state.memory_utilization)
        
        if pbar:
            pbar.close()
        
        # Compute results
        result = SimulationResult(
            completed_requests=list(self.state.completed_requests),
            total_time_ms=self.state.current_time,
        )
        result.compute_metrics()
        
        if self.memory_samples:
            result.avg_memory_utilization = sum(self.memory_samples) / len(self.memory_samples)
        
        return result
    
    def _handle_arrival(self, request: Request):
        """Handle request arrival event"""
        self.state.add_request(request)
        self.scheduler.on_request_arrival(self.state, request)
    
    def _handle_token_complete(self, request: Request):
        """Handle token generation complete event"""
        if request.status != RequestStatus.RUNNING:
            return
        
        request.generated_tokens += 1
        
        # Record first token time
        if request.generated_tokens == 1:
            request.first_token_time = self.state.current_time
        
        self.scheduler.on_token_generated(self.state, request)
        
        # Check if request is complete
        if request.is_completed:
            self.event_queue.push(Event(
                time=self.state.current_time,
                event_type=EventType.REQUEST_COMPLETE,
                data=request,
            ))
        else:
            # Schedule next token
            self._schedule_next_token(request)
    
    def _handle_request_complete(self, request: Request):
        """Handle request completion event"""
        if request.status != RequestStatus.RUNNING:
            return
        
        self.state.complete_request(request)
        self.scheduler.on_request_complete(self.state, request)
    
    def _try_schedule(self):
        """Try to schedule waiting requests"""
        max_iterations = 100  # Prevent infinite loop
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Check batch size limit
            if len(self.state.running_requests) >= self.system_config.max_batch_size:
                break
            
            # Check if there are waiting requests
            if not self.state.waiting_queue:
                break
            
            # Select next request
            next_req = self.scheduler.select_next(self.state)
            if next_req is None:
                break
            
            # Start the request
            self._start_request(next_req)
    
    def _start_request(self, request: Request):
        """Start executing a request"""
        self.state.start_request(request)
        
        # Schedule prefill completion (first token)
        prefill_time = request.prompt_tokens * self.system_config.prefill_time_per_token_ms
        
        self.event_queue.push(Event(
            time=self.state.current_time + prefill_time,
            event_type=EventType.TOKEN_COMPLETE,
            data=request,
        ))
    
    def _schedule_next_token(self, request: Request):
        """Schedule next token generation"""
        decode_time = self.system_config.decode_time_per_token_ms
        
        # Adjust for batch size (simplified batching model)
        batch_size = len(self.state.running_requests)
        if batch_size > 1:
            # Batching amortizes decode time
            decode_time = decode_time / batch_size * (1 + 0.1 * batch_size)
        
        self.event_queue.push(Event(
            time=self.state.current_time + decode_time,
            event_type=EventType.TOKEN_COMPLETE,
            data=request,
        ))
