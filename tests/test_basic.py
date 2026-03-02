#!/usr/bin/env python3
"""
Basic tests for memsched
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memsched import (
    Request, RequestStatus,
    get_model_config, SystemConfig, SystemState,
    get_scheduler, SCHEDULERS,
    Simulator,
    WorkloadConfig, WorkloadGenerator,
)


def test_request():
    """Test Request class"""
    print("Testing Request...")
    
    req = Request(
        id=0,
        arrival_time=0.0,
        prompt_tokens=100,
        max_output_tokens=200,
        actual_output_tokens=50,
    )
    
    assert req.total_tokens == 100  # prompt only initially
    assert req.remaining_tokens == 50
    assert req.progress == 0.0
    assert not req.is_completed
    
    req.generated_tokens = 25
    assert req.progress == 0.5
    assert req.total_tokens == 125
    
    req.generated_tokens = 50
    assert req.is_completed
    
    print("  PASSED")


def test_system_state():
    """Test SystemState class"""
    print("Testing SystemState...")
    
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig()
    state = SystemState(model_config, system_config)
    
    assert state.available_kv_memory_mb > 0
    assert state.memory_utilization == 0.0
    
    req = Request(
        id=0,
        arrival_time=0.0,
        prompt_tokens=100,
        max_output_tokens=200,
    )
    
    state.add_request(req)
    assert len(state.waiting_queue) == 1
    assert state.total_arrived == 1
    
    state.start_request(req)
    assert len(state.waiting_queue) == 0
    assert len(state.running_requests) == 1
    assert req.status == RequestStatus.RUNNING
    
    state.complete_request(req)
    assert len(state.running_requests) == 0
    assert len(state.completed_requests) == 1
    assert req.status == RequestStatus.COMPLETED
    
    print("  PASSED")


def test_schedulers():
    """Test all schedulers"""
    print("Testing Schedulers...")
    
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig()
    
    for name in SCHEDULERS:
        scheduler = get_scheduler(name)
        state = SystemState(model_config, system_config)
        
        # Add some requests
        for i in range(5):
            req = Request(
                id=i,
                arrival_time=i * 100.0,
                prompt_tokens=100 + i * 20,
                max_output_tokens=200,
            )
            state.add_request(req)
            scheduler.on_request_arrival(state, req)
        
        # Select next
        selected = scheduler.select_next(state)
        assert selected is not None, f"{name} failed to select"
        
        print(f"  {name}: PASSED")


def test_workload_generator():
    """Test WorkloadGenerator"""
    print("Testing WorkloadGenerator...")
    
    config = WorkloadConfig(
        num_requests=50,
        arrival_rate=10.0,
        seed=42,
    )
    
    generator = WorkloadGenerator(config)
    requests = generator.generate()
    
    assert len(requests) == 50
    assert all(r.prompt_tokens > 0 for r in requests)
    assert all(r.actual_output_tokens > 0 for r in requests)
    
    # Check arrival times are increasing
    for i in range(1, len(requests)):
        assert requests[i].arrival_time >= requests[i-1].arrival_time
    
    print("  PASSED")


def test_simulator():
    """Test Simulator"""
    print("Testing Simulator...")
    
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig(max_batch_size=8)
    scheduler = get_scheduler("fcfs")
    
    simulator = Simulator(model_config, system_config, scheduler)
    
    # Generate small workload
    config = WorkloadConfig(num_requests=10, seed=42)
    generator = WorkloadGenerator(config)
    requests = generator.generate()
    
    # Run simulation
    result = simulator.run(requests, show_progress=False)
    
    assert len(result.completed_requests) == 10
    assert result.avg_latency > 0
    assert result.throughput > 0
    
    print("  PASSED")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Running MemSched Tests")
    print("=" * 50)
    print()
    
    test_request()
    test_system_state()
    test_schedulers()
    test_workload_generator()
    test_simulator()
    
    print()
    print("=" * 50)
    print("All tests PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
