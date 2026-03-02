#!/usr/bin/env python3
"""
Stress test to show scheduler differences
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memsched import (
    get_model_config, SystemConfig,
    get_scheduler, SCHEDULERS,
    Simulator,
    WorkloadConfig, WorkloadGenerator,
    MetricsCollector,
    Request,
)
import random


def create_high_pressure_workload(num_requests: int = 200, seed: int = 42) -> list:
    """Create workload with high memory pressure"""
    random.seed(seed)
    requests = []
    
    current_time = 0.0
    
    for i in range(num_requests):
        # Mix of short and long requests
        if random.random() < 0.3:
            # Short request (30%)
            prompt = random.randint(50, 150)
            output = random.randint(20, 80)
        elif random.random() < 0.7:
            # Medium request (40%)
            prompt = random.randint(200, 500)
            output = random.randint(100, 300)
        else:
            # Long request (30%)
            prompt = random.randint(600, 1200)
            output = random.randint(300, 600)
        
        # High arrival rate with bursts
        if random.random() < 0.2:
            # Burst: multiple requests at same time
            inter_arrival = 0
        else:
            inter_arrival = random.expovariate(25)  # ~25 req/s average
        
        current_time += inter_arrival * 1000  # Convert to ms
        
        # Random priority
        priority = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
        
        req = Request(
            id=i,
            arrival_time=current_time,
            prompt_tokens=prompt,
            max_output_tokens=output + 100,
            actual_output_tokens=output,
            priority=priority,
        )
        requests.append(req)
    
    return requests


def run_stress_test():
    """Run stress test with memory pressure"""
    
    print("=" * 70)
    print("MemSched STRESS TEST - High Memory Pressure Scenario")
    print("=" * 70)
    
    # Configuration - VERY LIMITED MEMORY
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig(
        gpu_memory_mb=15 * 1024,      # Only 15GB GPU (1GB for KV!)
        model_memory_mb=14 * 1024,    # 14GB for model weights
        max_batch_size=8,             # Small batch
        prefill_time_per_token_ms=0.3,
        decode_time_per_token_ms=5.0,
    )
    
    print(f"\nSystem Configuration:")
    print(f"  GPU Memory: {system_config.gpu_memory_mb / 1024:.0f} GB")
    print(f"  Model Memory: {system_config.model_memory_mb / 1024:.0f} GB")
    print(f"  Available KV Memory: {system_config.available_kv_memory_mb:.0f} MB")
    print(f"  Max Batch Size: {system_config.max_batch_size}")
    
    # Create high pressure workload
    requests = create_high_pressure_workload(num_requests=150, seed=42)
    
    # Analyze workload
    prompt_tokens = [r.prompt_tokens for r in requests]
    output_tokens = [r.actual_output_tokens for r in requests]
    
    print(f"\nWorkload Statistics:")
    print(f"  Num Requests: {len(requests)}")
    print(f"  Prompt Tokens: min={min(prompt_tokens)}, max={max(prompt_tokens)}, avg={sum(prompt_tokens)/len(prompt_tokens):.0f}")
    print(f"  Output Tokens: min={min(output_tokens)}, max={max(output_tokens)}, avg={sum(output_tokens)/len(output_tokens):.0f}")
    
    # Calculate memory pressure
    avg_tokens = sum(prompt_tokens)/len(prompt_tokens) + sum(output_tokens)/len(output_tokens)
    kv_per_request_mb = avg_tokens * model_config.kv_bytes_per_token / 1024 / 1024
    print(f"  Avg KV cache per request: {kv_per_request_mb:.2f} MB")
    print(f"  Max concurrent (by memory): {system_config.available_kv_memory_mb / kv_per_request_mb:.1f}")
    
    # Metrics collector
    collector = MetricsCollector()
    
    # Schedulers to compare
    scheduler_names = ["fcfs", "sjf", "sjf_predicted", "mlfq", "memsched"]
    
    print("\n" + "-" * 70)
    print("Running experiments...")
    print("-" * 70)
    
    for scheduler_name in scheduler_names:
        print(f"\n[{scheduler_name.upper()}]")
        
        # Deep copy requests for each scheduler
        test_requests = [
            Request(
                id=r.id,
                arrival_time=r.arrival_time,
                prompt_tokens=r.prompt_tokens,
                max_output_tokens=r.max_output_tokens,
                actual_output_tokens=r.actual_output_tokens,
                priority=r.priority,
            )
            for r in requests
        ]
        
        # Create scheduler and simulator
        scheduler = get_scheduler(scheduler_name)
        simulator = Simulator(model_config, system_config, scheduler)
        
        # Run simulation
        result = simulator.run(test_requests, show_progress=True)
        collector.add_result(scheduler_name, result)
        
        print(f"  Avg Latency: {result.avg_latency:.1f}ms")
        print(f"  P50 Latency: {result.p50_latency:.1f}ms")
        print(f"  P99 Latency: {result.p99_latency:.1f}ms")
        print(f"  Throughput: {result.throughput:.2f} req/s")
        print(f"  Memory Util: {result.avg_memory_utilization:.1%}")
        print(f"  Preemptions: {result.preemption_count}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    summaries = collector.summarize_all("stress_test")
    
    # Header
    print(f"\n{'Scheduler':<15} {'Avg Lat':<10} {'P50 Lat':<10} {'P99 Lat':<10} "
          f"{'Thruput':<10} {'MemUtil':<8}")
    print("-" * 70)
    
    for summary in summaries:
        print(f"{summary.scheduler_name:<15} "
              f"{summary.avg_latency_mean:>7.0f}ms  "
              f"{summary.p50_latency_mean:>7.0f}ms  "
              f"{summary.p99_latency_mean:>7.0f}ms  "
              f"{summary.throughput_mean:>7.2f}    "
              f"{summary.memory_util_mean:>5.1%}")
    
    # Find best
    print("\n" + "-" * 70)
    best_latency = min(summaries, key=lambda s: s.avg_latency_mean)
    best_p99 = min(summaries, key=lambda s: s.p99_latency_mean)
    best_throughput = max(summaries, key=lambda s: s.throughput_mean)
    
    print(f"Best Avg Latency:  {best_latency.scheduler_name} ({best_latency.avg_latency_mean:.0f}ms)")
    print(f"Best P99 Latency:  {best_p99.scheduler_name} ({best_p99.p99_latency_mean:.0f}ms)")
    print(f"Best Throughput:   {best_throughput.scheduler_name} ({best_throughput.throughput_mean:.2f} req/s)")
    
    # Improvement analysis
    fcfs_summary = next(s for s in summaries if s.scheduler_name == "fcfs")
    
    print(f"\nImprovement over FCFS:")
    for summary in summaries:
        if summary.scheduler_name == "fcfs":
            continue
        lat_imp = (fcfs_summary.avg_latency_mean - summary.avg_latency_mean) / fcfs_summary.avg_latency_mean * 100
        p99_imp = (fcfs_summary.p99_latency_mean - summary.p99_latency_mean) / fcfs_summary.p99_latency_mean * 100
        thr_imp = (summary.throughput_mean - fcfs_summary.throughput_mean) / fcfs_summary.throughput_mean * 100
        print(f"  {summary.scheduler_name:<15}: AvgLat {lat_imp:+.1f}%, P99 {p99_imp:+.1f}%, Thru {thr_imp:+.1f}%")
    
    return collector


def run_extreme_test():
    """Extreme memory pressure test"""
    
    print("\n" + "=" * 70)
    print("EXTREME MEMORY PRESSURE TEST")
    print("=" * 70)
    
    model_config = get_model_config("llama-7b")
    
    # Extremely limited memory - only 512MB for KV cache
    system_config = SystemConfig(
        gpu_memory_mb=int(14.5 * 1024),  # 14.5GB total
        model_memory_mb=14 * 1024,        # 14GB for model
        max_batch_size=4,                 # Very small batch
        prefill_time_per_token_ms=0.2,
        decode_time_per_token_ms=3.0,
    )
    
    print(f"\nAvailable KV Memory: {system_config.available_kv_memory_mb:.0f} MB")
    print(f"Max Batch Size: {system_config.max_batch_size}")
    
    # Create workload with long sequences
    requests = []
    random.seed(123)
    current_time = 0.0
    
    for i in range(100):
        # Mostly long sequences
        prompt = random.randint(300, 800)
        output = random.randint(200, 500)
        
        inter_arrival = random.expovariate(15) * 1000
        current_time += inter_arrival
        
        requests.append(Request(
            id=i,
            arrival_time=current_time,
            prompt_tokens=prompt,
            max_output_tokens=output + 50,
            actual_output_tokens=output,
            priority=random.randint(0, 2),
        ))
    
    avg_tokens = sum(r.prompt_tokens + r.actual_output_tokens for r in requests) / len(requests)
    kv_per_req = avg_tokens * model_config.kv_bytes_per_token / 1024 / 1024
    print(f"Avg tokens per request: {avg_tokens:.0f}")
    print(f"Avg KV per request: {kv_per_req:.1f} MB")
    print(f"Max concurrent by memory: {system_config.available_kv_memory_mb / kv_per_req:.1f}")
    
    collector = MetricsCollector()
    scheduler_names = ["fcfs", "sjf", "sjf_predicted", "mlfq", "memsched"]
    
    print("\n" + "-" * 70)
    
    for scheduler_name in scheduler_names:
        print(f"\n[{scheduler_name.upper()}]")
        
        test_requests = [
            Request(
                id=r.id,
                arrival_time=r.arrival_time,
                prompt_tokens=r.prompt_tokens,
                max_output_tokens=r.max_output_tokens,
                actual_output_tokens=r.actual_output_tokens,
                priority=r.priority,
            )
            for r in requests
        ]
        
        scheduler = get_scheduler(scheduler_name)
        simulator = Simulator(model_config, system_config, scheduler)
        result = simulator.run(test_requests, show_progress=True)
        collector.add_result(scheduler_name, result)
        
        print(f"  Latency: avg={result.avg_latency:.0f}ms, p99={result.p99_latency:.0f}ms")
        print(f"  Throughput: {result.throughput:.2f} req/s, Memory: {result.avg_memory_utilization:.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXTREME TEST RESULTS")
    print("=" * 70)
    
    summaries = collector.summarize_all("extreme")
    
    print(f"\n{'Scheduler':<15} {'Avg Lat':<10} {'P99 Lat':<10} {'Throughput':<10}")
    print("-" * 50)
    
    for s in summaries:
        print(f"{s.scheduler_name:<15} {s.avg_latency_mean:>7.0f}ms  "
              f"{s.p99_latency_mean:>7.0f}ms  {s.throughput_mean:>7.2f}")


def run_bimodal_test():
    """Test with bimodal workload (short and long requests)"""
    
    print("\n" + "=" * 70)
    print("BIMODAL WORKLOAD TEST - Mixed Short and Long Requests")
    print("=" * 70)
    
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig(
        gpu_memory_mb=16 * 1024,      # 16GB
        model_memory_mb=14 * 1024,    # 14GB for model
        max_batch_size=12,
        prefill_time_per_token_ms=0.3,
        decode_time_per_token_ms=4.0,
    )
    
    print(f"\nAvailable KV Memory: {system_config.available_kv_memory_mb:.0f} MB")
    
    # Create bimodal workload
    requests = []
    random.seed(456)
    current_time = 0.0
    
    for i in range(120):
        # Bimodal: 50% short, 50% long
        if i % 2 == 0:
            # Short request
            prompt = random.randint(30, 100)
            output = random.randint(20, 60)
        else:
            # Long request
            prompt = random.randint(500, 1000)
            output = random.randint(300, 600)
        
        inter_arrival = random.expovariate(20) * 1000
        current_time += inter_arrival
        
        requests.append(Request(
            id=i,
            arrival_time=current_time,
            prompt_tokens=prompt,
            max_output_tokens=output + 50,
            actual_output_tokens=output,
            priority=0,
        ))
    
    short_reqs = [r for r in requests if r.prompt_tokens < 200]
    long_reqs = [r for r in requests if r.prompt_tokens >= 200]
    print(f"Short requests: {len(short_reqs)}, Long requests: {len(long_reqs)}")
    
    collector = MetricsCollector()
    scheduler_names = ["fcfs", "sjf", "sjf_predicted", "mlfq", "memsched"]
    
    print("\n" + "-" * 70)
    
    for scheduler_name in scheduler_names:
        print(f"\n[{scheduler_name.upper()}]")
        
        test_requests = [
            Request(
                id=r.id,
                arrival_time=r.arrival_time,
                prompt_tokens=r.prompt_tokens,
                max_output_tokens=r.max_output_tokens,
                actual_output_tokens=r.actual_output_tokens,
                priority=r.priority,
            )
            for r in requests
        ]
        
        scheduler = get_scheduler(scheduler_name)
        simulator = Simulator(model_config, system_config, scheduler)
        result = simulator.run(test_requests, show_progress=True)
        collector.add_result(scheduler_name, result)
        
        # Analyze by request type
        short_latencies = [r.latency for r in result.completed_requests if r.prompt_tokens < 200]
        long_latencies = [r.latency for r in result.completed_requests if r.prompt_tokens >= 200]
        
        avg_short = sum(short_latencies) / len(short_latencies) if short_latencies else 0
        avg_long = sum(long_latencies) / len(long_latencies) if long_latencies else 0
        
        print(f"  Overall: avg={result.avg_latency:.0f}ms, p99={result.p99_latency:.0f}ms")
        print(f"  Short requests avg: {avg_short:.0f}ms")
        print(f"  Long requests avg: {avg_long:.0f}ms")
        print(f"  Throughput: {result.throughput:.2f} req/s")
    
    # Summary
    print("\n" + "=" * 70)
    print("BIMODAL RESULTS")
    print("=" * 70)
    
    summaries = collector.summarize_all("bimodal")
    
    print(f"\n{'Scheduler':<15} {'Avg Lat':<10} {'P99 Lat':<10} {'Throughput':<10}")
    print("-" * 50)
    
    for s in summaries:
        print(f"{s.scheduler_name:<15} {s.avg_latency_mean:>7.0f}ms  "
              f"{s.p99_latency_mean:>7.0f}ms  {s.throughput_mean:>7.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Stress test")
    parser.add_argument("--test", type=str, default="stress",
                        choices=["stress", "extreme", "bimodal", "all"],
                        help="Test type")
    
    args = parser.parse_args()
    
    if args.test == "stress":
        run_stress_test()
    elif args.test == "extreme":
        run_extreme_test()
    elif args.test == "bimodal":
        run_bimodal_test()
    elif args.test == "all":
        run_stress_test()
        run_extreme_test()
        run_bimodal_test()


if __name__ == "__main__":
    main()
