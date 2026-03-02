#!/usr/bin/env python3
"""
Sensitivity analysis experiments
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memsched import (
    get_model_config, SystemConfig,
    get_scheduler,
    Simulator,
    WorkloadConfig, WorkloadGenerator,
)


def sensitivity_arrival_rate():
    """Analyze sensitivity to arrival rate"""
    
    print("=" * 60)
    print("Sensitivity Analysis: Arrival Rate")
    print("=" * 60)
    
    model_config = get_model_config("llama-7b")
    system_config = SystemConfig(
        gpu_memory_mb=24 * 1024,
        model_memory_mb=14 * 1024,
        max_batch_size=32,
    )
    
    arrival_rates = [5.0, 10.0, 15.0, 20.0, 25.0]
    schedulers = ["fcfs", "sjf_predicted", "mlfq", "memsched"]
    
    results = {s: {"latency": [], "throughput": []} for s in schedulers}
    
    for rate in arrival_rates:
        print(f"\nArrival rate: {rate} req/s")
        
        workload_config = WorkloadConfig(
            num_requests=100,
            arrival_rate=rate,
            seed=42,
        )
        generator = WorkloadGenerator(workload_config)
        requests = generator.generate()
        
        for sched_name in schedulers:
            scheduler = get_scheduler(sched_name)
            simulator = Simulator(model_config, system_config, scheduler)
            result = simulator.run(requests, show_progress=False)
            
            results[sched_name]["latency"].append(result.avg_latency)
            results[sched_name]["throughput"].append(result.throughput)
            
            print(f"  {sched_name}: latency={result.avg_latency:.1f}ms, "
                  f"throughput={result.throughput:.2f} req/s")
    
    # Summary tables
    print("\n" + "=" * 60)
    print("Summary: Average Latency (ms)")
    print("=" * 60)
    
    print(f"\n{'Rate':<10}", end="")
    for s in schedulers:
        print(f"{s:<15}", end="")
    print()
    print("-" * 70)
    
    for i, rate in enumerate(arrival_rates):
        print(f"{rate:<10.1f}", end="")
        for s in schedulers:
            print(f"{results[s]['latency'][i]:<15.1f}", end="")
        print()
    
    print("\n" + "=" * 60)
    print("Summary: Throughput (req/s)")
    print("=" * 60)
    
    print(f"\n{'Rate':<10}", end="")
    for s in schedulers:
        print(f"{s:<15}", end="")
    print()
    print("-" * 70)
    
    for i, rate in enumerate(arrival_rates):
        print(f"{rate:<10.1f}", end="")
        for s in schedulers:
            print(f"{results[s]['throughput'][i]:<15.2f}", end="")
        print()


def sensitivity_memory():
    """Analyze sensitivity to available memory"""
    
    print("=" * 60)
    print("Sensitivity Analysis: Available Memory")
    print("=" * 60)
    
    model_config = get_model_config("llama-7b")
    
    # Different GPU memory sizes (GB)
    memory_sizes = [16, 24, 32, 40, 48]
    schedulers = ["fcfs", "sjf_predicted", "memsched"]
    
    results = {s: {"latency": [], "throughput": []} for s in schedulers}
    
    for mem_gb in memory_sizes:
        print(f"\nGPU Memory: {mem_gb} GB (Available KV: {mem_gb - 14:.0f} GB)")
        
        system_config = SystemConfig(
            gpu_memory_mb=mem_gb * 1024,
            model_memory_mb=14 * 1024,
            max_batch_size=32,
        )
        
        workload_config = WorkloadConfig(
            num_requests=100,
            arrival_rate=15.0,
            seed=42,
        )
        generator = WorkloadGenerator(workload_config)
        requests = generator.generate()
        
        for sched_name in schedulers:
            scheduler = get_scheduler(sched_name)
            simulator = Simulator(model_config, system_config, scheduler)
            result = simulator.run(requests, show_progress=False)
            
            results[sched_name]["latency"].append(result.avg_latency)
            results[sched_name]["throughput"].append(result.throughput)
            
            print(f"  {sched_name}: latency={result.avg_latency:.1f}ms, "
                  f"throughput={result.throughput:.2f} req/s")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary: Average Latency (ms)")
    print("=" * 60)
    
    print(f"\n{'Memory(GB)':<12}", end="")
    for s in schedulers:
        print(f"{s:<15}", end="")
    print()
    print("-" * 60)
    
    for i, mem in enumerate(memory_sizes):
        print(f"{mem:<12}", end="")
        for s in schedulers:
            print(f"{results[s]['latency'][i]:<15.1f}", end="")
        print()


def sensitivity_batch_size():
    """Analyze sensitivity to batch size"""
    
    print("=" * 60)
    print("Sensitivity Analysis: Max Batch Size")
    print("=" * 60)
    
    model_config = get_model_config("llama-7b")
    
    batch_sizes = [4, 8, 16, 32, 64]
    schedulers = ["fcfs", "sjf_predicted", "memsched"]
    
    results = {s: {"latency": [], "throughput": []} for s in schedulers}
    
    for batch_size in batch_sizes:
        print(f"\nMax Batch Size: {batch_size}")
        
        system_config = SystemConfig(
            gpu_memory_mb=24 * 1024,
            model_memory_mb=14 * 1024,
            max_batch_size=batch_size,
        )
        
        workload_config = WorkloadConfig(
            num_requests=100,
            arrival_rate=15.0,
            seed=42,
        )
        generator = WorkloadGenerator(workload_config)
        requests = generator.generate()
        
        for sched_name in schedulers:
            scheduler = get_scheduler(sched_name)
            simulator = Simulator(model_config, system_config, scheduler)
            result = simulator.run(requests, show_progress=False)
            
            results[sched_name]["latency"].append(result.avg_latency)
            results[sched_name]["throughput"].append(result.throughput)
            
            print(f"  {sched_name}: latency={result.avg_latency:.1f}ms, "
                  f"throughput={result.throughput:.2f} req/s")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary: Average Latency (ms)")
    print("=" * 60)
    
    print(f"\n{'BatchSize':<12}", end="")
    for s in schedulers:
        print(f"{s:<15}", end="")
    print()
    print("-" * 60)
    
    for i, bs in enumerate(batch_sizes):
        print(f"{bs:<12}", end="")
        for s in schedulers:
            print(f"{results[s]['latency'][i]:<15.1f}", end="")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sensitivity analysis")
    parser.add_argument("--type", type=str, default="arrival_rate",
                        choices=["arrival_rate", "memory", "batch_size", "all"],
                        help="Type of sensitivity analysis")
    
    args = parser.parse_args()
    
    if args.type == "arrival_rate":
        sensitivity_arrival_rate()
    elif args.type == "memory":
        sensitivity_memory()
    elif args.type == "batch_size":
        sensitivity_batch_size()
    elif args.type == "all":
        sensitivity_arrival_rate()
        print("\n\n")
        sensitivity_memory()
        print("\n\n")
        sensitivity_batch_size()


if __name__ == "__main__":
    main()
