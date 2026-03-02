MemSched: Memory-Aware Scheduling for LLM Serving
A simulation framework for evaluating KV cache-aware scheduling algorithms in Large Language Model (LLM) serving systems.

Overview
MemSched provides a discrete-event simulator to study how different scheduling strategies affect LLM serving performance under memory constraints. The framework is designed to be extensible and can serve as a foundation for implementing real-world LLM serving schedulers.

Architecture
复制
┌─────────────────────────────────────────────────────────────────────┐
│                         Request Generator                          │
│              (Uniform / Bimodal / Bursty Patterns)                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            Scheduler                                │
│  ┌───────────┬───────────┬───────────┬───────────┬───────────┐     │
│  │   FCFS    │    SJF    │   MLFQ    │   vLLM    │  MemSched │     │
│  │           │           │           │   Style   │           │     │
│  └───────────┴───────────┴───────────┴───────────┴───────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          LLM Engine                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    KV Cache Manager                          │   │
│  │         (Memory Allocation / Eviction / Tracking)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Token Generator                           │   │
│  │              (Prefill / Decode Simulation)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Metrics Collector                            │
│     (Latency / Throughput / Memory Utilization / Preemptions)       │
└─────────────────────────────────────────────────────────────────────┘
Scheduling Algorithms
1. FCFS (First-Come-First-Served)
File: memsched/schedulers/fcfs.py

The simplest scheduling strategy that processes requests in arrival order.

Pros: Fair, no starvation, simple implementation
Cons: Head-of-line blocking, long requests delay short ones
Best for: Low-load scenarios with similar request sizes
2. SJF (Shortest Job First)
File: memsched/schedulers/sjf.py

Prioritizes requests with shorter estimated output lengths.

Pros: Minimizes average waiting time
Cons: Requires output length prediction, may starve long requests
Best for: Workloads with high variance in request sizes
3. MLFQ (Multi-Level Feedback Queue)
File: memsched/schedulers/mlfq.py

Uses multiple priority queues with feedback-based priority adjustment.

Pros: Balances responsiveness and throughput, adapts to request behavior
Cons: More complex, requires tuning of queue parameters
Configuration:
num_queues: Number of priority levels (default: 4)
time_quantum: Base time slice per queue level
4. vLLM-Style Scheduler
File: memsched/schedulers/vllm_scheduler.py

Implements continuous batching with memory-aware preemption, inspired by vLLM.

Pros: High throughput, efficient memory utilization
Cons: May preempt requests under memory pressure
Key Features:
Continuous batching (prefill + decode in same batch)
Memory-triggered preemption with recomputation
Waiting queue management
5. MemSched (Memory-Aware Scheduler)
File: memsched/schedulers/memsched_scheduler.py

Our proposed scheduler that combines memory awareness with intelligent request prioritization.

Pros: Best memory efficiency, minimal preemptions, adaptive
Cons: Higher scheduling overhead
Key Features:
Proactive memory pressure prediction
Completion-progress-aware prioritization
Adaptive batch sizing based on memory state
Module Description
Core Components
Module	File	Description
Request	memsched/core/request.py	Request data structure with timing and state tracking
KV Cache	memsched/core/kv_cache.py	Memory management for key-value caches
LLM Engine	memsched/core/engine.py	Simulates token generation (prefill/decode)
Simulator	memsched/simulator/simulator.py	Discrete-event simulation engine
Metrics	memsched/simulator/metrics.py	Performance measurement and statistics
Installation
bash
复制
git clone https://github.com/eldricwang/LLMServer-Scheduling.git
cd LLMServer-Scheduling
pip install -e .
Quick Start
Run Comparison Experiment
bash
复制
python experiments/run_comparison.py
Run Specific Workload
bash
复制
python experiments/run_comparison.py --workload bursty --num_requests 500
Extending to Real Systems
This simulator is designed as a stepping stone to real LLM serving systems.

Step 1: Replace Token Generator
Replace the simulated token generation with actual model inference:

python
复制
# Current (Simulation)
class SimulatedEngine:
    def generate_tokens(self, batch):
        time.sleep(self.compute_time)
        return [1] * len(batch)

# Real System
class RealEngine:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def generate_tokens(self, batch):
        with torch.no_grad():
            outputs = self.model.forward(batch)
        return outputs.next_tokens
Step 2: Integrate Real KV Cache
Connect to actual GPU memory management:

python
复制
# Current (Simulation)
class SimulatedKVCache:
    def allocate(self, size):
        self.used_memory += size

# Real System
class GPUKVCache:
    def allocate(self, num_tokens, num_layers, hidden_size):
        shape = (num_layers, 2, num_tokens, hidden_size)
        tensor = torch.empty(shape, device='cuda', dtype=torch.float16)
        return tensor
Step 3: Add Network Layer
Wrap the scheduler with an HTTP/gRPC server:

python
复制
from fastapi import FastAPI
from memsched.schedulers import MemSchedScheduler

app = FastAPI()
scheduler = MemSchedScheduler()

@app.post("/generate")
async def generate(prompt: str, max_tokens: int):
    request = Request(prompt=prompt, max_tokens=max_tokens)
    scheduler.add_request(request)
    result = await request.wait_for_completion()
    return {"generated_text": result}
Integration Checklist
Replace SimulatedEngine with real model inference
Replace SimulatedKVCache with GPU memory manager
Add request tokenization (prompt to token IDs)
Add response detokenization (token IDs to text)
Implement async request handling
Add HTTP/gRPC API layer
Add monitoring and logging
Project Structure
复制
memsched/
├── memsched/
│   ├── core/
│   │   ├── request.py
│   │   ├── kv_cache.py
│   │   └── engine.py
│   ├── schedulers/
│   │   ├── base.py
│   │   ├── fcfs.py
│   │   ├── sjf.py
│   │   ├── mlfq.py
│   │   ├── vllm_scheduler.py
│   │   └── memsched_scheduler.py
│   └── simulator/
│       ├── simulator.py
│       ├── metrics.py
│       └── workload.py
├── experiments/
│   └── run_comparison.py
├── tests/
│   └── test_schedulers.py
├── setup.py
├── requirements.txt
└── README.md
License
MIT License

References
vLLM: https://github.com/vllm-project/vllm
Orca: A Distributed Serving System for Transformer-Based Models (OSDI 2022)
PagedAttention: Efficient Memory Management for LLM Serving