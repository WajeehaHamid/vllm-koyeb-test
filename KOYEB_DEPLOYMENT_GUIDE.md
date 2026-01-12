# vLLM End-to-End Testing and Deployment Guide for Koyeb

## 📋 Table of Contents

1. [Overview](#overview)
2. [Understanding vLLM Optimizations](#understanding-vllm-optimizations)
3. [Local Testing](#local-testing)
4. [Koyeb Deployment](#koyeb-deployment)
5. [Testing the Deployment](#testing-the-deployment)
6. [Understanding Each Optimization Step](#understanding-each-optimization-step)

---

## Overview

This guide provides a comprehensive end-to-end setup for testing and deploying vLLM on Koyeb. You'll learn:

- How vLLM optimizes LLM inference (PagedAttention, Continuous Batching)
- How to test each optimization step locally
- How to deploy on Koyeb with GPU support
- How to interact with the deployed model

---

## Understanding vLLM Optimizations

### 1. **PagedAttention** (Memory Management)

**Problem it solves:**
- Traditional LLMs pre-allocate large contiguous memory blocks for KV cache
- This leads to 60-80% memory fragmentation and waste

**How vLLM solves it:**
```
Traditional approach:
┌─────────────────────────────┐
│ Sequence 1 [████░░░░░░░░░] │ 40% wasted
│ Sequence 2 [██████░░░░░░░] │ 50% wasted
└─────────────────────────────┘

PagedAttention:
┌─────────────────────────────┐
│ [████][████][████][████]    │ 0% wasted
│  S1    S1    S2    S2       │ Dynamic allocation
└─────────────────────────────┘
```

**Benefits:**
- 2-4x more requests fit in same GPU memory
- Zero fragmentation
- Dynamic block allocation as sequences grow

### 2. **Continuous Batching** (Scheduling)

**Problem it solves:**
- Static batching waits for ALL sequences to finish before processing new ones
- GPU sits idle when some sequences finish early

**How vLLM solves it:**
```
Static batching (naive):
Time →
[Req1 ████████████]  ← GPU busy
[Req2 █████]         ← Finishes early, but must wait
[Req3 ████████]
[Req4        ????????] ← Can't start until batch finishes

Continuous batching (vLLM):
Time →
[Req1 ████████████]
[Req2 █████]
[Req3      ████████]
[Req4      █████]      ← Starts as soon as Req2 frees space!
```

**Benefits:**
- 2-10x higher throughput
- Lower latency (no waiting for entire batch)
- 80-95% GPU utilization vs. 20-40%

### 3. **FlashAttention Integration**

**Problem it solves:**
- Standard attention is memory bandwidth-bound
- Multiple passes over data for softmax and output computation

**How vLLM solves it:**
- Fused attention kernels (compute in one pass)
- Reduced memory reads/writes by 4-5x
- 2-3x faster attention computation

### 4. **Prefix Caching**

**Problem it solves:**
- Many requests share common prefixes (e.g., system prompts)
- Redundant KV computation for identical prompt parts

**How vLLM solves it:**
- Cache KV blocks for shared prompt prefixes
- Reuse across multiple requests
- Up to 5x speedup for RAG workloads

---

## Local Testing

### Prerequisites

```bash
# Install vLLM
pip install vllm

# Install dependencies
pip install torch fastapi uvicorn openai requests
```

### Step 1: Run the Comprehensive Test Script

This script walks through each optimization step with detailed logging:

```bash
# Test with a small model (recommended for first run)
python koyeb_vllm_setup.py --model facebook/opt-125m

# Test with a larger model (requires more GPU memory)
python koyeb_vllm_setup.py --model meta-llama/Llama-2-7b-hf

# Test with multiple GPUs (if available)
python koyeb_vllm_setup.py --model meta-llama/Llama-2-7b-hf --tensor-parallel-size 2
```

**What this script does:**

1. **Step 1:** Imports vLLM and checks GPU availability
2. **Step 2:** Loads model and shows memory allocation
3. **Step 3:** Explains PagedAttention configuration
4. **Step 4:** Configures sampling parameters (temperature, top-p, etc.)
5. **Step 5:** Prepares test prompts for batching
6. **Step 6:** Executes inference with continuous batching
7. **Step 7:** Analyzes outputs and token generation
8. **Step 8:** Demonstrates streaming generation
9. **Step 9:** Benchmarks optimization impact
10. **Step 10:** Provides complete pipeline summary

**Expected output:**
```
==================================================================================
STEP 1: IMPORT AND SETUP
==================================================================================
✓ vLLM successfully imported
  Python version: 3.10.12
✓ CUDA available: NVIDIA A100-SXM4-40GB
  Total GPU memory: 40.00 GB

==================================================================================
STEP 2: MODEL LOADING AND MEMORY ALLOCATION
==================================================================================
Loading model: facebook/opt-125m
✓ Model loaded successfully in 12.34 seconds
  GPU memory allocated: 0.48 GB
  GPU memory reserved: 2.50 GB
  Available for KV cache: 37.50 GB

[... continues with all steps ...]
```

### Step 2: Run the API Server Locally

```bash
# Start the server (small model)
python koyeb_api_server.py --model facebook/opt-125m --port 8000

# Or with environment variables
MODEL_NAME=facebook/opt-125m python koyeb_api_server.py
```

**Expected output:**
```
==================================================================================
STARTING VLLM API SERVER
==================================================================================
Loading model: facebook/opt-125m
Tensor parallel size: 1
✓ Model loaded successfully in 12.34 seconds
GPU memory allocated: 0.48 GB
API server ready to accept requests
==================================================================================
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Test the API Locally

```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [
      {"role": "user", "content": "Explain machine learning"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Machine learning is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## Koyeb Deployment

### Option 1: Deploy from GitHub (Recommended)

1. **Fork the vLLM repository** (or push these files to your own repo)

2. **Add the deployment files:**
   - `Dockerfile.koyeb`
   - `koyeb_api_server.py`
   - `koyeb_vllm_setup.py`

3. **Push to GitHub:**
```bash
git add Dockerfile.koyeb koyeb_api_server.py koyeb_vllm_setup.py
git commit -m "Add Koyeb deployment files"
git push
```

4. **Deploy on Koyeb:**
   - Go to [Koyeb Dashboard](https://app.koyeb.com)
   - Click "Create App"
   - Select "GitHub" as source
   - Choose your repository
   - Configure:
     - **Dockerfile path:** `Dockerfile.koyeb`
     - **Instance type:** GPU instance (e.g., NVIDIA T4, A10, or A100)
     - **Environment variables:**
       - `MODEL_NAME=facebook/opt-125m` (or your preferred model)
       - `TENSOR_PARALLEL_SIZE=1`
       - `HF_HOME=/app/hf_cache`
     - **Port:** 8000
     - **Health check path:** `/health`

5. **Deploy and wait** (first deployment takes 15-20 minutes for model download)

### Option 2: Deploy from Docker Hub

1. **Build and push Docker image:**
```bash
# Build image
docker build -f Dockerfile.koyeb -t your-dockerhub-username/vllm-koyeb:latest .

# Push to Docker Hub
docker push your-dockerhub-username/vllm-koyeb:latest
```

2. **Deploy on Koyeb:**
   - Create app from Docker image
   - Use: `your-dockerhub-username/vllm-koyeb:latest`
   - Configure instance type and environment variables as above

### Recommended Instance Types

| Model Size | GPU Type | Memory | Cost/hr (approx) |
|------------|----------|--------|------------------|
| OPT-125M   | T4       | 16GB   | $0.25            |
| Llama-2-7B | T4       | 16GB   | $0.25            |
| Llama-2-7B | A10      | 24GB   | $0.75            |
| Llama-2-13B| A10      | 24GB   | $0.75            |
| Llama-2-70B| A100     | 40GB   | $2.50            |

---

## Testing the Deployment

Once deployed, you'll get a Koyeb URL like: `https://your-app-name.koyeb.app`

### Test 1: Health Check

```bash
curl https://your-app-name.koyeb.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model": "facebook/opt-125m",
  "tensor_parallel_size": 1
}
```

### Test 2: Metrics

```bash
curl https://your-app-name.koyeb.app/metrics
```

Expected response:
```json
{
  "model": "facebook/opt-125m",
  "tensor_parallel_size": 1,
  "gpu_memory_allocated_gb": 0.48,
  "gpu_memory_reserved_gb": 2.50,
  "gpu_name": "NVIDIA T4"
}
```

### Test 3: Chat Completion

```bash
curl https://your-app-name.koyeb.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

Expected response:
```json
{
  "id": "chatcmpl-1704234567890",
  "object": "chat.completion",
  "created": 1704234567,
  "model": "facebook/opt-125m",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing uses quantum bits (qubits)..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 123,
    "total_tokens": 168
  },
  "vllm_stats": {
    "inference_time_seconds": 1.234,
    "tokens_per_second": 99.68
  }
}
```

### Test 4: Batch Processing (Continuous Batching Demo)

```python
import asyncio
import aiohttp

async def send_request(session, prompt, request_id):
    url = "https://your-app-name.koyeb.app/v1/chat/completions"
    payload = {
        "model": "facebook/opt-125m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7
    }

    start = time.time()
    async with session.post(url, json=payload) as response:
        result = await response.json()
        elapsed = time.time() - start
        print(f"Request {request_id}: {elapsed:.2f}s - "
              f"{result['vllm_stats']['tokens_per_second']:.2f} tok/s")
        return result

async def test_continuous_batching():
    """Test continuous batching with concurrent requests"""
    prompts = [
        "Explain machine learning",
        "What is Python?",
        "Describe neural networks",
        "How does AI work?",
        "What is deep learning?"
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, prompt, i)
            for i, prompt in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks)

        print("\n" + "="*60)
        print("CONTINUOUS BATCHING DEMONSTRATION")
        print("="*60)
        print(f"Total requests: {len(results)}")
        print(f"All requests processed concurrently!")
        print(f"This demonstrates vLLM's continuous batching:")
        print(f"  • Requests enter/exit batch dynamically")
        print(f"  • No waiting for entire batch to finish")
        print(f"  • Maximum GPU utilization")

asyncio.run(test_continuous_batching())
```

---

## Understanding Each Optimization Step

### 1. PagedAttention in Action

To see PagedAttention working, look at the server logs when processing requests:

```
[12345] Processing chat completion request:
  Prompt length: 245 chars
  Max tokens: 256
Generation completed:
  Inference time: 1.234s
  Generated tokens: 123
  Throughput: 99.68 tokens/s
```

**What's happening behind the scenes:**
1. **Block allocation:** vLLM allocates blocks (typically 16 tokens each) as needed
2. **Non-contiguous storage:** Blocks can be anywhere in GPU memory
3. **Dynamic growth:** More blocks allocated as sequence grows
4. **Immediate release:** Blocks freed when sequence completes

**To observe:**
```bash
# Monitor GPU memory during inference
watch -n 0.5 nvidia-smi

# You'll see memory usage grow smoothly as sequences progress
# And drop sharply when sequences complete
```

### 2. Continuous Batching in Action

Send multiple requests simultaneously to see continuous batching:

```bash
# Send 5 requests in parallel
for i in {1..5}; do
  curl https://your-app-name.koyeb.app/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"facebook/opt-125m\", \"messages\": [{\"role\": \"user\", \"content\": \"Request $i\"}]}" &
done
wait
```

**Observe in logs:**
```
[12345] POST /v1/chat/completions
[12346] POST /v1/chat/completions
[12347] POST /v1/chat/completions
[12348] POST /v1/chat/completions
[12349] POST /v1/chat/completions

# All 5 requests processed together!
# As each finishes, it frees space for new requests

[12345] Completed in 1.234s - Status: 200
[12347] Completed in 1.456s - Status: 200  ← Finished early
[12350] POST /v1/chat/completions  ← New request can start immediately!
```

### 3. Token-Level Scheduling

Each "iteration" of the inference loop:
1. **Scheduler decides:** Which requests to include in next batch
2. **GPU executes:** One forward pass for all selected sequences
3. **Sampling:** Generate next token for each sequence
4. **Update:** Check which sequences finished, add new requests

This happens **every token** (not every sequence), maximizing GPU utilization.

### 4. Memory Efficiency Comparison

**Test this yourself:**

```python
# Without vLLM (naive implementation)
# - Pre-allocate 2048 tokens for each sequence
# - Memory needed: batch_size × 2048 × model_dim × dtype_size
# - 80% of this is wasted for short sequences!

# With vLLM (PagedAttention)
# - Allocate blocks dynamically (16 tokens each)
# - Memory needed: actual_tokens / 16 × block_size
# - Near-zero waste!
```

---

## Monitoring and Debugging

### Check vLLM Logs

```bash
# On Koyeb, view logs in dashboard
# Or via CLI:
koyeb logs your-app-name --follow

# Look for:
# - Model loading time
# - GPU memory allocation
# - Request processing times
# - Token throughput
```

### Common Issues

**1. Out of Memory (OOM)**
```
Solution:
- Use smaller model (opt-125m instead of llama-70b)
- Reduce gpu_memory_utilization (default 0.9 → 0.8)
- Reduce max_tokens in requests
```

**2. Slow First Request**
```
Reason: Model weights need to load from disk to GPU
Solution: This is normal, subsequent requests are fast
```

**3. Model Download Timeout**
```
Solution:
- Use smaller model for testing
- Increase timeout in Dockerfile
- Pre-download model in Docker build
```

---

## Advanced Topics

### 1. Using Larger Models

```bash
# Llama 2 7B (requires 16GB+ GPU)
MODEL_NAME=meta-llama/Llama-2-7b-hf python koyeb_api_server.py

# Mistral 7B
MODEL_NAME=mistralai/Mistral-7B-v0.1 python koyeb_api_server.py

# Quantized models (4-bit, requires less memory)
MODEL_NAME=TheBloke/Llama-2-7B-GPTQ python koyeb_api_server.py
```

### 2. Multi-GPU Deployment

```bash
# Tensor parallelism (split model across GPUs)
python koyeb_api_server.py \
  --model meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4
```

### 3. Prefix Caching for RAG

```python
# System prompt cached across requests
system_prompt = "You are an expert in quantum physics. Answer questions accurately."

# This system prompt KV cache is reused for all requests!
requests = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is superposition?"}
]
```

---

## Summary

This guide covered:

✅ **Understanding vLLM optimizations** (PagedAttention, Continuous Batching)
✅ **Local testing** with comprehensive logging at each step
✅ **Koyeb deployment** with GPU support
✅ **Testing deployment** with curl and Python clients
✅ **Observing optimizations** in action through logs and monitoring

**Key Takeaways:**

1. **PagedAttention** = 2-4x more requests in same memory
2. **Continuous Batching** = 2-10x higher throughput
3. **vLLM** = Production-ready, fast, easy to deploy

**Next Steps:**

- Deploy larger models (Llama 2, Mistral)
- Implement streaming responses
- Add authentication and rate limiting
- Monitor with Prometheus/Grafana
- Scale with multiple instances

---

## Resources

- [vLLM Documentation](https://docs.vllm.ai)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Koyeb Documentation](https://www.koyeb.com/docs)
- [vLLM Blog: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)

---

**Happy Testing! 🚀**
