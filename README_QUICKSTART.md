# vLLM End-to-End Testing - Quick Start Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional for testing, required for production)
- 16GB+ RAM

### 1. Install Dependencies

```bash
# Install vLLM and dependencies
pip install vllm torch fastapi uvicorn openai aiohttp

# Verify installation
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

### 2. Run Comprehensive Test (Recommended First Step)

This script walks you through each optimization step with detailed explanations:

```bash
# Test with small model (fast, requires ~2GB GPU memory)
python koyeb_vllm_setup.py --model facebook/opt-125m

# Output: 10 detailed steps showing how vLLM works
# Expected time: 2-3 minutes
```

**What you'll learn:**
- ✅ How PagedAttention manages memory
- ✅ How Continuous Batching works
- ✅ Token generation and sampling
- ✅ Performance metrics and optimization impact

### 3. Start API Server

```bash
# Start server locally
python koyeb_api_server.py --model facebook/opt-125m --port 8000

# Server starts at http://localhost:8000
```

### 4. Test the API

**Option A: Quick test with curl**
```bash
# Health check
curl http://localhost:8000/health

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Option B: Comprehensive test suite**
```bash
# Run all tests
python test_client.py --url http://localhost:8000

# Interactive chat mode
python test_client.py --url http://localhost:8000 --interactive

# Test continuous batching with 10 concurrent requests
python test_client.py --url http://localhost:8000 --test batch --num-requests 10
```

---

## 📊 Understanding the Optimizations

### Test Output Explanation

When you run `koyeb_vllm_setup.py`, you'll see output like:

```
==================================================================================
STEP 2: MODEL LOADING AND MEMORY ALLOCATION
==================================================================================
Loading model: facebook/opt-125m
✓ Model loaded successfully in 12.34 seconds
  GPU memory allocated: 0.48 GB      ← Model weights
  GPU memory reserved: 2.50 GB        ← Total GPU memory used
  Available for KV cache: 37.50 GB    ← For PagedAttention blocks
```

**What this means:**
- **Model weights (0.48 GB):** The neural network parameters
- **KV cache (37.50 GB available):** Space for PagedAttention blocks
- **PagedAttention divides this into blocks:** Typically 16 tokens per block

### Continuous Batching Demo

```
[Request 1] Sending: What is AI?
[Request 2] Sending: Explain ML:
[Request 3] Sending: What is DL?
[Request 2] ✓ Completed in 1.23s (89.2 tok/s)  ← Finished first
[Request 1] ✓ Completed in 1.45s (92.1 tok/s)
[Request 3] ✓ Completed in 1.67s (88.5 tok/s)

Total wall time: 1.67s (not 1.23+1.45+1.67 = 4.35s!)
```

**Why so fast?**
- All 3 requests processed **concurrently** in a dynamic batch
- As Request 2 finished, it freed GPU memory immediately
- No waiting for slowest request to finish

---

## 🐳 Docker Quick Start

### Build and Run

```bash
# Build Docker image
docker build -f Dockerfile.koyeb -t vllm-test .

# Run with Docker Compose (recommended)
docker-compose up

# Or run directly
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=facebook/opt-125m \
  vllm-test
```

### Test Docker Deployment

```bash
# Wait for model to load (check logs)
docker logs -f vllm-server

# Test once healthy
curl http://localhost:8000/health
```

---

## ☁️ Koyeb Deployment (2 Steps)

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add vLLM deployment files"
git push origin main
```

### Step 2: Deploy on Koyeb

1. Go to [Koyeb Dashboard](https://app.koyeb.com)
2. Click "Create App"
3. Select GitHub repository
4. Configure:
   - **Dockerfile:** `Dockerfile.koyeb`
   - **Instance type:** GPU (T4 recommended)
   - **Environment variables:**
     ```
     MODEL_NAME=facebook/opt-125m
     TENSOR_PARALLEL_SIZE=1
     ```
   - **Port:** 8000
   - **Health check:** `/health`

5. Deploy (takes 15-20 min for first deployment)

### Test Koyeb Deployment

```bash
# Replace with your Koyeb URL
export KOYEB_URL=https://your-app.koyeb.app

# Health check
curl $KOYEB_URL/health

# Run full test suite
python test_client.py --url $KOYEB_URL
```

---

## 📈 Performance Comparison

### Without vLLM (Naive Implementation)
```
Sequential processing:
Request 1: 2.0s ████████████████████
Request 2: 2.0s ████████████████████
Request 3: 2.0s ████████████████████
Total: 6.0s

Memory: 80% wasted (pre-allocated, unused)
GPU Utilization: 30-40%
```

### With vLLM (Optimized)
```
Continuous batching:
Request 1: █████████████████░░░
Request 2: ████████░░░░░░░░░░░░
Request 3: ██████████████████░░
Total: 2.0s (3x faster!)

Memory: <5% wasted (dynamic allocation)
GPU Utilization: 85-95%
```

---

## 🎯 Common Use Cases

### 1. Single Request (Low Latency)
```python
# Best for: Interactive chat, single-user apps
python test_client.py --url $URL --test single
```

### 2. Batch Processing (High Throughput)
```python
# Best for: Processing many documents, batch inference
python test_client.py --url $URL --test batch --num-requests 20
```

### 3. Streaming (Real-time)
```python
# Best for: ChatGPT-like interfaces
# (Use AsyncLLM for production streaming)
```

### 4. RAG with Prefix Caching
```python
# Best for: Q&A systems with common system prompts
# System prompt cached → 5x faster for subsequent requests
```

---

## 🔍 Monitoring and Debugging

### Check Metrics

```bash
# Get current server metrics
curl http://localhost:8000/metrics

# Example output:
{
  "model": "facebook/opt-125m",
  "gpu_memory_allocated_gb": 0.48,
  "gpu_memory_reserved_gb": 2.50,
  "gpu_name": "NVIDIA T4"
}
```

### Monitor GPU Usage

```bash
# Watch GPU memory in real-time
watch -n 0.5 nvidia-smi

# You'll see:
# - Memory spike during model loading
# - Dynamic allocation during inference
# - Immediate deallocation after completion
```

### Check Logs

```bash
# View detailed logs
tail -f vllm_api_server.log

# Look for:
[12345] Processing chat completion request:
  Prompt length: 50 chars
  Max tokens: 100
Generation completed:
  Inference time: 0.234s
  Generated tokens: 67
  Throughput: 286.32 tokens/s
```

---

## 🛠️ Troubleshooting

### Issue: "Out of Memory"
**Solution:**
```bash
# Use smaller model
python koyeb_api_server.py --model facebook/opt-125m

# Or reduce GPU memory utilization
# (Edit koyeb_api_server.py, line: gpu_memory_utilization=0.9 → 0.7)
```

### Issue: "Slow first request"
**Reason:** Model loading from disk to GPU
**Solution:** This is normal, subsequent requests are fast

### Issue: "Model download timeout"
**Solution:**
```bash
# Pre-download model
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
  AutoTokenizer.from_pretrained('facebook/opt-125m'); \
  AutoModelForCausalLM.from_pretrained('facebook/opt-125m')"
```

---

## 📚 Next Steps

### Try Larger Models
```bash
# Llama 2 7B (requires 16GB+ GPU)
python koyeb_api_server.py --model meta-llama/Llama-2-7b-hf

# Mistral 7B
python koyeb_api_server.py --model mistralai/Mistral-7B-v0.1

# Quantized (4-bit, less memory)
python koyeb_api_server.py --model TheBloke/Llama-2-7B-GPTQ
```

### Multi-GPU Deployment
```bash
# Tensor parallelism (split model across 2 GPUs)
python koyeb_api_server.py \
  --model meta-llama/Llama-2-13b-hf \
  --tensor-parallel-size 2
```

### Production Checklist
- [ ] Add authentication (API keys)
- [ ] Set up rate limiting
- [ ] Configure HTTPS/SSL
- [ ] Add monitoring (Prometheus/Grafana)
- [ ] Set up logging aggregation
- [ ] Implement request queuing
- [ ] Add load balancing

---

## 📖 Full Documentation

For complete details, see:
- **[KOYEB_DEPLOYMENT_GUIDE.md](KOYEB_DEPLOYMENT_GUIDE.md)** - Full deployment guide
- **[vLLM Docs](https://docs.vllm.ai)** - Official documentation
- **[Koyeb Docs](https://www.koyeb.com/docs)** - Platform documentation

---

## 💡 Key Takeaways

1. **PagedAttention** = 2-4x more requests in same GPU memory
2. **Continuous Batching** = 2-10x higher throughput
3. **vLLM** = Production-ready, OpenAI-compatible API
4. **Easy Deployment** = Works on Koyeb, AWS, GCP, Azure

**Start testing now:**
```bash
python koyeb_vllm_setup.py --model facebook/opt-125m
```

---

**Questions?** Check [KOYEB_DEPLOYMENT_GUIDE.md](KOYEB_DEPLOYMENT_GUIDE.md) for detailed explanations!

🚀 **Happy Testing!**
