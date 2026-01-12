# Step-by-Step Guide: Deploy and Test vLLM on Koyeb

**Situation:** You don't have a GPU locally, but you have Koyeb access with GPU instances.

**Goal:** Deploy vLLM on Koyeb and test all optimizations end-to-end.

---

## 📋 Prerequisites

- [ ] Koyeb account (with GPU access)
- [ ] GitHub account
- [ ] Git installed locally
- [ ] Python 3.10+ installed locally (for running test client)

---

## 🚀 Phase 1: Prepare Files for Deployment (5 minutes)

### Step 1.1: Verify You Have All Required Files

Check that these files exist in your `/home/ubuntu/Wajeeha-Data/CUDA/vLLM/` directory:

```bash
cd /home/ubuntu/Wajeeha-Data/CUDA/vLLM/

ls -la
```

You should see:
- ✅ `Dockerfile.koyeb`
- ✅ `koyeb_api_server.py`
- ✅ `koyeb_vllm_setup.py`
- ✅ `test_client.py`
- ✅ `requirements.txt`

### Step 1.2: Create a `.gitignore` File

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# vLLM cache
hf_cache/
*.log

# IDE
.vscode/
.idea/

# OS
.DS_Store
EOF
```

### Step 1.3: Initialize Git Repository (if not already done)

```bash
# Check if already a git repo
git status

# If not, initialize
git init

# Add files
git add Dockerfile.koyeb koyeb_api_server.py koyeb_vllm_setup.py test_client.py requirements.txt .gitignore

# Commit
git commit -m "Add vLLM deployment files for Koyeb"
```

---

## 📤 Phase 2: Push to GitHub (5 minutes)

### Step 2.1: Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click **"New repository"** (green button)
3. Repository name: `vllm-koyeb-deployment`
4. Make it **Public** (Koyeb needs access)
5. **Don't** initialize with README (we already have files)
6. Click **"Create repository"**

### Step 2.2: Push Your Code

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/vllm-koyeb-deployment.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Verify:** Go to your GitHub repository URL and confirm all files are visible.

---

## ☁️ Phase 3: Deploy on Koyeb (20 minutes)

### Step 3.1: Create New App

1. Go to [app.koyeb.com](https://app.koyeb.com)
2. Click **"Create App"** or **"Deploy"**
3. Choose **"Deploy from GitHub"**

### Step 3.2: Connect GitHub Repository

1. **Authorize GitHub** (if first time)
2. **Select repository:** `YOUR_USERNAME/vllm-koyeb-deployment`
3. **Branch:** `main`

### Step 3.3: Configure Build Settings

```
Builder: Docker
Dockerfile: Dockerfile.koyeb
```

### Step 3.4: Configure Instance Type

**IMPORTANT:** Select a GPU instance!

Recommended options:
- **For testing (cheapest):**
  - Instance: GPU Small (NVIDIA T4)
  - vCPU: 2-4
  - RAM: 8-16GB
  - GPU: T4 (16GB VRAM)
  - Model: `facebook/opt-125m` or `gpt2`

- **For production (better performance):**
  - Instance: GPU Medium (NVIDIA A10)
  - vCPU: 4-8
  - RAM: 16-32GB
  - GPU: A10 (24GB VRAM)
  - Model: `meta-llama/Llama-2-7b-hf`

### Step 3.5: Set Environment Variables

Add these environment variables:

```bash
MODEL_NAME=facebook/opt-125m
TENSOR_PARALLEL_SIZE=1
HF_HOME=/app/hf_cache
PYTHONUNBUFFERED=1
```

**Why these values?**
- `facebook/opt-125m`: Small model (125M parameters), fast to download, good for testing
- `TENSOR_PARALLEL_SIZE=1`: Use 1 GPU (change to 2+ if using multiple GPUs)
- `HF_HOME`: Cache directory for model weights

### Step 3.6: Configure Ports and Health Check

```
Port: 8000
Health check path: /health
Health check protocol: HTTP
```

### Step 3.7: Set Scaling

```
Min instances: 1
Max instances: 1
```

(Keep it simple for testing)

### Step 3.8: Deploy!

1. Review all settings
2. Click **"Deploy"**
3. Wait for deployment (15-20 minutes for first deployment)

**What happens during deployment:**
1. Koyeb pulls your code from GitHub
2. Builds Docker image (installs CUDA, PyTorch, vLLM)
3. Starts container with GPU
4. Downloads model from HuggingFace (12-500MB depending on model)
5. Loads model into GPU memory
6. Starts API server

### Step 3.9: Monitor Deployment

Click on your deployment to see:
- **Build logs:** Docker image building
- **Runtime logs:** Model loading, server startup
- **Health status:** Wait until it shows "Healthy"

**Look for these log messages:**
```
✓ Model loaded successfully in 12.34 seconds
GPU memory allocated: 0.48 GB
API server ready to accept requests
```

### Step 3.10: Get Your Deployment URL

Once deployed, Koyeb will give you a URL like:
```
https://your-app-name-YOUR_ORG.koyeb.app
```

**Copy this URL!** You'll need it for testing.

---

## ✅ Phase 4: Verify Deployment (5 minutes)

### Step 4.1: Test Health Endpoint

```bash
# Replace with your actual URL
export KOYEB_URL=https://your-app-name-YOUR_ORG.koyeb.app

# Test health check
curl $KOYEB_URL/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model": "facebook/opt-125m",
  "tensor_parallel_size": 1
}
```

✅ **If you see this, your deployment is working!**

### Step 4.2: Test Metrics Endpoint

```bash
curl $KOYEB_URL/metrics
```

**Expected response:**
```json
{
  "model": "facebook/opt-125m",
  "tensor_parallel_size": 1,
  "gpu_memory_allocated_gb": 0.48,
  "gpu_memory_reserved_gb": 2.50,
  "gpu_name": "NVIDIA Tesla T4"
}
```

✅ **This confirms GPU is being used!**

### Step 4.3: Test API Root

```bash
curl $KOYEB_URL/
```

**Expected response:**
```json
{
  "message": "vLLM API Server",
  "model": "facebook/opt-125m",
  "endpoints": {
    "health": "/health",
    "metrics": "/metrics",
    "chat": "/v1/chat/completions",
    "completions": "/v1/completions",
    "models": "/v1/models"
  }
}
```

---

## 🧪 Phase 5: Run Comprehensive Tests (10 minutes)

Now let's test all vLLM optimizations from your local laptop (without GPU)!

### Step 5.1: Install Test Client Dependencies Locally

```bash
# On your local laptop
cd /home/ubuntu/Wajeeha-Data/CUDA/vLLM/

pip install aiohttp requests
```

### Step 5.2: Run Full Test Suite

```bash
python test_client.py --url $KOYEB_URL
```

**What this does:**
1. ✅ Health check
2. ✅ Metrics check
3. ✅ **Test 1:** Single request (basic inference)
4. ✅ **Test 2:** Continuous batching (5 concurrent requests)
5. ✅ **Test 3:** Varying output lengths (PagedAttention demo)
6. ✅ **Test 4:** Temperature sampling (different creativity levels)

**Expected output:**
```
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀
VLLM END-TO-END TESTING
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀

================================================================================
HEALTH CHECK
================================================================================
✓ Status: healthy
  Model: facebook/opt-125m
  Tensor Parallel: 1

================================================================================
SERVER METRICS
================================================================================
Model: facebook/opt-125m
GPU: NVIDIA Tesla T4
GPU Memory Allocated: 0.48 GB
GPU Memory Reserved: 2.50 GB

================================================================================
TEST 1: SINGLE REQUEST
================================================================================
Testing basic inference pipeline...

Prompt: Explain what machine learning is in one paragraph:

✓ Response generated successfully

Response:
--------------------------------------------------------------------------------
Machine learning is a subset of artificial intelligence that involves...
--------------------------------------------------------------------------------

📊 Metrics:
  Prompt tokens: 11
  Completion tokens: 87
  Total time: 1.234s
  Throughput: 70.49 tokens/s
  Finish reason: stop

================================================================================
TEST 2: CONTINUOUS BATCHING (5 CONCURRENT REQUESTS)
================================================================================
Demonstrating vLLM's continuous batching optimization...
All requests sent simultaneously - vLLM will batch them dynamically

[Request 1] Sending: What is artificial intelligence?...
[Request 2] Sending: Explain quantum computing briefly:...
[Request 3] Sending: How do neural networks work?...
[Request 4] Sending: What is deep learning?...
[Request 5] Sending: Describe natural language processing:...
[Request 2] ✓ Completed in 1.23s (81.30 tok/s)
[Request 4] ✓ Completed in 1.45s (68.97 tok/s)
[Request 1] ✓ Completed in 1.67s (59.88 tok/s)
[Request 5] ✓ Completed in 1.78s (56.18 tok/s)
[Request 3] ✓ Completed in 1.89s (52.91 tok/s)

--------------------------------------------------------------------------------
📊 CONTINUOUS BATCHING RESULTS
--------------------------------------------------------------------------------
Total requests: 5
Total wall time: 1.89s (all concurrent)
Average request time: 1.60s
Total tokens generated: 325
Average throughput: 63.85 tokens/s
Aggregate throughput: 171.96 tokens/s

💡 What just happened:
  • All requests were batched together dynamically
  • As shorter sequences finished, they freed GPU memory
  • New requests could start immediately (continuous batching)
  • Total throughput is much higher than sequential processing

[... more tests ...]

================================================================================
✅ ALL TESTS COMPLETED SUCCESSFULLY
================================================================================

Key Observations:
  ✓ PagedAttention: Dynamic memory allocation
  ✓ Continuous Batching: Concurrent request processing
  ✓ FlashAttention: Fast attention computation
  ✓ Token Sampling: Flexible generation strategies

================================================================================
```

**Time taken:** ~2-3 minutes total

### Step 5.3: Run Individual Tests

```bash
# Test only continuous batching with 10 requests
python test_client.py --url $KOYEB_URL --test batch --num-requests 10

# Test only single request
python test_client.py --url $KOYEB_URL --test single

# Test sampling parameters
python test_client.py --url $KOYEB_URL --test sampling
```

### Step 5.4: Interactive Chat Mode

```bash
python test_client.py --url $KOYEB_URL --interactive
```

**Example session:**
```
================================================================================
INTERACTIVE CHAT MODE
================================================================================
Type your messages below. Enter 'quit' to exit.

You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence that
enables computers to learn from data without being explicitly programmed...

(87 tokens, 70.49 tok/s)

You: Explain it in simple terms
Assistant: It's like teaching a computer to recognize patterns, similar to
how you learned to recognize cats by seeing many examples...

(65 tokens, 75.23 tok/s)

You: quit

Goodbye!
```

---

## 📊 Phase 6: Understanding What You're Testing (10 minutes)

### Test 1: Single Request - Basic Inference Pipeline

**What's being tested:**
```
Your laptop → HTTP Request → Koyeb (GPU) → Model Inference → Response
```

**vLLM optimizations in action:**
1. **Tokenization:** "What is ML?" → [910, 525, 12500, ...]
2. **PagedAttention:** Allocates KV cache blocks dynamically
3. **Model forward pass:** Transformer layers with FlashAttention
4. **Sampling:** Temperature/top-p/top-k filtering
5. **Detokenization:** Token IDs → "Machine learning is..."

**What to observe:**
- Throughput: ~50-100 tokens/s (varies by model/GPU)
- Latency: ~1-2 seconds for 100 tokens
- GPU memory: Check metrics endpoint

### Test 2: Continuous Batching - Multiple Concurrent Requests

**What's being tested:**
```
5 requests sent simultaneously → vLLM batches dynamically
```

**Key insight:**
```
Traditional batching:
  Req 1: ████████████████ (2s)
  Req 2: ████████ (1s, but waits)
  Req 3: ████████████ (1.5s, but waits)
  Total: 2s + 2s + 2s = 6s (sequential)

vLLM continuous batching:
  Req 1: ████████████████ (2s)
  Req 2: ████████         (1s, frees memory)
  Req 3:   ████████████   (1.5s, uses freed memory)
  Total: 2s (concurrent!)
```

**What to observe:**
- Total wall time ≈ longest request time (NOT sum of all)
- Aggregate throughput >> single request throughput
- Shorter requests finish first, don't block others

### Test 3: Varying Lengths - PagedAttention Demo

**What's being tested:**
- Short (10 tokens), Medium (50 tokens), Long (200 tokens)

**Key insight:**
```
Without PagedAttention:
  Short:  Pre-allocate 2048 tokens → Waste 99.5%
  Medium: Pre-allocate 2048 tokens → Waste 97.5%
  Long:   Pre-allocate 2048 tokens → Waste 90%

With PagedAttention:
  Short:  Allocate 1 block (16 tokens) → Waste <40%
  Medium: Allocate 4 blocks (64 tokens) → Waste <22%
  Long:   Allocate 13 blocks (208 tokens) → Waste <4%
```

**What to observe:**
- Memory usage adapts to actual sequence length
- No pre-allocation waste
- Blocks freed immediately after completion

### Test 4: Temperature Sampling

**What's being tested:**
- Temperature 0.0 (deterministic)
- Temperature 0.5 (focused)
- Temperature 1.0 (balanced)
- Temperature 1.5 (creative)

**Key insight:**
```
Logits: [2.1, 1.8, 1.5, 0.9, 0.3, ...]

Temperature = 0.0:
  Always pick highest: token 0 (100%)
  Output: Deterministic, repeatable

Temperature = 1.0:
  Sample from distribution: [0.23, 0.19, 0.15, 0.08, ...]
  Output: Balanced creativity

Temperature = 1.5:
  Flatten distribution: [0.18, 0.17, 0.16, 0.12, ...]
  Output: Very creative, less coherent
```

**What to observe:**
- Temperature 0.0: Same output every time
- Higher temperature: More diverse, less predictable

---

## 🔍 Phase 7: Monitor Performance (Ongoing)

### Step 7.1: View Koyeb Logs

1. Go to Koyeb dashboard
2. Click on your app
3. Go to **"Logs"** tab
4. Enable **"Auto-refresh"**

**Look for:**
```
[12345] Processing chat completion request:
  Prompt length: 50 chars
  Max tokens: 100
Generation completed:
  Inference time: 0.234s
  Generated tokens: 67
  Throughput: 286.32 tokens/s
```

### Step 7.2: Check GPU Utilization

In Koyeb metrics (if available) or logs, look for:
- GPU memory usage: Should be 80-95% during inference
- GPU utilization: Should be high during forward pass
- Block allocation: Dynamic, adapts to requests

### Step 7.3: Benchmark Performance

```bash
# Stress test with many concurrent requests
python test_client.py --url $KOYEB_URL --test batch --num-requests 20

# Measure throughput
# Expected: 150-300 tokens/s aggregate (T4 GPU, OPT-125M)
```

---

## 🎓 Phase 8: Experiment and Learn (Optional)

### Experiment 1: Try a Larger Model

**Edit deployment on Koyeb:**
1. Go to your app settings
2. Update environment variable:
   ```
   MODEL_NAME=meta-llama/Llama-2-7b-hf
   ```
3. Redeploy

**Requirements:**
- GPU: A10 or A100 (24GB+ VRAM)
- First deployment: 30-40 minutes (model download ~13GB)

**Expected improvement:**
- Better quality responses
- Similar throughput (model is larger but optimized)

### Experiment 2: Multi-GPU Deployment

**Edit deployment on Koyeb:**
1. Select instance with 2+ GPUs
2. Update environment variables:
   ```
   MODEL_NAME=meta-llama/Llama-2-13b-hf
   TENSOR_PARALLEL_SIZE=2
   ```

**What happens:**
- Model split across 2 GPUs (tensor parallelism)
- Each GPU computes part of each layer
- Enables running larger models

### Experiment 3: Stress Test

```bash
# Create a stress test script
cat > stress_test.py << 'EOF'
import asyncio
import aiohttp
import time
import sys

async def stress_test(url, num_requests=50):
    async def send_request(session, req_id):
        start = time.time()
        async with session.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": f"Request {req_id}"}],
                "max_tokens": 50
            }
        ) as response:
            result = await response.json()
            elapsed = time.time() - start
            return elapsed

    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        tasks = [send_request(session, i) for i in range(num_requests)]
        times = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    print(f"Completed {num_requests} requests in {total_time:.2f}s")
    print(f"Average latency: {sum(times)/len(times):.2f}s")
    print(f"Throughput: {num_requests/total_time:.2f} req/s")

asyncio.run(stress_test(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50))
EOF

# Run stress test
python stress_test.py $KOYEB_URL 50
```

---

## 🐛 Troubleshooting

### Problem: Health check fails

**Symptoms:**
```
curl: (7) Failed to connect to host
```

**Solution:**
1. Check deployment logs in Koyeb
2. Wait for model to load (can take 20+ minutes first time)
3. Look for "API server ready to accept requests" in logs

### Problem: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use smaller model: `facebook/opt-125m` instead of `Llama-2-7b-hf`
2. Reduce `max_tokens` in requests (100 instead of 500)
3. Upgrade to GPU with more VRAM (A10 or A100)

### Problem: Slow responses

**Possible causes:**
1. Model still loading (check logs)
2. Using CPU instead of GPU (check metrics endpoint - should show GPU name)
3. Network latency (Koyeb server location)

**Solutions:**
1. Wait for full model load
2. Verify GPU is enabled in Koyeb instance settings
3. Accept network latency (typically 100-300ms)

### Problem: Model download timeout

**Symptoms:**
```
Timeout downloading model from HuggingFace
```

**Solutions:**
1. Use smaller model for testing
2. Increase build timeout in Koyeb settings
3. Pre-download model in Dockerfile (advanced)

---

## ✅ Success Checklist

- [ ] Files created and pushed to GitHub
- [ ] Koyeb app created with GPU instance
- [ ] Environment variables configured
- [ ] Deployment successful (status: Healthy)
- [ ] Health check passes (`/health`)
- [ ] Metrics show GPU usage (`/metrics`)
- [ ] Test 1 (single request) passes
- [ ] Test 2 (continuous batching) passes
- [ ] Test 3 (varying lengths) passes
- [ ] Test 4 (temperature sampling) passes
- [ ] Interactive chat works
- [ ] Logs show optimization metrics

---

## 📈 Expected Performance Benchmarks

### Small Model (OPT-125M, T4 GPU)
```
Single request:     50-100 tokens/s
Concurrent (5):     150-250 tokens/s aggregate
Latency (P50):      200-400ms
GPU utilization:    85-95%
```

### Medium Model (Llama-2-7B, A10 GPU)
```
Single request:     30-60 tokens/s
Concurrent (5):     100-200 tokens/s aggregate
Latency (P50):      400-800ms
GPU utilization:    90-95%
```

---

## 🎉 Next Steps

Once everything works:

1. **Explore the code:**
   - Read [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)
   - Study how PagedAttention works
   - Understand continuous batching algorithm

2. **Try different models:**
   - Mistral 7B (instruction-tuned)
   - Llama 2 13B (better quality)
   - Qwen models (multilingual)

3. **Add features:**
   - Authentication (API keys)
   - Rate limiting
   - Streaming responses
   - Prefix caching for RAG

4. **Monitor production:**
   - Set up alerting
   - Track token usage
   - Monitor costs
   - Log request patterns

---

## 📞 Getting Help

If you get stuck:

1. **Check Koyeb logs:** Most issues visible in deployment logs
2. **Test health endpoint:** Confirms basic connectivity
3. **Test metrics endpoint:** Confirms GPU usage
4. **Review this guide:** Step-by-step troubleshooting

**Common mistakes:**
- ❌ Forgot to select GPU instance (uses CPU instead)
- ❌ Wrong environment variable name (typo)
- ❌ Model name doesn't exist on HuggingFace
- ❌ Insufficient GPU memory for model size

---

**Ready to start?** Begin with Phase 1! 🚀

```bash
cd /home/ubuntu/Wajeeha-Data/CUDA/vLLM/
ls -la  # Verify files exist
```
