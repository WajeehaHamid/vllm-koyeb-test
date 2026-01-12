# vLLM on Koyeb - Deployment Checklist

Print this checklist and check off items as you complete them.

---

## Phase 1: Prepare Files ⏱️ 5 minutes

- [ ] Navigate to project directory: `cd /home/ubuntu/Wajeeha-Data/CUDA/vLLM/`
- [ ] Verify files exist: `ls -la`
- [ ] Create `.gitignore` file
- [ ] Initialize git repository: `git init`
- [ ] Add files: `git add .`
- [ ] Commit: `git commit -m "Add vLLM deployment files"`

---

## Phase 2: Push to GitHub ⏱️ 5 minutes

- [ ] Create new GitHub repository: `vllm-koyeb-deployment`
- [ ] Make it **Public**
- [ ] Copy repository URL
- [ ] Add remote: `git remote add origin <YOUR_URL>`
- [ ] Push: `git push -u origin main`
- [ ] Verify files visible on GitHub

---

## Phase 3: Deploy on Koyeb ⏱️ 20 minutes

### Setup
- [ ] Login to [app.koyeb.com](https://app.koyeb.com)
- [ ] Click **"Create App"**
- [ ] Select **"Deploy from GitHub"**
- [ ] Authorize GitHub (if needed)
- [ ] Select repository: `vllm-koyeb-deployment`
- [ ] Branch: `main`

### Configure Build
- [ ] Builder: **Docker**
- [ ] Dockerfile: `Dockerfile.koyeb`

### Configure Instance
- [ ] Instance type: **GPU Small (T4)** or **GPU Medium (A10)**
- [ ] vCPU: 2-4
- [ ] RAM: 8-16GB

### Set Environment Variables
- [ ] `MODEL_NAME=facebook/opt-125m`
- [ ] `TENSOR_PARALLEL_SIZE=1`
- [ ] `HF_HOME=/app/hf_cache`
- [ ] `PYTHONUNBUFFERED=1`

### Configure Service
- [ ] Port: `8000`
- [ ] Health check path: `/health`
- [ ] Health check protocol: `HTTP`

### Deploy
- [ ] Review all settings
- [ ] Click **"Deploy"**
- [ ] Wait 15-20 minutes
- [ ] Monitor logs for: "API server ready to accept requests"
- [ ] Status shows: **"Healthy"**
- [ ] Copy deployment URL: `https://your-app-name.koyeb.app`

---

## Phase 4: Verify Deployment ⏱️ 5 minutes

- [ ] Set URL: `export KOYEB_URL=https://your-app-name.koyeb.app`
- [ ] Test health: `curl $KOYEB_URL/health`
  - Expected: `{"status": "healthy", ...}`
- [ ] Test metrics: `curl $KOYEB_URL/metrics`
  - Expected: Shows GPU name and memory
- [ ] Test root: `curl $KOYEB_URL/`
  - Expected: Shows available endpoints

---

## Phase 5: Run Tests ⏱️ 10 minutes

### Install Client Dependencies
- [ ] On local machine: `pip install aiohttp requests`

### Run Test Suite
- [ ] Full test: `python test_client.py --url $KOYEB_URL`
- [ ] Test 1 passes: Single request ✅
- [ ] Test 2 passes: Continuous batching (5 requests) ✅
- [ ] Test 3 passes: Varying lengths ✅
- [ ] Test 4 passes: Temperature sampling ✅

### Observe Results
- [ ] See throughput metrics (tokens/s)
- [ ] See latency metrics (ms)
- [ ] Continuous batching demo shows concurrent processing
- [ ] All tests complete successfully

### Optional: Interactive Chat
- [ ] Run: `python test_client.py --url $KOYEB_URL --interactive`
- [ ] Test conversation flow
- [ ] Type `quit` to exit

---

## Phase 6: Understand Optimizations ⏱️ 10 minutes

### PagedAttention
- [ ] Read Test 3 results (varying lengths)
- [ ] Observe: Memory adapts to sequence length
- [ ] Understand: Dynamic block allocation

### Continuous Batching
- [ ] Read Test 2 results (concurrent requests)
- [ ] Observe: Total time ≈ longest request (not sum!)
- [ ] Understand: Requests enter/exit batch dynamically

### FlashAttention
- [ ] Check throughput in metrics
- [ ] Observe: 50-100+ tokens/s
- [ ] Understand: Optimized attention computation

### Token Sampling
- [ ] Read Test 4 results (temperature)
- [ ] Observe: Temperature affects creativity
- [ ] Understand: Higher temp = more random

---

## Phase 7: Monitor ⏱️ Ongoing

### Koyeb Dashboard
- [ ] View logs tab
- [ ] Enable auto-refresh
- [ ] Look for inference time metrics
- [ ] Monitor GPU memory usage

### Performance Metrics
- [ ] Check `/metrics` endpoint
- [ ] GPU memory allocated: ~0.5-5GB (depends on model)
- [ ] GPU utilization: 85-95% during inference

---

## Success Criteria ✅

All must be checked:

- [ ] ✅ Deployment status: Healthy
- [ ] ✅ Health check returns 200 OK
- [ ] ✅ Metrics show GPU name
- [ ] ✅ Single request completes successfully
- [ ] ✅ Continuous batching demo works
- [ ] ✅ Throughput: 50-100+ tokens/s
- [ ] ✅ Latency: <2s for 100 tokens
- [ ] ✅ Logs show optimization metrics

---

## Troubleshooting Quick Reference

| Problem | Quick Fix |
|---------|-----------|
| Health check fails | Wait 20 min, check logs for "API server ready" |
| No GPU in metrics | Verify GPU instance selected, not CPU |
| Out of memory | Use smaller model: `facebook/opt-125m` |
| Slow responses | Check if model still loading (view logs) |
| Download timeout | Use smaller model, increase timeout |

---

## Quick Commands Reference

```bash
# Set URL
export KOYEB_URL=https://your-app-name.koyeb.app

# Health check
curl $KOYEB_URL/health

# Metrics
curl $KOYEB_URL/metrics

# Full test
python test_client.py --url $KOYEB_URL

# Interactive chat
python test_client.py --url $KOYEB_URL --interactive

# Stress test (10 concurrent)
python test_client.py --url $KOYEB_URL --test batch --num-requests 10
```

---

## Time Breakdown

| Phase | Time | What Happens |
|-------|------|--------------|
| Phase 1 | 5 min | Prepare files locally |
| Phase 2 | 5 min | Push to GitHub |
| Phase 3 | 20 min | Deploy on Koyeb (first time) |
| Phase 4 | 5 min | Verify deployment |
| Phase 5 | 10 min | Run all tests |
| **TOTAL** | **45 min** | Complete setup and testing |

Subsequent deploys: ~5-10 minutes (model cached)

---

## Next Steps After Success

- [ ] Read [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) for deep dive
- [ ] Try larger model (Llama-2-7B)
- [ ] Experiment with different temperatures
- [ ] Test with your own prompts
- [ ] Monitor costs on Koyeb dashboard
- [ ] Plan production deployment (auth, rate limiting)

---

**Current Status:** __________

**Deployment URL:** __________

**Date Completed:** __________

---

🎉 **Congratulations on deploying vLLM!** 🎉
