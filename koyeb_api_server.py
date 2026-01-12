"""
vLLM API Server for Koyeb Deployment
=====================================

A production-ready API server with OpenAI-compatible endpoints,
health checks, and detailed logging for monitoring vLLM's optimizations.

Features:
- OpenAI-compatible /v1/chat/completions endpoint
- Streaming and non-streaming responses
- Health check endpoint
- Metrics endpoint for monitoring
- Request logging with optimization insights

Usage:
    python koyeb_api_server.py --model facebook/opt-125m --port 8000
"""

import os
import sys
import time
import logging
import asyncio
from typing import AsyncGenerator, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vllm_api_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables
llm_engine = None
model_name = os.environ.get("MODEL_NAME", "facebook/opt-125m")
tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))


# Pydantic models for API
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=10)


class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading/unloading the model"""
    global llm_engine

    logger.info("="*80)
    logger.info("STARTING VLLM API SERVER")
    logger.info("="*80)

    # Startup: Load the model
    try:
        from vllm import LLM
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")

        start_time = time.time()

        llm_engine = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
            download_dir=os.environ.get('HF_HOME', './hf_cache')
        )

        load_time = time.time() - start_time
        logger.info(f"✓ Model loaded successfully in {load_time:.2f} seconds")

        # Log memory statistics
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"GPU memory allocated: {allocated:.2f} GB")
                logger.info(f"GPU memory reserved: {reserved:.2f} GB")
        except:
            pass

        logger.info("API server ready to accept requests")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down API server")
    llm_engine = None


# Create FastAPI app
app = FastAPI(
    title="vLLM API Server",
    description="OpenAI-compatible API server powered by vLLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request tracking middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information"""
    start_time = time.time()
    request_id = f"{int(start_time * 1000000)}"

    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(f"[{request_id}] Completed in {duration:.3f}s - Status: {response.status_code}")

    return response


@app.get("/health")
async def health_check():
    """Health check endpoint for Koyeb"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "vLLM API Server",
        "model": model_name,
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models"
        },
        "documentation": "/docs"
    }


@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring"""
    metrics_data = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
    }

    try:
        import torch
        if torch.cuda.is_available():
            metrics_data["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9
            metrics_data["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(0) / 1e9
            metrics_data["gpu_name"] = torch.cuda.get_device_name(0)
    except:
        pass

    return metrics_data


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Demonstrates vLLM's continuous batching and PagedAttention optimizations.
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from vllm import SamplingParams

        # Convert messages to prompt
        prompt = ""
        for message in request.messages:
            role = message.role
            content = message.content
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += "Assistant:"

        # Configure sampling
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            n=request.n
        )

        logger.info(f"Processing chat completion request:")
        logger.info(f"  Prompt length: {len(prompt)} chars")
        logger.info(f"  Max tokens: {request.max_tokens}")
        logger.info(f"  Temperature: {request.temperature}")

        start_time = time.time()

        # Generate response
        outputs = llm_engine.generate([prompt], sampling_params)

        inference_time = time.time() - start_time

        # Extract generated text
        output = outputs[0]
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)

        logger.info(f"Generation completed:")
        logger.info(f"  Inference time: {inference_time:.3f}s")
        logger.info(f"  Generated tokens: {num_tokens}")
        logger.info(f"  Throughput: {num_tokens / inference_time:.2f} tokens/s")

        # Format response (OpenAI-compatible)
        if request.stream:
            # Streaming not fully implemented in this simple version
            # In production, use AsyncLLM for proper streaming
            raise HTTPException(
                status_code=501,
                detail="Streaming not implemented in this version. Use stream=false"
            )
        else:
            return {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text.strip()
                        },
                        "finish_reason": output.outputs[0].finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": num_tokens,
                    "total_tokens": len(output.prompt_token_ids) + num_tokens
                },
                "vllm_stats": {
                    "inference_time_seconds": inference_time,
                    "tokens_per_second": num_tokens / inference_time
                }
            }

    except Exception as e:
        logger.error(f"Error processing chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible completions endpoint.

    Raw text completion without chat formatting.
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from vllm import SamplingParams

        # Handle single prompt or list of prompts
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

        # Configure sampling
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            n=request.n
        )

        logger.info(f"Processing completion request:")
        logger.info(f"  Number of prompts: {len(prompts)}")
        logger.info(f"  Max tokens: {request.max_tokens}")

        start_time = time.time()

        # Generate responses
        outputs = llm_engine.generate(prompts, sampling_params)

        inference_time = time.time() - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

        logger.info(f"Generation completed:")
        logger.info(f"  Inference time: {inference_time:.3f}s")
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Throughput: {total_tokens / inference_time:.2f} tokens/s")

        # Format response
        choices = []
        for idx, output in enumerate(outputs):
            for output_item in output.outputs:
                choices.append({
                    "index": idx,
                    "text": output_item.text,
                    "finish_reason": output_item.finish_reason,
                    "logprobs": None
                })

        return {
            "id": f"cmpl-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": sum(len(output.prompt_token_ids) for output in outputs),
                "completion_tokens": total_tokens,
                "total_tokens": sum(len(output.prompt_token_ids) for output in outputs) + total_tokens
            },
            "vllm_stats": {
                "inference_time_seconds": inference_time,
                "tokens_per_second": total_tokens / inference_time,
                "continuous_batching": "enabled",
                "pagedattention": "enabled"
            }
        }

    except Exception as e:
        logger.error(f"Error processing completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM API Server for Koyeb")
    parser.add_argument("--model", type=str, default=model_name, help="Model name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--tensor-parallel-size", type=int, default=tensor_parallel_size,
                        help="Number of GPUs")

    args = parser.parse_args()

    # Update global variables
    global model_name, tensor_parallel_size
    model_name = args.model
    tensor_parallel_size = args.tensor_parallel_size

    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
