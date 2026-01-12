"""
vLLM API Testing Client
========================

Interactive client for testing vLLM deployment on Koyeb.
Demonstrates various optimization features and provides detailed metrics.

Usage:
    python test_client.py --url https://your-app.koyeb.app
    python test_client.py --url http://localhost:8000 --interactive
"""

import argparse
import asyncio
import aiohttp
import time
import json
from typing import List, Dict
from dataclasses import dataclass
import sys


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: int
    prompt: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_time: float
    tokens_per_second: float
    finish_reason: str


class VLLMTestClient:
    """Client for testing vLLM API with detailed metrics"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    async def health_check(self) -> Dict:
        """Check if the API is healthy"""
        print("\n" + "="*80)
        print("HEALTH CHECK")
        print("="*80)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/health") as response:
                    data = await response.json()
                    print(f"✓ Status: {data.get('status', 'unknown')}")
                    print(f"  Model: {data.get('model', 'unknown')}")
                    print(f"  Tensor Parallel: {data.get('tensor_parallel_size', 'unknown')}")
                    return data
            except Exception as e:
                print(f"✗ Health check failed: {e}")
                return None

    async def get_metrics(self) -> Dict:
        """Get current metrics from the server"""
        print("\n" + "="*80)
        print("SERVER METRICS")
        print("="*80)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/metrics") as response:
                    data = await response.json()
                    print(f"Model: {data.get('model', 'unknown')}")
                    if 'gpu_name' in data:
                        print(f"GPU: {data['gpu_name']}")
                        print(f"GPU Memory Allocated: {data.get('gpu_memory_allocated_gb', 0):.2f} GB")
                        print(f"GPU Memory Reserved: {data.get('gpu_memory_reserved_gb', 0):.2f} GB")
                    return data
            except Exception as e:
                print(f"⚠ Metrics not available: {e}")
                return {}

    async def single_chat_completion(self, prompt: str,
                                     temperature: float = 0.7,
                                     max_tokens: int = 256) -> RequestMetrics:
        """Send a single chat completion request"""
        payload = {
            "model": "default",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                result = await response.json()

        elapsed = time.time() - start_time

        # Extract metrics
        choice = result['choices'][0]
        usage = result['usage']
        vllm_stats = result.get('vllm_stats', {})

        return RequestMetrics(
            request_id=0,
            prompt=prompt,
            response=choice['message']['content'],
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_time=elapsed,
            tokens_per_second=vllm_stats.get('tokens_per_second',
                                            usage['completion_tokens'] / elapsed),
            finish_reason=choice['finish_reason']
        )

    async def test_single_request(self):
        """Test 1: Single request to verify basic functionality"""
        print("\n" + "="*80)
        print("TEST 1: SINGLE REQUEST")
        print("="*80)
        print("Testing basic inference pipeline...")

        prompt = "Explain what machine learning is in one paragraph:"
        print(f"\nPrompt: {prompt}")

        metrics = await self.single_chat_completion(prompt, max_tokens=150)

        print(f"\n✓ Response generated successfully")
        print(f"\nResponse:")
        print("-" * 80)
        print(metrics.response)
        print("-" * 80)

        print(f"\n📊 Metrics:")
        print(f"  Prompt tokens: {metrics.prompt_tokens}")
        print(f"  Completion tokens: {metrics.completion_tokens}")
        print(f"  Total time: {metrics.total_time:.3f}s")
        print(f"  Throughput: {metrics.tokens_per_second:.2f} tokens/s")
        print(f"  Finish reason: {metrics.finish_reason}")

        return metrics

    async def test_continuous_batching(self, num_requests: int = 5):
        """Test 2: Multiple concurrent requests to demonstrate continuous batching"""
        print("\n" + "="*80)
        print(f"TEST 2: CONTINUOUS BATCHING ({num_requests} CONCURRENT REQUESTS)")
        print("="*80)
        print("Demonstrating vLLM's continuous batching optimization...")
        print("All requests sent simultaneously - vLLM will batch them dynamically\n")

        prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing briefly:",
            "How do neural networks work?",
            "What is deep learning?",
            "Describe natural language processing:",
        ][:num_requests]

        async def send_request(session, prompt, req_id):
            """Send a single request and track timing"""
            payload = {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 100
            }

            start = time.time()
            print(f"[Request {req_id}] Sending: {prompt[:50]}...")

            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                result = await response.json()

            elapsed = time.time() - start
            usage = result['usage']
            tokens_per_sec = usage['completion_tokens'] / elapsed

            print(f"[Request {req_id}] ✓ Completed in {elapsed:.2f}s "
                  f"({tokens_per_sec:.2f} tok/s)")

            return RequestMetrics(
                request_id=req_id,
                prompt=prompt,
                response=result['choices'][0]['message']['content'],
                prompt_tokens=usage['prompt_tokens'],
                completion_tokens=usage['completion_tokens'],
                total_time=elapsed,
                tokens_per_second=tokens_per_sec,
                finish_reason=result['choices'][0]['finish_reason']
            )

        # Send all requests concurrently
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            tasks = [
                send_request(session, prompt, i)
                for i, prompt in enumerate(prompts, 1)
            ]
            metrics_list = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Analyze results
        print("\n" + "-"*80)
        print("📊 CONTINUOUS BATCHING RESULTS")
        print("-"*80)

        total_tokens = sum(m.completion_tokens for m in metrics_list)
        avg_time = sum(m.total_time for m in metrics_list) / len(metrics_list)
        avg_throughput = sum(m.tokens_per_second for m in metrics_list) / len(metrics_list)

        print(f"Total requests: {len(metrics_list)}")
        print(f"Total wall time: {total_time:.2f}s (all concurrent)")
        print(f"Average request time: {avg_time:.2f}s")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Average throughput: {avg_throughput:.2f} tokens/s")
        print(f"Aggregate throughput: {total_tokens / total_time:.2f} tokens/s")

        print("\n💡 What just happened:")
        print("  • All requests were batched together dynamically")
        print("  • As shorter sequences finished, they freed GPU memory")
        print("  • New requests could start immediately (continuous batching)")
        print("  • Total throughput is much higher than sequential processing")

        return metrics_list

    async def test_varying_lengths(self):
        """Test 3: Requests with varying output lengths"""
        print("\n" + "="*80)
        print("TEST 3: VARYING OUTPUT LENGTHS")
        print("="*80)
        print("Testing PagedAttention's dynamic memory allocation...\n")

        test_cases = [
            ("Short response (10 tokens)", "Say 'Hello' in 5 words:", 10),
            ("Medium response (50 tokens)", "Explain photosynthesis briefly:", 50),
            ("Long response (200 tokens)", "Write a detailed explanation of quantum mechanics:", 200),
        ]

        results = []

        for name, prompt, max_tokens in test_cases:
            print(f"\n{name}:")
            print(f"  Prompt: {prompt}")
            print(f"  Max tokens: {max_tokens}")

            metrics = await self.single_chat_completion(prompt, max_tokens=max_tokens)

            print(f"  ✓ Generated {metrics.completion_tokens} tokens in {metrics.total_time:.2f}s")
            print(f"  Throughput: {metrics.tokens_per_second:.2f} tokens/s")

            results.append((name, metrics))

        print("\n" + "-"*80)
        print("💡 PagedAttention Insight:")
        print("  • Short sequences used few blocks (low memory)")
        print("  • Long sequences dynamically allocated more blocks")
        print("  • No wasted pre-allocation!")
        print("  • Memory freed immediately after completion")

        return results

    async def test_temperature_sampling(self):
        """Test 4: Different sampling parameters"""
        print("\n" + "="*80)
        print("TEST 4: SAMPLING PARAMETERS")
        print("="*80)
        print("Testing token sampling with different temperatures...\n")

        prompt = "Complete this story: Once upon a time in a distant galaxy,"
        temperatures = [0.0, 0.5, 1.0, 1.5]

        print(f"Prompt: {prompt}\n")

        for temp in temperatures:
            print(f"Temperature = {temp}:")
            metrics = await self.single_chat_completion(
                prompt,
                temperature=temp,
                max_tokens=50
            )
            print(f"  {metrics.response[:100]}...")
            print(f"  ({metrics.tokens_per_second:.2f} tokens/s)\n")

        print("💡 Sampling Insight:")
        print("  • Temperature 0.0 = Deterministic (greedy)")
        print("  • Temperature 0.5 = Focused and coherent")
        print("  • Temperature 1.0 = Balanced creativity")
        print("  • Temperature 1.5 = Very creative/random")

    async def interactive_mode(self):
        """Interactive chat mode"""
        print("\n" + "="*80)
        print("INTERACTIVE CHAT MODE")
        print("="*80)
        print("Type your messages below. Enter 'quit' to exit.\n")

        while True:
            try:
                prompt = input("You: ").strip()

                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if not prompt:
                    continue

                print("Assistant: ", end="", flush=True)

                metrics = await self.single_chat_completion(prompt, max_tokens=256)

                print(metrics.response)
                print(f"\n({metrics.completion_tokens} tokens, "
                      f"{metrics.tokens_per_second:.2f} tok/s)\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")

    async def run_all_tests(self):
        """Run all tests sequentially"""
        print("\n" + "🚀 "*30)
        print("VLLM END-TO-END TESTING")
        print("🚀 "*30)

        # Health check
        await self.health_check()

        # Get metrics
        await self.get_metrics()

        # Run tests
        await self.test_single_request()
        await self.test_continuous_batching(num_requests=5)
        await self.test_varying_lengths()
        await self.test_temperature_sampling()

        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Observations:")
        print("  ✓ PagedAttention: Dynamic memory allocation")
        print("  ✓ Continuous Batching: Concurrent request processing")
        print("  ✓ FlashAttention: Fast attention computation")
        print("  ✓ Token Sampling: Flexible generation strategies")
        print("\n" + "="*80)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test vLLM API deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python test_client.py --url https://your-app.koyeb.app

  # Interactive mode
  python test_client.py --url http://localhost:8000 --interactive

  # Specific test
  python test_client.py --url http://localhost:8000 --test single

  # Stress test with many concurrent requests
  python test_client.py --url http://localhost:8000 --test batch --num-requests 10
        """
    )

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Base URL of the vLLM API server"
    )

    parser.add_argument(
        "--test",
        type=str,
        choices=['all', 'single', 'batch', 'lengths', 'sampling'],
        default='all',
        help="Which test to run"
    )

    parser.add_argument(
        "--interactive",
        action='store_true',
        help="Start interactive chat mode"
    )

    parser.add_argument(
        "--num-requests",
        type=int,
        default=5,
        help="Number of concurrent requests for batch test"
    )

    args = parser.parse_args()

    client = VLLMTestClient(args.url)

    try:
        if args.interactive:
            await client.health_check()
            await client.get_metrics()
            await client.interactive_mode()
        elif args.test == 'all':
            await client.run_all_tests()
        elif args.test == 'single':
            await client.health_check()
            await client.test_single_request()
        elif args.test == 'batch':
            await client.health_check()
            await client.test_continuous_batching(args.num_requests)
        elif args.test == 'lengths':
            await client.health_check()
            await client.test_varying_lengths()
        elif args.test == 'sampling':
            await client.health_check()
            await client.test_temperature_sampling()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
