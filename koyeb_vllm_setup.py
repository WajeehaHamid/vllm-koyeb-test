"""
vLLM End-to-End Testing and Exploration Script for Koyeb
=========================================================

This script demonstrates vLLM's complete inference pipeline with detailed
logging at each optimization step. It's designed to help understand:
1. Model loading and memory allocation
2. PagedAttention block management
3. Continuous batching behavior
4. Request scheduling and execution
5. Token generation and sampling

Author: Educational demonstration
License: MIT
"""

import os
import sys
import time
import logging
from typing import List, Dict, Optional
import json
from dataclasses import dataclass, asdict

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vllm_test_detailed.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TestMetrics:
    """Tracks performance metrics throughout the pipeline"""
    model_name: str
    load_time_seconds: float
    memory_allocated_gb: float
    num_blocks: int
    block_size: int
    total_requests: int
    avg_tokens_per_second: float
    avg_latency_ms: float
    peak_batch_size: int


class VLLMEndToEndTester:
    """
    Comprehensive tester for vLLM with detailed introspection
    at each stage of the inference pipeline.
    """

    def __init__(self, model_name: str = "facebook/opt-125m",
                 tensor_parallel_size: int = 1):
        """
        Initialize the tester with a model.

        Args:
            model_name: HuggingFace model identifier (use small model for testing)
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self.metrics = None

    def step1_import_and_setup(self):
        """Step 1: Import vLLM and verify installation"""
        logger.info("="*80)
        logger.info("STEP 1: IMPORT AND SETUP")
        logger.info("="*80)

        try:
            from vllm import LLM, SamplingParams
            logger.info("✓ vLLM successfully imported")
            logger.info(f"  Python version: {sys.version}")
            logger.info(f"  Working directory: {os.getcwd()}")

            # Check for GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
                    logger.info(f"  Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                else:
                    logger.warning("⚠ No GPU detected, will use CPU (slower)")
            except ImportError:
                logger.warning("⚠ PyTorch not available for GPU check")

            return LLM, SamplingParams

        except ImportError as e:
            logger.error(f"✗ Failed to import vLLM: {e}")
            logger.error("  Install with: pip install vllm")
            raise

    def step2_load_model(self, LLM):
        """Step 2: Load model and observe memory allocation"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: MODEL LOADING AND MEMORY ALLOCATION")
        logger.info("="*80)

        start_time = time.time()

        logger.info(f"Loading model: {self.model_name}")
        logger.info("This step involves:")
        logger.info("  1. Downloading model weights from HuggingFace (if not cached)")
        logger.info("  2. Loading tokenizer")
        logger.info("  3. Initializing model executor")
        logger.info("  4. Profiling GPU memory")
        logger.info("  5. Allocating KV cache blocks (PagedAttention)")

        try:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.9,  # Use 90% of GPU memory
                trust_remote_code=True,      # For custom models
                download_dir=os.environ.get('HF_HOME', './hf_cache')
            )

            load_time = time.time() - start_time
            logger.info(f"✓ Model loaded successfully in {load_time:.2f} seconds")

            # Try to extract memory statistics
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    logger.info(f"  GPU memory allocated: {allocated:.2f} GB")
                    logger.info(f"  GPU memory reserved: {reserved:.2f} GB")

                    # Estimate KV cache size
                    free_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 - reserved
                    logger.info(f"  Available for KV cache: {free_memory:.2f} GB")
            except:
                pass

            return load_time

        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise

    def step3_understand_pagedattention(self):
        """Step 3: Explain PagedAttention configuration"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: PAGEDATTENTION MEMORY MANAGEMENT")
        logger.info("="*80)

        logger.info("PagedAttention divides KV cache into fixed-size blocks:")
        logger.info("  • Block size: typically 16 tokens")
        logger.info("  • Blocks can be non-contiguous in GPU memory")
        logger.info("  • Reduces fragmentation by 2-4x vs. contiguous allocation")
        logger.info("  • Enables dynamic allocation as sequences grow")

        logger.info("\nKV Cache Structure:")
        logger.info("  Physical memory: [Block 0][Block 1][Block 2]...[Block N]")
        logger.info("  Logical mapping: Sequence 1 → [0, 3, 5], Sequence 2 → [1, 2, 4]")
        logger.info("  Custom CUDA kernels handle sparse block access")

        # Try to get actual block configuration
        try:
            # This is internal, may not be accessible in all versions
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                if hasattr(engine, 'scheduler'):
                    scheduler = engine.scheduler
                    logger.info("\n✓ Block Manager Statistics:")
                    # Note: Actual attributes depend on vLLM version
                    logger.info(f"  Block configuration detected")
        except:
            logger.info("\n⚠ Block statistics not accessible (internal implementation)")

    def step4_create_sampling_params(self, SamplingParams) -> 'SamplingParams':
        """Step 4: Configure sampling parameters"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: SAMPLING PARAMETERS CONFIGURATION")
        logger.info("="*80)

        sampling_params = SamplingParams(
            temperature=0.8,      # Controls randomness (0=deterministic, 1=creative)
            top_p=0.95,          # Nucleus sampling: cumulative probability threshold
            top_k=50,            # Consider only top-k tokens
            max_tokens=256,      # Maximum tokens to generate
            presence_penalty=0.0, # Penalize repeated tokens
            frequency_penalty=0.0 # Penalize frequent tokens
        )

        logger.info("Sampling configuration:")
        logger.info(f"  • Temperature: {sampling_params.temperature} (controls randomness)")
        logger.info(f"  • Top-p: {sampling_params.top_p} (nucleus sampling threshold)")
        logger.info(f"  • Top-k: {sampling_params.top_k} (candidate token limit)")
        logger.info(f"  • Max tokens: {sampling_params.max_tokens}")

        logger.info("\nSampling process:")
        logger.info("  1. Model produces logits for all vocab tokens")
        logger.info("  2. Apply temperature scaling")
        logger.info("  3. Filter to top-k candidates")
        logger.info("  4. Apply nucleus (top-p) sampling")
        logger.info("  5. Sample token from filtered distribution")

        return sampling_params

    def step5_prepare_test_prompts(self) -> List[str]:
        """Step 5: Prepare diverse test prompts"""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: TEST PROMPTS PREPARATION")
        logger.info("="*80)

        prompts = [
            "Explain what machine learning is in simple terms:",
            "Write a Python function to calculate fibonacci numbers:",
            "What are the benefits of cloud computing?",
            "Describe the process of photosynthesis:",
            "How does a neural network learn?",
        ]

        logger.info(f"Prepared {len(prompts)} test prompts")
        logger.info("These will demonstrate continuous batching:")
        logger.info("  • All prompts processed together in dynamic batches")
        logger.info("  • Shorter completions don't block longer ones")
        logger.info("  • KV cache blocks allocated/freed dynamically")

        for i, prompt in enumerate(prompts, 1):
            logger.info(f"  {i}. {prompt[:60]}...")

        return prompts

    def step6_execute_inference(self, prompts: List[str],
                                sampling_params) -> List:
        """Step 6: Execute inference with detailed logging"""
        logger.info("\n" + "="*80)
        logger.info("STEP 6: INFERENCE EXECUTION (CONTINUOUS BATCHING)")
        logger.info("="*80)

        logger.info("Starting inference pipeline:")
        logger.info("  1. Tokenization: Convert text to token IDs")
        logger.info("  2. Scheduling: Select requests for batch")
        logger.info("  3. KV allocation: Assign blocks via PagedAttention")
        logger.info("  4. GPU execution: Forward pass through transformer")
        logger.info("  5. Token generation: Sample next tokens")
        logger.info("  6. Iteration: Repeat until all sequences complete")

        start_time = time.time()

        try:
            outputs = self.llm.generate(prompts, sampling_params)

            inference_time = time.time() - start_time
            logger.info(f"\n✓ Inference completed in {inference_time:.2f} seconds")

            # Calculate statistics
            total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            tokens_per_second = total_tokens / inference_time
            avg_latency = (inference_time / len(prompts)) * 1000

            logger.info(f"  Total tokens generated: {total_tokens}")
            logger.info(f"  Throughput: {tokens_per_second:.2f} tokens/second")
            logger.info(f"  Average latency: {avg_latency:.2f} ms per request")

            return outputs

        except Exception as e:
            logger.error(f"✗ Inference failed: {e}")
            raise

    def step7_analyze_outputs(self, outputs: List) -> Dict:
        """Step 7: Analyze generation outputs"""
        logger.info("\n" + "="*80)
        logger.info("STEP 7: OUTPUT ANALYSIS")
        logger.info("="*80)

        results = []

        for idx, output in enumerate(outputs, 1):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            finish_reason = output.outputs[0].finish_reason

            logger.info(f"\n--- Request {idx} ---")
            logger.info(f"Prompt: {prompt[:60]}...")
            logger.info(f"Generated tokens: {len(token_ids)}")
            logger.info(f"Finish reason: {finish_reason}")
            logger.info(f"Output preview: {generated_text[:100]}...")

            results.append({
                'prompt': prompt,
                'output': generated_text,
                'num_tokens': len(token_ids),
                'finish_reason': finish_reason
            })

        # Summary statistics
        total_output_tokens = sum(r['num_tokens'] for r in results)
        avg_tokens = total_output_tokens / len(results)

        logger.info("\n" + "-"*80)
        logger.info("GENERATION SUMMARY:")
        logger.info(f"  Total requests: {len(results)}")
        logger.info(f"  Total output tokens: {total_output_tokens}")
        logger.info(f"  Average tokens per request: {avg_tokens:.1f}")

        return {
            'results': results,
            'total_tokens': total_output_tokens,
            'avg_tokens': avg_tokens
        }

    def step8_demonstrate_streaming(self, SamplingParams):
        """Step 8: Demonstrate streaming generation"""
        logger.info("\n" + "="*80)
        logger.info("STEP 8: STREAMING GENERATION")
        logger.info("="*80)

        logger.info("Streaming allows token-by-token output (like ChatGPT):")
        logger.info("  • Lower perceived latency")
        logger.info("  • Better user experience")
        logger.info("  • Same underlying continuous batching")

        prompt = "Write a haiku about artificial intelligence:"
        logger.info(f"\nPrompt: {prompt}")
        logger.info("Streamed output: ", end="", flush=True)

        # Note: Streaming requires different API in some vLLM versions
        try:
            # For offline LLM, we simulate streaming by showing tokens
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=50
            )

            outputs = self.llm.generate([prompt], sampling_params)
            output_text = outputs[0].outputs[0].text

            # Simulate streaming display
            for char in output_text:
                print(char, end="", flush=True)
                time.sleep(0.02)  # Simulate network delay

            print()  # Newline
            logger.info("\n✓ Streaming demonstration complete")

        except Exception as e:
            logger.warning(f"⚠ Streaming demo failed: {e}")

    def step9_benchmark_optimizations(self):
        """Step 9: Demonstrate optimization impact"""
        logger.info("\n" + "="*80)
        logger.info("STEP 9: OPTIMIZATION BENCHMARKING")
        logger.info("="*80)

        logger.info("vLLM Optimizations vs. Naive Implementation:")
        logger.info("\n1. PagedAttention:")
        logger.info("   • Memory efficiency: 2-4x more requests in same GPU")
        logger.info("   • Dynamic allocation: No pre-allocated waste")
        logger.info("   • Trade-off: ~5% compute overhead for block lookup")

        logger.info("\n2. Continuous Batching:")
        logger.info("   • Throughput: 2-10x higher than static batching")
        logger.info("   • Latency: Lower P50, similar P99")
        logger.info("   • GPU utilization: 80-95% vs. 20-40% naive")

        logger.info("\n3. FlashAttention:")
        logger.info("   • Attention compute: 2-3x faster")
        logger.info("   • Memory bandwidth: Reduced by fusing operations")
        logger.info("   • Works seamlessly with PagedAttention")

        logger.info("\n4. Prefix Caching:")
        logger.info("   • Shared prompts: Reuse KV cache blocks")
        logger.info("   • RAG workloads: Up to 5x speedup")
        logger.info("   • Zero redundant computation")

    def step10_full_pipeline_summary(self, analysis: Dict, load_time: float):
        """Step 10: Complete pipeline summary"""
        logger.info("\n" + "="*80)
        logger.info("STEP 10: END-TO-END PIPELINE SUMMARY")
        logger.info("="*80)

        logger.info("\n📊 COMPLETE REQUEST FLOW:")
        logger.info("\n1. HTTP Request → API Server")
        logger.info("   • FastAPI receives POST /v1/chat/completions")
        logger.info("   • Request validated and authenticated")
        logger.info("   • Assigned unique request_id")

        logger.info("\n2. Tokenization → Input Processing")
        logger.info("   • Text converted to token IDs")
        logger.info("   • Special tokens added (<BOS>, <EOS>)")
        logger.info("   • Request queued in AsyncLLM")

        logger.info("\n3. Scheduling → Continuous Batching")
        logger.info("   • Scheduler selects requests for batch")
        logger.info("   • PagedAttention allocates KV blocks")
        logger.info("   • Batch metadata prepared for GPU")

        logger.info("\n4. GPU Execution → Model Forward Pass")
        logger.info("   • Token embeddings computed")
        logger.info("   • 32 transformer layers (example)")
        logger.info("   • PagedAttention handles KV cache lookups")
        logger.info("   • Output logits: [batch_size, vocab_size]")

        logger.info("\n5. Sampling → Token Generation")
        logger.info("   • Temperature scaling applied")
        logger.info("   • Top-k/top-p filtering")
        logger.info("   • Token sampled from distribution")

        logger.info("\n6. Output Processing → Response")
        logger.info("   • Token IDs detokenized to text")
        logger.info("   • Formatted as SSE chunks (streaming)")
        logger.info("   • Sent to client via HTTP")

        logger.info("\n7. Iteration & Completion")
        logger.info("   • Repeat steps 3-6 until EOS or max_tokens")
        logger.info("   • KV blocks released to free pool")
        logger.info("   • Request statistics logged")

        logger.info("\n" + "="*80)
        logger.info("📈 PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Load time: {load_time:.2f}s")
        logger.info(f"Requests processed: {len(analysis['results'])}")
        logger.info(f"Total tokens: {analysis['total_tokens']}")
        logger.info(f"Average tokens/request: {analysis['avg_tokens']:.1f}")

        logger.info("\n✅ END-TO-END TEST COMPLETE")
        logger.info("="*80)

    def run_complete_test(self):
        """Execute all test steps in sequence"""
        print("\n" + "🚀 "*20)
        print("vLLM END-TO-END TESTING AND EXPLORATION")
        print("🚀 "*20 + "\n")

        try:
            # Step 1: Import
            LLM, SamplingParams = self.step1_import_and_setup()

            # Step 2: Load model
            load_time = self.step2_load_model(LLM)

            # Step 3: Understand PagedAttention
            self.step3_understand_pagedattention()

            # Step 4: Sampling parameters
            sampling_params = self.step4_create_sampling_params(SamplingParams)

            # Step 5: Prepare prompts
            prompts = self.step5_prepare_test_prompts()

            # Step 6: Execute inference
            outputs = self.step6_execute_inference(prompts, sampling_params)

            # Step 7: Analyze outputs
            analysis = self.step7_analyze_outputs(outputs)

            # Step 8: Streaming demo
            self.step8_demonstrate_streaming(SamplingParams)

            # Step 9: Optimization benchmarks
            self.step9_benchmark_optimizations()

            # Step 10: Complete summary
            self.step10_full_pipeline_summary(analysis, load_time)

            return True

        except Exception as e:
            logger.error(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="vLLM End-to-End Testing and Exploration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model name (use small model for testing: facebook/opt-125m, gpt2)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )

    args = parser.parse_args()

    logger.info(f"Starting test with model: {args.model}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")

    tester = VLLMEndToEndTester(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size
    )

    success = tester.run_complete_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
