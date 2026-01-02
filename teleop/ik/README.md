# IK Solver Testing Guide (DGX Spark)

This guide explains how to run the Inverse Kinematics (IK) test suite on high-performance compute environments like the **DGX Spark**.

## Prerequisites

1. **JAX with GPU Support:**
   On DGX (NVIDIA GPU), ensure you have the CUDA-enabled version of JAX:
   ```bash
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. **Dependencies:**
   ```bash
   pip install urdf-parser-py
   ```

## Running the Test

Run the pre-configured test script from the repository root:

```bash
python scripts/test_ik.py
```

### Options for DGX

- **Force CPU (for debugging):**
  If you encounter GPU driver issues, you can force JAX to use the CPU:
  ```bash
  JAX_PLATFORMS=cpu python scripts/test_ik.py
  ```

- **Enable High Precision:**
  JAX defaults to float32. For higher precision (if needed):
  ```bash
  export JAX_ENABLE_X64=True
  python scripts/test_ik.py
  ```

## What the Test Benchmarks

- **Robot Model Loading:** Parses the H1 Inspire URDF.
- **FK Throughput:** Measures how many forward kinematics solutions can be computed per second.
- **IK Latency:** Measures the time to reach a target pose from a cold start (with JIT compilation) and from a warm start (teleop simulation).
- **GPU Utilization:** On DGX, JAX will automatically JIT-compile the math to XLA/CUDA kernels for massive parallel speedup.

## Expected Performance on DGX

On a DGX A100/H100, you should expect:
- **Cold start (JIT):** ~1-2 seconds.
- **Warm start (Teleop):** **< 1ms** per solve (1000+ Hz).
