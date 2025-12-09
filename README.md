

# Local AI Recipes for vLLM

## Table of Contents

1. [About vLLM](#about-vllm)
2. [Hardware-Specific Parameters](#hardware-specific-parameters)
3. [Current hardware available](#current-hardware-available)
4. [Model Configurations](#model-configurations)
   - [GPT-OSS-120B](#gpt-oss-120b)
   - [GPT-OSS-20B](#gpt-oss-20b)
   - [Granite-4.0-H-Tiny](#granite-40-h-tiny)
   - [Whisper](#whisper)
   - [Deepseek OCR](#deepseek-ocr)
5. [Common Configuration Parameters](#common-configuration-parameters)
6. [Contributing](#contributing)

## About this Repo

I'm tired of searching the internet, reddit, github issues, you name it, to try to get models running on my hardware. I'm compiling an list on what work and how, hope this is usefull to someone.
All of the recipes run on `podman` so they should run on `docker` too. I always use AMD ready images from [Docker Hub](https://hub.docker.com/u/rocm).
If you have hardware that works and want to contribute, feel free to do so and create a pull request. For more information, check [Contributing](#contributing).
If you want to see if a specific model can run on the hardware available, create an issue and I'll look into it.

## Current hardware available

- 1 x AMD 9070 XT
- 4 x AMD R9700 AI PRO
- 1 x NVIDIA 5060 TI

## vLLM Hardware-Specific Parameters

This section explains the hardware-specific parameters used in the Docker commands for running vLLM with AMD GPUs.

- `-e HIP_VISIBLE_DEVICES=0,1,2,3`: Specifies which AMD GPUs should be visible to the application. The values are comma-separated indices of GPUs. Adjust this based on the number of GPUs you want to use.

- `-e NCCL_P2P_DISABLE=1`: Disables peer-to-peer (P2P) communication for NCCL (NVIDIA Collective Communications Library). This is sometimes necessary with AMD GPUs when P2P communication causes issues. Setting to `0` increases performance if the combo hardware/model/vllm version supports it

- `-e TORCH_BLAS_PREFER_HIPBLASLT=1`: Forces PyTorch to prefer HIPBLASLT (HIP Basic Linear Algebra Subprograms Library with Tensor Cores) over the standard BLAS library. This can improve performance on AMD GPUs that support tensor cores.

- `-e VLLM_ROCM_USE_AITER=0`: Disables the AITER optimization in vLLM for ROCm (AMD's GPU computing platform). This might be necessary if you encounter compatibility issues with certain models or hardware configurations. Ideally should be `1`, most of the times need to set to `0`.


## Model Configurations

### GPT-OSS-120B

**Hardware:**
- GPU: 4x AMD R9700

**Docker Image:**
- Image: `docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103`

**Command:**
```bash
podman run --ipc=host \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  -v /opt/models/hf:/workspace \
  -e NCCL_DEBUG=WARN \
  -e HUGGINGFACE_HUB_CACHE=/workspace \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e NCCL_P2P_DISABLE=0 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -e VLLM_ROCM_USE_AITER=0 \
  docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103 \
  vllm serve openai/gpt-oss-120b \
  --served-model-name gpt-oss-120b \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser openai
```

### GPT-OSS-20B

**Hardware:**
- GPU: AMD R9700 

**Docker Image:**
- Image: `docker.io/rocm/vllm-dev:open-r9700-08052025`

**Command:**
```bash
podman run --ipc=host \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  -v /opt/models/hf:/workspace \
  -e NCCL_DEBUG=WARN \
  -e HUGGINGFACE_HUB_CACHE=/workspace \
  -e HIP_VISIBLE_DEVICES=0 \
  -e NCCL_P2P_DISABLE=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 \
  docker.io/rocm/vllm-dev:open-r9700-08052025 \
  vllm serve openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser openai
```

### Granite-4.0-H-Tiny

**Hardware:**
- GPU: AMD Radeon RX 9070 XT

**Docker Image:**
- Image: `docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103`

**Command:**
```bash
podman run --ipc=host \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  -v /opt/models/hf:/workspace \
  -e NCCL_DEBUG=WARN \
  -e HUGGINGFACE_HUB_CACHE=/workspace \
  -e HIP_VISIBLE_DEVICES=0 \
  -e NCCL_P2P_DISABLE=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -e VLLM_ROCM_USE_AITER=0 \
  docker.io/rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103 \
  vllm serve ibm/granite-4.0-h-tiny \
  --served-model-name granite-4.0-h-tiny \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
```

### Whisper

**Hardware:**
- GPU: AMD Radeon RX 9070 XT (1 GPU)

**Docker Image:**
- Image: `docker.io/rocm/vllm-dev:nightly`

**Command:**
```bash
podman run --ipc=host \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  -v /opt/models/hf:/workspace \
  -e NCCL_DEBUG=WARN \
  -e HUGGINGFACE_HUB_CACHE=/workspace \
  -e HIP_VISIBLE_DEVICES=0 \
  -e NCCL_P2P_DISABLE=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=1 \
  -e VLLM_ROCM_USE_AITER=1 \
  docker.io/rocm/vllm-dev:nightly \
  /bin/bash -c "pip install vllm[audio] && vllm serve  openai/whisper-large-v3"  
```

### Deepseek OCR

**Hardware:**
- GPU: AMD Radeon RX 9070 XT

**Docker Image:**
- Image: `docker.io/rocm/vllm-dev:nightly`

**Command:**
```bash
 podman run --ipc=host \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  --security-opt seccomp=unconfined \
  -v /opt/models/hf:/workspace \
  -e NCCL_DEBUG=WARN \
  -e HUGGINGFACE_HUB_CACHE=/workspace \
  -e VLLM_ROCM_USE_AITER=0 \
  docker.io/rocm/vllm-dev:nightly \
  vllm serve deepseek-ai/DeepSeek-OCR \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 
```

## Common Configuration Parameters

- `--gpu-memory-utilization`: Percentage of GPU memory to use (0.0-1.0)
- `--max-model-len`: Maximum context length for the model
- `--tensor-parallel-size`: Number of GPUs to split the model across
- `--enable-auto-tool-choice`: Enable automatic tool choice for function calling
- `--tool-call-parser`: Parser to use for tool calling (openai, mistral, etc.)

## Contributing

We welcome contributions from the community! If you have a recipe for a different model or hardware configuration, please submit a pull request with the following information:

1. Model name
2. Hardware used
3. Docker image version
4. Command line to run the model
5. Any special notes about the configuration

Please ensure your contribution follows the format of existing recipes for consistency.

If you can provide a benchmark with it would be great!
