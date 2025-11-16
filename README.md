# SDXL Fine-Tuning for Naruto Art Style

This repository contains a complete implementation for fine-tuning Stable Diffusion XL (SDXL) to generate images in the Naruto anime art style, optimized to run within the 16GB VRAM constraint of Google Colab's free T4 GPU.

## 📋 Table of Contents

- [Overview](#overview)
- [The Challenge](#the-challenge)
- [High-Level Approach](#high-level-approach)
- [Memory Optimization Techniques](#memory-optimization-techniques)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Usage](#usage)
- [Repository Structure](#repository-structure)

##  Overview

The goal of this project is to fine-tune the Stable Diffusion XL base model (`stabilityai/stable-diffusion-xl-base-1.0`) to generate images in the distinct art style of Naruto anime, using the `lambdalabs/naruto-blip-captions` dataset. The entire training and inference process must run within the 16GB VRAM limit of Google Colab's free T4 GPU.

**Base Model**: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)  
**Dataset**: [lambdalabs/naruto-blip-captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)  
**Hardware**: Google Colab T4 GPU (~16GB VRAM)

### 🔗 Quick Links

- **Training Notebook (Colab)**: [Open in Google Colab](https://colab.research.google.com/drive/1PuQhg4BfjGSr5FE35yaAlCI8l10R8ws3?usp=sharing)
- **Inference Notebook (Colab)**: [Open in Google Colab](https://colab.research.google.com/drive/1k9ZWlg9CEaNQA9sJj5W9y0VbmImhKaOl?usp=sharing#scrollTo=FECi5EbShpiX)
- **Model Weights & Outputs (Google Drive)**: [Download from Drive](https://drive.google.com/drive/folders/1wsxr6C6JMRmhtA_PtCKzzb4ypNwYv2H0)

##  The Challenge

Stable Diffusion XL is a massive model with approximately **2.6 billion parameters**. A naive fine-tuning approach would require:

- **Full model training**: ~40-50GB VRAM at native 1024x1024 resolution
- **Standard batch size**: Even with batch size 1, full fine-tuning exceeds 20GB VRAM
- **Memory overhead**: Optimizer states, activations, and gradients multiply memory requirements

**The core challenge**: How do we adapt a 2.6B parameter model to learn a new artistic style within a 16GB VRAM budget?

##  High-Level Approach

Our solution employs a **multi-layered optimization strategy** that combines parameter-efficient fine-tuning with memory-efficient training techniques:

1. **Parameter-Efficient Fine-Tuning**: Use LoRA (Low-Rank Adaptation) to train only ~1.8% of model parameters
2. **Memory-Efficient Training**: Apply multiple techniques to reduce memory footprint during training
3. **Resolution Optimization**: Train at reduced resolution (512x512) while maintaining quality
4. **Smart Checkpointing**: Save progress frequently to prevent data loss

This approach reduces memory requirements from ~40GB to ~12-14GB, making training feasible on free-tier Colab.

##  Memory Optimization Techniques

### 1. LoRA (Low-Rank Adaptation)

**What it is**: LoRA is a parameter-efficient fine-tuning technique that adds trainable low-rank matrices to existing model weights instead of updating all parameters.

**How it works**:
- Original weight matrix `W` (shape: `[d, k]`) is frozen
- Two smaller matrices `A` (shape: `[d, r]`) and `B` (shape: `[r, k]`) are added
- During forward pass: `W' = W + BA`, where `r << min(d, k)` (rank)
- Only `A` and `B` are trained, reducing trainable parameters dramatically

**Memory Impact**:
- **Without LoRA**: 2.6B parameters × 4 bytes (FP32) = ~10.4GB just for weights
- **With LoRA (rank=32)**: ~46M trainable parameters × 4 bytes = ~184MB
- **Savings**: ~99.8% reduction in trainable parameters

**Why this technique**: LoRA has been proven to achieve comparable results to full fine-tuning for style transfer tasks while using a fraction of the memory. For artistic style adaptation, LoRA is particularly effective because style is often captured in attention mechanisms, which LoRA targets.

**Implementation**: Applied to UNet's attention layers (`to_k`, `to_q`, `to_v`, `to_out.0`) with rank=32, alpha=16.

---

### 2. Mixed Precision Training (FP16)

**What it is**: Training with 16-bit floating point precision instead of 32-bit, reducing memory usage by 50%.

**How it works**:
- Forward pass: Use FP16 for most operations
- Loss scaling: Scale loss to prevent underflow in FP16
- Backward pass: Compute gradients in FP16, convert to FP32 for optimizer
- Master weights: Optimizer maintains FP32 copies of weights

**Memory Impact**:
- **FP32**: 2.6B params × 4 bytes = 10.4GB
- **FP16**: 2.6B params × 2 bytes = 5.2GB
- **Savings**: ~5GB for model weights

**Why this technique**: Modern GPUs (including T4) have excellent FP16 performance. For diffusion models, FP16 training is stable and produces high-quality results. The Accelerate library handles mixed precision automatically, ensuring numerical stability.

**Implementation**: Enabled via `Accelerator(mixed_precision="fp16")`. VAE kept in FP32 for encoding stability, text encoders in FP16 (frozen), UNet uses mixed precision automatically.

---

### 3. Gradient Checkpointing

**What it is**: A technique that trades computation for memory by recomputing activations during backpropagation instead of storing them.

**How it works**:
- Forward pass: Store activations only at checkpoint points (e.g., every N layers)
- Backward pass: Recompute activations between checkpoints when needed
- Memory saved: ~50-70% reduction in activation memory

**Memory Impact**:
- **Without checkpointing**: Store all activations = ~8-10GB for SDXL
- **With checkpointing**: Store only checkpoints = ~2-3GB
- **Savings**: ~6-7GB during backpropagation

**Why this technique**: Gradient checkpointing is essential for large models. The computational overhead (~30% slower) is acceptable given the memory constraints. For SDXL, this is the difference between OOM and successful training.

**Implementation**: Enabled via `unet.enable_gradient_checkpointing()` before applying LoRA.

---

### 4. 8-bit AdamW Optimizer

**What it is**: Uses 8-bit quantization for optimizer states (momentum, variance) instead of 32-bit, reducing optimizer memory by 75%.

**How it works**:
- Standard AdamW: Stores FP32 momentum and variance for each parameter (~8 bytes per param)
- 8-bit AdamW: Quantizes optimizer states to 8-bit (~2 bytes per param)
- Maintains training stability through careful quantization

**Memory Impact**:
- **Standard AdamW**: 46M params × 8 bytes = ~368MB
- **8-bit AdamW**: 46M params × 2 bytes = ~92MB
- **Savings**: ~276MB (significant for our tight budget)

**Why this technique**: Optimizer state memory is often overlooked but can be substantial. For LoRA training, 8-bit AdamW provides significant savings with minimal impact on convergence.

**Implementation**: Uses `bitsandbytes.optim.AdamW8bit` for LoRA parameters.

---

### 5. xFormers Memory-Efficient Attention

**What it is**: An optimized attention implementation that reduces memory usage through flash attention and other optimizations.

**How it works**:
- Flash Attention: Computes attention in chunks, reducing peak memory
- Memory-efficient kernels: Optimized CUDA kernels for attention operations
- Reduces memory from O(n²) to O(n) for attention

**Memory Impact**:
- **Standard attention**: ~2-3GB for SDXL UNet
- **xFormers attention**: ~1-1.5GB
- **Savings**: ~1-1.5GB

**Why this technique**: Attention is the memory bottleneck in transformer-based models. xFormers provides both memory savings and speed improvements with no quality loss.

**Implementation**: Enabled via `unet.enable_xformers_memory_efficient_attention()`.

---

### 6. Reduced Training Resolution

**What it is**: Training at 512×512 resolution instead of SDXL's native 1024×1024.

**How it works**:
- Images are resized to 512×512 before encoding
- VAE encodes to 64×64 latents (instead of 128×128)
- UNet processes smaller latent tensors

**Memory Impact**:
- **1024×1024**: Latent size 128×128 = 16,384 elements per channel
- **512×512**: Latent size 64×64 = 4,096 elements per channel
- **Savings**: 4× reduction in spatial dimensions = ~4× less memory for latents and activations

**Why this technique**: Resolution has a quadratic impact on memory. Training at 512×512 is a necessary compromise that still produces high-quality results. The model can generate at higher resolutions during inference.

**Implementation**: Set `resolution = 512` in `TrainingConfig`.

---

### 7. VAE Slicing & Tiling

**What it is**: Processes images in chunks during VAE encoding/decoding to reduce peak memory.

**How it works**:
- **Slicing**: Split batch dimension into smaller chunks
- **Tiling**: Split spatial dimensions into tiles
- Process chunks sequentially instead of all at once

**Memory Impact**:
- **Without slicing**: VAE encoding can use 4-6GB
- **With slicing/tiling**: ~2-3GB peak
- **Savings**: ~2-3GB during encoding

**Why this technique**: VAE encoding is memory-intensive, especially for larger images. Slicing and tiling allow processing of images that wouldn't fit otherwise.

**Implementation**: Enabled via `vae.enable_slicing()` and `vae.enable_tiling()`.

---

### 8. Gradient Accumulation

**What it is**: Accumulate gradients over multiple batches before updating weights, simulating larger batch sizes.

**How it works**:
- Process batch size 1, accumulate gradients over 4 steps
- Effective batch size = 1 × 4 = 4
- Update weights only after accumulating 4 batches

**Memory Impact**:
- **Batch size 4**: Would require ~20GB VRAM
- **Batch size 1 + accumulation**: ~12GB VRAM
- **Savings**: Enables effective batch size 4 within memory limits

**Why this technique**: Larger batch sizes improve training stability and convergence. Gradient accumulation provides the benefits of larger batches without the memory cost.

**Implementation**: Set `gradient_accumulation_steps = 4` in config, handled by Accelerator.

---

##  Combined Memory Impact

| Component | Without Optimizations | With Optimizations | Savings |
|-----------|----------------------|-------------------|---------|
| Model Weights (FP32) | 10.4 GB | 0.2 GB (LoRA) | 10.2 GB |
| Activations | 8-10 GB | 2-3 GB (checkpointing) | 6-7 GB |
| Optimizer States | 0.4 GB | 0.1 GB (8-bit) | 0.3 GB |
| Attention Memory | 2-3 GB | 1-1.5 GB (xFormers) | 1-1.5 GB |
| Latents (512 vs 1024) | 4 GB | 1 GB | 3 GB |
| VAE Encoding | 4-6 GB | 2-3 GB (slicing) | 2-3 GB |
| **Total** | **~30-35 GB** | **~12-14 GB** | **~18-21 GB** |

**Result**: Training fits comfortably within 16GB VRAM with headroom for system overhead.

---

## 🔧 Implementation Details

### Training Configuration

```python
class TrainingConfig:
    # Model
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Dataset
    train_data_dir = "/content/naruto_train"
    resolution = 512  # Reduced from 1024
    
    # Training
    num_epochs = 5
    train_batch_size = 1
    gradient_accumulation_steps = 4  
    learning_rate = 1e-5
    mixed_precision = "fp16"
    
    # LoRA
    lora_rank = 32
    lora_alpha = 16
    lora_dropout = 0.1
    
    # Memory optimizations
    use_xformers = True
    gradient_checkpointing = True
    use_8bit_adam = True
    
    # SDXL-specific
    vae_scale_factor = 0.13025  # Critical for SDXL
```

### Key Implementation Choices

1. **Dtype Management**:
   - VAE: FP32 (encoding requires precision)
   - Text Encoders: FP16 (frozen, safe to quantize)
   - UNet: FP32 loaded, mixed precision handled by Accelerator

2. **LoRA Target Modules**:
   - Applied to attention layers: `to_k`, `to_q`, `to_v`, `to_out.0`
   - These layers capture style information effectively

3. **SDXL-Specific Handling**:
   - Dual text encoders with concatenated embeddings
   - Time embeddings for resolution conditioning
   - Correct VAE scaling factor (0.13025, not 0.18215)

4. **Error Handling**:
   - NaN detection and batch skipping
   - Gradient clipping (max_norm=1.0)
   - Frequent checkpointing to prevent data loss

---

##  Results

### Training Metrics

- **Trainable Parameters**: 46,448,640 (1.78% of total)
- **Training Time**: ~30-40 minutes per epoch on T4 GPU
- **Memory Usage**: ~12-14GB VRAM (within 16GB limit)
- **Final Loss**: Typically converges to 0.005-0.008 range

### Model Performance

The fine-tuned model successfully learns the Naruto art style and can generate:
- Characters in Naruto anime style
- Consistent artistic rendering
- Style transfer to new prompts (e.g., "Bill Gates in Naruto style")

### Visual Comparison: Base Model vs Fine-Tuned Model

The following comparison demonstrates the effectiveness of our fine-tuning approach. The fine-tuned model consistently generates images that strongly adhere to the Naruto anime art style, including accurate character design, facial features, clothing, and overall aesthetic.

**Test Prompts Used:**
1. `"Naruto Uzumaki eating ramen"`
2. `"Bill Gates in Naruto style"`
3. `"A boy with blue eyes in Naruto style"`

![Comparison Results](download.png)

The fine-tuned model demonstrates successful style adaptation, producing coherent and stylistically accurate images that capture the distinct visual characteristics of the Naruto anime series.

See `inference.ipynb` for more comparison examples and inference code.

---

##  Usage

### Training

**Option 1: Use Colab Notebook (Recommended)**
- **[Open Training Notebook in Google Colab](https://colab.research.google.com/drive/1PuQhg4BfjGSr5FE35yaAlCI8l10R8ws3?usp=sharing)**
- Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU: T4)
- Run all cells sequentially
- Training will save checkpoints to Google Drive automatically

**Option 2: Local Setup**
1. Clone this repository
2. Open `train.ipynb` in Jupyter or Google Colab
3. Follow the same steps as above

### Inference

**Option 1: Use Colab Notebook (Recommended)**
- **[Open Inference Notebook in Google Colab](https://colab.research.google.com/drive/1k9ZWlg9CEaNQA9sJj5W9y0VbmImhKaOl?usp=sharing#scrollTo=FECi5EbShpiX)**
- Load your trained LoRA weights (or download from Drive)
- Generate images with test prompts
- Compare with base SDXL outputs

**Option 2: Use Pre-trained Weights**
- **[Download Model Weights from Google Drive](https://drive.google.com/drive/folders/1wsxr6C6JMRmhtA_PtCKzzb4ypNwYv2H0)**
- The Drive folder contains:
  - `final_lora/` - Final trained LoRA weights
  - `checkpoint-500/`, `checkpoint-1000/`, etc. - Intermediate checkpoints
  - `epoch-2/`, `epoch-4/` - Epoch checkpoints
  - `test_0.png`, `test_1.png`, `test_2.png` - Sample generated images

### Loading Trained Model

```python
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
import torch

# Load base pipeline
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# Load LoRA weights
pipeline.unet = PeftModel.from_pretrained(
    pipeline.unet,
    "/path/to/final_lora"
)

# Generate
prompt = "Naruto Uzumaki eating ramen, anime style"
image = pipeline(prompt, num_inference_steps=30).images[0]
image.save("output.png")
```

---

## 📁 Repository Structure

```
listen2ti/
├── train.ipynb              # Main training notebook
├── inference.ipynb          # Inference and comparison 
├── README.md                
└── download.png             # Comparison results image
```

###  Model Weights & Outputs

All trained model weights, checkpoints, and generated images are available on Google Drive:

**[📁 Download from Google Drive](https://drive.google.com/drive/folders/1wsxr6C6JMRmhtA_PtCKzzb4ypNwYv2H0)**

The Drive folder contains:
- **`final_lora/`** - Final trained LoRA weights (recommended for inference)
- **`checkpoint-500/`**, **`checkpoint-1000/`**, **`checkpoint-1500/`** - Intermediate training checkpoints
- **`epoch-2/`**, **`epoch-4/`** - Epoch-based checkpoints
- **`test_0.png`**, **`test_1.png`**, **`test_2.png`** - Sample generated images from test prompts