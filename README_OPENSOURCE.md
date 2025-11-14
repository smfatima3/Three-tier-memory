# WebArena Evaluation - Open Source Models

Complete evaluation script for ICLR paper using **open-source LLMs** instead of proprietary APIs.

## Models Evaluated

### Open Source Baselines
- **Llama 3.1 8B Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)
- **Gemma 2 9B Instruct** (`google/gemma-2-9b-it`)
- **Qwen 2.5 7B Instruct** (`Qwen/Qwen2.5-7B-Instruct`)
- **Phi-3 Mini 4K** (`microsoft/Phi-3-mini-4k-instruct`)

### Your Proposed Method
- **LCA** (Layered Contextual Alignment) - 5 coordinated browser agents

## Features

✅ **Open Source Models**: HuggingFace Transformers with 8-bit quantization
✅ **Google Colab Optimized**: Auto-setup, GPU support, memory efficient
✅ **Real Browser Automation**: Selenium WebDriver for LCA
✅ **Statistical Validity**: 10 trials per task, Cohen's d effect sizes
✅ **NO Hardcoded Data**: All results from actual execution

## Quick Start (Google Colab)

### Option 1: Jupyter Notebook (Easiest)

1. **Open in Colab**: Upload `webarena_evaluation_opensource.ipynb`
2. **Set GPU Runtime**: Runtime → Change runtime type → GPU
3. **Upload Tasks**: Upload `webarena_task.json`
4. **Run All Cells**: Runtime → Run all

### Option 2: Python Script

```python
# In Colab cell:
!wget https://raw.githubusercontent.com/YOUR_USERNAME/Three-tier-memory/main/webarena_evaluation_opensource.py

# Upload your webarena_task.json
from google.colab import files
files.upload()

# Run evaluation
!python webarena_evaluation_opensource.py
```

## Local Setup (Non-Colab)

### Requirements

```bash
# Python 3.10+
pip install torch transformers accelerate bitsandbytes
pip install selenium pandas scipy numpy
```

### GPU Recommended

Models will run on CPU but much slower. GPU with 16GB+ VRAM recommended.

### Chrome Installation

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install chromium-chromedriver
```

**macOS:**
```bash
brew install chromedriver
```

### Run Evaluation

```bash
python webarena_evaluation_opensource.py
```

## Configuration

Edit `main()` function in the script:

```python
TASK_FILE = "webarena_task.json"
N_TRIALS = 10        # Trials per task (10 for statistical validity)
MAX_TASKS = 50       # Number of tasks to evaluate
n_agents = 5         # Number of LCA browser agents
```

For quick testing:
```python
N_TRIALS = 3         # Fewer trials
MAX_TASKS = 5        # Fewer tasks
n_agents = 3         # Fewer LCA agents
```

## What Gets Evaluated

Each agent is evaluated on:
- **50 tasks** from WebArena
- **10 trials** per task (different seeds)
- **Total**: 50 × 10 = 500 evaluations per agent

With 5 agents (4 LLMs + LCA):
- **Total evaluations**: 2,500

## Expected Runtime

On Google Colab with T4 GPU:
- **Model loading**: ~5 min per model
- **Evaluation**: ~2-3 min per task per agent
- **Total**: ~5-8 hours for full evaluation (50 tasks × 5 agents)

For faster testing, reduce `MAX_TASKS` to 5-10.

## Memory Requirements

### GPU Memory (with 8-bit quantization)
- Llama 3.1 8B: ~8 GB
- Gemma 2 9B: ~9 GB
- Qwen 2.5 7B: ~7 GB
- Phi-3 Mini: ~4 GB

Models are loaded one at a time, so peak usage = largest model (~9 GB).

### RAM
- Minimum: 12 GB
- Recommended: 16 GB+

## Output Files

Results saved to `webarena_results/`:

1. **CSV**: `webarena_opensource_YYYYMMDD_HHMMSS.csv`
   - All trial data (task_id, agent, success, time, quality, etc.)

2. **JSON**: `webarena_opensource_YYYYMMDD_HHMMSS.json`
   - Same data in JSON format

3. **Summary**: `webarena_summary_YYYYMMDD_HHMMSS.json`
   - Aggregated statistics per agent

## Statistical Analysis

The script computes:

### Descriptive Statistics
- Success rate (mean ± std)
- Execution time (mean ± std)
- Quality score (mean ± std)

### Effect Sizes (Cohen's d)
Computed for all pairwise comparisons:
- **Time**: Efficiency comparison
- **Success Rate**: Reliability comparison
- **Quality**: Overall performance

Interpretation:
- |d| < 0.2: negligible
- |d| < 0.5: small
- |d| < 0.8: medium
- |d| < 1.2: large
- |d| ≥ 2.0: very large

### Statistical Tests
- **t-tests**: Pairwise significance
- **p-values**: Statistical significance

## Model-Specific Notes

### Llama 3.1
- Requires Hugging Face token for gated model
- Set token: `huggingface-cli login`

### Gemma 2
- May require acceptance on Hugging Face
- Largest model (9B) - highest memory usage

### Qwen 2.5
- Multilingual capabilities
- Good instruction following

### Phi-3
- Smallest model (3.8B)
- Fastest inference
- Lower memory usage

## LCA Implementation

Same as proprietary version:
- 3-layer context embeddings (global, shared, individual)
- 5 coordinated browser agents
- Preference threshold τ = 0.65
- Success voting mechanism

## Differences from Proprietary Version

### Proprietary (`webarena_evaluation.py`)
- GPT-4 via OpenAI API
- Claude-3.5 via Anthropic API
- Requires API keys
- Pay per token

### Open Source (`webarena_evaluation_opensource.py`)
- Llama, Gemma, Qwen, Phi via HuggingFace
- No API keys needed (free)
- Requires GPU for reasonable speed
- One-time download, unlimited use

## Troubleshooting

### Out of Memory (Colab)

**Reduce memory usage:**
```python
# Evaluate models one at a time
# Comment out other models in main()

agents = []
# agents.append(Llama31Agent())  # Comment out
agents.append(Gemma2Agent())     # Evaluate only Gemma
# agents.append(Qwen25Agent())   # Comment out
# agents.append(Phi3Agent())     # Comment out
agents.append(LCAAgent())        # Keep LCA
```

**Or use fewer LCA agents:**
```python
agents.append(LCAAgent(n_agents=3))  # Instead of 5
```

### Model Loading Fails

**Gated models (Llama, Gemma):**
```bash
# Login to Hugging Face
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

**Download issues:**
```python
# In Colab, set token directly
from huggingface_hub import login
login(token="your_token_here")
```

### Chrome Driver Not Found

```bash
# Reinstall Chrome
!apt-get update
!apt-get install -y chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
```

### Slow Inference

**Using CPU instead of GPU:**
- Check: `torch.cuda.is_available()` returns `True`
- Colab: Set Runtime → Change runtime type → GPU

**Reduce MAX_TASKS for testing:**
```python
MAX_TASKS = 5  # Quick test with 5 tasks
```

## Comparing Results

To compare with proprietary models:

1. Run proprietary version: `python webarena_evaluation.py`
2. Run open-source version: `python webarena_evaluation_opensource.py`
3. Compare Cohen's d effect sizes between:
   - GPT-4 vs LCA
   - Claude vs LCA
   - Llama/Gemma/Qwen/Phi vs LCA

This shows whether open-source models can match proprietary performance.

## Citation

If you use this evaluation in your ICLR paper:

```bibtex
@inproceedings{your-paper,
  title={Your Paper Title},
  author={Your Name},
  booktitle={ICLR},
  year={2025}
}
```

## License

For academic research purposes (ICLR paper).
