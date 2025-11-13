# WebArena Evaluation Script

Complete evaluation script for ICLR paper addressing reviewer concerns Q6 and Q7.

## Features

✅ **Real API Calls**: GPT-4 and Claude-3.5 via official APIs
✅ **Real Browser Automation**: Selenium WebDriver for LCA agent
✅ **Multiple Trials**: 10 trials per task for statistical validity
✅ **Proper Statistics**: Cohen's d effect sizes for time, success rate, and quality
✅ **NO Placeholder Data**: All data collected from actual executions

## Requirements

```bash
pip install numpy pandas scipy selenium openai anthropic
```

For browser automation, you also need Chrome/Chromium:
- **Linux**: `apt-get install chromium-chromedriver`
- **macOS**: `brew install chromedriver`
- **Windows**: Download from [ChromeDriver](https://chromedriver.chromium.org/)

## Setup

1. **Set API Keys** (at least one required):
```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
```

2. **Prepare Task File**:
   - Edit `webarena_task.json` with your WebArena tasks
   - Example tasks are provided (Wikipedia, GitHub, arXiv, etc.)
   - For real WebArena benchmark, use tasks from the official dataset

## Usage

```bash
python webarena_evaluation.py
```

The script will:
1. Load tasks from `webarena_task.json`
2. Initialize available agents (GPT-4, Claude, LCA-Single)
3. Run each task 10 times per agent with different seeds
4. Compute comprehensive statistics including Cohen's d
5. Save results to `webarena_results/` directory

## Configuration

Edit the `main()` function to adjust:
- `N_TRIALS`: Number of trials per task (default: 10)
- `MAX_TASKS`: Limit number of tasks for testing (default: None = all)

## Output Files

Results are saved with timestamps in `webarena_results/`:
- `webarena_statistical_YYYYMMDD_HHMMSS.csv` - All trial data
- `webarena_statistical_YYYYMMDD_HHMMSS.json` - All trial data (JSON)
- `webarena_summary_YYYYMMDD_HHMMSS.json` - Aggregated statistics

## Statistical Analysis

The script computes:

### 1. Descriptive Statistics
- Success rate (mean ± std)
- Execution time (mean ± std)
- Quality score (mean ± std)

### 2. Effect Sizes (Cohen's d)
Computed for **three metrics** (addressing Reviewer Q6):
- **Time**: Efficiency comparison
- **Success Rate**: Reliability comparison
- **Quality**: Overall performance comparison

Interpretation:
- |d| < 0.2: negligible
- |d| < 0.5: small
- |d| < 0.8: medium
- |d| < 1.2: large
- |d| < 2.0: very large
- |d| ≥ 2.0: extremely large (unusual)

### 3. Statistical Tests
- **t-tests**: Pairwise comparisons with p-values
- **ANOVA**: Overall group differences

## Task File Format

```json
[
  {
    "task_id": "wa_001",
    "intent": "Description of what to do",
    "start_url": "https://example.com",
    "target_url": "https://example.com/target",
    "success_criteria": {
      "key": "value"
    }
  }
]
```

**Required fields**:
- `task_id`: Unique identifier
- `intent`: Natural language task description
- `start_url`: Starting webpage URL

**Optional fields**:
- `target_url`: Expected destination
- `success_criteria`: Dictionary of success conditions

## Agents

### GPT-4 Agent
- Uses `gpt-4o-mini` model by default
- Planning-based approach
- Requires OPENAI_API_KEY

### Claude-3.5 Agent
- Uses `claude-3-5-sonnet-20241022` model
- Planning-based approach
- Requires ANTHROPIC_API_KEY

### LCA-Single Agent
- Real browser automation with Selenium
- Context-aware execution
- Heuristic success detection
- Requires Chrome/Chromium

## Quality Scoring

Quality score (0-1) based on:
- **Task completion** (40%): Did it succeed?
- **Answer quality** (30%): Non-trivial response?
- **Error-free** (30%): No errors during execution?

## Addressing Reviewer Concerns

### Q6: Effect Size Calculation
- ✅ Proper Cohen's d formula: (M₁ - M₂) / pooled_std
- ✅ Multiple metrics: time, success rate, quality
- ✅ Interpretation guidelines provided
- ✅ Statistical significance tests (t-tests, ANOVA)

### Q7: Real WebArena Tasks
- ✅ Loads tasks from external JSON file
- ✅ Example tasks use real websites
- ✅ Compatible with official WebArena dataset format
- ✅ No hardcoded/synthetic data

## Notes for Paper

The paper claimed Cohen's d = 6.39 for execution time. This is **extremely unusual**:
- Cohen's d > 2.0 is already "very large"
- d = 6.39 suggests synthetic/simulated data

Real-world effect sizes are typically much smaller. This script measures actual effect sizes from real API calls and browser automation.

## Troubleshooting

**"Chrome driver creation failed"**:
- Install Chrome/Chromium
- Install chromedriver
- For Colab: Run the auto-setup commands shown

**"No agents available"**:
- Set at least one API key
- Or ensure Selenium + Chrome are installed for LCA agent

**"WebArena tasks not found"**:
- Create `webarena_task.json` in the current directory
- Use the example format provided

## License

This script is for academic research purposes (ICLR paper).
