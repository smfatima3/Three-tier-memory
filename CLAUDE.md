# CLAUDE.md - AI Assistant Guide

## Repository Overview

This repository contains the implementation of the **Layered Contextual Alignment (LCA)** framework, a research project for multi-agent browser automation with 3-tier memory context embeddings.

**Purpose**: Demonstrate and validate that multi-agent systems with layered contextual alignment can achieve better performance than single-agent approaches in web automation tasks.

**Key Research Claims**:
- Multi-agent coordination through 3-layer context embeddings (global, shared, individual)
- Improved scalability and efficiency compared to sequential or naive parallel approaches
- Statistical validation using real WebArena tasks

---

## Codebase Structure

### Core Implementation Files

```
Three-tier-memory/
├── lca_real_impl.py              # Real browser automation with Selenium
├── complete_rebuttal_eval.py     # Comprehensive evaluation framework
├── test_lca_logic.py             # Unit tests for core LCA logic
├── README.md                     # Minimal project description
├── LICENSE                       # Apache 2.0 license
├── 4268_Layered_Contextual_Alignm.pdf  # Research paper
└── webarena_task.json            # (Optional) Task definitions
```

### File Descriptions

#### `lca_real_impl.py` (691 lines)
**Purpose**: Real browser automation implementation using Selenium WebDriver

**Key Components**:
- `ContextEmbedding` (L109-134): 3-layer context representation (global, shared, individual)
- `BrowserAgent` (L148-288): Individual browser agent with context awareness
- `LCACoordinator` (L290-392): Multi-agent coordination through alignment
- `setup_chrome_driver()` (L49-104): Cross-platform Chrome/ChromeDriver setup
- Baseline implementations: Sequential (L396-413), Naive Parallel (L416-449)

**Usage Pattern**:
```python
# Initialize coordinator with N agents
coordinator = LCACoordinator(n_agents=5, coordination_threshold=0.65)
coordinator.initialize_agents()

# Process URLs with coordination
results = coordinator.process_urls_coordinated(urls, "extract_data")

# Cleanup
coordinator.shutdown_agents()
```

#### `complete_rebuttal_eval.py` (1140 lines)
**Purpose**: Comprehensive evaluation framework addressing research paper reviewer concerns

**Key Components**:
- `WebArenaTask` (L135-149): Task data structure
- `GPT4Agent` (L256-322): OpenAI GPT-4 baseline
- `ClaudeAgent` (L324-384): Anthropic Claude baseline
- `LCAAgent` (L387-611): Real LCA implementation with browser automation
- `StatisticalAnalyzer` (L613-879): Cohen's d effect size computation
- `WebArenaEvaluator` (L881-999): Main evaluation orchestrator

**Critical Features**:
- Multiple trials per task for statistical validity
- Cohen's d effect size for multiple metrics (time, success rate, quality)
- Proper pooled standard deviation computation
- ANOVA tests for multi-group comparisons

#### `test_lca_logic.py` (161 lines)
**Purpose**: Quick verification of core LCA logic

**Tests**:
- Context embedding creation and alignment computation
- Coordination threshold logic (τ=0.65)
- Task file loading
- Required module imports

---

## Core Concepts

### 1. Three-Layer Context Embedding

The LCA framework uses a hierarchical context representation:

```python
@dataclass
class ContextEmbedding:
    global_context: np.ndarray    # Task-level objectives (what needs to be done)
    shared_context: np.ndarray    # Session-level state (team knowledge)
    individual_context: np.ndarray # Agent-specific observations (personal experience)
```

**Alignment Computation**:
```
alignment_score = λ_g × sim_global + λ_s × sim_shared + λ_i × sim_individual

Default weights: λ_g=0.35, λ_s=0.30, λ_i=0.35
Similarity: Cosine similarity between context vectors
```

### 2. Coordination Threshold (τ)

**Value**: τ = 0.65 (configurable)

**Meaning**: Agents with alignment_score ≥ τ should coordinate on shared tasks

**Implementation**: Used in task assignment to find best-suited agent based on context similarity

### 3. Task Assignment Strategy

```python
def assign_task(self, url: str) -> BrowserAgent:
    # 1. Update shared context from agent experiences
    # 2. For each agent:
    #    - Update individual context
    #    - Compute suitability score (success_rate × load_balance)
    # 3. Return agent with highest score
```

---

## Environment Setup

### Dependencies

**Core Libraries** (install all):
```bash
pip install selenium numpy pandas scipy matplotlib seaborn psutil
```

**AI API Libraries** (for evaluation):
```bash
pip install openai anthropic
```

### Browser Automation Setup

**Google Colab**:
```python
!apt-get update
!apt-get install -y chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
```

**Local Machine**:
1. Install Chrome or Chromium browser
2. Download ChromeDriver from https://chromedriver.chromium.org/
3. Ensure ChromeDriver is in PATH

**Verification**:
```python
from lca_real_impl import setup_chrome_driver
driver = setup_chrome_driver()
driver.get("http://example.com")
print(driver.title)  # Should print "Example Domain"
driver.quit()
```

### API Keys (for evaluation only)

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Development Workflows

### Running Core Implementation

**Basic scalability test**:
```bash
python lca_real_impl.py
```

**What it does**:
- Tests Chrome/ChromeDriver setup
- Runs 3 experiments: Sequential, Naive Parallel, LCA Coordinated
- Uses 7 test URLs (HTTPBin, Example.com)
- Generates performance comparison metrics
- Saves results to `real_lca_results_TIMESTAMP.json`

**Expected Output**:
- Sequential baseline time
- Naive parallel speedup
- LCA coordinated speedup and efficiency
- Comparison table

### Running Full Evaluation

**Quick test** (5 tasks × 5 trials):
```bash
python complete_rebuttal_eval.py --tasks 5 --trials 5
```

**Full evaluation** (25 tasks × 10 trials):
```bash
python complete_rebuttal_eval.py --tasks 25 --trials 10
```

**What it does**:
1. Loads WebArena tasks from JSON
2. Initializes agents (GPT-4, Claude, LCA)
3. Runs each agent on each task multiple times with different seeds
4. Computes descriptive statistics
5. Calculates Cohen's d effect sizes for all metrics
6. Performs ANOVA tests
7. Saves results to `webarena_results/`

**Output Files**:
- `full_results_TIMESTAMP.csv` - All trial data
- `descriptive_statistics.csv` - Mean/SD per agent
- `effect_sizes.csv` - Cohen's d comparisons
- `summary_TIMESTAMP.json` - High-level metrics

### Running Tests

**Verify core logic**:
```bash
python test_lca_logic.py
```

**What it tests**:
- Context embedding alignment computation
- Coordination threshold logic
- Task file loading
- Import availability

---

## Key Conventions & Patterns

### Code Style

1. **Dataclasses for structured data**: All data structures use `@dataclass`
2. **Type hints**: All functions have type annotations
3. **Docstrings**: All classes and complex functions have docstrings
4. **Error handling**: Try-except with specific exception types
5. **Logging**: Use `print()` for user-facing messages (no logging framework)

### Important Patterns

#### Browser Driver Management

**Always use context or explicit cleanup**:
```python
# Good
agent = BrowserAgent("agent_0")
agent.initialize()
try:
    result = agent.process_url(url, "extract_data")
finally:
    agent.shutdown()

# Better (for coordinator)
coordinator = LCACoordinator(n_agents=5)
coordinator.initialize_agents()
try:
    results = coordinator.process_urls_coordinated(urls)
finally:
    coordinator.shutdown_agents()
```

#### Headless Mode

**Default**: All agents run in headless mode (no GUI)
**Override**: Set `headless=False` in ChromeOptions for debugging

#### Random Seeds

**Reproducibility**: All trials use deterministic seeds
```python
seed = 42 + trial_num + task_idx * 100
np.random.seed(seed)
```

### Common Gotchas

1. **ChromeDriver version mismatch**: Ensure ChromeDriver matches Chrome version
2. **Timeout errors**: Default page load timeout is 30s, adjust if needed
3. **Memory leaks**: Always call `driver.quit()` or `agent.shutdown()`
4. **API rate limits**: Add delays between API calls in evaluation
5. **Stale element references**: Re-fetch DOM elements after page changes

---

## Common Tasks

### Task 1: Add a New Baseline Agent

**File**: `complete_rebuttal_eval.py`

**Steps**:
1. Create new class inheriting from `BaseAgent` (L241-253)
2. Implement `run_trial(task, trial_num, seed)` method
3. Implement `cleanup()` method
4. Add to agent initialization in `main()` (L1036-1088)

**Example**:
```python
class MyNewAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("MyAgent")
        self.config = config

    def run_trial(self, task, trial_num, seed):
        start_time = time.time()
        # Your implementation here
        return TrialResult(
            task.task_id, self.name, trial_num, seed,
            success=True, execution_time=time.time()-start_time,
            quality_score=0.8
        )

    def cleanup(self):
        pass  # Cleanup resources
```

### Task 2: Modify Context Embedding Dimensionality

**Files**: `lca_real_impl.py`, `complete_rebuttal_eval.py`

**Change**:
```python
# Current (L297-298 in lca_real_impl.py)
self.global_context = np.random.randn(10)
self.shared_context = np.random.randn(10)

# Modified for 20-dimensional embeddings
self.global_context = np.random.randn(20)
self.shared_context = np.random.randn(20)
```

**Also update** all context update functions to match new dimensionality:
- `_update_global_context()` (L428-446)
- `_update_shared_context()` (L448-473)

### Task 3: Change Coordination Threshold

**File**: `lca_real_impl.py` (L293), `complete_rebuttal_eval.py` (L394)

**Modify**:
```python
# Default τ=0.65
coordinator = LCACoordinator(coordination_threshold=0.75)  # Stricter
coordinator = LCACoordinator(coordination_threshold=0.50)  # More lenient
```

**Impact**: Higher τ → fewer agents coordinate, more independent work

### Task 4: Add Custom Task Evaluation Metric

**File**: `complete_rebuttal_eval.py`

**Modify** `compute_quality_score()` (L212-238):
```python
def compute_quality_score(success, answer, error, has_content=True) -> float:
    quality = 0.0

    # Task completion (40%)
    if success:
        quality += 0.4

    # Answer quality (30%)
    if answer and len(answer) > 20:
        quality += 0.3

    # Error-free execution (30%)
    if not error:
        quality += 0.3

    # YOUR NEW METRIC (add weight, adjust others accordingly)
    # Example: Response time bonus
    # if execution_time < 5.0:
    #     quality += 0.1

    return min(1.0, quality)
```

### Task 5: Add New Test URLs

**File**: `lca_real_impl.py` (L611-623)

**Add to** `TEST_URLS`:
```python
TEST_URLS = [
    "https://httpbin.org/",
    "https://httpbin.org/html",
    # ... existing URLs ...

    # Your new URLs
    "https://your-test-site.com/page1",
    "https://your-test-site.com/page2",
]
```

---

## Testing & Validation

### Unit Testing Checklist

Before committing changes:

1. **Core logic test**:
   ```bash
   python test_lca_logic.py
   ```
   Expected: All tests pass, no errors

2. **Chrome setup test**:
   ```python
   from lca_real_impl import setup_chrome_driver
   driver = setup_chrome_driver()
   driver.get("http://example.com")
   assert "Example Domain" in driver.title
   driver.quit()
   ```

3. **Context alignment test**:
   ```python
   from lca_real_impl import ContextEmbedding
   ctx1 = ContextEmbedding(
       global_context=np.ones(10),
       shared_context=np.ones(10),
       individual_context=np.ones(10)
   )
   ctx2 = ContextEmbedding(
       global_context=np.ones(10),
       shared_context=np.ones(10),
       individual_context=np.zeros(10)
   )
   score = ctx1.compute_alignment(ctx2)
   assert 0.0 <= score <= 1.0
   ```

### Integration Testing

**Minimal working example**:
```bash
# Test with 2 tasks, 2 trials
python complete_rebuttal_eval.py --tasks 2 --trials 2
```

**Expected**:
- Agents initialize successfully
- Tasks execute without exceptions
- Results saved to `webarena_results/`
- Statistical analysis completes

### Performance Validation

**Expected ranges** (from paper claims):
- **Speedup**: 2x-4x vs sequential (depends on task parallelism)
- **Efficiency**: 60-80% (speedup / n_agents)
- **Success rate**: >90% for simple tasks
- **Cohen's d**: 0.5-1.5 for real tasks (NOT 6.39 from paper)

**Red flags**:
- Cohen's d > 2.0 → Check for data issues
- Success rate < 50% → Check Chrome/WebDriver setup
- Speedup < 1.5x → Check task parallelism or coordination overhead
- All failures → Check network, API keys, ChromeDriver

---

## Research Context

### Paper Claims to Validate

1. **3-layer context improves coordination** (Section 3.2)
   - Test: Compare LCA vs flat context embedding
   - Metric: Coordination overhead, task assignment quality

2. **Scalability up to 50 agents** (Section 4.3)
   - Test: Run with 2, 5, 10, 20, 30, 40, 50 agents
   - Metric: Speedup, efficiency, memory usage

3. **Cohen's d = 6.39** (Reviewer Q6 concern)
   - Issue: Extremely large, suggests synthetic data
   - Validation: Real evaluation shows d ∈ [0.5, 1.5]
   - Response: Acknowledge and provide corrected statistics

### Known Issues from Code Review

1. **Simulated browser mode**: Some tests use simulated delays, not real browser
   - Fix: Use `lca_real_impl.py` for real validation
   - Check: Look for `asyncio.sleep()` calls

2. **Hardcoded success probabilities** (complete_rebuttal_eval.py)
   - Line 489: `base_success_prob = 0.92`
   - Fix: Remove in real evaluation, use actual task outcomes

3. **Limited WebArena task coverage**
   - Paper claims: 812 tasks
   - Implementation: 5-25 tasks
   - Note: Full evaluation requires real WebArena environment

### Reviewer Concerns Addressed

**Q6: "Why only computed for time, not quality?"**
- Fixed in `StatisticalAnalyzer.analyze_results()` (L667-878)
- Now computes Cohen's d for: time, success rate, quality score
- Includes ANOVA tests for all metrics

**Q7: "Tasks are self-constructed, not WebArena"**
- Framework supports loading from `webarena_task.json`
- Can integrate with real WebArena environment
- Current: Sample tasks for demonstration

---

## Making Changes: Best Practices

### Before Modifying Code

1. **Read the relevant section** of the paper (PDF included)
2. **Understand the research claim** being implemented
3. **Check existing tests** to understand expected behavior
4. **Identify impact scope**: Core logic vs evaluation vs baselines

### Coding Guidelines

1. **Preserve statistical validity**:
   - Don't change random seeds without reason
   - Keep multiple trials for statistical power
   - Document any changes to metrics

2. **Maintain reproducibility**:
   - Use deterministic randomness (seeded)
   - Log all hyperparameters
   - Save complete experimental configurations

3. **Browser automation safety**:
   - Always use headless mode in production
   - Set reasonable timeouts (30s default)
   - Handle TimeoutException and WebDriverException
   - Clean up drivers in finally blocks

4. **API cost awareness**:
   - GPT-4 and Claude calls cost money
   - Use smaller models for testing (gpt-4o-mini, claude-3-haiku)
   - Add --dry-run flag for validation without API calls

### Testing Changes

1. **Unit tests first**: `python test_lca_logic.py`
2. **Small integration test**: `--tasks 2 --trials 2`
3. **Full evaluation**: `--tasks 25 --trials 10` (for paper results)

### Git Workflow

```bash
# Check status
git status

# Add changed files
git add lca_real_impl.py complete_rebuttal_eval.py

# Commit with descriptive message
git commit -m "feat: Add custom quality metric for task evaluation"

# Push to designated branch
git push -u origin claude/claude-md-mhzrkcfqe27khh6t-01RsuBND6fNSu7ZatVgpQ1sg
```

---

## Troubleshooting

### Chrome/ChromeDriver Issues

**Error**: "ChromeDriver version mismatch"
```bash
# Check Chrome version
google-chrome --version  # or chromium --version

# Download matching ChromeDriver from:
# https://chromedriver.chromium.org/downloads
```

**Error**: "WebDriverException: unknown error: DevToolsActivePort file doesn't exist"
```python
# Add to ChromeOptions:
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--remote-debugging-port=9222')
```

### API Errors

**Error**: "OpenAI API key not found"
```bash
export OPENAI_API_KEY="sk-..."
# Or create .env file (add to .gitignore!)
```

**Error**: "Rate limit exceeded"
```python
# Add delays in evaluation loop
import time
time.sleep(1)  # 1 second between API calls
```

### Memory Issues

**Error**: "Out of memory" with many agents
```python
# Reduce concurrent agents
coordinator = LCACoordinator(n_agents=3)  # Instead of 50

# Or increase system memory limit (Linux)
ulimit -v unlimited
```

### Statistical Analysis Issues

**Warning**: "Cohen's d is unusually large (>2.0)"
- Check for data quality issues
- Verify both groups have sufficient samples (n>5)
- Inspect raw data distributions
- Consider outliers or measurement errors

---

## Quick Reference

### File Line Number Guide

**lca_real_impl.py**:
- L49: Chrome driver setup
- L109: ContextEmbedding class
- L148: BrowserAgent class
- L290: LCACoordinator class
- L396: Sequential baseline
- L416: Naive parallel baseline
- L454: Main experiment runner

**complete_rebuttal_eval.py**:
- L135: WebArenaTask dataclass
- L152: TrialResult dataclass
- L166: Task loader
- L212: Quality scoring
- L256: GPT4Agent
- L324: ClaudeAgent
- L387: LCAAgent (real implementation)
- L613: StatisticalAnalyzer
- L881: WebArenaEvaluator
- L1002: Main entry point

**test_lca_logic.py**:
- L13: Context embedding tests
- L64: Coordination threshold tests
- L96: Task loading tests
- L115: Import tests

### Command Cheat Sheet

```bash
# Run basic implementation
python lca_real_impl.py

# Run full evaluation (quick)
python complete_rebuttal_eval.py --tasks 5 --trials 5

# Run full evaluation (complete)
python complete_rebuttal_eval.py --tasks 25 --trials 10

# Run tests
python test_lca_logic.py

# Install dependencies
pip install selenium numpy pandas scipy matplotlib seaborn psutil openai anthropic

# Setup Chrome (Colab)
apt-get update && apt-get install -y chromium-chromedriver
cp /usr/lib/chromium-browser/chromedriver /usr/bin
```

---

## Additional Resources

- **Research Paper**: `4268_Layered_Contextual_Alignm.pdf`
- **WebArena Project**: https://webarena.dev/
- **Selenium Docs**: https://selenium-python.readthedocs.io/
- **Cohen's d Reference**: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Maintainer**: LCA Research Team
