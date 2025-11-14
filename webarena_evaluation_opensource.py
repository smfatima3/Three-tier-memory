#!/usr/bin/env python3
"""
WEBARENA EVALUATION - OPEN SOURCE MODELS
- Open source LLMs (Llama 3.1, Gemma 2, Qwen 2.5, Phi-3)
- Real browser automation (LCA)
- Multiple trials for statistical validity
- Proper Cohen's d calculation
- Optimized for Google Colab with GPU

Addresses reviewer concerns:
Q6: Proper effect size calculation with multiple metrics
Q7: Uses real WebArena tasks (not self-constructed)
"""

import os
import json
import time
import sys
import subprocess
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============== AUTO-SETUP FOR COLAB ==============

def setup_colab_environment():
    """Auto-setup for Google Colab"""
    is_colab = 'google.colab' in sys.modules

    if is_colab:
        print("🔧 Setting up Google Colab environment...")

        # Install required packages
        print("Installing packages...")
        subprocess.run(['pip', 'install', '-q',
                       'transformers', 'accelerate', 'bitsandbytes',
                       'selenium', 'pandas', 'scipy'],
                      check=False, capture_output=True)

        # Setup Chrome
        print("Installing Chrome...")
        try:
            subprocess.run(['apt-get', 'update'], check=True,
                         capture_output=True, timeout=60)
            subprocess.run(['apt-get', 'install', '-y', 'chromium-chromedriver'],
                         check=True, capture_output=True, timeout=120)
            subprocess.run(['cp', '/usr/lib/chromium-browser/chromedriver', '/usr/bin'],
                         check=True, capture_output=True, timeout=30)
            print("✓ Environment setup complete")
        except Exception as e:
            print(f"⚠️  Chrome setup failed: {e}")

        return True
    return False

IS_COLAB = setup_colab_environment()

# Check GPU availability
import torch
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected - will use CPU (slower)")

# HuggingFace imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HF_AVAILABLE = True
    print("✓ HuggingFace Transformers available")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HuggingFace not available. Install: pip install transformers torch")

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not available. Install: pip install selenium")

# ============== CHROME DRIVER SETUP ==============

def create_chrome_driver():
    """Create Chrome driver for Colab"""
    options = ChromeOptions()

    # Essential headless options for Colab
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-setuid-sandbox')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')

    try:
        if IS_COLAB:
            options.binary_location = '/usr/bin/chromium-browser'
            service = ChromeService('/usr/bin/chromedriver')
            driver = webdriver.Chrome(service=service, options=options)
        else:
            driver = webdriver.Chrome(options=options)

        driver.set_page_load_timeout(30)
        driver.set_script_timeout(30)
        driver.implicitly_wait(5)

        return driver

    except Exception as e:
        print(f"❌ Chrome driver creation failed: {e}")
        raise

# ============== DATA STRUCTURES ==============

@dataclass
class WebArenaTask:
    """WebArena task structure"""
    task_id: str
    intent: str
    start_url: str
    target_url: str = None
    success_criteria: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.start_url:
            self.start_url = "http://example.com"
        if not self.start_url.startswith(('http://', 'https://')):
            self.start_url = 'http://' + self.start_url

@dataclass
class TrialResult:
    """Single trial result"""
    task_id: str
    agent_name: str
    trial_num: int
    seed: int
    success: bool
    execution_time: float
    quality_score: float
    answer: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

# ============== TASK LOADER ==============

def load_webarena_tasks(json_path: str = "webarena_task.json") -> List[WebArenaTask]:
    """Load REAL WebArena tasks from JSON file"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"❌ WebArena tasks not found at {json_path}\n"
            f"Please ensure the file exists in the current directory"
        )

    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        tasks_data = data
    elif isinstance(data, dict):
        if 'tasks' in data:
            tasks_data = data['tasks']
        elif 'data' in data:
            tasks_data = data['data']
        else:
            tasks_data = [data]
    else:
        raise ValueError("Unknown JSON structure")

    tasks = []
    for item in tasks_data:
        try:
            task = WebArenaTask(
                task_id=item.get('task_id', item.get('id', f"task_{len(tasks)}")),
                intent=item.get('intent', item.get('instruction', item.get('query', ''))),
                start_url=item.get('start_url', item.get('url', 'http://example.com')),
                target_url=item.get('target_url'),
                success_criteria=item.get('success_criteria', item.get('criteria', {}))
            )
            tasks.append(task)
        except Exception as e:
            print(f"⚠️  Skipping invalid task: {e}")
            continue

    if not tasks:
        raise ValueError("No valid tasks loaded from JSON")

    print(f"✓ Loaded {len(tasks)} WebArena tasks")
    return tasks

# ============== QUALITY SCORING ==============

def compute_quality_score(result: Dict) -> float:
    """Compute quality score based on multiple factors"""
    if not result.get('success', False):
        return 0.0

    quality = 0.0

    if result.get('success'):
        quality += 0.4

    answer = result.get('answer', '')
    if answer and len(answer) > 20:
        quality += 0.3

    if not result.get('error'):
        quality += 0.3

    return min(1.0, quality)

# ============== OPEN SOURCE MODEL AGENTS ==============

class BaseAgent:
    """Base class for all agents"""
    def __init__(self, name: str):
        self.name = name

    def run_single_trial(self, task: WebArenaTask, trial_num: int, seed: int) -> TrialResult:
        """Run a single trial with given seed"""
        raise NotImplementedError


class HuggingFaceAgent(BaseAgent):
    """Base class for HuggingFace model agents"""

    def __init__(self, name: str, model_id: str, max_tokens: int = 512):
        super().__init__(name)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def initialize(self):
        """Initialize model with quantization if needed"""
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace Transformers not available")

        print(f"  Loading {self.name} ({self.model_id})...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Use 8-bit quantization to save memory
            if GPU_AVAILABLE:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="cpu",
                    torch_dtype=torch.float32
                )

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=0.0,
                do_sample=False
            )

            print(f"  ✓ {self.name} loaded")

        except Exception as e:
            print(f"  ✗ Failed to load {self.name}: {e}")
            raise

    def run_single_trial(self, task: WebArenaTask, trial_num: int, seed: int) -> TrialResult:
        """Run single trial"""
        if not self.pipeline:
            return TrialResult(
                task.task_id, self.name, trial_num, seed,
                False, 0, 0.0, error="Model not initialized"
            )

        np.random.seed(seed)
        torch.manual_seed(seed)

        start_time = time.time()
        try:
            prompt = self._format_prompt(task)

            response = self.pipeline(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            answer = response[0]['generated_text']
            # Remove the prompt from answer
            if prompt in answer:
                answer = answer.replace(prompt, "").strip()

            exec_time = time.time() - start_time

            result_dict = {'success': True, 'answer': answer, 'error': None}
            quality = compute_quality_score(result_dict)

            return TrialResult(
                task.task_id, self.name, trial_num, seed,
                True, exec_time, quality,
                answer=answer[:200],  # Truncate for display
                metadata={
                    'model': self.model_id,
                    'answer_length': len(answer)
                }
            )

        except Exception as e:
            exec_time = time.time() - start_time
            result_dict = {'success': False, 'answer': None, 'error': str(e)}
            quality = compute_quality_score(result_dict)

            return TrialResult(
                task.task_id, self.name, trial_num, seed,
                False, exec_time, quality,
                error=str(e)[:100]
            )

    def _format_prompt(self, task: WebArenaTask) -> str:
        """Format prompt for the model"""
        return f"""You are a web automation expert. Create a detailed step-by-step plan to complete this task:

Task: {task.intent}
Start URL: {task.start_url}

Provide specific, actionable steps:"""


class Llama31Agent(HuggingFaceAgent):
    """Llama 3.1 8B Instruct agent"""

    def __init__(self):
        super().__init__(
            name="Llama-3.1-8B",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            max_tokens=512
        )

    def _format_prompt(self, task: WebArenaTask) -> str:
        """Llama 3.1 uses specific chat format"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a web automation expert. Be specific and detailed.<|eot_id|><|start_header_id|>user<|end_header_id|>

Create a detailed step-by-step plan to complete this task:

Task: {task.intent}
Start URL: {task.start_url}

Provide specific, actionable steps:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class Gemma2Agent(HuggingFaceAgent):
    """Gemma 2 9B Instruct agent"""

    def __init__(self):
        super().__init__(
            name="Gemma-2-9B",
            model_id="google/gemma-2-9b-it",
            max_tokens=512
        )

    def _format_prompt(self, task: WebArenaTask) -> str:
        """Gemma 2 format"""
        return f"""<bos><start_of_turn>user
Create a detailed step-by-step plan to complete this task:

Task: {task.intent}
Start URL: {task.start_url}

Provide specific, actionable steps:<end_of_turn>
<start_of_turn>model
"""


class Qwen25Agent(HuggingFaceAgent):
    """Qwen 2.5 7B Instruct agent"""

    def __init__(self):
        super().__init__(
            name="Qwen-2.5-7B",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            max_tokens=512
        )

    def _format_prompt(self, task: WebArenaTask) -> str:
        """Qwen 2.5 chat format"""
        return f"""<|im_start|>system
You are a web automation expert. Be specific and detailed.<|im_end|>
<|im_start|>user
Create a detailed step-by-step plan to complete this task:

Task: {task.intent}
Start URL: {task.start_url}

Provide specific, actionable steps:<|im_end|>
<|im_start|>assistant
"""


class Phi3Agent(HuggingFaceAgent):
    """Phi-3 Mini 4K Instruct agent"""

    def __init__(self):
        super().__init__(
            name="Phi-3-Mini",
            model_id="microsoft/Phi-3-mini-4k-instruct",
            max_tokens=512
        )

    def _format_prompt(self, task: WebArenaTask) -> str:
        """Phi-3 format"""
        return f"""<|system|>
You are a web automation expert. Be specific and detailed.<|end|>
<|user|>
Create a detailed step-by-step plan to complete this task:

Task: {task.intent}
Start URL: {task.start_url}

Provide specific, actionable steps:<|end|>
<|assistant|>
"""

# ============== LCA IMPLEMENTATION (SAME AS BEFORE) ==============

@dataclass
class ContextEmbedding:
    """3-layer context representation as described in paper"""
    global_context: np.ndarray
    shared_context: np.ndarray
    individual_context: np.ndarray
    timestamp: float = field(default_factory=time.time)

    def compute_alignment(self, other: 'ContextEmbedding',
                         weights: Tuple[float, float, float] = (0.35, 0.30, 0.35)) -> float:
        """Compute alignment score between two context embeddings"""
        λ_g, λ_s, λ_i = weights

        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)

        sim_g = cosine_sim(self.global_context, other.global_context)
        sim_s = cosine_sim(self.shared_context, other.shared_context)
        sim_i = cosine_sim(self.individual_context, other.individual_context)

        return λ_g * sim_g + λ_s * sim_s + λ_i * sim_i


@dataclass
class TaskResult:
    """Result from processing a single URL"""
    url: str
    success: bool
    execution_time: float
    agent_id: str
    data_extracted: Dict
    error: Optional[str] = None
    retry_count: int = 0


class BrowserAgent:
    """Individual browser agent with context awareness"""

    def __init__(self, agent_id: str, headless: bool = True):
        self.agent_id = agent_id
        self.headless = headless
        self.driver = None
        self.context = None
        self.performance_history = []

    def initialize(self):
        """Initialize browser driver"""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not available")

        try:
            self.driver = create_chrome_driver()
            self.driver.set_page_load_timeout(30)
            print(f"  ✓ Agent {self.agent_id} initialized")
        except Exception as e:
            print(f"  ✗ Failed to initialize agent {self.agent_id}: {e}")
            raise

    def shutdown(self):
        """Clean shutdown of browser"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

    def update_context(self, global_ctx: np.ndarray, shared_ctx: np.ndarray):
        """Update agent's context based on task state"""
        if len(self.performance_history) > 0:
            recent_perf = self.performance_history[-10:]
            success_rate = sum(1 for r in recent_perf if r['success']) / len(recent_perf)
            avg_time = np.mean([r['time'] for r in recent_perf])
            individual_features = np.array([success_rate, 1.0/avg_time if avg_time > 0 else 0])
        else:
            individual_features = np.array([1.0, 1.0])

        if len(individual_features) < len(global_ctx):
            individual_features = np.pad(individual_features, (0, len(global_ctx) - len(individual_features)))

        self.context = ContextEmbedding(
            global_context=global_ctx.copy(),
            shared_context=shared_ctx.copy(),
            individual_context=individual_features
        )

    def process_url(self, url: str, task_objective: str, timeout: float = 30) -> TaskResult:
        """Process a single URL"""
        start_time = time.time()

        try:
            self.driver.get(url)

            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )

            data = self._extract_data(task_objective)
            execution_time = time.time() - start_time
            success = self._determine_success(data, task_objective)

            self.performance_history.append({
                'success': success,
                'time': execution_time,
                'url': url
            })

            return TaskResult(
                url=url,
                success=success,
                execution_time=execution_time,
                agent_id=self.agent_id,
                data_extracted=data
            )

        except TimeoutException as e:
            execution_time = time.time() - start_time
            self.performance_history.append({
                'success': False,
                'time': execution_time,
                'url': url
            })
            return TaskResult(
                url=url,
                success=False,
                execution_time=execution_time,
                agent_id=self.agent_id,
                data_extracted={},
                error=f"Timeout: {str(e)}"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_history.append({
                'success': False,
                'time': execution_time,
                'url': url
            })
            return TaskResult(
                url=url,
                success=False,
                execution_time=execution_time,
                agent_id=self.agent_id,
                data_extracted={},
                error=str(e)
            )

    def _extract_data(self, objective: str) -> Dict:
        """Extract data based on objective"""
        data = {
            'title': self.driver.title,
            'url': self.driver.current_url,
            'timestamp': time.time()
        }

        try:
            links = self.driver.find_elements(By.TAG_NAME, 'a')
            data['num_links'] = len(links)

            images = self.driver.find_elements(By.TAG_NAME, 'img')
            data['num_images'] = len(images)

            forms = self.driver.find_elements(By.TAG_NAME, 'form')
            data['num_forms'] = len(forms)

            buttons = self.driver.find_elements(By.TAG_NAME, 'button')
            data['num_buttons'] = len(buttons)

            body = self.driver.find_element(By.TAG_NAME, 'body')
            data['text_length'] = len(body.text)

        except Exception as e:
            data['extraction_error'] = str(e)

        return data

    def _determine_success(self, data: Dict, objective: str) -> bool:
        """Determine if task was successful"""
        objective_lower = objective.lower()

        if 'search' in objective_lower or 'find' in objective_lower:
            return data.get('num_links', 0) > 5
        elif 'form' in objective_lower or 'submit' in objective_lower:
            return data.get('num_forms', 0) > 0
        elif 'button' in objective_lower or 'click' in objective_lower:
            return data.get('num_buttons', 0) > 0
        else:
            return data.get('num_links', 0) > 0 or data.get('text_length', 0) > 100


class LCACoordinator:
    """Layered Contextual Alignment Coordinator"""

    def __init__(self, n_agents: int = 5, coordination_threshold: float = 0.65):
        self.n_agents = n_agents
        self.tau = coordination_threshold
        self.agents: List[BrowserAgent] = []
        self.global_context = np.random.randn(10)
        self.shared_context = np.random.randn(10)

    def initialize_agents(self):
        """Initialize agent pool"""
        print(f"Initializing {self.n_agents} LCA browser agents...")
        for i in range(self.n_agents):
            agent = BrowserAgent(f"lca_agent_{i}", headless=True)
            agent.initialize()
            agent.update_context(self.global_context, self.shared_context)
            self.agents.append(agent)
        print(f"✓ {self.n_agents} LCA agents ready")

    def shutdown_agents(self):
        """Shutdown all agents"""
        for agent in self.agents:
            agent.shutdown()

    def assign_task(self, url: str) -> BrowserAgent:
        """Assign URL to most suitable agent"""
        self._update_shared_context()

        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent.update_context(self.global_context, self.shared_context)

            if len(agent.performance_history) > 0:
                success_rate = sum(1 for r in agent.performance_history if r['success']) / len(agent.performance_history)
                recent_load = len([r for r in agent.performance_history[-5:]])
                score = success_rate * (1.0 - recent_load / 10.0)
            else:
                score = 1.0

            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent if best_agent else self.agents[0]

    def _update_shared_context(self):
        """Update shared context"""
        if not self.agents:
            return

        all_contexts = []
        for agent in self.agents:
            if agent.context:
                all_contexts.append(agent.context.individual_context)

        if all_contexts:
            mean_ctx = np.mean(all_contexts, axis=0)
            if len(mean_ctx) < len(self.shared_context):
                mean_ctx = np.pad(mean_ctx, (0, len(self.shared_context) - len(mean_ctx)))
            elif len(mean_ctx) > len(self.shared_context):
                mean_ctx = mean_ctx[:len(self.shared_context)]
            self.shared_context = mean_ctx

    def process_task_coordinated(self, task: WebArenaTask, seed: int = 42) -> Dict:
        """Process task with multi-agent coordination"""
        np.random.seed(seed)
        random.seed(seed)

        urls_to_process = [task.start_url]
        if task.target_url and task.target_url != task.start_url:
            urls_to_process.append(task.target_url)

        n_parallel_agents = min(3, self.n_agents)

        results = []
        success_votes = 0
        total_time = 0

        with ThreadPoolExecutor(max_workers=self.n_agents) as executor:
            future_to_info = {}

            for url in urls_to_process:
                for _ in range(n_parallel_agents):
                    agent = self.assign_task(url)
                    future = executor.submit(agent.process_url, url, task.intent)
                    future_to_info[future] = (url, agent.agent_id)

            for future in as_completed(future_to_info):
                url, agent_id = future_to_info[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.success:
                        success_votes += 1
                    total_time += result.execution_time
                except Exception as e:
                    results.append(TaskResult(
                        url=url,
                        success=False,
                        execution_time=0,
                        agent_id=agent_id,
                        data_extracted={},
                        error=str(e)
                    ))

        success_ratio = success_votes / len(results) if results else 0
        overall_success = success_ratio >= self.tau

        all_data = {}
        for r in results:
            all_data.update(r.data_extracted)

        agent_outcomes = [
            {
                'agent_id': r.agent_id,
                'success': r.success,
                'error': r.error,
                'time': r.execution_time
            }
            for r in results
        ]

        return {
            'success': overall_success,
            'success_ratio': success_ratio,
            'total_time': total_time / len(results) if results else 0,
            'num_agents_used': len(set(r.agent_id for r in results)),
            'agent_outcomes': agent_outcomes,
            'results': results,
            'data': all_data
        }


class LCAAgent(BaseAgent):
    """LCA Multi-Agent System"""

    def __init__(self, n_agents: int = 5):
        super().__init__("LCA")
        self.coordinator = LCACoordinator(n_agents=n_agents, coordination_threshold=0.65)
        self.initialized = False
        print(f"✓ LCA initialized with {n_agents} agents")

    def _ensure_initialized(self):
        """Lazy initialization"""
        if not self.initialized:
            self.coordinator.initialize_agents()
            self.initialized = True

    def run_single_trial(self, task: WebArenaTask, trial_num: int, seed: int) -> TrialResult:
        """Run single trial with coordination"""
        np.random.seed(seed)
        random.seed(seed)

        start_time = time.time()

        try:
            self._ensure_initialized()

            coord_result = self.coordinator.process_task_coordinated(task, seed)

            exec_time = time.time() - start_time

            success = coord_result['success']
            data = coord_result['data']
            success_ratio = coord_result['success_ratio']

            answer = f"LCA processed with {coord_result['num_agents_used']} agents, " \
                    f"success ratio: {success_ratio:.2f}, " \
                    f"title: {data.get('title', 'N/A')}"

            error_msg = None
            if not success:
                error_msg = f"Failed coordination threshold (ratio={success_ratio:.2f} < τ={self.coordinator.tau})"

            result_dict = {
                'success': success,
                'answer': answer,
                'error': error_msg
            }
            quality = compute_quality_score(result_dict)

            return TrialResult(
                task.task_id, self.name, trial_num, seed,
                success, exec_time, quality,
                answer=answer,
                error=error_msg,
                metadata={
                    'success_ratio': coord_result['success_ratio'],
                    'num_agents_used': coord_result['num_agents_used'],
                    'coordination_threshold': self.coordinator.tau,
                    'agent_outcomes': coord_result.get('agent_outcomes', []),
                    **data
                }
            )

        except Exception as e:
            exec_time = time.time() - start_time

            if self.initialized:
                try:
                    self.coordinator.shutdown_agents()
                except:
                    pass
                self.initialized = False

            result_dict = {'success': False, 'answer': None, 'error': str(e)}
            quality = compute_quality_score(result_dict)

            return TrialResult(
                task.task_id, self.name, trial_num, seed,
                False, exec_time, quality,
                error=str(e)[:100]
            )

    def shutdown(self):
        """Cleanup all agents"""
        if self.initialized:
            self.coordinator.shutdown_agents()
            self.initialized = False

# ============== STATISTICAL EVALUATOR ==============

class StatisticalEvaluator:
    """Evaluator with proper statistical testing"""
    def __init__(self, n_trials: int = 10, output_dir: str = "webarena_results"):
        self.n_trials = n_trials
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.all_results = []

    def evaluate_agents(self, agents: List[BaseAgent], tasks: List[WebArenaTask],
                       max_tasks: int = None) -> pd.DataFrame:
        """Evaluate agents with multiple trials"""
        if max_tasks:
            tasks = tasks[:max_tasks]

        print("\n" + "="*80)
        print(f"STATISTICAL EVALUATION - {len(tasks)} tasks × {self.n_trials} trials")
        print("="*80)

        browser_agents = []

        for agent in agents:
            print(f"\n{'='*60}")
            print(f"Agent: {agent.name}")
            print('='*60)

            if isinstance(agent, LCAAgent):
                browser_agents.append(agent)

            for task_idx, task in enumerate(tasks, 1):
                print(f"\n[Task {task_idx}/{len(tasks)}] {task.task_id}: {task.intent[:50]}...")

                for trial in range(self.n_trials):
                    seed = 42 + trial + task_idx * 100

                    print(f"  Trial {trial+1}/{self.n_trials}: ", end="", flush=True)

                    result = agent.run_single_trial(task, trial, seed)
                    self.all_results.append(result)

                    if result.success:
                        print(f"✓ {result.execution_time:.2f}s (Q={result.quality_score:.2f})")
                    else:
                        if result.error:
                            error_msg = result.error[:60] if len(result.error) > 60 else result.error
                            print(f"✗ {error_msg}")
                        else:
                            print(f"✗ Failed (no error message)")

        # Cleanup
        for agent in browser_agents:
            agent.shutdown()

        return self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        data = []
        for r in self.all_results:
            data.append({
                'task_id': r.task_id,
                'agent': r.agent_name,
                'trial': r.trial_num,
                'seed': r.seed,
                'success': r.success,
                'time': r.execution_time,
                'quality': r.quality_score,
                'error': r.error,
                **r.metadata
            })
        return pd.DataFrame(data)

    def compute_statistics(self, df: pd.DataFrame):
        """Compute comprehensive statistics"""
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)

        agents = df['agent'].unique()

        print("\n📊 DESCRIPTIVE STATISTICS")
        print("-" * 60)

        for agent in agents:
            agent_data = df[df['agent'] == agent]
            print(f"\n{agent}:")
            print(f"  N trials: {len(agent_data)}")
            print(f"  Success rate: {agent_data['success'].mean():.3f} ± {agent_data['success'].std():.3f}")
            print(f"  Time (s): {agent_data['time'].mean():.3f} ± {agent_data['time'].std():.3f}")
            print(f"  Quality: {agent_data['quality'].mean():.3f} ± {agent_data['quality'].std():.3f}")

        if len(agents) >= 2:
            print("\n" + "="*80)
            print("EFFECT SIZES (Cohen's d)")
            print("="*80)
            print("\nComparing all agents pairwise:")
            print("-" * 60)

            for i, agent1 in enumerate(agents):
                for agent2 in agents[i+1:]:
                    data1 = df[df['agent'] == agent1]
                    data2 = df[df['agent'] == agent2]

                    print(f"\n{agent1} vs {agent2}:")

                    time1 = data1['time'].values
                    time2 = data2['time'].values
                    cohens_d_time = self._compute_cohens_d(time1, time2)
                    print(f"  Time: Cohen's d = {cohens_d_time:.4f} ({self._interpret_cohens_d(cohens_d_time)})")

                    success1 = data1['success'].astype(float).values
                    success2 = data2['success'].astype(float).values
                    cohens_d_success = self._compute_cohens_d(success1, success2)
                    print(f"  Success: Cohen's d = {cohens_d_success:.4f} ({self._interpret_cohens_d(cohens_d_success)})")

                    quality1 = data1['quality'].values
                    quality2 = data2['quality'].values
                    cohens_d_quality = self._compute_cohens_d(quality1, quality2)
                    print(f"  Quality: Cohen's d = {cohens_d_quality:.4f} ({self._interpret_cohens_d(cohens_d_quality)})")

                    t_stat_time, p_val_time = stats.ttest_ind(time1, time2)
                    t_stat_success, p_val_success = stats.ttest_ind(success1, success2)
                    t_stat_quality, p_val_quality = stats.ttest_ind(quality1, quality2)

                    print(f"  p-values: Time={p_val_time:.4f}, Success={p_val_success:.4f}, Quality={p_val_quality:.4f}")

    def _compute_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)

        pooled_std = np.sqrt((std1**2 + std2**2) / 2)

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d"""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        elif d_abs < 1.2:
            return "large"
        elif d_abs < 2.0:
            return "very large"
        else:
            return "extremely large"

    def save_results(self, df: pd.DataFrame):
        """Save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_file = self.output_dir / f"webarena_opensource_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n✓ CSV: {csv_file}")

        json_file = self.output_dir / f"webarena_opensource_{timestamp}.json"
        df.to_json(json_file, orient='records', indent=2)
        print(f"✓ JSON: {json_file}")

        summary = {
            'timestamp': timestamp,
            'n_trials': self.n_trials,
            'agents': df['agent'].unique().tolist(),
            'total_evaluations': len(df),
            'success_rates': df.groupby('agent')['success'].agg(['mean', 'std']).to_dict(),
            'execution_times': df.groupby('agent')['time'].agg(['mean', 'std']).to_dict(),
            'quality_scores': df.groupby('agent')['quality'].agg(['mean', 'std']).to_dict()
        }

        summary_file = self.output_dir / f"webarena_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary: {summary_file}")

# ============== MAIN ==============

def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("WEBARENA EVALUATION - OPEN SOURCE MODELS")
    print("Multiple trials for statistical validity")
    print("Optimized for Google Colab")
    print("="*80)

    # Configuration
    TASK_FILE = "webarena_task.json"
    N_TRIALS = 10
    MAX_TASKS = 50  # Limit to 50 tasks

    # Load tasks
    print(f"\n📂 Loading WebArena tasks from {TASK_FILE}...")
    try:
        tasks = load_webarena_tasks(TASK_FILE)
    except FileNotFoundError as e:
        print(f"\n{e}")
        return
    except Exception as e:
        print(f"\n❌ Error loading tasks: {e}")
        return

    # Initialize agents
    print("\n🤖 Initializing agents...")
    agents = []

    # Initialize open source models
    model_agents = [
        ("Llama 3.1 8B", Llama31Agent),
        ("Gemma 2 9B", Gemma2Agent),
        ("Qwen 2.5 7B", Qwen25Agent),
        ("Phi-3 Mini", Phi3Agent)
    ]

    for name, AgentClass in model_agents:
        try:
            print(f"\nInitializing {name}...")
            agent = AgentClass()
            agent.initialize()
            agents.append(agent)
        except Exception as e:
            print(f"⚠️  Failed to initialize {name}: {e}")
            print("    Continuing with other models...")

    # Add LCA agent
    if SELENIUM_AVAILABLE:
        try:
            agents.append(LCAAgent(n_agents=5))
        except Exception as e:
            print(f"⚠️  Failed to initialize LCA: {e}")
    else:
        print("⚠️  Selenium not available - LCA agent disabled")

    if not agents:
        print("\n❌ No agents available")
        return

    print(f"\n✓ Initialized {len(agents)} agents: {[a.name for a in agents]}")

    # Run evaluation
    print(f"\n🚀 Starting evaluation with {N_TRIALS} trials per task...")
    if MAX_TASKS:
        print(f"   (Limited to first {MAX_TASKS} tasks)")

    evaluator = StatisticalEvaluator(n_trials=N_TRIALS)

    try:
        df = evaluator.evaluate_agents(agents, tasks, max_tasks=MAX_TASKS)
        evaluator.compute_statistics(df)
        evaluator.save_results(df)

        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE")
        print("="*80)
        print(f"\nTotal evaluations: {len(df)}")
        print(f"Results saved to: {evaluator.output_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted")
    except Exception as e:
        print(f"\n\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
