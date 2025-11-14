# Google Colab Setup for WebArena Evaluation

## Complete Setup Code Block

Copy and paste this into a Google Colab cell to set up the complete environment:

```python
# ===================================================================
# GOOGLE COLAB SETUP FOR WEBARENA EVALUATION WITH LCA
# Complete prerequisite installation and configuration
# ===================================================================

# 1. Install System Dependencies
print("📦 Installing system dependencies...")
!apt-get update -qq
!apt-get install -y -qq chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
print("✓ Chrome and ChromeDriver installed\n")

# 2. Install Python Dependencies
print("📦 Installing Python packages...")
!pip install -q selenium anthropic openai numpy pandas scipy matplotlib seaborn
print("✓ Python packages installed\n")

# 3. Verify Chrome Installation
print("🔍 Verifying Chrome installation...")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

def test_chrome():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.binary_location = '/usr/bin/chromium-browser'
    service = Service('/usr/bin/chromedriver')

    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("data:text/html,<html><body>Test</body></html>")
        title = driver.title
        driver.quit()
        print(f"✓ Chrome working! Page title: '{title}'\n")
        return True
    except Exception as e:
        print(f"✗ Chrome test failed: {e}\n")
        return False

chrome_ok = test_chrome()

# 4. Set API Keys (Required)
print("🔑 Setting up API keys...")
print("Please set your API keys:\n")

import os
from getpass import getpass

# Option 1: Direct input (will be hidden)
if not os.getenv('OPENAI_API_KEY'):
    openai_key = getpass('Enter your OpenAI API key (or press Enter to skip): ')
    if openai_key:
        os.environ['OPENAI_API_KEY'] = openai_key
        print("✓ OpenAI API key set")
    else:
        print("⚠️  OpenAI API key skipped - GPT-4 will not be available")

if not os.getenv('ANTHROPIC_API_KEY'):
    anthropic_key = getpass('Enter your Anthropic API key (or press Enter to skip): ')
    if anthropic_key:
        os.environ['ANTHROPIC_API_KEY'] = anthropic_key
        print("✓ Anthropic API key set")
    else:
        print("⚠️  Anthropic API key skipped - Claude models will not be available")

print()

# 5. Clone/Download the Repository
print("📥 Setting up evaluation code...")
# Option A: If code is in a GitHub repo
# !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
# %cd YOUR_REPO

# Option B: Upload files manually or use Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/YOUR_FOLDER

# Option C: Download files directly
# !wget https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/complete_rebuttal_eval.py
# !wget https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/lca_real_impl.py

print("ℹ️  Please ensure the following files are in your working directory:")
print("   - complete_rebuttal_eval.py")
print("   - lca_real_impl.py")
print("   - webarena_task.json (optional)")
print()

# 6. Create Sample Task File (Optional)
print("📝 Creating sample task file...")
import json

sample_tasks = [
    {
        "task_id": "task_1",
        "intent": "Navigate to example website and extract title",
        "start_url": "http://example.com",
        "success_criteria": {"has_title": True}
    },
    {
        "task_id": "task_2",
        "intent": "Test HTTP endpoints",
        "start_url": "https://httpbin.org/",
        "success_criteria": {"status_code": 200}
    },
    {
        "task_id": "task_3",
        "intent": "Navigate to example.org",
        "start_url": "http://example.org",
        "success_criteria": {"has_content": True}
    },
    {
        "task_id": "task_4",
        "intent": "Test HTML rendering",
        "start_url": "https://httpbin.org/html",
        "success_criteria": {"has_html": True}
    },
    {
        "task_id": "task_5",
        "intent": "Navigate to example.net",
        "start_url": "http://example.net",
        "success_criteria": {"has_content": True}
    }
]

with open('webarena_task.json', 'w') as f:
    json.dump(sample_tasks, f, indent=2)
print("✓ Sample task file created\n")

# 7. Verify Setup
print("="*60)
print("SETUP VERIFICATION")
print("="*60)
print(f"✓ Chrome/ChromeDriver: {'OK' if chrome_ok else 'FAILED'}")
print(f"✓ OpenAI API Key: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
print(f"✓ Anthropic API Key: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
print(f"✓ Python packages: OK")
print()

if chrome_ok and (os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')):
    print("🎉 Setup complete! You can now run the evaluation.")
    print()
    print("Run the evaluation with:")
    print("  !python complete_rebuttal_eval.py --tasks 5 --trials 5")
else:
    print("⚠️  Setup incomplete. Please check the errors above.")
```

## Quick Start Commands

After running the setup block above, use these commands:

### 1. Quick Test (5 tasks, 5 trials)
```python
!python complete_rebuttal_eval.py --tasks 5 --trials 5
```

### 2. Full Evaluation (25 tasks, 10 trials)
```python
!python complete_rebuttal_eval.py --tasks 25 --trials 10
```

### 3. Custom Configuration
```python
!python complete_rebuttal_eval.py --tasks 10 --trials 5 --output my_results
```

## Alternative: Direct API Key Setting

If you prefer to set keys directly in code (less secure):

```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-...'
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
```

## Troubleshooting

### Chrome Issues
If Chrome fails:
```python
# Reinstall Chrome and ChromeDriver
!apt-get purge -y chromium-browser chromium-chromedriver
!apt-get update
!apt-get install -y chromium-browser chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
```

### DNS Resolution Errors (ERR_NAME_NO)
```python
# Test DNS resolution
import socket
try:
    socket.gethostbyname('example.com')
    print("✓ DNS working")
except:
    print("✗ DNS failed - network connectivity issue")
```

### Memory Issues
```python
# Monitor memory usage
!free -h

# Clear memory if needed
import gc
gc.collect()
```

## Expected Output

After successful setup and run, you should see:

```
COMPLETE WEBARENA EVALUATION FOR REBUTTAL
============================================================

Configuration:
  Tasks: 5
  Trials per task: 5
  Total evaluations: 25 per agent
  Output directory: webarena_results

📂 Loading tasks from webarena_task.json...
✓ Loaded 5 WebArena tasks

🤖 Initializing agents...
✓ GPT-4-Multi5 initialized with 5/5 agents
✓ Claude-Sonnet-4-Multi5 initialized with 5/5 agents
✓ Claude-Haiku-3.5-Multi5 initialized with 5/5 agents
✓ LCAAgent-Multi5 initialized with 5/5 agents

✓ 4 agents ready:
   - GPT-4-Multi5
   - Claude-Sonnet-4-Multi5
   - Claude-Haiku-3.5-Multi5
   - LCAAgent-Multi5

READY TO START
============================================================
```

## File Structure

After running, you'll have:
```
webarena_results/
├── full_results_YYYYMMDD_HHMMSS.csv
├── full_results_YYYYMMDD_HHMMSS.json
├── summary_YYYYMMDD_HHMMSS.json
├── descriptive_statistics.csv
└── effect_sizes.csv
```

## View Results in Colab

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('webarena_results/full_results_*.csv')  # Use actual filename

# Summary statistics
print(df.groupby('agent')[['success', 'time', 'quality']].mean())

# Plot success rates
df.groupby('agent')['success'].mean().plot(kind='bar')
plt.title('Success Rate by Agent')
plt.ylabel('Success Rate')
plt.show()

# Plot execution times
df.groupby('agent')['time'].mean().plot(kind='bar')
plt.title('Average Execution Time by Agent')
plt.ylabel('Time (seconds)')
plt.show()
```
