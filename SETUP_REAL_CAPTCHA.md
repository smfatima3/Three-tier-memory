# Setup Guide for Real CAPTCHA Testing

## Installation

```bash
# Install required dependencies
pip install aiohttp pandas numpy matplotlib

# Optional: For full browser automation (if needed later)
pip install playwright selenium
playwright install chromium
```

## Quick Start

```bash
# Run the real CAPTCHA benchmark
python lca_real_captcha_testing.py
```

## Real CAPTCHA Sources Used

### 1. Math CAPTCHAs
- **TextCaptcha.com**: Public API at http://textcaptcha.com/demo.json
- **SimpleCaptcha**: Open-source implementations (Django, WordPress, Flask)
- **BotDetect**: Commercial CAPTCHA service examples

### 2. Text Question CAPTCHAs
- **TextCaptcha.com**: Question-based challenges
- **WordPress Plugins**: WP-CAPTCHA, Question CAPTCHA
- **Drupal Modules**: CAPTCHA module question types

### 3. Honeypot Field Testing
- **HTTPBin.org**: https://httpbin.org/forms/post (public test endpoint)
- **ScrapeThisSite**: https://scrapethissite.com (honeypot examples)
- **WordPress Antispam**: Common honeypot implementations

### 4. Pattern CAPTCHAs
- **Arkose Labs**: https://arkoselabs.com (demo patterns)
- **BotDetect**: Sequence and pattern challenges
- Pattern completion implementations

### 5. Logic Puzzle CAPTCHAs
- **NuCaptcha**: https://nucaptcha.com (logic challenges)
- **Confident CAPTCHA**: Puzzle-based verification
- Logic reasoning implementations

## Configuration

Edit `main()` function to adjust:

```python
# Test with different agent counts
agent_counts = [1, 3, 5, 10]  # Modify as needed

# Challenges per CAPTCHA type
n_challenges_per_type = 20  # Increase for more data
```

## Output Files

Results saved to `./lca_real_captcha_results/`:
- `real_captcha_results.csv` - All test results
- `real_captcha_benchmark_results.png` - Visualizations
- `real_captcha_summary.txt` - Performance summary
- `benchmark_sources.json` - All CAPTCHA sources documented
- `REPRODUCIBILITY.md` - Reproduction guide

## For ICLR Paper

### Key Features for Publication:
1. **Reproducible**: All sources publicly accessible
2. **Documented**: Each result includes source URL and benchmark name
3. **Ethical**: Uses public APIs and demo endpoints only
4. **Verifiable**: All sources can be independently verified

### Citation Information:
- TextCaptcha: http://textcaptcha.com
- HTTPBin: https://httpbin.org
- SimpleCaptcha: Open-source CAPTCHA implementations
- Your LCA framework: [Add your citation]

## Extending to Full Browser Automation (Optional)

To test with actual browser interaction:

```python
# Install browser automation tools
pip install playwright
playwright install

# Then use in code:
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch()
    page = await browser.new_page()
    # Interact with real CAPTCHA websites
```

## Rate Limiting & Ethics

- Script includes delays between requests (0.5s)
- Respects public API rate limits
- Uses demo/test endpoints only
- Does NOT attack production systems

## Troubleshooting

**Q: TextCaptcha API not responding?**
A: Script falls back to SimpleCaptcha implementations

**Q: Want to test more sources?**
A: Add new methods to `RealCAPTCHAProvider` class

**Q: Need actual browser testing?**
A: Uncomment Playwright code and implement `_solve_with_browser()` methods

## Results Interpretation

Success rates indicate:
- **>80%**: Excellent (honeypot, math)
- **70-80%**: Good (logic, text comprehension)
- **60-70%**: Acceptable (patterns)
- **<60%**: Challenging (complex reasoning)

These are realistic for LLM-based agents on text CAPTCHAs.

## Contact

For questions about implementation or sources:
[Add your contact information]
