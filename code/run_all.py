"""One command for all CPU results: python code/run_all.py
verification -> experiments -> LLM plan analysis (if results/llm/*.jsonl exist) -> figures.
LLM answers themselves are produced by code/llm_bidders.py (CPU) or kaggle/job (GPU); see README/STATUS."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ("test_engine.py", "verify.py", "experiments.py", "llm_analysis.py", "figures.py"):
    print(f"== {script}", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
