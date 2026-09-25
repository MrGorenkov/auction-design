# Auction design: soft close, proxy bidding and sniping

Code and aggregate results for the article "Soft close, proxy bidding and sniping on an online art-auction platform:
an agent-based evaluation with rule-based and LLM bidders" (A. A. Gorenkov, Plekhanov Russian University of
Economics, 2026).

An agent-based model of the bidding engine of the ArtSphere auction platform
(https://github.com/MrGorenkov/ArtSphere-Auction): private-value bidders with proxy, incremental and sniping
strategies under hard close, soft close (extension windows) and proxy bidding. Revenue, allocative efficiency,
sniping share and auction duration are compared over many seeded replications; the engine is checked against revenue
equivalence and Myerson's optimal reserve price. LLM bidders (Qwen3.5 GGUF via llama.cpp) play the same rules.

## Run
```
pip install numpy pandas scipy matplotlib
python code/run_all.py            # engine tests, verification, experiments, LLM analysis, figures (~3 min, CPU)
python code/llm_bidders.py        # LLM bidders on CPU via llama.cpp (options in the module docstring)
python kaggle/make_job.py         # GPU job for the larger models (Kaggle 2x T4)
```
`results/` holds every number reported in the paper, `figures/` the figures.

## Licence
Code: MIT.
