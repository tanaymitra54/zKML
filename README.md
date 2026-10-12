# dpzkml

Reference implementation for *Zero-Knowledge Machine Learning with Differential
Privacy* (originally an analytical-only design study). See `SPEC.md` for
interfaces, acceptance numbers, and track ownership.

Quick start:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m analysis.figures        # regenerates results/fig3..fig9
```
