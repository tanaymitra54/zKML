# Demo layer — what to show, and what it honestly proves

A thin choreography over the frozen SPEC interfaces, built for the IDF pitch
video. It adds no protocol logic: `dp/`, `quant/` and `analysis/` still own
every guarantee. This layer only makes one release **watchable** — input,
process, output, and the four attacks that must fail.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r demo/requirements-demo.txt

python -m demo.run_release      # terminal, ~0.3 s
streamlit run demo/app.py       # browser console, four panels
pytest -q                       # 23 passed
```

## What is real, and what is not

| On screen | Status |
|---|---|
| Commitments `C_D`, `C_θ`, `C_ρ`, `C_y` | **real** — SHA-256 over the actual bytes |
| Quantize → clip → aggregate `z` | **real** — `quant/reference_query.py` |
| Noise `η` and excluded tail mass | **real** — `dp/sampler.py`, exact finite-support pmf |
| ε interval, enforced on the upper end | **real** — `dp/accountant.py` |
| Ledger chaining, replay / stale / branch rejection | **real** — `dp/ledger.py` |
| Artifact tamper detection | **real** — hash binding fails on one flipped integer |
| Seed `C_ρ` re-opening | **witness withheld** — only a proof can substitute |
| "proof backend binds the relation" | **NOT BOUND** — Track D, the part this grant funds |

The last two deliberately print `OPEN` / `NOT BOUND` instead of a green tick.
Never claim otherwise on camera: for a development fund, an honest TRL 4 with a
named missing component is stronger than an inferred one.

## Terminal settings for recording

- Font 20 pt, dark theme, window ≈ 1600×900 with generous margins
- Prompt shortened to `$`, `cd` to the repo root **before** recording
- Run it once first so imports are warm — the scripted run is ~0.3 s, so the
  output arrives at once and you narrate over it rather than waiting
- `--plain` gives dependency-free output; `--no-attacks` shows the happy path only

## Beat sheet against the 5-minute video

| Video section | Time | What to put on screen |
|---|---|---|
| 1 Introduction | 0:00–0:25 | `1/5` header: title, problem statement |
| 2 Solution overview | 0:25–1:00 | the three guarantees; the one-sentence novelty |
| 3 Core demo | 1:00–3:05 | `1/5` → `2/5` → `3/5` → `4/5`, then the four BLOCKED panels |
| 4 Technical details | 3:05–3:45 | `5/5` verify list + `analysis/figures` panels (5, 6, 7) |
| 5 Applications | 3:45–4:10 | Streamlit ①/② tabs, target sectors |
| 6 Commercial ask | 4:10–4:40 | Streamlit ③ budget panel + the `NOT BOUND` line |

The Streamlit console is the "UI/UX walkthrough" the rubric asks for; the
terminal run is the "live demo of a working model". Record both from the same
session so the numbers agree.

## Suggested submission filename

```
Karthikeyan PR_SELECT_DP-zkML - Verifiable Differentially-Private Inference.mp4
```

No `/` in the filename — the slash in the guideline is a path separator.
