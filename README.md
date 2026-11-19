# Swedish Language Fidelity Through Targeted Weight Editing

**SE.LLMA candidate assignment | Dhruhil Patel | 19 November 2026**

A controlled experiment testing whether a small rank-1 weight edit can improve the Swedish-language response rate of an open-weight language model, inspired by *Sparks of Weight-Space Programming*.

> **Main finding:** On 30 held-out prompts, the rank-1 edit increased Swedish-language detection from 86.7% to 90.0% (one additional response). Five random-direction edits matched the baseline rate, while a Swedish system prompt reached 100%. Qualitative inspection showed that Swedish-language detection did not imply good Swedish answers or factual accuracy.

## Start here

- **[Research note](research_note.md)** — critical assessment, hypothesis, methods, results and conclusions.
- **[Results summary](results/summary.md)** — held-out metrics and interpretation.
- **[Qualitative examples](results/examples.md)** — examples of output quality and failure cases.
- **[References](docs/REFERENCES.md)** — clickable links to the original research, model and related work.
- **[AI-use statement](AI_USAGE.md)** — transparent disclosure of AI assistance.

## Repository structure

```text
se-llma-weight-edit-swedish/
├── README.md
├── research_note.md
├── AI_USAGE.md
├── config.yaml
├── requirements.txt
├── run_experiment.py
├── data/
│   ├── dev_prompts.txt
│   ├── test_prompts.txt
│   ├── direction_pairs.tsv
│   ├── ppl_sv.txt
│   └── ppl_en.txt
├── docs/
│   └── REFERENCES.md
└── results/
    ├── summary.md
    └── examples.md
```

## Experiment at a glance

- **Model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Evaluation:** 30 development prompts and 30 held-out test prompts.
- **Direction:** mean Swedish-minus-English residual-stream contrast from 20 parallel sentence pairs.
- **Interventions:** rank-1 weight edit, additive activation steering and a Swedish system prompt.
- **Controls:** unchanged baseline and five random-direction weight edits.
- **Selection:** intervention settings selected on development data, then evaluated on the held-out test prompts.
- **Generation:** greedy decoding, 100 maximum new tokens, seed 0.

## Reproduce

Use Python 3.10+ and install the listed dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python run_experiment.py --config config.yaml
```

The script writes the raw machine-readable output to `results/results.json` and side-by-side generations to `results/examples.md`. Those runtime outputs are not currently included in this repository snapshot; the committed `results/summary.md` records the results supplied for this research note.

## Key held-out results

| Method | Swedish responses | Swedish perplexity | English perplexity |
|---|---:|---:|---:|
| Unchanged baseline | 26/30 (86.7%) | 89.64 | 23.53 |
| Swedish system prompt | 30/30 (100%) | Not evaluated conditionally | Not evaluated conditionally |
| Rank-1 weight edit | 27/30 (90.0%) | 93.35 | 23.53 |
| Random-direction edit (5-run mean) | 26/30 (86.7%) | 89.56 | 23.52 |
| Additive activation steering | 27/30 (90.0%) | 86.16 | 24.72 |

The rank-1 edit did not clearly outperform random edits and did not match the simple system prompt. The intervention was not an autonomous LLM weight-reasoning procedure: the direction and edit were specified algorithmically.

## Scope and limitations

This is a small, controlled mechanism-level experiment, not a full reproduction of the Weight-Space Programming paper. Language identification measures the language of an output, not its correctness, fluency or usefulness. The test set and perplexity samples are small; the model revision is configured as `main`, and the resolved model commit was not recorded (`commit: null`). See the research note for the full limitations.

## References

See [docs/REFERENCES.md](docs/REFERENCES.md) for the original Weight-Space Programming article, model card, related research and assignment context.
