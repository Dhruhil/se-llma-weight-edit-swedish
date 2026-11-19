# Swedish Language Fidelity Through Targeted Weight Editing

**SE.LLMA Candidate Assignment | Dhruhil Patel | 19 November 2026**

A controlled experiment testing whether a small rank-one weight edit can improve the Swedish-language response rate of an open-weight language model, inspired by *Sparks of Weight-Space Programming*.

**Main finding:** On 30 held-out prompts, the rank-one edit increased Swedish-language detection from 86.7% to 90.0% (one additional response). Five random-direction edits matched the baseline rate, while a Swedish system prompt reached 100%. Qualitative inspection showed that Swedish-language detection did not imply good Swedish answers or factual accuracy.

## Start Here

- **[Research note (PDF)](research_note.pdf)** – Final research note.
- **[Research note (Markdown)](research_note.md)** – Research note in Markdown.
- **[Results summary](results/summary.md)** – Held-out metrics and interpretation.
- **[Qualitative examples](results/examples.md)** – Examples of output quality and failure cases.
- **[References](docs/REFERENCES.md)** – Original research, model and related work.
- **[AI usage disclosure](AI_USAGE.md)** – Transparent disclosure of AI assistance.

## Research Question

Can a targeted weight edit improve the consistency of Swedish responses compared with an unchanged model and random weight edits, without substantially damaging language modelling?

## Experimental Design

The experiment uses `Qwen/Qwen2.5-0.5B-Instruct` and compares five conditions:

1. Unchanged baseline model.
2. Rank-one targeted weight editing.
3. Random-direction weight editing (five runs).
4. Additive activation steering.
5. Swedish-language system prompting.

A Swedish-English direction is estimated from 20 parallel sentence pairs. The experiment uses 30 development prompts for configuration selection and 30 separate held-out test prompts for final evaluation.

The primary metric is the proportion of generated responses classified as Swedish. Swedish and English perplexity are supplementary measures.

## Key Held-Out Results

| Method | Swedish responses | Swedish perplexity | English perplexity |
|---|---:|---:|---:|
| Unchanged baseline | 26/30 (86.7%) | 89.64 | 23.53 |
| Swedish system prompt | 30/30 (100%) | Not evaluated | Not evaluated |
| Rank-one weight edit | 27/30 (90.0%) | 93.35 | 23.53 |
| Random-direction edit (5-run mean) | 26/30 (86.7%) | 89.56 | 23.52 |
| Additive activation steering | 27/30 (90.0%) | 86.16 | 24.72 |

The rank-one edit produced only one additional Swedish response compared with the baseline. It did not clearly outperform random edits or match the simple Swedish system prompt. These results do not establish a reliable improvement in Swedish-language capability.

The intervention was not an autonomous LLM weight-reasoning procedure. The direction and edit were specified algorithmically.

## Repository Structure

```text
se-llma-weight-edit-swedish/
├── README.md
├── AI_USAGE.md
├── config.yaml
├── requirements.txt
├── run_experiment.py
├── make_tables.py
├── main.tex
├── references.bib
├── research_note.md
├── research_note.pdf
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
    ├── examples.md
    └── results.json
```

## Requirements and Reproduction

Use Python 3.10 or later.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python run_experiment.py --config config.yaml
```

Generate the result tables and plots:

```bash
python make_tables.py
```

The experiment uses Qwen2.5-0.5B-Instruct with the `main` model revision and random seed 0. The resolved model commit hash was not recorded for the reported run, which is a reproducibility limitation.

## Scope and Limitations

- The development and test sets each contain only 30 prompts.
- The language direction was estimated from 20 sentence pairs.
- The random-edit control contains five runs.
- Language classification measures output language, not correctness, fluency or usefulness.
- Perplexity does not establish factual accuracy or answer quality.
- The model revision was configured as `main`, and the resolved commit hash was not recorded.
- The experiment is a manually specified activation-derived weight intervention, not a complete autonomous implementation of Weight-Space Programming.

See the research note for the full methodology, results, critical discussion and limitations.

## AI Usage

See [AI_USAGE.md](AI_USAGE.md) for details of the AI tools used and the author's contributions.

## References

See [docs/REFERENCES.md](docs/REFERENCES.md) for the original Weight-Space Programming article, model card, related research and assignment context.