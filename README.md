# Can a rank-1 weight edit fix Swedish language fidelity? (SE.LLMA candidate assignment)

A small experiment testing one claim from *Sparks of Weight-Space Programming* (Takehi et al., 2026):
that a tiny, targeted weight edit can remove a specific failure of a small LM without damaging
other behaviour. Here the failure is "answers a Swedish prompt in another language".

See `research_note.md` (and `report.pdf`) for hypothesis, results and recommendation.

## Reproduce (about 15-30 min on a free Colab T4; CPU also works but is slow)
```bash
pip install -r requirements.txt
python run_experiment.py --config config.yaml
```
Outputs: `results/results.json` (all numbers, selected configuration, model commit hash) and
`results/examples.md` (side-by-side generations for 10 test prompts).

## Design in one paragraph
Baseline -> grid search on **dev** prompts/perplexity texts (layer range, edit strength) -> frozen
choice -> single evaluation on **test** prompts. Methods compared on test: unchanged model,
prompt ("Svara alltid på svenska."), rank-1 weight edit, random-direction edits (5 seeds, same
layers/strength), additive activation steering. Target metric: share of responses detected as
Swedish (`lingua`, 95% Wilson CI). Regression checks: Swedish and English perplexity on held-out
passages.

## Layout
- `data/` : 30 dev + 30 test Swedish prompts, 20 parallel sv/en sentence pairs (used only to
  estimate the direction), 10 sv + 10 en short passages for perplexity (first half dev, second half
  test). All written for this project, free to share.
- `config.yaml` : model, revision, seed, grids. **Pin `revision` to the commit hash printed by the script.**
- `run_experiment.py` : whole pipeline.

## Notes / limits
- If the baseline is already ~100% Swedish there is no failure to fix: switch `model_name` in
  `config.yaml` to a smaller model (e.g. `google/gemma-3-270m-it`) and re-run.
- Only 30 test prompts: effect sizes below ~20 percentage points are not distinguishable from noise
  (see CIs). Perplexity sets are tiny; replace with a larger permissively licensed Swedish sample
  (e.g. Swedish Wikipedia) if time allows.
