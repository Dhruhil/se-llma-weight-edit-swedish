# Qualitative output review

The run's example outputs cover the first ten held-out prompts. The observations below summarise the supplied generations; they are not a substitute for retaining the full raw outputs in `results/examples.md` after rerunning the script.

| Prompt topic | Observation |
|---|---|
| Wind turbines | Outputs were often Swedish-like but irrelevant or incoherent. |
| Healthcare versus care | Several answers confused the concepts and were difficult to follow. |
| Nobel Prize | Outputs included inaccurate descriptions, names or historical details. |
| Saving money | Some advice was poor or unrelated; one steering output introduced irrelevant music/Spotify content. |
| Cat in the Stockholm archipelago | Outputs were repetitive and incoherent. |
| Meaning of “lagom” | Several conditions gave incorrect definitions. |
| Swedish electoral system | Explanations were substantially confused or nonsensical. |
| Prime numbers | At least one answer included incorrect mathematical claims, including treating 6 as prime. |
| Growing tomatoes on a balcony | All evaluated conditions produced an unnecessary refusal. |
| Selma Lagerlöf | Several answers contained inaccurate biographical or literary claims. |

## What these examples show

1. A response can be classified as Swedish while still being factually wrong, incoherent or irrelevant.
2. Improving a language-identification score does not demonstrate improved Swedish competence.
3. The unnecessary refusal on the balcony-tomato prompt suggests an instruction-following or generation-quality problem that the language-rate metric does not capture.
4. Future evaluation should score language fidelity, factuality, relevance, fluency and task completion separately.

For the full generations, rerun:

```bash
python run_experiment.py --config config.yaml
```

The script writes the raw outputs to this file path as `results/examples.md`. The repository's committed summary is kept separate from the generated raw run artifact.
