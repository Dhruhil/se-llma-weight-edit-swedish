# Held-out experiment results

**Model:** Qwen/Qwen2.5-0.5B-Instruct  
**Test prompts:** 30  
**Generation:** greedy, max_new_tokens=100, seed=0

## Main comparison

| Method | Swedish | English | Other detected languages | Swedish rate | 95% Wilson CI |
|---|---:|---:|---:|---:|---:|
| Baseline | 26 | 3 | 1 Danish | 86.7% | 70.3–94.7% |
| Swedish system prompt | 30 | 0 | 0 | 100.0% | 88.7–100% |
| Rank-one edit | 27 | 2 | 1 Danish | 90.0% | 74.4–96.5% |
| Random-direction edit | 26 (each of 5 runs) | 3 (each) | 1 Danish (each) | 86.7% mean | Each run 26/30 |
| Activation steering | 27 | 2 | 1 Bokmål | 90.0% | 74.4–96.5% |

## Perplexity

| Method | Swedish PPL | Change vs baseline | English PPL | Change vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 89.6397 | Reference | 23.5300 | Reference |
| Rank-one edit | 93.3528 | +4.1% | 23.5333 | +0.0% |
| Random-direction edit, mean of 5 | 89.5616 | -0.1% | 23.5206 | -0.0% |
| Activation steering | 86.1550 | -3.9% | 24.7239 | +5.1% |

The prompt-conditioned perplexity was not separately measured. The raw run output's perplexity calculation is unconditioned, so baseline perplexity values should not be presented as prompt-conditioned results.

## Selected development configurations

- Rank-one edit: layers 12–17, centre layer 15, lambda = 0.5. Development language detection: 30/30 Swedish.
- Activation steering: layer 19, alpha = 0.1. Development language detection: 30/30 Swedish.
- Development baseline: 29/30 Swedish (96.7%).

Stronger settings could cause severe regressions. For example, the layers 16–21 edit with lambda = 1.0 produced 1/30 Swedish responses and Swedish PPL 149.659. Steering alpha = 0.4 also caused very large English perplexity increases in parts of the development grid.

## Interpretation

The targeted edit gained one Swedish-classified response relative to baseline. Five random edits matched baseline language counts. The Swedish system prompt achieved 30/30. The confidence intervals are broad; because conditions share the same prompts, prompt-level paired outcomes are needed for a valid paired significance test. These aggregate counts do not establish a statistically reliable improvement.

Language identification is not a quality metric. See [examples.md](examples.md) for qualitative failure cases.
