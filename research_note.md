# Can a rank-1 weight edit fix Swedish language fidelity in a small LM?

*Candidate assignment, SE.LLMA — Dhruhil — 19 Nov 2026*

## 1. Critical reading of "Sparks of Weight-Space Programming"
**Strongest evidence.** (i) Experiment 3: from only the four middle transformer blocks of three
character-level LMs (no input/output layers, no forward passes), the model names the training
language and exactly the induction heads that were measured independently by running the networks.
This is the cleanest case of genuinely reading weights. (ii) Experiment 8: the edited
Qwen3.5-0.8B was released and a third party (Hugging Face) rebuilt it from the logs and re-ran the
evaluation, so the headline number is reproducible.

**Main limitations.**
1. *Reasoning vs. feedback.* The Qwen edit used a Python kernel plus up to 50 forward passes on
   practice problems. The gain is not smooth in that budget: MATH-500 was 12.3% (10 passes) and 14.1%
   (25 passes), and only reached 48.1% at 50 passes (base 7.9%). This looks like iterative,
   feedback-guided search, not one-shot reading of weights. Experiments 4 and 5 (no tools / no
   rollouts) are closer to "pure reasoning", but they are small toy networks.
2. *The edit is mundane.* It subtracts one direction from the MLP output columns of two layers
   (0.016% of weights). A rank-1 projection of this kind is mathematically a special case of
   activation steering/direction ablation (a post by Wang et al. on LessWrong makes this point), so
   it does not show that weight-space editing is a new mechanism.
3. *Failure removal, not capability gain.* 93% of base answers hit the token limit (thinking loops).
   Fixing loops lets the model finish answers it could already produce; on Gemma 3 270M, which rarely
   loops, no edit moved MATH-500 by more than about a point (6.8 -> ~7.7). The authors state this
   themselves. Baselines such as non-thinking mode or loop-avoiding decoding were not compared.
4. *Narrow evidence base.* One closed frontier model, blog post (no peer review), small models, a
   single task per claim, and a negative result on identifying a sycophantic model from weights alone
   (it needed attention patterns from forward passes it designed).

**Justified:** a frontier LLM can sometimes find and remove a *single, localised* fault in a small
model, and can read coarse structure from weights. **Not justified:** that LLMs "natively" optimise
weights better than standard methods, that edits add capabilities, or that auditing from weights
alone works in general (sycophancy was invisible).

## 2. Hypothesis
SE.LLMA builds Swedish models on open models, so cheap, targeted fixes and audits are of interest.
**H:** A rank-1 weight edit derived from a Swedish-vs-English contrast increases the share of
Swedish-language answers to Swedish prompts (a *failure-removal* behaviour), without raising Swedish or
English perplexity by more than 10%, and beats random-direction edits and is competitive with
prompting and steering. **Supports H:** edit > random edit with non-overlapping CIs, perplexity
within the limit. **Challenges H:** edit ≈ random edit, edit < prompt baseline, or fidelity gains
come with a perplexity regression, or baseline has no failure to fix.

## 3. Experiment
Model `[MODEL @ COMMIT]`, greedy decoding, 100 new tokens, seed 0. Direction: mean residual-stream
difference between 20 Swedish sentences and their English translations at a middle layer. Edit:
`W' = (I - λ d dᵀ) W` on `mlp.down_proj` across a layer range. Controls on **test** (30 prompts, never
used for selection): unchanged model; system prompt "Svara alltid på svenska."; 5 random unit
directions with the same layers/λ; additive steering at one layer (conventional alternative).
Layer range and strength (λ, α) chosen on **dev** (30 prompts + half of perplexity texts) with the
rule: highest Swedish rate subject to Swedish perplexity ≤ 1.10× baseline. Language detection:
`lingua` (10 candidate languages). Code and data: this repository.

## 4. Results
<!-- Copy from results/results.json -> "test". Keep the CIs. -->
| Method | Swedish rate (95% CI) | Swedish ppl | English ppl |
|---|---|---|---|
| Baseline | [FILL] | [FILL] | [FILL] |
| Prompt "Svara på svenska" | [FILL] | – | – |
| Rank-1 weight edit (layers [FILL], λ=[FILL]) | [FILL] | [FILL] | [FILL] |
| Random-direction edit (mean ± sd, n=5) | [FILL] | [FILL] | [FILL] |
| Additive steering (layer [FILL], α=[FILL]) | [FILL] | [FILL] | [FILL] |

Examples (from `results/examples.md`): [FILL: 2-3 prompts where baseline answered in the wrong
language and what the edit / prompt produced; include one failure or degraded output].

## 5. Interpretation
<!-- Pick the sentence that matches your numbers; delete the rest. -->
- *If the edit clearly beats random and baseline:* "On this model, a one-direction edit raised Swedish
  fidelity from X to Y (CI ..), while random edits did not, and perplexity changed by Z%. This is
  failure removal: it shows the model already could answer in Swedish." Say that prompting [did / did
  not] match it, and that the edit is equivalent to a steering vector, not evidence of LLM-style
  weight reasoning (no LLM was used to choose the edit).
- *If the edit ≈ random or < prompt:* "The effect is not distinguishable from untargeted edits; a one-line
  system prompt is cheaper and at least as effective. The claim did not transfer to this setting."
- *If baseline ≈ 100% Swedish:* "There was no failure to fix on this model; the test is uninformative.
  Reran with [smaller model]."
**Limits:** one small model, 30 test prompts (CIs are wide), language-ID errors (Danish/Norwegian
confusion), tiny perplexity set, one direction-estimation method, no LLM-proposed edit, and fidelity
is not Swedish *quality* (grammar, culture, factuality).

## 6. Recommendation
[FILL, e.g.] Do not invest in "LLM edits weights" as a method for SE.LLMA's core goal (data and
continued pre-training remain the main lever for Swedish quality), but keep cheap direction-level
edits/steering in the toolbox for narrow failures (language drift, loops) and as audit probes.
**Most informative next experiment:** repeat with a larger open model on a real Swedish benchmark
(language fidelity *and* a competence benchmark), and add an arm where a frontier LLM proposes the
edit from weights only (0 forward passes) vs. with N forward passes, scoring each against a
conventional LoRA fine-tune on the same budget. That directly separates "reasoning about weights"
from "feedback-guided search".
