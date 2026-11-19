# Testing Targeted Weight Editing for Swedish Language Fidelity

**Candidate assignment:** SE.LLMA — AI Training Application Expert for Large Language Models  
**Author:** Dhruhil Patel  
**Date:** 19 November 2026

## Abstract

This note tests whether a rank-1 weight edit derived from a Swedish–English activation contrast can improve Swedish-language response fidelity in Qwen2.5-0.5B-Instruct. The experiment compares an unchanged baseline, a Swedish system prompt, a targeted weight edit, five random-direction edits and additive activation steering. On 30 held-out prompts, the baseline produced 26 Swedish responses (86.7%), while the selected weight edit produced 27 (90.0%). The random-edit control averaged the same 86.7%, and a simple Swedish system prompt produced 30/30 Swedish responses. The targeted edit increased Swedish perplexity by approximately 4.1%, while English perplexity was effectively unchanged. Qualitative examples reveal substantial factual and semantic errors even when outputs are classified as Swedish. The results therefore provide no convincing evidence that this targeted edit improves Swedish capability. They illustrate the importance of controls, held-out evaluation and measuring answer quality separately from language identity.

## 1. Critical reading of “Sparks of Weight-Space Programming”

The article by Takehi, Teoh, Zhang and Khattab introduces Weight-Space Programming as the idea that a general-purpose language model can inspect and directly modify neural-network weights in response to a natural-language objective, rather than relying on gradient descent or a conventional training loop.

### Strongest evidence

Two parts of the article are particularly interesting.

1. **Reading information from weights.** In the reported language-identification experiment, the model was given middle transformer blocks from small character-level language models and inferred properties of their training data and internal copying behaviour. This is the clearest evidence in the article for interpreting weights without relying on ordinary task-level generation.
2. **A consequential edit with external verification.** The Qwen3.5-0.8B experiment reports a substantial improvement on MATH-500 after a small, localised weight edit. The authors also released experiment logs, and a third party reconstructed the edited model and evaluated it. This makes the result more inspectable than an unsupported headline claim.

### Main limitations and alternative explanations

1. **Reasoning from weights versus feedback-guided search.** The Qwen experiment used a Python environment and repeated forward-pass feedback on practice problems. Its reported performance increased with the available evaluation budget. This is evidence for iterative diagnosis and search, but it is not the same as inferring a correct edit from weights alone. The article's toy experiments without rollouts are closer to the stronger claim of direct weight reasoning, but they are narrower.
2. **The edit is a familiar low-rank operation.** Removing or changing a direction in a layer's output can be expressed as a rank-one projection. Related activation-steering and direction-ablation work shows that low-dimensional directions can influence model behaviour. The interesting question is therefore not simply whether a rank-one edit works, but whether an LLM can identify a useful edit reliably and outperform simpler methods.
3. **Repairing a failure is not necessarily adding a capability.** If a model already has the relevant capability but fails to express it because of a local defect, removing that defect can improve benchmark performance without teaching the model new knowledge or skills.
4. **The evidence is early and task-specific.** The article is an early research blog post rather than a peer-reviewed paper. The reported experiments cover a limited number of architectures and tasks. Results on one model or benchmark should not be treated as evidence that the approach generalises broadly.
5. **Weight-only auditing has limits.** The reported difficulty identifying sycophantic behaviour from weights alone, and the use of additional activation information, are important qualifications to claims about inspecting a model without running it.

**Assessment.** The article provides promising evidence that frontier LLMs can sometimes interpret and edit neural-network parameters in useful ways. It does not establish that LLM-directed editing is generally more reliable or efficient than conventional steering, fine-tuning or controlled optimisation. In particular, iterative feedback should be distinguished from genuinely reasoning about weights without task feedback.

## 2. Hypothesis

A small language model may sometimes answer a Swedish prompt in another language. A targeted edit based on a Swedish–English representation contrast could reduce this language drift.

**Hypothesis (H):** A rank-1 weight edit derived from a Swedish-minus-English activation direction will increase the proportion of Swedish responses on held-out Swedish prompts, while keeping Swedish and English perplexity changes within a pre-defined 10% tolerance. The targeted edit should also outperform same-sized random-direction edits and be competitive with a Swedish system prompt and conventional activation steering.

The hypothesis is challenged if the targeted edit performs similarly to random edits, performs worse than prompting, or produces a meaningful perplexity regression. A baseline already close to 100% Swedish would also leave little room for improvement.

## 3. Experimental design

### 3.1 Model and data

- **Model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- **Generation:** greedy decoding, maximum 100 new tokens, seed 0.
- **Prompts:** 30 development prompts and 30 held-out test prompts.
- **Direction data:** 20 Swedish sentences paired with English translations.
- **Perplexity data:** short Swedish and English text samples, split into development and test portions by the script.

The configuration uses `revision: main`. The run output reported `commit: null`, so the exact resolved Hugging Face model revision was not recorded. This is a reproducibility limitation; no commit hash is inferred here.

### 3.2 Direction estimation and weight edit

For a selected transformer layer, the experiment computes mean residual-stream representations for the Swedish sentences and their English translations. The normalised difference defines a unit direction:

[
d = \frac{\mu_{sv}-\mu_{en}}{\|\mu_{sv}-\mu_{en}\|_2}.
]

The experiment applies a rank-one edit to the MLP `down_proj` weights:

[
W' = W - \lambda d(d^\top W)
]

where \(\lambda\) controls the edit strength. This operation attenuates the component of the MLP output weights along the selected direction. It is a manually specified, activation-derived intervention. It is **not** an LLM autonomously inspecting weights, proposing edits and improving them through repeated feedback.

### 3.3 Conditions and controls

The comparison includes:

- **Baseline:** unchanged model.
- **Prompting:** the system instruction `Svara alltid på svenska.`
- **Targeted edit:** the selected rank-one weight edit.
- **Random edit:** five random unit directions using the selected layers and edit strength.
- **Activation steering:** additive intervention using the Swedish–English direction at one selected layer.

The layer range and edit strength were selected using development prompts and development perplexity data. The selected configuration was then evaluated on the held-out test prompts.

### 3.4 Metrics

1. **Swedish response rate:** the proportion of outputs classified as Swedish by the Lingua language detector. The report includes 95% Wilson confidence intervals.
2. **Language counts:** outputs classified as Swedish, English, Danish, Bokmål or other detected languages.
3. **Perplexity:** Swedish and English text perplexity as a regression check.
4. **Qualitative inspection:** review of generated outputs to identify factual, semantic, fluency and instruction-following failures.

Language detection is only a proxy for language fidelity. It does not establish that a response is correct, fluent or relevant.

## 4. Results

### 4.1 Development-set selection

The development baseline classified 29/30 responses as Swedish (96.7%). The selected rank-one edit used layers 12–17, centred at layer 15, with \(\lambda=0.5\). The selected activation-steering condition used layer 19 with \(\alpha=0.1\). Both selected conditions reached 30/30 Swedish detections on the development prompts.

The grid search also showed that stronger interventions could be harmful. For example, a rank-one edit over layers 16–21 with \(\lambda=1.0\) produced only 1/30 Swedish responses and substantially increased Swedish perplexity. Strong activation steering also caused very large English-perplexity increases at some tested strengths. These results demonstrate sensitivity to layer choice and intervention strength.

### 4.2 Held-out test results

| Method | Swedish responses | 95% Wilson CI | Swedish PPL | English PPL |
|---|---:|---:|---:|---:|
| Baseline | 26/30 (86.7%) | [70.3%, 94.7%] | 89.64 | 23.53 |
| Swedish system prompt | 30/30 (100%) | [88.7%, 100%] | Not measured conditionally | Not measured conditionally |
| Rank-one edit, layers 12–17, \(\lambda=0.5\) | 27/30 (90.0%) | [74.4%, 96.5%] | 93.35 | 23.53 |
| Random-direction edit, mean of 5 runs | 26/30 (86.7%) | Each run: 26/30 | 89.56 | 23.52 |
| Activation steering, layer 19, \(\alpha=0.1\) | 27/30 (90.0%) | [74.4%, 96.5%] | 86.16 | 24.72 |

The prompt-conditioned perplexity values are marked as not measured because the reported perplexity calculation was unconditioned; copying the baseline perplexity into the prompt row would be misleading.

### 4.3 Perplexity changes

Relative to the baseline:

- The targeted weight edit increased Swedish perplexity from 89.64 to 93.35, approximately **4.1%**.
- English perplexity changed from 23.53 to 23.53, approximately **0.0%** at the displayed precision.
- Activation steering reduced Swedish perplexity by approximately **3.9%**, but increased English perplexity by approximately **5.1%**.
- The five random edits had mean perplexities close to baseline.

The weight edit stayed within the pre-defined 10% Swedish-perplexity tolerance, but satisfying that constraint does not establish a meaningful improvement in Swedish capability.

### 4.4 Qualitative inspection

The generated examples reveal a major gap between language identification and answer quality:

- Several outputs were recognisably Swedish but irrelevant, repetitive or semantically incoherent.
- Answers about *lagom*, the Swedish electoral system, the Nobel Prize and Selma Lagerlöf contained substantial inaccuracies or confused explanations.
- The prime-number example included incorrect mathematical claims.
- The balcony-tomato prompt received an unnecessary refusal under all evaluated conditions.
- The wind-turbine, healthcare and money-saving examples also showed poor relevance or content quality.

These examples are important negative evidence. A language detector can classify an answer as Swedish even when the answer is factually wrong or not useful.

## 5. Interpretation

The targeted rank-one edit increased the Swedish detection rate by one response, from 26/30 to 27/30. The five random-direction edits averaged exactly the baseline language rate. Activation steering also reached 27/30, while the simple Swedish system prompt reached 30/30.

The observed difference between baseline and targeted edit is small. The confidence intervals are wide and overlap. Because the same prompts were evaluated under each condition, a paired test such as McNemar's test would be more appropriate than treating the conditions as independent samples. The aggregate counts alone are insufficient to calculate that paired test; prompt-level paired outcomes would be required.

The targeted edit's Swedish perplexity increase was modest and within the pre-defined tolerance, but the edit did not clearly beat the random control. It also did not match the prompt baseline. Therefore, the experiment does **not** provide convincing evidence that this targeted weight edit improves Swedish language fidelity.

More importantly, the intervention does not reproduce the central agentic mechanism in the Weight-Space Programming article. No LLM was asked to inspect the model's weights, reason about the cause of the failure, propose an edit, observe feedback and revise its edit. This experiment evaluates a manually designed rank-one weight intervention inspired by the paper.

## 6. Limitations

1. Only one small instruction-tuned model was evaluated.
2. The held-out test set contains 30 prompts, resulting in wide confidence intervals.
3. The perplexity samples are small and cannot represent Swedish language quality broadly.
4. Language identification may confuse closely related languages, including Danish and Norwegian Bokmål.
5. Perplexity does not measure factual accuracy, usefulness, grammar or cultural appropriateness.
6. The model revision was not pinned to a resolved commit hash.
7. The selected edit was derived from a small set of parallel sentences and a limited grid search.
8. The experiment uses a manually specified intervention, not autonomous LLM reasoning about weights.
9. The experiment does not establish whether the intervention generalises to other Swedish tasks, domains or model sizes.

## 7. Recommendation and next experiment

For SE.LLMA, this result is not a reason to prioritise rank-one weight editing as a substitute for high-quality Swedish data, continued pre-training, instruction tuning or rigorous evaluation. Small direction-based interventions may still be useful as diagnostic probes or as tools for narrow, localised failure modes, but their effects should be compared with simple prompting and untargeted controls.

The most informative next experiment would use a stronger Swedish-capable open model and a larger, independently held-out Swedish evaluation set. It should measure both language fidelity and task quality, including factuality and instruction following. A direct comparison should include:

- a conventional prompt baseline;
- an activation-steering baseline;
- a controlled weight-editing method;
- size-matched random edits;
- an LLM-proposed edit using weights only, with zero forward-pass feedback;
- an LLM-proposed edit with a fixed forward-pass feedback budget; and
- a conventional fine-tuning or adapter baseline under a clearly stated compute/data budget.

All methods should be evaluated on the same paired prompts, with exact model revisions, seeds, prompts, edit diffs and raw outputs retained. This would more directly separate reasoning about weights from feedback-guided search.

## References

See [docs/REFERENCES.md](docs/REFERENCES.md) for full bibliographic details and clickable primary-source links.
