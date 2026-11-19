# References and official links

This file collects primary-source and project links used in the candidate assignment.

## 1. Weight-Space Programming

Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. (2026, November 2). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL research blog.

- [Original article (author-hosted)](https://rikiyat.github.io/blog/weight-space-programming/)
- [Authors' released experiment logs](https://github.com/RikiyaT/lm_understand-logs)

The article is a research blog post, not a peer-reviewed conference paper as of the cited publication.

## 2. Model used in this experiment

Qwen Team. *Qwen2.5-0.5B-Instruct*.

- [Official Hugging Face model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Qwen2 technical report (arXiv:2407.10671)](https://arxiv.org/abs/2407.10671)

The experiment configuration specifies `revision: main`. The run output supplied for this report has `commit: null`; therefore, the exact model snapshot is not pinned.

## 3. Related work on direction-based interventions

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* Advances in Neural Information Processing Systems 37.

- [NeurIPS proceedings page](https://papers.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html)
- [arXiv preprint](https://arxiv.org/abs/2406.11717)

This work is relevant background for low-dimensional activation directions and model interventions. It studies refusal behaviour, not Swedish language fidelity.

## 4. SE.LLMA assignment context

- [WASP / SE.LLMA position information](https://wasp-sweden.org/positions/ai-training-application-expert-for-large-language-models-se-llma/)
- [Linköping University](https://www.liu.se/)
- [NAISS](https://www.naiss.se/)

The position page provides background on the SE.LLMA initiative and the AI Training Application Expert role.

## Citation for the main article

```bibtex
@misc{takehi2026weightspace,
  title        = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}},
  note         = {Research blog post}
}
```
