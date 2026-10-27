# InferPriv: Evaluating Inferential Privacy Leakage in LLM Assistants

Pengrun Huang\*¹, Yiwei Fu\*², Luoxi Meng¹, Kamalika Chaudhuri¹<br>
¹UC San Diego · ²University of Illinois Urbana-Champaign · \*Equal contribution

[**Paper**](https://arxiv.org/abs/2610.34640)

Large language model (LLM)-based personal agents are increasingly deployed to handle everyday tasks on users' behalf. These agents have access to private user information, while their tasks involve communicating with third parties; this raises a critical privacy concern that agents may inappropriately reveal sensitive user information to these recipients.

We introduce \textsc{InferPriv}, a benchmark that measures whether an honest-but-curious recipient can infer a user's sensitive attributes from an LLM's reply on routine tasks.

In our paper, we find that inferential leakage substantially exceeds direct leakage across models and prompts. Even under a privacy-aware system prompt where direct leakage can be minimized, the inferential leakage rate remains between significant across frontier models.

This repository contains the benchmark's code and data, which spans 10 personas with 27 memory statements each, 10
sensitive attributes with 3 sensitive statements each, and 10 routine tasks, for 950 persona × attribute × task cells.

![Two examples of inferential leakage: an RSVP that avoids alcohol and unpasteurized cheese reveals a pregnancy, and a hotel reply asking for a refrigerator for medical supplies reveals diabetes](assets/examples.png)

## Setup

The code requires Python 3.10 or newer. Clone the repository and install the requirements:

```bash
git clone https://github.com/PengrunH/InferPriv.git
cd InferPriv
pip install -r requirements.txt
```

Each model is called through its provider's API. `config/providers.json` lists the supported providers and the
environment variable that holds each provider's API key, for example:

```json
"openai": {"api_key": "OPENAI_API_KEY", "base_url": "https://api.openai.com/v1"}
```

Set the key of every provider you use, either as an environment variable or in a `.env` file at the repository root with
one `KEY=value` line per key. The OpenAI key is always needed, since the guesser and the completeness judge are OpenAI
models.

To use a provider that is not in `config/providers.json`, it needs an OpenAI-compatible Chat Completions API. Add an
entry for it, for example:

```json
"mistral": {"api_key": "MISTRAL_API_KEY", "base_url": "https://api.mistral.ai/v1"}
```

Then set `MISTRAL_API_KEY` and put `"provider": "mistral"` in the model's config (see [Evaluate a model](#evaluate-a-model)).

## Evaluate a model

To evaluate a model, first add a config file for it to `config/models/`; the configs of the paper's models are already
there. For OpenAI-compatible providers, every field other than `provider` is sent as it is to the provider's Chat
Completions API, so use the provider's own option names. Anthropic models accept `model`, `effort`, `thinking`,
`temperature` and `max_tokens`. A config looks like this:

```json
{"provider": "openai", "model": "my-model-id", "reasoning_effort": "medium"}
```

Then run it under each of the two prompts, passing the config's path with `--model`. The config's `"provider"` field
selects the provider's entry in `config/providers.json`, and with it the API key. Each command runs the control and then
the attribute run, and prints the scores at the end:

```bash
python -m inferpriv.evaluate --model config/models/my-model.json --prompt baseline
python -m inferpriv.evaluate --model config/models/my-model.json --prompt privacy-aware
```

Use `--memory unnamed` or `--memory declaration` to run a memory-representation variant, and `--skip-completeness`
to skip the completeness judgments. `--results-dir` and `--env-file` change where results and API keys are kept.

Results are written to `results/<model>_<effort>_<prompt>[_control|_unnamed|_declaration]/`. `generation.json`,
`inference__*.json` and `sat__*.json` store every API call (request and response), which is what the scores are computed
from. `replies.jsonl` is a readable summary with one line per reply: the cell (`persona`, `attribute`, `task_file`,
`sample`), the `reply`, whether the cell is `protected`, the attribute words found in the reply (`direct`), the
guesser's `guesses`, and a 0/1 completeness verdict per Required statement (`sat`). A rerun reuses stored results, so an
interrupted run can be restarted; if the prompt, model settings or data changed, use a new `--results-dir`.

To print the scores of a finished run again, without any API calls, run:

```bash
python -m inferpriv.score --model config/models/my-model.json --prompt baseline [--json]
```

## Necessity labels

`data/need_labels.json` contains the final label of every memory statement × task pair, and it is the only label file
the code reads. `data/constraints.json` contains the extracted constraints, and `data/manual_overrides.json` contains
the pairs that were set by hand.

To redo the labeling, for example after adding personas, sensitive statements or tasks to `data/` and
`config/benchmark.json`, run the following command. It needs `OPENAI_API_KEY`, `DEEPSEEK_API_KEY` and
`ANTHROPIC_API_KEY`.

```bash
python -m inferpriv.labeling.run
```

The output is written to `results/labeling/` and never to `data/`. Check the split pairs by hand before copying the
output into `data/`.

## License

The code is released under the [MIT License](LICENSE), and the data in `data/` under CC BY-NC 4.0 (see
[data/LICENSE](data/LICENSE)). The personas are adapted from [CIMemories](https://huggingface.co/datasets/facebook/CIMemories)
(CC BY-NC 4.0). The sensitive statements are adapted from [Privasis-Zero](https://huggingface.co/datasets/nvidia/Privasis-Zero),
so they are also subject to the NVIDIA License and may be used only for non-commercial research or evaluation.

## Citation

```bibtex
@misc{huang2026inferprivevaluatinginferentialprivacy,
      title={InferPriv: Evaluating Inferential Privacy Leakage in LLM Assistants}, 
      author={Pengrun Huang and Yiwei Fu and Luoxi Meng and Kamalika Chaudhuri},
      year={2026},
      eprint={2610.34640},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.34640}, 
}
```