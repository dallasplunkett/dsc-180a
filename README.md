# DSC 180 Group B21-1

Contributors: Dallas Plunkett, Kendall Underwood, and Jeru Balares.

Project Website: https://dallasplunkett.github.io/dsc-180a/, with source files in `/docs`.

## Setup

Environment

```bash
git clone https://github.com/dallasplunkett/dsc-180a.git
cd dsc-180a
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Data

1. Subset Download: https://drive.google.com/file/d/1tyKNXRUkTtuYiNe5uwmFpqZP5C7FqDni/view?usp=sharing.
2. Once downloaded, unzip it.
3. Place unzipped `data` directory into the project's root directory (i.e. `dsc-180a/data`).

## LLM Usage

Tune

```bash
python -m llm.tune \
  --input data/reports/tune.csv \
  --output tune_artifact \
  --prompt data/prompts/binary.txt \
  --model mlx-community/medgemma-4b-it-4bit \
  --iters 1
```

Predict

```bash
python -m llm.predict \
  --input data/reports/tune.csv \
  --output debug_preds.csv \
  --prompt data/prompts/binary.txt \
  --model mlx-community/medgemma-4b-it-4bit \
  --adapters tune_artifact/adapters \
  --limit 1
```

Eval

```bash
python -m llm.eval \
  --input debug_preds.csv \
  --output eval_for_debug_preds
```

## CNN Usage

The CNN requires a Weights and Biases API key. Below are instructions for obtaining an account and finding your API key:

1. Go to [Weights & Biases (W&B)](https://wandb.ai/site) and sign up for a free account.
2. Once signed in, navigate to your [W&B homepage](https://wandb.ai/home).
3. Click your profile icon (top-right), scroll down, and click "API Key".
4. Copy your API key — you’ll need it when you run the CNN program.

```bash
python -m cnn.main --preset test
```

Follow the prompts:

1. For the first prompt __type 2 and press enter__.
2. For the second prompt __paste your W&B API key and press enter__.
3. Lastly, __click the link next to the rocket 🚀 emoji__ and watch the CNN run!
