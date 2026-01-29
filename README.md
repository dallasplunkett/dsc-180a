# DSC 180A — B21-1

Dallas Plunkett, Kendall Underwood, Jeru Balares

## Setup

```bash
git clone https://github.com/dallasplunkett/dsc-180a.git
cd dsc-180a
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download a subset of the [data]() and place it in the project's root directory (i.e. `dsc-180a/data`).

## LLM Usage

Tune

```bash
python -m llm.tune \
  --input data/reports/tune.csv \
  --output tuned_artifact \
  --prompt data/prompts/final.txt \
  --model mlx-community/medgemma-4b-it-4bit \
  --iters 1
```

Predict

```bash
python -m llm.predict \
  --input data/reports/test.csv \
  --output data/reports/preds.csv \
  --prompt data/prompts/final.txt \
  --model mlx-community/medgemma-4b-it-4bit \
  --adapters tuned_artifact/adapters \
  --limit 1
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
