# DSC 180A — B21-1

Dallas Plunkett, Kendall Underwood, Jeru Balares

## Environment Setup

```bash
git clone https://github.com/dallasplunkett/dsc-180a.git
cd dsc-180a
```

- macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Conda

```bash
conda create -n dsc180 python=3.11 -y
conda activate dsc180
pip install -r requirements.txt
```

A subset of the data used in our report can be downloaded from [Google Drive](https://drive.google.com/file/d/1ZANVqgPkUcJtffudfi34CwR-ynMEVl5q/view?usp=sharing). After download, extract the folder into the project root so it appears as `dsc-180a/data`.

To use the CNN part of this project you will need a Weights and Biases API key. Below are instructions for setting up an account if needed.

  1. Go to [Weights & Biases (W&B)](https://wandb.ai/site) and sign up for a free account.
  2. Once signed in, navigate to your [W&B homepage](https://wandb.ai/home).
  3. Click your profile icon (top-right), scroll down, and click "API Key".
  4. Copy your API key — you’ll need it when you run the CNN program.

## LLM Usage

```bash
cd llm
```

- Tune

```bash
python tune.py \
  --input ../data/reports/tune.csv \
  --output tuned_artifact \
  --prompt ../data/prompts/final.txt \
  --model mlx-community/medgemma-4b-it-4bit \
  --iters 1 # quick check
```

> __Note:__ This will create and write to `data/` and `adapters/` directories under the output directory specified and invoke `python -m mlx_lm.lora`.

- Predict

```bash
python3 predict.py \
  --input ../data/reports/test.csv \
  --output ../data/reports/preds.csv \
  --prompt ../data/prompts/final.txt \
  --model mlx-community/medgemma-4b-it-4bit \
  --adapters tuned_artifact/adapters \ # optional
  --limit 3 # quick check
```

> __Note:__ The optional adapters argument attaches the tuned model artifacts to the "base" model provided. Meaning the adapter must be compatible with the model selected.

## CNN Usage

Ensure you are in the root directory.

```bash
python cnn.main.py --preset test
```

You will be prompted in the terminal to select which mode you want to run W&B in. __Type 2 and press Enter__.

You will then be prompted to enter your API Key. __Paste it in and press Enter.__

From there the program will run and you should see a link to the W&B project. __Click the link next to the rocket 🚀 emoji__ and watch the CNN train!

Once the final epoch completes, predictions, examples, and more plots will be created within the W&B dashboard.

> **NOTE:** These steps uses a very small subset of the data and a lower quality configuration in the hopes to allow a grader to see in part, what we have done. So the performance and results are not what will be seen in the report — those took beefy GPUs and days to run on the DSMLP platform.

### Team DSMLP Notes for CNN

* Build a Docker Image

```bash
docker buildx build --platform [TARGET_OS]/[TARGET_CPU] -t [DOCKER_USERNAME]/[IMAGE]:[TAG] [CONTEXT_PATH]
```

Example,

```bash
docker buildx build --platform linux/amd64 -t dallasplunkett/train:latest .
```

- Push an Image to DockerHub

```bash
docker push [DOCKER_USERNAME]/[IMAGE]:[TAG]
```

Example,

```bash
docker push dallasplunkett/train:latest
```

- SSH into DSMLP

```bash
ssh [UCSD_USERNAME]@dsmlp-login.ucsd.edu
# or
ssh [UCSD_USERNAME]@128.54.65.160
```

- Set up W&B API Key

```bash
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- Launch Container

```bash
launch.sh \
    -W [COURSE_WORKSPACE] -G [GROUP_ID] \
    -i [DOCKER_USERNAME]/[IMAGE]:[TAG] \
    -c [NUMBER_OF_CPUs] -m [SIZE_OF_RAM] -g [NUMBER_OF_GPUs] -v [GPU_VARIANT] \
    -P Always -B -- \
    bash -lc 'cd /workspace && python main.py'
```

Example,

```bash
launch.sh \
  -W DSC180A_FA25_A00 -G b1100018875 \
  -i dallasplunkett/train:latest \
  -c 2 -m 16 -g 1 -v 2080ti \
  -P Always -B -- \
  bash -lc 'cd /workspace && python main.py --project test_run --model resnet34 --size 64 --epochs 5'
```

Other commonly used CLI arguments. See [Reference for DSMLP Launch Flags](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032273) for more.

| Flag                | Type    | Description                                                                                                |
| ------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `--preset`, `--cfg` | string  | Selects a base configuration preset (default = `remote`) — supported values: `remote`, `local`, `test`     |
| `--project`         | string  | Sets the Weights & Biases project name                                                                     |
| `--model`           | string  | CNN backbone architecture — supported values: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| `--size`            | integer | Input image resolution (square) — ranges: 0 to 1024                                                        |
| `--batch_size`      | integer | Number of samples per training batch                                                                       |
| `--epochs`          | integer | Number of training epochs to run                                                                           |
| `--learning_rate`   | float   | Optimizer learning rate for AdamW                                                                          |
| `--weight_decay`    | float   | L2 regularization strength                                                                                 |

- Watching Runs on W&B

Head to your W&B profile `https://wandb.ai/[PROFILE]` > Go to Projects tab > Select the Project Name you set.

- Kubernetes Pod Commands


| Command                       | Action           |
| ----------------------------- | ---------------- |
| `kubectl get pods`            | List all pods    |
| `kubesh [POD-ID]`             | "SSH" into a pod |
| `kubectl logs -f [POD-ID]`    | Stream pod logs  |
| `kubectl delete pod [POD-ID]` | Delete a pod     |
