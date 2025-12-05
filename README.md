# DSC 180A — B21

Dallas Plunkett, Kendall Underwood, Jeru Balares

## TA Usage

The guide below describes the most general and reliable way for TAs to run the program locally in a reasonable amount of time.

### Step 0. Create W&B Account and Note your API Key

1. Go to [Weights & Biases (W&B)](https://wandb.ai/site) and sign up for a free account.
2. Once signed in, navigate to your [W&B home page](https://wandb.ai/home).
3. Click your profile icon (top-right), scroll down, and click "API Key".
4. Copy your API key — you’ll need it when you run the program.

### Step 1. Clone the Repository and Create Checkpoints Folder Inside

```bash
git clone https://github.com/dallasplunkett/dsc-180a.git
cd dsc-180a
mkdir checkpoints
```

### Step 2. Create and Activate the Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To deactivate later, run `deactivate` or just exit the process.

### Step 3. Install the Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4. Download the Data

A subset of the data (8.8GB zipped ~20 GB unzipped) can be downloaded from [Google Drive](https://drive.google.com/file/d/1ZANVqgPkUcJtffudfi34CwR-ynMEVl5q/view?usp=sharing). After downloading, extract the folder into the project root so it appears as `dsc-180a/data`.

> **NOTE:** The data is anonymized, so no worries about Personal Identifiable Information (PII).

### Step 5. Run the Program

```bash
python3 main.py --preset local
```

### Step 6. Follow the Logs

You will be prompted to select which mode you want to run W&B in. *Enter option 2*. You will then be prompted to enter your API Key. From there the program should run and you will see a link to the W&B project. Click the link next to the rocket 🚀 emoji and watch the CNN train! Once the final epoch completes, predictions, examples, and more plots will be created within the W&B dashboard.

> **NOTE:** These steps uses a very small subset of the data and a lower quality configuration in the hopes to allow a grader to see in part, what we have done. So the performance and results are not what will be seen in the report — those took beefy GPUs and days to run on the DSMLP platform.

### Step 7. (Optional) Play with the Config!

For preset configurations of the training process, feel free to look into the `config.py` file. You can make changes here, or override them with the available cli flags on Step 5 of the team usage below.

Example,

```bash
python3 main.py --preset local --model resnet18 --size 512 --epochs 24
```

## Team Usage

### Step 0. Make Changes

### Step 1. Rebuild the Image

```bash
docker buildx build --platform [TARGET_OS]/[TARGET_CPU] -t [DOCKER_USERNAME]/[IMAGE]:[TAG] [CONTEXT_PATH]
```

Example,

```bash
docker buildx build --platform linux/amd64 -t dallasplunkett/train:latest .
```

### Step 2. Push the Image to DockerHub

```bash
docker push [DOCKER_USERNAME]/[IMAGE]:[TAG]
```

Example,

```bash
docker push dallasplunkett/train:latest
```

### Step 3. SSH into DSMLP

```bash
ssh [UCSD_USERNAME]@dsmlp-login.ucsd.edu
# or
ssh [UCSD_USERNAME]@128.54.65.160
```

### Step 4. Set your W&B API Key

```bash
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 5. Launch the Container

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

**Reference for Supported CLI Arguments:**

| Flag                | Type       | Description                                                                                                 |
| ------------------- | ---------- | ----------------------------------------------------------------------------------------------------------- |
| `--preset`, `--cfg` | string       | Selects a base configuration preset (default = `remote`) — supported values: `remote`, `local`, `test`      |
| `--project`         | string     | Sets the Weights & Biases project name                                                                      |
| `--model`           | string     | CNN backbone architecture — supported values: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`  |
| `--size`            | integer    | Input image resolution (square) — ranges: 0 to 1024                                                         |
| `--batch_size`      | integer    | Number of samples per training batch                                                                        |
| `--epochs`          | integer    | Number of training epochs to run                                                                            |
| `--learning_rate`   | float      | Optimizer learning rate for AdamW                                                                           |
| `--weight_decay`    | float      | L2 regularization strength                                                                                  |


[Reference for DSMLP Launch Flags](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032273)

### Step 6. Watch the Run on W&B

Head to your W&B profile `https://wandb.ai/[PROFILE]` > Go to Projects tab > Select the Project Name you set.

For trouble shooting (i.e. run is not appearing) you'll have too look at the pod. Below is as a little cheat sheet for Kubernetes pod commands to investigate and manage your pods.

**Kubernetes Pod Command Cheat Sheet**

| Command                       | Action           |
| ----------------------------- | ---------------- |
| `kubectl get pods`            | List all pods    |
| `kubesh [POD-ID]`             | "SSH" into a pod |
| `kubectl logs -f [POD-ID]`    | Stream pod logs  |
| `kubectl delete pod [POD-ID]` | Delete a pod     |
