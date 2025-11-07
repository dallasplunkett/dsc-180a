# DSC 180A — B21

Dallas Plunkett, Kendall Underwood, Jeru Balares

## TA Usage

The guide below describes the most general and reliable way for TAs to run the program locally in a reasonable amount of time.

#### Step 0. Create W&B Account and Note your API Key


1. Go to [Weights & Biases](https://wandb.ai/site) and sign up for a free account.
2. Once signed in, navigate to your [home](https://wandb.ai/home) page.
3. Click your profile icon (top-right), scroll down, and click "API Key".
4. Copy your API key — you’ll need it when you run the program.

#### Step 1. Clone the Repository and Create Checkpoints Folder Inside

```bash
git clone https://github.com/dallasplunkett/dsc-180a.git
cd dsc-180a
mkdir checkpoints
```

#### Step 2. Create and Activate the Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

To deactivate later, run `deactivate` or just exit the process.

#### Step 3. Install the Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4. Download the Data

A subset of the data (~20 GB) can be downloaded from [Google Drive](https://drive.google.com/file/d/1ZANVqgPkUcJtffudfi34CwR-ynMEVl5q/view?usp=sharing). After downloading, extract the folder into the project root so it appears as `dsc-180a/data`.

#### Step 5. Run the Program

```bash
python3 main.py
```

#### Step 6. Follow the Logs

You will be prompted to select which mode you want to run W&B in. Enter option 2. You will then be prompted to enter your API Key. From there the program should run and you will see a link to the W&B project. Click the link next to the 🚀 emoji and watch the CNN train!

#### Step 7. (Optional) Play with Config

The training process is driven by the `config.py` file. Feel free to play around with the `TestConfig` parameters and rerun the program again.

## Team Usage

#### Step 0. Make Changes

#### Step 1. Rebuild the Image

```bash
docker buildx build --platform [TARGET_OS]/[TARGET_CPU] -t [DOCKER_USERNAME]/[IMAGE]:[TAG] [CONTEXT_PATH]
```

Example,

```bash
docker buildx build --platform linux/amd64 -t dallasplunkett/train:latest .
```

#### Step 2. Push the Image to DockerHub

```bash
docker push [DOCKER_USERNAME]/[IMAGE]:[TAG]
```

Example,

```bash
docker push dallasplunkett/train:latest
```

#### Step 3. SSH into DSMLP

```bash
ssh [UCSD_USERNAME]@dsmlp-login.ucsd.edu
# or
ssh [UCSD_USERNAME]@128.54.65.160
```

#### Step 4. Launch the Container

```bash
launch.sh \
    -W [COURSE_WORKSPACE] -G [GROUP_ID] \
    -i [DOCKER_USERNAME]/[IMAGE]:[TAG] \
    -c [NUMBER_OF_CPUs] -m [SIZE_OF_RAM] -g [NUMBER_OF_GPUs] -v [GPU_VARIANT] \
    -P Always -T -s
```

Example,

```bash
launch.sh \
    -W DSC180A_FA25_A00 -G b1100018875 \
    -i dallasplunkett/train:latest \
    -c 2 -m 16 -g 1 -v 1080ti \
    -P Always -T -s
```

[Reference for DSMLP Launch Flags](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032273)

#### Step 5. Run the Program

```bash
cd /workspace
python main.py
```

#### Step 6. Follow the Logs
