## Usage

#### Step 0: Make any updates to your code.

#### Step 1: Build the Docker image for the target platform and tag it.

```bash
docker buildx build --platform [TARGET_OS]/[TARGET_CPU] -t [DOCKER_USERNAME]/[IMAGE]:[TAG] [CONTEXT_PATH]
```

Example,

```bash
docker buildx build --platform linux/amd64 -t dallasplunkett/train:latest .
```

#### Step 2: Push the image to Docker Hub (must be public for DSMLP).

```bash
docker push [DOCKER_USERNAME]/[IMAGE]:[TAG]
```

Example,

```bash
docker push dallasplunkett/train:latest
```

#### Step 3: SSH into the DSMLP jumpbox (VPN may be required if off campus).

```bash
ssh [UCSD_USERNAME]@dsmlp-login.ucsd.edu
# or
ssh [UCSD_USERNAME]@128.54.65.160
```

#### Step 4: Launch your container with the desired resources.

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

#### Step 5: Inside the container, start training.

```bash
cd /workspace
python main.py
```

#### Step 6: Watch logs, you will need to authenticate with Weights & Biases on first run.
