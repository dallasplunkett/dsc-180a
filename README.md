### How to Update Images

__Step 0:__ Make your code changes

Edit your source files locally, commit if needed, and ensure your project builds cleanly.

__Step 1:__ Build the updated image

```bash
docker buildx build --platform linux/amd64 -t <dockerhub_username>/<image_name>:latest .
```

Example

```bash
docker buildx build --platform linux/amd64 -t dallasplunkett/train:latest .
```

__Step 2:__ Push the image to DockerHub

```bash
docker push <dockerhub_username>/<image_name>:<tag>
```

- Make sure your image is public so DSMLP can pull it without authentication.

### How to Use the Image

__Step 3:__ SSH into the DSMLP jumpbox

```bash
ssh <ucsd_username>@dsmlp-login.ucsd.edu
# or
ssh <ucsd_username>@128.54.65.160
```

- VPN may be required when off-campus

__Step 4:__ Launch the container

Use the `launch.sh` script to start your container with the desired resources.

```bash
launch.sh \
    -W DSC180A_FA25_A00 -G b1100018875 \
    -i <dockerhub_username>/<image_name>:<tag> \
    -c 4 -m 16 -g 1 -v 1080ti \
    -P Always \
    -T -s
```

Example

```bash
launch.sh \
    -W DSC180A_FA25_A00 -G b1100018875 \
    -i dallasplunkett/train:latest \
    -c 4 -m 16 -g 1 -v 1080ti \
    -P Always \
    -T -s
```

> __Flag Reference:__\
> `-W` course workspace\
> `-G` your team/group ID\
> `-i` Docker image (on Docker Hub)\
> `-c` CPU cores\
> `-m` memory (GB)\
> `-g` GPUs requested\
> `-v` specific GPU type (e.g. 1080ti, 2080ti, V100)\
> `-P Always` always pull the latest image version\
> `-T` disable Jupyter Hub\
> `-s` start in shell mode

__Step 5:__ Run the training

Once inside the container, run:

```bash
cd /workspace
python main.py
```

__Step 6:__ Follow the logs
