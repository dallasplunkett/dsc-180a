### Usage

__Step 1:__ Clone the repo and cd into it.

```bash
git clone https://github.com/dallasplunkett/dsc-180a.git && cd dsc-180a
```

__Step 2:__ Create a `data/` folder in the repo's root that contains the `.csv` and `.hdf5` files. Your folder should look something like this:

```
dsc-180a/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
└── data/
    ├── train.csv
    ├── validation.csv
    ├── images_1.hdf5
    ├── images_2.hdf5
    └── ...
```

__Step 3:__ Build the container.

```bash
docker build -t bnpp-trainer .
```

__Step 4:__ Run the container.

```bash
docker run -it --rm \
  --cpus=8 \
  --memory=8g \
  --shm-size=8g \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/wandb:/app/wandb" \
  bnpp-trainer
```

__Step 5:__ Follow the logs.