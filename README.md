### Usage

__Step 1:__ Clone the repo and cd into it.

```bash
git clone https://github.com/dallasplunkett/dsc-180a.git && cd dsc-180a
```

__Step 2:__ Inside `config.py` update the `data_dir` to reflect where the *.hdf5 files are contained. Then update the `train_csv` and `test_csv` to specify the csv's specific paths. Below is where we have placed the directories and files along with their names.

```
dsc-180a/
    ...
    main.py
    data/
        train.csv
        test.csv
        images_1.hdf5
        images_2.hdf5
        ...
    src/
        config.py
        data.py
        ...
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