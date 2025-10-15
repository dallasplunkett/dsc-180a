## Usage (WIP)

1. Build the container.

```bash
docker build -t bnpp-trainer .
```

1. Create a `data` directory containing the `.csv` and `.hdf5` files in the project's root directory.

2. Add your Weights and Biases API key to your `.bashrc` or `.zshrc` and refresh it with source.

```bash
export WANDB_API_KEY=your_actual_key_here
```

```bash
source ~/.zshrc
```

4. Run the container.

```bash
docker run -it \
  -v ~Downloads/data:/app/data \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  --memory=0 \
  bnpp-trainer
```