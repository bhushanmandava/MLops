# Your Code Goes here:

# Step 1: Install
# pip install wandb

# Step 2: Setup the WandB library
# Importing the WandB library
import wandb
# Step 3: Initialize a new W&B run
wandb.init(project='first_MLops')
# Step 4: Log a sample metric
wandb.log({
    'acccuracy': 0.79
})
# Step 5: Finish the run
wandb.finish()
