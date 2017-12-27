"""Global constants shared throughout the code."""

# Truncation (in voxels).
TRUNCATION = 3

# Number of classes.
NUM_CLASSES = 13

# Class weights for training SUNCG data.
WEIGHT_CLASSES = [
    0.1, 2.0, 0.4, 2.0, 0.4, 0.6, 0.6, 2.0, 2.0, 2.0, 0.4, 0.5, 0.1
]