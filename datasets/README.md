# Downloaded Datasets

This directory contains local copies of benchmark datasets for analyzing layer-wise hidden-state similarity in plain MLPs. Data files are excluded from git by `datasets/.gitignore`.

## Dataset 1: MNIST

### Overview
- Source: Hugging Face dataset `mnist`
- Size: 60,000 train / 10,000 test
- Format: Hugging Face dataset saved with `save_to_disk`
- Task: 10-class image classification
- Splits: `train`, `test`
- Local location: `datasets/mnist/`
- Local size: about 17 MB

### Why it fits this project
- Small and fast to train.
- Standard first benchmark for dense MLPs on flattened pixels.
- Useful for checking whether similarity structure is stable in a relatively easy task.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("mnist")
dataset.save_to_disk("datasets/mnist")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mnist")
```

### Sample Data
- See `datasets/mnist/samples/samples.json`

## Dataset 2: Fashion-MNIST

### Overview
- Source: Hugging Face dataset `fashion_mnist`
- Size: 60,000 train / 10,000 test
- Format: Hugging Face dataset saved with `save_to_disk`
- Task: 10-class image classification
- Splits: `train`, `test`
- Local location: `datasets/fashion_mnist/`
- Local size: about 36 MB

### Why it fits this project
- Same resolution as MNIST but materially harder.
- Lets the experiment runner test whether more difficult supervision changes how quickly layers diverge.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("fashion_mnist")
dataset.save_to_disk("datasets/fashion_mnist")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/fashion_mnist")
```

### Sample Data
- See `datasets/fashion_mnist/samples/samples.json`

## Dataset 3: CIFAR-10

### Overview
- Source: Hugging Face dataset `cifar10`
- Size: 50,000 train / 10,000 test
- Format: Hugging Face dataset saved with `save_to_disk`
- Task: 10-class image classification
- Splits: `train`, `test`
- Local location: `datasets/cifar10/`
- Local size: about 131 MB

### Why it fits this project
- Harder natural-image benchmark.
- Useful stress test for fully connected models on flattened RGB inputs or low-dimensional learned projections.
- Gives a stronger setting for comparing early-layer versus late-layer similarity.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("cifar10")
dataset.save_to_disk("datasets/cifar10")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/cifar10")
```

### Sample Data
- See `datasets/cifar10/samples/samples.json`

## Notes
- Image fields decode to PIL objects, so `Pillow` is needed when iterating examples.
- For MLP experiments, the simplest preprocessing is flattening images to vectors and normalizing pixel values to `[0, 1]`.
- Recommended primary datasets for the first run are `MNIST` and `Fashion-MNIST`; use `CIFAR-10` as the harder follow-up.
