# UNIFIED-PROJECT-1.--ANIMAL-IMAGE-CLASSIFICATION


# Animal Image Classification Transfer Learning

## Objective

Train an image classification model using transfer learning on MobileNetV2 to classify animal images.

## Dataset

Organize image dataset in folders inside `dataset/` with one folder per class.

## Usage

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run training script:

```
python train_model.py
```

3. The model trains with early stopping and saves a Keras `.h5` model file.

## Model Architecture

- MobileNetV2 as base model with ImageNet weights (frozen).
- Additional custom classification layers added on top.

## Results

Shows training and validation accuracy and loss, saved model for later inference or fine-tuning.
