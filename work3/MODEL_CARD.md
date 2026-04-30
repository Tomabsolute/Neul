# Model Card: Conditional DCGAN for Fashion-MNIST

## Model

- Architecture: Conditional DCGAN
- Generator input: random noise vector plus class embedding
- Discriminator input: image plus label map channel
- Dataset: Fashion-MNIST
- Image size: 1 x 28 x 28
- Classes: 10 clothing categories

## Intended Use

This model is intended for a course assignment demonstration of GAN-based conditional image generation. It is not designed for high-quality production image synthesis.

## Training Setup

Default command:

```bash
python3 work3/train_cgan.py --dataset fashion-mnist --download --epochs 20 --batch-size 128 --num-samples 6000
```

The checkpoint saved at `work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt` contains:

- generator weights
- discriminator weights
- class names
- training arguments
- training history

## Inference

```bash
python3 work3/infer.py --checkpoint work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt
```

## Limitations

Fashion-MNIST is low resolution and grayscale. Generated images can show mode collapse, ambiguous class boundaries, or repeated shapes when the training budget is small.
