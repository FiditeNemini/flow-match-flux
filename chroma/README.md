# Chroma Model Training

This directory contains the necessary scripts and configurations to train the Chroma image generation model.

## Training

To start the training process, run the following command:

```bash
python train.py
```

This will automatically detect the number of available GPUs and start the training process using the settings defined in `config.json`.

## Training with CFG

To train with Classifier-Free Guidance (CFG), use the following command:

```bash
python train_cfg.py
```

This will use the configuration from `config_cfg.json`.

## Training with LoRA

To train with LoRA, use the following command:

```bash
python train_lora.py
```

This will use the configuration from `config_lora.json`.