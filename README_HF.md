---
title: AI Trust & Safety Demo
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ AI Trust & Safety Demo

A production-grade demonstration of trust and safety mechanisms for AI models, featuring **Out-of-Distribution (OOD) Detection** and **Adversarial Robustness Testing**.

## Features

### 🎯 Out-of-Distribution Detection
- Identifies inputs that fall outside the model's training distribution
- Uses Maximum Softmax Probability (MSP) method
- Prevents confident-but-wrong predictions on irrelevant content

### ⚔️ Adversarial Attack Generation
- Tests model robustness with character-level perturbations
- Uses DeepWordBug method via TextAttack
- Generates human-readable adversarial examples

## How It Works

1. **Enter text** in the input box
2. **Adjust threshold** for OOD detection sensitivity
3. **Click "Analyze"** to see:
   - Whether input is in-distribution or out-of-distribution
   - Model's prediction and confidence
   - Adversarial example that fools the model

## Model Details

- **Base Model**: DistilBERT
- **Training Data**: AG News (World, Sports, Business, Sci/Tech)
- **Framework**: PyTorch + Hugging Face Transformers
- **Attack Method**: DeepWordBug (TextAttack)

## Try It!

Use the examples provided or enter your own text to see how the model handles:
- ✅ In-distribution inputs (news articles)
- 🚫 Out-of-distribution inputs (movie reviews, random text)
- ⚔️ Adversarial attacks (perturbed text)

## Technologies

- **Model**: Hugging Face Transformers (DistilBERT)
- **Framework**: PyTorch
- **Adversarial Tools**: TextAttack
- **Interface**: Gradio

## Repository

[GitHub Repository](https://github.com/Sakeeb91/Adversarial-Robustness-and-OOD-Detection)
