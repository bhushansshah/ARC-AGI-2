# ARC Prize 2025 (ARC AGI 2) Solution

This repository contains code and documentation for my entry to the [Kaggle ARC Prize 2025 Competition](https://www.kaggle.com/competitions/arc-prize-2025/overview).

## 📚 Competition Overview

The ARC Prize 2025 challenges participants to develop AI systems that can **efficiently learn new skills and solve open-ended, novel reasoning tasks**, rather than just depending on pattern recognition from large datasets. Unlike previous LLM competitions, this focuses on abstraction and reasoning.

- **Dataset**: Collection of train/test grid transformation tasks — see [ARCPrize.org](http://arcprize.org/play) for an interactive demo.
- **Goal**: For each "test" grid, given several "train" demonstration input/output pairs, predict the corresponding output grid.
- **Evaluation**: Exact match on grid predictions across 2 attempts per task/test input. Final score is accuracy (percentage of correct predictions).

**See the [official competition page](https://www.kaggle.com/competitions/arc-prize-2025/overview) for full details.**

---

## 📝 Action Plan and Approach

### 1. CNN + Transformers for Grid Processing

1. **Convert every grid to image** format for consistency and potential use of vision models.
2. **Use pretrained vision models** (CNN backbones, e.g., ResNet, ViT) to extract features from each grid.
3. **Apply transformer blocks** on extracted vision features to model global relationships.
4. **Predict the output grid** (up to 30x30, values 0–9) based on multi-modal features.

### 2. Use of Large Language Models (LLMs)

  - **Two strategies:**  
    1. **Prompting:** Experiment with clever input prompts to elicit correct reasoning.
    2. **Finetuning:** Train or adapt LLMs on ARC-like data for improved reasoning.
  - **Variants:**  
    a. *Single model:* Use one strong LLM or vision transformer.  
    b. *Ensemble:* Aggregate results from several models (majority/weighted voting).  
    c. *LLM Ensemble + Previous Competition Solutions:* Combine LLM ensembles with ideas from top ARC 2024 solutions, e.g., weighted voting or stacking.

### 3. Agentic Workflow

- Implement a **Generator + Critique** loop:
  - The generator proposes candidate solutions for each task.
  - The critique agent evaluates and refines/rejects proposals.
  - This mimics human iterative reasoning and self-correction.

### 4. GANs (Generative Adversarial Networks)

- Explore generative models for complex reasoning or to produce diverse solution candidates.

---
