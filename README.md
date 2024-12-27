# MechaMap - Tool for Mechanistic Interpretability (MI) Research

**MechaMap** is a scanner and analysis framework designed to help researchers working on interpreting **"wtfisgoingon"** within transformer-based language models. It quickly surfaces which neurons (across all layers) might be responding strongly to different semantic categories—like *location*, *food*, *numeric*, *animal*, etc.—based on a single pass of a master text/s.

## What is Mechanistic Interpretability?

**Mechanistic Interpretability (MI)** is the discipline of **opening the black box** of large language models (and other neural networks) to understand the **underlying circuits**, **features** and/or **mechanisms** that give rise to specific behaviors. Instead of treating a model as a monolithic function, we:

1. **Trace** how input tokens propagate through attention heads, MLP layers, or neurons,  
2. **Identify** localized “circuit motifs” or subsets of weights that implement certain tasks,  
3. **Explain** how the distribution of learned parameters leads to emergent capabilities,  
4. **Develop** methods to systematically break down or “edit” these circuits to confirm we understand the causal structure.

Mechanistic Interpretability aspires to yield **human-understandable explanations** of how advanced models represent and manipulate concepts like “zero,” “red,” “lion,” or “London.” By doing so, we gain:

- **Trust & Reliability**: More confidence in model outputs if we know the circuits behind them.  
- **Safety & Alignment**: Early detection of harmful or unintended sub-circuits.  
- **Debugging**: Efficient fixes or interventions if a model shows undesired behaviors.

> **Reference & Kudos**: This project owes a great deal to the insights from [Neel Nanda’s Mechanistic Interpretability Glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary). 
Neel’s research and writing efforts have significantly helped the understanding of circuits and interpretability in large language models.

## Goals of MechaMap

- **Rapid Discovery**: Provide a **one-and-done** pass that highlights potentially interesting neurons—particularly those that strongly respond to high-level semantic categories (like “vehicle,” “food,” “numeric,” etc.).  
- **Foundational Baseline**: Act as a **launchpad** for deeper Mechanistic Interpretability experiments. Once MechaMap flags certain “candidate neurons,” researchers can do single-neuron hooking or more advanced circuit analysis.  
- **Usability**: Keep the scanning code straightforward, and produce easy-to-parse CSV/JSON files that can be quickly ingested into more advanced interpretability pipelines.  
- **Transparency**: Centralize all category tokens, scanning thresholds, and master text within a single config. This fosters reproducibility and allows for quick expansions (adding more categories or new domain tokens).

## Key Features

1. **Single-Pass “Master Text”**  
   A single text that includes sample tokens from each category. MechaMap runs one forward pass per neuron (hooked at MLP outputs), computing **average activation** on tokens for each category.

2. **Customizable Categories**  
   Default categories include *date*, *location*, *animal*, *food*, *numeric*, *language*, *vehicle*, *color*, but you can easily add *sports*, *finance*, or any domain tokens you care about.

3. **Partial Scanning for Large Models**  
   If a model is huge, you can limit to the **first N neurons** per layer. That speeds up scanning while preserving the same analysis code.

4. **Top Tokens**  
   For each neuron, MechaMap also saves a short list of the **top-activating tokens** from the text. This can reveal surprising structural triggers (like punctuation or stopwords).

5. **Interpretation Script**  
   A separate `interpret_map.py` tool helps you parse the CSV, show pivot tables, detect multi-category overlaps, or compute correlations among categories.

## Why Use MechaMap for Mechanistic Interpretability?

- **Discovery Layer**  
  Instead of manually investigating thousands of neurons, MechaMap quickly flags where the biggest domain-sensitive signals might be happening—particularly in later layers, or “multi-domain” neurons that consistently appear across categories.

- **Hypothesis Generation**  
  Once you find a neuron that strongly activates for *food* tokens, you can design follow-up tests (e.g., hooking that neuron in isolation, adding or removing certain tokens) to confirm if it truly encodes “foodness.”

- **Comparative Studies**  
  Scan different models (e.g., `gpt2`, `EleutherAI/pythia-70m`) with the same master text and see whether they converge on similarly specialized neurons or if they use distinct “circuits.”

- **Extendable**  
  MechaMap is **config-based**: just add tokens to a category, or define new categories, and re-run. You can also adapt the master text to reflect your specific research interests (e.g., adding legal terms, chemistry tokens, or coding keywords).
