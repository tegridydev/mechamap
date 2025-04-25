import sys
import time
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy

###############################################################################
# 1. RESEARCH BASELINE CONFIG
###############################################################################
# This section centralizes all the relevant "research knobs" so that
# you or your collaborators can easily tweak them without hunting
# through the rest of the script.

class NeuroMapConfig:
    """
    A baseline configuration object for the NeuroMap scanning tool.
    Modify fields as needed for your experiments.
    """
    def __init__(self):
        #
        # 1) Category tokens
        # Expand or adjust these lists to define the semantic categories
        # you're interested in. Each key is a category name, each value is
        # a list of tokens that represent that category.
        #
        self.CATEGORY_TOKENS: Dict[str, List[str]] = {
            "location": [
                "city", "country", "capital", "Paris", "London", "Australia", 
                "Canberra", "Tokyo", "Madrid", "Berlin"
            ],
            "numeric": [
                "1", "10", "100", "1000", "2023", "million", "zero", 
                "2", "3.14", "365", "year"
            ],
            "food": [
                "apple", "banana", "pizza", "sushi", "chocolate", "bread", 
                "cheese", "salad", "fruit", "vegetable", "hamburger", 
                "carrot", "steak"
            ],
            "date": [
                "Jan", "February", "Monday", "month", "2021", "2022", 
                "Tuesday", "week", "December"
            ],
            "animal": [
                "cat", "dog", "horse", "elephant", "lion", "tiger", 
                "giraffe", "monkey"
            ],
            "vehicle": [
                "car", "truck", "bicycle", "airplane", "train", 
                "ship", "motorcycle", "bus"
            ],
            "color": [
                "red", "blue", "green", "yellow", "purple", "orange", 
                "black", "white"
            ],
            "language": [
                "English", "French", "Spanish", "German", "Chinese", 
                "Japanese", "Arabic"
            ],
            # Feel free to add more categories as needed.
        }

        #
        # 2) Master text (the single input containing tokens from each category)
        # We have combined various tokens from each category into one or two 
        # coherent paragraphs so the text remains somewhat readable while 
        # hitting the key tokens you care about.
        #
        self.DEFAULT_MASTER_TEXT: str = (
            "The capital city of Australia is Canberra, a country known for vibrant cities like Sydney. "
            "I visited London last year and ate sushi, chocolate, and a banana for lunch. "
            "We also saw a red car and an airplane at the airport. "
            "My dog loves bread and cheese, while my cat prefers a steak. "
            "Zero is an interesting numeric concept, along with 3.14 for pi. "
            "In December of the year 2022, I traveled by train to Madrid and used French to communicate. "
            "Sometimes I enjoy a salad or a fruit, especially if it's an apple or a carrot. "
            "The lion at the zoo was near a giant purple giraffe statue. "
            "English is commonly spoken, but Japanese and Spanish are also widely learned. "
            "I saw 100 bicycles parked near a bus station, and Berlin is another big city. "
            "Vehicles like ships and motorcycles fascinate me. "
            "Monday is often considered the start of the week. "
            "Tokyo has millions of people and many airplanes crossing its skies."
        )

        #
        # 3) Default scanning thresholds & top tokens
        #
        self.DEFAULT_THRESHOLD: float = 0.2  # Min average activation to count a neuron as "matching" a category
        self.DEFAULT_TOP_K_TOKENS: int = 5   # How many top tokens to store per neuron
        self.DEFAULT_LIMIT_NEURONS: Optional[int] = None  # e.g. 50 if you want partial scanning

        #
        # 4) Output filenames (can remain None if not used)
        #
        self.DEFAULT_CSV: Optional[str] = None
        self.DEFAULT_JSON: Optional[str] = None

        #
        # 5) Verbosity / debug flags
        # For larger models, you might want to limit how much logging is printed.
        #
        self.VERBOSE: bool = True

    def __repr__(self):
        """Returns a string representation for debugging."""
        fields = [
            f"CATEGORY_TOKENS={list(self.CATEGORY_TOKENS.keys())}  (extended lists)",
            f"DEFAULT_MASTER_TEXT=({len(self.DEFAULT_MASTER_TEXT)} chars long)",
            f"DEFAULT_THRESHOLD={self.DEFAULT_THRESHOLD}",
            f"DEFAULT_TOP_K_TOKENS={self.DEFAULT_TOP_K_TOKENS}",
            f"DEFAULT_LIMIT_NEURONS={self.DEFAULT_LIMIT_NEURONS}",
            f"DEFAULT_CSV={self.DEFAULT_CSV}",
            f"DEFAULT_JSON={self.DEFAULT_JSON}",
            f"VERBOSE={self.VERBOSE}",
        ]
        return "NeuroMapConfig(\n  " + ",\n  ".join(fields) + "\n)"

###############################################################################
# 2. MAIN SCANNING LOGIC (INCORPORATING CONFIG)
###############################################################################

def map_neurons_with_config(
    model_name: str,
    config: NeuroMapConfig,
    threshold: float,
    limit_neurons: Optional[int],
    top_k_tokens: int,
    custom_text: Optional[str],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    A scanning function that uses the config object to gather category tokens,
    master text, etc. Returns a dictionary of (layer, neuron) -> {
      "categories": {cat: avg_val, ...},
      "top_tokens": [ (token, activation), ... ]
    }
    """

    # 1) Resolve which text to use
    if custom_text and custom_text.strip():
        master_text = custom_text.strip()
    else:
        master_text = config.DEFAULT_MASTER_TEXT

    # 2) Load model
    if config.VERBOSE:
        print(f"\nLoading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name)

    hidden_size = model.cfg.d_mlp
    n_layers = model.cfg.n_layers
    if config.VERBOSE:
        print(f"Model has {n_layers} layers, {hidden_size} MLP neurons per layer.")

    # 3) Tokenize text
    tokens = model.to_str_tokens(master_text)
    seq_len = len(tokens)
    if config.VERBOSE:
        print(f"Master text has {seq_len} tokens total.")

    # 4) Build list of (layer, neuron) pairs
    if limit_neurons is not None and limit_neurons > 0:
        layer_neuron_pairs = [
            (l, n) for l in range(n_layers) for n in range(min(hidden_size, limit_neurons))
        ]
        if config.VERBOSE:
            print(f"Scanning only first {limit_neurons} neurons per layer => {len(layer_neuron_pairs)} combos.")
    else:
        layer_neuron_pairs = [
            (l, n) for l in range(n_layers) for n in range(hidden_size)
        ]
        if config.VERBOSE:
            print(f"Scanning all {len(layer_neuron_pairs)} (layer, neuron) combos...")

    # 5) Hook collection
    activation_map = {}
    for (layer, neuron_idx) in tqdm(layer_neuron_pairs, desc="Collecting Activations"):
        cache = {}

        def caching_hook(act, hook):
            cache["activation"] = act[0, :, neuron_idx]

        model.run_with_hooks(
            master_text,
            fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", caching_hook)]
        )
        acts = to_numpy(cache["activation"])  

       
        activation_map[(layer, neuron_idx)] = [
            (tokens[i].strip(), float(acts[i])) for i in range(seq_len)
        ]

    # 6) Summarize categories + top tokens
    found_dict = {}
    if config.VERBOSE:
        print(f"\nAnalyzing categories with threshold={threshold} for {len(layer_neuron_pairs)} neurons...")
    time.sleep(0.3)

    for (layer, neuron_idx), tok_val_list in tqdm(activation_map.items(), desc="Categorizing"):
        cat_res = {}
        
        token_map = defaultdict(list)
        for (tok, val) in tok_val_list:
            token_map[tok].append(val)

        
        for cat_name, cat_tokens in config.CATEGORY_TOKENS.items():
            cat_sum = 0.0
            cat_count = 0
            for ctk in cat_tokens:
                if ctk in token_map:
                    cat_sum += sum(token_map[ctk])
                    cat_count += len(token_map[ctk])
            if cat_count > 0:
                avg_val = cat_sum / cat_count
                if avg_val >= threshold:
                    cat_res[cat_name] = avg_val

       
        sorted_tok_vals = sorted(tok_val_list, key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tok_vals[:top_k_tokens]

        if cat_res or top_tokens:
            found_dict[(layer, neuron_idx)] = {
                "categories": cat_res,
                "top_tokens": top_tokens,
            }

    return found_dict

###############################################################################
# 3. OUTPUT FUNCTIONS (CSV, JSON, SUMMARY)
###############################################################################

def save_to_csv(
    results: Dict[Tuple[int, int], Dict[str, Any]],
    model_name: str,
    csv_filename: str
):
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "layer", "neuron", "category", "avg_activation", "top_tokens"])

        for (layer, neuron_idx), data_dict in results.items():
            cat_dict = data_dict["categories"]
            top_tokens = data_dict["top_tokens"]
            top_tokens_str = str(top_tokens)

            if cat_dict:
                for cat_name, avg_val in cat_dict.items():
                    writer.writerow([
                        model_name,
                        layer,
                        neuron_idx,
                        cat_name,
                        f"{avg_val:.4f}",
                        top_tokens_str
                    ])
            else:
                
                writer.writerow([
                    model_name,
                    layer,
                    neuron_idx,
                    "",
                    "",
                    top_tokens_str
                ])
    print(f"Results saved to CSV: {csv_filename}")


def save_to_json(
    results: Dict[Tuple[int, int], Dict[str, Any]],
    model_name: str,
    json_filename: str
):
    data_out = {
        "model": model_name,
        "results": {}
    }
    for (layer, neuron_idx), data_dict in results.items():
        key_str = f"({layer},{neuron_idx})"
        data_out["results"][key_str] = {
            "categories": data_dict["categories"],
            "top_tokens": data_dict["top_tokens"]
        }
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(data_out, f, indent=2)
    print(f"Results saved to JSON: {json_filename}")


def print_summary(
    results: Dict[Tuple[int, int], Dict[str, Any]],
    model_name: str
):
    print(f"\n--- SUMMARY for {model_name} ---")
    if not results:
        print("No neurons matched any category or top_tokens.")
        return

    
    cat_count_map = defaultdict(int)
    for (layer, neuron_idx), data_dict in results.items():
        cat_dict = data_dict["categories"]
        top_tokens = data_dict["top_tokens"]
        cat_str = ", ".join(f"{c}:{v:.2f}" for c, v in cat_dict.items()) if cat_dict else "(none)"

        top_tok_str = ", ".join(f"'{t}'({val:.2f})" for t, val in top_tokens)
        print(f"Layer {layer}, Neuron {neuron_idx} => Categories[{cat_str}] | TopTokens[{top_tok_str}]")

        for c_name in cat_dict:
            cat_count_map[c_name] += 1

    if cat_count_map:
        print("\nNumber of neurons matched per category:")
        for c_name, c_val in cat_count_map.items():
            print(f"  {c_name}: {c_val}")

    print("--- END OF SUMMARY ---")

###############################################################################
# 4. MAIN ENTRY POINT
###############################################################################

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuroMap with a centralized config for research baseline (expanded categories & text)."
    )
    parser.add_argument(
        "model",
        help="Name of pretrained model to scan (e.g., 'gpt2-small', 'EleutherAI/pythia-70m')."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Activation threshold override (default uses config.DEFAULT_THRESHOLD)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="If provided, saves results to CSV."
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="If provided, saves results to JSON."
    )
    parser.add_argument(
        "--limit_neurons",
        type=int,
        default=None,
        help="If set, only scan the first N neurons per layer."
    )
    parser.add_argument(
        "--custom_text",
        type=str,
        default=None,
        help="Optionally override the master text from config."
    )
    parser.add_argument(
        "--top_k_tokens",
        type=int,
        default=None,
        help="Number of top tokens to store per neuron (default from config)."
    )

    args = parser.parse_args()

    
    config = NeuroMapConfig()
    if config.VERBOSE:
        print("=== NeuroMapConfig ===")
        print(config)

    
    threshold = args.threshold if args.threshold is not None else config.DEFAULT_THRESHOLD
    limit = args.limit_neurons if args.limit_neurons is not None else config.DEFAULT_LIMIT_NEURONS
    top_k = args.top_k_tokens if args.top_k_tokens is not None else config.DEFAULT_TOP_K_TOKENS
    custom_text = args.custom_text if args.custom_text else None
    csv_filename = args.csv if args.csv else config.DEFAULT_CSV
    json_filename = args.json if args.json else config.DEFAULT_JSON

    
    results = map_neurons_with_config(
        model_name=args.model,
        config=config,
        threshold=threshold,
        limit_neurons=limit,
        top_k_tokens=top_k,
        custom_text=custom_text
    )

   
    print_summary(results, args.model)

    
    if csv_filename:
        save_to_csv(results, args.model, csv_filename)
    if json_filename:
        save_to_json(results, args.model, json_filename)

if __name__ == "__main__":
    main()
