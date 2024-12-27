import argparse
import pandas as pd
import os
import ast
import numpy as np

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV into a pandas DataFrame.
    Expects columns: 
      model, layer, neuron, category, avg_activation, [top_tokens] (optional).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)

    numeric_cols = ["layer", "neuron", "avg_activation"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    if "top_tokens" not in df.columns:
        df["top_tokens"] = None  

    return df

def parse_top_tokens_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to parse 'top_tokens' as a Python list of (token, float) if it's in
    a string form like "[('pizza', 0.82), ('fruit', 0.79)]".
    We'll store it as a Python object in the DataFrame so we can analyze or print it.
    """
    if "top_tokens" not in df.columns:
        return df  
    
    def try_parse(val):
        if isinstance(val, str) and val.startswith("[(") and val.endswith(")]"):

            try:
                return ast.literal_eval(val)
            except:
                return None
        return None
    
    df["parsed_top_tokens"] = df["top_tokens"].apply(try_parse)
    return df

def summarize_data(df: pd.DataFrame, threshold: float):
    """
    Print basic info, top responding neurons per category, pivot table, etc.
    (Same as your original approach, plus potential references to top_tokens.)
    """
    print("\n=== BASIC INFO ===")
    print(f"Total Rows: {len(df)}")
    models = df["model"].unique()
    print(f"Models found: {models}")
    categories = df["category"].dropna().unique()
    print(f"Categories found: {categories}")

    print("\n=== TOP RESPONDING NEURONS PER CATEGORY ===")

    catdf = df.dropna(subset=["category"])
    grouped = catdf.groupby("category")
    for cat, cat_group in grouped:
        top_5 = cat_group.nlargest(5, "avg_activation")
        print(f"\nCategory: {cat}")
        for idx, row in top_5.iterrows():
            model = row["model"]
            layer = int(row["layer"])
            neuron = int(row["neuron"])
            activation = row["avg_activation"]
            tokens = row.get("parsed_top_tokens", None)
            tokens_str = ""
            if isinstance(tokens, list) and len(tokens) > 0:
                tokens_str = " | Tokens: " + str(tokens[:2])
            print(f"  Model={model}, Layer={layer}, Neuron={neuron}, Activation={activation:.4f}{tokens_str}")

    print(f"\n=== NEURONS EXCEEDING THRESHOLD ({threshold}) PER CATEGORY ===")
    above_thresh_df = catdf[catdf["avg_activation"] >= threshold]
    cat_counts = above_thresh_df.groupby("category")["avg_activation"].count().reset_index()
    for _, row in cat_counts.iterrows():
        cat = row["category"]
        count = row["avg_activation"]
        print(f"  Category '{cat}': {count} neurons above {threshold}")

    pivot_data = above_thresh_df.copy()
    pivot_data["count"] = 1
    pivot_table = pivot_data.pivot_table(
        index="layer",
        columns="category",
        values="count",
        aggfunc="sum",
        fill_value=0
    )
    print("\n=== LAYER-BASED SUMMARY ===")
    print("Number of (layer, neuron) pairs above threshold, by layer & category:\n")
    print(pivot_table)
    print("\n=== END OF SUMMARY ===")

def detect_multi_category_overlap(df: pd.DataFrame):
    """
    Identify neurons that appear in multiple categories.
    We'll group by (model, layer, neuron) and count how many distinct categories it has.
    """
    print("\n=== MULTI-CATEGORY OVERLAP ===")

    catdf = df.dropna(subset=["category"])
    group_cols = ["model", "layer", "neuron"]
    cat_map = catdf.groupby(group_cols)["category"].apply(set).reset_index()
    cat_map["num_categories"] = cat_map["category"].apply(len)
    multi_cats = cat_map[cat_map["num_categories"] >= 2]
    if len(multi_cats) == 0:
        print("No neurons found that match multiple categories.")
        return

    print(f"{len(multi_cats)} neurons match multiple categories.")
    for idx, row in multi_cats.iterrows():
        model = row["model"]
        layer = row["layer"]
        neuron = row["neuron"]
        cats = row["category"]
        print(f"  {model}: Layer {layer}, Neuron {neuron} => categories: {cats}")

def compute_neuron_activation_correlation(df: pd.DataFrame):
    """
    Build a pivot table of (layer, neuron) vs category => avg_activation
    Then compute correlation either across categories or across neurons.
    We'll do across categories for demonstration.
    """
    print("\n=== NEURON ACTIVATION CORRELATION AMONG CATEGORIES ===")
    catdf = df.dropna(subset=["category"])
    pivot = catdf.pivot_table(
        index=["layer", "neuron"],
        columns="category",
        values="avg_activation",
        aggfunc="mean",
        fill_value=0
    )
    cat_corr = pivot.corr()
    print("Category-Category Correlation Matrix:\n")
    print(cat_corr)
    print("\n(Interpretation: 1.0 => categories always co-activate the same neurons, 0 => no correlation)")

def main():
    parser = argparse.ArgumentParser(
        description="interpret_map.py (Advanced) - interpret CSV from neuromap"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--threshold", type=float, default=0.2, 
                        help="Threshold for counting a neuron in category summarization.")
    parser.add_argument("--multi_category", action="store_true", 
                        help="Show neurons that belong to multiple categories.")
    parser.add_argument("--correlation", action="store_true",
                        help="Compute correlation among categories.")
    args = parser.parse_args()

    df = load_csv(args.csv)
    df = parse_top_tokens_column(df)
    
    summarize_data(df, threshold=args.threshold)
    
    if args.multi_category:
        detect_multi_category_overlap(df)
    
    if args.correlation:
        compute_neuron_activation_correlation(df)

if __name__ == "__main__":
    main()
