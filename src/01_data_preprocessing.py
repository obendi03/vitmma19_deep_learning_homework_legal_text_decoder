import os
import re
import json
import zipfile
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_annotations(json_path):
    with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    records = []
    for item in data:
        text = item.get("data", {}).get("text", "")
        annotations = item.get("annotations", [])
        rating = None
        if annotations and "result" in annotations[0]:
            result = annotations[0]["result"]
            if result and "value" in result[0]:
                rating = result[0]["value"].get("choices")[0]
        records.append({"text": text, "rating": rating})

    return pd.DataFrame(records)


filter_digit = lambda x: re.search(r"(\d).+", x).group(1) if x and re.search(r"(\d).+", x) else None


def extract_zip_next_to_script(zip_path):
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    extract_to = script_dir / "extracted_zip"
    extract_to.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"ZIP extracted to: {extract_to}")
    return extract_to


def find_legal_text_decoder_dir(root):
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "legaltextdecoder":
            return p
    raise RuntimeError("Could not find 'legaltextdecoder' directory inside ZIP.")


def load_annotations_from_folder(root_dir, skip_consensus=True):
    all_dfs = []
    consensus_dfs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        is_consensus = "consensus" in dirpath.lower()
        if skip_consensus and is_consensus:
            continue
        if not filenames:
            continue
        json_files = [f for f in filenames if f.endswith(".json")]
        if not json_files:
            continue
        merged_files = [f for f in json_files if f.startswith("merged")]
        files_to_load = merged_files if merged_files else json_files
        for json_file in files_to_load:
            json_path = Path(dirpath) / json_file
            try:
                df = load_annotations(json_path)
                df["rating"] = df["rating"].apply(filter_digit).astype("Int64")
                if is_consensus:
                    consensus_dfs.append(df)
                else:
                    all_dfs.append(df)
            except Exception as e:
                print(f"ERROR loading {json_file}: {e}")

    base_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    consensus_df = pd.concat(consensus_dfs, ignore_index=True) if consensus_dfs else pd.DataFrame()

    return base_df, consensus_df




if __name__ == "__main__":
    DATA_DIR = "/app/data"
    os.makedirs(DATA_DIR, exist_ok=True)

    zip_path = os.path.join(DATA_DIR, "downloaded.zip")
    extracted_root = extract_zip_next_to_script(zip_path)
    legal_dir = find_legal_text_decoder_dir(extracted_root)

    # Betöltés
    base_df, consensus_df = load_annotations_from_folder(legal_dir, skip_consensus=False)


    # Külön kezeljük a consensus mappát
    consensus_only = consensus_df.copy()
    consensus_only.to_csv(os.path.join(DATA_DIR, "consensus.csv"), index=False)

    print("Base DF columns:", base_df.columns)
    print("Consensus DF columns:", consensus_only.columns)
    print("Base DF shape:", base_df.shape)
    print("Consensus DF shape:", consensus_only.shape)


    # Szűrés: a base_df-ből eltávolítjuk a consensusban lévő text-eket
    before_filter = len(base_df)
    base_df = base_df[~base_df["text"].isin(consensus_only["text"])].reset_index(drop=True)
    num_filtered_out_from_base = before_filter - len(base_df)
    print(f"Number of rows filtered out from base_df because they exist in consensus: {num_filtered_out_from_base}")

    # Átlagoljuk a ratinget azonos szövegekre a consensusban
    consensus_only = (
        consensus_only.groupby("text", as_index=False)
        .agg({"rating": "mean"})
        .rename(columns={"rating": "rating"})
    )
    num_aggregated = len(consensus_only)
    print(f"Number of aggregated consensus texts (after averaging): {num_aggregated}")

    # Mentés inference.csv-be
    inference_path = os.path.join(DATA_DIR, "inference.csv")
    consensus_only.to_csv(inference_path, index=False)
    print(f"Consensus inference saved to: {inference_path}")

    # Train/Val/Test split a szűrt base_df-ből
    train_val_df, test_df = train_test_split(base_df, test_size=0.1, random_state=42, shuffle=True)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, shuffle=True)

    # Mentés
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    print("Train/Val/Test split saved.")
