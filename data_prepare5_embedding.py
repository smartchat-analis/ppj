import os
import json
import time
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from openai import OpenAI

# =========================================================
# LOAD PAIRS
# =========================================================
def load_pairs(path):
    return pd.read_json(path)

PAIR_PATH = "data/pairs_perpanjangan_with_intent_and_score.json"
df_pairs = load_pairs(PAIR_PATH)

print("Rows loaded:", len(df_pairs))

# =========================================================
# MASUK API
# =========================================================
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================================================
# EMBEDDING
# =========================================================
def build_text_for_embedding(user_message: str):
    return user_message.strip().lower()

def generate_embedding(text):
    res = CLIENT.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return res.data[0].embedding

# Build text_for_embedding
df_pairs["text_for_embedding"] = df_pairs.apply(
    lambda r: build_text_for_embedding(
        r["user_message"]
    ),
    axis=1
)

# Generate embeddings
tqdm.pandas()
df_pairs["embedding"] = df_pairs["text_for_embedding"].progress_apply(generate_embedding)

print("Embedding complete:", len(df_pairs))

# =========================================================
# SAVE EMBEDDING
# =========================================================
SAVE_PATH = "model/pairs_perpanjangan_with_intent_embedding.json"

df_pairs.to_json(
    SAVE_PATH,
    orient="records",
    lines=True,
    force_ascii=False
)

print("Saved:", SAVE_PATH)

