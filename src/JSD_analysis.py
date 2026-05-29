import os
import json
import re
import csv
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
from scipy.spatial.distance import jensenshannon
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

INPUT_DIR  = "output/generations"
OUTPUT_DIR = "charts/jsd"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def freq_to_dist(freq: Counter, vocab: list[str]) -> np.ndarray:
    vec = np.array([freq.get(w, 0) for w in vocab], dtype=float)
    total = vec.sum()
    if total == 0:
        return np.ones(len(vocab)) / len(vocab)
    return vec / total


def jsd(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    return float(jensenshannon(dist_a, dist_b) ** 2)


def load_all_data(input_dir: str):
    records = []
    for model_folder in sorted(os.listdir(input_dir)):
        model_path = os.path.join(input_dir, model_folder)
        if not os.path.isdir(model_path):
            continue
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(model_path, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            model = data.get("model", model_folder)
            topic = data.get("topic", fname.replace(".json", ""))
            for gen in data.get("generations", []):
                for rtype, rdata in gen.get("replies", {}).items():
                    answer = rdata.get("answer", "")
                    if not answer:
                        continue
                    records.append({
                        "model": model.split("/")[1],
                        "topic": topic,
                        "reply_type": rtype,
                        "tokens": tokenize(answer),
                    })

    model_names = sorted({r["model"] for r in records})
    topics = sorted({r["topic"] for r in records})
    reply_types = sorted({r["reply_type"] for r in records})
    return records, model_names, topics, reply_types


def aggregate_tokens(records, key_fn):
    buckets: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        buckets[key_fn(r)].update(r["tokens"])
    return dict(buckets)


def build_shared_vocab(counters: dict) -> list[str]:
    vocab: set[str] = set()
    for c in counters.values():
        vocab.update(c.keys())
    return sorted(vocab)


def pairwise_jsd(labels: list[str], counters: dict) -> list[tuple]:
    """Return list of (label_a, label_b, jsd_score) for all pairs."""
    vocab = build_shared_vocab(counters)
    dists = {lbl: freq_to_dist(counters[lbl], vocab) for lbl in labels}
    rows = []
    for a, b in combinations(labels, 2):
        rows.append((a, b, round(jsd(dists[a], dists[b]),3)))
    return rows


def write_csv(path: str, header: list[str], rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  Saved → {path}  ({len(rows)} rows)")


def main():
    print("Loading data …")
    records, model_names, topics, reply_types = load_all_data(INPUT_DIR)
    print(f"  {len(records)} records | {len(model_names)} models | "
          f"{len(topics)} topics | {len(reply_types)} reply types")

    # ── 1. Overall pairwise JSD ────────────────────────────────────────────

    print("\n[1/4] Overall JSD …")

    # 1a. Model vs model
    model_counters = aggregate_tokens(records, lambda r: r["model"])
    rows = pairwise_jsd(model_names, model_counters)
    write_csv(
        os.path.join(OUTPUT_DIR, "overall_models.csv"),
        ["model1", "model2", "jsd"],
        rows,
    )

    # 1b. Topic vs topic
    topic_counters = aggregate_tokens(records, lambda r: r["topic"])
    rows = pairwise_jsd(topics, topic_counters)
    write_csv(
        os.path.join(OUTPUT_DIR, "overall_topics.csv"),
        ["topic1", "topic2", "jsd"],
        rows,
    )

    # 1c. Response type vs response type
    rtype_counters = aggregate_tokens(records, lambda r: r["reply_type"])
    rows = pairwise_jsd(reply_types, rtype_counters)
    write_csv(
        os.path.join(OUTPUT_DIR, "overall_response_types.csv"),
        ["response_type1", "response_type2", "jsd"],
        rows,
    )

    # ── 2. Cross model × topic ─────────────────────────────────────────────

    print("\n[2/4] Cross model × topic JSD …")
    rows = []
    for topic in topics:
        sub = [r for r in records if r["topic"] == topic]
        counters = aggregate_tokens(sub, lambda r: r["model"])
        active = [m for m in model_names if m in counters]
        if len(active) < 2:
            continue
        for a, b, score in pairwise_jsd(active, counters):
            rows.append((topic, a, b, score))
    write_csv(
        os.path.join(OUTPUT_DIR, "cross_model_topic.csv"),
        ["topic", "model1", "model2", "jsd"],
        rows,
    )

    # ── 3. Cross model × response type ────────────────────────────────────

    print("\n[3/4] Cross model × response type JSD …")
    rows = []
    for rtype in reply_types:
        sub = [r for r in records if r["reply_type"] == rtype]
        counters = aggregate_tokens(sub, lambda r: r["model"])
        active = [m for m in model_names if m in counters]
        if len(active) < 2:
            continue
        for a, b, score in pairwise_jsd(active, counters):
            rows.append((rtype, a, b, score))
    write_csv(
        os.path.join(OUTPUT_DIR, "cross_model_response_type.csv"),
        ["response_type", "model1", "model2", "jsd"],
        rows,
    )

    # ── 4. Cross model × topic × response type ────────────────────────────

    print("\n[4/4] Cross model × topic × response type JSD …")
    rows = []
    for topic in topics:
        for rtype in reply_types:
            sub = [r for r in records
                   if r["topic"] == topic and r["reply_type"] == rtype]
            if not sub:
                continue
            counters = aggregate_tokens(sub, lambda r: r["model"])
            active = [m for m in model_names if m in counters]
            if len(active) < 2:
                continue
            for a, b, score in pairwise_jsd(active, counters):
                rows.append((topic, rtype, a, b, score))
    write_csv(
        os.path.join(OUTPUT_DIR, "cross_model_topic_response_type.csv"),
        ["topic", "response_type", "model1", "model2", "jsd"],
        rows,
    )

    print("\nDone. All CSV files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
