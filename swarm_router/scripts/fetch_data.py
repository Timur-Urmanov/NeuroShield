"""Download external task sets for Phase 2 into data/ (gitignored) and write data/SOURCES.json.

  python3 scripts/fetch_data.py            # needs pyarrow and network access to huggingface.co

Sources (Hugging Face datasets, test splits):
  GSM8K   openai/gsm8k (main/test)                     MIT              -> data/gsm8k_test500.jsonl
  MBPP    google-research-datasets/mbpp (full/test)    CC BY 4.0        -> data/mbpp_test.jsonl
  AG News fancyzhx/ag_news (test)                      see dataset card -> data/text_agnews_test500.jsonl
Subsets (GSM8K 500 of 1319, AG News 500 of 7600, class-balanced) are drawn with a fixed seed so the
three kinds have equal weight in the task stream. SHA-256 of every downloaded file and every output
file is recorded in data/SOURCES.json.
"""
import hashlib, io, json, os, sys, urllib.request

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
HF = "https://huggingface.co/datasets/"
SRC = {
    "gsm8k": "openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
    "mbpp": "google-research-datasets/mbpp/resolve/main/full/test-00000-of-00001.parquet",
    "ag_news": "fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet",
}
SEED = 20261121
AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def get(path):
    with urllib.request.urlopen(HF + path, timeout=300) as r:
        b = r.read()
    return b, hashlib.sha256(b).hexdigest()


def write(name, rows):
    p = os.path.join(DATA, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(SEED)
    meta = {"seed": SEED, "files": {}}

    b, h = get(SRC["gsm8k"]); t = pq.read_table(io.BytesIO(b)).to_pylist()
    idx = sorted(rng.choice(len(t), 500, replace=False).tolist())
    rows = [dict(id=f"gsm8k-test-{i}", question=t[i]["question"], answer=t[i]["answer"]) for i in idx]
    meta["files"]["gsm8k_test500.jsonl"] = dict(source=SRC["gsm8k"], source_sha256=h, n_source=len(t),
                                               n=len(rows), sha256=write("gsm8k_test500.jsonl", rows))

    b, h = get(SRC["mbpp"]); t = pq.read_table(io.BytesIO(b)).to_pylist()
    rows = [dict(task_id=f"mbpp-{r['task_id']}", text=r["text"], code=r["code"], test_list=list(r["test_list"]),
                 test_setup_code=r.get("test_setup_code") or "",
                 challenge_test_list=list(r.get("challenge_test_list") or [])) for r in t]
    meta["files"]["mbpp_test.jsonl"] = dict(source=SRC["mbpp"], source_sha256=h, n_source=len(t), n=len(rows),
                                           sha256=write("mbpp_test.jsonl", rows))

    b, h = get(SRC["ag_news"]); t = pq.read_table(io.BytesIO(b)).to_pylist()
    by = {k: [i for i, r in enumerate(t) if r["label"] == k] for k in range(4)}
    idx = sorted(sum((rng.choice(by[k], 125, replace=False).tolist() for k in range(4)), []))
    rows = []
    for i in idx:
        lab = AG_LABELS[t[i]["label"]]
        rows.append(dict(id=f"agnews-test-{i}", answer=lab, labels=AG_LABELS,
                         aliases=["Science/Technology", "Sci-Tech", "Science and Technology"] if lab == "Sci/Tech" else [],
                         prompt=("Classify the news article into exactly one category: World, Sports, Business, "
                                 f"Sci/Tech.\n\nArticle: {t[i]['text'].strip()}")))
    meta["files"]["text_agnews_test500.jsonl"] = dict(source=SRC["ag_news"], source_sha256=h, n_source=len(t),
                                                     n=len(rows), sha256=write("text_agnews_test500.jsonl", rows))
    json.dump(meta, open(os.path.join(DATA, "SOURCES.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    sys.exit(main())
