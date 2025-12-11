# mia_pairwise.py
"""
Pair-wise Membership-Inference Attack for Sentence Embeddings
Paper: 'Exposing the limitations of embedding models', arXiv:2004.00053
Author: <you>
"""

from __future__ import annotations
import argparse, json, random, pathlib
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import nn, optim

# from sentence_transformers import SentenceTransformer, util

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


# -----------------------------------------------------------
# 1.  Embedding helper
# -----------------------------------------------------------
# def embed(model: SentenceTransformer, sentences: List[str]) -> np.ndarray:
#     """Return L2-normalised embeddings (batch-encoded)."""
#     return model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)


# -----------------------------------------------------------
# 2.  Data utilities
# -----------------------------------------------------------
Pair = Tuple[str, str]

def load_pairs(path: pathlib.Path) -> List[Pair]:
    """
    Each line: sentenceA<TAB>sentenceB
    """
    with open(path, encoding="utf8") as f:
        return [tuple(line.rstrip("\n").split("\t", 1)) for line in f]


def load_features(features_path):
    dfs = []
    for file in tqdm(sorted(glob.glob(os.path.join(features_path, '*.parquet')))[:1000]):
        df = pd.read_parquet(file)
        dfs.append(df)
    if len(dfs) > 1:
        return np.array(pd.concat(dfs)['features'].values.tolist())
    else:
        return np.zeros((0,))

def build_dataset(
    train_features_path: str,
    test_features_path: str,
    seed: int = 7,
    train_frac: float = 0.5,
) -> Tuple[List[Pair], List[int], List[Pair], List[int]]:
    """Return (train_pairs, train_labels, test_pairs, test_labels)."""
    rng = random.Random(seed)
    members = load_features(train_features_path)
    print(members.shape)
    nonmembers = load_features(test_features_path)
    print(nonmembers.shape)
    # Convert to lists for consistent shuffling and slicing
    members = list(members)
    nonmembers = list(nonmembers)
    # Shuffle and pair within each set
    rng.shuffle(members)
    rng.shuffle(nonmembers)
    # Make pairs (discard last if odd)
    def make_pairs(lst):
        n = len(lst) // 2 * 2  # even number
        return [(lst[i], lst[i+1]) for i in range(0, n, 2)]
    member_pairs = make_pairs(members)
    nonmember_pairs = make_pairs(nonmembers)
    # Balance the two classes
    n = min(len(member_pairs), len(nonmember_pairs))
    member_pairs = member_pairs[:n]
    nonmember_pairs = nonmember_pairs[:n]
    all_pairs  = member_pairs + nonmember_pairs
    all_labels = [1]*n + [0]*n
    # Simple random split
    idx = list(range(2*n))
    rng.shuffle(idx)
    split = int(train_frac * 2 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    train_pairs  = [all_pairs[i]  for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_pairs   = [all_pairs[i]  for i in test_idx]
    test_labels  = [all_labels[i] for i in test_idx]
    return train_pairs, train_labels, test_pairs, test_labels


# -----------------------------------------------------------
# 3.  Attack model
# -----------------------------------------------------------
class PairwiseMIA:
    """
    A single-feature logistic-regression attack:
        feature = cosine-similarity(poly Norm(u), Norm(v))
    """

    def __init__(self):
        # self.clf = LogisticRegression(solver="lbfgs")
        # self.clf = RandomForestClassifier()
        self.clf = KNeighborsClassifier()

    def fit(self, sims: np.ndarray, labels: List[int]):
        X = sims.reshape(-1, 1)
        self.clf.fit(X, labels)

    def predict_prob(self, sims: np.ndarray) -> np.ndarray:
        X = sims.reshape(-1, 1)
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, sims: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_prob(sims) >= threshold).astype(int)


def cosine_similarities(pairs: List[Pair]) -> np.ndarray:
    """Compute cosine similarity for each sentence pair."""
    emb_a, emb_b = zip(*pairs)
    emb_a = np.stack(emb_a)
    emb_b = np.stack(emb_b)
    dot_products = np.sum(emb_a * emb_b, axis=1)
    norms_a = np.linalg.norm(emb_a, axis=1)
    norms_b = np.linalg.norm(emb_b, axis=1)
    cosine_sim = dot_products / (norms_a * norms_b)
    return cosine_sim


# -----------------------------------------------------------
# 4.  Train & evaluate
# -----------------------------------------------------------
def adversarial_advantage(preds: np.ndarray, labels: List[int]) -> float:
    preds, labels = np.asarray(preds), np.asarray(labels)
    in_rate  = preds[labels == 1].mean()
    out_rate = preds[labels == 0].mean()
    return abs(in_rate - out_rate)


def main(
    train_features_path: str,
    test_features_path: str,
    train_frac: float = 0.33,
    random_seed: int = 42,
):
    # model = SentenceTransformer(model_name)

    tr_pairs, tr_labels, te_pairs, te_labels = build_dataset(
        train_features_path, test_features_path, seed=random_seed, train_frac=train_frac
    )

    # Compute similarities
    tr_sims = cosine_similarities(tr_pairs)
    te_sims = cosine_similarities(te_pairs)

    # Train attack
    attack = PairwiseMIA()
    attack.fit(tr_sims, tr_labels)

    # Evaluate
    prob_te  = attack.predict_prob(te_sims)
    pred_te  = (prob_te >= 0.5).astype(int)

    auc  = roc_auc_score(te_labels, prob_te)
    adv  = adversarial_advantage(pred_te, te_labels)

    metrics = {"AUC": auc, "Advantage": adv}
    print(json.dumps(metrics, indent=2))


class LearnedProjectionMIA(nn.Module):
    """
    W_m : (d,d) full matrix, learned to maximise BCE between
           similarity and membership label.
    """

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))  # init as identity

    def forward(self, u, v):
        # u,v : (batch,d) torch.float32, already L2-normalised
        u_p = torch.matmul(u, self.W)            # (batch,d)
        v_p = torch.matmul(v, self.W)            # (batch,d)
        # cosine (dot because all vectors are L2-norm 1 after proj?  → renorm)
        u_p = nn.functional.normalize(u_p, p=2, dim=1)
        v_p = nn.functional.normalize(v_p, p=2, dim=1)
        sim = torch.sum(u_p * v_p, dim=1)        # (batch,)
        return sim  # logit before sigmoid


def train_attack(model, u_tr, v_tr, y_tr, epochs=10, lr=1e-3, bs=256):
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(u_tr, v_tr, y_tr)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=bs, shuffle=True, drop_last=False
    )
    for _ in range(epochs):
        for u_b, v_b, y_b in loader:
            opt.zero_grad()
            logits = model(u_b, v_b)
            loss = crit(logits, y_b)
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate(model, u, v, y, roc_curve_path=None):
    logits = model(u, v)
    prob = torch.sigmoid(logits).cpu().numpy()
    auc = roc_auc_score(y.cpu().numpy(), prob)
    pred = (prob >= 0.5).astype(int)
    advantage = abs(pred[y == 1].mean() - pred[y == 0].mean())
    # Save ROC curve if requested
    if roc_curve_path is not None:
        fpr, tpr, _ = roc_curve(y.cpu().numpy(), prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(roc_curve_path)
        plt.close()
    return auc, advantage


# ---------------------------------------------------------------------
# 3. driver
# ---------------------------------------------------------------------
def main_torch(train_features_path, test_features_path, train_frac, random_seed, epochs, batch_size, lr):
    # 3.1 build splits
    tr_pairs, y_tr, te_pairs, y_te = build_dataset(
        train_features_path, test_features_path, seed=random_seed, train_frac=train_frac
    )

    # 3.2 embed once
    # enc = SentenceTransformer(args.model)
    u_tr_np = np.array([p[0] for p in tr_pairs])
    v_tr_np = np.array([p[1] for p in tr_pairs])
    u_te_np = np.array([p[0] for p in te_pairs])
    v_te_np = np.array([p[1] for p in te_pairs])
    print(u_tr_np.shape)
    print(v_tr_np.shape)
    print(u_te_np.shape)
    print(v_te_np.shape)
    y_tr = np.array(y_tr)
    y_te = np.array(y_te)

    # 3.3 torch tensors (ensure float32 for BCE)
    u_tr = torch.from_numpy(u_tr_np).float()
    v_tr = torch.from_numpy(v_tr_np).float()
    u_te = torch.from_numpy(u_te_np).float()
    v_te = torch.from_numpy(v_te_np).float()
    y_tr = torch.from_numpy(y_tr).float()
    y_te = torch.from_numpy(y_te).float()

    # 3.4 train attack
    mia = LearnedProjectionMIA(dim=u_tr.shape[1])
    train_attack(mia, u_tr, v_tr, y_tr, epochs=epochs, lr=lr, bs=batch_size)

    # 3.5 evaluate
    auc, adv = evaluate(mia, u_te, v_te, y_te, roc_curve_path='roc_curve.png')
    print(json.dumps({"AUC": auc, "Advantage": adv}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pair-wise MIA against sentence encoders")
    # parser.add_argument("--member_pairs", required=True, help="TSV file: observed-together pairs (label 1)")
    # parser.add_argument("--nonmember_pairs", required=True, help="TSV file: random pairs (label 0)")
    # parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--train_frac", type=float, default=0.3, help="Fraction of labelled data for attack training")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    args = parser.parse_args()


    base_folder_train = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_train_mia_20'
    base_folder_test = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_test_mia_20'

    # base_folder_train = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_train'
    # base_folder_test = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/cehrgpt_flo/features/patient_sequence/cehrgpt_test'

    train_features_path = os.path.join(base_folder_train, 'features_without_label/train_features')
    test_features_path = os.path.join(base_folder_test, 'features_without_label/train_features')

    main_torch(train_features_path, test_features_path, args.train_frac, args.random_seed, args.epochs, args.batch_size, args.lr)