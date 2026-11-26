"""Predict a drug's pharmacologic class from the text of its FDA label (Indications and Usage).
TF-IDF and softmax regression written from scratch with numpy. Cross-validation is grouped by drug so
the same drug never appears in both train and test."""
import re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
d = pd.read_csv("data/labels_raw.csv")
d["epc1"] = d.epc.str.split("; ").str[0].str.replace(" [EPC]", "", regex=False)
d["drug"] = d.generic.str.lower().str.split("; ").str[0]
# keep classes with at least 50 labels AND at least 6 different drugs, so a held-out drug group still leaves examples of the class to learn from
nl = d.epc1.value_counts(); nd = d.groupby("epc1").drug.nunique()
top = [c for c in nl.index if nl[c] >= 50 and nd[c] >= 6]
d = d[d.epc1.isin(top)].reset_index(drop=True)
print(len(d), "labels,", d.drug.nunique(), "drugs,", len(top), "classes")

STOP = set("a an and are as at be been by for from has have in is it its may of on or that the this to was were with who which will can use used usage indicated indication indications not also other than such these those when if into including per both only who their they more most all any each within after before during patients patient adult adults children pediatric years age old older".split())
def tokens(text, drug):
    own = set(re.findall(r"[a-z]+", drug))                      # hide the drug's own name
    t = re.findall(r"[a-z][a-z\-]{2,}", text.lower())
    return [w for w in t if w not in STOP and w not in own]
docs = [tokens(t, g) for t, g in zip(d.indications, d.drug)]

# vocabulary: unigrams seen in at least 5 labels, top 4000 by document frequency
df_count = {}
for toks in docs:
    for w in set(toks): df_count[w] = df_count.get(w, 0) + 1
vocab = [w for w, c in sorted(df_count.items(), key=lambda x: -x[1]) if c >= 5][:4000]
idx = {w: i for i, w in enumerate(vocab)}
N = len(docs); X = np.zeros((N, len(vocab)))
for i, toks in enumerate(docs):
    for w in toks:
        if w in idx: X[i, idx[w]] += 1
X = np.log1p(X)
idf = np.log((1 + N) / (1 + (X > 0).sum(0))) + 1
X = X * idf; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

cls = {c: k for k, c in enumerate(top)}; y = d.epc1.map(cls).values; K = len(top)

def softmax_fit(X, y, lam=1e-3, lr=2.0, iters=300):
    W = np.zeros((X.shape[1], K)); b = np.zeros(K); Y = np.eye(K)[y]
    for _ in range(iters):
        Z = X @ W + b; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(X); W -= lr * (X.T @ G + lam * W); b -= lr * G.sum(0)
    return W, b

def centroid_fit(X, y):
    C = np.array([X[y == k].mean(0) if (y == k).any() else np.zeros(X.shape[1]) for k in range(K)]); return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)

def macro_f1(y, p):
    f = []
    for k in range(K):
        tp = ((y == k) & (p == k)).sum(); fp = ((y != k) & (p == k)).sum(); fn = ((y == k) & (p != k)).sum()
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(f)

drugs = d.drug.unique(); rng.shuffle(drugs); fold_of = {g: i % 5 for i, g in enumerate(drugs)}
fold = d.drug.map(fold_of).values
pred = {"majority": np.zeros(N, int), "centroid": np.zeros(N, int), "softmax": np.zeros(N, int)}
for f in range(5):
    tr, te = fold != f, fold == f
    pred["majority"][te] = np.bincount(y[tr], minlength=K).argmax()
    C = centroid_fit(X[tr], y[tr]); pred["centroid"][te] = (X[te] @ C.T).argmax(1)
    W, b = softmax_fit(X[tr], y[tr]); pred["softmax"][te] = (X[te] @ W + b).argmax(1)
rows = []
for m, p in pred.items():
    f1 = macro_f1(y, p); rows.append({"model": m, "accuracy": round((p == y).mean(), 3), "macro_f1": round(f1.mean(), 3)})
res = pd.DataFrame(rows); res.to_csv("data/results.csv", index=False); print(res)

f1 = macro_f1(y, pred["centroid"])
pc = pd.DataFrame({"class": top, "n_labels": [int((y == k).sum()) for k in range(K)], "n_drugs": [int(d[y == k].drug.nunique()) for k in range(K)], "f1": f1.round(2)}).sort_values("f1")
pc.to_csv("data/per_class_f1.csv", index=False); print(pc.to_string())

# most-confused pairs
cm = np.zeros((K, K), int)
for a, b_ in zip(y, pred["centroid"]): cm[a, b_] += 1
conf = sorted([(cm[i, j], top[i], top[j]) for i in range(K) for j in range(K) if i != j], reverse=True)[:8]
pd.DataFrame(conf, columns=["count", "true_class", "predicted_class"]).to_csv("data/top_confusions.csv", index=False); print(conf)

# top terms per class from a model fit on everything
W, b = softmax_fit(X, y)
terms = pd.DataFrame({top[k]: [vocab[i] for i in np.argsort(-W[:, k])[:8]] for k in range(K)}).T
terms.columns = [f"term_{i+1}" for i in range(8)]; terms.to_csv("data/top_terms_per_class.csv"); print(terms.to_string())

# figures
fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(pc["class"], pc.f1, color="#4472c4"); ax.set_xlabel("F1 on drugs the model never saw (5-fold grouped CV)")
ax.set_title("Nearest-centroid model, by drug class"); fig.tight_layout(); fig.savefig("figures/f1_by_class.png", dpi=150)
fig, ax = plt.subplots(figsize=(7, 3)); ax.barh(res.model, res.macro_f1, color=["grey", "#f4b183", "#4472c4"]); ax.set_xlabel("Macro F1"); fig.tight_layout(); fig.savefig("figures/model_comparison.png", dpi=150)
