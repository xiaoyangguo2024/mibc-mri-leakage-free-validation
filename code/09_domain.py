"""09 - Domain shift: how identifiable is the centre, and how does centre membership confound pooled CV?

  A  centre identifiability: 4-class prediction of the centre from (i) all radiomics features,
     (ii) each frozen deep feature set, (iii) the tumour-volume features alone; patient-grouped 5-fold CV,
     macro one-vs-rest AUC and accuracy
  B  2-D t-SNE embeddings of radiomics and deep features (for Fig. 2), coloured by centre and label
  C  per-centre distributions of tumour volume and normalised tumour intensity
Outputs: results/tables/domain_identifiability.csv, results/tables/domain_embeddings.csv,
         results/tables/domain_distributions.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bcmri import config as C
from bcmri import data as D


def centre_cv(X, centre, groups, seed, pca=None):
    y = pd.Categorical(centre, categories=C.CENTRES).codes
    P = np.zeros((len(y), len(C.CENTRES)))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, groups):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=min(pca, len(tr) - 1), random_state=seed))
        steps.append(LogisticRegression(C=0.1, max_iter=5000))
        m = make_pipeline(*steps).fit(X[tr], y[tr])
        P[te] = m.predict_proba(X[te])
    return {"macro_auc_ovr": roc_auc_score(y, P, multi_class="ovr", average="macro"),
            "accuracy": accuracy_score(y, P.argmax(1)),
            **{f"auc_{c}": roc_auc_score((y == k).astype(int), P[:, k]) for k, c in enumerate(C.CENTRES)}}


def main():
    coh = D.load_cohort()
    R = D.load_radiomics("orig").reindex(coh.index)
    R = R.loc[:, np.isfinite(R.values).all(axis=0) & (R.std() > 1e-8).values]
    deep = {b: D.load_deep(b).reindex(coh.index) for b in D.BACKBONES}
    S = D.size_features(D.load_radiomics("orig").reindex(coh.index))
    g = coh["patient_id"].values
    rows = []
    sets = {"Radiomics (IT+PT, all)": (R.values, None),
            "Radiomics, first-order only": (R[[c for c in R.columns if "_firstorder_" in c]].values, None),
            "Radiomics, shape only": (R[[c for c in R.columns if "_shape_" in c]].values, None),
            "Tumour size (2 features)": (S.values, None)}
    for b, X in deep.items():
        sets[f"Deep: {b}"] = (X.values, 50)
    for name, (X, pca) in sets.items():
        res = centre_cv(X, coh["centre"].values, g, C.SEED, pca)
        rows.append({"feature_set": name, "n_features": X.shape[1], **res})
        print(name, {k: round(v, 3) for k, v in res.items()}, flush=True)
    pd.DataFrame(rows).to_csv(C.TABLES / "domain_identifiability.csv", index=False)

    emb = []
    for name, X, npc in (("Radiomics", R.values, 30), *[(b, deep[b].values, 30) for b in D.BACKBONES]):
        Z = PCA(n_components=npc, random_state=C.SEED).fit_transform(StandardScaler().fit_transform(X))
        T = TSNE(n_components=2, perplexity=30, init="pca", random_state=C.SEED).fit_transform(Z)
        emb.append(pd.DataFrame({"feature_set": name, "case_id": coh.index, "x": T[:, 0], "y_": T[:, 1],
                                 "centre": coh["centre"].values, "label": coh["y"].values}))
    pd.concat(emb).to_csv(C.TABLES / "domain_embeddings.csv", index=False)

    dist = pd.DataFrame({"case_id": coh.index, "centre": coh["centre"].values, "label": coh["y"].values,
                         "volume_ml": coh["tumour_ml"].values,
                         "IT_median_norm": R["IT_original_firstorder_Median"].values,
                         "PT_median_norm": R["PT_original_firstorder_Median"].values,
                         "IT_entropy": R["IT_original_firstorder_Entropy"].values})
    dist.to_csv(C.TABLES / "domain_distributions.csv", index=False)


if __name__ == "__main__":
    main()
