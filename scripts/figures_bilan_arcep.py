"""Figures de synthèse des expériences ARCEP (séances 8 à 11) pour le compte rendu 4.

Usage (depuis la racine du dépôt) : .venv\\Scripts\\python scripts\\figures_bilan_arcep.py
Écrit dans results/figures/ :
  - figures tirées des tableaux de results/tables (aucun réentraînement) ;
  - courbes d'apprentissage de XGBoost (échec web et débit DOWNLOAD), avec
    dérive (passé -> année suivante) et sans dérive (moitié de 2025 -> autre moitié) ;
  - courbe d'apprentissage du MLP (pertes relevées dans notebooks/10_mlp.ipynb, graine 0).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parents[1]
sys.path.append(str(RACINE / "src"))
TABLES = RACINE / "results" / "tables"
FIGURES = RACINE / "results" / "figures"


def enregistrer(fig, nom):
    fig.tight_layout()
    fig.savefig(FIGURES / nom, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("écrit :", nom)


def etiqueter(ax, barres, fmt="{:.3f}"):
    for b in barres:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.04), fmt.format(v),
                ha="center", fontsize=7.5)


# ------------------------------------------------------------------ figures tirées des tableaux
def classification_web():
    t = pd.read_csv(TABLES / "web_variables_contexte_test_2025.csv", index_col=[0, 1])
    m = pd.read_csv(TABLES / "mlp_web_test_2025.csv", index_col=0)
    lignes = [
        ("règle RSRP", t.loc[("règle", "RSRP")]),
        ("logistique\n(base)", t.loc[("base", "logistique")]),
        ("XGBoost\n(base)", t.loc[("base", "xgboost")]),
        ("logistique\n(+ test préc.)", t.loc[("prédiction", "logistique")]),
        ("XGBoost\n(+ test préc.)", t.loc[("prédiction", "xgboost")]),
        ("MLP\n(+ test préc.)", m.loc["moyenne"]),
    ]
    noms = [n for n, _ in lignes]
    x = np.arange(len(noms))
    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - 0.2, [float(r["f1"]) for _, r in lignes], 0.4, label="F1 (seuil réglé sur 2024)")
    b2 = ax.bar(x + 0.2, [float(r["ap"]) for _, r in lignes], 0.4, label="AP")
    etiqueter(ax, b1); etiqueter(ax, b2)
    ax.set_xticks(x, noms, fontsize=8.5); ax.set_ylim(0, 0.95)
    ax.set_ylabel("score sur le test 2025")
    ax.set_title("Échec d'un test web : modèles et jeux de variables (entraînement 2022-2024, test 2025)")
    ax.legend(loc="upper left", fontsize=8.5)
    enregistrer(fig, "arcep_web_modeles_test2025.png")


def derive_web():
    t = pd.read_csv(TABLES / "web_classification_derive.csv", index_col=0)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    b1 = ax.bar(x - 0.2, t["AP temporel"], 0.4, label="temporel (2022-2024 → 2025)")
    b2 = ax.bar(x + 0.2, t["AP intra-2025"], 0.4, label="intra-2025 (sans dérive)")
    etiqueter(ax, b1); etiqueter(ax, b2)
    ax.set_xticks(x, t.index); ax.set_ylim(0.6, 0.8)
    ax.set_ylabel("AP sur 2025")
    ax.set_title("Coût de la dérive (variables de base)")
    ax.legend(fontsize=8.5, loc="upper left")
    enregistrer(fig, "arcep_web_derive.png")


def importance_web():
    t = pd.read_csv(TABLES / "web_importance_permutation.csv", index_col=0).sort_values("baisse_AP")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(t.index, t["baisse_AP"], xerr=t["ecart_type"], color="#4C72B0")
    for i, v in enumerate(t["baisse_AP"]):
        ax.text(max(v, 0) + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("baisse de l'AP quand la variable est mélangée")
    ax.set_title("Importance par permutation (XGBoost, échec web, test 2025)")
    ax.set_xlim(-0.01, 0.52)
    enregistrer(fig, "arcep_web_importance.png")


def regression_debit():
    t = pd.read_csv(TABLES / "debit_regression_test_2025.csv", index_col=0)
    noms = ["médiane\nglobale", "médiane\nzone ×\nopérateur", "linéaire\n(base)", "XGBoost\n(base)",
            "linéaire\n+ débit\nprécédent", "XGBoost\n+ débit\nprécédent"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    b = axes[0].bar(noms, t["R2"], color=["grey", "grey", "#4C72B0", "#DD8452", "#4C72B0", "#DD8452"])
    etiqueter(axes[0], b)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("R² sur 2025"); axes[0].set_title("Qualité de la prédiction")
    b = axes[1].bar(noms, t["biais"], color=["grey", "grey", "#4C72B0", "#DD8452", "#4C72B0", "#DD8452"])
    etiqueter(axes[1], b, "{:.0f}")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("biais moyen (Mbit/s)"); axes[1].set_title("Sous-estimation due à la dérive")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Débit DOWNLOAD : entraînement 2022-2024, test 2025", fontsize=11)
    enregistrer(fig, "arcep_debit_modeles.png")


def strategies_debit():
    t = pd.read_csv(TABLES / "debit_strategies_entrainement.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    couleurs = ["#4C72B0", "#55A868", "grey"]
    b = axes[0].bar(t.index, t["R2"], color=couleurs); etiqueter(axes[0], b)
    axes[0].set_ylim(0.5, 0.72); axes[0].set_ylabel("R² (moitié de 2025)")
    b = axes[1].bar(t.index, t["biais"], color=couleurs); etiqueter(axes[1], b, "{:.0f}")
    axes[1].axhline(0, color="black", lw=0.8); axes[1].set_ylabel("biais (Mbit/s)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Débit DOWNLOAD (XGBoost + débit précédent) : quelles années pour l'entraînement ?", fontsize=11)
    enregistrer(fig, "arcep_debit_strategies.png")


# ------------------------------------------------------------------ courbes d'apprentissage
def courbe_mlp():
    # pertes relevées à la sortie de la cellule d'entraînement de notebooks/10_mlp.ipynb (graine 0)
    entr = [0.5979, 0.5692, 0.5673, 0.5656, 0.5647, 0.5638, 0.5633, 0.5633]
    val = [0.4814, 0.4762, 0.4648, 0.4667, 0.4658, 0.4711, 0.4659, 0.4759]
    ep = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(ep, entr, marker="o", label="entraînement 2022-2023 (avec dropout)")
    ax.plot(ep, val, marker="o", label="validation 2024")
    ax.axvline(3, color="grey", ls="--", lw=1)
    ax.set_xlabel("époque"); ax.set_ylabel("perte (entropie croisée pondérée)")
    ax.set_title("Courbe d'apprentissage du MLP (échec web)")
    ax.legend(fontsize=8.5)
    enregistrer(fig, "arcep_courbe_mlp.png")


def preparer(df, numeriques, booleennes, masque_entr):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    prep = ColumnTransformer([("num", "passthrough", numeriques + booleennes),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), ["operator", "zone", "situation"])])
    X = df[numeriques + booleennes + ["operator", "zone", "situation"]]
    prep.fit(X[masque_entr])
    return lambda m: prep.transform(X[m])


def courbes_xgb(h):
    from xgboost import XGBClassifier, XGBRegressor
    from variables import ajouter_contexte, ajouter_precedent
    rng = np.random.default_rng(0)

    # --- échec web, jeu « prédiction »
    web = h.query("protocole == 'WEB'").copy()
    web["echec"] = (1 - web["succes"]).astype(int)
    web = ajouter_contexte(web)
    annee, y = web["annee"].to_numpy(), web["echec"].to_numpy()
    moitie = rng.random(len(web)) < 0.5
    cas_web = {
        "avec dérive : 2022-2023 → 2024": (annee <= 2023, annee == 2024),
        "sans dérive : ½ 2025 → autre ½ 2025": ((annee == 2025) & moitie, (annee == 2025) & ~moitie),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (titre, (me, mv)) in zip(axes, cas_web.items()):
        tr = preparer(web, ["rsrp", "heure", "echec_prec"], ["rsrp_absent", "prec_absent"], me)
        m = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=0,
                          eval_metric="logloss")
        m.fit(tr(me), y[me], eval_set=[(tr(me), y[me]), (tr(mv), y[mv])], verbose=False)
        r = m.evals_result()
        ax.plot(r["validation_0"]["logloss"], label="entraînement")
        ax.plot(r["validation_1"]["logloss"], label="évaluation")
        ax.set_title(titre, fontsize=10); ax.set_xlabel("nombre d'arbres")
        print(titre, "logloss fin", round(r["validation_0"]["logloss"][-1], 4), round(r["validation_1"]["logloss"][-1], 4),
              "min éval", round(min(r["validation_1"]["logloss"]), 4), int(np.argmin(r["validation_1"]["logloss"])) + 1)
    axes[0].set_ylabel("log-loss"); axes[0].legend()
    fig.suptitle("Courbes d'apprentissage de XGBoost — échec web (test précédent inclus)", fontsize=11)
    enregistrer(fig, "arcep_courbes_xgb_web.png")

    # --- débit DOWNLOAD, jeu « + débit précédent »
    dl = h.query("protocole == 'DOWNLOAD'").copy()
    dl = ajouter_precedent(dl, "bitrate_dl", "debit_prec", delai_max=3600)
    annee, y = dl["annee"].to_numpy(), dl["bitrate_dl"].to_numpy()
    moitie = rng.random(len(dl)) < 0.5
    cas_dl = {
        "avec dérive : 2022-2024 → 2025": (annee <= 2024, annee == 2025),
        "sans dérive : ½ 2025 → autre ½ 2025": ((annee == 2025) & moitie, (annee == 2025) & ~moitie),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (titre, (me, mv)) in zip(axes, cas_dl.items()):
        tr = preparer(dl, ["rsrp", "heure", "debit_prec"], ["rsrp_absent", "debit_prec_absent"], me)
        m = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=0,
                         eval_metric="rmse")
        m.fit(tr(me), y[me], eval_set=[(tr(me), y[me]), (tr(mv), y[mv])], verbose=False)
        r = m.evals_result()
        ax.plot(r["validation_0"]["rmse"], label="entraînement")
        ax.plot(r["validation_1"]["rmse"], label="évaluation")
        ax.set_title(titre, fontsize=10); ax.set_xlabel("nombre d'arbres")
        print(titre, "rmse fin", round(r["validation_0"]["rmse"][-1], 1), round(r["validation_1"]["rmse"][-1], 1),
              "min éval", round(min(r["validation_1"]["rmse"]), 1), int(np.argmin(r["validation_1"]["rmse"])) + 1)
    axes[0].set_ylabel("RMSE (Mbit/s)"); axes[0].legend()
    fig.suptitle("Courbes d'apprentissage de XGBoost — débit DOWNLOAD (débit précédent inclus)", fontsize=11)
    enregistrer(fig, "arcep_courbes_xgb_debit.png")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    for f in [classification_web, derive_web, importance_web, regression_debit, strategies_debit, courbe_mlp]:
        f()
    courbes_xgb(pd.read_parquet(RACINE / "data" / "processed" / "arcep_2022_2025.parquet"))
