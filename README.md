# Analyse des performances et détection d'anomalies dans les réseaux 5G/6G par Machine Learning et Deep Learning

*Performance analysis and anomaly detection in 5G/6G networks with Machine Learning and Deep Learning*

Projet SMART-ICT S5 — INPT DE RABAT / SMART ICT INE3 — Année 2026-2027
Auteur : ABOUBAKAR SIDIKI

---

## Objectif

Étudier comment les techniques d'intelligence artificielle peuvent exploiter les données des réseaux 5G/6G pour :

1. **Prédire des indicateurs de performance (KPI)** — par exemple le débit descendant à court terme à partir des mesures radio (RSRP, RSRQ, SINR, CQI…).
2. **Détecter des comportements anormaux** — par exemple des attaques (DoS, scans de ports) dans le trafic d'un réseau 5G.

Les deux tâches sont traitées à part égale, et les différentes approches sont **comparées** selon la précision, la capacité de détection, le temps d'apprentissage et les ressources nécessaires.

## Données

| Tâche | Jeu de données | Contenu | Source |
|---|---|---|---|
| Prédiction de KPI | 5G dataset — Raca et al. (MMSys 2020) | Mesures réelles d'un opérateur 5G irlandais : KPI radio, contexte, cellule, débit | [University College Cork](https://cora.ucc.ie/handle/10468/9826) |
| Détection d'anomalies | 5G-NIDD (Université d'Oulu) | Trafic réseau 5G étiqueté : normal, DoS, scans de ports | [IEEE DataPort](https://ieee-dataport.org/node/10203) |

Les données ne sont **pas** incluses dans ce dépôt (taille et licences). Les instructions de téléchargement se trouvent dans [`docs/`](docs/).

## Méthodes étudiées

| Famille | Modèles |
|---|---|
| Machine Learning classique (références) | Régression linéaire, arbre de décision, Random Forest, XGBoost, Isolation Forest |
| Deep Learning | MLP, LSTM, GRU, Transformer pour séries temporelles |
| Pistes exploratoires | Reinforcement Learning, Transfer Learning, Federated Learning, IA explicable (XAI) |

**Métriques** — prédiction : MAE, RMSE, R² ; détection : précision, rappel, F1-score, matrice de confusion ; coût : temps d'apprentissage et d'inférence.

## Organisation du dépôt

```
data/raw/          données brutes téléchargées (non versionnées)
data/processed/    données nettoyées (non versionnées)
notebooks/         exploration et expériences (Jupyter / Google Colab)
src/               code Python réutilisable
models/            modèles entraînés (non versionnés)
results/figures/   graphiques
results/tables/    tableaux de comparaison
docs/              documentation du projet
```

## Installation

```bash
git clone https://github.com/A-SIDIKI/5g6g-perf-anomaly-ml-dl.git
cd 5g6g-perf-anomaly-ml-dl
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
```

Les modèles de Deep Learning sont entraînés sur **Google Colab**.

## Avancement

- [x] Semaine 1 — Cadrage, mise en place du dépôt, choix des données
- [ ] Semaine 2 — Exploration et préparation des données
- [ ] Semaine 3 — Modèles de Machine Learning classiques
- [ ] Semaine 4 — MLP
- [ ] Semaine 5 — LSTM et GRU
- [ ] Semaine 6 — Transformer, XAI, pistes exploratoires
- [ ] Semaine 7 — Comparaison finale, rapport et soutenance

## Licence

Code sous licence MIT. Les jeux de données restent soumis aux licences de leurs auteurs.