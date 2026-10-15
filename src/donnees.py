"""Chargement des jeux de données du projet."""
from pathlib import Path

import pandas as pd

# Racine du dépôt : le dossier parent de src/
RACINE_DEPOT = Path(__file__).resolve().parents[1]
DOSSIER_RACA = RACINE_DEPOT / "data" / "raw" / "raca5g" / "5G-production-dataset"


def charger_raca(dossier=DOSSIER_RACA):
    """Charge les 83 traces du jeu 5G de Raca et al. dans un seul DataFrame.

    Trois colonnes sont ajoutées pour garder l'origine de chaque ligne :
    application (Netflix, Amazon_Prime, Download), mobilite (Static, Driving)
    et trace (nom du fichier).
    """
    morceaux = []
    for fichier in sorted(Path(dossier).rglob("*.csv")):
        d = pd.read_csv(fichier, na_values="-")
        chemin = fichier.relative_to(dossier).parts
        d["application"] = chemin[0]
        d["mobilite"] = chemin[1]
        d["trace"] = fichier.stem
        morceaux.append(d)
    return pd.concat(morceaux, ignore_index=True)