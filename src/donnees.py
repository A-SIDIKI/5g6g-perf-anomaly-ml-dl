"""Chargement des jeux de données du projet."""
from pathlib import Path

import pandas as pd
import numpy as np


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

# Génération de réseau correspondant à chaque valeur de NetworkMode
GENERATION = {
    "GPRS": "2G", "EDGE": "2G",
    "UMTS": "3G", "HSDPA": "3G", "HSUPA": "3G", "HSPA+": "3G",
    "LTE": "4G",
    "5G": "5G",
}
COLONNES_RADIO = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI", "NRxRSRP", "NRxRSRQ"]


def nettoyer_raca(brut):
    """Applique les règles de nettoyage décidées lors du diagnostic.

    1. Supprime les colonnes PING* (vides à 98 %).
    2. Convertit l'horodatage texte en vraie date (colonne temps).
    3. Ajoute la colonne generation (2G, 3G, 4G, 5G).
    4. Remplace la valeur bouche-trou RSRP = -200 (et le SNR associé) par NaN.
    5. Vide les colonnes radio en 2G/3G : ce ne sont pas les mêmes grandeurs (RSCP...).
    """
    df = brut.copy()                       # on ne modifie jamais les données d'origine

    df = df.drop(columns=[c for c in df.columns if c.startswith("PING")])
    df["temps"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S")
    df["generation"] = df["NetworkMode"].map(GENERATION)

    sans_mesure = df["RSRP"] == -200
    df.loc[sans_mesure, ["RSRP", "SNR"]] = np.nan

    ancienne_techno = df["generation"].isin(["2G", "3G"])
    df.loc[ancienne_techno, COLONNES_RADIO] = np.nan

    return df


# Rééchantillonnage : colonnes moyennées et colonnes dont on garde la première valeur
MESURES = ["Longitude", "Latitude", "Speed", "RSRP", "RSRQ", "SNR", "CQI", "RSSI",
           "NRxRSRP", "NRxRSRQ", "DL_bitrate", "UL_bitrate"]
INFOS = ["NetworkMode", "generation", "CellID", "State", "application", "mobilite"]


def reechantillonner_raca(propre):
    """Construit une grille régulière d'une ligne par seconde pour chaque trace.

    - plusieurs mesures dans la même seconde : moyenne (MESURES) ou première valeur (INFOS) ;
    - seconde sans mesure : on recopie la mesure précédente, au plus 2 secondes d'affilée ;
    - la colonne seconde_completee indique les lignes recopiées.
    """
    resultats = []
    for nom, trace in propre.groupby("trace"):
        par_seconde = trace.set_index("temps").resample("1s")
        df = pd.concat([par_seconde[MESURES].mean(), par_seconde[INFOS].first()], axis=1)

        manquante = par_seconde.size() == 0       # secondes sans aucune mesure
        recopie = df.ffill(limit=2)               # valeur précédente, 2 fois au plus
        df[manquante] = recopie[manquante]        # on ne remplit QUE les secondes manquantes

        df["seconde_completee"] = manquante
        df["trace"] = nom
        resultats.append(df.reset_index())
    return pd.concat(resultats, ignore_index=True)