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

   # ---------------------------------------------------------------- ARCEP
DOSSIER_ARCEP = RACINE_DEPOT / "data" / "raw" / "arcep"
COLONNES_VIRGULE = ["acess_duration", "bitrate_dl", "bitrate_ul"]
COLONNES_CASSE = ["situation", "zone"]
FICHIERS_ARCEP = ["data_habitations", "data_transports", "voix_habitations", "voix_transports"]


def fichier_arcep(nom="data_habitations", annee=2025, territoire="Metropole"):
    """Chemin d'un fichier de mesures ARCEP, par exemple 2025_QoS_Metropole_voix_transports.csv."""
    return DOSSIER_ARCEP / f"{annee}_QoS_{territoire}_{nom}.csv"


def charger_arcep(fichier=None):
    """Charge un fichier de mesures QoS de l'ARCEP (campagne « Mon réseau mobile »).

    - séparateur ; et encodage cp1252 (le fichier n'est pas en UTF-8) ;
    - suppression des colonnes entièrement vides ;
    - conversion des nombres écrits avec une virgule décimale (si la colonne existe et est du texte) ;
    - casse uniformisée pour situation et zone ;
    - ajout de la colonne instant (date et heure du début du test).
    """
    if fichier is None:
        fichier = fichier_arcep()
    df = pd.read_csv(fichier, sep=";", encoding="cp1252", low_memory=False)
    df = df.dropna(axis=1, how="all")

    for col in COLONNES_VIRGULE:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")

    for col in COLONNES_CASSE:
        if col in df.columns:
            df[col] = df[col].str.capitalize()

    df["instant"] = pd.to_datetime(df["date_start"] + " " + df["hour_start"],
                                   format="%d/%m/%Y %H:%M:%S")
    return df

TERMINAUX = {
    "SM-G990B": "Galaxy S21 FE", "SM-G991B": "Galaxy S21",
    "Samsung Galaxy S22": "Galaxy S22", "Samsung Galaxy Zflip 4": "Galaxy Z Flip4",
    "Samsung Galaxy S23": "Galaxy S23", "SM-S911B": "Galaxy S23",
    "SM-F731B": "Galaxy Z Flip5", "SM-S921B": "Galaxy S24",
}
PROTOCOLES_COMMUNS = ["WEB", "STREAM", "DOWNLOAD"]


def harmoniser_arcep(tables):
    """Réunit plusieurs années (dictionnaire {année: tableau}) en un seul tableau comparable."""
    communes = sorted(set.intersection(*(set(df.columns) for df in tables.values())))
    morceaux = [df[communes].assign(annee=annee) for annee, df in tables.items()]
    tout = pd.concat(morceaux, ignore_index=True)

    tout = tout[tout["protocole"].isin(PROTOCOLES_COMMUNS)]
    dl_sans_debit = (tout["protocole"] == "DOWNLOAD") & tout["bitrate_dl"].isna()
    tout = tout[~dl_sans_debit].copy()

    tout["operator"] = tout["operator"].replace({"FREE": "Free"})
    tout["terminal"] = tout["terminal"].replace(TERMINAUX)
    tout["heure"] = tout["instant"].dt.hour
    tout["rsrp"] = tout["rsrp"].where(tout["rsrp"] < 0)    # 0 dBm = valeur de remplissage
    tout["rsrp_absent"] = tout["rsrp"].isna()               # l'absence de mesure est une information
    return tout.reset_index(drop=True)