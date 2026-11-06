"""Construction de la cible de prédiction du débit (tâche A)."""


def ajouter_cible(serie, horizon):
    """Ajoute la cible : le débit descendant (Mbit/s) dans `horizon` secondes, dans la même trace.

    Colonnes ajoutées :
    - debit     : débit actuel en Mbit/s ;
    - cible     : débit dans `horizon` secondes (NaN en fin de trace) ;
    - evaluable : True si la ligne peut servir à l'évaluation (cible connue, débit actuel connu,
                  et seconde cible réellement mesurée, pas recopiée au rééchantillonnage).
    """
    df = serie.copy()
    df["debit"] = df["DL_bitrate"] / 1000
    par_trace = df.groupby("trace")
    df["cible"] = par_trace["debit"].shift(-horizon)
    cible_completee = par_trace["seconde_completee"].shift(-horizon, fill_value=True)
    df["evaluable"] = df["cible"].notna() & df["debit"].notna() & ~cible_completee
    return df

# ---------------------------------------------------------------- ARCEP
import numpy as np

# Critère officiel de réussite de chaque protocole (notes méthodologiques de l'ARCEP)
CRITERE_SUCCES = {
    "WEB": "loaded_in_less_10_secondes",   # page chargée en moins de 10 s
    "STREAM": "quality_correct",           # vidéo de qualité correcte
    "ULH": "upload_ok",                    # fichier de 1 Mo envoyé en moins de 30 s
    "SMS": "sms_success",                  # SMS reçu en moins de 10 s
    "VOIX": "crspa",                       # appel maintenu 2 min sans perturbation audible
    "OTT": "crspa",                        # idem, appel par application (WhatsApp...)
}


def ajouter_succes(df):
    """Ajoute la colonne succes (1 = réussi, 0 = échec, NaN = pas de critère officiel).

    DOWNLOAD et UPLOAD n'ont pas de critère de réussite : on y mesure un débit.
    ICMP (ping) n'a pas d'indicateur officiel : on utilise la colonne Result.
    """
    df = df.copy()
    df["succes"] = np.nan
    for protocole, colonne in CRITERE_SUCCES.items():
        if colonne in df.columns:
            ligne = df["protocole"] == protocole
            df.loc[ligne, "succes"] = df.loc[ligne, colonne]
    if "Result" in df.columns:
        ping = df["protocole"] == "ICMP"
        df.loc[ping, "succes"] = (df.loc[ping, "Result"] == "Success").astype(int)
    return df