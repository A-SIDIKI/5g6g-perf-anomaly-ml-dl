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