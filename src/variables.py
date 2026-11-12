"""Variables de contexte construites à partir des tests voisins (ARCEP)."""
import numpy as np


def ajouter_contexte(df, delai_max=600):
    """Ajoute le résultat du test précédent (même opérateur, même lieu) et le taux d'échec
    des autres tests de la même minute au même lieu. df doit contenir la colonne `echec`."""
    df = df.copy()
    df["lat"] = df["latitude_start"].round(3)
    df["lon"] = df["longitude_start"].round(3)

    df = df.sort_values(["operator", "lat", "lon", "instant"])
    groupes = df.groupby(["operator", "lat", "lon"])
    df["echec_prec"] = groupes["echec"].shift(1)
    delai = (df["instant"] - groupes["instant"].shift(1)).dt.total_seconds()
    df.loc[delai > delai_max, "echec_prec"] = np.nan
    df["prec_absent"] = df["echec_prec"].isna()
    df = df.sort_index()

    df["minute"] = df["instant"].dt.floor("1min")
    voisins = df.groupby(["minute", "lat", "lon"])["echec"]
    n = voisins.transform("size")
    total = voisins.transform("sum")
    df["taux_echec_autres"] = ((total - df["echec"]) / (n - 1)).where(n > 1)
    return df