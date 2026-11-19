"""Fenêtres glissantes et entraînement des réseaux séquentiels (PyTorch)."""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def fenetres(df, variables, moy, ect, W=10, h=1, cible="debit"):
    """Découpe chaque trace en fenêtres de W secondes ; la cible est le débit h secondes après.

    Renvoie X (n, W, variables + 1), y (n,), evaluable (n,) et persistance (n,).
    """
    X, y, evaluable, persistance = [], [], [], []
    for _, g in df.groupby("trace"):                 # jamais de fenêtre à cheval sur deux traces
        v = (g[variables] - moy) / ect                # standardisation
        absent = v["RSRP"].isna().to_numpy(dtype="float32")[:, None]
        v = np.hstack([v.fillna(0).to_numpy(dtype="float32"), absent])
        debit = g[cible].to_numpy()
        completee = g["seconde_completee"].to_numpy()
        for t in range(W - 1, len(g) - h):            # t = dernière seconde de la fenêtre
            X.append(v[t - W + 1:t + 1])
            y.append(debit[t + h])
            evaluable.append(not completee[t + h])    # cible vraiment mesurée (pas recopiée)
            persistance.append(debit[t])              # prédiction « rien ne change »
    return (np.array(X, dtype="float32"), np.array(y, dtype="float32"),
            np.array(evaluable), np.array(persistance))


def entrainer_reseau(modele, X_entr, y_entr, X_val, y_val, echelle=100,
                     max_epoques=60, patience=8, taille_lot=256, lr=1e-3):
    """Perte MSE, Adam, arrêt précoce. La cible est divisée par `echelle` pour rester proche de 1.

    Renvoie le modèle (meilleurs poids) et l'historique des pertes à chaque époque.
    """
    lots = DataLoader(TensorDataset(torch.tensor(X_entr), torch.tensor(y_entr).unsqueeze(1) / echelle),
                      batch_size=taille_lot, shuffle=True)
    X_v, y_v = torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1) / echelle
    perte = nn.MSELoss()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=lr)
    meilleure, meilleurs_poids, attente = float("inf"), None, 0
    historique, debut = [], time.time()
    for epoque in range(1, max_epoques + 1):
        modele.train()
        total = 0.0
        for xb, yb in lots:
            optimiseur.zero_grad()
            l = perte(modele(xb), yb)
            l.backward()
            optimiseur.step()
            total += l.item() * len(xb)
        modele.eval()
        with torch.no_grad():
            l_val = perte(modele(X_v), y_v).item()
        historique.append({"epoque": epoque, "perte_entr": total / len(X_entr), "perte_val": l_val})
        if l_val < meilleure:
            meilleure, meilleurs_poids, attente = l_val, copy.deepcopy(modele.state_dict()), 0
        else:
            attente += 1
            if attente >= patience:
                break
    modele.load_state_dict(meilleurs_poids)
    print(f"{epoque} époques, meilleure à l'époque {epoque - attente}, {time.time() - debut:.0f} s")
    return modele, historique


def predire(modele, X, echelle=100):
    """Prédictions du réseau, remises dans l'unité de la cible (Mbit/s)."""
    modele.eval()
    with torch.no_grad():
        return modele(torch.tensor(X)).squeeze(1).numpy() * echelle


# ---------------------------------------------------------------- modèles séquentiels
class ReseauLSTM(nn.Module):
    def __init__(self, n_variables, n_cachees=32):
        super().__init__()
        self.lstm = nn.LSTM(n_variables, n_cachees, batch_first=True)
        self.sortie = nn.Linear(n_cachees, 1)

    def forward(self, x):                        # x : (lot, W, variables)
        etats, _ = self.lstm(x)                  # (lot, W, n_cachees)
        return self.sortie(etats[:, -1, :])      # état de la dernière seconde


class ReseauGRU(nn.Module):
    def __init__(self, n_variables, n_cachees=32):
        super().__init__()
        self.gru = nn.GRU(n_variables, n_cachees, batch_first=True)
        self.sortie = nn.Linear(n_cachees, 1)

    def forward(self, x):
        etats, _ = self.gru(x)
        return self.sortie(etats[:, -1, :])


class ReseauCNNLSTM(nn.Module):
    """Convolution d'abord (motifs locaux), puis LSTM (enchaînement des motifs)."""
    def __init__(self, n_variables, n_filtres=32, n_cachees=32):
        super().__init__()
        self.conv = nn.Conv1d(n_variables, n_filtres, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(n_filtres, n_cachees, batch_first=True)
        self.sortie = nn.Linear(n_cachees, 1)

    def forward(self, x):
        motifs = torch.relu(self.conv(x.permute(0, 2, 1)))   # (lot, filtres, W)
        etats, _ = self.lstm(motifs.permute(0, 2, 1))        # (lot, W, n_cachees)
        return self.sortie(etats[:, -1, :])


class ReseauLSTMCNN(nn.Module):
    """LSTM d'abord (un état par seconde), puis convolution sur ces états et pooling."""
    def __init__(self, n_variables, n_cachees=32, n_filtres=32):
        super().__init__()
        self.lstm = nn.LSTM(n_variables, n_cachees, batch_first=True)
        self.conv = nn.Conv1d(n_cachees, n_filtres, kernel_size=3, padding=1)
        self.sortie = nn.Linear(n_filtres, 1)

    def forward(self, x):
        etats, _ = self.lstm(x)                                 # (lot, W, n_cachees)
        motifs = torch.relu(self.conv(etats.permute(0, 2, 1)))  # (lot, filtres, W)
        return self.sortie(motifs.max(dim=2).values)            # pooling : max sur le temps
