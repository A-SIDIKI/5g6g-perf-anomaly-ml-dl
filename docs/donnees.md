# Téléchargement des données

Les données ne sont pas versionnées. Placer les fichiers comme indiqué ci-dessous.

## 1. Jeu 5G de Raca et al. (prédiction de KPI)

Source : https://github.com/uccmisl/5Gdataset — licence GPL-3.0.

```powershell
Invoke-WebRequest -Uri https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip -OutFile data\raw\5G-production-dataset.zip
Expand-Archive data\raw\5G-production-dataset.zip -DestinationPath data\raw\raca5g
Remove-Item -Recurse -Force data\raw\raca5g\__MACOSX
```

Résultat attendu : 83 fichiers CSV dans `data/raw/raca5g/5G-production-dataset/`.

## 2. Mesures QoS de l'ARCEP (détection de dysfonctionnements)

Source : https://data.arcep.fr/mobile/mesures_qualite_arcep/index.html (jeu « Mon réseau mobile », également sur data.gouv.fr).

Fichier utilisé : `2025_QoS_Metropole_data_habitations.csv` (93 Mo), à placer dans `data/raw/arcep/`.

Particularités : séparateur `;`, encodage **cp1252** (et non UTF-8 comme l'indique la documentation), virgule décimale dans certaines colonnes. Le chargement est géré par `src/donnees.py::charger_arcep()`.