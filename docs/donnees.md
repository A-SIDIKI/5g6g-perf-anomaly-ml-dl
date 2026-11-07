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

Source : https://data.arcep.fr/mobile/mesures_qualite_arcep/index.html (jeu « Mon réseau mobile », également sur data.gouv.fr). Fichiers à placer dans `data/raw/arcep/`.

Adresse d'un fichier : `https://data.arcep.fr/mobile/mesures_qualite_arcep/ANNEE/Metropole/NOM.csv`

| Année | Fichiers utilisés |
|---|---|
| 2025 | `2025_QoS_Metropole_data_habitations.csv`, `..._data_transports.csv`, `..._voix_habitations.csv`, `..._voix_transports.csv`, `2025_QoS_indicateurs_globaux.xlsx` |
| 2022, 2023, 2024 | `ANNEE_QoS_Metropole_data_habitations.csv` |

Exemple :

```powershell
curl.exe --ssl-no-revoke -L -o data\raw\arcep\2024_QoS_Metropole_data_habitations.csv https://data.arcep.fr/mobile/mesures_qualite_arcep/2024/Metropole/2024_QoS_Metropole_data_habitations.csv
```

Particularités : séparateur `;`, encodage **cp1252** (et non UTF-8 comme l'indique la documentation), virgule décimale dans certaines colonnes. Le chargement est géré par `src/donnees.py::charger_arcep()`.

## 3. Tableau harmonisé 2022-2025 (fichier dérivé)

`data/processed/arcep_2022_2025.parquet` n'est pas versionné : il est regénéré en exécutant `notebooks/06_harmonisation_annees.ipynb` (fonction `harmoniser_arcep()`).