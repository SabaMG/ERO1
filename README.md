# Projet ERO1 — Optimisation du déneigement à Montréal

## Bibliothèques utilisées

Le programme s’appuie sur les bibliothèques Python suivantes :

- [`osmnx`](https://github.com/gboeing/osmnx) : pour l’extraction et la manipulation des graphes routiers depuis OpenStreetMap.
- [`networkx`](https://networkx.org/) : pour les opérations de base sur les graphes.
- [`rustworkx`](https://qiskit.org/documentation/rustworkx/) : pour certaines optimisations algorithmiques.
- [`numpy`](https://numpy.org/) : pour les calculs numériques.
- [`matplotlib`](https://matplotlib.org/) : pour la visualisation des graphes.

Toutes les dépendances peuvent être installées avec :

```bash
pip install osmnx networkx rustworkx numpy matplotlib
```
## Fichiers principaux

- `drone.py` : génère un plan de vol de reconnaissance du drone.
- `snowplough1.py` : génère un plan de déneigement en fonction du budget et du nombre de machines.
- `snowplough2.py` : génère un plan de déneigement en fonction du temps de travail.

## Lancement des scripts

### 1. Génération des graphes et simulation du vol de drone
```bash
python3 drone.py
```
### 2. Simulation du déneigement
```bash
python3 snowplough1.py <budget> [<max_machines>]
```
```bash
python3 snowplough2.py <max_hour>
```
Par exemple :

```bash
python3 snowplough2.py 3200 10
```
Ce qui produit le meme resulats que dans le compte rendu

## Structure du projet

```
.
├── drone.py
├── snowplough1.py
├── snowplough2.py
├── plot_solutions1/              # images pour snowplough1.py
├── plot_solutions2/              # images pour snowplough2.py
├── montreal_directed.graphml
├── montreal_undirected.graphml
├── ERO1_CR.pdf
├── AUTHORS
└── README.md
```