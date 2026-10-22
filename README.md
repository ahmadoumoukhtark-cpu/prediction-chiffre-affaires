# 📊 Prédiction du chiffre d’affaires annuel des entreprises

Projet pédagogique de Data Science réalisé par **Ahmadou Moukhtar Kane**, dans le cadre de mon apprentissage de la régression supervisée.

## 🎯 Problématique

Peut-on estimer le chiffre d’affaires annuel d’une entreprise à partir de ses caractéristiques ? La régularisation améliore-t-elle les prédictions par rapport à une régression linéaire classique ?

Ce projet compare quatre approches : **régression linéaire, Ridge, Lasso et SGDRegressor**.

## 📁 Données

Le jeu de données décrit dans le notebook contient **1 200 observations et 13 colonnes**.

**Variable cible :** `chiffre_affaires_annuel`.

Les variables explicatives comprennent notamment :

- La taille de l’équipe.
- Les années d’expérience.
- Les dépenses publicitaires mensuelles.
- Les visites du site web.
- La note des clients.
- La région.
- Le type d’entreprise.
- La formule d’abonnement.

La source du dataset, son caractère réel ou synthétique, la devise et la définition de certaines variables restent à préciser. Ces informations limitent l’interprétation métier des résultats.

## 🛠️ Technologies utilisées

- **Python**
- **Pandas et NumPy** : manipulation des données.
- **Matplotlib et Seaborn** : visualisation.
- **Scikit-learn** : prétraitement, modèles et métriques.
- **SciPy et Statsmodels** : diagnostics statistiques.
- **Jupyter Notebook** : analyse et documentation.

## 🔎 Démarche

### 1. Analyse exploratoire

- Inspection des dimensions et des types de variables.
- Recherche des valeurs manquantes et des doublons.
- Analyse des statistiques descriptives.
- Visualisation des distributions et des valeurs atypiques.
- Étude des relations entre les variables et la cible.

### 2. Prétraitement

- Encodage One-Hot des variables catégorielles.
- Retrait d’une catégorie de référence par groupe pour éviter la redondance avec l’intercept.
- Séparation des données : **80 % pour l’entraînement et 20 % pour le test**.
- Standardisation des variables numériques avec `StandardScaler`, ajusté uniquement sur l’entraînement.

**Limite actuelle :** l’encodage est encore ajusté avant la séparation. Une prochaine version intégrera les transformations dans un pipeline ajusté uniquement sur l’entraînement.

### 3. Sélection des variables

Une sélection exploratoire utilise la corrélation de Pearson avec un seuil de **|r| ≥ 0,3** pour les variables numériques.

La redondance entre variables est ensuite examinée avec le **VIF**.

Ce seuil est un choix pédagogique : une faible corrélation individuelle ne prouve pas qu’une variable soit inutile. La comparaison avec une version conservant toutes les variables reste à réaliser en validation croisée.

### 4. Modélisation

| Approche | Principe | Objectif |
|---|---|---|
| Régression linéaire | Minimisation des erreurs au carré sans pénalité | Construire un modèle de référence |
| Ridge | Régularisation L2 | Réduire l’amplitude des coefficients |
| Lasso | Régularisation L1 | Examiner la possibilité d’annuler certains coefficients |
| SGDRegressor | Optimisation par descente de gradient stochastique | Étudier l’apprentissage itératif d’un modèle linéaire |

Ridge et Lasso utilisent une première valeur de `alpha=1`, sans optimisation des hyperparamètres.

SGDRegressor est un modèle linéaire entraîné par une autre méthode d’optimisation ; il ne représente pas ici une approche non linéaire.

### 5. Évaluation

Quatre métriques sont utilisées :

| Métrique | Interprétation |
|---|---|
| R² | Compare les erreurs du modèle à celles d’une prédiction constante égale à la moyenne des valeurs évaluées |
| MAE | Erreur absolue moyenne, dans l’unité de la cible |
| MSE | Erreur quadratique moyenne, sensible aux grandes erreurs |
| RMSE | Racine de la MSE, dans l’unité de la cible |

Un **R² de 0,77 ne signifie pas que 77 % des prédictions sont exactes**.

Une référence simple prédisant la moyenne de la cible d’entraînement a également été ajoutée au notebook.

### 6. Diagnostics

- **Linéarité** : examen graphique des résidus.
- **Homoscédasticité** : graphiques et test de Breusch-Pagan/Koenker.
- **Normalité des résidus** : test de Shapiro-Wilk et graphique Q-Q.
- **Multicolinéarité** : calcul du VIF.

Ces diagnostics concernent principalement la régression linéaire de référence. La normalité des résidus n’est pas une condition obligatoire pour produire des prédictions utiles.

## 📈 Résultats historiques — à confirmer

Les résultats ci-dessous proviennent de l’exécution initiale. Ils n’ont pas été recalculés après les corrections du notebook et ne constituent donc pas une validation de sa version actuelle.

| Approche | R² test | MAE test | RMSE test |
|---|---:|---:|---:|
| Régression linéaire | 0,768667 | 18 872,34 | 24 090,61 |
| Ridge | 0,768814 | 18 870,93 | 24 082,95 |
| Lasso | 0,768675 | 18 872,48 | 24 090,18 |
| SGDRegressor | 0,768146 | 18 910,26 | 24 117,72 |

Les quatre approches présentent des performances très proches dans cette exécution initiale. Le faible avantage de Ridge ne permet pas d’affirmer une supériorité robuste sur un seul découpage.

La devise et la tolérance métier étant inconnues, il n’est pas possible de conclure que les erreurs sont acceptables pour un usage en entreprise.

## ▶️ Exécuter le projet

### Prérequis

Installer Python et les dépendances :

    pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels jupyterlab

### Étapes

1. Télécharger ou cloner le dépôt.
2. Placer le fichier `dataset.csv` dans le même dossier que le notebook.
3. Lancer JupyterLab :

       jupyter lab

4. Ouvrir `Projet_Chiffre_Affaires_Documenter.ipynb`.
5. Redémarrer le noyau et exécuter toutes les cellules dans l’ordre.

Le fichier CSV est nécessaire pour recalculer les résultats. Le notebook seul ne contient pas les données nécessaires à une exécution complète.

## ⚠️ Limites

- Source et documentation des données à compléter.
- Exploration initiale réalisée sur l’ensemble des données.
- Encodage à déplacer après la séparation entraînement/test.
- Évaluation sur un seul découpage aléatoire.
- Absence de validation croisée et de réglage des hyperparamètres.
- Sélection par corrélation pouvant écarter des variables utiles.
- Résultats historiques à recalculer après les corrections.
- Projet pédagogique, non validé pour une utilisation opérationnelle.

.

## 💡 Enseignements

Ce projet m’a permis de comprendre que la modélisation ne se limite pas à entraîner un algorithme. La qualité des données, les choix de prétraitement, l’analyse des erreurs et la transparence sur les limites sont essentiels pour interpréter les résultats.

## 👤 Auteur

**Ahmadou Moukhtar Kane**  
Étudiant en mastère professionnel Ingénierie IA & Big Data — Université Centrale de Tunis.
