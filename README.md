# ML_Biometric_Signal_Classification
Conception d'instrumentation électronique, acquisition de signaux biométriques (83 volontaires) et classification par optimisation d'un modèle Random Forest en Python.


# Biometric Signal Classification & Inference Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange)](https://scikit-learn.org/)
[![SMOTE](https://img.shields.io/badge/Data-Imbalanced%20Learning-green)](https://imbalanced-learn.org/)

Ce projet implémente un système complet de diagnostic intelligent basé sur des signaux biométriques (Impédancemétrie, Phase).
Il entraîne des modèles de Machine Learning sur une base de données clinique augmentée pour prédire le profil (Sexe, Âge, échelle de Monk) de nouveaux patients.

## 📂 Architecture des Données

Le projet distingue clairement les données d'apprentissage des données de production :

* **`data/training_dataset.xlsx` (Training)** : Base de données historique de 73 volontaires contenant les features (153 colonnes) et les labels (Sexe, Âge, Monk/Activité).
* **`data/new_patients_input.xlsx` (Inference)** : Données brutes de nouveaux sujets (3 patients tests) sans labels, utilisées pour tester la capacité de prédiction du système en conditions réelles.

## 🧠 Workflow Technique

### 1. Augmentation de Données (Sequential SMOTE)
Le script `src/data_augmentation.py` traite le déséquilibre des classes dans le dataset d'entraînement.
* **Problème :** Certaines tranches d'âge ou activités étaient sous-représentées.
* **Solution :** Application de **SMOTE en cascade** (sur l'Âge, puis le Sexe, puis l'Activité) pour générer des profils synthétiques cohérents avant l'entraînement.

### 2. Pipeline de Prédiction
Le script `src/main_prediction.py` exécute la chaîne complète :
1.  **Chargement** de la base de connaissance (`training_dataset.xlsx`).
2.  **Entraînement** des modèles (Random Forest) sur les données augmentées.
3.  **Ingestion** du fichier de nouveaux patients (`new_patients_input.xlsx`).
4.  **Prédiction** et affichage des résultats :
    * Est-ce un Homme ou une Femme ?
    * Quelle est sa tranche d'âge ?
    * Quelle est son activité actuelle (Repos, Marche, Course...) ?

## 🚀 Installation et Utilisation

1.  **Cloner le dépôt :**
    ```bash
    git clone [https://github.com/celian-di-giovanni/Biometric-Activity-Classification-ML.git](https://github.com/celian-di-giovanni/Biometric-Activity-Classification-ML.git)
    cd Biometric-Activity-Classification-ML
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install pandas scikit-learn imbalanced-learn openpyxl numpy
    ```

3.  **Lancer une prédiction sur les nouveaux patients :**
    ```bash
    python src/main_prediction.py
    ```
    *Le script utilisera automatiquement `data/new_patients_input.xlsx` comme entrée pour la démonstration.*
