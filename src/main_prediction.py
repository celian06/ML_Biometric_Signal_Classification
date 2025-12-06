import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import warnings

# Supprimer les UserWarning de scikit-learn pour la division par zéro dans les métriques
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Correction pour get_cmap de Matplotlib ---
import matplotlib.cm as cm
import matplotlib

if hasattr(matplotlib, 'colormaps'):
    get_cmap = matplotlib.colormaps.get_cmap
else:
    get_cmap = cm.get_cmap
# --- Fin correction get_cmap ---

# --- Paramètres globaux ---
RANDOM_STATE = 42
N_SPLITS_CV = 5
N_ITER_RANDOM_SEARCH = 50 # Réduit pour un entraînement plus rapide
TEST_SIZE = 0.2
N_JOBS_FOR_PARALLEL = -1 # Utilise tous les cœurs disponibles pour accélérer RandomizedSearchCV
# --------------------------

# --- Fonctions utilitaires ---
def load_data(file_path):
    """Charge les données à partir d'un fichier Excel."""
    try:
        df = pd.read_excel(file_path)
        print(f"Fichier de donnees '{file_path}' charge avec succes. Nombre de lignes : {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{file_path}' est introuvable. Veuillez verifier le chemin.")
        return None
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement du fichier : {e}")
        return None

def prepare_multioutput_data(df, sex_column='Sexe', age_column='Age', age_range_target_column='Tranche_Age', monk_column='Monk'):
    """
    Prépare les données pour la classification multi-output (Sexe, Tranche d'Âge et Monk).
    """
    print(f"\n--- Preparation des données pour la classification multi-output ({sex_column}, {age_range_target_column} et {monk_column}) ---")
    
    df_cleaned = df.copy()
    print(f"Lignes au debut de prepare_multioutput_data: {len(df_cleaned)}")

    # 1. Création de la colonne Tranche_Age si elle n'existe pas
    if age_column in df_cleaned.columns and age_range_target_column not in df_cleaned.columns:
        print(f"Creation de la colonne '{age_range_target_column}' a partir de '{age_column}'...")
        bins = [0, 17, 24, 31, 38, 45, 52, 59, 66, np.inf]
        labels = ['<18', '18-24', '25-31', '32-38', '39-45', '46-52', '53-59', '60-66', '>66']
        df_cleaned[age_range_target_column] = pd.cut(df_cleaned[age_column], bins=bins, labels=labels, right=True)
    elif age_column not in df_cleaned.columns:
        print(f"AVERTISSEMENT: La colonne '{age_column}' est manquante dans le fichier d'entraînement. Impossible de creer '{age_range_target_column}'.")
        return None, None, None, None, None, None, None

    # 2. Supprimer les lignes où 'Tranche_Age' est NaN (peut arriver si 'Age' était NaN)
    initial_rows = len(df_cleaned)
    df_cleaned.dropna(subset=[age_range_target_column], inplace=True)
    print(f"Lignes après dropna sur '{age_range_target_column}': {len(df_cleaned)} (supprimé {initial_rows - len(df_cleaned)} lignes)")

    if df_cleaned.empty:
        print("AVERTISSEMENT: Le DataFrame est vide apres suppression des NaN dans 'Tranche_Age'. Impossible de continuer la preparation.")
        return None, None, None, None, None, None, None

    # Définir les tranches d'âge que nous voulons inclure dans le modèle (exclure <18 et >66 si non pertinents)
    tranches_a_exclure = ['<18', '>66']
    
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned[~df_cleaned[age_range_target_column].isin(tranches_a_exclure)].copy()
    print(f"Lignes apres exclusion des tranches d'age {tranches_a_exclure}: {len(df_cleaned)} (supprimé {initial_rows - len(df_cleaned)} lignes)")

    if df_cleaned.empty:
        print("AVERTISSEMENT: Le DataFrame est vide après exclusion des tranches d'age. Impossible de continuer la preparation.")
        return None, None, None, None, None, None, None

    # 3. Supprimer les lignes avec des valeurs manquantes pour les cibles ou les features clés
    required_cols_for_dropna = [sex_column, age_range_target_column]
    if monk_column in df_cleaned.columns:
        required_cols_for_dropna.append(monk_column)
    
    cols_to_check_for_nan_in_features = ['Taille', 'Poids']
    for col in cols_to_check_for_nan_in_features:
        if col in df_cleaned.columns:
            required_cols_for_dropna.append(col)
    
    impedance_phase_cols = [col for col in df_cleaned.columns if 'Impedance' in col or 'Phase' in col]
    required_cols_for_dropna.extend(impedance_phase_cols)
    
    actual_cols_for_dropna = [col for col in required_cols_for_dropna if col in df_cleaned.columns]

    initial_rows_after_age_filter = len(df_cleaned)
    df_cleaned.dropna(subset=actual_cols_for_dropna, inplace=True)
    print(f"Lignes après dropna sur les features et cibles principales: {len(df_cleaned)} (supprime {initial_rows_after_age_filter - len(df_cleaned)} lignes)")

    if df_cleaned.empty:
        print("AVERTISSEMENT: Le DataFrame est vide apres dropna sur les features/cibles. Impossible de continuer la preparation.")
        return None, None, None, None, None, None, None

    # 4. Encodage des cibles
    le_sex = LabelEncoder()
    y_sex_encoded = le_sex.fit_transform(df_cleaned[sex_column])
    
    unique_age_ranges_in_data = sorted(df_cleaned[age_range_target_column].astype(str).unique())
    le_age = LabelEncoder()
    le_age.fit(unique_age_ranges_in_data)
    y_age_encoded = le_age.transform(df_cleaned[age_range_target_column])

    le_monk = None
    y_monk_encoded = None
    if monk_column in df_cleaned.columns:
        if df_cleaned[monk_column].dtype == 'object':
            print(f"Encodage de la colonne '{monk_column}' (categorielle)...")
            le_monk = LabelEncoder()
            y_monk_encoded = le_monk.fit_transform(df_cleaned[monk_column])
            print(f"Classes de '{monk_column}' encodees : {list(le_monk.classes_)}")
        else:
            print(f"La colonne '{monk_column}' est deja numérique. Pas d'encodage necessaire.")
            y_monk_encoded = df_cleaned[monk_column].round().astype(int).values
        
        # Vérification du déséquilibre des classes pour Monk si elle est encodée
        if y_monk_encoded is not None:
            monk_counts = pd.Series(y_monk_encoded).value_counts(normalize=True)
            if monk_counts.min() < 0.05: # Si une classe représente moins de 5% des données
                print(f"ATTENTION: La distribution des classes pour '{monk_column}' est deséquilibrée:")
                for idx, val in monk_counts.items():
                    class_label = le_monk.inverse_transform([idx])[0] if le_monk else idx
                    print(f"  Classe '{class_label}': {val:.2%}")
                print("Cela pourrait affecter la précision des prédictions pour les classes minoritaires. 'class_weight=\"balanced\"' est utilisé pour compenser.")

    else:
        print(f"AVERTISSEMENT: La colonne '{monk_column}' est manquante dans les donnees d'entraînement. La prédiction de 'Monk' ne sera pas possible.")

    # Combiner les cibles
    if y_monk_encoded is not None:
        y_encoded = np.column_stack((y_sex_encoded, y_age_encoded, y_monk_encoded))
    else: # Si 'Monk' est absente de l'entraînement, on ne combine que sexe et âge
        y_encoded = np.column_stack((y_sex_encoded, y_age_encoded))

    print(f"Classes de sexe encodees : {list(le_sex.classes_)}")
    print(f"Classes de tranche d'age encodees : {list(le_age.classes_)}")
    print("Distribution du sexe original:")
    print(df_cleaned[sex_column].value_counts())
    print("Distribution de la tranche d'age originale:")
    print(df_cleaned[age_range_target_column].value_counts().sort_index())
    if y_monk_encoded is not None:
        print(f"Distribution de '{monk_column}' originale:")
        print(df_cleaned[monk_column].value_counts().sort_index())


    # 5. Sélection des features (MODIFICATION CLÉ ICI : 'Age' est retiré des features)
    feature_cols = [col for col in df_cleaned.columns if 'Impedance' in col or 'Phase' in col or col in ['Taille', 'Poids']]
    # On s'assure que 'Age' n'est PAS dans les features
    if 'Age' in feature_cols:
        feature_cols.remove('Age')
    
    X = df_cleaned[feature_cols]
    
    print(f"Features utilisees ({len(feature_cols)}) : {feature_cols}")
    print(f"Forme des features (X) : {X.shape}")
    print(f"Forme des cibles combinees (y_encoded) : {y_encoded.shape}")
    
    return X, y_encoded, le_sex, le_age, le_monk, feature_cols


def evaluate_multioutput_model(X, y_encoded, le_sex, le_age, le_monk, feature_columns_list, model_type='RandomForest'):
    """Fonction pour evaluer et tuner un modele MultiOutputClassifier."""
    
    print(f"\n--- Debut de l'evaluation du modele MultiOutputClassifier ({model_type}) ---")

    if len(X) == 0:
        print("ERREUR: Le dataset d'entrainement est vide avant l'evaluation. Impossible de proceder.")
        return None, None, None, None

    # Définition du modèle de base pour MultiOutputClassifier
    if model_type == 'RandomForest':
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
        # Paramètres pour RandomizedSearchCV (plages plus ciblées pour un entraînement plus rapide)
        param_dist = {
            'estimator__estimator__n_estimators': [100, 200, 300], # Réduit le nombre max d'estimateurs
            'estimator__estimator__max_features': ['sqrt', 'log2', 0.7], # Plage plus restreinte
            'estimator__estimator__max_depth': [10, 20, 30, None], # Maintien de l'exploration de profondeurs
            'estimator__estimator__min_samples_split': [2, 5], # Moins d'options
            'estimator__estimator__min_samples_leaf': [1, 2], # Moins d'options
            'estimator__estimator__bootstrap': [True], # Souvent True est préféré
        }
    elif model_type == 'GradientBoosting':
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_dist = {
            'estimator__estimator__n_estimators': [100, 200, 300],
            'estimator__estimator__learning_rate': [0.05, 0.1],
            'estimator__estimator__max_depth': [3, 5, 7],
            'estimator__estimator__subsample': [0.8, 0.9, 1.0],
        }
    else:
        raise ValueError("Type de modele non supporté. Choisissez 'RandomForest' ou 'GradientBoosting'.")

    # Pipeline avec StandardScaler et MultiOutputClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', MultiOutputClassifier(base_model, n_jobs=N_JOBS_FOR_PARALLEL))
    ])

    # S'assurer qu'il y a suffisamment de données pour la stratification
    if len(np.unique(y_encoded[:, 0])) < 2 or \
       (len(np.unique(y_encoded[:, 0])) >= 2 and np.min(np.bincount(y_encoded[:, 0])) < N_SPLITS_CV):
        print("AVERTISSEMENT: Pas assez de classes ou d'echantillons pour une stratification efficace sur le sexe. La stratification sera désactivée.")
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded[:, 0]
        )

    print(f"\n--- Division des donnees pour l'entraînement et le test ---")
    print(f"Taille train : {len(X_train)}, test: {len(X_test)}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("ERREUR: Apres train_test_split, l'un des ensembles (train ou test) est vide. Veuillez revoir la preparation des donnees ou la taille du dataset original.")
        return None, None, None, None

    # Custom scorer pour la première tâche (sexe) pour RandomizedSearchCV.
    def f1_score_sex_scorer(y_true_multi, y_pred_multi):
        return f1_score(y_true_multi[:, 0], y_pred_multi[:, 0], average='weighted', zero_division=0)
    
    # Custom scorer pour 'Monk'
    def f1_score_monk_scorer(y_true_multi, y_pred_multi):
        if y_true_multi.shape[1] > 2: # Si Monk est la troisième cible
            return f1_score(y_true_multi[:, 2], y_pred_multi[:, 2], average='weighted', zero_division=0)
        return 0.0 # Retourne 0.0 si Monk n'est pas une cible pour éviter les erreurs de scoring.

    # Choisis le scorer ici. Par défaut, on garde 'sex', mais tu peux passer à 'monk'
    # si la performance de Monk est ta priorité absolue pour la recherche d'hyperparamètres.
    scorer_for_random_search = make_scorer(f1_score_sex_scorer)
    # Pour optimiser sur Monk :
    # scorer_for_random_search = make_scorer(f1_score_monk_scorer)

    print("Lancement de RandomizedSearchCV pour le modèle MultiOutputClassifier...")
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_RANDOM_SEARCH,
        cv=N_SPLITS_CV,
        scoring=scorer_for_random_search,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS_FOR_PARALLEL,
        verbose=0, # MODIFIÉ ICI : de 2 à 0 pour masquer les logs détaillés CV
        error_score='raise'
    )
    random_search.fit(X_train, y_train_encoded)

    print(f"RandomizedSearchCV terminé en {random_search.cv_results_['mean_fit_time'].sum():.2f} secondes.")
    print("\nMeilleurs hyperparamètres trouvés :")
    print(random_search.best_params_)

    best_model = random_search.best_estimator_
    print(f"\nModele choisi pour l'evaluation finale : {best_model}")

    y_pred_encoded = best_model.predict(X_test)

    # Évaluation pour le Sexe
    print("\n== Resultats de l'evaluation de classification pour le SEXE ==")
    y_test_sex_original_labels = le_sex.inverse_transform(y_test_encoded[:, 0])
    y_pred_sex_original_labels = le_sex.inverse_transform(y_pred_encoded[:, 0])
    print(f"Accuracy (Sexe): {accuracy_score(y_test_sex_original_labels, y_pred_sex_original_labels):.2f}")
    print("Classification Report (Sexe):")
    print(classification_report(y_test_sex_original_labels, y_pred_sex_original_labels, digits=2, zero_division=0))
    print("Confusion Matrix (Sexe):")
    print(confusion_matrix(y_test_sex_original_labels, y_pred_sex_original_labels, labels=le_sex.classes_))

    # Évaluation pour la Tranche d'Âge
    print("\n== Resultats de l'evaluation de classification pour la TRANCHE D'ÂGE ==")
    y_test_age_original_labels = le_age.inverse_transform(y_test_encoded[:, 1])
    y_pred_age_original_labels = le_age.inverse_transform(y_pred_encoded[:, 1])
    print(f"Accuracy (Tranche d'Âge): {accuracy_score(y_test_age_original_labels, y_pred_age_original_labels):.2f}")
    print("Classification Report (Tranche d'Âge):")
    print(classification_report(y_test_age_original_labels, y_pred_age_original_labels, digits=2, zero_division=0))
    print("Confusion Matrix (Tranche d'Âge):")
    print(confusion_matrix(y_test_age_original_labels, y_pred_age_original_labels, labels=le_age.classes_))

    # Évaluation pour Monk (si présente)
    if y_encoded.shape[1] > 2: # S'il y a plus de 2 cibles (sexe, âge, monk)
        print("\n== Resultats de l'évaluation de classification pour MONK ==")
        y_test_monk_encoded = y_test_encoded[:, 2]
        y_pred_monk_encoded = y_pred_encoded[:, 2]

        if le_monk is not None: # Si Monk était catégorielle et encodée
            y_test_monk_original_labels = le_monk.inverse_transform(y_test_monk_encoded)
            y_pred_monk_original_labels = le_monk.inverse_transform(y_pred_monk_encoded)
            monk_classes = le_monk.classes_
        else: # Si Monk était déjà numérique
            y_test_monk_original_labels = y_test_monk_encoded.astype(int)
            y_pred_monk_original_labels = y_pred_monk_encoded.astype(int)
            monk_classes = np.unique(np.concatenate((y_test_monk_original_labels, y_pred_monk_original_labels)))

        print(f"Accuracy (Monk): {accuracy_score(y_test_monk_original_labels, y_pred_monk_original_labels):.2f}")
        print("Classification Report (Monk):")
        print(classification_report(y_test_monk_original_labels, y_pred_monk_original_labels, digits=2, zero_division=0))
        print("Confusion Matrix (Monk):")
        print(confusion_matrix(y_test_monk_original_labels, y_pred_monk_original_labels, labels=monk_classes))

    return best_model, le_sex, le_age, le_monk


# --- Script principal ---
if __name__ == "__main__":
    print("--- Debut du script de classification Multi-Output (Sexe, Tranche d'Age et Monk) ---")

    # Assure-toi que ce fichier contient bien la colonne 'Monk' avec des données.
    file_path = 'data/training_dataset.xlsx'
    df_train = load_data(file_path)

    if df_train is None:
        exit()

    monk_column_exists_in_train = 'Monk' in df_train.columns
    monk_column_name_train = 'Monk' if monk_column_exists_in_train else None

    X_train_multi, y_train_encoded_multi, le_sex, le_age, le_monk, feature_cols_multi = \
        prepare_multioutput_data(df_train, monk_column=monk_column_name_train)
    
    if X_train_multi is None:
        print("Arret du script en raison d'un probleme de preparation des donnees d'entrainement.")
        exit()

    # Le modèle est explicitement défini sur RandomForest ici.
    best_multioutput_model, final_le_sex, final_le_age, final_le_monk = evaluate_multioutput_model(
        X_train_multi, y_train_encoded_multi, le_sex, le_age, le_monk, feature_cols_multi, model_type='RandomForest'
    )

    if best_multioutput_model is None:
        print("Arret du script car le modele n'a pas pu être entraîne.")
        exit()

    print("\n--- Debut de la prediction du Sexe, de la Tranche d'Age et de Monk sur 'Data_1.xlsx' ---")

    new_data_path = 'data/new_patients_input.xlsx'
    new_data_df = load_data(new_data_path)

    if new_data_df is None:
        exit()

    num_samples_to_predict = min(5, len(new_data_df))
    samples_to_predict_df = new_data_df.head(num_samples_to_predict).copy()

    print(f"\nAnalyse des {num_samples_to_predict} premiers echantillons du fichier '{new_data_path}'...")

    missing_features_in_samples = [col for col in feature_cols_multi if col not in samples_to_predict_df.columns]
    
    if missing_features_in_samples:
        print(f"Erreur: Les features suivantes sont manquantes dans les echantillons a predire : {missing_features_in_samples}")
        print("La prediction ne peut pas continuer sans toutes les features requises.")
        exit()

    # Sélection des features pour la prédiction (doit correspondre à celles utilisées à l'entraînement)
    X_samples_to_predict = samples_to_predict_df[feature_cols_multi]

    if X_samples_to_predict.empty:
        print("AVERTISSEMENT: Aucun échantillon à prédire après sélection. Le DataFrame est vide.")
    else:
        predicted_encoded_labels = best_multioutput_model.predict(X_samples_to_predict)

        predicted_sex_labels = final_le_sex.inverse_transform(predicted_encoded_labels[:, 0])
        predicted_age_labels = final_le_age.inverse_transform(predicted_encoded_labels[:, 1])
        
        predicted_monk_labels = None
        # Vérifie si le modèle a été entraîné pour prédire 'Monk' (i.e., si y_encoded_multi avait 3 colonnes)
        if predicted_encoded_labels.shape[1] > 2:
            if final_le_monk is not None: # Si Monk était catégorielle et encodée
                predicted_monk_labels = final_le_monk.inverse_transform(predicted_encoded_labels[:, 2])
            else: # Si Monk était numérique
                predicted_monk_labels = predicted_encoded_labels[:, 2].astype(int)

        samples_to_predict_df['Sexe_prédit'] = predicted_sex_labels
        samples_to_predict_df['Tranche_Age_prédite'] = predicted_age_labels
        if predicted_monk_labels is not None:
            samples_to_predict_df['Monk_prédit'] = predicted_monk_labels
            print("Colonne 'Monk_prédit' ajoutée aux prédictions.")
        else:
            print("La colonne 'Monk_prédit' n'a pas été ajoutée, car le modèle n'a pas été entraîné pour prédire 'Monk'.")


        print("\n== Prédictions pour les échantillons sélectionnés de 'Data_1.xlsx' ==")
        
        cols_to_display_final = ['Sexe_prédit', 'Tranche_Age_prédite']
        if 'Monk_prédit' in samples_to_predict_df.columns:
            cols_to_display_final.append('Monk_prédit')
        
        # Afficher uniquement dans le terminal
        print(samples_to_predict_df[cols_to_display_final].to_string())

    print("\n--- Fin du script de classification ---")