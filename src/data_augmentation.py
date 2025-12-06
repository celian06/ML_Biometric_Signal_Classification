import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings

# Supprimer les avertissements de performance de Pandas (liés à l'ajout de colonnes)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def create_age_bin(age):
    """
    Crée des tranches d'âge à partir d'une valeur d'âge numérique.
    Gère également les cas où l'âge est manquant ou non numérique.
    """
    if pd.isna(age) or not isinstance(age, (int, float)):
        return 'Unknown'
    if 20 <= age <= 30: return '20-30'
    elif 30 < age <= 40: return '30-40'
    elif 40 < age <= 50: return '40-50'
    elif 50 < age <= 60: return '50-60'
    elif 60 < age <= 70: return '60-70'
    elif 70 < age <= 80: return '70-80'
    else: return 'Other' # Pour les âges en dehors des tranches définies (e.g., <20 ou >80)

def prepare_features_for_smote(df_input, target_column_name, encoder=None, is_target_encoded=False):
    """
    Prépare les features (X) et la cible (y) pour l'application de SMOTE.
    Gère l'encodage temporaire de la colonne cible si elle est catégorielle,
    convertit toutes les features en numérique et supprime les lignes avec NaNs.

    Args:
        df_input (pd.DataFrame): DataFrame d'entrée.
        target_column_name (str): Nom de la colonne cible pour SMOTE.
        encoder (LabelEncoder, optional): Encodeur à utiliser si la cible est déjà encodée
                                         et que l'on veut conserver le mapping.
        is_target_encoded (bool): True si la colonne cible est déjà encodée numériquement.

    Returns:
        tuple: (pd.DataFrame X_cleaned, pd.Series y_cleaned, LabelEncoder used_encoder)
               X_cleaned sont les features nettoyées, y_cleaned est la cible nettoyée,
               used_encoder est l'encodeur utilisé pour la cible (ou None).
    """
    print(f"\n--- Préparation des features pour SMOTE (lignes initiales: {len(df_input)}) ---")
    print(f"   Colonnes reçues par prepare_features_for_smote: {df_input.columns.tolist()}")

    df_temp = df_input.copy()

    # --- Étape 1: Gérer la colonne cible (y) ---
    y = None
    used_encoder = None
    target_encoded_col_name = target_column_name # Initialisation par défaut

    if target_column_name not in df_temp.columns:
        print(f"   ERREUR: La colonne cible '{target_column_name}' n'est pas présente dans le DataFrame d'entrée.")
        return pd.DataFrame(), pd.Series(), None

    if df_temp[target_column_name].dtype == 'object' and not is_target_encoded:
        # Cible catégorielle à encoder pour SMOTE
        print(f"   Encodage temporaire de la colonne cible '{target_column_name}'...")
        temp_encoder = LabelEncoder()
        target_encoded_col_name = target_column_name + '_encoded'
        df_temp[target_encoded_col_name] = temp_encoder.fit_transform(df_temp[target_column_name])
        y = df_temp[target_encoded_col_name]
        used_encoder = temp_encoder
        # Exclure la colonne cible originale (string) du DataFrame des features (X)
        X = df_temp.drop(columns=[target_column_name, target_encoded_col_name], errors='ignore')
    elif is_target_encoded:
        # Cible déjà encodée numériquement
        print(f"   La colonne cible '{target_column_name}' est déjà encodée numériquement.")
        y = df_temp[target_column_name]
        used_encoder = encoder # Utiliser l'encodeur fourni pour le décodage ultérieur
        # Exclure la colonne cible (encodée) du DataFrame des features (X)
        X = df_temp.drop(columns=[target_column_name], errors='ignore')
    else:
        # Cible déjà numérique (pas d'encodage nécessaire)
        print(f"   La colonne cible '{target_column_name}' est déjà numérique.")
        y = df_temp[target_column_name]
        X = df_temp.drop(columns=[target_column_name], errors='ignore')
        used_encoder = None

    # --- Étape 2: Gérer les features (X) ---
    # Identifier et encoder One-Hot les colonnes catégorielles restantes dans X
    categorical_cols_in_X = X.select_dtypes(include='object').columns.tolist()
    if categorical_cols_in_X:
        print(f"   Encodage One-Hot des colonnes catégorielles dans les features: {categorical_cols_in_X}")
        # Exclure les colonnes qui seraient des cibles ou qui ont déjà été encodées séparément
        cols_to_remove_from_ohe = ['Sexe', 'Tranche_Age'] # Add other target-like columns if they appear as objects here
        categorical_cols_in_X = [col for col in categorical_cols_in_X if col not in cols_to_remove_from_ohe]

        if categorical_cols_in_X: # Appliquer One-Hot seulement s'il reste des colonnes
            X = pd.get_dummies(X, columns=categorical_cols_in_X, drop_first=True)

    # Convertir toutes les features en numérique (avec gestion des erreurs)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        elif not np.issubdtype(X[col].dtype, np.number):
             X[col] = pd.to_numeric(X[col], errors='coerce')

    # --- Étape 3: Suppression des NaNs ---
    # Créer un DataFrame temporaire pour la suppression de NaN qui inclut X et y
    temp_df_for_nan_check = X.copy()
    temp_df_for_nan_check[y.name if y.name else target_encoded_col_name] = y

    print("\n   --- Compte des NaN après conversion numérique (avant suppression) ---")
    nan_counts = temp_df_for_nan_check.isnull().sum()
    print(nan_counts[nan_counts > 0]) # Affiche seulement les colonnes avec des NaNs

    initial_rows = len(temp_df_for_nan_check)
    temp_df_for_nan_check.dropna(inplace=True)
    rows_after_nan_drop = len(temp_df_for_nan_check)
    rows_dropped = initial_rows - rows_after_nan_drop

    print(f"   Lignes avant suppression des NaNs: {initial_rows}")
    print(f"   Lignes après suppression des NaNs: {rows_after_nan_drop}")
    print(f"   {rows_dropped} lignes supprimées en raison de valeurs manquantes.")

    if rows_after_nan_drop == 0:
        print("   ERREUR: Le DataFrame est devenu vide après la suppression des lignes avec valeurs manquantes. Cela signifie qu'aucune donnée valide ne reste pour SMOTE. Veuillez vérifier vos données pour des NaN excessifs ou des entrées non-numeriques dans les colonnes clés.")
        return pd.DataFrame(), pd.Series(), None

    # Séparer X et y nettoyés
    y_cleaned_name = y.name if y.name else target_encoded_col_name
    X_cleaned = temp_df_for_nan_check.drop(columns=[y_cleaned_name])
    y_cleaned = temp_df_for_nan_check[y_cleaned_name]
    y_cleaned.name = target_column_name # Renomme la série pour la cohérence des logs

    print(f"   Forme finale des features (X) pour SMOTE: {X_cleaned.shape}")
    print(f"   Forme finale de la cible (y) pour SMOTE: {y_cleaned.shape}")

    return X_cleaned, y_cleaned, used_encoder


def apply_smote(X, y, target_name_for_logs, k_neighbors_override=None, cols_to_int=[]):
    """
    Applique l'algorithme SMOTE pour suréchantillonner les classes minoritaires.
    Ajuste k_neighbors dynamiquement si nécessaire.
    Convertit les colonnes spécifiées en entiers après suréchantillonnage.

    Args:
        X (pd.DataFrame): Features pour SMOTE.
        y (pd.Series): Cible pour SMOTE (doit être numérique/encodée).
        target_name_for_logs (str): Nom de la colonne cible pour les logs.
        k_neighbors_override (int, optional): Si spécifié, force k_neighbors. Sinon, calculé dynamiquement.
        cols_to_int (list): Liste des noms de colonnes à convertir en entier après SMOTE.

    Returns:
        pd.DataFrame: DataFrame contenant les données (features + cible) après SMOTE.
    """
    print(f"\n--- Application de SMOTE pour équilibrer '{target_name_for_logs}' ---")

    # Afficher la distribution actuelle de la cible
    print(f"   Distribution actuelle pour '{target_name_for_logs}':")
    value_counts = y.value_counts().sort_index()
    print(value_counts)

    # Calculer k_neighbors dynamiquement
    min_samples_in_any_class = value_counts.min()

    k_neighbors_to_use = k_neighbors_override
    if k_neighbors_to_use is None:
        if min_samples_in_any_class < 2:
            print(f"   AVERTISSEMENT: La classe la plus rare pour '{target_name_for_logs}' a seulement {min_samples_in_any_class} échantillon(s).")
            print("   SMOTE nécessite au moins 2 échantillons pour générer des voisins. L'application de SMOTE ne sera pas effectuée pour cette cible.")
            df_resampled = X.copy()
            df_resampled[y.name] = y
            return df_resampled
        else:
            k_neighbors_to_use = max(1, min(5, min_samples_in_any_class - 1))

    print(f"   Stratégie SMOTE pour '{target_name_for_logs}': 'auto'")
    print(f"   k_neighbors utilisé pour '{target_name_for_logs}': {k_neighbors_to_use}")

    # Initialisation et application de SMOTE
    sm = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors_to_use, random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Reconstruire le DataFrame augmenté
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[y.name] = y_resampled # Ajoute la colonne cible encodée/numérique

    # Convertir les colonnes spécifiées en entiers (arrondi le plus proche)
    for col in cols_to_int:
        if col in df_resampled.columns:
            print(f"   Conversion de la colonne '{col}' en entiers après SMOTE...")
            df_resampled[col] = df_resampled[col].round().astype(int)

    print(f"   Distribution pour '{target_name_for_logs}' APRÈS SMOTE (encodée/numérique):")
    print(df_resampled[y.name].value_counts().sort_index())

    return df_resampled

# --- Démarrage du script d'augmentation de dataset avec SMOTE ---
print("================================================================")
print("             Début du script d'augmentation de dataset           ")
print("================================================================")

file_path = 'data/training_dataset.xlsx'
try:
    df = pd.read_excel(file_path)
    print(f"Fichier de données '{file_path}' chargé avec succès. Nombre de lignes : {len(df)}")
except FileNotFoundError:
    print(f"Erreur: Le fichier '{file_path}' n'a pas été trouvé. Veuillez vérifier le chemin.")
    exit()

# --- 1. Pré-traitement initial du DataFrame ---
print("\n--- Étape 1: Pré-traitement initial des données ---")

# Création de la colonne 'Tranche_Age' à partir de 'Age'
print("   Création de la colonne 'Tranche_Age' à partir de 'Age'...")
df['Tranche_Age'] = df['Age'].apply(create_age_bin)
print("   Colonne 'Tranche_Age' créée.")

# Nettoyage et uniformisation de la colonne 'Sexe'
print("   Nettoyage et encodage initial de 'Sexe'...")
df['Sexe'] = df['Sexe'].replace({'F': 'Femme', 'M': 'Homme'})
gender_le = LabelEncoder()
df['Gender_encoded'] = gender_le.fit_transform(df['Sexe'])
print(f"   Encodage de 'Sexe' en 'Gender_encoded': {dict(zip(gender_le.classes_, gender_le.transform(gender_le.classes_)))}")

# Assurez-vous que la colonne 'Monk' est numérique avant de la passer à SMOTE
monk_le = None
if 'Monk' in df.columns:
    if df['Monk'].dtype == 'object':
        print("   Encodage numérique de la colonne 'Monk' (catégorielle) pour SMOTE...")
        monk_le = LabelEncoder()
        df['Monk'] = monk_le.fit_transform(df['Monk'])
        print(f"   Mapping 'Monk': {dict(zip(monk_le.classes_, monk_le.transform(monk_le.classes_)))}")
    else:
        print("   La colonne 'Monk' est déjà numérique.")
        df['Monk'] = pd.to_numeric(df['Monk'], errors='coerce')
        df.dropna(subset=['Monk'], inplace=True) # Supprimer les NaN si coercés
else:
    print("   La colonne 'Monk' n'est pas présente dans le DataFrame initial. Impossible d'équilibrer 'Monk'.")
    exit()

# Filtrer les tranches d'âge pertinentes (pour la cohérence, si ce n'est pas déjà fait)
initial_rows_before_age_filter = len(df)
pertinent_age_bins = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80']
df_filtered_age = df[df['Tranche_Age'].isin(pertinent_age_bins)].copy()
print(f"   Dataset filtré aux tranches d'âge pertinentes ({len(df_filtered_age)} lignes). {initial_rows_before_age_filter - len(df_filtered_age)} lignes exclues.")


# --- 2. Premier SMOTE: Équilibrage des `Tranche_Age` ---
print("\n================================================================")
print("--- ÉTAPE 2: Application de SMOTE pour équilibrer les 'Tranche_Age' ---")
print("================================================================")

# 'Sexe' (string) doit être exclue des features. 'Tranche_Age' est la cible.
# 'Gender_encoded' et 'Monk' sont des features.
cols_to_exclude_from_features_age = ['Sexe']

df_for_age_smote_prep = df_filtered_age.drop(columns=cols_to_exclude_from_features_age, errors='ignore').copy()

X_age, y_age_encoded, age_bin_encoder = prepare_features_for_smote(
    df_for_age_smote_prep, 'Tranche_Age', is_target_encoded=False
)

if X_age.empty:
    print("Processus interrompu: DataFrame vide après préparation pour le SMOTE d'âge.")
    exit()

cols_to_int_after_smote_age = ['Age', 'Poids', 'Monk', 'Gender_encoded'] # Gender_encoded doit rester entier

df_resampled_age_smote = apply_smote(X_age, y_age_encoded, 'Tranche_Age', k_neighbors_override=None,
                                     cols_to_int=cols_to_int_after_smote_age)

df_augmented_age = df_resampled_age_smote.copy()

# Suppression de la colonne 'Tranche_Age' qui a servi de cible pour ce SMOTE
df_augmented_age.drop(columns=['Tranche_Age'], errors='ignore', inplace=True)


print(f"\n--- Résumé après SMOTE pour 'Tranche_Age': {len(df_augmented_age)} lignes ---")
# On ne peut pas afficher la distribution de 'Sexe' ici car 'Sexe' n'a pas encore été réintroduite
# et 'Gender_encoded' est une valeur numérique.
# print("   Nouvelle distribution de 'Sexe' après le premier SMOTE (pas encore équilibrée):")
# if 'Gender_encoded' in df_augmented_age.columns:
#     temp_sexe = gender_le.inverse_transform(df_augmented_age['Gender_encoded'].round().astype(int))
#     print(pd.Series(temp_sexe).value_counts())
# else:
#     print("   'Gender_encoded' non trouvée pour afficher la distribution de 'Sexe'.")

print("   Nouvelle distribution de 'Monk' après le premier SMOTE (pas encore équilibrée):")
if 'Monk' in df_augmented_age.columns:
    print(df_augmented_age['Monk'].value_counts().sort_index())
else:
    print("   'Monk' non trouvée.")


# --- 3. Deuxième SMOTE: Équilibrage du `Sexe` ---
print("\n================================================================")
print("--- ÉTAPE 3: Application de SMOTE pour équilibrer le 'Sexe' ---")
print("================================================================")

# 'Gender_encoded' est la cible. 'Sexe' (chaîne) n'est pas dans ce DataFrame.
# df_augmented_age contient 'Gender_encoded' et les autres features.
cols_to_exclude_from_features_gender = [] # Pas besoin d'exclure 'Sexe' string ici, car elle a été retirée plus tôt.

# Ici, nous utilisons df_augmented_age qui contient Gender_encoded comme une feature.
# La fonction prepare_features_for_smote va retirer Gender_encoded pour en faire la cible.
df_for_gender_smote_prep = df_augmented_age.copy()

X_gender, y_gender_encoded, _ = prepare_features_for_smote(
    df_for_gender_smote_prep, 'Gender_encoded', encoder=gender_le, is_target_encoded=True
)

if X_gender.empty:
    print("Processus interrompu: DataFrame vide après préparation pour le SMOTE de sexe.")
    exit()

cols_to_int_after_smote_gender = ['Age', 'Poids', 'Monk'] # Gender_encoded est la cible, pas une feature à convertir ici

df_resampled_gender_smote = apply_smote(X_gender, y_gender_encoded, 'Gender_encoded', k_neighbors_override=None,
                                        cols_to_int=cols_to_int_after_smote_gender)

df_augmented_gender = df_resampled_gender_smote.copy()

print(f"\n--- Résumé après SMOTE pour 'Sexe': {len(df_augmented_gender)} lignes ---")
# Pas d'affichage de 'Sexe' ici, car elle n'est pas encore décodée.
# print("   Nouvelle distribution de 'Sexe' après le deuxième SMOTE (devrait être équilibrée):")
# if 'Gender_encoded' in df_augmented_gender.columns:
#     temp_sexe = gender_le.inverse_transform(df_augmented_gender['Gender_encoded'].round().astype(int))
#     print(pd.Series(temp_sexe).value_counts())
# else:
#     print("   'Gender_encoded' non trouvée pour afficher la distribution de 'Sexe'.")

print("   Nouvelle distribution de 'Monk' après le deuxième SMOTE (pas encore équilibrée):")
if 'Monk' in df_augmented_gender.columns:
    print(df_augmented_gender['Monk'].value_counts().sort_index())
else:
    print("   'Monk' non trouvée.")


# --- 4. Troisième SMOTE: Équilibrage de `Monk` ---
print("\n================================================================")
print("--- ÉTAPE 4: Application de SMOTE pour équilibrer l'échelle de 'Monk' ---")
print("================================================================")

# 'Monk' est la cible. 'Gender_encoded' doit être une feature ici.
# Donc, pas besoin d'exclure 'Sexe' ou 'Gender_encoded'.
cols_to_exclude_from_features_monk = [] # Pas d'exclusion ici

df_for_monk_smote_prep = df_augmented_gender.copy() # Contient 'Gender_encoded'

X_monk, y_monk_encoded, _ = prepare_features_for_smote(
    df_for_monk_smote_prep, 'Monk', encoder=monk_le, is_target_encoded=True if monk_le else False
)

if X_monk.empty:
    print("Processus interrompu: DataFrame vide après préparation pour le SMOTE de Monk.")
    exit()

cols_to_int_after_smote_monk = ['Age', 'Poids', 'Gender_encoded'] # 'Monk' est la cible, Gender_encoded doit rester entier

df_final_augmented_data = apply_smote(X_monk, y_monk_encoded, 'Monk', k_neighbors_override=None,
                                      cols_to_int=cols_to_int_after_smote_monk)

# --- 5. Finalisation du DataFrame augmenté ---
print("\n================================================================")
print("--- Étape 5: Finalisation du dataset augmenté ---")
print("================================================================")

# C'est ici que nous décodons ENFIN 'Gender_encoded' en 'Sexe'
if 'Gender_encoded' in df_final_augmented_data.columns:
    df_final_augmented_data['Gender_encoded'] = df_final_augmented_data['Gender_encoded'].round().astype(int)
    df_final_augmented_data['Sexe'] = gender_le.inverse_transform(df_final_augmented_data['Gender_encoded'])
    df_final_augmented_data.drop(columns=['Gender_encoded'], errors='ignore', inplace=True)
    print("   Colonne 'Sexe' décodée et 'Gender_encoded' supprimée.")
else:
    print("   AVERTISSEMENT: 'Gender_encoded' non trouvée dans le DataFrame final. 'Sexe' ne sera pas recréée.")


# Décoder la colonne 'Monk' si elle a été encodée
if monk_le:
    # Assurez-vous que la colonne 'Monk' est un entier avant de la décoder
    if 'Monk' in df_final_augmented_data.columns:
        df_final_augmented_data['Monk'] = df_final_augmented_data['Monk'].round().astype(int)
        df_final_augmented_data['Monk'] = monk_le.inverse_transform(df_final_augmented_data['Monk'])
        print("   Colonnes 'Monk' décodées vers leurs valeurs originales.")
    else:
        print("   AVERTISSEMENT: 'Monk' (encodée) non trouvée dans le DataFrame final pour le décodage.")
else:
    # S'assurer que 'Monk' est un entier si elle n'a pas été encodée
    if 'Monk' in df_final_augmented_data.columns:
        df_final_augmented_data['Monk'] = df_final_augmented_data['Monk'].round().astype(int)
        print("   Colonne 'Monk' convertie en entiers.")
    else:
        print("   AVERTISSEMENT: 'Monk' non trouvée dans le DataFrame final pour la conversion en entier.")


# Recalculer 'Tranche_Age' une dernière fois (pour l'affichage uniquement)
# La colonne 'Age' devrait être présente après tous les SMOTE
if 'Age' in df_final_augmented_data.columns:
    df_final_augmented_data['Tranche_Age_temp_display'] = df_final_augmented_data['Age'].apply(create_age_bin)
    print(f"\nDataset final augmenté. Nombre de lignes : {len(df_final_augmented_data)}")
    print("\nDistribution finale de 'Tranche_Age' (pour information, non sauvegardée):")
    print(df_final_augmented_data['Tranche_Age_temp_display'].value_counts())
    df_final_augmented_data.drop(columns=['Tranche_Age_temp_display'], errors='ignore', inplace=True) # Suppression avant sauvegarde
else:
    print("\nAVERTISSEMENT: La colonne 'Age' est manquante. Impossible de calculer 'Tranche_Age' pour l'affichage.")
    print(f"\nDataset final augmenté. Nombre de lignes : {len(df_final_augmented_data)}")


print("\nDistribution finale de 'Sexe':")
if 'Sexe' in df_final_augmented_data.columns: # Vérification supplémentaire ici
    print(df_final_augmented_data['Sexe'].value_counts())
else:
    print("   La colonne 'Sexe' est manquante dans le DataFrame final.")

print("\nDistribution finale de 'Monk' (devrait être équilibrée):")
if 'Monk' in df_final_augmented_data.columns: # Vérification supplémentaire ici
    print(df_final_augmented_data['Monk'].value_counts().sort_index())
else:
    print("   La colonne 'Monk' est manquante dans le DataFrame final.")


# --- Réorganiser les colonnes ---
cols = df_final_augmented_data.columns.tolist()

# Assurez-vous que les colonnes clés sont présentes pour éviter les erreurs
required_cols_order = ['Sexe', 'Monk', 'Age', 'Poids', 'Taille']

new_cols_order = []
for col in required_cols_order:
    if col in cols:
        new_cols_order.append(col)
        cols.remove(col) # Supprime la colonne de la liste 'cols' pour éviter les doublons

new_cols_order.extend(cols) # Ajoute les colonnes restantes

# Filter new_cols_order to only include columns actually present in df_final_augmented_data
final_ordered_cols = [col for col in new_cols_order if col in df_final_augmented_data.columns]
df_final_augmented_data = df_final_augmented_data[final_ordered_cols]


# Sauvegarder le DataFrame augmenté dans un nouveau fichier Excel
output_file_path = 'Database_augmented_final_version.xlsx'
try:
    df_final_augmented_data.to_excel(output_file_path, index=False)
    print(f"\nDataset augmenté enregistré avec succès sous '{output_file_path}'")
except Exception as e:
    print(f"Erreur lors de l'enregistrement du fichier Excel: {e}")

print("\n================================================================")
print("             Script terminé avec succès.                         ")
print("================================================================")