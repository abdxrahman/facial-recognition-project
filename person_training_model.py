import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Chargement du jeu de données
df = pd.read_csv('person_facial_features.csv')

# Affichage des informations sur le jeu de données
print(f"Dimensions du jeu de données: {df.shape}")
print("\nDistribution des classes avant équilibrage:")
print(df['person_name'].value_counts())

X = df.drop(['person_name', 'image_path'], axis=1)
y = df['person_name']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nPersonnes encodées: {list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

# Standardisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Application de SMOTE pour équilibrer les données d'entraînement
print("\nApplication de SMOTE pour équilibrer les données d'entraînement...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("\nDistribution des classes après équilibrage:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for label, count in zip(label_encoder.classes_, counts):
    print(f"{label}: {count}")

dtrain = xgb.DMatrix(X_train_balanced, label=y_train_balanced)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

base_params = {
    'objective': 'multi:softprob',
    'num_class': len(label_encoder.classes_),
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'eval_metric': ['mlogloss', 'merror'],
    'seed': 42,
    'max_bin': 256,
    'grow_policy': 'lossguide',
    'predictor': 'gpu_predictor',

    'max_depth': 8,  # Profondeur augmentée pour des motifs plus complexes
    'min_child_weight': 2,  # Réduit pour permettre des motifs plus spécifiques
    'gamma': 0.2,  # Augmenté pour prévenir le surapprentissage
    'subsample': 0.9,  # Augmenté pour utiliser plus de données par arbre
    'colsample_bytree': 0.9,  # Augmenté pour utiliser plus de caractéristiques par arbre
    'reg_alpha': 0.2,  # Régularisation L1
    'reg_lambda': 1.0,  # Régularisation L2
    'learning_rate': 0.03,  # Réduit pour une meilleure généralisation
    'scale_pos_weight': 1  # Classes équilibrées
}

# Entraînement du modèle initial avec plus d'itérations
print("\nEntraînement du modèle initial...")
num_rounds = 500  # Nombre d'itérations augmenté
watchlist = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(
    base_params,
    dtrain,
    num_rounds,
    watchlist,
    early_stopping_rounds=30,  # Patience augmentée
    verbose_eval=50
)

xgb_clf = xgb.XGBClassifier(
    **base_params,
    n_estimators=500,
    early_stopping_rounds=30,
    verbose=0
)

param_grid = {
    'max_depth': [7, 8, 9],
    'min_child_weight': [1, 2],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_lambda': [0.8, 1.0, 1.2]
}

# Recherche par grille avec plus de validation croisée
print("\nExécution de la recherche par grille...")
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    cv=5, 
    verbose=1,
    n_jobs=1
)

# Ajustement avec ensemble d'évaluation
grid_search.fit(
    X_train_balanced,
    y_train_balanced,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score: {grid_search.best_score_:.4f}")

print("\nEntraînement du modèle final avec les meilleurs paramètres...")
final_params = base_params.copy()
final_params.update(grid_search.best_params_)

final_model = xgb.train(
    final_params,
    dtrain,
    num_boost_round=800,
    evals=watchlist,
    early_stopping_rounds=30,
    verbose_eval=50
)

y_pred_proba = final_model.predict(dtest)
confidence_threshold = 0.4 

# Conversion des probabilités en prédictions de classe
y_pred = np.argmax(y_pred_proba, axis=1)
max_probs = np.max(y_pred_proba, axis=1)

# Calcul de la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision du modèle final: {accuracy:.4f}")

# Rapport de classification
print("\nRapport de Classification:")
print(classification_report(
    y_test, 
    y_pred,
    target_names=label_encoder.classes_
))

# Visualisation de la matrice de confusion
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            square=True)
plt.xlabel("Personne Prédite")
plt.ylabel("Personne Réelle")
plt.title("Matrice de Confusion - Identification des Personnes")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig("person_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))
xgb.plot_importance(final_model, max_num_features=12, importance_type='gain')
plt.title('Importance des Caractéristiques pour l\'Identification des Personnes')
plt.tight_layout()
plt.savefig('person_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

final_model.save_model('person_identification_model.json')
import joblib
joblib.dump(scaler, 'person_scaler.pkl')
joblib.dump(label_encoder, 'person_label_encoder.pkl')

print("\nModèle et composants sauvegardés avec succès.")
print("Fichiers sauvegardés:")
print("- person_identification_model.json: Modèle XGBoost entraîné")
print("- person_scaler.pkl: Normalisateur de caractéristiques")
print("- person_label_encoder.pkl: Encodeur d'étiquettes")
print("- person_confusion_matrix.png: Visualisation de la matrice de confusion")
print("- person_feature_importance.png: Graphique d'importance des caractéristiques") 