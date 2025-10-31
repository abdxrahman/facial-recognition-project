import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
import joblib
import pandas as pd
import math
from pathlib import Path

# Chargement du modèle sauvegardé et des composants de prétraitement
model = xgb.Booster()
model.load_model("person_identification_model.json")
scaler = joblib.load("person_scaler.pkl")
label_encoder = joblib.load("person_label_encoder.pkl")

# initialisation de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_angle(point1, point2, point3):
    try:
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

        vector1_norm = np.linalg.norm(vector1)
        vector2_norm = np.linalg.norm(vector2)

        if vector1_norm == 0 or vector2_norm == 0:
            return 0.0

        vector1_normalized = vector1 / vector1_norm
        vector2_normalized = vector2 / vector2_norm
        dot_product = np.clip(np.dot(vector1_normalized, vector2_normalized), -1.0, 1.0)

        return math.degrees(math.acos(dot_product))
    except Exception as e:
        return 0.0

def calculate_area(points):
    try:
        if len(points) < 3:
            return 0.0

        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]

        return 0.5 * abs(
            np.sum(x[:-1] * y[1:] + x[-1] * y[0] - x[1:] * y[:-1] - x[0] * y[-1])
        )
    except Exception:
        return 0.0

def extract_features(landmarks, image_width, image_height):
    try:
        points = [
            (int(landmark.x * image_width), int(landmark.y * image_height))
            for landmark in landmarks
        ]

        # Facteur de normalisation
        left_eye_center = np.mean([points[33], points[133]], axis=0)
        right_eye_center = np.mean([points[362], points[263]], axis=0)
        normalize_factor = calculate_distance(left_eye_center, right_eye_center)

        if normalize_factor == 0:
            return None

        features = {}

        # 1. Caractéristiques de la structure osseuse
        features["face_width"] = (
            calculate_distance(points[454], points[234]) / normalize_factor
        )
        features["face_height"] = (
            calculate_distance(points[10], points[152]) / normalize_factor
        )
        features["jaw_width"] = (
            calculate_distance(points[132], points[361]) / normalize_factor
        )
        features["cheekbone_width"] = (
            calculate_distance(points[123], points[352]) / normalize_factor
        )
        features["face_depth"] = (
            calculate_distance(points[10], points[152]) /
            calculate_distance(points[454], points[234])
        )
        features["jaw_angle"] = calculate_angle(
            points[132], points[172], points[397]
        )

        # 2. Caractéristiques de la structure des yeux
        features["eye_socket_width"] = (
            calculate_distance(points[33], points[133]) / normalize_factor
        )
        features["eye_socket_height"] = (
            calculate_distance(points[27], points[23]) / normalize_factor
        )
        features["eye_spacing"] = (
            calculate_distance(points[133], points[362]) / normalize_factor
        )
        features["eyebrow_position"] = (
            calculate_distance(points[66], points[27]) / normalize_factor
        )
        features["eye_angle"] = calculate_angle(
            points[33], points[27], points[133]
        )

        # 3. Caractéristiques de la structure du nez
        features["nose_length"] = (
            calculate_distance(points[6], points[4]) / normalize_factor
        )
        features["nose_width"] = (
            calculate_distance(points[219], points[438]) / normalize_factor
        )
        features["nose_bridge_length"] = (
            calculate_distance(points[6], points[197]) / normalize_factor
        )
        features["nose_angle"] = calculate_angle(
            points[6], points[4], points[197]
        )

        # 4. Proportions faciales
        features["upper_face_ratio"] = (
            calculate_distance(points[10], points[6]) /
            calculate_distance(points[10], points[152])
        )
        features["middle_face_ratio"] = (
            calculate_distance(points[6], points[4]) /
            calculate_distance(points[10], points[152])
        )
        features["lower_face_ratio"] = (
            calculate_distance(points[4], points[152]) /
            calculate_distance(points[10], points[152])
        )
        features["face_width_height_ratio"] = (
            calculate_distance(points[454], points[234]) /
            calculate_distance(points[10], points[152])
        )
        features["eye_nose_ratio"] = (
            calculate_distance(points[133], points[362]) /
            calculate_distance(points[219], points[438])
        )

        return features

    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques: {e}")
        return None

def predict_person(features, scaler, model, label_encoder):
    try:
        feature_df = pd.DataFrame([features])
 
        expected_features = [col for col in scaler.feature_names_in_]

        feature_df = feature_df[expected_features]
        features_array = feature_df.values

        # Normalisation des caractéristiques
        features_scaled = scaler.transform(features_array)

        dfeatures = xgb.DMatrix(features_scaled)

        # Faire la prédiction
        probabilities = model.predict(dfeatures)
        predicted_class = np.argmax(probabilities[0])  
        person = label_encoder.inverse_transform([predicted_class])[0]
        
        # Obtenir le score de confiance
        confidence = float(probabilities[0][predicted_class])
        return person, confidence
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return "Inconnu", 0.0

def get_reference_image(person_name):
    """Obtenir une image de référence du jeu de données pour la personne prédite."""
    try:
        df = pd.read_csv('person_facial_features.csv')
        person_images = df[df['person_name'] == person_name]['image_path'].values
        
        if len(person_images) > 0:
            # Prendre la première image comme référence
            ref_image_path = person_images[0]
            ref_image = cv2.imread(ref_image_path)
            if ref_image is not None:
                target_size = (400, 400)
                ref_image = cv2.resize(ref_image, target_size)
                return ref_image
    except Exception as e:
        print(f"Erreur lors du chargement de l'image de référence: {e}")
    return None

def process_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Erreur: Impossible de lire l'image {image_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        print(f"Aucun visage détecté dans {image_path}")
        return None, None

    # Extraire les caractéristiques
    features = extract_features(results.multi_face_landmarks[0].landmark, w, h)
    if not features:
        print(f"Impossible d'extraire les caractéristiques de {image_path}")
        return None, None

    # Faire la prédiction
    prediction, confidence = predict_person(features, scaler, model, label_encoder)

    # Obtenir l'image de référence
    ref_image = get_reference_image(prediction)

    target_size = (400, 400)
    resized_image = cv2.resize(image, target_size)

    annotated_image = resized_image.copy()
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.multi_face_landmarks[0],
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )

    padding = 20 
    header_height = 35 

    final_width = (target_size[0] * 2) + padding
    final_height = target_size[1] + header_height
    final_image = np.zeros((final_height, final_width, 3), dtype=np.uint8)
    final_image.fill(240)

    final_image[header_height:header_height+target_size[1], 0:target_size[0]] = annotated_image

    if ref_image is not None:
        ref_start_x = target_size[0] + padding
        final_image[header_height:header_height+target_size[1], 
                   ref_start_x:ref_start_x+target_size[0]] = ref_image

    text_color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
    cv2.putText(
        final_image,
        f"Prédit: {prediction}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  
        text_color,
        2
    )

    if ref_image is not None:
        cv2.putText(
            final_image,
            "Référence",
            (target_size[0] + padding, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  
            (0, 0, 0),
            2
        )

    return prediction, final_image

def main():
    test_folder = Path("test_images")
    
    test_folder.mkdir(exist_ok=True)
    
    print(f"Recherche d'images de test dans: {test_folder.absolute()}")

    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = set() 
    for ext in image_extensions:
        image_files.update(test_folder.glob(ext))
        image_files.update(test_folder.glob(ext.upper()))

    image_files = sorted(list(image_files))

    if not image_files:
        print(f"Aucune image trouvée dans {test_folder}")
        print("Veuillez ajouter des images au dossier test_images et relancer le script")
        return

    print(f"\nTrouvé {len(image_files)} images à traiter")

    results_folder = Path("test_results")
    results_folder.mkdir(exist_ok=True)

    for img_path in image_files:
        print(f"\nTraitement de {img_path.name}...")
        prediction, annotated_image = process_image(img_path)
        
        if prediction and annotated_image is not None:
            output_path = results_folder / f"result_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_image)
            print(f"Prédiction: {prediction}")
            print(f"Résultat sauvegardé dans: {output_path}")

    print("\nTraitement terminé ! Vérifiez le dossier test_results pour les images annotées.")

if __name__ == "__main__":
    main() 