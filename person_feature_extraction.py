import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import math
import random


class FacialFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_angle(self, point1, point2, point3):
        try:
            vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
            vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

            # Normaliser les vecteurs
            vector1_norm = np.linalg.norm(vector1)
            vector2_norm = np.linalg.norm(vector2)

            if vector1_norm == 0 or vector2_norm == 0:
                return 0.0

            vector1_normalized = vector1 / vector1_norm
            vector2_normalized = vector2 / vector2_norm

            dot_product = np.clip(
                np.dot(vector1_normalized, vector2_normalized), -1.0, 1.0
            )

            return math.degrees(math.acos(dot_product))
        except Exception as e:
            print(f"Avertissement: Erreur lors du calcul de l'angle: {e}")
            return 0.0

    def calculate_area(self, points):
        try:
            if len(points) < 3:
                return 0.0

            points = np.array(points)
            x = points[:, 0]
            y = points[:, 1]

            return 0.5 * abs(
                np.sum(x[:-1] * y[1:] + x[-1] * y[0] - x[1:] * y[:-1] - x[0] * y[-1])
            )
        except Exception as e:
            print(f"Avertissement: Erreur lors du calcul de l'aire: {e}")
            return 0.0

    def extract_features(self, image_path):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Avertissement: Impossible de lire l'image: {image_path}")
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                print(f"Avertissement: Aucun visage détecté dans {image_path}")
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            points = [
                (int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks
            ]

            left_eye_center = np.mean([points[33], points[133]], axis=0)
            right_eye_center = np.mean([points[362], points[263]], axis=0)
            normalize_factor = self.calculate_distance(
                left_eye_center, right_eye_center
            )

            if normalize_factor == 0:
                print(f"Avertissement: Facteur de normalisation invalide dans {image_path}")
                return None

            features = {}

            # Basées sur la structure du crâne qui ne change pas avec l'expression
            features["face_width"] = (
                self.calculate_distance(points[454], points[234]) / normalize_factor
            )
            features["face_height"] = (
                self.calculate_distance(points[10], points[152]) / normalize_factor
            )
            features["jaw_width"] = (
                self.calculate_distance(points[132], points[361]) / normalize_factor
            )
            features["cheekbone_width"] = (
                self.calculate_distance(points[123], points[352]) / normalize_factor
            )
            features["face_depth"] = (
                self.calculate_distance(points[10], points[152]) /
                self.calculate_distance(points[454], points[234])
            )
            features["jaw_angle"] = self.calculate_angle(
                points[132], points[172], points[397]
            )

            # Focus sur la structure osseuse autour des yeux, pas l'ouverture de l'œil
            features["eye_socket_width"] = (
                self.calculate_distance(points[33], points[133]) / normalize_factor
            )
            features["eye_socket_height"] = (
                self.calculate_distance(points[27], points[23]) / normalize_factor
            )
            features["eye_spacing"] = (
                self.calculate_distance(points[133], points[362]) / normalize_factor
            )
            features["eyebrow_position"] = (
                self.calculate_distance(points[66], points[27]) / normalize_factor
            )
            features["eye_angle"] = self.calculate_angle(
                points[33], points[27], points[133]
            )

            # Basées sur la structure osseuse et cartilagineuse
            features["nose_length"] = (
                self.calculate_distance(points[6], points[4]) / normalize_factor
            )
            features["nose_width"] = (
                self.calculate_distance(points[219], points[438]) / normalize_factor
            )
            features["nose_bridge_length"] = (
                self.calculate_distance(points[6], points[197]) / normalize_factor
            )
            features["nose_angle"] = self.calculate_angle(
                points[6], points[4], points[197]
            )

            # Ratios entre les caractéristiques faciales stables
            features["upper_face_ratio"] = (
                self.calculate_distance(points[10], points[6]) /
                self.calculate_distance(points[10], points[152])
            )
            features["middle_face_ratio"] = (
                self.calculate_distance(points[6], points[4]) /
                self.calculate_distance(points[10], points[152])
            )
            features["lower_face_ratio"] = (
                self.calculate_distance(points[4], points[152]) /
                self.calculate_distance(points[10], points[152])
            )
            features["face_width_height_ratio"] = (
                self.calculate_distance(points[454], points[234]) /
                self.calculate_distance(points[10], points[152])
            )
            features["eye_nose_ratio"] = (
                self.calculate_distance(points[133], points[362]) /
                self.calculate_distance(points[219], points[438])
            )

            return features

        except Exception as e:
            print(f"Erreur lors du traitement de {image_path}: {e}")
            return None


def process_person_dataset(base_path):
   
    extractor = FacialFeatureExtractor()
    data = []

    base_path = Path(base_path)
    total_images = 0
    processed_images = 0

    for person_dir in base_path.glob("*"):
        if person_dir.is_dir():
            person_name = person_dir.name
            print(f"Traitement de la personne: {person_name}")

            image_paths = list(person_dir.glob("*.[jJ][pP][gG]")) + \
                          list(person_dir.glob("*.[jJ][pP][eE][gG]")) + \
                          list(person_dir.glob("*.[pP][nN][gG]"))

            for img_path in image_paths:
                total_images += 1
                features = extractor.extract_features(img_path)
                if features:
                    features["person_name"] = person_name
                    features["image_path"] = str(img_path)
                    data.append(features)
                    processed_images += 1
                else:
                    print(f"Impossible d'extraire les caractéristiques de {img_path}")

    print(f"\nRésumé du traitement:")
    print(f"Total des images trouvées: {total_images}")
    print(f"Traitées avec succès: {processed_images}")
    print(f"Échecs de traitement: {total_images - processed_images}")

    if not data:
        raise ValueError("Aucune caractéristique n'a pu être extraite des images du jeu de données")

    df = pd.DataFrame(data)
    return df


def visualize_person_examples(base_path, num_examples=1):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    base_path = Path(base_path)
   
    person_dirs = list(base_path.glob("*"))
    person_dirs = person_dirs[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:

        for person_dir in person_dirs:
            if person_dir.is_dir():
                person_name = person_dir.name
                print(f"\nPersonne: {person_name}")

                image_files = list(person_dir.glob("*.[jJ][pP][gG]")) + \
                              list(person_dir.glob("*.[jJ][pP][eE][gG]")) + \
                              list(person_dir.glob("*.[pP][nN][gG]"))

                if not image_files:
                    print(f"Aucune image trouvée pour {person_name}")
                    continue

                examples = random.sample(
                    image_files, min(num_examples, len(image_files))
                )

                for img_path in examples:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        print(f"Impossible de lire l'image: {img_path}")
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(image_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                            )
                    else:
                        print(f"Aucun visage détecté dans {img_path}")
                        continue

                    image = cv2.resize(
                        image, (0, 0), fx=0.5, fy=0.5
                    )  
                    cv2.imshow(f"{person_name} - {img_path.name}", image)
                    cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset_path = "dataset2"

    try:
        visualize_person_examples(dataset_path, num_examples=2)

        df = process_person_dataset(dataset_path)

        output_path = "person_facial_features.csv"
        df.to_csv(output_path, index=False)
        print(f"\nCaractéristiques extraites et sauvegardées dans {output_path}")
        print(f"Total des échantillons dans le CSV: {len(df)}")
        print("\nColonnes de caractéristiques:")
        cols_to_print = list(df.columns[:5]) + ['person_name', 'image_path']
        for col in cols_to_print:
             if col in df.columns:
                print(f"- {col}")
        if len(df.columns) > 7:
             print(f"- ... et {len(df.columns) - 7} autres colonnes de caractéristiques")

    except FileNotFoundError:
        print(f"Erreur: Répertoire du jeu de données non trouvé à '{dataset_path}'. Veuillez vérifier le chemin.")
    except ValueError as ve:
        print(f"Erreur: {ve}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite: {e}") 