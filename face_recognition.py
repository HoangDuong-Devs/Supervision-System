import cv2
import os
import numpy as np
from uniface import RetinaFace, ArcFace
from uniface import compute_similarity

# Initialize models
detector = RetinaFace()
recognizer = ArcFace()

# Dataset path
dataset_path = "faces_dataset"

# Load dataset and extract embeddings
def load_face_database():
    database = {}
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            embeddings = []
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_path, image_file)
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Since images are pre-cropped and aligned, use get_embedding directly
                        embedding = recognizer.get_embedding(image)
                        embeddings.append(embedding)
            if embeddings:
                database[person_name] = np.array(embeddings)  # Store as array for averaging
    return database

# Recognize face from query image
def recognize_face(query_image_path, database, conf_thresh=0.8, sim_thresh=0.6):
    query_image = cv2.imread(query_image_path)
    if query_image is None:
        return "Failed to load query image"
    
    # Detect faces in query image
    faces = detector.detect(query_image)
    if not faces:
        return "No faces detected in query image"
    
    # Take the first face (assuming single face)
    face = faces[0]
    if face['confidence'] < conf_thresh:
        return f"Face confidence too low: {face['confidence']:.2f}"
    
    # Extract embedding from query
    query_embedding = recognizer.get_normalized_embedding(query_image, face['landmarks'])
    
    # Compare with database
    best_match = None
    best_similarity = -1
    results = []
    for person_name, person_embeddings in database.items():
        # Average embeddings for the person
        avg_embedding = np.mean(person_embeddings, axis=0)
        similarity = compute_similarity(query_embedding, avg_embedding)
        results.append((person_name, similarity))
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = person_name
    
    # Check if best similarity is above threshold
    if best_similarity < sim_thresh:
        return f"No good match found. Best: {best_match}, Similarity: {best_similarity:.4f} (below {sim_thresh}). All: {results}"
    
    return f"Best match: {best_match}, Similarity: {best_similarity:.4f}"
    
    return f"Best match: {best_match}, Similarity: {best_similarity:.4f}"

# Main
if __name__ == "__main__":
    # Load database
    print("Loading face database...")
    database = load_face_database()
    print(f"Loaded {len(database)} persons: {list(database.keys())}")
    
    # Query image path (replace with your image)
    query_path = r"F:\VScode_NHD\boxmot\duong_selfie.jpg"
    
    # Recognize
    result = recognize_face(query_path, database, sim_thresh=0.3)
    print(result)