import cv2
import os
import numpy as np
import random
import requests
from urllib.parse import unquote
from uniface import ArcFace
from uniface import compute_similarity

# Initialize recognizer
recognizer = ArcFace()

# Dataset path
dataset_path = r"F:\VScode_NHD\boxmot\vgg_face_dataset"

# Number of persons to sample
num_persons = 50

# Load dataset for selected persons and extract embeddings
def load_sampled_database(num_persons):
    files_path = os.path.join(dataset_path, 'files')
    all_persons_files = [f for f in os.listdir(files_path) if f.endswith('.txt')]
    print(f"Found {len(all_persons_files)} person files in {files_path}")
    
    if len(all_persons_files) < num_persons:
        num_persons = len(all_persons_files)
    sampled_files = random.sample(all_persons_files, num_persons)
    print(f"Sampled {num_persons} persons: {[f[:-4] for f in sampled_files[:5]]}...")
    
    database = {}
    for person_file in sampled_files:
        person_name = person_file[:-4]  # Remove .txt
        file_path = os.path.join(files_path, person_file)
        embeddings = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Limit to 10 images per person
        for line in lines[:10]:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            url = unquote(parts[1])  # Decode URL
            left, top, right, bottom = map(int, map(float, parts[2:6]))  # Convert to int
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                image_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                
                # Crop face
                face = image[top:bottom, left:right]
                if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                    continue
                
                # Assuming images need alignment, but for simplicity, use as is
                embedding = recognizer.get_embedding(face)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                continue
        
        if embeddings:
            database[person_name] = np.array(embeddings)
        print(f"Person {person_name}: {len(embeddings)} embeddings")
    
    return database

# Experiment: Pick one image from one person, compare to all persons' avg embeddings
def run_similarity_experiment(database):
    if not database:
        return "Database empty"
    
    # Pick a random person with at least 2 embeddings
    eligible_persons = [p for p in database if len(database[p]) > 1]
    if not eligible_persons:
        print("No person with multiple embeddings, using any.")
        eligible_persons = list(database.keys())
    query_person = random.choice(eligible_persons)
    query_embeddings = database[query_person]
    query_image_idx = random.randint(0, len(query_embeddings) - 1)
    query_embedding = query_embeddings[query_image_idx]
    
    print(f"Query: Person '{query_person}', Image {query_image_idx + 1}")
    
    # Compare to all persons' average embeddings
    similarities = {}
    for person_name, person_embeddings in database.items():
        if person_name == query_person:
            # Exclude the query embedding from avg
            other_embeddings = [emb for i, emb in enumerate(person_embeddings) if i != query_image_idx]
            if other_embeddings:
                avg_embedding = np.mean(other_embeddings, axis=0)
            else:
                continue  # Skip if no other embeddings
        else:
            avg_embedding = np.mean(person_embeddings, axis=0)
        similarity = compute_similarity(query_embedding, avg_embedding)
        similarities[person_name] = similarity
    
    # Sort by similarity descending
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 similarities:")
    for i, (person, sim) in enumerate(sorted_similarities[:10]):
        match = " (MATCH)" if person == query_person else ""
        print(f"{i+1}. {person}: {sim:.4f}{match}")
    
    # Stats
    same_person_sim = similarities.get(query_person, 0)
    other_sims = [sim for p, sim in similarities.items() if p != query_person]
    if other_sims:
        avg_other_sim = np.mean(other_sims)
        max_other_sim = max(other_sims)
    else:
        avg_other_sim = 0
        max_other_sim = 0
    
    print(f"\nStats:")
    print(f"Same person similarity: {same_person_sim:.4f}")
    print(f"Avg other persons: {avg_other_sim:.4f}")
    print(f"Max other persons: {max_other_sim:.4f}")
    print(f"Gap (same - avg other): {same_person_sim - avg_other_sim:.4f}")

# Main
if __name__ == "__main__":
    try:
        print("Starting experiment...")
        print(f"Loading {num_persons} random persons from {dataset_path}...")
        database = load_sampled_database(num_persons)
        print(f"Loaded {len(database)} persons with embeddings.")
        
        run_similarity_experiment(database)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()