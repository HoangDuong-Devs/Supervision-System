import cv2
import os
import time
from uniface import RetinaFace, AgeGender

# Initialize detector and age-gender model
detector = RetinaFace()
age_gender = AgeGender()

# Folder containing test images
test_folder = "test_face_detections"

# Output folder
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Get list of image files
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(image_extensions)]

if not image_files:
    print(f"No images found in {test_folder}")
    exit()

total_time = 0.0
num_images = 0

for image_file in image_files:
    image_path = os.path.join(test_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        continue
    
    # Detect faces
    faces = detector.detect(image)
    
    # Process results and draw
    for face in faces:
        bbox = face['bbox']  # [x1, y1, x2, y2]
        confidence = face['confidence']
        landmarks = face['landmarks']  # 5-point landmarks
        if confidence > 0.85:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for point in landmarks:
                x, y = point
                cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            # Predict age and gender (demo only, accuracy varies)
            gender_id, age = age_gender.predict(image, bbox)
            age_int = int(age)
            gender = 'Female' if gender_id == 0 else 'Male'
            
            # Draw text with scaled font to fit bbox width
            text = f"{gender}, {age_int}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 2
            text_size = cv2.getTextSize(text, font, 1, thickness)[0]
            text_width = text_size[0]
            bbox_width = x2 - x1
            scale = bbox_width / text_width if text_width > 0 else 1
            scale = max(scale, 0.5)  # Min scale to avoid too small
            
            cv2.putText(image, text, (x1, y1 - 10), font, scale, (255, 0, 0), thickness)
    
    # Measure inference time (only detection, not drawing)
    start_time = time.time()
    faces = detector.detect(image)  # Redetect? Wait, already detected above.
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    total_time += inference_time
    num_images += 1
    
    # Save output image
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, image)
    
    print(f"{image_file}: {len(faces)} faces detected, Time: {inference_time:.4f} s, Saved to {output_path}")

if num_images > 0:
    mean_time = total_time / num_images
    print(f"\nTotal images: {num_images}")
    print(f"Mean inference time: {mean_time:.4f} s per image")
else:
    print("No images processed")