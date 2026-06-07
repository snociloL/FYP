import os
import cv2
import numpy as np
from mtcnn import MTCNN

# Initialize the MTCNN face detector
detector = MTCNN()

def extract_and_preprocess(video_path, sequence_length=30, img_size=224):
    """
    Extracts 30 evenly spaced frames, detects the face, crops, and resizes.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < sequence_length:
        print(f"Skipping {video_path}: Video too short.")
        return None

    # Calculate exactly which frames to pull to get an even spread of 30 frames
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
    
    processed_frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # OpenCV uses BGR, convert to RGB for MTCNN and ResNet
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = detector.detect_faces(rgb_frame)
        
        if results:
            # Assume the most prominent face is the first result
            bounding_box = results[0]['box']
            x, y, width, height = bounding_box
            
            # Ensure coordinates are within image bounds to prevent crashing
            x, y = max(0, x), max(0, y)
            
            # Crop the face
            cropped_face = rgb_frame[y:y+height, x:x+width]
            
            # Resize to 224x224 as required by ResNet-50
            try:
                resized_face = cv2.resize(cropped_face, (img_size, img_size))
                # Normalize pixel values to 0-1 range
                normalized_face = resized_face / 255.0
                processed_frames.append(normalized_face)
            except Exception as e:
                print(f"Error resizing face: {e}")
                
    cap.release()
    
    # Only return if we successfully got exactly 30 face frames
    if len(processed_frames) == sequence_length:
        return np.array(processed_frames)
    else:
        print(f"Could not extract {sequence_length} valid faces from {video_path}.")
        return None

def process_dataset(raw_videos_dir, output_dir, label_name):
    """
    Loops through a directory of videos, processes them, and saves as .npy files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(raw_videos_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(raw_videos_dir, filename)
            print(f"Processing: {filename}...")
            
            sequence_tensor = extract_and_preprocess(video_path)
            
            if sequence_tensor is not None:
                # Save the 4D tensor (30, 224, 224, 3) as a numpy file
                save_path = os.path.join(output_dir, f"{label_name}_{filename.split('.')[0]}.npy")
                np.save(save_path, sequence_tensor)
                print(f"Saved: {save_path}")

# --- Execution Setup ---
if __name__ == "__main__":
    print("Deepfake Video Validator - Data Preprocessing Pipeline")
    
    # You will need to create these folders and place your FF++ videos inside them!
    # Example:
    process_dataset("data/raw_videos/real", "data/processed_tensors/real", "real")
    process_dataset("data/raw_videos/fake", "data/processed_tensors/fake", "fake")
    print("Ready to process FaceForensics++ videos!")