import os
# Force TensorFlow to use Keras 2 compatibility mode. MUST BE SET BEFORE IMPORTING TENSORFLOW/DEEPFACE.
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import cv2
import numpy as np
import time
import sys
from deepface import DeepFace # For emotion analysis

# --- Configuration ---
# Detector Backend: 'retinaface' or 'mtcnn' generally offer a good balance of accuracy.
# 'ssd' or 'opencv' are faster but potentially less accurate face detectors.
# Ensure backend is installed: pip install retina-face OR pip install mtcnn
DEEPFACE_DETECTOR_BACKEND = 'retinaface' # Or 'mtcnn'
ENFORCE_DETECTION = False   # Keep False for real-time robustness
FRAME_SKIP = 5              # << ADJUST FOR PERFORMANCE >> Process emotions every Nth frame (e.g., 1, 5, 10)
                            # Higher value = less lag, less frequent updates

# --- Text and Display Configuration ---
# Main Window (Emotion BBox & ID)
BOX_COLOR = (0, 255, 0)       # Green for DeepFace bounding box
ID_TEXT_COLOR = (0, 255, 0)   # Green for Face ID (from DeepFace)
FONT = cv2.FONT_HERSHEY_SIMPLEX
ID_FONT_SCALE = 0.6
ID_TEXT_THICKNESS = 1
ID_OFFSET_Y = -10

# Detail Window Configuration
DETAIL_WINDOW_NAME = 'Emotion Analysis Details' # Updated Title
DETAIL_WINDOW_WIDTH = 450 # Slightly wider
DETAIL_WINDOW_HEIGHT = 600
DETAIL_BG_COLOR = (20, 20, 20) # Dark Gray Background
DETAIL_HEADER_COLOR = (255, 255, 255) # White
DETAIL_DOMINANT_COLOR = (0, 255, 0)   # Green
DETAIL_PERCENT_COLOR = (200, 200, 200) # Light Gray
DETAIL_FONT_SCALE = 0.5
DETAIL_LINE_HEIGHT = 18
DETAIL_FACE_V_SPACE = 210     # Vertical pixels allocated PER FACE (7 emotions * 18px + headers ≈ 196px)
DETAIL_LEFT_MARGIN = 15
DETAIL_SECTION_SPACE = 8      # Space between header/dominant/percentages

# --- Initialization ---
print("Initializing...")

# Initialize DeepFace (includes model pre-loading attempt)
print(f"Initializing DeepFace with detector backend: {DEEPFACE_DETECTOR_BACKEND}...")
# These are the emotions DeepFace model recognizes
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
print(f"Recognizing Basic Emotions: {', '.join(emotion_labels)}")
print("NOTE: Complex states like 'anxiety', 'stress', 'calm' or 'gaze' cannot be detected by this model.")
try:
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _ = DeepFace.analyze(
        img_path=dummy_frame, actions=['emotion'],
        detector_backend=DEEPFACE_DETECTOR_BACKEND,
        enforce_detection=False, silent=False # Show loading messages
    )
    print("DeepFace models pre-loaded successfully (or were already cached).")
except ValueError as e:
     print(f"\n--- ERROR initializing DeepFace ---")
     print(f"{e}")
     print(f"Ensure the '{DEEPFACE_DETECTOR_BACKEND}' backend is installed (e.g., pip install retina-face or mtcnn)")
     print(f"Check internet connection for model downloads.")
     if "KerasTensor" in str(e):
         print("\n>>> KerasTensor Error Hint: Try setting TF_USE_LEGACY_KERAS=1 (already attempted) or check TF/Keras versions.")
     sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during DeepFace initialization: {e}")
    print("Attempting to continue, but DeepFace might fail later.")

# Initialize Webcam
print("Initializing Webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Set lower resolution for performance gain
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"Webcam resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print("Webcam opened successfully.")

print("\nStarting real-time analysis...")
print(f"Analyzing emotions every {FRAME_SKIP} frame(s).")
print("Press 'q' in the main video window to quit.")

frame_count = 0
last_analysis_time = time.time()  # Tracks when last successful analysis occurred
deepface_results_cache = [] # Cache for DeepFace results

# --- Main Loop ---
while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame. Exiting.")
        break

    frame_height, frame_width, _ = frame.shape
    frame_count += 1
    current_time = time.time()

    # Create the blank canvas for the detail window
    detail_frame = np.full((DETAIL_WINDOW_HEIGHT, DETAIL_WINDOW_WIDTH, 3), DETAIL_BG_COLOR, dtype=np.uint8)

    # --- Run DeepFace Emotion Analysis (Skip frames for performance) ---
    process_this_frame_deepface = (frame_count % FRAME_SKIP == 0)
    if process_this_frame_deepface:
        try:
            # Analyze the original BGR frame
            analyze_start_time = time.time()
            deepface_results_cache = DeepFace.analyze(
                img_path=frame.copy(), # Pass a copy
                actions=['emotion'],
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=ENFORCE_DETECTION,
                silent=True
            )
            analyze_duration = time.time() - analyze_start_time
            # print(f"Analysis duration: {analyze_duration:.3f}s") # Optional performance timing

            # Ensure results format
            if isinstance(deepface_results_cache, dict):
                 deepface_results_cache = [deepface_results_cache]
            elif not isinstance(deepface_results_cache, list):
                 deepface_results_cache = [] # Make sure it's a list

            last_analysis_time = current_time

        except Exception as e:
            if frame_count % 30 == 0:  # Print error every 30 frames to avoid spam
                print(f"DeepFace Analysis Error: {e}", file=sys.stderr)
            deepface_results_cache = [] # Clear cache on error

    # --- Draw DeepFace Results (BBox, ID, Details) using cached results ---
    detail_y_base = DETAIL_LEFT_MARGIN # Start Y coord for first face details

    if deepface_results_cache:
        for idx, result in enumerate(deepface_results_cache):
            if not isinstance(result, dict) or 'region' not in result or 'emotion' not in result or 'dominant_emotion' not in result:
                continue # Skip malformed result

            # Draw DeepFace Bounding Box & ID on Main Frame
            id_text = f"Face #{idx + 1}"  # Assigned here to avoid NameError if try block fails
            try:
                x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
                # Calculate text size to prevent overlap if needed (optional)
                (text_width, text_height), _ = cv2.getTextSize(id_text, FONT, ID_FONT_SCALE, ID_TEXT_THICKNESS + 1)
                id_text_y = y + ID_OFFSET_Y if y + ID_OFFSET_Y > text_height else y + h + text_height + 5 # Adjust pos
                cv2.putText(frame, id_text, (x, id_text_y), FONT, ID_FONT_SCALE, ID_TEXT_COLOR, ID_TEXT_THICKNESS + 1, cv2.LINE_AA)
            except Exception as e:
                continue # Skip drawing if region data is bad

            # --- Populate Detail Window (Structured Layout) ---
            # Check if there's enough vertical space for this face's details
            if detail_y_base + DETAIL_FACE_V_SPACE <= DETAIL_WINDOW_HEIGHT:
                current_detail_y = detail_y_base

                # Draw Face ID Header
                cv2.putText(detail_frame, id_text, (DETAIL_LEFT_MARGIN, current_detail_y), FONT, DETAIL_FONT_SCALE, DETAIL_HEADER_COLOR, ID_TEXT_THICKNESS, cv2.LINE_AA)
                current_detail_y += DETAIL_LINE_HEIGHT + DETAIL_SECTION_SPACE

                # Draw Dominant Emotion
                dominant_emotion = result['dominant_emotion']
                dominant_score = result['emotion'].get(dominant_emotion, 0) # Get score too
                dominant_text = f"Dominant: {dominant_emotion} ({dominant_score:.1f}%)"
                cv2.putText(detail_frame, dominant_text, (DETAIL_LEFT_MARGIN + 10, current_detail_y), FONT, DETAIL_FONT_SCALE, DETAIL_DOMINANT_COLOR, ID_TEXT_THICKNESS, cv2.LINE_AA)
                current_detail_y += DETAIL_LINE_HEIGHT + DETAIL_SECTION_SPACE

                # Draw all basic emotion percentages
                emotion_scores = result['emotion']
                cv2.putText(detail_frame, "Percentages:", (DETAIL_LEFT_MARGIN + 10, current_detail_y), FONT, DETAIL_FONT_SCALE * 0.9, DETAIL_PERCENT_COLOR, ID_TEXT_THICKNESS, cv2.LINE_AA) # Sub-header
                current_detail_y += DETAIL_LINE_HEIGHT
                for emotion in emotion_labels:
                    score = emotion_scores.get(emotion, 0) # Use .get for safety
                    text = f"- {emotion}: {score:.1f}%"
                    cv2.putText(detail_frame, text, (DETAIL_LEFT_MARGIN + 20, current_detail_y), FONT, DETAIL_FONT_SCALE, DETAIL_PERCENT_COLOR, ID_TEXT_THICKNESS, cv2.LINE_AA)
                    current_detail_y += DETAIL_LINE_HEIGHT

                # Update base Y for the next potential face
                detail_y_base += DETAIL_FACE_V_SPACE
                # Draw a separator line (optional)
                if idx < len(deepface_results_cache) - 1 and detail_y_base < DETAIL_WINDOW_HEIGHT:
                     cv2.line(detail_frame, (DETAIL_LEFT_MARGIN // 2, detail_y_base - DETAIL_FACE_V_SPACE // 4),
                              (DETAIL_WINDOW_WIDTH - DETAIL_LEFT_MARGIN // 2, detail_y_base - DETAIL_FACE_V_SPACE // 4),
                              (50, 50, 50), 1) # Faint gray line

            else:
                # Indicate detail window is full if we haven't already
                if detail_y_base < DETAIL_WINDOW_HEIGHT + DETAIL_LINE_HEIGHT: # Check ensures we draw "..." only once near the bottom
                    cv2.putText(detail_frame, "...", (DETAIL_LEFT_MARGIN, detail_y_base), FONT, DETAIL_FONT_SCALE, DETAIL_HEADER_COLOR, ID_TEXT_THICKNESS, cv2.LINE_AA)
                    detail_y_base += DETAIL_WINDOW_HEIGHT # Prevent drawing "..." again

    # 4. Display the resulting frames
    cv2.imshow('Real-time Emotion Analysis (Press Q to Quit)', frame)
    cv2.imshow(DETAIL_WINDOW_NAME, detail_frame)

    # 5. Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'): # Use waitKey(1) for max responsiveness
        print("Exiting...")
        break

# --- Cleanup ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("Resources released. Goodbye!")