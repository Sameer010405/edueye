import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import time
import logging
from pathlib import Path
from PIL import Image
import face_recognition  # Still needed for encoding generation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(_BASE_DIR, "known_faces")
ENCODINGS_FILE_PATH = os.path.join(_BASE_DIR, "known_face_encodings.pkl")

# BlazeFace detection parameters
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

# Face encoding parameters  
NUM_JITTERS_FOR_ENCODING = 1
MAX_ENCODINGS_PER_PERSON = 10
MIN_FACE_SIZE = 50
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# Face crop padding (to ensure we get full face for encoding)
FACE_PADDING = 0.2  # 20% padding around detected face

class BlazeFaceEncoder:
    def __init__(self):
        """Initialize BlazeFace detector and face recognition encoder."""
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize BlazeFace detector
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (< 2m), 1 for full-range
            min_detection_confidence=DETECTION_CONFIDENCE
        )
        
        logger.info("BlazeFace encoder initialized successfully")

    def detect_faces_blazeface(self, image):
        """
        Detect faces using BlazeFace (MediaPipe).
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            list: List of face bounding boxes in (x, y, width, height) format
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_image)
        
        face_boxes = []
        if results.detections:
            h, w, _ = image.shape
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding to ensure full face is captured
                padding_x = int(width * FACE_PADDING)
                padding_y = int(height * FACE_PADDING)
                
                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                width = min(w - x, width + 2 * padding_x)
                height = min(h - y, height + 2 * padding_y)
                
                # Check minimum size
                if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                    face_boxes.append((x, y, width, height))
        
        return face_boxes

    def crop_face_for_encoding(self, image, face_box):
        """
        Crop face from image using BlazeFace detection box.
        
        Args:
            image: Original image
            face_box: (x, y, width, height) bounding box
            
        Returns:
            numpy.ndarray: Cropped face image
        """
        x, y, w, h = face_box
        cropped_face = image[y:y+h, x:x+w]
        return cropped_face

    def generate_face_encoding(self, face_image):
        """
        Generate face encoding using face_recognition library.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            numpy.ndarray: Face encoding or None if failed
        """
        try:
            # Convert BGR to RGB for face_recognition
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Generate encoding using face_recognition
            # Note: We don't use face_recognition for detection, only for encoding
            encodings = face_recognition.face_encodings(
                rgb_face, 
                num_jitters=NUM_JITTERS_FOR_ENCODING
            )
            
            if encodings:
                return encodings[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating encoding: {e}")
            return None

    def process_image(self, image_path):
        """
        Process single image: detect faces with BlazeFace, encode with face_recognition.
        
        Args:
            image_path: Path to image file
            
        Returns:
            list: List of face encodings
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            # Resize large images for faster processing
            h, w = image.shape[:2]
            if max(h, w) > 1600:
                scale = 1600 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Detect faces using BlazeFace
            face_boxes = self.detect_faces_blazeface(image)
            
            if not face_boxes:
                return []
            
            # Generate encodings for detected faces
            encodings = []
            for face_box in face_boxes:
                # Crop face
                face_image = self.crop_face_for_encoding(image, face_box)
                
                # Generate encoding
                encoding = self.generate_face_encoding(face_image)
                if encoding is not None:
                    encodings.append(encoding)
                    # Only take first valid encoding per image
                    break
            
            return encodings
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return []

    def encode_known_faces(self):
        """
        Main encoding function using BlazeFace for detection and face_recognition for encoding.
        """
        logger.info("=" * 60)
        logger.info("STARTING BLAZEFACE-BASED FACE ENCODING")
        logger.info("=" * 60)
        logger.info(f"Source directory: {KNOWN_FACES_DIR}")
        logger.info(f"Output file: {ENCODINGS_FILE_PATH}")
        logger.info(f"Detection: BlazeFace (MediaPipe)")
        logger.info(f"Encoding: face_recognition library")
        logger.info(f"Detection confidence: {DETECTION_CONFIDENCE}")
        logger.info(f"Face padding: {FACE_PADDING}")
        logger.info("-" * 60)

        # Validate input directory
        known_faces_path = Path(KNOWN_FACES_DIR)
        if not known_faces_path.exists():
            logger.error(f"Directory not found: {KNOWN_FACES_DIR}")
            return False

        # Initialize storage
        known_face_encodings = []
        known_face_names = []
        person_thumbnails = {}
        all_known_person_names = set()
        
        # Statistics
        stats = {
            'processed_people': 0,
            'successful_people': 0,
            'total_encodings': 0,
            'skipped_empty_folders': 0,
            'skipped_no_faces': 0,
            'failed_images': 0
        }

        start_time = time.time()

        try:
            # Process each person directory
            person_dirs = [d for d in known_faces_path.iterdir() if d.is_dir()]
            
            if not person_dirs:
                logger.warning("No person directories found.")
                return False

            for person_dir in sorted(person_dirs):
                person_name = person_dir.name
                logger.info(f"Processing: '{person_name}'")
                stats['processed_people'] += 1
                all_known_person_names.add(person_name)
                
                # Find image files
                image_files = []
                for ext in SUPPORTED_FORMATS:
                    image_files.extend(person_dir.glob(f"*{ext}"))
                    image_files.extend(person_dir.glob(f"*{ext.upper()}"))
                
                if not image_files:
                    logger.warning(f"  No images found for '{person_name}'")
                    stats['skipped_empty_folders'] += 1
                    continue

                # Process images for this person
                encodings_for_person = 0
                first_valid_image = None
                
                for img_path in sorted(image_files):
                    if encodings_for_person >= MAX_ENCODINGS_PER_PERSON:
                        break
                    
                    # Process image with BlazeFace
                    encodings = self.process_image(img_path)
                    
                    if encodings:
                        for encoding in encodings:
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                            encodings_for_person += 1
                            stats['total_encodings'] += 1
                            
                            if first_valid_image is None:
                                first_valid_image = str(img_path)
                                logger.info(f"  ✓ Found face in '{img_path.name}' (thumbnail)")
                            
                            # Only one encoding per image
                            break
                    else:
                        stats['failed_images'] += 1
                
                # Summary for this person
                if encodings_for_person > 0:
                    person_thumbnails[person_name] = first_valid_image
                    stats['successful_people'] += 1
                    logger.info(f"  ✓ Encoded {encodings_for_person} face(s) for '{person_name}'")
                else:
                    logger.warning(f"  ✗ No valid faces found for '{person_name}'")
                    stats['skipped_no_faces'] += 1

            # Processing complete
            elapsed_time = time.time() - start_time

            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("ENCODING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Processed people: {stats['processed_people']}")
            logger.info(f"Successful people: {stats['successful_people']}")
            logger.info(f"Total encodings: {stats['total_encodings']}")
            logger.info(f"Failed images: {stats['failed_images']}")
            logger.info(f"Processing time: {elapsed_time:.2f} seconds")

            if not known_face_encodings:
                logger.error("No faces encoded successfully!")
                return False

            # Save encodings
            logger.info(f"\nSaving encodings to: {ENCODINGS_FILE_PATH}")
            
            try:
                data_to_save = {
                    "encodings": known_face_encodings,
                    "names": known_face_names,
                    "thumbnails": person_thumbnails,
                    "all_person_names": list(all_known_person_names),
                    "encoding_stats": stats,
                    "detection_model": "BlazeFace",
                    "encoding_model": "face_recognition",
                    "detection_confidence": DETECTION_CONFIDENCE,
                    "face_padding": FACE_PADDING,
                    "encoding_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(ENCODINGS_FILE_PATH, "wb") as f:
                    pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.info("✓ Encodings saved successfully!")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save encodings: {e}")
                return False

        except Exception as e:
            logger.error(f"Fatal error during encoding: {e}")
            return False
        
        finally:
            # Clean up MediaPipe resources
            self.face_detection.close()

def main():
    """Main function."""
    try:
        encoder = BlazeFaceEncoder()
        success = encoder.encode_known_faces()
        
        if success:
            logger.info("BlazeFace encoding completed successfully!")
            return 0
        else:
            logger.error("BlazeFace encoding failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()