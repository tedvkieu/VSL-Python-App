# Configuration file for VSL Detection System
# Performance optimization settings

# Frame processing settings
FRAME_RATE = 30  # FPS for frame processing
FRAME_WIDTH = 160  # Target frame width for processing
FRAME_HEIGHT = 120  # Target frame height for processing
JPEG_QUALITY = 0.5  # JPEG compression quality (0.1-1.0)
FRAME_SKIP_THRESHOLD = 2  # Process every Nth frame

# Batch processing settings
BATCH_SIZE = 3  # Number of frames to process in batch
BATCH_TIMEOUT_MS = 50  # Timeout for batch processing (ms)

# Model settings
MIN_CONFIDENCE = 0.9  # Minimum confidence for prediction
MISSING_THRESHOLD = 10  # Frames to wait before prediction
MAX_SEQUENCE_LENGTH = 50  # Maximum sequence length

# MediaPipe settings
MEDIAPIPE_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_MAX_HANDS = 2

# Server settings
DEBUG_MODE = True
HOST = '0.0.0.0'
PORT = 5000
THREAD_POOL_WORKERS = 4

# Performance optimization flags
ENABLE_HARDWARE_ACCELERATION = True
ENABLE_LANDMARKS_CACHE = True
ENABLE_FRAME_SKIPPING = True
ENABLE_BATCH_PROCESSING = True

# UI settings
UI_ANIMATION_DURATION = 0.2  # seconds
UI_PULSE_ANIMATION_DURATION = 1.5  # seconds
ENABLE_SMOOTH_SCROLLING = True

# Data saving settings
SAVE_PREDICTIONS_TO_CSV = True
CSV_DATA_DIR = 'data' 