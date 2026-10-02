"""Central configuration for the local VisionQC demo."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"
CATEGORY = "bottle"  # MVTec AD category used by fit and evaluation.
BACKBONE = "wide_resnet50_2"  # Set to "resnet18" for a smaller CPU model.
INPUT_SIZE = 224  # Crop size after resizing the short edge to 256.
MAX_BANK_PATCHES = 20_000  # Upper bound for nearest-neighbour reference patches.
DISTANCE_CHUNK_SIZE = 128  # Query patches processed together during torch.cdist.
FIT_MAX_IMAGES = 30  # Maximum normal examples used for fit and evaluation.
HISTORY_MAX_LIMIT = 500  # Largest page of inspection history returned by the API.
DASHBOARD_TREND_POINTS = 100  # Recent score points shown in the dashboard chart.
ANALYTICS_SCAN_LIMIT = 5_000  # Recent rows scanned by hotspot and drift analytics.
ALERTS_LIMIT = 50  # Recent alerts returned to the dashboard.
CHAT_FAIL_LIMIT = 10  # Recent failed inspections passed to the chatbot context.
CHAT_ALERT_LIMIT = 10  # Recent alerts passed to the chatbot context.
GAUSSIAN_SIGMA = 4.0  # Blur radius for the upsampled anomaly map.
DEFAULT_THRESHOLD = 1.0  # Replaced by calibration's worst-good score plus margin.
SPREAD_THRESHOLD = 0.15  # Low anomaly-map value used by the diffuse-capture check.
SPREAD_FRACTION = 0.30  # More anomalous area than this requests a new capture.
DARK_PIXEL_LIMIT = 0.25  # Fraction of near-black pixels that rejects a capture.
BRIGHT_PIXEL_LIMIT = 0.25  # Fraction of near-white pixels that rejects a capture.
CAPTURE_PIXEL_FRACTION_MARGIN = 0.05  # Allowed pixel-fraction headroom above calibrated good photos.
CAPTURE_DARK_FALLBACK = 35.0  # Minimum accepted mean brightness before calibration.
CAPTURE_BRIGHT_FALLBACK = 225.0  # Maximum accepted mean brightness before calibration.
BLUR_FALLBACK = 20.0  # Laplacian variance floor before good-photo calibration.
EMPTY_DIFFERENCE_THRESHOLD = 2.0  # Mean grayscale difference from reference/empty scene.
ALIGN_ENABLED = False  # Enable for a fixed camera/reference rig; pre-framed demo clips need no ORB warp.
ALIGN_MIN_INLIERS = 8  # Minimum RANSAC inliers needed to accept alignment.
ALIGN_MAX_SHIFT_FRACTION = 0.35  # Reject homographies moving the image center too far.
ALIGN_MAX_ROTATION_DEGREES = 25.0  # Reject extreme in-plane rotations.
ORB_MAX_FEATURES = 1_200  # Maximum ORB keypoints per aligned image.
ORB_RATIO_TEST = 0.75  # Lowe ratio threshold for binary descriptor matches.
ORB_RANSAC_REPROJECTION = 4.0  # Homography inlier reprojection tolerance in pixels.
REPEAT_WINDOW = 25  # Recent failed inspections examined for repeated locations.
REPEAT_COUNT = 3  # Number of failures in one cell needed to raise an alert.
GRID_SIZE = 10  # Grid size shared by hotspot analytics and the UI.
DRIFT_WINDOW = 20  # Scores included in the moving average drift check.
DRIFT_PERCENT = 20.0  # Warning threshold above the calibration mean.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"  # OpenAI-compatible OpenRouter endpoint.
OPENROUTER_MODEL = "openai/gpt-4o-mini"  # Override with any model enabled for your OpenRouter account.
CHAT_MAX_TOKENS = 500  # Maximum length of a chat response.
INSPECT_INTERVAL_SECONDS = 1.0 / 3.0  # Frontend target cadence: three frames per second.
