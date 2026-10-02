import random
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parents[1]
TEST_GOOD_DIR = BASE / "data" / "mvtec" / "bottle" / "test" / "good"
BAD_CAPTURE_DIR = BASE / "demo" / "bad_capture"


def load_images(folder: Path):
    return sorted(path for path in folder.iterdir() if path.is_file())


if __name__ == "__main__":
    BAD_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in (BAD_CAPTURE_DIR / "dark", BAD_CAPTURE_DIR / "bright", BAD_CAPTURE_DIR / "blur"):
        subdir.mkdir(parents=True, exist_ok=True)
        for child in subdir.iterdir():
            if child.is_file():
                child.unlink()

    files = load_images(TEST_GOOD_DIR)
    rng = random.Random(42)
    rng.shuffle(files)
    selected = files[:9]

    for index, path in enumerate(selected[:3]):
        img = cv2.imread(str(path))
        dark = cv2.convertScaleAbs(img, alpha=0.25, beta=0)
        cv2.imwrite(str(BAD_CAPTURE_DIR / "dark" / f"dark_{index:02d}.png"), dark)

    for index, path in enumerate(selected[3:6]):
        img = cv2.imread(str(path))
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=140)
        cv2.imwrite(str(BAD_CAPTURE_DIR / "bright" / f"bright_{index:02d}.png"), bright)

    for index, path in enumerate(selected[6:9]):
        img = cv2.imread(str(path))
        blurred = cv2.GaussianBlur(img, (51, 51), 0)
        cv2.imwrite(str(BAD_CAPTURE_DIR / "blur" / f"blur_{index:02d}.png"), blurred)

    print(f"dark={len(list((BAD_CAPTURE_DIR / 'dark').iterdir()))}")
    print(f"bright={len(list((BAD_CAPTURE_DIR / 'bright').iterdir()))}")
    print(f"blur={len(list((BAD_CAPTURE_DIR / 'blur').iterdir()))}")
    print(f"bad_capture_dir={BAD_CAPTURE_DIR}")
