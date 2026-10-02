import random
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parents[1]
TEST_DIR = BASE / "data" / "mvtec" / "bottle" / "test"
DEMO_DIR = BASE / "demo"
PREVIEW_DIR = DEMO_DIR / "previews"
FPS = 10
SHOT_SECONDS = 1.5
FRAMES_PER_SHOT = int(round(FPS * SHOT_SECONDS))
SIZE = (640, 640)


def list_files(directory: Path):
    return sorted(path for path in directory.iterdir() if path.is_file())


def load_and_resize(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.shape[:2] != SIZE:
        img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    return img


def make_bad_frame(img, mode: str):
    if mode == "dark":
        return cv2.convertScaleAbs(img, alpha=0.25, beta=0)
    if mode == "bright":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=140)
    if mode == "blur":
        return cv2.GaussianBlur(img, (51, 51), 0)
    return img.copy()


def make_main_sequence():
    rng = random.Random(42)
    good = list_files(TEST_DIR / "good")
    broken_large = list_files(TEST_DIR / "broken_large")
    broken_small = list_files(TEST_DIR / "broken_small")
    contamination = list_files(TEST_DIR / "contamination")

    for image_list in (good, broken_large, broken_small, contamination):
        rng.shuffle(image_list)

    return [
        ("good", good[0]),
        ("good", good[1]),
        ("defect", broken_large[0]),
        ("good", good[2]),
        ("dark", good[3]),
        ("good", good[4]),
        ("defect", contamination[0]),
        ("good", good[5]),
        ("blur", good[6]),
        ("good", good[7]),
        ("defect", broken_small[0]),
        ("bright", good[8]),
        ("good", good[9]),
        ("defect", broken_large[1]),
        ("good", good[10]),
    ]


def make_repeat_sequence(good, broken_small):
    return [
        ("good", good[11]),
        ("defect", broken_small[0]),
        ("good", good[12]),
        ("defect", broken_small[0]),
        ("good", good[13]),
        ("defect", broken_small[0]),
        ("good", good[14]),
        ("defect", broken_small[0]),
        ("good", good[15]),
    ]


def write_video(path: Path, shots):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed for {path}")

    for kind, pic in shots:
        if kind == "good":
            frame = load_and_resize(pic)
        elif kind == "defect":
            frame = load_and_resize(pic)
        else:
            base_img = load_and_resize(pic)
            frame = make_bad_frame(base_img, kind)

        for _ in range(FRAMES_PER_SHOT):
            writer.write(frame)

    writer.release()
    print(f"written {path.name} ({len(shots)} shots, {len(shots) * SHOT_SECONDS:.1f}s)")


def save_previews(good, dark_src, blur_src):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    good_img = load_and_resize(good)
    dark_img = make_bad_frame(load_and_resize(dark_src), "dark")
    blur_img = make_bad_frame(load_and_resize(blur_src), "blur")

    cv2.imwrite(str(PREVIEW_DIR / "good_preview.png"), good_img)
    cv2.imwrite(str(PREVIEW_DIR / "dark_preview.png"), dark_img)
    cv2.imwrite(str(PREVIEW_DIR / "blur_preview.png"), blur_img)

    print(f"saved previews to {PREVIEW_DIR}")


if __name__ == "__main__":
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    good = list_files(TEST_DIR / "good")
    broken_large = list_files(TEST_DIR / "broken_large")
    broken_small = list_files(TEST_DIR / "broken_small")
    contamination = list_files(TEST_DIR / "contamination")
    for image_list in (good, broken_large, broken_small, contamination):
        rng.shuffle(image_list)

    main_shots = [
        ("good", good[0]),
        ("good", good[1]),
        ("defect", broken_large[0]),
        ("good", good[2]),
        ("dark", good[3]),
        ("good", good[4]),
        ("defect", contamination[0]),
        ("good", good[5]),
        ("blur", good[6]),
        ("good", good[7]),
        ("defect", broken_small[0]),
        ("bright", good[8]),
        ("good", good[9]),
        ("defect", broken_large[1]),
        ("good", good[10]),
    ]

    repeat_shots = [
        ("good", good[11]),
        ("defect", broken_small[0]),
        ("good", good[12]),
        ("defect", broken_small[0]),
        ("good", good[13]),
        ("defect", broken_small[0]),
        ("good", good[14]),
        ("defect", broken_small[0]),
        ("good", good[15]),
    ]

    write_video(DEMO_DIR / "main_demo.mp4", main_shots)
    write_video(DEMO_DIR / "repeat_defect.mp4", repeat_shots)
    save_previews(good[0], good[3], good[6])

    print(f"main_demo_shots={len(main_shots)}")
    print(f"repeat_defect_shots={len(repeat_shots)}")
    print(f"demo_dir={DEMO_DIR}")
