import csv
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TEST_DIR = BASE / "data" / "mvtec" / "bottle" / "test"
OUTPUT_PATH = BASE / "demo" / "shot_list.csv"


def list_files(directory: Path):
    return sorted(path for path in directory.iterdir() if path.is_file())


def main_sequence():
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


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    shot_order = main_sequence()
    for index, (kind, source_path) in enumerate(shot_order, start=1):
        start = (index - 1) * 1.5
        end = index * 1.5
        expected = {
            "good": "PASS",
            "defect": "FAIL",
            "dark": "RECAPTURE",
            "bright": "RECAPTURE",
            "blur": "RECAPTURE",
        }[kind]
        rows.append(
            {
                "shot_number": index,
                "source_image_filename": source_path.name,
                "type": kind,
                "expected_verdict": expected,
                "start_time_s": f"{start:.1f}",
                "end_time_s": f"{end:.1f}",
            }
        )

    with OUTPUT_PATH.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "shot_number",
                "source_image_filename",
                "type",
                "expected_verdict",
                "start_time_s",
                "end_time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"shot_list_rows={len(rows)}")
    print(f"shot_list_csv={OUTPUT_PATH}")
