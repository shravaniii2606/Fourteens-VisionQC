from pathlib import Path
import random
import shutil

BASE = Path(__file__).resolve().parents[1]
SRC_DIR = BASE / "data" / "mvtec" / "bottle" / "train" / "good"
FIT_DIR = BASE / "data" / "fit_set"
HOLDOUT_DIR = BASE / "data" / "holdout_good"
REF_PATH = BASE / "data" / "reference.jpg"

FIT_DIR.mkdir(parents=True, exist_ok=True)
HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

for folder in (FIT_DIR, HOLDOUT_DIR):
    for child in folder.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            shutil.rmtree(child)

files = sorted(p for p in SRC_DIR.iterdir() if p.is_file())
if len(files) < 36:
    raise RuntimeError(f"Not enough good training images for a 25+10+1 split: found {len(files)}")

rng = random.Random(42)
rng.shuffle(files)
fit_files = files[:25]
hold_files = files[25:35]
reference_file = files[35]

for src in fit_files:
    shutil.copy2(src, FIT_DIR / src.name)
for src in hold_files:
    shutil.copy2(src, HOLDOUT_DIR / src.name)
shutil.copy2(reference_file, REF_PATH)

print(f"fit_set={len(fit_files)}")
print(f"holdout_good={len(hold_files)}")
print(f"reference={REF_PATH.name}")
print(f"reference_source={reference_file.name}")
