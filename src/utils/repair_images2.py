from pathlib import Path
import cv2
from tqdm import tqdm


IN_DIR = Path(r"")
OUT_DIR = Path(r"")

def normalize_file(src, dst):
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return cv2.imwrite(str(dst), img)

ok = 0
fail = 0
for p in tqdm(IN_DIR.rglob("*")):
    if not p.is_file():
        continue
    out = OUT_DIR / (p.stem + ".png")
    if normalize_file(p, out):
        ok += 1
    else:
        fail += 1

print("Normalized:", ok, "Failed:", fail)
