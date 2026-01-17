import os
from pathlib import Path

import json

import cv2
import pytesseract

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


INPUT = Path("")
OUTPUT = ""

TESSERACT_CMD = r""

MAX_WORKERS = os.cpu_count() - 1


def preprocess_image(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        11
    )
    
    return gray

def initialize_worker():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def ocr_one(path_str):
    image_path = str(Path(path_str))
    image = cv2.imread(image_path)
    
    if image is None:
        return None

    processed_image = preprocess_image(image)

    text = pytesseract.image_to_string(
        processed_image,
        lang="eng",
        config="--oem 3 --psm 6"
    ).strip()

    return {"file": image_path, "text": text}

def main():
    paths = [str(path) for path in INPUT.rglob("*")]

    results = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=initialize_worker) as executor:
        for record in tqdm(executor.map(ocr_one, paths, chunksize=8), total=len(paths)):
            if record is not None:
                results.append(record)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
