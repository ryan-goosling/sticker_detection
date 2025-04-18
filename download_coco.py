import os
import zipfile
import shutil
import urllib.request
from tqdm import tqdm

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
TARGET_DIR = "backgrounds"
TEMP_ZIP = "val2017.zip"
EXTRACT_DIR = "val2017"

def download(url, filename):
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        total = int(response.getheader('Content-Length').strip())
        with tqdm(total=total, unit='B', unit_scale=True, desc="Downloading") as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Step 1: Download COCO val2017 zip
    print("📥 Downloading COCO val2017...")
    download(COCO_URL, TEMP_ZIP)

    # Step 2: Unzip
    print("📦 Extracting...")
    with zipfile.ZipFile(TEMP_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")

    # Step 3: Copy first 2000 images to backgrounds/
    print("📂 Copying 2000 images to 'backgrounds/'...")
    images = sorted(os.listdir(EXTRACT_DIR))[:2000]
    for img in tqdm(images):
        src = os.path.join(EXTRACT_DIR, img)
        dst = os.path.join(TARGET_DIR, img)
        shutil.copy(src, dst)

    # Step 4: Cleanup
    os.remove(TEMP_ZIP)
    shutil.rmtree(EXTRACT_DIR)

    print("✅ Done! Backgrounds saved to:", TARGET_DIR)

if __name__ == '__main__':
    main()
