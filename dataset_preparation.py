import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm

# --- CONFIG ---
CONFIG = {
    "image_size": (512, 512),
    "num_images": 2000,
    "num_stickers_per_image": (1, 5),
    "sticker_scale_range": (0.03, 0.3),
    "rotation_range": (0, 360),
    "class_distribution": {
        "happy": 1.0, "sad": 1.0, "dead": 1.0
    },
    "output_dir": "dataset",
    "stickers_dir": "stickers",
    "backgrounds_dir": "backgrounds"
}

CLASSES = list(CONFIG["class_distribution"].keys())

def crop_sticker_with_center(img):
    # Optional horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w, h = img.size
    x0 = int(w * 0.5)
    x1 = int(w * 0.7)
    y0 = int(h * 0.35)
    y1 = int(h * 0.65)

    expand_w = random.randint(0, min(x0, w - x1))
    expand_h = random.randint(0, min(y0, h - y1))

    left = x0 - expand_w
    right = x1 + expand_w
    top = y0 - expand_h
    bottom = y1 + expand_h

    return img.crop((left, top, right, bottom))

def apply_color_jitter(img):
    brightness_factor = random.uniform(0.3, 1.7)
    contrast_factor = random.uniform(0.3, 1.7)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img

def apply_rotation(img):
    angle = random.uniform(*CONFIG["rotation_range"])
    return img.rotate(angle, expand=True)

def apply_affine_distortion(img):
    w, h = img.size
    dx = int(w * 0.2)
    dy = int(h * 0.2)
    coeffs = [
        1 + random.uniform(-0.2, 0.2), random.uniform(-0.3, 0.3), random.randint(-dx, dx),
        random.uniform(-0.3, 0.3), 1 + random.uniform(-0.2, 0.2), random.randint(-dy, dy)
    ]
    # expand canvas to keep all content after transformation
    expanded_w = int(w * 1.5)
    expanded_h = int(h * 1.5)
    expanded = Image.new("RGBA", (expanded_w, expanded_h), (0, 0, 0, 0))
    expanded.paste(img, ((expanded_w - w) // 2, (expanded_h - h) // 2))
    transformed = expanded.transform((expanded_w, expanded_h), Image.AFFINE, coeffs, resample=Image.BICUBIC)
    # crop back to non-empty bounding box
    bbox = transformed.getbbox()
    return transformed.crop(bbox)

def paste_sticker(bg, sticker, x1, y1):
    if random.random() < 0.3:  # вероятность применения прозрачности
        alpha = sticker.getchannel("A")
        alpha = alpha.point(lambda p: int(p * random.uniform(0.7, 0.9)))  # 70% - 90% прозрачность
        sticker.putalpha(alpha)
    bg.paste(sticker, (x1, y1), sticker)
    return bg

def split_indices(total, train_ratio=0.8, val_ratio=0.1):
    indices = list(range(total))
    random.shuffle(indices)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    return {
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:]
    }

def boxes_overlap(x1, y1, w1, h1, boxes):
    x2, y2 = x1 + w1, y1 + h1
    for bx, by, bw, bh in boxes:
        bx2, by2 = bx + bw, by + bh
        if not (x2 < bx or x1 > bx2 or y2 < by or y1 > by2):
            return True
    return False

def generate():
    split = split_indices(CONFIG["num_images"])
    for split_name in split:
        os.makedirs(os.path.join(CONFIG["output_dir"], "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(CONFIG["output_dir"], "labels", split_name), exist_ok=True)

    sticker_paths = {cls: os.path.join(CONFIG["stickers_dir"], f"{cls}.png") for cls in CLASSES}
    backgrounds = os.listdir(CONFIG["backgrounds_dir"])

    for idx in tqdm(range(CONFIG["num_images"])):
        bg_path = os.path.join(CONFIG["backgrounds_dir"], random.choice(backgrounds))
        bg = Image.open(bg_path).convert("RGB").resize(CONFIG["image_size"])
        w_bg, h_bg = CONFIG["image_size"]

        num_stickers = random.randint(*CONFIG["num_stickers_per_image"])
        annotations = []
        occupied = []

        for _ in range(num_stickers):
            cls = random.choice(CLASSES)
            sticker_img = Image.open(sticker_paths[cls]).convert("RGBA")
            sticker_img = crop_sticker_with_center(sticker_img)
            if random.random() < 0.5:
                sticker_img = sticker_img.transpose(Image.FLIP_LEFT_RIGHT)
            sticker_img = apply_color_jitter(sticker_img)
            sticker_img = apply_rotation(sticker_img)
            sticker_img = apply_affine_distortion(sticker_img)

            scale = random.uniform(*CONFIG["sticker_scale_range"])
            new_w = int(w_bg * scale)
            aspect = sticker_img.width / sticker_img.height
            new_h = int(new_w / aspect)
            sticker_img = sticker_img.resize((new_w, new_h), resample=Image.BICUBIC)

            if new_w >= w_bg or new_h >= h_bg:
                continue
            for _ in range(20):  # try 20 times to find non-overlapping spot
                x1 = random.randint(0, w_bg - new_w)
                y1 = random.randint(0, h_bg - new_h)
                if not boxes_overlap(x1, y1, new_w, new_h, occupied):
                    occupied.append((x1, y1, new_w, new_h))
                    bg = paste_sticker(bg, sticker_img, x1, y1)

                    x_center = (x1 + new_w / 2) / w_bg
                    y_center = (y1 + new_h / 2) / h_bg
                    rel_w = new_w / w_bg
                    rel_h = new_h / h_bg

                    annotations.append(f"{CLASSES.index(cls)} {x_center:.6f} {y_center:.6f} {rel_w:.6f} {rel_h:.6f}")
                    break

        split_type = [k for k, v in split.items() if idx in v][0]
        img_name = f"{idx:05d}.jpg"
        label_name = f"{idx:05d}.txt"

        bg.save(os.path.join(CONFIG["output_dir"], "images", split_type, img_name))
        with open(os.path.join(CONFIG["output_dir"], "labels", split_type, label_name), 'w') as f:
            f.write("\n".join(annotations) + "\n")

    print("✅ Генерация завершена")

if __name__ == '__main__':
    generate()
