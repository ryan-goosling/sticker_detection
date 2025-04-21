import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
import numpy as np
import cv2

# --- CONFIG ---
CONFIG = {
    "image_size": (512, 512),
    "num_images": 2000,
    "num_stickers_per_image": (1, 5),
    "sticker_scale_range": (0.04, 0.30),
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
    w, h = img.size
    x0 = int(w * 0.2)
    x1 = int(w * 0.8)
    y0 = int(h * 0.2)
    y1 = int(h * 0.8)
    expand_w = random.randint(0, min(x0, w - x1))
    expand_h = random.randint(0, min(y0, h - y1))
    return img.crop((x0 - expand_w, y0 - expand_h, x1 + expand_w, y1 + expand_h))

def apply_color_jitter(img):
    w, h = img.size
    area = w * h
    scale = min(1.0, area / (128 * 128))

    brightness_factor = random.uniform(1 - 0.4 * scale, 1 + 0.4 * scale)
    contrast_factor   = random.uniform(1 - 0.4 * scale, 1 + 0.4 * scale)
    saturation_factor = random.uniform(1 - 0.2 * scale, 1 + 0.2 * scale)

    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    img = ImageEnhance.Color(img).enhance(saturation_factor)

    img_np = np.array(img).astype(np.float32)

    # 🔵 Эффект холодного/тёплого освещения
    if random.random() < 0.5:
        tone = random.choice(['warm', 'cool'])
        alpha = random.uniform(0.05, 0.4 * scale)
        if tone == 'warm':
            img_np[..., 0] *= 1 - alpha
            img_np[..., 2] *= 1 + alpha
        else:
            img_np[..., 0] *= 1 + alpha
            img_np[..., 2] *= 1 - alpha

    # 🌒 Понижение точки белого
    if random.random() < 0.4:
        gamma = random.uniform(1.2, 5.0)  # Чем выше — тем темнее
        img_np = 255 * ((img_np / 255) ** gamma)

    img_np = np.clip(img_np, 0, 255)
    return Image.fromarray(img_np.astype(np.uint8))




def apply_rotation(img):
    return img.rotate(random.uniform(*CONFIG["rotation_range"]), expand=True)

def apply_perspective_transform(pil_img):
    img = np.array(pil_img)
    h, w = img.shape[:2]
    area = w * h

    # Пороговые значения
    area_values = [32*32, 50*50, 75*75, 150*150]
    strength_values = [0.07, 0.15, 0.3, 0.4]

    # Интерполируем strength в зависимости от площади
    strength = float(np.interp(area, area_values, strength_values))

    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    delta = strength * min(w, h)
    pts2 = pts1 + np.random.uniform(-delta, delta, pts1.shape).astype(np.float32)

    shift = np.array([[-np.min(pts2[:, 0]), -np.min(pts2[:, 1])]], dtype=np.float32)
    shifted_pts2 = pts2 + shift

    matrix = cv2.getPerspectiveTransform(pts1, shifted_pts2)
    new_w = int(np.max(shifted_pts2[:, 0]))
    new_h = int(np.max(shifted_pts2[:, 1]))

    transformed = cv2.warpPerspective(
        img, matrix, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    return Image.fromarray(transformed)




def apply_photorealistic_effects(img):
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    area = w * h

    # Масштаб силы эффектов: меньше для маленьких картинок
    scale = min(1.0, area / (128 * 128))  # от 0 до 1

    if random.random() < 0.3 * scale:  # слабее вероятность
        ksize = 3 if area < 128 * 128 else 5
        sigma = random.uniform(0.3, 1.0) * scale
        img_np = cv2.GaussianBlur(img_np, (ksize, ksize), sigmaX=sigma)

    if random.random() < 0.5 * scale:
        noise_level = int(5 + 10 * scale)  # от 5 до 15
        noise = np.random.normal(0, noise_level, img_np.shape).astype(np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.3 * scale:
        ksize = random.choice([3, 5]) if area > 128 * 128 else 3
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = np.ones(ksize)
        kernel = kernel / ksize
        img_np = cv2.filter2D(img_np, -1, kernel)

    return Image.fromarray(img_np)


def paste_sticker(bg, sticker, x1, y1):
    if random.random() < 0.2:
        alpha = sticker.getchannel("A")
        alpha = alpha.point(lambda p: int(p * random.uniform(0.8, 0.95)))
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
        if not (x2 < bx or x1 > bx + bw or y2 < by or y1 > by + bh):
            return True
    return False

def generate():
    split = split_indices(CONFIG["num_images"])
    for split_name in split:
        os.makedirs(os.path.join(CONFIG["output_dir"], "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(CONFIG["output_dir"], "labels", split_name), exist_ok=True)

    backgrounds = os.listdir(CONFIG["backgrounds_dir"])
    sticker_dirs = {
        cls: [os.path.join(CONFIG["stickers_dir"], cls, f) for f in os.listdir(os.path.join(CONFIG["stickers_dir"], cls)) if f.endswith(".png")]
        for cls in CLASSES
    }

    for idx in tqdm(range(CONFIG["num_images"])):
        bg_path = os.path.join(CONFIG["backgrounds_dir"], random.choice(backgrounds))
        bg = Image.open(bg_path).convert("RGB").resize(CONFIG["image_size"])
        w_bg, h_bg = CONFIG["image_size"]

        num_stickers = random.randint(*CONFIG["num_stickers_per_image"])
        annotations = []
        occupied = []

        for _ in range(num_stickers):
            cls = random.choice(CLASSES)
            sticker_path = random.choice(sticker_dirs[cls])
            sticker_img = Image.open(sticker_path).convert("RGBA")

            if random.random() < 0.3:
                arr = np.array(sticker_img)
                arr[..., :3] = 255 - arr[..., :3]  # инвертируем только RGB, не трогаем альфу
                sticker_img = Image.fromarray(arr, mode="RGBA")


            if random.random() < 0.5:
                sticker_img = apply_photorealistic_effects(sticker_img)
            sticker_img = crop_sticker_with_center(sticker_img)
            if random.random() < 0.5:
                sticker_img = sticker_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.9:  
                sticker_img = apply_rotation(sticker_img)
            sticker_img = apply_color_jitter(sticker_img)
            if random.random() < 0.6:
                sticker_img = apply_perspective_transform(sticker_img)

            new_w, new_h = sticker_img.size
            if new_w >= w_bg or new_h >= h_bg:
                continue
            for _ in range(20):
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

    print("✅ Generation done")

if __name__ == '__main__':
    generate()
