# 程序化画面生成：用代码画图，不依赖 diffusion
import hashlib
import math
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

MEDIA = Path(__file__).resolve().parent / "data" / "media"
MEDIA.mkdir(parents=True, exist_ok=True)


def _seed(text):
    return int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8], 16)


def _gradient(width, height, top, bottom):
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = tuple(int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)
    return image


def _scene(text, style, t=0.0):
    rng = random.Random(_seed(text))
    width, height = 640, 480
    night = style in ("night", "rain") or "夜" in text or "晚" in text
    if night:
        image = _gradient(width, height, (18, 24, 48), (46, 38, 60))
    else:
        image = _gradient(width, height, (242, 196, 142), (120, 150, 190))
    draw = ImageDraw.Draw(image, "RGBA")

    # 星星 / 云
    for _ in range(40 if night else 12):
        x, y = rng.randrange(width), rng.randrange(height // 2)
        draw.ellipse([x, y, x + 2, y + 2], fill=(255, 255, 240, rng.randrange(80, 200)))

    # 窗户
    wx, wy, ww, wh = 110, 90, 200, 170
    draw.rectangle([wx, wy, wx + ww, wy + wh], outline=(220, 220, 230, 120), width=4)
    draw.line([wx + ww // 2, wy, wx + ww // 2, wy + wh], fill=(220, 220, 230, 90), width=2)
    draw.line([wx, wy + wh // 2, wx + ww, wy + wh // 2], fill=(220, 220, 230, 90), width=2)

    # 台灯与光晕
    lamp_x, lamp_y = 470, 330
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    radius = 90 + int(10 * math.sin(t * math.pi * 2))
    gdraw.ellipse([lamp_x - radius, lamp_y - radius, lamp_x + radius, lamp_y + radius],
                  fill=(255, 196, 96, 70))
    glow = glow.filter(ImageFilter.GaussianBlur(28))
    image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([lamp_x - 8, lamp_y - 70, lamp_x + 8, lamp_y], fill=(40, 40, 46, 255))
    draw.polygon([(lamp_x - 46, lamp_y - 70), (lamp_x + 46, lamp_y - 70), (lamp_x + 26, lamp_y - 104),
                  (lamp_x - 26, lamp_y - 104)], fill=(255, 214, 140, 255))

    # 她的剪影
    body_x = 300
    draw.ellipse([body_x - 46, 300, body_x + 46, 420], fill=(30, 30, 42, 235))
    draw.ellipse([body_x - 30, 252, body_x + 30, 316], fill=(30, 30, 42, 235))
    draw.ellipse([body_x - 18, 274, body_x - 10, 282], fill=(240, 240, 220, 220))
    draw.ellipse([body_x + 10, 274, body_x + 18, 282], fill=(240, 240, 220, 220))

    # 雨
    if style == "rain" or "雨" in text:
        for _ in range(120):
            x = rng.randrange(width)
            y = rng.randrange(height)
            draw.line([x, y, x - 4, y + 16], fill=(180, 200, 255, 90), width=1)

    return image


def generate(text, style="auto", animate=False):
    text = str(text or "安静的房间")
    if style == "auto":
        style = "night" if any(word in text for word in ("夜", "晚", "睡", "灯")) else "day"
    stamp = int(time.time() * 1000)
    if animate:
        frames = [_scene(text, style, t=i / 12).resize((480, 360)) for i in range(12)]
        path = MEDIA / ("anim-%d.gif" % stamp)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=140, loop=0)
        return {"ok": True, "path": str(path), "url": "/media/" + path.name,
                "kind": "gif", "style": style}
    image = _scene(text, style).resize((480, 360))
    path = MEDIA / ("img-%d.png" % stamp)
    image.save(path)
    return {"ok": True, "path": str(path), "url": "/media/" + path.name,
            "kind": "png", "style": style}
