import math
from PIL import Image, ImageDraw, ImageFont


def generate_logo(size=256):
    img = Image.new("RGBA", (size, size), (15, 15, 35, 255))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    r = size * 0.35

    # face outline - geometric polygon
    face_pts = []
    for i in range(6):
        angle = math.radians(60 * i - 90)
        face_pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(face_pts, outline=(0, 212, 255, 180), fill=None)
    draw.polygon(face_pts, outline=(123, 47, 247, 180), fill=None)

    # inner triangle
    inner_r = r * 0.55
    tri_pts = []
    for i in range(3):
        angle = math.radians(120 * i - 90)
        tri_pts.append((cx + inner_r * math.cos(angle), cy + inner_r * math.sin(angle)))
    draw.polygon(tri_pts, outline=(0, 212, 255, 120), fill=None)

    # landmark dots
    dot_positions = []
    for i in range(6):
        angle = math.radians(60 * i - 90)
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        dot_positions.append((px, py))
    for i in range(3):
        angle = math.radians(120 * i - 90)
        px = cx + inner_r * math.cos(angle)
        py = cy + inner_r * math.sin(angle)
        dot_positions.append((px, py))

    # center dot
    dot_positions.append((cx, cy))

    for px, py in dot_positions:
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(0, 212, 255, 255))
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 255, 255))

    # connecting lines
    for i in range(len(dot_positions)):
        for j in range(i + 1, len(dot_positions)):
            dist = math.hypot(dot_positions[i][0] - dot_positions[j][0],
                              dot_positions[i][1] - dot_positions[j][1])
            if dist < r * 1.2:
                draw.line([dot_positions[i], dot_positions[j]],
                          fill=(0, 212, 255, 60), width=1)

    # eye-like arcs inside
    eye_r = r * 0.18
    for side in [-1, 1]:
        ex = cx + side * r * 0.25
        ey = cy - r * 0.1
        draw.ellipse([ex - eye_r, ey - eye_r, ex + eye_r, ey + eye_r],
                     outline=(123, 47, 247, 200), width=2)

    # mouth arc
    mouth_y = cy + r * 0.35
    mouth_w = r * 0.3
    draw.arc([cx - mouth_w, mouth_y - mouth_w * 0.3, cx + mouth_w, mouth_y + mouth_w * 0.3],
             0, 180, fill=(0, 212, 255, 150), width=2)

    return img


def generate_banner(width=1280, height=640):
    img = Image.new("RGBA", (width, height), (15, 15, 35, 255))
    draw = ImageDraw.Draw(img)

    # gradient background
    for y in range(height):
        ratio = y / height
        r = int(15 + ratio * 10)
        g = int(15 + ratio * 5)
        b = int(35 + ratio * 30)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # wireface mesh in background
    mesh_pts = []
    num_cols = 12
    num_rows = 8
    for row in range(num_rows):
        for col in range(num_cols):
            x = width * 0.55 + (col - num_cols / 2) * 55 + (row % 2) * 25
            y = height * 0.5 + (row - num_rows / 2) * 50
            jitter_x = math.sin(row * 1.5 + col * 0.7) * 15
            jitter_y = math.cos(row * 0.8 + col * 1.2) * 10
            mesh_pts.append((x + jitter_x, y + jitter_y))

    # connect nearby points
    for i, p1 in enumerate(mesh_pts):
        for j, p2 in enumerate(mesh_pts):
            if j <= i:
                continue
            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if dist < 90:
                alpha = max(10, int(60 * (1 - dist / 90)))
                draw.line([p1, p2], fill=(0, 212, 255, alpha), width=1)

    # draw mesh points
    for px, py in mesh_pts:
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(0, 212, 255, 80))

    # accent lines - geometric triangles
    tri_sets = [
        [(width * 0.7, height * 0.15), (width * 0.85, height * 0.45), (width * 0.55, height * 0.45)],
        [(width * 0.75, height * 0.6), (width * 0.9, height * 0.85), (width * 0.6, height * 0.85)],
    ]
    for tri in tri_sets:
        draw.polygon(tri, outline=(123, 47, 247, 60), fill=None)

    # text
    try:
        font_title = ImageFont.truetype("arial.ttf", 64)
        font_sub = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 64)
            font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        except OSError:
            font_title = ImageFont.load_default()
            font_sub = ImageFont.load_default()

    title = "ORFormer-Lite"
    tagline = "Real-time Facial Expression Recognition for Assistive Communication"

    # title with shadow
    tx, ty = 80, height // 2 - 80
    draw.text((tx + 2, ty + 2), title, fill=(0, 0, 0, 120), font=font_title)
    draw.text((tx, ty), title, fill=(255, 255, 255, 240), font=font_title)

    # tagline
    draw.text((tx, ty + 90), tagline, fill=(180, 180, 200, 200), font=font_sub)

    # accent bar under title
    draw.rectangle([tx, ty + 80, tx + 300, ty + 84], fill=(0, 212, 255, 200))

    # small decorative dots
    for i in range(5):
        dx = tx + 320 + i * 20
        dy = ty + 82
        draw.ellipse([dx - 3, dy - 3, dx + 3, dy + 3], fill=(123, 47, 247, 150))

    return img


if __name__ == "__main__":
    logo = generate_logo(256)
    logo.save("assets/logo.png")
    print("Saved assets/logo.png")

    banner = generate_banner(1280, 640)
    banner.save("assets/banner.png")
    print("Saved assets/banner.png")
