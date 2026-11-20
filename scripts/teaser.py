"""Tile the policy panels of rendered rollouts (left half of each video) into a looping GIF.

    python scripts/teaser.py OUT.gif VIDEO... [--fps 12] [--cols 2] [--scale 0.5] [--crop-top 0.2]
"""
import argparse

import imageio_ffmpeg
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("output")
p.add_argument("videos", nargs="+")
p.add_argument("--fps", type=int, default=12)
p.add_argument("--cols", type=int, default=2)
p.add_argument("--scale", type=float, default=0.5)
p.add_argument("--colors", type=int, default=64)
p.add_argument("--crop-top", type=float, default=0.0, help="Fraction of each panel cut from the top (the empty sky)")
a = p.parse_args()

clips = []
for path in a.videos:
    reader = imageio_ffmpeg.read_frames(path)
    meta = next(reader)
    (w, h), fps = meta["size"], meta["fps"]
    step = max(1, round(fps / a.fps))
    frames = []
    for i, raw in enumerate(reader):
        if i % step:
            continue
        frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3)[int(a.crop_top * h):, : w // 2]
        img = Image.fromarray(frame).resize((int(frame.shape[1] * a.scale), int(frame.shape[0] * a.scale)), Image.LANCZOS)
        frames.append(np.asarray(img))
    clips.append(frames)
n = max(len(c) for c in clips)
rows = (len(clips) + a.cols - 1) // a.cols
ph, pw = clips[0][0].shape[:2]
canvases = []
for t in range(n):
    canvas = np.full((rows * ph, a.cols * pw, 3), 255, np.uint8)
    for k, c in enumerate(clips):
        r, col = divmod(k, a.cols)
        canvas[r * ph:(r + 1) * ph, col * pw:(col + 1) * pw] = c[min(t, len(c) - 1)]
    canvases.append(canvas)
# One palette for the whole animation, from a sample of its frames: colors stay put from frame to frame.
sample = np.concatenate(canvases[:: max(1, n // 8)], 0)
palette = Image.fromarray(sample).quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT)
out = [Image.fromarray(c).quantize(palette=palette, dither=Image.Dither.NONE) for c in canvases]
out[0].save(a.output, save_all=True, append_images=out[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
print(a.output, len(out), "frames", out[0].size)
