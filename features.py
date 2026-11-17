import cv2
import numpy as np
from skimage.morphology import skeletonize

SIZE, BOX = 28, 20
CELL, NBINS = 4, 9

GROUPS = [("Pixels", 784), ("HOG", 441), ("Hu moments", 7),
          ("Zoning", 16), ("Projections", 56), ("Topology", 10)]
N_FEATURES = sum(n for _, n in GROUPS)
TOPOLOGY_NAMES = ["0 holes", "1 hole", "2+ holes", "hole height", "hole area",
                  "end-points", "junctions", "ends top", "ends bottom", "log aspect"]


def normalize_digit(img):
    img = np.asarray(img, np.float32)
    if img.max() <= 0:
        return np.zeros((SIZE, SIZE), np.float32), 1.0
    img = img / img.max()
    ys, xs = np.nonzero(img > 0.1)
    if len(ys) == 0:
        return np.zeros((SIZE, SIZE), np.float32), 1.0
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = BOX / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    m = cv2.moments(small)
    cy, cx = (m["m01"] / m["m00"], m["m10"] / m["m00"]) if m["m00"] > 0 else (nh / 2, nw / 2)
    oy = int(np.clip(round(SIZE / 2 - cy), 0, SIZE - nh))
    ox = int(np.clip(round(SIZE / 2 - cx), 0, SIZE - nw))
    out = np.zeros((SIZE, SIZE), np.float32)
    out[oy:oy + nh, ox:ox + nw] = small
    return out / max(out.max(), 1e-6), w / h


def hog(X):
    N = len(X)
    gx = np.zeros_like(X); gy = np.zeros_like(X)
    gx[:, :, 1:-1] = X[:, :, 2:] - X[:, :, :-2]
    gy[:, 1:-1, :] = X[:, 2:, :] - X[:, :-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * NBINS - 0.5
    b0 = np.floor(ang).astype(np.int64)
    f = (ang - b0).astype(np.float32)
    b0 %= NBINS
    b1 = (b0 + 1) % NBINS
    g = SIZE // CELL
    cell_idx = (np.arange(SIZE) // CELL)
    cid = (cell_idx[:, None] * g + cell_idx[None, :]).ravel()
    base = (np.arange(N)[:, None] * g * g + cid[None, :]) * NBINS
    hist = np.zeros(N * g * g * NBINS, np.float32)
    m = mag.reshape(N, -1)
    np.add.at(hist, (base + b0.reshape(N, -1)).ravel(), (m * (1 - f.reshape(N, -1))).ravel())
    np.add.at(hist, (base + b1.reshape(N, -1)).ravel(), (m * f.reshape(N, -1)).ravel())
    hist = hist.reshape(N, g, g, NBINS)
    energy = (hist ** 2).sum(-1)
    local = np.pad(energy, ((0, 0), (1, 1), (1, 1)))
    local = sum(local[:, i:i + g, j:j + g] for i in range(3) for j in range(3))
    return hist / np.sqrt(local[..., None] / 9 + 1e-3)


def hu(img):
    h = cv2.HuMoments(cv2.moments(img)).ravel()
    return -np.sign(h) * np.log10(np.abs(h) + 1e-30)


_NB_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)


def topology(img, aspect):
    ink = (img > 0.35).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(1 - ink, connectivity=4)
    holes = [i for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= 3
             and stats[i, cv2.CC_STAT_LEFT] > 0 and stats[i, cv2.CC_STAT_TOP] > 0
             and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] < SIZE
             and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] < SIZE]
    nh = len(holes)
    if nh:
        big = max(holes, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        hole_y = cents[big][1] / SIZE
        hole_a = sum(stats[i, cv2.CC_STAT_AREA] for i in holes) / 100.0
    else:
        hole_y, hole_a = 0.5, 0.0
    sk = skeletonize(ink.astype(bool)).astype(np.uint8)
    nb = cv2.filter2D(sk.astype(np.float32), -1, _NB_KERNEL, borderType=cv2.BORDER_CONSTANT)
    ends = (sk == 1) & (nb == 1)
    junc = ((sk == 1) & (nb >= 3)).astype(np.uint8)
    n_junc = cv2.connectedComponents(junc, connectivity=8)[0] - 1 if junc.any() else 0
    ey = np.nonzero(ends)[0]
    return np.array([nh == 0, nh == 1, nh >= 2, hole_y, hole_a,
                     min(len(ey), 6) / 2, min(n_junc, 6) / 2,
                     (ey < SIZE / 2).sum() / 2, (ey >= SIZE / 2).sum() / 2,
                     np.log(np.clip(aspect, 0.05, 5))], np.float32)


def extract(X, aspects, chunk=4096):
    X = np.asarray(X, np.float32)
    out = np.empty((len(X), N_FEATURES), np.float32)
    for s in range(0, len(X), chunk):
        x, a = X[s:s + chunk], aspects[s:s + chunk]
        n = len(x)
        out[s:s + n] = np.concatenate([
            x.reshape(n, -1),
            hog(x).reshape(n, -1),
            np.array([hu(im) for im in x]),
            x.reshape(n, 4, 7, 4, 7).mean((2, 4)).reshape(n, -1),
            np.concatenate([x.mean(2), x.mean(1)], 1),
            np.array([topology(im, ar) for im, ar in zip(x, a)]),
        ], 1)
    return out


def group_slices():
    s, out = 0, []
    for name, n in GROUPS:
        out.append((name, slice(s, s + n)))
        s += n
    return out
