#!/usr/bin/env python3
"""CPU point trackers for the gripper in 17-frame clips, (row, col) like eef_px_raw. Validated on real
frames during the E4 audit: tracked vs projected EEF displacement cosine 0.98-0.99."""
import numpy as np, cv2

def _gray(v):
    return [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in v]

_DIS = None
def flow_pair(a, b, method):
    global _DIS
    if method == "farneback":
        return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 15, 3, 5, 1.2, 0)
    if method == "dis":
        if _DIS is None:
            _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return _DIS.calc(a, b, None)
    raise ValueError(method)

def track_flow(video, p0, method="farneback", h=8, weight="mag"):
    """Chain frame-to-frame dense flow; at each step move the point by the (magnitude-weighted) mean
    flow in a (2h+1)^2 neighbourhood. Returns (17,2) track in (row, col)."""
    g = _gray(video); H, W = g[0].shape
    p = np.array(p0, np.float64); tr = [p.copy()]
    for k in range(1, len(g)):
        fl = flow_pair(g[k - 1], g[k], method)  # fl[y,x] = (dx, dy)
        r, c = int(round(np.clip(p[0], h, H - 1 - h))), int(round(np.clip(p[1], h, W - 1 - h)))
        f = fl[r - h:r + h + 1, c - h:c + h + 1].reshape(-1, 2)
        if weight == "mag":
            w = np.linalg.norm(f, axis=1) + 1e-6
            v = (f * w[:, None]).sum(0) / w.sum()
        else:
            v = f.mean(0)
        p = p + np.array([v[1], v[0]])
        p = np.clip(p, 0, [H - 1, W - 1]); tr.append(p.copy())
    return np.array(tr)

def track_template(video, p0, half=14, search=12):
    """Frame-to-frame template matching (TM_CCOEFF_NORMED), template refreshed every frame."""
    g = _gray(video); H, W = g[0].shape
    p = np.array(p0, np.float64); tr = [p.copy()]
    for k in range(1, len(g)):
        r, c = int(round(np.clip(p[0], half, H - 1 - half))), int(round(np.clip(p[1], half, W - 1 - half)))
        T = g[k - 1][r - half:r + half + 1, c - half:c + half + 1]
        r0, r1 = max(r - half - search, 0), min(r + half + search + 1, H)
        c0, c1 = max(c - half - search, 0), min(c + half + search + 1, W)
        S = g[k][r0:r1, c0:c1]
        res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        nr, nc = r0 + loc[1] + half, c0 + loc[0] + half
        p = p + np.array([nr - r, nc - c], np.float64)
        p = np.clip(p, 0, [H - 1, W - 1]); tr.append(p.copy())
    return np.array(tr)

def cos(a, b, eps=1e-9):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    return float(np.dot(a, b) / (na * nb))
