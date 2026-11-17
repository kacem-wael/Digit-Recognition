import base64, json, os, sys, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
from features import normalize_digit, extract, hog, hu, group_slices, GROUPS
from nn import MLP

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL, METRICS = os.path.join(ROOT, "model/digit_net.npz"), os.path.join(ROOT, "model/metrics.json")
if not os.path.exists(MODEL):
    sys.exit("No trained model found. Run `python train.py` first.")
NET = MLP.load(MODEL)
INFO = json.load(open(METRICS))
INFO.pop("history", None)
SLICES = group_slices()


def top_edges(contrib, k, pos_share=0.7):
    out = []
    for sign, n in ((1, int(k * pos_share)), (-1, k - int(k * pos_share))):
        flat = (sign * contrib).ravel()
        n = min(n, int((flat > 0).sum()))
        if n == 0:
            continue
        idx = np.argpartition(-flat, n - 1)[:n]
        i, j = np.unravel_index(idx, contrib.shape)
        out += [[int(a), int(b), round(float(contrib[a, b]), 4)] for a, b in zip(i, j)]
    return out


def analyse(gray):
    img, aspect = normalize_digit(gray)
    if img.max() == 0:
        return {"empty": True}
    feats = extract(img[None], np.array([aspect], np.float32))
    probs, acts = NET.forward(feats, keep=True)
    z, h1, h2 = acts[0][0], acts[1][0], acts[2][0]
    W1, W2, W3 = NET.W
    group_to_h1 = np.stack([z[s] @ W1[s] for _, s in SLICES])
    topo = feats[0][SLICES[-1][1]]
    return {
        "empty": False,
        "probs": probs[0].round(5).tolist(),
        "image": img.round(3).ravel().tolist(),
        "h1": h1.round(4).tolist(), "h2": h2.round(4).tolist(),
        "out": acts[3][0].round(4).tolist(),
        "groups": [n for n, _ in GROUPS],
        "group_strength": np.abs(group_to_h1).sum(1).round(3).tolist(),
        "edges": {
            "in_h1": top_edges(group_to_h1, 140),
            "h1_h2": top_edges(h1[:, None] * W2, 220),
            "h2_out": top_edges(h2[:, None] * W3, 120),
        },
        "descriptors": {
            "hog": hog(img[None])[0].round(3).tolist(),
            "zoning": img.reshape(4, 7, 4, 7).mean((1, 3)).round(3).tolist(),
            "hu": hu(img).round(2).tolist(),
            "holes": int(round(topo[1] * 1 + topo[2] * 2)),
            "endpoints": int(round(topo[5] * 2)),
            "junctions": int(round(topo[6] * 2)),
            "aspect": round(float(aspect), 2),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "web/index.html"), "rb") as f:
                self.send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self.send(200, INFO)
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            png = base64.b64decode(body["image"].split(",", 1)[-1])
            rgba = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
            gray = rgba[..., 3] if rgba.ndim == 3 and rgba.shape[2] == 4 else \
                cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY) if rgba.ndim == 3 else rgba
            self.send(200, analyse(gray))
        except Exception as e:
            self.send(400, {"error": str(e)})


if __name__ == "__main__":
    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8000)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Digit recognizer running at {url}   (test accuracy {INFO['test_accuracy']:.2%})  Ctrl+C to stop")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
