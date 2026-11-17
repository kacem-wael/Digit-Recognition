import json, os
import cv2, numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

K, SEED = 5, 0
os.makedirs("figures", exist_ok=True)

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True, data_home="data")
X = X.reshape(-1, 28, 28).astype(np.uint8)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)

def hu(imgs):
    H = np.array([cv2.HuMoments(cv2.moments(im)).ravel() for im in imgs])
    return -np.sign(H) * np.log10(np.abs(H) + 1e-30)

Htr, Hte = hu(Xtr), hu(Xte)
sc = StandardScaler().fit(Htr)
Htr, Hte = sc.transform(Htr), sc.transform(Hte)

res = {"n_train": len(ytr), "n_test": len(yte), "k": K}
pca_full = PCA().fit(Htr)
res["explained_variance_ratio"] = pca_full.explained_variance_ratio_.round(5).tolist()

for d in (7, 6, 5, 4):
    pca = PCA(n_components=d).fit(Htr)
    Ztr, Zte = pca.transform(Htr), pca.transform(Hte)
    knn = KNeighborsClassifier(n_neighbors=K, n_jobs=-1).fit(Ztr, ytr)
    pred = knn.predict(Zte)
    acc = accuracy_score(yte, pred)
    res[f"acc_d{d}"] = round(acc, 4)
    res[f"var_d{d}"] = round(pca.explained_variance_ratio_.sum(), 4)
    res[f"per_class_d{d}"] = {c: round(float((pred[yte == c] == c).mean()), 4) for c in np.unique(yte)}
    print(f"d={d}  variance={res[f'var_d{d}']:.4f}  accuracy={acc:.4f}")
    if d == 4:
        ConfusionMatrixDisplay.from_predictions(yte, pred, normalize="true", values_format=".2f",
                                                cmap="Blues", colorbar=False)
        plt.title("Matrice de confusion (ACP d=4, KNN k=5)"); plt.tight_layout()
        plt.savefig("figures/confusion_d4.png", dpi=200); plt.close()

fig, ax = plt.subplots(2, 10, figsize=(10, 2.3))
for i, c in enumerate(sorted(np.unique(ytr), key=int)):
    for r, im in enumerate(Xtr[ytr == c][:2]):
        ax[r, i].imshow(im, cmap="gray"); ax[r, i].axis("off")
    ax[0, i].set_title(c)
plt.tight_layout(); plt.savefig("figures/samples.png", dpi=200); plt.close()

ev = pca_full.explained_variance_ratio_
plt.figure(figsize=(5, 3.2))
plt.bar(range(1, 8), ev, label="individuelle"); plt.step(range(1, 8), ev.cumsum(), where="mid", c="k", label="cumulée")
plt.xlabel("Composante principale"); plt.ylabel("Variance expliquée"); plt.legend(); plt.tight_layout()
plt.savefig("figures/pca_variance.png", dpi=200); plt.close()

dims = [7, 6, 5, 4]
plt.figure(figsize=(5, 3.2))
plt.plot(dims, [res[f"acc_d{d}"] for d in dims], "o-")
plt.gca().invert_xaxis(); plt.xticks(dims)
plt.xlabel("Dimension après ACP"); plt.ylabel("Précision (test)"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("figures/accuracy_vs_dim.png", dpi=200); plt.close()

json.dump(res, open("results.json", "w"), indent=2)
