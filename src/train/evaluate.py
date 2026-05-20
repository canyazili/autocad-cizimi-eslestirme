"""
Etiketli (foto, dwg) çiftleri üzerinde Recall@K, MRR raporu.

Kullanım:
    python -m src.train.evaluate --k 1 5 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMBEDDINGS, MODEL_BEST, CLIP_BACKBONE, CLIP_PRETRAINED, PHOTOS  # noqa: E402
from utils import load_rgb, foto_edge, labels_yukle  # noqa: E402


def model_yukle(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_PRETRAINED)
    ckpt = torch.load(MODEL_BEST, map_location="cpu", weights_only=False)
    state = {k.replace("clip_model.", "", 1) if k.startswith("clip_model.") else k: v for k, v in ckpt.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), preprocess


def bank_yukle():
    d = np.load(EMBEDDINGS, allow_pickle=True)
    edge = d["edge_embeddings"].astype(np.float32)
    edge /= np.clip(np.linalg.norm(edge, axis=1, keepdims=True), 1e-8, None)
    adlar = np.array([str(a) for a in d["dosya_adlari"]])
    # rotasyon birleştir: her baz_ad için indekslerin listesi
    bazlar = np.array([a.rsplit("__", 1)[0] for a in adlar])
    return edge, adlar, bazlar


@torch.no_grad()
def foto_gomme(model, preprocess, foto, device):
    img = foto_edge(load_rgb(str(foto)))
    x = preprocess(img).unsqueeze(0).to(device)
    v = model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v[0].cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = model_yukle(device)
    edge, adlar, bazlar = bank_yukle()

    labels = labels_yukle()
    eslesme = labels.get("eslesme", {})
    if not eslesme:
        print("Etiket yok — değerlendirilecek bir şey yok.")
        return

    print(f"Foto sayısı: {len(eslesme)}")
    K_max = max(args.k)
    recall_say = {k: 0 for k in args.k}
    rrs = []

    for foto, dogru_dwgler in eslesme.items():
        dogru = set(d.replace(".dwg", ".png").lower() if d.endswith(".dwg") else d.lower()
                    for d in dogru_dwgler)
        q = foto_gomme(model, preprocess, PHOTOS / foto, device)
        sims = edge @ q
        # baz başına en yüksek skoru indir
        en_iyi = {}
        for s, b in zip(sims, bazlar):
            if b not in en_iyi or s > en_iyi[b]:
                en_iyi[b] = float(s)
        sirali = sorted(en_iyi.items(), key=lambda kv: -kv[1])
        ust_sirali = [b for b, _ in sirali[:K_max]]

        bulundu_ilk = None
        for sira, b in enumerate(ust_sirali, 1):
            if b.lower() in dogru:
                bulundu_ilk = sira
                break
        if bulundu_ilk is not None:
            rrs.append(1.0 / bulundu_ilk)
            for k in args.k:
                if bulundu_ilk <= k:
                    recall_say[k] += 1
        else:
            rrs.append(0.0)

    n = len(eslesme)
    print(f"\n{'Metrik':<10} {'Değer':>8}")
    for k in args.k:
        print(f"Recall@{k:<3} {recall_say[k]/n*100:>6.1f}%")
    print(f"MRR        {sum(rrs)/n:>8.4f}")


if __name__ == "__main__":
    main()
