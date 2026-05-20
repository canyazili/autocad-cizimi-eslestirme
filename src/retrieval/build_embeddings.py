"""
Tüm CAD PNG'leri 4 rotasyonla raw + edge gömme bankası üretir.

Çıktı: models/embeddings.npz
    dosya_adlari: (N,) '<dwg_stem>.png__<rotasyon>'
    raw_embeddings:  (N, 512) float32 (L2 normalize)
    edge_embeddings: (N, 512) float32 (L2 normalize)

Kullanım:
    python -m src.retrieval.build_embeddings
"""

import sys
import functools
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# stdout buffer'ı kapat → ilerleme anında diske yazılsın
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CAD_PNG, EMBEDDINGS, MODEL_BEST, CLIP_BACKBONE, CLIP_PRETRAINED,
)
from utils import cad_raw, cad_edge  # noqa: E402


ROTASYONLAR = (0, 90, 180, 270)
BATCH = 64


def model_yukle(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_PRETRAINED)
    if Path(MODEL_BEST).exists():
        print(f"[init] {MODEL_BEST} yükleniyor")
        ckpt = torch.load(MODEL_BEST, map_location="cpu", weights_only=False)
        state = {k.replace("clip_model.", "", 1) if k.startswith("clip_model.") else k: v
                 for k, v in ckpt.items()}
        model.load_state_dict(state, strict=False)
    return model.to(device).eval(), preprocess


@torch.no_grad()
def batch_embed(model, preprocess, pil_listesi, device):
    x = torch.stack([preprocess(im) for im in pil_listesi]).to(device)
    v = model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = model_yukle(device)

    png_listesi = sorted(p for p in CAD_PNG.glob("*.png") if p.is_file())
    if not png_listesi:
        print(f"PNG yok: {CAD_PNG} — önce convert_dwg.py çalıştır.")
        return
    print(f"{len(png_listesi)} PNG × {len(ROTASYONLAR)} rotasyon × 2 kanal (raw, edge)")

    adlar = []
    raw_list = []
    edge_list = []

    tampon_pil_raw = []
    tampon_pil_edge = []
    tampon_ad = []

    def bosalt():
        if not tampon_ad:
            return
        re_arr = batch_embed(model, preprocess, tampon_pil_raw, device)
        ed_arr = batch_embed(model, preprocess, tampon_pil_edge, device)
        adlar.extend(tampon_ad)
        raw_list.append(re_arr)
        edge_list.append(ed_arr)
        tampon_pil_raw.clear()
        tampon_pil_edge.clear()
        tampon_ad.clear()

    for i, png in enumerate(png_listesi, 1):
        try:
            with Image.open(png) as im:
                im = im.convert("RGB")
        except Exception as e:
            print(f"[atla] {png.name}: {e}")
            continue
        for rot in ROTASYONLAR:
            rot_im = im if rot == 0 else im.rotate(rot, expand=True)
            tampon_pil_raw.append(cad_raw(rot_im))
            tampon_pil_edge.append(cad_edge(rot_im))
            tampon_ad.append(f"{png.name}__{rot}")
            if len(tampon_ad) >= BATCH:
                bosalt()
        if i % 100 == 0 or i == len(png_listesi):
            print(f"  {i}/{len(png_listesi)}")
    bosalt()

    raw = np.concatenate(raw_list, axis=0)
    edge = np.concatenate(edge_list, axis=0)
    adlar_arr = np.array(adlar)

    Path(EMBEDDINGS).parent.mkdir(parents=True, exist_ok=True)
    # Mevcut embeddings dosyasını yedekle
    if Path(EMBEDDINGS).exists():
        yedek = Path(EMBEDDINGS).with_suffix(".npz.backup")
        Path(EMBEDDINGS).replace(yedek)
        print(f"eski embeddings yedeklendi: {yedek}")

    np.savez_compressed(EMBEDDINGS,
                        raw_embeddings=raw, edge_embeddings=edge, dosya_adlari=adlar_arr)
    print(f"yazıldı: {EMBEDDINGS}  |  {raw.shape}, {edge.shape}, {adlar_arr.shape}")


if __name__ == "__main__":
    main()
