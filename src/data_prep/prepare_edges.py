"""
PNG klasörünü sketch dönüşümlü ikinci bir klasöre çevirir.

Kullanım:
    python -m src.data_prep.prepare_edges --target cad
    python -m src.data_prep.prepare_edges --target photos

Idempotent: çıktısı varsa atlar.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAD_PNG, CAD_PNG_EDGE, PHOTOS, PHOTOS_EDGE  # noqa: E402
from utils import cad_edge, foto_edge, SUPPORTED_EXTS  # noqa: E402


HEDEF = {
    "cad":    (CAD_PNG,    CAD_PNG_EDGE,  cad_edge),
    "photos": (PHOTOS,     PHOTOS_EDGE,   foto_edge),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(HEDEF.keys()))
    args = ap.parse_args()

    kaynak, cikti, fn = HEDEF[args.target]
    if not kaynak.exists():
        print(f"[HATA] kaynak yok: {kaynak}")
        sys.exit(1)
    cikti.mkdir(parents=True, exist_ok=True)

    ok = atla = bozuk = 0
    for src in sorted(kaynak.iterdir()):
        if not src.is_file() or not src.name.lower().endswith(SUPPORTED_EXTS):
            continue
        dst = cikti / (src.stem + ".png")
        if dst.exists():
            atla += 1
            continue
        try:
            with Image.open(src) as img:
                edge = fn(img.convert("RGB"))
                edge.save(dst, "PNG")
            ok += 1
        except Exception as e:
            bozuk += 1
            print(f"[bozuk] {src.name}: {e}")

    print(f"[{args.target}] yazıldı: {ok}  |  atlandı: {atla}  |  bozuk: {bozuk}")


if __name__ == "__main__":
    main()
