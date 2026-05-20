"""
Mevcut data/cad_png/*.png dosyalarını siyah-zemin/beyaz-çizgi formatına çevirir.

Grayscale → invert → threshold(>40 = beyaz, ≤40 = siyah).

Idempotent değildir — dönüştürülmüş dosya tekrar dönüştürülürse karışır.
Kontrol için: zaten siyah-zemin olanları atlamak amacıyla "köşe pikseli siyah mı?" diye bakar.

Kullanım:
    python -m src.data_prep.siyahla_pngler
    python -m src.data_prep.siyahla_pngler --paralel 8
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CAD_PNG  # noqa: E402


def _siyah_zemin_mi(p: Path) -> bool:
    """Köşe pikseli siyah ise (parlaklık < 30) zaten dönüştürülmüş varsay."""
    with Image.open(p) as im:
        g = im.convert("L")
        kose = g.getpixel((0, 0))
        return kose < 30


def _islem(p_str: str) -> tuple[str, str]:
    p = Path(p_str)
    try:
        if _siyah_zemin_mi(p):
            return ("atla", p.name)
        with Image.open(p) as im:
            g = im.convert("L")
            inv = ImageOps.invert(g)
            siyah_zemin = inv.point(lambda x: 255 if x > 40 else 0, mode="L")
        siyah_zemin.save(p, "PNG")
        return ("ok", p.name)
    except Exception as e:
        return (f"bozuk:{e}", p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paralel", type=int, default=8)
    args = ap.parse_args()

    if not CAD_PNG.exists():
        print(f"[hata] yok: {CAD_PNG}")
        sys.exit(1)

    pngler = sorted(CAD_PNG.glob("*.png"))
    if not pngler:
        print("PNG yok.")
        return
    print(f"{len(pngler)} PNG işlenecek (paralel={args.paralel})")

    sayim = {"ok": 0, "atla": 0, "bozuk": 0}
    with ProcessPoolExecutor(max_workers=args.paralel) as ex:
        for fut in as_completed(ex.submit(_islem, str(p)) for p in pngler):
            durum, ad = fut.result()
            if durum.startswith("bozuk"):
                sayim["bozuk"] += 1
                print(f"  [bozuk] {ad}: {durum}")
            else:
                sayim[durum] += 1
            tplm = sum(sayim.values())
            if tplm % 1000 == 0:
                print(f"  ilerleme: {tplm}/{len(pngler)}  ok={sayim['ok']} atla={sayim['atla']} bozuk={sayim['bozuk']}")
    print(f"\nBitti: ok={sayim['ok']}  atla={sayim['atla']}  bozuk={sayim['bozuk']}")


if __name__ == "__main__":
    main()
