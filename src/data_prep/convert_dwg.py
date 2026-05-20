"""
DWG → PNG dönüşümü iki adımda:
  1) DWG → DXF: ODA File Converter (toplu, idempotent)
  2) DXF → PNG: ezdxf + matplotlib (kanonik render: 1024×1024, beyaz zemin siyah çizgi)

Kullanım:
    # Tek dosya hızlı test:
    python -m src.data_prep.convert_dwg --tek data/cad_dwg/100A.dwg

    # Tümünü çevir:
    python -m src.data_prep.convert_dwg

Idempotent: çıktı PNG varsa atlanır.
"""

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Non-interactive backend — Tk dependency yok
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAD_DWG, CAD_PNG, ODA_CONVERTER  # noqa: E402


RENDER_BOYUT = 1024  # kanonik PNG boyutu (eğitim/inference tutarlılığı için sabit)


# ──────────────────────────────────────────────────────────────
# 1) DWG → DXF (ODA)
# ──────────────────────────────────────────────────────────────
def oda_dwg_to_dxf(kaynak_dir: Path, hedef_dir: Path):
    """
    ODA File Converter CLI imzası:
        ODAFileConverter <inputFolder> <outputFolder> <outputVersion>
                         <outputFormat> <recurse> <auditing> [<filter>]
    """
    hedef_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ODA_CONVERTER,
        str(kaynak_dir),
        str(hedef_dir),
        "ACAD2018",   # outputVersion
        "DXF",        # outputFormat
        "0",          # recurse
        "1",          # auditing
        "*.DWG",      # filter
    ]
    print("[ODA] başlıyor:", " ".join(cmd))
    sonuc = subprocess.run(cmd, capture_output=True, text=True)
    if sonuc.returncode != 0:
        print("[ODA] stderr:", sonuc.stderr[:500])
    return sonuc.returncode


# ──────────────────────────────────────────────────────────────
# 2) DXF → PNG (ezdxf + matplotlib)
# ──────────────────────────────────────────────────────────────
def dxf_to_png(dxf_yolu: Path, png_yolu: Path, boyut: int = RENDER_BOYUT) -> bool:
    """
    Tek bir DXF'i PNG'ye render et.
    Kanonik render: SİYAH zemin, BEYAZ çizgi (sabit boyut, kapı bbox'a fit).
    matplotlib renkli çizer; çıktıyı PIL ile grayscale + invert + threshold ile
    siyah-zemin/beyaz-çizgi'ye sıkıştırırız → labeler'da kontur net görünür.
    """
    import ezdxf
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    try:
        doc = ezdxf.readfile(str(dxf_yolu))
    except (IOError, ezdxf.DXFStructureError) as e:
        print(f"[dxf-okuma-hata] {dxf_yolu.name}: {e}")
        return False

    msp = doc.modelspace()

    # Yazıları sil — TEXT/MTEXT/ATTRIB/ATTDEF entity'lerini kaldır.
    # Sebep: foto'da yazı yok (foto_raw alt %12'yi maskeliyor); CAD'de yazı bırakırsak model
    # kapı geometrisi yerine yazıya odaklanır → retrieval ve fine-tune bozulur.
    # DIMENSION'a dokunmuyoruz — kapı ölçü işaretleri geometriye yakın bilgi.
    for ent in list(msp.query("TEXT MTEXT ATTRIB ATTDEF")):
        msp.delete_entity(ent)

    fig, ax = plt.subplots(figsize=(boyut / 100, boyut / 100), dpi=100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    try:
        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)
        # Önce geçici renkli PNG, sonra siyahlaştır
        gecici = png_yolu.with_suffix(".tmp.png")
        fig.savefig(gecici, dpi=100, facecolor="white", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        # Grayscale → invert → threshold (siyah zemin, beyaz çizgi)
        with Image.open(gecici) as im:
            g = im.convert("L")
            inv = ImageOps.invert(g)
            siyah_zemin = inv.point(lambda x: 255 if x > 40 else 0, mode="L")
            siyah_zemin.save(png_yolu, "PNG")
        gecici.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"[render-hata] {dxf_yolu.name}: {e}")
        plt.close(fig)
        return False


# ──────────────────────────────────────────────────────────────
# Worker (paralel)
# ──────────────────────────────────────────────────────────────
def _worker(arg):
    dxf, png = arg
    if Path(png).exists():
        return ("atla", str(dxf))
    ok = dxf_to_png(Path(dxf), Path(png))
    return ("ok" if ok else "bozuk", str(dxf))


def batch_dxf_to_png(dxf_dir: Path, png_dir: Path, paralel: int = 4):
    png_dir.mkdir(parents=True, exist_ok=True)
    isler = []
    for dxf in sorted(dxf_dir.iterdir()):
        if dxf.suffix.lower() != ".dxf":
            continue
        png = png_dir / (dxf.stem + ".png")
        isler.append((str(dxf), str(png)))

    print(f"[render] {len(isler)} DXF → PNG (paralel={paralel})")
    sayim = {"ok": 0, "atla": 0, "bozuk": 0}
    with ProcessPoolExecutor(max_workers=paralel) as ex:
        for fut in as_completed(ex.submit(_worker, j) for j in isler):
            durum, ad = fut.result()
            sayim[durum] += 1
            tplm = sayim["ok"] + sayim["atla"] + sayim["bozuk"]
            if tplm % 100 == 0:
                print(f"  ilerleme: {tplm}/{len(isler)}  ok={sayim['ok']} atla={sayim['atla']} bozuk={sayim['bozuk']}")
    print(f"[render] tamam: ok={sayim['ok']}  atla={sayim['atla']}  bozuk={sayim['bozuk']}")


# ──────────────────────────────────────────────────────────────
# Komut satırı
# ──────────────────────────────────────────────────────────────
def tek_dosya(dwg_yolu: Path):
    """Tek DWG için: geçici DXF → kanonik PNG. Hızlı test amaçlı."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        in_dir = tmp_dir / "in"
        out_dir = tmp_dir / "out"
        in_dir.mkdir()
        # ODA tek dosya değil klasör alır → linkleyelim
        import shutil
        shutil.copy2(dwg_yolu, in_dir / dwg_yolu.name)
        rc = oda_dwg_to_dxf(in_dir, out_dir)
        if rc != 0:
            print(f"[hata] ODA exit={rc}")
            return
        dxf = out_dir / (dwg_yolu.stem + ".dxf")
        if not dxf.exists():
            # ODA bazen büyük/küçük harf değiştirir
            dxf_list = list(out_dir.glob("*.dxf"))
            if not dxf_list:
                print("[hata] DXF üretilmedi")
                return
            dxf = dxf_list[0]
        png = CAD_PNG / (dwg_yolu.stem + ".png")
        CAD_PNG.mkdir(parents=True, exist_ok=True)
        ok = dxf_to_png(dxf, png)
        print(f"{'TAMAM' if ok else 'BOZUK'}: {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tek", help="Tek bir DWG dosyası (hızlı test)")
    ap.add_argument("--dxf-dir", default=None, help="DXF ara klasörü (vermezsek data/cad_dxf)")
    ap.add_argument("--paralel", type=int, default=4)
    ap.add_argument("--skip-oda", action="store_true", help="DWG→DXF adımını atla (zaten yapıldıysa)")
    args = ap.parse_args()

    if args.tek:
        tek_dosya(Path(args.tek))
        return

    dxf_dir = Path(args.dxf_dir) if args.dxf_dir else (CAD_DWG.parent / "cad_dxf")

    if not args.skip_oda:
        dxf_dir.mkdir(parents=True, exist_ok=True)
        rc = oda_dwg_to_dxf(CAD_DWG, dxf_dir)
        if rc != 0:
            print(f"[uyarı] ODA exit={rc}, yine de PNG denemesi yapılacak")

    batch_dxf_to_png(dxf_dir, CAD_PNG, paralel=args.paralel)


if __name__ == "__main__":
    main()
