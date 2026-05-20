"""
labels.json bakım scripti — istatistik, çift kontrolü, boş kayıt temizliği.

Komutlar:
    python -m src.labeling.label_manager stats              # özet
    python -m src.labeling.label_manager dedup              # eşlemelerdeki tekrarları temizle
    python -m src.labeling.label_manager bosalt-tamamlanan  # eşlemesi olmayan tamamlananları kaldır (DİKKAT)
    python -m src.labeling.label_manager export-train data/labels/train_pairs.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Windows konsol UTF-8 (Türkçe + ok karakterleri için)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LABELS_FILE  # noqa: E402
from utils import labels_yukle, labels_kaydet  # noqa: E402


def stats():
    d = labels_yukle()
    tamamlanan = d.get("tamamlanan", [])
    eslesme = d.get("eslesme", {})
    print(f"Tamamlanan foto:        {len(tamamlanan)}")
    print(f"Eşleşmesi olan foto:    {len(eslesme)}")
    print(f"Boş tamamlanan:         {len(tamamlanan) - len(eslesme)}")
    toplam_cift = sum(len(v) for v in eslesme.values())
    print(f"Toplam (foto,DWG) çifti: {toplam_cift}")
    if eslesme:
        per_foto = Counter(len(v) for v in eslesme.values())
        print(f"Foto başına DWG dağılımı (sayı→foto):")
        for k in sorted(per_foto):
            print(f"  {k:>3} DWG: {per_foto[k]} foto")
    # En çok geçen DWG'ler
    dwg_say = Counter()
    for v in eslesme.values():
        dwg_say.update(v)
    print(f"\nEn popüler 10 DWG eşleşmesi:")
    for d, n in dwg_say.most_common(10):
        print(f"  {n:>3}× {d}")


def dedup():
    d = labels_yukle()
    eslesme = d.get("eslesme", {})
    degisti = 0
    for foto in list(eslesme):
        once = eslesme[foto]
        sonra = list(dict.fromkeys(once))  # sıra-koruyan uniq
        if len(sonra) != len(once):
            eslesme[foto] = sonra
            degisti += 1
    if degisti:
        # labels_kaydet eşleşmeleri merge ediyor — burada doğrudan yazmamız gerek
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"{degisti} foto'da yinelenen DWG temizlendi.")
    else:
        print("Yinelenen DWG yok.")


def bosalt_tamamlanan():
    d = labels_yukle()
    tamamlanan = d.get("tamamlanan", [])
    eslesme = d.get("eslesme", {})
    yeni = [f for f in tamamlanan if f in eslesme]
    silinen = len(tamamlanan) - len(yeni)
    if silinen:
        d["tamamlanan"] = yeni
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"{silinen} eşlemesiz tamamlanan kaldırıldı.")
    else:
        print("Eşlemesiz tamamlanan yok.")


def export_train(out_path: str):
    """Eğitim verisi için düz (foto, dwg) çift listesi yaz."""
    d = labels_yukle()
    ciftler = []
    for foto, dwgler in d.get("eslesme", {}).items():
        for dwg in dwgler:
            ciftler.append({"foto": foto, "dwg": dwg})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ciftler, f, ensure_ascii=False, indent=2)
    print(f"{len(ciftler)} eğitim çifti → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    sub.add_parser("dedup")
    sub.add_parser("bosalt-tamamlanan")
    ep = sub.add_parser("export-train")
    ep.add_argument("out")
    args = ap.parse_args()

    {
        "stats": stats,
        "dedup": dedup,
        "bosalt-tamamlanan": bosalt_tamamlanan,
        "export-train": lambda: export_train(args.out),
    }[args.cmd]()


if __name__ == "__main__":
    main()
