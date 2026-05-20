"""
data/tüm modeller/ → data/photos/ ingest.

- Foto adlarından örtük meta bilgi çıkarır (foto_meta.json).
- Adlar GÜVENİLİR ETİKET DEĞİL; sadece UI'da iddia edilen-no göstermek için.
- Dosyaları photos/ altına kopyalar (idempotent; varsa atlar).

Kullanım:
    python -m src.data_prep.ingest_photos
"""

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PHOTOS_RAW, PHOTOS, FOTO_META_FILE  # noqa: E402
from utils import SUPPORTED_EXTS  # noqa: E402


# İki dosya-ad şeması destekleniyor:
#   (a) sayı-baş: <TABAN><AD?><ALT_NO?>(-<VARYANT>)?     örn. 1008AKEL-1, 1008BAKLAVA2, 1001, 1072-3GÖBEK-1
#   (b) ad-baş:   <AD><ALT_NO?>(-<VARYANT>)?              örn. AKELKADIR09, AFERİN02-1, ABSGÜZEL01
# "GİBİ" her ikisinde de yumuşak benzerlik işaretçisidir.
#
# DİKKAT: çıkardığımız meta GÜVENİLİR ETİKET DEĞİL — sadece UI'da "iddia edilen no/ad" göstermek için.

GIBI_RE   = re.compile(r"GİBİ|GIBI", flags=re.IGNORECASE)
HARF      = r"[A-ZÇĞİÖŞÜa-zçğıöşü]"
SAYI_BAS  = re.compile(rf"^(?P<taban>\d+(?:-\d+)?)(?P<ad>{HARF}*?)(?P<alt>\d+)?(?:-(?P<vary>\d+))?$")
AD_BAS    = re.compile(rf"^(?P<ad>{HARF}+)(?P<alt>\d+)?(?:-(?P<vary>\d+))?$")


def isim_parse(stem: str) -> dict:
    yumusak = bool(GIBI_RE.search(stem))
    s = GIBI_RE.sub("", stem)

    m = SAYI_BAS.match(s)
    if m:
        return {
            "tip": "sayi_bas",
            "taban":   m.group("taban") or "",
            "ad":      (m.group("ad") or "").strip(),
            "alt_no":  m.group("alt") or "",
            "varyant": m.group("vary") or "",
            "yumusak": yumusak,
            "ham": stem,
        }

    m = AD_BAS.match(s)
    if m:
        return {
            "tip": "ad_bas",
            "taban":   "",
            "ad":      m.group("ad"),
            "alt_no":  m.group("alt") or "",
            "varyant": m.group("vary") or "",
            "yumusak": yumusak,
            "ham": stem,
        }

    return {"tip": "bilinmiyor", "taban": "", "ad": "", "alt_no": "", "varyant": "", "yumusak": yumusak, "ham": stem}


def ingest():
    PHOTOS.mkdir(parents=True, exist_ok=True)
    Path(FOTO_META_FILE).parent.mkdir(parents=True, exist_ok=True)

    if not PHOTOS_RAW.exists():
        print(f"[HATA] Kaynak yok: {PHOTOS_RAW}")
        sys.exit(1)

    meta = {}
    kopya = 0
    atla = 0
    sayim = {"sayi_bas": 0, "ad_bas": 0, "bilinmiyor": 0}

    for src in sorted(PHOTOS_RAW.iterdir()):
        if not src.is_file():
            continue
        if not src.name.lower().endswith(SUPPORTED_EXTS):
            continue

        hedef = PHOTOS / src.name
        if not hedef.exists():
            shutil.copy2(src, hedef)
            kopya += 1
        else:
            atla += 1

        info = isim_parse(src.stem)
        sayim[info["tip"]] = sayim.get(info["tip"], 0) + 1
        meta[src.name] = info

    with open(FOTO_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Toplam: {len(meta)}  |  kopyalandı: {kopya}  |  atlandı: {atla}")
    print(f"  sayı-baş: {sayim['sayi_bas']}  |  ad-baş: {sayim['ad_bas']}  |  bilinmiyor: {sayim['bilinmiyor']}")
    print(f"Meta yazıldı: {FOTO_META_FILE}")


if __name__ == "__main__":
    ingest()
