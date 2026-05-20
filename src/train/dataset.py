"""
labels.json'dan (foto, dwg) çiftleri okuyan PyTorch dataset.

Beklenen veri:
  - data/photos/<foto>.jpg/png       (ham)
  - data/cad_png/<dwg>.png            (convert_dwg çıktısı)
  - labels.json["eslesme"]: {foto: [dwg, ...]}

Pozitif örnek: bir foto + bağlı DWG'lerinden biri (rastgele).
Negatif örnekler InfoNCE'de aynı batch'in diğer örneklerinden gelir.

`aile_negatif=True` ise pozitif DWG'nin aynı taban-numarasına sahip ama eşlemede
olmayan bir DWG batch'e zorla eklenir → hard negative.
"""

import json
import random
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PHOTOS, CAD_PNG, LABELS_FILE  # noqa: E402
from utils import load_rgb, foto_edge, cad_edge  # noqa: E402


TABAN_RE = re.compile(r"^(\d+(?:-\d+)?|[A-Za-zÇĞİÖŞÜçğıöşü]+)")


def _taban(ad: str) -> str:
    m = TABAN_RE.match(Path(ad).stem)
    return m.group(0).lower() if m else ""


class CiftDataset(Dataset):
    """
    Her __getitem__ → (foto_edge_tensor, cad_edge_tensor[, hard_neg_cad_tensor])
    preprocess: open_clip image transform.
    """
    def __init__(self, preprocess, aile_negatif: bool = True,
                 labels_yolu: str = LABELS_FILE):
        with open(labels_yolu, encoding="utf-8") as f:
            d = json.load(f)
        eslesme = d.get("eslesme", {})

        self.ciftler = []  # (foto_name, dwg_name)
        for foto, dwgler in eslesme.items():
            for dwg in dwgler:
                self.ciftler.append((foto, dwg))

        # taban → tüm DWG isimleri (hard negative seçimi için)
        tum_dwg = [p.name for p in CAD_PNG.glob("*.png")]
        self.taban_dwg: dict[str, list[str]] = {}
        for dwg in tum_dwg:
            self.taban_dwg.setdefault(_taban(dwg), []).append(dwg)

        self.preprocess = preprocess
        self.aile_negatif = aile_negatif
        self.eslesme = eslesme  # foto → [dwg, ...]

    def __len__(self):
        return len(self.ciftler)

    def _yukle_foto(self, foto_ad: str):
        img = foto_edge(load_rgb(str(PHOTOS / foto_ad)))
        return self.preprocess(img)

    def _yukle_cad(self, dwg_ad: str):
        with Image.open(CAD_PNG / dwg_ad) as img:
            edge = cad_edge(img.convert("RGB"))
        return self.preprocess(edge)

    def __getitem__(self, idx):
        foto, dwg_poz = self.ciftler[idx]
        try:
            foto_x = self._yukle_foto(foto)
            poz_x = self._yukle_cad(dwg_poz)
        except (FileNotFoundError, OSError) as e:
            # Eksik dosya: rastgele başka bir çifte düş
            return self.__getitem__((idx + 1) % len(self.ciftler))

        if not self.aile_negatif:
            return foto_x, poz_x

        # Hard negative: aynı taban_no'lu ama bu foto için eşleşmeyen DWG seç
        taban = _taban(dwg_poz)
        adaylar = [d for d in self.taban_dwg.get(taban, [])
                   if d != dwg_poz and d not in self.eslesme.get(foto, [])]
        if not adaylar:
            # fallback: tamamen rastgele DWG
            taban_listesi = list(self.taban_dwg.keys())
            taban_listesi.remove(taban) if taban in taban_listesi else None
            if taban_listesi:
                adaylar = self.taban_dwg[random.choice(taban_listesi)]
            else:
                # Hard negative yok — sadece pozitif çift döndür
                return foto_x, poz_x

        neg_dwg = random.choice(adaylar)
        try:
            neg_x = self._yukle_cad(neg_dwg)
        except (FileNotFoundError, OSError):
            return foto_x, poz_x
        return foto_x, poz_x, neg_x


def collate_ucl(batch):
    """3-elemanlı (foto, poz, neg) ile 2-elemanlı (foto, poz) örnekleri dengeli batch'le."""
    foto = torch.stack([b[0] for b in batch])
    poz  = torch.stack([b[1] for b in batch])
    if len(batch[0]) == 3:
        neg = torch.stack([b[2] for b in batch])
        return foto, poz, neg
    return foto, poz
