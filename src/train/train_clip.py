"""
Fine-tune CLIP (image-image contrastive, InfoNCE) — foto edge ↔ CAD edge.

Kullanım:
    python -m src.train.train_clip --epochs 10 --batch 64 --lr 1e-5

Çıktı:
    finetune_output/clip_finetuned_last.pt
    finetune_output/clip_finetuned_best.pt   (val loss en iyi)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    MODEL_BEST, FINETUNE_OUTPUT, MODEL_BEST_NAME, MODEL_LAST_NAME,
    CLIP_BACKBONE, CLIP_PRETRAINED,
)
from train.dataset import CiftDataset, collate_ucl  # noqa: E402


def model_yukle(device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_PRETRAINED)
    if Path(MODEL_BEST).exists():
        print(f"[init] {MODEL_BEST} yükleniyor")
        ckpt = torch.load(MODEL_BEST, map_location="cpu", weights_only=False)
        state = {k.replace("clip_model.", "", 1) if k.startswith("clip_model.") else k: v
                 for k, v in ckpt.items()}
        model.load_state_dict(state, strict=False)
    else:
        print("[init] mevcut .pt yok, OpenAI pretrained ağırlıkları kullanılacak")
    return model.to(device), preprocess


def encode_norm(model, x):
    v = model.encode_image(x)
    return v / v.norm(dim=-1, keepdim=True)


def info_nce(foto_emb, cad_emb, temp: float = 0.07):
    """
    Simetrik InfoNCE:
      logits = (foto · cadᵀ) / temp
      kayıp = 0.5 * (CE(rows, diag) + CE(cols, diag))
    """
    logits = (foto_emb @ cad_emb.T) / temp
    hedef = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, hedef) + F.cross_entropy(logits.T, hedef))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--val-orani", type=float, default=0.1)
    ap.add_argument("--isci", type=int, default=4)
    ap.add_argument("--cikti", default=str(FINETUNE_OUTPUT))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, preprocess = model_yukle(device)
    veri = CiftDataset(preprocess, aile_negatif=True)
    print(f"toplam çift: {len(veri)}")
    if len(veri) < 10:
        print("yetersiz veri (en az ~50 etiket lazım)")
        sys.exit(1)

    val_n = max(1, int(len(veri) * args.val_orani))
    train_n = len(veri) - val_n
    train_ds, val_ds = random_split(veri, [train_n, val_n], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_ucl, num_workers=args.isci, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate_ucl, num_workers=args.isci)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    cikti_dir = Path(args.cikti)
    cikti_dir.mkdir(parents=True, exist_ok=True)
    en_iyi_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        toplam_kayip = 0.0
        n = 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                if len(batch) == 3:
                    foto, poz, neg = (b.to(device, non_blocking=True) for b in batch)
                    fe = encode_norm(model, foto)
                    pe = encode_norm(model, poz)
                    ne = encode_norm(model, neg)
                    # foto ↔ poz contrast; ek olarak neg'leri "in-batch" negatif olarak ekle
                    cad_emb = torch.cat([pe, ne], dim=0)
                    foto_ext = fe  # hedef: foto i'ye karşılık index i olan poz
                    logits = (foto_ext @ cad_emb.T) / args.temp
                    hedef = torch.arange(fe.size(0), device=device)
                    kayip = F.cross_entropy(logits, hedef)
                else:
                    foto, poz = (b.to(device, non_blocking=True) for b in batch)
                    fe = encode_norm(model, foto)
                    pe = encode_norm(model, poz)
                    kayip = info_nce(fe, pe, temp=args.temp)
            scaler.scale(kayip).backward()
            scaler.step(opt)
            scaler.update()
            toplam_kayip += kayip.item() * foto.size(0)
            n += foto.size(0)
        ort_train = toplam_kayip / max(1, n)

        # Validation
        model.eval()
        v_top = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    foto, poz, _ = (b.to(device) for b in batch)
                else:
                    foto, poz = (b.to(device) for b in batch)
                fe = encode_norm(model, foto)
                pe = encode_norm(model, poz)
                kayip = info_nce(fe, pe, temp=args.temp)
                v_top += kayip.item() * foto.size(0)
                v_n += foto.size(0)
        ort_val = v_top / max(1, v_n)
        print(f"epoch {epoch:>2}: train={ort_train:.4f}  val={ort_val:.4f}")

        # Kayıt — wrapper formatına geri sar (clip_model. öneki)
        son_state = {f"clip_model.{k}": v for k, v in model.state_dict().items()}
        torch.save(son_state, cikti_dir / MODEL_LAST_NAME)
        if ort_val < en_iyi_val:
            en_iyi_val = ort_val
            torch.save(son_state, cikti_dir / MODEL_BEST_NAME)
            print(f"  → best güncellendi")


if __name__ == "__main__":
    main()
