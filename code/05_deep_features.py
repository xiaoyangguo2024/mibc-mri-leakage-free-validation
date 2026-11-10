"""05 - Frozen deep features (no training, label-free) from three pretrained backbones:
ImageNet ResNet-50 (supervised CNN), DINOv2 ViT-B/14 (self-supervised ViT, natural images),
BiomedCLIP ViT-B/16 (biomedical image-text). Weights are local files pinned in env/weights/MANIFEST.tsv.

Every tumour slice is embedded; tumour-level features = mean over the 5 largest tumour slices.
Outputs: data/processed/deep_{name}_slices.npz (slice-level) and deep_{name}.csv (tumour-level).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bcmri import config as C

W = C.WEIGHTS
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def build(name):
    import timm
    if name == "ResNet50_IN":
        m = timm.create_model("resnet50.tv2_in1k", pretrained=True, num_classes=0,
                              pretrained_cfg_overlay=dict(file=str(W / "timm_resnet50.tv2_in1k" / "model.safetensors")))
        return m, IMNET, 224
    if name == "DINOv2_B":
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224,
                              pretrained_cfg_overlay=dict(file=str(W / "timm_vit_base_patch14_dinov2.lvd142m" / "model.safetensors")))
        return m, IMNET, 224
    if name == "BiomedCLIP":
        from open_clip.model import _build_vision_tower
        d = W / "microsoft_BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        cfg = json.loads((d / "open_clip_config.json").read_text())
        mc = cfg["model_cfg"]
        visual = _build_vision_tower(mc["embed_dim"], mc["vision_cfg"])
        sd = torch.load(d / "open_clip_pytorch_model.bin", map_location="cpu", weights_only=True)
        vsd = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
        res = visual.load_state_dict(vsd, strict=True)
        pp = cfg.get("preprocess_cfg", {})
        return visual, (pp.get("mean", IMNET[0]), pp.get("std", IMNET[1])), mc["vision_cfg"].get("image_size", 224)
    raise ValueError(name)


@torch.no_grad()
def embed(model, norm, size, imgs, device, bs=32):
    mean = torch.tensor(norm[0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(norm[1], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(device).float() / 255.0
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        out.append(model((x - mean) / std).float().cpu().numpy())
    return np.concatenate(out)


def main():
    names = sys.argv[1:] or ["ResNet50_IN", "DINOv2_B", "BiomedCLIP"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    df = pd.read_csv(C.PROCESSED / "cohort.csv")
    idx = pd.read_csv(C.PROCESSED / "crops_index.csv")
    for name in names:
        t0 = time.time()
        model, norm, size = build(name)
        model.eval().to(device)
        feats, keys = [], []
        for cid in df["case_id"]:
            z = np.load(C.INTERIM / "crops" / f"{cid}.npz")
            f = embed(model, norm, size, z["img"], device)
            feats.append(f)
            keys += [(cid, i) for i in range(len(f))]
        S = np.concatenate(feats).astype(np.float32)
        kdf = pd.DataFrame(keys, columns=["case_id", "slice_idx"]).merge(idx, on=["case_id", "slice_idx"], how="left")
        np.savez_compressed(C.PROCESSED / f"deep_{name}_slices.npz", feats=S,
                            case_id=kdf["case_id"].values, slice_idx=kdf["slice_idx"].values,
                            area_rank=kdf["area_rank"].values)
        top = (kdf["area_rank"] < C.TOPK_SLICES).values
        agg = pd.DataFrame(S[top]).groupby(kdf.loc[top, "case_id"].values).mean()
        agg = agg.reindex(df["case_id"])
        agg.columns = [f"{name}_{j}" for j in range(agg.shape[1])]
        agg.index.name = "case_id"
        agg.to_csv(C.PROCESSED / f"deep_{name}.csv")
        print(f"{name}: device {device}, slices {S.shape}, tumour-level {agg.shape}, {time.time() - t0:.0f}s", flush=True)
        del model


if __name__ == "__main__":
    main()
