"""Load VAR-d{depth} given a local checkout of the VAR repo and its checkpoints.

`scripts/prepare.sh` clones the VAR code into ./VAR and downloads the checkpoints into
./checkpoints, so the defaults below work out of the box.
"""
import os.path as osp
import sys
import torch


def add_var_repo(var_repo):
    p = osp.abspath(var_repo)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_var(var_repo="VAR", ckpt_dir="checkpoints", device="cuda", depth=16):
    """Return (vae, var, patch_nums) for VAR-d{depth}, weights loaded and frozen."""
    add_var_repo(var_repo)
    # VAR re-initialises weights in __init__; skip it so loading is fast and deterministic.
    torch.nn.Linear.reset_parameters = lambda self: None
    torch.nn.LayerNorm.reset_parameters = lambda self: None
    from models import build_vae_var

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, device=device,
        patch_nums=patch_nums, num_classes=1000, depth=depth, shared_aln=False,
    )
    vae.load_state_dict(torch.load(osp.join(ckpt_dir, "vae_ch160v4096z32.pth"), map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(osp.join(ckpt_dir, f"var_d{depth}.pth"), map_location="cpu"), strict=True)
    vae.eval()
    var.eval()
    for p in var.parameters():
        p.requires_grad_(False)
    return vae, var, patch_nums
