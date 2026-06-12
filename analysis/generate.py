import os, os.path as osp, sys, time, json, argparse
import numpy as np
import torch
import PIL.Image as PImage

VAR_DIR  = 'VAR'
CKPT_DIR = 'checkpoints'
WORK_DIR = 'analysis'
sys.path.insert(0, VAR_DIR)
sys.path.insert(0, WORK_DIR)


def _expand(s):
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_classes(s):
    return list(range(1000)) if s == 'all' else _expand(s)


def parse_sel(s):
    return 'all' if s == 'all' else _expand(s)


def parse_scale_bits(s):
    if not s:
        return None
    d = {}
    for part in s.split(','):
        rng, bits = part.split(':')
        b = int(bits)
        for si in _expand(rng):
            d[si] = (b, b)
    return d


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--depth', type=int, default=16, choices=[16, 20, 24, 30])
    p.add_argument('--cfg', type=float, default=1.5)
    p.add_argument('--top_k', type=int, default=900)
    p.add_argument('--top_p', type=float, default=0.96)
    p.add_argument('--classes', type=str, default='all')
    p.add_argument('--n_per_class', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--layout', type=str, default='flat', choices=['flat', 'perclass'])
    p.add_argument('--more_smooth', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--kv_k_bits', type=int, default=16)
    p.add_argument('--kv_v_bits', type=int, default=16)
    p.add_argument('--kv_per', type=str, default='token', choices=['token', 'channel'])
    p.add_argument('--kv_per_k', type=str, default='', choices=['', 'token', 'channel'])
    p.add_argument('--kv_per_v', type=str, default='', choices=['', 'token', 'channel'])
    p.add_argument('--kv_sym', action='store_true')
    p.add_argument('--kv_layers', type=str, default='all')
    p.add_argument('--kv_scales', type=str, default='all')
    p.add_argument('--kv_scale_bits', type=str, default='')
    return p.parse_args()


def main():
    args = build_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setattr(torch.nn.Linear,    'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    from models import build_vae_var

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae_ckpt = osp.join(CKPT_DIR, 'vae_ch160v4096z32.pth')
    var_ckpt = osp.join(CKPT_DIR, f'var_d{args.depth}.pth')
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, device=device,
        patch_nums=patch_nums, num_classes=1000, depth=args.depth, shared_aln=False,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(); var.eval()
    for q in vae.parameters(): q.requires_grad_(False)
    for q in var.parameters(): q.requires_grad_(False)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    kv_meta = None
    scale_bits = parse_scale_bits(args.kv_scale_bits)
    if args.kv_k_bits < 16 or args.kv_v_bits < 16 or scale_bits is not None:
        from kv_compression.quant import KVQConfig, apply_kv_quant
        kvcfg = KVQConfig(k_bits=args.kv_k_bits, v_bits=args.kv_v_bits, per=args.kv_per,
                          per_k=args.kv_per_k or None, per_v=args.kv_per_v or None,
                          sym=args.kv_sym, layers=parse_sel(args.kv_layers),
                          scales=parse_sel(args.kv_scales), scale_bits=scale_bits,
                          patch_nums=patch_nums)
        apply_kv_quant(var, kvcfg)
        kv_meta = kvcfg.as_dict()
        print('KV-quant active:', json.dumps(kv_meta), flush=True)

    evict_meta = None
    es = os.environ.get('EVICT_STRIDE')
    if es and int(es) > 1:
        from kv_compression.evict import EvictConfig, apply_evict
        ecfg = EvictConfig(int(es), int(os.environ.get('EVICT_LATE_START', '5')), patch_nums)
        apply_evict(var, ecfg)
        evict_meta = ecfg.as_dict()
        print('Evict active:', json.dumps(evict_meta), flush=True)

    classes = parse_classes(args.classes)
    os.makedirs(args.out_dir, exist_ok=True)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); total = 0; stop = False

    for ci in classes:
        if stop:
            break
        if args.layout == 'perclass':
            cdir = osp.join(args.out_dir, f'{ci:04d}'); os.makedirs(cdir, exist_ok=True)
        n_done = 0
        while n_done < args.n_per_class:
            b = min(args.batch_size, args.n_per_class - n_done)
            g_seed = args.seed * 1_000_000 + ci * 1000 + n_done
            label_B = torch.full((b,), ci, dtype=torch.long, device=device)
            with torch.inference_mode():
                with torch.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16, cache_enabled=True):
                    imgs = var.autoregressive_infer_cfg(
                        B=b, label_B=label_B, cfg=args.cfg, top_k=args.top_k,
                        top_p=args.top_p, g_seed=g_seed, more_smooth=args.more_smooth)
            arr = imgs.permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            for j in range(b):
                idx = n_done + j
                if args.layout == 'perclass':
                    path = osp.join(cdir, f'{idx:03d}.png')
                else:
                    path = osp.join(args.out_dir, f'{ci:04d}_{idx:03d}.png')
                PImage.fromarray(arr[j]).save(path)
                total += 1
                if args.limit and total >= args.limit:
                    stop = True; break
            n_done += b
            if stop:
                break

    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == 'cuda' else 0.0
    meta = dict(depth=args.depth, cfg=args.cfg, top_k=args.top_k, top_p=args.top_p,
                seed=args.seed, n_per_class=args.n_per_class, n_classes=len(classes),
                total_images=total, seconds=round(dt, 2),
                imgs_per_sec=round(total / max(dt, 1e-9), 2),
                peak_gpu_gb=round(peak, 3), layout=args.layout, more_smooth=args.more_smooth,
                kv_quant=kv_meta, evict=evict_meta)
    with open(osp.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print('GEN_DONE')


if __name__ == '__main__':
    main()
