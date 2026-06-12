"""Load Infinity-2B once, sweep KV-quant configs (FP16 / uniform & KIVI x INT3/INT2),
generate the same prompts under each, save per-config dirs, report analytical KV memory.
Run with: CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 <infinity-venv>/bin/python analysis/run_infinity_all.py
"""
import os, os.path as osp, sys, json, time
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
import numpy as np
import torch

ROOT = '.'
INF = osp.join(ROOT, 'Infinity')
sys.path.insert(0, INF); sys.path.insert(0, osp.join(INF, 'tools')); sys.path.insert(0, osp.join(ROOT, 'analysis'))
import cv2
from run_infinity import (load_tokenizer, load_visual_tokenizer, load_transformer,
                          gen_one_img, dynamic_resolution_h_w, h_div_w_templates)
import argparse
from kv_compression import inf_quant

PROMPTS = {
    "explore_more": "Create an image with 'Explore More' in an adventurous font over a picturesque hiking trail.",
    "corgi_dog": "A close-up photograph of a Corgi dog wearing a black hat and round dark sunglasses, joyful expression, tongue out.",
    "cat_fashion": "Hyperrealistic black and white photography of cats fashion show in style of Helmut Newton.",
    "toy_car": "Close-up shot of a diecast toy car, diorama, night, lights from windows, bokeh, snow.",
    "perfume": "Product photography, a perfume placed on a white marble table with pineapple, coconut, lime, white curtains, intricate details, realistic.",
    "miniature_village": "An enchanted miniature village bustling with activity, featuring tiny houses, markets, and residents.",
    "robot_eggplant": "a robot holding a huge eggplant, sunny nature background",
    "fairy_house": "House: white; pink tinted windows; surrounded by flowers; cute; scenic; garden; fairy-like; photorealistic; insanely detailed.",
}
CONFIGS = [
    dict(name='fp16',      bits=16, perk='token',   perv='token'),
    dict(name='uni_int3',  bits=3,  perk='token',   perv='token'),
    dict(name='kivi_int3', bits=3,  perk='channel', perv='token'),
    dict(name='uni_int2',  bits=2,  perk='token',   perv='token'),
    dict(name='kivi_int2', bits=2,  perk='channel', perv='token'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--tau', type=float, default=0.5)
    a = ap.parse_args()

    args = argparse.Namespace(
        pn='1M', model_path=osp.join(ROOT, 'checkpoints/infinity/infinity_2b_reg.pth'),
        cfg_insertion_layer=0, vae_type=32,
        vae_path=osp.join(ROOT, 'checkpoints/infinity/infinity_vae_d32reg.pth'),
        add_lvl_embeding_only_first_block=1, use_bit_label=1, model_type='infinity_2b',
        rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2, use_scale_schedule_embedding=0,
        sampling_per_bits=1, text_encoder_ckpt=osp.join(ROOT, 'checkpoints/flan-t5-xl'),
        text_channels=2048, apply_spatial_patchify=0, h_div_w_template=1.000, use_flex_attn=0,
        cache_dir='/dev/shm', checkpoint_type='torch', seed=0, bf16=1, save_file='tmp.jpg',
        enable_model_cache=0,
    )
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    h_div_w = 1.0
    tmpl = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[tmpl][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # analytical KV memory (depth x 2(k,v) x tokens x embed_dim x bits/8), batch=1
    depth = 32; embed_dim = 2048
    total_tokens = int(sum(h * w for (_, h, w) in scale_schedule))
    def kv_gb(bits, B=1):
        return round(2 * depth * total_tokens * embed_dim * (bits / 8.0) * B / 1e9, 3)
    memtab = {c['name']: kv_gb(c['bits']) for c in CONFIGS}
    print('scale_schedule tokens=%d, KV-mem(GB,B=1): %s' % (total_tokens, json.dumps(memtab)), flush=True)

    items = list(PROMPTS.items())[:a.n]
    summary = dict(total_tokens=total_tokens, kv_mem_gb=memtab, configs=[])
    for c in CONFIGS:
        out = osp.join(ROOT, 'results/infinity', c['name'])
        os.makedirs(out, exist_ok=True)
        if c['bits'] >= 16:
            inf_quant.remove_inf_quant(infinity)
            qmeta = dict(bits=16)
        else:
            os.environ['INFQ_BITS'] = str(c['bits'])
            os.environ['INFQ_PERK'] = c['perk']; os.environ['INFQ_PERV'] = c['perv']
            os.environ['INFQ_STOCH'] = '0'
            qmeta = inf_quant.apply_inf_quant(infinity)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i, (key, prompt) in enumerate(items):
            img = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt,
                              g_seed=i, gt_leak=0, gt_ls_Bl=None, cfg_list=a.cfg, tau_list=a.tau,
                              scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer],
                              vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits,
                              enable_positive_prompt=0)
            cv2.imwrite(osp.join(out, '%s.jpg' % key), img.cpu().numpy())
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print('[%s] %s done %d imgs (%.1fs) peak=%.2fGB' % (c['name'], json.dumps(qmeta), len(items), time.time() - t0, peak), flush=True)
        summary['configs'].append(dict(name=c['name'], quant=qmeta, kv_mem_gb=memtab[c['name']], peak_gpu_gb=peak, out=out))
    inf_quant.remove_inf_quant(infinity)
    with open(osp.join(ROOT, 'results/infinity', 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary), flush=True)
    print('INF_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
