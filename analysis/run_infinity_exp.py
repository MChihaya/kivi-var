import os, os.path as osp, sys, json, argparse, time
import numpy as np
import torch

ROOT = '.'
INF = osp.join(ROOT, 'Infinity')
sys.path.insert(0, INF)
sys.path.insert(0, osp.join(INF, 'tools'))
sys.path.insert(0, osp.join(ROOT, 'analysis'))
import cv2
from run_infinity import (load_tokenizer, load_visual_tokenizer, load_transformer,
                          gen_one_img, dynamic_resolution_h_w, h_div_w_templates)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=osp.join(ROOT, 'results/infinity/fp16'))
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--n', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

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

    qmeta = None
    if int(os.environ.get('INFQ_BITS', '16')) < 16:
        from kv_compression.inf_quant import apply_inf_quant
        qmeta = apply_inf_quant(infinity)
        print('INF-quant active:', json.dumps(qmeta), flush=True)

    h_div_w = 1.0
    tmpl = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[tmpl][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    items = list(PROMPTS.items())[:a.n]
    t0 = time.time()
    for i, (key, prompt) in enumerate(items):
        img = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt,
                          g_seed=i, gt_leak=0, gt_ls_Bl=None, cfg_list=a.cfg, tau_list=a.tau,
                          scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer],
                          vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits,
                          enable_positive_prompt=0)
        cv2.imwrite(osp.join(a.out, '%s.jpg' % key), img.cpu().numpy())
        print('[%d/%d] %s (%.1fs)' % (i + 1, len(items), key, time.time() - t0), flush=True)
    meta = dict(out=a.out, n=len(items), cfg=a.cfg, tau=a.tau, quant=qmeta,
                peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    with open(osp.join(a.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta), flush=True)
    print('INF_GEN_DONE', flush=True)


if __name__ == '__main__':
    main()
