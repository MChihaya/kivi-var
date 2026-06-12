import os, os.path as osp, glob, argparse
from concurrent.futures import ThreadPoolExecutor
import PIL.Image as Image

S = 256

def process(pair):
    src, dst = pair
    if osp.exists(dst):
        return True
    try:
        im = Image.open(src).convert('RGB')
        w, h = im.size
        if w < h:
            nw, nh = S, round(h * S / w)
        else:
            nw, nh = round(w * S / h), S
        im = im.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - S) // 2, (nh - S) // 2
        im.crop((left, top, left + S, top + S)).save(dst)
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.environ.get('IMAGENET_VAL', ''))
    ap.add_argument('--dst', default='./data/imagenet_val_256')
    ap.add_argument('--per_class', type=int, default=0, help='0 = all')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    tasks = []
    for cd in sorted(d for d in glob.glob(osp.join(a.src, 'n*')) if osp.isdir(d)):
        wnid = osp.basename(cd)
        imgs = sorted(glob.glob(osp.join(cd, '*.JPEG')))
        if a.per_class:
            imgs = imgs[:a.per_class]
        for k, src in enumerate(imgs):
            tasks.append((src, osp.join(a.dst, f'{wnid}_{k:04d}.png')))
    print('total tasks', len(tasks), flush=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(process, tasks)):
            ok += int(r)
            if (i + 1) % 5000 == 0:
                print(i + 1, 'done', flush=True)
    print('RESIZE_DONE ok=', ok, '/', len(tasks), flush=True)

if __name__ == '__main__':
    main()
