import os, os.path as osp, glob, json, argparse
from concurrent.futures import ThreadPoolExecutor
import PIL.Image as Image

S = 256


def idx2wnid(train_root):
    wnids = sorted(osp.basename(d) for d in glob.glob(osp.join(train_root, 'n*')) if osp.isdir(d))
    return {i: w for i, w in enumerate(wnids)}


def resize_one(pair):
    src, dst = pair
    if osp.exists(dst):
        return True
    try:
        im = Image.open(src).convert('RGB'); w, h = im.size
        if w < h:
            nw, nh = S, round(h * S / w)
        else:
            nw, nh = round(w * S / h), S
        im = im.resize((nw, nh), Image.BICUBIC)
        l, t = (nw - S) // 2, (nh - S) // 2
        im.crop((l, t, l + S, t + S)).save(dst)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_root', default=os.environ.get('IMAGENET_TRAIN', ''))
    ap.add_argument('--classes_json', default='analysis/failure_classes.json')
    ap.add_argument('--dst', default='./data/ref_perclass')
    ap.add_argument('--per_class', type=int, default=500)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()

    i2w = idx2wnid(a.train_root)
    with open(a.classes_json) as f:
        cj = json.load(f)
    idxs = sorted({int(k) for grp, d in cj.items() if not grp.startswith('_') for k in d})

    tasks = []
    for idx in idxs:
        wnid = i2w[idx]
        outd = osp.join(a.dst, f'{idx:04d}'); os.makedirs(outd, exist_ok=True)
        imgs = sorted(glob.glob(osp.join(a.train_root, wnid, '*.JPEG')))[:a.per_class]
        for k, src in enumerate(imgs):
            tasks.append((src, osp.join(outd, f'{k:04d}.png')))
    print('classes', len(idxs), 'tasks', len(tasks), flush=True)

    ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(resize_one, tasks)):
            ok += int(r)
            if (i + 1) % 2000 == 0:
                print(i + 1, 'done', flush=True)
    print('PERCLASS_DONE ok=', ok, '/', len(tasks), flush=True)


if __name__ == '__main__':
    main()
