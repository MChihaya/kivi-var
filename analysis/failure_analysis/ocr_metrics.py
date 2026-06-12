import argparse, glob, os.path as osp, json, re
import numpy as np
import easyocr

WORD_RE = re.compile(r'^[A-Za-z]{2,}$')


def load_dict():
    for p in ('/usr/share/dict/words', '/usr/share/dict/american-english'):
        if osp.exists(p):
            with open(p, encoding='utf-8', errors='ignore') as f:
                return {w.strip().lower() for w in f if w.strip()}
    return None


def gather(d):
    fs = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.JPEG'):
        fs += glob.glob(osp.join(d, ext))
        fs += glob.glob(osp.join(d, '**', ext), recursive=True)
    return sorted(set(fs))


def is_valid_word(w, vocab):
    w = w.strip()
    if not WORD_RE.match(w):
        return False
    if vocab is not None:
        return w.lower() in vocab
    return bool(re.search(r'[aeiouAEIOU]', w))  # fallback: must contain a vowel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--conf', type=float, default=0.3, help='min region confidence to count')
    ap.add_argument('--gpu', action='store_true')
    a = ap.parse_args()

    vocab = load_dict()
    reader = easyocr.Reader(['en'], gpu=a.gpu, verbose=False)
    files = gather(a.dir)
    if a.limit:
        files = files[:a.limit]

    regions, chars, confs_img, valid_fracs, with_text = [], [], [], [], 0
    n = 0
    for f in files:
        try:
            res = reader.readtext(f)
        except Exception:
            continue
        n += 1
        regs = [(t, c) for (_, t, c) in res if c >= a.conf]
        regions.append(len(regs))
        chars.append(sum(len(t) for t, _ in regs))
        if regs:
            with_text += 1
            confs_img.append(float(np.mean([c for _, c in regs])))
            words = [t for t, _ in regs]
            valid_fracs.append(sum(is_valid_word(w, vocab) for w in words) / len(words))

    mean = lambda x: round(float(np.mean(x)), 4) if x else 0.0
    res = dict(dir=a.dir, n_images=n, dict_loaded=vocab is not None,
               frac_images_with_text=round(with_text / max(n, 1), 4),
               mean_regions_per_image=mean(regions),
               mean_chars_per_image=mean(chars),
               mean_region_conf=mean(confs_img),
               mean_valid_word_frac=mean(valid_fracs))
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, 'w') as fp:
            json.dump(res, fp, indent=2)
        print('saved', a.out)


if __name__ == '__main__':
    main()
