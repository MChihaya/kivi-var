import argparse, glob, os.path as osp, json, re
import numpy as np
import cv2

WORD_RE = re.compile(r'^[A-Za-z]{2,}$')


def gather(d, limit=0):
    fs = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.JPEG'):
        fs += glob.glob(osp.join(d, ext))
    fs = sorted(set(fs))
    return fs[:limit] if limit else fs


def load_vocab():
    for p in ('/usr/share/dict/words', '/usr/share/dict/american-english'):
        if osp.exists(p):
            return {w.strip().lower() for w in open(p, encoding='utf-8', errors='ignore') if w.strip()}
    return None


def is_word(w, vocab):
    w = w.strip()
    if not WORD_RE.match(w):
        return False
    return (w.lower() in vocab) if vocab is not None else bool(re.search(r'[aeiouAEIOU]', w))


def ocr_stats(files, reader, vocab, conf):
    regions, chars, confs, valid, with_text, n = [], [], [], [], 0, 0
    for f in files:
        try:
            res = reader.readtext(f)
        except Exception:
            continue
        n += 1
        regs = [(t, c) for (_, t, c) in res if c >= conf]
        regions.append(len(regs)); chars.append(sum(len(t) for t, _ in regs))
        if regs:
            with_text += 1
            confs.append(float(np.mean([c for _, c in regs])))
            words = [t for t, _ in regs]
            valid.append(sum(is_word(w, vocab) for w in words) / len(words))
    m = lambda x: round(float(np.mean(x)), 3) if x else 0.0
    return dict(n=n, frac_with_text=round(with_text / max(n, 1), 3),
                regions=m(regions), chars=m(chars), conf=m(confs), valid_word_frac=m(valid))


def face_stats(files, app):
    n, with_face, faces, dets = 0, 0, 0, []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        n += 1
        fs = app.get(img)
        if fs:
            with_face += 1; faces += len(fs)
            fs.sort(key=lambda x: x.det_score, reverse=True)
            dets.append(float(fs[0].det_score))
    m = lambda x: round(float(np.mean(x)), 3) if x else 0.0
    return dict(n=n, det_rate=round(with_face / max(n, 1), 3),
                faces_per_img=round(faces / max(n, 1), 3), det_score=m(dets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen_root', default='./results/failure1/gen')
    ap.add_argument('--real_root', default='./data/ref_perclass')
    ap.add_argument('--classes_json', default='analysis/failure_classes.json')
    ap.add_argument('--out', default='./results/failure1/failure1_table.json')
    ap.add_argument('--n', type=int, default=150, help='images/dir for OCR & face')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--gpu', action='store_true')
    a = ap.parse_args()

    with open(a.classes_json) as f:
        cj = json.load(f)
    vocab = load_vocab()

    import easyocr
    reader = easyocr.Reader(['en'], gpu=a.gpu, verbose=False)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                       allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(256, 256))
    import torch_fidelity

    rows = []
    for group, members in cj.items():
        if group.startswith('_'):
            continue
        for k, name in members.items():
            idx = int(k)
            gdir = osp.join(a.gen_root, f'{idx:04d}')
            rdir = osp.join(a.real_root, f'{idx:04d}')
            row = dict(group=group, idx=idx, name=name)
            try:
                fid = torch_fidelity.calculate_metrics(
                    input1=gdir, input2=rdir, fid=True, cuda=True, verbose=False
                )['frechet_inception_distance']
                row['fid_gen_vs_real'] = round(float(fid), 2)
            except Exception as e:
                row['fid_gen_vs_real'] = None
                row['fid_err'] = str(e)[:120]
            gfiles, rfiles = gather(gdir, a.n), gather(rdir, a.n)
            if group == 'text':
                row['ocr_real'] = ocr_stats(rfiles, reader, vocab, a.conf)
                row['ocr_gen'] = ocr_stats(gfiles, reader, vocab, a.conf)
            if group == 'human_face':
                row['face_real'] = face_stats(rfiles, app)
                row['face_gen'] = face_stats(gfiles, app)
            rows.append(row)
            print('[done]', group, idx, name, 'fid=', row.get('fid_gen_vs_real'), flush=True)

    out = dict(config=dict(gen_root=a.gen_root, real_root=a.real_root, n=a.n, conf=a.conf, vocab=vocab is not None),
               rows=rows)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=2)
    print('SAVED', a.out)


if __name__ == '__main__':
    main()
