import argparse, glob, os.path as osp, json
import numpy as np
import cv2
from insightface.app import FaceAnalysis


def gather(d):
    fs = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.JPEG'):
        fs += glob.glob(osp.join(d, ext))
        fs += glob.glob(osp.join(d, '**', ext), recursive=True)
    return sorted(set(fs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='image dir (flat or nested)')
    ap.add_argument('--out', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--det_thresh', type=float, default=0.5)
    a = ap.parse_args()

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                       allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(256, 256), det_thresh=a.det_thresh)

    files = gather(a.dir)
    if a.limit:
        files = files[:a.limit]

    n = 0; with_face = 0; faces_total = 0; dets = []; embs = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        n += 1
        fs = app.get(img)
        if fs:
            with_face += 1
            faces_total += len(fs)
            fs.sort(key=lambda x: x.det_score, reverse=True)
            dets.append(float(fs[0].det_score))
            if getattr(fs[0], 'normed_embedding', None) is not None:
                embs.append(fs[0].normed_embedding)

    res = dict(dir=a.dir, n_images=n, images_with_face=with_face,
               detection_rate=round(with_face / max(n, 1), 4),
               faces_per_image=round(faces_total / max(n, 1), 4),
               mean_det_score=round(float(np.mean(dets)) if dets else 0.0, 4))
    if len(embs) >= 2:
        E = np.stack(embs)
        sims = E @ E.T
        iu = np.triu_indices(len(E), 1)
        res['mean_pairwise_cossim'] = round(float(sims[iu].mean()), 4)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, 'w') as fp:
            json.dump(res, fp, indent=2)
        print('saved', a.out)


if __name__ == '__main__':
    main()
