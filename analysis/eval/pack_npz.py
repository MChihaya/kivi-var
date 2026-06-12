import glob, os.path as osp, sys
import numpy as np
import PIL.Image as Image

src, dst = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
files = sorted(glob.glob(osp.join(src, '*.png')))
if limit:
    files = files[:limit]
n = len(files)
arr = np.zeros((n, 256, 256, 3), dtype=np.uint8)
for i, f in enumerate(files):
    im = Image.open(f).convert('RGB')
    if im.size != (256, 256):
        im = im.resize((256, 256))
    arr[i] = np.asarray(im)
    if (i + 1) % 10000 == 0:
        print(i + 1, 'packed', flush=True)
np.savez(dst, arr_0=arr)
print('PACKED', arr.shape, '->', dst, flush=True)
