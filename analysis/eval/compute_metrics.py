import argparse, json
import torch_fidelity

CACHE_ROOT = './data/fid_cache'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', required=True, help='generated images dir (input1)')
    ap.add_argument('--real', default='', help='real reference dir (input2); empty => IS only')
    ap.add_argument('--fid', action='store_true')
    ap.add_argument('--isc', action='store_true', help='Inception Score on generated set')
    ap.add_argument('--kid', action='store_true')
    ap.add_argument('--prc', action='store_true', help='precision/recall')
    ap.add_argument('--deep', action='store_true', help='recurse into subdirs of inputs')
    ap.add_argument('--cache_real_name', default='', help='cache name for real stats (enables caching of input2)')
    ap.add_argument('--out', default='', help='optional json output path')
    a = ap.parse_args()

    kwargs = dict(
        input1=a.gen, cuda=True, verbose=True,
        isc=a.isc, fid=a.fid, kid=a.kid, prc=a.prc,
        samples_find_deep=a.deep,
    )
    if a.real:
        kwargs['input2'] = a.real
    if a.cache_real_name:
        kwargs['cache'] = True
        kwargs['cache_root'] = CACHE_ROOT
        kwargs['input2_cache_name'] = a.cache_real_name

    m = torch_fidelity.calculate_metrics(**kwargs)
    m = {k: float(v) for k, v in m.items()}
    print(json.dumps(m, indent=2))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(m, f, indent=2)
        print('saved', a.out)


if __name__ == '__main__':
    main()
