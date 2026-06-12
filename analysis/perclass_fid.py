"""Per-class FID with a consistent protocol (50 generated vs the real reference) for both
the failure classes and the VAR-strong "good" classes, so the high-frequency figure can
plot both on the same axes. Builds good-class real references from ImageNet train."""
import os, os.path as osp, glob, json, shutil, random
ROOT='.'; VAL=os.environ['IMAGENET_VAL']; TRAIN=os.environ['IMAGENET_TRAIN']
REF=osp.join(ROOT,'data/ref_perclass'); GFAIL=osp.join(ROOT,'results/failure1/gen'); GGOOD=osp.join(ROOT,'results/good_fid_gen')
WNIDS=sorted([d for d in os.listdir(VAL) if d.startswith('n')])
fc=json.load(open(osp.join(ROOT,'analysis/failure_classes.json')))
GOOD={207:'golden retriever',291:'lion',340:'zebra',388:'giant panda',779:'school bus',817:'sports car',963:'pizza',985:'daisy'}
classes=[]
for g in ['text','fine_structure','animal_detail','human_face','reference_easy']:
    for k,v in fc.get(g,{}).items(): classes.append((int(k),v,g))
for k,v in GOOD.items(): classes.append((k,v,'good_object'))
# build good-class real refs (symlink up to 500 train images)
for idx,name,grp in classes:
    rd=osp.join(REF,'%04d'%idx)
    if osp.isdir(rd) and len(os.listdir(rd))>=200: continue
    os.makedirs(rd,exist_ok=True)
    src=sorted(glob.glob(osp.join(TRAIN,WNIDS[idx],'*.JPEG')))[:500]
    for s in src:
        d=osp.join(rd,osp.basename(s))
        if not osp.exists(d):
            try: os.symlink(s,d)
            except Exception: pass
    print('ref %04d %s: %d real'%(idx,name,len(os.listdir(rd))),flush=True)
import torch_fidelity
SHM='/dev/shm/gfid'; os.makedirs(SHM,exist_ok=True)
rows=[]
for idx,name,grp in classes:
    gd=osp.join(GFAIL,'%04d'%idx)
    if not osp.isdir(gd): gd=osp.join(GGOOD,'%04d'%idx)
    gfiles=sorted(glob.glob(osp.join(gd,'*.png')))[:50]
    td=osp.join(SHM,'%04d'%idx); shutil.rmtree(td,ignore_errors=True); os.makedirs(td)
    for f in gfiles: os.symlink(f,osp.join(td,osp.basename(f)))
    rd=osp.join(REF,'%04d'%idx)
    try:
        fid=torch_fidelity.calculate_metrics(input1=td,input2=rd,fid=True,cuda=True,verbose=False)['frechet_inception_distance']
        fid=round(float(fid),2)
    except Exception as e:
        fid=None; print('ERR',idx,str(e)[:80])
    rows.append(dict(idx=idx,name=name,group=grp,fid=fid,n_gen=len(gfiles)))
    print('FID %04d %-16s %-15s = %s'%(idx,name[:16],grp,fid),flush=True)
json.dump(rows,open(osp.join(ROOT,'results/perclass_fid.json'),'w'),indent=2)
print('PCFID_DONE')
