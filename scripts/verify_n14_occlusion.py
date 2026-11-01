"""N14静态遮挡对照；只作离线夹具核验，不是控制或检测训练数据。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('H:/OccluActiSafeDataset/artifacts')


def run(blocked,clear,output):
    def sample(folder):
        p=Path(folder)/'reference';r=json.loads((p/'评测真值.jsonl').read_text(encoding='utf-8').splitlines()[0]);return p,r
    a,x=sample(blocked);b,y=sample(clear);output=Path(output);output.mkdir(exist_ok=True)
    checks={}
    checks['相同自车与目标平面位姿']=all(math.hypot(x[k]['x']-y[k]['x'],x[k]['y']-y[k]['y'])<1e-5 and abs(x[k]['yaw']-y[k]['yaw'])<1e-5 for k in ('ego','target'))
    checks['实体遮挡组造成完全遮挡']=x['target_pixels']['车端']==0 and y['target_pixels']['车端']>0
    checks['路侧两次均可见']=min(x['target_pixels']['路侧'],y['target_pixels']['路侧'])>0
    checks['两次安全收尾']=all(json.loads((p/'清理报告.json').read_text(encoding='utf-8'))['status']=='completed' for p in (a,b))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    fig,axes=plt.subplots(1,2,figsize=(16,5),constrained_layout=True)
    files=[]
    for ax,folder,row,label in zip(axes,(a,b),(x,y),('保留实体遮挡组','移开实体遮挡组')):
        f=folder/'车端画面_帧0000.png';files.append(f)
        ax.imshow(plt.imread(f));ax.axis('off');ax.set_title(f"{label}，目标可见像素={row['target_pixels']['车端']}")
    for ext in ('png','svg'):fig.savefig(output/f'大车遮挡静态对照.{ext}',dpi=150)
    plt.close(fig)
    result={'checks':checks,'passed':sum(checks.values()),'failed':sum(not v for v in checks.values()),
        'blocked_run':str(blocked),'clear_run':str(clear),'blocked_pixels':x['target_pixels'],'clear_pixels':y['target_pixels'],
        'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'scope':'固定自车与目标平面位姿的可见性对照，未严格复刻动态三维姿态，不是避撞反事实，未向在线控制提供隐藏状态或实例标签'}
    (output/'遮挡核验报告.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"遮挡核验：通过{result['passed']}，失败{result['failed']}")
    if result['failed']:raise RuntimeError('遮挡夹具无效')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--blocked-run',type=Path);parser.add_argument('--clear-run',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'n14_20261031');args=parser.parse_args()
    options=[]
    for p in ROOT.glob('n14_*'):
        f=p/'冻结配置.json'
        if f.is_file():
            c=json.loads(f.read_text(encoding='utf-8'))
            if c.get('probe_static') and (p/'reference'/'公共前缀.json').is_file():options.append((p,c))
    def latest(clear):
        selected=[p for p,c in options if (abs(c['occluder']['x'])>500)==clear]
        if not selected:raise FileNotFoundError('没有可用的静态遮挡核验输入')
        return max(selected,key=lambda p:p.stat().st_mtime)
    run(args.blocked_run or latest(False),args.clear_run or latest(True),args.output)
