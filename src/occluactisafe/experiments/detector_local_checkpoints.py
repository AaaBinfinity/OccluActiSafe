"""N17本机检查点：仅last/best原子发布；续训保留旧目录和未提交记录。"""
import csv
import json
import os
from pathlib import Path
import tempfile
import hashlib

from .run_store import read_json, write_json, sha256


def atomic_bytes(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.模型发布-',suffix='.tmp',delete=False) as f:
            temporary=Path(f.name);f.write(content);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path);temporary=None
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def publish_weights(directory, serialized, is_best, metadata):
    directory=Path(directory);weights=directory/'weights';weights.mkdir(parents=True,exist_ok=True)
    unexpected={p.name for p in weights.iterdir()}-{'last.pt','best.pt'}
    if unexpected:raise ValueError('本机weights包含非last/best文件，拒绝覆盖或自动删除')
    # 事务先落盘；中断后可核对last/best，不把不完整发布冒充完成。
    pending=directory/'检查点发布事务.json'
    write_json(pending,{'status':'publishing','is_best':is_best,'new_sha256':hashlib.sha256(serialized).hexdigest(),
                       'previous_best_sha256':sha256(weights/'best.pt') if (weights/'best.pt').is_file() else None,**metadata})
    atomic_bytes(weights/'last.pt',serialized)
    if is_best:atomic_bytes(weights/'best.pt',serialized)
    state={'status':'completed',**metadata,'last_sha256':sha256(weights/'last.pt'),
           'best_sha256':sha256(weights/'best.pt') if (weights/'best.pt').is_file() else None}
    write_json(directory/'检查点状态.json',state);write_json(pending,state)
    return state


def verify_checkpoint(directory):
    directory=Path(directory);state=directory/'检查点状态.json'
    record=read_json(state) if state.exists() else None
    pending=read_json(directory/'检查点发布事务.json')
    if pending['status']=='publishing' and (directory/'weights/last.pt').is_file() and sha256(directory/'weights/last.pt')==pending['new_sha256']:
        source='last.pt' if pending['is_best'] else 'best.pt'
        expected=pending['new_sha256'] if pending['is_best'] else pending['previous_best_sha256']
        if expected is None or sha256(directory/'weights'/source)!=expected:raise ValueError('中断发布事务的best来源无法核实')
        return {**pending,'status':'recoverable_pending','last_sha256':pending['new_sha256'],'best_sha256':expected,'best_source_file':source}
    if record is None:raise ValueError('没有已完成的检查点，不能伪造恢复')
    if record['status']!='completed':raise ValueError('检查点发布未完成')
    for name,key in (('last.pt','last_sha256'),('best.pt','best_sha256')):
        if record[key] is None or sha256(directory/'weights'/name)!=record[key]:raise ValueError('检查点SHA改变或缺少'+name)
    return {**record,'best_source_file':'best.pt'}


def carry_committed_logs(source, destination, epoch, counts=None):
    """复制已完成epoch的原始前缀到新续训目录；原失败目录和多出的尾部不修改。"""
    source,destination=Path(source),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in ('逐步优化指标.csv','逐批次优化尝试.csv','results.csv'):
        old=source/name
        if not old.exists():continue
        target=destination/name
        if target.exists():raise FileExistsError(target)
        with old.open(encoding='utf-8',newline='') as a,target.open('x',encoding='utf-8',newline='') as b:
            reader=csv.DictReader(a);writer=csv.DictWriter(b,fieldnames=reader.fieldnames);writer.writeheader();count=0
            limit=counts.get(name) if counts else None
            for index,row in enumerate(reader):
                if limit is not None and index>=limit:break
                value=row.get('epoch') or row.get(' epoch')
                if value is None:raise ValueError('待恢复CSV缺少epoch列')
                if int(float(value))<=epoch:writer.writerow(row);count+=1
            if limit is not None and count!=limit:raise ValueError('已提交CSV前缀少于检查点记录')
        records.append({'file':name,'source_sha256':sha256(old),'carried_sha256':sha256(target),'rows':count})
    for name in ('逐轮验证指标.jsonl','逐轮分类验证指标.jsonl','早停判定.jsonl','逐轮耗时.jsonl','逐轮优化与资源.jsonl','逐轮显卡监测.jsonl'):
        old=source/name
        if not old.exists():continue
        target=destination/name
        if target.exists():raise FileExistsError(target)
        count=0
        with old.open(encoding='utf-8') as a,target.open('x',encoding='utf-8') as b:
            for index,line in enumerate(a):
                if index>=epoch:break  # 未提交尾部可能是中断时的半行，原文件仍完整保留。
                row=json.loads(line)
                if row['epoch']<=epoch:b.write(line);count+=1
        if name in ('逐轮验证指标.jsonl','早停判定.jsonl') and count!=epoch:raise ValueError('已提交逐轮记录不完整')
        records.append({'file':name,'source_sha256':sha256(old),'carried_sha256':sha256(target),'rows':count})
    write_json(destination/'续训原始前缀来源.json',{'source':str(source),'last_completed_epoch':epoch,
               'old_directory_unchanged':True,'uncommitted_tail_retained_in_source':True,'files':records})
    return records
