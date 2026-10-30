"""N04 分文件NPZ读取与仅训练集拟合的掩码归一化。"""
import json
from pathlib import Path
import numpy as np
from occluactisafe.data.split import canonical_hash
from occluactisafe.data.windows import FEATURES, input_hash


class MaskedScaler:
    def __init__(self):
        self.count=np.zeros(len(FEATURES),dtype=np.int64)
        self.mean=np.zeros(len(FEATURES));self.m2=np.zeros(len(FEATURES))

    def update(self, values, mask, *, role):
        if role!='train':raise ValueError('只允许训练集合拟合归一化')
        values=np.asarray(values);mask=np.asarray(mask)
        if values.shape!=mask.shape or values.shape[-1]!=len(FEATURES) or mask.dtype.kind!='b':raise ValueError('特征/掩码形状不一致')
        if not np.isfinite(values[mask]).all():raise ValueError('有效特征包含非有限值')
        x=values.reshape(-1,len(FEATURES));m=mask.reshape(x.shape)
        n=m.sum(axis=0);safe=np.where(m,x,0)
        mean=safe.sum(axis=0)/np.maximum(n,1)
        m2=np.where(m,(safe-mean)**2,0).sum(axis=0)
        self.merge({'count':n.tolist(),'mean':mean.tolist(),'m2':m2.tolist()},role=role)

    def merge(self, statistics, *, role):
        if role!='train':raise ValueError('只允许训练统计合并')
        n=np.asarray(statistics['count'],dtype=np.int64);mean=np.asarray(statistics['mean']);m2=np.asarray(statistics['m2'])
        total=self.count+n;delta=mean-self.mean
        self.mean+=delta*n/np.maximum(total,1)
        self.m2+=m2+delta**2*self.count*n/np.maximum(total,1)
        self.count=total

    def statistics(self):
        return {'count':self.count.tolist(),'mean':self.mean.tolist(),'m2':self.m2.tolist()}

    def freeze(self):
        std=np.sqrt(np.maximum(self.m2/np.maximum(self.count,1),0));std=np.where(std>1e-12,std,1)
        result={'schema_version':'1.0','fit_role':'train','features':list(FEATURES),
                'count':self.count.tolist(),'mean':self.mean.tolist(),'scale':std.tolist(),
                'empty_feature_indices':np.flatnonzero(self.count==0).tolist()}
        result['sha256']=canonical_hash(result)
        return result

    @staticmethod
    def transform(window, scaler):
        if scaler['features']!=list(FEATURES) or scaler['fit_role']!='train':raise ValueError('归一化契约不匹配')
        return np.where(window['feature_mask'],(window['features']-np.asarray(scaler['mean']))/np.asarray(scaler['scale']),0)


def save_sample(path, window, labels):
    arrays={k:v for k,v in window.items() if isinstance(v,np.ndarray)}
    metadata={k:v for k,v in window.items() if not isinstance(v,np.ndarray)}
    np.savez_compressed(path,**arrays,metadata=np.array(json.dumps(metadata,separators=(',',':'))))
    np.savez_compressed(Path(path).with_suffix('.labels.npz'),**labels)


def load_sample(path):
    with np.load(path,allow_pickle=False) as archive:
        result={key:archive[key] for key in archive.files if key!='metadata'}
        result.update(json.loads(str(archive['metadata'])))
    if input_hash(result)!=result['input_hash']:raise ValueError('样本输入哈希不匹配')
    return result


class PredictionDataset:
    """按索引惰性读取单个场景，监督字段只由labels()提供。无Torch依赖。"""
    def __init__(self, run_dir, role, *, normalized=True):
        self.root=Path(run_dir).resolve()
        if role not in ('train','model_val','risk_calibration','test'):raise ValueError('集合角色非法')
        self.index=[]
        for path in sorted((self.root/'samples').glob('*/index.jsonl')):
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['project_role']==role and row['status']=='written':self.index.append(row)
        self.scaler=json.loads((self.root/'scaler.json').read_text(encoding='utf-8')) if normalized else None
        if self.scaler is not None:
            expected=self.scaler.get('sha256')
            if canonical_hash({k:v for k,v in self.scaler.items() if k!='sha256'})!=expected:
                raise ValueError('归一化文件哈希不匹配')

    def __len__(self):return len(self.index)

    def _load(self,index):
        path=(self.root/self.index[index]['sample_path']).resolve()
        if not path.is_relative_to(self.root):raise ValueError('样本路径越界')
        return load_sample(path)

    def inputs(self,index):
        row=self._load(index)
        result={key:row[key] for key in ('features','feature_mask','observed_mask','history_times','actual_dt',
                                         'categories','prediction_target_mask','track_ids','anchor_pose','input_hash')}
        if self.scaler is not None:result['features']=MaskedScaler.transform(row,self.scaler)
        return result

    def labels(self,index):
        path=(self.root/self.index[index]['sample_path']).resolve().with_suffix('.labels.npz')
        if not path.is_relative_to(self.root):raise ValueError('监督路径越界')
        with np.load(path,allow_pickle=False) as archive:
            return {key:archive[key] for key in archive.files}
