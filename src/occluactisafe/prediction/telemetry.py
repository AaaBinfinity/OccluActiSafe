"""逐优化更新的可复核CSV；恢复只继承检查点对应前缀，旧记录不改写。"""
import csv
import hashlib
import json
import math

FIELDS = ('global_step','epoch','batch_index','target_count','loss','regression_loss',
          'classification_loss','train_minADE_m','gradient_norm_before_clip','learning_rate',
          'step_seconds','elapsed_seconds','gpu_allocated_mib','gpu_reserved_mib',
          'host_working_set_mib','batch_target_indices')


class StepLog:
    def __init__(self, path, *, source=None, steps=0, prefix_sha256=None):
        self.path=path;self.rows=0;self.digest=hashlib.sha256()
        self.file=path.open('w',encoding='utf-8',newline='')
        self.writer=csv.DictWriter(self.file,fieldnames=FIELDS);self.writer.writeheader();self.file.flush()
        try:
            if steps:
                if source is None or not source.is_file():raise ValueError('恢复缺少逐步原始指标日志，不能补造')
                with source.open(encoding='utf-8',newline='') as f:
                    reader=csv.DictReader(f)
                    if tuple(reader.fieldnames or ())!=FIELDS:raise ValueError('恢复日志字段版本不兼容')
                    for row in reader:
                        if self.rows==steps:break
                        self.append(row)
                if self.rows!=steps or self.sha256!=prefix_sha256:raise ValueError('恢复日志前缀数量/哈希与检查点不符')
            elif prefix_sha256 not in (None,self.sha256):raise ValueError('空日志前缀哈希不符')
        except BaseException:
            self.file.close();raise

    @property
    def sha256(self):return self.digest.hexdigest()

    def append(self, row):
        if set(row)!=set(FIELDS) or int(row['global_step'])!=self.rows+1:raise ValueError('逐步指标字段不符或步号重复/不连续')
        for key in FIELDS[:-1]:
            if row[key] not in (None,'') and not math.isfinite(float(row[key])):raise ValueError('逐步指标非有限')
        values={k:'' if row[k] is None else str(row[k]) for k in FIELDS}
        self.writer.writerow(values);self.file.flush()
        self.digest.update((json.dumps(values,sort_keys=True,separators=(',',':'))+'\n').encode())
        self.rows+=1

    def close(self):self.file.close()
