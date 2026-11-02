"""N17耗时预算：明确区分假设区间与同设备pilot实测外推，不承诺收敛。"""
import json
import math
from pathlib import Path
import statistics

from .detector_models import expanded_models
from .run_store import read_json, sha256, write_json, now

# 960输入、单卡串行、batch16/AMP、17003训练+3185验证、含IO的条件假设，非A100实测。
ASSUMED_MINUTES = {'n': (1, 2.5), 's': (1.5, 3.5), 'm': (2.5, 5.5), 'l': (3.5, 7.5), 'x': (5, 11)}


def timing_budget(settings, pilot=None, destination=None):
    if settings.get('model_sizes'):
        models = expanded_models(settings['model_sizes'])
    else:
        from .detector_data import load_config
        models = [dict(m,scale='n') for m in load_config()['models']]
    per_model, sources = {}, []
    if pilot:
        pilot = Path(pilot).resolve(); state = read_json(pilot/'任务状态.json')
        if state['mode'] != 'pilot' or state['status'] != 'completed' or state['configuration_sha256'] != settings['_sha256']:
            raise ValueError('耗时外推须使用同站点配置、完整完成的pilot，不能使用计划或失败任务')
        expected = {m['family'] for m in models}
        for task in state['tasks']:
            if task['kind'] != 'train' or task['policy'] != settings['formal_policy']: continue
            if task['status'] != 'completed' or task['attempts'][-1].get('returncode') != 0:
                raise ValueError('pilot计时任务未成功')
            result = read_json(Path(task['attempts'][-1]['directory'])/'worker_result.json')
            path = Path(result['directory'])/'训练/逐轮耗时.jsonl'
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            if len(records) != settings['pilot_epochs'] or [r['epoch'] for r in records] != list(range(1,len(records)+1)):
                raise ValueError('pilot逐轮耗时记录不完整')
            values = [float(r['seconds_including_validation_and_save']) for r in records]
            if any(not math.isfinite(v) or v <= 0 for v in values): raise ValueError('实际耗时无效')
            # 有至少3轮才排除首次初始化/缓存轮，保留全部原始值和来源SHA。
            usable = values[1:] if len(values) >= 3 else values
            mean = statistics.mean(usable)
            per_model[task['family']] = (mean*.8, mean*1.3)
            sources.append({'family':task['family'],'path':str(path),'sha256':sha256(path),
                            'raw_seconds':values,'used_seconds':usable,'mean_seconds':mean})
        if set(per_model) != expected: raise ValueError('pilot计时缺少模型，不用其他型号代填')
    else:
        per_model = {m['family']: tuple(v*60 for v in ASSUMED_MINUTES[m['scale']]) for m in models}
    minimum = settings.get('minimum_epochs', 1) if settings.get('early_stopping_patience', 0) else settings['formal_epochs']
    epoch_cases = sorted({minimum, min(max(100,minimum),settings['formal_epochs']), settings['formal_epochs']})
    scenarios = []
    for epochs in epoch_cases:
        lower = sum(v[0] for v in per_model.values())*3*epochs*1.1
        upper = sum(v[1] for v in per_model.values())*3*epochs*1.1
        scenarios.append({'assumed_epochs_per_run':epochs,'training_runs':len(models)*3,
                          'hours_lower':lower/3600,'hours_upper':upper/3600,
                          'days_lower':lower/86400,'days_upper':upper/86400})
    result = {'schema_version':'n17-timing-budget-1','created_at':now(),
              'basis':'same_config_pilot_extrapolation' if pilot else 'unmeasured_conditional_assumptions',
              'actual_a100_measurement':bool(pilot),'server_hardware_verified_by_this_function':False,
              'estimated_hardware': '使用pilot记录的实际GPU' if pilot else '假设单卡A100，尚未实测',
              'assumptions':{'input_long_edge':960,'train_frames':17003,'validation_frames':3185,
                             'assumed_batch':16,'assumed_amp':True,'actual_config_batch':settings['batch'],
                             'actual_config_amp':settings.get('amp',False),'execution':'single_gpu_serial',
                             'seeds':3,'reserve_factor':1.1,'interval_is_confidence_interval':False},
              'per_model_epoch_seconds_interval':per_model,'pilot_run':str(pilot) if pilot else None,
              'pilot_sources':sources,'scenarios':scenarios,'convergence_guaranteed':False,
              'limitations':['平台期触发轮数未知；情景不是预计会在该轮收敛',
                             '不含首次环境/数据拷贝/下载和pilot成本，最终评测以10%余量粗留',
                             'GPU分片/共享、CPU/磁盘/杀毒、checkpoint保存会改变速度',
                             '服务器速度不证明本机部署p95；门槛不变']}
    if pilot:
        result['actual_a100_measurement'] = 'a100' in read_json(state['environment_report'])['gpu']['name'].lower()
        result['pilot_gpu'] = read_json(state['environment_report'])['gpu']
    if destination:
        destination = Path(destination); write_json(destination/'耗时预算.json', result)
        lines=['# N17训练耗时预算', '依据：'+result['basis']+'。不是收敛保证或统计置信区间。',
               '| 假设每次实际轮数 | 训练次数 | 小时区间 | 天数区间 |','| --- | --- | --- | --- |']
        lines += [f"| {r['assumed_epochs_per_run']} | {r['training_runs']} | {r['hours_lower']:.1f}–{r['hours_upper']:.1f} | {r['days_lower']:.1f}–{r['days_upper']:.1f} |" for r in scenarios]
        lines += ['含10%最终评测/输出余量，首次Setup、数据拷贝和pilot另计；完整公式与逐轮来源SHA见JSON。',
                  'pilot按真实同配置逐轮耗时外推；GPU身份仍以原预检为准。',
                  '实际学习进展、早停轮数及是否达标未知；保留全部失败。']
        (destination/'耗时预算.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    return result
