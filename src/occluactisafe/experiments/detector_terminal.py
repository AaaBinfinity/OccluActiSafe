"""N17纯终端交互：先选参数/显示摘要，明确确认后才调用既有执行器。"""
from pathlib import Path
import uuid

from .detector_local import ALL_MODELS,digest,selected_options
from .run_store import now,write_json

ACTIONS=[('check','环境与输入检查，不训练'),('plan','冻结本次计划，不训练'),
         ('probe','真实短探针，开发数据1轮'),('train','新任务正式训练，最多300轮'),
         ('resume','从同配方检查点续训')]


class SelectionCancelled(Exception):
    pass


def effective_options(opts):
    return {k:str(v) if isinstance(v,Path) else v for k,v in opts.items() if k!='options_source_bytes'}


def display_options(opts,*,dry_run=False,output=print):
    output('\n本次任务摘要'+('（仅预览，不执行）' if dry_run else ''))
    output(f"  模型：{opts['MODEL']}    seed：{opts['SEED']}    操作：{opts['ACTION']}")
    output(f"  SGD初始学习率lr0：{opts['LR0']:g}    batch/nbs：{opts['BATCH']}")
    output(f"  workers：{opts['WORKERS']}    CPU线程：{opts['CPU_THREADS'] or '框架默认'}    AMP：{opts['AMP']}")
    output(f"  原生验证batch上限：{opts['VALIDATION_BATCH'] or '框架默认'}    GPU：{opts['DEVICE']}")
    output(f"  上限：{opts['MAX_EPOCHS']}轮    最少：{opts['MIN_EPOCHS']}轮    早停耐心：{opts['PATIENCE']}轮")
    if opts['ACTION']=='probe':output('  probe实际使用真实开发子集1轮，不是完整数据训练或收敛证明。')
    if opts['ACTION'] in ('check','plan'):output('  本次操作不启动模型训练。')
    if opts['RESUME_RUN']:output('  续训来源：'+str(opts['RESUME_RUN']))
    output('  输出目录：'+str(opts['OUTPUT_ROOT']))
    output('  保留全部原始指标/中文图集；完整训练weights仅last.pt和best.pt。')


def choose_options(opts,*,dry_run=False,input_fn=None,output=print):
    reader=input if input_fn is None else input_fn
    def ask(label,default,parse):
        while True:
            text=reader(f'{label} [{default}]：').strip()
            if text.lower() in ('q','quit','exit','取消'):raise SelectionCancelled
            try:return parse(str(default) if not text else text)
            except (ValueError,KeyError,IndexError) as exc:output('输入无效，请重新输入：'+str(exc))
    def model(text):
        names=[m['family'] for m in ALL_MODELS]
        if text.isdigit():
            index=int(text)-1
            if not 0<=index<len(names):raise ValueError('型号编号须为1～15')
            return names[index]
        found={n.lower():n for n in names}
        if text.lower().removesuffix('.pt') not in found:raise ValueError('请输入编号或完整型号名')
        return found[text.lower().removesuffix('.pt')]
    def integer(text):
        v=int(text)
        if not 0<=v<2**32:raise ValueError('seed须为0～4294967295整数')
        return v
    def rate(text):
        v=float(text)
        import math
        if not math.isfinite(v) or not 0<v<=1:raise ValueError('学习率须大于0、不超过1且为有限值')
        return v
    def batch(text):
        v=int(text)
        if v<1:raise ValueError('batch须为正整数')
        return v
    def action(text):
        if text.isdigit():
            i=int(text)-1
            if not 0<=i<len(ACTIONS):raise ValueError('操作编号须为1～5')
            return ACTIONS[i][0]
        aliases={'检查':'check','计划':'plan','探针':'probe','训练':'train','续训':'resume'}
        v=aliases.get(text,text.lower())
        if v not in dict(ACTIONS):raise ValueError('请选择列出的操作')
        return v
    def resume_path(text):
        from occluactisafe.common.config import PROJECT_ROOT
        p=Path(text.strip('"')).expanduser()
        p=(p if p.is_absolute() else PROJECT_ROOT/p).resolve()
        if not (p/'本机训练任务.json').is_file():raise ValueError('目录中没有本机训练任务.json')
        return str(p)
    try:
        output('N17训练任务选择｜输入q或Ctrl+C随时取消；回车使用方括号内默认值')
        for start in range(0,len(ALL_MODELS),3):
            output('   '.join(f"{i+1:2d}. {ALL_MODELS[i]['family']:<9}" for i in range(start,min(start+3,len(ALL_MODELS)))))
        chosen_model=ask('模型编号或名称',opts['MODEL'],model)
        current=selected_options(opts,{'MODEL':chosen_model,'ACTION':'check'})
        seed=ask('seed（正式比较通常17/29/43）',current['SEED'],integer)
        lr=ask('SGD初始学习率lr0（例如0.001或0.0003）',f"{current['LR0']:g}",rate)
        size=ask('batch（大尺寸起点未实测，建议先probe）',current['BATCH'],batch)
        for i,(name,label) in enumerate(ACTIONS,1):output(f'  {i}. {name}：{label}')
        selected_action=ask('运行操作',1,action)
        changes={'MODEL':chosen_model,'SEED':seed,'LR0':lr,'BATCH':size,'ACTION':selected_action}
        if selected_action=='resume':
            changes['RESUME_RUN']=ask('原中断任务目录',current['RESUME_RUN'] or '',resume_path)
        result=selected_options(current,changes)
        display_options(result,dry_run=dry_run,output=output)
        if reader('输入 y 确认执行，其他输入取消 [N]：').strip().lower() not in ('y','yes','是','确认'):
            raise SelectionCancelled
        return result
    except (SelectionCancelled,EOFError,KeyboardInterrupt):
        output('\n已取消，未启动预检或训练。')
        return None


def execute_options(opts,runner,mode):
    chosen=effective_options(opts)
    selection={'mode':mode,'confirmed_at':now(),'effective_options':chosen,
               'effective_options_sha256':digest(chosen),'defaults_file_sha256':opts['options_sha256'],
               'status':'confirmed'}
    directory=opts['OUTPUT_ROOT']/'03_验收证据'/('n17_terminal_selection_'+uuid.uuid4().hex)
    directory.mkdir(parents=True)
    path=directory/'终端任务选择.json';write_json(path,selection)
    try:
        result=runner(opts)
        selection.update(status='completed',result_directory=str(result),finished_at=now())
        write_json(path,selection)
        print('本次有效参数已保存：'+str(path),flush=True)
        return result
    except BaseException as exc:
        selection.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                         error=repr(exc),finished_at=now())
        write_json(path,selection)
        raise
