"""N14在线俯瞰示意：只画合法自车、已知目标及当前CV预测，不读评测真值。"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont


SIZE = (1280, 720)
COLORS = {'ego': '#2563eb', 'target': '#c2410c', 'warning': '#9a6700', 'emergency': '#b91c1c'}


def render_overview(observation, decision, config, strategy):
    """一帧在线状态→PNG画布。道路为公开冻结地图，实车位置仅来自合法输入。"""
    image = Image.new('RGB', SIZE, '#f1f5f9')
    draw = ImageDraw.Draw(image)
    regular = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    title = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 26)
    cx, cy = config['conflict_xy'][0], -config['conflict_xy'][1]
    scale = 4.7

    def xy(x, y):
        return (int(420 + (x-cx-5)*scale), int(405-(y-cy+10)*scale))

    def route(points, color, width, dashed=False):
        ps = [xy(*p) for p in points]
        if len(ps)<2:return
        if dashed:
            for i in range(0,len(ps)-1,2):draw.line(ps[i:i+2], fill=color, width=width)
        else:draw.line(ps, fill=color, width=width)

    def vehicle(state, color):
        x,y,yaw=state['x'],state['y'],state['yaw'];c,s=math.cos(yaw),math.sin(yaw)
        corners=[xy(x+c*a-s*b,y+s*a+c*b) for a,b in
                 ((state['length']/2,state['width']/2),(state['length']/2,-state['width']/2),
                  (-state['length']/2,-state['width']/2),(-state['length']/2,state['width']/2))]
        draw.polygon(corners,fill=color,outline='#0f172a')

    # 仅用公开地图路线绘制道路底图，灰线不代表目标真实未来。
    for key in ('ego_route','target_route'):
        points=[[p['x'],-p['y']] for p in config[key]]
        route(points,'#cbd5e1',round(7*scale));route(points,'#ffffff',2)
    draw.rectangle((15,15,825,100), fill='#f8fafc')
    draw.text((28,24), '在线俯瞰示意 · 决策当时可知的信息',font=title,fill='#0f172a')
    draw.text((28,65),'蓝色：自车   橙色：已知目标   虚线：当前常速预测   灰色：地图',font=small,fill='#334155')
    ego=observation['ego'];now=observation['time_s']
    points=[[ego['x']+ego['vx']*i*.1,ego['y']+ego['vy']*i*.1] for i in range(31)]
    route(points,COLORS['ego'],3,True);vehicle(ego,COLORS['ego'])
    ex,ey=xy(ego['x'],ego['y']);draw.text((ex+14,ey+8),'本车',font=small,fill=COLORS['ego'])
    # N14固定单一横向目标：双来源估计不叠印标签，显示最紧迫估计；无风险时用最新估计。
    # 仅选择图形，不改动在线决策或其多来源保守风险计算。
    groups={}
    for target in decision['known_targets']:groups.setdefault(target['track_id'],[]).append(target)
    for i,target in enumerate(decision['known_targets']):
        source='路侧消息' if target['source_id']=='roadside-1' else '车端观测'
        draw.text((28,116+25*i),f"{source} · 数据年龄 {target['age_s']:.2f}s",font=small,fill=COLORS['target'])
    for estimates in groups.values():
        risk=[t for t in estimates if t['predicted_entry_s'] is not None]
        target=min(risk,key=lambda t:t['predicted_entry_s']) if risk else min(estimates,key=lambda t:t['age_s'])
        route(target['prediction_xy'],COLORS['target'],4,True);vehicle(target,COLORS['target'])
        x,y=xy(target['x'],target['y'])
        draw.text((x+14,y+10),'已知横向目标',font=small,fill=COLORS['target'])
        for i in (10,20,30):
            px,py=xy(*target['prediction_xy'][i]);draw.ellipse((px-3,py-3,px+3,py+3),fill=COLORS['target'])
            draw.text((px+5,py+2),f'+{i/10:g}s',font=small,fill=COLORS['target'])
    speed=math.hypot(ego['vx'],ego['vy']);budget=decision['stopping_budget_m']
    if speed>.1:
        end=[ego['x']+ego['vx']/speed*budget,ego['y']+ego['vy']/speed*budget]
        bx,by=xy(*end);draw.ellipse((bx-7,by-7,bx+7,by+7),outline='#7c3aed',width=3)
        draw.text((max(30,bx-180),by-48),'停车预算边界',font=small,fill='#7c3aed')
    draw.rectangle((840,0,1280,720), fill='#ffffff')
    branch={'reference':'参考行驶','local_aeb':'仅车端AEB','cooperative_aeb':'车路协同AEB'}[strategy]
    status=decision['mode'];color=COLORS['emergency'] if decision['brake'] else COLORS['warning'] if decision['warning'] else '#166534'
    draw.text((862,24),branch,font=title,fill='#0f172a')
    draw.rounded_rectangle((860,72,1260,122),radius=8,fill=color)
    draw.text((878,82),status,font=title,fill='white')
    number=lambda v,suffix='': '—' if v is None else f'{v:.2f}{suffix}'
    lines=[f'仿真时间  {now:.2f} s',f'当前速度  {speed:.2f} m/s',
           f"首次目标可用  {number(decision['first_target_available_s'],' s')}",
           f"首次风险预警  {number(decision['first_warning_s'],' s')}",
           f"紧急制动触发  {number(decision['first_emergency_s'],' s')}",
           f"预测包络接触  {number(decision['predicted_conflict_s'],' s后')}",
           f"至接触的行驶距离  {number(decision['distance_to_conflict_envelope_m'],' m')}",
           f'停车距离预算  {budget:.2f} m',f"距离余量 D−B  {number(decision['stopping_margin_m'],' m')}"]
    for i,line in enumerate(lines):draw.text((862,146+34*i),line,font=regular,fill='#0f172a')
    b=decision['braking_parameters']
    details=[f"实测减速度  {b['deceleration_mps2']:.3f} m/s²",
             f"执行延迟  {b['response_delay_s']:.2f}s + 周期{b['control_period_s']:.2f}s",
             f"安全余量  {b['margin_m']:.1f}m",'D > B：预警，保持参考行驶','D ≤ B：锁存紧急制动（前缀后）']
    for i,line in enumerate(details):draw.text((862,474+30*i),line,font=small,fill='#475569')
    draw.text((25,646),'预测为当前CV外推，不是车辆真实未来；未发现的目标不绘制',font=regular,fill='#0f172a')
    draw.text((25,681),'真实全场景与遮挡大车请看“俯瞰实景”；示意图不向控制注入信息',font=small,fill='#475569')
    draw.text((25,619),'多源重合时显示最紧迫估计；无冲突时显示最新估计',font=small,fill='#475569')
    draw.line((40,610,40+10*scale,610),fill='#0f172a',width=3)
    draw.text((40,578),'10 m · 右手系X向右、Y向上',font=small,fill='#334155')
    return image


class OverviewRecorder:
    def __init__(self, folder, config, strategy):
        self.folder=Path(folder);self.config=config;self.strategy=strategy
        self.images=self.folder/'实时俯瞰示意';self.images.mkdir()
        self.stream=(self.folder/'示意在线状态.jsonl').open('x',encoding='utf-8')
        self.records=[];self.snapshots=[];self.root=None;self.displayed=0;self.closed_by_user=False
        self.started=time.monotonic();self.events=set()
        if config.get('overview',{}).get('live_window',True):
            import tkinter as tk
            from PIL import ImageTk
            self.ImageTk=ImageTk;self.root=tk.Tk()
            self.root.title('OccluActiSafe · N14 在线俯瞰 · '+strategy)
            self.root.geometry('+20+20');self.label=tk.Label(self.root);self.label.pack()
            self.root.protocol('WM_DELETE_WINDOW',self.close_window)
            self.root.update()

    def close_window(self):
        self.closed_by_user=True
        if self.root is not None:self.root.destroy();self.root=None

    def update(self, observation, decision):
        index=round(observation['time_s']/.05)
        picture=render_overview(observation,decision,self.config,self.strategy)
        path=self.images/f'示意画面_帧{index:04d}.png';picture.save(path)
        self.records.append({'image':path.relative_to(self.folder).as_posix(),'source':'示意',
                             'index':index,'time_s':observation['time_s'],
                             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        row={'index':index,'time_s':observation['time_s'],'ego':observation['ego'],'decision':decision,
             'legal_observation_sha256':hashlib.sha256(json.dumps(observation,sort_keys=True).encode()).hexdigest()}
        self.stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.stream.flush()
        if self.root is not None:
            self.photo=self.ImageTk.PhotoImage(picture);self.label.configure(image=self.photo)
            self.root.update_idletasks();self.root.update();self.displayed+=1
            if self.root is None:return
            event=('首次紧急制动' if decision['brake'] else '首次风险预警' if decision['warning']
                   else '首次目标可用' if decision['known_targets'] else '初始画面')
            if event not in self.events:
                from PIL import ImageGrab
                self.root.lift();self.root.update()
                x,y=self.root.winfo_rootx(),self.root.winfo_rooty()
                shot=self.images/f'实时窗口_{event}.png'
                ImageGrab.grab(bbox=(x,y,x+self.root.winfo_width(),y+self.root.winfo_height())).save(shot)
                self.snapshots.append({'event':event,'time_s':observation['time_s'],'path':str(shot),
                                       'sha256':hashlib.sha256(shot.read_bytes()).hexdigest()})
                self.events.add(event)

    def close(self):
        if self.stream.closed:return
        self.stream.close()
        (self.folder/'示意帧索引.json').write_text(json.dumps(self.records,ensure_ascii=False,indent=2),encoding='utf-8')
        (self.folder/'实时示意窗口复核.json').write_text(json.dumps({'generated_frames':len(self.records),
            'displayed_frames':self.displayed,'closed_by_user':self.closed_by_user,'pid':os.getpid(),
            'wall_elapsed_s':time.monotonic()-self.started,'snapshots':self.snapshots,
            'source':'仅在线合法自车、已知目标和当前CV外推；不读评测真值'},ensure_ascii=False,indent=2),encoding='utf-8')
        if self.root is not None:self.root.destroy();self.root=None
