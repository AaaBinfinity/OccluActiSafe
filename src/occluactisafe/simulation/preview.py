"""N12 合法 RGB 的只读桌面预览与硬盘截图，不导入 CARLA。"""
import base64
import hashlib
import json
import os
from pathlib import Path
import time
from PIL import Image, ImageDraw


class FrameArchive:
    def __init__(self, output, *, interval=50,native_resolution=(320,180)):
        if type(interval) is not int or interval<1: raise ValueError('截图间隔须为正整数')
        self.output=Path(output); self.interval=interval
        self.records=(self.output/'预览帧记录.jsonl').open('x',encoding='utf-8')
        self.images=[];self.last=None;self.count=0;self.valid=0;self._invalid_saved=False;self._first_valid=False
        self.started=time.monotonic()
        self.native_resolution=tuple(native_resolution)

    def update(self,event,camera,control):
        # 输入来自 decode_tick，仍检查帧号，防止把旧画面标为当前帧。
        if camera['frame']!=event.frame_id:raise ValueError('预览帧不一致')
        self.count+=1;raw=None
        if camera['valid']:
            raw=camera.get('raw_bgra')
            if raw is None:raw=base64.b64decode(camera['bgra'],validate=True)
            if (camera['width'],camera['height'])!=self.native_resolution or len(raw)!=self.native_resolution[0]*self.native_resolution[1]*4:raise ValueError('图像尺寸或长度异常')
            picture=Image.frombytes('RGBA',self.native_resolution,raw,'raw','BGRA').convert('RGB');self.valid+=1
        else:
            picture=Image.new('RGB',self.native_resolution,'#252b36')
            ImageDraw.Draw(picture).text((20,75),'INVALID FRAME: '+str(camera['reason']),fill='white')
        row={'frame':event.frame_id,'sim_time':event.sim_time,'valid':camera['valid'],'reason':camera['reason'],
             'wall_elapsed_s':time.monotonic()-self.started,'raw_bgra_sha256':hashlib.sha256(raw).hexdigest() if raw else None,
             'camera_base64_sha256':(camera.get('transport_sha256') or hashlib.sha256(camera['bgra'].encode()).hexdigest()) if raw else None,
             'encoding':camera.get('encoding','base64_bgra'),'native_resolution':self.native_resolution,
             'control':{key:getattr(control,key) for key in ('throttle','brake','steer')}}
        self.records.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.records.flush()
        self.last=(picture,row)
        first_valid=camera['valid'] and not self._first_valid
        first_invalid=not camera['valid'] and not self._invalid_saved
        if self.count==1 or first_valid or first_invalid or self.count%self.interval==0:self.save('自动')
        self._first_valid=self._first_valid or camera['valid'];self._invalid_saved=self._invalid_saved or not camera['valid']
        return picture,row

    def save(self,reason='手动'):
        if self.last is None:return
        picture,row=self.last
        name=f'相机画面_第{row["frame"]:08d}帧_{reason}_{len(self.images)+1:03d}.png'
        path=self.output/name;picture.save(path)
        self.images.append({**row,'image':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'save_reason':reason,'kind':'camera_rgb' if row['valid'] else 'invalid_placeholder'})
        manifest={'source':'预览帧记录.jsonl','source_sha256':hashlib.sha256((self.output/'预览帧记录.jsonl').read_bytes()).hexdigest(),
                  'images':self.images,'native_resolution':self.native_resolution,'smoothing':False}
        (self.output/'预览图片来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

    def close(self):
        if not self.records.closed:self.save('结束');self.records.close()


class LivePreview:
    def __init__(self,output,*,interval=50):
        import tkinter as tk
        from PIL import ImageTk
        self.tk=tk;self.ImageTk=ImageTk;self.output=Path(output)
        self.root=tk.Tk();title='OccluActiSafe · N12 实时预览 · '+self.output.name;self.root.title(title)
        self.root.geometry('+40+40')
        self.root.configure(bg='#111827');self.root.resizable(False,False)
        self.root.protocol('WM_DELETE_WINDOW',self.request_stop)
        tk.Label(self.root,text='OccluActiSafe  仿真观察',font=('Microsoft YaHei UI',18,'bold'),bg='#111827',fg='white').pack(pady=(12,4))
        tk.Label(self.root,text='N12 同步诊断 · 车辆保持制动 · 独立跟车相机 1280×720',bg='#111827',fg='#a7b4c9').pack()
        self.photo=ImageTk.PhotoImage(Image.new('RGB',(960,540),'#252b36'))
        self.image_label=tk.Label(self.root,bg='#252b36',image=self.photo,width=960,height=540);self.image_label.pack(padx=16,pady=10)
        self.status=tk.Label(self.root,text='正在连接 CARLA…',font=('Microsoft YaHei UI',11),bg='#111827',fg='#a7b4c9')
        self.status.pack(pady=4)
        buttons=tk.Frame(self.root,bg='#111827');buttons.pack(pady=(4,12))
        tk.Button(buttons,text='保存画面 (F12)',command=lambda:self.archive.save(),width=18).pack(side='left',padx=8)
        self.stop=tk.Button(buttons,text='停止并清理 (Esc)',command=self.request_stop,width=18);self.stop.pack(side='left',padx=8)
        self.archive=FrameArchive(output,interval=interval,native_resolution=(1280,720));self.stop_requested=False;self.destroyed=False;self.complete=False
        self.next_frame=time.monotonic();self.render_count=0;self.pump()
        self.root.bind('<F12>',lambda _:self.archive.save());self.root.bind('<Escape>',lambda _:self.request_stop())
        (self.output/'预览窗口.json').write_text(json.dumps({'title':title,'window_id':self.root.winfo_id(),'pid':os.getpid()},ensure_ascii=False),encoding='utf-8')

    def pump(self):
        if self.destroyed:return
        try:self.root.update_idletasks();self.root.update()
        except self.tk.TclError:self.destroyed=True;self.stop_requested=True

    def request_stop(self):
        if self.complete:
            self.root.destroy();self.destroyed=True
        else:
            self.stop_requested=True;self.status.configure(text='正在结束本次运行并清理，请稍候…',fg='#fbbf24')

    def pace(self):
        while time.monotonic()<self.next_frame and not self.stop_requested:
            self.pump();time.sleep(min(.01,max(0,self.next_frame-time.monotonic())))
        self.next_frame=time.monotonic()+.05;self.pump()

    def update(self,event,camera,control,total):
        picture,row=self.archive.update(event,camera,control)
        if self.destroyed:return
        self.photo=self.ImageTk.PhotoImage(picture.resize((960,540),Image.Resampling.LANCZOS))
        self.image_label.configure(image=self.photo,width=960,height=540)
        state='图像有效' if row['valid'] else '图像无效：'+str(row['reason'])
        self.status.configure(text=f'帧 {event.frame_id}  |  仿真 {event.sim_time:.2f} s  |  {self.archive.count}/{total}  |  {state}',
                              fg='#6ee7b7' if row['valid'] else '#fbbf24')
        self.render_count+=1;self.pump()

    def finish(self,message,*,hold_seconds=None):
        self.complete=True;self.archive.close()
        result={'message':message,'displayed_frames':self.render_count,'received_frames':self.archive.count,
                'valid_frames':self.archive.valid,'stopped_by_user':self.stop_requested,
                'wall_elapsed_s':time.monotonic()-self.archive.started,'image_count':len(self.archive.images)}
        (self.output/'预览结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        if self.destroyed:return
        self.status.configure(text=message+'；画面已结束，截图保存在本次运行目录',fg='#a7b4c9');self.stop.configure(text='关闭窗口')
        if self.stop_requested:self.root.destroy();self.destroyed=True;return
        if hold_seconds is None:self.root.mainloop()
        else:
            deadline=time.monotonic()+hold_seconds
            while not self.destroyed and time.monotonic()<deadline:self.pump();time.sleep(.02)
            if not self.destroyed:self.root.destroy();self.destroyed=True
