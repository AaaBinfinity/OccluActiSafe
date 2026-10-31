"""唯一同步世界持有者：严格顺序请求与本次actor清理。"""
from threading import Lock
import json
from pathlib import Path
from occluactisafe.common.schemas import AdvanceRequest, TickEvent


class SynchronousRunner:
    def __init__(self, world, *, run_id, episode_id, generation, clock_domain, fixed_delta=.05, destroy_owned=None, cleanup_path=None):
        if not all(isinstance(s,str) and s.strip() for s in (run_id,episode_id,clock_domain)): raise ValueError('身份不能为空')
        if type(generation) is not int or generation < 0 or fixed_delta != .05: raise ValueError('generation或固定步长非法')
        self._world=world; self.run_id=run_id; self.episode_id=episode_id; self.generation=generation; self.clock_domain=clock_domain
        self._original=world.get_settings(); self._owned=[]; self._lock=Lock(); self._closed=False; self._failed=False
        self._destroy_owned=destroy_owned
        self._cleanup_path=Path(cleanup_path) if cleanup_path is not None else None
        self._cleanup_trigger=None
        snapshot=world.get_snapshot(); self.frame=snapshot.frame; self.sim_time=snapshot.timestamp.elapsed_seconds
        settings=world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=fixed_delta
        if hasattr(settings,'substepping'):
            settings.substepping=True; settings.max_substeps=10; settings.max_substep_delta_time=.01
        try:
            world.apply_settings(settings)
            snapshot=world.get_snapshot(); self.frame=snapshot.frame; self.sim_time=snapshot.timestamp.elapsed_seconds
        except BaseException:
            world.apply_settings(self._original); raise

    def spawn_ego_and_camera(self, carla, frames):
        """只创建并登记本运行的车辆和摄像头，失败时由 close 回收。"""
        blueprints=self._world.get_blueprint_library()
        candidates=blueprints.filter('vehicle.*')
        if not candidates: raise RuntimeError('地图没有车辆蓝图')
        ego=None
        for transform in self._world.get_map().get_spawn_points():
            ego=self._world.try_spawn_actor(candidates[0],transform)
            if ego is not None: break
        if ego is None: raise RuntimeError('没有可用的车辆生成点')
        # 新生成车辆尚未注册自动驾驶。0.9.16 的 set_autopilot(False) 也会启动 TM；
        # 其后台线程会在服务端失联时触发原生进程终止，因此此处保持默认手动控制。
        self.register_owned(ego)
        camera_bp=blueprints.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x','320'); camera_bp.set_attribute('image_size_y','180')
        camera_bp.set_attribute('sensor_tick','0.05')
        camera=self._world.spawn_actor(camera_bp,carla.Transform(carla.Location(x=1.5,z=2.4)),attach_to=ego)
        self.register_owned(camera); camera.listen(lambda image: frames.put(image.frame,image))
        return ego

    def register_owned(self, actor):
        if self._closed: raise RuntimeError('runner已关闭')
        self._owned.append(actor)

    def advance(self, request_json, *, apply_control):
        request=AdvanceRequest.from_json(request_json)
        with self._lock:
            if self._closed or self._failed: raise RuntimeError('runner已关闭或状态未知')
            if (request.run_id,request.episode_id,request.generation,request.clock_domain)!=(self.run_id,self.episode_id,self.generation,self.clock_domain):
                raise ValueError('运行身份/代次/时钟不匹配')
            if request.target_frame_id != self.frame+1: raise ValueError('重复、乱序或非下一帧请求')
            if not request.request_time <= self.sim_time <= request.valid_until: raise ValueError('请求未生效或已过期')
            try:
                apply_control(request.control)
                frame=self._world.tick(); snapshot=self._world.get_snapshot()
                if frame != request.target_frame_id or snapshot.frame != frame or snapshot.timestamp.elapsed_seconds <= self.sim_time:
                    raise RuntimeError('世界帧/时间异常，停止继续推进')
                self.frame=frame; self.sim_time=snapshot.timestamp.elapsed_seconds
            except BaseException:
                self._failed=True; raise
            return TickEvent('1.0',self.run_id,self.episode_id,self.generation,self.frame,self.clock_domain,self.sim_time,()).to_json()

    def close(self):
        with self._lock:
            if self._closed: return
            failures=[]
            report={'status':'in_progress','trigger':self._cleanup_trigger,'actors':[], 'settings_restored':False}
            def save():
                if self._cleanup_path is not None:
                    temporary=self._cleanup_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                    temporary.replace(self._cleanup_path)
            save()
            for actor in reversed(self._owned):
                entry={'actor_id':getattr(actor,'id',None),'destroyed':False,'errors':[]}
                try:
                    if hasattr(actor,'stop'): actor.stop()
                except Exception as exc: failures.append(str(exc)); entry['errors'].append(str(exc))
                try:
                    destroyed = self._destroy_owned(actor) if self._destroy_owned is not None else actor.destroy()
                    if destroyed is False: raise RuntimeError('actor 销毁返回失败')
                    entry['destroyed']=True
                except Exception as exc: failures.append(str(exc)); entry['errors'].append(str(exc))
                report['actors'].append(entry); save()
            try:
                self._world.apply_settings(self._original); report['settings_restored']=True
            except Exception as exc: failures.append(str(exc))
            self._closed=True
            report['status']='failed' if failures else 'completed'; report['errors']=failures; save()
            if failures: raise RuntimeError('清理或恢复设置失败：'+'; '.join(failures))

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        self._cleanup_trigger=None if exc is None else f'{exc_type.__name__}: {exc}'
        self.close()
