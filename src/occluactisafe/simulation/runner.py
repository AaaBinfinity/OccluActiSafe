"""唯一同步世界持有者：严格顺序请求与本次actor清理。"""
from threading import Lock
import json
import math
from pathlib import Path
import time
from occluactisafe.common.schemas import AdvanceRequest, TickEvent


class SynchronousRunner:
    @classmethod
    def reload_synchronously(cls,client,**kwargs):
        """在重载前固定步长，避免首帧异步物理破坏跨分支重复性。"""
        previous=client.get_world();original=previous.get_settings()
        settings=previous.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
        settings.substepping=True;settings.max_substeps=10;settings.max_substep_delta_time=.01
        previous.apply_settings(settings)
        try:
            world=client.reload_world(False)
            runner=cls(world,**kwargs)
            # 回收新世界自建actors后恢复本次调用前的设置。
            runner._original=original
            return runner
        except BaseException:
            client.get_world().apply_settings(original)
            raise

    def __init__(self, world, *, run_id, episode_id, generation, clock_domain, fixed_delta=.05, destroy_owned=None, cleanup_path=None,
                 reconnect=None, destroy_command=None, prepare_cleanup=None, recovery_budget_s=12., cleanup_rpc_s=.5):
        if not all(isinstance(s,str) and s.strip() for s in (run_id,episode_id,clock_domain)): raise ValueError('身份不能为空')
        if type(generation) is not int or generation < 0 or fixed_delta != .05: raise ValueError('generation或固定步长非法')
        self._world=world; self.run_id=run_id; self.episode_id=episode_id; self.generation=generation; self.clock_domain=clock_domain
        self._original=world.get_settings(); self._owned=[]; self._lock=Lock(); self._closed=False; self._failed=False
        self._destroy_owned=destroy_owned
        self._cleanup_path=Path(cleanup_path) if cleanup_path is not None else None
        self._cleanup_trigger=None
        if not 0 < cleanup_rpc_s <= recovery_budget_s <= 12: raise ValueError('清理期限非法')
        self._reconnect=reconnect; self._destroy_command=destroy_command; self._prepare_cleanup=prepare_cleanup
        self._recovery_budget_s=recovery_budget_s; self._cleanup_rpc_s=cleanup_rpc_s
        self._world_id=getattr(world,'id',None); self._role='oas:'+run_id
        self._spectator_original=None
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
        ego_bp=candidates[0]
        ego_bp.set_attribute('role_name',self._role)
        ego=None
        for transform in self._world.get_map().get_spawn_points():
            ego=self._world.try_spawn_actor(ego_bp,transform)
            if ego is not None: break
        if ego is None: raise RuntimeError('没有可用的车辆生成点')
        # 新生成车辆尚未注册自动驾驶。0.9.16 的 set_autopilot(False) 也会启动 TM；
        # 其后台线程会在服务端失联时触发原生进程终止，因此此处保持默认手动控制。
        self.register_owned(ego)
        camera_bp=blueprints.find('sensor.camera.rgb')
        camera_bp.set_attribute('role_name',self._role)
        camera_bp.set_attribute('image_size_x','320'); camera_bp.set_attribute('image_size_y','180')
        camera_bp.set_attribute('sensor_tick','0.05')
        camera=self._world.spawn_actor(camera_bp,carla.Transform(carla.Location(x=1.5,z=2.4)),attach_to=ego)
        self.register_owned(camera); camera.listen(lambda image: frames.put(image.frame,image))
        return ego

    def register_owned(self, actor):
        if self._closed: raise RuntimeError('runner已关闭')
        self._owned.append(actor)

    def spawn_preview_camera(self,carla,ego,frames):
        """独立720p跟车显示相机，核心320×180相机保持原配置。"""
        blueprint=self._world.get_blueprint_library().find('sensor.camera.rgb')
        for name,value in {'role_name':self._role,'image_size_x':'1280','image_size_y':'720','sensor_tick':'0.05'}.items():
            blueprint.set_attribute(name,value)
        camera=self._world.spawn_actor(blueprint,carla.Transform(carla.Location(x=-7,z=4),carla.Rotation(pitch=-18)),attach_to=ego)
        self.register_owned(camera);camera.listen(lambda image:frames.put(image.frame,image))

    def follow_ego(self,carla,ego):
        """仅供可视诊断；自车位姿不送入核心，旁观者不推进世界。"""
        spectator=self._world.get_spectator()
        if self._spectator_original is None:self._spectator_original=spectator.get_transform()
        transform=ego.get_transform();forward=transform.get_forward_vector();location=transform.location
        spectator.set_transform(carla.Transform(carla.Location(x=location.x-7*forward.x,y=location.y-7*forward.y,z=location.z+4),
                                                carla.Rotation(pitch=-18,yaw=transform.rotation.yaw)))

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

    def _save_cleanup(self, report):
        if self._cleanup_path is None: return
        try:
            temporary=self._cleanup_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(self._cleanup_path)
        except OSError as exc:
            # 证据盘写入故障不能阻止 actor 清理；最终仍明确报告失败。
            report.setdefault('evidence_errors',[]).append(str(exc))

    def _close_with_recovery(self):
        started=time.monotonic(); deadline=started+self._recovery_budget_s
        self._closed=True
        report={'status':'in_progress','trigger':self._cleanup_trigger,'world_id':self._world_id,
                'actors':[],'settings_restored':False,'attempts':[],'budget_s':self._recovery_budget_s}
        report['spectator_restored']=None if self._spectator_original is None else False
        report['owned']=[{'actor_id':a.id,'type_id':a.type_id,'role_name':self._role} for a in self._owned]
        self._save_cleanup(report)
        if self._prepare_cleanup: self._prepare_cleanup(self._cleanup_rpc_s)
        for actor in reversed(self._owned):
            try:
                if hasattr(actor,'stop'): actor.stop()
            except Exception as exc: report.setdefault('stop_errors',[]).append(str(exc))
        settings_keys=('synchronous_mode','fixed_delta_seconds','substepping','max_substeps',
                       'max_substep_delta_time','no_rendering_mode','max_culling_distance',
                       'deterministic_ragdolls','tile_stream_distance','actor_active_distance','spectator_as_ego')
        expected={key:getattr(self._original,key) for key in settings_keys if hasattr(self._original,key)}
        completed=set()
        while time.monotonic() < deadline:
            attempt={'elapsed_s':time.monotonic()-started}; report['attempts'].append(attempt)
            try:
                client=self._reconnect(min(self._cleanup_rpc_s,deadline-time.monotonic()))
                def check_time():
                    left=deadline-time.monotonic()
                    if left<=0: raise TimeoutError('恢复收尾期限已到')
                    client.set_timeout(min(self._cleanup_rpc_s,left))
                check_time(); world=client.get_world()
                check_time(); world.get_settings()  # 强制真实 RPC，不能只使用缓存的世界 ID。
                attempt['observed_world_id']=world.id
                if self._world_id is None or world.id != self._world_id:
                    report['reason']='world_changed'; break
                pending=[]
                for owned in reversed(self._owned):
                    if owned.id in completed: continue
                    check_time(); actor=world.get_actor(owned.id)
                    if actor is not None and (actor.type_id != owned.type_id or actor.attributes.get('role_name') != self._role):
                        attempt['conflict']={'actor_id':owned.id,'observed_type':actor.type_id,
                                             'observed_role':actor.attributes.get('role_name'),'expected_role':self._role}
                        report['reason']='ownership_mismatch'; break
                    pending.append((owned.id,actor))
                if report.get('reason'): break
                if self._spectator_original is not None:
                    check_time();world.get_spectator().set_transform(self._spectator_original)
                    report['spectator_restored']=True
                # 全部对象归属预检完成后才允许修改服务端。
                for actor_id,actor in pending:
                    if actor is not None:
                        check_time(); response=client.apply_batch_sync([self._destroy_command(actor_id)],False)[0]
                        if response.error: raise RuntimeError(response.error)
                    completed.add(actor_id)
                    report['actors'].append({'actor_id':actor_id,'destroyed':True,'already_absent':actor is None})
                    self._save_cleanup(report)
                check_time(); world.apply_settings(self._original)
                check_time(); actual=world.get_settings()
                def equal(a,b):
                    return math.isclose(a,b,rel_tol=0,abs_tol=1e-6) if type(a) is float and type(b) is float else a==b
                if any(not equal(value,getattr(actual,key)) for key,value in expected.items()):
                    raise RuntimeError('原设置恢复后复核不一致')
                report['settings_restored']=True; report['status']='completed'; break
            except Exception as exc:
                attempt['error']=str(exc); self._save_cleanup(report)
                left=deadline-time.monotonic()
                if left>0: time.sleep(min(.25,left))
        if report['status']!='completed':
            report['status']='failed'; report.setdefault('reason','recovery_deadline')
        report['elapsed_s']=time.monotonic()-started
        self._save_cleanup(report)
        if report.get('evidence_errors') or report['status']!='completed':
            raise RuntimeError('清理收尾失败：'+report.get('reason','evidence_write_failed'))

    def close(self):
        with self._lock:
            if self._closed: return
            if self._reconnect is not None: return self._close_with_recovery()
            failures=[]
            if self._spectator_original is not None:
                try:self._world.get_spectator().set_transform(self._spectator_original)
                except Exception as exc:failures.append(str(exc))
            report={'status':'in_progress','trigger':self._cleanup_trigger,'actors':[], 'settings_restored':False}
            def save():
                self._save_cleanup(report)
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
