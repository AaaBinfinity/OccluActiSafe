"""唯一同步世界持有者：严格顺序请求与本次actor清理。"""
from threading import Lock
from occluactisafe.common.schemas import AdvanceRequest, TickEvent


class SynchronousRunner:
    def __init__(self, world, *, run_id, episode_id, generation, clock_domain, fixed_delta=.05):
        if not all(isinstance(s,str) and s.strip() for s in (run_id,episode_id,clock_domain)): raise ValueError('身份不能为空')
        if type(generation) is not int or generation < 0 or fixed_delta != .05: raise ValueError('generation或固定步长非法')
        self._world=world; self.run_id=run_id; self.episode_id=episode_id; self.generation=generation; self.clock_domain=clock_domain
        self._original=world.get_settings(); self._owned=[]; self._lock=Lock(); self._closed=False; self._failed=False
        snapshot=world.get_snapshot(); self.frame=snapshot.frame; self.sim_time=snapshot.timestamp.elapsed_seconds
        settings=world.get_settings(); settings.synchronous_mode=True; settings.fixed_delta_seconds=fixed_delta
        try: world.apply_settings(settings)
        except BaseException:
            world.apply_settings(self._original); raise

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
            for actor in reversed(self._owned):
                try: actor.destroy()
                except Exception as exc: failures.append(str(exc))
            try: self._world.apply_settings(self._original)
            except Exception as exc: failures.append(str(exc))
            self._closed=True
            if failures: raise RuntimeError('清理或恢复设置失败：'+'; '.join(failures))

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
