"""FIFO发送服务、到达最小堆和身份绑定扰动的进程内信道。"""

from collections import Counter
import heapq
import hashlib
import math
import time

from occluactisafe.common.clock import Timestamp
from occluactisafe.common.errors import ContractError
from occluactisafe.common.schemas import ReceivedMessage
from .protocol import (
    ChannelConfig, Disturbance, check_direction, decode_message, disturbance_for,
    encode_message, message_key, service_seconds, target_count,
)


class SimChannel:
    def __init__(self, config: ChannelConfig, *, perturbations=None, event_sink=None):
        if not isinstance(config, ChannelConfig):
            raise ValueError("config须为ChannelConfig")
        self.config = config
        self._perturbations = dict(perturbations or {})
        for key, value in self._perturbations.items():
            if (not isinstance(key, tuple) or len(key) != 3
                    or any(not isinstance(v, str) or not v for v in key[:2])
                    or type(key[2]) is not int or key[2] < 0 or not isinstance(value, Disturbance)):
                raise ValueError("固定扰动表身份或值非法")
        self._heap = []
        self._counter = 0
        self._now = None
        self._service_end = 0.0
        self._event_sink = event_sink
        self.events = []
        self.stats = Counter()

    @property
    def pending_count(self):
        return len(self._heap)

    def _advance(self, now):
        if not isinstance(now, Timestamp) or now.clock_domain != self.config.clock_domain:
            raise ContractError("信道时钟域不一致")
        if now.seconds < 0 or (self._now is not None and now.seconds < self._now):
            raise ContractError("信道时钟不得为负或倒退")
        self._now = now.seconds

    def _record(self, event, **fields):
        row = {"event": event, "direction": self.config.direction, **fields}
        if self._event_sink is None:
            self.events.append(row)
        else:
            self._event_sink(row)

    def send(self, payload, now: Timestamp) -> bool:
        """返回是否已安排交付；已接纳的无线丢包同样占用发送服务。"""
        started = time.perf_counter()
        self._advance(now)
        self.stats["send_attempts"] += 1
        try:
            raw = payload if isinstance(payload, bytes) else encode_message(payload)
            if len(raw) > self.config.max_message_bytes:
                raise ContractError("消息超过max_message_bytes")
            message = decode_message(raw)  # 冻结不可变线上内容，重验字段。
            check_direction(message, self.config.direction)
            if message.clock_domain != self.config.clock_domain or message.transmitted_time != now.seconds:
                raise ContractError("消息声明的发送时刻/时钟域与调度器不一致")
        except (ValueError, TypeError) as exc:
            self.stats["schema_rejected"] += 1
            self._record("send_rejected", time=now.seconds, reason="invalid_message", detail=str(exc))
            raise
        key = message_key(message)
        byte_count = len(raw)
        count = target_count(message)
        base = {"time": now.seconds, "source_id": key[0], "session_id": key[1], "sequence": key[2],
                "message_type": message.message_type, "bytes": byte_count, "targets": count,
                "payload_sha256": hashlib.sha256(raw).hexdigest()}
        self.stats["offered_bytes"] += byte_count
        self.stats["offered_targets"] += count
        d = self._perturbations.get(key, disturbance_for(self.config, key))
        service = service_seconds(byte_count, self.config.capacity_bps)
        service_start = max(now.seconds, self._service_end)
        queue_wait = service_start - now.seconds
        end = service_start + service
        arrival = end + d.propagation_delay_s
        usable = arrival + self.config.processing_delay_s
        if not all(math.isfinite(v) for v in (end, arrival, usable)):
            raise ValueError("信道时间溢出")
        base.update(queue_wait_s=queue_wait, service_s=service, service_start=service_start,
                    service_end=end, propagation_s=d.propagation_delay_s, arrival_time=arrival,
                    processing_s=self.config.processing_delay_s, usable_time=usable)
        if len(self._heap) >= self.config.max_pending_messages:
            reason = "queue_full"
        elif queue_wait > self.config.max_queue_wait_s:
            reason = "queue_timeout"
        elif usable > message.valid_until:
            reason = "deadline_unreachable"
        else:
            reason = None
        if reason:
            self.stats[reason] += 1
            self._record("send_rejected", **base, reason=reason,
                         encode_schedule_wall_s=time.perf_counter() - started)
            return False
        self._service_end = end
        blackout = any(start <= now.seconds < stop for start, stop in self.config.blackout_intervals)
        loss_reason = "blackout" if blackout else ("random_loss" if d.lost else None)
        record = {**base, "loss_reason": loss_reason, "counter": self._counter}
        heapq.heappush(self._heap, (arrival, self._counter, raw, record))
        self._counter += 1
        self.stats["admitted_messages"] += 1
        self.stats["transmitted_bytes"] += byte_count
        self.stats["transmitted_targets"] += count
        self.stats["service_seconds"] += service
        self.stats["max_pending"] = max(self.stats["max_pending"], len(self._heap))
        if loss_reason:
            self.stats[loss_reason] += 1
        self._record("scheduled", **record, encode_schedule_wall_s=time.perf_counter() - started)
        return loss_reason is None

    def deliver(self, now: Timestamp) -> list[ReceivedMessage]:
        self._advance(now)
        ready = []
        while self._heap and self._heap[0][0] <= now.seconds:
            _, _, raw, row = heapq.heappop(self._heap)
            if row["loss_reason"]:
                self._record("dropped", **row, poll_time=now.seconds, reason=row["loss_reason"])
                continue
            started = time.perf_counter()
            message = decode_message(raw)
            received = ReceivedMessage(message, row["arrival_time"], row["usable_time"], now.clock_domain)
            elapsed = time.perf_counter() - started
            ready.append(received)
            self.stats["delivered_messages"] += 1
            self.stats["delivered_bytes"] += len(raw)
            self.stats["delivered_targets"] += target_count(message)
            self.stats["decode_wall_seconds"] += elapsed
            self.stats["injected_processing_seconds"] += self.config.processing_delay_s
            self._record("arrived", **row, poll_time=now.seconds, decode_wall_s=elapsed)
        return ready
