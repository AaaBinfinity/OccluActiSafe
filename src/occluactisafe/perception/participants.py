"""车辆/骑行者OBB与行人圆分派，拒绝猜测尺寸或参考点。"""

from dataclasses import dataclass
from occluactisafe.common.coordinates import Pose2D, shift_reference_point
from occluactisafe.common.schemas import SourceTrack
from occluactisafe.risk.obb import OBB, Circle, real


@dataclass(frozen=True)
class ParticipantGeometry:
    category: str
    length: float | None = None
    width: float | None = None
    radius: float | None = None
    margin: float = 0.
    reference_point: str = "geometric_center"
    center_offset_m: float | None = None

    def __post_init__(self):
        if self.category not in ("vehicle", "cyclist", "pedestrian"): raise ValueError("未知参与者几何类别")
        real(self.margin, "margin", nonnegative=True)
        if self.category == "pedestrian":
            real(self.radius, "pedestrian_radius", positive=True)
            if self.length is not None or self.width is not None: raise ValueError("行人圆不套用车辆长宽")
            if self.reference_point != "geometric_center": raise ValueError("行人圆必须声明中心")
        else:
            real(self.length, "length", positive=True); real(self.width, "width", positive=True)
            if self.radius is not None: raise ValueError("车辆/骑行者使用声明长宽OBB")
        if self.reference_point not in ("geometric_center", "rear_axle"): raise ValueError("未知参考点")
        if self.reference_point == "rear_axle": real(self.center_offset_m, "center_offset_m")
        elif self.center_offset_m is not None: raise ValueError("中心参考不重复施加后轴偏移")

    @property
    def motion_radius(self):
        base = self.radius + self.margin if self.category == "pedestrian" else OBB((0, 0), 0, self.length, self.width, self.margin).radius
        return base + abs(self.center_offset_m or 0.)

    def at(self, pose: Pose2D):
        if self.reference_point == "rear_axle": pose = shift_reference_point(pose, self.center_offset_m)
        if self.category == "pedestrian": return Circle((pose.x, pose.y), self.radius, self.margin)
        return OBB((pose.x, pose.y), pose.yaw, self.length, self.width, self.margin)


def geometry_from_track(track: SourceTrack, *, pedestrian_radius=None, margin=0., center_offset_m=None):
    if not isinstance(track, SourceTrack): raise ValueError("需要N02 SourceTrack")
    track.to_dict()
    if not track.state_valid_mask: raise ValueError("目标位置无效")
    if track.category == "pedestrian":
        return ParticipantGeometry("pedestrian", radius=pedestrian_radius, margin=margin, reference_point=track.reference_point)
    if not track.size_valid or not track.yaw_valid: raise ValueError("OBB目标缺少声明尺寸或有效航向")
    return ParticipantGeometry(track.category, length=track.length, width=track.width, margin=margin,
                               reference_point=track.reference_point, center_offset_m=center_offset_m)
