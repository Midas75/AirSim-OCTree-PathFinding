import typing

import airsim
import fusion_detection
import drone_request


class IDrone:
    pass


class AirSimDrone(IDrone):
    pass


class UnityDrone(IDrone):
    _ac: airsim.MultirotorClient
    _dr: drone_request.DroneRequestClient
    _id: int

    def __init__(self, _id: int = 1):
        super().__init__()
        self._id = _id

    def get_forward(self) -> list[float]:
        att = self._dr.getAttitude(self._id)
        return list(fusion_detection.getForward((att.rx, att.ry, att.rz, att.rw)))

    def get_position(self) -> list[float]:
        return list(self._dr.getAttitude(self._id).getPosition())

    def get_pointcloud_points(
        self, depth_threshold: float = 100, with_color: bool = True
    ) -> tuple[
        list[tuple[float, float, float, bool]], list[tuple[float, float, float]]
    ]:
        ps, cs = fusion_detection.getPointCloudPoints(
            self._dr.getDepth(self._id),
            self._dr.getAttitude(self._id),
            None if not with_color else self._dr.getImage(self._id),
            depth_threshold,
        )
        return ps, cs