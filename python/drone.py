from __future__ import annotations
import typing
import time
from asyncio import to_thread
from threading import Event
from numpy import (
    reshape,
    array,
    dtype,
    ndarray,
    concatenate,
    float16,
    dot,
    clip,
    arccos,
    cross,
    isnan,
    degrees,
)
from numpy.linalg import norm
from open3d import geometry, utility, visualization
from fusion_detection import getPointCloudPoints
import drone_request
from projectairsim import ProjectAirSimClient, Drone, World
from tqdm import tqdm
from airsim import MultirotorClient, YawMode
from drone_request import DroneRequestClient


class IDrone:
    def __init__(self):
        pass

    def rotate_vectors(
        self,
        quaternion: ndarray,  # [w,x,y,z]
        vectors: ndarray = array([1, 0, 0], float16),  # (n,3) or (x,y,z)
    ) -> ndarray:  # (n,3)
        w, x, y, z = quaternion
        rot_matrix = array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float16,
        )
        rotated = vectors @ rot_matrix.T
        return rotated

    def get_ground_truth_pose(self) -> tuple[ndarray, ndarray]:
        pass

    def get_pose(self, ground_truth: bool = True) -> tuple[ndarray, ndarray]:
        pass

    def get_lidar_data(
        self, ensure_new: bool = True, relative: bool = False
    ) -> ndarray:
        pass

    async def move_by_velocity(
        self,
        v_north: float = 1.0,
        v_east: float = 0.0,
        v_down: float = 0.0,
        duration: float = 0,
        face_to_forward: bool = True,
        up_vector: ndarray = array([0, 0, 1], dtype=float16),
        yaw_ratio: float = 0.5,
    ) -> None:
        pass

    def close(self) -> None:
        pass


class AdvancedAirSimDrone(IDrone):
    ac: MultirotorClient
    dr: DroneRequestClient
    drone_id: int

    def __init__(
        self,
        drone_id: int = 1,
        airsim_address: str = "127.0.0.1",
        airsim_port: int = 41451,
        drone_request_address: str = "127.0.0.1",
        drone_request_port: int = 8811,
    ):
        self.drone_id = drone_id
        self.ac = MultirotorClient(ip=airsim_address, port=airsim_port)
        self.dr = DroneRequestClient((drone_request_address, drone_request_port))
        self.ac.confirmConnection()
        self.ac.reset()
        self.ac.enableApiControl(True, vehicle_name=str(self.drone_id))
        self.ac.armDisarm(True, vehicle_name=str(self.drone_id))

    def get_ground_truth_pose(self) -> tuple[ndarray, ndarray]:
        att = self.dr.getAttitude(self.drone_id)
        return array([att.z, att.x, -att.y], dtype=float16), array(
            [-att.rw, att.rx, att.rz, -att.ry], dtype=float16
        )

    def get_pose(self, ground_truth: bool = True) -> tuple[ndarray, ndarray]:
        if ground_truth:
            return self.get_ground_truth_pose()

    def get_lidar_data(
        self, ensure_new: bool = True, relative: bool = False
    ) -> ndarray:
        depth = self.dr.getDepth(self.drone_id)
        att = self.dr.getAttitude(self.drone_id)
        results = []
        if relative:
            att.x = 0
            att.y = 0
            att.z = 0
        points, _ = getPointCloudPoints(depth, att)
        for x, y, z, b in points:
            if b:
                continue
            results.append([z, x, -y])
        return array(results, dtype=float16)

    async def move_by_velocity(
        self,
        v_north: float = 1.0,
        v_east: float = 0.0,
        v_down: float = 0.0,
        duration: float = 0,
        face_to_forward=True,
        up_vector=array([0, 0, 1], dtype=float16),
        yaw_ratio=0.5,
    ):
        if face_to_forward:
            _, q = self.get_pose()
            forward_vec = self.rotate_vectors(q)
            forward_vec[2] = 0
            v_vec = array([v_north, v_east, 0], dtype=float16)
            norm_f = norm(forward_vec)
            norm_v = norm(v_vec)
            if isnan(norm_v) or norm_v < 1e-4:
                theta = 0
            else:
                dot_fv = dot(forward_vec, v_vec)
                cos_theta = dot_fv / norm_f / norm_v
                cos_theta = clip(cos_theta, -1, 1)
                theta = float(arccos(cos_theta))
                if dot(cross(forward_vec, v_vec), up_vector) < 0:
                    theta = -theta
                theta *= yaw_ratio
        else:
            theta = 0
        await to_thread(
            self.ac.moveByVelocityAsync(
                vx=v_north,
                vy=v_east,
                vz=v_down,
                duration=duration,
                yaw_mode=YawMode(True, yaw_or_rate=degrees(theta)),
                vehicle_name=str(self.drone_id),
            ).join
        )

    def close(self):
        self.ac.client.close()
        del self.dr


class ProjectAirSimDrone(IDrone):
    client: ProjectAirSimClient
    world: World
    drone: Drone
    _lidar_data: ndarray
    _new_lidar_event: Event

    def init_world(
        self, scene_config_name: str = "../Unreal_ProjectAirSim/scene_lidar_drone.jsonc"
    ):
        self.world = World(
            self.client,
            scene_config_name=scene_config_name,
            delay_after_load_sec=2,
            sim_config_path="",
        )

    def init_drone(self, drone_name: str = "Drone1"):
        self.drone = Drone(self.client, self.world, drone_name)
        self.drone.enable_api_control()
        self.drone.arm()

    def __init__(
        self,
        address: str = "127.0.0.1",
        port_topics: int = 8989,
        port_services: int = 8990,
    ):
        self._new_lidar_event = Event()
        self.client = ProjectAirSimClient(
            address=address, port_topics=port_topics, port_services=port_services
        )
        self.client.connect()
        self.init_world()
        self.init_drone()
        self.subscribe_lidar()

    def subscribe_lidar(self):
        self.client.subscribe(
            self.drone.sensors["lidar1"]["lidar"],
            lambda _, lidar: self.process_lidar_data(lidar),
        )

    def get_ground_truth_pose(self) -> tuple[ndarray, ndarray]:
        pose = self.drone.get_ground_truth_pose()
        translation = pose["translation"]
        rotation = pose["rotation"]
        return (
            array(
                [translation["x"], translation["y"], translation["z"]],
                dtype=float16,
            ),
            array(
                [rotation["w"], rotation["x"], rotation["y"], rotation["z"]],
                dtype=float16,
            ),
        )

    def get_pose(self, ground_truth: bool = True) -> tuple[ndarray, ndarray]:
        if ground_truth:
            return self.get_ground_truth_pose()

    def process_lidar_data(self, lidar_data: dict):
        if lidar_data:
            points = array(lidar_data["point_cloud"], dtype=dtype("f4"))
            points = reshape(points, (int(points.shape[0] / 3), 3))
            self._lidar_data = points
            self._new_lidar_event.set()

    def get_lidar_data(
        self, ensure_new: bool = True, relative: bool = False
    ) -> ndarray:
        if ensure_new:
            self._new_lidar_event.wait(3)
            self._new_lidar_event.clear()
        if relative:
            return self._lidar_data.copy()
        else:
            rel = self._lidar_data.copy()
            pos, q = self.get_pose()
            rot_rel = self.rotate_vectors(q, rel)
            return rot_rel + pos

    async def move_by_velocity(
        self,
        v_north: float = 1.0,
        v_east: float = 0.0,
        v_down: float = 0.0,
        duration: float = 0,
        face_to_forward: bool = True,
        up_vector: ndarray = array([0, 0, 1], dtype=float16),
        yaw_ratio: float = 0.5,
    ) -> None:
        if face_to_forward:
            _, q = self.get_pose()
            forward_vec = self.rotate_vectors(q)
            forward_vec[2] = 0
            v_vec = array([v_north, v_east, 0], dtype=float16)
            norm_f = norm(forward_vec)
            norm_v = norm(v_vec)
            if isnan(norm_v) or norm_v < 1e-4:
                theta = 0
            else:
                dot_fv = dot(forward_vec, v_vec)
                cos_theta = dot_fv / norm_f / norm_v
                cos_theta = clip(cos_theta, -1, 1)
                theta = float(arccos(cos_theta))
                if dot(cross(forward_vec, v_vec), up_vector) < 0:
                    theta = -theta
                theta *= yaw_ratio
        else:
            theta = 0
        task = await self.drone.move_by_velocity_async(
            v_north=v_north, v_east=v_east, v_down=v_down, duration=duration, yaw=theta
        )
        await task

    def close(self):
        self.client.disconnect()
