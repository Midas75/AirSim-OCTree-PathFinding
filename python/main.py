import time
import math
import os
import datetime

import drone_request
import fusion_detection
import airsim

from msgpackrpc.future import Future

from tree import PathGraph, TreeNode
from ctree import CPathGraph, CTreeNode, init_ctree


all_points = []
all_colors = []
all_paths = []
all_path_colors = []
tn_cls = TreeNode
pg_cls = PathGraph


def get_path(
    root: tn_cls, pg: pg_cls, current: list[float], target: list[float]
) -> list[tuple[float, float, float]]:
    expand = [ml / 3 for ml in root.min_length]
    start = time.perf_counter()
    current_node = root.query(current, True)
    target_node = root.query(target, True)
    if current_node is None or target_node is None:
        print("is none", current, target)
    path = pg.get_path(current_node, target_node)
    c_path = [current] + root.interpolation_center(path) + [target]
    _, s_path = root.path_smoothing(c_path, expand)
    print(f"get path cost {time.perf_counter()*1000-start*1000:.2f}ms")
    return s_path


def fly_to(
    ac: airsim.MultirotorClient,
    dr: drone_request.DroneRequestClient,
    root: tn_cls,
    target: list[float],
    pg: pg_cls,
    dt: float = 0.1,
    velocity: float = 5,
    accept_distance: float = 4,
) -> tuple[bool, bool]:
    result = True
    att = dr.getAttitude(1)
    current_forward = fusion_detection.getForward((att.rx, att.ry, att.rz, att.rw))
    current_forward = fusion_detection.normalize(
        (current_forward[0], 0, current_forward[2])
    )
    current = att.getPosition()
    all_paths.append(current)
    all_path_colors.append((1, 0, 0))
    direction = [target[i] - current[i] for i in range(3)]
    direction = fusion_detection.normalize(direction)
    target_forward = fusion_detection.normalize((direction[0], 0, direction[2]))
    cross_product = (
        target_forward[0] * current_forward[0] + target_forward[2] * current_forward[2]
    )
    yaw_offset = math.acos(min(max(cross_product, -1), 1))
    yaw_offset = math.degrees(yaw_offset)
    if (
        target_forward[0] * current_forward[2] - target_forward[2] * current_forward[0]
        < 0
    ):
        yaw_offset = -yaw_offset
    distance = fusion_detection.getV3Distance(current, target)
    if distance / dt < velocity:
        real_velocity = distance / dt
    else:
        real_velocity = velocity

    ps, cs = fusion_detection.getPointCloudPoints(
        dr.getDepth(1), dr.getAttitude(1), dr.getImage(1), 100
    )
    for i, p in enumerate(ps):
        p = ps[i]
        c = cs[i]
        if not p[-1]:
            all_points.append((p[0], p[1], p[2]))
            all_colors.append(c)
        root.add_raycast(list(current), [p[0], p[1], p[2]], p[-1], dynamic_culling=1000)
    current_node = root.query(current, True)
    target_node = root.query(target, True)
    if current_node.state != root.EMPTY:
        current_node.clear_as()
    if target_node.state != root.EMPTY:
        result = False
        print(f"cannot fly to {target} because it is not empty:{target_node.center}")
    if result and root.cross_lca(current, target):
        result = False
        print(f"cannot fly to {target} because cross")
    real_target = [current[i] + direction[i] * real_velocity for i in range(3)]
    real_target_node = root.query(real_target, True)
    if result and real_target_node.state != root.EMPTY:
        result = False
        print(f"cannot fly to {target} by {real_target_node.center}")
    # print(f"moving to {real_target} distance: {distance}")
    if abs(yaw_offset) > 20:
        real_velocity = 0
    if not result:
        real_velocity = -real_velocity / 2

    f: Future = ac.moveByVelocityAsync(  # z x -y
        direction[2] * real_velocity,
        direction[0] * real_velocity,
        -direction[1] * real_velocity,
        dt,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(True, yaw_offset),
    )
    if not result:
        print("updating...")
        start = time.perf_counter() * 1000
        pg.update(root)
        print(f"update cost {time.perf_counter()*1000-start:.2f}ms")
        f.join()
        return False, False
    # pg.update(root)
    f.join()
    return (
        result,
        fusion_detection.getV3Distance(dr.getAttitude(1).getPosition(), target)
        < accept_distance,
    )


def main():
    init_ctree()
    ac = airsim.MultirotorClient()
    time.sleep(1)
    ac.confirmConnection()
    time.sleep(1)
    ac.reset()
    time.sleep(1)
    ac.enableApiControl(True)
    time.sleep(1)
    ac.takeoffAsync().join()
    ac.moveToZAsync(-10, velocity=5).join()
    dr = drone_request.DroneRequestClient()
    # x = 500
    # y = 100
    # z = 500
    x = 100
    y = 12
    z = 100
    ml = [2, 2, 2]

    pg = pg_cls()
    if os.path.exists("TreeNode.json.gz"):
        tn = tn_cls.load("TreeNode.json.gz")
    else:
        tn = tn_cls().as_root([-x, 1, -z], [x, y, z], ml)
    # target = (-297, 12.65, 246.5)
    target = (71, 12, 83)
    start_time = time.perf_counter()
    print("first updating...")
    pg.update(tn)
    print(f"first update cost {(time.perf_counter()-start_time)*1000:.2f}ms")
    start_point = position = dr.getAttitude(1).getPosition()
    path = get_path(tn, pg, position, target)
    current_index = 1
    while True:
        result, reach = fly_to(ac, dr, tn, path[current_index], pg)
        if not result:
            # if target != path[current_index]:
            #     root.add(path[current_index])
            position = dr.getAttitude(1).getPosition()
            path = get_path(tn, pg, position, target)
            if len(path) <= 2:
                pass
                # fusion_detection.renderPointCloud(
                #     all_points + all_paths, all_colors + all_path_colors
                # )
                # root.render3(with_coordinate=False, show_now=0)
            current_index = 1
            continue
        if reach:
            # print(f"reached {current_index}")
            current_index += 1
            if current_index >= len(path):
                print("reach")
                pg.update(tn)
                tn.save("TreeNode.json.gz")
                tn.save(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json.gz'
                )
                fusion_detection.renderPointCloud(
                    all_points + all_paths, all_colors + all_path_colors
                )
                p = pg.get_path(tn.query(start_point), tn.query(target))
                cp = tn.interpolation_center(p)
                _, sp = tn.path_smoothing(cp, [ml / 3 for ml in tn.min_length])
                tn.render3(with_path=sp, with_coordinate=False, show_now=0)
                break


if __name__ == "__main__":
    main()
