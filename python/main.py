from time import perf_counter
import os
import datetime
from asyncio import run

from numpy import array, float16, clip, arccos, degrees
from numpy.linalg import norm

from drone import ProjectAirSimDrone, AdvancedAirSimDrone
from tree import PathGraph, TreeNode

from ctree import CPathGraph, CTreeNode, init_ctree
from tree_utils import VisWindow3

all_points = []
all_colors = []
all_paths = []
all_path_colors = []
tn_cls = TreeNode
pg_cls = PathGraph
# tn_cls = CTreeNode
# pg_cls = CPathGraph


def get_path(
    root: tn_cls, pg: pg_cls, current: list[float], target: list[float]
) -> list[list[float]]:
    start = perf_counter()
    current_node = root.query(current, True)
    target_node = root.query(target, True)
    path = pg.get_path(current_node, target_node)
    c_path = [current] + root.interpolation_center(path) + [target]
    _, s_path = root.path_smoothing(
        c_path, expand=[ml / 2 for ml in root.min_length], break_length=2
    )
    print(f"get path cost {perf_counter()*1000-start*1000:.2f}ms")
    return s_path


def full_path_cross_check(root: tn_cls, path: list[list[float]]) -> bool:
    for i in range(len(path) - 1):
        if root.cross_lca(path[i], path[i + 1]):
            return False
    return True


def fly_to(
    d: ProjectAirSimDrone,
    root: tn_cls,
    pg: pg_cls,
    target: list[float],
    dt: float = 0.1,
    velocity: float = 5,
    accept_distance: float = 4,
) -> tuple[bool, bool]:
    result = True
    target_arr = array(target, dtype=float16)
    position, rotation = d.get_ground_truth_pose()
    current = [float(p) for p in position]
    current_forward = d.rotate_vectors(rotation)
    current_forward[2] = 0
    current_forward /= norm(current_forward)
    direction = target_arr - position
    target_forward = direction.copy()
    distance = norm(direction)
    direction /= distance
    target_forward[2] = 0
    target_forward /= norm(target_forward)

    cross_product = (
        target_forward[0] * current_forward[0] + target_forward[1] * current_forward[1]
    )
    yaw_offset = arccos(clip(cross_product, -1, 1))
    yaw_offset = degrees(yaw_offset)
    if distance / dt < velocity:
        real_vel = distance / dt
    else:
        real_vel = velocity
    point_cloud = d.get_lidar_data(ensure_new=True)
    root.next_ray_batch()
    for i, p in enumerate(point_cloud):
        root.add_raycast(current, list(p))
    current_node = root.query(current, True)
    target_node = root.query(target, True)
    # if current_node.state != root.EMPTY:
    #     current_node.clear_as()
    if target_node.state != root.EMPTY:
        result = False
        print(f"cannot fly to {target} because it is not empty:{target_node.center}")
    real_target = [current[i] + direction[i] * real_vel for i in range(3)]
    real_target_node = root.query(real_target, True)
    if result and real_target_node.state != root.EMPTY:
        result = False
        print(f"cannot fly to {target} by {real_target_node.center}")
    if result and root.cross_lca(current, target):
        result = False
        print(f"cannot fly to {target} because cross")
    if not result:
        real_vel = -real_vel / 4
    elif abs(yaw_offset) > 10:
        real_vel = real_vel / 10
    direction *= real_vel
    run(
        d.move_by_velocity(
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
            dt,
            True,
            yaw_ratio=0.75,
        )
    )
    if not result:
        print("updating...")
        start = perf_counter() * 1000
        pg.update(root)
        print(f"update cost {perf_counter()*1000-start:.2f}ms")
        return False, False
    return (result, norm(position - target_arr) < accept_distance)


def main():
    if tn_cls == CTreeNode:
        init_ctree(debug=False)
    # d = ProjectAirSimDrone()
    d = AdvancedAirSimDrone()
    run(d.move_by_velocity(0, 0, -5, 2))
    vis = True
    x = 300
    y = 300
    z = 15
    ml = [2, 2, 2]

    pg = pg_cls()
    if os.path.exists("TreeNode.json.gz"):
        tn = tn_cls.load("TreeNode.json.gz")
    else:
        tn = tn_cls().as_root([-x, -y, -z], [x, y, 0], ml)
    if vis:
        vw = VisWindow3()
    target = [84, 69, -12]
    start_time = perf_counter()
    print("first updating...")
    pg.update(tn)
    print(f"first update cost {(perf_counter()-start_time)*1000:.2f}ms")
    start_point = position = [float(p) for p in d.get_ground_truth_pose()[0]]
    path = get_path(tn, pg, position, target)
    # vw.update(tn, pg, path=path)
    current_index = 1
    try:
        input("press enter to avoid waitiing vw loading")
        while True:
            result, reach = fly_to(d, tn, pg, path[current_index])
            if result and not full_path_cross_check(tn, path):
                result = False
                pg.update(tn)
                print("cannot fly by path because cross")
            if vis:
                vw.update(tn, pg, path=path)
            if not result:
                # if target != path[current_index]:
                #     root.add(path[current_index])
                position = [float(p) for p in d.get_ground_truth_pose()[0]]
                path = get_path(tn, pg, position, target)
                if vis:
                    vw.update(tn, pg, path=path)
                if len(path) <= 2:
                    pass
                current_index = 1
                continue
            if reach:
                current_index += 1
                if current_index >= len(path):
                    print("reach")
                    pg.update(tn)
                    tn.save("TreeNode.json.gz")
                    tn.save(
                        f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json.gz'
                    )
                    p = get_path(tn, pg, start_point, target)
                    if vis:
                        vw.update(tn, pg, path=p)
                    input("press enter to exit")
                    # vw.stop()
                    d.close()
                    break
    except Exception as e:
        d.close()
        raise e
    except KeyboardInterrupt as e:
        d.close()
        raise e


if __name__ == "__main__":
    main()
