from tree import PathGraph, OCTreeNode
import drone_request
import fusion_detection
import airsim
import time
import math
from msgpackrpc.future import Future

all_points = []
all_colors = []
all_paths = []
all_path_colors = []


def getPath(
    root: OCTreeNode,
    pg: PathGraph,
    current: tuple[float, float, float],
    target: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    print("getting path...  ", end="")
    start = time.time()

    current_node = root.query(current)
    target_node = root.query(target)
    path = pg.get_path(current_node, target_node)
    combineIndex = 0
    for i in range(len(path)):
        if current_node == path[i].tree_node:
            combineIndex = i
        else:
            break

    path = (
        [current]
        + [item.tree_node.center for item in path[combineIndex + 1 :]]
        + [target]
    )
    print(
        f"get path cost {time.time()*1000-start*1000:.2f}ms, start: {current_node.center,current_node.state == current_node.empty} end: {target_node.center,target_node.state == target_node.empty}"
    )
    print(path)
    return path


def flyTo(
    ac: airsim.MultirotorClient,
    dr: drone_request.DroneRequestClient,
    root: OCTreeNode,
    target: tuple[float, float, float],
    pg: PathGraph,
    dt: float = 0.1,
    velocity: float = 7,
    accept_distance: float = 4,
) -> tuple[bool, bool]:
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
    yaw_offset = math.acos(
        target_forward[0] * current_forward[0] + target_forward[2] * current_forward[2]
    )
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
    real_target = [current[i] + direction[i] * real_velocity for i in range(3)]
    ps, cs = fusion_detection.getPointCloudPoints(
        dr.getDepth(1), dr.getAttitude(1), dr.getImage(1), 100
    )
    all_points.extend(ps)
    all_colors.extend(cs)
    for p in ps:
        root.add(p)
    current_node = root.query(current)
    target_node = root.query(target)
    if current_node.state != root.empty:
        current_node.clear()
    if target_node.state != root.empty:
        print(f"cannot fly to {target} because it is not empty:{target_node.center}")
        print("updating...", end="")
        start = time.time() * 1000
        pg.update(root)
        print(f"update cost {time.time()*1000-start:.2f}ms")
        return False, False
    real_target_node = root.query(real_target)
    if real_target_node.state != root.empty:
        print(
            f"cannot fly to {target} by {real_target_node.center,real_target_node.state == root.empty}"
        )
        real_target_node.clear()
        print("updating...", end="")
        start = time.time() * 1000
        pg.update(root)
        print(f"update cost {time.time()*1000-start:.2f}ms")
        return False, False
    # print(f"moving to {real_target} distance: {distance}")
    if abs(yaw_offset) > 30:
        real_velocity = 0
    f = ac.moveByVelocityAsync(  # z x -y
        direction[2] * real_velocity,
        direction[0] * real_velocity,
        -direction[1] * real_velocity,
        dt,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(True, yaw_offset),
    )
    # pg.update(root)
    f.join()
    return (
        True,
        fusion_detection.getV3Distance(dr.getAttitude(1).getPosition(), target)
        < accept_distance,
    )


if __name__ == "__main__":
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
    pg = PathGraph()
    # x = 500
    # y = 50
    # z = 500
    # ml = (1, 1, 1)
    x = 100
    y = 15
    z = 100
    ml = (1, 1, 1)
    root = OCTreeNode(-x, -y, -z, x, y, z, ml)
    # target = (-297, 12.65, 246.5)
    target = (71, 12, 83)
    root.query(target).divide(4)
    pg.update(root)
    f: Future = None
    position = dr.getAttitude(1).getPosition()
    path = getPath(root, pg, position, target)
    currentIndex = 1
    while True:
        result, reach = flyTo(ac, dr, root, path[currentIndex], pg)
        if not result:
            # if target != path[currentIndex]:
            #     root.add(path[currentIndex])
            position = dr.getAttitude(1).getPosition()
            path = getPath(root, pg, position, target)
            if len(path) <= 2:
                fusion_detection.renderPointCloud(
                    all_points + all_paths, all_colors + all_path_colors
                )
                root.render(show_now=True)
            currentIndex = 1
            continue
        if reach:
            # print(f"reached {currentIndex}")
            currentIndex += 1
            if currentIndex >= len(path):
                print("reach")
                fusion_detection.renderPointCloud(
                    all_points + all_paths, all_colors + all_path_colors
                )
                root.render(show_now=True)
                break
