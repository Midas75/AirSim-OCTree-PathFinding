import drone_request
import math
import typing
import open3d as o3d


def rotateWithQuaternion(
    quaternion: tuple[float, float, float, float],
    direction: tuple[float, float, float] = (0, 0, 1),
) -> tuple[float, float, float]:
    num = quaternion[0] * 2  # x
    num2 = quaternion[1] * 2  # y
    num3 = quaternion[2] * 2  # z

    num4 = quaternion[0] * num  # x
    num5 = quaternion[1] * num2  # y
    num6 = quaternion[2] * num3  # z

    num7 = quaternion[0] * num2  # x
    num8 = quaternion[0] * num3  # x
    num9 = quaternion[1] * num3  # y

    num10 = quaternion[3] * num  # w
    num11 = quaternion[3] * num2  # w
    num12 = quaternion[3] * num3  # w
    rx = (
        (1 - (num5 + num6)) * direction[0]
        + (num7 - num12) * direction[1]
        + (num8 + num11) * direction[2]
    )
    ry = (
        (num7 + num12) * direction[0]
        + (1 - (num4 + num6)) * direction[1]
        + (num9 - num10) * direction[2]
    )
    rz = (
        (num8 - num11) * direction[0]
        + (num9 + num10) * direction[1]
        + (1 - (num4 + num5)) * direction[2]
    )
    return (rx, ry, rz)


def getForward(
    quaternion: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    return rotateWithQuaternion(quaternion)


def getV3Distance(
    a: tuple[float, float, float], b: tuple[float, float, float] = (0, 0, 0)
) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)

def distance(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def normalize(
    v: typing.Union[tuple[float, float, float], list[float]]
) -> tuple[float, float, float]:
    norm = distance(v)
    if norm == 0:
        return v
    return (v[0] / norm, v[1] / norm, v[2] / norm)

def getRegionColor(
    image: drone_request.Image, position: tuple[float, float], regionSize: int = 10
) -> tuple[float, float, float]:
    x, y = int(position[0] * image.image.shape[1]), int(
        position[1] * image.image.shape[0]
    )
    halfSize = regionSize // 2
    region = image.image[
        y - halfSize : y + halfSize + 1, x - halfSize : x + halfSize + 1
    ]
    avgColor = region.mean(axis=(0, 1)) / 255
    # bgr -> rgb
    return (avgColor[2], avgColor[1], avgColor[0])

def getPointCloudPoints(
    depth: drone_request.Depth,
    attitude: drone_request.Attitude,
    image: drone_request.Image = None,
    depthThreshold: float = 100,
    whRatio: float = 1280 / 768,
) -> tuple[list[tuple[float, float, float,bool]], list[tuple[float, float, float]]]:
    points = []
    colors = []
    q = (attitude.rx, attitude.ry, attitude.rz, attitude.rw)
    p = (attitude.x, attitude.y, attitude.z)
    fovRatio = 2 * math.tan(math.radians(attitude.fov / 2))
    if image is not None:
        whRatio = image.image.shape[1] / image.image.shape[0]
    for r in range(depth.row):
        for c in range(depth.col):
            depthValue = depth.depths[r][c]
            empty = False
            if depthThreshold - depthValue < 1:
                empty = True
            center = [(c + 0.5) / depth.col, 1 - (r + 0.5) / depth.row]
            targetDirection = (
                whRatio * fovRatio * (center[0] - 0.5),
                fovRatio * (0.5 - center[1]),
                1,
            )
            targetDirection = normalize(targetDirection)
            targetDirection = rotateWithQuaternion(q, targetDirection)

            pointPosition = (
                p[0] + targetDirection[0] * depthValue,
                p[1] + targetDirection[1] * depthValue,
                p[2] + targetDirection[2] * depthValue,
                empty
            )
            points.append(pointPosition)
            if image is not None:
                colors.append(getRegionColor(image, center))
            else:
                colors.append((0.5, 0.5, 0.5))
    return points, colors


def renderPointCloud(
    points: list[tuple[float, float, float]], colors: list[tuple[float, float, float]] = None
) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors != None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])