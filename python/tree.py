# pylint: disable=C0114  # I'm lazy
# pylint: disable=C0115  # I'm lazy
# pylint: disable=C0116  # I'm lazy
# pylint: disable=C0302  # I'm long
from __future__ import annotations
from math import sqrt, ceil, log2, dist
from typing import Union, Literal, Any
from itertools import product
import heapq
import json
import gzip

from numpy import ndarray, uint8, ones, array

from cv2 import rectangle, line, imshow, waitKey  # pylint: disable=no-name-in-module
from open3d import geometry, utility, visualization
from tqdm import tqdm


class TreeNode:
    INF: float = float("inf")
    EMPTY: int = 0
    FULL: int = 1
    HALF_FULL: int = 2

    child: dict[int, TreeNode] = None
    state: int

    dynamic_culling: int = -1
    last_ray_id: int = 0
    ray_id: int = 0

    bound_min: list[float]
    bound_max: list[float]
    bound_size: list[float]
    center: list[float]

    min_length: list[float]
    parent: TreeNode
    root: TreeNode
    nodes: dict[int, TreeNode]

    bit_values: list[int] = [1, 2, 4, 8]
    dims: int
    rangel: list[int]
    directions: list[list[bool]]

    _min: bool = False
    is_leaf: bool = True
    known: bool = False

    i_bound_min: list[int]
    i_bound_max: list[int]
    i_bound_size: list[int]
    i_center: list[int]

    id: int = -1

    def __init__(
        self,
        parent: TreeNode = None,
        direction: int = None,
        divide: list[bool] = None,
    ) -> None:
        self.state = self.EMPTY
        if parent is not None:
            self.parent = parent
            self.root = parent.root
            self.nodes = parent.nodes

            self.dims = parent.dims
            self.rangel = parent.rangel
            self.directions = parent.directions

            self.min_length = parent.min_length

            self.i_bound_min = [0] * self.dims
            self.i_bound_max = [0] * self.dims
            self.i_bound_size = [0] * self.dims
            if direction is not None:
                for dim in self.rangel:
                    if not divide[dim]:
                        self.i_bound_min[dim] = parent.i_bound_min[dim]
                        self.i_bound_max[dim] = parent.i_bound_max[dim]

                    elif not self.directions[direction][dim]:
                        self.i_bound_min[dim] = parent.i_bound_min[dim]
                        self.i_bound_max[dim] = parent.i_center[dim]
                    else:
                        self.i_bound_min[dim] = parent.i_center[dim]
                        self.i_bound_max[dim] = parent.i_bound_max[dim]
                self.update_bound()

    def serialize(self, progress: bool = True) -> dict[str, Any]:
        if self.parent is not None:
            raise TypeError("cannot serialize from a non-root node")
        result = dict[str, Any]()
        result["min_length"] = self.min_length
        result["bound_min"] = self.bound_min
        result["bound_max"] = self.bound_max
        result["nodes"] = dict[str, Any]()
        for i_id, node in (
            tqdm(self.nodes.items(), desc="serializing")
            if progress
            else self.nodes.items()
        ):
            s_id = str(i_id)
            info = dict[str, Any]()
            info["i_bound_min"] = node.i_bound_min
            info["i_bound_max"] = node.i_bound_max
            info["known"] = node.known
            info["state"] = node.state
            info["is_leaf"] = node.is_leaf
            if node.child is not None:
                info["child"] = dict[str, Any]()
                for direction, c in node.child.items():
                    info["child"][str(direction)] = c.id
            result["nodes"][s_id] = info
        return result

    def save(self, path: str = None) -> None:
        if path is None:
            path = f"{self.__class__.__name__}.json.gz"
        with open(path, "wb") as f:
            j = json.dumps(self.serialize())
            gz = gzip.compress(j.encode("utf-8"))
            f.write(gz)

    @staticmethod
    def deserialize(
        obj: dict[str, Any],
        progress: bool = True,
        _current_id: int = None,
        _parent: TreeNode = None,
        _tqdm: tqdm = None,
    ) -> TreeNode:
        is_root = False
        if _current_id is None:
            node = TreeNode().as_root(
                obj["bound_min"], obj["bound_max"], obj["min_length"]
            )
            _current_id = node.id
            is_root = True
            if progress:
                _tqdm = tqdm(total=len(obj["nodes"]), desc="deserializing")
        info = obj["nodes"][str(_current_id)]
        if not is_root:
            node = TreeNode(_parent)
            node.i_bound_max = info["i_bound_max"]
            node.i_bound_min = info["i_bound_min"]
            node.update_bound()
        node.state = info["state"]
        node.known = info["known"]
        node.is_leaf = info["is_leaf"]
        if "child" in info:
            node.child = dict[int, TreeNode]()
            relink_node = dict[int, int]()
            for str_direction, int_id in info["child"].items():
                int_direction = int(str_direction)
                if int_id not in relink_node:
                    node.child[int_direction] = TreeNode.deserialize(
                        obj, progress, int_id, node, _tqdm
                    )
                    if _tqdm is not None:
                        _tqdm.update(1)
                    relink_node[int_id] = int_direction
                else:
                    node.child[int_direction] = node.child[relink_node[int_id]]
        return node

    @staticmethod
    def load(path: str = None) -> TreeNode:
        if path is None:
            path = f"{__class__.__name__}.json.gz"
        with open(path, "rb") as f:
            j = gzip.decompress(f.read()).decode("utf-8")
            return TreeNode.deserialize(json.loads(j))

    def as_root(
        self, bound_min: list[float], bound_max: list[float], min_length: list[float]
    ) -> TreeNode:
        self.min_length = min_length

        self.dims = len(min_length)
        self.rangel = range(self.dims)
        self.directions = [
            list(reversed(p)) for p in product([False, True], repeat=self.dims)
        ]
        self.root = self
        self.parent = None
        self.nodes = dict[int, TreeNode]()

        self.bound_min = bound_min
        self.bound_max = bound_max

        self.bound_size = [
            self.bound_max[dim] - self.bound_min[dim] for dim in self.rangel
        ]
        self.center = [
            self.bound_max[dim] / 2 + self.bound_min[dim] / 2 for dim in self.rangel
        ]

        self.i_bound_min = [0] * self.dims
        self.i_bound_max = [0] * self.dims
        self.i_bound_size = [0] * self.dims
        self.i_center = [0] * self.dims
        for dim in self.rangel:
            dim_ratio = self.bound_size[dim] / self.min_length[dim]
            if dim_ratio < 1:
                raise ValueError(
                    f"on dim:{dim}, min_length {self.min_length[dim]} "
                    f"is smaller than bound_size {self.bound_size[dim]}"
                )
            self.i_bound_size[dim] = 2 ** ceil(
                log2(dim_ratio)
            )  # assure i_center is int
            self.i_bound_max[dim] = self.i_bound_size[dim]
            self.i_bound_min[dim] = 0
            self.i_center[dim] = self.i_bound_max[dim] // 2
        self._min = self.is_min()
        self.id = self.gen_id()
        # python GC can handle this
        self.nodes[self.id] = self
        return self

    def update_bound(self):
        self.center = [0] * self.dims
        self.i_center = [0] * self.dims
        if self.parent is None:  # is not root
            raise ValueError("root cannot update bound")
        self.bound_max = [0] * self.dims
        self.bound_min = [0] * self.dims
        self.bound_size = [0] * self.dims
        self.i_bound_size = [0] * self.dims
        for dim in self.rangel:
            self.i_bound_size[dim] = self.i_bound_max[dim] - self.i_bound_min[dim]
            if self.parent is not None:
                bound_ratio = self.root.bound_size[dim] / self.root.i_bound_size[dim]
                self.bound_max[dim] = (
                    self.i_bound_max[dim] * bound_ratio + self.root.bound_min[dim]
                )
                self.bound_min[dim] = (
                    self.i_bound_min[dim] * bound_ratio + self.root.bound_min[dim]
                )
            self.bound_size[dim] = self.bound_max[dim] - self.bound_min[dim]
            self.center[dim] = self.bound_max[dim] / 2 + self.bound_min[dim] / 2
            self.i_center[dim] = (self.i_bound_max[dim] + self.i_bound_min[dim]) // 2

        self._min = self.is_min()
        if self.id in self.nodes:
            del self.nodes[self.id]
        self.id = self.gen_id()
        self.nodes[self.id] = self

    def gen_id(self) -> int:
        result = 0
        dim_range = 1
        for dim in self.rangel:
            result += dim_range * self.i_center[dim]
            dim_range *= self.root.i_bound_size[dim]
        return result

    def divide(self, depth: int = 1) -> None:
        if self._min:
            return
        if self.state != self.EMPTY:
            return
        if not self.is_leaf:
            return
        if depth <= 0:
            return
        if self.child is None:
            self.child = dict[int, TreeNode]()
        self.is_leaf = False
        for i, _direction in enumerate(self.directions):
            _r, ri, d = self.get_bound_by_direction(i)
            if ri not in self.child:
                c = self.__class__(self, i, d)
                self.child[ri] = c
                self.child[i] = c
                c.divide(depth - 1)
            else:
                self.child[i] = self.child[ri]

    def is_min(self) -> bool:
        for dim in self.rangel:
            if self.i_center[dim] & 1 == 0:
                return False
        return True

    def update_state(self) -> None:
        if self.child is None or len(self.child) <= 0:
            self.is_leaf = True
            return
        full_counter = 0
        empty_counter = 0
        half_full_counter = 0
        for i in self.child:
            c = self.child[i]
            if c.state == self.FULL:
                full_counter += 1
            elif c.state == self.HALF_FULL:
                half_full_counter += 1
            elif c.state == self.EMPTY:
                empty_counter += 1
        if empty_counter == 0 and half_full_counter == 0:
            self.state = self.FULL
            self.remove_child()
        elif full_counter == 0 and half_full_counter == 0:
            self.state = self.EMPTY
        else:
            self.state = self.HALF_FULL

    def get_direction(self, point: list[float], allow_oor: bool = False) -> int:
        if (not allow_oor) and self.out_of_region(point):
            return -1
        result = 0

        for dim in self.rangel:
            if point[dim] > self.center[dim]:
                result += self.bit_values[dim]
        return result

    def out_of_region(self, point: list[float]) -> bool:
        for dim in self.rangel:
            if self.bound_min[dim] > point[dim] or self.bound_max[dim] < point[dim]:
                return True
        return False

    def get_direction_i(self, point: list[int], allow_oor: bool = False) -> int:
        if (not allow_oor) and self.out_of_region_i(point):
            return -1
        result = 0
        for dim in self.rangel:
            if point[dim] > self.i_center[dim]:
                result += self.bit_values[dim]
        return result

    def out_of_region_i(self, point: list[int]) -> bool:
        for dim in self.rangel:
            if self.i_bound_min[dim] > point[dim] or self.i_bound_max[dim] < point[dim]:
                return True
        return False

    def get_bound_by_direction(self, direction: int) -> tuple[bool, int, list[bool]]:
        reduce = False
        index = 0
        divide = [True] * self.dims
        directions = self.directions[direction]
        for dim in self.rangel:
            if self.i_center[dim] & 1 == 1:
                reduce = True
                divide[dim] = False
            elif directions[dim]:
                index += self.bit_values[dim]
        return reduce, index, divide

    def query(
        self, point: list[float], nearest_on_oor: bool = False
    ) -> TreeNode | None:
        if (not nearest_on_oor) and self.out_of_region(point):
            return None
        if self.is_leaf:
            return self
        direction = self.get_direction(point, True)
        if direction in self.child:
            return self.child[direction].query(point, True)
        return self

    def query_i(
        self, point: list[int], nearest_on_oor: bool = False
    ) -> Union[TreeNode, None]:
        if (not nearest_on_oor) and self.out_of_region_i(point):
            return None
        if self.is_leaf:
            return self
        direction = self.get_direction_i(point, nearest_on_oor)
        if direction in self.child:
            return self.child[direction].query_i(point, True)
        return self

    def clear_as(self, state: Literal[0, 1, 2] = 0) -> None:
        self.remove_child()
        self.state = state
        self.known = False
        parent = self.parent
        while parent is not None:
            parent.update_state()
            parent = parent.parent

    def remove_child(self):
        if self.child is None:
            return
        for _, child in self.child.items():
            child.remove_child()
            if child.id in self.nodes:
                del self.nodes[child.id]
        self.child = None
        self.is_leaf = True

    def lca(self, node1: TreeNode, node2: TreeNode) -> TreeNode:
        p1 = node1
        p2 = node2
        while p1 != p2:
            p1 = node2 if p1 is None else p1.parent
            p2 = node1 if p2 is None else p2.parent
        if p1 is None:
            return self
        return p1

    def cross_self(
        self, start: list[float], inv_vector: list[float], expand: list[float]
    ) -> bool:
        tmin = -self.INF
        tmax = self.INF
        for dim in self.rangel:
            b_min = self.bound_min[dim] - expand[dim]
            b_max = self.bound_max[dim] + expand[dim]
            if inv_vector[dim] is None:
                if start[dim] < b_min or start[dim] > b_max:
                    return False
            else:
                t1 = (b_min - start[dim]) * inv_vector[dim]
                t2 = (b_max - start[dim]) * inv_vector[dim]
                # pylint: disable=R1731  # if is faster than min
                # pylint: disable=R1730  # if is faster than max
                if t1 < t2:
                    if tmin < t1:
                        tmin = t1
                    if tmax > t2:
                        tmax = t2
                else:
                    if tmin < t2:
                        tmin = t2
                    if tmax > t1:
                        tmax = t1
                if tmin > tmax:
                    return False
        return tmax >= 0 and tmin <= 1

    def cross(
        self, start: list[float], inv_vector: list[float], expand: list[float]
    ) -> bool:
        if self.state == self.EMPTY:
            return False
        self_cross = self.cross_self(start, inv_vector, expand)
        if self.state == self.FULL:
            return self_cross

        if self_cross and self.child is not None:
            for direction in self.child:
                if self.child[direction].cross(start, inv_vector, expand):
                    return True
        return False

    def cross_lca(
        self, start: list[float], end: list[float], expand: list[float] = None
    ) -> bool:
        if expand is None:
            expand = [0] * self.dims
        ex_start = start.copy()
        ex_end = end.copy()
        for i in self.rangel:
            if start[i] < end[i]:
                ex_start[i] -= expand[i]
                ex_end[i] += expand[i]
            else:
                ex_start[i] += expand[i]
                ex_end[i] -= expand[i]
        n1 = self.query(ex_start, True)
        n2 = self.query(ex_end, True)
        vector = [end[i] - start[i] for i in self.rangel]
        inv_vector = [(None if vector[i] == 0 else 1 / vector[i]) for i in self.rangel]
        return self.lca(n1, n2).cross(start, inv_vector, expand)

    def get_parent(self, number: int = 1) -> TreeNode:
        parent = self
        for _ in range(number):
            if parent.parent is not None:
                parent = parent.parent
            else:
                break
        return parent

    def intersect(self, other: TreeNode) -> bool:
        one_eq = False
        for dim in self.rangel:
            if (
                self.i_bound_min[dim] > other.i_bound_max[dim]
                or self.i_bound_max[dim] < other.i_bound_min[dim]
            ):
                return False
            if (
                self.i_bound_min[dim] == other.i_bound_max[dim]
                or self.i_bound_max[dim] == other.i_bound_min[dim]
            ):
                if one_eq is True:
                    return False
                one_eq = True
        return True

    def add(
        self, point: list[float], empty: bool = False
    ) -> bool:  # tell if node.state was changed
        if self.state == self.FULL:
            if not empty:
                return False
            else:
                return False
        direction = self.get_direction(point)
        if direction < 0:
            return False
        if self.state == self.EMPTY:
            if empty:
                if not self.known:
                    self.known = True
                    return False
                else:
                    return False
            else:
                if self._min:
                    self.state = self.FULL
                    self.dynamic_culling = -1
                    return True
                else:
                    pass
        self.divide()
        changed = self.child[direction].add(point, empty)
        if changed:
            self.update_state()
        return changed

    def add_i(self, point: list[int], empty: bool = False) -> bool:
        if self.state == self.FULL:
            if not empty:
                return False
            else:
                return False
        direction = self.get_direction_i(point)
        if direction < 0:
            return False
        if self.state == self.EMPTY:
            if empty:
                if not self.known:
                    self.known = True
                    return False
                else:
                    return False
            else:
                if self._min:
                    self.state = self.FULL
                    return True
                else:
                    pass
        self.divide()
        changed = self.child[direction].add_i(point, empty)
        if changed:
            self.update_state()
        return changed

    def ray_out_intersect(
        self, point: list[float], vector: list[float]
    ) -> tuple[int, list[float]]:
        t_min = -self.INF
        t_max = self.INF
        out_dim = 0
        for dim in self.rangel:
            if vector[dim] == 0:
                t1 = self.INF
                t2 = self.INF
            else:
                t1 = (self.bound_min[dim] - point[dim]) / vector[dim]
                t2 = (self.bound_max[dim] - point[dim]) / vector[dim]
            if t1 > t2:
                t_near = t2
                t_far = t1
            else:
                t_near = t1
                t_far = t2
            if t_max > t_far:
                out_dim = dim
                t_max = t_far
            if t_min < t_near:
                t_min = t_near
        exit_t = t_max
        return out_dim, [point[dim] + exit_t * vector[dim] for dim in self.rangel]

    def next_ray_batch(self) -> None:
        self.root.ray_id += 1

    def center_to_segment(self, start: list[float], point: list[float]) -> float:
        ab = [point[i] - start[i] for i in self.rangel]
        ap = [self.center[i] - start[i] for i in self.rangel]
        ab2 = sum(a * a for a in ab)
        if ab2 == 0:
            return dist(self.center, start)
        t = sum(ap[i] * ab[i] for i in self.rangel) / ab2
        t = max(0, min(1, t))
        q = [start[i] + t * ab[i] for i in self.rangel]
        return dist(self.center, q)

    def add_raycast(
        self,
        start: list[float],
        point: list[float],
        empty_end: bool = False,
        dynamic_culling: int = 10,  # set to negative to disable
        center_limit: float = 0.5,
    ) -> None:
        self.add(point, empty_end)
        end = self.query(point)
        current = start.copy()
        direction = [point[dim] - start[dim] for dim in self.rangel]
        sign = [0] * self.dims
        for dim in self.rangel:
            if direction[dim] > 0:
                sign[dim] = 1
            elif direction[dim] < 0:
                sign[dim] = -1
        q = False
        visit = set[int]()
        while True:
            cnode = self.query(current)
            if cnode is None or cnode is end or cnode.id in visit:
                break
            for dim in self.rangel:
                cd = abs(current[dim] - start[dim])
                ad = abs(direction[dim])
                if cd > ad:
                    q = True
                    break
            if q:
                break
            if (
                cnode.state == cnode.FULL
                and dynamic_culling > 0
                and cnode.last_ray_id != cnode.root.ray_id
                and cnode.center_to_segment(start, point) <= center_limit
            ):
                cnode.last_ray_id = cnode.root.ray_id
                if cnode.dynamic_culling < 0:
                    cnode.dynamic_culling = dynamic_culling
                else:
                    cnode.dynamic_culling -= 1
                if cnode.dynamic_culling == 0:
                    cnode.clear_as(self.EMPTY)
                    cnode.dynamic_culling = TreeNode.dynamic_culling
            visit.add(cnode.id)
            self.add(current, empty=True)
            out_dim, current = cnode.ray_out_intersect(start, direction)

            current[out_dim] += self.min_length[out_dim] * sign[out_dim]

    def path_smoothing(
        self,
        path: list[list[float]],
        expand: list[float] = None,
        break_length: float = 1,
    ) -> tuple[bool, list[list[float]]]:
        changed = False
        if len(path) <= 1:
            return changed, path
        if break_length > 0:
            broke_path = []
            broke_path.append(path[0])
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]
                d = dist(p1, p2)
                if d < break_length:
                    broke_path.append(p2)
                else:
                    n = ceil(d / break_length)
                    for j in range(1, n + 1):
                        t = j / n
                        new_point = [p1[k] + (p2[k] - p1[k]) * t for k in self.rangel]
                        broke_path.append(new_point)
            path = broke_path
        result = list[list[float]]()
        result.append(path[0])
        i = 0
        while i < len(path) - 1:
            j = i + 2
            while j <= len(path) - 1:
                cl = self.cross_lca(path[i], path[j], expand)
                if cl:
                    changed = True
                    break
                j += 1
            result.append(path[j - 1])
            i = j - 1
        return changed, result

    def get_neighbor(self) -> set[TreeNode]:
        result = set[TreeNode]()
        lower = [0] * self.dims
        upper = [0] * self.dims
        for dim in self.rangel:
            for _dim in self.rangel:
                if dim == _dim:
                    lower[_dim] = self.i_bound_min[_dim] - 1
                    upper[_dim] = self.i_bound_max[_dim] + 1
                else:
                    upper[_dim] = self.i_center[_dim]
                    lower[_dim] = self.i_center[_dim]
            lower_node = self.root.query_i(lower)
            upper_node = self.root.query_i(upper)
            if lower_node is not None:
                result.add(lower_node)
            if upper_node is not None:
                result.add(upper_node)
        return result

    def contact_with(self, other: TreeNode) -> list[bool]:
        result = [False] * (self.dims + 1)
        true_counter = 0
        for dim in self.rangel:
            size = (self.i_bound_size[dim] + other.i_bound_size[dim]) // 2
            c2c = abs(self.i_center[dim] - other.i_center[dim])
            if size < c2c:
                return result
            true_counter += 1
            if size == c2c:
                result[dim + 1] = True
            # size < c2c: result[dim+1]= False
        result[0] = True
        return result

    def contact_center(self, other: TreeNode) -> Union[list[float], None]:
        cw = self.contact_with(other)
        if not cw[0]:
            return None
        center = [0] * self.dims
        for dim in self.rangel:
            if cw[dim + 1]:
                if self.i_center[dim] < other.i_center[dim]:
                    center[dim] = self.center[dim] + self.bound_size[dim] / 2
                else:
                    center[dim] = self.center[dim] - self.bound_size[dim] / 2
            else:
                if self.i_bound_size[dim] < other.i_bound_size[dim]:
                    center[dim] = self.center[dim]
                else:
                    center[dim] = other.center[dim]
        return center

    def interpolation_center(self, path: list[TreeNode]) -> list[list[float]]:
        result = list[list[float]]()
        if len(path) <= 0:
            return result
        result.append(path[0].center)
        for i in range(1, len(path)):
            f = path[i - 1]
            t = path[i]
            c = f.contact_center(t)
            if c is not None:
                result.append(c)
            result.append(t.center)
        return result

    def render2(
        self,
        width: int = 720,
        show_now: int = -1,
        image: ndarray = None,
        with_graph: PathGraph = None,
        with_path: list[list[float]] = None,
        _root: TreeNode = None,
    ) -> ndarray:
        if self.dims != 2:
            raise ValueError(f"tree dim {self.dims} is not 2")
        is_root = False
        if _root is None:
            is_root = True
            _root = self
        if image is None:
            image = ones((width, width, 3), dtype=uint8) * 200
        x_ratio = 1 / (_root.i_bound_max[0] - _root.i_bound_min[0]) * width
        y_ratio = 1 / (_root.i_bound_max[1] - _root.i_bound_min[1]) * width

        lt = (
            int((self.i_bound_min[0] - _root.i_bound_min[0]) * x_ratio),
            int((self.i_bound_min[1] - _root.i_bound_min[1]) * y_ratio),
        )
        rb = (
            int((self.i_bound_max[0] - _root.i_bound_min[0]) * x_ratio),
            int((self.i_bound_max[1] - _root.i_bound_min[1]) * y_ratio),
        )

        if self.state == self.FULL:
            rectangle(  # pylint: disable=no-member
                image, lt, rb, (0, 0, 128), thickness=-1
            )
        elif self.known and self.is_leaf:
            rectangle(image, lt, rb, (128, 64, 64), thickness=-1)
        rectangle(image, lt, rb, (128, 0, 0), 1)  # pylint: disable=no-member
        if self.child is not None:
            for i in self.child:
                self.child[i].render2(width=width, image=image, _root=_root)

        if is_root:
            if with_graph is not None:
                for _, edge in with_graph.edges.items():
                    ca = edge.a.tree_node.i_center
                    cb = edge.b.tree_node.i_center
                    pa = (
                        int((ca[0] - _root.i_bound_min[0]) * x_ratio),
                        int((ca[1] - _root.i_bound_min[1]) * y_ratio),
                    )
                    pb = (
                        int((cb[0] - _root.i_bound_min[0]) * x_ratio),
                        int((cb[1] - _root.i_bound_min[1]) * y_ratio),
                    )
                    line(image, pa, pb, (0, 200, 0), 1)
            if with_path is not None:
                fx_ratio = 1 / (_root.bound_max[0] - _root.bound_min[0]) * width
                fy_ratio = 1 / (_root.bound_max[1] - _root.bound_min[1]) * width
                for i in range(1, len(with_path)):
                    pa = (
                        int((with_path[i - 1][0] - _root.bound_min[0]) * fx_ratio),
                        int((with_path[i - 1][1] - _root.bound_min[1]) * fy_ratio),
                    )
                    pb = (
                        int((with_path[i][0] - _root.bound_min[0]) * fx_ratio),
                        int((with_path[i][1] - _root.bound_min[1]) * fy_ratio),
                    )
                    line(image, pa, pb, (255, 255, 255), 1)
        if show_now >= 0:
            imshow(self.__class__.__name__, image)  # pylint: disable=no-member
            waitKey(show_now)  # pylint: disable=no-member
        return image


class PathNode:
    id: int
    edges: set[int]
    tree_node: TreeNode
    f: float
    g: float = float("inf")
    h: float
    from_node: PathNode = None

    def __init__(self, tree_node: TreeNode) -> None:
        self.tree_node = tree_node
        self.id = tree_node.id
        self.f = 0
        self.h = 0
        self.edges = set[int]()

    def distance(self, other: PathNode, unknown_penalty: bool = True) -> float:
        up_factor = 0.2
        if unknown_penalty:
            if (not self.tree_node.known) or (not other.tree_node.known):
                up_factor = 1
        v = [
            (self.tree_node.center[i] - other.tree_node.center[i]) ** 2
            for i in self.tree_node.rangel
        ]
        return sqrt(sum(v)) * up_factor

    def __eq__(self, value: object) -> bool:
        return self.id == value.id

    def __hash__(self):
        return self.id


class PathEdge:
    a: PathNode
    b: PathNode
    id: tuple[int, int]
    hash: int

    @staticmethod
    def pair_code_ll(pair: tuple[int, int]) -> int:
        sum_pair = pair[0] + pair[1]
        return (sum_pair * (sum_pair + 1)) // 2 + min(*pair)

    def __init__(self, a: PathNode, b: PathNode):
        self.a = a
        self.b = b
        self.id = min(a.id, b.id), max(a.id, b.id)
        self.hash = PathEdge.pair_code_ll(self.id)


class PathGraph:
    nodes: dict[int, PathNode]
    edges: dict[int, PathEdge]
    last_root: TreeNode = None
    last_leaves: set[TreeNode] = None

    def __init__(self) -> None:
        self.nodes = dict[int, PathNode]()
        self.edges = dict[int, PathEdge]()
        self.last_leaves = set[TreeNode]()

    def add_node(self, tree_node: TreeNode) -> int:
        if tree_node.id not in self.nodes:
            self.nodes[tree_node.id] = PathNode(tree_node)
        return tree_node.id

    def remove_node(self, node_id: int, remove_edge: bool = True) -> None:
        if node_id in self.nodes:
            pn = self.nodes[node_id]
            del self.nodes[node_id]
            if remove_edge:
                for edge_id in pn.edges:
                    e = self.edges[edge_id]
                    del e.a.edges[edge_id]
                    del e.b.edges[edge_id]

    def add_edge(self, a: TreeNode, b: TreeNode) -> None:
        edge_id = PathEdge.pair_code_ll((a.id, b.id))
        if edge_id not in self.edges:
            pn1 = self.nodes[self.add_node(a)]
            pn2 = self.nodes[self.add_node(b)]
            self.edges[edge_id] = PathEdge(pn1, pn2)
            pn1.edges.add(edge_id)
            pn2.edges.add(edge_id)

    def remove_edge(self, edge_id: int, remove_from_this: bool = False) -> None:
        if edge_id in self.edges:
            pe = self.edges[edge_id]
            if remove_from_this:
                del self.edges[edge_id]
            pe.a.edges.remove(edge_id)
            pe.b.edges.remove(edge_id)

    def update(self, root: TreeNode, full_reset: bool = False) -> None:
        now_leaves = self.get_empty_leaves(root)
        if (root is not self.last_root) or full_reset:
            self.last_leaves.clear()
            if full_reset:
                self.update_edges(now_leaves)
            else:
                self.update_edges_neighbor(now_leaves)
            self.last_root = root
        else:
            self.update_edges_neighbor(now_leaves)
        self.last_leaves = now_leaves

    def get_empty_leaves(
        self, tree_node: TreeNode, leaves: set[TreeNode] = None
    ) -> set[TreeNode]:
        if leaves is None:
            leaves = set[TreeNode]()
        if tree_node.is_leaf and tree_node.state == tree_node.EMPTY:
            leaves.add(tree_node)
        elif tree_node.state != tree_node.FULL and tree_node.child is not None:
            for direction in tree_node.child:
                self.get_empty_leaves(tree_node.child[direction], leaves)
        return leaves

    def update_edges(self, leaves: set[TreeNode]):
        self.nodes.clear()
        self.edges.clear()
        leaves_list = list(leaves)
        for i, leaf in enumerate(leaves_list):
            for j in range(i + 1, len(leaves_list)):
                leaf2 = leaves_list[j]
                if leaf.intersect(leaf2):
                    self.add_edge(leaf, leaf2)

    def update_edges_neighbor(self, leaves: set[TreeNode]):
        expire_edge_ids = set[int]()
        active_nodes = set[TreeNode]()
        for edge_id, e in self.edges.items():
            tna = e.a.tree_node
            tnb = e.b.tree_node
            a_expire = tna not in leaves
            b_expire = tnb not in leaves
            if a_expire or b_expire:
                expire_edge_ids.add(edge_id)
            if a_expire:
                self.remove_node(tna.id, False)
                if not b_expire:
                    active_nodes.add(tnb)
            if b_expire:
                self.remove_node(tnb.id, False)
                if not a_expire:
                    active_nodes.add(tna)
        for edge_id in expire_edge_ids:
            self.remove_edge(edge_id, True)
        for leaf in leaves:
            if (leaf in active_nodes) or (leaf not in self.last_leaves):
                neighbors = leaf.get_neighbor()
                for neighbor in neighbors:
                    if neighbor in leaves:
                        self.add_edge(leaf, neighbor)

    def get_path(
        self, tree_start: TreeNode, tree_end: TreeNode, unknown_penalty: bool = True
    ) -> list[TreeNode]:
        result = list[TreeNode]()
        start = self.nodes.get(tree_start.id, None)
        end = self.nodes.get(tree_end.id, None)
        if start is None or end is None:
            return result
        iter_count = 0
        max_iter_limit = 100_000
        open_heap = list[tuple[float, int, PathNode]]()

        open_set_ids = set[int]()
        close_set = set[PathNode]()
        start.g = 0
        start.h = start.distance(end, unknown_penalty)
        start.f = start.g + start.h
        start.from_node = None

        heapq.heappush(open_heap, (start.f, start.id, start))
        open_set_ids.add(start.id)

        while len(open_heap) > 0:
            iter_count += 1
            _, _, current = heapq.heappop(open_heap)
            open_set_ids.discard(current.id)

            if iter_count > max_iter_limit:
                self.construct_path(current, result)
                return result
            if current == end:
                self.construct_path(current, result)
                return result
            close_set.add(current)
            for eid in current.edges:
                if eid not in self.edges:
                    continue
                e = self.edges[eid]
                neighbor: PathNode = e.b if e.a == current else e.a
                if neighbor in close_set:
                    continue
                g_score = current.g + current.distance(neighbor, unknown_penalty)
                if g_score < neighbor.g or neighbor.id not in open_set_ids:
                    neighbor.g = g_score
                    neighbor.h = neighbor.distance(end, unknown_penalty)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.from_node = current
                    heapq.heappush(open_heap, (neighbor.f, neighbor.id, neighbor))
                    open_set_ids.add(neighbor.id)
        return result

    def construct_path(self, current: PathNode, path_list: list[TreeNode]):
        while current is not None:
            path_list.append(current.tree_node)
            current = current.from_node
        path_list.reverse()
