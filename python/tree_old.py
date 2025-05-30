from __future__ import annotations
import numpy
import typing
import math
from sortedcontainers import SortedSet
import open3d as o3d
import json

hash = 0
tuple_hash = dict[tuple, int]()


def get_tuple_hash(value: tuple) -> int:
    global hash
    if value in tuple_hash:
        return tuple_hash[value]
    hash += 1
    return tuple_hash.setdefault(value, hash)


class PathNode:
    id: int
    edges: set[PathEdge]
    tree_node: TreeNode
    f: float
    g: float
    h: float
    from_node: PathNode = None

    def __init__(self, tree_node: TreeNode) -> None:
        self.tree_node = tree_node
        self.id = get_tuple_hash(self.tree_node.center)
        self.f = 0
        self.g = 0
        self.h = 0
        self.edges = set[PathEdge]()

    def distance(self, other: PathNode, unknown_penalty: bool = True) -> float:
        up_factor = 0.2
        if unknown_penalty:
            if (not self.tree_node.known) or (not other.tree_node.known):
                up_factor = 1
        v = [
            (self.tree_node.center[i] - other.tree_node.center[i]) ** 2
            for i in self.tree_node.rangel
        ]
        return math.sqrt(sum(v)) * up_factor

    def __eq__(self, value: object) -> bool:
        return self.id == value.id

    def __hash__(self) -> int:
        return self.id


class PathEdge:
    a: PathNode
    b: PathNode
    id: tuple[int, int]
    hash: int

    def __init__(self, a: PathNode, b: PathNode):
        self.a = a
        self.b = b
        self.id = min(self.a.id, self.b.id), max(self.a.id, self.b.id)
        self.hash = self.id.__hash__()

    def __eq__(self, value: object) -> bool:
        return self.id == value.id

    def __hash__(self) -> int:
        return self.hash


class PathGraph:
    nodes: dict[tuple, PathNode]
    edges: set[PathEdge]
    last_root: TreeNode = None
    last_leaves: set[TreeNode] = None

    def __init__(self) -> None:
        self.nodes = dict[tuple, PathNode]()
        self.edges = set[PathEdge]()
        self.last_leaves = set[TreeNode]()

    def add_node(self, tree_node: TreeNode) -> None:
        if tree_node.center not in self.nodes:
            self.nodes[tree_node.center] = PathNode(tree_node)

    def add_edge(self, a: TreeNode, b: TreeNode) -> None:
        na = self.find_node(a)
        nb = self.find_node(b)
        if na is None or nb is None or a is b:
            return
        edge = PathEdge(na, nb)
        if edge not in self.edges:
            self.edges.add(edge)

    def find_node(self, tree_node: TreeNode) -> typing.Union[PathNode, None]:
        return self.nodes.get(tree_node.center, None)

    def update(self, root: TreeNode, full_reset: bool = False) -> None:
        if full_reset:
            self.last_leaves.clear()
            self.nodes.clear()
            self.edges.clear()
            now_leaves = self.get_empty_leaves(root)
            self.get_edges(now_leaves)
            # self.get_edges_neighbor(now_leaves, root)
            for e in self.edges:
                self.find_node(e.a.tree_node).edges.add(e)
                self.find_node(e.b.tree_node).edges.add(e)
            return
        
        if root is not self.last_root:
            self.last_leaves = set[TreeNode]()
            self.edges.clear()

        now_leaves = self.get_empty_leaves(root)
        now_centers = set([leaf.center for leaf in now_leaves])
        expire_centers = [c for c in self.nodes if c not in now_centers]
        for ec in expire_centers:
            del self.nodes[ec]
        for c in self.nodes:
            n = self.nodes[c]
            n.from_node = None
        self.get_edges_neighbor(now_leaves, root)
        for e in self.edges:
            e.a.edges.add(e)
            e.b.edges.add(e)
        self.last_root = root
        self.last_leaves = now_leaves

    def get_empty_leaves(
        self, tree_node: TreeNode, leaves: set[TreeNode] = None
    ) -> set[TreeNode]:
        if leaves is None:
            leaves = set[TreeNode]()
        if tree_node.is_leaf and tree_node.state == tree_node.empty:
            leaves.add(tree_node)
            self.add_node(tree_node)
            return leaves
        for direction in tree_node.child:
            self.get_empty_leaves(tree_node.child[direction], leaves)
        return leaves

    def get_edges(self, leaves: set[TreeNode]):
        leaves_list = list(leaves)
        for i in range(len(leaves_list)):
            leaf = leaves_list[i]
            for j in range(i + 1, len(leaves_list)):
                leaf2 = leaves_list[j]
                if leaf.intersect(leaf2):
                    self.add_edge(leaf, leaf2)

    def get_edges_neighbor(self, leaves: set[TreeNode], root: TreeNode):
        expire_edges = set[PathEdge]()
        active_nodes = set[TreeNode]()
        for e in self.edges:
            tna = e.a.tree_node
            tnb = e.b.tree_node
            a_expire = tna.state != tna.empty or not tna.is_leaf
            b_expire = tnb.state != tnb.empty or not tnb.is_leaf
            if a_expire or b_expire:
                expire_edges.add(e)
            if a_expire and not b_expire:
                e.b.edges.remove(e)
                active_nodes.add(tnb)
            if b_expire and not a_expire:
                e.a.edges.remove(e)
                active_nodes.add(tna)

        self.edges.difference_update(expire_edges)
        for leaf in leaves:
            if (leaf not in self.last_leaves) or (leaf in active_nodes):
                neighbors = leaf.get_neighbor(root)
                for neighbor in neighbors:
                    if neighbor in leaves:
                        self.add_edge(leaf, neighbor)

    def get_edges_fast(self, leaves: set[TreeNode]):
        l_list = list(leaves)
        if len(l_list) < 0:
            return

        dims = len(l_list[0].bound_min)
        for dim in range(dims):
            events = list[tuple[float, int, int]]()
            for i, tn in enumerate(l_list):
                events.append((tn.bound_min[dim] - tn.min_length[dim] / 2, 0, i))
                events.append((tn.bound_max[dim] + tn.min_length[dim] / 2, 1, i))
            events.sort(key=lambda o: o[0])
            active = set[int]()
            for event in events:
                x, et, idx = event
                if et == 0:
                    for other_idx in active:
                        if l_list[idx].intersect(l_list[other_idx]):
                            self.add_edge(l_list[idx], l_list[other_idx])
                    active.add(idx)
                elif et == 1:
                    active.remove(idx)

    def construct_path(self, current: PathNode, path_list: list[PathNode]):
        while current is not None:
            path_list.append(current)
            current = current.from_node
        path_list.reverse()

    def get_path(
        self, tree_start: TreeNode, tree_end: TreeNode, unknown_penalty: bool = True
    ) -> list[PathNode]:
        result = list[PathNode]()
        start = self.find_node(tree_start)
        end = self.find_node(tree_end)
        if start is None or end is None:
            return result
        iterCount = 0
        maxIterLimit = 100_000
        openSet = SortedSet[PathNode](key=lambda o: o.f)
        closedSet = set[PathNode]()
        start.g = 0
        start.h = start.distance(end, unknown_penalty)
        start.f = start.g + start.h
        openSet.add(start)
        while len(openSet) > 0:
            iterCount += 1
            if iterCount > maxIterLimit:
                self.construct_path(current, result)
                return result
            current: PathNode = openSet.pop(0)
            if current == end:
                self.construct_path(current, result)
                return result
            closedSet.add(current)
            for e in current.edges:
                neighbor: PathNode = e.b if e.a == current else e.a
                if neighbor in closedSet:
                    continue
                tentative_g_score = current.g + current.distance(neighbor)
                if tentative_g_score < neighbor.g or neighbor not in openSet:
                    neighbor.g = tentative_g_score
                    neighbor.h = neighbor.distance(end)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.from_node = current
                    openSet.add(neighbor)
        return result


class TreeNode:
    empty: int = 0
    full: int = 1
    half_full: int = 2
    child: dict[int, TreeNode]
    state: int
    bit_value: list[int]
    bound: list[tuple[float, ...]]
    bound_min: tuple[float, ...]
    bound_max: tuple[float, ...]
    bound_size: tuple[float, ...]
    center: tuple[float, ...]
    min_length: tuple[float, ...]
    parent: TreeNode
    _min: bool
    inf: float = float("inf")
    dims: int
    is_leaf: bool = True
    rangel: list[int]
    known: bool = False

    def __init__(self, min_length: tuple[float, ...]) -> None:
        self.state = self.empty
        self.min_length = min_length
        self.parent = None

    def serialize(self, is_top: bool = True) -> dict[str, typing.Any]:
        obj = dict[str, typing.Any]()
        obj["bound"] = [list(bound) for bound in self.bound]
        obj["state"] = self.state
        obj["known"] = self.known
        if is_top:
            obj["min_length"] = list(self.min_length)
            obj["_class"] = self.__class__.__name__
        obj["child"] = dict[str, typing.Any]()
        for k in self.child:
            obj["child"][str(k)] = self.child[k].serialize(is_top=False)
        return obj

    def deserialize(
        obj: dict[str, typing.Any],
        is_top: bool = True,
        with_min_length: list[float] = None,
        with_class: type[TreeNode] = None,
    ) -> typing.Union[OCTreeNode, QuadTreeNode]:
        if is_top:
            with_class = globals().get(obj["_class"])
            with_min_length = tuple(obj["min_length"])
        if with_class is None:
            raise NotImplementedError(obj["_class"])
        bound = list[float]()
        for b in obj["bound"]:
            for v in b:
                bound.append(v)
        node: TreeNode = with_class(*bound, with_min_length)
        node.known = obj["known"]
        node.state = obj["state"]
        if len(obj["child"]) > 0:
            node.is_leaf = False
        for k in obj["child"]:
            node.child[int(k)] = child_node = TreeNode.deserialize(
                obj["child"][k],
                is_top=False,
                with_min_length=with_min_length,
                with_class=with_class,
            )
            child_node.parent = node
        node.update_state()
        return node

    def save(self, path: str = None):
        if path is None:
            path = f"{self.__class__.__name__}.json"
        json.dump(self.serialize(), open(path, "w"))

    def load(path: str = None) -> typing.Union[OCTreeNode, QuadTreeNode]:
        return TreeNode.deserialize(json.load(open(path)))

    def path_smoothing(
        self, path: list[tuple[float, ...]]
    ) -> tuple[bool, list[tuple[float, ...]]]:
        changed = False
        if len(path) <= 2:
            return changed, path
        result = list[tuple[float, ...]]()
        result.append(path[0])

        i = 0
        while i < len(path)-1:
            j = len(path)-1
            while j > i+1:
                if not self.cross_lca(path[i],path[j]):
                    changed = True
                    break
                j-=1
            result.append(path[j])
            i = j
        return changed, result

    def clear(self) -> None:
        self.child.clear()
        self.state = self.empty
        self.known = False
        self.is_leaf = True
        parent = self.parent
        while True:
            if parent is None:
                break
            parent.update_state()
            parent = parent.parent

    def query(
        self, point: tuple, nearest_on_oor: bool = True
    ) -> typing.Union[TreeNode, None]:
        if (not nearest_on_oor) and self.out_of_region(point):
            return None

        if self.is_leaf:
            return self
        direction = self.get_direction(point, allow_oor=True)
        if direction in self.child:
            return self.child[direction].query(point, nearest_on_oor)
        return self

    def lca(self, node1: TreeNode, node2: TreeNode) -> TreeNode:
        node1chain = list[TreeNode]()
        node2chain = list[TreeNode]()
        node1p = node1
        node2p = node2
        while node1p is not None:
            node1chain.append(node1p)
            node1p = node1p.parent
        while node2p is not None:
            node2chain.append(node2p)
            node2p = node2p.parent
        node1chain.reverse()
        node2chain.reverse()
        min_len = min(len(node1chain), len(node2chain))
        lca_node = None
        for i in range(min_len):
            if node1chain[i] is node2chain[i]:
                lca_node = node1chain[i]
            else:
                break
        return lca_node

    def cross_self(
        self, start: tuple[float, ...], invVector: tuple[float, ...]
    ) -> bool:
        rl = range(self.dims)
        tmin = -self.inf
        tmax = self.inf
        for i in rl:
            bound_min = self.bound_min[i]
            bound_max = self.bound_max[i]
            if invVector[i] is None:
                if start[i] < bound_min or start[i] > bound_max:
                    return False
            else:
                invV = invVector[i]
                t1 = (bound_min - start[i]) * invV
                t2 = (bound_max - start[i]) * invV
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
                # tmin = max(tmin, min(t1, t2))
                # tmax = min(tmax, max(t1, t2))
                if tmin > tmax:
                    return False
        return tmax >= 0 and tmin <= 1

    def cross(self, start: tuple[float, ...], invVector: tuple[float, ...]) -> bool:
        if self.state == self.empty:
            return False
        self_cross = self.cross_self(start, invVector)
        if self.state == self.full:
            return self_cross
        elif self_cross:
            for child_dir in self.child:
                if self.child[child_dir].cross(start, invVector):
                    return True
        else:
            return False

    def cross_lca(self, start: tuple[float, ...], end: tuple[float, ...]) -> bool:
        n1 = self.query(start, True)
        n2 = self.query(end, True)
        vector = [end[i] - start[i] for i in self.rangel]
        invVector = [(None if vector[i] == 0 else 1 / vector[i]) for i in self.rangel]
        return self.lca(n1, n2).cross(start, invVector)

    def get_parent(self, number: int = 1) -> TreeNode:
        parent = self
        for i in range(number):
            if parent.parent is not None:
                parent = parent.parent
            else:
                break
        return parent

    def intersect(self, other: TreeNode) -> bool:
        for dim in self.rangel:
            if (
                self.bound_min[dim] > other.bound_max[dim]
                or self.bound_max[dim] < other.bound_min[dim]
            ):
                return False
        return True

    def divide(self, recursive_depth: int = -1) -> None:
        if recursive_depth == 0 or self._min or self.state != self.empty:
            return
        else:
            ri = range(2**self.dims)
            for d in ri:
                if d not in self.child:
                    b, rd = self.get_bound_by_direction(d)
                    if rd == d:
                        self.child[d] = self.__class__(*b, self.min_length)
                        self.child[d].parent = self
                        # self.child[d].known = self.known
                        self.child[d].divide(recursive_depth - 1)
                        self.is_leaf = False
                    else:
                        self.child[d] = self.child[rd]

    def get_bound_by_direction(self, direction: int) -> tuple[list[float], int]:
        value = [0] * self.dims * 2
        reduced_direction = direction
        for dim in self.rangel:
            if self.bound_size[dim] <= self.min_length[dim]:
                reduced_direction &= ~self.bit_value[dim]
                value[dim] = self.bound_min[dim]
                value[dim + self.dims] = self.bound_max[dim]
            elif direction & self.bit_value[dim] == 0:
                value[dim] = self.bound_min[dim]
                value[dim + self.dims] = self.center[dim]
            else:
                value[dim] = self.center[dim]
                value[dim + self.dims] = self.bound_max[dim]
        return value, reduced_direction

    def update_state(self):
        if len(self.child) <= 0:
            self.is_leaf = True
            return
        full_counter = 0
        empty_counter = 0
        half_full_counter = 0
        for child_dir in self.child:
            child = self.child[child_dir]
            if child.state == self.full:
                full_counter += 1
            elif child.state == self.half_full:
                half_full_counter += 1
            elif child.state == self.empty:
                empty_counter += 1
        if empty_counter == 0 and half_full_counter == 0:
            self.state = self.full
            self.is_leaf = True
            self.child.clear()
        elif full_counter == 0 and half_full_counter == 0:
            self.state = self.empty
        else:
            self.state = self.half_full

    def add(self, point: tuple[float, ...], empty: bool = False) -> bool:
        if self.state == self.full:
            return False
        direction = self.get_direction(point)
        if direction < 0:
            return False
        if self.state == self.empty and empty:
            self.known = True
            return True
        elif self._min and not empty:
            self.state = self.full
            return True
        self.divide(1)
        changed: bool = self.child[direction].add(point, empty)
        if changed:
            self.update_state()
        return changed

    # assume intersect
    def ray_out_intersect(
        self, point: tuple[float, ...], direction: tuple[float, ...]
    ) -> tuple[float, ...]:
        t_min = -self.inf
        t_max = self.inf
        for dim in self.rangel:
            t1 = (self.bound_min[dim] - point[dim]) / direction[dim]
            t2 = (self.bound_max[dim] - point[dim]) / direction[dim]
            if t1 > t2:
                t_near = t2
                t_far = t1
            else:
                t_near = t1
                t_far = t2
            if t_min < t_near:
                t_min = t_near
            if t_max > t_far:
                t_max = t_far
        exit_t = t_max
        return tuple(point[dim] + exit_t * direction[dim] for dim in self.rangel)

    def add_raycast(
        self,
        start: tuple[float, ...],
        point: tuple[float, ...],
        empty_end: bool = False,
    ) -> None:
        self.add(point, empty_end)
        current = list(start)
        direction = [point[dim] - start[dim] for dim in self.rangel]
        sign = [1 if direction[dim] > 0 else -1 for dim in self.rangel]
        unit_dir = [self.min_length[dim] / 2 * sign[dim] for dim in self.rangel]
        while True:
            cnode = self.query(current, False)
            if (cnode is None) or (cnode.state != self.empty):
                break
            self.add(current, empty=True)
            current = cnode.ray_out_intersect(start, direction)
            current = [unit_dir[dim] + current[dim] for dim in self.rangel]

        # self.add(point, empty_end)

    def is_min(self) -> bool:
        result = True
        for i in self.rangel:
            if self.bound_size[i] < self.min_length[i]:
                pass
            else:
                result = False
        return result

    def out_of_region(self, point: tuple) -> bool:
        for i in self.rangel:
            if self.bound_min[i] > point[i] or self.bound_max[i] < point[i]:
                return True
        return False

    def get_direction(self, point: tuple, allow_oor: bool = False) -> int:
        if (not allow_oor) and self.out_of_region(point):
            return -1
        result = 0
        for i in self.rangel:
            if point[i] > self.center[i]:
                result += self.bit_value[i]
        return result

    def get_neighbor(self, root: TreeNode) -> set[TreeNode]:
        raise NotImplementedError()

    def contact_with(self, other: TreeNode) -> list[bool]:
        result = [False] * (self.dims + 1)
        for i in self.rangel:
            max_size2 = self.bound_size[i] + other.bound_size[i] + self.min_length[i]
            min_size2 = self.bound_size[i] + other.bound_size[i] - self.min_length[i]
            c2c2 = 2 * abs(self.center[i] - other.center[i])
            if max_size2 < c2c2:
                return result
            if min_size2 > c2c2:
                result[i + 1] = True
        result[0] = True
        return result

    def get_contact_face_center(
        self, other: TreeNode
    ) -> typing.Union[tuple[float, ...], None]:
        cw = self.contact_with(other)
        if not cw[0]:
            return None
        center = [0] * self.dims
        for i in self.rangel:
            if cw[i + 1]:
                if self.bound_size[i] < other.bound_size[i]:
                    center[i] = self.center[i]
                else:
                    center[i] = other.center[i]
            else:
                if self.center[i] < other.center[i]:
                    center[i] = self.center[i] + self.bound_size[i] / 2
                else:
                    center[i] = self.center[i] - self.bound_size[i] / 2
        return center


class QuadTreeNode(TreeNode):
    lt: int = 0
    rt: int = 1
    lb: int = 2
    rb: int = 3
    bit_value: list[int] = [1, 2]
    rangel: list[int] = [0, 1]
    bound: list[tuple[float, float]]
    bound_min: tuple[float, float]
    bound_max: tuple[float, float]
    center: tuple[float, float]
    dims: int = 2

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        min_length: tuple[float, float] = (1, 1),
    ) -> None:
        super().__init__(min_length)
        self.child = dict[int, QuadTreeNode]()
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        self.bound = [(x1, y1), (x2, y2)]
        self.bound_min = self.bound[0]
        self.bound_max = self.bound[1]
        self.bound_size = (x2 - x1, y2 - y1)
        self.center = (x1 / 2 + x2 / 2, y1 / 2 + y2 / 2)
        self._min = self.is_min()

    def x1(self) -> float:
        return self.bound[0][0]

    def y1(self) -> float:
        return self.bound[0][1]

    def x2(self) -> float:
        return self.bound[1][0]

    def y2(self) -> float:
        return self.bound[1][1]

    def xc(self) -> float:
        return self.center[0]

    def yc(self) -> float:
        return self.center[1]

    def get_neighbor(self, root: TreeNode) -> set[TreeNode]:
        left = (self.x1() - self.min_length[0] / 2, self.yc())
        right = (self.x2() + self.min_length[0] / 2, self.yc())
        top = (self.xc(), self.y1() - self.min_length[1] / 2)
        bottom = (self.xc(), self.y2() + self.min_length[1] / 2)

        # lt = (left[0], top[1])
        # rt = (right[0], top[1])
        # lb = (left[0], bottom[1])
        # rb = (right[0], bottom[1])
        result = set[TreeNode]()
        for direction in [
            top,
            bottom,
            left,
            right,
            #   , lt, rt, lb, rb
        ]:
            item = root.query(direction)
            if item is not None:
                result.add(item)
        return result

    def render(
        self,
        width: int = 1280,
        image: numpy.ndarray = None,
        bound: list[tuple[float, float]] = None,
        with_graph: PathGraph = None,
        with_path: list[typing.Union[PathNode, tuple[float, float]]] = None,
        with_cotact_center: bool = True,
    ) -> numpy.ndarray:
        import cv2

        is_root = False
        if bound is None:
            bound = self.bound
        if image is None:
            is_root = True
            image = numpy.ones((width, width, 3), dtype=numpy.uint8) * 200
        x_ratio = 1 / (bound[1][0] - bound[0][0]) * width
        y_ratio = 1 / (bound[1][1] - bound[0][1]) * width
        lt = (
            int((self.x1() - bound[0][0]) * x_ratio),
            int((self.y1() - bound[0][1]) * y_ratio),
        )
        rb = (
            int((self.x2() - bound[0][0]) * x_ratio),
            int((self.y2() - bound[0][1]) * y_ratio),
        )
        cv2.rectangle(image, lt, rb, (128, 0, 0), 2)
        if self.state == self.full:
            cv2.rectangle(image, lt, rb, (0, 0, 128), thickness=-1)
        elif self.known and self.is_leaf:
            cv2.rectangle(image, lt, rb, (128, 64, 64), thickness=-1)
        for child_dir in self.child:
            self.child[child_dir].render(width=width, image=image, bound=bound)

        if is_root and with_graph != None:
            for edge in with_graph.edges:
                pa = edge.a.tree_node.center
                pa = (
                    int((pa[0] - bound[0][0]) * x_ratio),
                    int((pa[1] - bound[0][1]) * y_ratio),
                )
                pb = edge.b.tree_node.center
                pb = (
                    int((pb[0] - bound[0][0]) * x_ratio),
                    int((pb[1] - bound[0][1]) * y_ratio),
                )
                cv2.line(image, pa, pb, (0, 200, 0), 2)
        if is_root and with_path != None:
            if isinstance(with_path[0], PathNode):
                for i in range(len(with_path) - 1):
                    pa = with_path[i].tree_node.center
                    pa = (
                        int((pa[0] - bound[0][0]) * x_ratio),
                        int((pa[1] - bound[0][1]) * y_ratio),
                    )
                    pb = with_path[i + 1].tree_node.center
                    pb = (
                        int((pb[0] - bound[0][0]) * x_ratio),
                        int((pb[1] - bound[0][1]) * y_ratio),
                    )
                    if not with_cotact_center:
                        cv2.line(image, pa, pb, (255, 255, 255), 2)
                    else:
                        pc = with_path[i].tree_node.get_contact_face_center(
                            with_path[i + 1].tree_node
                        )
                        pc = (
                            int((pc[0] - bound[0][0]) * x_ratio),
                            int((pc[1] - bound[0][1]) * y_ratio),
                        )
                        cv2.line(image, pa, pc, (255, 255, 255), 2)
                        cv2.line(image, pc, pb, (255, 255, 255), 2)
            elif isinstance(with_path[0], tuple):
                for i in range(len(with_path) - 1):
                    pa = (
                        int((with_path[i][0] - bound[0][0]) * x_ratio),
                        int((with_path[i][1] - bound[0][1]) * y_ratio),
                    )
                    pb = (
                        int((with_path[i + 1][0] - bound[0][0]) * x_ratio),
                        int((with_path[i + 1][1] - bound[0][1]) * y_ratio),
                    )
                    cv2.line(image, pa, pb, (255, 255, 255), 2)
        return image


class OCTreeNode(TreeNode):
    ltf: int = 0
    rtf: int = 1
    lbf: int = 2
    rbf: int = 3
    ltb: int = 4
    rtb: int = 5
    lbb: int = 6
    rbb: int = 7
    bit_value: list[int] = [1, 2, 4]
    rangel: list[int] = [0, 1, 2]
    dims: int = 3

    def __init__(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        min_length: tuple[float, float, float] = (1, 1, 1),
    ) -> None:
        super().__init__(min_length)
        self.child = dict[int, OCTreeNode]()
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        z1, z2 = min(z1, z2), max(z1, z2)
        self.bound = [(x1, y1, z1), (x2, y2, z2)]
        self.bound_min, self.bound_max = self.bound[0], self.bound[1]
        self.bound_size = (x2 - x1, y2 - y1, z2 - z1)
        self.center = (x1 / 2 + x2 / 2, y1 / 2 + y2 / 2, z1 / 2 + z2 / 2)
        self._min = self.is_min()

    def get_neighbor(self, root: TreeNode) -> set[TreeNode]:
        left = (self.x1() - self.min_length[0] / 2, self.yc(), self.zc())
        right = (self.x2() + self.min_length[0] / 2, self.yc(), self.zc())
        top = (self.xc(), self.y1() - self.min_length[1] / 2, self.zc())
        bottom = (self.xc(), self.y2() + self.min_length[1] / 2, self.zc())
        front = (self.xc(), self.yc(), self.z1() - self.min_length[1] / 2)
        back = (self.xc(), self.yc(), self.z2() + self.min_length[1] / 2)

        # ltc = (left[0], top[1], self.zc())
        # rtc = (right[0], top[1], self.zc())
        # lbc = (left[0], bottom[1], self.zc())
        # rbc = (right[0], bottom[1], self.zc())

        # lcf = (left[0], self.yc(), front[2])
        # lcb = (left[0], self.yc(), back[2])
        # rcf = (right[0], self.yc(), front[2])
        # rcb = (right[0], self.yc(), back[2])

        # ctf = (self.xc(), top[1], front[2])
        # cbf = (self.xc(), bottom[1], front[2])
        # ctb = (self.xc(), top[1], back[2])
        # cbb = (self.xc(), bottom[1], back[2])

        # ltf = (left[0], top[1], front[2])
        # rtf = (right[0], top[1], front[2])
        # lbf = (left[0], bottom[1], front[2])
        # rbf = (right[0], bottom[1], front[2])

        # ltb = (left[0], top[1], back[2])
        # rtb = (right[0], top[1], back[2])
        # lbb = (left[0], bottom[1], back[2])
        # rbb = (right[0], bottom[1], back[2])

        result = set[TreeNode]()
        for direction in [
            top,
            bottom,
            left,
            right,
            front,
            back,
            # ltc,
            # rtc,
            # lbc,
            # rbc,
            # lcf,
            # lcb,
            # rcf,
            # rcb,
            # ctf,
            # cbf,
            # ctb,
            # cbb,
            # ltf,
            # rtf,
            # lbf,
            # rbf,
            # ltb,
            # rtb,
            # lbb,
            # rbb,
        ]:
            item = root.query(direction)
            if item is not None:
                result.add(item)
        return result

    def x1(self) -> float:
        return self.bound[0][0]

    def y1(self) -> float:
        return self.bound[0][1]

    def z1(self) -> float:
        return self.bound[0][2]

    def x2(self) -> float:
        return self.bound[1][0]

    def y2(self) -> float:
        return self.bound[1][1]

    def z2(self) -> float:
        return self.bound[1][2]

    def xc(self) -> float:
        return self.center[0]

    def yc(self) -> float:
        return self.center[1]

    def zc(self) -> float:
        return self.center[2]

    def get_cube_mesh(self) -> o3d.io.TriangleMesh:
        x1, y1, z1 = self.bound_min
        x2, y2, z2 = self.bound_max
        cube = o3d.geometry.TriangleMesh.create_box(x2 - x1, y2 - y1, z2 - z1)
        # cube.compute_vertex_normals()
        cube.translate(self.bound_min)
        return cube

    def render(
        self,
        geometries: list[o3d.geometry.Geometry] = None,
        with_graph: PathGraph = None,
        with_path: list[PathNode] = None,
        with_cotact_center: bool = True,
        show_now: bool = False,
    ) -> list[o3d.geometry.Geometry]:
        is_root = False
        if geometries is None:
            is_root = True
            geometries = list[o3d.geometry.Geometry]()
        if self.state == self.full:
            geometries.append(self.get_cube_mesh())
        elif self.state == self.half_full:
            for child_dir in self.child:
                self.child[child_dir].render(geometries, show_now=False)
        if is_root:
            merge_mesh = o3d.geometry.TriangleMesh()
            for tm in geometries:
                merge_mesh += tm
            merge_mesh.remove_duplicated_triangles()
            merge_mesh.compute_vertex_normals()
            geometries = [merge_mesh]
        if with_path is not None:
            points = list[tuple[float, float, float]]()
            if not with_cotact_center:
                for pn in with_path:
                    points.append(pn.tree_node.center)
            else:
                for i in range(len(with_path) - 1):
                    points.append(with_path[i].tree_node.center)
                    points.append(
                        with_path[i].tree_node.get_contact_face_center(
                            with_path[i + 1].tree_node
                        )
                    )
                points.append(with_path[-1].tree_node.center)
            connection = [[i, i + 1] for i in range(len(points) - 1)]
            ls = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(points),
                o3d.utility.Vector2iVector(connection),
            )
            ls.paint_uniform_color((1, 0, 0))
            geometries.append(ls)
        if show_now:
            o3d.visualization.draw_geometries(geometries)
        return geometries


def quad_tree_test():
    import cv2
    import random
    import time

    root = QuadTreeNode(0, 0, 50, 50, (15, 2))
    # root.divide(3)
    graph = PathGraph()
    for test_times in range(10):
        points = [(random.random() * 50, random.random() * 50) for i in range(1)]
        # points = [(23, 23), (26, 26)]
        start = time.perf_counter()
        for p in points:
            root.add(p)
        graph.update(root)
        path = None
        path = graph.get_path(root.query((0, 0)), root.query((50, 50)))
        print(f"{(time.perf_counter()-start)*1000:.2f}ms")
        cv2.imshow(
            "QuadTreeNode",
            root.render(
                width=840, with_graph=graph, with_path=path, with_cotact_center=True
            ),
        )
        if cv2.waitKey(0) == 27:
            break


def oc_tree_test():
    import random
    import time

    size = 50
    root = OCTreeNode(0, 0, 0, size, size, size, (1, 1, 1))
    graph = PathGraph()
    for test_times in range(20):
        points = [
            (random.random() * size, random.random() * size, random.random() * size)
            for i in range(20)
        ]
        for p in points:
            root.add(p)

        start = time.perf_counter()
        graph.update(root)
        print(f"{(time.perf_counter()-start)*1000:.2f}")
        path = graph.get_path(root.query((0, 0, 0)), root.query((size, size, size)))
    root.render(with_path=path, show_now=True)


def lca_test():
    import cv2
    import random

    while True:
        size = 50
        root = QuadTreeNode(0, 0, size, size, (1, 1))
        p1 = (random.random() * size, random.random() * size)
        p2 = (random.random() * size, random.random() * size)
        root.add(p1)
        root.add(p2)
        lca_node = root.lca(root.query(p1), root.query(p2))
        pic = lca_node.render(width=640)
        cv2.imshow("QuadTreeNode", pic)
        if cv2.waitKey(0) == 27:
            break


def cross_test():
    import cv2
    import random

    size = 10
    root = QuadTreeNode(0, 0, size, size, (1, 1))
    width = 640
    while True:
        print("adding point")
        for i in range(2):
            p1 = (random.random() * size, random.random() * size)
            root.add(p1)
        cross = root.cross_lca((0, 0), (size, size))
        pic = root.render(width=width)
        cv2.line(pic, (0, 0), (width, width), (0, 0, 255) if cross else (0, 255, 0), 2)
        cv2.imshow("QuadTreeNode", pic)
        if cv2.waitKey(0) == 27:
            break


def cross_test_bench():
    import random

    size = 500
    root = QuadTreeNode(0, 0, size, size, (1, 1))
    for i in range(size * size):
        p1 = (random.random() * size, random.random() * size)
        root.add(p1)
        cross = root.cross_lca((0, 0), (size, size))


def raycast_test():
    import cv2
    import random
    import time

    root = QuadTreeNode(0, 0, 50, 50, (1, 1))
    graph = PathGraph()

    for test_times in range(100):
        points = [(random.random() * 50, random.random() * 50) for i in range(1)]
        # points = [(23, 23), (26, 26)]
        start = time.perf_counter()
        for p in points:
            root.add_raycast((0, 0, 0), p)
        graph.update(root)
        path = None
        path = graph.get_path(root.query((0, 0)), root.query((50, 50)))
        print(f"{(time.perf_counter()-start)*1000:.2f}ms")
        cv2.imshow(
            "QuadTreeNode",
            root.render(
                width=840, with_graph=graph, with_path=path, with_cotact_center=True
            ),
        )
        if cv2.waitKey(0) == 27:
            break


def serialize_test():
    import random
    import json
    import cv2

    root = QuadTreeNode(0, 0, 50, 50, (1, 1))
    points = [(random.random() * 50, random.random() * 50) for i in range(10)]
    for p in points:
        root.add_raycast((0, 0, 0), p)
    open("t.json", "w").write(json.dumps(root.serialize()))
    cv2.imshow("QuadTree", root.render(width=840))


def deserialize_test():
    import json
    import cv2

    serialize_test()
    root: QuadTreeNode = TreeNode.deserialize(json.load(open("t.json")))
    cv2.imshow("Deserialized QuadTree", root.render(width=840))
    cv2.waitKey(0)


def path_smoothing_test():
    import cv2
    import random

    root = QuadTreeNode(0, 0, 50, 50, (1, 1))
    points = [(random.random() * 50, random.random() * 50) for i in range(50)]
    for p in points:
        root.add_raycast((0, 0, 0), p)
    graph = PathGraph()
    graph.update(root)
    path = graph.get_path(root.query((0, 0)), root.query((50, 50)))
    path_point = []
    path_point.append(path[0].tree_node.center)
    for i in range(1, len(path)):
        path_point.append(
            path[i - 1].tree_node.get_contact_face_center(path[i].tree_node)
        )
        path_point.append(path[i].tree_node.center)
    cv2.imshow(
        "Path",
        root.render(width=840, with_graph=graph, with_path=path_point),
    )
    _, path_smoothed = root.path_smoothing(path_point)
    print(path_smoothed)
    cv2.imshow(
        "Smoothed Path",
        root.render(width=840, with_graph=graph, with_path=path_smoothed),
    )
    cv2.waitKey(0)


if __name__ == "__main__":
    import cProfile

    # path_smoothing_test()
    # deserialize_test()
    # serialize_test()
    # raycast_test()
    quad_tree_test()
    # oc_tree_test()
    # cProfile.run("quad_tree_test()", sort="tottime")
    # cProfile.run("oc_tree_test()", sort="tottime")
    # cross_test()
    # cProfile.run("cross_test_bench()", sort="tottime")
