# pylint: disable=C0114  # I'm lazy
# pylint: disable=C0115  # I'm lazy
# pylint: disable=C0116  # I'm lazy
# pylint: disable=C0302  # I'm long
from __future__ import annotations

import os
from typing import Union, Iterator, Literal, Any
from time import perf_counter, sleep
from math import sin, cos, pi
import gc
import json
import gzip

from numpy import ones, ndarray, uint8
from cv2 import rectangle, line, imshow, waitKey  # pylint: disable=no-name-in-module
from cppyy import cppdef, include, gbl, addressof, set_debug
from open3d import geometry, utility, visualization

try:
    import tqdm
    import matplotlib.pyplot as plt
    import psutil
except ModuleNotFoundError as e:
    print(f"[{e}] Import test related modules failed!")
C_TREE_DIM = 3
CACHE_C_TREE_PROPS = True
_F_STD_ARRAY = None
_I_STD_ARRAY = None
_TREENODE_STD_VECTOR = None
_FARRAY_STD_VECTOR = None
_LL_STD_PAIR = None

_TREENODE_DATA_STD_VECTOR = None


def init_ctree(
    c_tree_dim: int = 3,
    cache_c_tree_props: bool = True,
    warm_up: bool = True,
    debug: bool = False,
):
    start = perf_counter()
    global C_TREE_DIM, CACHE_C_TREE_PROPS  # pylint: disable=global-statement
    global _F_STD_ARRAY, _I_STD_ARRAY, _TREENODE_STD_VECTOR, _FARRAY_STD_VECTOR, _LL_STD_PAIR  # pylint: disable=global-statement
    global _TREENODE_DATA_STD_VECTOR  # pylint: disable=global-statement
    C_TREE_DIM = c_tree_dim
    CACHE_C_TREE_PROPS = cache_c_tree_props
    if debug:
        set_debug()
    cppdef(f"#define TREE_DIM {C_TREE_DIM}")
    include(f"{os.path.dirname(__file__)}/../cpp/ctree.hpp")
    _F_STD_ARRAY = gbl.std.array["float", C_TREE_DIM]
    _I_STD_ARRAY = gbl.std.array["int", C_TREE_DIM]
    _TREENODE_STD_VECTOR = gbl.std.vector[gbl.ctree.TreeNode.Ptr]
    _FARRAY_STD_VECTOR = gbl.std.vector[_F_STD_ARRAY]
    _LL_STD_PAIR = gbl.std.pair["std::uint32_t", "std::uint32_t"]
    _TREENODE_DATA_STD_VECTOR = gbl.std.vector[gbl.ctree.TreeNodeData]
    print(f"init cost {1000*(perf_counter()-start)} ms")
    if warm_up:
        start = perf_counter()
        CTreeNode.warm_up()
        print(f"warmup cost {1000*(perf_counter()-start)} ms")


def farray(l: list[float]) -> object:
    lenl = len(l)
    if lenl == C_TREE_DIM:
        return _F_STD_ARRAY(l)
    if lenl > C_TREE_DIM:
        return _F_STD_ARRAY(l[:C_TREE_DIM])
    return _F_STD_ARRAY(l + [0] * (C_TREE_DIM - lenl))


def iarray(l: list[int]) -> object:
    lenl = len(l)
    if lenl == C_TREE_DIM:
        return _I_STD_ARRAY(l)
    if lenl > C_TREE_DIM:
        return _I_STD_ARRAY(l[:C_TREE_DIM])
    return _I_STD_ARRAY(l + [0] * (C_TREE_DIM - lenl))


def llpair(ll: tuple[int, int]) -> object:
    return _LL_STD_PAIR(*ll)


class CTreeNodeChildWrapper:
    holder: CTreeNode

    def __init__(self, holder: CTreeNode):
        self.holder = holder

    def __getitem__(self, i: int) -> Union[CTreeNode, None]:
        c = self.holder._c.child[i]
        if addressof(c) != 0:
            return CTreeNode(c)
        return None

    def __contains__(self, i: int) -> bool:
        return addressof(self.holder._c.child[i]) != 0

    def __iter__(self) -> Iterator[int]:
        result = []
        for i in range(gbl.ctree.TREE_CHILDS):
            if addressof(self.holder._c.child[i]) != 0:
                result.append(i)
        return iter(result)


class CPathEdgeWrapper:
    holder: CPathGraph

    def __init__(self, holder: CPathGraph):
        self.holder = holder

    def __iter__(self) -> Iterator[CPathEdge]:
        result = []
        it = self.holder._c.edges.begin()
        end = self.holder._c.edges.end()
        while it != end:
            result.append(CPathEdge(it.__deref__().second))
            it.__preinc__()
        return iter(result)


class CTreeNode:
    _c: object
    _i_bound_max: list[int] = None
    _i_bound_min: list[int] = None
    _i_bound_size: list[int] = None
    _i_center: list[int] = None
    _bound_max: list[float] = None
    _bound_min: list[float] = None
    _bound_size: list[float] = None
    _center: list[float] = None
    _min_length: list[float] = None
    _dims: int = None
    _ctncw: CTreeNodeChildWrapper = None
    _id: int = None

    @staticmethod
    def warm_up():
        tn = __class__()
        tn.as_root([0] * C_TREE_DIM, [100] * C_TREE_DIM, [1] * C_TREE_DIM)
        _ = (tn.EMPTY, tn.FULL, tn.HALF_FULL)
        tn.clear_as(tn.EMPTY)
        tn.divide()
        tn.add([50] * C_TREE_DIM)
        tn.add_i([0] * C_TREE_DIM)
        tn.add_raycast([0] * C_TREE_DIM, [100] * C_TREE_DIM)
        tn.deserialize(tn.serialize())
        _ = (
            tn.bound_max,
            tn.bound_min,
            tn.bound_size,
            tn.center,
            tn.i_bound_max,
            tn.i_bound_min,
            tn.i_bound_size,
            tn.i_center,
            tn.is_leaf,
        )
        for _ in tn.child:
            pass
        _ = tn.dims
        tn.cross_lca([0] * C_TREE_DIM, [1] * C_TREE_DIM)

    def __init__(self, _c: object = None):
        if _c is None:
            self._c = gbl.ctree.TreeNode.create()
        else:
            self._c = _c
        self._ctncw = CTreeNodeChildWrapper(self)

    def serialize(self, _result: dict[str, Any] = None) -> dict[str, Any]:
        if _result is not None:
            _result.clear()
        else:
            _result = dict[str, Any]()
        td = gbl.ctree.TreeData()
        tnd = _TREENODE_DATA_STD_VECTOR()
        self._c.serialize(td, tnd)
        dims = td.dims
        _result["min_length"] = [0] * dims
        _result["bound_min"] = [0] * dims
        _result["bound_max"] = [0] * dims
        for dim in range(dims):
            _result["min_length"][dim] = td.min_length[dim]
            _result["bound_min"][dim] = td.bound_min[dim]
            _result["bound_max"][dim] = td.bound_max[dim]
        _result["nodes"] = dict[str, Any]()
        tndl = list(tnd)
        for item in tndl:
            info = dict[str, Any]()
            info["i_bound_min"] = [0] * dims
            info["i_bound_max"] = [0] * dims
            for dim in range(dims):
                info["i_bound_min"][dim] = item.i_bound_min[dim]
                info["i_bound_max"][dim] = item.i_bound_max[dim]
            info["state"] = item.state
            info["known"] = bool(item.known)
            info["is_leaf"] = bool(item.is_leaf)
            info["child"] = dict[str, int]()
            for i in range(2**dims):
                if item.child[i] == 0:
                    continue
                info["child"][str(i)] = item.child[i]
            _result["nodes"][str(item.id)] = info
        return _result

    def save(self, path: str = None) -> None:
        if path is None:
            path = f"{__class__.__name__}.json.gz"
        with open(path, "wb") as f:
            j = json.dumps(self.serialize())
            gz = gzip.compress(j.encode("utf-8"))
            f.write(gz)

    @staticmethod
    def deserialize(obj: dict[str, Any]) -> CTreeNode:
        td = gbl.ctree.TreeData()
        tndv = _TREENODE_DATA_STD_VECTOR()
        dims = len(obj["bound_min"])
        td.dims = dims
        for dim in range(dims):
            td.min_length[dim] = obj["min_length"][dim]
            td.bound_min[dim] = obj["bound_min"][dim]
            td.bound_max[dim] = obj["bound_max"][dim]
        for str_id, item in obj["nodes"].items():
            tnd = gbl.ctree.TreeNodeData()
            tnd.id = int(str_id)
            tnd.state = item["state"]
            tnd.known = item["known"]
            tnd.is_leaf = item["is_leaf"]
            for dim in range(dims):
                tnd.i_bound_min[dim] = item["i_bound_min"][dim]
                tnd.i_bound_max[dim] = item["i_bound_max"][dim]
            if "child" in item:
                for str_direction, int_child_id in item["child"].items():
                    tnd.child[int(str_direction)] = int_child_id
            tndv.emplace_back(tnd)
        return CTreeNode(gbl.ctree.TreeNode.deserialize(td, tndv))

    @staticmethod
    def load(path: str = None) -> CTreeNode:
        if path is None:
            path = f"{__class__.__name__}.json.gz"
        with open(path, "rb") as f:
            j = gzip.decompress(f.read()).decode("utf-8")
            return CTreeNode.deserialize(json.loads(j))

    def as_root(
        self, bound_min: list[float], bound_max: list[float], min_length: list[float]
    ) -> CTreeNode:
        self._c.as_root(
            farray(bound_min), farray(bound_max), farray(min_length), len(bound_min)
        )
        return self

    def divide(self, depth: int = 1) -> None:
        return self._c.divide(depth)

    def query(self, point: list[float], nearest_on_oor: bool = False) -> CTreeNode:
        result = self._c.query(farray(point), nearest_on_oor)
        if addressof(result) == 0:
            return None
        return CTreeNode(result)

    def clear_as(self, state: Literal[0, 1, 2] = 0) -> None:
        return self._c.clear_as(state)

    def cross_lca(
        self, start: list[float], end: list[float], expand: list[float] = None
    ) -> bool:
        if expand is None:
            return self._c.cross_lca(farray(start), farray(end))
        return self._c.cross_lca(farray(start), farray(end), farray(expand))

    def add(self, point: list[float], empty: bool = False) -> bool:
        return self._c.add(farray(point), empty)

    def add_i(self, point: list[int], empty: bool = False) -> bool:
        return self._c.add_i(iarray(point), empty)

    def add_raycast(
        self,
        start: list[float],
        point: list[float],
        empty_end: bool = False,
        dynamic_culling: int = 20,
        culling_min_ratio: float = 0.2,
        culling_max_ratio: float = 0.8,
    ) -> None:
        return self._c.add_raycast(
            farray(start),
            farray(point),
            empty_end,
            dynamic_culling,
            culling_min_ratio,
            culling_max_ratio,
        )

    def path_smoothing(
        self, path: list[list[float]], expand: list[float] = None
    ) -> tuple[bool, list[list[float]]]:
        fsv = _FARRAY_STD_VECTOR([farray(item) for item in path])
        o_path = _FARRAY_STD_VECTOR()
        if expand is None:
            changed = self._c.path_smoothing(fsv, o_path)
        else:
            changed = self._c.path_smoothing(fsv, o_path, farray(expand))
        return changed, [list(o) for o in o_path]

    @property
    def i_center(self) -> list[int]:
        if (not CACHE_C_TREE_PROPS) or (self._i_center is None):
            self._i_center = list(self._c.i_center)
        return self._i_center[: self.dims]

    @property
    def i_bound_size(self) -> list[int]:
        if (not CACHE_C_TREE_PROPS) or (self._i_bound_size is None):
            self._i_bound_size = list(self._c.i_bound_size)
        return self._i_bound_size[: self.dims]

    @property
    def i_bound_max(self) -> list[int]:
        if (not CACHE_C_TREE_PROPS) or (self._i_bound_max is None):
            self._i_bound_max = list(self._c.i_bound_max)
        return self._i_bound_max[: self.dims]

    @property
    def i_bound_min(self) -> list[int]:
        if (not CACHE_C_TREE_PROPS) or (self._i_bound_min is None):
            self._i_bound_min = list(self._c.i_bound_min)
        return self._i_bound_min[: self.dims]

    @property
    def center(self) -> list[float]:
        if (not CACHE_C_TREE_PROPS) or (self._center is None):
            self._center = list(self._c.center)
        return self._center[: self.dims]

    @property
    def bound_size(self) -> list[float]:
        if (not CACHE_C_TREE_PROPS) or (self._bound_size is None):
            self._bound_size = list(self._c.bound_size)
        return self._bound_size[: self.dims]

    @property
    def bound_max(self) -> list[float]:
        if (not CACHE_C_TREE_PROPS) or (self._bound_max is None):
            self._bound_max = list(self._c.bound_max)
        return self._bound_max[: self.dims]

    @property
    def bound_min(self) -> list[float]:
        if (not CACHE_C_TREE_PROPS) or (self._bound_min is None):
            self._bound_min = list(self._c.bound_min)
        return self._bound_min[: self.dims]

    @property
    def min_length(self) -> list[float]:
        if (not CACHE_C_TREE_PROPS) or (self._min_length is None):
            self._min_length = list(self._c.min_length)
        return self._min_length[: self.dims]

    @property
    def dims(self) -> int:
        if (not CACHE_C_TREE_PROPS) or (self._dims is None):
            self._dims = self._c.dims
        return self._dims

    @property
    def FULL(self) -> int:  # pylint: disable=C0103
        return gbl.ctree.TreeNode.FULL

    @property
    def EMPTY(self) -> int:  # pylint: disable=C0103
        return gbl.ctree.TreeNode.EMPTY

    @property
    def HALF_FULL(self) -> int:  # pylint: disable=C0103
        return gbl.ctree.TreeNode.HALF_FULL

    @property
    def state(self) -> int:
        return self._c.state

    @property
    def known(self) -> bool:
        return self._c.known

    @property
    def is_leaf(self) -> bool:
        return self._c.is_leaf

    @property
    def child(self) -> Union[None, CTreeNodeChildWrapper]:
        if self._c.no_child:
            return None
        return self._ctncw

    @property
    def id(self) -> int:
        if (not CACHE_C_TREE_PROPS) or (self._id is None):
            self._id = self._c.id
        return self._id

    def interpolation_center(self, path: list[CTreeNode]) -> list[list[float]]:
        if len(path) < 1:
            return []
        fsv = _FARRAY_STD_VECTOR()
        tsv = _TREENODE_STD_VECTOR(
            [p._c for p in path]  # pylint: disable=protected-access
        )
        dims = path[0].dims
        self._c.interpolation_center(tsv, fsv)
        return [list(item)[:dims] for item in fsv]

    def render2(
        self,
        width: int = 720,
        show_now: int = -1,
        image: ndarray = None,
        with_graph: CPathGraph = None,
        with_path: list[list[float]] = None,
        _root: CTreeNode = None,
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
            rectangle(image, lt, rb, (0, 0, 128), thickness=-1)
        elif self.known and self.is_leaf:
            rectangle(image, lt, rb, (128, 64, 64), thickness=-1)
        rectangle(image, lt, rb, (128, 0, 0), 1)
        if self.child is not None:
            for i in self.child:
                self.child[i].render2(width=width, image=image, _root=_root)

        if is_root:
            if with_graph is not None:
                for edge in with_graph.edges:
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
            imshow(self.__class__.__name__, image)
            waitKey(show_now)
        return image

    def render3(
        self,
        show_now: int = -1,
        geometries: list[geometry.Geometry] = None,
        with_path: list[list[float]] = None,
        with_coordinate: bool = True,
    ) -> list[geometry.Geometry]:
        is_root = False
        if geometries is None:
            is_root = True
            geometries = list[geometry.Geometry]()
        if self.state == self.FULL:
            geometries.append(
                geometry.TriangleMesh.create_box(
                    self.bound_size[0], self.bound_size[1], self.bound_size[2]
                ).translate(self.bound_min)
            )
        elif self.state == self.HALF_FULL and self.child is not None:
            for direction in self.child:
                self.child[direction].render3(geometries=geometries)
        if is_root:
            merge_mesh = geometry.TriangleMesh()
            for tm in geometries:
                merge_mesh += tm
            merge_mesh.remove_duplicated_triangles()
            merge_mesh.compute_vertex_normals()
            geometries = [merge_mesh]
            if with_coordinate:
                geometries.append(
                    geometry.TriangleMesh.create_coordinate_frame(
                        size=max(*self.bound_size)
                    )
                )
            if with_path is not None and len(with_path) > 1:
                connection = [[i, i + 1] for i in range(len(with_path) - 1)]
                ls = geometry.LineSet(
                    utility.Vector3dVector(with_path),
                    utility.Vector2iVector(connection),
                )
                ls.paint_uniform_color((1, 0, 0))
                geometries.append(ls)
        if show_now >= 0:
            visualization.draw_geometries(geometries)  # pylint: disable=no-member
        return geometries


class CPathNode:
    _c: object

    def __init__(self, _c: object = None) -> None:
        self._c = _c

    @property
    def tree_node(self) -> CTreeNode:
        # self._c.tree_node is not a shared_ptr, so do not use it
        return CTreeNode(self._c.tree_node.shared_from_this())


class CPathEdge:
    _c: object

    def __init__(self, _c: object = None) -> None:
        self._c = _c

    @property
    def a(self) -> CPathEdge:
        return CPathNode(self._c.a)

    @property
    def b(self) -> CPathEdge:
        return CPathNode(self._c.b)


class CPathGraph:
    _c: object
    _cpew: CPathEdgeWrapper

    def __init__(self, _c: object = None) -> None:
        if _c is None:
            self._c = gbl.ctree.PathGraph()
        else:
            self._c = _c
        self._cpew = CPathEdgeWrapper(self)

    def update(self, root: CTreeNode, full_reset: bool = False) -> None:
        return self._c.update(root._c, full_reset)  # pylint: disable=protected-access

    def get_path(
        self, tree_start: CTreeNode, tree_end: CTreeNode, unknown_penalty: bool = True
    ) -> list[CTreeNode]:
        std_vector = _TREENODE_STD_VECTOR()
        self._c.get_path(
            tree_start._c,  # pylint: disable=protected-access
            tree_end._c,  # pylint: disable=protected-access
            std_vector,
            unknown_penalty,
        )
        if std_vector.size() > 0:
            return [CTreeNode(_c) for _c in std_vector]
        return []

    @property
    def edges(self) -> CPathEdgeWrapper:
        return self._cpew


class CTreeNodeTest:
    @staticmethod
    def add_test():
        start = perf_counter()
        tn = CTreeNode().as_root([0, 0, 0], [50, 50, 50], [1, 1, 1])
        print(f"as_root cost {1000*(perf_counter()-start)} ms")
        start = perf_counter()
        tn.add([1, 1, 1])
        print(f"add cost {1000*(perf_counter()-start)} ms")
        start = perf_counter()
        print(tn.i_center, tn.i_bound_size, tn.bound_size)
        print(f"print cost {1000*(perf_counter()-start)} ms")

    @staticmethod
    def query_test():
        start = perf_counter()
        tn = CTreeNode().as_root([0, 0, 0], [50, 50, 50], [1, 1, 1])
        print(f"as_root cost {1000*(perf_counter()-start)} ms")
        start = perf_counter()
        tn.add([1, 1, 1])
        print(f"add cost {1000*(perf_counter()-start)} ms")
        start = perf_counter()
        print(tn.query([1, 1, 1]).i_center)
        print(f"query cost {1000*(perf_counter()-start)} ms")

    @staticmethod
    def render2_test():
        tn = CTreeNode().as_root([0, 0], [50, 50], [2, 2])
        tn.add([1, 1])
        tn.add([30, 30])
        tn.render2(show_now=0)

    @staticmethod
    def render2_benchmark_test():
        tn = CTreeNode().as_root([0, 0], [640, 640], [2, 2])
        number = 500
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
            start = perf_counter()
            tn.render2(show_now=-1)
            print(f"render2 cost {1000*(perf_counter()-start)} ms")

    @staticmethod
    def raycast_test():
        tn = CTreeNode()
        tn.as_root([0, 0], [640, 640], [2, 2])
        number = 500
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            start = perf_counter()
            tn.add_raycast([0, 0], p, False)
            print(f"add_raycast cost {1000*(perf_counter()-start)} ms")
            tn.render2(show_now=0)

    @staticmethod
    def raycast_benchmark_test():
        tn = CTreeNode().as_root([0, 0], [640, 640], [2, 2])
        number = 500
        start = perf_counter()
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
        print(f"add_raycast {number} times cost {1000*(perf_counter()-start)} ms")

    @staticmethod
    def memory_safety_test():
        test_number = 10000
        rest_check = 5000
        rest_interval_ms = 1
        mem_mb = [0] * (test_number + rest_check)
        process = psutil.Process(os.getpid())
        number = 500
        ps = [
            [
                640 * sin(pi / 2 * i / number),
                640 * cos(pi / 2 * i / number),
            ]
            for i in range(number)
        ]
        for t in tqdm.tqdm(range(test_number)):
            tn = CTreeNode().as_root([0, 0], [640, 640], [2, 2])
            for i in range(number):
                tn.add_raycast([0, 0], ps[i], False)
            mem_mb[t] = process.memory_info().rss / 1024 / 1024
        gc.collect()
        for r in tqdm.tqdm(range(rest_check)):
            sleep(rest_interval_ms / 1000)
            mem_mb[test_number + r] = process.memory_info().rss / 1024 / 1024
        plt.plot(list(range(test_number + rest_check)), mem_mb)
        plt.xlabel("Test number")
        plt.ylabel("Memory (MB)")
        plt.show()

    @staticmethod
    def std_array_test():
        test_number = 1000000
        for _ in tqdm.tqdm(range(test_number), desc="f2"):
            farray([0, 0])
        for _ in tqdm.tqdm(range(test_number), desc="f3"):
            farray([0, 0, 0])
        for _ in tqdm.tqdm(range(test_number), desc="f4"):
            farray([0, 0, 0, 0])
        for _ in tqdm.tqdm(range(test_number), desc="i2"):
            iarray([0, 0])
        for _ in tqdm.tqdm(range(test_number), desc="i3"):
            iarray([0, 0, 0])
        for _ in tqdm.tqdm(range(test_number), desc="i4"):
            iarray([0, 0, 0, 0])

    @staticmethod
    def property_test():
        tn = CTreeNode().as_root([0, 0], [50, 50], [1, 1])
        tn.add([10, 10])
        p = tn.query([5, 5])
        print("bound_min:", p.bound_min)
        print("bound_max:", p.bound_max)
        print("bound_size:", p.bound_size)
        print("center", p.center)
        print("min_length", p.min_length)
        print("i_bound_min", p.i_bound_min)
        print("i_bound_max", p.i_bound_max)
        print("i_bound_size", p.i_bound_size)
        print("i_center", p.i_center)

    @staticmethod
    def render2_pathgraph_test():
        tn = CTreeNode().as_root([0, 0], [50, 50], [1, 1])
        pg = CPathGraph()
        tn.add_raycast([0, 0], [25, 49])
        pg.update(tn)
        path = pg.get_path(tn.query([0, 0], True), tn.query([50, 50], True))
        c_path = tn.interpolation_center(path)
        _, s_path = tn.path_smoothing(c_path)
        tn.render2(show_now=0, with_graph=pg, with_path=s_path)

    @staticmethod
    def save_test():
        tn = CTreeNode().as_root([0, 0], [640, 640], [2, 2])
        number = 500

        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
        start = perf_counter()
        tn.save()
        print(f"save cost {1000*(perf_counter()-start)} ms")

    @staticmethod
    def serialize_deserialize_test():
        tn = CTreeNode().as_root([0, 0], [50, 50], [1, 1])
        number = 50
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
        start = perf_counter()
        obj = tn.serialize()
        print(f"serialization cost {1000*(perf_counter()-start)} ms")

        start = perf_counter()
        deserialized = CTreeNode.deserialize(obj)
        print(f"deserialization cost {1000*(perf_counter()-start)} ms")
        imshow("Raw", tn.render2())
        imshow("New", deserialized.render2())
        waitKey(0)

    @staticmethod
    def save_load_test():
        tn = CTreeNode().as_root([0, 0], [50, 50], [1, 1])
        number = 50
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
        start = perf_counter()
        tn.save()
        print(f"save cost {1000*(perf_counter()-start)} ms")

        start = perf_counter()
        loaded = CTreeNode.load()
        print(f"load cost {1000*(perf_counter()-start)} ms")
        imshow("Raw", tn.render2())
        imshow("New", loaded.render2())
        waitKey(0)

    @staticmethod
    def ray3d_test():
        tn = CTreeNode().as_root([0, 0, 0], [50, 50, 50], [1, 1, 1])
        number = 50
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
                tn.bound_size[2] / 2,
            ]
            tn.add_raycast([0, 0, 0], p, False, -1)
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] / 2,
                tn.bound_size[2] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0, 0], p, False, -1)
            p = [
                tn.bound_size[0] / 2,
                tn.bound_size[1] * sin(pi / 2 * i / number),
                tn.bound_size[2] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0, 0], p, False, -1)
        tn.render3(0)


if __name__ == "__main__":
    init_ctree()
    CTreeNodeTest.render2_pathgraph_test()
