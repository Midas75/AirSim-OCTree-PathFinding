from __future__ import annotations

import os
from typing import Union, Iterator, Literal
from time import perf_counter, sleep
from math import sin, cos, pi
import gc

from numpy import ones, ndarray, uint8
from cv2 import rectangle, line, imshow, waitKey  # pylint: disable=no-name-in-module
from cppyy import cppdef, include, gbl, addressof

try:
    import tqdm
    import matplotlib.pyplot as plt
    import psutil
except ModuleNotFoundError as e:
    print(f"[{e}] Import test related modules failed!")
C_TREE_DIM = 3
CACHE_C_TREE_PROPS = True
_f_std_array = None
_i_std_array = None
_treenode_std_vector = None
_farray_std_vector = None
_ll_std_pair = None


def init_ctree(
    c_tree_dim: int = 3, cache_c_tree_props: bool = True, warm_up: bool = True
):
    start = perf_counter()
    global C_TREE_DIM, CACHE_C_TREE_PROPS, _f_std_array, _i_std_array, _treenode_std_vector, _farray_std_vector, _ll_std_pair  # pylint: disable=global-statement
    C_TREE_DIM = c_tree_dim
    CACHE_C_TREE_PROPS = cache_c_tree_props
    cppdef(f"#define TREE_DIM {C_TREE_DIM}")
    include(f"{os.path.dirname(__file__)}/../ctree/ctree.hpp")
    _f_std_array = gbl.std.array["float", C_TREE_DIM]
    _i_std_array = gbl.std.array["int", C_TREE_DIM]
    _treenode_std_vector = gbl.std.vector[gbl.ctree.TreeNode.Ptr]
    _farray_std_vector = gbl.std.vector[_f_std_array]
    _ll_std_pair = gbl.std.pair["long", "long"]
    print(f"init cost {1000*(perf_counter()-start)} ms")
    if warm_up:
        start = perf_counter()
        CTreeNode.warm_up()
        print(f"warmup cost {1000*(perf_counter()-start)} ms")


def farray(l: list[float]) -> object:
    lenl = len(l)
    if lenl == C_TREE_DIM:
        return _f_std_array(l)
    if lenl > C_TREE_DIM:
        return _f_std_array(l[:C_TREE_DIM])
    return _f_std_array(l + [0] * (C_TREE_DIM - lenl))


def iarray(l: list[int]) -> object:
    lenl = len(l)
    if lenl == C_TREE_DIM:
        return _i_std_array(l)
    if lenl > C_TREE_DIM:
        return _i_std_array(l[:C_TREE_DIM])
    return _i_std_array(l + [0] * (C_TREE_DIM - lenl))


def llpair(ll: tuple[int, int]) -> object:
    return _ll_std_pair(*ll)


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
        return CTreeNode(self._c.query(farray(point), nearest_on_oor))

    def clear_as(self, state: Literal[0, 1, 2]) -> None:
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
        fsv = _farray_std_vector([farray(item) for item in path])
        o_path = _farray_std_vector()
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
    def FULL(self) -> int:
        return gbl.ctree.TreeNode.FULL

    @property
    def EMPTY(self) -> int:
        return gbl.ctree.TreeNode.EMPTY

    @property
    def HALF_FULL(self) -> int:
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
                    print(pa, pb)
                    line(image, pa, pb, (255, 255, 255), 1)
        if show_now >= 0:
            imshow(self.__class__.__name__, image)
            waitKey(show_now)
        return image


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
        std_vector = _treenode_std_vector()
        self._c.get_path(
            tree_start._c,  # pylint: disable=protected-access
            tree_end._c,  # pylint: disable=protected-access
            std_vector,
            unknown_penalty,
        )
        return [CTreeNode(_c) for _c in std_vector]

    def interpolation_center(self, path: list[CTreeNode]) -> list[list[float]]:
        if len(path) < 1:
            return []
        fsv = _farray_std_vector()
        tsv = _treenode_std_vector(
            [p._c for p in path]
        )  # pylint: disable=protected-access
        dims = path[0].dims
        self._c.interpolation_center(tsv, fsv)
        return [list(item)[:dims] for item in fsv]

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
        path = pg.get_path(tn.query([0, 0]), tn.query([50, 50]))
        c_path = pg.interpolation_center(path)
        _, s_path = tn.path_smoothing(c_path)
        print(s_path)
        tn.render2(show_now=0, with_graph=pg, with_path=s_path)


if __name__ == "__main__":
    init_ctree()
    CTreeNodeTest.render2_pathgraph_test()
