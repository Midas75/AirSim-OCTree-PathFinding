from __future__ import annotations

import os
import typing
import time
import math

import numpy
import cv2
import cppyy


def init_ctree(c_tree_dim: int = 3):
    start = time.perf_counter()
    global C_TREE_DIM
    C_TREE_DIM = c_tree_dim
    cppyy.cppdef(f"#define TREE_DIM {C_TREE_DIM}")
    cppyy.include(f"{os.path.dirname(__file__)}/ctree.hpp")
    print(f"include cost {1000*(time.perf_counter()-start)} ms")


def farray(l: list[float]) -> object:
    a = cppyy.gbl.std.array["float", C_TREE_DIM]()
    rl = range(min(len(l), C_TREE_DIM))
    for i in rl:
        a[i] = l[i]
    return a


class CTreeNodeChildWrapper:
    holder: CTreeNode

    def __init__(self, holder: CTreeNode):
        self.holder = holder

    def __getitem__(self, i: int) -> typing.Union[CTreeNode, None]:
        c = self.holder._c.child[i]
        if cppyy.addressof(c) != 0:
            return CTreeNode(c)
        return None

    def __contains__(self, i: int) -> bool:
        return cppyy.addressof(self.holder._c.child[i]) != 0

    def __iter__(self) -> typing.Iterator[int]:
        result = []
        for i in range(cppyy.gbl.ctree.TREE_CHILDS):
            if cppyy.addressof(self.holder._c.child[i]) != 0:
                result.append(i)
        return iter(result)


class CTreeNode:
    _c: object

    def __init__(self, _c: object = None):
        if _c is None:
            self._c = cppyy.gbl.std.make_shared[cppyy.gbl.ctree.TreeNode]()
        else:
            self._c = _c

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

    def clear_as(self, state: typing.Literal[0, 1, 2]) -> None:
        return self._c.clear_as(state)

    def cross_lca(
        self, start: list[float], end: list[float], expand: list[float] = None
    ) -> bool:
        if expand is None:
            return self._c.cross_lca(farray(start), farray(end))
        return self._c.cross_lca(farray(start), farray(end), farray(expand))

    def add(self, point: list[float], empty: bool = False) -> bool:
        return self._c.add(farray(point), empty)

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

    @property
    def i_center(self) -> list[int]:
        return list(self._c.i_center)

    @property
    def i_bound_size(self) -> list[int]:
        return list(self._c.i_bound_size)

    @property
    def i_bound_max(self) -> list[int]:
        return list(self._c.i_bound_max)

    @property
    def i_bound_min(self) -> list[int]:
        return list(self._c.i_bound_min)

    @property
    def bound_size(self) -> list[float]:
        return list(self._c.bound_size)

    @property
    def dims(self) -> int:
        return self._c.dims

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
    def child(self) -> typing.Union[None, CTreeNodeChildWrapper]:
        if self._c.no_child:
            return None
        return CTreeNodeChildWrapper(self)

    @property
    def bound_max(self) -> list[float]:
        return list(self._c.bound_max)

    @property
    def bound_min(self) -> list[float]:
        return list(self._c.bound_min)

    @property
    def FULL(self) -> int:
        return cppyy.gbl.ctree.TreeNode.FULL

    @property
    def EMPTY(self) -> int:
        return cppyy.gbl.ctree.TreeNode.EMPTY

    @property
    def HALF_FULL(self) -> int:
        return cppyy.gbl.ctree.TreeNode.HALF_FULL

    def render2(
        self,
        width: int = 720,
        show_now: int = -1,
        image: numpy.ndarray = None,
        with_graph: PathGraph = None,
        with_path: list[list[float]] = None,
        _root: CTreeNode = None,
    ) -> numpy.ndarray:
        if self.dims != 2:
            raise ValueError(f"tree dim {self.dims} is not 2")
        is_root = False
        if _root is None:
            is_root = True
            _root = self
        if image is None:
            image = numpy.ones((width, width, 3), dtype=numpy.uint8) * 200
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
            cv2.rectangle(  # pylint: disable=no-member
                image, lt, rb, (0, 0, 128), thickness=-1
            )
        elif self.known and self.is_leaf:
            cv2.rectangle(image, lt, rb, (128, 64, 64), thickness=-1)
        cv2.rectangle(image, lt, rb, (128, 0, 0), 1)  # pylint: disable=no-member
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
                    cv2.line(image, pa, pb, (0, 200, 0), 1)
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
                    cv2.line(image, pa, pb, (255, 255, 255), 1)
        if show_now >= 0:
            cv2.imshow(self.__class__.__name__, image)  # pylint: disable=no-member
            cv2.waitKey(show_now)  # pylint: disable=no-member
        return image


class CTreeNodeTest:
    @staticmethod
    def add_test():
        start = time.perf_counter()
        tn = CTreeNode().as_root([0, 0, 0], [50, 50, 50], [1, 1, 1])
        print(f"as_root cost {1000*(time.perf_counter()-start)} ms")
        start = time.perf_counter()
        tn.add([1, 1, 1])
        print(f"add cost {1000*(time.perf_counter()-start)} ms")
        start = time.perf_counter()
        print(tn.i_center, tn.i_bound_size, tn.bound_size)
        print(f"print cost {1000*(time.perf_counter()-start)} ms")

    @staticmethod
    def query_test():
        start = time.perf_counter()
        tn = CTreeNode().as_root([0, 0, 0], [50, 50, 50], [1, 1, 1])
        print(f"as_root cost {1000*(time.perf_counter()-start)} ms")
        start = time.perf_counter()
        tn.add([1, 1, 1])
        print(f"add cost {1000*(time.perf_counter()-start)} ms")
        start = time.perf_counter()
        print(tn.query([1, 1, 1]).i_center)
        print(f"query cost {1000*(time.perf_counter()-start)} ms")

    @staticmethod
    def render2_test():
        tn = CTreeNode().as_root([0, 0], [50, 50], [1, 1])
        tn.add([1, 1])
        tn.add([30, 30])
        tn.render2(show_now=0)

    @staticmethod
    def raycast_test():
        tn = CTreeNode()
        tn.as_root([0, 0], [640, 640], [2, 2])
        number = 500
        for i in range(number):
            p = [
                tn.bound_size[0] * math.sin(math.pi / 2 * i / number),
                tn.bound_size[1] * math.cos(math.pi / 2 * i / number),
            ]
            start = time.perf_counter()
            tn.add_raycast([0, 0], p, False)
            print(f"add_raycast cost {1000*(time.perf_counter()-start)} ms")
            tn.render2(show_now=0)

    @staticmethod
    def raycast_benchmark_test():
        tn = CTreeNode().as_root([0, 0], [640, 640], [2, 2])
        number = 501
        for i in range(number):
            if i == 1:
                start = time.perf_counter()
            p = [
                tn.bound_size[0] * math.sin(math.pi / 2 * i / number),
                tn.bound_size[1] * math.cos(math.pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
        print(
            f"add_raycast {number-1} times cost {1000*(time.perf_counter()-start)} ms"
        )


if __name__ == "__main__":
    init_ctree()
    CTreeNodeTest.raycast_benchmark_test()
