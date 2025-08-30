# pylint: disable=C0114  # I'm lazy
# pylint: disable=C0115  # I'm lazy
# pylint: disable=C0116  # I'm lazy
# pylint: disable=C0302  # I'm long
from __future__ import annotations

import os
from typing import Iterator, Literal, Any
from time import perf_counter
from functools import cached_property
import json
import gzip
from tqdm import tqdm
from numpy import ones, ndarray, uint8
from cv2 import rectangle, line, imshow, waitKey  # pylint: disable=no-name-in-module
from cppyy import cppdef, include, gbl, addressof, set_debug

C_TREE_DIM = 3
_F_STD_ARRAY = None
_I_STD_ARRAY = None
_TREENODE_STD_VECTOR = None
_FARRAY_STD_VECTOR = None
_LL_STD_PAIR = None

_TREENODE_DATA_STD_VECTOR = None


def init_ctree(
    c_tree_dim: int = 3,
    debug: bool = False,
):
    start = perf_counter()
    global C_TREE_DIM  # pylint: disable=global-statement
    global _F_STD_ARRAY, _I_STD_ARRAY, _TREENODE_STD_VECTOR, _FARRAY_STD_VECTOR, _LL_STD_PAIR  # pylint: disable=global-statement
    global _TREENODE_DATA_STD_VECTOR  # pylint: disable=global-statement
    C_TREE_DIM = c_tree_dim
    if debug:
        set_debug()
    cppdef(f"#define TREE_DIM {C_TREE_DIM}")
    include(f"{os.path.dirname(__file__)}/../cpp/ctree.hpp")
    _F_STD_ARRAY = gbl.ctree.F_STD_ARRAY
    _I_STD_ARRAY = gbl.ctree.I_STD_ARRAY
    _TREENODE_STD_VECTOR = gbl.ctree.TREENODE_STD_VECTOR
    _FARRAY_STD_VECTOR = gbl.ctree.FARRAY_STD_VECTOR
    _LL_STD_PAIR = gbl.ctree.LL_STD_PAIR
    _TREENODE_DATA_STD_VECTOR = gbl.ctree.TREENODE_DATA_STD_VECTOR
    print(f"init cost {1000*(perf_counter()-start)} ms")


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


class CTreeNodeNodesWrapper:
    holder: CTreeNode

    def __init__(self, holder: CTreeNode):
        self.holder = holder

    def __getitem__(self, key: int) -> CTreeNode | None:
        if self.holder._c.nodes.count(key) > 0:
            return CTreeNode(self.holder._c.nodes.at(key))
        return None

    def __contains__(self, key: int) -> bool:
        return self.holder._c.nodes.count(key) > 0

    def __iter__(self) -> Iterator[int]:
        result = []
        for kv in self.holder._c.nodes:
            result.append(kv.first)
        return iter(result)

    def items(self) -> Iterator[tuple[int, CTreeNode]]:
        result = []
        for kv in self.holder._c.nodes:
            sp = kv.second.lock()
            if sp:
                result.append((kv.first, CTreeNode(sp)))
        return iter(result)


class CTreeNodeChildWrapper:
    holder: CTreeNode

    def __init__(self, holder: CTreeNode):
        self.holder = holder

    def __getitem__(self, i: int) -> CTreeNode | None:
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


class CPathGraphEdgesWrapper:
    holder: CPathGraph

    def __init__(self, holder: CPathGraph):
        self.holder = holder

    def __iter__(self) -> Iterator[CPathEdge]:
        result = []
        it = self.holder._c.edges.begin()
        end = self.holder._c.edges.end()
        while it != end:
            result.append(CPathEdge(it.__deref__().first))
            it.__preinc__()
        return iter(result)

    def items(self) -> Iterator[tuple[int, CPathEdge]]:
        result = []
        it = self.holder._c.edges.begin()
        end = self.holder._c.edges.end()
        while it != end:
            deref_it = it.__deref__()
            result.append((deref_it.first, CPathEdge(deref_it.second)))
            it.__preinc__()
        return iter(result)


class CTreeNode:
    _c: object
    _ctncw: CTreeNodeChildWrapper = None
    _ctnnw: CTreeNodeNodesWrapper = None

    @cached_property
    def nodes(self) -> CTreeNodeNodesWrapper:
        return self._ctnnw

    @cached_property
    def i_center(self) -> list[int]:
        return list(self._c.i_center)[: self.dims]

    @cached_property
    def i_bound_size(self) -> list[int]:
        return list(self._c.i_bound_size)[: self.dims]

    @cached_property
    def i_bound_max(self) -> list[int]:
        return list(self._c.i_bound_max)[: self.dims]

    @cached_property
    def i_bound_min(self) -> list[int]:
        return list(self._c.i_bound_min)[: self.dims]

    @cached_property
    def center(self) -> list[float]:
        return list(self._c.center)[: self.dims]

    @cached_property
    def bound_size(self) -> list[float]:
        return list(self._c.bound_size)[: self.dims]

    @cached_property
    def bound_max(self) -> list[float]:
        return list(self._c.bound_max)[: self.dims]

    @cached_property
    def bound_min(self) -> list[float]:
        return list(self._c.bound_min)[: self.dims]

    @cached_property
    def min_length(self) -> list[float]:
        return list(self._c.min_length)[: self.dims]

    @cached_property
    def dims(self) -> int:
        return self._c.dims

    @cached_property
    def FULL(self) -> int:  # pylint: disable=C0103
        return gbl.ctree.TreeNode.FULL

    @cached_property
    def EMPTY(self) -> int:  # pylint: disable=C0103
        return gbl.ctree.TreeNode.EMPTY

    @cached_property
    def HALF_FULL(self) -> int:  # pylint: disable=C0103
        return gbl.ctree.TreeNode.HALF_FULL

    @cached_property
    def state(self) -> int:
        return self._c.state

    @cached_property
    def known(self) -> bool:
        return self._c.known

    @cached_property
    def is_leaf(self) -> bool:
        return self._c.is_leaf

    @cached_property
    def parent(self) -> CTreeNode | None:
        if addressof(self._c.parent) == 0:
            return None
        return CTreeNode(self._c.parent)

    @property
    def child(self) -> None | CTreeNodeChildWrapper:
        if self._c.no_child:
            return None
        return self._ctncw

    @property
    def last_ray_id(self) -> int:
        return self._c.last_ray_id

    @property
    def ray_id(self) -> int:
        return self._c.root.ray_id

    @cached_property
    def id(self) -> int:
        return self._c.id

    def __init__(self, _c: object = None):
        if _c is None or addressof(_c) == 0:
            self._c = gbl.ctree.TreeNode.create()
        else:
            self._c = _c
        self._ctncw = CTreeNodeChildWrapper(self)
        self._ctnnw = CTreeNodeNodesWrapper(self)

    def serialize(self, progress: bool = True) -> dict[str, Any]:
        if self.parent is not None:
            raise TypeError("cannot serialize from a non-root node")
        result = dict[str, Any]()
        td = gbl.ctree.TreeData()
        tnd = _TREENODE_DATA_STD_VECTOR()
        self._c.serialize(td, tnd)
        dims = td.dims
        result["min_length"] = [0] * dims
        result["bound_min"] = [0] * dims
        result["bound_max"] = [0] * dims
        for dim in range(dims):
            result["min_length"][dim] = td.min_length[dim]
            result["bound_min"][dim] = td.bound_min[dim]
            result["bound_max"][dim] = td.bound_max[dim]
        result["nodes"] = dict[str, Any]()
        tndl = list(tnd)
        for item in tndl if not progress else tqdm(tndl):
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
            result["nodes"][str(item.id)] = info
        return result

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

    def query(
        self, point: list[float], nearest_on_oor: bool = False
    ) -> CTreeNode | None:
        result = self._c.query(farray(point), nearest_on_oor)
        if addressof(result) == 0:
            return None
        return CTreeNode(result)

    def query_i(
        self, point: list[int], nearest_on_oor: bool = False
    ) -> CTreeNode | None:
        result = self._c.query(iarray(point), nearest_on_oor)
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

    def next_ray_batch(self) -> None:
        self._c.next_ray_batch()

    def center_to_segment(self, start: list[float], point: list[float]) -> float:
        return float(self._c.center_to_segment(farray(start), farray(point)))

    def add_raycast(
        self,
        start: list[float],
        point: list[float],
        empty_end: bool = False,
        dynamic_culling: int = 10,
        center_limit: float = 0.5,
    ) -> None:
        self._c.add_raycast(
            farray(start), farray(point), empty_end, dynamic_culling, center_limit
        )

    def path_smoothing(
        self,
        path: list[list[float]],
        expand: list[float] = None,
        break_length: float = 1,
    ) -> tuple[bool, list[list[float]]]:
        fsv = _FARRAY_STD_VECTOR([farray(item) for item in path])
        o_path = _FARRAY_STD_VECTOR()
        if expand is None:
            changed = self._c.path_smoothing(
                fsv, o_path, farray([0, 0, 0]), break_length
            )
        else:
            changed = self._c.path_smoothing(fsv, o_path, farray(expand), break_length)
        return changed, [list(o) for o in o_path]

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
    def a(self) -> CPathNode:
        return CPathNode(self._c.a)

    @property
    def b(self) -> CPathNode:
        return CPathNode(self._c.b)


class CPathGraph:
    _c: object
    _cpgew: CPathGraphEdgesWrapper

    def __init__(self, _c: object = None) -> None:
        if _c is None:
            self._c = gbl.ctree.PathGraph()
        else:
            self._c = _c
        self._cpgew = CPathGraphEdgesWrapper(self)

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
    def edges(self) -> CPathGraphEdgesWrapper:
        return self._cpgew
