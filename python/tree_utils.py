from multiprocessing import Process, Queue
from traceback import print_exc
from time import perf_counter, sleep
from open3d import visualization, geometry, utility

from tree import TreeNode, PathGraph
from tqdm import tqdm

from ctree import CTreeNode, CPathGraph


class SerializedData2:
    node: dict[int, list[float]] = None


class SerializedData3:
    node: dict[int, list[float]] = None  # [x1,y1,z1,xb,yb,zb]
    points: list[list[float]] = None  # [x,y,z]
    add_points: list[list[float]] = None  # [x,y,z]
    graph: list[list[float]] = None  # [x1,y1,z1,x2,y2,z2]
    path: list[list[float]] = None  # [x,y,z]
    bound: list[float] = None  # [x1,y1,z1,x2,y2,z2]
    stop: bool = False

    def __init__(self):
        pass

    def update(
        self,
        tree_node: TreeNode | CTreeNode = None,
        points: list[list[float]] = None,
        add_points: list[list[float]] = None,
        path_graph: PathGraph | CPathGraph = None,
        path: list[list[float]] = None,
        bound: bool = True,
        contact_center: bool = False,
        stop: bool = False,
    ) -> float:
        start = perf_counter()
        if bound and tree_node is not None:
            self.bound = []
            self.bound.extend(tree_node.bound_min)
            self.bound.extend(tree_node.bound_max)
        else:
            self.bound = None
        if tree_node is not None:
            self.node = {}
            for node_id, node in tree_node.nodes.items():
                if node.is_leaf and node.state == node.FULL:
                    l = []
                    l.extend(node.bound_min)
                    l.extend(node.bound_size)
                    self.node[node_id] = l
        else:
            self.node = {}

        if path_graph is not None:
            self.graph = []
            for _, edge in path_graph.edges.items():
                if contact_center:
                    c = edge.a.tree_node.contact_center(edge.b.tree_node.center)
                    ac = []
                    cb = []
                    ac.extend(edge.a.tree_node.center)
                    ac.extend(c)
                    cb.extend(c)
                    cb.extend(edge.b.tree_node.center)
                    self.graph.append(ac)
                    self.graph.append(cb)
                else:
                    ab = []
                    ab.extend(edge.a.tree_node.center)
                    ab.extend(edge.b.tree_node.center)
                    self.graph.append(ab)
        else:
            self.graph = None
        if path is not None:
            self.path = path
        else:
            self.path = None
        if points is not None:
            self.points = points.copy()
        else:
            self.points = None
        if add_points is not None:
            self.add_points = add_points.copy()
        else:
            self.add_points = None
        self.stop = stop
        return perf_counter() - start


class VisWindow2:
    _process: Process = None
    _update_queue: Queue = None
    buffer_size: int = 2
    sd: SerializedData2


class VisWindow3:
    _process: Process = None
    _update_queue: Queue = None
    _tree_node_geometries: dict[int, geometry.TriangleMesh] = None
    _pcd_geometry: geometry.PointCloud = None
    _template_box_geometry: geometry.TriangleMesh = None
    _path_graph_geometry: geometry.LineSet = None
    _path_geometry: geometry.LineSet = None
    _bound_geometry: geometry.AxisAlignedBoundingBox = None
    _visualizer: visualization.Visualizer = None
    buffer_size: int = 2
    running: bool = True
    sd: SerializedData3

    def __init__(self, buffer_size: int = 2):
        self.buffer_size = buffer_size
        self._update_queue = Queue()
        self._process = Process(target=self._run, daemon=True)
        self.running = True
        self.sd = SerializedData3()
        self._process.start()

    def stop(self):
        self.update(stop=True)
        self._process.join(5)

    def update(
        self,
        tree_node: TreeNode = None,
        points: list[list[float]] = None,
        add_points: list[list[float]] = None,
        path_graph: PathGraph = None,
        path: list[list[float]] = None,
        bound: bool = True,
        contact_center: bool = False,
        stop: bool = False,
    ) -> float:
        self.clip_queue()
        cost = self.sd.update(
            tree_node=tree_node,
            points=points,
            add_points=add_points,
            path_graph=path_graph,
            path=path,
            bound=bound,
            contact_center=contact_center,
            stop=stop,
        )
        self._update_queue.put(self.sd)
        return cost

    def clip_queue(self):
        while (
            not self._update_queue.empty()
        ) and self._update_queue.qsize() > self.buffer_size:
            self._update_queue.get()

    def _run(self):
        self._visualizer = visualization.Visualizer()
        self._tree_node_geometries = dict[int, geometry.TriangleMesh]()
        merge_node = geometry.TriangleMesh()
        self._pcd_geometry = geometry.PointCloud()
        self._path_graph_geometry = geometry.LineSet()
        self._path_geometry = geometry.LineSet()
        self._bound_geometry = geometry.AxisAlignedBoundingBox()
        self._visualizer.create_window()
        self._visualizer.add_geometry(merge_node)
        self._visualizer.add_geometry(self._pcd_geometry)
        first_vis = True
        while self.running:
            try:
                if not self._update_queue.empty():
                    sd: SerializedData3 = self._update_queue.get()
                    if sd.stop:
                        self.running = False
                    need_remerge = False
                    for node_id in list(self._tree_node_geometries.keys()):
                        if node_id not in sd.node:
                            # tm = self._tree_node_geometries[node_id]
                            # self._visualizer.remove_geometry(tm, False)
                            need_remerge = True
                            del self._tree_node_geometries[node_id]
                    if need_remerge:
                        merge_node.clear()
                    for node_id, node_bound in (
                        tqdm(sd.node.items(), desc="merging")
                        if first_vis
                        else sd.node.items()
                    ):
                        if node_id not in self._tree_node_geometries:
                            box = geometry.TriangleMesh.create_box(
                                node_bound[3], node_bound[4], node_bound[5]
                            ).translate(node_bound[:3])
                            self._tree_node_geometries[node_id] = box
                            if not need_remerge:
                                merge_node += box
                        if need_remerge:
                            merge_node += self._tree_node_geometries[node_id]
                        if not need_remerge:
                            merge_node += box
                    merge_node.compute_vertex_normals()
                    self._visualizer.update_geometry(merge_node)
                    if sd.points is not None:
                        self._pcd_geometry.points = utility.Vector3dVector(sd.points)
                    if sd.add_points is not None:
                        self._pcd_geometry.points.extend(
                            utility.Vector3dVector(sd.add_points)
                        )
                    self._visualizer.update_geometry(self._pcd_geometry)
                    if sd.graph is not None and len(sd.graph) > 0:
                        points = list[list[float]]()
                        lines = list[list[int]]()

                        for i, p in (
                            tqdm(enumerate(sd.graph), desc="path")
                            if first_vis
                            else enumerate(sd.graph)
                        ):
                            points.append(p[:3])
                            points.append(p[3:])
                            lines.append([i * 2, i * 2 + 1])
                        self._path_graph_geometry.points = utility.Vector3dVector(
                            points
                        )
                        self._path_graph_geometry.lines = utility.Vector2iVector(lines)
                        self._path_graph_geometry.paint_uniform_color([0, 1, 0])
                        self._visualizer.add_geometry(
                            self._path_graph_geometry, first_vis
                        )
                    else:
                        self._visualizer.remove_geometry(
                            self._path_graph_geometry, False
                        )
                    if sd.path is not None and len(sd.path) > 0:
                        self._path_geometry.points = utility.Vector3dVector(sd.path)
                        self._path_geometry.lines = utility.Vector2iVector(
                            [[i, i + 1] for i in range(len(sd.path) - 1)]
                        )
                        self._path_geometry.paint_uniform_color([1, 0, 0])
                        self._visualizer.add_geometry(self._path_geometry, first_vis)
                    else:
                        self._visualizer.remove_geometry(self._path_geometry, False)
                    if sd.bound is not None:
                        self._bound_geometry.min_bound = sd.bound[:3]
                        self._bound_geometry.max_bound = sd.bound[3:]
                        self._bound_geometry.color = [0, 0, 1]
                        self._visualizer.add_geometry(self._bound_geometry, first_vis)
                    else:
                        self._visualizer.remove_geometry(self._bound_geometry, False)
                    # if first_vis:
                    #     self._visualizer.reset_view_point(True)
                    first_vis = False
                sleep(1 / 60)
                self._visualizer.poll_events()
                self._visualizer.update_renderer()
            except KeyboardInterrupt as _:
                self.running = False
                print_exc()
        # self._visualizer.destroy_window()
