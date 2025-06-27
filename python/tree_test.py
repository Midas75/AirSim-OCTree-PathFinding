from math import cos, sin, pi
import time

import random
from cv2 import imshow, waitKey  # pylint: disable=no-name-in-module
from tree import TreeNode, PathNode, PathGraph


class TreeTest:
    @staticmethod
    def i_root_test():
        tn = TreeNode()
        tn.as_root([0, 0, 0], [50, 50, 50], [5, 5, 5])
        print(f"i_center: {tn.i_center}")
        print(f"i_bound_size: {tn.i_bound_size}")
        print(f"i_bound_min: {tn.i_bound_min}")
        print(f"i_bound_max: {tn.i_bound_max}")
        print(f"center: {tn.center}")
        print(f"bound_size: {tn.bound_size}")
        print(f"bound_min: {tn.bound_min}")
        print(f"bound_max: {tn.bound_max}")

    @staticmethod
    def add_test():
        tn = TreeNode()
        tn.as_root([0, 0], [50, 50], [1, 1])
        tn.add([0, 0])
        tn.render2(show_now=0)

    @staticmethod
    def unbalance_test():

        tn = TreeNode()
        tn.as_root([0, 0], [50, 50], [4, 1])
        for _ in range(10):
            tn.add(
                [random.random() * tn.bound_size[0], random.random() * tn.bound_size[1]]
            )
        tn.render2(show_now=0)

    @staticmethod
    def raycast_test():
        tn = TreeNode()
        tn.as_root([0, 0], [640, 640], [2, 2])
        number = 500
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            start = time.perf_counter()
            tn.add_raycast([0, 0], p, False)
            print(f"add_raycast cost {1000*(time.perf_counter()-start)} ms")
            tn.render2(show_now=0)

    @staticmethod
    def path_node_id_test():
        tn = TreeNode()
        tn.as_root([0, 0], [32, 64], [1, 1])
        p1 = tn.add_i([16, 32])
        p2 = tn.add_i([0, 0])
        print(PathNode(p1).id, PathNode(p2).id)
        tn.render2(show_now=0)

    @staticmethod
    def gen_id_test():
        ids = set[int]()
        x, y, z = 10, 20, 30
        bound_size = [x, y, z]
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    i_center = [i, j, k]
                    result = 0
                    dim_range = 1
                    for dim in range(3):
                        result += dim_range * i_center[dim]
                        dim_range *= bound_size[dim]
                    if result in ids:
                        raise ValueError(result)
                    ids.add(id)
        print("passed")

    @staticmethod
    def neighbor_test():
        tn = TreeNode().as_root([0, 0], [1, 1], [0.1, 0.1])
        ns = tn.add_i([3, 3]).get_neighbor()
        print(tn.query_i([3, 3]).i_center)
        for n in ns:
            print(n.i_center)
            n.known = True
        tn.render2(show_now=0)

    @staticmethod
    def graph_test():
        tn = TreeNode().as_root([0, 0], [50, 50], [1, 1])
        tn.add_i([1, 1])
        tn.add_i([49, 49])
        tn.add_i([0, 20])
        tn.add_i([20, 10])
        pg = PathGraph()
        pg.update(tn, False)
        tn.render2(show_now=0, with_graph=pg)

    @staticmethod
    def graph_update_test():

        tn = TreeNode().as_root([0, 0], [50, 50], [0.25, 0.25])
        pg = PathGraph()
        for _i in range(10):
            for _j in range(1000):
                tn.add(
                    [
                        random.random() * tn.bound_size[0],
                        random.random() * tn.bound_size[1],
                    ]
                )
            start = time.perf_counter()
            pg.update(tn)
            print(f"update cost: {(time.perf_counter()-start)*1000:.2f} ms")
            tn.render2(show_now=0, with_graph=pg)

    @staticmethod
    def get_path_test():

        tn = TreeNode().as_root([0, 0], [50, 50], [0.5, 0.5])
        pg = PathGraph()
        for _i in range(10):
            for _j in range(100):
                tn.add(
                    [
                        1 + random.random() * (tn.bound_size[0] - 1),
                        1 + random.random() * (tn.bound_size[1] - 1),
                    ]
                )
            pg.update(tn)
            start = time.perf_counter()
            path = pg.get_path(tn.query([0, 0]), tn.query([50, 50]))
            print(f"get path cost: {(time.perf_counter()-start)*1000:.2f} ms")
            c_path = tn.interpolation_center(path)
            tn.render2(show_now=0, with_graph=pg, with_path=c_path)
            _, s_path = tn.path_smoothing(c_path, expand=tn.min_length)
            tn.render2(show_now=0, with_graph=pg, with_path=s_path)

    @staticmethod
    def render3_test():
        tn = TreeNode().as_root([0, 0, 0], [10, 10, 10], [1, 1, 1])
        tn.add([0, 0, 0])
        tn.add_i([3, 3, 3])
        tn.render3(show_now=0, with_path=[[0, 0, 0], [10, 10, 10]])

    @staticmethod
    def octree_test():
        tn = TreeNode().as_root([-100, -100, -100], [100, 100, 100], [1, 1, 1])
        pg = PathGraph()
        for _i in range(1000):
            tn.add(
                [
                    1 + random.random() * (tn.bound_size[0] - 1),
                    1 + random.random() * (tn.bound_size[1] - 1),
                    1 + random.random() * (tn.bound_size[1] - 1),
                ]
            )
        start = time.perf_counter()
        pg.update(tn)
        path = pg.get_path(tn.query([0, 0, 0]), tn.query([10, 10, 10]))
        r_path = [pn.tree_node.center for pn in path]
        c_path = tn.interpolation_center(path)
        _, s_path = tn.path_smoothing(c_path, expand=tn.min_length)
        print(f"update&path cost {1000*(time.perf_counter()-start):.2f} ms")
        print(r_path)
        print(c_path)
        print(s_path)
        tn.render3(show_now=0, with_path=r_path)

    @staticmethod
    def serialize_deserialize_test():
        tn = TreeNode().as_root([0, 0], [50, 50], [1, 1])
        number = 50
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
        start = time.perf_counter()
        obj = tn.serialize()
        print(f"serialization cost {1000*(time.perf_counter()-start)} ms")

        start = time.perf_counter()
        deserialized = TreeNode.deserialize(obj)
        print(f"deserialization cost {1000*(time.perf_counter()-start)} ms")
        imshow("Raw", tn.render2())
        imshow("New", deserialized.render2())
        waitKey(0)

    @staticmethod
    def save_load_test():
        tn = TreeNode().as_root([0, 0], [50, 50], [1, 1])
        number = 50
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
        start = time.perf_counter()
        tn.save()
        print(f"save cost {1000*(time.perf_counter()-start)} ms")

        start = time.perf_counter()
        loaded = TreeNode.load()
        print(f"load cost {1000*(time.perf_counter()-start)} ms")
        imshow("Raw", tn.render2())
        imshow("New", loaded.render2())
        waitKey(0)

    @staticmethod
    def dynamic_culling_test():
        tn = TreeNode().as_root([0, 0], [50, 50], [0.25, 0.25])
        number = 500
        moving_obstacle = [10, 0]
        obstacle_size = 2
        moving_speed = 0.5
        pg = PathGraph()
        for _ in range(number):
            print(_)
            moving_obstacle[1] += moving_speed
            if moving_obstacle[1] > 50:
                moving_obstacle[1] = 0
            for i in range(number):
                p = [
                    30,
                    tn.bound_size[1] * cos(pi / 2 * i / number),
                ]
                if (
                    p[1] > moving_obstacle[1] - obstacle_size / 2
                    and p[1] < moving_obstacle[1] + obstacle_size / 2
                ):
                    p[0] = moving_obstacle[0]
                tn.add_raycast([0, p[1]], p, False, 20)
            pg.update(tn)
            tn.render2(show_now=0, with_graph=pg)

    @staticmethod
    def raycast_benchmark_test():
        tn = TreeNode().as_root([0, 0], [640, 640], [2, 2])
        number = 500
        start = time.perf_counter()
        for i in range(number):
            p = [
                tn.bound_size[0] * sin(pi / 2 * i / number),
                tn.bound_size[1] * cos(pi / 2 * i / number),
            ]
            tn.add_raycast([0, 0], p, False)
        print(f"add_raycast {number} times cost {1000*(time.perf_counter()-start)} ms")


if __name__ == "__main__":
    tt = TreeTest
    # tt.i_root_test()
    # tt.add_test()
    # tt.unbalance_test()
    # tt.raycast_test()
    # tt.path_node_id_test()
    # tt.gen_id_test()
    # tt.neighbor_test()
    # tt.graph_test()
    # tt.graph_update_test()
    # tt.get_path_test()
    # tt.render3_test()
    # tt.octree_test()
    # tt.serialize_deserialize_test()
    # tt.save_load_test()
    tt.dynamic_culling_test()
    # tt.raycast_benchmark_test()
