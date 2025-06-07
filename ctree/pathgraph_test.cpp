#include <stdio.h>
#include "ctree.hpp"

int main()
{
    auto tn = ctree::TreeNode::create()->as_root({0, 0}, {50, 50}, {1, 1}, 2);
    auto pg = ctree::PathGraph();
    tn->add_raycast({0, 0}, {25, 49});
    pg.update(tn);
    std::vector<ctree::TreeNode::Ptr> path;
    pg.get_path(tn->query({0, 0},true), tn->query({50, 50},true), path);
    std::vector<std::array<float, TREE_DIM>> c_path;
    pg.interpolation_center(path, c_path);
    for (auto &fa : c_path)
    {
        printf("[%f,%f] ",fa[0],fa[1]);
    }
}