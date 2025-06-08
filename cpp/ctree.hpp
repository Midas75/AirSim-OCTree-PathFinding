#ifndef C_TREE
#define C_TREE
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <memory>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <set>
namespace ctree
{
#ifndef TREE_DIM
#define TREE_DIM 3
#endif
#ifndef for_dims
#define for_dims(_this) for (int8_t dim = 0; dim < (_this)->dims; dim++)

    constexpr int8_t TREE_CHILDS = 1 << TREE_DIM;
    constexpr std::array<float, TREE_DIM> ZEROS = {0};
    struct TreeData
    {
        float min_length[TREE_DIM] = {0}, bound_min[TREE_DIM] = {0}, bound_max[TREE_DIM] = {0};
        int8_t dims = TREE_DIM;
    };
    struct TreeNodeData
    {
        long id = 0;
        long child[TREE_CHILDS] = {0};
        int i_bound_min[TREE_DIM] = {0}, i_bound_max[TREE_DIM] = {0};
        int8_t state = 0;
        bool known = false, is_leaf = false;
    };
    class TreeNode : public std::enable_shared_from_this<TreeNode>
    {
    private:
        explicit TreeNode() {}
        TreeNode(const TreeNode &) = delete;
        TreeNode &operator=(const TreeNode &) = delete;

    public:
        using Ptr = std::shared_ptr<TreeNode>;
        using ConstPtr = const Ptr;
        using PtrRef = Ptr &;
        using ConstPtrRef = ConstPtr &;
        inline static ConstPtrRef Nullptr = nullptr;
        inline static const float INF = std::numeric_limits<float>::infinity();
        static const int8_t EMPTY = 0;
        static const int8_t FULL = 1;
        static const int8_t HALF_FULL = 2;

        Ptr child[TREE_CHILDS];
        bool no_child = true;

        int8_t state = TreeNode::EMPTY;
        int dynamic_culling = -1;

        std::array<float, TREE_DIM> bound_min = {0},
                                    bound_max = {0},
                                    bound_size = {0},
                                    center = {0},
                                    min_length = {0};
        TreeNode *parent = nullptr, *root = nullptr;
        int8_t dims = TREE_DIM;
        int8_t directions = TREE_CHILDS;

        bool _min = false, is_leaf = true, known = false;
        std::array<int, TREE_DIM> i_bound_min = {0},
                                  i_bound_max = {0},
                                  i_bound_size = {0},
                                  i_center = {0};
        long id = 0;
        void serialize(TreeData &out_tree_data, std::vector<TreeNodeData> &out_tree_node_data) const
        {
            out_tree_data.dims = this->dims;
            std::copy(this->min_length.begin(), this->min_length.end(), out_tree_data.min_length);
            std::copy(this->bound_min.begin(), this->bound_min.end(), out_tree_data.bound_min);
            std::copy(this->bound_max.begin(), this->bound_max.end(), out_tree_data.bound_max);
            this->serialize(out_tree_node_data);
        }
        void serialize(std::vector<TreeNodeData> &out_tree_node_data) const
        {
            auto otnd = TreeNodeData();
            std::copy(this->i_bound_min.begin(), this->i_bound_min.end(), otnd.i_bound_min);
            std::copy(this->i_bound_max.begin(), this->i_bound_max.end(), otnd.i_bound_max);
            otnd.id = this->id;
            otnd.is_leaf = this->is_leaf;
            otnd.state = this->state;
            otnd.known = this->known;
            for (int i = 0; i < TREE_CHILDS; i++)
            {
                auto &c = this->child[i];
                if (c != Nullptr)
                {
                    otnd.child[i] = c->id;
                    c->serialize(out_tree_node_data);
                }
            }
            out_tree_node_data.emplace_back(std::move(otnd));
        }
        static ConstPtr deserialize(const TreeData &tree_data, const std::vector<TreeNodeData> &tree_node_datas)
        {
            auto map = std::unordered_map<long, size_t>();
            map.reserve(tree_node_datas.size());
            for (size_t i = 0; i < tree_node_datas.size(); i++)
            {
                map.emplace(tree_node_datas[i].id, i);
            }
            return deserialize(tree_data, tree_node_datas, map);
        }
        static ConstPtr deserialize(const TreeData &tree_data, const std::vector<TreeNodeData> &tree_node_datas,
                                    const std::unordered_map<long, size_t> &_tree_nodes_map,
                                    long _current_id = 0, ConstPtrRef _parent = Nullptr)
        {
            Ptr node;
            bool is_root = false;
            if (_current_id == 0)
            {
                std::array<float, 3> abmin = {0}, abmax = {0}, aml = {0};
                for (int8_t dim = 0; dim < tree_data.dims; dim++)
                {
                    abmin[dim] = tree_data.bound_min[dim];
                    abmax[dim] = tree_data.bound_max[dim];
                    aml[dim] = tree_data.min_length[dim];
                }
                node = TreeNode::create()->as_root(abmin, abmax, aml, tree_data.dims);
                is_root = true;
                _current_id = node->id;
            }
            const TreeNodeData &info = tree_node_datas[_tree_nodes_map.at(_current_id)];
            if (!is_root)
            {
                node = TreeNode::create(_parent.get());
                for (int8_t dim = 0; dim < tree_data.dims; dim++)
                {
                    node->i_bound_max[dim] = info.i_bound_max[dim];
                    node->i_bound_min[dim] = info.i_bound_min[dim];
                }
                node->update_bound();
            }
            node->state = info.state;
            node->known = info.known;
            node->is_leaf = info.is_leaf;
            for (int8_t direction = 0; direction < TREE_CHILDS; direction++)
            {
                if (info.child[direction] > 0)
                {
                    node->no_child = false;
                    node->child[direction] = TreeNode::deserialize(
                        tree_data, tree_node_datas,
                        _tree_nodes_map,
                        info.child[direction], node);
                }
            }
            return node;
        }
        static ConstPtr create(TreeNode *parent = nullptr,
                               const int8_t direction = -1,
                               const int8_t divide = 0)
        {
            auto self = std::shared_ptr<TreeNode>(new TreeNode());
            self->state = TreeNode::EMPTY;
            for (int i = 0; i < TREE_CHILDS; i++)
            {
                self->child[i] = nullptr;
            }
            if (parent != nullptr)
            {
                self->parent = parent;
                self->root = parent->root;

                self->dims = parent->dims;
                self->directions = 1 << self->dims;
                self->min_length = parent->min_length;
                if (direction >= 0)
                {
                    int8_t _divide = divide, _direction = direction;
                    for_dims(self)
                    {
                        _divide = divide >> dim;
                        _direction = direction >> dim;
                        if (!(_divide & 1))
                        {
                            self->i_bound_min[dim] = parent->i_bound_min[dim];
                            self->i_bound_max[dim] = parent->i_bound_max[dim];
                        }
                        else if (!(_direction & 1))
                        {
                            self->i_bound_min[dim] = parent->i_bound_min[dim];
                            self->i_bound_max[dim] = parent->i_center[dim];
                        }
                        else
                        {
                            self->i_bound_min[dim] = parent->i_center[dim];
                            self->i_bound_max[dim] = parent->i_bound_max[dim];
                        }
                    }
                    self->update_bound();
                }
            }
            return self;
        }
        void update_bound()
        {
            this->center.fill(0);
            this->i_center.fill(0);
            if (this->parent != nullptr)
            {
                this->bound_max.fill(0);
                this->bound_min.fill(0);
                this->bound_size.fill(0);
                this->i_bound_size.fill(0);
            }
            for_dims(this)
            {
                this->i_bound_size[dim] = this->i_bound_max[dim] - this->i_bound_min[dim];
                if (this->parent != nullptr)
                {
                    float bound_ratio = this->root->bound_size[dim] / this->root->i_bound_size[dim];
                    this->bound_max[dim] = this->i_bound_max[dim] * bound_ratio + this->root->bound_min[dim];
                    this->bound_min[dim] = this->i_bound_min[dim] * bound_ratio + this->root->bound_min[dim];
                }
                this->bound_size[dim] = this->bound_max[dim] - this->bound_min[dim];
                this->center[dim] = this->bound_max[dim] / 2 + this->bound_min[dim] / 2;
                this->i_center[dim] = this->i_bound_max[dim] / 2 + this->i_bound_min[dim] / 2;
            }
            this->_min = this->is_min();
            this->id = this->gen_id();
        }
        bool is_min() const
        {
            for_dims(this)
            {
                if (!(this->i_center[dim] & 1))
                {
                    return false;
                }
            }
            return true;
        }
        long gen_id() const
        {
            long result = 0;
            long dim_range = 1;
            for_dims(this)
            {
                result += dim_range * this->i_center[dim];
                dim_range *= this->root->i_bound_size[dim];
            }
            return result;
        }
        ConstPtr as_root(
            const std::array<float, TREE_DIM> &bound_min,
            const std::array<float, TREE_DIM> &bound_max,
            const std::array<float, TREE_DIM> &min_length,
            const int8_t dims = TREE_DIM)
        {
            this->dims = dims;
            this->directions = 1 << this->dims;
            this->root = this;
            this->parent = nullptr;

            this->min_length = min_length;
            this->bound_max = bound_max;
            this->bound_min = bound_min;

            for_dims(this)
            {
                this->bound_size[dim] = this->bound_max[dim] - this->bound_min[dim];
                this->center[dim] = this->bound_max[dim] / 2 + this->bound_min[dim] / 2;
                float dim_ratio = this->bound_size[dim] / this->min_length[dim];
                if (dim_ratio < 1)
                {
                    dim_ratio = 1;
                }
                this->i_bound_size[dim] = 1 << (int)std::ceil(std::log2(dim_ratio));
                this->i_bound_max[dim] = this->i_bound_size[dim];
                this->i_bound_min[dim] = 0;
                this->i_center[dim] = this->i_bound_max[dim] / 2;
            }
            this->_min = this->is_min();
            this->id = this->gen_id();
            return shared_from_this();
        }
        void divide(const int depth = 1)
        {
            if (this->_min)
            {
                return;
            }
            if (this->state != TreeNode::EMPTY)
            {
                return;
            }
            if (!this->is_leaf)
            {
                return;
            }
            if (depth <= 0)
            {
                return;
            }
            this->is_leaf = false;
            for (int8_t i = 0; i < this->directions; i++)
            {
                auto reduced = this->get_bound_by_direction(i);
                auto ri = reduced.first, d = reduced.second;
                if (this->child[ri] == nullptr)
                {
                    this->no_child = false;
                    auto c = TreeNode::create(this, i, d);
                    this->child[ri] = c;
                    this->child[i] = c;
                    this->divide(depth - 1);
                }
                else
                {
                    this->child[i] = this->child[ri];
                }
            }
        }
        void update_state()
        {
            if (this->no_child)
            {
                this->is_leaf = true;
                return;
            }
            int8_t full_counter = 0, empty_counter = 0, half_full_counter = 0;
            for (int i = 0; i < this->directions; i++)
            {
                auto c = this->child[i];
                if (c == nullptr)
                {
                    continue;
                }
                switch (c->state)
                {
                case TreeNode::FULL:
                    full_counter += 1;
                    break;
                case TreeNode::HALF_FULL:
                    half_full_counter += 1;
                    break;
                case TreeNode::EMPTY:
                    empty_counter += 1;
                    break;
                }
            }
            if (empty_counter == 0 && half_full_counter == 0)
            {
                this->state = TreeNode::FULL;
                this->is_leaf = true;
                for (int i = 0; i < this->directions; i++)
                {
                    this->child[i] = nullptr;
                }
            }
            else if (full_counter == 0 && half_full_counter == 0)
            {
                this->state = TreeNode::EMPTY;
            }
            else
            {
                this->state = TreeNode::HALF_FULL;
            }
        }
        std::pair<int8_t, int8_t> get_bound_by_direction(const int8_t direction) const
        {
            int8_t index = direction;
            int8_t divide = 0xFFu;
            for_dims(this)
            {
                int8_t bit = 1u << dim;
                if (this->i_center[dim] & 1)
                {
                    divide &= ~bit;
                    index &= ~bit;
                }
            }
            return {index, divide};
        }
        int8_t get_direction(const std::array<float, TREE_DIM> &point, const bool allow_oor = false) const
        {
            if ((!allow_oor) && (this->out_of_region(point)))
            {
                return -1;
            }
            int8_t result = 0;
            for_dims(this)
            {
                if (point[dim] > this->center[dim])
                {
                    result |= 1 << dim;
                }
            }
            return result;
        }
        bool out_of_region(const std::array<float, TREE_DIM> &point) const
        {
            for_dims(this)
            {
                if (this->bound_min[dim] > point[dim] ||
                    this->bound_max[dim] < point[dim])
                {
                    return true;
                }
            }
            return false;
        }
        int8_t get_direction_i(const std::array<int, TREE_DIM> &point, const bool allow_oor = false) const
        {
            if ((!allow_oor) && (this->out_of_region_i(point)))
            {
                return -1;
            }
            int8_t result = 0;
            for_dims(this)
            {
                if (point[dim] > this->i_center[dim])
                {
                    result |= 1 << dim;
                }
            }
            return result;
        }
        bool out_of_region_i(const std::array<int, TREE_DIM> &point) const
        {
            for_dims(this)
            {
                if (this->i_bound_min[dim] > point[dim] ||
                    this->i_bound_max[dim] < point[dim])
                {
                    return true;
                }
            }
            return false;
        }
        ConstPtr query(
            const std::array<float, TREE_DIM> &point,
            bool nearest_on_oor = false)
        {
            if ((!nearest_on_oor) && this->out_of_region(point))
            {
                return TreeNode::Nullptr;
            }
            if (this->is_leaf)
            {
            }
            else
            {
                int8_t direction = this->get_direction(point, true);
                if (this->child[direction] != nullptr)
                {
                    return this->child[direction]->query(point, nearest_on_oor);
                }
            }
            return shared_from_this();
        }
        ConstPtr query_i(
            const std::array<int, TREE_DIM> &point,
            bool nearest_on_oor = false)
        {
            if ((!nearest_on_oor) && this->out_of_region_i(point))
            {
                return TreeNode::Nullptr;
            }
            if (this->is_leaf)
            {
            }
            else
            {
                int8_t direction = this->get_direction_i(point, true);
                if (this->child[direction] != nullptr)
                {
                    return this->child[direction]->query_i(point, nearest_on_oor);
                }
            }
            return shared_from_this();
        }
        void clear_as(const int8_t state = TreeNode::EMPTY)
        {
            this->no_child = true;
            for (int i = 0; i < TREE_CHILDS; i++)
            {
                this->child[i] = nullptr;
            }
            this->state = state;
            this->known = false;
            this->is_leaf = true;
            auto parent = this->parent;
            while (parent != nullptr)
            {
                parent->update_state();
                parent = parent->parent;
            }
        }
        static const std::shared_ptr<const TreeNode> lca(ConstPtrRef node1, ConstPtrRef node2)
        {
            auto p1 = node1.get();
            auto p2 = node2.get();
            if (!p1 || !p2)
                return nullptr;
            while (p1 != p2)
            {
                p1 = (p1 == nullptr) ? node2.get() : p1->parent;
                p2 = (p2 == nullptr) ? node1.get() : p2->parent;
            }
            if (p1 == nullptr)
            {
                return node1->root->shared_from_this();
            }
            else
            {
                return p1->shared_from_this();
            }
        }
        bool cross_self(const std::array<float, TREE_DIM> &start,
                        const std::array<float, TREE_DIM> &inv_vector,
                        const std::array<float, TREE_DIM> &expand) const
        {
            float tmin = -TreeNode::INF;
            float tmax = TreeNode::INF;

            for_dims(this)
            {
                float b_min = this->bound_min[dim] - expand[dim];
                float b_max = this->bound_max[dim] + expand[dim];
                if (inv_vector[dim] == TreeNode::INF)
                    if (start[dim] < b_min || start[dim] > b_max)
                    {
                        return false;
                    }
                    else
                    {
                        float t1 = (b_min - start[dim]) * inv_vector[dim];
                        float t2 = (b_max - start[dim]) * inv_vector[dim];
                        tmin = std::max(tmin, std::min(t1, t2));
                        tmax = std::min(tmax, std::max(t1, t2));
                    }
                if (tmin > tmax)
                {
                    return false;
                }
            }
            return tmax >= 0 && tmin <= 1;
        }
        bool cross(const std::array<float, TREE_DIM> &start,
                   const std::array<float, TREE_DIM> &inv_vector,
                   const std::array<float, TREE_DIM> &expand) const
        {
            if (this->state == TreeNode::EMPTY)
            {
                return false;
            }
            bool self_cross = this->cross_self(start, inv_vector, expand);
            if (this->state == TreeNode::FULL)
            {
                return self_cross;
            }
            if (self_cross && (!this->no_child))
            {
                for (int i = 0; i < TREE_CHILDS; i++)
                {
                    if (this->child[i] != nullptr)
                    {
                        if (this->child[i]->cross(start, inv_vector, expand))
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }
        bool cross_lca(
            const std::array<float, TREE_DIM> &start,
            const std::array<float, TREE_DIM> &end,
            const std::array<float, TREE_DIM> &expand = ZEROS)
        {
            auto &n1 = this->query(start, true);
            auto &n2 = this->query(end, true);
            std::array<float, TREE_DIM> inv_vector = {0};
            for_dims(this)
            {
                float v = end[dim] - start[dim];
                if (v == 0)
                {
                    inv_vector[dim] = TreeNode::INF;
                }
                else
                {
                    inv_vector[dim] = 1 / v;
                }
            }
            return TreeNode::lca(n1, n2)->cross(start, inv_vector, expand);
        }
        static const std::shared_ptr<const TreeNode> get_parent(ConstPtrRef self, int number = 1)
        {
            auto parent = self.get();
            for (int i = 0; i < number; i++)
            {
                if (parent->parent != nullptr)
                {
                    parent = parent->parent;
                }
                else
                {
                    break;
                }
            }
            return parent->shared_from_this();
        }
        bool intersect(ConstPtrRef other) const
        {
            bool one_eq = false;
            for_dims(this)
            {
                if (this->i_bound_min[dim] > other->i_bound_max[dim] ||
                    this->i_bound_max[dim] < other->i_bound_min[dim])
                {
                    return false;
                }
                if (this->i_bound_min[dim] == other->i_bound_max[dim] ||
                    this->i_bound_max[dim] == other->i_bound_min[dim])
                {
                    if (one_eq)
                    {
                        return false;
                    }
                    one_eq = true;
                }
            }
            return true;
        }
        bool add(const std::array<float, TREE_DIM> &point, bool empty = false)
        {
            if (this->state == TreeNode::FULL)
            {
                if (!empty)
                {
                    return false;
                }
                else
                {
                    return false;
                }
            }
            auto direction = this->get_direction(point);
            if (direction < 0)
            {
                return false;
            }
            if (this->state == TreeNode::EMPTY)
            {
                if (empty)
                {
                    if (!this->known)
                    {
                        this->known = true;
                        return false;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    if (this->_min)
                    {
                        this->state = TreeNode::FULL;
                        this->dynamic_culling = -1;
                        return true;
                    }
                    else
                    {
                    }
                }
            }
            this->divide();
            bool changed = this->child[direction]->add(point, empty);
            if (changed)
            {
                this->update_state();
            }
            return changed;
        }
        bool add_i(const std::array<int, TREE_DIM> &point, bool empty = false)
        {
            if (this->state == TreeNode::FULL)
            {
                if (!empty)
                {
                    return false;
                }
                else
                {
                    return false;
                }
            }
            auto direction = this->get_direction_i(point);
            if (direction < 0)
            {
                return false;
            }
            if (this->state == TreeNode::EMPTY)
            {
                if (empty)
                {
                    if (!this->known)
                    {
                        this->known = true;
                        return false;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    if (this->_min)
                    {
                        this->state = TreeNode::FULL;
                        this->dynamic_culling = -1;
                        return true;
                    }
                    else
                    {
                    }
                }
            }
            this->divide();
            bool changed = this->child[direction]->add_i(point, empty);
            if (changed)
            {
                this->update_state();
            }
            return changed;
        }
        int8_t ray_out_intersect(
            const std::array<float, TREE_DIM> &point,
            const std::array<float, TREE_DIM> &vector,
            std::array<float, TREE_DIM> &out) const
        {
            float t_min = -TreeNode::INF;
            float t_max = TreeNode::INF;
            int8_t out_dim = 0;
            for_dims(this)
            {
                float t1, t2;
                if (vector[dim] == 0)
                {
                    t1 = TreeNode::INF;
                    t2 = TreeNode::INF;
                }
                else
                {
                    t1 = (this->bound_min[dim] - point[dim]) / vector[dim];
                    t2 = (this->bound_max[dim] - point[dim]) / vector[dim];
                }
                float t_near = std::min(t1, t2);
                float t_far = std::max(t1, t2);
                if (t_max > t_far)
                {
                    out_dim = dim;
                    t_max = t_far;
                }
                t_min = std::max(t_min, t_near);
            }
            float exit_t = t_max;
            for_dims(this)
            {
                out[dim] = point[dim] + exit_t * vector[dim];
            }
            return out_dim;
        }
        void add_raycast(const std::array<float, TREE_DIM> &start,
                         const std::array<float, TREE_DIM> &point,
                         bool empty_end = false,
                         int dynamic_culling = 20,
                         float culling_min_ratio = 0.2,
                         float culling_max_ratio = 0.8)
        {
            this->add(point, empty_end);
            auto end = this->query(point);
            std::array<float, TREE_DIM> current = start;
            std::array<float, TREE_DIM> direction = {0};
            std::array<int8_t, TREE_DIM> sign = {0};
            for_dims(this)
            {
                direction[dim] = point[dim] - start[dim];
                if (direction[dim] > 0)
                {
                    sign[dim] = 1;
                }
                else if (direction[dim] < 0)
                {
                    sign[dim] = -1;
                }
            }
            bool quit = false;
            std::unordered_set<long> visit;
            while (true)
            {
                auto cnode = this->query(current);
                if (cnode == TreeNode::Nullptr ||
                    cnode == end ||
                    visit.count(cnode->id))
                {
                    break;
                }
                bool need_culling = false;
                for_dims(this)
                {
                    float cd = abs(current[dim] - start[dim]);
                    float ad = abs(direction[dim]);
                    if (cd > ad)
                    {
                        quit = true;
                        break;
                    }
                    if (cd > culling_min_ratio * ad && cd < culling_max_ratio * ad)
                    {
                        need_culling = true;
                    }
                }
                if (quit)
                {
                    break;
                }
                if (cnode->state != TreeNode::EMPTY && need_culling && dynamic_culling > 0)
                {
                    if (cnode->dynamic_culling < 0)
                    {
                        cnode->dynamic_culling = dynamic_culling;
                    }
                    else
                    {
                        cnode->dynamic_culling -= 1;
                    }
                    if (cnode->dynamic_culling == 0)
                    {
                        cnode->clear_as();
                        cnode->dynamic_culling = -1;
                    }
                }
                visit.emplace(cnode->id);
                this->add(current, true);
                int8_t out_dim = cnode->ray_out_intersect(start, direction, current);
                current[out_dim] += this->min_length[out_dim] * sign[out_dim];
            }
        }
        void get_neighbor(std::unordered_map<long, ConstPtr> &out_map) const
        {
            std::array<int, TREE_DIM> lower = {0}, upper = {0};
            for_dims(this)
            {
                for (int _dim = 0; _dim < this->dims; _dim++)
                {
                    if (dim == _dim)
                    {
                        lower[_dim] = this->i_bound_min[_dim] - 1;
                        upper[_dim] = this->i_bound_max[_dim] + 1;
                    }
                    else
                    {
                        upper[_dim] = this->i_center[_dim];
                        lower[_dim] = this->i_center[_dim];
                    }
                }
                auto lower_node = this->root->query_i(lower);
                auto upper_node = this->root->query_i(upper);
                if (lower_node != nullptr)
                {
                    out_map.emplace(lower_node->id, lower_node);
                }
                if (upper_node != nullptr)
                {
                    out_map.emplace(upper_node->id, upper_node);
                }
            }
            return;
        }
        uint8_t contact_with(ConstPtrRef other) const
        {
            uint8_t result = 0;
            for_dims(this)
            {
                auto size = (this->i_bound_size[dim] + other->i_bound_size[dim]) / 2;
                auto c2c = abs(this->i_center[dim] - other->i_center[dim]);
                if (size < c2c)
                {
                    return result;
                }
                result |= 1 << (dim + 1);
            }
            result |= 1;
            return result;
        }
        bool contact_center(ConstPtrRef other, std::array<float, TREE_DIM> &out_center) const
        {
            auto cw = this->contact_with(other);
            if (!(cw & 1))
            {
                return false;
            }
            for_dims(this)
            {
                if (cw & (1 << (dim + 1)))
                {
                    if (this->i_bound_size[dim] < other->i_bound_size[dim])
                    {
                        out_center[dim] = this->center[dim];
                    }
                    else
                    {
                        out_center[dim] = other->center[dim];
                    }
                }
                else
                {
                    if (this->i_center[dim] < other->i_center[dim])
                    {
                        out_center[dim] = this->center[dim] + this->bound_size[dim] / 2;
                    }
                    else
                    {
                        out_center[dim] = this->center[dim] - this->bound_size[dim] / 2;
                    }
                }
            }
            return true;
        }
        bool path_smoothing(
            const std::vector<std::array<float, TREE_DIM>> &path,
            std::vector<std::array<float, TREE_DIM>> &out_path,
            const std::array<float, TREE_DIM> &expand = ZEROS)
        {
            bool changed = false;
            if (path.size() <= 2)
            {
                out_path = path;
                return changed;
            }
            out_path.clear();
            out_path.emplace_back(path[0]);
            size_t size_1 = path.size() - 1;
            size_t i = 0;
            while (i < size_1)
            {
                size_t j = size_1;
                while (j > i + 1)
                {
                    if (!this->cross_lca(path[i], path[j], expand))
                    {
                        changed = true;
                        break;
                    }
                    j -= 1;
                }
                out_path.emplace_back(path[j]);
                i = j;
            }
            return changed;
        }
    };
    struct pair_hash_ll
    {
        std::size_t operator()(const std::pair<long, long> &p) const
        {
            std::size_t h1 = std::hash<long>{}(p.first);
            std::size_t h2 = std::hash<long>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    class PathNode
    {
    public:
        const long id;
        const int dims;
        std::unordered_set<std::pair<long, long>, pair_hash_ll> edges;
        TreeNode::ConstPtrRef tree_node;
        float f = 0, g = 0, h = 0;
        long from_node = 0;
        PathNode(TreeNode::ConstPtrRef tree_node) : tree_node(tree_node), id(tree_node->id), dims(tree_node->dims)
        {
        }
        float distance(std::shared_ptr<PathNode> &other, bool unknown_penalty = true) const
        {
            float up_factor = 0.2f;
            if (unknown_penalty)
            {
                if ((!this->tree_node->known) || (!other->tree_node->known))
                {
                    up_factor = 1;
                }
            }
            float v = 0;
            for_dims(this)
            {
                v += std::powf(this->tree_node->center[dim] - other->tree_node->center[dim], 2);
            }
            return std::sqrt(v) * up_factor;
        }
    };
    struct compare_f
    {
        bool operator()(const std::shared_ptr<PathNode> &a, const std::shared_ptr<PathNode> &b) const
        {
            return a->f < b->f;
        }
    };
    class PathEdge
    {
    public:
        const std::shared_ptr<PathNode> a, b;
        std::pair<long, long> id;
        PathEdge(std::shared_ptr<PathNode> a, std::shared_ptr<PathNode> b) : a(a), b(b)
        {
            if (a->id < b->id)
            {
                this->id = {a->id, b->id};
            }
            else
            {
                this->id = {b->id, a->id};
            }
        }
    };
    class PathGraph
    {
    public:
        std::unordered_map<long, std::shared_ptr<PathNode>> nodes;
        std::unordered_map<std::pair<long, long>, std::shared_ptr<PathEdge>, pair_hash_ll> edges;
        TreeNode::Ptr last_root = nullptr;
        std::unordered_map<long, TreeNode::Ptr> now_leaves, last_leaves;
        void add_node(TreeNode::ConstPtrRef tree_node)
        {
            if (!this->nodes.count(tree_node->id))
            {
                this->nodes.emplace(tree_node->id, std::make_shared<PathNode>(tree_node));
            }
        }
        std::shared_ptr<PathNode> find_node(TreeNode::ConstPtrRef tree_node)
        {
            if (this->nodes.count(tree_node->id))
            {
                return this->nodes.at(tree_node->id);
            }
            return nullptr;
        }
        void add_edge(TreeNode::ConstPtrRef a, TreeNode::ConstPtrRef b)
        {
            auto na = this->find_node(a);
            auto nb = this->find_node(b);
            if (na == nullptr || nb == nullptr || na->id == nb->id)
            {
                return;
            }
            auto edge = std::make_shared<PathEdge>(na, nb);
            this->edges.emplace(edge->id, edge);
        }
        void get_empty_leaves(TreeNode::ConstPtrRef tree_node, std::unordered_map<long, TreeNode::Ptr> &leaves)
        {
            if (tree_node->is_leaf && tree_node->state == TreeNode::EMPTY)
            {
                leaves.emplace(tree_node->id, tree_node);
                this->add_node(tree_node);
                return;
            }
            if (!tree_node->no_child)
            {
                for (int i = 0; i < TREE_CHILDS; i++)
                {
                    if (tree_node->child[i] != nullptr)
                    {
                        this->get_empty_leaves(tree_node->child[i], leaves);
                    }
                }
            }
            return;
        }
        void get_edges(std::unordered_map<long, TreeNode::Ptr> &leaves)
        {
            for (auto pair1 : leaves)
            {
                for (auto pair2 : leaves)
                {
                    if (pair1.first <= pair2.first)
                    {
                        continue;
                    }
                    if (pair1.second->intersect(pair2.second))
                    {
                        this->add_edge(pair1.second, pair2.second);
                    }
                }
            }
        }
        void get_edges_neighbor(std::unordered_map<long, TreeNode::Ptr> &leaves)
        {
            std::unordered_set<long> active_nodes;
            for (auto it = this->edges.begin(); it != this->edges.end();)
            {
                auto &edge = it->second;
                auto &tna = edge->a->tree_node;
                auto &tnb = edge->b->tree_node;

                bool a_expire = tna->state != TreeNode::EMPTY || (!tna->is_leaf);
                bool b_expire = tnb->state != TreeNode::EMPTY || (!tnb->is_leaf);

                if (a_expire || b_expire)
                {
                    if (a_expire && !b_expire)
                    {
                        edge->b->edges.erase(it->first);
                        active_nodes.emplace(tnb->id);
                    }
                    if (b_expire && !a_expire)
                    {
                        edge->a->edges.erase(it->first);
                        active_nodes.emplace(tna->id);
                    }

                    it = this->edges.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            std::unordered_map<long, TreeNode::ConstPtr> neighbors;
            for (auto &kv : leaves)
            {
                neighbors.clear();
                if ((!this->last_leaves.count(kv.first)) || active_nodes.count(kv.first))
                {
                    kv.second->get_neighbor(neighbors);
                    for (auto &nkv : neighbors)
                    {
                        if (leaves.count(nkv.first))
                        {
                            this->add_edge(kv.second, nkv.second);
                        }
                    }
                }
            }
        }
        void update(TreeNode::ConstPtrRef root, bool full_reset = false)
        {
            if (full_reset)
            {
                this->last_leaves.clear();
                this->nodes.clear();
                this->edges.clear();
                this->now_leaves.clear();
                this->get_empty_leaves(root, this->now_leaves);
                this->get_edges(this->now_leaves);

                for (auto &kv : this->edges)
                {
                    this->find_node(kv.second->a->tree_node)->edges.emplace(kv.first);
                    this->find_node(kv.second->b->tree_node)->edges.emplace(kv.first);
                }
                return;
            }
            if (root != this->last_root)
            {
                this->last_leaves.clear();
                this->edges.clear();
                this->nodes.clear();
            }
            this->now_leaves.clear();
            this->get_empty_leaves(root, this->now_leaves);

            for (auto it = this->nodes.begin(); it != this->nodes.end();)
            {
                if (!this->now_leaves.count(it->first))
                {
                    it = this->nodes.erase(it);
                }
                else
                {
                    it->second->from_node = 0;
                    ++it;
                }
            }
            this->get_edges_neighbor(this->now_leaves);
            for (auto &kv : this->edges)
            {
                kv.second->a->edges.emplace(kv.first);
                kv.second->b->edges.emplace(kv.first);
            }
            this->last_root = root;
            this->last_leaves = this->now_leaves;
        }
        void construct_path(std::shared_ptr<PathNode> current, std::vector<TreeNode::Ptr> &path_list)
        {

            while (true)
            {
                path_list.emplace_back(current->tree_node);
                if (current->from_node <= 0)
                {
                    break;
                }
                current = this->nodes.at(current->from_node);
            }
            std::reverse(path_list.begin(), path_list.end());
        }
        void get_path(TreeNode::ConstPtrRef tree_start,
                      TreeNode::ConstPtrRef tree_end,
                      std::vector<TreeNode::Ptr> &out_path,
                      bool unknown_penalty = true)
        {
            out_path.clear();
            auto start = this->find_node(tree_start);
            auto end = this->find_node(tree_end);

            if (start == nullptr || end == nullptr)
            {
                return;
            }
            int iter_count = 0, max_iter_limit = 100000;
            std::set<std::shared_ptr<PathNode>, compare_f> open_set;
            std::unordered_set<long> open_id_set;
            std::unordered_set<long> close_set;
            start->g = 0;
            start->h = start->distance(end, unknown_penalty);
            start->f = start->g + start->h;
            open_set.emplace(start);
            open_id_set.emplace(start->id);

            while (open_set.size() > 0)
            {
                iter_count += 1;
                auto current = *open_set.begin();
                open_set.erase(open_set.begin());
                open_id_set.erase(current->id);
                if (iter_count > max_iter_limit)
                {
                    this->construct_path(current, out_path);
                    return;
                }
                if (current->id == end->id)
                {
                    this->construct_path(current, out_path);
                    return;
                }
                close_set.emplace(current->id);
                for (auto &ll : current->edges)
                {
                    auto e = this->edges.at(ll);
                    auto neighbor = e->b;
                    if (current->id == neighbor->id)
                    {
                        neighbor = e->a;
                    }
                    if (close_set.count(neighbor->id))
                    {
                        continue;
                    }
                    auto g_score = current->g + current->distance(neighbor, unknown_penalty);
                    if (g_score < neighbor->g || (!open_id_set.count(neighbor->id)))
                    {
                        neighbor->g = g_score;
                        neighbor->h = neighbor->distance(end, unknown_penalty);
                        neighbor->f = neighbor->g + neighbor->h;
                        neighbor->from_node = current->id;

                        open_set.emplace(neighbor);
                        open_id_set.emplace(neighbor->id);
                    }
                }
            }
            return;
        }
        void interpolation_center(
            const std::vector<TreeNode::Ptr> &path,
            std::vector<std::array<float, TREE_DIM>> &out_path) const
        {
            out_path.clear();
            if (path.size() <= 0)
            {
                return;
            }
            out_path.emplace_back(path[0]->center);
            for (int i = 1; i < path.size(); i++)
            {
                auto f = path.at(i - 1);
                auto t = path.at(i);
                std::array<float, TREE_DIM> center;
                auto c = f->contact_center(t, center);
                if (c)
                {
                    out_path.emplace_back(center);
                }
                out_path.emplace_back(t->center);
            }
        }
    };
#undef for_dims
}
#endif
#endif