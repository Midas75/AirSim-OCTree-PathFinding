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
#include <queue>
namespace ctree
{
#ifndef TREE_DIM
#define TREE_DIM 3
#endif
#ifndef for_dims
#define for_dims(_this) for (int8_t dim = 0; dim < (_this)->dims; dim++)

    template <typename T, std::size_t N>
    inline T dist(const std::array<T, N> &a, const std::array<T, N> &b)
    {
        T sum = 0;
        for (size_t i = 0; i < N; ++i)
        {
            T diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    constexpr int8_t TREE_CHILDS = 1 << TREE_DIM;
    constexpr std::array<float, TREE_DIM> ZEROS = {0};
    static const float INF = std::numeric_limits<float>::infinity();
    struct TreeData
    {
        float min_length[TREE_DIM] = {0}, bound_min[TREE_DIM] = {0}, bound_max[TREE_DIM] = {0};
        int8_t dims = TREE_DIM;
    };
    struct TreeNodeData
    {
        uint32_t id = 0;
        uint32_t child[TREE_CHILDS] = {0};
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

        static const int8_t EMPTY = 0;
        static const int8_t FULL = 1;
        static const int8_t HALF_FULL = 2;

        Ptr child[TREE_CHILDS];
        bool no_child = true;

        int8_t state = TreeNode::EMPTY;
        int dynamic_culling = -1;
        uint32_t last_ray_id = 0;
        uint32_t ray_id = 0;

        std::array<float, TREE_DIM> bound_min = {0},
                                    bound_max = {0},
                                    bound_size = {0},
                                    center = {0},
                                    min_length = {0};
        TreeNode *parent = nullptr, *root = nullptr;
        std::shared_ptr<std::unordered_map<uint32_t, std::weak_ptr<TreeNode>>> nodes; // avoiding cycle ref

        int8_t dims = TREE_DIM;
        int8_t directions = TREE_CHILDS;

        bool _min = false, is_leaf = true, known = false;
        std::array<int, TREE_DIM> i_bound_min = {0},
                                  i_bound_max = {0},
                                  i_bound_size = {0},
                                  i_center = {0};
        uint32_t id = 0;
        void serialize(TreeData &out_tree_data, std::vector<TreeNodeData> &out_tree_node_data) const
        {
            if (this->parent != nullptr)
            {
                return;
            }
            out_tree_data.dims = this->dims;
            std::copy(this->min_length.begin(), this->min_length.end(), out_tree_data.min_length);
            std::copy(this->bound_min.begin(), this->bound_min.end(), out_tree_data.bound_min);
            std::copy(this->bound_max.begin(), this->bound_max.end(), out_tree_data.bound_max);
            for (auto &kv : *this->nodes)
            {
                auto otnd = TreeNodeData();

                otnd.id = kv.first;
                if (auto p = kv.second.lock())
                {
                    std::copy(p->i_bound_min.begin(), p->i_bound_min.end(), otnd.i_bound_min);
                    std::copy(p->i_bound_max.begin(), p->i_bound_max.end(), otnd.i_bound_max);
                    otnd.known = p->known;
                    otnd.state = p->state;
                    otnd.is_leaf = p->is_leaf;
                    if (!p->no_child)
                    {
                        for (int i = 0; i < TREE_CHILDS; i++)
                        {
                            auto &c = p->child[i];
                            if (c != nullptr)
                            {
                                otnd.child[i] = c->id;
                            }
                        }
                    }
                    out_tree_node_data.emplace_back(std::move(otnd));
                }
            }
        }

        static ConstPtr deserialize(const TreeData &tree_data, const std::vector<TreeNodeData> &tree_node_datas)
        {
            auto map = std::unordered_map<uint32_t, uint64_t>();
            map.reserve(tree_node_datas.size());
            for (uint64_t i = 0; i < tree_node_datas.size(); i++)
            {
                map.emplace(tree_node_datas[i].id, i);
            }
            return deserialize(tree_data, tree_node_datas, map);
        }
        static ConstPtr deserialize(const TreeData &tree_data, const std::vector<TreeNodeData> &tree_node_datas,
                                    const std::unordered_map<uint32_t, uint64_t> &_tree_nodes_map,
                                    uint32_t _current_id = 0, ConstPtr _parent = Nullptr)
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
            auto relink_node = std::unordered_map<uint32_t, int8_t>();
            for (int8_t direction = 0; direction < TREE_CHILDS; direction++)
            {
                auto c_id = info.child[direction];
                if (c_id > 0)
                {
                    node->no_child = false;
                    if (!relink_node.count(c_id))
                    {
                        node->child[direction] = TreeNode::deserialize(
                            tree_data, tree_node_datas,
                            _tree_nodes_map,
                            c_id, node);
                        relink_node.emplace(c_id, direction);
                    }
                    else
                    {
                        node->child[direction] = node->child[relink_node[c_id]];
                    }
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
                self->nodes = parent->nodes;

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
            if (this->parent == nullptr)
            {
                return;
            }
            this->bound_max.fill(0);
            this->bound_min.fill(0);
            this->bound_size.fill(0);
            this->i_bound_size.fill(0);
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
            this->nodes->erase(this->id);
            this->id = this->gen_id();
            this->nodes->emplace(this->id, shared_from_this());
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
        uint32_t gen_id() const
        {
            uint32_t result = 0;
            uint32_t dim_range = 1;
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
            this->nodes = std::make_shared<std::unordered_map<uint32_t, std::weak_ptr<TreeNode>>>();

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
            this->nodes->emplace(this->id, shared_from_this());
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
                this->remove_child();
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
            this->remove_child();
            this->state = state;
            this->known = false;
            auto parent = this->parent;
            while (parent != nullptr)
            {
                parent->update_state();
                parent = parent->parent;
            }
        }
        void remove_child()
        {
            if (this->no_child)
            {
                return;
            }
            for (int i = 0; i < TREE_CHILDS; i++)
            {
                if (this->child[i] != nullptr)
                {
                    this->child[i]->remove_child();
                    this->nodes->erase(this->child[i]->id);
                    this->child[i] = nullptr;
                }
            }
            this->no_child = true;
            this->is_leaf = true;
        }
        static const std::shared_ptr<const TreeNode> lca(ConstPtr node1, ConstPtr node2)
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
            float tmin = -INF;
            float tmax = INF;

            for_dims(this)
            {
                float b_min = this->bound_min[dim] - expand[dim];
                float b_max = this->bound_max[dim] + expand[dim];
                if (inv_vector[dim] == INF)
                {
                    if (start[dim] < b_min || start[dim] > b_max)
                    {
                        return false;
                    }
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
            auto ex_start = start;
            auto ex_end = end;
            for_dims(this)
            {
                if (start[dim] < end[dim])
                {
                    ex_start[dim] -= expand[dim];
                    ex_end[dim] += expand[dim];
                }
                else
                {
                    ex_start[dim] += expand[dim];
                    ex_end[dim] -= expand[dim];
                }
            }
            auto &n1 = this->query(ex_start, true);
            auto &n2 = this->query(ex_end, true);
            std::array<float, TREE_DIM> inv_vector = {0};
            for_dims(this)
            {
                float v = end[dim] - start[dim];
                if (v == 0)
                {
                    inv_vector[dim] = INF;
                }
                else
                {
                    inv_vector[dim] = 1 / v;
                }
            }
            return TreeNode::lca(n1, n2)->cross(start, inv_vector, expand);
        }
        static const std::shared_ptr<const TreeNode> get_parent(ConstPtr self, int number = 1)
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
        bool intersect(ConstPtr other) const
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
            float t_min = -INF;
            float t_max = INF;
            int8_t out_dim = 0;
            for_dims(this)
            {
                float t1, t2;
                if (vector[dim] == 0)
                {
                    t1 = INF;
                    t2 = INF;
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
        void next_ray_batch()
        {
            this->root->ray_id += 1;
        }
        const float center_to_segment(const std::array<float, TREE_DIM> &start, const std::array<float, TREE_DIM> &point)
        {
            auto ab = point, ap = this->center;
            float ab2 = 0, t = 0;
            for_dims(this)
            {
                ab[dim] -= start[dim];
                ap[dim] -= start[dim];
                ab2 += ab[dim] * ab[dim];
                t += ap[dim] * ab[dim];
            }
            if (ab2 == 0)
            {
                return dist(ap, start);
            }
            t /= ab2;
            t = std::max(std::min(t, 1.f), 0.f);
            auto q = start;
            for_dims(this)
            {
                q[dim] += t * ab[dim];
            }
            return dist(this->center, q);
        }
        void add_raycast(const std::array<float, TREE_DIM> &start,
                         const std::array<float, TREE_DIM> &point,
                         bool empty_end = false,
                         int dynamic_culling = 10,
                         float center_limit = 0.5f)
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
            std::unordered_set<uint32_t> visit;
            while (true)
            {
                auto cnode = this->query(current);
                if (cnode == TreeNode::Nullptr ||
                    cnode == end ||
                    visit.count(cnode->id))
                {
                    break;
                }
                for_dims(this)
                {
                    float cd = abs(current[dim] - start[dim]);
                    float ad = abs(direction[dim]);
                    if (cd > ad)
                    {
                        quit = true;
                        break;
                    }
                }
                if (quit)
                {
                    break;
                }
                if (
                    cnode->state == TreeNode::FULL && dynamic_culling > 0 && cnode->last_ray_id != cnode->root->ray_id && cnode->center_to_segment(start, point) <= center_limit)
                {
                    cnode->last_ray_id = cnode->root->ray_id;
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
        bool path_smoothing(
            const std::vector<std::array<float, TREE_DIM>> &path,
            std::vector<std::array<float, TREE_DIM>> &out_path,
            const std::array<float, TREE_DIM> &expand = ZEROS,
            float break_length = 1.f)
        {
            bool changed = false;
            if (path.size() <= 1)
            {
                out_path = path;
                return changed;
            }
            auto _path(path);
            out_path.clear();
            out_path.emplace_back(path[0]);
            if (break_length > 0)
            {
                _path.clear();
                for (int i = 0; i < path.size() - 1; i++)
                {
                    auto &p1 = path[i], &p2 = path[i + 1];
                    auto d = dist(p1, p2);
                    if (d < break_length)
                    {
                        _path.emplace_back(p2);
                    }
                    else
                    {
                        auto n = std::ceil(d / break_length);
                        for (int j = 1; j < n + 1; j++)
                        {
                            auto t = j / n;
                            auto np = std::array<float, TREE_DIM>();
                            for_dims(this)
                            {
                                np[dim] = p1[dim] + (p2[dim] - p1[dim]) * t;
                            }
                            _path.emplace_back(np);
                        }
                    }
                }
            }
            uint64_t size_1 = _path.size() - 1;
            uint64_t i = 0;
            while (i < size_1)
            {
                uint64_t j = i + 2;
                while (j <= size_1)
                {
                    bool cl = this->cross_lca(_path[i], _path[j], expand);
                    if (cl)
                    {
                        changed = true;
                        break;
                    }
                    j += 1;
                }
                out_path.emplace_back(_path[j - 1]);
                i = j - 1;
            }
            return changed;
        }
        void get_neighbor(std::unordered_map<uint32_t, ConstPtr> &out_map) const
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
        uint8_t contact_with(ConstPtr other) const
        {
            uint8_t result = 0;
            int true_counter = 0;
            for_dims(this)
            {
                auto size = (this->i_bound_size[dim] + other->i_bound_size[dim]) / 2;
                auto c2c = abs(this->i_center[dim] - other->i_center[dim]);
                if (size < c2c)
                {
                    return result;
                }
                true_counter += 1;
                if (size == c2c)
                {
                    result |= 1 << (dim + 1);
                }
            }
            result |= 1;
            return result;
        }
        bool contact_center(ConstPtr other, std::array<float, TREE_DIM> &out_center) const
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
                    if (this->i_center[dim] < other->i_center[dim])
                    {
                        out_center[dim] = this->center[dim] + this->bound_size[dim] / 2;
                    }
                    else
                    {
                        out_center[dim] = this->center[dim] - this->bound_size[dim] / 2;
                    }
                }
                else
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
            }
            return true;
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
    std::uint64_t pair_code_ll(const std::pair<uint32_t, uint32_t> &p)
    {
        std::uint64_t a = p.first, b = p.second;
        // Cantor pairing function
        std::uint64_t sum = a + b;
        return (sum * (sum + 1)) / 2 + std::min(a, b);
    };
    class PathNode
    {
    public:
        const uint32_t id;
        const int dims;
        std::unordered_set<std::uint64_t> edges;
        TreeNode::ConstPtr tree_node;
        float f = 0, g = INF, h = 0;
        uint32_t from_node = 0;
        PathNode(TreeNode::ConstPtr tree_node) : tree_node(tree_node), id(tree_node->id), dims(tree_node->dims)
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
            return dist(this->tree_node->center, other->tree_node->center) * up_factor;
        }
    };
    struct compare_f
    {
        bool operator()(const std::shared_ptr<PathNode> &a, const std::shared_ptr<PathNode> &b) const
        {
            return a->f > b->f;
        }
    };
    class PathEdge
    {
    public:
        const std::shared_ptr<PathNode> a, b;
        std::pair<uint32_t, uint32_t> id;
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
        std::unordered_map<uint32_t, std::shared_ptr<PathNode>> nodes;
        std::unordered_map<std::uint64_t, std::shared_ptr<PathEdge>> edges;
        TreeNode::Ptr last_root = nullptr;
        std::unordered_map<uint32_t, TreeNode::Ptr> now_leaves, last_leaves;

        uint32_t add_node(TreeNode::ConstPtrRef tree_node)
        {
            if (!this->nodes.count(tree_node->id))
            {
                this->nodes.emplace(tree_node->id, std::make_shared<PathNode>(tree_node));
            }
            return tree_node->id;
        }

        void remove_node(uint32_t node_id, bool remove_edge = true)
        {
            std::shared_ptr<PathNode> pn;
            if (this->nodes.count(node_id))
            {
                pn = this->nodes.at(node_id);
                this->nodes.erase(node_id);
                if (remove_edge)
                {
                    for (auto &edge_id : pn->edges)
                    {
                        auto &e = this->edges.at(edge_id);
                        e->a->edges.erase(edge_id);
                        e->b->edges.erase(edge_id);
                    }
                }
            }
        }
        void add_edge(TreeNode::ConstPtrRef a, TreeNode::ConstPtrRef b)
        {
            auto edge_id = pair_code_ll(std::make_pair(a->id, b->id));
            if (!this->edges.count(edge_id))
            {
                this->add_node(a);
                this->add_node(b);
                auto pn1 = this->nodes.at(a->id);
                auto pn2 = this->nodes.at(b->id);
                this->edges.emplace(edge_id, std::make_shared<PathEdge>(pn1, pn2));
                pn1->edges.emplace(edge_id);
                pn2->edges.emplace(edge_id);
            }
        }
        void remove_edge(const std::uint64_t edge_id, bool remove_from_this = false)
        {
            if (this->edges.count(edge_id))
            {
                auto &pe = this->edges.at(edge_id);
                if (remove_from_this)
                {
                    this->edges.erase(edge_id);
                }
                pe->a->edges.erase(edge_id);
                pe->b->edges.erase(edge_id);
            }
        }
        void get_empty_leaves(TreeNode::ConstPtrRef tree_node, std::unordered_map<uint32_t, TreeNode::Ptr> &leaves)
        {
            if (tree_node->is_leaf && tree_node->state == TreeNode::EMPTY)
            {
                leaves.emplace(tree_node->id, tree_node);
            }
            else if (tree_node->state != TreeNode::FULL && !tree_node->no_child)
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
        void update_edges(std::unordered_map<uint32_t, TreeNode::Ptr> &leaves)
        {
            this->nodes.clear();
            this->edges.clear();
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
        void update_edges_neighbor(std::unordered_map<uint32_t, TreeNode::Ptr> &leaves)
        {
            std::unordered_set<uint32_t> active_nodes;
            for (auto it = this->edges.begin(); it != this->edges.end();)
            {
                auto &e = it->second;
                auto &tna = e->a->tree_node;
                auto &tnb = e->b->tree_node;
                bool a_expire = !leaves.count(tna->id);
                bool b_expire = !leaves.count(tnb->id);
                if (a_expire)
                {
                    this->remove_node(tna->id, false);
                    if (!b_expire)
                    {
                        active_nodes.emplace(tnb->id);
                    }
                }
                if (b_expire)
                {
                    this->remove_node(tnb->id, false);
                    if (!a_expire)
                    {
                        active_nodes.emplace(tna->id);
                    }
                }
                if (a_expire || b_expire)
                {
                    this->remove_edge(it->first);
                    it = this->edges.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            std::unordered_map<uint32_t, TreeNode::ConstPtr> neighbors;
            for (auto &kv : leaves)
            {
                if (active_nodes.count(kv.first) || (!this->last_leaves.count(kv.first)))
                {
                    neighbors.clear();
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
            this->now_leaves.clear();
            this->get_empty_leaves(root, this->now_leaves);
            if ((root != this->last_root) || full_reset)
            {
                this->last_leaves.clear();
                if (full_reset)
                {
                    this->update_edges(this->now_leaves);
                }
                else
                {
                    this->update_edges_neighbor(this->now_leaves);
                }
                this->last_root = root;
            }
            else
            {
                this->update_edges_neighbor(this->now_leaves);
            }
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
        void get_path(TreeNode::ConstPtr tree_start,
                      TreeNode::ConstPtr tree_end,
                      std::vector<TreeNode::Ptr> &out_path,
                      bool unknown_penalty = true)
        {
            out_path.clear();
            auto start_it = this->nodes.find(tree_start->id);
            auto end_it = this->nodes.find(tree_end->id);
            if ((start_it == this->nodes.end()) || (end_it == this->nodes.end()))
            {
                return;
            }
            auto start = start_it->second;
            auto end = end_it->second;

            int iter_count = 0, max_iter_limit = 100000;
            std::priority_queue<std::shared_ptr<PathNode>, std::vector<std::shared_ptr<PathNode>>, compare_f> open_heap;
            std::unordered_set<uint32_t> open_set_ids;
            std::unordered_set<uint32_t> close_set;

            start->g = 0;
            start->h = start->distance(end, unknown_penalty);
            start->f = start->g + start->h;
            start->from_node = 0;

            open_heap.emplace(start);
            open_set_ids.emplace(start->id);

            while (open_heap.size() > 0)
            {

                iter_count += 1;
                auto current = open_heap.top();

                open_heap.pop();
                open_set_ids.erase(current->id);

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
                for (auto &eid : current->edges)
                {
                    if (!this->edges.count(eid))
                    {
                        continue;
                    }
                    auto &e = this->edges.at(eid);
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
                    if (g_score < neighbor->g || (!open_set_ids.count(neighbor->id)))
                    {
                        neighbor->g = g_score;
                        neighbor->h = neighbor->distance(end, unknown_penalty);
                        neighbor->f = neighbor->g + neighbor->h;
                        neighbor->from_node = current->id;
                        open_heap.emplace(neighbor);
                        open_set_ids.emplace(neighbor->id);
                    }
                }
            }
            return;
        }
    };
    typedef std::array<float, TREE_DIM> F_STD_ARRAY;
    typedef std::array<int, TREE_DIM> I_STD_ARRAY;
    typedef std::vector<TreeNode::Ptr> TREENODE_STD_VECTOR;
    typedef std::vector<F_STD_ARRAY> FARRAY_STD_VECTOR;
    typedef std::pair<uint32_t, uint32_t> LL_STD_PAIR;
    typedef std::vector<TreeNodeData> TREENODE_DATA_STD_VECTOR;
#undef for_dims
}
#endif
#endif