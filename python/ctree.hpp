#ifndef C_TREE
#define C_TREE
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <memory>
#include <cmath>

namespace ctree
{
#ifndef TREE_DIM
#define TREE_DIM 3
#endif
    constexpr uint8_t TREE_CHILDS = 1 << TREE_DIM;
    class TreeNode
    {
    public:
        static const float INF;
        static const uint8_t EMPTY = 0;
        static const uint8_t FULL = 1;
        static const uint8_t HALF_FULL = 2;

        std::shared_ptr<TreeNode> child[TREE_CHILDS];
        bool no_child = true;

        uint8_t state;
        int dynamic_culling = -1;

        float bound_min[TREE_DIM] = {0},
              bound_max[TREE_DIM] = {0},
              bound_size[TREE_DIM] = {0},
              center[TREE_DIM] = {0},
              min_length[TREE_DIM] = {0};
        TreeNode *parent = nullptr, *root = nullptr;
        uint8_t dims;
        uint8_t directions;

        bool _min = false, is_leaf = true, known = false;

        int i_bound_min[TREE_DIM] = {0},
            i_bound_max[TREE_DIM] = {0},
            i_bound_size[TREE_DIM] = {0},
            i_center[TREE_DIM] = {0};
        long id;

        TreeNode(TreeNode *parent = nullptr,
                 const uint8_t direction = 0,
                 const uint8_t divide = 0)
        {
            this->state = TreeNode::EMPTY;
            for (int i = 0; i < TREE_CHILDS; i++)
            {
                this->child[i] = nullptr;
            }
            if (parent != nullptr)
            {
                this->parent = parent;
                this->root = parent->root;

                this->dims = parent->dims;
                this->directions = 1 << this->dims;
                memcpy(this->min_length, parent->min_length, TREE_DIM);
                uint8_t _divide = divide, _direction = direction;
                for (int dim = 0; dim < this->dims; dim++)
                {
                    _divide >>= dim;
                    _direction >>= dim;
                    if (!(_divide & 1))
                    {
                        this->i_bound_min[dim] = parent->i_bound_min[dim];
                        this->i_bound_max[dim] = parent->i_bound_max[dim];
                    }
                    else if (!(_direction & 1))
                    {
                        this->i_bound_min[dim] = parent->i_bound_min[dim];
                        this->i_bound_max[dim] = parent->i_center[dim];
                    }
                    else
                    {
                        this->i_bound_min[dim] = parent->i_center[dim];
                        this->i_bound_max[dim] = parent->i_bound_max[dim];
                    }
                }
                this->update_bound();
            }
        }
        void update_bound()
        {
            memset(this->center, 0, sizeof(this->center));
            memset(this->i_center, 0, sizeof(this->i_center));

            if (this->parent != nullptr)
            {
                memset(this->bound_max, 0, sizeof(this->bound_max));
                memset(this->bound_min, 0, sizeof(this->bound_min));
                memset(this->bound_size, 0, sizeof(this->bound_size));
                memset(this->i_bound_size, 0, sizeof(this->i_bound_size));
            }
            for (int dim = 0; dim < this->dims; dim++)
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
        const bool is_min()
        {
            for (int dim = 0; dim < this->dims; dim++)
            {
                if (!(this->i_center[dim] & 1))
                {
                    return false;
                }
            }
            return true;
        }
        const long gen_id()
        {
            long result = 0;
            long dim_range = 1;
            for (int dim = 0; dim < this->dims; dim++)
            {
                result += dim_range * this->i_center[dim];
                dim_range *= this->root->i_bound_size[dim];
            }
            return result;
        }
        void as_root(const float *bound_min, const float *bound_max, const float *min_length, const int dims = TREE_DIM)
        {
            size_t dim_size_f = sizeof(float) * dims;
            size_t dim_size_i = sizeof(int) * dims;
            this->dims = dims;
            this->directions = 1 << this->dims;
            this->root = this;
            this->parent = nullptr;

            memcpy(this->min_length, min_length, dim_size_f);
            memcpy(this->bound_min, bound_min, dim_size_f);
            memcpy(this->bound_max, bound_max, dim_size_f);

            for (int dim = 0; dim < this->dims; dim++)
            {
                this->bound_size[dim] = this->bound_max[dim] - this->bound_min[dim];
                this->center[dim] = this->bound_max[dim] / 2 + this->bound_min[dim] / 2;
            }

            for (int dim = 0; dim < this->dims; dim++)
            {
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
            for (int i = 0; i < this->directions; i++)
            {
                auto reduced = this->get_bound_by_direction(i);
                auto ri = reduced.first, d = reduced.second;
                if (this->child[ri] == nullptr)
                {
                    this->no_child = false;
                    auto c = std::make_shared<TreeNode>(this, i, d);
                    this->child[ri] = c;
                    this->child[i] = c;
                    c->divide(depth - 1);
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
            uint8_t full_counter = 0, empty_counter = 0, half_full_counter = 0;
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
        const std::pair<uint8_t, uint8_t> get_bound_by_direction(const uint8_t direction)
        {
            uint8_t index = direction;
            uint8_t divide = 0xFFu;
            for (int dim = 0; dim < this->dims; dim++)
            {
                uint8_t bit = 1u << dim;
                if (this->i_center[dim] & 1)
                {
                    divide &= ~bit;
                    index &= ~bit;
                }
            }
            return {index, divide};
        }
    };
    const float TreeNode::INF = std::numeric_limits<float>::infinity();
    class PathNode
    {
    public:
        int id;
        std::unordered_set<int> edges;
    };
}
#endif