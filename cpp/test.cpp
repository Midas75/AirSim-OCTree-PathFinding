#include "ctree.hpp"
void update_test()
{
	auto tn = ctree::TreeNode::create()->as_root(
		{0, 0, 0}, {100, 100, 100}, {1, 1, 1}, 3);
	auto pg = ctree::PathGraph();
	int number = 50;
	for (int i = 0; i <= number; i++)
	{
		tn->add_raycast(
			{0, 0, 0},
			{tn->bound_size[0] * std::sin(3.1415926f / 2 * i / number),
			 tn->bound_size[1] * std::cos(3.1415926f / 2 * i / number),
			 0},
			false);
		pg.update(tn);
	}
}
void main()
{
	update_test();
}