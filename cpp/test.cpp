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
void hash_collide_test()
{
	std::unordered_set<std::pair<long, long>,ctree::pair_code_ll> set;
	long test_range = 10000;
	struct ctree::pair_code_ll hasher;
	for (long i = test_range; i >0; i--)
	{
		for (long j = i; j >0; j--)
		{
			auto pair = std::make_pair(i, j);
			auto id = hasher(pair);
			if (set.count(pair))
			{
				printf("collided! %lld\n",id);
				getchar();
			}
			else
			{
				set.emplace(pair);
			}
		}
		printf("%ld\n", i);
	}
}
void main()
{
	hash_collide_test();
}