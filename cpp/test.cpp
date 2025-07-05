#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <utility>
#include "ctree.hpp"
void update_test()
{
	auto tn = ctree::TreeNode::create()->as_root(
		{ 0, 0, 0 }, { 1000, 1000, 1000 }, { 1, 1, 1 }, 3);
	auto pg = ctree::PathGraph();
	int number = 5000;
	for (int i = 0; i <= number; i++)
	{
		tn->add_raycast(
			{ 0, 0, 0 },
			{ tn->bound_size[0] * std::sin(3.1415926f / 2 * i / number),
			 tn->bound_size[1] * std::cos(3.1415926f / 2 * i / number),
			 0 },
			false);
		pg.update(tn);
		auto ov = std::vector<ctree::TreeNode::Ptr>();
		pg.get_path(tn->query({ 2, 2, 2 }), tn->query({ 1000,1000,1000 }), ov);
	}
}
void test_pair_code_ll_performance(std::size_t test_size = 500'000'000)
{
	std::cout << "Testing performance of pair_code_ll with " << test_size << " pairs...\n";
	std::mt19937_64 rng(42);
	std::uniform_int_distribution<uint32_t> dist(0, 1'000'000);
	std::vector<std::pair<uint32_t, uint32_t>> test_data;
	test_data.reserve(test_size);
	for (std::size_t i = 0; i < test_size; ++i)
	{
		test_data.emplace_back(dist(rng), dist(rng));
	}
	auto start = std::chrono::high_resolution_clock::now();
	std::uint64_t checksum = 0;
	for (const auto& p : test_data)
	{
		checksum += ctree::pair_code_ll(p);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;

	std::cout << "Total time: " << duration.count() << " seconds\n";
	std::cout << "Checksum (to prevent optimization): " << checksum << "\n";
}
void hash_collide_test()
{
	std::unordered_set<std::uint64_t> set;
	uint32_t test_range = 10000;
	for (uint32_t i = test_range; i > 0; i--)
	{
		for (uint32_t j = i; j > 0; j--)
		{
			auto pair = std::make_pair(i, j);
			auto id = ctree::pair_code_ll(pair);
			if (set.count(id))
			{
				printf("collided! %lld\n", id);
				getchar();
			}
			else
			{
				set.emplace(id);
			}
		}
		printf("%ld\n", i);
	}
}
void find_test() {
	auto tn = ctree::TreeNode::create()->as_root(
		{ 0, 0, 0 }, { 1000, 1000, 1000 }, { 1, 1, 1 }, 3);
	auto pg = ctree::PathGraph();
	pg.update(tn);
	auto ov = std::vector<ctree::TreeNode::Ptr>();
	auto n = tn->query({ 0,0,0 });
	pg.get_path(n, n, ov);
}
int main()
{
	find_test();
	return 0;
}