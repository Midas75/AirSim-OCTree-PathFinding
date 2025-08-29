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
void dump_test() {
	auto tn = ctree::TreeNode::create()->as_root(
		{ 0,0,0 }, { 30,30,30 }, { 1,1,1 }, 3
	);
	auto pg = ctree::PathGraph();
	for (int i = tn->i_bound_min[0]; i < tn->i_bound_max[0]; i++) {
		for (int j = tn->i_bound_min[1]; j < tn->i_bound_max[1]; j++) {
			tn->add_i({ i,j,tn->i_bound_min[2] });
		}
	}
	for (int i = tn->i_bound_min[0] + 10; i < tn->i_bound_max[0] - 10; i++) {
		for (int j = tn->i_bound_min[1] + 10; j < tn->i_bound_max[1] - 10; j++) {
			for (int k = tn->i_bound_min[2]; k < tn->i_bound_max[2] - 10; k++) {
				tn->add_i({ i,j,k });
			}
			pg.update(tn);
		}
	}
	std::vector<ctree::TreeNode::Ptr> outpath;
	pg.get_path(
		tn->query({ 0,0,5 }, true),
		tn->query({ tn->bound_max[0],tn->bound_max[1],5 }, true),
		outpath
	);
	std::vector<std::array<float, 3>> ic, ps;
	tn->interpolation_center(outpath, ic);
	tn->path_smoothing(ic, ps);
}
int main()
{
	dump_test();
	return 0;
}