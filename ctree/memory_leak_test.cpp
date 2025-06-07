// windows crt is needed in visual studio
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include <memory>
#include "ctree.hpp"
int _main()
{
    int test_number = 10000;
    // int *ptr = new int[100];
    for (int i = 0; i < test_number; i++)
    {
        auto tn = ctree::TreeNode::create()->as_root({0, 0}, {640, 640}, {2, 2}, 2);
        tn->add_raycast({0, 0}, {640, 640});
    }
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();
    return 0;
}