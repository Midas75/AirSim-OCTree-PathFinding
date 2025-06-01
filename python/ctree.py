import os

import cppyy

cppyy.include(f"{os.path.dirname(__file__)}/ctree.hpp")
