pip install -r requirements.txt || exit /b
pip install cppyy-cling cppyy-backend || exit /b
pip install CPyCppyy --no-deps --no-build-isolation || exit /b
pip install cppyy || exit /b