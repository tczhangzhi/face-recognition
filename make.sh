cmake . -DCMAKE_BUILD_TYPE=Release  -DBLAS="Open" -Dpython_version=3 -DCUDA_HOST_COMPILER=/usr/bin/g++-5 -DCUDA_PROPAGATE_HOST_FLAGS=off -DCMAKE_CXX_FLAGS="-std=c++11"
make
./dist/main