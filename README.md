TODO:
- should we use global indices in queries to guarantee result independent of tile size?
- robust test suite to have high confidence in correctness
- do not support check if triplet, persistence, and persistencebelow input is a maximum
- implement the reference using the sweep algorithm for better performance (especially for triplets and persistence)
- make it C++ library (std::sort is 2-3x faster than qsort; maybe implement our own sort?)

Topological analysis based on localized data structures.


Requirements
---------------
Python 3.6, numpy 1.13
C99 compiler (MSVC 16, gcc 4.8, clang 3.4)