# movie_recommendation
Matrix factorization using alternating least squares for movie recommendation

The file ```recommend.jl``` contains a single run of the ALS algorithm with no cross-validation or parameter tuning.

The file ```recommend_full.jl``` contains the ALS algorithm with 10-fold cross-validation and considers ranges for three parameters: the number of features, the learning rate, and the maximum number of iterations of the algorithm.

Both files use the ```ratings.dat``` file from the 1M dataset from http://grouplens.org/datasets/movielens/

