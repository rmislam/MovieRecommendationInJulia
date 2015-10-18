# Movie Recommendation Engine
# Russ Islam
#############################

data = readdlm("ratings.dat");

# ratings matrix, with rows as users and columns as movies
R = zeros(6040, 3952);

# go through dataset and populate ratings matrix
for a in 1:length(data)
    entry = split(data[a], "::");
    R[int(entry[1]), int(entry[2])] = float(entry[3]); 
end

# weight matrix (1 if there is a rating, 0 otherwise)
W = float(R .> 0);
    
# hyperparameters
k = 5; # number of features
λ = 0.01; # learning rate
maxiter = 10; # maximum number of iterations of alternating least squares (ALS)

# Initialize X and Y to random values
X = rand(k, size(R)[1]); # users 
Y = rand(k, size(R)[2]); # movies

# Alternating least squares (ALS)
for j = 1:maxiter 

    # update users matrix
    for u in 1:size(R)[1]
        X[:,u] = (Y*diagm(vec(W[u,:]))*Y' + λ*eye(k))\(Y*R[u,:]'); 
    end
    
    # update movies matrix
    for i in 1:size(R)[2]
        Y[:,i] = (X*diagm(vec(W[:,i]))*X' + λ*eye(k))\(X*R[:,i]); 
    end    
end

# Construct the estimated ratings matrix Q from the users matrix and the movies matrix
Q = X'*Y;

# Save matrices
writedlm("users_matrix_estimate.txt", X);
writedlm("movies_matrix_estimate.txt", Y);
writedlm("ratings_matrix_estimate.txt", Q);


