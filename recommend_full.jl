# Movie Recommendation Engine
# Russ Islam
#############################

data = readdlm("ratings.dat");

# hyperparameters
k_vals = [5 10 15]; # number of features
λ_vals = [0.01 0.05 0.1]; # learning rate
maxiter_vals = [5 10 15]; # maximum number of iterations for alternating least squares (ALS)

# training and validation root-mean-square error matrices
training_error_mat = zeros(length(k_vals), length(λ_vals), length(maxiter_vals));
validation_error_mat = zeros(length(k_vals), length(λ_vals), length(maxiter_vals));

for k in k_vals
    for λ in λ_vals
        for maxiter in maxiter_vals

            total_training_error = 0;
            total_validation_error = 0;

            # 10-fold cross validation
            for fold = 1:10
    
                # ratings matrix, with rows as users and columns as movies
                R = zeros(6040, 3952);
                R_validation = zeros(6040,3952);

                # Create indices for training and validation sets
                indices = shuffle([1:length(data)]);
                training_indices = indices[(int(length(data) / 10) + 1):end];    
                validation_indices = indices[1:int(length(data) / 10)];

                # Populate training ratings matrix
                for a in training_indices
                    entry = split(data[a], "::");
                    R[int(entry[1]), int(entry[2])] = float(entry[3]);
                end

                # Populate validation ratings matrix
                for a in validation_indices
                    entry = split(data[a], "::");
                    R_validation[int(entry[1]), int(entry[2])] = float(entry[3]);
                end

                # weight matrix (1 if there is a rating, 0 otherwise) for training ratings matrix
                W = float(R .> 0);
    
                # Initialize X and Y to random values
                X = (5.0/k) * rand(k, size(R)[1]); # users 
                Y = (5.0/k) * rand(k, size(R)[2]); # movies
    
                # Alternating least squares (ALS)
                for j = 1:maxiter
                   
                    # Update users matrix
                    for u in 1:size(R)[1]
                        X[:,u] = (Y*diagm(vec(W[u,:]))*Y' + λ*eye(k))\(Y*R[u,:]');
                    end
        
                    # Update movies matrix
                    for i in 1:size(R)[2]
                        Y[:,i] = (X*diagm(vec(W[:,i]))*X' + λ*eye(k))\(X*R[:,i]);
                    end
                end
    
                # Estimated ratings matrix Q
                Q = X'*Y;
    
                # Compute training rmse
                training_error = 0;
                for a in training_indices
                    entry = split(data[a], "::");
                    training_error += (Q[int(entry[1]), int(entry[2])] - R[int(entry[1]), int(entry[2])])^2;
                end
                training_error /= length(training_indices);
                total_training_error += sqrt(training_error);

                # Compute validation rmse
                validation_error = 0;
                for a in validation_indices
                    entry = split(data[a], "::");
                    validation_error += (Q[int(entry[1]), int(entry[2])] - R_validation[int(entry[1]), int(entry[2])])^2;
                end
                validation_error /= length(validation_indices);
                total_validation_error += sqrt(validation_error);
            end

            # Average errors over the 10 folds and store in error matrices
            training_error_mat[find(k_vals .== k), find(λ_vals .== λ), find(maxiter_vals .== maxiter)] = total_training_error/10.0;
            validation_error_mat[find(k_vals .== k), find(λ_vals .== λ), find(maxiter_vals .== maxiter)] = total_validation_error/10.0;
            
            # Save matrices
            writedlm("training_results.txt", training_error_mat);
            writedlm("validation_results.txt", validation_error_mat);
        end
    end
end

# Examine the matrix stored in validation_results.txt and choose the minimum element. The indices of this element correspond to the optimal hyperparameters k, λ, and maxiter. 

