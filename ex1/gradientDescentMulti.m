function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    fprintf('tttttttttt');
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    new_theta = zeros(size(theta))
    elements = size(X, 2)
    fprintf('elements %f\n', elements);
    for ele = 1:elements
        temp = theta(ele, 1)
        temp = temp - (alpha / m) * sum((X * theta - y).*X(:,ele))
        new_theta(ele, 1) = temp
    end

    % ============================================================

    % Save the cost J in every iteration    
    cost = computeCostMulti(X, y, theta);
    fprintf('$$$$$$$$$is this going down: ');
    fprintf('%f\n', cost);

    J_history(iter) = cost
    theta = new_theta
end

end
