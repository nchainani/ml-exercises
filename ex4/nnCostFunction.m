function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X];

z2 = X * transpose(Theta1)

a2 =  (1 ./ (1 + (e .** -z2)))

m2 = size(a2, 1);1

a2 = [ones(m2, 1) a2];1

z3 = a2 * transpose(Theta2);1

a3 =  (1 ./ (1 + (e .** -z3)));1

[a, p] = max(a3, [], 2);1

total = 0

second = a3;1

for i = 1:m
  xx = transpose(a3(i,:)) %zeros([num_labels;1]);1 %
  %index1 = p(i,1)
  %xx(index1, 1) = 1;
  yy = zeros([num_labels;1]);1
  index2 = y(i,1)
  yy(index2,1) = 1;1
  total = total + sum((-yy .* log(xx) - ((1 - yy) .* log(1 - xx))));1
end

total = 1/m * total;

t = (lambda / (2 * m))
reg = (sum(sum(Theta1(:,2:size(Theta1)(2)) .** 2)) + sum(sum(Theta2(:,2:size(Theta2)(2)) .** 2)))
reg = t * reg
J = total + reg

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

delta2 = 0
delta1 = 0

for i = 1:m
  a1 = transpose(X(i,:))
  %a1 = [1 a1]
  z2 = Theta1 * a1
  a2 = (1 ./ (1 + (e .** -z2)))
  a2 = [1;a2]
  z3 = Theta2 * a2;1
  a3 = (1 ./ (1 + (e .** -z3)));1

  yy = zeros([num_labels;1]);1
  index2 = y(i,1)
  yy(index2,1) = 1;1

  sdelta3 = a3 - yy
  sdelta2 = (transpose(Theta2) * sdelta3)(2:end) .* sigmoidGradient(z2)

  delta2 = delta2 + (sdelta3 * transpose(a2))
  delta1 = delta1 + (sdelta2 * transpose(a1))
end


ltheta1 = Theta1
ltheta1(:,1) = 0

ltheta2 = Theta2
ltheta2(:,1) = 0
reg1 = ltheta1 .* (lambda / m)
reg2 = ltheta2 .* (lambda / m)

Theta1_grad
reg1

Theta1_grad = 1/m * delta1 + reg1
Theta2_grad = 1/m * delta2 + reg2

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
