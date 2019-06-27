function [T, Z] = RobustTransformLearning (X, numOfAtoms, mu, tau)

% solves ||TX - Z||_1 - mu*logdet(T) + eps*mu||T||_Fro + tau||Z||_1

% Inputs
% X          - Training Data
% numOfAtoms - dimensionaity after Transform
% mu         - regularizer for Tranform
% lambda     - regularizer for coefficient

% Output
% T          - learnt Transform
% Z          - learnt sparse coefficients

if nargin < 4
    tau = 0.1;
end
if nargin < 3
    mu = 1;
end

maxIter = 10;
type = 'soft'; % default 'soft'
lambda = 1; % controlling the intermediate variable

rng(1); % repeatable
T = randn(numOfAtoms, size(X,1));

invL = (X*X' + mu*eye(size(X,1)))^(-0.5);

for i = 1:maxIter
    
    % update Coefficient Z
    % sparse 
    Z = sign(T*X).*max(0,abs(T*X)-tau); % soft thresholding
    % dense
    % Z = T*X;
    
    % Update Intermediate Variable
    Y = sign(T*X-Z).*max(0,abs(T*X-Z)-lambda); 

    % update Transform T
    [U,S,V] = svd(invL*X*(Y+Z)');
    D = [diag(diag(S) + (diag(S).^2 + 2*mu/lambda).^0.5) zeros(numOfAtoms, size(X,1)-numOfAtoms)];
    T = 0.5*V*D*U'*invL;
    
end