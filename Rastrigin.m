function y = Rastrigin (X)
% Rastrigin's Function
A = 10;
n = length(X);
m = 0;
for i = 1:n
    m = m + X(i)^2 - A*cos(2*pi*X(i));
end
y = 10*n + m;
end