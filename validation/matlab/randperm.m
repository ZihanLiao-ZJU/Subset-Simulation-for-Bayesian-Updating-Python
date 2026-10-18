function r = randperm(n, k)
if nargin < 2, k = n; end
r = builtin('randperm', n);
r = r(1:k);
rec_push(3, r);
end
