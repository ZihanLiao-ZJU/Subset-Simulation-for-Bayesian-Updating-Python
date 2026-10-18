function r = randsample(pop, k, repl, w)
% weighted sampling with replacement; pop is n or a vector 1:n
if numel(pop) == 1, n = pop; else, n = numel(pop); end
w = w(:) / sum(w(:));
edges = [0; cumsum(w)]; edges(end) = 1;
u = builtin('rand', k, 1);
r = zeros(k,1);
for i = 1:k
  r(i) = find(u(i) < edges, 1, 'first') - 1;
end
if numel(pop) > 1, r = pop(r); end
r = r(:)';
rec_push(5, r);
end
