function r = randi(n, a, b)
r = floor(builtin('rand', a, b) * n) + 1;
rec_push(4, r);
end
