function r = randn(varargin)
r = builtin('randn', varargin{:});
rec_push(1, r);
end
