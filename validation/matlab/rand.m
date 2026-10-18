function r = rand(varargin)
r = builtin('rand', varargin{:});
rec_push(2, r);
end
