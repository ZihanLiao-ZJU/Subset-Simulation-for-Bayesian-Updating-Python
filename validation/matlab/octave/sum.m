function y = sum(varargin)
if nargin>=2 && ischar(varargin{2}) && strcmpi(varargin{2},'all')
  y = builtin('sum', varargin{1}(:), 1);
else
  y = builtin('sum', varargin{:});
end
end
