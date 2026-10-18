function varargout = min(varargin)
if nargin==3 && ischar(varargin{3}) && strcmpi(varargin{3},'all')
  [varargout{1:builtin("max",nargout,1)}] = builtin('min', varargin{1}(:), [], 1);
else
  [varargout{1:builtin("max",nargout,1)}] = builtin('min', varargin{:});
end
end
