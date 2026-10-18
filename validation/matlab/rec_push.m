function rec_push(code, v)
global REC_K REC_V
REC_K(end+1) = code;
REC_V{end+1} = v(:);
end
