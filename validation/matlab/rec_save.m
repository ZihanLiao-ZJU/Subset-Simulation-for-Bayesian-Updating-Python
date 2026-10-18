function rec_save(fname)
global REC_K REC_V
kinds = REC_K(:); lens = cellfun(@numel, REC_V(:)); vals = vertcat(REC_V{:});
save('-v7', fname, 'kinds', 'lens', 'vals');
end
