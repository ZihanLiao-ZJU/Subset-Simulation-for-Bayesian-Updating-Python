function C = pagemtimes(A,B)
sB = size(B);
C = reshape(A*reshape(B,sB(1),[]), [size(A,1), sB(2:end)]);
end
