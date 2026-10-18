function o = makedist(name, varargin)
% Minimal Octave stand-in for MATLAB makedist (struct with function handles)
o = struct('DistributionName',name,'mu',0,'sigma',1,'lower',0,'upper',1,'k',0, ...
           'icdf',[],'cdf',[],'pdf',[]);
for i=1:2:numel(varargin)
  o.(varargin{i}) = varargin{i+1};
end
switch lower(name)
  case 'normal'
    m=o.mu; s=o.sigma;
    o.icdf=@(p) norminv(p,m,s); o.cdf=@(x) normcdf(x,m,s); o.pdf=@(x) normpdf(x,m,s);
  case 'lognormal'
    m=o.mu; s=o.sigma;
    o.icdf=@(p) exp(norminv(p,m,s)); o.cdf=@(x) normcdf(log(x),m,s); o.pdf=@(x) lognpdf(x,m,s);
  case 'uniform'
    a=o.lower; b=o.upper;
    o.icdf=@(p) a+p.*(b-a); o.cdf=@(x) min(max((x-a)./(b-a),0),1); o.pdf=@(x) double(x>=a & x<=b)./(b-a);
  case 'generalizedextremevalue'
    o.DistributionName = 'Generalized Extreme Value';
    k=o.k; s=o.sigma; m=o.mu;
    o.icdf=@(p) gevinv(p,k,s,m); o.cdf=@(x) gevcdf(x,k,s,m); o.pdf=@(x) gevpdf(x,k,s,m);
end
end
