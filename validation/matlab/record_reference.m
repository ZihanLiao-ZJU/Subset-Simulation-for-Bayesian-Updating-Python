function record_reference(matlab_repo, case_name, Ndim, seed, outdir)
% Run the ORIGINAL MATLAB code once and record every random number it draws.
%
%   record_reference('/path/to/Subset-Simulation-for-Bayesian-Updating', 'shells', 2, 1, '../reference')
%
% case_name : 'shells' | 'loggamma' | 'eggbox'
% The functions randn/rand/randperm/randi/randsample in this folder shadow the
% built-in ones and log every draw; rec_*.mat holds the stream and res_*.mat
% the results. validate.py replays the stream through the Python port.
here = fileparts(mfilename('fullpath'));
addpath(genpath(matlab_repo));
addpath(here);                         % shims first on the path
if exist('OCTAVE_VERSION', 'builtin')
    addpath(fullfile(here, 'octave'));  % Octave compatibility shims (makedist, pagemtimes, ...)
    pkg load statistics
    builtin('rand', 'seed', seed); builtin('randn', 'seed', seed);
else
    rng(seed);
end
rec_reset();
switch case_name
    case 'shells',   Tfun = LKF_nD_GaussianShells(Ndim);
    case 'loggamma', Tfun = LKF_nD_GaussianLogGamma(Ndim);
    case 'eggbox',   Tfun = LKF_2D_Eggbox();
end
IntBay = SuS_Bay_Nataf(Tfun); sampler = aCS(IntBay); ss = main(sampler); ss.NumSam = 1000;
out_ss = ss.RunIte;
out_ps = NumSim(out_ss);
[x_pos, ~] = out_ps.PosSim;
[c_z_hat, c_z] = out_ps.EvlVar;
Z = out_ps.Z; Nite = out_ss.Nite; Ncal = out_ss.Ncal; G = cell2mat(out_ss.g); %#ok<NASGU>
tag = sprintf('%s_%d_%d', case_name, Ndim, seed);
rec_save(fullfile(outdir, ['rec_' tag '.mat']));
save(fullfile(outdir, ['res_' tag '.mat']), '-v7', 'Z', 'Nite', 'Ncal', 'G', 'x_pos', 'c_z_hat', 'c_z');
end
