% This function serves as the entry point for the NMF algorithm.
% If no arguments are provided, it calls the nmf_for_stability() function.
% If arguments are provided, it calls the work() function.
function nmf(varargin)
    if (nargin == 0)
        nmf_for_stability();
    else
        mkdir(varargin{3})
        work(varargin{1}, varargin{2}, sprintf('%s/area_result.mat',varargin{3}));
    end
end
    
function nmf_for_stability()
    for k = 20:2:22
        path = "mental_health_result\\cross_sectional\\adhd\\";
        %mkdir(sprintf("%s\\stability_results\\k%d",path, k));
        for metric = ["thk", "vol"]
            mkdir(sprintf("%s\\stability_results\\%s\\k%d",path, metric, k));
            parfor i = 0:9
                filename = sprintf('%s\\stability_splits\\%s_s_a_%d.mat', path, metric, i);
                output = sprintf('%s\\stability_results\\%s\\k%d\\a_%d.mat', path, metric, k, i);
                work(filename, k, output);
                filename = sprintf('%s\\stability_splits\\%s_s_b_%d.mat', path, metric, i);
                output = sprintf('%s\\stability_results\\%s\\k%d\\b_%d.mat', path, metric, k, i);
                work(filename, k, output);
            end
        end
    end
end
function work(filename, k, output)
    addpath('cobra_brainparts');

    % Create an input parser object
    p = inputParser;

    % Define the input parameters
    addRequired(p, 'filename', @ischar);
    addRequired(p, 'k', @isnumeric);
    addRequired(p, 'output', @ischar);

    % Parse the input parameters
    parse(p, filename, k, output);

    % Load the input file
    load(p.Results.filename);

    % Display the size of X
    disp(size(X));

    % Run pNMF
    [W,H,recon] = opnmf_mem_cobra(X, p.Results.k, [], 4,50000,[],[],100,[]);

    % Display the size of W
    disp(size(W));

    % Save the output
    save(p.Results.output, 'W', 'H', 'recon', '-v7');
end
