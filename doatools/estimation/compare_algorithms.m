% Clear and close all
clc; close all; clear;
rng('default');

% Define parameters
M = 5;                    % Number of array elements
N = 100;                  % Number of snapshots
c = 3e8;                  % Speed of light (m/s)
freq_GHz = 0.25;          % Frequency in GHz
th_deg = 30;              % Fixed target angle (degrees)
th_rad = deg2rad(th_deg); % Convert to radians

% SNR range
SNR_dB_vec = -20:2:20;    % SNR range from -20dB to 20dB
numSNR = length(SNR_dB_vec);

% Array configuration
x = [0;0.6;1.275;2.7;5.7];  % Array positions
% x = [0; 0.12; 0.6; 1.275; 2.7] 
% x = [1:5]';
d = diff(x);

% Signal parameters
f = freq_GHz * 1e9;       % Frequency in Hz
lambda = c / f;           % Wavelength in meters
sigma_n = 1;              % Noise power

% Initialize arrays for storing results
music_rmse = zeros(numSNR, 1);
interf_weighted_rmse = zeros(numSNR, 1);
interf_unweighted_rmse = zeros(numSNR, 1);
num_trials = 100;         % Number of Monte Carlo trials

% Monte Carlo simulation
for idxSNR = 1:numSNR
    SNR_dB = SNR_dB_vec(idxSNR);
    sigma_s = 10^(SNR_dB/10) * sigma_n;  % Signal power
    
    % Initialize arrays for storing estimates
    music_estimates = zeros(num_trials, 1);
    interf_weighted_estimates = zeros(num_trials, 1);
    interf_unweighted_estimates = zeros(num_trials, 1);
    
    for trial = 1:num_trials
        % Generate array manifold
        a = exp(-1j * 2 * pi * x / lambda * sin(th_rad));
        
        % Generate signal
        s = sqrt(sigma_s/2) * (randn(1, N) + 1j * randn(1, N));
        
        % Generate noise
        n = sqrt(sigma_n/2) * (randn(M, N) + 1j * randn(M, N));
        
        % Generate received signal
        X = a * s + n;
        
        % MUSIC algorithm
        R = X * X' / N;  % Sample covariance matrix
        [V, D] = eig(R);
        [D, idx] = sort(real(diag(D)), 'descend');
        V = V(:, idx);
        
        % Use all but the largest eigenvalue for noise subspace
        Un = V(:, 2:end);
        
        % MUSIC spectrum
        search_angles = deg2rad(-60:0.1:60);
        P_music = zeros(size(search_angles));
        for i = 1:length(search_angles)
            a_search = exp(-1j * 2 * pi * x / lambda * sin(search_angles(i)));
            P_music(i) = 1 / (a_search' * (Un * Un') * a_search);
        end
        [~, max_idx] = max(abs(P_music));
        music_estimates(trial) = rad2deg(search_angles(max_idx));

        % Interferometer algorithm (with weighted least squares)
        interf_weighted_estimates(trial) = interferometer_doa(X, x, lambda, true);
        
        % Interferometer algorithm (without weighted least squares)
        interf_unweighted_estimates(trial) = interferometer_doa(X, x, lambda, false);
    end
    
    % Calculate RMSE
    music_rmse(idxSNR) = sqrt(mean((music_estimates - th_deg).^2));
    interf_weighted_rmse(idxSNR) = sqrt(mean((interf_weighted_estimates - th_deg).^2));
    interf_unweighted_rmse(idxSNR) = sqrt(mean((interf_unweighted_estimates - th_deg).^2));
end

% Plot results
figure;
semilogy(SNR_dB_vec, music_rmse, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'MUSIC');
hold on;
semilogy(SNR_dB_vec, interf_weighted_rmse, 'r-s', 'LineWidth', 1.5, 'DisplayName', 'Interferometer (加权)');
semilogy(SNR_dB_vec, interf_unweighted_rmse, 'g-d', 'LineWidth', 1.5, 'DisplayName', 'Interferometer (非加权)');
xlabel('SNR (dB)');
ylabel('RMSE (度)');
title(sprintf('算法性能比较 (目标角度 = %d°)', th_deg));
legend('show', 'Location', 'best');
grid on;
hold off;

% Calculate average improvement
avg_improvement_weighted = mean(interf_weighted_rmse ./ music_rmse);
avg_improvement_unweighted = mean(interf_unweighted_rmse ./ music_rmse);
fprintf('MUSIC相对于加权干涉仪的平均性能提升: %.2f倍\n', avg_improvement_weighted);
fprintf('MUSIC相对于非加权干涉仪的平均性能提升: %.2f倍\n', avg_improvement_unweighted); 