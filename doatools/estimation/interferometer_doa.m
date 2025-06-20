function angle_est = interferometer_doa(X, x, lambda, use_weighted)
% INTERFEROMETER_DOA 使用干涉仪算法进行DOA估计
%   X: 接收信号矩阵 (M x N)，M为阵元数，N为快拍数
%   x: 阵元位置向量 (M x 1)
%   lambda: 信号波长
%   use_weighted: 是否使用加权最小二乘估计 (可选，默认为true)
%   返回: 估计的DOA角度（度）

% 参数检查
if nargin < 4
    use_weighted = true;  % 默认使用加权最小二乘
end

M = size(X, 1);  % 阵元数
d = diff(x);     % 阵元间距

% 1. 计算相邻阵元间的相位差
phase_diff = zeros(M-1, 1);
for i = 1:M-1
    R = mean(X(i,:) .* conj(X(i+1,:)));  % 复相关系数
    phase_diff(i) = angle(R);  % 平均相位差 [-π, π]
end

% 2. 分层解模糊策略 (从最小间距开始)
unwrapped_phase = phase_diff;
if M > 2
    % 按间距从小到大排序
    [sorted_d, sort_idx] = sort(d);
    sorted_phase = phase_diff(sort_idx);
    
    % 最小间距对作为基准
    base_sin_theta = lambda * sorted_phase(1) / (2 * pi * sorted_d(1));
    base_sin_theta = min(max(base_sin_theta, -1), 1);  % 限制范围
    
    % 用基准解其他阵元对的模糊
    for k = 2:length(sorted_d)
        % 预测相位差 (基于基准角度)
        pred_phase = 2 * pi * sorted_d(k) * base_sin_theta / lambda;
        
        % 计算相位差与预测值的偏差
        phase_error = sorted_phase(k) - pred_phase;
        
        % 解模糊 (找到最接近的2π整数倍)
        n = round(phase_error / (2*pi));
        unwrapped_phase(sort_idx(k)) = sorted_phase(k) - 2*pi*n;
        
        % 更新基准 (加权平均)
        current_sin_theta = lambda * unwrapped_phase(sort_idx(k)) / (2 * pi * sorted_d(k));
        base_sin_theta = (base_sin_theta + current_sin_theta) / 2;
        base_sin_theta = min(max(base_sin_theta, -1), 1);
    end
end

% 3. 最小二乘估计
A_matrix = d;
b_vector = unwrapped_phase * lambda / (2*pi);

if use_weighted
    % 加权最小二乘估计
    weights = d.^2;  % 克拉美罗界最优加权 (间距平方)
    sin_theta = (A_matrix' * diag(weights) * A_matrix) \ (A_matrix' * diag(weights) * b_vector);
else
    % 普通最小二乘估计
    sin_theta = (A_matrix' * A_matrix) \ (A_matrix' * b_vector);
end

% 4. 计算角度估计
sin_theta = min(max(sin_theta, -1), 1);  % 确保在有效范围内
angle_est = asind(sin_theta);
end 