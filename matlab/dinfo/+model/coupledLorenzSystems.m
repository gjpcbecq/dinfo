function [x, y, z] = coupledLorenzSystems(nObs, K, tau, b, sigma, r, ...
    tauMax)
% Simulate samples from three coupled Lorenz systems used in Frenzel and Pompe 2007
%
% Syntax
% 
% [x, y, z] = coupledLorenzSystems(nObs, K, tau, b, sigma, r, tauMax)
% 
% Input
%
% nObs: int
% K: 3-by-3, coupling matrix between systems
% tau: 3-by-3, delay betwwen systems
% b=8/3: float
% sigma=10: float
% r=28: float
% tauMax=60: int
%
% Output
% 
% x: 1-by-nObs
% y: 1-by-nObs
% z: 1-by-nObs
%
% Description
% 
% $$
% \left \{
% \begin{array}{l}
% \dot{X}_i(t) = \sigma \, (Y_i(t) - X_i(t)) \\
% \dot{Y}_i(t) = r \, X_i(t) - Y_i(t) - X_i(t) \, Z_i(t) + 
%     \sum_{j \neq i} K_{ij} \, Y_j^2(t - \tau_{ij})\\
% \dot{Z}_i(t) = X_i(t) \, Y_i(t) - b \, Z_i(t) 
% \end{array}
% \right . 
% $$
% Euler method for integration in 2 steps with integration steps of 0.006. 
% Runge-Kutta method like in the article is not used here for simplicity. 
% All data are returned with a sampling period of 0.003. 
% For getting a sampling period of $\Delta t =0.3$ like in the artcile, use 
% (X, Y, Z) = (x[:, ::100], y[:, ::100], z[:, ::100]) 
% Initialisation is realized by taking uniform values in [0, 0.01]    
%
% Example 
%
% nObs = 10000;
% K = zeros(3, 3);
% K(1, 2) = 0.5;
% K(2, 3) = 0.5;
% tau = zeros(3, 3);
% tau(1, 2) = 1000;
% tau(2, 3) = 1500;
% [x, y, z] = model.coupledLorenzSystems(nObs, K, tau, 8/3, 10, 28, 2000); 
% X = x(:, 1:100:end); 
% Y = y(:, 1:100:end); 
% Z = z(:, 1:100:end); 
% tau = 10;
% Y1 = Y(1, tau:end);
% Y2 = Y(2, 1:end-tau+1);
% miY1Y2 = dinfo.mi(Y1, Y2, 'Frenzel', {3, 'Euclidean'});
% disp([tau, miY1Y2]);
% 
% 10.0000    1.2854
% 
ampCondInit = 1; 
x = ampCondInit * rand(3, nObs + tauMax);
y = ampCondInit * rand(3, nObs + tauMax);
z = ampCondInit * rand(3, nObs + tauMax);
Ts = 0.003; 
Ts2 = Ts / 2;  
for t = 1 + tauMax : nObs + tauMax, 
    tm1 = t - 1;
    for i = 1 : 3, 
        sK = 0; 
        for j = 1 : 3,  
            yd = y(j, t - tau(i, j)); 
            sK = sK + K(i, j) * yd * yd; 
        end
        % double Euler integration for stability 
        % TODO use Runge-Kutta method 
        xp = sigma * (y(i, tm1) - x(i, tm1));  
        yp = x(i, tm1) * (r - z(i, tm1)) - y(i, tm1) + sK; 
        zp = x(i, tm1) * y(i, tm1) - b * z(i, tm1); 
        xt = x(i, tm1) + Ts2 * xp; 
        yt = y(i, tm1) + Ts2 * yp; 
        zt = z(i, tm1) + Ts2 * zp; 
        xp = sigma * (yt - xt); 
        yp = xt * (r - zt) - yt + sK; 
        zp = xt * yt - b * zt; 
        x(i, t) = xt + Ts2 * xp; 
        y(i, t) = yt + Ts2 * yp; 
        z(i, t) = zt + Ts2 * zp; 
    end
end
x = x(:, 1 + tauMax : nObs + tauMax); 
y = y(:, 1 + tauMax : nObs + tauMax); 
z = z(:, 1 + tauMax : nObs + tauMax); 
return 