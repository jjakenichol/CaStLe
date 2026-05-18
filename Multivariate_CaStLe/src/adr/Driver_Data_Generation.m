clear;
close all;
clc;
%run('../../src/Set_Paths');

% Beforehand: use pdeModeler to generate geometry and mesh.
% save('urban_canyon.mat', 'points', 'edges', 'triangles');

control_nodes = [
                 0.1000    0.5000
                 0.1000    0.9000
                 0.1000    1.1000
                 0.3000    0.7000
                 0.3000    0.9000
                 0.3000    1.1000
                 0.5000    0.3000
                 0.5000    0.5000
                 0.5000    0.7000
                 0.7000    0.7000
                 0.9000    0.3000
                 0.9000    1.1000
                 1.1000    0.7000
                 1.1000    0.9000
                ]';

%% Initialize the solver.
% mesh_shape = "square";
% init_plume_center = [-0.; 0.];
% plume_size = 10;
% diff_coeffs = [.004; .004];
% advection_coeffs = [1; 1];
% react_rate = 10;
% reaction_scaling = 1.0;
% velocity_field_type = "constant";
% alpha = 2.5;
% beta = -5.0;
% velocity_parameters = [alpha, beta];
% H = 0.05;
% radius = 1.0;
% t = linspace(0, .4, 51);
mesh_shape = "circle";
init_plume_center = [0.0; 0.0];
plume_size = 50;
diff_coeffs = [0.05; 0.05];
advection_coeffs = [5.0; 5.0];
react_rate = 1;
reaction_scaling = 1.0;
velocity_field_type = "sinusoidal";
alpha = 0.0;
beta = 0.0;
velocity_parameters = [alpha, beta];
H = 0.02;
radius = 3.0;
capture_apothem = 1.0;
t = linspace(0.0, 0.4, 51);

model = Transient_ADR_2D.model_default(H, radius, mesh_shape);
solver = Transient_ADR_2D(model, init_plume_center, plume_size, diff_coeffs, advection_coeffs, velocity_field_type, velocity_parameters, react_rate, reaction_scaling, control_nodes);

% init_plume_center = [.05; .85];
% diff_coeffs = [.1; .1];
% advection_coeffs = [4; 4];
% react_rate = 2;
% model = Transient_ADR_2D.model_fromfile('urban_canyon.mat');
% solver = Transient_ADR_2D(model, init_plume_center, diff_coeffs, vel_coeffs, react_rate, control_nodes);

% t = linspace(0, .4, 401);

%% Visualize the solver geometry.
solver.Plot_Control_Nodes();
solver.Plot_Velocity_Field();

%% Initialize the controller and visualize the controls.
% controller = randomspline(t, solver.n_q, 8); % Option to add a forcing
controller = nocontrol(solver.n_q);

figure;
plot(t, controller(t));
title('Controller');

%% Solve with the selected controller and visualize the results.
tic();
u = solver.State_Solve(controller, t);
solve_time = toc();

u_nodal = u.NodalSolution;

solver.Plot_Field(u.NodalSolution(:, :, 1), 'Initial condition');
solver.Animate_Solution(u.NodalSolution);

%% Save the results.
save('solver.mat', 'solver');
save('solution.mat', 'u');

% % For later: load and animate the results again.
% load('solver.mat', 'solver');
% load('solution.mat', 'u');
% solver.Animate_Solution(u.NodalSolution)

%% Controllers
function [out] = randomspline(t, dofs, num_nodes)
    % Random (but smooth) nonnegative inputs.
    nodes = linspace(min(t), max(t), num_nodes);
    vals = 50 * rand(dofs, num_nodes) + 5;
    pp = pchip(nodes, vals);
    out = @(tt) ppval(pp, tt);
end

function [out] = nocontrol(dofs)
    % Zero inputs: q(t) = 0 for all t/
    out = @(t) zeros(dofs, length(t));
end
