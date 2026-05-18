function solution = ADR_Driver_func(mesh_shape, init_center, plume_size, diff_coeffs, advection_coeffs, velocity_field_type, velocity_parameters, react_rate, reaction_scaling, init_concentration, t, H, radius, capture_apothem, capture_N, parallel_interpolation, plot, verbose)
    % Driver function to solve the two-species transient advection-diffusion-reaction problem.
    %
    % Parameters
    % ----------
    % mesh_shape : str
    %   Shape of mesh ("square" or "circle").
    % init_center : (2 x 1) float
    %   Center of the initial condition blob.
    % plume_size : float
    %   The size of the initial plume
    % diff_coeffs : (2 x 1) float
    %   Diffusion coefficients. Larger means more diffusion.
    % advection_coeffs : (2 x 1) float
    %   Coefficients that determine the strength and direction of advection in the system.
    %   Larger values indicate stronger advection effects.
    % velocity_field_type : str
    %   Specifies the type of velocity field used in the simulation.
    %   Supported types include "constant" and "sinusoidal".
    % velocity_parameters : (2 x 1) float
    %   Parameters related to the velocity field.
    %   For a constant velocity field, this might include the velocity components in the x and y directions.
    % react_rate : float
    %   Reaction coefficient. Larger means more reaction.
    % reaction_scaling : float
    %   Scaling factor for amount of a species becomes the other.
    % init_concentration : float
    %   Initial concentration magnitude (default = 50).
    % t : (1 x n) float
    %   Time points at which to solve the PDE.
    % H : float
    %   Maximum and minimum spacing in the spatial mesh. Smaller Hmax means a finer mesh.
    % radius : float
    %   Radius of the circular domain (only used if mesh_shape is "circle").
    % capture_apothem : float
    %   Half the length of the side of the square region to capture the solution.
    % capture_N : int
    %   Number of points in each dimension of the captured solution grid.
    % parallel_interpolation : bool
    %   Whether to compute the interpolation in parallel.
    % plot : logical
    %   Whether to plot the solution.
    % verbose : logical
    %   Whether to display verbose output.
    %
    % Returns
    % -------
    % solution : (capture_N x capture_N x length(t) x 2) float
    %   Solution array representing the species concentrations over time and space.

    % Suppress specific warnings
    warn_id = 'MATLAB:scatteredInterpolant:DupPtsAvValuesWarnId';
    warning('off', warn_id);

    % Control nodes need to be passed but don't have an effect on the
    % behavior of the model. These can be set to a default and passed to
    % Transient_ADR_2D
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

    % Initialize the PDE model with a default geometry and mesh.
    if verbose
        disp('MATLAB: Initializing the PDE model...');
    end
    model = Transient_ADR_2D.model_default(H, radius, mesh_shape);
    solver = Transient_ADR_2D(model, init_center, plume_size, diff_coeffs, advection_coeffs, velocity_field_type, velocity_parameters, react_rate, reaction_scaling, control_nodes, init_concentration);

    % Initialize a controller with no control inputs.
    controller = nocontrol(solver.n_q);

    % Solve the PDE with the selected controller.
    if verbose
        disp('MATLAB: Solving the PDE...');
    end
    u = solver.State_Solve(controller, t);

    % Extract the nodal solution from the solver.
    if verbose
        disp('MATLAB: Extracting nodal solution...');
    end
    u_nodal = u.NodalSolution;

    % Capture the solution in the specified region.
    if verbose
        disp('MATLAB: Capturing solution region...');
    end
    xl = linspace(-capture_apothem, capture_apothem, capture_N);
    yl = linspace(-capture_apothem, capture_apothem, capture_N);
    [X, Y] = meshgrid(xl, yl);
    solution = zeros(capture_N, capture_N, length(t), 2);


    if verbose
        disp('MATLAB: Interpolating raw data...');
    end

    % Temporary arrays to store the results of the parfor loop
    temp_solution_1 = zeros(capture_N, capture_N, length(t));
    temp_solution_2 = zeros(capture_N, capture_N, length(t));

    % Use parallel computing to speed up the interpolation process if requested
    if parallel_interpolation
        parfor k = 1:length(t)
            try
                % Interpolation to access raw data
                F1 = scatteredInterpolant(solver.x, solver.y, u_nodal(:, 1, k));
                F2 = scatteredInterpolant(solver.x, solver.y, u_nodal(:, 2, k));
                temp_solution_1(:, :, k) = F1(X, Y);
                temp_solution_2(:, :, k) = F2(X, Y);
            catch ME
                disp(['Error in parfor loop at iteration ', num2str(k), ': ', ME.message]);
            end
        end
    else
        for k = 1:length(t)
            try
                % Interpolation to access raw data
                F1 = scatteredInterpolant(solver.x, solver.y, u_nodal(:, 1, k));
                F2 = scatteredInterpolant(solver.x, solver.y, u_nodal(:, 2, k));
                temp_solution_1(:, :, k) = F1(X, Y);
                temp_solution_2(:, :, k) = F2(X, Y);
            catch ME
                disp(['Error in for loop at iteration ', num2str(k), ': ', ME.message]);
            end
        end
    end

    % Assign temporary arrays to the solution array
    solution(:, :, :, 1) = temp_solution_1;
    solution(:, :, :, 2) = temp_solution_2;

    % Plot the solution if requested.
    if plot
        solver.Plot_Field(u_nodal(:, :, 1), 'Initial condition', false, false);
        solver.Animate_Solution(u_nodal);
    end

    if verbose
        disp('MATLAB: Solution computed and data captured.');
    end
end

function [out] = nocontrol(dofs)
    % Generate zero inputs: q(t) = 0 for all t.
    %
    % Parameters
    % ----------
    % dofs : int
    %   Number of degrees of freedom (control nodes).
    %
    % Returns
    % -------
    % out : function handle
    %   Function handle that returns zero control inputs at given time points.
    out = @(t) zeros(dofs, length(t));
end