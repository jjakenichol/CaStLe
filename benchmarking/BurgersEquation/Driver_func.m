function out = Driver_func(init_diam, x_pos, y_pos, alpha, beta, c, radius, capture_apothem, capture_N, plot, verbose)
    warn_id = 'MATLAB:scatteredInterpolant:DupPtsAvValuesWarnId';
    warning('off',warn_id)

    h = .1; % This parameter determines the mesh resolution, small h means more accurate and more costly to solve
    pde = Burgers_Equation(h, alpha, beta, c, radius); % Instantiate the PDE class

    t = linspace(0,1,101)'; % Define the time mesh to solve on

    % Set the initial condition as a function of (x,y)
    u0 = @(x,y) 10*exp(init_diam*((x+x_pos).^2 + (y+y_pos).^2));

    u = pde.State_Solve(u0,t); % Solve the PDE

    if verbose == 1
       disp('Solution computed, getting data.') 
    end
    
    out = pde.Get_Data(u,t,capture_apothem,capture_N); % Get data to return
    % out = pde.Save_Data(u,t); % save the solution
    if plot == 1
        pde.Plot_Field(u,t,1,101); % Plot the solution
    end

end