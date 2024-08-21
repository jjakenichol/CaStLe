classdef Burgers_Equation < handle

    
    properties
        model; % PDE object
        x; % x nodes in the mesh
        y; % y nodes in the mesh
    end
    
    methods
        % Input: h is the mesh resolution, smaller h is more accurate, but
        % takes longer to solve the system
        function obj = Burgers_Equation(h, alpha, beta, c, radius)
            
            obj.model = createpde;
            R1 = [1. 0. 0. radius]'; % [shape, center_x, center_y, radius]
            gd = R1;
            ns = char('R1');
            ns = ns';
            sf = 'R1';
            geometryFromEdges(obj.model, decsg(gd, sf, ns));
            generateMesh(obj.model,"Hmin",h,"Hmax",h);

            % This defines the advection operator
            % The PDE is aCoef*u, so for example, 
            % aCoef = @(region,state) state.ux means the PDE is (du/dx)*u
            aCoef = @(region,state) alpha*state.ux + beta*state.uy;
            
            % c is the diffusion coefficient, smaller c makes the problem
            % more advective, taking c too small may cause solver problems
            specifyCoefficients(obj.model,"m",0,"d",1,"c",c,"a",aCoef,"f",0);
            
            % Currently we set f=0, meaning their is no source term, this
            % is another option if we want to add more to the system
            
            obj.x = obj.model.Mesh.Nodes(1,:)';
            obj.y = obj.model.Mesh.Nodes(2,:)';
        end
        
        % Input u0: a matlab function which inputs (x,y) and maps to the
        % initial condition which is defined for (x,y) in [-1,1]x[-1,1]
        % Input t: A matlab vector of time steps for which we want the
        % solution, for instance, t=linspace(0,1,101) gives 101 equally
        % space points in the time interval [0,1]
        % Output: u is a (num_space_nodes x num_time_steps) matrix of the
        % PDE solution, where num_time_steps is the length of t
        function [u] = State_Solve(obj,u0,t)
            % Set initial condtion
            u0_input = @(region) u0(region.x,region.y);
            setInitialConditions(obj.model,u0_input);
            
            % Set solver options
            obj.model.SolverOptions.MaxIterations = 1000;
            obj.model.SolverOptions.ReportStatistics = 'off';
            obj.model.SolverOptions.ResidualTolerance = 1.e-5;
            obj.model.SolverOptions.MinStep = 1.e-9;
            
            % Solve PDE
            result = solvepde(obj.model,t);
            u = result.NodalSolution;
        end
         
        % Plotting function, input u (the output of "State_Solve")
        % and t (an input of "State_Solve")
        function [] = Plot_Field(obj,u,t,min_t,max_t)
            % For circle:
            radius = 1;
            phi360 = (0:360)*pi/180;
            % phi360 = linspace(0,360,100)*pi/180;
            r = linspace(0,radius);
            [Phi360,R] = meshgrid(phi360,r);

            % Loop over time steps
            for k = min_t:max_t
                % For circle:
                F = scatteredInterpolant(obj.x,obj.y,u(:,k));
                f = F(R.*cos(Phi360), R.*sin(Phi360));

                % Plot PDE solution at kth time step
                figure(1)
                surf(R.*cos(Phi360), R.*sin(Phi360),f)
                view(2)
                shading interp
                colorbar()
                pbaspect([1 1 1])
                title(['t = ',num2str(t(k))])
                pause(.1)
            end
        end


        function [out] = Get_Data(obj,u,t,capture_apothem,capture_N)
            xl = linspace(-capture_apothem,capture_apothem,capture_N);
            yl = linspace(-capture_apothem,capture_apothem,capture_N);
            [X, Y] = meshgrid(xl, yl);
            out = zeros(size(X,1), size(Y,1), length(t));

            for k = 1:length(t)
                % Interpolation to access raw data
                F = scatteredInterpolant(obj.x,obj.y,u(:,k));
                f = F(X, Y);
                out(:,:,k) = f;
            end

        end

        function [out] = Save_Data(obj,u,t)

            % Mapping meshes to make the plotting tool happy
            xl = linspace(min(obj.x),max(obj.x),100)';
            yl = linspace(min(obj.y),max(obj.y),100)';
            [X,Y] = meshgrid(xl,yl);

            out = zeros(size(X,1), size(Y,1), length(t));

            % Loop over time steps
            for k = 1:length(t)
                % Interpolation to access raw data
                F = scatteredInterpolant(obj.x,obj.y,u(:,k));
                f = F(X,Y);

                out(:,:,k) = f;
            end

            save("burgersTest.mat","out")
        end
        
        
    end
end

