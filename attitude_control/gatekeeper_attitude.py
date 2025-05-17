import numpy as np
from .simple_attitude import SimpleAtt
from .velocity_tracking_yaw import VelocityTrackingYaw
from .visibility_promoting_yaw import VisibilityAtt
from shapely.geometry import LineString, Polygon

"""
Created on April 21th, 2025
@author: Taekyung Kim

@description: 
This code calls gatekeeper submodule and implements a gatekeeper-based attitude controller.
The intended use case is nominal: visibility promoting, backup: velocity tracking yaw, but not limited to it.


"""

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

class GatekeeperAtt:
    """
    Attitude (yaw) controller that wraps the Gatekeeper safety filter,
    selecting between a nominal and a backup yaw policy.
    """
    def __init__(self,
                 robot,
                 robot_spec: dict,
                 dt: float = 0.05,
                 ctrl_config: dict = {},
                 nominal_horizon: float = 1.0,
                 backup_horizon: float = 2.0,
                 event_offset: float = 1.0):
        """
        Parameters
        ----------
        robot : object
            Robot instance (provides state X, dynamics, etc.).
        robot_spec : dict
            Robot specifications
        ctrl_config : dict
            {'nominal': <'simple'|'velocity tracking yaw'>,
             'backup': <'simple'|'velocity tracking yaw'>}
        dt : float
            Integration timestep for Gatekeeper.
        nominal_horizon : float
        backup_horizon : float
        event_offset : float
        """
        self.robot = robot
        self.robot_spec = robot_spec
        self.dt = dt
        self.nominal_horizon = nominal_horizon
        self.backup_horizon = backup_horizon
        self.event_offset = event_offset
        self.horizon_discount = dt * 5

        self.next_event_time = 0.0
        self.current_time_idx = backup_horizon / dt # start from backup trajectory. If the initial canddidate traj is valid, it will be updated to 0
        self.candidate_time_idx = 0
        self.committed_horizon = 0.0

        # Initialize the committed trajectory
        self.committed_x_traj = None
        self.committed_u_traj = None
        self.pos_committed_x_traj = None # be updated using pos_controller instance
        self.pos_committed_u_traj = None

        # Map user keys to controller classes
        self._ctrl_map = {
            'simple': SimpleAtt,
            'velocity tracking yaw': VelocityTrackingYaw,
            'visibility promoting yaw': VisibilityAtt
        }

        nom_key = ctrl_config.get('nominal', 'visibility promoting yaw')
        if nom_key not in self._ctrl_map:
            raise ValueError(f"Unknown nominal controller '{nom_key}'")
        NominalCtrl = self._ctrl_map[nom_key]
        self.nominal_ctrl = NominalCtrl(robot, robot_spec)
        self._set_nominal_controller(self.nominal_ctrl.solve_control_problem)

        backup_key = ctrl_config.get('backup', 'velocity tracking yaw')
        if backup_key not in self._ctrl_map:
            raise ValueError(f"Unknown backup controller '{backup_key}'")
        BackupCtrl = self._ctrl_map[backup_key]
        self.backup_ctrl = BackupCtrl(robot, robot_spec)
        self._set_backup_controller(self.backup_ctrl.solve_control_problem)

    def _set_nominal_controller(self, nominal_controller):
        self.nominal_controller = nominal_controller

    def _set_backup_controller(self, backup_controller):
        self.backup_controller = backup_controller

    def setup_pos_controller(self, pos_controller):
        self.pos_controller = pos_controller

    def _dynamics(self, x, u):
        x_col = np.array(x)
        u_col = np.array(u)
        # f and g live under self.robot.robot
        dx = self.robot.robot.f(x_col) + self.robot.robot.g(x_col) @ u_col
        return dx
    
    def _update_pos_committed_trajectory(self):
        """
        Update the positional committed trajectory.
        """

        x_traj_casadi = self.pos_controller.mpc.opt_x_num['_x', :, 0, 0]
        x_traj_casadi = x_traj_casadi[:-1] # remove the last step to make it same length as u_traj_casadi
        u_traj_casadi = self.pos_controller.mpc.opt_x_num['_u', :, 0]
        
        # Convert to numpy arrays - initialization
        n_steps = len(x_traj_casadi)
        state_dim = x_traj_casadi[0].shape[0] if hasattr(x_traj_casadi[0], 'shape') else 1
        pos_x_traj = np.zeros((n_steps, state_dim))
        
        n_controls = len(u_traj_casadi)
        control_dim = u_traj_casadi[0].shape[0] if hasattr(u_traj_casadi[0], 'shape') else 1
        pos_u_traj = np.zeros((n_controls, control_dim))
        
        # Convert each state in the trajectory
        for i, x_dm in enumerate(x_traj_casadi):
            pos_x_traj[i, :] = np.array(x_dm.full()).flatten()
        
        for i, u_dm in enumerate(u_traj_casadi):
            pos_u_traj[i, :] = np.array(u_dm.full()).flatten()

        # Update the class attributes
        self.pos_committed_x_traj = pos_x_traj
        self.pos_committed_u_traj = pos_u_traj

        # --- now extend if too short ---
        required_steps = int((self.nominal_horizon + self.backup_horizon) / self.dt) + 1
        curr_len = self.pos_committed_x_traj.shape[0]
        if curr_len < required_steps:
            missing = required_steps - curr_len

            x_ext = np.zeros((missing, state_dim))
            u_ext = np.zeros((missing, control_dim))

            last_x = self.pos_committed_x_traj[-1].reshape(-1, 1)
            last_u = self.pos_committed_u_traj[-1].reshape(-1, 1)
            last_u = np.zeros_like(last_u) # TODO: maintain zero control input

            for k in range(missing):
                dx = self._dynamics(last_x, last_u)
                last_x = last_x + dx * self.dt

                x_ext[k, :] = last_x.flatten()
                u_ext[k, :] = last_u.flatten()

            # Concatenate in NumPy
            self.pos_committed_x_traj = np.concatenate(
                [self.pos_committed_x_traj, x_ext], axis=0
            )
            self.pos_committed_u_traj = np.concatenate(
                [self.pos_committed_u_traj, u_ext], axis=0
            )

    def _generate_trajectory(self, initial_yaw, horizon, controller):
        """
        Generate a backup trajectory that tracks velocity direction from the
        committed positional trajectory.
        
        Parameters:
        -----------
        initial_yaw : float
            Initial yaw angle to start the backup trajectory from
        horizon : float
            Time horizon for the backup trajectory
        controller : callable
            Function to compute the control input based on the current state
        """
        n_steps = int(horizon / self.dt) + 1
        x_traj = np.zeros(n_steps)
        u_traj = np.zeros(n_steps)
        
        current_yaw = initial_yaw
        x_traj[0] = current_yaw
        
        # Propagate yaw dynamics using defined controller
        for i in range(1, n_steps):
            # Get corresponding positional state and control
            pos_idx = self.candidate_time_idx + i
            if pos_idx < len(self.pos_committed_x_traj)-1 :
                pos_x = self.pos_committed_x_traj[pos_idx]
                pos_u = self.pos_committed_u_traj[pos_idx]
            else:
                # Use last available state/control if beyond pos trajectory
                pos_x = self.pos_committed_x_traj[-1]
                #pos_u = self.pos_committed_u_traj[-1]
                pos_u = np.zeros_like(self.pos_committed_u_traj[-1]) # TODO: maintain zero control input

            # Compute yaw control input
            u = controller(pos_x.reshape(-1, 1), current_yaw, pos_u.reshape(-1, 1))
            u = u.item()  # Extract scalar from 1x1 array
            u_traj[i-1] = u
            
            # Update yaw using simple Euler integration
            current_yaw = current_yaw + u * self.dt
            x_traj[i] = current_yaw
        
        return x_traj, u_traj

    def _generate_candidate_trajectory(self, discounted_nominal_horizon):

        # Generate the candidate trajectory using the nominal and backup controllers
        self.candidate_time_idx = 0
        current_yaw = self.robot.get_orientation()
        nominal_x_traj, nominal_u_traj = self._generate_trajectory(current_yaw, discounted_nominal_horizon, self.nominal_controller)

        self.candidate_time_idx = len(nominal_x_traj) - 1
        yaw_at_backup = nominal_x_traj[-1]  # last yaw of the nominal trajectory
        backup_x_traj, backup_u_traj = self._generate_trajectory(yaw_at_backup, self.backup_horizon, self.backup_controller)

        self.candidate_x_traj = np.hstack((nominal_x_traj, backup_x_traj))
        self.candidate_u_traj = np.hstack((nominal_u_traj, backup_u_traj))
        return self.candidate_x_traj
    
    def _is_candidate_valid(self, critical_point, candidate_x_traj, discounted_nominal_horizon):
        """
        Check if the candidate trajectory is valid by evaluating the safety condition.
        The trajectory is valid if the critical point becomes visible before the robot reaches it.
        
        Parameters
        ----------
        candidate_x_traj : numpy.ndarray
            The candidate yaw trajectory to evaluate
            
        Returns
        -------
        bool
            True if the trajectory is valid (critical point becomes visible in time), False otherwise
        """
        if critical_point is None:
            return True  # No critical point found, trajectory is valid
            
        # Check if critical point becomes visible before robot reaches it
        for i in range(len(candidate_x_traj)):
            # Get position and yaw at this timestep
            
            pos = self.pos_committed_x_traj[i]
            yaw = candidate_x_traj[i]
            
            # Check if critical point is in FOV at this state
            # Consider the discounted nominal horizon, only check during the backup phase
            if i > discounted_nominal_horizon/self.dt:
                if self._is_point_in_fov(pos, yaw, critical_point, is_in_cam_range=True):
                    # print("In FOV at time step", i, "at position", pos, "with yaw", yaw)
                    # print("critical point", critical_point)
                    return True
                    
            # Check if we've reached the stopping point before critical point
            if self.robot_spec['model'] == 'DoubleIntegrator2D':
                # Get current velocity
                # vx = pos[2]
                # vy = pos[3]
                # v = np.sqrt(vx**2 + vy**2)
                v = self.robot.robot_spec['v_max'] # TODO:
                
                # Calculate maximum braking distance
                a_max = self.robot_spec['a_max']
                braking_distance = v**2 / (2 * a_max)
                
                # Calculate distance to critical point
                dist_to_critical = np.linalg.norm(pos[:2] - critical_point)
                
                # If we're within braking distance of critical point, trajectory is invalid
                if dist_to_critical <= braking_distance:
                    return False
            elif self.robot_spec['model'] == 'SingleIntegrator2D':
                # If we've passed the critical point, trajectory is invalid
                if np.linalg.norm(pos[:2] - critical_point) < self.robot.robot_radius:
                    return False
            else:
                raise ValueError(f"Not implemented for robot model: {self.robot_spec['model']}")
                
        return False  # Critical point never became visible
        
    def _is_point_in_fov(self, robot_state, robot_yaw, point, is_in_cam_range=False):
        """
        Check if a point is within the robot's field of view at a given state and yaw.
        
        Parameters
        ----------
        robot_state : numpy.ndarray
            Robot state [x, y, ...]
        robot_yaw : float
            Robot yaw angle
        point : numpy.ndarray
            Point to check [x, y]
        is_in_cam_range : bool
            Whether to also check if point is within camera range
            
        Returns
        -------
        bool
            True if point is in FOV, False otherwise
        """
        robot_pos = robot_state[:2]
        to_point = point - robot_pos
        
        angle_to_point = np.arctan2(to_point[1], to_point[0])
        angle_diff = abs(angle_normalize(angle_to_point - robot_yaw))
        
        in_fov = angle_diff <= self.robot.fov_angle / 2
        #print("robot spec", self.robot_spec)
        
        if is_in_cam_range:
            dist_to_point = np.linalg.norm(to_point)
            return in_fov and dist_to_point <= self.robot.robot_spec["cam_range"]
            
        return in_fov

    def _compute_critical_point(self):
        """
        Compute the critical point where the planned trajectory intersects with the map boundary.
        
        Returns
        -------
        critical_point : numpy.ndarray or None
            The critical point where trajectory intersects map boundary, or None if no intersection
        """
        if self.robot.sensing_footprints.is_empty:
            print("No sensing footprints found")
            return None
            
        # Get the planned trajectory
        trajectory = self.pos_committed_x_traj
        
        # Convert trajectory points to LineString
        trajectory_line = LineString([(x[0], x[1]) for x in trajectory])
        
        # Find intersection with sensing footprints
        intersection = trajectory_line.intersection(self.robot.sensing_footprints)

        if intersection.is_empty:
            return None
            
        # Get the first intersection point
        if intersection.geom_type == 'Point':
            critical_point = np.array([intersection.x, intersection.y])
        elif intersection.geom_type == 'LineString':
            # For LineString intersection, get the furthest point
            # This is the point where we exit the explored area
            robot_pos = self.robot.get_position()
            coords = list(intersection.coords)
            distances = [np.linalg.norm(np.array(p) - robot_pos) for p in coords]
            furthest_idx = np.argmax(distances)
            critical_point = np.array(coords[furthest_idx])
        elif intersection.geom_type in ['GeometryCollection', 'MultiLineString', 'MultiPoint']:
            #print("GeometryCollection intersection")
            robot_pos = self.robot.get_position()
            multi_geom = list(intersection.geoms)
            coords = []
            for geom in multi_geom:
                if geom.geom_type == 'LineString':
                    coords.extend(list(geom.coords))
                elif geom.geom_type == 'Point':
                    coords.append((geom.x, geom.y))
            distances = [np.linalg.norm(np.array(p) - robot_pos) for p in coords]
            furthest_idx = np.argmax(distances)
            critical_point = np.array(coords[furthest_idx])
        else:
            print(f"Unexpected intersection type: {intersection.geom_type}")
            return None
            
        return critical_point

    def _update_committed_trajectory(self, discounted_nominal_horizon):
        """
        Update the committed trajectory with the candidate trajectory.
        """
        self.committed_x_traj = self.candidate_x_traj
        self.committed_u_traj = self.candidate_u_traj
        self.next_event_time = self.event_offset
        self.current_time_idx = 0
        self.committed_horizon = discounted_nominal_horizon

    def solve_control_problem(self,
                              robot_state: np.ndarray,
                              current_yaw: float,
                              u: np.ndarray) -> float:
        '''
            u: ignored here, and instead using pos_committed_u_traj
        '''

        self.current_time_idx += 1
        self._update_pos_committed_trajectory()

        if self.committed_x_traj is None and self.committed_u_traj is None:
            # initialize the committed trajectory
            init_x_traj, init_u_traj = self._generate_trajectory(current_yaw, self.backup_horizon, self.backup_controller)   
            self.committed_x_traj = init_x_traj
            self.committed_u_traj = init_u_traj 

        # try updating the committed trajectory
        if self.current_time_idx > self.next_event_time/self.dt:
            #print("Event triggered, generating new candidate trajectory")
            critical_point = self._compute_critical_point()

            for i in range(int(self.nominal_horizon//self.horizon_discount)):
                # discount the nominal horizon
                discounted_nominal_horizon = self.nominal_horizon - i * self.horizon_discount
                # Generate the candidate trajectory
                candidate_x_traj = self._generate_candidate_trajectory(discounted_nominal_horizon)
                # Check if the candidate trajectory is valid
                if self._is_candidate_valid(critical_point, candidate_x_traj, discounted_nominal_horizon):
                    #print("Candidate trajectory is valid")
                    self._update_committed_trajectory(discounted_nominal_horizon)
                    break
        
        if self.current_time_idx < self.committed_horizon/self.dt:
            # Use the committed trajectory for the next control input
            #print("in nominal: control input", self.committed_u_traj[self.current_time_idx])
            #print("in nominal: control input", self.nominal_controller(self.robot.X, goal))
            #print("nominal")
            return self.committed_u_traj[self.current_time_idx] if self.nominal_controller is None else self.nominal_controller(robot_state, current_yaw, u)
        else:
            #print("backup")
            #print("backup: control input", self.backup_controller(self.robot.X))
            return self.committed_u_traj[self.current_time_idx] if self.backup_controller is None else self.backup_controller(robot_state, current_yaw, u)