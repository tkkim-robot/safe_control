import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

"""
Created on January 27th, 2025
@author: Taekyung Kim

@description: 
Implement quadplane in 2D and visualize the rigid body plus elevators.
"""

def angle_normalize(x):
    """Normalize angle to (-pi, pi]."""
    if isinstance(x, (np.ndarray, float, int)):
        return ((x + np.pi) % (2*np.pi)) - np.pi
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        return ca.fmod(x + ca.pi, 2*ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


class VTOL2D:
    """
    X: [ x, z, theta, x_dot, z_dot, theta_dot ]
        - x,z      -> inertial positions
        - theta    -> pitch angle
        - x_dot,z_dot -> inertial velocities
        - theta_dot   -> pitch rate
    U:[ delta_front, delta_rear, delta_pusher, delta_elevator ]
    cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
    relative degree: 2

    We compute angle of attack alpha from body-frame velocity (u_b, w_b):
        alpha = atan2(w_b, u_b).
    Then we apply lift & drag in the "wind frame" where
        F_wind = [ -D, +L ]
    rotate from wind to body by alpha,
    then from body to inertial by theta.

    Rotors are assumed to produce linear thrust in body coordinates:
      front, rear rotor -> ±body_z,
      pusher rotor -> +body_x,
    and each is rotated to inertial via R(theta).

    Elevator modifies L, D, and also adds a pitch moment linearly => system remains control affine.
    """

    def __init__(self, dt, robot_spec):
        self.dt = dt
        self.spec = robot_spec

        # Default or user-set parameters
        self.spec.setdefault('mass', 11.0)
        self.spec.setdefault('inertia', 1.135)
        self.spec.setdefault('S_wing', 0.55)   # Wing area
        self.spec.setdefault('rho', 1.2682)   # Air density
        self.spec.setdefault('C_L0', 0.23)
        self.spec.setdefault('C_Lalpha', 5.61)
        self.spec.setdefault('C_Ldelta_e', 0.13)    # TODO: might be wrong, but it should be positive
        self.spec.setdefault('C_D0', 0.043)
        self.spec.setdefault('C_Dalpha', 0.03)      # e.g. alpha^2 coefficient
        self.spec.setdefault('C_Ddelta_e', 0.0)     # should be abs(), but makes it not affine
        self.spec.setdefault('C_m0', 0.0135)        # in 2D, use small m for pitch moment
        self.spec.setdefault('C_malpha', -2.74)     # coeff of pitch moment from alpha
        self.spec.setdefault('C_mdelta_e', -0.99)   # for pitch moment from elevator
        self.spec.setdefault('chord', 0.18994)      # mean chord length

        # linear rotor thrust
        self.spec.setdefault('k_front', 90.0)
        self.spec.setdefault('k_rear',  90.0)
        self.spec.setdefault('k_pusher',60.0)
        # geometry: lever arms
        self.spec.setdefault('ell_f', 0.5)
        self.spec.setdefault('ell_r', 0.5)

        # control limits
        self.spec.setdefault('throttle_min', 0.0)
        self.spec.setdefault('throttle_max', 1.0)
        self.spec.setdefault('elevator_min', -0.5)
        self.spec.setdefault('elevator_max', 0.5)

        # visualization
        self.spec.setdefault('plane_width', 1.0)
        self.spec.setdefault('plane_height', 0.3)
        self.spec.setdefault('front_width', 0.25)
        self.spec.setdefault('front_height', 0.1)
        self.spec.setdefault('rear_width', 0.25)
        self.spec.setdefault('rear_height', 0.1)
        self.spec.setdefault('pusher_width', 0.1)
        self.spec.setdefault('pusher_height', 0.4)
        self.spec.setdefault('elev_width', 0.35)
        self.spec.setdefault('elev_height', 0.1)

        # spec safety constraint
        self.spec.setdefault('v_max', 15.0)
        self.spec.setdefault('descent_speed_max', 2.0) # in negative z

        self.gravity = 9.81

    #--------------------------------------------------------------------------
    # Body-frame velocity => alpha => baseline (no-elevator) lift/drag => inertial
    #--------------------------------------------------------------------------
    def f(self, X, casadi=False):
        """
        Unforced dynamics: f(X)
          - includes baseline aerodynamic forces (delta_e = 0)
          - includes gravity
          - no rotor thrust
          - no elevator deflection
        """
        if casadi:
            theta = X[2,0]
            xdot  = X[3,0]
            zdot  = X[4,0]
            thetadot = X[5,0]

            # 1) Body-frame velocity (u_b, w_b)
            u_b, w_b = self._body_velocity_casadi(xdot, zdot, theta)
            V = ca.sqrt(u_b*u_b + w_b*w_b)  # actual airspeed magnitude
            alpha = ca.atan2(w_b, u_b)

            # 2) Baseline lift & drag (no elevator => delta_e=0)
            L0, D0, M0 = self._lift_drag_moment(V, alpha, delta_e=0.0)

            # 3) Wind->Body rotation by alpha, Body->Inertial by theta
            #    => net rotation by (theta + alpha)
            fx_aero, fz_aero = self._wind_to_inertial_casadi(theta, alpha, -D0, L0)

            # 4) Gravity in inertial is (0, -m*g)
            m = self.spec['mass']
            I = self.spec['inertia']
            fx_net = fx_aero
            fz_net = fz_aero - m*self.gravity

            x_ddot = fx_net / m
            z_ddot = fz_net / m
            theta_ddot = M0 / I

            return ca.vertcat(
                xdot,
                zdot,
                thetadot,
                x_ddot,
                z_ddot,
                theta_ddot
            )
        else:
            # NumPy version
            theta = X[2,0]
            xdot  = X[3,0]
            zdot  = X[4,0]
            thetadot = X[5,0]

            u_b, w_b = self._body_velocity_np(xdot, zdot, theta)
            V = np.sqrt(u_b**2 + w_b**2)
            alpha = np.arctan2(w_b, u_b)

            L0, D0, M0 = self._lift_drag_moment(V, alpha, 0.0)

            fx_aero, fz_aero = self._wind_to_inertial_np(theta, alpha, -D0, L0)

            m = self.spec['mass']
            I = self.spec['inertia']
            fx_net = fx_aero
            fz_net = fz_aero - m*self.gravity

            x_ddot = fx_net / m
            z_ddot = fz_net / m
            theta_ddot = M0 / I

            return np.array([
                xdot,
                zdot,
                thetadot,
                x_ddot,
                z_ddot,
                theta_ddot
            ]).reshape(-1,1)

    #--------------------------------------------------------------------------
    # g(X): partial wrt each of the 4 control inputs => columns in a 6x4 matrix
    #--------------------------------------------------------------------------
    def g(self, X, casadi=False):
        """
        Control-dependent dynamics.  U = [ delta_front, delta_rear, delta_pusher, delta_elevator ]^T.
        - front, rear, pusher = rotor forces in body coords
        - elevator => additional (L, D) + pitch moment
        """
        if casadi:
            theta = X[2,0]
            xdot  = X[3,0]
            zdot  = X[4,0]
            # body velocity
            u_b, w_b = self._body_velocity_casadi(xdot, zdot, theta)
            V = ca.sqrt(u_b*u_b + w_b*w_b)
            alpha = ca.atan2(w_b, u_b)

            #-----------------------------------
            # 1) front rotor => thrust along -body_z
            fx_front, fz_front, M_f = self._front_rotor_casadi(theta)
            # 2) rear rotor
            fx_rear, fz_rear, M_r = self._rear_rotor_casadi(theta)
            # 3) pusher rotor => +body_x
            fx_thr, fz_thr, M_t = self._pusher_rotor_casadi(theta)
            # 4) elevator => partial in lift/drag + pitch moment
            #    => difference from delta_e=1 vs 0
            L_de, D_de, M_de = self._lift_drag_moment(V, alpha, delta_e=1.0)  # "partial"
            fx_elev, fz_elev = self._wind_to_inertial_casadi(theta, alpha, -D_de, L_de)

            m = self.spec['mass']
            I = self.spec['inertia']

            #----------------------------------------------------------------------
            # Using block-building to do list-to-SX (otherwise, it reports error)
            # Build a 6x4 matrix (since our state dimension is 6)
            # row=0..5, col=0..3
            # We fill in the partial derivatives w.r.t. each input in each column.
            # Everything not affected by the input remains 0.
            #----------------------------------------------------------------------
            # Create a list of 6 rows, each row has 4 columns => (6x4)
            g_list = [[0]*4 for _ in range(6)]

            # Column 0: front rotor
            g_list[3][0] = fx_front/m
            g_list[4][0] = fz_front/m
            g_list[5][0] = M_f/I

            # Column 1: rear rotor
            g_list[3][1] = fx_rear/m
            g_list[4][1] = fz_rear/m
            g_list[5][1] = M_r/I

            # Column 2: pusher rotor
            g_list[3][2] = fx_thr/m
            g_list[4][2] = fz_thr/m
            g_list[5][2] = M_t/I

            # Column 3: elevator
            g_list[3][3] = fx_elev/m # because of this index is non-zero, it makes relative degree of 2 wrt h(x)
            g_list[4][3] = fz_elev/m # because of this index is non-zero, it makes relative degree of 2 wrt h(x)
            g_list[5][3] = M_de/I

            # Convert each row to a CasADi row vector, then stack them
            g_sx_rows = []
            for row_vals in g_list:
                # row_vals is something like [expr_col0, expr_col1, expr_col2, expr_col3]
                g_sx_rows.append(ca.horzcat(*row_vals))

            g_sx = ca.vertcat(*g_sx_rows)
            return g_sx

        else:
            theta = X[2,0]
            xdot  = X[3,0]
            zdot  = X[4,0]

            # body velocity
            u_b, w_b = self._body_velocity_np(xdot, zdot, theta)
            V = np.sqrt(u_b**2 + w_b**2)
            alpha = np.arctan2(w_b, u_b)

            fx_front, fz_front, M_f = self._front_rotor_np(theta)
            fx_rear,  fz_rear,  M_r = self._rear_rotor_np(theta)
            fx_thr,   fz_thr,   M_t = self._pusher_rotor_np(theta)

            # elevator partial
            L_de, D_de, M_de = self._lift_drag_moment(V, alpha, 1.0)
            fx_elev, fz_elev = self._wind_to_inertial_np(theta, alpha, -D_de, L_de)

            m = self.spec['mass']
            I = self.spec['inertia']

            g_out = np.zeros((6,4))
            # front
            g_out[3,0] = fx_front / m
            g_out[4,0] = fz_front / m
            g_out[5,0] = M_f / I

            # rear
            g_out[3,1] = fx_rear / m
            g_out[4,1] = fz_rear / m
            g_out[5,1] = M_r / I

            # pusher
            g_out[3,2] = fx_thr / m
            g_out[4,2] = fz_thr / m
            g_out[5,2] = M_t / I

            # elevator
            g_out[3,3] = fx_elev / m
            g_out[4,3] = fz_elev / m
            g_out[5,3] = M_de / I

            return g_out

    def step(self, X, U, casadi=False):
        """
        Euler step:
           X_{k+1} = X_k + [ f(X_k) + g(X_k)*U ] * dt
        """
        Xdot = self.f(X, casadi) + self.g(X, casadi) @ U
        Xnew = X + Xdot * self.dt
        Xnew[2,0] = angle_normalize(Xnew[2,0])
        return Xnew

    #--------------------------------------------------------------------------
    # TF utils: body velocity from inertial velocity
    #--------------------------------------------------------------------------
    def _body_velocity_np(self, xdot, zdot, theta):
        """
        Rotate inertial velocity (xdot, zdot) into body frame (u_b, w_b).
         [u_b] = [ cos(theta),  sin(theta)] [ xdot ]
         [w_b]   [-sin(theta), cos(theta)]  [ zdot ]
        """
        cth = np.cos(theta)
        sth = np.sin(theta)
        u_b = cth*xdot + sth*zdot
        w_b = -sth*xdot + cth*zdot
        return u_b, w_b

    def _body_velocity_casadi(self, xdot, zdot, theta):
        cth = ca.cos(theta)
        sth = ca.sin(theta)
        u_b = cth*xdot + sth*zdot
        w_b = -sth*xdot + cth*zdot
        return (u_b, w_b)

    #--------------------------------------------------------------------------
    # compute lift, drag in "wind frame"
    #--------------------------------------------------------------------------
    def _lift_drag_moment(self, V, alpha, delta_e):
        """
        NOTE: universal to np and casadi
        L,D from standard formula with linear elevator effect:
          L = 1/2 * rho * V^2 * S * (C_L0 + C_Lalpha*alpha + C_Ldelta_e * delta_e)
          D = 1/2 * rho * V^2 * S * (C_D0 + C_Dalpha*(alpha^2) + C_Ddelta_e * delta_e)
          M = 1/2 * rho * V^2 * S * chord * (C_m0 + C_malpha*alpha + C_mdelta_e * delta_e)

        Returns (L, D, M)  (magnitudes).
        """
        rho = self.spec['rho']
        S   = self.spec['S_wing']
        CL = (self.spec['C_L0']
              + self.spec['C_Lalpha']*alpha
              + self.spec['C_Ldelta_e']*delta_e)
        CD = (self.spec['C_D0']
              + self.spec['C_Dalpha']*(alpha**2)
              + self.spec['C_Ddelta_e']*delta_e)
        CM = (self.spec['C_m0']
              + self.spec['C_malpha']*alpha
              + self.spec['C_mdelta_e']*delta_e)
        chord = self.spec['chord']


        qbar = 0.5*rho*(V**2)
        L = qbar * S * CL           # N
        D = qbar * S * CD           # N
        M = qbar * S * CM * chord   # Nm
        return (L, D, M)

    #--------------------------------------------------------------------------
    # Rotation from wind frame to inertial
    #   wind->body: rotate by alpha
    #   body->inertial: rotate by theta
    #   => total rotation by (theta + alpha)
    #   =>  F_inertial = R(theta+alpha)*F_wind
    #--------------------------------------------------------------------------
    def _wind_to_inertial_np(self, theta, alpha, fx_w, fz_w):
        """
        wind frame axes: x_w = direction of velocity, z_w = +90 deg from that
        If (fx_w, fz_w) = (-D, L), then we rotate by (theta+alpha).
        """
        heading = theta + alpha
        c = np.cos(heading)
        s = np.sin(heading)
        fx_i = c*fx_w - s*fz_w
        fz_i = s*fx_w + c*fz_w
        return fx_i, fz_i

    def _wind_to_inertial_casadi(self, theta, alpha, fx_w, fz_w):
        heading = theta + alpha
        c = ca.cos(heading)
        s = ca.sin(heading)
        fx_i = c*fx_w - s*fz_w
        fz_i = s*fx_w + c*fz_w
        return fx_i, fz_i

    #--------------------------------------------------------------------------
    # Rotors utils: front, rear, pusher
    #--------------------------------------------------------------------------
    def _front_rotor_np(self, theta):
        """
        front rotor pusher = k_front * delta_front in +body_z
        We'll say +body_z => (0, +k_front). Then to inertial => R(theta).
        Also pitch moment => +ell_f * T_f.  We'll only return partial.
        """
        # partial w.r.t delta_front is (k_front)
        # in body coords => (fx_b, fz_b) = (0, +k_front)
        # rotate to inertial by R(theta).
        cth = np.cos(theta)
        sth = np.sin(theta)
        # inertial
        fx_i = -sth * self.spec['k_front']
        fz_i =  cth * self.spec['k_front']
        # pitch moment
        M_f  = +self.spec['ell_f'] * self.spec['k_front']
        return fx_i, fz_i, M_f

    def _rear_rotor_np(self, theta):
        cth = np.cos(theta)
        sth = np.sin(theta)
        fx_i = -sth * self.spec['k_rear']
        fz_i =  cth * self.spec['k_rear']
        M_r  = -self.spec['ell_r'] * self.spec['k_rear']
        return fx_i, fz_i, M_r

    def _pusher_rotor_np(self, theta):
        """
        pusher rotor => pusher along +body_x => rotate by theta
        """
        cth = np.cos(theta)
        sth = np.sin(theta)
        fx_i =  cth * self.spec['k_pusher']
        fz_i =  sth * self.spec['k_pusher']
        M_t  =  0.0
        return fx_i, fz_i, M_t

    # CasADi versions
    def _front_rotor_casadi(self, theta):
        cth = ca.cos(theta)
        sth = ca.sin(theta)
        fx_i = -sth*self.spec['k_front']
        fz_i =  cth*self.spec['k_front']
        M_f  =  self.spec['ell_f']*self.spec['k_front']
        return fx_i, fz_i, M_f

    def _rear_rotor_casadi(self, theta):
        cth = ca.cos(theta)
        sth = ca.sin(theta)
        fx_i = -sth*self.spec['k_rear']
        fz_i =  cth*self.spec['k_rear']
        M_r  = -self.spec['ell_r']*self.spec['k_rear']
        return fx_i, fz_i, M_r

    def _pusher_rotor_casadi(self, theta):
        cth = ca.cos(theta)
        sth = ca.sin(theta)
        fx_i = cth*self.spec['k_pusher']
        fz_i = sth*self.spec['k_pusher']
        M_t  = 0.0
        return fx_i, fz_i, M_t
    
    def nominal_input(self, X, G):
        # not imeplemented
        return np.zeros((4, 1))

    def stop(self, X):
        # not imeplemented
        return np.zeros((4, 1))
    
    def has_stopped(self, X, tol=0.05):
        """Check if quadrotor has stopped within tolerance."""
        return np.linalg.norm(X[3:5, 0]) < tol

    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        # Not implemented
        raise NotImplementedError("Not implemented")
    
    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k, casadi=True)
        x_k2 = self.step(x_k1, u_k, casadi=True)

        def h(x, obs, robot_radius, beta=1.01):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            z_obs = obs[1]
            r_obs = obs[2]
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - z_obs)**2 - beta*d_min**2
            return h

        h_k2 = h(x_k2, obs, robot_radius, beta)
        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        dd_h = h_k2 - 2 * h_k1 + h_k
        # hocbf_2nd_order = h_ddot + (gamma1 + gamma2) * h_dot + (gamma1 * gamma2) * h_k
        return h_k, d_h, dd_h
    
    def render_rigid_body(self, X, U):
        """
        Return transforms for body, front rotor, rear rotor,
        forward (pusher) throttle, and elevator.
        """
        x, z, theta, xdot, zdot, thetadot = X.flatten()
        delta_front, delta_rear, delta_pusher, delta_elev = U.flatten()

        # 1) Body transform: rotate by theta, then translate by (x,z)
        transform_body = Affine2D().rotate(theta).translate(x, z) + plt.gca().transData

        # 2) Front rotor: place it at the front of the plane
        #    Suppose the front is ~ +0.5 * plane_width from center
        #    We'll rotate the same theta, then translate to (x_front, z_front)
        front_offset = self.spec['plane_width']/2  # tune as needed
        x_front = x + front_offset * np.cos(theta)
        z_front = z + front_offset * np.sin(theta)
        transform_front = Affine2D().rotate(theta).translate(x_front, z_front) + plt.gca().transData

        # 3) Rear rotor: place it at the negative front_offset
        x_rear = x - front_offset * np.cos(theta)
        z_rear = z - front_offset * np.sin(theta)
        transform_rear = Affine2D().rotate(theta).translate(x_rear, z_rear) + plt.gca().transData

        # 4) Forward (pusher) throttle: 
        #    For simplicity, place it at the center and beneath the plane
        push_offset_x = 0.0
        push_offset_z = -self.spec['plane_height']/2 - self.spec['pusher_height']/2
        # Translate locally:
        transform_pusher = Affine2D().translate(push_offset_x, push_offset_z)
        # Rotate about local origin:
        transform_pusher = transform_pusher.rotate(theta)
        # Translate to global position (x,z):
        transform_pusher = transform_pusher.translate(x, z)
        # Finally, attach it to the axes:
        transform_pusher = transform_pusher + plt.gca().transData
        
        # 5) Elevator: hinge at the rear
        #    We want to rotate it by (theta + delta_elev) around the hinge
        #    => shift the coordinate so hinge is at origin, rotate, shift back
        #    But a simpler approach is: rotate by 'theta', then do a local rotate by 'delta_elev'
        #    about local rectangle corner. For illustration:
        elevator_offset = - (self.spec['plane_width']/2 + 0.25)  # plane length/2 plus a bit
        x_elev = x + elevator_offset * np.cos(theta)
        z_elev = z + elevator_offset * np.sin(theta)
        # first rotate by "theta" (plane orientation), then rotate about the local rectangle's pivot by 'delta_elev'
        transform_elev = Affine2D()
        transform_elev = transform_elev.rotate_around(self.spec['elev_width']/2, 0, delta_elev)  # local rotation about rectangle corner
        transform_elev = transform_elev.rotate(theta)
        transform_elev = transform_elev.translate(x_elev, z_elev) + plt.gca().transData

        return transform_body, transform_front, transform_rear, transform_pusher, transform_elev