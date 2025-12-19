
import numpy as np
import matplotlib.pyplot as plt

from safe_control.utils import plotting
from safe_control.utils import env
from safe_control.tracking import LocalTrackingController

def main():
    dt = 0.05
    model = 'VTOL2D'

    robot_spec = {
        'model': 'VTOL2D',
        'radius': 0.6,
        'v_max': 20.0,
        'reached_threshold': 1.0 # meter
    }
    
    # waypoints specific for VTOL
    waypoints = [
        [2, 10],
        [70, 10],
        [70, 0.5]
    ]
    
    pillar_1_x = 67.0
    pillar_2_x = 73.0
    known_obs = np.array([
        # [pillar_1_x, 1.0, 0.5],
        # [pillar_1_x, 2.0, 0.5],
        # [pillar_1_x, 3.0, 0.5],
        # [pillar_1_x, 4.0, 0.5],
        # [pillar_1_x, 5.0, 0.5],
        [pillar_1_x, 6.0, 0.5],
        [pillar_1_x, 7.0, 0.5],
        [pillar_1_x, 8.0, 0.5],
        [pillar_1_x, 9.0, 0.5],
        [pillar_2_x, 1.0, 0.5],
        [pillar_2_x, 2.0, 0.5],
        [pillar_2_x, 3.0, 0.5],
        [pillar_2_x, 4.0, 0.5],
        [pillar_2_x, 5.0, 0.5],
        [pillar_2_x, 6.0, 0.5],
        [pillar_2_x, 7.0, 0.5],
        [pillar_2_x, 8.0, 0.5],
        [pillar_2_x, 9.0, 0.5],
        [pillar_2_x, 10.0, 0.5],
        [pillar_2_x, 11.0, 0.5],
        [pillar_2_x, 12.0, 0.5],
        [pillar_2_x, 13.0, 0.5],
        [pillar_2_x, 14.0, 0.5],
        [pillar_2_x, 15.0, 0.5],
        [60.0, 12.0, 1.5]
    ])

    env_width = 75.0
    env_height = 20.0
    plt.rcParams['figure.figsize'] = [12, 5]

    waypoints = np.array(waypoints, dtype=np.float64)
    
    v_init = robot_spec['v_max'] # m/s
    x_init = np.hstack((waypoints[0][0:2], 0.0, v_init, 0.0, 0.0))

    if len(known_obs) > 0 and known_obs.shape[1] != 7:
        known_obs = np.hstack((known_obs, np.zeros((known_obs.shape[0], 4)))) # Set static obs velocity 0.0

    plot_handler = plotting.Plotting(width=env_width, height=env_height, known_obs=known_obs)
    ax, fig = plot_handler.plot_grid("VTOL Tracking")
    env_handler = env.Env()

    # Default controller for VTOL
    controller_type = {'pos': 'mpc_cbf'}

    tracking_controller = LocalTrackingController(x_init, robot_spec,
                                                  controller_type=controller_type,
                                                  dt=dt,
                                                  show_animation=True,
                                                  save_animation=False,
                                                  show_mpc_traj=False,
                                                  ax=ax, fig=fig,
                                                  env=env_handler)

    # Set obstacles
    tracking_controller.obs = known_obs
    tracking_controller.set_waypoints(waypoints)
    unexpected_beh = tracking_controller.run_all_steps(tf=100)
    
    if unexpected_beh == -1 or unexpected_beh == 0:
        print("Success!")
    else:
        print("Failed!")

if __name__ == '__main__':
    main()
