
import time
import math
import rclpy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import cv2
from PIL import Image, ImageFilter

from rclpy.node import Node
from rclpy.qos import QoSProfile
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, FloatingPointRange

from system_msgs.msg import SYSTEMSIMULATIONCOMMAND
from learn_msgs.msg import LearnData
from learn_msgs.msg import DesireSpeed
from learn_msgs.msg import MppiResult

from auto_learn.module import module
from auto_learn.dynamics.nn_vehicle import VehicleModel
from auto_learn.pytorch_mppi import smooth_mppi_deploy as mppi
# from auto_learn.pytorch_mppi import mppi_map as mppi

# ACTIVATION = 'softplus'
ACTIVATION = 'relu'
PATH = '/home/add/Desktop/auto_ws/src/auto_learn'

SUBFOLDER = 'rs_230123_300_50'
#SUBFOLDER = 'jrd_230103_100_50'
#SUBFOLDER = 'gn_230105_none_50'
#SUBFOLDER = 'jrdx_230105_100_50'
SUBFOLDER = 'jrd_230103_100_50'


MODELNAME = SUBFOLDER + '/step_300'

MODELSTEP = 1

n_states = 3
n_actions = 2
n_history = 4
n_hidden = 40
n_ensemble = 5
n_trajectory = 5000

TIMESTEPS = 35

# 2 at continuous # 2 at discrete # let MultivariateNormal to have steer_max sometimes
VAR_STEERING = 1.6  # 1.1 for w/o action_state cost # 0.7
# 3 at continuous # 30 at discrete # let MultivariateNormal to have v_max sometimes
VAR_DESIRED_V = 0.4
W_SPEED_COST = 1.0
W_P2E_COST = 3.0

W_TRACK_COST = 3.5
W_STABIL_COST = 0.5  # 1.0
W_ACTION_STEER = 0.4  # 0.0  # 0.8
W_ACTION_VX = 3.5  # 0.0  # 0.8

d_steer_max = 2.0  # 2.0
d_vx_max = 4.0  # 5.0

lambda_ = 20.0  # 0.1 -> 99% were rejected (omega->0)
gamma_ = 0.1

W_TRACK_PEN = 1000000.0/255.0  # divide by value of white pixel (255)
W_SLIP_PEN = 10000  # large penalty on large slip
THR_SLIP_PEN = 0.3
TRACK_COST_TIME_DECAY = 0.9
EDGE_PENALTY = 0.1  # smaller, more penalty

#########################
### GLOBAL PARAMETERS####
#########################
LOG_FLAG = True
TRAJ_VIS = True
PLOT_SIZE = 100  # 300  # 10
# ACTION_LOW, ACTION_HIGH = -2.0, 2.0
dt = 0.1  # 10hz

_VX = 40
DESIRED_SPEED = round(_VX*1000/3600, 2)  # m/s
# steering, desired_Vx
vx_max = round(_VX * 1000 / 3600, 2)  # m/s
steer_max = round(400 / 180 * math.pi, 2)

# maximum delta action on 0.1 seconds
U_MIN = torch.tensor([-steer_max, 0])
U_MAX = torch.tensor([steer_max, vx_max])
D_U_MIN = torch.tensor([-d_steer_max, -d_vx_max])
D_U_MAX = torch.tensor([d_steer_max, d_vx_max])
NOISE_MU = torch.tensor([0.0, 0.0])

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
noise_sigma = torch.tensor(
    [[VAR_STEERING, 0], [0, VAR_DESIRED_V]], device=device, dtype=torch.float32)  # shape[0] = n_actions

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

SELECT_MAP = 3
if SELECT_MAP == 0:
    img_name = 'map.png'
elif SELECT_MAP == 1:
    img_name = 'first.png'
elif SELECT_MAP == 2:
    img_name = 'second.png'
elif SELECT_MAP == 3:
    img_name = 'third.png'
else:
    img_name = 'fourth.png'


def img2costmap(img):

    lower_bound = np.array([176, 176, 176])
    upper_bound = np.array([179, 179, 179])
    mask = cv2.inRange(img, lower_bound, upper_bound)
    res = cv2.bitwise_and(img, img, mask=mask)

    image = Image.fromarray(res)
    image = image.filter(ImageFilter.ModeFilter(size=30))
    res = np.array(image)

    kernel = np.ones((3, 3), np.uint8)
    '''
    res = cv2.dilate(res, kernel, iterations=3)
    res = cv2.erode(res, kernel, iterations=3)
    '''

    ret, thr = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)
    thr = cv2.cvtColor(thr, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[1])
    # thr = thr[y:y+h, x:x+w]
    index_start_x = x+w/2
    index_start_y = y+h/2
    print(index_start_x, index_start_y)
    '''
    res = cv2.dilate(res, kernel, iterations=7)
    res = cv2.erode(res, kernel, iterations=7)
    '''
    ret, thr = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)
    thr = cv2.cvtColor(thr, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    # cv2.rectangle(thr, (x,y), (x+w, y+h), 255, 3)
    cost_map_bin = thr[y:y+h, x:x+w]
    # cost_map_color = cv2.cvtColor(cost_map_bin, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("Asdf", cost_map_color)
    # cv2.imwrite("a.png", cost_map_color)
    # cv2.waitKey(5000)

    # erode cost map more, for conservative driving
    cost_map_bin = cv2.erode(cost_map_bin, kernel, iterations=3)
    cost_map_edge = cost_map_bin.copy()*EDGE_PENALTY
    cost_map_edge = np.array(cost_map_edge, dtype='uint8')
    cost_map_bin = cv2.erode(cost_map_bin, kernel, iterations=3)  # 15

    cost_map_bin = np.array(cost_map_edge, dtype='int32') + \
        np.array(cost_map_bin, dtype='int32')
    cost_map_bin = np.clip(cost_map_bin, 0, 255)
    cost_map_bin = np.array(cost_map_bin, dtype='uint8')

    for y in range(1, len(cost_map_bin)):
        for x in range(7):
            cost_map_bin[y-1, len(cost_map_bin[0])-1-x] = 0
            cost_map_bin[y-1, x] = 0
    for x in range(1, len(cost_map_bin[0])):
        for y in range(7):
            cost_map_bin[len(cost_map_bin)-1-y, x-1] = 0
            cost_map_bin[y, x-1] = 0

    cost_map_color = cv2.cvtColor(cost_map_bin, cv2.COLOR_GRAY2BGR)

    # cv2.imshow("Asdf", cost_map_color)
    # # cv2.imwrite("a.png", cost_map_color)
    # cv2.waitKey(0)

    # cv2.circle(cost_map_color, ((int)(index_start_x),
    #                            (int)(index_start_y)), 3, (0, 255, 0), 3)

    return cost_map_color, cost_map_bin, index_start_x, index_start_y


img = cv2.imread(
    PATH + '/auto_learn/' + img_name, cv2.IMREAD_COLOR)
cost_map_color, cost_map_bin, index_start_x, index_start_y = img2costmap(img)
print(index_start_x, index_start_y)


# race_van_tree, clockwise
if SELECT_MAP == 0:
    index_start_x = 348.5
    index_start_y = 573.5
elif SELECT_MAP == 1:
    index_start_x = 371
    index_start_y = 536
elif SELECT_MAP == 2:
    index_start_x = 913
    index_start_y = 387.5
elif SELECT_MAP == 3:
    index_start_x = 424
    index_start_y = 24
#427, 25
else:
    index_start_x = 471
    index_start_y = 144


# ret, cost_map_bin = cv2.threshold(
#    cost_map_bin, 127, 255, cv2.THRESH_BINARY_INV)
cost_map_bin = 255 - cost_map_bin
# cv2.imshow("asdf",cost_map_bin)
# cv2.waitKey(0)

cost_map_bin = torch.tensor(cost_map_bin, device=device)

# real width of the map = 257 m
# print(len(cost_map_color), len(cost_map_color[0]))
# MAP_HEIGHT = len(cost_map_color) - 1
# MAP_WIDTH = len(cost_map_color[0]) - 1
# GRID2METER = 257/len(cost_map_color[0])
# print(GRID2METER)
# METER2GRID = len(cost_map_color[0])/257  # TODO

# real width of the map = 191.5 m
print(len(cost_map_color), len(cost_map_color[0]))
MAP_HEIGHT = len(cost_map_color) - 1
MAP_WIDTH = len(cost_map_color[0]) - 1
if SELECT_MAP == 0:
    real_map_width = 257
elif SELECT_MAP == 1:
    real_map_width = 326.5
elif SELECT_MAP == 2:
    real_map_width = 284.8
elif SELECT_MAP == 3:
    real_map_width = 191.0
else:
    real_map_width = 191.5
GRID2METER = real_map_width/len(cost_map_color[0])
print(GRID2METER)
METER2GRID = len(cost_map_color[0])/real_map_width  # TODO


class DeployClass(Node):
    def __init__(self):
        super().__init__('MpDeploy')
        qos_profile = QoSProfile(depth=1)

        self.timer = self.create_timer(dt, self.publish_mppi_action_10hz)

        self.state_subscriber = self.create_subscription(LearnData, '/TruckMaker/Car/LearnData', self.subscribe_car_state,
                                                         qos_profile)

        self.system_state_subscriber = self.create_subscription(SYSTEMSIMULATIONCOMMAND, '/MPPI/Deploy/System_SimulationCommand', self.subscribe_mppi_reset,
                                                                qos_profile)

        self.mppi_reuslt_publisher = self.create_publisher(
            MppiResult, '/Mppi/System/Result', qos_profile)

        self.mppi_result = MppiResult()

        # self.carmaker_running = False
        self.action_state = 0

        # [1000,5,4]
        self.vehicle_history_data = torch.zeros(
            n_trajectory, (n_states + n_actions), n_history).to(device)

        self.Vehicle = VehicleModel(n_states, n_actions, n_hidden, n_ensemble, n_history,
                                    n_trajectory, device, lr=0.0001, model_type='vanilla', incremental=False, activation=ACTIVATION)

        self.MPPI = mppi.MPPI(self.Vehicle.inference, self.running_cost, self.Vehicle.initialize_history,
                              n_states, noise_sigma, num_samples=n_trajectory, horizon=TIMESTEPS,
                              dt=dt, lambda_=lambda_, gamma_=gamma_, w_action_steer=W_ACTION_STEER, w_action_vx=W_ACTION_VX,
                              device=device, noise_mu=NOISE_MU,
                              u_min=U_MIN, u_max=U_MAX, d_u_min=D_U_MIN, d_u_max=D_U_MAX)

        self.checkpoint = torch.load(
            PATH + '/auto_learn/checkpoint_step/' + MODELNAME + '.pth')
        # self.Vehicle.load_state_dict(self.checkpoint)
        self.Vehicle.model.load_state_dict(self.checkpoint)

        self.mppi_lock = False
        self.model_num = 1

        self.global_x = 0.
        self.global_y = 0.
        self.global_z = 0.
        self.car_simtime = 0.

        if SELECT_MAP == 0:
            self.global_start_x = 485.987
            self.global_start_y = -178.777
        elif SELECT_MAP == 1:
            self.global_start_x = 12.56
            self.global_start_y = -37.77
        elif SELECT_MAP == 2:
            self.global_start_x = 170.15
            self.global_start_y = -6.74
        elif SELECT_MAP == 3:
            self.global_start_x = 54.14#46.04
            self.global_start_y = 49.09#7.75
        else:
            self.global_start_x = 120.0
            self.global_start_y = 60.72

        self.global_yaw = 0.
        self.global_vx = 0.
        self.global_vy = 0.
        self.global_yaw_rate = 0.
        self.global_roll_rate = 0.
        self.global_slip = 0.

        self.cost_map_color_vis = cost_map_color.copy()
        self.xs = torch.empty(n_trajectory).to(device)
        self.ys = torch.empty(n_trajectory).to(device)
        self.yaws = torch.empty(n_trajectory).to(device)

        if LOG_FLAG == True:
            self.dataframe = pd.DataFrame(
                columns=['total', 'track', 'speed', 'stabilizing', 'sim_time', 'distance', 'vx', 'vy', 'kph_speed'])
            if not os.path.isdir(PATH + 'auto_learn/data_inference/' + SUBFOLDER):
                os.makedirs(PATH + '/auto_learn/data_inference/' +
                            SUBFOLDER, exist_ok=True)
            self.dataframe.to_csv(PATH + '/auto_learn/data_inference/' + MODELNAME + '.csv',
                                  mode='w', header=True)
            self.track_cost = torch.Tensor([0]).to('cuda')
            self.speed_cost = torch.Tensor([0]).to('cuda')
            self.stabilizing_cost = torch.Tensor([0]).to('cuda')
        self.count = 0
        self.subscribe_count = 0

        self.params = {}
        self.params["var_steer"] = VAR_STEERING
        self.params["var_desired_v"] = VAR_DESIRED_V
        self.params["w_track"] = W_TRACK_COST
        self.params["w_speed"] = W_SPEED_COST
        self.params["w_stabil"] = W_STABIL_COST
        self.params["w_action_steer"] = W_ACTION_STEER
        self.params["w_action_vx"] = W_ACTION_VX
        self.params["desired_v"] = DESIRED_SPEED
        self.params["max_steer_handle"] = steer_max
        self.params["max_v"] = vx_max
        self.params["max_d_steer"] = d_steer_max
        self.params["max_d_v"] = d_vx_max
        self.params["lambda"] = lambda_
        self.params["gamma"] = gamma_
        self.params["map_visualize"] = TRAJ_VIS

        self.w_track_pen = W_TRACK_PEN
        self.w_slip_pen = W_SLIP_PEN
        self.thr_slip_pen = THR_SLIP_PEN
        self.time_decay = TRACK_COST_TIME_DECAY
        self.dt = dt

        self.max_pixel = torch.tensor([255]).to(device)

        '''
        self.desired_v = DESIRED_SPEED
        self.w_track = W_TRACK_COST
        self.w_track_pen = W_TRACK_PEN
        self.w_speed = W_SPEED_COST
        self.w_stabil = W_STABIL_COST
        '''

        _range = FloatingPointRange()
        _range.from_value = 0.0
        _range.to_value = 10.0
        _range.step = 0.01
        descriptor = ParameterDescriptor(
            floating_point_range=[_range])
        _range_large = FloatingPointRange()
        _range_large.from_value = 0.0
        _range_large.to_value = 50.0
        _range_large.step = 0.01
        descriptor_large = ParameterDescriptor(
            floating_point_range=[_range_large])
        _range_small = FloatingPointRange()
        _range_small.from_value = 0.0
        _range_small.to_value = 5.0
        _range_small.step = 0.01
        descriptor_small = ParameterDescriptor(
            floating_point_range=[_range_small])
        self.declare_parameter('var_steer', VAR_STEERING, descriptor_small)
        self.declare_parameter(
            'var_desired_v', VAR_DESIRED_V, descriptor_small)
        # Declare ROS2 Param for dynamic configuration
        self.declare_parameter('w_speed', W_SPEED_COST, descriptor_small)
        self.declare_parameter('w_track', W_TRACK_COST, descriptor_small)
        self.declare_parameter('w_stabil', W_STABIL_COST, descriptor_small)
        self.declare_parameter(
            'w_action_steer', W_ACTION_STEER, descriptor_small)
        self.declare_parameter(
            'w_action_vx', W_ACTION_VX, descriptor_large)
        self.declare_parameter('desired_v', DESIRED_SPEED, descriptor_large)
        self.declare_parameter('max_steer_handle', steer_max, descriptor)
        self.declare_parameter('max_v', vx_max, descriptor_large)
        self.declare_parameter('max_d_steer', d_steer_max, descriptor)
        self.declare_parameter('max_d_v', d_vx_max, descriptor)

        self.declare_parameter('lambda', lambda_, descriptor_large)
        self.declare_parameter('gamma', gamma_, descriptor_small)

        # rclpy.Parameter.Type.BOOL = 1
        descriptor = ParameterDescriptor(
            type=1)
        self.declare_parameter('map_visualize', TRAJ_VIS, descriptor)

        # this is deprecated in ros2 foxy
        # use "add_on_set_parameters_callback()" instead
        self.set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            self.params[param.name] = param.value

        self._noise_sigma = torch.tensor(
            [[self.params["var_steer"], 0], [0, self.params["var_desired_v"]]], device=device, dtype=torch.float32)
        self._U_MIN = torch.tensor([-self.params["max_steer_handle"], 0])
        self._U_MAX = torch.tensor(
            [self.params["max_steer_handle"], self.params["max_v"]])
        self._D_U_MIN = torch.tensor(
            [-self.params["max_d_steer"], -self.params["max_d_v"]])
        self._D_U_MAX = torch.tensor(
            [self.params["max_d_steer"], self.params["max_d_v"]])
        self.MPPI.update_parameters(noise_sigma=self._noise_sigma,
                                    lambda_=self.params["lambda"],
                                    gamma_=self.params["gamma"],
                                    w_action_steer=self.params["w_action_steer"],
                                    w_action_vx=self.params["w_action_vx"],
                                    u_min=self._U_MIN,
                                    u_max=self._U_MAX,
                                    d_u_min=self._D_U_MIN,
                                    d_u_max=self._D_U_MAX)
        print(" -------------- Parameter is Updated !!! ----------------")
        return SetParametersResult(successful=True)

    def initialize_vehicle_history(self):
        self.vehicle_history_data = torch.zeros_like(
            self.vehicle_history_data)

    def subscribe_mppi_reset(self, msg):
        self.model_num += 1
        self.get_logger().info('I receive Command:')
        # self.checkpoint = torch.load(
        #     '/home/add/robot_ws/src/auto_learn/auto_learn/checkpoint/' + MODELNAME + '.pth')
        try:
            self.checkpoint = torch.load(
                PATH + '/auto_learn/checkpoint_step/' + SUBFOLDER + '/step_' + str(self.model_num*MODELSTEP) + '.pth')
            self.dataframe = pd.DataFrame(
                columns=['total', 'track', 'speed', 'stabilizing', 'sim_time', 'distance', 'vx', 'vy', 'kph_speed'])
            self.dataframe.to_csv(PATH + '/auto_learn/data_inference/' + SUBFOLDER + '/step_' + str(self.model_num*MODELSTEP) + '.csv',
                                  mode='w', header=True)

            print("Change Dynamic Model: step_" +
                  str(self.model_num*MODELSTEP))
        except:
            print("There is no model to deploy -> System End")
            # TODO: Add Ros Topic to exit 'auto_deploy'
            # auto_deploy add in auto_ws and change audo_cmd -> auto_learn

            exit()
        self.mppi_lock = True

        self.Vehicle.model.load_state_dict(self.checkpoint)
        self.MPPI.update_dynamics(self.Vehicle.inference)
        # FIXME:
        self.initialize_vehicle_history()
        self.MPPI.reset()  # reset 'A'(action state) and 'U'(control) to zero
        self.mppi_lock = False

    def subscribe_car_state(self, msg):
        self.global_x = msg.veh_poi_x
        self.global_y = msg.veh_poi_y
        self.global_z = msg.veh_poi_z
        # didn't use quarternion for convinience

        self.car_simtime = msg.sim_time

        if self.car_simtime < 0.2:
            self.MPPI.reset()

        self.global_yaw = msg.car_yaw
        self.global_vx = msg.car_vx
        self.global_vy = msg.car_vy
        self.global_yaw_rate = msg.car_yaw_vel
        self.global_roll_rate = msg.car_roll_vel
        self.global_slip = msg.sideslip_angle

        self.sim_time = msg.sim_time
        self.vehicle_distance = msg.vehicle_distance
        # if self.global_vx == -100.0:
        #     self.carmaker_running = False
        #     print("="*20)
        #     print("TestRun Finished, Initialize History !!! ")
        #     print("="*20)
        #     self.initialize_vehicle_history()
        #     self.MPPI.reset()
        #     return 0
        # self.carmaker_running = True

        self.subscribe_count += 1

    def log_data_init(self):
        track = float(self.track_cost)
        speed = float(self.speed_cost)
        stabilize = float(self.stabilizing_cost)
        total = track + speed + stabilize

        kph_speed = math.sqrt(self.global_vx**2 + self.global_vy**2)*3.6

        if self.subscribe_count > self.count:
            dataframe2 = pd.DataFrame(
                data=[[total, track, speed, stabilize, self.sim_time, self.vehicle_distance, self.global_vx, self.global_vy, kph_speed]], columns=['total', 'track', 'speed', 'stabilizing', 'time', 'distance', 'vx', 'vy', 'kph_speed'])
            dataframe2.to_csv(PATH+'/auto_learn/data_inference/' + SUBFOLDER + '/step_' +
                              str(self.model_num*MODELSTEP) + '.csv', mode='a', header=False)

        self.track_cost = torch.Tensor([0]).to('cuda')
        self.speed_cost = torch.Tensor([0]).to('cuda')
        self.stabilizing_cost = torch.Tensor([0]).to('cuda')

        self.count = self.subscribe_count

    def publish_mppi_action_10hz(self):
        # if not self.carmaker_running:
        #     return 1

        if self.params["map_visualize"]:
            self.cost_map_color_vis = cost_map_color.copy()

        # xs, ys, yaws store mppi state for state cost
        self.xs = self.xs.fill_(self.global_x)
        self.ys = self.ys.fill_(self.global_y)
        self.yaws = self.yaws.fill_(self.global_yaw)

        # state = [self.global_vx, self.global_vy, self.global_yaw_rate,
        #          self.global_roll_rate, self.global_slip]

        state = [self.global_vx,
                 self.global_vy, self.global_yaw_rate]

        state_normalize = module.normalize(state)
        # state_normalize = state

        # state_normalize = [self.global_vx,
        #                    self.global_vy, self.global_yaw_rate]

        t1 = time.time()
        ## * ##
        actions, _, total_reward = mppi.run_mppi(
            self.MPPI, state=state_normalize,
            action_state=self.action_state,
            vehicle_history_data=self.vehicle_history_data)

        # to visualize
        if self.params["map_visualize"]:
            x, y = self.meter2grid_conversion(self.global_x, self.global_y)
            color = int(min(abs(self.global_yaw_rate) / 0.5, 1) * 255)
            cv2.circle(self.cost_map_color_vis,
                       ((int)(x), (int)(y)), 4, (0, 255-color, color), 4)
            # cost_map_color_vis = cv2.resize(
            #    self.cost_map_color_vis, dsize=(0, 0), fx=0.8, fy=0.8)
            # cv2.imshow("Asdf", cost_map_color_vis)
            cv2.imshow("Asdf", self.cost_map_color_vis)
            cv2.waitKey(1)

        action = actions[0]
        next_action = actions[1]

        self.action_state = action
        state = torch.tensor(state_normalize).to(device)  # .to('cpu')
        self.vehicle_history_data = self.update_vehicle_history_data(
            state=state, action=action)
        t2 = time.time()

        print("steering : " + str(float(action[0])))
        print("desired_vx: " + str(float(action[1])))
        print("took {:.6f} sec".format(t2-t1))
        print("reference_speed : ", DESIRED_SPEED*3.6)
        print("="*30)
        print()

        self.control_steer = float(action[0])
        self.control_speed = float(action[1])

        self.mppi_result.control_steer = float(self.control_steer)
        self.mppi_result.control_speed = float(self.control_speed)
        self.mppi_reuslt_publisher.publish(self.mppi_result)

        # cost calculation
        current_cost = self.running_cost_single(state, torch.tensor(
            [self.global_x, self.global_y, self.global_yaw]))
        if LOG_FLAG == True:
            self.log_data_init()
        # TODO: current_cost

    def update_vehicle_history_data(self, state, action):
        history_data = self.vehicle_history_data[:, :, 1:]
        current_data = torch.cat([state, action]).expand(n_trajectory, -1).unsqueeze(
            2)  # expand the 1-D tensor to K*(X+S) 2D tensor by repeating
        # current_data = torch.cat([state, action], dim=1).unsqueeze(2)  # shoule be K*(S+X)*1

        history_data = torch.cat(
            [history_data, current_data], dim=2)  # K*(X+S)*H
        # remain 3D tensorerd

        return history_data  # .view((n_trajectory, -1))

    def running_cost_single(self, state, global_state):
        global cost_map_bin
        # do not use 'u(action)' in state cost
        # print("state cost shape", state.shape)

        # Denormalize
        vx = state[0] * 10.0
        vy = state[1] * 5.0
        yaw_rate = state[2] * 1.0

        yaw = global_state[2] + yaw_rate*dt  # tensor + tensor*scalar
        x = global_state[0] + (vx*torch.cos(yaw) - vy*torch.sin(yaw))*dt
        y = global_state[1] + (vx*torch.sin(yaw) + vy*torch.cos(yaw))*dt

        xs, ys = self.meter2grid_conversion(x, y)

        xs = torch.clamp(xs, min=0, max=MAP_WIDTH)
        ys = torch.clamp(ys, min=0, max=MAP_HEIGHT)

        xs = np.array(xs.to('cpu'), dtype='int64')
        ys = np.array(ys.to('cpu'), dtype='int64')

        curr_map_mask = cost_map_bin[ys, xs]

        # map_mask = cost_map_bin[ys, xs]
        track_cost = torch.min(
            curr_map_mask, self.max_pixel) * self.w_track_pen
        # calculate speed cost
        v = torch.sqrt(torch.square(vx)+torch.square(vy))
        speed_cost = torch.square(self.params["desired_v"] - v)

        # calculate stabilizing cost
        slip_angle = torch.abs(torch.atan(vy/torch.abs(vx)))
        # make 1 where slip angle larger than 0.75, otherwise 0
        stabilizing_cost = torch.square(
            slip_angle) + torch.where(slip_angle > self.thr_slip_pen, 1.0, 0.0) * self.w_slip_pen  # slip angle penalty

        cost = self.params["w_track"] * track_cost * self.time_decay + self.params["w_speed"] * \
            speed_cost + self.params["w_stabil"] * 1000 * stabilizing_cost

        if LOG_FLAG == True:
            self.track_cost = self.params["w_track"] * \
                track_cost * self.time_decay
            self.speed_cost = self.params["w_speed"] * speed_cost
            self.stabilizing_cost = self.params["w_stabil"] * \
                1000 * stabilizing_cost

        # print(self.speed_cost_mean, self.speed_cost_mean.shape)

        return cost

    def running_cost(self, state, prev_map_mask, t):
        global cost_map_bin
        # do not use 'u(action)' in state cost
        # print("state cost shape", state.shape)

        vx = state[:, 0] * 10.0
        vy = state[:, 1] * 5.0
        yaw_rate = state[:, 2] * 1.0

        # vx = state[:, 0]
        # vy = state[:, 1]
        # yaw_rate = state[:, 2]

        # calculate track cost
        self.yaws += yaw_rate*dt  # tensor + tensor*scalar
        self.xs += (vx*torch.cos(self.yaws) - vy*torch.sin(self.yaws))*dt
        self.ys += (vx*torch.sin(self.yaws) + vy*torch.cos(self.yaws))*dt

        xs, ys = self.meter2grid_conversion(self.xs, self.ys)

        xs = torch.clamp(xs, min=0, max=MAP_WIDTH)
        ys = torch.clamp(ys, min=0, max=MAP_HEIGHT)

        xs = np.array(xs.to('cpu'), dtype='int64')
        ys = np.array(ys.to('cpu'), dtype='int64')
        # multiple indexing on cost map, which is loaded on CUDA memory
        # if a trajectory that was previously went out of the map,
        # then penalize the entire trajectory (permit road edge = 127.5)

        curr_map_mask = cost_map_bin[ys, xs]  # +prev_map_mask

        # map_mask = cost_map_bin[ys, xs]
        track_cost = torch.min(
            curr_map_mask, self.max_pixel) * self.w_track_pen
        map_mask = torch.where(curr_map_mask > 244.0, 255, 0)

        if self.params["map_visualize"]:
            # _track_cost = np.array(track_cost.to('cpu'))
            _map_mask = np.array(map_mask.to('cpu'))
            # print(_map_mask)
            for j in range(0, PLOT_SIZE):  # int(len(xs)/1000)
                cv2.circle(self.cost_map_color_vis, ((int)(xs[j]), (int)(
                    ys[j])), 1, (int(_map_mask[j]), 255-int(_map_mask[j]), 0), 1)

        # calculate speed cost
        v = torch.sqrt(torch.square(vx)+torch.square(vy))
        # print(v)
        # * 0.9 ** ((40 - t)*self.dt)
        speed_cost = torch.square(self.params["desired_v"] - v)

        # calculate stabilizing cost
        slip_angle = torch.abs(torch.atan(vy/torch.abs(vx)))
        # make 1 where slip angle larger than 0.75, otherwise 0
        stabilizing_cost = torch.square(
            slip_angle) + torch.where(slip_angle > self.thr_slip_pen, 1.0, 0.0) * self.w_slip_pen  # slip angle penalty

        # temp, _ = torch.sort(slip_angle, descending=True)
        # print(temp)

        cost = self.params["w_track"] * track_cost * self.time_decay**(t*self.dt) + self.params["w_speed"] * \
            speed_cost + self.params["w_stabil"] * 1000 * stabilizing_cost

        # print(track_cost, speed_cost)
        # penalize the entire trajectory (permit road edge = 127.5)
        return cost, map_mask

    def meter2grid_conversion(self, x, y):
        # both accept scalar state and tensor state
        global index_start_x, index_start_y

        _x = (x-self.global_start_x)*METER2GRID+index_start_x
        _y = -1*(y-self.global_start_y)*METER2GRID+index_start_y

        return _x, _y


def main(args=None):
    rclpy.init(args=args)
    node = DeployClass()

    # ROS Publishers
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
