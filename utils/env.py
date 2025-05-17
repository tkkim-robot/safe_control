import numpy as np
class Env:
    def __init__(self, width=20.0, height=20.0, resolution=0.1, known_obs=[]):
        self.width = width
        self.height = height
        # print("asdfasdf"*100)
        self.resolution = resolution  # meters per cell
        self.x_range = (0, width)
        self.y_range = (0, height)
        self.obs_boundary = self.set_obs_boundary(width, height)
        self.obs_circle = self.set_obs_circle(known_obs)
        self.obs_rectangle = self.set_obs_rectangle()
        self._discretize_map()

    def _discretize_map(self):
        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)

    def get_map_shape(self):
        return (self.grid_height, self.grid_width)
    
    def f_to_grid(self, points):
        points = np.array(points)
        original_shape = points.shape

        if points.ndim == 1:
            points = points.reshape(1, -1)

        grid_points = (points / self.resolution).astype(int)

        if original_shape == (2,):
            return grid_points[0]  # Return a 1D array for a single input point
        return grid_points

    def grid_to_f(self, grid_points):
        grid_points = np.array(grid_points)
        original_shape = grid_points.shape

        if grid_points.ndim == 1:
            grid_points = grid_points.reshape(1, -1)

        points = (grid_points * self.resolution) + (self.resolution / 2)

        if original_shape == (2,):
            return points[0]  # Return a 1D array for a single input point
        return points

    @staticmethod
    def set_obs_boundary(width, height):  # circle
        w = width
        h = height
        linewidth = 0.05
        obs_boundary = [
            [0, 0, linewidth, h],
            [0, h, w, linewidth],
            [linewidth, 0, w, linewidth],
            [w, linewidth, linewidth, h]
        ]
        return obs_boundary

    @staticmethod
    def set_obs_rectangle():
        # obs_rectangle = [
        #     [14, 12, 8, 2],
        #     [18, 22, 8, 3],
        #     [26, 7, 2, 12],
        #     [32, 14, 10, 2]
        # ]
        obs_rectangle = [[5, 5, 10, 10]]
        obs_rectangle = []
        return obs_rectangle
    @staticmethod
    def set_obs_circle(known_obs):
        # obs_cir = [[18, 18, 1]]
        # obs_cir = []
        obs_cir = known_obs
        return obs_cir