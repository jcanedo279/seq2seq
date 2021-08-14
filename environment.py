import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import noise
import numpy as np
from PIL import Image

from collections import deque

import copy


class GridWorld:
    class GridAgent:
        def __init__(self, loc=(1, 1, 2), dir="N", world_size=(1, 1, 1)):
            self.loc, self.dir = loc, dir
            world_width, world_depth, world_height = world_size

            edgecolors = [
                "red" if card == dir else "black"
                for card in ["Top", "W", "E", "N", "S"]
            ]

            ## top, left, right, front, back
            self.rects = [
                Rectangle(
                    (loc[0] * world_width, loc[1] * world_depth),
                    world_width,
                    world_depth,
                    facecolor="gold",
                    edgecolor=edgecolors[0],
                ),
                Rectangle(
                    (loc[1] * world_depth, loc[2] * world_width),
                    world_depth,
                    world_height,
                    facecolor="gold",
                    edgecolor=edgecolors[1],
                ),
                Rectangle(
                    (loc[1] * world_depth, loc[2] * world_width),
                    world_depth,
                    world_height,
                    facecolor="gold",
                    edgecolor=edgecolors[2],
                ),
                Rectangle(
                    (loc[0] * world_width, loc[2] * world_width),
                    world_width,
                    world_height,
                    facecolor="gold",
                    edgecolor=edgecolors[3],
                ),
                Rectangle(
                    (loc[0] * world_width, loc[2] * world_width),
                    world_width,
                    world_height,
                    facecolor="gold",
                    edgecolor=edgecolors[4],
                ),
            ]
            self.zs = [
                loc[2] * world_height + world_height,
                loc[0] * world_width,
                loc[0] * world_width + world_width,
                loc[1] * world_depth,
                loc[1] * world_depth + world_depth,
            ]
            self.z_dirs = ["z", "x", "x", "y", "y"]

        def set_dir(self, dir):
            self.dir = dir

    class GridGround:
        def __init__(self, loc, state):
            self.loc = loc
            self.state = state

    class GridSpace:
        def __init__(self, loc):
            self.loc = loc

    class GridObject:
        def __init__(self, loc, is_translucent=False, is_movable=False):
            self.loc = loc
            self.is_translucent, self.is_movable = is_translucent, is_movable

    def __init__(
        self,
        base_dim=10,
        num_layers=4,
        num_ground_layers=2,
        max_agent_view_depth=5,
        layers=[],
    ):
        self.world_width, self.world_depth, self.world_height = 1, 1, 1
        self.max_agent_view_depth=max_agent_view_depth
        self.val_to_col = {"ground": {0: "red", 1: "green", 2: "grey"}}
        self.base_dim, self.num_layers, self.num_ground_layers = base_dim, num_layers, num_ground_layers
        self.gen_world(layers)

        agey = self.GridAgent(
            world_size=(self.world_width, self.world_depth, self.world_height)
        )

        self.agents = [agey]
        agent_view = self.gen_agent_view(agey)

        self.display_world(agent_view=agent_view, agent_loc=agey.loc)

    def gen_world(self, layers):
        self.world_layers = []

        noise_mat = self.gen_perlin_noise()

        self.world_layers = []
        for layer_ind in range(self.num_layers):
            mat = []
            for line_ind, mat_line in enumerate(noise_mat):
                line = []
                for val_ind, mat_val in enumerate(mat_line):
                    top_layer = noise_mat[line_ind][val_ind]
                    ground_val = (
                        0
                        if (
                            val_ind == 0
                            or line_ind == 0
                            or val_ind + 1 == self.base_dim
                            or line_ind + 1 == self.base_dim
                        )
                        else 1
                    )
                    if layer_ind == top_layer:
                        line.append(
                            self.GridGround((val_ind, line_ind, layer_ind), ground_val)
                        )
                    else:
                        line.append(self.GridSpace((val_ind, line_ind, layer_ind)))
                mat.append(line)
            self.world_layers.append(mat)

        blocked_items = {(3, 1, 2), (3, 3, 2), (4, 3, 2)}
        self.not_visible = []
        for x, y, z in blocked_items:
            self.not_visible.append(
                self.GridObject((x, y, z), is_translucent=False, is_movable=False)
            )

    def display_world(self, agent_view=None, agent_loc=None):
        fig = plt.figure()
        ax = fig.gca(projection="3d")

        ## Ground
        self.display_ground(ax)
        ## Agent View
        self.display_agent_views(ax)
        ## Not Visible ////// GridElement
        self.display_elements(ax)
        ## Agent
        self.display_agents(ax)

        ax.grid(False)
        max_bound = max(
            self.world_width * (self.base_dim + 1) - self.world_width,
            self.world_depth * (self.base_dim + 1) - self.world_depth,
            self.world_height * (self.num_layers + 1) - self.world_height,
        )
        ax.set_xlim3d(-self.world_width, -self.world_width + max_bound)
        ax.set_ylim3d(-self.world_depth, -self.world_depth + max_bound)
        ax.set_zlim3d(-self.world_height, -self.world_height + max_bound)
        plt.show()

    #####################
    ### Display Block ###
    #####################
    def display_block(self, loc, facecolor="Black", edgecolor="Black", alpha=1):
        faces, zs, z_dirs = [], [], []
        ## Gather adjacent elements
        top_elem = (
            self.world_layers[loc[2] + 1][loc[1]][loc[0]]
            if loc[2] + 1 != self.num_layers
            else None
        )
        bottom_elem = (
            self.world_layers[loc[2] - 1][loc[2]][loc[0]] if loc[2] != 0 else None
        )
        left_elem = (
            self.world_layers[loc[2]][loc[1]][loc[0] - 1] if loc[0] != 0 else None
        )
        right_elem = (
            self.world_layers[loc[2]][loc[1]][loc[0] + 1]
            if loc[0] + 1 < self.base_dim
            else None
        )
        back_elem = (
            self.world_layers[loc[2]][loc[1] - 1][loc[0]] if loc[1] != 0 else None
        )
        front_elem = (
            self.world_layers[loc[2]][loc[1] + 1][loc[0]]
            if loc[1] + 1 < self.base_dim
            else None
        )
        ### Top Element ###
        if top_elem != None or type(top_elem) != self.GridGround:
            faces.append(
                Rectangle(
                    (loc[0], loc[1]),
                    self.world_width,
                    self.world_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                )
            )
            zs.append((loc[2] + 1) * self.world_height)
            z_dirs.append("z")
        ### Bottom Element ###
        if bottom_elem != None or type(bottom_elem) != self.GridGround:
            faces.append(
                Rectangle(
                    (loc[0], loc[1]),
                    self.world_width,
                    self.world_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                )
            )
            zs.append(loc[2] * self.world_height)
            z_dirs.append("z")
        ### Left Element ###
        if left_elem != None or type(left_elem) != self.GridGround:
            faces.append(
                Rectangle(
                    (loc[1], loc[2]),
                    self.world_depth,
                    self.world_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                )
            )
            zs.append(loc[0] * self.world_depth)
            z_dirs.append("x")
        ### Right Element ###
        if right_elem != None or type(right_elem) != self.GridGround:
            faces.append(
                Rectangle(
                    (loc[1], loc[2]),
                    self.world_depth,
                    self.world_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                )
            )
            zs.append((loc[0] + 1) * self.world_depth)
            z_dirs.append("x")
        ### Back Element ###
        if back_elem != None or type(back_elem) != self.GridGround:
            faces.append(
                Rectangle(
                    (loc[0], loc[2]),
                    self.world_width,
                    self.world_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                )
            )
            zs.append((loc[1]) * self.world_depth)
            z_dirs.append("y")
        ### Front Element ###
        if front_elem != None or type(front_elem) != self.GridGround:
            faces.append(
                Rectangle(
                    (loc[0], loc[2]),
                    self.world_width,
                    self.world_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                )
            )
            zs.append(((loc[1]) + 1) * self.world_depth)
            z_dirs.append("y")
        return faces, zs, z_dirs

    def display_ground(self, ax):
        for z_ind, layer in enumerate(self.world_layers):
            for y_ind, ground_line in enumerate(layer):
                for x_ind, grid_elem in enumerate(ground_line):
                    if type(grid_elem) == self.GridGround:
                        facecolor = self.val_to_col["ground"][grid_elem.state]
                        faces, zs, z_dirs = self.display_block(
                            (x_ind, y_ind, z_ind), facecolor=facecolor
                        )
                        for rect, z, z_dir in zip(faces, zs, z_dirs):
                            ax.add_patch(rect)
                            art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)

    def display_elements(self, ax):
        for grid_object in self.not_visible:
            faces, zs, z_dirs = self.display_block(
                (grid_object.loc[0], grid_object.loc[1], grid_object.loc[2]),
                facecolor="black",
            )
            for rect, z, z_dir in zip(faces, zs, z_dirs):
                ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)

    def display_agent_views(self, ax):
        colors = ["blue", "purple", "yellow", "orange", "pink"]
        for agent in self.agents:
            agent_color = colors.pop()
            agent_view = self.gen_agent_view(agent)
            sweep_step = 0.5 / (len(agent_view))
            for sweep_ind, view_sweep in enumerate(agent_view):
                cur_step = 0.5 - sweep_step * sweep_ind
                for loc in view_sweep:
                    if loc == agent.loc:
                        continue
                    faces, zs, z_dirs = self.display_block(
                        (loc[0], loc[1], loc[2]),
                        facecolor=agent_color,
                        edgecolor=agent_color,
                        alpha=cur_step,
                    )
                    for rect, z, z_dir in zip(faces, zs, z_dirs):
                        ax.add_patch(rect)
                        art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)

    def display_agents(self, ax):
        for agent in self.agents:
            faces, zs, z_dirs = self.display_block(
                (agent.loc[0], agent.loc[1], agent.loc[2]), facecolor="gold"
            )
            for rect, z, z_dir in zip(faces, zs, z_dirs):
                ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)

    def gen_agent_view(self, agent):
        def view_ind_to_pos(view_ind, cur_loc, dir, dir_orth):
            x_view = cur_loc[0] + dir[0] + view_ind * dir_orth[0]
            y_view = cur_loc[1] + dir[1] + view_ind * dir_orth[1]
            return (x_view, y_view, cur_loc[2] * self.world_height)

        ## Calculate agent direction
        dir = [0, 0, 0]
        if agent.dir in {"W", "E"}:
            dir[0] = -1 if agent.dir == "W" else 1
        if agent.dir in {"S", "N"}:
            dir[1] = -1 if agent.dir == "S" else 1
        ## Calculate orthogonal agent direction
        dir_orth = [dir[1], dir[0], dir[2]]
        ## Calculate agent direction dot world_size
        weighted_dir = (
            [dir[0] * self.world_width, dir[1] * self.world_depth, dir[2]]
            if agent.dir in {"N", "S"}
            else []
        )
        weighted_dir = (
            [dir[0] * self.world_depth, dir[1] * self.world_width, dir[2]]
            if agent.dir in {"W", "E"}
            else weighted_dir
        )
        ## Calculate orthogonal agent direction dot world_size
        w_orth_dir = (
            [
                dir_orth[0] * self.world_width,
                dir_orth[1] * self.world_depth,
                dir_orth[2],
            ]
            if agent.dir in {"N", "S"}
            else []
        )
        w_orth_dir = (
            [
                dir_orth[0] * self.world_depth,
                dir_orth[1] * self.world_width,
                dir_orth[2],
            ]
            if agent.dir in {"E", "W"}
            else w_orth_dir
        )

        a_loc, cur_loc = agent.loc, agent.loc
        not_visible = set([obj.loc for obj in self.not_visible])
        view_set = set([cur_loc])
        view = [[cur_loc]]

        view_width, d = 1, self.base_dim
        while cur_loc[0] >= 0 and cur_loc[1] >= 0 and cur_loc[0] < d and cur_loc[1] < d and view_width <= self.max_agent_view_depth:

            ## Trace line of sight sweep from previous line of sight sweep
            def is_viewable(loc):
                loc_below = (loc[0], loc[1], loc[2] - 1)
                loc_below_prec = (loc[0] - dir[0], loc[1] - dir[1], loc[2] - 1)
                loc_above = (loc[0], loc[1], loc[2] + 1)
                loc_above_prec = (loc[0] - dir[0], loc[1] - dir[1], loc[2] + 1)
                loc_prec = (loc[0] - dir[0], loc[1] - dir[1], loc[2])

                rel_pos_dir = (dir[1], -dir[0], 0)
                loc_left_prec = (
                    loc[0] - dir[0] - rel_pos_dir[0],
                    loc[1] - dir[1] - rel_pos_dir[1],
                    loc[2] - dir[2],
                )
                loc_right_prec = (
                    loc[0] - dir[0] + rel_pos_dir[0],
                    loc[1] - dir[1] + rel_pos_dir[1],
                    loc[2] - dir[2],
                )
                ## Not viewable if loc is out of world bounds
                if (
                    loc[0] < 0
                    or loc[0] >= self.base_dim
                    or loc[1] < 0
                    or loc[1] >= self.base_dim
                ):
                    return True
                ####### ## Vertical Tracing ##
                if loc[2] > a_loc[2]:
                    ## Iterate over cells between current block and blocks below, not visible if below not visible or ground
                    for layer_ind in range(max(a_loc[2], 0), loc[2] + 1):
                        if (
                            (loc[0], loc[1], layer_ind) in not_visible
                            or (loc_below_prec[0], loc_below_prec[1], layer_ind)
                            in not_visible
                            or type(self.world_layers[layer_ind][loc[1]][loc[0]])
                            == self.GridGround
                            or type(
                                self.world_layers[layer_ind][loc_below_prec[1]][
                                    loc_below_prec[0]
                                ]
                            )
                            == self.GridGround
                        ):
                            not_visible.add(loc)
                            return True
                if loc[2] < a_loc[2]:
                    ## Iterate over cells between current block and blocks above, not visible if above not visible or ground
                    for layer_ind in range(loc[2], min(a_loc[2] + 1, self.num_layers)):
                        if (
                            (loc[0], loc[1], layer_ind) in not_visible
                            or (loc_above_prec[0], loc_above_prec[1], layer_ind)
                            in not_visible
                            or type(self.world_layers[layer_ind][loc[1]][loc[0]])
                            == self.GridGround
                            or type(
                                self.world_layers[layer_ind][loc_above_prec[1]][loc_above_prec[0]]
                            )
                            == self.GridGround
                        ):
                            not_visible.add(loc)
                            return True
                ####### ## Horizontal Tracing ##
                ## Not viewable if we are behind something not viewable
                if loc_prec in not_visible:
                    not_visible.add(loc)
                    return True
                ## Not viewable if we are to the right of the agent's line of sight and to the right of something non viewable
                if (
                    rel_pos_dir[0] * (loc[0] - a_loc[0])
                    + rel_pos_dir[1] * (loc[1] - a_loc[1])
                    > 0
                ):
                    if loc_left_prec in not_visible and loc_prec not in view_set:
                        not_visible.add(loc)
                        return True
                ## Not viewable if we are to the left of the agent's line of sight and to the left of something non viewable
                if (
                    rel_pos_dir[0] * (loc[0] - a_loc[0])
                    + rel_pos_dir[1] * (loc[1] - a_loc[1])
                    < 0
                ):
                    if loc_right_prec in not_visible and loc_prec not in view_set:
                        not_visible.add(loc)
                        return True
                ## Not viewable if under ground
                if (
                    loc[2] + 1 == self.num_layers
                    and type(self.world_layers[loc[2]][loc[1]][loc[0]])
                    != self.GridSpace
                ):
                    not_visible.add(loc)
                    return True
                if (
                    loc[2] + 1 < self.num_layers
                    and type(self.world_layers[loc[2] + 1][loc[1]][loc[0]])
                    != self.GridSpace
                ):
                    not_visible.add(loc)
                    return True
                ## Not viewable if ground block
                if (
                    type(self.world_layers[loc[2]][loc[1]][loc[0]]) == self.GridGround
                    and type(self.world_layers[loc[2] + 1][loc[1]][loc[0]])
                    == self.GridSpace
                ):
                    return True
                ## If not we are viewable
                return False

            next_pos = []
            for layer_ind in range(self.num_layers):
                next_pos.extend(
                    [
                        view_ind_to_pos(
                            view_ind,
                            (cur_loc[0], cur_loc[1], layer_ind),
                            dir,
                            dir_orth,
                        )
                        for view_ind in range(-view_width, view_width + 1)
                    ]
                )

            next_pos = list(filter(lambda x: not is_viewable(x), next_pos))

            if not next_pos:
                break

            view_set.update(next_pos)
            view.append(next_pos)

            view_width += 1

            cur_loc = [cur_loc[0] + dir[0], cur_loc[1] + dir[1]]
        return view

    def gen_perlin_noise(self):
        shape = (self.base_dim, self.base_dim)
        ## altitude? 0 -> 1
        scale = 0.5
        ## number of passes, each pass adds more detail
        octaves = 2
        persistence = 1
        ## detail addedd per pass
        lacunarity = 2
        # seed = np.random.randint(0,100)
        seed = 0

        world = np.zeros(shape)

        # make coordinate grid on [0,1]^2
        x_idx = np.linspace(0, 1, shape[0])
        y_idx = np.linspace(0, 1, shape[1])
        world_x, world_y = np.meshgrid(x_idx, y_idx)

        # apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
        world = np.vectorize(noise.pnoise2)(
            world_x / scale,
            world_y / scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            repeatx=self.base_dim,
            repeaty=self.base_dim,
            base=seed,
        )

        # here was the error: one needs to normalize the image first. Could be done without copying the array, though
        grid = np.floor((world + 0.5) * 255).astype(
            np.uint8
        )  # <- Normalize world first

        norm_grid = (self.num_ground_layers * (grid / np.max(grid)) - 0.001).astype(int)
        return norm_grid


dim = 10
state0 = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 2, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
# state1 = [
#     [-1, -1, -1, -1, -1, -1, -1],
#     [-1, 1, -1, -1, -1, -1, -1],
#     [-1, 1, -1, -1, -1, -1, -1],
#     [-1, 1, -1, -1, -1, -1, -1],
#     [-1, 1, 1, -1, -1, -1, -1],
#     [-1, 1, 1, 1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1],
# ]
layers = [state0]

griddy = GridWorld(dim, layers=layers)

grid = griddy.gen_perlin_noise()
