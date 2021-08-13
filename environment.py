import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

from collections import deque

import copy


class GridWorld:
    class GridAgent:
        def __init__(self, loc=(1, 1, 1), dir="N", world_size=(1, 1, 1)):
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
        num_layers=3,
        ground_state=[],
    ):
        self.world_width, self.world_depth, self.world_height = 1, 1, 1
        self.val_to_col = {"ground": {0: "red", 1: "green", 2: "grey"}}
        self.base_dim, self.num_layers = base_dim, num_layers
        self.gen_world(ground_state)

        agey = self.GridAgent(
            world_size=(self.world_width, self.world_depth, self.world_height)
        )

        self.agents = [agey]
        agent_view = self.gen_agent_view(agey)

        self.display_world(agent_view=agent_view, agent_loc=agey.loc)

    def gen_world(self, ground_state):
        self.world_layers = []

        self.ground_layer = []
        for line_ind, ground_line in enumerate(ground_state):
            line = []
            for val_ind, ground_val in enumerate(ground_line):
                line.append(self.GridGround((val_ind, line_ind, 0), ground_val))
            self.ground_layer.append(line)
        self.world_layers.append(self.ground_layer)

        blocked_items = {(3, 1, 1), (3, 3, 1), (4, 3, 1)}
        self.not_visible = []
        for x, y, z in blocked_items:
            self.not_visible.append(
                self.GridObject((x, y, z), is_translucent=False, is_movable=False)
            )

        for layer_ind in range(self.num_layers - 1):
            layer = []
            for line_ind in range(self.base_dim):
                line = []
                for val_ind in range(self.base_dim):
                    line.append(self.GridSpace((val_ind, line_ind, layer_ind)))
                layer.append(line)
            self.world_layers.append(layer)

    def display_world(self, agent_view=None, agent_loc=None):
        fig = plt.figure()
        ax = fig.gca(projection="3d")

        ground_layer = self.world_layers[0]
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
        top_elem = self.world_layers[loc[2]+1][loc[1]][loc[0]] if loc[2] + 1 != self.num_layers else None
        bottom_elem = self.world_layers[loc[2]-1][loc[2]][loc[0]] if loc[2] != 0 else None
        left_elem = self.world_layers[loc[2]][loc[1]][loc[0]-1] if loc[0] != 0 else None
        right_elem = self.world_layers[loc[2]][loc[1]][loc[0]+1] if loc[0]+1 < self.base_dim else None
        back_elem = self.world_layers[loc[2]][loc[1]-1][loc[0]] if loc[1] != 0 else None
        front_elem = self.world_layers[loc[2]][loc[1]+1][loc[0]] if loc[1]+1 < self.base_dim else None
        ### Top Element ###
        if (
            top_elem != None
            or type(top_elem) != self.GridGround
        ):
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
        if (
            bottom_elem != None
            or type(bottom_elem) != self.GridGround
        ):
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
        if (
            left_elem != None
            or type(left_elem) != self.GridGround
        ):
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
            zs.append((loc[0]+1) * self.world_depth)
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
            zs.append(((loc[1])+1) * self.world_depth)
            z_dirs.append("y")
        return faces, zs, z_dirs

    def display_ground(self, ax):
        for z_ind, layer in enumerate(self.world_layers):
            for y_ind, ground_line in enumerate(layer):
                for x_ind, grid_elem in enumerate(ground_line):
                    if type(grid_elem) == self.GridGround:
                        facecolor = self.val_to_col['ground'][grid_elem.state]
                        faces, zs, z_dirs = self.display_block((x_ind, y_ind, z_ind), facecolor=facecolor)
                        for rect, z, z_dir in zip(faces, zs, z_dirs):
                            ax.add_patch(rect)
                            art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)
                            
    def display_elements(self, ax):
        for grid_object in self.not_visible:
            faces, zs, z_dirs = self.display_block((grid_object.loc[0], grid_object.loc[1], grid_object.loc[2]), facecolor="black")
            for rect, z, z_dir in zip(faces, zs, z_dirs):
                ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)
                
    def display_agent_views(self, ax):
        colors = ["blue", "purple", "yellow", "orange", "pink"]
        for agent in self.agents:
            agent_color = colors.pop()
            agent_view = self.gen_agent_view(agent)
            sweep_step = 0.5/(len(agent_view)-1)
            for sweep_ind, view_sweep in enumerate(agent_view):
                cur_step = 0.5 + sweep_step*sweep_ind
                for loc in view_sweep:
                    if loc == agent.loc:
                        continue
                    faces, zs, z_dirs = self.display_block((loc[0], loc[1], loc[2]), facecolor=agent_color, edgecolor=agent_color, alpha=cur_step)
                    for rect, z, z_dir in zip(faces, zs, z_dirs):
                        ax.add_patch(rect)
                        art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)
                    
    def display_agents(self, ax):
        for agent in self.agents:
            faces, zs, z_dirs = self.display_block((agent.loc[0], agent.loc[1], agent.loc[2]), facecolor='gold')
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
        while cur_loc[0] >= 0 and cur_loc[1] >= 0 and cur_loc[0] < d and cur_loc[1] < d:

            ## Trace line of sight sweep from previous line of sight sweep
            def is_viewable(loc):
                ## Not viewable if loc is out of world bounds
                if (
                    loc[0] < 0
                    or loc[0] >= self.base_dim
                    or loc[1] < 0
                    or loc[1] >= self.base_dim
                ):
                    return True
                ####### ## Vertical Tracing ##
                ## Not viewable if we are above something non viewable
                if loc[2] > a_loc[2]:
                    if (loc[0], loc[1], loc[2] - 1) in not_visible or (
                        loc[0] - dir[0],
                        loc[1] - dir[1],
                        loc[2] - 1,
                    ) in not_visible:
                        not_visible.add(loc)
                        return True
                if loc[2] < a_loc[2]:
                    if (loc[0], loc[1], loc[2] + 1) in not_visible or (
                        loc[0] + dir[0],
                        loc[1] + dir[1],
                        loc[2] + 1,
                    ) in not_visible:
                        not_visible.add(loc)
                        return True
                ####### ## Horizontal Tracing ##
                trace_prec = (loc[0] - dir[0], loc[1] - dir[1], loc[2])
                rel_pos_dir = (dir[1], -dir[0], 0)
                ## Not viewable if we are behind something not viewable
                if trace_prec in not_visible:
                    not_visible.add(loc)
                    return True
                ## Not viewable if we are to the right of the agent's line of sight and to the right of something non viewable
                if (
                    rel_pos_dir[0] * (loc[0] - a_loc[0])
                    + rel_pos_dir[1] * (loc[1] - a_loc[1])
                    > 0
                ):
                    if (
                        loc[0] - dir[0] - rel_pos_dir[0],
                        loc[1] - dir[1] - rel_pos_dir[1],
                        loc[2] - dir[2],
                    ) in not_visible and trace_prec not in view_set:
                        not_visible.add(loc)
                        return True
                ## Not viewable if we are to the left of the agent's line of sight and to the left of something non viewable
                if (
                    rel_pos_dir[0] * (loc[0] - a_loc[0])
                    + rel_pos_dir[1] * (loc[1] - a_loc[1])
                    < 0
                ):
                    if (
                        loc[0] - dir[0] + rel_pos_dir[0],
                        loc[1] - dir[1] + rel_pos_dir[1],
                        loc[2] - dir[2],
                    ) in not_visible and trace_prec not in view_set:
                        not_visible.add(loc)
                        return True
                ## Not viewable if under ground
                if (
                    loc[2] + 1 == self.num_layers
                    and type(self.world_layers[loc[2]][loc[1]][loc[0]])
                    != self.GridSpace
                ):
                    return True
                if (
                    loc[2] + 1 < self.num_layers
                    and type(self.world_layers[loc[2] + 1][loc[1]][loc[0]])
                    != self.GridSpace
                ):
                    return True
                ## Not viewable if ground block
                if type(self.world_layers[loc[2]][loc[1]][loc[0]]) == self.GridGround:
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


dim = 7
state = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 2, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
]

griddy = GridWorld(dim, ground_state=state)
