import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

from collections import deque

import copy


class GridWorld:
    class GridAgent:
        def __init__(self, loc=(1, 1, 0), dir="E"):
            self.loc, self.dir = loc, dir

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

        agey = self.GridAgent()

        agent_view = self.get_agent_view(agey)

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
            self.not_visible.append(self.GridObject((x,y,z), is_translucent=False, is_movable=False))

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
        for y_ind, ground_line in enumerate(ground_layer):
            for x_ind, grid_ground in enumerate(ground_line):
                ground_col = self.val_to_col["ground"][grid_ground.state]
                rect = Rectangle(
                    (x_ind, y_ind),
                    self.world_width,
                    self.world_height,
                    facecolor=ground_col,
                    edgecolor="black",
                )
                ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=0)
        ## Agent View
        if agent_view:
            for loc in agent_view:
                rect = Rectangle(
                    (loc[0], loc[1]),
                    self.world_width,
                    self.world_height,
                    facecolor="blue",
                    edgecolor="black",
                )
                ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=loc[2])
        ## Not Visible ////// GridElement
        for grid_object in self.not_visible:
            rect = Rectangle(
                (grid_object.loc[0], grid_object.loc[1]),
                self.world_width,
                self.world_height,
                facecolor="black",
                edgecolor="black",
            )
            ax.add_patch(rect)
            art3d.pathpatch_2d_to_3d(rect, z=grid_object.loc[2])
        ## Agent
        if agent_loc:
            rect = Rectangle(
                (agent_loc[0], agent_loc[1]),
                self.world_width,
                self.world_height,
                facecolor="gold",
                edgecolor="black",
            )
            ax.add_patch(rect)
            art3d.pathpatch_2d_to_3d(rect, z=0)

        ax.grid(False)
        ax.set_xlim3d(-self.world_width, self.world_width * (self.base_dim + 1))
        ax.set_ylim3d(-self.world_depth, self.world_depth * (self.base_dim + 1))
        ax.set_zlim3d(-self.world_height, self.world_height * (self.num_layers + 1))
        plt.show()

    def get_agent_view(self, agent):
        def view_ind_to_pos(view_ind, cur_loc, dir, dir_orth):
            x_view = cur_loc[0] + dir[0] + view_ind * dir_orth[0]
            y_view = cur_loc[1] + dir[1] + view_ind * dir_orth[1]
            return (x_view, y_view, cur_loc[2])

        dir = [0, 0, 0]
        if agent.dir in {"W", "E"}:
            dir[0] = -1 if agent.dir == "W" else 1
        if agent.dir in {"S", "N"}:
            dir[1] = -1 if agent.dir == "S" else 1
        dir_orth = [dir[1], dir[0], dir[2]]

        a_loc, cur_loc = agent.loc, agent.loc
        not_visible = set([obj.loc for obj in self.not_visible])
        view = set([cur_loc])
        view_width, d = 1, self.base_dim
        while cur_loc[0] >= 0 and cur_loc[1] >= 0 and cur_loc[0] < d and cur_loc[1] < d:

            ## Trace line of sight form previous line of sight row
            def is_viewable(loc):
                ## Not viewable if loc is OUT OF WORLD BOUND
                if loc[0] < 0 or loc[0] >= d or loc[1] < 0 or loc[1] >= d:
                    return True
                
                ## Vertical Tracing ##
                ######################
                ## Not viewable if we are above something non viewable
                if loc[2] > a_loc[2]:
                    if (loc[0], loc[1], loc[2]-1) in not_visible or (loc[0]-dir[0], loc[1]-dir[1], loc[2]-1) in not_visible:
                        not_visible.add(loc)
                        return True
                if loc[2] < a_loc[2]:
                    if (loc[0], loc[1], loc[2]+1) in not_visible or (loc[0]+dir[0], loc[1]+dir[1], loc[2]+1) in not_visible:
                        not_visible.add(loc)
                        return True
                ######################
                
                ## Horizontal Tracing ##
                ########################
                ## Not Viewable if we are along the line of sight of the agent and behind something non viewable
                trace_prec = (loc[0] - dir[0], loc[1] - dir[1], loc[2])
                rel_pos_dir = (dir[1], -dir[0], 0)
                if (
                    rel_pos_dir[0] * (loc[0] - a_loc[0]) == 0
                    and rel_pos_dir[1] * (loc[1] - a_loc[1]) == 0
                ):
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
                        loc[1] - dir[1] - rel_pos_dir[1], loc[2]-dir[2]
                    ) in not_visible and trace_prec not in view:
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
                        loc[1] - dir[1] + rel_pos_dir[1], loc[2]-dir[2]
                    ) in not_visible and trace_prec not in view:
                        not_visible.add(loc)
                        return True
                ########################
                
                ## If not we are viewable
                return False

            next_pos = []
            for layer_ind in range(self.num_layers):
                next_pos.extend( [
                    view_ind_to_pos(view_ind, (cur_loc[0], cur_loc[1], layer_ind), dir, dir_orth)
                    for view_ind in range(-view_width, view_width + 1)
                ] )

            next_pos = list(filter(lambda x: not is_viewable(x), next_pos))
            view.update(next_pos)

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
