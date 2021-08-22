import os
import time
import math
import random as rd
import copy


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

import noise
import numpy as np
from PIL import Image

from collections import deque

import copy



class GridWorld:
    class GridAgent:
        def __init__(self, loc=(1, 1, 2), dir="E", world_scale=(1, 1, 1)):
            ## GridWorld parameters
            self.loc, self.dir = loc, dir
            self.world_scale = world_scale
            
            cards = ["TOP", "W", "E", "S", "N"] # <- cardinal dirs
            world_width, world_depth, world_height = world_scale
            self.sizes = [
                (world_width, world_depth),
                (world_depth, world_height),
                (world_depth, world_height),
                (world_width, world_height),
                (world_width,  world_height)
            ]
            self.colors = [('blue', 'blue') if card==self.dir else ('gold', 'black') for card in cards]
            self.z_dirs = ["z", "x", "x", "y", "y"]
            
            self.update_faces()
            
        def update_faces(self):
            loc = self.loc
            world_width, world_depth, world_height = self.world_scale
            locs = [
                (loc[0] * world_width, loc[1] * world_depth), 
                (loc[1] * world_depth, loc[2] * world_height),
                (loc[1] * world_depth, loc[2] * world_height),
                (loc[0] * world_width, loc[2] * world_height),
                (loc[0] * world_width, loc[2] * world_height),
            ]
            
            self.faces = [Rectangle(loc, size[0], size[1], facecolor=color[0], edgecolor=color[1]) for loc, size, color in zip(locs, self.sizes, self.colors)]
            
            self.zs = [
                (loc[2]+1) * world_height,
                loc[0] * world_width,
                (loc[0]+1) * world_width,
                loc[1] * world_depth,
                (loc[1]+1) * world_depth,
            ]

    class GridElement:
        def __init__(self, loc, state, is_transparent, is_movable, label=None):
            self.loc, self.state = loc, state
            self.is_transparent, self.is_movable = is_transparent, is_movable
            self.label=label


    def __init__(
        self,
        config
    ):
        self.continue_animation = True
        self.fps = 1
        
        self.val_to_col = {"ground": {0: "red", 1: "green", 2: "grey"}}
        
        
        self.right_dir_seq = {"N":"E", "E":"S", "S":"W", "W":"N"}
        self.left_dir_seq = {"N":"W", "W":"S", "S":"E", "E":"N"}
        self.turn_dir_seq = {"N":"S", "S":"N", "E":"W", "W":"E"}
        
        self.out_to_orth_dir_vect = {0: (1,1,1), 1: (-1,1,1), 2: (1,-1,1), 3: (-1,-1,1)}
        self.dir_to_dir_vect = {"E": (1,0,0), "W": (-1,0,0), "N": (1,0,0), "S":(-1,0,0)}
        
        self.gen_ind = 0
        self.config = config
        
        current_time = time.localtime()
        fitted_time_arr = time.strftime('%a, %d %b %Y %H:%M:%S GMT', current_time).split(' ')
        local_time = fitted_time_arr[4]
        local_date = f'{fitted_time_arr[2]}:{fitted_time_arr[1]}'
        
        self.output_path = f'out/environment_{local_date}_{local_time}.gif'
        

    def make_animation(self):
        self.animating_figure = plt.figure()
        # self.ax = plt.gca(projection="3d")
        self.ax = plt.axes(projection="3d")
        self.ax.axis('auto')
        
        self.gen_world()
        
        self.gen_agents()
        
        # self.pop_iter(display=True)
        
        self.anim = FuncAnimation(self.animating_figure, self.update_animation,
                                    frames=self.gen_frames, init_func=self.init_plot(), repeat=False)
        self.anim.save(self.output_path, writer=PillowWriter(fps=self.fps))
        
        plt.close('all')
        del self.animating_figure
        
    
    def init_plot(self):
        self.ax.cla()
        
    def gen_frames(self):
        while (self.gen_ind < self.config['agent_config']['num_generations']) and self.continue_animation:
            self.ax.cla()
            yield self.gen_ind
        yield StopIteration
        
    def update_animation(self, i):
        print(f"\ranimating timestamp: {i}", end='')
        # self.ax.cla()

        agent_scores = self.hide_and_seek_score_rel()
        outputs = self.iter_agents_predict()
        self.update_state(outputs)
        for agent_ind, agent in enumerate(self.agents):
            agent.view = self.gen_agent_view(agent)
        self.display_world()

        # # Format animation title
        # title = f'dim={self.dim}, size={self.size}, gen={self.ptIndex}'
        # self.ax.set_title(title)
        # if not self.silence:
        #     print(f'Generation {self.gen_ind} complete', end='\r')
        self.gen_ind += 1
        return self.ax
    
    def pop_iter(self, display=False):
        num_generations = self.config['agent_config']['num_generations']
        
        for gen_ind in range(num_generations):
        
            agent_scores = self.hide_and_seek_score_rel()
            outputs = self.iter_agents_predict()
            self.update_state(outputs)
            for ind, agent in enumerate(self.agents):
                agent.view = self.gen_agent_view(agent)
            if display:
                self.display_world()
        
    def hide_and_seek_score_raw(self):
        agent_loc_to_ind = {agent.loc: agent_ind for agent_ind, agent in enumerate(self.agents)}
        agent_scores = {
            'scores': {agent.loc: 0 for agent in self.agents},
            'penalties': {agent.loc: 0 for agent in self.agents}
        }
        for view_agent in self.agents:
            view_agent_view = view_agent.view
            for view_sweep in view_agent_view:
                for loc in view_sweep:
                    if loc in agent_loc_to_ind and loc != view_agent.loc:
                        agent_scores['scores'][view_agent.loc] += 1
                        agent_scores['penalties'][loc] += 1
                        
        diff_scores = [agent_scores['scores'][agent.loc]-agent_scores['penalties'][agent.loc] for agent in self.agents]
        min_diff_score = min(diff_scores)
        norm_diff_scores = [diff_score-min_diff_score for diff_score in diff_scores] if min_diff_score != 0 else diff_scores
        return norm_diff_scores
        
        
    def hide_and_seek_score_rel(self):
        agent_loc_to_ind = {agent.loc: agent_ind for agent_ind, agent in enumerate(self.agents)}
        agent_scores = {
            'scores': {agent.loc: 0 for agent in self.agents},
            'penalties': {agent.loc: 0 for agent in self.agents}
        }
        for view_agent in self.agents:
            view_agent_view = view_agent.view
            for view_sweep in view_agent_view:
                for loc in view_sweep:
                    if loc in agent_loc_to_ind and loc != view_agent.loc:
                        agent_scores['scores'][view_agent.loc] += 1
                        agent_scores['penalties'][loc] += 1
                        
        diff_scores = [agent_scores['scores'][agent.loc]-agent_scores['penalties'][agent.loc] for agent in self.agents]
        min_diff_score = min(diff_scores)
        norm_diff_scores = [diff_score-min_diff_score for diff_score in diff_scores] if min_diff_score != 0 else diff_scores
        
        eul_scores = [(1-math.exp(-score)) for score in norm_diff_scores]
        eul_score_sum = sum(eul_scores)
        
        eul_norm_scores = [eul_score/eul_score_sum for eul_score in eul_scores] if eul_score_sum != 0 else [0]*len(eul_scores) 
        return eul_norm_scores
    
    def iter_agents_predict(self):
        outputs = []
        for agent in self.agents:
            rand_dir = rd.choice([0, 1, 2, 3])
            rand_loc = [rd.choice([-1,0,1]) for _ in range(3)]
            outputs.append([rand_dir, rand_loc, 0])
        return outputs
            
    
    ## 
    #  output[0] = {'', 'left', 'right', 'turn'}
    #  output[1] = [{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}]
    #  output[2] = {'', 'grab', 'release'}
    def update_state(self, outputs):
        world_dim = self.config['world_config']['world_dim']
        for output, agent in zip(outputs, self.agents):
            out_dir, out_loc, out_state = output
            new_orth_dir_vect = self.out_to_orth_dir_vect[out_dir]
            new_dir_vect = (new_orth_dir_vect[0]*agent.dir_vect[0], new_orth_dir_vect[1]*agent.dir_vect[1], new_orth_dir_vect[2]*agent.dir_vect[2])
            
            new_loc = (agent.loc[0]+out_loc[0], agent.loc[1]+out_loc[1], agent.loc[2]+out_loc[2])
            ## catch out of bounds
            new_loc = (max(0,min(new_loc[0],world_dim[0]-1)), max(0,min(new_loc[1],world_dim[1]-1)), max(0,min(new_loc[2],world_dim[2]-1)))
            ## bind agent to surface 
            new_loc = new_loc if new_loc[2] == self.ground_grid[new_loc[1]][new_loc[0]]+1 else agent.loc
            
            new_dir = agent.dir
            if out_dir == 1:
                new_dir = self.left_dir_seq[new_dir]
            elif out_dir == 2:
                new_dir = self.right_dir_seq[new_dir]
            elif out_dir == 3:
                new_dir = self.turn_dir_seq[new_dir]
            
            agent.dir = new_dir
            agent.dir_vect = new_dir_vect
            agent.loc = new_loc
            
            agent.update_faces()


    def gen_world(self):
        world_dim = self.config['world_config']['world_dim']
        
        self.world_layers = []

        self.gen_perlin_noise()

        self.world_layers = []
        for layer_ind in range(world_dim[2]):
            mat = []
            for line_ind, mat_line in enumerate(self.ground_grid):
                line = []
                for val_ind, mat_val in enumerate(mat_line):
                    ground_val = (
                        0
                        if (
                            val_ind == 0
                            or line_ind == 0
                            or val_ind + 1 == world_dim[0]
                            or line_ind + 1 == world_dim[1]
                        )
                        else 1
                    )
                    if layer_ind == mat_val:
                        line.append(
                            self.GridElement((val_ind, line_ind, layer_ind), ground_val, False, False, label='ground')
                        )
                    else:
                        line.append(self.GridElement((val_ind, line_ind, layer_ind), -1, True, False, label='space'))
                mat.append(line)
            self.world_layers.append(mat)

        # blocked_items = {(3, 1, 2), (3, 3, 2), (4, 3, 2)}
        blocked_items = {}
        self.not_visible = []
        for x, y, z in blocked_items:
            self.not_visible.append(
                self.GridElement((x, y, z), -2, False, True)
            )
            
            
    def gen_agents(self):
        world_dim = self.config['world_config']['world_dim']
        self.agents = []
        dirs = ['N', 'E', 'S', 'W']
        agent_locations = set([(0,0,0)])
        for _ in range(self.config['agent_config']['num_agents']):
            rand_dir = rd.choice(dirs)
            rand_dir = 'S'
            rand_loc = (0,0,0)
            while rand_loc in agent_locations:
                new_loc = [rd.randint(0, world_dim[dim]-1) for dim in range(2)]
                new_loc.append(self.ground_grid[rand_loc[1]][rand_loc[0]]+1)
                rand_loc = (new_loc[0], new_loc[1], new_loc[2])
            rand_loc = (rand_loc[0], rand_loc[1], rand_loc[2])
            rand_loc = (3, 3, 4)
            agent_locations.add(rand_loc)
            self.agents.append( self.GridAgent(loc=rand_loc, dir=rand_dir, world_scale=config['world_config']['world_scale']) )
            self.agents[-1].view = self.gen_agent_view(self.agents[-1])


    def display_world(self):
        ## Ground
        self.display_ground()
        ## Not Visible
        self.display_elements()
        ## Agent
        self.display_agents()
        ## Agent View
        self.display_agent_views()
        
        self.set_fig_extras()
        
        plt.show()
        
        # return self.ax

        
    def set_fig_extras(self):
        world_dim, world_scale = self.config['world_config']['world_dim'], self.config['world_config']['world_scale']
        self.ax.grid(False)
        max_bound = max(
            world_scale[0] * world_dim[0],
            world_scale[1] * world_dim[1],
            world_scale[2] * world_dim[2],
        )
        self.ax.set_xlim3d(-world_scale[0], -world_scale[0] + max_bound)
        self.ax.set_ylim3d(-world_scale[1], -world_scale[1] + max_bound)
        self.ax.set_zlim3d(-world_scale[2], -world_scale[2] + max_bound)
        

    #####################
    ### Display Block ###
    #####################
    def display_block(self, loc, facecolor="Black", edgecolor="Black", alpha=1):
        world_dim, world_scale = self.config['world_config']['world_dim'], self.config['world_config']['world_scale']
        
        faces, zs, z_dirs = [], [], []
        ## Gather adjacent elements
        top_elem = (
            self.world_layers[loc[2] + 1][loc[1]][loc[0]]
            if loc[2] + 1 != world_dim[2]
            else None
        )
        bottom_elem = (
            self.world_layers[loc[2] - 1][loc[1]][loc[0]] if loc[2] != 0 else None
        )
        left_elem = (
            self.world_layers[loc[2]][loc[1]][loc[0] - 1] if loc[0] != 0 else None
        )
        right_elem = (
            self.world_layers[loc[2]][loc[1]][loc[0] + 1]
            if loc[0] + 1 < world_dim[0]
            else None
        )
        back_elem = (
            self.world_layers[loc[2]][loc[1] - 1][loc[0]] if loc[1] != 0 else None
        )
        front_elem = (
            self.world_layers[loc[2]][loc[1] + 1][loc[0]]
            if loc[1] + 1 < world_dim[1]
            else None
        )
        loc_vals = [
            (loc[0]*world_scale[0], loc[1]*world_scale[1]),
            (loc[0]*world_scale[0], loc[1]*world_scale[1]),
            (loc[1]*world_scale[1], loc[2]*world_scale[2]),
            (loc[1]*world_scale[1], loc[2]*world_scale[2]),
            (loc[0]*world_scale[0], loc[2]*world_scale[2]),
            (loc[0]*world_scale[0], loc[2]*world_scale[2]),
        ]
        world_scale_vals = [
            (world_scale[0], world_scale[1]),
            (world_scale[0], world_scale[1]),
            (world_scale[1], world_scale[2]),
            (world_scale[1], world_scale[2]),
            (world_scale[0], world_scale[2])
        ]
        element_vals = [top_elem, bottom_elem, left_elem, right_elem, back_elem, front_elem]
        z_vals = [
            (loc[2] + 1) * world_scale[2],
            loc[2] * world_scale[2],
            loc[0] * world_scale[0],
            (loc[0] + 1) * world_scale[0],
            (loc[1]) * world_scale[1],
            ((loc[1]) + 1) * world_scale[1]
        ]
        z_dir_vals = ['z', 'z', 'x', 'x', 'y', 'y']
        
        for elem, l, scale, z, z_dir in zip(element_vals, loc_vals, world_scale_vals, z_vals, z_dir_vals):
            if elem == None or elem.is_transparent:
                faces.append(Rectangle(l, scale[0], scale[1], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha))
                zs.append(z), z_dirs.append(z_dir)
        return faces, zs, z_dirs

    def display_ground(self):
        for z_ind, layer in enumerate(self.world_layers):
            for y_ind, ground_line in enumerate(layer):
                for x_ind, grid_elem in enumerate(ground_line):
                    if grid_elem.label == 'ground':
                        facecolor = self.val_to_col["ground"][grid_elem.state]
                        faces, zs, z_dirs = self.display_block(
                            (x_ind, y_ind, z_ind), facecolor=facecolor
                        )
                        for rect, z, z_dir in zip(faces, zs, z_dirs):
                            self.ax.add_patch(rect)
                            art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)

    def display_elements(self):
        for grid_object in self.not_visible:
            faces, zs, z_dirs = self.display_block(
                (grid_object.loc[0], grid_object.loc[1], grid_object.loc[2]),
                facecolor="black",
            )
            for rect, z, z_dir in zip(faces, zs, z_dirs):
                self.ax.add_patch(rect)
                art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)

    def display_agent_views(self):
        colors = ["purple", "yellow", "orange", "pink", "grey", "aqua", "peru", "fuchsia", "whitesmoke", "yellowgreen"]
        for agent_ind, agent in enumerate(self.agents):
            agent_color = colors.pop()
            sweep_step = 0.5 / (len(agent.view))
            for sweep_ind, view_sweep in enumerate(agent.view):
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
                        self.ax.add_patch(rect)
                        art3d.pathpatch_2d_to_3d(rect, z=z, zdir=z_dir)

    def display_agents(self):
        for agent in self.agents:
            faces, zs, z_dirs = agent.faces, agent.zs, agent.z_dirs
            for rect, z, z_dir in zip(faces, zs, z_dirs):
                rect_copy = copy.deepcopy(rect)
                self.ax.add_patch(rect_copy)
                art3d.pathpatch_2d_to_3d(rect_copy, z=z, zdir=z_dir)

    def gen_agent_view(self, agent, padded=False):
        world_dim, world_scale = self.config['world_config']['world_dim'], self.config['world_config']['world_scale']
        max_agent_view_depth, max_agent_view_height = self.config['agent_config']['max_agent_view_depth'], self.config['agent_config']['max_agent_view_height']
        
        def view_ind_to_pos(view_ind, cur_loc, dir, dir_orth):
            x_view = cur_loc[0] + dir[0] + view_ind * dir_orth[0]
            y_view = cur_loc[1] + dir[1] + view_ind * dir_orth[1]
            # return (x_view, y_view, cur_loc[2] * self.world_height)
            return (x_view, y_view, cur_loc[2])
        
        ## Calculate agent direction
        dir = [0, 0, 0]
        if agent.dir in {"W", "E"}:
            dir[0] = -1 if agent.dir == "W" else 1
        if agent.dir in {"S", "N"}:
            dir[1] = -1 if agent.dir == "S" else 1
        agent.dir_vect = dir
        ## Calculate orthogonal agent direction
        dir_orth = [dir[1], dir[0], dir[2]]
        ## Calculate agent direction dot world_scale
        weighted_dir = (
            [dir[0] * world_scale[0], dir[1] * world_scale[1], dir[2]]
            if agent.dir in {"N", "S"}
            else []
        )
        weighted_dir = (
            [dir[0] * world_scale[1], dir[1] * world_scale[0], dir[2]]
            if agent.dir in {"W", "E"}
            else weighted_dir
        )
        ## Calculate orthogonal agent direction dot world_scale
        w_orth_dir = (
            [
                dir_orth[0] * world_scale[0],
                dir_orth[1] * world_scale[1],
                dir_orth[2],
            ]
            if agent.dir in {"N", "S"}
            else []
        )
        w_orth_dir = (
            [
                dir_orth[0] * world_scale[1],
                dir_orth[1] * world_scale[0],
                dir_orth[2],
            ]
            if agent.dir in {"E", "W"}
            else w_orth_dir
        )

        a_loc, cur_loc = agent.loc, agent.loc
        not_visible = set([obj.loc for obj in self.not_visible])
        view_set = set([cur_loc])
        view = [[cur_loc]]
        padded_view = [[cur_loc]]

        view_width = 1
        while view_width < max_agent_view_depth:

            ## Trace line of sight sweep from previous line of sight sweep
            def is_viewable(loc):
                loc_below_prec = (loc[0] - dir[0], loc[1] - dir[1], loc[2] - 1)
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
                    loc[0] < 0 or loc[0] >= world_dim[0]
                    or loc[1] < 0 or loc[1] >= world_dim[1]
                    or loc[2] < 0 or loc[2] >= world_dim[2]
                ):
                    return True
                
                # if not self.world_layers[loc[2]][loc[1]][loc[0]].is_transparent:
                #     return False
                
                ####### ## Vertical Tracing ##
                if loc[2] > a_loc[2]:
                    ## Iterate over cells between current block and blocks below, not visible if below not visible or ground
                    ##  check from [max(0, agent loc)] -> [block below loc]
                    for layer_ind in range(max(a_loc[2], 0), loc[2]):
                        if (
                            (loc[0], loc[1], layer_ind) in not_visible
                            or (loc_below_prec[0], loc_below_prec[1], layer_ind) in not_visible
                            or not self.world_layers[layer_ind][loc[1]][loc[0]].is_transparent
                            or not self.world_layers[layer_ind][loc_below_prec[1]][loc_below_prec[0]].is_transparent
                        ):
                            not_visible.add(loc)
                            return True
                if loc[2] < a_loc[2]:
                    ## Iterate over cells between current block and blocks above, not visible if above not visible or ground
                    ##  check from [block above loc] -> [min(agent loc, world_ceil_ind)]
                    for layer_ind in range(loc[2]+1, min(a_loc[2]+1, world_dim[2])):
                        if (
                            (loc[0], loc[1], layer_ind) in not_visible
                            or (loc_above_prec[0], loc_above_prec[1], layer_ind) in not_visible
                            or not self.world_layers[layer_ind][loc[1]][loc[0]].is_transparent
                            or not self.world_layers[layer_ind][loc_above_prec[1]][loc_above_prec[0]].is_transparent
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
                    rel_pos_dir[0] * (loc[0] - a_loc[0]) + rel_pos_dir[1] * (loc[1] - a_loc[1]) > 0
                ):
                    if loc_left_prec in not_visible and loc_prec not in view_set:
                        not_visible.add(loc)
                        return True
                ## Not viewable if we are to the left of the agent's line of sight and to the left of something non viewable
                if (
                    rel_pos_dir[0] * (loc[0] - a_loc[0]) + rel_pos_dir[1] * (loc[1] - a_loc[1]) < 0
                ):
                    if loc_right_prec in not_visible and loc_prec not in view_set:
                        not_visible.add(loc)
                        return True
                return False

            next_pos = []
            for layer_ind in range(a_loc[2]+1-max_agent_view_height, a_loc[2]+max_agent_view_height):
                next_pos.extend(
                    [
                        view_ind_to_pos(
                            view_ind, (cur_loc[0], cur_loc[1], layer_ind),
                            dir, dir_orth
                        ) for view_ind in range(-view_width, view_width + 1)
                    ]
                )

            proc_pos = list(filter(lambda x: not is_viewable(x), next_pos))
            
            padded_pos = [(not is_viewable(x), x) for x in next_pos]
            
            if padded:
                padded_view.append(padded_pos)
            elif not proc_pos:
                break

            view_set.update(proc_pos)
            view.append(proc_pos)

            view_width += 1

            cur_loc = [cur_loc[0] + dir[0], cur_loc[1] + dir[1]]
        if padded:
            padded_view[0] = [(True, padded_view[0][0])]
            return padded_view
        return view

    def gen_perlin_noise(self):
        world_dim = self.config['world_config']['world_dim']
        scale = self.config['world_terrain_config']['scale']
        octaves = self.config['world_terrain_config']['octaves']
        persistence = self.config['world_terrain_config']['persistence']
        lacunarity = self.config['world_terrain_config']['lacunarity']
        seed = self.config['world_terrain_config']['seed']
        num_ground_layers = self.config['world_terrain_config']['num_ground_layers']
        
        shape = (world_dim[0], world_dim[1])

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
            repeatx=world_dim[0],
            repeaty=world_dim[1],
            base=seed,
        )
        grid = np.floor((world + 0.5) * 255).astype(np.uint8)  # <- Normalize world first
        self.ground_grid = (num_ground_layers * (grid / np.max(grid)) - 0.001).astype(int)





dim_x, dim_y, dim_z = 7, 7, 8
scale_x, scale_y, scale_z = 1, 1, 1


scale = 0.5
octaves = 2
persistence = 1
lacunarity = 2
seed = 0
num_ground_layers=3


num_agents = 1
max_agent_view_depth=5
max_agent_view_height=3

num_generations=8



world_config = {
    'world_dim': (dim_x, dim_y, dim_z),
    'world_scale': (scale_x, scale_y, scale_z)
}

world_terrain_config = {
    ## altitude? 0 -> 1
    'scale': scale,
    ## number of passes, each pass adds more detail
    'octaves': octaves,
    'persistence': persistence,
    ## detail addedd per pass
    'lacunarity': lacunarity,
    # seed = np.random.randint(0,100)
    'seed': seed,
    'num_ground_layers': num_ground_layers
}

agent_config = {
    'num_agents': num_agents,
    'max_agent_view_depth': max_agent_view_depth,
    'max_agent_view_height': max_agent_view_height,
    'num_generations': num_generations
}



config = {
    'world_config': world_config,
    'world_terrain_config': world_terrain_config,
    'agent_config': agent_config
}


def main():
    griddy = GridWorld(config)
    griddy.make_animation()


if __name__ == "__main__":
    main()