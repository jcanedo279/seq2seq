from __future__ import print_function
import os
import time
import neat
import numpy as np
import random as rd
from wandb.sdk import wandb_config
import visualize
import math
import copy
import wandb
from GridWorld import GridWorld
import matplotlib
matplotlib.use('agg')
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
from PIL import Image

class EnvWrapper:
    def __init__(self, config_neat_path, config_env, display=False):
        ## Initialize wandb
        os.environ['WANDB_NOTEBOOK_NAME'] = 'OpenEndedEnv'
        wandb.login()
        
        self.wandb_config = wandb_config
        
        self.possible_outputs = [
            ['', 'left', 'right', 'turn'],
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ]
        
        self.eval_gen = 0
        self.agent_history = []
        self.gen_ind_to_best_agent = {}
        
        local_dir = os.path.dirname(__file__)
        self.config_neat_path = os.path.join(local_dir, 'config-feedforward')
        self.config_env = config_env
        
        self.parse_config()
        
        self.env = GridWorld(config_env)
        self.env.gen_world()
        self.env.gen_spaced_agents()
        
        ## dictionary maps 
        self.persistent_refs = {}
        self.previous_genome_ids = set()
        
        self.display = display
        self.run()
        
    
    def parse_config(self):
        max_agent_view_depth = self.config_env['agent_config']['max_agent_view_depth']-1
        max_agent_view_height = self.config_env['agent_config']['max_agent_view_height']
        view_height = 2*max_agent_view_height-1
        view_size = (view_height*(max_agent_view_depth*(max_agent_view_depth+2))) + 1
        self.config_env['agent_config']['num_inputs'] = view_size
        
        re_writes = {
            'pop_size': self.config_env['agent_config']['num_agents'],
            'num_inputs': view_size,
            'num_outputs': 13,
        }
        
        new_lines = []
        with open(self.config_neat_path) as neat_config_file:
            for line in neat_config_file:
                if '=' not in line:
                    new_lines.append(line)
                    continue
                key_value_pair = [unf_str.strip() for unf_str in line.split('=')]
                if key_value_pair[0] in re_writes:
                    un_stripped_line = line.split('=')
                    new_line = f'{un_stripped_line[0]}= {re_writes[key_value_pair[0]]}\n'
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
        
        self.config_neat = {}
        with open(self.config_neat_path, 'w') as neat_config_file:
            for line in new_lines:
                neat_config_file.write(line)
                if '=' in line:
                    key, val = [val.strip() for val in line.split('=')]
                    self.config_neat[key] = val
    
    class Dummy:
        def __init__(self, state):
            self.state = state
        
        
    def display_history(self):
        self.continue_animation = True
        self.fps = 1
        self.display_ind = 0
        
        current_time = time.localtime()
        fitted_time_arr = time.strftime('%a, %d %b %Y %H:%M:%S GMT', current_time).split(' ')
        local_time = fitted_time_arr[4]
        local_date = f'{fitted_time_arr[2]}:{fitted_time_arr[1]}'
        self.output_path = f'out/neat_environment_{local_date}_{local_time}.gif'
        
        self.env.animating_figure = plt.figure()
        self.env.ax = plt.axes(projection="3d")
        self.env.ax.axis('auto')
        
        # self.env.gen_world()
        
        self.display_iter = iter(self.agent_history)
        
        self.anim = FuncAnimation(self.env.animating_figure, self.update_display,
                                    frames=self.gen_frames(), init_func=self.init_plot(), repeat=False)
        self.anim.save(self.output_path, writer=PillowWriter(fps=self.fps))

        wandb.log({"Fittest Member Animation": wandb.Video(self.output_path, fps=self.fps, format="gif")})
        
        plt.close('all')
        del self.env.animating_figure
        
    def update_display(self, i):
        if i == StopIteration:
            return
        print(f'Displaying sample generation [{i+1} / {len(self.agent_history)}]')
        self.env.agents = next(self.display_iter)
        gen_ind = math.floor(i/self.config_env['agent_config']['num_eval_generations'])
        self.env.display_world(best_agent=self.gen_ind_to_best_agent[gen_ind], gen_ind=gen_ind)
        self.display_ind += 1
        return self.env.ax
        
    def init_plot(self):
        self.env.ax.cla()
        
    def gen_frames(self):
        while (self.display_ind < len(self.agent_history) and self.continue_animation):
            self.env.ax.cla()
            yield self.display_ind
        yield StopIteration
        
    
    def eval_genomes(self, genomes, config):     
        """
        outputs = [
            dir in {'', 'left', 'right', 'turn'},
            d_x in {-1, 0, 1},
            d_y in {-1, 0, 1},
            d_z in {-1, 0, 1},
            state in {'', 'grab', 'release'}
        ]
        """
        rd.shuffle(genomes)
        temp_genomes = genomes[:self.config_env['agent_config']['num_agents']]
        
        total_agent_outputs = np.array([0]*len(self.env.agents))
        total_agent_scores, total_agent_penalties = np.array([0]*len(self.env.agents)), np.array([0]*len(self.env.agents))
        for eval_gen in range(self.config_env['agent_config']['num_eval_generations']):
            
            if eval_gen == 0:
                self.env.gen_spaced_agents()
            
            ## Define mappings from new gene ids to agent indexes
            if not self.previous_genome_ids:
                genome_ind = 0
                for genome_id, genome in temp_genomes:
                    self.persistent_refs[genome_id] = genome_ind
                    self.previous_genome_ids.add(genome_id)
                    genome_ind += 1
                    
            current_genome_ids = set()
            current_new_ids = set()
            id_to_output = {}
            ## Eval existing agents
            for genome_id, genome in temp_genomes:
                current_genome_ids.add(genome_id)
                if genome_id not in self.previous_genome_ids:
                    current_new_ids.add(genome_id)
                    continue
                agent_ind = self.persistent_refs[genome_id]
                
                agent = self.env.agents[agent_ind]
                agent_view = self.env.gen_agent_view(agent, padded=True)
                self.env.agents[agent_ind].view = self.env.gen_agent_view(agent)
            
                
                agent_locs = set([a.loc for a in self.env.agents])
                masked_view = []
                for view_sweep in agent_view:
                    for view, loc in view_sweep:
                        x, y, z = loc
                        """
                        -1: agent
                        0:  air / transparent
                        1:  masked / hidden
                        2:  element
                        3:  ground
                        """
                        element =  self.env.world_layers[z][y][x] if view else self.Dummy(1)
                        element = self.Dummy(-1) if (x,y,z) in agent_locs and (x,y,z) != agent.loc else element
                        # element = self.Dummy(99) if (x,y,z) in agent_locs else self.env.world_layers[z][y][x]
                        # element = element if view else self.Dummy(-99)
                        masked_view.append(float(element.state))
                
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                output = net.activate(masked_view)
                
                p_dirs, p_xs, p_ys, p_zs = output[:4], output[4:7], output[7:10], output[10:13]
                d_d, d_x, d_y, d_z = np.argmax(np.array(p_dirs)), np.argmax(np.array(p_xs)), np.argmax(np.array(p_ys)), np.argmax(np.array(p_zs))
                output = [d_d, [d_x, d_y, d_z], 0]
                id_to_output[genome_id] = output
            
                
            ## Eval new agents
            used_inds = set([self.persistent_refs[g_id] for g_id in current_genome_ids-current_new_ids])
            new_agent_inds = iter([ind for ind in range(len(self.env.agents)) if ind not in used_inds])
            for genome_id, genome in temp_genomes:
                if genome_id in self.previous_genome_ids:
                    continue
                next_available_agent = next(new_agent_inds)
                self.persistent_refs[genome_id] = next_available_agent
                agent = self.env.agents[next_available_agent]
                agent_view = self.env.gen_agent_view(agent, padded=True)
                self.env.agents[next_available_agent].view = self.env.gen_agent_view(agent)
                
                masked_view = []
                for view_sweep in agent_view:
                    for view, loc in view_sweep:
                        x, y, z = loc
                        element = self.env.world_layers[z][y][x] if view else self.Dummy(-99)
                        masked_view.append(float(element.state))
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                output = net.activate(masked_view)
                
                p_dirs, p_xs, p_ys, p_zs = output[:4], output[4:7], output[7:10], output[10:13]
                d_d, d_x, d_y, d_z = np.argmax(np.array(p_dirs)), np.argmax(np.array(p_xs)), np.argmax(np.array(p_ys)), np.argmax(np.array(p_zs))
                id_to_output[genome_id] = [d_d, [d_x, d_y, d_z], 0]
                
            outputs = [None]*len(temp_genomes)
            for id, output in id_to_output.items():
                outputs[self.persistent_refs[id]] = output
                
            self.env.update_state(outputs)
            self.agent_history.append(copy.deepcopy(self.env.agents))
            agent_scores_dict = self.env.hide_and_seek_score(method='raw', extra_data=True)
            
            total_agent_outputs = total_agent_outputs + np.array(agent_scores_dict['outputs'])
            total_agent_scores = total_agent_scores + np.array(agent_scores_dict['scores'])
            total_agent_penalties = total_agent_penalties + np.array(agent_scores_dict['penalties'])
            
        best_agent = np.argmax(total_agent_outputs)
        self.gen_ind_to_best_agent[self.eval_gen] = best_agent
        
        total_agent_outputs = total_agent_outputs.astype(float)
        agent_norm_outputs = [output/self.config_env['agent_config']['num_generations'] for output in total_agent_outputs]
        agent_norm_scores = [score/self.config_env['agent_config']['num_generations'] for score in total_agent_scores]
        agent_norm_penalties = [penalty/self.config_env['agent_config']['num_generations'] for penalty in total_agent_penalties]
        
        self.eval_gen += 1
        
        used_genome_ids = set([genome_id for genome_id, _ in temp_genomes])
        for genome_id, genome in genomes:
            if genome_id in used_genome_ids:
                genome.fitness = agent_norm_outputs[self.persistent_refs[genome_id]]
            else:
                genome.fitness = 0
        
        log_params = {
            'avg_hide_and_seek_fitness': sum(agent_norm_outputs)/len(agent_norm_outputs),
            'best_hide_and_seek_fitness': max(agent_norm_outputs),
            'avg_hide_and_seek_score': sum(agent_norm_scores)/len(agent_norm_scores),
            'best_hide_and_seek_score': max(agent_norm_scores),
            'avg_hide_and_seek_penalties': sum(agent_norm_penalties)/len(agent_norm_penalties),
            'best_hide_and_seek_penalty': max(agent_norm_penalties),
            'num_agents': self.config_wandb['agent_config']['num_agents'],
            'agent_density': self.config_wandb['agent_config']['agent_density'],
            'world_area': self.env.config['world_config']['world_dim'][0]*self.env.config['world_config']['world_dim'][1],
            'num_generations': self.config_wandb['agent_config']['num_generations']
        }
          
        wandb.log(log_params)
    
    
    
    def run(self):
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            self.config_neat_path)
        
        self.config = copy.deepcopy(self.config_env)
        self.config.update(self.config_neat)
        run_context = wandb.init(project=os.environ['WANDB_NOTEBOOK_NAME'], config=self.config)
        with run_context:
            self.config_wandb = wandb.config

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)

            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(gen_interval:=5, filename_prefix='checkpoints/neat-checkpoint-'))

            # Run for up to 300 generations.
            winner = p.run(self.eval_genomes, self.config_env['agent_config']['num_generations'])
            
            # # Show output of the most fit genome against training data.
            # print('\nOutput:')
            # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            # for xi, xo in zip(xor_inputs, xor_outputs):
            #     output = winner_net.activate(xi)
            #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

            node_names = {-(i+1): f'{i+1}' for i in range(self.config_env['agent_config']['num_inputs']+1)}
            node_names.update({i: f'{i}' for i in range(13)})
            visualize.draw_net(config, winner, True, node_names=node_names)
            visualize.plot_stats(stats, ylog=False, view=False)
            visualize.plot_species(stats, view=False)

            # num_checks = math.floor(self.config_env['agent_config']['num_generations']/gen_interval)-1
            # p = neat.Checkpointer.restore_checkpoint(f'checkpoints/neat-checkpoint-{4+5*num_checks}')
            # p.run(self.eval_genomes, 10)
            
            if self.display:
                self.display_history()
            






def main():
    scale = 0.5
    octaves = 2
    persistence = 1
    lacunarity = 2
    seed = 0
    num_ground_layers = 3


    num_agents = 30
    agent_density = 0.01
    max_agent_view_depth = 5
    max_agent_view_height = 3


    base_len = math.ceil(math.sqrt(num_agents/agent_density))
    dim_x, dim_y, dim_z = base_len, base_len, 8
    scale_x, scale_y, scale_z = 1, 1, 1
    
    
    num_generations = 4
    num_eval_generations = 5
    
    display = True
    

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
        'agent_density': agent_density,
        'max_agent_view_depth': max_agent_view_depth,
        'max_agent_view_height': max_agent_view_height,
        'num_generations': num_generations,
        'num_eval_generations': num_eval_generations,
    }



    config_env = {
        'world_config': world_config,
        'world_terrain_config': world_terrain_config,
        'agent_config': agent_config
    }
    
    config_neat_path = 'config-feedforward'
    env_wrapper = EnvWrapper(config_neat_path, config_env, display=display)
    env_wrapper.run()
    
    

if __name__ == '__main__':
    main()


