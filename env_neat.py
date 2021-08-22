"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import neat
import numpy as np
import visualize
import copy
from GridWorld import GridWorld


class EnvWrapper:
    def __init__(self, config_env):
        
        self.possible_outputs = [
            ['', 'left', 'right', 'turn'],
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ]
        
        self.config_env = config_env
        
        self.env = GridWorld(config_env)
        self.env.gen_world()
        self.env.gen_agents()
        
        ## dictionary maps 
        self.persistent_refs = {}
        self.previous_genome_ids = set()
        
        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')
        self.run(config_path)
    
    class Dummy:
        def __init__(self, state):
            self.state = state
        
    
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
        
        if not self.previous_genome_ids:
            genome_ind = 0
            for genome_id, genome in genomes:
                self.persistent_refs[genome_id] = genome_ind
                self.previous_genome_ids.add(genome_id)
                genome_ind += 1
                
        current_genome_ids = set()
        current_new_ids = set()
        id_to_output = {}
        for genome_id, genome in genomes:
            current_genome_ids.add(genome_id)
            if genome_id not in self.previous_genome_ids:
                current_new_ids.add(genome_id)
                continue
            agent_ind = self.persistent_refs[genome_id]
            
            agent = self.env.agents[agent_ind]
            agent_view = self.env.gen_agent_view(agent, padded=True)
        
            
            masked_view = []
            # print('agent_view[1]', agent_view[1])
            for view_sweep in agent_view:
                for view, loc in view_sweep:
                    x, y, z = loc
                    element = self.env.world_layers[z][y][x] if view else self.Dummy(-99)
                    masked_view.append(float(element.state))
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            output = net.activate(masked_view)
            
            p_dirs, p_xs, p_ys, p_zs = output[:4], output[4:7], output[7:10], output[10:13]
            d_d, d_x, d_y, d_z = np.argmax(np.array(p_dirs)), np.argmax(np.array(p_xs)), np.argmax(np.array(p_ys)), np.argmax(np.array(p_zs))
            output = [d_d, [d_x, d_y, d_z], 0]
            id_to_output[genome_id] = output
            
        used_inds = set([self.persistent_refs[g_id] for g_id in self.previous_genome_ids if g_id in current_genome_ids])
        new_agent_inds = iter([ind for ind in range(len(self.env.agents)) if ind not in used_inds])
        for genome_id, genome in genomes:
            if genome_id in self.previous_genome_ids:
                continue
            next_available_agent = next(new_agent_inds)
            self.persistent_refs[genome_id] = next_available_agent
            agent = self.env.agents[next_available_agent]
            agent_view = self.env.gen_agent_view(agent, padded=True)
            
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
            
        outputs = [None]*len(genomes)
        for id, output in id_to_output.items():
            outputs[self.persistent_refs[id]] = output
            
        self.env.update_state(outputs)
        agent_scores = self.env.hide_and_seek_score()
        
        for genome_id, genome in genomes:
            genome.fitness = agent_scores[self.persistent_refs[genome_id]]
        
        self.env.gen_ind += 1
    
    
    
    def run(self, config_file):
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
        
        config.pop_size = self.config_env['agent_config']['num_agents']
        max_agent_view_depth = self.config_env['agent_config']['max_agent_view_depth']-1
        max_agent_view_height = self.config_env['agent_config']['max_agent_view_height']
        view_height = 2*max_agent_view_height-1
        view_size = (view_height*(max_agent_view_depth*(max_agent_view_depth+2))) + 1
        config.num_inputs = view_size
        config.num_outputs = 13
        

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(gen_interval:=5, filename_prefix='checkpoints/neat-checkpoint-'))

        # Run for up to 300 generations.
        winner = p.run(self.eval_genomes, 30)
        
        print(dir(winner))




def main():
    dim_x, dim_y, dim_z = 20, 15, 8
    scale_x, scale_y, scale_z = 1, 1, 1


    scale = 0.5
    octaves = 2
    persistence = 1
    lacunarity = 2
    seed = 0
    num_ground_layers = 3


    num_agents = 10
    max_agent_view_depth = 5
    max_agent_view_height = 3

    num_generations = 8



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
    
    env_wrapper = EnvWrapper(config)
    env_wrapper.run('config-feedforward')
    
    

if __name__ == '__main__':
    main()
