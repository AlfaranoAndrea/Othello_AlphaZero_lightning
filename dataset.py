from torch.utils.data.dataset import IterableDataset
from typing import Tuple
import numpy as np
import torch
from MCTS import MCTS
from OthelloGame import Game
import random
from torch.autograd import Variable

from collections import deque
class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """
    def __init__(self, neuralNet) -> None:
        self.buffer_size = 700000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.n_playout = 1600
        self.c_puct = 5
        self.minibatch_steps = 0
        self.batch_size = 250
        self.learn_rate = 1e-2
        self.lr_multiplier = 1.0
        self.ff = 0.9
        self.L2penalty = 1e-4
        self.epslion = 0.25
        self.eta = 0.3

        self.temperature = 0.001
        self.eval_freq = 100
        self.check_freq = 20
  
        self.neuralNet=neuralNet
        self.update_batch(torch.device("cuda"))
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, mcts_prob, score = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(state), np.array(mcts_prob), np.array(score, dtype=np.float32))



    def add_self_play_data(self, play_data):
        extend_data = []
        for state, mcts_prob, score in play_data:
            for i in [1,2,3,4]:
                rot_state = np.rot90(state, i, (1,2))
                rot_mcts_prob = np.append(np.rot90(mcts_prob[:-1].reshape((8,8)),i).reshape(-1), mcts_prob[-1])
                extend_data.append((rot_state,rot_mcts_prob,score))
                flp_state = np.flip(rot_state, 1)
                flp_mcts_prob = np.append(np.flip(rot_mcts_prob[:-1].reshape((8,8)),1).reshape(-1), rot_mcts_prob[-1])
                extend_data.append((flp_state,flp_mcts_prob,score))
                #print('state\n',state,sep='')
                #print('rot_state\n',rot_state,sep='')
                #print('flp_state\n',flp_state,sep='')
        self.data_buffer.extend(extend_data)        
        #print('debug in add self play:', extend_data[0][0], extend_data[0][1], extend_data[0][2])

    def self_play(self, net, device):
        net.eval()
        mcts = MCTS(net, self.n_playout, self.c_puct)
        states, mcts_probs, scores = [],[],[]
        moves_count = 0
        game = Game(device)
        while game.get_game_result()[0] == 0:
            # game.print_game_state()
            # For the first 8 moves of each game, the
            # temperature is set to τ = 1; this selects moves proportionally to their visit count in MCTS, and
            # ensures a diverse set of positions are encountered. For the remainder of the game, an infinitesimal
            # temperature is used, τ → 0. 
            if moves_count < 8:
                acts, act_probs = mcts.play(game, 1)
            else:
                acts, act_probs = mcts.play(game, self.temperature)

            # Additional exploration is achieved by adding Dirichlet noise to the prior probabilities in the root node.
            # This noise ensures that all moves may be tried, but the search may still overrule bad moves.
            action = int(np.random.choice(
                acts, 
                p=act_probs * (1-self.epslion) + self.epslion*np.random.dirichlet(self.eta*np.ones(act_probs.shape))
            ))
            mcts.update_with_move(-1)
            
            mcts_prob = np.zeros((65))
            mcts_prob.flat[np.asarray(acts)] = act_probs
            
            states.append(game.get_numpy_format())
            mcts_probs.append(mcts_prob)
            scores.append(game.cur_player)
           # game.print_game_state()
            game.set_next_state(action)
            moves_count += 1
        #game.print_game_state()
        #game.print_game_state()
        
        res = game.get_game_result()[1]
        for i in range(len(scores)):
            if res == game.Draw:
                scores[i] = game.Draw
            else:
                scores[i] = res * game.cur_player * scores[i]

        return zip(states, mcts_probs, scores)

    def __iter__(self) -> Tuple:
        mini_batch = random.sample(
            self.data_buffer, 
            self.batch_size
            )

        state_batch = np.array([data[0] for data in mini_batch])
        mcts_prob_batch =  np.array([data[1] for data in mini_batch])
        score_batch =  np.array([data[2] for data in mini_batch])
        
        state_batch = Variable(torch.FloatTensor(state_batch)) #.cuda()
        mcts_prob_batch = Variable(torch.FloatTensor(mcts_prob_batch)) #.cuda()
        score_batch = Variable(torch.FloatTensor(score_batch))#.cuda()
        for i in range(self.batch_size):
            yield state_batch[i], mcts_prob_batch[i], score_batch[i]


    def update_batch(self, device):
        buffer=self.self_play(self.neuralNet, device=device)
        self.add_self_play_data(buffer)