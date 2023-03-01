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
    Iterable Dataset containing the self played matches
    """
    def __init__(self, n_playout=1, batch_size=50) -> None:
        self.buffer_size = 700000
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.n_playout = n_playout
        self.c_puct = 5
        self.batch_size = batch_size
        self.epslion = 0.25
        self.eta = 0.3
        self.temperature = 0.001

        
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, mcts_prob, score = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(state), np.array(mcts_prob), np.array(score, dtype=np.float32))
  

    def self_play(self, net, device):
        net.eval()
        player = MCTS(net, self.n_playout, self.c_puct)
        states, mcts_probs, scores = [],[],[]
        game = Game(device)
        states, mcts_probs, scores= self.play_a_match(game, player)
        res = game.get_game_result()[1]
        for i in range(len(scores)):
            if res == game.Draw:
                scores[i] = game.Draw
            else:
                scores[i] = res * game.cur_player * scores[i]

        return zip(states, mcts_probs, scores)
    
    
    def play_a_match(self, game,player):
        moves_count = 0
        states, mcts_probs, scores = [],[],[]
        while game.get_game_result()[0] == 0:
            if moves_count < 8:
                acts, act_probs = player.play(game, 1)
            else:
                acts, act_probs = player.play(game, self.temperature)
            action = int(np.random.choice(
                acts, 
                p=act_probs * (1-self.epslion) + self.epslion*np.random.dirichlet(self.eta*np.ones(act_probs.shape))
            ))
            player.update_with_move(-1)
            mcts_prob = np.zeros((65))
            mcts_prob.flat[np.asarray(acts)] = act_probs
            
            states.append(game.get_numpy_format())
            mcts_probs.append(mcts_prob)
            scores.append(game.cur_player)
            game.set_next_state(action)
            moves_count += 1
        return states, mcts_probs, scores


    def __iter__(self) -> Tuple:
        mini_batch = random.sample(
            self.replay_buffer, 
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

    def update_batch(self, device, net):
        self.replay_buffer.extend(
            self.self_play(net, device=device)
            )
