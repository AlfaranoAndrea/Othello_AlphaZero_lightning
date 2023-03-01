import numpy as np
from MCTS import MCTS, puremcts_policy
from ModelPipeline import ModelPipeline


class Player ():
    def __init__(self,  verbose=True) -> None:
        super().__init__()
        self.verbose= verbose
    def convert_single(self,num):
        s = ""
        if 0 <= num < 64:
            s += chr(ord('A') + num%8)
            s += chr(ord('1') + 7-num//8)
        return s    

    def iconvert_single(self,s):
        return ord(s[0])-ord('A') + (7 - ord(s[1]) + ord('1')) * 8 

    def convert(self, act_list):
        s = ""
        for num in act_list:
            s += self.convert_single(num)
        return s

    
    def play(game):
        pass

    def log(self,message):
        if self.verbose:
            print(message)

class HumanPlayer(Player):
    def __init__(self, verbose=False) -> None:
        super().__init__(verbose)
    def play(self, game):
        playable = np.where(game.get_vaild_moves() == 1)[0]
        playable.sort()
        if playable[-1] == 64:
            self.log('player pass.')
            action = 64
        else:
            action = -1
            print('player:',end=' ')
            while action not in playable:
                ss = [self.convert_single(num) for num in playable]
                print(ss)
                s = input().upper()
                action = self.iconvert_single(s)
        return action


class Alphazero(Player):
    def __init__(self, n_playout, c_puct, checkpoint,verbose=True) -> None:
        super().__init__(verbose)
        self.model=   ModelPipeline()#.load_from_checkpoint(checkpoint)
        self.policy=MCTS(self.model.net.cuda(),  n_playout ,c_puct,)
        self.checkpoint=checkpoint
        self.moves_count=0
        self.temperature = 0.00001
        self.n_playout=n_playout 
        self.c_puct =c_puct

    def __str__(self):
        return f"Alphazero with model: {self.checkpoint}, n_playout:{self.n_playout}  c_puct:{self.c_puct}"

    def play(self, game):
        if self.moves_count < 4:
            acts, act_probs = self.policy.play(game, 0.1)
        else:
            acts, act_probs = self.policy.play(game, self.temperature)
        action = int(np.random.choice(acts, p=act_probs))
        if action == 64: 
             self.log('nnet pass.')
        else:
            self.log(f"nnet: {self.convert_single(action)}")
        self.log(f"score:,{self.policy.get_root_score()}")
        self.policy.update_with_move(-1)
        self.moves_count+=1
        return action
    def deallocate(self):
        self.model.net.cpu()

class MCTS_player(Player):
    def __init__(self, n_playout, c_puct, verbose=True) -> None:
        super().__init__(verbose)
        self.policy=MCTS(puremcts_policy, n_playout, c_puct)
        self.policy.nnet
        self.moves_count=0
        self.temperature = 0.00001
        self.n_playout=n_playout 
        self.c_puct =c_puct


    def play(self, game):
        acts, act_probs = self.policy.play(game, self.temperature)
        self.log(acts)
        self.log(act_probs)
        action = int(np.random.choice(acts, p=act_probs))
        self.policy.update_with_move(-1)
        return action
    
    
    def __str__(self):
        return f"MCTS_player with n_playout: {self.n_playout}  c_puct: {self.c_puct}"

class RandomPlayer(Player):
    def __init__(self, verbose=False) -> None:
        super().__init__(verbose)

    def play(self, game):
        acts = np.nonzero(game.get_vaild_moves())[0]
        self.log(acts)
        action = int(np.random.choice(acts))
        return action