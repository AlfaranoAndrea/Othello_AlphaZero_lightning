from OthelloGame import Game
import numpy as np
import torch
from tqdm import tqdm
from player import HumanPlayer, Alphazero, MCTS_player, RandomPlayer

def play_match(player1, player2, verbose=True):
    use_gpu = torch.cuda.is_available() 
    device = torch.device("cuda" if use_gpu else "cpu")
    game = Game(device)
    if verbose: 
        print('device:',device)
        game.print_game_state()
    
    for _ in tqdm(range(100)):
        if game.get_game_result()[0] != 0:
            break
        if game.cur_player == 1:
            action=player1.play(game)
        else:
            action=player2.play(game)
        
        game.set_next_state(action)
        if verbose: game.print_game_state()
 
    res = game.get_game_result()[1]*game.cur_player 
    if verbose: 
        print(res)
    return res


def play_tournament(player1,player2, n_match=100, verbose=False):
    results=[]
    for _ in tqdm(range(n_match)):
        res=play_match(player1,player2, verbose)
        results.append(res)

    print(
        f"PLAYER1 = {player1}  \
        \n pLAYER2 = {player2} \
        \n results are{results}: \
        player 1 won {results.count(1)}, \
        draw {results.count(0) },\
        lose {results.count(-1)}"
        )
    if(results.count(1)>results.count(-1)):
        return player1, player2
    else:
        return player2, player1

if __name__ == '__main__':
    player1 = Alphazero(
        checkpoint="./checkpoints/checkpoint10000.ckpt", 
        n_playout=400, 
        c_puct=5,
        verbose= False)

    player2= Alphazero(
        checkpoint="./checkpoints/checkpoint2000.ckpt",
        n_playout=400, 
        c_puct=5 ,
        verbose= False)
    player3 = Alphazero(
        checkpoint="./checkpoints/checkpoint1000.ckpt", 
        n_playout=400, 
        c_puct=5,
        verbose= False)

    player4= MCTS_player(
       # checkpoint="./checkpoints/checkpoint4000.ckpt",
        n_playout=400, 
        c_puct=5 ,
        verbose= False)
    player5 = RandomPlayer()




    play_tournament(player1,player2,verbose=False,n_match=11)
    play_tournament(player1,player3,verbose=False,n_match=11)
    play_tournament(player1,player4,verbose=False,n_match=11)
    play_tournament(player1,player5,verbose=False,n_match=11)


    play_tournament(player2,player3,verbose=False,n_match=11)
    play_tournament(player2,player4,verbose=False,n_match=11)
    play_tournament(player2,player5,verbose=False,n_match=11)


    play_tournament(player3,player4,verbose=False,n_match=11)
    play_tournament(player3,player5,verbose=False,n_match=11)


    play_tournament(player4,player5,verbose=False,n_match=11)    
    

    # if __name__ == '__main__':
    # player7 = Alphazero(
    #     checkpoint="./checkpoints/checkpoint10000.ckpt", 
    #     n_playout=400, 
    #     c_puct=5,
    #     verbose= False)

    # player6= Alphazero(
    #     checkpoint="./checkpoints/checkpoint8000.ckpt",
    #     n_playout=400, 
    #     c_puct=5 ,
    #     verbose= False)
    # player5 = Alphazero(
    #     checkpoint="./checkpoints/checkpoint6000.ckpt", 
    #     n_playout=400, 
    #     c_puct=5,
    #     verbose= False)

    # player4= Alphazero(
    #     checkpoint="./checkpoints/checkpoint4000.ckpt",
    #     n_playout=400, 
    #     c_puct=5 ,
    #     verbose= False)
    # player3 = Alphazero(
    #     checkpoint="./checkpoints/checkpoint2000.ckpt", 
    #     n_playout=400, 
    #     c_puct=5,
    #     verbose= False)

    # player2= Alphazero(
    #     checkpoint="./checkpoints/checkpoint1000.ckpt",
    #     n_playout=400, 
    #     c_puct=5 ,
    #     verbose= False)
    # player1 = Alphazero(
    #     checkpoint="./checkpoints/checkpoint20.ckpt", 
    #     n_playout=400, 
    #     c_puct=5,
    #     verbose= False)


    # winnerR1, loserR1= play_tournament(player1,player2,verbose=False,n_match=11)
    # loserR1.deallocate()
    # winnerR2, loserR2= play_tournament(winnerR1,player3,verbose=False,n_match=11)
    # loserR2.deallocate()
    # winnerR3, loserR3= play_tournament(winnerR2,player4,verbose=False,n_match=11)
    # loserR3.deallocate()
    # winnerR4, loserR4= play_tournament(winnerR3,player5,verbose=False,n_match=11)
    # loserR4.deallocate()
    # winnerR5, loserR5= play_tournament(winnerR4,player6,verbose=False,n_match=11)
    # loserR5.deallocate()
    # winnerR6, loserR6= play_tournament(winnerR5,player7,verbose=False,n_match=11)
    # loserR6.deallocate()