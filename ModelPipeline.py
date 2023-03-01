from DNN import Network
import torch
from torch import optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import  LightningModule
from torch import optim
from dataset import RLDataset

class ModelPipeline(LightningModule):
    def __init__(self, params, training=False):
        super(ModelPipeline,self).__init__()
        self.params=params
        self.batch_size = 250
        self.learn_rate = 1e-3
        self.net = Network(4, 128, 15, 64, 65)  #.to(self.device)
        
        if training:
            self.dataset=RLDataset(
                    n_playout=self.params.n_playout,
                    batch_size=self.batch_size  
                    )
            device = self.get_device(self.net)
            
            #play ten games to load the replay buffer
            for _ in range(3):
                self.dataset.update_batch(
                device= device, 
                net= self.net, 
                )
    def train_dataloader(self):
        dataloader = DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                num_workers= 8
                                )
        return dataloader
    
    def training_step(self,batch,batch_idx):
        device = self.get_device(batch)
        self.dataset.update_batch(device= device, net= self.net)
        state_batch, mcts_prob_batch, score_batch= batch
       
        log_actprobs, score =  self.net(state_batch)
        value_loss = F.mse_loss(score, score_batch.reshape(-1,1))
        policy_loss = -torch.mean(torch.sum(mcts_prob_batch*log_actprobs, 1))
        loss = value_loss + policy_loss

        self.log('train_loss', loss,on_step=True, on_epoch=True,prog_bar=True)
        return loss

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'
    
    def configure_optimizers(self):
        opt=optim.SGD(
            self.net.parameters(), 
            lr = self.learn_rate, 
            momentum=0.9, 
            weight_decay=1e-4)
        return opt


