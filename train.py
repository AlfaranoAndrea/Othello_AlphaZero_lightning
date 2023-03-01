from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl
import argparse

from ModelPipeline import ModelPipeline
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=0, help="gpus devices")
    parser.add_argument("--epochs", type=int, default=1, help="# epochs")
    parser.add_argument("--n_playout", type=int, default=1, help="# n of playout for each MCTS simulation")
    
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints',
        filename='file',
        save_last=True, 
        every_n_epochs=1, 
        save_top_k=5
        )
    
    logger = TensorBoardLogger(
            "tb_logs", 
            name="my_model"
            )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
  
    
    cli , _ = parser.parse_known_args()
    model = ModelPipeline(training=True, params=cli)

    if(len(cli.gpus) <= 1):
        print(cli.gpus)
        trainer = pl.Trainer(
                gpus=cli.gpus, 
                max_epochs=cli.epochs, 
                callbacks=[checkpoint_callback,lr_monitor]
                )
    else:
        print("b")
        ddp= DDPStrategy(
            find_unused_parameters=False, 
            process_group_backend="GLOO")

        trainer = pl.Trainer(
                gpus=cli.gpus, 
                max_epochs=cli.epochs, 
                auto_lr_find=False, 
                auto_scale_batch_size=False,
                precision=16,
                callbacks=[checkpoint_callback, lr_monitor], 
                logger=logger,
                strategy= ddp
                )

    trainer.fit(model )

if __name__ == '__main__':
    main()