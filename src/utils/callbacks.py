from pytorch_lightning.callbacks import ProgressBar

# Epochごとに上書きが起きないようにしている．
class PerEpochProgressBar(ProgressBar):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch:
            print()  # 上書き禁止
        super().on_train_epoch_start(trainer, pl_module)
