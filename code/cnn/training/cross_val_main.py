from models.multi import ImagenetTransferLearning
from preprocessing.dataloaders import KFoldMemeDataModule

import os
import torch
import lightning as L
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BatchSizeFinder, Callback
from argparse import ArgumentParser
from datetime import datetime
import mlflow.pytorch
import mlflow as mlf


def get_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
      exp_id = mlflow.create_experiment(name)
      return exp_id
    return exp.experiment_id

class KFoldLogger(Callback):
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        print(f"Fold: {trainer.current_epoch}, Loss: {trainer.callback_metrics['val/loss']}")
        print(f"Fold: {trainer.current_epoch}, Accuracy: {trainer.callback_metrics['val/acc']}")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-m","--model_name", type=str)
    parser.add_argument("-b","--batch_size", type=int, default=32)
    parser.add_argument("-f","--folds", type=int, default=5)
    parser.add_argument("-df","--data_df_path", type=str, default="/home/hsdslab/murgi/meme-research-2024/data/processed/gpu_server_meme_entries.parquet")
    parser.add_argument("-c","--checkpoint_dir", type=str, default="/home/hsdslab/murgi/meme-research-2024/src/models/cnns")

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    # Auto log all MLflow entities
    mlf.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.pytorch.autolog(checkpoint_monitor="val/loss")

    L.seed_everything(42,workers=True)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = "gpu" if torch.cuda.is_available() else "cpu"

    for fold in range(args.folds):
        model = ImagenetTransferLearning(model_name=args.model_name, num_target_classes=1145)
        datamodule = KFoldMemeDataModule(batch_size=args.batch_size, data_df_path=args.data_df_path, k=fold, num_splits=args.folds, split_seed=42)
        now = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir+"-"+now+"-"+str(fold), save_top_k=2, monitor="val/loss", mode="min", filename=f"{args.model_name}"+"{epoch:02d}-{val_loss:.2f}")

        # Debug
        # trainer = Trainer(deterministic=True,accelerator=device, max_epochs=10,fast_dev_run=True,
        #                   callbacks=[EarlyStopping(monitor="val/loss", mode="min"), checkpoint_callback,BatchSizeFinder(mode="binsearch")])
        trainer = Trainer(
        deterministic=True,
        accelerator=device,
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="val/loss", mode="min"),checkpoint_callback, KFoldLogger()]
        )

        exp_id = get_experiment_id(f"{args.model_name}_cross_val_experiment")
        print(exp_id)
        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.set_tag("model_name", args.model_name)
            trainer.fit(model, datamodule)
        


