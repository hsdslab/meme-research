import torchmetrics.classification
import torchvision.models as models
import torch.nn as nn
import torch
import lightning as L
import torchmetrics
import pytorch_lightning as pl
from sklearn.metrics import roc_curve
import numpy as np
import mlflow
from huggingface_hub import PyTorchModelHubMixin

class BinaryMemeClassifier(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self, model_name, learning_rate=2e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.train_accuracy = torchmetrics.classification.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.classification.Accuracy(task='binary')
        self.mcc = torchmetrics.classification.MatthewsCorrCoef(task="binary")
        self.kappa = torchmetrics.classification.CohenKappa(task="binary")
        self.f1 = torchmetrics.classification.F1Score(task="binary", average="weighted")

        final_layer_in_dim = self._load_pretrained_model()
        self.classifier = nn.Linear(final_layer_in_dim, 1)

        self.best_threshold = 0.5
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _load_pretrained_model(self) -> int:
        '''
        Load the pretrained model. Possible models are:
        - resnet18
        \\
        Returns: num_filters - the number of filters of the last layer of the model
        '''
        if self.model_name == 'resnet18':
            backbone = models.resnet18(weights="DEFAULT")
            input_shape = (224, 224)
    
        if self.model_name == 'densenet121':
            backbone = models.densenet121(weights="DEFAULT")
            input_shape = (224, 224)

        if self.model_name == 'efficientnet_v2_s':
            backbone = models.efficientnet_v2_s(weights="DEFAULT")
            input_shape = (224, 224)

        
        # Remove the last classification layer of the model used for ImageNet
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # Freeze the feature extractor
        final_layer_in_dim = self._get_conv_output(input_shape)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        return final_layer_in_dim
            
    def _get_conv_output(self, shape):
        '''Test the forward pass of the model to get the shape of the output tensor after the conv block
        \\
        Returns: n_size - the number of features in the output tensor
        '''

        batch_size = 1
        # Ensure the input tensor has 3 channels (RGB)
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, 3, *shape))


        output_feat = self._forward_features(tmp_input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        '''Pass the input through the feature extractor'''
        x = self.feature_extractor(x)
        return x


    # will be used during inference
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       x = torch.sigmoid(x)
       return x.squeeze(1)


    def training_step(self, batch):
        X, y = batch[0], batch[1].float()
        out = self.forward(X)
        loss = self.criterion(out, y)
        self.train_accuracy(out, y)
        self.log("train/loss", loss)
        self.log("train/acc",  self.train_accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y  = batch[0], batch[1].float()
        out = self.forward(X)
        loss = self.criterion(out, y)
        self.log("val/loss", loss)
        self.val_accuracy(out, y)
        self.log("val/acc", self.val_accuracy)

        self.mcc(out, y)
        self.log("val/mcc", self.mcc)
        self.f1(out, y)
        self.log("val/f1", self.f1)
        self.kappa(out, y)
        self.log("val/kappa", self.kappa)

        self.validation_step_outputs.append({"loss": loss, "probs": out, "targets": y})

        return loss
    
    def predict_step(self, batch, batch_idx):
        X, y = batch[0], batch[1].float()
        out = self.forward(X)
        loss = self.criterion(out, y)
        
        # Use the best threshold for predictions
        predictions = (out > self.best_threshold).float()
        
        return {"loss": loss, "outputs": out, "predictions": predictions, "y": y}
    
    def test_step(self, batch, batch_idx):
        X, y = batch[0], batch[1].float()
        out = self.forward(X)
        loss = self.criterion(out, y)
        
        # Use the best threshold for predictions
        predictions = (out > self.best_threshold).float()

        self.test_step_outputs.append({"loss": loss, "outputs": out, "predictions": predictions, "y": y})
        
        return {"loss": loss, "outputs": out, "predictions": predictions, "y": y}
    
    def on_validation_epoch_end(self):
        # Concatenate all validation outputs
        probs = torch.cat([x["probs"] for x in self.validation_step_outputs])
        targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        # Convert to numpy for sklearn functions
        probs = probs.cpu().numpy()
        targets = targets.cpu().numpy()

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(targets, probs)

        # Calculate G-Mean
        tnr = 1 - fpr
        gmeans = np.sqrt(tpr * tnr)

        # Find the threshold that gives the best G-Mean
        best_idx = np.argmax(gmeans)
        best_threshold = thresholds[best_idx]

        # Log metrics
        self.log("best_threshold", best_threshold)
        self.log("best_gmean", gmeans[best_idx])
        self.log("best_tpr", tpr[best_idx])
        self.log("best_fpr", fpr[best_idx])

        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_gmean", gmeans[best_idx])
        mlflow.log_metric("best_tpr", tpr[best_idx])
        mlflow.log_metric("best_fpr", fpr[best_idx])

        # Update the best threshold
        self.best_threshold = best_threshold
    
    def on_test_epoch_end(self):
        loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        output = torch.cat([x['outputs'] for x in self.test_step_outputs], dim=0)
        predictions = torch.cat([x['predictions'] for x in self.test_step_outputs], dim=0)
        y = torch.cat([x['y'] for x in self.test_step_outputs], dim=0)
        
        self.log("test/loss", loss)
        acc = self.accuracy(predictions, y)
        self.log("test/acc", acc)
        
        # Calculate and log test metrics
        fpr, tpr, _ = roc_curve(y.cpu().numpy(), output.cpu().numpy())
        tnr = 1 - fpr
        gmean = np.sqrt(tpr * tnr)
        
        self.log("test/gmean", gmean.max())
        self.log("test/tpr", tpr[np.argmax(gmean)])
        self.log("test/fpr", fpr[np.argmax(gmean)])
        
        self.test_ys = y
        self.test_output = output
        self.test_predictions = predictions


    def configure_optimizers(self):
        filtered_params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(filtered_params, lr=self.learning_rate)