import torchmetrics.classification
import torchvision.models as models
import torch.nn as nn
import torch
import lightning as L
import torchmetrics
import pytorch_lightning as pl
from huggingface_hub import PyTorchModelHubMixin

class ImagenetTransferLearning(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self, model_name, learning_rate=2e-4, num_target_classes=1145):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_target_classes = num_target_classes
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_target_classes)
        self.val_accuracy = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_target_classes)
        self.mcc = torchmetrics.classification.MatthewsCorrCoef(task="multiclass",num_classes=num_target_classes)
        self.kappa = torchmetrics.classification.CohenKappa(task="multiclass",num_classes=num_target_classes)
        self.f1 = torchmetrics.classification.F1Score(task="multiclass",num_classes=num_target_classes, average="weighted")

        final_layer_in_dim = self._load_pretrained_model()
        self.classifier = nn.Linear(final_layer_in_dim, num_target_classes)

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
       
       return x


    def training_step(self, batch):
        X, y = batch[0], batch[1]
        out = self.forward(X)
        loss = self.criterion(out, y)
        self.train_accuracy(out, y)
        self.log("train/loss", loss)
        self.log("train/acc",  self.train_accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y  = batch[0], batch[1]
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

        return loss
    
    def predict_step(self, batch, batch_idx):
        X = batch
        logits = self.forward(X)
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        predicted_classes = torch.argmax(probabilities, dim=1)  # Get the predicted classes
        return {"logits": logits, "probabilities": probabilities, "predicted_classes": predicted_classes}
    
    def test_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
        out = self.forward(X)
        loss = self.criterion(out, y)
        
        return {"loss": loss, "outputs": out, "y": y}
    
    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        output = torch.cat([x['outputs'] for x in outputs], dim=0)
        
        y = torch.cat([x['y'] for x in outputs], dim=0)
        
        self.log("test/loss", loss)
        acc = self.accuracy(output, y)
        self.log("test/acc", acc)
        
        self.test_ys = y
        self.test_output = output


    def configure_optimizers(self):
        filtered_params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(filtered_params, lr=self.learning_rate)