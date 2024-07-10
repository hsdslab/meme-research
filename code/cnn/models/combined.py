import torchmetrics.classification
import torchvision.models as models
import torch.nn as nn
import torch
import lightning as L
import torchmetrics
import pytorch_lightning as pl

class CombinedMemeClassifier(pl.LightningModule):
    def __init__(self, binary_classifier, template_classifier,  num_target_classes=1145):
        super().__init__()
        # Load pre-trained models
        self.binary_classifier = binary_classifier
        self.template_classifier = template_classifier
        # Set models to evaluation mode
        self.binary_classifier.eval()
        self._freeze_model(self.binary_classifier)
        self.template_classifier.eval()
        self._freeze_model(self.template_classifier)
        self.val_accuracy = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_target_classes)
        self.mcc = torchmetrics.classification.MatthewsCorrCoef(task="multiclass",num_classes=num_target_classes)
        self.kappa = torchmetrics.classification.CohenKappa(task="multiclass",num_classes=num_target_classes)
        self.f1 = torchmetrics.classification.F1Score(task="multiclass",num_classes=num_target_classes, average="weighted")

        self.best_threshold = binary_classifier.best_threshold

    def _freeze_model(self,model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Perform inference with the binary classifier
        meme_likelihood = self.binary_classifier(x)
        # Initialize default output (using a marker for no template)
        template_output = torch.full((x.size(0),), -1, dtype=torch.long, device=self.device)

        # Check which items are likely memes and process only those
        meme_indices = meme_likelihood > self.best_threshold
        if meme_indices.any():
            memes = x[meme_indices]
            if memes.size(0) > 0:
                # Perform inference with the template classifier
                prediction_results = self.template_classifier.predict_step(memes, None)
                predicted_classes = prediction_results['predicted_classes']
                template_output[meme_indices] = predicted_classes

        return meme_likelihood, template_output

    # for nonlabeled meme
    # def predict_step(self, batch, batch_idx):
    #     X, paths = batch  # Assume the batch is just the images
    #     meme_likelihood, template_output = self(X)
    #     return {"meme_likelihood": meme_likelihood, "template_output": template_output, "paths": paths}
    
    # for labeled memes
    def predict_step(self, batch, batch_idx):
        X, y = batch[0], batch[1]
        meme_likelihood, template_output = self(X)
        return {"meme_likelihood": meme_likelihood, "template_output": template_output, "y": y}
