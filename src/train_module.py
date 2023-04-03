
import torch
import pytorch_lightning as pl


class TrainModule(pl.LightningModule):
    def __init__(self, model, loss, metric, optimizer, scheduler):
        super(TrainModule, self).__init__()

        self.model = model
        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        """Get output from model.

        Args:
            x: torch.Tensor - batch of images.

        Returns:
            output: torch.Tensor - predicted logits.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):

        input_images, targets, target_lengths = batch
        output = self(input_images)

        input_lengths = [output.size(0) for _ in input_images]
        input_lengths = torch.LongTensor(input_lengths)
        target_lengths = torch.flatten(target_lengths)

        loss = self.loss(output, targets, input_lengths, target_lengths)

        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True)

        return loss

    # def training_epoch_end(self, outputs):

    def validation_step(self, batch, batch_idx):

        input_images, targets, target_lengths = batch
        output = self(input_images)

        input_lengths = [output.size(0) for _ in input_images]
        input_lengths = torch.LongTensor(input_lengths)
        target_lengths = torch.flatten(target_lengths)

        loss = self.loss(output, targets, input_lengths, target_lengths)

        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True)

        return loss

    # def validation_epoch_end(self, outputs):

    def test_step(self, batch, batch_idx):
        input_images, targets, target_lengths = batch
        output = self(input_images)

        input_lengths = [output.size(0) for _ in input_images]
        input_lengths = torch.LongTensor(input_lengths)
        target_lengths = torch.flatten(target_lengths)

        loss = self.loss(output, targets, input_lengths, target_lengths)

        self.log('test_loss', loss, prog_bar=True, logger=True, on_step=True)

        return loss

    def configure_optimizers(self):
        """To callback for configuring optimizers.

        Returns:
            optimizer: torch.optim - optimizer for PL.
        """
        return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.scheduler}}

    def on_save_checkpoint(self, checkpoint):
        """Save custom state dict.

        Function is needed, because we want only timm state_dict for scripting.

        Args:
            checkpoint: pl.checkpoint - checkpoint from PL.
        """
        checkpoint['my_state_dict'] = self.model.state_dict()
