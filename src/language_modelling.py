import torch
import torch.optim as optim
import pytorch_lightning as pl

from .metrics import get_compute_metrics_lm

OPTIMIZER_DIC = {"Adam": optim.Adam}

class LMLightningModule(pl.LightningModule):
    """Language Modeling (CLM and MLM) Model"""
    def __init__(
        self,
        model,
        task : str,
        optimizer_name: str,
        learning_rate: float,
        lr_factor: float,
        lr_patience: int,
        decoder_start_token_id : int = None
    ):
        """
        model : transformer model
        task (str) : mlm or clm
        optimizer_name (str) : optimizer name : Adam, ...
        learning_rate (float) : learning rate
        lr_factor (float) : learning rate scheduler factor
        lr_patience (int) : learning rate scheduler patience
        decoder_start_token_id (int) : start token of sentences for text generation (clm only)
        """
        super(LMLightningModule, self).__init__()
        self.model = model
        self.task = task
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.optimizer_name = optimizer_name
        self.decoder_start_token_id = decoder_start_token_id
        self.compute_metrics_lm = get_compute_metrics_lm(task)
        
    def configure_optimizers(self):
        optimizer = OPTIMIZER_DIC[self.optimizer_name](
            self.model.parameters(), lr=self.learning_rate
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=self.lr_factor, patience=self.lr_patience
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            #'interval': 'step',
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }
        return output
            
    def _compute_loss(self, features, prefix=""):
        mask_token_index = features.pop("mask_token_index", None)
        output = self.model(**features)
        loss = output.loss 
        output = self.compute_metrics_lm(features["labels"], output.logits, loss, mask_token_index, prefix)
        output["%sloss"%prefix] = loss.item()
        return loss, output
    
    def training_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch, prefix="train_")
        output["loss"] = loss
        self.log_dict(output, prog_bar=True)
        return output

    def validation_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch, prefix="val_")
        self.log_dict(output, prog_bar=True)
        return output

    def test_step(self, batch, batch_idx):
        loss, output = self._compute_loss(batch, prefix="test_")
        self.log_dict(output)
        return output

    def predict_step(self, batch, batch_idx):
        return self.fill_mask(batch) if self.task == "mlm" else self.generate(batch)

    def fill_mask(self, features):
        "MLM only"
        mask_token_index = features.pop("mask_token_index", None)
        output = self.model(**features)
        logits = output.logits.cpu().detach()
        y_hat = torch.log_softmax(logits, dim=-1).argmax(dim=-1)
        output = {
            "input_ids" : features["input_ids"], 
            #"labels" : features["labels"][:,mask_token_index], 
            "labels" : features["labels"][mask_token_index], 
            "output_ids" : y_hat, 
            #"pred_labels" : y_hat[:,mask_token_index],
            "pred_labels" : y_hat[mask_token_index]
        }
        return output

    def generate(self, features):
        """CLM only : Greedy search / Beam search / Top-K sampling / Top-p sampling."""
        input_ids = features["input_ids"]
        type_ = getattr(self, "type", None)
        max_length = self.max_length
        if type_ is None :
            output_ids = self.model.generate(input_ids, max_length=max_length, decoder_start_token_id=self.decoder_start_token_id)
        # https://huggingface.co/blog/how-to-generate
        elif type_ == "greedy_search" :
            # Greedy search
            output_ids = self.model.generate(input_ids, max_length=max_length)
        elif type_ == "beam_search" :
            # Beam search
            #output_ids = self.model.generate(input_ids, max_length=max_length, num_beams=getattr(self, "num_beams", 2), early_stopping=getattr(self, "early_stopping", True))
            # Beam search with n-gram penalties
            output_ids = self.model.generate(
                input_ids, max_length=max_length, 
                num_beams=getattr(self, "num_beams", 2), 
                early_stopping=getattr(self, "early_stopping", True), 
                no_repeat_ngram_size=getattr(self, "no_repeat_ngram_size", 1)
            )
            # # Beam search with n-gram penalties, compare the top beams after generation and choose the generated beam that fits our purpose best (num_return_sequences <= num_beams).
            # output_ids = self.model.generate(input_ids, max_length=max_length, 
            #     num_beams=getattr(self, "num_beams", 2), early_stopping=getattr(self, "early_stopping", True), 
            #     no_repeat_ngram_size=getattr(self, "no_repeat_ngram_size", 1), num_return_sequences=5
            # )#[0]
        elif type_ == "top-k_sampling" :
            # Sampling
            #output_ids = self.model.generate(input_ids, max_length=max_length, do_sample=True, top_k=0)
            # Sampling, use temperature to decrease the sensitivity to low probability candidates
            output_ids = self.model.generate(input_ids, max_length=max_length, 
                do_sample=getattr(self, "do_sample", True), top_k=getattr(self, "top_k", 0), temperature=getattr(self, "temperature", None))
            # Top-K Sampling
            #output_ids = self.model.generate(input_ids, max_length=max_length, do_sample=getattr(self, "do_sample", True), top_k=getattr(self, "top_k", 0))
        elif type_ == "top-p_sampling" :
            # Top-p (nucleus) sampling
            output_ids = self.model.generate(input_ids, max_length=max_length, do_sample=getattr(self, "do_sample", True), 
                    top_k=getattr(self, "top_k", 0), top_p=getattr(self, "top_p", 0))
            # # Top-p (nucleus) sampling, compare the top beams after generation and choose the generated beam that fits our purpose best 
            # output_ids = self.model.generate(input_ids, max_length=max_length, do_sample=getattr(self, "do_sample", True), 
            #         top_k=getattr(self, "top_k", 0), top_p=getattr(self, "top_p", 0), num_return_sequences=5)#[0]
        else :
            output_ids = self.model.generate(input_ids, max_length=max_length, decoder_start_token_id=self.decoder_start_token_id)

        output = {
            "input_ids" : input_ids, 
            "output_ids" : output_ids
        }
        
        if features.get("labels", None) is not None : output["labels"] = features["labels"]
        return output


