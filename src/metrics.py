import torch
import math

def multi_acc(y, y_hat):
    """y_hat, y : (bs, n_labels) or (bs*n_labels,)"""
    return 100. * (y_hat == y).float().mean().item()

def comupte_perplexity(loss):
    try: return math.exp(loss)
    except OverflowError: return float("inf")

def get_compute_metrics_lm(task):
    assert task in ["clm", "mlm"]
    if task == "clm" :
        def compute_metrics_lm(y, logits, loss, mask_token_index=None, prefix=""):
            y = y.cpu()
            y_hat = logits.cpu().detach().argmax(dim=-1)
            # Shift so that tokens < n predict n
            y = y[:, 1:].reshape(-1) # (bs*n_labels,)
            y_hat = y_hat[:, :-1].reshape(-1) # (bs*n_labels,)
            return {
                '%sacc'%prefix : multi_acc(y=y, y_hat=y_hat), 
                "%sppl"%prefix : comupte_perplexity(loss)
            }
    else :
        def compute_metrics_lm(y, logits, loss, mask_token_index, prefix=""):
            y = y.cpu()
            y_hat = logits.cpu().detach().argmax(dim=-1)
            # Select masked tokens
            y = y[mask_token_index].reshape(-1) # (bs*n_labels,)
            y_hat = y_hat[mask_token_index].reshape(-1) # (bs*n_labels,)
            return {
                '%sacc'%prefix : multi_acc(y=y, y_hat=y_hat), 
                "%sppl"%prefix : comupte_perplexity(loss)
            }
        
    return compute_metrics_lm