from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    prompt_ids = []
    output_ids = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids.append(torch.tensor(tokenizer.encode(prompt, add_special_tokens=False)))
        output_ids.append(torch.tensor(tokenizer.encode(output, add_special_tokens=False)))

    prompt_and_output_lens =[len(p) + len(o) for p, o in zip(prompt_ids, output_ids)]
    max_len = max(prompt_and_output_lens)
    
    input_ids = []
    labels = []
    response_masks = []
    
    for p,o in zip(prompt_ids, output_ids):
        input_id = torch.cat([p,o])
        response_mask = torch.cat([torch.zeros_like(p,dtype=torch.bool), torch.ones_like(o,dtype=torch.bool)])
        padding_len = max_len - len(input_id)
        pad_input_id = F.pad(input_id, (0,padding_len), value=tokenizer.pad_token_id)
        pad_response_mask = F.pad(response_mask, (0,padding_len), value=False)
        
        input_ids.append(pad_input_id[:-1])  # slice off final token
        labels.append(pad_input_id[1:])  # shifted input_ids
        response_masks.append(pad_response_mask[1:])

    input_ids_tensor = torch.stack(input_ids)
    labels_tensor = torch.stack(labels)
    response_mask_tensor = torch.stack(response_masks)

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "response_mask": response_mask_tensor
    }
        

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    logp = F.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    entropy = -(p * logp).sum(dim=-1)
    return entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_softmax = F.log_softmax(logits, dim=-1)
        # 根据labels获取每个token对应的log prob
        token_log_probs = log_softmax.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    if not return_token_entropy:
        return {"log_probs": token_log_probs}
    else:
        return {"log_probs": token_log_probs, "token_entropy": compute_entropy(logits)}

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked = tensor * mask
    return torch.sum(masked,dim=dim) / normalize_constant



def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    
    # ce loss
    loss = -masked_normalize(policy_log_probs,response_mask,None,normalize_constant)

    loss = loss / gradient_accumulation_steps
    # Backward pass.
    loss.backward()
    
    return loss, {}
