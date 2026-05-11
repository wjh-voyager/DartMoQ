# /home/daodao/Develop/DartMoQ/dartmoq_hybridmoe.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DartMoQHybridMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # First level: original experts (routers)
        self.experts = nn.ModuleList()
        self.gate = None
        
        # Meta information
        self.sub_expert_sizes = []
        self.sub_expert_bit_configs = []
        self.expert_to_subexperts = []
        
        # Shared experts
        self.shared_experts = None
        
        # Return type flag
        self.return_tuple = False  # Default to single tensor return

    def forward(self, hidden_states):
        """Forward pass through hybrid MoE (inference only)"""
        identity = hidden_states
        orig_shape = hidden_states.shape
        
        # First level routing (handle different gate return types)
        gate_output = self.gate(hidden_states)
        
        # Handle different gate return formats
        if isinstance(gate_output, tuple):
            if len(gate_output) == 3:
                topk_idx, topk_weight, _ = gate_output
            elif len(gate_output) == 2:
                topk_idx, topk_weight = gate_output
            else:
                topk_idx, topk_weight = gate_output[0], gate_output[1]
        else:
            # For simple Linear gates, output is just logits
            router_logits = gate_output
            topk_weight, topk_idx = torch.topk(F.softmax(router_logits, dim=-1), self.num_experts_per_tok)
        
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        # Inference mode with efficient implementation
        y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # Add shared experts output if present
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        
        if self.return_tuple:
            return y, None
        else:
            return y
        
    def _forward_sub_experts(self, sub_experts, hidden_states):
        """
        Forward through sub-experts within an expert group.
        Sub-experts may have different sizes (intermediate_size).
        """
        if len(sub_experts) == 1:
            return sub_experts[0](hidden_states)
        
        # For multiple sub-experts, concatenate outputs along hidden dimension
        outputs = []
        for sub_expert in sub_experts:
            outputs.append(sub_expert(hidden_states))
        return torch.cat(outputs, dim=-1)

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """Efficient inference implementation for hybrid MoE"""
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            
            # Get sub-experts for this expert
            sub_experts = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            
            # Forward through sub-experts
            expert_out = self._forward_sub_experts(sub_experts, expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        
        return expert_cache

    def set_shared_experts(self, shared_experts):
        """Set shared experts from original model"""
        self.shared_experts = shared_experts


def restructure_hybrid_qscheme(qscheme_expert, slice_expert_num):
    """
    Restructure qscheme for hybrid MoE: group sub-experts by bit config.
    
    Each expert's sub-experts with the same bit config are merged into a single sub-expert.
    
    Args:
        qscheme_expert: Original qscheme with shape [n_experts, slice_expert_num]
        slice_expert_num: Number of slices per expert
    
    Returns:
        Restructured qscheme with shape [n_experts, n_unique_bits_per_expert]
    """
    restructured = []
    for expert_idx in range(len(qscheme_expert)):
        # Count occurrences of each bit config
        bit_counts = {}
        for bit in qscheme_expert[expert_idx]:
            bit_counts[bit] = bit_counts.get(bit, 0) + 1
        
        # Create list of unique bits, sorted by bit value (descending), consistent with sub-expert creation order
        expert_bits = sorted(bit_counts.items(), reverse=True)
        restructured.append([bit for bit, count in expert_bits])
        
        print(f"Expert {expert_idx} original: {qscheme_expert[expert_idx]} -> restructured: {restructured[-1]}")
    
    return restructured