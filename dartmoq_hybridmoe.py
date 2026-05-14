import torch
import torch.nn as nn
import torch.nn.functional as F

class DartMoQHybridMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        
        self.experts = nn.ModuleList()
        self.gate = None
        
        self.sub_expert_sizes = []
        self.sub_expert_bit_configs = []
        self.expert_to_subexperts = []
        
        self.shared_experts = None
        
        self.return_tuple = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        num_experts = len(self.experts)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

        for expert_idx in range(num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            output = self._forward_sub_experts(expert_layer, current_state)
            current_hidden_states = output * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)

        if self.return_tuple:
            return final_hidden_states, router_logits
        else:
            return final_hidden_states

    def _forward_sub_experts(self, sub_experts, hidden_states):
        assert len(sub_experts) > 0
        
        # Process each sub-expert and accumulate results
        total_output = torch.zeros_like(hidden_states)
        
        for sub_expert in sub_experts:
            sub_expert_output = sub_expert(hidden_states)
            total_output = total_output + sub_expert_output
            
        return total_output

    def set_shared_experts(self, shared_experts):
        self.shared_experts = shared_experts

class DartMoQMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class DartMoQLinear(nn.Module):
    def __init__(self, all_new_experts):
        super().__init__()
        self.experts = nn.ModuleList([nn.ModuleList(sub_experts) for sub_experts in all_new_experts])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert len(self.experts) > 0
        
        # Process each sub-expert and accumulate results
        total_output = torch.zeros_like(hidden_states)
        
        for sub_expert in self.experts:
            sub_expert_output = sub_expert(hidden_states)
            total_output = total_output + sub_expert_output
        
        return total_output

def restructure_hybrid_qscheme(qscheme_expert, slice_expert_num):

    restructured = []
    for expert_idx in range(len(qscheme_expert)):
        # Count occurrences of each bit config
        bit_counts = {}
        for bit in qscheme_expert[expert_idx]:
            bit_counts[bit] = bit_counts.get(bit, 0) + 1
        
        # Create list of unique bits, sorted by bit value (descending), consistent with sub-expert creation order
        expert_bits = sorted(bit_counts.items(), reverse=True)
        restructured.append([bit for bit, count in expert_bits])
        
        # print(f"Expert {expert_idx} original: {qscheme_expert[expert_idx]} -> restructured: {restructured[expert_idx]} {expert_bits}")
    
    return restructured