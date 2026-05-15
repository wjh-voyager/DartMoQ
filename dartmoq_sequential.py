import torch
import torch.nn as nn
import os
import time
import re
from dartmoq_utils import *
from data_utils import *
from eval_dartmoq import cmoe_ppl_eval
from camera_utils import analyze_expert_energy
from dp_utils import enum_optimal_m_scheme_fast_general
from dp_utils import extrapolate_0bit_loss
from tool_utils import *
from collections import Counter
from dartmoq_hybridmoe import DartMoQHybridWrapper
from dartmoq_hybridmoe import restructure_hybrid_qscheme

DEV = torch.device('cuda:0')

@torch.no_grad()
def reconstruct_moe_from_existing(model, layer, layer_idx, inps, 
                                  n_experts, n_activated, slice_expert_num, 
                                  ori_activated, device, qscheme, use_hybrid_moe, args):

    if "global" in args.quant_scheme :
        expert_activation_rates = analyze_experts_activation(layer, layer_idx, inps, ori_activated, model.config.model_type)

    ori_expert_num = len(layer.mlp.experts)
    
    if use_hybrid_moe:
        # Hybrid MoE: keep original expert count at first level
        new_expert_num = ori_expert_num
    else:
        new_expert_num = ori_expert_num * slice_expert_num 
        scaling_factor = slice_expert_num

    ori_router_gate = layer.mlp.gate.weight
    
    if use_hybrid_moe:
        # Hybrid MoE uses nested structure: experts -> sub-experts
        all_new_experts = []  # List of lists (each expert has sub-experts)
    else:
        if type(layer.mlp.gate) == nn.Linear:
            new_router = nn.Linear(model.config.hidden_size, new_expert_num, dtype=ori_router_gate.dtype, bias=False).to(device)
        else:
            new_router = layer.mlp.gate.__class__(model.config).to(device).to(layer.mlp.gate.weight.dtype)
        all_new_experts = nn.ModuleList()

    total_neurons_processed = 0
    gate_start_idx = 0
    
    # For hybrid MoE, we need to track sub-expert bit configs
    sub_expert_bit_configs = []
    expert_to_subexperts = []

    probe_bit = 2
    if args.rank_mode == "quant_outlier":
        tick0 = time.time()

        q_rates = {}
        if 'target_bpw' not in qscheme:
            outlier_bits = {probe_bit}
        else:
            outlier_bits = {0, 1, 2, 3, 4}
        print(f"simulate quant outlier_bits {outlier_bits}")

        cache_dir = f"quant_outlier_/{model.model_id}"
        os.makedirs(cache_dir, exist_ok=True)
        
        for x in sorted(outlier_bits, reverse=True):  ## 0 bit should be extrapolated from other bit data, so we compute it at last
            cache_path = os.path.join(cache_dir, f"{model.model_id}_L{layer_idx}_b{x}.pt")
            if os.path.exists(cache_path):
                try:
                    cached_data = torch.load(cache_path, map_location=device)
                    print(f"Loading cached quant outlier data for layer {layer_idx}, wbits={x}", flush=True)
                    q_rates[x] = cached_data
                    continue
                except Exception as e:
                    print(f"Failed to load cached data {e}")
            
            if x == 0:
                print(f"Computing extrapolate 0 bit loss for layer {layer_idx}")
                q_rates[0] = extrapolate_0bit_loss(q_rates)
                q_rates[0] = [torch.from_numpy(q_rates[0][i]).to(device) for i in range(len(q_rates[0]))]
            else:
                print(f"Computing quant outlier for layer {layer_idx}, wbits={x}")
                q_rates[x] = analyze_quant_outlier(layer, layer_idx, inps, ori_expert_num, wbits=x, save_path=None)
            torch.save(q_rates[x], cache_path)
            print(f"Saved quant outlier data to {cache_path}")
        
        if 'target_bpw' not in qscheme:
            all_rates = q_rates[probe_bit]
        else:
            all_rates = []
            dpscheme_list = []
            for expert_idx in range(ori_expert_num):
                rates_x = {}
                for x in outlier_bits:
                    rates_x[x] = q_rates[x][expert_idx].detach().cpu().numpy()
                # print(f"expert_idx {expert_idx} scheme search:")
                dpscheme, rates = enum_optimal_m_scheme_fast_general(rates_x, slice_expert_num, target_bpw=qscheme['target_bpw'])
                dpscheme_list.append(dpscheme)
                rates = torch.from_numpy(rates).to(device)
                all_rates.append(rates)
            
        # from visual_utils import plot_diff_wbits_correlation, plot_spearman_rank_correlation
        # # plot_diff_wbits_correlation(model.config.model_type, layer_idx, ori_expert_num, q_rates[2], q_rates[3], q_rates[4])
        # plot_spearman_rank_correlation(model.config.model_type, layer_idx, ori_expert_num, q_rates[2], q_rates[3], q_rates[4])
        tick1 = time.time()
        print(f"analyze quant outlier time {tick1 - tick0}", flush=True)

    tick0 = time.time()

    all_new_expert_rates = []
    all_expert_groups = []  # Store groups for each expert
    
    for expert_idx, expert in enumerate(layer.mlp.experts):
        # print(f"\nProcessing original expert {expert_idx} / {ori_expert_num}")
        if args.rank_mode == "activation":
            ori_gate_proj_weights = expert.gate_proj.weight
            ori_up_proj_weights = expert.up_proj.weight
            ori_down_proj_weights = expert.down_proj.weight

            analyze_sparsity = 0.1
            rates = analyze_neuron_activations(expert.act_fn, inps, ori_gate_proj_weights, ori_up_proj_weights, sparsity=analyze_sparsity)
        elif args.rank_mode == "energy":
            rates = analyze_expert_energy(expert, inps)
        elif args.rank_mode == "quant_outlier":
            rates = all_rates[expert_idx]
        elif args.rank_mode == "random":
            rates = torch.randn(layer.mlp.intermediate_size, device=device)
        elif args.rank_mode == "neuron_index":
            rates = torch.arange(layer.mlp.intermediate_size, device=device)
        else:
            assert False, f"Unknown rank mode: {args.rank_mode}"
        
        expert_groups, expert_rates = construct_experts_by_rates(
            rates,
            num_experts = slice_expert_num
        )
        
        expert_groups = expert_groups[1:]
        all_expert_groups.append(expert_groups)
        
        if "global" in args.quant_scheme :
            _rates = [e * expert_activation_rates[expert_idx] for e in expert_rates[1:]]
            all_new_expert_rates.extend(_rates)
        else:
            all_new_expert_rates.extend(expert_rates[1:])

    # print(qscheme)
    if 'target_bpw' in qscheme:
        qscheme['expert'] = dpscheme_list
        counter = Counter(dpscheme_list)
        print(f"layer {layer_idx} {qscheme['target_bpw']} dpscheme_list scheme type count: {counter}")
    elif "global" in args.quant_scheme :
        ee = qscheme['econfig']
        e_bits = [int(e) for e in ee]

        if all_new_expert_rates is not None:
            _, sorted_index = torch.sort(torch.tensor(all_new_expert_rates), descending=True)
            # print(e_bits, sect_, new_expert_num)
            qscheme['expert'] = [[0] * slice_expert_num for i in range(ori_expert_num)]
            for i, idx in enumerate(sorted_index):
                # print(idx, all_new_expert_rates[idx])
                xi = int(idx // slice_expert_num)
                xj = int(idx % slice_expert_num)
                qscheme['expert'][xi][xj] = e_bits[i // ori_expert_num]
    else:
        qscheme['expert'] = [qscheme['econfig'] for i in range(ori_expert_num)]
    
    # For hybrid MoE: restructure qscheme to group by bit config
    if use_hybrid_moe:
        qscheme['slice_expert'] = qscheme['expert']
        qscheme['expert'] = restructure_hybrid_qscheme(qscheme['slice_expert'], slice_expert_num)

    for expert_idx, expert in enumerate(layer.mlp.experts):
        ori_gate_proj_weights = expert.gate_proj.weight
        ori_up_proj_weights = expert.up_proj.weight
        ori_down_proj_weights = expert.down_proj.weight
        
        # Get groups for this specific expert
        expert_groups = all_expert_groups[expert_idx]

        if use_hybrid_moe:
            # Hybrid MoE: group sub-experts by bit config
            expert_sub_experts = []
            expert_sub_sizes = []

            orig_bit_config = qscheme['slice_expert'][expert_idx]
            restructured_config = qscheme['expert'][expert_idx]
            # print("orig_bit_config:", orig_bit_config, "restructured_config:", restructured_config)

            bit_to_indices = {}
            bit_to_slice_count = {}
            
            for bit, group_indices in zip(orig_bit_config, expert_groups):
                if bit not in bit_to_indices:
                    bit_to_indices[bit] = []
                    bit_to_slice_count[bit] = 0
                bit_to_indices[bit].extend(group_indices)
                bit_to_slice_count[bit] += 1
            
            for bit in restructured_config:
                indices = bit_to_indices[bit]
                n_neurons = len(indices)

                # print(f"layer {layer_idx} expert {expert_idx} bit={bit} n_neurons={n_neurons}")
                # print(bit, indices)
                # if expert_idx < 2:
                #     print(f"layer {layer_idx} expert {expert_idx} bit={bit} n_neurons={n_neurons}, indices[:5]={indices[:5]} {indices[-5:]}")
                new_config = model.config
                new_config.intermediate_size = n_neurons
                expert_mlp = expert.__class__(new_config).to(device)
                
                with torch.no_grad():
                    indices_tensor = torch.tensor(indices, dtype=torch.long, device=ori_gate_proj_weights.device)
                    expert_mlp.gate_proj.weight.data = ori_gate_proj_weights[indices_tensor, :].detach().clone()
                    expert_mlp.up_proj.weight.data = ori_up_proj_weights[indices_tensor, :].detach().clone()
                    expert_mlp.down_proj.weight.data = ori_down_proj_weights[:, indices_tensor].detach().clone()
                
                expert_sub_experts.append(expert_mlp)
                expert_sub_sizes.append(n_neurons)
                total_neurons_processed += n_neurons
            
            all_new_experts.append(expert_sub_experts)
            sub_expert_bit_configs.append(tuple(restructured_config))
            expert_to_subexperts.append(list(range(len(expert_sub_experts))))
            
            # For hybrid MoE, router stays the same (one entry per original expert)
        else:
            # Original behavior: create separate expert for each slice
            for ii, group_indices in enumerate(expert_groups):
                n_neurons = len(group_indices)
                # if expert_idx < 2:
                #     print(f"layer {layer_idx} expert {expert_idx} slice {ii} n_neurons={n_neurons}, group_indices={group_indices[:5]} {group_indices[-5:]}")
                new_config = model.config
                new_config.intermediate_size = n_neurons
                expert_mlp = expert.__class__(new_config).to(device)
                
                with torch.no_grad():
                    group_indices_tensor = torch.tensor(group_indices, dtype=torch.long, device=ori_gate_proj_weights.device)
                    expert_mlp.gate_proj.weight.data = ori_gate_proj_weights[group_indices_tensor, :].detach().clone()
                    expert_mlp.up_proj.weight.data = ori_up_proj_weights[group_indices_tensor, :].detach().clone()
                    expert_mlp.down_proj.weight.data = ori_down_proj_weights[:, group_indices_tensor].detach().clone() * scaling_factor
                
                all_new_experts.append(expert_mlp)
                new_expert_intermediate_size = expert_mlp.up_proj.weight.shape[0]
                total_neurons_processed += new_expert_intermediate_size
            
            expanded_gate = ori_router_gate.data[expert_idx, :].unsqueeze(0).repeat(slice_expert_num, 1).to(device).detach().clone()
            new_router.weight.data[gate_start_idx: gate_start_idx + slice_expert_num, :] = expanded_gate
            gate_start_idx += slice_expert_num

        del ori_gate_proj_weights, ori_up_proj_weights, ori_down_proj_weights
        if 'group_indices_tensor' in locals():
            del group_indices_tensor
        if 'expanded_gate' in locals():
            del expanded_gate
        gc.collect()
        torch.cuda.empty_cache()

    tick1 = time.time()
    print(f"Layer {layer_idx}, {args.rank_mode} expert re- sort time: {tick1 - tick0}", flush=True)
    print("all_new_expert_rates:", len(all_new_expert_rates))

    if use_hybrid_moe:
        # Create hybrid MoE using the original model's MLP class
        moe = layer.mlp.__class__(model.config).to(device)
        
        # Keep gate and top_k configuration consistent with original
        moe.gate = layer.mlp.gate
        moe.num_experts = len(all_new_experts)
        
        # Replace experts with nn.ModuleList of DartMoQHybridWrapper wrappers
        # Each DartMoQHybridWrapper wraps multiple sub-experts with different bit configs
        moe.experts = nn.ModuleList([DartMoQHybridWrapper(sub_experts) for sub_experts in all_new_experts])
        
        counter = Counter(sub_expert_bit_configs)
        print("reconstruct moe with sub_expert_bit_configs: ", counter) 
        
        # Copy shared_experts if exists
        if hasattr(layer.mlp, 'shared_experts'):
            moe.shared_experts = layer.mlp.shared_experts
        moe.training = False
    else:
        # Original behavior
        moe = layer.mlp.__class__(model.config).to(device)
        moe.num_experts = len(all_new_experts)
        moe.top_k = n_activated
        moe.gate = new_router
        moe.experts = all_new_experts
        if hasattr(layer.mlp, 'shared_experts'):
            moe.shared_experts = layer.mlp.shared_experts
    gc.collect()
    torch.cuda.empty_cache()

    return moe

@torch.no_grad()
def construct_moe(model, moe_model_flag, layer, layer_idx, inp, 
                    attention_mask, position_ids, position_embeddings, 
                    n_experts, n_activated, slice_expert_num, ori_activated, 
                    qscheme, args):
    
    modeltype = model.config.model_type
    batchsize = inp.shape[0]

    device = next(layer.parameters()).device
    # print(layer, device)
    # print(inp.shape)

    # Forward attention
    inp = inp.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
    residual = inp
    with torch.no_grad():
        hidden_states_inorm = layer.input_layernorm(inp)

    # tick0 = time.time()
    attn_out = torch.zeros_like(hidden_states_inorm)
    for b_i in range(0, batchsize):
        # print(modeltype)
        if modeltype == 'olmoe' or modeltype == 'llama' or modeltype == 'qwen3' or modeltype == 'qwen3_moe' or modeltype == 'deepseek_v3':
            with torch.no_grad():
                attn_out[b_i:b_i+1] = layer.self_attn(
                    hidden_states=hidden_states_inorm[b_i:b_i+1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings)[0]
        else:
            with torch.no_grad():
                attn_out[b_i:b_i+1] = layer.self_attn(
                    hidden_states=hidden_states_inorm[b_i:b_i+1],
                    attention_mask=attention_mask, 
                    position_ids=position_ids)[0]
    # tick1 = time.time()
    # print(f"Inference in origin attention layer {layer_idx} with batch size {batchsize} time: {tick1 - tick0}")

    hidden_states = residual + attn_out
    residual = hidden_states
    with torch.no_grad():
        hidden_states = layer.post_attention_layernorm(hidden_states)

    # print(hidden_states.shape)
    is_moe_layer = hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts') ## some moe model has no expert layer in the first few layers,
    
    tick0 = time.time()
    use_hybrid_moe = getattr(args, 'use_hybrid_moe', False)

    if moe_model_flag:
        if is_moe_layer:
            moe = reconstruct_moe_from_existing(model, layer, layer_idx, hidden_states, 
                                                n_experts, n_activated, slice_expert_num, ori_activated, device,
                                                qscheme, use_hybrid_moe, args)
            layer.mlp = moe
    else:
        # moe = reconstruct_moe_from_dense(model, layer, layer_idx, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
        # layer.mlp = moe
        assert False, "Dense model is not supported"
    gc.collect()
    torch.cuda.empty_cache()
    tick1 = time.time()
    print(f"reconstruct_moe_from_existing layer {layer_idx} time: {tick1 - tick0}", flush=True)
    
    tick0 = time.time()
    if_quant_attn = True
    quant_layer_mix_precision(layer, layer_idx, if_quant_attn, n_experts, slice_expert_num,
                hidden_states_inorm, hidden_states, attention_mask, position_ids, position_embeddings, 
                qscheme, use_hybrid_moe)
    gc.collect()
    torch.cuda.empty_cache()
    tick1 = time.time()
    print(f"quant_layer_mix_precision layer {layer_idx} time: {tick1 - tick0}", flush=True)

    # tick0 = time.time()
    moe_out = torch.zeros_like(hidden_states)
    for b_i in range(0, batchsize):
        mlp_out = layer.mlp(hidden_states[b_i:b_i+1])
        if isinstance(mlp_out, tuple):
            moe_out[b_i:b_i+1] = mlp_out[0]
        else:
            moe_out[b_i:b_i+1] = mlp_out

    with torch.no_grad():
        moe_out = moe_out + residual

    del hidden_states, hidden_states_inorm, residual, attn_out

    gc.collect()
    torch.cuda.empty_cache()

    # tick1 = time.time()
    # print(f"Inference in new moe layer {layer_idx} with batch size {batchsize} time: {tick1 - tick0}", flush=True)
    return moe_out

@torch.no_grad()
def cmoe_sequential(model, tokenizer, dataloader, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 1
    
    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    print(inps.shape)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    if args.standby_layer_cpu:
        model.model.embed_tokens = model.model.embed_tokens.to(DEV)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):

            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(DEV))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    # print("position_embeddings:", position_embeddings)
    # print(cache)

    print('Ready.')
    # model.cuda()
    # layers.cuda()

    # MoE Carving
    moe_model_flag = False
    for layer in layers:
        moe_model_flag = moe_model_flag or hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')
    
    use_hybrid_moe = getattr(args, 'use_hybrid_moe', False)
    
    if moe_model_flag:
        slice_expert_num = args.slices

        if hasattr(model.config, 'num_experts'):
            ori_num_experts = model.config.num_experts
            if use_hybrid_moe:
                # Hybrid MoE keeps original expert count at first level
                new_num_expert = model.config.num_experts
            else:
                new_num_expert = slice_expert_num * model.config.num_experts
                model.config.num_experts = new_num_expert
        elif hasattr(model.config, 'n_routed_experts'):
            ori_num_experts = model.config.n_routed_experts
            if use_hybrid_moe:
                new_num_expert = model.config.n_routed_experts
            else:
                new_num_expert = slice_expert_num * model.config.n_routed_experts
                model.config.n_routed_experts = new_num_expert
        
        ori_num_experts_per_tok = model.config.num_experts_per_tok
        if use_hybrid_moe:
            # Hybrid MoE keeps original activation count
            new_num_experts_per_tok = model.config.num_experts_per_tok
        else:
            new_num_experts_per_tok = slice_expert_num * model.config.num_experts_per_tok
            model.config.num_experts_per_tok = new_num_experts_per_tok
        
        # For hybrid MoE, we don't change intermediate_size since sub-experts have different sizes
        if not use_hybrid_moe:
            if hasattr(model.config, 'moe_intermediate_size'):
                model.config.moe_intermediate_size = model.config.moe_intermediate_size // slice_expert_num
            elif hasattr(model.config, 'intermediate_size'):
                model.config.intermediate_size = model.config.intermediate_size // slice_expert_num
        
        if use_hybrid_moe:
            print("The model is already a MoE model. Proceeding to create hybrid MoE structure.")
            print(f"Hybrid MoE: {ori_num_experts} experts with sub-experts sliced by {slice_expert_num}")
        else:
            print("The model is already a MoE model. Proceeding to split experts. ")
            print(f"Slice expert by {slice_expert_num}: to {new_num_expert}, with {new_num_experts_per_tok} activated experts.")
    else:
        assert False, "Dense model is not supported."
    
    inps = inps.squeeze(1)

    if args.standby_layer_cpu:
        layers_device = []
        for layer_idx, layer in enumerate(model.model.layers):
            dev = next(layer.parameters()).device
            layers_device.append(dev)
            # print(layer_idx, dev)
            if dev.type == 'cuda':
                layer = layer.to('cpu')
        for i in range(torch.cuda.device_count()):
            force_release_inactive_splits(device=i)
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")       
        # print(layers_device)
    
    qscheme_str = args.quant_scheme
    qscheme = {}
    # qscheme['attn'] = [8]
    # qscheme['share'] = [4]
    # qscheme['expert'] = [2, 2, 2, 2, 2, 2, 2, 2]
    try:
        # sample: "a8s4m3221", "a8s4m33222222"
        match = re.search(r'a(\d)s(\d)m([\d.]+)', qscheme_str)
        aa = match.group(1)
        ss = match.group(2)
        ee = match.group(3)
        qscheme['attn'] = [int(aa)]
        qscheme['share'] = [int(ss)]
        if 'bpw' not in qscheme_str:
            assert len(ee) == slice_expert_num
            qscheme['econfig'] = [int(e) for e in ee]
            bpw = sum(qscheme['econfig']) * 1.0 / slice_expert_num
        else:
            bpw = float(ee)
            qscheme['target_bpw'] = bpw
        print(f"Quant expert scheme (ppl): {qscheme_str} with bpw {bpw} qscheme {qscheme}")
    except:
        assert False, f"Quant scheme {qscheme_str} is not valid."

    for layer_idx, layer in enumerate(layers):
        tick0 = time.time()
        if args.standby_layer_cpu:
            layer = layer.to(layers_device[layer_idx])

        moe_out = construct_moe(model,
            moe_model_flag,
            layer, 
            layer_idx,
            inps, 
            attention_mask, 
            position_ids,
            position_embeddings,
            n_experts = new_num_expert,
            n_activated = new_num_experts_per_tok,
            slice_expert_num = slice_expert_num,
            ori_activated = ori_num_experts_per_tok,
            qscheme = qscheme,
            args = args
        )

        inps = moe_out

        if args.standby_layer_cpu:
            layer = layer.to('cpu')

        for i in range(torch.cuda.device_count()):
            # force_release_inactive_splits(device=i) # force to release inactive reserved memory
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")

        tick1 = time.time()
        print(f"Layer {layer_idx} total reconstruct and quantization time: {tick1 - tick0:.2f} s", flush=True)
    
    print("MoE reconstruction and quantization done. Moving layers to GPU for evaluation...")

    if args.standby_layer_cpu:
        for i in range(torch.cuda.device_count()):
            force_release_inactive_splits(device=i)
        for layer_idx, layer in enumerate(model.model.layers):
            if layers_device[layer_idx].type == 'cuda':
                layer = layer.to(layers_device[layer_idx])
            for i in range(torch.cuda.device_count()):
                print(f"layer {layer_idx} CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
                print(f"layer {layer_idx} CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")
        
        # for name, param in model.named_parameters():
        #     print(f"{name:<40} → {param.device}")

    # print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    
    return model