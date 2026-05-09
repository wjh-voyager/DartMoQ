from pyexpat import model
import os
import time
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict

def _exp_decay_func(b: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    return A * np.exp(B * b) + C

def _poly2_func(b: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    return A * (b ** 2) + B * b + C

def _power_growth_func(x: np.ndarray, A: float, C: float, D: float, x_max: float) -> np.ndarray:
    return A * np.power(x_max - x + 1, C) + D

def extrapolate_0bit_loss(rates: Dict[int, List[np.ndarray]], save_plots: bool = False) -> List[np.ndarray]:
    bits = sorted(rates.keys())
    if 0 in bits:
        bits.remove(0)
        rates.pop(0)
    # print(bits)
    assert len(bits) >= 2, "at least 2 bits are required for extrapolation of 0bit loss"
    
    x_max = max(bits) + 1.0

    n_experts = len(rates[bits[0]])
    for b in bits:
        assert len(rates[b]) == n_experts, f"bit {b} has inconsistent number of experts"
    
    L0 = []
    
    for expert_idx in range(n_experts):
        print(f"Processing extrapolate 0bit loss for expert {expert_idx}")

        expert_rates = {}
        n_neurons = None
        for b in bits:
            expert_rates[b] = rates[b][expert_idx].detach().cpu().numpy()
            if n_neurons is None:
                n_neurons = len(expert_rates[b])
            else:
                assert len(expert_rates[b]) == n_neurons, \
                    f"expert {expert_idx}, bit {b} has inconsistent loss array length"
        
        b_array = np.array(bits, dtype=float)
        expert_L0 = np.zeros(n_neurons, dtype=float)
        
        for i in range(n_neurons):
            loss_array = np.array([expert_rates[b][i] for b in bits])

            # --------------------- log quadratic fit ---------------------
            # function: log(loss) = p*b^2 + q*b + r
            # loss(b) = exp(p*b^2 + q*b + r)
            # -------------------------------------------------------------
            try:
                log_loss = np.log(loss_array)

                p, q, r = np.polyfit(b_array, log_loss, deg=2)

                l0 = np.exp(r)

                l1 = expert_rates[1][i] if 1 in expert_rates else expert_rates[bits[0]][i]
                if l0 < l1:
                    l0 = l1 * 2.0

            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                l1 = expert_rates[1][i] if 1 in expert_rates else expert_rates[bits[0]][i]
                l0 = l1 * 2.0

            expert_L0[i] = l0

            if save_plots:
                print(p, q, r)
                plt.figure(figsize=(7,4))

                plt.scatter(bits, loss_array, color='red', s=10, label='Original loss (1,2,3,4...)')

                b_dense = np.linspace(0.001, max(bits), 100)
                y_dense = np.exp(p * b_dense**2 + q * b_dense + r)
                plt.plot(b_dense, y_dense, 'b-', label=f'Log-quad fit (exp(pb²+qb+r))')

                plt.scatter(0, l0, color='green', s=10, label=f'L0 = {l0:.2f}')
                plt.scatter(1, l1, color='orange', s=10, label=f'L1 = {l1:.2f}')

                plt.title(f'Expert {expert_idx} | Neuron {i} | L0={l0:.2f}, L1={l1:.2f}')
                plt.xlabel('bit')
                plt.ylabel('loss')
                plt.yscale('log')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'plot/bit_loss_fit/exp2_expert_{expert_idx}_neuron_{i}.png', dpi=150)
                plt.close()
        
        L0.append(expert_L0)

    return L0

def generate_valid_m_schemes_general(bits, s, target_bpw, epsilon):
    """
    generate_valid_m_schemes_general(bits, s, target_bpw, epsilon)
    incremental backtracking enumeration to generate all valid m-schemes that satisfy the bpw constraint (general bit version)
    """
    bits_sorted = sorted(bits, reverse=True)
    min_bit = bits_sorted[-1]
    max_bit = bits_sorted[0]
    
    target_total = target_bpw * s
    min_total = min_bit * s
    max_total = max_bit * s
    target_total_clipped = np.clip(target_total, min_total, max_total)
    valid_schemes = []
    
    def backtrack(pos, curr_total, curr_scheme, last_bit):
        if pos == s:
            if abs(curr_total - target_total_clipped) <= epsilon * s:
                valid_schemes.append(tuple(curr_scheme))
            return
        
        remaining = s - pos
        if curr_total + min_bit * remaining > target_total_clipped + epsilon * s:
            return
        if curr_total + max_bit * remaining < target_total_clipped - epsilon * s:
            return
        
        for bit in bits_sorted:
            if bit <= last_bit:
                backtrack(pos + 1, curr_total + bit, curr_scheme + [bit], bit)
    
    for first_bit in bits_sorted:
        backtrack(1, first_bit, [first_bit], first_bit)
    
    return valid_schemes

def get_unified_sorted_idx_general(rates: Dict[int, np.ndarray], bits: List[int]) -> np.ndarray:
    """
    core step 1: unified marginal gain sorting (general bit version)
    sorting criterion: combined marginal gain of adjacent bits
    """
    bits_sorted = sorted(bits)
    n_neurons = len(rates[bits_sorted[0]])
    idx = np.arange(n_neurons)
    
    if len(bits_sorted) == 1:
        return idx
    
    # compute combined score of marginal gains: sum(adjacent bit gains)
    combined_score = np.zeros(n_neurons)
    for i in range(len(bits_sorted)-1):
        low_bit = bits_sorted[i]
        high_bit = bits_sorted[i+1]
        # gain: high_bit replacing low_bit (rates[low] - rates[high])
        gain = rates[low_bit] - rates[high_bit]
        combined_score += gain
    
    sorted_idx = idx[np.argsort(-combined_score)]
    return sorted_idx

def precompute_block_losses_general(sorted_idx, rates, bits, s):
    n_neurons = len(sorted_idx)
    assert n_neurons % s == 0, "number of neurons must be divisible by the number of blocks"
    m = n_neurons // s
    
    bit_to_idx = {b: i for i, b in enumerate(bits)}
    block_losses = np.zeros((len(bits), s))
    
    for k in range(s):
        start = k * m
        end = start + m
        idx_in_block = sorted_idx[start:end]
        for bit in bits:
            bit_idx = bit_to_idx[bit]
            block_losses[bit_idx, k] = rates[bit][idx_in_block].sum()
    
    return block_losses, bit_to_idx

def enum_optimal_m_scheme_fast_general(rates, s, target_bpw, epsilon = 0):

    bits = list(rates.keys())
    n_neurons = len(rates[bits[0]])
    for b in bits[1:]:
        assert len(rates[b]) == n_neurons, f"rates[{b}] length must be consistent with other bits"
    
    sorted_idx = get_unified_sorted_idx_general(rates, bits)
    
    block_losses, bit_to_idx = precompute_block_losses_general(sorted_idx, rates, bits, s)
    
    valid_schemes = generate_valid_m_schemes_general(bits, s, target_bpw, epsilon)
    if not valid_schemes:
        raise ValueError(f"No valid m scheme found for target_bpw={target_bpw}, please adjust parameters")
    
    best_loss = float('inf')
    best_scheme = None
    
    for scheme in valid_schemes:
        total_loss = 0.0
        for k, bit in enumerate(scheme):
            bit_idx = bit_to_idx[bit]
            total_loss += block_losses[bit_idx, k]
        
        # print(f"Scheme: {scheme}, Total Loss: {total_loss:.4f}")
        if total_loss < best_loss:
            best_loss = total_loss
            best_scheme = scheme
    
    print(f"{len(valid_schemes)} valid schemes... Optimal Scheme: {best_scheme}, Minimum Loss: {best_loss:.4f}")
    
    m = n_neurons // s
    neuron_bits = np.zeros(n_neurons, dtype=int)
    for k, bit in enumerate(best_scheme):
        start = k * m
        end = start + m
        neuron_bits[sorted_idx[start:end]] = bit
    
    # print("Neuron Bit Width Statistics:")
    # for b in sorted(bits):
    #     count = np.sum(neuron_bits == b)
    #     print(f"  {b}bit: {count} neurons")
    # actual_bpw = np.mean(neuron_bits)
    # print(f"Actual Average Bit Width (BPW): {actual_bpw:.4f}")
    
    return best_scheme, neuron_bits

def neuron_level_dp_general(
    rates: Dict[int, np.ndarray],
    bits: List[int],
    target_bpw: float,
    epsilon: float = 0
) -> np.ndarray:

    bits_sorted = sorted(bits)
    n_neurons = len(rates[bits_sorted[0]])
    min_bit = bits_sorted[0]
    max_bit = bits_sorted[-1]
    
    assert min_bit <= target_bpw <= max_bit, f"target_bpw must be in [{min_bit}, {max_bit}]"
    
    target_total_w = round(target_bpw * n_neurons)
    min_total_w = min_bit * n_neurons
    max_total_w = max_bit * n_neurons
    target_total_w = int(np.clip(target_total_w, min_total_w, max_total_w))
    
    offset = min_total_w
    max_offset_w = max_total_w - offset
    target_offset_w = target_total_w - offset
    
    INF = float('inf')
    prev_dp = np.full(max_offset_w + 1, INF)
    prev_dp[0] = 0.0
    choice_history = []
    
    for i in range(n_neurons):
        curr_dp = np.full(max_offset_w + 1, INF)
        curr_choice = np.full(max_offset_w + 1, -1, dtype=int)
        
        for w_prev in range(max_offset_w + 1):
            if prev_dp[w_prev] == INF:
                continue
            
            for bit in bits_sorted:
                w_curr = w_prev + (bit - min_bit)
                if w_curr <= max_offset_w:
                    new_loss = prev_dp[w_prev] + rates[bit][i]
                    if new_loss < curr_dp[w_curr]:
                        curr_dp[w_curr] = new_loss
                        curr_choice[w_curr] = bit
        
        prev_dp = curr_dp
        choice_history.append(curr_choice)
    
    search_range = int(epsilon * n_neurons)
    best_w = -1
    best_loss = INF
    for w in range(max(0, target_offset_w - search_range),
                   min(max_offset_w, target_offset_w + search_range) + 1):
        if prev_dp[w] < best_loss:
            best_loss = prev_dp[w]
            best_w = w
    
    if best_w == -1:
        raise ValueError("No feasible solution found, please check target_bpw and epsilon")
    
    neuron_bits = np.zeros(n_neurons, dtype=int)
    current_w = best_w
    for i in reversed(range(n_neurons)):
        choice = choice_history[i][current_w]
        neuron_bits[i] = choice
        current_w -= (choice - min_bit)
    
    # print("Neuron Level DP Results Statistics:")
    # for b in sorted(bits):
    #     count = np.sum(neuron_bits == b)
    #     print(f"  {b}bit: {count} neurons")
    # actual_bpw = np.mean(neuron_bits)
    # print(f"Actual BPW: {actual_bpw:.4f}, Total Loss: {best_loss:.4f}")
    
    return neuron_bits

def test_dp_utils():
    np.random.seed(42)
    n_neurons = 1024
    
    # bits = [0, 1, 2, 3, 4, 5, 6]
    
    bits = [2, 3, 4]
    
    rates = {}
    r_base = np.random.rand(n_neurons)

    for bit in sorted(bits, reverse=True):
        if bit == max(bits):
            rates[bit] = r_base
        else:
            higher_bit = bit + 1
            while higher_bit not in rates and higher_bit <= max(bits):
                higher_bit += 1
            if higher_bit not in rates:
                rates[bit] = r_base * (1.5 ** (max(bits) - bit)) + np.random.rand(n_neurons) * 0.1
            else:
                rates[bit] = rates[higher_bit] * 1.3 + np.random.rand(n_neurons) * 0.1
    
    if 0 in bits:
        rates[0] = np.full(n_neurons, 10.0)
    
    s = 8 
    target_bpw = 2.5
    epsilon = 0.1
    
    print(f"Neuron Level DP Config:bits={bits}, s={s}, target_bpw={target_bpw}, epsilon={epsilon}")
    
    print("\n--- Fast m-scheme Search ---")
    tick = time.time()
    best_scheme, neuron_bits_fast = enum_optimal_m_scheme_fast_general(
        rates, s, target_bpw, epsilon
    )
    print(f"Fast m-scheme Search Time: {time.time() - tick:.4f} s")
    

def test_read_rates_from_file():
    outlier_bits = {1, 2, 3, 4, }
    print(f"simulate quant outlier_bits {outlier_bits}")

    model_id = "deepseek-v1-moe-16b"
    layer_idx = 1
    cache_dir = f"quant_outlier_/{model_id}"
    
    p = 20
    rates = {}
    for x in outlier_bits:
        cache_path = os.path.join(cache_dir, f"{model_id}_L{layer_idx}_b{x}.pt")
        # print(cache_path)
        if os.path.exists(cache_path):
            try:
                import torch
                cached_data = torch.load(cache_path, map_location='cpu')
                print(f"Loading cached quant outlier data for layer {layer_idx}, wbits={x}")
                rates[x] = [cached_data[0][:p]]
            except Exception as e:
                print(f"Failed to load cached data: {e}")

    rates[0] = extrapolate_0bit_loss(rates, save_plots=True)
    for i in range(p):
        print(i, end=',')
        print(f"{rates[4][0][i].item():.4f}", end=',')
        print(f"{rates[3][0][i].item():.4f}", end=',')
        print(f"{rates[2][0][i].item():.4f}", end=',')
        print(f"{rates[1][0][i].item():.4f}", end=',')
        print(f"{rates[0][0][i].item():.4f}", end=',')
        print()
    # print(L0[0][:10])

if __name__ == "__main__":
    # test_extrapolate_0bit_loss()
    test_read_rates_from_file()
    # test_dp_utils()
