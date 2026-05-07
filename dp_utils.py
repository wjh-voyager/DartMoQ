import time
import numpy as np
from typing import List, Dict, Tuple

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

if __name__ == "__main__":
    np.random.seed(42)
    n_neurons = 1024
    
    # ================= 配置搜索空间（灵活调整这里） =================
    # 示例1：完整搜索空间 0-6
    # bits = [0, 1, 2, 3, 4, 5, 6]
    
    # 示例2：只搜索 2-4（和原代码一致）
    # bits = [2, 3, 4]
    
    # 示例3：搜索 1,2,3,4（常用场景）
    bits = [2, 3, 4]
    # =================================================================
    
    # 生成模拟rates（0bit可以设一个较大的惩罚值）
    rates = {}
    # 先生成基础r（比如对应4bit）
    r_base = np.random.rand(n_neurons)
    # 按bit数越低，损失越高的原则生成（模拟量化损失）
    # 这里简单模拟：bit数每降1，损失乘以1.3 + 噪声
    for bit in sorted(bits, reverse=True):
        if bit == max(bits):
            rates[bit] = r_base
        else:
            higher_bit = bit + 1
            # 确保higher_bit在rates中（如果bits不连续，需要调整逻辑）
            while higher_bit not in rates and higher_bit <= max(bits):
                higher_bit += 1
            if higher_bit not in rates:
                rates[bit] = r_base * (1.5 ** (max(bits) - bit)) + np.random.rand(n_neurons) * 0.1
            else:
                rates[bit] = rates[higher_bit] * 1.3 + np.random.rand(n_neurons) * 0.1
    
    # 特殊处理0bit（如果包含）：设为一个固定的高惩罚
    if 0 in bits:
        rates[0] = np.full(n_neurons, 10.0)  # 0bit惩罚
    
    # 参数设置
    s = 8  # 分块数
    target_bpw = 2.5  # 目标平均位宽
    epsilon = 0.1  # 允许的误差范围
    
    print(f"Neuron Level DP Config:bits={bits}, s={s}, target_bpw={target_bpw}, epsilon={epsilon}")
    
    # 运行极速版m-scheme搜索
    print("\n--- Fast m-scheme Search ---")
    tick = time.time()
    best_scheme, neuron_bits_fast = enum_optimal_m_scheme_fast_general(
        rates, s, target_bpw, epsilon
    )
    print(f"Fast m-scheme Search Time: {time.time() - tick:.4f} s")
    
    # （可选）运行神经元级DP作为对比
    # print("\n--- 运行神经元级DP（对比） ---")运行神经元级DP作为对比
    # print("\n--- 运行神经元级DP（对比） ---")
    # tick = time.time()
    # neuron_bits_dp = neuron_level_dp_general(rates, bits, target_bpw, epsilon)
    # print(f"耗时: {time.time() - tick:.4f} s")