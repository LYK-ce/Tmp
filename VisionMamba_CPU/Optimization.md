# Presented by KeJi
# Date: 2025-12-29

# Vision Mamba CPU Optimization Guide

## Performance Summary

| Configuration | Time (ms) | Speedup | Description |
|--------------|-----------|---------|-------------|
| Python-Original | 714.64 | 1.00x | Baseline |
| Python-Fixlen | 444.57 | 1.61x | Two-stage algorithm |
| Python-Fused | 291.76 | 2.45x | N-dimension concat |
| Python-Fused-Fixlen | 296.80 | 2.41x | Combined |
| CPP-Original | 384.64 | 1.86x | C++ basic |
| CPP-Fixlen | 243.70 | 2.93x | C++ two-stage |
| CPP-Fused | 198.61 | 3.60x | C++ fused |
| **CPP-Fused-Fixlen** | **179.78** | **3.98x** | **Best** |

Test Platform: RK3588 (ARM aarch64), PyTorch 2.8.0+cpu

---

## Optimization Techniques

### 1. Two-Stage Algorithm (Fixlen)

**Problem**: Original selective scan uses a sequential loop computing both state and output at each step.

```python
# Original: O(L) sequential operations, each with state update + output computation
for i in range(L):
    x = deltaA[:,:,i] * x + deltaB_u[:,:,i]  # state update
    y = einsum('bdn,bn->bd', x, C[:,:,i])    # output computation
```

**Solution**: Separate state propagation and output computation into two phases.

```python
# Phase 1: State propagation only (in-place)
for i in range(1, L):
    deltaB_u[:,:,i] += deltaA[:,:,i] * deltaB_u[:,:,i-1]

# Phase 2: Batch output computation (vectorized)
y = einsum('bdln,bnl->bdl', deltaB_u, C)
```

**Benefits**:
- Phase 1: Sequential but minimal computation per step
- Phase 2: Fully vectorized, leverages SIMD

**Speedup**: 1.61x (Python), 2.93x (C++)

---

### 2. Fused Bidirectional Scan

**Problem**: Vision Mamba requires bidirectional scan, which traditionally calls `selective_scan` twice (forward + backward).

```python
# Original: Two separate scans
y_fwd = selective_scan(x, delta, A_fwd, B, C, ...)
y_bwd = selective_scan(x.flip(-1), delta.flip(-1), A_bwd, B.flip(-1), C.flip(-1), ...)
y = y_fwd + y_bwd.flip(-1)
```

**Solution**: Concatenate forward and backward parameters along N dimension, compute both directions in a single scan.

```python
# Fused: Single scan with 2N state dimension
deltaA_bi = cat([deltaA_fwd, deltaA_bwd], dim=-1)      # (B,D,L,2N)
deltaB_u_bi = cat([deltaB_u_fwd, deltaB_u_bwd], dim=-1) # (B,D,L,2N)

# Single recurrence loop handles both directions
for i in range(1, L):
    deltaB_u_bi[:,:,i] += deltaA_bi[:,:,i] * deltaB_u_bi[:,:,i-1]
```

**Benefits**:
- Eliminates duplicate overhead (memory allocation, loop setup)
- Better cache utilization (single buffer vs two separate buffers)
- Reduces Python interpreter overhead

**Speedup**: 2.45x (Python), 3.60x (C++)

---

### 3. Memory Pre-allocation (Eliminate torch.cat)

**Problem**: `torch.cat` allocates new memory and copies data.

```python
# Slow: cat allocates new tensor + copies twice
deltaA_bi = torch.cat([
    torch.exp(einsum('bdl,dn->bdln', dt_fwd, A_fwd)),
    torch.exp(einsum('bdl,dn->bdln', dt_bwd, A_bwd))
], dim=-1)
```

**Solution**: Pre-allocate buffer and write directly to slices.

```python
# Fast: Pre-allocate + direct write
deltaA_bi = torch.empty(B, D, L, 2*N, ...)
deltaA_bi[:,:,:,:N] = torch.exp(einsum('bdl,dn->bdln', dt_fwd, A_fwd))
deltaA_bi[:,:,:,N:] = torch.exp(einsum('bdl,dn->bdln', dt_bwd, A_bwd))
```

**Benefits**:
- Eliminates memory allocation overhead
- Avoids data copy
- Reduces memory fragmentation

---

### 4. Memory Layout Optimization (Avoid Permute)

**Problem**: Initial optimization attempt used `(B,D,2N,L)` layout for cache-friendly L-dimension access, but required `permute` to match einsum output.

```python
# Bad: permute adds overhead
deltaA_bi[:,:,:N,:] = einsum('bdl,dn->bdln', ...).permute(0,1,3,2)
```

**Solution**: Use `(B,D,L,2N)` layout which matches einsum output directly.

```python
# Good: No permute needed
deltaA_bi[:,:,:,:N] = einsum('bdl,dn->bdln', ...)  # Output is already (B,D,L,N)
```

**Trade-off Analysis**:
- `(B,D,2N,L)`: Better cache for recurrence loop, but permute overhead
- `(B,D,L,2N)`: No permute overhead, recurrence still efficient

**Result**: Permute overhead > cache benefit on ARM, use `(B,D,L,2N)`.

---

### 5. C++ Extension

**Problem**: Python interpreter overhead in tight loops.

```python
# Python: Each iteration has interpreter overhead
for i in range(1, L):
    deltaB_u[:,:,i] += deltaA[:,:,i] * deltaB_u[:,:,i-1]
```

**Solution**: Implement in C++ with PyTorch C++ API.

```cpp
// C++: Native loop, no interpreter overhead
for (int64_t i = 1; i < seq_len; i++) {
    auto deltaA_i = deltaA_bi.select(2, i);
    auto prev = deltaB_u_bi.select(2, i - 1);
    deltaB_u_bi.select(2, i).add_(deltaA_i * prev);
}
```

**Benefits**:
- No Python GIL
- Better compiler optimization
- Tensor view operations (select/narrow) are zero-copy

**Speedup**: 1.86x over Python baseline

---

### 6. In-place Operations

**Problem**: Creating new tensors for intermediate results.

```python
# Allocates new tensor
deltaB_u_bi[:,:,i] = deltaA_bi[:,:,i] * deltaB_u_bi[:,:,i-1] + deltaB_u_bi[:,:,i]
```

**Solution**: Use in-place operations.

```python
# In-place: No allocation
deltaB_u_bi[:,:,i] += deltaA_bi[:,:,i] * deltaB_u_bi[:,:,i-1]
```

```cpp
// C++ in-place
deltaB_u_bi.select(2, i).add_(deltaA_i * prev);
```

**Benefits**:
- Reduces memory allocation
- Better cache utilization

---

## Implementation Files

- **Python**: [`selective_scan_interface.py`](mamba-1p1p1/mamba_ssm/ops/selective_scan_interface.py)
  - `selective_scan_ref`: Original implementation
  - `selective_scan_ref_fixlen`: Two-stage optimization
  - `selective_fused_scan_ref`: Fused bidirectional
  - `selective_fused_scan_ref_fixlen`: Best Python version

- **C++**: [`selective_scan.cpp`](mamba-1p1p1/mamba_ssm/ops/selective_scan.cpp)
  - `Selective_Scan_Ref_Cpu`: Original implementation
  - `Selective_Scan_Ref_Fixlen_Cpu`: Two-stage optimization
  - `Selective_Fused_Scan_Cpu`: Fused bidirectional
  - `Selective_Fused_Scan_Fixlen_Cpu`: Best C++ version (3.98x)

---

## Usage

### Build C++ Extension

```bash
cd VisionMamba_CPU/mamba-1p1p1/mamba_ssm/ops
python setup.py build_ext --inplace
```

### Use in Model

```python
from mamba_ssm.ops.selective_scan_interface import selective_fused_scan_fn

# Best performance: use_cpp=True, use_fixlen=True
out = selective_fused_scan_fn(
    dt_fwd, dt_bwd, A_fwd, A_bwd, B_fwd, B_bwd,
    x_fwd_conv, x_bwd_conv_flip,
    C_fwd, C_bwd, D_fwd, D_bwd,
    z_fwd, z_bwd_flip,
    use_cpp=True, use_fixlen=True
)
```

---

## Future Optimizations

1. **Parallel Scan Algorithm**: For GPU or long sequences
2. **Chunked Recurrence**: Process L in chunks for better cache
3. **Loop Unrolling**: Manual unroll in C++ for reduced branch overhead
4. **NEON/SVE Intrinsics**: ARM-specific SIMD optimization
5. **Memory Pool**: Reuse allocated buffers across calls
