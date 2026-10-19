#include <metal_stdlib>
using namespace metal;

struct HnetGradShape {
    uint tokens;
    uint proposal_start;
    uint vocabulary;
    float loss_scale;
};

inline void hnet_add(device atomic_uint* address, float value) {
    uint expected = atomic_load_explicit(address, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        address,
        &expected,
        as_type<uint>(as_type<float>(expected) + value),
        memory_order_relaxed,
        memory_order_relaxed)) {}
}

inline void hnet_bad(device atomic_uint* status, uint index, bool bad) {
    if (bad) atomic_fetch_add_explicit(status + index, 1u, memory_order_relaxed);
}

inline uint hnet_offset(uint level) {
    constexpr uint offsets[5] = {0, 8, 72, 584, 4680};
    return offsets[level];
}

inline float hnet_retention(float coordinate) {
    const float value = coordinate + 1.0986122886681098f;
    const float e = exp(-abs(value));
    return value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

inline float hnet_input_scale(float retention) {
    return sqrt(max(1.0f - retention * retention, 1e-6f));
}

// Exact cross-entropy adjoint for the six base-8 H-Net decisions. Generated
// token choices remain a fixed causal trace; reference-token probabilities
// supply the gradient at every decision.
kernel void hnet_loss_grad(
    device const half* hidden [[buffer(0)]],
    device const half* qkv [[buffer(1)]],
    device const half* positions [[buffer(2)]],
    device const half* readout [[buffer(3)]],
    device const half* centroids [[buffer(4)]],
    device const uint* expected [[buffer(5)]],
    device const uchar* score_mask [[buffer(6)]],
    device atomic_uint* hidden_grad [[buffer(7)]],
    device atomic_uint* qkv_grad [[buffer(8)]],
    device atomic_uint* position_grad [[buffer(9)]],
    device atomic_uint* readout_grad [[buffer(10)]],
    device atomic_uint* centroid_grad [[buffer(11)]],
    constant HnetGradShape& p [[buffer(12)]],
    device atomic_uint* status [[buffer(13)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint first = block * 64;
    if (first >= p.tokens) return;
    const uint row = p.proposal_start + block;
    threadgroup float query[64];
    threadgroup float logits[8];
    threadgroup float probabilities[8];
    threadgroup float query_grad[64];
    float prefix_carry = 0.0f;
    float base_grad = 0.0f;
    float previous_prefix = 0.0f;
    for (int local = int(min(64u, p.tokens - first)) - 1; local >= 0; --local) {
        const uint token = first + uint(local);
        const uint target = expected[token];
        if (tid < 64) {
            float base = 0.0f;
            for (uint k = 0; k < 512; ++k)
                base = fma(float(hidden[ulong(row) * 512 + k]),
                           float(qkv[ulong(k) * 640 + tid]), base);
            float prefix = 0.0f;
            previous_prefix = 0.0f;
            for (uint step = 0; step < uint(local); ++step) {
                const uint previous = expected[first + step];
                const float retention = hnet_retention(
                    float(positions[ulong(step) * 512 + tid]));
                previous_prefix = prefix;
                prefix = fma(retention, prefix, hnet_input_scale(retention)
                    * float(readout[ulong(tid) * p.vocabulary + previous]));
            }
            query[tid] = base * 0.0441941738f
                + float(positions[ulong(local) * 512 + tid]) + prefix;
            hnet_bad(status, 0, !isfinite(query[tid]));
            query_grad[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (score_mask[token] != 0) {
            for (uint level = 0; level < 6; ++level) {
                if (tid < 8) {
                    const uint parent = target >> ((6 - level) * 3);
                    const uint index = parent * 8 + tid;
                    float value = 0.0f;
                    if (level < 5) {
                        const uint node = hnet_offset(level) + index;
                        for (uint dim = 0; dim < 64; ++dim)
                            value = fma(query[dim],
                                float(centroids[ulong(node) * 64 + dim]), value);
                    } else {
                        for (uint dim = 0; dim < 64; ++dim)
                            value = fma(query[dim],
                                float(readout[ulong(dim) * p.vocabulary + index]), value);
                    }
                    logits[tid] = value;
                    hnet_bad(status, 1, !isfinite(value));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float maximum = logits[0];
                    for (uint child = 1; child < 8; ++child)
                        maximum = max(maximum, logits[child]);
                    float total = 0.0f;
                    for (uint child = 0; child < 8; ++child) {
                        probabilities[child] = exp(logits[child] - maximum);
                        total += probabilities[child];
                    }
                    for (uint child = 0; child < 8; ++child)
                        probabilities[child] /= total;
                    probabilities[(target >> ((5 - level) * 3)) & 7u] -= 1.0f;
                    bool bad = !isfinite(maximum) || !isfinite(total) || total <= 0.0f;
                    for (uint child = 0; child < 8; ++child)
                        bad = bad || !isfinite(probabilities[child]);
                    hnet_bad(status, 2, bad);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < 64) {
                    float dq = 0.0f;
                    const uint parent = target >> ((6 - level) * 3);
                    for (uint child = 0; child < 8; ++child) {
                        const uint index = parent * 8 + child;
                        const float coefficient = p.loss_scale * probabilities[child];
                        if (level < 5) {
                            const uint node = hnet_offset(level) + index;
                            dq = fma(coefficient,
                                float(centroids[ulong(node) * 64 + tid]), dq);
                            hnet_add(centroid_grad + ulong(node) * 64 + tid,
                                coefficient * query[tid]);
                        } else {
                            dq = fma(coefficient,
                                float(readout[ulong(tid) * p.vocabulary + index]), dq);
                            hnet_add(readout_grad + ulong(tid) * p.vocabulary + index,
                                coefficient * query[tid]);
                        }
                    }
                    query_grad[tid] += dq;
                    hnet_bad(status, 3, !isfinite(query_grad[tid]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (tid < 64) {
            const float dq = query_grad[tid];
            base_grad += dq;
            hnet_add(position_grad + ulong(local) * 512 + tid, dq);
            if (local > 0) {
                const float prefix_grad = dq + prefix_carry;
                hnet_bad(status, 4, !isfinite(prefix_grad));
                const uint previous_local = uint(local - 1);
                const uint previous = expected[token - 1];
                const float coordinate = float(
                    positions[ulong(previous_local) * 512 + tid]);
                const float retention = hnet_retention(coordinate);
                const float input_scale = hnet_input_scale(retention);
                const float embedding = float(
                    readout[ulong(tid) * p.vocabulary + previous]);
                hnet_add(readout_grad + ulong(tid) * p.vocabulary + previous,
                    input_scale * prefix_grad);
                const float transition_grad = prefix_grad
                    * retention * (1.0f - retention)
                    * (previous_prefix - retention * embedding / input_scale);
                hnet_bad(status, 5, !isfinite(transition_grad));
                hnet_add(position_grad + ulong(previous_local) * 512 + tid,
                    transition_grad);
                prefix_carry = retention * prefix_grad;
            }
            hnet_bad(status, 6, !isfinite(base_grad));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < 64) {
        for (uint k = 0; k < 512; ++k) {
            hnet_bad(status, 7,
                !isfinite(0.0441941738f * float(hidden[ulong(row) * 512 + k]) * base_grad)
                || !isfinite(0.0441941738f * float(qkv[ulong(k) * 640 + tid]) * base_grad));
            hnet_add(qkv_grad + ulong(k) * 640 + tid,
                0.0441941738f * float(hidden[ulong(row) * 512 + k]) * base_grad);
            hnet_add(hidden_grad + ulong(row) * 512 + k,
                0.0441941738f * float(qkv[ulong(k) * 640 + tid]) * base_grad);
        }
    }
}

// Reverse the normalized subtree sums used to construct hierarchy centroids.
// One thread owns one readout coordinate, so the final accumulation is atomic
// only because leaf-level loss gradients may already occupy the same address.
kernel void hnet_centroid_grad(
    device const float* centroid_grad [[buffer(0)]],
    device atomic_uint* readout_grad [[buffer(1)]],
    constant HnetGradShape& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(p.vocabulary) * 64;
    if (gid >= elements) return;
    const uint token = gid / 64;
    const uint dim = gid % 64;
    float value = 0.0f;
    for (uint level = 0; level < 5; ++level) {
        const uint shift = (5 - level) * 3;
        const uint node = hnet_offset(level) + (token >> shift);
        const uint leaves = 1u << shift;
        value += centroid_grad[ulong(node) * 64 + dim] * rsqrt(float(leaves));
    }
    hnet_add(readout_grad + ulong(dim) * p.vocabulary + token, value);
}
