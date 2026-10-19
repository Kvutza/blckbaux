#include <metal_stdlib>
using namespace metal;

struct DraftCaptureShape {
    uint vectors;
    uint slot;
    uint row_start;
    uint context;
};

// Capture target hidden states without scalar conversion. The target scorer
// overwrites its activation buffer at every layer, so selected visits must be
// copied before the next layer starts.
kernel void fbt_draft_capture(
    device const half4* input [[buffer(0)]],
    device half4* features [[buffer(1)]],
    constant DraftCaptureShape& shape [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < shape.vectors) {
        const ulong slot_vectors = ulong(shape.context) * 512ul / 4ul;
        const ulong row_vectors = ulong(shape.row_start) * 512ul / 4ul;
        features[ulong(shape.slot) * slot_vectors + row_vectors + gid] = input[gid];
    }
}

// EncodeShape and its layout assertions come from blckbaux_semantics::hnet.

// Candidate weights use the canonical dimension-major readout layout. H-Net's
// fine encoder instead consumes complete token vectors, so transpose once per
// candidate and make both the transpose reads and encoder reads contiguous.
kernel void fbt_hnet_transpose(
    device const half* readout [[buffer(0)]],
    device half* tokens [[buffer(1)]],
    constant EncodeShape& shape [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup half tile[16][17];
    const uint token = group.x * 16 + tid.x;
    const uint dimension = group.y * 16 + tid.y;
    if (token < shape.vocabulary && dimension < shape.width)
        tile[tid.y][tid.x] = readout[ulong(dimension) * shape.vocabulary + token];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint output_token = group.x * 16 + tid.y;
    const uint output_dimension = group.y * 16 + tid.x;
    if (output_token < shape.vocabulary && output_dimension < shape.width)
        tokens[ulong(output_token) * shape.width + output_dimension] = tile[tid.x][tid.y];
}

// Compress every observed 64-token block with a learned, order-sensitive
// projection. Future blocks are represented by the learned mask vector. The
// operation is linear in fine tokens and produces the coarse sequence consumed
// by the global H-Net core.
kernel void fbt_hnet_encode(
    device const half* readout [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device const half* positions [[buffer(2)]],
    device const half* mask [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant EncodeShape& shape [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(shape.rows) * shape.width;
    if (ulong(gid) >= elements) return;
    const uint row = gid / shape.width;
    const uint dimension = gid % shape.width;
    float value = 0.0f;
    uint count = 0;
    for (uint intra = 0; intra < shape.patch; ++intra) {
        const uint token = tokens[ulong(row) * shape.patch + intra];
        if (token < shape.vocabulary) {
            value = fma(
                float(positions[ulong(intra) * shape.width + dimension]),
                float(readout[ulong(token) * shape.width + dimension]),
                value);
            ++count;
        }
    }
    output[gid] = count == 0
        ? mask[dimension]
        : half(value * sqrt(float(shape.patch) / float(count)));
}

struct HnetCaptureShape {
    uint rows;
    uint row_start;
    uint width;
};

kernel void fbt_hnet_capture(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant HnetCaptureShape& shape [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong elements = ulong(shape.rows) * shape.width;
    if (ulong(gid) < elements)
        output[ulong(shape.row_start) * shape.width + gid] = input[gid];
}

struct HnetExpandShape {
    uint tokens;
    uint proposal_start;
    uint patch;
    uint qkv_width;
    uint index_width;
    uint vocabulary;
    float temperature;
    uint score;
};

METAL_FUNC ulong fbt_hnet_mix(ulong value) {
    value += 0x9e3779b97f4a7c15ul;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

// Blocks execute in parallel. Tokens within a block execute causally, using a
// recurrent summary of the sampled prefix. This preserves the 64x H-Net
// contraction without imposing conditional independence inside each block.
// The online log-sum-exp is the exact free-running token objective; it shares
// every dot product with Gumbel-max and never materializes a logits tensor.
kernel void fbt_hnet_expand(
    device const half* coarse [[buffer(0)]],
    device const half* qkv [[buffer(1)]],
    device const half* positions [[buffer(2)]],
    device const half* readout [[buffer(3)]],
    device const ulong* seeds [[buffer(4)]],
    device const uint* expected [[buffer(5)]],
    device uint* output [[buffer(6)]],
    device float* losses [[buffer(7)]],
    constant HnetExpandShape& shape [[buffer(8)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint first = block * shape.patch;
    if (first >= shape.tokens) return;
    threadgroup half query[64];
    threadgroup float base[64];
    threadgroup float prefix[64];
    threadgroup float logit_maxima[8];
    threadgroup float logit_sums[8];
    threadgroup float expected_logits[8];
    threadgroup float sample_maxima[8];
    threadgroup uint sample_indices[8];
    const uint row = shape.proposal_start + block;
    if (tid < 64) {
        float value = 0.0f;
        for (uint input = 0; input < 512; ++input)
            value += float(coarse[ulong(row) * 512 + input])
                * float(qkv[ulong(input) * shape.qkv_width + tid]);
        base[tid] = value * 0.0441941738f;
        prefix[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint intra = 0; intra < shape.patch; ++intra) {
        const uint position = first + intra;
        if (position >= shape.tokens) break;
        if (tid < 64) {
            if (intra > 0) {
                const uint previous = output[position - 1];
                prefix[tid] = fma(
                    0.75f,
                    prefix[tid],
                    float(readout[ulong(tid) * shape.vocabulary + previous]));
            }
            const float scale = intra == 0 ? 0.0f : rsqrt(float(intra));
            query[tid] = half(
                base[tid]
                + float(positions[ulong(intra) * shape.index_width + tid])
                + prefix[tid] * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float logit_max = -INFINITY;
        float logit_sum = 0.0f;
        float expected_logit = -INFINITY;
        float sample_max = -INFINITY;
        uint sample_index = UINT_MAX;
        for (uint token = tid; token < shape.vocabulary; token += 256) {
            float logit = 0.0f;
            for (uint dimension = 0; dimension < 64; ++dimension)
                logit = fma(
                    float(query[dimension]),
                    float(readout[ulong(dimension) * shape.vocabulary + token]),
                    logit);
            if (shape.score != 0 && token == expected[position])
                expected_logit = logit;
            if (logit > logit_max) {
                logit_sum = logit_sum * exp(logit_max - logit) + 1.0f;
                logit_max = logit;
            } else {
                logit_sum += exp(logit - logit_max);
            }
            float sampled = logit;
            if (shape.temperature > 0.0f) {
                const ulong bits = fbt_hnet_mix(
                    seeds[0] ^ (ulong(position) << 32) ^ ulong(token));
                const float uniform =
                    (float(uint(bits >> 41)) + 0.5f) / 8388608.0f;
                sampled -= shape.temperature * log(-log(uniform));
            }
            if (sampled > sample_max || (sampled == sample_max && token < sample_index)) {
                sample_max = sampled;
                sample_index = token;
            }
        }
        for (uint offset = 16; offset > 0; offset >>= 1) {
            const float other_max = simd_shuffle_down(logit_max, offset);
            const float other_sum = simd_shuffle_down(logit_sum, offset);
            if (lane < offset) {
                const float combined = max(logit_max, other_max);
                logit_sum = logit_sum * exp(logit_max - combined)
                    + other_sum * exp(other_max - combined);
                logit_max = combined;
            }
        }
        const float group_expected = simd_max(expected_logit);
        const float group_sample = simd_max(sample_max);
        const uint group_index = simd_min(
            sample_max == group_sample ? sample_index : UINT_MAX);
        if (lane == 0) {
            logit_maxima[simdgroup] = logit_max;
            logit_sums[simdgroup] = logit_sum;
            expected_logits[simdgroup] = group_expected;
            sample_maxima[simdgroup] = group_sample;
            sample_indices[simdgroup] = group_index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total_max = logit_maxima[0];
            float total_sum = logit_sums[0];
            float target = expected_logits[0];
            float best = sample_maxima[0];
            uint chosen = sample_indices[0];
            for (uint group = 1; group < 8; ++group) {
                const float combined = max(total_max, logit_maxima[group]);
                total_sum = total_sum * exp(total_max - combined)
                    + logit_sums[group] * exp(logit_maxima[group] - combined);
                total_max = combined;
                target = max(target, expected_logits[group]);
                if (sample_maxima[group] > best
                    || (sample_maxima[group] == best && sample_indices[group] < chosen)) {
                    best = sample_maxima[group];
                    chosen = sample_indices[group];
                }
            }
            output[position] = chosen;
            if (shape.score != 0)
                losses[position] = total_max + log(total_sum) - target;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// PairShape and ObjectiveStats come from blckbaux_semantics::hnet.

// Reduce the full free-running trajectory without returning per-token losses
// or scanning generated tokens on the CPU.
kernel void fbt_hnet_objective(
    device const float* losses [[buffer(0)]],
    device const uint* generated [[buffer(1)]],
    device const uint* expected [[buffer(2)]],
    device ObjectiveStats* output [[buffer(3)]],
    device const float* baseline_losses [[buffer(4)]],
    constant PairShape& shape [[buffer(5)]],
    device float* paired_patches [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sums[8];
    threadgroup float batch_centered_sums[8];
    threadgroup float squared_batch_centered_sums[8];
    threadgroup float difference_sums[8];
    threadgroup float difference_batch_sums[8];
    threadgroup float squared_difference_batch_sums[8];
    threadgroup float maxima[8];
    threadgroup float mismatch_sums[8];
    threadgroup uint matches[8];
    threadgroup uint mismatches[8];
    threadgroup uint first_mismatches[8];
    float sum = 0.0f;
    float difference_sum = 0.0f;
    float maximum = -INFINITY;
    float mismatch_sum = 0.0f;
    uint matched = 0;
    uint mismatched = 0;
    uint first = UINT_MAX;
    for (uint position = tid; position < shape.tokens; position += 256) {
        const float loss = losses[position];
        const bool equal = generated[position] == expected[position];
        sum += loss;
        const float difference = baseline_losses[position] - loss;
        difference_sum += shape.paired != 0 ? difference : 0.0f;
        maximum = max(maximum, loss);
        matched += uint(equal);
        mismatched += uint(!equal);
        mismatch_sum += equal ? 0.0f : loss;
        first = !equal ? min(first, position) : first;
    }
    const uint batch_tokens = max(shape.batch_tokens, 1u);
    const uint batches = (shape.tokens + batch_tokens - 1u) / batch_tokens;
    float batch_centered_sum = 0.0f;
    float squared_batch_centered_sum = 0.0f;
    float difference_batch_sum = 0.0f;
    float squared_difference_batch_sum = 0.0f;
    for (uint batch = tid; batch < batches; batch += 256) {
        const uint start = batch * batch_tokens;
        const uint end = min(start + batch_tokens, shape.tokens);
        float batch_loss = 0.0f;
        float batch_difference = 0.0f;
        for (uint position = start; position < end; position++) {
            batch_loss += losses[position];
            batch_difference += baseline_losses[position] - losses[position];
        }
        const float inverse = 1.0f / float(end - start);
        const float centered = batch_loss * inverse - 12.0f;
        const float difference = batch_difference * inverse;
        if (shape.paired != 0)
            paired_patches[batch] = difference;
        batch_centered_sum += centered;
        squared_batch_centered_sum += centered * centered;
        difference_batch_sum += shape.paired != 0 ? difference : 0.0f;
        squared_difference_batch_sum +=
            shape.paired != 0 ? difference * difference : 0.0f;
    }
    sum = simd_sum(sum);
    batch_centered_sum = simd_sum(batch_centered_sum);
    squared_batch_centered_sum = simd_sum(squared_batch_centered_sum);
    difference_sum = simd_sum(difference_sum);
    difference_batch_sum = simd_sum(difference_batch_sum);
    squared_difference_batch_sum = simd_sum(squared_difference_batch_sum);
    maximum = simd_max(maximum);
    mismatch_sum = simd_sum(mismatch_sum);
    matched = simd_sum(matched);
    mismatched = simd_sum(mismatched);
    first = simd_min(first);
    if (lane == 0) {
        sums[simdgroup] = sum;
        batch_centered_sums[simdgroup] = batch_centered_sum;
        squared_batch_centered_sums[simdgroup] = squared_batch_centered_sum;
        difference_sums[simdgroup] = difference_sum;
        difference_batch_sums[simdgroup] = difference_batch_sum;
        squared_difference_batch_sums[simdgroup] = squared_difference_batch_sum;
        maxima[simdgroup] = maximum;
        mismatch_sums[simdgroup] = mismatch_sum;
        matches[simdgroup] = matched;
        mismatches[simdgroup] = mismatched;
        first_mismatches[simdgroup] = first;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        sum = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        batch_centered_sum = simd_sum(
            lane < 8 ? batch_centered_sums[lane] : 0.0f);
        squared_batch_centered_sum = simd_sum(
            lane < 8 ? squared_batch_centered_sums[lane] : 0.0f);
        difference_sum = simd_sum(lane < 8 ? difference_sums[lane] : 0.0f);
        difference_batch_sum = simd_sum(
            lane < 8 ? difference_batch_sums[lane] : 0.0f);
        squared_difference_batch_sum = simd_sum(
            lane < 8 ? squared_difference_batch_sums[lane] : 0.0f);
        maximum = simd_max(lane < 8 ? maxima[lane] : -INFINITY);
        mismatch_sum = simd_sum(lane < 8 ? mismatch_sums[lane] : 0.0f);
        matched = simd_sum(lane < 8 ? matches[lane] : 0u);
        mismatched = simd_sum(lane < 8 ? mismatches[lane] : 0u);
        first = simd_min(lane < 8 ? first_mismatches[lane] : UINT_MAX);
        if (lane == 0) {
            output[0].sum = sum;
            output[0].batch_centered_sum = batch_centered_sum;
            output[0].squared_batch_centered_sum = squared_batch_centered_sum;
            output[0].difference_sum = difference_sum;
            output[0].difference_batch_sum = difference_batch_sum;
            output[0].squared_difference_batch_sum = squared_difference_batch_sum;
            output[0].maximum = maximum;
            output[0].mismatch_sum = mismatch_sum;
            output[0].batches = batches;
            output[0].matches = matched;
            output[0].mismatches = mismatched;
            output[0].first_mismatch = first;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid < 8) {
        const uint begin = tid * batch_tokens / 8;
        const uint finish = (tid + 1) * batch_tokens / 8;
        float octile_sum = 0.0f;
        uint octile_count = 0;
        for (uint base = 0; base < shape.tokens; base += batch_tokens) {
            const uint end = min(base + finish, shape.tokens);
            for (uint position = base + begin; position < end; ++position) {
                octile_sum += losses[position];
                octile_count++;
            }
        }
        output[0].patch_nll_octiles[tid] = octile_count == 0
            ? NAN
            : octile_sum / float(octile_count);
    }
}
