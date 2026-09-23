// Exact, non-periodic 3D KD-tree query, K=16, FP64. No fast-math.
// Tree layout and stack-free traversal follow CuPy 14.2's _kdtree_utils.py
// (Wald's left-balanced KD-tree). See cupy_knn_LICENSE.txt for the MIT license.
// Specialization: squared Euclidean distance, fixed-size per-thread top-K,
// int32 tree positions, and a single final write of the int64 original IDs.

extern "C" __global__ void query_knn16(
    const double* __restrict__ queries,
    const double* __restrict__ tree,
    const long long* __restrict__ original_ids,
    const int count, const int query_count,
    long long* __restrict__ result
) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query_count) return;
    const double qx = queries[3LL * q];
    const double qy = queries[3LL * q + 1];
    const double qz = queries[3LL * q + 2];

    // Constant loop indices allow scalarization; there is no global distance
    // buffer or global candidate update inside the traversal loop.
    double best_distance[16];
    int best_position[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        best_distance[i] = __longlong_as_double(0x7ff0000000000000LL);
        best_position[i] = 0;
    }

    int node = 0;
    int previous = -1;
    while (node >= 0) {
        const int parent = (node + 1) / 2 - 1;
        // A left-balanced tree can have a missing right child.
        if (node >= count) {
            previous = node;
            node = parent;
            continue;
        }
        const int left = 2 * node + 1;
        const int right = left + 1;
        const double dx = qx - tree[3LL * node];
        const double dy = qy - tree[3LL * node + 1];
        const double dz = qz - tree[3LL * node + 2];

        // Process a node once, on descent from its parent.
        if (previous < left) {
            double distance = (dx * dx + dy * dy) + dz * dz;
            if (distance < best_distance[15]) {
                int position = node;
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (distance < best_distance[i]) {
                        const double displaced_distance = best_distance[i];
                        const int displaced_position = best_position[i];
                        best_distance[i] = distance;
                        best_position[i] = position;
                        distance = displaced_distance;
                        position = displaced_position;
                    }
                }
            }
        }

        const int depth = 31 - __clz(static_cast<unsigned int>(node + 1));
        const int axis = depth % 3;
        const double split_delta = axis == 0 ? dx : (axis == 1 ? dy : dz);
        const int near_child = split_delta > 0.0 ? right : left;
        const int far_child = split_delta > 0.0 ? left : right;

        int next;
        if (previous == near_child) {
            // The split-plane distance is a lower bound for the far subtree.
            // Only prune if it cannot IMPROVE the current 16 candidates.
            // Equal-distance IDs are intentionally unspecified (as in SciPy).
            next = far_child < count && split_delta * split_delta < best_distance[15]
                ? far_child : parent;
        } else if (previous == far_child) {
            next = parent;
        } else {
            next = left < count ? near_child : parent;
        }
        previous = node;
        node = next;
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[16LL * q + i] = original_ids[best_position[i]];
    }
}
