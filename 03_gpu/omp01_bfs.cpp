#include <vector>
#include <queue>
#include <omp.h>
#include <iostream>
#include <atomic>

void bfs(const std::vector<std::vector<int>>& adj, 
         int src,
         std::vector<int>& dist) {
    int INF = adj.size() + 1; // Use a value larger than the maximum possible distance
    dist.assign(adj.size(), INF);
    dist[src] = 0;

    std::vector<int> frontier, next_frontier;
    frontier.push_back(src);

    int level = 1;
    while (!frontier.empty()) {
        next_frontier.clear();

        // Parallel processing of the frontier
        #pragma omp parallel
        {
            std::vector<int> local_next;
            #pragma omp for nowait
            for (int i = 0; i < (int)frontier.size(); ++i) {
                int u = frontier[i];
                for (int v : adj[u]) {
                    if (dist[v] == INF) {
                        // try to update the distance atomically
						if (std::atomic_ref(dist[v]).compare_exchange_strong(INF, level)) {
							local_next.push_back(v);
						}
                    }
                }
            }
            // local fusion of next_frontier
            #pragma omp critical
            next_frontier.insert(next_frontier.end(),
                                 local_next.begin(),
                                 local_next.end());
        }

        frontier.swap(next_frontier);
        ++level;
    }
}

int main() {
    int N = 6; // Number of nodes
    std::vector<std::vector<int>> adj = {
        {1, 2},    // 0
        {0, 3, 4}, // 1
        {0, 4},    // 2
        {1, 5},    // 3
        {1, 2, 5}, // 4
        {3, 4}     // 5
    };

    std::vector<int> dist;
    bfs(adj, /*src=*/0, dist);

    for (int i = 0; i < N; ++i)
        std::cout << "Dist[" << i << "] = " << dist[i] << "\n";

    return 0;
}