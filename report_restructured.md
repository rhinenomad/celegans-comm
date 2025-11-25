# 1. Background Survey of the Literature

The study of neural connectomes has been revolutionized by advances in electron microscopy and computational graph theory. The C. elegans connectome, being the first complete neural wiring diagram of an animal, has served as a model system for understanding how network structure relates to function (White et al., 1986; Varshney et al., 2011). The WormWiring project (Cook et al., 2019) has provided comprehensive connectivity data, enabling algorithmic analysis of information flow in neural networks.

Graph algorithms have been widely applied to analyze neural connectivity. Breadth-first search (BFS) has been used to study information propagation and network reachability (Sporns et al., 2004). Shortest path algorithms, particularly Dijkstra's algorithm, have been employed to understand routing efficiency in neural networks (Bassett & Bullmore, 2006). The trade-off between topological and geometric routing strategies has been explored in various network contexts, suggesting that biological networks may optimize for both connectivity patterns and spatial constraints (Kaiser & Hilgetag, 2006).

Decentralized navigation algorithms, inspired by biological routing mechanisms, have been studied to understand how local information can guide global navigation (Kleinberg, 2000). Greedy geometric routing, where nodes make local decisions based on proximity to the target, has been shown to work well in certain network topologies but can fail in others (Duchon et al., 2004).

Diffusion-based communication models, particularly the heat kernel, have been used to quantify information flow in networks (Gămănuț et al., 2018). The heat kernel H = exp(-τL) captures how information diffuses through a network over time, providing a continuous analog to discrete random walks. The relationship between diffusion strength and geodesic distance has been studied to understand how network structure supports efficient communication (Estrada & Hatano, 2008).

Network flow theory, particularly max-flow min-cut analysis, has been applied to identify critical bottlenecks in information transmission (Ford & Fulkerson, 1956). The relationship between flow-based bottlenecks and centrality measures such as betweenness centrality has been explored to understand which nodes are most critical for network function (Freeman, 1977; Girvan & Newman, 2002).

This work builds upon these foundations to provide a comprehensive algorithmic analysis of communication patterns in the C. elegans connectome, examining multiple aspects of information flow through a unified experimental framework.

---

# 2. Implementation Work

This section describes the software implementation, data processing pipeline, and algorithmic components developed for this project. The complete codebase is available in the project repository.

## 2.1 Data Processing Pipeline

The implementation begins with processing the raw WormWiring SI5 dataset (Cook et al., 2019), which is provided as an Excel file containing a connectivity matrix. We developed a Python script (`parse_wormwiring_final.py`) that extracts neural connectivity data while excluding non-neuronal cells such as body wall muscles, pharynx cells, and sex-specific cells.

The parser identifies three neuron types based on the Excel file's grouping structure: sensory neurons (S), interneurons (I), and motor neurons (M). It processes both row and column headers to extract neuron names and their classifications, ensuring that only neurons explicitly categorized in the Excel file are included. The final processed network contains 272 neurons (83 sensory, 81 interneurons, 108 motor neurons) and 3,995 directed edges representing chemical synapses, with edge weights corresponding to the number of synaptic connections between neuron pairs.

## 2.2 Graph Variant Construction

We implemented a graph construction pipeline (`src/pipeline_prepare_graphs.py`) that generates five graph variants to accommodate different algorithmic requirements:

- **V1 (Directed, Unweighted)**: A directed graph with unit edge weights, suitable for algorithms focusing on topological connectivity.

- **V2 (Directed, Weighted)**: A directed graph preserving the original edge weights (synaptic connection counts), serving as the primary variant for weighted pathfinding and flow analysis.

- **V3 (Undirected, Weighted)**: An undirected version of V2, created by symmetrizing the directed edges, useful for algorithms treating the network as symmetric.

- **V4 (Directed, Weighted, with Geometry)**: Based on V2 with algorithmically generated coordinates, combining weighted connectivity with spatial information. This variant is used in experiments requiring geometric coordinates (E2 and E3).

- **V5 (Directed, Unweighted, with Geometry)**: Similar to V1 but includes algorithmically generated 2D coordinates using NetworkX's spring layout algorithm. This variant is defined but not used in the current experiments.

The V4 variant is generated using a dedicated script (`src/generate_v5.py`) that applies the spring layout algorithm (with seed=42 for reproducibility) to compute 2D node positions based on the graph's connectivity structure. The script takes V2 as input and adds algorithmically generated coordinates to create V4. V4 is the primary variant used for experiments requiring geometric coordinates (E2 and E3), as it combines the weighted connectivity structure of V2 with spatial information. All graph variants are stored as NetworkX pickle files, preserving node attributes (neuron types) and edge attributes (weights, connection types).

## 2.3 Algorithm Implementations

### Breadth-First Search (E1)
We implemented a directed BFS algorithm (`src/e1_bfs/run_bfs.py`) that computes layer-wise coverage from seed nodes. The implementation uses NetworkX's `single_source_shortest_path_length` function, respecting the directed nature of the graph to capture actual information flow direction. The algorithm tracks cumulative coverage at each BFS layer, aggregating results across multiple seeds to compute average coverage curves.

### Alpha-Sweep Shortest Path (E2)
We implemented a parameterized shortest path algorithm (`src/e2_dijkstra/run_sp.py`) using Dijkstra's algorithm with a hybrid cost function: `cost = α × geometric_distance + (1 - α) × topological_cost`. The topological cost is computed as the inverse of edge weight, while the geometric cost uses Euclidean distance between node positions from the V4 graph variant. The implementation supports alpha values from 0.0 (pure topological) to 1.0 (pure geometric), enabling systematic exploration of the topology-geometry trade-off.

### Greedy Geometric Navigation (E3)
We implemented a decentralized navigation algorithm (`src/e3_nav/run_nav.py`) that simulates local routing based on geometric proximity. The algorithm starts from a source node and, at each step, selects the neighbor that minimizes Euclidean distance to the target. The implementation uses the V4 graph variant, which provides both weighted connectivity and geometric coordinates. The implementation includes support for multi-cue navigation, combining geometric distance with local node properties (degree, edge weights, homophily). We also implemented metrics for geometric congruence and greedy routing efficiency (`src/e3_nav/geometric_congruence.py`) to evaluate navigation performance.

### Heat Kernel Diffusion (E4)
We implemented the heat kernel diffusion model (`src/e4_diffusion/run_diff.py`) to compute the diffusion matrix H = exp(-τL), where L is the graph Laplacian and τ = 0.5 is the time scale parameter. The implementation uses SciPy's sparse matrix exponential function for efficient computation. We also implemented geodesic distance matrix computation using NetworkX's shortest path algorithms, and correlation analysis using Spearman rank correlation to compare diffusion values with inverse geodesic distances.

### Max-Flow Min-Cut Analysis (E5)
We implemented network flow analysis (`src/e5_flow/run_flow.py`) using NetworkX's maximum flow algorithm (Ford-Fulkerson method). The implementation computes max-flow and identifies min-cut nodes for each sensory-motor pair. We also implemented betweenness centrality computation and configuration model null graph generation for statistical comparison. The implementation includes FDR correction for multiple hypothesis testing using the Benjamini-Hochberg method.

## 2.4 Code Repository

The complete implementation is available in the project repository: **https://github.com/rhinenomad/celegans-comm.git**

The codebase is organized into modular components:
- `src/pipeline_prepare_graphs.py`: Graph construction pipeline
- `src/e1_bfs/`: BFS coverage analysis
- `src/e2_dijkstra/`: Alpha-sweep shortest path analysis
- `src/e3_nav/`: Greedy geometric navigation
- `src/e4_diffusion/`: Heat kernel diffusion analysis
- `src/e5_flow/`: Max-flow min-cut bottleneck analysis
- `configs/params.yaml`: Configuration file for all experiments
- `makefile`: Build automation for running experiments

All scripts follow a consistent interface using command-line arguments and YAML configuration files, enabling reproducible experiments across different graph variants and parameter settings.

---

# 3. Experiments and Results

This section describes the specific experiments performed, including experimental design, parameter settings, procedures, results, and discussion for each of the five experiments.

## Experiment 1: Breadth-First Search Coverage Analysis

**Objective**: To analyze how information propagates from sensory neurons through the network.

**Experimental Design**: We ran directed BFS starting from each of the 83 identified sensory neurons as seed nodes. For each seed, we computed the layer-wise coverage, tracking how many new nodes are reached at each BFS layer (hop distance from the seed).

**Parameters**: 
- Graph variant: V2 (directed, weighted)
- Seeds: 83 sensory neurons
- Metric: Cumulative coverage at each BFS layer

**Procedure**: For each sensory neuron seed, we computed the shortest path lengths to all reachable nodes using NetworkX's directed shortest path algorithm. We then grouped nodes by their distance (layer) from the seed and computed cumulative coverage. Results were aggregated across all seeds to compute average coverage curves.

**Results**: The BFS coverage analysis demonstrates that the network exhibits efficient information propagation from sensory neurons. Starting from 83 sensory neurons as seeds, the analysis shows that most nodes in the network can be reached within 3 to 4 hops. On average, sensory neurons reach approximately 20 nodes in the first layer, 100 nodes in the second layer, and over 250 nodes by the third layer, with near-complete coverage (270 out of 272 nodes) achieved by layer 4 or 5.

**Discussion**: This rapid coverage indicates that the network structure is well-suited for fast information propagation, with sensory inputs able to reach the vast majority of the network within a small number of synaptic hops. The layer-wise progression shows a characteristic pattern where coverage increases rapidly in the first few layers and then plateaus as the network becomes saturated. This suggests that the network's connectivity structure provides multiple redundant pathways for information to spread from sensory inputs, which is consistent with the network's role in processing sensory information and coordinating motor responses.

## Experiment 2: Alpha-Sweep Shortest Path Analysis

**Objective**: To explore the trade-off between topological and geometric routing strategies.

**Experimental Design**: We performed an alpha-sweep across five values (0.0, 0.25, 0.5, 0.75, 1.0), where α controls the balance between topological cost (inverse edge weight) and geometric cost (Euclidean distance). For each alpha value, we computed shortest paths from all sensory neurons to all motor neurons.

**Parameters**:
- Graph variant: V4 (V2 with algorithmically generated coordinates)
- Alpha values: [0.0, 0.25, 0.5, 0.75, 1.0]
- Sources: 83 sensory neurons
- Targets: 108 motor neurons
- Cost function: `cost = α × geometric_distance + (1 - α) × inverse_weight`

**Procedure**: For each alpha value, we constructed a cost graph where edge costs combine topological and geometric components according to the alpha parameter. We then computed shortest paths from each sensory neuron to each motor neuron using Dijkstra's algorithm. We measured both the reachability rate (fraction of connected pairs) and the mean path cost.

**Results**: The alpha-sweep experiment reveals how the balance between topological and geometric routing affects pathfinding efficiency. Across all alpha values tested (0.0 to 1.0), the reachability rate remains constant at 45%, indicating that approximately half of the sensory-motor pairs are connected in the directed graph. However, the mean path cost increases monotonically with alpha, from 0.29 at α = 0.0 (pure topological) to 0.90 at α = 1.0 (pure geometric), representing a 3.09-fold increase.

**Discussion**: This cost increase suggests that pure geometric routing, while intuitive, is significantly less efficient than topological routing in this network. The topological structure, encoded through edge weights representing synaptic connection counts, provides more efficient paths than simply following geometric proximity. The intermediate alpha values show a gradual transition, with costs increasing smoothly as more weight is placed on geometric distance. This indicates that the network's topology and geometry (as represented by the algorithmically generated coordinates) are not perfectly aligned, and that the weighted connectivity structure contains information that cannot be captured by geometric distance alone. This finding suggests that the C. elegans connectome has evolved to optimize for topological efficiency rather than spatial proximity, which may reflect the constraints of neural development and the functional requirements of information processing.

## Experiment 3: Greedy Geometric Navigation

**Objective**: To evaluate the performance of decentralized navigation using only local geometric information.

**Experimental Design**: We tested greedy geometric navigation on 8,964 source-target pairs sampled from the network. For each pair, we attempted to navigate from source to target using a greedy algorithm that selects the neighbor minimizing Euclidean distance to the target at each step.

**Parameters**:
- Graph variant: V4 (V2 with algorithmically generated coordinates)
- Number of pairs: 8,964
- Step limit: Based on network diameter
- Navigation strategy: Pure geometric (minimize distance to target)

**Procedure**: For each source-target pair, we ran the greedy navigation algorithm, recording whether the navigation succeeded (reached the target) and the path taken. We computed path stretch (ratio of actual path length to shortest path length) for successful navigations. We also computed geometric congruence (ratio of straight-line distance to geometric path length) and greedy routing efficiency (combining success rate and normalized stretch).

**Results**: The greedy geometric navigation algorithm achieves a success rate of 81.5% across 8,964 source-target pairs, demonstrating that simple local routing based on geometric proximity can successfully navigate the network in most cases. However, the median path stretch is 4.5, meaning that successful paths are on average more than four times longer than the optimal shortest paths. The mean path stretch is even higher at 5.75, indicating substantial inefficiency in the greedy approach. The geometric congruence analysis shows a mean value of 0.81, indicating that the straight-line distance is on average 81% of the geometric path length (i.e., the geometric path length is approximately 23% longer than the straight-line distance). The greedy routing efficiency (GRE) is 0.14, reflecting the trade-off between the algorithm's reasonable success rate and its suboptimal path lengths.

**Discussion**: These results suggest that while decentralized geometric navigation is possible in the C. elegans connectome, the network structure does not strongly favor this simple routing strategy. The high path stretch indicates that local greedy decisions often lead to suboptimal paths, which may explain why biological neural systems likely employ more sophisticated routing mechanisms. The 81.5% success rate, while reasonable, leaves room for improvement, and the substantial path inefficiency suggests that purely geometric routing may not be the primary mechanism for information routing in this network. The relatively high geometric congruence (0.81) suggests that the algorithmically generated geometric embedding is reasonably consistent with the network's connectivity structure, but this consistency alone is not sufficient to enable efficient greedy navigation.

## Experiment 4: Heat Kernel Diffusion Analysis

**Objective**: To investigate the relationship between diffusion-based communication and geodesic distance.

**Experimental Design**: We computed the heat kernel diffusion matrix H = exp(-τL) with τ = 0.5 and compared diffusion values against geodesic distances (shortest path lengths) for all node pairs.

**Parameters**:
- Graph variant: V2 (directed, weighted)
- Diffusion model: Heat kernel
- Time scale parameter τ: 0.5
- Correlation method: Spearman rank correlation

**Procedure**: We computed the graph Laplacian matrix L and applied the matrix exponential to obtain the diffusion matrix H. We also computed the geodesic distance matrix using shortest path algorithms. We then computed the Spearman rank correlation coefficient between diffusion values and inverse geodesic distances (using inverse distance because diffusion should increase as nodes get closer).

**Results**: The heat kernel diffusion analysis reveals a moderate positive correlation (Spearman ρ = 0.333, p < 0.0001) between diffusion strength and inverse geodesic distance across 36,816 valid node pairs. This correlation indicates that nodes that are closer together in terms of hop distance tend to have stronger diffusion-based communication, as expected. The scatter plot visualization shows distinct patterns for different distance ranges. Nodes at distance 1 (directly connected) show the highest diffusion values, with a wide spread. As distance increases, diffusion values generally decrease, but with considerable variation. The distance groups 4-5 and 6-10 contain the majority of node pairs, reflecting the network's small-world properties where most nodes are within a few hops of each other.

**Discussion**: The moderate correlation suggests that while geodesic distance is a good predictor of diffusion strength, the heat kernel diffusion model captures additional information beyond simple shortest-path distance. However, the correlation is not perfect, suggesting that diffusion captures additional network structure that goes beyond simple path length. This may include the effects of multiple parallel paths, network motifs, and local connectivity patterns that influence how information diffuses through the network. The imperfect correlation indicates that diffusion-based communication provides a richer model of information flow than simple shortest-path routing, potentially capturing aspects of network redundancy and alternative pathways.

## Experiment 5: Max-Flow Min-Cut Bottleneck Analysis

**Objective**: To identify critical bottlenecks in information flow between sensory and motor neurons.

**Experimental Design**: We computed max-flow and min-cut for 100 sensory-motor pairs, then compared min-cut nodes against nodes with high betweenness centrality. We generated 20 configuration model null graphs to assess statistical significance.

**Parameters**:
- Graph variant: V2 (directed, weighted)
- Number of pairs tested: 100
- Capacity: Edge weights
- Null model: Configuration model (20 graphs)
- Statistical test: One-tailed t-test with FDR correction (α = 0.05)

**Procedure**: For each sensory-motor pair, we computed the maximum flow using NetworkX's max-flow algorithm, identifying the min-cut nodes (source-side of the residual graph). We computed betweenness centrality for all nodes and selected the top-k nodes (where k is approximately twice the average min-cut size). We computed the overlap between min-cut nodes and top-k betweenness nodes. For statistical testing, we generated 20 configuration model null graphs preserving the degree distribution, computed the same overlap metric on each, and performed statistical testing with FDR correction.

**Results**: The max-flow min-cut analysis on 100 sensory-motor pairs reveals that the network exhibits high redundancy in information flow. The mean maximum flow value is 40.09, and the mean min-cut size is 108.51 nodes, indicating that a large number of nodes must be removed to completely disconnect sensory-motor communication. This large cut size relative to the total network size (272 nodes) suggests that information flow between sensory and motor neurons is highly redundant, with many alternative pathways available. However, the overlap analysis between min-cut nodes and high betweenness centrality nodes shows a mean overlap ratio of only 4%, which is not significantly different from the null model expectation after FDR correction. None of the 100 tested pairs showed statistically significant overlap.

**Discussion**: The high redundancy in information flow suggests that the network is robust to node removal, with many alternative pathways available for sensory-motor communication. However, the lack of significant overlap between min-cut nodes and high betweenness nodes indicates that these two measures identify different sets of critical nodes. The min-cut identifies nodes that are specifically critical for particular source-target pairs, representing bottlenecks in point-to-point communication. In contrast, betweenness centrality identifies nodes that are globally important across all paths, representing hubs in the overall network structure. The lack of significant overlap indicates that these two measures capture complementary aspects of network importance rather than identifying the same bottleneck nodes. This finding suggests that the C. elegans connectome may have evolved to distribute critical nodes across different functional roles, with some nodes serving as local bottlenecks for specific pathways and others serving as global hubs for overall network connectivity.

---

# 4. Overall Discussion

## Summary of Findings

Taken together, these experimental results paint a comprehensive picture of the C. elegans connectome as a network optimized for efficient information propagation through topological structure rather than spatial proximity. The network exhibits high redundancy, enabling robust communication through multiple alternative pathways. While simple routing strategies (geometric navigation) can achieve reasonable success rates, they are far from optimal, suggesting that biological neural systems likely employ more sophisticated mechanisms for information routing.

The experiments reveal several key insights: (1) The network achieves rapid coverage from sensory inputs, with most nodes reachable within 3-4 hops, indicating efficient information propagation; (2) Topological routing significantly outperforms geometric routing, suggesting the network is optimized for connectivity patterns rather than spatial proximity; (3) Decentralized geometric navigation is feasible but inefficient, with high path stretch indicating the need for more sophisticated routing mechanisms; (4) Diffusion-based communication shows moderate correlation with geodesic distance, capturing additional network structure beyond simple path length; (5) The network exhibits high redundancy in information flow, with critical nodes distributed across different functional roles rather than concentrated in a single set of bottleneck nodes.

These findings suggest that the C. elegans connectome has evolved to support efficient and robust information processing through a combination of topological optimization, network redundancy, and distributed critical nodes. The moderate correlation between diffusion and geodesic distance indicates that network structure supports efficient communication through multiple mechanisms, including direct connections, short paths, and diffusion-based spreading.

## Limitations and Future Work

Several limitations of this study warrant discussion and suggest directions for future research. First, our geometric routing experiments (E2 and E3) rely on algorithmically generated coordinates using spring layout, rather than actual anatomical positions. While this approach enables geometric analysis when anatomical data is unavailable, it may not accurately reflect true spatial relationships in the neural tissue. The strong preference for topological routing over geometric routing (3.09-fold difference) could partially reflect this limitation, as the algorithmically generated coordinates may not align well with the network's connectivity structure. Future work should incorporate actual anatomical coordinates when available to validate these findings.

Second, our analysis focuses on a simplified network of 272 neurons, excluding non-neuronal cells such as muscles and pharynx cells. While this simplification enables clearer analysis of neural communication, it may miss important connectivity patterns that involve these excluded components. Additionally, we consider only chemical synapses, excluding gap junctions which provide direct electrical coupling. Future work could explore how including gap junctions affects routing efficiency and network redundancy.

Third, our experiments reveal some discrepancies with pre-registered success criteria. For E3, while the success rate (81.5%) exceeds the threshold (≥70%), the median path stretch (4.5) far exceeds the target (≤1.25). This suggests that the success criteria may need refinement, as achieving high success rates does not guarantee efficient routing. For E5, we found no statistically significant overlap between min-cut nodes and high betweenness nodes, contrary to the pre-registered expectation. This finding challenges the assumption that critical nodes are concentrated in a single set of hub nodes, instead suggesting a distributed model of network criticality.

Fourth, the moderate correlation (ρ = 0.333) between diffusion and geodesic distance in E4, while statistically significant, is not as strong as might be expected for a small-world network. This suggests that diffusion captures additional network structure beyond simple path length, potentially including the effects of parallel paths, network motifs, and local connectivity patterns. However, it also raises questions about whether the heat kernel model with τ = 0.5 is the optimal choice, or whether alternative diffusion models or time scales might better capture the network's communication properties.

Fifth, our analysis assumes a static network structure, ignoring potential temporal dynamics, plasticity, or context-dependent connectivity changes. Real neural networks exhibit activity-dependent plasticity and may have different effective connectivity patterns under different behavioral states. Future work could explore how dynamic connectivity affects routing efficiency and network robustness.

Finally, the lack of significant overlap between min-cut nodes and betweenness centrality nodes in E5 represents a novel finding that challenges conventional hub-based models of network importance. This suggests that the C. elegans connectome may have evolved a distributed criticality model, where different nodes serve different roles (local bottlenecks vs. global hubs) rather than concentrating importance in a single set of nodes. This finding warrants further investigation to understand the evolutionary and functional implications of this distributed structure.

These limitations and findings open several avenues for future research: (1) incorporating anatomical coordinates to validate geometric routing findings; (2) expanding the network to include gap junctions and non-neuronal components; (3) exploring alternative diffusion models and time scales; (4) investigating dynamic network properties and their effects on communication; (5) developing more sophisticated null models that better capture network structure; and (6) refining success criteria to better reflect the trade-offs between different routing strategies.

---

# Appendix A: Note on Dataset Revision

It should be noted that the experimental results presented in this report differ slightly from those presented in an earlier poster presentation. Following the presentation, we conducted a careful review of the dataset construction process and identified that a small number of muscle cells had been incorrectly classified as motor neurons during the initial data processing. 

This misclassification has been corrected in the current analysis, with neurons now strictly classified based on their explicit categorization in the WormWiring Excel file (SENSORY NEURONS, INTERNEURONS, and MOTOR NEURONS groupings), excluding non-neuronal cells such as body wall muscles, pharynx cells, and sex-specific cells. The final dataset used in this report contains 272 neurons (83 sensory, 81 interneurons, 108 motor neurons) and has been carefully validated to ensure accurate neuron classification.

In the event of any discrepancies between this report and earlier presentations, the results and conclusions presented in this report should be considered authoritative, as they are based on the corrected and validated dataset.

---

# Appendix B: Complete Experimental Parameters

This appendix provides a comprehensive listing of all parameters and configuration settings used in the experiments. All parameters are defined in the project configuration file (`configs/params.yaml`) and can be reproduced by running the experiments with the provided configuration.

## B.1 Project-Wide Settings

- **Random seed**: 42 (for reproducibility)
- **Dataset**: WormWiring SI5 (Cook et al., 2019)
- **Total nodes**: 272 neurons (83 sensory, 81 interneurons, 108 motor neurons)
- **Total edges**: 3,995 directed edges (chemical synapses)

## B.2 Graph Variants

The following graph variants were constructed for different experimental requirements:

- **V1**: Directed, unweighted (unit edge weights)
- **V2**: Directed, weighted (edge weights = synaptic connection counts)
- **V3**: Undirected, weighted (symmetrized version of V2)
- **V4**: Directed, weighted, with algorithmically generated 2D coordinates (used in E2 and E3)
- **V5**: Directed, unweighted, with algorithmically generated 2D coordinates (defined but not used)

The V4 variant was generated using NetworkX's spring layout algorithm with seed=42 for reproducibility.

## B.3 Experiment 1: BFS Coverage Analysis

- **Graph variant**: V2 (directed, weighted)
- **Seed nodes**: 83 sensory neurons (from `data/interim/sensory_nodes.txt`)
- **Algorithm**: Directed breadth-first search using NetworkX's `single_source_shortest_path_length`
- **Metric**: Cumulative coverage at each BFS layer (hop distance)
- **Output**: Layer-wise coverage statistics aggregated across all seeds

## B.4 Experiment 2: Alpha-Sweep Shortest Path Analysis

- **Graph variant**: V4 (directed, weighted with geometric coordinates)
- **Alpha values**: [0.0, 0.25, 0.5, 0.75, 1.0]
- **Source nodes**: 83 sensory neurons
- **Target nodes**: 108 motor neurons
- **Cost function**: `cost = α × geometric_distance + (1 - α) × topological_cost`
  - Topological cost: inverse of edge weight (`inverse_weight`)
  - Geometric cost: Euclidean distance between node positions
  - Distance key: `euclidean`
- **Algorithm**: Dijkstra's shortest path algorithm
- **Metrics**: Reachability rate, mean path cost

## B.5 Experiment 3: Greedy Geometric Navigation

- **Graph variant**: V4 (directed, weighted with geometric coordinates)
- **Number of pairs tested**: 8,964 source-target pairs
- **Navigation strategy**: Pure geometric (minimize Euclidean distance to target at each step)
- **Step limit**: Based on network diameter (max_steps_factor = 1.0)
- **Backtracking**: Allowed (`allow_backtrack: true`)
- **Multi-cue navigation weights** (defined but not used in pure geometric mode):
  - Degree weight: 0.2
  - Edge weight: 0.4
  - Homophily weight: 0.2
  - Distance to target weight: 0.2
- **Metrics**: Success rate, path stretch (ratio of actual path length to shortest path length), geometric congruence, greedy routing efficiency

## B.6 Experiment 4: Heat Kernel Diffusion Analysis

- **Graph variant**: V2 (directed, weighted)
- **Diffusion model**: Heat kernel (`heat_kernel`)
- **Time scale parameter τ**: 0.5
- **Diffusion matrix**: H = exp(-τL), where L is the graph Laplacian
- **Computation method**: SciPy sparse matrix exponential
- **Geodesic distance**: Shortest path length (hop count) between all node pairs
- **Correlation method**: Spearman rank correlation
- **Comparison**: Diffusion values vs. inverse geodesic distances
- **Valid pairs**: 36,816 node pairs (excluding self-loops and unreachable pairs)

## B.7 Experiment 5: Max-Flow Min-Cut Bottleneck Analysis

- **Graph variant**: V2 (directed, weighted)
- **Number of pairs tested**: 100 sensory-motor pairs
- **Capacity**: Edge weights (synaptic connection counts)
- **Algorithm**: NetworkX maximum flow (Ford-Fulkerson method)
- **Min-cut identification**: Source-side nodes in the residual graph after max-flow computation
- **Betweenness centrality**: Computed for all nodes, top-k nodes selected (where k ≈ 2 × average min-cut size)
- **Null model**: Configuration model
  - Type: `configuration`
  - Number of null graphs: 20
  - Preserve in/out degree: `true`
- **Statistical testing**:
  - Method: One-tailed t-test
  - Multiple comparison correction: FDR (Benjamini-Hochberg method)
  - Significance threshold: α = 0.05
- **Metrics**: Maximum flow value, min-cut size, overlap ratio between min-cut nodes and high betweenness nodes

## B.8 Statistical Analysis Parameters

- **Correlation method**: Spearman rank correlation (non-parametric)
- **Paired test**: Wilcoxon signed-rank test
- **Multiple comparison correction**: FDR (False Discovery Rate) with α = 0.05
- **Bootstrap iterations**: 1,000 (for confidence intervals, if applicable)

## B.9 Output Directories

- **Figures directory**: `results/figures/report_figures/`
- **Tables directory**: `results/tables/`

All experimental results are saved to these directories with standardized naming conventions for reproducibility.

