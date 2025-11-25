# WormWiring Data Download Guide

## Manual Download Steps

Since the WormWiring website (https://wormwiring.org/pages/) data may require access through the website interface, please follow these steps:

### 1. Access the Website
Open your browser and visit: https://wormwiring.org/pages/

### 2. Find Data Download Links
On the website, look for the following types of data files:
- Chemical Synapse Data (Chemical Synapses)
- Gap Junction Data (Gap Junctions)
- Adjacency Matrix
- Node List (Neuron List/Nodes)
- Edge List (Edges)

### 3. Download and Rename
Save the downloaded files to the `data/raw/` directory and rename them as follows:

| Original Filename | Suggested Rename | Description |
|-------------------|------------------|-------------|
| Chemical Synapse Data | `wormwiring_chemical_synapses.csv` | Chemical synapse connections |
| Gap Junction Data | `wormwiring_gap_junctions.csv` | Gap junctions (undirected) |
| Complete Edge Data | `wormwiring_edges.csv` | All edge data |
| Node/Neuron List | `wormwiring_nodes.csv` | Node metadata (names, types, etc.) |
| Adjacency Matrix | `wormwiring_adjacency_matrix.csv` | Adjacency matrix format |

### 4. Expected File Format

**Edge data files should contain:**
- `source` or `from` column: Source node
- `target` or `to` column: Target node
- `weight` column (optional): Edge weight
- `type` column (optional): Edge type (chemical/gap)

**Node data files should contain:**
- `id` or `node_id`: Node ID
- `name` or `neuron_name`: Neuron name
- `type` column (optional): Node type (sensory/motor/interneuron)

## Automated Download Script (if URL is known)

If you find direct download links for the data files, you can use the following commands:

```bash
# Example: Download CSV files
cd data/raw
curl -L -o wormwiring_chemical_synapses.csv "https://wormwiring.org/data/chemical_synapses.csv"
curl -L -o wormwiring_gap_junctions.csv "https://wormwiring.org/data/gap_junctions.csv"
curl -L -o wormwiring_nodes.csv "https://wormwiring.org/data/nodes.csv"
```

## Check Downloaded Files

Run the following command to check the files:

```bash
cd data/raw
ls -lh wormwiring_*
```

## Using Excel Files from WormWiring SI5

The project primarily uses the WormWiring SI5 dataset, which is provided as an Excel file. The Excel file contains multiple worksheets with connectivity matrices.

### Excel File Structure

The Excel file (`wormwiring_SI5_connectome_adjacency_corrected_2020.xlsx`) should be placed in `data/raw/` directory. This file contains:

- **Chemical synapse worksheets**: Connectivity matrices for chemical synapses (hermaphrodite)
- **Gap junction worksheets**: Connectivity matrices for gap junctions (hermaphrodite, asymmetric)

### Processing the Excel File

The project includes a Python script (`parse_wormwiring_final.py`) that automatically processes the Excel file:

```bash
python parse_wormwiring_final.py
```

This script will:
1. Extract neuron connectivity data from the Excel file
2. Filter to include only neurons (S, I, M types)
3. Exclude non-neuronal cells (muscles, pharynx, sex-specific cells)
4. Generate processed graph files in `data/processed/`

### Required Excel File

The script expects the following file:
- `data/raw/wormwiring_SI5_connectome_adjacency_corrected_2020.xlsx`

If you have a different version or filename, you may need to update the script or rename the file accordingly.
