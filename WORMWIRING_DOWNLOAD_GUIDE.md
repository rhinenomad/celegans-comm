# WormWiring 数据下载指南

## 手动下载步骤

由于 WormWiring 网站（https://wormwiring.org/pages/）的数据可能需要通过网站界面访问，建议按以下步骤操作：

### 1. 访问网站
打开浏览器访问：https://wormwiring.org/pages/

### 2. 查找数据下载链接
在网站上查找以下类型的数据文件：
- 化学突触数据（Chemical Synapses）
- 间隙连接数据（Gap Junctions）
- 邻接矩阵（Adjacency Matrix）
- 节点列表（Neuron List/Nodes）
- 边列表（Edges）

### 3. 下载并重命名
将下载的文件保存到 `data/raw/` 目录，并重命名为：

| 原始文件名 | 建议重命名 | 描述 |
|-----------|-----------|------|
| 化学突触数据 | `wormwiring_chemical_synapses.csv` | 化学突触连接 |
| 间隙连接数据 | `wormwiring_gap_junctions.csv` | 间隙连接（无向） |
| 完整边数据 | `wormwiring_edges.csv` | 所有边的数据 |
| 节点/神经元列表 | `wormwiring_nodes.csv` | 节点元数据（名称、类型等） |
| 邻接矩阵 | `wormwiring_adjacency_matrix.csv` | 邻接矩阵格式 |

### 4. 文件格式期望

**边数据文件应包含：**
- `source` 或 `from` 列：源节点
- `target` 或 `to` 列：目标节点  
- `weight` 列（可选）：边权重
- `type` 列（可选）：边类型（chemical/gap）

**节点数据文件应包含：**
- `id` 或 `node_id`：节点ID
- `name` 或 `neuron_name`：神经元名称
- `type` 列（可选）：节点类型（sensory/motor/interneuron）

## 自动下载脚本（如果已知URL）

如果找到数据文件的直接下载链接，可以使用以下命令：

```bash
# 示例：下载CSV文件
cd data/raw
curl -L -o wormwiring_chemical_synapses.csv "https://wormwiring.org/data/chemical_synapses.csv"
curl -L -o wormwiring_gap_junctions.csv "https://wormwiring.org/data/gap_junctions.csv"
curl -L -o wormwiring_nodes.csv "https://wormwiring.org/data/nodes.csv"
```

## 检查已下载的文件

运行以下命令检查文件：

```bash
cd data/raw
ls -lh wormwiring_*
```

