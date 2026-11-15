import argparse
import os

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, to_networkx


DEFAULT_DATASET = ""
DEFAULT_DATA_DIR = ""


def find_first_existing(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def list_files_by_suffix(directory, suffix):
    if not directory or not os.path.isdir(directory):
        return []
    return [os.path.join(directory, name) for name in os.listdir(directory) if name.lower().endswith(suffix.lower())]


def resolve_dataset_dir(data_dir, dataset):
    if not data_dir:
        return os.getcwd()

    if dataset and os.path.basename(os.path.normpath(data_dir)) != dataset:
        child = os.path.join(data_dir, dataset)
        if os.path.isdir(child):
            return child
    return data_dir


def get_cycle_clique_matrix(adj_mx):
    num_nodes = adj_mx.shape[0]
    print("Building cycle-clique matrix with PyG-compatible graph conversion...")

    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(adj_tensor)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    graph = to_networkx(data, to_undirected=True)

    if graph.number_of_edges() > num_nodes * (num_nodes - 1) * 0.2:
        print("Warning: graph is dense; cycle_basis may be slow.")

    try:
        cycles = nx.cycle_basis(graph)
    except Exception as exc:
        print(f"Failed to compute cycles: {exc}")
        cycles = []

    print(f"Found {len(cycles)} cycles.")

    cycle_adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for cycle in cycles:
        for i in cycle:
            cycle_adj[i, cycle] = 1.0
        cycle_adj[cycle, cycle] = 0.0

    return cycle_adj


def save_outputs(adj_mx, cycle_mx, output_a_path, output_cycle_path):
    np.save(output_a_path, adj_mx)
    np.save(output_cycle_path, cycle_mx)
    print(f"Saved adjacency matrix: {output_a_path}")
    print(f"Saved cycle matrix: {output_cycle_path}")


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_node_ids(npz_path, csv_df, data_dir):
    if npz_path and os.path.exists(npz_path):
        try:
            raw_data = np.load(npz_path)
            if "ids" in raw_data.files:
                return raw_data["ids"]
            if "nodes" in raw_data.files:
                return raw_data["nodes"]
            if "data" in raw_data.files:
                num_nodes = raw_data["data"].shape[1]
                print(f"Detected NPZ data shape, inferred {num_nodes} nodes.")
                return np.arange(num_nodes)
        except Exception as exc:
            print(f"Failed to read NPZ: {exc}")

    txt_path = os.path.join(data_dir, "graph_sensor_ids.txt") if data_dir else ""
    if txt_path and os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                raise ValueError("graph_sensor_ids.txt is empty")
            if "," in content:
                ids = [int(x) for x in content.split(",") if x.strip()]
            else:
                ids = [int(x) for x in content.splitlines() if x.strip()]
            return np.array(ids)
        except Exception as exc:
            print(f"Failed to read txt IDs: {exc}")

    unique_ids = pd.unique(csv_df[["from", "to"]].values.ravel("K"))
    unique_ids.sort()
    return unique_ids


def read_edge_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        try:
            float(df.columns[0])
            df = pd.read_csv(csv_path, header=None)
            df.columns = ["from", "to", "cost"]
        except ValueError:
            if "from" not in df.columns:
                df.columns = ["from", "to", "cost"]
    except Exception:
        df = pd.read_csv(csv_path, header=None)
        df.columns = ["from", "to", "cost"]
    return df


def get_binary_adjacency_matrix(distance_df, sensor_ids):
    num_sensors = len(sensor_ids)
    adj_mx = np.zeros((num_sensors, num_sensors), dtype=np.float64)
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}

    values = distance_df.values
    for row in values:
        s_from = int(row[0])
        s_to = int(row[1])
        dist = row[2]

        if s_from in sensor_id_to_ind and s_to in sensor_id_to_ind:
            if not np.isinf(dist):
                u = sensor_id_to_ind[s_from]
                v = sensor_id_to_ind[s_to]
                adj_mx[u, v] = 1.0
                adj_mx[v, u] = 1.0

    np.fill_diagonal(adj_mx, 0)
    print(f"Built binary adjacency matrix. Non-zero entries: {np.count_nonzero(adj_mx)}")
    return adj_mx


def build_from_h5(h5_path, h5_key, top_k, sparsity_threshold):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        key = h5_key
        if key not in f.keys():
            keys = list(f.keys())
            print(f"Key '{key}' not found. Available keys: {keys}")
            if not keys:
                raise ValueError("No datasets found in h5 file")
            key = keys[0]
        raw_matrix = f[key][:]
        print(f"Loaded h5 key '{key}', shape={raw_matrix.shape}")

    num_nodes = raw_matrix.shape[0]
    binary_adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    non_zero_count = np.count_nonzero(raw_matrix)
    sparsity = non_zero_count / (num_nodes * num_nodes)
    print(f"Raw matrix non-zero ratio: {sparsity:.2%}")

    if sparsity < sparsity_threshold:
        print("Using sparse rule: non-zero means connected.")
        binary_adj[raw_matrix != 0] = 1.0
    else:
        print(f"Using dense rule: top-k nearest neighbors, k={top_k}.")
        temp_matrix = raw_matrix.copy()
        temp_matrix[temp_matrix == 0] = np.inf
        for i in range(num_nodes):
            nearest_indices = np.argsort(temp_matrix[i, :])[:top_k]
            binary_adj[i, nearest_indices] = 1.0

    binary_adj = np.maximum(binary_adj, binary_adj.T)
    np.fill_diagonal(binary_adj, 0)
    return binary_adj


def build_from_matrix_csv(matrix_csv_path, traffic_csv_path=None):
    df_adj = pd.read_csv(matrix_csv_path, header=None)
    adj_raw = df_adj.values
    rows, cols = adj_raw.shape
    print(f"Loaded matrix csv, shape={rows}x{cols}")

    if rows != cols:
        if cols == rows + 1:
            print("Detected index column, dropping first column to make square matrix.")
            adj_raw = adj_raw[:, 1:]
            rows, cols = adj_raw.shape
        else:
            raise ValueError(f"Matrix is not square: {rows}x{cols}")

    num_nodes = rows
    if traffic_csv_path and os.path.exists(traffic_csv_path):
        try:
            df_flow = pd.read_csv(traffic_csv_path, nrows=2, header=None)
            print(f"Traffic file columns: {df_flow.shape[1]}, matrix nodes: {num_nodes}")
        except Exception:
            pass

    adj_binary = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            val = adj_raw[i, j]
            if val != 0 and not np.isnan(val) and not np.isinf(val):
                adj_binary[i, j] = 1.0
                adj_binary[j, i] = 1.0

    np.fill_diagonal(adj_binary, 0)
    if np.count_nonzero(adj_binary) == 0:
        raise ValueError("Matrix is all zeros after binarization")
    return adj_binary


def process_w228_matrix(csv_path, threshold=2000):
    df_matrix = pd.read_csv(csv_path, header=None)
    matrix = df_matrix.values
    num_nodes = matrix.shape[0]
    rows, cols = np.where((matrix > 0) & (matrix < threshold))

    edges = []
    for r, c in zip(rows, cols):
        edges.append([r, c, matrix[r, c]])

    edge_df = pd.DataFrame(edges, columns=["from", "to", "cost"])
    sensor_ids = list(range(num_nodes))
    print(f"Built edge list from W_228 matrix with threshold={threshold}, edges={len(edges)}")
    return edge_df, sensor_ids


def infer_mode(args):
    if args.mode != "auto":
        return args.mode
    if args.h5:
        return "h5"
    if args.w228_csv:
        return "pems07m"
    if args.matrix_csv:
        return "matrix_csv"
    return "edge_csv"


def resolve_auto_inputs(args):
    mode = infer_mode(args)
    resolved = {
        "csv": args.csv,
        "npz": args.npz,
        "matrix_csv": args.matrix_csv,
        "traffic_csv": args.traffic_csv,
        "h5": args.h5,
        "w228_csv": args.w228_csv,
    }

    if args.mode != "auto":
        return mode, resolved

    dataset = args.dataset
    dataset_dir = resolve_dataset_dir(args.data_dir or os.getcwd(), dataset)

    # User-provided paths have highest priority.
    if resolved["h5"]:
        return "h5", resolved
    if resolved["w228_csv"]:
        return "pems07m", resolved
    if resolved["matrix_csv"]:
        return "matrix_csv", resolved
    if resolved["csv"]:
        return "edge_csv", resolved

    dataset_lower = dataset.lower() if dataset else ""
    hint_mode = ""
    if dataset_lower in {"nyctaxi"}:
        hint_mode = "h5"
    elif dataset_lower in {"nycbike"}:
        hint_mode = "matrix_csv"
    elif dataset_lower in {"pems07(m)", "pems07m"}:
        hint_mode = "pems07m"
    elif dataset_lower.startswith("pems"):
        hint_mode = "edge_csv"

    # Build candidate file paths under dataset directory.
    resolved["w228_csv"] = find_first_existing(
        [
            os.path.join(dataset_dir, "PEMS07_W_228.csv"),
            os.path.join(dataset_dir, "V_228.csv"),
            os.path.join(args.data_dir or "", "PEMS07_W_228.csv"),
        ]
    )

    resolved["h5"] = find_first_existing(
        [
            os.path.join(dataset_dir, "all_graph.h5"),
            os.path.join(dataset_dir, f"{dataset}.h5") if dataset else "",
            os.path.join(args.data_dir or "", f"{dataset}.h5") if dataset else "",
        ]
    )
    if not resolved["h5"]:
        h5_candidates = list_files_by_suffix(dataset_dir, ".h5")
        resolved["h5"] = h5_candidates[0] if h5_candidates else ""

    resolved["matrix_csv"] = find_first_existing(
        [
            os.path.join(dataset_dir, "dis_bb.csv"),
            os.path.join(dataset_dir, "distance.csv"),
            os.path.join(args.data_dir or "", "dis_bb.csv"),
        ]
    )
    resolved["traffic_csv"] = find_first_existing(
        [
            os.path.join(dataset_dir, "bike_pick.csv"),
            os.path.join(dataset_dir, "traffic.csv"),
            os.path.join(args.data_dir or "", "bike_pick.csv"),
        ]
    )

    resolved["csv"] = find_first_existing(
        [
            os.path.join(dataset_dir, f"{dataset}.csv") if dataset else "",
            os.path.join(args.data_dir or "", f"{dataset}.csv") if dataset else "",
        ]
    )
    resolved["npz"] = find_first_existing(
        [
            os.path.join(dataset_dir, f"{dataset}.npz") if dataset else "",
            os.path.join(args.data_dir or "", f"{dataset}.npz") if dataset else "",
        ]
    )

    # If dataset name has a strong hint, respect it when matching files are available.
    if hint_mode == "pems07m" and resolved["w228_csv"]:
        return "pems07m", resolved
    if hint_mode == "h5" and resolved["h5"]:
        return "h5", resolved
    if hint_mode == "matrix_csv" and resolved["matrix_csv"]:
        return "matrix_csv", resolved
    if hint_mode == "edge_csv" and resolved["csv"]:
        return "edge_csv", resolved

    # File-based fallback detection.
    if resolved["w228_csv"]:
        return "pems07m", resolved
    if resolved["h5"]:
        return "h5", resolved
    if resolved["matrix_csv"]:
        return "matrix_csv", resolved
    if resolved["csv"]:
        return "edge_csv", resolved

    raise ValueError(
        "Auto mode failed to detect input files. Provide --mode with explicit path parameters."
    )


def default_output_paths(args, mode):
    data_dir = args.data_dir or os.getcwd()
    dataset = args.dataset or "dataset"

    if args.suffix is None:
        if mode == "edge_csv":
            suffix = "_binary"
        elif mode == "pems07m":
            suffix = "_M"
        else:
            suffix = ""
    else:
        suffix = args.suffix

    output_a = args.output_a or os.path.join(data_dir, f"{dataset}{suffix}_A.npy")
    output_cycle = args.output_cycle or os.path.join(data_dir, f"{dataset}{suffix}_cycle.npy")
    return output_a, output_cycle


def main():
    parser = argparse.ArgumentParser(
        description="Unified generator for adjacency matrix (A) and cycle-clique matrix (Ac)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "edge_csv", "matrix_csv", "h5", "pems07m"],
        help="Input mode",
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Dataset directory")
    parser.add_argument("--suffix", type=str, default=None, help="Output filename suffix")

    parser.add_argument("--csv", type=str, default="", help="Edge-list csv path")
    parser.add_argument("--npz", type=str, default="", help="Optional npz path for node ids")
    parser.add_argument("--matrix_csv", type=str, default="", help="Square matrix csv path")
    parser.add_argument("--traffic_csv", type=str, default="", help="Optional traffic csv path")
    parser.add_argument("--h5", type=str, default="", help="H5 path")
    parser.add_argument("--h5_key", type=str, default="dis_tt", help="H5 dataset key")
    parser.add_argument("--w228_csv", type=str, default="", help="PEMS07(M) W_228 csv path")

    parser.add_argument("--threshold", type=float, default=2000.0, help="Distance threshold for W_228")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k for dense h5 matrix")
    parser.add_argument(
        "--sparsity_threshold",
        type=float,
        default=0.1,
        help="If non-zero ratio < threshold, treat h5 matrix as sparse",
    )

    parser.add_argument("--output_a", type=str, default="", help="Output path for adjacency matrix")
    parser.add_argument("--output_cycle", type=str, default="", help="Output path for cycle matrix")

    args = parser.parse_args()
    mode, resolved = resolve_auto_inputs(args)
    output_a_path, output_cycle_path = default_output_paths(args, mode)
    ensure_parent_dir(output_a_path)
    ensure_parent_dir(output_cycle_path)

    print(f"Mode: {mode}")

    if mode == "edge_csv":
        data_dir = args.data_dir or os.getcwd()
        csv_path = resolved["csv"] or os.path.join(resolve_dataset_dir(data_dir, args.dataset), f"{args.dataset}.csv")
        npz_path = resolved["npz"] or os.path.join(resolve_dataset_dir(data_dir, args.dataset), f"{args.dataset}.npz")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        edge_df = read_edge_csv(csv_path)
        sensor_ids = get_node_ids(npz_path, edge_df, data_dir)
        print(f"Detected node count: {len(sensor_ids)}")
        adj_mx = get_binary_adjacency_matrix(edge_df, sensor_ids)

    elif mode == "matrix_csv":
        if not resolved["matrix_csv"]:
            raise ValueError("--matrix_csv is required for matrix_csv mode")
        adj_mx = build_from_matrix_csv(resolved["matrix_csv"], resolved["traffic_csv"])

    elif mode == "h5":
        if not resolved["h5"]:
            raise ValueError("--h5 is required for h5 mode")
        adj_mx = build_from_h5(resolved["h5"], args.h5_key, args.top_k, args.sparsity_threshold)

    elif mode == "pems07m":
        if resolved["w228_csv"]:
            w228_csv = resolved["w228_csv"]
        else:
            w228_csv = os.path.join(resolve_dataset_dir(args.data_dir or os.getcwd(), args.dataset), "PEMS07_W_228.csv")
        if not os.path.exists(w228_csv):
            raise FileNotFoundError(f"W_228 csv not found: {w228_csv}")

        edge_df, sensor_ids = process_w228_matrix(w228_csv, args.threshold)
        adj_mx = get_binary_adjacency_matrix(edge_df, sensor_ids)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cycle_mx = get_cycle_clique_matrix(adj_mx)
    save_outputs(adj_mx, cycle_mx, output_a_path, output_cycle_path)
    print("All done.")


if __name__ == "__main__":
    main()