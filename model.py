import copy
import random
import warnings
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import Tk, filedialog
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, RGCNConv
from torch_geometric.utils import from_scipy_sparse_matrix

warnings.filterwarnings("ignore")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_COL = "y_true(fc)"
SOURCE_COL = "source_name"
TIME_COL = "time"

HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.30
LR = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 350
RUNS = 5
K_NEIGHBORS = 10

# Generate the five shared seeds once.
SEEDS = [42, 52, 62, 72, 82]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def select_csv_file():
    root = Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select 5G Telemetry CSV",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    if not path:
        raise FileNotFoundError("No telemetry CSV selected.")
    return path


def infer_nf_type(value):
    """
    Infer 5G NF type from source_name when possible.
    Unknown values are kept as OTHER rather than guessed.
    """
    s = str(value).lower()

    if "amf" in s:
        return "AMF"
    if "smf" in s:
        return "SMF"
    if "udm" in s:
        return "UDM"
    if "upf" in s:
        return "UPF"
    if "ausf" in s:
        return "AUSF"

    return "OTHER"


def load_data(path=None):
    path = path or select_csv_file()
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found."
        )
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(
        df[TARGET_COL].values
    )
    if SOURCE_COL in df.columns:
        nf_types = (
            df[SOURCE_COL]
            .astype(str)
            .map(infer_nf_type)
            .values
        )
    else:
        nf_types = np.array(
            ["NF"] * len(df),
            dtype=object
        )
    node_type_encoder = LabelEncoder()

    node_type = node_type_encoder.fit_transform(
        nf_types
    )
    feature_df = df.drop(
        columns=[
            TARGET_COL,
            SOURCE_COL,
            TIME_COL
        ],
        errors="ignore"
    )

    feature_df = feature_df.select_dtypes(
        include=[np.number]
    )
    if feature_df.shape[1] == 0:
        raise ValueError(
            "No numeric telemetry features found."
        )
    X = MinMaxScaler().fit_transform(
        feature_df.values
    )
    x = torch.tensor(
        X,
        dtype=torch.float32
    )
    y = torch.tensor(
        y,
        dtype=torch.long
    )
    node_type = torch.tensor(
        node_type,
        dtype=torch.long
    )
    print("\nDATASET")
    print("-" * 65)
    print(f"Nodes        : {x.size(0)}")
    print(f"Features     : {x.size(1)}")
    print(f"Classes      : {len(label_encoder.classes_)}")
    print(f"Node types   : {list(node_type_encoder.classes_)}")
    print(f"Device       : {DEVICE}")
    print(f"Random seeds : {SEEDS}")
    print("-" * 65)
    timestamps = pd.to_datetime(df[TIME_COL], errors="coerce") if TIME_COL in df else None
    return (x, y, node_type, label_encoder, node_type_encoder, timestamps)

def build_relation_graph(x):
    adj = kneighbors_graph(
        x.cpu().numpy(),
        n_neighbors=K_NEIGHBORS,
        mode="distance",
        include_self=False
    )
    edge_index, edge_weight = (
        from_scipy_sparse_matrix(adj)
    )
    edge_index = edge_index.long()
    distances = (
        edge_weight.float()
        .cpu()
        .numpy()
    )
    q1 = np.quantile(
        distances,
        1.0 / 3.0
    )
    q2 = np.quantile(
        distances,
        2.0 / 3.0
    )
    edge_type = np.zeros(
        len(distances),
        dtype=np.int64
    )
    edge_type[
        (distances > q1)
        & (distances <= q2)
    ] = 1
    edge_type[
        distances > q2
    ] = 2
    edge_type = torch.tensor(
        edge_type,
        dtype=torch.long
    )
    relation_names = [
        "strong_similarity",
        "moderate_similarity",
        "weak_similarity"
    ]
    print("\nGRAPH")
    print("-" * 65)
    print(f"Edges             : {edge_index.size(1)}")
    print(f"Relation channels : {len(relation_names)}")
    print(f"Relations         : {relation_names}")
    print("-" * 65)

    return (
        edge_index,
        edge_type,
        relation_names
    )
def create_masks(y, seed):
    indices = np.arange(
        len(y)
    )
    labels = y.cpu().numpy()
    try:
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=0.60,
            stratify=labels,
            random_state=seed,
            shuffle=True
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            stratify=labels[temp_idx],
            random_state=seed,
            shuffle=True
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=0.60,
            random_state=seed,
            shuffle=True
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=seed,
            shuffle=True
        )
    n = len(y)
    train_mask = torch.zeros(
        n,
        dtype=torch.bool
    )
    val_mask = torch.zeros(
        n,
        dtype=torch.bool
    )
    test_mask = torch.zeros(
        n,
        dtype=torch.bool
    )
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return (
        train_mask,
        val_mask,
        test_mask
    )
class TypeSpecificLinear(nn.Module):
    """
    Implements W_{tau(v)} or P_{tau(v)}^(l).
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_node_types
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(
                in_dim,
                out_dim
            )
            for _ in range(num_node_types)
        ])

    def forward(
        self,
        x,
        node_type
    ):
        out_dim = self.layers[0].out_features
        out = torch.zeros(
            x.size(0),
            out_dim,
            dtype=x.dtype,
            device=x.device
        )
        for t, layer in enumerate(
            self.layers
        ):
            mask = (
                node_type == t
            )
            if mask.any():
                out[mask] = layer(
                    x[mask]
                )
        return out
class RelationPropagation(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_relations,
        heads=2,
        semantic_fusion=True
    ):
        super().__init__()
        self.num_relations = num_relations
        self.semantic_fusion = semantic_fusion
       
        self.rel_convs = nn.ModuleList([
            GATConv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=DROPOUT,
                add_self_loops=False
            )
            for _ in range(num_relations)
        ])
        # q^T tanh(W_s z_{v,r})
        self.semantic_score = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        h,
        edge_index,
        edge_type
    ):
        relation_embeddings = []
        relation_available = []

        for r, conv in enumerate(
            self.rel_convs
        ):
            relation_mask = (
                edge_type == r
            )
            edge_r = edge_index[
                :,
                relation_mask
            ]
            if edge_r.size(1) == 0:
                z_r = torch.zeros_like(
                    h
                )
                available = torch.zeros(
                    h.size(0),
                    dtype=torch.bool,
                    device=h.device
                )
            else:
                
                z_r = F.elu(
                    conv(
                        h,
                        edge_r
                    )
                )
                available = torch.zeros(
                    h.size(0),
                    dtype=torch.bool,
                    device=h.device
                )
                available[
                    edge_r[1].unique()
                ] = True
            relation_embeddings.append(
                z_r
            )
            relation_available.append(
                available
            )
        # Z[v,r,:] = z_{v,r}
        Z = torch.stack(
            relation_embeddings,
            dim=1
        )
        # A[v,r] indicates whether r is incident on v
        A = torch.stack(
            relation_available,
            dim=1
        )
        if self.semantic_fusion:
            score = (
                self.semantic_score(Z)
                .squeeze(-1)
            )
            score = score.masked_fill(
                ~A,
                -1e9
            )
            beta = torch.softmax(
                score,
                dim=1
            )
            valid = A.any(
                dim=1,
                keepdim=True
            )

            beta = torch.where(
                valid,
                beta,
                torch.zeros_like(beta)
            )
            h_rel = torch.sum(
                beta.unsqueeze(-1) * Z,
                dim=1
            )
        else:
            weights = (
                A.float()
                .unsqueeze(-1)
            )
            denominator = (
                weights.sum(dim=1)
                .clamp(min=1.0)
            )
            h_rel = (
                (Z * weights)
                .sum(dim=1)
                / denominator
            )
        return h_rel

class NodeClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dim
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                hidden_dim,
                out_dim
            )
        )
    def forward(
        self,
        h
    ):
        return self.net(h)

class MTLGNN(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()
        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )
        self.propagation = RelationPropagation(
            hidden_dim,
            num_relations,
            semantic_fusion=False
        )
        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )
        self.auxiliary = nn.Linear(
            hidden_dim,
            1
        )
    def forward(
        self,
        data
    ):
        h = F.relu(
            self.encoder(
                data.x
            )
        )
        h = self.propagation(
            h,
            data.edge_index,
            data.edge_type
        )
        return (
            self.classifier(h),
            self.auxiliary(h)
        )
class HeteroGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()

        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )
        self.propagation = RelationPropagation(
            hidden_dim,
            num_relations,
            semantic_fusion=False
        )
        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )
    def forward(
        self,
        data
    ):
        h = F.relu(
            self.encoder(
                data.x
            )
        )
        h = self.propagation(
            h,
            data.edge_index,
            data.edge_type
        )
        return self.classifier(h)
class HPN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()
        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )
        self.propagation = RelationPropagation(
            hidden_dim,
            num_relations,
            heads=1,
            semantic_fusion=True
        )
        self.gate = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),
            nn.Sigmoid()
        )
        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )
    def forward(
        self,
        data
    ):
        h0 = F.relu(
            self.encoder(
                data.x
            )
        )
        hp = self.propagation(
            h0,
            data.edge_index,
            data.edge_type
        )
        gate = self.gate(
            torch.cat(
                [h0, hp],
                dim=1
            )
        )

        h = (
            gate * h0
            + (1.0 - gate) * hp
        )

        return self.classifier(h)


class RLPHGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()

        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )
        self.propagation = RelationPropagation(
            hidden_dim,
            num_relations,
            semantic_fusion=True
        )

        self.pre_classifier = nn.Linear(
            hidden_dim,
            out_dim
        )

        self.label_projection = nn.Linear(
            out_dim,
            hidden_dim
        )

        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )

    def forward(
        self,
        data
    ):
        h = F.relu(
            self.encoder(
                data.x
            )
        )

        h = self.propagation(
            h,
            data.edge_index,
            data.edge_type
        )

        pseudo = torch.softmax(
            self.pre_classifier(h),
            dim=1
        )

        label_h = F.relu(
            self.label_projection(
                pseudo
            )
        )

        h = (
            h
            + 0.2 * label_h
        )

        return self.classifier(h)


class HFDG(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()

        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )

        self.propagation = RelationPropagation(
            hidden_dim,
            num_relations,
            semantic_fusion=False
        )

        self.failure_gate = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.Sigmoid()
        )

        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )

    def forward(
        self,
        data
    ):
        h = F.relu(
            self.encoder(
                data.x
            )
        )

        h = self.propagation(
            h,
            data.edge_index,
            data.edge_type
        )

        h = (
            self.failure_gate(h)
            * h
        )

        return self.classifier(h)


class IFTSHGPN(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()

        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )

        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=2,
            concat=False,
            dropout=DROPOUT
        )

        self.gate = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.Sigmoid()
        )

        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )

    def forward(
        self,
        data
    ):
        h = F.relu(
            self.encoder(
                data.x
            )
        )

        h = F.elu(
            self.gat(
                h,
                data.edge_index
            )
        )

        h = (
            h
            + self.gate(h) * h
        )

        return self.classifier(h)


class HAN(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()

        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )

        self.propagation = RelationPropagation(
            hidden_dim,
            num_relations,
            semantic_fusion=True
        )

        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )

    def forward(
        self,
        data
    ):
        h = F.elu(
            self.encoder(
                data.x
            )
        )

        h = self.propagation(
            h,
            data.edge_index,
            data.edge_type
        )

        return self.classifier(h)


class HGT(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()

        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )

        self.propagation = RelationPropagation(
            hidden_dim,
            num_relations,
            heads=4,
            semantic_fusion=True
        )

        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            )
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )

    def forward(
        self,
        data
    ):
        h0 = F.relu(
            self.encoder(
                data.x
            )
        )

        hp = self.propagation(
            h0,
            data.edge_index,
            data.edge_type
        )

        h = self.norm1(
            h0 + hp
        )

        h = self.norm2(
            h + self.ffn(h)
        )

        return self.classifier(h)


class MAGNN(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()

        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )

        self.path1 = RelationPropagation(
            hidden_dim,
            num_relations,
            heads=1,
            semantic_fusion=True
        )

        self.path2 = RelationPropagation(
            hidden_dim,
            num_relations,
            heads=1,
            semantic_fusion=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                2
            ),
            nn.Softmax(
                dim=1
            )
        )

        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )

    def forward(
        self,
        data
    ):
        h0 = F.relu(
            self.encoder(
                data.x
            )
        )

        h1 = self.path1(
            h0,
            data.edge_index,
            data.edge_type
        )

        h2 = self.path2(
            h1,
            data.edge_index,
            data.edge_type
        )

        gamma = self.fusion(
            torch.cat(
                [h1, h2],
                dim=1
            )
        )

        h = (
            gamma[:, 0:1] * h1
            + gamma[:, 1:2] * h2
        )

        return self.classifier(h)


class KGBP(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    ):
        super().__init__()

        self.encoder = nn.Linear(
            in_dim,
            hidden_dim
        )

        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=DROPOUT
        )

        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=DROPOUT
        )

        self.residual = nn.Linear(
            in_dim,
            hidden_dim
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.classifier = NodeClassifier(
            hidden_dim,
            out_dim
        )

    def forward(
        self,
        data
    ):
        h = F.relu(
            self.encoder(
                data.x
            )
        )

        h = F.elu(
            self.gat1(
                h,
                data.edge_index
            )
        )

        h = F.elu(
            self.gat2(
                h,
                data.edge_index
            )
        )

        h = self.norm(
            h
            + self.residual(
                data.x
            )
        )

        return self.classifier(h)



class RGCN(nn.Module):
    """Schlichtkrull-style relation-specific graph convolution baseline."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.classifier = NodeClassifier(hidden_dim, out_dim)

    def forward(self, data):
        h = F.relu(self.encoder(data.x))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.conv1(h, data.edge_index, data.edge_type))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.conv2(h, data.edge_index, data.edge_type))
        return self.classifier(h)




class HGPNF(nn.Module):
    """
    h_v^(0) = sigma(W_{tau(v)} x_v + b_{tau(v)})

    for l = 0,...,L-1:
        z_{v,r}^{(l+1)} from relation-specific GAT
        beta_{v,r}^{(l)} from semantic softmax
        h_v^(l+1) =
            LN(
                P_{tau(v)}^(l) h_v^(l)
                + sum_r beta_{v,r}^{(l)} z_{v,r}^{(l+1)}
            )

    logits_v = W_o h_v^(L) + b_o
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_node_types,
        num_relations,
        num_layers=NUM_LAYERS
    ):
        super().__init__()

        self.num_layers = num_layers

        # W_{tau(v)}
        self.type_encoder = TypeSpecificLinear(
            in_dim,
            hidden_dim,
            num_node_types
        )

        # Relation-specific propagation for every l
        self.propagation_layers = nn.ModuleList([
            RelationPropagation(
                hidden_dim,
                num_relations,
                heads=2,
                semantic_fusion=True
            )
            for _ in range(num_layers)
        ])

        # P_{tau(v)}^(l)
        self.residual_layers = nn.ModuleList([
            TypeSpecificLinear(
                hidden_dim,
                hidden_dim,
                num_node_types
            )
            for _ in range(num_layers)
        ])

        # LN
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(
                hidden_dim
            )
            for _ in range(num_layers)
        ])

        # W_o h_v^(L) + b_o
        self.classifier = nn.Linear(
            hidden_dim,
            out_dim
        )

    def forward(
        self,
        data
    ):
        # ----------------------------------------------------
        # h_v^(0)
        # ----------------------------------------------------
        h = F.relu(
            self.type_encoder(
                data.x,
                data.node_type
            )
        )

        h = F.dropout(
            h,
            p=DROPOUT,
            training=self.training
        )

        # ----------------------------------------------------
        # l = 0,...,L-1
        # ----------------------------------------------------
        for l in range(
            self.num_layers
        ):
            h_prev = h

            # sum_r beta_{v,r} z_{v,r}
            h_rel = self.propagation_layers[l](
                h_prev,
                data.edge_index,
                data.edge_type
            )

            # P_{tau(v)}^(l) h_v^(l)
            h_res = self.residual_layers[l](
                h_prev,
                data.node_type
            )

            # h_v^(l+1)
            h = self.norm_layers[l](
                h_res + h_rel
            )

            h = F.dropout(
                h,
                p=DROPOUT,
                training=self.training
            )

        # Raw logits. Softmax is not applied here because
        # PyTorch cross_entropy expects logits directly.
        logits = self.classifier(
            h
        )

        return logits

BASELINE_CLASSES = {
    "MTL-GNN": MTLGNN,
    "Hetero-GNN": HeteroGNN,
    "HPN": HPN,
    "RLP-HGNN": RLPHGNN,
    "HFDG": HFDG,
    "IFT-SHGPN": IFTSHGPN,
    "HAN": HAN,
    "HGT": HGT,
    "MAGNN": MAGNN,
    "KGBP": KGBP,
    "R-GCN": RGCN
}

MODEL_ORDER = [
    "MTL-GNN",
    "Hetero-GNN",
    "HPN",
    "RLP-HGNN",
    "HFDG",
    "IFT-SHGPN",
    "HAN",
    "HGT",
    "MAGNN",
    "KGBP",
    "R-GCN",
    "HGPNF (Ours)"
]


def build_model(
    model_name,
    in_dim,
    hidden_dim,
    out_dim,
    num_node_types,
    num_relations
):
    if model_name == "HGPNF (Ours)":
        return HGPNF(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_node_types=num_node_types,
            num_relations=num_relations,
            num_layers=NUM_LAYERS
        )

    ModelClass = BASELINE_CLASSES[
        model_name
    ]

    return ModelClass(
        in_dim,
        hidden_dim,
        out_dim,
        num_relations
    )


def get_output(
    model,
    model_name,
    data
):
    out = model(data)

    if model_name == "MTL-GNN":
        return (
            out[0],
            out[1]
        )

    return (
        out,
        None
    )


def evaluate(
    model,
    model_name,
    data,
    mask
):
    model.eval()

    with torch.no_grad():
        logits, _ = get_output(
            model,
            model_name,
            data
        )

        pred = logits.argmax(
            dim=1
        )

    y_true = (
        data.y[mask]
        .cpu()
        .numpy()
    )

    y_pred = (
        pred[mask]
        .cpu()
        .numpy()
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=np.arange(int(data.y.max().item()) + 1),
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        labels=np.arange(int(data.y.max().item()) + 1),
        zero_division=0
    )

    return (
        micro_f1,
        macro_f1,
        macro_precision
    )

def train_model(
    model,
    model_name,
    data
):
    # Adam weight_decay is 5e-4; this is not necessarily the same
    # coefficient as an explicitly written lambda * ||Theta||^2 objective.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_macro = -np.inf
    best_state = None

    for epoch in range(
        EPOCHS
    ):
        model.train()
        optimizer.zero_grad()

        logits, aux = get_output(
            model,
            model_name,
            data
        )

        cls_loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask]
        )

        if (
            model_name == "MTL-GNN"
            and aux is not None
        ):
            aux_loss = torch.mean(
                aux[data.train_mask] ** 2
            )

            loss = (
                cls_loss
                + 1e-4 * aux_loss
            )
        else:
            loss = cls_loss

        loss.backward()
        optimizer.step()

        _, val_macro, _ = evaluate(
            model,
            model_name,
            data,
            data.val_mask
        )

        if val_macro > best_val_macro:
            best_val_macro = val_macro

            best_state = copy.deepcopy(
                model.state_dict()
            )

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return evaluate(
        model,
        model_name,
        data,
        data.test_mask
    )

def print_latex_rows(results_df):
    """Columns match manuscript: Macro-F1, Micro-F1, Macro-Precision."""
    print("\nLATEX TABLE ROWS (Macro-F1, Micro-F1, Macro-Precision)\n")
    for _, row in results_df.iterrows():
        name = row["Model"]
        cells = []
        for key in ("Macro-F1", "Micro-F1", "Macro-Precision"):
            value = f"{row[key + ' Mean']:.4f} \\pm {row[key + ' Std']:.4f}"
            cells.append(f"$\\mathbf{{{value}}}$" if name == "HGPNF (Ours)" else f"${value}$")
        print(name + " & " + " & ".join(cells) + " " + "\\\\")


def prepare_features(raw_x, train_mask):
    """Train-only median imputation and MinMax scaling; prevents fit leakage."""
    x = raw_x.cpu().numpy().copy()
    train = x[train_mask.cpu().numpy()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isfinite(x), x, medians[None, :])
    scaler = MinMaxScaler().fit(x[train_mask.cpu().numpy()])
    return torch.tensor(scaler.transform(x), dtype=torch.float32)


def chronological_masks(timestamps, n):
    """Chronological 60/20/20 split; no overlapping indices."""
    if timestamps is None or timestamps.isna().any():
        raise ValueError("Chronological splitting requires complete parseable time values.")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    a, b = int(.6*n), int(.8*n)
    masks = []
    for inds in (order[:a], order[a:b], order[b:]):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[inds] = True
        masks.append(mask)
    return tuple(masks)


def print_class_distribution(y, encoder, masks):
    labels = y.cpu().numpy()
    print("\nPER-CLASS SAMPLE DISTRIBUTION (full/train/val/test)")
    print("-"*100)
    for c, name in enumerate(encoder.classes_):
        counts = [int(np.sum(labels == c))]
        counts += [int(np.sum(labels[m.cpu().numpy()] == c)) for m in masks]
        print(f"{name:<48} {counts[0]:>5} {counts[1]:>5} {counts[2]:>5} {counts[3]:>5}")


def print_hgpnf_diagnostics(model, data, encoder, run):
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
    yt = data.y[data.test_mask].cpu().numpy()
    yp = pred[data.test_mask].cpu().numpy()
    labels = np.arange(len(encoder.classes_))
    p = precision_score(yt, yp, labels=labels, average=None, zero_division=0)
    r = recall_score(yt, yp, labels=labels, average=None, zero_division=0)
    f = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels)
    print(f"\nHGPNF TEST PER-CLASS METRICS — SEED {run}")
    print(f"{'Class':<47} {'Support':>7} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for i, name in enumerate(encoder.classes_):
        print(f"{name:<47} {int((yt==i).sum()):>7} {p[i]:>10.4f} {r[i]:>10.4f} {f[i]:>10.4f}")
    print("\nHGPNF TEST CONFUSION MATRIX (rows=true, columns=predicted)")
    print("Label indices:", "; ".join(f"{i}={name}" for i, name in enumerate(encoder.classes_)))
    print(cm)
    return p, r, f, cm



def run_benchmark(csv_path=None, split_mode="random", diagnostic_seed=42):
    raw_x, y, node_type, label_encoder, node_type_encoder, timestamps = load_data(csv_path)
    print("\nEVALUATION MODE:", split_mode)
    print("Dataset:", csv_path or "selected interactively")
    print("NOTE: Similarity graph uses all nodes (transductive setting).")
    print("NOTE: Relations are telemetry-distance terciles, NOT protocol-defined edges.")
    print("NOTE: Baselines use shared fixed training settings, not per-model grid search.")
    print("NOTE: Chronological mode does not guarantee event independence or eliminate transductive edges.")
    in_dim = raw_x.size(1)
    out_dim = len(label_encoder.classes_)
    num_node_types = len(node_type_encoder.classes_)
    num_relations = 3
    per_seed_results = []
    results = []
    # Identical masks, preprocessing, and graph for every method within each seed.
    datasets = {}
    for seed in SEEDS:
        set_seed(seed)
        masks = (chronological_masks(timestamps, len(y)) if split_mode == "chronological"
                 else create_masks(y, seed))
        train_mask, val_mask, test_mask = masks
        x = prepare_features(raw_x, train_mask)
        edge_index, edge_type, _ = build_relation_graph(x)
        data = Data(x=x, y=y, node_type=node_type, edge_index=edge_index,
                    edge_type=edge_type, train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask).to(DEVICE)
        datasets[seed] = data
        if seed == SEEDS[0]:
            print_class_distribution(y, label_encoder, masks)
    for model_name in MODEL_ORDER:
        print("\n" + "="*95 + "\nMODEL:", model_name, "\n" + "="*95)
        micro_scores, macro_scores, precision_scores = [], [], []
        for run, seed in enumerate(SEEDS, start=1):
            set_seed(seed)
            data = datasets[seed]
            model = build_model(model_name, in_dim, HIDDEN_DIM, out_dim,
                                num_node_types, num_relations).to(DEVICE)
            micro, macro, precision = train_model(model, model_name, data)
            micro_scores.append(micro)
            macro_scores.append(macro)
            precision_scores.append(precision)
            per_seed_results.append({"Model":model_name, "Seed":seed,
                                     "Micro-F1":micro, "Macro-F1":macro,
                                     "Macro-Precision":precision})
            print(f"Run {run}/{RUNS} seed={seed}: Micro-F1={micro:.4f}, "
                  f"Macro-F1={macro:.4f}, Macro-Precision={precision:.4f}")
            if model_name == "HGPNF (Ours)" and seed == diagnostic_seed:
                print_hgpnf_diagnostics(model, data, label_encoder, seed)
        results.append({"Model":model_name,
                        "Micro-F1 Mean":np.mean(micro_scores),
                        "Micro-F1 Std":np.std(micro_scores, ddof=1),
                        "Macro-F1 Mean":np.mean(macro_scores),
                        "Macro-F1 Std":np.std(macro_scores, ddof=1),
                        "Macro-Precision Mean":np.mean(precision_scores),
                        "Macro-Precision Std":np.std(precision_scores, ddof=1)})
    results_df = pd.DataFrame(results)
    per_seed_df = pd.DataFrame(per_seed_results)
    print("\nFINAL RESULTS — mean ± sample SD over 5 shared seeds")
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<20} | Macro-F1 {row['Macro-F1 Mean']:.4f} ± {row['Macro-F1 Std']:.4f}"
              f" | Micro-F1 {row['Micro-F1 Mean']:.4f} ± {row['Micro-F1 Std']:.4f}"
              f" | Macro-P {row['Macro-Precision Mean']:.4f} ± {row['Macro-Precision Std']:.4f}")
    print_latex_rows(results_df)
    print("\nNo CSV, PDF or PNG files are created by this script.")
    return results_df, per_seed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HGPNF five-seed reviewer-revision benchmark")
    parser.add_argument("--csv", default=None, help="Dataset CSV (omit for Tk file chooser)")
    parser.add_argument("--split", choices=["random", "chronological"], default="random",
                        help="Chronological mode requires complete timestamps")
    parser.add_argument("--diagnostic-seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(args.csv, args.split, args.diagnostic_seed)
