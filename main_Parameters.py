
import os
import random
import numpy as np
import torch
from torch import nn

from Model.models import *
from utils import get_data


# --------------------------------------------------
# DEVICE CONFIGURATION
# --------------------------------------------------

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    return device


# --------------------------------------------------
# RANDOM SEED
# --------------------------------------------------

def SET_Random(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

Random_seed = 0

hidden_num = 32
Dim_emb = 32
dropout_date = 0.6
K = 1

GCN_Model = "Cheby_GCN"

SET_Random(Random_seed)
device = get_device()


# --------------------------------------------------
# DATASET PATHS
# --------------------------------------------------

Root_path = os.path.dirname(os.path.abspath(__file__))

DATA_SET = "TADPOLE"
Task = "AD_CN_SMCI"

if DATA_SET == "TADPOLE":

    Feature_Data_path = os.path.join(
        Root_path,
        "DATASET",
        "TADPOLE",
        f"{Task}_ADNI_processed_standard_data.csv"
    )

    Feature_dict_path = os.path.join(
        Root_path,
        "DATASET",
        "TADPOLE",
        f"{Task}_ADNI_modal_feat_dict.npy"
    )

else:
    raise ValueError("This configuration is currently for TADPOLE only.")


# Verify files
for path in [Feature_Data_path, Feature_dict_path]:

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

Data_X, Data_Y, Adj, Mask_list, in_channels, Class_num, train_num, test_num, Feature_index_list, label_weight, Feature_Data_index = get_data(
    Feature_Data_path,
    Feature_dict_path,
    Device=device,
    k_nearest_neighobrs=7,
    rm_common_neighbors=2,
    Random_seed=Random_seed
)


# --------------------------------------------------
# MULTIMODAL STATE SPACE NETWORK
# --------------------------------------------------

M_SSNs_Encoder = MultimodalStateSpaceNetworks(
    Feature_index_list,
    device,
    dim_embedding=Dim_emb
).to(device)


M_SSNs_Encoder_p = nn.Sequential(
    nn.LayerNorm(Dim_emb),
    nn.Dropout(p=dropout_date),
    nn.LeakyReLU()
).to(device)


M_SSNs_Decoder = nn.Sequential(
    nn.Linear(
        in_features=Dim_emb * len(Feature_index_list),
        out_features=Dim_emb
    ),
    nn.LayerNorm(Dim_emb),
    nn.LeakyReLU()
).to(device)


# --------------------------------------------------
# GRAPH STRUCTURE LEARNING
# --------------------------------------------------

M_GSL = MultimodalGraphStructureLearning(
    Adj,
    Feature_index_list,
    device,
    Dim_emb,
    rate=0
).to(device)


M_GF = MultimodalGraphFusion(
    Adj.shape,
    device
).to(device)


# --------------------------------------------------
# GRAPH CONVOLUTION NETWORK
# --------------------------------------------------

if GCN_Model == "GCN":

    GCN = GCN(
        Dim_emb=Dim_emb,
        hidden=hidden_num,
        out_channels=Class_num,
        P=dropout_date
    ).to(device)

elif GCN_Model == "Cheby_GCN":

    GCN = Cheb_GCN(
        Dim_emb=Dim_emb,
        hidden=hidden_num,
        out_channels=Class_num,
        P=dropout_date,
        K=K
    ).to(device)

else:
    raise ValueError(f"Unknown GCN model: {GCN_Model}")


# --------------------------------------------------
# PARAMETER COUNT
# --------------------------------------------------

M_SSNs_params = sum(
    p.numel() for p in M_SSNs_Encoder.parameters()
)

M_SSNs_p_params = sum(
    p.numel() for p in M_SSNs_Encoder_p.parameters()
)

M_SSNs_Decoder_params = sum(
    p.numel() for p in M_SSNs_Decoder.parameters()
)

n = Adj.shape[0]

M_GSL_params = (
    sum(p.numel() for p in M_GSL.parameters())
    - ((n * n - n) // 2) * len(Feature_index_list)
)

M_GF_params = sum(
    p.numel() for p in M_GF.parameters()
)

GCN_params = sum(
    p.numel() for p in GCN.parameters()
)


M_SSNs_params = (
    M_SSNs_params
    + M_SSNs_p_params
    + M_SSNs_Decoder_params
)


print("\n========== MODEL PARAMETERS ==========")
print("M_SSNs_params:", M_SSNs_params)
print("M_GSL_params:", M_GSL_params)
print("M_GF_params:", M_GF_params)
print("GCN_params:", GCN_params)

print(
    "Total:",
    M_SSNs_params + M_GSL_params + M_GF_params + GCN_params
)

print("======================================")
