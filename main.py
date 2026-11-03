
import json
import os
import random
import string
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn

from Model.models import MMFL
from Loss.loss_fn import Loss_fn
from utils import get_data, get_auc


# --------------------------------------------------
# DEVICE CONFIGURATION: APPLE SILICON / CUDA / CPU
# --------------------------------------------------

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


# --------------------------------------------------
# REPRODUCIBILITY
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
# DATASET PATHS
# --------------------------------------------------

def get_dataset_paths(root_path, dataset, task):

    if dataset == "TADPOLE":
        data_path = os.path.join(
            root_path,
            "DATASET",
            "TADPOLE",
            "ADNI_processed_standard_data.csv"
        )

        dict_path = os.path.join(
            root_path,
            "DATASET",
            "TADPOLE",
            "ADNI_modal_feat_dict.npy"
        )

    elif dataset == "ADNI":
        data_path = os.path.join(
            root_path,
            "DATASET",
            "ADNI",
            f"Select_NEW_{task}__ADNI_data.csv"
        )

        dict_path = os.path.join(
            root_path,
            "DATASET",
            "ADNI",
            f"Select_NEW_{task}__ADNI_modal_feat_dict.npy"
        )

    elif dataset == "ABIDE":
        data_path = os.path.join(
            root_path,
            "DATASET",
            "ABIDE",
            "ABIDE_processed_data_modal.csv"
        )

        dict_path = os.path.join(
            root_path,
            "DATASET",
            "ABIDE",
            "ABIDE_modal_feat_dict.npy"
        )

    elif dataset == "ABIDE-5":
        data_path = os.path.join(
            root_path,
            "DATASET",
            "ABIDE-5",
            "ABIDE_processed_data_modal.csv"
        )

        dict_path = os.path.join(
            root_path,
            "DATASET",
            "ABIDE-5",
            "ABIDE_modal_feat_dict.npy"
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for path in [data_path, dict_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required dataset file not found: {path}"
            )

    return data_path, dict_path


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MMFL Multimodal Learning"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="TADPOLE"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="AD_CN_SMCI"
    )

    parser.add_argument(
        "--knn",
        type=int,
        default=7
    )

    args = parser.parse_args()

    DATA_SET = args.dataset
    Task = args.task
    k_nearest_neighobrs = args.knn

    print("\n========== MMFL CONFIGURATION ==========")
    print("Dataset:", DATA_SET)
    print("Task:", Task)
    print("KNN:", k_nearest_neighobrs)
    print("========================================\n")

    # Original hyperparameters
    Save_Model = False
    CALR = True

    weight_decay = 0.0009
    EPOCH = 500

    Shuffle = True
    n_splits = 10
    Random_seed = 0

    if CALR:
        lr = 0.01
        Lr_Min = 0.0001
    else:
        lr = 0.001

    hidden_num = 32
    Dim_emb = 32
    dropout_date = 0.6

    K = 1
    GCN_Model = "Cheby_GCN"

    # Device and seed
    SET_Random(Random_seed)
    device = get_device()

    Root_path = os.path.dirname(
        os.path.abspath(__file__)
    )

    # Dataset paths
    Feature_Data_path, Feature_dict_path = get_dataset_paths(
        Root_path,
        DATA_SET,
        Task
    )

    print("Feature data:", Feature_Data_path)
    print("Feature dictionary:", Feature_dict_path)

    # Load data
    print("\nLoading dataset...")

    Data_X, Data_Y, Adj, Mask_list, in_channels, Class_num, train_num, test_num, Feature_index_list, label_weight, Feature_Data_index = get_data(
        Feature_Data_path,
        Feature_dict_path,
        Device=device,
        k_nearest_neighobrs=k_nearest_neighobrs,
        rm_common_neighbors=2,
        Random_seed=Random_seed
    )

    print("\nDataset loaded successfully.")
    print("Number of classes:", Class_num)
    print("Number of splits:", len(Mask_list))
    print("Input channels:", in_channels)

    All_ACC_list = []
    All_AUC_list = []

    # --------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------

    for splits_num, (train_mask, test_mask) in enumerate(Mask_list):

        print(f"\nStarting split {splits_num + 1}")

        model = MMFL(
            Adj_=Adj,
            out_channels=Class_num,
            feature_list=Feature_index_list,
            Device=device,
            GCN_Model=GCN_Model,
            hidden=hidden_num,
            K=K,
            P=dropout_date,
            Dim_emb=Dim_emb
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        Loss_F = nn.CrossEntropyLoss(
            weight=label_weight
        ).to(device)

        if CALR:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,
                eta_min=Lr_Min
            )

        Best_ACC = 0.0
        Best_AUC = 0.0
        Best_model_state = None

        with tqdm(range(EPOCH), total=EPOCH) as pbar:

            for epoch in pbar:

                # Training
                model.train()
                optimizer.zero_grad()

                out, Adj_ = model(Data_X)

                if torch.isnan(out).any() or torch.isnan(Adj_).any():
                    raise RuntimeError(
                        f"NaN detected during training, split {splits_num + 1}, epoch {epoch + 1}"
                    )

                loss = Loss_fn(
                    Data_X,
                    Adj_,
                    out,
                    Data_Y,
                    train_mask,
                    Loss_F,
                    label_weight,
                    device
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.8,
                    norm_type=2
                )

                optimizer.step()

                _, train_labels_predict = out[train_mask].max(-1)

                train_acc = (
                    Data_Y[train_mask] == train_labels_predict
                ).sum().item()

                # Evaluation
                model.eval()

                with torch.no_grad():

                    test_output, Adj_ = model(Data_X)

                    if torch.isnan(test_output).any() or torch.isnan(Adj_).any():
                        raise RuntimeError(
                            f"NaN detected during evaluation, split {splits_num + 1}, epoch {epoch + 1}"
                        )

                _, test_labels_predict = test_output[test_mask].max(-1)

                test_acc = (
                    Data_Y[test_mask] == test_labels_predict
                ).sum().item()

                if CALR:
                    scheduler.step()

                Test_ACC = test_acc / test_num[splits_num]
                Train_ACC = train_acc / train_num[splits_num]

                Test_AUC = get_auc(
                    y_true=F.one_hot(
                        Data_Y[test_mask].cpu(),
                        num_classes=Class_num
                    ),
                    y_pred=test_output[test_mask].cpu()
                )

                if Test_ACC > Best_ACC:
                    Best_ACC = Test_ACC

                    if Test_AUC > Best_AUC:
                        Best_AUC = Test_AUC

                        if Save_Model:
                            Best_model_state = {
                                key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()
                            }

                pbar.set_description(
                    f"Split: {splits_num + 1} "
                    f"Epoch: {epoch + 1} "
                    f"Loss: {loss.item():.4f} "
                    f"Train ACC: {Train_ACC:.4f} "
                    f"Test ACC: {Test_ACC:.4f}"
                )

        All_ACC_list.append(Best_ACC)
        All_AUC_list.append(Best_AUC)

        print(
            f"Split {splits_num + 1} "
            f"Best ACC: {Best_ACC:.4f} "
            f"Best AUC: {Best_AUC:.4f}"
        )

        if Save_Model and Best_model_state is not None:

            model_dir = os.path.join(
                Root_path,
                "RESULT",
                "Model_Weights"
            )

            os.makedirs(model_dir, exist_ok=True)

            torch.save(
                Best_model_state,
                os.path.join(
                    model_dir,
                    f"{DATA_SET}_{Task}_Model_Splits_{splits_num}.pth"
                )
            )

    # --------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------

    ACC_mean = np.mean(All_ACC_list)
    AUC_mean = np.mean(All_AUC_list)

    ACC_std = np.std(All_ACC_list)
    AUC_std = np.std(All_AUC_list)

    print("\n========== FINAL RESULTS ==========")
    print(f"ACC: Mean: {ACC_mean:.4f} Std: {ACC_std:.4f}")
    print(f"AUC: Mean: {AUC_mean:.4f} Std: {AUC_std:.4f}")
    print("===================================\n")

    result = {
        "ACC": {
            "Mean": float(ACC_mean),
            "Std": float(ACC_std)
        },
        "AUC": {
            "Mean": float(AUC_mean),
            "Std": float(AUC_std)
        }
    }

    letters = string.ascii_letters + string.digits
    result_id = "".join(
        random.choice(letters) for _ in range(10)
    )

    Save_History_path = os.path.join(
        Root_path,
        "RESULT",
        DATA_SET,
        Task,
        f"{result_id}.json"
    )

    os.makedirs(
        os.path.dirname(Save_History_path),
        exist_ok=True
    )

    with open(Save_History_path, "w") as file:
        json.dump(result, file, indent=4)

    print("Results saved to:", Save_History_path)
