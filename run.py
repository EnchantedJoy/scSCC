from preprocess import *
from train import *
import os
from sklearn.manifold import TSNE
import random
import pandas as pd
import time
from sklearn import metrics
from parse import createParser
import warnings

warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = createParser()

    curSavePath = os.path.join(os.getcwd(), args.save_path, args.dataset)
    curEmbdPath = os.path.join(os.getcwd(), args.embd_path, args.dataset)
    # make sure saving path exists
    if not os.path.exists(curSavePath):
        os.makedirs(curSavePath)
    if not os.path.exists(curEmbdPath):
        os.makedirs(curEmbdPath)
    with h5py.File("./XYData/" + args.dataset + ".h5", "r") as fr:
        tempRawData = fr["X"][:]
        if args.dataset.split("_")[-1] == "MM":
            tempADTData = fr["X1"][:]
            tempRawData = np.concatenate([tempRawData, tempADTData], axis=1)
    datasetShape = tempRawData.shape

    del tempRawData

    # ++ print dataset information ++
    print("-" * 50 + "->")
    print(f"dataset: {args.dataset}")
    print(f"dataset's shape: {datasetShape}")
    print(f"HVG selected: {args.highly_genes}")
    X_adt = None

    ###  + all the data is h5py with "X" and "Y", hvg: "XRaw", "XPre", "Y"
    hvgDataPath = "./XYData/" + args.dataset + "HVG" + ".h5"
    if not os.path.exists(hvgDataPath):
        data_h5 = h5py.File("./XYData/" + args.dataset + ".h5", 'r')
        X = np.array(data_h5["X"])
        Y = np.array(data_h5["Y"]).reshape([
            -1,
        ])
        X = np.ceil(X).astype(np.int32)
        count_X = X
        X = X.astype(np.float32)
        adata = sc.AnnData(X)
        adata = normalize_(adata,
                           ifcopy=True,
                           highly_genes=args.highly_genes,
                           filter_min_counts=True,
                           size_factors=True,
                           normalize_input=True,
                           logtrans_input=True)
        X = adata.X.astype(np.float32)  ## + get cell-hvg expression
        high_variable = np.array(adata.var.highly_variable.index,
                                 dtype=np.int32)
        count_X = count_X[:, high_variable]
        ## - save hvg data
        with h5py.File(hvgDataPath, 'w') as fwHVG:
            fwHVG.create_dataset("XRaw", data=count_X, compression="gzip")
            fwHVG.create_dataset("XPre", data=X, compression="gzip")
            fwHVG.create_dataset("Y", data=Y, compression="gzip")
            if args.dataset.split("_")[-1] == "MM":
                X_adt = np.array(data_h5["X1"]).astype(np.float32)
                adata_adt = sc.AnnData(X_adt)
                adata_adt = normalize_(adata_adt,
                                       ifcopy=True,
                                       filter_min_counts=True,
                                       size_factors=True,
                                       normalize_input=True,
                                       logtrans_input=True)
                X_adt = adata_adt.X.astype(np.float32)
                X = np.concatenate([X, X_adt], axis=1)
                Y_idx = data_h5["Y_idx"][:].reshape([
                    -1,
                ])
                args.y_idx = Y_idx
                args.protein_dim = X_adt.shape[1]
                fwHVG.create_dataset("XADT", data=X_adt, compression="gzip")
                fwHVG.create_dataset("Y_idx", data=Y_idx, compression="gzip")
        data_h5.close()
    else:
        data_h5 = h5py.File(hvgDataPath, 'r')
        count_X = np.array(data_h5["XRaw"])
        X = np.array(data_h5["XPre"])
        Y = np.array(data_h5["Y"]).reshape([
            -1,
        ])
        if args.dataset.split("_")[-1] == "MM":
            X_adt = np.array(data_h5["XADT"])
            Y_idx = np.array(data_h5["Y_idx"])
            args.y_idx = Y_idx
            args.protein_dim = X_adt.shape[1]
            X = np.concatenate([X, X_adt], axis=1)
        data_h5.close()

    # - select kappa, alpha and epochs
    givenKappa = args.kappa
    if datasetShape[0] < 10000 and args.dataset.split("_")[-1] != "MM":
        args.kappa = 0.01
    else:
        args.kappa = 1.
    if givenKappa != -1:
        args.kappa = givenKappa + 0.0
    if givenKappa == 0:
        args.kappa = 0.0
        print("training without instance loss!")
    # == print hyperparameters here ==
    print(f"kappa: {args.kappa}, alpha: {args.alpha}")
    print(f"pretrain epochs: {args.preEpochs}, total epochs: {args.epochs}")

    cluster_number = int(max(Y) - min(Y) + 1)
    if args.numClasses != 0:
        cluster_number = args.numClasses

    saveTsneEmbd = False   ###  save embedding or not

    ## = test case:
    indicesDataFrame = pd.DataFrame({
        "dataset": [],
        "seed": [],
        "kmeansARI": [],
        "kmeansNMI": [],
        "ARI": [],
        "NMI": [],
        "epoch": [],
        "time": []
    })
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if args.seed != -1:
        seeds = [args.seed]
    for seed in seeds:
        print(f"=== current seed: {seed} ===")
        results = {}
        setup_seed(seed)
        startTime = time.time()
        features, best_model = train(X,
                                     cluster_number,
                                     lab_full=Y,
                                     args=args,
                                     protein_data=X_adt)
        endTime = time.time()
        runningTime = endTime - startTime
        with h5py.File(curEmbdPath + "/embd.h5", "r") as fr:
            features = fr["Z"][:]
            pred_labels = fr["YPre"][:].flatten()
            if args.dataset.split(
                    "_")[-1] == "MM" and len(pred_labels) != len(Y):
                features = features[Y_idx]
                pred_labels = pred_labels[Y_idx]
        if args.save_pred:
            results[f"features"] = features
            results[f"max_epoch"] = best_model
        res_indices = cluster_embedding(features,
                                        cluster_number,
                                        Y,
                                        save_pred=args.save_pred)
        results = {**results, **res_indices, "dataset: ": args.dataset}
        kmeansARI = res_indices["ari"]
        kmeansNMI = res_indices["nmi"]
        predARI = np.round(metrics.adjusted_rand_score(Y, pred_labels), 4)
        predNMI = np.round(metrics.normalized_mutual_info_score(Y, pred_labels),
                           4)
        kmeansPredLabel = res_indices["pred"]
        print("----" * 10)
        print(
            f"seed: {seed}, kmeansARI: {kmeansARI}, kmeansNMI: {kmeansNMI}, ARI: {predARI}, NMI: {predNMI}, epoch: {best_model}, time: {runningTime}"
        )
        indicesDataFrame = indicesDataFrame.append(
            {
                "dataset": args.dataset,
                "seed": seed,
                "kmeansARI": kmeansARI,
                "kmeansNMI": kmeansNMI,
                "ARI": predARI,
                "NMI": predNMI,
                "epoch": best_model,
                "time": runningTime
            },
            ignore_index=True)
        if (seed == 0 or seed == args.seed) and saveTsneEmbd == True:
            print("========= save embedding for visualization ===========")
            tsne = TSNE(n_components=2).fit_transform(features)
            # save data
            if not os.path.exists("./scSCCRes"):
                os.makedirs("./scSCCRes")
            np.savez("./scSCCRes/" + args.dataset + ".npz",
                     TSNE=tsne,
                     kmeansARI=kmeansARI,
                     kmeansNMI=kmeansNMI,
                     predARI=predARI,
                     predNMI=predNMI,
                     classNum=cluster_number,
                     KPredLabel=kmeansPredLabel,
                     predLabel=pred_labels,
                     TrueLabel=Y)

        # print(results)

    ## -save indices
    if not os.path.exists("./scSCCRes"):
        os.makedirs("./scSCCRes")
    if not os.path.exists("./scSCCResAll"):
        os.makedirs("./scSCCResAll")
    if args.seed == -1:
        indicesDataFrame.to_csv("./scSCCResAll/" + args.dataset + f"_idx_{args.kappa}_nclasses_{args.numClasses}.csv",
                                index=False)
    else:
        indicesDataFrame.to_csv("./scSCCRes/" + args.dataset + str(args.seed) + f"_idx_{args.kappa}.csv",
                                index=False)
    print(indicesDataFrame)


if __name__ == "__main__":
    main()
