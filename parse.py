import argparse

def createParser():
    parser = argparse.ArgumentParser(description="Training netAE")
    # Training settings:
    parser.add_argument("-epath",
                        "--embd-path",
                        default="Embd",
                        help="path to the dataset folder")
    parser.add_argument("-spath",
                        "--save-path",
                        type=str,
                        default="save",
                        help="path to output directory")
    parser.add_argument("-ds",
                        "--dataset",
                        type=str,
                        default="QSTrachea",
                        help="name of the dataset (default: QSTrachea)")
    parser.add_argument("-hvg",
                        "--highly_genes",
                        default=2000,
                        help="dim of highly variance genes(default: 2000)")
    parser.add_argument("-se",
                        "--save-embd",
                        type=bool,
                        default=True,
                        help="saves the embedded space along with the model")

    # Model property:
    parser.add_argument("-s",
                        "--seed",
                        type=int,
                        default=0,
                        help="random seed for loading dataset (default: 0)")

    # Model parameters:
    parser.add_argument("-opt",
                        "--optim",
                        default="adam",
                        help="name of the optimizer to use (default: adam)")
    parser.add_argument("-bs",
                        "--batch-size",
                        type=int,
                        default=256,
                        help="batch size (default: 256)")

    parser.add_argument("--lr",
                        type=float,
                        default=0.01,
                        help="learning rate (default: 0.01)")

    parser.add_argument("-ed",
                        "--encoderDim",
                        type=list,
                        default=[200, 40],
                        help="encoding dimension")
    parser.add_argument("--dim",
                        type=int,
                        default=60,
                        help="latent dimension (default: 60)")
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        default=200,
                        help="number of epochs (default: 250)")
    parser.add_argument("-pe",
                        "--preEpochs",
                        type=int,
                        default=150,
                        help="number of pretrain epochs (default: 200)")

    # Loss parameters
    parser.add_argument("-dr",
                        "--dropoutRate",
                        type=float,
                        default=0.9,
                        help="dropout rate of augmentation (default: 0.9).")

    parser.add_argument("-eps",
                        "--epsilon",
                        type=float,
                        default=0.05,
                        help="regularization of Sinkhorn-Knopp algorithm")
    parser.add_argument("-sinkIter",
                        "--sinkhorn_iterations",
                        type=float,
                        default=3,
                        help="regularization of Sinkhorn-Knopp algorithm")
    parser.add_argument("-al",
                        "--alpha",
                        type=float,
                        default=0.1,
                        help="ratio for noise (default: 0.1)")
    parser.add_argument("-kp",
                        "--kappa",
                        type=float,
                        default=-1,
                        help="weight for instance loss (default: auto select)")
    parser.add_argument("-act",
                        "--activation",
                        type=str,
                        default="relu",
                        help="weight for instance loss")

    parser.add_argument("-ld",
                        "--lambd",
                        type=float,
                        default=1,
                        help="weight for swav loss")
    parser.add_argument("-insTemp",
                        "--instance_temperature",
                        type=float,
                        default=.1,
                        help="temperature of instance loss (default: 0.1)")
    parser.add_argument("-swTemp",
                        "--swav_temperature",
                        type=float,
                        default=.1,
                        help="temperature of swav loss (default: 0.1)")
    parser.add_argument("-hd", "--headDim", type=float, default=40)
    parser.add_argument("-sp", "--save-pred", type=bool, default=True)
    parser.add_argument("-nc", "--numClasses", type=int, default=0)
    parser.add_argument("-td", "--threshold", type=float, default=0.94)

    args = parser.parse_args()
    return args
