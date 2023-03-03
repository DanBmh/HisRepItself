import sys
import time

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt

from model import AttModel
from utils.opt import Options

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

# datapath_save_out = "/datasets/tmp/human36m/{}_forecast_samples.json"
# config = {
#     "item_step": 2,
#     "window_step": 2,
#     "input_n": 50,
#     "output_n": 25,
#     "select_joints": [
#         "hip_middle",
#         "hip_right",
#         "knee_right",
#         "ankle_right",
#         # "middlefoot_right",
#         # "forefoot_right",
#         "hip_left",
#         "knee_left",
#         "ankle_left",
#         # "middlefoot_left",
#         # "forefoot_left",
#         # "spine_upper",
#         # "neck",
#         "nose",
#         # "head",
#         "shoulder_left",
#         "elbow_left",
#         "wrist_left",
#         # "hand_left",
#         # "thumb_left",
#         "shoulder_right",
#         "elbow_right",
#         "wrist_right",
#         # "hand_right",
#         # "thumb_right",
#         "shoulder_middle",
#     ],
# }

datapath_save_out = "/datasets/preprocessed/mocap/{}_forecast_samples.json"
config = {
    "item_step": 2,
    "window_step": 2,
    # "input_n": 30,
    # "output_n": 15,
    "input_n": 90,
    "output_n": 45,
    "select_joints": [
        "hip_middle",
        # "spine_lower",
        "hip_right",
        "knee_right",
        "ankle_right",
        # "middlefoot_right",
        # "forefoot_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        # "middlefoot_left",
        # "forefoot_left",
        # "spine2",
        # "spine3",
        # "spine_upper",
        # "neck",
        # "head_lower",
        "head_upper",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        # "hand_right1",
        # "hand_right2",
        # "hand_right3",
        # "hand_right4",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        # "hand_left1",
        # "hand_left2",
        # "hand_left3",
        # "hand_left4"
        "shoulder_middle",
    ],
}

viz_action = ""
# viz_action = "walking"

# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device):
    sequences = utils_pipeline.make_input_sequence(batch, split, "gt-gt")

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    sequences = torch.from_numpy(sequences).to(device)

    return sequences


# ==================================================================================================


def viz_joints_3d(sequences_predict, batch):
    batch = batch[0]
    vis_seq_pred = (
        sequences_predict.cpu()
        .detach()
        .numpy()
        .reshape(sequences_predict.shape[0], sequences_predict.shape[1], -1, 3)
    )[0]
    utils_pipeline.visualize_pose_trajectories(
        np.array([cs["bodies3D"][0] for cs in batch["input"]]),
        np.array([cs["bodies3D"][0] for cs in batch["target"]]),
        utils_pipeline.make_absolute_with_last_input(vis_seq_pred, batch),
        batch["joints"],
        {"room_size": [3200, 4800, 2000], "room_center": [0, 0, 1000]},
    )
    plt.show()


# ==================================================================================================


def run_test(model, opt):

    model.eval()
    action_losses = []
    itera = 1
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size

    # Load preprocessed datasets
    dataset_test, dlen = utils_pipeline.load_dataset(datapath_save_out, "test", config)
    label_gen_test = utils_pipeline.create_labels_generator(dataset_test, config)

    stime = time.time()
    frame_losses = np.zeros([out_n])
    nitems = 0

    with torch.no_grad():
        nbatch = 1
        batch_size = nbatch

        for batch in tqdm.tqdm(label_gen_test, total=dlen):

            if nbatch == 1:
                batch = [batch]

            if viz_action != "" and viz_action != batch[0]["action"]:
                continue

            nitems += nbatch
            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)
            seq_all = torch.cat([sequences_train, sequences_gt], dim=1)

            p3d_out_all = model(seq_all, input_n=in_n, output_n=out_n, itera=itera)

            sequences_predict = (
                p3d_out_all[:, seq_in:]
                .transpose(1, 2)
                .reshape([batch_size, out_n * itera, -1])[:, :out_n]
            )

            if viz_action != "":
                viz_joints_3d(sequences_predict, batch)

            pose_dim = sequences_predict.shape[-1]
            loss = torch.sqrt(
                torch.sum(
                    (
                        sequences_predict.view(nbatch, -1, pose_dim // 3, 3)
                        - sequences_gt.view(nbatch, -1, pose_dim // 3, 3)
                    )
                    ** 2,
                    dim=-1,
                )
            )
            loss = torch.sum(torch.mean(loss, dim=2), dim=0)
            frame_losses += loss.cpu().data.numpy()

    avg_losses = frame_losses / nitems
    print("Averaged frame losses in mm are:", avg_losses)

    ftime = time.time()
    print("Testing took {} seconds".format(int(ftime - stime)))


# ==================================================================================================


def main(opt):
    print(">>> create models")
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = AttModel.AttModel(
        in_features=in_features,
        kernel_size=kernel_size,
        d_model=d_model,
        num_stage=opt.num_stage,
        dct_n=opt.dct_n,
    )
    net_pred.cuda()
    model_path_len = "{}/ckpt_best.pth.tar".format(opt.ckpt)
    model_path_len = model_path_len.replace("_eval", "")
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt["state_dict"])
    print(
        ">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt["epoch"], ckpt["err"])
    )

    run_test(net_pred, opt)


# ==================================================================================================


if __name__ == "__main__":
    option = Options().parse()
    main(option)
