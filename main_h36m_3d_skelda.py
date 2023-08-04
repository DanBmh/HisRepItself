import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from model import AttModel
from utils import log, util
from utils.opt import Options

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

datapath_save_out = "/datasets/preprocessed/human36m/{}_forecast_kppspose.json"
config = {
    "item_step": 2,
    "window_step": 2,
    "input_n": 50,
    "output_n": 25,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        "nose",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        "shoulder_middle",
    ],
}


# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device):
    sequences = utils_pipeline.make_input_sequence(batch, split, "gt-gt")

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    sequences = torch.from_numpy(sequences).to(device)

    return sequences


# ==================================================================================================


def main(opt):
    lr_now = opt.lr_now
    start_epoch = 1
    # opt.is_eval = True
    print(">>> create models")
    in_features = opt.in_features  # 66
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

    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now
    )
    print(
        ">>> total params: {:.2f}M".format(
            sum(p.numel() for p in net_pred.parameters()) / 1000000.0
        )
    )

    if opt.is_load or opt.is_eval:
        model_path_len = "./{}/ckpt_best.pth.tar".format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt["epoch"] + 1
        err_best = ckpt["err"]
        lr_now = ckpt["lr"]
        net_pred.load_state_dict(ckpt["state_dict"])
        # net.load_state_dict(ckpt)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # lr_now = util.lr_decay_mine(optimizer, lr_now, 0.2)
        print(
            ">>> ckpt len loaded (epoch: {} | err: {})".format(
                ckpt["epoch"], ckpt["err"]
            )
        )

    print(">>> loading datasets")

    # Load preprocessed datasets
    print("Loading datasets ...")
    dataset_train, dlen_train = utils_pipeline.load_dataset(
        datapath_save_out, "train", config
    )
    esplit = "test" if "mocap" in datapath_save_out else "eval"

    dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        datapath_save_out, esplit, config
    )

    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print(">>> training epoch: {:d}".format(epo))

            label_gen_train = utils_pipeline.create_labels_generator(
                dataset_train["sequences"], config
            )
            label_gen_eval = utils_pipeline.create_labels_generator(
                dataset_eval["sequences"], config
            )

            ret_train = run_model(
                net_pred,
                optimizer,
                is_train=0,
                data_loader=label_gen_train,
                epo=epo,
                opt=opt,
                dlen=dlen_train,
            )
            print("train error: {:.3f}".format(ret_train["m_p3d_h36"]))
            ret_valid = run_model(
                net_pred,
                is_train=1,
                data_loader=label_gen_eval,
                opt=opt,
                epo=epo,
                dlen=dlen_eval,
            )
            print("validation error: {:.3f}".format(ret_valid["m_p3d_h36"]))

            ret_log = np.array([epo, lr_now])
            head = np.array(["epoch", "lr"])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ["valid_" + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            if ret_valid["m_p3d_h36"] < err_best:
                err_best = ret_valid["m_p3d_h36"]
                is_best = True
            log.save_ckpt(
                {
                    "epoch": epo,
                    "lr": lr_now,
                    "err": ret_valid["m_p3d_h36"],
                    "state_dict": net_pred.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                opt=opt,
            )


# ==================================================================================================


def run_model(
    net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None, dlen=0
):
    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    itera = 1

    nbatch = opt.batch_size
    for batch in tqdm.tqdm(
        utils_pipeline.batch_iterate(data_loader, batch_size=nbatch),
        total=int(dlen / nbatch),
    ):

        batch_size = nbatch
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue

        n += batch_size

        sequences_train = prepare_sequences(batch, nbatch, "input", device)
        sequences_gt = prepare_sequences(batch, nbatch, "target", device)

        seq_all = torch.cat([sequences_train, sequences_gt], dim=1)
        # seq_all = seq_all.float().cuda()
        # print(seq_all.shape)

        p3d_out_all = net_pred(seq_all, input_n=in_n, output_n=out_n, itera=itera)

        p3d_out = p3d_out_all[:, seq_in:, 0]
        p3d_out = p3d_out.reshape([-1, out_n, p3d_out_all.shape[-1] // 3, 3])

        p3d_h36 = seq_all.reshape([-1, in_n + out_n, seq_all.shape[-1] // 3, 3])
        p3d_sup = p3d_h36[:, -out_n - seq_in :, :, :]

        p3d_out_all = p3d_out_all.reshape(
            [batch_size, seq_in + out_n, itera, p3d_out_all.shape[-1] // 3, 3]
        )

        # 2d joint loss:
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))
            loss_all = loss_p3d
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if (
            is_train <= 1
        ):  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(
                torch.norm(p3d_h36[:, in_n : in_n + out_n] - p3d_out, dim=3)
            )
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(
                torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0
            )
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]
    return ret


# ==================================================================================================

if __name__ == "__main__":
    option = Options().parse()

    stime = time.time()
    main(option)

    ftime = time.time()
    print("Training took {} seconds".format(int(ftime - stime)))
