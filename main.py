from __future__ import print_function
import os
import argparse

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

import utils
from covi_trainer import train_covi, train_src

parser = argparse.ArgumentParser(
    description="CoVi: Contrastive Vicinal Space for Unsupervised Domain Adaptation (ECCV-22)"
)
parser.add_argument(
    "-db_path",
    type=str,
    default="/mnt/asgard2/data/kevin",
    help="path to dataset repository",
)
parser.add_argument("-source", type=str, default="amazon", help="source domain")
parser.add_argument("-target", type=str, default="dslr", help="target domain")
parser.add_argument(
    "-workers", default=12, type=int, metavar="N", help="number of data loading workers"
)
parser.add_argument("-train_source", help="train only on source", action="store_true", default=False)
parser.add_argument("-gpu", help="gpu number", type=str, default="0")
parser.add_argument("-batch_size", default=32, type=int)
parser.add_argument("-baseline_path", type=str, help="path to pre-trainined model")
parser.add_argument("-epochs", default=1, type=int, help="total training epochs")
parser.add_argument("-cmin", type=float, default=0.1, help="EMP boundary:min")
parser.add_argument("-cmax", type=float, default=0.9, help="EMP boundary:max")
parser.add_argument(
    "-swap_margin", type=float, default=0.1, help="margin for contrastive space"
)
parser.add_argument("-swap_upper", type=float, default=0.9)
parser.add_argument("-swap_lower", type=float, default=0.1)
parser.add_argument("-swap_th", default=1.0, type=float, help="confidence threshold")
parser.add_argument("-consensus_ratio", help="consensus_ratio", type=float, default=0.1)


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("Use GPU(s): {} for training".format(args.gpu))
    print(args)

    num_classes, resnet_type = utils.get_data_info(args)
    src_trainset, src_testset = utils.get_dataset(args.source, db_path=args.db_path)
    tgt_trainset, tgt_testset = utils.get_dataset(args.target, db_path=args.db_path)

    src_train_loader = torchdata.DataLoader(
        src_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    tgt_train_loader = torchdata.DataLoader(
        tgt_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    src_test_loader = torchdata.DataLoader(
        src_testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    tgt_test_loader = torchdata.DataLoader(
        tgt_testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    lr, l2_decay, momentum = utils.get_train_info()

    net, head, classifier, emp_learner = utils.get_net_info(num_classes)
    net_part1 = nn.Sequential(*(list(net.children())[:-2]))

    learnable_params = (
        list(net.parameters()) + list(head.parameters()) + list(classifier.parameters())
    )
    if not args.train_source:
        net, head, classifier = utils.load_net(net, head, classifier, load_path=args.baseline_path)
    model = nn.Sequential(net, head, classifier)

    optimizer = optim.SGD(
        learnable_params, lr=lr, momentum=momentum, weight_decay=l2_decay
    )
    optimizer_emp = optim.SGD(
        list(emp_learner.parameters()), lr=lr, momentum=momentum, weight_decay=l2_decay
    )

    entropy = utils.EntropyLoss().cuda()
    cross_entropy = nn.CrossEntropyLoss().cuda()

    if args.train_source:
        for epoch in range(args.epochs):
            train_src(
                args=args,
                src_train_loader=src_train_loader,
                optimizer=optimizer,
                model=model,
                cross_entropy=cross_entropy,
                epoch=epoch
            )
            acc_src = utils.evaluate(model, src_test_loader)
            print("* Accuracy source {:.2f}%".format(acc_src))
        
        utils.save_net(model, save_path=args.baseline_path)
        return

    best_acc_tgt = utils.evaluate(model, tgt_test_loader)
    acc_src = utils.evaluate(model, src_test_loader)
    print("* Before training: Accuracy target {:.2f}%".format(best_acc_tgt))
    print("* Before training: Accuracy source {:.2f}%".format(acc_src))

    for epoch in range(args.epochs):
        train_covi(
            args,
            src_train_loader,
            tgt_train_loader,
            optimizer,
            optimizer_emp,
            model,
            net_part1,
            emp_learner,
            entropy,
            cross_entropy,
            epoch,
        )
        acc = utils.evaluate(model, tgt_test_loader)
        acc_src = utils.evaluate(model, src_test_loader)

        if acc > best_acc_tgt:
            best_acc_tgt = max(acc, best_acc_tgt)

        print("* Latest Accuracy target {:.2f}%".format(acc))
        print("* Best Accuracy target {:.2f}%".format(best_acc_tgt))
        print("* Accuracy source {:.2f}%".format(acc_src))


if __name__ == "__main__":
    main()
