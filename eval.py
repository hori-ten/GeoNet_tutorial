import torch
import os
import numpy as np
import argparse

from model.model import Encoder
from data_loader.load_images import ImageList
import data_loader.transforms as transforms


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MemSAC')

    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataset', type=str, nargs='?', default='c', help="target dataset")
    parser.add_argument('--target', type=str, nargs='?', default='c', help="target domain")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size should be samples * classes")
    parser.add_argument('--nClasses', type=int, help="#Classes")
    parser.add_argument('--checkpoint' , type=str, help="Checkpoint to load from.")
    parser.add_argument('--multi_gpu', type=int, default=0, help="use dataparallel if 1")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--resnet', default="resnet50", help="Resnet backbone")

    args = parser.parse_args() 



    if args.dataset == "GeoImNet":
        file_path = {
            "asia": "./data_files/GeoImNet/asia_test.txt",
            "usa": "./data_files/GeoImNet/usa_test.txt"
            }

    elif args.dataset == "GeoPlaces":
        file_path = {
            "asia": "./data_files/GeoPlaces/asia_test.txt",
            "usa": "./data_files/GeoPlaces/usa_test.txt"
            }

    else:
        raise NotImplementedError


    dataset_test = file_path[args.target]
    print("Target" , args.target)

    dataset_loaders = {}

    dataset_list = ImageList(args.data_dir, open(dataset_test).readlines(), transform=transforms.image_test(resize_size=256, crop_size=224))
    print("Size of target dataset:" , len(dataset_list))
    dataset_loaders["test"] = torch.utils.data.DataLoader(dataset_list, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=16, drop_last=False)

    # network construction
    print(args.nClasses)
    base_network = Encoder(args.resnet, 256, args.nClasses).cuda()
        
    accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    saved_state_dict = torch.load(args.checkpoint)
    base_network.load_state_dict(saved_state_dict, strict=True)
    base_network.eval()
    start_test = True
    iter_test = iter(dataset_loaders["test"])

    conf_matrix = torch.zeros(args.nClasses, 3)  # TP, FP, FN


    with torch.no_grad():
        for i in range(len(dataset_loaders['test'])):
            print("{0}/{1}".format(i,len(dataset_loaders['test'])) , end="\r")
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()

            # calculate Top-1 accuracy
            _, outputs = base_network(inputs)
            predictions = outputs.argmax(1)
            correct = torch.sum((predictions == labels).float())
            accuracy.update(correct, len(outputs))

            # calculate Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            top5_correct = torch.sum((top5_pred == labels.view(1, -1).expand_as(top5_pred)).float())
            top5_accuracy.update(top5_correct, len(outputs))


            # update confusion matrix
            for i in range(len(outputs)):
                if labels[i] == predictions[i]:
                    conf_matrix[labels[i], 0] += 1
                else:
                    conf_matrix[predictions[i], 1] += 1
                    conf_matrix[labels[i], 2] += 1

            # calculate macro-F1
            macro_f1 = 0
            for i in range(args.nClasses):
                f1_score = (2 * conf_matrix[i, 0]) / (2 * conf_matrix[i, 0] + conf_matrix[i, 1] + conf_matrix[i, 2])
                macro_f1 += f1_score
            macro_f1 /= args.nClasses

    # print Top-1 Acc
    print_str = "\nCorrect Predictions(Top-1): {}/{}".format(int(accuracy.sum), accuracy.count)
    print_str1 = '\ntest_acc(Top-1):{:.4f}'.format(accuracy.avg)
    print(print_str + print_str1)

    # print Top-5 Acc
    print_str3 = "\nCorrect Predictions(Top-5): {}/{}".format(int(top5_accuracy.sum), top5_accuracy.count)
    print_str4 = '\ntest_acc(Top-5):{:.4f}'.format(top5_accuracy.avg)
    print(print_str3 + print_str4)

    # print macro-F1
    print(macro_f1)
