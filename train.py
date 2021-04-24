import datetime
import os
import time

import pickle
import copy
import shutil
from pathlib import Path
import tempfile

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import pandas as pd
from PIL import Image

import utils

import mlflow
import mlflow.pytorch
import optuna

def calc_log_ROC(model, data_loader, device, classes):
    #calculate roc curve, create and save matplotlib fig, send to mlflow
    
    # only tested for binary classification, not distributed mode and gpu
    model.eval()
    with torch.no_grad():
        result = torch.cuda.FloatTensor()
        targets = torch.cuda.LongTensor()
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            m = nn.Softmax(dim=1)
            output =  m(output)
            result = torch.cat((result,output),dim=0)
            targets = torch.cat((targets,target),dim=0)
            
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt

    class_of_interst = len(classes)-1

    y = targets.cpu()
    scores = result[:,class_of_interst].cpu()
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=class_of_interst)
    roc_auc = auc(fpr, tpr)
            
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        savepath = os.path.join(tmpdirname,'ROC.png')
        plt.savefig(savepath, dpi=1000)
        mlflow.log_artifact(savepath, artifact_path="ROC_Curve")

def log_data_split(args, indices, splitpoint):
    df = pd.read_csv(args.data_path)
    df_train = df.iloc[indices[:splitpoint],:]
    df_test = df.iloc[indices[splitpoint:],:]
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        df_train.to_csv(tmpdirname+"/train_set.csv", index=False)
        df_test.to_csv(tmpdirname+"/test_set.csv", index=False)
        
        mlflow.log_artifacts(tmpdirname, artifact_path="data")
        mlflow.log_artifact(args.data_path, artifact_path="data")

def reshape_classification_head(model, args, class_names):
    num_classes = len(class_names)
    if args.model == 'resnet18' or args.model == 'resnet34' or args.model == 'resnet50' or args.model == 'resnet101' or args.model == 'resnet152' or args.model == 'resnext50_32x4d' or args.model == 'resnext101_32x8d' or args.model == 'wide_resnet50_2' or args.model == 'wide_resnet101_2' or args.model == 'shufflenet_v2_x0_5' or args.model == 'shufflenet_v2_x1_0':
        num_ftrs = model.fc.in_features
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif args.model == 'squeezenet1_1':
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    elif args.model == 'densenet121' or args.model == 'densenet169' or args.model == 'densenet201' or args.model == 'densenet161':
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f'changing classification head for model "{args.model}" not supported, exiting...')
    return model
        

class CSVDataset(object):
    def __init__(self, root, path_to_csv, transforms, dataset_type='binary'):
        self.root = root
        self.transforms = transforms
        df = pd.read_csv(path_to_csv)
        
        self.imgs = list(df.iloc[:, 0])
        
        if dataset_type == 'binary':
            self.labels = list(df[df.columns[1]])
            self.classes = ['non_'+df.columns[1], df.columns[1]]
        elif dataset_type == 'multi_class':
            raise NotImplementedError(f'dataset_type "{dataset_type}" not yet implemented')
        elif dataset_type == 'multi_label':
            raise NotImplementedError(f'dataset_type "{dataset_type}" not yet implemented')
        else:
            raise ValueError(f'invalid value for dataset_type "{dataset_type}". Valid values are "binary", "multi_class" and "multi_label"')

    def __getitem__(self, idx):
        # load images and labels
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path)
        target = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, class_names, print_freq=100):
    num_classes = len(class_names)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            
            #calc acc per class
            _, preds = torch.max(output, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size) 
        # check for distributed mode, if yes, synchronize confusion matrix across all nodes
        if utils.is_dist_avail_and_initialized():
            confusion_matrix = confusion_matrix.to(device)
            torch.distributed.barrier()
            torch.distributed.all_reduce(confusion_matrix)
        recall_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
        recall_per_class[recall_per_class != recall_per_class] = 0 # set all nans to 0
        precision_per_class = confusion_matrix.diag()/confusion_matrix.sum(0)
        precision_per_class[precision_per_class != precision_per_class] = 0 # set all nans to 0
        
        recall_per_class_list = []
        precision_per_class_list = []
        for i in range(len(recall_per_class)):
            metric_logger.meters['recall_per_class_'+str(i)].update(recall_per_class[i].item(), n=1)
            metric_logger.meters['precision_per_class_'+str(i)].update(precision_per_class[i].item(), n=1)
            recall_per_class_list.append(recall_per_class[i].item())
            precision_per_class_list.append(precision_per_class[i].item())
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Acc@1 {top1.global_avg:.3f}'
          .format(top1=metric_logger.acc1))
    
    recall_dict = {}
    precision_dict = {}
    for i in range(len(recall_per_class_list)):
        recall_dict['Recall '+class_names[i]] = recall_per_class_list[i]
        precision_dict['Precision '+class_names[i]] = precision_per_class_list[i]
    
    return (metric_logger.acc1.global_avg)/100, recall_dict, precision_dict

def load_data(args):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()

    dataset_train = CSVDataset(
        args.imgs_path,
        args.data_path if args.data_path else args.data_paths[0],
        transforms.Compose([
            transforms.RandomResizedCrop(tuple(args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'binary')
    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = CSVDataset(
        args.imgs_path,
        args.data_path if args.data_path else args.data_paths[1],
        transforms.Compose([
            transforms.Resize(tuple(args.input_size)),
            transforms.ToTensor(),
            normalize,]),
        'binary')

    class_names = dataset_train.classes
    
    if args.data_path:
        splitpoint=int(len(dataset_train)*(1-args.test_split))
        indices = torch.randperm(len(dataset_train)).tolist()

        dataset_train = torch.utils.data.Subset(dataset_train, indices[:splitpoint])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[splitpoint:])
    
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        # oversample minority class, so that model sees roughly same num_samples from each class
        #train_sampler = torch.utils.data.WeightedRandomSampler(weights[train_ys], (max(class_sample_count)*2))
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        
    if args.data_path:
        log_data_split(args, indices, splitpoint)
    else:
        mlflow.log_artifact(args.data_paths[0], artifact_path="data")
        mlflow.log_artifact(args.data_paths[1], artifact_path="data")
    
    return dataset_train, dataset_test, train_sampler, test_sampler, class_names


def main(args, trial=None):
    
    torch.manual_seed(1)

    if utils.is_main_process():
        if args.output_dir:
            if Path(args.output_dir).exists():
                shutil.rmtree(args.output_dir)
            utils.mkdir(args.output_dir)

#    utils.init_distributed_mode(args)
    print(args)

    dataset, dataset_test, train_sampler, test_sampler, class_names = load_data(args)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

    model = reshape_classification_head(model, args, class_names)

    device = torch.device(args.device)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return
    
    print("Start training")
    start_time = time.time()
    best_model_performance = 0.0
    best_model_epoch = 0
    
    if utils.is_main_process():
        # Log parameters
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        acc, recall_per_class, precision_per_class = evaluate(model, criterion, data_loader_test, device=device, class_names=class_names)
                
        if utils.is_main_process():
            # log metrics
            mlflow.log_metric('Accuracy', acc, step=epoch)
            mlflow.log_metrics(recall_per_class, step=epoch)
            mlflow.log_metrics(precision_per_class, step=epoch)
                
            # prune run (trial) if its not going well
            if trial is not None:
                intermediate_value = acc
                trial.report(intermediate_value, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            #track best model
            if acc > best_model_performance:
                best_model = copy.deepcopy(model_without_ddp)
                best_model_optimizer = copy.deepcopy(optimizer)
                best_model_lr_scheduler = copy.deepcopy(lr_scheduler)
                best_model_performance = acc
                best_model_epoch = epoch

    #save best model locally
    if args.output_dir:
        checkpoint = {
            'model': best_model.state_dict(),
            'optimizer': best_model_optimizer.state_dict(),
            'lr_scheduler': best_model_lr_scheduler.state_dict(),
            'args': args}
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'model.pth'))

    # log model
    if utils.is_main_process():
        print("\nLogging the trained model as a run artifact...")
        mlflow.pytorch.log_model(best_model, artifact_path="pytorch-model", pickle_module=pickle)
        print("\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-model"))
            
        # also save best model performance as last step performance
        mlflow.log_metric('Accuracy', best_model_performance, step=args.epochs)
            
        calc_log_ROC(best_model, data_loader_test, device=args.device, classes=class_names)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return best_model_performance


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-path', help='path to csv containing dataset')
    group.add_argument('--data-paths', nargs=2, help='path to train- and test-csv containing labels and paths to images')
    parser.add_argument('--imgs-path', default='/beegfs/global0/ws/s7740678-apl_classification_repo/apl_classification/apl_classification/upload/kaggle/wsi', help='folder containing all the images')
    parser.add_argument('--name', default='default', help='name of the training run')
    parser.add_argument('--experiment-name', help='name of the experiment')
    parser.add_argument('--input-size', type=int, nargs=2, default=[150,150], help='shape to which images are resized for training: (h, w)')
    parser.add_argument('--test-split', type=float, default=0.2, help='portion of data to use for testing')
    parser.add_argument('--model', default='resnet50', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=10, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        #action="store_true",
        default=True
    )
    # distributed training parameters
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.name):
        main(args)
    #main(args)
