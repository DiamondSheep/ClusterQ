import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.io import read_image
import utils as utils
from options import Option
import os
import sys
import copy
import logging
import argparse
from dataloader import DataLoader as DLR
from gan import Generator, Generator_imagenet, Qimera_Generator
from pytorchcv.model_provider import get_model as ptcv_get_model
from quantization_utils.quant_modules import *
from sklearn.manifold import TSNE
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

#mpl.rcParams['lines.markersize'] = np.sqrt(20)

def distilled_dataloader(data_path, batchsize, num_workers, for_inception=False):
    dataset = Distilled(path=data_path)
    dataloader = DataLoader(dataset, batch_size=batchsize,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader

class Distilled(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.img_dir = path
        self.transform = transform
        self.iter_idx = -1
        self.files = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        self.iter_idx += 1
        if (self.iter_idx >= len(self)):
            raise StopIteration
        return self[self.iter_idx]

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.files[index])
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)

        return img.float()


class Visual(object):
    def __init__(self, generator, model, train_loader, test_loader, options=None, conf_path=None, args=None):
        super().__init__()
        self.settings = options or Option(conf_path)
        self.generator = utils.data_parallel(generator, self.settings.nGPU, self.settings.GPU)
        self.model = model.cuda()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tsne = TSNE(n_components=2, init='pca', random_state=0)
        self.args = args

        self.features = []
        self.mean_list = []
        self.var_list = []
        self.running_mean = []
        self.running_var = []
        # inistialization of hooks
        self.generator.eval()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3]).cpu()
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False).cpu()
        self.mean_list.append(mean)
        self.var_list.append(var)

        self.running_mean.append(module.running_mean) # number_of_layers x channels
        self.running_var.append(module.running_var)   # number_of_layers x channels
        self.features.append(output.cpu())

    def feature_analysis(self, labels, synthetic = False):
        if (synthetic):
            save_path = f"visual_result/{self.settings.dataset}/synthetic/"
        else:
            save_path = f"visual_result/{self.settings.dataset}/original/"
        print(f"feature analysis saved : {save_path}")
        for i, feature in enumerate(self.features):
            print(f"layer {i}")
            feature = feature.reshape(feature.shape[0], -1)
            feature_sne = self.tsne.fit_transform(feature)
            # normalization
            feature_sne = (feature_sne - feature_sne.mean()) / feature_sne.std()

            plt.figure(i)
            plt.axis('on')

            labels = labels.cpu()
            plt.scatter(feature_sne[:, 0], feature_sne[:, 1], alpha=0.5, label=labels, c=labels)
            plt.savefig(os.path.join(save_path, f"{self.settings.model}_layer_{i}.eps"), dpi=600, format='eps')
            plt.legend()
            plt.clf();
        self.features = []
        cont = input()

    def BNS_analysis(self, running_times = 200, save_path = "visual_result/cifar10/BNS", generator=None):
        print (f"Save path: {save_path}")
        mean_sne_list = []
        var_sne_list = []
        label_list = []
        if (generator == None):
            print ("BNS Analysis for Original Data")
            """
            save BNS in each layer to a mat [label, layer, channel] -> [layer, label, channel]
            """
            with torch.no_grad():
                for i, (images, labels) in enumerate(self.test_loader):
                    if running_times == 0:
                        break
                    images = images.cuda()
                    label_list.append(labels[0])
                    output = self.model(images)
                    if (len(mean_sne_list) == 0 & len(var_sne_list) == 0):
                        # initialization
                        for mean in self.mean_list:
                            mean_sne_list.append(mean.reshape(1,-1).cpu())
                        for var in self.var_list:
                            var_sne_list.append(var.reshape(1,-1).cpu())
                    else:
                        # concat
                        assert len(mean_sne_list) == len(var_sne_list)
                        for j in range(len(mean_sne_list)):
                            mean_sne_list[j] = torch.cat((mean_sne_list[j], self.mean_list[j].reshape(1,-1).cpu()), 0)
                            var_sne_list[j] = torch.cat((var_sne_list[j], self.var_list[j].reshape(1,-1).cpu()), 0)
                    running_times -= 1
                    self.features = []
                    self.running_mean = []
                    self.running_var = []
        else:
            for i in range(len(self.features)):
                self.model()
                print (self.running_mean[i].shape)
                conti = input()

        for layer in range(len(mean_sne_list)):
            print(f"layer {layer}")
            mean_sne = self.tsne.fit_transform(mean_sne_list[layer])
            mean_sne = (mean_sne - mean_sne.min()) / (mean_sne.max() - mean_sne.min())
            plt.figure(layer)
            plt.axis('on')
            plt.scatter(mean_sne[:, 0], mean_sne[:, 1], alpha=0.5, label=label_list, c=label_list)
            plt.savefig(os.path.join(save_path, f"{self.settings.model}_layer_{layer}"))
            plt.legend()
            plt.clf()

    def test_original(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                labels = labels.cuda()
                images = images.cuda()
                output = self.model(images)
                self.feature_analysis(labels, synthetic=True)

    def test_generator(self, generator_path=None):
        #print(self.generator)
        generator_state_dict = torch.load(generator_path)
        #for key in generator_state_dict.keys():
        #    print(f"{key}: {generator_state_dict[key].shape}")
        self.generator.load_state_dict(generator_state_dict)
        self.model.eval()
        labels = torch.randint(0, self.settings.nClasses, (self.settings.batchSize,)).cuda()
        labels = labels.contiguous()
        z = torch.randn(self.settings.batchSize, self.settings.latent_dim).cuda()
        z = z.contiguous()
        images = self.generator(z, labels)
        with torch.no_grad():
            output = self.model(images)
            self.feature_analysis(labels, synthetic=True)


    def generate_images(self, generator_path = None):
        generator_state_dict = torch.load(generator_path)

        self.generator.load_state_dict(torch.load(generator_path))
        print("state loaded.")
        labels = torch.randint(0, self.settings.nClasses, (self.settings.batchSize,)).cuda()
        labels = labels.contiguous()
        z = torch.randn(self.settings.batchSize, self.settings.latent_dim).cuda()
        z = z.contiguous()
        images = self.generator(z, labels)
        synthetic_images_path = f"visual_result/{self.settings.dataset}/synthetic_images/"
        print(f"images saved : {synthetic_images_path}")
        for i, image in enumerate(images):
            save_image(image, os.path.join(synthetic_images_path, f"pic_{i}_label_{labels[i]}.png"))

    def save_original(self):
        for images, labels in self.test_loader:
            for i in range(len(images)):
                save_image(images[i], f"visual_result/cifar10/original_images/label_{labels[i]}_{i}.png")
            conti = input()

    def test_zeroq(self):
        distilled = distilled_dataloader(data_path="data/ZeroQ_syn/distilled_data", batchsize=self.settings.batchSize, num_workers=4)
        self.model.eval()
        with torch.no_grad():
            for images in distilled:
                images = images.cuda()
                output = self.model(images)
                _, index = torch.max(output, dim=1)
                self.feature_analysis(index, synthetic = True) # No label given by ZeroQ

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--conf_path', type=str, metavar='conf_path',
                        help='input the path of config file')
    parser.add_argument('--id', type=int, metavar='experiment_id',
                        help='Experiment ID')
    parser.add_argument('--gen', type=str, metavar='Generator',
                        help='generator select')
    args = parser.parse_args()
    # Options
    option = Option(args.conf_path)
    option.manualSeed = args.id + 1
    option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)
    option.set_save_path()
    # Logger
    logger = logging.getLogger('baseline')
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    # file log
    file_handler = logging.FileHandler(os.path.join(option.save_path, "train_test.log"))
    file_handler.setFormatter(file_formatter)
    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    # Model
    if option.dataset in ["cifar10"]:
        model = ptcv_get_model(f'{option.model}_cifar10', pretrained=True)
    elif option.dataset in ["cifar100"]:
        model = ptcv_get_model(f'{option.model}_cifar100', pretrained=True)
    elif option.dataset in ["imagenet"]:
        model = ptcv_get_model(option.model, pretrained=True)
    else:
        print("Illegal model")
        exit(1)
    model.eval()
    # Generator
    if option.dataset in ["cifar100", "cifar10"]:
        if (args.gen in ['qimera']):
            print("------using Qimera GAN------")
            generator = Qimera_Generator(option)
        else:
            generator = Generator(option)
    elif option.dataset in ["imagenet"]:
        generator = Generator_imagenet(option)
    else:
        assert False, "invalid dataset"
    # Dataset
    data_loader = DLR(dataset=option.dataset,
                            batch_size=option.batchSize, # NOTE: set 1 for inference
                            data_path=option.dataPath,
                            n_threads=option.nThreads,
                            ten_crop=option.tenCrop,
                            logger=logger)
    train_loader, test_loader = data_loader.getloader()
    logger.info("dataset loaded.")
    # Visual
    visual = Visual(generator, model, train_loader, test_loader, option, args)

    # output images
    #visual.save_original()
    visual.generate_images("generated_model/resnet20/cifar10_resnet20_generator.pth")

    # t-SNE analysis
    #visual.test_original()
    #visual.test_generator(generator_path = "generated_model/resnet20_16layer/cifar10_resnet20_generator.pth")
    #visual.test_zeroq()

    #visual.BNS_analysis()

