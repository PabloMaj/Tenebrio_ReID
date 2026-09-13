import cv2
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "re_identification"


class BeetleDataset(Dataset):

    def __init__(self, image_paths=None, labels=None, transform=False, label_to_num=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_to_num = label_to_num

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath).astype("float") / 255
        image = cv2.resize(image, (128, 128))
        label = self.labels[idx]

        label_num = self.label_to_num[label]

        if self.transform is not None:
            image = self.transform(image)
        return image, label_num


class Net(nn.Module):
    def __init__(self, kernel_size=None, number_of_CNN_blocks=None, architecture_type=None, number_of_fc_layers=None, use_dropout=None, type_pool=None):
        super(Net, self).__init__()
        self.architecture_type = architecture_type
        self.number_of_CNN_blocks = number_of_CNN_blocks
        self.number_of_fc_layers = number_of_fc_layers
        self.use_dropout = use_dropout
        if kernel_size == 3:
            padding_val = 1
        elif kernel_size == 5:
            padding_val = 2
        if architecture_type == "conv_pool_conv":
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding_val)
        elif architecture_type == "2conv_pool_2conv":
            self.conv1_add = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv2_add = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv3_add = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv4_add = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding_val)
            self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding_val)
        if type_pool == "max_pool":
            self.pool = nn.MaxPool2d(kernel_size=2)
        elif type_pool == "average_pool":
            self.pool = nn.AvgPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        basic_dim = int(128 / 2**number_of_CNN_blocks)
        self.fc1 = nn.Linear(in_features=basic_dim * basic_dim * [16, 32, 64, 128][number_of_CNN_blocks - 1], out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)

    def forward(self, x):
        if self.architecture_type == "2conv_pool_2conv":
            x = F.relu(self.conv1_add(x))
        x = F.relu(self.pool(self.conv1(x)))
        if self.architecture_type == "2conv_pool_2conv":
            x = F.relu(self.conv2_add(x))
        x = F.relu(self.pool(self.conv2(x)))
        if self.number_of_CNN_blocks >= 3:
            if self.architecture_type == "2conv_pool_2conv":
                x = F.relu(self.conv3_add(x))
            x = F.relu(self.pool(self.conv3(x)))
        if self.number_of_CNN_blocks >= 4:
            if self.architecture_type == "2conv_pool_2conv":
                x = F.relu(self.conv4_add(x))
            x = F.relu(self.pool(self.conv4(x)))
        if self.use_dropout:
            x = F.dropout(self.drop(x), training=self.training)
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc1(x)
        if self.number_of_fc_layers >= 2:
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(self.drop(x), training=self.training)
            x = self.fc2(x)
        if self.number_of_fc_layers >= 3:
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(self.drop(x), training=self.training)
            x = self.fc3(x)
        return x


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        embeddings = model(data)
        # print(type(embeddings))
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator, epoch=None, path_to_save=None):

    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)

    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)

    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        query=test_embeddings, reference=train_embeddings, query_labels=test_labels,
        reference_labels=train_labels, ref_includes_query=False)

    # print(accuracies.keys())
    test_labels = np.unique(test_labels.cpu().detach().numpy())
    # mAP_values = accuracies["mean_average_precision"]

    if epoch == 10:
        f = open(path_to_save, "a")
        f.write("epoch\tlabel\t")
        for j, key_ in enumerate(accuracies.keys()):
            f.write(key_)
            if j == (len(accuracies.keys()) - 1):
                f.write("\n")
            else:
                f.write("\t")
        f.close()
    if epoch % 10 == 0:
        f = open(path_to_save, "a")
        for i, label in enumerate(list(test_labels)):
            f.write(f"{epoch}\t{test_labels[i]}\t")
            for j, key_ in enumerate(accuracies.keys()):
                if key_ in ["NMI", "AMI"]:
                    f.write(f"{np.round(accuracies[key_], 4)}")
                else:
                    f.write(f"{np.round(accuracies[key_][i], 4)}")
                if j == (len(accuracies.keys()) - 1):
                    f.write("\n")
                else:
                    f.write("\t")
        f.close()
    # print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    print("mAP = {}".format(accuracies["mean_average_precision"]))


def oversampling_for_train_samples(image_train_paths=None, train_labels=None):

    image_train_paths_updated = image_train_paths.copy()
    train_labels_updated = train_labels.copy()

    counts = dict()
    indexes_of_elements_in_list = dict()
    max_count = 0

    for label in np.unique(train_labels):
        counts[label] = train_labels.count(label)
        if counts[label] > max_count:
            max_count = counts[label]
        indexes_of_elements_in_list[label] = [i for i, el in enumerate(train_labels) if el == label]
    for label in np.unique(train_labels):
        no_samples_to_copy = max_count - counts[label]
        while no_samples_to_copy >= len(indexes_of_elements_in_list[label]):
            no_samples_to_copy -= len(indexes_of_elements_in_list[label])
            for index_ in indexes_of_elements_in_list[label]:
                image_train_paths_updated.append(image_train_paths[index_])
                train_labels_updated.append(train_labels[index_])
        while no_samples_to_copy != 0:
            no_samples_to_copy -= 1
            index_ = random.choice(indexes_of_elements_in_list[label])
            image_train_paths_updated.append(image_train_paths[index_])
            train_labels_updated.append(train_labels[index_])

    # checking
    # print(Counter(train_labels_updated))

    return image_train_paths_updated, train_labels_updated


if __name__ == '__main__':

    # settings
    part_for_analysis = "thorax"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_epochs = 300
    train_val_ratio = 0.2

    # parameters for training
    margin_value = 0.2
    # margin values: [0.02, 0.05, 0.1, 0.2, 0.4] (5 options)
    distance_type = "cosine"
    # distance_type: euclidean or cosine (2 options)
    type_of_triplets = "semihard"
    # type_of_triplets: ["all", "hard", "semihard", "easy"] (4 options)

    # mean and std vectors for normalize
    mean_normalize_values = dict()
    std_normalize_values = dict()
    mean_normalize_values["thorax"] = (0.149, 0.176, 0.211)
    std_normalize_values["thorax"] = (0.093, 0.114, 0.112)
    mean_normalize_values["beetle"] = (0.186, 0.219, 0.256)
    std_normalize_values["beetle"] = (0.175, 0.196, 0.199)

    # transformation definition
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize(mean_normalize_values[part_for_analysis], std_normalize_values[part_for_analysis])]
    )

    path_to_dataset = str(DATASET_DIR / "re_identification_dataset") + "/"
    path_to_save = "results/"

    series_names = ["seria_2", "seria_3", "seria_4", "seria_5"]

    counter_experiment = 0
    for kernel_size in [3, 5]:
        for number_of_CNN_blocks in [4, 3, 2]:
            for architecture_type in ["conv_pool_conv", "2conv_pool_2conv"]:
                for number_of_fc_layers in [3, 2, 1]:
                    for use_dropout in [1, 0]:
                        for type_pool in ["average_pool", "max_pool"]:
                            for repeat_id in [1, 2, 3]:

                                # prepare dataset, splitting into train/val/test
                                image_train_paths = []
                                image_val_paths = []
                                image_test_paths = []

                                train_labels = []
                                val_labels = []
                                test_labels = []

                                label_to_num = dict()
                                counter = 1

                                for serie_name in series_names:
                                    for folder_name in os.listdir(path_to_dataset + serie_name):

                                        label = serie_name + "_" + folder_name

                                        if label not in label_to_num:
                                            label_to_num[label] = counter
                                            counter += 1

                                        for stage_type in ["free", "isolated"]:
                                            for filename in os.listdir(path_to_dataset + f"{serie_name}/{folder_name}/{stage_type}"):

                                                if part_for_analysis in filename:
                                                    if stage_type == "isolated":
                                                        if random.random() > train_val_ratio:
                                                            image_train_paths.append(path_to_dataset + f"{serie_name}/{folder_name}/{stage_type}/{filename}")
                                                            train_labels.append(label)
                                                        else:
                                                            image_val_paths.append(path_to_dataset + f"{serie_name}/{folder_name}/{stage_type}/{filename}")
                                                            val_labels.append(label)
                                                    elif stage_type == "free":
                                                        image_test_paths.append(path_to_dataset + f"{serie_name}/{folder_name}/{stage_type}/{filename}")
                                                        test_labels.append(label)
                                # print(label_to_num)
                                # print(f"No of samples in train/val/test before oversampling: {len(image_train_paths)} {len(image_val_paths)} {len(image_test_paths)}")
                                # oversampling for train set
                                image_train_paths, train_labels = oversampling_for_train_samples(image_train_paths=image_train_paths, train_labels=train_labels)
                                # print(f"No of samples in train/val/test after oversampling: {len(image_train_paths)} {len(image_val_paths)} {len(image_test_paths)}")

                                # creation datasets and dataloaders
                                dataset_train = BeetleDataset(image_train_paths, train_labels, transform=transform, label_to_num=label_to_num)
                                dataset_val = BeetleDataset(image_val_paths, val_labels, transform=transform, label_to_num=label_to_num)
                                dataset_test = BeetleDataset(image_test_paths, test_labels, transform=transform, label_to_num=label_to_num)
                                train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
                                val_loader = DataLoader(dataset_val, batch_size=batch_size)
                                test_loader = DataLoader(dataset_test, batch_size=batch_size)

                                # initialization of model and optimizer
                                model = Net(kernel_size=kernel_size, number_of_CNN_blocks=number_of_CNN_blocks, architecture_type=architecture_type,
                                            number_of_fc_layers=number_of_fc_layers, use_dropout=use_dropout, type_pool=type_pool)
                                model = model.to(device)
                                optimizer = optim.Adam(model.parameters(), lr=0.01)

                                # settings for loss
                                if distance_type == "cosine":
                                    distance = distances.CosineSimilarity()
                                elif distance_type == "euclidean":
                                    distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
                                reducer = reducers.ThresholdReducer(low=0)
                                loss_func = losses.TripletMarginLoss(margin=margin_value, distance=distance, reducer=reducer)
                                mining_func = miners.TripletMarginMiner(margin=margin_value, distance=distance, type_of_triplets=type_of_triplets)

                                accuracy_calculator = AccuracyCalculator(include=(), avg_of_avgs=False, return_per_class=True, device=torch.device("cpu"))

                                folder_name_to_save = f"kernel_size.{kernel_size}.number_of_CNN_blocks.{number_of_CNN_blocks}."
                                folder_name_to_save += f"architecture_type.{architecture_type}.number_of_fc_layers.{number_of_fc_layers}."
                                folder_name_to_save += f"use_dropout.{use_dropout}.type_pool.{type_pool}.repeat_id.{repeat_id}."

                                for epoch in range(1, num_epochs + 1):
                                    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
                                    if epoch == 1:
                                        try:
                                            os.makedirs(path_to_save + folder_name_to_save)
                                        except Exception as e:
                                            print(e)
                                            pass
                                        f1 = open(path_to_save + folder_name_to_save + "/results_eval_val.txt", "w")
                                        f2 = open(path_to_save + folder_name_to_save + "/results_eval_test.txt", "w")
                                        f3 = open(path_to_save + folder_name_to_save + "/params.txt", "w")

                                        f3.write(f"kernel_size={kernel_size}\nnumber_of_CNN_blocks={number_of_CNN_blocks}\n")
                                        f3.write(f"architecture_type={architecture_type}\nnumber_of_fc_layers={number_of_fc_layers}\n")
                                        f3.write(f"use_dropout={use_dropout}\ntype_pool={type_pool}\nrepeat_id={repeat_id}\n")
                                        f3.write(f"part_for_analysis={part_for_analysis}\nbatch_size={batch_size}\nnum_epochs={num_epochs}\n")
                                        f3.write("loss_func=TripletMarginLoss\nreducer=ThresholdReducer(low=0)\nmining_func=TripletMarginMiner\n")
                                        f3.write(f"train_val_ratio={train_val_ratio}\nmargin_value={margin_value}\ndistance_type={distance_type}\n")
                                        f3.write(f"type_of_triplets={type_of_triplets}\noptimizer=Adam_0.01\n")

                                        f1.close()
                                        f2.close()
                                        f3.close()

                                    if epoch % 10 == 0:
                                        test(dataset_train, dataset_val, model, accuracy_calculator, epoch=epoch, path_to_save=path_to_save + folder_name_to_save + "/results_eval_val.txt")
                                        test(dataset_train, dataset_test, model, accuracy_calculator, epoch=epoch, path_to_save=path_to_save + folder_name_to_save + "/results_eval_test.txt")

                                print(counter_experiment)
                                counter_experiment += 1
