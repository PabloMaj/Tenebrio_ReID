import cv2
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from torchvision import models
from efficientnet_pytorch import EfficientNet

import pickle

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

    def __init__(self, num_ftrs=None, number_of_fc_layers=None):
        super(Net, self).__init__()
        self.number_of_fc_layers = number_of_fc_layers

        self.fc1 = nn.Linear(in_features=num_ftrs, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)

    def forward(self, x):
        if self.number_of_fc_layers >= 1:
            x = self.fc1(x)
        if self.number_of_fc_layers >= 2:
            x = F.relu(x)
            x = self.fc2(x)
        if self.number_of_fc_layers >= 3:
            x = F.relu(x)
            x = self.fc3(x)
        # print(x.size())
        return x


def read_data(split_id=None, path_to_dataset=None, part_for_analysis=None, use_oversampling=True):

    series_names = ["seria_2", "seria_3", "seria_4", "seria_5"]

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

            # test dataset
            path_to_test_samples = path_to_dataset + f"{serie_name}/{folder_name}/free/"
            image_test_paths_to_add = [path_to_test_samples + filename for filename in os.listdir(path_to_test_samples) if part_for_analysis in filename]
            image_test_paths += image_test_paths_to_add
            test_labels += [label for i in range(len(image_test_paths_to_add))]

            # train and val datatest
            path_to_train_val_samples = path_to_dataset + f"{serie_name}/{folder_name}/isolated/"
            image_train_val_paths = [path_to_train_val_samples + filename for filename in os.listdir(path_to_train_val_samples) if part_for_analysis in filename]

            n = len(image_train_val_paths)
            id_start = int(n * (split_id - 1) / 5)
            id_end = int(n * split_id / 5)

            image_train_paths_to_add = image_train_val_paths[:id_start] + image_train_val_paths[id_end:]
            image_val_paths_to_add = image_train_val_paths[id_start:id_end]
            image_train_paths += image_train_paths_to_add
            image_val_paths += image_val_paths_to_add
            train_labels += [label for i in range(len(image_train_paths_to_add))]
            val_labels += [label for i in range(len(image_val_paths_to_add))]

    print(f"{len(list(np.unique(train_labels)))}\t{len(list(np.unique(val_labels)))}")
    print("--------------------")
    # os.system("pause")

    # print(label_to_num)
    print(f"No of samples in train/val/test before oversampling: {len(image_train_paths)} {len(image_val_paths)} {len(image_test_paths)}")
    # oversampling for train set
    if use_oversampling:
        image_train_paths, train_labels = oversampling_for_train_samples(image_train_paths=image_train_paths, train_labels=train_labels)
        print(f"No of samples in train/val/test after oversampling: {len(image_train_paths)} {len(image_val_paths)} {len(image_test_paths)}")

    return image_train_paths, train_labels, image_val_paths, val_labels, image_test_paths, test_labels, label_to_num


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
        # print(label)
        # print(len(train_labels))
        # print(len(indexes_of_elements_in_list[label]))
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


def init_extractor(model_name=None, number_of_fc_layers=None, feature_extract=None):

    use_pretrained = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # definition of relevant model
    if "efficientnet" in model_name:
        model_ft = EfficientNet.from_pretrained(model_name)
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
    if model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
    if model_name == "mobilenet_v2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)

    # check dimension of output
    if "resnet" in model_name:
        num_ftrs = model_ft.fc.in_features
    if model_name == "mobilenet_v2":
        num_ftrs = model_ft.classifier[1].in_features
    if "efficientnet" in model_name:
        num_ftrs = model_ft._fc.in_features

    set_parameter_requires_grad(model_ft, feature_extract)

    net_add = Net(num_ftrs=num_ftrs, number_of_fc_layers=number_of_fc_layers)

    if "resnet" in model_name:
        model_ft.fc = net_add
    if model_name == "vgg11":
        model_ft.classifier[6] = net_add
    if model_name == "mobilenet_v2":
        model_ft.classifier[1] = net_add
    if "efficientnet" in model_name:
        model_ft._fc = net_add

    model_ft.to(device)

    return model_ft, num_ftrs


def set_parameter_requires_grad(model, feature_extract):
    for param in model.parameters():
        if feature_extract:
            param.requires_grad = False
        else:
            param.requires_grad = True


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

    print(train_embeddings.shape)
    print(test_embeddings.shape)

    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)

    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        query=test_embeddings, reference=train_embeddings, query_labels=test_labels,
        reference_labels=train_labels, ref_includes_query=False)

    # print(accuracies.keys())
    test_labels = np.unique(test_labels.cpu().detach().numpy())
    # mAP_values = accuracies["mean_average_precision"]

    if epoch in [0, 10]:
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


if __name__ == '__main__':

    # settings
    part_for_analysis = "thorax"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    num_epochs = 200
    train_val_ratio = 0.2

    # mean and std vectors for normalize
    mean_normalize_values = dict()
    std_normalize_values = dict()
    mean_normalize_values["thorax"] = (0.485, 0.456, 0.406)
    std_normalize_values["thorax"] = (0.229, 0.224, 0.225)
    # mean_normalize_values["beetle"] = (0.186, 0.219, 0.256)
    # std_normalize_values["beetle"]  = (0.175, 0.196, 0.199)

    # parameters for training
    margin_value = 0.2
    # margin values: [0.02, 0.05, 0.1, 0.2, 0.4] (5 options)
    distance_type = "cosine"
    # distance_type: euclidean or cosine (2 options)
    type_of_triplets = "semihard"
    # type_of_triplets: ["all", "hard", "semihard", "easy"] (4 options)

    # transformation definition
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize(mean_normalize_values[part_for_analysis], std_normalize_values[part_for_analysis])]
    )

    path_to_dataset = str(DATASET_DIR / "re_identification_dataset_only_thorax") + "/"
    path_to_save = "results/"
    counter_experiment = 0

    # part related to model training
    for split_id in [1, 2, 3, 4, 5]:
        for number_of_fc_layers in [1]:
            for model_name in ["mobilenet_v2",]:
                for feature_extract in [False]:

                    image_train_paths, train_labels, image_val_paths, val_labels, image_test_paths, test_labels, label_to_num = read_data(
                        split_id=split_id, path_to_dataset=path_to_dataset, part_for_analysis=part_for_analysis, use_oversampling=False)

                    print("-------------------------")
                    # print(image_train_paths)
                    # check sequence in time
                    for label in tqdm(np.unique(train_labels)):
                        mask = [el == label for el in train_labels]
                        paths_choosen = [image_train_paths[i] for i in range(len(mask)) if mask[i]]
                        image_ids = [int(os.path.basename(path)[:5]) for path in paths_choosen]
                        increase_check = np.sum([image_ids[i] <= image_ids[i + 1] for i in range(len(image_ids) - 1)])
                        if (increase_check + 1) != len(image_ids):
                            print("NOT OK")
                            # print(image_ids)
                            # print("-------------------------")
                        # print(image_ids)
                        # print(train_labels)
                        # print("-------------------------")

                    # creation datasets and dataloaders
                    dataset_train = BeetleDataset(image_train_paths, train_labels, transform=transform, label_to_num=label_to_num)
                    dataset_val = BeetleDataset(image_val_paths, val_labels, transform=transform, label_to_num=label_to_num)
                    dataset_test = BeetleDataset(image_test_paths, test_labels, transform=transform, label_to_num=label_to_num)
                    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(dataset_val, batch_size=batch_size)
                    test_loader = DataLoader(dataset_test, batch_size=batch_size)

                    # init model
                    model, _ = init_extractor(model_name=model_name, number_of_fc_layers=number_of_fc_layers, feature_extract=feature_extract)

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

                    folder_name_to_save = f"model_name.{model_name}.number_of_fc_layers.{number_of_fc_layers}.feature_extract.{feature_extract}.split_id.{split_id}"

                    try:
                        os.makedirs(path_to_save + folder_name_to_save)
                    except Exception as e:
                        print(e)
                        pass

                    with open(path_to_save + folder_name_to_save + "/label_to_num.pkl", 'wb') as f:
                        pickle.dump(label_to_num, f)

                    for epoch in range(1, num_epochs + 1):
                        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)

                        if epoch == 1:

                            f1 = open(path_to_save + folder_name_to_save + "/results_eval_val.txt", "w")
                            f2 = open(path_to_save + folder_name_to_save + "/results_eval_test.txt", "w")
                            f3 = open(path_to_save + folder_name_to_save + "/params.txt", "w")

                            f3.write(f"model_name={model_name}\nnumber_of_fc_layers={number_of_fc_layers}\nfeature_extract={feature_extract}\nsplit_id={split_id}\n")
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

    """
    # part related to embeddings from extractor with frozen weights

    feature_extract = True
    number_of_fc_layers = 0

    for repeat_id in [1, 2, 3]:
        for model_name in ["resnet18", "resnet50", "resnet101", "mobilenet_v2", "efficientnet-b0", "efficientnet-b4"]:

            image_train_paths, train_labels, image_val_paths, val_labels, image_test_paths, test_labels, label_to_num = read_data(path_to_dataset)

            # creation datasets and dataloaders
            dataset_train = BeetleDataset(image_train_paths, train_labels, transform=transform, label_to_num=label_to_num)
            dataset_val = BeetleDataset(image_val_paths, val_labels, transform=transform, label_to_num=label_to_num)
            dataset_test = BeetleDataset(image_test_paths, test_labels, transform=transform, label_to_num=label_to_num)
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(dataset_val, batch_size=batch_size)
            test_loader = DataLoader(dataset_test, batch_size=batch_size)

            # init model
            model, _ = init_extractor(model_name=model_name, number_of_fc_layers=number_of_fc_layers, feature_extract=feature_extract)

            optimizer = optim.Adam(model.parameters(), lr=0.01)

            accuracy_calculator = AccuracyCalculator(include=(), avg_of_avgs=False, return_per_class=True, device=torch.device("cpu"))

            folder_name_to_save = f"model_name.{model_name}.number_of_fc_layers.{number_of_fc_layers}.feature_extract.{feature_extract}.repeat_id.{repeat_id}"

            try:
                os.makedirs(path_to_save + folder_name_to_save)
            except:
                pass
            f1 = open(path_to_save + folder_name_to_save + "/results_eval_val.txt", "w")
            f2 = open(path_to_save + folder_name_to_save + "/results_eval_test.txt", "w")
            f3 = open(path_to_save + folder_name_to_save + "/params.txt", "w")

            f3.write(f"model_name={model_name}\nnumber_of_fc_layers={number_of_fc_layers}\nfeature_extract={feature_extract}\nrepeat_id={repeat_id}\n")
            f3.write(f"part_for_analysis={part_for_analysis}\nbatch_size={batch_size}\nnum_epochs={num_epochs}\n")
            f3.write(f"train_val_ratio={train_val_ratio}")

            f1.close()
            f2.close()
            f3.close()

            test(dataset_train, dataset_val, model, accuracy_calculator, epoch=0, path_to_save=path_to_save + folder_name_to_save + "/results_eval_val.txt")
            test(dataset_train, dataset_test, model, accuracy_calculator, epoch=0, path_to_save=path_to_save + folder_name_to_save + "/results_eval_test.txt")
    """
