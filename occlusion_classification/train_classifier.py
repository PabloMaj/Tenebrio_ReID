import os
import shutil
import random
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from pathlib import Path

ROOT_PATH = str(Path(__file__).resolve().parent.parent / "datasets" / "occlusion_classification") + "/"

no_neurons_in_hl = {
    "512": [256, 128, 64],
    "1024": [512, 256, 128],
    "1280": [512, 256, 128],
    "1792": [1024, 512, 128],
    "2048": [1024, 512, 128],
    "2560": [1024, 512, 128],
    "4096": [1024, 512, 128]
}

label_to_num = {
    "non_occluded": 0,
    "occluded": 1
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class net(nn.Module):

    def __init__(self, num_ftrs=None, no_neurons=None):
        super(net, self).__init__()

        self.no_hidden_layers = len(no_neurons)
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=no_neurons[0]),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=no_neurons[0], out_features=no_neurons[1]),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=no_neurons[1], out_features=no_neurons[2]),
            nn.ReLU())
        self.fc_last = nn.Sequential(
            nn.Linear(in_features=no_neurons[-1], out_features=1),
            nn.Sigmoid())

    def forward(self, x):

        out = self.fc(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc_last(out)

        return out


def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        if feature_extracting:
            param.requires_grad = False
        else:
            param.requires_grad = True


def create_model(model_name=None):

    use_pretrained = True
    feature_extract = False

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

    if "resnet" in model_name:
        num_ftrs = model_ft.fc.in_features
    if model_name == "mobilenet_v2":
        num_ftrs = model_ft.classifier[1].in_features
    if "efficientnet" in model_name:
        num_ftrs = model_ft._fc.in_features

    no_neurons = no_neurons_in_hl[str(num_ftrs)]
    net_add = net(num_ftrs=num_ftrs, no_neurons=no_neurons)

    if "efficientnet" not in model_name:
        set_parameter_requires_grad(model_ft, feature_extract)
    else:
        for param in model_ft._fc.parameters():
            param.require_grad = True

    if "resnet" in model_name:
        model_ft.fc = net_add
    if model_name == "mobilenet_v2":
        model_ft.classifier[1] = net_add
    if "efficientnet" in model_name:
        model_ft._fc = net_add

    model = model_ft
    model.to(device)

    return model


class BeetleClassificationDataset(Dataset):

    def __init__(self, image_paths=None, labels=None, transform=False, label_to_num=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_to_num = label_to_num

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = image.astype("float") / 255
        # image = cv2.resize(image, (128, 128))
        label = self.labels[idx]

        label_num = np.array([self.label_to_num[label]])

        if self.transform is not None:
            image = self.transform(image)

        # print(f"{image}\t{label_num}")
        return image, label_num


class DatasetCreator:

    def __init__(self, repeat):
        self.repeat = repeat

    def create_dataset(self):

        all_paths_with_labels = []

        path_to_dataset = ROOT_PATH + "samples/"
        path_to_save = ROOT_PATH + "dataset/"

        for subset_name in ["train", "val", "test"]:
            for filename in os.listdir(path_to_save + subset_name):
                os.remove(path_to_save + subset_name + f"/{filename}")

        for folder_name in os.listdir(path_to_dataset):
            for filename in os.listdir(path_to_dataset + folder_name):
                path_ = path_to_dataset + folder_name + f"/{filename}"
                label_ = folder_name
                all_paths_with_labels.append((path_, label_))

        random_state = 42
        random.seed(random_state)
        random.shuffle(all_paths_with_labels)

        all_paths = [el[0] for el in all_paths_with_labels]
        all_labels = [el[1] for el in all_paths_with_labels]
        # print(all_paths[:5])

        split_train_val = 0.7

        id_start = int(len(all_paths_with_labels) * (repeat - 1) / 3)
        id_end = int(len(all_paths_with_labels) * repeat / 3)

        test_paths = all_paths[id_start:id_end]
        train_val_paths = all_paths[:id_start] + all_paths[id_end:]
        test_labels = all_labels[id_start:id_end]
        train_val_labels = all_labels[:id_start] + all_labels[id_end:]

        id_loc = int(split_train_val * len(train_val_paths))
        train_paths = train_val_paths[:id_loc]
        val_paths = train_val_paths[id_loc:]
        train_labels = train_val_labels[:id_loc]
        val_labels = train_val_labels[id_loc:]

        print(f"{len(train_paths)}\t{len(train_labels)}\t{len(val_paths)}\t{len(val_labels)}\t{len(test_paths)}\t{len(test_labels)}")
        print(train_labels.count("occluded"))
        print(val_labels.count("occluded"))
        print(test_labels.count("occluded"))

        for subset_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            for path_ in paths:
                filename = os.path.basename(path_)
                shutil.copy(path_, path_to_save + subset_name + f"/{filename}")

        """
        train_paths_occluded = [path_ for path_ in train_paths if "non_occluded" not in path_]
        no_occluded_samples = len(train_paths_occluded)
        no_non_occluded_samples = len([path_ for path_ in train_paths if "non_occluded" in path_])
        no_copies = int(no_non_occluded_samples / no_occluded_samples)
        print(f"no_copies={no_copies}")
        print("-------------------------------------------")
        for copy_id in range(no_copies):
            for path_ in train_paths_occluded:
                filename = os.path.basename(path_)
                shutil.copy(path_, path_to_save + "train" + "/" + filename.replace(".png", f"_{copy_id}.png"))
                train_paths.append(path_to_save + "train" + "/" + filename.replace(".png", f"_{copy_id}.png"))
                train_labels.append("occluded")
        """

        return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

    def create_additional_train_samples(self):

        path_to_save = ROOT_PATH + "dataset/train/"

        min_overlap = 500

        train_paths_add = []
        train_labels_add = []

        print("HERE")

        counter = 1
        for img_name in os.listdir(path_to_save):

            img = cv2.imread(path_to_save + img_name)

            binary_mask = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2] > 0).astype("int")

            while (1):

                rows_bbox = random.choice(list(range(16, 64)))
                cols_bbox = random.choice(list(range(16, 64)))

                x_start_bbox = random.choice(list(range(0, 128 - rows_bbox)))
                x_end_bbox = x_start_bbox + rows_bbox
                y_start_bbox = random.choice(list(range(0, 128 - cols_bbox)))
                y_end_bbox = y_start_bbox + cols_bbox

                overlap = np.sum(binary_mask[x_start_bbox:x_end_bbox, y_start_bbox:y_end_bbox])
                if overlap > min_overlap and np.sum(np.logical_not(binary_mask[x_start_bbox:x_end_bbox, y_start_bbox:y_end_bbox])) > 100:
                    img[x_start_bbox:x_end_bbox, y_start_bbox:y_end_bbox, :] = 0
                    filename_out = f"occluded_augmented_{counter}.png"
                    cv2.imwrite(ROOT_PATH + "dataset/train/" + filename_out, img)
                    train_paths_add.append(ROOT_PATH + "dataset/train/" + filename_out)
                    train_labels_add.append("occluded")
                    counter += 1
                    break

        return train_paths_add, train_labels_add


if __name__ == "__main__":

    learning_rate = 10**-5
    num_epochs = 200

    batch_size = 16

    feature_extract = False
    use_pretrained = True
    num_classes = 1
    no_hidden_layers = 3

    mean_normalize_values = (0.186, 0.219, 0.256)
    std_normalize_values = (0.175, 0.196, 0.199)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize(mean_normalize_values, std_normalize_values)]
    )

    for model_name in ["resnet18", "resnet50", "resnet101", "mobilenet_v2", "efficientnet-b0", "efficientnet-b4"][:1]:
        for repeat in [1, 2, 3]:

            dataset_creator = DatasetCreator(repeat=repeat)
            image_train_paths, image_val_paths, image_test_paths, train_labels, val_labels, test_labels = dataset_creator.create_dataset()
            train_paths_add, train_labels_add = dataset_creator.create_additional_train_samples()
            image_train_paths += train_paths_add
            train_labels += train_labels_add

            # create datasets and dataloaders
            dataset_train = BeetleClassificationDataset(image_train_paths, train_labels, transform=transform, label_to_num=label_to_num)
            dataset_val = BeetleClassificationDataset(image_val_paths, val_labels, transform=transform, label_to_num=label_to_num)
            dataset_test = BeetleClassificationDataset(image_test_paths, test_labels, transform=transform, label_to_num=label_to_num)
            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(dataset_val, batch_size=batch_size)
            test_dataloader = DataLoader(dataset_test, batch_size=batch_size)

            # create model
            model = create_model(model_name=model_name)
            # print(summary(model, (3, 128, 128)))

            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            count = 0
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(train_dataloader):

                    images = images.to(device, dtype=torch.float32)
                    labels = labels.to(device, dtype=torch.float32)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

                print("epoch\tsubset_name\tF1\tprecision\trecall\tAUCPR\tAUCROC\tthresh")
                thresh_opt = None
                for subset_name, eval_dataloader in [("val", val_dataloader), ("test", test_dataloader)]:

                    predictions_list = []
                    scores_list = []
                    labels_list = []

                    for images, labels in eval_dataloader:

                        images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
                        outputs = model(images)

                        for val in list(outputs.cpu().detach().numpy()):
                            # print(val)
                            scores_list.append(val[0])
                        for val in list(labels.cpu().detach().numpy()):
                            labels_list.append(int(val[0]))

                    # print(labels_list)
                    # print(predictions_list)

                    if subset_name == "val":
                        # pick the decision threshold on val only; test reuses it below so
                        # the reported test metrics aren't tuned on the test set itself
                        thresh_values = list(np.arange(0, 1, 0.001))
                        F1_values = []
                        for thresh in thresh_values:
                            predictions_list = [score > thresh for score in scores_list]
                            F1 = round(f1_score(y_true=labels_list, y_pred=predictions_list), 3)
                            F1_values.append(F1)
                        thresh_opt = thresh_values[np.argmax(F1_values)]

                    predictions_list = [score > thresh_opt for score in scores_list]
                    F1 = round(f1_score(y_true=labels_list, y_pred=predictions_list), 3)

                    precision = round(precision_score(y_true=labels_list, y_pred=predictions_list), 3)
                    recall = round(recall_score(y_true=labels_list, y_pred=predictions_list), 3)

                    AUCPR = round(average_precision_score(y_true=labels_list, y_score=scores_list), 3)
                    AUCROC = round(roc_auc_score(y_true=labels_list, y_score=scores_list), 3)

                    print(f"{epoch}\t{subset_name}\t{F1}\t{precision}\t{recall}\t{AUCPR}\t{AUCROC}\t{thresh_opt}")

                    if F1 > 0.8 and subset_name == "val":
                        torch.save(model.state_dict(), "example_model_occlusion_classification.pt")
                        with open("example_model_occlusion_classification_threshold.txt", "w") as f:
                            f.write(str(thresh_opt))
                        print("Problematic samples")
                        for i, path_ in enumerate(image_val_paths):
                            img_name = os.path.basename(path_)
                            if predictions_list[i] != labels_list[i]:
                                print(img_name)
                        print("--------------------------------")
                print("------------------------------------------------")
