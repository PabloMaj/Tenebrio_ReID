import os
import numpy as np
import cv2
import random
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent / "datasets" / "marker_recognition") + "\\"


class ModelSemSegMarkers():

    def __init__(self, color_space=None, model_name=None):
        self.color_space = color_space
        self.model_name = model_name

    def change_color_space(self, img=None):

        if self.color_space == "RGB":
            return img
        elif color_space == "HSV":
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == "Lab":
            return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif color_space == "Luv":
            return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == "YCrCb":
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    def read_data_to_arrays(self, path_to_samples=None):

        X_for_splits = dict()
        Y_for_splits = dict()
        for subset_name in ["train", "val"]:
            X_for_splits[subset_name] = []
            Y_for_splits[subset_name] = []

        for split_id in [1, 2, 3, 4, 5]:

            X = dict()
            Y = dict()
            for subset_name in ["train", "val"]:
                X[subset_name] = []
                Y[subset_name] = []

            for class_name in os.listdir(path_to_samples):

                filenames_all = os.listdir(path_to_samples + class_name)
                random_state = 42
                random.seed(random_state)
                random.shuffle(filenames_all)

                id_start = int((split_id - 1) * len(filenames_all) / 5)
                id_end = int(split_id * len(filenames_all) / 5)

                filenames_choosen = dict()
                filenames_choosen["val"] = filenames_all[id_start:id_end]
                filenames_choosen["train"] = filenames_all[:id_start] + filenames_all[id_end:]

                for subset_name in ["train", "val"]:
                    for filename in filenames_choosen[subset_name]:

                        img_oryg = cv2.imread(path_to_samples + f"{class_name}\\{filename}")
                        img = self.change_color_space(img_oryg)
                        for x_loc in range(img.shape[0]):
                            for y_loc in range(img.shape[1]):
                                if np.sum(img_oryg[x_loc, y_loc, :]) != 0:
                                    if random.random() < 0.2:
                                        X[subset_name].append(list(img[x_loc, y_loc, :]))
                                        Y[subset_name].append(class_name)

            for subset_name in ["train", "val"]:
                X[subset_name] = np.array(X[subset_name])
                Y[subset_name] = np.array(Y[subset_name])
                X_for_splits[subset_name].append(X[subset_name])
                Y_for_splits[subset_name].append(Y[subset_name])

        self.X_for_splits = X_for_splits
        self.Y_for_splits = Y_for_splits

        # sprawdzenie
        for subset_name in ["train", "val"]:
            # print(f"{len(self.X_for_splits[subset_name])}\t{len(self.Y_for_splits[subset_name])}")
            for i in range(5):
                pass
                # print(f"{self.X_for_splits[subset_name][i].shape}\t{self.Y_for_splits[subset_name][i].shape}")

    def init_ML_model(self):

        if self.model_name == "LogReg":
            model_ML = LogisticRegression(max_iter=1000)
        elif self.model_name == "LDA":
            model_ML = LinearDiscriminantAnalysis()
        elif self.model_name == "SVM_linear":
            model_ML = SVC(kernel="linear")
        elif self.model_name == "SVM_rbf":
            model_ML = SVC(kernel="rbf")

        return model_ML

    def cross_val(self):

        scores = []

        for split_id in [1, 2, 3, 4, 5]:

            X_train = self.X_for_splits["train"][split_id - 1]
            Y_train = self.Y_for_splits["train"][split_id - 1]
            X_val = self.X_for_splits["val"][split_id - 1]
            Y_val = self.Y_for_splits["val"][split_id - 1]

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model_ML = self.init_ML_model()
            model_ML.fit(X_train_scaled, Y_train)
            Y_predicted = model_ML.predict(X_val_scaled)
            score = f1_score(y_true=Y_val, y_pred=Y_predicted, average="macro")

            scores.append(score)

        f1_mean = np.round(np.mean(scores), 3)
        f1_std = np.round(np.std(scores), 3)

        return f1_mean, f1_std

    def train_and_save_model(self, path_to_save="model_and_scaler\\"):

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.model_ML.fit(X_train_scaled, Y_train)

        Y_test_predicted = self.model_ML.predict(X_test_scaled)

        metric = f1_score(y_true=Y_test, y_pred=Y_test_predicted, average="macro")

        print(metric)

        # save model and scaler
        os.makedirs(path_to_save, exist_ok=True)
        pickle.dump(self.model_ML, open(path_to_save + f"model_{self.color_space}_{self.model_name}.sav", 'wb'))
        pickle.dump(scaler, open(path_to_save + f"scaler_{self.color_space}_{self.model_name}.sav", 'wb'))


if __name__ == "__main__":

    path_to_samples = root_path + "marker_colors_samples\\"

    # parameters fine-tuning
    rows = []
    for color_space in ["RGB", "HSV", "Lab", "Luv", "YCrCb"]:
        for model_name in ["LogReg", "LDA", "SVM_linear", "SVM_rbf"]:

            print(f"{color_space}\t{model_name}")

            model_semseg = ModelSemSegMarkers(
                color_space=color_space,
                model_name=model_name
            )

            model_semseg.read_data_to_arrays(path_to_samples=path_to_samples)
            # model_semseg.init_ML_model()
            f1_mean, f1_std = model_semseg.cross_val()

            print(f"{f1_mean}+-{f1_std}")
            print("--------------")

            row = dict()
            row["color_space"] = color_space
            row["model_name"] = model_name
            row["f1_mean"] = f1_mean
            row["f1_std"] = f1_std
            rows.append(row)

    df_results = pd.DataFrame(rows)
    df_results.to_csv("results_hyper_fine_tuning.csv", index=False, sep=";")

    """
    # train model under best settings
    color_space = "HSV"
    model_name = "SVM_rbf"

    model_semseg = ModelSemSegMarkers(
        color_space=color_space,
        model_name=model_name
    )

    model_semseg.read_data_to_arrays(path_to_samples=path_to_samples)
    model_semseg.init_ML_model()
    model_semseg.train_and_save_model()
    """
