import os
import numpy as np
import cv2
import random
import pickle
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
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

        X = []
        Y = []
        for class_name in os.listdir(path_to_samples):
            for filename in os.listdir(path_to_samples + class_name):

                img_oryg = cv2.imread(path_to_samples + f"{class_name}\\{filename}")
                img = self.change_color_space(img_oryg)
                for x_loc in range(img.shape[0]):
                    for y_loc in range(img.shape[1]):
                        if np.sum(img_oryg[x_loc, y_loc, :]) != 0:
                            if random.random() <= 1:
                                X.append(list(img[x_loc, y_loc, :]))
                                Y.append(class_name)
        self.X = np.array(X)
        self.Y = np.array(Y)
        print(f"{self.X.shape}\t{self.Y.shape}")

    def init_ML_model(self):

        if self.model_name == "LogReg":
            self.model_ML = LogisticRegression(max_iter=1000)
        elif self.model_name == "LDA":
            self.model_ML = LinearDiscriminantAnalysis()
        elif self.model_name == "SVM_linear":
            self.model_ML = SVC(kernel="linear")
        elif self.model_name == "SVM_rbf":
            self.model_ML = SVC(kernel="rbf")
        self.pipe = Pipeline([('scaler', StandardScaler()), ('model_ML', self.model_ML)])

    def cross_val(self):

        cv = StratifiedKFold(n_splits=5, shuffle=True)
        scores = cross_val_score(
            estimator=self.pipe,
            X=self.X,
            y=self.Y,
            scoring="f1_macro",
            cv=cv
        )

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

    """
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
            model_semseg.init_ML_model()
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
