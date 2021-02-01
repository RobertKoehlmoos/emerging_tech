import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree, neighbors, ensemble, svm, discriminant_analysis, neural_network
from sklearn import model_selection
from sklearn import metrics
from typing import Tuple


def get_frames(power: np.ndarray) -> np.ndarray:
    """Returns a list of the lengths of the frames in this power series"""
    frames = []
    in_frame = False  # handles starting in the middle of a frame
    frame_start = 0
    for i in range(power.shape[0] - 1):
        if (not in_frame) and power[i] == 0 and power[i + 1] == 1:
            frame_start = i
            in_frame = True
        elif in_frame and power[i] == 1 and power[i + 1] == 0:
            frames.append(i - frame_start)
            in_frame = False
    return np.asarray(frames)


def get_features(power: np.ndarray) -> Tuple[float, float, float]:
    """Returns a list of features for the power series
    average frame length, frame length variance, percent transmitting
    """
    frames = get_frames(power)
    return frames.mean(), frames.var(), np.count_nonzero(power)/power.shape[0]


def make_dataset(num_splits=50):
    """Converts the signal captures into measurable features and stores them as a csv
    num_splits specifies how many observations each capture is split into"""

    data_files = ["droneA_cap_1Msps_20dbSNR.u8",
                  "droneB_cap_1Msps_20dbSNR.u8",
                  "random-signal_cap_1Msps_20dbSNR.u8",
                  "wifi_cap_1Msps_20dbSNR.u8"]
    classes = [name.split("_")[0] for name in data_files]  # getting user friendly class names
    observations = pd.DataFrame(columns=["Mean", "Variance", "Up", "Class"])
    captures = {source: np.fromfile(f"data/{file}", dtype=np.uint8)
                for source, file in zip(classes, data_files)}

    for source, power in captures.items():
        power_sections = np.split(power, num_splits)
        for section in power_sections:
            mean, var, percent_up = get_features(section)
            observations = observations.append({"Mean": mean, "Variance": var, "Up": percent_up, "Class": source},
                                               ignore_index=True)
    observations.to_csv(f"features_{num_splits}_splits.csv")


def evaluate_models(num_splits=10):
    """Test out of sample performance of 6 different models
    Returns a dictionary of classification reports for each model"""
    models = {"Decision Tree": tree.DecisionTreeClassifier(),
              "Nearest Neighbor": neighbors.KNeighborsClassifier(),
              "Random Forest": ensemble.RandomForestClassifier(),
              "Linear SVM": svm.SVC(kernel="linear"),  # the linear kernel shows best performance
              "LDA": discriminant_analysis.LinearDiscriminantAnalysis(),
              "Neural Net": neural_network.MLPClassifier(solver="lbfgs")}  # small datasets favor an lbfgs solver

    data = pd.read_csv(f"features_{num_splits}_splits.csv", index_col=[0])
    # All the models can achieve near perfect accuracy without normalization except for neural networks
    for feature in ["Mean", "Variance", "Up"]:
        data[feature] = (data[feature] - data[feature].mean())/data[feature].std()
    y = data["Class"]
    x = data.drop(["Class"], axis=1)

    # performing the model testing
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=1)
    performance = {}
    sources = ["droneA", "droneB", "wifi", "random-signal"]
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        report = metrics.classification_report(predictions, y_test, output_dict=True, zero_division=0)
        # The report gives summary results that are not used, so those are filtered out
        performance[name] = {source: report[source] for source in sources}

    return performance


def get_performance(splits=(10, 25, 50)) -> pd.DataFrame:
    """Takes an iterable of the number of splits for each signal capture
    returns a dataframe of the models' performance"""
    performance = {num_splits: evaluate_models(num_splits) for num_splits in splits}
    performance_flattened = {(model, num_splits, source, performance[num_splits][model][source]['f1-score'])
                             for num_splits in splits
                             for model in performance[num_splits]
                             for source in performance[num_splits][model]}
    return pd.DataFrame(performance_flattened, columns=("Model", "Splits", "Source", "F Score"))


def main():
    # splits = [10, 25, 50]
    # for num_splits in splits:
        # make_dataset(num_splits=num_splits)
        # features = pd.read_csv(f"features_{num_splits}_splits.csv", index_col=[0])
        # g = sns.pairplot(features, hue="Class")
        # g.map_diag(sns.histplot)
        # g.map_offdiag(sns.scatterplot)
        # g.add_legend()
        # plt.savefig(f"img/feature_facets_{num_splits}_splits.png")

    #get_performance(splits=splits).to_csv("model_performance.csv")

    performance_df = pd.read_csv("model_performance.csv")
    # Making values reader friendly
    source_map = {"droneA": "Drone A", "droneB": "Drone B", "random-signal": "Noise", "wifi": "Wifi"}
    performance_df["Source"] = performance_df["Source"].map(lambda x: source_map[x])
    performance_df["Sample Size"] = performance_df["Splits"].map(lambda x: f"{5/x:0.1f}s")

    # plotting performance
    plt.clf()
    g = sns.FacetGrid(performance_df, col="Sample Size", row="Model",
                      col_order=["0.1s", "0.2s", "0.5s"], margin_titles=True)
    g.map(sns.barplot, "Source", "F Score", order=["Drone A", "Drone B", "Wifi", "Noise"])
    plt.subplots_adjust(top=0.95)  # Value found by manual adjustment
    g.fig.suptitle("Signal Classification Performance")
    plt.savefig("img/performance.png")


if __name__ == "__main__":
    main()
