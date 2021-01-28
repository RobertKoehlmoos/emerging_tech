import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_frames(power: np.array) -> np.array:
    """Returns a list of the lengths of the frames in this power series"""
    frames = []
    in_frame = False
    frame_start = 0
    for i in range(power.shape[0] - 1):
        if (not in_frame) and power[i] == 0 and power[i + 1] == 1:
            frame_start = i
            in_frame = True
        elif in_frame and power[i] == 1 and power[i + 1] == 0:
            frames.append(i - frame_start)
            in_frame = False
    return np.asarray(frames)


def get_features(power: np.array) -> (float, float, float):
    """Returns a list of features for the power series
    average frame length, frame length variance, percent transmitting
    """
    frames = get_frames(power)
    return frames.mean(), frames.var(), np.count_nonzero(power)/power.shape[0]


def get_features():
    data_files = ["droneA_cap_1Msps_20dbSNR.u8",
                  "droneB_cap_1Msps_20dbSNR.u8",
                  "random-signal_cap_1Msps_20dbSNR.u8",
                  "wifi_cap_1Msps_20dbSNR.u8"]
    observations = pd.DataFrame(columns=["Mean", "Variance", "Up", "Class"])
    for file in data_files:
        power = np.fromfile(f"data/{file}", dtype=np.uint8)
        power_sections = np.split(power, 10)
        for section in power_sections:
            mean, var, percent_up = get_features(section)
            observations = observations.append({"Mean": mean, "Variance": var, "Up": percent_up, "Class": file}, ignore_index=True)
    observations.to_csv("features.csv")


def main():
    #get_features()
    features = pd.read_csv("features.csv", index_col=[0])
    features["Class"] = features["Class"].map(lambda x: x.split("_")[0])
    g = sns.pairplot(features, hue="Class")
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    plt.savefig("img/feature_facets.png")

if __name__ == "__main__":
    main()
