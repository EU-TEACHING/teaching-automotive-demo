import os, warnings

warnings.filterwarnings(action="ignore")

from typing import List, Literal, Optional
import pandas as pd

import numpy as np
import torch

from scipy.stats import zscore
from sklearn.model_selection import train_test_split

DATA_DIR = "/raid/decaro/datasets/processed/AVLStudy"
SCENARIOS = [i for i in range(1, 12)]
STRESS_QUEST = [f"S{i}M1Q8" for i in range(1, 7)]


class AVLDataset(torch.utils.data.Dataset):
    SCENARIOS = {
        "all": [i for i in range(1, 12)],
        "train": [3, 5, 6, 7, 11, 10],
        "eval": [4, 8, 9],
        "test": [1, 2],
    }

    def __init__(
        self,
        user_id: str,
        split: str,
        columns: List[str],
        norm: Literal["baseline", "sequence"],
        classification: bool,
    ):
        self.user_id = user_id
        self.scenario_ids = AVLDataset.SCENARIOS[split]
        self.norm = norm
        self.columns = columns
        self.classification = classification
        self.user_path = f"{DATA_DIR}/SUBJ_{user_id}/"

        ft_path = os.path.join(self.user_path, f"SUBJ_{user_id}_ALL_SCENARIOS.csv")
        self.features = pd.read_csv(ft_path)
        self.features = self.features[
            [
                "Time",
                "ScenarioID",
                "Slider_value",
            ]
            + self.columns
        ]
        self.preprocess()

    def __len__(self):
        return len(self.scenario_ids)

    def __getitem__(self, idx):
        scenario_data = self.features[
            (self.features["ScenarioID"] == self.scenario_ids[idx])
        ]

        if self.norm == "sequence":
            features = (features - np.mean(features)) / np.std(features)

        scenario_data = scenario_data.iloc[50:]
        features = scenario_data[self.columns].values

        if self.classification:
            target = scenario_data["stressed"].values
        else:
            target = scenario_data["Slider_value"].values

        return torch.tensor(features, dtype=torch.float), torch.tensor(
            target, dtype=torch.float
        ).unsqueeze(1)

    def preprocess(self):
        self.features["Slider_value"] /= 1024
        if self.norm == "baseline":
            bs_path = os.path.join(
                self.user_path, f"SUBJ_{self.user_id}_SCEN_00_MODE_FreeDriving.csv"
            )
            baseline = pd.read_csv(bs_path).iloc[50:]
            baseline = baseline[self.columns]
            baseline = baseline.values

            mean, std = np.mean(baseline, axis=0), np.std(baseline, axis=0)
            self.features[self.columns] = (self.features[self.columns] - mean) / std
        # if self.classification:
        #     for i in self.scenario_ids:
        #         scenario_data = self.features[self.features["ScenarioID"] == i]
        #         scenario_data["slider_diff"] = scenario_data["Slider_value"].diff()
        #         scenario_data["slider_diff"].iloc[:2] = scenario_data[
        #             "slider_diff"
        #         ].iloc[2]
        #         scenario_data["slider_diff_rolling_sum"] = (
        #             scenario_data["slider_diff"].rolling(500).sum()
        #         )
        #         scenario_data["stressed"] = scenario_data.apply(to_class, axis=1)
        #         scenario_data["stressed"].iloc[0] = 0

        #         scenario_data["stressed"] = scenario_data["stressed"].fillna(
        #             method="ffill"
        #         )
        #         self.features.loc[
        #             self.features["ScenarioID"] == i, "stressed"
        #         ] = scenario_data["stressed"].values


def to_class(x):
    if x["slider_diff_rolling_sum"] > 0.05:
        return 1
    elif x["slider_diff_rolling_sum"] < -0.05:
        return 0
    else:
        return np.NaN


class AVLDatasetV1(torch.utils.data.Dataset):
    USERS = [
        15,
        50,
        1001,
        1004,
        1017,
        1023,
        1044,
        1056,
        1089,
        2001,
        2017,
        2066,
        2078,
        2099,
        2101,
        2111,
        2180,
        3001,
        3002,
        3049,
        3050,
        3080,
        3099,
        3101,
        4001,
        4002,
        4010,
        4030,
        4040,
        4069,
        4087,
        4099,
    ]

    def __init__(
        self,
        users: Literal["train", "eval", "test", "all"] = "all",
        scenarios: Literal["train", "eval", "test", "all"] = "all",
        drop_features: Optional[List[str]] = ["ecg_data_diff", "gsr_data"],
        sequence_prediction: bool = False,
    ) -> None:
        self.users_type = users
        self.scenarios_type = scenarios
        self.sequence_prediction = sequence_prediction

        features = pd.read_csv(DATA_DIR + "merged/ecg_gsr_scenario_cleaned_v5.csv")
        targets = pd.read_csv(DATA_DIR + "quest/raw.csv", sep=";")

        if drop_features is not None:
            features = features.drop(drop_features, axis=1)

        targets = targets[["prob_id"] + STRESS_QUEST]
        targets[STRESS_QUEST] = targets[STRESS_QUEST] / 5

        features = features[features["prob_id"].isin(AVLDataset.USERS)]
        targets = targets[targets["prob_id"].isin(AVLDataset.USERS)]
        # features[["gsr_data_diff", "ecg_data"]] = features[
        #     ["gsr_data_diff", "ecg_data"]
        # ].apply(zscore)

        self.features, self.targets = self._get_split(
            features, targets, self.users_type, self.scenarios_type
        )

    def __len__(self):
        return len(self.users) * len(self.scenarios)

    def __getitem__(self, idx):
        user_idx = idx // len(self.scenarios)
        prob_id = self.users[user_idx]
        scenario_id = idx % len(self.scenarios)

        features = self.features[
            (self.features["prob_id"] == prob_id)
            & (self.features["scenario_id"] == self.scenarios[scenario_id])
        ].drop(["scenario_id", "prob_id", "ts_corr_time_offset"], axis=1)
        target = (
            features["events"].apply(lambda x: 0 if x == "normal" else 1).values[10:]
        )
        features = features.drop(["events"], axis=1)
        features = features.values[10:]

        if not self.sequence_prediction:
            features = torch.tensor(features)
            mean, std = torch.mean(features, dim=0, keepdim=True), torch.std(
                features, dim=0, keepdim=True
            )
            features = (features - mean) / std
            target = torch.tensor(target)
            target[20:50] = 1
            target = target.unsqueeze(1)

        else:
            features = np.concatenate(
                [
                    np.mean(features, axis=0),
                    np.std(features, axis=0),
                ],
                axis=0,
            )
            target = self.targets[self.targets["prob_id"] == prob_id]
            target = np.round(target[STRESS_QUEST[scenario_id]].values[0])
        return features, target

    def _get_split(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        users_type: Literal["all", "train", "eval", "test"],
        scenarios_type: Literal["all", "train", "eval", "test"],
    ):
        if users_type != "all":
            labels = targets[STRESS_QUEST].values
            labels = np.round(np.mean(labels, axis=1))

            train_users, test_users = train_test_split(
                AVLDataset.USERS, test_size=0.2, stratify=labels, random_state=1
            )
            labels = targets[targets["prob_id"].isin(train_users)][STRESS_QUEST].values
            labels = np.round(np.mean(labels, axis=1))
            train_users, val_users = train_test_split(
                train_users, test_size=0.2, stratify=labels, random_state=2
            )

            if users_type == "train":
                self.users = train_users
            elif users_type == "eval":
                self.users = val_users
            elif users_type == "test":
                self.users = test_users

            self.users = sorted(self.users)
            features = features[features["prob_id"].isin(self.users)]
            targets = targets[targets["prob_id"].isin(self.users)]
        else:
            self.users = AVLDataset.USERS

        if scenarios_type != "all":
            if scenarios_type == "train":
                self.scenarios = [1, 2, 3, 4]
            elif scenarios_type == "eval":
                self.scenarios = [5]
            elif scenarios_type == "test":
                self.scenarios = [6]

            features = features[features["scenario_id"].isin(self.scenarios)]
            targets = targets[targets["prob_id"].isin(self.users)]
        else:
            self.scenarios = [1, 2, 3, 4, 5, 6]

        return features, targets


def seq_collate_fn(model: str = "esn"):
    if model == "esn":

        def aux_fn(batch):
            features, targets = zip(*batch)
            return torch.stack(features, 1), torch.stack(targets)

    elif model == "rnn":

        def aux_fn(batch):
            features, targets = zip(*batch)
            # pad features sequence with zeros
            features = torch.nn.utils.rnn.pad_sequence(features)
            return features, torch.stack(targets)

    return aux_fn
