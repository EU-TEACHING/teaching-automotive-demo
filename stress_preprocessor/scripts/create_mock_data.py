import pandas as pd
import os
import numpy as np

# get the parent directory of the current working directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

ecg_gsr_path = r"automotive_study_2/SUBJ_01_DATA/Scenario_01.Case_set_1.Scenario01_Sport/results/icos/Subsystem_1.Sensors_IN_res.csv"
cond_path = r"automotive_study_2/SUBJ_01_DATA/Scenario_01.Case_set_1.Scenario01_Sport/results/icos/Subsystem_1.Sim_State_in.csv"
err_path = r"automotive_study_2/SUBJ_01_DATA/Scenario_01.Case_set_1.Scenario01_Sport/results/icos/Subsystem_1.Error_Report_res.csv"

ecg_gsr_path = os.path.join(parent_dir, ecg_gsr_path)
cond_path = os.path.join(parent_dir, cond_path)
err_path = os.path.join(parent_dir, err_path)

df_ecg_gsr = pd.read_csv(ecg_gsr_path, header=1, sep=';')
df_cond = pd.read_csv(cond_path, header=1, sep=';')
df_err = pd.read_csv(err_path, header=1, sep=';')

# drop the first row (index 0)
df_ecg_gsr = df_ecg_gsr.drop(0)
df_cond = df_cond.drop(0)
df_err = df_err.drop(0)

df_cond = df_cond.loc[:, ["Time", "RepetitionID", "ScenarioID", "subID"]]
df_ecg_gsr = df_ecg_gsr.loc[:, ["Time", "ECG", "GSR"]]

# merge the DataFrames on the 'Time' column, preserving the original column names
merged_df = pd.merge(df_cond, df_ecg_gsr, on='Time', how='outer').merge(df_err, on='Time', how='outer')

# check if any missing values exist in the merged DataFrame
if merged_df.isnull().any().any():
    print("Missing values exist in the merged DataFrame")
else:
    print("No missing values exist in the merged DataFrame")

# add mock stress col
random_values = np.random.randint(0, 101, size=len(merged_df))
merged_df['Stress'] = random_values

save_path = r"automotive_study_2/SUBJ_01_DATA/Scenario_01.Case_set_1.Scenario01_Sport/results/icos/SUBJ_01_MOCK_01.csv"
save_path = os.path.join(parent_dir, save_path)

merged_df.to_csv(save_path, index=False, sep=';')



