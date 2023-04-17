# teaching-automotive-demo

Repository for the Simulator Study #2

## Stress Preprocessor

Data preprocessing classes for both offline and online pipelines

### Run the offline preprocessor

After updating the config.py according to the latest data schema information,  
```
python offline_preprocess_run.py -sid [subj_id] -sp [save_path]
```

Make sure to replace [subj_id] with the actual ID of the participant and [save_path] with the path where you want to
save the processed dataframes. The -sid option is required and specifies the participant's ID, while the -sp option is
optional and specifies the path to save the processed dataframes. If the -sp option is not provided, the processed
dataframes will be saved in the current working directory.

### Extracted features

Neurokit2 is used for the ECG and EDA feature extraction. 

#### ECG
- `"ECG_Raw"`: the raw signal.
- `"ECG_Clean"`: the cleaned signal.
- `"ECG_R_Peaks"`: the R-peaks marked as “1” in a list of zeros.
- `"ECG_Rate"`: heart rate interpolated between R-peaks.
- `"ECG_P_Peaks"`: the P-peaks marked as “1” in a list of zeros
- `"ECG_Q_Peaks"`: the Q-peaks marked as “1” in a list of zeros .
- `"ECG_S_Peaks"`: the S-peaks marked as “1” in a list of zeros.
- `"ECG_T_Peaks"`: the T-peaks marked as “1” in a list of zeros.
- `"ECG_P_Onsets"`: the P-onsets marked as “1” in a list of zeros.
- `"ECG_P_Offsets"`: the P-offsets marked as “1” in a list of zeros (only when method in ecg_delineate() is wavelet).
- `"ECG_T_Onsets"`: the T-onsets marked as “1” in a list of zeros (only when method in ecg_delineate() is wavelet).
- `"ECG_T_Offsets"`: the T-offsets marked as “1” in a list of zeros.
- `"ECG_R_Onsets"`: the R-onsets marked as “1” in a list of zeros (only when method in ecg_delineate() is wavelet).
- `"ECG_R_Offsets"`: the R-offsets marked as “1” in a list of zeros (only when method in ecg_delineate() is wavelet).
- `"ECG_Phase_Atrial"`: cardiac phase, marked by “1” for systole and “0” for diastole.
- `"ECG_Phase_Ventricular"`: cardiac phase, marked by “1” for systole and “0” for diastole.
- `"ECG_Atrial_PhaseCompletion"`: cardiac phase (atrial) completion, expressed in percentage (from 0 to 1), representing the stage of the current cardiac phase.
- `"ECG_Ventricular_PhaseCompletion"`: cardiac phase (ventricular) completion, expressed in percentage (from 0 to 1), representing the stage of the current cardiac phase.
```
ecg_feats_dfs[0].columns
Out[2]: 
Index(['ECG_Raw', 'ECG_Clean', 'ECG_Rate', 'ECG_Quality', 'ECG_R_Peaks',
       'ECG_P_Peaks', 'ECG_P_Onsets', 'ECG_P_Offsets', 'ECG_Q_Peaks',
       'ECG_R_Onsets', 'ECG_R_Offsets', 'ECG_S_Peaks', 'ECG_T_Peaks',
       'ECG_T_Onsets', 'ECG_T_Offsets', 'ECG_Phase_Atrial',
       'ECG_Phase_Completion_Atrial', 'ECG_Phase_Ventricular',
       'ECG_Phase_Completion_Ventricular'],
      dtype='object')
```
#### EDA
-    `"EDA_Raw"`: the raw signal.
-    `"EDA_Clean"`: the cleaned signal.
-    `"EDA_Tonic"`: the tonic component of the signal, or the Tonic Skin Conductance Level (SCL).
-    `"EDA_Phasic"`: the phasic component of the signal, or the Phasic Skin Conductance Response (SCR).
-    `"SCR_Onsets"`: the samples at which the onsets of the peaks occur, marked as “1” in a list of zeros.
-    `"SCR_Peaks"`: the samples at which the peaks occur, marked as “1” in a list of zeros.
-    `"SCR_Height"`: the SCR amplitude of the signal including the Tonic component. Note that cumulative effects of close-occurring SCRs might lead to an underestimation of the amplitude.
-    `"SCR_Amplitude"`: the SCR amplitude of the signal excluding the Tonic component.
-    `"SCR_RiseTime"`: the time taken for SCR onset to reach peak amplitude within the SCR.
-    `"SCR_Recovery"`: the samples at which SCR peaks recover (decline) to half amplitude, marked as “1” in a list of zeros.

```
eda_feats_dfs[0].columns
Out[4]: 
Index(['EDA_Raw', 'EDA_Clean', 'EDA_Tonic', 'EDA_Phasic', 'EDA_SCR_Onsets',
       'EDA_SCR_Peaks', 'EDA_SCR_Height', 'EDA_SCR_Amplitude',
       'EDA_SCR_RiseTime', 'EDA_SCR_Recovery', 'EDA_SCR_RecoveryTime'],
      dtype='object')
```
#### First order differences

#### All new features

```
new_feats_dfs[0].columns
Out[5]: 
Index(['ECG_Raw', 'ECG_Clean', 'ECG_Rate', 'ECG_Quality', 'ECG_R_Peaks',
       'ECG_P_Peaks', 'ECG_P_Onsets', 'ECG_P_Offsets', 'ECG_Q_Peaks',
       'ECG_R_Onsets', 'ECG_R_Offsets', 'ECG_S_Peaks', 'ECG_T_Peaks',
       'ECG_T_Onsets', 'ECG_T_Offsets', 'ECG_Phase_Atrial',
       'ECG_Phase_Completion_Atrial', 'ECG_Phase_Ventricular',
       'ECG_Phase_Completion_Ventricular', 'EDA_Raw', 'EDA_Clean', 'EDA_Tonic',
       'EDA_Phasic', 'EDA_SCR_Onsets', 'EDA_SCR_Peaks', 'EDA_SCR_Height',
       'EDA_SCR_Amplitude', 'EDA_SCR_RiseTime', 'EDA_SCR_Recovery',
       'EDA_SCR_RecoveryTime', 'ECG_Raw_diff', 'EDA_Raw_diff',
       'ECG_Clean_diff', 'EDA_Clean_diff'],
      dtype='object')
```

