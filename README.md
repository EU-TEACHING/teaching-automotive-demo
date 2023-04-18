# teaching-automotive-demo

Repository for the Simulator Study #2

## Stress Preprocessor

Data preprocessing classes for offline and online pipelines

### Run the offline preprocessor

First make sure `stress_preprocessor/config/offline_config.json` is up to date with the latest data schema information.    
Run the offline preprocessor: 

```
python offline_preprocess_run.py -sid [subj_id]
```

The -sid option is required and specifies the participant's ID.


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
Defined in 'stress_preprocessor/config/offline_config.json' with key "fod_feats". 
The features in this list are selected from EDA and ECG feature names. 

#### Final dataframe

```
final_df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 124000 entries, 0 to 15499
Data columns (total 38 columns):
 #   Column                            Non-Null Count   Dtype  
---  ------                            --------------   -----  
 0   Time                              124000 non-null  object 
 1   ScenarioID                        123999 non-null  object 
 2   RepetitionID                      123999 non-null  object 
 3   Stress                            123999 non-null  float32
 4   ECG_Raw                           123999 non-null  float32
 5   ECG_Clean                         124000 non-null  float64
 6   ECG_Rate                          124000 non-null  float64
 7   ECG_Quality                       124000 non-null  float64
 8   ECG_R_Peaks                       124000 non-null  int64  
 9   ECG_P_Peaks                       124000 non-null  int64  
 10  ECG_P_Onsets                      124000 non-null  int64  
 11  ECG_P_Offsets                     124000 non-null  int64  
 12  ECG_Q_Peaks                       124000 non-null  int64  
 13  ECG_R_Onsets                      124000 non-null  int64  
 14  ECG_R_Offsets                     124000 non-null  int64  
 15  ECG_S_Peaks                       124000 non-null  int64  
 16  ECG_T_Peaks                       124000 non-null  int64  
 17  ECG_T_Onsets                      124000 non-null  int64  
 18  ECG_T_Offsets                     124000 non-null  int64  
 19  ECG_Phase_Atrial                  123504 non-null  float64
 20  ECG_Phase_Completion_Atrial       124000 non-null  float64
 21  ECG_Phase_Ventricular             123504 non-null  float64
 22  ECG_Phase_Completion_Ventricular  124000 non-null  float64
 23  EDA_Raw                           123999 non-null  float32
 24  EDA_Clean                         124000 non-null  float64
 25  EDA_Tonic                         124000 non-null  float64
 26  EDA_Phasic                        124000 non-null  float64
 27  EDA_SCR_Onsets                    124000 non-null  int64  
 28  EDA_SCR_Peaks                     124000 non-null  int64  
 29  EDA_SCR_Height                    124000 non-null  float64
 30  EDA_SCR_Amplitude                 124000 non-null  float64
 31  EDA_SCR_RiseTime                  124000 non-null  float64
 32  EDA_SCR_Recovery                  124000 non-null  int64  
 33  EDA_SCR_RecoveryTime              124000 non-null  float64
 34  ECG_Raw_diff                      123998 non-null  float32
 35  EDA_Raw_diff                      123998 non-null  float32
 36  ECG_Clean_diff                    124000 non-null  float64
 37  EDA_Clean_diff                    124000 non-null  float64
dtypes: float32(5), float64(16), int64(14), object(3)
memory usage: 34.5+ MB
```

#### ECG AND eda PLOTS
Saved at `stress_preprocessor/graphs`

#### Remaining issues
1. Visualize raw time series, must be adapted from previous version
2. Error handling based on the error_col, replace row with null values
3. Imputation is currently handled by Neurokit2. Use visualization if something is really wrong to remove whole participant's data
4. Null values concerning the initial features exist in the final df 
