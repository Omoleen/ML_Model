from joblib import dump, load
import pandas as pd


data = {'Age_at_Diagnosis': [46],
        'Mutation_Count': [2],
        'M_Stage': [0],
        'Race_Category': [2],
        'Smoking': [2],
        'Tissue_Type': [3],
        'TMB_(nonsynonymous)': [0.35564]}
new = pd.DataFrame.from_dict(data)


saved_model = load('Origin/cancer_race_gridsearch.joblib')
result = saved_model.predict(new)
result = result[0]
if result == 0:
        print('Dead')
elif result == 1:
        print('Alive')
# print(result)
# print("After save:",saved_model.predict(new))

# Race:
# Black = 0
# White = 1
# Asian = 2
#
# Smoking:
# Never = 0
# Current = 1
# Former = 2
#
# Status:
# Dead = 0
# Alive = 1
#
# Tissue:
# Liver = 0
# Lymph_Node = 1
# Bone = 2
# Prostate = 3
# Other_soft_tissue = 4
# Lung = 5
#
# Status:
# M0 = 0
# M1 = 1