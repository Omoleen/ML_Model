import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from tkinter import *
import threading
import os

result = ""
screen = Tk()
screen.geometry("1000x750")
screen.title("Predict Likelihood of death in Head and Neck cancer")


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def predict():
    dataset = pd.read_csv(resource_path('hnsc - hnsc_tcga_pan_can_atlas_2018_clinical_data.csv'))
    dataset = dataset.drop(['Study ID', 'Patient ID', 'Sample ID', 'Neoplasm Disease Stage American Joint Committee on Cancer Code', 'American Joint Committee on Cancer Publication Version Type', 'Cancer Type', 'TCGA PanCanAtlas Cancer Type Acronym', 'Cancer Type Detailed', 'Tissue Source Site', 'Tissue Source Site Code', 'Tumor Disease Anatomic Site', 'Tumor Type', 'Patient Weight', 'Disease Free (Months)', 'Last Communication Contact from Initial Pathologic Diagnosis Date', 'Disease Free Status', 'Ethnicity Category', 'Somatic Status'],  axis=1)
    dataset = dataset.drop(['Form completion date'], axis=1)
    dataset = dataset.drop(['Neoplasm Histologic Grade'], axis=1)
    dataset = dataset.drop(['Subtype'], axis=1)
    dataset = dataset.drop(['Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value', 'Neoadjuvant Therapy Type Administered Prior To Resection Text', 'ICD-10 Classification', 'Prior Diagnosis', 'Race Category', 'Radiation Therapy', 'Sample Type', 'Tissue Prospective Collection Indicator', 'Tissue Retrospective Collection Indicator'], axis=1)
    dataset = dataset.drop(['International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code', 'International Classification of Diseases for Oncology, Third Edition ICD-O-3 Site Code', 'Informed consent verified', 'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code', 'American Joint Committee on Cancer Tumor Stage Code', 'Progression Free Status', 'Primary Lymph Node Presentation Assessment'], axis=1)
    dataset['Sex'].replace(['Male', 'Female'], [0, 1], inplace=True)
    dataset['Person Neoplasm Cancer Status'].replace(['Tumor Free', 'With Tumor'], [0, 1], inplace=True)
    dataset['Overall Survival Status'].replace(['0:LIVING', '1:DECEASED'], [0, 1], inplace=True)
    dataset = dataset.drop(['Disease-specific Survival status', 'In PanCan Pathway Analysis', 'Other Patient ID', 'American Joint Committee on Cancer Metastasis Stage Code'], axis=1)
    dataset = dataset.drop(['New Neoplasm Event Post Initial Therapy Indicator', 'Oncotree Code'], axis=1)
    dataset = dataset.drop(['Number of Samples Per Patient'], axis=1)
    dataset = dataset.dropna()
    print(dataset.loc[dataset.index[0]])
    
    X = dataset.drop(['Overall Survival Status'], axis=1)
    Y = dataset['Overall Survival Status']
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    print(X.shape)
    print(Y.shape)
    clf = RandomForestClassifier()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf.fit(X_train, Y_train)
    print(clf.predict(X_test))
    print(Y_test)

    print(fgaa)
    

    input_data = [agee.get(), anss.get(), bhss.get(), nbdd.get(), monthss.get(), fgaa.get(), msii.get(), msss.get(), mcc.get(), monthss.get(), pncss.get(), monthss.get(), rhss.get(), gg.get(), tmbb.get(), whss.get()]
    
    prediction = clf.predict([input_data])
    print(prediction)
    

    if (prediction[0] == 0):
        result = "This person is alive and most likely going to live"
        result_label = Label(text = result, fg = "green", width = "100")
        result_label.place(x = 140, y = 525)
        print("This person is alive and most likely going to live")
    else:
        result = "This person is dead or is most likely going to die"
        result_label = Label(text = result, fg = "red", width = "100")
        result_label.place(x = 140, y = 525)
        screen.update()
        print("This person is dead or is most likely going to die")
    


heading = Label(text="Predict Likelihood of death in Head and Neck cancer", bg="white", fg="black")
heading.pack()


age = Label(text="Age *")
ans = Label(text="Aneuploidy Score *")
bhs = Label(text="Buffer Hypoxia Score *")
nbd = Label(text="Number of days from birth to diagnosis(Negative) *")
months = Label(text="Months patient has had the cancer for *")
fga = Label(text="Fraction genome altered*")
msi = Label(text="MSI MANTIS score *")
mss = Label(text="MSI Sensor score *")
mc = Label(text="Mutation count *")
pncs = Label(text="Person Neoplasm Cancer Status (0 for tumor free, 1 for tumor presence) *")
rhs = Label(text="Ragnum Hypoxia score *")
g = Label(text="Gender (0 for Male, 1 for Female) *")
tmb = Label(text="TMB *")
whs = Label(text="Winter Hypoxia Score *")
age.place(x = 15, y = 50)
ans.place(x = 15, y = 100)
bhs.place(x = 15, y = 150)
nbd.place(x = 15, y = 200)
months.place(x = 15, y = 250)
fga.place(x = 15, y = 300)
msi.place(x = 15, y = 350)
mss.place(x = 15, y = 400)
mc.place(x = 15, y = 450)
pncs.place(x = 550, y = 50)
rhs.place(x = 550, y = 100)
g.place(x = 550, y = 150)
tmb.place(x = 550, y = 200)
whs.place(x = 550, y = 250)

agee = DoubleVar()
anss = DoubleVar()
bhss = DoubleVar()
nbdd = DoubleVar()
monthss = DoubleVar()
fgaa = DoubleVar()
msii = DoubleVar()
msss = DoubleVar()
mcc = DoubleVar()
pncss = DoubleVar()
rhss = DoubleVar()
gg = DoubleVar()
tmbb = DoubleVar()
whss = DoubleVar()

age_entry = Entry(textvariable = agee)
ans_entry = Entry(textvariable = anss)
bhs_entry = Entry(textvariable = bhss)
nbd_entry = Entry(textvariable = nbdd)
month_entry = Entry(textvariable = monthss)
fga_entry = Entry(textvariable = fgaa)
msi_entry = Entry(textvariable = msii)
mss_entry = Entry(textvariable = msss)
mc_entry = Entry(textvariable = mcc)
pncs_entry = Entry(textvariable = pncss)
rhs_entry = Entry(textvariable = rhss)
g_entry = Entry(textvariable = gg)
tmb_entry = Entry(textvariable = tmbb)
whs_entry = Entry(textvariable =whss)

age_entry.place(x = 15, y = 75)
ans_entry.place(x = 15, y = 125)
bhs_entry.place(x = 15, y = 175)
nbd_entry.place(x = 15, y = 225)
month_entry.place(x = 15, y = 275)
fga_entry.place(x = 15, y = 325)
msi_entry.place(x = 15, y = 375)
mss_entry.place(x = 15, y = 425)
mc_entry.place(x = 15, y = 475)
pncs_entry.place(x = 550, y = 75)
rhs_entry.place(x = 550, y = 125)
g_entry.place(x = 550, y = 175)
tmb_entry.place(x = 550, y = 225)
whs_entry.place(x = 550, y = 275)

result_label = Label(text = result)
result_label.place(x = 140, y = 550)

predict = Button(text = "Predict", width="30", height="2", command = predict)
predict.place(x = 375, y = 620)



screen.mainloop()
