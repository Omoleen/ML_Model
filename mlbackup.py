import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QDialog, QApplication
from joblib import load
import pandas as pd

saved_model = load('Origin/cancer_race_gridsearch.joblib')
class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        uic.loadUi("ml.ui", self)
        self.predict.clicked.connect(self.runtest)

    def runtest(self):
        Tissue_Type = self.Tissue_Type.text()
        Race_Category = self.Race_Category.text()
        Mutation_Count = self.Mutation_Count.text()
        Age_at_Diagnosis = self.Age_at_Diagnosis.text()
        Smoking = self.Smoking.text()
        TMB_ = self.TMB_.text()
        M_Stage = self.M_Stage.text()
        #
        if len(Tissue_Type) == 0 or len(Race_Category) == 0 or len(Mutation_Count) == 0 or len(Age_at_Diagnosis) == 0 or len(Mutation_Count) == 0 or len(Smoking) == 0 or len(TMB_) == 0 or len(M_Stage) == 0:
            self.error.setText("Please input all fields")
        else:
            Tissue_Type = int(self.Tissue_Type.text())
            Race_Category = int(self.Race_Category.text())
            Mutation_Count = int(self.Mutation_Count.text())
            Age_at_Diagnosis = int(self.Age_at_Diagnosis.text())
            Smoking = int(self.Smoking.text())
            TMB_ = float(self.TMB_.text())
            M_Stage = int(self.M_Stage.text())
            data = {'Age_at_Diagnosis': [Age_at_Diagnosis],
                    'Mutation_Count': [Mutation_Count],
                    'M_Stage': [M_Stage],
                    'Race_Category': [Race_Category],
                    'Smoking': [Smoking],
                    'Tissue_Type': [Tissue_Type],
                    'TMB_(nonsynonymous)': [TMB_]}
            new = pd.DataFrame.from_dict(data)
            result = saved_model.predict(new)
            result = result[0]
            if result == 0:
                self.Result.setText("Will not Survive")
            elif result == 1:
                self.Result.setText("Will Survive")



app = QApplication(sys.argv)
mainwindow = Login()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(950)
widget.setFixedHeight(900)
widget.show()
app.exec_()
