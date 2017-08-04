#--------------------Imported Modules-------------------
import ConfigParser

import bcidataset
import preprocessing

import numpy as np
import math

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing as pp
#--------------------Class Definition start here---------------

class proccontrol:
	'Drives all the other code'

	def __init__(self, ConfigFile = None, TemporaryDirectory = None):
		self.config = ConfigParser.ConfigParser()
		if ConfigFile is None:
			self.config.readfp(open('bciconfig.txt'))
			self.ConfigFile = 'bciconfig.txt'
		else:
			self.config.readfp(open(ConfigFile))
			self.ConfigFile = ConfigFile
		if TemporaryDirectory is None:
			self.TemporaryDirectory = self.config.get('BCIData', 'TemporaryDirectory')
		else:
			self.TemporaryDirectory = TemporaryDirectory

	def EvaluateMSE(self):
		SubjectIDList = ['a','b','f','g']
		XfilternumArray = np.array([5])
		C_range = np.logspace(-3, 3, 7)
		gamma_range = np.logspace(-3, 3, 7)
		param_grid = dict(gamma=gamma_range, C=C_range)
		cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

		for SubID in SubjectIDList:
			for Xfiltnum in XfilternumArray:
				BCIdata = bcidataset.bcidataset(ConfigFile = self.ConfigFile, Xfilternum = Xfiltnum, SubjectID = SubID)
				PreProc = preprocessing.preprocessing(ConfigFile = self.ConfigFile, Xfilternum = Xfiltnum, SubjectID = SubID)
				if not BCIdata.CheckTempComputed():
					PreProc.MahmoudPreProcessingAlgorithm()
				BestFrequencyRange = BCIdata.LoadBestFrequencyRange()
				CSPfilters = BCIdata.LoadBestXFilters()
				#print BestFrequencyRange
				#print CSPfilters
				#PreProc.FilteredOutputDemo(BestFrequencyRange, CSPfilters)
				
				[FeatureVectors, OutputVector] = PreProc.GetFeatureVectorsTrain(BestFrequencyRange, CSPfilters)
				#print len(FeatureVectors), len(OutputVector)
				#print len(FeatureVectors[OutputVector == -1]), len(FeatureVectors[OutputVector == 1])
				#PreProc.FeatureDemo(FeatureVectors, OutputVector)
				scaler = pp.StandardScaler().fit(FeatureVectors)
				scaler.transform(FeatureVectors)

				grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)

				grid.fit(FeatureVectors, OutputVector)

				#print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
				del FeatureVectors, OutputVector

				[TestFeatureVectors, TestOutputVector] = PreProc.GetFeatureVectorsTest(BestFrequencyRange, CSPfilters)
				#print len(TestFeatureVectors), len(TestOutputVector)
				#print len(TestFeatureVectors[TestOutputVector == -1]), len(TestFeatureVectors[TestOutputVector == 1])

				scaler.transform(TestFeatureVectors)
				PredictedOutputVector = grid.predict(TestFeatureVectors)
				
				#print len(PredictedOutputVector), len(TestOutputVector)
				if len(PredictedOutputVector) != len(TestOutputVector):
					#print 'EEG data file and Class file Size Mismatch!'
					exit()

				MSE = 0
				count = 0
				checkdata = []
				for index, miclass in enumerate(TestOutputVector):
					if not math.isnan(miclass):
						count = count + 1
						if PredictedOutputVector[index] != miclass:
							MSE = MSE + (miclass-PredictedOutputVector[index])**2
				MSE = MSE / count

				printstr = 'NumFilters = %s | SubID = %s | MSE = %s' %(Xfiltnum, SubID, MSE)
				print printstr
		return;

#Default config file is used if nothing is passed.
pc = proccontrol()
pc.EvaluateMSE()