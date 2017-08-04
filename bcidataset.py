#--------------------Imported Modules-------------------
import ConfigParser
import re
import sys
import itertools
import numpy as np
import os.path

#--------------------Class Definition start here---------------

class bcidataset:
	'Takes the data from the ASCII format and puts it into variables for firther processing'

	def __init__(self, ConfigFile = None, DataDirectory = None, SamplingRate = None, SubjectID = None, TemporaryDirectory = None, Xfilternum = None):
		self.config = ConfigParser.ConfigParser()
		if ConfigFile is None:
			self.config.readfp(open('bciconfig.txt'))
			self.ConfigFile = 'bciconfig.txt'
		else:
			self.config.readfp(open(ConfigFile))
			self.ConfigFile = ConfigFile

		if DataDirectory is None:
			self.DataDirectory = self.config.get('BCIData', 'DataDirectory')
		else:
			self.DataDirectory = DataDirectory
		if SamplingRate is None:
			self.SamplingRate = int(self.config.get('BCIData', 'SamplingRate'))
		else:
			self.SamplingRate = int(SamplingRate)
		if SubjectID is None:
			self.SubjectID = self.config.get('BCIData', 'SubjectID')
		else:
			self.SubjectID = SubjectID
		if TemporaryDirectory is None:
			self.TemporaryDirectory = self.config.get('BCIData', 'TemporaryDirectory')
		else:
			self.TemporaryDirectory = TemporaryDirectory
		if Xfilternum is None:
			self.Xfilternum = int(self.config.get('MahmoudAlgorithm', 'Xfilternum'))
		else:
			self.Xfilternum = int(Xfilternum)

	def GetFeatureVectorPerSample(self, cntFile):
		with open(cntFile) as cntFileHandle:
			SampleList = cntFileHandle.readlines()
		
		SampleInputVectorList = []

		for Sample in SampleList:
			SampleVector = [float(s) for s in re.findall(r'[-+]?\d+', Sample)]
			SampleInputVectorList.append(SampleVector)

		return SampleInputVectorList;

	def GetClassVectorPerSample(self, mrkFile, NumSamples):
		with open(mrkFile) as mrkFileHandle:
			CueList = mrkFileHandle.readlines()

		SampleClassList = [float(0)] * NumSamples
		
		for Sample in CueList:
			SampleCue = [float(s) for s in re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", Sample)]
			SampleClassList[(int(SampleCue[0]) - 1):(int(SampleCue[0]) + 4*self.SamplingRate - 1)] = [SampleCue[1]]*(4*self.SamplingRate)

		return SampleClassList;

	def GetClassVectorPerSampleTest(self, evalmrkFile, NumSamples):
		with open(evalmrkFile) as evalmrkFileHandle:
			SampleList = evalmrkFileHandle.readlines()

		SampleClassListTest = [float('NaN')] * NumSamples 

		index = 0
		for Sample in SampleList:
			if (Sample != 'NaN\n') and (index % 10 == 0):
				SampleClassListTest[(index/10)] = float(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", Sample)[0])
			index = index + 1

		return SampleClassListTest;

	def ReadSubjectFile(self):
		cntFile = self.DataDirectory + '/' + 'BCICIV_calib_ds1' + self.SubjectID + '_cnt.txt'
		mrkFile = self.DataDirectory + '/' + 'BCICIV_calib_ds1' + self.SubjectID + '_mrk.txt'
		#col_idx = np.array([3,5,7,26,28,30])
		#SampleInputVectorList = FeatureVectorPerSample = self.GetFeatureVectorPerSample(cntFile)
		SampleInputVectorList = np.array(self.GetFeatureVectorPerSample(cntFile))
		#SampleInputVectorList = SampleInputVectorList[ :, col_idx]
		SampleClassList = self.GetClassVectorPerSample(mrkFile, len(SampleInputVectorList))

		return [SampleInputVectorList, SampleClassList];

	def ReadSubjectFileTest(self):
		evalcntFile = self.DataDirectory + '/' + 'BCICIV_eval_ds1' + self.SubjectID + '_cnt.txt'
		evalmrkFile = self.DataDirectory + '/' + 'BCICIV_eval_ds1' + self.SubjectID + '_1000Hz_true_y.txt'
		#col_idx = np.array([3,5,7,26,28,30])

		SampleInputVectorListTest = np.array(self.GetFeatureVectorPerSample(evalcntFile))
		#SampleInputVectorListTest = SampleInputVectorListTest[ :, col_idx]
		SampleClassListTest = self.GetClassVectorPerSampleTest(evalmrkFile, len(SampleInputVectorListTest))

		return [SampleInputVectorListTest, SampleClassListTest]

	def StoreBestFrequencyRange(self, BestFrequencyRange):
		BFRtextfile = self.TemporaryDirectory + '/' + 'BFR' + self.SubjectID + str(self.Xfilternum) + '.txt'
		np.savetxt(BFRtextfile, BestFrequencyRange)
		return;

	def StoreBestXFilters(self, CSPfilters):
		BXFtextfile = self.TemporaryDirectory + '/' + 'BXF' + self.SubjectID + str(self.Xfilternum) + '.txt'
		np.savetxt(BXFtextfile, CSPfilters)
		return;

	def LoadBestFrequencyRange(self):
		BFRtextfile = self.TemporaryDirectory +  '/' + 'BFR' + self.SubjectID + str(self.Xfilternum) + '.txt'
		BestFrequencyRange = np.loadtxt(BFRtextfile)
		return BestFrequencyRange;

	def LoadBestXFilters(self):
		BXFtextfile = self.TemporaryDirectory + '/' + 'BXF' + self.SubjectID + str(self.Xfilternum) + '.txt'
		CSPfilters = np.loadtxt(BXFtextfile)
		return CSPfilters;

	def CheckTempComputed(self):
		BFRtextfile = self.TemporaryDirectory + '/' + 'BFR' + self.SubjectID + str(self.Xfilternum) + '.txt'
		BXFtextfile = self.TemporaryDirectory + '/' + 'BXF' + self.SubjectID + str(self.Xfilternum) + '.txt'
		return os.path.isfile(BFRtextfile) and os.path.isfile(BXFtextfile);

# test = bcidataset()
# [SampleInputVectorListTest, SampleClassListTest] = test.ReadSubjectFileTest()
# print len(SampleInputVectorListTest), len(SampleClassListTest)