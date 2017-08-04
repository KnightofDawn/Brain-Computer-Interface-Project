#--------------------Imported Modules-------------------
import ConfigParser

import bcidataset
import numpy as np
import eegtools

from scipy.signal import butter, lfilter
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import pylab

#--------------------Class Definition start here---------------

class preprocessing:
	'Performs the preprocessing functions for the BCI EEG data'

	def __init__(self, ConfigFile = None, CovariancePeriod = None, SkipPeriod = None, NumFilters = None, Order = None, NumBinsPerDim = None, SamplingRate = None, Xfilternum = None, SubjectID = None):
		self.config = ConfigParser.ConfigParser()
		if ConfigFile is None:
			self.config.readfp(open('bciconfig.txt'))
			self.ConfigFile = 'bciconfig.txt'
		else:
			self.config.readfp(open(ConfigFile))
			self.ConfigFile = ConfigFile


		if CovariancePeriod is None:
			self.CovariancePeriod = int(self.config.get('CSP', 'CovariancePeriod'))
		else:
			self.CovariancePeriod = int(CovariancePeriod)
		if SkipPeriod is None:
			self.SkipPeriod = int(self.config.get('CSP', 'SkipPeriod'))
		else:
			self.SkipPeriod = int(SkipPeriod)
		if NumFilters is None:
			self.NumFilters = int(self.config.get('CSP', 'NumFilters'))
		else:
			self.NumFilters = int(NumFilters)
		if Order is None:
			self.Order = int(self.config.get('Filtering', 'Order'))
		else:
			self.Order = int(Order)
		if NumBinsPerDim is None:
			self.NumBinsPerDim = int(self.config.get('ProbabilityDensity', 'NumBinsPerDim'))
		else:
			self.NumBinsPerDim = int(NumBinsPerDim)
		if SamplingRate is None:
			self.SamplingRate = int(self.config.get('BCIData', 'SamplingRate'))
		else:
			self.SamplingRate = int(SamplingRate)
		if Xfilternum is None:
			self.Xfilternum = int(self.config.get('MahmoudAlgorithm', 'Xfilternum'))
		else:
			self.Xfilternum = int(Xfilternum)
		if SubjectID is None:
			self.SubjectID = self.config.get('BCIData', 'SubjectID')
		else:
			self.SubjectID = SubjectID

	def TrainCSPfilters(self, SampleInputArray, SampleClassArray):
		[AverageCovarClassOneTrials, AverageCovarClassTwoTrials] = self.GetAverageCovar(SampleInputArray, SampleClassArray)
		CSPfilters = eegtools.spatfilt.csp(AverageCovarClassOneTrials, AverageCovarClassTwoTrials, self.NumFilters)
		return CSPfilters;

	def GetAverageCovar(self, SampleInputArray, SampleClassArray):
		NumChannels = SampleInputArray.shape[1]

		AverageCovarClassOneTrials = np.zeros((NumChannels, NumChannels))
		AverageCovarClassTwoTrials = np.zeros((NumChannels, NumChannels))
		SampleData = np.zeros((NumChannels, self.CovariancePeriod))
		NumTrialsOne = 0
		NumTrialsTwo = 0
		SampleIndex = 0

		while SampleIndex <= SampleInputArray.shape[0] - self.CovariancePeriod:
			if SampleClassArray[SampleIndex] == -1:
				SampleIndex = SampleIndex + 1*self.SamplingRate
				SampleData = SampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SpatCovar = np.mat(SampleData) * np.mat(SampleData.transpose())
				AverageCovarClassOneTrials = AverageCovarClassOneTrials + np.array(SpatCovar / (np.trace(SpatCovar)))
				NumTrialsOne = NumTrialsOne + 1

				SampleIndex = SampleIndex + self.CovariancePeriod

				SampleData = SampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SpatCovar = np.mat(SampleData) * np.mat(SampleData.transpose())
				AverageCovarClassOneTrials = AverageCovarClassOneTrials + np.array(SpatCovar / (np.trace(SpatCovar)))
				NumTrialsOne = NumTrialsOne + 1

				SampleIndex = SampleIndex + self.CovariancePeriod

			elif SampleClassArray[SampleIndex] == 1:
				SampleIndex = SampleIndex + 1*self.SamplingRate
				SampleData = SampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SpatCovar = np.mat(SampleData) * np.mat(SampleData.transpose())
				AverageCovarClassTwoTrials = AverageCovarClassTwoTrials + np.array(SpatCovar / (np.trace(SpatCovar)))
				NumTrialsTwo = NumTrialsTwo + 1

				SampleIndex = SampleIndex + self.CovariancePeriod

				SampleData = SampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SpatCovar = np.mat(SampleData) * np.mat(SampleData.transpose())
				AverageCovarClassTwoTrials = AverageCovarClassTwoTrials + np.array(SpatCovar / (np.trace(SpatCovar)))
				NumTrialsTwo = NumTrialsTwo + 1

				SampleIndex = SampleIndex + self.CovariancePeriod
			else:
				SampleIndex = SampleIndex + 1

		AverageCovarClassOneTrials = AverageCovarClassOneTrials / NumTrialsOne
		AverageCovarClassTwoTrials = AverageCovarClassTwoTrials / NumTrialsTwo

		return [AverageCovarClassOneTrials, AverageCovarClassTwoTrials];

	def BandPassFilter(self, lowfreq, highfreq, UnfilteredData):
		FilteredData = np.zeros(UnfilteredData.shape)
		NumChannels = UnfilteredData.shape[1]

		NyquistFrequency = 0.5 * float(self.SamplingRate)
		normlowfreq = float(lowfreq) / NyquistFrequency
		normhighfreq = float(highfreq) / NyquistFrequency
		b, a = butter(self.Order, [normlowfreq, normhighfreq], btype = 'bandpass', analog = False)

		for ChannelIter in xrange(0, NumChannels):
			FilteredData[:, ChannelIter] = lfilter(b, a, UnfilteredData[:, ChannelIter])

		return FilteredData;


	def BandPassFilterDemo(self, lowfreq, highfreq):
		bcidataobject = bcidataset.bcidataset(ConfigFile = self.ConfigFile, SamplingRate = self.SamplingRate, Xfilternum = self.Xfilternum, SubjectID = self.SubjectID)
		[SampleInputVectorList, SampleClassList] = bcidataobject.ReadSubjectFile()		
		NumSamples = len(SampleInputVectorList)

		UnfilteredData = np.zeros((NumSamples, len(SampleInputVectorList[0])))
		FilteredData = np.zeros((NumSamples, len(SampleInputVectorList[0])))

		for SampleIter in xrange(0, NumSamples):
			UnfilteredData[SampleIter, :] = np.array(SampleInputVectorList[SampleIter])

		NyquistFrequency = 0.5 * bcidataobject.SamplingRate

		normlowfreq = lowfreq / NyquistFrequency
		normhighfreq = highfreq / NyquistFrequency

		print normhighfreq, normlowfreq
		b, a = butter(self.Order, [normlowfreq, normhighfreq], btype = 'bandpass', analog = False)
		FilteredData[:,13] = lfilter(b, a, UnfilteredData[:,13])

		fftfilt = fft(FilteredData[:,13])
		fftunfilt = fft(UnfilteredData[:,13])

		print fftfilt[0:5]
		print fftfilt[-4:]
		plt.plot(20*np.log10(abs(fftunfilt)))
		plt.show()
		plt.plot(20*np.log10(abs(fftfilt)))
		plt.show()

		return;

	def FilteredOutputDemo(self, BestFrequencyRange, CSPfilters):
		print 'Reading Subject Data'
		bcidataobject = bcidataset.bcidataset(ConfigFile = self.ConfigFile, SamplingRate = self.SamplingRate, Xfilternum = self.Xfilternum, SubjectID = self.SubjectID)
		[SampleInputVectorList, SampleClassList] = bcidataobject.ReadSubjectFile()
		SampleInputArray = np.array(SampleInputVectorList)
		SampleClassArray = np.array(SampleClassList)
		del SampleInputVectorList, SampleClassList
		print 'Reading Completed'

		print 'Preforming Frequency Filtering in the BFR'
		FFSampleInputArray = self.BandPassFilter(BestFrequencyRange[0], BestFrequencyRange[1], SampleInputArray)
		print 'Performing Spatial Filtering with the best Spatial Filters'
		SFSampleInputArray = np.array(np.mat(FFSampleInputArray) * np.mat(CSPfilters.transpose()))
		del FFSampleInputArray

		NumPointsToPlot = 30000
		#NumParts should be able to integer divide NumPointsToPlot
		NumParts = 10
		PartLength = NumPointsToPlot / NumParts
		for Part in xrange(0, NumParts):
			fig = plt.figure()
			for NewChannelIndex in xrange(0, SFSampleInputArray.shape[1]):
				# #plot SFSampleInputArray[(Part*PartLength):((Part+1)*PartLength), NewChannelIndex] in color of SampleClassList[(Part*PartLength):((Part+1)*PartLength)]
				
				# pylab.xlabel('Time Samples')
				# pylab.ylabel('Filtered Value')
				# pylab.title('Filtered Channel: ' + str(NewChannelIndex + 1))

				ClassOneInd = [(index+1+Part*PartLength) for index, x in enumerate(SampleClassArray[(Part*PartLength):((Part+1)*PartLength)]) if x == -1]
				ClassTwoInd = [(index+1+Part*PartLength) for index, x in enumerate(SampleClassArray[(Part*PartLength):((Part+1)*PartLength)]) if x == 1]

				ax = fig.add_subplot(5,1,NewChannelIndex+1)
				ax.plot(np.array(xrange(Part*PartLength, (Part+1)*PartLength)), SFSampleInputArray[(Part*PartLength):((Part+1)*PartLength), NewChannelIndex])
				ax.plot(ClassOneInd, SFSampleInputArray[ClassOneInd, NewChannelIndex], color="red")
				ax.plot(ClassTwoInd, SFSampleInputArray[ClassTwoInd, NewChannelIndex], color="green")
				ax.set_title('Channel ' + str(NewChannelIndex + 1))
			fig.tight_layout()
			ImageDir = '/home/amit/data/BCI/VisualizeData/FilteredOutput/' + str(Part + 1) +'.jpg'
			pylab.savefig(ImageDir, bbox_inches='tight')
			# pylab.show()
		return;

	def FeatureDemo(self, FeatureVectors, OutputVector):
		NumPointsToPlot = 30000
		#NumParts should be able to integer divide NumPointsToPlot
		NumParts = 10
		PartLength = NumPointsToPlot / NumParts
		for Part in xrange(0, NumParts):
			fig = plt.figure()
			for NewChannelIndex in xrange(0, FeatureVectors.shape[1]):
				#plot FeatureVectors[(Part*PartLength):((Part+1)*PartLength), NewChannelIndex] in color of SampleClassList[(Part*PartLength):((Part+1)*PartLength)]
				ClassOneInd = [(index+1+Part*PartLength) for index, x in enumerate(OutputVector[(Part*PartLength):((Part+1)*PartLength)]) if x == -1]
				ClassTwoInd = [(index+1+Part*PartLength) for index, x in enumerate(OutputVector[(Part*PartLength):((Part+1)*PartLength)]) if x == 1]

				ax = fig.add_subplot(FeatureVectors.shape[1], 1, NewChannelIndex+1)
				ax.plot(np.array(xrange(Part*PartLength, (Part+1)*PartLength)), FeatureVectors[(Part*PartLength):((Part+1)*PartLength), NewChannelIndex])
				ax.plot(ClassOneInd, FeatureVectors[ClassOneInd, NewChannelIndex], color="red")
				ax.plot(ClassTwoInd, FeatureVectors[ClassTwoInd, NewChannelIndex], color="green")
				ax.set_title('Channel ' + str(NewChannelIndex + 1))
			fig.tight_layout()
			ImageDir = '/home/amit/data/BCI/VisualizeData/FeatureOutput/' + str(Part + 1) +'.jpg'
			pylab.savefig(ImageDir, bbox_inches='tight')
			# pylab.show()
		return;

	def ComputeProbabilityDistribution(self, SelectedFilterData):
		if SelectedFilterData.ndim == 1:
			Histogram, BinEdges = np.histogram(SelectedFilterData, self.NumBinsPerDim)
		elif SelectedFilterData.ndim == 2:
			Histogram, xedges, yedges = np.histogram2d(SelectedFilterData[:, 0], SelectedFilterData[:, 1], self.NumBinsPerDim)
			BinEdges = np.column_stack((xedges, yedges))

		return Histogram.astype(float) / float(np.sum(Histogram)), BinEdges;

	# def FindBin(self, PointComponent, EdgesDim):
	# 	for edgeiter in xrange(0, EdgesDim.shape[0] - 1):
	# 		if (PointComponent >= EdgesDim[edgeiter]) and (PointComponent < EdgesDim[edgeiter + 1]):
	# 			return edgeiter;

	# 	return edgeiter;

	def ComputeMutualInformation(self, SelectedFilterData):
		#SelectedFilterData is the data from one filter, it is one dimensional and has the class output in the second column.
		#It is selected based on FilterScoringFunction
		MI = 0

		ProbabilityDensityXY, BinEdgesXY = self.ComputeProbabilityDistribution(SelectedFilterData)
		ProbabilityDensityX, BinEdgesX = self.ComputeProbabilityDistribution(SelectedFilterData[:, 0])
		ProbabilityDensityY, BinEdgesY = self.ComputeProbabilityDistribution(SelectedFilterData[:, 1])

		for xbin in xrange(0, ProbabilityDensityXY.shape[0]):
			for ybin in xrange(0, ProbabilityDensityXY.shape[1]):
				if ProbabilityDensityXY[xbin, ybin] != 0:
					MI = MI + (ProbabilityDensityXY[xbin, ybin] * np.log2(ProbabilityDensityXY[xbin, ybin] / (ProbabilityDensityX[xbin] * ProbabilityDensityY[ybin])))
		# for SampleIter in xrange(0, SelectedFilterData.shape[0]):
		# 	X = SelectedFilterData[SampleIter, 0]
		# 	Y = SelectedFilterData[SampleIter, 1]
		# 	pX = ProbabilityDensityX[self.FindBin(X, BinEdgesX)]
		# 	pY = ProbabilityDensityY[self.FindBin(Y, BinEdgesY)]
		# 	pXY = ProbabilityDensityXY[self.FindBin(X, BinEdgesXY[:, 0]), self.FindBin(Y, BinEdgesXY[:, 1])]

		# 	MI = MI + (pXY * np.log2(pXY / (pX * pY)))

		return MI;

	def FilterScoringFunction(self, FilterData):
		#FilterData has two clomuns. First is the filtered data. Second is the output class.
		ClassOneData = FilterData[FilterData[:, 1] == -1]
		ClassTwoData = FilterData[FilterData[:, 1] == 1]
		IdleClassData = FilterData[FilterData[:, 1] == 0]

		MIscore = self.ComputeMutualInformation(np.concatenate((ClassOneData, ClassTwoData), axis = 0)) + max(self.ComputeMutualInformation(np.concatenate((ClassOneData, IdleClassData), axis = 0)), self.ComputeMutualInformation(np.concatenate((ClassTwoData, IdleClassData), axis = 0)))

		return MIscore;

	def GetMIFeatureVector(self, SFSampleInputArray):
		VarianceWidth = int(self.CovariancePeriod / 2)
		NumSamples = SFSampleInputArray.shape[0] - (2 * VarianceWidth)
		MIFeatureVector = np.zeros((NumSamples, SFSampleInputArray.shape[1]))

		for SampleIndex in xrange(VarianceWidth, NumSamples - VarianceWidth):
			SampleData = np.zeros((SFSampleInputArray.shape[1], self.CovariancePeriod))
			SampleData = SFSampleInputArray[(SampleIndex - VarianceWidth):(SampleIndex + VarianceWidth + 1), :].transpose()
			SampleVariance = np.var(SampleData, axis = 1)
			MIFeatureVector[SampleIndex - VarianceWidth, :] = np.log10(SampleVariance / np.sum(SampleVariance))

		return MIFeatureVector;

	def ComputeBestXFilters(self, SFSampleInputArray, SampleClassArray):
		VarianceWidth = int(self.CovariancePeriod / 2)
		MIFeatureVector = self.GetMIFeatureVector(SFSampleInputArray)
		OutputArray = SampleClassArray[VarianceWidth:-VarianceWidth]
		FilterScores = np.zeros(MIFeatureVector.shape[1])

		#print 'Scoring Filters : '
		for FilterIndex in xrange(0, MIFeatureVector.shape[1]):
			#print (FilterIndex + 1)
			FilterScores[FilterIndex] = self.FilterScoringFunction(np.column_stack((MIFeatureVector[:, FilterIndex], OutputArray)))

		ChosenFilters = np.argsort(FilterScores)[-self.Xfilternum:]

		#print 'Filter Scores'
		#print FilterScores

		return ChosenFilters;

	def ComputeFrequencyRangeScore(self, SFSampleInputArray, SampleClassArray):
		VarianceWidth = int(self.CovariancePeriod / 2)
		MIFeatureVector = self.GetMIFeatureVector(SFSampleInputArray)
		OutputArray = SampleClassArray[VarianceWidth:-VarianceWidth]
		FilterScores = np.zeros(MIFeatureVector.shape[1])

		#print 'Scoring Filters for BestFrequencyRange: '
		for FilterIndex in xrange(0, MIFeatureVector.shape[1]):
			#print (FilterIndex + 1)
			FilterScores[FilterIndex] = self.FilterScoringFunction(np.column_stack((MIFeatureVector[:, FilterIndex], OutputArray)))

		return np.sum(FilterScores);

	def ComputeBestFrequencyRange(self, SampleInputArray, SampleClassArray, CSPfilters):
		VarianceWidth = int(self.CovariancePeriod / 2)
		FrequencyRanges = np.array([[6,11], [7,12], [8,13], [9,14], [10,15], [11,16], [12,17], [17,25], [25,32]]).astype(float)
		FrequencyScores = np.zeros(FrequencyRanges.shape[0])

		for FreqIter in xrange(0, FrequencyRanges.shape[0]):
			#print 'Frequency Range : '
			#print FrequencyRanges[FreqIter]
			FFSampleInputArray = self.BandPassFilter(FrequencyRanges[FreqIter, 0], FrequencyRanges[FreqIter, 1], SampleInputArray)
			SFSampleInputArray = np.array(np.mat(FFSampleInputArray) * np.mat(CSPfilters.transpose()))
			del FFSampleInputArray
			FrequencyScores[FreqIter] = self.ComputeFrequencyRangeScore(SFSampleInputArray, SampleClassArray)
			del SFSampleInputArray

		#print 'Frequency Range Scores'
		#print FrequencyScores

		return FrequencyRanges[np.argmax(FrequencyScores)];

	def GetFeatureVectorsTrain(self, BestFrequencyRange, CSPfilters):
		#print '-----------------------------------'
		#print 'Reading Subject Data'
		bcidataobject = bcidataset.bcidataset(ConfigFile = self.ConfigFile, SamplingRate = self.SamplingRate, Xfilternum = self.Xfilternum, SubjectID = self.SubjectID)

		[SampleInputVectorList, SampleClassList] = bcidataobject.ReadSubjectFile()

		SampleInputArray = np.array(SampleInputVectorList)
		SampleClassArray = np.array(SampleClassList)
		del SampleInputVectorList, SampleClassList
		#print 'Reading Completed'

		#print 'Performing frequency and spatial filtering.'
		FFSampleInputArray = self.BandPassFilter(BestFrequencyRange[0], BestFrequencyRange[1], SampleInputArray)
		del SampleInputArray
		SFSampleInputArray = np.array(np.mat(FFSampleInputArray) * np.mat(CSPfilters.transpose()))
		del FFSampleInputArray

		FeatureList = []
		OutputList = []
		SampleIndex = 0

		while SampleIndex <= SFSampleInputArray.shape[0] - self.CovariancePeriod:
			SampleData = np.zeros((SFSampleInputArray.shape[1], self.CovariancePeriod))
			if (SampleClassArray[SampleIndex] == 0) and (SampleClassArray[SampleIndex + self.CovariancePeriod - 1] == 0):
				SampleData = SFSampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SampleEnergy = np.linalg.norm(SampleData, axis=1)
				FeatureList.append(2 * np.log10(SampleEnergy))
				OutputList.append(0)

				SampleIndex = SampleIndex + self.CovariancePeriod
			elif SampleClassArray[SampleIndex] == -1:
				SampleIndex = SampleIndex + 1*self.SamplingRate
				SampleData = SFSampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SampleEnergy = np.linalg.norm(SampleData, axis=1)
				FeatureList.append(2 * np.log10(SampleEnergy))
				OutputList.append(-1)

				SampleIndex = SampleIndex + self.CovariancePeriod
				SampleData = SFSampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SampleEnergy = np.linalg.norm(SampleData, axis=1)
				FeatureList.append(2 * np.log10(SampleEnergy))
				OutputList.append(-1)

				SampleIndex = SampleIndex + self.CovariancePeriod
			elif SampleClassArray[SampleIndex] == 1:
				SampleIndex = SampleIndex + 1*self.SamplingRate
				SampleData = SFSampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SampleEnergy = np.linalg.norm(SampleData, axis=1)
				FeatureList.append(2 * np.log10(SampleEnergy))
				OutputList.append(1)

				SampleIndex = SampleIndex + self.CovariancePeriod
				SampleData = SFSampleInputArray[SampleIndex : SampleIndex + self.CovariancePeriod, :].transpose()
				SampleEnergy = np.linalg.norm(SampleData, axis=1)
				FeatureList.append(2 * np.log10(SampleEnergy))
				OutputList.append(1)

				SampleIndex = SampleIndex + self.CovariancePeriod
			else:
				SampleIndex = SampleIndex + 1

		return [np.array(FeatureList), np.array(OutputList)];


	def GetFeatureVectorsTest(self, BestFrequencyRange, CSPfilters):
		#print '-----------------------------------'
		#print 'Reading Subject Data'
		bcidataobject = bcidataset.bcidataset(ConfigFile = self.ConfigFile, SamplingRate = self.SamplingRate, Xfilternum = self.Xfilternum, SubjectID = self.SubjectID)

		[SampleInputVectorList, SampleClassList] = bcidataobject.ReadSubjectFileTest()

		SampleInputArray = np.array(SampleInputVectorList)
		SampleClassArray = np.array(SampleClassList)
		del SampleInputVectorList, SampleClassList
		#print 'Reading Completed'

		#print 'Performing frequency and spatial filtering.'
		FFSampleInputArray = self.BandPassFilter(BestFrequencyRange[0], BestFrequencyRange[1], SampleInputArray)
		del SampleInputArray
		SFSampleInputArray = np.array(np.mat(FFSampleInputArray) * np.mat(CSPfilters.transpose()))
		del FFSampleInputArray

		NumSamples = SFSampleInputArray.shape[0] - self.CovariancePeriod + 1
		FeatureVectors = np.zeros((NumSamples, SFSampleInputArray.shape[1]))
		OutputVector = SampleClassArray[0:NumSamples]

		for SampleIndex in xrange(0, NumSamples):
			SampleData = np.zeros((SFSampleInputArray.shape[1], self.CovariancePeriod))
			SampleData = SFSampleInputArray[SampleIndex:SampleIndex + self.CovariancePeriod, :].transpose()
			SampleEnergy = np.linalg.norm(SampleData, axis=1)
			FeatureVectors[SampleIndex, :] = 2 * np.log10(SampleEnergy)

		return [FeatureVectors, OutputVector];



	def MahmoudPreProcessingAlgorithm(self):
		#print 'Reading Subject Data'
		bcidataobject = bcidataset.bcidataset(ConfigFile = self.ConfigFile, SamplingRate = self.SamplingRate, Xfilternum = self.Xfilternum, SubjectID = self.SubjectID)
		[SampleInputVectorList, SampleClassList] = bcidataobject.ReadSubjectFile()
		SampleInputArray = np.array(SampleInputVectorList)
		SampleClassArray = np.array(SampleClassList)
		del SampleInputVectorList, SampleClassList
		#print 'Reading Completed'

		#print 'Initial Frequency Filtering'
		FFSampleInputArray = self.BandPassFilter(8.0, 30.0, SampleInputArray)
		#print 'Computing CSP Filters'
		CSPfilters = self.TrainCSPfilters(FFSampleInputArray, SampleClassArray)
		#print 'Applying CSP Filters'
		SFSampleInputArray = np.array(np.mat(FFSampleInputArray) * np.mat(CSPfilters.transpose()))
		del FFSampleInputArray
		#print 'Computing Best X Filters'
		ChosenFilters = self.ComputeBestXFilters(SFSampleInputArray, SampleClassArray)
		del SFSampleInputArray

		#print 'Computing Best Frequency Range'
		BestFrequencyRange = self.ComputeBestFrequencyRange(SampleInputArray, SampleClassArray, CSPfilters[ChosenFilters])
		del CSPfilters

		#print 'Final Frequency Filtering'
		FFSampleInputArray = self.BandPassFilter(BestFrequencyRange[0], BestFrequencyRange[1], SampleInputArray)
		#print 'Re-computing CSP Filters'
		CSPfilters = self.TrainCSPfilters(FFSampleInputArray, SampleClassArray)
		#print 'Applying CSP Filters'
		SFSampleInputArray = np.array(np.mat(FFSampleInputArray) * np.mat(CSPfilters.transpose()))
		del FFSampleInputArray
		#print 'Computing Best X Filters'
		ChosenFilters = self.ComputeBestXFilters(SFSampleInputArray, SampleClassArray)
		del SFSampleInputArray

		#print 'The best frequency range obtained is : '
		#print BestFrequencyRange
		#print 'The best X filters for this range are : '
		#print CSPfilters[ChosenFilters]

		bcidataobject.StoreBestFrequencyRange(BestFrequencyRange)
		bcidataobject.StoreBestXFilters(CSPfilters[ChosenFilters])

		return;


# test = preprocessing()
# print test.MahmoudPreProcessingAlgorithm()
# test.TrainCSPfilters()
# test.BandPassFilterDemo(8,30)