import numpy as np
import matplotlib.pyplot as plt
import random

class Class:
	def __init__(self, label, numOfSpecification=1):
		self.dataList = []
		self.mean = []
		self.covariance = np.zeros((numOfSpecification, numOfSpecification), dtype=float)
		self.numOfSpecification = numOfSpecification
		self.label = label
		self.numOfData = 0
 
	def CalculateMean(self):
		self.mean = np.mean(self.dataList, axis=0)
 
	def CalculateCovariance(self):
		self.covariance = (self.dataList[0] - self.mean).T.dot((self.datalist[0] - self.mean)) / (self.numOfData - 1)
 
	def AddDataList(self, sliceOfList):
		self.dataList.append(sliceOfList)
		self.numOfData += 1
 
	def ConvertListToArray(self):
		self.covariance = np.cov(np.array(self.dataList).T)
 
	def GetMeanVector(self):
		return self.mean
 
	def GetCovariance(self):
		return self.covariance
 
	def CalculateDiscriminantFunction(self, x):
		return np.dot(np.dot(x,-0.5*np.linalg.inv(self.covariance)),np.transpose(x)) + np.dot(np.transpose(np.dot(np.linalg.inv(self.covariance),np.transpose(self.mean))),np.transpose(x)) + ((-0.5*np.dot(np.dot(self.mean,(np.linalg.inv(self.covariance))), np.transpose(self.mean)))-0.5*(np.log(np.linalg.det(self.covariance))))
 
	def GetMahalanobisDistance(self, x):
		return np.dot(np.dot((x - self.mean),np.linalg.inv(self.covariance)),np.transpose(x - self.mean))
	
	def GetDataList(self):
		return self.dataList
 
	def GetNumOfData(self):
		return self.numOfData
 
	def ShowClassInfo(self):
		print("%d 's Class" % self.label)
		print("\nDatas >>\n" + str(self.dataList))
		print("Mean Vector: " + str(self.mean))
		print("Covariance Matrix >>\n" + str(self.covariance))
		print("numOfData : %d\n\n\n" % self.numOfData)
 
 
###############################################   READ DATA    #################################################
 
def ReadData(inputFile):
	dataList = []
	
	with open(inputFile, "r") as f:
		lines = f.readlines()
		for line in lines:
			datas = [float(data) for data in line.split(' ')[:-1]]
			datas.append(int(line.split(' ')[-1]))
			dataList.append(datas)
	return dataList
 
def DataClassification(dataList, classList):
	for data in dataList[:]:
		temp = data[:-1]
		label = data[-1]
		classList[label - 1].AddDataList(temp)
 
 
############################################ Plot Function #############################################
 
def PlotData(classList):
	shape = ['ro', 'bo', 'go'] #class1 = red, class2 = blue, class3 = green
	j=0
	for _class in classList:
		list = _class.GetDataList()
		for i in list:
			plt.plot(i[0], i[1], shape[j])
		j+=1
	plt.grid()
	plt.show()
 
 
############################################ Perceptron Algorithm ######################################

def ActivationFunction(value, threshold):
	if(value >= threshold):
		return 1
	return 0

def UpdateWeightvector(w, learningRate, data, targetValue, result):
	j=0
	w[0] = w[0] + learningRate * data[0] * (targetValue - result)
	w[1] = w[2] + learningRate * data[1] * (targetValue - result)
	w[2] = w[2] + learningRate * data[2] * (targetValue - result)
	print(w)
	return w

def CalculatePerceptron(classList, num1, num2, learningRate, threshold):
	dataList1 = classList[num1].GetDataList()
	dataList2 = classList[num2].GetDataList()
	w = [-2, 1, 1]
	epoch = 0
	sum = 0

#	for i in dataList1[0]:
#		w.append(random.uniform(-0.5, 0.5))


	print("initial value of Weight Vector" + str(w))

	while(True):
		# Learning
		errorCount = 0
		for i in range(classList[num1].GetNumOfData()):
			temp = [1]
			temp += dataList1[i]
			temp = np.transpose(np.array(temp))
			sum = np.dot(w, temp)
			result = ActivationFunction(sum, threshold)
			if(result == 0):
				#targetvalue == 1
				w = UpdateWeightvector(w,learningRate,temp,1,result)
				errorCount+=1

		for i in range(classList[num2].GetNumOfData()):
			temp = [1]
			temp += dataList2[i]
			temp = np.transpose(np.array(temp))
			sum = np.dot(w, temp)
			result = ActivationFunction(sum, threshold)
			if(result == 1):
				#targetvalue == 0
				w = UpdateWeightvector(w, learningRate, temp, 0, result)
				errorCount+=1
		print("epoch is >> " + str(epoch))
		print("errorcount is >> " + str(errorCount))
		print("weight is >> " + str(w))
 		if(errorCount == 0):
			print("finished Classification")
			print("epoch is >>  " + str(epoch))
			print("weight Vector is >>  " + str(w))
			#TODO plot
			break


		epoch += 1
 
 
############################################ main #############################################
 
TRAIN_DATA = "train_data.txt"
NUMOFCLASSES = 3
NUMOFSPECIFICATION = 2
LEARNINGRATE = 0.05
 
trainDataList = ReadData(TRAIN_DATA)
classList = [Class(i + 1, NUMOFSPECIFICATION) for i in range(NUMOFCLASSES)]
 
DataClassification(trainDataList, classList)
PlotData(classList)

CalculatePerceptron(classList, 1, 2, LEARNINGRATE, 1.0)