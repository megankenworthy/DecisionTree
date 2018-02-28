#Megan Kenworthy
#Artificial Intelligence
#September 27, 2017


import csv
import math 
import numpy as np
import matplotlib.pyplot as plt 


#data is nXfeatures
#classes is nX1 (class assignments indexed off of data)
#feature is just 1d array of feature names 


def calc_entropy(p):
	if p!=0:
		return -p * np.log2(p)
	else:
		return 0



def infogain(data,classes,feature): 
	Gain = 0                        #setting Gain
	nData = len(data) 				#getting the length 

	#values list finds the different values that this feature can take (based on input)
	values = []

	#goes through each input
	for datapoint in data:  
		if datapoint[feature] not in values: 
			values.append(datapoint[feature]) 	#adds unique values to list

	featureCounts = np.zeros(len(values))       #array of 0s length of values
	entropy = np.zeros(len(values))
	valueIndex = 0 

	#trying to figure out based on the feature and entropy which class they fall into




	#find where values appear in data[feature] and corresponding class
	for value in values: 	#for each unique value that is given for a feature (prob sldnt include url)
		dataIndex = 0
		newClasses = []
		for datapoint in data: #iterating through all the inputs to see if the feature=value
			if datapoint[feature] == value: 
				featureCounts[valueIndex]+=1    #increment the count for featurecounts
				newClasses.append(classes[dataIndex]) #newclasses is holding the class assignment of the input at this index with the value
			dataIndex+=1

		
		#getting unique classifications 
		classvalues = [0,0,0,0]
		for classification in newClasses:   #iterating through list
			if classvalues.count(classification)==0: 
				if classification=='1':
					classvalues[0] = '1'
				elif classification=='2':
					classvalues[1] = '2'
				elif classification=='3':
					classvalues[2] = '3'
				else:
					classvalues[3] = '4'
				#classvalues.append(classification) #appending unique class names 


		#print('classes for that value in that feature', classvalues)

		classCounts = np.zeros(len(classvalues)) #list the length of the classes 
		classIndex = 0 
		#getting number of inputs in each class (4 in this case)
		for classValue in classvalues: 
			for classification in newClasses:
				if classification == classValue: 
					classCounts[classIndex]+=1     #gets the number of inputs with feature for each class
			classIndex+=1



		#you have to know the probability of getting a distinct value for each feature to calculate the 
		#entropy for that feature 
		#iterating through the number of classes 	
		for CIndex in range(len(classvalues)):
			if classCounts[CIndex]!=0:
				entropy[valueIndex] += calc_entropy(float(classCounts[CIndex])/sum(classCounts)) #calculating entropy for that value in feature

		#featureCounts = number of inputs that have this value for this feature
		#calc prob of value for feature * entropy for that value   (basically will do this 2x for binary decision before returning gain)
		

		Gain += float(featureCounts[valueIndex])/nData * entropy[valueIndex]
		valueIndex+=1

	#print(Gain)

	return Gain 


def make_tree(data,classes,featureNames,height):

	classLabels = ['1','2','3','4'] #to call in the base case for the recursion

	#we want to know the datapoints assigned to each class, held in the frequency table
	frequency = [0,0,0,0]

	freqiterator = len(classes)
	for freqoption in range(freqiterator):
		if classes[freqoption]=='1':
			frequency[0]+=1
		elif classes[freqoption]=='2':
			frequency[1]+=1
		elif classes[freqoption]=='3':
			frequency[2]+=1
		else:
			frequency[3]+=1

	#print(frequency)


	
	numData = len(data)            #getting the # of points that have yet to be accounted for
	nFeatures = len(featureNames)  #getting the # of features that haven't been accounted for


	#print('checkingnumdata',numData)
	#print('checking num features',nFeatures)


	#start with like 3 --- go up check accuracy 

	#if there is nothing left to test, return the most common classification
	if (numData==0)or(nFeatures==0)or(height==0):
		return np.argmax(frequency)+1

	#checking to see if only one class remains in the data by counting 
	#all of the class assignments for the first input in class where
	#if the number is the same as the number of input data, then they are all part of that class
	elif classes.count(classes[0])==numData: 
		return int(classes[0])						
												

	#all the building business is about to happen										
	else: 

		totalentropy = 0
		for totalE in range(len(frequency)):
			totalentropy += calc_entropy(float(frequency[totalE])/numData) # p = num of points classified to that class/total num points

		#print(totalentropy)
		#we have to choose which feature is best to split on 

		#trying to find info gain
		gain = np.zeros(nFeatures)    #makes an arrray size of features to calc each gain

		#iterate through each of the features 
		for feature in range(nFeatures):
			gain_calc = infogain(data,classes,feature) #get the gain for that feature										
			gain[feature] = totalentropy - gain_calc
		bestFeature = np.argmax(gain)   			   #finds the index of the best feature
	#cchange back to argmax
		#actually start the tree now, dictionaries are easier to work with in python 
		#and access the information through 
		tree = {featureNames[bestFeature]:{}}  		   #using the index of best feature to 
													   #start the tree

		#now we find the possible feature values for each feature 
		#these should be binary decisions now since we discretized the data (0 or 1)

		values = ['0','1']


		for value in values:

			#find the datapoint with each feature value, 0,1 
			newData = []
			newClasses = []
			new_Feature_Names = []
			index = 0
			for datapoint in data:
				if datapoint[bestFeature]==value: #if the datapoint has the val ur looking at
					
					if bestFeature==0: 			  #if index of best feature is in the start
						datapoint = datapoint[1:] #take out that column in the data
						new_Feature_Names = featureNames[1:]
					elif bestFeature==nFeatures: #if index of bestfeature is at the end
						datapoint = datapoint[:-1]
						new_Feature_Names = featureNames[:-1]
					
					#kind of have to finess the two parts together 
					else: 
						datapoint = datapoint[:bestFeature]+datapoint[bestFeature+1:]
						new_Feature_Names = featureNames[:bestFeature]+featureNames[bestFeature+1:]
						#print(new_Feature_Names)
				

					#accounting for the newdata and classes to be used in recursion
					newData.append(datapoint)
					newClasses.append(classes[index]) 
		
				index=index+1  		

		#recursing to next level
			subtree = make_tree(newData,newClasses,new_Feature_Names,height-1) 
		#at the point where you return, want to add all the subtrees onto the tree 
			tree[featureNames[bestFeature]][value] = subtree  

	return tree  


def TestTree(treedic,featurelist,testdata):

		#if data[root] value is 0, you want the next key to be dict[root][o] else dict[root][1]
		

	#if data[root] value is 0, you want the next key to be dict[root][o] else dict[root][1]


	if (treedic==1) or (treedic==2) or (treedic==3) or (treedic==4):
		return treedic

	else:
		rootlist = treedic.keys()
		#print(rootlist)
		root = rootlist[0]
		checkvalindex = featurelist.index(root)
		inputvalue = testdata[checkvalindex]
		newdicplace = treedic.get(root)
			#print(newdicplace)
		nextdic = newdicplace.get(inputvalue) #want it specific to datapoint
		#print(nextdic)

		return TestTree(nextdic,featurelist,testdata)

	#recursively call on rootlist = nextdic.keys() -- root = rootlist[0
	#rootindex = featurenames[root[0]]
	#print(rootindex)

	#recursively call on rootlist = nextdic.keys() -- root = rootlist[0
	#rootindex = featurenames[root[0]]
	#print(rootindex)
			

def Accuracy(treeclassified,realclassified):
	
	total = len(treeclassified)
	correct = 0
	#print(treeclassified+realclassified)
	perclass = [0,0,0,0]
	for dataclassify in range(total):
		if (treeclassified[dataclassify])==(int(realclassified[dataclassify])):
			correct=correct+1
			if treeclassified[dataclassify]==1:
				perclass[0]+=1
			elif treeclassified[dataclassify]==2:
				perclass[1]+=1
			elif treeclassified[dataclassify]==3:
				perclass[2]+=1
			else:
				perclass[3]+=1
	

	print('perclass',perclass)
	print('numbercorrect',correct)
	print('totalnumber',total)
	percentcorrect = (float(correct)/float(total))
	print('percent correct',percentcorrect*100)
	return percentcorrect*100



def main():
	#want to read in the csv files here


	with open('TrainingDataNewsPop.csv','rb') as trainingdata:
		
		data = []
		reader = csv.reader(trainingdata)
		for row in reader:
			data.append(row)  

	with open('BucketClassifiedtrain.csv','rb') as trainclasses:

		classes = []
		reader = csv.reader(trainclasses)
		for row in reader:
			classes.append(row[0])

	
	with open('TestingDataNewsPop.csv','rb') as testingdata:
		Testingdata = []
		reader = csv.reader(testingdata)
		for row in reader: 
			Testingdata.append(row)

	with open('BucketClassifiedtest.csv','rb') as testclasses:

		Testclasses = []
		reader = csv.reader(testclasses)
		for row in reader:
			Testclasses.append(row[0])

	Data = []
	for newdata in data:
		newdata = newdata[:-1]
		Data.append(newdata)

	TestData = []
	for tdata in Testingdata:
		tdata = tdata[:-1]
		TestData.append(tdata)


	featurenames = ["# Words in Title","# Words in Content","Rate of Unique Words",
	"Rate of Nonstop Words","Rate of Unique Nonstop Words","Number of Links","Number Internal Links",
	"Number of Images","Number of Videos","Average Word Length","# Keywords in Metadata",
	"Data-Channel: Lifestyle","Data-Channel:Entertainment","Data-Channel:Business",
	"Data-Channel:Social Media","Data-Channel:Tech","Data-Channel:World","Worst Keyword Min Shares",
	"Worst Keyword Max Shares", "Worst Keyword Average Shares","Best Keyword Min Shares",
	"Best Keyword Max Shares","Best Keyword Average Shares","Average Keyword Min Shares",
	"Average Keyword Max Shares","Average Keyword Average Shares",
	"Min Shares of referenced Mashable articles","Max shares of referenced Mashable articles",
	"Avg Shares of references Mashable articles","Published on Monday","Published on Tuesday",
	"Published on Wednesday","Published on Thursday","Published on Friday",
	"Published on Saturday","Published on Sunday", "Published on Weekend",
	"Closeness to LDA topic 0","Closeness to LDA topic 1","Closeness to LDA topic 2",
	"Closeness to LDA topic 3","Closeness to LDA topic 4","Text Subjectivity",
	"Text Sentiment Polarity","Rate of Positive Words","Rate of Negative Words",
	"Rate of Positive words among non-neutral tokens","Rate of Negative Words among non-neutral Tokens",
	"Average Polarity of Positive Words","Min Polarity of Positive Words","Max Polarity of Positive Words",
	"Average Polarity of Negative Words","Min Polarity of Negative Words",
	"Max Polarity of Negative Words","Title Subjectivity","Title Polarity",
	"Absolute Subjectivity (Title)","Absolute Polarity (Title)"]



	TestClassCounts = [0,0,0,0]
	TestClassCounts[0]=Testclasses.count('1')
	TestClassCounts[1]=Testclasses.count('2')
	TestClassCounts[2]=Testclasses.count('3')
	TestClassCounts[3]=Testclasses.count('4')

	print(TestClassCounts)
	#plotting accuracy w different heights
	#AcToPlot = []
	heightmax = 6
	#for i in range(heightmax):
	#print(i)
	DecisionTree = make_tree(Data,classes,featurenames,heightmax)

	#print(DecisionTree)

	testingClassifications = [] #setting this up 
	for testdatapoint in TestData:
		classification = TestTree(DecisionTree,featurenames,testdatapoint)
		testingClassifications.append(classification)


	AcToPlot = []
	addacc = Accuracy(testingClassifications,Testclasses)
	AcToPlot.append(addacc)
	print(AcToPlot)

#	plt.plot(range(heightmax),AcToPlot)
#	plt.axis([0,heightmax,0,100])
#	plt.xlabel('Tree Height')
#	plt.ylabel('Percent Classification Accuracy')
#	plt.title('Classification Accuracy By Tree Height')
#	plt.show()




main()
