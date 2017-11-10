# Rohitash Chandra, 2017 c.rohitash@gmail.conm

#!/usr/bin/python

# built using: https://github.com/rohitash-chandra/VanillaFNN-Python


#Sigmoid units used in hidden and output layer. gradient descent and stocastic gradient descent functions implemented with momentum


#Multi-task learning for modular neural networks.
 

import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time

#An example of a class
class Network:

    def __init__(self, Topo, Train, Test, MaxTime,  MinPer):

        self.Top  = Topo  # NN topology [input, hidden, output]
        self.Max = MaxTime # max epocs
        self.TrainData = Train
        self.TestData = Test
        self.NumSamples = Train.shape[0]

        self.lrate  = 0 # will be updated later with BP call

        self.momenRate = 0

        self.minPerf = MinPer
                                        #initialize weights ( W1 W2 ) and bias ( b1 b2 ) of the network
    	np.random.seed()
        self.W1 = np.random.randn(self.Top[0]  , self.Top[1])  / np.sqrt(self.Top[0] )
        self.B1 = np.random.randn(1  , self.Top[1])  / np.sqrt(self.Top[1] ) # bias first layer
        self.BestB1 = self.B1
        self.BestW1 = self.W1
        self.W2 = np.random.randn(self.Top[1] , self.Top[2]) / np.sqrt(self.Top[1] )
        self.B2 = np.random.randn(1  , self.Top[2])  / np.sqrt(self.Top[1] ) # bias second layer
        self.BestB2 = self.B2
        self.BestW2 = self.W2
        self.hidout = np.zeros((1, self.Top[1] )) # output of first hidden layer
        self.out = np.zeros((1, self.Top[2])) #  output last layer

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def printNet(self):
        print self.Top
        print self.W1

    def sampleEr(self,actualout):
        error = np.subtract(self.out, actualout)
        sqerror= np.sum(np.square(error))/self.Top[2]
        #print sqerror
        return sqerror

    def ForwardPass(self, X ):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1) # output of first hidden layer#
        z2 = self.hidout.dot(self.W2)  - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer



    def BackwardPassMomentum(self, Input, desired, vanilla):
        out_delta =   (desired - self.out)*(self.out*(1-self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1-self.hidout))

        if vanilla == 1: #no momentum
            self.W2+= (self.hidout.T.dot(out_delta) * self.lrate)
            self.B2+=  (-1 * self.lrate * out_delta)
            self.W1 += (Input.T.dot(hid_delta) * self.lrate)
            self.B1+=  (-1 * self.lrate * hid_delta)

        else:
            # momentum http://cs231n.github.io/neural-networks-3/#sgd
            self.W2 += ( self.W2 *self.momenRate) + (self.hidout.T.dot(out_delta) * self.lrate)       # velocity update
            self.W1 += ( self.W1 *self.momenRate) + (Input.T.dot(hid_delta) * self.lrate)
            self.B2 += ( self.B2 *self.momenRate) + (-1 * self.lrate * out_delta)       # velocity update
            self.B1 += ( self.B1 *self.momenRate) + (-1 * self.lrate * hid_delta)





    def TestNetwork(self, phase,   erTolerance):
        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        nOutput = np.zeros((1, self.Top[2]))
        if phase == 1:
            Data = self.TestData
        if phase == 0:
            Data = self.TrainData
        clasPerf = 0
        sse = 0
        testSize = Data.shape[0]
        self.W1 = self.BestW1
        self.W2 = self.BestW2 #load best knowledge
        self.B1 = self.BestB1
        self.B2 = self.BestB2 #load best knowledge

        mape = 0 # calculate mape in future

        for s in xrange(0, testSize):

            Input[:]  =   Data[s,0:self.Top[0]]
            Desired[:] =  Data[s,self.Top[0]:]

            self.ForwardPass(Input )
            sse = sse+ self.sampleEr(Desired)


            return ( np.sqrt(sse/testSize),  0 )


    def saveKnowledge(self):
        self.BestW1 = self.W1
        self.BestW2 = self.W2
        self.BestB1 = self.B1
        self.BestB2 = self.B2

    def BP_GD(self, learnRate, mRate,    stocastic, vanilla, depth): # BP with SGD (Stocastic BP)
        self.lrate = learnRate
        self.momenRate = mRate

        Input = np.zeros((1, self.Top[0])) # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        #Er = []#np.zeros((1, self.Max))
        epoch = 0
        bestrmse = 1
        bestmape = 1 #  Calc Mean Absolute Percentage Error

        #while  epoch < self.Max and bestTrain > self.minPerf :
        while epoch < depth:
            sse = 0
            for s in xrange(0, self.NumSamples):

                if(stocastic):
                   pat = random.randint(0, self.NumSamples-1)
                else:
                   pat = s

                Input[:]  =  self.TrainData[pat,0:self.Top[0]]
                Desired[:] = self.TrainData[pat,self.Top[0]:]

                self.ForwardPass(Input )
                self.BackwardPassMomentum(Input , Desired, vanilla)
                sse = sse+ self.sampleEr(Desired)

            rmse = np.sqrt(sse/self.NumSamples*self.Top[2])

            if rmse < bestrmse:
               bestrmse = rmse
               self.saveKnowledge()




            epoch=epoch+1

        return (rmse,bestrmse, bestmape, epoch)

#--------------------------------------------------------------------------------------------------------

class MTnetwork: # Multi-Task leaning using Stocastic GD

    def __init__(self, mtaskNet, trainData, testData, maxTime, minPerf, learnRate, numModules, transKnow):
        #trainData and testData could also be different datasets. this example considers one dataset
        self.transKnowlege = transKnow
        self.trainData = trainData
        self.testData = testData
        self.maxTime = maxTime
        self.minCriteria = minPerf
        self.numModules = numModules # number of network modules (i.e tasks with shared knowledge representation)
                           # need to define network toplogies for the different tasks.

        self.mtaskNet = mtaskNet

        self.learnRate = learnRate
        self.trainTolerance = 0.20 # [eg 0.15 output would be seen as 0] [ 0.81 would be seen as 1]
        self.testTolerance = 0.49


    def transferKnowledge(self, Wprev, Wnext): # transfer knowledge (weights from given layer) from  Task n (Wprev) to Task n+1 (Wnext)
        x=0
        y = 0 
        Wnext[x:x+Wprev.shape[0], y:y+Wprev.shape[1]] = Wprev                                   #(Netlist[n].W1 ->  Netlist[n+1].W1)
        return Wnext


    def restoreKnowledge(self, Wprev, Wnext): #  restore the knowledge from previous task since transferKnowledge refines past tas knowledge 
        x=0
        y = 0 
        Wnext[x:x+Wprev.shape[0], y:y+Wprev.shape[1]] = Wprev                                   
        return Wnext

    def mainAlg(self):

        mRate = 0.05

        stocastic = 1 # 0 for vanilla BP. 1 for Stocastic BP
        vanilla = 1 # 1 for Vanilla Gradient Descent, 0 for Gradient Descent with momentum


        Netlist = [None]*10  # create list of Network objects ( just max size of 10 for now )


        trainPerf = np.zeros(self.numModules)
        trainRMSE =  np.zeros(self.numModules)
        testPerf = np.zeros(self.numModules)
        testRMSE =  np.zeros(self.numModules)

        erPlot = np.random.randn(self.maxTime ,self.numModules)
         # plot of convergence for each module (Netlist[n] )

        depthSearch = 5 #declare


        for n in xrange(0, self.numModules):
            module = self.mtaskNet[n]
            taskfeatures = module[0]
            trdata = taskdata(self.trainData, module[0],  module[2] )   # make the partitions for task data
            testdata = taskdata(self.testData, module[0],  module[2] )
            Netlist[n] = Network(self.mtaskNet[n], trdata, testdata, depthSearch,  self.minCriteria)


        cycles = 0
        index = 0
        current = 0
        while(current) <(self.maxTime): #*self.numModules
            cycles =cycles + 1
           
            for n in xrange(0, self.numModules-1):
                   #if n == 0:
                depthSearch =  10-(n*3) # random.randint(1, 5) #stocastic depth
                #print depthSearch
                #else:
                   #depthSearch = 1
                
                current = current + depthSearch
                #print current, depthSearch
            	(erPlot[index, n],  trainRMSE[n], trainPerf[n], Epochs) = Netlist[n].BP_GD(self.learnRate, mRate,   stocastic, vanilla, depthSearch)
                 
                if(self.transKnowlege ==1):

                    Netlist[n+1].W1 = self.transferKnowledge(Netlist[n].W1, Netlist[n+1].W1) 
                    #print Netlist[n+1].W1, 'transfered'
                    Netlist[n+1].B1 = self.transferKnowledge(Netlist[n].B1, Netlist[n+1].B1)
                    Netlist[n+1].W2 = self.transferKnowledge(Netlist[n].W2, Netlist[n+1].W2)
                    Netlist[n+1].B2 = self.transferKnowledge(Netlist[n].B2, Netlist[n+1].B2)

                    if n >=1: 
                    	Netlist[n].W1 = self.restoreKnowledge(Netlist[n-1].W1, Netlist[n].W1)  
                    	Netlist[n].B1 = self.restoreKnowledge(Netlist[n-1].B1, Netlist[n].B1)   
                    	Netlist[n].W2 = self.restoreKnowledge(Netlist[n-1].W2, Netlist[n].W2)  
                    	Netlist[n].B2 = self.restoreKnowledge(Netlist[n-1].B2, Netlist[n].B2) 
                    	#print Netlist[n].W1, 'restored' 



            (erPlot[index, self.numModules-1],  trainRMSE[self.numModules-1], trainPerf[self.numModules-1], Epochs) = Netlist[self.numModules-1].BP_GD(self.learnRate, mRate, stocastic, vanilla, depthSearch) # BP for last module

    #print Netlist[n+1].W1
            index = index + 1
            #print trainMSE, cycles, current

        for n in xrange(0, self.numModules):
            (testRMSE[n], testPerf[n]) = Netlist[n].TestNetwork(1, self.testTolerance) # 1 in argument means to use testdata


        return (erPlot, trainRMSE, trainPerf, testRMSE, testPerf)


def taskdata(data, taskfeatures, output):
    # group taskdata from main data source.
    # note that the grouping is done in accending order fin terms of features.
    # the way the data is grouped as tasks can change for different applications.
    # there is some motivation to keep the features with highest contribution as first  feature space for module 1.
    datacols = data.shape[1]
    featuregroup = data[:,0:taskfeatures]
    return np.concatenate(( featuregroup[:,range(0,taskfeatures)], data[:,range(datacols-output,datacols)]), axis=1)



# ------------------------------------------------------------------------------------------------------



def main():
 
    np.random.seed()  
    fileout1 =  open('out1_v2.txt','a')
    fileout2 =  open('out2_v2.txt','a')

    moduledecomp = [0.25, 0.5, 0.75, 1]  # decide what will be number of features for each group of taskdata correpond to module


    for problem in range(1, 8):

        hidden = 2
        input = 4  #
        output = 1
        learnRate = 0.2
        mRate = 0.01
        MaxTime = 4000

        if problem == 1:
           traindata  = np.loadtxt("Data_OneStepAhead/Lazer/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Lazer/test.txt") #
        if problem == 2:
           traindata  = np.loadtxt("Data_OneStepAhead/Sunspot/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Sunspot/test.txt") #
        if problem == 3:
           traindata  = np.loadtxt("Data_OneStepAhead/Mackey/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Mackey/test.txt") #
        if problem == 4:
           traindata  = np.loadtxt("Data_OneStepAhead/Lorenz/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Lorenz/test.txt") #
        if problem == 5:
           traindata  = np.loadtxt("Data_OneStepAhead/Rossler/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Rossler/test.txt") #
        if problem == 6:
           traindata  = np.loadtxt("Data_OneStepAhead/Henon/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/Henon/test.txt") #
        if problem == 7:
           traindata  = np.loadtxt("Data_OneStepAhead/ACFinance/train.txt")
           testdata  = np.loadtxt("Data_OneStepAhead/ACFinance/test.txt") #



        print traindata

        print problem


        TrSamples =  np.size(traindata,0)
        TestSize = np.size(testdata,0)
        inputfeatures = np.size(traindata,1) - 1




        MaxRun = 30  # number of experimental runs

        MinCriteria = 0.0000001 #stop when RMSE reches this point

        numModules = 4 # first decide number of  modules (or ensembles for comparison)

        baseNet = [input, hidden, output]

        inputfeatures = baseNet[0] # total num inputfeatures for the prob

        mtaskNet =   np.array([baseNet, baseNet,baseNet,baseNet])

        for i in xrange(1, numModules):
            mtaskNet[i-1][0]  =  moduledecomp[i-1] * inputfeatures
            mtaskNet[i][1] += (i*2) # in this example, we have fixed numner  output neurons. input for each task is termined by feature group size.
                                         # we adapt the number of hidden neurons for each task.
        print mtaskNet # print network topology of all the modules that make the respective tasks. Note in this example, the tasks aredifferent network topologies given by hiddent number of hidden layers.

        trainPerf = np.zeros((MaxRun,numModules))
        testPerf =  np.zeros((MaxRun,numModules))
        meanTrain =  np.zeros(numModules)
        stdTrain =  np.zeros(numModules)
        meanTest =  np.zeros(numModules)
        stdTest =  np.zeros(numModules)
        x =   np.zeros(numModules)

        trainRMSE =  np.zeros((MaxRun,numModules))
        testRMSE =  np.zeros((MaxRun,numModules))
        Epochs =  np.zeros(MaxRun)
        Time =  np.zeros(MaxRun)



        for transKnow in xrange(1, 2  ): # transKnow = 0 # 1 is for MT knowledge transfer. 0 is for no transfer (simple ensemble 'learning)
            for run in xrange(0, MaxRun  ):
                print run
                mt = MTnetwork(mtaskNet, traindata, testdata, MaxTime,MinCriteria,learnRate, numModules,transKnow)
                (erPlot, trainRMSE[run,:], trainPerf[run,:], testRMSE[run,:], testPerf[run,:]) = mt.mainAlg()
                x = [problem, transKnow, run]
                print x, trainRMSE[run,:]
                print x, testRMSE[run,:]


            for module in xrange(0, numModules ):
                meanTrain[module] = np.mean(trainRMSE[:,module])
                stdTrain[module] = np.std(trainRMSE[:,module])
                meanTest[module] = np.mean(testRMSE[:,module])
                stdTest[module] = np.std(testRMSE[:,module])


            print meanTrain
            print stdTrain
            print meanTest
            print stdTest

            np.savetxt(fileout2, (problem, transKnow ), fmt='%1.1f'     )
            np.savetxt(fileout2, (meanTrain,stdTrain,meanTest,stdTest ), fmt='%1.5f')

if __name__ == "__main__": main()
