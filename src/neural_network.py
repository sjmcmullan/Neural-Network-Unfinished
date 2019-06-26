from Neuron import Neuron
from Weight import Weight
import matplotlib.pyplot as plt
# import numpy as np
import sys
import csv
import gzip
import io
import os
import random
currentPath = os.path.dirname(__file__)
# Experimentation parameters.
epochs = 30
numberOfBatches = 20
learningRate = 3.0
batchSize = 0
# Neuron numbers.
numberOfInputNeurons = int(sys.argv[1])
numberOfHiddenNeurons = int(sys.argv[2])
numberOfOutputNeurons = int(sys.argv[3])
# Data lists.
trainingInputData = []
trainingLabelData = []
testInputData = []
testLabelData = []
# Lists to hold network components.
neuronList = []
weightList = []
# Miscellaneous variables.
batchTotalError = 0
epochTotalError = 0
numberOfSamplesRun = 0
numberofWeights = 0
errorPlotXValues = [x for x in range(1, 31)]
errorPlotYValues = []
# Reading in the data.
newPath = os.path.relpath("..\\data\\" + sys.argv[4], currentPath)
with gzip.open(newPath, "rt", newline="") as file:
    reader = csv.reader(file)
    trainingInputData = list(reader)
file.close()
newPath = os.path.relpath("..\\data\\" + sys.argv[5], currentPath)
with gzip.open(newPath, "rt", newline="") as file:
    reader = csv.reader(file)
    trainingLabelData = list(reader)
file.close()
newPath = os.path.relpath("..\\data\\" + sys.argv[6], currentPath)
with gzip.open(newPath, "rt", newline="") as file:
    reader = csv.reader(file)
    testInputData = list(reader)
file.close()
newPath = os.path.relpath("..\\data\\" + sys.argv[7], currentPath)
with gzip.open(newPath, "rt", newline="") as file:
    reader = csv.reader(file)
    testLabelData = list(reader)
file.close()
# Creating the network before working on it.
# First, generate all of the neurons.
for i in range(0, numberOfInputNeurons):
    tmpNeuron = Neuron("INPUT", "I"+str(i))
    neuronList.append(tmpNeuron)
for i in range(0, numberOfHiddenNeurons):
    tmpNeuron = Neuron("HIDDEN", "H"+str(i))
    neuronList.append(tmpNeuron)
for i in range(0, numberOfOutputNeurons):
    tmpNeuron = Neuron("OUTPUT", "O"+str(i))
    neuronList.append(tmpNeuron)
# Now, generate all of the weights.
# Start from the hidden layer.
for i in range(numberOfInputNeurons, numberOfInputNeurons + numberOfHiddenNeurons):
    outputNeuron = neuronList[i]
    for j in range(0, numberOfInputNeurons):
        inputNeuron = neuronList[j]
        tmpWeight = Weight("W"+str(numberofWeights), random.uniform(0, 1), inputNeuron, outputNeuron)
        inputNeuron.UpdateOutputWeights(tmpWeight)
        outputNeuron.UpdateInputWeights(tmpWeight)
        weightList.append(tmpWeight)
        numberofWeights += 1;
for i in range(numberOfInputNeurons + numberOfHiddenNeurons, numberOfInputNeurons + numberOfHiddenNeurons + numberOfOutputNeurons):
    outputNeuron = neuronList[i]
    for j in range(numberOfInputNeurons, numberOfInputNeurons + numberOfHiddenNeurons):
        inputNeuron = neuronList[j]
        tmpWeight = Weight("W"+str(numberofWeights), random.uniform(0, 1), inputNeuron, outputNeuron)
        inputNeuron.UpdateOutputWeights(tmpWeight)
        outputNeuron.UpdateInputWeights(tmpWeight)
        weightList.append(tmpWeight)
        numberofWeights += 1;
numberofBiasWeights = 0
for i in range(numberOfInputNeurons, numberOfInputNeurons + numberOfHiddenNeurons):
    outputNeuron = neuronList[i]
    tmpWeight = Weight("B"+str(numberofBiasWeights), 1, None, outputNeuron)
    outputNeuron.UpdateInputWeights(tmpWeight)
    weightList.append(tmpWeight)
    numberofBiasWeights += 1;
for i in range(numberOfInputNeurons + numberOfHiddenNeurons, numberOfInputNeurons + numberOfHiddenNeurons + numberOfOutputNeurons):
    outputNeuron = neuronList[i]
    tmpWeight = Weight("B"+str(numberofBiasWeights), 1, None, outputNeuron)
    outputNeuron.UpdateInputWeights(tmpWeight)
    weightList.append(tmpWeight)
    numberofBiasWeights += 1;
batchSize = len(trainingInputData)//numberOfBatches
for epoch in range(0, 1):
    for batch in range(0, numberOfBatches):
        for sample in range(0, batchSize):
            # Before performing the forward pass, assign the training data to the input neurons.
            for neuron in range(0, numberOfInputNeurons):
                neuronList[neuron].SetOutValue(float(trainingInputData[sample + numberOfSamplesRun][neuron]))
            # Also assign the target data to the output neurons.
            targetValueCounter = 0
            for neuron in range(numberOfInputNeurons + numberOfHiddenNeurons, numberOfInputNeurons + numberOfHiddenNeurons + numberOfOutputNeurons):
                # Need to create a list of 1s and 0s so that we can put that into each output neuron for this sample.
                targetList = [0 for x in range(0, 10)]
                targetList[int(trainingLabelData[sample + numberOfSamplesRun][0])] = 1
                neuronList[neuron].SetTargetValue(targetList[targetValueCounter])
                targetValueCounter += 1
            # Now do the forward pass, starting from the hidden neurons.
            for neuron in range(numberOfInputNeurons, numberOfInputNeurons + numberOfHiddenNeurons):
                neuronList[neuron].CalculateNet()
                neuronList[neuron].CalculateOut()
            # Continue the forward pass on the output layer neurons.
            for neuron in range(numberOfInputNeurons + numberOfHiddenNeurons, numberOfInputNeurons + numberOfHiddenNeurons + numberOfOutputNeurons):
                neuronList[neuron].CalculateNet()
                neuronList[neuron].CalculateOut()
                neuronList[neuron].CalculateNeuronError()
            # Now perform the backwards pass.
            # Start from the last weight in the network (i.e. don't perform this task on the bias weights).
            # Go backwards from there.
            for weight in range(numberofWeights - 1, -1, -1):
                weightList[weight].CalculateGradient()
        # Update numberOfSamplesRun so that on the next run, the correct data is retrieved.
        numberOfSamplesRun += batchSize
        # Update all of the weights' values.
        for weight in range(0, numberofWeights - numberofBiasWeights):
            weightList[weight].CalculateNewValue(learningRate, batchSize)
        # Perform a test on the neural network in it's current state for this batch.
        for testSample in range(0, len(testInputData)):
            # Map the test data to each input neuron.
            for neuron in range(0, numberOfInputNeurons):
                neuronList[neuron].SetOutValue(float(testInputData[testSample][neuron]))
            # Now assign the target value data to the output neurons.
            targetValueCounter = 0
            for neuron in range(numberOfInputNeurons + numberOfHiddenNeurons, numberOfInputNeurons + numberOfHiddenNeurons + numberOfOutputNeurons):
                # Need to create a list of 1s and 0s so that we can put that into each output neuron for this sample.
                targetList = [0 for x in range(0, 10)]
                targetList[int(testLabelData[testSample][0])] = 1
                neuronList[neuron].SetTargetValue(targetList[targetValueCounter])
                targetValueCounter += 1
            # Perform forward pass on test data.
            for neuron in range(numberOfInputNeurons, numberOfInputNeurons + numberOfHiddenNeurons):
                neuronList[neuron].CalculateNet()
                neuronList[neuron].CalculateOut()
            # Continue the forward pass on the output layer neurons.
            for neuron in range(numberOfInputNeurons + numberOfHiddenNeurons, numberOfInputNeurons + numberOfHiddenNeurons + numberOfOutputNeurons):
                neuronList[neuron].CalculateNet()
                neuronList[neuron].CalculateOut()
                neuronList[neuron].CalculateNeuronError()
                batchTotalError += neuronList[neuron].GetErrorValue()
        # Get the error for this batch as the average of all test sample errors.
        batchTotalError = (1/len(testInputData)) * batchTotalError
    # Get the error for this epoch as the average of all of the batchs.
    epochTotalError = (1/numberOfBatches) * batchTotalError
    errorPlotYValues.append(epochTotalError)
    # Reset the error trackers.
    batchTotalError = 0
    epochTotalError = 0
    numberOfSamplesRun = 0

# Plot the neural network results.
plt.xlabel("Number of epochs.")
plt.ylabel("Error.")
plt.title("Accuracy of neural network.")
plt.plot(errorPlotXValues, errorPlotYValues)
plt.grid(True)
plt.show()            