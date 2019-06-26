from Neuron import Neuron
from Weight import Weight

i1 = Neuron("INPUT", "I1")
i2 = Neuron("INPUT", "I2")
h1 = Neuron("HIDDEN", "H1")
h2 = Neuron("HIDDEN", "H2")
o1 = Neuron("OUTPUT", "O1")
o2 = Neuron("OUTPUT", "O2")
neuronList = [i1, i2, h1, h2, o1, o2]
w1 = Weight("W1", 0.1, i1, h1)
w2 = Weight("W2", 0.2, i1, h2)
w3 = Weight("W3", 0.1, i2, h1)
w4 = Weight("W4", 0.1, i2, h2)
w5 = Weight("W5", 0.1, h1, o1)
w6 = Weight("W6", 0.1, h1, o2)
w7 = Weight("W7", 0.1, h2, o1)
w8 = Weight("W8", 0.2, h2, o2)
# Bias weights.
w9 = Weight("W9", 0.1, None, h1)
w10 = Weight("W10", 0.1, None, h2)
w11 = Weight("W11", 0.1, None, o1)
w12 = Weight("W12", 0.1, None, o2)
weightList = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12]

i1.SetOutputWeights([w1, w2])
i2.SetOutputWeights([w3, w4])
h1.SetInputWeights([w1, w3, w9])
h1.SetOutputWeights([w5, w6])
h2.SetInputWeights([w2, w4, w10])
h2.SetOutputWeights([w7, w8])
o1.SetInputWeights([w5, w7, w11])
o2.SetInputWeights([w6, w8, w12])

sampleInput1 = (0.1, 0.1)
sampleInput2 = (0.1, 0.2)
sampleOutput1 = (1, 0)
sampleOutput2 = (0, 1)

trainingRate = 0.1
epochs = 1
batchs = 1
batchSize = 2

inputSamples = [sampleInput1, sampleInput2]
outputSamples = [sampleOutput1, sampleOutput2]

sampleErrorTotal = 0
for epoch in range(0, epochs):
    for batch in range(0, batchs):
        for sample in range(0, batchSize):
            # Forward pass.
            print("Performing forward pass...")
            for neuron in range(0, 2):  
                neuronList[neuron].SetOutValue(inputSamples[sample][neuron])
                neuronList[-(neuron+1)].SetTargetValue(outputSamples[sample][-(neuron+1)])

            for neuron in range(2, 4):
                neuronList[neuron].CalculateNet()
                neuronList[neuron].CalculateOut()
            
            for neuron in range(4, 6):
                neuronList[neuron].CalculateNet()
                neuronList[neuron].CalculateOut()
                neuronList[neuron].CalculateNeuronError()
                sampleErrorTotal += neuronList[neuron].GetErrorValue()
            
            print("Performing backwards pass...")
            for weight in range(len(weightList)-5, -1, -1):
                weightList[weight].CalculateGradient()
            sampleErrorTotal = 0
        for weight in range(0, len(weightList) - 4):
            weightList[weight].CalculateNewValue(trainingRate, batchSize)
        print()

        print("Performing test pass...")
        for outputSample in range(0, batchSize):
            # Forward pass.
            for neuron in range(0, 2):  
                neuronList[neuron].SetOutValue(outputSamples[outputSample][neuron])
                neuronList[-(neuron+1)].SetTargetValue(outputSamples[sample][-(neuron+1)])

            for neuron in range(2, 4):
                neuronList[neuron].CalculateNet()
                neuronList[neuron].CalculateOut()
            
            for neuron in range(4, 6):
                neuronList[neuron].CalculateNet()
                neuronList[neuron].CalculateOut()
                neuronList[neuron].CalculateNeuronError()
                sampleErrorTotal += neuronList[neuron].GetErrorValue()
            print("The total error for this output sample is", sampleErrorTotal)
        print()