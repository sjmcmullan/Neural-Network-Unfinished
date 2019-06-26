import math

class Neuron:
    def __init__(self, layer, name):
        self.layer = layer
        self.name = name
        self.netValue = 0
        self.outValue = 0
        self.errorValue = 0
        self.targetValue = 0
        self.inputWeights = []
        self.outputWeights = []

    def SetLayer(self, layer):
        self.layer = layer

    def SetName(self, name):
        self.name = name
    
    def SetNetValue(self, value):
        self.netValue = value
    
    def SetOutValue(self, value):
        self.outValue = value
    
    def SetErrorValue(self, value):
        self.errorValue = value

    def SetTargetValue(self, value):
        self.targetValue = value

    def SetInputWeights(self, weights):
        self.inputWeights = weights
    
    def SetOutputWeights(self, weights):
        self.outputWeights = weights
    
    def UpdateInputWeights(self, weight):
        self.GetInputWeights().append(weight)

    def UpdateOutputWeights(self, weight):
        self.GetOutputWeights().append(weight)

    def GetLayer(self):
        return self.layer

    def GetName(self):
        return self.name
    
    def GetNetValue(self):
        return self.netValue
    
    def GetOutValue(self):
        return self.outValue

    def GetErrorValue(self):
        return self.errorValue

    def GetTargetValue(self):
        return self.targetValue
    
    def GetInputWeights(self):
        return self.inputWeights

    def GetOutputWeights(self):
        return self.outputWeights

    def CalculateNet(self):
        netSum = 0
        for weight in self.GetInputWeights():
            if weight.GetInputNode() != None:
                netSum += weight.GetValue() * weight.GetInputNode().GetOutValue()
            else:
                netSum += weight.GetValue()

        self.SetNetValue(netSum)

    def CalculateOut(self):
        self.SetOutValue(1 / (1 + math.exp(-self.GetNetValue())))
    
    def CalculateNeuronError(self):
        self.errorValue = 0.5 * ((self.GetTargetValue() - self.GetOutValue()) ** 2)
    