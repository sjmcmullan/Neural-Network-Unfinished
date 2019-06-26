class Weight:
    def __init__(self, name, value, inputNode, outputNode):
        self.name = name
        self.value = value
        self.inputNode = inputNode
        self.outputNode = outputNode
        self.backwardsPassGradient = 0
        self.sampleGradients = []
    
    def SetName(self, name):
        self.name = name
    
    def SetValue(self, value):
        self.value = value
    
    def SetInputNode(self, node):
        self.inputNode = node
    
    def SetOutputNode(self, node):
        self.outputNode = node
    
    def SetBackwardsPassGradient(self, gradient):
        self.backwardsPassGradient = gradient

    def SetSampleGradients(self, gradients):
        self.sampleGradients = gradients
    
    def GetName(self):
        return self.name

    def GetValue(self):
        return self.value

    def GetInputNode(self):
        return self.inputNode

    def GetOutputNode(self):
        return self.outputNode
    
    def GetSampleGradients(self):
        return self.sampleGradients

    def GetBackwardsPassGradient(self):
        return self.backwardsPassGradient

    def AddToSampleGradients(self, gradient):
        self.GetSampleGradients().append(gradient)
        
    def CalculateGradient(self):
        outputNode = self.GetOutputNode()
        outputNodeLayer = outputNode.GetLayer()
        newGradient = 0

        if outputNodeLayer == "OUTPUT":
            outputOutValue = outputNode.GetOutValue()
            newGradient = -(outputNode.GetTargetValue() - outputOutValue)
            newGradient *= outputOutValue * (1 - outputOutValue)
            newGradient *= self.GetInputNode().GetOutValue()
            self.AddToSampleGradients(newGradient)
        elif outputNodeLayer == "HIDDEN":
            # Partial derivative of etotal in terms of outh(i).
            newGradient = 0

            for weight in outputNode.GetOutputWeights():
                weightOutputNode = weight.GetOutputNode()
                outputOutValue = weightOutputNode.GetOutValue()
                temp = -(weightOutputNode.GetTargetValue() - outputOutValue)
                temp *= outputOutValue * (1 - outputOutValue)
                temp *= weight.GetValue()

                newGradient = newGradient + temp
            # Partial derivative of outh(i) in terms of neth(i).
            outValue = outputNode.GetOutValue()
            newGradient *= outValue * (1 - outValue)
            # Partial derivative of neth(i) in terms of w(i).
            newGradient *= self.GetInputNode().GetOutValue()
            self.AddToSampleGradients(newGradient)

    def CalculateNewValue(self, trainRate, batchSize):
        totalBatchGradient = 1/batchSize * (sum(self.GetSampleGradients()))
        self.SetValue(self.GetValue() - (trainRate * totalBatchGradient))
