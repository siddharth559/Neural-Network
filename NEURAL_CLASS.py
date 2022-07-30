import numpy as np

def sigmoid(j):
    return 1/(1+np.e**(-1*j))



class neural_net:

    def __init__(self, inputs, outputs, **layers):
        '''
        Put number of input neurons in input and the 
        layers(name = number of neurons) at layers.
        later the weights of repective layers will be
        stored there, thence can be accessed the layer name.
        '''
        self.inputs = inputs
        self.outputs = outputs
        self.weights = [inputs]+[layers[i] for i in layers]+[outputs]
        self.activation = np.vectorize(sigmoid)

        for i in range(len(self.weights)-1):
            self.weights[i] = np.matrix(np.random.random((self.weights[i],self.weights[i+1])))
        self.weights = self.weights[0:-1]


    def set_activation(self,fun):
        self.activation = np.vectorize(fun)


    def calc_out(self):
        '''
        calculates the expected output
        '''
        self.layer = [self.train_set]
        for i in self.weights:
            self.layer.append(self.activation(i.transpose()*self.layer[-1]))
        return self.layer[-1]


    def train(self,train_set,output_set): 
        '''
        enter a numpy array or list(one_dimensional) as train_set
        put the (one dimensional)iterator of expected output in output_set
        '''
        self.train_set = np.matrix(train_set).transpose()
        self.desired_output = np.matrix(output_set).transpose()
        self.ΔWeight = []
        self.calc_out() #calculated the output
        _prev_ = 0
        for i in range(-1,-1*len(self.weights)-1,-1): 
            if len(self.ΔWeight) == 0:
                O = self.layer[-1]
                O = np.array(O.transpose()) * np.array((1-O).transpose()) * np.array((self.desired_output-O).transpose())
                _prev_ = np.matrix(O).transpose()
            else:
                O = self.weights[i+1]
                O = O * _prev_
                O = np.array(self.layer[i].transpose())*np.array(1-self.layer[i].transpose())*np.array(O.transpose())
                _prev_ = O

            self.ΔWeight.insert(0,self.layer[i-1]*np.matrix(O))

        #print('weights are:',[i for i in self.weights],'delta weights are:',[i for i in self.ΔWeight])
        for i in range(len(self.weights)):
            self.weights[i] -= self.ΔWeight[i]

if __name__ == "__main__":
    a = neural_net(2,1,hidden = 2)
    #a.train_set = np.matrix([0.4,-0.7]).transpose()
    #print(a.calc_out())

    weights = [ np.matrix([[0.1,0.4],
                          [-0.2,0.2]]), np.matrix([[0.2],
                                                   [0.5]])  ]

    a.weights = weights
    
    a.train([0.4,-0.7],[0.1])
    print(a.ΔWeight)
    print(a.layer)
    a.train([0.3,-0.5],[0.05])
    a.train([0.6,0.1],[0.3])
    a.train([0.2,0.4],[0.25])
    a.train([0.1,0.2],[0.12])
    

        

