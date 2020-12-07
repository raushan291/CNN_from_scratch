import numpy as np
import os
from PIL import Image
import pickle
from cnn_scratch_v3 import Conv_op, Max_pool, Softmax, relu

Train = False
Labels_known = True

train_data_path = '/home/rakumar/NN/TEST/'
test_dir_path = '/home/rakumar/NN/TEST/'
model_path = '/home/rakumar/NN/leNetModel_parameters.clf'


def parameter_initalization(in_features=None, out_features=None):
    convFilter = np.random.randn( 5, 5, out_features) / (5*5)
    softmaxWeight = None
    softmaxBias = None

    if in_features != None:
        softmaxWeight = np.random.randn(in_features, out_features) / in_features  # 16*16*4
        softmaxBias = np.zeros(out_features)

    return convFilter, softmaxWeight, softmaxBias

class LeNet5:
    def __init__(self, num_classes):
        self.convFilter1, _, _ = parameter_initalization(None, 6)
        self.convFilter2, _, _ = parameter_initalization(None, 16)

        _, self.softmaxWeight1, self.softmaxBias1 = parameter_initalization(16*4*4, 120)
        _, self.softmaxWeight2, self.softmaxBias2 = parameter_initalization(120, 84)
        _, self.softmaxWeight3, self.softmaxBias3 = parameter_initalization(84, num_classes)

        self.conv1 = Conv_op(num_filters=6, filter_size=5, convFilter=self.convFilter1)
        self.conv2 = Conv_op(num_filters=16, filter_size=5, convFilter=self.convFilter2)

        self.relu = relu
        self.maxpool = Max_pool(filter_size=2)

        self.fc1 = Softmax(softmaxWeight=self.softmaxWeight1, softmaxBias=self.softmaxBias1)
        self.fc2 = Softmax(softmaxWeight=self.softmaxWeight2, softmaxBias=self.softmaxBias2)
        self.fc3 = Softmax(softmaxWeight=self.softmaxWeight3, softmaxBias=self.softmaxBias3)

    def forward(self, x):
        x = self.conv1.forward_prop((x / 255) - 0.5)
        x = self.relu(x)
        x = self.maxpool.forward_prop(x)

        x = self.conv2.forward_prop((x / 255) - 0.5)
        x = self.relu(x)
        x = self.maxpool.forward_prop(x)

        x = self.fc1.forward_prop(x)
        x = self.relu(x)

        x = self.fc2.forward_prop(x)
        x = self.relu(x)

        x = self.fc3.forward_prop(x)

        return x
    
    def backpropagation(self, x, label, gradient, learning_rate = 0.001):

        grad_back = self.fc3.back_prop(gradient, learning_rate)
        grad_back = self.fc2.back_prop(grad_back, learning_rate)
        grad_back = self.fc1.back_prop(grad_back, learning_rate)

        grad_back = self.maxpool.back_prop(grad_back)
        grad_back = self.conv2.back_prop(grad_back, learning_rate)
        grad_back = self.maxpool.back_prop(grad_back.T)
        grad_back = self.conv1.back_prop(grad_back, learning_rate)

  
net = LeNet5(num_classes=10)

### Train the network ###
if Train:
    images_dir = sorted(os.listdir(train_data_path))

    for epoch in range(1):
        loss = 0
        num_correct = 0
        for i, img in enumerate(images_dir):
            # get the inputs
            im = Image.open(train_data_path+img)
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.asarray(im)

            label=int(img.split('_')[0])

            # forward prop
            out = net.forward(im)

            cross_ent_loss = -np.log(out[label])
            accuracy_eval = 1 if np.argmax(out) == label else 0

            # calculate initial gradient
            gradient = np.zeros(10)
            gradient[label] = -1/out[label]

            # backPropagation
            net.backpropagation(im, label, gradient, learning_rate = 0.001)

            loss += cross_ent_loss
            num_correct += accuracy_eval

            # print statistics
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss / 100))
                loss = 0.0

    parameters = {  'conv_filter1' : net.conv1.conv_filter,
                    'conv_filter2' : net.conv2.conv_filter,

                    'softmax_weight1' : net.fc1.weight,
                    'softmax_weight2' : net.fc2.weight,
                    'softmax_weight3' : net.fc3.weight,

                    'softmax_bias1' : net.fc1.bias,
                    'softmax_bias2' : net.fc2.bias,
                    'softmax_bias3' : net.fc3.bias
        }

    with open(model_path, 'wb') as f:
                pickle.dump(parameters, f)


def test():
    net = LeNet5(num_classes=10)

    with open(model_path, 'rb') as f:
                  model_parameters = pickle.load(f)

    convFilter1 = model_parameters['conv_filter1']
    convFilter2 = model_parameters['conv_filter2']

    softmaxWeight1 = model_parameters['softmax_weight1']
    softmaxWeight2 = model_parameters['softmax_weight2']
    softmaxWeight3 = model_parameters['softmax_weight3']

    softmaxBias1 = model_parameters['softmax_bias1']
    softmaxBias2 = model_parameters['softmax_bias2']
    softmaxBias3 = model_parameters['softmax_bias3']


    net.conv1 = Conv_op(num_filters=6, filter_size=5, convFilter=convFilter1)
    net.conv2 = Conv_op(num_filters=16, filter_size=5, convFilter=convFilter2)

    net.fc1 = Softmax(softmaxWeight=softmaxWeight1, softmaxBias=softmaxBias1)
    net.fc2 = Softmax(softmaxWeight=softmaxWeight2, softmaxBias=softmaxBias2)
    net.fc3 = Softmax(softmaxWeight=softmaxWeight3, softmaxBias=softmaxBias3)

    images_dir = sorted(os.listdir(test_dir_path))
    totalCorrectCount = 0
    totalImgs = len(images_dir)

    for img in images_dir:
      x = Image.open(test_dir_path+img)
      x = x.resize((28, 28), Image.ANTIALIAS)
      x = np.asarray(x)

      output = net.forward(x)
      # print(output)
      pred = np.argmax(output)
      prob = round(output[pred], 2)
      if Labels_known:
        if pred == int(img.split('_')[0]):
            totalCorrectCount += 1
      print(img.split('/')[-1]+" ==> "+str(pred) + '  (prob = '+ str(prob) + ')')
    
    if Labels_known:
      accuracy = totalCorrectCount/totalImgs
      print("Test accuracy = ", accuracy)

test()