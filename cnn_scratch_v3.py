import numpy as np
import matplotlib.pyplot as plt 
import pickle
import os
from PIL import Image

learning_rate = 0.001
num_epoch = 5

training_data_path = '/home/rakumar/NN/train/'
test_data_path = '/home/rakumar/NN/TEST/'

Train = False
Test = False
Labels_known = False

model_config = {
    'filter_size' : 3,
    'num_filters' : 8,
    'softmax_input_node' : 13 * 13 * 8,
    'softmax_output_node' : 10
}

def relu(Z):
    A = np.maximum(0,Z)
    return A

class Conv_op:
    def __init__(self, num_filters, filter_size, convFilter):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = convFilter

    def image_region(self, image):    # generator
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width, _  = image.shape

        self.image = image

        for j in range(height - self.filter_size + 1):
            for k in range(width - self.filter_size + 1):
                image_patch = image[j:(j+self.filter_size), k:(k+self.filter_size)]
                yield image_patch.T, j, k
    
    def forward_prop(self, image):
        self.image = image
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width, _  = image.shape
            image = image.T[0]

        conv_out = np.zeros((height-self.filter_size+1, width-self.filter_size+1, self.num_filters)) #(26,26,5)

        for image_patch, i, j in self.image_region(image):
            conv_out[i, j] = np.sum(image_patch*self.conv_filter.T, axis=(1,2)) # (3,3)*(5,3,3) ==> (5,3,3)
                                                                                # where, num_filter=5, filter_size=3
                                                                                # conv_out[j, k].shape = (1,5)                 
        return conv_out                                                         # conv_out.shape = (26, 26, 5)

    def back_prop(self, dL_out, learning_rate):             # dL_out --> output of maxPool Backprpagation
        dL_dF_params = np.zeros((self.conv_filter.T.shape))
        for image_patch, i, j in self.image_region(self.image):
            for k in range(self.num_filters):
                if i < dL_out.shape[0] and j < dL_out.shape[1] and k < dL_out.shape[2]:   ##### ??? Lenet5--  #####
                    dL_dF_params[k] = dL_dF_params[k] + image_patch*dL_out[i,j,k]

        # filter parameres update
        self.conv_filter -= learning_rate*dL_dF_params.T
        return dL_dF_params



class Max_pool:
    def __init__(self, filter_size):
        self.filter_size = filter_size

    def image_region(self, image):    # generator
        new_height = image.shape[0] // self.filter_size
        new_width  = image.shape[1] // self.filter_size
        self.image = image

        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[ (i*self.filter_size) : (i*self.filter_size + self.filter_size), (j*self.filter_size) : (j*self.filter_size + self.filter_size)]
                yield image_patch, i, j
    
    def forward_prop(self, image):
        if len(image.shape) == 2:
            height, width = image.shape
            num_filters = 1
        else:
            height, width, num_filters = image.shape
        maxpool_out = np.zeros((height//self.filter_size, width//self.filter_size, num_filters))

        for image_patch, i, j in self.image_region(image):
            maxpool_out[i, j] = np.sum(image_patch, axis=(0,1)) 

        return maxpool_out

    def back_prop(self, dL_out):             # dL_out --> output of softmax Backprpagation
        dL_dmax_pool = np.zeros(self.image.shape)
        for image_patch, i, j in self.image_region(self.image):
            height, width, num_filters = image_patch.shape
            maximum_val = np.amax(image_patch, axis=(0,1))

            for i1 in range(height):
                for j1 in range(width):
                    for k1 in range(num_filters):
                        if image_patch[i1,j1,k1] == maximum_val[k1]:
                            dL_dmax_pool[i*self.filter_size +i1, j*self.filter_size+j1, k1] = dL_out[i,j,k1]

        return dL_dmax_pool


class Softmax:
    def __init__(self, softmaxWeight, softmaxBias):
        self.weight = softmaxWeight
        self.bias = softmaxBias
    
    def forward_prop(self, image):
        self.original_im_shape = image.shape   # used in backProp
        image_modified = image.flatten()
        self.modified_input = image_modified   # to be used in backProp
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)              # used Softmax Transformation..
        return exp_out/np.sum(exp_out, axis=0)    # logits .. value is prpbability(i.e, between 0 and 1)

    def back_prop(self, dL_out, learning_rate):             # dL_out --> ?
        for i, grad in enumerate(dL_out):
            if grad == 0:
                continue

            transformation_eq = np.exp(self.out)
            S_total = np.sum(transformation_eq)

            # gradients w.r.t out (z)
            dy_dz = -transformation_eq[i] * transformation_eq / (S_total ** 2)
            dy_dz[i] = transformation_eq[i] * (S_total - transformation_eq[i]) / (S_total ** 2)

            # gradients of totals against weight/bias/input
            dz_dw = self.modified_input
            dz_db = 1
            dz_d_inp = self.weight

            # gradients of loss against totals
            dL_dz = grad * dy_dz

            # gradients of loss against weight/bias/input
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz

        # update weight and bias
        self.weight -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db

        return dL_d_inp.reshape(self.original_im_shape)


def cnn_forward_prop(conv, pool, softmax, image, label=None):
    out_p = conv.forward_prop((image / 255) - 0.5)
    out_p = relu(out_p)
    out_p = pool.forward_prop(out_p)
    out_p = softmax.forward_prop(out_p)

    # calculate cross-entropy loss and accuracy
    if label != None:
        cross_ent_loss = -np.log(out_p[label])
        accuracy_eval = 1 if np.argmax(out_p) == label else 0
        return out_p, cross_ent_loss, accuracy_eval
    else:
        return out_p

def training_cnn(conv, pool, softmax, image, label, learning_rate = learning_rate):
    # forward prop
    out, loss, acc = cnn_forward_prop(conv, pool, softmax,image, label)

    # calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1/out[label]

    # backPropagation
    grad_back = softmax.back_prop(gradient, learning_rate)
    grad_back = pool.back_prop(grad_back)
    grad_back = conv.back_prop(grad_back, learning_rate)

    return loss, acc



def train(conv, pool, softmax, train_images, train_labels, num_epoch):
    for epoch in range(num_epoch):
        print('=========== Epoch %d =========== '%(epoch+1))

        # shuffle the training data
        shuffle_data = np.random.permutation(len(train_images))
        train_images = train_images[shuffle_data]
        train_labels = train_labels[shuffle_data]

        # training the CNN
        loss = 0
        num_correct = 0

        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i % 100 == 0:
                print('[%d, %d/%d] : Average loss: %.3f   Accuracy: %d%%' %(epoch+1, i, len(train_images), loss/100, num_correct))
                loss = 0
                num_correct = 0

            l, accu = training_cnn(conv, pool, softmax, im, label)
            loss += l
            num_correct += accu


def get_dataset_for_training(training_data_path):
    train_list = sorted(os.listdir(training_data_path))

    data = []
    for imgdir in train_list:
        images = sorted(os.listdir(training_data_path+imgdir))
        label = imgdir
        for image in images:
            data.append((training_data_path+imgdir+'/'+image, label))

    imgs=[]
    lbs=[]
    for eachdata in data:
        im = np.asarray(Image.open(eachdata[0]))
        imgs.append(im)
        l = eachdata[1]
        lbs.append(int(l))

    X = np.dstack(imgs)
    X = np.rollaxis(X,-1)

    Y = np.stack(lbs)

    return X, Y

if Train:
    X, Y = get_dataset_for_training(training_data_path= training_data_path)

    filter_size = model_config['filter_size']
    num_filters = model_config['num_filters']
    softmax_input_node = model_config['softmax_input_node']
    softmax_output_node = model_config['softmax_output_node']

    convFilter = np.random.randn( filter_size, filter_size, num_filters) / (filter_size*filter_size)
    softmaxWeight = np.random.randn(softmax_input_node, softmax_output_node) / softmax_input_node
    softmaxBias = np.zeros(softmax_output_node)

    conv = Conv_op(num_filters=num_filters, filter_size=filter_size, convFilter=convFilter)      # 28x28x1 --> 26x26x8
    pool = Max_pool(filter_size=2)                                                               # 26x26x8 --> 13x13x8
    softmax = Softmax(softmaxWeight=softmaxWeight, softmaxBias=softmaxBias)                      # 13x13x8 --> 10
    
    train(conv, pool, softmax, train_images=X, train_labels=Y, num_epoch=num_epoch)

    parameters = {  'conv_filter' : conv.conv_filter,
                    'softmax_weight' : softmax.weight,
                    'softmax_bias' : softmax.bias
    }

    with open('/home/rakumar/NN/cnnModel_parameters.clf', 'wb') as f:
                pickle.dump(parameters, f)


if Test:
    # load pre-trained model
    with open('/home/rakumar/NN/cnnModel_parameters.clf', 'rb') as f:
                model_parameters = pickle.load(f)

    convFilter = model_parameters['conv_filter']
    softmaxWeight = model_parameters['softmax_weight']
    softmaxBias = model_parameters['softmax_bias']

    conv = Conv_op(num_filters=model_config['num_filters'], filter_size=model_config['filter_size'], convFilter=convFilter)
    pool = Max_pool(filter_size=2)                                       
    softmax = Softmax(softmaxWeight=softmaxWeight, softmaxBias=softmaxBias)

    images_dir = sorted(os.listdir(test_data_path))

    loss = 0
    num_correct = 0
    for i, img in enumerate(images_dir):
        im = Image.open(test_data_path+img)
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = np.asarray(im)
        

        if Labels_known:
            pred, l, accu = cnn_forward_prop(conv, pool, softmax,im, label=int(img.split('_')[0]))
            print('image : ' + img + '     prediction: '+str(np.argmax(pred)))
            loss += l
            num_correct += accu

        else:
            pred = cnn_forward_prop(conv, pool, softmax, im)
            p = np.argmax(pred)
            print(img, p)

    if Labels_known:
        num_tests = len(images_dir)
        print('Test Loss: ', loss/num_tests)
        print('Test Accuracy: ', num_correct/num_tests)
