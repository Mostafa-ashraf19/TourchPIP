# Build a Deep Learning Framework
 -  A DIY Deep Learning Framework that simulate real- world frame works such as  that's able to handle datasets, split data, train and test models.

## Problem Statement

 - A DIY DL framework is a simpler verion of the available live frameworks such as PyTorch and TensorFlow. SO the goal here is to build a frame work that has 
     * a data module to read and process datasets
     * A NN module to design different architectures
     * An optimization module for training
     * A visualization module to track the training and testing processes
     * A utils module for reading and saving models

## Metrics
 - According to the precision and recall concept when we used in binary class  classification, will apply this concept into multiclass classification,  a typical multi-class classification problem, we need to categorize each sample into 1 of N different classes,
Similar to a binary case, we can define precision and recall for each of the classes. 
So, Accuracy = correct / total size
Or  = true positive + true negative / dataset size 

## Data Exploration and Visualization
  * ### Data Download 
     - Our DataFrame supports downloading and using (MNIST and CIFAR-10) dataset for training and testing till now. The images are divided into a training set and a validation set
  * ### Data Preprocessing 
      - After loading the data we have the option to perform some operation on the data. We Pass images to the data loader with batch size as desired normalize it, convert them into tensors and shuffle it. 
  * ### Data Visualisation  

    <div class="row">
       <div class="column">
         <img src="https://raw.githubusercontent.com/Mostafa-ashraf19/DL_framework-/master/Images/birdy.png?token=AN55WNTF74ET6SEXSCHFXTTAC3UFY"                       height="300" width="300" />
            <div class="caption" style="width: 300; text-align: center" > Data plotting </div>  
       </div>
       <div class="column">
         <img src="https://github.com/Mostafa-ashraf19/DL_framework-/blob/master/Images/CIFAR%20data%20vis.png" height="300" width="300"/>
                  <div class="caption" style="width: 300; text-align: center" > CIFAR_Data_Visualization </div>  
       </div>
       <div class="column">
        <img src="https://github.com/Mostafa-ashraf19/DL_framework-/blob/master/Images/MNIST_Data_vis.png"  height="300" width="300" />
       </div>     
    </div>
##  Implementation
   -The Dataframe core is divided into modules as follows :
   * ### Layers 
       - The dense layer is a neural network layer that is connected deeply, which means each neuron in the dense layer receives input from all neurons of its previous layer. The dense layer is found to be the most commonly used layer in the models.
       - The neurons, within each of the layer of a neural network, perform the same function. They simply calculate the weighted sum of inputs and weights, add the bias and execute an activation function.
       
  * ### Activation
       - __Sigmoid / Logistic__
         - Range: Between 0 and 1

            ****Problems****:
           1. Vanishing Gradient problem: function is flat near 0 and 1 → during back-propagation, the gradients in neurons whose output is near 0 or 1 are nearly 0 (a.k.a saturated neurons). It causes the weights in these neurons unable to update
           2. Output is not zero-centered: makes gradient updates go too far in different directions
           3. Saturates and kills gradients
           4. Slow convergence
           
       - __ReLU__
           1. Simple and efficient: It is said to have 6 times improvement in convergence from tanh function
           2. Range: [0, infinity)
           3. Avoids Vanishing Gradient problem
           4. Can only be used within hidden layer in Neural Network model (output is not scaled)
           
                 ****Problems****:
                1. Some gradients are fragile during training and can die. It causes weight update which will make it never activate on any data point again.
       
       - __TanH__
           1. Tanh(x) = 2Sigmoid(2x)-1
           2. Also known as scaled sigmoid functions. Made to solve the problem of zero-centered
           3. Range: Between -1 to 1 (zero-centered)
           4. Usually used in classification between two classes
           
                 ****Problems****:
                 1. Vanishing Gradient problem
       - __Softmax__
         1. Able to handle multiple classes only one class in other activation functions—normalizes the outputs for each class between 0 and 1, and divides by their sum, giving the probability of the input value being in a specific class.
         2. Useful for output neurons—typically Softmax is used only for the output layer, for neural networks that need to classify inputs into multiple categories.
          
         
  ![activation](https://raw.githubusercontent.com/Mostafa-ashraf19/DL_framework-/master/Images/activation.png?token=AN55WNU3NIXNW22K24ISVHLAC3T5A) \
  * ### Losses 
      - we seek to minimize the error. As such, the objective function is often referred to as a cost function or a loss function and the value calculated by the loss function is referred to as simply “loss.”
The cost or loss function has an important job in that it must faithfully distill all aspects of the model down into a single number in such a way that improvements in that number are a sign of a better model.
In calculating the error of the model during the optimization process, a loss function must be chosen.
This can be a challenging problem as the function must capture the properties of the problem and be motivated by concerns that are important to the project and stakeholders.
 
  * ### Optimization
      - Optimisers are supposed to avoid local minima and head for the global minimum as fast as possible. Different optimisers perform better for different surfaces, but there are a few that have proven useful for common use cases. 
         -  __Gradient Descent__: 
           The basis for pretty much all other optimisation algorithms. As the name suggests, it is the most basic version of a gradient-based optimiser, naively using the same learning rate to update all parameters based on their gradients from the last step.
          - __Adam Optimization__:
          1. Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models.
          2. Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.
          3. Adam is relatively easy to configure where the default configuration parameters do well on most problems.
          
          - __Momentum Optimization__:
           Momentum [1] or SGD with momentum is method which helps accelerate gradients vectors in the right directions so instead of using only the gradient of the current step to guide the search, momentum also accumulates the gradient of the past steps to determine the direction to go,thus leading to faster converging. It is one of the most popular optimization algorithms and many state-of-the-art models are trained using it.
           
            
  
  
 

![DL_chart](https://raw.githubusercontent.com/Mostafa-ashraf19/DL_framework-/master/Images/DL_chart.jpg?token=AN55WNTF3PLJ3TSKGJYPOCLAC3UZC)

