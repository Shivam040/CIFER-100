# CIFAR-100
Models for CIFAR-100
CIFAR-100 Classification 

This project focuses on solving the CIFAR-100 image classification problem using two distinct architectures: 

ResNet Architecture 

Pretrained-Xception Architecture   

Dataset 

The CIFAR-100 dataset consists of 60,000 color images (32x32 pixels) in 100 classes, with 6,000 images per class. The dataset is split into: 

Training Set: 50,000 images 

Test Set: 10,000 images 

  

Training Data Variety 

Data Augmentation: For each original image 2 more images were generated and then merged with the original training dataset. 

The training dataset contains 15000 instances of each class: 

Labels:  

beaver, dolphin, otter, seal, whale, aquarium fish, flatfish, ray, shark, trout, orchids, poppies, roses, sunflowers, tulips, bottles, bowls, cans, cups, plates, apples, mushrooms, oranges, pears, sweet peppers, clock, computer keyboard, lamp, telephone, television, bed, chair, couch, table, wardrobe, bee, beetle, butterfly, caterpillar, cockroach, skyscraper, cloud, forest, mountain, plain, sea, camel, cattle, chimpanzee, elephant, kangaroo, fox, porcupine, possum, raccoon, skunk, crab, lobster, snail, spider, worm, baby, boy, girl, man, woman, crocodile, dinosaur, lizard, snake, turtle, hamster, mouse, rabbit, shrew, squirrel, maple, oak, palm, pine, willow, bicycle, bus, motorcycle, pickup truck, train, lawn-mower, rocket, streetcar, tank, tractor, bear, leopard, lion, tiger, wolf, bridge, castle, house, road. 

  

  

Data Loading Functions 

To read the dataset, the following functions were implemented: 

 
 
  

GPU Configuration 

To optimize the GPU memory usage, the following configuration was applied: 

Title: Inserting image... 
 This ensures that TensorFlow dynamically allocates memory as needed during training instead of pre-allocating the entire GPU memory. 

  

Approaches 

1. ResNet Architecture 

A residual neural network is a deep learning architecture in which the layers learn residual functions with reference to the layer inputs. 

Architecture Highlights: 

Residual Blocks: Enables deeper networks by using shortcut connections to prevent vanishing gradients. 

Shortcut Connection: If conv_shortcut=True, the input is used as the shortcut, a 1x1 convolution is applied to match the dimensions.  

Followed by Batch Normalization. 

First Convolution Layer: 

Uses a 3×3 kernel with stride=1 

Followed by Batch Normalization and ReLU activation. 

Second Convolution Layer: 

Uses another 3×3 kernel with padding='same'. 

Followed by Batch Normalization. 

The shortcut is added to the processed tensor to allow gradient flow. 

Final Activation: ReLU activation. 

ModifyNet Block: This function builds a complete ResNet model using multiple residual blocks. 

Input Layer: Takes 32x32x3 color images as input. 

1st Convolutional Layer: 128 filters, 3×3 kernel 

Followed with Batch Normalization. 

Activation: ReLU activation. 

Dropout: Reduces overfitting, dropout_rate=0.5. 

Residual Blocks: Configured using num_blocks_list=[2, 2, 2, 2], meaning: 

2 blocks in each stage. 

Each stage has double the number of filters from the previous one. 

Strides=2 for the first block in each new stage (downsampling) 

Global Average Pooling: Reduces model complexity instead of using fully connected layers. 

Final Dense Layer: 100 output neurons with a softmax activation 

Optimizer: Model was compiled 3 times, adam(learning_rate=0.001), adam(learning_rate=0.0001) and  adam(learning_rate=0.00005) 

Loss Function: Sparse Categorical Crossentropy 

Metrics: Sparce Categorical Accuracy 

  

2. Pretrained Xception Architecture 

The Xception (Extreme Inception) model is a deep convolutional neural network that utilizes depthwise separable convolutions for more efficient feature extraction. This implementation transfers learning from ImageNet and fine-tunes it for CIFAR-100. 

Architecture Highlights: 

Preprocessing Function: Resizes images to 71x71 to match the input requirements of Xception. 

Uses tf.keras.applications.xception.preprocess_input() to normalize pixel values. 

Base Model: Xception (Pretrained on ImageNet) 

include_top=False: Removes fully connected layers. 

Global Average Pooling (GlobalAveragePooling2D): Reduces feature maps into a single vector. 

Dense Layer with 512 neurons with ReLU activation. 

Dropout Layer (0.5) to prevent overfitting. 

Output Layer with 100 neurons with softmax activation. 

Freezing Pretrained Layers: All layers of Xception are frozen to retain pretrained knowledge from ImageNet for 20 epochs. This allows the model to focus on learning CIFAR-100-specific features in the final layers. 

Unfreezing Pretrained Layers: For 26 epochs. 

Optimizer: Model was compiled 3 times, adam(learning_rate=0.0002) , adam(learning_rate=0.000005) and  adam(learning_rate=0. 000002)  

Loss Function: Sparse Categorical Crossentropy 

Metrics: Sparce Categorical Accuracy 

  

Results 

Model 

Test Sparse Categorical Accuracy (%) 

Training Time (epochs) 

Parameters 

ResNet Architecture 

65.49 

26 

46,203,236 

Pretrained Xception 

72.93 

45 

21,961,868 

  

Requirements 

Python 3.8+ 

TensorFlow 2.x 

NumPy 

Matplotlib 

Scikit-learn 

pickle 

Conclusion 

Pretrained Xception  outperformed the others with the highest test Sparse Categorical Accuracy of 72.93%, followed closely by ResNet with 65.49%. 
