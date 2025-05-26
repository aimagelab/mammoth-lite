Creating Custom Backbones
========================

This example shows how to create custom neural network backbones in Mammoth Lite. You'll learn to implement your own architectures, register them with the framework, and use them in continual learning experiments.

.. note::
   This example is based on the Jupyter notebook ``examples/notebooks/create_a_backbone.ipynb``. You can run it interactively or follow along here.

Learning Objectives
------------------

By the end of this example, you'll understand:

* How to create custom neural network architectures for continual learning
* The ``MammothBackbone`` base class and its requirements
* How to implement flexible return types for different algorithms
* How to register multiple backbone variants
* How to use custom backbones in experiments

Understanding Backbone Architecture
----------------------------------

What is a Backbone?
~~~~~~~~~~~~~~~~~

In Mammoth Lite, a **backbone** is the neural network architecture that:

* Extracts features from input data
* Provides the final classification layer
* Can return different outputs (logits, features, both) depending on algorithm needs
* Serves as the foundation for continual learning models

Backbone vs Model Distinction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Backbone**: The neural network architecture (ResNet, CNN, Transformer)

**Model**: The continual learning algorithm (SGD, replay, regularization) that uses the backbone

This separation allows you to:

* Test the same continual learning algorithm with different architectures
* Use the same backbone with different continual learning approaches
* Easily swap architectures without changing algorithm logic

Backbone Requirements
--------------------

All custom backbones must:

**Inherit from MammothBackbone**
  Provides the base functionality and interface consistency

**Implement Flexible Forward Pass**
  Support different return types for various continual learning algorithms

**Accept num_classes Parameter**
  The number of output classes (set automatically by the dataset)

**Handle Different Return Types**
  Support at minimum ``"out"`` and preferably ``"features"`` return types

Creating a Simple Custom Backbone
---------------------------------

Let's start with a basic CNN backbone:

.. code-block:: python

   from mammoth_lite import register_backbone, MammothBackbone, ReturnTypes
   from torch import nn
   from torch.nn import functional as F

   class CustomCNN(MammothBackbone):
       """
       A custom CNN backbone for continual learning.
       
       Simple architecture with two convolutional layers and a classifier.
       """

       def __init__(self, num_classes, num_channels=32, input_size=32):
           """
           Initialize the custom CNN.
           
           Args:
               num_classes: Number of output classes (set by dataset)
               num_channels: Number of filters in first conv layer
               input_size: Input image size (height/width)
           """
           super().__init__()

           # Feature extraction layers
           self.layer1 = nn.Conv2d(
               in_channels=3,              # RGB input
               out_channels=num_channels,
               kernel_size=3,
               stride=1,
               padding=1
           )
           
           self.layer2 = nn.Conv2d(
               in_channels=num_channels,
               out_channels=num_channels * 2,
               kernel_size=3,
               stride=1,
               padding=1
           )
           
           self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

           # Classifier layer
           # After two maxpool operations: input_size // 4
           classifier_input_size = num_channels * 2 * (input_size // 4) ** 2
           self.classifier = nn.Linear(classifier_input_size, num_classes)

       def forward(self, x, returnt=ReturnTypes.OUT):
           """
           Forward pass with flexible return types.
           
           Args:
               x: Input tensor [batch_size, 3, height, width]
               returnt: What to return ("out", "features", "both")
               
           Returns:
               Depends on returnt parameter:
               - "out": Classification logits
               - "features": Feature representation
               - "both": Tuple of (logits, features)
           """
           # Feature extraction
           out1 = self.maxpool(F.relu(self.layer1(x)))     # [B, C, H/2, W/2]
           out2 = self.maxpool(F.relu(self.layer2(out1)))  # [B, 2C, H/4, W/4]
           
           # Flatten features
           features = out2.view(out2.size(0), -1)          # [B, 2C*(H/4)*(W/4)]
           
           # Classification
           logits = self.classifier(features)               # [B, num_classes]
           
           # Return based on requested type
           if returnt == ReturnTypes.OUT:
               return logits
           elif returnt == ReturnTypes.FEATURES:
               return features
           elif returnt == ReturnTypes.BOTH:
               return logits, features
           else:
               raise ValueError(f"Unknown return type: {returnt}")

Key Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~

**Parameter Handling**
  Only ``num_classes`` is mandatory - other parameters can be customized during registration.

**Return Type Flexibility**
  Different continual learning algorithms need different outputs:
  
  * Standard training: ``"out"`` (logits)
  * Rehearsal methods: ``"features"`` (for memory storage)
  * Advanced methods: ``"both"`` (logits and features)

**Feature Flattening**
  Convert 2D feature maps to 1D vectors for the classifier.

**Error Handling**
  Raise clear errors for unsupported return types.

Registering Backbone Variants
----------------------------

Unlike models and datasets, backbone registration uses functions that return instances:

.. code-block:: python

   @register_backbone(name='custom-cnn-v1')
   def custom_cnn_v1(num_classes):
       """
       Register a small version of the custom CNN.
       
       Args:
           num_classes: Number of output classes (passed automatically)
           
       Returns:
           CustomCNN instance with specific parameters
       """
       return CustomCNN(
           num_classes=num_classes,
           num_channels=32,        # Small version
           input_size=32
       )

   @register_backbone(name='custom-cnn-v2')
   def custom_cnn_v2(num_classes):
       """
       Register a larger version of the custom CNN.
       """
       return CustomCNN(
           num_classes=num_classes,
           num_channels=64,        # Larger version
           input_size=32
       )

Why Function-Based Registration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function-based registration allows:

* **Multiple Variants**: Create different versions with different parameters
* **Dynamic Configuration**: Potentially accept additional CLI arguments
* **Lazy Initialization**: Only create instances when needed
* **Parameter Validation**: Check parameters before creating instances

Using Custom Backbones
----------------------

Once registered, use your custom backbones in experiments:

.. code-block:: python

   from mammoth_lite import load_runner, train

   # Use the small version
   model, dataset = load_runner(
       model='sgd',
       dataset='seq-cifar10',
       args={
           'lr': 0.1,
           'n_epochs': 1,
           'batch_size': 32,
           'backbone': 'custom-cnn-v1'  # Specify custom backbone
       }
   )

   print(f"Using backbone: {type(model.net).__name__}")
   print(f"Number of parameters: {sum(p.numel() for p in model.net.parameters()):,}")

   train(model, dataset)

**Expected Output:**

.. code-block:: text

   Using backbone: CustomCNN
   Number of parameters: 85,962

   Task 1: 100%|██████████| 1563/1563 [01:25<00:00, 18.32it/s]
   Accuracy on task 1: 64.2%

Comparing Backbone Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare different backbone sizes
   for backbone_name in ['custom-cnn-v1', 'custom-cnn-v2', 'resnet18']:
       print(f"\\nTesting {backbone_name}:")
       
       model, dataset = load_runner(
           'sgd', 'seq-cifar10',
           {'n_epochs': 1, 'backbone': backbone_name}
       )
       
       # Count parameters
       n_params = sum(p.numel() for p in model.net.parameters())
       print(f"Parameters: {n_params:,}")
       
       # Quick training test
       # train(model, dataset)  # Uncomment to actually train

Advanced Backbone Examples
-------------------------

ResNet-Style Backbone
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomResNetBlock(nn.Module):
       """Basic ResNet block with skip connections."""
       
       def __init__(self, in_channels, out_channels, stride=1):
           super().__init__()
           
           self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
           self.bn1 = nn.BatchNorm2d(out_channels)
           self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
           self.bn2 = nn.BatchNorm2d(out_channels)
           
           # Skip connection
           self.skip = nn.Sequential()
           if stride != 1 or in_channels != out_channels:
               self.skip = nn.Sequential(
                   nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                   nn.BatchNorm2d(out_channels)
               )

       def forward(self, x):
           out = F.relu(self.bn1(self.conv1(x)))
           out = self.bn2(self.conv2(out))
           out += self.skip(x)  # Skip connection
           return F.relu(out)

   class CustomResNet(MammothBackbone):
       """Custom ResNet-style backbone."""
       
       def __init__(self, num_classes, layers=[2, 2, 2, 2], channels=[64, 128, 256, 512]):
           super().__init__()
           
           self.in_channels = channels[0]
           
           # Initial convolution
           self.conv1 = nn.Conv2d(3, channels[0], 7, 2, 3, bias=False)
           self.bn1 = nn.BatchNorm2d(channels[0])
           self.maxpool = nn.MaxPool2d(3, 2, 1)
           
           # ResNet layers
           self.layer1 = self._make_layer(channels[0], layers[0], 1)
           self.layer2 = self._make_layer(channels[1], layers[1], 2)
           self.layer3 = self._make_layer(channels[2], layers[2], 2)
           self.layer4 = self._make_layer(channels[3], layers[3], 2)
           
           # Classifier
           self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
           self.fc = nn.Linear(channels[3], num_classes)

       def _make_layer(self, out_channels, blocks, stride):
           layers = []
           layers.append(CustomResNetBlock(self.in_channels, out_channels, stride))
           self.in_channels = out_channels
           
           for _ in range(1, blocks):
               layers.append(CustomResNetBlock(out_channels, out_channels))
               
           return nn.Sequential(*layers)

       def forward(self, x, returnt=ReturnTypes.OUT):
           # Initial processing
           x = F.relu(self.bn1(self.conv1(x)))
           x = self.maxpool(x)
           
           # ResNet layers
           x = self.layer1(x)
           x = self.layer2(x)
           x = self.layer3(x)
           x = self.layer4(x)
           
           # Global average pooling
           x = self.avgpool(x)
           features = x.view(x.size(0), -1)
           
           # Classification
           logits = self.fc(features)
           
           if returnt == ReturnTypes.OUT:
               return logits
           elif returnt == ReturnTypes.FEATURES:
               return features
           elif returnt == ReturnTypes.BOTH:
               return logits, features
           else:
               raise ValueError(f"Unknown return type: {returnt}")

   @register_backbone(name='custom-resnet-small')
   def custom_resnet_small(num_classes):
       return CustomResNet(num_classes, layers=[1, 1, 1, 1], channels=[32, 64, 128, 256])

Attention-Based Backbone
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class SelfAttention(nn.Module):
       """Simple self-attention mechanism."""
       
       def __init__(self, in_dim):
           super().__init__()
           self.query = nn.Linear(in_dim, in_dim)
           self.key = nn.Linear(in_dim, in_dim)
           self.value = nn.Linear(in_dim, in_dim)
           self.scale = in_dim ** -0.5

       def forward(self, x):
           B, C, H, W = x.shape
           x_flat = x.view(B, C, H*W).transpose(1, 2)  # [B, HW, C]
           
           Q = self.query(x_flat)
           K = self.key(x_flat)
           V = self.value(x_flat)
           
           attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
           out = torch.bmm(attn, V)
           
           return out.transpose(1, 2).view(B, C, H, W)

   class AttentionCNN(MammothBackbone):
       """CNN with self-attention layers."""
       
       def __init__(self, num_classes):
           super().__init__()
           
           self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
           self.attention1 = SelfAttention(64)
           
           self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
           self.attention2 = SelfAttention(128)
           
           self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
           self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
           
           self.classifier = nn.Linear(256, num_classes)

       def forward(self, x, returnt=ReturnTypes.OUT):
           # Conv + Attention layers
           x = F.relu(self.conv1(x))
           x = self.attention1(x)
           
           x = F.relu(self.conv2(x))
           x = self.attention2(x)
           
           x = F.relu(self.conv3(x))
           x = self.global_pool(x)
           
           features = x.view(x.size(0), -1)
           logits = self.classifier(features)
           
           if returnt == ReturnTypes.OUT:
               return logits
           elif returnt == ReturnTypes.FEATURES:
               return features
           elif returnt == ReturnTypes.BOTH:
               return logits, features

   @register_backbone(name='attention-cnn')
   def attention_cnn(num_classes):
       return AttentionCNN(num_classes)

Backbone Design Considerations
-----------------------------

Architecture Choices
~~~~~~~~~~~~~~~~~~~

**Depth vs Width Trade-off**
  Deeper networks can learn more complex features but may be harder to train continuously.

**Skip Connections**
  Help with gradient flow and can reduce catastrophic forgetting.

**Normalization**
  BatchNorm can help but may interfere with some continual learning algorithms.

**Pooling Strategy**
  Global average pooling often works better than fully connected layers for continual learning.

Continual Learning Specific Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Feature Reusability**
  Design features that can be useful across multiple tasks.

**Modular Architecture**
  Consider whether parts of the network can be frozen or adapted independently.

**Memory Efficiency**
  Balance capacity with memory constraints for rehearsal-based methods.

**Gradient Flow**
  Ensure gradients can flow effectively for stable continual learning.

Testing and Validation
---------------------

Backbone Testing
~~~~~~~~~~~~~~

.. code-block:: python

   def test_backbone():
       """Test custom backbone functionality."""
       import torch
       
       # Create backbone instance
       backbone = CustomCNN(num_classes=10)
       
       # Test input shapes
       x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
       
       # Test different return types
       logits = backbone(x, ReturnTypes.OUT)
       features = backbone(x, ReturnTypes.FEATURES)
       both = backbone(x, ReturnTypes.BOTH)
       
       print(f"Input shape: {x.shape}")
       print(f"Logits shape: {logits.shape}")
       print(f"Features shape: {features.shape}")
       print(f"Both output: {type(both)}, lengths: {len(both)}")
       
       # Test parameter count
       n_params = sum(p.numel() for p in backbone.parameters())
       print(f"Parameters: {n_params:,}")
       
       assert logits.shape == (4, 10)  # Correct output shape
       assert len(both) == 2           # Both returns tuple
       
       print("✓ Backbone tests passed")

   test_backbone()

Performance Comparison
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def compare_backbones():
       """Compare different backbone architectures."""
       import time
       import torch
       
       backbones = {
           'custom-cnn-v1': custom_cnn_v1(10),
           'custom-cnn-v2': custom_cnn_v2(10),
       }
       
       x = torch.randn(32, 3, 32, 32)  # Batch for timing
       
       for name, backbone in backbones.items():
           # Parameter count
           n_params = sum(p.numel() for p in backbone.parameters())
           
           # Forward pass timing
           backbone.eval()
           with torch.no_grad():
               start_time = time.time()
               for _ in range(100):
                   _ = backbone(x)
               end_time = time.time()
           
           avg_time = (end_time - start_time) / 100 * 1000  # ms
           
           print(f"{name}:")
           print(f"  Parameters: {n_params:,}")
           print(f"  Forward time: {avg_time:.2f}ms")
           print()

   compare_backbones()

Best Practices
-------------

**Start Simple**
  Begin with basic architectures before adding complexity.

**Consistent Interface**
  Always implement the required return types correctly.

**Parameter Efficiency**
  Consider parameter count vs performance trade-offs.

**Gradient Health**
  Monitor for vanishing/exploding gradients during training.

**Memory Awareness**
  Consider GPU memory usage, especially for large models.

**Reproducibility**
  Initialize weights consistently for fair comparisons.

**Documentation**
  Document architecture choices and expected use cases.

Complete Example Script
----------------------

.. code-block:: python

   """
   Complete example: Creating and testing custom neural network backbones
   """

   from mammoth_lite import (register_backbone, MammothBackbone, ReturnTypes,
                            load_runner, train)
   from torch import nn
   from torch.nn import functional as F
   import torch

   # Step 1: Define backbone class
   class MyCustomCNN(MammothBackbone):
       def __init__(self, num_classes, num_channels=32):
           super().__init__()
           
           self.features = nn.Sequential(
               nn.Conv2d(3, num_channels, 3, 1, 1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               nn.Conv2d(num_channels, num_channels*2, 3, 1, 1),
               nn.ReLU(),
               nn.MaxPool2d(2),
           )
           
           self.classifier = nn.Linear(num_channels*2*8*8, num_classes)

       def forward(self, x, returnt=ReturnTypes.OUT):
           feat_maps = self.features(x)
           features = feat_maps.view(feat_maps.size(0), -1)
           logits = self.classifier(features)
           
           if returnt == ReturnTypes.OUT:
               return logits
           elif returnt == ReturnTypes.FEATURES:
               return features
           elif returnt == ReturnTypes.BOTH:
               return logits, features
           else:
               raise ValueError(f"Unknown return type: {returnt}")

   # Step 2: Register backbone variants
   @register_backbone(name='my-cnn-small')
   def my_cnn_small(num_classes):
       return MyCustomCNN(num_classes, num_channels=16)

   @register_backbone(name='my-cnn-large')
   def my_cnn_large(num_classes):
       return MyCustomCNN(num_classes, num_channels=64)

   # Step 3: Test the backbone
   print("Testing custom backbone...")
   
   # Test instantiation
   backbone = my_cnn_small(10)
   x = torch.randn(4, 3, 32, 32)
   
   logits = backbone(x, ReturnTypes.OUT)
   features = backbone(x, ReturnTypes.FEATURES)
   
   print(f"Logits shape: {logits.shape}")
   print(f"Features shape: {features.shape}")
   print(f"Parameters: {sum(p.numel() for p in backbone.parameters()):,}")

   # Step 4: Use in continual learning
   model, dataset = load_runner(
       'sgd', 'seq-cifar10',
       {'lr': 0.1, 'n_epochs': 1, 'backbone': 'my-cnn-small'}
   )

   print(f"\\nUsing backbone: {type(model.net).__name__}")
   train(model, dataset)

Common Issues and Solutions
--------------------------

**Shape Mismatch Errors**
  Calculate feature dimensions carefully after convolutions and pooling.

**Memory Issues**
  Reduce model size or batch size. Monitor GPU memory usage.

**Gradient Problems**
  Add skip connections or adjust learning rates for deep networks.

**Registration Errors**
  Ensure the registration function signature is correct (only ``num_classes`` parameter).

**Return Type Errors**
  Implement all required return types and handle edge cases properly.

**Performance Issues**
  Profile your backbone and optimize bottlenecks. Consider more efficient operations.

Next Steps
----------

Now that you can create custom backbones:

1. **Experiment with Architectures**: Try different CNN, ResNet, or Transformer designs
2. **Optimize for Continual Learning**: Design architectures that work well with specific algorithms
3. **Advanced Techniques**: Explore neural architecture search or adaptive architectures
4. **Benchmark Performance**: Compare your backbones across different datasets and algorithms
5. **Share Your Work**: Contribute useful architectures to the research community

Custom backbones give you complete control over the neural network architecture, enabling you to test how different designs affect continual learning performance!
