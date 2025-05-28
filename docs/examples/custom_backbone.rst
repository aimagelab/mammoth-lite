Creating Custom Backbones
=========================

This example shows how to create custom neural network backbones in Mammoth Lite. You'll learn to implement your own architectures, register them with the framework, and use them in continual learning experiments.

.. note::

   This example is based on the Jupyter notebook ``examples/notebooks/create_a_backbone.ipynb``. You can run it interactively or follow along here.

Learning Objectives
-------------------

By the end of this example, you'll understand:

* How to create custom neural network architectures for continual learning  
* The ``MammothBackbone`` base class and its requirements  
* How to implement flexible return types for different algorithms  
* How to register multiple backbone variants  
* How to use custom backbones in experiments  

Understanding Backbone Architecture
-----------------------------------

What is a Backbone?
~~~~~~~~~~~~~~~~~~~

In Mammoth Lite, a **backbone** is the neural network architecture that:

* Extracts features from input data  
* Provides the final classification layer  
* Can return different outputs (logits, features, both) depending on algorithm needs  
* Serves as the foundation for continual learning models  

Backbone vs Model Distinction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Backbone**: The neural network architecture (ResNet, CNN, Transformer)

**Model**: The continual learning algorithm (SGD, replay, regularization) that uses the backbone

This separation allows you to:

* Test the same continual learning algorithm with different architectures  
* Use the same backbone with different continual learning approaches  
* Easily swap architectures without changing algorithm logic  

Backbone Requirements
---------------------

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
-----------------------------

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function-based registration allows:

* **Multiple Variants**: Create different versions with different parameters  
* **Dynamic Configuration**: Potentially accept additional CLI arguments (this is ony available in the full Mammoth framework)  
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

    Loading model:  sgd
    - Using CustomCNN as backbone

    Task 1: 100%|██████████| 1563/1563 [01:20<00:00, 19.42it/s]
    Accuracy on task 1:	[Class-IL]: 72.40% 	[Task-IL]: 72.40%

Comparing Backbone Variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Testing and Validation
----------------------

Backbone Testing
~~~~~~~~~~~~~~~~

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

Next Steps
----------

Now that you can create custom backbones:

1. **Experiment with Architectures**: Try different CNN, ResNet, or Transformer designs
2. **Benchmark Performance**: Compare your backbones across different datasets and algorithms
3. **Share Your Work**: Contribute useful architectures to the research community

Custom backbones give you complete control over the neural network architecture, enabling you to test how different designs affect continual learning performance!
