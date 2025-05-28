Creating Custom Models
======================

This example shows how to create custom continual learning algorithms in Mammoth Lite. You'll learn to implement your own model, register it with the framework, and use it in experiments.

.. note::

    This example is based on the Jupyter notebook ``examples/notebooks/create_a_model.ipynb``. You can run it interactively or follow along here.

Learning Objectives
-------------------

By the end of this example, you'll understand:

* The structure of continual learning models in Mammoth Lite  
* How to inherit from the ``ContinualModel`` base class  
* How to implement the ``observe`` method for your algorithm  
* How to register custom models with the framework  
* How to test and use your custom model  

Understanding Continual Learning Models
---------------------------------------

Model Structure
~~~~~~~~~~~~~~~

All continual learning models in Mammoth Lite share a common structure:

* **Inheritance**: Inherit from ``ContinualModel`` base class  
* **Registration**: Use the ``@register_model`` decorator  
* **Core Method**: Implement the ``observe`` method  
* **Compatibility**: Optionally specify supported scenarios  

The ``observe`` method is the heart of any continual learning algorithm - it defines how the model processes each batch of training data.

Base Class Features
~~~~~~~~~~~~~~~~~~~

The ``ContinualModel`` class provides:

* ``self.net``: The backbone neural network (e.g., ResNet-18)  
* ``self.opt``: The optimizer (e.g., SGD, Adam)    
* ``self.loss``: The loss function (usually CrossEntropyLoss)  
* ``self.device``: The device (CPU or GPU)  
* ``self.args``: All configuration arguments  

Creating a Simple Custom Model
------------------------------

Let's create a custom SGD-based model to understand the process:

.. code-block:: python

   from mammoth_lite import register_model, ContinualModel

   @register_model('new-sgd')  # Register with name 'new-sgd'
   class NewSgd(ContinualModel):
       """
       A custom SGD-based continual learning model.
       
       This model implements standard SGD training without any
       specific continual learning techniques. It will exhibit
       catastrophic forgetting but serves as a good baseline.
       """
       
       # Specify compatible scenarios (optional)
       COMPATIBILITY = ['class-il', 'task-il']
       
       def observe(self, inputs, labels, not_aug_inputs, epoch=None):
           """
           Process a batch of training data.
           
           Args:
               inputs: Augmented input images (tensor)
               labels: Target labels (tensor)  
               not_aug_inputs: Non-augmented images (tensor)
               epoch: Current epoch number (optional)
               
           Returns:
               Loss value (for logging/monitoring)
           """
           # Set model to training mode
           self.net.train()
           
           # Zero gradients from previous batch
           self.opt.zero_grad()
           
           # Forward pass through the network
           outputs = self.net(inputs)
           
           # Compute loss
           loss = self.loss(outputs, labels)
           
           # Backward pass
           loss.backward()
           
           # Update weights
           self.opt.step()
           
           # Return loss for monitoring
           return loss.item()

Key Components Explained
~~~~~~~~~~~~~~~~~~~~~~~~

**@register_model Decorator**
  Registers your model with Mammoth Lite so it can be used with ``load_runner()``.

**COMPATIBILITY Attribute**
  Specifies which continual learning scenarios your model supports:
  
  * ``'class-il'``: Class-incremental learning. This is also the default and most common scenario.  
  * ``'task-il'``: Task-incremental learning    
  * More scenarios are available in the full Mammoth framework.  

**observe Method Arguments**
  
  * ``inputs``: Augmented training images (data augmentation applied)  
  * ``labels``: Ground truth class labels  
  * ``not_aug_inputs``: Original images without augmentation (useful for replay-based algorithms)  
  * ``epoch``: (optional) Current epoch number (useful for scheduling)  

Testing Your Custom Model
-------------------------

Once defined, you can use your custom model like any built-in model:

.. code-block:: python

   from mammoth_lite import load_runner, train

   # Load your custom model
   model, dataset = load_runner(
       model='new-sgd',          # Use your custom model name
       dataset='seq-cifar10',
       args={
           'lr': 0.1,
           'n_epochs': 2,
           'batch_size': 32
       }
   )

   # Train and evaluate
   train(model, dataset)

**Expected Output:**

.. code-block:: text

   Task 1: 100%|██████████| 1563/1563 [01:20<00:00, 19.42it/s]
   Accuracy on task 1:	[Class-IL]: 68.20% 	[Task-IL]: 68.20%
   
   Task 2: 100%|██████████| 1563/1563 [01:18<00:00, 19.95it/s]  
   Accuracy on task 2:	[Class-IL]: 32.90% 	[Task-IL]: 62.62%

   ...

Advanced Custom Model Example
-----------------------------

Let's create a more sophisticated model that uses experience replay to mitigate forgetting:

.. code-block:: python

    from argparse import ArgumentParser
    from mammoth_lite import register_model, ContinualModel, Buffer, add_rehearsal_args

    @register_model('experience-replay')
    class SimpleReplay(ContinualModel):
        """
        A simple experience replay model.
        
        Stores a small buffer of previous examples and replays
        them when learning new tasks to reduce forgetting.
        """

        @staticmethod
        def get_parser(parser: ArgumentParser):
            """
            This method is used to define additional command line arguments for the model.
            It is called by the `load_runner` function to parse the arguments.
            """

            add_rehearsal_args(parser)  # This includes the `buffer_size` and `minibatch_size` arguments
            parser.add_argument('--alpha', type=float, default=0.5,
                                help='Weight of replay loss in total loss')
            return parser
        
        def __init__(self, backbone, loss, args, device, dataset):
            super().__init__(backbone, loss, args, device, dataset)
            
            # Initialize experience buffer
            self.buffer = Buffer(
                buffer_size=args.buffer_size  # Custom buffer size
            )
            
        def observe(self, inputs, labels, not_aug_inputs, epoch=None):
            """
            Training step with experience replay.
            """
            self.net.train()
            
            # Standard training on current batch
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            
            # Sample a batch from the buffer
            if len(self.buffer) > 0:
                buffer_inputs, buffer_labels = self.buffer.get_data(
                    size=self.args.minibatch_size, device=self.device)
                
                # Forward pass on the buffer data
                buffer_outputs = self.net(buffer_inputs)
                # Compute the loss on the buffer data
                buffer_loss = self.loss(buffer_outputs, buffer_labels)
                # Combine the losses from the current batch and the buffer
                loss = loss + self.args.alpha * buffer_loss

            # backward pass and update the weights
            loss.backward()
            self.opt.step()
            
            # Store the current batch in the buffer
            self.buffer.add_data(inputs, labels)
            
            return total_loss.item()

Using the Advanced Model
~~~~~~~~~~~~~~~~~~~~~~~~

Your replay model can now be used with additional parameters:

.. code-block:: python

   # Load replay model with custom parameters
   model, dataset = load_runner(
       model='simple-replay',
       dataset='seq-cifar10',
       args={
           'lr': 0.1,
           'n_epochs': 2,
           'batch_size': 32,
           'buffer_size': 1000,      # Custom parameter
           'alpha': 0.5       # Custom parameter
           'minibatch_size': 32  # Size of replay batch
       }
   )

   train(model, dataset)


Adding Custom Arguments
~~~~~~~~~~~~~~~~~~~~~~~

You can add custom arguments for your model:

.. code-block:: python

   from mammoth_lite.utils.args import add_rehearsal_args

   @register_model('my-model')
   class MyModel(ContinualModel):
       
       def get_parser(parser):
           """
           Add custom arguments for this model.
           """
           parser.add_argument('--my_custom_arg', type=int, default=42,
                               help='An example custom argument')
           return parser

Next Steps
----------

Now that you can create custom models:

1. **Implement Advanced Algorithms**: Try implementing additional rehearsal methods, such as Dark Experience Replay, or regularization methods such as Learning without Forgetting. You can find their complete code in the `models/` directory.
2. **Create Custom Datasets**: Learn to build custom benchmarks in :doc:`custom_dataset`
3. **Design Custom Backbones**: Explore custom architectures in :doc:`custom_backbone`  
4. **Contribute**: Share your models with the Mammoth Lite community

The ability to create custom models opens up endless possibilities for continual learning research. Experiment with different approaches and see how they perform on various benchmarks!
