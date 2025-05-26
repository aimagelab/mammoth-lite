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
  
  * ``'class-il'``: Class-incremental learning
  * ``'task-il'``: Task-incremental learning  
  * ``'domain-il'``: Domain-incremental learning

**observe Method Arguments**
  
  * ``inputs``: Augmented training images (data augmentation applied)
  * ``labels``: Ground truth class labels
  * ``not_aug_inputs``: Original images without augmentation (useful for some algorithms)
  * ``epoch``: Current epoch number (useful for scheduling)

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
   Accuracy on task 1: 67.8%
   
   Task 2: 100%|██████████| 1563/1563 [01:18<00:00, 19.95it/s]  
   Accuracy on task 1: 24.6%  # Catastrophic forgetting
   Accuracy on task 2: 69.1%

Advanced Custom Model Example
-----------------------------

Let's create a more sophisticated model that uses experience replay to mitigate forgetting:

.. code-block:: python

   import torch
   from mammoth_lite import register_model, ContinualModel

   @register_model('simple-replay')
   class SimpleReplay(ContinualModel):
       """
       A simple experience replay model.
       
       Stores a small buffer of previous examples and replays
       them when learning new tasks to reduce forgetting.
       """
       
       def __init__(self, backbone, loss, args, device):
           super().__init__(backbone, loss, args, device)
           
           # Initialize experience buffer
           self.buffer_size = getattr(args, 'buffer_size', 500)
           self.buffer_inputs = []
           self.buffer_labels = []
           
           # Replay parameters
           self.replay_ratio = getattr(args, 'replay_ratio', 0.5)
           
       def observe(self, inputs, labels, not_aug_inputs, epoch=None):
           """
           Training step with experience replay.
           """
           self.net.train()
           
           # Standard training on current batch
           self.opt.zero_grad()
           outputs = self.net(inputs)
           loss = self.loss(outputs, labels)
           
           # Add replay loss if buffer has examples
           if len(self.buffer_inputs) > 0:
               replay_loss = self._get_replay_loss()
               total_loss = loss + self.replay_ratio * replay_loss
           else:
               total_loss = loss
           
           total_loss.backward()
           self.opt.step()
           
           # Update buffer with current examples
           self._update_buffer(not_aug_inputs, labels)
           
           return total_loss.item()
       
       def _get_replay_loss(self):
           """Compute loss on replayed examples."""
           # Sample from buffer
           n_replay = min(len(self.buffer_inputs), 32)
           indices = torch.randperm(len(self.buffer_inputs))[:n_replay]
           
           replay_inputs = torch.stack([self.buffer_inputs[i] for i in indices])
           replay_labels = torch.stack([self.buffer_labels[i] for i in indices])
           
           replay_inputs = replay_inputs.to(self.device)
           replay_labels = replay_labels.to(self.device)
           
           # Forward pass on replay data
           replay_outputs = self.net(replay_inputs)
           replay_loss = self.loss(replay_outputs, replay_labels)
           
           return replay_loss
       
       def _update_buffer(self, inputs, labels):
           """Add new examples to buffer, removing oldest if necessary."""
           batch_size = inputs.size(0)
           
           for i in range(batch_size):
               # Add to buffer
               self.buffer_inputs.append(inputs[i].cpu())
               self.buffer_labels.append(labels[i].cpu())
               
               # Remove oldest if buffer is full
               if len(self.buffer_inputs) > self.buffer_size:
                   self.buffer_inputs.pop(0)
                   self.buffer_labels.pop(0)

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
           'replay_ratio': 0.5       # Custom parameter  
       }
   )

   train(model, dataset)

Model Implementation Patterns
-----------------------------

Common Patterns
~~~~~~~~~~~~~~~

**Memory-based Methods**

.. code-block:: python

   def __init__(self, backbone, loss, args, device):
       super().__init__(backbone, loss, args, device)
       self.memory_buffer = []  # Store previous examples
       
   def observe(self, inputs, labels, not_aug_inputs, epoch=None):
       # Train on current data + replay from memory
       pass

**Regularization-based Methods**

.. code-block:: python

   def __init__(self, backbone, loss, args, device):
       super().__init__(backbone, loss, args, device)
       self.previous_weights = None  # Store important weights
       
   def observe(self, inputs, labels, not_aug_inputs, epoch=None):
       # Add regularization term to prevent weight changes
       main_loss = self.loss(self.net(inputs), labels)
       reg_loss = self._compute_regularization()
       total_loss = main_loss + reg_loss
       pass

**Meta-learning Methods**

.. code-block:: python

   def observe(self, inputs, labels, not_aug_inputs, epoch=None):
       # Implement MAML-style meta-learning updates
       # Fast adaptation on current task
       # Meta-update to preserve previous knowledge
       pass

Adding Custom Arguments
~~~~~~~~~~~~~~~~~~~~~~~

You can add custom arguments for your model:

.. code-block:: python

   from mammoth_lite.utils.args import add_rehearsal_args

   @register_model('my-model')
   class MyModel(ContinualModel):
       def __init__(self, backbone, loss, args, device):
           super().__init__(backbone, loss, args, device)
           
           # Access custom arguments
           self.custom_param = getattr(args, 'custom_param', 1.0)
           self.another_param = getattr(args, 'another_param', 'default')

Testing and Validation
----------------------

Unit Testing Your Model
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_custom_model():
       """Test that custom model can be loaded and trained."""
       from mammoth_lite import load_runner, train
       
       # Test loading
       model, dataset = load_runner(
           'new-sgd', 
           'seq-cifar10',
           {'n_epochs': 1}
       )
       
       assert model is not None
       assert dataset is not None
       
       # Test that training doesn't crash
       try:
           train(model, dataset)
           print("✓ Model works correctly")
       except Exception as e:
           print(f"✗ Model failed: {e}")

   test_custom_model()

Comparing Models
~~~~~~~~~~~~~~~~

Compare your custom model against baselines:

.. code-block:: python

   def compare_models():
       """Compare custom model against SGD baseline."""
       
       results = {}
       
       for model_name in ['sgd', 'new-sgd', 'simple-replay']:
           print(f"\\nTesting {model_name}...")
           model, dataset = load_runner(
               model_name, 
               'seq-cifar10',
               {'n_epochs': 1}
           )
           # Run and collect results
           # results[model_name] = evaluate(model, dataset)
       
       # Compare results
       # print("Model comparison:", results)

   compare_models()

Best Practices
--------------

**Error Handling**

.. code-block:: python

   def observe(self, inputs, labels, not_aug_inputs, epoch=None):
       try:
           # Your training logic
           pass
       except RuntimeError as e:
           print(f"Training error: {e}")
           return float('inf')  # Return high loss on error

**Memory Management**

.. code-block:: python

   def observe(self, inputs, labels, not_aug_inputs, epoch=None):
       # Clear unnecessary gradients
       self.opt.zero_grad()
       
       # Your training logic
       
       # Clean up if needed
       torch.cuda.empty_cache()  # For GPU memory

**Logging and Monitoring**

.. code-block:: python

   def observe(self, inputs, labels, not_aug_inputs, epoch=None):
       # Your training logic
       loss_value = loss.item()
       
       # Optional: Log additional metrics
       if hasattr(self, 'task_losses'):
           self.task_losses.append(loss_value)
       
       return loss_value

Complete Example Script
-----------------------

.. code-block:: python

   """
   Complete example: Creating and testing a custom continual learning model
   """

   from mammoth_lite import register_model, ContinualModel, load_runner, train
   import torch

   @register_model('my-custom-sgd')
   class MyCustomSgd(ContinualModel):
       """Custom SGD model with additional logging."""
       
       def __init__(self, backbone, loss, args, device):
           super().__init__(backbone, loss, args, device)
           self.training_losses = []
           
       def observe(self, inputs, labels, not_aug_inputs, epoch=None):
           self.net.train()
           self.opt.zero_grad()
           
           outputs = self.net(inputs)
           loss = self.loss(outputs, labels)
           
           loss.backward()
           self.opt.step()
           
           loss_value = loss.item()
           self.training_losses.append(loss_value)
           
           return loss_value

   # Test the custom model
   print("Testing custom model...")
   model, dataset = load_runner(
       'my-custom-sgd',
       'seq-cifar10', 
       {'n_epochs': 1, 'lr': 0.1}
   )

   train(model, dataset)
   print(f"Average training loss: {sum(model.training_losses) / len(model.training_losses):.4f}")

Common Issues and Solutions
---------------------------

**Model Not Found Error**
  Make sure you've run the cell with ``@register_model`` before trying to use it.

**GPU Memory Issues**  
  Add ``torch.cuda.empty_cache()`` in your observe method.

**Slow Training**
  Check that you're not accidentally keeping gradients or large tensors in memory.

**Gradient Explosion**
  Add gradient clipping: ``torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)``

Next Steps
----------

Now that you can create custom models:

1. **Implement Advanced Algorithms**: Try implementing rehearsal, regularization, or meta-learning methods
2. **Create Custom Datasets**: Learn to build custom benchmarks in :doc:`custom_dataset`
3. **Design Custom Backbones**: Explore custom architectures in :doc:`custom_backbone`  
4. **Advanced Evaluation**: Set up comprehensive evaluation and analysis
5. **Contribute**: Share your models with the Mammoth Lite community

The ability to create custom models opens up endless possibilities for continual learning research. Experiment with different approaches and see how they perform on various benchmarks!
