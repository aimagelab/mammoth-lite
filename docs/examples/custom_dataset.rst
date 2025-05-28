Creating Custom Datasets
========================

This example shows how to create custom continual learning datasets in Mammoth Lite. You'll learn to implement your own benchmarks, register them with the framework, and use them in experiments.

.. note::

    This example is based on the Jupyter notebook ``examples/notebooks/create_a_dataset.ipynb``. You can run it interactively or follow along here.

Learning Objectives
-------------------

By the end of this example, you'll understand:

* The two-layer structure of datasets in Mammoth Lite  
* How to create a data source class inheriting from ``MammothDataset``  
* How to create a continual dataset class inheriting from ``ContinualDataset``  
* How to register custom datasets with the framework  
* How to configure data transformations and task structures  

Understanding Dataset Architecture
----------------------------------

Mammoth Lite uses a two-layer architecture for datasets:

**Layer 1: Data Source (MammothDataset)**
  Handles loading and accessing individual data samples. Similar to PyTorch's ``Dataset`` class but with continual learning requirements.

**Layer 2: Continual Dataset (ContinualDataset)**  
  Defines the continual learning scenario: how data is split into tasks, transformations, and evaluation protocol.

This separation allows you to:

* Reuse the same data source for different continual learning scenarios  
* Easily modify task structures without changing data loading logic  
* Support different evaluation protocols on the same underlying data  

Data Source Requirements
------------------------

The data source class must provide:

**Required Attributes:**
* ``data``: Training/testing data samples  
* ``targets``: Corresponding labels  
* ``not_aug_transform``: Transformation without data augmentation  

**Required Methods:**
* ``__len__()``: Return number of samples  
* ``__getitem__(index)``: Return ``(augmented_image, label, original_image)``  

The triple return format is crucial for continual learning algorithms that need both augmented data for training and original data for memory storage (rehearsal).

Creating a Custom Data Source
-----------------------------

Let's create a custom CIFAR-10 data source that meets Mammoth Lite's requirements:

.. code-block:: python

   from mammoth_lite import MammothDataset
   from torchvision.datasets import CIFAR10
   from torchvision import transforms
   from PIL import Image

   class MammothCIFAR10(MammothDataset, CIFAR10):
       """
       Custom CIFAR-10 dataset that returns the required triple format.
       
       Inherits from both MammothDataset (for continual learning features)
       and CIFAR10 (for data loading functionality).
       """

       def __init__(self, root, is_train=True, transform=None):
           """
           Initialize the dataset.
           
           Args:
               root: Directory to store/load data
               is_train: Whether to load training or test set
               transform: Data augmentation transforms
           """
           self.root = root
           # Download data if not present (avoid debug messages)
           super().__init__(root, is_train, transform, 
                          download=not self._check_integrity())

       def __getitem__(self, index):
           """
           Get a sample from the dataset.
           
           Returns:
               tuple: (augmented_image, label, original_image)
           """
           # Get raw data and label
           img, target = self.data[index], self.targets[index]

           # Convert numpy array to PIL Image for transformations
           img = Image.fromarray(img, mode='RGB')
           original_img = img.copy()  # Important: copy to avoid modification

           # Apply non-augmenting transform to original image
           not_aug_img = self.not_aug_transform(original_img)

           # Apply augmenting transforms if specified
           if self.transform is not None:
               img = self.transform(img)

           return img, target, not_aug_img

Key Implementation Details
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Multiple Inheritance**
  Inheriting from both ``MammothDataset`` and ``CIFAR10`` gives you continual learning features and CIFAR-10 data loading.

**Image Copying**
  ``original_img = img.copy()`` is crucial - without this, transforms would modify the original image.

**Transform Application**
  Apply ``not_aug_transform`` to get normalized but non-augmented images for memory storage.

**Triple Return Format**
  Always return ``(augmented_image, label, original_image)`` from ``__getitem__``.

Creating a Custom Continual Dataset
-----------------------------------

Now create the continual learning scenario definition:

.. code-block:: python

   from mammoth_lite import register_dataset, ContinualDataset, base_path
   from torchvision import transforms

   @register_dataset(name='custom-cifar10')
   class CustomSeqCifar10(ContinualDataset):
       """
       Custom Sequential CIFAR-10 continual learning benchmark.
       
       Splits CIFAR-10 into 5 tasks of 2 classes each.
       """

       # Required dataset metadata
       NAME = 'custom-cifar10'
       SETTING = 'class-il'          # Class-incremental learning
       N_CLASSES_PER_TASK = 2        # 2 classes per task
       N_TASKS = 5                   # 5 total tasks
       
       # Dataset statistics for normalization
       MEAN = (0.4914, 0.4822, 0.4465)
       STD = (0.2470, 0.2435, 0.2615)

       # Training transformations (with data augmentation)
       TRANSFORM = transforms.Compose([
           transforms.RandomCrop(32, padding=4),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize(MEAN, STD)
       ])

       # Test transformations (no augmentation)
       TEST_TRANSFORM = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize(MEAN, STD)
       ])

       def get_data_loaders(self):
           """
           Create and return train and test datasets.
           
           Returns:
               tuple: (train_dataset, test_dataset)
           """
           # Create datasets using our custom data source
           train_dataset = MammothCIFAR10(
               root=base_path() + 'CIFAR10',
               is_train=True,
               transform=self.TRANSFORM
           )
           
           test_dataset = MammothCIFAR10(
               root=base_path() + 'CIFAR10', 
               is_train=False,
               transform=self.TEST_TRANSFORM
           )

           return train_dataset, test_dataset

       @staticmethod
       def get_backbone():
           """
           Specify the default backbone architecture.
           
           Returns:
               str: Name of registered backbone
           """
           return "resnet18"

Required Attributes Explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NAME**: Unique identifier for your dataset (by default, will use the same name as in the ``@register_dataset`` decorator)

**SETTING**: Continual learning scenario type:  
  * ``'class-il'``: Class-incremental learning  
  * ``'task-il'``: Task-incremental learning    
  * ... other settings are available in the full Mammoth framework or can be defined as needed  

**N_CLASSES_PER_TASK**: Number of classes introduced in each task

**N_TASKS**: Total number of sequential tasks

**MEAN/STD**: Dataset statistics for proper normalization

**TRANSFORM/TEST_TRANSFORM**: Data augmentation for training/testing

Required Methods
~~~~~~~~~~~~~~~~

**get_data_loaders()**: Returns train and test dataset instances

**get_backbone()**: Specifies default neural network architecture

Testing Your Custom Dataset
---------------------------

Once defined, use your custom dataset like any built-in benchmark:

.. code-block:: python

   from mammoth_lite import load_runner, train

   # Load SGD model with your custom dataset
   model, dataset = load_runner(
       model='sgd',
       dataset='custom-cifar10',  # Use your custom dataset
       args={
           'lr': 0.1,
           'n_epochs': 1,
           'batch_size': 32
       }
   )

   # Verify dataset properties
   print(f"Dataset: {dataset.NAME}")
   print(f"Setting: {dataset.SETTING}")
   print(f"Tasks: {dataset.N_TASKS}")
   print(f"Classes per task: {dataset.N_CLASSES_PER_TASK}")

   # Run training
   train(model, dataset)

**Expected Output:**

.. code-block:: text

   Dataset: custom-cifar10
   Setting: class-il
   Tasks: 5
   Classes per task: 2

   Task 1: 100%|██████████| 313/313 [01:18<00:00, 19.95it/s]
   Accuracy for task 1	[Class-IL]: 87.60% 	[Task-IL]: 87.60%

Data Augmentation Strategies
----------------------------

Different Augmentations per Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TaskSpecificTransforms(ContinualDataset):
       """
       Apply different augmentations for different tasks.
       """
       
       def get_transforms(self, task_id):
           """Return task-specific transforms."""
           base_transform = [transforms.ToTensor(), 
                           transforms.Normalize(self.MEAN, self.STD)]
           
           if task_id == 0:
               # Minimal augmentation for first task
               return transforms.Compose([
                   transforms.RandomHorizontalFlip(0.3),
                   *base_transform
               ])
           else:
               # Stronger augmentation for later tasks
               return transforms.Compose([
                   transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                   *base_transform
               ])

Progressive Difficulty
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ProgressiveDifficultyDataset(ContinualDataset):
       """
       Increase task difficulty over time.
       """
       
       def get_data_loaders(self, task_id):
           """Return data loaders with task-specific difficulty."""
           
           # Base dataset
           train_data = self.load_base_data(train=True)
           test_data = self.load_base_data(train=False)
           
           # Apply task-specific modifications
           if task_id > 0:
               # Add noise, blur, or other corruptions
               corruption_level = task_id * 0.1
               train_data = self.add_corruption(train_data, corruption_level)
               test_data = self.add_corruption(test_data, corruption_level)
           
           return train_data, test_data

Validation and Testing
----------------------

Dataset Validation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validate_dataset():
       """Test that custom dataset works correctly."""
       
       # Test loading
       model, dataset = load_runner('sgd', 'custom-cifar10', {'n_epochs': 1})
       
       # Validate properties
       assert dataset.N_TASKS == 5
       assert dataset.N_CLASSES_PER_TASK == 2
       assert dataset.SETTING == 'class-il'
       
       # Test data loading
       train_loader, test_loader = dataset.get_data_loaders()
       
       # Check sample format
       sample = next(iter(train_loader))
       assert len(sample) == 3  # (img, label, not_aug_img)
       
       print("✓ Dataset validation passed")

   validate_dataset()

Next Steps
----------

Now that you can create custom datasets:

1. **Explore Different Scenarios**: Try domain-incremental or task-incremental settings
2. **Create Complex Benchmarks**: Combine multiple datasets or add synthetic corruptions
3. **Custom Backbones**: Learn to design architectures in :doc:`custom_backbone`
4. **Share Your Work**: Contribute interesting datasets to the community

Custom datasets enable you to test continual learning algorithms on your specific problem domains and research questions!
