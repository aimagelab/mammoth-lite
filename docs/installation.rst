Installation
============

System Requirements
-------------------

Mammoth Lite requires:

* **Python**: 3.10 or higher  
* **Operating System**: Linux, macOS, or Windows  
* **GPU**: Optional but recommended for faster training

Dependencies
------------

Mammoth Lite has minimal dependencies to keep the framework lightweight:

**Core Dependencies:**

* ``torch`` - PyTorch for deep learning  
* ``torchvision`` - Computer vision utilities  
* ``numpy`` - Numerical computing  
* ``tqdm`` - Progress bars  
* ``gitpython`` - Git repository handling  
* ``six`` - Python 2/3 compatibility  

**Development Dependencies:**

* ``pytest`` - Testing framework  
* ``ipywidgets`` - Jupyter notebook widgets  
* ``mypy`` - Type checking  

Installation Methods
--------------------

Method 1: Development Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development or if you want to modify the code:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/aimagelab/mammoth-lite.git
   cd mammoth-lite

   # Install in development mode
   pip install -e .

This installs Mammoth Lite in "editable" mode, so changes to the source code are immediately available.

Method 2: Direct Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you just want to use Mammoth Lite without modifying it:

.. code-block:: bash

   pip install mammoth-lite

Method 3: From Source
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Download and extract the source
   wget https://github.com/aimagelab/mammoth-lite/archive/main.zip
   unzip main.zip
   cd mammoth-lite-main

   # Install
   pip install .

Using with UV (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're using the `uv <https://docs.astral.sh/uv/>`_ package manager:

.. code-block:: bash

   # Install dependencies
   uv sync

   # Run with uv
   uv run python mammoth_lite/main.py --model sgd --dataset seq-cifar10

GPU Support
-----------

Mammoth Lite automatically uses GPU acceleration if available. To install PyTorch with CUDA support:

.. code-block:: bash

   # For CUDA 11.8 (adjust version as needed)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Verification
------------

To verify your installation works correctly:

.. code-block:: python

   # Test basic import
   import mammoth_lite
   
   # Check available models and datasets
   from mammoth_lite import get_model_names, get_dataset_names
   
   print("Available models:", get_model_names())
   print("Available datasets:", get_dataset_names())

You should see output listing the available models and datasets.

Running a Quick Test
--------------------

Test that everything works with a quick training run:

.. code-block:: bash

    cd mammoth_lite
    python main.py --model sgd --dataset seq-cifar10 --n_epochs 1

This should start training an SGD model on Sequential CIFAR-10 for 1 epoch.

Next Steps
----------

Once you have Mammoth Lite installed, continue to the :doc:`quickstart` guide to run your first continual learning experiment!
