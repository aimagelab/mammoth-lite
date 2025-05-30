# 🦣 Mammoth Lite - A *liter* Framework for Continual Learning

Mammoth Lite is a lightweight and simplified version of the [Mammoth](https://github.com/aimagelab/mammoth) framework for continual learning research. This version prioritizes ease of use and minimal dependencies while maintaining the core functionality for developing and testing continual learning algorithms.

Mammoth Lite is designed for didactic purposes and to provide a simple and clean interface for experimenting with continual learning algorithms.  
It should not be used for academic research or development. For that, you should refer to the complete [Mammoth](https://github.com/aimagelab/mammoth).

<p align="center">
  <img width="800" height="125" src="docs/static/mammoth_banner.svg" alt="logo">
</p>

## ⚙️ Setup

- 📥 Install with `pip install -e .` or run directly with `python main.py ...`
- 🚀 Use `main.py` to run experiments
- 🧩 Models can be added to the `mammoth_lite/models` folder
- 📊 Datasets can be added to the `mammoth_lite/datasets` folder
- ⚡ Supports [uv](https://docs.astral.sh/uv/) for fast package management

## 🧪 Examples

### Running a model

To run a basic experiment:

```bash
cd mammoth_lite
python main.py --model sgd --dataset seq-cifar10
```

You can customize various parameters:

```bash
cd mammoth_lite
python main.py --model sgd --dataset seq-cifar10 --lr 0.01 --n_epochs 50
```

### 📓 Interactive Notebooks

For a more interactive learning experience, check out the Jupyter notebooks in the `examples/notebooks/` folder:

- **`basics.ipynb`**: Learn the fundamentals of using Mammoth Lite with step-by-step examples
- **`create_a_model.ipynb`**: Build your own continual learning algorithm from scratch
- **`create_a_backbone.ipynb`**: Design custom neural network architectures
- **`create_a_dataset.ipynb`**: Implement your own continual learning datasets

These notebooks provide hands-on tutorials with detailed explanations and code examples.

To use them, you will need to build Mammoth Lite:

```bash
pip install -e . 
# or
uv build # <- faster, preferred if you have uv
```

### Project Structure

- `mammoth_lite/main.py`: Main entry point for running experiments
- `mammoth_lite/models`: Contains continual learning models
- `mammoth_lite/datasets`: Contains dataset implementations
- `mammoth_lite/backbone`: Neural network architectures
- `mammoth_lite/utils`: Utility functions and training code

## 🔧 Datasets

**NOTE**: Datasets are automatically downloaded in `mammoth/data/`.  
- This can be changed by modifying the `base_path` function in `mammoth/utils/conf.py` or using the `--base_path` argument.
- The `data/` folder should not be tracked by _git_ and is created automatically if missing.

Currently included datasets:
- Sequential CIFAR-10 (`seq-cifar10`): CIFAR-10 split into a sequence of tasks for Class-IL or Task-IL

## 📝 Citing

If you use Mammoth Lite in your research, please cite the original Mammoth papers:

```bibtex
@article{boschini2022class,
  title={Class-Incremental Continual Learning into the eXtended DER-verse},
  author={Boschini, Matteo and Bonicelli, Lorenzo and Buzzega, Pietro and Porrello, Angelo and Calderara, Simone},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}

@inproceedings{buzzega2020dark,
 author = {Buzzega, Pietro and Boschini, Matteo and Porrello, Angelo and Abati, Davide and Calderara, Simone},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Dark Experience for General Continual Learning: a Strong, Simple Baseline},
 volume = {33},
 year = {2020}
}
```
