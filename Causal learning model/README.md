# One-Shot Learning Model Implementation

This repository contains a Python implementation of the one-shot learning model proposed in the paper "Neural Computations Mediating One-Shot Learning in the Human Brain."

## Project structure
The project consists of the following Python files:

### Oneshot.py:

Contains the Oneshot class, which has a one-shot model as a property.
The gen_exp function creates a one-shot experiment and outputs the estimated learning rate from the one-shot model.

### Oneshot_sangwan.py:

Implements the one-shot model from the referenced paper.
This code is utilized by Oneshot.py to create the one-shot model.

### gen_exp.py:

The main file that instantiates the Oneshot class from Oneshot.py and demonstrates examples of its usage.


## How to Run the Project
1. Clone the repository:
```
git clone https://github.com/ckdghk77/oneshot_sangwan_python.git
```

2. Install the required package:
```
pip install numpy
```

3. Run the main script:
```
python gen_exp.py
```

This will show you examples of oneshot and incremental experiment.
