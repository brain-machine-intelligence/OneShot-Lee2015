# Causal Learning Model Implementation

This repository provides a Python implementation of the one-shot learning model introduced in the paper *Neural Computations Mediating One-Shot Learning in the Human Brain*.

## Project Structure

The project contains the following Python files:

### `Oneshot.py`

Defines the `Oneshot` class, which includes the one-shot model as a core component. The `gen_exp` function creates a one-shot experiment and outputs the estimated learning rate derived from the one-shot model.

### `Oneshot_sangwan.py`

Implements the core one-shot model based on the referenced paper. This code is utilized within `Oneshot.py` to instantiate the one-shot model.

### `gen_exp.py`

The main script that instantiates the `Oneshot` class from `Oneshot.py` and demonstrates usage examples.

## How to Run the Project

1. Clone the repository:

    ```bash
    git clone https://github.com/ckdghk77/oneshot_sangwan_python.git
    ```

2. Install the required packages:

    ```bash
    pip install numpy
    ```

3. Run the main script:

    ```bash
    python gen_exp.py
    ```

   This will display examples of both one-shot and incremental experiments.
