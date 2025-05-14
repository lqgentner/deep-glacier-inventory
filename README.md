# Deep Glacier Inventory

## Replicating the analysis

Clone this repository to a directory of your choice:

    git clone https://github.com/lqgentner/deep-glacier-inventory.git
    cd deep-glacier-inventory

This repository uses [uv](https://docs.astral.sh/uv/) to manage its environment. To recreate the environment, install uv according to the [guide on the developer's website](https://docs.astral.sh/uv/getting-started/installation/). Afterwards, run the following command in the root directory of this repository:

    uv sync

This will install the required version of Python and all dependencies in a virtual environment located in the `.venv` directory. To run any of the scripts, you can use the following command:

    uv run <script-name>.py

To run Jupyter notebooks, open the repository in an editor of your choice, like Visual Studio Code:

    code .

When prompted, select the virtual environment created earlier (`.venv/bin/python` on macOS and Linux, or `.venv\Scripts\python` on Windows) as the kernel for the notebook.
