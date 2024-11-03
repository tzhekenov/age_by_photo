# Age classification from face image
Classification of age from face image project

## Project Setup

To set up the project, follow these steps:
1. **Install Poetry**:
    If you don't have Poetry installed, you can install it by following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

    For example, using `curl`:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Or using `pip`:
    ```sh
    pip install poetry
    ```

2. **Clone the repository**:
    ```sh
    git clone https://github.com/tzhekenov/age_by_photo.git
    cd age_by_photo
    ```

3. **Initialize the Poetry environment**:
    ```sh
    poetry install
    ```

4. **Activate the Poetry shell**:
    ```sh
    poetry shell
    ```

This will activate the virtual environment created by Poetry.

5. **Add the environment to Jupyter kernel**:
    ```sh
    python -m ipykernel install --user --name age-by-photo-py3.11
    ```

This will install all the dependencies and set up the Jupyter kernel for the project.