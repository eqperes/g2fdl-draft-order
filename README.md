# Dependencies 

- Python 3.8
- Pipenv: https://pipenv.pypa.io/en/latest/

# How to use

- Clone the repository
- In the repository folder, install the virtual environment: `pipenv install`
- Modify the constants in the main file `main.py`
- Run the script: `pipenv run python main.py`

# FYI

- This algorithm is based on the [Metropolis-Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
- The principle is to sample random solutions, each solution having a probability proportional to `1 / (cost ** acceptance_power)`
- For 12 players and 9 tiers, it is not difficult to find solutions with cost below 0.05
