# gym-bandits
The [Multi-Armed Bandits Domain](https://github.com/ruirmv/gym-bandits) is a single-agent stateless environment with k actions.
The implementation here provided follows the statement as in [Richard S. Sutton and Andrew G. Barto](https://mitpress.mit.edu/books/reinforcement-learning-second-edition).

### Installation
```bash
git clone https://github.com/ruirmv/gym-bandits.git
cd gym-bandits
pip install -e .
```

### Creating Instance of Environment

After installation, you can create an instance of the environment with ```gym.make('gym_bandits:bandits-v0')```
