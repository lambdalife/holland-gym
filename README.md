# holland-gym
Evolving Neural Net Agents on [OpenAI's Gym Environments](http://gym.openai.com/) using [Holland](https://github.com/lambdalife/holland).

Each environment tested has its own directory within the root directory of this project. Each of these environment directories will generally contain the following files:

- `[env]_agent.py` — contains the Agent class for the environment
- `evolve.py` — contains all code related to evolution of agents
- `genomes.json` — stores results from the best (or most recent) runs of evolution (storage parameters can be configured in `evolve.py`)
- `display.py` — displays a trial in the environment of the best individual from `genomes.json`

The root-level file `agent.py` contains the `Agent` class from which all specific environment Agents inherit.

## Running Evolution

To evolve new individuals for a particular environment, simply run `[env]/evolve.py`. Note that you may want to configure the stoarge parameters in this file so that the tracked `[env]/genomes.json` is not overwritten.

## Displaying Individuals

To run a trial (with rendering on) of the most fit individual stored in `[env]/genomes.json`, run `[env]/display.py`.