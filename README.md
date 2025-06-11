# Tetris Heuristic AI

This project is a Tetris bot that uses heuristic evaluation and genetic algorithms to optimize its gameplay strategy. It features a GUI built with Pygame and Pygame GUI.

## Features

* Interactive GUI with sliders, toggles, and real-time graphs
* Genetic algorithm to evolve the best heuristic weights
* Heuristics include: lines cleared, aggregate height, bumpiness, well depth, holes, support

## Installation

```
git clone https://github.com/VeerSahai/TetrisAI.git
cd TetrisAI
pip install -r requirements.txt
```

## Running the Program

```
python main.py
```

## Controls

* Start Game: Start the AI
* Restart Game: Clear and restart
* Heuristics / GA toggle: Show/hide controls
* Run GA: Evolve best weights
* Replay Best: Replay the best genome

## Customization

* Adjust heuristic weights and toggles live
* Set lookahead depth and beam width
* Configure GA population, generations, mutation rate, and max pieces

## Graphs & Visualization

* Heuristic Breakdown
* Score Over Time

### In Progress

* Hold Piece Funtionality (currently only press shift)
