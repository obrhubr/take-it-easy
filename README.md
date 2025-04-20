https://github.com/kschweig/take_it_easy

### Solving a game



### Creating the lookup table

For each possible combination of 19 tiles (out of 27), find the optimal end-state.

Create a map from `bit_repr` (27 bits) to `optimal_solution, score`. 

### Using the lookup table

 - Draw a random tile from the stack
	- Construct the bit representation
 - From the lookup table -> Get all matching sets of tiles which are still possible (using & operation on the lookup table)
 - From those, check which are still possible given your current board setup.
 - Remove all impossible boards from the lookup table
 - Select the position to place the tile which maximises score and amount of possible final states
 - Eliminate all other boards from the lookup table

