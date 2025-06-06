package ticTacToe;

import java.util.List;
import java.util.Random;
import java.util.Set;
/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	 
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=10000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.2;
	
//	double minEpsilon=0.1;
//	
//	double EpsilonDecay=0.9999;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount) 
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initiating q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent() 
	{
		this(new RandomAgent(), 0.1, 50000, 0.9);
		
	}
	 
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method. 
	 */
	
	public void train()
	{
		// This method implements the training loop for a reinforcement learning agent using the Q-learning algorithm.
		
		/* 
		 * YOUR CODE HERE
		 */
		
		Random random = new Random(); // Create a Random object for generating random numbers used in the epsilon-greedy policy.
		
		// Iterate over the specified number of episodes (training cycles).
		for (int i = 0; i < numEpisodes; i++) {
			// Reset the environment to its initial state at the start of each episode.
			env.reset();
			// Get the current game state from the environment, which serves as the starting state for this episode.
			Game key = env.getCurrentGameState();
			
			// Continue the loop until the environment reaches a terminal state (e.g., game over or win/loss condition).
			while (!env.isTerminal()) {
				Move action = null; // Variable to store the selected action.
				// Get the list of all possible moves from the current state.
				List<Move> moves = env.getPossibleMoves();
				
				// Implement the epsilon-greedy policy for action selection.
				if (epsilon > random.nextDouble()) {
					// With probability epsilon, choose a random action (exploration).
					int randompos = env.getPossibleMoves().size(); // Number of available moves.
					int rand = random.nextInt(randompos); // Generate a random index within the range of available moves.
					action = moves.get(rand); // Selects a random move.
				} else {
					// With probability 1 - epsilon, choose the action with the highest Q-value (exploitation).
                    double max = Double.NEGATIVE_INFINITY; // Initialise the maximum Q-value as negative infinity.
                    
                    // Iterate through all possible moves to find the one with the highest Q-value.
                    for (Move move : moves) {
                        double value = qTable.getQValue(key, move); // Get the Q-value for the current state-action pair.
                        if (value > max) {
                            max = value; // Update the maximum Q-value.
                            action = move; // Update the action corresponding to the maximum Q-value.
                        }
                    }
				}

				try {
					// Execute the selected action in the environment and observe the outcome.
					Outcome outcome = env.executeMove(action);
					// Calculate the maximum Q-value for the next state (s').
					double maxQ = outcome.sPrime.isTerminal() ? 0 // If the next state is terminal, the max Q-value is 0.
							: outcome.sPrime.getPossibleMoves().stream()
									.mapToDouble(move -> qTable.getQValue(outcome.sPrime, move)).max() // Get Q-values for all possible moves in s'.
									.orElse(Double.NEGATIVE_INFINITY); // Handle the case where no moves are possible.
					// Compute the Q-learning update for the state-action pair.
					double sample = (outcome.localReward + (discount * maxQ)); // Bellman equation for Q-learning.
					double rAverage = (((1 - alpha) * qTable.getQValue(outcome.s, outcome.move)) + (alpha * sample)); // Bellman equation for Q-learning.
					// Update the Q-value table with the new value.
					qTable.addQValue(outcome.s, outcome.move, rAverage);
					// Update the current state to the next state (s').
					key = outcome.sPrime;
				} catch (IllegalMoveException e) {
					// Handle exceptions for illegal moves, which could occur due to environment constraints or logic errors.
					e.printStackTrace(); // Print the stack trace to debug the issue.
				}
			}
		}


		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		// This method generates a deterministic policy based on the Q-table. 
	    // The policy specifies the best action for each state to maximise the expected reward.
		
		/* 
		 * YOUR CODE HERE 
		 */
 
		Policy p = new Policy(); // Create a new Policy object to store the extracted policy.
		// Get all the states currently stored in the Q-table.
	    // Each state in the Q-table has an associated set of Q-values for its possible actions.
		Set<Game> states = qTable.keySet();
		
		// Iterate through all states in the Q-table to determine the best action for each state.
		for (Game key : states) {
			// Check if the current state is a terminal state.
			if (key.isTerminal()) {
				// If the state is terminal, no action is needed; set the policy for this state to null.
				p.policy.put(key, null);
			} else {
				// For non-terminal states, find the action with the highest Q-value.
				Move action = null; // Variable to store the best action for the current state.
				double max = Double.NEGATIVE_INFINITY; // Initialise the maximum Q-value as negative infinity.
				// Get the list of all possible moves (actions) for the current state.
				List<Move> moves = key.getPossibleMoves();
				// Iterate through all possible moves to determine the one with the highest Q-value.
				for (Move move : moves) {
					// Retrieve the Q-value for the current state-action pair.
					double value = qTable.getQValue(key, move);
					// Check if the current Q-value is greater than the maximum Q-value seen so far.
					if (value > max) {
						// Update the maximum Q-value and the corresponding action.
						max = value;
						action = move;
					}
				}
				// After evaluating all possible moves, set the policy for the current state
	            // to the action with the highest Q-value.
				p.policy.put(key, action);
			}
		}
		// Return the extracted policy. The policy maps each state to the best action determined by the Q-table.
		return p;

	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
	    // Test method to play your agent against a human agent (yourself).
	    QLearningAgent agent = new QLearningAgent();  // Initialise the QLearning agent.
	    
	    HumanAgent human = new HumanAgent();  // Initialise the HumanAgent to play against.

	    // Initialise the game with the QLearningAgent as 'X' and the HumanAgent as 'O'
	    Game game = new Game(agent, human, human);

	    // Play out the game, alternating moves between the QLearningAgent and HumanAgent
	    game.playOut(); 
	}

	
	
	
}
