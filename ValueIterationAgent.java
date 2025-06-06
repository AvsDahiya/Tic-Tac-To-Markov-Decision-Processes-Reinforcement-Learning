package ticTacToe;


import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9; 
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=50;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate()
	{
		// This method implements Value Iteration, a core algorithm in reinforcement learning for computing optimal state values.
	    // The goal is to iteratively refine the value function, which estimates the maximum expected reward for each state.
		
		/* 
		 * YOUR CODE HERE 
		 */
		
		// Repeat the value iteration process for `k` iterations.
		for (int i = 0; i < k; i++) {
			// Create a new map to store the updated value function for this iteration.
			Map<Game, Double> newValueFunction = new HashMap<>();
			// Iterate through all states in the current value function.
			for (Game state : valueFunction.keySet()) {
				// Check if the current state is terminal.
				if (state.isTerminal()) {
					// Terminal states have a value of 0 since no future rewards are possible.
					newValueFunction.put(state, 0.0);
					continue; // Skip further computation for terminal states.
				}
				// Initialise the maximum value for this state as negative infinity.
	            // This represents the best expected reward achievable from this state.
				double maxValue = Double.NEGATIVE_INFINITY;
				// Iterate through all possible moves (actions) for the current state.
				for (Move move : state.getPossibleMoves()) {
					// Compute the expected value of the current action by considering all possible outcomes.
					double expectedValue = 0.0;
					// Get the transition probabilities and outcomes for the current state and action.
					for (TransitionProb tp : mdp.generateTransitions(state, move)) {
						// Add the weighted contribution of this outcome to the expected value.
	                    // The weight is the probability of the transition, and the contribution includes:
	                    // - The immediate reward (`tp.outcome.localReward`).
	                    // - The discounted value of the next state (`discount * valueFunction.get(tp.outcome.sPrime)`).
						expectedValue += tp.prob * (tp.outcome.localReward + discount * valueFunction.get(tp.outcome.sPrime));
					}
					// Update the maximum value if the expected value of this action is greater.
					if (expectedValue > maxValue) {
						maxValue = expectedValue;
					}
				}
				// After evaluating all actions, store the maximum value in the new value function.
				newValueFunction.put(state, maxValue);
			}
			// Replace the old value function with the updated one for the next iteration.
			valueFunction = newValueFunction;
		}
	}
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		// This method extracts a policy from the current value function.
	    // A policy maps each state to the action (Move) that maximises the expected reward.
		
		/*
		 * YOUR CODE HERE
		 */
		
		Policy policy = new Policy(); // Create a new Policy object to store the extracted policy.
		// Iterate through all states in the value function.
	    for (Game state : valueFunction.keySet()) {
	    	// Skip terminal states since no action is needed for them.
	        if (state.isTerminal()) continue; 
	        Move bestMove = null; // Variable to store the best action for the current state.
	        double maxValue = Double.NEGATIVE_INFINITY;  // Initialise the maximum value as negative infinity.
	        // Iterate through all possible moves (actions) for the current state.
	        for (Move move : state.getPossibleMoves()) {
	        	// Compute the expected value of taking this action.
	            double expectedValue = 0.0;
	            // Generate transition probabilities and outcomes for the current state-action pair.
	            for (TransitionProb tp : mdp.generateTransitions(state, move)) {
	            	// Add the weighted contribution of this transition to the expected value.
	                // The weight is the transition probability (`tp.prob`), and the contribution includes:
	                // - The immediate reward (`tp.outcome.localReward`).
	                // - The discounted value of the next state (`discount * valueFunction.get(tp.outcome.sPrime)`).
	                expectedValue += tp.prob * (tp.outcome.localReward + discount * valueFunction.get(tp.outcome.sPrime));
	            }
	            // Update the maximum value and corresponding action if this action's expected value is higher.
	            if (expectedValue > maxValue) {
	                maxValue = expectedValue;
	                bestMove = move;
	            }
	        }
	        // After evaluating all possible moves, store the best move for the current state in the policy.
	        policy.policy.put(state, bestMove);
	    }
	    // Return the extracted policy, which maps each state to its optimal action.
	    return policy;
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
