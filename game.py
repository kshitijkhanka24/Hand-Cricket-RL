import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time

# ============================================================
# HAND CRICKET ENVIRONMENT
# ============================================================
class HandCricketEnv:
    """
    Hand Cricket Game with lives system
    - Batting: Score runs
    - Bowling: Get opponents out
    - Match has 2 innings
    """
    
    def __init__(self, max_lives=2):
        self.max_lives = max_lives
        self.reset_game()
    
    def reset_game(self):
        """Reset game for new innings"""
        self.batting_score = 0
        self.bowling_outs = 0
        self.current_lives = self.max_lives
        self.rounds_played = 0
        self.game_over = False
    
    def step(self, batting_throw, bowling_throw):
        """
        Execute one round of Hand Cricket
        Returns: runs_scored, out_status, lives_remaining
        """
        self.rounds_played += 1
        out = False
        runs = 0
        
        if batting_throw == bowling_throw:
            # Numbers match - Batter is OUT
            out = True
            self.current_lives -= 1
            self.bowling_outs += 1
            result = "OUT"
        else:
            # Numbers differ - Batter scores runs
            runs = abs(batting_throw - bowling_throw)
            self.batting_score += runs
            result = "RUNS"
        
        # Check if game is over (all lives lost)
        if self.current_lives <= 0:
            self.game_over = True
        
        return runs, out, self.current_lives, result

# ============================================================
# Q-LEARNING AGENT
# ============================================================
class QLearningAgent:
    """
    Q-Learning Agent for Hand Cricket
    Learns optimal bowling/batting strategy
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.5, mode='bowling'):
        """
        mode: 'bowling' (defensive) or 'batting' (aggressive)
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1
        self.mode = mode
        
        # Q-table: maps (state, action) -> expected reward
        self.q_table = defaultdict(lambda: np.zeros(6))
        
        self.human_history = []
        self.agent_history = []
        self.rewards_history = []
    
    def encode_state(self, last_throws):
        """Encode game state from last N throws"""
        if len(last_throws) == 0:
            return (0, 0)
        elif len(last_throws) == 1:
            return (0, last_throws[0])
        else:
            return (last_throws[-2], last_throws[-1])
    
    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(1, 6)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0] + 1
            return np.random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """Q-Learning update formula"""
        current_q = self.q_table[state][action - 1]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action - 1] = new_q
    
    def decay_epsilon(self):
        """Reduce exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def remember(self, human_action, agent_action, reward):
        """Track history"""
        self.human_history.append(human_action)
        self.agent_history.append(agent_action)
        self.rewards_history.append(reward)

# ============================================================
# TRAINING PHASE
# ============================================================
def train_agent(agent, episodes=100):
    """Train agent against simulated human"""
    print("\n" + "="*70)
    print("🎓 TRAINING PHASE: AI Learning from 100 Simulated Matches")
    print("="*70)
    
    game_stats = []
    
    for episode in range(episodes):
        env = HandCricketEnv(max_lives=2)
        agent.human_history = []
        agent.agent_history = []
        agent.rewards_history = []
        
        # Simulate human with bias
        bias = random.choice(["low", "high", "mid", "random"])
        
        while not env.game_over:
            # Simulated human throw
            if bias == "low":
                human_throw = random.choice([1, 2, 3, 1, 2])
            elif bias == "high":
                human_throw = random.choice([4, 5, 6, 5, 6])
            elif bias == "mid":
                human_throw = random.choice([3, 4])
            else:
                human_throw = random.randint(1, 6)
            
            state = agent.encode_state(agent.human_history)
            agent_throw = agent.choose_action(state, training=True)
            
            # Execute round
            runs, out, lives, result = env.step(human_throw, agent_throw)
            
            # Calculate reward
            if agent.mode == 'bowling':
                reward = 10 if out else -runs  # Reward for getting out
            else:
                reward = runs if not out else -10  # Reward for scoring
            
            next_state = agent.encode_state(agent.human_history + [human_throw])
            agent.update_q_value(state, agent_throw, reward, next_state)
            agent.remember(human_throw, agent_throw, reward)
        
        agent.decay_epsilon()
        
        game_stats.append({
            'episode': episode + 1,
            'final_score': env.batting_score,
            'total_outs': env.bowling_outs,
            'total_reward': sum(agent.rewards_history)
        })
        
        if (episode + 1) % 20 == 0:
            avg_score = np.mean([g['final_score'] for g in game_stats[-20:]])
            print(f"  Episode {episode + 1:3d}/100 | Avg Score: {avg_score:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("\n✅ Training Complete! AI is now ready to play.\n")
    return game_stats

# ============================================================
# COIN FLIP
# ============================================================
def coin_flip():
    """Interactive coin flip to decide who bats first"""
    print("\n" + "="*70)
    print("🪙 COIN FLIP - Decide who bats first")
    print("="*70)
    
    while True:
        choice = input("\nCall the coin flip - Heads (H) or Tails (T)? ").strip().upper()
        if choice in ['H', 'T']:
            break
        print("❌ Invalid! Enter H or T")
    
    flip_result = random.choice(['H', 'T'])
    print(f"\n🎲 Coin flip result: {flip_result}")
    
    if choice == flip_result:
        print("✅ YOU WIN THE TOSS! Choose your role:")
        print("  1. BATTING (Score runs)")
        print("  2. BOWLING (Get opponent out)")
        
        while True:
            role_choice = input("\nEnter 1 or 2: ").strip()
            if role_choice in ['1', '2']:
                human_role = 'batting' if role_choice == '1' else 'bowling'
                ai_role = 'bowling' if human_role == 'batting' else 'batting'
                print(f"\n✅ You chose: {human_role.upper()}")
                print(f"   AI will: {ai_role.upper()}")
                break
            print("❌ Invalid! Enter 1 or 2")
    else:
        print("❌ AI WINS THE TOSS! AI chooses its role...")
        # AI chooses strategically (usually bowling to get outs)
        ai_role = random.choice(['batting', 'bowling'])
        human_role = 'bowling' if ai_role == 'batting' else 'batting'
        print(f"   AI chooses: {ai_role.upper()}")
        print(f"   You will: {human_role.upper()}")
    
    return human_role, ai_role

# ============================================================
# INTERACTIVE MATCH
# ============================================================
def play_interactive_match(ai_batting, ai_bowling):
    """Play interactive match with human"""
    print("\n" + "="*70)
    print("🏏 LET'S PLAY HAND CRICKET!")
    print("="*70)
    
    # Innings 1
    print("\n" + "="*70)
    print("🔴 INNINGS 1 - First Team Bats")
    print("="*70)
    
    if ai_batting.mode == 'batting':
        print("\n🤖 AI is BATTING | 👤 You are BOWLING")
        inning1_ai_score = play_inning(ai_agent=ai_batting, human_role='bowling', target_score=None)
        inning1_human_score = 0
    else:
        print("\n👤 You are BATTING | 🤖 AI is BOWLING")
        inning1_human_score = play_inning(ai_agent=ai_bowling, human_role='batting', target_score=None)
        inning1_ai_score = 0
    
    print(f"\n📊 Innings 1 Result: AI={inning1_ai_score} | Human={inning1_human_score}")
    
    # Innings 2 - Need to beat Innings 1 score
    target_score = max(inning1_ai_score, inning1_human_score)
    print("\n" + "="*70)
    print("🔵 INNINGS 2 - Second Team Bats (Need to beat Innings 1)")
    print(f"⭐ TARGET SCORE: {target_score} runs")
    print("="*70)
    
    if ai_batting.mode == 'batting':
        print("\n👤 You are BATTING | 🤖 AI is BOWLING")
        inning2_human_score = play_inning(ai_agent=ai_bowling, human_role='batting', target_score=target_score)
        inning2_ai_score = 0
    else:
        print("\n🤖 AI is BATTING | 👤 You are BOWLING")
        inning2_ai_score = play_inning(ai_agent=ai_batting, human_role='bowling', target_score=target_score)
        inning2_human_score = 0
    
    print(f"\n📊 Innings 2 Result: Human={inning2_human_score} | AI={inning2_ai_score}")
    
    # Final Result
    total_human = inning1_human_score + inning2_human_score
    total_ai = inning1_ai_score + inning2_ai_score
    
    print("\n" + "="*70)
    print("🏆 MATCH RESULT")
    print("="*70)
    print(f"\nInnings 1: AI={inning1_ai_score} | Human={inning1_human_score}")
    print(f"Innings 2: AI={inning2_ai_score} | Human={inning2_human_score}")
    print(f"\nTotal Scores:")
    print(f"  👤 You:  {total_human} runs")
    print(f"  🤖 AI:   {total_ai} runs")
    
    if total_human > total_ai:
        print(f"\n🎉 CONGRATULATIONS! You WON by {total_human - total_ai} runs! 🏆")
        return "HUMAN"
    elif total_ai > total_human:
        print(f"\n🤖 AI WINS by {total_ai - total_human} runs! You got dominated! 🤖")
        return "AI"
    else:
        print(f"\n🤝 TIED MATCH! Both scored equally! 🤝")
        return "TIE"
    
    print("\n" + "="*70)

def play_inning(ai_agent, human_role, target_score=None):
    """Play one inning (batting side gets 2 lives)"""
    env = HandCricketEnv(max_lives=2)
    ai_agent.epsilon = 0.0  # No exploration during match
    ai_agent.human_history = []
    ai_agent.agent_history = []
    
    round_num = 0
    
    while not env.game_over:
        round_num += 1
        print(f"\n{'─'*70}")
        print(f"🔄 Ball {round_num} | Lives: {'❤️ ' * env.current_lives}")
        
        # Human input
        while True:
            try:
                human_throw = int(input(f"You pick (1-6): "))
                if 1 <= human_throw <= 6:
                    break
                print("❌ Invalid! Pick 1-6")
            except ValueError:
                print("❌ Invalid! Enter a number")
        
        # AI decision
        state = ai_agent.encode_state(ai_agent.human_history)
        ai_throw = ai_agent.choose_action(state, training=False)
        
        # Execute round
        runs, out, lives, result = env.step(human_throw, ai_throw)
        ai_agent.remember(human_throw, ai_throw, runs if not out else -10)
        
        print(f"AI picks: {ai_throw}")
        
        if result == "OUT":
            print(f"💥 OUT! Numbers matched ({human_throw} = {ai_throw})")
            print(f"❌ Life lost! Lives remaining: {lives}")
        else:
            print(f"✅ {runs} runs! ({human_throw} vs {ai_throw} = diff of {runs})")
            print(f"📈 Total Score: {env.batting_score}")
        
        # Check if target score is beaten (chase completed)
        if target_score is not None and env.batting_score > target_score:
            print(f"\n✨ TARGET BEATEN! Score {env.batting_score} > {target_score}")
            print(f"🏁 Innings Ends early!")
            break
    
    print(f"\n{'─'*70}")
    print(f"🏁 Innings End! Final Score: {env.batting_score} runs | Outs: {env.bowling_outs}")
    
    return env.batting_score

# ============================================================
# VISUALIZATION
# ============================================================
def plot_training_stats(game_stats):
    """Visualize training progress"""
    episodes = [g['episode'] for g in game_stats]
    scores = [g['final_score'] for g in game_stats]
    
    # Moving average
    window = 10
    scores_ma = np.convolve(scores, np.ones(window)/window, mode='valid')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hand Cricket - AI Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Scores
    axes[0].plot(episodes, scores, alpha=0.3, label='Per Game', color='blue')
    axes[0].plot(range(window, len(scores) + 1), scores_ma, linewidth=2.5, 
                 label=f'{window}-game average', color='darkblue')
    axes[0].set_xlabel('Episode', fontsize=11)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title('Training Scores Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Statistics
    avg_score = np.mean(scores)
    final_score = scores[-1]
    
    text = f"""
    TRAINING SUMMARY
    ━━━━━━━━━━━━━━━━━━━━
    Episodes Trained: 100
    
    Average Score: {avg_score:.1f} runs
    Final Score: {final_score:.1f} runs
    
    Starting Epsilon: 0.5
    Final Epsilon: {game_stats[-1]['episode']*0.995:.3f}
    
    Algorithm: Q-Learning
    State: Last 2 throws
    Reward: ±Runs/Outs
    """
    
    axes[1].text(0.1, 0.5, text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.7))
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "🏏 HAND CRICKET - Q-LEARNING AI 🏏" + " "*19 + "║")
    print("║" + " "*20 + "Module-I: Reinforcement Learning" + " "*17 + "║")
    print("╚" + "="*68 + "╝")
    
    # Train both agents (batting & bowling)
    print("\n🚀 Initializing AI agents...")
    ai_batting = QLearningAgent(learning_rate=0.15, discount_factor=0.95, epsilon=0.8, mode='batting')
    ai_bowling = QLearningAgent(learning_rate=0.15, discount_factor=0.95, epsilon=0.8, mode='bowling')
    
    # Training
    stats_batting = train_agent(ai_batting, episodes=100)
    stats_bowling = train_agent(ai_bowling, episodes=100)
    
    plot_training_stats(stats_batting)
    
    # Play interactive match
    print("\n\n" + "="*70)
    print("⏰ AI Training Complete! Ready to play!")
    print("="*70)
    
    total_matches_played = 0
    human_match_wins = 0
    ai_match_wins = 0
    
    while True:
        play_match = input("\nWant to play against the AI? (yes/no): ").strip().lower()
        
        if play_match == 'yes':
            total_matches_played += 1
            print(f"\n📊 Match #{total_matches_played}")
            
            human_role, ai_role = coin_flip()
            
            # Update AI modes based on roles
            ai_batting.mode = 'batting' if ai_role == 'batting' else 'bowling'
            ai_bowling.mode = 'bowling' if ai_role == 'bowling' else 'batting'
            
            # Enable learning from human's current match
            ai_batting.epsilon = 0.05  # Minimal exploration, mostly exploit
            ai_bowling.epsilon = 0.05
            
            input("\nPress ENTER to start the match...")
            match_winner = play_interactive_match(ai_batting, ai_bowling)
            
            if match_winner == "HUMAN":
                human_match_wins += 1
            elif match_winner == "AI":
                ai_match_wins += 1
            
            # Learn from human's patterns in this match
            print("\n" + "="*70)
            print("🧠 AI LEARNING FROM YOUR PLAY PATTERNS...")
            print("="*70)
            
            # Update Q-values based on human's playing pattern
            for i, (human_throw, ai_throw, reward) in enumerate(zip(
                ai_batting.human_history, ai_batting.agent_history, ai_batting.rewards_history)):
                state = ai_batting.encode_state(ai_batting.human_history[:i])
                next_state = ai_batting.encode_state(ai_batting.human_history[:i+1])
                ai_batting.update_q_value(state, ai_throw, reward * 2, next_state)  # Double learning rate
            
            for i, (human_throw, ai_throw, reward) in enumerate(zip(
                ai_bowling.human_history, ai_bowling.agent_history, ai_bowling.rewards_history)):
                state = ai_bowling.encode_state(ai_bowling.human_history[:i])
                next_state = ai_bowling.encode_state(ai_bowling.human_history[:i+1])
                ai_bowling.update_q_value(state, ai_throw, reward * 2, next_state)  # Double learning rate
            
            print("✅ AI has learned from your playing style!")
            print(f"   - Your batting patterns analyzed")
            print(f"   - Your bowling patterns analyzed")
            print(f"   - AI strategy updated for next match\n")
            
            # Show overall stats
            print("="*70)
            print("📈 OVERALL MATCH STATISTICS")
            print("="*70)
            print(f"Total Matches Played: {total_matches_played}")
            print(f"Your Match Wins: {human_match_wins}")
            print(f"AI Match Wins: {ai_match_wins}")
            if total_matches_played > 0:
                print(f"Your Win Rate: {human_match_wins/total_matches_played*100:.1f}%")
                print(f"AI Win Rate: {ai_match_wins/total_matches_played*100:.1f}%")
                
                if ai_match_wins > human_match_wins:
                    print(f"\n🤖 AI is DOMINATING! Winning by {ai_match_wins - human_match_wins} matches!")
                elif human_match_wins > ai_match_wins:
                    print(f"\n🎉 You're on fire! Leading by {human_match_wins - ai_match_wins} matches!")
                else:
                    print(f"\n🤝 It's a tie series!")
            print("="*70)
        else:
            print("\n✨ Thanks for playing! ✨")
            print(f"\nFinal Statistics:")
            print(f"  Total Matches: {total_matches_played}")
            print(f"  Your Wins: {human_match_wins}")
            print(f"  AI Wins: {ai_match_wins}")
            if total_matches_played > 0 and ai_match_wins > human_match_wins:
                print(f"\n🤖 AI successfully learned and dominated!")
            break
