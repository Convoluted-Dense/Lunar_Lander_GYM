import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import layers
import time

# === Recreate the same model architecture ===
def create_q_model():
    inputs = layers.Input(shape=(8,), dtype=tf.float32)
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(4, activation="linear", dtype='float32')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# === Load the trained model ===
print("Loading trained model...")
try:
    # Try to load the model
    model = tf.keras.models.load_model("fast_lunar_lander_dqn.h5")
    print("Model loaded successfully!")
except:
    print("Model file not found! Creating a new model and loading weights...")
    model = create_q_model()
    model.load_weights("fast_lunar_lander_dqn.h5")
    print("Weights loaded successfully!")

# === Create environment (with rendering) ===
env = gym.make("LunarLander-v3", render_mode="human")

# === Test the model ===
def test_model(num_episodes=5, delay=0.01):
    """Test the trained model"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        print(f"\n=== Episode {episode + 1} ===")
        
        while not done:
            # Get Q-values from the model
            state_tensor = tf.constant(state.reshape(1, -1), dtype=tf.float32)
            q_values = model(state_tensor, training=False)
            action = int(tf.argmax(q_values[0]).numpy())
            
            # Take action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            # Optional: Add small delay to watch the agent
            if delay > 0:
                time.sleep(delay)
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {step_count}")
    
    # Print summary
    avg_reward = np.mean(total_rewards)
    print(f"\n=== TEST RESULTS ===")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    
    return total_rewards

# === Run the test ===
if __name__ == "__main__":
    print("Starting model evaluation...")
    print("Close the environment window to stop testing")
    
    try:
        rewards = test_model(num_episodes=10, delay=0.02)  # 10 episodes with slight delay
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    finally:
        env.close()
        print("Environment closed")