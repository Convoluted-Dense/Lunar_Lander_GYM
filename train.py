import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import layers
from collections import deque
import random
import time

# === FAST GPU CONFIGURATION ===
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # Mixed precision (new API)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
    except RuntimeError as e:
        print(e)

# === OPTIMIZED Q-NETWORK ===
def create_q_model():
    inputs = layers.Input(shape=(8,), dtype=tf.float32)
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(4, activation="linear", dtype='float32')(x)  # Force float32 output
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# === HYPERPARAMETERS  ===
env = gym.make("LunarLander-v3")
model = create_q_model()
target_model = create_q_model()
target_model.set_weights(model.get_weights())

# Faster optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
loss_fn = tf.keras.losses.Huber()

# Smaller replay buffer for faster sampling
replay_buffer = deque(maxlen=50_000)
batch_size = 32
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99
gamma = 0.99
num_episodes = 300 
target_update_freq = 100
train_frequency = 4

# === VECTORIED TRAINING STEP ===
@tf.function
def train_step(states, actions, rewards, next_states, dones):
    """Vectorized training step - much faster!"""
    next_q_values = target_model(next_states, training=False)
    max_next_q_values = tf.reduce_max(next_q_values, axis=1)
    target_q_values = rewards + (1.0 - dones) * gamma * max_next_q_values
    
    masks = tf.one_hot(actions, 4)
    
    with tf.GradientTape() as tape:
        q_values = model(states, training=True)
        q_values_selected = tf.reduce_sum(q_values * masks, axis=1)
        loss = loss_fn(target_q_values, q_values_selected)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ===TRAINING LOOP ===
start_time = time.time()
steps_done = 0

print("Starting training...")
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    episode_steps = 0
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            state_tensor = tf.constant(state.reshape(1, -1), dtype=tf.float32)
            q_values = model(state_tensor, training=False)
            action = int(tf.argmax(q_values[0]).numpy())
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store experience
        replay_buffer.append((
            state.astype(np.float32),
            action,
            float(reward),
            next_state.astype(np.float32),
            float(done)
        ))
        
        state = next_state
        total_reward += reward
        steps_done += 1
        episode_steps += 1
        
        # Train less frequently for speed
        if len(replay_buffer) >= batch_size and steps_done % train_frequency == 0:
            # Sample batch
            batch = random.sample(replay_buffer, batch_size)
            states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
            
            # Convert to tensors
            states_b = tf.constant(states_b, dtype=tf.float32)
            actions_b = tf.constant(actions_b, dtype=tf.int32)
            rewards_b = tf.constant(rewards_b, dtype=tf.float32)
            next_states_b = tf.constant(next_states_b, dtype=tf.float32)
            dones_b = tf.constant(dones_b, dtype=tf.float32)
            
            
            train_step(states_b, actions_b, rewards_b, next_states_b, dones_b)
        
        # Update target network
        if steps_done % target_update_freq == 0:
            target_model.set_weights(model.get_weights())
    
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Print progress every 10 episodes
    if episode % 10 == 0:
        elapsed_time = time.time() - start_time
        print(f"Episode {episode}: Reward={total_reward:.1f}, "
              f"Epsilon={epsilon:.3f}, Steps={steps_done}, "
              f"Time={elapsed_time:.1f}s")

print(f"Training completed in {time.time() - start_time:.1f} seconds")
model.save("lunar_lander_dqn.h5")
env.close()