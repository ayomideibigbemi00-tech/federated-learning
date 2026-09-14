import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import traceback
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TelecomEdgeEnv(gym.Env):
    """
    Telecom edge resource-allocation environment.

    Observation:
        [network_load, latency, anomaly_severity]
        Each value is in the range 0-4.

    Actions:
        0 = Normal operation
        1 = Increase bandwidth
        2 = Prioritize critical traffic
        3 = Throttle suspicious traffic

    The environment is intentionally small and discrete so that:
        - PPO can train on it easily
        - a tabular Q-learning implementation could be added later
        - the behavior is easy to explain during the hackathon
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=50):
        super().__init__()

        self.max_steps = max_steps

        self.observation_space = spaces.MultiDiscrete(
            np.array([5, 5, 5], dtype=np.int32)
        )

        self.action_space = spaces.Discrete(4)

        self.state = np.zeros(3, dtype=np.int32)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        self.state = self.np_random.integers(
            low=np.array([0, 0, 0], dtype=np.int32),
            high=np.array([5, 5, 5], dtype=np.int32),
            dtype=np.int32,
        )

        return self.state.copy(), {}

    def step(self, action):
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1

        load, latency, severity = map(int, self.state)

        # ---------------------------------------------------------
        # 1. APPLY AGENT ACTION
        # ---------------------------------------------------------

        if action == 0:
            # Normal operation.
            pass

        elif action == 1:
            # Increase bandwidth.
            # Reduces network load but consumes resources.
            load = max(0, load - 2)

        elif action == 2:
            # Prioritize critical traffic.
            # Improves latency but increases resource/load pressure.
            latency = max(0, latency - 2)
            load = min(4, load + 1)

        elif action == 3:
            # Throttle suspicious traffic.
            # Strongly reduces anomaly severity, but may increase latency.
            if severity > 0:
                severity = max(0, severity - 2)
                latency = min(4, latency + 1)

        # ---------------------------------------------------------
        # 2. NATURAL ENVIRONMENT DYNAMICS
        # ---------------------------------------------------------

        # Network traffic changes naturally.
        traffic_change = int(self.np_random.integers(-1, 2))
        load = int(np.clip(load + traffic_change, 0, 4))

        # High network load increases latency.
        if load >= 3:
            latency += 1
        elif load == 0:
            latency -= 1

        latency = int(np.clip(latency, 0, 4))

        # New anomaly events occasionally appear.
        if self.np_random.random() < 0.15:
            severity += int(self.np_random.integers(1, 3))

        severity = int(np.clip(severity, 0, 4))

        # ---------------------------------------------------------
        # 3. UPDATE STATE
        # ---------------------------------------------------------

        self.state = np.array(
            [load, latency, severity],
            dtype=np.int32,
        )

        # ---------------------------------------------------------
        # 4. REWARD
        # ---------------------------------------------------------

        reward = (
            -1.0 * load
            -1.2 * latency
            -1.5 * severity
        )

        # Action costs.
        if action == 1:
            reward -= 0.5

        elif action == 2:
            reward -= 0.3

        elif action == 3:
            reward -= 0.7

        # Small bonus for achieving a good network state.
        if load <= 1 and latency <= 1 and severity <= 1:
            reward += 2.0

        terminated = False
        truncated = self.steps >= self.max_steps

        return (
            self.state.copy(),
            float(reward),
            terminated,
            truncated,
            {},
        )


class EdgeAgent:
    """
    Unified interface for all edge agents.

    The Streamlit application only needs:
        action = agent.act(observation)

    This keeps the UI independent of whether the underlying
    implementation is PPO, Q-learning, or the deterministic fallback.
    """

    def __init__(self, model=None, use_ppo=True):
        self.model = model
        self.use_ppo = use_ppo

    def act(self, observation):
        observation = np.asarray(observation, dtype=np.int32)

        if self.use_ppo and self.model is not None:
            action, _ = self.model.predict(
                observation,
                deterministic=True,
            )

            action = int(action)

            if not 0 <= action < 4:
                raise ValueError(
                    f"PPO returned invalid action: {action}"
                )

            return action

        return self.heuristic_action(observation)

    @staticmethod
    def heuristic_action(observation):
        load, latency, severity = map(int, observation)

        # Severe anomaly gets highest priority.
        if severity >= 3:
            return 3

        # Very high latency gets priority handling.
        if latency >= 3:
            return 2

        # Very high network load gets more bandwidth.
        if load >= 3:
            return 1

        return 0


def check_environment():
    """
    Validate the custom Gymnasium environment with Stable-Baselines3.
    """

    from stable_baselines3.common.env_checker import check_env

    print("--- Environment Validation ---")

    env = TelecomEdgeEnv()

    check_env(
        env,
        warn=True,
        skip_render_check=True,
    )

    print("[OK] Gymnasium environment validation passed.")

    observation, info = env.reset(seed=42)

    print(f"[OK] Initial observation: {observation}")

    for action in range(env.action_space.n):
        observation, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(observation)
        assert env.action_space.contains(action)

        print(
            f"[OK] Action {action} -> "
            f"observation={observation}, "
            f"reward={reward:.2f}"
        )

        env.reset(seed=42)

    env.close()


def run_smoke_test():
    """
    Day 1 PPO/SB3 compatibility and checkpoint smoke test.

    Tests:
        1. Stable-Baselines3 import
        2. Gymnasium environment validation
        3. PPO construction
        4. PPO micro-training
        5. Model save/load
        6. Unified EdgeAgent interface
        7. Valid action prediction

    If PPO fails, a deterministic heuristic fallback is verified.
    """

    print("=" * 60)
    print("DAY 1 EDGE RL SMOKE TEST")
    print("=" * 60)

    checkpoint_path = "ppo_smoke_test"

    try:
        # ---------------------------------------------------------
        # 1. IMPORT PPO
        # ---------------------------------------------------------

        from stable_baselines3 import PPO

        print("[OK] Stable-Baselines3 imported successfully.")

        # ---------------------------------------------------------
        # 2. VALIDATE ENVIRONMENT
        # ---------------------------------------------------------

        check_environment()

        # ---------------------------------------------------------
        # 3. CREATE ENVIRONMENT
        # ---------------------------------------------------------

        env = TelecomEdgeEnv()

        # ---------------------------------------------------------
        # 4. CREATE PPO
        # ---------------------------------------------------------

        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=64,
            batch_size=16,
            learning_rate=3e-4,
            verbose=0,
            seed=42,
        )

        print("[OK] PPO model initialized.")

        # ---------------------------------------------------------
        # 5. MICRO TRAINING
        # ---------------------------------------------------------

        model.learn(
            total_timesteps=512,
            progress_bar=False,
        )

        print("[OK] PPO micro-training completed.")

        # ---------------------------------------------------------
        # 6. CHECKPOINT SAVE
        # ---------------------------------------------------------

        model.save(checkpoint_path)

        print("[OK] PPO checkpoint saved.")

        # ---------------------------------------------------------
        # 7. CHECKPOINT LOAD
        # ---------------------------------------------------------

        loaded_model = PPO.load(
            checkpoint_path,
            env=env,
            device="cpu"
        )

        print("[OK] PPO checkpoint loaded.")

        # ---------------------------------------------------------
        # 8. TEST UNIFIED AGENT INTERFACE
        # ---------------------------------------------------------

        agent = EdgeAgent(
            model=loaded_model,
            use_ppo=True,
        )

        observation, _ = env.reset(seed=42)

        action = agent.act(observation)

        assert env.action_space.contains(action)

        print(
            f"[OK] EdgeAgent returned valid action: "
            f"{action}"
        )

        # ---------------------------------------------------------
        # 9. RUN SEVERAL INFERENCE STEPS
        # ---------------------------------------------------------

        print("\n--- PPO Inference Test ---")

        observation, _ = env.reset(seed=123)

        total_reward = 0.0

        for step in range(10):
            action = agent.act(observation)

            assert env.action_space.contains(action)

            observation, reward, terminated, truncated, _ = env.step(
                action
            )

            total_reward += reward

            print(
                f"Step {step + 1:02d} | "
                f"State={observation} | "
                f"Action={action} | "
                f"Reward={reward:.2f}"
            )

            if terminated or truncated:
                break

        print(
            f"\n[OK] PPO inference completed. "
            f"Total reward: {total_reward:.2f}"
        )

        env.close()

        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED")
        print("PPO architecture is available.")
        print("=" * 60)

        return {
            "status": "ppo",
            "model": loaded_model,
        }

    except Exception:
        print("\n" + "=" * 60)
        print("PPO / SB3 SMOKE TEST FAILED")
        print("=" * 60)

        print(traceback.format_exc())

        print("\nSwitching to deterministic heuristic fallback.")

        # ---------------------------------------------------------
        # FALLBACK TEST
        # ---------------------------------------------------------

        try:
            fallback_env = TelecomEdgeEnv()

            fallback_agent = EdgeAgent(
                model=None,
                use_ppo=False,
            )

            observation, _ = fallback_env.reset(seed=42)

            action = fallback_agent.act(observation)

            assert fallback_env.action_space.contains(action)

            print(
                f"[OK] Heuristic fallback returned valid "
                f"action: {action}"
            )

            fallback_env.close()

            print("\n" + "=" * 60)
            print("FALLBACK READY")
            print("=" * 60)

            return {
                "status": "heuristic",
                "model": None,
            }

        except Exception:
            print("\n[FAILED] Even the heuristic fallback failed.")
            print(traceback.format_exc())

            raise

    finally:
        # ---------------------------------------------------------
        # CLEAN UP SMOKE-TEST CHECKPOINT
        # ---------------------------------------------------------

        smoke_checkpoint = f"{checkpoint_path}.zip"

        if os.path.exists(smoke_checkpoint):
            os.remove(smoke_checkpoint)


if __name__ == "__main__":
    run_smoke_test()