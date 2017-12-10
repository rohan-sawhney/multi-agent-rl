import gym


class MyMonitor(gym.wrappers.Monitor):

    def _after_step(self, observation, reward, done, info):
        if not self.enabled:
            return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation
            # will be the first one of the new episode
            self._reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Semisupervised envs modify the rewards, but we want the original when
        # scoring
        if info.get('true_reward', None):
            reward = info['true_reward']

        # Record stats
        self.stats_recorder.after_step(
            observation, np.sum(reward), any(done), info)
        # Record video
        self.video_recorder.capture_frame()

        return done
