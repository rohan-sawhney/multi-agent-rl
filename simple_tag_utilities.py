import numpy as np
def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False

def count_agent_collisions(env):
    """
    Returns a count of all collisions between two non-cooperating agents.
    """
    counts = []
    for i in range(env.n):
        collide_i = 0
        for j in range(env.n):
            is_collide = is_collision(env.agents[i], env.agents[j])
            if is_collide and env.agents[i].adversary is not env.agents[j].adversary:
                collide_i += 1
        counts.append(collide_i)
    return counts
