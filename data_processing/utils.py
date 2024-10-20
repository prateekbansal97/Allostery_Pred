def load_windows(timestep):
    return {
        'train': [0, int(timestep * 0.8)],
        'valid': [int(timestep * 0.8), int(timestep * 0.9) + 1],
        'test': [int(timestep * 0.9) + 1, int(timestep) - 1]
    }
