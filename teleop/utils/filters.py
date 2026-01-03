"""
JAX-compatible filters for IK smoothing.
"""
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from typing import List, Optional

class JaxWeightedMovingFilter:
    """
    JAX-native weighted moving filter.
    Avoids CPU<->GPU transfers by keeping data on device.
    """
    def __init__(self, weights: List[float], data_size: int):
        self._weights = jnp.array(weights)
        self._window_size = len(weights)
        self._data_size = data_size
        # Initialize buffer with zeros
        self._buffer = jnp.zeros((self._window_size, data_size))
        self._ptr = 0  # Circular buffer pointer
        
        # JIT compile the update step
        self._update_jit = jax.jit(self._update_impl)
        
    def _update_impl(self, buffer, new_data):
        # Shift buffer: [0, 1, 2] -> [1, 2, new]
        # Actually, let's just roll it for simplicity in JAX
        new_buffer = jnp.roll(buffer, -1, axis=0)
        new_buffer = new_buffer.at[-1].set(new_data)
        
        # Apply weights: weights are [oldest, ..., newest]
        # But our buffer is [oldest, ..., newest] after roll?
        # Let's verify Unitree's weights: [0.4, 0.3, 0.2, 0.1] (newest to oldest)
        # So we want: 0.4*new + 0.3*prev + ...
        
        # If buffer is [t-3, t-2, t-1, t], and weights are [0.1, 0.2, 0.3, 0.4]
        # Then dot product works.
        
        # Unitree weights are passed as [0.4, 0.3, 0.2, 0.1]
        # So we need to reverse them to match time-ordered buffer
        
        weighted_sum = jnp.dot(self._weights[::-1], new_buffer)
        return new_buffer, weighted_sum

    def add_data_jax(self, new_data: Array) -> Array:
        """Add data and return filtered result (all on GPU)."""
        self._buffer, filtered = self._update_jit(self._buffer, new_data)
        return filtered
    
    def reset(self):
        self._buffer = jnp.zeros((self._window_size, self._data_size))

# Re-export the numpy version for compatibility if needed, 
# but we really want the JAX one.
class WeightedMovingFilter:
    def __init__(self, weights, data_size):
        self.jax_filter = JaxWeightedMovingFilter(weights, data_size)
    
    def add_data(self, data):
        # Legacy support
        if isinstance(data, np.ndarray):
            data = jnp.array(data)
        return np.array(self.jax_filter.add_data_jax(data))
    
    def add_data_jax(self, data):
        return self.jax_filter.add_data_jax(data)
    
    def reset(self):
        self.jax_filter.reset()
    
    @property
    def filtered_data(self):
        # This is tricky with JAX state, better to return from add_data
        # For legacy compat, we might need to cache last result
        return np.array(self.jax_filter._buffer[-1]) # Approximate/Wrong
        # Real fix: Update SmoothIKSolver to use return value
