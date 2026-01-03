"""
WeightedMovingFilter for smooth IK output.

Ported from Unitree's xr_teleoperate implementation for reducing
jitter in joint angle commands during VR teleoperation.

The filter applies exponential smoothing over the last N frames,
with higher weights given to more recent values.
"""

import numpy as np
from typing import Optional, List


class WeightedMovingFilter:
    """
    Applies weighted moving average filtering to joint data.
    
    This filter smooths IK output to reduce jitter while maintaining
    responsiveness. Uses a weighted combination of the last N frames
    where more recent frames have higher weight.
    
    Example:
        >>> filter = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], data_size=7)
        >>> for frame in ik_solutions:
        ...     filter.add_data(frame)
        ...     smooth_solution = filter.filtered_data
        ...     send_to_robot(smooth_solution)
    """
    
    def __init__(
        self,
        weights: List[float] = [0.4, 0.3, 0.2, 0.1],
        data_size: int = 7
    ):
        """
        Initialize the filter.
        
        Args:
            weights: Filter weights from newest to oldest. Must sum to 1.0.
                    Default [0.4, 0.3, 0.2, 0.1] gives 4-frame smoothing.
            data_size: Number of elements in each data frame (e.g., 7 for 7-DOF arm).
        """
        self._weights = np.array(weights)
        self._window_size = len(weights)
        
        if not np.isclose(np.sum(self._weights), 1.0):
            raise ValueError(
                f"[WeightedMovingFilter] Weights must sum to 1.0, got {np.sum(self._weights)}"
            )
        
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue: List[np.ndarray] = []
    
    def _apply_filter(self) -> np.ndarray:
        """Apply the weighted filter to the data queue."""
        if len(self._data_queue) < self._window_size:
            # Not enough history, return most recent
            return self._data_queue[-1]
        
        # Stack data and apply weighted convolution
        data_array = np.array(self._data_queue)
        filtered = np.zeros(self._data_size)
        
        for i in range(self._data_size):
            filtered[i] = np.convolve(
                data_array[:, i], 
                self._weights, 
                mode='valid'
            )[-1]
        
        return filtered
    
    def add_data(self, new_data: np.ndarray) -> None:
        """
        Add a new data frame to the filter.
        
        Args:
            new_data: Array of shape (data_size,) with new joint values.
        """
        new_data = np.asarray(new_data).flatten()
        
        if len(new_data) != self._data_size:
            raise ValueError(
                f"Expected data of size {self._data_size}, got {len(new_data)}"
            )
        
        # Skip duplicate consecutive data
        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return
        
        # Maintain sliding window
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)
        
        self._data_queue.append(new_data.copy())
        self._filtered_data = self._apply_filter()
    
    @property
    def filtered_data(self) -> np.ndarray:
        """Get the current filtered output."""
        return self._filtered_data.copy()
    
    @property
    def is_ready(self) -> bool:
        """Whether the filter has enough history for full smoothing."""
        return len(self._data_queue) >= self._window_size
    
    def reset(self) -> None:
        """Clear the filter history."""
        self._data_queue.clear()
        self._filtered_data = np.zeros(self._data_size)


class ExponentialFilter:
    """
    Simple exponential smoothing filter (alternative to WeightedMovingFilter).
    
    Uses: filtered = alpha * new_data + (1 - alpha) * filtered
    Lower alpha = more smoothing but more latency.
    
    For VR teleoperation:
    - alpha = 0.5: Good balance (default)
    - alpha = 0.7: More responsive, more jitter
    - alpha = 0.3: Very smooth, noticeable lag
    """
    
    def __init__(self, alpha: float = 0.5, data_size: int = 7):
        """
        Initialize exponential filter.
        
        Args:
            alpha: Smoothing factor in [0, 1]. Higher = more responsive.
            data_size: Number of elements per frame.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"Alpha must be in (0, 1], got {alpha}")
        
        self._alpha = alpha
        self._data_size = data_size
        self._filtered_data: Optional[np.ndarray] = None
    
    def add_data(self, new_data: np.ndarray) -> None:
        """Add new data and update filtered output."""
        new_data = np.asarray(new_data).flatten()
        
        if self._filtered_data is None:
            self._filtered_data = new_data.copy()
        else:
            self._filtered_data = (
                self._alpha * new_data + 
                (1 - self._alpha) * self._filtered_data
            )
    
    @property
    def filtered_data(self) -> np.ndarray:
        """Get current filtered output."""
        if self._filtered_data is None:
            return np.zeros(self._data_size)
        return self._filtered_data.copy()
    
    def reset(self) -> None:
        """Clear filter state."""
        self._filtered_data = None
