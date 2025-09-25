import numpy as np
from typing import Callable, Union, Tuple, Dict, Any

def reshape_coord_and_calculate(
    func: Callable[..., Union[np.ndarray, Tuple, Dict]], 
    coord: np.ndarray,
    *args: Any,
    **kwargs: Any
) -> Union[np.ndarray, Tuple, Dict]:
    """
    Reshape coordinates to (-1, 3), compute using func, then reshape results back.
    
    Args:
        func: Computation function that takes (N,3) array and returns:
              - np.ndarray (scalar/vector/matrix)
              - tuple of arrays
              - dictionary of arrays
        coord: Input coordinate array with last dim=3
        *args: Additional positional arguments for func
        **kwargs: Additional keyword arguments for func
    
    Returns:
        Results with same structure as func's output but reshaped to match input shape
    
    Raises:
        AssertionError: If coord's last dimension is not 3
    """
    assert coord.shape[-1] == 3, f'reshape coord to (-1, 3) failed: coord shape {coord.shape}'
    point_shape = coord.shape[:-1]  # Original shape (excluding last dim)
    coord = coord.reshape((-1, 3))  # Flatten to (N,3)
    
    ret = func(coord, *args, **kwargs)
    
    def _reshape_array(arr: np.ndarray) -> np.ndarray:
        """Reshape array to match original point cloud structure"""
        if arr.ndim == 1:
            return arr.reshape(point_shape)
        return arr.reshape((*point_shape, arr.shape[-1]))
    
    if isinstance(ret, dict):
        # Handle dictionary case
        return {k: _reshape_array(v) for k, v in ret.items()}
    elif isinstance(ret, tuple):
        # Handle tuple case
        return tuple(_reshape_array(r) for r in ret)
    else:
        # Handle single array case
        return _reshape_array(ret)
