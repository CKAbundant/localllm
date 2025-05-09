"""Descriptor class to be used as function decorator to compute time taken
for class methods."""

import time
from functools import wraps
from typing import Any, Callable, Type, TypeVar

# Create generic type variable 'T'
T = TypeVar("T")


class TimedMethod:
    """Descriptor class to compute time taken to invoke a particlar class method.

    - Used as function decorator for instance methods ONLY.
    - NOT meant for static or class methods.

    Args:
        func (Callable):
            Specific class method, which we are interested to determine
            its computation time.

    Attributes:
        func (Callable):
            Specific class method, which we are interested to determine
            its computation time.
    """

    def __init__(self, func: Callable):
        self.func = func

    def __get__(self, instance: T, owner: Type[T]) -> Callable:
        """Invoked when decorated method (i.e. decorated by 'TimedMethod') is
        accessed via an class instance.

        Args:
            instance (T): Instance of class 'owner'.
            owner (Type[T]): Class containing methods to be assessed.

        Returns:
            wrapper (Callable): Callable that wraps the original method.
        """

        @wraps(self.func)  # Copy 'self.func' metadata to 'wrapper' function
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()

            # Manually input class instance to 'self.func'
            result = self.func(instance, *args, **kwargs)

            # Compute elapsed time
            elapsed = time.perf_counter() - start

            # Create 'timings' attribute if not present in 'instance'
            if not hasattr(instance, "timings"):
                instance.timings = []

            instance.timings.append(elapsed)

            return result

        return wrapper
