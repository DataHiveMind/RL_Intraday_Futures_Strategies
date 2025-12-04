from typing import Any, Dict, Optional


class Registry:
    """Simple decorator-based registry for classes/functions.

    Example:
            registry = Registry("agents")

            @registry.register()
            class MyAgent: ...

            cls = registry.get("MyAgent")
    """

    def __init__(self, name: str):
        self._name = name
        self._dict: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None):
        """Decorator to register a class or function under `name` (or its __name__)."""

        def decorator(obj: Any):
            key = name or getattr(obj, "__name__", None)
            if key is None:
                raise ValueError("Cannot register object without a name")
            if key in self._dict:
                raise KeyError(f"An object is already registered under name '{key}'")
            self._dict[key] = obj
            return obj

        return decorator

    def add(self, obj: Any, name: Optional[str] = None):
        """Directly add an object to the registry."""
        key = name or getattr(obj, "__name__", None)
        if key is None:
            raise ValueError("Cannot add object without a name")
        if key in self._dict:
            raise KeyError(f"An object is already registered under name '{key}'")
        self._dict[key] = obj
        return obj

    def get(self, name: str) -> Any:
        return self._dict[name]

    def create(self, name: str, *args, **kwargs) -> Any:
        """If the registered object is callable, instantiate/call it with provided args."""
        obj = self.get(name)
        if callable(obj):
            return obj(*args, **kwargs)
        return obj

    def list(self):
        return list(self._dict.keys())

    def clear(self):
        self._dict.clear()


# Convenience default registry instance
default_registry = Registry("default")

__all__ = ["Registry", "default_registry"]
