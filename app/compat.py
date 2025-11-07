import sys
import typing


def patch_forward_ref():
    if sys.version_info < (3, 13):
        return

    forward_ref = getattr(typing, "ForwardRef", None)
    if forward_ref is None:
        return

    original = getattr(forward_ref, "_evaluate", None)
    if original is None:
        return

    if getattr(original, "__patched_for_py313__", False):
        return

    def _evaluate(self, globalns, localns, recursive_guard=None):
        if recursive_guard is None:
            recursive_guard = set()
        return original(self, globalns, localns, recursive_guard)

    _evaluate.__patched_for_py313__ = True
    forward_ref._evaluate = _evaluate  # type: ignore[attr-defined]


patch_forward_ref()
