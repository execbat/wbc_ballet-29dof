"""A small, dependency-free re-implementation of IsaacLab's ``configclass``.

Why this exists
----------------
mjlab's own manager configs (``ManagerBasedRlEnvCfg``, ``ObservationGroupCfg``,
``RewardTermCfg``, ...) are plain ``dict``-keyed dataclasses: you build a
``dict[str, RewardTermCfg]`` by hand and hand it to the env cfg. That is fine
for a library, but it makes *task authoring* hard to keep organized: every
task ends up as one giant function that pokes at ``cfg.rewards["name"]``.

IsaacLab instead lets you *declare* each manager's terms as named attributes
on a small class, e.g.::

    @configclass
    class RewardsCfg:
        track_lin_vel_xy_exp = RewTerm(func=..., weight=1.0)

    @configclass
    class G1Rewards(RewardsCfg):
        track_lin_vel_xy_exp = RewTerm(func=..., weight=2.0)  # override
        extra_term = RewTerm(func=...)                         # add
        some_term = None                                        # disable

This module reproduces just enough of that mechanism to write mjlab task
configs the same way, without depending on Isaac Sim / Isaac Lab at all:

- ``configclass``: turns a plain class into a ``dataclasses.dataclass``,
  recursing into nested classes and making every non-primitive default
  (nested configclass instances, term configs, tuples, dicts, ...) safe to
  share across instances by wrapping it in a ``default_factory`` that
  deep-copies it. This is what lets ``PolicyCfg: PolicyCfg = PolicyCfg()``
  work as a class attribute without every instance secretly sharing state.
- ``MISSING``: sentinel for "must be overridden by a subclass/robot cfg".

``mjlab_microduck.utils.manager_compat`` builds on top of this to convert a
tree of configclass instances into the plain dicts mjlab's managers expect.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from typing import Any, TypeVar

__all__ = ["MISSING", "configclass"]

T = TypeVar("T")

MISSING: Any = dataclasses.MISSING
"""Sentinel marking a field that a subclass is required to fill in.

Usage mirrors ``dataclasses.MISSING`` / IsaacLab's ``configclass``:

    @configclass
    class SceneCfg:
        robot: EntityCfg = MISSING  # must be set by a robot-specific subclass
"""

_PRIMITIVE_TYPES = (int, float, str, bool, bytes, type(None))


def _is_safe_as_bare_default(value: Any) -> bool:
    """True if ``value`` can be used directly as a dataclass default.

    Immutable/primitive values (including enums) are safe to share across
    instances as-is. Everything else (dataclass instances, dicts, lists,
    tuples of non-primitives, ...) gets deep-copied per-instance via a
    ``default_factory`` instead, so mutating one instance's config never
    leaks into another's.
    """
    if value is MISSING:
        return True
    if isinstance(value, _PRIMITIVE_TYPES):
        return True
    if isinstance(value, tuple):
        return all(_is_safe_as_bare_default(v) for v in value)
    import enum

    if isinstance(value, enum.Enum):
        return True
    return False


def _make_default_factory(value: Any):
    # Bind a *copy* of value at decoration time so later mutation of the
    # original object (e.g. a module-level constant) doesn't leak in.
    template = copy.deepcopy(value)

    def _factory(_template=template):
        return copy.deepcopy(_template)

    return _factory


def configclass(cls: type[T]) -> type[T]:
    """Decorator turning ``cls`` into a mutable-default-safe dataclass.

    Behaves like IsaacLab's ``@configclass``:

    - Nested plain classes (not yet dataclasses) are recursively decorated,
      so you can write ``class PolicyCfg(ObsGroup): ...`` inside an outer
      ``@configclass class ObservationsCfg:`` body without decorating the
      inner class explicitly.
    - Any class-attribute default that isn't a primitive/immutable value is
      wrapped in a ``default_factory`` that deep-copies it, so instances
      never share mutable state.
    - ``__post_init__`` (if defined) is preserved and called by the
      generated ``__init__`` as usual for dataclasses.
    """
    # 1) Recursively promote nested plain classes to configclasses.
    for name, value in list(vars(cls).items()):
        if (
            inspect.isclass(value)
            and not dataclasses.is_dataclass(value)
            and not name.startswith("__")
        ):
            setattr(cls, name, configclass(value))

    # 2) Rewrite defaults for annotated fields so mutable ones get a
    #    default_factory instead of being shared across instances.
    annotations = dict(cls.__dict__.get("__annotations__", {}))
    for name in annotations:
        if name not in cls.__dict__:
            continue  # annotated but no default -> required field, leave it.
        current = cls.__dict__[name]
        if isinstance(current, dataclasses.Field):
            continue  # already explicit field()/MISSING handling.
        if not _is_safe_as_bare_default(current):
            setattr(
                cls,
                name,
                dataclasses.field(default_factory=_make_default_factory(current)),
            )

    return dataclasses.dataclass(cls)  # type: ignore[return-value]
