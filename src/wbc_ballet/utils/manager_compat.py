"""Bridge between IsaacLab-flavored task configs and mjlab's manager dicts.

mjlab's ``ManagerBasedRlEnvCfg`` (and every manager under it) is configured
with plain dicts, e.g. ``rewards: dict[str, RewardTermCfg]``. This module lets
task authors instead declare those terms as named attributes on a
``@configclass`` (see ``mjlab_microduck.utils.configclass``), the way IsaacLab
tasks do, and converts that declarative tree into the dicts mjlab actually
consumes at env-construction time.

Convention used throughout ``mjlab_microduck.tasks``:

- A "flat" term group (rewards, events, terminations, curriculum, commands,
  actions) is a ``@configclass`` whose fields are each either a mjlab
  ``...TermCfg`` instance (term enabled) or ``None`` (term disabled/removed).
  ``group_to_dict`` turns this into ``dict[str, TermCfg]``, dropping the
  ``None`` entries -- so disabling a term is just ``some_term: RewTerm | None
  = None`` in a subclass, exactly like IsaacLab.
- Observations are two-level: an ``ObservationsCfg`` whose fields are
  observation *groups* (e.g. ``actor``, ``critic``), each an ``ObsGroup``
  subclass whose own fields are ``ObservationTermCfg`` instances (or
  ``None``) plus a handful of group-level settings (``enable_corruption``,
  ``concatenate_terms``, ...). ``observations_to_dict`` handles both levels.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from wbc_ballet.utils.configclass import configclass

__all__ = ["ObsGroup", "group_to_dict", "observations_to_dict"]


@configclass
class ObsGroup:
    """Base class for an observation group (e.g. ``PolicyCfg``/``CriticCfg``).

    Subclass this and add ``ObservationTermCfg`` (``ObsTerm``) instances as
    class attributes -- one per observation term, in the order they should be
    concatenated. The handful of attributes declared here configure the
    *group* itself and are peeled off separately by ``observations_to_dict``;
    everything else on the subclass is treated as a term.
    """

    concatenate_terms: bool = True
    concatenate_dim: int = -1
    enable_corruption: bool = False
    history_length: int | None = None
    flatten_history_dim: bool = True
    nan_policy: str = "disabled"
    nan_check_per_term: bool = True


# Field names that belong to the group itself, not to an individual term.
_OBS_GROUP_OWN_FIELDS = {f.name for f in dataclasses.fields(ObsGroup)}


def group_to_dict(group: Any) -> dict[str, Any]:
    """Convert a flat ``@configclass`` term group into ``dict[str, TermCfg]``.

    Fields whose value is ``None`` are treated as "disabled" and dropped --
    this is how a robot-specific subclass removes a term inherited from its
    base class (``some_term: RewTerm | None = None``).
    """
    if group is None:
        return {}
    result: dict[str, Any] = {}
    for f in dataclasses.fields(group):
        value = getattr(group, f.name)
        if value is None:
            continue
        result[f.name] = value
    return result


def _obs_group_to_mjlab(group: Any) -> ObservationGroupCfg:
    terms: dict[str, ObservationTermCfg] = {}
    group_kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(group):
        value = getattr(group, f.name)
        if f.name in _OBS_GROUP_OWN_FIELDS:
            group_kwargs[f.name] = value
        elif value is not None:
            if not isinstance(value, ObservationTermCfg):
                raise TypeError(
                    f"Observation group field '{f.name}' must be an "
                    f"ObservationTermCfg (ObsTerm) or None, got {type(value)!r}."
                )
            terms[f.name] = value
    return ObservationGroupCfg(terms=terms, **group_kwargs)


def observations_to_dict(observations: Any) -> dict[str, ObservationGroupCfg]:
    """Convert an ``ObservationsCfg`` into ``dict[str, ObservationGroupCfg]``."""
    if observations is None:
        return {}
    result: dict[str, ObservationGroupCfg] = {}
    for f in dataclasses.fields(observations):
        group = getattr(observations, f.name)
        if group is None:
            continue
        result[f.name] = _obs_group_to_mjlab(group)
    return result
