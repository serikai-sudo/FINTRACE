"""Pydantic v2 data models. Placeholder module.

TODO:
    - Every model uses ``model_config = ConfigDict(extra="forbid")``.
    - Money fields are ``Decimal``; timestamps are ISO-8601 with UTC offset.
    - ``PublicEpisode`` (agent-visible) vs ``PrivateEpisode`` (truth + provenance, evaluator-only).
"""

__all__: tuple[str, ...] = ()
