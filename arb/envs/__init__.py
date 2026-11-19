"""Environment registry."""

from __future__ import annotations

from arb.envs.base import Environment
from arb.envs.tickets import TicketsEnv
from arb.envs.migration import MigrationEnv
from arb.envs.inventory import InventoryEnv
from arb.envs.procurement import ProcurementEnv

SINGLE_AGENT_FAMILIES: tuple[str, ...] = ("tickets", "migration", "inventory")
MULTI_AGENT_FAMILIES: tuple[str, ...] = ("procurement",)

FAMILIES: dict[str, type[Environment]] = {
    "tickets": TicketsEnv,
    "migration": MigrationEnv,
    "inventory": InventoryEnv,
    "procurement": ProcurementEnv,
}


def make_env(family: str) -> Environment:
    return FAMILIES[family]()


def all_envs(track: str | None = None) -> list[Environment]:
    out = []
    for fam in SINGLE_AGENT_FAMILIES + MULTI_AGENT_FAMILIES:
        env = make_env(fam)
        if track is None or env.track == track:
            out.append(env)
    return out


__all__ = [
    "Environment",
    "SINGLE_AGENT_FAMILIES",
    "MULTI_AGENT_FAMILIES",
    "FAMILIES",
    "make_env",
    "all_envs",
]
