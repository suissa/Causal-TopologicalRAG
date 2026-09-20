"""Controlled eroded-path rescue experiment with adversarial baselines."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ctrag import (
    CausalProvenance, CausalTopology, CTRetriever, DynamicTerrain, Edge, EdgeKind,
    MemoryNode, QueryMode, TerrainAwareRetriever, TerrainConfig,
)


@dataclass(frozen=True)
class ErodedPathConfig:
    failure_repetitions: int = 20
    provider_successes: int = 8
    provider_failures: int = 2
    manual_successes: int = 1
    manual_failures: int = 9
    script_successes: int = 2
    script_failures: int = 8
    cache_successes: int = 1
    cache_failures: int = 3
    global_successes: int = 9
    global_failures: int = 1
    decay_elapsed: float = 20.0


def _edge(source: str, target: str) -> Edge:
    return Edge(source, target, EdgeKind.CAUSAL, provenance=CausalProvenance.EXECUTION)


def _build() -> tuple[CausalTopology, dict[str, Edge]]:
    t = CausalTopology()
    now = datetime(2026, 9, 20, tzinfo=timezone.utc)
    specs = {
        "retry": ("payment retry", "retry", 0),
        "timeout": ("payment path", "timeout", 0),
        "human": ("payment outcome", "human_intervention", 0),
        "provider": ("payment recovery route", "fallback", 30),
        "provider_recovered": ("payment recovered", "recovered", 30),
        "provider_failed": ("payment failed", "failed", 29),
        "manual": ("payment recovery route", "manual_patch", 5),
        "manual_recovered": ("payment recovered", "recovered", 5),
        "manual_failed": ("payment failed", "failed", 4),
        "script": ("payment recovery route", "script_patch", 7),
        "script_recovered": ("payment recovered", "recovered", 7),
        "script_failed": ("payment failed", "failed", 6),
        "cache": ("payment recovery route", "cache_reset", 9),
        "cache_recovered": ("payment recovered", "recovered", 9),
        "cache_failed": ("payment failed", "failed", 8),
        "other_root": ("different incident", "failure", 1),
        "global_fix": ("global recovery route", "fallback", 2),
        "global_recovered": ("payment recovered", "recovered", 2),
        "global_failed": ("payment failed", "failed", 2),
    }
    for node_id, (textv, status, days) in specs.items():
        t.add_node(MemoryNode(id=node_id,text=textv,timestamp=now-timedelta(days=days),metadata={"status":status}))
    edges={
        "failure_1":_edge("retry","timeout"),"failure_2":_edge("timeout","human"),
        "provider_1":_edge("retry","provider"),"provider_success":_edge("provider","provider_recovered"),"provider_failure":_edge("provider","provider_failed"),
        "manual_1":_edge("retry","manual"),"manual_success":_edge("manual","manual_recovered"),"manual_failure":_edge("manual","manual_failed"),
        "script_1":_edge("retry","script"),"script_success":_edge("script","script_recovered"),"script_failure":_edge("script","script_failed"),
        "cache_1":_edge("retry","cache"),"cache_success":_edge("cache","cache_recovered"),"cache_failure":_edge("cache","cache_failed"),
        "global_1":_edge("other_root","global_fix"),"global_success":_edge("global_fix","global_recovered"),"global_failure":_edge("global_fix","global_failed"),
    }
    for e in edges.values(): t.add_edge(e)
    return t,edges


def _observe_branch(terrain:DynamicTerrain, first:Edge, success:Edge, failure:Edge, s:int, f:int)->None:
    for _ in range(s):
        terrain.reinforce(first); terrain.reinforce(success)
    for _ in range(f):
        terrain.reinforce(first); terrain.reinforce(failure)


def _path_score(terrain:DynamicTerrain, edges:tuple[Edge,...])->float:
    p=1.0
    for e in edges:p*=terrain.influence(e)
    return p ** (1/len(edges))


def _reachable_recovered(t:CausalTopology,anchor:str)->list[str]:
    out=[]
    for node_id,node in t.nodes.items():
        if str(node.metadata.get("status","")).lower()!="recovered": continue
        if t.causal_path_evidence(anchor,node_id,direction="out",max_hops=4) is not None: out.append(node_id)
    return sorted(out)


def _raw_success_rates(cfg:ErodedPathConfig)->dict[str,float]:
    return {
        "provider_recovered": cfg.provider_successes/(cfg.provider_successes+cfg.provider_failures),
        "manual_recovered": cfg.manual_successes/(cfg.manual_successes+cfg.manual_failures),
        "script_recovered": cfg.script_successes/(cfg.script_successes+cfg.script_failures),
        "cache_recovered": cfg.cache_successes/(cfg.cache_successes+cfg.cache_failures),
        "global_recovered": cfg.global_successes/(cfg.global_successes+cfg.global_failures),
    }


def _precision_at_k(ranking:list[str],gold:set[str],k:int)->float:
    chosen=ranking[:k]
    return 0.0 if not chosen else sum(x in gold for x in chosen)/len(chosen)


def run(config:ErodedPathConfig, output:Path)->dict[str,object]:
    output.mkdir(parents=True,exist_ok=True)
    topology,edges=_build()
    terrain=DynamicTerrain(topology,config=TerrainConfig(reinforcement_step=.5,decay_rate=.15,maximum_influence=30.0))

    _observe_branch(terrain,edges["provider_1"],edges["provider_success"],edges["provider_failure"],config.provider_successes,config.provider_failures)
    _observe_branch(terrain,edges["manual_1"],edges["manual_success"],edges["manual_failure"],config.manual_successes,config.manual_failures)
    _observe_branch(terrain,edges["script_1"],edges["script_success"],edges["script_failure"],config.script_successes,config.script_failures)
    _observe_branch(terrain,edges["cache_1"],edges["cache_success"],edges["cache_failure"],config.cache_successes,config.cache_failures)
    _observe_branch(terrain,edges["global_1"],edges["global_success"],edges["global_failure"],config.global_successes,config.global_failures)

    terrain.decay(config.decay_elapsed)
    for _ in range(config.failure_repetitions):
        terrain.reinforce(edges["failure_1"]);terrain.reinforce(edges["failure_2"])

    reachable=_reachable_recovered(topology,"retry")
    recency=sorted(reachable,key=lambda n:(-topology.nodes[n].timestamp.timestamp(),n))
    raw_rates=_raw_success_rates(config)
    raw_success=sorted(raw_rates,key=lambda n:(-raw_rates[n],n))

    terrain_only=sorted(
        [
            ("human",_path_score(terrain,(edges["failure_1"],edges["failure_2"]))),
            ("provider_recovered",_path_score(terrain,(edges["provider_1"],edges["provider_success"]))),
            ("manual_recovered",_path_score(terrain,(edges["manual_1"],edges["manual_success"]))),
            ("script_recovered",_path_score(terrain,(edges["script_1"],edges["script_success"]))),
            ("cache_recovered",_path_score(terrain,(edges["cache_1"],edges["cache_success"]))),
        ],
        key=lambda x:(-x[1],x[0])
    )
    terrain_rank=[x[0] for x in terrain_only]

    result=TerrainAwareRetriever(CTRetriever(topology),terrain).search_staged(
        "payment recovery path",mode=QueryMode.RECOVERY,anchor_ids=["retry"],k=20
    )
    recovered_ids={n for n,node in topology.nodes.items() if node.metadata.get("status")=="recovered"}
    ctrag=[h.node.id for h in result.hits if h.node.id in recovered_ids]
    components={
        h.node.id:{
            "score":h.score,
            "historical_success_rate":h.components.get("historical_success_rate",0.0),
            "historical_support_n":h.components.get("historical_support_n",0.0),
            "historical_wilson_lower_95":h.components.get("historical_wilson_lower_95",0.0),
            "recovery_causally_reachable":h.components.get("recovery_causally_reachable",0.0),
        }
        for h in result.hits if h.node.id in recovered_ids
    }

    gold={"provider_recovered"}
    false_rescue={}
    for k in (1,2,3,4):
        false_rescue[f"false_rescue_at_{k}"]=1.0-_precision_at_k(ctrag,gold,k)

    report={
        "schema_version":3,
        "claim_scope":"controlled eroded-path rescue discrimination mechanism",
        "rankings":{
            "terrain_only":terrain_rank,
            "recency_recovered_only":recency,
            "raw_success_global":raw_success,
            "structural_recovered_unranked":reachable,
            "ctrag_recovery":ctrag,
        },
        "raw_success_rates":raw_rates,
        "ctrag_components":components,
        "metrics":{
            "terrain_only_precision_at_1":_precision_at_k(terrain_rank,gold,1),
            "recency_precision_at_1":_precision_at_k(recency,gold,1),
            "raw_success_precision_at_1":_precision_at_k(raw_success,gold,1),
            "structural_precision_at_4":_precision_at_k(reachable,gold,4),
            "ctrag_precision_at_1":_precision_at_k(ctrag,gold,1),
            **false_rescue,
        },
        "outcomes":{
            "raw_success_prefers_unreachable_path": raw_success[0]=="global_recovered",
            "global_high_success_is_unreachable_from_retry": "global_recovered" not in reachable,
            "ctrag_excludes_unreachable_high_success_path": "global_recovered" not in ctrag,
            "ctrag_prefers_provider": bool(ctrag and ctrag[0]=="provider_recovered"),
            "recency_prefers_distractor": bool(recency and recency[0]!="provider_recovered"),
        },
    }
    rows=[{"candidate":n,"raw_success_rate":raw_rates[n],"reachable_from_retry":n in reachable,"ctrag_rank":(ctrag.index(n)+1 if n in ctrag else None)} for n in raw_success]
    (output/"eroded-path-rescue.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (output/"eroded-path-rescue.csv").open("w",encoding="utf-8",newline="") as s:
        w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return report


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("benchmark-results/eroded-path-rescue"))
    a=p.parse_args(argv);r=run(ErodedPathConfig(),a.output);print(json.dumps({"rankings":r["rankings"],"metrics":r["metrics"],"outcomes":r["outcomes"]},sort_keys=True));return 0


if __name__=="__main__": raise SystemExit(main())
