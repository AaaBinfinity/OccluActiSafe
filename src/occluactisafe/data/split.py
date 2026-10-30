"""N04 分组优先的四集合划分；保留原目录角色，不随机拆分窗口。"""
from collections import Counter
import hashlib
import json
import math

ROLES = ("train", "model_val", "risk_calibration", "test")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def build_group_split(records, *, seed=20261030, ratios=(0.65, 0.15, 0.10, 0.10)):
    """输入文件/episode元数据，按路口、明确录制ID和基础episode的连通分量划分。

    location_id是原始地理标识，不与地图文件编号建立对应。未知地理/录制身份
    且无基础episode时明确失败；同名不同来源不得由调用方擅自视为同一基础episode。
    原始val/test文件保留分组身份，但不作为项目四集合的训练/校准/测试窗口输入。
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed须为非负整数")
    if len(ratios) != 4 or any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("四种角色比例须为和为1的有限正数")
    records = sorted((dict(r) for r in records), key=lambda r: r["source_file"])
    if not records or len({r["source_file"] for r in records}) != len(records):
        raise ValueError("文件列表为空或有重复文件键")
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(records):
        keys = []
        for field in ("location_id", "recording_id", "base_episode_id"):
            value = row.get(field)
            if value:
                keys.append((field, str(value)))
        if not keys:
            raise ValueError(f"缺少可核验分组身份: {row['source_file']}")
        for key in keys:
            if key in owners:
                a, b = find(i), find(owners[key]); parent[max(a,b)] = min(a,b)
            else:
                owners[key] = i
    members = {}
    for i, row in enumerate(records):
        members.setdefault(find(i), []).append(row["source_file"])
    groups = {i: canonical_hash(files) for i, files in members.items()}
    eligible = {find(i) for i,r in enumerate(records) if r.get("original_split") == "train"}
    if len(eligible) < 4:
        raise ValueError(f"只有{len(eligible)}个合格独立分组，不能形成四集合")
    order = sorted(eligible, key=lambda i: hashlib.sha256(f"{seed}:{groups[i]}".encode()).hexdigest())
    # 至少一个组/角色，剩余组按最大余数法分配；不读取类别或未来监督选组。
    allocation = [1]*4; rest = len(order)-4
    extra = [math.floor(rest*r) for r in ratios]
    for j in sorted(range(4), key=lambda j: (-(rest*ratios[j]-extra[j]),j))[:rest-sum(extra)]:
        extra[j] += 1
    allocation = [a+b for a,b in zip(allocation,extra)]
    assignment = {}; start = 0
    for role, count in zip(ROLES, allocation):
        for i in order[start:start+count]: assignment[i] = role
        start += count
    entries = []
    for i,row in enumerate(records):
        group = find(i)
        entries.append({**row,"group_id":groups[group],"project_role":assignment.get(group,"quarantined"),
                        "project_input_allowed":row.get("original_split")=="train"})
    manifest = {"schema_version":"1.0","protocol":"project_intersection_holdout_v1",
                "seed":seed,"ratios":dict(zip(ROLES,ratios)),"group_count":len(groups),
                "group_role_counts":dict(zip(ROLES,allocation)),"files":entries,
                "original_split_counts":dict(Counter(r.get("original_split","unknown") for r in entries)),
                "note":"原目录划分单列；项目独立集合只从原train留出。原val/test不参与项目拟合。分组不代表已核实原始录制ID。"}
    audit_group_split(manifest)
    manifest["sha256"] = canonical_hash(manifest)
    return manifest


def audit_group_split(manifest):
    seen = {}
    for row in manifest["files"]:
        for field in ("group_id","location_id","recording_id","base_episode_id"):
            value = row.get(field)
            if value:
                key=(field,value)
                if key in seen and seen[key] != row["project_role"]:
                    raise ValueError(f"跨集合泄漏: {field}={value}")
                seen[key]=row["project_role"]
    return {"cross_split_shared_keys":0,"file_count":len(manifest["files"])}
