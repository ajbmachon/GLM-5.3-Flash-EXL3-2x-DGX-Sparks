#!/usr/bin/env python3
"""HEAD_GID / WORKER_GID = auto: the real preflight resolves each rank's RoCE v2
IPv4 GID index from a fake sysfs, or refuses when the rank's HCAs disagree."""

import re
from pathlib import Path
import subprocess

import pytest

from test_nccl_multi_hca import HCAS, STUBS, ROOT

LINK_LOCAL = "fe80:0000:0000:0000:32c5:99ff:fe40:c420"
ZERO = "0000:0000:0000:0000:0000:0000:0000:0000"


def ipv4_gid(last_octet):
    return f"0000:0000:0000:0000:0000:ffff:c0a8:64{last_octet:02x}"


def write_table(port, rows):
    (port / "gids").mkdir(parents=True, exist_ok=True)
    (port / "gid_attrs/types").mkdir(parents=True, exist_ok=True)
    for index in range(16):
        gid, kind = rows.get(index, (ZERO, ""))
        (port / "gids" / str(index)).write_text(gid + "\n")
        (port / "gid_attrs/types" / str(index)).write_text(kind + "\n")


def real_table(v2_index, octet):
    # The GB10 CX7 layout seen on 2026-09-15: link-local v1/v2 at 0/1, the
    # IPv4-mapped entry as RoCE v1 one index below its RoCE v2 twin.
    return {
        0: (LINK_LOCAL, "IB/RoCE v1"), 1: (LINK_LOCAL, "RoCE v2"),
        v2_index - 1: (ipv4_gid(octet), "IB/RoCE v1"),
        v2_index: (ipv4_gid(octet), "RoCE v2"),
    }


def run_preflight(tmp_path, tables, head_gid="auto", worker_gid="auto"):
    source = (ROOT / "start.sh").read_text()
    begin = source.index("preflight() {")
    end = source.index("\n}\n", begin) + 3
    # preflight calls the memory guard; load it too so the body runs unchanged.
    mem_begin = source.index("# GLM53 preflight memory guard (begin)")
    mem_end = source.index("# GLM53 preflight memory guard (end)")
    source = source[:begin] + source[mem_begin:mem_end] + source[begin:]
    end += mem_end - mem_begin
    begin = source.index("read_meminfo_kib() {")
    for node in HCAS:
        for hca in HCAS[node]:
            write_table(tmp_path / "sysfs" / node / hca / "ports/1", tables[node][hca])
    for name in ("overlay/patch_ablit.py", "overlay/ablit_runtime.py", "ablit/LAYER_MAP.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    placeholder = tmp_path / "overlay-placeholder"
    placeholder.touch()
    env = {
        "PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": str(tmp_path),
        "USER": "fake-user", "FAKE_SYSFS": str(tmp_path / "sysfs"),
        "FAKE_NODE": "head", "HEAD_IP": "192.0.2.1", "WORKER_SSH": "fake-worker",
        "HEAD_CX7_IB": ",".join(HCAS["head"]), "WORKER_CX7_IB": ",".join(HCAS["worker"]),
        "HEAD_GID": head_gid, "WORKER_GID": worker_gid,
        "TP": "2", "NNODES": "2", "CONTAINER_WORKER": "fake-worker",
        "GPU_MEM_UTIL": "0.01", "GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB": "0",
        "PORT": "8888", "MASTER_PORT": "29521", "SCRIPT_DIR": str(tmp_path),
        "ABLIT": "0", "HF_CACHE_DIR": str(tmp_path / "head-cache"),
        "WORKER_HOME": str(tmp_path), "WORKER_CACHE_DIR": str(tmp_path / "worker-cache"),
    }
    # Every overlay path preflight checks gets a placeholder; derive the list
    # from the exercised body so a new overlay cannot silently break this harness.
    overlay_keys = set(re.findall(r"\b([A-Z0-9_]+_PATCH_HOST)\b", source[begin:end])) | {"EXL3_OVERLAY_HOST"}
    for key in sorted(overlay_keys):
        env[key] = str(placeholder)
    # Echo the resolved indices so the test asserts on what the launch would use.
    script = STUBS + source[begin:end] + "\npreflight\nprintf 'RESOLVED %s %s\\n' \"$HEAD_GID\" \"$WORKER_GID\"\n"
    return subprocess.run(["bash", "-c", script], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=15)


def resolved(result):
    for line in result.stdout.splitlines():
        if line.startswith("RESOLVED "):
            return tuple(line.split()[1:])
    return None


def test_auto_picks_the_roce_v2_ipv4_index_per_rank(tmp_path):
    # Head at 3 on both rails, worker at 4 on both rails: the ranks differ and
    # a RoCE v1 twin sits one index below each. auto must land on 3 and 4.
    tables = {
        "head": {hca: real_table(3, 0x0a) for hca in HCAS["head"]},
        "worker": {hca: real_table(4, 0x0b) for hca in HCAS["worker"]},
    }
    result = run_preflight(tmp_path, tables)
    assert result.returncode == 0, result.stderr
    assert resolved(result) == ("3", "4"), result.stdout + result.stderr


def test_auto_follows_a_renumbered_worker_table(tmp_path):
    # The 2026-09-15 incident: after a head reboot the worker's RoCE v2 IPv4
    # entry moved from index 4 to 3 while .env still pinned 4.
    tables = {
        "head": {hca: real_table(3, 0x0a) for hca in HCAS["head"]},
        "worker": {hca: real_table(3, 0x0b) for hca in HCAS["worker"]},
    }
    pinned = run_preflight(tmp_path, tables, head_gid="3", worker_gid="4")
    assert pinned.returncode != 0, "a stale pinned index must still be refused"
    auto = run_preflight(tmp_path, tables)
    assert auto.returncode == 0, auto.stderr
    assert resolved(auto) == ("3", "3")


def test_auto_refuses_when_a_ranks_rails_disagree(tmp_path):
    # NCCL applies one index per rank, so two rails at different indices are
    # not launchable; auto must fail loudly instead of picking one rail.
    head_a, head_b = HCAS["head"]
    tables = {
        "head": {head_a: real_table(3, 0x0a), head_b: real_table(5, 0x0a)},
        "worker": {hca: real_table(3, 0x0b) for hca in HCAS["worker"]},
    }
    result = run_preflight(tmp_path, tables)
    assert result.returncode != 0
    assert "head" in result.stderr and head_b in result.stderr, result.stderr


def test_auto_ignores_a_v1_only_ipv4_entry(tmp_path):
    # A table with the IPv4 entry only as RoCE v1 has no usable index.
    tables = {
        "head": {hca: {0: (LINK_LOCAL, "IB/RoCE v1"), 2: (ipv4_gid(0x0a), "IB/RoCE v1")}
                 for hca in HCAS["head"]},
        "worker": {hca: real_table(3, 0x0b) for hca in HCAS["worker"]},
    }
    result = run_preflight(tmp_path, tables)
    assert result.returncode != 0
    assert "RoCE v2" in result.stderr, result.stderr


def test_literal_indices_still_work_unchanged(tmp_path):
    tables = {
        "head": {hca: real_table(3, 0x0a) for hca in HCAS["head"]},
        "worker": {hca: real_table(4, 0x0b) for hca in HCAS["worker"]},
    }
    result = run_preflight(tmp_path, tables, head_gid="3", worker_gid="4")
    assert result.returncode == 0, result.stderr
    assert resolved(result) == ("3", "4")
