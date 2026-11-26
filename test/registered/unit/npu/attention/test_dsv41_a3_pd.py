"""CPU byte-copy contracts for A3 V4.1 PD, without NPU/network imports.

Use the production pool descriptors and Ascend/Mooncake address builders.
Only the transport is replaced by a bounds-checked CPU memory copy. This
does not exercise device stream ordering, distributed bootstrap or ACLGraph.
"""

import ast
import ctypes
import enum
import unittest
from types import SimpleNamespace

import numpy as np
import test_dsv41_a3_cache as cache_tests
import torch
from test_dsv41_a3_cache import (
    ROOT,
    _index_pool,
    _load_definitions,
    _pool_classes,
    _single_pool,
)


def _load_methods(relative_path, class_name, names, namespace):
    path = ROOT / relative_path
    cls = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    nodes = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def _manager():
    ns = {"np": np, "enum": enum}
    base = ROOT / "python/sglang/srt/disaggregation"
    _load_definitions(base / "base/conn.py", {"StateType"}, ns)
    _load_definitions(base / "common/utils.py", {"group_concurrent_contiguous"}, ns)
    _load_definitions(base / "utils.py", {"is_elided_entry"}, ns)
    _load_definitions(base / "mooncake/conn.py", {"_uses_flat_entry_layout"}, ns)
    _load_methods(
        "python/sglang/srt/disaggregation/ascend/conn.py",
        "AscendKVManager",
        {"send_kvcache", "get_mla_kv_ptrs_with_pp"},
        ns,
    )
    _load_methods(
        "python/sglang/srt/disaggregation/mooncake/conn.py",
        "MooncakeKVManager",
        {"_send_kvcache_generic"},
        ns,
    )
    cls = type(
        "CpuAscendTransfer",
        (),
        {
            name: ns[name]
            for name in (
                "send_kvcache",
                "get_mla_kv_ptrs_with_pp",
                "_send_kvcache_generic",
            )
        },
    )
    mgr = cls()
    mgr.pp_size = 1
    mgr.kv_args = SimpleNamespace(mla_compression_ratios=[1, 2])
    mgr.is_mla_backend = True
    mgr.is_hybrid_mla_backend = False
    mgr.enable_custom_mem_pool = False
    mgr.max_transfer_batch_indices = 0
    mgr.state_strides_validated = {"decode"}
    mgr._validate_envelope_kv_layout = lambda *args: None
    return mgr, ns["StateType"]


def _owner(size):
    classes = _pool_classes()
    owner = object.__new__(classes["DSV4NPUTokenToKVPool"])
    owner.page_size = 128
    owner.c4_kv_pool = None
    owner.kv_pools = {r: _single_pool(classes, r, size=size // r) for r in (1, 2)}
    indexers = {r: _index_pool(classes, size=size // r) for r in (1, 2)}
    for pool in indexers.values():
        pool.page_size = 64
    owner._indexer_pool = indexers.__getitem__
    owner.swa_kv_pool = _single_pool(classes, compressed=False, size=size)
    owner.compress_state_pools = []
    owner.indexer_compress_state_pools = []
    return owner


def _main_buffers(owner):
    return [
        buf
        for ratio in (1, 2)
        for buf in (
            *owner.kv_pools[ratio].kv_buffer,
            *owner._indexer_pool(ratio).index_k_buffer,
        )
    ]


class TestA3PDTransfer(unittest.TestCase):
    def _copy_engine(self, mgr, src_buffers, dst_buffers):
        copies = []

        def transfer(session, blocks):
            self.assertEqual(session, "decode")
            for src, dst, length in blocks:
                self.assertTrue(
                    any(
                        b.data_ptr() <= src and src + length <= b.data_ptr() + b.nbytes
                        for b in src_buffers
                    )
                )
                self.assertTrue(
                    any(
                        b.data_ptr() <= dst and dst + length <= b.data_ptr() + b.nbytes
                        for b in dst_buffers
                    )
                )
                ctypes.memmove(dst, src, length)
                copies.append((src, dst, length))
            return 0

        mgr._transfer_data = transfer
        return copies

    def test_compressed_kv_and_index_pages_use_receiver_page_ids(self):
        # Distinct capacities and noncontiguous page numbers on the two peers.
        src, dst = _owner(1024), _owner(1536)
        src_bufs, dst_bufs = _main_buffers(src), _main_buffers(dst)
        mgr, _ = _manager()
        src_ptrs, _, strides = src.get_contiguous_buf_infos()
        dst_ptrs, _, dst_strides = dst.get_contiguous_buf_infos()
        self.assertEqual(strides, dst_strides)
        mgr.kv_args = SimpleNamespace(kv_data_ptrs=src_ptrs, kv_item_lens=strides)
        self._copy_engine(mgr, src_bufs, dst_bufs)
        for i, (s, d) in enumerate(zip(src_bufs, dst_bufs)):
            s.view(torch.uint8).reshape(-1).copy_(
                torch.arange(s.nbytes, dtype=torch.int64)
                .add(i * 31)
                .remainder(251)
                .to(torch.uint8)
            )
            d.view(torch.uint8).fill_(255)
        # A chunk boundary may split a contiguous run; each call must preserve
        # the full-page stride of BOTH compression ratios and their index pages.
        for src_ids, dst_ids in (([1, 2], [5, 6]), ([4], [3])):
            self.assertEqual(
                mgr.send_kvcache(
                    "decode", np.array(src_ids), dst_ptrs, np.array(dst_ids), None
                ),
                0,
            )
        for s, d, stride in zip(src_bufs, dst_bufs, strides):
            s_bytes, d_bytes = (
                s.view(torch.uint8).reshape(-1),
                d.view(torch.uint8).reshape(-1),
            )
            expected = torch.full_like(d_bytes, 255)
            for si, di in ((1, 5), (2, 6), (4, 3)):
                expected[di * stride : (di + 1) * stride] = s_bytes[
                    si * stride : (si + 1) * stride
                ]
            torch.testing.assert_close(d_bytes, expected)

    def test_swa_uses_its_own_page_mapping(self):
        src, dst = _owner(1024), _owner(1536)
        mgr, state_type = _manager()
        s, d = src.swa_kv_pool.kv_buffer[0], dst.swa_kv_pool.kv_buffer[0]
        s[2].fill_(12)
        s[3].fill_(13)
        d.fill_(-1)
        self._copy_engine(mgr, [s], [d])
        sp, _, lengths = src.get_state_buf_infos()
        dp, _, dst_lengths = dst.get_state_buf_infos()
        mgr._send_kvcache_generic(
            "decode",
            sp,
            dp,
            lengths,
            np.array([2, 3]),
            np.array([7, 4]),
            None,
            state_type=state_type.SWA,
            dst_item_lens=dst_lengths,
        )
        torch.testing.assert_close(d[7], s[2])
        torch.testing.assert_close(d[4], s[3])
        self.assertTrue(torch.all(d[[0, 1, 2, 3, 5, 6, 8]] == -1))

    def test_pending_pair_moves_to_preallocated_decode_slot(self):
        self._check_pending_pair_transfer(ring_size=2)

    def test_dspark_pending_pair_moves_to_preallocated_decode_slot(self):
        ns = {}
        _load_definitions(
            ROOT / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
            {"get_compress_state_ring_size"},
            ns,
        )
        # User's gamma=6 means seven verify tokens and a 16-row pair ring.
        ring_size = ns["get_compress_state_ring_size"](2, num_draft_tokens=7)
        self.assertEqual(ring_size, 16)
        self._check_pending_pair_transfer(ring_size=ring_size)

    def _check_pending_pair_transfer(self, ring_size):
        ns = {}
        _load_methods(
            "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
            "DeepSeekV4TokenToKVPool",
            {
                "get_request_state_buf_infos",
                "request_state_transfer_indices",
                "clear_request_scoped_state",
            },
            ns,
        )
        owner_cls = type("RequestStateOwner", (), ns)
        # D has extra receive slots beyond its running-request capacity.
        factory = cache_tests.TestA3PairStatePool()
        src_pool, _ = factory._make(3, 4, ring_size=ring_size)
        dst_pool, _ = factory._make(9, 10, ring_size=ring_size)
        src, dst = owner_cls(), owner_cls()
        src.compress_state_pools = [src_pool]
        dst.compress_state_pools = [dst_pool]
        mgr, state_type = _manager()
        s, d = src_pool.kv_score_buffer.kv_score, dst_pool.kv_score_buffer.kv_score
        copies = self._copy_engine(mgr, [s], [d])
        for seq_len in (1, 2, 127, 128, 129, 8195, 8196, 8197):
            with self.subTest(seq_len=seq_len):
                # Distinct ring rows expose wrong bank strides and offsets.
                s.copy_(torch.arange(s.shape[0], dtype=s.dtype)[:, None].expand_as(s))
                d.fill_(99)
                dst.clear_request_scoped_state(7)
                sp, _, lengths = src.get_request_state_buf_infos()
                dp, _, dst_lengths = dst.get_request_state_buf_infos()
                copies.clear()
                mgr._send_kvcache_generic(
                    "decode",
                    sp,
                    dp,
                    lengths,
                    src.request_state_transfer_indices(1, seq_len),
                    dst.request_state_transfer_indices(7, seq_len),
                    None,
                    state_type=state_type.DSV4_REQUEST_STATE,
                    dst_item_lens=dst_lengths,
                )
                ring = dst_pool.state_cache_3d[7]
                if seq_len % 2:
                    torch.testing.assert_close(ring, src_pool.state_cache_3d[1])
                    self.assertEqual(len(copies), 1)
                else:
                    self.assertEqual(copies, [])
                    self.assertTrue(torch.all(ring[:, :512] == 0))
                    self.assertTrue(torch.all(torch.isneginf(ring[:, 512:])))
                self.assertTrue(torch.all(dst_pool.state_cache_3d[6] == 99))


if __name__ == "__main__":
    unittest.main()
