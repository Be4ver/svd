"""Check the proposed static task graph before changing device synchronization."""
import unittest

from run_sweep import block_rounds


def task_graph(blocks, cores):
    previous = [None] * blocks
    graph = []
    for round_id, pairs in enumerate(block_rounds(blocks)):
        for pair_id, (a, b) in enumerate(pairs):
            task = len(graph)
            dependencies = {p for p in (previous[a], previous[b]) if p is not None}
            graph.append((round_id, pair_id % cores, a, b, dependencies))
            previous[a] = previous[b] = task
    return graph


class AsyncDependenciesTests(unittest.TestCase):
    def test_dependencies_cover_each_block_and_are_acyclic(self):
        for blocks, cores in ((2, 1), (4, 2), (6, 3), (64, 20)):
            graph = task_graph(blocks, cores)
            last = [None] * blocks
            pairs = set()
            for task, (round_id, owner, a, b, deps) in enumerate(graph):
                self.assertEqual(deps, {p for p in (last[a], last[b]) if p is not None})
                self.assertTrue(all(p < task and graph[p][0] < round_id for p in deps))
                pairs.add(tuple(sorted((a, b))))
                last[a] = last[b] = task
            self.assertEqual(len(graph), blocks * (blocks - 1) // 2)
            self.assertEqual(len(pairs), len(graph))

    def test_static_assignment_requires_cross_core_handoffs(self):
        graph = task_graph(64, 20)
        handoffs = sum(graph[p][1] != owner for _, owner, _, _, deps in graph for p in deps)
        self.assertGreater(handoffs, 0)


if __name__ == '__main__':
    unittest.main()
