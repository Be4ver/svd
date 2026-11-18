"""Host-check the device schedule and stress the two-mailbox protocol model.

This tests ordering/liveness, not device DMA completion or cache visibility.
Real NPU one-sweep/serial comparison is required in addition.
"""
from collections import deque
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from run_sweep import block_rounds


class CoreLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        source = root / 'schedule.cpp'
        source.write_text(r'''
#include <iostream>
#define __aicore__
#include "core_local_schedule.hpp"
int main(int argc, char** argv) {
    using namespace blocked_jacobi;
    int nb=std::stoi(argv[1]), batch=std::stoi(argv[2]), groups=std::stoi(argv[3]);
    int jobs=batch*(nb/2);
    if(argc>4) {
      for(int w=0; w<2*groups; ++w) {
        int j=WorkerJob(w,0,groups), r=0;
        if(j>=jobs) continue;
        while(r<3*(nb-1)) {
          std::cout<<r<<' '<<j<<' '<<w<<'\n';
          AdvanceWorkerJob(j,r,w,groups,jobs);
        }
      }
      return 0;
    }
    for(int w=0; w<2*groups; ++w)
      for(int r=0; r<nb-1; ++r)
        for(int k=0; WorkerJob(w,k,groups)<jobs; ++k) {
          int j=WorkerJob(w,k,groups), m=j/(nb/2), p=j%(nb/2);
          if(OwnerWorker(j,groups)!=w) return 2;
          std::cout<<r<<' '<<j<<' '<<w<<' '
                   <<m*nb+PairBlock(p,0,r,nb)<<' '
                   <<m*nb+PairBlock(p,1,r,nb)<<'\n';
        }
}
''')
        cls.exe = root / 'schedule'
        subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(Path(__file__).resolve().parents[1]),
                        str(source), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def tasks(self, nb, batch, groups):
        run = subprocess.run([str(self.exe), str(nb), str(batch), str(groups)],
                             check=True, capture_output=True, text=True)
        return [tuple(map(int, row.split())) for row in run.stdout.splitlines()]

    def test_device_formula_matches_independent_tournament(self):
        for nb, batch, groups in ((2,1,20),(4,3,2),(6,5,3),(64,1,20),(8,7,1),(10,3,4)):
            with self.subTest(nb=nb,batch=batch,groups=groups):
                tasks = self.tasks(nb,batch,groups)
                actual = {(r,j,a,b) for r,j,w,a,b in tasks}
                expected = {(r,m*(nb//2)+p,m*nb+a,m*nb+b)
                            for r,pairs in enumerate(block_rounds(nb))
                            for m in range(batch) for p,(a,b) in enumerate(pairs)}
                self.assertEqual(actual, expected)
                self.assertEqual(len(tasks), len(expected))

    def test_static_consumer_cursor_matches_producer_across_sweeps(self):
        for nb, batch, groups in ((2,1,20),(4,3,2),(6,5,3),(64,1,20),(8,7,1),(10,3,4)):
            with self.subTest(nb=nb, batch=batch, groups=groups):
                run = subprocess.run([str(self.exe), str(nb), str(batch), str(groups), 'cursor'],
                                     check=True, capture_output=True, text=True)
                actual = [tuple(map(int, row.split())) for row in run.stdout.splitlines()]
                expected = [(r+s*(nb-1),j,w) for s in range(3)
                            for r,j,w,a,b in self.tasks(nb,batch,groups)]
                self.assertEqual(sorted(actual), sorted(expected))

    def test_two_slots_progress_and_never_read_unpublished_block(self):
        for nb,batch,groups in ((2,1,20),(4,3,2),(6,5,3),(64,1,20),(8,7,1),(10,3,4)):
            # Continue versions and mailbox sequence numbers across sweeps.
            # An old epoch/token must never satisfy a reused slot.
            tasks = [(r+s*(nb-1),j,w,a,b)
                     for s in range(3) for r,j,w,a,b in self.tasks(nb,batch,groups)]
            for seed in range(4):
                rng = random.Random(seed)
                queues = [deque() for _ in range(groups*2)]
                for task in tasks:
                    queues[task[2]].append(task)
                version = [0]*(batch*nb)
                phase = [0]*len(queues)  # dep, pack ready, gram done, R ready
                sequence = [0]*len(queues)
                acknowledged = [0]*len(queues)
                completed = 0
                while completed < len(tasks):
                    ready = []
                    for w,q in enumerate(queues):
                        if not q:
                            continue
                        r,j,_,a,b = q[0]
                        if phase[w] == 0:
                            self.assertLessEqual(version[a], r)
                            self.assertLessEqual(version[b], r)
                            if version[a] == version[b] == r:
                                ready.append((w,1))
                        else:
                            ready.append((w,phase[w]+1))
                    self.assertTrue(ready, 'mailbox scheduler deadlocked')
                    w,next_phase = rng.choice(ready)
                    if next_phase == 4:
                        r,j,_,a,b = queues[w].popleft()
                        self.assertEqual((version[a],version[b]), (r,r))
                        version[a] = version[b] = r+1
                        self.assertEqual(sequence[w], acknowledged[w]+1)
                        acknowledged[w] = sequence[w]
                        phase[w] = 0
                        completed += 1
                    else:
                        if next_phase in (1,3):
                            self.assertEqual(sequence[w], acknowledged[w])
                            sequence[w] += 1
                        elif next_phase == 2:
                            self.assertEqual(sequence[w], acknowledged[w]+1)
                            acknowledged[w] = sequence[w]
                        phase[w] = next_phase
                self.assertEqual(version, [3*(nb-1)]*(batch*nb))
                self.assertEqual(sequence, acknowledged)


if __name__ == '__main__':
    unittest.main()
