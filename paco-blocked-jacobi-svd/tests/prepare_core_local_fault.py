"""Create a separate build source with an impossible dependency to test unwind.

The production kernel and its timeout remain unchanged. Build this copy in a
separate directory; it must report the dependency watchdog error, not succeed.
"""
from pathlib import Path
import argparse
import shutil

project = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifacts', type=Path, default=project/'artifacts/core-local/fault-source')
root = p.parse_args().artifacts.resolve()
root.mkdir(parents=True, exist_ok=True)
for pattern in ('*.hpp', '*.cpp', 'CMakeLists.txt'):
    for source in project.glob(pattern):
        shutil.copy2(source, root / source.name)
for name in ('core_local_sync.hpp', 'core_local_pipeline.hpp'):
    source = root / name
    text = source.read_text().replace('2000000000LL', '5000000LL')
    if name == 'core_local_pipeline.hpp':
        old = 'signals.Wait(signals.Version(a), epoch)'
        assert text.count(old) == 1
        text = text.replace(old,
            'signals.Wait(signals.Version(a), epoch + (worker == 0 && sweep == 0 && round == 0 && ordinal == 0))')
    source.write_text(text)
main = root / 'main.cpp'
main.write_text(main.read_text().replace('ACL(aclrtCreateContext(&context, device));',
    'ACL(aclrtSetOpExecuteTimeOut(5)); ACL(aclrtCreateContext(&context, device));'))
print(root)
