"""Run one CPU suite with temp-only writes, no network and no subprocesses."""
import os
from pathlib import Path
import runpy
import sys
import tempfile

script = Path(sys.argv[1]).resolve()
allowed = {"test_operations.py", "test_inputs.py", "test_science_s1.py", "test_science_s2.py"}
if script.parent != Path(__file__).resolve().parent or script.name not in allowed:
    raise SystemExit("Unknown test suite")

with tempfile.TemporaryDirectory(prefix="augseg-cpu-isolated-") as directory:
    sandbox = Path(directory).resolve()
    os.chdir(sandbox)
    tempfile.tempdir = str(sandbox)
    os.environ.update(TMPDIR=str(sandbox), HOME=str(sandbox),
                      XDG_CACHE_HOME=str(sandbox / "cache"),
                      AUGSEG_TEST_SANDBOX=str(sandbox),
                      PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                      WANDB_MODE="disabled")
    sys.dont_write_bytecode = True

    def writable(path, directory_fd=None):
        if isinstance(path, int):
            path = os.readlink("/proc/self/fd/" + str(path))
        value = Path(os.fsdecode(path))
        if not value.is_absolute():
            base = Path(os.readlink("/proc/self/fd/" + str(directory_fd))) if directory_fd is not None and directory_fd >= 0 else Path.cwd()
            value = base / value
        value = value.resolve()
        if value != sandbox and sandbox not in value.parents:
            raise PermissionError("CPU tests may write only inside their temporary sandbox")

    def audit(event, args):
        if event in ("socket.connect", "socket.bind", "subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"):
            raise PermissionError("CPU tests cannot use network or launch subprocesses")
        if event == "open":
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                writable(path)
        elif event in ("os.remove", "os.rmdir"):
            writable(args[0], args[1])
        elif event in ("os.mkdir", "os.chmod", "os.chown", "os.utime"):
            writable(args[0], args[-1] if isinstance(args[-1], int) else None)
        elif event in ("os.rename", "os.link", "os.symlink"):
            writable(args[0], args[2] if len(args) > 2 else None)
            writable(args[1], args[3] if len(args) > 3 else None)

    sys.addaudithook(audit)
    # Prove the guard rejects a write outside this sandbox before running suites.
    try:
        with (sandbox.parent / (sandbox.name + "-forbidden-write")).open("w"):
            pass
    except PermissionError:
        pass
    else:
        raise AssertionError("Isolation guard did not block an external write")
    sys.argv = [str(script)]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code:
            raise
    print("ISOLATION_PASS", script.name, "temp-only writes; network/subprocess blocked")
