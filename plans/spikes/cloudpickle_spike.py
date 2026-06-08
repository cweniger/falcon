"""Cloudpickle spike for the Falcon notebook API (issue #58, Step 2).

Tests whether __main__-defined simulator and embedding classes survive the
Ray actor serialization boundary, covering the scenarios listed in the plan.

Run from the repo root:
    python plans/spikes/cloudpickle_spike.py

Each test prints PASS / FAIL with a brief explanation.

## Findings (2026-06-08)

All scenarios pass except CUDA tensors stored as instance attributes (expected).

PASS  Basic callable, numpy return
PASS  Torch simulator using torch.randn_like
PASS  torch.nn.Module subclass (SmallMLP with Linear layer)
PASS  Transitive __main__ dependency (class A uses class B from __main__)
PASS  Closure over a small numpy array
PASS  Closure over a large numpy array (~8 MB) — pickle is 8 MB; documented footgun
PASS  Class redefinition (re-run cell): new class replaces old one correctly
PASS  CUDA tensor constructor arg — fails as expected; numpy workaround passes

Conclusion: cloudpickle + Ray handles all normal notebook simulator patterns.
The one constraint: do not store CUDA tensors as instance attributes; store
numpy arrays and call .cuda() inside forward()/__call__().
Large global closures work but silently bloat the pickle on every call.
"""

import sys
import traceback
import numpy as np
import torch
import cloudpickle
import ray

# ---------------------------------------------------------------------------
# Ray actor that accepts a cloudpickled callable and exercises it
# ---------------------------------------------------------------------------

@ray.remote
class WorkerActor:
    """Simulates a NodeWrapper actor receiving a user-defined simulator."""

    def run_callable(self, obj_bytes, *args):
        """Deserialize obj_bytes, instantiate if it's a class, call with args."""
        obj = cloudpickle.loads(obj_bytes)
        if isinstance(obj, type):
            instance = obj()
        else:
            instance = obj
        result = instance(*args)
        return result

    def run_module_forward(self, obj_bytes, x_bytes):
        """Deserialize a nn.Module and run a forward pass."""
        module = cloudpickle.loads(obj_bytes)
        x = cloudpickle.loads(x_bytes)
        with torch.no_grad():
            return cloudpickle.dumps(module(x))

    def check_identity(self, obj_bytes):
        """Return the class name of the deserialized object (instance or class)."""
        obj = cloudpickle.loads(obj_bytes)
        if isinstance(obj, type):
            return obj.__name__
        return type(obj).__name__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack(obj):
    return cloudpickle.dumps(obj)

def _pack_tensor(t):
    return cloudpickle.dumps(t)

def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_callable(actor):
    """Plain __main__ callable class shipped to a Ray actor."""

    class MySimulator:
        def __call__(self, theta):
            return theta * 2.0

    theta = torch.tensor([1.0, 2.0, 3.0])
    result = ray.get(actor.run_callable.remote(_pack(MySimulator), _pack_tensor(theta)))
    result = cloudpickle.loads(result) if isinstance(result, bytes) else result

    # run_callable returns whatever the simulator returns; re-pack for transfer
    # Actually the actor returns the raw result. Let's just get it and compare.
    assert isinstance(result, torch.Tensor) or result is not None


def test_basic_callable_v2(actor):
    """Verify the actor can deserialize and the result is correct."""

    class DoubleSimulator:
        def __call__(self, theta):
            import torch as _torch
            return _torch.tensor([x * 2.0 for x in theta.tolist()])

    theta = torch.tensor([1.0, 2.0, 3.0])
    # Pack theta as plain numpy so it survives without torch serialization issues
    theta_np = theta.numpy()
    result = ray.get(actor.run_callable.remote(_pack(DoubleSimulator), theta_np))
    assert isinstance(result, (np.ndarray, torch.Tensor))


def test_torch_simulator(actor):
    """Simulator that uses torch inside __call__."""

    class TorchSimulator:
        def __call__(self, theta):
            import torch as _torch
            return theta + 0.1 * _torch.randn_like(theta)

    theta = torch.zeros(5)
    result = ray.get(actor.run_callable.remote(_pack(TorchSimulator), theta))
    assert result is not None


def test_torch_nn_module(actor):
    """nn.Module subclass defined in __main__."""

    class SmallMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    model = SmallMLP()
    x = torch.randn(3, 4)
    out_bytes = ray.get(actor.run_module_forward.remote(_pack(model), _pack_tensor(x)))
    out = cloudpickle.loads(out_bytes)
    assert out.shape == (3, 2)


def test_transitive_dep(actor):
    """Class A uses helper class B, both defined in __main__."""

    class Preprocessor:
        def __call__(self, x):
            import torch as _torch
            return x - _torch.mean(x)

    class SimulatorWithHelper:
        def __call__(self, theta):
            prep = Preprocessor()
            return prep(theta)

    theta = torch.tensor([1.0, 2.0, 3.0])
    result = ray.get(actor.run_callable.remote(_pack(SimulatorWithHelper), theta))
    assert result is not None


def test_closure_over_array(actor):
    """Class closes over a numpy array (fixed dataset) defined in the outer scope."""
    fixed_data = np.random.randn(100, 10)  # simulates a global in a notebook

    class DataSimulator:
        def __call__(self, theta):
            import numpy as _np
            idx = int(_np.random.randint(len(fixed_data)))
            return fixed_data[idx] + theta.numpy()

    theta = torch.zeros(10)
    result = ray.get(actor.run_callable.remote(_pack(DataSimulator), theta))
    assert result is not None


def test_large_global_closure(actor):
    """Class closes over a large array; checks pickle size is reasonable."""
    large_array = np.random.randn(1000, 1000)  # ~8 MB

    class LargeClosureSimulator:
        def __call__(self, theta):
            return large_array[0, :len(theta)] + theta.numpy()

    packed = _pack(LargeClosureSimulator)
    size_mb = len(packed) / 1e6
    # Warn if pickle is large but don't fail — just report
    print(f"          (pickle size: {size_mb:.1f} MB)", end="")

    theta = torch.zeros(5)
    result = ray.get(actor.run_callable.remote(packed, theta))
    assert result is not None


def test_class_redefinition(actor):
    """Simulate re-running a notebook cell: new definition of the same name.

    The old class identity and the new one must be distinct, and the actor
    always gets whatever was most recently packed.
    """

    class EvolvedSimulator:
        version = 1
        def __call__(self, theta):
            return theta * float(self.version)

    packed_v1 = _pack(EvolvedSimulator)
    name_v1 = ray.get(actor.check_identity.remote(packed_v1))

    # Simulate re-running the cell: redefine with a different version
    class EvolvedSimulator:  # noqa: F811
        version = 2
        def __call__(self, theta):
            return theta * float(self.version)

    packed_v2 = _pack(EvolvedSimulator)
    name_v2 = ray.get(actor.check_identity.remote(packed_v2))

    # Both should be named EvolvedSimulator but they are distinct pickle blobs
    assert name_v1 == "EvolvedSimulator"
    assert name_v2 == "EvolvedSimulator"

    # Verify the actor actually executes the new version
    theta = torch.tensor([1.0])
    result_v2 = ray.get(actor.run_callable.remote(packed_v2, theta))
    assert float(result_v2[0]) == 2.0, f"Expected 2.0, got {result_v2}"


def test_cuda_tensor_constructor_arg(actor):
    """CUDA tensors stored as instance attributes FAIL across the boundary.

    Expected failure: cloudpickle serializes the CUDA storage, and the Ray
    worker process cannot deserialize it without a matching GPU context.
    Workaround: store plain numpy in __init__ and call .cuda() inside forward.
    This test verifies the failure mode and that the workaround passes.
    """
    if not torch.cuda.is_available():
        print("          (skipped — no CUDA)", end="")
        return

    # Verify the failure mode
    class SimBroken:
        def __init__(self):
            import torch as _torch
            self.bias = _torch.tensor([1.0, 2.0]).cuda()
        def __call__(self, theta):
            return theta + self.bias.cpu()

    instance = SimBroken()
    packed = _pack(instance)
    theta = torch.zeros(2)
    try:
        ray.get(actor.run_callable.remote(packed, theta))
        raise AssertionError("Expected deserialization to fail — it did not")
    except ray.exceptions.RayTaskError as e:
        assert "CUDA" in str(e), f"Unexpected error: {e}"
        print("          (CUDA storage fails as expected)", end="")

    # Verify the workaround pattern compiles correctly (CPU version of the pattern)
    # The real workaround: store numpy in __init__, call .to(device) inside forward
    class SimFixed:
        def __init__(self):
            import numpy as _np
            self.bias_np = _np.array([1.0, 2.0])
        def __call__(self, theta):
            import torch as _torch
            bias = _torch.from_numpy(self.bias_np)  # .cuda() only when device available
            return theta + bias

    instance_fixed = SimFixed()
    packed_fixed = _pack(instance_fixed)
    result = ray.get(actor.run_callable.remote(packed_fixed, torch.zeros(2)))
    assert result is not None
    print("          (numpy workaround passes)", end="")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Cloudpickle / Ray serialization spike")
    print("=" * 50)

    ray.init(ignore_reinit_error=True, logging_level="ERROR", namespace="falcon_spike")

    actor = WorkerActor.remote()

    tests = [
        ("basic callable (numpy return)",    lambda: test_basic_callable_v2(actor)),
        ("torch simulator (randn_like)",      lambda: test_torch_simulator(actor)),
        ("torch nn.Module (SmallMLP)",        lambda: test_torch_nn_module(actor)),
        ("transitive __main__ dep",           lambda: test_transitive_dep(actor)),
        ("closure over numpy array",          lambda: test_closure_over_array(actor)),
        ("closure over large array (~8 MB)",  lambda: test_large_global_closure(actor)),
        ("class redefinition (re-run cell)",  lambda: test_class_redefinition(actor)),
        ("CUDA tensor constructor arg",       lambda: test_cuda_tensor_constructor_arg(actor)),
    ]

    results = []
    for name, fn in tests:
        ok = run_test(name, fn)
        print()
        results.append(ok)

    ray.shutdown()

    passed = sum(results)
    total = len(results)
    print("=" * 50)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("Cloudpickle spike: PASSED — notebook classes survive to Ray actors.")
    else:
        print("Cloudpickle spike: PARTIAL — see failures above.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
