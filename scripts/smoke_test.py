#!/usr/bin/env python3
"""
Smoke test for A1 synthetic guard.
"""

import sys
sys.path.insert(0, "src")


def check_synthetic_guard_acceptance() -> bool:
    """ALLOW_REAL_PHI=false + is_synthetic=True + known ID must pass."""
    from utils.synthetic import enforce_synthetic_guard, SyntheticGuardError
    try:
        enforce_synthetic_guard("P001", is_synthetic=True)
        print("✅ A1 OK: guard accepts known synthetic P001")
        return True
    except SyntheticGuardError as e:
        print(f"❌ A1 FAILED: guard rejected known synthetic P001: {e}")
        return False


def check_guard_rejects_lying_caller() -> bool:
    """is_synthetic=True with an unknown ID must be refused (trust-but-verify)."""
    from utils.synthetic import enforce_synthetic_guard, SyntheticGuardError
    try:
        enforce_synthetic_guard("REAL_1", is_synthetic=True)
        print("❌ A1 FAILED: guard trusted caller's is_synthetic=True for unknown ID")
        return False
    except SyntheticGuardError as e:
        print(f"✅ A1 OK (trust-but-verify): {e}")
        return True


def main() -> int:
    ok = True
    ok &= check_synthetic_guard_acceptance()
    ok &= check_guard_rejects_lying_caller()
    print()
    if ok:
        print("All smoke tests passed")
        return 0
    else:
        print("Some smoke tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())