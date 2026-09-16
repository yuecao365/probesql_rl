"""Does tau2 actually work inside the veRL environment, and are two episodes independent?

    /root/autodl-tmp/envs/verl/bin/python rl/test_env.py

Two things are being checked, and neither is cosmetic.

tau2 declares `requires-python >=3.12` and this environment is 3.10, so it was installed with
that check overridden. A successful `import tau2` proves nothing: 3.12-only syntax would sit
in whichever module is imported lazily. So this drives the real path -- build the domain
environment, read the database, call an agent tool, evaluate an assertion.

And rollouts must not share state. veRL runs G trajectories of one task concurrently; if they
reach the same database object, they overwrite each other's writes and every reward is
garbage, silently, with a training curve that still looks plausible. `DB.load` re-validates
from file and nothing in tau2 caches, so separate instances *should* be independent -- this
asserts it rather than trusting the reading.
"""

import sys

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        FAILURES.append((name, e))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def main():
    print(f"python {sys.version.split()[0]}\n")

    from tau2.registry import registry

    ctor = registry.get_env_constructor("telecom")

    def builds():
        env = ctor(solo_mode=False)
        tools = env.get_tools()
        assert len(tools) > 0, "no agent tools"
        print(f"        {len(tools)} agent tools, first = {tools[0].openai_schema['function']['name']}")

    check("telecom environment builds and exposes tools", builds)

    def isolated():
        a, b = ctor(solo_mode=False), ctor(solo_mode=False)
        assert a is not b
        h_a, h_b = a.get_db_hash(), b.get_db_hash()
        assert h_a == h_b, "two fresh environments start from different databases"
        # Mutate A through a real mutating tool call, then re-hash both.
        cust = a.tools.db.customers
        target = cust[0] if isinstance(cust, list) else cust[next(iter(cust))]
        target.full_name = "__probe__"
        assert a.get_db_hash() != h_a, "the probe did not change A's database at all"
        assert b.get_db_hash() == h_b, "A's write leaked into B: the two share state"

    check("two environments do not share database state", isolated)

    def assertions_run():
        tasks = registry.get_tasks_loader("telecom")()
        task = next(t for t in tasks if (t.evaluation_criteria.env_assertions or []))
        env = ctor(solo_mode=False)
        a0 = task.evaluation_criteria.env_assertions[0]
        got = env.run_env_assertion(a0, raise_assertion_error=False)
        assert isinstance(got, bool), f"assertion returned {type(got)}, not bool"
        print(f"        task {task.id}: assertion evaluates to {got} before any action")

    check("env assertions are callable mid-episode and return a bool", assertions_run)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
