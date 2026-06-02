"""Azure Container Apps sandbox example scenarios.

Three scenarios build on top of ``ACASandboxClient`` and progressively show:

- ``single_sandbox``    — one agent inside one sandbox.
- ``parallel_sandboxes`` — many agents, each in its own sandbox.
- ``nested_sandboxes``  — an orchestrator agent that itself runs inside a
  sandbox and spawns more sandboxes for its workers.
"""
