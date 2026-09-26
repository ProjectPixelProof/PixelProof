"""Model-free Oracle agent that performs the scientific clean recycle."""

from __future__ import annotations

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from offline_verifier import recycle_for_offline_verifier


class OfflineRecycleOracle(BaseAgent):
    """Run the task's deterministic solution, then replace the builder container."""

    @staticmethod
    def name() -> str:
        return "offline-recycle-oracle"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        result = await environment.exec(
            command=(
                "set -eu; "
                "test -x /workspace/oracle_solution/solve.sh; "
                "bash /workspace/oracle_solution/solve.sh"
            )
        )
        if result.return_code != 0:
            raise RuntimeError("sequential Oracle fixture failed")
        await recycle_for_offline_verifier(
            environment,
            required_artifact="submission/portfolio.json",
        )
