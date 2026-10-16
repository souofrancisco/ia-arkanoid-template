"""Latest-state-only client loop. BaseAgent.run answers every queued state in order, so a decision slower than one tick
builds a growing backlog of stale states; here only the newest state is ever deliberated on (the assignment's
"a slow agent simply reacts to fewer states")."""

import asyncio
import json
import logging

import websockets


async def run_latest(agent) -> None:
    try:
        async with websockets.connect(agent.server_uri) as ws:
            await ws.send(json.dumps({"client": "agent", "name": agent.name, "id": agent.agent_id}))
            logging.info("Connected to %s as %s (id %s)", agent.server_uri, agent.name, agent.agent_id)
            slot: dict = {"msg": None, "closed": False}
            wake = asyncio.Event()

            async def reader() -> None:
                try:
                    async for raw in ws:
                        data = json.loads(raw)
                        kind = data.get("type")
                        if kind == "setup":
                            agent.player_id, agent.setup_data = data.get("player_id"), data
                        elif kind in ("state", "update"):
                            slot["msg"] = data
                            wake.set()
                finally:
                    slot["closed"] = True
                    wake.set()

            task = asyncio.create_task(reader())
            try:
                while True:
                    await wake.wait()
                    wake.clear()
                    data, slot["msg"] = slot["msg"], None
                    if data is None:
                        if slot["closed"]:
                            break
                        continue
                    agent.current_state = data
                    if data.get("game_won"):
                        agent.stop_reason = "game won"
                        break
                    if data.get("game_over"):
                        agent.stop_reason = "time limit reached" if data.get("timed_out") else "game over"
                        break
                    action = await agent.deliberate()
                    if action is not None:
                        await ws.send(json.dumps(action))
            finally:
                task.cancel()
    except Exception as e:  # same contract as BaseAgent.run
        logging.error("Connection error: %s", e)
        agent.stop_reason = f"connection error: {e}"
    agent.report()
