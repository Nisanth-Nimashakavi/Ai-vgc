"""Coalesce compatible Model19 forward requests across live battles."""
import asyncio

import torch


class AsyncInference19:
    def __init__(self, model, wait_seconds=0.001):
        self.model = model
        self.wait_seconds = wait_seconds
        self.pending = []
        self.calls = 0
        self.requests = 0
        self.max_batch = 0

    async def forward(self, categorical, numeric, **context):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if not self.pending:
            loop.call_later(self.wait_seconds, self._flush)
        self.pending.append((categorical, numeric, context, future))
        return await future

    def _flush(self):
        pending, self.pending = self.pending, []
        groups = {}
        for c, n, context, future in pending:
            if future.cancelled():
                continue
            # Slot-A and conditioned slot-B requests must not be combined.
            key = tuple(sorted(context))
            groups.setdefault(key, []).append((c, n, context, future))
        for group in groups.values():
            try:
                sizes = [c.shape[0] for c, _, _, _ in group]
                c = torch.cat([entry[0] for entry in group])
                n = torch.cat([entry[1] for entry in group])
                context = {key: torch.cat([entry[2][key] for entry in group])
                           for key in group[0][2]}
                with torch.inference_mode():
                    output = self.model(c, n, **context)
                pieces = {key: value.split(sizes, dim=0)
                          for key, value in output.items()}
                self.calls += 1
                self.requests += len(group)
                self.max_batch = max(self.max_batch, sum(sizes))
                for index, entry in enumerate(group):
                    if not entry[3].done():
                        entry[3].set_result({key: values[index] for key, values in pieces.items()})
            except Exception as error:
                for entry in group:
                    if not entry[3].done():
                        entry[3].set_exception(error)
