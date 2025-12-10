# server/sse_bus.py
import asyncio
class Bus:
    def __init__(self): self.subs = {}; self.lock = asyncio.Lock()
    async def subscribe(self, job):
        q = asyncio.Queue(maxsize=256)
        async with self.lock: self.subs.setdefault(job, []).append(q)
        return q
    async def unsubscribe(self, job, q):
        async with self.lock:
            if job in self.subs and q in self.subs[job]: self.subs[job].remove(q)
    async def publish(self, job, msg):
        async with self.lock:
            for q in self.subs.get(job, []):
                try: q.put_nowait(msg)
                except asyncio.QueueFull: _=q.get_nowait(); await q.put(msg)
BUS = Bus()

