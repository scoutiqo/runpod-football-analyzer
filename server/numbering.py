# server/numbering.py
import time

class SquadNumberer:
    def __init__(self):
        self.map = {}           # (team, tid) -> squad_no
        self.used = {0:set(),1:set()}
        self.last_seen = {}     # (team, tid) -> ts
        self.max_age = 8.0

    def _next_free(self, team:int)->int:
        for n in range(1,12):
            if n not in self.used[team]:
                return n
        # recycle the stalest
        oldest=None; age=-1
        for (t,tid),ts in self.last_seen.items():
            if t!=team: continue
            a=time.time()-ts
            if a>age: age=a; oldest=(t,tid)
        if oldest:
            n=self.map.pop(oldest)
            self.used[team].discard(n)
            return n
        return 11

    def touch(self, team:int, tid:int):
        key=(team,tid)
        if key not in self.map:
            n=self._next_free(team)
            self.map[key]=n; self.used[team].add(n)
        self.last_seen[key]=time.time()

    def gc(self):
        now=time.time()
        for key,ts in list(self.last_seen.items()):
            if now-ts>self.max_age:
                n=self.map.pop(key, None)
                if n is not None: self.used[key[0]].discard(n)
                self.last_seen.pop(key, None)

    def get(self, team:int, tid:int)->int:
        return self.map.get((team,tid), 0)
