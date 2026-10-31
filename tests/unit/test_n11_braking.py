import unittest
from occluactisafe.planning.braking import braking_profile
from occluactisafe.control.executor import AnalyticExecutor


class BrakingTests(unittest.TestCase):
    def test_c02_exact_distance(self):
        p = braking_profile(12, delay=.15)
        self.assertAlmostEqual(p.at(10).s, 21.175, delta=1e-6)
        self.assertAlmostEqual(p.duration, .15 + 3.2291666666666665)
        self.assertEqual(p.at(10).v, 0)
        for i in range(1001):
            state = p.at(i / 100)
            self.assertGreaterEqual(state.v, -1e-9); self.assertGreaterEqual(state.a, -4.5 - 1e-8)
            self.assertLessEqual(abs(state.j), 8)

    def test_triangular_negative_acceleration_and_hold(self):
        for v, a in ((12, -1), (.3, 0), (0, 0)):
            p = braking_profile(v, a)
            self.assertEqual(p.at(10).v, 0); self.assertEqual(p.at(10).a, 0)
            self.assertEqual(p.at(10).s, p.at(20).s)
        moving = braking_profile(12, -1, delay=.15)
        self.assertAlmostEqual(moving.at(.15).s, 12 * .15 - .15**2 / 2)

    def test_invalid_boundaries(self):
        for kwargs in ({'v0':-1},{'v0':.01,'a0':-1},{'v0':12,'a0':-5},{'v0':12,'B':0},{'v0':12,'J':float('nan')},{'v0':True}):
            with self.assertRaises(ValueError): braking_profile(**kwargs)

    def test_queued_prefix_and_no_delay_reset(self):
        e = AnalyticExecutor(); e.request(0, 'DEGRADED')
        before = [e.at(t) for t in (.2,.25,.3,.35)]
        e.request(.2, 'AEB')
        for t, state in zip((.2,.25,.3,.35), before):
            self.assertAlmostEqual(e.at(t).s, state.s); self.assertAlmostEqual(e.at(t).v, state.v)
            self.assertAlmostEqual(e.at(t).a, state.a)
        for i in range(8,30): self.assertFalse(e.request(i*.05,'AEB'))
        self.assertEqual(len(e.requests),2)
        profile, started = e.preview(1.,4.5,8.)
        self.assertIs(profile,e.profile);self.assertEqual(started,e.started)
        self.assertEqual(e.at(10).v,0)
        with self.assertRaises(ValueError): e.control(0)
        for i in range(4,201):
            c=e.control(i*.05);self.assertEqual(c['throttle']*c['brake'],0)
        from occluactisafe.planning.candidates import longitudinal_candidates
        candidate=longitudinal_candidates(e,.5,'sim','map')[0]
        for i in range(1001):
            t=.5 + 3.15*i/1000
            self.assertLessEqual(abs(candidate.path.at(t).x-e.at(t).s),candidate.geometry.margin+1e-9)


if __name__ == '__main__': unittest.main()
