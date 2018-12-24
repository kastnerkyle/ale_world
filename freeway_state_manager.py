import sys
from atari_py.ale_python_interface import ALEInterface
from collections import Counter

class FreewayStateManager(object):
    """
    A state manager with a goal state and action space of 2

    State manager must, itself have *NO* state / updating behavior
    internally. Otherwise we need deepcopy() or cloneSystemState in get_action_probs, making it slower
    """
    def __init__(self, random_state, rollout_limit=1000):
        self.rollout_limit = rollout_limit
        self.random_state = random_state
        self.ale = ALEInterface()
        self.ale.setInt('random_seed', 123)
        rom_path = "atari_roms/freeway.bin"
        self.ale.loadROM(rom_path)

        # Set USE_SDL to true to display the screen. ALE must be compilied
        # with SDL enabled for this to work. On OSX, pygame init is used to
        # proxy-call SDL_main.
        USE_SDL = False
        if USE_SDL:
          if sys.platform == 'darwin':
            import pygame
            pygame.init()
            self.ale.setBool('sound', False) # Sound doesn't work on OSX
          elif sys.platform.startswith('linux'):
            # the sound is awful
            self.ale.setBool('sound', False)
          self.ale.setBool('display_screen', True)

        # get background subtraction
        arr = []
        s = self.ale.getScreenGrayscale()
        num = 1000
        empty = np.zeros((num, s.shape[0], s.shape[1]))
        for ii in range(num):
             self.ale.act(0)
             s = self.ale.getScreenGrayscale()
             empty[ii] = s[..., 0]

        # max and min are 214, 142
        o = np.zeros((empty.shape[1], empty.shape[2]), np.int)
        for i in range(o.shape[0]):
            for j in range(o.shape[1]):
                this_pixel = empty[:,i,j]
                (values,cnts) = np.unique(this_pixel, return_counts=True)
                o[i,j] = int(values[np.argmax(cnts)])
        self.background = o
        self.color = 233
        self.vert_l = 45
        self.vert_r = 50
        self.ale.reset_game()

    def _extract(self, s):
        if hasattr(s, "shape"):
            s = s[..., 0]
            s_a = [s]
        else:
            s_a = [s_i[..., 0] for s_i in s]
        """
        if hasattr(s, "shape"):
            s = s[..., 0]
        else:
            # list of frames in
            s = np.concatenate(s, axis=-1)
            s = np.max(s, axis=-1)
        """
        all_pos = []
        for s in s_a:
            # try to find ourselves
            vert = s[:, self.vert_l:self.vert_r]
            # "racecar" strip detector
            u_d = np.zeros((vert.shape[1]))
            hodl = 0. * u_d - 1
            strip_dets = [[] for i in range(len(u_d))]
            for ii in list(range(len(vert)))[::-1]:
                detect = (vert[ii] == 233)
                for jj in range(len(detect)):
                    if u_d[jj] == 1:
                        strip_dets[jj].append(ii)
                    if u_d[jj] == 0 and detect[jj] == True:
                        u_d[jj] = 1.
                        hodl[jj] = ii
                        strip_dets[jj].append(ii)
                    elif u_d[jj] == 1 and detect[jj] == False:
                        u_d[jj] = 0.
                        hodl[jj] = -1
            flat_dets = [(n, sl) for n, l in enumerate(strip_dets) for sl in l]
            # count of occurence for every pixel - max should be vert_l - vert_r
            counts = Counter(flat_dets)
            # aggregate counts across pixel window
            # pseudoconv 3x3
            c_counts = {}
            h = 3
            w = 3
            for k1 in counts.keys():
                c_counts[k1] = 0
                for k2 in counts.keys():
                    if abs(k1[0] - k2[0]) <= w and abs(k1[1] - k2[1]) <= h:
                        c_counts[k1] += 1
            mx = max(c_counts.values())
            all_mx = [k for k in c_counts if c_counts[k] == mx]
            # for now, assume med_w always == 2 since we move vertically
            #med_w = int(np.median([k[0] for k in all_mx]))
            med_w = 2
            med_h = int(np.median([k[1] for k in all_mx]))

            all_pos.append(med_h)

            ib = 7
            ob = 20
            mb = 3
            uu = med_h - ob
            u = med_h - ib
            d = med_h + ib
            dd = med_h + ob
            l = 45 + med_w - ib
            ll = 45 + med_w - ob
            r = 45 + med_w + ib
            rr = 45 + med_w + ob
            u_diff = s[uu:u, ll:rr] - self.background[uu:u, ll:rr]
            d_diff = s[d:dd, ll:rr] - self.background[d:dd, ll:rr]
            lu_diff = s[u:med_h-mb, ll:l] - self.background[u:med_h-mb, ll:l]
            lm_diff = s[med_h-mb:med_h+mb, ll:l] - self.background[med_h-mb:med_h+mb, ll:l]
            ld_diff = s[med_h+mb:d, ll:l] - self.background[med_h+mb:d, ll:l]
            ru_diff = s[u:med_h-mb, r:rr] - self.background[u:med_h-mb, r:rr]
            rm_diff = s[med_h-mb:med_h+mb, r:rr] - self.background[med_h-mb:med_h+mb, r:rr]
            rd_diff = s[med_h+mb:d, r:rr] - self.background[med_h+mb:d, r:rr]
            u_det = (np.abs(u_diff).sum() > 0)
            d_det = (np.abs(d_diff).sum() > 0)
            lu_det = (np.abs(lu_diff).sum() > 0)
            lm_det = (np.abs(lm_diff).sum() > 0)
            ld_det = (np.abs(ld_diff).sum() > 0)
            ru_det = (np.abs(ru_diff).sum() > 0)
            rm_det = (np.abs(rm_diff).sum() > 0)
            rd_det = (np.abs(rd_diff).sum() > 0)
            # buffer detector around our position
            s_min = [med_h, u_det, d_det,
                     lu_det, lm_det, ld_det,
                     ru_det, rm_det, rd_det]
        s_min = [int(np.min(all_pos)), int(np.max(all_pos))]
        return s_min

    def get_next_state_reward(self, state, action):
        # ignores state input due to ale being stateful
        frameskip = 48
        total_reward = 0

        all_s = []
        for i in range(frameskip):
            reward = self.ale.act(action)
            s = self.ale.getScreenGrayscale()
            all_s.append(s)
            total_reward += reward
        # use last 2 frames?
        s_min = self._extract(all_s)
        return s_min, total_reward

    def get_action_space(self):
        return list([a for a in self.ale.getMinimalActionSet()])

    def get_valid_actions(self, state):
        return list([a for a in self.ale.getMinimalActionSet()])

    def get_ale_clone(self):
        # if state manager is stateful, will use this to clone
        return self.ale.cloneState()

    def get_init_state(self):
        ss = self.ale.getScreenGrayscale()
        s = self._extract(ss)
        s[0] = ss.shape[0]
        s[1] = ss.shape[0]
        return s

    def rollout_fn(self, state):
        # can define custom rollout function
        return self.random_state.choice(self.get_valid_actions(state))

    def score(self, state):
        return 0.

    def is_finished(self, state):
        # if this check is slow
        # can rewrite as _is_finished
        # then add
        # self.is_finished = MemoizeMutable(self._is_finished)
        # to __init__ instead

        # return winner, score, end
        # winner normally in [-1, 0, 1]
        # if it's one player, can just use [0, 1] and it's fine
        # score arbitrary float value
        # end in [True, False]
        #return (1, 1., True) if state == self.goal_state else (0, 0., False)
        fin = self.ale.game_over()
        return (1, 1., True) if fin else (0, 0., False)

    def rollout_from_state(self, state):
        # example rollout function
        s = state
        w, sc, e = self.is_finished(state)
        if e:
            return sc

        c = 0
        t_r = 0
        while True:
            a = self.rollout_fn(s)
            s_n, r = self.get_next_state_reward(s, a)
            t_r += r

            e = self.is_finished(s_n)
            s = s_n
            c += 1
            if e:
                return np.abs(s[1] - s[0]) / 200.
                #return t_r

            if c > self.rollout_limit:
                
                return np.abs(s[1] - s[0]) / 200.
                # can also return different score if rollout limit hit
                #return t_r


if __name__ == "__main__":
    from puct_mcts import MCTS, MemoizeMutable
    import numpy as np
    mcts_random = np.random.RandomState(1110)
    state_random = np.random.RandomState(11)
    exact = True

    state_man = FreewayStateManager(state_random, rollout_limit=100)
    mcts = MCTS(state_man, use_ale_clone=True, n_playout=100, random_state=mcts_random)
    state = mcts.state_manager.get_init_state()
    winner, score, end = mcts.state_manager.is_finished(state)
    states = [state]
    while True:
        t_r = 0
        if not end:
            if not exact:
                a, ap = mcts.sample_action(state, temp=temp, add_noise=noise)
            else:
                a, ap = mcts.get_action(state)

            for i in mcts.root.children_.keys():
                print(i, mcts.root.children_[i].__dict__)
                print("")
            mcts.update_tree_root(a)
            state, reward = mcts.state_manager.get_next_state_reward(state, a)
            states.append(state)
            t_r += reward
            print(t_r, states[-10:])
            winner, score, end = mcts.state_manager.is_finished(state)
        if end:
            print(states[-1])
            print(t_r)
            print("Ended")
            from IPython import embed; embed(); raise ValueError()
            mcts.reconstruct_tree()
            break

