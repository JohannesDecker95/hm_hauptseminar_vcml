import torch
import lietorch
import numpy as np

from tqdm import trange
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops


class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)


        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            c = (ii == jj).long()
            fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            if hasattr(self.video, 'pgobuf'):
                self.video.pgobuf.add_rel_poses(self.ii[mask], self.jj[mask], self.target[:,mask], self.weight[:,mask])
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """

        with self.video.get_lock():
            self.video.tstamp[ix] = self.video.tstamp[ix+1]
            self.video.images[ix] = self.video.images[ix+1]
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.disps_up[ix] = self.video.disps_up[ix+1]
            self.video.disps_prior[ix] = self.video.disps_prior[ix+1]
            self.video.disps_prior_up[ix] = self.video.disps_prior_up[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]
            self.video.normals[ix] = self.video.normals[ix+1]
            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)


    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False, use_mono=False):
        """ run update operator on factor graph """

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)
        
        # correlation features
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask = \
            self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

        if t0 is None:
            # fix at least the first two poses to keep mono scale
            t0 = max(2, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target = coords1 + delta.to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float)
            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

            else:
                ii, jj, target, weight = self.ii, self.jj, self.target, self.weight


            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            # dense bundle adjustment
            self.video.cuda_ba(target, weight, damping, ii, jj, t0, t1, 
                itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only, use_mono=use_mono)

            self.video.upsample(torch.unique(self.ii), upmask)

        self.age += 1


    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value

        num, rig, ch, ht, wd = self.video.fmaps.shape
        corr_op = AltCorrBlock(self.video.fmaps.view(1, num*rig, ch, ht, wd))

        for step in (pbar := trange(steps)):
            pbar.set_description(f"Global BA Iteration #{step+1} with {t} keyframes {len(self.ii)} edges")
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 2
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:,v], rig * iis, rig * jjs + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                 
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis], corr1, motn[:,v], iis, jjs)

                    self.video.upsample(torch.unique(iis), upmask)

                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping

            damping = .2 * self.damping[torch.unique(self.ii)].contiguous() + EP

            # dense bundle adjustment
            self.video.cuda_ba(self.target, self.weight, damping, self.ii, self.jj, 10, t, 
                itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False, use_mono=False)

            self.video.dirty[:t] = True

    @torch.cuda.amp.autocast(enabled=True)
    def update_pgba(self, t0=None, t1=None, itrs=2, EP=1e-7, steps=6):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        if t1 is None:
            t1 = self.video.counter.value

        for step in range(steps):
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj, sim3=True)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            # correlation features
            corr = self.corr(coords1)
            self.net, delta, weight, damping, upmask = \
                self.update_op(self.net, self.inp, corr, motn, self.ii, self.jj)

            with torch.cuda.amp.autocast(enabled=False):
                self.target = coords1 + delta.to(dtype=torch.float)
                self.weight = weight.to(dtype=torch.float)
                self.damping[torch.unique(self.ii)] = damping

                ii = torch.cat([self.ii_inac, self.ii], 0)
                jj = torch.cat([self.jj_inac, self.jj], 0)
                target = torch.cat([self.target_inac, self.target], 1)
                weight = torch.cat([self.weight_inac, self.weight], 1)
                damping = .2 * self.damping[torch.unique(torch.cat((torch.arange(t0, t1, device='cuda'), ii)))].contiguous() + EP

                self.video.cuda_pgba(target, weight, damping, ii, jj, t0, t1, itrs=itrs, lm=1e-5, ep=1e-3)

        self.video.poses[:t1] = self.video.poses_sim3[:t1,:7]
        self.video.poses[:t1,:3] /= self.video.poses_sim3[:t1,-1:]
        self.video.pgobuf.rel_poses[:self.video.pgobuf.rel_N.value,:3] /= self.video.poses_sim3[self.video.pgobuf.rel_ii[:self.video.pgobuf.rel_N.value],-1:].cpu()

        ss = self.video.poses_sim3[:t1,-1:,None]
        self.video.disps[:t1] *= ss
        self.video.disps_up[:t1] *= ss.cpu()
        self.video.dscales[:t1] *= ss
        self.video.doffset[:t1] *= ss

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1), indexing='ij')
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.3, thresh=16.0, remove=False, backend=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


        es = []
        for i in range(t0, t):
            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        if backend:
            from scipy.spatial.transform import Rotation as R
            iii = ii[d.cpu() < thresh]
            jjj = jj[d.cpu() < thresh]
            if len(iii) > 0:
                Gij = (SE3(self.video.poses[jjj]) * SE3(self.video.poses[iii]).inv()).data
                euls = R.from_quat(Gij[:, 3:].cpu().numpy()).as_euler('zxy', degrees=True)
                oris = np.linalg.norm(euls, axis=1)
                d[d < thresh][oris > 150] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)
