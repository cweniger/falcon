"""Truncated-prior proposal: nested regions of the prior read off a conditional flow.

Port of the hysteretic two-region ladder of the LDC MBHB reference runs
(``run_trunc7b.py``, "trunc7"). Everything lives in the standard-normal latent
space of the prior, where the prior is N(0, I).

A region is the level set ``{ln q_c(u) > thr}`` of a conditional flow ``q_c``
at the observation, intersected with its parent region. It is frozen when it is
minted: it keeps copies of the conditional and marginal flows it was read off,
the embedded observation and its threshold, so it means the same thing whenever
it is used, however far the networks have moved since.

Two regions are live, nested: the inner region ``I`` must contain the
``x_sigma`` region of the current conditional flow, and the outer region ``O``
is what the proposal samples, so the training data reaches beyond ``I`` and the
flows never lose data at its boundary. Wider regions wait on a stack. Once per
round ``m_out``, the current conditional flow's mass outside ``I``, moves the
ladder::

    m_out > leak(x)            EXPAND:   I <- O, O <- pop the stack (or the prior)
    otherwise                  mint a candidate at leak(x + delta) inside I;
      R <= vratio and          CONTRACT: push O, O <- I, I <- candidate
      R_ess >= v_min_ess

``R`` is the prior mass of the candidate relative to ``I``, estimated on one
common set of marginal-flow draws so that the errors of numerator and
denominator cancel. Expanding promotes a stored region, which travels with the
marginal flow that was trained while the buffer still reached out there;
contracting mints a tighter region that the current marginal flow already
covers. Fast retreat, cautious approach.

Sampling a region is importance sampling: draw from the region's marginal
flow, keep the draws inside the region, weight them by prior over marginal
flow, and resample without replacement so that no simulation is spent twice.
The pool of weighted draws keeps growing until its effective sample size is
``ess_factor`` times the number of draws taken from it, so a pass that hits a
hole in the marginal flow buys more passes instead of feeding a tilted batch
to the buffer.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from falcon.core.logger import info, log, warning

PRIOR = "prior"


def leak(x: float) -> float:
    """Two-sided standard-normal tail mass at ``x`` sigma (x=3: 2.7e-3, x=4: 6.3e-5).

    ``x`` states contained mass rather than a density contour, which makes the
    criterion independent of the dimension.
    """
    return math.erfc(x / math.sqrt(2.0))


def log_normal(u: torch.Tensor) -> torch.Tensor:
    """Log density of the latent prior N(0, I), in float64."""
    u = u.double()
    return -0.5 * (u.pow(2).sum(-1) + u.shape[-1] * math.log(2 * math.pi))


def gumbel_top_k(logw: torch.Tensor, k: int) -> torch.Tensor:
    """``k`` indices drawn without replacement with probability proportional to ``exp(logw)``."""
    u = torch.rand(len(logw), dtype=torch.float64, device=logw.device).clamp_min(1e-300)
    return torch.topk(logw - torch.log(-torch.log(u)), k).indices


def _ess(w: torch.Tensor) -> float:
    """Kish effective sample size of non-negative weights."""
    total, square = float(w.sum()), float(w.pow(2).sum())
    return total ** 2 / square if len(w) and square > 0 else 0.0


@dataclass
class LadderConfig:
    """Parameters of the region ladder (see ``FlowMatching`` for their meaning)."""

    x_sigma: float = 3.0
    delta_sigma: float = 1.0
    min_keep_frac: float = 0.01
    vratio: float = 0.8
    v_min_ess: float = 1000
    v_max_draws: int = 1048576
    chain_depth: int = 1
    n_region: int = 65536
    n_mout: int = 65536
    max_sample_passes: int = 64
    ess_factor: float = 4.0


class FlowPair:
    """Conditional and marginal flow at one observation, with densities in the latent frame.

    Args:
        conditional, marginal: Flows with ``sample(s, **sample_kw)`` returning
            latent draws and ``log_prob(u, s, **density_kw)`` returning latent
            log densities (float64), one per row of the conditioning ``s``.
        s_obs: Embedded observation, shape ``(1, C)``; the marginal flow is
            conditioned on zeros of the same width.
    """

    def __init__(self, conditional, marginal, s_obs: torch.Tensor,
                 sample_kw: Optional[dict] = None, density_kw: Optional[dict] = None):
        self.conditional = conditional
        self.marginal = marginal
        self.s_obs = s_obs
        self.sample_kw = dict(sample_kw or {})
        self.density_kw = dict(density_kw or {})

    def _flow_and_condition(self, which: str, n: int):
        s = self.s_obs.expand(n, -1)
        if which == "c":
            return self.conditional, s
        return self.marginal, torch.zeros_like(s)

    def sample(self, which: str, n: int) -> torch.Tensor:
        flow, s = self._flow_and_condition(which, n)
        return flow.sample(s, **self.sample_kw)

    def log_prob(self, which: str, u: torch.Tensor) -> torch.Tensor:
        flow, s = self._flow_and_condition(which, len(u))
        return flow.log_prob(u, s, **self.density_kw)


class RegionLadder:
    """Inner region ``I``, outer region ``O`` and a stack of wider regions.

    Regions are records (plain values and numpy trees) keyed by id; the prior
    is the permanent region ``"prior"``, which is never stored.

    Args:
        config: Ladder parameters.
        load_pair: Builds the ``FlowPair`` of a region record.
        param_dim: Dimension of the latent space.
        device: Device of the flows.
    """

    def __init__(self, config: LadderConfig, load_pair: Callable[[Dict[str, Any]], FlowPair],
                 param_dim: int, device):
        self.config = config
        self._load_pair = load_pair
        self.param_dim = param_dim
        self.device = device
        self.regions: Dict[str, Dict[str, Any]] = {}
        self.inner = PRIOR
        self.outer = PRIOR
        self.stack: List[str] = []
        self.minted = 0
        self._pairs: Dict[str, FlowPair] = {}
        self._pool = None

    # ==================== Regions ====================

    def volume(self, rid: str) -> float:
        """Prior mass of a region, including the cut by its parent."""
        return 1.0 if rid == PRIOR else float(self.regions[rid]["V_cut"])

    def _pair(self, rid: str) -> FlowPair:
        if rid not in self._pairs:
            self._pairs[rid] = self._load_pair(self.regions[rid])
        return self._pairs[rid]

    @torch.no_grad()
    def cut(self, rid: str, u: torch.Tensor, depth: Optional[int] = None) -> torch.Tensor:
        """Membership of latent points in a region, including its parents up to ``depth``.

        A region is the intersection of its own level set with its parent, so
        a point can only be in a child if it is in the parent; ``depth`` (the
        ``chain_depth`` by default, -1 for all) limits how many ancestors are
        tested. Ancestors are evaluated only on the survivors.
        """
        if rid == PRIOR:
            return torch.ones(len(u), dtype=torch.bool, device=u.device)
        region = self.regions[rid]
        inside = self._pair(rid).log_prob("c", u) > region["thr"]
        depth = self.config.chain_depth if depth is None else depth
        parent = region["parent"]
        if depth != 0 and parent != PRIOR and inside.any():
            if parent not in self.regions:
                warning(f"Region {rid}: parent {parent} is missing; testing the region alone")
                return inside
            sub = torch.zeros_like(inside)
            sub[inside] = self.cut(parent, u[inside], depth - 1 if depth > 0 else -1)
            inside = sub
        return inside

    @torch.no_grad()
    def mass_outside(self, current: FlowPair, rid: str) -> float:
        """The current conditional flow's mass outside a region.

        The draws come from the mixture Q = (q_c + q_m) / 2 and are weighted by
        q_c / Q (at most 2): the marginal half populates the outside densely,
        so a tail mass of 1e-6 is an average of many small terms instead of a
        count of rare ones.
        """
        if rid == PRIOR:
            return 0.0
        half = max(1, self.config.n_mout // 2)
        u = torch.cat([current.sample("c", half), current.sample("m", half)])
        lqc, lqm = current.log_prob("c", u), current.log_prob("m", u)
        finite = torch.isfinite(lqc) & torch.isfinite(lqm) & torch.isfinite(u).all(1)
        if not finite.any():
            return float("nan")
        lw = (lqc - (torch.logaddexp(lqc, lqm) - math.log(2.0)))[finite]
        w = torch.exp(lw - lw.max())
        inside = self.cut(rid, u[finite])
        return float(w[~inside].sum() / w.sum())

    @torch.no_grad()
    def mint(self, current: FlowPair, target_leak: float, parent: str) -> Optional[Dict[str, float]]:
        """Threshold and prior mass of a candidate region at ``target_leak`` inside ``parent``.

        The threshold is the ``target_leak`` quantile of ``ln q_c`` over draws
        from the mixture Q = (q_c + q_m) / 2, weighted by q_c / Q. It is then
        lowered, if necessary, so that at least ``min_keep_frac`` of the
        marginal flow's draws survive the cut, which widens the region (the
        safe direction) and keeps sampling it affordable.

        The prior mass ``V`` of the level set and the ratio ``R`` of the
        candidate's mass inside the parent to the parent's mass are estimated
        on marginal-flow draws weighted by prior / q_m; more draws are added
        until ``R`` carries ``v_min_ess`` or ``v_max_draws`` is reached.

        Returns:
            The candidate's statistics, or None if no draw had a finite density.
        """
        cfg = self.config
        n = cfg.n_region
        u_m = current.sample("m", n)
        u = torch.cat([current.sample("c", n), u_m])
        lqc, lqm = current.log_prob("c", u), current.log_prob("m", u)
        finite = torch.isfinite(lqc) & torch.isfinite(lqm) & torch.isfinite(u).all(1)
        if not finite[n:].any():
            return None

        # Nominal contour: the target_leak quantile of ln q_c under Q, weights q_c / Q
        lw_mix = (lqc - (torch.logaddexp(lqc, lqm) - math.log(2.0)))[finite]
        lqc_finite = lqc[finite]
        order = torch.argsort(lqc_finite)
        wq = torch.exp(lw_mix[order] - lw_mix.max())
        cum = torch.cumsum(wq, 0) / wq.sum()
        k = int(torch.searchsorted(cum, torch.tensor(target_leak, dtype=cum.dtype, device=cum.device)))
        contour = float(lqc_finite[order[min(k, len(order) - 1)]])

        # Floor: at least min_keep_frac of the marginal flow's own draws survive
        ok = finite[n:]
        lqc_m, lqm_m, u_m = lqc[n:][ok], lqm[n:][ok], u_m[ok]
        floor = float(torch.quantile(lqc_m, 1.0 - cfg.min_keep_frac))
        thr = min(contour, floor)

        # Prior mass by prior / q_m importance sampling, on one common set of draws
        lw = log_normal(u_m) - lqm_m
        in_cand = lqc_m > thr
        in_parent = self.cut(parent, u_m)
        n_drawn = n
        while True:
            w = torch.exp(lw - lw.max())
            V = float(w[in_cand].sum() / w.sum())
            num, den = w[in_cand & in_parent], w[in_parent]
            R = float(num.sum() / den.sum()) if float(den.sum()) > 0 else float("nan")
            # R = A / B with A inside B: both sums must be resolved
            R_ess = min(_ess(num), _ess(den))
            if R_ess >= cfg.v_min_ess or n_drawn >= cfg.v_max_draws:
                break
            u_x = current.sample("m", n)
            lqc_x, lqm_x = current.log_prob("c", u_x), current.log_prob("m", u_x)
            ok_x = torch.isfinite(lqc_x) & torch.isfinite(lqm_x) & torch.isfinite(u_x).all(1)
            lw = torch.cat([lw, (log_normal(u_x) - lqm_x)[ok_x]])
            in_cand = torch.cat([in_cand, lqc_x[ok_x] > thr])
            in_parent = torch.cat([in_parent, self.cut(parent, u_x[ok_x])])
            n_drawn += n

        stats = {
            "thr": thr, "contour": contour, "floor": floor,
            "V": V, "V_ess": _ess(w), "R": R, "R_ess": R_ess,
            "V_cut": R * self.volume(parent), "draws": n_drawn,
        }
        info(
            f"[mint] leak={target_leak:.3g} thr={thr:.3f} (contour {contour:.3f}, floor {floor:.3f}"
            f"{'  FLOOR BINDS' if floor < contour else ''})  V={V:.4g} V_ess={stats['V_ess']:.0f}/{n_drawn}"
            f"  R={R:.4g} R_ess={R_ess:.0f}"
        )
        return stats

    def step(self, current: FlowPair, trees: Dict[str, Any]) -> str:
        """Move the ladder once, with the current best flows; returns the action.

        Args:
            current: The current best flows at the observation.
            trees: What a region minted now stores to rebuild ``current``
                (``c``, ``m`` and ``s_obs``).

        Returns:
            ``"EXPAND"``, ``"CONTRACT"`` or ``"hold (...)"``.
        """
        cfg = self.config
        hi, lo = leak(cfg.x_sigma), leak(cfg.x_sigma + 2 * cfg.delta_sigma)
        outer_before = self.outer
        m_out = self.mass_outside(current, self.inner)
        candidate = None
        if m_out > hi:
            action = "EXPAND"
            self.inner = self.outer
            self.outer = self.stack.pop() if self.stack else PRIOR
        else:
            target = leak(cfg.x_sigma + cfg.delta_sigma)
            candidate = self.mint(current, target, self.inner)
            if candidate is None:
                action = "hold (no finite draws)"
            elif candidate["R_ess"] < cfg.v_min_ess:
                action = f"hold (R_ess {candidate['R_ess']:.0f} < {cfg.v_min_ess:g})"
            elif not candidate["R"] <= cfg.vratio:
                action = f"hold (R {candidate['R']:.3g} > {cfg.vratio:g})"
            else:
                action = "CONTRACT"
                self.minted += 1
                rid = f"r{self.minted:03d}"
                self.regions[rid] = {
                    **candidate, "leak": target, "parent": self.inner, "minted": self.minted,
                    "c": trees["c"], "m": trees["m"], "s_obs": trees["s_obs"],
                }
                self.stack.append(self.outer)
                self.outer = self.inner
                self.inner = rid
        self._collect()
        if self.outer != outer_before:
            self._pool = None

        info(
            f"[ladder] m_out={m_out:.3e} vs [{lo:.3e}, {hi:.3e}] -> {action}   "
            f"I={self.inner}(V={self.volume(self.inner):.3g}) O={self.outer}(V={self.volume(self.outer):.3g}) "
            f"stack={len(self.stack)}"
        )
        metrics = {
            "ladder:m_out": m_out,
            "ladder:action": {"EXPAND": -1, "CONTRACT": 1}.get(action, 0),
            "ladder:V_inner": self.volume(self.inner),
            "ladder:V_outer": self.volume(self.outer),
            "ladder:stack": len(self.stack),
        }
        if candidate is not None:
            metrics.update({f"ladder:candidate_{k}": candidate[k] for k in ("thr", "V", "R", "R_ess")})
        log(metrics)
        return action

    def _ids(self, live: bool) -> List[str]:
        """Stored regions in use: all reachable ones, or only those sampling needs."""
        roots = [self.inner, self.outer] + ([] if live else list(self.stack))
        keep: List[str] = []
        for rid in roots:
            depth = self.config.chain_depth if live else -1
            while rid != PRIOR and rid in self.regions:
                if rid not in keep:
                    keep.append(rid)
                if depth == 0:
                    break
                depth = depth - 1 if depth > 0 else -1
                rid = self.regions[rid]["parent"]
        return keep

    def _collect(self) -> None:
        """Forget regions that can no longer be reached."""
        keep = set(self._ids(live=False))
        self.regions = {rid: r for rid, r in self.regions.items() if rid in keep}
        self._pairs = {rid: p for rid, p in self._pairs.items() if rid in keep}

    # ==================== Sampling ====================

    @torch.no_grad()
    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """``n`` latent draws from the prior truncated to ``O``, with their log density.

        Draws are taken without replacement from a pool of weighted draws of
        ``O``, which lives until ``O`` changes. Before taking them, the pool
        is extended until its effective sample size is ``ess_factor`` times
        everything taken from it so far, this request included (at most
        ``max_sample_passes`` passes; a pool that misses the target stops
        enforcing it).
        """
        if self.outer == PRIOR:
            u = torch.randn(n, self.param_dim, dtype=torch.float64, device=self.device)
            return u, log_normal(u)
        pool = self._pool
        if pool is None or pool["region"] != self.outer:
            empty = torch.zeros(0, dtype=torch.float64, device=self.device)
            pool = self._pool = {
                "region": self.outer,
                "u": torch.zeros(0, self.param_dim, dtype=torch.float64, device=self.device),
                "logw": empty,       # draws still available
                "logw_all": empty,   # every draw the pool ever held, for its ESS
                "taken": 0,
                "gate_failed": False,
            }
        cfg = self.config

        def short():
            gated = not pool["gate_failed"] and self.pool_ess() < cfg.ess_factor * (pool["taken"] + n)
            return len(pool["u"]) < n or gated

        passes = 0
        while passes < cfg.max_sample_passes and short():
            passes += 1
            self._fill(pool)
        if not pool["gate_failed"] and self.pool_ess() < cfg.ess_factor * (pool["taken"] + n):
            # Give up on the gate for this region instead of paying the passes on every request
            pool["gate_failed"] = True
            warning(f"Region {self.outer}: pool ESS {self.pool_ess():.0f} after {passes} passes, "
                    f"below {cfg.ess_factor:g} x {pool['taken'] + n} draws taken; "
                    "not enforced again until the region changes")
        take = min(n, len(pool["u"]))
        idx = gumbel_top_k(pool["logw"], take)
        u = pool["u"][idx]
        remaining = torch.ones(len(pool["u"]), dtype=torch.bool, device=self.device)
        remaining[idx] = False
        pool["u"], pool["logw"] = pool["u"][remaining], pool["logw"][remaining]
        pool["taken"] += take
        if take < n:
            if take == 0:
                warning(f"Region {self.outer}: no accepted draws in {passes} passes; "
                        "proposing from the prior")
                u = torch.randn(n, self.param_dim, dtype=torch.float64, device=self.device)
                return u, log_normal(u)
            warning(f"Region {self.outer}: only {take}/{n} distinct draws in {passes} passes; "
                    "repeating some")
            u = torch.cat([u, u[torch.randint(take, (n - take,), device=self.device)]])
        return u, log_normal(u) - math.log(self.volume(self.outer))

    def pool_ess(self) -> float:
        """Effective sample size of every draw the current proposal pool has held."""
        if self._pool is None or len(self._pool["logw_all"]) == 0:
            return 0.0
        logw = self._pool["logw_all"]
        return _ess(torch.exp(logw - logw.max()))

    def _fill(self, pool) -> None:
        """One pass of ``n_region`` marginal-flow draws of ``O`` into the pool."""
        pair = self._pair(pool["region"])
        u = pair.sample("m", self.config.n_region)
        keep = torch.isfinite(u).all(1)
        keep[keep.clone()] = self.cut(pool["region"], u[keep])
        u = u[keep]
        lqm = pair.log_prob("m", u)
        finite = torch.isfinite(lqm)
        u, logw = u[finite], (log_normal(u) - lqm)[finite]
        pool["u"] = torch.cat([pool["u"], u])
        pool["logw"] = torch.cat([pool["logw"], logw])
        pool["logw_all"] = torch.cat([pool["logw_all"], logw])
        eps = _ess(torch.exp(logw - logw.max())) / len(logw) if len(logw) else float("nan")
        log({
            "proposal:acceptance": len(u) / self.config.n_region,
            "proposal:eps": eps,
            "proposal:pool_ess": self.pool_ess(),
        })

    # ==================== State ====================

    def export(self, full: bool) -> Dict[str, Any]:
        """The ladder as a state tree: everything, or only what sampling needs."""
        ids = self._ids(live=not full)
        return {
            "inner": self.inner,
            "outer": self.outer,
            "stack": list(self.stack) if full else [],
            "minted": self.minted,
            "regions": {rid: self.regions[rid] for rid in ids},
        }

    def load(self, tree: Dict[str, Any]) -> None:
        """Install a tree from ``export()``."""
        self.inner = tree.get("inner", PRIOR)
        self.outer = tree.get("outer", PRIOR)
        self.stack = list(tree.get("stack", []))
        self.minted = int(tree.get("minted", 0))
        self.regions = dict(tree.get("regions", {}))
        self._pairs = {}
        self._pool = None
