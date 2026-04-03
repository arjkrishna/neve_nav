"""Simple centerline-following heuristic controller.

Generates "okay" trajectories by:
  - Translating forward at a modest constant speed
  - Rotating to minimize cross-track error and align with the path tangent

Used to seed the replay buffer before SAC training starts, avoiding
the early "collapse to do-nothing" failure mode.

Usage:
    heuristic = CenterlineFollowerHeuristic(pathfinder, intervention)
    heuristic.reset()  # after env.reset()
    action = heuristic.get_action(rng)
"""

import numpy as np

from eve.util.coordtransform import tracking3d_to_vessel_cs
from eve.util.polyline import (
    compute_cumulative_arclength,
    compute_segment_tangents,
    project_onto_polyline,
)


class CenterlineFollowerHeuristic:
    """Heuristic controller that follows the correct-path centerline.

    Args:
        pathfinder: A FixedPathfinder (must already have been reset so that
            ``path_points_vessel_cs`` is populated).
        intervention: The intervention object (provides fluoroscopy).
        max_translation: Maximum forward translation speed (mm/s).
        heading_kp: Proportional gain for heading-error correction (tangent alignment).
        crosstrack_kp: Proportional gain for cross-track error correction
            (steers back toward the centerline when off-path).
        noise_std_frac: Gaussian noise as a fraction of the action magnitude.
        catheter_follow_ratio: Catheter translation = gw * this ratio.
    """

    def __init__(
        self,
        pathfinder,
        intervention,
        max_translation: float = 20.0,
        heading_kp: float = 1.0,
        crosstrack_kp: float = 0.05,
        noise_std_frac: float = 0.1,
        catheter_follow_ratio: float = 0.8,
    ) -> None:
        self.pathfinder = pathfinder
        self.intervention = intervention
        self.max_translation = max_translation
        self.heading_kp = heading_kp
        self.crosstrack_kp = crosstrack_kp
        self.noise_std_frac = noise_std_frac
        self.catheter_follow_ratio = catheter_follow_ratio

        # Precomputed after reset
        self._polyline = np.empty((0, 3))
        self._cumlen = np.empty(0)
        self._tangents = np.empty((0, 3))
        self._total_length = 0.0

    def reset(self) -> None:
        """Recompute path data from the pathfinder (call after env.reset)."""
        self._polyline = self.pathfinder.path_points_vessel_cs
        if len(self._polyline) < 2:
            self._cumlen = np.zeros(max(1, len(self._polyline)))
            self._tangents = np.empty((0, 3))
            self._total_length = 0.0
            return
        self._cumlen = compute_cumulative_arclength(self._polyline)
        self._total_length = float(self._cumlen[-1])
        self._tangents = compute_segment_tangents(self._polyline)

    def get_action(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Compute a heuristic action for the current state.

        Translation: proportional to remaining distance, capped at max_translation.
        Rotation: combination of (a) tangent alignment and (b) cross-track
        error correction — steers toward the centerline when off-path
        AND aligns with the path direction.

        Returns:
            (4,) action array: [gw_trans, gw_rot, cath_trans, cath_rot].
        """
        if rng is None:
            rng = np.random.default_rng()

        if self._total_length < 1e-6:
            return np.zeros(4, dtype=np.float32)

        fluoro = self.intervention.fluoroscopy
        tip_3d = fluoro.tracking3d[0]
        tip_vessel = tracking3d_to_vessel_cs(
            tip_3d, fluoro.image_rot_zx, fluoro.image_center
        )

        proj = project_onto_polyline(tip_vessel, self._polyline, self._cumlen)

        d_rem = self._total_length - proj.s

        # ---- Translation: proportional to remaining distance, capped ----
        gw_trans = min(self.max_translation, d_rem * 0.1)
        gw_trans = max(gw_trans, 5.0)  # minimum forward push

        # ---- Rotation: align with tangent + correct cross-track error ----
        seg_idx = min(proj.segment_idx, len(self._tangents) - 1)
        tangent = self._tangents[seg_idx]

        # (a) Heading error: angle between device direction and path tangent
        heading_error = 0.0
        tracking = fluoro.tracking3d
        if len(tracking) >= 2:
            p0_v = tracking3d_to_vessel_cs(
                tracking[0], fluoro.image_rot_zx, fluoro.image_center
            )
            p1_v = tracking3d_to_vessel_cs(
                tracking[1], fluoro.image_rot_zx, fluoro.image_center
            )
            device_dir = p0_v - p1_v
            d_norm = np.linalg.norm(device_dir)
            if d_norm > 1e-8:
                device_dir = device_dir / d_norm
                cross = np.cross(device_dir, tangent)
                heading_error = float(cross[1])

        # (b) Cross-track error: signed lateral offset from centerline
        # Vector from projected point on path to actual tip position
        offset_vec = tip_vessel - proj.proj_point
        # Signed cross-track: positive = one side, negative = other side
        # Use cross product with tangent to determine sign
        cross_track_signed = float(np.cross(tangent, offset_vec)[1])

        # Combined rotation: steer to align with tangent AND reduce cross-track error
        gw_rot = -self.heading_kp * heading_error - self.crosstrack_kp * cross_track_signed
        gw_rot = float(np.clip(gw_rot, -np.pi, np.pi))

        # ---- Add noise for trajectory diversity ----
        gw_trans += rng.normal(0, abs(gw_trans) * self.noise_std_frac)
        gw_rot += rng.normal(0, max(abs(gw_rot), 0.1) * self.noise_std_frac)
        gw_trans = max(gw_trans, 0.0)  # never retract

        # ---- Catheter follows guidewire ----
        cath_trans = gw_trans * self.catheter_follow_ratio
        cath_rot = 0.0

        return np.array([gw_trans, gw_rot, cath_trans, cath_rot], dtype=np.float32)
