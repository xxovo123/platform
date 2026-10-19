"""Correct the preflight-only double counting of the bounded target motion.

No scene, trajectory, cache key, solver input or estimator setting is modified.
The frozen package's nominal clearance formula already includes 2*r_motion;
its old assertion subtracted that allowance again. This explicit overlay uses
the nominal bound once and checks actual states against 2*r_target+margin.
"""
import numpy as np

VERSION='nominal-motion-margin-once-20261017'


def assert_geometry_rules(scene, traj=None):
    from fdtd import scenes
    rule=scenes.separation_rule(scene)
    centers=scenes.target_centers_2d(scene)
    count=len(centers)
    nominal=min((float(np.linalg.norm(centers[i]-centers[j])) for i in range(count)
                 for j in range(i+1,count)),default=float('inf'))
    motion=rule['trajectory_radius_max_m']
    physical=2*rule['target_radius_m']+rule['overlap_margin_m']
    worst=nominal-2*motion if count>1 else float('inf')
    problems=[]
    if count>1 and nominal<rule['required_min_separation_m']:
        problems.append('Nominal spacing cannot guarantee nonoverlap for the motion bound')
    minimum=float('inf')
    if traj is not None:
        array=np.asarray(traj,dtype=float)
        if array.ndim!=3 or array.shape[1:]!=centers.shape or not np.isfinite(array).all():
            raise ValueError('Invalid target trajectory')
        radii=np.asarray(scenes._traj_radius_list(scene))
        if np.any(np.linalg.norm(array-centers[None,:,:],axis=2)>radii[None,:]+1e-12):
            raise ValueError('Trajectory exceeds its declared bound')
        minimum=min((float(np.linalg.norm(array[:,i]-array[:,j],axis=1).min())
                     for i in range(count) for j in range(i+1,count)),default=float('inf'))
        if count>1 and minimum<physical:
            problems.append('Actual target states overlap or violate the physical margin')
    reach=rule['target_radius_m']+motion
    inner=scene['domain']['inner_bounds_m']
    slack=min(min(float(p[0]-reach-inner['x'][0]),float(inner['x'][1]-p[0]-reach),
                  float(p[1]-reach-inner['y'][0]),float(inner['y'][1]-p[1]-reach)) for p in centers)
    if slack<0:
        problems.append('Target including its motion bound exceeds the inner domain')
    if problems:
        raise ValueError('Geometry rejected: '+ '; '.join(problems))
    return {'rule':rule,'min_nominal_separation_m':None if count<2 else nominal,
            'worst_case_separation_m':None if count<2 else worst,
            'min_state_separation_m':None if minimum==float('inf') else minimum,
            'min_inner_slack_m':slack,'physical_state_clearance_required_m':physical,
            'validation_overlay':VERSION}


def install():
    from fdtd import scenes
    scenes.assert_geometry_rules=assert_geometry_rules
