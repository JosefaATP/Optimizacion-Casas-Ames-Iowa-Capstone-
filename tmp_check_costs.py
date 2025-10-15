from optimization.remodel.model_spec import CostTables
ct = CostTables.from_yaml('costs/materials.yaml')
print('central_air_install =', ct.central_air_install)
print('cost_finish_bsmt_ft2 =', ct.cost_finish_bsmt_ft2)
print('cost_pool_ft2 =', ct.cost_pool_ft2)
print('fence_build_psf =', getattr(ct,'fence_build_psf',None))
print('BsmtCond keys =', list(ct.bsmt_cond.keys())[:8])
print('PoolQC keys =', list(ct.pool_qc.keys())[:8])
