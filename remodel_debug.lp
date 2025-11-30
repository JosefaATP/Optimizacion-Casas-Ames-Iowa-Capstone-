\ Model remodel_embed
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xfacc2ccccad465ca
Maximize
  0 x_delta_KitchenQual_TA + 0 x_delta_KitchenQual_EX + 0 x_u_util_ELO
   + 0 x_u_util_NoSeWa + 0 x_u_util_NoSewr + 0 x_u_util_AllPub
   + 0 x_Roof_Matl + 0 x_UpgGarage + 0 x_poolqc_is_Ex + 0 x_poolqc_is_Gd
   + 0 x_poolqc_is_TA + 0 x_poolqc_is_Fa + 0 x_poolqc_is_Po
   + 0 x_poolqc_is_No_aplica + 0 x_upg_pool_qc + 0 x_z10_GarageArea
   + 0 x_z20_GarageArea + 0 x_z30_GarageArea + 0 x_z10_WoodDeckSF
   + 0 x_z20_WoodDeckSF + 0 x_z30_WoodDeckSF + 0 x_z10_OpenPorchSF
   + 0 x_z20_OpenPorchSF + 0 x_z30_OpenPorchSF + 0 x_z10_EnclosedPorch
   + 0 x_z20_EnclosedPorch + 0 x_z30_EnclosedPorch + 0 x_z10_3SsnPorch
   + 0 x_z20_3SsnPorch + 0 x_z30_3SsnPorch + 0 x_z10_ScreenPorch
   + 0 x_z20_ScreenPorch + 0 x_z30_ScreenPorch + 0 x_z10_PoolArea
   + 0 x_z20_PoolArea + 0 x_z30_PoolArea + 0 x_Heating_Floor
   + 0 x_Heating_GasA + 0 x_Heating_GasW + 0 x_Heating_Grav
   + 0 x_Heating_OthW + 0 x_Heating_Wall + 0 x_b1_is_No_aplica
   + 0 x_b2_is_No_aplica + 0 B1_upg_flag + 0 B2_upg_flag + y_price
   - cost_model - 240266.984375 Constant
Subject To
 OverallQual_from_upgrades_float: - 0.0714285714285714 x_Kitchen_Qual
   - 0.0714285714285714 x_Exter_Qual - 0.0714285714285714 x_Exter_Cond
   - 0.0714285714285714 x_Heating_QC - 0.0714285714285714 x_Bsmt_Cond
   - 0.0714285714285714 x_Garage_Qual - 0.0714285714285714 x_Garage_Cond
   + Overall_Qual_float = 4.928571428571429
 OverallQual_floor_ub: - Overall_Qual_float + Overall_Qual_calc <= 0
 OverallQual_floor_lb: - Overall_Qual_float + Overall_Qual_calc >= -1
 Kitchen_Qual_upgrade_only: x_Kitchen_Qual >= 3
 Exter_Qual_upgrade_only: x_Exter_Qual >= 2
 Exter_Cond_upgrade_only: x_Exter_Cond >= 2
 Heating_QC_upgrade_only: x_Heating_QC >= 2
 Bsmt_Cond_upgrade_only: x_Bsmt_Cond >= 2
 Garage_Qual_upgrade_only: x_Garage_Qual >= 2
 Garage_Cond_upgrade_only: x_Garage_Cond >= 2
 KIT_onehot: kit_is_Po + kit_is_Fa + kit_is_TA + kit_is_Gd + kit_is_Ex = 1
 KIT_link_int: x_Kitchen_Qual - kit_is_Fa - 2 kit_is_TA - 3 kit_is_Gd
   - 4 kit_is_Ex = 0
 EXT_ex1_pick_one: x_ex1_is_AsbShng + x_ex1_is_AsphShn + x_ex1_is_BrkComm
   + x_ex1_is_BrkFace + x_ex1_is_CBlock + x_ex1_is_CemntBd
   + x_ex1_is_HdBoard + x_ex1_is_ImStucc + x_ex1_is_MetalSd
   + x_ex1_is_Other + x_ex1_is_Plywood + x_ex1_is_PreCast + x_ex1_is_Stone
   + x_ex1_is_Stucco + x_ex1_is_VinylSd + x_ex1_is_Wd_Sdng
   + x_ex1_is_WdShngl = 1
 EXT_ex2_pick_ilas2: x_ex2_is_AsbShng + x_ex2_is_AsphShn + x_ex2_is_BrkComm
   + x_ex2_is_BrkFace + x_ex2_is_CBlock + x_ex2_is_CemntBd
   + x_ex2_is_HdBoard + x_ex2_is_ImStucc + x_ex2_is_MetalSd
   + x_ex2_is_Other + x_ex2_is_Plywood + x_ex2_is_PreCast + x_ex2_is_Stone
   + x_ex2_is_Stucco + x_ex2_is_VinylSd + x_ex2_is_Wd_Sdng
   + x_ex2_is_WdShngl = 1
 EXT_EQ_onehot: exterqual_is_Po + exterqual_is_Fa + exterqual_is_TA
   + exterqual_is_Gd + exterqual_is_Ex = 1
 EXT_EC_onehot: extercond_is_Po + extercond_is_Fa + extercond_is_TA
   + extercond_is_Gd + extercond_is_Ex = 1
 EXT_EQ_link_int: x_Exter_Qual - exterqual_is_Fa - 2 exterqual_is_TA
   - 3 exterqual_is_Gd - 4 exterqual_is_Ex = 0
 EXT_EC_link_int: x_Exter_Cond - extercond_is_Fa - 2 extercond_is_TA
   - 3 extercond_is_Gd - 4 extercond_is_Ex = 0
 EXT_exclusive_paths: ext_upg_material + ext_upg_qc <= 1
 MVT_pick_one: x_mvt_is_BrkCmn + x_mvt_is_BrkFace + x_mvt_is_CBlock
   + x_mvt_is_No_aplica + x_mvt_is_Stone = 1
 MVT_forbid_noaplica: x_mvt_is_No_aplica = 0
 MVT_forbid_cheaper_BrkCmn: x_mvt_is_BrkCmn = 0
 MVT_area_no_decrease: x_Mas_Vnr_Area >= 108
 MVT_lin_p_le_a_BrkCmn: - x_Mas_Vnr_Area + mv_area_mvt_BrkCmn <= 0
 MVT_lin_p_le_UBb_BrkCmn: - 2000 x_mvt_is_BrkCmn + mv_area_mvt_BrkCmn <= 0
 MVT_lin_p_ge_a_minus_UB1b_BrkCmn: - x_Mas_Vnr_Area - 2000 x_mvt_is_BrkCmn
   + mv_area_mvt_BrkCmn >= -2000
 MVT_lin_p_le_a_CBlock: - x_Mas_Vnr_Area + mv_area_mvt_CBlock <= 0
 MVT_lin_p_le_UBb_CBlock: - 2000 x_mvt_is_CBlock + mv_area_mvt_CBlock <= 0
 MVT_lin_p_ge_a_minus_UB1b_CBlock: - x_Mas_Vnr_Area - 2000 x_mvt_is_CBlock
   + mv_area_mvt_CBlock >= -2000
 MVT_lin_p_le_a_Stone: - x_Mas_Vnr_Area + mv_area_mvt_Stone <= 0
 MVT_lin_p_le_UBb_Stone: - 2000 x_mvt_is_Stone + mv_area_mvt_Stone <= 0
 MVT_lin_p_ge_a_minus_UB1b_Stone: - x_Mas_Vnr_Area - 2000 x_mvt_is_Stone
   + mv_area_mvt_Stone >= -2000
 ROOF_pick_one_matl: x_roof_matl_is_ClyTile + x_roof_matl_is_CompShg
   + x_roof_matl_is_Membran + x_roof_matl_is_Metal + x_roof_matl_is_Roll
   + x_roof_matl_is_Tar&Grv + x_roof_matl_is_WdShake
   + x_roof_matl_is_WdShngl = 1
 ROOF_incompat_Hip_Membran: x_roof_matl_is_Membran = 0
 ROOF_change_eq_sum_nonbase: - x_roof_matl_is_ClyTile
   - x_roof_matl_is_Membran - x_roof_matl_is_Metal - x_roof_matl_is_Roll
   - x_roof_matl_is_Tar&Grv - x_roof_matl_is_WdShake
   - x_roof_matl_is_WdShngl + roof_change_mat = 0
 GaFin_pick_one: x_garage_finish_is_Fin + x_garage_finish_is_RFn
   + x_garage_finish_is_Unf + x_garage_finish_is_No_aplica = 1
 UpgGa_allowed_if_eligible: x_UpgGarage <= 1
 GaFin_change_only_if_upg: x_garage_finish_is_Fin + x_garage_finish_is_RFn
   + x_garage_finish_is_No_aplica - x_UpgGarage <= 0
 GaFin_Fin_requires_Upg: x_garage_finish_is_Fin - x_UpgGarage <= 0
 GaFin_RFnUnf_disallowed_when_upg: x_garage_finish_is_RFn
   + x_garage_finish_is_Unf + x_UpgGarage <= 1
 CentralAir_onehot: central_air_yes + central_air_no = 1
 PoolQC_pick_one: poolqc_is_Po + poolqc_is_Fa + poolqc_is_TA + poolqc_is_Gd
   + poolqc_is_Ex + poolqc_is_No_aplica = 1
 PoolQC_link_ord: x_Pool_QC - poolqc_is_Fa - 2 poolqc_is_TA
   - 3 poolqc_is_Gd - 4 poolqc_is_Ex + poolqc_is_No_aplica = 0
 PoolQC_eligibility: UpgPool <= 0
 AMPL_one_scale_GarageArea: x_z10_GarageArea + x_z20_GarageArea
   + x_z30_GarageArea <= 1
 AMPL_one_scale_WoodDeckSF: x_z10_WoodDeckSF + x_z20_WoodDeckSF
   + x_z30_WoodDeckSF <= 1
 AMPL_one_scale_OpenPorchSF: x_z10_OpenPorchSF + x_z20_OpenPorchSF
   + x_z30_OpenPorchSF <= 1
 AMPL_one_scale_EnclosedPorch: x_z10_EnclosedPorch + x_z20_EnclosedPorch
   + x_z30_EnclosedPorch <= 1
 AMPL_one_scale_3SsnPorch: x_z10_3SsnPorch + x_z20_3SsnPorch
   + x_z30_3SsnPorch <= 1
 AMPL_one_scale_ScreenPorch: x_z10_ScreenPorch + x_z20_ScreenPorch
   + x_z30_ScreenPorch <= 1
 AMPL_one_scale_PoolArea: x_z10_PoolArea + x_z20_PoolArea + x_z30_PoolArea
   <= 1
 AREA_libre_no_negativa: - 40 x_AddFull - 20 x_AddHalf - 75 x_AddKitch
   - 70 x_AddBed - 31.2 x_z10_GarageArea - 62.4 x_z20_GarageArea
   - 93.6 x_z30_GarageArea - 39.3 x_z10_WoodDeckSF - 78.6 x_z20_WoodDeckSF
   - 117.9 x_z30_WoodDeckSF - 3.6 x_z10_OpenPorchSF - 7.2 x_z20_OpenPorchSF
   - 10.8 x_z30_OpenPorchSF >= -12197
 AMPL_link_GarageArea: x_Garage_Area - 31.2 x_z10_GarageArea
   - 62.4 x_z20_GarageArea - 93.6 x_z30_GarageArea = 312
 AMPL_link_WoodDeckSF: x_Wood_Deck_SF - 39.3 x_z10_WoodDeckSF
   - 78.6 x_z20_WoodDeckSF - 117.9 x_z30_WoodDeckSF = 393
 AMPL_link_OpenPorchSF: x_Open_Porch_SF - 3.6 x_z10_OpenPorchSF
   - 7.2 x_z20_OpenPorchSF - 10.8 x_z30_OpenPorchSF = 36
 AMPL_link_EnclosedPorch: x_Enclosed_Porch = 0
 AMPL_link_3SsnPorch: x_3Ssn_Porch = 0
 AMPL_link_ScreenPorch: x_Screen_Porch = 0
 AMPL_link_PoolArea: x_Pool_Area = 0
 AMPL_link_1stflr: x_1st_Flr_SF - 40 x_AddFull - 20 x_AddHalf
   - 75 x_AddKitch - 70 x_AddBed = 1329
 AMPL_1stflr_upgrade: x_1st_Flr_SF - 40 x_Add1stFlr >= 1329
 COUNT_fullbath: x_Full_Bath - x_AddFull = 1
 COUNT_halfbath: - x_AddHalf + x_Half_Bath = 1
 COUNT_bedroom: x_Bedroom_AbvGr - x_AddBed = 3
 COUNT_kitchen: - x_AddKitch + x_Kitchen_AbvGr = 1
 COUNT_totrooms: x_TotRms_AbvGrd - x_AddKitch - x_AddBed = 6
 GQ_pick_one: x_garage_qual_is_Ex + x_garage_qual_is_Gd
   + x_garage_qual_is_TA + x_garage_qual_is_Fa + x_garage_qual_is_Po
   + x_garage_qual_is_No_aplica = 1
 GC_pick_one: x_garage_cond_is_Ex + x_garage_cond_is_Gd
   + x_garage_cond_is_TA + x_garage_cond_is_Fa + x_garage_cond_is_Po
   + x_garage_cond_is_No_aplica = 1
 PAVED_pick_one: x_paved_drive_is_Y + x_paved_drive_is_P
   + x_paved_drive_is_N = 1
 FENCE_pick_one: x_fence_is_GdPrv + x_fence_is_MnPrv + x_fence_is_GdWo
   + x_fence_is_MnWw + x_fence_is_No_aplica = 1
 ELEC_pick_one_full: x_elect_is_SBrkr + x_elect_is_FuseA + x_elect_is_FuseF
   + x_elect_is_FuseP + x_elect_is_Mix = 1
 HEAT_pick_one_type: heat_is_Floor + heat_is_GasA + heat_is_GasW
   + heat_is_Grav + heat_is_OthW + heat_is_Wall = 1
 HEAT_qc_upgrade_only_if_eligible: x_heat_upg_qc <= 1
 HEAT_qc_upgrade_only: x_Heating_QC >= 2
 R74: - x_heat_upg_type + HEAT_any_rebuild >= 0
 R75: - x_heat_upg_qc + HEAT_any_rebuild >= 0
 R76: - x_heat_upg_type - x_heat_upg_qc + HEAT_any_rebuild <= 0
 HEAT_qc_min_if_any_rebuild: x_Heating_QC - 2 HEAT_any_rebuild >= 0
 HEAT_change_def: - heat_is_Floor - heat_is_GasW - heat_is_Grav
   - heat_is_OthW - heat_is_Wall + heat_change_type = 0
 HEAT_qc_onehot: heat_qc_is_Po + heat_qc_is_Fa + heat_qc_is_TA
   + heat_qc_is_Gd + heat_qc_is_Ex = 1
 HEAT_qc_link: x_Heating_QC - heat_qc_is_Fa - 2 heat_qc_is_TA
   - 3 heat_qc_is_Gd - 4 heat_qc_is_Ex = 0
 HEAT_exclusive_paths: heat_upg_type_flag + heat_upg_qc_flag <= 1
 FQ_pick_one: fireplace_is_Po + fireplace_is_Fa + fireplace_is_TA
   + fireplace_is_Gd + fireplace_is_Ex + fireplace_is_No_aplica = 1
 FQ_level_match: x_Fireplace_Qu - fireplace_is_Fa - 2 fireplace_is_TA
   - 3 fireplace_is_Gd - 4 fireplace_is_Ex + fireplace_is_No_aplica = 0
 BSMT_all_or_nothing: - 406 bsmt_finish_all + bsmt_to_fin1 + bsmt_to_fin2
   = 0
 BSMT_link_fin1: x_BsmtFin_SF_1 - bsmt_to_fin1 = 923
 BSMT_link_fin2: x_BsmtFin_SF_2 - bsmt_to_fin2 = 0
 BSMT_link_unf: x_Bsmt_Unf_SF + 406 bsmt_finish_all = 406
 BSMT_conservation_sum: x_BsmtFin_SF_1 + x_BsmtFin_SF_2 + x_Bsmt_Unf_SF
   = 1329
 BSMTCOND_onehot: bsmtcond_is_Po + bsmtcond_is_Fa + bsmtcond_is_TA
   + bsmtcond_is_Gd + bsmtcond_is_Ex = 1
 BSMTCOND_link_int: x_Bsmt_Cond - bsmtcond_is_Fa - 2 bsmtcond_is_TA
   - 3 bsmtcond_is_Gd - 4 bsmtcond_is_Ex = 0
 UTIL_onehot: util_ELO + util_NoSeWa + util_NoSewr + util_AllPub = 1
 UTIL_link_int: x_Utilities - util_NoSeWa - 2 util_NoSewr - 3 util_AllPub
   = 0
 R2_grliv_le_lot: x_Gr_Liv_Area <= 14267
 R4_baths_le_bedrooms: - x_Bedroom_AbvGr + x_Full_Bath + x_Half_Bath <= 0
 R5_min_fullbath: x_Full_Bath >= 1
 R5_min_bedrooms: x_Bedroom_AbvGr >= 1
 R5_min_kitchen: x_Kitchen_AbvGr >= 1
 R7_gr_liv_equality: x_Gr_Liv_Area - x_1st_Flr_SF = 0
 R8_rooms_balance: - x_Bedroom_AbvGr + x_TotRms_AbvGrd - x_Kitchen_AbvGr
   - R8_other_rooms = 0
 TREE_0_ONE_PATH: t0_path0 + t0_path1 + t0_path2 + t0_path3 + t0_path4
   + t0_path5 + t0_path6 + t0_path7 = 1
 TREE_1_ONE_PATH: t1_path0 + t1_path1 + t1_path2 + t1_path3 + t1_path4
   + t1_path5 + t1_path6 + t1_path7 = 1
 TREE_2_ONE_PATH: t2_path0 + t2_path1 + t2_path2 + t2_path3 + t2_path4
   + t2_path5 + t2_path6 + t2_path7 = 1
 TREE_3_ONE_PATH: t3_path0 + t3_path1 + t3_path2 + t3_path3 + t3_path4
   + t3_path5 + t3_path6 + t3_path7 = 1
 TREE_4_ONE_PATH: t4_path0 + t4_path1 + t4_path2 + t4_path3 + t4_path4
   + t4_path5 + t4_path6 + t4_path7 = 1
 TREE_5_ONE_PATH: t5_path0 + t5_path1 + t5_path2 + t5_path3 + t5_path4
   + t5_path5 + t5_path6 + t5_path7 = 1
 TREE_6_ONE_PATH: t6_path0 + t6_path1 + t6_path2 + t6_path3 + t6_path4
   + t6_path5 + t6_path6 + t6_path7 = 1
 TREE_7_ONE_PATH: t7_path0 + t7_path1 + t7_path2 + t7_path3 + t7_path4
   + t7_path5 + t7_path6 + t7_path7 = 1
 TREE_8_ONE_PATH: t8_path0 + t8_path1 + t8_path2 + t8_path3 + t8_path4
   + t8_path5 + t8_path6 + t8_path7 = 1
 TREE_9_ONE_PATH: t9_path0 + t9_path1 + t9_path2 + t9_path3 + t9_path4
   + t9_path5 + t9_path6 + t9_path7 = 1
 TREE_10_ONE_PATH: t10_path0 + t10_path1 + t10_path2 + t10_path3
   + t10_path4 + t10_path5 + t10_path6 + t10_path7 = 1
 TREE_11_ONE_PATH: t11_path0 + t11_path1 + t11_path2 + t11_path3
   + t11_path4 + t11_path5 + t11_path6 + t11_path7 = 1
 TREE_12_ONE_PATH: t12_path0 + t12_path1 + t12_path2 + t12_path3
   + t12_path4 + t12_path5 + t12_path6 + t12_path7 = 1
 TREE_13_ONE_PATH: t13_path0 + t13_path1 + t13_path2 + t13_path3
   + t13_path4 + t13_path5 + t13_path6 + t13_path7 = 1
 TREE_14_ONE_PATH: t14_path0 + t14_path1 + t14_path2 + t14_path3
   + t14_path4 + t14_path5 + t14_path6 + t14_path7 = 1
 TREE_15_ONE_PATH: t15_path0 + t15_path1 + t15_path2 + t15_path3
   + t15_path4 + t15_path5 + t15_path6 + t15_path7 = 1
 TREE_16_ONE_PATH: t16_path0 + t16_path1 + t16_path2 + t16_path3
   + t16_path4 + t16_path5 + t16_path6 + t16_path7 = 1
 TREE_17_ONE_PATH: t17_path0 + t17_path1 + t17_path2 + t17_path3
   + t17_path4 + t17_path5 + t17_path6 = 1
 TREE_18_ONE_PATH: t18_path0 + t18_path1 + t18_path2 + t18_path3
   + t18_path4 + t18_path5 + t18_path6 + t18_path7 = 1
 TREE_19_ONE_PATH: t19_path0 + t19_path1 + t19_path2 + t19_path3
   + t19_path4 + t19_path5 + t19_path6 + t19_path7 = 1
 TREE_20_ONE_PATH: t20_path0 + t20_path1 + t20_path2 + t20_path3
   + t20_path4 + t20_path5 + t20_path6 + t20_path7 = 1
 TREE_21_ONE_PATH: t21_path0 + t21_path1 + t21_path2 + t21_path3
   + t21_path4 + t21_path5 + t21_path6 + t21_path7 = 1
 TREE_22_ONE_PATH: t22_path0 + t22_path1 + t22_path2 + t22_path3
   + t22_path4 + t22_path5 + t22_path6 + t22_path7 = 1
 TREE_23_ONE_PATH: t23_path0 + t23_path1 + t23_path2 + t23_path3
   + t23_path4 + t23_path5 + t23_path6 + t23_path7 = 1
 TREE_24_ONE_PATH: t24_path0 + t24_path1 + t24_path2 + t24_path3
   + t24_path4 + t24_path5 + t24_path6 + t24_path7 = 1
 TREE_25_ONE_PATH: t25_path0 + t25_path1 + t25_path2 + t25_path3
   + t25_path4 + t25_path5 + t25_path6 + t25_path7 = 1
 TREE_26_ONE_PATH: t26_path0 + t26_path1 + t26_path2 + t26_path3
   + t26_path4 + t26_path5 + t26_path6 + t26_path7 = 1
 TREE_27_ONE_PATH: t27_path0 + t27_path1 + t27_path2 + t27_path3
   + t27_path4 + t27_path5 + t27_path6 + t27_path7 = 1
 TREE_28_ONE_PATH: t28_path0 + t28_path1 + t28_path2 + t28_path3
   + t28_path4 + t28_path5 + t28_path6 + t28_path7 = 1
 TREE_29_ONE_PATH: t29_path0 + t29_path1 + t29_path2 + t29_path3
   + t29_path4 + t29_path5 + t29_path6 + t29_path7 = 1
 TREE_30_ONE_PATH: t30_path0 + t30_path1 + t30_path2 + t30_path3
   + t30_path4 + t30_path5 + t30_path6 + t30_path7 = 1
 TREE_31_ONE_PATH: t31_path0 + t31_path1 + t31_path2 + t31_path3
   + t31_path4 + t31_path5 + t31_path6 + t31_path7 = 1
 TREE_32_ONE_PATH: t32_path0 + t32_path1 + t32_path2 + t32_path3
   + t32_path4 + t32_path5 + t32_path6 + t32_path7 = 1
 TREE_33_ONE_PATH: t33_path0 + t33_path1 + t33_path2 + t33_path3
   + t33_path4 + t33_path5 + t33_path6 + t33_path7 = 1
 TREE_34_ONE_PATH: t34_path0 + t34_path1 + t34_path2 + t34_path3
   + t34_path4 + t34_path5 + t34_path6 + t34_path7 = 1
 TREE_35_ONE_PATH: t35_path0 + t35_path1 + t35_path2 + t35_path3
   + t35_path4 + t35_path5 + t35_path6 + t35_path7 = 1
 TREE_36_ONE_PATH: t36_path0 + t36_path1 + t36_path2 + t36_path3
   + t36_path4 + t36_path5 + t36_path6 + t36_path7 = 1
 TREE_37_ONE_PATH: t37_path0 + t37_path1 + t37_path2 + t37_path3
   + t37_path4 + t37_path5 + t37_path6 + t37_path7 = 1
 TREE_38_ONE_PATH: t38_path0 + t38_path1 + t38_path2 + t38_path3
   + t38_path4 + t38_path5 + t38_path6 + t38_path7 = 1
 TREE_39_ONE_PATH: t39_path0 + t39_path1 + t39_path2 + t39_path3
   + t39_path4 + t39_path5 + t39_path6 + t39_path7 = 1
 TREE_40_ONE_PATH: t40_path0 + t40_path1 + t40_path2 + t40_path3
   + t40_path4 + t40_path5 + t40_path6 + t40_path7 = 1
 TREE_41_ONE_PATH: t41_path0 + t41_path1 + t41_path2 + t41_path3
   + t41_path4 + t41_path5 + t41_path6 + t41_path7 = 1
 TREE_42_ONE_PATH: t42_path0 + t42_path1 + t42_path2 + t42_path3
   + t42_path4 + t42_path5 + t42_path6 + t42_path7 = 1
 TREE_43_ONE_PATH: t43_path0 + t43_path1 + t43_path2 + t43_path3
   + t43_path4 + t43_path5 + t43_path6 + t43_path7 = 1
 TREE_44_ONE_PATH: t44_path0 + t44_path1 + t44_path2 + t44_path3
   + t44_path4 + t44_path5 + t44_path6 + t44_path7 = 1
 TREE_45_ONE_PATH: t45_path0 + t45_path1 + t45_path2 + t45_path3
   + t45_path4 + t45_path5 + t45_path6 + t45_path7 = 1
 TREE_46_ONE_PATH: t46_path0 + t46_path1 + t46_path2 + t46_path3
   + t46_path4 + t46_path5 + t46_path6 + t46_path7 = 1
 TREE_47_ONE_PATH: t47_path0 + t47_path1 + t47_path2 + t47_path3
   + t47_path4 + t47_path5 + t47_path6 + t47_path7 = 1
 TREE_48_ONE_PATH: t48_path0 + t48_path1 + t48_path2 + t48_path3
   + t48_path4 + t48_path5 + t48_path6 + t48_path7 = 1
 TREE_49_ONE_PATH: t49_path0 + t49_path1 + t49_path2 + t49_path3
   + t49_path4 + t49_path5 + t49_path6 + t49_path7 = 1
 TREE_50_ONE_PATH: t50_path0 + t50_path1 + t50_path2 + t50_path3
   + t50_path4 + t50_path5 + t50_path6 = 1
 TREE_51_ONE_PATH: t51_path0 + t51_path1 + t51_path2 + t51_path3
   + t51_path4 + t51_path5 + t51_path6 = 1
 TREE_52_ONE_PATH: t52_path0 + t52_path1 + t52_path2 + t52_path3
   + t52_path4 + t52_path5 + t52_path6 + t52_path7 = 1
 TREE_53_ONE_PATH: t53_path0 + t53_path1 + t53_path2 + t53_path3
   + t53_path4 + t53_path5 + t53_path6 = 1
 TREE_54_ONE_PATH: t54_path0 + t54_path1 + t54_path2 + t54_path3
   + t54_path4 + t54_path5 + t54_path6 = 1
 TREE_55_ONE_PATH: t55_path0 + t55_path1 + t55_path2 + t55_path3
   + t55_path4 + t55_path5 + t55_path6 + t55_path7 = 1
 TREE_56_ONE_PATH: t56_path0 + t56_path1 + t56_path2 + t56_path3
   + t56_path4 + t56_path5 + t56_path6 + t56_path7 = 1
 TREE_57_ONE_PATH: t57_path0 + t57_path1 + t57_path2 + t57_path3
   + t57_path4 + t57_path5 + t57_path6 + t57_path7 = 1
 TREE_58_ONE_PATH: t58_path0 + t58_path1 + t58_path2 + t58_path3
   + t58_path4 + t58_path5 + t58_path6 + t58_path7 = 1
 TREE_59_ONE_PATH: t59_path0 + t59_path1 + t59_path2 + t59_path3
   + t59_path4 + t59_path5 + t59_path6 + t59_path7 = 1
 TREE_60_ONE_PATH: t60_path0 + t60_path1 + t60_path2 + t60_path3
   + t60_path4 + t60_path5 + t60_path6 = 1
 TREE_61_ONE_PATH: t61_path0 + t61_path1 + t61_path2 + t61_path3
   + t61_path4 + t61_path5 + t61_path6 = 1
 TREE_62_ONE_PATH: t62_path0 + t62_path1 + t62_path2 + t62_path3
   + t62_path4 + t62_path5 + t62_path6 = 1
 TREE_63_ONE_PATH: t63_path0 + t63_path1 + t63_path2 + t63_path3
   + t63_path4 + t63_path5 + t63_path6 + t63_path7 = 1
 TREE_64_ONE_PATH: t64_path0 + t64_path1 + t64_path2 + t64_path3
   + t64_path4 + t64_path5 + t64_path6 + t64_path7 = 1
 TREE_65_ONE_PATH: t65_path0 + t65_path1 + t65_path2 + t65_path3
   + t65_path4 + t65_path5 + t65_path6 + t65_path7 = 1
 TREE_66_ONE_PATH: t66_path0 + t66_path1 + t66_path2 + t66_path3
   + t66_path4 + t66_path5 + t66_path6 + t66_path7 = 1
 TREE_67_ONE_PATH: t67_path0 + t67_path1 + t67_path2 + t67_path3
   + t67_path4 + t67_path5 = 1
 TREE_68_ONE_PATH: t68_path0 + t68_path1 + t68_path2 + t68_path3
   + t68_path4 + t68_path5 + t68_path6 + t68_path7 = 1
 TREE_69_ONE_PATH: t69_path0 + t69_path1 + t69_path2 + t69_path3
   + t69_path4 + t69_path5 + t69_path6 = 1
 TREE_70_ONE_PATH: t70_path0 + t70_path1 + t70_path2 + t70_path3
   + t70_path4 + t70_path5 + t70_path6 + t70_path7 = 1
 TREE_71_ONE_PATH: t71_path0 + t71_path1 + t71_path2 + t71_path3
   + t71_path4 + t71_path5 + t71_path6 + t71_path7 = 1
 TREE_72_ONE_PATH: t72_path0 + t72_path1 + t72_path2 + t72_path3
   + t72_path4 + t72_path5 + t72_path6 + t72_path7 = 1
 TREE_73_ONE_PATH: t73_path0 + t73_path1 + t73_path2 + t73_path3
   + t73_path4 + t73_path5 + t73_path6 = 1
 TREE_74_ONE_PATH: t74_path0 + t74_path1 + t74_path2 + t74_path3
   + t74_path4 + t74_path5 + t74_path6 = 1
 TREE_75_ONE_PATH: t75_path0 + t75_path1 + t75_path2 + t75_path3
   + t75_path4 + t75_path5 + t75_path6 = 1
 TREE_76_ONE_PATH: t76_path0 + t76_path1 + t76_path2 + t76_path3
   + t76_path4 + t76_path5 + t76_path6 + t76_path7 = 1
 TREE_77_ONE_PATH: t77_path0 + t77_path1 + t77_path2 + t77_path3
   + t77_path4 + t77_path5 + t77_path6 = 1
 TREE_78_ONE_PATH: t78_path0 + t78_path1 + t78_path2 + t78_path3
   + t78_path4 + t78_path5 + t78_path6 + t78_path7 = 1
 TREE_79_ONE_PATH: t79_path0 + t79_path1 + t79_path2 + t79_path3
   + t79_path4 + t79_path5 + t79_path6 = 1
 TREE_80_ONE_PATH: t80_path0 + t80_path1 + t80_path2 + t80_path3
   + t80_path4 + t80_path5 + t80_path6 = 1
 TREE_81_ONE_PATH: t81_path0 + t81_path1 + t81_path2 + t81_path3
   + t81_path4 = 1
 TREE_82_ONE_PATH: t82_path0 + t82_path1 + t82_path2 + t82_path3
   + t82_path4 + t82_path5 + t82_path6 = 1
 TREE_83_ONE_PATH: t83_path0 + t83_path1 + t83_path2 + t83_path3
   + t83_path4 = 1
 TREE_84_ONE_PATH: t84_path0 + t84_path1 + t84_path2 + t84_path3
   + t84_path4 + t84_path5 + t84_path6 = 1
 TREE_85_ONE_PATH: t85_path0 + t85_path1 + t85_path2 + t85_path3
   + t85_path4 + t85_path5 + t85_path6 = 1
 TREE_86_ONE_PATH: t86_path0 + t86_path1 + t86_path2 + t86_path3
   + t86_path4 + t86_path5 + t86_path6 = 1
 TREE_87_ONE_PATH: t87_path0 + t87_path1 + t87_path2 + t87_path3
   + t87_path4 + t87_path5 + t87_path6 = 1
 TREE_88_ONE_PATH: t88_path0 + t88_path1 + t88_path2 + t88_path3
   + t88_path4 + t88_path5 + t88_path6 = 1
 TREE_89_ONE_PATH: t89_path0 + t89_path1 + t89_path2 + t89_path3
   + t89_path4 + t89_path5 + t89_path6 + t89_path7 = 1
 TREE_90_ONE_PATH: t90_path0 + t90_path1 + t90_path2 + t90_path3
   + t90_path4 + t90_path5 + t90_path6 + t90_path7 = 1
 TREE_91_ONE_PATH: t91_path0 + t91_path1 + t91_path2 + t91_path3
   + t91_path4 + t91_path5 + t91_path6 + t91_path7 = 1
 TREE_92_ONE_PATH: t92_path0 + t92_path1 + t92_path2 + t92_path3
   + t92_path4 + t92_path5 + t92_path6 = 1
 TREE_93_ONE_PATH: t93_path0 + t93_path1 + t93_path2 + t93_path3
   + t93_path4 + t93_path5 + t93_path6 + t93_path7 = 1
 TREE_94_ONE_PATH: t94_path0 + t94_path1 + t94_path2 + t94_path3
   + t94_path4 + t94_path5 + t94_path6 = 1
 TREE_95_ONE_PATH: t95_path0 + t95_path1 + t95_path2 + t95_path3
   + t95_path4 + t95_path5 + t95_path6 + t95_path7 = 1
 TREE_96_ONE_PATH: t96_path0 + t96_path1 + t96_path2 + t96_path3
   + t96_path4 + t96_path5 + t96_path6 = 1
 TREE_97_ONE_PATH: t97_path0 + t97_path1 + t97_path2 + t97_path3
   + t97_path4 + t97_path5 + t97_path6 + t97_path7 = 1
 TREE_98_ONE_PATH: t98_path0 + t98_path1 + t98_path2 + t98_path3
   + t98_path4 + t98_path5 + t98_path6 = 1
 TREE_99_ONE_PATH: t99_path0 + t99_path1 + t99_path2 + t99_path3
   + t99_path4 + t99_path5 + t99_path6 = 1
 TREE_100_ONE_PATH: t100_path0 + t100_path1 + t100_path2 + t100_path3
   + t100_path4 + t100_path5 + t100_path6 + t100_path7 = 1
 TREE_101_ONE_PATH: t101_path0 + t101_path1 + t101_path2 + t101_path3
   + t101_path4 + t101_path5 + t101_path6 = 1
 TREE_102_ONE_PATH: t102_path0 + t102_path1 + t102_path2 + t102_path3
   + t102_path4 + t102_path5 + t102_path6 + t102_path7 = 1
 TREE_103_ONE_PATH: t103_path0 + t103_path1 + t103_path2 + t103_path3
   + t103_path4 + t103_path5 = 1
 TREE_104_ONE_PATH: t104_path0 + t104_path1 + t104_path2 + t104_path3
   + t104_path4 + t104_path5 = 1
 TREE_105_ONE_PATH: t105_path0 + t105_path1 + t105_path2 + t105_path3
   + t105_path4 + t105_path5 + t105_path6 + t105_path7 = 1
 TREE_106_ONE_PATH: t106_path0 + t106_path1 + t106_path2 + t106_path3
   + t106_path4 + t106_path5 + t106_path6 = 1
 TREE_107_ONE_PATH: t107_path0 + t107_path1 + t107_path2 + t107_path3
   + t107_path4 + t107_path5 + t107_path6 + t107_path7 = 1
 TREE_108_ONE_PATH: t108_path0 + t108_path1 + t108_path2 + t108_path3
   + t108_path4 + t108_path5 + t108_path6 + t108_path7 = 1
 TREE_109_ONE_PATH: t109_path0 + t109_path1 + t109_path2 + t109_path3
   + t109_path4 + t109_path5 + t109_path6 = 1
 TREE_110_ONE_PATH: t110_path0 + t110_path1 + t110_path2 + t110_path3
   + t110_path4 + t110_path5 = 1
 TREE_111_ONE_PATH: t111_path0 + t111_path1 + t111_path2 + t111_path3
   + t111_path4 + t111_path5 + t111_path6 + t111_path7 = 1
 TREE_112_ONE_PATH: t112_path0 + t112_path1 + t112_path2 + t112_path3
   + t112_path4 + t112_path5 + t112_path6 + t112_path7 = 1
 TREE_113_ONE_PATH: t113_path0 + t113_path1 + t113_path2 + t113_path3
   + t113_path4 + t113_path5 + t113_path6 = 1
 TREE_114_ONE_PATH: t114_path0 + t114_path1 + t114_path2 + t114_path3
   + t114_path4 + t114_path5 + t114_path6 + t114_path7 = 1
 TREE_115_ONE_PATH: t115_path0 + t115_path1 + t115_path2 + t115_path3
   + t115_path4 + t115_path5 = 1
 TREE_116_ONE_PATH: t116_path0 + t116_path1 + t116_path2 + t116_path3
   + t116_path4 + t116_path5 + t116_path6 = 1
 TREE_117_ONE_PATH: t117_path0 + t117_path1 + t117_path2 + t117_path3
   + t117_path4 = 1
 TREE_118_ONE_PATH: t118_path0 + t118_path1 + t118_path2 + t118_path3
   + t118_path4 + t118_path5 = 1
 TREE_119_ONE_PATH: t119_path0 + t119_path1 + t119_path2 + t119_path3 = 1
 TREE_120_ONE_PATH: t120_path0 + t120_path1 + t120_path2 + t120_path3
   + t120_path4 + t120_path5 = 1
 TREE_121_ONE_PATH: t121_path0 + t121_path1 + t121_path2 + t121_path3
   + t121_path4 + t121_path5 + t121_path6 = 1
 TREE_122_ONE_PATH: t122_path0 + t122_path1 + t122_path2 + t122_path3
   + t122_path4 = 1
 TREE_123_ONE_PATH: t123_path0 + t123_path1 + t123_path2 + t123_path3
   + t123_path4 + t123_path5 + t123_path6 + t123_path7 = 1
 TREE_124_ONE_PATH: t124_path0 + t124_path1 + t124_path2 + t124_path3
   + t124_path4 + t124_path5 = 1
 TREE_125_ONE_PATH: t125_path0 + t125_path1 + t125_path2 + t125_path3
   + t125_path4 + t125_path5 = 1
 TREE_126_ONE_PATH: t126_path0 + t126_path1 + t126_path2 + t126_path3
   + t126_path4 = 1
 TREE_127_ONE_PATH: t127_path0 + t127_path1 + t127_path2 + t127_path3
   + t127_path4 + t127_path5 = 1
 TREE_128_ONE_PATH: t128_path0 + t128_path1 + t128_path2 + t128_path3
   + t128_path4 = 1
 TREE_129_ONE_PATH: t129_path0 + t129_path1 + t129_path2 + t129_path3
   + t129_path4 + t129_path5 = 1
 TREE_130_ONE_PATH: t130_path0 + t130_path1 + t130_path2 + t130_path3
   + t130_path4 + t130_path5 + t130_path6 + t130_path7 = 1
 TREE_131_ONE_PATH: t131_path0 + t131_path1 + t131_path2 + t131_path3
   + t131_path4 + t131_path5 + t131_path6 + t131_path7 = 1
 TREE_132_ONE_PATH: t132_path0 + t132_path1 + t132_path2 + t132_path3
   + t132_path4 = 1
 TREE_133_ONE_PATH: t133_path0 + t133_path1 + t133_path2 + t133_path3
   + t133_path4 = 1
 TREE_134_ONE_PATH: t134_path0 + t134_path1 + t134_path2 + t134_path3
   + t134_path4 = 1
 TREE_135_ONE_PATH: t135_path0 + t135_path1 + t135_path2 + t135_path3
   + t135_path4 + t135_path5 + t135_path6 = 1
 TREE_136_ONE_PATH: t136_path0 + t136_path1 + t136_path2 + t136_path3
   + t136_path4 = 1
 TREE_137_ONE_PATH: t137_path0 + t137_path1 + t137_path2 + t137_path3
   + t137_path4 + t137_path5 + t137_path6 + t137_path7 = 1
 TREE_138_ONE_PATH: t138_path0 + t138_path1 + t138_path2 + t138_path3
   + t138_path4 + t138_path5 + t138_path6 = 1
 TREE_139_ONE_PATH: t139_path0 + t139_path1 + t139_path2 + t139_path3
   + t139_path4 + t139_path5 + t139_path6 + t139_path7 = 1
 TREE_140_ONE_PATH: t140_path0 + t140_path1 + t140_path2 + t140_path3
   + t140_path4 + t140_path5 + t140_path6 + t140_path7 = 1
 TREE_141_ONE_PATH: t141_path0 + t141_path1 + t141_path2 + t141_path3
   + t141_path4 + t141_path5 = 1
 TREE_142_ONE_PATH: t142_path0 + t142_path1 + t142_path2 + t142_path3
   + t142_path4 = 1
 TREE_143_ONE_PATH: t143_path0 + t143_path1 + t143_path2 + t143_path3
   + t143_path4 + t143_path5 + t143_path6 = 1
 TREE_144_ONE_PATH: t144_path0 + t144_path1 + t144_path2 + t144_path3
   + t144_path4 + t144_path5 = 1
 TREE_145_ONE_PATH: t145_path0 + t145_path1 + t145_path2 + t145_path3
   + t145_path4 + t145_path5 = 1
 TREE_146_ONE_PATH: t146_path0 + t146_path1 + t146_path2 + t146_path3
   + t146_path4 + t146_path5 + t146_path6 + t146_path7 = 1
 TREE_147_ONE_PATH: t147_path0 + t147_path1 + t147_path2 + t147_path3
   + t147_path4 + t147_path5 = 1
 TREE_148_ONE_PATH: t148_path0 + t148_path1 + t148_path2 + t148_path3
   + t148_path4 + t148_path5 = 1
 TREE_149_ONE_PATH: t149_path0 + t149_path1 + t149_path2 + t149_path3
   + t149_path4 + t149_path5 + t149_path6 = 1
 TREE_150_ONE_PATH: t150_path0 + t150_path1 + t150_path2 + t150_path3
   + t150_path4 + t150_path5 + t150_path6 = 1
 TREE_151_ONE_PATH: t151_path0 + t151_path1 + t151_path2 + t151_path3
   + t151_path4 = 1
 TREE_152_ONE_PATH: t152_path0 + t152_path1 + t152_path2 + t152_path3
   + t152_path4 + t152_path5 = 1
 TREE_153_ONE_PATH: t153_path0 + t153_path1 + t153_path2 + t153_path3
   + t153_path4 + t153_path5 + t153_path6 = 1
 TREE_154_ONE_PATH: t154_path0 + t154_path1 + t154_path2 + t154_path3
   + t154_path4 + t154_path5 + t154_path6 + t154_path7 = 1
 TREE_155_ONE_PATH: t155_path0 + t155_path1 + t155_path2 + t155_path3
   + t155_path4 = 1
 TREE_156_ONE_PATH: t156_path0 + t156_path1 + t156_path2 + t156_path3
   + t156_path4 = 1
 TREE_157_ONE_PATH: t157_path0 + t157_path1 + t157_path2 + t157_path3
   + t157_path4 + t157_path5 + t157_path6 = 1
 TREE_158_ONE_PATH: t158_path0 + t158_path1 + t158_path2 + t158_path3
   + t158_path4 + t158_path5 + t158_path6 = 1
 TREE_159_ONE_PATH: t159_path0 + t159_path1 + t159_path2 + t159_path3
   + t159_path4 = 1
 TREE_160_ONE_PATH: t160_path0 + t160_path1 + t160_path2 + t160_path3
   + t160_path4 + t160_path5 + t160_path6 = 1
 TREE_161_ONE_PATH: t161_path0 + t161_path1 + t161_path2 + t161_path3
   + t161_path4 + t161_path5 + t161_path6 = 1
 TREE_162_ONE_PATH: t162_path0 + t162_path1 + t162_path2 + t162_path3
   + t162_path4 + t162_path5 + t162_path6 = 1
 TREE_163_ONE_PATH: t163_path0 + t163_path1 + t163_path2 + t163_path3
   + t163_path4 + t163_path5 = 1
 TREE_164_ONE_PATH: t164_path0 + t164_path1 + t164_path2 + t164_path3
   + t164_path4 = 1
 TREE_165_ONE_PATH: t165_path0 + t165_path1 + t165_path2 + t165_path3
   + t165_path4 = 1
 TREE_166_ONE_PATH: t166_path0 + t166_path1 + t166_path2 + t166_path3 = 1
 TREE_167_ONE_PATH: t167_path0 + t167_path1 + t167_path2 + t167_path3
   + t167_path4 = 1
 TREE_168_ONE_PATH: t168_path0 + t168_path1 + t168_path2 + t168_path3
   + t168_path4 = 1
 TREE_169_ONE_PATH: t169_path0 + t169_path1 + t169_path2 + t169_path3
   + t169_path4 + t169_path5 + t169_path6 = 1
 TREE_170_ONE_PATH: t170_path0 + t170_path1 + t170_path2 + t170_path3
   + t170_path4 = 1
 TREE_171_ONE_PATH: t171_path0 + t171_path1 + t171_path2 + t171_path3
   + t171_path4 = 1
 TREE_172_ONE_PATH: t172_path0 + t172_path1 + t172_path2 + t172_path3
   + t172_path4 + t172_path5 + t172_path6 = 1
 TREE_173_ONE_PATH: t173_path0 + t173_path1 + t173_path2 + t173_path3
   + t173_path4 + t173_path5 + t173_path6 = 1
 TREE_174_ONE_PATH: t174_path0 + t174_path1 + t174_path2 + t174_path3 = 1
 TREE_175_ONE_PATH: t175_path0 + t175_path1 + t175_path2 + t175_path3
   + t175_path4 = 1
 TREE_176_ONE_PATH: t176_path0 + t176_path1 + t176_path2 + t176_path3
   + t176_path4 + t176_path5 + t176_path6 = 1
 TREE_177_ONE_PATH: t177_path0 + t177_path1 + t177_path2 + t177_path3
   + t177_path4 + t177_path5 + t177_path6 = 1
 TREE_178_ONE_PATH: t178_path0 + t178_path1 + t178_path2 + t178_path3
   + t178_path4 = 1
 TREE_179_ONE_PATH: t179_path0 + t179_path1 + t179_path2 + t179_path3
   + t179_path4 + t179_path5 = 1
 TREE_180_ONE_PATH: t180_path0 + t180_path1 + t180_path2 + t180_path3
   + t180_path4 = 1
 TREE_181_ONE_PATH: t181_path0 + t181_path1 + t181_path2 + t181_path3
   + t181_path4 + t181_path5 + t181_path6 = 1
 TREE_182_ONE_PATH: t182_path0 + t182_path1 + t182_path2 + t182_path3
   + t182_path4 + t182_path5 + t182_path6 + t182_path7 = 1
 TREE_183_ONE_PATH: t183_path0 + t183_path1 + t183_path2 + t183_path3
   + t183_path4 + t183_path5 = 1
 TREE_184_ONE_PATH: t184_path0 + t184_path1 + t184_path2 + t184_path3
   + t184_path4 + t184_path5 = 1
 TREE_185_ONE_PATH: t185_path0 + t185_path1 + t185_path2 + t185_path3
   + t185_path4 + t185_path5 = 1
 TREE_186_ONE_PATH: t186_path0 + t186_path1 + t186_path2 + t186_path3
   + t186_path4 + t186_path5 + t186_path6 = 1
 TREE_187_ONE_PATH: t187_path0 + t187_path1 + t187_path2 + t187_path3
   + t187_path4 + t187_path5 + t187_path6 = 1
 TREE_188_ONE_PATH: t188_path0 + t188_path1 + t188_path2 + t188_path3
   + t188_path4 + t188_path5 + t188_path6 + t188_path7 = 1
 TREE_189_ONE_PATH: t189_path0 + t189_path1 + t189_path2 + t189_path3
   + t189_path4 = 1
 TREE_190_ONE_PATH: t190_path0 + t190_path1 + t190_path2 + t190_path3
   + t190_path4 + t190_path5 = 1
 TREE_191_ONE_PATH: t191_path0 + t191_path1 + t191_path2 + t191_path3
   + t191_path4 + t191_path5 + t191_path6 = 1
 TREE_192_ONE_PATH: t192_path0 + t192_path1 + t192_path2 + t192_path3
   + t192_path4 = 1
 TREE_193_ONE_PATH: t193_path0 + t193_path1 + t193_path2 + t193_path3
   + t193_path4 + t193_path5 + t193_path6 + t193_path7 = 1
 TREE_194_ONE_PATH: t194_path0 + t194_path1 + t194_path2 + t194_path3 = 1
 TREE_195_ONE_PATH: t195_path0 + t195_path1 + t195_path2 + t195_path3 = 1
 TREE_196_ONE_PATH: t196_path0 + t196_path1 + t196_path2 + t196_path3
   + t196_path4 + t196_path5 = 1
 TREE_197_ONE_PATH: t197_path0 + t197_path1 + t197_path2 + t197_path3
   + t197_path4 + t197_path5 = 1
 TREE_198_ONE_PATH: t198_path0 + t198_path1 + t198_path2 + t198_path3
   + t198_path4 + t198_path5 = 1
 TREE_199_ONE_PATH: t199_path0 + t199_path1 + t199_path2 + t199_path3
   + t199_path4 = 1
 TREE_200_ONE_PATH: t200_path0 + t200_path1 + t200_path2 + t200_path3
   + t200_path4 = 1
 TREE_201_ONE_PATH: t201_path0 + t201_path1 + t201_path2 + t201_path3
   + t201_path4 = 1
 TREE_202_ONE_PATH: t202_path0 + t202_path1 + t202_path2 + t202_path3
   + t202_path4 + t202_path5 + t202_path6 = 1
 TREE_203_ONE_PATH: t203_path0 + t203_path1 + t203_path2 + t203_path3
   + t203_path4 + t203_path5 + t203_path6 + t203_path7 = 1
 TREE_204_ONE_PATH: t204_path0 + t204_path1 + t204_path2 + t204_path3
   + t204_path4 = 1
 TREE_205_ONE_PATH: t205_path0 + t205_path1 + t205_path2 + t205_path3 = 1
 TREE_206_ONE_PATH: t206_path0 + t206_path1 + t206_path2 + t206_path3 = 1
 TREE_207_ONE_PATH: t207_path0 + t207_path1 + t207_path2 + t207_path3
   + t207_path4 + t207_path5 = 1
 TREE_208_ONE_PATH: t208_path0 + t208_path1 + t208_path2 + t208_path3
   + t208_path4 = 1
 TREE_209_ONE_PATH: t209_path0 + t209_path1 + t209_path2 + t209_path3
   + t209_path4 = 1
 TREE_210_ONE_PATH: t210_path0 + t210_path1 + t210_path2 + t210_path3
   + t210_path4 = 1
 TREE_211_ONE_PATH: t211_path0 + t211_path1 + t211_path2 + t211_path3
   + t211_path4 + t211_path5 + t211_path6 = 1
 TREE_212_ONE_PATH: t212_path0 + t212_path1 + t212_path2 + t212_path3
   + t212_path4 + t212_path5 = 1
 TREE_213_ONE_PATH: t213_path0 + t213_path1 + t213_path2 + t213_path3 = 1
 TREE_214_ONE_PATH: t214_path0 + t214_path1 + t214_path2 + t214_path3
   + t214_path4 + t214_path5 + t214_path6 = 1
 TREE_215_ONE_PATH: t215_path0 + t215_path1 + t215_path2 + t215_path3
   + t215_path4 + t215_path5 = 1
 TREE_216_ONE_PATH: t216_path0 + t216_path1 + t216_path2 + t216_path3
   + t216_path4 = 1
 TREE_217_ONE_PATH: t217_path0 + t217_path1 + t217_path2 + t217_path3
   + t217_path4 = 1
 TREE_218_ONE_PATH: t218_path0 + t218_path1 + t218_path2 + t218_path3 = 1
 TREE_219_ONE_PATH: t219_path0 + t219_path1 + t219_path2 + t219_path3
   + t219_path4 + t219_path5 + t219_path6 = 1
 TREE_220_ONE_PATH: t220_path0 + t220_path1 + t220_path2 + t220_path3
   + t220_path4 + t220_path5 + t220_path6 = 1
 TREE_221_ONE_PATH: t221_path0 + t221_path1 + t221_path2 + t221_path3
   + t221_path4 + t221_path5 = 1
 TREE_222_ONE_PATH: t222_path0 + t222_path1 + t222_path2 + t222_path3
   + t222_path4 = 1
 TREE_223_ONE_PATH: t223_path0 + t223_path1 + t223_path2 + t223_path3
   + t223_path4 + t223_path5 + t223_path6 = 1
 TREE_224_ONE_PATH: t224_path0 + t224_path1 + t224_path2 + t224_path3 = 1
 TREE_225_ONE_PATH: t225_path0 + t225_path1 + t225_path2 + t225_path3
   + t225_path4 = 1
 TREE_226_ONE_PATH: t226_path0 + t226_path1 + t226_path2 + t226_path3
   + t226_path4 + t226_path5 + t226_path6 + t226_path7 = 1
 TREE_227_ONE_PATH: t227_path0 + t227_path1 + t227_path2 + t227_path3
   + t227_path4 = 1
 TREE_228_ONE_PATH: t228_path0 + t228_path1 + t228_path2 + t228_path3
   + t228_path4 = 1
 TREE_229_ONE_PATH: t229_path0 + t229_path1 + t229_path2 + t229_path3
   + t229_path4 = 1
 TREE_230_ONE_PATH: t230_path0 + t230_path1 + t230_path2 + t230_path3
   + t230_path4 + t230_path5 + t230_path6 = 1
 TREE_231_ONE_PATH: t231_path0 + t231_path1 + t231_path2 + t231_path3
   + t231_path4 + t231_path5 + t231_path6 + t231_path7 = 1
 TREE_232_ONE_PATH: t232_path0 + t232_path1 + t232_path2 + t232_path3
   + t232_path4 = 1
 TREE_233_ONE_PATH: t233_path0 + t233_path1 + t233_path2 + t233_path3
   + t233_path4 + t233_path5 + t233_path6 = 1
 TREE_234_ONE_PATH: t234_path0 + t234_path1 + t234_path2 + t234_path3
   + t234_path4 + t234_path5 = 1
 TREE_235_ONE_PATH: t235_path0 + t235_path1 + t235_path2 + t235_path3
   + t235_path4 = 1
 TREE_236_ONE_PATH: t236_path0 + t236_path1 + t236_path2 + t236_path3
   + t236_path4 + t236_path5 = 1
 TREE_237_ONE_PATH: t237_path0 + t237_path1 + t237_path2 + t237_path3 = 1
 TREE_238_ONE_PATH: t238_path0 + t238_path1 + t238_path2 + t238_path3
   + t238_path4 + t238_path5 = 1
 TREE_239_ONE_PATH: t239_path0 + t239_path1 + t239_path2 + t239_path3
   + t239_path4 + t239_path5 + t239_path6 = 1
 TREE_240_ONE_PATH: t240_path0 + t240_path1 + t240_path2 + t240_path3
   + t240_path4 + t240_path5 = 1
 TREE_241_ONE_PATH: t241_path0 + t241_path1 + t241_path2 + t241_path3
   + t241_path4 = 1
 TREE_242_ONE_PATH: t242_path0 + t242_path1 + t242_path2 + t242_path3
   + t242_path4 = 1
 TREE_243_ONE_PATH: t243_path0 + t243_path1 + t243_path2 + t243_path3
   + t243_path4 = 1
 TREE_244_ONE_PATH: t244_path0 + t244_path1 + t244_path2 + t244_path3 = 1
 TREE_245_ONE_PATH: t245_path0 + t245_path1 + t245_path2 + t245_path3
   + t245_path4 + t245_path5 + t245_path6 = 1
 TREE_246_ONE_PATH: t246_path0 + t246_path1 + t246_path2 + t246_path3 = 1
 TREE_247_ONE_PATH: t247_path0 + t247_path1 + t247_path2 + t247_path3
   + t247_path4 = 1
 TREE_248_ONE_PATH: t248_path0 + t248_path1 + t248_path2 + t248_path3
   + t248_path4 + t248_path5 + t248_path6 + t248_path7 = 1
 TREE_249_ONE_PATH: t249_path0 + t249_path1 + t249_path2 + t249_path3
   + t249_path4 = 1
 TREE_250_ONE_PATH: t250_path0 + t250_path1 + t250_path2 + t250_path3 = 1
 TREE_251_ONE_PATH: t251_path0 + t251_path1 + t251_path2 + t251_path3
   + t251_path4 = 1
 TREE_252_ONE_PATH: t252_path0 + t252_path1 + t252_path2 + t252_path3
   + t252_path4 + t252_path5 + t252_path6 + t252_path7 = 1
 TREE_253_ONE_PATH: t253_path0 + t253_path1 + t253_path2 + t253_path3
   + t253_path4 + t253_path5 = 1
 TREE_254_ONE_PATH: t254_path0 + t254_path1 + t254_path2 + t254_path3 = 1
 TREE_255_ONE_PATH: t255_path0 + t255_path1 + t255_path2 + t255_path3
   + t255_path4 + t255_path5 + t255_path6 = 1
 TREE_256_ONE_PATH: t256_path0 + t256_path1 + t256_path2 + t256_path3
   + t256_path4 + t256_path5 + t256_path6 = 1
 TREE_257_ONE_PATH: t257_path0 + t257_path1 + t257_path2 + t257_path3
   + t257_path4 + t257_path5 + t257_path6 = 1
 TREE_258_ONE_PATH: t258_path0 + t258_path1 + t258_path2 + t258_path3
   + t258_path4 = 1
 TREE_259_ONE_PATH: t259_path0 + t259_path1 + t259_path2 + t259_path3
   + t259_path4 = 1
 TREE_260_ONE_PATH: t260_path0 + t260_path1 + t260_path2 + t260_path3
   + t260_path4 + t260_path5 = 1
 TREE_261_ONE_PATH: t261_path0 + t261_path1 + t261_path2 + t261_path3 = 1
 TREE_262_ONE_PATH: t262_path0 + t262_path1 + t262_path2 + t262_path3
   + t262_path4 = 1
 TREE_263_ONE_PATH: t263_path0 + t263_path1 + t263_path2 + t263_path3
   + t263_path4 + t263_path5 = 1
 TREE_264_ONE_PATH: t264_path0 + t264_path1 + t264_path2 + t264_path3
   + t264_path4 = 1
 TREE_265_ONE_PATH: t265_path0 + t265_path1 + t265_path2 + t265_path3
   + t265_path4 = 1
 TREE_266_ONE_PATH: t266_path0 + t266_path1 + t266_path2 + t266_path3
   + t266_path4 + t266_path5 + t266_path6 = 1
 TREE_267_ONE_PATH: t267_path0 + t267_path1 + t267_path2 + t267_path3
   + t267_path4 + t267_path5 = 1
 TREE_268_ONE_PATH: t268_path0 + t268_path1 + t268_path2 + t268_path3
   + t268_path4 + t268_path5 = 1
 TREE_269_ONE_PATH: t269_path0 + t269_path1 + t269_path2 + t269_path3
   + t269_path4 = 1
 TREE_270_ONE_PATH: t270_path0 + t270_path1 + t270_path2 + t270_path3
   + t270_path4 + t270_path5 = 1
 TREE_271_ONE_PATH: t271_path0 + t271_path1 + t271_path2 + t271_path3
   + t271_path4 + t271_path5 = 1
 TREE_272_ONE_PATH: t272_path0 + t272_path1 + t272_path2 + t272_path3
   + t272_path4 + t272_path5 + t272_path6 + t272_path7 = 1
 TREE_273_ONE_PATH: t273_path0 + t273_path1 + t273_path2 + t273_path3
   + t273_path4 = 1
 TREE_274_ONE_PATH: t274_path0 + t274_path1 + t274_path2 + t274_path3
   + t274_path4 = 1
 TREE_275_ONE_PATH: t275_path0 + t275_path1 + t275_path2 + t275_path3
   + t275_path4 + t275_path5 = 1
 TREE_276_ONE_PATH: t276_path0 + t276_path1 + t276_path2 + t276_path3
   + t276_path4 = 1
 TREE_277_ONE_PATH: t277_path0 + t277_path1 + t277_path2 + t277_path3
   + t277_path4 + t277_path5 + t277_path6 = 1
 TREE_278_ONE_PATH: t278_path0 + t278_path1 + t278_path2 + t278_path3 = 1
 TREE_279_ONE_PATH: t279_path0 + t279_path1 + t279_path2 + t279_path3
   + t279_path4 + t279_path5 = 1
 TREE_280_ONE_PATH: t280_path0 + t280_path1 + t280_path2 + t280_path3
   + t280_path4 + t280_path5 + t280_path6 = 1
 TREE_281_ONE_PATH: t281_path0 + t281_path1 + t281_path2 + t281_path3
   + t281_path4 = 1
 TREE_282_ONE_PATH: t282_path0 + t282_path1 + t282_path2 + t282_path3 = 1
 TREE_283_ONE_PATH: t283_path0 + t283_path1 + t283_path2 + t283_path3
   + t283_path4 + t283_path5 = 1
 TREE_284_ONE_PATH: t284_path0 + t284_path1 + t284_path2 + t284_path3
   + t284_path4 = 1
 TREE_285_ONE_PATH: t285_path0 + t285_path1 + t285_path2 + t285_path3
   + t285_path4 + t285_path5 = 1
 TREE_286_ONE_PATH: t286_path0 + t286_path1 + t286_path2 + t286_path3
   + t286_path4 + t286_path5 = 1
 TREE_287_ONE_PATH: t287_path0 + t287_path1 + t287_path2 + t287_path3
   + t287_path4 + t287_path5 = 1
 TREE_288_ONE_PATH: t288_path0 + t288_path1 + t288_path2 + t288_path3
   + t288_path4 + t288_path5 = 1
 TREE_289_ONE_PATH: t289_path0 + t289_path1 + t289_path2 + t289_path3
   + t289_path4 + t289_path5 = 1
 TREE_290_ONE_PATH: t290_path0 + t290_path1 + t290_path2 + t290_path3
   + t290_path4 + t290_path5 = 1
 TREE_291_ONE_PATH: t291_path0 + t291_path1 + t291_path2 + t291_path3
   + t291_path4 = 1
 TREE_292_ONE_PATH: t292_path0 + t292_path1 + t292_path2 + t292_path3
   + t292_path4 = 1
 TREE_293_ONE_PATH: t293_path0 + t293_path1 + t293_path2 + t293_path3
   + t293_path4 = 1
 TREE_294_ONE_PATH: t294_path0 + t294_path1 + t294_path2 + t294_path3
   + t294_path4 = 1
 TREE_295_ONE_PATH: t295_path0 + t295_path1 + t295_path2 + t295_path3
   + t295_path4 + t295_path5 = 1
 TREE_296_ONE_PATH: t296_path0 + t296_path1 + t296_path2 + t296_path3
   + t296_path4 + t296_path5 = 1
 TREE_297_ONE_PATH: t297_path0 + t297_path1 + t297_path2 + t297_path3 = 1
 TREE_298_ONE_PATH: t298_path0 + t298_path1 + t298_path2 + t298_path3
   + t298_path4 = 1
 TREE_299_ONE_PATH: t299_path0 + t299_path1 + t299_path2 + t299_path3
   + t299_path4 + t299_path5 + t299_path6 = 1
 TREE_300_ONE_PATH: t300_path0 + t300_path1 + t300_path2 + t300_path3
   + t300_path4 + t300_path5 = 1
 TREE_301_ONE_PATH: t301_path0 + t301_path1 + t301_path2 + t301_path3
   + t301_path4 = 1
 TREE_302_ONE_PATH: t302_path0 + t302_path1 + t302_path2 + t302_path3 = 1
 TREE_303_ONE_PATH: t303_path0 + t303_path1 + t303_path2 + t303_path3
   + t303_path4 = 1
 TREE_304_ONE_PATH: t304_path0 + t304_path1 + t304_path2 + t304_path3
   + t304_path4 = 1
 TREE_305_ONE_PATH: t305_path0 + t305_path1 + t305_path2 + t305_path3
   + t305_path4 = 1
 TREE_306_ONE_PATH: t306_path0 + t306_path1 + t306_path2 + t306_path3
   + t306_path4 + t306_path5 = 1
 TREE_307_ONE_PATH: t307_path0 + t307_path1 + t307_path2 + t307_path3
   + t307_path4 + t307_path5 = 1
 TREE_308_ONE_PATH: t308_path0 + t308_path1 + t308_path2 + t308_path3
   + t308_path4 = 1
 TREE_309_ONE_PATH: t309_path0 + t309_path1 + t309_path2 + t309_path3
   + t309_path4 + t309_path5 = 1
 TREE_310_ONE_PATH: t310_path0 + t310_path1 + t310_path2 + t310_path3
   + t310_path4 = 1
 TREE_311_ONE_PATH: t311_path0 + t311_path1 + t311_path2 + t311_path3 = 1
 TREE_312_ONE_PATH: t312_path0 + t312_path1 + t312_path2 + t312_path3
   + t312_path4 = 1
 TREE_313_ONE_PATH: t313_path0 + t313_path1 + t313_path2 + t313_path3
   + t313_path4 + t313_path5 + t313_path6 = 1
 TREE_314_ONE_PATH: t314_path0 + t314_path1 + t314_path2 + t314_path3
   + t314_path4 + t314_path5 + t314_path6 = 1
 TREE_315_ONE_PATH: t315_path0 + t315_path1 + t315_path2 + t315_path3
   + t315_path4 = 1
 TREE_316_ONE_PATH: t316_path0 + t316_path1 + t316_path2 + t316_path3
   + t316_path4 = 1
 TREE_317_ONE_PATH: t317_path0 + t317_path1 + t317_path2 + t317_path3 = 1
 TREE_318_ONE_PATH: t318_path0 + t318_path1 + t318_path2 + t318_path3 = 1
 TREE_319_ONE_PATH: t319_path0 + t319_path1 + t319_path2 + t319_path3
   + t319_path4 = 1
 TREE_320_ONE_PATH: t320_path0 + t320_path1 + t320_path2 + t320_path3
   + t320_path4 + t320_path5 + t320_path6 = 1
 TREE_321_ONE_PATH: t321_path0 + t321_path1 + t321_path2 + t321_path3
   + t321_path4 = 1
 TREE_322_ONE_PATH: t322_path0 + t322_path1 + t322_path2 + t322_path3
   + t322_path4 + t322_path5 + t322_path6 = 1
 TREE_323_ONE_PATH: t323_path0 + t323_path1 + t323_path2 + t323_path3 = 1
 TREE_324_ONE_PATH: t324_path0 + t324_path1 + t324_path2 + t324_path3
   + t324_path4 + t324_path5 + t324_path6 = 1
 TREE_325_ONE_PATH: t325_path0 + t325_path1 + t325_path2 + t325_path3
   + t325_path4 = 1
 TREE_326_ONE_PATH: t326_path0 + t326_path1 + t326_path2 + t326_path3 = 1
 TREE_327_ONE_PATH: t327_path0 + t327_path1 + t327_path2 + t327_path3
   + t327_path4 = 1
 TREE_328_ONE_PATH: t328_path0 + t328_path1 + t328_path2 + t328_path3
   + t328_path4 + t328_path5 = 1
 TREE_329_ONE_PATH: t329_path0 + t329_path1 + t329_path2 + t329_path3
   + t329_path4 + t329_path5 + t329_path6 = 1
 TREE_330_ONE_PATH: t330_path0 + t330_path1 + t330_path2 + t330_path3
   + t330_path4 + t330_path5 = 1
 TREE_331_ONE_PATH: t331_path0 + t331_path1 + t331_path2 + t331_path3
   + t331_path4 = 1
 TREE_332_ONE_PATH: t332_path0 + t332_path1 + t332_path2 + t332_path3 = 1
 TREE_333_ONE_PATH: t333_path0 + t333_path1 + t333_path2 = 1
 TREE_334_ONE_PATH: t334_path0 + t334_path1 + t334_path2 + t334_path3
   + t334_path4 + t334_path5 + t334_path6 = 1
 TREE_335_ONE_PATH: t335_path0 + t335_path1 + t335_path2 + t335_path3
   + t335_path4 + t335_path5 = 1
 TREE_336_ONE_PATH: t336_path0 + t336_path1 + t336_path2 + t336_path3
   + t336_path4 + t336_path5 + t336_path6 + t336_path7 = 1
 TREE_337_ONE_PATH: t337_path0 + t337_path1 + t337_path2 + t337_path3
   + t337_path4 + t337_path5 = 1
 TREE_338_ONE_PATH: t338_path0 + t338_path1 + t338_path2 + t338_path3
   + t338_path4 + t338_path5 + t338_path6 = 1
 TREE_339_ONE_PATH: t339_path0 + t339_path1 + t339_path2 + t339_path3
   + t339_path4 + t339_path5 = 1
 TREE_340_ONE_PATH: t340_path0 + t340_path1 + t340_path2 + t340_path3
   + t340_path4 + t340_path5 + t340_path6 = 1
 TREE_341_ONE_PATH: t341_path0 + t341_path1 + t341_path2 = 1
 TREE_342_ONE_PATH: t342_path0 + t342_path1 + t342_path2 + t342_path3
   + t342_path4 = 1
 TREE_343_ONE_PATH: t343_path0 + t343_path1 + t343_path2 + t343_path3
   + t343_path4 = 1
 TREE_344_ONE_PATH: t344_path0 + t344_path1 + t344_path2 + t344_path3
   + t344_path4 + t344_path5 + t344_path6 = 1
 TREE_345_ONE_PATH: t345_path0 + t345_path1 + t345_path2 + t345_path3
   + t345_path4 + t345_path5 + t345_path6 = 1
 TREE_346_ONE_PATH: t346_path0 + t346_path1 + t346_path2 + t346_path3
   + t346_path4 = 1
 TREE_347_ONE_PATH: t347_path0 + t347_path1 + t347_path2 + t347_path3 = 1
 TREE_348_ONE_PATH: t348_path0 + t348_path1 + t348_path2 + t348_path3
   + t348_path4 = 1
 TREE_349_ONE_PATH: t349_path0 + t349_path1 + t349_path2 + t349_path3
   + t349_path4 = 1
 TREE_350_ONE_PATH: t350_path0 + t350_path1 + t350_path2 + t350_path3
   + t350_path4 + t350_path5 + t350_path6 = 1
 TREE_351_ONE_PATH: t351_path0 + t351_path1 + t351_path2 + t351_path3
   + t351_path4 + t351_path5 = 1
 TREE_352_ONE_PATH: t352_path0 + t352_path1 = 1
 TREE_353_ONE_PATH: t353_path0 + t353_path1 + t353_path2 + t353_path3
   + t353_path4 = 1
 TREE_354_ONE_PATH: t354_path0 + t354_path1 + t354_path2 = 1
 TREE_355_ONE_PATH: t355_path0 + t355_path1 + t355_path2 + t355_path3
   + t355_path4 + t355_path5 + t355_path6 = 1
 TREE_356_ONE_PATH: t356_path0 + t356_path1 + t356_path2 + t356_path3
   + t356_path4 + t356_path5 + t356_path6 = 1
 TREE_357_ONE_PATH: t357_path0 + t357_path1 + t357_path2 + t357_path3 = 1
 TREE_358_ONE_PATH: t358_path0 + t358_path1 + t358_path2 + t358_path3
   + t358_path4 + t358_path5 + t358_path6 = 1
 TREE_359_ONE_PATH: t359_path0 + t359_path1 + t359_path2 + t359_path3
   + t359_path4 + t359_path5 + t359_path6 = 1
 TREE_360_ONE_PATH: t360_path0 + t360_path1 = 1
 TREE_361_ONE_PATH: t361_path0 + t361_path1 + t361_path2 + t361_path3 = 1
 TREE_362_ONE_PATH: t362_path0 + t362_path1 + t362_path2 + t362_path3
   + t362_path4 = 1
 TREE_363_ONE_PATH: t363_path0 + t363_path1 + t363_path2 + t363_path3
   + t363_path4 = 1
 TREE_364_ONE_PATH: t364_path0 + t364_path1 + t364_path2 + t364_path3
   + t364_path4 = 1
 TREE_365_ONE_PATH: t365_path0 + t365_path1 + t365_path2 + t365_path3
   + t365_path4 = 1
 TREE_366_ONE_PATH: t366_path0 + t366_path1 + t366_path2 + t366_path3
   + t366_path4 = 1
 TREE_367_ONE_PATH: t367_path0 + t367_path1 + t367_path2 + t367_path3 = 1
 TREE_368_ONE_PATH: t368_path0 + t368_path1 + t368_path2 + t368_path3
   + t368_path4 = 1
 TREE_369_ONE_PATH: t369_path0 + t369_path1 + t369_path2 + t369_path3
   + t369_path4 + t369_path5 = 1
 TREE_370_ONE_PATH: t370_path0 + t370_path1 + t370_path2 = 1
 TREE_371_ONE_PATH: t371_path0 + t371_path1 + t371_path2 + t371_path3
   + t371_path4 = 1
 TREE_372_ONE_PATH: t372_path0 + t372_path1 + t372_path2 = 1
 TREE_373_ONE_PATH: t373_path0 + t373_path1 + t373_path2 + t373_path3
   + t373_path4 = 1
 TREE_374_ONE_PATH: t374_path0 + t374_path1 + t374_path2 + t374_path3
   + t374_path4 + t374_path5 = 1
 TREE_375_ONE_PATH: t375_path0 + t375_path1 + t375_path2 + t375_path3 = 1
 TREE_376_ONE_PATH: t376_path0 + t376_path1 + t376_path2 + t376_path3 = 1
 TREE_377_ONE_PATH: t377_path0 = 1
 TREE_378_ONE_PATH: t378_path0 + t378_path1 + t378_path2 = 1
 TREE_379_ONE_PATH: t379_path0 + t379_path1 + t379_path2 + t379_path3
   + t379_path4 = 1
 TREE_380_ONE_PATH: t380_path0 + t380_path1 = 1
 TREE_381_ONE_PATH: t381_path0 + t381_path1 + t381_path2 + t381_path3
   + t381_path4 = 1
 TREE_382_ONE_PATH: t382_path0 + t382_path1 + t382_path2 + t382_path3
   + t382_path4 + t382_path5 = 1
 TREE_383_ONE_PATH: t383_path0 + t383_path1 + t383_path2 + t383_path3
   + t383_path4 + t383_path5 + t383_path6 = 1
 TREE_384_ONE_PATH: t384_path0 + t384_path1 + t384_path2 + t384_path3
   + t384_path4 = 1
 TREE_385_ONE_PATH: t385_path0 + t385_path1 + t385_path2 + t385_path3
   + t385_path4 = 1
 TREE_386_ONE_PATH: t386_path0 + t386_path1 + t386_path2 + t386_path3 = 1
 TREE_387_ONE_PATH: t387_path0 + t387_path1 + t387_path2 + t387_path3
   + t387_path4 = 1
 TREE_388_ONE_PATH: t388_path0 + t388_path1 + t388_path2 + t388_path3 = 1
 TREE_389_ONE_PATH: t389_path0 + t389_path1 + t389_path2 + t389_path3 = 1
 TREE_390_ONE_PATH: t390_path0 + t390_path1 + t390_path2 = 1
 TREE_391_ONE_PATH: t391_path0 + t391_path1 + t391_path2 + t391_path3 = 1
 TREE_392_ONE_PATH: t392_path0 + t392_path1 + t392_path2 + t392_path3 = 1
 TREE_393_ONE_PATH: t393_path0 + t393_path1 + t393_path2 + t393_path3
   + t393_path4 = 1
 TREE_394_ONE_PATH: t394_path0 + t394_path1 = 1
 TREE_395_ONE_PATH: t395_path0 + t395_path1 + t395_path2 + t395_path3
   + t395_path4 + t395_path5 = 1
 TREE_396_ONE_PATH: t396_path0 + t396_path1 + t396_path2 + t396_path3
   + t396_path4 = 1
 TREE_397_ONE_PATH: t397_path0 + t397_path1 + t397_path2 + t397_path3
   + t397_path4 = 1
 TREE_398_ONE_PATH: t398_path0 + t398_path1 + t398_path2 + t398_path3 = 1
 TREE_399_ONE_PATH: t399_path0 + t399_path1 + t399_path2 + t399_path3 = 1
 TREE_400_ONE_PATH: t400_path0 + t400_path1 + t400_path2 = 1
 TREE_401_ONE_PATH: t401_path0 = 1
 TREE_402_ONE_PATH: t402_path0 + t402_path1 + t402_path2 + t402_path3
   + t402_path4 = 1
 TREE_403_ONE_PATH: t403_path0 + t403_path1 + t403_path2 + t403_path3
   + t403_path4 = 1
 TREE_404_ONE_PATH: t404_path0 + t404_path1 + t404_path2 + t404_path3
   + t404_path4 + t404_path5 = 1
 TREE_405_ONE_PATH: t405_path0 + t405_path1 + t405_path2 + t405_path3
   + t405_path4 = 1
 TREE_406_ONE_PATH: t406_path0 = 1
 TREE_407_ONE_PATH: t407_path0 = 1
 TREE_408_ONE_PATH: t408_path0 + t408_path1 + t408_path2 + t408_path3
   + t408_path4 = 1
 TREE_409_ONE_PATH: t409_path0 + t409_path1 + t409_path2 + t409_path3
   + t409_path4 + t409_path5 = 1
 TREE_410_ONE_PATH: t410_path0 + t410_path1 + t410_path2 + t410_path3 = 1
 TREE_411_ONE_PATH: t411_path0 + t411_path1 + t411_path2 = 1
 TREE_412_ONE_PATH: t412_path0 + t412_path1 + t412_path2 = 1
 TREE_413_ONE_PATH: t413_path0 + t413_path1 + t413_path2 + t413_path3
   + t413_path4 = 1
 TREE_414_ONE_PATH: t414_path0 + t414_path1 + t414_path2 + t414_path3 = 1
 TREE_415_ONE_PATH: t415_path0 + t415_path1 + t415_path2 + t415_path3
   + t415_path4 = 1
 TREE_416_ONE_PATH: t416_path0 + t416_path1 + t416_path2 + t416_path3
   + t416_path4 + t416_path5 + t416_path6 = 1
 TREE_417_ONE_PATH: t417_path0 = 1
 TREE_418_ONE_PATH: t418_path0 + t418_path1 = 1
 TREE_419_ONE_PATH: t419_path0 + t419_path1 + t419_path2 + t419_path3
   + t419_path4 + t419_path5 + t419_path6 + t419_path7 = 1
 TREE_420_ONE_PATH: t420_path0 = 1
 TREE_421_ONE_PATH: t421_path0 + t421_path1 + t421_path2 + t421_path3 = 1
 TREE_422_ONE_PATH: t422_path0 + t422_path1 + t422_path2 + t422_path3
   + t422_path4 = 1
 TREE_423_ONE_PATH: t423_path0 + t423_path1 + t423_path2 + t423_path3 = 1
 TREE_424_ONE_PATH: t424_path0 + t424_path1 + t424_path2 + t424_path3
   + t424_path4 + t424_path5 = 1
 TREE_425_ONE_PATH: t425_path0 + t425_path1 + t425_path2 = 1
 TREE_426_ONE_PATH: t426_path0 + t426_path1 + t426_path2 + t426_path3
   + t426_path4 + t426_path5 + t426_path6 = 1
 TREE_427_ONE_PATH: t427_path0 + t427_path1 + t427_path2 + t427_path3 = 1
 TREE_428_ONE_PATH: t428_path0 + t428_path1 + t428_path2 + t428_path3
   + t428_path4 = 1
 TREE_429_ONE_PATH: t429_path0 + t429_path1 + t429_path2 + t429_path3
   + t429_path4 + t429_path5 = 1
 TREE_430_ONE_PATH: t430_path0 + t430_path1 + t430_path2 + t430_path3
   + t430_path4 + t430_path5 + t430_path6 = 1
 TREE_431_ONE_PATH: t431_path0 + t431_path1 + t431_path2 + t431_path3 = 1
 TREE_432_ONE_PATH: t432_path0 + t432_path1 + t432_path2 + t432_path3
   + t432_path4 + t432_path5 = 1
 TREE_433_ONE_PATH: t433_path0 + t433_path1 + t433_path2 = 1
 TREE_434_ONE_PATH: t434_path0 = 1
 TREE_435_ONE_PATH: t435_path0 + t435_path1 + t435_path2 + t435_path3
   + t435_path4 = 1
 TREE_436_ONE_PATH: t436_path0 = 1
 TREE_437_ONE_PATH: t437_path0 + t437_path1 + t437_path2 = 1
 TREE_438_ONE_PATH: t438_path0 + t438_path1 = 1
 TREE_439_ONE_PATH: t439_path0 + t439_path1 + t439_path2 + t439_path3
   + t439_path4 = 1
 TREE_440_ONE_PATH: t440_path0 + t440_path1 + t440_path2 + t440_path3
   + t440_path4 + t440_path5 = 1
 TREE_441_ONE_PATH: t441_path0 + t441_path1 + t441_path2 + t441_path3
   + t441_path4 = 1
 TREE_442_ONE_PATH: t442_path0 + t442_path1 + t442_path2 = 1
 TREE_443_ONE_PATH: t443_path0 = 1
 TREE_444_ONE_PATH: t444_path0 + t444_path1 + t444_path2 + t444_path3
   + t444_path4 = 1
 TREE_445_ONE_PATH: t445_path0 + t445_path1 + t445_path2 = 1
 TREE_446_ONE_PATH: t446_path0 + t446_path1 + t446_path2 + t446_path3 = 1
 TREE_447_ONE_PATH: t447_path0 + t447_path1 + t447_path2 = 1
 TREE_448_ONE_PATH: t448_path0 + t448_path1 + t448_path2 = 1
 TREE_449_ONE_PATH: t449_path0 + t449_path1 + t449_path2 + t449_path3
   + t449_path4 + t449_path5 = 1
 TREE_450_ONE_PATH: t450_path0 = 1
 TREE_451_ONE_PATH: t451_path0 + t451_path1 + t451_path2 + t451_path3
   + t451_path4 + t451_path5 = 1
 TREE_452_ONE_PATH: t452_path0 + t452_path1 + t452_path2 + t452_path3
   + t452_path4 = 1
 TREE_453_ONE_PATH: t453_path0 + t453_path1 + t453_path2 + t453_path3
   + t453_path4 + t453_path5 = 1
 TREE_454_ONE_PATH: t454_path0 + t454_path1 = 1
 TREE_455_ONE_PATH: t455_path0 = 1
 TREE_456_ONE_PATH: t456_path0 = 1
 TREE_457_ONE_PATH: t457_path0 + t457_path1 + t457_path2 = 1
 TREE_458_ONE_PATH: t458_path0 + t458_path1 + t458_path2 + t458_path3 = 1
 TREE_459_ONE_PATH: t459_path0 + t459_path1 = 1
 TREE_460_ONE_PATH: t460_path0 = 1
 TREE_461_ONE_PATH: t461_path0 + t461_path1 + t461_path2 + t461_path3
   + t461_path4 = 1
 TREE_462_ONE_PATH: t462_path0 + t462_path1 = 1
 TREE_463_ONE_PATH: t463_path0 + t463_path1 + t463_path2 + t463_path3
   + t463_path4 + t463_path5 + t463_path6 = 1
 TREE_464_ONE_PATH: t464_path0 + t464_path1 + t464_path2 = 1
 TREE_465_ONE_PATH: t465_path0 + t465_path1 + t465_path2 + t465_path3 = 1
 TREE_466_ONE_PATH: t466_path0 + t466_path1 + t466_path2 + t466_path3
   + t466_path4 = 1
 TREE_467_ONE_PATH: t467_path0 = 1
 TREE_468_ONE_PATH: t468_path0 = 1
 TREE_469_ONE_PATH: t469_path0 = 1
 TREE_470_ONE_PATH: t470_path0 + t470_path1 + t470_path2 = 1
 TREE_471_ONE_PATH: t471_path0 + t471_path1 + t471_path2 + t471_path3 = 1
 TREE_472_ONE_PATH: t472_path0 = 1
 TREE_473_ONE_PATH: t473_path0 + t473_path1 + t473_path2 = 1
 TREE_474_ONE_PATH: t474_path0 + t474_path1 + t474_path2 = 1
 TREE_475_ONE_PATH: t475_path0 + t475_path1 + t475_path2 = 1
 TREE_476_ONE_PATH: t476_path0 + t476_path1 + t476_path2 + t476_path3 = 1
 TREE_477_ONE_PATH: t477_path0 + t477_path1 + t477_path2 + t477_path3 = 1
 TREE_478_ONE_PATH: t478_path0 + t478_path1 + t478_path2 + t478_path3 = 1
 TREE_479_ONE_PATH: t479_path0 + t479_path1 + t479_path2 + t479_path3
   + t479_path4 + t479_path5 = 1
 TREE_480_ONE_PATH: t480_path0 + t480_path1 + t480_path2 + t480_path3
   + t480_path4 = 1
 TREE_481_ONE_PATH: t481_path0 + t481_path1 = 1
 TREE_482_ONE_PATH: t482_path0 = 1
 TREE_483_ONE_PATH: t483_path0 = 1
 TREE_484_ONE_PATH: t484_path0 + t484_path1 + t484_path2 + t484_path3 = 1
 TREE_485_ONE_PATH: t485_path0 = 1
 TREE_486_ONE_PATH: t486_path0 = 1
 TREE_487_ONE_PATH: t487_path0 = 1
 TREE_488_ONE_PATH: t488_path0 + t488_path1 + t488_path2 + t488_path3 = 1
 TREE_489_ONE_PATH: t489_path0 = 1
 TREE_490_ONE_PATH: t490_path0 + t490_path1 + t490_path2 = 1
 TREE_491_ONE_PATH: t491_path0 + t491_path1 + t491_path2 + t491_path3 = 1
 TREE_492_ONE_PATH: t492_path0 + t492_path1 + t492_path2 = 1
 TREE_493_ONE_PATH: t493_path0 + t493_path1 + t493_path2 + t493_path3
   + t493_path4 = 1
 TREE_494_ONE_PATH: t494_path0 + t494_path1 + t494_path2 = 1
 TREE_495_ONE_PATH: t495_path0 + t495_path1 = 1
 TREE_496_ONE_PATH: t496_path0 + t496_path1 = 1
 TREE_497_ONE_PATH: t497_path0 + t497_path1 + t497_path2 + t497_path3 = 1
 TREE_498_ONE_PATH: t498_path0 + t498_path1 + t498_path2 + t498_path3
   + t498_path4 = 1
 TREE_499_ONE_PATH: t499_path0 + t499_path1 = 1
 TREE_500_ONE_PATH: t500_path0 + t500_path1 + t500_path2 + t500_path3 = 1
 TREE_501_ONE_PATH: t501_path0 + t501_path1 + t501_path2 + t501_path3
   + t501_path4 + t501_path5 + t501_path6 = 1
 TREE_502_ONE_PATH: t502_path0 + t502_path1 = 1
 TREE_503_ONE_PATH: t503_path0 = 1
 TREE_504_ONE_PATH: t504_path0 + t504_path1 + t504_path2 + t504_path3 = 1
 TREE_505_ONE_PATH: t505_path0 + t505_path1 = 1
 TREE_506_ONE_PATH: t506_path0 + t506_path1 + t506_path2 + t506_path3
   + t506_path4 = 1
 TREE_507_ONE_PATH: t507_path0 + t507_path1 = 1
 TREE_508_ONE_PATH: t508_path0 + t508_path1 + t508_path2 + t508_path3
   + t508_path4 + t508_path5 = 1
 TREE_509_ONE_PATH: t509_path0 + t509_path1 + t509_path2 = 1
 TREE_510_ONE_PATH: t510_path0 = 1
 TREE_511_ONE_PATH: t511_path0 = 1
 TREE_512_ONE_PATH: t512_path0 = 1
 TREE_513_ONE_PATH: t513_path0 + t513_path1 + t513_path2 + t513_path3
   + t513_path4 + t513_path5 = 1
 TREE_514_ONE_PATH: t514_path0 = 1
 TREE_515_ONE_PATH: t515_path0 + t515_path1 + t515_path2 + t515_path3 = 1
 TREE_516_ONE_PATH: t516_path0 + t516_path1 + t516_path2 + t516_path3
   + t516_path4 = 1
 TREE_517_ONE_PATH: t517_path0 + t517_path1 + t517_path2 + t517_path3
   + t517_path4 + t517_path5 = 1
 TREE_518_ONE_PATH: t518_path0 + t518_path1 + t518_path2 + t518_path3
   + t518_path4 + t518_path5 = 1
 TREE_519_ONE_PATH: t519_path0 + t519_path1 = 1
 TREE_520_ONE_PATH: t520_path0 + t520_path1 + t520_path2 + t520_path3
   + t520_path4 + t520_path5 = 1
 TREE_521_ONE_PATH: t521_path0 + t521_path1 + t521_path2 + t521_path3
   + t521_path4 = 1
 TREE_522_ONE_PATH: t522_path0 + t522_path1 + t522_path2 + t522_path3
   + t522_path4 + t522_path5 = 1
 TREE_523_ONE_PATH: t523_path0 + t523_path1 + t523_path2 + t523_path3
   + t523_path4 = 1
 TREE_524_ONE_PATH: t524_path0 + t524_path1 + t524_path2 + t524_path3 = 1
 TREE_525_ONE_PATH: t525_path0 + t525_path1 = 1
 TREE_526_ONE_PATH: t526_path0 + t526_path1 + t526_path2 + t526_path3
   + t526_path4 + t526_path5 = 1
 TREE_527_ONE_PATH: t527_path0 + t527_path1 + t527_path2 + t527_path3
   + t527_path4 + t527_path5 = 1
 TREE_528_ONE_PATH: t528_path0 + t528_path1 + t528_path2 = 1
 TREE_529_ONE_PATH: t529_path0 + t529_path1 + t529_path2 + t529_path3
   + t529_path4 = 1
 TREE_530_ONE_PATH: t530_path0 = 1
 TREE_531_ONE_PATH: t531_path0 + t531_path1 + t531_path2 = 1
 TREE_532_ONE_PATH: t532_path0 = 1
 TREE_533_ONE_PATH: t533_path0 + t533_path1 + t533_path2 = 1
 TREE_534_ONE_PATH: t534_path0 = 1
 TREE_535_ONE_PATH: t535_path0 + t535_path1 + t535_path2 + t535_path3
   + t535_path4 = 1
 TREE_536_ONE_PATH: t536_path0 = 1
 TREE_537_ONE_PATH: t537_path0 + t537_path1 = 1
 TREE_538_ONE_PATH: t538_path0 + t538_path1 + t538_path2 + t538_path3
   + t538_path4 = 1
 TREE_539_ONE_PATH: t539_path0 + t539_path1 = 1
 TREE_540_ONE_PATH: t540_path0 + t540_path1 + t540_path2 + t540_path3
   + t540_path4 = 1
 TREE_541_ONE_PATH: t541_path0 + t541_path1 + t541_path2 = 1
 TREE_542_ONE_PATH: t542_path0 = 1
 TREE_543_ONE_PATH: t543_path0 + t543_path1 + t543_path2 + t543_path3
   + t543_path4 + t543_path5 + t543_path6 = 1
 TREE_544_ONE_PATH: t544_path0 = 1
 TREE_545_ONE_PATH: t545_path0 + t545_path1 + t545_path2 + t545_path3
   + t545_path4 = 1
 TREE_546_ONE_PATH: t546_path0 + t546_path1 = 1
 TREE_547_ONE_PATH: t547_path0 + t547_path1 = 1
 TREE_548_ONE_PATH: t548_path0 + t548_path1 + t548_path2 + t548_path3 = 1
 TREE_549_ONE_PATH: t549_path0 + t549_path1 + t549_path2 + t549_path3
   + t549_path4 + t549_path5 = 1
 TREE_550_ONE_PATH: t550_path0 + t550_path1 + t550_path2 + t550_path3 = 1
 TREE_551_ONE_PATH: t551_path0 = 1
 TREE_552_ONE_PATH: t552_path0 + t552_path1 + t552_path2 + t552_path3
   + t552_path4 = 1
 TREE_553_ONE_PATH: t553_path0 + t553_path1 + t553_path2 + t553_path3 = 1
 TREE_554_ONE_PATH: t554_path0 = 1
 TREE_555_ONE_PATH: t555_path0 = 1
 TREE_556_ONE_PATH: t556_path0 + t556_path1 = 1
 TREE_557_ONE_PATH: t557_path0 + t557_path1 + t557_path2 + t557_path3 = 1
 TREE_558_ONE_PATH: t558_path0 + t558_path1 + t558_path2 + t558_path3 = 1
 TREE_559_ONE_PATH: t559_path0 + t559_path1 + t559_path2 + t559_path3
   + t559_path4 = 1
 TREE_560_ONE_PATH: t560_path0 + t560_path1 + t560_path2 + t560_path3 = 1
 TREE_561_ONE_PATH: t561_path0 + t561_path1 + t561_path2 = 1
 TREE_562_ONE_PATH: t562_path0 + t562_path1 + t562_path2 = 1
 TREE_563_ONE_PATH: t563_path0 = 1
 TREE_564_ONE_PATH: t564_path0 + t564_path1 + t564_path2 = 1
 TREE_565_ONE_PATH: t565_path0 = 1
 TREE_566_ONE_PATH: t566_path0 + t566_path1 + t566_path2 + t566_path3
   + t566_path4 + t566_path5 = 1
 TREE_567_ONE_PATH: t567_path0 + t567_path1 + t567_path2 + t567_path3
   + t567_path4 + t567_path5 = 1
 TREE_568_ONE_PATH: t568_path0 + t568_path1 + t568_path2 + t568_path3 = 1
 TREE_569_ONE_PATH: t569_path0 = 1
 TREE_570_ONE_PATH: t570_path0 = 1
 TREE_571_ONE_PATH: t571_path0 + t571_path1 + t571_path2 + t571_path3 = 1
 TREE_572_ONE_PATH: t572_path0 + t572_path1 + t572_path2 = 1
 TREE_573_ONE_PATH: t573_path0 = 1
 TREE_574_ONE_PATH: t574_path0 + t574_path1 + t574_path2 + t574_path3 = 1
 TREE_575_ONE_PATH: t575_path0 + t575_path1 = 1
 TREE_576_ONE_PATH: t576_path0 + t576_path1 + t576_path2 + t576_path3
   + t576_path4 + t576_path5 = 1
 TREE_577_ONE_PATH: t577_path0 = 1
 TREE_578_ONE_PATH: t578_path0 + t578_path1 + t578_path2 = 1
 TREE_579_ONE_PATH: t579_path0 + t579_path1 = 1
 TREE_580_ONE_PATH: t580_path0 + t580_path1 + t580_path2 + t580_path3
   + t580_path4 + t580_path5 = 1
 TREE_581_ONE_PATH: t581_path0 + t581_path1 = 1
 TREE_582_ONE_PATH: t582_path0 + t582_path1 + t582_path2 = 1
 TREE_583_ONE_PATH: t583_path0 = 1
 TREE_584_ONE_PATH: t584_path0 + t584_path1 + t584_path2 + t584_path3
   + t584_path4 = 1
 TREE_585_ONE_PATH: t585_path0 + t585_path1 + t585_path2 + t585_path3
   + t585_path4 = 1
 TREE_586_ONE_PATH: t586_path0 + t586_path1 + t586_path2 + t586_path3
   + t586_path4 = 1
 TREE_587_ONE_PATH: t587_path0 + t587_path1 + t587_path2 + t587_path3
   + t587_path4 + t587_path5 = 1
 TREE_588_ONE_PATH: t588_path0 + t588_path1 + t588_path2 + t588_path3
   + t588_path4 + t588_path5 + t588_path6 + t588_path7 = 1
 TREE_589_ONE_PATH: t589_path0 = 1
 TREE_590_ONE_PATH: t590_path0 + t590_path1 = 1
 TREE_591_ONE_PATH: t591_path0 = 1
 TREE_592_ONE_PATH: t592_path0 = 1
 TREE_593_ONE_PATH: t593_path0 = 1
 TREE_594_ONE_PATH: t594_path0 = 1
 TREE_595_ONE_PATH: t595_path0 + t595_path1 + t595_path2 + t595_path3
   + t595_path4 = 1
 TREE_596_ONE_PATH: t596_path0 = 1
 TREE_597_ONE_PATH: t597_path0 + t597_path1 + t597_path2 + t597_path3
   + t597_path4 = 1
 TREE_598_ONE_PATH: t598_path0 + t598_path1 = 1
 TREE_599_ONE_PATH: t599_path0 = 1
 TREE_600_ONE_PATH: t600_path0 + t600_path1 + t600_path2 + t600_path3 = 1
 TREE_601_ONE_PATH: t601_path0 + t601_path1 + t601_path2 + t601_path3 = 1
 TREE_602_ONE_PATH: t602_path0 = 1
 TREE_603_ONE_PATH: t603_path0 + t603_path1 = 1
 TREE_604_ONE_PATH: t604_path0 = 1
 TREE_605_ONE_PATH: t605_path0 + t605_path1 + t605_path2 + t605_path3 = 1
 TREE_606_ONE_PATH: t606_path0 = 1
 TREE_607_ONE_PATH: t607_path0 + t607_path1 = 1
 TREE_608_ONE_PATH: t608_path0 + t608_path1 = 1
 TREE_609_ONE_PATH: t609_path0 = 1
 TREE_610_ONE_PATH: t610_path0 = 1
 TREE_611_ONE_PATH: t611_path0 + t611_path1 = 1
 TREE_612_ONE_PATH: t612_path0 = 1
 TREE_613_ONE_PATH: t613_path0 + t613_path1 = 1
 TREE_614_ONE_PATH: t614_path0 = 1
 TREE_615_ONE_PATH: t615_path0 = 1
 TREE_616_ONE_PATH: t616_path0 + t616_path1 + t616_path2 + t616_path3 = 1
 TREE_617_ONE_PATH: t617_path0 + t617_path1 + t617_path2 = 1
 TREE_618_ONE_PATH: t618_path0 + t618_path1 = 1
 TREE_619_ONE_PATH: t619_path0 + t619_path1 + t619_path2 + t619_path3
   + t619_path4 = 1
 TREE_620_ONE_PATH: t620_path0 + t620_path1 = 1
 TREE_621_ONE_PATH: t621_path0 + t621_path1 + t621_path2 + t621_path3 = 1
 TREE_622_ONE_PATH: t622_path0 + t622_path1 + t622_path2 + t622_path3 = 1
 TREE_623_ONE_PATH: t623_path0 + t623_path1 + t623_path2 + t623_path3
   + t623_path4 + t623_path5 = 1
 TREE_624_ONE_PATH: t624_path0 + t624_path1 = 1
 TREE_625_ONE_PATH: t625_path0 + t625_path1 + t625_path2 = 1
 TREE_626_ONE_PATH: t626_path0 + t626_path1 + t626_path2 + t626_path3 = 1
 TREE_627_ONE_PATH: t627_path0 + t627_path1 = 1
 TREE_628_ONE_PATH: t628_path0 = 1
 TREE_629_ONE_PATH: t629_path0 = 1
 TREE_630_ONE_PATH: t630_path0 = 1
 TREE_631_ONE_PATH: t631_path0 = 1
 TREE_632_ONE_PATH: t632_path0 + t632_path1 + t632_path2 + t632_path3 = 1
 TREE_633_ONE_PATH: t633_path0 = 1
 TREE_634_ONE_PATH: t634_path0 = 1
 TREE_635_ONE_PATH: t635_path0 = 1
 TREE_636_ONE_PATH: t636_path0 = 1
 TREE_637_ONE_PATH: t637_path0 = 1
 TREE_638_ONE_PATH: t638_path0 = 1
 TREE_639_ONE_PATH: t639_path0 = 1
 TREE_640_ONE_PATH: t640_path0 + t640_path1 + t640_path2 + t640_path3 = 1
 TREE_641_ONE_PATH: t641_path0 = 1
 TREE_642_ONE_PATH: t642_path0 = 1
 TREE_643_ONE_PATH: t643_path0 = 1
 TREE_644_ONE_PATH: t644_path0 = 1
 TREE_645_ONE_PATH: t645_path0 = 1
 TREE_646_ONE_PATH: t646_path0 + t646_path1 + t646_path2 = 1
 TREE_647_ONE_PATH: t647_path0 + t647_path1 + t647_path2 = 1
 TREE_648_ONE_PATH: t648_path0 = 1
 TREE_649_ONE_PATH: t649_path0 + t649_path1 = 1
 TREE_650_ONE_PATH: t650_path0 = 1
 TREE_651_ONE_PATH: t651_path0 = 1
 TREE_652_ONE_PATH: t652_path0 + t652_path1 + t652_path2 = 1
 TREE_653_ONE_PATH: t653_path0 + t653_path1 = 1
 TREE_654_ONE_PATH: t654_path0 = 1
 TREE_655_ONE_PATH: t655_path0 + t655_path1 + t655_path2 = 1
 TREE_656_ONE_PATH: t656_path0 + t656_path1 + t656_path2 = 1
 TREE_657_ONE_PATH: t657_path0 + t657_path1 = 1
 TREE_658_ONE_PATH: t658_path0 = 1
 TREE_659_ONE_PATH: t659_path0 + t659_path1 + t659_path2 + t659_path3
   + t659_path4 = 1
 TREE_660_ONE_PATH: t660_path0 + t660_path1 + t660_path2 + t660_path3 = 1
 TREE_661_ONE_PATH: t661_path0 + t661_path1 + t661_path2 + t661_path3
   + t661_path4 = 1
 TREE_662_ONE_PATH: t662_path0 = 1
 TREE_663_ONE_PATH: t663_path0 + t663_path1 + t663_path2 + t663_path3
   + t663_path4 + t663_path5 + t663_path6 = 1
 TREE_664_ONE_PATH: t664_path0 + t664_path1 + t664_path2 = 1
 TREE_665_ONE_PATH: t665_path0 + t665_path1 + t665_path2 = 1
 TREE_666_ONE_PATH: t666_path0 + t666_path1 + t666_path2 = 1
 TREE_667_ONE_PATH: t667_path0 + t667_path1 + t667_path2 + t667_path3 = 1
 TREE_668_ONE_PATH: t668_path0 + t668_path1 + t668_path2 + t668_path3 = 1
 TREE_669_ONE_PATH: t669_path0 = 1
 TREE_670_ONE_PATH: t670_path0 = 1
 TREE_671_ONE_PATH: t671_path0 + t671_path1 = 1
 TREE_672_ONE_PATH: t672_path0 = 1
 TREE_673_ONE_PATH: t673_path0 + t673_path1 + t673_path2 + t673_path3
   + t673_path4 = 1
 TREE_674_ONE_PATH: t674_path0 + t674_path1 = 1
 TREE_675_ONE_PATH: t675_path0 = 1
 TREE_676_ONE_PATH: t676_path0 + t676_path1 = 1
 TREE_677_ONE_PATH: t677_path0 = 1
 TREE_678_ONE_PATH: t678_path0 + t678_path1 + t678_path2 + t678_path3
   + t678_path4 = 1
 TREE_679_ONE_PATH: t679_path0 = 1
 TREE_680_ONE_PATH: t680_path0 + t680_path1 + t680_path2 + t680_path3
   + t680_path4 = 1
 TREE_681_ONE_PATH: t681_path0 + t681_path1 = 1
 TREE_682_ONE_PATH: t682_path0 = 1
 TREE_683_ONE_PATH: t683_path0 + t683_path1 + t683_path2 + t683_path3 = 1
 TREE_684_ONE_PATH: t684_path0 + t684_path1 + t684_path2 = 1
 TREE_685_ONE_PATH: t685_path0 = 1
 TREE_686_ONE_PATH: t686_path0 = 1
 TREE_687_ONE_PATH: t687_path0 + t687_path1 + t687_path2 = 1
 TREE_688_ONE_PATH: t688_path0 = 1
 TREE_689_ONE_PATH: t689_path0 = 1
 TREE_690_ONE_PATH: t690_path0 = 1
 TREE_691_ONE_PATH: t691_path0 = 1
 TREE_692_ONE_PATH: t692_path0 + t692_path1 = 1
 TREE_693_ONE_PATH: t693_path0 + t693_path1 = 1
 TREE_694_ONE_PATH: t694_path0 = 1
 TREE_695_ONE_PATH: t695_path0 = 1
 TREE_696_ONE_PATH: t696_path0 = 1
 TREE_697_ONE_PATH: t697_path0 + t697_path1 = 1
 TREE_698_ONE_PATH: t698_path0 + t698_path1 = 1
 TREE_699_ONE_PATH: t699_path0 = 1
 TREE_700_ONE_PATH: t700_path0 = 1
 TREE_701_ONE_PATH: t701_path0 = 1
 TREE_702_ONE_PATH: t702_path0 + t702_path1 + t702_path2 + t702_path3 = 1
 TREE_703_ONE_PATH: t703_path0 = 1
 TREE_704_ONE_PATH: t704_path0 = 1
 TREE_705_ONE_PATH: t705_path0 + t705_path1 + t705_path2 + t705_path3
   + t705_path4 = 1
 TREE_706_ONE_PATH: t706_path0 + t706_path1 + t706_path2 = 1
 TREE_707_ONE_PATH: t707_path0 = 1
 TREE_708_ONE_PATH: t708_path0 = 1
 TREE_709_ONE_PATH: t709_path0 + t709_path1 + t709_path2 = 1
 TREE_710_ONE_PATH: t710_path0 + t710_path1 + t710_path2 + t710_path3 = 1
 TREE_711_ONE_PATH: t711_path0 = 1
 TREE_712_ONE_PATH: t712_path0 + t712_path1 + t712_path2 + t712_path3 = 1
 TREE_713_ONE_PATH: t713_path0 + t713_path1 = 1
 TREE_714_ONE_PATH: t714_path0 = 1
 TREE_715_ONE_PATH: t715_path0 + t715_path1 + t715_path2 + t715_path3 = 1
 TREE_716_ONE_PATH: t716_path0 = 1
 TREE_717_ONE_PATH: t717_path0 = 1
 TREE_718_ONE_PATH: t718_path0 = 1
 TREE_719_ONE_PATH: t719_path0 = 1
 TREE_720_ONE_PATH: t720_path0 + t720_path1 = 1
 TREE_721_ONE_PATH: t721_path0 = 1
 TREE_722_ONE_PATH: t722_path0 = 1
 TREE_723_ONE_PATH: t723_path0 = 1
 TREE_724_ONE_PATH: t724_path0 = 1
 TREE_725_ONE_PATH: t725_path0 = 1
 TREE_726_ONE_PATH: t726_path0 + t726_path1 + t726_path2 = 1
 TREE_727_ONE_PATH: t727_path0 + t727_path1 + t727_path2 = 1
 TREE_728_ONE_PATH: t728_path0 + t728_path1 = 1
 TREE_729_ONE_PATH: t729_path0 = 1
 TREE_730_ONE_PATH: t730_path0 + t730_path1 + t730_path2 + t730_path3
   + t730_path4 = 1
 TREE_731_ONE_PATH: t731_path0 = 1
 TREE_732_ONE_PATH: t732_path0 = 1
 TREE_733_ONE_PATH: t733_path0 = 1
 TREE_734_ONE_PATH: t734_path0 = 1
 TREE_735_ONE_PATH: t735_path0 = 1
 TREE_736_ONE_PATH: t736_path0 = 1
 TREE_737_ONE_PATH: t737_path0 = 1
 TREE_738_ONE_PATH: t738_path0 = 1
 TREE_739_ONE_PATH: t739_path0 + t739_path1 = 1
 TREE_740_ONE_PATH: t740_path0 = 1
 TREE_741_ONE_PATH: t741_path0 + t741_path1 + t741_path2 = 1
 TREE_742_ONE_PATH: t742_path0 + t742_path1 + t742_path2 + t742_path3
   + t742_path4 + t742_path5 + t742_path6 = 1
 TREE_743_ONE_PATH: t743_path0 = 1
 TREE_744_ONE_PATH: t744_path0 = 1
 TREE_745_ONE_PATH: t745_path0 = 1
 TREE_746_ONE_PATH: t746_path0 = 1
 TREE_747_ONE_PATH: t747_path0 = 1
 TREE_748_ONE_PATH: t748_path0 = 1
 TREE_749_ONE_PATH: t749_path0 + t749_path1 + t749_path2 + t749_path3 = 1
 TREE_750_ONE_PATH: t750_path0 = 1
 TREE_751_ONE_PATH: t751_path0 = 1
 TREE_752_ONE_PATH: t752_path0 = 1
 TREE_753_ONE_PATH: t753_path0 = 1
 TREE_754_ONE_PATH: t754_path0 = 1
 TREE_755_ONE_PATH: t755_path0 = 1
 TREE_756_ONE_PATH: t756_path0 = 1
 TREE_757_ONE_PATH: t757_path0 + t757_path1 + t757_path2 + t757_path3 = 1
 TREE_758_ONE_PATH: t758_path0 = 1
 TREE_759_ONE_PATH: t759_path0 + t759_path1 = 1
 TREE_760_ONE_PATH: t760_path0 = 1
 TREE_761_ONE_PATH: t761_path0 + t761_path1 + t761_path2 + t761_path3 = 1
 TREE_762_ONE_PATH: t762_path0 = 1
 TREE_763_ONE_PATH: t763_path0 + t763_path1 + t763_path2 + t763_path3
   + t763_path4 = 1
 TREE_764_ONE_PATH: t764_path0 + t764_path1 + t764_path2 + t764_path3 = 1
 TREE_765_ONE_PATH: t765_path0 = 1
 TREE_766_ONE_PATH: t766_path0 + t766_path1 = 1
 TREE_767_ONE_PATH: t767_path0 = 1
 TREE_768_ONE_PATH: t768_path0 + t768_path1 + t768_path2 = 1
 TREE_769_ONE_PATH: t769_path0 = 1
 TREE_770_ONE_PATH: t770_path0 = 1
 TREE_771_ONE_PATH: t771_path0 = 1
 TREE_772_ONE_PATH: t772_path0 + t772_path1 + t772_path2 = 1
 TREE_773_ONE_PATH: t773_path0 = 1
 TREE_774_ONE_PATH: t774_path0 = 1
 TREE_775_ONE_PATH: t775_path0 + t775_path1 = 1
 TREE_776_ONE_PATH: t776_path0 = 1
 TREE_777_ONE_PATH: t777_path0 + t777_path1 = 1
 TREE_778_ONE_PATH: t778_path0 + t778_path1 + t778_path2 + t778_path3 = 1
 TREE_779_ONE_PATH: t779_path0 = 1
 TREE_780_ONE_PATH: t780_path0 = 1
 TREE_781_ONE_PATH: t781_path0 + t781_path1 + t781_path2 = 1
 TREE_782_ONE_PATH: t782_path0 = 1
 TREE_783_ONE_PATH: t783_path0 = 1
 TREE_784_ONE_PATH: t784_path0 + t784_path1 + t784_path2 + t784_path3
   + t784_path4 = 1
 TREE_785_ONE_PATH: t785_path0 + t785_path1 = 1
 TREE_786_ONE_PATH: t786_path0 = 1
 TREE_787_ONE_PATH: t787_path0 + t787_path1 = 1
 TREE_788_ONE_PATH: t788_path0 = 1
 TREE_789_ONE_PATH: t789_path0 = 1
 TREE_790_ONE_PATH: t790_path0 = 1
 TREE_791_ONE_PATH: t791_path0 = 1
 TREE_792_ONE_PATH: t792_path0 = 1
 TREE_793_ONE_PATH: t793_path0 + t793_path1 + t793_path2 + t793_path3 = 1
 TREE_794_ONE_PATH: t794_path0 = 1
 TREE_795_ONE_PATH: t795_path0 + t795_path1 + t795_path2 + t795_path3
   + t795_path4 = 1
 TREE_796_ONE_PATH: t796_path0 = 1
 TREE_797_ONE_PATH: t797_path0 = 1
 TREE_798_ONE_PATH: t798_path0 = 1
 TREE_799_ONE_PATH: t799_path0 = 1
 TREE_800_ONE_PATH: t800_path0 = 1
 TREE_801_ONE_PATH: t801_path0 = 1
 TREE_802_ONE_PATH: t802_path0 = 1
 TREE_803_ONE_PATH: t803_path0 = 1
 TREE_804_ONE_PATH: t804_path0 = 1
 TREE_805_ONE_PATH: t805_path0 + t805_path1 = 1
 TREE_806_ONE_PATH: t806_path0 + t806_path1 = 1
 TREE_807_ONE_PATH: t807_path0 = 1
 TREE_808_ONE_PATH: t808_path0 = 1
 TREE_809_ONE_PATH: t809_path0 = 1
 TREE_810_ONE_PATH: t810_path0 = 1
 TREE_811_ONE_PATH: t811_path0 = 1
 TREE_812_ONE_PATH: t812_path0 = 1
 TREE_813_ONE_PATH: t813_path0 = 1
 TREE_814_ONE_PATH: t814_path0 + t814_path1 + t814_path2 + t814_path3
   + t814_path4 = 1
 TREE_815_ONE_PATH: t815_path0 = 1
 TREE_816_ONE_PATH: t816_path0 = 1
 TREE_817_ONE_PATH: t817_path0 + t817_path1 + t817_path2 = 1
 TREE_818_ONE_PATH: t818_path0 = 1
 TREE_819_ONE_PATH: t819_path0 = 1
 TREE_820_ONE_PATH: t820_path0 = 1
 TREE_821_ONE_PATH: t821_path0 = 1
 TREE_822_ONE_PATH: t822_path0 = 1
 TREE_823_ONE_PATH: t823_path0 = 1
 TREE_824_ONE_PATH: t824_path0 = 1
 TREE_825_ONE_PATH: t825_path0 + t825_path1 = 1
 TREE_826_ONE_PATH: t826_path0 = 1
 TREE_827_ONE_PATH: t827_path0 + t827_path1 + t827_path2 + t827_path3 = 1
 TREE_828_ONE_PATH: t828_path0 = 1
 TREE_829_ONE_PATH: t829_path0 + t829_path1 + t829_path2 + t829_path3
   + t829_path4 = 1
 TREE_830_ONE_PATH: t830_path0 = 1
 TREE_831_ONE_PATH: t831_path0 + t831_path1 = 1
 TREE_832_ONE_PATH: t832_path0 + t832_path1 = 1
 TREE_833_ONE_PATH: t833_path0 = 1
 TREE_834_ONE_PATH: t834_path0 = 1
 TREE_835_ONE_PATH: t835_path0 + t835_path1 + t835_path2 = 1
 TREE_836_ONE_PATH: t836_path0 = 1
 TREE_837_ONE_PATH: t837_path0 = 1
 TREE_838_ONE_PATH: t838_path0 = 1
 TREE_839_ONE_PATH: t839_path0 + t839_path1 = 1
 TREE_840_ONE_PATH: t840_path0 = 1
 TREE_841_ONE_PATH: t841_path0 + t841_path1 = 1
 TREE_842_ONE_PATH: t842_path0 + t842_path1 = 1
 TREE_843_ONE_PATH: t843_path0 = 1
 TREE_844_ONE_PATH: t844_path0 + t844_path1 = 1
 TREE_845_ONE_PATH: t845_path0 + t845_path1 + t845_path2 = 1
 TREE_846_ONE_PATH: t846_path0 = 1
 TREE_847_ONE_PATH: t847_path0 = 1
 TREE_848_ONE_PATH: t848_path0 + t848_path1 + t848_path2 = 1
 TREE_849_ONE_PATH: t849_path0 + t849_path1 + t849_path2 = 1
 TREE_850_ONE_PATH: t850_path0 = 1
 TREE_851_ONE_PATH: t851_path0 = 1
 TREE_852_ONE_PATH: t852_path0 = 1
 TREE_853_ONE_PATH: t853_path0 = 1
 TREE_854_ONE_PATH: t854_path0 + t854_path1 = 1
 TREE_855_ONE_PATH: t855_path0 = 1
 TREE_856_ONE_PATH: t856_path0 + t856_path1 = 1
 TREE_857_ONE_PATH: t857_path0 = 1
 TREE_858_ONE_PATH: t858_path0 + t858_path1 + t858_path2 + t858_path3 = 1
 TREE_859_ONE_PATH: t859_path0 = 1
 TREE_860_ONE_PATH: t860_path0 + t860_path1 + t860_path2 = 1
 TREE_861_ONE_PATH: t861_path0 = 1
 TREE_862_ONE_PATH: t862_path0 = 1
 TREE_863_ONE_PATH: t863_path0 = 1
 TREE_864_ONE_PATH: t864_path0 = 1
 TREE_865_ONE_PATH: t865_path0 + t865_path1 + t865_path2 = 1
 TREE_866_ONE_PATH: t866_path0 = 1
 TREE_867_ONE_PATH: t867_path0 = 1
 TREE_868_ONE_PATH: t868_path0 = 1
 TREE_869_ONE_PATH: t869_path0 = 1
 TREE_870_ONE_PATH: t870_path0 = 1
 TREE_871_ONE_PATH: t871_path0 = 1
 TREE_872_ONE_PATH: t872_path0 = 1
 TREE_873_ONE_PATH: t873_path0 = 1
 TREE_874_ONE_PATH: t874_path0 = 1
 TREE_875_ONE_PATH: t875_path0 + t875_path1 = 1
 TREE_876_ONE_PATH: t876_path0 + t876_path1 + t876_path2 + t876_path3 = 1
 TREE_877_ONE_PATH: t877_path0 + t877_path1 = 1
 TREE_878_ONE_PATH: t878_path0 = 1
 TREE_879_ONE_PATH: t879_path0 = 1
 TREE_880_ONE_PATH: t880_path0 + t880_path1 + t880_path2 + t880_path3 = 1
 TREE_881_ONE_PATH: t881_path0 + t881_path1 + t881_path2 + t881_path3
   + t881_path4 = 1
 TREE_882_ONE_PATH: t882_path0 = 1
 TREE_883_ONE_PATH: t883_path0 + t883_path1 = 1
 TREE_884_ONE_PATH: t884_path0 = 1
 TREE_885_ONE_PATH: t885_path0 = 1
 TREE_886_ONE_PATH: t886_path0 + t886_path1 + t886_path2 + t886_path3
   + t886_path4 + t886_path5 = 1
 TREE_887_ONE_PATH: t887_path0 = 1
 TREE_888_ONE_PATH: t888_path0 + t888_path1 = 1
 TREE_889_ONE_PATH: t889_path0 + t889_path1 + t889_path2 + t889_path3
   + t889_path4 = 1
 TREE_890_ONE_PATH: t890_path0 = 1
 TREE_891_ONE_PATH: t891_path0 = 1
 TREE_892_ONE_PATH: t892_path0 = 1
 TREE_893_ONE_PATH: t893_path0 = 1
 TREE_894_ONE_PATH: t894_path0 + t894_path1 = 1
 TREE_895_ONE_PATH: t895_path0 = 1
 TREE_896_ONE_PATH: t896_path0 + t896_path1 = 1
 TREE_897_ONE_PATH: t897_path0 = 1
 TREE_898_ONE_PATH: t898_path0 = 1
 TREE_899_ONE_PATH: t899_path0 = 1
 TREE_900_ONE_PATH: t900_path0 = 1
 TREE_901_ONE_PATH: t901_path0 = 1
 TREE_902_ONE_PATH: t902_path0 = 1
 TREE_903_ONE_PATH: t903_path0 + t903_path1 + t903_path2 = 1
 TREE_904_ONE_PATH: t904_path0 + t904_path1 = 1
 TREE_905_ONE_PATH: t905_path0 = 1
 TREE_906_ONE_PATH: t906_path0 + t906_path1 + t906_path2 = 1
 TREE_907_ONE_PATH: t907_path0 = 1
 TREE_908_ONE_PATH: t908_path0 = 1
 TREE_909_ONE_PATH: t909_path0 + t909_path1 = 1
 TREE_910_ONE_PATH: t910_path0 = 1
 TREE_911_ONE_PATH: t911_path0 = 1
 TREE_912_ONE_PATH: t912_path0 = 1
 TREE_913_ONE_PATH: t913_path0 = 1
 YLOG_XGB_SUM: y_log_raw + 0.0227153655 t0_path0 + 0.0109316884 t0_path1
   + 0.00458499743 t0_path2 - 0.010366804 t0_path3 - 0.00269366521 t0_path4
   - 0.0123194279 t0_path5 - 0.0137853688 t0_path6 - 0.0280528124 t0_path7
   + 0.0257327575 t1_path0 + 0.012289213 t1_path1 + 0.0106788166 t1_path2
   + 0.00107296975 t1_path3 - 0.00125729048 t1_path4
   - 0.0100781415 t1_path5 - 0.0159680694 t1_path6 - 0.0299721193 t1_path7
   + 0.02208459 t2_path0 + 0.00984619744 t2_path1 + 0.00488707656 t2_path2
   - 0.00307397358 t2_path3 + 4.70025e-05 t2_path4 - 0.0104035893 t2_path5
   - 0.0102329962 t2_path6 - 0.0241005123 t2_path7 + 0.0244713817 t3_path0
   + 0.0123096956 t3_path1 + 0.00931277592 t3_path2 + 0.000628208 t3_path3
   - 0.00212428719 t3_path4 - 0.0101810889 t3_path5 - 0.0167293586 t3_path6
   - 0.0306091532 t3_path7 + 0.0118720299 t4_path0 + 0.0308137815 t4_path1
   + 0.00745445536 t4_path2 - 5.1375233999999997e-04 t4_path3
   - 0.00407241331 t4_path4 - 0.0137299579 t4_path5 - 0.0126338294 t4_path6
   - 0.0269263107 t4_path7 + 0.0214245245 t5_path0 + 0.006503365 t5_path1
   + 0.0094019305 t5_path2 + 0.00117268786 t5_path3
   - 8.9457724200000003e-04 t5_path4 - 0.00873220619 t5_path5
   - 0.0165350884 t5_path6 - 0.0269878004 t5_path7 + 0.0258362964 t6_path0
   + 0.00968578085 t6_path1 + 0.00945897959 t6_path2
   + 0.00201151217 t6_path3 - 2.7977881799999999e-04 t6_path4
   - 0.00826565176 t6_path5 - 0.0145302918 t6_path6 - 0.0263090581 t6_path7
   + 0.0316822976 t7_path0 + 0.0145802516 t7_path1 + 0.00734207034 t7_path2
   - 7.7541382100000005e-04 t7_path3 - 0.00437049242 t7_path4
   - 0.0136460699 t7_path5 - 0.0134710008 t7_path6 - 0.0235562176 t7_path7
   + 0.0089117866 t8_path0 + 0.020513352 t8_path1 + 0.00342838 t8_path2
   - 0.00765790418 t8_path3 - 0.000545633 t8_path4 - 0.00960193388 t8_path5
   - 0.0156558808 t8_path6 - 0.0258980691 t8_path7 + 0.00848270953 t9_path0
   + 0.0215684231 t9_path1 + 0.00273803 t9_path2 - 0.00590120209 t9_path3
   - 0.00314983702 t9_path4 - 0.0100015 t9_path5 - 0.0130673386 t9_path6
   - 0.0223948359 t9_path7 + 0.0277508087 t10_path0
   + 0.0141743738 t10_path1 + 0.00870797224 t10_path2
   + 0.00262025231 t10_path3 + 0.00703274645 t10_path4
   - 0.00118900684 t10_path5 - 0.00884083938 t10_path6
   - 0.0196685623 t10_path7 + 0.0221577026 t11_path0
   + 0.00698288763 t11_path1 + 0.00663986569 t11_path2
   - 1.16937277e-04 t11_path3 - 0.00215779711 t11_path4
   - 0.00982607622 t11_path5 - 0.00888452493 t11_path6
   - 0.0192282423 t11_path7 + 0.0192659479 t12_path0
   + 0.0064635952 t12_path1 + 0.00787145365 t12_path2
   + 0.00151863252 t12_path3 - 0.00502124149 t12_path4
   - 0.0146028595 t12_path5 - 0.0139762852 t12_path6
   - 0.0237245914 t12_path7 + 0.0225325394 t13_path0
   + 0.00886209123 t13_path1 + 0.00353924115 t13_path2
   - 0.0039666486 t13_path3 + 0.00126351533 t13_path4
   - 0.00731827179 t13_path5 - 0.00989489164 t13_path6
   - 0.0186696183 t13_path7 + 0.0274305008 t14_path0
   + 0.0118557578 t14_path1 + 0.00622334797 t14_path2
   - 0.00126592466 t14_path3 + 8.0659211400000005e-04 t14_path4
   - 0.00815387443 t14_path5 - 0.00930652022 t14_path6
   - 0.0186197665 t14_path7 + 0.00878201984 t15_path0
   + 0.0191700589 t15_path1 + 0.00606038515 t15_path2
   - 0.000511884 t15_path3 - 0.00604679575 t15_path4
   + 0.00104142074 t15_path5 - 0.0128101651 t15_path6
   - 0.0219680332 t15_path7 + 0.033626996 t16_path0
   + 0.0122924922 t16_path1 + 0.00504348194 t16_path2
   - 0.00162049884 t16_path3 - 0.00309862709 t16_path4
   - 0.0100407386 t16_path5 - 0.0130267804 t16_path6 - 0.02132928 t16_path7
   + 0.0325403064 t17_path0 + 0.0117814485 t17_path1
   + 0.00543196313 t17_path2 + 4.59072879e-04 t17_path3
   - 0.00167758879 t17_path4 - 0.00777553162 t17_path5
   - 0.0174500383 t17_path6 + 0.00633081142 t18_path0
   + 3.1245258300000001e-04 t18_path1 + 0.018252641 t18_path2
   + 0.00481385505 t18_path3 - 8.26768344e-04 t18_path4
   - 0.00854970608 t18_path5 - 0.0102690775 t18_path6
   - 0.0177027192 t18_path7 + 0.0195962638 t19_path0
   + 0.00677934196 t19_path1 + 0.00220389059 t19_path2
   - 0.00740419654 t19_path3 - 0.00260248478 t19_path4
   - 0.0102004157 t19_path5 - 0.00938276388 t19_path6
   - 0.0184724126 t19_path7 + 0.0188416783 t20_path0
   + 0.0104141226 t20_path1 + 0.00697504822 t20_path2
   + 0.00121598167 t20_path3 + 0.00344684091 t20_path4
   - 0.00370885315 t20_path5 - 0.00951877516 t20_path6
   - 0.0169056635 t20_path7 + 0.0196709931 t21_path0
   + 0.0081847189 t21_path1 + 0.0044476781 t21_path2
   - 0.00141097861 t21_path3 - 0.00209141336 t21_path4
   - 0.00728301657 t21_path5 - 0.0103179729 t21_path6
   - 0.0178762339 t21_path7 + 0.0130721973 t22_path0
   + 0.00586968102 t22_path1 + 0.00712332316 t22_path2
   - 4.61305e-05 t22_path3 + 0.00270687207 t22_path4
   - 0.00609189086 t22_path5 - 0.00470262952 t22_path6
   - 0.0117900008 t22_path7 + 0.0241041016 t23_path0
   + 0.00399510749 t23_path1 + 0.00963845942 t23_path2
   + 0.00364134507 t23_path3 - 0.00110748026 t23_path4
   - 0.00853515882 t23_path5 - 0.00907738693 t23_path6
   - 0.0155859552 t23_path7 + 0.0106579522 t24_path0
   + 0.00428464776 t24_path1 + 0.00870386045 t24_path2
   - 9.9255e-05 t24_path3 + 0.000860073 t24_path4 - 0.00440375926 t24_path5
   - 0.00689919759 t24_path6 - 0.0144275175 t24_path7
   + 0.0166552104 t25_path0 + 0.00561338896 t25_path1
   + 0.00518423039 t25_path2 - 3.2448841400000003e-04 t25_path3
   - 0.00109107571 t25_path4 - 0.00792970601 t25_path5
   - 0.0045954017 t25_path6 - 0.0124895899 t25_path7
   + 0.00617979141 t26_path0 + 0.00120182964 t26_path1
   + 0.0210638065 t26_path2 + 0.00887472183 t26_path3
   + 0.00159580633 t26_path4 - 0.00377549301 t26_path5
   - 0.00729638292 t26_path6 - 0.0145469727 t26_path7
   + 0.00489378301 t27_path0 - 4.9468432600000001e-04 t27_path1
   + 0.0157187432 t27_path2 + 0.00481182523 t27_path3
   + 0.00166690641 t27_path4 - 0.00311546191 t27_path5
   - 0.00613677455 t27_path6 - 0.0116045447 t27_path7
   + 0.0187104642 t28_path0 + 0.00741076889 t28_path1
   + 0.00565502374 t28_path2 + 3.3919763500000002e-04 t28_path3
   - 3.66133201e-04 t28_path4 - 0.0058907 t28_path5
   - 0.00767616974 t28_path6 - 0.0140224807 t28_path7
   + 0.0127238724 t29_path0 + 0.0300854705 t29_path1
   + 0.00493100937 t29_path2 - 7.28432e-05 t29_path3
   - 0.00213713478 t29_path4 - 0.00753479823 t29_path5
   - 0.00429897569 t29_path6 - 0.0106065618 t29_path7
   + 0.0130724311 t30_path0 + 0.000492796 t30_path1
   + 0.00796774589 t30_path2 + 0.00256570778 t30_path3
   + 0.00279190508 t30_path4 - 0.0026802551 t30_path5
   - 0.0071412297 t30_path6 - 0.0154138179 t30_path7
   + 0.0136442063 t31_path0 + 0.00502445176 t31_path1
   + 0.00457269605 t31_path2 + 2.11014863e-04 t31_path3
   - 0.00304262736 t31_path4 + 0.00594458589 t31_path5
   - 0.00728029804 t31_path6 - 0.0127771311 t31_path7
   + 0.0117484964 t32_path0 + 0.00496501196 t32_path1
   + 0.00283318176 t32_path2 + 0.0104719298 t32_path3
   + 3.7698636899999999e-04 t32_path4 - 0.00623146445 t32_path5
   - 0.00143284304 t32_path6 - 0.00855198875 t32_path7
   + 0.00918977335 t33_path0 + 0.00278635556 t33_path1
   - 0.002842488 t33_path2 + 0.00327161793 t33_path3
   - 0.005830999 t33_path4 - 0.00105623575 t33_path5
   - 0.0125997476 t33_path6 - 0.00234522973 t33_path7
   + 0.011665591 t34_path0 - 1.6888847999999999e-04 t34_path1
   + 0.00336432108 t34_path2 - 0.00126298319 t34_path3
   + 0.00412542606 t34_path4 - 0.00121779356 t34_path5
   - 0.00205302821 t34_path6 - 0.00703571457 t34_path7
   + 0.00844715722 t35_path0 + 0.00257450715 t35_path1
   + 0.0043825726 t35_path2 + 3.8339942700000001e-04 t35_path3
   - 6.98630814e-04 t35_path4 - 0.00462945038 t35_path5
   - 0.00643907581 t35_path6 - 0.0185931567 t35_path7
   + 0.0215764325 t36_path0 + 0.0040037795 t36_path1
   + 0.00556176435 t36_path2 + 8.1962760300000005e-04 t36_path3
   + 0.000127487 t36_path4 - 0.00550210755 t36_path5
   - 0.00624468969 t36_path6 - 0.0121824732 t36_path7
   + 0.0216833055 t37_path0 + 0.00404957728 t37_path1
   + 0.0045885928 t37_path2 + 0.000298595 t37_path3
   + 0.00412288681 t37_path4 - 0.00167904631 t37_path5 - 0.002872 t37_path6
   - 0.00819734484 t37_path7 + 0.0223066155 t38_path0
   + 0.00783404242 t38_path1 + 0.0017124014 t38_path2
   + 0.00551982131 t38_path3 + 0.00288653257 t38_path4
   - 0.00144843012 t38_path5 - 0.00463070674 t38_path6
   - 0.00932648312 t38_path7 + 0.0118375253 t39_path0
   + 0.00443984661 t39_path1 + 0.00377490488 t39_path2
   + 6.2661047600000004e-04 t39_path3 + 0.00727402512 t39_path4
   - 0.0015061337 t39_path5 - 0.00533777429 t39_path6
   - 0.0111084916 t39_path7 + 0.0114943217 t40_path0
   + 0.00330723636 t40_path1 + 0.00426313 t40_path2
   + 2.1022485500000001e-04 t40_path3 + 0.001674289 t40_path4
   - 0.0026396485 t40_path5 - 0.00464140065 t40_path6
   - 0.0132546667 t40_path7 + 0.00552448304 t41_path0
   + 6.59313169e-04 t41_path1 + 0.0129046207 t41_path2
   + 0.00264035817 t41_path3 - 0.0018652149 t41_path4
   - 0.00722075673 t41_path5 - 0.011348038 t41_path6
   + 0.00206290511 t41_path7 + 0.0167772565 t42_path0
   + 0.0066267834 t42_path1 + 0.00158261228 t42_path2
   - 0.00250198576 t42_path3 - 3.1887e-05 t42_path4
   - 0.00438540801 t42_path5 - 0.0101558948 t42_path6
   - 2.00129e-05 t42_path7 + 0.0113324579 t43_path0
   + 0.00410116185 t43_path1 + 0.00276523875 t43_path2
   - 0.00125339138 t43_path3 + 2.03384e-05 t43_path4
   - 0.0023150458 t43_path5 - 0.00568786589 t43_path6
   + 0.00415098574 t43_path7 + 0.00910151284 t44_path0
   + 0.00383293 t44_path1 + 0.00838313438 t44_path2
   + 0.00111966394 t44_path3 + 6.8685371700000003e-04 t44_path4
   - 0.0025440969 t44_path5 - 0.00495358044 t44_path6
   - 0.0103776949 t44_path7 + 0.00505575724 t45_path0
   + 0.0133783706 t45_path1 + 0.00235957187 t45_path2
   - 0.00308252987 t45_path3 + 0.0025718275 t45_path4
   - 0.0010436374 t45_path5 - 0.00292644696 t45_path6
   - 0.00666281069 t45_path7 + 0.00606718473 t46_path0
   + 0.0139334304 t46_path1 + 0.00274072331 t46_path2
   - 0.00140766613 t46_path3 - 1.23683421e-04 t46_path4
   - 0.00369302696 t46_path5 - 0.0123702735 t46_path6
   - 0.00301604951 t46_path7 + 0.0103797326 t47_path0
   + 0.00405490818 t47_path1 + 0.00373325124 t47_path2
   - 1.47079e-05 t47_path3 + 0.0017519109 t47_path4
   - 0.00119341386 t47_path5 - 0.00297605479 t47_path6
   - 0.00741331046 t47_path7 + 0.00300765014 t48_path0
   - 2.5583570799999998e-04 t48_path1 + 0.017837951 t48_path2
   + 0.0059100776 t48_path3 - 0.00412723 t48_path4 - 4.90748e-05 t48_path5
   - 0.00476859184 t48_path6 - 0.0105427271 t48_path7
   + 0.00625494123 t49_path0 + 0.0207809079 t49_path1
   + 0.00286215893 t49_path2 - 1.93393891e-04 t49_path3
   - 0.00456466852 t49_path4 - 0.00116436416 t49_path5
   + 0.00281661237 t49_path6 + 0.00884123892 t49_path7
   + 0.0128198247 t50_path0 + 0.00562385656 t50_path1
   + 0.00149687182 t50_path2 + 0.00145047752 t50_path3
   - 0.00191480585 t50_path4 - 0.00130254554 t50_path5
   - 0.00572970975 t50_path6 + 0.0133413933 t51_path0
   + 0.00468616839 t51_path1 + 6.5414927700000001e-04 t51_path2
   + 1.8289328699999999e-04 t51_path3 - 0.00304884836 t51_path4
   - 0.00831197295 t51_path5 + 0.00446023932 t51_path6
   + 0.0102805989 t52_path0 + 0.00449622981 t52_path1
   + 0.0010973023 t52_path2 - 0.00289466628 t52_path3
   - 0.00134074665 t52_path4 - 0.00471787807 t52_path5
   - 0.00390934804 t52_path6 - 0.0110397693 t52_path7
   + 0.0137361148 t53_path0 + 0.00302655576 t53_path1
   + 0.00305952691 t53_path2 - 2.29396552e-04 t53_path3
   + 0.0121354312 t53_path4 - 0.00170094613 t53_path5
   - 0.00812260713 t53_path6 + 0.0118717821 t54_path0
   + 3.7218356700000001e-04 t54_path1 + 0.00659488374 t54_path2
   + 0.0139810806 t54_path3 + 0.00348631665 t54_path4
   - 6.7772343799999996e-04 t54_path5 - 0.0063156914 t54_path6
   + 0.0167653784 t55_path0 + 0.00640684459 t55_path1
   + 0.00477095274 t55_path2 + 6.8369280799999998e-04 t55_path3
   - 6.7105662299999996e-04 t55_path4 - 0.00455359 t55_path5
   - 0.00631834986 t55_path6 + 2.00520837e-04 t55_path7
   + 0.0051939562 t56_path0 + 0.00217358186 t56_path1
   + 0.00191095902 t56_path2 - 0.00131825136 t56_path3
   - 0.000650858 t56_path4 - 0.00426714588 t56_path5
   - 0.00665138615 t56_path6 + 0.00171647721 t56_path7
   + 0.00746972952 t57_path0 + 0.00238190661 t57_path1
   + 0.002080312 t57_path2 - 4.1165237799999999e-04 t57_path3
   + 0.00140763901 t57_path4 - 0.00196782686 t57_path5
   - 0.00520120608 t57_path6 - 0.0133087216 t57_path7
   + 0.00952006876 t58_path0 + 0.00484250579 t58_path1
   + 0.00562144769 t58_path2 + 9.891671360000001e-04 t58_path3
   + 0.0013535422 t58_path4 - 0.00116737094 t58_path5
   - 0.00770844752 t58_path6 - 0.00276972773 t58_path7
   + 0.00428174948 t59_path0 + 0.00959212054 t59_path1
   + 0.00258154864 t59_path2 - 0.00209887489 t59_path3
   + 0.00171466765 t59_path4 - 0.00122465508 t59_path5
   - 0.00219453918 t59_path6 - 0.00580362277 t59_path7
   + 0.0137389749 t60_path0 - 9.12745e-05 t60_path1
   + 0.00242593791 t60_path2 + 5.5487582000000001e-04 t60_path3
   - 0.00191126345 t60_path4 - 0.00447982782 t60_path5
   - 0.0106911445 t60_path6 + 0.0166750643 t61_path0
   + 0.00588289043 t61_path1 - 7.68841e-05 t61_path2
   + 0.00399805512 t61_path3 + 0.000492941 t61_path4 - 0.00135926 t61_path5
   - 0.00517891441 t61_path6 + 0.00278435485 t62_path0
   + 8.05036e-06 t62_path1 + 4.8905145400000003e-04 t62_path2
   - 0.0020591733 t62_path3 - 0.00273880013 t62_path4
   - 0.00738351699 t62_path5 + 0.00123838743 t62_path6
   + 0.0139392773 t63_path0 + 0.00307171978 t63_path1
   - 5.6445144599999995e-04 t63_path2 + 0.00348932249 t63_path3
   - 0.00120525237 t63_path4 + 0.00358211459 t63_path5
   - 0.011650878 t63_path6 - 0.00175457832 t63_path7
   + 0.00906343479 t64_path0 + 0.00216957089 t64_path1
   + 0.00260281283 t64_path2 + 7.45388e-06 t64_path3
   + 6.01040665e-04 t64_path4 - 0.00208192132 t64_path5
   - 0.00478206389 t64_path6 - 0.0122264596 t64_path7
   + 0.00363004324 t65_path0 + 0.0149856992 t65_path1
   + 0.00504682679 t65_path2 + 0.00130402483 t65_path3
   + 0.00190944399 t65_path4 - 0.000804587 t65_path5
   - 0.000859871 t65_path6 - 0.00730189588 t65_path7
   - 0.00129302987 t66_path0 - 0.00543237804 t66_path1
   + 0.0030434099 t66_path2 - 2.58147833e-04 t66_path3
   + 0.00397123164 t66_path4 - 0.00184395711 t66_path5
   + 0.0130365873 t66_path6 + 0.00392097048 t66_path7
   + 0.0201685596 t67_path0 + 0.00411708979 t67_path1
   - 4.32555651e-04 t67_path2 + 0.00268459343 t67_path3
   - 8.7189522999999998e-04 t67_path4 - 0.00441960758 t67_path5
   + 0.00331276678 t68_path0 + 5.3622655099999997e-04 t68_path1
   + 3.8457987799999998e-04 t68_path2 + 0.00598658575 t68_path3
   - 0.00162188488 t68_path4 - 0.00564902 t68_path5
   + 0.00114679604 t68_path6 - 0.00125117938 t68_path7
   + 0.00617063232 t69_path0 + 4.0900977899999998e-04 t69_path1
   - 8.4713159599999999e-04 t69_path2 + 0.00203201384 t69_path3
   - 5.0446041900000004e-04 t69_path4 + 0.00140916137 t69_path5
   - 0.0043247398 t69_path6 + 0.0017048649 t70_path0
   + 0.0063395747 t70_path1 - 1.97213478e-04 t70_path2
   + 0.00210716901 t70_path3 - 0.00111249671 t70_path4
   - 0.00610058 t70_path5 + 0.000328411 t70_path6 + 0.00390894292 t70_path7
   + 0.00365128741 t71_path0 + 0.0166816153 t71_path1
   + 0.00154998933 t71_path2 - 5.61618421e-04 t71_path3
   + 0.0019131751 t71_path4 - 0.00305133243 t71_path5
   - 0.00337120914 t71_path6 - 0.0137052732 t71_path7
   - 0.000894219 t72_path0 + 0.00323132798 t72_path1
   - 9.9367741499999996e-04 t72_path2 - 0.00468653347 t72_path3
   + 0.00752118 t72_path4 + 0.00180251023 t72_path5
   + 2.16908869e-04 t72_path6 - 0.00418398902 t72_path7
   + 0.00169463211 t73_path0 + 0.00151673495 t73_path1
   + 0.0116114831 t73_path2 + 1.89168117e-04 t73_path3
   - 0.00488520134 t73_path4 - 0.00175018841 t73_path5
   - 0.00896445382 t73_path6 - 0.000846754 t74_path0
   + 0.0013139901 t74_path1 + 0.00843074173 t74_path2
   - 6.8333034900000002e-04 t74_path4 - 0.00364214228 t74_path5
   - 0.00901089888 t74_path6 + 0.0145026492 t75_path0
   + 0.00583441183 t75_path1 + 0.000311075 t75_path2
   + 0.00396964606 t75_path3 + 1.15244882e-04 t75_path4
   - 5.41288e-05 t75_path5 - 0.00320201 t75_path6
   + 3.71103437e-04 t76_path0 + 0.00804256927 t76_path1
   + 0.00229042652 t76_path2 - 0.000584338 t76_path3
   - 0.0019381796 t76_path4 - 0.0060423594 t76_path5
   - 6.4304471000000004e-04 t76_path6 + 0.00686827907 t76_path7
   + 0.00310458755 t77_path0 + 0.0084863361 t77_path1
   - 0.00107428979 t77_path2 - 5.64173e-05 t77_path3
   - 0.00262625772 t77_path4 + 0.00188998773 t77_path5
   - 6.0425448500000005e-04 t77_path6 + 0.0037126604 t78_path0
   - 0.00103696983 t78_path1 + 0.00254035578 t78_path2
   - 0.00393573428 t78_path3 + 0.00230369112 t78_path4
   - 2.1204933e-04 t78_path5 + 0.0131892404 t78_path6
   + 0.0032081469 t78_path7 + 0.0118674766 t79_path0
   + 0.00278372038 t79_path1 + 0.00196123403 t79_path2
   + 0.00104786688 t79_path3 - 0.000998914 t79_path4
   - 1.6466221099999999e-04 t79_path5 - 0.00537316827 t79_path6
   + 0.00408427278 t80_path0 + 0.0115598226 t80_path1
   - 0.00346721523 t80_path2 + 0.00155047537 t80_path3
   - 5.3645472500000004e-04 t80_path4 - 0.00145784079 t80_path5
   - 0.00435985625 t80_path6 + 0.0126114096 t81_path0
   + 0.00317987264 t81_path1 + 2.9505565200000001e-04 t81_path2
   - 8.1865757199999997e-04 t81_path3 - 0.00478470093 t81_path4
   + 0.00138328411 t82_path0 - 5.13060833e-04 t82_path1
   + 0.00596186705 t82_path2 - 3.4265813900000001e-04 t82_path3
   - 0.00139608781 t82_path4 - 0.00507387379 t82_path5
   - 0.00607924769 t82_path6 + 0.0161681436 t83_path0
   + 0.00318679377 t83_path1 + 2.7705565999999997e-04 t83_path2
   - 0.000813273 t83_path3 - 0.00479911407 t83_path4
   + 0.00569548365 t84_path0 - 6.8245083099999996e-04 t84_path1
   - 1.0185119800000001e-04 t84_path2 + 0.00200898363 t84_path3
   + 0.00741924066 t84_path4 - 4.3529251699999999e-04 t84_path5
   - 0.00559918536 t84_path6 + 0.0024415676 t85_path0
   - 1.9678450200000001e-04 t85_path1 + 0.00990337133 t85_path2
   + 0.00253942795 t85_path3 - 0.00175697659 t85_path4
   - 0.0054518478 t85_path5 + 0.000706311 t85_path6
   + 0.00417325553 t86_path0 - 7.7961053499999995e-04 t86_path1
   - 5.26916e-05 t86_path2 - 0.00265474245 t86_path3
   + 0.00421062391 t86_path4 + 0.0128922788 t86_path5
   + 0.0023644492 t86_path6 + 5.2344531300000004e-04 t87_path0
   + 0.0101360129 t87_path1 + 0.00182129955 t87_path2
   - 0.00143441337 t87_path3 - 0.0113042705 t87_path4
   + 0.00276995986 t87_path5 + 1.06550309e-04 t87_path6
   + 0.0104326783 t88_path0 + 0.00349468016 t88_path1
   - 6.0304388199999999e-04 t88_path2 - 9.9965173299999995e-04 t88_path3
   - 0.00575711764 t88_path4 + 0.00302681676 t88_path5
   + 5.54507e-05 t88_path6 + 0.0029954412 t89_path0
   - 0.00184305757 t89_path1 + 0.0197300445 t89_path2
   + 0.00545269577 t89_path3 - 4.6298e-06 t89_path4
   + 0.00453223055 t89_path5 - 0.00165447441 t89_path6
   - 0.00513537228 t89_path7 + 0.0023959598 t90_path0
   - 3.6118688900000002e-04 t90_path1 - 0.00135328935 t90_path2
   - 0.00525943935 t90_path3 + 0.00132536911 t90_path4
   - 4.8496131799999998e-04 t90_path5 + 9.7724120100000007e-04 t90_path6
   + 0.00664515 t90_path7 + 0.00519820862 t91_path0
   + 0.00132001971 t91_path1 + 0.00183331547 t91_path2
   - 3.82477243e-04 t91_path3 + 0.00139598083 t91_path4
   - 6.2231312000000005e-04 t91_path5 - 0.00298366975 t91_path6
   + 0.00321752625 t91_path7 + 0.00538262213 t92_path0
   + 5.6847970699999998e-04 t92_path1 + 0.0131767411 t92_path2
   - 0.00115153077 t92_path3 + 9.3280977999999997e-04 t92_path4
   - 0.00817385502 t92_path5 - 0.00200225553 t92_path6
   + 0.00434960192 t93_path0 - 4.6701548899999998e-04 t93_path1
   + 0.00272078183 t93_path2 - 0.00353141618 t93_path3
   + 0.00255698874 t93_path4 + 1.8024683199999999e-04 t93_path5
   + 0.00225737621 t93_path6 + 0.00874399487 t93_path7
   + 0.00269197836 t94_path0 + 3.02250788e-04 t94_path1
   - 0.0042550033 t94_path2 - 0.00142196496 t94_path3
   - 0.00210566702 t94_path4 - 0.00612526154 t94_path5
   + 0.00509647606 t94_path6 + 0.00824853778 t95_path0
   + 0.00248237513 t95_path1 + 0.00287000346 t95_path2
   + 7.9185835799999997e-04 t95_path4 - 0.00232427544 t95_path5
   - 6.0222053400000003e-04 t95_path6 - 0.00253305282 t95_path7
   + 0.00530554587 t96_path0 + 4.5696625600000001e-04 t96_path1
   + 0.0155092906 t96_path2 + 0.00247159973 t96_path3
   - 0.000173313 t96_path4 - 0.00207905 t96_path5 - 5.28007e-05 t96_path6
   + 0.0010239135 t97_path0 - 5.4656568700000002e-04 t97_path1
   - 0.00038952 t97_path2 - 0.00598687539 t97_path3
   + 0.0107287895 t97_path4 + 0.00213185046 t97_path5
   + 0.000275558 t97_path6 + 0.00430138642 t97_path7
   + 0.00250352104 t98_path0 + 0.0104057463 t98_path1
   + 9.97669e-05 t98_path2 + 9.07754e-05 t98_path3
   - 0.00222849217 t98_path4 + 0.00113806943 t98_path5
   - 0.00624600751 t98_path6 + 0.00241860095 t99_path0
   + 2.74929334e-04 t99_path1 + 0.000198231 t99_path2
   - 0.00134948047 t99_path3 - 0.00594925508 t99_path4
   + 7.8023423000000004e-04 t99_path5 - 0.00317678647 t99_path6
   + 0.00424996624 t100_path0 + 5.6870427199999995e-04 t100_path1
   + 0.0134092942 t100_path2 + 0.00213709287 t100_path3
   + 0.00314891571 t100_path4 + 3.8545057799999999e-04 t100_path5
   - 0.00103069737 t100_path6 + 0.00649810024 t100_path7
   + 0.00950837415 t101_path1 + 0.00077336 t101_path2
   + 4.20378e-05 t101_path3 + 0.00847122911 t101_path4
   - 0.000745398 t101_path5 - 0.00468973722 t101_path6
   + 0.00249119429 t102_path0 - 0.00117949373 t102_path1
   - 0.00203969935 t102_path2 + 0.00983485207 t102_path3
   - 6.62943e-05 t102_path4 - 0.00394211523 t102_path5
   - 0.00328013371 t102_path6 + 0.00288914074 t102_path7
   + 0.00384016265 t103_path0 - 2.32807113e-04 t103_path1
   - 7.96181e-05 t103_path2 - 0.00369127491 t103_path3
   - 0.00650277548 t103_path5 + 0.00154805754 t104_path0
   + 0.015939299 t104_path1 + 0.00336470571 t104_path2
   + 1.77849e-05 t104_path3 - 0.00219529774 t104_path4
   - 0.00881142356 t104_path5 - 7.23776e-05 t105_path0
   + 0.00287546078 t105_path1 + 0.0176807456 t105_path2
   + 0.00329083623 t105_path3 + 9.3594589299999999e-04 t105_path4
   - 0.00058116 t105_path5 - 0.000646476 t105_path6
   - 0.00355234812 t105_path7 + 0.00560936565 t106_path0
   + 0.00162962312 t106_path1 + 6.2943965900000005e-04 t106_path2
   - 0.00158329483 t106_path3 - 6.8990374000000004e-04 t106_path4
   - 0.00523410924 t106_path5 + 0.00725439098 t106_path6
   + 2.8199874200000001e-04 t107_path0 - 0.00136412017 t107_path1
   - 0.00377105712 t107_path2 + 0.00513667101 t107_path3
   + 1.2968918599999999e-04 t107_path4 + 0.00346692535 t107_path5
   + 0.00230826112 t107_path6 + 0.0111443391 t107_path7
   - 7.70105864e-04 t108_path0 - 0.00540410262 t108_path1
   + 0.00302288379 t108_path2 + 2.2209329499999999e-04 t108_path3
   + 0.0125533557 t108_path4 + 1.07306761e-04 t108_path5
   + 0.00201867265 t108_path6 - 0.00394007843 t108_path7
   + 0.0131585998 t109_path0 + 0.00525323441 t109_path1
   - 5.6794471999999996e-04 t109_path2 - 0.000836595 t109_path3
   + 7.2191387899999998e-04 t109_path4 - 8.0795417300000001e-04 t109_path5
   - 0.00391848106 t109_path6 + 6.8536069000000005e-04 t110_path0
   + 0.00591684505 t110_path1 - 0.000400256 t110_path2
   - 0.00840347726 t110_path3 - 0.00137121067 t110_path4
   - 0.0042779753 t110_path5 + 6.1652279699999995e-04 t111_path0
   - 0.00216934807 t111_path1 - 8.1799062899999997e-04 t111_path2
   - 0.00293535623 t111_path3 + 8.3927088399999997e-04 t111_path4
   - 0.00100846973 t111_path5 + 0.00890953839 t111_path6
   - 1.70183106e-04 t111_path7 + 0.00387369888 t112_path0
   + 5.3194788100000001e-04 t112_path1 - 0.00156509131 t112_path2
   - 0.00548292603 t112_path3 - 0.00155563245 t112_path4
   - 0.00444311183 t112_path5 - 8.32999e-05 t112_path6
   - 0.0033379 t112_path7 + 0.00389579497 t113_path0
   + 0.0129936766 t113_path2 + 4.6076640100000001e-04 t113_path3
   - 0.00320562324 t113_path4 - 0.00171589525 t113_path5
   + 0.00293459836 t113_path6 + 0.00343916262 t114_path0
   + 0.00140096608 t114_path1 + 0.0029606109 t114_path2
   - 0.00177769549 t114_path3 + 4.20475e-05 t114_path4
   - 0.00253912341 t114_path5 - 0.00222346955 t114_path6
   + 0.00464104488 t114_path7 + 0.0039138454 t115_path0
   - 4.3325129099999998e-04 t115_path1 + 6.8369769699999996e-04 t115_path2
   - 0.00145596988 t115_path3 - 0.00129996589 t115_path4
   + 0.00609500613 t115_path5 - 0.000587087 t116_path0
   - 0.00244218414 t116_path1 - 0.00813035481 t116_path2
   + 0.000477441 t116_path3 - 0.00100072054 t116_path4
   + 0.0140368547 t116_path5 + 0.00169876 t116_path6
   + 0.00469063083 t117_path0 + 3.7003177599999998e-04 t117_path2
   - 0.000823772 t117_path3 - 0.00668247836 t117_path4
   - 8.52337515e-04 t118_path0 - 0.00346091599 t118_path1
   + 0.00194121373 t118_path2 + 2.0217e-05 t118_path3
   + 0.0124354204 t118_path4 - 0.00113880471 t118_path5
   + 0.00813435297 t119_path0 + 0.000812568 t119_path1
   - 4.2833312199999998e-04 t119_path2 - 0.0072644027 t119_path3
   + 0.014609674 t120_path1 + 0.000148536 t120_path2
   + 0.00747601036 t120_path3 + 0.00106170145 t120_path4
   - 0.00397162698 t120_path5 + 0.0123893321 t121_path0
   + 0.00191624311 t121_path1 + 0.00268007908 t121_path2
   + 2.1115451800000001e-04 t121_path3 + 3.1029852100000001e-04 t121_path4
   - 8.76532751e-04 t121_path5 - 0.00303396909 t121_path6
   + 0.00243653287 t122_path0 - 6.7409244400000001e-04 t122_path1
   + 0.000143893 t122_path2 - 0.00192043663 t122_path3
   - 0.00321921869 t122_path4 + 1.76089423e-04 t123_path0
   - 0.00127760717 t123_path1 - 0.00550385797 t123_path2
   - 9.8484172499999989e-04 t123_path3 + 0.00171859399 t123_path4
   + 0.00801094715 t123_path5 + 4.8628283500000001e-04 t123_path6
   - 0.00287363213 t123_path7 + 0.00116190047 t124_path0
   - 1.80805218e-04 t124_path1 - 0.00382135366 t124_path2
   - 0.00116886781 t124_path4 - 0.00347349839 t124_path5
   + 0.00177458406 t125_path0 + 0.00960239209 t125_path1
   + 2.4653444399999998e-04 t125_path2 - 0.00219590357 t125_path3
   - 0.00281869085 t125_path4 + 0.00246447884 t125_path5
   + 0.0038692567 t126_path0 - 1.3273856900000001e-04 t126_path1
   + 0.00289387209 t126_path2 + 8.3683879299999999e-04 t126_path3
   - 0.00484327972 t126_path4 - 0.000827319 t127_path0
   + 0.0148607343 t127_path1 + 0.00430076662 t127_path2
   + 9.7087235200000001e-04 t127_path3 + 0.00212545763 t127_path4
   - 2.1923164600000001e-04 t127_path5 + 0.00807339326 t128_path0
   + 6.66278589e-04 t128_path1 - 0.00104274566 t128_path2
   - 5.7749310500000001e-04 t128_path3 - 0.00346494932 t128_path4
   + 0.0127552152 t129_path0 - 0.000351672 t129_path1
   - 0.000155821 t129_path2 - 0.00435514795 t129_path3
   + 5.8101856799999999e-04 t129_path4 + 0.00364471599 t129_path5
   + 9.8536640899999999e-04 t130_path0 - 4.2817671800000001e-04 t130_path1
   + 0.00155460928 t130_path2 - 0.00388227194 t130_path3
   - 0.000745447 t130_path4 - 0.00387294171 t130_path5
   + 4.92481e-05 t130_path6 - 0.00362609443 t130_path7
   + 0.0164996125 t131_path0 + 6.1366969000000002e-04 t131_path1
   - 0.00130882789 t131_path2 + 0.00341900345 t131_path3
   - 9.6837611600000004e-04 t131_path4 + 6.3013675400000002e-04 t131_path5
   - 0.00131754228 t131_path6 - 0.00589310424 t131_path7
   + 0.000501106 t132_path0 - 0.00103556376 t132_path1
   + 0.0067718057 t132_path2 - 0.00656205043 t132_path3
   - 6.0167774800000005e-04 t132_path4 + 0.00384620437 t133_path0
   - 0.00138416921 t133_path1 - 6.86593e-05 t133_path2
   - 0.00227153627 t133_path3 + 0.00617895694 t133_path4
   + 0.0022956226 t134_path0 - 6.2762631599999998e-04 t134_path1
   - 0.000735431 t134_path2 + 5.0324550799999996e-04 t134_path3
   - 0.00601193681 t134_path4 + 0.00223605451 t135_path0
   - 0.00155308784 t135_path1 + 0.00148947118 t135_path2
   + 0.00583571196 t135_path3 + 8.74444202e-04 t135_path4
   - 0.000375838 t135_path5 + 0.00445225183 t135_path6
   - 6.6615169599999996e-04 t136_path0 + 4.2879630900000002e-04 t136_path1
   + 0.00708312215 t136_path2 - 0.00471841451 t136_path3
   - 1.6588933099999999e-04 t136_path4 + 0.00440930529 t137_path0
   + 0.00149541919 t137_path1 + 0.00148240733 t137_path2
   - 0.0023806931 t137_path3 - 0.00117226562 t137_path4
   + 0.000152184 t137_path5 - 4.65111e-05 t137_path6
   - 0.0062578083 t137_path7 + 0.00992888212 t138_path0
   - 1.08826818e-04 t138_path1 + 0.0039262888 t138_path2
   + 0.00283156522 t138_path3 + 6.2334770300000002e-04 t138_path4
   - 4.5266345800000003e-04 t138_path5 + 0.00525639392 t138_path6
   + 8.50913872e-04 t139_path0 - 3.8085674200000002e-04 t139_path1
   + 0.00113696326 t139_path2 + 0.00554104149 t139_path3
   + 0.00246050628 t139_path4 - 0.00175151671 t139_path5
   + 3.1314615600000001e-04 t139_path6 - 0.00222654128 t139_path7
   + 0.00154242804 t140_path0 + 0.0124624632 t140_path1
   + 7.54184672e-04 t140_path2 + 0.00370411738 t140_path3
   + 0.000923204 t140_path4 - 4.3876149000000002e-04 t140_path5
   - 0.001051725 t140_path6 - 0.0061641084 t140_path7
   + 0.00404748367 t141_path0 + 4.16536233e-04 t141_path1
   + 0.0065574795 t141_path2 + 0.00249515404 t141_path3
   - 1.3307880699999999e-04 t141_path4 - 0.00717616407 t141_path5
   + 1.44768783e-04 t142_path0 - 0.00229809224 t142_path1
   + 1.8105332899999999e-04 t142_path2 - 0.00263043516 t142_path3
   + 0.00564735383 t142_path4 + 0.00383990351 t143_path0
   - 3.3651170100000002e-04 t143_path1 + 0.00904309284 t143_path2
   + 4.7012424299999999e-04 t143_path3 - 6.3238438500000003e-04 t143_path4
   + 0.00101561949 t143_path5 - 0.00377840595 t143_path6
   - 4.1747884800000002e-04 t144_path0 - 0.00421067467 t144_path1
   + 0.00271595479 t144_path2 + 2.5746374699999998e-04 t144_path3
   + 0.00167537411 t144_path4 - 0.00361242797 t144_path5
   + 4.67970327e-04 t145_path0 - 0.00331039634 t145_path1
   - 9.0656889399999996e-04 t145_path2 + 0.00333343935 t145_path3
   + 0.000228045 t145_path4 + 0.0117751723 t145_path5
   + 0.00125302398 t146_path0 + 0.00491212076 t146_path1
   + 2.53636885e-04 t146_path2 - 0.00295299874 t146_path3
   - 1.0158356599999999e-04 t146_path4 - 0.00248405733 t146_path5
   - 0.00221785437 t146_path6 + 0.00130692043 t146_path7
   + 0.00670928461 t147_path0 + 0.00143308355 t147_path1
   - 2.80722e-05 t147_path2 - 0.00108349102 t147_path3
   + 0.00128695776 t147_path4 + 0.0066680559 t147_path5
   + 0.00189711095 t148_path0 - 2.2969320700000001e-04 t148_path1
   - 0.00374605716 t148_path2 + 0.00977600832 t148_path3
   - 1.8778567e-04 t148_path4 + 0.00391730852 t148_path5
   + 0.00519103697 t149_path0 + 0.00105129613 t149_path1
   - 8.0310588199999997e-04 t149_path2 + 0.000287164 t149_path3
   - 6.5617461199999998e-04 t149_path4 - 0.00459218211 t149_path5
   - 0.000290911 t149_path6 + 0.0100056566 t150_path1
   + 0.000745856 t150_path2 + 0.0025771081 t150_path3
   + 6.6731940000000001e-04 t150_path4 - 0.000203765 t150_path5
   - 0.00303622219 t150_path6 + 0.00190708 t151_path0
   + 2.15112683e-04 t151_path1 - 0.00057727 t151_path2
   - 0.00464307098 t151_path3 - 0.00193217641 t151_path4
   - 0.000404638 t152_path0 + 0.00943173654 t152_path1
   - 0.0022599895 t152_path2 + 3.5593e-05 t152_path3
   - 0.00134276634 t152_path4 - 0.00549311098 t152_path5
   + 0.00475836545 t153_path0 - 0.00123234745 t153_path1
   + 0.00131855812 t153_path2 + 0.00307313376 t153_path3
   - 0.000167871 t153_path4 - 0.00109569973 t153_path5
   - 0.00541100558 t153_path6 + 0.0128901443 t154_path0
   + 0.00256861979 t154_path1 + 9.0384035100000005e-04 t154_path2
   - 8.6343852999999995e-04 t154_path3 + 0.00305730267 t154_path4
   - 0.0019004409 t154_path5 - 9.9845463400000003e-04 t154_path6
   + 8.13391e-05 t154_path7 + 2.0216322399999999e-04 t155_path0
   - 0.00289092213 t155_path1 - 0.00339708594 t155_path2
   + 0.00257968367 t155_path3 + 0.00782286096 t155_path4
   + 0.00103155756 t156_path0 - 0.00170895655 t156_path2
   - 0.000935968 t156_path3 - 0.00475443574 t156_path4
   + 0.00262764352 t157_path0 + 1.5708665899999999e-04 t157_path1
   - 8.96444e-06 t157_path2 + 0.00409145746 t157_path3
   + 9.0533064200000002e-04 t157_path4 - 6.45026797e-04 t157_path5
   - 0.00538952323 t157_path6 + 0.00128950353 t158_path0
   + 0.00497971522 t158_path1 - 0.00238245493 t158_path2
   + 2.32776e-05 t158_path3 - 0.00131484552 t158_path4
   + 8.5022358700000003e-04 t158_path5 - 0.000171738 t158_path6
   + 0.00685757771 t159_path0 + 1.48523803e-04 t159_path1
   + 0.000576754 t159_path2 - 5.52274694e-04 t159_path3
   - 0.00271712895 t159_path4 - 9.0906472200000005e-04 t160_path0
   - 0.00358924293 t160_path1 + 3.640184e-04 t160_path2
   - 0.00157650188 t160_path3 + 0.0123870578 t160_path4
   + 0.00301926606 t160_path5 - 6.48189e-05 t160_path6
   - 2.94476165e-04 t161_path0 - 0.0043304693 t161_path1
   + 2.19343e-05 t161_path2 - 0.00423778314 t161_path3
   + 0.00254894095 t161_path4 - 0.00142773858 t161_path5
   + 0.00995922927 t161_path6 + 0.00277962233 t162_path0
   - 0.0010286 t162_path1 + 0.00924222916 t162_path2
   + 1.9550281299999999e-04 t162_path3 - 0.00194780191 t162_path4
   - 1.1168448e-04 t162_path5 - 0.0040239715 t162_path6
   + 6.3215522100000001e-04 t163_path0 - 0.00231857831 t163_path1
   - 0.00110649515 t163_path2 + 0.00803782139 t163_path3
   - 7.7047327100000004e-04 t163_path4 - 0.004270704 t163_path5
   + 8.1543158699999996e-04 t164_path0 + 0.00594227761 t164_path1
   + 4.15493e-06 t164_path2 - 0.00151229242 t164_path3
   - 0.00441542594 t164_path4 + 4.72425e-05 t165_path0
   + 0.00250822585 t165_path1 - 0.00402288139 t165_path2
   - 1.1790121100000001e-04 t165_path3 + 0.00679614907 t165_path4
   + 1.99581875e-04 t166_path0 - 0.00118928764 t166_path1
   - 0.00316333212 t166_path2 - 0.00668793404 t166_path3
   + 0.00766464556 t167_path0 + 0.00420584157 t167_path1
   + 9.89351e-05 t167_path2 - 0.0018240317 t167_path3
   + 3.8032620799999999e-04 t167_path4 + 1.7208413899999999e-04 t168_path0
   - 9.0946885800000001e-04 t168_path1 - 0.00216228724 t168_path2
   + 0.00345744845 t168_path3 - 0.00236554514 t168_path4
   + 0.00364929414 t169_path0 - 0.00132622477 t169_path1
   + 4.6319028499999999e-04 t169_path2 - 0.00136351411 t169_path3
   - 0.00212590164 t169_path4 + 0.00151024479 t169_path5
   - 7.6797971299999998e-04 t169_path6 - 2.87673e-05 t170_path0
   - 0.00225284765 t170_path1 + 0.000315003 t170_path2
   + 0.00405591261 t170_path3 - 0.00202639261 t170_path4
   + 5.47664e-05 t171_path0 - 0.00262519903 t171_path1
   + 0.00253719394 t171_path2 - 0.0014689028 t171_path3
   - 0.00208207848 t171_path4 + 0.00113251607 t172_path0
   - 2.11031118e-04 t172_path1 - 0.000490008 t172_path2
   + 0.00244770618 t172_path3 + 0.00142520433 t172_path4
   - 0.00198743143 t172_path5 - 3.0158512499999998e-04 t172_path6
   + 0.0072251833 t173_path0 - 0.00204607402 t173_path1
   + 0.00241500279 t173_path2 + 0.00157694973 t173_path3
   - 6.17756e-05 t173_path4 - 0.00114715099 t173_path5
   + 0.00197835197 t173_path6 - 8.1820873299999995e-04 t174_path0
   + 2.3724029600000001e-04 t174_path1 - 0.00166332279 t174_path2
   + 0.00546094449 t174_path3 + 1.9622159e-04 t175_path0
   + 0.00540297572 t175_path1 + 2.16569e-05 t175_path2
   + 0.00286504603 t175_path3 - 0.00478268228 t175_path4
   - 7.3244056000000004e-04 t176_path0 - 0.00493534375 t176_path1
   + 8.5767684500000002e-04 t176_path2 - 0.00511272019 t176_path3
   + 0.00214686012 t176_path4 + 5.14896e-05 t176_path5
   + 0.0065221414 t176_path6 + 9.8397349899999992e-04 t177_path0
   - 0.00282414583 t177_path1 - 0.00186355936 t177_path2
   + 6.0287542900000002e-04 t177_path3 - 6.8606977600000002e-04 t177_path4
   + 2.3832917200000001e-04 t177_path5 - 0.00187534431 t177_path6
   - 0.00115839101 t178_path0 + 0.00196731486 t178_path1
   + 0.00514182774 t178_path2 + 2.2467914099999999e-04 t178_path3
   - 0.00207689381 t178_path4 + 0.0025010996 t179_path0
   - 0.00188148813 t179_path1 + 0.0075530149 t179_path2
   - 0.00114242511 t179_path3 + 1.6533964699999999e-04 t179_path4
   - 0.00189401931 t179_path5 + 0.00905951206 t180_path0
   + 0.000731928 t180_path1 - 0.000128799 t180_path2
   - 0.00292870682 t180_path3 + 6.6335080199999999e-04 t180_path4
   + 9.71544e-05 t181_path0 - 0.00143028074 t181_path1
   - 0.00304660876 t181_path2 + 0.00574098201 t181_path3
   + 0.00124600821 t181_path4 + 2.67431256e-04 t181_path5
   - 0.00320238806 t181_path6 + 0.00326526631 t182_path0
   + 2.61599e-05 t182_path1 - 0.00202956027 t182_path2
   + 0.00352951256 t182_path3 + 0.00961248111 t182_path4
   - 0.00224315352 t182_path6 + 0.00111086702 t182_path7
   + 0.000287012 t183_path0 - 9.6278119599999999e-04 t183_path1
   - 0.00483966758 t183_path2 + 0.00765790977 t183_path3
   + 1.9218541299999999e-04 t183_path4 + 0.00129547226 t183_path5
   + 0.000459992 t184_path0 - 7.9059024599999997e-04 t184_path1
   - 0.00159231713 t184_path2 - 9.04268178e-04 t184_path3
   - 0.00402761158 t184_path4 + 0.00337277 t184_path5
   + 2.07902878e-04 t185_path0 - 9.7563635799999997e-04 t185_path1
   - 8.3005614599999998e-04 t185_path2 - 0.00311585655 t185_path3
   + 0.00895266607 t185_path4 - 0.00164797914 t185_path5
   - 4.49474814e-04 t186_path0 + 5.4404145300000002e-04 t186_path1
   - 0.00297788437 t186_path2 - 0.000905289 t186_path3
   + 0.00635075895 t186_path4 + 0.00145009172 t186_path5
   - 0.00142081606 t186_path6 + 9.08691e-05 t187_path0
   - 8.3337322599999996e-04 t187_path1 + 0.0038740125 t187_path3
   + 0.00763477059 t187_path4 + 0.00193067873 t187_path5
   - 0.00215014885 t187_path6 - 0.00301195169 t188_path0
   - 8.6531275900000003e-04 t188_path1 - 0.00107223075 t188_path2
   + 0.00172887882 t188_path3 + 4.29223612e-04 t188_path4
   - 0.00225110166 t188_path5 + 6.7669572299999998e-04 t188_path6
   - 0.00299918419 t188_path7 + 0.00181068573 t189_path0
   + 1.10858316e-04 t189_path1 - 0.00149154337 t189_path2
   + 0.00204217 t189_path3 + 0.00593167311 t189_path4
   + 0.00549565488 t190_path0 + 1.9519748499999999e-04 t190_path1
   + 5.54534e-05 t190_path2 - 0.00258282013 t190_path3
   - 0.0011386812 t190_path4 - 0.00408697082 t190_path5
   + 0.005940089 t191_path0 - 8.06453e-05 t191_path2
   - 0.00509836851 t191_path3 + 0.00930203777 t191_path4
   + 0.00265856227 t191_path5 - 0.00259114569 t191_path6
   + 8.1273919200000003e-04 t192_path0 - 0.000115221 t192_path1
   - 0.0015672195 t192_path2 - 0.00165862311 t192_path3
   + 3.4220164500000001e-04 t192_path4 + 4.9974629700000003e-04 t193_path0
   - 0.00185153738 t193_path1 - 5.07829e-05 t193_path2
   + 0.00837193616 t193_path3 + 0.00290075387 t193_path4
   - 5.0091650299999995e-04 t193_path5 - 0.0015977869 t193_path6
   + 0.00216280879 t193_path7 + 0.00278056832 t194_path0
   + 1.1121e-05 t194_path1 - 0.00393475406 t194_path2
   - 0.00507996464 t194_path3 + 0.00199575373 t195_path0
   - 4.8313647899999999e-04 t195_path1 + 3.4862189199999998e-04 t195_path2
   - 0.0044179908 t195_path3 - 1.8382e-05 t196_path0
   - 0.00166235492 t196_path1 - 0.00447970582 t196_path2
   - 0.00100848381 t196_path3 + 0.000311528 t196_path4
   + 0.00330254831 t196_path5 + 7.38665e-05 t197_path0
   - 0.00194279698 t197_path1 - 0.00255087134 t197_path2
   + 0.00347314868 t197_path3 + 0.00895583071 t197_path4
   - 0.0011915277 t197_path5 - 0.00235328381 t198_path0
   + 0.00785529707 t198_path1 - 2.2192248400000001e-04 t198_path2
   + 0.000825873 t198_path3 - 0.00155206083 t198_path4
   + 0.00381226582 t198_path5 + 0.00230843481 t199_path0
   - 0.00318638654 t199_path1 + 2.1901678699999999e-04 t199_path2
   - 6.9844577199999996e-04 t199_path3 - 0.00185636024 t199_path4
   + 0.00106137909 t200_path0 - 7.76621e-05 t200_path1
   + 0.0014294571 t200_path2 + 0.00133561 t200_path3
   - 0.00268011773 t200_path4 - 2.43966e-05 t201_path0
   - 0.00136049569 t201_path1 + 0.00387548306 t201_path2
   - 0.00443078484 t201_path3 + 0.00166257948 t201_path4
   + 0.00185461331 t202_path0 - 0.00111788954 t202_path1
   - 0.00119412947 t202_path2 - 3.8466451200000002e-04 t202_path3
   - 0.00192728778 t202_path4 + 0.000387415 t202_path5
   + 0.00337574072 t202_path6 - 8.72986e-05 t203_path0
   - 0.00162577338 t203_path1 - 3.1842160400000002e-04 t203_path2
   - 0.00312998472 t203_path3 + 0.00169050321 t203_path4
   + 3.58721e-05 t203_path5 - 0.0002982 t203_path6
   + 0.00514168944 t203_path7 + 0.00862085633 t204_path0
   + 1.87115627e-04 t204_path1 - 0.000524832 t204_path2
   + 2.2800113899999999e-04 t204_path3 - 0.00282363733 t204_path4
   + 0.00190226838 t205_path0 + 5.46596e-05 t205_path1
   - 0.00214858679 t205_path2 - 0.00185900053 t205_path3
   + 0.00374794379 t206_path0 + 2.2710754999999999e-04 t206_path1
   - 0.00203852868 t206_path2 - 0.00450836169 t206_path3
   - 0.000120231 t207_path0 + 0.00177580479 t207_path1
   - 0.0025022414 t207_path2 + 0.0090105515 t207_path3
   - 7.5113872299999997e-04 t207_path4 + 0.00283448235 t207_path5
   + 4.7174433700000002e-04 t208_path0 - 7.0005719300000001e-04 t208_path1
   - 0.0022903448 t208_path2 + 0.00250066514 t208_path3
   + 0.00579069834 t208_path4 + 1.9255558400000001e-04 t209_path0
   + 0.00261323969 t209_path1 + 2.4063047e-04 t209_path2
   - 0.00325495284 t209_path3 - 0.00209606276 t209_path4
   - 4.92342e-05 t210_path0 - 0.00511628389 t210_path1
   + 0.00533114048 t210_path2 - 9.7858719500000003e-04 t210_path3
   + 0.00317438738 t210_path4 - 0.00174759305 t211_path0
   + 2.26691598e-04 t211_path1 - 0.00213410635 t211_path2
   - 2.3378941099999999e-04 t211_path3 + 6.6552014300000004e-04 t211_path4
   - 9.2880940099999997e-04 t211_path5 + 0.00954541843 t211_path6
   - 6.2729057399999995e-04 t212_path0 + 0.0105997743 t212_path1
   - 5.61784e-06 t212_path2 + 0.0014280465 t212_path3
   - 0.00124600285 t212_path4 + 6.76768e-05 t212_path5
   + 0.00145896478 t213_path0 + 2.84885e-05 t213_path1
   - 0.00157784426 t213_path2 - 0.00294638495 t213_path3
   + 5.1357131500000004e-04 t214_path0 - 6.1345967700000001e-04 t214_path1
   + 0.00172968546 t214_path2 - 0.000907142 t214_path3
   + 0.00159568107 t214_path4 - 0.00229590479 t214_path5
   + 0.00449209707 t214_path6 - 8.4975245399999998e-04 t215_path0
   + 0.00873776618 t215_path1 + 0.000858947 t215_path2
   + 0.0050591873 t215_path3 + 0.00024748 t215_path4
   - 7.4304960400000005e-04 t215_path5 + 3.9571285000000001e-04 t216_path0
   - 0.00166836358 t216_path1 + 0.00218727533 t216_path2
   - 7.4188149299999999e-04 t216_path3 + 0.00650743116 t216_path4
   + 0.00273647765 t217_path0 - 9.8437687800000009e-04 t217_path1
   - 0.000746225 t217_path2 + 1.03198799e-04 t217_path3
   + 0.00569834094 t217_path4 + 1.3599146e-04 t218_path0
   - 0.00106601778 t218_path1 - 0.00182674988 t218_path2
   - 0.00399381341 t218_path3 - 0.00208459538 t219_path0
   + 4.0899412099999999e-04 t219_path1 + 0.00313806557 t219_path2
   + 1.082e-05 t219_path3 + 0.00195136259 t219_path4
   - 0.00214015972 t219_path5 + 0.00132694491 t219_path6
   - 5.5559322900000001e-04 t220_path0 + 0.00294971443 t220_path1
   + 2.1886872100000001e-04 t220_path2 + 0.00475296704 t220_path3
   + 1.0583546800000001e-04 t220_path4 + 0.00165646453 t220_path5
   + 0.00464753713 t220_path6 + 4.32632e-05 t221_path0
   - 0.00177175936 t221_path1 - 0.00361449132 t221_path2
   + 0.00453913817 t221_path3 + 2.62517686e-04 t221_path4
   - 0.00248517911 t221_path5 - 0.00112314161 t222_path0
   + 0.00132448284 t222_path1 + 0.000299122 t222_path2
   - 0.00120631733 t222_path3 - 0.0046050013 t222_path4
   + 0.00145228324 t223_path0 - 0.00076052 t223_path1
   + 0.00489732297 t223_path2 - 7.7823438900000001e-04 t223_path3
   + 1.1775062000000001e-04 t223_path4 - 8.1301643499999995e-04 t223_path5
   - 0.00437305588 t223_path6 + 0.00506596453 t224_path0
   + 5.28848e-05 t224_path1 - 0.0013440731 t224_path2
   + 0.00567819877 t224_path3 + 2.7219415599999999e-04 t225_path0
   + 0.0019484068 t225_path1 - 9.9844683399999998e-04 t225_path2
   + 0.0037483694 t225_path3 - 0.00348433154 t225_path4
   + 0.00112925225 t226_path0 - 0.00265713152 t226_path1
   - 0.00175249623 t226_path2 + 0.0114988554 t226_path3
   - 4.6732911099999999e-04 t226_path4 + 0.000527025 t226_path5
   + 0.00225942791 t226_path6 - 0.000889518 t226_path7
   + 4.79336e-05 t227_path0 - 0.00312280934 t227_path1
   - 6.0751882799999998e-04 t227_path2 + 0.00566353789 t227_path3
   - 0.00178707391 t227_path4 + 0.00131278683 t228_path0
   - 1.9790265699999999e-04 t228_path1 - 0.00241944077 t228_path2
   + 0.00119158579 t228_path3 + 0.00522818789 t228_path4
   + 0.00132759777 t229_path0 + 0.000140202 t229_path1
   - 0.00498079043 t229_path3 + 0.00558081362 t229_path4
   - 0.00486052455 t230_path0 - 6.374e-05 t230_path1
   + 0.00917181652 t230_path2 + 4.9809686599999996e-04 t230_path3
   - 2.91e-05 t230_path4 - 0.00127094961 t230_path5
   - 0.00314872479 t230_path6 + 3.0519277800000001e-04 t231_path0
   + 0.00241700769 t231_path1 + 3.69585439e-04 t231_path2
   - 0.00103738694 t231_path3 - 0.00527172815 t231_path5
   + 0.00363997114 t231_path6 - 0.00187012309 t231_path7
   + 1.19347817e-04 t232_path0 - 0.00206339057 t232_path1
   - 0.00365805672 t232_path2 + 3.9213246799999998e-04 t232_path3
   + 0.00925439969 t232_path4 + 3.8949237199999999e-04 t233_path0
   - 9.8585244299999992e-04 t233_path1 - 0.00224351976 t233_path2
   + 0.00191873219 t233_path3 - 0.00107608573 t233_path4
   - 0.00382866967 t233_path5 + 0.00289950566 t233_path6
   - 0.00447130436 t234_path0 + 3.3622118600000001e-04 t234_path1
   - 1.39880154e-04 t234_path2 + 0.00181967823 t234_path3
   - 0.00210449332 t234_path4 + 0.0086479187 t234_path5
   + 0.00440281443 t235_path0 - 8.4987556299999996e-04 t235_path1
   - 1.00995509e-04 t235_path2 - 0.00273077027 t235_path3
   + 0.00544502027 t235_path4 + 3.4689163999999999e-04 t236_path0
   - 0.00124781742 t236_path1 - 0.00139640109 t236_path2
   - 3.0295952500000001e-04 t236_path3 - 0.00251525431 t236_path4
   + 2.2119721699999999e-04 t236_path5 - 0.00194301479 t237_path0
   + 2.1809065899999999e-04 t237_path1 - 0.00289399223 t237_path2
   - 0.00145072269 t237_path3 + 0.00148865231 t238_path0
   - 0.000135139 t238_path1 - 0.00138299179 t238_path2
   - 0.00114893075 t238_path3 + 0.0021455856 t238_path4
   + 0.00932715461 t238_path5 + 3.3013135499999999e-04 t239_path0
   + 0.0026459191 t239_path1 + 0.00109056965 t239_path2
   - 0.000585942 t239_path3 - 0.00176710845 t239_path4
   - 4.43362864e-04 t239_path5 + 0.00413649902 t239_path6
   + 0.00145143794 t240_path0 - 8.8383746300000005e-04 t240_path1
   + 7.22954e-05 t240_path2 + 0.00183427252 t240_path3
   - 0.00214131107 t240_path4 - 0.0044108904 t240_path5
   + 0.00514017139 t241_path0 + 6.4703787200000003e-04 t241_path1
   + 0.00512567908 t241_path2 - 3.49348e-05 t241_path3
   - 0.00348596415 t241_path4 - 0.00228291983 t242_path0
   - 3.2956415100000001e-04 t242_path1 + 0.00150711066 t242_path2
   + 5.88582e-05 t242_path3 - 0.00249068718 t242_path4
   - 2.23588e-05 t243_path0 + 0.0018569564 t243_path1
   - 0.00363136549 t243_path2 + 5.5037211899999999e-04 t243_path3
   + 0.00508007454 t243_path4 - 1.1657823800000001e-04 t244_path0
   - 0.0042176689 t244_path1 + 0.00409394363 t244_path2
   + 0.0065394952 t244_path3 + 0.00361436768 t245_path0
   - 8.0776674399999997e-04 t245_path1 - 2.9887745000000001e-04 t245_path2
   + 3.6450629700000002e-04 t245_path3 - 0.00120888627 t245_path4
   - 3.0983827299999999e-04 t245_path5 - 0.0021545 t245_path6
   + 0.00117985089 t246_path0 - 4.3443069400000002e-04 t246_path1
   + 2.2287925799999999e-04 t246_path2 - 0.00289490656 t246_path3
   + 0.000458228 t247_path0 + 0.00774591 t247_path1
   - 0.000959201 t247_path2 + 1.4756934299999999e-04 t247_path3
   - 0.00176002213 t247_path4 + 2.1451164499999999e-04 t248_path0
   + 0.00531575875 t248_path1 + 0.0112847826 t248_path2
   - 1.6135735500000001e-04 t248_path4 - 0.00177354785 t248_path5
   + 6.1185460099999998e-04 t248_path6 - 0.00113312027 t248_path7
   - 3.5636423899999999e-04 t249_path0 - 0.0033805382 t249_path1
   + 0.00014462 t249_path2 + 0.00265427376 t249_path3
   + 0.00472077494 t249_path4 - 0.000293176 t250_path0
   + 0.00113418349 t250_path1 - 0.00345080416 t250_path2
   + 0.000641144 t250_path3 - 0.00118860265 t251_path0
   + 7.17064e-05 t251_path1 + 0.00118327641 t251_path2
   - 0.00238022674 t251_path3 + 0.00396060292 t251_path4
   + 0.00335873431 t252_path0 - 0.00270171743 t252_path2
   + 0.00239010598 t252_path3 - 2.57805e-06 t252_path4
   - 0.00163466483 t252_path5 - 8.01882e-06 t252_path6
   + 0.00502102589 t252_path7 - 4.56738e-05 t253_path0
   - 0.00193453778 t253_path1 - 0.00242305128 t253_path2
   + 0.00371805113 t253_path3 - 0.00237564743 t253_path4
   + 0.0103309685 t253_path5 + 0.00549242 t254_path0
   - 7.11977e-05 t254_path1 + 0.00373166031 t254_path2
   - 0.00193886086 t254_path3 - 6.23599452e-04 t255_path0
   + 0.00361407129 t255_path1 + 1.0774e-05 t255_path2
   + 9.6435233699999998e-04 t255_path3 - 0.00104780344 t255_path4
   + 0.00139654393 t255_path5 + 0.00989516824 t255_path6
   - 0.00191599713 t256_path0 + 0.00181284233 t256_path1
   - 1.2294459199999999e-04 t256_path2 + 0.0020435038 t256_path3
   - 0.00151783053 t256_path4 + 0.00672136107 t256_path5
   + 6.9036911000000002e-04 t257_path0 - 0.00340480544 t257_path1
   + 0.00903323106 t257_path2 + 0.00112355198 t257_path3
   + 0.00250752852 t257_path4 - 1.58828392e-04 t257_path5
   - 0.00289178523 t257_path6 + 0.00164522417 t258_path0
   + 7.44413e-05 t258_path1 - 0.00129946624 t258_path2
   - 0.00243842648 t258_path3 + 6.0318387099999996e-04 t258_path4
   + 0.00117152603 t259_path0 - 9.8077e-05 t259_path1
   - 2.28508e-05 t259_path2 - 9.8764523900000002e-04 t259_path3
   - 0.00210561021 t259_path4 + 6.3973333500000004e-04 t260_path0
   + 0.00532145519 t260_path1 - 1.88202786e-04 t260_path2
   + 0.00267905765 t260_path3 - 3.2032e-05 t260_path4
   + 0.00604429469 t260_path5 - 3.73411e-05 t261_path0
   + 0.00154237507 t261_path1 - 0.00379871484 t261_path2
   + 0.00427913107 t261_path3 + 0.00419620611 t262_path0
   - 5.5934e-05 t262_path1 - 0.00194463949 t262_path2
   - 4.1574824700000001e-04 t262_path3 + 0.00261541316 t262_path4
   - 0.00200992543 t263_path0 - 4.8798e-05 t263_path1
   - 0.00213942723 t263_path2 + 7.56919e-05 t263_path3
   + 0.00904431287 t263_path4 + 0.00177308626 t263_path5
   + 0.00408255216 t264_path0 + 0.00186593214 t264_path1
   - 0.00300318166 t264_path2 - 1.5956540300000001e-04 t264_path3
   + 0.0024440689 t264_path4 - 5.1107222699999996e-04 t265_path0
   - 0.00538049592 t265_path1 + 0.00201728591 t265_path2
   + 9.50387e-05 t265_path3 + 0.00454309583 t265_path4
   + 0.000190981 t266_path0 - 0.00143905089 t266_path1
   - 0.00546398014 t266_path2 + 0.00130339409 t266_path3
   - 7.3680351499999996e-04 t266_path4 - 0.00364414044 t266_path5
   + 0.00233197 t266_path6 + 1.2354415999999999e-04 t267_path0
   - 7.3559622999999998e-04 t267_path1 - 0.00405685883 t267_path2
   + 0.000329477 t267_path3 + 0.00359073491 t267_path4
   + 0.00554084405 t267_path5 + 2.21941358e-04 t268_path0
   - 0.000819201 t268_path1 - 0.00242301845 t268_path3
   + 0.00757386303 t268_path4 - 4.2457255800000001e-04 t268_path5
   - 4.97551e-05 t269_path0 - 0.00175292813 t269_path1
   - 0.000707501 t269_path2 + 0.00326993829 t269_path3
   - 0.00375807681 t269_path4 - 2.4939863900000001e-04 t270_path0
   + 0.00639411481 t270_path1 + 4.71886e-05 t270_path2
   + 0.00334672257 t270_path3 - 6.2326487399999998e-04 t270_path4
   + 0.00716247 t270_path5 - 0.00442833919 t271_path0
   + 1.4622333400000001e-04 t271_path1 + 9.82355e-05 t271_path2
   - 9.3874812600000003e-04 t271_path3 - 0.00105954439 t271_path4
   + 0.00927618705 t271_path5 - 0.00433454383 t272_path0
   - 1.03450657e-04 t272_path1 + 0.0024835 t272_path2
   - 0.00135959743 t272_path3 + 0.00113346463 t272_path4
   - 0.00266743265 t272_path5 + 0.00201791618 t272_path6
   + 0.00742451521 t272_path7 - 3.4346629400000003e-04 t273_path0
   + 0.00359063689 t273_path1 - 0.00330649433 t273_path2
   + 6.6896458200000003e-04 t273_path3 + 0.00547504332 t273_path4
   - 4.00486373e-04 t274_path0 - 0.00366037036 t274_path1
   - 1.84193821e-04 t274_path2 + 6.7177915499999998e-04 t274_path3
   - 0.0028903354 t274_path4 + 0.00621055719 t275_path0
   - 6.3371221800000003e-04 t275_path1 + 7.4485671899999996e-04 t275_path2
   - 2.2396835300000001e-04 t275_path3 + 0.00468770042 t275_path4
   + 0.00147050933 t276_path0 + 1.62687e-06 t276_path1
   - 0.00136045087 t276_path2 + 0.00377469463 t276_path4
   - 2.33378e-05 t277_path0 + 0.00617275201 t277_path1
   - 0.00147706945 t277_path2 + 0.00146086968 t277_path3
   - 0.00259306352 t277_path5 + 0.00432141405 t277_path6
   - 0.00337299518 t278_path0 + 4.2686378599999999e-04 t278_path1
   - 0.00063616 t278_path2 - 0.0021892956 t278_path3
   - 0.00507613178 t279_path0 - 1.09629713e-04 t279_path1
   - 0.000168877 t279_path2 + 0.000606462 t279_path3
   - 0.00160181394 t279_path4 + 0.00150534371 t279_path5
   - 0.00207138457 t280_path0 + 0.00102000986 t280_path1
   + 0.0051493044 t280_path2 - 0.000170987 t280_path3
   + 0.00345125422 t280_path4 + 0.00247156643 t280_path5
   - 0.00201363 t280_path6 + 0.0038503136 t281_path0
   + 0.000510216 t281_path1 - 0.000483895 t281_path2
   + 3.9146453499999998e-04 t281_path3 + 0.00475307135 t281_path4
   - 6.4398453100000005e-04 t282_path0 + 1.74858811e-04 t282_path1
   - 0.00318361796 t282_path2 + 0.00484835077 t282_path3
   + 0.00386169809 t283_path0 - 6.49289e-05 t283_path1
   - 9.1451447200000002e-04 t283_path2 + 8.05617e-05 t283_path3
   + 2.2698544500000001e-04 t283_path4 + 0.00418088026 t283_path5
   + 1.4291050299999999e-04 t284_path0 - 0.00288824295 t284_path1
   - 0.00158694596 t284_path2 + 0.00739445258 t284_path3
   - 0.00380889582 t284_path4 + 0.00223277416 t285_path0
   + 3.35463e-05 t285_path1 - 4.9345928699999996e-04 t285_path2
   + 0.00218630931 t285_path3 + 0.00751385605 t285_path4
   - 6.4810167500000001e-04 t285_path5 - 0.000123455 t286_path0
   + 0.00700934045 t286_path1 + 1.55661e-05 t286_path2
   - 0.0022793063 t286_path3 + 0.00366440089 t286_path4
   - 7.9380057299999998e-04 t286_path5 + 0.00752031105 t287_path0
   - 0.00123848696 t287_path1 - 1.58184281e-04 t287_path2
   + 5.1618518799999996e-04 t287_path3 - 0.00377571257 t287_path4
   + 8.8647031200000004e-04 t287_path5 - 0.000256852 t288_path0
   - 0.00362383109 t288_path1 + 0.000524015 t288_path2
   - 0.000748236 t288_path3 + 0.00531673897 t288_path4
   - 2.49406e-05 t288_path5 + 0.000418141 t289_path0
   - 0.00140096224 t289_path1 - 0.00253972411 t289_path2
   + 0.00114682 t289_path3 - 0.00379769574 t289_path4
   - 0.000173191 t289_path5 + 1.47766215e-04 t290_path0
   - 0.00314578507 t290_path1 - 3.7835980800000001e-04 t290_path2
   - 0.00253154058 t290_path3 + 8.4055145299999996e-04 t290_path4
   + 0.00486761937 t290_path5 + 6.27366e-05 t291_path0
   - 0.00346763525 t291_path1 - 0.000826925 t291_path2
   - 0.00401142985 t291_path3 + 0.00395620847 t291_path4
   - 0.00352033926 t292_path0 + 0.000517997 t292_path1
   + 8.52992e-05 t292_path2 - 0.00401593931 t292_path3
   - 0.00239975704 t292_path4 + 3.9791179000000002e-04 t293_path0
   - 2.53668346e-04 t293_path1 - 0.00182697643 t293_path2
   + 8.9287053599999995e-04 t293_path3 - 0.00282241148 t293_path4
   - 1.1387186499999999e-04 t294_path0 - 0.00339871133 t294_path1
   + 0.00483922428 t294_path2 + 0.00464526471 t294_path4
   + 1.59116855e-04 t295_path0 - 0.00321378186 t295_path1
   + 0.00157341524 t295_path2 - 0.00182101491 t295_path3
   + 0.000188932 t295_path4 - 6.7284429700000003e-04 t295_path5
   - 0.000122089 t296_path0 + 0.00173406524 t296_path1
   - 0.00340278423 t296_path2 - 0.00151683472 t296_path3
   + 0.00193712453 t296_path4 + 0.00594010809 t296_path5
   + 2.0711979600000001e-04 t297_path0 - 0.00253592432 t297_path1
   - 0.0010065902 t297_path2 - 0.00152814423 t297_path3
   + 0.00102922658 t298_path0 - 3.7586656999999998e-04 t298_path1
   + 6.9690641200000005e-04 t298_path2 - 7.10505177e-04 t298_path3
   + 0.0053263274 t298_path4 + 0.00252569746 t299_path0
   - 4.4855123299999999e-04 t299_path1 + 0.00213559112 t299_path2
   + 0.000180841 t299_path3 - 0.00135933131 t299_path4
   + 0.0019838782 t299_path5 + 0.00720849726 t299_path6
   + 1.58873358e-04 t300_path0 - 0.00115301809 t300_path1
   - 0.00322986091 t300_path2 - 0.00398380356 t300_path3
   - 0.00120848475 t300_path4 + 0.00484970491 t300_path5
   - 3.4877139799999998e-04 t301_path0 + 0.00296033896 t301_path1
   - 6.1520020200000005e-04 t301_path2 + 1.71133317e-04 t301_path3
   + 0.00489132106 t301_path4 - 7.3663832199999998e-04 t302_path0
   + 1.9608880400000001e-04 t302_path1 - 0.00118370447 t302_path2
   - 0.00326862186 t302_path3 + 1.6021485499999999e-04 t303_path0
   - 9.9454948199999998e-04 t303_path1 - 0.00291492417 t303_path2
   - 0.00104555523 t303_path3 + 0.00812408794 t303_path4
   - 9.5099623999999998e-04 t304_path0 + 1.80124349e-04 t304_path1
   + 7.3888147000000001e-04 t304_path2 + 0.00383320893 t304_path3
   - 0.00166422955 t304_path4 + 0.00199022307 t305_path1
   - 1.23078702e-04 t305_path2 + 0.00148243934 t305_path3
   + 0.00520661892 t305_path4 - 9.0381765000000001e-04 t306_path0
   + 0.00788646191 t306_path1 - 0.000841667 t306_path2
   + 0.00276447274 t306_path3 - 0.0017813443 t306_path4
   - 1.71705e-05 t306_path5 + 1.28116e-05 t307_path0
   - 0.00395599427 t307_path1 + 0.00189880235 t307_path2
   + 1.77854847e-04 t307_path3 - 0.000747022 t307_path4
   - 0.00428922288 t307_path5 + 2.6954617400000002e-04 t308_path0
   - 8.0888014099999997e-04 t308_path1 - 0.00206334912 t308_path2
   + 8.1080343800000001e-04 t308_path3 - 0.00147733313 t308_path4
   - 1.24125378e-04 t309_path0 - 0.00386918196 t309_path1
   + 2.5943131200000002e-04 t309_path2 + 0.00634328229 t309_path3
   + 0.00678100064 t309_path4 - 7.8538153299999997e-04 t309_path5
   + 0.000536933 t310_path0 - 0.0036119679 t310_path1
   + 0.00215767254 t310_path2 - 3.6949e-05 t310_path3
   + 0.00365578989 t310_path4 + 3.9581954499999999e-04 t311_path0
   - 0.00208416721 t311_path1 - 4.1437486700000002e-04 t311_path2
   - 0.00160176645 t311_path3 + 0.00165890774 t312_path0
   + 2.6512972499999999e-04 t312_path1 - 0.00102446356 t312_path2
   - 3.22167e-05 t312_path3 + 0.00560738845 t312_path4
   + 0.00133982208 t313_path0 - 0.00257796329 t313_path1
   + 0.00833278615 t313_path2 + 5.7960068799999996e-04 t313_path3
   - 0.00151081663 t313_path4 + 1.28523e-05 t313_path5
   + 0.00454950612 t313_path6 + 0.00200938107 t314_path0
   - 5.40281e-05 t314_path1 - 5.7770241999999998e-04 t314_path2
   + 0.000150492 t314_path3 + 7.6886039499999995e-04 t314_path4
   + 0.00423714146 t314_path5 - 0.0017895808 t314_path6
   - 0.00257282355 t315_path0 + 8.5290061500000001e-04 t315_path1
   + 0.00119856338 t315_path2 - 4.15998e-05 t315_path3
   - 0.00303742406 t315_path4 + 0.00216715457 t316_path0
   + 9.87014e-05 t316_path1 + 3.4176014000000002e-04 t316_path2
   + 0.0073144543 t316_path3 - 0.00108855055 t316_path4
   - 5.17360168e-04 t317_path0 + 1.7728147200000001e-04 t317_path1
   - 0.00172652153 t317_path2 + 0.00171588711 t317_path3
   + 0.00354841375 t318_path0 - 6.97737e-05 t318_path1
   - 0.0016958334 t318_path2 + 0.00260068662 t318_path3
   - 2.6291116999999998e-04 t319_path0 + 0.00295241131 t319_path1
   + 8.23687e-06 t319_path2 + 0.00240364322 t319_path3
   + 0.00154703076 t319_path4 + 0.00477596652 t320_path0
   - 4.6765865400000002e-04 t320_path1 - 1.68398183e-04 t320_path2
   - 0.00200196262 t320_path3 + 0.00496319355 t320_path4
   + 0.000490511 t320_path5 + 0.00439156778 t320_path6
   + 1.8520858399999999e-04 t321_path0 - 6.1468756800000005e-04 t321_path1
   + 0.00425115647 t321_path2 - 0.00546862464 t321_path3
   - 0.00547229545 t322_path0 - 2.39806e-05 t322_path1
   - 0.00170430134 t322_path2 + 9.1492035400000005e-04 t322_path3
   + 0.00176575838 t322_path4 + 4.60345e-05 t322_path5
   - 0.00120398414 t322_path6 + 1.8011948900000001e-04 t323_path0
   - 4.5238624399999999e-04 t323_path1 + 0.00341132563 t323_path2
   - 0.00141889043 t323_path3 - 0.000321079 t324_path0
   - 0.0022959339 t324_path1 + 0.00267543341 t324_path2
   + 0.00258223596 t324_path3 - 0.000492911 t324_path4
   - 3.74294e-05 t324_path5 + 0.00250418065 t324_path6
   - 3.2058279599999999e-04 t325_path0 + 2.9336058699999998e-04 t325_path1
   + 0.00478067528 t325_path2 - 0.00189606089 t325_path4
   + 0.00127413636 t326_path0 - 1.75343e-05 t326_path1
   - 0.00161477295 t326_path2 - 0.0024579973 t326_path3
   + 0.00132498739 t327_path0 + 1.73690671e-04 t327_path1
   - 7.45741e-05 t327_path2 - 0.00146552688 t327_path3
   - 0.00120858278 t327_path4 + 2.2750486100000001e-04 t328_path0
   + 0.00507151894 t328_path1 + 2.04943717e-04 t328_path2
   - 0.00254847761 t328_path3 - 1.42290141e-04 t328_path4
   + 0.00169705914 t328_path5 + 0.00238882774 t329_path1
   + 0.00281935278 t329_path2 - 0.00112520216 t329_path3
   + 0.000571332 t329_path4 + 0.00366683095 t329_path5
   - 0.00142618536 t329_path6 + 1.2879574200000001e-04 t330_path0
   + 0.00585116027 t330_path1 + 0.00014276 t330_path2
   - 0.00132541044 t330_path3 - 0.00240255473 t330_path4
   + 6.8793044199999998e-04 t330_path5 - 8.38667e-05 t331_path0
   - 0.0024010248 t331_path1 - 0.00171459641 t331_path2
   + 0.00263206474 t331_path3 + 0.00419934792 t331_path4
   + 9.6414849400000004e-04 t332_path0 + 3.47737e-05 t332_path1
   - 0.00109788124 t332_path2 + 0.0062632584 t332_path3
   + 4.09268e-05 t333_path0 - 0.00394835416 t333_path1
   + 0.00231287722 t333_path2 + 1.6537464400000001e-04 t334_path0
   + 0.00309188012 t334_path1 - 2.53202e-05 t334_path2
   - 0.00109486771 t334_path3 - 8.8897277599999996e-04 t334_path4
   + 9.6699077399999998e-04 t334_path5 + 0.00886969082 t334_path6
   + 5.2639783799999997e-04 t335_path0 - 0.0016338682 t335_path1
   - 0.00104343833 t335_path2 + 4.48553183e-04 t335_path3
   - 1.5567545799999999e-04 t335_path4 - 0.00187155872 t335_path5
   - 0.00160752167 t336_path0 + 3.1359141600000002e-04 t336_path1
   - 9.6286024199999996e-04 t336_path2 + 5.9848232200000002e-04 t336_path3
   - 0.00189860119 t336_path4 + 0.00199509435 t336_path5
   + 0.000590066 t336_path6 - 0.00220652088 t336_path7
   + 0.000795731 t337_path0 - 2.9138824899999999e-04 t337_path1
   - 0.00142083864 t337_path2 + 3.64473e-05 t337_path3
   + 0.00032395 t337_path4 - 0.00113447732 t337_path5
   - 0.00367750041 t338_path0 + 0.00122654194 t338_path1
   - 2.2252574899999999e-04 t338_path2 - 0.00193526736 t338_path3
   + 0.00471969973 t338_path4 - 4.8723581099999999e-04 t338_path5
   + 5.0463870899999995e-04 t338_path6 - 1.17629337e-04 t339_path0
   + 9.6140347900000003e-04 t339_path1 - 0.00278302026 t339_path2
   + 0.00128122873 t339_path3 - 3.8778586900000003e-04 t339_path4
   + 0.00536413677 t339_path5 - 0.00132108782 t340_path0
   + 0.00147629 t340_path1 - 0.00117163546 t340_path2
   + 0.00104023027 t340_path3 + 0.00484620966 t340_path4
   + 1.2337842799999999e-04 t340_path5 + 0.00254626852 t340_path6
   - 2.86039e-05 t341_path0 - 0.00111355365 t341_path1
   + 0.00553426892 t341_path2 - 0.00123084709 t342_path0
   - 3.6714240599999999e-04 t342_path1 + 0.00451623881 t342_path2
   + 1.78188202e-04 t342_path3 + 0.00235758652 t342_path4
   + 8.9472380899999996e-04 t343_path0 - 1.01114449e-04 t343_path1
   - 0.00394161139 t343_path2 + 0.00475047017 t343_path4
   + 3.8082379599999999e-04 t344_path0 - 0.00262577645 t344_path1
   + 0.00469167205 t344_path2 + 0.00045507 t344_path3
   - 1.82502874e-04 t344_path4 + 5.7015364300000003e-04 t344_path5
   + 0.00414394168 t344_path6 - 5.4144015299999995e-04 t345_path0
   + 0.00437270664 t345_path1 - 0.00313811586 t345_path2
   + 2.6947187100000001e-04 t345_path3 - 9.11347859e-04 t345_path4
   - 9.3254e-05 t345_path5 + 0.00405573705 t345_path6
   + 0.00139602 t346_path0 - 3.6008350400000002e-04 t346_path1
   + 3.2259989500000002e-04 t346_path2 + 0.0024625957 t346_path3
   - 0.000830595 t346_path4 + 8.99141e-05 t347_path0
   - 0.00227420754 t347_path1 - 0.00256343302 t347_path2
   - 0.00367106148 t347_path3 - 3.2897887299999999e-04 t348_path0
   + 3.9366612299999998e-04 t348_path1 - 0.00101511332 t348_path2
   - 5.4611428600000005e-04 t348_path3 - 0.003374035 t348_path4
   + 9.19571e-05 t349_path0 - 0.00240908726 t349_path1
   - 0.00245076302 t349_path2 + 9.8818866500000003e-04 t349_path3
   + 0.00690844795 t349_path4 - 3.2475148299999999e-04 t350_path0
   + 0.00973814446 t350_path1 - 2.9134179899999998e-04 t350_path2
   + 6.0967245400000004e-04 t350_path3 - 0.00144730578 t350_path4
   - 2.1780614e-04 t350_path5 + 4.5981982799999998e-04 t350_path6
   - 0.00104373146 t351_path0 - 1.92786e-06 t351_path1
   + 0.00208109221 t351_path2 - 0.00127779273 t351_path3
   + 0.00186203606 t351_path4 - 0.00113667944 t351_path5
   + 7.73825e-05 t352_path0 - 0.00135260168 t352_path1
   + 6.2375e-05 t353_path0 - 0.00325105037 t353_path1
   - 0.00306166941 t353_path2 + 0.00588754145 t353_path4
   - 1.15324125e-04 t354_path0 + 0.00272124121 t354_path1
   - 3.3509e-07 t354_path2 + 0.00180337729 t355_path0
   - 0.00101348816 t355_path1 - 0.0018810227 t355_path2
   + 8.35823e-06 t355_path3 - 3.6670724500000002e-04 t355_path4
   + 0.00122590235 t355_path5 - 0.00281038252 t355_path6
   + 9.4813259799999996e-04 t356_path0 - 0.00461664191 t356_path1
   + 1.6723187599999999e-04 t356_path2 + 2.66287389e-04 t356_path3
   - 0.00154892378 t356_path4 - 0.00185120688 t356_path5
   - 1.93905711e-04 t356_path6 + 1.69247e-05 t357_path0
   + 0.00356943929 t357_path1 - 0.00181192113 t357_path2
   + 0.0013361465 t357_path3 + 6.07915e-06 t358_path0
   + 8.0617424100000001e-04 t358_path1 - 0.00104942918 t358_path2
   - 0.00565980794 t358_path3 - 0.000246672 t358_path4
   + 6.4163270900000004e-04 t358_path5 - 0.00307267508 t358_path6
   - 0.000198361 t359_path0 + 4.4679967700000001e-04 t359_path1
   + 7.1381719300000004e-04 t359_path2 - 0.00323616038 t359_path3
   - 9.7323471e-04 t359_path4 + 0.00727135781 t359_path5
   + 5.4081249999999995e-04 t359_path6 + 0.00438145734 t360_path0
   + 2.39074e-05 t360_path1 + 0.00131875603 t361_path0
   - 8.9658022599999998e-04 t361_path1 - 1.57023e-05 t361_path2
   + 0.00226311898 t361_path3 + 0.0030483352 t362_path0
   + 2.7732460900000002e-04 t362_path1 + 1.9260935399999999e-04 t362_path2
   - 0.00100800581 t362_path3 + 2.6348570799999998e-04 t362_path4
   - 1.1531873999999999e-04 t363_path0 + 8.7609561199999996e-04 t363_path1
   + 0.00123268459 t363_path2 - 0.00192875951 t363_path3
   + 0.0044528665 t363_path4 + 0.000244283 t364_path0
   - 0.0030041954 t364_path1 + 1.29988432e-04 t364_path2
   - 7.7662832300000003e-04 t364_path3 + 0.0029657376 t364_path4
   - 1.2490571000000001e-04 t365_path0 - 0.00339467661 t365_path1
   - 1.43663e-05 t365_path2 + 0.00242181052 t365_path3
   + 0.00363880442 t365_path4 + 7.0282735400000005e-04 t366_path0
   - 0.00060398 t366_path1 + 0.00358657353 t366_path2
   + 3.0503582100000002e-04 t366_path3 - 0.00132889487 t366_path4
   - 0.00200078031 t367_path0 - 3.3182e-05 t367_path1
   + 0.00164411799 t367_path2 + 0.00364097278 t367_path3
   + 0.00169384549 t368_path1 - 0.00221303455 t368_path2
   - 1.78963252e-04 t368_path3 + 0.00350425486 t368_path4
   + 3.6625811499999998e-04 t369_path0 - 5.9993780499999997e-04 t369_path1
   - 9.86127881e-04 t369_path2 + 0.00108041416 t369_path3
   - 0.00350803626 t369_path4 - 6.66822e-05 t370_path0
   + 0.00289346185 t370_path1 + 0.0029481696 t370_path2
   + 0.000934239 t371_path0 - 0.000146695 t371_path1
   - 0.00176510413 t371_path2 + 0.002228759 t371_path3
   + 0.000246258 t371_path4 - 9.53049e-06 t372_path0
   + 0.00344563974 t372_path1 - 0.00110441621 t372_path2
   + 0.00420444692 t373_path0 + 8.3548715300000003e-04 t373_path1
   - 0.00163631851 t373_path2 - 2.46095e-05 t373_path3
   - 0.00291510974 t373_path4 + 6.00542e-05 t374_path0
   - 0.0018150229 t374_path1 - 0.00175649358 t374_path2
   - 2.5747044100000001e-04 t374_path3 + 7.6758617099999997e-04 t374_path4
   + 0.00571370265 t374_path5 + 1.5558078300000001e-04 t375_path0
   - 0.00259509939 t375_path1 + 0.00570314098 t375_path2
   - 0.00162639795 t375_path3 + 2.23543e-05 t376_path0
   - 1.11490153e-04 t376_path1 + 0.00270636939 t376_path2
   - 0.000870607 t376_path3 - 4.10596e-05 t377_path0
   - 0.000119988 t378_path0 + 0.0062052994 t378_path1
   - 8.10791e-05 t378_path2 + 0.00154903985 t379_path0
   - 1.5084838399999999e-04 t379_path1 - 0.00115351565 t379_path2
   + 0.00167667412 t379_path3 - 0.00125871366 t379_path4
   + 9.66369e-05 t380_path0 - 0.00128897745 t380_path1
   - 0.00360955158 t381_path0 + 4.2576077999999999e-04 t381_path1
   + 0.00407692138 t381_path2 + 2.0025757800000001e-04 t381_path3
   - 7.91411e-05 t381_path4 + 0.00573411537 t382_path1
   - 0.00182594161 t382_path2 - 1.02225873e-04 t382_path3
   - 5.58846e-05 t382_path4 + 0.003298203 t382_path5
   + 0.00284459046 t383_path0 - 1.26051324e-04 t383_path1
   + 0.000514235 t383_path2 + 0.00816446356 t383_path3
   - 0.000126954 t383_path4 + 0.00122088904 t383_path5
   + 0.00349958916 t383_path6 + 0.00107319234 t384_path0
   - 0.00131920539 t384_path1 + 0.00109354616 t384_path2
   + 6.04226e-05 t384_path3 - 0.001343232 t384_path4
   - 0.00037517 t385_path0 + 0.000280224 t385_path1
   + 0.00290644704 t385_path2 + 0.00811601803 t385_path3
   + 4.1769247000000002e-04 t385_path4 - 0.0034200931 t386_path0
   - 3.64248e-05 t386_path1 - 0.00115289621 t386_path2
   + 0.00493885716 t386_path3 - 7.49526e-05 t387_path0
   - 0.00137458125 t387_path1 + 0.00326618855 t387_path2
   + 0.00619797874 t387_path3 + 4.5852363099999999e-04 t387_path4
   + 0.00339740049 t388_path0 - 0.00158210401 t388_path1
   - 0.000143282 t388_path2 + 0.00127880485 t388_path3
   - 0.000378296 t389_path0 - 0.00329512078 t389_path1
   + 0.000125669 t389_path2 - 0.00159044052 t389_path3
   + 8.4482930800000003e-04 t390_path0 - 9.58778e-06 t390_path1
   - 0.00288443617 t390_path2 - 0.00379512133 t391_path0
   + 8.15551e-05 t391_path1 + 0.00296520512 t391_path2
   + 0.00389235443 t391_path3 + 1.20911893e-04 t392_path0
   - 0.00109406374 t392_path1 - 6.9375260500000003e-04 t392_path2
   + 0.00197936734 t392_path3 + 3.77216e-05 t393_path0
   - 0.00111777405 t393_path1 + 3.0965116500000003e-04 t393_path2
   + 0.00172916299 t393_path3 + 0.00397514738 t393_path4
   - 3.50809e-05 t394_path0 + 8.9998776099999998e-04 t394_path1
   + 2.9985737600000001e-04 t395_path0 - 4.7913927100000001e-04 t395_path1
   - 0.00120976509 t395_path2 + 0.0027840538 t395_path3
   + 0.00522371242 t395_path4 - 0.00284729502 t396_path0
   + 0.00709749153 t396_path1 - 0.00123687042 t396_path2
   - 3.00129e-05 t396_path3 + 0.0036496541 t396_path4
   + 6.2671e-05 t397_path0 - 0.00157591247 t397_path1
   + 0.00552642578 t397_path2 + 0.00674045598 t397_path4
   + 3.96722e-05 t398_path0 - 0.00319606625 t398_path1
   - 1.01516111e-04 t398_path2 - 0.00150868483 t398_path3
   - 0.00152912701 t399_path0 + 0.00251390133 t399_path1
   + 1.1560982200000001e-04 t399_path2 + 0.00351572316 t399_path3
   + 1.5940518600000001e-04 t400_path0 - 5.99127263e-04 t400_path1
   + 0.00381097454 t400_path2 - 4.53919e-05 t401_path0
   + 0.00564953871 t402_path0 - 0.00271584233 t402_path2
   - 3.18203e-05 t402_path3 + 0.0015112285 t402_path4
   - 6.86742e-05 t403_path0 - 0.00143347366 t403_path1
   + 0.00622800551 t403_path2 + 3.8811678000000001e-04 t403_path3
   + 0.00357764028 t403_path4 + 1.4120044900000001e-04 t404_path0
   - 0.00316424388 t404_path1 - 9.43218e-05 t404_path2
   + 8.0756814000000003e-04 t404_path3 - 5.67092e-05 t404_path4
   - 0.0022607781 t404_path5 + 0.00405680668 t405_path0
   - 6.5469043299999999e-04 t405_path1 - 0.00115582917 t405_path2
   - 4.26802e-05 t405_path3 + 0.00216861605 t405_path4
   - 5.18724e-05 t406_path0 + 1.3569834100000001e-04 t407_path0
   + 0.00133405684 t408_path0 - 3.89816e-05 t408_path1
   + 0.00642907387 t408_path2 - 0.00327889412 t408_path4
   + 0.00154963485 t409_path0 - 0.00291140634 t409_path1
   + 0.00187251053 t409_path2 + 7.90314e-05 t409_path3
   - 0.00199257606 t409_path4 + 0.00182182936 t409_path5
   - 0.000123291 t410_path0 + 0.000493505 t410_path1
   - 0.00122180837 t410_path2 - 0.00296994438 t410_path3
   + 1.02676255e-04 t411_path0 - 0.00183412735 t411_path1
   - 0.00147866795 t411_path2 - 1.3781222500000001e-04 t412_path0
   + 1.657424e-04 t412_path1 + 0.00224904669 t412_path2
   + 7.0841325200000005e-04 t413_path0 - 1.7018542e-04 t413_path1
   - 0.00127953081 t413_path2 - 0.0031722053 t413_path3
   - 9.6181e-05 t413_path4 + 0.000149366 t414_path0
   + 0.00138157711 t414_path1 - 0.00148376951 t414_path2
   - 0.00133324484 t414_path3 + 0.00179355696 t415_path0
   + 1.8799344300000001e-04 t415_path1 - 0.000723069 t415_path2
   + 1.4499225700000001e-04 t415_path3 - 0.00152141228 t415_path4
   + 8.1245641899999998e-04 t416_path0 - 0.00248082401 t416_path1
   + 0.00874423329 t416_path2 + 0.000800903 t416_path3
   - 9.07759e-05 t416_path4 + 0.00321902148 t416_path5
   - 0.00285623595 t416_path6 + 2.53374e-05 t417_path0
   - 7.10422e-06 t418_path0 - 0.00395803805 t418_path1
   - 4.5501574600000002e-04 t419_path0 + 0.00510223862 t419_path1
   + 0.00102367788 t419_path2 - 0.0017822017 t419_path3
   + 0.00154519407 t419_path4 + 2.02048861e-04 t419_path5
   - 7.4598717000000001e-04 t419_path6 + 1.89153579e-04 t419_path7
   - 2.20677e-05 t420_path0 + 0.00704177702 t421_path1
   - 0.001020064 t421_path2 - 9.79082e-05 t421_path3
   + 0.00362774753 t422_path0 + 2.5098555500000002e-04 t422_path1
   - 2.6379738100000002e-04 t422_path2 + 0.00156404963 t422_path3
   + 0.00287776673 t422_path4 + 0.00159647933 t423_path0
   + 5.85356e-05 t423_path1 - 0.00161538564 t423_path2
   - 0.00311105628 t423_path3 + 0.00656476011 t424_path0
   - 0.00122000021 t424_path1 + 0.00125343387 t424_path2
   - 8.5437722699999998e-04 t424_path3 - 2.21706e-06 t424_path4
   - 0.00196333393 t424_path5 + 0.00112953514 t425_path0
   - 6.66121e-05 t425_path1 - 0.00167220412 t425_path2
   - 0.00111985102 t426_path0 + 4.23271e-05 t426_path1
   + 0.00418563979 t426_path2 - 0.00165348849 t426_path3
   + 0.00351644098 t426_path4 - 0.00367743196 t426_path5
   + 5.8089330600000005e-04 t426_path6 - 5.99333e-05 t427_path0
   + 0.00223484146 t427_path1 + 0.0037957069 t427_path2
   - 0.00265531242 t427_path3 + 0.00112425699 t428_path0
   - 0.00129858858 t428_path1 - 0.00202616863 t428_path2
   + 4.20229e-05 t428_path3 - 0.00239025755 t428_path4
   - 2.41433765e-04 t429_path0 - 0.00178461825 t429_path1
   + 1.2924273300000001e-04 t429_path2 + 0.00388094597 t429_path3
   + 5.4773938599999997e-04 t429_path4 + 0.00592315244 t429_path5
   - 9.4874214799999998e-04 t430_path0 + 0.00206570025 t430_path1
   + 0.00333172013 t430_path2 + 4.0651843299999998e-04 t430_path3
   - 2.6200636099999998e-04 t430_path4 + 0.00361612928 t430_path5
   - 0.00229765777 t430_path6 + 1.20824e-05 t431_path0
   - 0.00164907135 t431_path1 + 0.00275625475 t431_path2
   + 0.00231019733 t431_path3 + 7.57146e-05 t432_path0
   + 0.00198957277 t432_path1 - 0.00233209273 t432_path2
   + 0.0031641759 t432_path3 - 0.00101634313 t432_path4
   - 0.0036887941 t432_path5 + 2.15669e-05 t433_path0
   - 0.0018848211 t433_path1 - 0.00173518609 t433_path2
   + 1.14133e-05 t434_path0 + 0.00197329186 t435_path0
   - 0.00116715243 t435_path1 + 1.7193031099999999e-04 t435_path2
   - 6.3971267099999999e-04 t435_path3 - 0.00151438336 t435_path4
   - 5.75961e-05 t436_path0 + 0.00225907238 t437_path0
   - 0.00330276927 t437_path1 + 5.73463e-05 t437_path2
   + 6.7041115800000001e-04 t438_path0 - 1.2953938900000001e-04 t438_path1
   + 9.94534e-05 t439_path0 - 0.00250694621 t439_path1
   + 0.00289349793 t439_path3 - 0.00102203828 t439_path4
   - 6.5353076200000003e-04 t440_path0 - 5.3174415400000005e-04 t440_path1
   + 0.00432404736 t440_path2 + 0.00378699205 t440_path3
   + 4.6106247499999998e-04 t440_path4 + 7.34576e-05 t440_path5
   + 1.66527549e-04 t441_path0 - 0.00294450903 t441_path1
   - 2.827e-05 t441_path2 - 8.1279623599999999e-04 t441_path3
   + 0.00276922062 t441_path4 + 0.000136291 t442_path0
   - 0.00235576206 t442_path1 - 0.00293414062 t442_path2
   - 1.34487e-05 t443_path0 + 0.00102591398 t444_path0
   - 1.2149760700000001e-04 t444_path1 - 0.00137945823 t444_path2
   + 0.00161178317 t444_path3 + 3.8472726000000001e-04 t444_path4
   + 8.32588e-05 t445_path0 + 0.00105086854 t445_path1
   - 0.00349826342 t445_path2 - 6.2377279399999995e-04 t446_path0
   + 0.00429079775 t446_path1 + 6.7870889299999999e-04 t446_path2
   - 1.9123719500000001e-04 t446_path3 + 2.56255e-05 t447_path0
   - 0.00159084669 t447_path1 + 0.0024496105 t447_path2
   + 1.33694e-05 t448_path0 - 0.00188902114 t448_path1
   + 0.00372307422 t448_path2 - 2.8449261999999998e-04 t449_path0
   + 0.00147053401 t449_path1 + 2.24038e-05 t449_path2
   - 0.00169612793 t449_path3 - 0.00170219922 t449_path4
   + 0.00181391276 t449_path5 - 7.1119e-05 t450_path0
   - 9.4291230199999995e-04 t451_path0 + 4.75934e-05 t451_path1
   + 0.00346613443 t451_path2 + 0.00653374707 t451_path3
   - 0.00102886849 t451_path4 + 1.8928557999999999e-04 t451_path5
   - 0.001233013 t452_path0 + 0.0015077726 t452_path1
   + 2.04389e-05 t452_path2 - 0.0044702366 t452_path3
   + 0.00181529694 t452_path4 + 1.52129185e-04 t453_path0
   + 0.00203759572 t453_path1 - 0.00161340751 t453_path2
   - 1.0846368199999999e-04 t453_path3 + 0.00585953332 t453_path4
   + 3.3895132899999997e-04 t453_path5 + 3.29889e-05 t454_path0
   + 0.00272291969 t454_path1 - 1.73655e-05 t455_path0
   + 2.44113e-05 t456_path0 + 9.32154e-05 t457_path0
   + 0.00420576334 t457_path1 - 5.9827574299999996e-04 t457_path2
   + 7.3989847399999996e-04 t458_path0 - 1.2489067700000001e-04 t458_path1
   + 0.00260046916 t458_path2 - 0.00241280044 t458_path3
   - 4.89966187e-04 t459_path0 + 1.8072089000000001e-04 t459_path1
   + 2.68539e-05 t460_path0 - 0.00365525507 t461_path0
   + 1.11956e-05 t461_path1 + 0.00167014566 t461_path2
   + 0.00211073062 t461_path3 - 0.00182310585 t461_path4
   + 3.84204e-05 t462_path0 - 0.00330180908 t462_path1
   + 0.0026623155 t463_path0 - 0.00154995278 t463_path2
   - 0.000113712 t463_path3 + 0.00272678374 t463_path4
   + 1.98402515e-04 t463_path5 + 0.00614634017 t463_path6
   + 3.2911344900000001e-04 t464_path0 - 0.00144829368 t464_path1
   - 2.991418e-04 t464_path2 - 0.00351026165 t465_path0
   + 0.00177727162 t465_path1 + 6.13312e-05 t465_path2
   - 0.0013338984 t465_path3 + 0.00141628331 t466_path0
   - 0.00202802522 t466_path1 + 0.000123557 t466_path2
   + 1.19685e-05 t466_path3 - 0.00191104854 t466_path4
   - 7.3682e-05 t467_path0 - 2.89562e-05 t468_path0
   - 3.13081e-05 t469_path0 + 0.00376264658 t470_path0
   + 0.000159792 t470_path1 - 0.00231862627 t470_path2
   - 1.4720457000000001e-04 t471_path0 + 0.00247577345 t471_path1
   - 6.4353575000000002e-04 t471_path2 + 0.00312447967 t471_path3
   + 7.76921e-05 t472_path0 - 2.63081e-05 t473_path0
   + 0.00129485235 t473_path1 - 0.00110369059 t473_path2
   - 0.00197753659 t474_path0 + 7.7988e-05 t474_path1
   - 0.00148342282 t474_path2 - 0.00272118254 t475_path1
   + 0.000145236 t475_path2 + 0.00314609 t476_path0
   + 1.4161242899999999e-04 t476_path1 - 0.00178857194 t476_path2
   - 8.7832e-05 t476_path3 + 0.00122044783 t477_path0
   - 2.8364255600000002e-04 t477_path1 + 0.000864946 t477_path2
   - 8.26902e-05 t477_path3 + 8.57525e-05 t478_path0
   + 0.00348374667 t478_path1 - 0.000378845 t478_path2
   - 0.00267980504 t478_path3 + 8.7085965799999995e-04 t479_path0
   - 1.81842552e-04 t479_path1 - 0.00293440721 t479_path2
   + 0.00111641025 t479_path3 - 0.00176628842 t479_path4
   + 0.00552282017 t479_path5 - 1.29792e-05 t480_path0
   + 0.00113970845 t480_path1 - 0.00158561114 t480_path2
   + 0.000766751 t480_path3 - 0.00192926906 t480_path4
   + 0.00127546699 t481_path0 - 1.4352117400000001e-04 t481_path1
   - 1.2552221600000001e-04 t482_path0 - 3.2512e-05 t483_path0
   + 1.02750164e-04 t484_path0 - 0.00226877071 t484_path1
   + 0.00341905886 t484_path2 + 0.00394834392 t484_path3
   - 1.0345382899999999e-04 t485_path0 - 3.88308e-05 t487_path0
   - 0.00292010279 t488_path0 - 0.00347999088 t488_path1
   + 5.75834e-05 t488_path2 + 2.0868027099999999e-04 t488_path3
   - 4.66644e-05 t489_path0 + 0.0017986407 t490_path1
   - 0.00103469985 t490_path2 + 3.5199627700000001e-04 t491_path0
   - 0.00136651401 t491_path1 + 6.6548e-06 t491_path2
   + 3.73000832e-04 t491_path3 + 1.41125522e-04 t492_path0
   - 0.00266780611 t492_path1 - 0.00200483133 t492_path2
   + 9.8460342299999996e-04 t493_path0 - 0.00134595216 t493_path1
   - 2.0456e-05 t493_path2 - 0.00149051705 t493_path3
   + 0.0010529228 t493_path4 - 4.54372e-06 t494_path0
   + 5.6632346199999998e-04 t494_path1 + 0.00397269428 t494_path2
   - 0.00321686221 t495_path0 - 6.21794e-05 t495_path1
   - 0.00301175099 t496_path0 - 2.88821e-05 t496_path1
   + 0.00180305575 t497_path0 - 9.6271414100000002e-04 t497_path1
   - 5.18376e-05 t497_path2 + 0.002087828 t497_path3
   + 0.000159188 t498_path0 - 0.000818547 t498_path1
   + 8.6316186899999998e-04 t498_path2 - 0.00242004986 t498_path3
   - 5.5514072300000001e-04 t498_path4 + 1.2989759900000001e-04 t499_path0
   - 0.00142675801 t499_path1 + 4.07941e-05 t500_path0
   + 0.00300787762 t500_path1 - 0.00129570067 t500_path2
   - 0.00146840478 t500_path3 + 0.00254795491 t501_path0
   - 3.93916e-05 t501_path1 + 0.00644635316 t501_path2
   - 2.5085546100000002e-04 t501_path3 + 0.00360364304 t501_path4
   - 3.8084667199999998e-04 t501_path5 + 0.00160909793 t501_path6
   + 0.00377556635 t502_path0 - 1.9624820600000001e-04 t502_path1
   - 8.75945e-05 t503_path0 - 0.000511391 t504_path0 + 0.0038615 t504_path1
   - 3.7141461500000001e-04 t504_path2 + 2.67432e-06 t504_path3
   + 2.4011217500000001e-04 t505_path0 - 5.2660086700000001e-04 t505_path1
   + 2.33877363e-04 t506_path0 - 0.00147870381 t506_path1
   - 0.00134411163 t506_path2 - 0.00290927896 t506_path3
   + 1.3383560899999999e-04 t507_path0 - 7.1626342799999999e-04 t507_path1
   - 5.3725310100000004e-04 t508_path0 + 8.8796170900000003e-04 t508_path1
   - 0.000206968 t508_path2 + 0.00105185027 t508_path3
   - 0.0010938691 t508_path4 + 0.00482804701 t508_path5
   + 1.4307655499999999e-04 t509_path0 + 0.001189298 t509_path1
   - 4.4647930199999999e-04 t509_path2 - 6.60479e-05 t510_path0
   - 3.43176e-05 t512_path0 - 0.00199722941 t513_path0
   + 9.4344420400000004e-04 t513_path1 + 0.000176679 t513_path2
   - 0.00154657848 t513_path3 - 2.02906158e-04 t513_path4
   + 0.00518112211 t513_path5 + 6.07054e-05 t514_path0
   - 0.00107792637 t515_path0 + 0.00353061664 t515_path1
   - 9.84066e-05 t515_path2 + 0.001209234 t515_path3
   - 4.41536657e-04 t516_path0 + 0.00154177763 t516_path1
   + 1.7024911400000001e-04 t516_path2 + 0.00423171325 t516_path3
   + 0.00274650869 t516_path4 - 3.1581765500000001e-04 t517_path0
   - 0.00351536577 t517_path1 - 0.00408586068 t517_path2
   + 0.00178645935 t517_path3 + 1.9385012299999999e-04 t517_path4
   - 0.00115624163 t517_path5 - 0.00106289273 t518_path0
   + 0.00153160316 t518_path1 - 1.02088714e-04 t518_path2
   - 0.00156842894 t518_path3 - 0.00458377134 t518_path4
   + 7.4460613499999998e-04 t518_path5 + 0.00367111573 t519_path0
   - 3.83902e-05 t519_path1 - 8.9053367300000001e-04 t520_path0
   + 7.7219965199999999e-04 t520_path1 - 1.9234554199999999e-04 t520_path2
   + 2.1022613499999999e-04 t520_path3 - 3.49421403e-04 t520_path4
   + 0.00566709554 t520_path5 + 0.00620563 t521_path1
   - 6.86561e-05 t521_path2 + 6.8957672900000003e-04 t521_path3
   - 0.00141615234 t521_path4 - 4.8562241099999999e-04 t522_path0
   + 0.00607134681 t522_path1 - 3.30541e-05 t522_path2
   + 0.00167784735 t522_path3 - 4.91105253e-04 t522_path4
   - 0.00327840028 t522_path5 + 6.9879135100000005e-04 t523_path0
   - 0.00181738602 t523_path1 - 0.00172861025 t523_path2
   + 0.00421786495 t523_path3 + 1.55353e-05 t523_path4
   + 0.00317366421 t524_path0 - 6.5695674900000003e-04 t524_path1
   + 3.4179657800000003e-04 t524_path2 - 3.4004950399999997e-04 t524_path3
   + 3.93534e-05 t525_path0 + 0.00219327747 t525_path1
   - 0.00127791369 t526_path0 - 2.92764e-05 t526_path1
   + 2.2527154900000001e-04 t526_path2 - 0.00143341743 t526_path3
   + 0.00648676883 t526_path5 + 3.6766988299999999e-04 t527_path0
   - 0.000275858 t527_path1 + 0.000200251 t527_path2
   - 0.00213273405 t527_path3 + 1.7803646900000001e-04 t527_path4
   + 0.00472342456 t527_path5 + 0.000176469 t528_path0
   - 0.00442330306 t528_path2 - 0.00158713059 t529_path0
   - 4.99624e-05 t529_path1 + 5.4691475800000002e-04 t529_path2
   - 7.9812639199999997e-04 t529_path3 - 0.000896441 t529_path4
   + 4.60335e-06 t530_path0 - 0.000924574 t531_path0
   + 0.00253596297 t531_path1 - 7.68673e-05 t531_path2
   - 2.28338e-05 t532_path0 - 2.95257e-06 t533_path0
   - 6.2296568600000003e-04 t533_path1 + 0.00415514037 t533_path2
   - 1.6025e-05 t534_path0 - 3.0042944099999998e-04 t535_path0
   - 0.00392391114 t535_path1 + 0.00195275829 t535_path2
   + 4.78747534e-04 t535_path3 - 0.00248388969 t535_path4
   + 6.4586e-05 t536_path0 - 4.67767e-05 t537_path0
   + 0.00341072236 t537_path1 - 4.4578479700000003e-04 t538_path0
   + 7.2296452700000005e-04 t538_path1 + 4.8754861899999999e-04 t538_path2
   - 0.000606259 t538_path3 + 0.0031384984 t538_path4
   - 4.8732239499999999e-04 t539_path0 + 2.0426273100000001e-04 t539_path1
   + 1.68165803e-04 t540_path0 - 0.00337305199 t540_path1
   + 0.00388450897 t540_path2 + 1.3507531500000001e-04 t540_path3
   - 9.0746884200000003e-04 t540_path4 - 2.6875076599999998e-04 t541_path0
   + 0.0052269008 t541_path1 - 1.30752e-05 t541_path2
   - 6.90614e-05 t542_path0 - 0.00131369 t543_path0
   + 7.72315e-05 t543_path1 - 0.00173644105 t543_path2
   + 8.3307345599999995e-04 t543_path3 + 0.00134683051 t543_path4
   + 1.66499114e-04 t543_path5 + 0.004729365 t543_path6
   + 8.8539e-05 t544_path0 - 1.77745445e-04 t545_path0
   - 1.70226e-07 t545_path1 + 0.000948351 t545_path2
   - 4.03220591e-04 t545_path3 + 0.00361446152 t545_path4
   + 3.77067e-05 t546_path0 - 0.00304618152 t546_path1
   + 0.0012369107 t547_path0 - 1.06485764e-04 t547_path1
   - 4.41002e-05 t548_path0 - 0.00100922049 t548_path1
   - 0.00493960269 t548_path2 + 0.00284574181 t548_path3
   + 1.15695722e-04 t549_path0 + 0.00560277607 t549_path1
   - 0.00115313451 t549_path2 + 4.79214e-05 t549_path3
   - 0.00159952627 t549_path4 + 0.00193451648 t549_path5
   + 0.00164477609 t550_path0 - 0.000130768 t550_path1
   + 6.8304553899999998e-04 t550_path2 - 0.0028234187 t550_path3
   + 1.9745e-05 t551_path0 + 0.00205912162 t552_path0
   + 2.3069044999999999e-04 t552_path1 - 0.00159665616 t552_path2
   + 0.00279208645 t552_path3 - 5.9856072799999995e-04 t552_path4
   - 0.00314678508 t553_path0 + 0.00162353681 t553_path1
   + 4.17552e-05 t553_path2 - 4.1509248000000002e-04 t553_path3
   - 3.80871e-05 t554_path0 - 5.6915e-05 t555_path0
   + 0.00198085629 t556_path0 + 2.9309e-05 t556_path1
   + 6.2720844300000005e-04 t557_path0 - 0.00148981495 t557_path1
   - 1.7695018299999999e-04 t557_path2 + 0.00346032134 t557_path3
   - 1.8889833900000001e-04 t558_path0 + 9.1021781600000005e-04 t558_path1
   + 0.00167126919 t558_path2 - 0.00175395014 t558_path3
   + 0.000282303 t559_path0 + 0.00203023921 t559_path1
   + 2.9544832099999999e-04 t559_path2 - 0.000810566 t559_path3
   - 7.8644760699999997e-04 t559_path4 + 0.000142536 t560_path0
   + 0.00101987505 t560_path1 - 0.00159159 t560_path2
   - 5.5332051099999996e-04 t560_path3 + 7.72139e-05 t561_path0
   + 0.00542624481 t561_path1 - 6.34887e-05 t561_path2
   + 6.37661e-05 t562_path0 - 0.000311467 t562_path1
   - 0.00196128688 t562_path2 - 1.16428615e-04 t563_path0
   + 2.0515310500000001e-04 t564_path0 - 0.00482314359 t564_path1
   + 4.1603285399999999e-04 t564_path2 - 1.5195e-06 t565_path0
   - 0.00323019456 t566_path1 + 2.5759689700000001e-04 t566_path2
   - 0.00215753773 t566_path3 + 0.00398665061 t566_path4
   + 1.2567664100000001e-04 t566_path5 - 8.2781183300000005e-04 t567_path0
   + 0.00487138052 t567_path1 - 5.0017231800000004e-04 t567_path2
   - 2.1788768900000001e-04 t567_path3 - 0.00289756502 t567_path4
   + 3.2167922499999998e-04 t567_path5 - 1.19055323e-04 t568_path0
   - 0.00121519773 t568_path1 + 0.00260676257 t568_path2
   - 0.00157682376 t568_path3 + 4.23164e-05 t569_path0
   - 5.20453e-05 t570_path0 - 3.62942e-05 t571_path0
   - 8.7844137999999999e-04 t571_path1 + 0.00307952263 t571_path2
   - 0.00254928297 t571_path3 - 0.00253927941 t572_path1
   - 0.00199716701 t572_path2 - 6.67014e-05 t573_path0
   - 8.45809e-05 t574_path0 + 9.4545417200000005e-04 t574_path1
   - 0.00365613378 t574_path2 + 0.0031480852 t574_path3
   + 2.2494e-05 t575_path0 + 0.00162698282 t575_path1
   - 0.00219258294 t576_path0 + 0.000598109 t576_path1
   - 0.00207897346 t576_path2 + 2.2867335099999999e-04 t576_path3
   - 9.8742823999999996e-04 t576_path4 + 5.7297e-05 t576_path5
   + 1.27677253e-04 t577_path0 - 2.28782e-05 t578_path0
   - 0.00125513389 t578_path1 + 0.00101989054 t578_path2
   + 2.0294227599999999e-04 t579_path0 - 0.000583454 t579_path1
   - 1.2005634200000001e-04 t580_path0 - 0.00179917179 t580_path1
   + 0.00325733516 t580_path2 - 1.11995112e-04 t580_path3
   + 0.00312657072 t580_path4 + 5.2764930300000003e-04 t580_path5
   - 7.77127e-05 t581_path0 + 0.00298648793 t581_path1
   + 0.00151435041 t582_path0 - 4.53056e-05 t582_path1
   - 0.00259841839 t582_path2 - 6.19326e-07 t583_path0
   + 0.00159761496 t584_path0 + 0.0015324289 t584_path1
   + 2.25533e-05 t584_path2 - 0.00191564544 t584_path3
   - 1.21621408e-04 t585_path0 + 0.00129398273 t585_path1
   - 0.00209812913 t585_path2 + 0.00464958185 t585_path3
   + 0.000406847 t585_path4 + 0.0040965667 t586_path0
   - 4.71489e-05 t586_path1 - 0.00130873569 t586_path2
   + 2.6557187100000003e-04 t586_path3 + 0.00291670836 t586_path4
   - 6.99891e-05 t587_path0 + 0.00214311946 t587_path1
   - 0.00101804419 t587_path2 + 0.00436223717 t587_path3
   - 0.00124903326 t587_path4 + 0.00140586018 t587_path5
   + 0.00163603551 t588_path0 - 2.47692e-05 t588_path1
   - 3.3532275100000002e-04 t588_path2 - 0.00239840336 t588_path3
   + 0.00474706152 t588_path4 + 9.64133826e-04 t588_path5
   - 4.78732e-05 t588_path6 + 0.00343720964 t588_path7
   - 1.53807589e-04 t589_path0 - 2.9445e-05 t590_path0
   - 0.0017337529 t590_path1 - 3.39261e-05 t591_path0
   - 1.51668748e-04 t592_path0 - 1.52265e-05 t593_path0
   - 3.16768e-05 t594_path0 + 9.7461277599999996e-04 t595_path0
   + 0.00218158518 t595_path1 - 3.8273370600000001e-04 t595_path2
   - 6.51235e-05 t595_path3 + 0.00193780789 t595_path4
   + 6.25188e-05 t596_path0 - 3.2200469300000002e-04 t597_path0
   - 0.00320540881 t597_path1 + 1.25142513e-04 t597_path2
   + 0.00342430663 t597_path3 + 0.00239073788 t597_path4
   + 0.00147878507 t598_path0 - 1.2817417199999999e-04 t598_path1
   - 2.93934e-05 t599_path0 - 4.32474e-05 t600_path0
   + 0.00256722537 t600_path1 + 5.73431e-05 t600_path2
   - 0.000302245 t600_path3 - 1.35182854e-04 t601_path0
   - 0.000204421 t601_path1 + 0.00396685628 t601_path2
   + 8.3671067799999995e-04 t601_path3 + 1.68624e-05 t602_path0
   - 2.4055213699999999e-04 t603_path0 + 3.7220050599999998e-04 t603_path1
   + 8.9496e-06 t604_path0 + 9.57886e-06 t605_path0
   + 0.0010060796 t605_path1 - 0.00140665355 t605_path2
   + 0.00350159896 t605_path3 + 5.91163e-05 t606_path0
   - 2.13607e-05 t607_path0 - 0.00263399235 t607_path1
   - 1.9839710100000001e-04 t608_path0 + 0.00185567967 t608_path1
   - 1.41431e-05 t609_path0 - 1.66412e-05 t610_path0
   - 5.83672e-05 t611_path0 + 8.9526869100000001e-04 t611_path1
   + 1.20831e-05 t612_path0 + 0.000340316 t613_path0
   - 3.45610228e-04 t613_path1 + 2.24826e-05 t614_path0
   + 7.39887e-05 t615_path0 + 3.0573425500000001e-04 t616_path0
   - 8.70854e-05 t616_path1 - 0.00224715378 t616_path2
   - 0.000422832 t616_path3 + 0.00127459015 t617_path0
   - 0.00109007605 t617_path1 + 0.000116252 t617_path2
   + 5.42735e-08 t618_path0 + 0.00251163985 t618_path1
   - 1.27026e-05 t619_path0 - 0.0015222755 t619_path1
   + 9.6685037700000001e-04 t619_path2 + 0.00605672598 t619_path3
   + 2.4036846299999999e-04 t619_path4 + 7.3315e-05 t620_path0
   - 0.00213920418 t620_path1 + 1.3073127800000001e-04 t621_path0
   - 0.0029416082 t621_path2 - 1.7472318700000001e-04 t621_path3
   - 2.5530817200000001e-04 t622_path0 - 0.00156886294 t622_path1
   + 3.9943083500000001e-04 t622_path2 - 0.00186442037 t622_path3
   - 1.6841778400000001e-04 t623_path0 + 4.2586680499999999e-04 t623_path1
   + 0.00475928048 t623_path2 + 4.0905145599999998e-04 t623_path3
   + 0.00368787092 t623_path4 - 0.00186316215 t623_path5
   - 4.14198e-05 t624_path0 + 0.0020051985 t624_path1
   - 2.8728842200000001e-04 t625_path0 + 0.00315824826 t625_path1
   - 4.27315e-05 t625_path2 + 0.0035433867 t626_path1
   - 7.4106449000000004e-04 t626_path2 - 5.30406e-05 t626_path3
   - 5.01426e-05 t627_path0 - 0.00242259121 t627_path1
   - 4.87566e-05 t628_path0 - 5.10348e-05 t629_path0
   + 4.94083e-05 t630_path0 + 3.81261e-05 t631_path0
   + 0.00130104809 t632_path0 - 1.17781587e-04 t632_path1
   + 0.00565262558 t632_path3 - 8.86921e-06 t633_path0
   + 3.55426e-05 t634_path0 + 4.96217e-05 t635_path0
   + 5.66404e-06 t636_path0 + 4.27218e-05 t637_path0
   + 1.7494321199999999e-04 t638_path0 - 1.13147558e-04 t639_path0
   - 0.00185629714 t640_path0 + 1.33203e-05 t640_path1
   + 0.00613524625 t640_path2 - 3.83288e-05 t640_path3
   + 1.9781e-05 t641_path0 - 1.02989186e-04 t643_path0
   - 3.51006e-05 t644_path0 - 1.15778515e-04 t645_path0
   + 1.7900850800000001e-04 t646_path0 - 8.22664e-05 t646_path1
   - 0.00118274183 t646_path2 + 2.8086660299999998e-04 t647_path0
   - 3.39212333e-04 t647_path1 - 8.9756818499999998e-04 t647_path2
   + 1.95097e-05 t648_path0 - 3.03837e-05 t649_path0
   + 0.00182671263 t649_path1 - 2.0085256300000001e-04 t650_path0
   - 3.32028e-05 t651_path0 - 0.00230509043 t652_path1
   + 0.00236267853 t652_path2 + 1.68224e-05 t653_path0
   - 0.00166174909 t653_path1 + 3.70911e-06 t654_path0
   - 0.000121166 t655_path0 + 0.00397962844 t655_path1
   - 0.00103974401 t655_path2 + 3.03282e-05 t656_path0
   + 0.00268163951 t656_path1 - 0.00121936668 t656_path2
   + 7.6161e-05 t657_path0 - 0.00334497401 t657_path1
   + 3.09134e-05 t658_path0 + 0.00144463463 t659_path0
   - 3.6553133400000001e-04 t659_path1 - 0.00132835796 t659_path2
   - 0.00215905276 t659_path3 - 5.06413e-05 t659_path4
   + 3.5245e-05 t660_path0 + 7.8711414199999997e-04 t660_path1
   - 6.4149161300000005e-04 t660_path2 - 0.00286604767 t660_path3
   + 0.00172058155 t661_path0 - 0.00109011866 t661_path1
   - 7.8215e-05 t661_path2 + 0.00460131606 t661_path3
   - 9.5819658599999996e-04 t661_path4 + 2.33046e-06 t662_path0
   + 8.6160842299999997e-04 t663_path0 + 0.000143396 t663_path1
   - 0.00151998352 t663_path2 + 0.00543501507 t663_path3
   + 6.3389132300000004e-04 t663_path4 + 1.2378668200000001e-04 t663_path5
   - 0.00186408113 t663_path6 + 7.9294381399999995e-04 t664_path0
   - 0.00136463833 t664_path1 - 9.90472e-05 t664_path2
   + 3.67361819e-04 t665_path0 - 0.0016232183 t665_path1
   - 0.00033041 t665_path2 + 1.3996817899999999e-04 t666_path0
   + 4.0358543600000003e-04 t666_path1 - 0.00319703529 t666_path2
   - 4.4140065399999998e-04 t667_path0 + 0.00113000919 t667_path1
   + 0.00626973296 t667_path2 - 1.17209478e-04 t667_path3
   - 2.9240435100000002e-04 t668_path0 + 0.00383215095 t668_path1
   + 2.2067647699999999e-04 t668_path2 + 0.00543374242 t668_path3
   - 4.45153e-06 t669_path0 - 1.32121e-05 t670_path0
   + 2.78213e-05 t671_path0 - 0.00170903176 t671_path1
   + 6.89707e-05 t672_path0 + 0.00211846502 t673_path0
   - 9.3480525499999999e-04 t673_path1 - 0.00121623895 t673_path2
   + 8.4174156700000003e-04 t673_path3 - 0.000114582 t673_path4
   - 9.41094e-05 t674_path0 - 0.00270194747 t674_path1
   - 5.71125e-05 t675_path0 - 0.00303465081 t676_path0
   + 4.29067e-05 t676_path1 + 2.88531e-05 t677_path0
   - 3.25697e-05 t678_path0 + 0.000810209 t678_path1
   - 0.00225501112 t678_path2 - 0.00321772229 t678_path3
   - 0.000493416 t678_path4 - 4.57929e-05 t679_path0
   - 2.41172136e-04 t680_path0 - 0.00166071032 t680_path1
   + 2.9718561600000003e-04 t680_path2 - 0.00288230018 t680_path3
   - 0.00235753623 t680_path4 + 1.5446273e-04 t681_path0
   + 0.00329994829 t681_path1 - 1.39994e-05 t682_path0
   + 7.55557e-05 t683_path0 - 0.00354629103 t683_path1
   - 5.85615e-05 t683_path2 - 0.0017985818 t683_path3
   + 0.00401771395 t684_path0 - 0.000187505 t684_path1
   - 5.72283e-05 t684_path2 + 4.6655e-05 t685_path0
   - 3.94479e-05 t686_path0 + 0.00341447699 t687_path0
   + 1.02338498e-04 t687_path1 - 7.4143771800000001e-04 t687_path2
   + 1.59545e-05 t688_path0 + 7.5935e-05 t689_path0
   + 8.90341e-05 t690_path0 - 5.06326e-05 t691_path0
   + 1.12805414e-04 t692_path0 + 0.00327045657 t692_path1
   + 0.00245969556 t693_path0 - 1.3797616700000001e-04 t693_path1
   - 3.00627e-05 t694_path0 + 5.64036e-05 t695_path0
   - 1.3639789499999999e-04 t696_path0 - 3.1400666900000001e-04 t697_path0
   + 3.1261085000000002e-04 t697_path1 - 7.98502e-05 t698_path0
   + 0.00377371325 t698_path1 + 1.03841e-05 t699_path0
   + 1.91525e-05 t700_path0 + 5.08115e-05 t701_path0
   + 0.00135571905 t702_path0 - 5.9090985499999998e-04 t702_path1
   - 6.78367e-05 t702_path2 - 0.00289679901 t702_path3
   - 4.71973e-05 t703_path0 - 3.76481e-05 t704_path0
   - 1.35943e-05 t705_path0 + 0.00204694876 t705_path1
   - 0.00423646113 t705_path2 - 0.00364252692 t705_path3
   - 4.4542763499999999e-04 t705_path4 - 1.63703662e-04 t706_path0
   + 3.13693308e-04 t706_path1 + 0.00333039602 t706_path2
   + 2.41983e-05 t707_path0 - 1.68705e-05 t708_path0
   - 6.73504e-05 t709_path0 - 2.05909e-05 t709_path1
   + 0.00176071504 t709_path2 + 1.54333262e-04 t710_path0
   - 0.00332748494 t710_path1 + 4.0204724100000001e-04 t710_path2
   - 2.7995664300000002e-04 t710_path3 - 6.88676e-05 t711_path0
   - 0.00205248059 t712_path0 - 3.0062251600000001e-04 t712_path1
   + 0.00394420652 t712_path2 + 2.64586211e-04 t712_path3
   + 8.69266e-05 t713_path0 + 0.00198078365 t713_path1
   - 1.91327e-05 t714_path0 + 0.0019353847 t715_path1
   + 3.2512139299999999e-04 t715_path2 - 0.00133221119 t715_path3
   + 6.5844e-06 t716_path0 + 1.747153e-04 t717_path0
   + 1.63254e-05 t718_path0 - 1.31894e-05 t719_path0
   + 0.00303673791 t720_path0 + 3.98427e-05 t720_path1
   - 4.48759e-05 t721_path0 - 4.79159e-05 t722_path0
   + 1.5885065699999999e-04 t723_path0 - 4.81961e-05 t724_path0
   - 3.17488e-05 t725_path0 + 0.00244568195 t726_path0
   - 1.16781e-05 t726_path1 - 0.00163566624 t726_path2
   + 1.3390828100000001e-04 t727_path0 - 0.00513767684 t727_path1
   - 0.00164370693 t728_path1 - 1.93313521e-04 t729_path0
   + 8.3565863300000005e-04 t730_path0 - 8.2540471300000004e-04 t730_path1
   + 0.00161223742 t730_path2 + 1.4494759700000001e-04 t730_path3
   + 0.00345601351 t730_path4 - 5.82928e-05 t731_path0
   - 9.4503e-05 t732_path0 + 4.01409e-06 t733_path0
   + 7.71063e-05 t734_path0 + 7.2224e-05 t735_path0
   + 2.08127443e-04 t736_path0 - 6.67851e-05 t737_path0
   + 3.00794e-05 t738_path0 + 3.93748e-05 t739_path0
   - 0.00117612816 t739_path1 + 9.18681e-06 t740_path0
   - 5.65664e-05 t741_path0 + 0.00620048307 t741_path1
   - 4.4327616299999999e-04 t741_path2 - 0.00146190461 t742_path0
   + 0.00260374602 t742_path1 + 8.7459030300000005e-04 t742_path2
   - 3.09016e-05 t742_path3 + 0.00646517752 t742_path4
   + 0.000592359 t742_path6 - 8.91243e-06 t743_path0
   - 2.74526e-05 t744_path0 - 5.23045e-05 t745_path0
   + 1.00894e-05 t746_path0 - 2.31419e-05 t747_path0
   - 9.32162e-06 t748_path0 + 6.4837036199999996e-04 t749_path0
   - 0.00207467703 t749_path1 + 0.00306605967 t749_path2
   - 1.3687382999999999e-04 t749_path3 + 5.97513e-05 t750_path0
   - 3.40493e-05 t751_path0 - 1.4192868499999999e-04 t752_path0
   - 1.38671e-06 t753_path0 - 5.40337e-06 t754_path0
   + 1.08947e-05 t755_path0 + 1.72353e-05 t756_path0
   - 1.86329868e-04 t757_path0 + 0.00516315177 t757_path1
   + 9.00723157e-04 t757_path2 + 1.60954107e-04 t757_path3
   - 6.73389e-05 t758_path0 + 0.00015149 t759_path0
   - 0.0017015069 t759_path1 + 1.2294620700000001e-04 t760_path0
   - 6.67682e-05 t761_path0 - 0.00191103143 t761_path1
   - 5.7752965899999999e-04 t761_path2 + 0.00157087587 t761_path3
   + 8.58141e-05 t762_path0 - 3.07945e-05 t763_path0
   - 0.00197148137 t763_path1 - 1.2163622200000001e-04 t763_path2
   + 3.93870636e-04 t763_path3 + 0.00232828921 t763_path4
   - 0.00305568567 t764_path0 - 3.7642868099999998e-04 t764_path1
   + 0.00211131899 t764_path2 + 0.000226889 t764_path3
   - 4.9542e-05 t765_path0 + 1.9492457799999999e-04 t766_path0
   - 0.00173935061 t766_path1 - 2.03227e-05 t767_path0
   + 5.09111e-05 t768_path0 - 0.00189218577 t768_path1
   + 0.00236037141 t768_path2 - 3.1772e-05 t769_path0
   - 1.3172623600000001e-04 t770_path0 - 6.63884e-06 t771_path0
   + 1.79716852e-04 t772_path0 - 0.00314741069 t772_path1
   - 2.97736202e-04 t772_path2 + 1.5233812199999999e-04 t773_path0
   - 5.23566e-05 t774_path0 + 2.12635583e-04 t775_path0
   - 0.00143921957 t775_path1 + 2.67903e-05 t776_path0
   - 6.34962e-05 t777_path0 + 0.00239414 t777_path1
   - 2.05344e-05 t778_path0 - 0.00526554463 t778_path1
   + 1.59439e-05 t778_path2 + 0.00317724166 t778_path3
   - 6.49324e-05 t779_path0 - 3.595e-05 t780_path0
   + 0.00363019365 t781_path0 + 6.35494e-05 t781_path1
   - 0.00111368322 t781_path2 - 5.81681e-05 t782_path0
   - 1.92205e-05 t783_path0 - 0.00146728428 t784_path0
   + 6.3343520700000005e-04 t784_path1 + 1.1299551999999999e-04 t784_path2
   + 0.00331801292 t784_path3 - 0.00283829262 t784_path4
   + 0.00183699862 t785_path0 - 9.63388e-05 t785_path1
   - 8.14462e-05 t786_path0 + 5.78842e-05 t787_path0
   - 0.00339002116 t787_path1 + 6.38706e-06 t788_path0
   + 1.2692887699999999e-04 t789_path0 + 5.22696e-06 t790_path0
   + 1.4929116899999999e-04 t791_path0 - 1.48858e-05 t792_path0
   + 2.51086e-05 t793_path0 - 0.00331524736 t793_path1
   + 0.00442092028 t793_path2 + 3.74112e-05 t793_path3
   - 1.12779264e-04 t794_path0 + 0.00206198427 t795_path1
   - 3.7376178099999998e-04 t795_path2 + 1.80031e-05 t795_path3
   - 0.00176201877 t795_path4 - 6.21064e-05 t796_path0
   + 2.81242e-05 t797_path0 - 6.74817e-05 t798_path0
   + 1.2516773100000001e-04 t799_path0 - 1.62891447e-04 t800_path0
   - 3.00696e-05 t801_path0 + 7.3264e-05 t802_path0
   - 4.45435e-05 t803_path0 + 6.04631e-05 t804_path0
   - 0.00174679 t805_path0 + 1.79584e-05 t805_path1
   + 8.3087221699999997e-04 t806_path0 - 3.84749e-05 t806_path1
   + 1.04881474e-04 t807_path0 - 3.51333e-05 t808_path0
   + 9.56046e-06 t809_path0 - 5.1986e-05 t810_path0
   - 1.21441e-05 t811_path0 + 9.15348e-05 t812_path0
   + 0.000117883 t813_path0 - 2.1007812799999999e-04 t814_path0
   + 5.8581749900000004e-04 t814_path1 + 0.00186009903 t814_path2
   + 0.00355380168 t814_path3 - 4.4575668299999999e-04 t814_path4
   - 3.98216e-05 t815_path0 - 7.58876e-05 t816_path0
   - 7.58534e-05 t817_path0 + 0.00592026114 t817_path1
   + 2.4496673699999998e-04 t817_path2 + 2.5795e-05 t818_path0
   + 4.85721e-05 t819_path0 - 9.35011e-05 t820_path0
   + 5.67381e-06 t821_path0 + 5.19976e-05 t822_path0
   - 5.34172e-05 t823_path0 - 2.23862e-05 t824_path0
   - 3.01992e-05 t825_path0 - 0.00172661943 t825_path1
   + 8.59429e-05 t826_path0 - 3.9412928200000002e-04 t827_path0
   + 0.0015367259 t827_path1 - 0.0019078427 t827_path2
   + 1.7429774700000001e-04 t827_path3 + 4.18846e-05 t828_path0
   + 0.00125556637 t829_path0 - 9.90314e-05 t829_path1
   - 1.8633784200000001e-04 t829_path2 + 0.00217220676 t829_path3
   - 0.00154137623 t829_path4 + 3.21379e-05 t830_path0
   + 5.04298e-05 t831_path0 - 0.00215259194 t831_path1
   - 0.00369863166 t832_path0 - 4.28939e-05 t832_path1
   + 6.52045e-06 t833_path0 - 2.03854666e-04 t834_path0
   - 1.06950567e-04 t835_path0 - 0.0018394246 t835_path1
   + 0.00156592927 t835_path2 + 1.3048e-05 t836_path0
   + 2.09775e-05 t837_path0 + 3.89339e-05 t838_path0
   - 1.01981714e-04 t839_path0 + 0.00322737521 t839_path1
   - 6.40916e-08 t840_path0 - 6.03732e-05 t841_path0
   + 0.00112336886 t841_path1 - 5.491e-05 t842_path0
   + 0.00127638178 t842_path1 - 6.37554e-06 t843_path0
   + 0.00101418013 t844_path0 - 7.68308e-05 t844_path1
   - 2.02956173e-04 t845_path0 - 6.1968714000000004e-04 t845_path1
   + 9.3402277000000002e-04 t845_path2 - 1.67296e-05 t846_path0
   - 2.71912e-05 t847_path0 + 0.00175209437 t848_path0
   - 8.8514969700000004e-04 t848_path1 - 5.13455e-05 t848_path2
   + 9.53182e-05 t849_path0 - 0.00115381903 t849_path1
   - 8.83821223e-04 t849_path2 - 7.96345e-05 t850_path0
   - 7.88005e-06 t851_path0 + 2.35185e-05 t852_path0
   - 9.56815e-06 t853_path0 + 1.01649712e-04 t854_path0
   - 0.00261323177 t854_path1 - 7.56838e-05 t855_path0
   - 7.89397e-08 t856_path0 - 0.00186690688 t856_path1
   + 7.54399e-05 t857_path0 - 3.47688e-05 t858_path0
   - 0.0022359693 t858_path1 + 0.000603889 t858_path2
   + 0.00388221676 t858_path3 - 4.57757e-05 t859_path0
   + 0.00481479382 t860_path0 + 1.02130725e-04 t860_path2
   + 6.06652e-05 t861_path0 - 1.0676956299999999e-04 t862_path0
   + 1.3420476200000001e-04 t863_path0 + 4.7329e-05 t864_path0
   - 3.65058e-05 t865_path0 + 0.00540651614 t865_path2
   + 2.88547e-05 t866_path0 + 4.98076e-05 t867_path0
   + 4.69418e-05 t868_path0 - 3.77602e-05 t869_path0
   + 1.22322e-05 t870_path0 - 2.96809e-05 t871_path0
   + 1.49157e-05 t872_path0 + 7.27154e-05 t873_path0
   + 5.29768e-05 t874_path0 - 0.0014122345 t875_path0
   + 1.28257685e-04 t875_path1 + 4.5178542499999999e-04 t876_path0
   - 0.00186321908 t876_path1 + 1.74142391e-04 t876_path2
   - 0.00159601681 t876_path3 - 3.51588e-06 t877_path0
   - 0.00146434631 t877_path1 - 8.37409e-05 t878_path0
   + 3.44417e-05 t879_path0 + 2.45524e-05 t880_path0
   - 8.6292089000000001e-04 t880_path1 + 0.00399752706 t880_path2
   - 6.4143119400000002e-04 t880_path3 + 0.003140938 t881_path0
   - 0.000293376 t881_path2 - 4.1023155799999999e-04 t881_path3
   + 2.02451381e-04 t881_path4 + 1.4837e-05 t882_path0
   - 7.7574211200000004e-04 t883_path0 + 3.23797e-05 t883_path1
   + 2.24807e-06 t884_path0 + 3.82079e-05 t885_path0
   - 0.00118095614 t886_path0 + 1.69205e-05 t886_path1
   - 0.00339460955 t886_path2 + 0.00153407396 t886_path3
   - 8.8644446799999999e-04 t886_path4 - 0.000731332 t886_path5
   - 0.000104384 t887_path0 + 2.4452211800000002e-04 t888_path0
   - 4.4280648600000003e-04 t888_path1 + 0.00047766 t889_path0
   + 0.00487284176 t889_path1 - 2.5142284000000001e-04 t889_path2
   + 3.6270721499999999e-04 t889_path3 - 0.00309592066 t889_path4
   - 2.05314e-05 t890_path0 + 1.84283e-05 t891_path0
   + 4.44824e-05 t892_path0 + 3.05993e-05 t893_path0
   + 1.8142779300000001e-04 t894_path0 - 0.00136716268 t894_path1
   + 1.4151226800000001e-04 t895_path0 + 9.03398e-05 t896_path0
   - 9.7510457299999995e-04 t896_path1 + 9.31598e-06 t897_path0
   - 7.41732e-05 t898_path0 + 2.9616e-05 t899_path0 + 2.8436e-05 t900_path0
   + 2.14811e-05 t901_path0 - 2.54415e-05 t902_path0
   - 0.00147752825 t903_path0 + 0.00257980474 t903_path2
   + 0.00294329342 t904_path0 - 9.32581e-06 t904_path1
   - 4.59413e-05 t905_path0 - 0.000347855 t906_path0
   + 5.3914968200000004e-04 t906_path1 + 9.1764662600000004e-04 t906_path2
   + 3.6586e-05 t907_path0 - 1.6305962299999999e-04 t908_path0
   + 0.00130810367 t909_path0 - 1.22262e-06 t909_path1
   + 2.62518e-05 t910_path0 - 4.45145e-05 t911_path0 - 7.128e-05 t912_path0
   + 9.69016e-05 t913_path0 = 0
 YLOG_with_offset: - y_log_raw + y_log = 12.437748
 COST_eq_linexpr: - 17352 x_roof_matl_is_ClyTile
   - 8011.95 x_roof_matl_is_Membran - 11739 x_roof_matl_is_Metal
   - 7600 x_roof_matl_is_Roll - 8550 x_roof_matl_is_Tar&Grv
   - 22500 x_roof_matl_is_WdShake - 19500 x_roof_matl_is_WdShngl
   - 1712.5 x_elect_is_FuseA - 887.5 x_elect_is_FuseF
   - 62.5 x_elect_is_FuseP - 287.5 x_elect_is_Mix
   - 24038 x_garage_finish_is_Fin - 20769 x_garage_finish_is_RFn
   - 28000 x_AddFull - 11200 x_AddHalf - 16800 x_AddKitch - 25480 x_AddBed
   - 51659 x_garage_qual_is_Ex - 51659 x_garage_cond_is_Ex
   - 37849 x_garage_qual_is_Gd - 37849 x_garage_cond_is_Gd
   - 14113 x_garage_qual_is_Fa - 14113 x_garage_cond_is_Fa
   - 4188 x_garage_qual_is_Po - 4188 x_garage_cond_is_Po
   - 3354 x_paved_drive_is_P - 1800 x_paved_drive_is_N
   - 9540 x_fence_is_GdPrv - 7940 x_fence_is_MnPrv - 3232 x_fence_is_GdWo
   - 2400 x_fence_is_MnWw - 19000 x_ex1_is_AsbShng - 22500 x_ex1_is_AsphShn
   - 26000 x_ex1_is_BrkComm - 22000 x_ex1_is_BrkFace
   - 10300 x_ex1_is_CBlock - 14674 x_ex1_is_CemntBd
   - 21300 x_ex1_is_HdBoard - 16500 x_ex1_is_ImStucc
   - 11196 x_ex1_is_MetalSd - 21765.3125 x_ex1_is_Other
   - 3461.81 x_ex1_is_Plywood - 17625 x_ex1_is_PreCast
   - 106250 x_ex1_is_Stone - 5629 x_ex1_is_Stucco - 17410 x_ex1_is_VinylSd
   - 12500 x_ex1_is_Wd_Sdng - 21900 x_ex1_is_WdShngl
   - 19000 x_ex2_is_AsbShng - 22500 x_ex2_is_AsphShn
   - 26000 x_ex2_is_BrkComm - 22000 x_ex2_is_BrkFace
   - 10300 x_ex2_is_CBlock - 14674 x_ex2_is_CemntBd
   - 21300 x_ex2_is_HdBoard - 16500 x_ex2_is_ImStucc
   - 11196 x_ex2_is_MetalSd - 21765.3125 x_ex2_is_Other
   - 3461.81 x_ex2_is_Plywood - 17625 x_ex2_is_PreCast
   - 106250 x_ex2_is_Stone - 5629 x_ex2_is_Stucco - 17410 x_ex2_is_VinylSd
   - 12500 x_ex2_is_Wd_Sdng - 21900 x_ex2_is_WdShngl - 75000 x_b1_is_GLQ
   - 32000 x_b1_is_BLQ - 23500 x_b1_is_Rec - 15000 x_b1_is_LwQ
   - 11250 x_b1_is_Unf - 75000 x_b2_is_GLQ - 53500 x_b2_is_ALQ
   - 32000 x_b2_is_BLQ - 23500 x_b2_is_Rec - 15000 x_b2_is_LwQ
   - 2875.08 x_z10_GarageArea - 7442.447999999999 x_z20_GarageArea
   - 13701.168 x_z30_GarageArea - 3621.495 x_z10_WoodDeckSF
   - 9374.621999999999 x_z20_WoodDeckSF - 17258.202 x_z30_WoodDeckSF
   - 331.74 x_z10_OpenPorchSF - 858.744 x_z20_OpenPorchSF
   - 1580.904 x_z30_OpenPorchSF - 180000 kit_is_Ex
   - 3949.31 exterqual_is_Gd - 87366.25 exterqual_is_Ex
   - 3949.31 extercond_is_Gd - 87366.25 extercond_is_Ex
   + 13.79 mv_area_mvt_BrkCmn - 7.5 mv_area_mvt_CBlock
   - 12.5 mv_area_mvt_Stone - 10850 roof_change_mat - 10304 x_Add1stFlr
   - 1773 heat_is_Floor - 8500 heat_is_GasW - 6300 heat_is_Grav
   - 5000 heat_is_OthW - 3700 heat_is_Wall - 3750 heat_qc_is_Po
   - 5125 heat_qc_is_Fa - 8250 heat_qc_is_Gd - 10000 heat_qc_is_Ex
   - 1500 fireplace_is_Po - 2000 fireplace_is_Fa - 2500 fireplace_is_TA
   - 3525 fireplace_is_Gd - 4550 fireplace_is_Ex - 12992 bsmt_finish_all
   - 10750 bsmtcond_is_Gd - 21500 bsmtcond_is_Ex + cost_model = 0
 BUDGET: cost_model <= 500000
 NOCHG_cost_link: cost_model - 500000 z_change <= 0
 NOCHG_cost_lb: cost_model - z_change >= 0
 NOCHG_pos_Lot_Frontage: const_Lot_Frontage <= 81
 NOCHG_neg_Lot_Frontage: - const_Lot_Frontage <= -81
 NOCHG_pos_Lot_Area: const_Lot_Area <= 14267
 NOCHG_neg_Lot_Area: - const_Lot_Area <= -14267
 NOCHG_pos_Utilities: x_Utilities <= 3
 NOCHG_neg_Utilities: - x_Utilities - 3 z_change <= -3
 NOCHG_pos_Overall_Qual: Overall_Qual_calc - 4 z_change <= 6
 NOCHG_neg_Overall_Qual: - Overall_Qual_calc - 5 z_change <= -6
 NOCHG_pos_Overall_Cond: const_Overall_Cond <= 6
 NOCHG_neg_Overall_Cond: - const_Overall_Cond <= -6
 NOCHG_pos_Year_Built: const_Year_Built <= 1958
 NOCHG_neg_Year_Built: - const_Year_Built <= -1958
 NOCHG_pos_Year_Remod/Add: const_Year_Remod_Add <= 1958
 NOCHG_neg_Year_Remod/Add: - const_Year_Remod_Add <= -1958
 NOCHG_pos_Mas_Vnr_Area: x_Mas_Vnr_Area <= 108
 NOCHG_neg_Mas_Vnr_Area: - x_Mas_Vnr_Area <= -108
 NOCHG_pos_Exter_Qual: x_Exter_Qual - 2 z_change <= 2
 NOCHG_neg_Exter_Qual: - x_Exter_Qual - 2 z_change <= -2
 NOCHG_pos_Exter_Cond: x_Exter_Cond - 2 z_change <= 2
 NOCHG_neg_Exter_Cond: - x_Exter_Cond - 2 z_change <= -2
 NOCHG_pos_Bsmt_Qual: const_Bsmt_Qual <= 2
 NOCHG_neg_Bsmt_Qual: - const_Bsmt_Qual <= -2
 NOCHG_pos_Bsmt_Cond: x_Bsmt_Cond - 2 z_change <= 2
 NOCHG_neg_Bsmt_Cond: - x_Bsmt_Cond - 3 z_change <= -2
 NOCHG_pos_BsmtFin_SF_1: x_BsmtFin_SF_1 - 999077 z_change <= 923
 NOCHG_neg_BsmtFin_SF_1: - x_BsmtFin_SF_1 - 923 z_change <= -923
 NOCHG_pos_BsmtFin_SF_2: x_BsmtFin_SF_2 - 100000 z_change <= 0
 NOCHG_neg_BsmtFin_SF_2: - x_BsmtFin_SF_2 <= 0
 NOCHG_pos_Bsmt_Unf_SF: x_Bsmt_Unf_SF - 99594 z_change <= 406
 NOCHG_neg_Bsmt_Unf_SF: - x_Bsmt_Unf_SF - 406 z_change <= -406
 NOCHG_pos_Total_Bsmt_SF: const_Total_Bsmt_SF <= 1329
 NOCHG_neg_Total_Bsmt_SF: - const_Total_Bsmt_SF <= -1329
 NOCHG_pos_Heating_QC: x_Heating_QC - 2 z_change <= 2
 NOCHG_neg_Heating_QC: - x_Heating_QC - 2 z_change <= -2
 NOCHG_pos_1st_Flr_SF: x_1st_Flr_SF - 3671 z_change <= 1329
 NOCHG_neg_1st_Flr_SF: - x_1st_Flr_SF - 1329 z_change <= -1329
 NOCHG_pos_2nd_Flr_SF: const_2nd_Flr_SF <= 0
 NOCHG_neg_2nd_Flr_SF: - const_2nd_Flr_SF <= 0
 NOCHG_pos_Low_Qual_Fin_SF: const_Low_Qual_Fin_SF <= 0
 NOCHG_neg_Low_Qual_Fin_SF: - const_Low_Qual_Fin_SF <= 0
 NOCHG_pos_Gr_Liv_Area: x_Gr_Liv_Area - 8671 z_change <= 1329
 NOCHG_neg_Gr_Liv_Area: - x_Gr_Liv_Area - 1329 z_change <= -1329
 NOCHG_pos_Bsmt_Full_Bath: const_Bsmt_Full_Bath <= 0
 NOCHG_neg_Bsmt_Full_Bath: - const_Bsmt_Full_Bath <= 0
 NOCHG_pos_Bsmt_Half_Bath: const_Bsmt_Half_Bath <= 0
 NOCHG_neg_Bsmt_Half_Bath: - const_Bsmt_Half_Bath <= 0
 NOCHG_pos_Full_Bath: x_Full_Bath - 3 z_change <= 1
 NOCHG_neg_Full_Bath: - x_Full_Bath - z_change <= -1
 NOCHG_pos_Half_Bath: x_Half_Bath - 4 z_change <= 1
 NOCHG_neg_Half_Bath: - x_Half_Bath - z_change <= -1
 NOCHG_pos_Bedroom_AbvGr: x_Bedroom_AbvGr - 3 z_change <= 3
 NOCHG_neg_Bedroom_AbvGr: - x_Bedroom_AbvGr - 3 z_change <= -3
 NOCHG_pos_Kitchen_AbvGr: x_Kitchen_AbvGr - 2 z_change <= 1
 NOCHG_neg_Kitchen_AbvGr: - x_Kitchen_AbvGr - z_change <= -1
 NOCHG_pos_Kitchen_Qual: x_Kitchen_Qual - z_change <= 3
 NOCHG_neg_Kitchen_Qual: - x_Kitchen_Qual - 3 z_change <= -3
 NOCHG_pos_TotRms_AbvGrd: x_TotRms_AbvGrd - 9 z_change <= 6
 NOCHG_neg_TotRms_AbvGrd: - x_TotRms_AbvGrd - 6 z_change <= -6
 NOCHG_pos_Fireplaces: const_Fireplaces <= 0
 NOCHG_neg_Fireplaces: - const_Fireplaces <= 0
 NOCHG_pos_Fireplace_Qu: x_Fireplace_Qu - 5 z_change <= -1
 NOCHG_neg_Fireplace_Qu: - x_Fireplace_Qu <= 1
 NOCHG_pos_Garage_Yr_Blt: const_Garage_Yr_Blt <= 1958
 NOCHG_neg_Garage_Yr_Blt: - const_Garage_Yr_Blt <= -1958
 NOCHG_pos_Garage_Cars: const_Garage_Cars <= 1
 NOCHG_neg_Garage_Cars: - const_Garage_Cars <= -1
 NOCHG_pos_Garage_Area: x_Garage_Area - 99688 z_change <= 312
 NOCHG_neg_Garage_Area: - x_Garage_Area - 312 z_change <= -312
 NOCHG_pos_Garage_Qual: x_Garage_Qual - 2 z_change <= 2
 NOCHG_neg_Garage_Qual: - x_Garage_Qual - 3 z_change <= -2
 NOCHG_pos_Garage_Cond: x_Garage_Cond - 2 z_change <= 2
 NOCHG_neg_Garage_Cond: - x_Garage_Cond - 3 z_change <= -2
 NOCHG_pos_Wood_Deck_SF: x_Wood_Deck_SF - 99607 z_change <= 393
 NOCHG_neg_Wood_Deck_SF: - x_Wood_Deck_SF - 393 z_change <= -393
 NOCHG_pos_Open_Porch_SF: x_Open_Porch_SF - 99964 z_change <= 36
 NOCHG_neg_Open_Porch_SF: - x_Open_Porch_SF - 36 z_change <= -36
 NOCHG_pos_Enclosed_Porch: x_Enclosed_Porch - 100000 z_change <= 0
 NOCHG_neg_Enclosed_Porch: - x_Enclosed_Porch <= 0
 NOCHG_pos_3Ssn_Porch: x_3Ssn_Porch - 100000 z_change <= 0
 NOCHG_neg_3Ssn_Porch: - x_3Ssn_Porch <= 0
 NOCHG_pos_Screen_Porch: x_Screen_Porch - 100000 z_change <= 0
 NOCHG_neg_Screen_Porch: - x_Screen_Porch <= 0
 NOCHG_pos_Pool_Area: x_Pool_Area - 100000 z_change <= 0
 NOCHG_neg_Pool_Area: - x_Pool_Area <= 0
 NOCHG_pos_Pool_QC: x_Pool_QC - 5 z_change <= -1
 NOCHG_neg_Pool_QC: - x_Pool_QC <= 1
 NOCHG_pos_Misc_Val: const_Misc_Val <= 12500
 NOCHG_neg_Misc_Val: - const_Misc_Val <= -12500
 NOCHG_pos_Yr_Sold: const_Yr_Sold <= 2010
 NOCHG_neg_Yr_Sold: - const_Yr_Sold <= -2010
 NOCHG_pos_MS_SubClass_20: const_MS_SubClass_20 <= 1
 NOCHG_neg_MS_SubClass_20: - const_MS_SubClass_20 <= -1
 NOCHG_pos_MS_SubClass_30: const_MS_SubClass_30 <= 0
 NOCHG_neg_MS_SubClass_30: - const_MS_SubClass_30 <= 0
 NOCHG_pos_MS_SubClass_40: const_MS_SubClass_40 <= 0
 NOCHG_neg_MS_SubClass_40: - const_MS_SubClass_40 <= 0
 NOCHG_pos_MS_SubClass_45: const_MS_SubClass_45 <= 0
 NOCHG_neg_MS_SubClass_45: - const_MS_SubClass_45 <= 0
 NOCHG_pos_MS_SubClass_50: const_MS_SubClass_50 <= 0
 NOCHG_neg_MS_SubClass_50: - const_MS_SubClass_50 <= 0
 NOCHG_pos_MS_SubClass_60: const_MS_SubClass_60 <= 0
 NOCHG_neg_MS_SubClass_60: - const_MS_SubClass_60 <= 0
 NOCHG_pos_MS_SubClass_70: const_MS_SubClass_70 <= 0
 NOCHG_neg_MS_SubClass_70: - const_MS_SubClass_70 <= 0
 NOCHG_pos_MS_SubClass_75: const_MS_SubClass_75 <= 0
 NOCHG_neg_MS_SubClass_75: - const_MS_SubClass_75 <= 0
 NOCHG_pos_MS_SubClass_80: const_MS_SubClass_80 <= 0
 NOCHG_neg_MS_SubClass_80: - const_MS_SubClass_80 <= 0
 NOCHG_pos_MS_SubClass_85: const_MS_SubClass_85 <= 0
 NOCHG_neg_MS_SubClass_85: - const_MS_SubClass_85 <= 0
 NOCHG_pos_MS_SubClass_90: const_MS_SubClass_90 <= 0
 NOCHG_neg_MS_SubClass_90: - const_MS_SubClass_90 <= 0
 NOCHG_pos_MS_SubClass_120: const_MS_SubClass_120 <= 0
 NOCHG_neg_MS_SubClass_120: - const_MS_SubClass_120 <= 0
 NOCHG_pos_MS_SubClass_150: const_MS_SubClass_150 <= 0
 NOCHG_neg_MS_SubClass_150: - const_MS_SubClass_150 <= 0
 NOCHG_pos_MS_SubClass_160: const_MS_SubClass_160 <= 0
 NOCHG_neg_MS_SubClass_160: - const_MS_SubClass_160 <= 0
 NOCHG_pos_MS_SubClass_180: const_MS_SubClass_180 <= 0
 NOCHG_neg_MS_SubClass_180: - const_MS_SubClass_180 <= 0
 NOCHG_pos_MS_SubClass_190: const_MS_SubClass_190 <= 0
 NOCHG_neg_MS_SubClass_190: - const_MS_SubClass_190 <= 0
 NOCHG_pos_MS_Zoning_A_(agr): const_MS_Zoning_A_(agr) <= 0
 NOCHG_neg_MS_Zoning_A_(agr): - const_MS_Zoning_A_(agr) <= 0
 NOCHG_pos_MS_Zoning_C_(all): const_MS_Zoning_C_(all) <= 0
 NOCHG_neg_MS_Zoning_C_(all): - const_MS_Zoning_C_(all) <= 0
 NOCHG_pos_MS_Zoning_FV: const_MS_Zoning_FV <= 0
 NOCHG_neg_MS_Zoning_FV: - const_MS_Zoning_FV <= 0
 NOCHG_pos_MS_Zoning_I_(all): const_MS_Zoning_I_(all) <= 0
 NOCHG_neg_MS_Zoning_I_(all): - const_MS_Zoning_I_(all) <= 0
 NOCHG_pos_MS_Zoning_RH: const_MS_Zoning_RH <= 0
 NOCHG_neg_MS_Zoning_RH: - const_MS_Zoning_RH <= 0
 NOCHG_pos_MS_Zoning_RL: const_MS_Zoning_RL <= 1
 NOCHG_neg_MS_Zoning_RL: - const_MS_Zoning_RL <= -1
 NOCHG_pos_MS_Zoning_RM: const_MS_Zoning_RM <= 0
 NOCHG_neg_MS_Zoning_RM: - const_MS_Zoning_RM <= 0
 NOCHG_pos_Street_Grvl: const_Street_Grvl <= 0
 NOCHG_neg_Street_Grvl: - const_Street_Grvl <= 0
 NOCHG_pos_Street_Pave: const_Street_Pave <= 1
 NOCHG_neg_Street_Pave: - const_Street_Pave <= -1
 NOCHG_pos_Alley_Grvl: const_Alley_Grvl <= 0
 NOCHG_neg_Alley_Grvl: - const_Alley_Grvl <= 0
 NOCHG_pos_Alley_No_aplica: const_Alley_No_aplica <= 1
 NOCHG_neg_Alley_No_aplica: - const_Alley_No_aplica <= -1
 NOCHG_pos_Alley_Pave: const_Alley_Pave <= 0
 NOCHG_neg_Alley_Pave: - const_Alley_Pave <= 0
 NOCHG_pos_Lot_Shape_IR1: const_Lot_Shape_IR1 <= 1
 NOCHG_neg_Lot_Shape_IR1: - const_Lot_Shape_IR1 <= -1
 NOCHG_pos_Lot_Shape_IR2: const_Lot_Shape_IR2 <= 0
 NOCHG_neg_Lot_Shape_IR2: - const_Lot_Shape_IR2 <= 0
 NOCHG_pos_Lot_Shape_IR3: const_Lot_Shape_IR3 <= 0
 NOCHG_neg_Lot_Shape_IR3: - const_Lot_Shape_IR3 <= 0
 NOCHG_pos_Lot_Shape_Reg: const_Lot_Shape_Reg <= 0
 NOCHG_neg_Lot_Shape_Reg: - const_Lot_Shape_Reg <= 0
 NOCHG_pos_Land_Contour_Bnk: const_Land_Contour_Bnk <= 0
 NOCHG_neg_Land_Contour_Bnk: - const_Land_Contour_Bnk <= 0
 NOCHG_pos_Land_Contour_HLS: const_Land_Contour_HLS <= 0
 NOCHG_neg_Land_Contour_HLS: - const_Land_Contour_HLS <= 0
 NOCHG_pos_Land_Contour_Low: const_Land_Contour_Low <= 0
 NOCHG_neg_Land_Contour_Low: - const_Land_Contour_Low <= 0
 NOCHG_pos_Land_Contour_Lvl: const_Land_Contour_Lvl <= 1
 NOCHG_neg_Land_Contour_Lvl: - const_Land_Contour_Lvl <= -1
 NOCHG_pos_Lot_Config_Corner: const_Lot_Config_Corner <= 1
 NOCHG_neg_Lot_Config_Corner: - const_Lot_Config_Corner <= -1
 NOCHG_pos_Lot_Config_CulDSac: const_Lot_Config_CulDSac <= 0
 NOCHG_neg_Lot_Config_CulDSac: - const_Lot_Config_CulDSac <= 0
 NOCHG_pos_Lot_Config_FR2: const_Lot_Config_FR2 <= 0
 NOCHG_neg_Lot_Config_FR2: - const_Lot_Config_FR2 <= 0
 NOCHG_pos_Lot_Config_FR3: const_Lot_Config_FR3 <= 0
 NOCHG_neg_Lot_Config_FR3: - const_Lot_Config_FR3 <= 0
 NOCHG_pos_Lot_Config_Inside: const_Lot_Config_Inside <= 0
 NOCHG_neg_Lot_Config_Inside: - const_Lot_Config_Inside <= 0
 NOCHG_pos_Land_Slope_Gtl: const_Land_Slope_Gtl <= 1
 NOCHG_neg_Land_Slope_Gtl: - const_Land_Slope_Gtl <= -1
 NOCHG_pos_Land_Slope_Mod: const_Land_Slope_Mod <= 0
 NOCHG_neg_Land_Slope_Mod: - const_Land_Slope_Mod <= 0
 NOCHG_pos_Land_Slope_Sev: const_Land_Slope_Sev <= 0
 NOCHG_neg_Land_Slope_Sev: - const_Land_Slope_Sev <= 0
 NOCHG_pos_Neighborhood_Blmngtn: const_Neighborhood_Blmngtn <= 0
 NOCHG_neg_Neighborhood_Blmngtn: - const_Neighborhood_Blmngtn <= 0
 NOCHG_pos_Neighborhood_Blueste: const_Neighborhood_Blueste <= 0
 NOCHG_neg_Neighborhood_Blueste: - const_Neighborhood_Blueste <= 0
 NOCHG_pos_Neighborhood_BrDale: const_Neighborhood_BrDale <= 0
 NOCHG_neg_Neighborhood_BrDale: - const_Neighborhood_BrDale <= 0
 NOCHG_pos_Neighborhood_BrkSide: const_Neighborhood_BrkSide <= 0
 NOCHG_neg_Neighborhood_BrkSide: - const_Neighborhood_BrkSide <= 0
 NOCHG_pos_Neighborhood_ClearCr: const_Neighborhood_ClearCr <= 0
 NOCHG_neg_Neighborhood_ClearCr: - const_Neighborhood_ClearCr <= 0
 NOCHG_pos_Neighborhood_CollgCr: const_Neighborhood_CollgCr <= 0
 NOCHG_neg_Neighborhood_CollgCr: - const_Neighborhood_CollgCr <= 0
 NOCHG_pos_Neighborhood_Crawfor: const_Neighborhood_Crawfor <= 0
 NOCHG_neg_Neighborhood_Crawfor: - const_Neighborhood_Crawfor <= 0
 NOCHG_pos_Neighborhood_Edwards: const_Neighborhood_Edwards <= 0
 NOCHG_neg_Neighborhood_Edwards: - const_Neighborhood_Edwards <= 0
 NOCHG_pos_Neighborhood_Gilbert: const_Neighborhood_Gilbert <= 0
 NOCHG_neg_Neighborhood_Gilbert: - const_Neighborhood_Gilbert <= 0
 NOCHG_pos_Neighborhood_Greens: const_Neighborhood_Greens <= 0
 NOCHG_neg_Neighborhood_Greens: - const_Neighborhood_Greens <= 0
 NOCHG_pos_Neighborhood_GrnHill: const_Neighborhood_GrnHill <= 0
 NOCHG_neg_Neighborhood_GrnHill: - const_Neighborhood_GrnHill <= 0
 NOCHG_pos_Neighborhood_IDOTRR: const_Neighborhood_IDOTRR <= 0
 NOCHG_neg_Neighborhood_IDOTRR: - const_Neighborhood_IDOTRR <= 0
 NOCHG_pos_Neighborhood_Landmrk: const_Neighborhood_Landmrk <= 0
 NOCHG_neg_Neighborhood_Landmrk: - const_Neighborhood_Landmrk <= 0
 NOCHG_pos_Neighborhood_MeadowV: const_Neighborhood_MeadowV <= 0
 NOCHG_neg_Neighborhood_MeadowV: - const_Neighborhood_MeadowV <= 0
 NOCHG_pos_Neighborhood_Mitchel: const_Neighborhood_Mitchel <= 0
 NOCHG_neg_Neighborhood_Mitchel: - const_Neighborhood_Mitchel <= 0
 NOCHG_pos_Neighborhood_NPkVill: const_Neighborhood_NPkVill <= 0
 NOCHG_neg_Neighborhood_NPkVill: - const_Neighborhood_NPkVill <= 0
 NOCHG_pos_Neighborhood_NWAmes: const_Neighborhood_NWAmes <= 0
 NOCHG_neg_Neighborhood_NWAmes: - const_Neighborhood_NWAmes <= 0
 NOCHG_pos_Neighborhood_No_aplicames: const_Neighborhood_No_aplicames <= 1
 NOCHG_neg_Neighborhood_No_aplicames: - const_Neighborhood_No_aplicames
   <= -1
 NOCHG_pos_Neighborhood_NoRidge: const_Neighborhood_NoRidge <= 0
 NOCHG_neg_Neighborhood_NoRidge: - const_Neighborhood_NoRidge <= 0
 NOCHG_pos_Neighborhood_NridgHt: const_Neighborhood_NridgHt <= 0
 NOCHG_neg_Neighborhood_NridgHt: - const_Neighborhood_NridgHt <= 0
 NOCHG_pos_Neighborhood_OldTown: const_Neighborhood_OldTown <= 0
 NOCHG_neg_Neighborhood_OldTown: - const_Neighborhood_OldTown <= 0
 NOCHG_pos_Neighborhood_SWISU: const_Neighborhood_SWISU <= 0
 NOCHG_neg_Neighborhood_SWISU: - const_Neighborhood_SWISU <= 0
 NOCHG_pos_Neighborhood_Sawyer: const_Neighborhood_Sawyer <= 0
 NOCHG_neg_Neighborhood_Sawyer: - const_Neighborhood_Sawyer <= 0
 NOCHG_pos_Neighborhood_SawyerW: const_Neighborhood_SawyerW <= 0
 NOCHG_neg_Neighborhood_SawyerW: - const_Neighborhood_SawyerW <= 0
 NOCHG_pos_Neighborhood_Somerst: const_Neighborhood_Somerst <= 0
 NOCHG_neg_Neighborhood_Somerst: - const_Neighborhood_Somerst <= 0
 NOCHG_pos_Neighborhood_StoneBr: const_Neighborhood_StoneBr <= 0
 NOCHG_neg_Neighborhood_StoneBr: - const_Neighborhood_StoneBr <= 0
 NOCHG_pos_Neighborhood_Timber: const_Neighborhood_Timber <= 0
 NOCHG_neg_Neighborhood_Timber: - const_Neighborhood_Timber <= 0
 NOCHG_pos_Neighborhood_Veenker: const_Neighborhood_Veenker <= 0
 NOCHG_neg_Neighborhood_Veenker: - const_Neighborhood_Veenker <= 0
 NOCHG_pos_Condition_1_Artery: const_Condition_1_Artery <= 0
 NOCHG_neg_Condition_1_Artery: - const_Condition_1_Artery <= 0
 NOCHG_pos_Condition_1_Feedr: const_Condition_1_Feedr <= 0
 NOCHG_neg_Condition_1_Feedr: - const_Condition_1_Feedr <= 0
 NOCHG_pos_Condition_1_Norm: const_Condition_1_Norm <= 1
 NOCHG_neg_Condition_1_Norm: - const_Condition_1_Norm <= -1
 NOCHG_pos_Condition_1_PosA: const_Condition_1_PosA <= 0
 NOCHG_neg_Condition_1_PosA: - const_Condition_1_PosA <= 0
 NOCHG_pos_Condition_1_PosN: const_Condition_1_PosN <= 0
 NOCHG_neg_Condition_1_PosN: - const_Condition_1_PosN <= 0
 NOCHG_pos_Condition_1_RRAe: const_Condition_1_RRAe <= 0
 NOCHG_neg_Condition_1_RRAe: - const_Condition_1_RRAe <= 0
 NOCHG_pos_Condition_1_RRAn: const_Condition_1_RRAn <= 0
 NOCHG_neg_Condition_1_RRAn: - const_Condition_1_RRAn <= 0
 NOCHG_pos_Condition_1_RRNe: const_Condition_1_RRNe <= 0
 NOCHG_neg_Condition_1_RRNe: - const_Condition_1_RRNe <= 0
 NOCHG_pos_Condition_1_RRNn: const_Condition_1_RRNn <= 0
 NOCHG_neg_Condition_1_RRNn: - const_Condition_1_RRNn <= 0
 NOCHG_pos_Condition_2_Artery: const_Condition_2_Artery <= 0
 NOCHG_neg_Condition_2_Artery: - const_Condition_2_Artery <= 0
 NOCHG_pos_Condition_2_Feedr: const_Condition_2_Feedr <= 0
 NOCHG_neg_Condition_2_Feedr: - const_Condition_2_Feedr <= 0
 NOCHG_pos_Condition_2_Norm: const_Condition_2_Norm <= 1
 NOCHG_neg_Condition_2_Norm: - const_Condition_2_Norm <= -1
 NOCHG_pos_Condition_2_PosA: const_Condition_2_PosA <= 0
 NOCHG_neg_Condition_2_PosA: - const_Condition_2_PosA <= 0
 NOCHG_pos_Condition_2_PosN: const_Condition_2_PosN <= 0
 NOCHG_neg_Condition_2_PosN: - const_Condition_2_PosN <= 0
 NOCHG_pos_Condition_2_RRAe: const_Condition_2_RRAe <= 0
 NOCHG_neg_Condition_2_RRAe: - const_Condition_2_RRAe <= 0
 NOCHG_pos_Condition_2_RRAn: const_Condition_2_RRAn <= 0
 NOCHG_neg_Condition_2_RRAn: - const_Condition_2_RRAn <= 0
 NOCHG_pos_Condition_2_RRNn: const_Condition_2_RRNn <= 0
 NOCHG_neg_Condition_2_RRNn: - const_Condition_2_RRNn <= 0
 NOCHG_pos_Bldg_Type_1Fam: const_Bldg_Type_1Fam <= 1
 NOCHG_neg_Bldg_Type_1Fam: - const_Bldg_Type_1Fam <= -1
 NOCHG_pos_Bldg_Type_2fmCon: const_Bldg_Type_2fmCon <= 0
 NOCHG_neg_Bldg_Type_2fmCon: - const_Bldg_Type_2fmCon <= 0
 NOCHG_pos_Bldg_Type_Duplex: const_Bldg_Type_Duplex <= 0
 NOCHG_neg_Bldg_Type_Duplex: - const_Bldg_Type_Duplex <= 0
 NOCHG_pos_Bldg_Type_Twnhs: const_Bldg_Type_Twnhs <= 0
 NOCHG_neg_Bldg_Type_Twnhs: - const_Bldg_Type_Twnhs <= 0
 NOCHG_pos_Bldg_Type_TwnhsE: const_Bldg_Type_TwnhsE <= 0
 NOCHG_neg_Bldg_Type_TwnhsE: - const_Bldg_Type_TwnhsE <= 0
 NOCHG_pos_House_Style_1.5Fin: const_House_Style_1.5Fin <= 0
 NOCHG_neg_House_Style_1.5Fin: - const_House_Style_1.5Fin <= 0
 NOCHG_pos_House_Style_1.5Unf: const_House_Style_1.5Unf <= 0
 NOCHG_neg_House_Style_1.5Unf: - const_House_Style_1.5Unf <= 0
 NOCHG_pos_House_Style_1Story: const_House_Style_1Story <= 1
 NOCHG_neg_House_Style_1Story: - const_House_Style_1Story <= -1
 NOCHG_pos_House_Style_2.5Fin: const_House_Style_2.5Fin <= 0
 NOCHG_neg_House_Style_2.5Fin: - const_House_Style_2.5Fin <= 0
 NOCHG_pos_House_Style_2.5Unf: const_House_Style_2.5Unf <= 0
 NOCHG_neg_House_Style_2.5Unf: - const_House_Style_2.5Unf <= 0
 NOCHG_pos_House_Style_2Story: const_House_Style_2Story <= 0
 NOCHG_neg_House_Style_2Story: - const_House_Style_2Story <= 0
 NOCHG_pos_House_Style_SFoyer: const_House_Style_SFoyer <= 0
 NOCHG_neg_House_Style_SFoyer: - const_House_Style_SFoyer <= 0
 NOCHG_pos_House_Style_SLvl: const_House_Style_SLvl <= 0
 NOCHG_neg_House_Style_SLvl: - const_House_Style_SLvl <= 0
 NOCHG_pos_Roof_Style_Flat: const_Roof_Style_Flat <= 0
 NOCHG_neg_Roof_Style_Flat: - const_Roof_Style_Flat <= 0
 NOCHG_pos_Roof_Style_Gable: const_Roof_Style_Gable <= 0
 NOCHG_neg_Roof_Style_Gable: - const_Roof_Style_Gable <= 0
 NOCHG_pos_Roof_Style_Gambrel: const_Roof_Style_Gambrel <= 0
 NOCHG_neg_Roof_Style_Gambrel: - const_Roof_Style_Gambrel <= 0
 NOCHG_pos_Roof_Style_Hip: const_Roof_Style_Hip <= 1
 NOCHG_neg_Roof_Style_Hip: - const_Roof_Style_Hip <= -1
 NOCHG_pos_Roof_Style_Mansard: const_Roof_Style_Mansard <= 0
 NOCHG_neg_Roof_Style_Mansard: - const_Roof_Style_Mansard <= 0
 NOCHG_pos_Roof_Style_Shed: const_Roof_Style_Shed <= 0
 NOCHG_neg_Roof_Style_Shed: - const_Roof_Style_Shed <= 0
 NOCHG_pos_Roof_Matl_ClyTile: x_roof_matl_is_ClyTile - z_change <= 0
 NOCHG_neg_Roof_Matl_ClyTile: - x_roof_matl_is_ClyTile <= 0
 NOCHG_pos_Roof_Matl_CompShg: x_roof_matl_is_CompShg <= 1
 NOCHG_neg_Roof_Matl_CompShg: - x_roof_matl_is_CompShg - z_change <= -1
 NOCHG_pos_Roof_Matl_Membran: x_roof_matl_is_Membran - z_change <= 0
 NOCHG_neg_Roof_Matl_Membran: - x_roof_matl_is_Membran <= 0
 NOCHG_pos_Roof_Matl_Metal: x_roof_matl_is_Metal - z_change <= 0
 NOCHG_neg_Roof_Matl_Metal: - x_roof_matl_is_Metal <= 0
 NOCHG_pos_Roof_Matl_Roll: x_roof_matl_is_Roll - z_change <= 0
 NOCHG_neg_Roof_Matl_Roll: - x_roof_matl_is_Roll <= 0
 NOCHG_pos_Roof_Matl_Tar&Grv: x_roof_matl_is_Tar&Grv - z_change <= 0
 NOCHG_neg_Roof_Matl_Tar&Grv: - x_roof_matl_is_Tar&Grv <= 0
 NOCHG_pos_Roof_Matl_WdShake: x_roof_matl_is_WdShake - z_change <= 0
 NOCHG_neg_Roof_Matl_WdShake: - x_roof_matl_is_WdShake <= 0
 NOCHG_pos_Roof_Matl_WdShngl: x_roof_matl_is_WdShngl - z_change <= 0
 NOCHG_neg_Roof_Matl_WdShngl: - x_roof_matl_is_WdShngl <= 0
 NOCHG_pos_Exterior_1st_AsbShng: x_ex1_is_AsbShng - z_change <= 0
 NOCHG_neg_Exterior_1st_AsbShng: - x_ex1_is_AsbShng <= 0
 NOCHG_pos_Exterior_1st_AsphShn: x_ex1_is_AsphShn - z_change <= 0
 NOCHG_neg_Exterior_1st_AsphShn: - x_ex1_is_AsphShn <= 0
 NOCHG_pos_Exterior_1st_BrkComm: x_ex1_is_BrkComm - z_change <= 0
 NOCHG_neg_Exterior_1st_BrkComm: - x_ex1_is_BrkComm <= 0
 NOCHG_pos_Exterior_1st_BrkFace: x_ex1_is_BrkFace - z_change <= 0
 NOCHG_neg_Exterior_1st_BrkFace: - x_ex1_is_BrkFace <= 0
 NOCHG_pos_Exterior_1st_CBlock: x_ex1_is_CBlock - z_change <= 0
 NOCHG_neg_Exterior_1st_CBlock: - x_ex1_is_CBlock <= 0
 NOCHG_pos_Exterior_1st_CemntBd: x_ex1_is_CemntBd - z_change <= 0
 NOCHG_neg_Exterior_1st_CemntBd: - x_ex1_is_CemntBd <= 0
 NOCHG_pos_Exterior_1st_HdBoard: x_ex1_is_HdBoard - z_change <= 0
 NOCHG_neg_Exterior_1st_HdBoard: - x_ex1_is_HdBoard <= 0
 NOCHG_pos_Exterior_1st_ImStucc: x_ex1_is_ImStucc - z_change <= 0
 NOCHG_neg_Exterior_1st_ImStucc: - x_ex1_is_ImStucc <= 0
 NOCHG_pos_Exterior_1st_MetalSd: x_ex1_is_MetalSd - z_change <= 0
 NOCHG_neg_Exterior_1st_MetalSd: - x_ex1_is_MetalSd <= 0
 NOCHG_pos_Exterior_1st_Plywood: x_ex1_is_Plywood - z_change <= 0
 NOCHG_neg_Exterior_1st_Plywood: - x_ex1_is_Plywood <= 0
 NOCHG_pos_Exterior_1st_PreCast: x_ex1_is_PreCast - z_change <= 0
 NOCHG_neg_Exterior_1st_PreCast: - x_ex1_is_PreCast <= 0
 NOCHG_pos_Exterior_1st_Stone: x_ex1_is_Stone - z_change <= 0
 NOCHG_neg_Exterior_1st_Stone: - x_ex1_is_Stone <= 0
 NOCHG_pos_Exterior_1st_Stucco: x_ex1_is_Stucco - z_change <= 0
 NOCHG_neg_Exterior_1st_Stucco: - x_ex1_is_Stucco <= 0
 NOCHG_pos_Exterior_1st_VinylSd: x_ex1_is_VinylSd - z_change <= 0
 NOCHG_neg_Exterior_1st_VinylSd: - x_ex1_is_VinylSd <= 0
 NOCHG_pos_Exterior_1st_Wd_Sdng: x_ex1_is_Wd_Sdng <= 1
 NOCHG_neg_Exterior_1st_Wd_Sdng: - x_ex1_is_Wd_Sdng - z_change <= -1
 NOCHG_pos_Exterior_1st_WdShing: const_Exterior_1st_WdShing <= 0
 NOCHG_neg_Exterior_1st_WdShing: - const_Exterior_1st_WdShing <= 0
 NOCHG_pos_Exterior_2nd_AsbShng: x_ex2_is_AsbShng - z_change <= 0
 NOCHG_neg_Exterior_2nd_AsbShng: - x_ex2_is_AsbShng <= 0
 NOCHG_pos_Exterior_2nd_AsphShn: x_ex2_is_AsphShn - z_change <= 0
 NOCHG_neg_Exterior_2nd_AsphShn: - x_ex2_is_AsphShn <= 0
 NOCHG_pos_Exterior_2nd_Brk_Cmn: const_Exterior_2nd_Brk_Cmn <= 0
 NOCHG_neg_Exterior_2nd_Brk_Cmn: - const_Exterior_2nd_Brk_Cmn <= 0
 NOCHG_pos_Exterior_2nd_BrkFace: x_ex2_is_BrkFace - z_change <= 0
 NOCHG_neg_Exterior_2nd_BrkFace: - x_ex2_is_BrkFace <= 0
 NOCHG_pos_Exterior_2nd_CBlock: x_ex2_is_CBlock - z_change <= 0
 NOCHG_neg_Exterior_2nd_CBlock: - x_ex2_is_CBlock <= 0
 NOCHG_pos_Exterior_2nd_CmentBd: const_Exterior_2nd_CmentBd <= 0
 NOCHG_neg_Exterior_2nd_CmentBd: - const_Exterior_2nd_CmentBd <= 0
 NOCHG_pos_Exterior_2nd_HdBoard: x_ex2_is_HdBoard - z_change <= 0
 NOCHG_neg_Exterior_2nd_HdBoard: - x_ex2_is_HdBoard <= 0
 NOCHG_pos_Exterior_2nd_ImStucc: x_ex2_is_ImStucc - z_change <= 0
 NOCHG_neg_Exterior_2nd_ImStucc: - x_ex2_is_ImStucc <= 0
 NOCHG_pos_Exterior_2nd_MetalSd: x_ex2_is_MetalSd - z_change <= 0
 NOCHG_neg_Exterior_2nd_MetalSd: - x_ex2_is_MetalSd <= 0
 NOCHG_pos_Exterior_2nd_Other: x_ex2_is_Other - z_change <= 0
 NOCHG_neg_Exterior_2nd_Other: - x_ex2_is_Other <= 0
 NOCHG_pos_Exterior_2nd_Plywood: x_ex2_is_Plywood - z_change <= 0
 NOCHG_neg_Exterior_2nd_Plywood: - x_ex2_is_Plywood <= 0
 NOCHG_pos_Exterior_2nd_PreCast: x_ex2_is_PreCast - z_change <= 0
 NOCHG_neg_Exterior_2nd_PreCast: - x_ex2_is_PreCast <= 0
 NOCHG_pos_Exterior_2nd_Stone: x_ex2_is_Stone - z_change <= 0
 NOCHG_neg_Exterior_2nd_Stone: - x_ex2_is_Stone <= 0
 NOCHG_pos_Exterior_2nd_Stucco: x_ex2_is_Stucco - z_change <= 0
 NOCHG_neg_Exterior_2nd_Stucco: - x_ex2_is_Stucco <= 0
 NOCHG_pos_Exterior_2nd_VinylSd: x_ex2_is_VinylSd - z_change <= 0
 NOCHG_neg_Exterior_2nd_VinylSd: - x_ex2_is_VinylSd <= 0
 NOCHG_pos_Exterior_2nd_Wd_Sdng: x_ex2_is_Wd_Sdng <= 1
 NOCHG_neg_Exterior_2nd_Wd_Sdng: - x_ex2_is_Wd_Sdng - z_change <= -1
 NOCHG_pos_Exterior_2nd_Wd_Shng: const_Exterior_2nd_Wd_Shng <= 0
 NOCHG_neg_Exterior_2nd_Wd_Shng: - const_Exterior_2nd_Wd_Shng <= 0
 NOCHG_pos_Mas_Vnr_Type_BrkCmn: x_mvt_is_BrkCmn - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_BrkCmn: - x_mvt_is_BrkCmn <= 0
 NOCHG_pos_Mas_Vnr_Type_BrkFace: x_mvt_is_BrkFace <= 1
 NOCHG_neg_Mas_Vnr_Type_BrkFace: - x_mvt_is_BrkFace - z_change <= -1
 NOCHG_pos_Mas_Vnr_Type_CBlock: x_mvt_is_CBlock - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_CBlock: - x_mvt_is_CBlock <= 0
 NOCHG_pos_Mas_Vnr_Type_No_aplica: x_mvt_is_No_aplica - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_No_aplica: - x_mvt_is_No_aplica <= 0
 NOCHG_pos_Mas_Vnr_Type_NoAplica: x_mvt_is_No_aplica - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_NoAplica: - x_mvt_is_No_aplica <= 0
 NOCHG_pos_Mas_Vnr_Type_Stone: x_mvt_is_Stone - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_Stone: - x_mvt_is_Stone <= 0
 NOCHG_pos_Foundation_BrkTil: const_Foundation_BrkTil <= 0
 NOCHG_neg_Foundation_BrkTil: - const_Foundation_BrkTil <= 0
 NOCHG_pos_Foundation_CBlock: const_Foundation_CBlock <= 1
 NOCHG_neg_Foundation_CBlock: - const_Foundation_CBlock <= -1
 NOCHG_pos_Foundation_PConc: const_Foundation_PConc <= 0
 NOCHG_neg_Foundation_PConc: - const_Foundation_PConc <= 0
 NOCHG_pos_Foundation_Slab: const_Foundation_Slab <= 0
 NOCHG_neg_Foundation_Slab: - const_Foundation_Slab <= 0
 NOCHG_pos_Foundation_Stone: const_Foundation_Stone <= 0
 NOCHG_neg_Foundation_Stone: - const_Foundation_Stone <= 0
 NOCHG_pos_Foundation_Wood: const_Foundation_Wood <= 0
 NOCHG_neg_Foundation_Wood: - const_Foundation_Wood <= 0
 NOCHG_pos_Bsmt_Exposure_Av: const_Bsmt_Exposure_Av <= 0
 NOCHG_neg_Bsmt_Exposure_Av: - const_Bsmt_Exposure_Av <= 0
 NOCHG_pos_Bsmt_Exposure_Gd: const_Bsmt_Exposure_Gd <= 0
 NOCHG_neg_Bsmt_Exposure_Gd: - const_Bsmt_Exposure_Gd <= 0
 NOCHG_pos_Bsmt_Exposure_Mn: const_Bsmt_Exposure_Mn <= 0
 NOCHG_neg_Bsmt_Exposure_Mn: - const_Bsmt_Exposure_Mn <= 0
 NOCHG_pos_Bsmt_Exposure_No: const_Bsmt_Exposure_No <= 1
 NOCHG_neg_Bsmt_Exposure_No: - const_Bsmt_Exposure_No <= -1
 NOCHG_pos_Bsmt_Exposure_No_aplica: const_Bsmt_Exposure_No_aplica <= 0
 NOCHG_neg_Bsmt_Exposure_No_aplica: - const_Bsmt_Exposure_No_aplica <= 0
 NOCHG_pos_BsmtFin_Type_1_ALQ: x_b1_is_ALQ <= 1
 NOCHG_neg_BsmtFin_Type_1_ALQ: - x_b1_is_ALQ - z_change <= -1
 NOCHG_pos_BsmtFin_Type_1_BLQ: x_b1_is_BLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_BLQ: - x_b1_is_BLQ <= 0
 NOCHG_pos_BsmtFin_Type_1_GLQ: x_b1_is_GLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_GLQ: - x_b1_is_GLQ <= 0
 NOCHG_pos_BsmtFin_Type_1_LwQ: x_b1_is_LwQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_LwQ: - x_b1_is_LwQ <= 0
 NOCHG_pos_BsmtFin_Type_1_No_aplica: const_BsmtFin_Type_1_No_aplica <= 0
 NOCHG_neg_BsmtFin_Type_1_No_aplica: - const_BsmtFin_Type_1_No_aplica <= 0
 NOCHG_pos_BsmtFin_Type_1_Rec: x_b1_is_Rec - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_Rec: - x_b1_is_Rec <= 0
 NOCHG_pos_BsmtFin_Type_1_Unf: x_b1_is_Unf - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_Unf: - x_b1_is_Unf <= 0
 NOCHG_pos_BsmtFin_Type_2_ALQ: x_b2_is_ALQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_ALQ: - x_b2_is_ALQ <= 0
 NOCHG_pos_BsmtFin_Type_2_BLQ: x_b2_is_BLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_BLQ: - x_b2_is_BLQ <= 0
 NOCHG_pos_BsmtFin_Type_2_GLQ: x_b2_is_GLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_GLQ: - x_b2_is_GLQ <= 0
 NOCHG_pos_BsmtFin_Type_2_LwQ: x_b2_is_LwQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_LwQ: - x_b2_is_LwQ <= 0
 NOCHG_pos_BsmtFin_Type_2_No_aplica: const_BsmtFin_Type_2_No_aplica <= 0
 NOCHG_neg_BsmtFin_Type_2_No_aplica: - const_BsmtFin_Type_2_No_aplica <= 0
 NOCHG_pos_BsmtFin_Type_2_Rec: x_b2_is_Rec - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_Rec: - x_b2_is_Rec <= 0
 NOCHG_pos_BsmtFin_Type_2_Unf: x_b2_is_Unf <= 1
 NOCHG_neg_BsmtFin_Type_2_Unf: - x_b2_is_Unf - z_change <= -1
 NOCHG_pos_Heating_Floor: heat_is_Floor <= 0
 NOCHG_neg_Heating_Floor: - heat_is_Floor <= 0
 NOCHG_pos_Heating_GasA: heat_is_GasA <= 1
 NOCHG_neg_Heating_GasA: - heat_is_GasA - z_change <= -1
 NOCHG_pos_Heating_GasW: heat_is_GasW - z_change <= 0
 NOCHG_neg_Heating_GasW: - heat_is_GasW <= 0
 NOCHG_pos_Heating_Grav: heat_is_Grav - z_change <= 0
 NOCHG_neg_Heating_Grav: - heat_is_Grav <= 0
 NOCHG_pos_Heating_OthW: heat_is_OthW <= 0
 NOCHG_neg_Heating_OthW: - heat_is_OthW <= 0
 NOCHG_pos_Heating_Wall: heat_is_Wall <= 0
 NOCHG_neg_Heating_Wall: - heat_is_Wall <= 0
 NOCHG_pos_Central_Air_N: central_air_no - z_change <= 0
 NOCHG_neg_Central_Air_N: - central_air_no <= 0
 NOCHG_pos_Central_Air_Y: central_air_yes <= 1
 NOCHG_neg_Central_Air_Y: - central_air_yes <= -1
 NOCHG_pos_Electrical_FuseA: x_elect_is_FuseA - z_change <= 0
 NOCHG_neg_Electrical_FuseA: - x_elect_is_FuseA <= 0
 NOCHG_pos_Electrical_FuseF: x_elect_is_FuseF - z_change <= 0
 NOCHG_neg_Electrical_FuseF: - x_elect_is_FuseF <= 0
 NOCHG_pos_Electrical_FuseP: x_elect_is_FuseP - z_change <= 0
 NOCHG_neg_Electrical_FuseP: - x_elect_is_FuseP <= 0
 NOCHG_pos_Electrical_Mix: x_elect_is_Mix - z_change <= 0
 NOCHG_neg_Electrical_Mix: - x_elect_is_Mix <= 0
 NOCHG_pos_Electrical_SBrkr: x_elect_is_SBrkr <= 1
 NOCHG_neg_Electrical_SBrkr: - x_elect_is_SBrkr - z_change <= -1
 NOCHG_pos_FunctioNo_aplical_Maj1: const_FunctioNo_aplical_Maj1 <= 0
 NOCHG_neg_FunctioNo_aplical_Maj1: - const_FunctioNo_aplical_Maj1 <= 0
 NOCHG_pos_FunctioNo_aplical_Maj2: const_FunctioNo_aplical_Maj2 <= 0
 NOCHG_neg_FunctioNo_aplical_Maj2: - const_FunctioNo_aplical_Maj2 <= 0
 NOCHG_pos_FunctioNo_aplical_Min1: const_FunctioNo_aplical_Min1 <= 0
 NOCHG_neg_FunctioNo_aplical_Min1: - const_FunctioNo_aplical_Min1 <= 0
 NOCHG_pos_FunctioNo_aplical_Min2: const_FunctioNo_aplical_Min2 <= 0
 NOCHG_neg_FunctioNo_aplical_Min2: - const_FunctioNo_aplical_Min2 <= 0
 NOCHG_pos_FunctioNo_aplical_Mod: const_FunctioNo_aplical_Mod <= 0
 NOCHG_neg_FunctioNo_aplical_Mod: - const_FunctioNo_aplical_Mod <= 0
 NOCHG_pos_FunctioNo_aplical_Sal: const_FunctioNo_aplical_Sal <= 0
 NOCHG_neg_FunctioNo_aplical_Sal: - const_FunctioNo_aplical_Sal <= 0
 NOCHG_pos_FunctioNo_aplical_Sev: const_FunctioNo_aplical_Sev <= 0
 NOCHG_neg_FunctioNo_aplical_Sev: - const_FunctioNo_aplical_Sev <= 0
 NOCHG_pos_FunctioNo_aplical_Typ: const_FunctioNo_aplical_Typ <= 1
 NOCHG_neg_FunctioNo_aplical_Typ: - const_FunctioNo_aplical_Typ <= -1
 NOCHG_pos_Garage_Type_2Types: const_Garage_Type_2Types <= 0
 NOCHG_neg_Garage_Type_2Types: - const_Garage_Type_2Types <= 0
 NOCHG_pos_Garage_Type_Attchd: const_Garage_Type_Attchd <= 1
 NOCHG_neg_Garage_Type_Attchd: - const_Garage_Type_Attchd <= -1
 NOCHG_pos_Garage_Type_Basment: const_Garage_Type_Basment <= 0
 NOCHG_neg_Garage_Type_Basment: - const_Garage_Type_Basment <= 0
 NOCHG_pos_Garage_Type_BuiltIn: const_Garage_Type_BuiltIn <= 0
 NOCHG_neg_Garage_Type_BuiltIn: - const_Garage_Type_BuiltIn <= 0
 NOCHG_pos_Garage_Type_CarPort: const_Garage_Type_CarPort <= 0
 NOCHG_neg_Garage_Type_CarPort: - const_Garage_Type_CarPort <= 0
 NOCHG_pos_Garage_Type_Detchd: const_Garage_Type_Detchd <= 0
 NOCHG_neg_Garage_Type_Detchd: - const_Garage_Type_Detchd <= 0
 NOCHG_pos_Garage_Type_No_aplica: const_Garage_Type_No_aplica <= 0
 NOCHG_neg_Garage_Type_No_aplica: - const_Garage_Type_No_aplica <= 0
 NOCHG_pos_Garage_Finish_Fin: const_Garage_Finish_Fin <= 0
 NOCHG_neg_Garage_Finish_Fin: - const_Garage_Finish_Fin <= 0
 NOCHG_pos_Garage_Finish_No_aplica: const_Garage_Finish_No_aplica <= 0
 NOCHG_neg_Garage_Finish_No_aplica: - const_Garage_Finish_No_aplica <= 0
 NOCHG_pos_Garage_Finish_RFn: const_Garage_Finish_RFn <= 0
 NOCHG_neg_Garage_Finish_RFn: - const_Garage_Finish_RFn <= 0
 NOCHG_pos_Garage_Finish_Unf: const_Garage_Finish_Unf <= 1
 NOCHG_neg_Garage_Finish_Unf: - const_Garage_Finish_Unf <= -1
 NOCHG_pos_Paved_Drive_N: x_paved_drive_is_N - z_change <= 0
 NOCHG_neg_Paved_Drive_N: - x_paved_drive_is_N <= 0
 NOCHG_pos_Paved_Drive_P: x_paved_drive_is_P - z_change <= 0
 NOCHG_neg_Paved_Drive_P: - x_paved_drive_is_P <= 0
 NOCHG_pos_Paved_Drive_Y: x_paved_drive_is_Y <= 1
 NOCHG_neg_Paved_Drive_Y: - x_paved_drive_is_Y - z_change <= -1
 NOCHG_pos_Fence_GdPrv: x_fence_is_GdPrv - z_change <= 0
 NOCHG_neg_Fence_GdPrv: - x_fence_is_GdPrv <= 0
 NOCHG_pos_Fence_GdWo: x_fence_is_GdWo - z_change <= 0
 NOCHG_neg_Fence_GdWo: - x_fence_is_GdWo <= 0
 NOCHG_pos_Fence_MnPrv: x_fence_is_MnPrv - z_change <= 0
 NOCHG_neg_Fence_MnPrv: - x_fence_is_MnPrv <= 0
 NOCHG_pos_Fence_MnWw: x_fence_is_MnWw - z_change <= 0
 NOCHG_neg_Fence_MnWw: - x_fence_is_MnWw <= 0
 NOCHG_pos_Fence_No_aplica: x_fence_is_No_aplica <= 1
 NOCHG_neg_Fence_No_aplica: - x_fence_is_No_aplica - z_change <= -1
 NOCHG_pos_Misc_Feature_Elev: const_Misc_Feature_Elev <= 0
 NOCHG_neg_Misc_Feature_Elev: - const_Misc_Feature_Elev <= 0
 NOCHG_pos_Misc_Feature_Gar2: const_Misc_Feature_Gar2 <= 1
 NOCHG_neg_Misc_Feature_Gar2: - const_Misc_Feature_Gar2 <= -1
 NOCHG_pos_Misc_Feature_No_aplica: const_Misc_Feature_No_aplica <= 0
 NOCHG_neg_Misc_Feature_No_aplica: - const_Misc_Feature_No_aplica <= 0
 NOCHG_pos_Misc_Feature_Othr: const_Misc_Feature_Othr <= 0
 NOCHG_neg_Misc_Feature_Othr: - const_Misc_Feature_Othr <= 0
 NOCHG_pos_Misc_Feature_Shed: const_Misc_Feature_Shed <= 0
 NOCHG_neg_Misc_Feature_Shed: - const_Misc_Feature_Shed <= 0
 NOCHG_pos_Misc_Feature_TenC: const_Misc_Feature_TenC <= 0
 NOCHG_neg_Misc_Feature_TenC: - const_Misc_Feature_TenC <= 0
 NOCHG_pos_Mo_Sold_1: const_Mo_Sold_1 <= 0
 NOCHG_neg_Mo_Sold_1: - const_Mo_Sold_1 <= 0
 NOCHG_pos_Mo_Sold_2: const_Mo_Sold_2 <= 0
 NOCHG_neg_Mo_Sold_2: - const_Mo_Sold_2 <= 0
 NOCHG_pos_Mo_Sold_3: const_Mo_Sold_3 <= 0
 NOCHG_neg_Mo_Sold_3: - const_Mo_Sold_3 <= 0
 NOCHG_pos_Mo_Sold_4: const_Mo_Sold_4 <= 0
 NOCHG_neg_Mo_Sold_4: - const_Mo_Sold_4 <= 0
 NOCHG_pos_Mo_Sold_5: const_Mo_Sold_5 <= 0
 NOCHG_neg_Mo_Sold_5: - const_Mo_Sold_5 <= 0
 NOCHG_pos_Mo_Sold_6: const_Mo_Sold_6 <= 1
 NOCHG_neg_Mo_Sold_6: - const_Mo_Sold_6 <= -1
 NOCHG_pos_Mo_Sold_7: const_Mo_Sold_7 <= 0
 NOCHG_neg_Mo_Sold_7: - const_Mo_Sold_7 <= 0
 NOCHG_pos_Mo_Sold_8: const_Mo_Sold_8 <= 0
 NOCHG_neg_Mo_Sold_8: - const_Mo_Sold_8 <= 0
 NOCHG_pos_Mo_Sold_9: const_Mo_Sold_9 <= 0
 NOCHG_neg_Mo_Sold_9: - const_Mo_Sold_9 <= 0
 NOCHG_pos_Mo_Sold_10: const_Mo_Sold_10 <= 0
 NOCHG_neg_Mo_Sold_10: - const_Mo_Sold_10 <= 0
 NOCHG_pos_Mo_Sold_11: const_Mo_Sold_11 <= 0
 NOCHG_neg_Mo_Sold_11: - const_Mo_Sold_11 <= 0
 NOCHG_pos_Mo_Sold_12: const_Mo_Sold_12 <= 0
 NOCHG_neg_Mo_Sold_12: - const_Mo_Sold_12 <= 0
 NOCHG_pos_Sale_Type_COD: const_Sale_Type_COD <= 0
 NOCHG_neg_Sale_Type_COD: - const_Sale_Type_COD <= 0
 NOCHG_pos_Sale_Type_CWD: const_Sale_Type_CWD <= 0
 NOCHG_neg_Sale_Type_CWD: - const_Sale_Type_CWD <= 0
 NOCHG_pos_Sale_Type_Con: const_Sale_Type_Con <= 0
 NOCHG_neg_Sale_Type_Con: - const_Sale_Type_Con <= 0
 NOCHG_pos_Sale_Type_ConLD: const_Sale_Type_ConLD <= 0
 NOCHG_neg_Sale_Type_ConLD: - const_Sale_Type_ConLD <= 0
 NOCHG_pos_Sale_Type_ConLI: const_Sale_Type_ConLI <= 0
 NOCHG_neg_Sale_Type_ConLI: - const_Sale_Type_ConLI <= 0
 NOCHG_pos_Sale_Type_ConLw: const_Sale_Type_ConLw <= 0
 NOCHG_neg_Sale_Type_ConLw: - const_Sale_Type_ConLw <= 0
 NOCHG_pos_Sale_Type_New: const_Sale_Type_New <= 0
 NOCHG_neg_Sale_Type_New: - const_Sale_Type_New <= 0
 NOCHG_pos_Sale_Type_Oth: const_Sale_Type_Oth <= 0
 NOCHG_neg_Sale_Type_Oth: - const_Sale_Type_Oth <= 0
 NOCHG_pos_Sale_Type_VWD: const_Sale_Type_VWD <= 0
 NOCHG_neg_Sale_Type_VWD: - const_Sale_Type_VWD <= 0
 NOCHG_pos_Sale_Type_WD_: const_Sale_Type_WD_ <= 1
 NOCHG_neg_Sale_Type_WD_: - const_Sale_Type_WD_ <= -1
 NOCHG_pos_Sale_Condition_Abnorml: const_Sale_Condition_Abnorml <= 0
 NOCHG_neg_Sale_Condition_Abnorml: - const_Sale_Condition_Abnorml <= 0
 NOCHG_pos_Sale_Condition_AdjLand: const_Sale_Condition_AdjLand <= 0
 NOCHG_neg_Sale_Condition_AdjLand: - const_Sale_Condition_AdjLand <= 0
 NOCHG_pos_Sale_Condition_Alloca: const_Sale_Condition_Alloca <= 0
 NOCHG_neg_Sale_Condition_Alloca: - const_Sale_Condition_Alloca <= 0
 NOCHG_pos_Sale_Condition_Family: const_Sale_Condition_Family <= 0
 NOCHG_neg_Sale_Condition_Family: - const_Sale_Condition_Family <= 0
 NOCHG_pos_Sale_Condition_Normal: const_Sale_Condition_Normal <= 1
 NOCHG_neg_Sale_Condition_Normal: - const_Sale_Condition_Normal <= -1
 NOCHG_pos_Sale_Condition_Partial: const_Sale_Condition_Partial <= 0
 NOCHG_neg_Sale_Condition_Partial: - const_Sale_Condition_Partial <= 0
 NOCHG_price_pos: y_price - 2e+06 z_change <= 240266.984375
 NOCHG_price_neg: - y_price - 2e+06 z_change <= -240266.984375
 MIN_PRICE_BASE: y_price >= 240266.984374
 NO_NEGATIVE_ROI: y_price - cost_model >= 240266.984374
 ylog_penalty_ub_iter1: y_log <= 12.44124771118164
 ylog_penalty_lb_iter1: y_log >= 12.42124771118164
 ylog_penalty_ub_iter2: y_log <= 12.35893035888672
 ylog_penalty_lb_iter2: y_log >= 12.33893035888672
 ind_0: t0_path0 = 1 -> const_Year_Built <= 1986
 ind_1: t0_path0 = 1 -> const_Fireplaces <= 1
 ind_2: t0_path0 = 1 -> const_Year_Remod_Add <= 1954
 ind_3: t0_path1 = 1 -> const_Year_Built <= 1986
 ind_4: t0_path1 = 1 -> const_Fireplaces <= 1
 ind_5: t0_path1 = 1 -> const_Year_Remod_Add >= 1954
 ind_6: t0_path2 = 1 -> const_Year_Built <= 1986
 ind_7: t0_path2 = 1 -> const_Fireplaces >= 1
 ind_8: t0_path2 = 1 -> x_Gr_Liv_Area <= 1740
 ind_9: t0_path3 = 1 -> const_Year_Built <= 1986
 ind_10: t0_path3 = 1 -> const_Fireplaces >= 1
 ind_11: t0_path3 = 1 -> x_Gr_Liv_Area >= 1740
 ind_12: t0_path4 = 1 -> const_Year_Built >= 1986
 ind_13: t0_path4 = 1 -> x_Garage_Area <= 650
 ind_14: t0_path4 = 1 -> x_Gr_Liv_Area <= 1484
 ind_15: t0_path5 = 1 -> const_Year_Built >= 1986
 ind_16: t0_path5 = 1 -> x_Garage_Area <= 650
 ind_17: t0_path5 = 1 -> x_Gr_Liv_Area >= 1484
 ind_18: t0_path6 = 1 -> const_Year_Built >= 1986
 ind_19: t0_path6 = 1 -> x_Garage_Area >= 650
 ind_20: t0_path6 = 1 -> x_Gr_Liv_Area <= 1661
 ind_21: t0_path7 = 1 -> const_Year_Built >= 1986
 ind_22: t0_path7 = 1 -> x_Garage_Area >= 650
 ind_23: t0_path7 = 1 -> x_Gr_Liv_Area >= 1661
 ind_24: t1_path0 = 1 -> Overall_Qual_calc <= 7
 ind_25: t1_path0 = 1 -> Overall_Qual_calc <= 5
 ind_26: t1_path0 = 1 -> const_Total_Bsmt_SF <= 864
 ind_27: t1_path1 = 1 -> Overall_Qual_calc <= 7
 ind_28: t1_path1 = 1 -> Overall_Qual_calc <= 5
 ind_29: t1_path1 = 1 -> const_Total_Bsmt_SF >= 864
 ind_30: t1_path2 = 1 -> Overall_Qual_calc <= 7
 ind_31: t1_path2 = 1 -> Overall_Qual_calc >= 5
 ind_32: t1_path2 = 1 -> const_Garage_Cars <= 2
 ind_33: t1_path3 = 1 -> Overall_Qual_calc <= 7
 ind_34: t1_path3 = 1 -> Overall_Qual_calc >= 5
 ind_35: t1_path3 = 1 -> const_Garage_Cars >= 2
 ind_36: t1_path4 = 1 -> Overall_Qual_calc >= 7
 ind_37: t1_path4 = 1 -> Overall_Qual_calc <= 8
 ind_38: t1_path4 = 1 -> const_Total_Bsmt_SF <= 845
 ind_39: t1_path5 = 1 -> Overall_Qual_calc >= 7
 ind_40: t1_path5 = 1 -> Overall_Qual_calc <= 8
 ind_41: t1_path5 = 1 -> const_Total_Bsmt_SF >= 845
 ind_42: t1_path6 = 1 -> Overall_Qual_calc >= 7
 ind_43: t1_path6 = 1 -> Overall_Qual_calc >= 8
 ind_44: t1_path6 = 1 -> x_Gr_Liv_Area <= 1866
 ind_45: t1_path7 = 1 -> Overall_Qual_calc >= 7
 ind_46: t1_path7 = 1 -> Overall_Qual_calc >= 8
 ind_47: t1_path7 = 1 -> x_Gr_Liv_Area >= 1866
 ind_48: t2_path0 = 1 -> x_Exter_Qual <= 3
 ind_49: t2_path0 = 1 -> x_Gr_Liv_Area <= 1308
 ind_50: t2_path0 = 1 -> const_Year_Built <= 1953
 ind_51: t2_path1 = 1 -> x_Exter_Qual <= 3
 ind_52: t2_path1 = 1 -> x_Gr_Liv_Area <= 1308
 ind_53: t2_path1 = 1 -> const_Year_Built >= 1953
 ind_54: t2_path2 = 1 -> x_Exter_Qual <= 3
 ind_55: t2_path2 = 1 -> x_Gr_Liv_Area >= 1308
 ind_56: t2_path2 = 1 -> x_BsmtFin_SF_1 <= 390
 ind_57: t2_path3 = 1 -> x_Exter_Qual <= 3
 ind_58: t2_path3 = 1 -> x_Gr_Liv_Area >= 1308
 ind_59: t2_path3 = 1 -> x_BsmtFin_SF_1 >= 390
 ind_60: t2_path4 = 1 -> x_Exter_Qual >= 3
 ind_61: t2_path4 = 1 -> x_Gr_Liv_Area <= 1627
 ind_62: t2_path4 = 1 -> x_1st_Flr_SF <= 1269
 ind_63: t2_path5 = 1 -> x_Exter_Qual >= 3
 ind_64: t2_path5 = 1 -> x_Gr_Liv_Area <= 1627
 ind_65: t2_path5 = 1 -> x_1st_Flr_SF >= 1269
 ind_66: t2_path6 = 1 -> x_Exter_Qual >= 3
 ind_67: t2_path6 = 1 -> x_Gr_Liv_Area >= 1627
 ind_68: t2_path6 = 1 -> x_1st_Flr_SF <= 1085
 ind_69: t2_path7 = 1 -> x_Exter_Qual >= 3
 ind_70: t2_path7 = 1 -> x_Gr_Liv_Area >= 1627
 ind_71: t2_path7 = 1 -> x_1st_Flr_SF >= 1085
 ind_72: t3_path0 = 1 -> Overall_Qual_calc <= 7
 ind_73: t3_path0 = 1 -> Overall_Qual_calc <= 5
 ind_74: t3_path0 = 1 -> const_Total_Bsmt_SF <= 800
 ind_75: t3_path1 = 1 -> Overall_Qual_calc <= 7
 ind_76: t3_path1 = 1 -> Overall_Qual_calc <= 5
 ind_77: t3_path1 = 1 -> const_Total_Bsmt_SF >= 800
 ind_78: t3_path2 = 1 -> Overall_Qual_calc <= 7
 ind_79: t3_path2 = 1 -> Overall_Qual_calc >= 5
 ind_80: t3_path2 = 1 -> x_Gr_Liv_Area <= 1362
 ind_81: t3_path3 = 1 -> Overall_Qual_calc <= 7
 ind_82: t3_path3 = 1 -> Overall_Qual_calc >= 5
 ind_83: t3_path3 = 1 -> x_Gr_Liv_Area >= 1362
 ind_84: t3_path4 = 1 -> Overall_Qual_calc >= 7
 ind_85: t3_path4 = 1 -> Overall_Qual_calc <= 8
 ind_86: t3_path4 = 1 -> x_Gr_Liv_Area <= 1478
 ind_87: t3_path5 = 1 -> Overall_Qual_calc >= 7
 ind_88: t3_path5 = 1 -> Overall_Qual_calc <= 8
 ind_89: t3_path5 = 1 -> x_Gr_Liv_Area >= 1478
 ind_90: t3_path6 = 1 -> Overall_Qual_calc >= 7
 ind_91: t3_path6 = 1 -> Overall_Qual_calc >= 8
 ind_92: t3_path6 = 1 -> Overall_Qual_calc <= 9
 ind_93: t3_path7 = 1 -> Overall_Qual_calc >= 7
 ind_94: t3_path7 = 1 -> Overall_Qual_calc >= 8
 ind_95: t3_path7 = 1 -> Overall_Qual_calc >= 9
 ind_96: t4_path0 = 1 -> x_Exter_Qual <= 3
 ind_97: t4_path0 = 1 -> x_Gr_Liv_Area <= 1096
 ind_98: t4_path0 = 1 -> central_air_no <= 1
 ind_99: t4_path1 = 1 -> x_Exter_Qual <= 3
 ind_100: t4_path1 = 1 -> x_Gr_Liv_Area <= 1096
 ind_101: t4_path1 = 1 -> central_air_no >= 1
 ind_102: t4_path2 = 1 -> x_Exter_Qual <= 3
 ind_103: t4_path2 = 1 -> x_Gr_Liv_Area >= 1096
 ind_104: t4_path2 = 1 -> const_Fireplaces <= 1
 ind_105: t4_path3 = 1 -> x_Exter_Qual <= 3
 ind_106: t4_path3 = 1 -> x_Gr_Liv_Area >= 1096
 ind_107: t4_path3 = 1 -> const_Fireplaces >= 1
 ind_108: t4_path4 = 1 -> x_Exter_Qual >= 3
 ind_109: t4_path4 = 1 -> const_Garage_Cars <= 3
 ind_110: t4_path4 = 1 -> x_1st_Flr_SF <= 1414
 ind_111: t4_path5 = 1 -> x_Exter_Qual >= 3
 ind_112: t4_path5 = 1 -> const_Garage_Cars <= 3
 ind_113: t4_path5 = 1 -> x_1st_Flr_SF >= 1414
 ind_114: t4_path6 = 1 -> x_Exter_Qual >= 3
 ind_115: t4_path6 = 1 -> const_Garage_Cars >= 3
 ind_116: t4_path6 = 1 -> x_Gr_Liv_Area <= 1716
 ind_117: t4_path7 = 1 -> x_Exter_Qual >= 3
 ind_118: t4_path7 = 1 -> const_Garage_Cars >= 3
 ind_119: t4_path7 = 1 -> x_Gr_Liv_Area >= 1716
 ind_120: t5_path0 = 1 -> Overall_Qual_calc <= 7
 ind_121: t5_path0 = 1 -> Overall_Qual_calc <= 5
 ind_122: t5_path0 = 1 -> x_1st_Flr_SF <= 1155
 ind_123: t5_path1 = 1 -> Overall_Qual_calc <= 7
 ind_124: t5_path1 = 1 -> Overall_Qual_calc <= 5
 ind_125: t5_path1 = 1 -> x_1st_Flr_SF >= 1155
 ind_126: t5_path2 = 1 -> Overall_Qual_calc <= 7
 ind_127: t5_path2 = 1 -> Overall_Qual_calc >= 5
 ind_128: t5_path2 = 1 -> const_Garage_Cars <= 2
 ind_129: t5_path3 = 1 -> Overall_Qual_calc <= 7
 ind_130: t5_path3 = 1 -> Overall_Qual_calc >= 5
 ind_131: t5_path3 = 1 -> const_Garage_Cars >= 2
 ind_132: t5_path4 = 1 -> Overall_Qual_calc >= 7
 ind_133: t5_path4 = 1 -> Overall_Qual_calc <= 8
 ind_134: t5_path4 = 1 -> x_1st_Flr_SF <= 932
 ind_135: t5_path5 = 1 -> Overall_Qual_calc >= 7
 ind_136: t5_path5 = 1 -> Overall_Qual_calc <= 8
 ind_137: t5_path5 = 1 -> x_1st_Flr_SF >= 932
 ind_138: t5_path6 = 1 -> Overall_Qual_calc >= 7
 ind_139: t5_path6 = 1 -> Overall_Qual_calc >= 8
 ind_140: t5_path6 = 1 -> x_1st_Flr_SF <= 1766
 ind_141: t5_path7 = 1 -> Overall_Qual_calc >= 7
 ind_142: t5_path7 = 1 -> Overall_Qual_calc >= 8
 ind_143: t5_path7 = 1 -> x_1st_Flr_SF >= 1766
 ind_144: t6_path0 = 1 -> const_Bsmt_Qual <= 3
 ind_145: t6_path0 = 1 -> x_Garage_Cond <= 2
 ind_146: t6_path0 = 1 -> x_TotRms_AbvGrd <= 7
 ind_147: t6_path1 = 1 -> const_Bsmt_Qual <= 3
 ind_148: t6_path1 = 1 -> x_Garage_Cond <= 2
 ind_149: t6_path1 = 1 -> x_TotRms_AbvGrd >= 7
 ind_150: t6_path2 = 1 -> const_Bsmt_Qual <= 3
 ind_151: t6_path2 = 1 -> x_Garage_Cond >= 2
 ind_152: t6_path2 = 1 -> const_Fireplaces <= 1
 ind_153: t6_path3 = 1 -> const_Bsmt_Qual <= 3
 ind_154: t6_path3 = 1 -> x_Garage_Cond >= 2
 ind_155: t6_path3 = 1 -> const_Fireplaces >= 1
 ind_156: t6_path4 = 1 -> const_Bsmt_Qual >= 3
 ind_157: t6_path4 = 1 -> const_Garage_Cars <= 3
 ind_158: t6_path4 = 1 -> const_Fireplaces <= 1
 ind_159: t6_path5 = 1 -> const_Bsmt_Qual >= 3
 ind_160: t6_path5 = 1 -> const_Garage_Cars <= 3
 ind_161: t6_path5 = 1 -> const_Fireplaces >= 1
 ind_162: t6_path6 = 1 -> const_Bsmt_Qual >= 3
 ind_163: t6_path6 = 1 -> const_Garage_Cars >= 3
 ind_164: t6_path6 = 1 -> x_Kitchen_Qual <= 4
 ind_165: t6_path7 = 1 -> const_Bsmt_Qual >= 3
 ind_166: t6_path7 = 1 -> const_Garage_Cars >= 3
 ind_167: t6_path7 = 1 -> x_Kitchen_Qual >= 4
 ind_168: t7_path0 = 1 -> Overall_Qual_calc <= 7
 ind_169: t7_path0 = 1 -> Overall_Qual_calc <= 5
 ind_170: t7_path0 = 1 -> x_Gr_Liv_Area <= 864
 ind_171: t7_path1 = 1 -> Overall_Qual_calc <= 7
 ind_172: t7_path1 = 1 -> Overall_Qual_calc <= 5
 ind_173: t7_path1 = 1 -> x_Gr_Liv_Area >= 864
 ind_174: t7_path2 = 1 -> Overall_Qual_calc <= 7
 ind_175: t7_path2 = 1 -> Overall_Qual_calc >= 5
 ind_176: t7_path2 = 1 -> x_Gr_Liv_Area <= 1484
 ind_177: t7_path3 = 1 -> Overall_Qual_calc <= 7
 ind_178: t7_path3 = 1 -> Overall_Qual_calc >= 5
 ind_179: t7_path3 = 1 -> x_Gr_Liv_Area >= 1484
 ind_180: t7_path4 = 1 -> Overall_Qual_calc >= 7
 ind_181: t7_path4 = 1 -> x_1st_Flr_SF <= 1508
 ind_182: t7_path4 = 1 -> x_Gr_Liv_Area <= 1866
 ind_183: t7_path5 = 1 -> Overall_Qual_calc >= 7
 ind_184: t7_path5 = 1 -> x_1st_Flr_SF <= 1508
 ind_185: t7_path5 = 1 -> x_Gr_Liv_Area >= 1866
 ind_186: t7_path6 = 1 -> Overall_Qual_calc >= 7
 ind_187: t7_path6 = 1 -> x_1st_Flr_SF >= 1508
 ind_188: t7_path6 = 1 -> x_Gr_Liv_Area <= 1720
 ind_189: t7_path7 = 1 -> Overall_Qual_calc >= 7
 ind_190: t7_path7 = 1 -> x_1st_Flr_SF >= 1508
 ind_191: t7_path7 = 1 -> x_Gr_Liv_Area >= 1720
 ind_192: t8_path0 = 1 -> const_Bsmt_Qual <= 3
 ind_193: t8_path0 = 1 -> const_Fireplaces <= 1
 ind_194: t8_path0 = 1 -> central_air_no <= 1
 ind_195: t8_path1 = 1 -> const_Bsmt_Qual <= 3
 ind_196: t8_path1 = 1 -> const_Fireplaces <= 1
 ind_197: t8_path1 = 1 -> central_air_no >= 1
 ind_198: t8_path2 = 1 -> const_Bsmt_Qual <= 3
 ind_199: t8_path2 = 1 -> const_Fireplaces >= 1
 ind_200: t8_path2 = 1 -> x_TotRms_AbvGrd <= 8
 ind_201: t8_path3 = 1 -> const_Bsmt_Qual <= 3
 ind_202: t8_path3 = 1 -> const_Fireplaces >= 1
 ind_203: t8_path3 = 1 -> x_TotRms_AbvGrd >= 8
 ind_204: t8_path4 = 1 -> const_Bsmt_Qual >= 3
 ind_205: t8_path4 = 1 -> const_Garage_Cars <= 3
 ind_206: t8_path4 = 1 -> x_1st_Flr_SF <= 1236
 ind_207: t8_path5 = 1 -> const_Bsmt_Qual >= 3
 ind_208: t8_path5 = 1 -> const_Garage_Cars <= 3
 ind_209: t8_path5 = 1 -> x_1st_Flr_SF >= 1236
 ind_210: t8_path6 = 1 -> const_Bsmt_Qual >= 3
 ind_211: t8_path6 = 1 -> const_Garage_Cars >= 3
 ind_212: t8_path6 = 1 -> x_1st_Flr_SF <= 1800
 ind_213: t8_path7 = 1 -> const_Bsmt_Qual >= 3
 ind_214: t8_path7 = 1 -> const_Garage_Cars >= 3
 ind_215: t8_path7 = 1 -> x_1st_Flr_SF >= 1800
 ind_216: t9_path0 = 1 -> const_Year_Built <= 1981
 ind_217: t9_path0 = 1 -> x_1st_Flr_SF <= 1094
 ind_218: t9_path0 = 1 -> central_air_no <= 1
 ind_219: t9_path1 = 1 -> const_Year_Built <= 1981
 ind_220: t9_path1 = 1 -> x_1st_Flr_SF <= 1094
 ind_221: t9_path1 = 1 -> central_air_no >= 1
 ind_222: t9_path2 = 1 -> const_Year_Built <= 1981
 ind_223: t9_path2 = 1 -> x_1st_Flr_SF >= 1094
 ind_224: t9_path2 = 1 -> const_Lot_Area <= 12216
 ind_225: t9_path3 = 1 -> const_Year_Built <= 1981
 ind_226: t9_path3 = 1 -> x_1st_Flr_SF >= 1094
 ind_227: t9_path3 = 1 -> const_Lot_Area >= 12216
 ind_228: t9_path4 = 1 -> const_Year_Built >= 1981
 ind_229: t9_path4 = 1 -> x_Garage_Area <= 668
 ind_230: t9_path4 = 1 -> const_Total_Bsmt_SF <= 1268
 ind_231: t9_path5 = 1 -> const_Year_Built >= 1981
 ind_232: t9_path5 = 1 -> x_Garage_Area <= 668
 ind_233: t9_path5 = 1 -> const_Total_Bsmt_SF >= 1268
 ind_234: t9_path6 = 1 -> const_Year_Built >= 1981
 ind_235: t9_path6 = 1 -> x_Garage_Area >= 668
 ind_236: t9_path6 = 1 -> const_Lot_Area <= 11980
 ind_237: t9_path7 = 1 -> const_Year_Built >= 1981
 ind_238: t9_path7 = 1 -> x_Garage_Area >= 668
 ind_239: t9_path7 = 1 -> const_Lot_Area >= 11980
 ind_240: t10_path0 = 1 -> x_Gr_Liv_Area <= 1320
 ind_241: t10_path0 = 1 -> const_Garage_Yr_Blt <= 1953
 ind_242: t10_path0 = 1 -> x_Gr_Liv_Area <= 833
 ind_243: t10_path1 = 1 -> x_Gr_Liv_Area <= 1320
 ind_244: t10_path1 = 1 -> const_Garage_Yr_Blt <= 1953
 ind_245: t10_path1 = 1 -> x_Gr_Liv_Area >= 833
 ind_246: t10_path2 = 1 -> x_Gr_Liv_Area <= 1320
 ind_247: t10_path2 = 1 -> const_Garage_Yr_Blt >= 1953
 ind_248: t10_path2 = 1 -> x_Gr_Liv_Area <= 1124
 ind_249: t10_path3 = 1 -> x_Gr_Liv_Area <= 1320
 ind_250: t10_path3 = 1 -> const_Garage_Yr_Blt >= 1953
 ind_251: t10_path3 = 1 -> x_Gr_Liv_Area >= 1124
 ind_252: t10_path4 = 1 -> x_Gr_Liv_Area >= 1320
 ind_253: t10_path4 = 1 -> x_Exter_Qual <= 3
 ind_254: t10_path4 = 1 -> const_MS_Zoning_RL <= 1
 ind_255: t10_path5 = 1 -> x_Gr_Liv_Area >= 1320
 ind_256: t10_path5 = 1 -> x_Exter_Qual <= 3
 ind_257: t10_path5 = 1 -> const_MS_Zoning_RL >= 1
 ind_258: t10_path6 = 1 -> x_Gr_Liv_Area >= 1320
 ind_259: t10_path6 = 1 -> x_Exter_Qual >= 3
 ind_260: t10_path6 = 1 -> const_Bsmt_Qual <= 4
 ind_261: t10_path7 = 1 -> x_Gr_Liv_Area >= 1320
 ind_262: t10_path7 = 1 -> x_Exter_Qual >= 3
 ind_263: t10_path7 = 1 -> const_Bsmt_Qual >= 4
 ind_264: t11_path0 = 1 -> Overall_Qual_calc <= 7
 ind_265: t11_path0 = 1 -> Overall_Qual_calc <= 5
 ind_266: t11_path0 = 1 -> const_Total_Bsmt_SF <= 912
 ind_267: t11_path1 = 1 -> Overall_Qual_calc <= 7
 ind_268: t11_path1 = 1 -> Overall_Qual_calc <= 5
 ind_269: t11_path1 = 1 -> const_Total_Bsmt_SF >= 912
 ind_270: t11_path2 = 1 -> Overall_Qual_calc <= 7
 ind_271: t11_path2 = 1 -> Overall_Qual_calc >= 5
 ind_272: t11_path2 = 1 -> x_Gr_Liv_Area <= 1411
 ind_273: t11_path3 = 1 -> Overall_Qual_calc <= 7
 ind_274: t11_path3 = 1 -> Overall_Qual_calc >= 5
 ind_275: t11_path3 = 1 -> x_Gr_Liv_Area >= 1411
 ind_276: t11_path4 = 1 -> Overall_Qual_calc >= 7
 ind_277: t11_path4 = 1 -> Overall_Qual_calc <= 8
 ind_278: t11_path4 = 1 -> x_Gr_Liv_Area <= 1726
 ind_279: t11_path5 = 1 -> Overall_Qual_calc >= 7
 ind_280: t11_path5 = 1 -> Overall_Qual_calc <= 8
 ind_281: t11_path5 = 1 -> x_Gr_Liv_Area >= 1726
 ind_282: t11_path6 = 1 -> Overall_Qual_calc >= 7
 ind_283: t11_path6 = 1 -> Overall_Qual_calc >= 8
 ind_284: t11_path6 = 1 -> x_Gr_Liv_Area <= 1670
 ind_285: t11_path7 = 1 -> Overall_Qual_calc >= 7
 ind_286: t11_path7 = 1 -> Overall_Qual_calc >= 8
 ind_287: t11_path7 = 1 -> x_Gr_Liv_Area >= 1670
 ind_288: t12_path0 = 1 -> Overall_Qual_calc <= 7
 ind_289: t12_path0 = 1 -> Overall_Qual_calc <= 5
 ind_290: t12_path0 = 1 -> x_Gr_Liv_Area <= 1155
 ind_291: t12_path1 = 1 -> Overall_Qual_calc <= 7
 ind_292: t12_path1 = 1 -> Overall_Qual_calc <= 5
 ind_293: t12_path1 = 1 -> x_Gr_Liv_Area >= 1155
 ind_294: t12_path2 = 1 -> Overall_Qual_calc <= 7
 ind_295: t12_path2 = 1 -> Overall_Qual_calc >= 5
 ind_296: t12_path2 = 1 -> x_Garage_Area <= 384
 ind_297: t12_path3 = 1 -> Overall_Qual_calc <= 7
 ind_298: t12_path3 = 1 -> Overall_Qual_calc >= 5
 ind_299: t12_path3 = 1 -> x_Garage_Area >= 384
 ind_300: t12_path4 = 1 -> Overall_Qual_calc >= 7
 ind_301: t12_path4 = 1 -> const_Bsmt_Qual <= 4
 ind_302: t12_path4 = 1 -> x_Gr_Liv_Area <= 2134
 ind_303: t12_path5 = 1 -> Overall_Qual_calc >= 7
 ind_304: t12_path5 = 1 -> const_Bsmt_Qual <= 4
 ind_305: t12_path5 = 1 -> x_Gr_Liv_Area >= 2134
 ind_306: t12_path6 = 1 -> Overall_Qual_calc >= 7
 ind_307: t12_path6 = 1 -> const_Bsmt_Qual >= 4
 ind_308: t12_path6 = 1 -> Overall_Qual_calc <= 9
 ind_309: t12_path7 = 1 -> Overall_Qual_calc >= 7
 ind_310: t12_path7 = 1 -> const_Bsmt_Qual >= 4
 ind_311: t12_path7 = 1 -> Overall_Qual_calc >= 9
 ind_312: t13_path0 = 1 -> x_Kitchen_Qual <= 3
 ind_313: t13_path0 = 1 -> const_Garage_Cars <= 2
 ind_314: t13_path0 = 1 -> const_Overall_Cond <= 4
 ind_315: t13_path1 = 1 -> x_Kitchen_Qual <= 3
 ind_316: t13_path1 = 1 -> const_Garage_Cars <= 2
 ind_317: t13_path1 = 1 -> const_Overall_Cond >= 4
 ind_318: t13_path2 = 1 -> x_Kitchen_Qual <= 3
 ind_319: t13_path2 = 1 -> const_Garage_Cars >= 2
 ind_320: t13_path2 = 1 -> x_Gr_Liv_Area <= 1779
 ind_321: t13_path3 = 1 -> x_Kitchen_Qual <= 3
 ind_322: t13_path3 = 1 -> const_Garage_Cars >= 2
 ind_323: t13_path3 = 1 -> x_Gr_Liv_Area >= 1779
 ind_324: t13_path4 = 1 -> x_Kitchen_Qual >= 3
 ind_325: t13_path4 = 1 -> x_Gr_Liv_Area <= 1726
 ind_326: t13_path4 = 1 -> x_1st_Flr_SF <= 1282
 ind_327: t13_path5 = 1 -> x_Kitchen_Qual >= 3
 ind_328: t13_path5 = 1 -> x_Gr_Liv_Area <= 1726
 ind_329: t13_path5 = 1 -> x_1st_Flr_SF >= 1282
 ind_330: t13_path6 = 1 -> x_Kitchen_Qual >= 3
 ind_331: t13_path6 = 1 -> x_Gr_Liv_Area >= 1726
 ind_332: t13_path6 = 1 -> const_Garage_Cars <= 3
 ind_333: t13_path7 = 1 -> x_Kitchen_Qual >= 3
 ind_334: t13_path7 = 1 -> x_Gr_Liv_Area >= 1726
 ind_335: t13_path7 = 1 -> const_Garage_Cars >= 3
 ind_336: t14_path0 = 1 -> x_Exter_Qual <= 3
 ind_337: t14_path0 = 1 -> central_air_yes <= 1
 ind_338: t14_path0 = 1 -> x_Gr_Liv_Area <= 876
 ind_339: t14_path1 = 1 -> x_Exter_Qual <= 3
 ind_340: t14_path1 = 1 -> central_air_yes <= 1
 ind_341: t14_path1 = 1 -> x_Gr_Liv_Area >= 876
 ind_342: t14_path2 = 1 -> x_Exter_Qual <= 3
 ind_343: t14_path2 = 1 -> central_air_yes >= 1
 ind_344: t14_path2 = 1 -> x_Gr_Liv_Area <= 1416
 ind_345: t14_path3 = 1 -> x_Exter_Qual <= 3
 ind_346: t14_path3 = 1 -> central_air_yes >= 1
 ind_347: t14_path3 = 1 -> x_Gr_Liv_Area >= 1416
 ind_348: t14_path4 = 1 -> x_Exter_Qual >= 3
 ind_349: t14_path4 = 1 -> x_Gr_Liv_Area <= 1740
 ind_350: t14_path4 = 1 -> const_Total_Bsmt_SF <= 1268
 ind_351: t14_path5 = 1 -> x_Exter_Qual >= 3
 ind_352: t14_path5 = 1 -> x_Gr_Liv_Area <= 1740
 ind_353: t14_path5 = 1 -> const_Total_Bsmt_SF >= 1268
 ind_354: t14_path6 = 1 -> x_Exter_Qual >= 3
 ind_355: t14_path6 = 1 -> x_Gr_Liv_Area >= 1740
 ind_356: t14_path6 = 1 -> const_Total_Bsmt_SF <= 1206
 ind_357: t14_path7 = 1 -> x_Exter_Qual >= 3
 ind_358: t14_path7 = 1 -> x_Gr_Liv_Area >= 1740
 ind_359: t14_path7 = 1 -> const_Total_Bsmt_SF >= 1206
 ind_360: t15_path0 = 1 -> x_Gr_Liv_Area <= 1411
 ind_361: t15_path0 = 1 -> const_Total_Bsmt_SF <= 970
 ind_362: t15_path0 = 1 -> central_air_no <= 1
 ind_363: t15_path1 = 1 -> x_Gr_Liv_Area <= 1411
 ind_364: t15_path1 = 1 -> const_Total_Bsmt_SF <= 970
 ind_365: t15_path1 = 1 -> central_air_no >= 1
 ind_366: t15_path2 = 1 -> x_Gr_Liv_Area <= 1411
 ind_367: t15_path2 = 1 -> const_Total_Bsmt_SF >= 970
 ind_368: t15_path2 = 1 -> const_Garage_Type_Attchd <= 1
 ind_369: t15_path3 = 1 -> x_Gr_Liv_Area <= 1411
 ind_370: t15_path3 = 1 -> const_Total_Bsmt_SF >= 970
 ind_371: t15_path3 = 1 -> const_Garage_Type_Attchd >= 1
 ind_372: t15_path4 = 1 -> x_Gr_Liv_Area >= 1411
 ind_373: t15_path4 = 1 -> const_Bsmt_Qual <= 4
 ind_374: t15_path4 = 1 -> const_Garage_Finish_Unf <= 1
 ind_375: t15_path5 = 1 -> x_Gr_Liv_Area >= 1411
 ind_376: t15_path5 = 1 -> const_Bsmt_Qual <= 4
 ind_377: t15_path5 = 1 -> const_Garage_Finish_Unf >= 1
 ind_378: t15_path6 = 1 -> x_Gr_Liv_Area >= 1411
 ind_379: t15_path6 = 1 -> const_Bsmt_Qual >= 4
 ind_380: t15_path6 = 1 -> x_Gr_Liv_Area <= 1995
 ind_381: t15_path7 = 1 -> x_Gr_Liv_Area >= 1411
 ind_382: t15_path7 = 1 -> const_Bsmt_Qual >= 4
 ind_383: t15_path7 = 1 -> x_Gr_Liv_Area >= 1995
 ind_384: t16_path0 = 1 -> Overall_Qual_calc <= 7
 ind_385: t16_path0 = 1 -> Overall_Qual_calc <= 5
 ind_386: t16_path0 = 1 -> Overall_Qual_calc <= 3
 ind_387: t16_path1 = 1 -> Overall_Qual_calc <= 7
 ind_388: t16_path1 = 1 -> Overall_Qual_calc <= 5
 ind_389: t16_path1 = 1 -> Overall_Qual_calc >= 3
 ind_390: t16_path2 = 1 -> Overall_Qual_calc <= 7
 ind_391: t16_path2 = 1 -> Overall_Qual_calc >= 5
 ind_392: t16_path2 = 1 -> const_Lot_Area <= 10800
 ind_393: t16_path3 = 1 -> Overall_Qual_calc <= 7
 ind_394: t16_path3 = 1 -> Overall_Qual_calc >= 5
 ind_395: t16_path3 = 1 -> const_Lot_Area >= 10800
 ind_396: t16_path4 = 1 -> Overall_Qual_calc >= 7
 ind_397: t16_path4 = 1 -> const_Bsmt_Qual <= 4
 ind_398: t16_path4 = 1 -> x_Gr_Liv_Area <= 1795
 ind_399: t16_path5 = 1 -> Overall_Qual_calc >= 7
 ind_400: t16_path5 = 1 -> const_Bsmt_Qual <= 4
 ind_401: t16_path5 = 1 -> x_Gr_Liv_Area >= 1795
 ind_402: t16_path6 = 1 -> Overall_Qual_calc >= 7
 ind_403: t16_path6 = 1 -> const_Bsmt_Qual >= 4
 ind_404: t16_path6 = 1 -> x_Gr_Liv_Area <= 2198
 ind_405: t16_path7 = 1 -> Overall_Qual_calc >= 7
 ind_406: t16_path7 = 1 -> const_Bsmt_Qual >= 4
 ind_407: t16_path7 = 1 -> x_Gr_Liv_Area >= 2198
 ind_408: t17_path0 = 1 -> Overall_Qual_calc <= 7
 ind_409: t17_path0 = 1 -> Overall_Qual_calc <= 5
 ind_410: t17_path0 = 1 -> Overall_Qual_calc <= 3
 ind_411: t17_path1 = 1 -> Overall_Qual_calc <= 7
 ind_412: t17_path1 = 1 -> Overall_Qual_calc <= 5
 ind_413: t17_path1 = 1 -> Overall_Qual_calc >= 3
 ind_414: t17_path2 = 1 -> Overall_Qual_calc <= 7
 ind_415: t17_path2 = 1 -> Overall_Qual_calc >= 5
 ind_416: t17_path2 = 1 -> x_Fireplace_Qu <= 0
 ind_417: t17_path3 = 1 -> Overall_Qual_calc <= 7
 ind_418: t17_path3 = 1 -> Overall_Qual_calc >= 5
 ind_419: t17_path3 = 1 -> x_Fireplace_Qu >= 0
 ind_420: t17_path4 = 1 -> Overall_Qual_calc >= 7
 ind_421: t17_path4 = 1 -> x_Kitchen_Qual <= 4
 ind_422: t17_path4 = 1 -> x_1st_Flr_SF <= 1064
 ind_423: t17_path5 = 1 -> Overall_Qual_calc >= 7
 ind_424: t17_path5 = 1 -> x_Kitchen_Qual <= 4
 ind_425: t17_path5 = 1 -> x_1st_Flr_SF >= 1064
 ind_426: t17_path6 = 1 -> Overall_Qual_calc >= 7
 ind_427: t17_path6 = 1 -> x_Kitchen_Qual >= 4
 ind_428: t18_path0 = 1 -> x_Exter_Qual <= 3
 ind_429: t18_path0 = 1 -> central_air_no <= 1
 ind_430: t18_path0 = 1 -> const_Total_Bsmt_SF <= 924
 ind_431: t18_path1 = 1 -> x_Exter_Qual <= 3
 ind_432: t18_path1 = 1 -> central_air_no <= 1
 ind_433: t18_path1 = 1 -> const_Total_Bsmt_SF >= 924
 ind_434: t18_path2 = 1 -> x_Exter_Qual <= 3
 ind_435: t18_path2 = 1 -> central_air_no >= 1
 ind_436: t18_path2 = 1 -> const_Overall_Cond <= 6
 ind_437: t18_path3 = 1 -> x_Exter_Qual <= 3
 ind_438: t18_path3 = 1 -> central_air_no >= 1
 ind_439: t18_path3 = 1 -> const_Overall_Cond >= 6
 ind_440: t18_path4 = 1 -> x_Exter_Qual >= 3
 ind_441: t18_path4 = 1 -> const_Total_Bsmt_SF <= 1389
 ind_442: t18_path4 = 1 -> const_2nd_Flr_SF <= 828
 ind_443: t18_path5 = 1 -> x_Exter_Qual >= 3
 ind_444: t18_path5 = 1 -> const_Total_Bsmt_SF <= 1389
 ind_445: t18_path5 = 1 -> const_2nd_Flr_SF >= 828
 ind_446: t18_path6 = 1 -> x_Exter_Qual >= 3
 ind_447: t18_path6 = 1 -> const_Total_Bsmt_SF >= 1389
 ind_448: t18_path6 = 1 -> const_Lot_Area <= 12099
 ind_449: t18_path7 = 1 -> x_Exter_Qual >= 3
 ind_450: t18_path7 = 1 -> const_Total_Bsmt_SF >= 1389
 ind_451: t18_path7 = 1 -> const_Lot_Area >= 12099
 ind_452: t19_path0 = 1 -> const_Year_Built <= 1981
 ind_453: t19_path0 = 1 -> x_Fireplace_Qu <= 0
 ind_454: t19_path0 = 1 -> x_Gr_Liv_Area <= 810
 ind_455: t19_path1 = 1 -> const_Year_Built <= 1981
 ind_456: t19_path1 = 1 -> x_Fireplace_Qu <= 0
 ind_457: t19_path1 = 1 -> x_Gr_Liv_Area >= 810
 ind_458: t19_path2 = 1 -> const_Year_Built <= 1981
 ind_459: t19_path2 = 1 -> x_Fireplace_Qu >= 0
 ind_460: t19_path2 = 1 -> x_Gr_Liv_Area <= 1779
 ind_461: t19_path3 = 1 -> const_Year_Built <= 1981
 ind_462: t19_path3 = 1 -> x_Fireplace_Qu >= 0
 ind_463: t19_path3 = 1 -> x_Gr_Liv_Area >= 1779
 ind_464: t19_path4 = 1 -> const_Year_Built >= 1981
 ind_465: t19_path4 = 1 -> x_Gr_Liv_Area <= 1961
 ind_466: t19_path4 = 1 -> const_Total_Bsmt_SF <= 1494
 ind_467: t19_path5 = 1 -> const_Year_Built >= 1981
 ind_468: t19_path5 = 1 -> x_Gr_Liv_Area <= 1961
 ind_469: t19_path5 = 1 -> const_Total_Bsmt_SF >= 1494
 ind_470: t19_path6 = 1 -> const_Year_Built >= 1981
 ind_471: t19_path6 = 1 -> x_Gr_Liv_Area >= 1961
 ind_472: t19_path6 = 1 -> const_Total_Bsmt_SF <= 1187
 ind_473: t19_path7 = 1 -> const_Year_Built >= 1981
 ind_474: t19_path7 = 1 -> x_Gr_Liv_Area >= 1961
 ind_475: t19_path7 = 1 -> const_Total_Bsmt_SF >= 1187
 ind_476: t20_path0 = 1 -> Overall_Qual_calc <= 6
 ind_477: t20_path0 = 1 -> const_MS_Zoning_RL <= 1
 ind_478: t20_path0 = 1 -> x_TotRms_AbvGrd <= 5
 ind_479: t20_path1 = 1 -> Overall_Qual_calc <= 6
 ind_480: t20_path1 = 1 -> const_MS_Zoning_RL <= 1
 ind_481: t20_path1 = 1 -> x_TotRms_AbvGrd >= 5
 ind_482: t20_path2 = 1 -> Overall_Qual_calc <= 6
 ind_483: t20_path2 = 1 -> const_MS_Zoning_RL >= 1
 ind_484: t20_path2 = 1 -> x_BsmtFin_SF_1 <= 557
 ind_485: t20_path3 = 1 -> Overall_Qual_calc <= 6
 ind_486: t20_path3 = 1 -> const_MS_Zoning_RL >= 1
 ind_487: t20_path3 = 1 -> x_BsmtFin_SF_1 >= 557
 ind_488: t20_path4 = 1 -> Overall_Qual_calc >= 6
 ind_489: t20_path4 = 1 -> Overall_Qual_calc <= 8
 ind_490: t20_path4 = 1 -> const_Lot_Area <= 7500
 ind_491: t20_path5 = 1 -> Overall_Qual_calc >= 6
 ind_492: t20_path5 = 1 -> Overall_Qual_calc <= 8
 ind_493: t20_path5 = 1 -> const_Lot_Area >= 7500
 ind_494: t20_path6 = 1 -> Overall_Qual_calc >= 6
 ind_495: t20_path6 = 1 -> Overall_Qual_calc >= 8
 ind_496: t20_path6 = 1 -> x_Wood_Deck_SF <= 240
 ind_497: t20_path7 = 1 -> Overall_Qual_calc >= 6
 ind_498: t20_path7 = 1 -> Overall_Qual_calc >= 8
 ind_499: t20_path7 = 1 -> x_Wood_Deck_SF >= 240
 ind_500: t21_path0 = 1 -> Overall_Qual_calc <= 7
 ind_501: t21_path0 = 1 -> const_MS_Zoning_RL <= 1
 ind_502: t21_path0 = 1 -> const_Overall_Cond <= 5
 ind_503: t21_path1 = 1 -> Overall_Qual_calc <= 7
 ind_504: t21_path1 = 1 -> const_MS_Zoning_RL <= 1
 ind_505: t21_path1 = 1 -> const_Overall_Cond >= 5
 ind_506: t21_path2 = 1 -> Overall_Qual_calc <= 7
 ind_507: t21_path2 = 1 -> const_MS_Zoning_RL >= 1
 ind_508: t21_path2 = 1 -> Overall_Qual_calc <= 6
 ind_509: t21_path3 = 1 -> Overall_Qual_calc <= 7
 ind_510: t21_path3 = 1 -> const_MS_Zoning_RL >= 1
 ind_511: t21_path3 = 1 -> Overall_Qual_calc >= 6
 ind_512: t21_path4 = 1 -> Overall_Qual_calc >= 7
 ind_513: t21_path4 = 1 -> x_BsmtFin_SF_1 <= 1097
 ind_514: t21_path4 = 1 -> x_Garage_Area <= 546
 ind_515: t21_path5 = 1 -> Overall_Qual_calc >= 7
 ind_516: t21_path5 = 1 -> x_BsmtFin_SF_1 <= 1097
 ind_517: t21_path5 = 1 -> x_Garage_Area >= 546
 ind_518: t21_path6 = 1 -> Overall_Qual_calc >= 7
 ind_519: t21_path6 = 1 -> x_BsmtFin_SF_1 >= 1097
 ind_520: t21_path6 = 1 -> x_Kitchen_Qual <= 4
 ind_521: t21_path7 = 1 -> Overall_Qual_calc >= 7
 ind_522: t21_path7 = 1 -> x_BsmtFin_SF_1 >= 1097
 ind_523: t21_path7 = 1 -> x_Kitchen_Qual >= 4
 ind_524: t22_path0 = 1 -> x_Gr_Liv_Area <= 1491
 ind_525: t22_path0 = 1 -> x_1st_Flr_SF <= 1025
 ind_526: t22_path0 = 1 -> const_Year_Remod_Add <= 1951
 ind_527: t22_path1 = 1 -> x_Gr_Liv_Area <= 1491
 ind_528: t22_path1 = 1 -> x_1st_Flr_SF <= 1025
 ind_529: t22_path1 = 1 -> const_Year_Remod_Add >= 1951
 ind_530: t22_path2 = 1 -> x_Gr_Liv_Area <= 1491
 ind_531: t22_path2 = 1 -> x_1st_Flr_SF >= 1025
 ind_532: t22_path2 = 1 -> const_Garage_Yr_Blt <= 1953
 ind_533: t22_path3 = 1 -> x_Gr_Liv_Area <= 1491
 ind_534: t22_path3 = 1 -> x_1st_Flr_SF >= 1025
 ind_535: t22_path3 = 1 -> const_Garage_Yr_Blt >= 1953
 ind_536: t22_path4 = 1 -> x_Gr_Liv_Area >= 1491
 ind_537: t22_path4 = 1 -> const_Year_Remod_Add <= 1983
 ind_538: t22_path4 = 1 -> const_Fireplaces <= 2
 ind_539: t22_path5 = 1 -> x_Gr_Liv_Area >= 1491
 ind_540: t22_path5 = 1 -> const_Year_Remod_Add <= 1983
 ind_541: t22_path5 = 1 -> const_Fireplaces >= 2
 ind_542: t22_path6 = 1 -> x_Gr_Liv_Area >= 1491
 ind_543: t22_path6 = 1 -> const_Year_Remod_Add >= 1983
 ind_544: t22_path6 = 1 -> x_Gr_Liv_Area <= 1902
 ind_545: t22_path7 = 1 -> x_Gr_Liv_Area >= 1491
 ind_546: t22_path7 = 1 -> const_Year_Remod_Add >= 1983
 ind_547: t22_path7 = 1 -> x_Gr_Liv_Area >= 1902
 ind_548: t23_path0 = 1 -> Overall_Qual_calc <= 6
 ind_549: t23_path0 = 1 -> Overall_Qual_calc <= 4
 ind_550: t23_path0 = 1 -> const_Year_Built <= 1950
 ind_551: t23_path1 = 1 -> Overall_Qual_calc <= 6
 ind_552: t23_path1 = 1 -> Overall_Qual_calc <= 4
 ind_553: t23_path1 = 1 -> const_Year_Built >= 1950
 ind_554: t23_path2 = 1 -> Overall_Qual_calc <= 6
 ind_555: t23_path2 = 1 -> Overall_Qual_calc >= 4
 ind_556: t23_path2 = 1 -> const_MS_Zoning_RL <= 1
 ind_557: t23_path3 = 1 -> Overall_Qual_calc <= 6
 ind_558: t23_path3 = 1 -> Overall_Qual_calc >= 4
 ind_559: t23_path3 = 1 -> const_MS_Zoning_RL >= 1
 ind_560: t23_path4 = 1 -> Overall_Qual_calc >= 6
 ind_561: t23_path4 = 1 -> const_Garage_Cars <= 3
 ind_562: t23_path4 = 1 -> const_Lot_Area <= 13262
 ind_563: t23_path5 = 1 -> Overall_Qual_calc >= 6
 ind_564: t23_path5 = 1 -> const_Garage_Cars <= 3
 ind_565: t23_path5 = 1 -> const_Lot_Area >= 13262
 ind_566: t23_path6 = 1 -> Overall_Qual_calc >= 6
 ind_567: t23_path6 = 1 -> const_Garage_Cars >= 3
 ind_568: t23_path6 = 1 -> Overall_Qual_calc <= 9
 ind_569: t23_path7 = 1 -> Overall_Qual_calc >= 6
 ind_570: t23_path7 = 1 -> const_Garage_Cars >= 3
 ind_571: t23_path7 = 1 -> Overall_Qual_calc >= 9
 ind_572: t24_path0 = 1 -> Overall_Qual_calc <= 7
 ind_573: t24_path0 = 1 -> const_Garage_Cars <= 2
 ind_574: t24_path0 = 1 -> x_1st_Flr_SF <= 894
 ind_575: t24_path1 = 1 -> Overall_Qual_calc <= 7
 ind_576: t24_path1 = 1 -> const_Garage_Cars <= 2
 ind_577: t24_path1 = 1 -> x_1st_Flr_SF >= 894
 ind_578: t24_path2 = 1 -> Overall_Qual_calc <= 7
 ind_579: t24_path2 = 1 -> const_Garage_Cars >= 2
 ind_580: t24_path2 = 1 -> const_Bsmt_Qual <= 2
 ind_581: t24_path3 = 1 -> Overall_Qual_calc <= 7
 ind_582: t24_path3 = 1 -> const_Garage_Cars >= 2
 ind_583: t24_path3 = 1 -> const_Bsmt_Qual >= 2
 ind_584: t24_path4 = 1 -> Overall_Qual_calc >= 7
 ind_585: t24_path4 = 1 -> Overall_Qual_calc <= 8
 ind_586: t24_path4 = 1 -> const_Lot_Area <= 7500
 ind_587: t24_path5 = 1 -> Overall_Qual_calc >= 7
 ind_588: t24_path5 = 1 -> Overall_Qual_calc <= 8
 ind_589: t24_path5 = 1 -> const_Lot_Area >= 7500
 ind_590: t24_path6 = 1 -> Overall_Qual_calc >= 7
 ind_591: t24_path6 = 1 -> Overall_Qual_calc >= 8
 ind_592: t24_path6 = 1 -> const_Lot_Area <= 12144
 ind_593: t24_path7 = 1 -> Overall_Qual_calc >= 7
 ind_594: t24_path7 = 1 -> Overall_Qual_calc >= 8
 ind_595: t24_path7 = 1 -> const_Lot_Area >= 12144
 ind_596: t25_path0 = 1 -> x_Exter_Qual <= 3
 ind_597: t25_path0 = 1 -> x_Gr_Liv_Area <= 1024
 ind_598: t25_path0 = 1 -> const_Total_Bsmt_SF <= 761
 ind_599: t25_path1 = 1 -> x_Exter_Qual <= 3
 ind_600: t25_path1 = 1 -> x_Gr_Liv_Area <= 1024
 ind_601: t25_path1 = 1 -> const_Total_Bsmt_SF >= 761
 ind_602: t25_path2 = 1 -> x_Exter_Qual <= 3
 ind_603: t25_path2 = 1 -> x_Gr_Liv_Area >= 1024
 ind_604: t25_path2 = 1 -> x_BsmtFin_SF_1 <= 138
 ind_605: t25_path3 = 1 -> x_Exter_Qual <= 3
 ind_606: t25_path3 = 1 -> x_Gr_Liv_Area >= 1024
 ind_607: t25_path3 = 1 -> x_BsmtFin_SF_1 >= 138
 ind_608: t25_path4 = 1 -> x_Exter_Qual >= 3
 ind_609: t25_path4 = 1 -> x_Gr_Liv_Area <= 1961
 ind_610: t25_path4 = 1 -> x_BsmtFin_SF_1 <= 748
 ind_611: t25_path5 = 1 -> x_Exter_Qual >= 3
 ind_612: t25_path5 = 1 -> x_Gr_Liv_Area <= 1961
 ind_613: t25_path5 = 1 -> x_BsmtFin_SF_1 >= 748
 ind_614: t25_path6 = 1 -> x_Exter_Qual >= 3
 ind_615: t25_path6 = 1 -> x_Gr_Liv_Area >= 1961
 ind_616: t25_path6 = 1 -> const_Total_Bsmt_SF <= 1055
 ind_617: t25_path7 = 1 -> x_Exter_Qual >= 3
 ind_618: t25_path7 = 1 -> x_Gr_Liv_Area >= 1961
 ind_619: t25_path7 = 1 -> const_Total_Bsmt_SF >= 1055
 ind_620: t26_path0 = 1 -> Overall_Qual_calc <= 6
 ind_621: t26_path0 = 1 -> x_paved_drive_is_N <= 1
 ind_622: t26_path0 = 1 -> x_1st_Flr_SF <= 1050
 ind_623: t26_path1 = 1 -> Overall_Qual_calc <= 6
 ind_624: t26_path1 = 1 -> x_paved_drive_is_N <= 1
 ind_625: t26_path1 = 1 -> x_1st_Flr_SF >= 1050
 ind_626: t26_path2 = 1 -> Overall_Qual_calc <= 6
 ind_627: t26_path2 = 1 -> x_paved_drive_is_N >= 1
 ind_628: t26_path2 = 1 -> x_Gr_Liv_Area <= 960
 ind_629: t26_path3 = 1 -> Overall_Qual_calc <= 6
 ind_630: t26_path3 = 1 -> x_paved_drive_is_N >= 1
 ind_631: t26_path3 = 1 -> x_Gr_Liv_Area >= 960
 ind_632: t26_path4 = 1 -> Overall_Qual_calc >= 6
 ind_633: t26_path4 = 1 -> Overall_Qual_calc <= 8
 ind_634: t26_path4 = 1 -> x_Gr_Liv_Area <= 1488
 ind_635: t26_path5 = 1 -> Overall_Qual_calc >= 6
 ind_636: t26_path5 = 1 -> Overall_Qual_calc <= 8
 ind_637: t26_path5 = 1 -> x_Gr_Liv_Area >= 1488
 ind_638: t26_path6 = 1 -> Overall_Qual_calc >= 6
 ind_639: t26_path6 = 1 -> Overall_Qual_calc >= 8
 ind_640: t26_path6 = 1 -> x_Gr_Liv_Area <= 2224
 ind_641: t26_path7 = 1 -> Overall_Qual_calc >= 6
 ind_642: t26_path7 = 1 -> Overall_Qual_calc >= 8
 ind_643: t26_path7 = 1 -> x_Gr_Liv_Area >= 2224
 ind_644: t27_path0 = 1 -> x_Exter_Qual <= 3
 ind_645: t27_path0 = 1 -> central_air_no <= 1
 ind_646: t27_path0 = 1 -> const_Lot_Area <= 9116
 ind_647: t27_path1 = 1 -> x_Exter_Qual <= 3
 ind_648: t27_path1 = 1 -> central_air_no <= 1
 ind_649: t27_path1 = 1 -> const_Lot_Area >= 9116
 ind_650: t27_path2 = 1 -> x_Exter_Qual <= 3
 ind_651: t27_path2 = 1 -> central_air_no >= 1
 ind_652: t27_path2 = 1 -> const_Overall_Cond <= 6
 ind_653: t27_path3 = 1 -> x_Exter_Qual <= 3
 ind_654: t27_path3 = 1 -> central_air_no >= 1
 ind_655: t27_path3 = 1 -> const_Overall_Cond >= 6
 ind_656: t27_path4 = 1 -> x_Exter_Qual >= 3
 ind_657: t27_path4 = 1 -> x_Fireplace_Qu <= 1
 ind_658: t27_path4 = 1 -> x_Garage_Area <= 497
 ind_659: t27_path5 = 1 -> x_Exter_Qual >= 3
 ind_660: t27_path5 = 1 -> x_Fireplace_Qu <= 1
 ind_661: t27_path5 = 1 -> x_Garage_Area >= 497
 ind_662: t27_path6 = 1 -> x_Exter_Qual >= 3
 ind_663: t27_path6 = 1 -> x_Fireplace_Qu >= 1
 ind_664: t27_path6 = 1 -> x_Garage_Area <= 706
 ind_665: t27_path7 = 1 -> x_Exter_Qual >= 3
 ind_666: t27_path7 = 1 -> x_Fireplace_Qu >= 1
 ind_667: t27_path7 = 1 -> x_Garage_Area >= 706
 ind_668: t28_path0 = 1 -> x_Gr_Liv_Area <= 1416
 ind_669: t28_path0 = 1 -> const_Year_Built <= 1953
 ind_670: t28_path0 = 1 -> x_Gr_Liv_Area <= 833
 ind_671: t28_path1 = 1 -> x_Gr_Liv_Area <= 1416
 ind_672: t28_path1 = 1 -> const_Year_Built <= 1953
 ind_673: t28_path1 = 1 -> x_Gr_Liv_Area >= 833
 ind_674: t28_path2 = 1 -> x_Gr_Liv_Area <= 1416
 ind_675: t28_path2 = 1 -> const_Year_Built >= 1953
 ind_676: t28_path2 = 1 -> x_1st_Flr_SF <= 951
 ind_677: t28_path3 = 1 -> x_Gr_Liv_Area <= 1416
 ind_678: t28_path3 = 1 -> const_Year_Built >= 1953
 ind_679: t28_path3 = 1 -> x_1st_Flr_SF >= 951
 ind_680: t28_path4 = 1 -> x_Gr_Liv_Area >= 1416
 ind_681: t28_path4 = 1 -> x_Gr_Liv_Area <= 2224
 ind_682: t28_path4 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_683: t28_path5 = 1 -> x_Gr_Liv_Area >= 1416
 ind_684: t28_path5 = 1 -> x_Gr_Liv_Area <= 2224
 ind_685: t28_path5 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_686: t28_path6 = 1 -> x_Gr_Liv_Area >= 1416
 ind_687: t28_path6 = 1 -> x_Gr_Liv_Area >= 2224
 ind_688: t28_path6 = 1 -> const_Lot_Area <= 11700
 ind_689: t28_path7 = 1 -> x_Gr_Liv_Area >= 1416
 ind_690: t28_path7 = 1 -> x_Gr_Liv_Area >= 2224
 ind_691: t28_path7 = 1 -> const_Lot_Area >= 11700
 ind_692: t29_path0 = 1 -> Overall_Qual_calc <= 7
 ind_693: t29_path0 = 1 -> Overall_Qual_calc <= 4
 ind_694: t29_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_695: t29_path1 = 1 -> Overall_Qual_calc <= 7
 ind_696: t29_path1 = 1 -> Overall_Qual_calc <= 4
 ind_697: t29_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_698: t29_path2 = 1 -> Overall_Qual_calc <= 7
 ind_699: t29_path2 = 1 -> Overall_Qual_calc >= 4
 ind_700: t29_path2 = 1 -> x_BsmtFin_SF_1 <= 351
 ind_701: t29_path3 = 1 -> Overall_Qual_calc <= 7
 ind_702: t29_path3 = 1 -> Overall_Qual_calc >= 4
 ind_703: t29_path3 = 1 -> x_BsmtFin_SF_1 >= 351
 ind_704: t29_path4 = 1 -> Overall_Qual_calc >= 7
 ind_705: t29_path4 = 1 -> const_Total_Bsmt_SF <= 1475
 ind_706: t29_path4 = 1 -> const_Lot_Area <= 11980
 ind_707: t29_path5 = 1 -> Overall_Qual_calc >= 7
 ind_708: t29_path5 = 1 -> const_Total_Bsmt_SF <= 1475
 ind_709: t29_path5 = 1 -> const_Lot_Area >= 11980
 ind_710: t29_path6 = 1 -> Overall_Qual_calc >= 7
 ind_711: t29_path6 = 1 -> const_Total_Bsmt_SF >= 1475
 ind_712: t29_path6 = 1 -> x_Fireplace_Qu <= 2
 ind_713: t29_path7 = 1 -> Overall_Qual_calc >= 7
 ind_714: t29_path7 = 1 -> const_Total_Bsmt_SF >= 1475
 ind_715: t29_path7 = 1 -> x_Fireplace_Qu >= 2
 ind_716: t30_path0 = 1 -> Overall_Qual_calc <= 6
 ind_717: t30_path0 = 1 -> x_Bsmt_Cond <= 2
 ind_718: t30_path0 = 1 -> x_Garage_Area <= 618
 ind_719: t30_path1 = 1 -> Overall_Qual_calc <= 6
 ind_720: t30_path1 = 1 -> x_Bsmt_Cond <= 2
 ind_721: t30_path1 = 1 -> x_Garage_Area >= 618
 ind_722: t30_path2 = 1 -> Overall_Qual_calc <= 6
 ind_723: t30_path2 = 1 -> x_Bsmt_Cond >= 2
 ind_724: t30_path2 = 1 -> const_MS_Zoning_RL <= 1
 ind_725: t30_path3 = 1 -> Overall_Qual_calc <= 6
 ind_726: t30_path3 = 1 -> x_Bsmt_Cond >= 2
 ind_727: t30_path3 = 1 -> const_MS_Zoning_RL >= 1
 ind_728: t30_path4 = 1 -> Overall_Qual_calc >= 6
 ind_729: t30_path4 = 1 -> const_Garage_Cars <= 3
 ind_730: t30_path4 = 1 -> const_Lot_Area <= 6373
 ind_731: t30_path5 = 1 -> Overall_Qual_calc >= 6
 ind_732: t30_path5 = 1 -> const_Garage_Cars <= 3
 ind_733: t30_path5 = 1 -> const_Lot_Area >= 6373
 ind_734: t30_path6 = 1 -> Overall_Qual_calc >= 6
 ind_735: t30_path6 = 1 -> const_Garage_Cars >= 3
 ind_736: t30_path6 = 1 -> const_Lot_Area <= 13518
 ind_737: t30_path7 = 1 -> Overall_Qual_calc >= 6
 ind_738: t30_path7 = 1 -> const_Garage_Cars >= 3
 ind_739: t30_path7 = 1 -> const_Lot_Area >= 13518
 ind_740: t31_path0 = 1 -> x_Gr_Liv_Area <= 1488
 ind_741: t31_path0 = 1 -> x_Garage_Cond <= 2
 ind_742: t31_path0 = 1 -> x_Gr_Liv_Area <= 1155
 ind_743: t31_path1 = 1 -> x_Gr_Liv_Area <= 1488
 ind_744: t31_path1 = 1 -> x_Garage_Cond <= 2
 ind_745: t31_path1 = 1 -> x_Gr_Liv_Area >= 1155
 ind_746: t31_path2 = 1 -> x_Gr_Liv_Area <= 1488
 ind_747: t31_path2 = 1 -> x_Garage_Cond >= 2
 ind_748: t31_path2 = 1 -> x_1st_Flr_SF <= 988
 ind_749: t31_path3 = 1 -> x_Gr_Liv_Area <= 1488
 ind_750: t31_path3 = 1 -> x_Garage_Cond >= 2
 ind_751: t31_path3 = 1 -> x_1st_Flr_SF >= 988
 ind_752: t31_path4 = 1 -> x_Gr_Liv_Area >= 1488
 ind_753: t31_path4 = 1 -> x_BsmtFin_SF_1 <= 773
 ind_754: t31_path4 = 1 -> x_Kitchen_AbvGr <= 2
 ind_755: t31_path5 = 1 -> x_Gr_Liv_Area >= 1488
 ind_756: t31_path5 = 1 -> x_BsmtFin_SF_1 <= 773
 ind_757: t31_path5 = 1 -> x_Kitchen_AbvGr >= 2
 ind_758: t31_path6 = 1 -> x_Gr_Liv_Area >= 1488
 ind_759: t31_path6 = 1 -> x_BsmtFin_SF_1 >= 773
 ind_760: t31_path6 = 1 -> x_Gr_Liv_Area <= 1873
 ind_761: t31_path7 = 1 -> x_Gr_Liv_Area >= 1488
 ind_762: t31_path7 = 1 -> x_BsmtFin_SF_1 >= 773
 ind_763: t31_path7 = 1 -> x_Gr_Liv_Area >= 1873
 ind_764: t32_path0 = 1 -> const_Garage_Cars <= 2
 ind_765: t32_path0 = 1 -> const_Total_Bsmt_SF <= 648
 ind_766: t32_path0 = 1 -> const_Year_Remod_Add <= 1978
 ind_767: t32_path1 = 1 -> const_Garage_Cars <= 2
 ind_768: t32_path1 = 1 -> const_Total_Bsmt_SF <= 648
 ind_769: t32_path1 = 1 -> const_Year_Remod_Add >= 1978
 ind_770: t32_path2 = 1 -> const_Garage_Cars <= 2
 ind_771: t32_path2 = 1 -> const_Total_Bsmt_SF >= 648
 ind_772: t32_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_773: t32_path3 = 1 -> const_Garage_Cars <= 2
 ind_774: t32_path3 = 1 -> const_Total_Bsmt_SF >= 648
 ind_775: t32_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_776: t32_path4 = 1 -> const_Garage_Cars >= 2
 ind_777: t32_path4 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_778: t32_path4 = 1 -> const_2nd_Flr_SF <= 983
 ind_779: t32_path5 = 1 -> const_Garage_Cars >= 2
 ind_780: t32_path5 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_781: t32_path5 = 1 -> const_2nd_Flr_SF >= 983
 ind_782: t32_path6 = 1 -> const_Garage_Cars >= 2
 ind_783: t32_path6 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_784: t32_path6 = 1 -> const_Fireplaces <= 1
 ind_785: t32_path7 = 1 -> const_Garage_Cars >= 2
 ind_786: t32_path7 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_787: t32_path7 = 1 -> const_Fireplaces >= 1
 ind_788: t33_path0 = 1 -> const_Total_Bsmt_SF <= 1240
 ind_789: t33_path0 = 1 -> const_2nd_Flr_SF <= 604
 ind_790: t33_path0 = 1 -> const_Total_Bsmt_SF <= 774
 ind_791: t33_path1 = 1 -> const_Total_Bsmt_SF <= 1240
 ind_792: t33_path1 = 1 -> const_2nd_Flr_SF <= 604
 ind_793: t33_path1 = 1 -> const_Total_Bsmt_SF >= 774
 ind_794: t33_path2 = 1 -> const_Total_Bsmt_SF <= 1240
 ind_795: t33_path2 = 1 -> const_2nd_Flr_SF >= 604
 ind_796: t33_path2 = 1 -> const_MS_Zoning_RM <= 1
 ind_797: t33_path3 = 1 -> const_Total_Bsmt_SF <= 1240
 ind_798: t33_path3 = 1 -> const_2nd_Flr_SF >= 604
 ind_799: t33_path3 = 1 -> const_MS_Zoning_RM >= 1
 ind_800: t33_path4 = 1 -> const_Total_Bsmt_SF >= 1240
 ind_801: t33_path4 = 1 -> Overall_Qual_calc <= 9
 ind_802: t33_path4 = 1 -> x_Bsmt_Unf_SF <= 1214
 ind_803: t33_path5 = 1 -> const_Total_Bsmt_SF >= 1240
 ind_804: t33_path5 = 1 -> Overall_Qual_calc <= 9
 ind_805: t33_path5 = 1 -> x_Bsmt_Unf_SF >= 1214
 ind_806: t33_path6 = 1 -> const_Total_Bsmt_SF >= 1240
 ind_807: t33_path6 = 1 -> Overall_Qual_calc >= 9
 ind_808: t33_path6 = 1 -> const_Lot_Frontage <= 114
 ind_809: t33_path7 = 1 -> const_Total_Bsmt_SF >= 1240
 ind_810: t33_path7 = 1 -> Overall_Qual_calc >= 9
 ind_811: t33_path7 = 1 -> const_Lot_Frontage >= 114
 ind_812: t34_path0 = 1 -> x_Kitchen_Qual <= 3
 ind_813: t34_path0 = 1 -> x_paved_drive_is_Y <= 1
 ind_814: t34_path0 = 1 -> const_Total_Bsmt_SF <= 1024
 ind_815: t34_path1 = 1 -> x_Kitchen_Qual <= 3
 ind_816: t34_path1 = 1 -> x_paved_drive_is_Y <= 1
 ind_817: t34_path1 = 1 -> const_Total_Bsmt_SF >= 1024
 ind_818: t34_path2 = 1 -> x_Kitchen_Qual <= 3
 ind_819: t34_path2 = 1 -> x_paved_drive_is_Y >= 1
 ind_820: t34_path2 = 1 -> x_Gr_Liv_Area <= 1498
 ind_821: t34_path3 = 1 -> x_Kitchen_Qual <= 3
 ind_822: t34_path3 = 1 -> x_paved_drive_is_Y >= 1
 ind_823: t34_path3 = 1 -> x_Gr_Liv_Area >= 1498
 ind_824: t34_path4 = 1 -> x_Kitchen_Qual >= 3
 ind_825: t34_path4 = 1 -> x_1st_Flr_SF <= 1055
 ind_826: t34_path4 = 1 -> const_Lot_Area <= 6373
 ind_827: t34_path5 = 1 -> x_Kitchen_Qual >= 3
 ind_828: t34_path5 = 1 -> x_1st_Flr_SF <= 1055
 ind_829: t34_path5 = 1 -> const_Lot_Area >= 6373
 ind_830: t34_path6 = 1 -> x_Kitchen_Qual >= 3
 ind_831: t34_path6 = 1 -> x_1st_Flr_SF >= 1055
 ind_832: t34_path6 = 1 -> x_Gr_Liv_Area <= 1558
 ind_833: t34_path7 = 1 -> x_Kitchen_Qual >= 3
 ind_834: t34_path7 = 1 -> x_1st_Flr_SF >= 1055
 ind_835: t34_path7 = 1 -> x_Gr_Liv_Area >= 1558
 ind_836: t35_path0 = 1 -> x_Kitchen_Qual <= 3
 ind_837: t35_path0 = 1 -> const_MS_Zoning_RL <= 1
 ind_838: t35_path0 = 1 -> const_Year_Remod_Add <= 1993
 ind_839: t35_path1 = 1 -> x_Kitchen_Qual <= 3
 ind_840: t35_path1 = 1 -> const_MS_Zoning_RL <= 1
 ind_841: t35_path1 = 1 -> const_Year_Remod_Add >= 1993
 ind_842: t35_path2 = 1 -> x_Kitchen_Qual <= 3
 ind_843: t35_path2 = 1 -> const_MS_Zoning_RL >= 1
 ind_844: t35_path2 = 1 -> const_Year_Remod_Add <= 1959
 ind_845: t35_path3 = 1 -> x_Kitchen_Qual <= 3
 ind_846: t35_path3 = 1 -> const_MS_Zoning_RL >= 1
 ind_847: t35_path3 = 1 -> const_Year_Remod_Add >= 1959
 ind_848: t35_path4 = 1 -> x_Kitchen_Qual >= 3
 ind_849: t35_path4 = 1 -> const_Lot_Area <= 11765
 ind_850: t35_path4 = 1 -> x_Mas_Vnr_Area <= 174
 ind_851: t35_path5 = 1 -> x_Kitchen_Qual >= 3
 ind_852: t35_path5 = 1 -> const_Lot_Area <= 11765
 ind_853: t35_path5 = 1 -> x_Mas_Vnr_Area >= 174
 ind_854: t35_path6 = 1 -> x_Kitchen_Qual >= 3
 ind_855: t35_path6 = 1 -> const_Lot_Area >= 11765
 ind_856: t35_path6 = 1 -> const_2nd_Flr_SF <= 1368
 ind_857: t35_path7 = 1 -> x_Kitchen_Qual >= 3
 ind_858: t35_path7 = 1 -> const_Lot_Area >= 11765
 ind_859: t35_path7 = 1 -> const_2nd_Flr_SF >= 1368
 ind_860: t36_path0 = 1 -> x_Gr_Liv_Area <= 1282
 ind_861: t36_path0 = 1 -> Overall_Qual_calc <= 4
 ind_862: t36_path0 = 1 -> x_1st_Flr_SF <= 868
 ind_863: t36_path1 = 1 -> x_Gr_Liv_Area <= 1282
 ind_864: t36_path1 = 1 -> Overall_Qual_calc <= 4
 ind_865: t36_path1 = 1 -> x_1st_Flr_SF >= 868
 ind_866: t36_path2 = 1 -> x_Gr_Liv_Area <= 1282
 ind_867: t36_path2 = 1 -> Overall_Qual_calc >= 4
 ind_868: t36_path2 = 1 -> x_BsmtFin_SF_1 <= 557
 ind_869: t36_path3 = 1 -> x_Gr_Liv_Area <= 1282
 ind_870: t36_path3 = 1 -> Overall_Qual_calc >= 4
 ind_871: t36_path3 = 1 -> x_BsmtFin_SF_1 >= 557
 ind_872: t36_path4 = 1 -> x_Gr_Liv_Area >= 1282
 ind_873: t36_path4 = 1 -> x_Gr_Liv_Area <= 2263
 ind_874: t36_path4 = 1 -> x_BsmtFin_SF_1 <= 784
 ind_875: t36_path5 = 1 -> x_Gr_Liv_Area >= 1282
 ind_876: t36_path5 = 1 -> x_Gr_Liv_Area <= 2263
 ind_877: t36_path5 = 1 -> x_BsmtFin_SF_1 >= 784
 ind_878: t36_path6 = 1 -> x_Gr_Liv_Area >= 1282
 ind_879: t36_path6 = 1 -> x_Gr_Liv_Area >= 2263
 ind_880: t36_path6 = 1 -> x_1st_Flr_SF <= 1502
 ind_881: t36_path7 = 1 -> x_Gr_Liv_Area >= 1282
 ind_882: t36_path7 = 1 -> x_Gr_Liv_Area >= 2263
 ind_883: t36_path7 = 1 -> x_1st_Flr_SF >= 1502
 ind_884: t37_path0 = 1 -> x_Gr_Liv_Area <= 1494
 ind_885: t37_path0 = 1 -> Overall_Qual_calc <= 4
 ind_886: t37_path0 = 1 -> x_1st_Flr_SF <= 908
 ind_887: t37_path1 = 1 -> x_Gr_Liv_Area <= 1494
 ind_888: t37_path1 = 1 -> Overall_Qual_calc <= 4
 ind_889: t37_path1 = 1 -> x_1st_Flr_SF >= 908
 ind_890: t37_path2 = 1 -> x_Gr_Liv_Area <= 1494
 ind_891: t37_path2 = 1 -> Overall_Qual_calc >= 4
 ind_892: t37_path2 = 1 -> const_Total_Bsmt_SF <= 975
 ind_893: t37_path3 = 1 -> x_Gr_Liv_Area <= 1494
 ind_894: t37_path3 = 1 -> Overall_Qual_calc >= 4
 ind_895: t37_path3 = 1 -> const_Total_Bsmt_SF >= 975
 ind_896: t37_path4 = 1 -> x_Gr_Liv_Area >= 1494
 ind_897: t37_path4 = 1 -> const_Year_Remod_Add <= 1983
 ind_898: t37_path4 = 1 -> x_BsmtFin_SF_1 <= 192
 ind_899: t37_path5 = 1 -> x_Gr_Liv_Area >= 1494
 ind_900: t37_path5 = 1 -> const_Year_Remod_Add <= 1983
 ind_901: t37_path5 = 1 -> x_BsmtFin_SF_1 >= 192
 ind_902: t37_path6 = 1 -> x_Gr_Liv_Area >= 1494
 ind_903: t37_path6 = 1 -> const_Year_Remod_Add >= 1983
 ind_904: t37_path6 = 1 -> x_Gr_Liv_Area <= 1968
 ind_905: t37_path7 = 1 -> x_Gr_Liv_Area >= 1494
 ind_906: t37_path7 = 1 -> const_Year_Remod_Add >= 1983
 ind_907: t37_path7 = 1 -> x_Gr_Liv_Area >= 1968
 ind_908: t38_path0 = 1 -> Overall_Qual_calc <= 6
 ind_909: t38_path0 = 1 -> central_air_yes <= 1
 ind_910: t38_path0 = 1 -> Overall_Qual_calc <= 3
 ind_911: t38_path1 = 1 -> Overall_Qual_calc <= 6
 ind_912: t38_path1 = 1 -> central_air_yes <= 1
 ind_913: t38_path1 = 1 -> Overall_Qual_calc >= 3
 ind_914: t38_path2 = 1 -> Overall_Qual_calc <= 6
 ind_915: t38_path2 = 1 -> central_air_yes >= 1
 ind_916: t38_path2 = 1 -> x_b1_is_Unf <= 1
 ind_917: t38_path3 = 1 -> Overall_Qual_calc <= 6
 ind_918: t38_path3 = 1 -> central_air_yes >= 1
 ind_919: t38_path3 = 1 -> x_b1_is_Unf >= 1
 ind_920: t38_path4 = 1 -> Overall_Qual_calc >= 6
 ind_921: t38_path4 = 1 -> x_Gr_Liv_Area <= 1800
 ind_922: t38_path4 = 1 -> x_Gr_Liv_Area <= 1144
 ind_923: t38_path5 = 1 -> Overall_Qual_calc >= 6
 ind_924: t38_path5 = 1 -> x_Gr_Liv_Area <= 1800
 ind_925: t38_path5 = 1 -> x_Gr_Liv_Area >= 1144
 ind_926: t38_path6 = 1 -> Overall_Qual_calc >= 6
 ind_927: t38_path6 = 1 -> x_Gr_Liv_Area >= 1800
 ind_928: t38_path6 = 1 -> x_Gr_Liv_Area <= 2327
 ind_929: t38_path7 = 1 -> Overall_Qual_calc >= 6
 ind_930: t38_path7 = 1 -> x_Gr_Liv_Area >= 1800
 ind_931: t38_path7 = 1 -> x_Gr_Liv_Area >= 2327
 ind_932: t39_path0 = 1 -> x_Gr_Liv_Area <= 1308
 ind_933: t39_path0 = 1 -> const_Year_Built <= 1941
 ind_934: t39_path0 = 1 -> const_Overall_Cond <= 7
 ind_935: t39_path1 = 1 -> x_Gr_Liv_Area <= 1308
 ind_936: t39_path1 = 1 -> const_Year_Built <= 1941
 ind_937: t39_path1 = 1 -> const_Overall_Cond >= 7
 ind_938: t39_path2 = 1 -> x_Gr_Liv_Area <= 1308
 ind_939: t39_path2 = 1 -> const_Year_Built >= 1941
 ind_940: t39_path2 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_941: t39_path3 = 1 -> x_Gr_Liv_Area <= 1308
 ind_942: t39_path3 = 1 -> const_Year_Built >= 1941
 ind_943: t39_path3 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_944: t39_path4 = 1 -> x_Gr_Liv_Area >= 1308
 ind_945: t39_path4 = 1 -> x_BsmtFin_SF_1 <= 856
 ind_946: t39_path4 = 1 -> const_Overall_Cond <= 5
 ind_947: t39_path5 = 1 -> x_Gr_Liv_Area >= 1308
 ind_948: t39_path5 = 1 -> x_BsmtFin_SF_1 <= 856
 ind_949: t39_path5 = 1 -> const_Overall_Cond >= 5
 ind_950: t39_path6 = 1 -> x_Gr_Liv_Area >= 1308
 ind_951: t39_path6 = 1 -> x_BsmtFin_SF_1 >= 856
 ind_952: t39_path6 = 1 -> Overall_Qual_calc <= 9
 ind_953: t39_path7 = 1 -> x_Gr_Liv_Area >= 1308
 ind_954: t39_path7 = 1 -> x_BsmtFin_SF_1 >= 856
 ind_955: t39_path7 = 1 -> Overall_Qual_calc >= 9
 ind_956: t40_path0 = 1 -> const_Year_Remod_Add <= 1985
 ind_957: t40_path0 = 1 -> x_Garage_Cond <= 2
 ind_958: t40_path0 = 1 -> const_Total_Bsmt_SF <= 858
 ind_959: t40_path1 = 1 -> const_Year_Remod_Add <= 1985
 ind_960: t40_path1 = 1 -> x_Garage_Cond <= 2
 ind_961: t40_path1 = 1 -> const_Total_Bsmt_SF >= 858
 ind_962: t40_path2 = 1 -> const_Year_Remod_Add <= 1985
 ind_963: t40_path2 = 1 -> x_Garage_Cond >= 2
 ind_964: t40_path2 = 1 -> const_Lot_Frontage <= 66
 ind_965: t40_path3 = 1 -> const_Year_Remod_Add <= 1985
 ind_966: t40_path3 = 1 -> x_Garage_Cond >= 2
 ind_967: t40_path3 = 1 -> const_Lot_Frontage >= 66
 ind_968: t40_path4 = 1 -> const_Year_Remod_Add >= 1985
 ind_969: t40_path4 = 1 -> x_Gr_Liv_Area <= 1638
 ind_970: t40_path4 = 1 -> const_Total_Bsmt_SF <= 1055
 ind_971: t40_path5 = 1 -> const_Year_Remod_Add >= 1985
 ind_972: t40_path5 = 1 -> x_Gr_Liv_Area <= 1638
 ind_973: t40_path5 = 1 -> const_Total_Bsmt_SF >= 1055
 ind_974: t40_path6 = 1 -> const_Year_Remod_Add >= 1985
 ind_975: t40_path6 = 1 -> x_Gr_Liv_Area >= 1638
 ind_976: t40_path6 = 1 -> const_2nd_Flr_SF <= 1368
 ind_977: t40_path7 = 1 -> const_Year_Remod_Add >= 1985
 ind_978: t40_path7 = 1 -> x_Gr_Liv_Area >= 1638
 ind_979: t40_path7 = 1 -> const_2nd_Flr_SF >= 1368
 ind_980: t41_path0 = 1 -> Overall_Qual_calc <= 7
 ind_981: t41_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_982: t41_path0 = 1 -> Overall_Qual_calc <= 5
 ind_983: t41_path1 = 1 -> Overall_Qual_calc <= 7
 ind_984: t41_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_985: t41_path1 = 1 -> Overall_Qual_calc >= 5
 ind_986: t41_path2 = 1 -> Overall_Qual_calc <= 7
 ind_987: t41_path2 = 1 -> const_MS_SubClass_30 >= 1
 ind_988: t41_path2 = 1 -> x_BsmtFin_SF_1 <= 206
 ind_989: t41_path3 = 1 -> Overall_Qual_calc <= 7
 ind_990: t41_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_991: t41_path3 = 1 -> x_BsmtFin_SF_1 >= 206
 ind_992: t41_path4 = 1 -> Overall_Qual_calc >= 7
 ind_993: t41_path4 = 1 -> Overall_Qual_calc <= 9
 ind_994: t41_path4 = 1 -> x_BsmtFin_SF_1 <= 1097
 ind_995: t41_path5 = 1 -> Overall_Qual_calc >= 7
 ind_996: t41_path5 = 1 -> Overall_Qual_calc <= 9
 ind_997: t41_path5 = 1 -> x_BsmtFin_SF_1 >= 1097
 ind_998: t41_path6 = 1 -> Overall_Qual_calc >= 7
 ind_999: t41_path6 = 1 -> Overall_Qual_calc >= 9
 ind_1000: t41_path6 = 1 -> x_Open_Porch_SF <= 236
 ind_1001: t41_path7 = 1 -> Overall_Qual_calc >= 7
 ind_1002: t41_path7 = 1 -> Overall_Qual_calc >= 9
 ind_1003: t41_path7 = 1 -> x_Open_Porch_SF >= 236
 ind_1004: t42_path0 = 1 -> const_Total_Bsmt_SF <= 1200
 ind_1005: t42_path0 = 1 -> x_Garage_Cond <= 2
 ind_1006: t42_path0 = 1 -> x_Exter_Cond <= 2
 ind_1007: t42_path1 = 1 -> const_Total_Bsmt_SF <= 1200
 ind_1008: t42_path1 = 1 -> x_Garage_Cond <= 2
 ind_1009: t42_path1 = 1 -> x_Exter_Cond >= 2
 ind_1010: t42_path2 = 1 -> const_Total_Bsmt_SF <= 1200
 ind_1011: t42_path2 = 1 -> x_Garage_Cond >= 2
 ind_1012: t42_path2 = 1 -> x_Gr_Liv_Area <= 1766
 ind_1013: t42_path3 = 1 -> const_Total_Bsmt_SF <= 1200
 ind_1014: t42_path3 = 1 -> x_Garage_Cond >= 2
 ind_1015: t42_path3 = 1 -> x_Gr_Liv_Area >= 1766
 ind_1016: t42_path4 = 1 -> const_Total_Bsmt_SF >= 1200
 ind_1017: t42_path4 = 1 -> x_Gr_Liv_Area <= 2052
 ind_1018: t42_path4 = 1 -> x_BsmtFin_SF_1 <= 324
 ind_1019: t42_path5 = 1 -> const_Total_Bsmt_SF >= 1200
 ind_1020: t42_path5 = 1 -> x_Gr_Liv_Area <= 2052
 ind_1021: t42_path5 = 1 -> x_BsmtFin_SF_1 >= 324
 ind_1022: t42_path6 = 1 -> const_Total_Bsmt_SF >= 1200
 ind_1023: t42_path6 = 1 -> x_Gr_Liv_Area >= 2052
 ind_1024: t42_path6 = 1 -> x_Open_Porch_SF <= 230
 ind_1025: t42_path7 = 1 -> const_Total_Bsmt_SF >= 1200
 ind_1026: t42_path7 = 1 -> x_Gr_Liv_Area >= 2052
 ind_1027: t42_path7 = 1 -> x_Open_Porch_SF >= 230
 ind_1028: t43_path0 = 1 -> const_Garage_Yr_Blt <= 1982
 ind_1029: t43_path0 = 1 -> x_BsmtFin_SF_1 <= 223
 ind_1030: t43_path0 = 1 -> const_Overall_Cond <= 5
 ind_1031: t43_path1 = 1 -> const_Garage_Yr_Blt <= 1982
 ind_1032: t43_path1 = 1 -> x_BsmtFin_SF_1 <= 223
 ind_1033: t43_path1 = 1 -> const_Overall_Cond >= 5
 ind_1034: t43_path2 = 1 -> const_Garage_Yr_Blt <= 1982
 ind_1035: t43_path2 = 1 -> x_BsmtFin_SF_1 >= 223
 ind_1036: t43_path2 = 1 -> const_Overall_Cond <= 6
 ind_1037: t43_path3 = 1 -> const_Garage_Yr_Blt <= 1982
 ind_1038: t43_path3 = 1 -> x_BsmtFin_SF_1 >= 223
 ind_1039: t43_path3 = 1 -> const_Overall_Cond >= 6
 ind_1040: t43_path4 = 1 -> const_Garage_Yr_Blt >= 1982
 ind_1041: t43_path4 = 1 -> Overall_Qual_calc <= 8
 ind_1042: t43_path4 = 1 -> x_Open_Porch_SF <= 42
 ind_1043: t43_path5 = 1 -> const_Garage_Yr_Blt >= 1982
 ind_1044: t43_path5 = 1 -> Overall_Qual_calc <= 8
 ind_1045: t43_path5 = 1 -> x_Open_Porch_SF >= 42
 ind_1046: t43_path6 = 1 -> const_Garage_Yr_Blt >= 1982
 ind_1047: t43_path6 = 1 -> Overall_Qual_calc >= 8
 ind_1048: t43_path6 = 1 -> x_Open_Porch_SF <= 274
 ind_1049: t43_path7 = 1 -> const_Garage_Yr_Blt >= 1982
 ind_1050: t43_path7 = 1 -> Overall_Qual_calc >= 8
 ind_1051: t43_path7 = 1 -> x_Open_Porch_SF >= 274
 ind_1052: t44_path0 = 1 -> const_Bsmt_Qual <= 3
 ind_1053: t44_path0 = 1 -> const_Total_Bsmt_SF <= 630
 ind_1054: t44_path0 = 1 -> x_Gr_Liv_Area <= 1211
 ind_1055: t44_path1 = 1 -> const_Bsmt_Qual <= 3
 ind_1056: t44_path1 = 1 -> const_Total_Bsmt_SF <= 630
 ind_1057: t44_path1 = 1 -> x_Gr_Liv_Area >= 1211
 ind_1058: t44_path2 = 1 -> const_Bsmt_Qual <= 3
 ind_1059: t44_path2 = 1 -> const_Total_Bsmt_SF >= 630
 ind_1060: t44_path2 = 1 -> const_Garage_Yr_Blt <= 1921
 ind_1061: t44_path3 = 1 -> const_Bsmt_Qual <= 3
 ind_1062: t44_path3 = 1 -> const_Total_Bsmt_SF >= 630
 ind_1063: t44_path3 = 1 -> const_Garage_Yr_Blt >= 1921
 ind_1064: t44_path4 = 1 -> const_Bsmt_Qual >= 3
 ind_1065: t44_path4 = 1 -> x_1st_Flr_SF <= 1652
 ind_1066: t44_path4 = 1 -> const_Total_Bsmt_SF <= 926
 ind_1067: t44_path5 = 1 -> const_Bsmt_Qual >= 3
 ind_1068: t44_path5 = 1 -> x_1st_Flr_SF <= 1652
 ind_1069: t44_path5 = 1 -> const_Total_Bsmt_SF >= 926
 ind_1070: t44_path6 = 1 -> const_Bsmt_Qual >= 3
 ind_1071: t44_path6 = 1 -> x_1st_Flr_SF >= 1652
 ind_1072: t44_path6 = 1 -> x_Gr_Liv_Area <= 2028
 ind_1073: t44_path7 = 1 -> const_Bsmt_Qual >= 3
 ind_1074: t44_path7 = 1 -> x_1st_Flr_SF >= 1652
 ind_1075: t44_path7 = 1 -> x_Gr_Liv_Area >= 2028
 ind_1076: t45_path0 = 1 -> Overall_Qual_calc <= 6
 ind_1077: t45_path0 = 1 -> x_BsmtFin_SF_1 <= 75
 ind_1078: t45_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1079: t45_path1 = 1 -> Overall_Qual_calc <= 6
 ind_1080: t45_path1 = 1 -> x_BsmtFin_SF_1 <= 75
 ind_1081: t45_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1082: t45_path2 = 1 -> Overall_Qual_calc <= 6
 ind_1083: t45_path2 = 1 -> x_BsmtFin_SF_1 >= 75
 ind_1084: t45_path2 = 1 -> const_Total_Bsmt_SF <= 1356
 ind_1085: t45_path3 = 1 -> Overall_Qual_calc <= 6
 ind_1086: t45_path3 = 1 -> x_BsmtFin_SF_1 >= 75
 ind_1087: t45_path3 = 1 -> const_Total_Bsmt_SF >= 1356
 ind_1088: t45_path4 = 1 -> Overall_Qual_calc >= 6
 ind_1089: t45_path4 = 1 -> x_BsmtFin_SF_1 <= 727
 ind_1090: t45_path4 = 1 -> const_Garage_Cars <= 2
 ind_1091: t45_path5 = 1 -> Overall_Qual_calc >= 6
 ind_1092: t45_path5 = 1 -> x_BsmtFin_SF_1 <= 727
 ind_1093: t45_path5 = 1 -> const_Garage_Cars >= 2
 ind_1094: t45_path6 = 1 -> Overall_Qual_calc >= 6
 ind_1095: t45_path6 = 1 -> x_BsmtFin_SF_1 >= 727
 ind_1096: t45_path6 = 1 -> Overall_Qual_calc <= 8
 ind_1097: t45_path7 = 1 -> Overall_Qual_calc >= 6
 ind_1098: t45_path7 = 1 -> x_BsmtFin_SF_1 >= 727
 ind_1099: t45_path7 = 1 -> Overall_Qual_calc >= 8
 ind_1100: t46_path0 = 1 -> x_1st_Flr_SF <= 1155
 ind_1101: t46_path0 = 1 -> x_Garage_Cond <= 2
 ind_1102: t46_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_1103: t46_path1 = 1 -> x_1st_Flr_SF <= 1155
 ind_1104: t46_path1 = 1 -> x_Garage_Cond <= 2
 ind_1105: t46_path1 = 1 -> const_MS_SubClass_30 >= 1
 ind_1106: t46_path2 = 1 -> x_1st_Flr_SF <= 1155
 ind_1107: t46_path2 = 1 -> x_Garage_Cond >= 2
 ind_1108: t46_path2 = 1 -> const_2nd_Flr_SF <= 676
 ind_1109: t46_path3 = 1 -> x_1st_Flr_SF <= 1155
 ind_1110: t46_path3 = 1 -> x_Garage_Cond >= 2
 ind_1111: t46_path3 = 1 -> const_2nd_Flr_SF >= 676
 ind_1112: t46_path4 = 1 -> x_1st_Flr_SF >= 1155
 ind_1113: t46_path4 = 1 -> const_2nd_Flr_SF <= 1168
 ind_1114: t46_path4 = 1 -> const_Year_Remod_Add <= 1993
 ind_1115: t46_path5 = 1 -> x_1st_Flr_SF >= 1155
 ind_1116: t46_path5 = 1 -> const_2nd_Flr_SF <= 1168
 ind_1117: t46_path5 = 1 -> const_Year_Remod_Add >= 1993
 ind_1118: t46_path6 = 1 -> x_1st_Flr_SF >= 1155
 ind_1119: t46_path6 = 1 -> const_2nd_Flr_SF >= 1168
 ind_1120: t46_path6 = 1 -> x_Kitchen_AbvGr <= 2
 ind_1121: t46_path7 = 1 -> x_1st_Flr_SF >= 1155
 ind_1122: t46_path7 = 1 -> const_2nd_Flr_SF >= 1168
 ind_1123: t46_path7 = 1 -> x_Kitchen_AbvGr >= 2
 ind_1124: t47_path0 = 1 -> const_Year_Remod_Add <= 1981
 ind_1125: t47_path0 = 1 -> const_Overall_Cond <= 5
 ind_1126: t47_path0 = 1 -> const_Year_Remod_Add <= 1959
 ind_1127: t47_path1 = 1 -> const_Year_Remod_Add <= 1981
 ind_1128: t47_path1 = 1 -> const_Overall_Cond <= 5
 ind_1129: t47_path1 = 1 -> const_Year_Remod_Add >= 1959
 ind_1130: t47_path2 = 1 -> const_Year_Remod_Add <= 1981
 ind_1131: t47_path2 = 1 -> const_Overall_Cond >= 5
 ind_1132: t47_path2 = 1 -> const_Lot_Frontage <= 69
 ind_1133: t47_path3 = 1 -> const_Year_Remod_Add <= 1981
 ind_1134: t47_path3 = 1 -> const_Overall_Cond >= 5
 ind_1135: t47_path3 = 1 -> const_Lot_Frontage >= 69
 ind_1136: t47_path4 = 1 -> const_Year_Remod_Add >= 1981
 ind_1137: t47_path4 = 1 -> x_Fireplace_Qu <= 2
 ind_1138: t47_path4 = 1 -> const_Lot_Frontage <= 61
 ind_1139: t47_path5 = 1 -> const_Year_Remod_Add >= 1981
 ind_1140: t47_path5 = 1 -> x_Fireplace_Qu <= 2
 ind_1141: t47_path5 = 1 -> const_Lot_Frontage >= 61
 ind_1142: t47_path6 = 1 -> const_Year_Remod_Add >= 1981
 ind_1143: t47_path6 = 1 -> x_Fireplace_Qu >= 2
 ind_1144: t47_path6 = 1 -> Overall_Qual_calc <= 9
 ind_1145: t47_path7 = 1 -> const_Year_Remod_Add >= 1981
 ind_1146: t47_path7 = 1 -> x_Fireplace_Qu >= 2
 ind_1147: t47_path7 = 1 -> Overall_Qual_calc >= 9
 ind_1148: t48_path0 = 1 -> x_Gr_Liv_Area <= 1712
 ind_1149: t48_path0 = 1 -> x_paved_drive_is_N <= 1
 ind_1150: t48_path0 = 1 -> x_Garage_Area <= 384
 ind_1151: t48_path1 = 1 -> x_Gr_Liv_Area <= 1712
 ind_1152: t48_path1 = 1 -> x_paved_drive_is_N <= 1
 ind_1153: t48_path1 = 1 -> x_Garage_Area >= 384
 ind_1154: t48_path2 = 1 -> x_Gr_Liv_Area <= 1712
 ind_1155: t48_path2 = 1 -> x_paved_drive_is_N >= 1
 ind_1156: t48_path2 = 1 -> const_Sale_Condition_Normal <= 1
 ind_1157: t48_path3 = 1 -> x_Gr_Liv_Area <= 1712
 ind_1158: t48_path3 = 1 -> x_paved_drive_is_N >= 1
 ind_1159: t48_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_1160: t48_path4 = 1 -> x_Gr_Liv_Area >= 1712
 ind_1161: t48_path4 = 1 -> x_b1_is_GLQ <= 1
 ind_1162: t48_path4 = 1 -> const_Yr_Sold <= 2008
 ind_1163: t48_path5 = 1 -> x_Gr_Liv_Area >= 1712
 ind_1164: t48_path5 = 1 -> x_b1_is_GLQ <= 1
 ind_1165: t48_path5 = 1 -> const_Yr_Sold >= 2008
 ind_1166: t48_path6 = 1 -> x_Gr_Liv_Area >= 1712
 ind_1167: t48_path6 = 1 -> x_b1_is_GLQ >= 1
 ind_1168: t48_path6 = 1 -> const_Total_Bsmt_SF <= 1850
 ind_1169: t48_path7 = 1 -> x_Gr_Liv_Area >= 1712
 ind_1170: t48_path7 = 1 -> x_b1_is_GLQ >= 1
 ind_1171: t48_path7 = 1 -> const_Total_Bsmt_SF >= 1850
 ind_1172: t49_path0 = 1 -> x_Gr_Liv_Area <= 1494
 ind_1173: t49_path0 = 1 -> x_paved_drive_is_Y <= 1
 ind_1174: t49_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1175: t49_path1 = 1 -> x_Gr_Liv_Area <= 1494
 ind_1176: t49_path1 = 1 -> x_paved_drive_is_Y <= 1
 ind_1177: t49_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1178: t49_path2 = 1 -> x_Gr_Liv_Area <= 1494
 ind_1179: t49_path2 = 1 -> x_paved_drive_is_Y >= 1
 ind_1180: t49_path2 = 1 -> x_1st_Flr_SF <= 1064
 ind_1181: t49_path3 = 1 -> x_Gr_Liv_Area <= 1494
 ind_1182: t49_path3 = 1 -> x_paved_drive_is_Y >= 1
 ind_1183: t49_path3 = 1 -> x_1st_Flr_SF >= 1064
 ind_1184: t49_path4 = 1 -> x_Gr_Liv_Area >= 1494
 ind_1185: t49_path4 = 1 -> const_Neighborhood_OldTown <= 1
 ind_1186: t49_path4 = 1 -> const_Yr_Sold <= 2008
 ind_1187: t49_path5 = 1 -> x_Gr_Liv_Area >= 1494
 ind_1188: t49_path5 = 1 -> const_Neighborhood_OldTown <= 1
 ind_1189: t49_path5 = 1 -> const_Yr_Sold >= 2008
 ind_1190: t49_path6 = 1 -> x_Gr_Liv_Area >= 1494
 ind_1191: t49_path6 = 1 -> const_Neighborhood_OldTown >= 1
 ind_1192: t49_path6 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1193: t49_path7 = 1 -> x_Gr_Liv_Area >= 1494
 ind_1194: t49_path7 = 1 -> const_Neighborhood_OldTown >= 1
 ind_1195: t49_path7 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1196: t50_path0 = 1 -> x_Gr_Liv_Area <= 1183
 ind_1197: t50_path0 = 1 -> Overall_Qual_calc <= 4
 ind_1198: t50_path1 = 1 -> x_Gr_Liv_Area <= 1183
 ind_1199: t50_path1 = 1 -> Overall_Qual_calc >= 4
 ind_1200: t50_path1 = 1 -> const_MS_Zoning_RL <= 1
 ind_1201: t50_path2 = 1 -> x_Gr_Liv_Area <= 1183
 ind_1202: t50_path2 = 1 -> Overall_Qual_calc >= 4
 ind_1203: t50_path2 = 1 -> const_MS_Zoning_RL >= 1
 ind_1204: t50_path3 = 1 -> x_Gr_Liv_Area >= 1183
 ind_1205: t50_path3 = 1 -> const_Lot_Area <= 11428
 ind_1206: t50_path3 = 1 -> const_Year_Built <= 1986
 ind_1207: t50_path4 = 1 -> x_Gr_Liv_Area >= 1183
 ind_1208: t50_path4 = 1 -> const_Lot_Area <= 11428
 ind_1209: t50_path4 = 1 -> const_Year_Built >= 1986
 ind_1210: t50_path5 = 1 -> x_Gr_Liv_Area >= 1183
 ind_1211: t50_path5 = 1 -> const_Lot_Area >= 11428
 ind_1212: t50_path5 = 1 -> x_Gr_Liv_Area <= 1734
 ind_1213: t50_path6 = 1 -> x_Gr_Liv_Area >= 1183
 ind_1214: t50_path6 = 1 -> const_Lot_Area >= 11428
 ind_1215: t50_path6 = 1 -> x_Gr_Liv_Area >= 1734
 ind_1216: t51_path0 = 1 -> Overall_Qual_calc <= 6
 ind_1217: t51_path0 = 1 -> x_Gr_Liv_Area <= 768
 ind_1218: t51_path1 = 1 -> Overall_Qual_calc <= 6
 ind_1219: t51_path1 = 1 -> x_Gr_Liv_Area >= 768
 ind_1220: t51_path1 = 1 -> x_BsmtFin_SF_1 <= 27
 ind_1221: t51_path2 = 1 -> Overall_Qual_calc <= 6
 ind_1222: t51_path2 = 1 -> x_Gr_Liv_Area >= 768
 ind_1223: t51_path2 = 1 -> x_BsmtFin_SF_1 >= 27
 ind_1224: t51_path3 = 1 -> Overall_Qual_calc >= 6
 ind_1225: t51_path3 = 1 -> x_Gr_Liv_Area <= 2327
 ind_1226: t51_path3 = 1 -> x_BsmtFin_SF_1 <= 705
 ind_1227: t51_path4 = 1 -> Overall_Qual_calc >= 6
 ind_1228: t51_path4 = 1 -> x_Gr_Liv_Area <= 2327
 ind_1229: t51_path4 = 1 -> x_BsmtFin_SF_1 >= 705
 ind_1230: t51_path5 = 1 -> Overall_Qual_calc >= 6
 ind_1231: t51_path5 = 1 -> x_Gr_Liv_Area >= 2327
 ind_1232: t51_path5 = 1 -> x_Open_Porch_SF <= 274
 ind_1233: t51_path6 = 1 -> Overall_Qual_calc >= 6
 ind_1234: t51_path6 = 1 -> x_Gr_Liv_Area >= 2327
 ind_1235: t51_path6 = 1 -> x_Open_Porch_SF >= 274
 ind_1236: t52_path0 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_1237: t52_path0 = 1 -> const_Overall_Cond <= 5
 ind_1238: t52_path0 = 1 -> const_Overall_Cond <= 4
 ind_1239: t52_path1 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_1240: t52_path1 = 1 -> const_Overall_Cond <= 5
 ind_1241: t52_path1 = 1 -> const_Overall_Cond >= 4
 ind_1242: t52_path2 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_1243: t52_path2 = 1 -> const_Overall_Cond >= 5
 ind_1244: t52_path2 = 1 -> x_Gr_Liv_Area <= 1720
 ind_1245: t52_path3 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_1246: t52_path3 = 1 -> const_Overall_Cond >= 5
 ind_1247: t52_path3 = 1 -> x_Gr_Liv_Area >= 1720
 ind_1248: t52_path4 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_1249: t52_path4 = 1 -> x_Gr_Liv_Area <= 2028
 ind_1250: t52_path4 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_1251: t52_path5 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_1252: t52_path5 = 1 -> x_Gr_Liv_Area <= 2028
 ind_1253: t52_path5 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_1254: t52_path6 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_1255: t52_path6 = 1 -> x_Gr_Liv_Area >= 2028
 ind_1256: t52_path6 = 1 -> const_Lot_Area <= 13688
 ind_1257: t52_path7 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_1258: t52_path7 = 1 -> x_Gr_Liv_Area >= 2028
 ind_1259: t52_path7 = 1 -> const_Lot_Area >= 13688
 ind_1260: t53_path0 = 1 -> x_Fireplace_Qu <= 1
 ind_1261: t53_path0 = 1 -> x_Exter_Cond <= 2
 ind_1262: t53_path0 = 1 -> x_Bedroom_AbvGr <= 4
 ind_1263: t53_path1 = 1 -> x_Fireplace_Qu <= 1
 ind_1264: t53_path1 = 1 -> x_Exter_Cond <= 2
 ind_1265: t53_path1 = 1 -> x_Bedroom_AbvGr >= 4
 ind_1266: t53_path2 = 1 -> x_Fireplace_Qu <= 1
 ind_1267: t53_path2 = 1 -> x_Exter_Cond >= 2
 ind_1268: t53_path2 = 1 -> const_Garage_Type_Attchd <= 1
 ind_1269: t53_path3 = 1 -> x_Fireplace_Qu <= 1
 ind_1270: t53_path3 = 1 -> x_Exter_Cond >= 2
 ind_1271: t53_path3 = 1 -> const_Garage_Type_Attchd >= 1
 ind_1272: t53_path4 = 1 -> x_Fireplace_Qu >= 1
 ind_1273: t53_path4 = 1 -> const_Overall_Cond <= 4
 ind_1274: t53_path5 = 1 -> x_Fireplace_Qu >= 1
 ind_1275: t53_path5 = 1 -> const_Overall_Cond >= 4
 ind_1276: t53_path5 = 1 -> Overall_Qual_calc <= 9
 ind_1277: t53_path6 = 1 -> x_Fireplace_Qu >= 1
 ind_1278: t53_path6 = 1 -> const_Overall_Cond >= 4
 ind_1279: t53_path6 = 1 -> Overall_Qual_calc >= 9
 ind_1280: t54_path0 = 1 -> const_Overall_Cond <= 5
 ind_1281: t54_path0 = 1 -> const_Overall_Cond <= 4
 ind_1282: t54_path1 = 1 -> const_Overall_Cond <= 5
 ind_1283: t54_path1 = 1 -> const_Overall_Cond >= 4
 ind_1284: t54_path1 = 1 -> const_Yr_Sold <= 2007
 ind_1285: t54_path2 = 1 -> const_Overall_Cond <= 5
 ind_1286: t54_path2 = 1 -> const_Overall_Cond >= 4
 ind_1287: t54_path2 = 1 -> const_Yr_Sold >= 2007
 ind_1288: t54_path3 = 1 -> const_Overall_Cond >= 5
 ind_1289: t54_path3 = 1 -> const_Total_Bsmt_SF <= 648
 ind_1290: t54_path3 = 1 -> x_Exter_Cond <= 2
 ind_1291: t54_path4 = 1 -> const_Overall_Cond >= 5
 ind_1292: t54_path4 = 1 -> const_Total_Bsmt_SF <= 648
 ind_1293: t54_path4 = 1 -> x_Exter_Cond >= 2
 ind_1294: t54_path5 = 1 -> const_Overall_Cond >= 5
 ind_1295: t54_path5 = 1 -> const_Total_Bsmt_SF >= 648
 ind_1296: t54_path5 = 1 -> x_Kitchen_Qual <= 4
 ind_1297: t54_path6 = 1 -> const_Overall_Cond >= 5
 ind_1298: t54_path6 = 1 -> const_Total_Bsmt_SF >= 648
 ind_1299: t54_path6 = 1 -> x_Kitchen_Qual >= 4
 ind_1300: t55_path0 = 1 -> const_Fireplaces <= 1
 ind_1301: t55_path0 = 1 -> x_Exter_Cond <= 2
 ind_1302: t55_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_1303: t55_path1 = 1 -> const_Fireplaces <= 1
 ind_1304: t55_path1 = 1 -> x_Exter_Cond <= 2
 ind_1305: t55_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_1306: t55_path2 = 1 -> const_Fireplaces <= 1
 ind_1307: t55_path2 = 1 -> x_Exter_Cond >= 2
 ind_1308: t55_path2 = 1 -> x_Garage_Area <= 270
 ind_1309: t55_path3 = 1 -> const_Fireplaces <= 1
 ind_1310: t55_path3 = 1 -> x_Exter_Cond >= 2
 ind_1311: t55_path3 = 1 -> x_Garage_Area >= 270
 ind_1312: t55_path4 = 1 -> const_Fireplaces >= 1
 ind_1313: t55_path4 = 1 -> x_BsmtFin_SF_1 <= 1097
 ind_1314: t55_path4 = 1 -> const_2nd_Flr_SF <= 933
 ind_1315: t55_path5 = 1 -> const_Fireplaces >= 1
 ind_1316: t55_path5 = 1 -> x_BsmtFin_SF_1 <= 1097
 ind_1317: t55_path5 = 1 -> const_2nd_Flr_SF >= 933
 ind_1318: t55_path6 = 1 -> const_Fireplaces >= 1
 ind_1319: t55_path6 = 1 -> x_BsmtFin_SF_1 >= 1097
 ind_1320: t55_path6 = 1 -> x_Wood_Deck_SF <= 371
 ind_1321: t55_path7 = 1 -> const_Fireplaces >= 1
 ind_1322: t55_path7 = 1 -> x_BsmtFin_SF_1 >= 1097
 ind_1323: t55_path7 = 1 -> x_Wood_Deck_SF >= 371
 ind_1324: t56_path0 = 1 -> const_Total_Bsmt_SF <= 1046
 ind_1325: t56_path0 = 1 -> const_Lot_Area <= 6442
 ind_1326: t56_path0 = 1 -> x_BsmtFin_SF_1 <= 370
 ind_1327: t56_path1 = 1 -> const_Total_Bsmt_SF <= 1046
 ind_1328: t56_path1 = 1 -> const_Lot_Area <= 6442
 ind_1329: t56_path1 = 1 -> x_BsmtFin_SF_1 >= 370
 ind_1330: t56_path2 = 1 -> const_Total_Bsmt_SF <= 1046
 ind_1331: t56_path2 = 1 -> const_Lot_Area >= 6442
 ind_1332: t56_path2 = 1 -> x_Half_Bath <= 1
 ind_1333: t56_path3 = 1 -> const_Total_Bsmt_SF <= 1046
 ind_1334: t56_path3 = 1 -> const_Lot_Area >= 6442
 ind_1335: t56_path3 = 1 -> x_Half_Bath >= 1
 ind_1336: t56_path4 = 1 -> const_Total_Bsmt_SF >= 1046
 ind_1337: t56_path4 = 1 -> x_Gr_Liv_Area <= 2237
 ind_1338: t56_path4 = 1 -> x_BsmtFin_SF_1 <= 1097
 ind_1339: t56_path5 = 1 -> const_Total_Bsmt_SF >= 1046
 ind_1340: t56_path5 = 1 -> x_Gr_Liv_Area <= 2237
 ind_1341: t56_path5 = 1 -> x_BsmtFin_SF_1 >= 1097
 ind_1342: t56_path6 = 1 -> const_Total_Bsmt_SF >= 1046
 ind_1343: t56_path6 = 1 -> x_Gr_Liv_Area >= 2237
 ind_1344: t56_path6 = 1 -> const_Misc_Val <= 1200
 ind_1345: t56_path7 = 1 -> const_Total_Bsmt_SF >= 1046
 ind_1346: t56_path7 = 1 -> x_Gr_Liv_Area >= 2237
 ind_1347: t56_path7 = 1 -> const_Misc_Val >= 1200
 ind_1348: t57_path0 = 1 -> x_Kitchen_Qual <= 3
 ind_1349: t57_path0 = 1 -> x_BsmtFin_SF_1 <= 56
 ind_1350: t57_path0 = 1 -> x_Gr_Liv_Area <= 1146
 ind_1351: t57_path1 = 1 -> x_Kitchen_Qual <= 3
 ind_1352: t57_path1 = 1 -> x_BsmtFin_SF_1 <= 56
 ind_1353: t57_path1 = 1 -> x_Gr_Liv_Area >= 1146
 ind_1354: t57_path2 = 1 -> x_Kitchen_Qual <= 3
 ind_1355: t57_path2 = 1 -> x_BsmtFin_SF_1 >= 56
 ind_1356: t57_path2 = 1 -> const_Lot_Area <= 8626
 ind_1357: t57_path3 = 1 -> x_Kitchen_Qual <= 3
 ind_1358: t57_path3 = 1 -> x_BsmtFin_SF_1 >= 56
 ind_1359: t57_path3 = 1 -> const_Lot_Area >= 8626
 ind_1360: t57_path4 = 1 -> x_Kitchen_Qual >= 3
 ind_1361: t57_path4 = 1 -> x_Gr_Liv_Area <= 2198
 ind_1362: t57_path4 = 1 -> const_Lot_Area <= 6240
 ind_1363: t57_path5 = 1 -> x_Kitchen_Qual >= 3
 ind_1364: t57_path5 = 1 -> x_Gr_Liv_Area <= 2198
 ind_1365: t57_path5 = 1 -> const_Lot_Area >= 6240
 ind_1366: t57_path6 = 1 -> x_Kitchen_Qual >= 3
 ind_1367: t57_path6 = 1 -> x_Gr_Liv_Area >= 2198
 ind_1368: t57_path6 = 1 -> const_2nd_Flr_SF <= 1405
 ind_1369: t57_path7 = 1 -> x_Kitchen_Qual >= 3
 ind_1370: t57_path7 = 1 -> x_Gr_Liv_Area >= 2198
 ind_1371: t57_path7 = 1 -> const_2nd_Flr_SF >= 1405
 ind_1372: t58_path0 = 1 -> x_Garage_Area <= 371
 ind_1373: t58_path0 = 1 -> const_Lot_Area <= 5976
 ind_1374: t58_path0 = 1 -> x_Garage_Cond <= 2
 ind_1375: t58_path1 = 1 -> x_Garage_Area <= 371
 ind_1376: t58_path1 = 1 -> const_Lot_Area <= 5976
 ind_1377: t58_path1 = 1 -> x_Garage_Cond >= 2
 ind_1378: t58_path2 = 1 -> x_Garage_Area <= 371
 ind_1379: t58_path2 = 1 -> const_Lot_Area >= 5976
 ind_1380: t58_path2 = 1 -> x_Gr_Liv_Area <= 883
 ind_1381: t58_path3 = 1 -> x_Garage_Area <= 371
 ind_1382: t58_path3 = 1 -> const_Lot_Area >= 5976
 ind_1383: t58_path3 = 1 -> x_Gr_Liv_Area >= 883
 ind_1384: t58_path4 = 1 -> x_Garage_Area >= 371
 ind_1385: t58_path4 = 1 -> x_1st_Flr_SF <= 1811
 ind_1386: t58_path4 = 1 -> x_Heating_QC <= 3
 ind_1387: t58_path5 = 1 -> x_Garage_Area >= 371
 ind_1388: t58_path5 = 1 -> x_1st_Flr_SF <= 1811
 ind_1389: t58_path5 = 1 -> x_Heating_QC >= 3
 ind_1390: t58_path6 = 1 -> x_Garage_Area >= 371
 ind_1391: t58_path6 = 1 -> x_1st_Flr_SF >= 1811
 ind_1392: t58_path6 = 1 -> x_Bsmt_Unf_SF <= 785
 ind_1393: t58_path7 = 1 -> x_Garage_Area >= 371
 ind_1394: t58_path7 = 1 -> x_1st_Flr_SF >= 1811
 ind_1395: t58_path7 = 1 -> x_Bsmt_Unf_SF >= 785
 ind_1396: t59_path0 = 1 -> const_Garage_Yr_Blt <= 1929
 ind_1397: t59_path0 = 1 -> x_Bsmt_Unf_SF <= 803
 ind_1398: t59_path0 = 1 -> const_House_Style_1Story <= 1
 ind_1399: t59_path1 = 1 -> const_Garage_Yr_Blt <= 1929
 ind_1400: t59_path1 = 1 -> x_Bsmt_Unf_SF <= 803
 ind_1401: t59_path1 = 1 -> const_House_Style_1Story >= 1
 ind_1402: t59_path2 = 1 -> const_Garage_Yr_Blt <= 1929
 ind_1403: t59_path2 = 1 -> x_Bsmt_Unf_SF >= 803
 ind_1404: t59_path2 = 1 -> x_BsmtFin_SF_1 <= 81
 ind_1405: t59_path3 = 1 -> const_Garage_Yr_Blt <= 1929
 ind_1406: t59_path3 = 1 -> x_Bsmt_Unf_SF >= 803
 ind_1407: t59_path3 = 1 -> x_BsmtFin_SF_1 >= 81
 ind_1408: t59_path4 = 1 -> const_Garage_Yr_Blt >= 1929
 ind_1409: t59_path4 = 1 -> const_Lot_Area <= 11980
 ind_1410: t59_path4 = 1 -> const_Year_Built <= 1980
 ind_1411: t59_path5 = 1 -> const_Garage_Yr_Blt >= 1929
 ind_1412: t59_path5 = 1 -> const_Lot_Area <= 11980
 ind_1413: t59_path5 = 1 -> const_Year_Built >= 1980
 ind_1414: t59_path6 = 1 -> const_Garage_Yr_Blt >= 1929
 ind_1415: t59_path6 = 1 -> const_Lot_Area >= 11980
 ind_1416: t59_path6 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_1417: t59_path7 = 1 -> const_Garage_Yr_Blt >= 1929
 ind_1418: t59_path7 = 1 -> const_Lot_Area >= 11980
 ind_1419: t59_path7 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_1420: t60_path0 = 1 -> x_Fireplace_Qu <= 1
 ind_1421: t60_path0 = 1 -> Overall_Qual_calc <= 3
 ind_1422: t60_path1 = 1 -> x_Fireplace_Qu <= 1
 ind_1423: t60_path1 = 1 -> Overall_Qual_calc >= 3
 ind_1424: t60_path1 = 1 -> const_Yr_Sold <= 2008
 ind_1425: t60_path2 = 1 -> x_Fireplace_Qu <= 1
 ind_1426: t60_path2 = 1 -> Overall_Qual_calc >= 3
 ind_1427: t60_path2 = 1 -> const_Yr_Sold >= 2008
 ind_1428: t60_path3 = 1 -> x_Fireplace_Qu >= 1
 ind_1429: t60_path3 = 1 -> x_BsmtFin_SF_1 <= 1097
 ind_1430: t60_path3 = 1 -> const_Lot_Frontage <= 65
 ind_1431: t60_path4 = 1 -> x_Fireplace_Qu >= 1
 ind_1432: t60_path4 = 1 -> x_BsmtFin_SF_1 <= 1097
 ind_1433: t60_path4 = 1 -> const_Lot_Frontage >= 65
 ind_1434: t60_path5 = 1 -> x_Fireplace_Qu >= 1
 ind_1435: t60_path5 = 1 -> x_BsmtFin_SF_1 >= 1097
 ind_1436: t60_path5 = 1 -> x_TotRms_AbvGrd <= 10
 ind_1437: t60_path6 = 1 -> x_Fireplace_Qu >= 1
 ind_1438: t60_path6 = 1 -> x_BsmtFin_SF_1 >= 1097
 ind_1439: t60_path6 = 1 -> x_TotRms_AbvGrd >= 10
 ind_1440: t61_path0 = 1 -> const_Overall_Cond <= 4
 ind_1441: t61_path0 = 1 -> const_Total_Bsmt_SF <= 1400
 ind_1442: t61_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_1443: t61_path1 = 1 -> const_Overall_Cond <= 4
 ind_1444: t61_path1 = 1 -> const_Total_Bsmt_SF <= 1400
 ind_1445: t61_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_1446: t61_path2 = 1 -> const_Overall_Cond <= 4
 ind_1447: t61_path2 = 1 -> const_Total_Bsmt_SF >= 1400
 ind_1448: t61_path3 = 1 -> const_Overall_Cond >= 4
 ind_1449: t61_path3 = 1 -> const_Total_Bsmt_SF <= 1024
 ind_1450: t61_path3 = 1 -> const_Lot_Area <= 6125
 ind_1451: t61_path4 = 1 -> const_Overall_Cond >= 4
 ind_1452: t61_path4 = 1 -> const_Total_Bsmt_SF <= 1024
 ind_1453: t61_path4 = 1 -> const_Lot_Area >= 6125
 ind_1454: t61_path5 = 1 -> const_Overall_Cond >= 4
 ind_1455: t61_path5 = 1 -> const_Total_Bsmt_SF >= 1024
 ind_1456: t61_path5 = 1 -> const_2nd_Flr_SF <= 880
 ind_1457: t61_path6 = 1 -> const_Overall_Cond >= 4
 ind_1458: t61_path6 = 1 -> const_Total_Bsmt_SF >= 1024
 ind_1459: t61_path6 = 1 -> const_2nd_Flr_SF >= 880
 ind_1460: t62_path0 = 1 -> x_Gr_Liv_Area <= 2198
 ind_1461: t62_path0 = 1 -> x_BsmtFin_SF_1 <= 329
 ind_1462: t62_path0 = 1 -> const_Year_Built <= 1984
 ind_1463: t62_path1 = 1 -> x_Gr_Liv_Area <= 2198
 ind_1464: t62_path1 = 1 -> x_BsmtFin_SF_1 <= 329
 ind_1465: t62_path1 = 1 -> const_Year_Built >= 1984
 ind_1466: t62_path2 = 1 -> x_Gr_Liv_Area <= 2198
 ind_1467: t62_path2 = 1 -> x_BsmtFin_SF_1 >= 329
 ind_1468: t62_path2 = 1 -> x_Gr_Liv_Area <= 1416
 ind_1469: t62_path3 = 1 -> x_Gr_Liv_Area <= 2198
 ind_1470: t62_path3 = 1 -> x_BsmtFin_SF_1 >= 329
 ind_1471: t62_path3 = 1 -> x_Gr_Liv_Area >= 1416
 ind_1472: t62_path4 = 1 -> x_Gr_Liv_Area >= 2198
 ind_1473: t62_path4 = 1 -> const_Condition_1_Feedr <= 1
 ind_1474: t62_path4 = 1 -> const_Lot_Area <= 13518
 ind_1475: t62_path5 = 1 -> x_Gr_Liv_Area >= 2198
 ind_1476: t62_path5 = 1 -> const_Condition_1_Feedr <= 1
 ind_1477: t62_path5 = 1 -> const_Lot_Area >= 13518
 ind_1478: t62_path6 = 1 -> x_Gr_Liv_Area >= 2198
 ind_1479: t62_path6 = 1 -> const_Condition_1_Feedr >= 1
 ind_1480: t63_path0 = 1 -> const_Year_Remod_Add <= 1987
 ind_1481: t63_path0 = 1 -> x_BsmtFin_SF_1 <= 450
 ind_1482: t63_path0 = 1 -> Overall_Qual_calc <= 3
 ind_1483: t63_path1 = 1 -> const_Year_Remod_Add <= 1987
 ind_1484: t63_path1 = 1 -> x_BsmtFin_SF_1 <= 450
 ind_1485: t63_path1 = 1 -> Overall_Qual_calc >= 3
 ind_1486: t63_path2 = 1 -> const_Year_Remod_Add <= 1987
 ind_1487: t63_path2 = 1 -> x_BsmtFin_SF_1 >= 450
 ind_1488: t63_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1489: t63_path3 = 1 -> const_Year_Remod_Add <= 1987
 ind_1490: t63_path3 = 1 -> x_BsmtFin_SF_1 >= 450
 ind_1491: t63_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1492: t63_path4 = 1 -> const_Year_Remod_Add >= 1987
 ind_1493: t63_path4 = 1 -> const_2nd_Flr_SF <= 1368
 ind_1494: t63_path4 = 1 -> const_Neighborhood_Edwards <= 1
 ind_1495: t63_path5 = 1 -> const_Year_Remod_Add >= 1987
 ind_1496: t63_path5 = 1 -> const_2nd_Flr_SF <= 1368
 ind_1497: t63_path5 = 1 -> const_Neighborhood_Edwards >= 1
 ind_1498: t63_path6 = 1 -> const_Year_Remod_Add >= 1987
 ind_1499: t63_path6 = 1 -> const_2nd_Flr_SF >= 1368
 ind_1500: t63_path6 = 1 -> const_Year_Remod_Add <= 1999
 ind_1501: t63_path7 = 1 -> const_Year_Remod_Add >= 1987
 ind_1502: t63_path7 = 1 -> const_2nd_Flr_SF >= 1368
 ind_1503: t63_path7 = 1 -> const_Year_Remod_Add >= 1999
 ind_1504: t64_path0 = 1 -> x_Gr_Liv_Area <= 1164
 ind_1505: t64_path0 = 1 -> x_BsmtFin_SF_1 <= 56
 ind_1506: t64_path0 = 1 -> x_Bsmt_Unf_SF <= 860
 ind_1507: t64_path1 = 1 -> x_Gr_Liv_Area <= 1164
 ind_1508: t64_path1 = 1 -> x_BsmtFin_SF_1 <= 56
 ind_1509: t64_path1 = 1 -> x_Bsmt_Unf_SF >= 860
 ind_1510: t64_path2 = 1 -> x_Gr_Liv_Area <= 1164
 ind_1511: t64_path2 = 1 -> x_BsmtFin_SF_1 >= 56
 ind_1512: t64_path2 = 1 -> const_Year_Remod_Add <= 1977
 ind_1513: t64_path3 = 1 -> x_Gr_Liv_Area <= 1164
 ind_1514: t64_path3 = 1 -> x_BsmtFin_SF_1 >= 56
 ind_1515: t64_path3 = 1 -> const_Year_Remod_Add >= 1977
 ind_1516: t64_path4 = 1 -> x_Gr_Liv_Area >= 1164
 ind_1517: t64_path4 = 1 -> x_Gr_Liv_Area <= 2544
 ind_1518: t64_path4 = 1 -> x_BsmtFin_SF_1 <= 292
 ind_1519: t64_path5 = 1 -> x_Gr_Liv_Area >= 1164
 ind_1520: t64_path5 = 1 -> x_Gr_Liv_Area <= 2544
 ind_1521: t64_path5 = 1 -> x_BsmtFin_SF_1 >= 292
 ind_1522: t64_path6 = 1 -> x_Gr_Liv_Area >= 1164
 ind_1523: t64_path6 = 1 -> x_Gr_Liv_Area >= 2544
 ind_1524: t64_path6 = 1 -> x_Gr_Liv_Area <= 3086
 ind_1525: t64_path7 = 1 -> x_Gr_Liv_Area >= 1164
 ind_1526: t64_path7 = 1 -> x_Gr_Liv_Area >= 2544
 ind_1527: t64_path7 = 1 -> x_Gr_Liv_Area >= 3086
 ind_1528: t65_path0 = 1 -> const_Total_Bsmt_SF <= 648
 ind_1529: t65_path0 = 1 -> Overall_Qual_calc <= 4
 ind_1530: t65_path0 = 1 -> x_ex2_is_VinylSd <= 1
 ind_1531: t65_path1 = 1 -> const_Total_Bsmt_SF <= 648
 ind_1532: t65_path1 = 1 -> Overall_Qual_calc <= 4
 ind_1533: t65_path1 = 1 -> x_ex2_is_VinylSd >= 1
 ind_1534: t65_path2 = 1 -> const_Total_Bsmt_SF <= 648
 ind_1535: t65_path2 = 1 -> Overall_Qual_calc >= 4
 ind_1536: t65_path2 = 1 -> const_Lot_Area <= 6762
 ind_1537: t65_path3 = 1 -> const_Total_Bsmt_SF <= 648
 ind_1538: t65_path3 = 1 -> Overall_Qual_calc >= 4
 ind_1539: t65_path3 = 1 -> const_Lot_Area >= 6762
 ind_1540: t65_path4 = 1 -> const_Total_Bsmt_SF >= 648
 ind_1541: t65_path4 = 1 -> x_1st_Flr_SF <= 1830
 ind_1542: t65_path4 = 1 -> const_Year_Remod_Add <= 1963
 ind_1543: t65_path5 = 1 -> const_Total_Bsmt_SF >= 648
 ind_1544: t65_path5 = 1 -> x_1st_Flr_SF <= 1830
 ind_1545: t65_path5 = 1 -> const_Year_Remod_Add >= 1963
 ind_1546: t65_path6 = 1 -> const_Total_Bsmt_SF >= 648
 ind_1547: t65_path6 = 1 -> x_1st_Flr_SF >= 1830
 ind_1548: t65_path6 = 1 -> const_Lot_Area <= 11388
 ind_1549: t65_path7 = 1 -> const_Total_Bsmt_SF >= 648
 ind_1550: t65_path7 = 1 -> x_1st_Flr_SF >= 1830
 ind_1551: t65_path7 = 1 -> const_Lot_Area >= 11388
 ind_1552: t66_path0 = 1 -> x_paved_drive_is_N <= 1
 ind_1553: t66_path0 = 1 -> const_Yr_Sold <= 2008
 ind_1554: t66_path0 = 1 -> x_Gr_Liv_Area <= 1968
 ind_1555: t66_path1 = 1 -> x_paved_drive_is_N <= 1
 ind_1556: t66_path1 = 1 -> const_Yr_Sold <= 2008
 ind_1557: t66_path1 = 1 -> x_Gr_Liv_Area >= 1968
 ind_1558: t66_path2 = 1 -> x_paved_drive_is_N <= 1
 ind_1559: t66_path2 = 1 -> const_Yr_Sold >= 2008
 ind_1560: t66_path2 = 1 -> x_Heating_QC <= 3
 ind_1561: t66_path3 = 1 -> x_paved_drive_is_N <= 1
 ind_1562: t66_path3 = 1 -> const_Yr_Sold >= 2008
 ind_1563: t66_path3 = 1 -> x_Heating_QC >= 3
 ind_1564: t66_path4 = 1 -> x_paved_drive_is_N >= 1
 ind_1565: t66_path4 = 1 -> const_Yr_Sold <= 2008
 ind_1566: t66_path4 = 1 -> x_Gr_Liv_Area <= 1488
 ind_1567: t66_path5 = 1 -> x_paved_drive_is_N >= 1
 ind_1568: t66_path5 = 1 -> const_Yr_Sold <= 2008
 ind_1569: t66_path5 = 1 -> x_Gr_Liv_Area >= 1488
 ind_1570: t66_path6 = 1 -> x_paved_drive_is_N >= 1
 ind_1571: t66_path6 = 1 -> const_Yr_Sold >= 2008
 ind_1572: t66_path6 = 1 -> x_Gr_Liv_Area <= 1105
 ind_1573: t66_path7 = 1 -> x_paved_drive_is_N >= 1
 ind_1574: t66_path7 = 1 -> const_Yr_Sold >= 2008
 ind_1575: t66_path7 = 1 -> x_Gr_Liv_Area >= 1105
 ind_1576: t67_path0 = 1 -> Overall_Qual_calc <= 4
 ind_1577: t67_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_1578: t67_path1 = 1 -> Overall_Qual_calc <= 4
 ind_1579: t67_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_1580: t67_path2 = 1 -> Overall_Qual_calc >= 4
 ind_1581: t67_path2 = 1 -> const_Year_Remod_Add <= 1993
 ind_1582: t67_path2 = 1 -> const_Yr_Sold <= 2008
 ind_1583: t67_path3 = 1 -> Overall_Qual_calc >= 4
 ind_1584: t67_path3 = 1 -> const_Year_Remod_Add <= 1993
 ind_1585: t67_path3 = 1 -> const_Yr_Sold >= 2008
 ind_1586: t67_path4 = 1 -> Overall_Qual_calc >= 4
 ind_1587: t67_path4 = 1 -> const_Year_Remod_Add >= 1993
 ind_1588: t67_path4 = 1 -> x_Kitchen_Qual <= 4
 ind_1589: t67_path5 = 1 -> Overall_Qual_calc >= 4
 ind_1590: t67_path5 = 1 -> const_Year_Remod_Add >= 1993
 ind_1591: t67_path5 = 1 -> x_Kitchen_Qual >= 4
 ind_1592: t68_path0 = 1 -> x_Fireplace_Qu <= 1
 ind_1593: t68_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_1594: t68_path0 = 1 -> const_Lot_Area <= 6120
 ind_1595: t68_path1 = 1 -> x_Fireplace_Qu <= 1
 ind_1596: t68_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_1597: t68_path1 = 1 -> const_Lot_Area >= 6120
 ind_1598: t68_path2 = 1 -> x_Fireplace_Qu <= 1
 ind_1599: t68_path2 = 1 -> const_MS_SubClass_30 >= 1
 ind_1600: t68_path2 = 1 -> x_fence_is_No_aplica <= 1
 ind_1601: t68_path3 = 1 -> x_Fireplace_Qu <= 1
 ind_1602: t68_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_1603: t68_path3 = 1 -> x_fence_is_No_aplica >= 1
 ind_1604: t68_path4 = 1 -> x_Fireplace_Qu >= 1
 ind_1605: t68_path4 = 1 -> const_Yr_Sold <= 2008
 ind_1606: t68_path4 = 1 -> x_1st_Flr_SF <= 1489
 ind_1607: t68_path5 = 1 -> x_Fireplace_Qu >= 1
 ind_1608: t68_path5 = 1 -> const_Yr_Sold <= 2008
 ind_1609: t68_path5 = 1 -> x_1st_Flr_SF >= 1489
 ind_1610: t68_path6 = 1 -> x_Fireplace_Qu >= 1
 ind_1611: t68_path6 = 1 -> const_Yr_Sold >= 2008
 ind_1612: t68_path6 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_1613: t68_path7 = 1 -> x_Fireplace_Qu >= 1
 ind_1614: t68_path7 = 1 -> const_Yr_Sold >= 2008
 ind_1615: t68_path7 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_1616: t69_path0 = 1 -> const_Overall_Cond <= 5
 ind_1617: t69_path0 = 1 -> const_Garage_Yr_Blt <= 1995
 ind_1618: t69_path0 = 1 -> x_Open_Porch_SF <= 72
 ind_1619: t69_path1 = 1 -> const_Overall_Cond <= 5
 ind_1620: t69_path1 = 1 -> const_Garage_Yr_Blt <= 1995
 ind_1621: t69_path1 = 1 -> x_Open_Porch_SF >= 72
 ind_1622: t69_path2 = 1 -> const_Overall_Cond <= 5
 ind_1623: t69_path2 = 1 -> const_Garage_Yr_Blt >= 1995
 ind_1624: t69_path3 = 1 -> const_Overall_Cond >= 5
 ind_1625: t69_path3 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_1626: t69_path3 = 1 -> const_Lot_Area <= 7200
 ind_1627: t69_path4 = 1 -> const_Overall_Cond >= 5
 ind_1628: t69_path4 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_1629: t69_path4 = 1 -> const_Lot_Area >= 7200
 ind_1630: t69_path5 = 1 -> const_Overall_Cond >= 5
 ind_1631: t69_path5 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_1632: t69_path5 = 1 -> x_1st_Flr_SF <= 948
 ind_1633: t69_path6 = 1 -> const_Overall_Cond >= 5
 ind_1634: t69_path6 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_1635: t69_path6 = 1 -> x_1st_Flr_SF >= 948
 ind_1636: t70_path0 = 1 -> const_Year_Remod_Add <= 1984
 ind_1637: t70_path0 = 1 -> const_MS_Zoning_RL <= 1
 ind_1638: t70_path0 = 1 -> const_Yr_Sold <= 2008
 ind_1639: t70_path1 = 1 -> const_Year_Remod_Add <= 1984
 ind_1640: t70_path1 = 1 -> const_MS_Zoning_RL <= 1
 ind_1641: t70_path1 = 1 -> const_Yr_Sold >= 2008
 ind_1642: t70_path2 = 1 -> const_Year_Remod_Add <= 1984
 ind_1643: t70_path2 = 1 -> const_MS_Zoning_RL >= 1
 ind_1644: t70_path2 = 1 -> x_Bsmt_Unf_SF <= 520
 ind_1645: t70_path3 = 1 -> const_Year_Remod_Add <= 1984
 ind_1646: t70_path3 = 1 -> const_MS_Zoning_RL >= 1
 ind_1647: t70_path3 = 1 -> x_Bsmt_Unf_SF >= 520
 ind_1648: t70_path4 = 1 -> const_Year_Remod_Add >= 1984
 ind_1649: t70_path4 = 1 -> const_Neighborhood_OldTown <= 1
 ind_1650: t70_path4 = 1 -> const_2nd_Flr_SF <= 1296
 ind_1651: t70_path5 = 1 -> const_Year_Remod_Add >= 1984
 ind_1652: t70_path5 = 1 -> const_Neighborhood_OldTown <= 1
 ind_1653: t70_path5 = 1 -> const_2nd_Flr_SF >= 1296
 ind_1654: t70_path6 = 1 -> const_Year_Remod_Add >= 1984
 ind_1655: t70_path6 = 1 -> const_Neighborhood_OldTown >= 1
 ind_1656: t70_path6 = 1 -> const_Foundation_BrkTil <= 1
 ind_1657: t70_path7 = 1 -> const_Year_Remod_Add >= 1984
 ind_1658: t70_path7 = 1 -> const_Neighborhood_OldTown >= 1
 ind_1659: t70_path7 = 1 -> const_Foundation_BrkTil >= 1
 ind_1660: t71_path0 = 1 -> x_Gr_Liv_Area <= 1922
 ind_1661: t71_path0 = 1 -> const_Bsmt_Qual <= 2
 ind_1662: t71_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1663: t71_path1 = 1 -> x_Gr_Liv_Area <= 1922
 ind_1664: t71_path1 = 1 -> const_Bsmt_Qual <= 2
 ind_1665: t71_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1666: t71_path2 = 1 -> x_Gr_Liv_Area <= 1922
 ind_1667: t71_path2 = 1 -> const_Bsmt_Qual >= 2
 ind_1668: t71_path2 = 1 -> x_BsmtFin_SF_1 <= 292
 ind_1669: t71_path3 = 1 -> x_Gr_Liv_Area <= 1922
 ind_1670: t71_path3 = 1 -> const_Bsmt_Qual >= 2
 ind_1671: t71_path3 = 1 -> x_BsmtFin_SF_1 >= 292
 ind_1672: t71_path4 = 1 -> x_Gr_Liv_Area >= 1922
 ind_1673: t71_path4 = 1 -> const_2nd_Flr_SF <= 1368
 ind_1674: t71_path4 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_1675: t71_path5 = 1 -> x_Gr_Liv_Area >= 1922
 ind_1676: t71_path5 = 1 -> const_2nd_Flr_SF <= 1368
 ind_1677: t71_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_1678: t71_path6 = 1 -> x_Gr_Liv_Area >= 1922
 ind_1679: t71_path6 = 1 -> const_2nd_Flr_SF >= 1368
 ind_1680: t71_path6 = 1 -> const_Lot_Area <= 12384
 ind_1681: t71_path7 = 1 -> x_Gr_Liv_Area >= 1922
 ind_1682: t71_path7 = 1 -> const_2nd_Flr_SF >= 1368
 ind_1683: t71_path7 = 1 -> const_Lot_Area >= 12384
 ind_1684: t72_path0 = 1 -> const_Yr_Sold <= 2008
 ind_1685: t72_path0 = 1 -> x_1st_Flr_SF <= 1320
 ind_1686: t72_path0 = 1 -> const_MS_SubClass_160 <= 1
 ind_1687: t72_path1 = 1 -> const_Yr_Sold <= 2008
 ind_1688: t72_path1 = 1 -> x_1st_Flr_SF <= 1320
 ind_1689: t72_path1 = 1 -> const_MS_SubClass_160 >= 1
 ind_1690: t72_path2 = 1 -> const_Yr_Sold <= 2008
 ind_1691: t72_path2 = 1 -> x_1st_Flr_SF >= 1320
 ind_1692: t72_path2 = 1 -> const_Year_Remod_Add <= 1991
 ind_1693: t72_path3 = 1 -> const_Yr_Sold <= 2008
 ind_1694: t72_path3 = 1 -> x_1st_Flr_SF >= 1320
 ind_1695: t72_path3 = 1 -> const_Year_Remod_Add >= 1991
 ind_1696: t72_path4 = 1 -> const_Yr_Sold >= 2008
 ind_1697: t72_path4 = 1 -> const_Year_Remod_Add <= 1984
 ind_1698: t72_path4 = 1 -> const_Sale_Condition_Normal <= 1
 ind_1699: t72_path5 = 1 -> const_Yr_Sold >= 2008
 ind_1700: t72_path5 = 1 -> const_Year_Remod_Add <= 1984
 ind_1701: t72_path5 = 1 -> const_Sale_Condition_Normal >= 1
 ind_1702: t72_path6 = 1 -> const_Yr_Sold >= 2008
 ind_1703: t72_path6 = 1 -> const_Year_Remod_Add >= 1984
 ind_1704: t72_path6 = 1 -> x_Gr_Liv_Area <= 2345
 ind_1705: t72_path7 = 1 -> const_Yr_Sold >= 2008
 ind_1706: t72_path7 = 1 -> const_Year_Remod_Add >= 1984
 ind_1707: t72_path7 = 1 -> x_Gr_Liv_Area >= 2345
 ind_1708: t73_path0 = 1 -> const_Overall_Cond <= 4
 ind_1709: t73_path0 = 1 -> x_Bsmt_Unf_SF <= 414
 ind_1710: t73_path1 = 1 -> const_Overall_Cond <= 4
 ind_1711: t73_path1 = 1 -> x_Bsmt_Unf_SF >= 414
 ind_1712: t73_path1 = 1 -> x_Exter_Qual <= 2
 ind_1713: t73_path2 = 1 -> const_Overall_Cond <= 4
 ind_1714: t73_path2 = 1 -> x_Bsmt_Unf_SF >= 414
 ind_1715: t73_path2 = 1 -> x_Exter_Qual >= 2
 ind_1716: t73_path3 = 1 -> const_Overall_Cond >= 4
 ind_1717: t73_path3 = 1 -> const_2nd_Flr_SF <= 1296
 ind_1718: t73_path3 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_1719: t73_path4 = 1 -> const_Overall_Cond >= 4
 ind_1720: t73_path4 = 1 -> const_2nd_Flr_SF <= 1296
 ind_1721: t73_path4 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_1722: t73_path5 = 1 -> const_Overall_Cond >= 4
 ind_1723: t73_path5 = 1 -> const_2nd_Flr_SF >= 1296
 ind_1724: t73_path5 = 1 -> x_Bsmt_Unf_SF <= 426
 ind_1725: t73_path6 = 1 -> const_Overall_Cond >= 4
 ind_1726: t73_path6 = 1 -> const_2nd_Flr_SF >= 1296
 ind_1727: t73_path6 = 1 -> x_Bsmt_Unf_SF >= 426
 ind_1728: t74_path0 = 1 -> Overall_Qual_calc <= 8
 ind_1729: t74_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_1730: t74_path0 = 1 -> const_Yr_Sold <= 2008
 ind_1731: t74_path1 = 1 -> Overall_Qual_calc <= 8
 ind_1732: t74_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_1733: t74_path1 = 1 -> const_Yr_Sold >= 2008
 ind_1734: t74_path2 = 1 -> Overall_Qual_calc <= 8
 ind_1735: t74_path2 = 1 -> const_MS_SubClass_30 >= 1
 ind_1736: t74_path2 = 1 -> x_1st_Flr_SF <= 970
 ind_1737: t74_path3 = 1 -> Overall_Qual_calc <= 8
 ind_1738: t74_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_1739: t74_path3 = 1 -> x_1st_Flr_SF >= 970
 ind_1740: t74_path4 = 1 -> Overall_Qual_calc >= 8
 ind_1741: t74_path4 = 1 -> const_2nd_Flr_SF <= 1368
 ind_1742: t74_path4 = 1 -> x_1st_Flr_SF <= 1464
 ind_1743: t74_path5 = 1 -> Overall_Qual_calc >= 8
 ind_1744: t74_path5 = 1 -> const_2nd_Flr_SF <= 1368
 ind_1745: t74_path5 = 1 -> x_1st_Flr_SF >= 1464
 ind_1746: t74_path6 = 1 -> Overall_Qual_calc >= 8
 ind_1747: t74_path6 = 1 -> const_2nd_Flr_SF >= 1368
 ind_1748: t75_path0 = 1 -> const_Overall_Cond <= 4
 ind_1749: t75_path0 = 1 -> x_Wood_Deck_SF <= 22
 ind_1750: t75_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_1751: t75_path1 = 1 -> const_Overall_Cond <= 4
 ind_1752: t75_path1 = 1 -> x_Wood_Deck_SF <= 22
 ind_1753: t75_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_1754: t75_path2 = 1 -> const_Overall_Cond <= 4
 ind_1755: t75_path2 = 1 -> x_Wood_Deck_SF >= 22
 ind_1756: t75_path3 = 1 -> const_Overall_Cond >= 4
 ind_1757: t75_path3 = 1 -> x_Gr_Liv_Area <= 1712
 ind_1758: t75_path3 = 1 -> const_Garage_Yr_Blt <= 1921
 ind_1759: t75_path4 = 1 -> const_Overall_Cond >= 4
 ind_1760: t75_path4 = 1 -> x_Gr_Liv_Area <= 1712
 ind_1761: t75_path4 = 1 -> const_Garage_Yr_Blt >= 1921
 ind_1762: t75_path5 = 1 -> const_Overall_Cond >= 4
 ind_1763: t75_path5 = 1 -> x_Gr_Liv_Area >= 1712
 ind_1764: t75_path5 = 1 -> x_BsmtFin_SF_1 <= 188
 ind_1765: t75_path6 = 1 -> const_Overall_Cond >= 4
 ind_1766: t75_path6 = 1 -> x_Gr_Liv_Area >= 1712
 ind_1767: t75_path6 = 1 -> x_BsmtFin_SF_1 >= 188
 ind_1768: t76_path0 = 1 -> const_Year_Remod_Add <= 1984
 ind_1769: t76_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_1770: t76_path0 = 1 -> const_Yr_Sold <= 2008
 ind_1771: t76_path1 = 1 -> const_Year_Remod_Add <= 1984
 ind_1772: t76_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_1773: t76_path1 = 1 -> const_Yr_Sold >= 2008
 ind_1774: t76_path2 = 1 -> const_Year_Remod_Add <= 1984
 ind_1775: t76_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_1776: t76_path2 = 1 -> const_Lot_Frontage <= 72
 ind_1777: t76_path3 = 1 -> const_Year_Remod_Add <= 1984
 ind_1778: t76_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_1779: t76_path3 = 1 -> const_Lot_Frontage >= 72
 ind_1780: t76_path4 = 1 -> const_Year_Remod_Add >= 1984
 ind_1781: t76_path4 = 1 -> const_Yr_Sold <= 2007
 ind_1782: t76_path4 = 1 -> x_Wood_Deck_SF <= 193
 ind_1783: t76_path5 = 1 -> const_Year_Remod_Add >= 1984
 ind_1784: t76_path5 = 1 -> const_Yr_Sold <= 2007
 ind_1785: t76_path5 = 1 -> x_Wood_Deck_SF >= 193
 ind_1786: t76_path6 = 1 -> const_Year_Remod_Add >= 1984
 ind_1787: t76_path6 = 1 -> const_Yr_Sold >= 2007
 ind_1788: t76_path6 = 1 -> x_TotRms_AbvGrd <= 12
 ind_1789: t76_path7 = 1 -> const_Year_Remod_Add >= 1984
 ind_1790: t76_path7 = 1 -> const_Yr_Sold >= 2007
 ind_1791: t76_path7 = 1 -> x_TotRms_AbvGrd >= 12
 ind_1792: t77_path0 = 1 -> const_Overall_Cond <= 5
 ind_1793: t77_path0 = 1 -> x_Open_Porch_SF <= 158
 ind_1794: t77_path0 = 1 -> x_Bsmt_Unf_SF <= 402
 ind_1795: t77_path1 = 1 -> const_Overall_Cond <= 5
 ind_1796: t77_path1 = 1 -> x_Open_Porch_SF <= 158
 ind_1797: t77_path1 = 1 -> x_Bsmt_Unf_SF >= 402
 ind_1798: t77_path2 = 1 -> const_Overall_Cond <= 5
 ind_1799: t77_path2 = 1 -> x_Open_Porch_SF >= 158
 ind_1800: t77_path3 = 1 -> const_Overall_Cond >= 5
 ind_1801: t77_path3 = 1 -> const_Yr_Sold <= 2008
 ind_1802: t77_path3 = 1 -> const_Lot_Area <= 9590
 ind_1803: t77_path4 = 1 -> const_Overall_Cond >= 5
 ind_1804: t77_path4 = 1 -> const_Yr_Sold <= 2008
 ind_1805: t77_path4 = 1 -> const_Lot_Area >= 9590
 ind_1806: t77_path5 = 1 -> const_Overall_Cond >= 5
 ind_1807: t77_path5 = 1 -> const_Yr_Sold >= 2008
 ind_1808: t77_path5 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_1809: t77_path6 = 1 -> const_Overall_Cond >= 5
 ind_1810: t77_path6 = 1 -> const_Yr_Sold >= 2008
 ind_1811: t77_path6 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_1812: t78_path0 = 1 -> const_Yr_Sold <= 2008
 ind_1813: t78_path0 = 1 -> x_Gr_Liv_Area <= 1734
 ind_1814: t78_path0 = 1 -> const_Lot_Frontage <= 31
 ind_1815: t78_path1 = 1 -> const_Yr_Sold <= 2008
 ind_1816: t78_path1 = 1 -> x_Gr_Liv_Area <= 1734
 ind_1817: t78_path1 = 1 -> const_Lot_Frontage >= 31
 ind_1818: t78_path2 = 1 -> const_Yr_Sold <= 2008
 ind_1819: t78_path2 = 1 -> x_Gr_Liv_Area >= 1734
 ind_1820: t78_path2 = 1 -> const_Year_Built <= 1901
 ind_1821: t78_path3 = 1 -> const_Yr_Sold <= 2008
 ind_1822: t78_path3 = 1 -> x_Gr_Liv_Area >= 1734
 ind_1823: t78_path3 = 1 -> const_Year_Built >= 1901
 ind_1824: t78_path4 = 1 -> const_Yr_Sold >= 2008
 ind_1825: t78_path4 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1826: t78_path4 = 1 -> x_Gr_Liv_Area <= 1118
 ind_1827: t78_path5 = 1 -> const_Yr_Sold >= 2008
 ind_1828: t78_path5 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1829: t78_path5 = 1 -> x_Gr_Liv_Area >= 1118
 ind_1830: t78_path6 = 1 -> const_Yr_Sold >= 2008
 ind_1831: t78_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1832: t78_path6 = 1 -> Overall_Qual_calc <= 5
 ind_1833: t78_path7 = 1 -> const_Yr_Sold >= 2008
 ind_1834: t78_path7 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1835: t78_path7 = 1 -> Overall_Qual_calc >= 5
 ind_1836: t79_path0 = 1 -> const_Overall_Cond <= 4
 ind_1837: t79_path0 = 1 -> x_Bedroom_AbvGr <= 4
 ind_1838: t79_path0 = 1 -> x_Heating_QC <= 3
 ind_1839: t79_path1 = 1 -> const_Overall_Cond <= 4
 ind_1840: t79_path1 = 1 -> x_Bedroom_AbvGr <= 4
 ind_1841: t79_path1 = 1 -> x_Heating_QC >= 3
 ind_1842: t79_path2 = 1 -> const_Overall_Cond <= 4
 ind_1843: t79_path2 = 1 -> x_Bedroom_AbvGr >= 4
 ind_1844: t79_path3 = 1 -> const_Overall_Cond >= 4
 ind_1845: t79_path3 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_1846: t79_path3 = 1 -> const_Total_Bsmt_SF <= 1046
 ind_1847: t79_path4 = 1 -> const_Overall_Cond >= 4
 ind_1848: t79_path4 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_1849: t79_path4 = 1 -> const_Total_Bsmt_SF >= 1046
 ind_1850: t79_path5 = 1 -> const_Overall_Cond >= 4
 ind_1851: t79_path5 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_1852: t79_path5 = 1 -> const_Year_Built <= 1919
 ind_1853: t79_path6 = 1 -> const_Overall_Cond >= 4
 ind_1854: t79_path6 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_1855: t79_path6 = 1 -> const_Year_Built >= 1919
 ind_1856: t80_path0 = 1 -> const_Overall_Cond <= 5
 ind_1857: t80_path0 = 1 -> x_Screen_Porch <= 104
 ind_1858: t80_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1859: t80_path1 = 1 -> const_Overall_Cond <= 5
 ind_1860: t80_path1 = 1 -> x_Screen_Porch <= 104
 ind_1861: t80_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1862: t80_path2 = 1 -> const_Overall_Cond <= 5
 ind_1863: t80_path2 = 1 -> x_Screen_Porch >= 104
 ind_1864: t80_path3 = 1 -> const_Overall_Cond >= 5
 ind_1865: t80_path3 = 1 -> const_Lot_Area <= 11900
 ind_1866: t80_path3 = 1 -> x_BsmtFin_SF_1 <= 170
 ind_1867: t80_path4 = 1 -> const_Overall_Cond >= 5
 ind_1868: t80_path4 = 1 -> const_Lot_Area <= 11900
 ind_1869: t80_path4 = 1 -> x_BsmtFin_SF_1 >= 170
 ind_1870: t80_path5 = 1 -> const_Overall_Cond >= 5
 ind_1871: t80_path5 = 1 -> const_Lot_Area >= 11900
 ind_1872: t80_path5 = 1 -> x_1st_Flr_SF <= 1508
 ind_1873: t80_path6 = 1 -> const_Overall_Cond >= 5
 ind_1874: t80_path6 = 1 -> const_Lot_Area >= 11900
 ind_1875: t80_path6 = 1 -> x_1st_Flr_SF >= 1508
 ind_1876: t81_path0 = 1 -> const_Overall_Cond <= 3
 ind_1877: t81_path1 = 1 -> const_Overall_Cond >= 3
 ind_1878: t81_path1 = 1 -> x_Fireplace_Qu <= 1
 ind_1879: t81_path1 = 1 -> const_Lot_Area <= 5976
 ind_1880: t81_path2 = 1 -> const_Overall_Cond >= 3
 ind_1881: t81_path2 = 1 -> x_Fireplace_Qu <= 1
 ind_1882: t81_path2 = 1 -> const_Lot_Area >= 5976
 ind_1883: t81_path3 = 1 -> const_Overall_Cond >= 3
 ind_1884: t81_path3 = 1 -> x_Fireplace_Qu >= 1
 ind_1885: t81_path3 = 1 -> const_Lot_Area <= 17227
 ind_1886: t81_path4 = 1 -> const_Overall_Cond >= 3
 ind_1887: t81_path4 = 1 -> x_Fireplace_Qu >= 1
 ind_1888: t81_path4 = 1 -> const_Lot_Area >= 17227
 ind_1889: t82_path0 = 1 -> x_Gr_Liv_Area <= 1800
 ind_1890: t82_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_1891: t82_path0 = 1 -> x_BsmtFin_SF_1 <= 360
 ind_1892: t82_path1 = 1 -> x_Gr_Liv_Area <= 1800
 ind_1893: t82_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_1894: t82_path1 = 1 -> x_BsmtFin_SF_1 >= 360
 ind_1895: t82_path2 = 1 -> x_Gr_Liv_Area <= 1800
 ind_1896: t82_path2 = 1 -> const_MS_SubClass_30 >= 1
 ind_1897: t82_path2 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_1898: t82_path3 = 1 -> x_Gr_Liv_Area <= 1800
 ind_1899: t82_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_1900: t82_path3 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_1901: t82_path4 = 1 -> x_Gr_Liv_Area >= 1800
 ind_1902: t82_path4 = 1 -> x_Screen_Porch <= 40
 ind_1903: t82_path4 = 1 -> x_Full_Bath <= 3
 ind_1904: t82_path5 = 1 -> x_Gr_Liv_Area >= 1800
 ind_1905: t82_path5 = 1 -> x_Screen_Porch <= 40
 ind_1906: t82_path5 = 1 -> x_Full_Bath >= 3
 ind_1907: t82_path6 = 1 -> x_Gr_Liv_Area >= 1800
 ind_1908: t82_path6 = 1 -> x_Screen_Porch >= 40
 ind_1909: t83_path0 = 1 -> Overall_Qual_calc <= 3
 ind_1910: t83_path1 = 1 -> Overall_Qual_calc >= 3
 ind_1911: t83_path1 = 1 -> const_Total_Bsmt_SF <= 1053
 ind_1912: t83_path1 = 1 -> const_Lot_Area <= 5976
 ind_1913: t83_path2 = 1 -> Overall_Qual_calc >= 3
 ind_1914: t83_path2 = 1 -> const_Total_Bsmt_SF <= 1053
 ind_1915: t83_path2 = 1 -> const_Lot_Area >= 5976
 ind_1916: t83_path3 = 1 -> Overall_Qual_calc >= 3
 ind_1917: t83_path3 = 1 -> const_Total_Bsmt_SF >= 1053
 ind_1918: t83_path3 = 1 -> x_Mas_Vnr_Area <= 456
 ind_1919: t83_path4 = 1 -> Overall_Qual_calc >= 3
 ind_1920: t83_path4 = 1 -> const_Total_Bsmt_SF >= 1053
 ind_1921: t83_path4 = 1 -> x_Mas_Vnr_Area >= 456
 ind_1922: t84_path0 = 1 -> const_Overall_Cond <= 5
 ind_1923: t84_path0 = 1 -> x_Open_Porch_SF <= 72
 ind_1924: t84_path0 = 1 -> x_Screen_Porch <= 104
 ind_1925: t84_path1 = 1 -> const_Overall_Cond <= 5
 ind_1926: t84_path1 = 1 -> x_Open_Porch_SF <= 72
 ind_1927: t84_path1 = 1 -> x_Screen_Porch >= 104
 ind_1928: t84_path2 = 1 -> const_Overall_Cond <= 5
 ind_1929: t84_path2 = 1 -> x_Open_Porch_SF >= 72
 ind_1930: t84_path3 = 1 -> const_Overall_Cond >= 5
 ind_1931: t84_path3 = 1 -> x_Gr_Liv_Area <= 988
 ind_1932: t84_path3 = 1 -> x_fence_is_GdWo <= 1
 ind_1933: t84_path4 = 1 -> const_Overall_Cond >= 5
 ind_1934: t84_path4 = 1 -> x_Gr_Liv_Area <= 988
 ind_1935: t84_path4 = 1 -> x_fence_is_GdWo >= 1
 ind_1936: t84_path5 = 1 -> const_Overall_Cond >= 5
 ind_1937: t84_path5 = 1 -> x_Gr_Liv_Area >= 988
 ind_1938: t84_path5 = 1 -> x_Full_Bath <= 3
 ind_1939: t84_path6 = 1 -> const_Overall_Cond >= 5
 ind_1940: t84_path6 = 1 -> x_Gr_Liv_Area >= 988
 ind_1941: t84_path6 = 1 -> x_Full_Bath >= 3
 ind_1942: t85_path0 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_1943: t85_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1944: t85_path0 = 1 -> const_Total_Bsmt_SF <= 663
 ind_1945: t85_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_1946: t85_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_1947: t85_path1 = 1 -> const_Total_Bsmt_SF >= 663
 ind_1948: t85_path2 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_1949: t85_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1950: t85_path2 = 1 -> Overall_Qual_calc <= 5
 ind_1951: t85_path3 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_1952: t85_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_1953: t85_path3 = 1 -> Overall_Qual_calc >= 5
 ind_1954: t85_path4 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_1955: t85_path4 = 1 -> const_Total_Bsmt_SF <= 1438
 ind_1956: t85_path5 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_1957: t85_path5 = 1 -> const_Total_Bsmt_SF >= 1438
 ind_1958: t85_path5 = 1 -> x_Bsmt_Unf_SF <= 996
 ind_1959: t85_path6 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_1960: t85_path6 = 1 -> const_Total_Bsmt_SF >= 1438
 ind_1961: t85_path6 = 1 -> x_Bsmt_Unf_SF >= 996
 ind_1962: t86_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_1963: t86_path0 = 1 -> const_Total_Bsmt_SF <= 576
 ind_1964: t86_path0 = 1 -> const_Garage_Finish_Fin <= 1
 ind_1965: t86_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_1966: t86_path1 = 1 -> const_Total_Bsmt_SF <= 576
 ind_1967: t86_path1 = 1 -> const_Garage_Finish_Fin >= 1
 ind_1968: t86_path2 = 1 -> const_MS_SubClass_30 <= 1
 ind_1969: t86_path2 = 1 -> const_Total_Bsmt_SF >= 576
 ind_1970: t86_path2 = 1 -> const_Lot_Area <= 13811
 ind_1971: t86_path3 = 1 -> const_MS_SubClass_30 <= 1
 ind_1972: t86_path3 = 1 -> const_Total_Bsmt_SF >= 576
 ind_1973: t86_path3 = 1 -> const_Lot_Area >= 13811
 ind_1974: t86_path4 = 1 -> const_MS_SubClass_30 >= 1
 ind_1975: t86_path4 = 1 -> const_Total_Bsmt_SF <= 761
 ind_1976: t86_path4 = 1 -> const_Total_Bsmt_SF <= 663
 ind_1977: t86_path5 = 1 -> const_MS_SubClass_30 >= 1
 ind_1978: t86_path5 = 1 -> const_Total_Bsmt_SF <= 761
 ind_1979: t86_path5 = 1 -> const_Total_Bsmt_SF >= 663
 ind_1980: t86_path6 = 1 -> const_MS_SubClass_30 >= 1
 ind_1981: t86_path6 = 1 -> const_Total_Bsmt_SF >= 761
 ind_1982: t87_path0 = 1 -> const_Overall_Cond <= 4
 ind_1983: t87_path0 = 1 -> const_Lot_Shape_IR1 <= 1
 ind_1984: t87_path0 = 1 -> heat_is_GasA <= 1
 ind_1985: t87_path1 = 1 -> const_Overall_Cond <= 4
 ind_1986: t87_path1 = 1 -> const_Lot_Shape_IR1 <= 1
 ind_1987: t87_path1 = 1 -> heat_is_GasA >= 1
 ind_1988: t87_path2 = 1 -> const_Overall_Cond <= 4
 ind_1989: t87_path2 = 1 -> const_Lot_Shape_IR1 >= 1
 ind_1990: t87_path3 = 1 -> const_Overall_Cond >= 4
 ind_1991: t87_path3 = 1 -> const_Yr_Sold <= 2008
 ind_1992: t87_path3 = 1 -> const_2nd_Flr_SF <= 1405
 ind_1993: t87_path4 = 1 -> const_Overall_Cond >= 4
 ind_1994: t87_path4 = 1 -> const_Yr_Sold <= 2008
 ind_1995: t87_path4 = 1 -> const_2nd_Flr_SF >= 1405
 ind_1996: t87_path5 = 1 -> const_Overall_Cond >= 4
 ind_1997: t87_path5 = 1 -> const_Yr_Sold >= 2008
 ind_1998: t87_path5 = 1 -> x_Gr_Liv_Area <= 1146
 ind_1999: t87_path6 = 1 -> const_Overall_Cond >= 4
 ind_2000: t87_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2001: t87_path6 = 1 -> x_Gr_Liv_Area >= 1146
 ind_2002: t88_path0 = 1 -> const_Overall_Cond <= 4
 ind_2003: t88_path0 = 1 -> const_Total_Bsmt_SF <= 1400
 ind_2004: t88_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_2005: t88_path1 = 1 -> const_Overall_Cond <= 4
 ind_2006: t88_path1 = 1 -> const_Total_Bsmt_SF <= 1400
 ind_2007: t88_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_2008: t88_path2 = 1 -> const_Overall_Cond <= 4
 ind_2009: t88_path2 = 1 -> const_Total_Bsmt_SF >= 1400
 ind_2010: t88_path3 = 1 -> const_Overall_Cond >= 4
 ind_2011: t88_path3 = 1 -> const_Yr_Sold <= 2008
 ind_2012: t88_path3 = 1 -> x_1st_Flr_SF <= 1740
 ind_2013: t88_path4 = 1 -> const_Overall_Cond >= 4
 ind_2014: t88_path4 = 1 -> const_Yr_Sold <= 2008
 ind_2015: t88_path4 = 1 -> x_1st_Flr_SF >= 1740
 ind_2016: t88_path5 = 1 -> const_Overall_Cond >= 4
 ind_2017: t88_path5 = 1 -> const_Yr_Sold >= 2008
 ind_2018: t88_path5 = 1 -> const_Total_Bsmt_SF <= 765
 ind_2019: t88_path6 = 1 -> const_Overall_Cond >= 4
 ind_2020: t88_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2021: t88_path6 = 1 -> const_Total_Bsmt_SF >= 765
 ind_2022: t89_path0 = 1 -> x_Bsmt_Cond <= 2
 ind_2023: t89_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2024: t89_path0 = 1 -> x_Enclosed_Porch <= 192
 ind_2025: t89_path1 = 1 -> x_Bsmt_Cond <= 2
 ind_2026: t89_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2027: t89_path1 = 1 -> x_Enclosed_Porch >= 192
 ind_2028: t89_path2 = 1 -> x_Bsmt_Cond <= 2
 ind_2029: t89_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2030: t89_path2 = 1 -> x_TotRms_AbvGrd <= 6
 ind_2031: t89_path3 = 1 -> x_Bsmt_Cond <= 2
 ind_2032: t89_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2033: t89_path3 = 1 -> x_TotRms_AbvGrd >= 6
 ind_2034: t89_path4 = 1 -> x_Bsmt_Cond >= 2
 ind_2035: t89_path4 = 1 -> x_Gr_Liv_Area <= 1749
 ind_2036: t89_path4 = 1 -> const_Condition_1_Artery <= 1
 ind_2037: t89_path5 = 1 -> x_Bsmt_Cond >= 2
 ind_2038: t89_path5 = 1 -> x_Gr_Liv_Area <= 1749
 ind_2039: t89_path5 = 1 -> const_Condition_1_Artery >= 1
 ind_2040: t89_path6 = 1 -> x_Bsmt_Cond >= 2
 ind_2041: t89_path6 = 1 -> x_Gr_Liv_Area >= 1749
 ind_2042: t89_path6 = 1 -> x_Screen_Porch <= 40
 ind_2043: t89_path7 = 1 -> x_Bsmt_Cond >= 2
 ind_2044: t89_path7 = 1 -> x_Gr_Liv_Area >= 1749
 ind_2045: t89_path7 = 1 -> x_Screen_Porch >= 40
 ind_2046: t90_path0 = 1 -> const_Yr_Sold <= 2008
 ind_2047: t90_path0 = 1 -> const_Bldg_Type_1Fam <= 1
 ind_2048: t90_path0 = 1 -> x_Open_Porch_SF <= 4
 ind_2049: t90_path1 = 1 -> const_Yr_Sold <= 2008
 ind_2050: t90_path1 = 1 -> const_Bldg_Type_1Fam <= 1
 ind_2051: t90_path1 = 1 -> x_Open_Porch_SF >= 4
 ind_2052: t90_path2 = 1 -> const_Yr_Sold <= 2008
 ind_2053: t90_path2 = 1 -> const_Bldg_Type_1Fam >= 1
 ind_2054: t90_path2 = 1 -> const_Total_Bsmt_SF <= 1689
 ind_2055: t90_path3 = 1 -> const_Yr_Sold <= 2008
 ind_2056: t90_path3 = 1 -> const_Bldg_Type_1Fam >= 1
 ind_2057: t90_path3 = 1 -> const_Total_Bsmt_SF >= 1689
 ind_2058: t90_path4 = 1 -> const_Yr_Sold >= 2008
 ind_2059: t90_path4 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2060: t90_path4 = 1 -> x_Open_Porch_SF <= 31
 ind_2061: t90_path5 = 1 -> const_Yr_Sold >= 2008
 ind_2062: t90_path5 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2063: t90_path5 = 1 -> x_Open_Porch_SF >= 31
 ind_2064: t90_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2065: t90_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2066: t90_path6 = 1 -> const_Lot_Config_Inside <= 1
 ind_2067: t90_path7 = 1 -> const_Yr_Sold >= 2008
 ind_2068: t90_path7 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2069: t90_path7 = 1 -> const_Lot_Config_Inside >= 1
 ind_2070: t91_path0 = 1 -> x_BsmtFin_SF_1 <= 292
 ind_2071: t91_path0 = 1 -> x_Heating_QC <= 3
 ind_2072: t91_path0 = 1 -> const_MS_Zoning_RL <= 1
 ind_2073: t91_path1 = 1 -> x_BsmtFin_SF_1 <= 292
 ind_2074: t91_path1 = 1 -> x_Heating_QC <= 3
 ind_2075: t91_path1 = 1 -> const_MS_Zoning_RL >= 1
 ind_2076: t91_path2 = 1 -> x_BsmtFin_SF_1 <= 292
 ind_2077: t91_path2 = 1 -> x_Heating_QC >= 3
 ind_2078: t91_path2 = 1 -> const_Lot_Area <= 7440
 ind_2079: t91_path3 = 1 -> x_BsmtFin_SF_1 <= 292
 ind_2080: t91_path3 = 1 -> x_Heating_QC >= 3
 ind_2081: t91_path3 = 1 -> const_Lot_Area >= 7440
 ind_2082: t91_path4 = 1 -> x_BsmtFin_SF_1 >= 292
 ind_2083: t91_path4 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_2084: t91_path4 = 1 -> const_Lot_Area <= 6125
 ind_2085: t91_path5 = 1 -> x_BsmtFin_SF_1 >= 292
 ind_2086: t91_path5 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_2087: t91_path5 = 1 -> const_Lot_Area >= 6125
 ind_2088: t91_path6 = 1 -> x_BsmtFin_SF_1 >= 292
 ind_2089: t91_path6 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_2090: t91_path6 = 1 -> const_Condition_1_Feedr <= 1
 ind_2091: t91_path7 = 1 -> x_BsmtFin_SF_1 >= 292
 ind_2092: t91_path7 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_2093: t91_path7 = 1 -> const_Condition_1_Feedr >= 1
 ind_2094: t92_path0 = 1 -> const_Overall_Cond <= 4
 ind_2095: t92_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2096: t92_path0 = 1 -> central_air_no <= 1
 ind_2097: t92_path1 = 1 -> const_Overall_Cond <= 4
 ind_2098: t92_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2099: t92_path1 = 1 -> central_air_no >= 1
 ind_2100: t92_path2 = 1 -> const_Overall_Cond <= 4
 ind_2101: t92_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2102: t92_path3 = 1 -> const_Overall_Cond >= 4
 ind_2103: t92_path3 = 1 -> x_Full_Bath <= 3
 ind_2104: t92_path3 = 1 -> const_Yr_Sold <= 2008
 ind_2105: t92_path4 = 1 -> const_Overall_Cond >= 4
 ind_2106: t92_path4 = 1 -> x_Full_Bath <= 3
 ind_2107: t92_path4 = 1 -> const_Yr_Sold >= 2008
 ind_2108: t92_path5 = 1 -> const_Overall_Cond >= 4
 ind_2109: t92_path5 = 1 -> x_Full_Bath >= 3
 ind_2110: t92_path5 = 1 -> x_ex1_is_VinylSd <= 1
 ind_2111: t92_path6 = 1 -> const_Overall_Cond >= 4
 ind_2112: t92_path6 = 1 -> x_Full_Bath >= 3
 ind_2113: t92_path6 = 1 -> x_ex1_is_VinylSd >= 1
 ind_2114: t93_path0 = 1 -> const_Yr_Sold <= 2008
 ind_2115: t93_path0 = 1 -> x_Gr_Liv_Area <= 1968
 ind_2116: t93_path0 = 1 -> const_Overall_Cond <= 4
 ind_2117: t93_path1 = 1 -> const_Yr_Sold <= 2008
 ind_2118: t93_path1 = 1 -> x_Gr_Liv_Area <= 1968
 ind_2119: t93_path1 = 1 -> const_Overall_Cond >= 4
 ind_2120: t93_path2 = 1 -> const_Yr_Sold <= 2008
 ind_2121: t93_path2 = 1 -> x_Gr_Liv_Area >= 1968
 ind_2122: t93_path2 = 1 -> const_Misc_Feature_No_aplica <= 1
 ind_2123: t93_path3 = 1 -> const_Yr_Sold <= 2008
 ind_2124: t93_path3 = 1 -> x_Gr_Liv_Area >= 1968
 ind_2125: t93_path3 = 1 -> const_Misc_Feature_No_aplica >= 1
 ind_2126: t93_path4 = 1 -> const_Yr_Sold >= 2008
 ind_2127: t93_path4 = 1 -> const_Land_Contour_Bnk <= 1
 ind_2128: t93_path4 = 1 -> x_Garage_Area <= 351
 ind_2129: t93_path5 = 1 -> const_Yr_Sold >= 2008
 ind_2130: t93_path5 = 1 -> const_Land_Contour_Bnk <= 1
 ind_2131: t93_path5 = 1 -> x_Garage_Area >= 351
 ind_2132: t93_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2133: t93_path6 = 1 -> const_Land_Contour_Bnk >= 1
 ind_2134: t93_path6 = 1 -> x_1st_Flr_SF <= 1017
 ind_2135: t93_path7 = 1 -> const_Yr_Sold >= 2008
 ind_2136: t93_path7 = 1 -> const_Land_Contour_Bnk >= 1
 ind_2137: t93_path7 = 1 -> x_1st_Flr_SF >= 1017
 ind_2138: t94_path0 = 1 -> const_Fireplaces <= 2
 ind_2139: t94_path0 = 1 -> const_Sale_Condition_Partial <= 1
 ind_2140: t94_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_2141: t94_path1 = 1 -> const_Fireplaces <= 2
 ind_2142: t94_path1 = 1 -> const_Sale_Condition_Partial <= 1
 ind_2143: t94_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_2144: t94_path2 = 1 -> const_Fireplaces <= 2
 ind_2145: t94_path2 = 1 -> const_Sale_Condition_Partial >= 1
 ind_2146: t94_path2 = 1 -> x_Bsmt_Unf_SF <= 638
 ind_2147: t94_path3 = 1 -> const_Fireplaces <= 2
 ind_2148: t94_path3 = 1 -> const_Sale_Condition_Partial >= 1
 ind_2149: t94_path3 = 1 -> x_Bsmt_Unf_SF >= 638
 ind_2150: t94_path4 = 1 -> const_Fireplaces >= 2
 ind_2151: t94_path4 = 1 -> const_Misc_Val <= 54
 ind_2152: t94_path4 = 1 -> x_Gr_Liv_Area <= 2134
 ind_2153: t94_path5 = 1 -> const_Fireplaces >= 2
 ind_2154: t94_path5 = 1 -> const_Misc_Val <= 54
 ind_2155: t94_path5 = 1 -> x_Gr_Liv_Area >= 2134
 ind_2156: t94_path6 = 1 -> const_Fireplaces >= 2
 ind_2157: t94_path6 = 1 -> const_Misc_Val >= 54
 ind_2158: t95_path0 = 1 -> const_Garage_Yr_Blt <= 1929
 ind_2159: t95_path0 = 1 -> x_Gr_Liv_Area <= 1200
 ind_2160: t95_path0 = 1 -> const_Year_Built <= 1924
 ind_2161: t95_path1 = 1 -> const_Garage_Yr_Blt <= 1929
 ind_2162: t95_path1 = 1 -> x_Gr_Liv_Area <= 1200
 ind_2163: t95_path1 = 1 -> const_Year_Built >= 1924
 ind_2164: t95_path2 = 1 -> const_Garage_Yr_Blt <= 1929
 ind_2165: t95_path2 = 1 -> x_Gr_Liv_Area >= 1200
 ind_2166: t95_path2 = 1 -> const_Year_Built <= 1911
 ind_2167: t95_path3 = 1 -> const_Garage_Yr_Blt <= 1929
 ind_2168: t95_path3 = 1 -> x_Gr_Liv_Area >= 1200
 ind_2169: t95_path3 = 1 -> const_Year_Built >= 1911
 ind_2170: t95_path4 = 1 -> const_Garage_Yr_Blt >= 1929
 ind_2171: t95_path4 = 1 -> const_Year_Remod_Add <= 1995
 ind_2172: t95_path4 = 1 -> x_BsmtFin_SF_2 <= 532
 ind_2173: t95_path5 = 1 -> const_Garage_Yr_Blt >= 1929
 ind_2174: t95_path5 = 1 -> const_Year_Remod_Add <= 1995
 ind_2175: t95_path5 = 1 -> x_BsmtFin_SF_2 >= 532
 ind_2176: t95_path6 = 1 -> const_Garage_Yr_Blt >= 1929
 ind_2177: t95_path6 = 1 -> const_Year_Remod_Add >= 1995
 ind_2178: t95_path6 = 1 -> x_Gr_Liv_Area <= 1968
 ind_2179: t95_path7 = 1 -> const_Garage_Yr_Blt >= 1929
 ind_2180: t95_path7 = 1 -> const_Year_Remod_Add >= 1995
 ind_2181: t95_path7 = 1 -> x_Gr_Liv_Area >= 1968
 ind_2182: t96_path0 = 1 -> x_Exter_Cond <= 2
 ind_2183: t96_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2184: t96_path0 = 1 -> const_Garage_Yr_Blt <= 1921
 ind_2185: t96_path1 = 1 -> x_Exter_Cond <= 2
 ind_2186: t96_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2187: t96_path1 = 1 -> const_Garage_Yr_Blt >= 1921
 ind_2188: t96_path2 = 1 -> x_Exter_Cond <= 2
 ind_2189: t96_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2190: t96_path3 = 1 -> x_Exter_Cond >= 2
 ind_2191: t96_path3 = 1 -> x_Gr_Liv_Area <= 988
 ind_2192: t96_path3 = 1 -> const_Overall_Cond <= 7
 ind_2193: t96_path4 = 1 -> x_Exter_Cond >= 2
 ind_2194: t96_path4 = 1 -> x_Gr_Liv_Area <= 988
 ind_2195: t96_path4 = 1 -> const_Overall_Cond >= 7
 ind_2196: t96_path5 = 1 -> x_Exter_Cond >= 2
 ind_2197: t96_path5 = 1 -> x_Gr_Liv_Area >= 988
 ind_2198: t96_path5 = 1 -> const_Yr_Sold <= 2007
 ind_2199: t96_path6 = 1 -> x_Exter_Cond >= 2
 ind_2200: t96_path6 = 1 -> x_Gr_Liv_Area >= 988
 ind_2201: t96_path6 = 1 -> const_Yr_Sold >= 2007
 ind_2202: t97_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_2203: t97_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2204: t97_path0 = 1 -> x_Heating_QC <= 3
 ind_2205: t97_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_2206: t97_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2207: t97_path1 = 1 -> x_Heating_QC >= 3
 ind_2208: t97_path2 = 1 -> const_MS_SubClass_30 <= 1
 ind_2209: t97_path2 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2210: t97_path2 = 1 -> x_Gr_Liv_Area <= 1504
 ind_2211: t97_path3 = 1 -> const_MS_SubClass_30 <= 1
 ind_2212: t97_path3 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2213: t97_path3 = 1 -> x_Gr_Liv_Area >= 1504
 ind_2214: t97_path4 = 1 -> const_MS_SubClass_30 >= 1
 ind_2215: t97_path4 = 1 -> x_Heating_QC <= 3
 ind_2216: t97_path4 = 1 -> const_Year_Built <= 1925
 ind_2217: t97_path5 = 1 -> const_MS_SubClass_30 >= 1
 ind_2218: t97_path5 = 1 -> x_Heating_QC <= 3
 ind_2219: t97_path5 = 1 -> const_Year_Built >= 1925
 ind_2220: t97_path6 = 1 -> const_MS_SubClass_30 >= 1
 ind_2221: t97_path6 = 1 -> x_Heating_QC >= 3
 ind_2222: t97_path6 = 1 -> x_Bedroom_AbvGr <= 3
 ind_2223: t97_path7 = 1 -> const_MS_SubClass_30 >= 1
 ind_2224: t97_path7 = 1 -> x_Heating_QC >= 3
 ind_2225: t97_path7 = 1 -> x_Bedroom_AbvGr >= 3
 ind_2226: t98_path0 = 1 -> const_Overall_Cond <= 4
 ind_2227: t98_path0 = 1 -> x_Heating_QC <= 4
 ind_2228: t98_path0 = 1 -> const_Yr_Sold <= 2007
 ind_2229: t98_path1 = 1 -> const_Overall_Cond <= 4
 ind_2230: t98_path1 = 1 -> x_Heating_QC <= 4
 ind_2231: t98_path1 = 1 -> const_Yr_Sold >= 2007
 ind_2232: t98_path2 = 1 -> const_Overall_Cond <= 4
 ind_2233: t98_path2 = 1 -> x_Heating_QC >= 4
 ind_2234: t98_path3 = 1 -> const_Overall_Cond >= 4
 ind_2235: t98_path3 = 1 -> x_Gr_Liv_Area <= 2649
 ind_2236: t98_path3 = 1 -> const_Year_Built <= 2006
 ind_2237: t98_path4 = 1 -> const_Overall_Cond >= 4
 ind_2238: t98_path4 = 1 -> x_Gr_Liv_Area <= 2649
 ind_2239: t98_path4 = 1 -> const_Year_Built >= 2006
 ind_2240: t98_path5 = 1 -> const_Overall_Cond >= 4
 ind_2241: t98_path5 = 1 -> x_Gr_Liv_Area >= 2649
 ind_2242: t98_path5 = 1 -> x_Bedroom_AbvGr <= 4
 ind_2243: t98_path6 = 1 -> const_Overall_Cond >= 4
 ind_2244: t98_path6 = 1 -> x_Gr_Liv_Area >= 2649
 ind_2245: t98_path6 = 1 -> x_Bedroom_AbvGr >= 4
 ind_2246: t99_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2247: t99_path0 = 1 -> x_BsmtFin_SF_1 <= 292
 ind_2248: t99_path0 = 1 -> const_Year_Remod_Add <= 1984
 ind_2249: t99_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2250: t99_path1 = 1 -> x_BsmtFin_SF_1 <= 292
 ind_2251: t99_path1 = 1 -> const_Year_Remod_Add >= 1984
 ind_2252: t99_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2253: t99_path2 = 1 -> x_BsmtFin_SF_1 >= 292
 ind_2254: t99_path2 = 1 -> const_Year_Remod_Add <= 1994
 ind_2255: t99_path3 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2256: t99_path3 = 1 -> x_BsmtFin_SF_1 >= 292
 ind_2257: t99_path3 = 1 -> const_Year_Remod_Add >= 1994
 ind_2258: t99_path4 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2259: t99_path4 = 1 -> x_Bsmt_Unf_SF <= 410
 ind_2260: t99_path5 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2261: t99_path5 = 1 -> x_Bsmt_Unf_SF >= 410
 ind_2262: t99_path5 = 1 -> const_Overall_Cond <= 6
 ind_2263: t99_path6 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2264: t99_path6 = 1 -> x_Bsmt_Unf_SF >= 410
 ind_2265: t99_path6 = 1 -> const_Overall_Cond >= 6
 ind_2266: t100_path0 = 1 -> x_Gr_Liv_Area <= 833
 ind_2267: t100_path0 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_2268: t100_path0 = 1 -> const_Total_Bsmt_SF <= 648
 ind_2269: t100_path1 = 1 -> x_Gr_Liv_Area <= 833
 ind_2270: t100_path1 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_2271: t100_path1 = 1 -> const_Total_Bsmt_SF >= 648
 ind_2272: t100_path2 = 1 -> x_Gr_Liv_Area <= 833
 ind_2273: t100_path2 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_2274: t100_path2 = 1 -> x_b1_is_Unf <= 1
 ind_2275: t100_path3 = 1 -> x_Gr_Liv_Area <= 833
 ind_2276: t100_path3 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_2277: t100_path3 = 1 -> x_b1_is_Unf >= 1
 ind_2278: t100_path4 = 1 -> x_Gr_Liv_Area >= 833
 ind_2279: t100_path4 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_2280: t100_path4 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_2281: t100_path5 = 1 -> x_Gr_Liv_Area >= 833
 ind_2282: t100_path5 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_2283: t100_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_2284: t100_path6 = 1 -> x_Gr_Liv_Area >= 833
 ind_2285: t100_path6 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_2286: t100_path6 = 1 -> x_Garage_Area <= 1154
 ind_2287: t100_path7 = 1 -> x_Gr_Liv_Area >= 833
 ind_2288: t100_path7 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_2289: t100_path7 = 1 -> x_Garage_Area >= 1154
 ind_2290: t101_path0 = 1 -> const_Overall_Cond <= 4
 ind_2291: t101_path0 = 1 -> x_Bsmt_Unf_SF <= 370
 ind_2292: t101_path1 = 1 -> const_Overall_Cond <= 4
 ind_2293: t101_path1 = 1 -> x_Bsmt_Unf_SF >= 370
 ind_2294: t101_path1 = 1 -> const_Mo_Sold_5 <= 1
 ind_2295: t101_path2 = 1 -> const_Overall_Cond <= 4
 ind_2296: t101_path2 = 1 -> x_Bsmt_Unf_SF >= 370
 ind_2297: t101_path2 = 1 -> const_Mo_Sold_5 >= 1
 ind_2298: t101_path3 = 1 -> const_Overall_Cond >= 4
 ind_2299: t101_path3 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2300: t101_path3 = 1 -> x_Garage_Area <= 1154
 ind_2301: t101_path4 = 1 -> const_Overall_Cond >= 4
 ind_2302: t101_path4 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2303: t101_path4 = 1 -> x_Garage_Area >= 1154
 ind_2304: t101_path5 = 1 -> const_Overall_Cond >= 4
 ind_2305: t101_path5 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2306: t101_path5 = 1 -> x_Gr_Liv_Area <= 1504
 ind_2307: t101_path6 = 1 -> const_Overall_Cond >= 4
 ind_2308: t101_path6 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2309: t101_path6 = 1 -> x_Gr_Liv_Area >= 1504
 ind_2310: t102_path0 = 1 -> Overall_Qual_calc <= 5
 ind_2311: t102_path0 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_2312: t102_path0 = 1 -> x_Open_Porch_SF <= 39
 ind_2313: t102_path1 = 1 -> Overall_Qual_calc <= 5
 ind_2314: t102_path1 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_2315: t102_path1 = 1 -> x_Open_Porch_SF >= 39
 ind_2316: t102_path2 = 1 -> Overall_Qual_calc <= 5
 ind_2317: t102_path2 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_2318: t102_path2 = 1 -> const_Yr_Sold <= 2007
 ind_2319: t102_path3 = 1 -> Overall_Qual_calc <= 5
 ind_2320: t102_path3 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_2321: t102_path3 = 1 -> const_Yr_Sold >= 2007
 ind_2322: t102_path4 = 1 -> Overall_Qual_calc >= 5
 ind_2323: t102_path4 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_2324: t102_path4 = 1 -> x_Full_Bath <= 3
 ind_2325: t102_path5 = 1 -> Overall_Qual_calc >= 5
 ind_2326: t102_path5 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_2327: t102_path5 = 1 -> x_Full_Bath >= 3
 ind_2328: t102_path6 = 1 -> Overall_Qual_calc >= 5
 ind_2329: t102_path6 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_2330: t102_path6 = 1 -> const_Year_Remod_Add <= 2009
 ind_2331: t102_path7 = 1 -> Overall_Qual_calc >= 5
 ind_2332: t102_path7 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_2333: t102_path7 = 1 -> const_Year_Remod_Add >= 2009
 ind_2334: t103_path0 = 1 -> const_Overall_Cond <= 5
 ind_2335: t103_path0 = 1 -> const_Total_Bsmt_SF <= 1362
 ind_2336: t103_path1 = 1 -> const_Overall_Cond <= 5
 ind_2337: t103_path1 = 1 -> const_Total_Bsmt_SF >= 1362
 ind_2338: t103_path2 = 1 -> const_Overall_Cond >= 5
 ind_2339: t103_path2 = 1 -> x_Full_Bath <= 3
 ind_2340: t103_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2341: t103_path3 = 1 -> const_Overall_Cond >= 5
 ind_2342: t103_path3 = 1 -> x_Full_Bath <= 3
 ind_2343: t103_path3 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2344: t103_path4 = 1 -> const_Overall_Cond >= 5
 ind_2345: t103_path4 = 1 -> x_Full_Bath >= 3
 ind_2346: t103_path4 = 1 -> x_Wood_Deck_SF <= 116
 ind_2347: t103_path5 = 1 -> const_Overall_Cond >= 5
 ind_2348: t103_path5 = 1 -> x_Full_Bath >= 3
 ind_2349: t103_path5 = 1 -> x_Wood_Deck_SF >= 116
 ind_2350: t104_path0 = 1 -> Overall_Qual_calc <= 3
 ind_2351: t104_path0 = 1 -> x_Garage_Area <= 286
 ind_2352: t104_path1 = 1 -> Overall_Qual_calc <= 3
 ind_2353: t104_path1 = 1 -> x_Garage_Area >= 286
 ind_2354: t104_path2 = 1 -> Overall_Qual_calc >= 3
 ind_2355: t104_path2 = 1 -> const_Lot_Area <= 14680
 ind_2356: t104_path2 = 1 -> const_Garage_Yr_Blt <= 1900
 ind_2357: t104_path3 = 1 -> Overall_Qual_calc >= 3
 ind_2358: t104_path3 = 1 -> const_Lot_Area <= 14680
 ind_2359: t104_path3 = 1 -> const_Garage_Yr_Blt >= 1900
 ind_2360: t104_path4 = 1 -> Overall_Qual_calc >= 3
 ind_2361: t104_path4 = 1 -> const_Lot_Area >= 14680
 ind_2362: t104_path4 = 1 -> const_2nd_Flr_SF <= 1276
 ind_2363: t104_path5 = 1 -> Overall_Qual_calc >= 3
 ind_2364: t104_path5 = 1 -> const_Lot_Area >= 14680
 ind_2365: t104_path5 = 1 -> const_2nd_Flr_SF >= 1276
 ind_2366: t105_path0 = 1 -> x_Bsmt_Cond <= 2
 ind_2367: t105_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2368: t105_path0 = 1 -> x_1st_Flr_SF <= 980
 ind_2369: t105_path1 = 1 -> x_Bsmt_Cond <= 2
 ind_2370: t105_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2371: t105_path1 = 1 -> x_1st_Flr_SF >= 980
 ind_2372: t105_path2 = 1 -> x_Bsmt_Cond <= 2
 ind_2373: t105_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2374: t105_path2 = 1 -> x_Heating_QC <= 3
 ind_2375: t105_path3 = 1 -> x_Bsmt_Cond <= 2
 ind_2376: t105_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2377: t105_path3 = 1 -> x_Heating_QC >= 3
 ind_2378: t105_path4 = 1 -> x_Bsmt_Cond >= 2
 ind_2379: t105_path4 = 1 -> const_Overall_Cond <= 7
 ind_2380: t105_path4 = 1 -> const_Year_Built <= 1978
 ind_2381: t105_path5 = 1 -> x_Bsmt_Cond >= 2
 ind_2382: t105_path5 = 1 -> const_Overall_Cond <= 7
 ind_2383: t105_path5 = 1 -> const_Year_Built >= 1978
 ind_2384: t105_path6 = 1 -> x_Bsmt_Cond >= 2
 ind_2385: t105_path6 = 1 -> const_Overall_Cond >= 7
 ind_2386: t105_path6 = 1 -> x_1st_Flr_SF <= 1100
 ind_2387: t105_path7 = 1 -> x_Bsmt_Cond >= 2
 ind_2388: t105_path7 = 1 -> const_Overall_Cond >= 7
 ind_2389: t105_path7 = 1 -> x_1st_Flr_SF >= 1100
 ind_2390: t106_path0 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_2391: t106_path0 = 1 -> x_Gr_Liv_Area <= 1146
 ind_2392: t106_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_2393: t106_path1 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_2394: t106_path1 = 1 -> x_Gr_Liv_Area <= 1146
 ind_2395: t106_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_2396: t106_path2 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_2397: t106_path2 = 1 -> x_Gr_Liv_Area >= 1146
 ind_2398: t106_path2 = 1 -> const_Overall_Cond <= 7
 ind_2399: t106_path3 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_2400: t106_path3 = 1 -> x_Gr_Liv_Area >= 1146
 ind_2401: t106_path3 = 1 -> const_Overall_Cond >= 7
 ind_2402: t106_path4 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_2403: t106_path4 = 1 -> x_TotRms_AbvGrd <= 12
 ind_2404: t106_path4 = 1 -> const_Lot_Area <= 18386
 ind_2405: t106_path5 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_2406: t106_path5 = 1 -> x_TotRms_AbvGrd <= 12
 ind_2407: t106_path5 = 1 -> const_Lot_Area >= 18386
 ind_2408: t106_path6 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_2409: t106_path6 = 1 -> x_TotRms_AbvGrd >= 12
 ind_2410: t107_path0 = 1 -> const_Yr_Sold <= 2008
 ind_2411: t107_path0 = 1 -> const_Lot_Area <= 13811
 ind_2412: t107_path0 = 1 -> x_BsmtFin_SF_1 <= 223
 ind_2413: t107_path1 = 1 -> const_Yr_Sold <= 2008
 ind_2414: t107_path1 = 1 -> const_Lot_Area <= 13811
 ind_2415: t107_path1 = 1 -> x_BsmtFin_SF_1 >= 223
 ind_2416: t107_path2 = 1 -> const_Yr_Sold <= 2008
 ind_2417: t107_path2 = 1 -> const_Lot_Area >= 13811
 ind_2418: t107_path2 = 1 -> x_Open_Porch_SF <= 273
 ind_2419: t107_path3 = 1 -> const_Yr_Sold <= 2008
 ind_2420: t107_path3 = 1 -> const_Lot_Area >= 13811
 ind_2421: t107_path3 = 1 -> x_Open_Porch_SF >= 273
 ind_2422: t107_path4 = 1 -> const_Yr_Sold >= 2008
 ind_2423: t107_path4 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_2424: t107_path4 = 1 -> const_Condition_1_Feedr <= 1
 ind_2425: t107_path5 = 1 -> const_Yr_Sold >= 2008
 ind_2426: t107_path5 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_2427: t107_path5 = 1 -> const_Condition_1_Feedr >= 1
 ind_2428: t107_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2429: t107_path6 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_2430: t107_path6 = 1 -> const_Lot_Frontage <= 63
 ind_2431: t107_path7 = 1 -> const_Yr_Sold >= 2008
 ind_2432: t107_path7 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_2433: t107_path7 = 1 -> const_Lot_Frontage >= 63
 ind_2434: t108_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2435: t108_path0 = 1 -> const_Lot_Shape_Reg <= 1
 ind_2436: t108_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_2437: t108_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2438: t108_path1 = 1 -> const_Lot_Shape_Reg <= 1
 ind_2439: t108_path1 = 1 -> x_ex1_is_BrkFace >= 1
 ind_2440: t108_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2441: t108_path2 = 1 -> const_Lot_Shape_Reg >= 1
 ind_2442: t108_path2 = 1 -> const_Lot_Area <= 2645
 ind_2443: t108_path3 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2444: t108_path3 = 1 -> const_Lot_Shape_Reg >= 1
 ind_2445: t108_path3 = 1 -> const_Lot_Area >= 2645
 ind_2446: t108_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2447: t108_path4 = 1 -> Overall_Qual_calc <= 5
 ind_2448: t108_path4 = 1 -> x_ex1_is_MetalSd <= 1
 ind_2449: t108_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2450: t108_path5 = 1 -> Overall_Qual_calc <= 5
 ind_2451: t108_path5 = 1 -> x_ex1_is_MetalSd >= 1
 ind_2452: t108_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2453: t108_path6 = 1 -> Overall_Qual_calc >= 5
 ind_2454: t108_path6 = 1 -> x_Kitchen_Qual <= 4
 ind_2455: t108_path7 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2456: t108_path7 = 1 -> Overall_Qual_calc >= 5
 ind_2457: t108_path7 = 1 -> x_Kitchen_Qual >= 4
 ind_2458: t109_path0 = 1 -> x_Exter_Cond <= 2
 ind_2459: t109_path0 = 1 -> Overall_Qual_calc <= 3
 ind_2460: t109_path1 = 1 -> x_Exter_Cond <= 2
 ind_2461: t109_path1 = 1 -> Overall_Qual_calc >= 3
 ind_2462: t109_path1 = 1 -> x_Garage_Area <= 351
 ind_2463: t109_path2 = 1 -> x_Exter_Cond <= 2
 ind_2464: t109_path2 = 1 -> Overall_Qual_calc >= 3
 ind_2465: t109_path2 = 1 -> x_Garage_Area >= 351
 ind_2466: t109_path3 = 1 -> x_Exter_Cond >= 2
 ind_2467: t109_path3 = 1 -> x_Gr_Liv_Area <= 2263
 ind_2468: t109_path3 = 1 -> const_Yr_Sold <= 2008
 ind_2469: t109_path4 = 1 -> x_Exter_Cond >= 2
 ind_2470: t109_path4 = 1 -> x_Gr_Liv_Area <= 2263
 ind_2471: t109_path4 = 1 -> const_Yr_Sold >= 2008
 ind_2472: t109_path5 = 1 -> x_Exter_Cond >= 2
 ind_2473: t109_path5 = 1 -> x_Gr_Liv_Area >= 2263
 ind_2474: t109_path5 = 1 -> x_1st_Flr_SF <= 1241
 ind_2475: t109_path6 = 1 -> x_Exter_Cond >= 2
 ind_2476: t109_path6 = 1 -> x_Gr_Liv_Area >= 2263
 ind_2477: t109_path6 = 1 -> x_1st_Flr_SF >= 1241
 ind_2478: t110_path0 = 1 -> x_Screen_Porch <= 104
 ind_2479: t110_path0 = 1 -> x_BsmtFin_SF_1 <= 223
 ind_2480: t110_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2481: t110_path1 = 1 -> x_Screen_Porch <= 104
 ind_2482: t110_path1 = 1 -> x_BsmtFin_SF_1 <= 223
 ind_2483: t110_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2484: t110_path2 = 1 -> x_Screen_Porch <= 104
 ind_2485: t110_path2 = 1 -> x_BsmtFin_SF_1 >= 223
 ind_2486: t110_path2 = 1 -> const_2nd_Flr_SF <= 1629
 ind_2487: t110_path3 = 1 -> x_Screen_Porch <= 104
 ind_2488: t110_path3 = 1 -> x_BsmtFin_SF_1 >= 223
 ind_2489: t110_path3 = 1 -> const_2nd_Flr_SF >= 1629
 ind_2490: t110_path4 = 1 -> x_Screen_Porch >= 104
 ind_2491: t110_path4 = 1 -> x_Full_Bath <= 2
 ind_2492: t110_path5 = 1 -> x_Screen_Porch >= 104
 ind_2493: t110_path5 = 1 -> x_Full_Bath >= 2
 ind_2494: t111_path0 = 1 -> const_Yr_Sold <= 2008
 ind_2495: t111_path0 = 1 -> x_BsmtFin_SF_1 <= 201
 ind_2496: t111_path0 = 1 -> x_Wood_Deck_SF <= 158
 ind_2497: t111_path1 = 1 -> const_Yr_Sold <= 2008
 ind_2498: t111_path1 = 1 -> x_BsmtFin_SF_1 <= 201
 ind_2499: t111_path1 = 1 -> x_Wood_Deck_SF >= 158
 ind_2500: t111_path2 = 1 -> const_Yr_Sold <= 2008
 ind_2501: t111_path2 = 1 -> x_BsmtFin_SF_1 >= 201
 ind_2502: t111_path2 = 1 -> const_Overall_Cond <= 6
 ind_2503: t111_path3 = 1 -> const_Yr_Sold <= 2008
 ind_2504: t111_path3 = 1 -> x_BsmtFin_SF_1 >= 201
 ind_2505: t111_path3 = 1 -> const_Overall_Cond >= 6
 ind_2506: t111_path4 = 1 -> const_Yr_Sold >= 2008
 ind_2507: t111_path4 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_2508: t111_path4 = 1 -> const_Overall_Cond <= 7
 ind_2509: t111_path5 = 1 -> const_Yr_Sold >= 2008
 ind_2510: t111_path5 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_2511: t111_path5 = 1 -> const_Overall_Cond >= 7
 ind_2512: t111_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2513: t111_path6 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_2514: t111_path6 = 1 -> x_Gr_Liv_Area <= 1134
 ind_2515: t111_path7 = 1 -> const_Yr_Sold >= 2008
 ind_2516: t111_path7 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_2517: t111_path7 = 1 -> x_Gr_Liv_Area >= 1134
 ind_2518: t112_path0 = 1 -> const_Lot_Shape_IR1 <= 1
 ind_2519: t112_path0 = 1 -> const_2nd_Flr_SF <= 978
 ind_2520: t112_path0 = 1 -> x_Bsmt_Cond <= 2
 ind_2521: t112_path1 = 1 -> const_Lot_Shape_IR1 <= 1
 ind_2522: t112_path1 = 1 -> const_2nd_Flr_SF <= 978
 ind_2523: t112_path1 = 1 -> x_Bsmt_Cond >= 2
 ind_2524: t112_path2 = 1 -> const_Lot_Shape_IR1 <= 1
 ind_2525: t112_path2 = 1 -> const_2nd_Flr_SF >= 978
 ind_2526: t112_path2 = 1 -> const_Overall_Cond <= 6
 ind_2527: t112_path3 = 1 -> const_Lot_Shape_IR1 <= 1
 ind_2528: t112_path3 = 1 -> const_2nd_Flr_SF >= 978
 ind_2529: t112_path3 = 1 -> const_Overall_Cond >= 6
 ind_2530: t112_path4 = 1 -> const_Lot_Shape_IR1 >= 1
 ind_2531: t112_path4 = 1 -> const_Yr_Sold <= 2008
 ind_2532: t112_path4 = 1 -> const_Lot_Area <= 15138
 ind_2533: t112_path5 = 1 -> const_Lot_Shape_IR1 >= 1
 ind_2534: t112_path5 = 1 -> const_Yr_Sold <= 2008
 ind_2535: t112_path5 = 1 -> const_Lot_Area >= 15138
 ind_2536: t112_path6 = 1 -> const_Lot_Shape_IR1 >= 1
 ind_2537: t112_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2538: t112_path6 = 1 -> const_Sale_Condition_Partial <= 1
 ind_2539: t112_path7 = 1 -> const_Lot_Shape_IR1 >= 1
 ind_2540: t112_path7 = 1 -> const_Yr_Sold >= 2008
 ind_2541: t112_path7 = 1 -> const_Sale_Condition_Partial >= 1
 ind_2542: t113_path0 = 1 -> x_Gr_Liv_Area <= 833
 ind_2543: t113_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2544: t113_path0 = 1 -> x_Enclosed_Porch <= 35
 ind_2545: t113_path1 = 1 -> x_Gr_Liv_Area <= 833
 ind_2546: t113_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2547: t113_path1 = 1 -> x_Enclosed_Porch >= 35
 ind_2548: t113_path2 = 1 -> x_Gr_Liv_Area <= 833
 ind_2549: t113_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2550: t113_path3 = 1 -> x_Gr_Liv_Area >= 833
 ind_2551: t113_path3 = 1 -> const_Total_Bsmt_SF <= 1438
 ind_2552: t113_path3 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2553: t113_path4 = 1 -> x_Gr_Liv_Area >= 833
 ind_2554: t113_path4 = 1 -> const_Total_Bsmt_SF <= 1438
 ind_2555: t113_path4 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2556: t113_path5 = 1 -> x_Gr_Liv_Area >= 833
 ind_2557: t113_path5 = 1 -> const_Total_Bsmt_SF >= 1438
 ind_2558: t113_path5 = 1 -> x_Open_Porch_SF <= 235
 ind_2559: t113_path6 = 1 -> x_Gr_Liv_Area >= 833
 ind_2560: t113_path6 = 1 -> const_Total_Bsmt_SF >= 1438
 ind_2561: t113_path6 = 1 -> x_Open_Porch_SF >= 235
 ind_2562: t114_path0 = 1 -> const_Lot_Area <= 7264
 ind_2563: t114_path0 = 1 -> x_BsmtFin_SF_1 <= 770
 ind_2564: t114_path0 = 1 -> Overall_Qual_calc <= 5
 ind_2565: t114_path1 = 1 -> const_Lot_Area <= 7264
 ind_2566: t114_path1 = 1 -> x_BsmtFin_SF_1 <= 770
 ind_2567: t114_path1 = 1 -> Overall_Qual_calc >= 5
 ind_2568: t114_path2 = 1 -> const_Lot_Area <= 7264
 ind_2569: t114_path2 = 1 -> x_BsmtFin_SF_1 >= 770
 ind_2570: t114_path2 = 1 -> const_Year_Built <= 1925
 ind_2571: t114_path3 = 1 -> const_Lot_Area <= 7264
 ind_2572: t114_path3 = 1 -> x_BsmtFin_SF_1 >= 770
 ind_2573: t114_path3 = 1 -> const_Year_Built >= 1925
 ind_2574: t114_path4 = 1 -> const_Lot_Area >= 7264
 ind_2575: t114_path4 = 1 -> x_Wood_Deck_SF <= 199
 ind_2576: t114_path4 = 1 -> x_Screen_Porch <= 115
 ind_2577: t114_path5 = 1 -> const_Lot_Area >= 7264
 ind_2578: t114_path5 = 1 -> x_Wood_Deck_SF <= 199
 ind_2579: t114_path5 = 1 -> x_Screen_Porch >= 115
 ind_2580: t114_path6 = 1 -> const_Lot_Area >= 7264
 ind_2581: t114_path6 = 1 -> x_Wood_Deck_SF >= 199
 ind_2582: t114_path6 = 1 -> const_Lot_Frontage <= 150
 ind_2583: t114_path7 = 1 -> const_Lot_Area >= 7264
 ind_2584: t114_path7 = 1 -> x_Wood_Deck_SF >= 199
 ind_2585: t114_path7 = 1 -> const_Lot_Frontage >= 150
 ind_2586: t115_path0 = 1 -> const_Overall_Cond <= 5
 ind_2587: t115_path0 = 1 -> x_Screen_Porch <= 88
 ind_2588: t115_path1 = 1 -> const_Overall_Cond <= 5
 ind_2589: t115_path1 = 1 -> x_Screen_Porch >= 88
 ind_2590: t115_path2 = 1 -> const_Overall_Cond >= 5
 ind_2591: t115_path2 = 1 -> x_Wood_Deck_SF <= 152
 ind_2592: t115_path2 = 1 -> x_Screen_Porch <= 100
 ind_2593: t115_path3 = 1 -> const_Overall_Cond >= 5
 ind_2594: t115_path3 = 1 -> x_Wood_Deck_SF <= 152
 ind_2595: t115_path3 = 1 -> x_Screen_Porch >= 100
 ind_2596: t115_path4 = 1 -> const_Overall_Cond >= 5
 ind_2597: t115_path4 = 1 -> x_Wood_Deck_SF >= 152
 ind_2598: t115_path4 = 1 -> const_Land_Contour_Bnk <= 1
 ind_2599: t115_path5 = 1 -> const_Overall_Cond >= 5
 ind_2600: t115_path5 = 1 -> x_Wood_Deck_SF >= 152
 ind_2601: t115_path5 = 1 -> const_Land_Contour_Bnk >= 1
 ind_2602: t116_path0 = 1 -> const_Yr_Sold <= 2008
 ind_2603: t116_path0 = 1 -> x_Gr_Liv_Area <= 3086
 ind_2604: t116_path0 = 1 -> x_BsmtFin_SF_1 <= 881
 ind_2605: t116_path1 = 1 -> const_Yr_Sold <= 2008
 ind_2606: t116_path1 = 1 -> x_Gr_Liv_Area <= 3086
 ind_2607: t116_path1 = 1 -> x_BsmtFin_SF_1 >= 881
 ind_2608: t116_path2 = 1 -> const_Yr_Sold <= 2008
 ind_2609: t116_path2 = 1 -> x_Gr_Liv_Area >= 3086
 ind_2610: t116_path3 = 1 -> const_Yr_Sold >= 2008
 ind_2611: t116_path3 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2612: t116_path3 = 1 -> const_Overall_Cond <= 7
 ind_2613: t116_path4 = 1 -> const_Yr_Sold >= 2008
 ind_2614: t116_path4 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2615: t116_path4 = 1 -> const_Overall_Cond >= 7
 ind_2616: t116_path5 = 1 -> const_Yr_Sold >= 2008
 ind_2617: t116_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2618: t116_path5 = 1 -> x_Gr_Liv_Area <= 976
 ind_2619: t116_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2620: t116_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2621: t116_path6 = 1 -> x_Gr_Liv_Area >= 976
 ind_2622: t117_path0 = 1 -> x_Gr_Liv_Area <= 3086
 ind_2623: t117_path0 = 1 -> const_Overall_Cond <= 5
 ind_2624: t117_path0 = 1 -> x_Heating_QC <= 3
 ind_2625: t117_path1 = 1 -> x_Gr_Liv_Area <= 3086
 ind_2626: t117_path1 = 1 -> const_Overall_Cond <= 5
 ind_2627: t117_path1 = 1 -> x_Heating_QC >= 3
 ind_2628: t117_path2 = 1 -> x_Gr_Liv_Area <= 3086
 ind_2629: t117_path2 = 1 -> const_Overall_Cond >= 5
 ind_2630: t117_path2 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_2631: t117_path3 = 1 -> x_Gr_Liv_Area <= 3086
 ind_2632: t117_path3 = 1 -> const_Overall_Cond >= 5
 ind_2633: t117_path3 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_2634: t117_path4 = 1 -> x_Gr_Liv_Area >= 3086
 ind_2635: t118_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_2636: t118_path0 = 1 -> const_Yr_Sold <= 2007
 ind_2637: t118_path0 = 1 -> x_Wood_Deck_SF <= 199
 ind_2638: t118_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_2639: t118_path1 = 1 -> const_Yr_Sold <= 2007
 ind_2640: t118_path1 = 1 -> x_Wood_Deck_SF >= 199
 ind_2641: t118_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_2642: t118_path2 = 1 -> const_Yr_Sold >= 2007
 ind_2643: t118_path2 = 1 -> const_Lot_Area <= 5976
 ind_2644: t118_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_2645: t118_path3 = 1 -> const_Yr_Sold >= 2007
 ind_2646: t118_path3 = 1 -> const_Lot_Area >= 5976
 ind_2647: t118_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_2648: t118_path4 = 1 -> x_Bedroom_AbvGr <= 3
 ind_2649: t118_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_2650: t118_path5 = 1 -> x_Bedroom_AbvGr >= 3
 ind_2651: t119_path0 = 1 -> const_Overall_Cond <= 3
 ind_2652: t119_path1 = 1 -> const_Overall_Cond >= 3
 ind_2653: t119_path1 = 1 -> const_2nd_Flr_SF <= 1427
 ind_2654: t119_path1 = 1 -> x_BsmtFin_SF_1 <= 75
 ind_2655: t119_path2 = 1 -> const_Overall_Cond >= 3
 ind_2656: t119_path2 = 1 -> const_2nd_Flr_SF <= 1427
 ind_2657: t119_path2 = 1 -> x_BsmtFin_SF_1 >= 75
 ind_2658: t119_path3 = 1 -> const_Overall_Cond >= 3
 ind_2659: t119_path3 = 1 -> const_2nd_Flr_SF >= 1427
 ind_2660: t120_path0 = 1 -> Overall_Qual_calc <= 3
 ind_2661: t120_path0 = 1 -> const_Year_Remod_Add <= 1951
 ind_2662: t120_path1 = 1 -> Overall_Qual_calc <= 3
 ind_2663: t120_path1 = 1 -> const_Year_Remod_Add >= 1951
 ind_2664: t120_path2 = 1 -> Overall_Qual_calc >= 3
 ind_2665: t120_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2666: t120_path2 = 1 -> x_Garage_Area <= 1154
 ind_2667: t120_path3 = 1 -> Overall_Qual_calc >= 3
 ind_2668: t120_path3 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2669: t120_path3 = 1 -> x_Garage_Area >= 1154
 ind_2670: t120_path4 = 1 -> Overall_Qual_calc >= 3
 ind_2671: t120_path4 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2672: t120_path4 = 1 -> const_Lot_Area <= 4426
 ind_2673: t120_path5 = 1 -> Overall_Qual_calc >= 3
 ind_2674: t120_path5 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2675: t120_path5 = 1 -> const_Lot_Area >= 4426
 ind_2676: t121_path0 = 1 -> x_Gr_Liv_Area <= 1155
 ind_2677: t121_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_2678: t121_path0 = 1 -> x_paved_drive_is_Y <= 1
 ind_2679: t121_path1 = 1 -> x_Gr_Liv_Area <= 1155
 ind_2680: t121_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_2681: t121_path1 = 1 -> x_paved_drive_is_Y >= 1
 ind_2682: t121_path2 = 1 -> x_Gr_Liv_Area <= 1155
 ind_2683: t121_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_2684: t121_path2 = 1 -> const_Garage_Yr_Blt <= 1931
 ind_2685: t121_path3 = 1 -> x_Gr_Liv_Area <= 1155
 ind_2686: t121_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_2687: t121_path3 = 1 -> const_Garage_Yr_Blt >= 1931
 ind_2688: t121_path4 = 1 -> x_Gr_Liv_Area >= 1155
 ind_2689: t121_path4 = 1 -> x_Mas_Vnr_Area <= 504
 ind_2690: t121_path4 = 1 -> x_BsmtFin_SF_1 <= 219
 ind_2691: t121_path5 = 1 -> x_Gr_Liv_Area >= 1155
 ind_2692: t121_path5 = 1 -> x_Mas_Vnr_Area <= 504
 ind_2693: t121_path5 = 1 -> x_BsmtFin_SF_1 >= 219
 ind_2694: t121_path6 = 1 -> x_Gr_Liv_Area >= 1155
 ind_2695: t121_path6 = 1 -> x_Mas_Vnr_Area >= 504
 ind_2696: t122_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2697: t122_path0 = 1 -> const_Lot_Area <= 5976
 ind_2698: t122_path0 = 1 -> const_MS_Zoning_FV <= 1
 ind_2699: t122_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2700: t122_path1 = 1 -> const_Lot_Area <= 5976
 ind_2701: t122_path1 = 1 -> const_MS_Zoning_FV >= 1
 ind_2702: t122_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2703: t122_path2 = 1 -> const_Lot_Area >= 5976
 ind_2704: t122_path2 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_2705: t122_path3 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2706: t122_path3 = 1 -> const_Lot_Area >= 5976
 ind_2707: t122_path3 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_2708: t122_path4 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2709: t123_path0 = 1 -> const_Yr_Sold <= 2008
 ind_2710: t123_path0 = 1 -> x_1st_Flr_SF <= 1811
 ind_2711: t123_path0 = 1 -> const_Overall_Cond <= 6
 ind_2712: t123_path1 = 1 -> const_Yr_Sold <= 2008
 ind_2713: t123_path1 = 1 -> x_1st_Flr_SF <= 1811
 ind_2714: t123_path1 = 1 -> const_Overall_Cond >= 6
 ind_2715: t123_path2 = 1 -> const_Yr_Sold <= 2008
 ind_2716: t123_path2 = 1 -> x_1st_Flr_SF >= 1811
 ind_2717: t123_path2 = 1 -> const_Lot_Shape_Reg <= 1
 ind_2718: t123_path3 = 1 -> const_Yr_Sold <= 2008
 ind_2719: t123_path3 = 1 -> x_1st_Flr_SF >= 1811
 ind_2720: t123_path3 = 1 -> const_Lot_Shape_Reg >= 1
 ind_2721: t123_path4 = 1 -> const_Yr_Sold >= 2008
 ind_2722: t123_path4 = 1 -> const_Year_Remod_Add <= 1954
 ind_2723: t123_path4 = 1 -> const_Land_Slope_Mod <= 1
 ind_2724: t123_path5 = 1 -> const_Yr_Sold >= 2008
 ind_2725: t123_path5 = 1 -> const_Year_Remod_Add <= 1954
 ind_2726: t123_path5 = 1 -> const_Land_Slope_Mod >= 1
 ind_2727: t123_path6 = 1 -> const_Yr_Sold >= 2008
 ind_2728: t123_path6 = 1 -> const_Year_Remod_Add >= 1954
 ind_2729: t123_path6 = 1 -> x_ex1_is_BrkFace <= 1
 ind_2730: t123_path7 = 1 -> const_Yr_Sold >= 2008
 ind_2731: t123_path7 = 1 -> const_Year_Remod_Add >= 1954
 ind_2732: t123_path7 = 1 -> x_ex1_is_BrkFace >= 1
 ind_2733: t124_path0 = 1 -> x_Wood_Deck_SF <= 216
 ind_2734: t124_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2735: t124_path0 = 1 -> const_Year_Remod_Add <= 1993
 ind_2736: t124_path1 = 1 -> x_Wood_Deck_SF <= 216
 ind_2737: t124_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2738: t124_path1 = 1 -> const_Year_Remod_Add >= 1993
 ind_2739: t124_path2 = 1 -> x_Wood_Deck_SF <= 216
 ind_2740: t124_path2 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2741: t124_path2 = 1 -> x_Garage_Area <= 497
 ind_2742: t124_path3 = 1 -> x_Wood_Deck_SF <= 216
 ind_2743: t124_path3 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2744: t124_path3 = 1 -> x_Garage_Area >= 497
 ind_2745: t124_path4 = 1 -> x_Wood_Deck_SF >= 216
 ind_2746: t124_path4 = 1 -> x_1st_Flr_SF <= 1811
 ind_2747: t124_path5 = 1 -> x_Wood_Deck_SF >= 216
 ind_2748: t124_path5 = 1 -> x_1st_Flr_SF >= 1811
 ind_2749: t125_path0 = 1 -> x_Heating_QC <= 2
 ind_2750: t125_path0 = 1 -> x_ex2_is_VinylSd <= 1
 ind_2751: t125_path1 = 1 -> x_Heating_QC <= 2
 ind_2752: t125_path1 = 1 -> x_ex2_is_VinylSd >= 1
 ind_2753: t125_path2 = 1 -> x_Heating_QC >= 2
 ind_2754: t125_path2 = 1 -> const_Fireplaces <= 2
 ind_2755: t125_path2 = 1 -> const_Sale_Type_New <= 1
 ind_2756: t125_path3 = 1 -> x_Heating_QC >= 2
 ind_2757: t125_path3 = 1 -> const_Fireplaces <= 2
 ind_2758: t125_path3 = 1 -> const_Sale_Type_New >= 1
 ind_2759: t125_path4 = 1 -> x_Heating_QC >= 2
 ind_2760: t125_path4 = 1 -> const_Fireplaces >= 2
 ind_2761: t125_path4 = 1 -> const_Lot_Frontage <= 117
 ind_2762: t125_path5 = 1 -> x_Heating_QC >= 2
 ind_2763: t125_path5 = 1 -> const_Fireplaces >= 2
 ind_2764: t125_path5 = 1 -> const_Lot_Frontage >= 117
 ind_2765: t126_path0 = 1 -> x_1st_Flr_SF <= 576
 ind_2766: t126_path1 = 1 -> x_1st_Flr_SF >= 576
 ind_2767: t126_path1 = 1 -> const_Lot_Area <= 19522
 ind_2768: t126_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_2769: t126_path2 = 1 -> x_1st_Flr_SF >= 576
 ind_2770: t126_path2 = 1 -> const_Lot_Area <= 19522
 ind_2771: t126_path2 = 1 -> const_MS_SubClass_30 >= 1
 ind_2772: t126_path3 = 1 -> x_1st_Flr_SF >= 576
 ind_2773: t126_path3 = 1 -> const_Lot_Area >= 19522
 ind_2774: t126_path3 = 1 -> const_Year_Remod_Add <= 1962
 ind_2775: t126_path4 = 1 -> x_1st_Flr_SF >= 576
 ind_2776: t126_path4 = 1 -> const_Lot_Area >= 19522
 ind_2777: t126_path4 = 1 -> const_Year_Remod_Add >= 1962
 ind_2778: t127_path0 = 1 -> Overall_Qual_calc <= 3
 ind_2779: t127_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_2780: t127_path1 = 1 -> Overall_Qual_calc <= 3
 ind_2781: t127_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_2782: t127_path2 = 1 -> Overall_Qual_calc >= 3
 ind_2783: t127_path2 = 1 -> const_Year_Built <= 1913
 ind_2784: t127_path2 = 1 -> x_Heating_QC <= 3
 ind_2785: t127_path3 = 1 -> Overall_Qual_calc >= 3
 ind_2786: t127_path3 = 1 -> const_Year_Built <= 1913
 ind_2787: t127_path3 = 1 -> x_Heating_QC >= 3
 ind_2788: t127_path4 = 1 -> Overall_Qual_calc >= 3
 ind_2789: t127_path4 = 1 -> const_Year_Built >= 1913
 ind_2790: t127_path4 = 1 -> const_Lot_Area <= 3675
 ind_2791: t127_path5 = 1 -> Overall_Qual_calc >= 3
 ind_2792: t127_path5 = 1 -> const_Year_Built >= 1913
 ind_2793: t127_path5 = 1 -> const_Lot_Area >= 3675
 ind_2794: t128_path0 = 1 -> Overall_Qual_calc <= 3
 ind_2795: t128_path1 = 1 -> Overall_Qual_calc >= 3
 ind_2796: t128_path1 = 1 -> const_Overall_Cond <= 7
 ind_2797: t128_path1 = 1 -> const_Garage_Yr_Blt <= 2005
 ind_2798: t128_path2 = 1 -> Overall_Qual_calc >= 3
 ind_2799: t128_path2 = 1 -> const_Overall_Cond <= 7
 ind_2800: t128_path2 = 1 -> const_Garage_Yr_Blt >= 2005
 ind_2801: t128_path3 = 1 -> Overall_Qual_calc >= 3
 ind_2802: t128_path3 = 1 -> const_Overall_Cond >= 7
 ind_2803: t128_path3 = 1 -> x_1st_Flr_SF <= 1320
 ind_2804: t128_path4 = 1 -> Overall_Qual_calc >= 3
 ind_2805: t128_path4 = 1 -> const_Overall_Cond >= 7
 ind_2806: t128_path4 = 1 -> x_1st_Flr_SF >= 1320
 ind_2807: t129_path0 = 1 -> x_Exter_Qual <= 2
 ind_2808: t129_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_2809: t129_path1 = 1 -> x_Exter_Qual <= 2
 ind_2810: t129_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_2811: t129_path2 = 1 -> x_Exter_Qual >= 2
 ind_2812: t129_path2 = 1 -> const_Neighborhood_Edwards <= 1
 ind_2813: t129_path2 = 1 -> Overall_Qual_calc <= 10
 ind_2814: t129_path3 = 1 -> x_Exter_Qual >= 2
 ind_2815: t129_path3 = 1 -> const_Neighborhood_Edwards <= 1
 ind_2816: t129_path3 = 1 -> Overall_Qual_calc >= 10
 ind_2817: t129_path4 = 1 -> x_Exter_Qual >= 2
 ind_2818: t129_path4 = 1 -> const_Neighborhood_Edwards >= 1
 ind_2819: t129_path4 = 1 -> const_Lot_Frontage <= 66
 ind_2820: t129_path5 = 1 -> x_Exter_Qual >= 2
 ind_2821: t129_path5 = 1 -> const_Neighborhood_Edwards >= 1
 ind_2822: t129_path5 = 1 -> const_Lot_Frontage >= 66
 ind_2823: t130_path0 = 1 -> x_Wood_Deck_SF <= 195
 ind_2824: t130_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2825: t130_path0 = 1 -> const_Year_Remod_Add <= 1995
 ind_2826: t130_path1 = 1 -> x_Wood_Deck_SF <= 195
 ind_2827: t130_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_2828: t130_path1 = 1 -> const_Year_Remod_Add >= 1995
 ind_2829: t130_path2 = 1 -> x_Wood_Deck_SF <= 195
 ind_2830: t130_path2 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2831: t130_path2 = 1 -> const_Lot_Frontage <= 54
 ind_2832: t130_path3 = 1 -> x_Wood_Deck_SF <= 195
 ind_2833: t130_path3 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_2834: t130_path3 = 1 -> const_Lot_Frontage >= 54
 ind_2835: t130_path4 = 1 -> x_Wood_Deck_SF >= 195
 ind_2836: t130_path4 = 1 -> x_Mas_Vnr_Area <= 300
 ind_2837: t130_path4 = 1 -> x_Bsmt_Unf_SF <= 1240
 ind_2838: t130_path5 = 1 -> x_Wood_Deck_SF >= 195
 ind_2839: t130_path5 = 1 -> x_Mas_Vnr_Area <= 300
 ind_2840: t130_path5 = 1 -> x_Bsmt_Unf_SF >= 1240
 ind_2841: t130_path6 = 1 -> x_Wood_Deck_SF >= 195
 ind_2842: t130_path6 = 1 -> x_Mas_Vnr_Area >= 300
 ind_2843: t130_path6 = 1 -> x_Fireplace_Qu <= 2
 ind_2844: t130_path7 = 1 -> x_Wood_Deck_SF >= 195
 ind_2845: t130_path7 = 1 -> x_Mas_Vnr_Area >= 300
 ind_2846: t130_path7 = 1 -> x_Fireplace_Qu >= 2
 ind_2847: t131_path0 = 1 -> x_Exter_Cond <= 2
 ind_2848: t131_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_2849: t131_path0 = 1 -> x_TotRms_AbvGrd <= 6
 ind_2850: t131_path1 = 1 -> x_Exter_Cond <= 2
 ind_2851: t131_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_2852: t131_path1 = 1 -> x_TotRms_AbvGrd >= 6
 ind_2853: t131_path2 = 1 -> x_Exter_Cond <= 2
 ind_2854: t131_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_2855: t131_path2 = 1 -> x_1st_Flr_SF <= 951
 ind_2856: t131_path3 = 1 -> x_Exter_Cond <= 2
 ind_2857: t131_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_2858: t131_path3 = 1 -> x_1st_Flr_SF >= 951
 ind_2859: t131_path4 = 1 -> x_Exter_Cond >= 2
 ind_2860: t131_path4 = 1 -> x_Screen_Porch <= 142
 ind_2861: t131_path4 = 1 -> const_Yr_Sold <= 2007
 ind_2862: t131_path5 = 1 -> x_Exter_Cond >= 2
 ind_2863: t131_path5 = 1 -> x_Screen_Porch <= 142
 ind_2864: t131_path5 = 1 -> const_Yr_Sold >= 2007
 ind_2865: t131_path6 = 1 -> x_Exter_Cond >= 2
 ind_2866: t131_path6 = 1 -> x_Screen_Porch >= 142
 ind_2867: t131_path6 = 1 -> const_Lot_Area <= 13052
 ind_2868: t131_path7 = 1 -> x_Exter_Cond >= 2
 ind_2869: t131_path7 = 1 -> x_Screen_Porch >= 142
 ind_2870: t131_path7 = 1 -> const_Lot_Area >= 13052
 ind_2871: t132_path0 = 1 -> x_Full_Bath <= 3
 ind_2872: t132_path0 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_2873: t132_path0 = 1 -> x_BsmtFin_SF_1 <= 748
 ind_2874: t132_path1 = 1 -> x_Full_Bath <= 3
 ind_2875: t132_path1 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_2876: t132_path1 = 1 -> x_BsmtFin_SF_1 >= 748
 ind_2877: t132_path2 = 1 -> x_Full_Bath <= 3
 ind_2878: t132_path2 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_2879: t132_path3 = 1 -> x_Full_Bath >= 3
 ind_2880: t132_path3 = 1 -> x_ex1_is_VinylSd <= 1
 ind_2881: t132_path4 = 1 -> x_Full_Bath >= 3
 ind_2882: t132_path4 = 1 -> x_ex1_is_VinylSd >= 1
 ind_2883: t133_path0 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_2884: t133_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_2885: t133_path0 = 1 -> x_Half_Bath <= 1
 ind_2886: t133_path1 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_2887: t133_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_2888: t133_path1 = 1 -> x_Half_Bath >= 1
 ind_2889: t133_path2 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_2890: t133_path2 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_2891: t133_path2 = 1 -> const_Fireplaces <= 2
 ind_2892: t133_path3 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_2893: t133_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_2894: t133_path3 = 1 -> const_Fireplaces >= 2
 ind_2895: t133_path4 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_2896: t134_path0 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_2897: t134_path0 = 1 -> x_Heating_QC <= 3
 ind_2898: t134_path0 = 1 -> const_Overall_Cond <= 6
 ind_2899: t134_path1 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_2900: t134_path1 = 1 -> x_Heating_QC <= 3
 ind_2901: t134_path1 = 1 -> const_Overall_Cond >= 6
 ind_2902: t134_path2 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_2903: t134_path2 = 1 -> x_Heating_QC >= 3
 ind_2904: t134_path2 = 1 -> const_Yr_Sold <= 2009
 ind_2905: t134_path3 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_2906: t134_path3 = 1 -> x_Heating_QC >= 3
 ind_2907: t134_path3 = 1 -> const_Yr_Sold >= 2009
 ind_2908: t134_path4 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_2909: t135_path0 = 1 -> const_Yr_Sold <= 2007
 ind_2910: t135_path0 = 1 -> const_Lot_Frontage <= 31
 ind_2911: t135_path1 = 1 -> const_Yr_Sold <= 2007
 ind_2912: t135_path1 = 1 -> const_Lot_Frontage >= 31
 ind_2913: t135_path1 = 1 -> const_Condition_1_Artery <= 1
 ind_2914: t135_path2 = 1 -> const_Yr_Sold <= 2007
 ind_2915: t135_path2 = 1 -> const_Lot_Frontage >= 31
 ind_2916: t135_path2 = 1 -> const_Condition_1_Artery >= 1
 ind_2917: t135_path3 = 1 -> const_Yr_Sold >= 2007
 ind_2918: t135_path3 = 1 -> x_Heating_QC <= 3
 ind_2919: t135_path3 = 1 -> x_Exter_Cond <= 2
 ind_2920: t135_path4 = 1 -> const_Yr_Sold >= 2007
 ind_2921: t135_path4 = 1 -> x_Heating_QC <= 3
 ind_2922: t135_path4 = 1 -> x_Exter_Cond >= 2
 ind_2923: t135_path5 = 1 -> const_Yr_Sold >= 2007
 ind_2924: t135_path5 = 1 -> x_Heating_QC >= 3
 ind_2925: t135_path5 = 1 -> const_FunctioNo_aplical_Maj2 <= 1
 ind_2926: t135_path6 = 1 -> const_Yr_Sold >= 2007
 ind_2927: t135_path6 = 1 -> x_Heating_QC >= 3
 ind_2928: t135_path6 = 1 -> const_FunctioNo_aplical_Maj2 >= 1
 ind_2929: t136_path0 = 1 -> x_Full_Bath <= 3
 ind_2930: t136_path0 = 1 -> x_Garage_Area <= 1154
 ind_2931: t136_path0 = 1 -> const_Yr_Sold <= 2008
 ind_2932: t136_path1 = 1 -> x_Full_Bath <= 3
 ind_2933: t136_path1 = 1 -> x_Garage_Area <= 1154
 ind_2934: t136_path1 = 1 -> const_Yr_Sold >= 2008
 ind_2935: t136_path2 = 1 -> x_Full_Bath <= 3
 ind_2936: t136_path2 = 1 -> x_Garage_Area >= 1154
 ind_2937: t136_path3 = 1 -> x_Full_Bath >= 3
 ind_2938: t136_path3 = 1 -> const_Garage_Finish_RFn <= 1
 ind_2939: t136_path4 = 1 -> x_Full_Bath >= 3
 ind_2940: t136_path4 = 1 -> const_Garage_Finish_RFn >= 1
 ind_2941: t137_path0 = 1 -> const_Lot_Area <= 6000
 ind_2942: t137_path0 = 1 -> Overall_Qual_calc <= 8
 ind_2943: t137_path0 = 1 -> x_Garage_Area <= 184
 ind_2944: t137_path1 = 1 -> const_Lot_Area <= 6000
 ind_2945: t137_path1 = 1 -> Overall_Qual_calc <= 8
 ind_2946: t137_path1 = 1 -> x_Garage_Area >= 184
 ind_2947: t137_path2 = 1 -> const_Lot_Area <= 6000
 ind_2948: t137_path2 = 1 -> Overall_Qual_calc >= 8
 ind_2949: t137_path2 = 1 -> x_BsmtFin_SF_1 <= 324
 ind_2950: t137_path3 = 1 -> const_Lot_Area <= 6000
 ind_2951: t137_path3 = 1 -> Overall_Qual_calc >= 8
 ind_2952: t137_path3 = 1 -> x_BsmtFin_SF_1 >= 324
 ind_2953: t137_path4 = 1 -> const_Lot_Area >= 6000
 ind_2954: t137_path4 = 1 -> const_2nd_Flr_SF <= 1427
 ind_2955: t137_path4 = 1 -> const_Yr_Sold <= 2007
 ind_2956: t137_path5 = 1 -> const_Lot_Area >= 6000
 ind_2957: t137_path5 = 1 -> const_2nd_Flr_SF <= 1427
 ind_2958: t137_path5 = 1 -> const_Yr_Sold >= 2007
 ind_2959: t137_path6 = 1 -> const_Lot_Area >= 6000
 ind_2960: t137_path6 = 1 -> const_2nd_Flr_SF >= 1427
 ind_2961: t137_path6 = 1 -> const_Lot_Area <= 9800
 ind_2962: t137_path7 = 1 -> const_Lot_Area >= 6000
 ind_2963: t137_path7 = 1 -> const_2nd_Flr_SF >= 1427
 ind_2964: t137_path7 = 1 -> const_Lot_Area >= 9800
 ind_2965: t138_path0 = 1 -> x_Exter_Cond <= 2
 ind_2966: t138_path0 = 1 -> Overall_Qual_calc <= 3
 ind_2967: t138_path1 = 1 -> x_Exter_Cond <= 2
 ind_2968: t138_path1 = 1 -> Overall_Qual_calc >= 3
 ind_2969: t138_path1 = 1 -> x_Bsmt_Unf_SF <= 720
 ind_2970: t138_path2 = 1 -> x_Exter_Cond <= 2
 ind_2971: t138_path2 = 1 -> Overall_Qual_calc >= 3
 ind_2972: t138_path2 = 1 -> x_Bsmt_Unf_SF >= 720
 ind_2973: t138_path3 = 1 -> x_Exter_Cond >= 2
 ind_2974: t138_path3 = 1 -> const_Lot_Area <= 5976
 ind_2975: t138_path3 = 1 -> x_Garage_Area <= 384
 ind_2976: t138_path4 = 1 -> x_Exter_Cond >= 2
 ind_2977: t138_path4 = 1 -> const_Lot_Area <= 5976
 ind_2978: t138_path4 = 1 -> x_Garage_Area >= 384
 ind_2979: t138_path5 = 1 -> x_Exter_Cond >= 2
 ind_2980: t138_path5 = 1 -> const_Lot_Area >= 5976
 ind_2981: t138_path5 = 1 -> const_Fireplaces <= 3
 ind_2982: t138_path6 = 1 -> x_Exter_Cond >= 2
 ind_2983: t138_path6 = 1 -> const_Lot_Area >= 5976
 ind_2984: t138_path6 = 1 -> const_Fireplaces >= 3
 ind_2985: t139_path0 = 1 -> const_Overall_Cond <= 7
 ind_2986: t139_path0 = 1 -> const_Neighborhood_OldTown <= 1
 ind_2987: t139_path0 = 1 -> x_Open_Porch_SF <= 45
 ind_2988: t139_path1 = 1 -> const_Overall_Cond <= 7
 ind_2989: t139_path1 = 1 -> const_Neighborhood_OldTown <= 1
 ind_2990: t139_path1 = 1 -> x_Open_Porch_SF >= 45
 ind_2991: t139_path2 = 1 -> const_Overall_Cond <= 7
 ind_2992: t139_path2 = 1 -> const_Neighborhood_OldTown >= 1
 ind_2993: t139_path2 = 1 -> const_Foundation_BrkTil <= 1
 ind_2994: t139_path3 = 1 -> const_Overall_Cond <= 7
 ind_2995: t139_path3 = 1 -> const_Neighborhood_OldTown >= 1
 ind_2996: t139_path3 = 1 -> const_Foundation_BrkTil >= 1
 ind_2997: t139_path4 = 1 -> const_Overall_Cond >= 7
 ind_2998: t139_path4 = 1 -> const_Condition_1_Norm <= 1
 ind_2999: t139_path4 = 1 -> const_Total_Bsmt_SF <= 904
 ind_3000: t139_path5 = 1 -> const_Overall_Cond >= 7
 ind_3001: t139_path5 = 1 -> const_Condition_1_Norm <= 1
 ind_3002: t139_path5 = 1 -> const_Total_Bsmt_SF >= 904
 ind_3003: t139_path6 = 1 -> const_Overall_Cond >= 7
 ind_3004: t139_path6 = 1 -> const_Condition_1_Norm >= 1
 ind_3005: t139_path6 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3006: t139_path7 = 1 -> const_Overall_Cond >= 7
 ind_3007: t139_path7 = 1 -> const_Condition_1_Norm >= 1
 ind_3008: t139_path7 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3009: t140_path0 = 1 -> x_Bsmt_Cond <= 2
 ind_3010: t140_path0 = 1 -> x_Exter_Cond <= 2
 ind_3011: t140_path0 = 1 -> const_Lot_Frontage <= 65
 ind_3012: t140_path1 = 1 -> x_Bsmt_Cond <= 2
 ind_3013: t140_path1 = 1 -> x_Exter_Cond <= 2
 ind_3014: t140_path1 = 1 -> const_Lot_Frontage >= 65
 ind_3015: t140_path2 = 1 -> x_Bsmt_Cond <= 2
 ind_3016: t140_path2 = 1 -> x_Exter_Cond >= 2
 ind_3017: t140_path2 = 1 -> x_1st_Flr_SF <= 1124
 ind_3018: t140_path3 = 1 -> x_Bsmt_Cond <= 2
 ind_3019: t140_path3 = 1 -> x_Exter_Cond >= 2
 ind_3020: t140_path3 = 1 -> x_1st_Flr_SF >= 1124
 ind_3021: t140_path4 = 1 -> x_Bsmt_Cond >= 2
 ind_3022: t140_path4 = 1 -> const_Overall_Cond <= 7
 ind_3023: t140_path4 = 1 -> x_Open_Porch_SF <= 30
 ind_3024: t140_path5 = 1 -> x_Bsmt_Cond >= 2
 ind_3025: t140_path5 = 1 -> const_Overall_Cond <= 7
 ind_3026: t140_path5 = 1 -> x_Open_Porch_SF >= 30
 ind_3027: t140_path6 = 1 -> x_Bsmt_Cond >= 2
 ind_3028: t140_path6 = 1 -> const_Overall_Cond >= 7
 ind_3029: t140_path6 = 1 -> const_Lot_Area <= 18386
 ind_3030: t140_path7 = 1 -> x_Bsmt_Cond >= 2
 ind_3031: t140_path7 = 1 -> const_Overall_Cond >= 7
 ind_3032: t140_path7 = 1 -> const_Lot_Area >= 18386
 ind_3033: t141_path0 = 1 -> const_Year_Built <= 1925
 ind_3034: t141_path0 = 1 -> x_Garage_Area <= 625
 ind_3035: t141_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_3036: t141_path1 = 1 -> const_Year_Built <= 1925
 ind_3037: t141_path1 = 1 -> x_Garage_Area <= 625
 ind_3038: t141_path1 = 1 -> const_Condition_1_Norm >= 1
 ind_3039: t141_path2 = 1 -> const_Year_Built <= 1925
 ind_3040: t141_path2 = 1 -> x_Garage_Area >= 625
 ind_3041: t141_path3 = 1 -> const_Year_Built >= 1925
 ind_3042: t141_path3 = 1 -> const_2nd_Flr_SF <= 1629
 ind_3043: t141_path3 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3044: t141_path4 = 1 -> const_Year_Built >= 1925
 ind_3045: t141_path4 = 1 -> const_2nd_Flr_SF <= 1629
 ind_3046: t141_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3047: t141_path5 = 1 -> const_Year_Built >= 1925
 ind_3048: t141_path5 = 1 -> const_2nd_Flr_SF >= 1629
 ind_3049: t142_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_3050: t142_path0 = 1 -> const_Sale_Type_New <= 1
 ind_3051: t142_path0 = 1 -> x_Gr_Liv_Area <= 2633
 ind_3052: t142_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_3053: t142_path1 = 1 -> const_Sale_Type_New <= 1
 ind_3054: t142_path1 = 1 -> x_Gr_Liv_Area >= 2633
 ind_3055: t142_path2 = 1 -> x_1st_Flr_SF <= 2515
 ind_3056: t142_path2 = 1 -> const_Sale_Type_New >= 1
 ind_3057: t142_path2 = 1 -> x_Gr_Liv_Area <= 1360
 ind_3058: t142_path3 = 1 -> x_1st_Flr_SF <= 2515
 ind_3059: t142_path3 = 1 -> const_Sale_Type_New >= 1
 ind_3060: t142_path3 = 1 -> x_Gr_Liv_Area >= 1360
 ind_3061: t142_path4 = 1 -> x_1st_Flr_SF >= 2515
 ind_3062: t143_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_3063: t143_path0 = 1 -> x_Gr_Liv_Area <= 720
 ind_3064: t143_path0 = 1 -> x_TotRms_AbvGrd <= 4
 ind_3065: t143_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_3066: t143_path1 = 1 -> x_Gr_Liv_Area <= 720
 ind_3067: t143_path1 = 1 -> x_TotRms_AbvGrd >= 4
 ind_3068: t143_path2 = 1 -> x_Gr_Liv_Area <= 747
 ind_3069: t143_path2 = 1 -> x_Gr_Liv_Area >= 720
 ind_3070: t143_path3 = 1 -> x_Gr_Liv_Area >= 747
 ind_3071: t143_path3 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3072: t143_path3 = 1 -> const_Overall_Cond <= 6
 ind_3073: t143_path4 = 1 -> x_Gr_Liv_Area >= 747
 ind_3074: t143_path4 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3075: t143_path4 = 1 -> const_Overall_Cond >= 6
 ind_3076: t143_path5 = 1 -> x_Gr_Liv_Area >= 747
 ind_3077: t143_path5 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3078: t143_path5 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3079: t143_path6 = 1 -> x_Gr_Liv_Area >= 747
 ind_3080: t143_path6 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3081: t143_path6 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3082: t144_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3083: t144_path0 = 1 -> const_Yr_Sold <= 2008
 ind_3084: t144_path0 = 1 -> x_Full_Bath <= 3
 ind_3085: t144_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3086: t144_path1 = 1 -> const_Yr_Sold <= 2008
 ind_3087: t144_path1 = 1 -> x_Full_Bath >= 3
 ind_3088: t144_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3089: t144_path2 = 1 -> const_Yr_Sold >= 2008
 ind_3090: t144_path2 = 1 -> const_Sale_Condition_Normal <= 1
 ind_3091: t144_path3 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3092: t144_path3 = 1 -> const_Yr_Sold >= 2008
 ind_3093: t144_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3094: t144_path4 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_3095: t144_path4 = 1 -> const_Lot_Area <= 4835
 ind_3096: t144_path5 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_3097: t144_path5 = 1 -> const_Lot_Area >= 4835
 ind_3098: t145_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_3099: t145_path0 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_3100: t145_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3101: t145_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_3102: t145_path1 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_3103: t145_path1 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3104: t145_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_3105: t145_path2 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_3106: t145_path2 = 1 -> x_ex2_is_Stucco <= 1
 ind_3107: t145_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_3108: t145_path3 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_3109: t145_path3 = 1 -> x_ex2_is_Stucco >= 1
 ind_3110: t145_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_3111: t145_path4 = 1 -> const_Lot_Area <= 8930
 ind_3112: t145_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_3113: t145_path5 = 1 -> const_Lot_Area >= 8930
 ind_3114: t146_path0 = 1 -> x_BsmtFin_SF_1 <= 425
 ind_3115: t146_path0 = 1 -> x_Heating_QC <= 3
 ind_3116: t146_path0 = 1 -> x_ex2_is_VinylSd <= 1
 ind_3117: t146_path1 = 1 -> x_BsmtFin_SF_1 <= 425
 ind_3118: t146_path1 = 1 -> x_Heating_QC <= 3
 ind_3119: t146_path1 = 1 -> x_ex2_is_VinylSd >= 1
 ind_3120: t146_path2 = 1 -> x_BsmtFin_SF_1 <= 425
 ind_3121: t146_path2 = 1 -> x_Heating_QC >= 3
 ind_3122: t146_path2 = 1 -> const_Lot_Area <= 16779
 ind_3123: t146_path3 = 1 -> x_BsmtFin_SF_1 <= 425
 ind_3124: t146_path3 = 1 -> x_Heating_QC >= 3
 ind_3125: t146_path3 = 1 -> const_Lot_Area >= 16779
 ind_3126: t146_path4 = 1 -> x_BsmtFin_SF_1 >= 425
 ind_3127: t146_path4 = 1 -> const_Total_Bsmt_SF <= 1498
 ind_3128: t146_path4 = 1 -> x_Bsmt_Cond <= 3
 ind_3129: t146_path5 = 1 -> x_BsmtFin_SF_1 >= 425
 ind_3130: t146_path5 = 1 -> const_Total_Bsmt_SF <= 1498
 ind_3131: t146_path5 = 1 -> x_Bsmt_Cond >= 3
 ind_3132: t146_path6 = 1 -> x_BsmtFin_SF_1 >= 425
 ind_3133: t146_path6 = 1 -> const_Total_Bsmt_SF >= 1498
 ind_3134: t146_path6 = 1 -> const_Bsmt_Exposure_Mn <= 1
 ind_3135: t146_path7 = 1 -> x_BsmtFin_SF_1 >= 425
 ind_3136: t146_path7 = 1 -> const_Total_Bsmt_SF >= 1498
 ind_3137: t146_path7 = 1 -> const_Bsmt_Exposure_Mn >= 1
 ind_3138: t147_path0 = 1 -> x_1st_Flr_SF <= 616
 ind_3139: t147_path0 = 1 -> const_Year_Remod_Add <= 1956
 ind_3140: t147_path1 = 1 -> x_1st_Flr_SF <= 616
 ind_3141: t147_path1 = 1 -> const_Year_Remod_Add >= 1956
 ind_3142: t147_path2 = 1 -> x_1st_Flr_SF >= 616
 ind_3143: t147_path2 = 1 -> const_Condition_1_Feedr <= 1
 ind_3144: t147_path2 = 1 -> x_BsmtFin_SF_1 <= 700
 ind_3145: t147_path3 = 1 -> x_1st_Flr_SF >= 616
 ind_3146: t147_path3 = 1 -> const_Condition_1_Feedr <= 1
 ind_3147: t147_path3 = 1 -> x_BsmtFin_SF_1 >= 700
 ind_3148: t147_path4 = 1 -> x_1st_Flr_SF >= 616
 ind_3149: t147_path4 = 1 -> const_Condition_1_Feedr >= 1
 ind_3150: t147_path4 = 1 -> const_Mo_Sold_1 <= 1
 ind_3151: t147_path5 = 1 -> x_1st_Flr_SF >= 616
 ind_3152: t147_path5 = 1 -> const_Condition_1_Feedr >= 1
 ind_3153: t147_path5 = 1 -> const_Mo_Sold_1 >= 1
 ind_3154: t148_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3155: t148_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3156: t148_path0 = 1 -> const_Total_Bsmt_SF <= 585
 ind_3157: t148_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3158: t148_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3159: t148_path1 = 1 -> const_Total_Bsmt_SF >= 585
 ind_3160: t148_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3161: t148_path2 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3162: t148_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3163: t148_path3 = 1 -> Overall_Qual_calc <= 4
 ind_3164: t148_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3165: t148_path4 = 1 -> Overall_Qual_calc >= 4
 ind_3166: t148_path4 = 1 -> const_Fireplaces <= 1
 ind_3167: t148_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3168: t148_path5 = 1 -> Overall_Qual_calc >= 4
 ind_3169: t148_path5 = 1 -> const_Fireplaces >= 1
 ind_3170: t149_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3171: t149_path0 = 1 -> x_ex2_is_Wd_Sdng <= 1
 ind_3172: t149_path0 = 1 -> x_BsmtFin_SF_1 <= 188
 ind_3173: t149_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3174: t149_path1 = 1 -> x_ex2_is_Wd_Sdng <= 1
 ind_3175: t149_path1 = 1 -> x_BsmtFin_SF_1 >= 188
 ind_3176: t149_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3177: t149_path2 = 1 -> x_ex2_is_Wd_Sdng >= 1
 ind_3178: t149_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3179: t149_path3 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3180: t149_path3 = 1 -> const_Overall_Cond <= 6
 ind_3181: t149_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3182: t149_path4 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3183: t149_path4 = 1 -> const_Overall_Cond >= 6
 ind_3184: t149_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3185: t149_path5 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3186: t149_path5 = 1 -> const_Neighborhood_No_aplicames <= 1
 ind_3187: t149_path6 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3188: t149_path6 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3189: t149_path6 = 1 -> const_Neighborhood_No_aplicames >= 1
 ind_3190: t150_path0 = 1 -> x_Exter_Cond <= 2
 ind_3191: t150_path0 = 1 -> x_Garage_Area <= 297
 ind_3192: t150_path1 = 1 -> x_Exter_Cond <= 2
 ind_3193: t150_path1 = 1 -> x_Garage_Area >= 297
 ind_3194: t150_path1 = 1 -> const_Total_Bsmt_SF <= 817
 ind_3195: t150_path2 = 1 -> x_Exter_Cond <= 2
 ind_3196: t150_path2 = 1 -> x_Garage_Area >= 297
 ind_3197: t150_path2 = 1 -> const_Total_Bsmt_SF >= 817
 ind_3198: t150_path3 = 1 -> x_Exter_Cond >= 2
 ind_3199: t150_path3 = 1 -> const_Lot_Area <= 5976
 ind_3200: t150_path3 = 1 -> x_Gr_Liv_Area <= 996
 ind_3201: t150_path4 = 1 -> x_Exter_Cond >= 2
 ind_3202: t150_path4 = 1 -> const_Lot_Area <= 5976
 ind_3203: t150_path4 = 1 -> x_Gr_Liv_Area >= 996
 ind_3204: t150_path5 = 1 -> x_Exter_Cond >= 2
 ind_3205: t150_path5 = 1 -> const_Lot_Area >= 5976
 ind_3206: t150_path5 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3207: t150_path6 = 1 -> x_Exter_Cond >= 2
 ind_3208: t150_path6 = 1 -> const_Lot_Area >= 5976
 ind_3209: t150_path6 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_3210: t151_path0 = 1 -> x_Garage_Area <= 804
 ind_3211: t151_path0 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_3212: t151_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3213: t151_path1 = 1 -> x_Garage_Area <= 804
 ind_3214: t151_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_3215: t151_path1 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3216: t151_path2 = 1 -> x_Garage_Area <= 804
 ind_3217: t151_path2 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_3218: t151_path2 = 1 -> const_Lot_Area <= 16560
 ind_3219: t151_path3 = 1 -> x_Garage_Area <= 804
 ind_3220: t151_path3 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_3221: t151_path3 = 1 -> const_Lot_Area >= 16560
 ind_3222: t151_path4 = 1 -> x_Garage_Area >= 804
 ind_3223: t152_path0 = 1 -> x_Screen_Porch <= 150
 ind_3224: t152_path0 = 1 -> x_Exter_Cond <= 2
 ind_3225: t152_path0 = 1 -> const_Lot_Area <= 9171
 ind_3226: t152_path1 = 1 -> x_Screen_Porch <= 150
 ind_3227: t152_path1 = 1 -> x_Exter_Cond <= 2
 ind_3228: t152_path1 = 1 -> const_Lot_Area >= 9171
 ind_3229: t152_path2 = 1 -> x_Screen_Porch <= 150
 ind_3230: t152_path2 = 1 -> x_Exter_Cond >= 2
 ind_3231: t152_path2 = 1 -> const_Land_Slope_Gtl <= 1
 ind_3232: t152_path3 = 1 -> x_Screen_Porch <= 150
 ind_3233: t152_path3 = 1 -> x_Exter_Cond >= 2
 ind_3234: t152_path3 = 1 -> const_Land_Slope_Gtl >= 1
 ind_3235: t152_path4 = 1 -> x_Screen_Porch >= 150
 ind_3236: t152_path4 = 1 -> const_2nd_Flr_SF <= 978
 ind_3237: t152_path5 = 1 -> x_Screen_Porch >= 150
 ind_3238: t152_path5 = 1 -> const_2nd_Flr_SF >= 978
 ind_3239: t153_path0 = 1 -> const_Lot_Area <= 6000
 ind_3240: t153_path0 = 1 -> const_Garage_Yr_Blt <= 1895
 ind_3241: t153_path1 = 1 -> const_Lot_Area <= 6000
 ind_3242: t153_path1 = 1 -> const_Garage_Yr_Blt >= 1895
 ind_3243: t153_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_3244: t153_path2 = 1 -> const_Lot_Area <= 6000
 ind_3245: t153_path2 = 1 -> const_Garage_Yr_Blt >= 1895
 ind_3246: t153_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3247: t153_path3 = 1 -> const_Lot_Area >= 6000
 ind_3248: t153_path3 = 1 -> x_Screen_Porch <= 142
 ind_3249: t153_path3 = 1 -> const_Year_Built <= 1913
 ind_3250: t153_path4 = 1 -> const_Lot_Area >= 6000
 ind_3251: t153_path4 = 1 -> x_Screen_Porch <= 142
 ind_3252: t153_path4 = 1 -> const_Year_Built >= 1913
 ind_3253: t153_path5 = 1 -> const_Lot_Area >= 6000
 ind_3254: t153_path5 = 1 -> x_Screen_Porch >= 142
 ind_3255: t153_path5 = 1 -> x_Gr_Liv_Area <= 1930
 ind_3256: t153_path6 = 1 -> const_Lot_Area >= 6000
 ind_3257: t153_path6 = 1 -> x_Screen_Porch >= 142
 ind_3258: t153_path6 = 1 -> x_Gr_Liv_Area >= 1930
 ind_3259: t154_path0 = 1 -> x_Heating_QC <= 3
 ind_3260: t154_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_3261: t154_path0 = 1 -> x_Gr_Liv_Area <= 976
 ind_3262: t154_path1 = 1 -> x_Heating_QC <= 3
 ind_3263: t154_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_3264: t154_path1 = 1 -> x_Gr_Liv_Area >= 976
 ind_3265: t154_path2 = 1 -> x_Heating_QC <= 3
 ind_3266: t154_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3267: t154_path2 = 1 -> const_Lot_Area <= 10896
 ind_3268: t154_path3 = 1 -> x_Heating_QC <= 3
 ind_3269: t154_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3270: t154_path3 = 1 -> const_Lot_Area >= 10896
 ind_3271: t154_path4 = 1 -> x_Heating_QC >= 3
 ind_3272: t154_path4 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3273: t154_path4 = 1 -> const_House_Style_1.5Fin <= 1
 ind_3274: t154_path5 = 1 -> x_Heating_QC >= 3
 ind_3275: t154_path5 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3276: t154_path5 = 1 -> const_House_Style_1.5Fin >= 1
 ind_3277: t154_path6 = 1 -> x_Heating_QC >= 3
 ind_3278: t154_path6 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3279: t154_path6 = 1 -> x_Exter_Qual <= 3
 ind_3280: t154_path7 = 1 -> x_Heating_QC >= 3
 ind_3281: t154_path7 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3282: t154_path7 = 1 -> x_Exter_Qual >= 3
 ind_3283: t155_path0 = 1 -> x_Garage_Area <= 1154
 ind_3284: t155_path0 = 1 -> const_Lot_Area <= 17871
 ind_3285: t155_path0 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_3286: t155_path1 = 1 -> x_Garage_Area <= 1154
 ind_3287: t155_path1 = 1 -> const_Lot_Area <= 17871
 ind_3288: t155_path1 = 1 -> const_Garage_Yr_Blt >= 2008
 ind_3289: t155_path2 = 1 -> x_Garage_Area <= 1154
 ind_3290: t155_path2 = 1 -> const_Lot_Area >= 17871
 ind_3291: t155_path2 = 1 -> const_Neighborhood_Mitchel <= 1
 ind_3292: t155_path3 = 1 -> x_Garage_Area <= 1154
 ind_3293: t155_path3 = 1 -> const_Lot_Area >= 17871
 ind_3294: t155_path3 = 1 -> const_Neighborhood_Mitchel >= 1
 ind_3295: t155_path4 = 1 -> x_Garage_Area >= 1154
 ind_3296: t156_path0 = 1 -> x_Screen_Porch <= 40
 ind_3297: t156_path0 = 1 -> const_Year_Built <= 2007
 ind_3298: t156_path0 = 1 -> x_BsmtFin_SF_1 <= 56
 ind_3299: t156_path1 = 1 -> x_Screen_Porch <= 40
 ind_3300: t156_path1 = 1 -> const_Year_Built <= 2007
 ind_3301: t156_path1 = 1 -> x_BsmtFin_SF_1 >= 56
 ind_3302: t156_path2 = 1 -> x_Screen_Porch <= 40
 ind_3303: t156_path2 = 1 -> const_Year_Built >= 2007
 ind_3304: t156_path3 = 1 -> x_Screen_Porch >= 40
 ind_3305: t156_path3 = 1 -> x_Gr_Liv_Area <= 1977
 ind_3306: t156_path4 = 1 -> x_Screen_Porch >= 40
 ind_3307: t156_path4 = 1 -> x_Gr_Liv_Area >= 1977
 ind_3308: t157_path0 = 1 -> const_Lot_Area <= 7950
 ind_3309: t157_path0 = 1 -> x_Gr_Liv_Area <= 1128
 ind_3310: t157_path0 = 1 -> x_BsmtFin_SF_1 <= 533
 ind_3311: t157_path1 = 1 -> const_Lot_Area <= 7950
 ind_3312: t157_path1 = 1 -> x_Gr_Liv_Area <= 1128
 ind_3313: t157_path1 = 1 -> x_BsmtFin_SF_1 >= 533
 ind_3314: t157_path2 = 1 -> const_Lot_Area <= 7950
 ind_3315: t157_path2 = 1 -> x_Gr_Liv_Area >= 1128
 ind_3316: t157_path2 = 1 -> const_FunctioNo_aplical_Maj1 <= 1
 ind_3317: t157_path3 = 1 -> const_Lot_Area <= 7950
 ind_3318: t157_path3 = 1 -> x_Gr_Liv_Area >= 1128
 ind_3319: t157_path3 = 1 -> const_FunctioNo_aplical_Maj1 >= 1
 ind_3320: t157_path4 = 1 -> const_Lot_Area >= 7950
 ind_3321: t157_path4 = 1 -> const_2nd_Flr_SF <= 1427
 ind_3322: t157_path4 = 1 -> const_Condition_1_Norm <= 1
 ind_3323: t157_path5 = 1 -> const_Lot_Area >= 7950
 ind_3324: t157_path5 = 1 -> const_2nd_Flr_SF <= 1427
 ind_3325: t157_path5 = 1 -> const_Condition_1_Norm >= 1
 ind_3326: t157_path6 = 1 -> const_Lot_Area >= 7950
 ind_3327: t157_path6 = 1 -> const_2nd_Flr_SF >= 1427
 ind_3328: t158_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3329: t158_path0 = 1 -> x_Screen_Porch <= 121
 ind_3330: t158_path0 = 1 -> const_Lot_Area <= 13518
 ind_3331: t158_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3332: t158_path1 = 1 -> x_Screen_Porch <= 121
 ind_3333: t158_path1 = 1 -> const_Lot_Area >= 13518
 ind_3334: t158_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3335: t158_path2 = 1 -> x_Screen_Porch >= 121
 ind_3336: t158_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3337: t158_path3 = 1 -> const_Yr_Sold <= 2007
 ind_3338: t158_path3 = 1 -> const_Lot_Area <= 8127
 ind_3339: t158_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3340: t158_path4 = 1 -> const_Yr_Sold <= 2007
 ind_3341: t158_path4 = 1 -> const_Lot_Area >= 8127
 ind_3342: t158_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3343: t158_path5 = 1 -> const_Yr_Sold >= 2007
 ind_3344: t158_path5 = 1 -> x_Gr_Liv_Area <= 1155
 ind_3345: t158_path6 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3346: t158_path6 = 1 -> const_Yr_Sold >= 2007
 ind_3347: t158_path6 = 1 -> x_Gr_Liv_Area >= 1155
 ind_3348: t159_path0 = 1 -> Overall_Qual_calc <= 9
 ind_3349: t159_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_3350: t159_path0 = 1 -> x_Heating_QC <= 3
 ind_3351: t159_path1 = 1 -> Overall_Qual_calc <= 9
 ind_3352: t159_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_3353: t159_path1 = 1 -> x_Heating_QC >= 3
 ind_3354: t159_path2 = 1 -> Overall_Qual_calc <= 9
 ind_3355: t159_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_3356: t159_path2 = 1 -> const_Overall_Cond <= 6
 ind_3357: t159_path3 = 1 -> Overall_Qual_calc <= 9
 ind_3358: t159_path3 = 1 -> x_Gr_Liv_Area >= 747
 ind_3359: t159_path3 = 1 -> const_Overall_Cond >= 6
 ind_3360: t159_path4 = 1 -> Overall_Qual_calc >= 9
 ind_3361: t160_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3362: t160_path0 = 1 -> const_Yr_Sold <= 2007
 ind_3363: t160_path0 = 1 -> const_Lot_Area <= 16779
 ind_3364: t160_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3365: t160_path1 = 1 -> const_Yr_Sold <= 2007
 ind_3366: t160_path1 = 1 -> const_Lot_Area >= 16779
 ind_3367: t160_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3368: t160_path2 = 1 -> const_Yr_Sold >= 2007
 ind_3369: t160_path2 = 1 -> x_Screen_Porch <= 142
 ind_3370: t160_path3 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3371: t160_path3 = 1 -> const_Yr_Sold >= 2007
 ind_3372: t160_path3 = 1 -> x_Screen_Porch >= 142
 ind_3373: t160_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3374: t160_path4 = 1 -> Overall_Qual_calc <= 4
 ind_3375: t160_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3376: t160_path5 = 1 -> Overall_Qual_calc >= 4
 ind_3377: t160_path5 = 1 -> x_Heating_QC <= 3
 ind_3378: t160_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3379: t160_path6 = 1 -> Overall_Qual_calc >= 4
 ind_3380: t160_path6 = 1 -> x_Heating_QC >= 3
 ind_3381: t161_path0 = 1 -> const_Land_Contour_Bnk <= 1
 ind_3382: t161_path0 = 1 -> const_Land_Slope_Gtl <= 1
 ind_3383: t161_path0 = 1 -> const_Lot_Area <= 10044
 ind_3384: t161_path1 = 1 -> const_Land_Contour_Bnk <= 1
 ind_3385: t161_path1 = 1 -> const_Land_Slope_Gtl <= 1
 ind_3386: t161_path1 = 1 -> const_Lot_Area >= 10044
 ind_3387: t161_path2 = 1 -> const_Land_Contour_Bnk <= 1
 ind_3388: t161_path2 = 1 -> const_Land_Slope_Gtl >= 1
 ind_3389: t161_path2 = 1 -> x_1st_Flr_SF <= 2121
 ind_3390: t161_path3 = 1 -> const_Land_Contour_Bnk <= 1
 ind_3391: t161_path3 = 1 -> const_Land_Slope_Gtl >= 1
 ind_3392: t161_path3 = 1 -> x_1st_Flr_SF >= 2121
 ind_3393: t161_path4 = 1 -> const_Land_Contour_Bnk >= 1
 ind_3394: t161_path4 = 1 -> const_Lot_Frontage <= 111
 ind_3395: t161_path4 = 1 -> const_Lot_Frontage <= 69
 ind_3396: t161_path5 = 1 -> const_Land_Contour_Bnk >= 1
 ind_3397: t161_path5 = 1 -> const_Lot_Frontage <= 111
 ind_3398: t161_path5 = 1 -> const_Lot_Frontage >= 69
 ind_3399: t161_path6 = 1 -> const_Land_Contour_Bnk >= 1
 ind_3400: t161_path6 = 1 -> const_Lot_Frontage >= 111
 ind_3401: t162_path0 = 1 -> x_Exter_Cond <= 2
 ind_3402: t162_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3403: t162_path0 = 1 -> x_Garage_Area <= 345
 ind_3404: t162_path1 = 1 -> x_Exter_Cond <= 2
 ind_3405: t162_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3406: t162_path1 = 1 -> x_Garage_Area >= 345
 ind_3407: t162_path2 = 1 -> x_Exter_Cond <= 2
 ind_3408: t162_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3409: t162_path3 = 1 -> x_Exter_Cond >= 2
 ind_3410: t162_path3 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3411: t162_path3 = 1 -> x_Screen_Porch <= 150
 ind_3412: t162_path4 = 1 -> x_Exter_Cond >= 2
 ind_3413: t162_path4 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3414: t162_path4 = 1 -> x_Screen_Porch >= 150
 ind_3415: t162_path5 = 1 -> x_Exter_Cond >= 2
 ind_3416: t162_path5 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3417: t162_path5 = 1 -> x_Gr_Liv_Area <= 1242
 ind_3418: t162_path6 = 1 -> x_Exter_Cond >= 2
 ind_3419: t162_path6 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3420: t162_path6 = 1 -> x_Gr_Liv_Area >= 1242
 ind_3421: t163_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3422: t163_path0 = 1 -> x_Wood_Deck_SF <= 199
 ind_3423: t163_path0 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_3424: t163_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3425: t163_path1 = 1 -> x_Wood_Deck_SF <= 199
 ind_3426: t163_path1 = 1 -> const_Garage_Yr_Blt >= 2008
 ind_3427: t163_path2 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3428: t163_path2 = 1 -> x_Wood_Deck_SF >= 199
 ind_3429: t163_path2 = 1 -> x_Mas_Vnr_Area <= 754
 ind_3430: t163_path3 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3431: t163_path3 = 1 -> x_Wood_Deck_SF >= 199
 ind_3432: t163_path3 = 1 -> x_Mas_Vnr_Area >= 754
 ind_3433: t163_path4 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3434: t163_path4 = 1 -> const_Lot_Area <= 9605
 ind_3435: t163_path5 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3436: t163_path5 = 1 -> const_Lot_Area >= 9605
 ind_3437: t164_path0 = 1 -> const_2nd_Flr_SF <= 1427
 ind_3438: t164_path0 = 1 -> x_Bsmt_Cond <= 0
 ind_3439: t164_path0 = 1 -> x_Garage_Area <= 457
 ind_3440: t164_path1 = 1 -> const_2nd_Flr_SF <= 1427
 ind_3441: t164_path1 = 1 -> x_Bsmt_Cond <= 0
 ind_3442: t164_path1 = 1 -> x_Garage_Area >= 457
 ind_3443: t164_path2 = 1 -> const_2nd_Flr_SF <= 1427
 ind_3444: t164_path2 = 1 -> x_Bsmt_Cond >= 0
 ind_3445: t164_path2 = 1 -> const_Sale_Type_New <= 1
 ind_3446: t164_path3 = 1 -> const_2nd_Flr_SF <= 1427
 ind_3447: t164_path3 = 1 -> x_Bsmt_Cond >= 0
 ind_3448: t164_path3 = 1 -> const_Sale_Type_New >= 1
 ind_3449: t164_path4 = 1 -> const_2nd_Flr_SF >= 1427
 ind_3450: t165_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_3451: t165_path0 = 1 -> const_Total_Bsmt_SF <= 1824
 ind_3452: t165_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3453: t165_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_3454: t165_path1 = 1 -> const_Total_Bsmt_SF <= 1824
 ind_3455: t165_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3456: t165_path2 = 1 -> x_1st_Flr_SF <= 2515
 ind_3457: t165_path2 = 1 -> const_Total_Bsmt_SF >= 1824
 ind_3458: t165_path2 = 1 -> const_Neighborhood_NridgHt <= 1
 ind_3459: t165_path3 = 1 -> x_1st_Flr_SF <= 2515
 ind_3460: t165_path3 = 1 -> const_Total_Bsmt_SF >= 1824
 ind_3461: t165_path3 = 1 -> const_Neighborhood_NridgHt >= 1
 ind_3462: t165_path4 = 1 -> x_1st_Flr_SF >= 2515
 ind_3463: t166_path0 = 1 -> const_2nd_Flr_SF <= 1629
 ind_3464: t166_path0 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_3465: t166_path0 = 1 -> const_Sale_Condition_Partial <= 1
 ind_3466: t166_path1 = 1 -> const_2nd_Flr_SF <= 1629
 ind_3467: t166_path1 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_3468: t166_path1 = 1 -> const_Sale_Condition_Partial >= 1
 ind_3469: t166_path2 = 1 -> const_2nd_Flr_SF <= 1629
 ind_3470: t166_path2 = 1 -> const_Neighborhood_ClearCr >= 1
 ind_3471: t166_path3 = 1 -> const_2nd_Flr_SF >= 1629
 ind_3472: t167_path0 = 1 -> Overall_Qual_calc <= 3
 ind_3473: t167_path1 = 1 -> Overall_Qual_calc >= 3
 ind_3474: t167_path1 = 1 -> const_Overall_Cond <= 7
 ind_3475: t167_path1 = 1 -> const_Year_Built <= 1911
 ind_3476: t167_path2 = 1 -> Overall_Qual_calc >= 3
 ind_3477: t167_path2 = 1 -> const_Overall_Cond <= 7
 ind_3478: t167_path2 = 1 -> const_Year_Built >= 1911
 ind_3479: t167_path3 = 1 -> Overall_Qual_calc >= 3
 ind_3480: t167_path3 = 1 -> const_Overall_Cond >= 7
 ind_3481: t167_path3 = 1 -> const_Yr_Sold <= 2009
 ind_3482: t167_path4 = 1 -> Overall_Qual_calc >= 3
 ind_3483: t167_path4 = 1 -> const_Overall_Cond >= 7
 ind_3484: t167_path4 = 1 -> const_Yr_Sold >= 2009
 ind_3485: t168_path0 = 1 -> const_Condition_1_Artery <= 1
 ind_3486: t168_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3487: t168_path0 = 1 -> const_Overall_Cond <= 7
 ind_3488: t168_path1 = 1 -> const_Condition_1_Artery <= 1
 ind_3489: t168_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3490: t168_path1 = 1 -> const_Overall_Cond >= 7
 ind_3491: t168_path2 = 1 -> const_Condition_1_Artery <= 1
 ind_3492: t168_path2 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_3493: t168_path3 = 1 -> const_Condition_1_Artery >= 1
 ind_3494: t168_path3 = 1 -> x_Garage_Area <= 625
 ind_3495: t168_path4 = 1 -> const_Condition_1_Artery >= 1
 ind_3496: t168_path4 = 1 -> x_Garage_Area >= 625
 ind_3497: t169_path0 = 1 -> const_Overall_Cond <= 7
 ind_3498: t169_path0 = 1 -> const_Year_Built <= 1932
 ind_3499: t169_path0 = 1 -> x_Wood_Deck_SF <= 168
 ind_3500: t169_path1 = 1 -> const_Overall_Cond <= 7
 ind_3501: t169_path1 = 1 -> const_Year_Built <= 1932
 ind_3502: t169_path1 = 1 -> x_Wood_Deck_SF >= 168
 ind_3503: t169_path2 = 1 -> const_Overall_Cond <= 7
 ind_3504: t169_path2 = 1 -> const_Year_Built >= 1932
 ind_3505: t169_path2 = 1 -> const_Year_Remod_Add <= 2007
 ind_3506: t169_path3 = 1 -> const_Overall_Cond <= 7
 ind_3507: t169_path3 = 1 -> const_Year_Built >= 1932
 ind_3508: t169_path3 = 1 -> const_Year_Remod_Add >= 2007
 ind_3509: t169_path4 = 1 -> const_Overall_Cond >= 7
 ind_3510: t169_path4 = 1 -> const_Yr_Sold <= 2008
 ind_3511: t169_path5 = 1 -> const_Overall_Cond >= 7
 ind_3512: t169_path5 = 1 -> const_Yr_Sold >= 2008
 ind_3513: t169_path5 = 1 -> const_Condition_1_Norm <= 1
 ind_3514: t169_path6 = 1 -> const_Overall_Cond >= 7
 ind_3515: t169_path6 = 1 -> const_Yr_Sold >= 2008
 ind_3516: t169_path6 = 1 -> const_Condition_1_Norm >= 1
 ind_3517: t170_path0 = 1 -> x_Kitchen_Qual <= 4
 ind_3518: t170_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3519: t170_path0 = 1 -> const_Lot_Area <= 17227
 ind_3520: t170_path1 = 1 -> x_Kitchen_Qual <= 4
 ind_3521: t170_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3522: t170_path1 = 1 -> const_Lot_Area >= 17227
 ind_3523: t170_path2 = 1 -> x_Kitchen_Qual <= 4
 ind_3524: t170_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3525: t170_path2 = 1 -> const_Lot_Area <= 9900
 ind_3526: t170_path3 = 1 -> x_Kitchen_Qual <= 4
 ind_3527: t170_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3528: t170_path3 = 1 -> const_Lot_Area >= 9900
 ind_3529: t170_path4 = 1 -> x_Kitchen_Qual >= 4
 ind_3530: t171_path0 = 1 -> x_Bsmt_Cond <= 3
 ind_3531: t171_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_3532: t171_path0 = 1 -> x_Screen_Porch <= 210
 ind_3533: t171_path1 = 1 -> x_Bsmt_Cond <= 3
 ind_3534: t171_path1 = 1 -> const_Neighborhood_Edwards <= 1
 ind_3535: t171_path1 = 1 -> x_Screen_Porch >= 210
 ind_3536: t171_path2 = 1 -> x_Bsmt_Cond <= 3
 ind_3537: t171_path2 = 1 -> const_Neighborhood_Edwards >= 1
 ind_3538: t171_path2 = 1 -> const_Lot_Config_Corner <= 1
 ind_3539: t171_path3 = 1 -> x_Bsmt_Cond <= 3
 ind_3540: t171_path3 = 1 -> const_Neighborhood_Edwards >= 1
 ind_3541: t171_path3 = 1 -> const_Lot_Config_Corner >= 1
 ind_3542: t171_path4 = 1 -> x_Bsmt_Cond >= 3
 ind_3543: t172_path0 = 1 -> x_b1_is_ALQ <= 1
 ind_3544: t172_path0 = 1 -> x_Wood_Deck_SF <= 56
 ind_3545: t172_path0 = 1 -> x_Half_Bath <= 1
 ind_3546: t172_path1 = 1 -> x_b1_is_ALQ <= 1
 ind_3547: t172_path1 = 1 -> x_Wood_Deck_SF <= 56
 ind_3548: t172_path1 = 1 -> x_Half_Bath >= 1
 ind_3549: t172_path2 = 1 -> x_b1_is_ALQ <= 1
 ind_3550: t172_path2 = 1 -> x_Wood_Deck_SF >= 56
 ind_3551: t172_path2 = 1 -> x_TotRms_AbvGrd <= 11
 ind_3552: t172_path3 = 1 -> x_b1_is_ALQ <= 1
 ind_3553: t172_path3 = 1 -> x_Wood_Deck_SF >= 56
 ind_3554: t172_path3 = 1 -> x_TotRms_AbvGrd >= 11
 ind_3555: t172_path4 = 1 -> x_b1_is_ALQ >= 1
 ind_3556: t172_path4 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3557: t172_path5 = 1 -> x_b1_is_ALQ >= 1
 ind_3558: t172_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3559: t172_path5 = 1 -> const_Garage_Yr_Blt <= 1977
 ind_3560: t172_path6 = 1 -> x_b1_is_ALQ >= 1
 ind_3561: t172_path6 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3562: t172_path6 = 1 -> const_Garage_Yr_Blt >= 1977
 ind_3563: t173_path0 = 1 -> x_Exter_Cond <= 2
 ind_3564: t173_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_3565: t173_path1 = 1 -> x_Exter_Cond <= 2
 ind_3566: t173_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3567: t173_path1 = 1 -> x_TotRms_AbvGrd <= 5
 ind_3568: t173_path2 = 1 -> x_Exter_Cond <= 2
 ind_3569: t173_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3570: t173_path2 = 1 -> x_TotRms_AbvGrd >= 5
 ind_3571: t173_path3 = 1 -> x_Exter_Cond >= 2
 ind_3572: t173_path3 = 1 -> const_Overall_Cond <= 7
 ind_3573: t173_path3 = 1 -> const_Year_Built <= 1954
 ind_3574: t173_path4 = 1 -> x_Exter_Cond >= 2
 ind_3575: t173_path4 = 1 -> const_Overall_Cond <= 7
 ind_3576: t173_path4 = 1 -> const_Year_Built >= 1954
 ind_3577: t173_path5 = 1 -> x_Exter_Cond >= 2
 ind_3578: t173_path5 = 1 -> const_Overall_Cond >= 7
 ind_3579: t173_path5 = 1 -> const_Garage_Yr_Blt <= 1995
 ind_3580: t173_path6 = 1 -> x_Exter_Cond >= 2
 ind_3581: t173_path6 = 1 -> const_Overall_Cond >= 7
 ind_3582: t173_path6 = 1 -> const_Garage_Yr_Blt >= 1995
 ind_3583: t174_path0 = 1 -> x_Garage_Area <= 1154
 ind_3584: t174_path0 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_3585: t174_path0 = 1 -> const_Yr_Sold <= 2007
 ind_3586: t174_path1 = 1 -> x_Garage_Area <= 1154
 ind_3587: t174_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_3588: t174_path1 = 1 -> const_Yr_Sold >= 2007
 ind_3589: t174_path2 = 1 -> x_Garage_Area <= 1154
 ind_3590: t174_path2 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_3591: t174_path3 = 1 -> x_Garage_Area >= 1154
 ind_3592: t175_path0 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_3593: t175_path0 = 1 -> const_Bsmt_Qual <= 2
 ind_3594: t175_path0 = 1 -> x_Garage_Area <= 487
 ind_3595: t175_path1 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_3596: t175_path1 = 1 -> const_Bsmt_Qual <= 2
 ind_3597: t175_path1 = 1 -> x_Garage_Area >= 487
 ind_3598: t175_path2 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_3599: t175_path2 = 1 -> const_Bsmt_Qual >= 2
 ind_3600: t175_path2 = 1 -> const_Condition_1_Artery <= 1
 ind_3601: t175_path3 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_3602: t175_path3 = 1 -> const_Bsmt_Qual >= 2
 ind_3603: t175_path3 = 1 -> const_Condition_1_Artery >= 1
 ind_3604: t175_path4 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_3605: t176_path0 = 1 -> const_Yr_Sold <= 2007
 ind_3606: t176_path0 = 1 -> central_air_no <= 1
 ind_3607: t176_path0 = 1 -> const_Overall_Cond <= 8
 ind_3608: t176_path1 = 1 -> const_Yr_Sold <= 2007
 ind_3609: t176_path1 = 1 -> central_air_no <= 1
 ind_3610: t176_path1 = 1 -> const_Overall_Cond >= 8
 ind_3611: t176_path2 = 1 -> const_Yr_Sold <= 2007
 ind_3612: t176_path2 = 1 -> central_air_no >= 1
 ind_3613: t176_path2 = 1 -> const_Lot_Area <= 6240
 ind_3614: t176_path3 = 1 -> const_Yr_Sold <= 2007
 ind_3615: t176_path3 = 1 -> central_air_no >= 1
 ind_3616: t176_path3 = 1 -> const_Lot_Area >= 6240
 ind_3617: t176_path4 = 1 -> const_Yr_Sold >= 2007
 ind_3618: t176_path4 = 1 -> x_Garage_Area <= 1154
 ind_3619: t176_path4 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_3620: t176_path5 = 1 -> const_Yr_Sold >= 2007
 ind_3621: t176_path5 = 1 -> x_Garage_Area <= 1154
 ind_3622: t176_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_3623: t176_path6 = 1 -> const_Yr_Sold >= 2007
 ind_3624: t176_path6 = 1 -> x_Garage_Area >= 1154
 ind_3625: t177_path0 = 1 -> x_Heating_QC <= 3
 ind_3626: t177_path0 = 1 -> x_Wood_Deck_SF <= 159
 ind_3627: t177_path0 = 1 -> x_Garage_Area <= 784
 ind_3628: t177_path1 = 1 -> x_Heating_QC <= 3
 ind_3629: t177_path1 = 1 -> x_Wood_Deck_SF <= 159
 ind_3630: t177_path1 = 1 -> x_Garage_Area >= 784
 ind_3631: t177_path2 = 1 -> x_Heating_QC <= 3
 ind_3632: t177_path2 = 1 -> x_Wood_Deck_SF >= 159
 ind_3633: t177_path2 = 1 -> const_Garage_Yr_Blt <= 1978
 ind_3634: t177_path3 = 1 -> x_Heating_QC <= 3
 ind_3635: t177_path3 = 1 -> x_Wood_Deck_SF >= 159
 ind_3636: t177_path3 = 1 -> const_Garage_Yr_Blt >= 1978
 ind_3637: t177_path4 = 1 -> x_Heating_QC >= 3
 ind_3638: t177_path4 = 1 -> x_Open_Porch_SF <= 134
 ind_3639: t177_path4 = 1 -> Overall_Qual_calc <= 7
 ind_3640: t177_path5 = 1 -> x_Heating_QC >= 3
 ind_3641: t177_path5 = 1 -> x_Open_Porch_SF <= 134
 ind_3642: t177_path5 = 1 -> Overall_Qual_calc >= 7
 ind_3643: t177_path6 = 1 -> x_Heating_QC >= 3
 ind_3644: t177_path6 = 1 -> x_Open_Porch_SF >= 134
 ind_3645: t178_path0 = 1 -> const_Yr_Sold <= 2007
 ind_3646: t178_path0 = 1 -> const_Neighborhood_Mitchel <= 1
 ind_3647: t178_path1 = 1 -> const_Yr_Sold <= 2007
 ind_3648: t178_path1 = 1 -> const_Neighborhood_Mitchel >= 1
 ind_3649: t178_path2 = 1 -> const_Yr_Sold >= 2007
 ind_3650: t178_path2 = 1 -> const_MS_Zoning_FV <= 1
 ind_3651: t178_path2 = 1 -> const_Street_Pave <= 1
 ind_3652: t178_path3 = 1 -> const_Yr_Sold >= 2007
 ind_3653: t178_path3 = 1 -> const_MS_Zoning_FV <= 1
 ind_3654: t178_path3 = 1 -> const_Street_Pave >= 1
 ind_3655: t178_path4 = 1 -> const_Yr_Sold >= 2007
 ind_3656: t178_path4 = 1 -> const_MS_Zoning_FV >= 1
 ind_3657: t179_path0 = 1 -> x_Gr_Liv_Area <= 768
 ind_3658: t179_path0 = 1 -> x_1st_Flr_SF <= 717
 ind_3659: t179_path0 = 1 -> x_ex2_is_Wd_Sdng <= 1
 ind_3660: t179_path1 = 1 -> x_Gr_Liv_Area <= 768
 ind_3661: t179_path1 = 1 -> x_1st_Flr_SF <= 717
 ind_3662: t179_path1 = 1 -> x_ex2_is_Wd_Sdng >= 1
 ind_3663: t179_path2 = 1 -> x_Gr_Liv_Area <= 768
 ind_3664: t179_path2 = 1 -> x_1st_Flr_SF >= 717
 ind_3665: t179_path3 = 1 -> x_Gr_Liv_Area >= 768
 ind_3666: t179_path3 = 1 -> const_Yr_Sold <= 2007
 ind_3667: t179_path4 = 1 -> x_Gr_Liv_Area >= 768
 ind_3668: t179_path4 = 1 -> const_Yr_Sold >= 2007
 ind_3669: t179_path4 = 1 -> x_Open_Porch_SF <= 183
 ind_3670: t179_path5 = 1 -> x_Gr_Liv_Area >= 768
 ind_3671: t179_path5 = 1 -> const_Yr_Sold >= 2007
 ind_3672: t179_path5 = 1 -> x_Open_Porch_SF >= 183
 ind_3673: t180_path0 = 1 -> Overall_Qual_calc <= 3
 ind_3674: t180_path1 = 1 -> Overall_Qual_calc >= 3
 ind_3675: t180_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3676: t180_path1 = 1 -> x_Heating_QC <= 3
 ind_3677: t180_path2 = 1 -> Overall_Qual_calc >= 3
 ind_3678: t180_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_3679: t180_path2 = 1 -> x_Heating_QC >= 3
 ind_3680: t180_path3 = 1 -> Overall_Qual_calc >= 3
 ind_3681: t180_path3 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_3682: t180_path3 = 1 -> const_Lot_Config_Corner <= 1
 ind_3683: t180_path4 = 1 -> Overall_Qual_calc >= 3
 ind_3684: t180_path4 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_3685: t180_path4 = 1 -> const_Lot_Config_Corner >= 1
 ind_3686: t181_path0 = 1 -> x_Bsmt_Unf_SF <= 113
 ind_3687: t181_path0 = 1 -> x_Bsmt_Cond <= 3
 ind_3688: t181_path0 = 1 -> x_BsmtFin_SF_1 <= 625
 ind_3689: t181_path1 = 1 -> x_Bsmt_Unf_SF <= 113
 ind_3690: t181_path1 = 1 -> x_Bsmt_Cond <= 3
 ind_3691: t181_path1 = 1 -> x_BsmtFin_SF_1 >= 625
 ind_3692: t181_path2 = 1 -> x_Bsmt_Unf_SF <= 113
 ind_3693: t181_path2 = 1 -> x_Bsmt_Cond >= 3
 ind_3694: t181_path3 = 1 -> x_Bsmt_Unf_SF >= 113
 ind_3695: t181_path3 = 1 -> x_Gr_Liv_Area <= 864
 ind_3696: t181_path3 = 1 -> const_Year_Built <= 1924
 ind_3697: t181_path4 = 1 -> x_Bsmt_Unf_SF >= 113
 ind_3698: t181_path4 = 1 -> x_Gr_Liv_Area <= 864
 ind_3699: t181_path4 = 1 -> const_Year_Built >= 1924
 ind_3700: t181_path5 = 1 -> x_Bsmt_Unf_SF >= 113
 ind_3701: t181_path5 = 1 -> x_Gr_Liv_Area >= 864
 ind_3702: t181_path5 = 1 -> const_2nd_Flr_SF <= 1368
 ind_3703: t181_path6 = 1 -> x_Bsmt_Unf_SF >= 113
 ind_3704: t181_path6 = 1 -> x_Gr_Liv_Area >= 864
 ind_3705: t181_path6 = 1 -> const_2nd_Flr_SF >= 1368
 ind_3706: t182_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3707: t182_path0 = 1 -> const_Lot_Area <= 14680
 ind_3708: t182_path0 = 1 -> x_1st_Flr_SF <= 576
 ind_3709: t182_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3710: t182_path1 = 1 -> const_Lot_Area <= 14680
 ind_3711: t182_path1 = 1 -> x_1st_Flr_SF >= 576
 ind_3712: t182_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3713: t182_path2 = 1 -> const_Lot_Area >= 14680
 ind_3714: t182_path2 = 1 -> x_Open_Porch_SF <= 274
 ind_3715: t182_path3 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3716: t182_path3 = 1 -> const_Lot_Area >= 14680
 ind_3717: t182_path3 = 1 -> x_Open_Porch_SF >= 274
 ind_3718: t182_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3719: t182_path4 = 1 -> Overall_Qual_calc <= 5
 ind_3720: t182_path4 = 1 -> const_Overall_Cond <= 6
 ind_3721: t182_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3722: t182_path5 = 1 -> Overall_Qual_calc <= 5
 ind_3723: t182_path5 = 1 -> const_Overall_Cond >= 6
 ind_3724: t182_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3725: t182_path6 = 1 -> Overall_Qual_calc >= 5
 ind_3726: t182_path6 = 1 -> x_elect_is_SBrkr <= 1
 ind_3727: t182_path7 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3728: t182_path7 = 1 -> Overall_Qual_calc >= 5
 ind_3729: t182_path7 = 1 -> x_elect_is_SBrkr >= 1
 ind_3730: t183_path0 = 1 -> const_Condition_1_Feedr <= 1
 ind_3731: t183_path0 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_3732: t183_path0 = 1 -> const_Overall_Cond <= 7
 ind_3733: t183_path1 = 1 -> const_Condition_1_Feedr <= 1
 ind_3734: t183_path1 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_3735: t183_path1 = 1 -> const_Overall_Cond >= 7
 ind_3736: t183_path2 = 1 -> const_Condition_1_Feedr <= 1
 ind_3737: t183_path2 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_3738: t183_path3 = 1 -> const_Condition_1_Feedr >= 1
 ind_3739: t183_path3 = 1 -> const_Land_Contour_Lvl <= 1
 ind_3740: t183_path3 = 1 -> x_ex1_is_Wd_Sdng <= 1
 ind_3741: t183_path4 = 1 -> const_Condition_1_Feedr >= 1
 ind_3742: t183_path4 = 1 -> const_Land_Contour_Lvl <= 1
 ind_3743: t183_path4 = 1 -> x_ex1_is_Wd_Sdng >= 1
 ind_3744: t183_path5 = 1 -> const_Condition_1_Feedr >= 1
 ind_3745: t183_path5 = 1 -> const_Land_Contour_Lvl >= 1
 ind_3746: t184_path0 = 1 -> x_Gr_Liv_Area <= 1961
 ind_3747: t184_path0 = 1 -> const_Neighborhood_Somerst <= 1
 ind_3748: t184_path0 = 1 -> x_b1_is_ALQ <= 1
 ind_3749: t184_path1 = 1 -> x_Gr_Liv_Area <= 1961
 ind_3750: t184_path1 = 1 -> const_Neighborhood_Somerst <= 1
 ind_3751: t184_path1 = 1 -> x_b1_is_ALQ >= 1
 ind_3752: t184_path2 = 1 -> x_Gr_Liv_Area <= 1961
 ind_3753: t184_path2 = 1 -> const_Neighborhood_Somerst >= 1
 ind_3754: t184_path3 = 1 -> x_Gr_Liv_Area >= 1961
 ind_3755: t184_path3 = 1 -> const_Neighborhood_Edwards <= 1
 ind_3756: t184_path3 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_3757: t184_path4 = 1 -> x_Gr_Liv_Area >= 1961
 ind_3758: t184_path4 = 1 -> const_Neighborhood_Edwards <= 1
 ind_3759: t184_path4 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_3760: t184_path5 = 1 -> x_Gr_Liv_Area >= 1961
 ind_3761: t184_path5 = 1 -> const_Neighborhood_Edwards >= 1
 ind_3762: t185_path0 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_3763: t185_path0 = 1 -> const_Fireplaces <= 2
 ind_3764: t185_path0 = 1 -> x_Wood_Deck_SF <= 209
 ind_3765: t185_path1 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_3766: t185_path1 = 1 -> const_Fireplaces <= 2
 ind_3767: t185_path1 = 1 -> x_Wood_Deck_SF >= 209
 ind_3768: t185_path2 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_3769: t185_path2 = 1 -> const_Fireplaces >= 2
 ind_3770: t185_path2 = 1 -> const_Total_Bsmt_SF <= 1482
 ind_3771: t185_path3 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_3772: t185_path3 = 1 -> const_Fireplaces >= 2
 ind_3773: t185_path3 = 1 -> const_Total_Bsmt_SF >= 1482
 ind_3774: t185_path4 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_3775: t185_path4 = 1 -> const_Lot_Shape_Reg <= 1
 ind_3776: t185_path5 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_3777: t185_path5 = 1 -> const_Lot_Shape_Reg >= 1
 ind_3778: t186_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3779: t186_path0 = 1 -> const_Year_Built <= 2005
 ind_3780: t186_path0 = 1 -> const_House_Style_1Story <= 1
 ind_3781: t186_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3782: t186_path1 = 1 -> const_Year_Built <= 2005
 ind_3783: t186_path1 = 1 -> const_House_Style_1Story >= 1
 ind_3784: t186_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3785: t186_path2 = 1 -> const_Year_Built >= 2005
 ind_3786: t186_path2 = 1 -> const_MS_Zoning_RL <= 1
 ind_3787: t186_path3 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3788: t186_path3 = 1 -> const_Year_Built >= 2005
 ind_3789: t186_path3 = 1 -> const_MS_Zoning_RL >= 1
 ind_3790: t186_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3791: t186_path4 = 1 -> Overall_Qual_calc <= 5
 ind_3792: t186_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3793: t186_path5 = 1 -> Overall_Qual_calc >= 5
 ind_3794: t186_path5 = 1 -> x_Enclosed_Porch <= 32
 ind_3795: t186_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3796: t186_path6 = 1 -> Overall_Qual_calc >= 5
 ind_3797: t186_path6 = 1 -> x_Enclosed_Porch >= 32
 ind_3798: t187_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3799: t187_path0 = 1 -> const_Condition_1_Artery <= 1
 ind_3800: t187_path0 = 1 -> const_Overall_Cond <= 6
 ind_3801: t187_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3802: t187_path1 = 1 -> const_Condition_1_Artery <= 1
 ind_3803: t187_path1 = 1 -> const_Overall_Cond >= 6
 ind_3804: t187_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3805: t187_path2 = 1 -> const_Condition_1_Artery >= 1
 ind_3806: t187_path2 = 1 -> x_Heating_QC <= 3
 ind_3807: t187_path3 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3808: t187_path3 = 1 -> const_Condition_1_Artery >= 1
 ind_3809: t187_path3 = 1 -> x_Heating_QC >= 3
 ind_3810: t187_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3811: t187_path4 = 1 -> const_Overall_Cond <= 4
 ind_3812: t187_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3813: t187_path5 = 1 -> const_Overall_Cond >= 4
 ind_3814: t187_path5 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_3815: t187_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3816: t187_path6 = 1 -> const_Overall_Cond >= 4
 ind_3817: t187_path6 = 1 -> const_Bsmt_Exposure_Av >= 1
 ind_3818: t188_path0 = 1 -> const_Yr_Sold <= 2007
 ind_3819: t188_path0 = 1 -> x_Bsmt_Unf_SF <= 967
 ind_3820: t188_path0 = 1 -> const_Year_Remod_Add <= 1951
 ind_3821: t188_path1 = 1 -> const_Yr_Sold <= 2007
 ind_3822: t188_path1 = 1 -> x_Bsmt_Unf_SF <= 967
 ind_3823: t188_path1 = 1 -> const_Year_Remod_Add >= 1951
 ind_3824: t188_path2 = 1 -> const_Yr_Sold <= 2007
 ind_3825: t188_path2 = 1 -> x_Bsmt_Unf_SF >= 967
 ind_3826: t188_path2 = 1 -> x_ex2_is_VinylSd <= 1
 ind_3827: t188_path3 = 1 -> const_Yr_Sold <= 2007
 ind_3828: t188_path3 = 1 -> x_Bsmt_Unf_SF >= 967
 ind_3829: t188_path3 = 1 -> x_ex2_is_VinylSd >= 1
 ind_3830: t188_path4 = 1 -> const_Yr_Sold >= 2007
 ind_3831: t188_path4 = 1 -> const_Land_Contour_HLS <= 1
 ind_3832: t188_path4 = 1 -> x_b2_is_ALQ <= 1
 ind_3833: t188_path5 = 1 -> const_Yr_Sold >= 2007
 ind_3834: t188_path5 = 1 -> const_Land_Contour_HLS <= 1
 ind_3835: t188_path5 = 1 -> x_b2_is_ALQ >= 1
 ind_3836: t188_path6 = 1 -> const_Yr_Sold >= 2007
 ind_3837: t188_path6 = 1 -> const_Land_Contour_HLS >= 1
 ind_3838: t188_path6 = 1 -> x_BsmtFin_SF_1 <= 81
 ind_3839: t188_path7 = 1 -> const_Yr_Sold >= 2007
 ind_3840: t188_path7 = 1 -> const_Land_Contour_HLS >= 1
 ind_3841: t188_path7 = 1 -> x_BsmtFin_SF_1 >= 81
 ind_3842: t189_path0 = 1 -> x_Open_Porch_SF <= 444
 ind_3843: t189_path0 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_3844: t189_path0 = 1 -> const_Lot_Area <= 3876
 ind_3845: t189_path1 = 1 -> x_Open_Porch_SF <= 444
 ind_3846: t189_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_3847: t189_path1 = 1 -> const_Lot_Area >= 3876
 ind_3848: t189_path2 = 1 -> x_Open_Porch_SF <= 444
 ind_3849: t189_path2 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_3850: t189_path2 = 1 -> x_Enclosed_Porch <= 35
 ind_3851: t189_path3 = 1 -> x_Open_Porch_SF <= 444
 ind_3852: t189_path3 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_3853: t189_path3 = 1 -> x_Enclosed_Porch >= 35
 ind_3854: t189_path4 = 1 -> x_Open_Porch_SF >= 444
 ind_3855: t190_path0 = 1 -> const_Garage_Yr_Blt <= 2005
 ind_3856: t190_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_3857: t190_path0 = 1 -> x_Bsmt_Unf_SF <= 273
 ind_3858: t190_path1 = 1 -> const_Garage_Yr_Blt <= 2005
 ind_3859: t190_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_3860: t190_path1 = 1 -> x_Bsmt_Unf_SF >= 273
 ind_3861: t190_path2 = 1 -> const_Garage_Yr_Blt <= 2005
 ind_3862: t190_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_3863: t190_path2 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_3864: t190_path3 = 1 -> const_Garage_Yr_Blt <= 2005
 ind_3865: t190_path3 = 1 -> x_Gr_Liv_Area >= 747
 ind_3866: t190_path3 = 1 -> const_Neighborhood_ClearCr >= 1
 ind_3867: t190_path4 = 1 -> const_Garage_Yr_Blt >= 2005
 ind_3868: t190_path4 = 1 -> x_Screen_Porch <= 40
 ind_3869: t190_path5 = 1 -> const_Garage_Yr_Blt >= 2005
 ind_3870: t190_path5 = 1 -> x_Screen_Porch >= 40
 ind_3871: t191_path0 = 1 -> const_Lot_Area <= 23257
 ind_3872: t191_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_3873: t191_path0 = 1 -> const_MS_Zoning_RL <= 1
 ind_3874: t191_path1 = 1 -> const_Lot_Area <= 23257
 ind_3875: t191_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_3876: t191_path1 = 1 -> const_MS_Zoning_RL >= 1
 ind_3877: t191_path2 = 1 -> const_Lot_Area <= 23257
 ind_3878: t191_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_3879: t191_path2 = 1 -> x_Gr_Liv_Area <= 3086
 ind_3880: t191_path3 = 1 -> const_Lot_Area <= 23257
 ind_3881: t191_path3 = 1 -> x_Gr_Liv_Area >= 747
 ind_3882: t191_path3 = 1 -> x_Gr_Liv_Area >= 3086
 ind_3883: t191_path4 = 1 -> const_Lot_Area >= 23257
 ind_3884: t191_path4 = 1 -> const_Sale_Condition_Normal <= 1
 ind_3885: t191_path5 = 1 -> const_Lot_Area >= 23257
 ind_3886: t191_path5 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3887: t191_path5 = 1 -> x_Wood_Deck_SF <= 22
 ind_3888: t191_path6 = 1 -> const_Lot_Area >= 23257
 ind_3889: t191_path6 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3890: t191_path6 = 1 -> x_Wood_Deck_SF >= 22
 ind_3891: t192_path0 = 1 -> x_b1_is_ALQ <= 1
 ind_3892: t192_path0 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_3893: t192_path0 = 1 -> x_Wood_Deck_SF <= 56
 ind_3894: t192_path1 = 1 -> x_b1_is_ALQ <= 1
 ind_3895: t192_path1 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_3896: t192_path1 = 1 -> x_Wood_Deck_SF >= 56
 ind_3897: t192_path2 = 1 -> x_b1_is_ALQ <= 1
 ind_3898: t192_path2 = 1 -> const_Lot_Config_CulDSac >= 1
 ind_3899: t192_path3 = 1 -> x_b1_is_ALQ >= 1
 ind_3900: t192_path3 = 1 -> const_Yr_Sold <= 2009
 ind_3901: t192_path4 = 1 -> x_b1_is_ALQ >= 1
 ind_3902: t192_path4 = 1 -> const_Yr_Sold >= 2009
 ind_3903: t193_path0 = 1 -> x_BsmtFin_SF_1 <= 65
 ind_3904: t193_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3905: t193_path0 = 1 -> x_Garage_Area <= 836
 ind_3906: t193_path1 = 1 -> x_BsmtFin_SF_1 <= 65
 ind_3907: t193_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_3908: t193_path1 = 1 -> x_Garage_Area >= 836
 ind_3909: t193_path2 = 1 -> x_BsmtFin_SF_1 <= 65
 ind_3910: t193_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3911: t193_path2 = 1 -> x_Garage_Area <= 451
 ind_3912: t193_path3 = 1 -> x_BsmtFin_SF_1 <= 65
 ind_3913: t193_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_3914: t193_path3 = 1 -> x_Garage_Area >= 451
 ind_3915: t193_path4 = 1 -> x_BsmtFin_SF_1 >= 65
 ind_3916: t193_path4 = 1 -> x_b1_is_LwQ <= 1
 ind_3917: t193_path4 = 1 -> const_Year_Built <= 1919
 ind_3918: t193_path5 = 1 -> x_BsmtFin_SF_1 >= 65
 ind_3919: t193_path5 = 1 -> x_b1_is_LwQ <= 1
 ind_3920: t193_path5 = 1 -> const_Year_Built >= 1919
 ind_3921: t193_path6 = 1 -> x_BsmtFin_SF_1 >= 65
 ind_3922: t193_path6 = 1 -> x_b1_is_LwQ >= 1
 ind_3923: t193_path6 = 1 -> const_Bsmt_Exposure_No <= 1
 ind_3924: t193_path7 = 1 -> x_BsmtFin_SF_1 >= 65
 ind_3925: t193_path7 = 1 -> x_b1_is_LwQ >= 1
 ind_3926: t193_path7 = 1 -> const_Bsmt_Exposure_No >= 1
 ind_3927: t194_path0 = 1 -> x_Gr_Liv_Area <= 3086
 ind_3928: t194_path0 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_3929: t194_path0 = 1 -> const_Overall_Cond <= 4
 ind_3930: t194_path1 = 1 -> x_Gr_Liv_Area <= 3086
 ind_3931: t194_path1 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_3932: t194_path1 = 1 -> const_Overall_Cond >= 4
 ind_3933: t194_path2 = 1 -> x_Gr_Liv_Area <= 3086
 ind_3934: t194_path2 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_3935: t194_path3 = 1 -> x_Gr_Liv_Area >= 3086
 ind_3936: t195_path0 = 1 -> x_Gr_Liv_Area <= 3086
 ind_3937: t195_path0 = 1 -> const_Lot_Frontage <= 28
 ind_3938: t195_path1 = 1 -> x_Gr_Liv_Area <= 3086
 ind_3939: t195_path1 = 1 -> const_Lot_Frontage >= 28
 ind_3940: t195_path1 = 1 -> Overall_Qual_calc <= 7
 ind_3941: t195_path2 = 1 -> x_Gr_Liv_Area <= 3086
 ind_3942: t195_path2 = 1 -> const_Lot_Frontage >= 28
 ind_3943: t195_path2 = 1 -> Overall_Qual_calc >= 7
 ind_3944: t195_path3 = 1 -> x_Gr_Liv_Area >= 3086
 ind_3945: t196_path0 = 1 -> const_Condition_1_Feedr <= 1
 ind_3946: t196_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3947: t196_path0 = 1 -> x_Kitchen_Qual <= 4
 ind_3948: t196_path1 = 1 -> const_Condition_1_Feedr <= 1
 ind_3949: t196_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_3950: t196_path1 = 1 -> x_Kitchen_Qual >= 4
 ind_3951: t196_path2 = 1 -> const_Condition_1_Feedr <= 1
 ind_3952: t196_path2 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3953: t196_path2 = 1 -> const_Neighborhood_No_aplicames <= 1
 ind_3954: t196_path3 = 1 -> const_Condition_1_Feedr <= 1
 ind_3955: t196_path3 = 1 -> x_ex1_is_BrkFace >= 1
 ind_3956: t196_path3 = 1 -> const_Neighborhood_No_aplicames >= 1
 ind_3957: t196_path4 = 1 -> const_Condition_1_Feedr >= 1
 ind_3958: t196_path4 = 1 -> const_Yr_Sold <= 2008
 ind_3959: t196_path5 = 1 -> const_Condition_1_Feedr >= 1
 ind_3960: t196_path5 = 1 -> const_Yr_Sold >= 2008
 ind_3961: t197_path0 = 1 -> x_Open_Porch_SF <= 282
 ind_3962: t197_path0 = 1 -> const_Lot_Area <= 17227
 ind_3963: t197_path0 = 1 -> Overall_Qual_calc <= 9
 ind_3964: t197_path1 = 1 -> x_Open_Porch_SF <= 282
 ind_3965: t197_path1 = 1 -> const_Lot_Area <= 17227
 ind_3966: t197_path1 = 1 -> Overall_Qual_calc >= 9
 ind_3967: t197_path2 = 1 -> x_Open_Porch_SF <= 282
 ind_3968: t197_path2 = 1 -> const_Lot_Area >= 17227
 ind_3969: t197_path2 = 1 -> const_Sale_Type_COD <= 1
 ind_3970: t197_path3 = 1 -> x_Open_Porch_SF <= 282
 ind_3971: t197_path3 = 1 -> const_Lot_Area >= 17227
 ind_3972: t197_path3 = 1 -> const_Sale_Type_COD >= 1
 ind_3973: t197_path4 = 1 -> x_Open_Porch_SF >= 282
 ind_3974: t197_path4 = 1 -> const_Sale_Condition_Normal <= 1
 ind_3975: t197_path5 = 1 -> x_Open_Porch_SF >= 282
 ind_3976: t197_path5 = 1 -> const_Sale_Condition_Normal >= 1
 ind_3977: t198_path0 = 1 -> Overall_Qual_calc <= 3
 ind_3978: t198_path0 = 1 -> const_Garage_Yr_Blt <= 1921
 ind_3979: t198_path1 = 1 -> Overall_Qual_calc <= 3
 ind_3980: t198_path1 = 1 -> const_Garage_Yr_Blt >= 1921
 ind_3981: t198_path2 = 1 -> Overall_Qual_calc >= 3
 ind_3982: t198_path2 = 1 -> const_Sale_Condition_Partial <= 1
 ind_3983: t198_path2 = 1 -> Overall_Qual_calc <= 7
 ind_3984: t198_path3 = 1 -> Overall_Qual_calc >= 3
 ind_3985: t198_path3 = 1 -> const_Sale_Condition_Partial <= 1
 ind_3986: t198_path3 = 1 -> Overall_Qual_calc >= 7
 ind_3987: t198_path4 = 1 -> Overall_Qual_calc >= 3
 ind_3988: t198_path4 = 1 -> const_Sale_Condition_Partial >= 1
 ind_3989: t198_path4 = 1 -> x_Open_Porch_SF <= 244
 ind_3990: t198_path5 = 1 -> Overall_Qual_calc >= 3
 ind_3991: t198_path5 = 1 -> const_Sale_Condition_Partial >= 1
 ind_3992: t198_path5 = 1 -> x_Open_Porch_SF >= 244
 ind_3993: t199_path0 = 1 -> x_Screen_Porch <= 150
 ind_3994: t199_path0 = 1 -> const_Year_Built <= 1911
 ind_3995: t199_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_3996: t199_path1 = 1 -> x_Screen_Porch <= 150
 ind_3997: t199_path1 = 1 -> const_Year_Built <= 1911
 ind_3998: t199_path1 = 1 -> x_ex1_is_AsbShng >= 1
 ind_3999: t199_path2 = 1 -> x_Screen_Porch <= 150
 ind_4000: t199_path2 = 1 -> const_Year_Built >= 1911
 ind_4001: t199_path2 = 1 -> const_2nd_Flr_SF <= 584
 ind_4002: t199_path3 = 1 -> x_Screen_Porch <= 150
 ind_4003: t199_path3 = 1 -> const_Year_Built >= 1911
 ind_4004: t199_path3 = 1 -> const_2nd_Flr_SF >= 584
 ind_4005: t199_path4 = 1 -> x_Screen_Porch >= 150
 ind_4006: t200_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_4007: t200_path0 = 1 -> x_b1_is_Rec <= 1
 ind_4008: t200_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_4009: t200_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_4010: t200_path1 = 1 -> x_b1_is_Rec <= 1
 ind_4011: t200_path1 = 1 -> const_Condition_1_Norm >= 1
 ind_4012: t200_path2 = 1 -> x_ex1_is_BrkFace <= 1
 ind_4013: t200_path2 = 1 -> x_b1_is_Rec >= 1
 ind_4014: t200_path3 = 1 -> x_ex1_is_BrkFace >= 1
 ind_4015: t200_path3 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4016: t200_path4 = 1 -> x_ex1_is_BrkFace >= 1
 ind_4017: t200_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4018: t201_path0 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_4019: t201_path0 = 1 -> const_FunctioNo_aplical_Maj2 <= 1
 ind_4020: t201_path0 = 1 -> x_elect_is_FuseA <= 1
 ind_4021: t201_path1 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_4022: t201_path1 = 1 -> const_FunctioNo_aplical_Maj2 <= 1
 ind_4023: t201_path1 = 1 -> x_elect_is_FuseA >= 1
 ind_4024: t201_path2 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_4025: t201_path2 = 1 -> const_FunctioNo_aplical_Maj2 >= 1
 ind_4026: t201_path3 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_4027: t201_path3 = 1 -> const_Overall_Cond <= 7
 ind_4028: t201_path4 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_4029: t201_path4 = 1 -> const_Overall_Cond >= 7
 ind_4030: t202_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_4031: t202_path0 = 1 -> x_Bsmt_Unf_SF <= 996
 ind_4032: t202_path0 = 1 -> const_Garage_Cars <= 3
 ind_4033: t202_path1 = 1 -> const_Condition_1_Norm <= 1
 ind_4034: t202_path1 = 1 -> x_Bsmt_Unf_SF <= 996
 ind_4035: t202_path1 = 1 -> const_Garage_Cars >= 3
 ind_4036: t202_path2 = 1 -> const_Condition_1_Norm <= 1
 ind_4037: t202_path2 = 1 -> x_Bsmt_Unf_SF >= 996
 ind_4038: t202_path3 = 1 -> const_Condition_1_Norm >= 1
 ind_4039: t202_path3 = 1 -> const_Yr_Sold <= 2009
 ind_4040: t202_path3 = 1 -> const_Overall_Cond <= 7
 ind_4041: t202_path4 = 1 -> const_Condition_1_Norm >= 1
 ind_4042: t202_path4 = 1 -> const_Yr_Sold <= 2009
 ind_4043: t202_path4 = 1 -> const_Overall_Cond >= 7
 ind_4044: t202_path5 = 1 -> const_Condition_1_Norm >= 1
 ind_4045: t202_path5 = 1 -> const_Yr_Sold >= 2009
 ind_4046: t202_path5 = 1 -> central_air_no <= 1
 ind_4047: t202_path6 = 1 -> const_Condition_1_Norm >= 1
 ind_4048: t202_path6 = 1 -> const_Yr_Sold >= 2009
 ind_4049: t202_path6 = 1 -> central_air_no >= 1
 ind_4050: t203_path0 = 1 -> const_Yr_Sold <= 2009
 ind_4051: t203_path0 = 1 -> const_Fireplaces <= 2
 ind_4052: t203_path0 = 1 -> const_Sale_Type_New <= 1
 ind_4053: t203_path1 = 1 -> const_Yr_Sold <= 2009
 ind_4054: t203_path1 = 1 -> const_Fireplaces <= 2
 ind_4055: t203_path1 = 1 -> const_Sale_Type_New >= 1
 ind_4056: t203_path2 = 1 -> const_Yr_Sold <= 2009
 ind_4057: t203_path2 = 1 -> const_Fireplaces >= 2
 ind_4058: t203_path2 = 1 -> const_Overall_Cond <= 6
 ind_4059: t203_path3 = 1 -> const_Yr_Sold <= 2009
 ind_4060: t203_path3 = 1 -> const_Fireplaces >= 2
 ind_4061: t203_path3 = 1 -> const_Overall_Cond >= 6
 ind_4062: t203_path4 = 1 -> const_Yr_Sold >= 2009
 ind_4063: t203_path4 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4064: t203_path4 = 1 -> const_Roof_Style_Gable <= 1
 ind_4065: t203_path5 = 1 -> const_Yr_Sold >= 2009
 ind_4066: t203_path5 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4067: t203_path5 = 1 -> const_Roof_Style_Gable >= 1
 ind_4068: t203_path6 = 1 -> const_Yr_Sold >= 2009
 ind_4069: t203_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4070: t203_path6 = 1 -> const_Roof_Style_Gable <= 1
 ind_4071: t203_path7 = 1 -> const_Yr_Sold >= 2009
 ind_4072: t203_path7 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4073: t203_path7 = 1 -> const_Roof_Style_Gable >= 1
 ind_4074: t204_path0 = 1 -> const_2nd_Flr_SF <= 1296
 ind_4075: t204_path0 = 1 -> x_Exter_Cond <= 2
 ind_4076: t204_path0 = 1 -> x_Gr_Liv_Area <= 1017
 ind_4077: t204_path1 = 1 -> const_2nd_Flr_SF <= 1296
 ind_4078: t204_path1 = 1 -> x_Exter_Cond <= 2
 ind_4079: t204_path1 = 1 -> x_Gr_Liv_Area >= 1017
 ind_4080: t204_path2 = 1 -> const_2nd_Flr_SF <= 1296
 ind_4081: t204_path2 = 1 -> x_Exter_Cond >= 2
 ind_4082: t204_path2 = 1 -> const_Bsmt_Exposure_No <= 1
 ind_4083: t204_path3 = 1 -> const_2nd_Flr_SF <= 1296
 ind_4084: t204_path3 = 1 -> x_Exter_Cond >= 2
 ind_4085: t204_path3 = 1 -> const_Bsmt_Exposure_No >= 1
 ind_4086: t204_path4 = 1 -> const_2nd_Flr_SF >= 1296
 ind_4087: t205_path0 = 1 -> const_Neighborhood_Somerst <= 1
 ind_4088: t205_path0 = 1 -> const_Lot_Area <= 4500
 ind_4089: t205_path1 = 1 -> const_Neighborhood_Somerst <= 1
 ind_4090: t205_path1 = 1 -> const_Lot_Area >= 4500
 ind_4091: t205_path1 = 1 -> x_Screen_Porch <= 143
 ind_4092: t205_path2 = 1 -> const_Neighborhood_Somerst <= 1
 ind_4093: t205_path2 = 1 -> const_Lot_Area >= 4500
 ind_4094: t205_path2 = 1 -> x_Screen_Porch >= 143
 ind_4095: t205_path3 = 1 -> const_Neighborhood_Somerst >= 1
 ind_4096: t206_path0 = 1 -> x_Gr_Liv_Area <= 3086
 ind_4097: t206_path0 = 1 -> const_Year_Remod_Add <= 2008
 ind_4098: t206_path0 = 1 -> const_Overall_Cond <= 3
 ind_4099: t206_path1 = 1 -> x_Gr_Liv_Area <= 3086
 ind_4100: t206_path1 = 1 -> const_Year_Remod_Add <= 2008
 ind_4101: t206_path1 = 1 -> const_Overall_Cond >= 3
 ind_4102: t206_path2 = 1 -> x_Gr_Liv_Area <= 3086
 ind_4103: t206_path2 = 1 -> const_Year_Remod_Add >= 2008
 ind_4104: t206_path3 = 1 -> x_Gr_Liv_Area >= 3086
 ind_4105: t207_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4106: t207_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_4107: t207_path0 = 1 -> const_Land_Contour_Bnk <= 1
 ind_4108: t207_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4109: t207_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_4110: t207_path1 = 1 -> const_Land_Contour_Bnk >= 1
 ind_4111: t207_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4112: t207_path2 = 1 -> x_ex1_is_BrkFace >= 1
 ind_4113: t207_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4114: t207_path3 = 1 -> Overall_Qual_calc <= 4
 ind_4115: t207_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4116: t207_path4 = 1 -> Overall_Qual_calc >= 4
 ind_4117: t207_path4 = 1 -> x_Bsmt_Unf_SF <= 672
 ind_4118: t207_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4119: t207_path5 = 1 -> Overall_Qual_calc >= 4
 ind_4120: t207_path5 = 1 -> x_Bsmt_Unf_SF >= 672
 ind_4121: t208_path0 = 1 -> x_Open_Porch_SF <= 484
 ind_4122: t208_path0 = 1 -> x_Mas_Vnr_Area <= 473
 ind_4123: t208_path0 = 1 -> x_b1_is_ALQ <= 1
 ind_4124: t208_path1 = 1 -> x_Open_Porch_SF <= 484
 ind_4125: t208_path1 = 1 -> x_Mas_Vnr_Area <= 473
 ind_4126: t208_path1 = 1 -> x_b1_is_ALQ >= 1
 ind_4127: t208_path2 = 1 -> x_Open_Porch_SF <= 484
 ind_4128: t208_path2 = 1 -> x_Mas_Vnr_Area >= 473
 ind_4129: t208_path2 = 1 -> const_2nd_Flr_SF <= 1306
 ind_4130: t208_path3 = 1 -> x_Open_Porch_SF <= 484
 ind_4131: t208_path3 = 1 -> x_Mas_Vnr_Area >= 473
 ind_4132: t208_path3 = 1 -> const_2nd_Flr_SF >= 1306
 ind_4133: t208_path4 = 1 -> x_Open_Porch_SF >= 484
 ind_4134: t209_path0 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_4135: t209_path0 = 1 -> const_Lot_Area <= 19522
 ind_4136: t209_path0 = 1 -> const_Condition_1_Artery <= 1
 ind_4137: t209_path1 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_4138: t209_path1 = 1 -> const_Lot_Area <= 19522
 ind_4139: t209_path1 = 1 -> const_Condition_1_Artery >= 1
 ind_4140: t209_path2 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_4141: t209_path2 = 1 -> const_Lot_Area >= 19522
 ind_4142: t209_path2 = 1 -> const_Lot_Frontage <= 80.5
 ind_4143: t209_path3 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_4144: t209_path3 = 1 -> const_Lot_Area >= 19522
 ind_4145: t209_path3 = 1 -> const_Lot_Frontage >= 80.5
 ind_4146: t209_path4 = 1 -> const_Garage_Yr_Blt >= 2008
 ind_4147: t210_path0 = 1 -> const_Condition_1_Artery <= 1
 ind_4148: t210_path0 = 1 -> x_Open_Porch_SF <= 418
 ind_4149: t210_path0 = 1 -> x_BsmtFin_SF_1 <= 1965
 ind_4150: t210_path1 = 1 -> const_Condition_1_Artery <= 1
 ind_4151: t210_path1 = 1 -> x_Open_Porch_SF <= 418
 ind_4152: t210_path1 = 1 -> x_BsmtFin_SF_1 >= 1965
 ind_4153: t210_path2 = 1 -> const_Condition_1_Artery <= 1
 ind_4154: t210_path2 = 1 -> x_Open_Porch_SF >= 418
 ind_4155: t210_path3 = 1 -> const_Condition_1_Artery >= 1
 ind_4156: t210_path3 = 1 -> Overall_Qual_calc <= 4
 ind_4157: t210_path4 = 1 -> const_Condition_1_Artery >= 1
 ind_4158: t210_path4 = 1 -> Overall_Qual_calc >= 4
 ind_4159: t211_path0 = 1 -> const_Yr_Sold <= 2007
 ind_4160: t211_path0 = 1 -> x_Wood_Deck_SF <= 150
 ind_4161: t211_path0 = 1 -> x_Garage_Area <= 379
 ind_4162: t211_path1 = 1 -> const_Yr_Sold <= 2007
 ind_4163: t211_path1 = 1 -> x_Wood_Deck_SF <= 150
 ind_4164: t211_path1 = 1 -> x_Garage_Area >= 379
 ind_4165: t211_path2 = 1 -> const_Yr_Sold <= 2007
 ind_4166: t211_path2 = 1 -> x_Wood_Deck_SF >= 150
 ind_4167: t211_path3 = 1 -> const_Yr_Sold >= 2007
 ind_4168: t211_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4169: t211_path3 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_4170: t211_path4 = 1 -> const_Yr_Sold >= 2007
 ind_4171: t211_path4 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4172: t211_path4 = 1 -> x_Bsmt_Unf_SF >= 476
 ind_4173: t211_path5 = 1 -> const_Yr_Sold >= 2007
 ind_4174: t211_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_4175: t211_path5 = 1 -> x_Garage_Area <= 366
 ind_4176: t211_path6 = 1 -> const_Yr_Sold >= 2007
 ind_4177: t211_path6 = 1 -> x_ex1_is_AsbShng >= 1
 ind_4178: t211_path6 = 1 -> x_Garage_Area >= 366
 ind_4179: t212_path0 = 1 -> x_Exter_Cond <= 2
 ind_4180: t212_path0 = 1 -> const_Lot_Frontage <= 66
 ind_4181: t212_path1 = 1 -> x_Exter_Cond <= 2
 ind_4182: t212_path1 = 1 -> const_Lot_Frontage >= 66
 ind_4183: t212_path1 = 1 -> x_Open_Porch_SF <= 4
 ind_4184: t212_path2 = 1 -> x_Exter_Cond <= 2
 ind_4185: t212_path2 = 1 -> const_Lot_Frontage >= 66
 ind_4186: t212_path2 = 1 -> x_Open_Porch_SF >= 4
 ind_4187: t212_path3 = 1 -> x_Exter_Cond >= 2
 ind_4188: t212_path3 = 1 -> const_Lot_Area <= 3675
 ind_4189: t212_path4 = 1 -> x_Exter_Cond >= 2
 ind_4190: t212_path4 = 1 -> const_Lot_Area >= 3675
 ind_4191: t212_path4 = 1 -> const_Garage_Yr_Blt <= 1941
 ind_4192: t212_path5 = 1 -> x_Exter_Cond >= 2
 ind_4193: t212_path5 = 1 -> const_Lot_Area >= 3675
 ind_4194: t212_path5 = 1 -> const_Garage_Yr_Blt >= 1941
 ind_4195: t213_path0 = 1 -> x_Enclosed_Porch <= 265
 ind_4196: t213_path0 = 1 -> x_Open_Porch_SF <= 191
 ind_4197: t213_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4198: t213_path1 = 1 -> x_Enclosed_Porch <= 265
 ind_4199: t213_path1 = 1 -> x_Open_Porch_SF <= 191
 ind_4200: t213_path1 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4201: t213_path2 = 1 -> x_Enclosed_Porch <= 265
 ind_4202: t213_path2 = 1 -> x_Open_Porch_SF >= 191
 ind_4203: t213_path3 = 1 -> x_Enclosed_Porch >= 265
 ind_4204: t214_path0 = 1 -> const_Condition_1_Artery <= 1
 ind_4205: t214_path0 = 1 -> const_Overall_Cond <= 7
 ind_4206: t214_path0 = 1 -> const_Garage_Yr_Blt <= 2005
 ind_4207: t214_path1 = 1 -> const_Condition_1_Artery <= 1
 ind_4208: t214_path1 = 1 -> const_Overall_Cond <= 7
 ind_4209: t214_path1 = 1 -> const_Garage_Yr_Blt >= 2005
 ind_4210: t214_path2 = 1 -> const_Condition_1_Artery <= 1
 ind_4211: t214_path2 = 1 -> const_Overall_Cond >= 7
 ind_4212: t214_path2 = 1 -> const_Garage_Yr_Blt <= 1908
 ind_4213: t214_path3 = 1 -> const_Condition_1_Artery <= 1
 ind_4214: t214_path3 = 1 -> const_Overall_Cond >= 7
 ind_4215: t214_path3 = 1 -> const_Garage_Yr_Blt >= 1908
 ind_4216: t214_path4 = 1 -> const_Condition_1_Artery >= 1
 ind_4217: t214_path4 = 1 -> const_Fireplaces <= 1
 ind_4218: t214_path4 = 1 -> const_Lot_Area <= 10624
 ind_4219: t214_path5 = 1 -> const_Condition_1_Artery >= 1
 ind_4220: t214_path5 = 1 -> const_Fireplaces <= 1
 ind_4221: t214_path5 = 1 -> const_Lot_Area >= 10624
 ind_4222: t214_path6 = 1 -> const_Condition_1_Artery >= 1
 ind_4223: t214_path6 = 1 -> const_Fireplaces >= 1
 ind_4224: t215_path0 = 1 -> Overall_Qual_calc <= 3
 ind_4225: t215_path0 = 1 -> const_Garage_Yr_Blt <= 1923
 ind_4226: t215_path1 = 1 -> Overall_Qual_calc <= 3
 ind_4227: t215_path1 = 1 -> const_Garage_Yr_Blt >= 1923
 ind_4228: t215_path2 = 1 -> Overall_Qual_calc >= 3
 ind_4229: t215_path2 = 1 -> const_Condition_1_Norm <= 1
 ind_4230: t215_path2 = 1 -> const_Lot_Frontage <= 121
 ind_4231: t215_path3 = 1 -> Overall_Qual_calc >= 3
 ind_4232: t215_path3 = 1 -> const_Condition_1_Norm <= 1
 ind_4233: t215_path3 = 1 -> const_Lot_Frontage >= 121
 ind_4234: t215_path4 = 1 -> Overall_Qual_calc >= 3
 ind_4235: t215_path4 = 1 -> const_Condition_1_Norm >= 1
 ind_4236: t215_path4 = 1 -> const_Overall_Cond <= 6
 ind_4237: t215_path5 = 1 -> Overall_Qual_calc >= 3
 ind_4238: t215_path5 = 1 -> const_Condition_1_Norm >= 1
 ind_4239: t215_path5 = 1 -> const_Overall_Cond >= 6
 ind_4240: t216_path0 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_4241: t216_path0 = 1 -> const_Overall_Cond <= 6
 ind_4242: t216_path0 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_4243: t216_path1 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_4244: t216_path1 = 1 -> const_Overall_Cond <= 6
 ind_4245: t216_path1 = 1 -> const_Garage_Yr_Blt >= 2008
 ind_4246: t216_path2 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_4247: t216_path2 = 1 -> const_Overall_Cond >= 6
 ind_4248: t216_path2 = 1 -> const_Garage_Yr_Blt <= 1908
 ind_4249: t216_path3 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_4250: t216_path3 = 1 -> const_Overall_Cond >= 6
 ind_4251: t216_path3 = 1 -> const_Garage_Yr_Blt >= 1908
 ind_4252: t216_path4 = 1 -> const_Total_Bsmt_SF >= 2492
 ind_4253: t217_path0 = 1 -> x_Garage_Area <= 1154
 ind_4254: t217_path0 = 1 -> const_Year_Built <= 1911
 ind_4255: t217_path0 = 1 -> x_Wood_Deck_SF <= 88
 ind_4256: t217_path1 = 1 -> x_Garage_Area <= 1154
 ind_4257: t217_path1 = 1 -> const_Year_Built <= 1911
 ind_4258: t217_path1 = 1 -> x_Wood_Deck_SF >= 88
 ind_4259: t217_path2 = 1 -> x_Garage_Area <= 1154
 ind_4260: t217_path2 = 1 -> const_Year_Built >= 1911
 ind_4261: t217_path2 = 1 -> const_Year_Remod_Add <= 1965
 ind_4262: t217_path3 = 1 -> x_Garage_Area <= 1154
 ind_4263: t217_path3 = 1 -> const_Year_Built >= 1911
 ind_4264: t217_path3 = 1 -> const_Year_Remod_Add >= 1965
 ind_4265: t217_path4 = 1 -> x_Garage_Area >= 1154
 ind_4266: t218_path0 = 1 -> Overall_Qual_calc <= 10
 ind_4267: t218_path0 = 1 -> x_Open_Porch_SF <= 194
 ind_4268: t218_path0 = 1 -> const_Sale_Type_New <= 1
 ind_4269: t218_path1 = 1 -> Overall_Qual_calc <= 10
 ind_4270: t218_path1 = 1 -> x_Open_Porch_SF <= 194
 ind_4271: t218_path1 = 1 -> const_Sale_Type_New >= 1
 ind_4272: t218_path2 = 1 -> Overall_Qual_calc <= 10
 ind_4273: t218_path2 = 1 -> x_Open_Porch_SF >= 194
 ind_4274: t218_path3 = 1 -> Overall_Qual_calc >= 10
 ind_4275: t219_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4276: t219_path0 = 1 -> const_Land_Slope_Gtl <= 1
 ind_4277: t219_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4278: t219_path1 = 1 -> const_Land_Slope_Gtl >= 1
 ind_4279: t219_path1 = 1 -> const_Total_Bsmt_SF <= 964
 ind_4280: t219_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4281: t219_path2 = 1 -> const_Land_Slope_Gtl >= 1
 ind_4282: t219_path2 = 1 -> const_Total_Bsmt_SF >= 964
 ind_4283: t219_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4284: t219_path3 = 1 -> const_Lot_Area <= 15312
 ind_4285: t219_path3 = 1 -> const_MS_SubClass_30 <= 1
 ind_4286: t219_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4287: t219_path4 = 1 -> const_Lot_Area <= 15312
 ind_4288: t219_path4 = 1 -> const_MS_SubClass_30 >= 1
 ind_4289: t219_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4290: t219_path5 = 1 -> const_Lot_Area >= 15312
 ind_4291: t219_path5 = 1 -> x_BsmtFin_SF_1 <= 1319
 ind_4292: t219_path6 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4293: t219_path6 = 1 -> const_Lot_Area >= 15312
 ind_4294: t219_path6 = 1 -> x_BsmtFin_SF_1 >= 1319
 ind_4295: t220_path0 = 1 -> const_Yr_Sold <= 2009
 ind_4296: t220_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_4297: t220_path0 = 1 -> const_Garage_Type_CarPort <= 1
 ind_4298: t220_path1 = 1 -> const_Yr_Sold <= 2009
 ind_4299: t220_path1 = 1 -> const_Mo_Sold_2 <= 1
 ind_4300: t220_path1 = 1 -> const_Garage_Type_CarPort >= 1
 ind_4301: t220_path2 = 1 -> const_Yr_Sold <= 2009
 ind_4302: t220_path2 = 1 -> const_Mo_Sold_2 >= 1
 ind_4303: t220_path2 = 1 -> const_Total_Bsmt_SF <= 1489
 ind_4304: t220_path3 = 1 -> const_Yr_Sold <= 2009
 ind_4305: t220_path3 = 1 -> const_Mo_Sold_2 >= 1
 ind_4306: t220_path3 = 1 -> const_Total_Bsmt_SF >= 1489
 ind_4307: t220_path4 = 1 -> const_Yr_Sold >= 2009
 ind_4308: t220_path4 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4309: t220_path4 = 1 -> const_Roof_Style_Hip <= 1
 ind_4310: t220_path5 = 1 -> const_Yr_Sold >= 2009
 ind_4311: t220_path5 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4312: t220_path5 = 1 -> const_Roof_Style_Hip >= 1
 ind_4313: t220_path6 = 1 -> const_Yr_Sold >= 2009
 ind_4314: t220_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4315: t221_path0 = 1 -> const_Yr_Sold <= 2007
 ind_4316: t221_path0 = 1 -> central_air_no <= 1
 ind_4317: t221_path0 = 1 -> const_Overall_Cond <= 6
 ind_4318: t221_path1 = 1 -> const_Yr_Sold <= 2007
 ind_4319: t221_path1 = 1 -> central_air_no <= 1
 ind_4320: t221_path1 = 1 -> const_Overall_Cond >= 6
 ind_4321: t221_path2 = 1 -> const_Yr_Sold <= 2007
 ind_4322: t221_path2 = 1 -> central_air_no >= 1
 ind_4323: t221_path3 = 1 -> const_Yr_Sold >= 2007
 ind_4324: t221_path3 = 1 -> x_roof_matl_is_CompShg <= 1
 ind_4325: t221_path4 = 1 -> const_Yr_Sold >= 2007
 ind_4326: t221_path4 = 1 -> x_roof_matl_is_CompShg >= 1
 ind_4327: t221_path4 = 1 -> x_1st_Flr_SF <= 2036
 ind_4328: t221_path5 = 1 -> const_Yr_Sold >= 2007
 ind_4329: t221_path5 = 1 -> x_roof_matl_is_CompShg >= 1
 ind_4330: t221_path5 = 1 -> x_1st_Flr_SF >= 2036
 ind_4331: t222_path0 = 1 -> const_2nd_Flr_SF <= 1629
 ind_4332: t222_path0 = 1 -> const_Yr_Sold <= 2007
 ind_4333: t222_path0 = 1 -> const_Lot_Frontage <= 90
 ind_4334: t222_path1 = 1 -> const_2nd_Flr_SF <= 1629
 ind_4335: t222_path1 = 1 -> const_Yr_Sold <= 2007
 ind_4336: t222_path1 = 1 -> const_Lot_Frontage >= 90
 ind_4337: t222_path2 = 1 -> const_2nd_Flr_SF <= 1629
 ind_4338: t222_path2 = 1 -> const_Yr_Sold >= 2007
 ind_4339: t222_path2 = 1 -> const_Year_Remod_Add <= 2007
 ind_4340: t222_path3 = 1 -> const_2nd_Flr_SF <= 1629
 ind_4341: t222_path3 = 1 -> const_Yr_Sold >= 2007
 ind_4342: t222_path3 = 1 -> const_Year_Remod_Add >= 2007
 ind_4343: t222_path4 = 1 -> const_2nd_Flr_SF >= 1629
 ind_4344: t223_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_4345: t223_path0 = 1 -> const_Lot_Frontage <= 121
 ind_4346: t223_path0 = 1 -> const_Lot_Frontage <= 80
 ind_4347: t223_path1 = 1 -> const_Condition_1_Norm <= 1
 ind_4348: t223_path1 = 1 -> const_Lot_Frontage <= 121
 ind_4349: t223_path1 = 1 -> const_Lot_Frontage >= 80
 ind_4350: t223_path2 = 1 -> const_Condition_1_Norm <= 1
 ind_4351: t223_path2 = 1 -> const_Lot_Frontage >= 121
 ind_4352: t223_path3 = 1 -> const_Condition_1_Norm >= 1
 ind_4353: t223_path3 = 1 -> const_Fireplaces <= 2
 ind_4354: t223_path3 = 1 -> const_Yr_Sold <= 2007
 ind_4355: t223_path4 = 1 -> const_Condition_1_Norm >= 1
 ind_4356: t223_path4 = 1 -> const_Fireplaces <= 2
 ind_4357: t223_path4 = 1 -> const_Yr_Sold >= 2007
 ind_4358: t223_path5 = 1 -> const_Condition_1_Norm >= 1
 ind_4359: t223_path5 = 1 -> const_Fireplaces >= 2
 ind_4360: t223_path5 = 1 -> const_Lot_Area <= 17871
 ind_4361: t223_path6 = 1 -> const_Condition_1_Norm >= 1
 ind_4362: t223_path6 = 1 -> const_Fireplaces >= 2
 ind_4363: t223_path6 = 1 -> const_Lot_Area >= 17871
 ind_4364: t224_path0 = 1 -> x_Garage_Area <= 1154
 ind_4365: t224_path0 = 1 -> const_Street_Pave <= 1
 ind_4366: t224_path1 = 1 -> x_Garage_Area <= 1154
 ind_4367: t224_path1 = 1 -> const_Street_Pave >= 1
 ind_4368: t224_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_4369: t224_path2 = 1 -> x_Garage_Area <= 1154
 ind_4370: t224_path2 = 1 -> const_Street_Pave >= 1
 ind_4371: t224_path2 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_4372: t224_path3 = 1 -> x_Garage_Area >= 1154
 ind_4373: t225_path0 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_4374: t225_path0 = 1 -> x_Wood_Deck_SF <= 212
 ind_4375: t225_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_4376: t225_path1 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_4377: t225_path1 = 1 -> x_Wood_Deck_SF <= 212
 ind_4378: t225_path1 = 1 -> const_Neighborhood_Edwards >= 1
 ind_4379: t225_path2 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_4380: t225_path2 = 1 -> x_Wood_Deck_SF >= 212
 ind_4381: t225_path2 = 1 -> const_Lot_Frontage <= 115
 ind_4382: t225_path3 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_4383: t225_path3 = 1 -> x_Wood_Deck_SF >= 212
 ind_4384: t225_path3 = 1 -> const_Lot_Frontage >= 115
 ind_4385: t225_path4 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_4386: t226_path0 = 1 -> x_Gr_Liv_Area <= 988
 ind_4387: t226_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4388: t226_path0 = 1 -> central_air_no <= 1
 ind_4389: t226_path1 = 1 -> x_Gr_Liv_Area <= 988
 ind_4390: t226_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4391: t226_path1 = 1 -> central_air_no >= 1
 ind_4392: t226_path2 = 1 -> x_Gr_Liv_Area <= 988
 ind_4393: t226_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4394: t226_path2 = 1 -> const_Yr_Sold <= 2008
 ind_4395: t226_path3 = 1 -> x_Gr_Liv_Area <= 988
 ind_4396: t226_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4397: t226_path3 = 1 -> const_Yr_Sold >= 2008
 ind_4398: t226_path4 = 1 -> x_Gr_Liv_Area >= 988
 ind_4399: t226_path4 = 1 -> const_Overall_Cond <= 6
 ind_4400: t226_path4 = 1 -> x_Bsmt_Unf_SF <= 466
 ind_4401: t226_path5 = 1 -> x_Gr_Liv_Area >= 988
 ind_4402: t226_path5 = 1 -> const_Overall_Cond <= 6
 ind_4403: t226_path5 = 1 -> x_Bsmt_Unf_SF >= 466
 ind_4404: t226_path6 = 1 -> x_Gr_Liv_Area >= 988
 ind_4405: t226_path6 = 1 -> const_Overall_Cond >= 6
 ind_4406: t226_path6 = 1 -> const_Year_Built <= 1892
 ind_4407: t226_path7 = 1 -> x_Gr_Liv_Area >= 988
 ind_4408: t226_path7 = 1 -> const_Overall_Cond >= 6
 ind_4409: t226_path7 = 1 -> const_Year_Built >= 1892
 ind_4410: t227_path0 = 1 -> x_Screen_Porch <= 143
 ind_4411: t227_path0 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_4412: t227_path0 = 1 -> const_Lot_Frontage <= 130
 ind_4413: t227_path1 = 1 -> x_Screen_Porch <= 143
 ind_4414: t227_path1 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_4415: t227_path1 = 1 -> const_Lot_Frontage >= 130
 ind_4416: t227_path2 = 1 -> x_Screen_Porch <= 143
 ind_4417: t227_path2 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_4418: t227_path2 = 1 -> const_Lot_Area <= 6600
 ind_4419: t227_path3 = 1 -> x_Screen_Porch <= 143
 ind_4420: t227_path3 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_4421: t227_path3 = 1 -> const_Lot_Area >= 6600
 ind_4422: t227_path4 = 1 -> x_Screen_Porch >= 143
 ind_4423: t228_path0 = 1 -> const_Condition_1_Feedr <= 1
 ind_4424: t228_path0 = 1 -> x_Full_Bath <= 3
 ind_4425: t228_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4426: t228_path1 = 1 -> const_Condition_1_Feedr <= 1
 ind_4427: t228_path1 = 1 -> x_Full_Bath <= 3
 ind_4428: t228_path1 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4429: t228_path2 = 1 -> const_Condition_1_Feedr <= 1
 ind_4430: t228_path2 = 1 -> x_Full_Bath >= 3
 ind_4431: t228_path3 = 1 -> const_Condition_1_Feedr >= 1
 ind_4432: t228_path3 = 1 -> const_Land_Contour_Bnk <= 1
 ind_4433: t228_path4 = 1 -> const_Condition_1_Feedr >= 1
 ind_4434: t228_path4 = 1 -> const_Land_Contour_Bnk >= 1
 ind_4435: t229_path0 = 1 -> const_Fireplaces <= 3
 ind_4436: t229_path0 = 1 -> x_Gr_Liv_Area <= 3086
 ind_4437: t229_path0 = 1 -> const_Lot_Area <= 4835
 ind_4438: t229_path1 = 1 -> const_Fireplaces <= 3
 ind_4439: t229_path1 = 1 -> x_Gr_Liv_Area <= 3086
 ind_4440: t229_path1 = 1 -> const_Lot_Area >= 4835
 ind_4441: t229_path2 = 1 -> const_Fireplaces <= 3
 ind_4442: t229_path2 = 1 -> x_Gr_Liv_Area >= 3086
 ind_4443: t229_path2 = 1 -> x_Open_Porch_SF <= 35
 ind_4444: t229_path3 = 1 -> const_Fireplaces <= 3
 ind_4445: t229_path3 = 1 -> x_Gr_Liv_Area >= 3086
 ind_4446: t229_path3 = 1 -> x_Open_Porch_SF >= 35
 ind_4447: t229_path4 = 1 -> const_Fireplaces >= 3
 ind_4448: t230_path0 = 1 -> const_Overall_Cond <= 6
 ind_4449: t230_path0 = 1 -> x_Garage_Area <= 253
 ind_4450: t230_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_4451: t230_path1 = 1 -> const_Overall_Cond <= 6
 ind_4452: t230_path1 = 1 -> x_Garage_Area <= 253
 ind_4453: t230_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_4454: t230_path2 = 1 -> const_Overall_Cond <= 6
 ind_4455: t230_path2 = 1 -> x_Garage_Area >= 253
 ind_4456: t230_path2 = 1 -> Overall_Qual_calc <= 3
 ind_4457: t230_path3 = 1 -> const_Overall_Cond <= 6
 ind_4458: t230_path3 = 1 -> x_Garage_Area >= 253
 ind_4459: t230_path3 = 1 -> Overall_Qual_calc >= 3
 ind_4460: t230_path4 = 1 -> const_Overall_Cond >= 6
 ind_4461: t230_path4 = 1 -> const_Land_Contour_HLS <= 1
 ind_4462: t230_path4 = 1 -> x_Exter_Cond <= 3
 ind_4463: t230_path5 = 1 -> const_Overall_Cond >= 6
 ind_4464: t230_path5 = 1 -> const_Land_Contour_HLS <= 1
 ind_4465: t230_path5 = 1 -> x_Exter_Cond >= 3
 ind_4466: t230_path6 = 1 -> const_Overall_Cond >= 6
 ind_4467: t230_path6 = 1 -> const_Land_Contour_HLS >= 1
 ind_4468: t231_path0 = 1 -> const_Lot_Area <= 19522
 ind_4469: t231_path0 = 1 -> const_Overall_Cond <= 6
 ind_4470: t231_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4471: t231_path1 = 1 -> const_Lot_Area <= 19522
 ind_4472: t231_path1 = 1 -> const_Overall_Cond <= 6
 ind_4473: t231_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4474: t231_path2 = 1 -> const_Lot_Area <= 19522
 ind_4475: t231_path2 = 1 -> const_Overall_Cond >= 6
 ind_4476: t231_path2 = 1 -> const_Lot_Area <= 7400
 ind_4477: t231_path3 = 1 -> const_Lot_Area <= 19522
 ind_4478: t231_path3 = 1 -> const_Overall_Cond >= 6
 ind_4479: t231_path3 = 1 -> const_Lot_Area >= 7400
 ind_4480: t231_path4 = 1 -> const_Lot_Area >= 19522
 ind_4481: t231_path4 = 1 -> x_b1_is_GLQ <= 1
 ind_4482: t231_path4 = 1 -> x_mvt_is_No_aplica <= 1
 ind_4483: t231_path5 = 1 -> const_Lot_Area >= 19522
 ind_4484: t231_path5 = 1 -> x_b1_is_GLQ <= 1
 ind_4485: t231_path5 = 1 -> x_mvt_is_No_aplica >= 1
 ind_4486: t231_path6 = 1 -> const_Lot_Area >= 19522
 ind_4487: t231_path6 = 1 -> x_b1_is_GLQ >= 1
 ind_4488: t231_path6 = 1 -> x_Open_Porch_SF <= 72
 ind_4489: t231_path7 = 1 -> const_Lot_Area >= 19522
 ind_4490: t231_path7 = 1 -> x_b1_is_GLQ >= 1
 ind_4491: t231_path7 = 1 -> x_Open_Porch_SF >= 72
 ind_4492: t232_path0 = 1 -> x_Open_Porch_SF <= 291
 ind_4493: t232_path0 = 1 -> x_1st_Flr_SF <= 2200
 ind_4494: t232_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_4495: t232_path1 = 1 -> x_Open_Porch_SF <= 291
 ind_4496: t232_path1 = 1 -> x_1st_Flr_SF <= 2200
 ind_4497: t232_path1 = 1 -> x_ex1_is_BrkFace >= 1
 ind_4498: t232_path2 = 1 -> x_Open_Porch_SF <= 291
 ind_4499: t232_path2 = 1 -> x_1st_Flr_SF >= 2200
 ind_4500: t232_path3 = 1 -> x_Open_Porch_SF >= 291
 ind_4501: t232_path3 = 1 -> x_BsmtFin_SF_1 <= 397
 ind_4502: t232_path4 = 1 -> x_Open_Porch_SF >= 291
 ind_4503: t232_path4 = 1 -> x_BsmtFin_SF_1 >= 397
 ind_4504: t233_path0 = 1 -> x_Open_Porch_SF <= 150
 ind_4505: t233_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_4506: t233_path0 = 1 -> const_Overall_Cond <= 8
 ind_4507: t233_path1 = 1 -> x_Open_Porch_SF <= 150
 ind_4508: t233_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_4509: t233_path1 = 1 -> const_Overall_Cond >= 8
 ind_4510: t233_path2 = 1 -> x_Open_Porch_SF <= 150
 ind_4511: t233_path2 = 1 -> x_ex1_is_BrkFace >= 1
 ind_4512: t233_path2 = 1 -> const_Condition_1_Feedr <= 1
 ind_4513: t233_path3 = 1 -> x_Open_Porch_SF <= 150
 ind_4514: t233_path3 = 1 -> x_ex1_is_BrkFace >= 1
 ind_4515: t233_path3 = 1 -> const_Condition_1_Feedr >= 1
 ind_4516: t233_path4 = 1 -> x_Open_Porch_SF >= 150
 ind_4517: t233_path4 = 1 -> x_BsmtFin_SF_2 <= 491
 ind_4518: t233_path4 = 1 -> x_Gr_Liv_Area <= 2393
 ind_4519: t233_path5 = 1 -> x_Open_Porch_SF >= 150
 ind_4520: t233_path5 = 1 -> x_BsmtFin_SF_2 <= 491
 ind_4521: t233_path5 = 1 -> x_Gr_Liv_Area >= 2393
 ind_4522: t233_path6 = 1 -> x_Open_Porch_SF >= 150
 ind_4523: t233_path6 = 1 -> x_BsmtFin_SF_2 >= 491
 ind_4524: t234_path0 = 1 -> Overall_Qual_calc <= 4
 ind_4525: t234_path0 = 1 -> const_Total_Bsmt_SF <= 716
 ind_4526: t234_path1 = 1 -> Overall_Qual_calc <= 4
 ind_4527: t234_path1 = 1 -> const_Total_Bsmt_SF >= 716
 ind_4528: t234_path2 = 1 -> Overall_Qual_calc >= 4
 ind_4529: t234_path2 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_4530: t234_path2 = 1 -> const_Neighborhood_SawyerW <= 1
 ind_4531: t234_path3 = 1 -> Overall_Qual_calc >= 4
 ind_4532: t234_path3 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_4533: t234_path3 = 1 -> const_Neighborhood_SawyerW >= 1
 ind_4534: t234_path4 = 1 -> Overall_Qual_calc >= 4
 ind_4535: t234_path4 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_4536: t234_path4 = 1 -> x_Wood_Deck_SF <= 172
 ind_4537: t234_path5 = 1 -> Overall_Qual_calc >= 4
 ind_4538: t234_path5 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_4539: t234_path5 = 1 -> x_Wood_Deck_SF >= 172
 ind_4540: t235_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_4541: t235_path0 = 1 -> x_Gr_Liv_Area <= 768
 ind_4542: t235_path0 = 1 -> const_MS_Zoning_RL <= 1
 ind_4543: t235_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_4544: t235_path1 = 1 -> x_Gr_Liv_Area <= 768
 ind_4545: t235_path1 = 1 -> const_MS_Zoning_RL >= 1
 ind_4546: t235_path2 = 1 -> x_1st_Flr_SF <= 2515
 ind_4547: t235_path2 = 1 -> x_Gr_Liv_Area >= 768
 ind_4548: t235_path2 = 1 -> x_Mas_Vnr_Area <= 673
 ind_4549: t235_path3 = 1 -> x_1st_Flr_SF <= 2515
 ind_4550: t235_path3 = 1 -> x_Gr_Liv_Area >= 768
 ind_4551: t235_path3 = 1 -> x_Mas_Vnr_Area >= 673
 ind_4552: t235_path4 = 1 -> x_1st_Flr_SF >= 2515
 ind_4553: t236_path0 = 1 -> const_Overall_Cond <= 7
 ind_4554: t236_path0 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_4555: t236_path0 = 1 -> const_MS_Zoning_FV <= 1
 ind_4556: t236_path1 = 1 -> const_Overall_Cond <= 7
 ind_4557: t236_path1 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_4558: t236_path1 = 1 -> const_MS_Zoning_FV >= 1
 ind_4559: t236_path2 = 1 -> const_Overall_Cond <= 7
 ind_4560: t236_path2 = 1 -> const_Lot_Config_CulDSac >= 1
 ind_4561: t236_path3 = 1 -> const_Overall_Cond >= 7
 ind_4562: t236_path3 = 1 -> const_Garage_Yr_Blt <= 1978
 ind_4563: t236_path3 = 1 -> const_Lot_Area <= 9738
 ind_4564: t236_path4 = 1 -> const_Overall_Cond >= 7
 ind_4565: t236_path4 = 1 -> const_Garage_Yr_Blt <= 1978
 ind_4566: t236_path4 = 1 -> const_Lot_Area >= 9738
 ind_4567: t236_path5 = 1 -> const_Overall_Cond >= 7
 ind_4568: t236_path5 = 1 -> const_Garage_Yr_Blt >= 1978
 ind_4569: t237_path0 = 1 -> x_Screen_Porch <= 104
 ind_4570: t237_path0 = 1 -> Overall_Qual_calc <= 10
 ind_4571: t237_path0 = 1 -> Overall_Qual_calc <= 4
 ind_4572: t237_path1 = 1 -> x_Screen_Porch <= 104
 ind_4573: t237_path1 = 1 -> Overall_Qual_calc <= 10
 ind_4574: t237_path1 = 1 -> Overall_Qual_calc >= 4
 ind_4575: t237_path2 = 1 -> x_Screen_Porch <= 104
 ind_4576: t237_path2 = 1 -> Overall_Qual_calc >= 10
 ind_4577: t237_path3 = 1 -> x_Screen_Porch >= 104
 ind_4578: t238_path0 = 1 -> const_Lot_Frontage <= 120
 ind_4579: t238_path0 = 1 -> const_Lot_Area <= 2760
 ind_4580: t238_path1 = 1 -> const_Lot_Frontage <= 120
 ind_4581: t238_path1 = 1 -> const_Lot_Area >= 2760
 ind_4582: t238_path1 = 1 -> x_elect_is_FuseA <= 1
 ind_4583: t238_path2 = 1 -> const_Lot_Frontage <= 120
 ind_4584: t238_path2 = 1 -> const_Lot_Area >= 2760
 ind_4585: t238_path2 = 1 -> x_elect_is_FuseA >= 1
 ind_4586: t238_path3 = 1 -> const_Lot_Frontage >= 120
 ind_4587: t238_path3 = 1 -> x_Exter_Qual <= 3
 ind_4588: t238_path4 = 1 -> const_Lot_Frontage >= 120
 ind_4589: t238_path4 = 1 -> x_Exter_Qual >= 3
 ind_4590: t238_path4 = 1 -> x_Wood_Deck_SF <= 150
 ind_4591: t238_path5 = 1 -> const_Lot_Frontage >= 120
 ind_4592: t238_path5 = 1 -> x_Exter_Qual >= 3
 ind_4593: t238_path5 = 1 -> x_Wood_Deck_SF >= 150
 ind_4594: t239_path0 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_4595: t239_path0 = 1 -> const_Overall_Cond <= 6
 ind_4596: t239_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4597: t239_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_4598: t239_path1 = 1 -> const_Overall_Cond <= 6
 ind_4599: t239_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4600: t239_path2 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_4601: t239_path2 = 1 -> const_Overall_Cond >= 6
 ind_4602: t239_path2 = 1 -> const_Condition_1_Norm <= 1
 ind_4603: t239_path3 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_4604: t239_path3 = 1 -> const_Overall_Cond >= 6
 ind_4605: t239_path3 = 1 -> const_Condition_1_Norm >= 1
 ind_4606: t239_path4 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_4607: t239_path4 = 1 -> x_Garage_Area <= 630
 ind_4608: t239_path5 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_4609: t239_path5 = 1 -> x_Garage_Area >= 630
 ind_4610: t239_path5 = 1 -> const_Lot_Frontage <= 120
 ind_4611: t239_path6 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_4612: t239_path6 = 1 -> x_Garage_Area >= 630
 ind_4613: t239_path6 = 1 -> const_Lot_Frontage >= 120
 ind_4614: t240_path0 = 1 -> x_Full_Bath <= 3
 ind_4615: t240_path0 = 1 -> const_Lot_Area <= 3675
 ind_4616: t240_path1 = 1 -> x_Full_Bath <= 3
 ind_4617: t240_path1 = 1 -> const_Lot_Area >= 3675
 ind_4618: t240_path1 = 1 -> x_1st_Flr_SF <= 831
 ind_4619: t240_path2 = 1 -> x_Full_Bath <= 3
 ind_4620: t240_path2 = 1 -> const_Lot_Area >= 3675
 ind_4621: t240_path2 = 1 -> x_1st_Flr_SF >= 831
 ind_4622: t240_path3 = 1 -> x_Full_Bath >= 3
 ind_4623: t240_path3 = 1 -> x_mvt_is_No_aplica <= 1
 ind_4624: t240_path3 = 1 -> x_Mas_Vnr_Area <= 188
 ind_4625: t240_path4 = 1 -> x_Full_Bath >= 3
 ind_4626: t240_path4 = 1 -> x_mvt_is_No_aplica <= 1
 ind_4627: t240_path4 = 1 -> x_Mas_Vnr_Area >= 188
 ind_4628: t240_path5 = 1 -> x_Full_Bath >= 3
 ind_4629: t240_path5 = 1 -> x_mvt_is_No_aplica >= 1
 ind_4630: t241_path0 = 1 -> Overall_Qual_calc <= 3
 ind_4631: t241_path1 = 1 -> Overall_Qual_calc >= 3
 ind_4632: t241_path1 = 1 -> const_Condition_1_Norm <= 1
 ind_4633: t241_path1 = 1 -> const_Mo_Sold_1 <= 1
 ind_4634: t241_path2 = 1 -> Overall_Qual_calc >= 3
 ind_4635: t241_path2 = 1 -> const_Condition_1_Norm <= 1
 ind_4636: t241_path2 = 1 -> const_Mo_Sold_1 >= 1
 ind_4637: t241_path3 = 1 -> Overall_Qual_calc >= 3
 ind_4638: t241_path3 = 1 -> const_Condition_1_Norm >= 1
 ind_4639: t241_path3 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_4640: t241_path4 = 1 -> Overall_Qual_calc >= 3
 ind_4641: t241_path4 = 1 -> const_Condition_1_Norm >= 1
 ind_4642: t241_path4 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_4643: t242_path0 = 1 -> x_Open_Porch_SF <= 195
 ind_4644: t242_path0 = 1 -> const_Yr_Sold <= 2007
 ind_4645: t242_path0 = 1 -> const_Year_Built <= 1950
 ind_4646: t242_path1 = 1 -> x_Open_Porch_SF <= 195
 ind_4647: t242_path1 = 1 -> const_Yr_Sold <= 2007
 ind_4648: t242_path1 = 1 -> const_Year_Built >= 1950
 ind_4649: t242_path2 = 1 -> x_Open_Porch_SF <= 195
 ind_4650: t242_path2 = 1 -> const_Yr_Sold >= 2007
 ind_4651: t242_path2 = 1 -> const_Lot_Area <= 4500
 ind_4652: t242_path3 = 1 -> x_Open_Porch_SF <= 195
 ind_4653: t242_path3 = 1 -> const_Yr_Sold >= 2007
 ind_4654: t242_path3 = 1 -> const_Lot_Area >= 4500
 ind_4655: t242_path4 = 1 -> x_Open_Porch_SF >= 195
 ind_4656: t243_path0 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_4657: t243_path0 = 1 -> const_Lot_Area <= 19522
 ind_4658: t243_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_4659: t243_path1 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_4660: t243_path1 = 1 -> const_Lot_Area <= 19522
 ind_4661: t243_path1 = 1 -> const_Mo_Sold_2 >= 1
 ind_4662: t243_path2 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_4663: t243_path2 = 1 -> const_Lot_Area >= 19522
 ind_4664: t243_path2 = 1 -> x_mvt_is_BrkFace <= 1
 ind_4665: t243_path3 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_4666: t243_path3 = 1 -> const_Lot_Area >= 19522
 ind_4667: t243_path3 = 1 -> x_mvt_is_BrkFace >= 1
 ind_4668: t243_path4 = 1 -> const_Total_Bsmt_SF >= 2492
 ind_4669: t244_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_4670: t244_path0 = 1 -> const_Street_Grvl <= 1
 ind_4671: t244_path0 = 1 -> const_2nd_Flr_SF <= 1629
 ind_4672: t244_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_4673: t244_path1 = 1 -> const_Street_Grvl <= 1
 ind_4674: t244_path1 = 1 -> const_2nd_Flr_SF >= 1629
 ind_4675: t244_path2 = 1 -> x_1st_Flr_SF <= 2515
 ind_4676: t244_path2 = 1 -> const_Street_Grvl >= 1
 ind_4677: t244_path3 = 1 -> x_1st_Flr_SF >= 2515
 ind_4678: t245_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4679: t245_path0 = 1 -> x_Wood_Deck_SF <= 22
 ind_4680: t245_path0 = 1 -> x_Enclosed_Porch <= 91
 ind_4681: t245_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4682: t245_path1 = 1 -> x_Wood_Deck_SF <= 22
 ind_4683: t245_path1 = 1 -> x_Enclosed_Porch >= 91
 ind_4684: t245_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4685: t245_path2 = 1 -> x_Wood_Deck_SF >= 22
 ind_4686: t245_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4687: t245_path3 = 1 -> const_Overall_Cond <= 6
 ind_4688: t245_path3 = 1 -> const_MS_Zoning_FV <= 1
 ind_4689: t245_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4690: t245_path4 = 1 -> const_Overall_Cond <= 6
 ind_4691: t245_path4 = 1 -> const_MS_Zoning_FV >= 1
 ind_4692: t245_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4693: t245_path5 = 1 -> const_Overall_Cond >= 6
 ind_4694: t245_path5 = 1 -> x_1st_Flr_SF <= 1358
 ind_4695: t245_path6 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4696: t245_path6 = 1 -> const_Overall_Cond >= 6
 ind_4697: t245_path6 = 1 -> x_1st_Flr_SF >= 1358
 ind_4698: t246_path0 = 1 -> x_Gr_Liv_Area <= 3086
 ind_4699: t246_path0 = 1 -> const_Lot_Area <= 4000
 ind_4700: t246_path1 = 1 -> x_Gr_Liv_Area <= 3086
 ind_4701: t246_path1 = 1 -> const_Lot_Area >= 4000
 ind_4702: t246_path1 = 1 -> const_Fireplaces <= 1
 ind_4703: t246_path2 = 1 -> x_Gr_Liv_Area <= 3086
 ind_4704: t246_path2 = 1 -> const_Lot_Area >= 4000
 ind_4705: t246_path2 = 1 -> const_Fireplaces >= 1
 ind_4706: t246_path3 = 1 -> x_Gr_Liv_Area >= 3086
 ind_4707: t247_path0 = 1 -> x_Kitchen_Qual <= 2
 ind_4708: t247_path0 = 1 -> x_elect_is_FuseA <= 1
 ind_4709: t247_path1 = 1 -> x_Kitchen_Qual <= 2
 ind_4710: t247_path1 = 1 -> x_elect_is_FuseA >= 1
 ind_4711: t247_path2 = 1 -> x_Kitchen_Qual >= 2
 ind_4712: t247_path2 = 1 -> x_Garage_Area <= 871
 ind_4713: t247_path2 = 1 -> x_Garage_Qual <= 2
 ind_4714: t247_path3 = 1 -> x_Kitchen_Qual >= 2
 ind_4715: t247_path3 = 1 -> x_Garage_Area <= 871
 ind_4716: t247_path3 = 1 -> x_Garage_Qual >= 2
 ind_4717: t247_path4 = 1 -> x_Kitchen_Qual >= 2
 ind_4718: t247_path4 = 1 -> x_Garage_Area >= 871
 ind_4719: t248_path0 = 1 -> x_Gr_Liv_Area <= 988
 ind_4720: t248_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4721: t248_path0 = 1 -> x_Enclosed_Porch <= 121
 ind_4722: t248_path1 = 1 -> x_Gr_Liv_Area <= 988
 ind_4723: t248_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4724: t248_path1 = 1 -> x_Enclosed_Porch >= 121
 ind_4725: t248_path2 = 1 -> x_Gr_Liv_Area <= 988
 ind_4726: t248_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4727: t248_path2 = 1 -> x_Garage_Cond <= 2
 ind_4728: t248_path3 = 1 -> x_Gr_Liv_Area <= 988
 ind_4729: t248_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4730: t248_path3 = 1 -> x_Garage_Cond >= 2
 ind_4731: t248_path4 = 1 -> x_Gr_Liv_Area >= 988
 ind_4732: t248_path4 = 1 -> Overall_Qual_calc <= 7
 ind_4733: t248_path4 = 1 -> const_2nd_Flr_SF <= 672
 ind_4734: t248_path5 = 1 -> x_Gr_Liv_Area >= 988
 ind_4735: t248_path5 = 1 -> Overall_Qual_calc <= 7
 ind_4736: t248_path5 = 1 -> const_2nd_Flr_SF >= 672
 ind_4737: t248_path6 = 1 -> x_Gr_Liv_Area >= 988
 ind_4738: t248_path6 = 1 -> Overall_Qual_calc >= 7
 ind_4739: t248_path6 = 1 -> Overall_Qual_calc <= 9
 ind_4740: t248_path7 = 1 -> x_Gr_Liv_Area >= 988
 ind_4741: t248_path7 = 1 -> Overall_Qual_calc >= 7
 ind_4742: t248_path7 = 1 -> Overall_Qual_calc >= 9
 ind_4743: t249_path0 = 1 -> x_Mas_Vnr_Area <= 922
 ind_4744: t249_path0 = 1 -> const_Yr_Sold <= 2008
 ind_4745: t249_path0 = 1 -> x_Gr_Liv_Area <= 2794
 ind_4746: t249_path1 = 1 -> x_Mas_Vnr_Area <= 922
 ind_4747: t249_path1 = 1 -> const_Yr_Sold <= 2008
 ind_4748: t249_path1 = 1 -> x_Gr_Liv_Area >= 2794
 ind_4749: t249_path2 = 1 -> x_Mas_Vnr_Area <= 922
 ind_4750: t249_path2 = 1 -> const_Yr_Sold >= 2008
 ind_4751: t249_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4752: t249_path3 = 1 -> x_Mas_Vnr_Area <= 922
 ind_4753: t249_path3 = 1 -> const_Yr_Sold >= 2008
 ind_4754: t249_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4755: t249_path4 = 1 -> x_Mas_Vnr_Area >= 922
 ind_4756: t250_path0 = 1 -> x_Bsmt_Unf_SF <= 702
 ind_4757: t250_path0 = 1 -> x_1st_Flr_SF <= 2200
 ind_4758: t250_path0 = 1 -> x_1st_Flr_SF <= 1565
 ind_4759: t250_path1 = 1 -> x_Bsmt_Unf_SF <= 702
 ind_4760: t250_path1 = 1 -> x_1st_Flr_SF <= 2200
 ind_4761: t250_path1 = 1 -> x_1st_Flr_SF >= 1565
 ind_4762: t250_path2 = 1 -> x_Bsmt_Unf_SF <= 702
 ind_4763: t250_path2 = 1 -> x_1st_Flr_SF >= 2200
 ind_4764: t250_path3 = 1 -> x_Bsmt_Unf_SF >= 702
 ind_4765: t251_path0 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_4766: t251_path0 = 1 -> const_MS_SubClass_70 <= 1
 ind_4767: t251_path0 = 1 -> Overall_Qual_calc <= 5
 ind_4768: t251_path1 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_4769: t251_path1 = 1 -> const_MS_SubClass_70 <= 1
 ind_4770: t251_path1 = 1 -> Overall_Qual_calc >= 5
 ind_4771: t251_path2 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_4772: t251_path2 = 1 -> const_MS_SubClass_70 >= 1
 ind_4773: t251_path2 = 1 -> const_Year_Built <= 1892
 ind_4774: t251_path3 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_4775: t251_path3 = 1 -> const_MS_SubClass_70 >= 1
 ind_4776: t251_path3 = 1 -> const_Year_Built >= 1892
 ind_4777: t251_path4 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_4778: t252_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4779: t252_path0 = 1 -> x_Enclosed_Porch <= 91
 ind_4780: t252_path0 = 1 -> x_Wood_Deck_SF <= 22
 ind_4781: t252_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4782: t252_path1 = 1 -> x_Enclosed_Porch <= 91
 ind_4783: t252_path1 = 1 -> x_Wood_Deck_SF >= 22
 ind_4784: t252_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4785: t252_path2 = 1 -> x_Enclosed_Porch >= 91
 ind_4786: t252_path2 = 1 -> x_ex1_is_MetalSd <= 1
 ind_4787: t252_path3 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4788: t252_path3 = 1 -> x_Enclosed_Porch >= 91
 ind_4789: t252_path3 = 1 -> x_ex1_is_MetalSd >= 1
 ind_4790: t252_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4791: t252_path4 = 1 -> const_Sale_Condition_Family <= 1
 ind_4792: t252_path4 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_4793: t252_path5 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4794: t252_path5 = 1 -> const_Sale_Condition_Family <= 1
 ind_4795: t252_path5 = 1 -> const_Neighborhood_BrkSide >= 1
 ind_4796: t252_path6 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4797: t252_path6 = 1 -> const_Sale_Condition_Family >= 1
 ind_4798: t252_path6 = 1 -> x_Garage_Area <= 506
 ind_4799: t252_path7 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4800: t252_path7 = 1 -> const_Sale_Condition_Family >= 1
 ind_4801: t252_path7 = 1 -> x_Garage_Area >= 506
 ind_4802: t253_path0 = 1 -> const_Lot_Area <= 23257
 ind_4803: t253_path0 = 1 -> const_Lot_Area <= 18386
 ind_4804: t253_path0 = 1 -> const_Alley_Pave <= 1
 ind_4805: t253_path1 = 1 -> const_Lot_Area <= 23257
 ind_4806: t253_path1 = 1 -> const_Lot_Area <= 18386
 ind_4807: t253_path1 = 1 -> const_Alley_Pave >= 1
 ind_4808: t253_path2 = 1 -> const_Lot_Area <= 23257
 ind_4809: t253_path2 = 1 -> const_Lot_Area >= 18386
 ind_4810: t253_path3 = 1 -> const_Lot_Area >= 23257
 ind_4811: t253_path3 = 1 -> x_Gr_Liv_Area <= 2544
 ind_4812: t253_path3 = 1 -> x_Gr_Liv_Area <= 1491
 ind_4813: t253_path4 = 1 -> const_Lot_Area >= 23257
 ind_4814: t253_path4 = 1 -> x_Gr_Liv_Area <= 2544
 ind_4815: t253_path4 = 1 -> x_Gr_Liv_Area >= 1491
 ind_4816: t253_path5 = 1 -> const_Lot_Area >= 23257
 ind_4817: t253_path5 = 1 -> x_Gr_Liv_Area >= 2544
 ind_4818: t254_path0 = 1 -> Overall_Qual_calc <= 3
 ind_4819: t254_path1 = 1 -> Overall_Qual_calc >= 3
 ind_4820: t254_path1 = 1 -> const_MS_Zoning_FV <= 1
 ind_4821: t254_path1 = 1 -> x_Open_Porch_SF <= 291
 ind_4822: t254_path2 = 1 -> Overall_Qual_calc >= 3
 ind_4823: t254_path2 = 1 -> const_MS_Zoning_FV <= 1
 ind_4824: t254_path2 = 1 -> x_Open_Porch_SF >= 291
 ind_4825: t254_path3 = 1 -> Overall_Qual_calc >= 3
 ind_4826: t254_path3 = 1 -> const_MS_Zoning_FV >= 1
 ind_4827: t255_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4828: t255_path0 = 1 -> x_Bedroom_AbvGr <= 3
 ind_4829: t255_path0 = 1 -> x_Bsmt_Unf_SF <= 1510
 ind_4830: t255_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4831: t255_path1 = 1 -> x_Bedroom_AbvGr <= 3
 ind_4832: t255_path1 = 1 -> x_Bsmt_Unf_SF >= 1510
 ind_4833: t255_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4834: t255_path2 = 1 -> x_Bedroom_AbvGr >= 3
 ind_4835: t255_path2 = 1 -> x_1st_Flr_SF <= 1402
 ind_4836: t255_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4837: t255_path3 = 1 -> x_Bedroom_AbvGr >= 3
 ind_4838: t255_path3 = 1 -> x_1st_Flr_SF >= 1402
 ind_4839: t255_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_4840: t255_path4 = 1 -> const_Garage_Yr_Blt <= 1923
 ind_4841: t255_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_4842: t255_path5 = 1 -> const_Garage_Yr_Blt >= 1923
 ind_4843: t255_path5 = 1 -> const_Lot_Area <= 9605
 ind_4844: t255_path6 = 1 -> x_ex1_is_AsbShng >= 1
 ind_4845: t255_path6 = 1 -> const_Garage_Yr_Blt >= 1923
 ind_4846: t255_path6 = 1 -> const_Lot_Area >= 9605
 ind_4847: t256_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_4848: t256_path0 = 1 -> const_Land_Slope_Gtl <= 1
 ind_4849: t256_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_4850: t256_path1 = 1 -> const_Land_Slope_Gtl >= 1
 ind_4851: t256_path1 = 1 -> const_Year_Built <= 1911
 ind_4852: t256_path2 = 1 -> const_MS_SubClass_30 <= 1
 ind_4853: t256_path2 = 1 -> const_Land_Slope_Gtl >= 1
 ind_4854: t256_path2 = 1 -> const_Year_Built >= 1911
 ind_4855: t256_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_4856: t256_path3 = 1 -> x_Garage_Area <= 497
 ind_4857: t256_path3 = 1 -> const_Lot_Area <= 8094
 ind_4858: t256_path4 = 1 -> const_MS_SubClass_30 >= 1
 ind_4859: t256_path4 = 1 -> x_Garage_Area <= 497
 ind_4860: t256_path4 = 1 -> const_Lot_Area >= 8094
 ind_4861: t256_path5 = 1 -> const_MS_SubClass_30 >= 1
 ind_4862: t256_path5 = 1 -> x_Garage_Area >= 497
 ind_4863: t256_path5 = 1 -> x_Open_Porch_SF <= 4
 ind_4864: t256_path6 = 1 -> const_MS_SubClass_30 >= 1
 ind_4865: t256_path6 = 1 -> x_Garage_Area >= 497
 ind_4866: t256_path6 = 1 -> x_Open_Porch_SF >= 4
 ind_4867: t257_path0 = 1 -> x_Gr_Liv_Area <= 988
 ind_4868: t257_path0 = 1 -> x_Garage_Area <= 216
 ind_4869: t257_path0 = 1 -> central_air_no <= 1
 ind_4870: t257_path1 = 1 -> x_Gr_Liv_Area <= 988
 ind_4871: t257_path1 = 1 -> x_Garage_Area <= 216
 ind_4872: t257_path1 = 1 -> central_air_no >= 1
 ind_4873: t257_path2 = 1 -> x_Gr_Liv_Area <= 988
 ind_4874: t257_path2 = 1 -> x_Garage_Area >= 216
 ind_4875: t257_path2 = 1 -> x_Exter_Cond <= 2
 ind_4876: t257_path3 = 1 -> x_Gr_Liv_Area <= 988
 ind_4877: t257_path3 = 1 -> x_Garage_Area >= 216
 ind_4878: t257_path3 = 1 -> x_Exter_Cond >= 2
 ind_4879: t257_path4 = 1 -> x_Gr_Liv_Area >= 988
 ind_4880: t257_path4 = 1 -> const_Overall_Cond <= 4
 ind_4881: t257_path5 = 1 -> x_Gr_Liv_Area >= 988
 ind_4882: t257_path5 = 1 -> const_Overall_Cond >= 4
 ind_4883: t257_path5 = 1 -> x_Mas_Vnr_Area <= 673
 ind_4884: t257_path6 = 1 -> x_Gr_Liv_Area >= 988
 ind_4885: t257_path6 = 1 -> const_Overall_Cond >= 4
 ind_4886: t257_path6 = 1 -> x_Mas_Vnr_Area >= 673
 ind_4887: t258_path0 = 1 -> const_Year_Built <= 1914
 ind_4888: t258_path1 = 1 -> const_Year_Built >= 1914
 ind_4889: t258_path1 = 1 -> central_air_no <= 1
 ind_4890: t258_path1 = 1 -> const_MS_Zoning_FV <= 1
 ind_4891: t258_path2 = 1 -> const_Year_Built >= 1914
 ind_4892: t258_path2 = 1 -> central_air_no <= 1
 ind_4893: t258_path2 = 1 -> const_MS_Zoning_FV >= 1
 ind_4894: t258_path3 = 1 -> const_Year_Built >= 1914
 ind_4895: t258_path3 = 1 -> central_air_no >= 1
 ind_4896: t258_path3 = 1 -> x_ex2_is_MetalSd <= 1
 ind_4897: t258_path4 = 1 -> const_Year_Built >= 1914
 ind_4898: t258_path4 = 1 -> central_air_no >= 1
 ind_4899: t258_path4 = 1 -> x_ex2_is_MetalSd >= 1
 ind_4900: t259_path0 = 1 -> const_2nd_Flr_SF <= 1296
 ind_4901: t259_path0 = 1 -> const_Overall_Cond <= 6
 ind_4902: t259_path0 = 1 -> x_Heating_QC <= 3
 ind_4903: t259_path1 = 1 -> const_2nd_Flr_SF <= 1296
 ind_4904: t259_path1 = 1 -> const_Overall_Cond <= 6
 ind_4905: t259_path1 = 1 -> x_Heating_QC >= 3
 ind_4906: t259_path2 = 1 -> const_2nd_Flr_SF <= 1296
 ind_4907: t259_path2 = 1 -> const_Overall_Cond >= 6
 ind_4908: t259_path2 = 1 -> const_2nd_Flr_SF <= 167
 ind_4909: t259_path3 = 1 -> const_2nd_Flr_SF <= 1296
 ind_4910: t259_path3 = 1 -> const_Overall_Cond >= 6
 ind_4911: t259_path3 = 1 -> const_2nd_Flr_SF >= 167
 ind_4912: t259_path4 = 1 -> const_2nd_Flr_SF >= 1296
 ind_4913: t260_path0 = 1 -> const_Lot_Frontage <= 121
 ind_4914: t260_path0 = 1 -> x_Gr_Liv_Area <= 894
 ind_4915: t260_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_4916: t260_path1 = 1 -> const_Lot_Frontage <= 121
 ind_4917: t260_path1 = 1 -> x_Gr_Liv_Area <= 894
 ind_4918: t260_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_4919: t260_path2 = 1 -> const_Lot_Frontage <= 121
 ind_4920: t260_path2 = 1 -> x_Gr_Liv_Area >= 894
 ind_4921: t260_path2 = 1 -> x_mvt_is_BrkCmn <= 1
 ind_4922: t260_path3 = 1 -> const_Lot_Frontage <= 121
 ind_4923: t260_path3 = 1 -> x_Gr_Liv_Area >= 894
 ind_4924: t260_path3 = 1 -> x_mvt_is_BrkCmn >= 1
 ind_4925: t260_path4 = 1 -> const_Lot_Frontage >= 121
 ind_4926: t260_path4 = 1 -> x_Bsmt_Unf_SF <= 448
 ind_4927: t260_path5 = 1 -> const_Lot_Frontage >= 121
 ind_4928: t260_path5 = 1 -> x_Bsmt_Unf_SF >= 448
 ind_4929: t261_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_4930: t261_path0 = 1 -> const_Lot_Area <= 27697
 ind_4931: t261_path0 = 1 -> const_MS_SubClass_30 <= 1
 ind_4932: t261_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_4933: t261_path1 = 1 -> const_Lot_Area <= 27697
 ind_4934: t261_path1 = 1 -> const_MS_SubClass_30 >= 1
 ind_4935: t261_path2 = 1 -> x_1st_Flr_SF <= 2515
 ind_4936: t261_path2 = 1 -> const_Lot_Area >= 27697
 ind_4937: t261_path3 = 1 -> x_1st_Flr_SF >= 2515
 ind_4938: t262_path0 = 1 -> Overall_Qual_calc <= 3
 ind_4939: t262_path1 = 1 -> Overall_Qual_calc >= 3
 ind_4940: t262_path1 = 1 -> x_Kitchen_AbvGr <= 2
 ind_4941: t262_path1 = 1 -> x_Gr_Liv_Area <= 2649
 ind_4942: t262_path2 = 1 -> Overall_Qual_calc >= 3
 ind_4943: t262_path2 = 1 -> x_Kitchen_AbvGr <= 2
 ind_4944: t262_path2 = 1 -> x_Gr_Liv_Area >= 2649
 ind_4945: t262_path3 = 1 -> Overall_Qual_calc >= 3
 ind_4946: t262_path3 = 1 -> x_Kitchen_AbvGr >= 2
 ind_4947: t262_path3 = 1 -> x_Garage_Cond <= 1
 ind_4948: t262_path4 = 1 -> Overall_Qual_calc >= 3
 ind_4949: t262_path4 = 1 -> x_Kitchen_AbvGr >= 2
 ind_4950: t262_path4 = 1 -> x_Garage_Cond >= 1
 ind_4951: t263_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_4952: t263_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_4953: t263_path0 = 1 -> x_Gr_Liv_Area <= 810
 ind_4954: t263_path1 = 1 -> const_Mo_Sold_2 <= 1
 ind_4955: t263_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_4956: t263_path1 = 1 -> x_Gr_Liv_Area >= 810
 ind_4957: t263_path2 = 1 -> const_Mo_Sold_2 <= 1
 ind_4958: t263_path2 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_4959: t263_path3 = 1 -> const_Mo_Sold_2 >= 1
 ind_4960: t263_path3 = 1 -> const_Lot_Area <= 9965
 ind_4961: t263_path4 = 1 -> const_Mo_Sold_2 >= 1
 ind_4962: t263_path4 = 1 -> const_Lot_Area >= 9965
 ind_4963: t263_path4 = 1 -> const_Sale_Condition_Normal <= 1
 ind_4964: t263_path5 = 1 -> const_Mo_Sold_2 >= 1
 ind_4965: t263_path5 = 1 -> const_Lot_Area >= 9965
 ind_4966: t263_path5 = 1 -> const_Sale_Condition_Normal >= 1
 ind_4967: t264_path0 = 1 -> Overall_Qual_calc <= 3
 ind_4968: t264_path1 = 1 -> Overall_Qual_calc >= 3
 ind_4969: t264_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4970: t264_path1 = 1 -> x_Open_Porch_SF <= 182
 ind_4971: t264_path2 = 1 -> Overall_Qual_calc >= 3
 ind_4972: t264_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_4973: t264_path2 = 1 -> x_Open_Porch_SF >= 182
 ind_4974: t264_path3 = 1 -> Overall_Qual_calc >= 3
 ind_4975: t264_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4976: t264_path3 = 1 -> const_Sale_Condition_Family <= 1
 ind_4977: t264_path4 = 1 -> Overall_Qual_calc >= 3
 ind_4978: t264_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_4979: t264_path4 = 1 -> const_Sale_Condition_Family >= 1
 ind_4980: t265_path0 = 1 -> const_Lot_Frontage <= 152
 ind_4981: t265_path0 = 1 -> Overall_Qual_calc <= 5
 ind_4982: t265_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4983: t265_path1 = 1 -> const_Lot_Frontage <= 152
 ind_4984: t265_path1 = 1 -> Overall_Qual_calc <= 5
 ind_4985: t265_path1 = 1 -> x_ex1_is_AsbShng >= 1
 ind_4986: t265_path2 = 1 -> const_Lot_Frontage <= 152
 ind_4987: t265_path2 = 1 -> Overall_Qual_calc >= 5
 ind_4988: t265_path2 = 1 -> const_Year_Built <= 1913
 ind_4989: t265_path3 = 1 -> const_Lot_Frontage <= 152
 ind_4990: t265_path3 = 1 -> Overall_Qual_calc >= 5
 ind_4991: t265_path3 = 1 -> const_Year_Built >= 1913
 ind_4992: t265_path4 = 1 -> const_Lot_Frontage >= 152
 ind_4993: t266_path0 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_4994: t266_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4995: t266_path0 = 1 -> x_Garage_Area <= 811
 ind_4996: t266_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_4997: t266_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_4998: t266_path1 = 1 -> x_Garage_Area >= 811
 ind_4999: t266_path2 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_5000: t266_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5001: t266_path2 = 1 -> const_Total_Bsmt_SF <= 727
 ind_5002: t266_path3 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_5003: t266_path3 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5004: t266_path3 = 1 -> const_Total_Bsmt_SF >= 727
 ind_5005: t266_path4 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_5006: t266_path4 = 1 -> x_Wood_Deck_SF <= 511
 ind_5007: t266_path4 = 1 -> const_Fireplaces <= 2
 ind_5008: t266_path5 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_5009: t266_path5 = 1 -> x_Wood_Deck_SF <= 511
 ind_5010: t266_path5 = 1 -> const_Fireplaces >= 2
 ind_5011: t266_path6 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_5012: t266_path6 = 1 -> x_Wood_Deck_SF >= 511
 ind_5013: t267_path0 = 1 -> Overall_Qual_calc <= 7
 ind_5014: t267_path0 = 1 -> const_2nd_Flr_SF <= 1276
 ind_5015: t267_path0 = 1 -> x_Open_Porch_SF <= 21
 ind_5016: t267_path1 = 1 -> Overall_Qual_calc <= 7
 ind_5017: t267_path1 = 1 -> const_2nd_Flr_SF <= 1276
 ind_5018: t267_path1 = 1 -> x_Open_Porch_SF >= 21
 ind_5019: t267_path2 = 1 -> Overall_Qual_calc <= 7
 ind_5020: t267_path2 = 1 -> const_2nd_Flr_SF >= 1276
 ind_5021: t267_path3 = 1 -> Overall_Qual_calc >= 7
 ind_5022: t267_path3 = 1 -> const_Sale_Condition_Family <= 1
 ind_5023: t267_path3 = 1 -> const_Lot_Frontage <= 114
 ind_5024: t267_path4 = 1 -> Overall_Qual_calc >= 7
 ind_5025: t267_path4 = 1 -> const_Sale_Condition_Family <= 1
 ind_5026: t267_path4 = 1 -> const_Lot_Frontage >= 114
 ind_5027: t267_path5 = 1 -> Overall_Qual_calc >= 7
 ind_5028: t267_path5 = 1 -> const_Sale_Condition_Family >= 1
 ind_5029: t268_path0 = 1 -> x_Open_Porch_SF <= 282
 ind_5030: t268_path0 = 1 -> const_Fireplaces <= 2
 ind_5031: t268_path0 = 1 -> x_ex2_is_MetalSd <= 1
 ind_5032: t268_path1 = 1 -> x_Open_Porch_SF <= 282
 ind_5033: t268_path1 = 1 -> const_Fireplaces <= 2
 ind_5034: t268_path1 = 1 -> x_ex2_is_MetalSd >= 1
 ind_5035: t268_path2 = 1 -> x_Open_Porch_SF <= 282
 ind_5036: t268_path2 = 1 -> const_Fireplaces >= 2
 ind_5037: t268_path2 = 1 -> x_Open_Porch_SF <= 31
 ind_5038: t268_path3 = 1 -> x_Open_Porch_SF <= 282
 ind_5039: t268_path3 = 1 -> const_Fireplaces >= 2
 ind_5040: t268_path3 = 1 -> x_Open_Porch_SF >= 31
 ind_5041: t268_path4 = 1 -> x_Open_Porch_SF >= 282
 ind_5042: t268_path4 = 1 -> const_Garage_Finish_Unf <= 1
 ind_5043: t268_path5 = 1 -> x_Open_Porch_SF >= 282
 ind_5044: t268_path5 = 1 -> const_Garage_Finish_Unf >= 1
 ind_5045: t269_path0 = 1 -> const_2nd_Flr_SF <= 1611
 ind_5046: t269_path0 = 1 -> const_Lot_Frontage <= 117
 ind_5047: t269_path0 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_5048: t269_path1 = 1 -> const_2nd_Flr_SF <= 1611
 ind_5049: t269_path1 = 1 -> const_Lot_Frontage <= 117
 ind_5050: t269_path1 = 1 -> const_Garage_Yr_Blt >= 2008
 ind_5051: t269_path2 = 1 -> const_2nd_Flr_SF <= 1611
 ind_5052: t269_path2 = 1 -> const_Lot_Frontage >= 117
 ind_5053: t269_path2 = 1 -> const_Garage_Yr_Blt <= 1966
 ind_5054: t269_path3 = 1 -> const_2nd_Flr_SF <= 1611
 ind_5055: t269_path3 = 1 -> const_Lot_Frontage >= 117
 ind_5056: t269_path3 = 1 -> const_Garage_Yr_Blt >= 1966
 ind_5057: t269_path4 = 1 -> const_2nd_Flr_SF >= 1611
 ind_5058: t270_path0 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_5059: t270_path0 = 1 -> x_Exter_Cond <= 2
 ind_5060: t270_path0 = 1 -> const_Year_Remod_Add <= 1952
 ind_5061: t270_path1 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_5062: t270_path1 = 1 -> x_Exter_Cond <= 2
 ind_5063: t270_path1 = 1 -> const_Year_Remod_Add >= 1952
 ind_5064: t270_path2 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_5065: t270_path2 = 1 -> x_Exter_Cond >= 2
 ind_5066: t270_path2 = 1 -> const_Garage_Type_CarPort <= 1
 ind_5067: t270_path3 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_5068: t270_path3 = 1 -> x_Exter_Cond >= 2
 ind_5069: t270_path3 = 1 -> const_Garage_Type_CarPort >= 1
 ind_5070: t270_path4 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_5071: t270_path4 = 1 -> x_Mas_Vnr_Area <= 567
 ind_5072: t270_path5 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_5073: t270_path5 = 1 -> x_Mas_Vnr_Area >= 567
 ind_5074: t271_path0 = 1 -> x_Mas_Vnr_Area <= 788
 ind_5075: t271_path0 = 1 -> Overall_Qual_calc <= 4
 ind_5076: t271_path0 = 1 -> x_Bsmt_Unf_SF <= 266
 ind_5077: t271_path1 = 1 -> x_Mas_Vnr_Area <= 788
 ind_5078: t271_path1 = 1 -> Overall_Qual_calc <= 4
 ind_5079: t271_path1 = 1 -> x_Bsmt_Unf_SF >= 266
 ind_5080: t271_path2 = 1 -> x_Mas_Vnr_Area <= 788
 ind_5081: t271_path2 = 1 -> Overall_Qual_calc >= 4
 ind_5082: t271_path2 = 1 -> const_Sale_Condition_Partial <= 1
 ind_5083: t271_path3 = 1 -> x_Mas_Vnr_Area <= 788
 ind_5084: t271_path3 = 1 -> Overall_Qual_calc >= 4
 ind_5085: t271_path3 = 1 -> const_Sale_Condition_Partial >= 1
 ind_5086: t271_path4 = 1 -> x_Mas_Vnr_Area >= 788
 ind_5087: t271_path4 = 1 -> x_Gr_Liv_Area <= 2690
 ind_5088: t271_path5 = 1 -> x_Mas_Vnr_Area >= 788
 ind_5089: t271_path5 = 1 -> x_Gr_Liv_Area >= 2690
 ind_5090: t272_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5091: t272_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_5092: t272_path0 = 1 -> Overall_Qual_calc <= 3
 ind_5093: t272_path1 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5094: t272_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_5095: t272_path1 = 1 -> Overall_Qual_calc >= 3
 ind_5096: t272_path2 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5097: t272_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_5098: t272_path2 = 1 -> x_Bsmt_Unf_SF <= 702
 ind_5099: t272_path3 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5100: t272_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_5101: t272_path3 = 1 -> x_Bsmt_Unf_SF >= 702
 ind_5102: t272_path4 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5103: t272_path4 = 1 -> x_Mas_Vnr_Area <= 3
 ind_5104: t272_path4 = 1 -> const_Lot_Area <= 16285
 ind_5105: t272_path5 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5106: t272_path5 = 1 -> x_Mas_Vnr_Area <= 3
 ind_5107: t272_path5 = 1 -> const_Lot_Area >= 16285
 ind_5108: t272_path6 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5109: t272_path6 = 1 -> x_Mas_Vnr_Area >= 3
 ind_5110: t272_path6 = 1 -> const_Foundation_PConc <= 1
 ind_5111: t272_path7 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5112: t272_path7 = 1 -> x_Mas_Vnr_Area >= 3
 ind_5113: t272_path7 = 1 -> const_Foundation_PConc >= 1
 ind_5114: t273_path0 = 1 -> x_Garage_Area <= 1154
 ind_5115: t273_path0 = 1 -> const_Yr_Sold <= 2009
 ind_5116: t273_path0 = 1 -> x_Half_Bath <= 2
 ind_5117: t273_path1 = 1 -> x_Garage_Area <= 1154
 ind_5118: t273_path1 = 1 -> const_Yr_Sold <= 2009
 ind_5119: t273_path1 = 1 -> x_Half_Bath >= 2
 ind_5120: t273_path2 = 1 -> x_Garage_Area <= 1154
 ind_5121: t273_path2 = 1 -> const_Yr_Sold >= 2009
 ind_5122: t273_path2 = 1 -> x_Gr_Liv_Area <= 816
 ind_5123: t273_path3 = 1 -> x_Garage_Area <= 1154
 ind_5124: t273_path3 = 1 -> const_Yr_Sold >= 2009
 ind_5125: t273_path3 = 1 -> x_Gr_Liv_Area >= 816
 ind_5126: t273_path4 = 1 -> x_Garage_Area >= 1154
 ind_5127: t274_path0 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_5128: t274_path0 = 1 -> const_Garage_Yr_Blt <= 1953
 ind_5129: t274_path0 = 1 -> x_1st_Flr_SF <= 1392
 ind_5130: t274_path1 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_5131: t274_path1 = 1 -> const_Garage_Yr_Blt <= 1953
 ind_5132: t274_path1 = 1 -> x_1st_Flr_SF >= 1392
 ind_5133: t274_path2 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_5134: t274_path2 = 1 -> const_Garage_Yr_Blt >= 1953
 ind_5135: t274_path2 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_5136: t274_path3 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_5137: t274_path3 = 1 -> const_Garage_Yr_Blt >= 1953
 ind_5138: t274_path3 = 1 -> x_Bsmt_Unf_SF >= 476
 ind_5139: t274_path4 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_5140: t275_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_5141: t275_path0 = 1 -> x_Enclosed_Porch <= 16
 ind_5142: t275_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_5143: t275_path1 = 1 -> x_Enclosed_Porch >= 16
 ind_5144: t275_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_5145: t275_path2 = 1 -> const_Misc_Val <= 54
 ind_5146: t275_path2 = 1 -> const_Condition_1_Norm <= 1
 ind_5147: t275_path3 = 1 -> x_Gr_Liv_Area >= 747
 ind_5148: t275_path3 = 1 -> const_Misc_Val <= 54
 ind_5149: t275_path3 = 1 -> const_Condition_1_Norm >= 1
 ind_5150: t275_path4 = 1 -> x_Gr_Liv_Area >= 747
 ind_5151: t275_path4 = 1 -> const_Misc_Val >= 54
 ind_5152: t275_path4 = 1 -> x_Bedroom_AbvGr <= 3
 ind_5153: t275_path5 = 1 -> x_Gr_Liv_Area >= 747
 ind_5154: t275_path5 = 1 -> const_Misc_Val >= 54
 ind_5155: t275_path5 = 1 -> x_Bedroom_AbvGr >= 3
 ind_5156: t276_path0 = 1 -> const_Lot_Area <= 3182
 ind_5157: t276_path1 = 1 -> const_Lot_Area >= 3182
 ind_5158: t276_path1 = 1 -> x_ex2_is_Stucco <= 1
 ind_5159: t276_path1 = 1 -> x_1st_Flr_SF <= 1811
 ind_5160: t276_path2 = 1 -> const_Lot_Area >= 3182
 ind_5161: t276_path2 = 1 -> x_ex2_is_Stucco <= 1
 ind_5162: t276_path2 = 1 -> x_1st_Flr_SF >= 1811
 ind_5163: t276_path3 = 1 -> const_Lot_Area >= 3182
 ind_5164: t276_path3 = 1 -> x_ex2_is_Stucco >= 1
 ind_5165: t276_path3 = 1 -> x_BsmtFin_SF_1 <= 138
 ind_5166: t276_path4 = 1 -> const_Lot_Area >= 3182
 ind_5167: t276_path4 = 1 -> x_ex2_is_Stucco >= 1
 ind_5168: t276_path4 = 1 -> x_BsmtFin_SF_1 >= 138
 ind_5169: t277_path0 = 1 -> const_Overall_Cond <= 5
 ind_5170: t277_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_5171: t277_path0 = 1 -> const_Bsmt_Exposure_No <= 1
 ind_5172: t277_path1 = 1 -> const_Overall_Cond <= 5
 ind_5173: t277_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_5174: t277_path1 = 1 -> const_Bsmt_Exposure_No >= 1
 ind_5175: t277_path2 = 1 -> const_Overall_Cond <= 5
 ind_5176: t277_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_5177: t277_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_5178: t277_path3 = 1 -> const_Overall_Cond <= 5
 ind_5179: t277_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_5180: t277_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_5181: t277_path4 = 1 -> const_Overall_Cond >= 5
 ind_5182: t277_path4 = 1 -> const_Lot_Frontage <= 152
 ind_5183: t277_path4 = 1 -> x_Open_Porch_SF <= 276
 ind_5184: t277_path5 = 1 -> const_Overall_Cond >= 5
 ind_5185: t277_path5 = 1 -> const_Lot_Frontage <= 152
 ind_5186: t277_path5 = 1 -> x_Open_Porch_SF >= 276
 ind_5187: t277_path6 = 1 -> const_Overall_Cond >= 5
 ind_5188: t277_path6 = 1 -> const_Lot_Frontage >= 152
 ind_5189: t278_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5190: t278_path0 = 1 -> Overall_Qual_calc <= 3
 ind_5191: t278_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5192: t278_path1 = 1 -> Overall_Qual_calc >= 3
 ind_5193: t278_path1 = 1 -> const_Mo_Sold_7 <= 1
 ind_5194: t278_path2 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5195: t278_path2 = 1 -> Overall_Qual_calc >= 3
 ind_5196: t278_path2 = 1 -> const_Mo_Sold_7 >= 1
 ind_5197: t278_path3 = 1 -> x_ex1_is_BrkFace >= 1
 ind_5198: t279_path0 = 1 -> Overall_Qual_calc <= 4
 ind_5199: t279_path0 = 1 -> x_paved_drive_is_Y <= 1
 ind_5200: t279_path1 = 1 -> Overall_Qual_calc <= 4
 ind_5201: t279_path1 = 1 -> x_paved_drive_is_Y >= 1
 ind_5202: t279_path2 = 1 -> Overall_Qual_calc >= 4
 ind_5203: t279_path2 = 1 -> const_Year_Remod_Add <= 2007
 ind_5204: t279_path2 = 1 -> const_MS_SubClass_20 <= 1
 ind_5205: t279_path3 = 1 -> Overall_Qual_calc >= 4
 ind_5206: t279_path3 = 1 -> const_Year_Remod_Add <= 2007
 ind_5207: t279_path3 = 1 -> const_MS_SubClass_20 >= 1
 ind_5208: t279_path4 = 1 -> Overall_Qual_calc >= 4
 ind_5209: t279_path4 = 1 -> const_Year_Remod_Add >= 2007
 ind_5210: t279_path4 = 1 -> const_Lot_Frontage <= 95
 ind_5211: t279_path5 = 1 -> Overall_Qual_calc >= 4
 ind_5212: t279_path5 = 1 -> const_Year_Remod_Add >= 2007
 ind_5213: t279_path5 = 1 -> const_Lot_Frontage >= 95
 ind_5214: t280_path0 = 1 -> const_Overall_Cond <= 4
 ind_5215: t280_path0 = 1 -> x_Bsmt_Unf_SF <= 336
 ind_5216: t280_path1 = 1 -> const_Overall_Cond <= 4
 ind_5217: t280_path1 = 1 -> x_Bsmt_Unf_SF >= 336
 ind_5218: t280_path1 = 1 -> const_Year_Remod_Add <= 1952
 ind_5219: t280_path2 = 1 -> const_Overall_Cond <= 4
 ind_5220: t280_path2 = 1 -> x_Bsmt_Unf_SF >= 336
 ind_5221: t280_path2 = 1 -> const_Year_Remod_Add >= 1952
 ind_5222: t280_path3 = 1 -> const_Overall_Cond >= 4
 ind_5223: t280_path3 = 1 -> const_Condition_1_Artery <= 1
 ind_5224: t280_path3 = 1 -> const_Fireplaces <= 3
 ind_5225: t280_path4 = 1 -> const_Overall_Cond >= 4
 ind_5226: t280_path4 = 1 -> const_Condition_1_Artery <= 1
 ind_5227: t280_path4 = 1 -> const_Fireplaces >= 3
 ind_5228: t280_path5 = 1 -> const_Overall_Cond >= 4
 ind_5229: t280_path5 = 1 -> const_Condition_1_Artery >= 1
 ind_5230: t280_path5 = 1 -> x_Garage_Area <= 804
 ind_5231: t280_path6 = 1 -> const_Overall_Cond >= 4
 ind_5232: t280_path6 = 1 -> const_Condition_1_Artery >= 1
 ind_5233: t280_path6 = 1 -> x_Garage_Area >= 804
 ind_5234: t281_path0 = 1 -> Overall_Qual_calc <= 3
 ind_5235: t281_path1 = 1 -> Overall_Qual_calc >= 3
 ind_5236: t281_path1 = 1 -> Overall_Qual_calc <= 7
 ind_5237: t281_path1 = 1 -> const_Lot_Area <= 8000
 ind_5238: t281_path2 = 1 -> Overall_Qual_calc >= 3
 ind_5239: t281_path2 = 1 -> Overall_Qual_calc <= 7
 ind_5240: t281_path2 = 1 -> const_Lot_Area >= 8000
 ind_5241: t281_path3 = 1 -> Overall_Qual_calc >= 3
 ind_5242: t281_path3 = 1 -> Overall_Qual_calc >= 7
 ind_5243: t281_path3 = 1 -> const_Land_Contour_Bnk <= 1
 ind_5244: t281_path4 = 1 -> Overall_Qual_calc >= 3
 ind_5245: t281_path4 = 1 -> Overall_Qual_calc >= 7
 ind_5246: t281_path4 = 1 -> const_Land_Contour_Bnk >= 1
 ind_5247: t282_path0 = 1 -> x_Open_Porch_SF <= 418
 ind_5248: t282_path0 = 1 -> x_Mas_Vnr_Area <= 705
 ind_5249: t282_path0 = 1 -> const_Yr_Sold <= 2007
 ind_5250: t282_path1 = 1 -> x_Open_Porch_SF <= 418
 ind_5251: t282_path1 = 1 -> x_Mas_Vnr_Area <= 705
 ind_5252: t282_path1 = 1 -> const_Yr_Sold >= 2007
 ind_5253: t282_path2 = 1 -> x_Open_Porch_SF <= 418
 ind_5254: t282_path2 = 1 -> x_Mas_Vnr_Area >= 705
 ind_5255: t282_path3 = 1 -> x_Open_Porch_SF >= 418
 ind_5256: t283_path0 = 1 -> const_Bsmt_Qual <= 0
 ind_5257: t283_path0 = 1 -> const_Garage_Type_Detchd <= 1
 ind_5258: t283_path1 = 1 -> const_Bsmt_Qual <= 0
 ind_5259: t283_path1 = 1 -> const_Garage_Type_Detchd >= 1
 ind_5260: t283_path2 = 1 -> const_Bsmt_Qual >= 0
 ind_5261: t283_path2 = 1 -> const_MS_SubClass_30 <= 1
 ind_5262: t283_path2 = 1 -> const_Garage_Yr_Blt <= 1941
 ind_5263: t283_path3 = 1 -> const_Bsmt_Qual >= 0
 ind_5264: t283_path3 = 1 -> const_MS_SubClass_30 <= 1
 ind_5265: t283_path3 = 1 -> const_Garage_Yr_Blt >= 1941
 ind_5266: t283_path4 = 1 -> const_Bsmt_Qual >= 0
 ind_5267: t283_path4 = 1 -> const_MS_SubClass_30 >= 1
 ind_5268: t283_path4 = 1 -> const_Year_Remod_Add <= 1951
 ind_5269: t283_path5 = 1 -> const_Bsmt_Qual >= 0
 ind_5270: t283_path5 = 1 -> const_MS_SubClass_30 >= 1
 ind_5271: t283_path5 = 1 -> const_Year_Remod_Add >= 1951
 ind_5272: t284_path0 = 1 -> x_Enclosed_Porch <= 248
 ind_5273: t284_path0 = 1 -> x_Garage_Area <= 1154
 ind_5274: t284_path0 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_5275: t284_path1 = 1 -> x_Enclosed_Porch <= 248
 ind_5276: t284_path1 = 1 -> x_Garage_Area <= 1154
 ind_5277: t284_path1 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_5278: t284_path2 = 1 -> x_Enclosed_Porch <= 248
 ind_5279: t284_path2 = 1 -> x_Garage_Area >= 1154
 ind_5280: t284_path2 = 1 -> const_Bsmt_Qual <= 4
 ind_5281: t284_path3 = 1 -> x_Enclosed_Porch <= 248
 ind_5282: t284_path3 = 1 -> x_Garage_Area >= 1154
 ind_5283: t284_path3 = 1 -> const_Bsmt_Qual >= 4
 ind_5284: t284_path4 = 1 -> x_Enclosed_Porch >= 248
 ind_5285: t285_path0 = 1 -> x_TotRms_AbvGrd <= 12
 ind_5286: t285_path0 = 1 -> x_Heating_QC <= 3
 ind_5287: t285_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_5288: t285_path1 = 1 -> x_TotRms_AbvGrd <= 12
 ind_5289: t285_path1 = 1 -> x_Heating_QC <= 3
 ind_5290: t285_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_5291: t285_path2 = 1 -> x_TotRms_AbvGrd <= 12
 ind_5292: t285_path2 = 1 -> x_Heating_QC >= 3
 ind_5293: t285_path2 = 1 -> const_Condition_1_Artery <= 1
 ind_5294: t285_path3 = 1 -> x_TotRms_AbvGrd <= 12
 ind_5295: t285_path3 = 1 -> x_Heating_QC >= 3
 ind_5296: t285_path3 = 1 -> const_Condition_1_Artery >= 1
 ind_5297: t285_path4 = 1 -> x_TotRms_AbvGrd >= 12
 ind_5298: t285_path4 = 1 -> const_2nd_Flr_SF <= 975
 ind_5299: t285_path5 = 1 -> x_TotRms_AbvGrd >= 12
 ind_5300: t285_path5 = 1 -> const_2nd_Flr_SF >= 975
 ind_5301: t286_path0 = 1 -> Overall_Qual_calc <= 3
 ind_5302: t286_path0 = 1 -> x_Garage_Area <= 286
 ind_5303: t286_path1 = 1 -> Overall_Qual_calc <= 3
 ind_5304: t286_path1 = 1 -> x_Garage_Area >= 286
 ind_5305: t286_path2 = 1 -> Overall_Qual_calc >= 3
 ind_5306: t286_path2 = 1 -> const_Lot_Area <= 20431
 ind_5307: t286_path2 = 1 -> const_Lot_Area <= 17541
 ind_5308: t286_path3 = 1 -> Overall_Qual_calc >= 3
 ind_5309: t286_path3 = 1 -> const_Lot_Area <= 20431
 ind_5310: t286_path3 = 1 -> const_Lot_Area >= 17541
 ind_5311: t286_path4 = 1 -> Overall_Qual_calc >= 3
 ind_5312: t286_path4 = 1 -> const_Lot_Area >= 20431
 ind_5313: t286_path4 = 1 -> x_Bedroom_AbvGr <= 4
 ind_5314: t286_path5 = 1 -> Overall_Qual_calc >= 3
 ind_5315: t286_path5 = 1 -> const_Lot_Area >= 20431
 ind_5316: t286_path5 = 1 -> x_Bedroom_AbvGr >= 4
 ind_5317: t287_path0 = 1 -> Overall_Qual_calc <= 3
 ind_5318: t287_path0 = 1 -> x_Garage_Cond <= 2
 ind_5319: t287_path1 = 1 -> Overall_Qual_calc <= 3
 ind_5320: t287_path1 = 1 -> x_Garage_Cond >= 2
 ind_5321: t287_path2 = 1 -> Overall_Qual_calc >= 3
 ind_5322: t287_path2 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_5323: t287_path2 = 1 -> Overall_Qual_calc <= 7
 ind_5324: t287_path3 = 1 -> Overall_Qual_calc >= 3
 ind_5325: t287_path3 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_5326: t287_path3 = 1 -> Overall_Qual_calc >= 7
 ind_5327: t287_path4 = 1 -> Overall_Qual_calc >= 3
 ind_5328: t287_path4 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_5329: t287_path4 = 1 -> const_2nd_Flr_SF <= 505
 ind_5330: t287_path5 = 1 -> Overall_Qual_calc >= 3
 ind_5331: t287_path5 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_5332: t287_path5 = 1 -> const_2nd_Flr_SF >= 505
 ind_5333: t288_path0 = 1 -> const_Lot_Shape_IR3 <= 1
 ind_5334: t288_path0 = 1 -> x_Bsmt_Unf_SF <= 408
 ind_5335: t288_path0 = 1 -> x_BsmtFin_SF_1 <= 1573
 ind_5336: t288_path1 = 1 -> const_Lot_Shape_IR3 <= 1
 ind_5337: t288_path1 = 1 -> x_Bsmt_Unf_SF <= 408
 ind_5338: t288_path1 = 1 -> x_BsmtFin_SF_1 >= 1573
 ind_5339: t288_path2 = 1 -> const_Lot_Shape_IR3 <= 1
 ind_5340: t288_path2 = 1 -> x_Bsmt_Unf_SF >= 408
 ind_5341: t288_path2 = 1 -> x_Bedroom_AbvGr <= 4
 ind_5342: t288_path3 = 1 -> const_Lot_Shape_IR3 <= 1
 ind_5343: t288_path3 = 1 -> x_Bsmt_Unf_SF >= 408
 ind_5344: t288_path3 = 1 -> x_Bedroom_AbvGr >= 4
 ind_5345: t288_path4 = 1 -> const_Lot_Shape_IR3 >= 1
 ind_5346: t288_path4 = 1 -> const_Garage_Type_BuiltIn <= 1
 ind_5347: t288_path5 = 1 -> const_Lot_Shape_IR3 >= 1
 ind_5348: t288_path5 = 1 -> const_Garage_Type_BuiltIn >= 1
 ind_5349: t289_path0 = 1 -> x_Half_Bath <= 1
 ind_5350: t289_path0 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_5351: t289_path0 = 1 -> const_Mo_Sold_1 <= 1
 ind_5352: t289_path1 = 1 -> x_Half_Bath <= 1
 ind_5353: t289_path1 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_5354: t289_path1 = 1 -> const_Mo_Sold_1 >= 1
 ind_5355: t289_path2 = 1 -> x_Half_Bath <= 1
 ind_5356: t289_path2 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_5357: t289_path3 = 1 -> x_Half_Bath >= 1
 ind_5358: t289_path3 = 1 -> x_elect_is_SBrkr <= 1
 ind_5359: t289_path3 = 1 -> const_Condition_1_Norm <= 1
 ind_5360: t289_path4 = 1 -> x_Half_Bath >= 1
 ind_5361: t289_path4 = 1 -> x_elect_is_SBrkr <= 1
 ind_5362: t289_path4 = 1 -> const_Condition_1_Norm >= 1
 ind_5363: t289_path5 = 1 -> x_Half_Bath >= 1
 ind_5364: t289_path5 = 1 -> x_elect_is_SBrkr >= 1
 ind_5365: t290_path0 = 1 -> const_Land_Contour_Bnk <= 1
 ind_5366: t290_path0 = 1 -> x_Open_Porch_SF <= 75
 ind_5367: t290_path0 = 1 -> const_Year_Built <= 2008
 ind_5368: t290_path1 = 1 -> const_Land_Contour_Bnk <= 1
 ind_5369: t290_path1 = 1 -> x_Open_Porch_SF <= 75
 ind_5370: t290_path1 = 1 -> const_Year_Built >= 2008
 ind_5371: t290_path2 = 1 -> const_Land_Contour_Bnk <= 1
 ind_5372: t290_path2 = 1 -> x_Open_Porch_SF >= 75
 ind_5373: t290_path2 = 1 -> const_Lot_Area <= 14680
 ind_5374: t290_path3 = 1 -> const_Land_Contour_Bnk <= 1
 ind_5375: t290_path3 = 1 -> x_Open_Porch_SF >= 75
 ind_5376: t290_path3 = 1 -> const_Lot_Area >= 14680
 ind_5377: t290_path4 = 1 -> const_Land_Contour_Bnk >= 1
 ind_5378: t290_path4 = 1 -> x_Exter_Qual <= 3
 ind_5379: t290_path5 = 1 -> const_Land_Contour_Bnk >= 1
 ind_5380: t290_path5 = 1 -> x_Exter_Qual >= 3
 ind_5381: t291_path0 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_5382: t291_path0 = 1 -> const_Year_Remod_Add <= 2007
 ind_5383: t291_path0 = 1 -> const_Lot_Frontage <= 134
 ind_5384: t291_path1 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_5385: t291_path1 = 1 -> const_Year_Remod_Add <= 2007
 ind_5386: t291_path1 = 1 -> const_Lot_Frontage >= 134
 ind_5387: t291_path2 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_5388: t291_path2 = 1 -> const_Year_Remod_Add >= 2007
 ind_5389: t291_path2 = 1 -> const_Mo_Sold_10 <= 1
 ind_5390: t291_path3 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_5391: t291_path3 = 1 -> const_Year_Remod_Add >= 2007
 ind_5392: t291_path3 = 1 -> const_Mo_Sold_10 >= 1
 ind_5393: t291_path4 = 1 -> const_Total_Bsmt_SF >= 2492
 ind_5394: t292_path0 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_5395: t292_path0 = 1 -> x_Exter_Cond <= 2
 ind_5396: t292_path0 = 1 -> x_Open_Porch_SF <= 37
 ind_5397: t292_path1 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_5398: t292_path1 = 1 -> x_Exter_Cond <= 2
 ind_5399: t292_path1 = 1 -> x_Open_Porch_SF >= 37
 ind_5400: t292_path2 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_5401: t292_path2 = 1 -> x_Exter_Cond >= 2
 ind_5402: t292_path2 = 1 -> x_Pool_Area <= 368
 ind_5403: t292_path3 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_5404: t292_path3 = 1 -> x_Exter_Cond >= 2
 ind_5405: t292_path3 = 1 -> x_Pool_Area >= 368
 ind_5406: t292_path4 = 1 -> const_Neighborhood_ClearCr >= 1
 ind_5407: t293_path0 = 1 -> x_Gr_Liv_Area <= 2649
 ind_5408: t293_path0 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_5409: t293_path0 = 1 -> const_Bsmt_Full_Bath <= 1
 ind_5410: t293_path1 = 1 -> x_Gr_Liv_Area <= 2649
 ind_5411: t293_path1 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_5412: t293_path1 = 1 -> const_Bsmt_Full_Bath >= 1
 ind_5413: t293_path2 = 1 -> x_Gr_Liv_Area <= 2649
 ind_5414: t293_path2 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_5415: t293_path3 = 1 -> x_Gr_Liv_Area >= 2649
 ind_5416: t293_path3 = 1 -> x_Open_Porch_SF <= 31
 ind_5417: t293_path4 = 1 -> x_Gr_Liv_Area >= 2649
 ind_5418: t293_path4 = 1 -> x_Open_Porch_SF >= 31
 ind_5419: t294_path0 = 1 -> const_Sale_Condition_Family <= 1
 ind_5420: t294_path0 = 1 -> x_Open_Porch_SF <= 418
 ind_5421: t294_path0 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_5422: t294_path1 = 1 -> const_Sale_Condition_Family <= 1
 ind_5423: t294_path1 = 1 -> x_Open_Porch_SF <= 418
 ind_5424: t294_path1 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_5425: t294_path2 = 1 -> const_Sale_Condition_Family <= 1
 ind_5426: t294_path2 = 1 -> x_Open_Porch_SF >= 418
 ind_5427: t294_path3 = 1 -> const_Sale_Condition_Family >= 1
 ind_5428: t294_path3 = 1 -> x_Garage_Area <= 441
 ind_5429: t294_path4 = 1 -> const_Sale_Condition_Family >= 1
 ind_5430: t294_path4 = 1 -> x_Garage_Area >= 441
 ind_5431: t295_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_5432: t295_path0 = 1 -> x_Fireplace_Qu <= 2
 ind_5433: t295_path0 = 1 -> x_Bedroom_AbvGr <= 5
 ind_5434: t295_path1 = 1 -> const_Condition_1_Norm <= 1
 ind_5435: t295_path1 = 1 -> x_Fireplace_Qu <= 2
 ind_5436: t295_path1 = 1 -> x_Bedroom_AbvGr >= 5
 ind_5437: t295_path2 = 1 -> const_Condition_1_Norm <= 1
 ind_5438: t295_path2 = 1 -> x_Fireplace_Qu >= 2
 ind_5439: t295_path3 = 1 -> const_Condition_1_Norm >= 1
 ind_5440: t295_path3 = 1 -> const_Overall_Cond <= 6
 ind_5441: t295_path3 = 1 -> const_Garage_Yr_Blt <= 1923
 ind_5442: t295_path4 = 1 -> const_Condition_1_Norm >= 1
 ind_5443: t295_path4 = 1 -> const_Overall_Cond <= 6
 ind_5444: t295_path4 = 1 -> const_Garage_Yr_Blt >= 1923
 ind_5445: t295_path5 = 1 -> const_Condition_1_Norm >= 1
 ind_5446: t295_path5 = 1 -> const_Overall_Cond >= 6
 ind_5447: t296_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5448: t296_path0 = 1 -> x_1st_Flr_SF <= 2200
 ind_5449: t296_path0 = 1 -> const_Condition_1_Artery <= 1
 ind_5450: t296_path1 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5451: t296_path1 = 1 -> x_1st_Flr_SF <= 2200
 ind_5452: t296_path1 = 1 -> const_Condition_1_Artery >= 1
 ind_5453: t296_path2 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5454: t296_path2 = 1 -> x_1st_Flr_SF >= 2200
 ind_5455: t296_path3 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5456: t296_path3 = 1 -> x_Open_Porch_SF <= 238
 ind_5457: t296_path3 = 1 -> const_Lot_Config_Inside <= 1
 ind_5458: t296_path4 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5459: t296_path4 = 1 -> x_Open_Porch_SF <= 238
 ind_5460: t296_path4 = 1 -> const_Lot_Config_Inside >= 1
 ind_5461: t296_path5 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5462: t296_path5 = 1 -> x_Open_Porch_SF >= 238
 ind_5463: t297_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_5464: t297_path0 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_5465: t297_path0 = 1 -> const_Condition_1_PosA <= 1
 ind_5466: t297_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_5467: t297_path1 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_5468: t297_path1 = 1 -> const_Condition_1_PosA >= 1
 ind_5469: t297_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_5470: t297_path2 = 1 -> const_Lot_Config_CulDSac >= 1
 ind_5471: t297_path3 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_5472: t298_path0 = 1 -> x_Mas_Vnr_Area <= 922
 ind_5473: t298_path0 = 1 -> const_MS_SubClass_20 <= 1
 ind_5474: t298_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_5475: t298_path1 = 1 -> x_Mas_Vnr_Area <= 922
 ind_5476: t298_path1 = 1 -> const_MS_SubClass_20 <= 1
 ind_5477: t298_path1 = 1 -> const_Condition_1_Norm >= 1
 ind_5478: t298_path2 = 1 -> x_Mas_Vnr_Area <= 922
 ind_5479: t298_path2 = 1 -> const_MS_SubClass_20 >= 1
 ind_5480: t298_path2 = 1 -> x_Open_Porch_SF <= 91
 ind_5481: t298_path3 = 1 -> x_Mas_Vnr_Area <= 922
 ind_5482: t298_path3 = 1 -> const_MS_SubClass_20 >= 1
 ind_5483: t298_path3 = 1 -> x_Open_Porch_SF >= 91
 ind_5484: t298_path4 = 1 -> x_Mas_Vnr_Area >= 922
 ind_5485: t299_path0 = 1 -> const_Lot_Frontage <= 115
 ind_5486: t299_path0 = 1 -> const_Yr_Sold <= 2009
 ind_5487: t299_path0 = 1 -> const_Overall_Cond <= 4
 ind_5488: t299_path1 = 1 -> const_Lot_Frontage <= 115
 ind_5489: t299_path1 = 1 -> const_Yr_Sold <= 2009
 ind_5490: t299_path1 = 1 -> const_Overall_Cond >= 4
 ind_5491: t299_path2 = 1 -> const_Lot_Frontage <= 115
 ind_5492: t299_path2 = 1 -> const_Yr_Sold >= 2009
 ind_5493: t299_path2 = 1 -> const_Sale_Condition_Normal <= 1
 ind_5494: t299_path3 = 1 -> const_Lot_Frontage <= 115
 ind_5495: t299_path3 = 1 -> const_Yr_Sold >= 2009
 ind_5496: t299_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_5497: t299_path4 = 1 -> const_Lot_Frontage >= 115
 ind_5498: t299_path4 = 1 -> const_Land_Contour_Bnk <= 1
 ind_5499: t299_path4 = 1 -> x_1st_Flr_SF <= 1230
 ind_5500: t299_path5 = 1 -> const_Lot_Frontage >= 115
 ind_5501: t299_path5 = 1 -> const_Land_Contour_Bnk <= 1
 ind_5502: t299_path5 = 1 -> x_1st_Flr_SF >= 1230
 ind_5503: t299_path6 = 1 -> const_Lot_Frontage >= 115
 ind_5504: t299_path6 = 1 -> const_Land_Contour_Bnk >= 1
 ind_5505: t300_path0 = 1 -> const_Lot_Area <= 17541
 ind_5506: t300_path0 = 1 -> x_BsmtFin_SF_1 <= 1573
 ind_5507: t300_path0 = 1 -> const_MS_Zoning_FV <= 1
 ind_5508: t300_path1 = 1 -> const_Lot_Area <= 17541
 ind_5509: t300_path1 = 1 -> x_BsmtFin_SF_1 <= 1573
 ind_5510: t300_path1 = 1 -> const_MS_Zoning_FV >= 1
 ind_5511: t300_path2 = 1 -> const_Lot_Area <= 17541
 ind_5512: t300_path2 = 1 -> x_BsmtFin_SF_1 >= 1573
 ind_5513: t300_path3 = 1 -> const_Lot_Area >= 17541
 ind_5514: t300_path3 = 1 -> const_Garage_Type_Attchd <= 1
 ind_5515: t300_path4 = 1 -> const_Lot_Area >= 17541
 ind_5516: t300_path4 = 1 -> const_Garage_Type_Attchd >= 1
 ind_5517: t300_path4 = 1 -> const_Year_Remod_Add <= 2005
 ind_5518: t300_path5 = 1 -> const_Lot_Area >= 17541
 ind_5519: t300_path5 = 1 -> const_Garage_Type_Attchd >= 1
 ind_5520: t300_path5 = 1 -> const_Year_Remod_Add >= 2005
 ind_5521: t301_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_5522: t301_path0 = 1 -> const_Year_Built <= 1915
 ind_5523: t301_path0 = 1 -> x_1st_Flr_SF <= 813
 ind_5524: t301_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_5525: t301_path1 = 1 -> const_Year_Built <= 1915
 ind_5526: t301_path1 = 1 -> x_1st_Flr_SF >= 813
 ind_5527: t301_path2 = 1 -> x_1st_Flr_SF <= 2515
 ind_5528: t301_path2 = 1 -> const_Year_Built >= 1915
 ind_5529: t301_path2 = 1 -> const_Yr_Sold <= 2007
 ind_5530: t301_path3 = 1 -> x_1st_Flr_SF <= 2515
 ind_5531: t301_path3 = 1 -> const_Year_Built >= 1915
 ind_5532: t301_path3 = 1 -> const_Yr_Sold >= 2007
 ind_5533: t301_path4 = 1 -> x_1st_Flr_SF >= 2515
 ind_5534: t302_path0 = 1 -> x_1st_Flr_SF <= 2200
 ind_5535: t302_path0 = 1 -> const_Neighborhood_Somerst <= 1
 ind_5536: t302_path0 = 1 -> x_Bsmt_Unf_SF <= 108
 ind_5537: t302_path1 = 1 -> x_1st_Flr_SF <= 2200
 ind_5538: t302_path1 = 1 -> const_Neighborhood_Somerst <= 1
 ind_5539: t302_path1 = 1 -> x_Bsmt_Unf_SF >= 108
 ind_5540: t302_path2 = 1 -> x_1st_Flr_SF <= 2200
 ind_5541: t302_path2 = 1 -> const_Neighborhood_Somerst >= 1
 ind_5542: t302_path3 = 1 -> x_1st_Flr_SF >= 2200
 ind_5543: t303_path0 = 1 -> x_Mas_Vnr_Area <= 788
 ind_5544: t303_path0 = 1 -> Overall_Qual_calc <= 10
 ind_5545: t303_path0 = 1 -> x_Screen_Porch <= 110
 ind_5546: t303_path1 = 1 -> x_Mas_Vnr_Area <= 788
 ind_5547: t303_path1 = 1 -> Overall_Qual_calc <= 10
 ind_5548: t303_path1 = 1 -> x_Screen_Porch >= 110
 ind_5549: t303_path2 = 1 -> x_Mas_Vnr_Area <= 788
 ind_5550: t303_path2 = 1 -> Overall_Qual_calc >= 10
 ind_5551: t303_path3 = 1 -> x_Mas_Vnr_Area >= 788
 ind_5552: t303_path3 = 1 -> x_Gr_Liv_Area <= 2728
 ind_5553: t303_path4 = 1 -> x_Mas_Vnr_Area >= 788
 ind_5554: t303_path4 = 1 -> x_Gr_Liv_Area >= 2728
 ind_5555: t304_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5556: t304_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5557: t304_path0 = 1 -> const_Alley_No_aplica <= 1
 ind_5558: t304_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5559: t304_path1 = 1 -> const_Neighborhood_Edwards <= 1
 ind_5560: t304_path1 = 1 -> const_Alley_No_aplica >= 1
 ind_5561: t304_path2 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5562: t304_path2 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5563: t304_path2 = 1 -> x_Bsmt_Unf_SF <= 952
 ind_5564: t304_path3 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5565: t304_path3 = 1 -> const_Neighborhood_Edwards >= 1
 ind_5566: t304_path3 = 1 -> x_Bsmt_Unf_SF >= 952
 ind_5567: t304_path4 = 1 -> x_ex1_is_BrkFace >= 1
 ind_5568: t305_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_5569: t305_path0 = 1 -> const_Lot_Area <= 6000
 ind_5570: t305_path0 = 1 -> const_MS_Zoning_RM <= 1
 ind_5571: t305_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_5572: t305_path1 = 1 -> const_Lot_Area <= 6000
 ind_5573: t305_path1 = 1 -> const_MS_Zoning_RM >= 1
 ind_5574: t305_path2 = 1 -> x_1st_Flr_SF <= 2515
 ind_5575: t305_path2 = 1 -> const_Lot_Area >= 6000
 ind_5576: t305_path2 = 1 -> const_Neighborhood_SawyerW <= 1
 ind_5577: t305_path3 = 1 -> x_1st_Flr_SF <= 2515
 ind_5578: t305_path3 = 1 -> const_Lot_Area >= 6000
 ind_5579: t305_path3 = 1 -> const_Neighborhood_SawyerW >= 1
 ind_5580: t305_path4 = 1 -> x_1st_Flr_SF >= 2515
 ind_5581: t306_path0 = 1 -> Overall_Qual_calc <= 3
 ind_5582: t306_path0 = 1 -> const_Year_Remod_Add <= 1951
 ind_5583: t306_path1 = 1 -> Overall_Qual_calc <= 3
 ind_5584: t306_path1 = 1 -> const_Year_Remod_Add >= 1951
 ind_5585: t306_path2 = 1 -> Overall_Qual_calc >= 3
 ind_5586: t306_path2 = 1 -> const_Year_Built <= 1914
 ind_5587: t306_path2 = 1 -> x_1st_Flr_SF <= 813
 ind_5588: t306_path3 = 1 -> Overall_Qual_calc >= 3
 ind_5589: t306_path3 = 1 -> const_Year_Built <= 1914
 ind_5590: t306_path3 = 1 -> x_1st_Flr_SF >= 813
 ind_5591: t306_path4 = 1 -> Overall_Qual_calc >= 3
 ind_5592: t306_path4 = 1 -> const_Year_Built >= 1914
 ind_5593: t306_path4 = 1 -> central_air_yes <= 1
 ind_5594: t306_path5 = 1 -> Overall_Qual_calc >= 3
 ind_5595: t306_path5 = 1 -> const_Year_Built >= 1914
 ind_5596: t306_path5 = 1 -> central_air_yes >= 1
 ind_5597: t307_path0 = 1 -> Overall_Qual_calc <= 4
 ind_5598: t307_path0 = 1 -> const_Lot_Frontage <= 56
 ind_5599: t307_path1 = 1 -> Overall_Qual_calc <= 4
 ind_5600: t307_path1 = 1 -> const_Lot_Frontage >= 56
 ind_5601: t307_path2 = 1 -> Overall_Qual_calc >= 4
 ind_5602: t307_path2 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_5603: t307_path2 = 1 -> const_Year_Built <= 1915
 ind_5604: t307_path3 = 1 -> Overall_Qual_calc >= 4
 ind_5605: t307_path3 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_5606: t307_path3 = 1 -> const_Year_Built >= 1915
 ind_5607: t307_path4 = 1 -> Overall_Qual_calc >= 4
 ind_5608: t307_path4 = 1 -> const_Bsmt_Exposure_Av >= 1
 ind_5609: t307_path4 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_5610: t307_path5 = 1 -> Overall_Qual_calc >= 4
 ind_5611: t307_path5 = 1 -> const_Bsmt_Exposure_Av >= 1
 ind_5612: t307_path5 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_5613: t308_path0 = 1 -> const_MS_Zoning_FV <= 1
 ind_5614: t308_path0 = 1 -> Overall_Qual_calc <= 7
 ind_5615: t308_path0 = 1 -> x_Full_Bath <= 2
 ind_5616: t308_path1 = 1 -> const_MS_Zoning_FV <= 1
 ind_5617: t308_path1 = 1 -> Overall_Qual_calc <= 7
 ind_5618: t308_path1 = 1 -> x_Full_Bath >= 2
 ind_5619: t308_path2 = 1 -> const_MS_Zoning_FV <= 1
 ind_5620: t308_path2 = 1 -> Overall_Qual_calc >= 7
 ind_5621: t308_path2 = 1 -> const_Land_Slope_Gtl <= 1
 ind_5622: t308_path3 = 1 -> const_MS_Zoning_FV <= 1
 ind_5623: t308_path3 = 1 -> Overall_Qual_calc >= 7
 ind_5624: t308_path3 = 1 -> const_Land_Slope_Gtl >= 1
 ind_5625: t308_path4 = 1 -> const_MS_Zoning_FV >= 1
 ind_5626: t309_path0 = 1 -> x_TotRms_AbvGrd <= 12
 ind_5627: t309_path0 = 1 -> x_fence_is_GdWo <= 1
 ind_5628: t309_path0 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_5629: t309_path1 = 1 -> x_TotRms_AbvGrd <= 12
 ind_5630: t309_path1 = 1 -> x_fence_is_GdWo <= 1
 ind_5631: t309_path1 = 1 -> const_Total_Bsmt_SF >= 2492
 ind_5632: t309_path2 = 1 -> x_TotRms_AbvGrd <= 12
 ind_5633: t309_path2 = 1 -> x_fence_is_GdWo >= 1
 ind_5634: t309_path2 = 1 -> x_Enclosed_Porch <= 101
 ind_5635: t309_path3 = 1 -> x_TotRms_AbvGrd <= 12
 ind_5636: t309_path3 = 1 -> x_fence_is_GdWo >= 1
 ind_5637: t309_path3 = 1 -> x_Enclosed_Porch >= 101
 ind_5638: t309_path4 = 1 -> x_TotRms_AbvGrd >= 12
 ind_5639: t309_path4 = 1 -> const_2nd_Flr_SF <= 975
 ind_5640: t309_path5 = 1 -> x_TotRms_AbvGrd >= 12
 ind_5641: t309_path5 = 1 -> const_2nd_Flr_SF >= 975
 ind_5642: t310_path0 = 1 -> const_Street_Grvl <= 1
 ind_5643: t310_path0 = 1 -> x_Kitchen_Qual <= 2
 ind_5644: t310_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_5645: t310_path1 = 1 -> const_Street_Grvl <= 1
 ind_5646: t310_path1 = 1 -> x_Kitchen_Qual <= 2
 ind_5647: t310_path1 = 1 -> const_Condition_1_Norm >= 1
 ind_5648: t310_path2 = 1 -> const_Street_Grvl <= 1
 ind_5649: t310_path2 = 1 -> x_Kitchen_Qual >= 2
 ind_5650: t310_path2 = 1 -> const_Overall_Cond <= 4
 ind_5651: t310_path3 = 1 -> const_Street_Grvl <= 1
 ind_5652: t310_path3 = 1 -> x_Kitchen_Qual >= 2
 ind_5653: t310_path3 = 1 -> const_Overall_Cond >= 4
 ind_5654: t310_path4 = 1 -> const_Street_Grvl >= 1
 ind_5655: t311_path0 = 1 -> const_Year_Remod_Add <= 2008
 ind_5656: t311_path0 = 1 -> const_Overall_Cond <= 6
 ind_5657: t311_path0 = 1 -> x_1st_Flr_SF <= 1940
 ind_5658: t311_path1 = 1 -> const_Year_Remod_Add <= 2008
 ind_5659: t311_path1 = 1 -> const_Overall_Cond <= 6
 ind_5660: t311_path1 = 1 -> x_1st_Flr_SF >= 1940
 ind_5661: t311_path2 = 1 -> const_Year_Remod_Add <= 2008
 ind_5662: t311_path2 = 1 -> const_Overall_Cond >= 6
 ind_5663: t311_path3 = 1 -> const_Year_Remod_Add >= 2008
 ind_5664: t312_path0 = 1 -> x_Garage_Area <= 1154
 ind_5665: t312_path0 = 1 -> const_Foundation_PConc <= 1
 ind_5666: t312_path0 = 1 -> const_Year_Built <= 1925
 ind_5667: t312_path1 = 1 -> x_Garage_Area <= 1154
 ind_5668: t312_path1 = 1 -> const_Foundation_PConc <= 1
 ind_5669: t312_path1 = 1 -> const_Year_Built >= 1925
 ind_5670: t312_path2 = 1 -> x_Garage_Area <= 1154
 ind_5671: t312_path2 = 1 -> const_Foundation_PConc >= 1
 ind_5672: t312_path2 = 1 -> x_ex1_is_VinylSd <= 1
 ind_5673: t312_path3 = 1 -> x_Garage_Area <= 1154
 ind_5674: t312_path3 = 1 -> const_Foundation_PConc >= 1
 ind_5675: t312_path3 = 1 -> x_ex1_is_VinylSd >= 1
 ind_5676: t312_path4 = 1 -> x_Garage_Area >= 1154
 ind_5677: t313_path0 = 1 -> x_Exter_Cond <= 2
 ind_5678: t313_path0 = 1 -> const_Lot_Area <= 9171
 ind_5679: t313_path0 = 1 -> const_Year_Built <= 1931
 ind_5680: t313_path1 = 1 -> x_Exter_Cond <= 2
 ind_5681: t313_path1 = 1 -> const_Lot_Area <= 9171
 ind_5682: t313_path1 = 1 -> const_Year_Built >= 1931
 ind_5683: t313_path2 = 1 -> x_Exter_Cond <= 2
 ind_5684: t313_path2 = 1 -> const_Lot_Area >= 9171
 ind_5685: t313_path2 = 1 -> x_Open_Porch_SF <= 4
 ind_5686: t313_path3 = 1 -> x_Exter_Cond <= 2
 ind_5687: t313_path3 = 1 -> const_Lot_Area >= 9171
 ind_5688: t313_path3 = 1 -> x_Open_Porch_SF >= 4
 ind_5689: t313_path4 = 1 -> x_Exter_Cond >= 2
 ind_5690: t313_path4 = 1 -> const_Land_Slope_Gtl <= 1
 ind_5691: t313_path5 = 1 -> x_Exter_Cond >= 2
 ind_5692: t313_path5 = 1 -> const_Land_Slope_Gtl >= 1
 ind_5693: t313_path5 = 1 -> const_Lot_Area <= 35133
 ind_5694: t313_path6 = 1 -> x_Exter_Cond >= 2
 ind_5695: t313_path6 = 1 -> const_Land_Slope_Gtl >= 1
 ind_5696: t313_path6 = 1 -> const_Lot_Area >= 35133
 ind_5697: t314_path0 = 1 -> const_Garage_Cars <= 3
 ind_5698: t314_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_5699: t314_path0 = 1 -> const_Year_Built <= 1942
 ind_5700: t314_path1 = 1 -> const_Garage_Cars <= 3
 ind_5701: t314_path1 = 1 -> const_Condition_1_Norm <= 1
 ind_5702: t314_path1 = 1 -> const_Year_Built >= 1942
 ind_5703: t314_path2 = 1 -> const_Garage_Cars <= 3
 ind_5704: t314_path2 = 1 -> const_Condition_1_Norm >= 1
 ind_5705: t314_path2 = 1 -> const_MS_SubClass_20 <= 1
 ind_5706: t314_path3 = 1 -> const_Garage_Cars <= 3
 ind_5707: t314_path3 = 1 -> const_Condition_1_Norm >= 1
 ind_5708: t314_path3 = 1 -> const_MS_SubClass_20 >= 1
 ind_5709: t314_path4 = 1 -> const_Garage_Cars >= 3
 ind_5710: t314_path4 = 1 -> Overall_Qual_calc <= 10
 ind_5711: t314_path4 = 1 -> const_Mo_Sold_2 <= 1
 ind_5712: t314_path5 = 1 -> const_Garage_Cars >= 3
 ind_5713: t314_path5 = 1 -> Overall_Qual_calc <= 10
 ind_5714: t314_path5 = 1 -> const_Mo_Sold_2 >= 1
 ind_5715: t314_path6 = 1 -> const_Garage_Cars >= 3
 ind_5716: t314_path6 = 1 -> Overall_Qual_calc >= 10
 ind_5717: t315_path0 = 1 -> const_Land_Slope_Gtl <= 1
 ind_5718: t315_path0 = 1 -> const_Garage_Finish_RFn <= 1
 ind_5719: t315_path1 = 1 -> const_Land_Slope_Gtl <= 1
 ind_5720: t315_path1 = 1 -> const_Garage_Finish_RFn >= 1
 ind_5721: t315_path2 = 1 -> const_Land_Slope_Gtl >= 1
 ind_5722: t315_path2 = 1 -> x_Gr_Liv_Area <= 3086
 ind_5723: t315_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_5724: t315_path3 = 1 -> const_Land_Slope_Gtl >= 1
 ind_5725: t315_path3 = 1 -> x_Gr_Liv_Area <= 3086
 ind_5726: t315_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_5727: t315_path4 = 1 -> const_Land_Slope_Gtl >= 1
 ind_5728: t315_path4 = 1 -> x_Gr_Liv_Area >= 3086
 ind_5729: t316_path0 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_5730: t316_path0 = 1 -> const_Sale_Condition_Family <= 1
 ind_5731: t316_path0 = 1 -> x_Exter_Cond <= 2
 ind_5732: t316_path1 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_5733: t316_path1 = 1 -> const_Sale_Condition_Family <= 1
 ind_5734: t316_path1 = 1 -> x_Exter_Cond >= 2
 ind_5735: t316_path2 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_5736: t316_path2 = 1 -> const_Sale_Condition_Family >= 1
 ind_5737: t316_path2 = 1 -> x_Mas_Vnr_Area <= 192
 ind_5738: t316_path3 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_5739: t316_path3 = 1 -> const_Sale_Condition_Family >= 1
 ind_5740: t316_path3 = 1 -> x_Mas_Vnr_Area >= 192
 ind_5741: t316_path4 = 1 -> const_Bsmt_Exposure_Av >= 1
 ind_5742: t317_path0 = 1 -> const_Misc_Val <= 500
 ind_5743: t317_path0 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_5744: t317_path0 = 1 -> const_Bsmt_Exposure_No <= 1
 ind_5745: t317_path1 = 1 -> const_Misc_Val <= 500
 ind_5746: t317_path1 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_5747: t317_path1 = 1 -> const_Bsmt_Exposure_No >= 1
 ind_5748: t317_path2 = 1 -> const_Misc_Val <= 500
 ind_5749: t317_path2 = 1 -> const_Neighborhood_BrkSide >= 1
 ind_5750: t317_path3 = 1 -> const_Misc_Val >= 500
 ind_5751: t318_path0 = 1 -> const_Street_Pave <= 1
 ind_5752: t318_path1 = 1 -> const_Street_Pave >= 1
 ind_5753: t318_path1 = 1 -> x_Wood_Deck_SF <= 511
 ind_5754: t318_path1 = 1 -> x_Mas_Vnr_Area <= 459
 ind_5755: t318_path2 = 1 -> const_Street_Pave >= 1
 ind_5756: t318_path2 = 1 -> x_Wood_Deck_SF <= 511
 ind_5757: t318_path2 = 1 -> x_Mas_Vnr_Area >= 459
 ind_5758: t318_path3 = 1 -> const_Street_Pave >= 1
 ind_5759: t318_path3 = 1 -> x_Wood_Deck_SF >= 511
 ind_5760: t319_path0 = 1 -> const_Yr_Sold <= 2009
 ind_5761: t319_path0 = 1 -> x_Wood_Deck_SF <= 467
 ind_5762: t319_path1 = 1 -> const_Yr_Sold <= 2009
 ind_5763: t319_path1 = 1 -> x_Wood_Deck_SF >= 467
 ind_5764: t319_path2 = 1 -> const_Yr_Sold >= 2009
 ind_5765: t319_path2 = 1 -> const_Lot_Frontage <= 85
 ind_5766: t319_path2 = 1 -> x_fence_is_GdWo <= 1
 ind_5767: t319_path3 = 1 -> const_Yr_Sold >= 2009
 ind_5768: t319_path3 = 1 -> const_Lot_Frontage <= 85
 ind_5769: t319_path3 = 1 -> x_fence_is_GdWo >= 1
 ind_5770: t319_path4 = 1 -> const_Yr_Sold >= 2009
 ind_5771: t319_path4 = 1 -> const_Lot_Frontage >= 85
 ind_5772: t320_path0 = 1 -> x_Garage_Area <= 630
 ind_5773: t320_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_5774: t320_path0 = 1 -> x_Heating_QC <= 3
 ind_5775: t320_path1 = 1 -> x_Garage_Area <= 630
 ind_5776: t320_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_5777: t320_path1 = 1 -> x_Heating_QC >= 3
 ind_5778: t320_path2 = 1 -> x_Garage_Area <= 630
 ind_5779: t320_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_5780: t320_path2 = 1 -> x_Bsmt_Cond <= 3
 ind_5781: t320_path3 = 1 -> x_Garage_Area <= 630
 ind_5782: t320_path3 = 1 -> x_Gr_Liv_Area >= 747
 ind_5783: t320_path3 = 1 -> x_Bsmt_Cond >= 3
 ind_5784: t320_path4 = 1 -> x_Garage_Area >= 630
 ind_5785: t320_path4 = 1 -> const_Overall_Cond <= 4
 ind_5786: t320_path5 = 1 -> x_Garage_Area >= 630
 ind_5787: t320_path5 = 1 -> const_Overall_Cond >= 4
 ind_5788: t320_path5 = 1 -> x_Garage_Area <= 1154
 ind_5789: t320_path6 = 1 -> x_Garage_Area >= 630
 ind_5790: t320_path6 = 1 -> const_Overall_Cond >= 4
 ind_5791: t320_path6 = 1 -> x_Garage_Area >= 1154
 ind_5792: t321_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_5793: t321_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5794: t321_path0 = 1 -> const_Overall_Cond <= 7
 ind_5795: t321_path1 = 1 -> x_ex2_is_AsbShng <= 1
 ind_5796: t321_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5797: t321_path1 = 1 -> const_Overall_Cond >= 7
 ind_5798: t321_path2 = 1 -> x_ex2_is_AsbShng <= 1
 ind_5799: t321_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5800: t321_path3 = 1 -> x_ex2_is_AsbShng >= 1
 ind_5801: t321_path3 = 1 -> const_Total_Bsmt_SF <= 721
 ind_5802: t321_path4 = 1 -> x_ex2_is_AsbShng >= 1
 ind_5803: t321_path4 = 1 -> const_Total_Bsmt_SF >= 721
 ind_5804: t322_path0 = 1 -> central_air_yes <= 1
 ind_5805: t322_path0 = 1 -> const_Overall_Cond <= 5
 ind_5806: t322_path0 = 1 -> x_Bsmt_Unf_SF <= 727
 ind_5807: t322_path1 = 1 -> central_air_yes <= 1
 ind_5808: t322_path1 = 1 -> const_Overall_Cond <= 5
 ind_5809: t322_path1 = 1 -> x_Bsmt_Unf_SF >= 727
 ind_5810: t322_path2 = 1 -> central_air_yes <= 1
 ind_5811: t322_path2 = 1 -> const_Overall_Cond >= 5
 ind_5812: t322_path2 = 1 -> x_Garage_Qual <= 2
 ind_5813: t322_path3 = 1 -> central_air_yes <= 1
 ind_5814: t322_path3 = 1 -> const_Overall_Cond >= 5
 ind_5815: t322_path3 = 1 -> x_Garage_Qual >= 2
 ind_5816: t322_path4 = 1 -> central_air_yes >= 1
 ind_5817: t322_path4 = 1 -> x_Screen_Porch <= 150
 ind_5818: t322_path4 = 1 -> const_Overall_Cond <= 5
 ind_5819: t322_path5 = 1 -> central_air_yes >= 1
 ind_5820: t322_path5 = 1 -> x_Screen_Porch <= 150
 ind_5821: t322_path5 = 1 -> const_Overall_Cond >= 5
 ind_5822: t322_path6 = 1 -> central_air_yes >= 1
 ind_5823: t322_path6 = 1 -> x_Screen_Porch >= 150
 ind_5824: t323_path0 = 1 -> const_MS_SubClass_70 <= 1
 ind_5825: t323_path0 = 1 -> const_Roof_Style_Flat <= 1
 ind_5826: t323_path0 = 1 -> x_b1_is_GLQ <= 1
 ind_5827: t323_path1 = 1 -> const_MS_SubClass_70 <= 1
 ind_5828: t323_path1 = 1 -> const_Roof_Style_Flat <= 1
 ind_5829: t323_path1 = 1 -> x_b1_is_GLQ >= 1
 ind_5830: t323_path2 = 1 -> const_MS_SubClass_70 <= 1
 ind_5831: t323_path2 = 1 -> const_Roof_Style_Flat >= 1
 ind_5832: t323_path3 = 1 -> const_MS_SubClass_70 >= 1
 ind_5833: t324_path0 = 1 -> const_Yr_Sold <= 2008
 ind_5834: t324_path0 = 1 -> const_Sale_Type_COD <= 1
 ind_5835: t324_path0 = 1 -> x_elect_is_FuseA <= 1
 ind_5836: t324_path1 = 1 -> const_Yr_Sold <= 2008
 ind_5837: t324_path1 = 1 -> const_Sale_Type_COD <= 1
 ind_5838: t324_path1 = 1 -> x_elect_is_FuseA >= 1
 ind_5839: t324_path2 = 1 -> const_Yr_Sold <= 2008
 ind_5840: t324_path2 = 1 -> const_Sale_Type_COD >= 1
 ind_5841: t324_path3 = 1 -> const_Yr_Sold >= 2008
 ind_5842: t324_path3 = 1 -> const_Year_Built <= 1925
 ind_5843: t324_path3 = 1 -> const_Year_Remod_Add <= 1997
 ind_5844: t324_path4 = 1 -> const_Yr_Sold >= 2008
 ind_5845: t324_path4 = 1 -> const_Year_Built <= 1925
 ind_5846: t324_path4 = 1 -> const_Year_Remod_Add >= 1997
 ind_5847: t324_path5 = 1 -> const_Yr_Sold >= 2008
 ind_5848: t324_path5 = 1 -> const_Year_Built >= 1925
 ind_5849: t324_path5 = 1 -> x_b2_is_BLQ <= 1
 ind_5850: t324_path6 = 1 -> const_Yr_Sold >= 2008
 ind_5851: t324_path6 = 1 -> const_Year_Built >= 1925
 ind_5852: t324_path6 = 1 -> x_b2_is_BLQ >= 1
 ind_5853: t325_path0 = 1 -> x_Garage_Area <= 895
 ind_5854: t325_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5855: t325_path0 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_5856: t325_path1 = 1 -> x_Garage_Area <= 895
 ind_5857: t325_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5858: t325_path1 = 1 -> x_Bsmt_Unf_SF >= 476
 ind_5859: t325_path2 = 1 -> x_Garage_Area <= 895
 ind_5860: t325_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5861: t325_path2 = 1 -> x_Bsmt_Unf_SF <= 507
 ind_5862: t325_path3 = 1 -> x_Garage_Area <= 895
 ind_5863: t325_path3 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5864: t325_path3 = 1 -> x_Bsmt_Unf_SF >= 507
 ind_5865: t325_path4 = 1 -> x_Garage_Area >= 895
 ind_5866: t326_path0 = 1 -> const_Condition_1_PosN <= 1
 ind_5867: t326_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5868: t326_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_5869: t326_path1 = 1 -> const_Condition_1_PosN <= 1
 ind_5870: t326_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_5871: t326_path1 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_5872: t326_path2 = 1 -> const_Condition_1_PosN <= 1
 ind_5873: t326_path2 = 1 -> x_ex1_is_BrkFace >= 1
 ind_5874: t326_path3 = 1 -> const_Condition_1_PosN >= 1
 ind_5875: t327_path0 = 1 -> const_Neighborhood_Somerst <= 1
 ind_5876: t327_path0 = 1 -> const_Overall_Cond <= 6
 ind_5877: t327_path0 = 1 -> const_Garage_Yr_Blt <= 1966
 ind_5878: t327_path1 = 1 -> const_Neighborhood_Somerst <= 1
 ind_5879: t327_path1 = 1 -> const_Overall_Cond <= 6
 ind_5880: t327_path1 = 1 -> const_Garage_Yr_Blt >= 1966
 ind_5881: t327_path2 = 1 -> const_Neighborhood_Somerst <= 1
 ind_5882: t327_path2 = 1 -> const_Overall_Cond >= 6
 ind_5883: t327_path2 = 1 -> x_1st_Flr_SF <= 1414
 ind_5884: t327_path3 = 1 -> const_Neighborhood_Somerst <= 1
 ind_5885: t327_path3 = 1 -> const_Overall_Cond >= 6
 ind_5886: t327_path3 = 1 -> x_1st_Flr_SF >= 1414
 ind_5887: t327_path4 = 1 -> const_Neighborhood_Somerst >= 1
 ind_5888: t328_path0 = 1 -> x_Heating_QC <= 2
 ind_5889: t328_path0 = 1 -> x_ex2_is_VinylSd <= 1
 ind_5890: t328_path1 = 1 -> x_Heating_QC <= 2
 ind_5891: t328_path1 = 1 -> x_ex2_is_VinylSd >= 1
 ind_5892: t328_path2 = 1 -> x_Heating_QC >= 2
 ind_5893: t328_path2 = 1 -> central_air_yes <= 1
 ind_5894: t328_path2 = 1 -> x_b1_is_Unf <= 1
 ind_5895: t328_path3 = 1 -> x_Heating_QC >= 2
 ind_5896: t328_path3 = 1 -> central_air_yes <= 1
 ind_5897: t328_path3 = 1 -> x_b1_is_Unf >= 1
 ind_5898: t328_path4 = 1 -> x_Heating_QC >= 2
 ind_5899: t328_path4 = 1 -> central_air_yes >= 1
 ind_5900: t328_path4 = 1 -> const_Exterior_2nd_Wd_Shng <= 1
 ind_5901: t328_path5 = 1 -> x_Heating_QC >= 2
 ind_5902: t328_path5 = 1 -> central_air_yes >= 1
 ind_5903: t328_path5 = 1 -> const_Exterior_2nd_Wd_Shng >= 1
 ind_5904: t329_path0 = 1 -> Overall_Qual_calc <= 7
 ind_5905: t329_path0 = 1 -> x_Full_Bath <= 2
 ind_5906: t329_path0 = 1 -> const_Total_Bsmt_SF <= 1389
 ind_5907: t329_path1 = 1 -> Overall_Qual_calc <= 7
 ind_5908: t329_path1 = 1 -> x_Full_Bath <= 2
 ind_5909: t329_path1 = 1 -> const_Total_Bsmt_SF >= 1389
 ind_5910: t329_path2 = 1 -> Overall_Qual_calc <= 7
 ind_5911: t329_path2 = 1 -> x_Full_Bath >= 2
 ind_5912: t329_path2 = 1 -> x_Gr_Liv_Area <= 1017
 ind_5913: t329_path3 = 1 -> Overall_Qual_calc <= 7
 ind_5914: t329_path3 = 1 -> x_Full_Bath >= 2
 ind_5915: t329_path3 = 1 -> x_Gr_Liv_Area >= 1017
 ind_5916: t329_path4 = 1 -> Overall_Qual_calc >= 7
 ind_5917: t329_path4 = 1 -> const_MS_SubClass_120 <= 1
 ind_5918: t329_path4 = 1 -> const_Lot_Frontage <= 111
 ind_5919: t329_path5 = 1 -> Overall_Qual_calc >= 7
 ind_5920: t329_path5 = 1 -> const_MS_SubClass_120 <= 1
 ind_5921: t329_path5 = 1 -> const_Lot_Frontage >= 111
 ind_5922: t329_path6 = 1 -> Overall_Qual_calc >= 7
 ind_5923: t329_path6 = 1 -> const_MS_SubClass_120 >= 1
 ind_5924: t330_path0 = 1 -> const_Lot_Area <= 18386
 ind_5925: t330_path0 = 1 -> x_Kitchen_Qual <= 2
 ind_5926: t330_path0 = 1 -> const_Lot_Area <= 8520
 ind_5927: t330_path1 = 1 -> const_Lot_Area <= 18386
 ind_5928: t330_path1 = 1 -> x_Kitchen_Qual <= 2
 ind_5929: t330_path1 = 1 -> const_Lot_Area >= 8520
 ind_5930: t330_path2 = 1 -> const_Lot_Area <= 18386
 ind_5931: t330_path2 = 1 -> x_Kitchen_Qual >= 2
 ind_5932: t330_path2 = 1 -> const_MS_Zoning_FV <= 1
 ind_5933: t330_path3 = 1 -> const_Lot_Area <= 18386
 ind_5934: t330_path3 = 1 -> x_Kitchen_Qual >= 2
 ind_5935: t330_path3 = 1 -> const_MS_Zoning_FV >= 1
 ind_5936: t330_path4 = 1 -> const_Lot_Area >= 18386
 ind_5937: t330_path4 = 1 -> x_Enclosed_Porch <= 109
 ind_5938: t330_path5 = 1 -> const_Lot_Area >= 18386
 ind_5939: t330_path5 = 1 -> x_Enclosed_Porch >= 109
 ind_5940: t331_path0 = 1 -> const_Year_Remod_Add <= 2007
 ind_5941: t331_path0 = 1 -> const_Lot_Frontage <= 130
 ind_5942: t331_path1 = 1 -> const_Year_Remod_Add <= 2007
 ind_5943: t331_path1 = 1 -> const_Lot_Frontage >= 130
 ind_5944: t331_path2 = 1 -> const_Year_Remod_Add >= 2007
 ind_5945: t331_path2 = 1 -> const_Lot_Frontage <= 114
 ind_5946: t331_path2 = 1 -> x_ex2_is_Wd_Sdng <= 1
 ind_5947: t331_path3 = 1 -> const_Year_Remod_Add >= 2007
 ind_5948: t331_path3 = 1 -> const_Lot_Frontage <= 114
 ind_5949: t331_path3 = 1 -> x_ex2_is_Wd_Sdng >= 1
 ind_5950: t331_path4 = 1 -> const_Year_Remod_Add >= 2007
 ind_5951: t331_path4 = 1 -> const_Lot_Frontage >= 114
 ind_5952: t332_path0 = 1 -> x_Garage_Area <= 1154
 ind_5953: t332_path0 = 1 -> const_Lot_Area <= 4435
 ind_5954: t332_path1 = 1 -> x_Garage_Area <= 1154
 ind_5955: t332_path1 = 1 -> const_Lot_Area >= 4435
 ind_5956: t332_path1 = 1 -> const_Year_Remod_Add <= 2007
 ind_5957: t332_path2 = 1 -> x_Garage_Area <= 1154
 ind_5958: t332_path2 = 1 -> const_Lot_Area >= 4435
 ind_5959: t332_path2 = 1 -> const_Year_Remod_Add >= 2007
 ind_5960: t332_path3 = 1 -> x_Garage_Area >= 1154
 ind_5961: t333_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5962: t333_path1 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5963: t333_path1 = 1 -> x_BsmtFin_SF_1 <= 192
 ind_5964: t333_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5965: t333_path2 = 1 -> x_BsmtFin_SF_1 >= 192
 ind_5966: t334_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5967: t334_path0 = 1 -> const_Overall_Cond <= 6
 ind_5968: t334_path0 = 1 -> const_Land_Contour_Bnk <= 1
 ind_5969: t334_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5970: t334_path1 = 1 -> const_Overall_Cond <= 6
 ind_5971: t334_path1 = 1 -> const_Land_Contour_Bnk >= 1
 ind_5972: t334_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5973: t334_path2 = 1 -> const_Overall_Cond >= 6
 ind_5974: t334_path2 = 1 -> const_2nd_Flr_SF <= 376
 ind_5975: t334_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_5976: t334_path3 = 1 -> const_Overall_Cond >= 6
 ind_5977: t334_path3 = 1 -> const_2nd_Flr_SF >= 376
 ind_5978: t334_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5979: t334_path4 = 1 -> const_Garage_Yr_Blt <= 1921
 ind_5980: t334_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5981: t334_path5 = 1 -> const_Garage_Yr_Blt >= 1921
 ind_5982: t334_path5 = 1 -> const_Lot_Area <= 9605
 ind_5983: t334_path6 = 1 -> x_ex1_is_AsbShng >= 1
 ind_5984: t334_path6 = 1 -> const_Garage_Yr_Blt >= 1921
 ind_5985: t334_path6 = 1 -> const_Lot_Area >= 9605
 ind_5986: t335_path0 = 1 -> const_Mo_Sold_7 <= 1
 ind_5987: t335_path0 = 1 -> x_Half_Bath <= 1
 ind_5988: t335_path0 = 1 -> x_Bsmt_Cond <= 3
 ind_5989: t335_path1 = 1 -> const_Mo_Sold_7 <= 1
 ind_5990: t335_path1 = 1 -> x_Half_Bath <= 1
 ind_5991: t335_path1 = 1 -> x_Bsmt_Cond >= 3
 ind_5992: t335_path2 = 1 -> const_Mo_Sold_7 <= 1
 ind_5993: t335_path2 = 1 -> x_Half_Bath >= 1
 ind_5994: t335_path2 = 1 -> Overall_Qual_calc <= 7
 ind_5995: t335_path3 = 1 -> const_Mo_Sold_7 <= 1
 ind_5996: t335_path3 = 1 -> x_Half_Bath >= 1
 ind_5997: t335_path3 = 1 -> Overall_Qual_calc >= 7
 ind_5998: t335_path4 = 1 -> const_Mo_Sold_7 >= 1
 ind_5999: t335_path4 = 1 -> const_Lot_Area <= 11428
 ind_6000: t335_path5 = 1 -> const_Mo_Sold_7 >= 1
 ind_6001: t335_path5 = 1 -> const_Lot_Area >= 11428
 ind_6002: t336_path0 = 1 -> const_Lot_Frontage <= 85
 ind_6003: t336_path0 = 1 -> x_b1_is_GLQ <= 1
 ind_6004: t336_path0 = 1 -> x_Garage_Cond <= 1
 ind_6005: t336_path1 = 1 -> const_Lot_Frontage <= 85
 ind_6006: t336_path1 = 1 -> x_b1_is_GLQ <= 1
 ind_6007: t336_path1 = 1 -> x_Garage_Cond >= 1
 ind_6008: t336_path2 = 1 -> const_Lot_Frontage <= 85
 ind_6009: t336_path2 = 1 -> x_b1_is_GLQ >= 1
 ind_6010: t336_path2 = 1 -> const_Lot_Area <= 12435
 ind_6011: t336_path3 = 1 -> const_Lot_Frontage <= 85
 ind_6012: t336_path3 = 1 -> x_b1_is_GLQ >= 1
 ind_6013: t336_path3 = 1 -> const_Lot_Area >= 12435
 ind_6014: t336_path4 = 1 -> const_Lot_Frontage >= 85
 ind_6015: t336_path4 = 1 -> const_Roof_Style_Gable <= 1
 ind_6016: t336_path4 = 1 -> const_Land_Slope_Gtl <= 1
 ind_6017: t336_path5 = 1 -> const_Lot_Frontage >= 85
 ind_6018: t336_path5 = 1 -> const_Roof_Style_Gable <= 1
 ind_6019: t336_path5 = 1 -> const_Land_Slope_Gtl >= 1
 ind_6020: t336_path6 = 1 -> const_Lot_Frontage >= 85
 ind_6021: t336_path6 = 1 -> const_Roof_Style_Gable >= 1
 ind_6022: t336_path6 = 1 -> const_Lot_Area <= 18386
 ind_6023: t336_path7 = 1 -> const_Lot_Frontage >= 85
 ind_6024: t336_path7 = 1 -> const_Roof_Style_Gable >= 1
 ind_6025: t336_path7 = 1 -> const_Lot_Area >= 18386
 ind_6026: t337_path0 = 1 -> x_b1_is_ALQ <= 1
 ind_6027: t337_path0 = 1 -> const_Year_Built <= 1996
 ind_6028: t337_path0 = 1 -> x_Bsmt_Unf_SF <= 817
 ind_6029: t337_path1 = 1 -> x_b1_is_ALQ <= 1
 ind_6030: t337_path1 = 1 -> const_Year_Built <= 1996
 ind_6031: t337_path1 = 1 -> x_Bsmt_Unf_SF >= 817
 ind_6032: t337_path2 = 1 -> x_b1_is_ALQ <= 1
 ind_6033: t337_path2 = 1 -> const_Year_Built >= 1996
 ind_6034: t337_path2 = 1 -> x_Exter_Qual <= 3
 ind_6035: t337_path3 = 1 -> x_b1_is_ALQ <= 1
 ind_6036: t337_path3 = 1 -> const_Year_Built >= 1996
 ind_6037: t337_path3 = 1 -> x_Exter_Qual >= 3
 ind_6038: t337_path4 = 1 -> x_b1_is_ALQ >= 1
 ind_6039: t337_path4 = 1 -> const_Lot_Config_Inside <= 1
 ind_6040: t337_path5 = 1 -> x_b1_is_ALQ >= 1
 ind_6041: t337_path5 = 1 -> const_Lot_Config_Inside >= 1
 ind_6042: t338_path0 = 1 -> x_Fireplace_Qu <= 2
 ind_6043: t338_path0 = 1 -> x_Exter_Cond <= 2
 ind_6044: t338_path0 = 1 -> x_ex1_is_Wd_Sdng <= 1
 ind_6045: t338_path1 = 1 -> x_Fireplace_Qu <= 2
 ind_6046: t338_path1 = 1 -> x_Exter_Cond <= 2
 ind_6047: t338_path1 = 1 -> x_ex1_is_Wd_Sdng >= 1
 ind_6048: t338_path2 = 1 -> x_Fireplace_Qu <= 2
 ind_6049: t338_path2 = 1 -> x_Exter_Cond >= 2
 ind_6050: t338_path2 = 1 -> const_Bsmt_Half_Bath <= 1
 ind_6051: t338_path3 = 1 -> x_Fireplace_Qu <= 2
 ind_6052: t338_path3 = 1 -> x_Exter_Cond >= 2
 ind_6053: t338_path3 = 1 -> const_Bsmt_Half_Bath >= 1
 ind_6054: t338_path4 = 1 -> x_Fireplace_Qu >= 2
 ind_6055: t338_path4 = 1 -> x_Gr_Liv_Area <= 876
 ind_6056: t338_path5 = 1 -> x_Fireplace_Qu >= 2
 ind_6057: t338_path5 = 1 -> x_Gr_Liv_Area >= 876
 ind_6058: t338_path5 = 1 -> x_Bsmt_Unf_SF <= 348
 ind_6059: t338_path6 = 1 -> x_Fireplace_Qu >= 2
 ind_6060: t338_path6 = 1 -> x_Gr_Liv_Area >= 876
 ind_6061: t338_path6 = 1 -> x_Bsmt_Unf_SF >= 348
 ind_6062: t339_path0 = 1 -> const_Sale_Condition_Family <= 1
 ind_6063: t339_path0 = 1 -> const_Lot_Area <= 19522
 ind_6064: t339_path0 = 1 -> const_Neighborhood_OldTown <= 1
 ind_6065: t339_path1 = 1 -> const_Sale_Condition_Family <= 1
 ind_6066: t339_path1 = 1 -> const_Lot_Area <= 19522
 ind_6067: t339_path1 = 1 -> const_Neighborhood_OldTown >= 1
 ind_6068: t339_path2 = 1 -> const_Sale_Condition_Family <= 1
 ind_6069: t339_path2 = 1 -> const_Lot_Area >= 19522
 ind_6070: t339_path2 = 1 -> const_Land_Contour_Bnk <= 1
 ind_6071: t339_path3 = 1 -> const_Sale_Condition_Family <= 1
 ind_6072: t339_path3 = 1 -> const_Lot_Area >= 19522
 ind_6073: t339_path3 = 1 -> const_Land_Contour_Bnk >= 1
 ind_6074: t339_path4 = 1 -> const_Sale_Condition_Family >= 1
 ind_6075: t339_path4 = 1 -> x_Mas_Vnr_Area <= 22
 ind_6076: t339_path5 = 1 -> const_Sale_Condition_Family >= 1
 ind_6077: t339_path5 = 1 -> x_Mas_Vnr_Area >= 22
 ind_6078: t340_path0 = 1 -> const_Year_Built <= 1952
 ind_6079: t340_path0 = 1 -> const_Year_Built <= 1915
 ind_6080: t340_path0 = 1 -> const_Bsmt_Qual <= 2
 ind_6081: t340_path1 = 1 -> const_Year_Built <= 1952
 ind_6082: t340_path1 = 1 -> const_Year_Built <= 1915
 ind_6083: t340_path1 = 1 -> const_Bsmt_Qual >= 2
 ind_6084: t340_path2 = 1 -> const_Year_Built <= 1952
 ind_6085: t340_path2 = 1 -> const_Year_Built >= 1915
 ind_6086: t340_path2 = 1 -> const_Condition_1_Artery <= 1
 ind_6087: t340_path3 = 1 -> const_Year_Built <= 1952
 ind_6088: t340_path3 = 1 -> const_Year_Built >= 1915
 ind_6089: t340_path3 = 1 -> const_Condition_1_Artery >= 1
 ind_6090: t340_path4 = 1 -> const_Year_Built >= 1952
 ind_6091: t340_path4 = 1 -> const_Year_Remod_Add <= 1953
 ind_6092: t340_path5 = 1 -> const_Year_Built >= 1952
 ind_6093: t340_path5 = 1 -> const_Year_Remod_Add >= 1953
 ind_6094: t340_path5 = 1 -> const_Sale_Condition_Family <= 1
 ind_6095: t340_path6 = 1 -> const_Year_Built >= 1952
 ind_6096: t340_path6 = 1 -> const_Year_Remod_Add >= 1953
 ind_6097: t340_path6 = 1 -> const_Sale_Condition_Family >= 1
 ind_6098: t341_path0 = 1 -> const_Sale_Condition_Family <= 1
 ind_6099: t341_path1 = 1 -> const_Sale_Condition_Family >= 1
 ind_6100: t341_path1 = 1 -> x_Garage_Area <= 506
 ind_6101: t341_path2 = 1 -> const_Sale_Condition_Family >= 1
 ind_6102: t341_path2 = 1 -> x_Garage_Area >= 506
 ind_6103: t342_path0 = 1 -> const_Alley_No_aplica <= 1
 ind_6104: t342_path1 = 1 -> const_Alley_No_aplica >= 1
 ind_6105: t342_path1 = 1 -> x_Bsmt_Unf_SF <= 429
 ind_6106: t342_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6107: t342_path2 = 1 -> const_Alley_No_aplica >= 1
 ind_6108: t342_path2 = 1 -> x_Bsmt_Unf_SF <= 429
 ind_6109: t342_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6110: t342_path3 = 1 -> const_Alley_No_aplica >= 1
 ind_6111: t342_path3 = 1 -> x_Bsmt_Unf_SF >= 429
 ind_6112: t342_path3 = 1 -> const_Neighborhood_Edwards <= 1
 ind_6113: t342_path4 = 1 -> const_Alley_No_aplica >= 1
 ind_6114: t342_path4 = 1 -> x_Bsmt_Unf_SF >= 429
 ind_6115: t342_path4 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6116: t343_path0 = 1 -> const_Foundation_Slab <= 1
 ind_6117: t343_path0 = 1 -> x_BsmtFin_SF_1 <= 1965
 ind_6118: t343_path0 = 1 -> const_Lot_Area <= 5000
 ind_6119: t343_path1 = 1 -> const_Foundation_Slab <= 1
 ind_6120: t343_path1 = 1 -> x_BsmtFin_SF_1 <= 1965
 ind_6121: t343_path1 = 1 -> const_Lot_Area >= 5000
 ind_6122: t343_path2 = 1 -> const_Foundation_Slab <= 1
 ind_6123: t343_path2 = 1 -> x_BsmtFin_SF_1 >= 1965
 ind_6124: t343_path3 = 1 -> const_Foundation_Slab >= 1
 ind_6125: t343_path3 = 1 -> const_Lot_Frontage <= 66
 ind_6126: t343_path4 = 1 -> const_Foundation_Slab >= 1
 ind_6127: t343_path4 = 1 -> const_Lot_Frontage >= 66
 ind_6128: t344_path0 = 1 -> Overall_Qual_calc <= 5
 ind_6129: t344_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_6130: t344_path0 = 1 -> const_Lot_Area <= 5976
 ind_6131: t344_path1 = 1 -> Overall_Qual_calc <= 5
 ind_6132: t344_path1 = 1 -> const_Mo_Sold_2 <= 1
 ind_6133: t344_path1 = 1 -> const_Lot_Area >= 5976
 ind_6134: t344_path2 = 1 -> Overall_Qual_calc <= 5
 ind_6135: t344_path2 = 1 -> const_Mo_Sold_2 >= 1
 ind_6136: t344_path3 = 1 -> Overall_Qual_calc >= 5
 ind_6137: t344_path3 = 1 -> x_BsmtFin_SF_1 <= 936
 ind_6138: t344_path3 = 1 -> const_Lot_Area <= 8400
 ind_6139: t344_path4 = 1 -> Overall_Qual_calc >= 5
 ind_6140: t344_path4 = 1 -> x_BsmtFin_SF_1 <= 936
 ind_6141: t344_path4 = 1 -> const_Lot_Area >= 8400
 ind_6142: t344_path5 = 1 -> Overall_Qual_calc >= 5
 ind_6143: t344_path5 = 1 -> x_BsmtFin_SF_1 >= 936
 ind_6144: t344_path5 = 1 -> const_Lot_Area <= 21370
 ind_6145: t344_path6 = 1 -> Overall_Qual_calc >= 5
 ind_6146: t344_path6 = 1 -> x_BsmtFin_SF_1 >= 936
 ind_6147: t344_path6 = 1 -> const_Lot_Area >= 21370
 ind_6148: t345_path0 = 1 -> x_Heating_QC <= 3
 ind_6149: t345_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_6150: t345_path0 = 1 -> const_Bsmt_Exposure_No <= 1
 ind_6151: t345_path1 = 1 -> x_Heating_QC <= 3
 ind_6152: t345_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_6153: t345_path1 = 1 -> const_Bsmt_Exposure_No >= 1
 ind_6154: t345_path2 = 1 -> x_Heating_QC <= 3
 ind_6155: t345_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_6156: t345_path2 = 1 -> Overall_Qual_calc <= 4
 ind_6157: t345_path3 = 1 -> x_Heating_QC <= 3
 ind_6158: t345_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_6159: t345_path3 = 1 -> Overall_Qual_calc >= 4
 ind_6160: t345_path4 = 1 -> x_Heating_QC >= 3
 ind_6161: t345_path4 = 1 -> Overall_Qual_calc <= 6
 ind_6162: t345_path5 = 1 -> x_Heating_QC >= 3
 ind_6163: t345_path5 = 1 -> Overall_Qual_calc >= 6
 ind_6164: t345_path5 = 1 -> const_Neighborhood_Edwards <= 1
 ind_6165: t345_path6 = 1 -> x_Heating_QC >= 3
 ind_6166: t345_path6 = 1 -> Overall_Qual_calc >= 6
 ind_6167: t345_path6 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6168: t346_path0 = 1 -> x_Mas_Vnr_Area <= 252
 ind_6169: t346_path0 = 1 -> x_Exter_Qual <= 3
 ind_6170: t346_path0 = 1 -> x_Heating_QC <= 2
 ind_6171: t346_path1 = 1 -> x_Mas_Vnr_Area <= 252
 ind_6172: t346_path1 = 1 -> x_Exter_Qual <= 3
 ind_6173: t346_path1 = 1 -> x_Heating_QC >= 2
 ind_6174: t346_path2 = 1 -> x_Mas_Vnr_Area <= 252
 ind_6175: t346_path2 = 1 -> x_Exter_Qual >= 3
 ind_6176: t346_path2 = 1 -> x_1st_Flr_SF <= 1782
 ind_6177: t346_path3 = 1 -> x_Mas_Vnr_Area <= 252
 ind_6178: t346_path3 = 1 -> x_Exter_Qual >= 3
 ind_6179: t346_path3 = 1 -> x_1st_Flr_SF >= 1782
 ind_6180: t346_path4 = 1 -> x_Mas_Vnr_Area >= 252
 ind_6181: t347_path0 = 1 -> x_Enclosed_Porch <= 290
 ind_6182: t347_path0 = 1 -> const_Lot_Frontage <= 130
 ind_6183: t347_path0 = 1 -> const_Condition_1_PosN <= 1
 ind_6184: t347_path1 = 1 -> x_Enclosed_Porch <= 290
 ind_6185: t347_path1 = 1 -> const_Lot_Frontage <= 130
 ind_6186: t347_path1 = 1 -> const_Condition_1_PosN >= 1
 ind_6187: t347_path2 = 1 -> x_Enclosed_Porch <= 290
 ind_6188: t347_path2 = 1 -> const_Lot_Frontage >= 130
 ind_6189: t347_path3 = 1 -> x_Enclosed_Porch >= 290
 ind_6190: t348_path0 = 1 -> x_Screen_Porch <= 108
 ind_6191: t348_path0 = 1 -> x_Mas_Vnr_Area <= 459
 ind_6192: t348_path0 = 1 -> const_Yr_Sold <= 2007
 ind_6193: t348_path1 = 1 -> x_Screen_Porch <= 108
 ind_6194: t348_path1 = 1 -> x_Mas_Vnr_Area <= 459
 ind_6195: t348_path1 = 1 -> const_Yr_Sold >= 2007
 ind_6196: t348_path2 = 1 -> x_Screen_Porch <= 108
 ind_6197: t348_path2 = 1 -> x_Mas_Vnr_Area >= 459
 ind_6198: t348_path3 = 1 -> x_Screen_Porch >= 108
 ind_6199: t348_path3 = 1 -> x_b1_is_Unf <= 1
 ind_6200: t348_path4 = 1 -> x_Screen_Porch >= 108
 ind_6201: t348_path4 = 1 -> x_b1_is_Unf >= 1
 ind_6202: t349_path0 = 1 -> const_Lot_Frontage <= 120
 ind_6203: t349_path0 = 1 -> x_1st_Flr_SF <= 2121
 ind_6204: t349_path0 = 1 -> x_Enclosed_Porch <= 265
 ind_6205: t349_path1 = 1 -> const_Lot_Frontage <= 120
 ind_6206: t349_path1 = 1 -> x_1st_Flr_SF <= 2121
 ind_6207: t349_path1 = 1 -> x_Enclosed_Porch >= 265
 ind_6208: t349_path2 = 1 -> const_Lot_Frontage <= 120
 ind_6209: t349_path2 = 1 -> x_1st_Flr_SF >= 2121
 ind_6210: t349_path3 = 1 -> const_Lot_Frontage >= 120
 ind_6211: t349_path3 = 1 -> const_Land_Contour_Bnk <= 1
 ind_6212: t349_path4 = 1 -> const_Lot_Frontage >= 120
 ind_6213: t349_path4 = 1 -> const_Land_Contour_Bnk >= 1
 ind_6214: t350_path0 = 1 -> x_Exter_Cond <= 2
 ind_6215: t350_path0 = 1 -> const_Lot_Frontage <= 66
 ind_6216: t350_path1 = 1 -> x_Exter_Cond <= 2
 ind_6217: t350_path1 = 1 -> const_Lot_Frontage >= 66
 ind_6218: t350_path1 = 1 -> const_2nd_Flr_SF <= 167
 ind_6219: t350_path2 = 1 -> x_Exter_Cond <= 2
 ind_6220: t350_path2 = 1 -> const_Lot_Frontage >= 66
 ind_6221: t350_path2 = 1 -> const_2nd_Flr_SF >= 167
 ind_6222: t350_path3 = 1 -> x_Exter_Cond >= 2
 ind_6223: t350_path3 = 1 -> const_Garage_Yr_Blt <= 1941
 ind_6224: t350_path3 = 1 -> x_Bsmt_Unf_SF <= 383
 ind_6225: t350_path4 = 1 -> x_Exter_Cond >= 2
 ind_6226: t350_path4 = 1 -> const_Garage_Yr_Blt <= 1941
 ind_6227: t350_path4 = 1 -> x_Bsmt_Unf_SF >= 383
 ind_6228: t350_path5 = 1 -> x_Exter_Cond >= 2
 ind_6229: t350_path5 = 1 -> const_Garage_Yr_Blt >= 1941
 ind_6230: t350_path5 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_6231: t350_path6 = 1 -> x_Exter_Cond >= 2
 ind_6232: t350_path6 = 1 -> const_Garage_Yr_Blt >= 1941
 ind_6233: t350_path6 = 1 -> x_Bsmt_Unf_SF >= 476
 ind_6234: t351_path0 = 1 -> const_Yr_Sold <= 2010
 ind_6235: t351_path0 = 1 -> Overall_Qual_calc <= 5
 ind_6236: t351_path1 = 1 -> const_Yr_Sold <= 2010
 ind_6237: t351_path1 = 1 -> Overall_Qual_calc >= 5
 ind_6238: t351_path1 = 1 -> const_MS_SubClass_190 <= 1
 ind_6239: t351_path2 = 1 -> const_Yr_Sold <= 2010
 ind_6240: t351_path2 = 1 -> Overall_Qual_calc >= 5
 ind_6241: t351_path2 = 1 -> const_MS_SubClass_190 >= 1
 ind_6242: t351_path3 = 1 -> const_Yr_Sold >= 2010
 ind_6243: t351_path3 = 1 -> x_Mas_Vnr_Area <= 324
 ind_6244: t351_path3 = 1 -> const_Total_Bsmt_SF <= 530
 ind_6245: t351_path4 = 1 -> const_Yr_Sold >= 2010
 ind_6246: t351_path4 = 1 -> x_Mas_Vnr_Area <= 324
 ind_6247: t351_path4 = 1 -> const_Total_Bsmt_SF >= 530
 ind_6248: t351_path5 = 1 -> const_Yr_Sold >= 2010
 ind_6249: t351_path5 = 1 -> x_Mas_Vnr_Area >= 324
 ind_6250: t352_path0 = 1 -> x_Open_Porch_SF <= 189
 ind_6251: t352_path1 = 1 -> x_Open_Porch_SF >= 189
 ind_6252: t353_path0 = 1 -> x_Open_Porch_SF <= 287
 ind_6253: t353_path0 = 1 -> x_Gr_Liv_Area <= 3086
 ind_6254: t353_path0 = 1 -> const_Roof_Style_Flat <= 1
 ind_6255: t353_path1 = 1 -> x_Open_Porch_SF <= 287
 ind_6256: t353_path1 = 1 -> x_Gr_Liv_Area <= 3086
 ind_6257: t353_path1 = 1 -> const_Roof_Style_Flat >= 1
 ind_6258: t353_path2 = 1 -> x_Open_Porch_SF <= 287
 ind_6259: t353_path2 = 1 -> x_Gr_Liv_Area >= 3086
 ind_6260: t353_path3 = 1 -> x_Open_Porch_SF >= 287
 ind_6261: t353_path3 = 1 -> x_TotRms_AbvGrd <= 10
 ind_6262: t353_path4 = 1 -> x_Open_Porch_SF >= 287
 ind_6263: t353_path4 = 1 -> x_TotRms_AbvGrd >= 10
 ind_6264: t354_path0 = 1 -> const_Year_Built <= 1911
 ind_6265: t354_path0 = 1 -> x_1st_Flr_SF <= 860
 ind_6266: t354_path1 = 1 -> const_Year_Built <= 1911
 ind_6267: t354_path1 = 1 -> x_1st_Flr_SF >= 860
 ind_6268: t354_path2 = 1 -> const_Year_Built >= 1911
 ind_6269: t355_path0 = 1 -> const_Garage_Type_Detchd <= 1
 ind_6270: t355_path0 = 1 -> const_Total_Bsmt_SF <= 648
 ind_6271: t355_path0 = 1 -> const_Garage_Yr_Blt <= 1995
 ind_6272: t355_path1 = 1 -> const_Garage_Type_Detchd <= 1
 ind_6273: t355_path1 = 1 -> const_Total_Bsmt_SF <= 648
 ind_6274: t355_path1 = 1 -> const_Garage_Yr_Blt >= 1995
 ind_6275: t355_path2 = 1 -> const_Garage_Type_Detchd <= 1
 ind_6276: t355_path2 = 1 -> const_Total_Bsmt_SF >= 648
 ind_6277: t355_path2 = 1 -> const_Total_Bsmt_SF <= 721
 ind_6278: t355_path3 = 1 -> const_Garage_Type_Detchd <= 1
 ind_6279: t355_path3 = 1 -> const_Total_Bsmt_SF >= 648
 ind_6280: t355_path3 = 1 -> const_Total_Bsmt_SF >= 721
 ind_6281: t355_path4 = 1 -> const_Garage_Type_Detchd >= 1
 ind_6282: t355_path4 = 1 -> const_Lot_Area <= 11025
 ind_6283: t355_path5 = 1 -> const_Garage_Type_Detchd >= 1
 ind_6284: t355_path5 = 1 -> const_Lot_Area >= 11025
 ind_6285: t355_path5 = 1 -> const_Sale_Type_WD_ <= 1
 ind_6286: t355_path6 = 1 -> const_Garage_Type_Detchd >= 1
 ind_6287: t355_path6 = 1 -> const_Lot_Area >= 11025
 ind_6288: t355_path6 = 1 -> const_Sale_Type_WD_ >= 1
 ind_6289: t356_path0 = 1 -> const_Overall_Cond <= 5
 ind_6290: t356_path0 = 1 -> const_Total_Bsmt_SF <= 727
 ind_6291: t356_path0 = 1 -> x_1st_Flr_SF <= 616
 ind_6292: t356_path1 = 1 -> const_Overall_Cond <= 5
 ind_6293: t356_path1 = 1 -> const_Total_Bsmt_SF <= 727
 ind_6294: t356_path1 = 1 -> x_1st_Flr_SF >= 616
 ind_6295: t356_path2 = 1 -> const_Overall_Cond <= 5
 ind_6296: t356_path2 = 1 -> const_Total_Bsmt_SF >= 727
 ind_6297: t356_path3 = 1 -> const_Overall_Cond >= 5
 ind_6298: t356_path3 = 1 -> const_2nd_Flr_SF <= 834
 ind_6299: t356_path3 = 1 -> const_Alley_Pave <= 1
 ind_6300: t356_path4 = 1 -> const_Overall_Cond >= 5
 ind_6301: t356_path4 = 1 -> const_2nd_Flr_SF <= 834
 ind_6302: t356_path4 = 1 -> const_Alley_Pave >= 1
 ind_6303: t356_path5 = 1 -> const_Overall_Cond >= 5
 ind_6304: t356_path5 = 1 -> const_2nd_Flr_SF >= 834
 ind_6305: t356_path5 = 1 -> Overall_Qual_calc <= 7
 ind_6306: t356_path6 = 1 -> const_Overall_Cond >= 5
 ind_6307: t356_path6 = 1 -> const_2nd_Flr_SF >= 834
 ind_6308: t356_path6 = 1 -> Overall_Qual_calc >= 7
 ind_6309: t357_path0 = 1 -> const_Mo_Sold_3 <= 1
 ind_6310: t357_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_6311: t357_path0 = 1 -> const_Street_Grvl <= 1
 ind_6312: t357_path1 = 1 -> const_Mo_Sold_3 <= 1
 ind_6313: t357_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_6314: t357_path1 = 1 -> const_Street_Grvl >= 1
 ind_6315: t357_path2 = 1 -> const_Mo_Sold_3 <= 1
 ind_6316: t357_path2 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_6317: t357_path3 = 1 -> const_Mo_Sold_3 >= 1
 ind_6318: t358_path0 = 1 -> central_air_no <= 1
 ind_6319: t358_path0 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_6320: t358_path0 = 1 -> x_1st_Flr_SF <= 1453
 ind_6321: t358_path1 = 1 -> central_air_no <= 1
 ind_6322: t358_path1 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_6323: t358_path1 = 1 -> x_1st_Flr_SF >= 1453
 ind_6324: t358_path2 = 1 -> central_air_no <= 1
 ind_6325: t358_path2 = 1 -> const_Lot_Config_CulDSac >= 1
 ind_6326: t358_path3 = 1 -> central_air_no >= 1
 ind_6327: t358_path3 = 1 -> const_Overall_Cond <= 5
 ind_6328: t358_path3 = 1 -> x_1st_Flr_SF <= 936
 ind_6329: t358_path4 = 1 -> central_air_no >= 1
 ind_6330: t358_path4 = 1 -> const_Overall_Cond <= 5
 ind_6331: t358_path4 = 1 -> x_1st_Flr_SF >= 936
 ind_6332: t358_path5 = 1 -> central_air_no >= 1
 ind_6333: t358_path5 = 1 -> const_Overall_Cond >= 5
 ind_6334: t358_path5 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_6335: t358_path6 = 1 -> central_air_no >= 1
 ind_6336: t358_path6 = 1 -> const_Overall_Cond >= 5
 ind_6337: t358_path6 = 1 -> const_Neighborhood_BrkSide >= 1
 ind_6338: t359_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6339: t359_path0 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_6340: t359_path0 = 1 -> x_Exter_Qual <= 3
 ind_6341: t359_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6342: t359_path1 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_6343: t359_path1 = 1 -> x_Exter_Qual >= 3
 ind_6344: t359_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6345: t359_path2 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_6346: t359_path2 = 1 -> const_Total_Bsmt_SF <= 1400
 ind_6347: t359_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6348: t359_path3 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_6349: t359_path3 = 1 -> const_Total_Bsmt_SF >= 1400
 ind_6350: t359_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6351: t359_path4 = 1 -> x_Garage_Area <= 250
 ind_6352: t359_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6353: t359_path5 = 1 -> x_Garage_Area >= 250
 ind_6354: t359_path5 = 1 -> const_Total_Bsmt_SF <= 791
 ind_6355: t359_path6 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6356: t359_path6 = 1 -> x_Garage_Area >= 250
 ind_6357: t359_path6 = 1 -> const_Total_Bsmt_SF >= 791
 ind_6358: t360_path0 = 1 -> Overall_Qual_calc <= 3
 ind_6359: t360_path1 = 1 -> Overall_Qual_calc >= 3
 ind_6360: t361_path0 = 1 -> x_ex1_is_Stucco <= 1
 ind_6361: t361_path0 = 1 -> x_1st_Flr_SF <= 646
 ind_6362: t361_path1 = 1 -> x_ex1_is_Stucco <= 1
 ind_6363: t361_path1 = 1 -> x_1st_Flr_SF >= 646
 ind_6364: t361_path1 = 1 -> x_Garage_Area <= 297
 ind_6365: t361_path2 = 1 -> x_ex1_is_Stucco <= 1
 ind_6366: t361_path2 = 1 -> x_1st_Flr_SF >= 646
 ind_6367: t361_path2 = 1 -> x_Garage_Area >= 297
 ind_6368: t361_path3 = 1 -> x_ex1_is_Stucco >= 1
 ind_6369: t362_path0 = 1 -> const_Year_Built <= 1915
 ind_6370: t362_path0 = 1 -> const_Overall_Cond <= 7
 ind_6371: t362_path1 = 1 -> const_Year_Built <= 1915
 ind_6372: t362_path1 = 1 -> const_Overall_Cond >= 7
 ind_6373: t362_path2 = 1 -> const_Year_Built >= 1915
 ind_6374: t362_path2 = 1 -> const_2nd_Flr_SF <= 358
 ind_6375: t362_path3 = 1 -> const_Year_Built >= 1915
 ind_6376: t362_path3 = 1 -> const_2nd_Flr_SF >= 358
 ind_6377: t362_path3 = 1 -> Overall_Qual_calc <= 7
 ind_6378: t362_path4 = 1 -> const_Year_Built >= 1915
 ind_6379: t362_path4 = 1 -> const_2nd_Flr_SF >= 358
 ind_6380: t362_path4 = 1 -> Overall_Qual_calc >= 7
 ind_6381: t363_path0 = 1 -> const_Sale_Condition_Family <= 1
 ind_6382: t363_path0 = 1 -> x_Bsmt_Unf_SF <= 1393
 ind_6383: t363_path0 = 1 -> x_b1_is_Rec <= 1
 ind_6384: t363_path1 = 1 -> const_Sale_Condition_Family <= 1
 ind_6385: t363_path1 = 1 -> x_Bsmt_Unf_SF <= 1393
 ind_6386: t363_path1 = 1 -> x_b1_is_Rec >= 1
 ind_6387: t363_path2 = 1 -> const_Sale_Condition_Family <= 1
 ind_6388: t363_path2 = 1 -> x_Bsmt_Unf_SF >= 1393
 ind_6389: t363_path3 = 1 -> const_Sale_Condition_Family >= 1
 ind_6390: t363_path3 = 1 -> x_Bsmt_Unf_SF <= 507
 ind_6391: t363_path4 = 1 -> const_Sale_Condition_Family >= 1
 ind_6392: t363_path4 = 1 -> x_Bsmt_Unf_SF >= 507
 ind_6393: t364_path0 = 1 -> const_Street_Grvl <= 1
 ind_6394: t364_path0 = 1 -> central_air_yes <= 1
 ind_6395: t364_path0 = 1 -> const_Lot_Area <= 7920
 ind_6396: t364_path1 = 1 -> const_Street_Grvl <= 1
 ind_6397: t364_path1 = 1 -> central_air_yes <= 1
 ind_6398: t364_path1 = 1 -> const_Lot_Area >= 7920
 ind_6399: t364_path2 = 1 -> const_Street_Grvl <= 1
 ind_6400: t364_path2 = 1 -> central_air_yes >= 1
 ind_6401: t364_path2 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_6402: t364_path3 = 1 -> const_Street_Grvl <= 1
 ind_6403: t364_path3 = 1 -> central_air_yes >= 1
 ind_6404: t364_path3 = 1 -> const_Bsmt_Exposure_Av >= 1
 ind_6405: t364_path4 = 1 -> const_Street_Grvl >= 1
 ind_6406: t365_path0 = 1 -> const_FunctioNo_aplical_Maj2 <= 1
 ind_6407: t365_path0 = 1 -> x_Gr_Liv_Area <= 816
 ind_6408: t365_path0 = 1 -> x_paved_drive_is_N <= 1
 ind_6409: t365_path1 = 1 -> const_FunctioNo_aplical_Maj2 <= 1
 ind_6410: t365_path1 = 1 -> x_Gr_Liv_Area <= 816
 ind_6411: t365_path1 = 1 -> x_paved_drive_is_N >= 1
 ind_6412: t365_path2 = 1 -> const_FunctioNo_aplical_Maj2 <= 1
 ind_6413: t365_path2 = 1 -> x_Gr_Liv_Area >= 816
 ind_6414: t365_path2 = 1 -> const_MS_SubClass_30 <= 1
 ind_6415: t365_path3 = 1 -> const_FunctioNo_aplical_Maj2 <= 1
 ind_6416: t365_path3 = 1 -> x_Gr_Liv_Area >= 816
 ind_6417: t365_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_6418: t365_path4 = 1 -> const_FunctioNo_aplical_Maj2 >= 1
 ind_6419: t366_path0 = 1 -> x_Fireplace_Qu <= 2
 ind_6420: t366_path0 = 1 -> const_Lot_Area <= 4500
 ind_6421: t366_path1 = 1 -> x_Fireplace_Qu <= 2
 ind_6422: t366_path1 = 1 -> const_Lot_Area >= 4500
 ind_6423: t366_path2 = 1 -> x_Fireplace_Qu >= 2
 ind_6424: t366_path2 = 1 -> x_Exter_Cond <= 2
 ind_6425: t366_path3 = 1 -> x_Fireplace_Qu >= 2
 ind_6426: t366_path3 = 1 -> x_Exter_Cond >= 2
 ind_6427: t366_path3 = 1 -> const_Land_Contour_HLS <= 1
 ind_6428: t366_path4 = 1 -> x_Fireplace_Qu >= 2
 ind_6429: t366_path4 = 1 -> x_Exter_Cond >= 2
 ind_6430: t366_path4 = 1 -> const_Land_Contour_HLS >= 1
 ind_6431: t367_path0 = 1 -> Overall_Qual_calc <= 4
 ind_6432: t367_path1 = 1 -> Overall_Qual_calc >= 4
 ind_6433: t367_path1 = 1 -> const_Street_Grvl <= 1
 ind_6434: t367_path1 = 1 -> x_b2_is_BLQ <= 1
 ind_6435: t367_path2 = 1 -> Overall_Qual_calc >= 4
 ind_6436: t367_path2 = 1 -> const_Street_Grvl <= 1
 ind_6437: t367_path2 = 1 -> x_b2_is_BLQ >= 1
 ind_6438: t367_path3 = 1 -> Overall_Qual_calc >= 4
 ind_6439: t367_path3 = 1 -> const_Street_Grvl >= 1
 ind_6440: t368_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_6441: t368_path1 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6442: t368_path1 = 1 -> const_Total_Bsmt_SF <= 761
 ind_6443: t368_path1 = 1 -> x_Garage_Area <= 225
 ind_6444: t368_path2 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6445: t368_path2 = 1 -> const_Total_Bsmt_SF <= 761
 ind_6446: t368_path2 = 1 -> x_Garage_Area >= 225
 ind_6447: t368_path3 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6448: t368_path3 = 1 -> const_Total_Bsmt_SF >= 761
 ind_6449: t368_path3 = 1 -> x_Garage_Area <= 250
 ind_6450: t368_path4 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6451: t368_path4 = 1 -> const_Total_Bsmt_SF >= 761
 ind_6452: t368_path4 = 1 -> x_Garage_Area >= 250
 ind_6453: t369_path0 = 1 -> const_Alley_Pave <= 1
 ind_6454: t369_path0 = 1 -> x_b1_is_ALQ <= 1
 ind_6455: t369_path0 = 1 -> const_Year_Built <= 2005
 ind_6456: t369_path1 = 1 -> const_Alley_Pave <= 1
 ind_6457: t369_path1 = 1 -> x_b1_is_ALQ <= 1
 ind_6458: t369_path1 = 1 -> const_Year_Built >= 2005
 ind_6459: t369_path2 = 1 -> const_Alley_Pave <= 1
 ind_6460: t369_path2 = 1 -> x_b1_is_ALQ >= 1
 ind_6461: t369_path2 = 1 -> Overall_Qual_calc <= 7
 ind_6462: t369_path3 = 1 -> const_Alley_Pave <= 1
 ind_6463: t369_path3 = 1 -> x_b1_is_ALQ >= 1
 ind_6464: t369_path3 = 1 -> Overall_Qual_calc >= 7
 ind_6465: t369_path4 = 1 -> const_Alley_Pave >= 1
 ind_6466: t369_path4 = 1 -> Overall_Qual_calc <= 7
 ind_6467: t369_path5 = 1 -> const_Alley_Pave >= 1
 ind_6468: t369_path5 = 1 -> Overall_Qual_calc >= 7
 ind_6469: t370_path0 = 1 -> const_Lot_Shape_IR3 <= 1
 ind_6470: t370_path0 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_6471: t370_path1 = 1 -> const_Lot_Shape_IR3 <= 1
 ind_6472: t370_path1 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_6473: t370_path2 = 1 -> const_Lot_Shape_IR3 >= 1
 ind_6474: t371_path0 = 1 -> x_Gr_Liv_Area <= 936
 ind_6475: t371_path1 = 1 -> x_Gr_Liv_Area >= 936
 ind_6476: t371_path1 = 1 -> x_Gr_Liv_Area <= 1646
 ind_6477: t371_path1 = 1 -> central_air_no <= 1
 ind_6478: t371_path2 = 1 -> x_Gr_Liv_Area >= 936
 ind_6479: t371_path2 = 1 -> x_Gr_Liv_Area <= 1646
 ind_6480: t371_path2 = 1 -> central_air_no >= 1
 ind_6481: t371_path3 = 1 -> x_Gr_Liv_Area >= 936
 ind_6482: t371_path3 = 1 -> x_Gr_Liv_Area >= 1646
 ind_6483: t371_path3 = 1 -> const_Year_Built <= 1911
 ind_6484: t371_path4 = 1 -> x_Gr_Liv_Area >= 936
 ind_6485: t371_path4 = 1 -> x_Gr_Liv_Area >= 1646
 ind_6486: t371_path4 = 1 -> const_Year_Built >= 1911
 ind_6487: t372_path0 = 1 -> x_Screen_Porch <= 108
 ind_6488: t372_path0 = 1 -> const_Lot_Frontage <= 174
 ind_6489: t372_path1 = 1 -> x_Screen_Porch <= 108
 ind_6490: t372_path1 = 1 -> const_Lot_Frontage >= 174
 ind_6491: t372_path2 = 1 -> x_Screen_Porch >= 108
 ind_6492: t373_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_6493: t373_path0 = 1 -> const_Total_Bsmt_SF <= 506
 ind_6494: t373_path0 = 1 -> x_Garage_Cond <= 1
 ind_6495: t373_path1 = 1 -> x_ex2_is_AsbShng <= 1
 ind_6496: t373_path1 = 1 -> const_Total_Bsmt_SF <= 506
 ind_6497: t373_path1 = 1 -> x_Garage_Cond >= 1
 ind_6498: t373_path2 = 1 -> x_ex2_is_AsbShng <= 1
 ind_6499: t373_path2 = 1 -> const_Total_Bsmt_SF >= 506
 ind_6500: t373_path2 = 1 -> const_Land_Slope_Gtl <= 1
 ind_6501: t373_path3 = 1 -> x_ex2_is_AsbShng <= 1
 ind_6502: t373_path3 = 1 -> const_Total_Bsmt_SF >= 506
 ind_6503: t373_path3 = 1 -> const_Land_Slope_Gtl >= 1
 ind_6504: t373_path4 = 1 -> x_ex2_is_AsbShng >= 1
 ind_6505: t374_path0 = 1 -> x_b2_is_BLQ <= 1
 ind_6506: t374_path0 = 1 -> x_Full_Bath <= 3
 ind_6507: t374_path0 = 1 -> x_BsmtFin_SF_2 <= 712
 ind_6508: t374_path1 = 1 -> x_b2_is_BLQ <= 1
 ind_6509: t374_path1 = 1 -> x_Full_Bath <= 3
 ind_6510: t374_path1 = 1 -> x_BsmtFin_SF_2 >= 712
 ind_6511: t374_path2 = 1 -> x_b2_is_BLQ <= 1
 ind_6512: t374_path2 = 1 -> x_Full_Bath >= 3
 ind_6513: t374_path3 = 1 -> x_b2_is_BLQ >= 1
 ind_6514: t374_path3 = 1 -> x_Garage_Area <= 338
 ind_6515: t374_path4 = 1 -> x_b2_is_BLQ >= 1
 ind_6516: t374_path4 = 1 -> x_Garage_Area >= 338
 ind_6517: t374_path4 = 1 -> const_Garage_Type_Detchd <= 1
 ind_6518: t374_path5 = 1 -> x_b2_is_BLQ >= 1
 ind_6519: t374_path5 = 1 -> x_Garage_Area >= 338
 ind_6520: t374_path5 = 1 -> const_Garage_Type_Detchd >= 1
 ind_6521: t375_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_6522: t375_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6523: t375_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_6524: t375_path1 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6525: t375_path1 = 1 -> const_Garage_Yr_Blt <= 1921
 ind_6526: t375_path2 = 1 -> x_ex1_is_BrkFace <= 1
 ind_6527: t375_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6528: t375_path2 = 1 -> const_Garage_Yr_Blt >= 1921
 ind_6529: t375_path3 = 1 -> x_ex1_is_BrkFace >= 1
 ind_6530: t376_path0 = 1 -> const_Year_Built <= 2005
 ind_6531: t376_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_6532: t376_path1 = 1 -> const_Year_Built <= 2005
 ind_6533: t376_path1 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6534: t376_path1 = 1 -> const_Garage_Yr_Blt <= 1956
 ind_6535: t376_path2 = 1 -> const_Year_Built <= 2005
 ind_6536: t376_path2 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6537: t376_path2 = 1 -> const_Garage_Yr_Blt >= 1956
 ind_6538: t376_path3 = 1 -> const_Year_Built >= 2005
 ind_6539: t378_path0 = 1 -> Overall_Qual_calc <= 3
 ind_6540: t378_path0 = 1 -> x_Garage_Area <= 273
 ind_6541: t378_path1 = 1 -> Overall_Qual_calc <= 3
 ind_6542: t378_path1 = 1 -> x_Garage_Area >= 273
 ind_6543: t378_path2 = 1 -> Overall_Qual_calc >= 3
 ind_6544: t379_path0 = 1 -> const_Garage_Cars <= 3
 ind_6545: t379_path0 = 1 -> x_1st_Flr_SF <= 576
 ind_6546: t379_path1 = 1 -> const_Garage_Cars <= 3
 ind_6547: t379_path1 = 1 -> x_1st_Flr_SF >= 576
 ind_6548: t379_path2 = 1 -> const_Garage_Cars >= 3
 ind_6549: t379_path2 = 1 -> x_Fireplace_Qu <= 0
 ind_6550: t379_path3 = 1 -> const_Garage_Cars >= 3
 ind_6551: t379_path3 = 1 -> x_Fireplace_Qu >= 0
 ind_6552: t379_path3 = 1 -> x_Screen_Porch <= 40
 ind_6553: t379_path4 = 1 -> const_Garage_Cars >= 3
 ind_6554: t379_path4 = 1 -> x_Fireplace_Qu >= 0
 ind_6555: t379_path4 = 1 -> x_Screen_Porch >= 40
 ind_6556: t380_path0 = 1 -> x_Screen_Porch <= 156
 ind_6557: t380_path1 = 1 -> x_Screen_Porch >= 156
 ind_6558: t381_path0 = 1 -> Overall_Qual_calc <= 4
 ind_6559: t381_path0 = 1 -> x_ex1_is_MetalSd <= 1
 ind_6560: t381_path1 = 1 -> Overall_Qual_calc <= 4
 ind_6561: t381_path1 = 1 -> x_ex1_is_MetalSd >= 1
 ind_6562: t381_path2 = 1 -> Overall_Qual_calc >= 4
 ind_6563: t381_path2 = 1 -> const_Year_Built <= 1911
 ind_6564: t381_path2 = 1 -> const_Overall_Cond <= 7
 ind_6565: t381_path3 = 1 -> Overall_Qual_calc >= 4
 ind_6566: t381_path3 = 1 -> const_Year_Built <= 1911
 ind_6567: t381_path3 = 1 -> const_Overall_Cond >= 7
 ind_6568: t381_path4 = 1 -> Overall_Qual_calc >= 4
 ind_6569: t381_path4 = 1 -> const_Year_Built >= 1911
 ind_6570: t382_path0 = 1 -> x_Kitchen_Qual <= 2
 ind_6571: t382_path0 = 1 -> const_Garage_Cars <= 2
 ind_6572: t382_path1 = 1 -> x_Kitchen_Qual <= 2
 ind_6573: t382_path1 = 1 -> const_Garage_Cars >= 2
 ind_6574: t382_path2 = 1 -> x_Kitchen_Qual >= 2
 ind_6575: t382_path2 = 1 -> const_Year_Remod_Add <= 1965
 ind_6576: t382_path2 = 1 -> x_1st_Flr_SF <= 916
 ind_6577: t382_path3 = 1 -> x_Kitchen_Qual >= 2
 ind_6578: t382_path3 = 1 -> const_Year_Remod_Add <= 1965
 ind_6579: t382_path3 = 1 -> x_1st_Flr_SF >= 916
 ind_6580: t382_path4 = 1 -> x_Kitchen_Qual >= 2
 ind_6581: t382_path4 = 1 -> const_Year_Remod_Add >= 1965
 ind_6582: t382_path4 = 1 -> const_MS_SubClass_30 <= 1
 ind_6583: t382_path5 = 1 -> x_Kitchen_Qual >= 2
 ind_6584: t382_path5 = 1 -> const_Year_Remod_Add >= 1965
 ind_6585: t382_path5 = 1 -> const_MS_SubClass_30 >= 1
 ind_6586: t383_path0 = 1 -> x_Gr_Liv_Area <= 894
 ind_6587: t383_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_6588: t383_path0 = 1 -> const_Lot_Shape_Reg <= 1
 ind_6589: t383_path1 = 1 -> x_Gr_Liv_Area <= 894
 ind_6590: t383_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_6591: t383_path1 = 1 -> const_Lot_Shape_Reg >= 1
 ind_6592: t383_path2 = 1 -> x_Gr_Liv_Area <= 894
 ind_6593: t383_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_6594: t383_path2 = 1 -> x_Garage_Area <= 371
 ind_6595: t383_path3 = 1 -> x_Gr_Liv_Area <= 894
 ind_6596: t383_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_6597: t383_path3 = 1 -> x_Garage_Area >= 371
 ind_6598: t383_path4 = 1 -> x_Gr_Liv_Area >= 894
 ind_6599: t383_path4 = 1 -> x_TotRms_AbvGrd <= 12
 ind_6600: t383_path4 = 1 -> const_Neighborhood_SawyerW <= 1
 ind_6601: t383_path5 = 1 -> x_Gr_Liv_Area >= 894
 ind_6602: t383_path5 = 1 -> x_TotRms_AbvGrd <= 12
 ind_6603: t383_path5 = 1 -> const_Neighborhood_SawyerW >= 1
 ind_6604: t383_path6 = 1 -> x_Gr_Liv_Area >= 894
 ind_6605: t383_path6 = 1 -> x_TotRms_AbvGrd >= 12
 ind_6606: t384_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_6607: t384_path1 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_6608: t384_path1 = 1 -> x_Bsmt_Unf_SF <= 122
 ind_6609: t384_path1 = 1 -> const_House_Style_1.5Fin <= 1
 ind_6610: t384_path2 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_6611: t384_path2 = 1 -> x_Bsmt_Unf_SF <= 122
 ind_6612: t384_path2 = 1 -> const_House_Style_1.5Fin >= 1
 ind_6613: t384_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_6614: t384_path3 = 1 -> x_Bsmt_Unf_SF >= 122
 ind_6615: t384_path3 = 1 -> x_Garage_Area <= 871
 ind_6616: t384_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_6617: t384_path4 = 1 -> x_Bsmt_Unf_SF >= 122
 ind_6618: t384_path4 = 1 -> x_Garage_Area >= 871
 ind_6619: t385_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6620: t385_path0 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_6621: t385_path0 = 1 -> Overall_Qual_calc <= 7
 ind_6622: t385_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6623: t385_path1 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_6624: t385_path1 = 1 -> Overall_Qual_calc >= 7
 ind_6625: t385_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6626: t385_path2 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_6627: t385_path3 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6628: t385_path3 = 1 -> x_Gr_Liv_Area <= 883
 ind_6629: t385_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6630: t385_path4 = 1 -> x_Gr_Liv_Area >= 883
 ind_6631: t386_path0 = 1 -> x_Garage_Area <= 1154
 ind_6632: t386_path0 = 1 -> const_Year_Built <= 2006
 ind_6633: t386_path0 = 1 -> Overall_Qual_calc <= 3
 ind_6634: t386_path1 = 1 -> x_Garage_Area <= 1154
 ind_6635: t386_path1 = 1 -> const_Year_Built <= 2006
 ind_6636: t386_path1 = 1 -> Overall_Qual_calc >= 3
 ind_6637: t386_path2 = 1 -> x_Garage_Area <= 1154
 ind_6638: t386_path2 = 1 -> const_Year_Built >= 2006
 ind_6639: t386_path3 = 1 -> x_Garage_Area >= 1154
 ind_6640: t387_path0 = 1 -> const_Lot_Frontage <= 121
 ind_6641: t387_path0 = 1 -> heat_is_Wall <= 1
 ind_6642: t387_path0 = 1 -> central_air_no <= 1
 ind_6643: t387_path1 = 1 -> const_Lot_Frontage <= 121
 ind_6644: t387_path1 = 1 -> heat_is_Wall <= 1
 ind_6645: t387_path1 = 1 -> central_air_no >= 1
 ind_6646: t387_path2 = 1 -> const_Lot_Frontage <= 121
 ind_6647: t387_path2 = 1 -> heat_is_Wall >= 1
 ind_6648: t387_path3 = 1 -> const_Lot_Frontage >= 121
 ind_6649: t387_path3 = 1 -> const_Sale_Type_WD_ <= 1
 ind_6650: t387_path4 = 1 -> const_Lot_Frontage >= 121
 ind_6651: t387_path4 = 1 -> const_Sale_Type_WD_ >= 1
 ind_6652: t388_path0 = 1 -> x_Kitchen_AbvGr <= 2
 ind_6653: t388_path0 = 1 -> const_Overall_Cond <= 3
 ind_6654: t388_path1 = 1 -> x_Kitchen_AbvGr <= 2
 ind_6655: t388_path1 = 1 -> const_Overall_Cond >= 3
 ind_6656: t388_path1 = 1 -> central_air_yes <= 1
 ind_6657: t388_path2 = 1 -> x_Kitchen_AbvGr <= 2
 ind_6658: t388_path2 = 1 -> const_Overall_Cond >= 3
 ind_6659: t388_path2 = 1 -> central_air_yes >= 1
 ind_6660: t388_path3 = 1 -> x_Kitchen_AbvGr >= 2
 ind_6661: t389_path0 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_6662: t389_path0 = 1 -> const_Yr_Sold <= 2007
 ind_6663: t389_path0 = 1 -> central_air_no <= 1
 ind_6664: t389_path1 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_6665: t389_path1 = 1 -> const_Yr_Sold <= 2007
 ind_6666: t389_path1 = 1 -> central_air_no >= 1
 ind_6667: t389_path2 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_6668: t389_path2 = 1 -> const_Yr_Sold >= 2007
 ind_6669: t389_path3 = 1 -> const_Neighborhood_BrkSide >= 1
 ind_6670: t390_path0 = 1 -> const_Lot_Area <= 5976
 ind_6671: t390_path1 = 1 -> const_Lot_Area >= 5976
 ind_6672: t390_path1 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_6673: t390_path2 = 1 -> const_Lot_Area >= 5976
 ind_6674: t390_path2 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_6675: t391_path0 = 1 -> Overall_Qual_calc <= 3
 ind_6676: t391_path1 = 1 -> Overall_Qual_calc >= 3
 ind_6677: t391_path1 = 1 -> const_Garage_Type_CarPort <= 1
 ind_6678: t391_path1 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_6679: t391_path2 = 1 -> Overall_Qual_calc >= 3
 ind_6680: t391_path2 = 1 -> const_Garage_Type_CarPort <= 1
 ind_6681: t391_path2 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_6682: t391_path3 = 1 -> Overall_Qual_calc >= 3
 ind_6683: t391_path3 = 1 -> const_Garage_Type_CarPort >= 1
 ind_6684: t392_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_6685: t392_path0 = 1 -> x_Mas_Vnr_Area <= 456
 ind_6686: t392_path1 = 1 -> const_Neighborhood_Edwards <= 1
 ind_6687: t392_path1 = 1 -> x_Mas_Vnr_Area >= 456
 ind_6688: t392_path2 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6689: t392_path2 = 1 -> const_Total_Bsmt_SF <= 698
 ind_6690: t392_path3 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6691: t392_path3 = 1 -> const_Total_Bsmt_SF >= 698
 ind_6692: t393_path0 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_6693: t393_path0 = 1 -> x_Exter_Qual <= 3
 ind_6694: t393_path0 = 1 -> x_Gr_Liv_Area <= 1716
 ind_6695: t393_path1 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_6696: t393_path1 = 1 -> x_Exter_Qual <= 3
 ind_6697: t393_path1 = 1 -> x_Gr_Liv_Area >= 1716
 ind_6698: t393_path2 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_6699: t393_path2 = 1 -> x_Exter_Qual >= 3
 ind_6700: t393_path2 = 1 -> x_1st_Flr_SF <= 1782
 ind_6701: t393_path3 = 1 -> x_BsmtFin_SF_1 <= 1732
 ind_6702: t393_path3 = 1 -> x_Exter_Qual >= 3
 ind_6703: t393_path3 = 1 -> x_1st_Flr_SF >= 1782
 ind_6704: t393_path4 = 1 -> x_BsmtFin_SF_1 >= 1732
 ind_6705: t394_path0 = 1 -> x_b1_is_Rec <= 1
 ind_6706: t394_path1 = 1 -> x_b1_is_Rec >= 1
 ind_6707: t395_path0 = 1 -> x_Open_Porch_SF <= 291
 ind_6708: t395_path0 = 1 -> const_Year_Remod_Add <= 2007
 ind_6709: t395_path0 = 1 -> const_Overall_Cond <= 7
 ind_6710: t395_path1 = 1 -> x_Open_Porch_SF <= 291
 ind_6711: t395_path1 = 1 -> const_Year_Remod_Add <= 2007
 ind_6712: t395_path1 = 1 -> const_Overall_Cond >= 7
 ind_6713: t395_path2 = 1 -> x_Open_Porch_SF <= 291
 ind_6714: t395_path2 = 1 -> const_Year_Remod_Add >= 2007
 ind_6715: t395_path2 = 1 -> const_Neighborhood_Blmngtn <= 1
 ind_6716: t395_path3 = 1 -> x_Open_Porch_SF <= 291
 ind_6717: t395_path3 = 1 -> const_Year_Remod_Add >= 2007
 ind_6718: t395_path3 = 1 -> const_Neighborhood_Blmngtn >= 1
 ind_6719: t395_path4 = 1 -> x_Open_Porch_SF >= 291
 ind_6720: t395_path4 = 1 -> const_Sale_Condition_Normal <= 1
 ind_6721: t395_path5 = 1 -> x_Open_Porch_SF >= 291
 ind_6722: t395_path5 = 1 -> const_Sale_Condition_Normal >= 1
 ind_6723: t396_path0 = 1 -> Overall_Qual_calc <= 3
 ind_6724: t396_path0 = 1 -> const_Garage_Yr_Blt <= 1895
 ind_6725: t396_path1 = 1 -> Overall_Qual_calc <= 3
 ind_6726: t396_path1 = 1 -> const_Garage_Yr_Blt >= 1895
 ind_6727: t396_path2 = 1 -> Overall_Qual_calc >= 3
 ind_6728: t396_path2 = 1 -> x_Open_Porch_SF <= 484
 ind_6729: t396_path2 = 1 -> x_Garage_Cond <= 2
 ind_6730: t396_path3 = 1 -> Overall_Qual_calc >= 3
 ind_6731: t396_path3 = 1 -> x_Open_Porch_SF <= 484
 ind_6732: t396_path3 = 1 -> x_Garage_Cond >= 2
 ind_6733: t396_path4 = 1 -> Overall_Qual_calc >= 3
 ind_6734: t396_path4 = 1 -> x_Open_Porch_SF >= 484
 ind_6735: t397_path0 = 1 -> const_Sale_Condition_Family <= 1
 ind_6736: t397_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_6737: t397_path1 = 1 -> const_Sale_Condition_Family <= 1
 ind_6738: t397_path1 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6739: t397_path1 = 1 -> const_Garage_Yr_Blt <= 1923
 ind_6740: t397_path2 = 1 -> const_Sale_Condition_Family <= 1
 ind_6741: t397_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_6742: t397_path2 = 1 -> const_Garage_Yr_Blt >= 1923
 ind_6743: t397_path3 = 1 -> const_Sale_Condition_Family >= 1
 ind_6744: t397_path3 = 1 -> x_BsmtFin_SF_1 <= 167
 ind_6745: t397_path4 = 1 -> const_Sale_Condition_Family >= 1
 ind_6746: t397_path4 = 1 -> x_BsmtFin_SF_1 >= 167
 ind_6747: t398_path0 = 1 -> x_Open_Porch_SF <= 159
 ind_6748: t398_path0 = 1 -> x_Bsmt_Cond <= 3
 ind_6749: t398_path1 = 1 -> x_Open_Porch_SF <= 159
 ind_6750: t398_path1 = 1 -> x_Bsmt_Cond >= 3
 ind_6751: t398_path1 = 1 -> const_Garage_Yr_Blt <= 1977
 ind_6752: t398_path2 = 1 -> x_Open_Porch_SF <= 159
 ind_6753: t398_path2 = 1 -> x_Bsmt_Cond >= 3
 ind_6754: t398_path2 = 1 -> const_Garage_Yr_Blt >= 1977
 ind_6755: t398_path3 = 1 -> x_Open_Porch_SF >= 159
 ind_6756: t399_path0 = 1 -> const_Land_Slope_Gtl <= 1
 ind_6757: t399_path1 = 1 -> const_Land_Slope_Gtl >= 1
 ind_6758: t399_path1 = 1 -> const_Lot_Area <= 25286
 ind_6759: t399_path1 = 1 -> x_1st_Flr_SF <= 576
 ind_6760: t399_path2 = 1 -> const_Land_Slope_Gtl >= 1
 ind_6761: t399_path2 = 1 -> const_Lot_Area <= 25286
 ind_6762: t399_path2 = 1 -> x_1st_Flr_SF >= 576
 ind_6763: t399_path3 = 1 -> const_Land_Slope_Gtl >= 1
 ind_6764: t399_path3 = 1 -> const_Lot_Area >= 25286
 ind_6765: t400_path0 = 1 -> const_Garage_Type_CarPort <= 1
 ind_6766: t400_path0 = 1 -> const_Overall_Cond <= 7
 ind_6767: t400_path1 = 1 -> const_Garage_Type_CarPort <= 1
 ind_6768: t400_path1 = 1 -> const_Overall_Cond >= 7
 ind_6769: t400_path2 = 1 -> const_Garage_Type_CarPort >= 1
 ind_6770: t402_path0 = 1 -> const_Neighborhood_SawyerW <= 1
 ind_6771: t402_path0 = 1 -> heat_is_GasA <= 1
 ind_6772: t402_path0 = 1 -> x_1st_Flr_SF <= 932
 ind_6773: t402_path1 = 1 -> const_Neighborhood_SawyerW <= 1
 ind_6774: t402_path1 = 1 -> heat_is_GasA <= 1
 ind_6775: t402_path1 = 1 -> x_1st_Flr_SF >= 932
 ind_6776: t402_path2 = 1 -> const_Neighborhood_SawyerW <= 1
 ind_6777: t402_path2 = 1 -> heat_is_GasA >= 1
 ind_6778: t402_path2 = 1 -> Overall_Qual_calc <= 4
 ind_6779: t402_path3 = 1 -> const_Neighborhood_SawyerW <= 1
 ind_6780: t402_path3 = 1 -> heat_is_GasA >= 1
 ind_6781: t402_path3 = 1 -> Overall_Qual_calc >= 4
 ind_6782: t402_path4 = 1 -> const_Neighborhood_SawyerW >= 1
 ind_6783: t403_path0 = 1 -> const_Street_Grvl <= 1
 ind_6784: t403_path0 = 1 -> x_Garage_Area <= 630
 ind_6785: t403_path0 = 1 -> const_2nd_Flr_SF <= 940
 ind_6786: t403_path1 = 1 -> const_Street_Grvl <= 1
 ind_6787: t403_path1 = 1 -> x_Garage_Area <= 630
 ind_6788: t403_path1 = 1 -> const_2nd_Flr_SF >= 940
 ind_6789: t403_path2 = 1 -> const_Street_Grvl <= 1
 ind_6790: t403_path2 = 1 -> x_Garage_Area >= 630
 ind_6791: t403_path2 = 1 -> const_Overall_Cond <= 5
 ind_6792: t403_path3 = 1 -> const_Street_Grvl <= 1
 ind_6793: t403_path3 = 1 -> x_Garage_Area >= 630
 ind_6794: t403_path3 = 1 -> const_Overall_Cond >= 5
 ind_6795: t403_path4 = 1 -> const_Street_Grvl >= 1
 ind_6796: t404_path0 = 1 -> Overall_Qual_calc <= 5
 ind_6797: t404_path0 = 1 -> const_Year_Remod_Add <= 1953
 ind_6798: t404_path0 = 1 -> const_Lot_Frontage <= 51
 ind_6799: t404_path1 = 1 -> Overall_Qual_calc <= 5
 ind_6800: t404_path1 = 1 -> const_Year_Remod_Add <= 1953
 ind_6801: t404_path1 = 1 -> const_Lot_Frontage >= 51
 ind_6802: t404_path2 = 1 -> Overall_Qual_calc <= 5
 ind_6803: t404_path2 = 1 -> const_Year_Remod_Add >= 1953
 ind_6804: t404_path3 = 1 -> Overall_Qual_calc >= 5
 ind_6805: t404_path3 = 1 -> const_Condition_1_PosN <= 1
 ind_6806: t404_path3 = 1 -> const_Condition_1_Norm <= 1
 ind_6807: t404_path4 = 1 -> Overall_Qual_calc >= 5
 ind_6808: t404_path4 = 1 -> const_Condition_1_PosN <= 1
 ind_6809: t404_path4 = 1 -> const_Condition_1_Norm >= 1
 ind_6810: t404_path5 = 1 -> Overall_Qual_calc >= 5
 ind_6811: t404_path5 = 1 -> const_Condition_1_PosN >= 1
 ind_6812: t405_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_6813: t405_path0 = 1 -> x_Heating_QC <= 3
 ind_6814: t405_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_6815: t405_path1 = 1 -> x_Heating_QC >= 3
 ind_6816: t405_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_6817: t405_path2 = 1 -> Overall_Qual_calc <= 5
 ind_6818: t405_path3 = 1 -> x_Gr_Liv_Area >= 747
 ind_6819: t405_path3 = 1 -> Overall_Qual_calc >= 5
 ind_6820: t405_path3 = 1 -> x_Open_Porch_SF <= 260
 ind_6821: t405_path4 = 1 -> x_Gr_Liv_Area >= 747
 ind_6822: t405_path4 = 1 -> Overall_Qual_calc >= 5
 ind_6823: t405_path4 = 1 -> x_Open_Porch_SF >= 260
 ind_6824: t408_path0 = 1 -> const_2nd_Flr_SF <= 1611
 ind_6825: t408_path0 = 1 -> x_Gr_Liv_Area <= 2872
 ind_6826: t408_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_6827: t408_path1 = 1 -> const_2nd_Flr_SF <= 1611
 ind_6828: t408_path1 = 1 -> x_Gr_Liv_Area <= 2872
 ind_6829: t408_path1 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_6830: t408_path2 = 1 -> const_2nd_Flr_SF <= 1611
 ind_6831: t408_path2 = 1 -> x_Gr_Liv_Area >= 2872
 ind_6832: t408_path2 = 1 -> x_Bedroom_AbvGr <= 4
 ind_6833: t408_path3 = 1 -> const_2nd_Flr_SF <= 1611
 ind_6834: t408_path3 = 1 -> x_Gr_Liv_Area >= 2872
 ind_6835: t408_path3 = 1 -> x_Bedroom_AbvGr >= 4
 ind_6836: t408_path4 = 1 -> const_2nd_Flr_SF >= 1611
 ind_6837: t409_path0 = 1 -> const_Garage_Yr_Blt <= 1931
 ind_6838: t409_path0 = 1 -> const_Lot_Area <= 10337
 ind_6839: t409_path1 = 1 -> const_Garage_Yr_Blt <= 1931
 ind_6840: t409_path1 = 1 -> const_Lot_Area >= 10337
 ind_6841: t409_path1 = 1 -> x_ex2_is_VinylSd <= 1
 ind_6842: t409_path2 = 1 -> const_Garage_Yr_Blt <= 1931
 ind_6843: t409_path2 = 1 -> const_Lot_Area >= 10337
 ind_6844: t409_path2 = 1 -> x_ex2_is_VinylSd >= 1
 ind_6845: t409_path3 = 1 -> const_Garage_Yr_Blt >= 1931
 ind_6846: t409_path3 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_6847: t409_path4 = 1 -> const_Garage_Yr_Blt >= 1931
 ind_6848: t409_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_6849: t409_path4 = 1 -> x_Wood_Deck_SF <= 181
 ind_6850: t409_path5 = 1 -> const_Garage_Yr_Blt >= 1931
 ind_6851: t409_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_6852: t409_path5 = 1 -> x_Wood_Deck_SF >= 181
 ind_6853: t410_path0 = 1 -> x_1st_Flr_SF <= 2200
 ind_6854: t410_path0 = 1 -> const_MS_Zoning_FV <= 1
 ind_6855: t410_path0 = 1 -> x_Exter_Qual <= 3
 ind_6856: t410_path1 = 1 -> x_1st_Flr_SF <= 2200
 ind_6857: t410_path1 = 1 -> const_MS_Zoning_FV <= 1
 ind_6858: t410_path1 = 1 -> x_Exter_Qual >= 3
 ind_6859: t410_path2 = 1 -> x_1st_Flr_SF <= 2200
 ind_6860: t410_path2 = 1 -> const_MS_Zoning_FV >= 1
 ind_6861: t410_path3 = 1 -> x_1st_Flr_SF >= 2200
 ind_6862: t411_path0 = 1 -> const_Alley_Pave <= 1
 ind_6863: t411_path0 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_6864: t411_path1 = 1 -> const_Alley_Pave <= 1
 ind_6865: t411_path1 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_6866: t411_path2 = 1 -> const_Alley_Pave >= 1
 ind_6867: t412_path0 = 1 -> const_Mo_Sold_4 <= 1
 ind_6868: t412_path1 = 1 -> const_Mo_Sold_4 >= 1
 ind_6869: t412_path1 = 1 -> const_Lot_Frontage <= 77
 ind_6870: t412_path2 = 1 -> const_Mo_Sold_4 >= 1
 ind_6871: t412_path2 = 1 -> const_Lot_Frontage >= 77
 ind_6872: t413_path0 = 1 -> const_Lot_Area <= 6000
 ind_6873: t413_path1 = 1 -> const_Lot_Area >= 6000
 ind_6874: t413_path1 = 1 -> const_MS_SubClass_30 <= 1
 ind_6875: t413_path1 = 1 -> x_Screen_Porch <= 108
 ind_6876: t413_path2 = 1 -> const_Lot_Area >= 6000
 ind_6877: t413_path2 = 1 -> const_MS_SubClass_30 <= 1
 ind_6878: t413_path2 = 1 -> x_Screen_Porch >= 108
 ind_6879: t413_path3 = 1 -> const_Lot_Area >= 6000
 ind_6880: t413_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_6881: t413_path3 = 1 -> x_Enclosed_Porch <= 77
 ind_6882: t413_path4 = 1 -> const_Lot_Area >= 6000
 ind_6883: t413_path4 = 1 -> const_MS_SubClass_30 >= 1
 ind_6884: t413_path4 = 1 -> x_Enclosed_Porch >= 77
 ind_6885: t414_path0 = 1 -> const_MS_Zoning_FV <= 1
 ind_6886: t414_path0 = 1 -> x_Bsmt_Cond <= 3
 ind_6887: t414_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_6888: t414_path1 = 1 -> const_MS_Zoning_FV <= 1
 ind_6889: t414_path1 = 1 -> x_Bsmt_Cond <= 3
 ind_6890: t414_path1 = 1 -> const_Neighborhood_Edwards >= 1
 ind_6891: t414_path2 = 1 -> const_MS_Zoning_FV <= 1
 ind_6892: t414_path2 = 1 -> x_Bsmt_Cond >= 3
 ind_6893: t414_path3 = 1 -> const_MS_Zoning_FV >= 1
 ind_6894: t415_path0 = 1 -> x_Mas_Vnr_Area <= 470
 ind_6895: t415_path0 = 1 -> x_Full_Bath <= 2
 ind_6896: t415_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_6897: t415_path1 = 1 -> x_Mas_Vnr_Area <= 470
 ind_6898: t415_path1 = 1 -> x_Full_Bath <= 2
 ind_6899: t415_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_6900: t415_path2 = 1 -> x_Mas_Vnr_Area <= 470
 ind_6901: t415_path2 = 1 -> x_Full_Bath >= 2
 ind_6902: t415_path2 = 1 -> x_Exter_Qual <= 3
 ind_6903: t415_path3 = 1 -> x_Mas_Vnr_Area <= 470
 ind_6904: t415_path3 = 1 -> x_Full_Bath >= 2
 ind_6905: t415_path3 = 1 -> x_Exter_Qual >= 3
 ind_6906: t415_path4 = 1 -> x_Mas_Vnr_Area >= 470
 ind_6907: t416_path0 = 1 -> x_Garage_Cond <= 2
 ind_6908: t416_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_6909: t416_path0 = 1 -> const_Lot_Area <= 10990
 ind_6910: t416_path1 = 1 -> x_Garage_Cond <= 2
 ind_6911: t416_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_6912: t416_path1 = 1 -> const_Lot_Area >= 10990
 ind_6913: t416_path2 = 1 -> x_Garage_Cond <= 2
 ind_6914: t416_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_6915: t416_path2 = 1 -> x_TotRms_AbvGrd <= 7
 ind_6916: t416_path3 = 1 -> x_Garage_Cond <= 2
 ind_6917: t416_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_6918: t416_path3 = 1 -> x_TotRms_AbvGrd >= 7
 ind_6919: t416_path4 = 1 -> x_Garage_Cond >= 2
 ind_6920: t416_path4 = 1 -> x_Enclosed_Porch <= 264
 ind_6921: t416_path4 = 1 -> const_Roof_Style_Flat <= 1
 ind_6922: t416_path5 = 1 -> x_Garage_Cond >= 2
 ind_6923: t416_path5 = 1 -> x_Enclosed_Porch <= 264
 ind_6924: t416_path5 = 1 -> const_Roof_Style_Flat >= 1
 ind_6925: t416_path6 = 1 -> x_Garage_Cond >= 2
 ind_6926: t416_path6 = 1 -> x_Enclosed_Porch >= 264
 ind_6927: t418_path0 = 1 -> x_BsmtFin_SF_1 <= 1965
 ind_6928: t418_path1 = 1 -> x_BsmtFin_SF_1 >= 1965
 ind_6929: t419_path0 = 1 -> Overall_Qual_calc <= 5
 ind_6930: t419_path0 = 1 -> const_Bsmt_Qual <= 0
 ind_6931: t419_path0 = 1 -> const_Lot_Frontage <= 66
 ind_6932: t419_path1 = 1 -> Overall_Qual_calc <= 5
 ind_6933: t419_path1 = 1 -> const_Bsmt_Qual <= 0
 ind_6934: t419_path1 = 1 -> const_Lot_Frontage >= 66
 ind_6935: t419_path2 = 1 -> Overall_Qual_calc <= 5
 ind_6936: t419_path2 = 1 -> const_Bsmt_Qual >= 0
 ind_6937: t419_path2 = 1 -> const_Lot_Frontage <= 22
 ind_6938: t419_path3 = 1 -> Overall_Qual_calc <= 5
 ind_6939: t419_path3 = 1 -> const_Bsmt_Qual >= 0
 ind_6940: t419_path3 = 1 -> const_Lot_Frontage >= 22
 ind_6941: t419_path4 = 1 -> Overall_Qual_calc >= 5
 ind_6942: t419_path4 = 1 -> x_Full_Bath <= 2
 ind_6943: t419_path4 = 1 -> const_Condition_1_Norm <= 1
 ind_6944: t419_path5 = 1 -> Overall_Qual_calc >= 5
 ind_6945: t419_path5 = 1 -> x_Full_Bath <= 2
 ind_6946: t419_path5 = 1 -> const_Condition_1_Norm >= 1
 ind_6947: t419_path6 = 1 -> Overall_Qual_calc >= 5
 ind_6948: t419_path6 = 1 -> x_Full_Bath >= 2
 ind_6949: t419_path6 = 1 -> Overall_Qual_calc <= 7
 ind_6950: t419_path7 = 1 -> Overall_Qual_calc >= 5
 ind_6951: t419_path7 = 1 -> x_Full_Bath >= 2
 ind_6952: t419_path7 = 1 -> Overall_Qual_calc >= 7
 ind_6953: t421_path0 = 1 -> x_Exter_Cond <= 2
 ind_6954: t421_path0 = 1 -> x_Full_Bath <= 2
 ind_6955: t421_path0 = 1 -> const_Lot_Frontage <= 61
 ind_6956: t421_path1 = 1 -> x_Exter_Cond <= 2
 ind_6957: t421_path1 = 1 -> x_Full_Bath <= 2
 ind_6958: t421_path1 = 1 -> const_Lot_Frontage >= 61
 ind_6959: t421_path2 = 1 -> x_Exter_Cond <= 2
 ind_6960: t421_path2 = 1 -> x_Full_Bath >= 2
 ind_6961: t421_path3 = 1 -> x_Exter_Cond >= 2
 ind_6962: t422_path0 = 1 -> x_Open_Porch_SF <= 258
 ind_6963: t422_path0 = 1 -> x_Bsmt_Cond <= 2
 ind_6964: t422_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_6965: t422_path1 = 1 -> x_Open_Porch_SF <= 258
 ind_6966: t422_path1 = 1 -> x_Bsmt_Cond <= 2
 ind_6967: t422_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_6968: t422_path2 = 1 -> x_Open_Porch_SF <= 258
 ind_6969: t422_path2 = 1 -> x_Bsmt_Cond >= 2
 ind_6970: t422_path2 = 1 -> const_Condition_1_Artery <= 1
 ind_6971: t422_path3 = 1 -> x_Open_Porch_SF <= 258
 ind_6972: t422_path3 = 1 -> x_Bsmt_Cond >= 2
 ind_6973: t422_path3 = 1 -> const_Condition_1_Artery >= 1
 ind_6974: t422_path4 = 1 -> x_Open_Porch_SF >= 258
 ind_6975: t423_path0 = 1 -> x_Enclosed_Porch <= 290
 ind_6976: t423_path0 = 1 -> x_Bedroom_AbvGr <= 5
 ind_6977: t423_path0 = 1 -> x_Bsmt_Cond <= 1
 ind_6978: t423_path1 = 1 -> x_Enclosed_Porch <= 290
 ind_6979: t423_path1 = 1 -> x_Bedroom_AbvGr <= 5
 ind_6980: t423_path1 = 1 -> x_Bsmt_Cond >= 1
 ind_6981: t423_path2 = 1 -> x_Enclosed_Porch <= 290
 ind_6982: t423_path2 = 1 -> x_Bedroom_AbvGr >= 5
 ind_6983: t423_path3 = 1 -> x_Enclosed_Porch >= 290
 ind_6984: t424_path0 = 1 -> Overall_Qual_calc <= 3
 ind_6985: t424_path0 = 1 -> x_Enclosed_Porch <= 26
 ind_6986: t424_path1 = 1 -> Overall_Qual_calc <= 3
 ind_6987: t424_path1 = 1 -> x_Enclosed_Porch >= 26
 ind_6988: t424_path2 = 1 -> Overall_Qual_calc >= 3
 ind_6989: t424_path2 = 1 -> const_Condition_1_Norm <= 1
 ind_6990: t424_path2 = 1 -> x_Bsmt_Unf_SF <= 916
 ind_6991: t424_path3 = 1 -> Overall_Qual_calc >= 3
 ind_6992: t424_path3 = 1 -> const_Condition_1_Norm <= 1
 ind_6993: t424_path3 = 1 -> x_Bsmt_Unf_SF >= 916
 ind_6994: t424_path4 = 1 -> Overall_Qual_calc >= 3
 ind_6995: t424_path4 = 1 -> const_Condition_1_Norm >= 1
 ind_6996: t424_path4 = 1 -> x_ex1_is_BrkFace <= 1
 ind_6997: t424_path5 = 1 -> Overall_Qual_calc >= 3
 ind_6998: t424_path5 = 1 -> const_Condition_1_Norm >= 1
 ind_6999: t424_path5 = 1 -> x_ex1_is_BrkFace >= 1
 ind_7000: t425_path0 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_7001: t425_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_7002: t425_path1 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_7003: t425_path1 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_7004: t425_path2 = 1 -> const_Neighborhood_BrkSide >= 1
 ind_7005: t426_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_7006: t426_path0 = 1 -> x_Gr_Liv_Area <= 3395
 ind_7007: t426_path0 = 1 -> x_Garage_Cond <= 2
 ind_7008: t426_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_7009: t426_path1 = 1 -> x_Gr_Liv_Area <= 3395
 ind_7010: t426_path1 = 1 -> x_Garage_Cond >= 2
 ind_7011: t426_path2 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_7012: t426_path2 = 1 -> x_Gr_Liv_Area >= 3395
 ind_7013: t426_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_7014: t426_path3 = 1 -> const_Lot_Config_Corner <= 1
 ind_7015: t426_path3 = 1 -> const_Lot_Area <= 4671
 ind_7016: t426_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_7017: t426_path4 = 1 -> const_Lot_Config_Corner <= 1
 ind_7018: t426_path4 = 1 -> const_Lot_Area >= 4671
 ind_7019: t426_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_7020: t426_path5 = 1 -> const_Lot_Config_Corner >= 1
 ind_7021: t426_path5 = 1 -> x_Gr_Liv_Area <= 1128
 ind_7022: t426_path6 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_7023: t426_path6 = 1 -> const_Lot_Config_Corner >= 1
 ind_7024: t426_path6 = 1 -> x_Gr_Liv_Area >= 1128
 ind_7025: t427_path0 = 1 -> const_Lot_Area <= 27697
 ind_7026: t427_path0 = 1 -> const_Lot_Area <= 23257
 ind_7027: t427_path0 = 1 -> x_mvt_is_BrkCmn <= 1
 ind_7028: t427_path1 = 1 -> const_Lot_Area <= 27697
 ind_7029: t427_path1 = 1 -> const_Lot_Area <= 23257
 ind_7030: t427_path1 = 1 -> x_mvt_is_BrkCmn >= 1
 ind_7031: t427_path2 = 1 -> const_Lot_Area <= 27697
 ind_7032: t427_path2 = 1 -> const_Lot_Area >= 23257
 ind_7033: t427_path3 = 1 -> const_Lot_Area >= 27697
 ind_7034: t428_path0 = 1 -> x_Enclosed_Porch <= 268
 ind_7035: t428_path0 = 1 -> const_Year_Built <= 1946
 ind_7036: t428_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_7037: t428_path1 = 1 -> x_Enclosed_Porch <= 268
 ind_7038: t428_path1 = 1 -> const_Year_Built <= 1946
 ind_7039: t428_path1 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_7040: t428_path2 = 1 -> x_Enclosed_Porch <= 268
 ind_7041: t428_path2 = 1 -> const_Year_Built >= 1946
 ind_7042: t428_path2 = 1 -> const_Year_Built <= 1950
 ind_7043: t428_path3 = 1 -> x_Enclosed_Porch <= 268
 ind_7044: t428_path3 = 1 -> const_Year_Built >= 1946
 ind_7045: t428_path3 = 1 -> const_Year_Built >= 1950
 ind_7046: t428_path4 = 1 -> x_Enclosed_Porch >= 268
 ind_7047: t429_path0 = 1 -> x_Fireplace_Qu <= 2
 ind_7048: t429_path0 = 1 -> const_Bsmt_Half_Bath <= 1
 ind_7049: t429_path1 = 1 -> x_Fireplace_Qu <= 2
 ind_7050: t429_path1 = 1 -> const_Bsmt_Half_Bath >= 1
 ind_7051: t429_path2 = 1 -> x_Fireplace_Qu >= 2
 ind_7052: t429_path2 = 1 -> const_Neighborhood_Edwards <= 1
 ind_7053: t429_path2 = 1 -> const_Sale_Condition_Family <= 1
 ind_7054: t429_path3 = 1 -> x_Fireplace_Qu >= 2
 ind_7055: t429_path3 = 1 -> const_Neighborhood_Edwards <= 1
 ind_7056: t429_path3 = 1 -> const_Sale_Condition_Family >= 1
 ind_7057: t429_path4 = 1 -> x_Fireplace_Qu >= 2
 ind_7058: t429_path4 = 1 -> const_Neighborhood_Edwards >= 1
 ind_7059: t429_path4 = 1 -> const_Roof_Style_Hip <= 1
 ind_7060: t429_path5 = 1 -> x_Fireplace_Qu >= 2
 ind_7061: t429_path5 = 1 -> const_Neighborhood_Edwards >= 1
 ind_7062: t429_path5 = 1 -> const_Roof_Style_Hip >= 1
 ind_7063: t430_path0 = 1 -> x_Bsmt_Unf_SF <= 137
 ind_7064: t430_path0 = 1 -> x_Enclosed_Porch <= 224
 ind_7065: t430_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_7066: t430_path1 = 1 -> x_Bsmt_Unf_SF <= 137
 ind_7067: t430_path1 = 1 -> x_Enclosed_Porch <= 224
 ind_7068: t430_path1 = 1 -> const_Mo_Sold_2 >= 1
 ind_7069: t430_path2 = 1 -> x_Bsmt_Unf_SF <= 137
 ind_7070: t430_path2 = 1 -> x_Enclosed_Porch >= 224
 ind_7071: t430_path3 = 1 -> x_Bsmt_Unf_SF >= 137
 ind_7072: t430_path3 = 1 -> const_Lot_Area <= 20431
 ind_7073: t430_path3 = 1 -> const_2nd_Flr_SF <= 640
 ind_7074: t430_path4 = 1 -> x_Bsmt_Unf_SF >= 137
 ind_7075: t430_path4 = 1 -> const_Lot_Area <= 20431
 ind_7076: t430_path4 = 1 -> const_2nd_Flr_SF >= 640
 ind_7077: t430_path5 = 1 -> x_Bsmt_Unf_SF >= 137
 ind_7078: t430_path5 = 1 -> const_Lot_Area >= 20431
 ind_7079: t430_path5 = 1 -> x_Enclosed_Porch <= 194
 ind_7080: t430_path6 = 1 -> x_Bsmt_Unf_SF >= 137
 ind_7081: t430_path6 = 1 -> const_Lot_Area >= 20431
 ind_7082: t430_path6 = 1 -> x_Enclosed_Porch >= 194
 ind_7083: t431_path0 = 1 -> const_Foundation_Slab <= 1
 ind_7084: t431_path0 = 1 -> x_Open_Porch_SF <= 287
 ind_7085: t431_path0 = 1 -> x_Gr_Liv_Area <= 2649
 ind_7086: t431_path1 = 1 -> const_Foundation_Slab <= 1
 ind_7087: t431_path1 = 1 -> x_Open_Porch_SF <= 287
 ind_7088: t431_path1 = 1 -> x_Gr_Liv_Area >= 2649
 ind_7089: t431_path2 = 1 -> const_Foundation_Slab <= 1
 ind_7090: t431_path2 = 1 -> x_Open_Porch_SF >= 287
 ind_7091: t431_path3 = 1 -> const_Foundation_Slab >= 1
 ind_7092: t432_path0 = 1 -> const_Lot_Area <= 19522
 ind_7093: t432_path0 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_7094: t432_path0 = 1 -> x_Enclosed_Porch <= 224
 ind_7095: t432_path1 = 1 -> const_Lot_Area <= 19522
 ind_7096: t432_path1 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_7097: t432_path1 = 1 -> x_Enclosed_Porch >= 224
 ind_7098: t432_path2 = 1 -> const_Lot_Area <= 19522
 ind_7099: t432_path2 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_7100: t432_path3 = 1 -> const_Lot_Area >= 19522
 ind_7101: t432_path3 = 1 -> x_Bedroom_AbvGr <= 4
 ind_7102: t432_path3 = 1 -> const_Sale_Condition_Normal <= 1
 ind_7103: t432_path4 = 1 -> const_Lot_Area >= 19522
 ind_7104: t432_path4 = 1 -> x_Bedroom_AbvGr <= 4
 ind_7105: t432_path4 = 1 -> const_Sale_Condition_Normal >= 1
 ind_7106: t432_path5 = 1 -> const_Lot_Area >= 19522
 ind_7107: t432_path5 = 1 -> x_Bedroom_AbvGr >= 4
 ind_7108: t433_path0 = 1 -> x_Open_Porch_SF <= 208
 ind_7109: t433_path0 = 1 -> x_b2_is_ALQ <= 1
 ind_7110: t433_path1 = 1 -> x_Open_Porch_SF <= 208
 ind_7111: t433_path1 = 1 -> x_b2_is_ALQ >= 1
 ind_7112: t433_path2 = 1 -> x_Open_Porch_SF >= 208
 ind_7113: t435_path0 = 1 -> x_Enclosed_Porch <= 194
 ind_7114: t435_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_7115: t435_path0 = 1 -> x_Enclosed_Porch <= 64
 ind_7116: t435_path1 = 1 -> x_Enclosed_Porch <= 194
 ind_7117: t435_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_7118: t435_path1 = 1 -> x_Enclosed_Porch >= 64
 ind_7119: t435_path2 = 1 -> x_Enclosed_Porch <= 194
 ind_7120: t435_path2 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_7121: t435_path2 = 1 -> const_Overall_Cond <= 7
 ind_7122: t435_path3 = 1 -> x_Enclosed_Porch <= 194
 ind_7123: t435_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_7124: t435_path3 = 1 -> const_Overall_Cond >= 7
 ind_7125: t435_path4 = 1 -> x_Enclosed_Porch >= 194
 ind_7126: t437_path0 = 1 -> Overall_Qual_calc <= 4
 ind_7127: t437_path0 = 1 -> const_Land_Contour_Lvl <= 1
 ind_7128: t437_path1 = 1 -> Overall_Qual_calc <= 4
 ind_7129: t437_path1 = 1 -> const_Land_Contour_Lvl >= 1
 ind_7130: t437_path2 = 1 -> Overall_Qual_calc >= 4
 ind_7131: t438_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_7132: t438_path1 = 1 -> const_Condition_1_Norm >= 1
 ind_7133: t439_path0 = 1 -> central_air_no <= 1
 ind_7134: t439_path1 = 1 -> central_air_no >= 1
 ind_7135: t439_path1 = 1 -> x_BsmtFin_SF_1 <= 316
 ind_7136: t439_path1 = 1 -> const_Mo_Sold_5 <= 1
 ind_7137: t439_path2 = 1 -> central_air_no >= 1
 ind_7138: t439_path2 = 1 -> x_BsmtFin_SF_1 <= 316
 ind_7139: t439_path2 = 1 -> const_Mo_Sold_5 >= 1
 ind_7140: t439_path3 = 1 -> central_air_no >= 1
 ind_7141: t439_path3 = 1 -> x_BsmtFin_SF_1 >= 316
 ind_7142: t439_path3 = 1 -> x_BsmtFin_SF_1 <= 637
 ind_7143: t439_path4 = 1 -> central_air_no >= 1
 ind_7144: t439_path4 = 1 -> x_BsmtFin_SF_1 >= 316
 ind_7145: t439_path4 = 1 -> x_BsmtFin_SF_1 >= 637
 ind_7146: t440_path0 = 1 -> x_Bedroom_AbvGr <= 3
 ind_7147: t440_path0 = 1 -> x_Gr_Liv_Area <= 1795
 ind_7148: t440_path1 = 1 -> x_Bedroom_AbvGr <= 3
 ind_7149: t440_path1 = 1 -> x_Gr_Liv_Area >= 1795
 ind_7150: t440_path1 = 1 -> x_ex1_is_VinylSd <= 1
 ind_7151: t440_path2 = 1 -> x_Bedroom_AbvGr <= 3
 ind_7152: t440_path2 = 1 -> x_Gr_Liv_Area >= 1795
 ind_7153: t440_path2 = 1 -> x_ex1_is_VinylSd >= 1
 ind_7154: t440_path3 = 1 -> x_Bedroom_AbvGr >= 3
 ind_7155: t440_path3 = 1 -> x_Gr_Liv_Area <= 976
 ind_7156: t440_path3 = 1 -> const_Garage_Yr_Blt <= 1967
 ind_7157: t440_path4 = 1 -> x_Bedroom_AbvGr >= 3
 ind_7158: t440_path4 = 1 -> x_Gr_Liv_Area <= 976
 ind_7159: t440_path4 = 1 -> const_Garage_Yr_Blt >= 1967
 ind_7160: t440_path5 = 1 -> x_Bedroom_AbvGr >= 3
 ind_7161: t440_path5 = 1 -> x_Gr_Liv_Area >= 976
 ind_7162: t441_path0 = 1 -> x_Garage_Qual <= 1
 ind_7163: t441_path0 = 1 -> const_Lot_Area <= 8800
 ind_7164: t441_path1 = 1 -> x_Garage_Qual <= 1
 ind_7165: t441_path1 = 1 -> const_Lot_Area >= 8800
 ind_7166: t441_path2 = 1 -> x_Garage_Qual >= 1
 ind_7167: t441_path2 = 1 -> const_MS_SubClass_30 <= 1
 ind_7168: t441_path3 = 1 -> x_Garage_Qual >= 1
 ind_7169: t441_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_7170: t441_path3 = 1 -> x_1st_Flr_SF <= 820
 ind_7171: t441_path4 = 1 -> x_Garage_Qual >= 1
 ind_7172: t441_path4 = 1 -> const_MS_SubClass_30 >= 1
 ind_7173: t441_path4 = 1 -> x_1st_Flr_SF >= 820
 ind_7174: t442_path0 = 1 -> const_Roof_Style_Flat <= 1
 ind_7175: t442_path0 = 1 -> const_Condition_1_PosA <= 1
 ind_7176: t442_path1 = 1 -> const_Roof_Style_Flat <= 1
 ind_7177: t442_path1 = 1 -> const_Condition_1_PosA >= 1
 ind_7178: t442_path2 = 1 -> const_Roof_Style_Flat >= 1
 ind_7179: t444_path0 = 1 -> x_Fireplace_Qu <= 1
 ind_7180: t444_path0 = 1 -> const_2nd_Flr_SF <= 664
 ind_7181: t444_path0 = 1 -> const_Year_Built <= 1936
 ind_7182: t444_path1 = 1 -> x_Fireplace_Qu <= 1
 ind_7183: t444_path1 = 1 -> const_2nd_Flr_SF <= 664
 ind_7184: t444_path1 = 1 -> const_Year_Built >= 1936
 ind_7185: t444_path2 = 1 -> x_Fireplace_Qu <= 1
 ind_7186: t444_path2 = 1 -> const_2nd_Flr_SF >= 664
 ind_7187: t444_path2 = 1 -> const_Mo_Sold_9 <= 1
 ind_7188: t444_path3 = 1 -> x_Fireplace_Qu <= 1
 ind_7189: t444_path3 = 1 -> const_2nd_Flr_SF >= 664
 ind_7190: t444_path3 = 1 -> const_Mo_Sold_9 >= 1
 ind_7191: t444_path4 = 1 -> x_Fireplace_Qu >= 1
 ind_7192: t445_path0 = 1 -> const_Sale_Type_Oth <= 1
 ind_7193: t445_path0 = 1 -> x_Bsmt_Unf_SF <= 1257
 ind_7194: t445_path1 = 1 -> const_Sale_Type_Oth <= 1
 ind_7195: t445_path1 = 1 -> x_Bsmt_Unf_SF >= 1257
 ind_7196: t445_path2 = 1 -> const_Sale_Type_Oth >= 1
 ind_7197: t446_path0 = 1 -> x_Gr_Liv_Area <= 987
 ind_7198: t446_path0 = 1 -> const_Year_Remod_Add <= 1953
 ind_7199: t446_path1 = 1 -> x_Gr_Liv_Area <= 987
 ind_7200: t446_path1 = 1 -> const_Year_Remod_Add >= 1953
 ind_7201: t446_path1 = 1 -> const_Year_Built <= 1936
 ind_7202: t446_path2 = 1 -> x_Gr_Liv_Area <= 987
 ind_7203: t446_path2 = 1 -> const_Year_Remod_Add >= 1953
 ind_7204: t446_path2 = 1 -> const_Year_Built >= 1936
 ind_7205: t446_path3 = 1 -> x_Gr_Liv_Area >= 987
 ind_7206: t447_path0 = 1 -> const_Year_Built <= 2007
 ind_7207: t447_path1 = 1 -> const_Year_Built >= 2007
 ind_7208: t447_path1 = 1 -> const_Lot_Frontage <= 113
 ind_7209: t447_path2 = 1 -> const_Year_Built >= 2007
 ind_7210: t447_path2 = 1 -> const_Lot_Frontage >= 113
 ind_7211: t448_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_7212: t448_path0 = 1 -> const_Lot_Area <= 19522
 ind_7213: t448_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_7214: t448_path1 = 1 -> const_Lot_Area >= 19522
 ind_7215: t448_path2 = 1 -> x_1st_Flr_SF >= 2515
 ind_7216: t449_path0 = 1 -> x_Garage_Area <= 871
 ind_7217: t449_path0 = 1 -> const_Roof_Style_Gable <= 1
 ind_7218: t449_path0 = 1 -> const_Yr_Sold <= 2008
 ind_7219: t449_path1 = 1 -> x_Garage_Area <= 871
 ind_7220: t449_path1 = 1 -> const_Roof_Style_Gable <= 1
 ind_7221: t449_path1 = 1 -> const_Yr_Sold >= 2008
 ind_7222: t449_path2 = 1 -> x_Garage_Area <= 871
 ind_7223: t449_path2 = 1 -> const_Roof_Style_Gable >= 1
 ind_7224: t449_path2 = 1 -> const_Land_Contour_HLS <= 1
 ind_7225: t449_path3 = 1 -> x_Garage_Area <= 871
 ind_7226: t449_path3 = 1 -> const_Roof_Style_Gable >= 1
 ind_7227: t449_path3 = 1 -> const_Land_Contour_HLS >= 1
 ind_7228: t449_path4 = 1 -> x_Garage_Area >= 871
 ind_7229: t449_path4 = 1 -> x_Mas_Vnr_Area <= 724
 ind_7230: t449_path5 = 1 -> x_Garage_Area >= 871
 ind_7231: t449_path5 = 1 -> x_Mas_Vnr_Area >= 724
 ind_7232: t451_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_7233: t451_path0 = 1 -> const_Garage_Type_CarPort <= 1
 ind_7234: t451_path0 = 1 -> x_Bsmt_Unf_SF <= 113
 ind_7235: t451_path1 = 1 -> const_Mo_Sold_2 <= 1
 ind_7236: t451_path1 = 1 -> const_Garage_Type_CarPort <= 1
 ind_7237: t451_path1 = 1 -> x_Bsmt_Unf_SF >= 113
 ind_7238: t451_path2 = 1 -> const_Mo_Sold_2 <= 1
 ind_7239: t451_path2 = 1 -> const_Garage_Type_CarPort >= 1
 ind_7240: t451_path3 = 1 -> const_Mo_Sold_2 >= 1
 ind_7241: t451_path3 = 1 -> x_BsmtFin_SF_1 <= 219
 ind_7242: t451_path3 = 1 -> const_2nd_Flr_SF <= 760
 ind_7243: t451_path4 = 1 -> const_Mo_Sold_2 >= 1
 ind_7244: t451_path4 = 1 -> x_BsmtFin_SF_1 <= 219
 ind_7245: t451_path4 = 1 -> const_2nd_Flr_SF >= 760
 ind_7246: t451_path5 = 1 -> const_Mo_Sold_2 >= 1
 ind_7247: t451_path5 = 1 -> x_BsmtFin_SF_1 >= 219
 ind_7248: t452_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7249: t452_path0 = 1 -> const_Land_Slope_Gtl <= 1
 ind_7250: t452_path1 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7251: t452_path1 = 1 -> const_Land_Slope_Gtl >= 1
 ind_7252: t452_path1 = 1 -> const_Year_Built <= 1911
 ind_7253: t452_path2 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7254: t452_path2 = 1 -> const_Land_Slope_Gtl >= 1
 ind_7255: t452_path2 = 1 -> const_Year_Built >= 1911
 ind_7256: t452_path3 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7257: t452_path3 = 1 -> x_BsmtFin_SF_1 <= 176
 ind_7258: t452_path4 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7259: t452_path4 = 1 -> x_BsmtFin_SF_1 >= 176
 ind_7260: t453_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_7261: t453_path0 = 1 -> const_Lot_Area <= 6000
 ind_7262: t453_path0 = 1 -> x_Bsmt_Unf_SF <= 595
 ind_7263: t453_path1 = 1 -> const_Mo_Sold_2 <= 1
 ind_7264: t453_path1 = 1 -> const_Lot_Area <= 6000
 ind_7265: t453_path1 = 1 -> x_Bsmt_Unf_SF >= 595
 ind_7266: t453_path2 = 1 -> const_Mo_Sold_2 <= 1
 ind_7267: t453_path2 = 1 -> const_Lot_Area >= 6000
 ind_7268: t453_path2 = 1 -> Overall_Qual_calc <= 5
 ind_7269: t453_path3 = 1 -> const_Mo_Sold_2 <= 1
 ind_7270: t453_path3 = 1 -> const_Lot_Area >= 6000
 ind_7271: t453_path3 = 1 -> Overall_Qual_calc >= 5
 ind_7272: t453_path4 = 1 -> const_Mo_Sold_2 >= 1
 ind_7273: t453_path4 = 1 -> const_Sale_Condition_Normal <= 1
 ind_7274: t453_path5 = 1 -> const_Mo_Sold_2 >= 1
 ind_7275: t453_path5 = 1 -> const_Sale_Condition_Normal >= 1
 ind_7276: t454_path0 = 1 -> x_mvt_is_BrkCmn <= 1
 ind_7277: t454_path1 = 1 -> x_mvt_is_BrkCmn >= 1
 ind_7278: t457_path0 = 1 -> const_Sale_Condition_Family <= 1
 ind_7279: t457_path1 = 1 -> const_Sale_Condition_Family >= 1
 ind_7280: t457_path1 = 1 -> const_Lot_Config_Corner <= 1
 ind_7281: t457_path2 = 1 -> const_Sale_Condition_Family >= 1
 ind_7282: t457_path2 = 1 -> const_Lot_Config_Corner >= 1
 ind_7283: t458_path0 = 1 -> const_Lot_Area <= 6442
 ind_7284: t458_path1 = 1 -> const_Lot_Area >= 6442
 ind_7285: t458_path1 = 1 -> x_Enclosed_Porch <= 254
 ind_7286: t458_path1 = 1 -> x_Enclosed_Porch <= 236
 ind_7287: t458_path2 = 1 -> const_Lot_Area >= 6442
 ind_7288: t458_path2 = 1 -> x_Enclosed_Porch <= 254
 ind_7289: t458_path2 = 1 -> x_Enclosed_Porch >= 236
 ind_7290: t458_path3 = 1 -> const_Lot_Area >= 6442
 ind_7291: t458_path3 = 1 -> x_Enclosed_Porch >= 254
 ind_7292: t459_path0 = 1 -> const_Yr_Sold <= 2007
 ind_7293: t459_path1 = 1 -> const_Yr_Sold >= 2007
 ind_7294: t461_path0 = 1 -> Overall_Qual_calc <= 3
 ind_7295: t461_path1 = 1 -> Overall_Qual_calc >= 3
 ind_7296: t461_path1 = 1 -> const_Lot_Area <= 16779
 ind_7297: t461_path1 = 1 -> x_Enclosed_Porch <= 231
 ind_7298: t461_path2 = 1 -> Overall_Qual_calc >= 3
 ind_7299: t461_path2 = 1 -> const_Lot_Area <= 16779
 ind_7300: t461_path2 = 1 -> x_Enclosed_Porch >= 231
 ind_7301: t461_path3 = 1 -> Overall_Qual_calc >= 3
 ind_7302: t461_path3 = 1 -> const_Lot_Area >= 16779
 ind_7303: t461_path3 = 1 -> const_Misc_Feature_No_aplica <= 1
 ind_7304: t461_path4 = 1 -> Overall_Qual_calc >= 3
 ind_7305: t461_path4 = 1 -> const_Lot_Area >= 16779
 ind_7306: t461_path4 = 1 -> const_Misc_Feature_No_aplica >= 1
 ind_7307: t462_path0 = 1 -> const_Lot_Frontage <= 140
 ind_7308: t462_path1 = 1 -> const_Lot_Frontage >= 140
 ind_7309: t463_path0 = 1 -> x_Gr_Liv_Area <= 894
 ind_7310: t463_path0 = 1 -> const_Lot_Area <= 10182
 ind_7311: t463_path0 = 1 -> x_Wood_Deck_SF <= 22
 ind_7312: t463_path1 = 1 -> x_Gr_Liv_Area <= 894
 ind_7313: t463_path1 = 1 -> const_Lot_Area <= 10182
 ind_7314: t463_path1 = 1 -> x_Wood_Deck_SF >= 22
 ind_7315: t463_path2 = 1 -> x_Gr_Liv_Area <= 894
 ind_7316: t463_path2 = 1 -> const_Lot_Area >= 10182
 ind_7317: t463_path3 = 1 -> x_Gr_Liv_Area >= 894
 ind_7318: t463_path3 = 1 -> const_Lot_Area <= 21370
 ind_7319: t463_path3 = 1 -> const_Garage_Type_Basment <= 1
 ind_7320: t463_path4 = 1 -> x_Gr_Liv_Area >= 894
 ind_7321: t463_path4 = 1 -> const_Lot_Area <= 21370
 ind_7322: t463_path4 = 1 -> const_Garage_Type_Basment >= 1
 ind_7323: t463_path5 = 1 -> x_Gr_Liv_Area >= 894
 ind_7324: t463_path5 = 1 -> const_Lot_Area >= 21370
 ind_7325: t463_path5 = 1 -> x_Garage_Area <= 843
 ind_7326: t463_path6 = 1 -> x_Gr_Liv_Area >= 894
 ind_7327: t463_path6 = 1 -> const_Lot_Area >= 21370
 ind_7328: t463_path6 = 1 -> x_Garage_Area >= 843
 ind_7329: t464_path0 = 1 -> const_Foundation_PConc <= 1
 ind_7330: t464_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_7331: t464_path1 = 1 -> const_Foundation_PConc <= 1
 ind_7332: t464_path1 = 1 -> x_ex1_is_BrkFace >= 1
 ind_7333: t464_path2 = 1 -> const_Foundation_PConc >= 1
 ind_7334: t465_path0 = 1 -> Overall_Qual_calc <= 4
 ind_7335: t465_path0 = 1 -> x_1st_Flr_SF <= 813
 ind_7336: t465_path1 = 1 -> Overall_Qual_calc <= 4
 ind_7337: t465_path1 = 1 -> x_1st_Flr_SF >= 813
 ind_7338: t465_path2 = 1 -> Overall_Qual_calc >= 4
 ind_7339: t465_path2 = 1 -> x_Open_Porch_SF <= 189
 ind_7340: t465_path3 = 1 -> Overall_Qual_calc >= 4
 ind_7341: t465_path3 = 1 -> x_Open_Porch_SF >= 189
 ind_7342: t466_path0 = 1 -> Overall_Qual_calc <= 5
 ind_7343: t466_path0 = 1 -> const_Lot_Area <= 4270
 ind_7344: t466_path1 = 1 -> Overall_Qual_calc <= 5
 ind_7345: t466_path1 = 1 -> const_Lot_Area >= 4270
 ind_7346: t466_path1 = 1 -> x_Garage_Area <= 500
 ind_7347: t466_path2 = 1 -> Overall_Qual_calc <= 5
 ind_7348: t466_path2 = 1 -> const_Lot_Area >= 4270
 ind_7349: t466_path2 = 1 -> x_Garage_Area >= 500
 ind_7350: t466_path3 = 1 -> Overall_Qual_calc >= 5
 ind_7351: t466_path3 = 1 -> x_b2_is_ALQ <= 1
 ind_7352: t466_path4 = 1 -> Overall_Qual_calc >= 5
 ind_7353: t466_path4 = 1 -> x_b2_is_ALQ >= 1
 ind_7354: t470_path0 = 1 -> x_Exter_Qual <= 2
 ind_7355: t470_path1 = 1 -> x_Exter_Qual >= 2
 ind_7356: t470_path1 = 1 -> x_1st_Flr_SF <= 2200
 ind_7357: t470_path2 = 1 -> x_Exter_Qual >= 2
 ind_7358: t470_path2 = 1 -> x_1st_Flr_SF >= 2200
 ind_7359: t471_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_7360: t471_path1 = 1 -> const_Neighborhood_Edwards >= 1
 ind_7361: t471_path1 = 1 -> x_1st_Flr_SF <= 986
 ind_7362: t471_path2 = 1 -> const_Neighborhood_Edwards >= 1
 ind_7363: t471_path2 = 1 -> x_1st_Flr_SF >= 986
 ind_7364: t471_path2 = 1 -> x_1st_Flr_SF <= 1518
 ind_7365: t471_path3 = 1 -> const_Neighborhood_Edwards >= 1
 ind_7366: t471_path3 = 1 -> x_1st_Flr_SF >= 986
 ind_7367: t471_path3 = 1 -> x_1st_Flr_SF >= 1518
 ind_7368: t473_path0 = 1 -> x_mvt_is_Stone <= 1
 ind_7369: t473_path0 = 1 -> x_Bsmt_Unf_SF <= 1393
 ind_7370: t473_path1 = 1 -> x_mvt_is_Stone <= 1
 ind_7371: t473_path1 = 1 -> x_Bsmt_Unf_SF >= 1393
 ind_7372: t473_path2 = 1 -> x_mvt_is_Stone >= 1
 ind_7373: t474_path0 = 1 -> const_Total_Bsmt_SF <= 1824
 ind_7374: t474_path0 = 1 -> Overall_Qual_calc <= 4
 ind_7375: t474_path1 = 1 -> const_Total_Bsmt_SF <= 1824
 ind_7376: t474_path1 = 1 -> Overall_Qual_calc >= 4
 ind_7377: t474_path2 = 1 -> const_Total_Bsmt_SF >= 1824
 ind_7378: t475_path0 = 1 -> const_Total_Bsmt_SF <= 481
 ind_7379: t475_path0 = 1 -> const_Total_Bsmt_SF <= 245
 ind_7380: t475_path1 = 1 -> const_Total_Bsmt_SF <= 481
 ind_7381: t475_path1 = 1 -> const_Total_Bsmt_SF >= 245
 ind_7382: t475_path2 = 1 -> const_Total_Bsmt_SF >= 481
 ind_7383: t476_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_7384: t476_path0 = 1 -> x_Enclosed_Porch <= 91
 ind_7385: t476_path0 = 1 -> const_Overall_Cond <= 6
 ind_7386: t476_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_7387: t476_path1 = 1 -> x_Enclosed_Porch <= 91
 ind_7388: t476_path1 = 1 -> const_Overall_Cond >= 6
 ind_7389: t476_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_7390: t476_path2 = 1 -> x_Enclosed_Porch >= 91
 ind_7391: t476_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_7392: t477_path0 = 1 -> x_Bsmt_Unf_SF <= 576
 ind_7393: t477_path0 = 1 -> x_elect_is_SBrkr <= 1
 ind_7394: t477_path1 = 1 -> x_Bsmt_Unf_SF <= 576
 ind_7395: t477_path1 = 1 -> x_elect_is_SBrkr >= 1
 ind_7396: t477_path2 = 1 -> x_Bsmt_Unf_SF >= 576
 ind_7397: t477_path2 = 1 -> const_2nd_Flr_SF <= 581
 ind_7398: t477_path3 = 1 -> x_Bsmt_Unf_SF >= 576
 ind_7399: t477_path3 = 1 -> const_2nd_Flr_SF >= 581
 ind_7400: t478_path0 = 1 -> x_Enclosed_Porch <= 154
 ind_7401: t478_path1 = 1 -> x_Enclosed_Porch >= 154
 ind_7402: t478_path1 = 1 -> const_2nd_Flr_SF <= 308
 ind_7403: t478_path1 = 1 -> x_Bsmt_Unf_SF <= 28
 ind_7404: t478_path2 = 1 -> x_Enclosed_Porch >= 154
 ind_7405: t478_path2 = 1 -> const_2nd_Flr_SF <= 308
 ind_7406: t478_path2 = 1 -> x_Bsmt_Unf_SF >= 28
 ind_7407: t478_path3 = 1 -> x_Enclosed_Porch >= 154
 ind_7408: t478_path3 = 1 -> const_2nd_Flr_SF >= 308
 ind_7409: t479_path0 = 1 -> const_Neighborhood_Edwards <= 1
 ind_7410: t479_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7411: t479_path0 = 1 -> x_Gr_Liv_Area <= 894
 ind_7412: t479_path1 = 1 -> const_Neighborhood_Edwards <= 1
 ind_7413: t479_path1 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7414: t479_path1 = 1 -> x_Gr_Liv_Area >= 894
 ind_7415: t479_path2 = 1 -> const_Neighborhood_Edwards <= 1
 ind_7416: t479_path2 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7417: t479_path3 = 1 -> const_Neighborhood_Edwards >= 1
 ind_7418: t479_path3 = 1 -> x_Garage_Area <= 610
 ind_7419: t479_path3 = 1 -> const_Lot_Shape_IR1 <= 1
 ind_7420: t479_path4 = 1 -> const_Neighborhood_Edwards >= 1
 ind_7421: t479_path4 = 1 -> x_Garage_Area <= 610
 ind_7422: t479_path4 = 1 -> const_Lot_Shape_IR1 >= 1
 ind_7423: t479_path5 = 1 -> const_Neighborhood_Edwards >= 1
 ind_7424: t479_path5 = 1 -> x_Garage_Area >= 610
 ind_7425: t480_path0 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_7426: t480_path0 = 1 -> x_Open_Porch_SF <= 168
 ind_7427: t480_path0 = 1 -> x_Bsmt_Unf_SF <= 1373
 ind_7428: t480_path1 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_7429: t480_path1 = 1 -> x_Open_Porch_SF <= 168
 ind_7430: t480_path1 = 1 -> x_Bsmt_Unf_SF >= 1373
 ind_7431: t480_path2 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_7432: t480_path2 = 1 -> x_Open_Porch_SF >= 168
 ind_7433: t480_path2 = 1 -> x_TotRms_AbvGrd <= 9
 ind_7434: t480_path3 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_7435: t480_path3 = 1 -> x_Open_Porch_SF >= 168
 ind_7436: t480_path3 = 1 -> x_TotRms_AbvGrd >= 9
 ind_7437: t480_path4 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_7438: t481_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_7439: t481_path1 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_7440: t484_path0 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_7441: t484_path0 = 1 -> const_Garage_Type_CarPort <= 1
 ind_7442: t484_path0 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_7443: t484_path1 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_7444: t484_path1 = 1 -> const_Garage_Type_CarPort <= 1
 ind_7445: t484_path1 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_7446: t484_path2 = 1 -> const_Total_Bsmt_SF <= 2492
 ind_7447: t484_path2 = 1 -> const_Garage_Type_CarPort >= 1
 ind_7448: t484_path3 = 1 -> const_Total_Bsmt_SF >= 2492
 ind_7449: t488_path0 = 1 -> x_Garage_Area <= 308
 ind_7450: t488_path0 = 1 -> const_Sale_Condition_Normal <= 1
 ind_7451: t488_path1 = 1 -> x_Garage_Area <= 308
 ind_7452: t488_path1 = 1 -> const_Sale_Condition_Normal >= 1
 ind_7453: t488_path1 = 1 -> Overall_Qual_calc <= 3
 ind_7454: t488_path2 = 1 -> x_Garage_Area <= 308
 ind_7455: t488_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_7456: t488_path2 = 1 -> Overall_Qual_calc >= 3
 ind_7457: t488_path3 = 1 -> x_Garage_Area >= 308
 ind_7458: t490_path0 = 1 -> x_Open_Porch_SF <= 164
 ind_7459: t490_path0 = 1 -> x_paved_drive_is_P <= 1
 ind_7460: t490_path1 = 1 -> x_Open_Porch_SF <= 164
 ind_7461: t490_path1 = 1 -> x_paved_drive_is_P >= 1
 ind_7462: t490_path2 = 1 -> x_Open_Porch_SF >= 164
 ind_7463: t491_path0 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_7464: t491_path0 = 1 -> x_Bedroom_AbvGr <= 3
 ind_7465: t491_path0 = 1 -> const_Lot_Area <= 5436
 ind_7466: t491_path1 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_7467: t491_path1 = 1 -> x_Bedroom_AbvGr <= 3
 ind_7468: t491_path1 = 1 -> const_Lot_Area >= 5436
 ind_7469: t491_path2 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_7470: t491_path2 = 1 -> x_Bedroom_AbvGr >= 3
 ind_7471: t491_path3 = 1 -> x_Bsmt_Unf_SF >= 476
 ind_7472: t492_path0 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_7473: t492_path0 = 1 -> x_Enclosed_Porch <= 265
 ind_7474: t492_path1 = 1 -> const_Neighborhood_ClearCr <= 1
 ind_7475: t492_path1 = 1 -> x_Enclosed_Porch >= 265
 ind_7476: t492_path2 = 1 -> const_Neighborhood_ClearCr >= 1
 ind_7477: t493_path0 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7478: t493_path0 = 1 -> x_Garage_Area <= 252
 ind_7479: t493_path0 = 1 -> const_Lot_Shape_Reg <= 1
 ind_7480: t493_path1 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7481: t493_path1 = 1 -> x_Garage_Area <= 252
 ind_7482: t493_path1 = 1 -> const_Lot_Shape_Reg >= 1
 ind_7483: t493_path2 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7484: t493_path2 = 1 -> x_Garage_Area >= 252
 ind_7485: t493_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_7486: t493_path3 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7487: t493_path3 = 1 -> x_Garage_Area >= 252
 ind_7488: t493_path3 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_7489: t493_path4 = 1 -> x_Kitchen_AbvGr >= 2
 ind_7490: t494_path0 = 1 -> const_Roof_Style_Hip <= 1
 ind_7491: t494_path1 = 1 -> const_Roof_Style_Hip >= 1
 ind_7492: t494_path1 = 1 -> const_Land_Contour_Bnk <= 1
 ind_7493: t494_path2 = 1 -> const_Roof_Style_Hip >= 1
 ind_7494: t494_path2 = 1 -> const_Land_Contour_Bnk >= 1
 ind_7495: t495_path0 = 1 -> Overall_Qual_calc <= 3
 ind_7496: t495_path1 = 1 -> Overall_Qual_calc >= 3
 ind_7497: t496_path0 = 1 -> Overall_Qual_calc <= 3
 ind_7498: t496_path1 = 1 -> Overall_Qual_calc >= 3
 ind_7499: t497_path0 = 1 -> const_Lot_Frontage <= 117
 ind_7500: t497_path0 = 1 -> x_Bsmt_Cond <= 2
 ind_7501: t497_path0 = 1 -> x_Enclosed_Porch <= 105
 ind_7502: t497_path1 = 1 -> const_Lot_Frontage <= 117
 ind_7503: t497_path1 = 1 -> x_Bsmt_Cond <= 2
 ind_7504: t497_path1 = 1 -> x_Enclosed_Porch >= 105
 ind_7505: t497_path2 = 1 -> const_Lot_Frontage <= 117
 ind_7506: t497_path2 = 1 -> x_Bsmt_Cond >= 2
 ind_7507: t497_path3 = 1 -> const_Lot_Frontage >= 117
 ind_7508: t498_path0 = 1 -> const_Year_Built <= 2005
 ind_7509: t498_path0 = 1 -> Overall_Qual_calc <= 7
 ind_7510: t498_path0 = 1 -> x_TotRms_AbvGrd <= 8
 ind_7511: t498_path1 = 1 -> const_Year_Built <= 2005
 ind_7512: t498_path1 = 1 -> Overall_Qual_calc <= 7
 ind_7513: t498_path1 = 1 -> x_TotRms_AbvGrd >= 8
 ind_7514: t498_path2 = 1 -> const_Year_Built <= 2005
 ind_7515: t498_path2 = 1 -> Overall_Qual_calc >= 7
 ind_7516: t498_path2 = 1 -> const_Lot_Area <= 25286
 ind_7517: t498_path3 = 1 -> const_Year_Built <= 2005
 ind_7518: t498_path3 = 1 -> Overall_Qual_calc >= 7
 ind_7519: t498_path3 = 1 -> const_Lot_Area >= 25286
 ind_7520: t498_path4 = 1 -> const_Year_Built >= 2005
 ind_7521: t499_path0 = 1 -> const_Neighborhood_BrkSide <= 1
 ind_7522: t499_path1 = 1 -> const_Neighborhood_BrkSide >= 1
 ind_7523: t500_path0 = 1 -> x_Garage_Area <= 888
 ind_7524: t500_path0 = 1 -> const_Land_Contour_Low <= 1
 ind_7525: t500_path1 = 1 -> x_Garage_Area <= 888
 ind_7526: t500_path1 = 1 -> const_Land_Contour_Low >= 1
 ind_7527: t500_path1 = 1 -> x_ex1_is_Plywood <= 1
 ind_7528: t500_path2 = 1 -> x_Garage_Area <= 888
 ind_7529: t500_path2 = 1 -> const_Land_Contour_Low >= 1
 ind_7530: t500_path2 = 1 -> x_ex1_is_Plywood >= 1
 ind_7531: t500_path3 = 1 -> x_Garage_Area >= 888
 ind_7532: t501_path0 = 1 -> x_Gr_Liv_Area <= 876
 ind_7533: t501_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_7534: t501_path0 = 1 -> const_Lot_Shape_Reg <= 1
 ind_7535: t501_path1 = 1 -> x_Gr_Liv_Area <= 876
 ind_7536: t501_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_7537: t501_path1 = 1 -> const_Lot_Shape_Reg >= 1
 ind_7538: t501_path2 = 1 -> x_Gr_Liv_Area <= 876
 ind_7539: t501_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_7540: t501_path3 = 1 -> x_Gr_Liv_Area >= 876
 ind_7541: t501_path3 = 1 -> x_Garage_Area <= 644
 ind_7542: t501_path3 = 1 -> const_Garage_Type_CarPort <= 1
 ind_7543: t501_path4 = 1 -> x_Gr_Liv_Area >= 876
 ind_7544: t501_path4 = 1 -> x_Garage_Area <= 644
 ind_7545: t501_path4 = 1 -> const_Garage_Type_CarPort >= 1
 ind_7546: t501_path5 = 1 -> x_Gr_Liv_Area >= 876
 ind_7547: t501_path5 = 1 -> x_Garage_Area >= 644
 ind_7548: t501_path5 = 1 -> x_Gr_Liv_Area <= 1779
 ind_7549: t501_path6 = 1 -> x_Gr_Liv_Area >= 876
 ind_7550: t501_path6 = 1 -> x_Garage_Area >= 644
 ind_7551: t501_path6 = 1 -> x_Gr_Liv_Area >= 1779
 ind_7552: t502_path0 = 1 -> Overall_Qual_calc <= 3
 ind_7553: t502_path1 = 1 -> Overall_Qual_calc >= 3
 ind_7554: t504_path0 = 1 -> const_Bsmt_Qual <= 0
 ind_7555: t504_path0 = 1 -> const_Lot_Frontage <= 66
 ind_7556: t504_path1 = 1 -> const_Bsmt_Qual <= 0
 ind_7557: t504_path1 = 1 -> const_Lot_Frontage >= 66
 ind_7558: t504_path1 = 1 -> const_Mo_Sold_6 <= 1
 ind_7559: t504_path2 = 1 -> const_Bsmt_Qual <= 0
 ind_7560: t504_path2 = 1 -> const_Lot_Frontage >= 66
 ind_7561: t504_path2 = 1 -> const_Mo_Sold_6 >= 1
 ind_7562: t504_path3 = 1 -> const_Bsmt_Qual >= 0
 ind_7563: t505_path0 = 1 -> x_Open_Porch_SF <= 93
 ind_7564: t505_path1 = 1 -> x_Open_Porch_SF >= 93
 ind_7565: t506_path0 = 1 -> const_Lot_Area <= 16779
 ind_7566: t506_path0 = 1 -> const_Year_Remod_Add <= 2008
 ind_7567: t506_path0 = 1 -> x_Gr_Liv_Area <= 2633
 ind_7568: t506_path1 = 1 -> const_Lot_Area <= 16779
 ind_7569: t506_path1 = 1 -> const_Year_Remod_Add <= 2008
 ind_7570: t506_path1 = 1 -> x_Gr_Liv_Area >= 2633
 ind_7571: t506_path2 = 1 -> const_Lot_Area <= 16779
 ind_7572: t506_path2 = 1 -> const_Year_Remod_Add >= 2008
 ind_7573: t506_path3 = 1 -> const_Lot_Area >= 16779
 ind_7574: t506_path3 = 1 -> x_1st_Flr_SF <= 1402
 ind_7575: t506_path4 = 1 -> const_Lot_Area >= 16779
 ind_7576: t506_path4 = 1 -> x_1st_Flr_SF >= 1402
 ind_7577: t507_path0 = 1 -> x_BsmtFin_SF_2 <= 60
 ind_7578: t507_path1 = 1 -> x_BsmtFin_SF_2 >= 60
 ind_7579: t508_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7580: t508_path0 = 1 -> const_Roof_Style_Gable <= 1
 ind_7581: t508_path0 = 1 -> const_Yr_Sold <= 2007
 ind_7582: t508_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7583: t508_path1 = 1 -> const_Roof_Style_Gable <= 1
 ind_7584: t508_path1 = 1 -> const_Yr_Sold >= 2007
 ind_7585: t508_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7586: t508_path2 = 1 -> const_Roof_Style_Gable >= 1
 ind_7587: t508_path2 = 1 -> const_Mo_Sold_3 <= 1
 ind_7588: t508_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7589: t508_path3 = 1 -> const_Roof_Style_Gable >= 1
 ind_7590: t508_path3 = 1 -> const_Mo_Sold_3 >= 1
 ind_7591: t508_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_7592: t508_path4 = 1 -> x_Garage_Area <= 351
 ind_7593: t508_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_7594: t508_path5 = 1 -> x_Garage_Area >= 351
 ind_7595: t509_path0 = 1 -> const_2nd_Flr_SF <= 797
 ind_7596: t509_path0 = 1 -> const_Yr_Sold <= 2010
 ind_7597: t509_path1 = 1 -> const_2nd_Flr_SF <= 797
 ind_7598: t509_path1 = 1 -> const_Yr_Sold >= 2010
 ind_7599: t509_path2 = 1 -> const_2nd_Flr_SF >= 797
 ind_7600: t513_path0 = 1 -> x_Mas_Vnr_Area <= 788
 ind_7601: t513_path0 = 1 -> const_Overall_Cond <= 5
 ind_7602: t513_path0 = 1 -> x_Fireplace_Qu <= 1
 ind_7603: t513_path1 = 1 -> x_Mas_Vnr_Area <= 788
 ind_7604: t513_path1 = 1 -> const_Overall_Cond <= 5
 ind_7605: t513_path1 = 1 -> x_Fireplace_Qu >= 1
 ind_7606: t513_path2 = 1 -> x_Mas_Vnr_Area <= 788
 ind_7607: t513_path2 = 1 -> const_Overall_Cond >= 5
 ind_7608: t513_path2 = 1 -> x_ex1_is_BrkFace <= 1
 ind_7609: t513_path3 = 1 -> x_Mas_Vnr_Area <= 788
 ind_7610: t513_path3 = 1 -> const_Overall_Cond >= 5
 ind_7611: t513_path3 = 1 -> x_ex1_is_BrkFace >= 1
 ind_7612: t513_path4 = 1 -> x_Mas_Vnr_Area >= 788
 ind_7613: t513_path4 = 1 -> const_Lot_Area <= 14115
 ind_7614: t513_path5 = 1 -> x_Mas_Vnr_Area >= 788
 ind_7615: t513_path5 = 1 -> const_Lot_Area >= 14115
 ind_7616: t515_path0 = 1 -> x_b1_is_LwQ <= 1
 ind_7617: t515_path0 = 1 -> const_Year_Remod_Add <= 1954
 ind_7618: t515_path1 = 1 -> x_b1_is_LwQ <= 1
 ind_7619: t515_path1 = 1 -> const_Year_Remod_Add >= 1954
 ind_7620: t515_path1 = 1 -> const_Overall_Cond <= 4
 ind_7621: t515_path2 = 1 -> x_b1_is_LwQ <= 1
 ind_7622: t515_path2 = 1 -> const_Year_Remod_Add >= 1954
 ind_7623: t515_path2 = 1 -> const_Overall_Cond >= 4
 ind_7624: t515_path3 = 1 -> x_b1_is_LwQ >= 1
 ind_7625: t516_path0 = 1 -> const_Sale_Condition_Alloca <= 1
 ind_7626: t516_path0 = 1 -> Overall_Qual_calc <= 7
 ind_7627: t516_path0 = 1 -> x_b1_is_LwQ <= 1
 ind_7628: t516_path1 = 1 -> const_Sale_Condition_Alloca <= 1
 ind_7629: t516_path1 = 1 -> Overall_Qual_calc <= 7
 ind_7630: t516_path1 = 1 -> x_b1_is_LwQ >= 1
 ind_7631: t516_path2 = 1 -> const_Sale_Condition_Alloca <= 1
 ind_7632: t516_path2 = 1 -> Overall_Qual_calc >= 7
 ind_7633: t516_path2 = 1 -> const_Sale_Condition_Family <= 1
 ind_7634: t516_path3 = 1 -> const_Sale_Condition_Alloca <= 1
 ind_7635: t516_path3 = 1 -> Overall_Qual_calc >= 7
 ind_7636: t516_path3 = 1 -> const_Sale_Condition_Family >= 1
 ind_7637: t516_path4 = 1 -> const_Sale_Condition_Alloca >= 1
 ind_7638: t517_path0 = 1 -> x_Garage_Area <= 320
 ind_7639: t517_path0 = 1 -> x_BsmtFin_SF_1 <= 976
 ind_7640: t517_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7641: t517_path1 = 1 -> x_Garage_Area <= 320
 ind_7642: t517_path1 = 1 -> x_BsmtFin_SF_1 <= 976
 ind_7643: t517_path1 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7644: t517_path2 = 1 -> x_Garage_Area <= 320
 ind_7645: t517_path2 = 1 -> x_BsmtFin_SF_1 >= 976
 ind_7646: t517_path3 = 1 -> x_Garage_Area >= 320
 ind_7647: t517_path3 = 1 -> x_Gr_Liv_Area <= 883
 ind_7648: t517_path4 = 1 -> x_Garage_Area >= 320
 ind_7649: t517_path4 = 1 -> x_Gr_Liv_Area >= 883
 ind_7650: t517_path4 = 1 -> x_Garage_Area <= 871
 ind_7651: t517_path5 = 1 -> x_Garage_Area >= 320
 ind_7652: t517_path5 = 1 -> x_Gr_Liv_Area >= 883
 ind_7653: t517_path5 = 1 -> x_Garage_Area >= 871
 ind_7654: t518_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7655: t518_path0 = 1 -> const_Garage_Yr_Blt <= 1908
 ind_7656: t518_path0 = 1 -> const_Bsmt_Qual <= 2
 ind_7657: t518_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7658: t518_path1 = 1 -> const_Garage_Yr_Blt <= 1908
 ind_7659: t518_path1 = 1 -> const_Bsmt_Qual >= 2
 ind_7660: t518_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7661: t518_path2 = 1 -> const_Garage_Yr_Blt >= 1908
 ind_7662: t518_path2 = 1 -> x_Open_Porch_SF <= 188
 ind_7663: t518_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7664: t518_path3 = 1 -> const_Garage_Yr_Blt >= 1908
 ind_7665: t518_path3 = 1 -> x_Open_Porch_SF >= 188
 ind_7666: t518_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_7667: t518_path4 = 1 -> const_Total_Bsmt_SF <= 727
 ind_7668: t518_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_7669: t518_path5 = 1 -> const_Total_Bsmt_SF >= 727
 ind_7670: t519_path0 = 1 -> const_Overall_Cond <= 3
 ind_7671: t519_path1 = 1 -> const_Overall_Cond >= 3
 ind_7672: t520_path0 = 1 -> x_BsmtFin_SF_1 <= 1259
 ind_7673: t520_path0 = 1 -> const_Garage_Yr_Blt <= 1960
 ind_7674: t520_path0 = 1 -> x_1st_Flr_SF <= 798
 ind_7675: t520_path1 = 1 -> x_BsmtFin_SF_1 <= 1259
 ind_7676: t520_path1 = 1 -> const_Garage_Yr_Blt <= 1960
 ind_7677: t520_path1 = 1 -> x_1st_Flr_SF >= 798
 ind_7678: t520_path2 = 1 -> x_BsmtFin_SF_1 <= 1259
 ind_7679: t520_path2 = 1 -> const_Garage_Yr_Blt >= 1960
 ind_7680: t520_path3 = 1 -> x_BsmtFin_SF_1 >= 1259
 ind_7681: t520_path3 = 1 -> const_Lot_Area <= 15498
 ind_7682: t520_path4 = 1 -> x_BsmtFin_SF_1 >= 1259
 ind_7683: t520_path4 = 1 -> const_Lot_Area >= 15498
 ind_7684: t520_path4 = 1 -> const_Lot_Frontage <= 60
 ind_7685: t520_path5 = 1 -> x_BsmtFin_SF_1 >= 1259
 ind_7686: t520_path5 = 1 -> const_Lot_Area >= 15498
 ind_7687: t520_path5 = 1 -> const_Lot_Frontage >= 60
 ind_7688: t521_path0 = 1 -> Overall_Qual_calc <= 3
 ind_7689: t521_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_7690: t521_path1 = 1 -> Overall_Qual_calc <= 3
 ind_7691: t521_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_7692: t521_path2 = 1 -> Overall_Qual_calc >= 3
 ind_7693: t521_path2 = 1 -> x_ex1_is_BrkFace <= 1
 ind_7694: t521_path2 = 1 -> const_Lot_Area <= 13052
 ind_7695: t521_path3 = 1 -> Overall_Qual_calc >= 3
 ind_7696: t521_path3 = 1 -> x_ex1_is_BrkFace <= 1
 ind_7697: t521_path3 = 1 -> const_Lot_Area >= 13052
 ind_7698: t521_path4 = 1 -> Overall_Qual_calc >= 3
 ind_7699: t521_path4 = 1 -> x_ex1_is_BrkFace >= 1
 ind_7700: t522_path0 = 1 -> Overall_Qual_calc <= 3
 ind_7701: t522_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_7702: t522_path1 = 1 -> Overall_Qual_calc <= 3
 ind_7703: t522_path1 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_7704: t522_path2 = 1 -> Overall_Qual_calc >= 3
 ind_7705: t522_path2 = 1 -> x_ex2_is_MetalSd <= 1
 ind_7706: t522_path2 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7707: t522_path3 = 1 -> Overall_Qual_calc >= 3
 ind_7708: t522_path3 = 1 -> x_ex2_is_MetalSd <= 1
 ind_7709: t522_path3 = 1 -> x_Kitchen_AbvGr >= 2
 ind_7710: t522_path4 = 1 -> Overall_Qual_calc >= 3
 ind_7711: t522_path4 = 1 -> x_ex2_is_MetalSd >= 1
 ind_7712: t522_path4 = 1 -> const_Garage_Yr_Blt <= 2006
 ind_7713: t522_path5 = 1 -> Overall_Qual_calc >= 3
 ind_7714: t522_path5 = 1 -> x_ex2_is_MetalSd >= 1
 ind_7715: t522_path5 = 1 -> const_Garage_Yr_Blt >= 2006
 ind_7716: t523_path0 = 1 -> Overall_Qual_calc <= 5
 ind_7717: t523_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7718: t523_path0 = 1 -> x_mvt_is_No_aplica <= 1
 ind_7719: t523_path1 = 1 -> Overall_Qual_calc <= 5
 ind_7720: t523_path1 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7721: t523_path1 = 1 -> x_mvt_is_No_aplica >= 1
 ind_7722: t523_path2 = 1 -> Overall_Qual_calc <= 5
 ind_7723: t523_path2 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7724: t523_path2 = 1 -> const_Garage_Type_Detchd <= 1
 ind_7725: t523_path3 = 1 -> Overall_Qual_calc <= 5
 ind_7726: t523_path3 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7727: t523_path3 = 1 -> const_Garage_Type_Detchd >= 1
 ind_7728: t523_path4 = 1 -> Overall_Qual_calc >= 5
 ind_7729: t524_path0 = 1 -> x_Bsmt_Cond <= 1
 ind_7730: t524_path0 = 1 -> const_Mo_Sold_6 <= 1
 ind_7731: t524_path1 = 1 -> x_Bsmt_Cond <= 1
 ind_7732: t524_path1 = 1 -> const_Mo_Sold_6 >= 1
 ind_7733: t524_path2 = 1 -> x_Bsmt_Cond >= 1
 ind_7734: t524_path2 = 1 -> x_Full_Bath <= 2
 ind_7735: t524_path3 = 1 -> x_Bsmt_Cond >= 1
 ind_7736: t524_path3 = 1 -> x_Full_Bath >= 2
 ind_7737: t525_path0 = 1 -> const_MS_SubClass_190 <= 1
 ind_7738: t525_path1 = 1 -> const_MS_SubClass_190 >= 1
 ind_7739: t526_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7740: t526_path0 = 1 -> x_Bedroom_AbvGr <= 3
 ind_7741: t526_path0 = 1 -> const_Bsmt_Exposure_No <= 1
 ind_7742: t526_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7743: t526_path1 = 1 -> x_Bedroom_AbvGr <= 3
 ind_7744: t526_path1 = 1 -> const_Bsmt_Exposure_No >= 1
 ind_7745: t526_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7746: t526_path2 = 1 -> x_Bedroom_AbvGr >= 3
 ind_7747: t526_path2 = 1 -> x_Bsmt_Cond <= 3
 ind_7748: t526_path3 = 1 -> x_ex1_is_AsbShng <= 1
 ind_7749: t526_path3 = 1 -> x_Bedroom_AbvGr >= 3
 ind_7750: t526_path3 = 1 -> x_Bsmt_Cond >= 3
 ind_7751: t526_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_7752: t526_path4 = 1 -> const_Lot_Frontage <= 66
 ind_7753: t526_path5 = 1 -> x_ex1_is_AsbShng >= 1
 ind_7754: t526_path5 = 1 -> const_Lot_Frontage >= 66
 ind_7755: t527_path0 = 1 -> const_Lot_Area <= 21780
 ind_7756: t527_path0 = 1 -> central_air_no <= 1
 ind_7757: t527_path0 = 1 -> x_b1_is_GLQ <= 1
 ind_7758: t527_path1 = 1 -> const_Lot_Area <= 21780
 ind_7759: t527_path1 = 1 -> central_air_no <= 1
 ind_7760: t527_path1 = 1 -> x_b1_is_GLQ >= 1
 ind_7761: t527_path2 = 1 -> const_Lot_Area <= 21780
 ind_7762: t527_path2 = 1 -> central_air_no >= 1
 ind_7763: t527_path2 = 1 -> const_MS_Zoning_RL <= 1
 ind_7764: t527_path3 = 1 -> const_Lot_Area <= 21780
 ind_7765: t527_path3 = 1 -> central_air_no >= 1
 ind_7766: t527_path3 = 1 -> const_MS_Zoning_RL >= 1
 ind_7767: t527_path4 = 1 -> const_Lot_Area >= 21780
 ind_7768: t527_path4 = 1 -> x_b1_is_GLQ <= 1
 ind_7769: t527_path5 = 1 -> const_Lot_Area >= 21780
 ind_7770: t527_path5 = 1 -> x_b1_is_GLQ >= 1
 ind_7771: t528_path0 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_7772: t528_path1 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_7773: t528_path1 = 1 -> const_Sale_Type_New <= 1
 ind_7774: t528_path2 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_7775: t528_path2 = 1 -> const_Sale_Type_New >= 1
 ind_7776: t529_path0 = 1 -> x_mvt_is_Stone <= 1
 ind_7777: t529_path0 = 1 -> x_Garage_Area <= 320
 ind_7778: t529_path0 = 1 -> central_air_yes <= 1
 ind_7779: t529_path1 = 1 -> x_mvt_is_Stone <= 1
 ind_7780: t529_path1 = 1 -> x_Garage_Area <= 320
 ind_7781: t529_path1 = 1 -> central_air_yes >= 1
 ind_7782: t529_path2 = 1 -> x_mvt_is_Stone <= 1
 ind_7783: t529_path2 = 1 -> x_Garage_Area >= 320
 ind_7784: t529_path2 = 1 -> x_Wood_Deck_SF <= 274
 ind_7785: t529_path3 = 1 -> x_mvt_is_Stone <= 1
 ind_7786: t529_path3 = 1 -> x_Garage_Area >= 320
 ind_7787: t529_path3 = 1 -> x_Wood_Deck_SF >= 274
 ind_7788: t529_path4 = 1 -> x_mvt_is_Stone >= 1
 ind_7789: t531_path0 = 1 -> const_Year_Built <= 1914
 ind_7790: t531_path0 = 1 -> const_Bsmt_Qual <= 2
 ind_7791: t531_path1 = 1 -> const_Year_Built <= 1914
 ind_7792: t531_path1 = 1 -> const_Bsmt_Qual >= 2
 ind_7793: t531_path2 = 1 -> const_Year_Built >= 1914
 ind_7794: t533_path0 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_7795: t533_path1 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_7796: t533_path1 = 1 -> const_Lot_Area <= 6600
 ind_7797: t533_path2 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_7798: t533_path2 = 1 -> const_Lot_Area >= 6600
 ind_7799: t535_path0 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_7800: t535_path0 = 1 -> const_Lot_Area <= 21370
 ind_7801: t535_path0 = 1 -> x_BsmtFin_SF_1 <= 1965
 ind_7802: t535_path1 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_7803: t535_path1 = 1 -> const_Lot_Area <= 21370
 ind_7804: t535_path1 = 1 -> x_BsmtFin_SF_1 >= 1965
 ind_7805: t535_path2 = 1 -> x_Bsmt_Unf_SF <= 476
 ind_7806: t535_path2 = 1 -> const_Lot_Area >= 21370
 ind_7807: t535_path3 = 1 -> x_Bsmt_Unf_SF >= 476
 ind_7808: t535_path3 = 1 -> const_2nd_Flr_SF <= 1371
 ind_7809: t535_path4 = 1 -> x_Bsmt_Unf_SF >= 476
 ind_7810: t535_path4 = 1 -> const_2nd_Flr_SF >= 1371
 ind_7811: t537_path0 = 1 -> const_Lot_Shape_IR3 <= 1
 ind_7812: t537_path1 = 1 -> const_Lot_Shape_IR3 >= 1
 ind_7813: t538_path0 = 1 -> Overall_Qual_calc <= 7
 ind_7814: t538_path0 = 1 -> x_BsmtFin_SF_1 <= 918
 ind_7815: t538_path1 = 1 -> Overall_Qual_calc <= 7
 ind_7816: t538_path1 = 1 -> x_BsmtFin_SF_1 >= 918
 ind_7817: t538_path2 = 1 -> Overall_Qual_calc >= 7
 ind_7818: t538_path2 = 1 -> x_Bsmt_Unf_SF <= 1795
 ind_7819: t538_path2 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_7820: t538_path3 = 1 -> Overall_Qual_calc >= 7
 ind_7821: t538_path3 = 1 -> x_Bsmt_Unf_SF <= 1795
 ind_7822: t538_path3 = 1 -> const_Bsmt_Exposure_Av >= 1
 ind_7823: t538_path4 = 1 -> Overall_Qual_calc >= 7
 ind_7824: t538_path4 = 1 -> x_Bsmt_Unf_SF >= 1795
 ind_7825: t539_path0 = 1 -> const_Yr_Sold <= 2007
 ind_7826: t539_path1 = 1 -> const_Yr_Sold >= 2007
 ind_7827: t540_path0 = 1 -> const_Mo_Sold_7 <= 1
 ind_7828: t540_path0 = 1 -> x_Garage_Cond <= 1
 ind_7829: t540_path0 = 1 -> const_Lot_Area <= 8800
 ind_7830: t540_path1 = 1 -> const_Mo_Sold_7 <= 1
 ind_7831: t540_path1 = 1 -> x_Garage_Cond <= 1
 ind_7832: t540_path1 = 1 -> const_Lot_Area >= 8800
 ind_7833: t540_path2 = 1 -> const_Mo_Sold_7 <= 1
 ind_7834: t540_path2 = 1 -> x_Garage_Cond >= 1
 ind_7835: t540_path2 = 1 -> Overall_Qual_calc <= 3
 ind_7836: t540_path3 = 1 -> const_Mo_Sold_7 <= 1
 ind_7837: t540_path3 = 1 -> x_Garage_Cond >= 1
 ind_7838: t540_path3 = 1 -> Overall_Qual_calc >= 3
 ind_7839: t540_path4 = 1 -> const_Mo_Sold_7 >= 1
 ind_7840: t541_path0 = 1 -> x_Exter_Qual <= 2
 ind_7841: t541_path0 = 1 -> x_Garage_Area <= 297
 ind_7842: t541_path1 = 1 -> x_Exter_Qual <= 2
 ind_7843: t541_path1 = 1 -> x_Garage_Area >= 297
 ind_7844: t541_path2 = 1 -> x_Exter_Qual >= 2
 ind_7845: t543_path0 = 1 -> x_Exter_Qual <= 3
 ind_7846: t543_path0 = 1 -> x_Gr_Liv_Area <= 1734
 ind_7847: t543_path0 = 1 -> const_Alley_No_aplica <= 1
 ind_7848: t543_path1 = 1 -> x_Exter_Qual <= 3
 ind_7849: t543_path1 = 1 -> x_Gr_Liv_Area <= 1734
 ind_7850: t543_path1 = 1 -> const_Alley_No_aplica >= 1
 ind_7851: t543_path2 = 1 -> x_Exter_Qual <= 3
 ind_7852: t543_path2 = 1 -> x_Gr_Liv_Area >= 1734
 ind_7853: t543_path2 = 1 -> const_MS_Zoning_RM <= 1
 ind_7854: t543_path3 = 1 -> x_Exter_Qual <= 3
 ind_7855: t543_path3 = 1 -> x_Gr_Liv_Area >= 1734
 ind_7856: t543_path3 = 1 -> const_MS_Zoning_RM >= 1
 ind_7857: t543_path4 = 1 -> x_Exter_Qual >= 3
 ind_7858: t543_path4 = 1 -> const_Land_Contour_Bnk <= 1
 ind_7859: t543_path4 = 1 -> const_Garage_Yr_Blt <= 1993
 ind_7860: t543_path5 = 1 -> x_Exter_Qual >= 3
 ind_7861: t543_path5 = 1 -> const_Land_Contour_Bnk <= 1
 ind_7862: t543_path5 = 1 -> const_Garage_Yr_Blt >= 1993
 ind_7863: t543_path6 = 1 -> x_Exter_Qual >= 3
 ind_7864: t543_path6 = 1 -> const_Land_Contour_Bnk >= 1
 ind_7865: t545_path0 = 1 -> const_Lot_Frontage <= 115
 ind_7866: t545_path0 = 1 -> const_Foundation_CBlock <= 1
 ind_7867: t545_path1 = 1 -> const_Lot_Frontage <= 115
 ind_7868: t545_path1 = 1 -> const_Foundation_CBlock >= 1
 ind_7869: t545_path1 = 1 -> x_Fireplace_Qu <= 0
 ind_7870: t545_path2 = 1 -> const_Lot_Frontage <= 115
 ind_7871: t545_path2 = 1 -> const_Foundation_CBlock >= 1
 ind_7872: t545_path2 = 1 -> x_Fireplace_Qu >= 0
 ind_7873: t545_path3 = 1 -> const_Lot_Frontage >= 115
 ind_7874: t545_path3 = 1 -> x_Exter_Qual <= 3
 ind_7875: t545_path4 = 1 -> const_Lot_Frontage >= 115
 ind_7876: t545_path4 = 1 -> x_Exter_Qual >= 3
 ind_7877: t546_path0 = 1 -> const_Condition_1_PosA <= 1
 ind_7878: t546_path1 = 1 -> const_Condition_1_PosA >= 1
 ind_7879: t547_path0 = 1 -> const_Lot_Frontage <= 34
 ind_7880: t547_path1 = 1 -> const_Lot_Frontage >= 34
 ind_7881: t548_path0 = 1 -> x_elect_is_FuseA <= 1
 ind_7882: t548_path1 = 1 -> x_elect_is_FuseA >= 1
 ind_7883: t548_path1 = 1 -> const_Misc_Val <= 54
 ind_7884: t548_path1 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7885: t548_path2 = 1 -> x_elect_is_FuseA >= 1
 ind_7886: t548_path2 = 1 -> const_Misc_Val <= 54
 ind_7887: t548_path2 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7888: t548_path3 = 1 -> x_elect_is_FuseA >= 1
 ind_7889: t548_path3 = 1 -> const_Misc_Val >= 54
 ind_7890: t549_path0 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7891: t549_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_7892: t549_path0 = 1 -> x_1st_Flr_SF <= 712
 ind_7893: t549_path1 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7894: t549_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_7895: t549_path1 = 1 -> x_1st_Flr_SF >= 712
 ind_7896: t549_path2 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7897: t549_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_7898: t549_path2 = 1 -> Overall_Qual_calc <= 5
 ind_7899: t549_path3 = 1 -> x_Kitchen_AbvGr <= 2
 ind_7900: t549_path3 = 1 -> x_Gr_Liv_Area >= 747
 ind_7901: t549_path3 = 1 -> Overall_Qual_calc >= 5
 ind_7902: t549_path4 = 1 -> x_Kitchen_AbvGr >= 2
 ind_7903: t549_path4 = 1 -> x_Exter_Cond <= 2
 ind_7904: t549_path5 = 1 -> x_Kitchen_AbvGr >= 2
 ind_7905: t549_path5 = 1 -> x_Exter_Cond >= 2
 ind_7906: t550_path0 = 1 -> x_Heating_QC <= 2
 ind_7907: t550_path1 = 1 -> x_Heating_QC >= 2
 ind_7908: t550_path1 = 1 -> x_Mas_Vnr_Area <= 922
 ind_7909: t550_path1 = 1 -> const_Lot_Frontage <= 85
 ind_7910: t550_path2 = 1 -> x_Heating_QC >= 2
 ind_7911: t550_path2 = 1 -> x_Mas_Vnr_Area <= 922
 ind_7912: t550_path2 = 1 -> const_Lot_Frontage >= 85
 ind_7913: t550_path3 = 1 -> x_Heating_QC >= 2
 ind_7914: t550_path3 = 1 -> x_Mas_Vnr_Area >= 922
 ind_7915: t552_path0 = 1 -> const_Mo_Sold_7 <= 1
 ind_7916: t552_path0 = 1 -> const_Lot_Area <= 19522
 ind_7917: t552_path0 = 1 -> const_Total_Bsmt_SF <= 245
 ind_7918: t552_path1 = 1 -> const_Mo_Sold_7 <= 1
 ind_7919: t552_path1 = 1 -> const_Lot_Area <= 19522
 ind_7920: t552_path1 = 1 -> const_Total_Bsmt_SF >= 245
 ind_7921: t552_path2 = 1 -> const_Mo_Sold_7 <= 1
 ind_7922: t552_path2 = 1 -> const_Lot_Area >= 19522
 ind_7923: t552_path2 = 1 -> const_Lot_Frontage <= 174
 ind_7924: t552_path3 = 1 -> const_Mo_Sold_7 <= 1
 ind_7925: t552_path3 = 1 -> const_Lot_Area >= 19522
 ind_7926: t552_path3 = 1 -> const_Lot_Frontage >= 174
 ind_7927: t552_path4 = 1 -> const_Mo_Sold_7 >= 1
 ind_7928: t553_path0 = 1 -> const_Overall_Cond <= 6
 ind_7929: t553_path0 = 1 -> const_Year_Built <= 1953
 ind_7930: t553_path0 = 1 -> const_Year_Built <= 1902
 ind_7931: t553_path1 = 1 -> const_Overall_Cond <= 6
 ind_7932: t553_path1 = 1 -> const_Year_Built <= 1953
 ind_7933: t553_path1 = 1 -> const_Year_Built >= 1902
 ind_7934: t553_path2 = 1 -> const_Overall_Cond <= 6
 ind_7935: t553_path2 = 1 -> const_Year_Built >= 1953
 ind_7936: t553_path3 = 1 -> const_Overall_Cond >= 6
 ind_7937: t556_path0 = 1 -> x_Gr_Liv_Area <= 768
 ind_7938: t556_path1 = 1 -> x_Gr_Liv_Area >= 768
 ind_7939: t557_path0 = 1 -> const_Lot_Area <= 6000
 ind_7940: t557_path1 = 1 -> const_Lot_Area >= 6000
 ind_7941: t557_path1 = 1 -> const_Fireplaces <= 3
 ind_7942: t557_path1 = 1 -> x_TotRms_AbvGrd <= 5
 ind_7943: t557_path2 = 1 -> const_Lot_Area >= 6000
 ind_7944: t557_path2 = 1 -> const_Fireplaces <= 3
 ind_7945: t557_path2 = 1 -> x_TotRms_AbvGrd >= 5
 ind_7946: t557_path3 = 1 -> const_Lot_Area >= 6000
 ind_7947: t557_path3 = 1 -> const_Fireplaces >= 3
 ind_7948: t558_path0 = 1 -> x_Garage_Area <= 888
 ind_7949: t558_path0 = 1 -> const_Condition_1_Artery <= 1
 ind_7950: t558_path0 = 1 -> const_Garage_Cars <= 3
 ind_7951: t558_path1 = 1 -> x_Garage_Area <= 888
 ind_7952: t558_path1 = 1 -> const_Condition_1_Artery <= 1
 ind_7953: t558_path1 = 1 -> const_Garage_Cars >= 3
 ind_7954: t558_path2 = 1 -> x_Garage_Area <= 888
 ind_7955: t558_path2 = 1 -> const_Condition_1_Artery >= 1
 ind_7956: t558_path3 = 1 -> x_Garage_Area >= 888
 ind_7957: t559_path0 = 1 -> x_Wood_Deck_SF <= 216
 ind_7958: t559_path0 = 1 -> const_Overall_Cond <= 6
 ind_7959: t559_path0 = 1 -> x_b1_is_Rec <= 1
 ind_7960: t559_path1 = 1 -> x_Wood_Deck_SF <= 216
 ind_7961: t559_path1 = 1 -> const_Overall_Cond <= 6
 ind_7962: t559_path1 = 1 -> x_b1_is_Rec >= 1
 ind_7963: t559_path2 = 1 -> x_Wood_Deck_SF <= 216
 ind_7964: t559_path2 = 1 -> const_Overall_Cond >= 6
 ind_7965: t559_path2 = 1 -> const_Lot_Area <= 8626
 ind_7966: t559_path3 = 1 -> x_Wood_Deck_SF <= 216
 ind_7967: t559_path3 = 1 -> const_Overall_Cond >= 6
 ind_7968: t559_path3 = 1 -> const_Lot_Area >= 8626
 ind_7969: t559_path4 = 1 -> x_Wood_Deck_SF >= 216
 ind_7970: t560_path0 = 1 -> x_ex1_is_MetalSd <= 1
 ind_7971: t560_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_7972: t560_path0 = 1 -> const_Lot_Area <= 12853
 ind_7973: t560_path1 = 1 -> x_ex1_is_MetalSd <= 1
 ind_7974: t560_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_7975: t560_path1 = 1 -> const_Lot_Area >= 12853
 ind_7976: t560_path2 = 1 -> x_ex1_is_MetalSd <= 1
 ind_7977: t560_path2 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_7978: t560_path3 = 1 -> x_ex1_is_MetalSd >= 1
 ind_7979: t561_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_7980: t561_path0 = 1 -> x_1st_Flr_SF <= 717
 ind_7981: t561_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_7982: t561_path1 = 1 -> x_1st_Flr_SF >= 717
 ind_7983: t561_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_7984: t562_path0 = 1 -> x_1st_Flr_SF <= 1502
 ind_7985: t562_path1 = 1 -> x_1st_Flr_SF >= 1502
 ind_7986: t562_path1 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_7987: t562_path2 = 1 -> x_1st_Flr_SF >= 1502
 ind_7988: t562_path2 = 1 -> const_Bsmt_Exposure_Av >= 1
 ind_7989: t564_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_7990: t564_path1 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7991: t564_path1 = 1 -> const_Total_Bsmt_SF <= 727
 ind_7992: t564_path2 = 1 -> x_ex2_is_AsbShng >= 1
 ind_7993: t564_path2 = 1 -> const_Total_Bsmt_SF >= 727
 ind_7994: t566_path0 = 1 -> const_Year_Remod_Add <= 1958
 ind_7995: t566_path0 = 1 -> const_Total_Bsmt_SF <= 712
 ind_7996: t566_path0 = 1 -> const_Lot_Area <= 8094
 ind_7997: t566_path1 = 1 -> const_Year_Remod_Add <= 1958
 ind_7998: t566_path1 = 1 -> const_Total_Bsmt_SF <= 712
 ind_7999: t566_path1 = 1 -> const_Lot_Area >= 8094
 ind_8000: t566_path2 = 1 -> const_Year_Remod_Add <= 1958
 ind_8001: t566_path2 = 1 -> const_Total_Bsmt_SF >= 712
 ind_8002: t566_path2 = 1 -> x_Half_Bath <= 1
 ind_8003: t566_path3 = 1 -> const_Year_Remod_Add <= 1958
 ind_8004: t566_path3 = 1 -> const_Total_Bsmt_SF >= 712
 ind_8005: t566_path3 = 1 -> x_Half_Bath >= 1
 ind_8006: t566_path4 = 1 -> const_Year_Remod_Add >= 1958
 ind_8007: t566_path4 = 1 -> const_Overall_Cond <= 4
 ind_8008: t566_path5 = 1 -> const_Year_Remod_Add >= 1958
 ind_8009: t566_path5 = 1 -> const_Overall_Cond >= 4
 ind_8010: t567_path0 = 1 -> const_Overall_Cond <= 4
 ind_8011: t567_path0 = 1 -> const_Year_Remod_Add <= 1951
 ind_8012: t567_path1 = 1 -> const_Overall_Cond <= 4
 ind_8013: t567_path1 = 1 -> const_Year_Remod_Add >= 1951
 ind_8014: t567_path1 = 1 -> x_ex2_is_HdBoard <= 1
 ind_8015: t567_path2 = 1 -> const_Overall_Cond <= 4
 ind_8016: t567_path2 = 1 -> const_Year_Remod_Add >= 1951
 ind_8017: t567_path2 = 1 -> x_ex2_is_HdBoard >= 1
 ind_8018: t567_path3 = 1 -> const_Overall_Cond >= 4
 ind_8019: t567_path3 = 1 -> x_Exter_Qual <= 3
 ind_8020: t567_path3 = 1 -> const_Alley_Pave <= 1
 ind_8021: t567_path4 = 1 -> const_Overall_Cond >= 4
 ind_8022: t567_path4 = 1 -> x_Exter_Qual <= 3
 ind_8023: t567_path4 = 1 -> const_Alley_Pave >= 1
 ind_8024: t567_path5 = 1 -> const_Overall_Cond >= 4
 ind_8025: t567_path5 = 1 -> x_Exter_Qual >= 3
 ind_8026: t568_path0 = 1 -> x_TotRms_AbvGrd <= 11
 ind_8027: t568_path0 = 1 -> const_Fireplaces <= 2
 ind_8028: t568_path1 = 1 -> x_TotRms_AbvGrd <= 11
 ind_8029: t568_path1 = 1 -> const_Fireplaces >= 2
 ind_8030: t568_path2 = 1 -> x_TotRms_AbvGrd >= 11
 ind_8031: t568_path2 = 1 -> x_Bsmt_Unf_SF <= 1088
 ind_8032: t568_path3 = 1 -> x_TotRms_AbvGrd >= 11
 ind_8033: t568_path3 = 1 -> x_Bsmt_Unf_SF >= 1088
 ind_8034: t571_path0 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_8035: t571_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_8036: t571_path1 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_8037: t571_path1 = 1 -> const_Mo_Sold_2 >= 1
 ind_8038: t571_path1 = 1 -> x_Garage_Area <= 474
 ind_8039: t571_path2 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_8040: t571_path2 = 1 -> const_Mo_Sold_2 >= 1
 ind_8041: t571_path2 = 1 -> x_Garage_Area >= 474
 ind_8042: t571_path3 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_8043: t572_path0 = 1 -> const_Year_Built <= 2008
 ind_8044: t572_path0 = 1 -> const_Lot_Area <= 27697
 ind_8045: t572_path1 = 1 -> const_Year_Built <= 2008
 ind_8046: t572_path1 = 1 -> const_Lot_Area >= 27697
 ind_8047: t572_path2 = 1 -> const_Year_Built >= 2008
 ind_8048: t574_path0 = 1 -> x_BsmtFin_SF_1 <= 1259
 ind_8049: t574_path1 = 1 -> x_BsmtFin_SF_1 >= 1259
 ind_8050: t574_path1 = 1 -> x_Bsmt_Unf_SF <= 448
 ind_8051: t574_path1 = 1 -> x_BsmtFin_SF_1 <= 1965
 ind_8052: t574_path2 = 1 -> x_BsmtFin_SF_1 >= 1259
 ind_8053: t574_path2 = 1 -> x_Bsmt_Unf_SF <= 448
 ind_8054: t574_path2 = 1 -> x_BsmtFin_SF_1 >= 1965
 ind_8055: t574_path3 = 1 -> x_BsmtFin_SF_1 >= 1259
 ind_8056: t574_path3 = 1 -> x_Bsmt_Unf_SF >= 448
 ind_8057: t575_path0 = 1 -> x_b1_is_LwQ <= 1
 ind_8058: t575_path1 = 1 -> x_b1_is_LwQ >= 1
 ind_8059: t576_path0 = 1 -> Overall_Qual_calc <= 5
 ind_8060: t576_path0 = 1 -> const_Year_Remod_Add <= 1954
 ind_8061: t576_path1 = 1 -> Overall_Qual_calc <= 5
 ind_8062: t576_path1 = 1 -> const_Year_Remod_Add >= 1954
 ind_8063: t576_path1 = 1 -> x_Open_Porch_SF <= 35
 ind_8064: t576_path2 = 1 -> Overall_Qual_calc <= 5
 ind_8065: t576_path2 = 1 -> const_Year_Remod_Add >= 1954
 ind_8066: t576_path2 = 1 -> x_Open_Porch_SF >= 35
 ind_8067: t576_path3 = 1 -> Overall_Qual_calc >= 5
 ind_8068: t576_path3 = 1 -> x_Garage_Area <= 544
 ind_8069: t576_path4 = 1 -> Overall_Qual_calc >= 5
 ind_8070: t576_path4 = 1 -> x_Garage_Area >= 544
 ind_8071: t576_path4 = 1 -> x_Garage_Area <= 636
 ind_8072: t576_path5 = 1 -> Overall_Qual_calc >= 5
 ind_8073: t576_path5 = 1 -> x_Garage_Area >= 544
 ind_8074: t576_path5 = 1 -> x_Garage_Area >= 636
 ind_8075: t578_path0 = 1 -> const_Mo_Sold_3 <= 1
 ind_8076: t578_path0 = 1 -> x_Screen_Porch <= 154
 ind_8077: t578_path1 = 1 -> const_Mo_Sold_3 <= 1
 ind_8078: t578_path1 = 1 -> x_Screen_Porch >= 154
 ind_8079: t578_path2 = 1 -> const_Mo_Sold_3 >= 1
 ind_8080: t579_path0 = 1 -> const_2nd_Flr_SF <= 834
 ind_8081: t579_path1 = 1 -> const_2nd_Flr_SF >= 834
 ind_8082: t580_path0 = 1 -> x_b1_is_Rec <= 1
 ind_8083: t580_path0 = 1 -> const_Fireplaces <= 3
 ind_8084: t580_path0 = 1 -> x_Mas_Vnr_Area <= 567
 ind_8085: t580_path1 = 1 -> x_b1_is_Rec <= 1
 ind_8086: t580_path1 = 1 -> const_Fireplaces <= 3
 ind_8087: t580_path1 = 1 -> x_Mas_Vnr_Area >= 567
 ind_8088: t580_path2 = 1 -> x_b1_is_Rec <= 1
 ind_8089: t580_path2 = 1 -> const_Fireplaces >= 3
 ind_8090: t580_path3 = 1 -> x_b1_is_Rec >= 1
 ind_8091: t580_path3 = 1 -> x_BsmtFin_SF_1 <= 390
 ind_8092: t580_path4 = 1 -> x_b1_is_Rec >= 1
 ind_8093: t580_path4 = 1 -> x_BsmtFin_SF_1 >= 390
 ind_8094: t580_path4 = 1 -> x_BsmtFin_SF_1 <= 588
 ind_8095: t580_path5 = 1 -> x_b1_is_Rec >= 1
 ind_8096: t580_path5 = 1 -> x_BsmtFin_SF_1 >= 390
 ind_8097: t580_path5 = 1 -> x_BsmtFin_SF_1 >= 588
 ind_8098: t581_path0 = 1 -> x_Garage_Area <= 1154
 ind_8099: t581_path1 = 1 -> x_Garage_Area >= 1154
 ind_8100: t582_path0 = 1 -> Overall_Qual_calc <= 10
 ind_8101: t582_path0 = 1 -> x_Heating_QC <= 2
 ind_8102: t582_path1 = 1 -> Overall_Qual_calc <= 10
 ind_8103: t582_path1 = 1 -> x_Heating_QC >= 2
 ind_8104: t582_path2 = 1 -> Overall_Qual_calc >= 10
 ind_8105: t584_path0 = 1 -> const_Year_Built <= 1914
 ind_8106: t584_path1 = 1 -> const_Year_Built >= 1914
 ind_8107: t584_path1 = 1 -> const_2nd_Flr_SF <= 988
 ind_8108: t584_path1 = 1 -> const_Bsmt_Qual <= 2
 ind_8109: t584_path2 = 1 -> const_Year_Built >= 1914
 ind_8110: t584_path2 = 1 -> const_2nd_Flr_SF <= 988
 ind_8111: t584_path2 = 1 -> const_Bsmt_Qual >= 2
 ind_8112: t584_path3 = 1 -> const_Year_Built >= 1914
 ind_8113: t584_path3 = 1 -> const_2nd_Flr_SF >= 988
 ind_8114: t584_path3 = 1 -> x_Garage_Area <= 622
 ind_8115: t584_path4 = 1 -> const_Year_Built >= 1914
 ind_8116: t584_path4 = 1 -> const_2nd_Flr_SF >= 988
 ind_8117: t584_path4 = 1 -> x_Garage_Area >= 622
 ind_8118: t585_path0 = 1 -> const_Lot_Frontage <= 117
 ind_8119: t585_path0 = 1 -> x_Bedroom_AbvGr <= 5
 ind_8120: t585_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_8121: t585_path1 = 1 -> const_Lot_Frontage <= 117
 ind_8122: t585_path1 = 1 -> x_Bedroom_AbvGr <= 5
 ind_8123: t585_path1 = 1 -> const_Mo_Sold_2 >= 1
 ind_8124: t585_path2 = 1 -> const_Lot_Frontage <= 117
 ind_8125: t585_path2 = 1 -> x_Bedroom_AbvGr >= 5
 ind_8126: t585_path3 = 1 -> const_Lot_Frontage >= 117
 ind_8127: t585_path3 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8128: t585_path4 = 1 -> const_Lot_Frontage >= 117
 ind_8129: t585_path4 = 1 -> const_Sale_Condition_Normal >= 1
 ind_8130: t586_path0 = 1 -> Overall_Qual_calc <= 3
 ind_8131: t586_path1 = 1 -> Overall_Qual_calc >= 3
 ind_8132: t586_path1 = 1 -> Overall_Qual_calc <= 7
 ind_8133: t586_path1 = 1 -> x_Gr_Liv_Area <= 1701
 ind_8134: t586_path2 = 1 -> Overall_Qual_calc >= 3
 ind_8135: t586_path2 = 1 -> Overall_Qual_calc <= 7
 ind_8136: t586_path2 = 1 -> x_Gr_Liv_Area >= 1701
 ind_8137: t586_path3 = 1 -> Overall_Qual_calc >= 3
 ind_8138: t586_path3 = 1 -> Overall_Qual_calc >= 7
 ind_8139: t586_path3 = 1 -> const_Neighborhood_NWAmes <= 1
 ind_8140: t586_path4 = 1 -> Overall_Qual_calc >= 3
 ind_8141: t586_path4 = 1 -> Overall_Qual_calc >= 7
 ind_8142: t586_path4 = 1 -> const_Neighborhood_NWAmes >= 1
 ind_8143: t587_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_8144: t587_path0 = 1 -> x_BsmtFin_SF_1 <= 1474
 ind_8145: t587_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_8146: t587_path1 = 1 -> x_BsmtFin_SF_1 >= 1474
 ind_8147: t587_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_8148: t587_path2 = 1 -> const_Year_Built <= 1954
 ind_8149: t587_path2 = 1 -> const_Roof_Style_Gable <= 1
 ind_8150: t587_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_8151: t587_path3 = 1 -> const_Year_Built <= 1954
 ind_8152: t587_path3 = 1 -> const_Roof_Style_Gable >= 1
 ind_8153: t587_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_8154: t587_path4 = 1 -> const_Year_Built >= 1954
 ind_8155: t587_path4 = 1 -> const_Yr_Sold <= 2009
 ind_8156: t587_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_8157: t587_path5 = 1 -> const_Year_Built >= 1954
 ind_8158: t587_path5 = 1 -> const_Yr_Sold >= 2009
 ind_8159: t588_path0 = 1 -> Overall_Qual_calc <= 7
 ind_8160: t588_path0 = 1 -> const_Lot_Area <= 8520
 ind_8161: t588_path0 = 1 -> x_Garage_Area <= 215
 ind_8162: t588_path1 = 1 -> Overall_Qual_calc <= 7
 ind_8163: t588_path1 = 1 -> const_Lot_Area <= 8520
 ind_8164: t588_path1 = 1 -> x_Garage_Area >= 215
 ind_8165: t588_path2 = 1 -> Overall_Qual_calc <= 7
 ind_8166: t588_path2 = 1 -> const_Lot_Area >= 8520
 ind_8167: t588_path2 = 1 -> central_air_no <= 1
 ind_8168: t588_path3 = 1 -> Overall_Qual_calc <= 7
 ind_8169: t588_path3 = 1 -> const_Lot_Area >= 8520
 ind_8170: t588_path3 = 1 -> central_air_no >= 1
 ind_8171: t588_path4 = 1 -> Overall_Qual_calc >= 7
 ind_8172: t588_path4 = 1 -> const_Year_Remod_Add <= 1993
 ind_8173: t588_path4 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8174: t588_path5 = 1 -> Overall_Qual_calc >= 7
 ind_8175: t588_path5 = 1 -> const_Year_Remod_Add <= 1993
 ind_8176: t588_path5 = 1 -> const_Sale_Condition_Normal >= 1
 ind_8177: t588_path6 = 1 -> Overall_Qual_calc >= 7
 ind_8178: t588_path6 = 1 -> const_Year_Remod_Add >= 1993
 ind_8179: t588_path6 = 1 -> const_Neighborhood_Edwards <= 1
 ind_8180: t588_path7 = 1 -> Overall_Qual_calc >= 7
 ind_8181: t588_path7 = 1 -> const_Year_Remod_Add >= 1993
 ind_8182: t588_path7 = 1 -> const_Neighborhood_Edwards >= 1
 ind_8183: t590_path0 = 1 -> x_Gr_Liv_Area <= 2633
 ind_8184: t590_path1 = 1 -> x_Gr_Liv_Area >= 2633
 ind_8185: t595_path0 = 1 -> const_Year_Built <= 1960
 ind_8186: t595_path0 = 1 -> const_2nd_Flr_SF <= 472
 ind_8187: t595_path1 = 1 -> const_Year_Built <= 1960
 ind_8188: t595_path1 = 1 -> const_2nd_Flr_SF >= 472
 ind_8189: t595_path1 = 1 -> x_Bsmt_Unf_SF <= 190
 ind_8190: t595_path2 = 1 -> const_Year_Built <= 1960
 ind_8191: t595_path2 = 1 -> const_2nd_Flr_SF >= 472
 ind_8192: t595_path2 = 1 -> x_Bsmt_Unf_SF >= 190
 ind_8193: t595_path3 = 1 -> const_Year_Built >= 1960
 ind_8194: t595_path3 = 1 -> const_Lot_Frontage <= 115
 ind_8195: t595_path4 = 1 -> const_Year_Built >= 1960
 ind_8196: t595_path4 = 1 -> const_Lot_Frontage >= 115
 ind_8197: t597_path0 = 1 -> x_mvt_is_BrkCmn <= 1
 ind_8198: t597_path0 = 1 -> x_Fireplace_Qu <= 2
 ind_8199: t597_path0 = 1 -> x_Gr_Liv_Area <= 2237
 ind_8200: t597_path1 = 1 -> x_mvt_is_BrkCmn <= 1
 ind_8201: t597_path1 = 1 -> x_Fireplace_Qu <= 2
 ind_8202: t597_path1 = 1 -> x_Gr_Liv_Area >= 2237
 ind_8203: t597_path2 = 1 -> x_mvt_is_BrkCmn <= 1
 ind_8204: t597_path2 = 1 -> x_Fireplace_Qu >= 2
 ind_8205: t597_path2 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_8206: t597_path3 = 1 -> x_mvt_is_BrkCmn <= 1
 ind_8207: t597_path3 = 1 -> x_Fireplace_Qu >= 2
 ind_8208: t597_path3 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_8209: t597_path4 = 1 -> x_mvt_is_BrkCmn >= 1
 ind_8210: t598_path0 = 1 -> const_Year_Built <= 1915
 ind_8211: t598_path1 = 1 -> const_Year_Built >= 1915
 ind_8212: t600_path0 = 1 -> x_b1_is_Rec <= 1
 ind_8213: t600_path1 = 1 -> x_b1_is_Rec >= 1
 ind_8214: t600_path1 = 1 -> x_ex2_is_Wd_Sdng <= 1
 ind_8215: t600_path1 = 1 -> const_Garage_Yr_Blt <= 1966
 ind_8216: t600_path2 = 1 -> x_b1_is_Rec >= 1
 ind_8217: t600_path2 = 1 -> x_ex2_is_Wd_Sdng <= 1
 ind_8218: t600_path2 = 1 -> const_Garage_Yr_Blt >= 1966
 ind_8219: t600_path3 = 1 -> x_b1_is_Rec >= 1
 ind_8220: t600_path3 = 1 -> x_ex2_is_Wd_Sdng >= 1
 ind_8221: t601_path0 = 1 -> x_b1_is_Rec <= 1
 ind_8222: t601_path1 = 1 -> x_b1_is_Rec >= 1
 ind_8223: t601_path1 = 1 -> x_BsmtFin_SF_1 <= 366
 ind_8224: t601_path2 = 1 -> x_b1_is_Rec >= 1
 ind_8225: t601_path2 = 1 -> x_BsmtFin_SF_1 >= 366
 ind_8226: t601_path2 = 1 -> x_Gr_Liv_Area <= 965
 ind_8227: t601_path3 = 1 -> x_b1_is_Rec >= 1
 ind_8228: t601_path3 = 1 -> x_BsmtFin_SF_1 >= 366
 ind_8229: t601_path3 = 1 -> x_Gr_Liv_Area >= 965
 ind_8230: t603_path0 = 1 -> x_Exter_Qual <= 3
 ind_8231: t603_path1 = 1 -> x_Exter_Qual >= 3
 ind_8232: t605_path0 = 1 -> x_1st_Flr_SF <= 2515
 ind_8233: t605_path0 = 1 -> const_Lot_Frontage <= 110
 ind_8234: t605_path0 = 1 -> const_Garage_Cars <= 3
 ind_8235: t605_path1 = 1 -> x_1st_Flr_SF <= 2515
 ind_8236: t605_path1 = 1 -> const_Lot_Frontage <= 110
 ind_8237: t605_path1 = 1 -> const_Garage_Cars >= 3
 ind_8238: t605_path2 = 1 -> x_1st_Flr_SF <= 2515
 ind_8239: t605_path2 = 1 -> const_Lot_Frontage >= 110
 ind_8240: t605_path3 = 1 -> x_1st_Flr_SF >= 2515
 ind_8241: t607_path0 = 1 -> x_roof_matl_is_Tar&Grv <= 1
 ind_8242: t607_path1 = 1 -> x_roof_matl_is_Tar&Grv >= 1
 ind_8243: t608_path0 = 1 -> const_MS_SubClass_190 <= 1
 ind_8244: t608_path1 = 1 -> const_MS_SubClass_190 >= 1
 ind_8245: t611_path0 = 1 -> const_Mo_Sold_4 <= 1
 ind_8246: t611_path1 = 1 -> const_Mo_Sold_4 >= 1
 ind_8247: t613_path0 = 1 -> x_Half_Bath <= 1
 ind_8248: t613_path1 = 1 -> x_Half_Bath >= 1
 ind_8249: t616_path0 = 1 -> const_Lot_Config_Inside <= 1
 ind_8250: t616_path1 = 1 -> const_Lot_Config_Inside >= 1
 ind_8251: t616_path1 = 1 -> const_Lot_Area <= 10800
 ind_8252: t616_path2 = 1 -> const_Lot_Config_Inside >= 1
 ind_8253: t616_path2 = 1 -> const_Lot_Area >= 10800
 ind_8254: t616_path2 = 1 -> x_Garage_Area <= 366
 ind_8255: t616_path3 = 1 -> const_Lot_Config_Inside >= 1
 ind_8256: t616_path3 = 1 -> const_Lot_Area >= 10800
 ind_8257: t616_path3 = 1 -> x_Garage_Area >= 366
 ind_8258: t617_path0 = 1 -> Overall_Qual_calc <= 5
 ind_8259: t617_path0 = 1 -> const_Land_Contour_Lvl <= 1
 ind_8260: t617_path1 = 1 -> Overall_Qual_calc <= 5
 ind_8261: t617_path1 = 1 -> const_Land_Contour_Lvl >= 1
 ind_8262: t617_path2 = 1 -> Overall_Qual_calc >= 5
 ind_8263: t618_path0 = 1 -> const_FunctioNo_aplical_Maj1 <= 1
 ind_8264: t618_path1 = 1 -> const_FunctioNo_aplical_Maj1 >= 1
 ind_8265: t619_path0 = 1 -> const_Fireplaces <= 1
 ind_8266: t619_path0 = 1 -> x_Garage_Area <= 624
 ind_8267: t619_path1 = 1 -> const_Fireplaces <= 1
 ind_8268: t619_path1 = 1 -> x_Garage_Area >= 624
 ind_8269: t619_path2 = 1 -> const_Fireplaces >= 1
 ind_8270: t619_path2 = 1 -> const_Overall_Cond <= 5
 ind_8271: t619_path2 = 1 -> x_Garage_Area <= 644
 ind_8272: t619_path3 = 1 -> const_Fireplaces >= 1
 ind_8273: t619_path3 = 1 -> const_Overall_Cond <= 5
 ind_8274: t619_path3 = 1 -> x_Garage_Area >= 644
 ind_8275: t619_path4 = 1 -> const_Fireplaces >= 1
 ind_8276: t619_path4 = 1 -> const_Overall_Cond >= 5
 ind_8277: t620_path0 = 1 -> x_Mas_Vnr_Area <= 673
 ind_8278: t620_path1 = 1 -> x_Mas_Vnr_Area >= 673
 ind_8279: t621_path0 = 1 -> const_Year_Built <= 2005
 ind_8280: t621_path1 = 1 -> const_Year_Built >= 2005
 ind_8281: t621_path1 = 1 -> const_Lot_Area <= 8158
 ind_8282: t621_path1 = 1 -> const_Lot_Frontage <= 45
 ind_8283: t621_path2 = 1 -> const_Year_Built >= 2005
 ind_8284: t621_path2 = 1 -> const_Lot_Area <= 8158
 ind_8285: t621_path2 = 1 -> const_Lot_Frontage >= 45
 ind_8286: t621_path3 = 1 -> const_Year_Built >= 2005
 ind_8287: t621_path3 = 1 -> const_Lot_Area >= 8158
 ind_8288: t622_path0 = 1 -> x_Gr_Liv_Area <= 2649
 ind_8289: t622_path0 = 1 -> const_Fireplaces <= 1
 ind_8290: t622_path1 = 1 -> x_Gr_Liv_Area <= 2649
 ind_8291: t622_path1 = 1 -> const_Fireplaces >= 1
 ind_8292: t622_path1 = 1 -> x_TotRms_AbvGrd <= 5
 ind_8293: t622_path2 = 1 -> x_Gr_Liv_Area <= 2649
 ind_8294: t622_path2 = 1 -> const_Fireplaces >= 1
 ind_8295: t622_path2 = 1 -> x_TotRms_AbvGrd >= 5
 ind_8296: t622_path3 = 1 -> x_Gr_Liv_Area >= 2649
 ind_8297: t623_path0 = 1 -> const_Yr_Sold <= 2009
 ind_8298: t623_path0 = 1 -> const_Land_Contour_Bnk <= 1
 ind_8299: t623_path1 = 1 -> const_Yr_Sold <= 2009
 ind_8300: t623_path1 = 1 -> const_Land_Contour_Bnk >= 1
 ind_8301: t623_path1 = 1 -> x_Garage_Area <= 577
 ind_8302: t623_path2 = 1 -> const_Yr_Sold <= 2009
 ind_8303: t623_path2 = 1 -> const_Land_Contour_Bnk >= 1
 ind_8304: t623_path2 = 1 -> x_Garage_Area >= 577
 ind_8305: t623_path3 = 1 -> const_Yr_Sold >= 2009
 ind_8306: t623_path3 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_8307: t623_path4 = 1 -> const_Yr_Sold >= 2009
 ind_8308: t623_path4 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_8309: t623_path4 = 1 -> const_Lot_Config_Corner <= 1
 ind_8310: t623_path5 = 1 -> const_Yr_Sold >= 2009
 ind_8311: t623_path5 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_8312: t623_path5 = 1 -> const_Lot_Config_Corner >= 1
 ind_8313: t624_path0 = 1 -> const_Land_Contour_Low <= 1
 ind_8314: t624_path1 = 1 -> const_Land_Contour_Low >= 1
 ind_8315: t625_path0 = 1 -> const_Total_Bsmt_SF <= 245
 ind_8316: t625_path0 = 1 -> const_Lot_Frontage <= 66
 ind_8317: t625_path1 = 1 -> const_Total_Bsmt_SF <= 245
 ind_8318: t625_path1 = 1 -> const_Lot_Frontage >= 66
 ind_8319: t625_path2 = 1 -> const_Total_Bsmt_SF >= 245
 ind_8320: t626_path0 = 1 -> const_Bsmt_Qual <= 2
 ind_8321: t626_path0 = 1 -> const_Overall_Cond <= 6
 ind_8322: t626_path0 = 1 -> const_Lot_Config_Inside <= 1
 ind_8323: t626_path1 = 1 -> const_Bsmt_Qual <= 2
 ind_8324: t626_path1 = 1 -> const_Overall_Cond <= 6
 ind_8325: t626_path1 = 1 -> const_Lot_Config_Inside >= 1
 ind_8326: t626_path2 = 1 -> const_Bsmt_Qual <= 2
 ind_8327: t626_path2 = 1 -> const_Overall_Cond >= 6
 ind_8328: t626_path3 = 1 -> const_Bsmt_Qual >= 2
 ind_8329: t627_path0 = 1 -> const_Total_Bsmt_SF <= 2042
 ind_8330: t627_path1 = 1 -> const_Total_Bsmt_SF >= 2042
 ind_8331: t632_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_8332: t632_path0 = 1 -> const_Year_Built <= 1915
 ind_8333: t632_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_8334: t632_path1 = 1 -> const_Year_Built >= 1915
 ind_8335: t632_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8336: t632_path2 = 1 -> const_Lot_Area <= 9605
 ind_8337: t632_path3 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8338: t632_path3 = 1 -> const_Lot_Area >= 9605
 ind_8339: t640_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_8340: t640_path0 = 1 -> const_Overall_Cond <= 4
 ind_8341: t640_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_8342: t640_path1 = 1 -> const_Overall_Cond >= 4
 ind_8343: t640_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8344: t640_path2 = 1 -> x_Gr_Liv_Area <= 883
 ind_8345: t640_path3 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8346: t640_path3 = 1 -> x_Gr_Liv_Area >= 883
 ind_8347: t646_path0 = 1 -> const_Overall_Cond <= 6
 ind_8348: t646_path1 = 1 -> const_Overall_Cond >= 6
 ind_8349: t646_path1 = 1 -> x_Full_Bath <= 2
 ind_8350: t646_path2 = 1 -> const_Overall_Cond >= 6
 ind_8351: t646_path2 = 1 -> x_Full_Bath >= 2
 ind_8352: t647_path0 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_8353: t647_path0 = 1 -> const_Lot_Frontage <= 76
 ind_8354: t647_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_8355: t647_path1 = 1 -> const_Lot_Frontage >= 76
 ind_8356: t647_path2 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_8357: t649_path0 = 1 -> x_b2_is_BLQ <= 1
 ind_8358: t649_path1 = 1 -> x_b2_is_BLQ >= 1
 ind_8359: t652_path0 = 1 -> x_Open_Porch_SF <= 288
 ind_8360: t652_path0 = 1 -> const_Lot_Frontage <= 131
 ind_8361: t652_path1 = 1 -> x_Open_Porch_SF <= 288
 ind_8362: t652_path1 = 1 -> const_Lot_Frontage >= 131
 ind_8363: t652_path2 = 1 -> x_Open_Porch_SF >= 288
 ind_8364: t653_path0 = 1 -> x_BsmtFin_SF_1 <= 1433
 ind_8365: t653_path1 = 1 -> x_BsmtFin_SF_1 >= 1433
 ind_8366: t655_path0 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_8367: t655_path1 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_8368: t655_path1 = 1 -> const_House_Style_1.5Fin <= 1
 ind_8369: t655_path2 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_8370: t655_path2 = 1 -> const_House_Style_1.5Fin >= 1
 ind_8371: t656_path0 = 1 -> x_Garage_Area <= 830
 ind_8372: t656_path0 = 1 -> const_Low_Qual_Fin_SF <= 108
 ind_8373: t656_path1 = 1 -> x_Garage_Area <= 830
 ind_8374: t656_path1 = 1 -> const_Low_Qual_Fin_SF >= 108
 ind_8375: t656_path2 = 1 -> x_Garage_Area >= 830
 ind_8376: t657_path0 = 1 -> const_Sale_Type_Oth <= 1
 ind_8377: t657_path1 = 1 -> const_Sale_Type_Oth >= 1
 ind_8378: t659_path0 = 1 -> const_Garage_Yr_Blt <= 1943
 ind_8379: t659_path0 = 1 -> x_b1_is_ALQ <= 1
 ind_8380: t659_path0 = 1 -> x_TotRms_AbvGrd <= 8
 ind_8381: t659_path1 = 1 -> const_Garage_Yr_Blt <= 1943
 ind_8382: t659_path1 = 1 -> x_b1_is_ALQ <= 1
 ind_8383: t659_path1 = 1 -> x_TotRms_AbvGrd >= 8
 ind_8384: t659_path2 = 1 -> const_Garage_Yr_Blt <= 1943
 ind_8385: t659_path2 = 1 -> x_b1_is_ALQ >= 1
 ind_8386: t659_path3 = 1 -> const_Garage_Yr_Blt >= 1943
 ind_8387: t659_path3 = 1 -> x_Garage_Qual <= 2
 ind_8388: t659_path4 = 1 -> const_Garage_Yr_Blt >= 1943
 ind_8389: t659_path4 = 1 -> x_Garage_Qual >= 2
 ind_8390: t660_path0 = 1 -> const_Lot_Frontage <= 140
 ind_8391: t660_path0 = 1 -> x_Open_Porch_SF <= 122
 ind_8392: t660_path0 = 1 -> const_Lot_Area <= 11765
 ind_8393: t660_path1 = 1 -> const_Lot_Frontage <= 140
 ind_8394: t660_path1 = 1 -> x_Open_Porch_SF <= 122
 ind_8395: t660_path1 = 1 -> const_Lot_Area >= 11765
 ind_8396: t660_path2 = 1 -> const_Lot_Frontage <= 140
 ind_8397: t660_path2 = 1 -> x_Open_Porch_SF >= 122
 ind_8398: t660_path3 = 1 -> const_Lot_Frontage >= 140
 ind_8399: t661_path0 = 1 -> x_1st_Flr_SF <= 576
 ind_8400: t661_path1 = 1 -> x_1st_Flr_SF >= 576
 ind_8401: t661_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_8402: t661_path1 = 1 -> x_Garage_Qual <= 2
 ind_8403: t661_path2 = 1 -> x_1st_Flr_SF >= 576
 ind_8404: t661_path2 = 1 -> x_ex1_is_AsbShng <= 1
 ind_8405: t661_path2 = 1 -> x_Garage_Qual >= 2
 ind_8406: t661_path3 = 1 -> x_1st_Flr_SF >= 576
 ind_8407: t661_path3 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8408: t661_path3 = 1 -> x_Bsmt_Unf_SF <= 684
 ind_8409: t661_path4 = 1 -> x_1st_Flr_SF >= 576
 ind_8410: t661_path4 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8411: t661_path4 = 1 -> x_Bsmt_Unf_SF >= 684
 ind_8412: t663_path0 = 1 -> const_Year_Built <= 1939
 ind_8413: t663_path0 = 1 -> const_Condition_1_Norm <= 1
 ind_8414: t663_path1 = 1 -> const_Year_Built <= 1939
 ind_8415: t663_path1 = 1 -> const_Condition_1_Norm >= 1
 ind_8416: t663_path1 = 1 -> const_Year_Built <= 1915
 ind_8417: t663_path2 = 1 -> const_Year_Built <= 1939
 ind_8418: t663_path2 = 1 -> const_Condition_1_Norm >= 1
 ind_8419: t663_path2 = 1 -> const_Year_Built >= 1915
 ind_8420: t663_path3 = 1 -> const_Year_Built >= 1939
 ind_8421: t663_path3 = 1 -> x_elect_is_SBrkr <= 1
 ind_8422: t663_path3 = 1 -> const_Land_Slope_Gtl <= 1
 ind_8423: t663_path4 = 1 -> const_Year_Built >= 1939
 ind_8424: t663_path4 = 1 -> x_elect_is_SBrkr <= 1
 ind_8425: t663_path4 = 1 -> const_Land_Slope_Gtl >= 1
 ind_8426: t663_path5 = 1 -> const_Year_Built >= 1939
 ind_8427: t663_path5 = 1 -> x_elect_is_SBrkr >= 1
 ind_8428: t663_path5 = 1 -> central_air_no <= 1
 ind_8429: t663_path6 = 1 -> const_Year_Built >= 1939
 ind_8430: t663_path6 = 1 -> x_elect_is_SBrkr >= 1
 ind_8431: t663_path6 = 1 -> central_air_no >= 1
 ind_8432: t664_path0 = 1 -> const_Lot_Area <= 6000
 ind_8433: t664_path1 = 1 -> const_Lot_Area >= 6000
 ind_8434: t664_path1 = 1 -> Overall_Qual_calc <= 5
 ind_8435: t664_path2 = 1 -> const_Lot_Area >= 6000
 ind_8436: t664_path2 = 1 -> Overall_Qual_calc >= 5
 ind_8437: t665_path0 = 1 -> x_b1_is_GLQ <= 1
 ind_8438: t665_path0 = 1 -> const_Lot_Area <= 19522
 ind_8439: t665_path1 = 1 -> x_b1_is_GLQ <= 1
 ind_8440: t665_path1 = 1 -> const_Lot_Area >= 19522
 ind_8441: t665_path2 = 1 -> x_b1_is_GLQ >= 1
 ind_8442: t666_path0 = 1 -> const_Neighborhood_StoneBr <= 1
 ind_8443: t666_path1 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_8444: t666_path1 = 1 -> const_Total_Bsmt_SF <= 1342
 ind_8445: t666_path2 = 1 -> const_Neighborhood_StoneBr >= 1
 ind_8446: t666_path2 = 1 -> const_Total_Bsmt_SF >= 1342
 ind_8447: t667_path0 = 1 -> x_Gr_Liv_Area <= 883
 ind_8448: t667_path0 = 1 -> x_Garage_Area <= 286
 ind_8449: t667_path1 = 1 -> x_Gr_Liv_Area <= 883
 ind_8450: t667_path1 = 1 -> x_Garage_Area >= 286
 ind_8451: t667_path1 = 1 -> x_ex1_is_AsbShng <= 1
 ind_8452: t667_path2 = 1 -> x_Gr_Liv_Area <= 883
 ind_8453: t667_path2 = 1 -> x_Garage_Area >= 286
 ind_8454: t667_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8455: t667_path3 = 1 -> x_Gr_Liv_Area >= 883
 ind_8456: t668_path0 = 1 -> x_Garage_Area <= 644
 ind_8457: t668_path1 = 1 -> x_Garage_Area >= 644
 ind_8458: t668_path1 = 1 -> const_Neighborhood_Edwards <= 1
 ind_8459: t668_path1 = 1 -> const_Garage_Yr_Blt <= 1942
 ind_8460: t668_path2 = 1 -> x_Garage_Area >= 644
 ind_8461: t668_path2 = 1 -> const_Neighborhood_Edwards <= 1
 ind_8462: t668_path2 = 1 -> const_Garage_Yr_Blt >= 1942
 ind_8463: t668_path3 = 1 -> x_Garage_Area >= 644
 ind_8464: t668_path3 = 1 -> const_Neighborhood_Edwards >= 1
 ind_8465: t671_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_8466: t671_path1 = 1 -> x_ex1_is_BrkFace >= 1
 ind_8467: t673_path0 = 1 -> x_Bsmt_Cond <= 2
 ind_8468: t673_path0 = 1 -> const_Garage_Yr_Blt <= 1974
 ind_8469: t673_path0 = 1 -> x_b1_is_LwQ <= 1
 ind_8470: t673_path1 = 1 -> x_Bsmt_Cond <= 2
 ind_8471: t673_path1 = 1 -> const_Garage_Yr_Blt <= 1974
 ind_8472: t673_path1 = 1 -> x_b1_is_LwQ >= 1
 ind_8473: t673_path2 = 1 -> x_Bsmt_Cond <= 2
 ind_8474: t673_path2 = 1 -> const_Garage_Yr_Blt >= 1974
 ind_8475: t673_path3 = 1 -> x_Bsmt_Cond >= 2
 ind_8476: t673_path3 = 1 -> const_Lot_Area <= 4500
 ind_8477: t673_path4 = 1 -> x_Bsmt_Cond >= 2
 ind_8478: t673_path4 = 1 -> const_Lot_Area >= 4500
 ind_8479: t674_path0 = 1 -> Overall_Qual_calc <= 10
 ind_8480: t674_path1 = 1 -> Overall_Qual_calc >= 10
 ind_8481: t676_path0 = 1 -> Overall_Qual_calc <= 3
 ind_8482: t676_path1 = 1 -> Overall_Qual_calc >= 3
 ind_8483: t678_path0 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_8484: t678_path0 = 1 -> x_Enclosed_Porch <= 265
 ind_8485: t678_path0 = 1 -> const_Lot_Area <= 11792
 ind_8486: t678_path1 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_8487: t678_path1 = 1 -> x_Enclosed_Porch <= 265
 ind_8488: t678_path1 = 1 -> const_Lot_Area >= 11792
 ind_8489: t678_path2 = 1 -> const_Bsmt_Exposure_Gd <= 1
 ind_8490: t678_path2 = 1 -> x_Enclosed_Porch >= 265
 ind_8491: t678_path3 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_8492: t678_path3 = 1 -> const_Garage_Yr_Blt <= 1961
 ind_8493: t678_path4 = 1 -> const_Bsmt_Exposure_Gd >= 1
 ind_8494: t678_path4 = 1 -> const_Garage_Yr_Blt >= 1961
 ind_8495: t680_path0 = 1 -> Overall_Qual_calc <= 10
 ind_8496: t680_path0 = 1 -> const_Fireplaces <= 1
 ind_8497: t680_path0 = 1 -> x_Gr_Liv_Area <= 1795
 ind_8498: t680_path1 = 1 -> Overall_Qual_calc <= 10
 ind_8499: t680_path1 = 1 -> const_Fireplaces <= 1
 ind_8500: t680_path1 = 1 -> x_Gr_Liv_Area >= 1795
 ind_8501: t680_path2 = 1 -> Overall_Qual_calc <= 10
 ind_8502: t680_path2 = 1 -> const_Fireplaces >= 1
 ind_8503: t680_path2 = 1 -> const_MS_SubClass_30 <= 1
 ind_8504: t680_path3 = 1 -> Overall_Qual_calc <= 10
 ind_8505: t680_path3 = 1 -> const_Fireplaces >= 1
 ind_8506: t680_path3 = 1 -> const_MS_SubClass_30 >= 1
 ind_8507: t680_path4 = 1 -> Overall_Qual_calc >= 10
 ind_8508: t681_path0 = 1 -> const_Total_Bsmt_SF <= 2223
 ind_8509: t681_path1 = 1 -> const_Total_Bsmt_SF >= 2223
 ind_8510: t683_path0 = 1 -> x_1st_Flr_SF <= 1434
 ind_8511: t683_path1 = 1 -> x_1st_Flr_SF >= 1434
 ind_8512: t683_path1 = 1 -> x_Garage_Area <= 288
 ind_8513: t683_path2 = 1 -> x_1st_Flr_SF >= 1434
 ind_8514: t683_path2 = 1 -> x_Garage_Area >= 288
 ind_8515: t683_path2 = 1 -> const_Bsmt_Exposure_Av <= 1
 ind_8516: t683_path3 = 1 -> x_1st_Flr_SF >= 1434
 ind_8517: t683_path3 = 1 -> x_Garage_Area >= 288
 ind_8518: t683_path3 = 1 -> const_Bsmt_Exposure_Av >= 1
 ind_8519: t684_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_8520: t684_path0 = 1 -> x_Bsmt_Unf_SF <= 273
 ind_8521: t684_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_8522: t684_path1 = 1 -> x_Bsmt_Unf_SF >= 273
 ind_8523: t684_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_8524: t687_path0 = 1 -> Overall_Qual_calc <= 3
 ind_8525: t687_path1 = 1 -> Overall_Qual_calc >= 3
 ind_8526: t687_path1 = 1 -> const_Year_Built <= 2005
 ind_8527: t687_path2 = 1 -> Overall_Qual_calc >= 3
 ind_8528: t687_path2 = 1 -> const_Year_Built >= 2005
 ind_8529: t692_path0 = 1 -> const_Lot_Frontage <= 152
 ind_8530: t692_path1 = 1 -> const_Lot_Frontage >= 152
 ind_8531: t693_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_8532: t693_path1 = 1 -> x_Gr_Liv_Area >= 747
 ind_8533: t697_path0 = 1 -> x_Bedroom_AbvGr <= 3
 ind_8534: t697_path1 = 1 -> x_Bedroom_AbvGr >= 3
 ind_8535: t698_path0 = 1 -> x_Garage_Area <= 1154
 ind_8536: t698_path1 = 1 -> x_Garage_Area >= 1154
 ind_8537: t702_path0 = 1 -> const_Year_Built <= 1932
 ind_8538: t702_path0 = 1 -> const_Foundation_PConc <= 1
 ind_8539: t702_path1 = 1 -> const_Year_Built <= 1932
 ind_8540: t702_path1 = 1 -> const_Foundation_PConc >= 1
 ind_8541: t702_path2 = 1 -> const_Year_Built >= 1932
 ind_8542: t702_path2 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_8543: t702_path3 = 1 -> const_Year_Built >= 1932
 ind_8544: t702_path3 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_8545: t705_path0 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_8546: t705_path0 = 1 -> const_Total_Bsmt_SF <= 1380
 ind_8547: t705_path1 = 1 -> const_Total_Bsmt_SF <= 1419
 ind_8548: t705_path1 = 1 -> const_Total_Bsmt_SF >= 1380
 ind_8549: t705_path2 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_8550: t705_path2 = 1 -> x_1st_Flr_SF <= 1422
 ind_8551: t705_path3 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_8552: t705_path3 = 1 -> x_1st_Flr_SF >= 1422
 ind_8553: t705_path3 = 1 -> x_Garage_Area <= 288
 ind_8554: t705_path4 = 1 -> const_Total_Bsmt_SF >= 1419
 ind_8555: t705_path4 = 1 -> x_1st_Flr_SF >= 1422
 ind_8556: t705_path4 = 1 -> x_Garage_Area >= 288
 ind_8557: t706_path0 = 1 -> x_Exter_Qual <= 3
 ind_8558: t706_path1 = 1 -> x_Exter_Qual >= 3
 ind_8559: t706_path1 = 1 -> x_Gr_Liv_Area <= 2794
 ind_8560: t706_path2 = 1 -> x_Exter_Qual >= 3
 ind_8561: t706_path2 = 1 -> x_Gr_Liv_Area >= 2794
 ind_8562: t709_path0 = 1 -> const_Mo_Sold_4 <= 1
 ind_8563: t709_path1 = 1 -> const_Mo_Sold_4 >= 1
 ind_8564: t709_path1 = 1 -> const_Fireplaces <= 1
 ind_8565: t709_path2 = 1 -> const_Mo_Sold_4 >= 1
 ind_8566: t709_path2 = 1 -> const_Fireplaces >= 1
 ind_8567: t710_path0 = 1 -> x_Enclosed_Porch <= 18
 ind_8568: t710_path1 = 1 -> x_Enclosed_Porch >= 18
 ind_8569: t710_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8570: t710_path1 = 1 -> const_Sale_Type_COD <= 1
 ind_8571: t710_path2 = 1 -> x_Enclosed_Porch >= 18
 ind_8572: t710_path2 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8573: t710_path2 = 1 -> const_Sale_Type_COD >= 1
 ind_8574: t710_path3 = 1 -> x_Enclosed_Porch >= 18
 ind_8575: t710_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_8576: t712_path0 = 1 -> const_Lot_Shape_Reg <= 1
 ind_8577: t712_path0 = 1 -> x_TotRms_AbvGrd <= 12
 ind_8578: t712_path0 = 1 -> const_Land_Slope_Gtl <= 1
 ind_8579: t712_path1 = 1 -> const_Lot_Shape_Reg <= 1
 ind_8580: t712_path1 = 1 -> x_TotRms_AbvGrd <= 12
 ind_8581: t712_path1 = 1 -> const_Land_Slope_Gtl >= 1
 ind_8582: t712_path2 = 1 -> const_Lot_Shape_Reg <= 1
 ind_8583: t712_path2 = 1 -> x_TotRms_AbvGrd >= 12
 ind_8584: t712_path3 = 1 -> const_Lot_Shape_Reg >= 1
 ind_8585: t713_path0 = 1 -> x_paved_drive_is_P <= 1
 ind_8586: t713_path1 = 1 -> x_paved_drive_is_P >= 1
 ind_8587: t715_path0 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_8588: t715_path0 = 1 -> const_Lot_Area <= 11792
 ind_8589: t715_path1 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_8590: t715_path1 = 1 -> const_Lot_Area >= 11792
 ind_8591: t715_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8592: t715_path2 = 1 -> const_Neighborhood_Crawfor <= 1
 ind_8593: t715_path2 = 1 -> const_Lot_Area >= 11792
 ind_8594: t715_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_8595: t715_path3 = 1 -> const_Neighborhood_Crawfor >= 1
 ind_8596: t720_path0 = 1 -> Overall_Qual_calc <= 3
 ind_8597: t720_path1 = 1 -> Overall_Qual_calc >= 3
 ind_8598: t726_path0 = 1 -> x_roof_matl_is_CompShg <= 1
 ind_8599: t726_path1 = 1 -> x_roof_matl_is_CompShg >= 1
 ind_8600: t726_path1 = 1 -> const_Garage_Yr_Blt <= 2008
 ind_8601: t726_path2 = 1 -> x_roof_matl_is_CompShg >= 1
 ind_8602: t726_path2 = 1 -> const_Garage_Yr_Blt >= 2008
 ind_8603: t727_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_8604: t727_path1 = 1 -> x_ex2_is_AsbShng >= 1
 ind_8605: t727_path1 = 1 -> const_Overall_Cond <= 5
 ind_8606: t727_path2 = 1 -> x_ex2_is_AsbShng >= 1
 ind_8607: t727_path2 = 1 -> const_Overall_Cond >= 5
 ind_8608: t728_path0 = 1 -> x_Full_Bath <= 3
 ind_8609: t728_path1 = 1 -> x_Full_Bath >= 3
 ind_8610: t730_path0 = 1 -> x_Gr_Liv_Area <= 988
 ind_8611: t730_path1 = 1 -> x_Gr_Liv_Area >= 988
 ind_8612: t730_path1 = 1 -> const_Year_Remod_Add <= 1972
 ind_8613: t730_path1 = 1 -> x_Kitchen_AbvGr <= 2
 ind_8614: t730_path2 = 1 -> x_Gr_Liv_Area >= 988
 ind_8615: t730_path2 = 1 -> const_Year_Remod_Add <= 1972
 ind_8616: t730_path2 = 1 -> x_Kitchen_AbvGr >= 2
 ind_8617: t730_path3 = 1 -> x_Gr_Liv_Area >= 988
 ind_8618: t730_path3 = 1 -> const_Year_Remod_Add >= 1972
 ind_8619: t730_path3 = 1 -> x_Mas_Vnr_Area <= 788
 ind_8620: t730_path4 = 1 -> x_Gr_Liv_Area >= 988
 ind_8621: t730_path4 = 1 -> const_Year_Remod_Add >= 1972
 ind_8622: t730_path4 = 1 -> x_Mas_Vnr_Area >= 788
 ind_8623: t739_path0 = 1 -> x_Open_Porch_SF <= 169
 ind_8624: t739_path1 = 1 -> x_Open_Porch_SF >= 169
 ind_8625: t741_path0 = 1 -> const_Lot_Area <= 23257
 ind_8626: t741_path1 = 1 -> const_Lot_Area >= 23257
 ind_8627: t741_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8628: t741_path2 = 1 -> const_Lot_Area >= 23257
 ind_8629: t741_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_8630: t742_path0 = 1 -> const_Mo_Sold_2 <= 1
 ind_8631: t742_path0 = 1 -> central_air_yes <= 1
 ind_8632: t742_path0 = 1 -> x_Screen_Porch <= 53
 ind_8633: t742_path1 = 1 -> const_Mo_Sold_2 <= 1
 ind_8634: t742_path1 = 1 -> central_air_yes <= 1
 ind_8635: t742_path1 = 1 -> x_Screen_Porch >= 53
 ind_8636: t742_path2 = 1 -> const_Mo_Sold_2 <= 1
 ind_8637: t742_path2 = 1 -> central_air_yes >= 1
 ind_8638: t742_path2 = 1 -> x_Garage_Area <= 270
 ind_8639: t742_path3 = 1 -> const_Mo_Sold_2 <= 1
 ind_8640: t742_path3 = 1 -> central_air_yes >= 1
 ind_8641: t742_path3 = 1 -> x_Garage_Area >= 270
 ind_8642: t742_path4 = 1 -> const_Mo_Sold_2 >= 1
 ind_8643: t742_path4 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8644: t742_path4 = 1 -> x_Open_Porch_SF <= 50
 ind_8645: t742_path5 = 1 -> const_Mo_Sold_2 >= 1
 ind_8646: t742_path5 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8647: t742_path5 = 1 -> x_Open_Porch_SF >= 50
 ind_8648: t742_path6 = 1 -> const_Mo_Sold_2 >= 1
 ind_8649: t742_path6 = 1 -> const_Sale_Condition_Normal >= 1
 ind_8650: t749_path0 = 1 -> x_paved_drive_is_Y <= 1
 ind_8651: t749_path0 = 1 -> x_Garage_Area <= 403
 ind_8652: t749_path0 = 1 -> const_Lot_Area <= 10337
 ind_8653: t749_path1 = 1 -> x_paved_drive_is_Y <= 1
 ind_8654: t749_path1 = 1 -> x_Garage_Area <= 403
 ind_8655: t749_path1 = 1 -> const_Lot_Area >= 10337
 ind_8656: t749_path2 = 1 -> x_paved_drive_is_Y <= 1
 ind_8657: t749_path2 = 1 -> x_Garage_Area >= 403
 ind_8658: t749_path3 = 1 -> x_paved_drive_is_Y >= 1
 ind_8659: t757_path0 = 1 -> const_Yr_Sold <= 2010
 ind_8660: t757_path1 = 1 -> const_Yr_Sold >= 2010
 ind_8661: t757_path1 = 1 -> const_Total_Bsmt_SF <= 761
 ind_8662: t757_path1 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8663: t757_path2 = 1 -> const_Yr_Sold >= 2010
 ind_8664: t757_path2 = 1 -> const_Total_Bsmt_SF <= 761
 ind_8665: t757_path2 = 1 -> const_Sale_Condition_Normal >= 1
 ind_8666: t757_path3 = 1 -> const_Yr_Sold >= 2010
 ind_8667: t757_path3 = 1 -> const_Total_Bsmt_SF >= 761
 ind_8668: t759_path0 = 1 -> const_FunctioNo_aplical_Min1 <= 1
 ind_8669: t759_path1 = 1 -> const_FunctioNo_aplical_Min1 >= 1
 ind_8670: t761_path0 = 1 -> const_Bsmt_Qual <= 4
 ind_8671: t761_path0 = 1 -> x_Garage_Area <= 866
 ind_8672: t761_path1 = 1 -> const_Bsmt_Qual <= 4
 ind_8673: t761_path1 = 1 -> x_Garage_Area >= 866
 ind_8674: t761_path2 = 1 -> const_Bsmt_Qual >= 4
 ind_8675: t761_path2 = 1 -> x_Garage_Area <= 644
 ind_8676: t761_path3 = 1 -> const_Bsmt_Qual >= 4
 ind_8677: t761_path3 = 1 -> x_Garage_Area >= 644
 ind_8678: t763_path0 = 1 -> const_Roof_Style_Hip <= 1
 ind_8679: t763_path0 = 1 -> const_Lot_Area <= 19522
 ind_8680: t763_path1 = 1 -> const_Roof_Style_Hip <= 1
 ind_8681: t763_path1 = 1 -> const_Lot_Area >= 19522
 ind_8682: t763_path2 = 1 -> const_Roof_Style_Hip >= 1
 ind_8683: t763_path2 = 1 -> const_Yr_Sold <= 2008
 ind_8684: t763_path3 = 1 -> const_Roof_Style_Hip >= 1
 ind_8685: t763_path3 = 1 -> const_Yr_Sold >= 2008
 ind_8686: t763_path3 = 1 -> x_Bsmt_Unf_SF <= 466
 ind_8687: t763_path4 = 1 -> const_Roof_Style_Hip >= 1
 ind_8688: t763_path4 = 1 -> const_Yr_Sold >= 2008
 ind_8689: t763_path4 = 1 -> x_Bsmt_Unf_SF >= 466
 ind_8690: t764_path0 = 1 -> x_Bsmt_Unf_SF <= 441
 ind_8691: t764_path0 = 1 -> const_Overall_Cond <= 4
 ind_8692: t764_path1 = 1 -> x_Bsmt_Unf_SF <= 441
 ind_8693: t764_path1 = 1 -> const_Overall_Cond >= 4
 ind_8694: t764_path1 = 1 -> const_Land_Contour_Low <= 1
 ind_8695: t764_path2 = 1 -> x_Bsmt_Unf_SF <= 441
 ind_8696: t764_path2 = 1 -> const_Overall_Cond >= 4
 ind_8697: t764_path2 = 1 -> const_Land_Contour_Low >= 1
 ind_8698: t764_path3 = 1 -> x_Bsmt_Unf_SF >= 441
 ind_8699: t766_path0 = 1 -> const_Alley_Pave <= 1
 ind_8700: t766_path1 = 1 -> const_Alley_Pave >= 1
 ind_8701: t768_path0 = 1 -> x_Open_Porch_SF <= 260
 ind_8702: t768_path0 = 1 -> x_1st_Flr_SF <= 1973
 ind_8703: t768_path1 = 1 -> x_Open_Porch_SF <= 260
 ind_8704: t768_path1 = 1 -> x_1st_Flr_SF >= 1973
 ind_8705: t768_path2 = 1 -> x_Open_Porch_SF >= 260
 ind_8706: t772_path0 = 1 -> const_Lot_Area <= 10200
 ind_8707: t772_path1 = 1 -> const_Lot_Area >= 10200
 ind_8708: t772_path1 = 1 -> const_Total_Bsmt_SF <= 481
 ind_8709: t772_path2 = 1 -> const_Lot_Area >= 10200
 ind_8710: t772_path2 = 1 -> const_Total_Bsmt_SF >= 481
 ind_8711: t775_path0 = 1 -> const_Neighborhood_IDOTRR <= 1
 ind_8712: t775_path1 = 1 -> const_Neighborhood_IDOTRR >= 1
 ind_8713: t777_path0 = 1 -> x_Garage_Area <= 972
 ind_8714: t777_path1 = 1 -> x_Garage_Area >= 972
 ind_8715: t778_path0 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_8716: t778_path0 = 1 -> x_ex2_is_AsbShng <= 1
 ind_8717: t778_path1 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_8718: t778_path1 = 1 -> x_ex2_is_AsbShng >= 1
 ind_8719: t778_path1 = 1 -> const_Overall_Cond <= 5
 ind_8720: t778_path2 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_8721: t778_path2 = 1 -> x_ex2_is_AsbShng >= 1
 ind_8722: t778_path2 = 1 -> const_Overall_Cond >= 5
 ind_8723: t778_path3 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_8724: t781_path0 = 1 -> Overall_Qual_calc <= 3
 ind_8725: t781_path1 = 1 -> Overall_Qual_calc >= 3
 ind_8726: t781_path1 = 1 -> x_Garage_Area <= 804
 ind_8727: t781_path2 = 1 -> Overall_Qual_calc >= 3
 ind_8728: t781_path2 = 1 -> x_Garage_Area >= 804
 ind_8729: t784_path0 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_8730: t784_path0 = 1 -> x_Garage_Area <= 252
 ind_8731: t784_path0 = 1 -> Overall_Qual_calc <= 6
 ind_8732: t784_path1 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_8733: t784_path1 = 1 -> x_Garage_Area <= 252
 ind_8734: t784_path1 = 1 -> Overall_Qual_calc >= 6
 ind_8735: t784_path2 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_8736: t784_path2 = 1 -> x_Garage_Area >= 252
 ind_8737: t784_path2 = 1 -> const_MS_Zoning_C_(all) <= 1
 ind_8738: t784_path3 = 1 -> const_Total_Bsmt_SF <= 2136
 ind_8739: t784_path3 = 1 -> x_Garage_Area >= 252
 ind_8740: t784_path3 = 1 -> const_MS_Zoning_C_(all) >= 1
 ind_8741: t784_path4 = 1 -> const_Total_Bsmt_SF >= 2136
 ind_8742: t785_path0 = 1 -> x_1st_Flr_SF <= 576
 ind_8743: t785_path1 = 1 -> x_1st_Flr_SF >= 576
 ind_8744: t787_path0 = 1 -> x_BsmtFin_SF_1 <= 1965
 ind_8745: t787_path1 = 1 -> x_BsmtFin_SF_1 >= 1965
 ind_8746: t793_path0 = 1 -> x_Open_Porch_SF <= 260
 ind_8747: t793_path0 = 1 -> x_BsmtFin_SF_1 <= 1965
 ind_8748: t793_path1 = 1 -> x_Open_Porch_SF <= 260
 ind_8749: t793_path1 = 1 -> x_BsmtFin_SF_1 >= 1965
 ind_8750: t793_path2 = 1 -> x_Open_Porch_SF >= 260
 ind_8751: t793_path2 = 1 -> const_Sale_Condition_Normal <= 1
 ind_8752: t793_path3 = 1 -> x_Open_Porch_SF >= 260
 ind_8753: t793_path3 = 1 -> const_Sale_Condition_Normal >= 1
 ind_8754: t795_path0 = 1 -> const_Year_Built <= 1945
 ind_8755: t795_path0 = 1 -> const_Overall_Cond <= 7
 ind_8756: t795_path0 = 1 -> const_Overall_Cond <= 5
 ind_8757: t795_path1 = 1 -> const_Year_Built <= 1945
 ind_8758: t795_path1 = 1 -> const_Overall_Cond <= 7
 ind_8759: t795_path1 = 1 -> const_Overall_Cond >= 5
 ind_8760: t795_path2 = 1 -> const_Year_Built <= 1945
 ind_8761: t795_path2 = 1 -> const_Overall_Cond >= 7
 ind_8762: t795_path3 = 1 -> const_Year_Built >= 1945
 ind_8763: t795_path3 = 1 -> central_air_no <= 1
 ind_8764: t795_path4 = 1 -> const_Year_Built >= 1945
 ind_8765: t795_path4 = 1 -> central_air_no >= 1
 ind_8766: t805_path0 = 1 -> x_Exter_Cond <= 2
 ind_8767: t805_path1 = 1 -> x_Exter_Cond >= 2
 ind_8768: t806_path0 = 1 -> x_Gr_Liv_Area <= 936
 ind_8769: t806_path1 = 1 -> x_Gr_Liv_Area >= 936
 ind_8770: t814_path0 = 1 -> x_Enclosed_Porch <= 236
 ind_8771: t814_path0 = 1 -> x_paved_drive_is_P <= 1
 ind_8772: t814_path0 = 1 -> const_Neighborhood_No_aplicames <= 1
 ind_8773: t814_path1 = 1 -> x_Enclosed_Porch <= 236
 ind_8774: t814_path1 = 1 -> x_paved_drive_is_P <= 1
 ind_8775: t814_path1 = 1 -> const_Neighborhood_No_aplicames >= 1
 ind_8776: t814_path2 = 1 -> x_Enclosed_Porch <= 236
 ind_8777: t814_path2 = 1 -> x_paved_drive_is_P >= 1
 ind_8778: t814_path3 = 1 -> x_Enclosed_Porch >= 236
 ind_8779: t814_path3 = 1 -> const_Lot_Area <= 9750
 ind_8780: t814_path4 = 1 -> x_Enclosed_Porch >= 236
 ind_8781: t814_path4 = 1 -> const_Lot_Area >= 9750
 ind_8782: t817_path0 = 1 -> x_ex1_is_AsbShng <= 1
 ind_8783: t817_path1 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8784: t817_path1 = 1 -> x_Gr_Liv_Area <= 1226
 ind_8785: t817_path2 = 1 -> x_ex1_is_AsbShng >= 1
 ind_8786: t817_path2 = 1 -> x_Gr_Liv_Area >= 1226
 ind_8787: t825_path0 = 1 -> x_BsmtFin_SF_1 <= 1387
 ind_8788: t825_path1 = 1 -> x_BsmtFin_SF_1 >= 1387
 ind_8789: t827_path0 = 1 -> const_Year_Built <= 1952
 ind_8790: t827_path0 = 1 -> const_Year_Built <= 1947
 ind_8791: t827_path0 = 1 -> const_Yr_Sold <= 2010
 ind_8792: t827_path1 = 1 -> const_Year_Built <= 1952
 ind_8793: t827_path1 = 1 -> const_Year_Built <= 1947
 ind_8794: t827_path1 = 1 -> const_Yr_Sold >= 2010
 ind_8795: t827_path2 = 1 -> const_Year_Built <= 1952
 ind_8796: t827_path2 = 1 -> const_Year_Built >= 1947
 ind_8797: t827_path3 = 1 -> const_Year_Built >= 1952
 ind_8798: t829_path0 = 1 -> x_1st_Flr_SF <= 1782
 ind_8799: t829_path0 = 1 -> x_1st_Flr_SF <= 624
 ind_8800: t829_path1 = 1 -> x_1st_Flr_SF <= 1782
 ind_8801: t829_path1 = 1 -> x_1st_Flr_SF >= 624
 ind_8802: t829_path2 = 1 -> x_1st_Flr_SF >= 1782
 ind_8803: t829_path2 = 1 -> const_2nd_Flr_SF <= 956
 ind_8804: t829_path2 = 1 -> x_Open_Porch_SF <= 33
 ind_8805: t829_path3 = 1 -> x_1st_Flr_SF >= 1782
 ind_8806: t829_path3 = 1 -> const_2nd_Flr_SF <= 956
 ind_8807: t829_path3 = 1 -> x_Open_Porch_SF >= 33
 ind_8808: t829_path4 = 1 -> x_1st_Flr_SF >= 1782
 ind_8809: t829_path4 = 1 -> const_2nd_Flr_SF >= 956
 ind_8810: t831_path0 = 1 -> x_Enclosed_Porch <= 265
 ind_8811: t831_path1 = 1 -> x_Enclosed_Porch >= 265
 ind_8812: t832_path0 = 1 -> Overall_Qual_calc <= 3
 ind_8813: t832_path1 = 1 -> Overall_Qual_calc >= 3
 ind_8814: t835_path0 = 1 -> central_air_no <= 1
 ind_8815: t835_path1 = 1 -> central_air_no >= 1
 ind_8816: t835_path1 = 1 -> const_2nd_Flr_SF <= 812
 ind_8817: t835_path2 = 1 -> central_air_no >= 1
 ind_8818: t835_path2 = 1 -> const_2nd_Flr_SF >= 812
 ind_8819: t839_path0 = 1 -> const_Roof_Style_Flat <= 1
 ind_8820: t839_path1 = 1 -> const_Roof_Style_Flat >= 1
 ind_8821: t841_path0 = 1 -> x_b1_is_LwQ <= 1
 ind_8822: t841_path1 = 1 -> x_b1_is_LwQ >= 1
 ind_8823: t842_path0 = 1 -> const_Neighborhood_SawyerW <= 1
 ind_8824: t842_path1 = 1 -> const_Neighborhood_SawyerW >= 1
 ind_8825: t844_path0 = 1 -> x_Garage_Area <= 240
 ind_8826: t844_path1 = 1 -> x_Garage_Area >= 240
 ind_8827: t845_path0 = 1 -> x_Garage_Area <= 644
 ind_8828: t845_path1 = 1 -> x_Garage_Area >= 644
 ind_8829: t845_path1 = 1 -> const_Fireplaces <= 1
 ind_8830: t845_path2 = 1 -> x_Garage_Area >= 644
 ind_8831: t845_path2 = 1 -> const_Fireplaces >= 1
 ind_8832: t848_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_8833: t848_path0 = 1 -> x_Enclosed_Porch <= 91
 ind_8834: t848_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_8835: t848_path1 = 1 -> x_Enclosed_Porch >= 91
 ind_8836: t848_path2 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_8837: t849_path0 = 1 -> x_BsmtFin_SF_2 <= 60
 ind_8838: t849_path0 = 1 -> x_Open_Porch_SF <= 189
 ind_8839: t849_path1 = 1 -> x_BsmtFin_SF_2 <= 60
 ind_8840: t849_path1 = 1 -> x_Open_Porch_SF >= 189
 ind_8841: t849_path2 = 1 -> x_BsmtFin_SF_2 >= 60
 ind_8842: t854_path0 = 1 -> x_Mas_Vnr_Area <= 860
 ind_8843: t854_path1 = 1 -> x_Mas_Vnr_Area >= 860
 ind_8844: t856_path0 = 1 -> const_Alley_Pave <= 1
 ind_8845: t856_path1 = 1 -> const_Alley_Pave >= 1
 ind_8846: t858_path0 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_8847: t858_path0 = 1 -> const_Bsmt_Full_Bath <= 2
 ind_8848: t858_path1 = 1 -> const_Sale_Condition_Abnorml <= 1
 ind_8849: t858_path1 = 1 -> const_Bsmt_Full_Bath >= 2
 ind_8850: t858_path2 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_8851: t858_path2 = 1 -> x_Fireplace_Qu <= 3
 ind_8852: t858_path3 = 1 -> const_Sale_Condition_Abnorml >= 1
 ind_8853: t858_path3 = 1 -> x_Fireplace_Qu >= 3
 ind_8854: t860_path0 = 1 -> x_Gr_Liv_Area <= 747
 ind_8855: t860_path0 = 1 -> x_Garage_Cond <= 2
 ind_8856: t860_path1 = 1 -> x_Gr_Liv_Area <= 747
 ind_8857: t860_path1 = 1 -> x_Garage_Cond >= 2
 ind_8858: t860_path2 = 1 -> x_Gr_Liv_Area >= 747
 ind_8859: t865_path0 = 1 -> const_Sale_Condition_Family <= 1
 ind_8860: t865_path1 = 1 -> const_Sale_Condition_Family >= 1
 ind_8861: t865_path1 = 1 -> x_BsmtFin_SF_1 <= 167
 ind_8862: t865_path2 = 1 -> const_Sale_Condition_Family >= 1
 ind_8863: t865_path2 = 1 -> x_BsmtFin_SF_1 >= 167
 ind_8864: t875_path0 = 1 -> x_Gr_Liv_Area <= 816
 ind_8865: t875_path1 = 1 -> x_Gr_Liv_Area >= 816
 ind_8866: t876_path0 = 1 -> x_ex1_is_BrkFace <= 1
 ind_8867: t876_path0 = 1 -> x_Garage_Cond <= 2
 ind_8868: t876_path0 = 1 -> const_Lot_Area <= 8125
 ind_8869: t876_path1 = 1 -> x_ex1_is_BrkFace <= 1
 ind_8870: t876_path1 = 1 -> x_Garage_Cond <= 2
 ind_8871: t876_path1 = 1 -> const_Lot_Area >= 8125
 ind_8872: t876_path2 = 1 -> x_ex1_is_BrkFace <= 1
 ind_8873: t876_path2 = 1 -> x_Garage_Cond >= 2
 ind_8874: t876_path3 = 1 -> x_ex1_is_BrkFace >= 1
 ind_8875: t877_path0 = 1 -> const_Lot_Frontage <= 109
 ind_8876: t877_path1 = 1 -> const_Lot_Frontage >= 109
 ind_8877: t880_path0 = 1 -> const_Lot_Area <= 20431
 ind_8878: t880_path1 = 1 -> const_Lot_Area >= 20431
 ind_8879: t880_path1 = 1 -> x_1st_Flr_SF <= 1241
 ind_8880: t880_path2 = 1 -> const_Lot_Area >= 20431
 ind_8881: t880_path2 = 1 -> x_1st_Flr_SF >= 1241
 ind_8882: t880_path2 = 1 -> const_Lot_Config_CulDSac <= 1
 ind_8883: t880_path3 = 1 -> const_Lot_Area >= 20431
 ind_8884: t880_path3 = 1 -> x_1st_Flr_SF >= 1241
 ind_8885: t880_path3 = 1 -> const_Lot_Config_CulDSac >= 1
 ind_8886: t881_path0 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_8887: t881_path0 = 1 -> const_Yr_Sold <= 2009
 ind_8888: t881_path0 = 1 -> x_BsmtFin_SF_1 <= 188
 ind_8889: t881_path1 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_8890: t881_path1 = 1 -> const_Yr_Sold <= 2009
 ind_8891: t881_path1 = 1 -> x_BsmtFin_SF_1 >= 188
 ind_8892: t881_path2 = 1 -> const_FunctioNo_aplical_Typ <= 1
 ind_8893: t881_path2 = 1 -> const_Yr_Sold >= 2009
 ind_8894: t881_path3 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_8895: t881_path3 = 1 -> x_Fireplace_Qu <= 0
 ind_8896: t881_path4 = 1 -> const_FunctioNo_aplical_Typ >= 1
 ind_8897: t881_path4 = 1 -> x_Fireplace_Qu >= 0
 ind_8898: t883_path0 = 1 -> const_Sale_Type_WD_ <= 1
 ind_8899: t883_path1 = 1 -> const_Sale_Type_WD_ >= 1
 ind_8900: t886_path0 = 1 -> x_Enclosed_Porch <= 78
 ind_8901: t886_path0 = 1 -> x_Enclosed_Porch <= 56
 ind_8902: t886_path0 = 1 -> x_TotRms_AbvGrd <= 5
 ind_8903: t886_path1 = 1 -> x_Enclosed_Porch <= 78
 ind_8904: t886_path1 = 1 -> x_Enclosed_Porch <= 56
 ind_8905: t886_path1 = 1 -> x_TotRms_AbvGrd >= 5
 ind_8906: t886_path2 = 1 -> x_Enclosed_Porch <= 78
 ind_8907: t886_path2 = 1 -> x_Enclosed_Porch >= 56
 ind_8908: t886_path3 = 1 -> x_Enclosed_Porch >= 78
 ind_8909: t886_path3 = 1 -> const_MS_SubClass_50 <= 1
 ind_8910: t886_path3 = 1 -> const_Mo_Sold_5 <= 1
 ind_8911: t886_path4 = 1 -> x_Enclosed_Porch >= 78
 ind_8912: t886_path4 = 1 -> const_MS_SubClass_50 <= 1
 ind_8913: t886_path4 = 1 -> const_Mo_Sold_5 >= 1
 ind_8914: t886_path5 = 1 -> x_Enclosed_Porch >= 78
 ind_8915: t886_path5 = 1 -> const_MS_SubClass_50 >= 1
 ind_8916: t888_path0 = 1 -> const_Overall_Cond <= 7
 ind_8917: t888_path1 = 1 -> const_Overall_Cond >= 7
 ind_8918: t889_path0 = 1 -> x_Gr_Liv_Area <= 883
 ind_8919: t889_path0 = 1 -> const_Fireplaces <= 1
 ind_8920: t889_path1 = 1 -> x_Gr_Liv_Area <= 883
 ind_8921: t889_path1 = 1 -> const_Fireplaces >= 1
 ind_8922: t889_path2 = 1 -> x_Gr_Liv_Area >= 883
 ind_8923: t889_path2 = 1 -> const_Total_Bsmt_SF <= 2223
 ind_8924: t889_path2 = 1 -> const_Total_Bsmt_SF <= 1136
 ind_8925: t889_path3 = 1 -> x_Gr_Liv_Area >= 883
 ind_8926: t889_path3 = 1 -> const_Total_Bsmt_SF <= 2223
 ind_8927: t889_path3 = 1 -> const_Total_Bsmt_SF >= 1136
 ind_8928: t889_path4 = 1 -> x_Gr_Liv_Area >= 883
 ind_8929: t889_path4 = 1 -> const_Total_Bsmt_SF >= 2223
 ind_8930: t894_path0 = 1 -> x_Garage_Area <= 871
 ind_8931: t894_path1 = 1 -> x_Garage_Area >= 871
 ind_8932: t896_path0 = 1 -> const_Fireplaces <= 2
 ind_8933: t896_path1 = 1 -> const_Fireplaces >= 2
 ind_8934: t903_path0 = 1 -> const_Total_Bsmt_SF <= 483
 ind_8935: t903_path1 = 1 -> const_Total_Bsmt_SF >= 483
 ind_8936: t903_path1 = 1 -> x_Open_Porch_SF <= 288
 ind_8937: t903_path2 = 1 -> const_Total_Bsmt_SF >= 483
 ind_8938: t903_path2 = 1 -> x_Open_Porch_SF >= 288
 ind_8939: t904_path0 = 1 -> Overall_Qual_calc <= 3
 ind_8940: t904_path1 = 1 -> Overall_Qual_calc >= 3
 ind_8941: t906_path0 = 1 -> const_Mo_Sold_3 <= 1
 ind_8942: t906_path0 = 1 -> const_Garage_Cars <= 3
 ind_8943: t906_path1 = 1 -> const_Mo_Sold_3 <= 1
 ind_8944: t906_path1 = 1 -> const_Garage_Cars >= 3
 ind_8945: t906_path2 = 1 -> const_Mo_Sold_3 >= 1
 ind_8946: t909_path0 = 1 -> const_Year_Built <= 1911
 ind_8947: t909_path1 = 1 -> const_Year_Built >= 1911
Bounds
 x_Bedroom_AbvGr <= 6
 x_Full_Bath <= 4
 x_BsmtFin_SF_1 <= 1e+06
 x_BsmtFin_SF_2 <= 100000
 x_Bsmt_Unf_SF <= 100000
 x_Gr_Liv_Area <= 10000
 x_1st_Flr_SF <= 5000
 x_TotRms_AbvGrd <= 15
 x_Mas_Vnr_Area = 108
 x_Kitchen_Qual <= 4
 x_Exter_Qual <= 4
 x_Exter_Cond <= 4
 x_Heating_QC <= 4
 -1 <= x_Fireplace_Qu <= 4
 -1 <= x_Bsmt_Cond <= 4
 -1 <= x_Garage_Qual <= 4
 -1 <= x_Garage_Cond <= 4
 -1 <= x_Pool_QC <= 4
 x_Utilities <= 3
 x_Roof_Matl <= 7
 x_garage_finish_is_No_aplica = 0
 x_Garage_Area <= 100000
 x_Wood_Deck_SF <= 100000
 x_Open_Porch_SF <= 100000
 x_Enclosed_Porch <= 100000
 x_3Ssn_Porch <= 100000
 x_Screen_Porch <= 100000
 x_Pool_Area <= 100000
 x_Half_Bath <= 5
 x_Kitchen_AbvGr <= 3
 x_garage_qual_is_Fa = 0
 x_garage_cond_is_Fa = 0
 x_garage_qual_is_Po = 0
 x_garage_cond_is_Po = 0
 x_garage_qual_is_No_aplica = 0
 x_garage_cond_is_No_aplica = 0
 1 <= Overall_Qual_float <= 10
 1 <= Overall_Qual_calc <= 10
 kit_is_Po = 0
 kit_is_Fa = 0
 kit_is_TA = 0
 exterqual_is_Po = 0
 exterqual_is_Fa = 0
 extercond_is_Po = 0
 extercond_is_Fa = 0
 mv_area_mvt_BrkCmn <= 2000
 mv_area_mvt_CBlock <= 2000
 mv_area_mvt_Stone <= 2000
 central_air_yes = 1
 poolqc_is_Po = 0
 poolqc_is_Fa = 0
 poolqc_is_TA = 0
 poolqc_is_Gd = 0
 poolqc_is_Ex = 0
 poolqc_is_No_aplica = 1
 UpgPool = 0
 heat_is_Floor = 0
 heat_is_OthW = 0
 heat_is_Wall = 0
 fireplace_is_Po = 0
 fireplace_is_Fa = 0
 fireplace_is_TA = 0
 fireplace_is_Gd = 0
 fireplace_is_Ex = 0
 fireplace_is_No_aplica = 1
 bsmtcond_is_Po = 0
 bsmtcond_is_Fa = 0
 B1_upg_flag = 0
 B2_upg_flag = 1
 util_ELO = 0
 util_NoSeWa = 0
 util_NoSewr = 0
 R8_other_rooms <= 15
 const_Lot_Frontage = 81
 const_Lot_Area = 14267
 const_Overall_Cond = 6
 const_Year_Built = 1958
 const_Year_Remod_Add = 1958
 const_Bsmt_Qual = 2
 const_Total_Bsmt_SF = 1329
 const_2nd_Flr_SF = 0
 const_Low_Qual_Fin_SF = 0
 const_Bsmt_Full_Bath = 0
 const_Bsmt_Half_Bath = 0
 const_Fireplaces = 0
 const_Garage_Yr_Blt = 1958
 const_Garage_Cars = 1
 const_Misc_Val = 12500
 const_Yr_Sold = 2010
 const_MS_SubClass_20 = 1
 const_MS_SubClass_30 = 0
 const_MS_SubClass_40 = 0
 const_MS_SubClass_45 = 0
 const_MS_SubClass_50 = 0
 const_MS_SubClass_60 = 0
 const_MS_SubClass_70 = 0
 const_MS_SubClass_75 = 0
 const_MS_SubClass_80 = 0
 const_MS_SubClass_85 = 0
 const_MS_SubClass_90 = 0
 const_MS_SubClass_120 = 0
 const_MS_SubClass_150 = 0
 const_MS_SubClass_160 = 0
 const_MS_SubClass_180 = 0
 const_MS_SubClass_190 = 0
 const_MS_Zoning_A_(agr) = 0
 const_MS_Zoning_C_(all) = 0
 const_MS_Zoning_FV = 0
 const_MS_Zoning_I_(all) = 0
 const_MS_Zoning_RH = 0
 const_MS_Zoning_RL = 1
 const_MS_Zoning_RM = 0
 const_Street_Grvl = 0
 const_Street_Pave = 1
 const_Alley_Grvl = 0
 const_Alley_No_aplica = 1
 const_Alley_Pave = 0
 const_Lot_Shape_IR1 = 1
 const_Lot_Shape_IR2 = 0
 const_Lot_Shape_IR3 = 0
 const_Lot_Shape_Reg = 0
 const_Land_Contour_Bnk = 0
 const_Land_Contour_HLS = 0
 const_Land_Contour_Low = 0
 const_Land_Contour_Lvl = 1
 const_Lot_Config_Corner = 1
 const_Lot_Config_CulDSac = 0
 const_Lot_Config_FR2 = 0
 const_Lot_Config_FR3 = 0
 const_Lot_Config_Inside = 0
 const_Land_Slope_Gtl = 1
 const_Land_Slope_Mod = 0
 const_Land_Slope_Sev = 0
 const_Neighborhood_Blmngtn = 0
 const_Neighborhood_Blueste = 0
 const_Neighborhood_BrDale = 0
 const_Neighborhood_BrkSide = 0
 const_Neighborhood_ClearCr = 0
 const_Neighborhood_CollgCr = 0
 const_Neighborhood_Crawfor = 0
 const_Neighborhood_Edwards = 0
 const_Neighborhood_Gilbert = 0
 const_Neighborhood_Greens = 0
 const_Neighborhood_GrnHill = 0
 const_Neighborhood_IDOTRR = 0
 const_Neighborhood_Landmrk = 0
 const_Neighborhood_MeadowV = 0
 const_Neighborhood_Mitchel = 0
 const_Neighborhood_NPkVill = 0
 const_Neighborhood_NWAmes = 0
 const_Neighborhood_No_aplicames = 1
 const_Neighborhood_NoRidge = 0
 const_Neighborhood_NridgHt = 0
 const_Neighborhood_OldTown = 0
 const_Neighborhood_SWISU = 0
 const_Neighborhood_Sawyer = 0
 const_Neighborhood_SawyerW = 0
 const_Neighborhood_Somerst = 0
 const_Neighborhood_StoneBr = 0
 const_Neighborhood_Timber = 0
 const_Neighborhood_Veenker = 0
 const_Condition_1_Artery = 0
 const_Condition_1_Feedr = 0
 const_Condition_1_Norm = 1
 const_Condition_1_PosA = 0
 const_Condition_1_PosN = 0
 const_Condition_1_RRAe = 0
 const_Condition_1_RRAn = 0
 const_Condition_1_RRNe = 0
 const_Condition_1_RRNn = 0
 const_Condition_2_Artery = 0
 const_Condition_2_Feedr = 0
 const_Condition_2_Norm = 1
 const_Condition_2_PosA = 0
 const_Condition_2_PosN = 0
 const_Condition_2_RRAe = 0
 const_Condition_2_RRAn = 0
 const_Condition_2_RRNn = 0
 const_Bldg_Type_1Fam = 1
 const_Bldg_Type_2fmCon = 0
 const_Bldg_Type_Duplex = 0
 const_Bldg_Type_Twnhs = 0
 const_Bldg_Type_TwnhsE = 0
 const_House_Style_1.5Fin = 0
 const_House_Style_1.5Unf = 0
 const_House_Style_1Story = 1
 const_House_Style_2.5Fin = 0
 const_House_Style_2.5Unf = 0
 const_House_Style_2Story = 0
 const_House_Style_SFoyer = 0
 const_House_Style_SLvl = 0
 const_Roof_Style_Flat = 0
 const_Roof_Style_Gable = 0
 const_Roof_Style_Gambrel = 0
 const_Roof_Style_Hip = 1
 const_Roof_Style_Mansard = 0
 const_Roof_Style_Shed = 0
 const_Exterior_1st_WdShing = 0
 const_Exterior_2nd_Brk_Cmn = 0
 const_Exterior_2nd_CmentBd = 0
 const_Exterior_2nd_Wd_Shng = 0
 const_Foundation_BrkTil = 0
 const_Foundation_CBlock = 1
 const_Foundation_PConc = 0
 const_Foundation_Slab = 0
 const_Foundation_Stone = 0
 const_Foundation_Wood = 0
 const_Bsmt_Exposure_Av = 0
 const_Bsmt_Exposure_Gd = 0
 const_Bsmt_Exposure_Mn = 0
 const_Bsmt_Exposure_No = 1
 const_Bsmt_Exposure_No_aplica = 0
 const_BsmtFin_Type_1_No_aplica = 0
 const_BsmtFin_Type_2_No_aplica = 0
 const_FunctioNo_aplical_Maj1 = 0
 const_FunctioNo_aplical_Maj2 = 0
 const_FunctioNo_aplical_Min1 = 0
 const_FunctioNo_aplical_Min2 = 0
 const_FunctioNo_aplical_Mod = 0
 const_FunctioNo_aplical_Sal = 0
 const_FunctioNo_aplical_Sev = 0
 const_FunctioNo_aplical_Typ = 1
 const_Garage_Type_2Types = 0
 const_Garage_Type_Attchd = 1
 const_Garage_Type_Basment = 0
 const_Garage_Type_BuiltIn = 0
 const_Garage_Type_CarPort = 0
 const_Garage_Type_Detchd = 0
 const_Garage_Type_No_aplica = 0
 const_Garage_Finish_Fin = 0
 const_Garage_Finish_No_aplica = 0
 const_Garage_Finish_RFn = 0
 const_Garage_Finish_Unf = 1
 const_Misc_Feature_Elev = 0
 const_Misc_Feature_Gar2 = 1
 const_Misc_Feature_No_aplica = 0
 const_Misc_Feature_Othr = 0
 const_Misc_Feature_Shed = 0
 const_Misc_Feature_TenC = 0
 const_Mo_Sold_1 = 0
 const_Mo_Sold_2 = 0
 const_Mo_Sold_3 = 0
 const_Mo_Sold_4 = 0
 const_Mo_Sold_5 = 0
 const_Mo_Sold_6 = 1
 const_Mo_Sold_7 = 0
 const_Mo_Sold_8 = 0
 const_Mo_Sold_9 = 0
 const_Mo_Sold_10 = 0
 const_Mo_Sold_11 = 0
 const_Mo_Sold_12 = 0
 const_Sale_Type_COD = 0
 const_Sale_Type_CWD = 0
 const_Sale_Type_Con = 0
 const_Sale_Type_ConLD = 0
 const_Sale_Type_ConLI = 0
 const_Sale_Type_ConLw = 0
 const_Sale_Type_New = 0
 const_Sale_Type_Oth = 0
 const_Sale_Type_VWD = 0
 const_Sale_Type_WD_ = 1
 const_Sale_Condition_Abnorml = 0
 const_Sale_Condition_AdjLand = 0
 const_Sale_Condition_Alloca = 0
 const_Sale_Condition_Family = 0
 const_Sale_Condition_Normal = 1
 const_Sale_Condition_Partial = 0
 y_log_raw free
 y_log free
 Constant = 1
Binaries
 x_mvt_is_BrkCmn x_mvt_is_BrkFace x_mvt_is_CBlock x_mvt_is_No_aplica
 x_mvt_is_Stone x_delta_KitchenQual_TA x_delta_KitchenQual_EX x_u_util_ELO
 x_u_util_NoSeWa x_u_util_NoSewr x_u_util_AllPub x_roof_matl_is_ClyTile
 x_roof_matl_is_CompShg x_roof_matl_is_Membran x_roof_matl_is_Metal
 x_roof_matl_is_Roll x_roof_matl_is_Tar&Grv x_roof_matl_is_WdShake
 x_roof_matl_is_WdShngl x_elect_is_SBrkr x_elect_is_FuseA x_elect_is_FuseF
 x_elect_is_FuseP x_elect_is_Mix x_garage_finish_is_Fin
 x_garage_finish_is_RFn x_garage_finish_is_Unf x_garage_finish_is_No_aplica
 x_UpgGarage x_poolqc_is_Ex x_poolqc_is_Gd x_poolqc_is_TA x_poolqc_is_Fa
 x_poolqc_is_Po x_poolqc_is_No_aplica x_upg_pool_qc x_AddFull x_AddHalf
 x_AddKitch x_AddBed x_z10_GarageArea x_z20_GarageArea x_z30_GarageArea
 x_z10_WoodDeckSF x_z20_WoodDeckSF x_z30_WoodDeckSF x_z10_OpenPorchSF
 x_z20_OpenPorchSF x_z30_OpenPorchSF x_z10_EnclosedPorch
 x_z20_EnclosedPorch x_z30_EnclosedPorch x_z10_3SsnPorch x_z20_3SsnPorch
 x_z30_3SsnPorch x_z10_ScreenPorch x_z20_ScreenPorch x_z30_ScreenPorch
 x_z10_PoolArea x_z20_PoolArea x_z30_PoolArea x_garage_qual_is_Ex
 x_garage_cond_is_Ex x_garage_qual_is_Gd x_garage_cond_is_Gd
 x_garage_qual_is_TA x_garage_cond_is_TA x_garage_qual_is_Fa
 x_garage_cond_is_Fa x_garage_qual_is_Po x_garage_cond_is_Po
 x_garage_qual_is_No_aplica x_garage_cond_is_No_aplica x_UpgGarage
 x_paved_drive_is_Y x_paved_drive_is_P x_paved_drive_is_N x_fence_is_GdPrv
 x_fence_is_MnPrv x_fence_is_GdWo x_fence_is_MnWw x_fence_is_No_aplica
 x_ex1_is_AsbShng x_ex1_is_AsphShn x_ex1_is_BrkComm x_ex1_is_BrkFace
 x_ex1_is_CBlock x_ex1_is_CemntBd x_ex1_is_HdBoard x_ex1_is_ImStucc
 x_ex1_is_MetalSd x_ex1_is_Other x_ex1_is_Plywood x_ex1_is_PreCast
 x_ex1_is_Stone x_ex1_is_Stucco x_ex1_is_VinylSd x_ex1_is_Wd_Sdng
 x_ex1_is_WdShngl x_ex2_is_AsbShng x_ex2_is_AsphShn x_ex2_is_BrkComm
 x_ex2_is_BrkFace x_ex2_is_CBlock x_ex2_is_CemntBd x_ex2_is_HdBoard
 x_ex2_is_ImStucc x_ex2_is_MetalSd x_ex2_is_Other x_ex2_is_Plywood
 x_ex2_is_PreCast x_ex2_is_Stone x_ex2_is_Stucco x_ex2_is_VinylSd
 x_ex2_is_Wd_Sdng x_ex2_is_WdShngl x_Heating_Floor x_Heating_GasA
 x_Heating_GasW x_Heating_Grav x_Heating_OthW x_Heating_Wall
 x_heat_upg_type x_heat_upg_qc x_b1_is_GLQ x_b1_is_ALQ x_b1_is_BLQ
 x_b1_is_Rec x_b1_is_LwQ x_b1_is_Unf x_b1_is_No_aplica x_b2_is_GLQ
 x_b2_is_ALQ x_b2_is_BLQ x_b2_is_Rec x_b2_is_LwQ x_b2_is_Unf
 x_b2_is_No_aplica x_z10_GarageArea x_z20_GarageArea x_z30_GarageArea
 x_z10_WoodDeckSF x_z20_WoodDeckSF x_z30_WoodDeckSF x_z10_OpenPorchSF
 x_z20_OpenPorchSF x_z30_OpenPorchSF x_z10_EnclosedPorch
 x_z20_EnclosedPorch x_z30_EnclosedPorch x_z10_3SsnPorch x_z20_3SsnPorch
 x_z30_3SsnPorch x_z10_ScreenPorch x_z20_ScreenPorch x_z30_ScreenPorch
 x_z10_PoolArea x_z20_PoolArea x_z30_PoolArea kit_is_Po kit_is_Fa kit_is_TA
 kit_is_Gd kit_is_Ex exterqual_is_Po exterqual_is_Fa exterqual_is_TA
 exterqual_is_Gd exterqual_is_Ex extercond_is_Po extercond_is_Fa
 extercond_is_TA extercond_is_Gd extercond_is_Ex ext_upg_material
 ext_upg_qc roof_change_mat central_air_yes central_air_no poolqc_is_Po
 poolqc_is_Fa poolqc_is_TA poolqc_is_Gd poolqc_is_Ex poolqc_is_No_aplica
 UpgPool x_Add1stFlr heat_is_Floor heat_is_GasA heat_is_GasW heat_is_Grav
 heat_is_OthW heat_is_Wall HEAT_any_rebuild heat_change_type heat_qc_is_Po
 heat_qc_is_Fa heat_qc_is_TA heat_qc_is_Gd heat_qc_is_Ex heat_upg_type_flag
 heat_upg_qc_flag fireplace_is_Po fireplace_is_Fa fireplace_is_TA
 fireplace_is_Gd fireplace_is_Ex fireplace_is_No_aplica bsmt_finish_all
 bsmtcond_is_Po bsmtcond_is_Fa bsmtcond_is_TA bsmtcond_is_Gd bsmtcond_is_Ex
 B1_upg_flag B2_upg_flag util_ELO util_NoSeWa util_NoSewr util_AllPub
 t0_path0 t0_path1 t0_path2 t0_path3 t0_path4 t0_path5 t0_path6 t0_path7
 t1_path0 t1_path1 t1_path2 t1_path3 t1_path4 t1_path5 t1_path6 t1_path7
 t2_path0 t2_path1 t2_path2 t2_path3 t2_path4 t2_path5 t2_path6 t2_path7
 t3_path0 t3_path1 t3_path2 t3_path3 t3_path4 t3_path5 t3_path6 t3_path7
 t4_path0 t4_path1 t4_path2 t4_path3 t4_path4 t4_path5 t4_path6 t4_path7
 t5_path0 t5_path1 t5_path2 t5_path3 t5_path4 t5_path5 t5_path6 t5_path7
 t6_path0 t6_path1 t6_path2 t6_path3 t6_path4 t6_path5 t6_path6 t6_path7
 t7_path0 t7_path1 t7_path2 t7_path3 t7_path4 t7_path5 t7_path6 t7_path7
 t8_path0 t8_path1 t8_path2 t8_path3 t8_path4 t8_path5 t8_path6 t8_path7
 t9_path0 t9_path1 t9_path2 t9_path3 t9_path4 t9_path5 t9_path6 t9_path7
 t10_path0 t10_path1 t10_path2 t10_path3 t10_path4 t10_path5 t10_path6
 t10_path7 t11_path0 t11_path1 t11_path2 t11_path3 t11_path4 t11_path5
 t11_path6 t11_path7 t12_path0 t12_path1 t12_path2 t12_path3 t12_path4
 t12_path5 t12_path6 t12_path7 t13_path0 t13_path1 t13_path2 t13_path3
 t13_path4 t13_path5 t13_path6 t13_path7 t14_path0 t14_path1 t14_path2
 t14_path3 t14_path4 t14_path5 t14_path6 t14_path7 t15_path0 t15_path1
 t15_path2 t15_path3 t15_path4 t15_path5 t15_path6 t15_path7 t16_path0
 t16_path1 t16_path2 t16_path3 t16_path4 t16_path5 t16_path6 t16_path7
 t17_path0 t17_path1 t17_path2 t17_path3 t17_path4 t17_path5 t17_path6
 t18_path0 t18_path1 t18_path2 t18_path3 t18_path4 t18_path5 t18_path6
 t18_path7 t19_path0 t19_path1 t19_path2 t19_path3 t19_path4 t19_path5
 t19_path6 t19_path7 t20_path0 t20_path1 t20_path2 t20_path3 t20_path4
 t20_path5 t20_path6 t20_path7 t21_path0 t21_path1 t21_path2 t21_path3
 t21_path4 t21_path5 t21_path6 t21_path7 t22_path0 t22_path1 t22_path2
 t22_path3 t22_path4 t22_path5 t22_path6 t22_path7 t23_path0 t23_path1
 t23_path2 t23_path3 t23_path4 t23_path5 t23_path6 t23_path7 t24_path0
 t24_path1 t24_path2 t24_path3 t24_path4 t24_path5 t24_path6 t24_path7
 t25_path0 t25_path1 t25_path2 t25_path3 t25_path4 t25_path5 t25_path6
 t25_path7 t26_path0 t26_path1 t26_path2 t26_path3 t26_path4 t26_path5
 t26_path6 t26_path7 t27_path0 t27_path1 t27_path2 t27_path3 t27_path4
 t27_path5 t27_path6 t27_path7 t28_path0 t28_path1 t28_path2 t28_path3
 t28_path4 t28_path5 t28_path6 t28_path7 t29_path0 t29_path1 t29_path2
 t29_path3 t29_path4 t29_path5 t29_path6 t29_path7 t30_path0 t30_path1
 t30_path2 t30_path3 t30_path4 t30_path5 t30_path6 t30_path7 t31_path0
 t31_path1 t31_path2 t31_path3 t31_path4 t31_path5 t31_path6 t31_path7
 t32_path0 t32_path1 t32_path2 t32_path3 t32_path4 t32_path5 t32_path6
 t32_path7 t33_path0 t33_path1 t33_path2 t33_path3 t33_path4 t33_path5
 t33_path6 t33_path7 t34_path0 t34_path1 t34_path2 t34_path3 t34_path4
 t34_path5 t34_path6 t34_path7 t35_path0 t35_path1 t35_path2 t35_path3
 t35_path4 t35_path5 t35_path6 t35_path7 t36_path0 t36_path1 t36_path2
 t36_path3 t36_path4 t36_path5 t36_path6 t36_path7 t37_path0 t37_path1
 t37_path2 t37_path3 t37_path4 t37_path5 t37_path6 t37_path7 t38_path0
 t38_path1 t38_path2 t38_path3 t38_path4 t38_path5 t38_path6 t38_path7
 t39_path0 t39_path1 t39_path2 t39_path3 t39_path4 t39_path5 t39_path6
 t39_path7 t40_path0 t40_path1 t40_path2 t40_path3 t40_path4 t40_path5
 t40_path6 t40_path7 t41_path0 t41_path1 t41_path2 t41_path3 t41_path4
 t41_path5 t41_path6 t41_path7 t42_path0 t42_path1 t42_path2 t42_path3
 t42_path4 t42_path5 t42_path6 t42_path7 t43_path0 t43_path1 t43_path2
 t43_path3 t43_path4 t43_path5 t43_path6 t43_path7 t44_path0 t44_path1
 t44_path2 t44_path3 t44_path4 t44_path5 t44_path6 t44_path7 t45_path0
 t45_path1 t45_path2 t45_path3 t45_path4 t45_path5 t45_path6 t45_path7
 t46_path0 t46_path1 t46_path2 t46_path3 t46_path4 t46_path5 t46_path6
 t46_path7 t47_path0 t47_path1 t47_path2 t47_path3 t47_path4 t47_path5
 t47_path6 t47_path7 t48_path0 t48_path1 t48_path2 t48_path3 t48_path4
 t48_path5 t48_path6 t48_path7 t49_path0 t49_path1 t49_path2 t49_path3
 t49_path4 t49_path5 t49_path6 t49_path7 t50_path0 t50_path1 t50_path2
 t50_path3 t50_path4 t50_path5 t50_path6 t51_path0 t51_path1 t51_path2
 t51_path3 t51_path4 t51_path5 t51_path6 t52_path0 t52_path1 t52_path2
 t52_path3 t52_path4 t52_path5 t52_path6 t52_path7 t53_path0 t53_path1
 t53_path2 t53_path3 t53_path4 t53_path5 t53_path6 t54_path0 t54_path1
 t54_path2 t54_path3 t54_path4 t54_path5 t54_path6 t55_path0 t55_path1
 t55_path2 t55_path3 t55_path4 t55_path5 t55_path6 t55_path7 t56_path0
 t56_path1 t56_path2 t56_path3 t56_path4 t56_path5 t56_path6 t56_path7
 t57_path0 t57_path1 t57_path2 t57_path3 t57_path4 t57_path5 t57_path6
 t57_path7 t58_path0 t58_path1 t58_path2 t58_path3 t58_path4 t58_path5
 t58_path6 t58_path7 t59_path0 t59_path1 t59_path2 t59_path3 t59_path4
 t59_path5 t59_path6 t59_path7 t60_path0 t60_path1 t60_path2 t60_path3
 t60_path4 t60_path5 t60_path6 t61_path0 t61_path1 t61_path2 t61_path3
 t61_path4 t61_path5 t61_path6 t62_path0 t62_path1 t62_path2 t62_path3
 t62_path4 t62_path5 t62_path6 t63_path0 t63_path1 t63_path2 t63_path3
 t63_path4 t63_path5 t63_path6 t63_path7 t64_path0 t64_path1 t64_path2
 t64_path3 t64_path4 t64_path5 t64_path6 t64_path7 t65_path0 t65_path1
 t65_path2 t65_path3 t65_path4 t65_path5 t65_path6 t65_path7 t66_path0
 t66_path1 t66_path2 t66_path3 t66_path4 t66_path5 t66_path6 t66_path7
 t67_path0 t67_path1 t67_path2 t67_path3 t67_path4 t67_path5 t68_path0
 t68_path1 t68_path2 t68_path3 t68_path4 t68_path5 t68_path6 t68_path7
 t69_path0 t69_path1 t69_path2 t69_path3 t69_path4 t69_path5 t69_path6
 t70_path0 t70_path1 t70_path2 t70_path3 t70_path4 t70_path5 t70_path6
 t70_path7 t71_path0 t71_path1 t71_path2 t71_path3 t71_path4 t71_path5
 t71_path6 t71_path7 t72_path0 t72_path1 t72_path2 t72_path3 t72_path4
 t72_path5 t72_path6 t72_path7 t73_path0 t73_path1 t73_path2 t73_path3
 t73_path4 t73_path5 t73_path6 t74_path0 t74_path1 t74_path2 t74_path3
 t74_path4 t74_path5 t74_path6 t75_path0 t75_path1 t75_path2 t75_path3
 t75_path4 t75_path5 t75_path6 t76_path0 t76_path1 t76_path2 t76_path3
 t76_path4 t76_path5 t76_path6 t76_path7 t77_path0 t77_path1 t77_path2
 t77_path3 t77_path4 t77_path5 t77_path6 t78_path0 t78_path1 t78_path2
 t78_path3 t78_path4 t78_path5 t78_path6 t78_path7 t79_path0 t79_path1
 t79_path2 t79_path3 t79_path4 t79_path5 t79_path6 t80_path0 t80_path1
 t80_path2 t80_path3 t80_path4 t80_path5 t80_path6 t81_path0 t81_path1
 t81_path2 t81_path3 t81_path4 t82_path0 t82_path1 t82_path2 t82_path3
 t82_path4 t82_path5 t82_path6 t83_path0 t83_path1 t83_path2 t83_path3
 t83_path4 t84_path0 t84_path1 t84_path2 t84_path3 t84_path4 t84_path5
 t84_path6 t85_path0 t85_path1 t85_path2 t85_path3 t85_path4 t85_path5
 t85_path6 t86_path0 t86_path1 t86_path2 t86_path3 t86_path4 t86_path5
 t86_path6 t87_path0 t87_path1 t87_path2 t87_path3 t87_path4 t87_path5
 t87_path6 t88_path0 t88_path1 t88_path2 t88_path3 t88_path4 t88_path5
 t88_path6 t89_path0 t89_path1 t89_path2 t89_path3 t89_path4 t89_path5
 t89_path6 t89_path7 t90_path0 t90_path1 t90_path2 t90_path3 t90_path4
 t90_path5 t90_path6 t90_path7 t91_path0 t91_path1 t91_path2 t91_path3
 t91_path4 t91_path5 t91_path6 t91_path7 t92_path0 t92_path1 t92_path2
 t92_path3 t92_path4 t92_path5 t92_path6 t93_path0 t93_path1 t93_path2
 t93_path3 t93_path4 t93_path5 t93_path6 t93_path7 t94_path0 t94_path1
 t94_path2 t94_path3 t94_path4 t94_path5 t94_path6 t95_path0 t95_path1
 t95_path2 t95_path3 t95_path4 t95_path5 t95_path6 t95_path7 t96_path0
 t96_path1 t96_path2 t96_path3 t96_path4 t96_path5 t96_path6 t97_path0
 t97_path1 t97_path2 t97_path3 t97_path4 t97_path5 t97_path6 t97_path7
 t98_path0 t98_path1 t98_path2 t98_path3 t98_path4 t98_path5 t98_path6
 t99_path0 t99_path1 t99_path2 t99_path3 t99_path4 t99_path5 t99_path6
 t100_path0 t100_path1 t100_path2 t100_path3 t100_path4 t100_path5
 t100_path6 t100_path7 t101_path0 t101_path1 t101_path2 t101_path3
 t101_path4 t101_path5 t101_path6 t102_path0 t102_path1 t102_path2
 t102_path3 t102_path4 t102_path5 t102_path6 t102_path7 t103_path0
 t103_path1 t103_path2 t103_path3 t103_path4 t103_path5 t104_path0
 t104_path1 t104_path2 t104_path3 t104_path4 t104_path5 t105_path0
 t105_path1 t105_path2 t105_path3 t105_path4 t105_path5 t105_path6
 t105_path7 t106_path0 t106_path1 t106_path2 t106_path3 t106_path4
 t106_path5 t106_path6 t107_path0 t107_path1 t107_path2 t107_path3
 t107_path4 t107_path5 t107_path6 t107_path7 t108_path0 t108_path1
 t108_path2 t108_path3 t108_path4 t108_path5 t108_path6 t108_path7
 t109_path0 t109_path1 t109_path2 t109_path3 t109_path4 t109_path5
 t109_path6 t110_path0 t110_path1 t110_path2 t110_path3 t110_path4
 t110_path5 t111_path0 t111_path1 t111_path2 t111_path3 t111_path4
 t111_path5 t111_path6 t111_path7 t112_path0 t112_path1 t112_path2
 t112_path3 t112_path4 t112_path5 t112_path6 t112_path7 t113_path0
 t113_path1 t113_path2 t113_path3 t113_path4 t113_path5 t113_path6
 t114_path0 t114_path1 t114_path2 t114_path3 t114_path4 t114_path5
 t114_path6 t114_path7 t115_path0 t115_path1 t115_path2 t115_path3
 t115_path4 t115_path5 t116_path0 t116_path1 t116_path2 t116_path3
 t116_path4 t116_path5 t116_path6 t117_path0 t117_path1 t117_path2
 t117_path3 t117_path4 t118_path0 t118_path1 t118_path2 t118_path3
 t118_path4 t118_path5 t119_path0 t119_path1 t119_path2 t119_path3
 t120_path0 t120_path1 t120_path2 t120_path3 t120_path4 t120_path5
 t121_path0 t121_path1 t121_path2 t121_path3 t121_path4 t121_path5
 t121_path6 t122_path0 t122_path1 t122_path2 t122_path3 t122_path4
 t123_path0 t123_path1 t123_path2 t123_path3 t123_path4 t123_path5
 t123_path6 t123_path7 t124_path0 t124_path1 t124_path2 t124_path3
 t124_path4 t124_path5 t125_path0 t125_path1 t125_path2 t125_path3
 t125_path4 t125_path5 t126_path0 t126_path1 t126_path2 t126_path3
 t126_path4 t127_path0 t127_path1 t127_path2 t127_path3 t127_path4
 t127_path5 t128_path0 t128_path1 t128_path2 t128_path3 t128_path4
 t129_path0 t129_path1 t129_path2 t129_path3 t129_path4 t129_path5
 t130_path0 t130_path1 t130_path2 t130_path3 t130_path4 t130_path5
 t130_path6 t130_path7 t131_path0 t131_path1 t131_path2 t131_path3
 t131_path4 t131_path5 t131_path6 t131_path7 t132_path0 t132_path1
 t132_path2 t132_path3 t132_path4 t133_path0 t133_path1 t133_path2
 t133_path3 t133_path4 t134_path0 t134_path1 t134_path2 t134_path3
 t134_path4 t135_path0 t135_path1 t135_path2 t135_path3 t135_path4
 t135_path5 t135_path6 t136_path0 t136_path1 t136_path2 t136_path3
 t136_path4 t137_path0 t137_path1 t137_path2 t137_path3 t137_path4
 t137_path5 t137_path6 t137_path7 t138_path0 t138_path1 t138_path2
 t138_path3 t138_path4 t138_path5 t138_path6 t139_path0 t139_path1
 t139_path2 t139_path3 t139_path4 t139_path5 t139_path6 t139_path7
 t140_path0 t140_path1 t140_path2 t140_path3 t140_path4 t140_path5
 t140_path6 t140_path7 t141_path0 t141_path1 t141_path2 t141_path3
 t141_path4 t141_path5 t142_path0 t142_path1 t142_path2 t142_path3
 t142_path4 t143_path0 t143_path1 t143_path2 t143_path3 t143_path4
 t143_path5 t143_path6 t144_path0 t144_path1 t144_path2 t144_path3
 t144_path4 t144_path5 t145_path0 t145_path1 t145_path2 t145_path3
 t145_path4 t145_path5 t146_path0 t146_path1 t146_path2 t146_path3
 t146_path4 t146_path5 t146_path6 t146_path7 t147_path0 t147_path1
 t147_path2 t147_path3 t147_path4 t147_path5 t148_path0 t148_path1
 t148_path2 t148_path3 t148_path4 t148_path5 t149_path0 t149_path1
 t149_path2 t149_path3 t149_path4 t149_path5 t149_path6 t150_path0
 t150_path1 t150_path2 t150_path3 t150_path4 t150_path5 t150_path6
 t151_path0 t151_path1 t151_path2 t151_path3 t151_path4 t152_path0
 t152_path1 t152_path2 t152_path3 t152_path4 t152_path5 t153_path0
 t153_path1 t153_path2 t153_path3 t153_path4 t153_path5 t153_path6
 t154_path0 t154_path1 t154_path2 t154_path3 t154_path4 t154_path5
 t154_path6 t154_path7 t155_path0 t155_path1 t155_path2 t155_path3
 t155_path4 t156_path0 t156_path1 t156_path2 t156_path3 t156_path4
 t157_path0 t157_path1 t157_path2 t157_path3 t157_path4 t157_path5
 t157_path6 t158_path0 t158_path1 t158_path2 t158_path3 t158_path4
 t158_path5 t158_path6 t159_path0 t159_path1 t159_path2 t159_path3
 t159_path4 t160_path0 t160_path1 t160_path2 t160_path3 t160_path4
 t160_path5 t160_path6 t161_path0 t161_path1 t161_path2 t161_path3
 t161_path4 t161_path5 t161_path6 t162_path0 t162_path1 t162_path2
 t162_path3 t162_path4 t162_path5 t162_path6 t163_path0 t163_path1
 t163_path2 t163_path3 t163_path4 t163_path5 t164_path0 t164_path1
 t164_path2 t164_path3 t164_path4 t165_path0 t165_path1 t165_path2
 t165_path3 t165_path4 t166_path0 t166_path1 t166_path2 t166_path3
 t167_path0 t167_path1 t167_path2 t167_path3 t167_path4 t168_path0
 t168_path1 t168_path2 t168_path3 t168_path4 t169_path0 t169_path1
 t169_path2 t169_path3 t169_path4 t169_path5 t169_path6 t170_path0
 t170_path1 t170_path2 t170_path3 t170_path4 t171_path0 t171_path1
 t171_path2 t171_path3 t171_path4 t172_path0 t172_path1 t172_path2
 t172_path3 t172_path4 t172_path5 t172_path6 t173_path0 t173_path1
 t173_path2 t173_path3 t173_path4 t173_path5 t173_path6 t174_path0
 t174_path1 t174_path2 t174_path3 t175_path0 t175_path1 t175_path2
 t175_path3 t175_path4 t176_path0 t176_path1 t176_path2 t176_path3
 t176_path4 t176_path5 t176_path6 t177_path0 t177_path1 t177_path2
 t177_path3 t177_path4 t177_path5 t177_path6 t178_path0 t178_path1
 t178_path2 t178_path3 t178_path4 t179_path0 t179_path1 t179_path2
 t179_path3 t179_path4 t179_path5 t180_path0 t180_path1 t180_path2
 t180_path3 t180_path4 t181_path0 t181_path1 t181_path2 t181_path3
 t181_path4 t181_path5 t181_path6 t182_path0 t182_path1 t182_path2
 t182_path3 t182_path4 t182_path5 t182_path6 t182_path7 t183_path0
 t183_path1 t183_path2 t183_path3 t183_path4 t183_path5 t184_path0
 t184_path1 t184_path2 t184_path3 t184_path4 t184_path5 t185_path0
 t185_path1 t185_path2 t185_path3 t185_path4 t185_path5 t186_path0
 t186_path1 t186_path2 t186_path3 t186_path4 t186_path5 t186_path6
 t187_path0 t187_path1 t187_path2 t187_path3 t187_path4 t187_path5
 t187_path6 t188_path0 t188_path1 t188_path2 t188_path3 t188_path4
 t188_path5 t188_path6 t188_path7 t189_path0 t189_path1 t189_path2
 t189_path3 t189_path4 t190_path0 t190_path1 t190_path2 t190_path3
 t190_path4 t190_path5 t191_path0 t191_path1 t191_path2 t191_path3
 t191_path4 t191_path5 t191_path6 t192_path0 t192_path1 t192_path2
 t192_path3 t192_path4 t193_path0 t193_path1 t193_path2 t193_path3
 t193_path4 t193_path5 t193_path6 t193_path7 t194_path0 t194_path1
 t194_path2 t194_path3 t195_path0 t195_path1 t195_path2 t195_path3
 t196_path0 t196_path1 t196_path2 t196_path3 t196_path4 t196_path5
 t197_path0 t197_path1 t197_path2 t197_path3 t197_path4 t197_path5
 t198_path0 t198_path1 t198_path2 t198_path3 t198_path4 t198_path5
 t199_path0 t199_path1 t199_path2 t199_path3 t199_path4 t200_path0
 t200_path1 t200_path2 t200_path3 t200_path4 t201_path0 t201_path1
 t201_path2 t201_path3 t201_path4 t202_path0 t202_path1 t202_path2
 t202_path3 t202_path4 t202_path5 t202_path6 t203_path0 t203_path1
 t203_path2 t203_path3 t203_path4 t203_path5 t203_path6 t203_path7
 t204_path0 t204_path1 t204_path2 t204_path3 t204_path4 t205_path0
 t205_path1 t205_path2 t205_path3 t206_path0 t206_path1 t206_path2
 t206_path3 t207_path0 t207_path1 t207_path2 t207_path3 t207_path4
 t207_path5 t208_path0 t208_path1 t208_path2 t208_path3 t208_path4
 t209_path0 t209_path1 t209_path2 t209_path3 t209_path4 t210_path0
 t210_path1 t210_path2 t210_path3 t210_path4 t211_path0 t211_path1
 t211_path2 t211_path3 t211_path4 t211_path5 t211_path6 t212_path0
 t212_path1 t212_path2 t212_path3 t212_path4 t212_path5 t213_path0
 t213_path1 t213_path2 t213_path3 t214_path0 t214_path1 t214_path2
 t214_path3 t214_path4 t214_path5 t214_path6 t215_path0 t215_path1
 t215_path2 t215_path3 t215_path4 t215_path5 t216_path0 t216_path1
 t216_path2 t216_path3 t216_path4 t217_path0 t217_path1 t217_path2
 t217_path3 t217_path4 t218_path0 t218_path1 t218_path2 t218_path3
 t219_path0 t219_path1 t219_path2 t219_path3 t219_path4 t219_path5
 t219_path6 t220_path0 t220_path1 t220_path2 t220_path3 t220_path4
 t220_path5 t220_path6 t221_path0 t221_path1 t221_path2 t221_path3
 t221_path4 t221_path5 t222_path0 t222_path1 t222_path2 t222_path3
 t222_path4 t223_path0 t223_path1 t223_path2 t223_path3 t223_path4
 t223_path5 t223_path6 t224_path0 t224_path1 t224_path2 t224_path3
 t225_path0 t225_path1 t225_path2 t225_path3 t225_path4 t226_path0
 t226_path1 t226_path2 t226_path3 t226_path4 t226_path5 t226_path6
 t226_path7 t227_path0 t227_path1 t227_path2 t227_path3 t227_path4
 t228_path0 t228_path1 t228_path2 t228_path3 t228_path4 t229_path0
 t229_path1 t229_path2 t229_path3 t229_path4 t230_path0 t230_path1
 t230_path2 t230_path3 t230_path4 t230_path5 t230_path6 t231_path0
 t231_path1 t231_path2 t231_path3 t231_path4 t231_path5 t231_path6
 t231_path7 t232_path0 t232_path1 t232_path2 t232_path3 t232_path4
 t233_path0 t233_path1 t233_path2 t233_path3 t233_path4 t233_path5
 t233_path6 t234_path0 t234_path1 t234_path2 t234_path3 t234_path4
 t234_path5 t235_path0 t235_path1 t235_path2 t235_path3 t235_path4
 t236_path0 t236_path1 t236_path2 t236_path3 t236_path4 t236_path5
 t237_path0 t237_path1 t237_path2 t237_path3 t238_path0 t238_path1
 t238_path2 t238_path3 t238_path4 t238_path5 t239_path0 t239_path1
 t239_path2 t239_path3 t239_path4 t239_path5 t239_path6 t240_path0
 t240_path1 t240_path2 t240_path3 t240_path4 t240_path5 t241_path0
 t241_path1 t241_path2 t241_path3 t241_path4 t242_path0 t242_path1
 t242_path2 t242_path3 t242_path4 t243_path0 t243_path1 t243_path2
 t243_path3 t243_path4 t244_path0 t244_path1 t244_path2 t244_path3
 t245_path0 t245_path1 t245_path2 t245_path3 t245_path4 t245_path5
 t245_path6 t246_path0 t246_path1 t246_path2 t246_path3 t247_path0
 t247_path1 t247_path2 t247_path3 t247_path4 t248_path0 t248_path1
 t248_path2 t248_path3 t248_path4 t248_path5 t248_path6 t248_path7
 t249_path0 t249_path1 t249_path2 t249_path3 t249_path4 t250_path0
 t250_path1 t250_path2 t250_path3 t251_path0 t251_path1 t251_path2
 t251_path3 t251_path4 t252_path0 t252_path1 t252_path2 t252_path3
 t252_path4 t252_path5 t252_path6 t252_path7 t253_path0 t253_path1
 t253_path2 t253_path3 t253_path4 t253_path5 t254_path0 t254_path1
 t254_path2 t254_path3 t255_path0 t255_path1 t255_path2 t255_path3
 t255_path4 t255_path5 t255_path6 t256_path0 t256_path1 t256_path2
 t256_path3 t256_path4 t256_path5 t256_path6 t257_path0 t257_path1
 t257_path2 t257_path3 t257_path4 t257_path5 t257_path6 t258_path0
 t258_path1 t258_path2 t258_path3 t258_path4 t259_path0 t259_path1
 t259_path2 t259_path3 t259_path4 t260_path0 t260_path1 t260_path2
 t260_path3 t260_path4 t260_path5 t261_path0 t261_path1 t261_path2
 t261_path3 t262_path0 t262_path1 t262_path2 t262_path3 t262_path4
 t263_path0 t263_path1 t263_path2 t263_path3 t263_path4 t263_path5
 t264_path0 t264_path1 t264_path2 t264_path3 t264_path4 t265_path0
 t265_path1 t265_path2 t265_path3 t265_path4 t266_path0 t266_path1
 t266_path2 t266_path3 t266_path4 t266_path5 t266_path6 t267_path0
 t267_path1 t267_path2 t267_path3 t267_path4 t267_path5 t268_path0
 t268_path1 t268_path2 t268_path3 t268_path4 t268_path5 t269_path0
 t269_path1 t269_path2 t269_path3 t269_path4 t270_path0 t270_path1
 t270_path2 t270_path3 t270_path4 t270_path5 t271_path0 t271_path1
 t271_path2 t271_path3 t271_path4 t271_path5 t272_path0 t272_path1
 t272_path2 t272_path3 t272_path4 t272_path5 t272_path6 t272_path7
 t273_path0 t273_path1 t273_path2 t273_path3 t273_path4 t274_path0
 t274_path1 t274_path2 t274_path3 t274_path4 t275_path0 t275_path1
 t275_path2 t275_path3 t275_path4 t275_path5 t276_path0 t276_path1
 t276_path2 t276_path3 t276_path4 t277_path0 t277_path1 t277_path2
 t277_path3 t277_path4 t277_path5 t277_path6 t278_path0 t278_path1
 t278_path2 t278_path3 t279_path0 t279_path1 t279_path2 t279_path3
 t279_path4 t279_path5 t280_path0 t280_path1 t280_path2 t280_path3
 t280_path4 t280_path5 t280_path6 t281_path0 t281_path1 t281_path2
 t281_path3 t281_path4 t282_path0 t282_path1 t282_path2 t282_path3
 t283_path0 t283_path1 t283_path2 t283_path3 t283_path4 t283_path5
 t284_path0 t284_path1 t284_path2 t284_path3 t284_path4 t285_path0
 t285_path1 t285_path2 t285_path3 t285_path4 t285_path5 t286_path0
 t286_path1 t286_path2 t286_path3 t286_path4 t286_path5 t287_path0
 t287_path1 t287_path2 t287_path3 t287_path4 t287_path5 t288_path0
 t288_path1 t288_path2 t288_path3 t288_path4 t288_path5 t289_path0
 t289_path1 t289_path2 t289_path3 t289_path4 t289_path5 t290_path0
 t290_path1 t290_path2 t290_path3 t290_path4 t290_path5 t291_path0
 t291_path1 t291_path2 t291_path3 t291_path4 t292_path0 t292_path1
 t292_path2 t292_path3 t292_path4 t293_path0 t293_path1 t293_path2
 t293_path3 t293_path4 t294_path0 t294_path1 t294_path2 t294_path3
 t294_path4 t295_path0 t295_path1 t295_path2 t295_path3 t295_path4
 t295_path5 t296_path0 t296_path1 t296_path2 t296_path3 t296_path4
 t296_path5 t297_path0 t297_path1 t297_path2 t297_path3 t298_path0
 t298_path1 t298_path2 t298_path3 t298_path4 t299_path0 t299_path1
 t299_path2 t299_path3 t299_path4 t299_path5 t299_path6 t300_path0
 t300_path1 t300_path2 t300_path3 t300_path4 t300_path5 t301_path0
 t301_path1 t301_path2 t301_path3 t301_path4 t302_path0 t302_path1
 t302_path2 t302_path3 t303_path0 t303_path1 t303_path2 t303_path3
 t303_path4 t304_path0 t304_path1 t304_path2 t304_path3 t304_path4
 t305_path0 t305_path1 t305_path2 t305_path3 t305_path4 t306_path0
 t306_path1 t306_path2 t306_path3 t306_path4 t306_path5 t307_path0
 t307_path1 t307_path2 t307_path3 t307_path4 t307_path5 t308_path0
 t308_path1 t308_path2 t308_path3 t308_path4 t309_path0 t309_path1
 t309_path2 t309_path3 t309_path4 t309_path5 t310_path0 t310_path1
 t310_path2 t310_path3 t310_path4 t311_path0 t311_path1 t311_path2
 t311_path3 t312_path0 t312_path1 t312_path2 t312_path3 t312_path4
 t313_path0 t313_path1 t313_path2 t313_path3 t313_path4 t313_path5
 t313_path6 t314_path0 t314_path1 t314_path2 t314_path3 t314_path4
 t314_path5 t314_path6 t315_path0 t315_path1 t315_path2 t315_path3
 t315_path4 t316_path0 t316_path1 t316_path2 t316_path3 t316_path4
 t317_path0 t317_path1 t317_path2 t317_path3 t318_path0 t318_path1
 t318_path2 t318_path3 t319_path0 t319_path1 t319_path2 t319_path3
 t319_path4 t320_path0 t320_path1 t320_path2 t320_path3 t320_path4
 t320_path5 t320_path6 t321_path0 t321_path1 t321_path2 t321_path3
 t321_path4 t322_path0 t322_path1 t322_path2 t322_path3 t322_path4
 t322_path5 t322_path6 t323_path0 t323_path1 t323_path2 t323_path3
 t324_path0 t324_path1 t324_path2 t324_path3 t324_path4 t324_path5
 t324_path6 t325_path0 t325_path1 t325_path2 t325_path3 t325_path4
 t326_path0 t326_path1 t326_path2 t326_path3 t327_path0 t327_path1
 t327_path2 t327_path3 t327_path4 t328_path0 t328_path1 t328_path2
 t328_path3 t328_path4 t328_path5 t329_path0 t329_path1 t329_path2
 t329_path3 t329_path4 t329_path5 t329_path6 t330_path0 t330_path1
 t330_path2 t330_path3 t330_path4 t330_path5 t331_path0 t331_path1
 t331_path2 t331_path3 t331_path4 t332_path0 t332_path1 t332_path2
 t332_path3 t333_path0 t333_path1 t333_path2 t334_path0 t334_path1
 t334_path2 t334_path3 t334_path4 t334_path5 t334_path6 t335_path0
 t335_path1 t335_path2 t335_path3 t335_path4 t335_path5 t336_path0
 t336_path1 t336_path2 t336_path3 t336_path4 t336_path5 t336_path6
 t336_path7 t337_path0 t337_path1 t337_path2 t337_path3 t337_path4
 t337_path5 t338_path0 t338_path1 t338_path2 t338_path3 t338_path4
 t338_path5 t338_path6 t339_path0 t339_path1 t339_path2 t339_path3
 t339_path4 t339_path5 t340_path0 t340_path1 t340_path2 t340_path3
 t340_path4 t340_path5 t340_path6 t341_path0 t341_path1 t341_path2
 t342_path0 t342_path1 t342_path2 t342_path3 t342_path4 t343_path0
 t343_path1 t343_path2 t343_path3 t343_path4 t344_path0 t344_path1
 t344_path2 t344_path3 t344_path4 t344_path5 t344_path6 t345_path0
 t345_path1 t345_path2 t345_path3 t345_path4 t345_path5 t345_path6
 t346_path0 t346_path1 t346_path2 t346_path3 t346_path4 t347_path0
 t347_path1 t347_path2 t347_path3 t348_path0 t348_path1 t348_path2
 t348_path3 t348_path4 t349_path0 t349_path1 t349_path2 t349_path3
 t349_path4 t350_path0 t350_path1 t350_path2 t350_path3 t350_path4
 t350_path5 t350_path6 t351_path0 t351_path1 t351_path2 t351_path3
 t351_path4 t351_path5 t352_path0 t352_path1 t353_path0 t353_path1
 t353_path2 t353_path3 t353_path4 t354_path0 t354_path1 t354_path2
 t355_path0 t355_path1 t355_path2 t355_path3 t355_path4 t355_path5
 t355_path6 t356_path0 t356_path1 t356_path2 t356_path3 t356_path4
 t356_path5 t356_path6 t357_path0 t357_path1 t357_path2 t357_path3
 t358_path0 t358_path1 t358_path2 t358_path3 t358_path4 t358_path5
 t358_path6 t359_path0 t359_path1 t359_path2 t359_path3 t359_path4
 t359_path5 t359_path6 t360_path0 t360_path1 t361_path0 t361_path1
 t361_path2 t361_path3 t362_path0 t362_path1 t362_path2 t362_path3
 t362_path4 t363_path0 t363_path1 t363_path2 t363_path3 t363_path4
 t364_path0 t364_path1 t364_path2 t364_path3 t364_path4 t365_path0
 t365_path1 t365_path2 t365_path3 t365_path4 t366_path0 t366_path1
 t366_path2 t366_path3 t366_path4 t367_path0 t367_path1 t367_path2
 t367_path3 t368_path0 t368_path1 t368_path2 t368_path3 t368_path4
 t369_path0 t369_path1 t369_path2 t369_path3 t369_path4 t369_path5
 t370_path0 t370_path1 t370_path2 t371_path0 t371_path1 t371_path2
 t371_path3 t371_path4 t372_path0 t372_path1 t372_path2 t373_path0
 t373_path1 t373_path2 t373_path3 t373_path4 t374_path0 t374_path1
 t374_path2 t374_path3 t374_path4 t374_path5 t375_path0 t375_path1
 t375_path2 t375_path3 t376_path0 t376_path1 t376_path2 t376_path3
 t377_path0 t378_path0 t378_path1 t378_path2 t379_path0 t379_path1
 t379_path2 t379_path3 t379_path4 t380_path0 t380_path1 t381_path0
 t381_path1 t381_path2 t381_path3 t381_path4 t382_path0 t382_path1
 t382_path2 t382_path3 t382_path4 t382_path5 t383_path0 t383_path1
 t383_path2 t383_path3 t383_path4 t383_path5 t383_path6 t384_path0
 t384_path1 t384_path2 t384_path3 t384_path4 t385_path0 t385_path1
 t385_path2 t385_path3 t385_path4 t386_path0 t386_path1 t386_path2
 t386_path3 t387_path0 t387_path1 t387_path2 t387_path3 t387_path4
 t388_path0 t388_path1 t388_path2 t388_path3 t389_path0 t389_path1
 t389_path2 t389_path3 t390_path0 t390_path1 t390_path2 t391_path0
 t391_path1 t391_path2 t391_path3 t392_path0 t392_path1 t392_path2
 t392_path3 t393_path0 t393_path1 t393_path2 t393_path3 t393_path4
 t394_path0 t394_path1 t395_path0 t395_path1 t395_path2 t395_path3
 t395_path4 t395_path5 t396_path0 t396_path1 t396_path2 t396_path3
 t396_path4 t397_path0 t397_path1 t397_path2 t397_path3 t397_path4
 t398_path0 t398_path1 t398_path2 t398_path3 t399_path0 t399_path1
 t399_path2 t399_path3 t400_path0 t400_path1 t400_path2 t401_path0
 t402_path0 t402_path1 t402_path2 t402_path3 t402_path4 t403_path0
 t403_path1 t403_path2 t403_path3 t403_path4 t404_path0 t404_path1
 t404_path2 t404_path3 t404_path4 t404_path5 t405_path0 t405_path1
 t405_path2 t405_path3 t405_path4 t406_path0 t407_path0 t408_path0
 t408_path1 t408_path2 t408_path3 t408_path4 t409_path0 t409_path1
 t409_path2 t409_path3 t409_path4 t409_path5 t410_path0 t410_path1
 t410_path2 t410_path3 t411_path0 t411_path1 t411_path2 t412_path0
 t412_path1 t412_path2 t413_path0 t413_path1 t413_path2 t413_path3
 t413_path4 t414_path0 t414_path1 t414_path2 t414_path3 t415_path0
 t415_path1 t415_path2 t415_path3 t415_path4 t416_path0 t416_path1
 t416_path2 t416_path3 t416_path4 t416_path5 t416_path6 t417_path0
 t418_path0 t418_path1 t419_path0 t419_path1 t419_path2 t419_path3
 t419_path4 t419_path5 t419_path6 t419_path7 t420_path0 t421_path0
 t421_path1 t421_path2 t421_path3 t422_path0 t422_path1 t422_path2
 t422_path3 t422_path4 t423_path0 t423_path1 t423_path2 t423_path3
 t424_path0 t424_path1 t424_path2 t424_path3 t424_path4 t424_path5
 t425_path0 t425_path1 t425_path2 t426_path0 t426_path1 t426_path2
 t426_path3 t426_path4 t426_path5 t426_path6 t427_path0 t427_path1
 t427_path2 t427_path3 t428_path0 t428_path1 t428_path2 t428_path3
 t428_path4 t429_path0 t429_path1 t429_path2 t429_path3 t429_path4
 t429_path5 t430_path0 t430_path1 t430_path2 t430_path3 t430_path4
 t430_path5 t430_path6 t431_path0 t431_path1 t431_path2 t431_path3
 t432_path0 t432_path1 t432_path2 t432_path3 t432_path4 t432_path5
 t433_path0 t433_path1 t433_path2 t434_path0 t435_path0 t435_path1
 t435_path2 t435_path3 t435_path4 t436_path0 t437_path0 t437_path1
 t437_path2 t438_path0 t438_path1 t439_path0 t439_path1 t439_path2
 t439_path3 t439_path4 t440_path0 t440_path1 t440_path2 t440_path3
 t440_path4 t440_path5 t441_path0 t441_path1 t441_path2 t441_path3
 t441_path4 t442_path0 t442_path1 t442_path2 t443_path0 t444_path0
 t444_path1 t444_path2 t444_path3 t444_path4 t445_path0 t445_path1
 t445_path2 t446_path0 t446_path1 t446_path2 t446_path3 t447_path0
 t447_path1 t447_path2 t448_path0 t448_path1 t448_path2 t449_path0
 t449_path1 t449_path2 t449_path3 t449_path4 t449_path5 t450_path0
 t451_path0 t451_path1 t451_path2 t451_path3 t451_path4 t451_path5
 t452_path0 t452_path1 t452_path2 t452_path3 t452_path4 t453_path0
 t453_path1 t453_path2 t453_path3 t453_path4 t453_path5 t454_path0
 t454_path1 t455_path0 t456_path0 t457_path0 t457_path1 t457_path2
 t458_path0 t458_path1 t458_path2 t458_path3 t459_path0 t459_path1
 t460_path0 t461_path0 t461_path1 t461_path2 t461_path3 t461_path4
 t462_path0 t462_path1 t463_path0 t463_path1 t463_path2 t463_path3
 t463_path4 t463_path5 t463_path6 t464_path0 t464_path1 t464_path2
 t465_path0 t465_path1 t465_path2 t465_path3 t466_path0 t466_path1
 t466_path2 t466_path3 t466_path4 t467_path0 t468_path0 t469_path0
 t470_path0 t470_path1 t470_path2 t471_path0 t471_path1 t471_path2
 t471_path3 t472_path0 t473_path0 t473_path1 t473_path2 t474_path0
 t474_path1 t474_path2 t475_path0 t475_path1 t475_path2 t476_path0
 t476_path1 t476_path2 t476_path3 t477_path0 t477_path1 t477_path2
 t477_path3 t478_path0 t478_path1 t478_path2 t478_path3 t479_path0
 t479_path1 t479_path2 t479_path3 t479_path4 t479_path5 t480_path0
 t480_path1 t480_path2 t480_path3 t480_path4 t481_path0 t481_path1
 t482_path0 t483_path0 t484_path0 t484_path1 t484_path2 t484_path3
 t485_path0 t486_path0 t487_path0 t488_path0 t488_path1 t488_path2
 t488_path3 t489_path0 t490_path0 t490_path1 t490_path2 t491_path0
 t491_path1 t491_path2 t491_path3 t492_path0 t492_path1 t492_path2
 t493_path0 t493_path1 t493_path2 t493_path3 t493_path4 t494_path0
 t494_path1 t494_path2 t495_path0 t495_path1 t496_path0 t496_path1
 t497_path0 t497_path1 t497_path2 t497_path3 t498_path0 t498_path1
 t498_path2 t498_path3 t498_path4 t499_path0 t499_path1 t500_path0
 t500_path1 t500_path2 t500_path3 t501_path0 t501_path1 t501_path2
 t501_path3 t501_path4 t501_path5 t501_path6 t502_path0 t502_path1
 t503_path0 t504_path0 t504_path1 t504_path2 t504_path3 t505_path0
 t505_path1 t506_path0 t506_path1 t506_path2 t506_path3 t506_path4
 t507_path0 t507_path1 t508_path0 t508_path1 t508_path2 t508_path3
 t508_path4 t508_path5 t509_path0 t509_path1 t509_path2 t510_path0
 t511_path0 t512_path0 t513_path0 t513_path1 t513_path2 t513_path3
 t513_path4 t513_path5 t514_path0 t515_path0 t515_path1 t515_path2
 t515_path3 t516_path0 t516_path1 t516_path2 t516_path3 t516_path4
 t517_path0 t517_path1 t517_path2 t517_path3 t517_path4 t517_path5
 t518_path0 t518_path1 t518_path2 t518_path3 t518_path4 t518_path5
 t519_path0 t519_path1 t520_path0 t520_path1 t520_path2 t520_path3
 t520_path4 t520_path5 t521_path0 t521_path1 t521_path2 t521_path3
 t521_path4 t522_path0 t522_path1 t522_path2 t522_path3 t522_path4
 t522_path5 t523_path0 t523_path1 t523_path2 t523_path3 t523_path4
 t524_path0 t524_path1 t524_path2 t524_path3 t525_path0 t525_path1
 t526_path0 t526_path1 t526_path2 t526_path3 t526_path4 t526_path5
 t527_path0 t527_path1 t527_path2 t527_path3 t527_path4 t527_path5
 t528_path0 t528_path1 t528_path2 t529_path0 t529_path1 t529_path2
 t529_path3 t529_path4 t530_path0 t531_path0 t531_path1 t531_path2
 t532_path0 t533_path0 t533_path1 t533_path2 t534_path0 t535_path0
 t535_path1 t535_path2 t535_path3 t535_path4 t536_path0 t537_path0
 t537_path1 t538_path0 t538_path1 t538_path2 t538_path3 t538_path4
 t539_path0 t539_path1 t540_path0 t540_path1 t540_path2 t540_path3
 t540_path4 t541_path0 t541_path1 t541_path2 t542_path0 t543_path0
 t543_path1 t543_path2 t543_path3 t543_path4 t543_path5 t543_path6
 t544_path0 t545_path0 t545_path1 t545_path2 t545_path3 t545_path4
 t546_path0 t546_path1 t547_path0 t547_path1 t548_path0 t548_path1
 t548_path2 t548_path3 t549_path0 t549_path1 t549_path2 t549_path3
 t549_path4 t549_path5 t550_path0 t550_path1 t550_path2 t550_path3
 t551_path0 t552_path0 t552_path1 t552_path2 t552_path3 t552_path4
 t553_path0 t553_path1 t553_path2 t553_path3 t554_path0 t555_path0
 t556_path0 t556_path1 t557_path0 t557_path1 t557_path2 t557_path3
 t558_path0 t558_path1 t558_path2 t558_path3 t559_path0 t559_path1
 t559_path2 t559_path3 t559_path4 t560_path0 t560_path1 t560_path2
 t560_path3 t561_path0 t561_path1 t561_path2 t562_path0 t562_path1
 t562_path2 t563_path0 t564_path0 t564_path1 t564_path2 t565_path0
 t566_path0 t566_path1 t566_path2 t566_path3 t566_path4 t566_path5
 t567_path0 t567_path1 t567_path2 t567_path3 t567_path4 t567_path5
 t568_path0 t568_path1 t568_path2 t568_path3 t569_path0 t570_path0
 t571_path0 t571_path1 t571_path2 t571_path3 t572_path0 t572_path1
 t572_path2 t573_path0 t574_path0 t574_path1 t574_path2 t574_path3
 t575_path0 t575_path1 t576_path0 t576_path1 t576_path2 t576_path3
 t576_path4 t576_path5 t577_path0 t578_path0 t578_path1 t578_path2
 t579_path0 t579_path1 t580_path0 t580_path1 t580_path2 t580_path3
 t580_path4 t580_path5 t581_path0 t581_path1 t582_path0 t582_path1
 t582_path2 t583_path0 t584_path0 t584_path1 t584_path2 t584_path3
 t584_path4 t585_path0 t585_path1 t585_path2 t585_path3 t585_path4
 t586_path0 t586_path1 t586_path2 t586_path3 t586_path4 t587_path0
 t587_path1 t587_path2 t587_path3 t587_path4 t587_path5 t588_path0
 t588_path1 t588_path2 t588_path3 t588_path4 t588_path5 t588_path6
 t588_path7 t589_path0 t590_path0 t590_path1 t591_path0 t592_path0
 t593_path0 t594_path0 t595_path0 t595_path1 t595_path2 t595_path3
 t595_path4 t596_path0 t597_path0 t597_path1 t597_path2 t597_path3
 t597_path4 t598_path0 t598_path1 t599_path0 t600_path0 t600_path1
 t600_path2 t600_path3 t601_path0 t601_path1 t601_path2 t601_path3
 t602_path0 t603_path0 t603_path1 t604_path0 t605_path0 t605_path1
 t605_path2 t605_path3 t606_path0 t607_path0 t607_path1 t608_path0
 t608_path1 t609_path0 t610_path0 t611_path0 t611_path1 t612_path0
 t613_path0 t613_path1 t614_path0 t615_path0 t616_path0 t616_path1
 t616_path2 t616_path3 t617_path0 t617_path1 t617_path2 t618_path0
 t618_path1 t619_path0 t619_path1 t619_path2 t619_path3 t619_path4
 t620_path0 t620_path1 t621_path0 t621_path1 t621_path2 t621_path3
 t622_path0 t622_path1 t622_path2 t622_path3 t623_path0 t623_path1
 t623_path2 t623_path3 t623_path4 t623_path5 t624_path0 t624_path1
 t625_path0 t625_path1 t625_path2 t626_path0 t626_path1 t626_path2
 t626_path3 t627_path0 t627_path1 t628_path0 t629_path0 t630_path0
 t631_path0 t632_path0 t632_path1 t632_path2 t632_path3 t633_path0
 t634_path0 t635_path0 t636_path0 t637_path0 t638_path0 t639_path0
 t640_path0 t640_path1 t640_path2 t640_path3 t641_path0 t642_path0
 t643_path0 t644_path0 t645_path0 t646_path0 t646_path1 t646_path2
 t647_path0 t647_path1 t647_path2 t648_path0 t649_path0 t649_path1
 t650_path0 t651_path0 t652_path0 t652_path1 t652_path2 t653_path0
 t653_path1 t654_path0 t655_path0 t655_path1 t655_path2 t656_path0
 t656_path1 t656_path2 t657_path0 t657_path1 t658_path0 t659_path0
 t659_path1 t659_path2 t659_path3 t659_path4 t660_path0 t660_path1
 t660_path2 t660_path3 t661_path0 t661_path1 t661_path2 t661_path3
 t661_path4 t662_path0 t663_path0 t663_path1 t663_path2 t663_path3
 t663_path4 t663_path5 t663_path6 t664_path0 t664_path1 t664_path2
 t665_path0 t665_path1 t665_path2 t666_path0 t666_path1 t666_path2
 t667_path0 t667_path1 t667_path2 t667_path3 t668_path0 t668_path1
 t668_path2 t668_path3 t669_path0 t670_path0 t671_path0 t671_path1
 t672_path0 t673_path0 t673_path1 t673_path2 t673_path3 t673_path4
 t674_path0 t674_path1 t675_path0 t676_path0 t676_path1 t677_path0
 t678_path0 t678_path1 t678_path2 t678_path3 t678_path4 t679_path0
 t680_path0 t680_path1 t680_path2 t680_path3 t680_path4 t681_path0
 t681_path1 t682_path0 t683_path0 t683_path1 t683_path2 t683_path3
 t684_path0 t684_path1 t684_path2 t685_path0 t686_path0 t687_path0
 t687_path1 t687_path2 t688_path0 t689_path0 t690_path0 t691_path0
 t692_path0 t692_path1 t693_path0 t693_path1 t694_path0 t695_path0
 t696_path0 t697_path0 t697_path1 t698_path0 t698_path1 t699_path0
 t700_path0 t701_path0 t702_path0 t702_path1 t702_path2 t702_path3
 t703_path0 t704_path0 t705_path0 t705_path1 t705_path2 t705_path3
 t705_path4 t706_path0 t706_path1 t706_path2 t707_path0 t708_path0
 t709_path0 t709_path1 t709_path2 t710_path0 t710_path1 t710_path2
 t710_path3 t711_path0 t712_path0 t712_path1 t712_path2 t712_path3
 t713_path0 t713_path1 t714_path0 t715_path0 t715_path1 t715_path2
 t715_path3 t716_path0 t717_path0 t718_path0 t719_path0 t720_path0
 t720_path1 t721_path0 t722_path0 t723_path0 t724_path0 t725_path0
 t726_path0 t726_path1 t726_path2 t727_path0 t727_path1 t727_path2
 t728_path0 t728_path1 t729_path0 t730_path0 t730_path1 t730_path2
 t730_path3 t730_path4 t731_path0 t732_path0 t733_path0 t734_path0
 t735_path0 t736_path0 t737_path0 t738_path0 t739_path0 t739_path1
 t740_path0 t741_path0 t741_path1 t741_path2 t742_path0 t742_path1
 t742_path2 t742_path3 t742_path4 t742_path5 t742_path6 t743_path0
 t744_path0 t745_path0 t746_path0 t747_path0 t748_path0 t749_path0
 t749_path1 t749_path2 t749_path3 t750_path0 t751_path0 t752_path0
 t753_path0 t754_path0 t755_path0 t756_path0 t757_path0 t757_path1
 t757_path2 t757_path3 t758_path0 t759_path0 t759_path1 t760_path0
 t761_path0 t761_path1 t761_path2 t761_path3 t762_path0 t763_path0
 t763_path1 t763_path2 t763_path3 t763_path4 t764_path0 t764_path1
 t764_path2 t764_path3 t765_path0 t766_path0 t766_path1 t767_path0
 t768_path0 t768_path1 t768_path2 t769_path0 t770_path0 t771_path0
 t772_path0 t772_path1 t772_path2 t773_path0 t774_path0 t775_path0
 t775_path1 t776_path0 t777_path0 t777_path1 t778_path0 t778_path1
 t778_path2 t778_path3 t779_path0 t780_path0 t781_path0 t781_path1
 t781_path2 t782_path0 t783_path0 t784_path0 t784_path1 t784_path2
 t784_path3 t784_path4 t785_path0 t785_path1 t786_path0 t787_path0
 t787_path1 t788_path0 t789_path0 t790_path0 t791_path0 t792_path0
 t793_path0 t793_path1 t793_path2 t793_path3 t794_path0 t795_path0
 t795_path1 t795_path2 t795_path3 t795_path4 t796_path0 t797_path0
 t798_path0 t799_path0 t800_path0 t801_path0 t802_path0 t803_path0
 t804_path0 t805_path0 t805_path1 t806_path0 t806_path1 t807_path0
 t808_path0 t809_path0 t810_path0 t811_path0 t812_path0 t813_path0
 t814_path0 t814_path1 t814_path2 t814_path3 t814_path4 t815_path0
 t816_path0 t817_path0 t817_path1 t817_path2 t818_path0 t819_path0
 t820_path0 t821_path0 t822_path0 t823_path0 t824_path0 t825_path0
 t825_path1 t826_path0 t827_path0 t827_path1 t827_path2 t827_path3
 t828_path0 t829_path0 t829_path1 t829_path2 t829_path3 t829_path4
 t830_path0 t831_path0 t831_path1 t832_path0 t832_path1 t833_path0
 t834_path0 t835_path0 t835_path1 t835_path2 t836_path0 t837_path0
 t838_path0 t839_path0 t839_path1 t840_path0 t841_path0 t841_path1
 t842_path0 t842_path1 t843_path0 t844_path0 t844_path1 t845_path0
 t845_path1 t845_path2 t846_path0 t847_path0 t848_path0 t848_path1
 t848_path2 t849_path0 t849_path1 t849_path2 t850_path0 t851_path0
 t852_path0 t853_path0 t854_path0 t854_path1 t855_path0 t856_path0
 t856_path1 t857_path0 t858_path0 t858_path1 t858_path2 t858_path3
 t859_path0 t860_path0 t860_path1 t860_path2 t861_path0 t862_path0
 t863_path0 t864_path0 t865_path0 t865_path1 t865_path2 t866_path0
 t867_path0 t868_path0 t869_path0 t870_path0 t871_path0 t872_path0
 t873_path0 t874_path0 t875_path0 t875_path1 t876_path0 t876_path1
 t876_path2 t876_path3 t877_path0 t877_path1 t878_path0 t879_path0
 t880_path0 t880_path1 t880_path2 t880_path3 t881_path0 t881_path1
 t881_path2 t881_path3 t881_path4 t882_path0 t883_path0 t883_path1
 t884_path0 t885_path0 t886_path0 t886_path1 t886_path2 t886_path3
 t886_path4 t886_path5 t887_path0 t888_path0 t888_path1 t889_path0
 t889_path1 t889_path2 t889_path3 t889_path4 t890_path0 t891_path0
 t892_path0 t893_path0 t894_path0 t894_path1 t895_path0 t896_path0
 t896_path1 t897_path0 t898_path0 t899_path0 t900_path0 t901_path0
 t902_path0 t903_path0 t903_path1 t903_path2 t904_path0 t904_path1
 t905_path0 t906_path0 t906_path1 t906_path2 t907_path0 t908_path0
 t909_path0 t909_path1 t910_path0 t911_path0 t912_path0 t913_path0 z_change
Generals
 x_Bedroom_AbvGr x_Full_Bath x_TotRms_AbvGrd x_Kitchen_Qual x_Exter_Qual
 x_Exter_Cond x_Heating_QC x_Fireplace_Qu x_Bsmt_Cond x_Garage_Qual
 x_Garage_Cond x_Pool_QC x_Utilities x_Roof_Matl x_Half_Bath
 x_Kitchen_AbvGr Overall_Qual_calc R8_other_rooms
General Constraints
 exp_expm1: y_price = PWL ( y_log ) : (10, 22025.46579480672) 
  (10.025, 22583.06843112946) (10.05, 23154.78684539547) 
  (10.075, 23740.9783802246) (10.1, 24342.00942440838) 
  (10.125, 24958.25564191459) (10.15, 25590.1022066897) 
  (10.175, 26237.94404340504) (10.2, 26902.18607429754) 
  (10.225, 27583.24347225953) (10.25, 28281.54192033498) 
  (10.275, 28997.51787778548) (10.3, 29731.61885289144) 
  (10.325, 30484.30368265924) (10.35, 31256.04281960959) 
  (10.375, 32047.31862582525) (10.4, 32858.62567444333) 
  (10.425, 33690.47105877948) (10.45, 34543.37470927775) 
  (10.475, 35417.86971848416) (10.5, 36314.50267424664) 
  (10.525, 37233.83400135042) (10.55, 38176.43831180172) 
  (10.575, 39142.90476397888) (10.6, 40133.83743087578) 
  (10.625, 41149.85567766677) (10.65, 42191.5945488303) 
  (10.675, 43259.70516507232) (10.7, 44354.85513029784) 
  (10.725, 45477.72894888544) (10.75, 46629.02845352433) 
  (10.775, 47809.47324388286) (10.8, 49019.80113638175) 
  (10.825, 50260.76862535324) (10.85, 51533.15135587504) 
  (10.875, 52837.74460857319) (10.9, 54175.36379669875) 
  (10.925, 55546.84497578751) (10.95, 56953.04536622178) 
  (10.975, 58394.84388902149) (11, 59873.14171519782) 
  (11.025, 61388.8628290148) (11.05, 62942.95460550949) 
  (11.075, 64536.38840263205) (11.1, 66170.16016837658) 
  (11.125, 67845.29106328034) (11.15, 69562.82809868279) 
  (11.175, 71323.84479114162) (11.2, 73129.44183341545) 
  (11.225, 74980.74778243311) (11.25, 76878.91976467776) 
  (11.275, 78825.14419942902) (11.3, 80820.6375403136) 
  (11.325, 82866.64703562824) (11.35, 84964.45150791103) 
  (11.375, 87115.36215324633) (11.4, 89320.72336080558) 
  (11.425, 91581.9135531341) (11.45, 93900.34604770991) 
  (11.475, 96277.46994031382) (11.5, 98714.77101076049) 
  (11.525, 101213.7726515598) (11.55, 103776.0368200868) 
  (11.575, 106403.1650148552) (11.6, 109096.7992765075) 
  (11.625, 111858.6232141423) (11.65, 114690.3630576256) 
  (11.675, 117593.7887365404) (11.7, 120570.7149864505) 
  (11.725, 123623.0024831697) (11.75, 126752.5590057434) 
  (11.775, 129961.3406288742) (11.8, 133251.352945531) 
  (11.825, 136624.6523205107) (11.85, 140083.3471757332) 
  (11.875, 143629.5993080729) (11.9, 147265.6252405527) 
  (11.925, 150993.6976077433) (11.95, 154816.1465762327) 
  (11.975, 158735.3613010583) (12, 162753.7914190039) 
  (12.025, 166873.9485797043) (12.05, 171098.4080155075) 
  (12.075, 175429.8101510782) (12.1, 179870.862253751) 
  (12.125, 184424.3401256583) (12.15, 189093.0898386982) 
  (12.175, 193880.0295134221) (12.2, 198788.1511429544) 
  (12.225, 203820.5224630873) (12.25, 208980.288869713) 
  (12.275, 214270.6753848003) (12.3, 219694.9886721379) 
  (12.325, 225256.6191041063) (12.35, 230959.0428807727) 
  (12.375, 236805.8242026268) (12.4, 242800.6174983236) 
  (12.425, 248947.1697088201) (12.45, 255249.3226293344) 
  (12.475, 261711.0153105955) (12.5, 268336.2865208745) 
  (12.525, 275129.2772703487) (12.55, 282094.2333993672) 
  (12.575, 289235.5082322386) (12.6, 296557.5652982028) 
  (12.625, 304064.9811212787) (12.65, 311762.4480807426) 
  (12.675, 319654.7773440182) (12.7, 327746.9018738116) 
  (12.725, 336043.8795113759) (12.75, 344550.8961378237) 
  (12.775, 353273.2689154733) (12.8, 362216.4496112482) 
  (12.825, 371386.0280042076) (12.85, 380787.7353793437) 
  (12.875, 390427.4481098168) (12.9, 400311.1913298826) 
  (12.925, 410445.1427007969) (12.95, 420835.6362720539) 
  (12.975, 431489.1664403764) (13, 442412.3920089205) 
  (13.025, 453612.1403492478) (13.05, 465095.4116686539) 
  (13.075, 476869.3833855246) (13.1, 488941.4146154599) 
  (13.125, 501319.0507709557) (13.15, 514010.0282775354) 
  (13.175, 527022.279409267) (13.2, 540363.9372466916) 
  (13.225, 554043.3407602672) (13.25, 568069.0400224912) 
  (13.275, 582449.801551978) (13.3, 597194.6137928166) 
  (13.325, 612312.6927326412) (13.35, 627813.4876629263) 
  (13.375, 643706.6870850961) (13.4, 660002.2247661569) 
  (13.425, 676710.285947624) (13.45, 693841.3137116284) 
  (13.475, 711406.0155081871) (13.5, 729415.3698477013) 
  (13.525, 747880.6331628875) (13.55, 766813.346844413) 
  (13.575, 786225.3444546394) (13.6, 806128.75912399) 
  (13.625, 826536.0311345428) (13.65, 847459.9156956177) 
  (13.675, 868913.4909161953) (13.7, 890910.1659791603) 
  (13.725, 913463.6895224825) (13.75, 936588.1582325544) 
  (13.775, 960298.0256550852) (13.8, 984608.1112290357) 
  (13.825, 1.009533609549248e+06) (13.85, 1.0350900998635698e+06) 
  (13.875, 1.06129355581038e+06) (13.9, 1.0881603554026405e+06) 
  (13.925, 1.1157072912646844e+06) (13.95, 1.143951581128153e+06) 
  (13.975, 1.1729108785936439e+06) (14, 1.2026032841647768e+06)
End
