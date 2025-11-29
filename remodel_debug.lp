\ Model remodel_embed
\ LP format - for model browsing. Use MPS format to capture full model detail.
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
   + 0 x_b2_is_No_aplica + 0 B1_upg_flag + 0 B2_upg_flag
   + 0 const_MS_SubClass_40 + 0 const_MS_SubClass_45
   + 0 const_MS_SubClass_60 + 0 const_MS_SubClass_75
   + 0 const_MS_SubClass_80 + 0 const_MS_SubClass_85
   + 0 const_MS_SubClass_90 + 0 const_MS_SubClass_150
   + 0 const_MS_SubClass_180 + 0 const_MS_Zoning_A_(agr)
   + 0 const_MS_Zoning_I_(all) + 0 const_MS_Zoning_RH + 0 const_Alley_Grvl
   + 0 const_Lot_Shape_IR2 + 0 const_Lot_Config_FR2
   + 0 const_Lot_Config_FR3 + 0 const_Land_Slope_Sev
   + 0 const_Neighborhood_Blueste + 0 const_Neighborhood_BrDale
   + 0 const_Neighborhood_CollgCr + 0 const_Neighborhood_Gilbert
   + 0 const_Neighborhood_Greens + 0 const_Neighborhood_GrnHill
   + 0 const_Neighborhood_Landmrk + 0 const_Neighborhood_MeadowV
   + 0 const_Neighborhood_NPkVill + 0 const_Neighborhood_NoRidge
   + 0 const_Neighborhood_SWISU + 0 const_Neighborhood_Sawyer
   + 0 const_Neighborhood_Timber + 0 const_Neighborhood_Veenker
   + 0 const_Condition_1_RRAe + 0 const_Condition_1_RRAn
   + 0 const_Condition_1_RRNe + 0 const_Condition_1_RRNn
   + 0 const_Condition_2_Artery + 0 const_Condition_2_Feedr
   + 0 const_Condition_2_Norm + 0 const_Condition_2_PosA
   + 0 const_Condition_2_PosN + 0 const_Condition_2_RRAe
   + 0 const_Condition_2_RRAn + 0 const_Condition_2_RRNn
   + 0 const_Bldg_Type_2fmCon + 0 const_Bldg_Type_Duplex
   + 0 const_Bldg_Type_Twnhs + 0 const_Bldg_Type_TwnhsE
   + 0 const_House_Style_1.5Unf + 0 const_House_Style_2.5Fin
   + 0 const_House_Style_2.5Unf + 0 const_House_Style_2Story
   + 0 const_House_Style_SFoyer + 0 const_House_Style_SLvl
   + 0 const_Roof_Style_Gambrel + 0 const_Roof_Style_Mansard
   + 0 const_Roof_Style_Shed + 0 const_Exterior_1st_WdShing
   + 0 const_Exterior_2nd_Brk_Cmn + 0 const_Exterior_2nd_CmentBd
   + 0 const_Foundation_Stone + 0 const_Foundation_Wood
   + 0 const_Bsmt_Exposure_No_aplica + 0 const_BsmtFin_Type_1_No_aplica
   + 0 const_BsmtFin_Type_2_No_aplica + 0 const_FunctioNo_aplical_Min2
   + 0 const_FunctioNo_aplical_Mod + 0 const_FunctioNo_aplical_Sal
   + 0 const_FunctioNo_aplical_Sev + 0 const_Garage_Type_2Types
   + 0 const_Garage_Type_No_aplica + 0 const_Garage_Finish_No_aplica
   + 0 const_Misc_Feature_Elev + 0 const_Misc_Feature_Gar2
   + 0 const_Misc_Feature_Othr + 0 const_Misc_Feature_Shed
   + 0 const_Misc_Feature_TenC + 0 const_Mo_Sold_8 + 0 const_Mo_Sold_11
   + 0 const_Mo_Sold_12 + 0 const_Sale_Type_CWD + 0 const_Sale_Type_Con
   + 0 const_Sale_Type_ConLD + 0 const_Sale_Type_ConLI
   + 0 const_Sale_Type_ConLw + 0 const_Sale_Type_VWD
   + 0 const_Sale_Condition_AdjLand + y_price - cost_model
   - 237393.28125 Constant
Subject To
 OverallQual_from_upgrades_float: - 0.0714285714285714 x_Kitchen_Qual
   - 0.0714285714285714 x_Exter_Qual - 0.0714285714285714 x_Exter_Cond
   - 0.0714285714285714 x_Heating_QC - 0.0714285714285714 x_Bsmt_Cond
   - 0.0714285714285714 x_Garage_Qual - 0.0714285714285714 x_Garage_Cond
   + Overall_Qual_float = 5.714285714285714
 OverallQual_floor_ub: - Overall_Qual_float + Overall_Qual_calc <= 0
 OverallQual_floor_lb: - Overall_Qual_float + Overall_Qual_calc >= -1
 Kitchen_Qual_upgrade_only: x_Kitchen_Qual >= 3
 Exter_Qual_upgrade_only: x_Exter_Qual >= 3
 Exter_Cond_upgrade_only: x_Exter_Cond >= 2
 Heating_QC_upgrade_only: x_Heating_QC >= 4
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
 MVT_forbid_BrkCmn: x_mvt_is_BrkCmn = 0
 MVT_forbid_BrkFace: x_mvt_is_BrkFace = 0
 MVT_forbid_CBlock: x_mvt_is_CBlock = 0
 MVT_forbid_Stone: x_mvt_is_Stone = 0
 MVT_stay_noaplica: x_mvt_is_No_aplica = 1
 MVT_lin_p_le_a_BrkCmn: - x_Mas_Vnr_Area + mv_area_mvt_BrkCmn <= 0
 MVT_lin_p_le_UBb_BrkCmn: - 2000 x_mvt_is_BrkCmn + mv_area_mvt_BrkCmn <= 0
 MVT_lin_p_ge_a_minus_UB1b_BrkCmn: - x_Mas_Vnr_Area - 2000 x_mvt_is_BrkCmn
   + mv_area_mvt_BrkCmn >= -2000
 MVT_lin_p_le_a_BrkFace: - x_Mas_Vnr_Area + mv_area_mvt_BrkFace <= 0
 MVT_lin_p_le_UBb_BrkFace: - 2000 x_mvt_is_BrkFace + mv_area_mvt_BrkFace
   <= 0
 MVT_lin_p_ge_a_minus_UB1b_BrkFace: - x_Mas_Vnr_Area
   - 2000 x_mvt_is_BrkFace + mv_area_mvt_BrkFace >= -2000
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
 ROOF_incompat_Gable_Membran: x_roof_matl_is_Membran = 0
 ROOF_change_eq_sum_nonbase: - x_roof_matl_is_ClyTile
   - x_roof_matl_is_Membran - x_roof_matl_is_Metal - x_roof_matl_is_Roll
   - x_roof_matl_is_Tar&Grv - x_roof_matl_is_WdShake
   - x_roof_matl_is_WdShngl + roof_change_mat = 0
 GaFin_pick_one: x_garage_finish_is_Fin + x_garage_finish_is_RFn
   + x_garage_finish_is_Unf + x_garage_finish_is_No_aplica = 1
 UpgGa_allowed_if_eligible: x_UpgGarage <= 1
 GaFin_change_only_if_upg: x_garage_finish_is_Fin + x_garage_finish_is_Unf
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
   - 70 x_AddBed - 48 x_z10_GarageArea - 96 x_z20_GarageArea
   - 144 x_z30_GarageArea - 16.6 x_z10_OpenPorchSF - 33.2 x_z20_OpenPorchSF
   - 49.8 x_z30_OpenPorchSF >= -1378
 AMPL_link_GarageArea: x_Garage_Area - 48 x_z10_GarageArea
   - 96 x_z20_GarageArea - 144 x_z30_GarageArea = 480
 AMPL_link_WoodDeckSF: x_Wood_Deck_SF = 0
 AMPL_link_OpenPorchSF: x_Open_Porch_SF - 16.6 x_z10_OpenPorchSF
   - 33.2 x_z20_OpenPorchSF - 49.8 x_z30_OpenPorchSF = 166
 AMPL_link_EnclosedPorch: x_Enclosed_Porch = 0
 AMPL_link_3SsnPorch: x_3Ssn_Porch = 0
 AMPL_link_ScreenPorch: x_Screen_Porch = 0
 AMPL_link_PoolArea: x_Pool_Area = 0
 AMPL_link_1stflr: x_1st_Flr_SF - 40 x_AddFull - 20 x_AddHalf
   - 75 x_AddKitch - 70 x_AddBed = 520
 AMPL_1stflr_upgrade: x_1st_Flr_SF - 40 x_Add1stFlr >= 520
 COUNT_fullbath: x_Full_Bath - x_AddFull = 2
 COUNT_halfbath: - x_AddHalf + x_Half_Bath = 1
 COUNT_bedroom: x_Bedroom_AbvGr - x_AddBed = 2
 COUNT_kitchen: - x_AddKitch + x_Kitchen_AbvGr = 1
 COUNT_totrooms: x_TotRms_AbvGrd - x_AddKitch - x_AddBed = 4
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
 HEAT_qc_upgrade_only_if_eligible: x_heat_upg_qc <= 0
 HEAT_qc_upgrade_only: x_Heating_QC >= 4
 R79: - x_heat_upg_type + HEAT_any_rebuild >= 0
 R80: - x_heat_upg_qc + HEAT_any_rebuild >= 0
 R81: - x_heat_upg_type - x_heat_upg_qc + HEAT_any_rebuild <= 0
 HEAT_qc_min_if_any_rebuild: x_Heating_QC - 2 HEAT_any_rebuild >= 0
 HEAT_change_def: - heat_is_Floor - heat_is_GasW - heat_is_Grav
   - heat_is_OthW - heat_is_Wall + heat_change_type = 0
 HEAT_qc_onehot: heat_qc_is_Po + heat_qc_is_Fa + heat_qc_is_TA
   + heat_qc_is_Gd + heat_qc_is_Ex = 1
 HEAT_qc_link: x_Heating_QC - heat_qc_is_Fa - 2 heat_qc_is_TA
   - 3 heat_qc_is_Gd - 4 heat_qc_is_Ex = 0
 FQ_pick_one: fireplace_is_Po + fireplace_is_Fa + fireplace_is_TA
   + fireplace_is_Gd + fireplace_is_Ex + fireplace_is_No_aplica = 1
 FQ_level_match: x_Fireplace_Qu - fireplace_is_Fa - 2 fireplace_is_TA
   - 3 fireplace_is_Gd - 4 fireplace_is_Ex + fireplace_is_No_aplica = 0
 BSMT_all_or_nothing: - 600 bsmt_finish_all + bsmt_to_fin1 + bsmt_to_fin2
   = 0
 BSMT_link_fin1: x_BsmtFin_SF_1 - bsmt_to_fin1 = 0
 BSMT_link_fin2: x_BsmtFin_SF_2 - bsmt_to_fin2 = 0
 BSMT_link_unf: x_Bsmt_Unf_SF + 600 bsmt_finish_all = 600
 BSMT_conservation_sum: x_BsmtFin_SF_1 + x_BsmtFin_SF_2 + x_Bsmt_Unf_SF
   = 600
 BSMTCOND_onehot: bsmtcond_is_Po + bsmtcond_is_Fa + bsmtcond_is_TA
   + bsmtcond_is_Gd + bsmtcond_is_Ex = 1
 BSMTCOND_link_int: x_Bsmt_Cond - bsmtcond_is_Fa - 2 bsmtcond_is_TA
   - 3 bsmtcond_is_Gd - 4 bsmtcond_is_Ex = 0
 UTIL_onehot: util_ELO + util_NoSeWa + util_NoSewr + util_AllPub = 1
 UTIL_link_int: x_Utilities - util_NoSeWa - 2 util_NoSewr - 3 util_AllPub
   = 0
 R2_grliv_le_lot: x_Gr_Liv_Area <= 2544
 R4_baths_le_bedrooms: - x_Bedroom_AbvGr + x_Full_Bath + x_Half_Bath <= 0
 R5_min_fullbath: x_Full_Bath >= 1
 R5_min_bedrooms: x_Bedroom_AbvGr >= 1
 R5_min_kitchen: x_Kitchen_AbvGr >= 1
 R7_gr_liv_equality: x_Gr_Liv_Area - x_1st_Flr_SF = 703
 R8_rooms_balance: - x_Bedroom_AbvGr + x_TotRms_AbvGrd - x_Kitchen_AbvGr
   - R8_other_rooms = 0
 TREE_0_ONEHOT: t0_leaf0 + t0_leaf1 + t0_leaf2 + t0_leaf3 + t0_leaf4
   + t0_leaf5 + t0_leaf6 + t0_leaf7 = 1
 T0_L0_f5_le: const_Year_Built + 998014 t0_leaf0 <= 1e+06
 T0_L0_f29_le: const_Fireplaces + 999999 t0_leaf0 <= 1e+06
 T0_L0_f6_le: const_Year_Remod_Add + 998046 t0_leaf0 <= 1e+06
 T0_L1_f5_le: const_Year_Built + 998014 t0_leaf1 <= 1e+06
 T0_L1_f29_le: const_Fireplaces + 999999 t0_leaf1 <= 1e+06
 T0_R1_f6_ge: const_Year_Remod_Add - 1.001954e+06 t0_leaf1 >= -1e+06
 T0_L2_f5_le: const_Year_Built + 998014 t0_leaf2 <= 1e+06
 T0_R2_f29_ge: const_Fireplaces - 1.000001e+06 t0_leaf2 >= -1e+06
 T0_L2_f20_le: x_Gr_Liv_Area + 8260 t0_leaf2 <= 10000
 T0_L3_f5_le: const_Year_Built + 998014 t0_leaf3 <= 1e+06
 T0_R3_f29_ge: const_Fireplaces - 1.000001e+06 t0_leaf3 >= -1e+06
 T0_R3_f20_ge: x_Gr_Liv_Area - 1740 t0_leaf3 >= 0
 T0_R4_f5_ge: const_Year_Built - 1.001986e+06 t0_leaf4 >= -1e+06
 T0_L4_f33_le: x_Garage_Area + 99350 t0_leaf4 <= 100000
 T0_L4_f20_le: x_Gr_Liv_Area + 8516 t0_leaf4 <= 10000
 T0_R5_f5_ge: const_Year_Built - 1.001986e+06 t0_leaf5 >= -1e+06
 T0_L5_f33_le: x_Garage_Area + 99350 t0_leaf5 <= 100000
 T0_R5_f20_ge: x_Gr_Liv_Area - 1484 t0_leaf5 >= 0
 T0_R6_f5_ge: const_Year_Built - 1.001986e+06 t0_leaf6 >= -1e+06
 T0_R6_f33_ge: x_Garage_Area - 650 t0_leaf6 >= 0
 T0_L6_f20_le: x_Gr_Liv_Area + 8339 t0_leaf6 <= 10000
 T0_R7_f5_ge: const_Year_Built - 1.001986e+06 t0_leaf7 >= -1e+06
 T0_R7_f33_ge: x_Garage_Area - 650 t0_leaf7 >= 0
 T0_R7_f20_ge: x_Gr_Liv_Area - 1661 t0_leaf7 >= 0
 TREE_1_ONEHOT: t1_leaf0 + t1_leaf1 + t1_leaf2 + t1_leaf3 + t1_leaf4
   + t1_leaf5 + t1_leaf6 + t1_leaf7 = 1
 T1_L0_f3_le: Overall_Qual_calc + 999993 t1_leaf0 <= 1e+06
 T1_L0_f3_le: Overall_Qual_calc + 999995 t1_leaf0 <= 1e+06
 T1_L0_f15_le: const_Total_Bsmt_SF + 999136 t1_leaf0 <= 1e+06
 T1_L1_f3_le: Overall_Qual_calc + 999993 t1_leaf1 <= 1e+06
 T1_L1_f3_le: Overall_Qual_calc + 999995 t1_leaf1 <= 1e+06
 T1_R1_f15_ge: const_Total_Bsmt_SF - 1.000864e+06 t1_leaf1 >= -1e+06
 T1_L2_f3_le: Overall_Qual_calc + 999993 t1_leaf2 <= 1e+06
 T1_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t1_leaf2 >= -1e+06
 T1_L2_f32_le: const_Garage_Cars + 999998 t1_leaf2 <= 1e+06
 T1_L3_f3_le: Overall_Qual_calc + 999993 t1_leaf3 <= 1e+06
 T1_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t1_leaf3 >= -1e+06
 T1_R3_f32_ge: const_Garage_Cars - 1.000002e+06 t1_leaf3 >= -1e+06
 T1_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t1_leaf4 >= -1e+06
 T1_L4_f3_le: Overall_Qual_calc + 999992 t1_leaf4 <= 1e+06
 T1_L4_f15_le: const_Total_Bsmt_SF + 999155 t1_leaf4 <= 1e+06
 T1_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t1_leaf5 >= -1e+06
 T1_L5_f3_le: Overall_Qual_calc + 999992 t1_leaf5 <= 1e+06
 T1_R5_f15_ge: const_Total_Bsmt_SF - 1.000845e+06 t1_leaf5 >= -1e+06
 T1_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t1_leaf6 >= -1e+06
 T1_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t1_leaf6 >= -1e+06
 T1_L6_f20_le: x_Gr_Liv_Area + 8134 t1_leaf6 <= 10000
 T1_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t1_leaf7 >= -1e+06
 T1_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t1_leaf7 >= -1e+06
 T1_R7_f20_ge: x_Gr_Liv_Area - 1866 t1_leaf7 >= 0
 TREE_2_ONEHOT: t2_leaf0 + t2_leaf1 + t2_leaf2 + t2_leaf3 + t2_leaf4
   + t2_leaf5 + t2_leaf6 + t2_leaf7 = 1
 T2_L0_f8_le: x_Exter_Qual + t2_leaf0 <= 4
 T2_L0_f20_le: x_Gr_Liv_Area + 8692 t2_leaf0 <= 10000
 T2_L0_f5_le: const_Year_Built + 998047 t2_leaf0 <= 1e+06
 T2_L1_f8_le: x_Exter_Qual + t2_leaf1 <= 4
 T2_L1_f20_le: x_Gr_Liv_Area + 8692 t2_leaf1 <= 10000
 T2_R1_f5_ge: const_Year_Built - 1.001953e+06 t2_leaf1 >= -1e+06
 T2_L2_f8_le: x_Exter_Qual + t2_leaf2 <= 4
 T2_R2_f20_ge: x_Gr_Liv_Area - 1308 t2_leaf2 >= 0
 T2_L2_f12_le: x_BsmtFin_SF_1 + 999610 t2_leaf2 <= 1e+06
 T2_L3_f8_le: x_Exter_Qual + t2_leaf3 <= 4
 T2_R3_f20_ge: x_Gr_Liv_Area - 1308 t2_leaf3 >= 0
 T2_R3_f12_ge: x_BsmtFin_SF_1 - 390 t2_leaf3 >= 0
 T2_R4_f8_ge: x_Exter_Qual - 4 t2_leaf4 >= -1
 T2_L4_f20_le: x_Gr_Liv_Area + 8373 t2_leaf4 <= 10000
 T2_L4_f17_le: x_1st_Flr_SF + 3731 t2_leaf4 <= 5000
 T2_R5_f8_ge: x_Exter_Qual - 4 t2_leaf5 >= -1
 T2_L5_f20_le: x_Gr_Liv_Area + 8373 t2_leaf5 <= 10000
 T2_R5_f17_ge: x_1st_Flr_SF - 1269 t2_leaf5 >= 0
 T2_R6_f8_ge: x_Exter_Qual - 4 t2_leaf6 >= -1
 T2_R6_f20_ge: x_Gr_Liv_Area - 1627 t2_leaf6 >= 0
 T2_L6_f17_le: x_1st_Flr_SF + 3915 t2_leaf6 <= 5000
 T2_R7_f8_ge: x_Exter_Qual - 4 t2_leaf7 >= -1
 T2_R7_f20_ge: x_Gr_Liv_Area - 1627 t2_leaf7 >= 0
 T2_R7_f17_ge: x_1st_Flr_SF - 1085 t2_leaf7 >= 0
 TREE_3_ONEHOT: t3_leaf0 + t3_leaf1 + t3_leaf2 + t3_leaf3 + t3_leaf4
   + t3_leaf5 + t3_leaf6 + t3_leaf7 = 1
 T3_L0_f3_le: Overall_Qual_calc + 999993 t3_leaf0 <= 1e+06
 T3_L0_f3_le: Overall_Qual_calc + 999995 t3_leaf0 <= 1e+06
 T3_L0_f15_le: const_Total_Bsmt_SF + 999200 t3_leaf0 <= 1e+06
 T3_L1_f3_le: Overall_Qual_calc + 999993 t3_leaf1 <= 1e+06
 T3_L1_f3_le: Overall_Qual_calc + 999995 t3_leaf1 <= 1e+06
 T3_R1_f15_ge: const_Total_Bsmt_SF - 1.0008e+06 t3_leaf1 >= -1e+06
 T3_L2_f3_le: Overall_Qual_calc + 999993 t3_leaf2 <= 1e+06
 T3_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t3_leaf2 >= -1e+06
 T3_L2_f20_le: x_Gr_Liv_Area + 8638 t3_leaf2 <= 10000
 T3_L3_f3_le: Overall_Qual_calc + 999993 t3_leaf3 <= 1e+06
 T3_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t3_leaf3 >= -1e+06
 T3_R3_f20_ge: x_Gr_Liv_Area - 1362 t3_leaf3 >= 0
 T3_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t3_leaf4 >= -1e+06
 T3_L4_f3_le: Overall_Qual_calc + 999992 t3_leaf4 <= 1e+06
 T3_L4_f20_le: x_Gr_Liv_Area + 8522 t3_leaf4 <= 10000
 T3_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t3_leaf5 >= -1e+06
 T3_L5_f3_le: Overall_Qual_calc + 999992 t3_leaf5 <= 1e+06
 T3_R5_f20_ge: x_Gr_Liv_Area - 1478 t3_leaf5 >= 0
 T3_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t3_leaf6 >= -1e+06
 T3_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t3_leaf6 >= -1e+06
 T3_L6_f3_le: Overall_Qual_calc + 999991 t3_leaf6 <= 1e+06
 T3_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t3_leaf7 >= -1e+06
 T3_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t3_leaf7 >= -1e+06
 T3_R7_f3_ge: Overall_Qual_calc - 1.000009e+06 t3_leaf7 >= -1e+06
 TREE_4_ONEHOT: t4_leaf0 + t4_leaf1 + t4_leaf2 + t4_leaf3 + t4_leaf4
   + t4_leaf5 + t4_leaf6 + t4_leaf7 = 1
 T4_L0_f8_le: x_Exter_Qual + t4_leaf0 <= 4
 T4_L0_f20_le: x_Gr_Liv_Area + 8904 t4_leaf0 <= 10000
 T4_L0_f231_le: central_air_no + 999999 t4_leaf0 <= 1e+06
 T4_L1_f8_le: x_Exter_Qual + t4_leaf1 <= 4
 T4_L1_f20_le: x_Gr_Liv_Area + 8904 t4_leaf1 <= 10000
 T4_R1_f231_ge: central_air_no - 1.000001e+06 t4_leaf1 >= -1e+06
 T4_L2_f8_le: x_Exter_Qual + t4_leaf2 <= 4
 T4_R2_f20_ge: x_Gr_Liv_Area - 1096 t4_leaf2 >= 0
 T4_L2_f29_le: const_Fireplaces + 999999 t4_leaf2 <= 1e+06
 T4_L3_f8_le: x_Exter_Qual + t4_leaf3 <= 4
 T4_R3_f20_ge: x_Gr_Liv_Area - 1096 t4_leaf3 >= 0
 T4_R3_f29_ge: const_Fireplaces - 1.000001e+06 t4_leaf3 >= -1e+06
 T4_R4_f8_ge: x_Exter_Qual - 4 t4_leaf4 >= -1
 T4_L4_f32_le: const_Garage_Cars + 999997 t4_leaf4 <= 1e+06
 T4_L4_f17_le: x_1st_Flr_SF + 3586 t4_leaf4 <= 5000
 T4_R5_f8_ge: x_Exter_Qual - 4 t4_leaf5 >= -1
 T4_L5_f32_le: const_Garage_Cars + 999997 t4_leaf5 <= 1e+06
 T4_R5_f17_ge: x_1st_Flr_SF - 1414 t4_leaf5 >= 0
 T4_R6_f8_ge: x_Exter_Qual - 4 t4_leaf6 >= -1
 T4_R6_f32_ge: const_Garage_Cars - 1.000003e+06 t4_leaf6 >= -1e+06
 T4_L6_f20_le: x_Gr_Liv_Area + 8284 t4_leaf6 <= 10000
 T4_R7_f8_ge: x_Exter_Qual - 4 t4_leaf7 >= -1
 T4_R7_f32_ge: const_Garage_Cars - 1.000003e+06 t4_leaf7 >= -1e+06
 T4_R7_f20_ge: x_Gr_Liv_Area - 1716 t4_leaf7 >= 0
 TREE_5_ONEHOT: t5_leaf0 + t5_leaf1 + t5_leaf2 + t5_leaf3 + t5_leaf4
   + t5_leaf5 + t5_leaf6 + t5_leaf7 = 1
 T5_L0_f3_le: Overall_Qual_calc + 999993 t5_leaf0 <= 1e+06
 T5_L0_f3_le: Overall_Qual_calc + 999995 t5_leaf0 <= 1e+06
 T5_L0_f17_le: x_1st_Flr_SF + 3845 t5_leaf0 <= 5000
 T5_L1_f3_le: Overall_Qual_calc + 999993 t5_leaf1 <= 1e+06
 T5_L1_f3_le: Overall_Qual_calc + 999995 t5_leaf1 <= 1e+06
 T5_R1_f17_ge: x_1st_Flr_SF - 1155 t5_leaf1 >= 0
 T5_L2_f3_le: Overall_Qual_calc + 999993 t5_leaf2 <= 1e+06
 T5_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t5_leaf2 >= -1e+06
 T5_L2_f32_le: const_Garage_Cars + 999998 t5_leaf2 <= 1e+06
 T5_L3_f3_le: Overall_Qual_calc + 999993 t5_leaf3 <= 1e+06
 T5_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t5_leaf3 >= -1e+06
 T5_R3_f32_ge: const_Garage_Cars - 1.000002e+06 t5_leaf3 >= -1e+06
 T5_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t5_leaf4 >= -1e+06
 T5_L4_f3_le: Overall_Qual_calc + 999992 t5_leaf4 <= 1e+06
 T5_L4_f17_le: x_1st_Flr_SF + 4068 t5_leaf4 <= 5000
 T5_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t5_leaf5 >= -1e+06
 T5_L5_f3_le: Overall_Qual_calc + 999992 t5_leaf5 <= 1e+06
 T5_R5_f17_ge: x_1st_Flr_SF - 932 t5_leaf5 >= 0
 T5_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t5_leaf6 >= -1e+06
 T5_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t5_leaf6 >= -1e+06
 T5_L6_f17_le: x_1st_Flr_SF + 3234 t5_leaf6 <= 5000
 T5_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t5_leaf7 >= -1e+06
 T5_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t5_leaf7 >= -1e+06
 T5_R7_f17_ge: x_1st_Flr_SF - 1766 t5_leaf7 >= 0
 TREE_6_ONEHOT: t6_leaf0 + t6_leaf1 + t6_leaf2 + t6_leaf3 + t6_leaf4
   + t6_leaf5 + t6_leaf6 + t6_leaf7 = 1
 T6_L0_f10_le: const_Bsmt_Qual + 999997 t6_leaf0 <= 1e+06
 T6_L0_f35_le: x_Garage_Cond + 2 t6_leaf0 <= 4
 T6_L0_f28_le: x_TotRms_AbvGrd + 8 t6_leaf0 <= 15
 T6_L1_f10_le: const_Bsmt_Qual + 999997 t6_leaf1 <= 1e+06
 T6_L1_f35_le: x_Garage_Cond + 2 t6_leaf1 <= 4
 T6_R1_f28_ge: x_TotRms_AbvGrd - 7 t6_leaf1 >= 0
 T6_L2_f10_le: const_Bsmt_Qual + 999997 t6_leaf2 <= 1e+06
 T6_R2_f35_ge: x_Garage_Cond - 3 t6_leaf2 >= -1
 T6_L2_f29_le: const_Fireplaces + 999999 t6_leaf2 <= 1e+06
 T6_L3_f10_le: const_Bsmt_Qual + 999997 t6_leaf3 <= 1e+06
 T6_R3_f35_ge: x_Garage_Cond - 3 t6_leaf3 >= -1
 T6_R3_f29_ge: const_Fireplaces - 1.000001e+06 t6_leaf3 >= -1e+06
 T6_R4_f10_ge: const_Bsmt_Qual - 1.000003e+06 t6_leaf4 >= -1e+06
 T6_L4_f32_le: const_Garage_Cars + 999997 t6_leaf4 <= 1e+06
 T6_L4_f29_le: const_Fireplaces + 999999 t6_leaf4 <= 1e+06
 T6_R5_f10_ge: const_Bsmt_Qual - 1.000003e+06 t6_leaf5 >= -1e+06
 T6_L5_f32_le: const_Garage_Cars + 999997 t6_leaf5 <= 1e+06
 T6_R5_f29_ge: const_Fireplaces - 1.000001e+06 t6_leaf5 >= -1e+06
 T6_R6_f10_ge: const_Bsmt_Qual - 1.000003e+06 t6_leaf6 >= -1e+06
 T6_R6_f32_ge: const_Garage_Cars - 1.000003e+06 t6_leaf6 >= -1e+06
 T6_L6_f27_le: x_Kitchen_Qual <= 4
 T6_R7_f10_ge: const_Bsmt_Qual - 1.000003e+06 t6_leaf7 >= -1e+06
 T6_R7_f32_ge: const_Garage_Cars - 1.000003e+06 t6_leaf7 >= -1e+06
 T6_R7_f27_ge: x_Kitchen_Qual - 5 t6_leaf7 >= -1
 TREE_7_ONEHOT: t7_leaf0 + t7_leaf1 + t7_leaf2 + t7_leaf3 + t7_leaf4
   + t7_leaf5 + t7_leaf6 + t7_leaf7 = 1
 T7_L0_f3_le: Overall_Qual_calc + 999993 t7_leaf0 <= 1e+06
 T7_L0_f3_le: Overall_Qual_calc + 999995 t7_leaf0 <= 1e+06
 T7_L0_f20_le: x_Gr_Liv_Area + 9136 t7_leaf0 <= 10000
 T7_L1_f3_le: Overall_Qual_calc + 999993 t7_leaf1 <= 1e+06
 T7_L1_f3_le: Overall_Qual_calc + 999995 t7_leaf1 <= 1e+06
 T7_R1_f20_ge: x_Gr_Liv_Area - 864 t7_leaf1 >= 0
 T7_L2_f3_le: Overall_Qual_calc + 999993 t7_leaf2 <= 1e+06
 T7_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t7_leaf2 >= -1e+06
 T7_L2_f20_le: x_Gr_Liv_Area + 8516 t7_leaf2 <= 10000
 T7_L3_f3_le: Overall_Qual_calc + 999993 t7_leaf3 <= 1e+06
 T7_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t7_leaf3 >= -1e+06
 T7_R3_f20_ge: x_Gr_Liv_Area - 1484 t7_leaf3 >= 0
 T7_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t7_leaf4 >= -1e+06
 T7_L4_f17_le: x_1st_Flr_SF + 3492 t7_leaf4 <= 5000
 T7_L4_f20_le: x_Gr_Liv_Area + 8134 t7_leaf4 <= 10000
 T7_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t7_leaf5 >= -1e+06
 T7_L5_f17_le: x_1st_Flr_SF + 3492 t7_leaf5 <= 5000
 T7_R5_f20_ge: x_Gr_Liv_Area - 1866 t7_leaf5 >= 0
 T7_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t7_leaf6 >= -1e+06
 T7_R6_f17_ge: x_1st_Flr_SF - 1508 t7_leaf6 >= 0
 T7_L6_f20_le: x_Gr_Liv_Area + 8280 t7_leaf6 <= 10000
 T7_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t7_leaf7 >= -1e+06
 T7_R7_f17_ge: x_1st_Flr_SF - 1508 t7_leaf7 >= 0
 T7_R7_f20_ge: x_Gr_Liv_Area - 1720 t7_leaf7 >= 0
 TREE_8_ONEHOT: t8_leaf0 + t8_leaf1 + t8_leaf2 + t8_leaf3 + t8_leaf4
   + t8_leaf5 + t8_leaf6 + t8_leaf7 = 1
 T8_L0_f10_le: const_Bsmt_Qual + 999997 t8_leaf0 <= 1e+06
 T8_L0_f29_le: const_Fireplaces + 999999 t8_leaf0 <= 1e+06
 T8_L0_f231_le: central_air_no + 999999 t8_leaf0 <= 1e+06
 T8_L1_f10_le: const_Bsmt_Qual + 999997 t8_leaf1 <= 1e+06
 T8_L1_f29_le: const_Fireplaces + 999999 t8_leaf1 <= 1e+06
 T8_R1_f231_ge: central_air_no - 1.000001e+06 t8_leaf1 >= -1e+06
 T8_L2_f10_le: const_Bsmt_Qual + 999997 t8_leaf2 <= 1e+06
 T8_R2_f29_ge: const_Fireplaces - 1.000001e+06 t8_leaf2 >= -1e+06
 T8_L2_f28_le: x_TotRms_AbvGrd + 7 t8_leaf2 <= 15
 T8_L3_f10_le: const_Bsmt_Qual + 999997 t8_leaf3 <= 1e+06
 T8_R3_f29_ge: const_Fireplaces - 1.000001e+06 t8_leaf3 >= -1e+06
 T8_R3_f28_ge: x_TotRms_AbvGrd - 8 t8_leaf3 >= 0
 T8_R4_f10_ge: const_Bsmt_Qual - 1.000003e+06 t8_leaf4 >= -1e+06
 T8_L4_f32_le: const_Garage_Cars + 999997 t8_leaf4 <= 1e+06
 T8_L4_f17_le: x_1st_Flr_SF + 3764 t8_leaf4 <= 5000
 T8_R5_f10_ge: const_Bsmt_Qual - 1.000003e+06 t8_leaf5 >= -1e+06
 T8_L5_f32_le: const_Garage_Cars + 999997 t8_leaf5 <= 1e+06
 T8_R5_f17_ge: x_1st_Flr_SF - 1236 t8_leaf5 >= 0
 T8_R6_f10_ge: const_Bsmt_Qual - 1.000003e+06 t8_leaf6 >= -1e+06
 T8_R6_f32_ge: const_Garage_Cars - 1.000003e+06 t8_leaf6 >= -1e+06
 T8_L6_f17_le: x_1st_Flr_SF + 3200 t8_leaf6 <= 5000
 T8_R7_f10_ge: const_Bsmt_Qual - 1.000003e+06 t8_leaf7 >= -1e+06
 T8_R7_f32_ge: const_Garage_Cars - 1.000003e+06 t8_leaf7 >= -1e+06
 T8_R7_f17_ge: x_1st_Flr_SF - 1800 t8_leaf7 >= 0
 TREE_9_ONEHOT: t9_leaf0 + t9_leaf1 + t9_leaf2 + t9_leaf3 + t9_leaf4
   + t9_leaf5 + t9_leaf6 + t9_leaf7 = 1
 T9_L0_f5_le: const_Year_Built + 998019 t9_leaf0 <= 1e+06
 T9_L0_f17_le: x_1st_Flr_SF + 3906 t9_leaf0 <= 5000
 T9_L0_f231_le: central_air_no + 999999 t9_leaf0 <= 1e+06
 T9_L1_f5_le: const_Year_Built + 998019 t9_leaf1 <= 1e+06
 T9_L1_f17_le: x_1st_Flr_SF + 3906 t9_leaf1 <= 5000
 T9_R1_f231_ge: central_air_no - 1.000001e+06 t9_leaf1 >= -1e+06
 T9_L2_f5_le: const_Year_Built + 998019 t9_leaf2 <= 1e+06
 T9_R2_f17_ge: x_1st_Flr_SF - 1094 t9_leaf2 >= 0
 T9_L2_f1_le: const_Lot_Area + 987784 t9_leaf2 <= 1e+06
 T9_L3_f5_le: const_Year_Built + 998019 t9_leaf3 <= 1e+06
 T9_R3_f17_ge: x_1st_Flr_SF - 1094 t9_leaf3 >= 0
 T9_R3_f1_ge: const_Lot_Area - 1.012216e+06 t9_leaf3 >= -1e+06
 T9_R4_f5_ge: const_Year_Built - 1.001981e+06 t9_leaf4 >= -1e+06
 T9_L4_f33_le: x_Garage_Area + 99332 t9_leaf4 <= 100000
 T9_L4_f15_le: const_Total_Bsmt_SF + 998732 t9_leaf4 <= 1e+06
 T9_R5_f5_ge: const_Year_Built - 1.001981e+06 t9_leaf5 >= -1e+06
 T9_L5_f33_le: x_Garage_Area + 99332 t9_leaf5 <= 100000
 T9_R5_f15_ge: const_Total_Bsmt_SF - 1.001268e+06 t9_leaf5 >= -1e+06
 T9_R6_f5_ge: const_Year_Built - 1.001981e+06 t9_leaf6 >= -1e+06
 T9_R6_f33_ge: x_Garage_Area - 668 t9_leaf6 >= 0
 T9_L6_f1_le: const_Lot_Area + 988020 t9_leaf6 <= 1e+06
 T9_R7_f5_ge: const_Year_Built - 1.001981e+06 t9_leaf7 >= -1e+06
 T9_R7_f33_ge: x_Garage_Area - 668 t9_leaf7 >= 0
 T9_R7_f1_ge: const_Lot_Area - 1.01198e+06 t9_leaf7 >= -1e+06
 TREE_10_ONEHOT: t10_leaf0 + t10_leaf1 + t10_leaf2 + t10_leaf3 + t10_leaf4
   + t10_leaf5 + t10_leaf6 + t10_leaf7 = 1
 T10_L0_f20_le: x_Gr_Liv_Area + 8680 t10_leaf0 <= 10000
 T10_L0_f31_le: const_Garage_Yr_Blt + 998047 t10_leaf0 <= 1e+06
 T10_L0_f20_le: x_Gr_Liv_Area + 9167 t10_leaf0 <= 10000
 T10_L1_f20_le: x_Gr_Liv_Area + 8680 t10_leaf1 <= 10000
 T10_L1_f31_le: const_Garage_Yr_Blt + 998047 t10_leaf1 <= 1e+06
 T10_R1_f20_ge: x_Gr_Liv_Area - 833 t10_leaf1 >= 0
 T10_L2_f20_le: x_Gr_Liv_Area + 8680 t10_leaf2 <= 10000
 T10_R2_f31_ge: const_Garage_Yr_Blt - 1.001953e+06 t10_leaf2 >= -1e+06
 T10_L2_f20_le: x_Gr_Liv_Area + 8876 t10_leaf2 <= 10000
 T10_L3_f20_le: x_Gr_Liv_Area + 8680 t10_leaf3 <= 10000
 T10_R3_f31_ge: const_Garage_Yr_Blt - 1.001953e+06 t10_leaf3 >= -1e+06
 T10_R3_f20_ge: x_Gr_Liv_Area - 1124 t10_leaf3 >= 0
 T10_R4_f20_ge: x_Gr_Liv_Area - 1320 t10_leaf4 >= 0
 T10_L4_f8_le: x_Exter_Qual + t10_leaf4 <= 4
 T10_L4_f66_le: const_MS_Zoning_RL + 999999 t10_leaf4 <= 1e+06
 T10_R5_f20_ge: x_Gr_Liv_Area - 1320 t10_leaf5 >= 0
 T10_L5_f8_le: x_Exter_Qual + t10_leaf5 <= 4
 T10_R5_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t10_leaf5 >= -1e+06
 T10_R6_f20_ge: x_Gr_Liv_Area - 1320 t10_leaf6 >= 0
 T10_R6_f8_ge: x_Exter_Qual - 4 t10_leaf6 >= -1
 T10_L6_f10_le: const_Bsmt_Qual + 999996 t10_leaf6 <= 1e+06
 T10_R7_f20_ge: x_Gr_Liv_Area - 1320 t10_leaf7 >= 0
 T10_R7_f8_ge: x_Exter_Qual - 4 t10_leaf7 >= -1
 T10_R7_f10_ge: const_Bsmt_Qual - 1.000004e+06 t10_leaf7 >= -1e+06
 TREE_11_ONEHOT: t11_leaf0 + t11_leaf1 + t11_leaf2 + t11_leaf3 + t11_leaf4
   + t11_leaf5 + t11_leaf6 + t11_leaf7 = 1
 T11_L0_f3_le: Overall_Qual_calc + 999993 t11_leaf0 <= 1e+06
 T11_L0_f3_le: Overall_Qual_calc + 999995 t11_leaf0 <= 1e+06
 T11_L0_f15_le: const_Total_Bsmt_SF + 999088 t11_leaf0 <= 1e+06
 T11_L1_f3_le: Overall_Qual_calc + 999993 t11_leaf1 <= 1e+06
 T11_L1_f3_le: Overall_Qual_calc + 999995 t11_leaf1 <= 1e+06
 T11_R1_f15_ge: const_Total_Bsmt_SF - 1.000912e+06 t11_leaf1 >= -1e+06
 T11_L2_f3_le: Overall_Qual_calc + 999993 t11_leaf2 <= 1e+06
 T11_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t11_leaf2 >= -1e+06
 T11_L2_f20_le: x_Gr_Liv_Area + 8589 t11_leaf2 <= 10000
 T11_L3_f3_le: Overall_Qual_calc + 999993 t11_leaf3 <= 1e+06
 T11_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t11_leaf3 >= -1e+06
 T11_R3_f20_ge: x_Gr_Liv_Area - 1411 t11_leaf3 >= 0
 T11_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t11_leaf4 >= -1e+06
 T11_L4_f3_le: Overall_Qual_calc + 999992 t11_leaf4 <= 1e+06
 T11_L4_f20_le: x_Gr_Liv_Area + 8274 t11_leaf4 <= 10000
 T11_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t11_leaf5 >= -1e+06
 T11_L5_f3_le: Overall_Qual_calc + 999992 t11_leaf5 <= 1e+06
 T11_R5_f20_ge: x_Gr_Liv_Area - 1726 t11_leaf5 >= 0
 T11_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t11_leaf6 >= -1e+06
 T11_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t11_leaf6 >= -1e+06
 T11_L6_f20_le: x_Gr_Liv_Area + 8330 t11_leaf6 <= 10000
 T11_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t11_leaf7 >= -1e+06
 T11_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t11_leaf7 >= -1e+06
 T11_R7_f20_ge: x_Gr_Liv_Area - 1670 t11_leaf7 >= 0
 TREE_12_ONEHOT: t12_leaf0 + t12_leaf1 + t12_leaf2 + t12_leaf3 + t12_leaf4
   + t12_leaf5 + t12_leaf6 + t12_leaf7 = 1
 T12_L0_f3_le: Overall_Qual_calc + 999993 t12_leaf0 <= 1e+06
 T12_L0_f3_le: Overall_Qual_calc + 999995 t12_leaf0 <= 1e+06
 T12_L0_f20_le: x_Gr_Liv_Area + 8845 t12_leaf0 <= 10000
 T12_L1_f3_le: Overall_Qual_calc + 999993 t12_leaf1 <= 1e+06
 T12_L1_f3_le: Overall_Qual_calc + 999995 t12_leaf1 <= 1e+06
 T12_R1_f20_ge: x_Gr_Liv_Area - 1155 t12_leaf1 >= 0
 T12_L2_f3_le: Overall_Qual_calc + 999993 t12_leaf2 <= 1e+06
 T12_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t12_leaf2 >= -1e+06
 T12_L2_f33_le: x_Garage_Area + 99616 t12_leaf2 <= 100000
 T12_L3_f3_le: Overall_Qual_calc + 999993 t12_leaf3 <= 1e+06
 T12_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t12_leaf3 >= -1e+06
 T12_R3_f33_ge: x_Garage_Area - 384 t12_leaf3 >= 0
 T12_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t12_leaf4 >= -1e+06
 T12_L4_f10_le: const_Bsmt_Qual + 999996 t12_leaf4 <= 1e+06
 T12_L4_f20_le: x_Gr_Liv_Area + 7866 t12_leaf4 <= 10000
 T12_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t12_leaf5 >= -1e+06
 T12_L5_f10_le: const_Bsmt_Qual + 999996 t12_leaf5 <= 1e+06
 T12_R5_f20_ge: x_Gr_Liv_Area - 2134 t12_leaf5 >= 0
 T12_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t12_leaf6 >= -1e+06
 T12_R6_f10_ge: const_Bsmt_Qual - 1.000004e+06 t12_leaf6 >= -1e+06
 T12_L6_f3_le: Overall_Qual_calc + 999991 t12_leaf6 <= 1e+06
 T12_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t12_leaf7 >= -1e+06
 T12_R7_f10_ge: const_Bsmt_Qual - 1.000004e+06 t12_leaf7 >= -1e+06
 T12_R7_f3_ge: Overall_Qual_calc - 1.000009e+06 t12_leaf7 >= -1e+06
 TREE_13_ONEHOT: t13_leaf0 + t13_leaf1 + t13_leaf2 + t13_leaf3 + t13_leaf4
   + t13_leaf5 + t13_leaf6 + t13_leaf7 = 1
 T13_L0_f27_le: x_Kitchen_Qual + t13_leaf0 <= 4
 T13_L0_f32_le: const_Garage_Cars + 999998 t13_leaf0 <= 1e+06
 T13_L0_f4_le: const_Overall_Cond + 999996 t13_leaf0 <= 1e+06
 T13_L1_f27_le: x_Kitchen_Qual + t13_leaf1 <= 4
 T13_L1_f32_le: const_Garage_Cars + 999998 t13_leaf1 <= 1e+06
 T13_R1_f4_ge: const_Overall_Cond - 1.000004e+06 t13_leaf1 >= -1e+06
 T13_L2_f27_le: x_Kitchen_Qual + t13_leaf2 <= 4
 T13_R2_f32_ge: const_Garage_Cars - 1.000002e+06 t13_leaf2 >= -1e+06
 T13_L2_f20_le: x_Gr_Liv_Area + 8221 t13_leaf2 <= 10000
 T13_L3_f27_le: x_Kitchen_Qual + t13_leaf3 <= 4
 T13_R3_f32_ge: const_Garage_Cars - 1.000002e+06 t13_leaf3 >= -1e+06
 T13_R3_f20_ge: x_Gr_Liv_Area - 1779 t13_leaf3 >= 0
 T13_R4_f27_ge: x_Kitchen_Qual - 4 t13_leaf4 >= -1
 T13_L4_f20_le: x_Gr_Liv_Area + 8274 t13_leaf4 <= 10000
 T13_L4_f17_le: x_1st_Flr_SF + 3718 t13_leaf4 <= 5000
 T13_R5_f27_ge: x_Kitchen_Qual - 4 t13_leaf5 >= -1
 T13_L5_f20_le: x_Gr_Liv_Area + 8274 t13_leaf5 <= 10000
 T13_R5_f17_ge: x_1st_Flr_SF - 1282 t13_leaf5 >= 0
 T13_R6_f27_ge: x_Kitchen_Qual - 4 t13_leaf6 >= -1
 T13_R6_f20_ge: x_Gr_Liv_Area - 1726 t13_leaf6 >= 0
 T13_L6_f32_le: const_Garage_Cars + 999997 t13_leaf6 <= 1e+06
 T13_R7_f27_ge: x_Kitchen_Qual - 4 t13_leaf7 >= -1
 T13_R7_f20_ge: x_Gr_Liv_Area - 1726 t13_leaf7 >= 0
 T13_R7_f32_ge: const_Garage_Cars - 1.000003e+06 t13_leaf7 >= -1e+06
 TREE_14_ONEHOT: t14_leaf0 + t14_leaf1 + t14_leaf2 + t14_leaf3 + t14_leaf4
   + t14_leaf5 + t14_leaf6 + t14_leaf7 = 1
 T14_L0_f8_le: x_Exter_Qual + t14_leaf0 <= 4
 T14_L0_f232_le: central_air_yes + 999999 t14_leaf0 <= 1e+06
 T14_L0_f20_le: x_Gr_Liv_Area + 9124 t14_leaf0 <= 10000
 T14_L1_f8_le: x_Exter_Qual + t14_leaf1 <= 4
 T14_L1_f232_le: central_air_yes + 999999 t14_leaf1 <= 1e+06
 T14_R1_f20_ge: x_Gr_Liv_Area - 876 t14_leaf1 >= 0
 T14_L2_f8_le: x_Exter_Qual + t14_leaf2 <= 4
 T14_R2_f232_ge: central_air_yes - 1.000001e+06 t14_leaf2 >= -1e+06
 T14_L2_f20_le: x_Gr_Liv_Area + 8584 t14_leaf2 <= 10000
 T14_L3_f8_le: x_Exter_Qual + t14_leaf3 <= 4
 T14_R3_f232_ge: central_air_yes - 1.000001e+06 t14_leaf3 >= -1e+06
 T14_R3_f20_ge: x_Gr_Liv_Area - 1416 t14_leaf3 >= 0
 T14_R4_f8_ge: x_Exter_Qual - 4 t14_leaf4 >= -1
 T14_L4_f20_le: x_Gr_Liv_Area + 8260 t14_leaf4 <= 10000
 T14_L4_f15_le: const_Total_Bsmt_SF + 998732 t14_leaf4 <= 1e+06
 T14_R5_f8_ge: x_Exter_Qual - 4 t14_leaf5 >= -1
 T14_L5_f20_le: x_Gr_Liv_Area + 8260 t14_leaf5 <= 10000
 T14_R5_f15_ge: const_Total_Bsmt_SF - 1.001268e+06 t14_leaf5 >= -1e+06
 T14_R6_f8_ge: x_Exter_Qual - 4 t14_leaf6 >= -1
 T14_R6_f20_ge: x_Gr_Liv_Area - 1740 t14_leaf6 >= 0
 T14_L6_f15_le: const_Total_Bsmt_SF + 998794 t14_leaf6 <= 1e+06
 T14_R7_f8_ge: x_Exter_Qual - 4 t14_leaf7 >= -1
 T14_R7_f20_ge: x_Gr_Liv_Area - 1740 t14_leaf7 >= 0
 T14_R7_f15_ge: const_Total_Bsmt_SF - 1.001206e+06 t14_leaf7 >= -1e+06
 TREE_15_ONEHOT: t15_leaf0 + t15_leaf1 + t15_leaf2 + t15_leaf3 + t15_leaf4
   + t15_leaf5 + t15_leaf6 + t15_leaf7 = 1
 T15_L0_f20_le: x_Gr_Liv_Area + 8589 t15_leaf0 <= 10000
 T15_L0_f15_le: const_Total_Bsmt_SF + 999030 t15_leaf0 <= 1e+06
 T15_L0_f231_le: central_air_no + 999999 t15_leaf0 <= 1e+06
 T15_L1_f20_le: x_Gr_Liv_Area + 8589 t15_leaf1 <= 10000
 T15_L1_f15_le: const_Total_Bsmt_SF + 999030 t15_leaf1 <= 1e+06
 T15_R1_f231_ge: central_air_no - 1.000001e+06 t15_leaf1 >= -1e+06
 T15_L2_f20_le: x_Gr_Liv_Area + 8589 t15_leaf2 <= 10000
 T15_R2_f15_ge: const_Total_Bsmt_SF - 1.00097e+06 t15_leaf2 >= -1e+06
 T15_L2_f247_le: const_Garage_Type_Attchd + 999999 t15_leaf2 <= 1e+06
 T15_L3_f20_le: x_Gr_Liv_Area + 8589 t15_leaf3 <= 10000
 T15_R3_f15_ge: const_Total_Bsmt_SF - 1.00097e+06 t15_leaf3 >= -1e+06
 T15_R3_f247_ge: const_Garage_Type_Attchd - 1.000001e+06 t15_leaf3
   >= -1e+06
 T15_R4_f20_ge: x_Gr_Liv_Area - 1411 t15_leaf4 >= 0
 T15_L4_f10_le: const_Bsmt_Qual + 999996 t15_leaf4 <= 1e+06
 T15_L4_f256_le: const_Garage_Finish_Unf + 999999 t15_leaf4 <= 1e+06
 T15_R5_f20_ge: x_Gr_Liv_Area - 1411 t15_leaf5 >= 0
 T15_L5_f10_le: const_Bsmt_Qual + 999996 t15_leaf5 <= 1e+06
 T15_R5_f256_ge: const_Garage_Finish_Unf - 1.000001e+06 t15_leaf5
   >= -1e+06
 T15_R6_f20_ge: x_Gr_Liv_Area - 1411 t15_leaf6 >= 0
 T15_R6_f10_ge: const_Bsmt_Qual - 1.000004e+06 t15_leaf6 >= -1e+06
 T15_L6_f20_le: x_Gr_Liv_Area + 8005 t15_leaf6 <= 10000
 T15_R7_f20_ge: x_Gr_Liv_Area - 1411 t15_leaf7 >= 0
 T15_R7_f10_ge: const_Bsmt_Qual - 1.000004e+06 t15_leaf7 >= -1e+06
 T15_R7_f20_ge: x_Gr_Liv_Area - 1995 t15_leaf7 >= 0
 TREE_16_ONEHOT: t16_leaf0 + t16_leaf1 + t16_leaf2 + t16_leaf3 + t16_leaf4
   + t16_leaf5 + t16_leaf6 + t16_leaf7 = 1
 T16_L0_f3_le: Overall_Qual_calc + 999993 t16_leaf0 <= 1e+06
 T16_L0_f3_le: Overall_Qual_calc + 999995 t16_leaf0 <= 1e+06
 T16_L0_f3_le: Overall_Qual_calc + 999997 t16_leaf0 <= 1e+06
 T16_L1_f3_le: Overall_Qual_calc + 999993 t16_leaf1 <= 1e+06
 T16_L1_f3_le: Overall_Qual_calc + 999995 t16_leaf1 <= 1e+06
 T16_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t16_leaf1 >= -1e+06
 T16_L2_f3_le: Overall_Qual_calc + 999993 t16_leaf2 <= 1e+06
 T16_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t16_leaf2 >= -1e+06
 T16_L2_f1_le: const_Lot_Area + 989200 t16_leaf2 <= 1e+06
 T16_L3_f3_le: Overall_Qual_calc + 999993 t16_leaf3 <= 1e+06
 T16_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t16_leaf3 >= -1e+06
 T16_R3_f1_ge: const_Lot_Area - 1.0108e+06 t16_leaf3 >= -1e+06
 T16_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t16_leaf4 >= -1e+06
 T16_L4_f10_le: const_Bsmt_Qual + 999996 t16_leaf4 <= 1e+06
 T16_L4_f20_le: x_Gr_Liv_Area + 8205 t16_leaf4 <= 10000
 T16_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t16_leaf5 >= -1e+06
 T16_L5_f10_le: const_Bsmt_Qual + 999996 t16_leaf5 <= 1e+06
 T16_R5_f20_ge: x_Gr_Liv_Area - 1795 t16_leaf5 >= 0
 T16_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t16_leaf6 >= -1e+06
 T16_R6_f10_ge: const_Bsmt_Qual - 1.000004e+06 t16_leaf6 >= -1e+06
 T16_L6_f20_le: x_Gr_Liv_Area + 7802 t16_leaf6 <= 10000
 T16_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t16_leaf7 >= -1e+06
 T16_R7_f10_ge: const_Bsmt_Qual - 1.000004e+06 t16_leaf7 >= -1e+06
 T16_R7_f20_ge: x_Gr_Liv_Area - 2198 t16_leaf7 >= 0
 TREE_17_ONEHOT: t17_leaf0 + t17_leaf1 + t17_leaf2 + t17_leaf3 + t17_leaf4
   + t17_leaf5 + t17_leaf6 = 1
 T17_L0_f3_le: Overall_Qual_calc + 999993 t17_leaf0 <= 1e+06
 T17_L0_f3_le: Overall_Qual_calc + 999995 t17_leaf0 <= 1e+06
 T17_L0_f3_le: Overall_Qual_calc + 999997 t17_leaf0 <= 1e+06
 T17_L1_f3_le: Overall_Qual_calc + 999993 t17_leaf1 <= 1e+06
 T17_L1_f3_le: Overall_Qual_calc + 999995 t17_leaf1 <= 1e+06
 T17_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t17_leaf1 >= -1e+06
 T17_L2_f3_le: Overall_Qual_calc + 999993 t17_leaf2 <= 1e+06
 T17_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t17_leaf2 >= -1e+06
 T17_L2_f30_le: x_Fireplace_Qu + 4 t17_leaf2 <= 4
 T17_L3_f3_le: Overall_Qual_calc + 999993 t17_leaf3 <= 1e+06
 T17_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t17_leaf3 >= -1e+06
 T17_R3_f30_ge: x_Fireplace_Qu - t17_leaf3 >= -1
 T17_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t17_leaf4 >= -1e+06
 T17_L4_f27_le: x_Kitchen_Qual <= 4
 T17_L4_f17_le: x_1st_Flr_SF + 3936 t17_leaf4 <= 5000
 T17_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t17_leaf5 >= -1e+06
 T17_L5_f27_le: x_Kitchen_Qual <= 4
 T17_R5_f17_ge: x_1st_Flr_SF - 1064 t17_leaf5 >= 0
 T17_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t17_leaf6 >= -1e+06
 T17_R6_f27_ge: x_Kitchen_Qual - 5 t17_leaf6 >= -1
 TREE_18_ONEHOT: t18_leaf0 + t18_leaf1 + t18_leaf2 + t18_leaf3 + t18_leaf4
   + t18_leaf5 + t18_leaf6 + t18_leaf7 = 1
 T18_L0_f8_le: x_Exter_Qual + t18_leaf0 <= 4
 T18_L0_f231_le: central_air_no + 999999 t18_leaf0 <= 1e+06
 T18_L0_f15_le: const_Total_Bsmt_SF + 999076 t18_leaf0 <= 1e+06
 T18_L1_f8_le: x_Exter_Qual + t18_leaf1 <= 4
 T18_L1_f231_le: central_air_no + 999999 t18_leaf1 <= 1e+06
 T18_R1_f15_ge: const_Total_Bsmt_SF - 1.000924e+06 t18_leaf1 >= -1e+06
 T18_L2_f8_le: x_Exter_Qual + t18_leaf2 <= 4
 T18_R2_f231_ge: central_air_no - 1.000001e+06 t18_leaf2 >= -1e+06
 T18_L2_f4_le: const_Overall_Cond + 999994 t18_leaf2 <= 1e+06
 T18_L3_f8_le: x_Exter_Qual + t18_leaf3 <= 4
 T18_R3_f231_ge: central_air_no - 1.000001e+06 t18_leaf3 >= -1e+06
 T18_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t18_leaf3 >= -1e+06
 T18_R4_f8_ge: x_Exter_Qual - 4 t18_leaf4 >= -1
 T18_L4_f15_le: const_Total_Bsmt_SF + 998611 t18_leaf4 <= 1e+06
 T18_L4_f18_le: const_2nd_Flr_SF + 999172 t18_leaf4 <= 1e+06
 T18_R5_f8_ge: x_Exter_Qual - 4 t18_leaf5 >= -1
 T18_L5_f15_le: const_Total_Bsmt_SF + 998611 t18_leaf5 <= 1e+06
 T18_R5_f18_ge: const_2nd_Flr_SF - 1.000828e+06 t18_leaf5 >= -1e+06
 T18_R6_f8_ge: x_Exter_Qual - 4 t18_leaf6 >= -1
 T18_R6_f15_ge: const_Total_Bsmt_SF - 1.001389e+06 t18_leaf6 >= -1e+06
 T18_L6_f1_le: const_Lot_Area + 987901 t18_leaf6 <= 1e+06
 T18_R7_f8_ge: x_Exter_Qual - 4 t18_leaf7 >= -1
 T18_R7_f15_ge: const_Total_Bsmt_SF - 1.001389e+06 t18_leaf7 >= -1e+06
 T18_R7_f1_ge: const_Lot_Area - 1.012099e+06 t18_leaf7 >= -1e+06
 TREE_19_ONEHOT: t19_leaf0 + t19_leaf1 + t19_leaf2 + t19_leaf3 + t19_leaf4
   + t19_leaf5 + t19_leaf6 + t19_leaf7 = 1
 T19_L0_f5_le: const_Year_Built + 998019 t19_leaf0 <= 1e+06
 T19_L0_f30_le: x_Fireplace_Qu + 4 t19_leaf0 <= 4
 T19_L0_f20_le: x_Gr_Liv_Area + 9190 t19_leaf0 <= 10000
 T19_L1_f5_le: const_Year_Built + 998019 t19_leaf1 <= 1e+06
 T19_L1_f30_le: x_Fireplace_Qu + 4 t19_leaf1 <= 4
 T19_R1_f20_ge: x_Gr_Liv_Area - 810 t19_leaf1 >= 0
 T19_L2_f5_le: const_Year_Built + 998019 t19_leaf2 <= 1e+06
 T19_R2_f30_ge: x_Fireplace_Qu - t19_leaf2 >= -1
 T19_L2_f20_le: x_Gr_Liv_Area + 8221 t19_leaf2 <= 10000
 T19_L3_f5_le: const_Year_Built + 998019 t19_leaf3 <= 1e+06
 T19_R3_f30_ge: x_Fireplace_Qu - t19_leaf3 >= -1
 T19_R3_f20_ge: x_Gr_Liv_Area - 1779 t19_leaf3 >= 0
 T19_R4_f5_ge: const_Year_Built - 1.001981e+06 t19_leaf4 >= -1e+06
 T19_L4_f20_le: x_Gr_Liv_Area + 8039 t19_leaf4 <= 10000
 T19_L4_f15_le: const_Total_Bsmt_SF + 998506 t19_leaf4 <= 1e+06
 T19_R5_f5_ge: const_Year_Built - 1.001981e+06 t19_leaf5 >= -1e+06
 T19_L5_f20_le: x_Gr_Liv_Area + 8039 t19_leaf5 <= 10000
 T19_R5_f15_ge: const_Total_Bsmt_SF - 1.001494e+06 t19_leaf5 >= -1e+06
 T19_R6_f5_ge: const_Year_Built - 1.001981e+06 t19_leaf6 >= -1e+06
 T19_R6_f20_ge: x_Gr_Liv_Area - 1961 t19_leaf6 >= 0
 T19_L6_f15_le: const_Total_Bsmt_SF + 998813 t19_leaf6 <= 1e+06
 T19_R7_f5_ge: const_Year_Built - 1.001981e+06 t19_leaf7 >= -1e+06
 T19_R7_f20_ge: x_Gr_Liv_Area - 1961 t19_leaf7 >= 0
 T19_R7_f15_ge: const_Total_Bsmt_SF - 1.001187e+06 t19_leaf7 >= -1e+06
 TREE_20_ONEHOT: t20_leaf0 + t20_leaf1 + t20_leaf2 + t20_leaf3 + t20_leaf4
   + t20_leaf5 + t20_leaf6 + t20_leaf7 = 1
 T20_L0_f3_le: Overall_Qual_calc + 999994 t20_leaf0 <= 1e+06
 T20_L0_f66_le: const_MS_Zoning_RL + 999999 t20_leaf0 <= 1e+06
 T20_L0_f28_le: x_TotRms_AbvGrd + 10 t20_leaf0 <= 15
 T20_L1_f3_le: Overall_Qual_calc + 999994 t20_leaf1 <= 1e+06
 T20_L1_f66_le: const_MS_Zoning_RL + 999999 t20_leaf1 <= 1e+06
 T20_R1_f28_ge: x_TotRms_AbvGrd - 5 t20_leaf1 >= 0
 T20_L2_f3_le: Overall_Qual_calc + 999994 t20_leaf2 <= 1e+06
 T20_R2_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t20_leaf2 >= -1e+06
 T20_L2_f12_le: x_BsmtFin_SF_1 + 999443 t20_leaf2 <= 1e+06
 T20_L3_f3_le: Overall_Qual_calc + 999994 t20_leaf3 <= 1e+06
 T20_R3_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t20_leaf3 >= -1e+06
 T20_R3_f12_ge: x_BsmtFin_SF_1 - 557 t20_leaf3 >= 0
 T20_R4_f3_ge: Overall_Qual_calc - 1.000006e+06 t20_leaf4 >= -1e+06
 T20_L4_f3_le: Overall_Qual_calc + 999992 t20_leaf4 <= 1e+06
 T20_L4_f1_le: const_Lot_Area + 992500 t20_leaf4 <= 1e+06
 T20_R5_f3_ge: Overall_Qual_calc - 1.000006e+06 t20_leaf5 >= -1e+06
 T20_L5_f3_le: Overall_Qual_calc + 999992 t20_leaf5 <= 1e+06
 T20_R5_f1_ge: const_Lot_Area - 1.0075e+06 t20_leaf5 >= -1e+06
 T20_R6_f3_ge: Overall_Qual_calc - 1.000006e+06 t20_leaf6 >= -1e+06
 T20_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t20_leaf6 >= -1e+06
 T20_L6_f36_le: x_Wood_Deck_SF + 99760 t20_leaf6 <= 100000
 T20_R7_f3_ge: Overall_Qual_calc - 1.000006e+06 t20_leaf7 >= -1e+06
 T20_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t20_leaf7 >= -1e+06
 T20_R7_f36_ge: x_Wood_Deck_SF - 240 t20_leaf7 >= 0
 TREE_21_ONEHOT: t21_leaf0 + t21_leaf1 + t21_leaf2 + t21_leaf3 + t21_leaf4
   + t21_leaf5 + t21_leaf6 + t21_leaf7 = 1
 T21_L0_f3_le: Overall_Qual_calc + 999993 t21_leaf0 <= 1e+06
 T21_L0_f66_le: const_MS_Zoning_RL + 999999 t21_leaf0 <= 1e+06
 T21_L0_f4_le: const_Overall_Cond + 999995 t21_leaf0 <= 1e+06
 T21_L1_f3_le: Overall_Qual_calc + 999993 t21_leaf1 <= 1e+06
 T21_L1_f66_le: const_MS_Zoning_RL + 999999 t21_leaf1 <= 1e+06
 T21_R1_f4_ge: const_Overall_Cond - 1.000005e+06 t21_leaf1 >= -1e+06
 T21_L2_f3_le: Overall_Qual_calc + 999993 t21_leaf2 <= 1e+06
 T21_R2_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t21_leaf2 >= -1e+06
 T21_L2_f3_le: Overall_Qual_calc + 999994 t21_leaf2 <= 1e+06
 T21_L3_f3_le: Overall_Qual_calc + 999993 t21_leaf3 <= 1e+06
 T21_R3_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t21_leaf3 >= -1e+06
 T21_R3_f3_ge: Overall_Qual_calc - 1.000006e+06 t21_leaf3 >= -1e+06
 T21_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t21_leaf4 >= -1e+06
 T21_L4_f12_le: x_BsmtFin_SF_1 + 998903 t21_leaf4 <= 1e+06
 T21_L4_f33_le: x_Garage_Area + 99454 t21_leaf4 <= 100000
 T21_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t21_leaf5 >= -1e+06
 T21_L5_f12_le: x_BsmtFin_SF_1 + 998903 t21_leaf5 <= 1e+06
 T21_R5_f33_ge: x_Garage_Area - 546 t21_leaf5 >= 0
 T21_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t21_leaf6 >= -1e+06
 T21_R6_f12_ge: x_BsmtFin_SF_1 - 1097 t21_leaf6 >= 0
 T21_L6_f27_le: x_Kitchen_Qual <= 4
 T21_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t21_leaf7 >= -1e+06
 T21_R7_f12_ge: x_BsmtFin_SF_1 - 1097 t21_leaf7 >= 0
 T21_R7_f27_ge: x_Kitchen_Qual - 5 t21_leaf7 >= -1
 TREE_22_ONEHOT: t22_leaf0 + t22_leaf1 + t22_leaf2 + t22_leaf3 + t22_leaf4
   + t22_leaf5 + t22_leaf6 + t22_leaf7 = 1
 T22_L0_f20_le: x_Gr_Liv_Area + 8509 t22_leaf0 <= 10000
 T22_L0_f17_le: x_1st_Flr_SF + 3975 t22_leaf0 <= 5000
 T22_L0_f6_le: const_Year_Remod_Add + 998049 t22_leaf0 <= 1e+06
 T22_L1_f20_le: x_Gr_Liv_Area + 8509 t22_leaf1 <= 10000
 T22_L1_f17_le: x_1st_Flr_SF + 3975 t22_leaf1 <= 5000
 T22_R1_f6_ge: const_Year_Remod_Add - 1.001951e+06 t22_leaf1 >= -1e+06
 T22_L2_f20_le: x_Gr_Liv_Area + 8509 t22_leaf2 <= 10000
 T22_R2_f17_ge: x_1st_Flr_SF - 1025 t22_leaf2 >= 0
 T22_L2_f31_le: const_Garage_Yr_Blt + 998047 t22_leaf2 <= 1e+06
 T22_L3_f20_le: x_Gr_Liv_Area + 8509 t22_leaf3 <= 10000
 T22_R3_f17_ge: x_1st_Flr_SF - 1025 t22_leaf3 >= 0
 T22_R3_f31_ge: const_Garage_Yr_Blt - 1.001953e+06 t22_leaf3 >= -1e+06
 T22_R4_f20_ge: x_Gr_Liv_Area - 1491 t22_leaf4 >= 0
 T22_L4_f6_le: const_Year_Remod_Add + 998017 t22_leaf4 <= 1e+06
 T22_L4_f29_le: const_Fireplaces + 999998 t22_leaf4 <= 1e+06
 T22_R5_f20_ge: x_Gr_Liv_Area - 1491 t22_leaf5 >= 0
 T22_L5_f6_le: const_Year_Remod_Add + 998017 t22_leaf5 <= 1e+06
 T22_R5_f29_ge: const_Fireplaces - 1.000002e+06 t22_leaf5 >= -1e+06
 T22_R6_f20_ge: x_Gr_Liv_Area - 1491 t22_leaf6 >= 0
 T22_R6_f6_ge: const_Year_Remod_Add - 1.001983e+06 t22_leaf6 >= -1e+06
 T22_L6_f20_le: x_Gr_Liv_Area + 8098 t22_leaf6 <= 10000
 T22_R7_f20_ge: x_Gr_Liv_Area - 1491 t22_leaf7 >= 0
 T22_R7_f6_ge: const_Year_Remod_Add - 1.001983e+06 t22_leaf7 >= -1e+06
 T22_R7_f20_ge: x_Gr_Liv_Area - 1902 t22_leaf7 >= 0
 TREE_23_ONEHOT: t23_leaf0 + t23_leaf1 + t23_leaf2 + t23_leaf3 + t23_leaf4
   + t23_leaf5 + t23_leaf6 + t23_leaf7 = 1
 T23_L0_f3_le: Overall_Qual_calc + 999994 t23_leaf0 <= 1e+06
 T23_L0_f3_le: Overall_Qual_calc + 999996 t23_leaf0 <= 1e+06
 T23_L0_f5_le: const_Year_Built + 998050 t23_leaf0 <= 1e+06
 T23_L1_f3_le: Overall_Qual_calc + 999994 t23_leaf1 <= 1e+06
 T23_L1_f3_le: Overall_Qual_calc + 999996 t23_leaf1 <= 1e+06
 T23_R1_f5_ge: const_Year_Built - 1.00195e+06 t23_leaf1 >= -1e+06
 T23_L2_f3_le: Overall_Qual_calc + 999994 t23_leaf2 <= 1e+06
 T23_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t23_leaf2 >= -1e+06
 T23_L2_f66_le: const_MS_Zoning_RL + 999999 t23_leaf2 <= 1e+06
 T23_L3_f3_le: Overall_Qual_calc + 999994 t23_leaf3 <= 1e+06
 T23_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t23_leaf3 >= -1e+06
 T23_R3_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t23_leaf3 >= -1e+06
 T23_R4_f3_ge: Overall_Qual_calc - 1.000006e+06 t23_leaf4 >= -1e+06
 T23_L4_f32_le: const_Garage_Cars + 999997 t23_leaf4 <= 1e+06
 T23_L4_f1_le: const_Lot_Area + 986738 t23_leaf4 <= 1e+06
 T23_R5_f3_ge: Overall_Qual_calc - 1.000006e+06 t23_leaf5 >= -1e+06
 T23_L5_f32_le: const_Garage_Cars + 999997 t23_leaf5 <= 1e+06
 T23_R5_f1_ge: const_Lot_Area - 1.013262e+06 t23_leaf5 >= -1e+06
 T23_R6_f3_ge: Overall_Qual_calc - 1.000006e+06 t23_leaf6 >= -1e+06
 T23_R6_f32_ge: const_Garage_Cars - 1.000003e+06 t23_leaf6 >= -1e+06
 T23_L6_f3_le: Overall_Qual_calc + 999991 t23_leaf6 <= 1e+06
 T23_R7_f3_ge: Overall_Qual_calc - 1.000006e+06 t23_leaf7 >= -1e+06
 T23_R7_f32_ge: const_Garage_Cars - 1.000003e+06 t23_leaf7 >= -1e+06
 T23_R7_f3_ge: Overall_Qual_calc - 1.000009e+06 t23_leaf7 >= -1e+06
 TREE_24_ONEHOT: t24_leaf0 + t24_leaf1 + t24_leaf2 + t24_leaf3 + t24_leaf4
   + t24_leaf5 + t24_leaf6 + t24_leaf7 = 1
 T24_L0_f3_le: Overall_Qual_calc + 999993 t24_leaf0 <= 1e+06
 T24_L0_f32_le: const_Garage_Cars + 999998 t24_leaf0 <= 1e+06
 T24_L0_f17_le: x_1st_Flr_SF + 4106 t24_leaf0 <= 5000
 T24_L1_f3_le: Overall_Qual_calc + 999993 t24_leaf1 <= 1e+06
 T24_L1_f32_le: const_Garage_Cars + 999998 t24_leaf1 <= 1e+06
 T24_R1_f17_ge: x_1st_Flr_SF - 894 t24_leaf1 >= 0
 T24_L2_f3_le: Overall_Qual_calc + 999993 t24_leaf2 <= 1e+06
 T24_R2_f32_ge: const_Garage_Cars - 1.000002e+06 t24_leaf2 >= -1e+06
 T24_L2_f10_le: const_Bsmt_Qual + 999998 t24_leaf2 <= 1e+06
 T24_L3_f3_le: Overall_Qual_calc + 999993 t24_leaf3 <= 1e+06
 T24_R3_f32_ge: const_Garage_Cars - 1.000002e+06 t24_leaf3 >= -1e+06
 T24_R3_f10_ge: const_Bsmt_Qual - 1.000002e+06 t24_leaf3 >= -1e+06
 T24_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t24_leaf4 >= -1e+06
 T24_L4_f3_le: Overall_Qual_calc + 999992 t24_leaf4 <= 1e+06
 T24_L4_f1_le: const_Lot_Area + 992500 t24_leaf4 <= 1e+06
 T24_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t24_leaf5 >= -1e+06
 T24_L5_f3_le: Overall_Qual_calc + 999992 t24_leaf5 <= 1e+06
 T24_R5_f1_ge: const_Lot_Area - 1.0075e+06 t24_leaf5 >= -1e+06
 T24_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t24_leaf6 >= -1e+06
 T24_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t24_leaf6 >= -1e+06
 T24_L6_f1_le: const_Lot_Area + 987856 t24_leaf6 <= 1e+06
 T24_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t24_leaf7 >= -1e+06
 T24_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t24_leaf7 >= -1e+06
 T24_R7_f1_ge: const_Lot_Area - 1.012144e+06 t24_leaf7 >= -1e+06
 TREE_25_ONEHOT: t25_leaf0 + t25_leaf1 + t25_leaf2 + t25_leaf3 + t25_leaf4
   + t25_leaf5 + t25_leaf6 + t25_leaf7 = 1
 T25_L0_f8_le: x_Exter_Qual + t25_leaf0 <= 4
 T25_L0_f20_le: x_Gr_Liv_Area + 8976 t25_leaf0 <= 10000
 T25_L0_f15_le: const_Total_Bsmt_SF + 999239 t25_leaf0 <= 1e+06
 T25_L1_f8_le: x_Exter_Qual + t25_leaf1 <= 4
 T25_L1_f20_le: x_Gr_Liv_Area + 8976 t25_leaf1 <= 10000
 T25_R1_f15_ge: const_Total_Bsmt_SF - 1.000761e+06 t25_leaf1 >= -1e+06
 T25_L2_f8_le: x_Exter_Qual + t25_leaf2 <= 4
 T25_R2_f20_ge: x_Gr_Liv_Area - 1024 t25_leaf2 >= 0
 T25_L2_f12_le: x_BsmtFin_SF_1 + 999862 t25_leaf2 <= 1e+06
 T25_L3_f8_le: x_Exter_Qual + t25_leaf3 <= 4
 T25_R3_f20_ge: x_Gr_Liv_Area - 1024 t25_leaf3 >= 0
 T25_R3_f12_ge: x_BsmtFin_SF_1 - 138 t25_leaf3 >= 0
 T25_R4_f8_ge: x_Exter_Qual - 4 t25_leaf4 >= -1
 T25_L4_f20_le: x_Gr_Liv_Area + 8039 t25_leaf4 <= 10000
 T25_L4_f12_le: x_BsmtFin_SF_1 + 999252 t25_leaf4 <= 1e+06
 T25_R5_f8_ge: x_Exter_Qual - 4 t25_leaf5 >= -1
 T25_L5_f20_le: x_Gr_Liv_Area + 8039 t25_leaf5 <= 10000
 T25_R5_f12_ge: x_BsmtFin_SF_1 - 748 t25_leaf5 >= 0
 T25_R6_f8_ge: x_Exter_Qual - 4 t25_leaf6 >= -1
 T25_R6_f20_ge: x_Gr_Liv_Area - 1961 t25_leaf6 >= 0
 T25_L6_f15_le: const_Total_Bsmt_SF + 998945 t25_leaf6 <= 1e+06
 T25_R7_f8_ge: x_Exter_Qual - 4 t25_leaf7 >= -1
 T25_R7_f20_ge: x_Gr_Liv_Area - 1961 t25_leaf7 >= 0
 T25_R7_f15_ge: const_Total_Bsmt_SF - 1.001055e+06 t25_leaf7 >= -1e+06
 TREE_26_ONEHOT: t26_leaf0 + t26_leaf1 + t26_leaf2 + t26_leaf3 + t26_leaf4
   + t26_leaf5 + t26_leaf6 + t26_leaf7 = 1
 T26_L0_f3_le: Overall_Qual_calc + 999994 t26_leaf0 <= 1e+06
 T26_L0_f257_le: x_paved_drive_is_N <= 1
 T26_L0_f17_le: x_1st_Flr_SF + 3950 t26_leaf0 <= 5000
 T26_L1_f3_le: Overall_Qual_calc + 999994 t26_leaf1 <= 1e+06
 T26_L1_f257_le: x_paved_drive_is_N <= 1
 T26_R1_f17_ge: x_1st_Flr_SF - 1050 t26_leaf1 >= 0
 T26_L2_f3_le: Overall_Qual_calc + 999994 t26_leaf2 <= 1e+06
 T26_R2_f257_ge: x_paved_drive_is_N - t26_leaf2 >= 0
 T26_L2_f20_le: x_Gr_Liv_Area + 9040 t26_leaf2 <= 10000
 T26_L3_f3_le: Overall_Qual_calc + 999994 t26_leaf3 <= 1e+06
 T26_R3_f257_ge: x_paved_drive_is_N - t26_leaf3 >= 0
 T26_R3_f20_ge: x_Gr_Liv_Area - 960 t26_leaf3 >= 0
 T26_R4_f3_ge: Overall_Qual_calc - 1.000006e+06 t26_leaf4 >= -1e+06
 T26_L4_f3_le: Overall_Qual_calc + 999992 t26_leaf4 <= 1e+06
 T26_L4_f20_le: x_Gr_Liv_Area + 8512 t26_leaf4 <= 10000
 T26_R5_f3_ge: Overall_Qual_calc - 1.000006e+06 t26_leaf5 >= -1e+06
 T26_L5_f3_le: Overall_Qual_calc + 999992 t26_leaf5 <= 1e+06
 T26_R5_f20_ge: x_Gr_Liv_Area - 1488 t26_leaf5 >= 0
 T26_R6_f3_ge: Overall_Qual_calc - 1.000006e+06 t26_leaf6 >= -1e+06
 T26_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t26_leaf6 >= -1e+06
 T26_L6_f20_le: x_Gr_Liv_Area + 7776 t26_leaf6 <= 10000
 T26_R7_f3_ge: Overall_Qual_calc - 1.000006e+06 t26_leaf7 >= -1e+06
 T26_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t26_leaf7 >= -1e+06
 T26_R7_f20_ge: x_Gr_Liv_Area - 2224 t26_leaf7 >= 0
 TREE_27_ONEHOT: t27_leaf0 + t27_leaf1 + t27_leaf2 + t27_leaf3 + t27_leaf4
   + t27_leaf5 + t27_leaf6 + t27_leaf7 = 1
 T27_L0_f8_le: x_Exter_Qual + t27_leaf0 <= 4
 T27_L0_f231_le: central_air_no + 999999 t27_leaf0 <= 1e+06
 T27_L0_f1_le: const_Lot_Area + 990884 t27_leaf0 <= 1e+06
 T27_L1_f8_le: x_Exter_Qual + t27_leaf1 <= 4
 T27_L1_f231_le: central_air_no + 999999 t27_leaf1 <= 1e+06
 T27_R1_f1_ge: const_Lot_Area - 1.009116e+06 t27_leaf1 >= -1e+06
 T27_L2_f8_le: x_Exter_Qual + t27_leaf2 <= 4
 T27_R2_f231_ge: central_air_no - 1.000001e+06 t27_leaf2 >= -1e+06
 T27_L2_f4_le: const_Overall_Cond + 999994 t27_leaf2 <= 1e+06
 T27_L3_f8_le: x_Exter_Qual + t27_leaf3 <= 4
 T27_R3_f231_ge: central_air_no - 1.000001e+06 t27_leaf3 >= -1e+06
 T27_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t27_leaf3 >= -1e+06
 T27_R4_f8_ge: x_Exter_Qual - 4 t27_leaf4 >= -1
 T27_L4_f30_le: x_Fireplace_Qu + 3 t27_leaf4 <= 4
 T27_L4_f33_le: x_Garage_Area + 99503 t27_leaf4 <= 100000
 T27_R5_f8_ge: x_Exter_Qual - 4 t27_leaf5 >= -1
 T27_L5_f30_le: x_Fireplace_Qu + 3 t27_leaf5 <= 4
 T27_R5_f33_ge: x_Garage_Area - 497 t27_leaf5 >= 0
 T27_R6_f8_ge: x_Exter_Qual - 4 t27_leaf6 >= -1
 T27_R6_f30_ge: x_Fireplace_Qu - 2 t27_leaf6 >= -1
 T27_L6_f33_le: x_Garage_Area + 99294 t27_leaf6 <= 100000
 T27_R7_f8_ge: x_Exter_Qual - 4 t27_leaf7 >= -1
 T27_R7_f30_ge: x_Fireplace_Qu - 2 t27_leaf7 >= -1
 T27_R7_f33_ge: x_Garage_Area - 706 t27_leaf7 >= 0
 TREE_28_ONEHOT: t28_leaf0 + t28_leaf1 + t28_leaf2 + t28_leaf3 + t28_leaf4
   + t28_leaf5 + t28_leaf6 + t28_leaf7 = 1
 T28_L0_f20_le: x_Gr_Liv_Area + 8584 t28_leaf0 <= 10000
 T28_L0_f5_le: const_Year_Built + 998047 t28_leaf0 <= 1e+06
 T28_L0_f20_le: x_Gr_Liv_Area + 9167 t28_leaf0 <= 10000
 T28_L1_f20_le: x_Gr_Liv_Area + 8584 t28_leaf1 <= 10000
 T28_L1_f5_le: const_Year_Built + 998047 t28_leaf1 <= 1e+06
 T28_R1_f20_ge: x_Gr_Liv_Area - 833 t28_leaf1 >= 0
 T28_L2_f20_le: x_Gr_Liv_Area + 8584 t28_leaf2 <= 10000
 T28_R2_f5_ge: const_Year_Built - 1.001953e+06 t28_leaf2 >= -1e+06
 T28_L2_f17_le: x_1st_Flr_SF + 4049 t28_leaf2 <= 5000
 T28_L3_f20_le: x_Gr_Liv_Area + 8584 t28_leaf3 <= 10000
 T28_R3_f5_ge: const_Year_Built - 1.001953e+06 t28_leaf3 >= -1e+06
 T28_R3_f17_ge: x_1st_Flr_SF - 951 t28_leaf3 >= 0
 T28_R4_f20_ge: x_Gr_Liv_Area - 1416 t28_leaf4 >= 0
 T28_L4_f20_le: x_Gr_Liv_Area + 7776 t28_leaf4 <= 10000
 T28_L4_f21_le: const_Bsmt_Full_Bath + 999999 t28_leaf4 <= 1e+06
 T28_R5_f20_ge: x_Gr_Liv_Area - 1416 t28_leaf5 >= 0
 T28_L5_f20_le: x_Gr_Liv_Area + 7776 t28_leaf5 <= 10000
 T28_R5_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t28_leaf5 >= -1e+06
 T28_R6_f20_ge: x_Gr_Liv_Area - 1416 t28_leaf6 >= 0
 T28_R6_f20_ge: x_Gr_Liv_Area - 2224 t28_leaf6 >= 0
 T28_L6_f1_le: const_Lot_Area + 988300 t28_leaf6 <= 1e+06
 T28_R7_f20_ge: x_Gr_Liv_Area - 1416 t28_leaf7 >= 0
 T28_R7_f20_ge: x_Gr_Liv_Area - 2224 t28_leaf7 >= 0
 T28_R7_f1_ge: const_Lot_Area - 1.0117e+06 t28_leaf7 >= -1e+06
 TREE_29_ONEHOT: t29_leaf0 + t29_leaf1 + t29_leaf2 + t29_leaf3 + t29_leaf4
   + t29_leaf5 + t29_leaf6 + t29_leaf7 = 1
 T29_L0_f3_le: Overall_Qual_calc + 999993 t29_leaf0 <= 1e+06
 T29_L0_f3_le: Overall_Qual_calc + 999996 t29_leaf0 <= 1e+06
 T29_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t29_leaf0 <= 1e+06
 T29_L1_f3_le: Overall_Qual_calc + 999993 t29_leaf1 <= 1e+06
 T29_L1_f3_le: Overall_Qual_calc + 999996 t29_leaf1 <= 1e+06
 T29_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t29_leaf1
   >= -1e+06
 T29_L2_f3_le: Overall_Qual_calc + 999993 t29_leaf2 <= 1e+06
 T29_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t29_leaf2 >= -1e+06
 T29_L2_f12_le: x_BsmtFin_SF_1 + 999649 t29_leaf2 <= 1e+06
 T29_L3_f3_le: Overall_Qual_calc + 999993 t29_leaf3 <= 1e+06
 T29_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t29_leaf3 >= -1e+06
 T29_R3_f12_ge: x_BsmtFin_SF_1 - 351 t29_leaf3 >= 0
 T29_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t29_leaf4 >= -1e+06
 T29_L4_f15_le: const_Total_Bsmt_SF + 998525 t29_leaf4 <= 1e+06
 T29_L4_f1_le: const_Lot_Area + 988020 t29_leaf4 <= 1e+06
 T29_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t29_leaf5 >= -1e+06
 T29_L5_f15_le: const_Total_Bsmt_SF + 998525 t29_leaf5 <= 1e+06
 T29_R5_f1_ge: const_Lot_Area - 1.01198e+06 t29_leaf5 >= -1e+06
 T29_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t29_leaf6 >= -1e+06
 T29_R6_f15_ge: const_Total_Bsmt_SF - 1.001475e+06 t29_leaf6 >= -1e+06
 T29_L6_f30_le: x_Fireplace_Qu + 2 t29_leaf6 <= 4
 T29_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t29_leaf7 >= -1e+06
 T29_R7_f15_ge: const_Total_Bsmt_SF - 1.001475e+06 t29_leaf7 >= -1e+06
 T29_R7_f30_ge: x_Fireplace_Qu - 3 t29_leaf7 >= -1
 TREE_30_ONEHOT: t30_leaf0 + t30_leaf1 + t30_leaf2 + t30_leaf3 + t30_leaf4
   + t30_leaf5 + t30_leaf6 + t30_leaf7 = 1
 T30_L0_f3_le: Overall_Qual_calc + 999994 t30_leaf0 <= 1e+06
 T30_L0_f11_le: x_Bsmt_Cond + 2 t30_leaf0 <= 4
 T30_L0_f33_le: x_Garage_Area + 99382 t30_leaf0 <= 100000
 T30_L1_f3_le: Overall_Qual_calc + 999994 t30_leaf1 <= 1e+06
 T30_L1_f11_le: x_Bsmt_Cond + 2 t30_leaf1 <= 4
 T30_R1_f33_ge: x_Garage_Area - 618 t30_leaf1 >= 0
 T30_L2_f3_le: Overall_Qual_calc + 999994 t30_leaf2 <= 1e+06
 T30_R2_f11_ge: x_Bsmt_Cond - 3 t30_leaf2 >= -1
 T30_L2_f66_le: const_MS_Zoning_RL + 999999 t30_leaf2 <= 1e+06
 T30_L3_f3_le: Overall_Qual_calc + 999994 t30_leaf3 <= 1e+06
 T30_R3_f11_ge: x_Bsmt_Cond - 3 t30_leaf3 >= -1
 T30_R3_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t30_leaf3 >= -1e+06
 T30_R4_f3_ge: Overall_Qual_calc - 1.000006e+06 t30_leaf4 >= -1e+06
 T30_L4_f32_le: const_Garage_Cars + 999997 t30_leaf4 <= 1e+06
 T30_L4_f1_le: const_Lot_Area + 993627 t30_leaf4 <= 1e+06
 T30_R5_f3_ge: Overall_Qual_calc - 1.000006e+06 t30_leaf5 >= -1e+06
 T30_L5_f32_le: const_Garage_Cars + 999997 t30_leaf5 <= 1e+06
 T30_R5_f1_ge: const_Lot_Area - 1.006373e+06 t30_leaf5 >= -1e+06
 T30_R6_f3_ge: Overall_Qual_calc - 1.000006e+06 t30_leaf6 >= -1e+06
 T30_R6_f32_ge: const_Garage_Cars - 1.000003e+06 t30_leaf6 >= -1e+06
 T30_L6_f1_le: const_Lot_Area + 986482 t30_leaf6 <= 1e+06
 T30_R7_f3_ge: Overall_Qual_calc - 1.000006e+06 t30_leaf7 >= -1e+06
 T30_R7_f32_ge: const_Garage_Cars - 1.000003e+06 t30_leaf7 >= -1e+06
 T30_R7_f1_ge: const_Lot_Area - 1.013518e+06 t30_leaf7 >= -1e+06
 TREE_31_ONEHOT: t31_leaf0 + t31_leaf1 + t31_leaf2 + t31_leaf3 + t31_leaf4
   + t31_leaf5 + t31_leaf6 + t31_leaf7 = 1
 T31_L0_f20_le: x_Gr_Liv_Area + 8512 t31_leaf0 <= 10000
 T31_L0_f35_le: x_Garage_Cond + 2 t31_leaf0 <= 4
 T31_L0_f20_le: x_Gr_Liv_Area + 8845 t31_leaf0 <= 10000
 T31_L1_f20_le: x_Gr_Liv_Area + 8512 t31_leaf1 <= 10000
 T31_L1_f35_le: x_Garage_Cond + 2 t31_leaf1 <= 4
 T31_R1_f20_ge: x_Gr_Liv_Area - 1155 t31_leaf1 >= 0
 T31_L2_f20_le: x_Gr_Liv_Area + 8512 t31_leaf2 <= 10000
 T31_R2_f35_ge: x_Garage_Cond - 3 t31_leaf2 >= -1
 T31_L2_f17_le: x_1st_Flr_SF + 4012 t31_leaf2 <= 5000
 T31_L3_f20_le: x_Gr_Liv_Area + 8512 t31_leaf3 <= 10000
 T31_R3_f35_ge: x_Garage_Cond - 3 t31_leaf3 >= -1
 T31_R3_f17_ge: x_1st_Flr_SF - 988 t31_leaf3 >= 0
 T31_R4_f20_ge: x_Gr_Liv_Area - 1488 t31_leaf4 >= 0
 T31_L4_f12_le: x_BsmtFin_SF_1 + 999227 t31_leaf4 <= 1e+06
 T31_L4_f26_le: x_Kitchen_AbvGr + t31_leaf4 <= 3
 T31_R5_f20_ge: x_Gr_Liv_Area - 1488 t31_leaf5 >= 0
 T31_L5_f12_le: x_BsmtFin_SF_1 + 999227 t31_leaf5 <= 1e+06
 T31_R5_f26_ge: x_Kitchen_AbvGr - 2 t31_leaf5 >= 0
 T31_R6_f20_ge: x_Gr_Liv_Area - 1488 t31_leaf6 >= 0
 T31_R6_f12_ge: x_BsmtFin_SF_1 - 773 t31_leaf6 >= 0
 T31_L6_f20_le: x_Gr_Liv_Area + 8127 t31_leaf6 <= 10000
 T31_R7_f20_ge: x_Gr_Liv_Area - 1488 t31_leaf7 >= 0
 T31_R7_f12_ge: x_BsmtFin_SF_1 - 773 t31_leaf7 >= 0
 T31_R7_f20_ge: x_Gr_Liv_Area - 1873 t31_leaf7 >= 0
 TREE_32_ONEHOT: t32_leaf0 + t32_leaf1 + t32_leaf2 + t32_leaf3 + t32_leaf4
   + t32_leaf5 + t32_leaf6 + t32_leaf7 = 1
 T32_L0_f32_le: const_Garage_Cars + 999998 t32_leaf0 <= 1e+06
 T32_L0_f15_le: const_Total_Bsmt_SF + 999352 t32_leaf0 <= 1e+06
 T32_L0_f6_le: const_Year_Remod_Add + 998022 t32_leaf0 <= 1e+06
 T32_L1_f32_le: const_Garage_Cars + 999998 t32_leaf1 <= 1e+06
 T32_L1_f15_le: const_Total_Bsmt_SF + 999352 t32_leaf1 <= 1e+06
 T32_R1_f6_ge: const_Year_Remod_Add - 1.001978e+06 t32_leaf1 >= -1e+06
 T32_L2_f32_le: const_Garage_Cars + 999998 t32_leaf2 <= 1e+06
 T32_R2_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t32_leaf2 >= -1e+06
 T32_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t32_leaf2 <= 1e+06
 T32_L3_f32_le: const_Garage_Cars + 999998 t32_leaf3 <= 1e+06
 T32_R3_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t32_leaf3 >= -1e+06
 T32_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t32_leaf3
   >= -1e+06
 T32_R4_f32_ge: const_Garage_Cars - 1.000002e+06 t32_leaf4 >= -1e+06
 T32_L4_f15_le: const_Total_Bsmt_SF + 998581 t32_leaf4 <= 1e+06
 T32_L4_f18_le: const_2nd_Flr_SF + 999017 t32_leaf4 <= 1e+06
 T32_R5_f32_ge: const_Garage_Cars - 1.000002e+06 t32_leaf5 >= -1e+06
 T32_L5_f15_le: const_Total_Bsmt_SF + 998581 t32_leaf5 <= 1e+06
 T32_R5_f18_ge: const_2nd_Flr_SF - 1.000983e+06 t32_leaf5 >= -1e+06
 T32_R6_f32_ge: const_Garage_Cars - 1.000002e+06 t32_leaf6 >= -1e+06
 T32_R6_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t32_leaf6 >= -1e+06
 T32_L6_f29_le: const_Fireplaces + 999999 t32_leaf6 <= 1e+06
 T32_R7_f32_ge: const_Garage_Cars - 1.000002e+06 t32_leaf7 >= -1e+06
 T32_R7_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t32_leaf7 >= -1e+06
 T32_R7_f29_ge: const_Fireplaces - 1.000001e+06 t32_leaf7 >= -1e+06
 TREE_33_ONEHOT: t33_leaf0 + t33_leaf1 + t33_leaf2 + t33_leaf3 + t33_leaf4
   + t33_leaf5 + t33_leaf6 + t33_leaf7 = 1
 T33_L0_f15_le: const_Total_Bsmt_SF + 998760 t33_leaf0 <= 1e+06
 T33_L0_f18_le: const_2nd_Flr_SF + 999396 t33_leaf0 <= 1e+06
 T33_L0_f15_le: const_Total_Bsmt_SF + 999226 t33_leaf0 <= 1e+06
 T33_L1_f15_le: const_Total_Bsmt_SF + 998760 t33_leaf1 <= 1e+06
 T33_L1_f18_le: const_2nd_Flr_SF + 999396 t33_leaf1 <= 1e+06
 T33_R1_f15_ge: const_Total_Bsmt_SF - 1.000774e+06 t33_leaf1 >= -1e+06
 T33_L2_f15_le: const_Total_Bsmt_SF + 998760 t33_leaf2 <= 1e+06
 T33_R2_f18_ge: const_2nd_Flr_SF - 1.000604e+06 t33_leaf2 >= -1e+06
 T33_L2_f67_le: const_MS_Zoning_RM + 999999 t33_leaf2 <= 1e+06
 T33_L3_f15_le: const_Total_Bsmt_SF + 998760 t33_leaf3 <= 1e+06
 T33_R3_f18_ge: const_2nd_Flr_SF - 1.000604e+06 t33_leaf3 >= -1e+06
 T33_R3_f67_ge: const_MS_Zoning_RM - 1.000001e+06 t33_leaf3 >= -1e+06
 T33_R4_f15_ge: const_Total_Bsmt_SF - 1.00124e+06 t33_leaf4 >= -1e+06
 T33_L4_f3_le: Overall_Qual_calc + 999991 t33_leaf4 <= 1e+06
 T33_L4_f14_le: x_Bsmt_Unf_SF + 98786 t33_leaf4 <= 100000
 T33_R5_f15_ge: const_Total_Bsmt_SF - 1.00124e+06 t33_leaf5 >= -1e+06
 T33_L5_f3_le: Overall_Qual_calc + 999991 t33_leaf5 <= 1e+06
 T33_R5_f14_ge: x_Bsmt_Unf_SF - 1214 t33_leaf5 >= 0
 T33_R6_f15_ge: const_Total_Bsmt_SF - 1.00124e+06 t33_leaf6 >= -1e+06
 T33_R6_f3_ge: Overall_Qual_calc - 1.000009e+06 t33_leaf6 >= -1e+06
 T33_L6_f0_le: const_Lot_Frontage + 999886 t33_leaf6 <= 1e+06
 T33_R7_f15_ge: const_Total_Bsmt_SF - 1.00124e+06 t33_leaf7 >= -1e+06
 T33_R7_f3_ge: Overall_Qual_calc - 1.000009e+06 t33_leaf7 >= -1e+06
 T33_R7_f0_ge: const_Lot_Frontage - 1.000114e+06 t33_leaf7 >= -1e+06
 TREE_34_ONEHOT: t34_leaf0 + t34_leaf1 + t34_leaf2 + t34_leaf3 + t34_leaf4
   + t34_leaf5 + t34_leaf6 + t34_leaf7 = 1
 T34_L0_f27_le: x_Kitchen_Qual + t34_leaf0 <= 4
 T34_L0_f259_le: x_paved_drive_is_Y <= 1
 T34_L0_f15_le: const_Total_Bsmt_SF + 998976 t34_leaf0 <= 1e+06
 T34_L1_f27_le: x_Kitchen_Qual + t34_leaf1 <= 4
 T34_L1_f259_le: x_paved_drive_is_Y <= 1
 T34_R1_f15_ge: const_Total_Bsmt_SF - 1.001024e+06 t34_leaf1 >= -1e+06
 T34_L2_f27_le: x_Kitchen_Qual + t34_leaf2 <= 4
 T34_R2_f259_ge: x_paved_drive_is_Y - t34_leaf2 >= 0
 T34_L2_f20_le: x_Gr_Liv_Area + 8502 t34_leaf2 <= 10000
 T34_L3_f27_le: x_Kitchen_Qual + t34_leaf3 <= 4
 T34_R3_f259_ge: x_paved_drive_is_Y - t34_leaf3 >= 0
 T34_R3_f20_ge: x_Gr_Liv_Area - 1498 t34_leaf3 >= 0
 T34_R4_f27_ge: x_Kitchen_Qual - 4 t34_leaf4 >= -1
 T34_L4_f17_le: x_1st_Flr_SF + 3945 t34_leaf4 <= 5000
 T34_L4_f1_le: const_Lot_Area + 993627 t34_leaf4 <= 1e+06
 T34_R5_f27_ge: x_Kitchen_Qual - 4 t34_leaf5 >= -1
 T34_L5_f17_le: x_1st_Flr_SF + 3945 t34_leaf5 <= 5000
 T34_R5_f1_ge: const_Lot_Area - 1.006373e+06 t34_leaf5 >= -1e+06
 T34_R6_f27_ge: x_Kitchen_Qual - 4 t34_leaf6 >= -1
 T34_R6_f17_ge: x_1st_Flr_SF - 1055 t34_leaf6 >= 0
 T34_L6_f20_le: x_Gr_Liv_Area + 8442 t34_leaf6 <= 10000
 T34_R7_f27_ge: x_Kitchen_Qual - 4 t34_leaf7 >= -1
 T34_R7_f17_ge: x_1st_Flr_SF - 1055 t34_leaf7 >= 0
 T34_R7_f20_ge: x_Gr_Liv_Area - 1558 t34_leaf7 >= 0
 TREE_35_ONEHOT: t35_leaf0 + t35_leaf1 + t35_leaf2 + t35_leaf3 + t35_leaf4
   + t35_leaf5 + t35_leaf6 + t35_leaf7 = 1
 T35_L0_f27_le: x_Kitchen_Qual + t35_leaf0 <= 4
 T35_L0_f66_le: const_MS_Zoning_RL + 999999 t35_leaf0 <= 1e+06
 T35_L0_f6_le: const_Year_Remod_Add + 998007 t35_leaf0 <= 1e+06
 T35_L1_f27_le: x_Kitchen_Qual + t35_leaf1 <= 4
 T35_L1_f66_le: const_MS_Zoning_RL + 999999 t35_leaf1 <= 1e+06
 T35_R1_f6_ge: const_Year_Remod_Add - 1.001993e+06 t35_leaf1 >= -1e+06
 T35_L2_f27_le: x_Kitchen_Qual + t35_leaf2 <= 4
 T35_R2_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t35_leaf2 >= -1e+06
 T35_L2_f6_le: const_Year_Remod_Add + 998041 t35_leaf2 <= 1e+06
 T35_L3_f27_le: x_Kitchen_Qual + t35_leaf3 <= 4
 T35_R3_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t35_leaf3 >= -1e+06
 T35_R3_f6_ge: const_Year_Remod_Add - 1.001959e+06 t35_leaf3 >= -1e+06
 T35_R4_f27_ge: x_Kitchen_Qual - 4 t35_leaf4 >= -1
 T35_L4_f1_le: const_Lot_Area + 988235 t35_leaf4 <= 1e+06
 T35_L4_f7_le: x_Mas_Vnr_Area + 1826 t35_leaf4 <= 2000
 T35_R5_f27_ge: x_Kitchen_Qual - 4 t35_leaf5 >= -1
 T35_L5_f1_le: const_Lot_Area + 988235 t35_leaf5 <= 1e+06
 T35_R5_f7_ge: x_Mas_Vnr_Area - 174 t35_leaf5 >= 0
 T35_R6_f27_ge: x_Kitchen_Qual - 4 t35_leaf6 >= -1
 T35_R6_f1_ge: const_Lot_Area - 1.011765e+06 t35_leaf6 >= -1e+06
 T35_L6_f18_le: const_2nd_Flr_SF + 998632 t35_leaf6 <= 1e+06
 T35_R7_f27_ge: x_Kitchen_Qual - 4 t35_leaf7 >= -1
 T35_R7_f1_ge: const_Lot_Area - 1.011765e+06 t35_leaf7 >= -1e+06
 T35_R7_f18_ge: const_2nd_Flr_SF - 1.001368e+06 t35_leaf7 >= -1e+06
 TREE_36_ONEHOT: t36_leaf0 + t36_leaf1 + t36_leaf2 + t36_leaf3 + t36_leaf4
   + t36_leaf5 + t36_leaf6 + t36_leaf7 = 1
 T36_L0_f20_le: x_Gr_Liv_Area + 8718 t36_leaf0 <= 10000
 T36_L0_f3_le: Overall_Qual_calc + 999996 t36_leaf0 <= 1e+06
 T36_L0_f17_le: x_1st_Flr_SF + 4132 t36_leaf0 <= 5000
 T36_L1_f20_le: x_Gr_Liv_Area + 8718 t36_leaf1 <= 10000
 T36_L1_f3_le: Overall_Qual_calc + 999996 t36_leaf1 <= 1e+06
 T36_R1_f17_ge: x_1st_Flr_SF - 868 t36_leaf1 >= 0
 T36_L2_f20_le: x_Gr_Liv_Area + 8718 t36_leaf2 <= 10000
 T36_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t36_leaf2 >= -1e+06
 T36_L2_f12_le: x_BsmtFin_SF_1 + 999443 t36_leaf2 <= 1e+06
 T36_L3_f20_le: x_Gr_Liv_Area + 8718 t36_leaf3 <= 10000
 T36_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t36_leaf3 >= -1e+06
 T36_R3_f12_ge: x_BsmtFin_SF_1 - 557 t36_leaf3 >= 0
 T36_R4_f20_ge: x_Gr_Liv_Area - 1282 t36_leaf4 >= 0
 T36_L4_f20_le: x_Gr_Liv_Area + 7737 t36_leaf4 <= 10000
 T36_L4_f12_le: x_BsmtFin_SF_1 + 999216 t36_leaf4 <= 1e+06
 T36_R5_f20_ge: x_Gr_Liv_Area - 1282 t36_leaf5 >= 0
 T36_L5_f20_le: x_Gr_Liv_Area + 7737 t36_leaf5 <= 10000
 T36_R5_f12_ge: x_BsmtFin_SF_1 - 784 t36_leaf5 >= 0
 T36_R6_f20_ge: x_Gr_Liv_Area - 1282 t36_leaf6 >= 0
 T36_R6_f20_ge: x_Gr_Liv_Area - 2263 t36_leaf6 >= 0
 T36_L6_f17_le: x_1st_Flr_SF + 3498 t36_leaf6 <= 5000
 T36_R7_f20_ge: x_Gr_Liv_Area - 1282 t36_leaf7 >= 0
 T36_R7_f20_ge: x_Gr_Liv_Area - 2263 t36_leaf7 >= 0
 T36_R7_f17_ge: x_1st_Flr_SF - 1502 t36_leaf7 >= 0
 TREE_37_ONEHOT: t37_leaf0 + t37_leaf1 + t37_leaf2 + t37_leaf3 + t37_leaf4
   + t37_leaf5 + t37_leaf6 + t37_leaf7 = 1
 T37_L0_f20_le: x_Gr_Liv_Area + 8506 t37_leaf0 <= 10000
 T37_L0_f3_le: Overall_Qual_calc + 999996 t37_leaf0 <= 1e+06
 T37_L0_f17_le: x_1st_Flr_SF + 4092 t37_leaf0 <= 5000
 T37_L1_f20_le: x_Gr_Liv_Area + 8506 t37_leaf1 <= 10000
 T37_L1_f3_le: Overall_Qual_calc + 999996 t37_leaf1 <= 1e+06
 T37_R1_f17_ge: x_1st_Flr_SF - 908 t37_leaf1 >= 0
 T37_L2_f20_le: x_Gr_Liv_Area + 8506 t37_leaf2 <= 10000
 T37_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t37_leaf2 >= -1e+06
 T37_L2_f15_le: const_Total_Bsmt_SF + 999025 t37_leaf2 <= 1e+06
 T37_L3_f20_le: x_Gr_Liv_Area + 8506 t37_leaf3 <= 10000
 T37_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t37_leaf3 >= -1e+06
 T37_R3_f15_ge: const_Total_Bsmt_SF - 1.000975e+06 t37_leaf3 >= -1e+06
 T37_R4_f20_ge: x_Gr_Liv_Area - 1494 t37_leaf4 >= 0
 T37_L4_f6_le: const_Year_Remod_Add + 998017 t37_leaf4 <= 1e+06
 T37_L4_f12_le: x_BsmtFin_SF_1 + 999808 t37_leaf4 <= 1e+06
 T37_R5_f20_ge: x_Gr_Liv_Area - 1494 t37_leaf5 >= 0
 T37_L5_f6_le: const_Year_Remod_Add + 998017 t37_leaf5 <= 1e+06
 T37_R5_f12_ge: x_BsmtFin_SF_1 - 192 t37_leaf5 >= 0
 T37_R6_f20_ge: x_Gr_Liv_Area - 1494 t37_leaf6 >= 0
 T37_R6_f6_ge: const_Year_Remod_Add - 1.001983e+06 t37_leaf6 >= -1e+06
 T37_L6_f20_le: x_Gr_Liv_Area + 8032 t37_leaf6 <= 10000
 T37_R7_f20_ge: x_Gr_Liv_Area - 1494 t37_leaf7 >= 0
 T37_R7_f6_ge: const_Year_Remod_Add - 1.001983e+06 t37_leaf7 >= -1e+06
 T37_R7_f20_ge: x_Gr_Liv_Area - 1968 t37_leaf7 >= 0
 TREE_38_ONEHOT: t38_leaf0 + t38_leaf1 + t38_leaf2 + t38_leaf3 + t38_leaf4
   + t38_leaf5 + t38_leaf6 + t38_leaf7 = 1
 T38_L0_f3_le: Overall_Qual_calc + 999994 t38_leaf0 <= 1e+06
 T38_L0_f232_le: central_air_yes + 999999 t38_leaf0 <= 1e+06
 T38_L0_f3_le: Overall_Qual_calc + 999997 t38_leaf0 <= 1e+06
 T38_L1_f3_le: Overall_Qual_calc + 999994 t38_leaf1 <= 1e+06
 T38_L1_f232_le: central_air_yes + 999999 t38_leaf1 <= 1e+06
 T38_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t38_leaf1 >= -1e+06
 T38_L2_f3_le: Overall_Qual_calc + 999994 t38_leaf2 <= 1e+06
 T38_R2_f232_ge: central_air_yes - 1.000001e+06 t38_leaf2 >= -1e+06
 T38_L2_f217_le: x_b1_is_Unf <= 1
 T38_L3_f3_le: Overall_Qual_calc + 999994 t38_leaf3 <= 1e+06
 T38_R3_f232_ge: central_air_yes - 1.000001e+06 t38_leaf3 >= -1e+06
 T38_R3_f217_ge: x_b1_is_Unf - t38_leaf3 >= 0
 T38_R4_f3_ge: Overall_Qual_calc - 1.000006e+06 t38_leaf4 >= -1e+06
 T38_L4_f20_le: x_Gr_Liv_Area + 8200 t38_leaf4 <= 10000
 T38_L4_f20_le: x_Gr_Liv_Area + 8856 t38_leaf4 <= 10000
 T38_R5_f3_ge: Overall_Qual_calc - 1.000006e+06 t38_leaf5 >= -1e+06
 T38_L5_f20_le: x_Gr_Liv_Area + 8200 t38_leaf5 <= 10000
 T38_R5_f20_ge: x_Gr_Liv_Area - 1144 t38_leaf5 >= 0
 T38_R6_f3_ge: Overall_Qual_calc - 1.000006e+06 t38_leaf6 >= -1e+06
 T38_R6_f20_ge: x_Gr_Liv_Area - 1800 t38_leaf6 >= 0
 T38_L6_f20_le: x_Gr_Liv_Area + 7673 t38_leaf6 <= 10000
 T38_R7_f3_ge: Overall_Qual_calc - 1.000006e+06 t38_leaf7 >= -1e+06
 T38_R7_f20_ge: x_Gr_Liv_Area - 1800 t38_leaf7 >= 0
 T38_R7_f20_ge: x_Gr_Liv_Area - 2327 t38_leaf7 >= 0
 TREE_39_ONEHOT: t39_leaf0 + t39_leaf1 + t39_leaf2 + t39_leaf3 + t39_leaf4
   + t39_leaf5 + t39_leaf6 + t39_leaf7 = 1
 T39_L0_f20_le: x_Gr_Liv_Area + 8692 t39_leaf0 <= 10000
 T39_L0_f5_le: const_Year_Built + 998059 t39_leaf0 <= 1e+06
 T39_L0_f4_le: const_Overall_Cond + 999993 t39_leaf0 <= 1e+06
 T39_L1_f20_le: x_Gr_Liv_Area + 8692 t39_leaf1 <= 10000
 T39_L1_f5_le: const_Year_Built + 998059 t39_leaf1 <= 1e+06
 T39_R1_f4_ge: const_Overall_Cond - 1.000007e+06 t39_leaf1 >= -1e+06
 T39_L2_f20_le: x_Gr_Liv_Area + 8692 t39_leaf2 <= 10000
 T39_R2_f5_ge: const_Year_Built - 1.001941e+06 t39_leaf2 >= -1e+06
 T39_L2_f21_le: const_Bsmt_Full_Bath + 999999 t39_leaf2 <= 1e+06
 T39_L3_f20_le: x_Gr_Liv_Area + 8692 t39_leaf3 <= 10000
 T39_R3_f5_ge: const_Year_Built - 1.001941e+06 t39_leaf3 >= -1e+06
 T39_R3_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t39_leaf3 >= -1e+06
 T39_R4_f20_ge: x_Gr_Liv_Area - 1308 t39_leaf4 >= 0
 T39_L4_f12_le: x_BsmtFin_SF_1 + 999144 t39_leaf4 <= 1e+06
 T39_L4_f4_le: const_Overall_Cond + 999995 t39_leaf4 <= 1e+06
 T39_R5_f20_ge: x_Gr_Liv_Area - 1308 t39_leaf5 >= 0
 T39_L5_f12_le: x_BsmtFin_SF_1 + 999144 t39_leaf5 <= 1e+06
 T39_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t39_leaf5 >= -1e+06
 T39_R6_f20_ge: x_Gr_Liv_Area - 1308 t39_leaf6 >= 0
 T39_R6_f12_ge: x_BsmtFin_SF_1 - 856 t39_leaf6 >= 0
 T39_L6_f3_le: Overall_Qual_calc + 999991 t39_leaf6 <= 1e+06
 T39_R7_f20_ge: x_Gr_Liv_Area - 1308 t39_leaf7 >= 0
 T39_R7_f12_ge: x_BsmtFin_SF_1 - 856 t39_leaf7 >= 0
 T39_R7_f3_ge: Overall_Qual_calc - 1.000009e+06 t39_leaf7 >= -1e+06
 TREE_40_ONEHOT: t40_leaf0 + t40_leaf1 + t40_leaf2 + t40_leaf3 + t40_leaf4
   + t40_leaf5 + t40_leaf6 + t40_leaf7 = 1
 T40_L0_f6_le: const_Year_Remod_Add + 998015 t40_leaf0 <= 1e+06
 T40_L0_f35_le: x_Garage_Cond + 2 t40_leaf0 <= 4
 T40_L0_f15_le: const_Total_Bsmt_SF + 999142 t40_leaf0 <= 1e+06
 T40_L1_f6_le: const_Year_Remod_Add + 998015 t40_leaf1 <= 1e+06
 T40_L1_f35_le: x_Garage_Cond + 2 t40_leaf1 <= 4
 T40_R1_f15_ge: const_Total_Bsmt_SF - 1.000858e+06 t40_leaf1 >= -1e+06
 T40_L2_f6_le: const_Year_Remod_Add + 998015 t40_leaf2 <= 1e+06
 T40_R2_f35_ge: x_Garage_Cond - 3 t40_leaf2 >= -1
 T40_L2_f0_le: const_Lot_Frontage + 999934 t40_leaf2 <= 1e+06
 T40_L3_f6_le: const_Year_Remod_Add + 998015 t40_leaf3 <= 1e+06
 T40_R3_f35_ge: x_Garage_Cond - 3 t40_leaf3 >= -1
 T40_R3_f0_ge: const_Lot_Frontage - 1.000066e+06 t40_leaf3 >= -1e+06
 T40_R4_f6_ge: const_Year_Remod_Add - 1.001985e+06 t40_leaf4 >= -1e+06
 T40_L4_f20_le: x_Gr_Liv_Area + 8362 t40_leaf4 <= 10000
 T40_L4_f15_le: const_Total_Bsmt_SF + 998945 t40_leaf4 <= 1e+06
 T40_R5_f6_ge: const_Year_Remod_Add - 1.001985e+06 t40_leaf5 >= -1e+06
 T40_L5_f20_le: x_Gr_Liv_Area + 8362 t40_leaf5 <= 10000
 T40_R5_f15_ge: const_Total_Bsmt_SF - 1.001055e+06 t40_leaf5 >= -1e+06
 T40_R6_f6_ge: const_Year_Remod_Add - 1.001985e+06 t40_leaf6 >= -1e+06
 T40_R6_f20_ge: x_Gr_Liv_Area - 1638 t40_leaf6 >= 0
 T40_L6_f18_le: const_2nd_Flr_SF + 998632 t40_leaf6 <= 1e+06
 T40_R7_f6_ge: const_Year_Remod_Add - 1.001985e+06 t40_leaf7 >= -1e+06
 T40_R7_f20_ge: x_Gr_Liv_Area - 1638 t40_leaf7 >= 0
 T40_R7_f18_ge: const_2nd_Flr_SF - 1.001368e+06 t40_leaf7 >= -1e+06
 TREE_41_ONEHOT: t41_leaf0 + t41_leaf1 + t41_leaf2 + t41_leaf3 + t41_leaf4
   + t41_leaf5 + t41_leaf6 + t41_leaf7 = 1
 T41_L0_f3_le: Overall_Qual_calc + 999993 t41_leaf0 <= 1e+06
 T41_L0_f46_le: const_MS_SubClass_30 + 999999 t41_leaf0 <= 1e+06
 T41_L0_f3_le: Overall_Qual_calc + 999995 t41_leaf0 <= 1e+06
 T41_L1_f3_le: Overall_Qual_calc + 999993 t41_leaf1 <= 1e+06
 T41_L1_f46_le: const_MS_SubClass_30 + 999999 t41_leaf1 <= 1e+06
 T41_R1_f3_ge: Overall_Qual_calc - 1.000005e+06 t41_leaf1 >= -1e+06
 T41_L2_f3_le: Overall_Qual_calc + 999993 t41_leaf2 <= 1e+06
 T41_R2_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t41_leaf2 >= -1e+06
 T41_L2_f12_le: x_BsmtFin_SF_1 + 999794 t41_leaf2 <= 1e+06
 T41_L3_f3_le: Overall_Qual_calc + 999993 t41_leaf3 <= 1e+06
 T41_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t41_leaf3 >= -1e+06
 T41_R3_f12_ge: x_BsmtFin_SF_1 - 206 t41_leaf3 >= 0
 T41_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t41_leaf4 >= -1e+06
 T41_L4_f3_le: Overall_Qual_calc + 999991 t41_leaf4 <= 1e+06
 T41_L4_f12_le: x_BsmtFin_SF_1 + 998903 t41_leaf4 <= 1e+06
 T41_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t41_leaf5 >= -1e+06
 T41_L5_f3_le: Overall_Qual_calc + 999991 t41_leaf5 <= 1e+06
 T41_R5_f12_ge: x_BsmtFin_SF_1 - 1097 t41_leaf5 >= 0
 T41_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t41_leaf6 >= -1e+06
 T41_R6_f3_ge: Overall_Qual_calc - 1.000009e+06 t41_leaf6 >= -1e+06
 T41_L6_f37_le: x_Open_Porch_SF + 99764 t41_leaf6 <= 100000
 T41_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t41_leaf7 >= -1e+06
 T41_R7_f3_ge: Overall_Qual_calc - 1.000009e+06 t41_leaf7 >= -1e+06
 T41_R7_f37_ge: x_Open_Porch_SF - 236 t41_leaf7 >= 0
 TREE_42_ONEHOT: t42_leaf0 + t42_leaf1 + t42_leaf2 + t42_leaf3 + t42_leaf4
   + t42_leaf5 + t42_leaf6 + t42_leaf7 = 1
 T42_L0_f15_le: const_Total_Bsmt_SF + 998800 t42_leaf0 <= 1e+06
 T42_L0_f35_le: x_Garage_Cond + 2 t42_leaf0 <= 4
 T42_L0_f9_le: x_Exter_Cond + 2 t42_leaf0 <= 4
 T42_L1_f15_le: const_Total_Bsmt_SF + 998800 t42_leaf1 <= 1e+06
 T42_L1_f35_le: x_Garage_Cond + 2 t42_leaf1 <= 4
 T42_R1_f9_ge: x_Exter_Cond - 3 t42_leaf1 >= -1
 T42_L2_f15_le: const_Total_Bsmt_SF + 998800 t42_leaf2 <= 1e+06
 T42_R2_f35_ge: x_Garage_Cond - 3 t42_leaf2 >= -1
 T42_L2_f20_le: x_Gr_Liv_Area + 8234 t42_leaf2 <= 10000
 T42_L3_f15_le: const_Total_Bsmt_SF + 998800 t42_leaf3 <= 1e+06
 T42_R3_f35_ge: x_Garage_Cond - 3 t42_leaf3 >= -1
 T42_R3_f20_ge: x_Gr_Liv_Area - 1766 t42_leaf3 >= 0
 T42_R4_f15_ge: const_Total_Bsmt_SF - 1.0012e+06 t42_leaf4 >= -1e+06
 T42_L4_f20_le: x_Gr_Liv_Area + 7948 t42_leaf4 <= 10000
 T42_L4_f12_le: x_BsmtFin_SF_1 + 999676 t42_leaf4 <= 1e+06
 T42_R5_f15_ge: const_Total_Bsmt_SF - 1.0012e+06 t42_leaf5 >= -1e+06
 T42_L5_f20_le: x_Gr_Liv_Area + 7948 t42_leaf5 <= 10000
 T42_R5_f12_ge: x_BsmtFin_SF_1 - 324 t42_leaf5 >= 0
 T42_R6_f15_ge: const_Total_Bsmt_SF - 1.0012e+06 t42_leaf6 >= -1e+06
 T42_R6_f20_ge: x_Gr_Liv_Area - 2052 t42_leaf6 >= 0
 T42_L6_f37_le: x_Open_Porch_SF + 99770 t42_leaf6 <= 100000
 T42_R7_f15_ge: const_Total_Bsmt_SF - 1.0012e+06 t42_leaf7 >= -1e+06
 T42_R7_f20_ge: x_Gr_Liv_Area - 2052 t42_leaf7 >= 0
 T42_R7_f37_ge: x_Open_Porch_SF - 230 t42_leaf7 >= 0
 TREE_43_ONEHOT: t43_leaf0 + t43_leaf1 + t43_leaf2 + t43_leaf3 + t43_leaf4
   + t43_leaf5 + t43_leaf6 + t43_leaf7 = 1
 T43_L0_f31_le: const_Garage_Yr_Blt + 998018 t43_leaf0 <= 1e+06
 T43_L0_f12_le: x_BsmtFin_SF_1 + 999777 t43_leaf0 <= 1e+06
 T43_L0_f4_le: const_Overall_Cond + 999995 t43_leaf0 <= 1e+06
 T43_L1_f31_le: const_Garage_Yr_Blt + 998018 t43_leaf1 <= 1e+06
 T43_L1_f12_le: x_BsmtFin_SF_1 + 999777 t43_leaf1 <= 1e+06
 T43_R1_f4_ge: const_Overall_Cond - 1.000005e+06 t43_leaf1 >= -1e+06
 T43_L2_f31_le: const_Garage_Yr_Blt + 998018 t43_leaf2 <= 1e+06
 T43_R2_f12_ge: x_BsmtFin_SF_1 - 223 t43_leaf2 >= 0
 T43_L2_f4_le: const_Overall_Cond + 999994 t43_leaf2 <= 1e+06
 T43_L3_f31_le: const_Garage_Yr_Blt + 998018 t43_leaf3 <= 1e+06
 T43_R3_f12_ge: x_BsmtFin_SF_1 - 223 t43_leaf3 >= 0
 T43_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t43_leaf3 >= -1e+06
 T43_R4_f31_ge: const_Garage_Yr_Blt - 1.001982e+06 t43_leaf4 >= -1e+06
 T43_L4_f3_le: Overall_Qual_calc + 999992 t43_leaf4 <= 1e+06
 T43_L4_f37_le: x_Open_Porch_SF + 99958 t43_leaf4 <= 100000
 T43_R5_f31_ge: const_Garage_Yr_Blt - 1.001982e+06 t43_leaf5 >= -1e+06
 T43_L5_f3_le: Overall_Qual_calc + 999992 t43_leaf5 <= 1e+06
 T43_R5_f37_ge: x_Open_Porch_SF - 42 t43_leaf5 >= 0
 T43_R6_f31_ge: const_Garage_Yr_Blt - 1.001982e+06 t43_leaf6 >= -1e+06
 T43_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t43_leaf6 >= -1e+06
 T43_L6_f37_le: x_Open_Porch_SF + 99726 t43_leaf6 <= 100000
 T43_R7_f31_ge: const_Garage_Yr_Blt - 1.001982e+06 t43_leaf7 >= -1e+06
 T43_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t43_leaf7 >= -1e+06
 T43_R7_f37_ge: x_Open_Porch_SF - 274 t43_leaf7 >= 0
 TREE_44_ONEHOT: t44_leaf0 + t44_leaf1 + t44_leaf2 + t44_leaf3 + t44_leaf4
   + t44_leaf5 + t44_leaf6 + t44_leaf7 = 1
 T44_L0_f10_le: const_Bsmt_Qual + 999997 t44_leaf0 <= 1e+06
 T44_L0_f15_le: const_Total_Bsmt_SF + 999370 t44_leaf0 <= 1e+06
 T44_L0_f20_le: x_Gr_Liv_Area + 8789 t44_leaf0 <= 10000
 T44_L1_f10_le: const_Bsmt_Qual + 999997 t44_leaf1 <= 1e+06
 T44_L1_f15_le: const_Total_Bsmt_SF + 999370 t44_leaf1 <= 1e+06
 T44_R1_f20_ge: x_Gr_Liv_Area - 1211 t44_leaf1 >= 0
 T44_L2_f10_le: const_Bsmt_Qual + 999997 t44_leaf2 <= 1e+06
 T44_R2_f15_ge: const_Total_Bsmt_SF - 1.00063e+06 t44_leaf2 >= -1e+06
 T44_L2_f31_le: const_Garage_Yr_Blt + 998079 t44_leaf2 <= 1e+06
 T44_L3_f10_le: const_Bsmt_Qual + 999997 t44_leaf3 <= 1e+06
 T44_R3_f15_ge: const_Total_Bsmt_SF - 1.00063e+06 t44_leaf3 >= -1e+06
 T44_R3_f31_ge: const_Garage_Yr_Blt - 1.001921e+06 t44_leaf3 >= -1e+06
 T44_R4_f10_ge: const_Bsmt_Qual - 1.000003e+06 t44_leaf4 >= -1e+06
 T44_L4_f17_le: x_1st_Flr_SF + 3348 t44_leaf4 <= 5000
 T44_L4_f15_le: const_Total_Bsmt_SF + 999074 t44_leaf4 <= 1e+06
 T44_R5_f10_ge: const_Bsmt_Qual - 1.000003e+06 t44_leaf5 >= -1e+06
 T44_L5_f17_le: x_1st_Flr_SF + 3348 t44_leaf5 <= 5000
 T44_R5_f15_ge: const_Total_Bsmt_SF - 1.000926e+06 t44_leaf5 >= -1e+06
 T44_R6_f10_ge: const_Bsmt_Qual - 1.000003e+06 t44_leaf6 >= -1e+06
 T44_R6_f17_ge: x_1st_Flr_SF - 1652 t44_leaf6 >= 0
 T44_L6_f20_le: x_Gr_Liv_Area + 7972 t44_leaf6 <= 10000
 T44_R7_f10_ge: const_Bsmt_Qual - 1.000003e+06 t44_leaf7 >= -1e+06
 T44_R7_f17_ge: x_1st_Flr_SF - 1652 t44_leaf7 >= 0
 T44_R7_f20_ge: x_Gr_Liv_Area - 2028 t44_leaf7 >= 0
 TREE_45_ONEHOT: t45_leaf0 + t45_leaf1 + t45_leaf2 + t45_leaf3 + t45_leaf4
   + t45_leaf5 + t45_leaf6 + t45_leaf7 = 1
 T45_L0_f3_le: Overall_Qual_calc + 999994 t45_leaf0 <= 1e+06
 T45_L0_f12_le: x_BsmtFin_SF_1 + 999925 t45_leaf0 <= 1e+06
 T45_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t45_leaf0 <= 1e+06
 T45_L1_f3_le: Overall_Qual_calc + 999994 t45_leaf1 <= 1e+06
 T45_L1_f12_le: x_BsmtFin_SF_1 + 999925 t45_leaf1 <= 1e+06
 T45_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t45_leaf1
   >= -1e+06
 T45_L2_f3_le: Overall_Qual_calc + 999994 t45_leaf2 <= 1e+06
 T45_R2_f12_ge: x_BsmtFin_SF_1 - 75 t45_leaf2 >= 0
 T45_L2_f15_le: const_Total_Bsmt_SF + 998644 t45_leaf2 <= 1e+06
 T45_L3_f3_le: Overall_Qual_calc + 999994 t45_leaf3 <= 1e+06
 T45_R3_f12_ge: x_BsmtFin_SF_1 - 75 t45_leaf3 >= 0
 T45_R3_f15_ge: const_Total_Bsmt_SF - 1.001356e+06 t45_leaf3 >= -1e+06
 T45_R4_f3_ge: Overall_Qual_calc - 1.000006e+06 t45_leaf4 >= -1e+06
 T45_L4_f12_le: x_BsmtFin_SF_1 + 999273 t45_leaf4 <= 1e+06
 T45_L4_f32_le: const_Garage_Cars + 999998 t45_leaf4 <= 1e+06
 T45_R5_f3_ge: Overall_Qual_calc - 1.000006e+06 t45_leaf5 >= -1e+06
 T45_L5_f12_le: x_BsmtFin_SF_1 + 999273 t45_leaf5 <= 1e+06
 T45_R5_f32_ge: const_Garage_Cars - 1.000002e+06 t45_leaf5 >= -1e+06
 T45_R6_f3_ge: Overall_Qual_calc - 1.000006e+06 t45_leaf6 >= -1e+06
 T45_R6_f12_ge: x_BsmtFin_SF_1 - 727 t45_leaf6 >= 0
 T45_L6_f3_le: Overall_Qual_calc + 999992 t45_leaf6 <= 1e+06
 T45_R7_f3_ge: Overall_Qual_calc - 1.000006e+06 t45_leaf7 >= -1e+06
 T45_R7_f12_ge: x_BsmtFin_SF_1 - 727 t45_leaf7 >= 0
 T45_R7_f3_ge: Overall_Qual_calc - 1.000008e+06 t45_leaf7 >= -1e+06
 TREE_46_ONEHOT: t46_leaf0 + t46_leaf1 + t46_leaf2 + t46_leaf3 + t46_leaf4
   + t46_leaf5 + t46_leaf6 + t46_leaf7 = 1
 T46_L0_f17_le: x_1st_Flr_SF + 3845 t46_leaf0 <= 5000
 T46_L0_f35_le: x_Garage_Cond + 2 t46_leaf0 <= 4
 T46_L0_f46_le: const_MS_SubClass_30 + 999999 t46_leaf0 <= 1e+06
 T46_L1_f17_le: x_1st_Flr_SF + 3845 t46_leaf1 <= 5000
 T46_L1_f35_le: x_Garage_Cond + 2 t46_leaf1 <= 4
 T46_R1_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t46_leaf1 >= -1e+06
 T46_L2_f17_le: x_1st_Flr_SF + 3845 t46_leaf2 <= 5000
 T46_R2_f35_ge: x_Garage_Cond - 3 t46_leaf2 >= -1
 T46_L2_f18_le: const_2nd_Flr_SF + 999324 t46_leaf2 <= 1e+06
 T46_L3_f17_le: x_1st_Flr_SF + 3845 t46_leaf3 <= 5000
 T46_R3_f35_ge: x_Garage_Cond - 3 t46_leaf3 >= -1
 T46_R3_f18_ge: const_2nd_Flr_SF - 1.000676e+06 t46_leaf3 >= -1e+06
 T46_R4_f17_ge: x_1st_Flr_SF - 1155 t46_leaf4 >= 0
 T46_L4_f18_le: const_2nd_Flr_SF + 998832 t46_leaf4 <= 1e+06
 T46_L4_f6_le: const_Year_Remod_Add + 998007 t46_leaf4 <= 1e+06
 T46_R5_f17_ge: x_1st_Flr_SF - 1155 t46_leaf5 >= 0
 T46_L5_f18_le: const_2nd_Flr_SF + 998832 t46_leaf5 <= 1e+06
 T46_R5_f6_ge: const_Year_Remod_Add - 1.001993e+06 t46_leaf5 >= -1e+06
 T46_R6_f17_ge: x_1st_Flr_SF - 1155 t46_leaf6 >= 0
 T46_R6_f18_ge: const_2nd_Flr_SF - 1.001168e+06 t46_leaf6 >= -1e+06
 T46_L6_f26_le: x_Kitchen_AbvGr + t46_leaf6 <= 3
 T46_R7_f17_ge: x_1st_Flr_SF - 1155 t46_leaf7 >= 0
 T46_R7_f18_ge: const_2nd_Flr_SF - 1.001168e+06 t46_leaf7 >= -1e+06
 T46_R7_f26_ge: x_Kitchen_AbvGr - 2 t46_leaf7 >= 0
 TREE_47_ONEHOT: t47_leaf0 + t47_leaf1 + t47_leaf2 + t47_leaf3 + t47_leaf4
   + t47_leaf5 + t47_leaf6 + t47_leaf7 = 1
 T47_L0_f6_le: const_Year_Remod_Add + 998019 t47_leaf0 <= 1e+06
 T47_L0_f4_le: const_Overall_Cond + 999995 t47_leaf0 <= 1e+06
 T47_L0_f6_le: const_Year_Remod_Add + 998041 t47_leaf0 <= 1e+06
 T47_L1_f6_le: const_Year_Remod_Add + 998019 t47_leaf1 <= 1e+06
 T47_L1_f4_le: const_Overall_Cond + 999995 t47_leaf1 <= 1e+06
 T47_R1_f6_ge: const_Year_Remod_Add - 1.001959e+06 t47_leaf1 >= -1e+06
 T47_L2_f6_le: const_Year_Remod_Add + 998019 t47_leaf2 <= 1e+06
 T47_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t47_leaf2 >= -1e+06
 T47_L2_f0_le: const_Lot_Frontage + 999931 t47_leaf2 <= 1e+06
 T47_L3_f6_le: const_Year_Remod_Add + 998019 t47_leaf3 <= 1e+06
 T47_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t47_leaf3 >= -1e+06
 T47_R3_f0_ge: const_Lot_Frontage - 1.000069e+06 t47_leaf3 >= -1e+06
 T47_R4_f6_ge: const_Year_Remod_Add - 1.001981e+06 t47_leaf4 >= -1e+06
 T47_L4_f30_le: x_Fireplace_Qu + 2 t47_leaf4 <= 4
 T47_L4_f0_le: const_Lot_Frontage + 999939 t47_leaf4 <= 1e+06
 T47_R5_f6_ge: const_Year_Remod_Add - 1.001981e+06 t47_leaf5 >= -1e+06
 T47_L5_f30_le: x_Fireplace_Qu + 2 t47_leaf5 <= 4
 T47_R5_f0_ge: const_Lot_Frontage - 1.000061e+06 t47_leaf5 >= -1e+06
 T47_R6_f6_ge: const_Year_Remod_Add - 1.001981e+06 t47_leaf6 >= -1e+06
 T47_R6_f30_ge: x_Fireplace_Qu - 3 t47_leaf6 >= -1
 T47_L6_f3_le: Overall_Qual_calc + 999991 t47_leaf6 <= 1e+06
 T47_R7_f6_ge: const_Year_Remod_Add - 1.001981e+06 t47_leaf7 >= -1e+06
 T47_R7_f30_ge: x_Fireplace_Qu - 3 t47_leaf7 >= -1
 T47_R7_f3_ge: Overall_Qual_calc - 1.000009e+06 t47_leaf7 >= -1e+06
 TREE_48_ONEHOT: t48_leaf0 + t48_leaf1 + t48_leaf2 + t48_leaf3 + t48_leaf4
   + t48_leaf5 + t48_leaf6 + t48_leaf7 = 1
 T48_L0_f20_le: x_Gr_Liv_Area + 8288 t48_leaf0 <= 10000
 T48_L0_f257_le: x_paved_drive_is_N <= 1
 T48_L0_f33_le: x_Garage_Area + 99616 t48_leaf0 <= 100000
 T48_L1_f20_le: x_Gr_Liv_Area + 8288 t48_leaf1 <= 10000
 T48_L1_f257_le: x_paved_drive_is_N <= 1
 T48_R1_f33_ge: x_Garage_Area - 384 t48_leaf1 >= 0
 T48_L2_f20_le: x_Gr_Liv_Area + 8288 t48_leaf2 <= 10000
 T48_R2_f257_ge: x_paved_drive_is_N - t48_leaf2 >= 0
 T48_L2_f297_le: const_Sale_Condition_Normal + 999999 t48_leaf2 <= 1e+06
 T48_L3_f20_le: x_Gr_Liv_Area + 8288 t48_leaf3 <= 10000
 T48_R3_f257_ge: x_paved_drive_is_N - t48_leaf3 >= 0
 T48_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t48_leaf3
   >= -1e+06
 T48_R4_f20_ge: x_Gr_Liv_Area - 1712 t48_leaf4 >= 0
 T48_L4_f213_le: x_b1_is_GLQ <= 1
 T48_L4_f44_le: const_Yr_Sold + 997992 t48_leaf4 <= 1e+06
 T48_R5_f20_ge: x_Gr_Liv_Area - 1712 t48_leaf5 >= 0
 T48_L5_f213_le: x_b1_is_GLQ <= 1
 T48_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t48_leaf5 >= -1e+06
 T48_R6_f20_ge: x_Gr_Liv_Area - 1712 t48_leaf6 >= 0
 T48_R6_f213_ge: x_b1_is_GLQ - t48_leaf6 >= 0
 T48_L6_f15_le: const_Total_Bsmt_SF + 998150 t48_leaf6 <= 1e+06
 T48_R7_f20_ge: x_Gr_Liv_Area - 1712 t48_leaf7 >= 0
 T48_R7_f213_ge: x_b1_is_GLQ - t48_leaf7 >= 0
 T48_R7_f15_ge: const_Total_Bsmt_SF - 1.00185e+06 t48_leaf7 >= -1e+06
 TREE_49_ONEHOT: t49_leaf0 + t49_leaf1 + t49_leaf2 + t49_leaf3 + t49_leaf4
   + t49_leaf5 + t49_leaf6 + t49_leaf7 = 1
 T49_L0_f20_le: x_Gr_Liv_Area + 8506 t49_leaf0 <= 10000
 T49_L0_f259_le: x_paved_drive_is_Y <= 1
 T49_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t49_leaf0 <= 1e+06
 T49_L1_f20_le: x_Gr_Liv_Area + 8506 t49_leaf1 <= 10000
 T49_L1_f259_le: x_paved_drive_is_Y <= 1
 T49_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t49_leaf1
   >= -1e+06
 T49_L2_f20_le: x_Gr_Liv_Area + 8506 t49_leaf2 <= 10000
 T49_R2_f259_ge: x_paved_drive_is_Y - t49_leaf2 >= 0
 T49_L2_f17_le: x_1st_Flr_SF + 3936 t49_leaf2 <= 5000
 T49_L3_f20_le: x_Gr_Liv_Area + 8506 t49_leaf3 <= 10000
 T49_R3_f259_ge: x_paved_drive_is_Y - t49_leaf3 >= 0
 T49_R3_f17_ge: x_1st_Flr_SF - 1064 t49_leaf3 >= 0
 T49_R4_f20_ge: x_Gr_Liv_Area - 1494 t49_leaf4 >= 0
 T49_L4_f109_le: const_Neighborhood_OldTown + 999999 t49_leaf4 <= 1e+06
 T49_L4_f44_le: const_Yr_Sold + 997992 t49_leaf4 <= 1e+06
 T49_R5_f20_ge: x_Gr_Liv_Area - 1494 t49_leaf5 >= 0
 T49_L5_f109_le: const_Neighborhood_OldTown + 999999 t49_leaf5 <= 1e+06
 T49_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t49_leaf5 >= -1e+06
 T49_R6_f20_ge: x_Gr_Liv_Area - 1494 t49_leaf6 >= 0
 T49_R6_f109_ge: const_Neighborhood_OldTown - 1.000001e+06 t49_leaf6
   >= -1e+06
 T49_L6_f293_le: const_Sale_Condition_Abnorml + 999999 t49_leaf6 <= 1e+06
 T49_R7_f20_ge: x_Gr_Liv_Area - 1494 t49_leaf7 >= 0
 T49_R7_f109_ge: const_Neighborhood_OldTown - 1.000001e+06 t49_leaf7
   >= -1e+06
 T49_R7_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t49_leaf7
   >= -1e+06
 TREE_50_ONEHOT: t50_leaf0 + t50_leaf1 + t50_leaf2 + t50_leaf3 + t50_leaf4
   + t50_leaf5 + t50_leaf6 = 1
 T50_L0_f20_le: x_Gr_Liv_Area + 8817 t50_leaf0 <= 10000
 T50_L0_f3_le: Overall_Qual_calc + 999996 t50_leaf0 <= 1e+06
 T50_L1_f20_le: x_Gr_Liv_Area + 8817 t50_leaf1 <= 10000
 T50_R1_f3_ge: Overall_Qual_calc - 1.000004e+06 t50_leaf1 >= -1e+06
 T50_L1_f66_le: const_MS_Zoning_RL + 999999 t50_leaf1 <= 1e+06
 T50_L2_f20_le: x_Gr_Liv_Area + 8817 t50_leaf2 <= 10000
 T50_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t50_leaf2 >= -1e+06
 T50_R2_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t50_leaf2 >= -1e+06
 T50_R3_f20_ge: x_Gr_Liv_Area - 1183 t50_leaf3 >= 0
 T50_L3_f1_le: const_Lot_Area + 988572 t50_leaf3 <= 1e+06
 T50_L3_f5_le: const_Year_Built + 998014 t50_leaf3 <= 1e+06
 T50_R4_f20_ge: x_Gr_Liv_Area - 1183 t50_leaf4 >= 0
 T50_L4_f1_le: const_Lot_Area + 988572 t50_leaf4 <= 1e+06
 T50_R4_f5_ge: const_Year_Built - 1.001986e+06 t50_leaf4 >= -1e+06
 T50_R5_f20_ge: x_Gr_Liv_Area - 1183 t50_leaf5 >= 0
 T50_R5_f1_ge: const_Lot_Area - 1.011428e+06 t50_leaf5 >= -1e+06
 T50_L5_f20_le: x_Gr_Liv_Area + 8266 t50_leaf5 <= 10000
 T50_R6_f20_ge: x_Gr_Liv_Area - 1183 t50_leaf6 >= 0
 T50_R6_f1_ge: const_Lot_Area - 1.011428e+06 t50_leaf6 >= -1e+06
 T50_R6_f20_ge: x_Gr_Liv_Area - 1734 t50_leaf6 >= 0
 TREE_51_ONEHOT: t51_leaf0 + t51_leaf1 + t51_leaf2 + t51_leaf3 + t51_leaf4
   + t51_leaf5 + t51_leaf6 = 1
 T51_L0_f3_le: Overall_Qual_calc + 999994 t51_leaf0 <= 1e+06
 T51_L0_f20_le: x_Gr_Liv_Area + 9232 t51_leaf0 <= 10000
 T51_L1_f3_le: Overall_Qual_calc + 999994 t51_leaf1 <= 1e+06
 T51_R1_f20_ge: x_Gr_Liv_Area - 768 t51_leaf1 >= 0
 T51_L1_f12_le: x_BsmtFin_SF_1 + 999973 t51_leaf1 <= 1e+06
 T51_L2_f3_le: Overall_Qual_calc + 999994 t51_leaf2 <= 1e+06
 T51_R2_f20_ge: x_Gr_Liv_Area - 768 t51_leaf2 >= 0
 T51_R2_f12_ge: x_BsmtFin_SF_1 - 27 t51_leaf2 >= 0
 T51_R3_f3_ge: Overall_Qual_calc - 1.000006e+06 t51_leaf3 >= -1e+06
 T51_L3_f20_le: x_Gr_Liv_Area + 7673 t51_leaf3 <= 10000
 T51_L3_f12_le: x_BsmtFin_SF_1 + 999295 t51_leaf3 <= 1e+06
 T51_R4_f3_ge: Overall_Qual_calc - 1.000006e+06 t51_leaf4 >= -1e+06
 T51_L4_f20_le: x_Gr_Liv_Area + 7673 t51_leaf4 <= 10000
 T51_R4_f12_ge: x_BsmtFin_SF_1 - 705 t51_leaf4 >= 0
 T51_R5_f3_ge: Overall_Qual_calc - 1.000006e+06 t51_leaf5 >= -1e+06
 T51_R5_f20_ge: x_Gr_Liv_Area - 2327 t51_leaf5 >= 0
 T51_L5_f37_le: x_Open_Porch_SF + 99726 t51_leaf5 <= 100000
 T51_R6_f3_ge: Overall_Qual_calc - 1.000006e+06 t51_leaf6 >= -1e+06
 T51_R6_f20_ge: x_Gr_Liv_Area - 2327 t51_leaf6 >= 0
 T51_R6_f37_ge: x_Open_Porch_SF - 274 t51_leaf6 >= 0
 TREE_52_ONEHOT: t52_leaf0 + t52_leaf1 + t52_leaf2 + t52_leaf3 + t52_leaf4
   + t52_leaf5 + t52_leaf6 + t52_leaf7 = 1
 T52_L0_f15_le: const_Total_Bsmt_SF + 998581 t52_leaf0 <= 1e+06
 T52_L0_f4_le: const_Overall_Cond + 999995 t52_leaf0 <= 1e+06
 T52_L0_f4_le: const_Overall_Cond + 999996 t52_leaf0 <= 1e+06
 T52_L1_f15_le: const_Total_Bsmt_SF + 998581 t52_leaf1 <= 1e+06
 T52_L1_f4_le: const_Overall_Cond + 999995 t52_leaf1 <= 1e+06
 T52_R1_f4_ge: const_Overall_Cond - 1.000004e+06 t52_leaf1 >= -1e+06
 T52_L2_f15_le: const_Total_Bsmt_SF + 998581 t52_leaf2 <= 1e+06
 T52_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t52_leaf2 >= -1e+06
 T52_L2_f20_le: x_Gr_Liv_Area + 8280 t52_leaf2 <= 10000
 T52_L3_f15_le: const_Total_Bsmt_SF + 998581 t52_leaf3 <= 1e+06
 T52_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t52_leaf3 >= -1e+06
 T52_R3_f20_ge: x_Gr_Liv_Area - 1720 t52_leaf3 >= 0
 T52_R4_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t52_leaf4 >= -1e+06
 T52_L4_f20_le: x_Gr_Liv_Area + 7972 t52_leaf4 <= 10000
 T52_L4_f21_le: const_Bsmt_Full_Bath + 999999 t52_leaf4 <= 1e+06
 T52_R5_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t52_leaf5 >= -1e+06
 T52_L5_f20_le: x_Gr_Liv_Area + 7972 t52_leaf5 <= 10000
 T52_R5_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t52_leaf5 >= -1e+06
 T52_R6_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t52_leaf6 >= -1e+06
 T52_R6_f20_ge: x_Gr_Liv_Area - 2028 t52_leaf6 >= 0
 T52_L6_f1_le: const_Lot_Area + 986312 t52_leaf6 <= 1e+06
 T52_R7_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t52_leaf7 >= -1e+06
 T52_R7_f20_ge: x_Gr_Liv_Area - 2028 t52_leaf7 >= 0
 T52_R7_f1_ge: const_Lot_Area - 1.013688e+06 t52_leaf7 >= -1e+06
 TREE_53_ONEHOT: t53_leaf0 + t53_leaf1 + t53_leaf2 + t53_leaf3 + t53_leaf4
   + t53_leaf5 + t53_leaf6 = 1
 T53_L0_f30_le: x_Fireplace_Qu + 3 t53_leaf0 <= 4
 T53_L0_f9_le: x_Exter_Cond + 2 t53_leaf0 <= 4
 T53_L0_f25_le: x_Bedroom_AbvGr + 2 t53_leaf0 <= 6
 T53_L1_f30_le: x_Fireplace_Qu + 3 t53_leaf1 <= 4
 T53_L1_f9_le: x_Exter_Cond + 2 t53_leaf1 <= 4
 T53_R1_f25_ge: x_Bedroom_AbvGr - 4 t53_leaf1 >= 0
 T53_L2_f30_le: x_Fireplace_Qu + 3 t53_leaf2 <= 4
 T53_R2_f9_ge: x_Exter_Cond - 3 t53_leaf2 >= -1
 T53_L2_f247_le: const_Garage_Type_Attchd + 999999 t53_leaf2 <= 1e+06
 T53_L3_f30_le: x_Fireplace_Qu + 3 t53_leaf3 <= 4
 T53_R3_f9_ge: x_Exter_Cond - 3 t53_leaf3 >= -1
 T53_R3_f247_ge: const_Garage_Type_Attchd - 1.000001e+06 t53_leaf3
   >= -1e+06
 T53_R4_f30_ge: x_Fireplace_Qu - 2 t53_leaf4 >= -1
 T53_L4_f4_le: const_Overall_Cond + 999996 t53_leaf4 <= 1e+06
 T53_R5_f30_ge: x_Fireplace_Qu - 2 t53_leaf5 >= -1
 T53_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t53_leaf5 >= -1e+06
 T53_L5_f3_le: Overall_Qual_calc + 999991 t53_leaf5 <= 1e+06
 T53_R6_f30_ge: x_Fireplace_Qu - 2 t53_leaf6 >= -1
 T53_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t53_leaf6 >= -1e+06
 T53_R6_f3_ge: Overall_Qual_calc - 1.000009e+06 t53_leaf6 >= -1e+06
 TREE_54_ONEHOT: t54_leaf0 + t54_leaf1 + t54_leaf2 + t54_leaf3 + t54_leaf4
   + t54_leaf5 + t54_leaf6 = 1
 T54_L0_f4_le: const_Overall_Cond + 999995 t54_leaf0 <= 1e+06
 T54_L0_f4_le: const_Overall_Cond + 999996 t54_leaf0 <= 1e+06
 T54_L1_f4_le: const_Overall_Cond + 999995 t54_leaf1 <= 1e+06
 T54_R1_f4_ge: const_Overall_Cond - 1.000004e+06 t54_leaf1 >= -1e+06
 T54_L1_f44_le: const_Yr_Sold + 997993 t54_leaf1 <= 1e+06
 T54_L2_f4_le: const_Overall_Cond + 999995 t54_leaf2 <= 1e+06
 T54_R2_f4_ge: const_Overall_Cond - 1.000004e+06 t54_leaf2 >= -1e+06
 T54_R2_f44_ge: const_Yr_Sold - 1.002007e+06 t54_leaf2 >= -1e+06
 T54_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t54_leaf3 >= -1e+06
 T54_L3_f15_le: const_Total_Bsmt_SF + 999352 t54_leaf3 <= 1e+06
 T54_L3_f9_le: x_Exter_Cond + 2 t54_leaf3 <= 4
 T54_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t54_leaf4 >= -1e+06
 T54_L4_f15_le: const_Total_Bsmt_SF + 999352 t54_leaf4 <= 1e+06
 T54_R4_f9_ge: x_Exter_Cond - 3 t54_leaf4 >= -1
 T54_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t54_leaf5 >= -1e+06
 T54_R5_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t54_leaf5 >= -1e+06
 T54_L5_f27_le: x_Kitchen_Qual <= 4
 T54_R6_f4_ge: const_Overall_Cond - 1.000005e+06 t54_leaf6 >= -1e+06
 T54_R6_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t54_leaf6 >= -1e+06
 T54_R6_f27_ge: x_Kitchen_Qual - 5 t54_leaf6 >= -1
 TREE_55_ONEHOT: t55_leaf0 + t55_leaf1 + t55_leaf2 + t55_leaf3 + t55_leaf4
   + t55_leaf5 + t55_leaf6 + t55_leaf7 = 1
 T55_L0_f29_le: const_Fireplaces + 999999 t55_leaf0 <= 1e+06
 T55_L0_f9_le: x_Exter_Cond + 2 t55_leaf0 <= 4
 T55_L0_f297_le: const_Sale_Condition_Normal + 999999 t55_leaf0 <= 1e+06
 T55_L1_f29_le: const_Fireplaces + 999999 t55_leaf1 <= 1e+06
 T55_L1_f9_le: x_Exter_Cond + 2 t55_leaf1 <= 4
 T55_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t55_leaf1
   >= -1e+06
 T55_L2_f29_le: const_Fireplaces + 999999 t55_leaf2 <= 1e+06
 T55_R2_f9_ge: x_Exter_Cond - 3 t55_leaf2 >= -1
 T55_L2_f33_le: x_Garage_Area + 99730 t55_leaf2 <= 100000
 T55_L3_f29_le: const_Fireplaces + 999999 t55_leaf3 <= 1e+06
 T55_R3_f9_ge: x_Exter_Cond - 3 t55_leaf3 >= -1
 T55_R3_f33_ge: x_Garage_Area - 270 t55_leaf3 >= 0
 T55_R4_f29_ge: const_Fireplaces - 1.000001e+06 t55_leaf4 >= -1e+06
 T55_L4_f12_le: x_BsmtFin_SF_1 + 998903 t55_leaf4 <= 1e+06
 T55_L4_f18_le: const_2nd_Flr_SF + 999067 t55_leaf4 <= 1e+06
 T55_R5_f29_ge: const_Fireplaces - 1.000001e+06 t55_leaf5 >= -1e+06
 T55_L5_f12_le: x_BsmtFin_SF_1 + 998903 t55_leaf5 <= 1e+06
 T55_R5_f18_ge: const_2nd_Flr_SF - 1.000933e+06 t55_leaf5 >= -1e+06
 T55_R6_f29_ge: const_Fireplaces - 1.000001e+06 t55_leaf6 >= -1e+06
 T55_R6_f12_ge: x_BsmtFin_SF_1 - 1097 t55_leaf6 >= 0
 T55_L6_f36_le: x_Wood_Deck_SF + 99629 t55_leaf6 <= 100000
 T55_R7_f29_ge: const_Fireplaces - 1.000001e+06 t55_leaf7 >= -1e+06
 T55_R7_f12_ge: x_BsmtFin_SF_1 - 1097 t55_leaf7 >= 0
 T55_R7_f36_ge: x_Wood_Deck_SF - 371 t55_leaf7 >= 0
 TREE_56_ONEHOT: t56_leaf0 + t56_leaf1 + t56_leaf2 + t56_leaf3 + t56_leaf4
   + t56_leaf5 + t56_leaf6 + t56_leaf7 = 1
 T56_L0_f15_le: const_Total_Bsmt_SF + 998954 t56_leaf0 <= 1e+06
 T56_L0_f1_le: const_Lot_Area + 993558 t56_leaf0 <= 1e+06
 T56_L0_f12_le: x_BsmtFin_SF_1 + 999630 t56_leaf0 <= 1e+06
 T56_L1_f15_le: const_Total_Bsmt_SF + 998954 t56_leaf1 <= 1e+06
 T56_L1_f1_le: const_Lot_Area + 993558 t56_leaf1 <= 1e+06
 T56_R1_f12_ge: x_BsmtFin_SF_1 - 370 t56_leaf1 >= 0
 T56_L2_f15_le: const_Total_Bsmt_SF + 998954 t56_leaf2 <= 1e+06
 T56_R2_f1_ge: const_Lot_Area - 1.006442e+06 t56_leaf2 >= -1e+06
 T56_L2_f24_le: x_Half_Bath + 4 t56_leaf2 <= 5
 T56_L3_f15_le: const_Total_Bsmt_SF + 998954 t56_leaf3 <= 1e+06
 T56_R3_f1_ge: const_Lot_Area - 1.006442e+06 t56_leaf3 >= -1e+06
 T56_R3_f24_ge: x_Half_Bath - t56_leaf3 >= 0
 T56_R4_f15_ge: const_Total_Bsmt_SF - 1.001046e+06 t56_leaf4 >= -1e+06
 T56_L4_f20_le: x_Gr_Liv_Area + 7763 t56_leaf4 <= 10000
 T56_L4_f12_le: x_BsmtFin_SF_1 + 998903 t56_leaf4 <= 1e+06
 T56_R5_f15_ge: const_Total_Bsmt_SF - 1.001046e+06 t56_leaf5 >= -1e+06
 T56_L5_f20_le: x_Gr_Liv_Area + 7763 t56_leaf5 <= 10000
 T56_R5_f12_ge: x_BsmtFin_SF_1 - 1097 t56_leaf5 >= 0
 T56_R6_f15_ge: const_Total_Bsmt_SF - 1.001046e+06 t56_leaf6 >= -1e+06
 T56_R6_f20_ge: x_Gr_Liv_Area - 2237 t56_leaf6 >= 0
 T56_L6_f43_le: const_Misc_Val + 998800 t56_leaf6 <= 1e+06
 T56_R7_f15_ge: const_Total_Bsmt_SF - 1.001046e+06 t56_leaf7 >= -1e+06
 T56_R7_f20_ge: x_Gr_Liv_Area - 2237 t56_leaf7 >= 0
 T56_R7_f43_ge: const_Misc_Val - 1.0012e+06 t56_leaf7 >= -1e+06
 TREE_57_ONEHOT: t57_leaf0 + t57_leaf1 + t57_leaf2 + t57_leaf3 + t57_leaf4
   + t57_leaf5 + t57_leaf6 + t57_leaf7 = 1
 T57_L0_f27_le: x_Kitchen_Qual + t57_leaf0 <= 4
 T57_L0_f12_le: x_BsmtFin_SF_1 + 999944 t57_leaf0 <= 1e+06
 T57_L0_f20_le: x_Gr_Liv_Area + 8854 t57_leaf0 <= 10000
 T57_L1_f27_le: x_Kitchen_Qual + t57_leaf1 <= 4
 T57_L1_f12_le: x_BsmtFin_SF_1 + 999944 t57_leaf1 <= 1e+06
 T57_R1_f20_ge: x_Gr_Liv_Area - 1146 t57_leaf1 >= 0
 T57_L2_f27_le: x_Kitchen_Qual + t57_leaf2 <= 4
 T57_R2_f12_ge: x_BsmtFin_SF_1 - 56 t57_leaf2 >= 0
 T57_L2_f1_le: const_Lot_Area + 991374 t57_leaf2 <= 1e+06
 T57_L3_f27_le: x_Kitchen_Qual + t57_leaf3 <= 4
 T57_R3_f12_ge: x_BsmtFin_SF_1 - 56 t57_leaf3 >= 0
 T57_R3_f1_ge: const_Lot_Area - 1.008626e+06 t57_leaf3 >= -1e+06
 T57_R4_f27_ge: x_Kitchen_Qual - 4 t57_leaf4 >= -1
 T57_L4_f20_le: x_Gr_Liv_Area + 7802 t57_leaf4 <= 10000
 T57_L4_f1_le: const_Lot_Area + 993760 t57_leaf4 <= 1e+06
 T57_R5_f27_ge: x_Kitchen_Qual - 4 t57_leaf5 >= -1
 T57_L5_f20_le: x_Gr_Liv_Area + 7802 t57_leaf5 <= 10000
 T57_R5_f1_ge: const_Lot_Area - 1.00624e+06 t57_leaf5 >= -1e+06
 T57_R6_f27_ge: x_Kitchen_Qual - 4 t57_leaf6 >= -1
 T57_R6_f20_ge: x_Gr_Liv_Area - 2198 t57_leaf6 >= 0
 T57_L6_f18_le: const_2nd_Flr_SF + 998595 t57_leaf6 <= 1e+06
 T57_R7_f27_ge: x_Kitchen_Qual - 4 t57_leaf7 >= -1
 T57_R7_f20_ge: x_Gr_Liv_Area - 2198 t57_leaf7 >= 0
 T57_R7_f18_ge: const_2nd_Flr_SF - 1.001405e+06 t57_leaf7 >= -1e+06
 TREE_58_ONEHOT: t58_leaf0 + t58_leaf1 + t58_leaf2 + t58_leaf3 + t58_leaf4
   + t58_leaf5 + t58_leaf6 + t58_leaf7 = 1
 T58_L0_f33_le: x_Garage_Area + 99629 t58_leaf0 <= 100000
 T58_L0_f1_le: const_Lot_Area + 994024 t58_leaf0 <= 1e+06
 T58_L0_f35_le: x_Garage_Cond + 2 t58_leaf0 <= 4
 T58_L1_f33_le: x_Garage_Area + 99629 t58_leaf1 <= 100000
 T58_L1_f1_le: const_Lot_Area + 994024 t58_leaf1 <= 1e+06
 T58_R1_f35_ge: x_Garage_Cond - 3 t58_leaf1 >= -1
 T58_L2_f33_le: x_Garage_Area + 99629 t58_leaf2 <= 100000
 T58_R2_f1_ge: const_Lot_Area - 1.005976e+06 t58_leaf2 >= -1e+06
 T58_L2_f20_le: x_Gr_Liv_Area + 9117 t58_leaf2 <= 10000
 T58_L3_f33_le: x_Garage_Area + 99629 t58_leaf3 <= 100000
 T58_R3_f1_ge: const_Lot_Area - 1.005976e+06 t58_leaf3 >= -1e+06
 T58_R3_f20_ge: x_Gr_Liv_Area - 883 t58_leaf3 >= 0
 T58_R4_f33_ge: x_Garage_Area - 371 t58_leaf4 >= 0
 T58_L4_f17_le: x_1st_Flr_SF + 3189 t58_leaf4 <= 5000
 T58_L4_f16_le: x_Heating_QC + t58_leaf4 <= 4
 T58_R5_f33_ge: x_Garage_Area - 371 t58_leaf5 >= 0
 T58_L5_f17_le: x_1st_Flr_SF + 3189 t58_leaf5 <= 5000
 T58_R5_f16_ge: x_Heating_QC - 4 t58_leaf5 >= -1
 T58_R6_f33_ge: x_Garage_Area - 371 t58_leaf6 >= 0
 T58_R6_f17_ge: x_1st_Flr_SF - 1811 t58_leaf6 >= 0
 T58_L6_f14_le: x_Bsmt_Unf_SF + 99215 t58_leaf6 <= 100000
 T58_R7_f33_ge: x_Garage_Area - 371 t58_leaf7 >= 0
 T58_R7_f17_ge: x_1st_Flr_SF - 1811 t58_leaf7 >= 0
 T58_R7_f14_ge: x_Bsmt_Unf_SF - 785 t58_leaf7 >= 0
 TREE_59_ONEHOT: t59_leaf0 + t59_leaf1 + t59_leaf2 + t59_leaf3 + t59_leaf4
   + t59_leaf5 + t59_leaf6 + t59_leaf7 = 1
 T59_L0_f31_le: const_Garage_Yr_Blt + 998071 t59_leaf0 <= 1e+06
 T59_L0_f14_le: x_Bsmt_Unf_SF + 99197 t59_leaf0 <= 100000
 T59_L0_f141_le: const_House_Style_1Story + 999999 t59_leaf0 <= 1e+06
 T59_L1_f31_le: const_Garage_Yr_Blt + 998071 t59_leaf1 <= 1e+06
 T59_L1_f14_le: x_Bsmt_Unf_SF + 99197 t59_leaf1 <= 100000
 T59_R1_f141_ge: const_House_Style_1Story - 1.000001e+06 t59_leaf1
   >= -1e+06
 T59_L2_f31_le: const_Garage_Yr_Blt + 998071 t59_leaf2 <= 1e+06
 T59_R2_f14_ge: x_Bsmt_Unf_SF - 803 t59_leaf2 >= 0
 T59_L2_f12_le: x_BsmtFin_SF_1 + 999919 t59_leaf2 <= 1e+06
 T59_L3_f31_le: const_Garage_Yr_Blt + 998071 t59_leaf3 <= 1e+06
 T59_R3_f14_ge: x_Bsmt_Unf_SF - 803 t59_leaf3 >= 0
 T59_R3_f12_ge: x_BsmtFin_SF_1 - 81 t59_leaf3 >= 0
 T59_R4_f31_ge: const_Garage_Yr_Blt - 1.001929e+06 t59_leaf4 >= -1e+06
 T59_L4_f1_le: const_Lot_Area + 988020 t59_leaf4 <= 1e+06
 T59_L4_f5_le: const_Year_Built + 998020 t59_leaf4 <= 1e+06
 T59_R5_f31_ge: const_Garage_Yr_Blt - 1.001929e+06 t59_leaf5 >= -1e+06
 T59_L5_f1_le: const_Lot_Area + 988020 t59_leaf5 <= 1e+06
 T59_R5_f5_ge: const_Year_Built - 1.00198e+06 t59_leaf5 >= -1e+06
 T59_R6_f31_ge: const_Garage_Yr_Blt - 1.001929e+06 t59_leaf6 >= -1e+06
 T59_R6_f1_ge: const_Lot_Area - 1.01198e+06 t59_leaf6 >= -1e+06
 T59_L6_f207_le: const_Bsmt_Exposure_Gd + 999999 t59_leaf6 <= 1e+06
 T59_R7_f31_ge: const_Garage_Yr_Blt - 1.001929e+06 t59_leaf7 >= -1e+06
 T59_R7_f1_ge: const_Lot_Area - 1.01198e+06 t59_leaf7 >= -1e+06
 T59_R7_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t59_leaf7 >= -1e+06
 TREE_60_ONEHOT: t60_leaf0 + t60_leaf1 + t60_leaf2 + t60_leaf3 + t60_leaf4
   + t60_leaf5 + t60_leaf6 = 1
 T60_L0_f30_le: x_Fireplace_Qu + 3 t60_leaf0 <= 4
 T60_L0_f3_le: Overall_Qual_calc + 999997 t60_leaf0 <= 1e+06
 T60_L1_f30_le: x_Fireplace_Qu + 3 t60_leaf1 <= 4
 T60_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t60_leaf1 >= -1e+06
 T60_L1_f44_le: const_Yr_Sold + 997992 t60_leaf1 <= 1e+06
 T60_L2_f30_le: x_Fireplace_Qu + 3 t60_leaf2 <= 4
 T60_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t60_leaf2 >= -1e+06
 T60_R2_f44_ge: const_Yr_Sold - 1.002008e+06 t60_leaf2 >= -1e+06
 T60_R3_f30_ge: x_Fireplace_Qu - 2 t60_leaf3 >= -1
 T60_L3_f12_le: x_BsmtFin_SF_1 + 998903 t60_leaf3 <= 1e+06
 T60_L3_f0_le: const_Lot_Frontage + 999935 t60_leaf3 <= 1e+06
 T60_R4_f30_ge: x_Fireplace_Qu - 2 t60_leaf4 >= -1
 T60_L4_f12_le: x_BsmtFin_SF_1 + 998903 t60_leaf4 <= 1e+06
 T60_R4_f0_ge: const_Lot_Frontage - 1.000065e+06 t60_leaf4 >= -1e+06
 T60_R5_f30_ge: x_Fireplace_Qu - 2 t60_leaf5 >= -1
 T60_R5_f12_ge: x_BsmtFin_SF_1 - 1097 t60_leaf5 >= 0
 T60_L5_f28_le: x_TotRms_AbvGrd + 5 t60_leaf5 <= 15
 T60_R6_f30_ge: x_Fireplace_Qu - 2 t60_leaf6 >= -1
 T60_R6_f12_ge: x_BsmtFin_SF_1 - 1097 t60_leaf6 >= 0
 T60_R6_f28_ge: x_TotRms_AbvGrd - 10 t60_leaf6 >= 0
 TREE_61_ONEHOT: t61_leaf0 + t61_leaf1 + t61_leaf2 + t61_leaf3 + t61_leaf4
   + t61_leaf5 + t61_leaf6 = 1
 T61_L0_f4_le: const_Overall_Cond + 999996 t61_leaf0 <= 1e+06
 T61_L0_f15_le: const_Total_Bsmt_SF + 998600 t61_leaf0 <= 1e+06
 T61_L0_f297_le: const_Sale_Condition_Normal + 999999 t61_leaf0 <= 1e+06
 T61_L1_f4_le: const_Overall_Cond + 999996 t61_leaf1 <= 1e+06
 T61_L1_f15_le: const_Total_Bsmt_SF + 998600 t61_leaf1 <= 1e+06
 T61_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t61_leaf1
   >= -1e+06
 T61_L2_f4_le: const_Overall_Cond + 999996 t61_leaf2 <= 1e+06
 T61_R2_f15_ge: const_Total_Bsmt_SF - 1.0014e+06 t61_leaf2 >= -1e+06
 T61_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t61_leaf3 >= -1e+06
 T61_L3_f15_le: const_Total_Bsmt_SF + 998976 t61_leaf3 <= 1e+06
 T61_L3_f1_le: const_Lot_Area + 993875 t61_leaf3 <= 1e+06
 T61_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t61_leaf4 >= -1e+06
 T61_L4_f15_le: const_Total_Bsmt_SF + 998976 t61_leaf4 <= 1e+06
 T61_R4_f1_ge: const_Lot_Area - 1.006125e+06 t61_leaf4 >= -1e+06
 T61_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t61_leaf5 >= -1e+06
 T61_R5_f15_ge: const_Total_Bsmt_SF - 1.001024e+06 t61_leaf5 >= -1e+06
 T61_L5_f18_le: const_2nd_Flr_SF + 999120 t61_leaf5 <= 1e+06
 T61_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t61_leaf6 >= -1e+06
 T61_R6_f15_ge: const_Total_Bsmt_SF - 1.001024e+06 t61_leaf6 >= -1e+06
 T61_R6_f18_ge: const_2nd_Flr_SF - 1.00088e+06 t61_leaf6 >= -1e+06
 TREE_62_ONEHOT: t62_leaf0 + t62_leaf1 + t62_leaf2 + t62_leaf3 + t62_leaf4
   + t62_leaf5 + t62_leaf6 = 1
 T62_L0_f20_le: x_Gr_Liv_Area + 7802 t62_leaf0 <= 10000
 T62_L0_f12_le: x_BsmtFin_SF_1 + 999671 t62_leaf0 <= 1e+06
 T62_L0_f5_le: const_Year_Built + 998016 t62_leaf0 <= 1e+06
 T62_L1_f20_le: x_Gr_Liv_Area + 7802 t62_leaf1 <= 10000
 T62_L1_f12_le: x_BsmtFin_SF_1 + 999671 t62_leaf1 <= 1e+06
 T62_R1_f5_ge: const_Year_Built - 1.001984e+06 t62_leaf1 >= -1e+06
 T62_L2_f20_le: x_Gr_Liv_Area + 7802 t62_leaf2 <= 10000
 T62_R2_f12_ge: x_BsmtFin_SF_1 - 329 t62_leaf2 >= 0
 T62_L2_f20_le: x_Gr_Liv_Area + 8584 t62_leaf2 <= 10000
 T62_L3_f20_le: x_Gr_Liv_Area + 7802 t62_leaf3 <= 10000
 T62_R3_f12_ge: x_BsmtFin_SF_1 - 329 t62_leaf3 >= 0
 T62_R3_f20_ge: x_Gr_Liv_Area - 1416 t62_leaf3 >= 0
 T62_R4_f20_ge: x_Gr_Liv_Area - 2198 t62_leaf4 >= 0
 T62_L4_f118_le: const_Condition_1_Feedr + 999999 t62_leaf4 <= 1e+06
 T62_L4_f1_le: const_Lot_Area + 986482 t62_leaf4 <= 1e+06
 T62_R5_f20_ge: x_Gr_Liv_Area - 2198 t62_leaf5 >= 0
 T62_L5_f118_le: const_Condition_1_Feedr + 999999 t62_leaf5 <= 1e+06
 T62_R5_f1_ge: const_Lot_Area - 1.013518e+06 t62_leaf5 >= -1e+06
 T62_R6_f20_ge: x_Gr_Liv_Area - 2198 t62_leaf6 >= 0
 T62_R6_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t62_leaf6
   >= -1e+06
 TREE_63_ONEHOT: t63_leaf0 + t63_leaf1 + t63_leaf2 + t63_leaf3 + t63_leaf4
   + t63_leaf5 + t63_leaf6 + t63_leaf7 = 1
 T63_L0_f6_le: const_Year_Remod_Add + 998013 t63_leaf0 <= 1e+06
 T63_L0_f12_le: x_BsmtFin_SF_1 + 999550 t63_leaf0 <= 1e+06
 T63_L0_f3_le: Overall_Qual_calc + 999997 t63_leaf0 <= 1e+06
 T63_L1_f6_le: const_Year_Remod_Add + 998013 t63_leaf1 <= 1e+06
 T63_L1_f12_le: x_BsmtFin_SF_1 + 999550 t63_leaf1 <= 1e+06
 T63_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t63_leaf1 >= -1e+06
 T63_L2_f6_le: const_Year_Remod_Add + 998013 t63_leaf2 <= 1e+06
 T63_R2_f12_ge: x_BsmtFin_SF_1 - 450 t63_leaf2 >= 0
 T63_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t63_leaf2 <= 1e+06
 T63_L3_f6_le: const_Year_Remod_Add + 998013 t63_leaf3 <= 1e+06
 T63_R3_f12_ge: x_BsmtFin_SF_1 - 450 t63_leaf3 >= 0
 T63_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t63_leaf3
   >= -1e+06
 T63_R4_f6_ge: const_Year_Remod_Add - 1.001987e+06 t63_leaf4 >= -1e+06
 T63_L4_f18_le: const_2nd_Flr_SF + 998632 t63_leaf4 <= 1e+06
 T63_L4_f96_le: const_Neighborhood_Edwards + 999999 t63_leaf4 <= 1e+06
 T63_R5_f6_ge: const_Year_Remod_Add - 1.001987e+06 t63_leaf5 >= -1e+06
 T63_L5_f18_le: const_2nd_Flr_SF + 998632 t63_leaf5 <= 1e+06
 T63_R5_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t63_leaf5
   >= -1e+06
 T63_R6_f6_ge: const_Year_Remod_Add - 1.001987e+06 t63_leaf6 >= -1e+06
 T63_R6_f18_ge: const_2nd_Flr_SF - 1.001368e+06 t63_leaf6 >= -1e+06
 T63_L6_f6_le: const_Year_Remod_Add + 998001 t63_leaf6 <= 1e+06
 T63_R7_f6_ge: const_Year_Remod_Add - 1.001987e+06 t63_leaf7 >= -1e+06
 T63_R7_f18_ge: const_2nd_Flr_SF - 1.001368e+06 t63_leaf7 >= -1e+06
 T63_R7_f6_ge: const_Year_Remod_Add - 1.001999e+06 t63_leaf7 >= -1e+06
 TREE_64_ONEHOT: t64_leaf0 + t64_leaf1 + t64_leaf2 + t64_leaf3 + t64_leaf4
   + t64_leaf5 + t64_leaf6 + t64_leaf7 = 1
 T64_L0_f20_le: x_Gr_Liv_Area + 8836 t64_leaf0 <= 10000
 T64_L0_f12_le: x_BsmtFin_SF_1 + 999944 t64_leaf0 <= 1e+06
 T64_L0_f14_le: x_Bsmt_Unf_SF + 99140 t64_leaf0 <= 100000
 T64_L1_f20_le: x_Gr_Liv_Area + 8836 t64_leaf1 <= 10000
 T64_L1_f12_le: x_BsmtFin_SF_1 + 999944 t64_leaf1 <= 1e+06
 T64_R1_f14_ge: x_Bsmt_Unf_SF - 860 t64_leaf1 >= 0
 T64_L2_f20_le: x_Gr_Liv_Area + 8836 t64_leaf2 <= 10000
 T64_R2_f12_ge: x_BsmtFin_SF_1 - 56 t64_leaf2 >= 0
 T64_L2_f6_le: const_Year_Remod_Add + 998023 t64_leaf2 <= 1e+06
 T64_L3_f20_le: x_Gr_Liv_Area + 8836 t64_leaf3 <= 10000
 T64_R3_f12_ge: x_BsmtFin_SF_1 - 56 t64_leaf3 >= 0
 T64_R3_f6_ge: const_Year_Remod_Add - 1.001977e+06 t64_leaf3 >= -1e+06
 T64_R4_f20_ge: x_Gr_Liv_Area - 1164 t64_leaf4 >= 0
 T64_L4_f20_le: x_Gr_Liv_Area + 7456 t64_leaf4 <= 10000
 T64_L4_f12_le: x_BsmtFin_SF_1 + 999708 t64_leaf4 <= 1e+06
 T64_R5_f20_ge: x_Gr_Liv_Area - 1164 t64_leaf5 >= 0
 T64_L5_f20_le: x_Gr_Liv_Area + 7456 t64_leaf5 <= 10000
 T64_R5_f12_ge: x_BsmtFin_SF_1 - 292 t64_leaf5 >= 0
 T64_R6_f20_ge: x_Gr_Liv_Area - 1164 t64_leaf6 >= 0
 T64_R6_f20_ge: x_Gr_Liv_Area - 2544 t64_leaf6 >= 0
 T64_L6_f20_le: x_Gr_Liv_Area + 6914 t64_leaf6 <= 10000
 T64_R7_f20_ge: x_Gr_Liv_Area - 1164 t64_leaf7 >= 0
 T64_R7_f20_ge: x_Gr_Liv_Area - 2544 t64_leaf7 >= 0
 T64_R7_f20_ge: x_Gr_Liv_Area - 3086 t64_leaf7 >= 0
 TREE_65_ONEHOT: t65_leaf0 + t65_leaf1 + t65_leaf2 + t65_leaf3 + t65_leaf4
   + t65_leaf5 + t65_leaf6 + t65_leaf7 = 1
 T65_L0_f15_le: const_Total_Bsmt_SF + 999352 t65_leaf0 <= 1e+06
 T65_L0_f3_le: Overall_Qual_calc + 999996 t65_leaf0 <= 1e+06
 T65_L0_f191_le: x_ex2_is_VinylSd <= 1
 T65_L1_f15_le: const_Total_Bsmt_SF + 999352 t65_leaf1 <= 1e+06
 T65_L1_f3_le: Overall_Qual_calc + 999996 t65_leaf1 <= 1e+06
 T65_R1_f191_ge: x_ex2_is_VinylSd - t65_leaf1 >= 0
 T65_L2_f15_le: const_Total_Bsmt_SF + 999352 t65_leaf2 <= 1e+06
 T65_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t65_leaf2 >= -1e+06
 T65_L2_f1_le: const_Lot_Area + 993238 t65_leaf2 <= 1e+06
 T65_L3_f15_le: const_Total_Bsmt_SF + 999352 t65_leaf3 <= 1e+06
 T65_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t65_leaf3 >= -1e+06
 T65_R3_f1_ge: const_Lot_Area - 1.006762e+06 t65_leaf3 >= -1e+06
 T65_R4_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t65_leaf4 >= -1e+06
 T65_L4_f17_le: x_1st_Flr_SF + 3170 t65_leaf4 <= 5000
 T65_L4_f6_le: const_Year_Remod_Add + 998037 t65_leaf4 <= 1e+06
 T65_R5_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t65_leaf5 >= -1e+06
 T65_L5_f17_le: x_1st_Flr_SF + 3170 t65_leaf5 <= 5000
 T65_R5_f6_ge: const_Year_Remod_Add - 1.001963e+06 t65_leaf5 >= -1e+06
 T65_R6_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t65_leaf6 >= -1e+06
 T65_R6_f17_ge: x_1st_Flr_SF - 1830 t65_leaf6 >= 0
 T65_L6_f1_le: const_Lot_Area + 988612 t65_leaf6 <= 1e+06
 T65_R7_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t65_leaf7 >= -1e+06
 T65_R7_f17_ge: x_1st_Flr_SF - 1830 t65_leaf7 >= 0
 T65_R7_f1_ge: const_Lot_Area - 1.011388e+06 t65_leaf7 >= -1e+06
 TREE_66_ONEHOT: t66_leaf0 + t66_leaf1 + t66_leaf2 + t66_leaf3 + t66_leaf4
   + t66_leaf5 + t66_leaf6 + t66_leaf7 = 1
 T66_L0_f257_le: x_paved_drive_is_N <= 1
 T66_L0_f44_le: const_Yr_Sold + 997992 t66_leaf0 <= 1e+06
 T66_L0_f20_le: x_Gr_Liv_Area + 8032 t66_leaf0 <= 10000
 T66_L1_f257_le: x_paved_drive_is_N <= 1
 T66_L1_f44_le: const_Yr_Sold + 997992 t66_leaf1 <= 1e+06
 T66_R1_f20_ge: x_Gr_Liv_Area - 1968 t66_leaf1 >= 0
 T66_L2_f257_le: x_paved_drive_is_N <= 1
 T66_R2_f44_ge: const_Yr_Sold - 1.002008e+06 t66_leaf2 >= -1e+06
 T66_L2_f16_le: x_Heating_QC + t66_leaf2 <= 4
 T66_L3_f257_le: x_paved_drive_is_N <= 1
 T66_R3_f44_ge: const_Yr_Sold - 1.002008e+06 t66_leaf3 >= -1e+06
 T66_R3_f16_ge: x_Heating_QC - 4 t66_leaf3 >= -1
 T66_R4_f257_ge: x_paved_drive_is_N - t66_leaf4 >= 0
 T66_L4_f44_le: const_Yr_Sold + 997992 t66_leaf4 <= 1e+06
 T66_L4_f20_le: x_Gr_Liv_Area + 8512 t66_leaf4 <= 10000
 T66_R5_f257_ge: x_paved_drive_is_N - t66_leaf5 >= 0
 T66_L5_f44_le: const_Yr_Sold + 997992 t66_leaf5 <= 1e+06
 T66_R5_f20_ge: x_Gr_Liv_Area - 1488 t66_leaf5 >= 0
 T66_R6_f257_ge: x_paved_drive_is_N - t66_leaf6 >= 0
 T66_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t66_leaf6 >= -1e+06
 T66_L6_f20_le: x_Gr_Liv_Area + 8895 t66_leaf6 <= 10000
 T66_R7_f257_ge: x_paved_drive_is_N - t66_leaf7 >= 0
 T66_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t66_leaf7 >= -1e+06
 T66_R7_f20_ge: x_Gr_Liv_Area - 1105 t66_leaf7 >= 0
 TREE_67_ONEHOT: t67_leaf0 + t67_leaf1 + t67_leaf2 + t67_leaf3 + t67_leaf4
   + t67_leaf5 = 1
 T67_L0_f3_le: Overall_Qual_calc + 999996 t67_leaf0 <= 1e+06
 T67_L0_f297_le: const_Sale_Condition_Normal + 999999 t67_leaf0 <= 1e+06
 T67_L1_f3_le: Overall_Qual_calc + 999996 t67_leaf1 <= 1e+06
 T67_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t67_leaf1
   >= -1e+06
 T67_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t67_leaf2 >= -1e+06
 T67_L2_f6_le: const_Year_Remod_Add + 998007 t67_leaf2 <= 1e+06
 T67_L2_f44_le: const_Yr_Sold + 997992 t67_leaf2 <= 1e+06
 T67_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t67_leaf3 >= -1e+06
 T67_L3_f6_le: const_Year_Remod_Add + 998007 t67_leaf3 <= 1e+06
 T67_R3_f44_ge: const_Yr_Sold - 1.002008e+06 t67_leaf3 >= -1e+06
 T67_R4_f3_ge: Overall_Qual_calc - 1.000004e+06 t67_leaf4 >= -1e+06
 T67_R4_f6_ge: const_Year_Remod_Add - 1.001993e+06 t67_leaf4 >= -1e+06
 T67_L4_f27_le: x_Kitchen_Qual <= 4
 T67_R5_f3_ge: Overall_Qual_calc - 1.000004e+06 t67_leaf5 >= -1e+06
 T67_R5_f6_ge: const_Year_Remod_Add - 1.001993e+06 t67_leaf5 >= -1e+06
 T67_R5_f27_ge: x_Kitchen_Qual - 5 t67_leaf5 >= -1
 TREE_68_ONEHOT: t68_leaf0 + t68_leaf1 + t68_leaf2 + t68_leaf3 + t68_leaf4
   + t68_leaf5 + t68_leaf6 + t68_leaf7 = 1
 T68_L0_f30_le: x_Fireplace_Qu + 3 t68_leaf0 <= 4
 T68_L0_f46_le: const_MS_SubClass_30 + 999999 t68_leaf0 <= 1e+06
 T68_L0_f1_le: const_Lot_Area + 993880 t68_leaf0 <= 1e+06
 T68_L1_f30_le: x_Fireplace_Qu + 3 t68_leaf1 <= 4
 T68_L1_f46_le: const_MS_SubClass_30 + 999999 t68_leaf1 <= 1e+06
 T68_R1_f1_ge: const_Lot_Area - 1.00612e+06 t68_leaf1 >= -1e+06
 T68_L2_f30_le: x_Fireplace_Qu + 3 t68_leaf2 <= 4
 T68_R2_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t68_leaf2 >= -1e+06
 T68_L2_f264_le: x_fence_is_No_aplica <= 1
 T68_L3_f30_le: x_Fireplace_Qu + 3 t68_leaf3 <= 4
 T68_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t68_leaf3 >= -1e+06
 T68_R3_f264_ge: x_fence_is_No_aplica - t68_leaf3 >= 0
 T68_R4_f30_ge: x_Fireplace_Qu - 2 t68_leaf4 >= -1
 T68_L4_f44_le: const_Yr_Sold + 997992 t68_leaf4 <= 1e+06
 T68_L4_f17_le: x_1st_Flr_SF + 3511 t68_leaf4 <= 5000
 T68_R5_f30_ge: x_Fireplace_Qu - 2 t68_leaf5 >= -1
 T68_L5_f44_le: const_Yr_Sold + 997992 t68_leaf5 <= 1e+06
 T68_R5_f17_ge: x_1st_Flr_SF - 1489 t68_leaf5 >= 0
 T68_R6_f30_ge: x_Fireplace_Qu - 2 t68_leaf6 >= -1
 T68_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t68_leaf6 >= -1e+06
 T68_L6_f21_le: const_Bsmt_Full_Bath + 999999 t68_leaf6 <= 1e+06
 T68_R7_f30_ge: x_Fireplace_Qu - 2 t68_leaf7 >= -1
 T68_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t68_leaf7 >= -1e+06
 T68_R7_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t68_leaf7 >= -1e+06
 TREE_69_ONEHOT: t69_leaf0 + t69_leaf1 + t69_leaf2 + t69_leaf3 + t69_leaf4
   + t69_leaf5 + t69_leaf6 = 1
 T69_L0_f4_le: const_Overall_Cond + 999995 t69_leaf0 <= 1e+06
 T69_L0_f31_le: const_Garage_Yr_Blt + 998005 t69_leaf0 <= 1e+06
 T69_L0_f37_le: x_Open_Porch_SF + 99928 t69_leaf0 <= 100000
 T69_L1_f4_le: const_Overall_Cond + 999995 t69_leaf1 <= 1e+06
 T69_L1_f31_le: const_Garage_Yr_Blt + 998005 t69_leaf1 <= 1e+06
 T69_R1_f37_ge: x_Open_Porch_SF - 72 t69_leaf1 >= 0
 T69_L2_f4_le: const_Overall_Cond + 999995 t69_leaf2 <= 1e+06
 T69_R2_f31_ge: const_Garage_Yr_Blt - 1.001995e+06 t69_leaf2 >= -1e+06
 T69_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t69_leaf3 >= -1e+06
 T69_L3_f207_le: const_Bsmt_Exposure_Gd + 999999 t69_leaf3 <= 1e+06
 T69_L3_f1_le: const_Lot_Area + 992800 t69_leaf3 <= 1e+06
 T69_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t69_leaf4 >= -1e+06
 T69_L4_f207_le: const_Bsmt_Exposure_Gd + 999999 t69_leaf4 <= 1e+06
 T69_R4_f1_ge: const_Lot_Area - 1.0072e+06 t69_leaf4 >= -1e+06
 T69_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t69_leaf5 >= -1e+06
 T69_R5_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t69_leaf5 >= -1e+06
 T69_L5_f17_le: x_1st_Flr_SF + 4052 t69_leaf5 <= 5000
 T69_R6_f4_ge: const_Overall_Cond - 1.000005e+06 t69_leaf6 >= -1e+06
 T69_R6_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t69_leaf6 >= -1e+06
 T69_R6_f17_ge: x_1st_Flr_SF - 948 t69_leaf6 >= 0
 TREE_70_ONEHOT: t70_leaf0 + t70_leaf1 + t70_leaf2 + t70_leaf3 + t70_leaf4
   + t70_leaf5 + t70_leaf6 + t70_leaf7 = 1
 T70_L0_f6_le: const_Year_Remod_Add + 998016 t70_leaf0 <= 1e+06
 T70_L0_f66_le: const_MS_Zoning_RL + 999999 t70_leaf0 <= 1e+06
 T70_L0_f44_le: const_Yr_Sold + 997992 t70_leaf0 <= 1e+06
 T70_L1_f6_le: const_Year_Remod_Add + 998016 t70_leaf1 <= 1e+06
 T70_L1_f66_le: const_MS_Zoning_RL + 999999 t70_leaf1 <= 1e+06
 T70_R1_f44_ge: const_Yr_Sold - 1.002008e+06 t70_leaf1 >= -1e+06
 T70_L2_f6_le: const_Year_Remod_Add + 998016 t70_leaf2 <= 1e+06
 T70_R2_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t70_leaf2 >= -1e+06
 T70_L2_f14_le: x_Bsmt_Unf_SF + 99480 t70_leaf2 <= 100000
 T70_L3_f6_le: const_Year_Remod_Add + 998016 t70_leaf3 <= 1e+06
 T70_R3_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t70_leaf3 >= -1e+06
 T70_R3_f14_ge: x_Bsmt_Unf_SF - 520 t70_leaf3 >= 0
 T70_R4_f6_ge: const_Year_Remod_Add - 1.001984e+06 t70_leaf4 >= -1e+06
 T70_L4_f109_le: const_Neighborhood_OldTown + 999999 t70_leaf4 <= 1e+06
 T70_L4_f18_le: const_2nd_Flr_SF + 998704 t70_leaf4 <= 1e+06
 T70_R5_f6_ge: const_Year_Remod_Add - 1.001984e+06 t70_leaf5 >= -1e+06
 T70_L5_f109_le: const_Neighborhood_OldTown + 999999 t70_leaf5 <= 1e+06
 T70_R5_f18_ge: const_2nd_Flr_SF - 1.001296e+06 t70_leaf5 >= -1e+06
 T70_R6_f6_ge: const_Year_Remod_Add - 1.001984e+06 t70_leaf6 >= -1e+06
 T70_R6_f109_ge: const_Neighborhood_OldTown - 1.000001e+06 t70_leaf6
   >= -1e+06
 T70_L6_f200_le: const_Foundation_BrkTil + 999999 t70_leaf6 <= 1e+06
 T70_R7_f6_ge: const_Year_Remod_Add - 1.001984e+06 t70_leaf7 >= -1e+06
 T70_R7_f109_ge: const_Neighborhood_OldTown - 1.000001e+06 t70_leaf7
   >= -1e+06
 T70_R7_f200_ge: const_Foundation_BrkTil - 1.000001e+06 t70_leaf7
   >= -1e+06
 TREE_71_ONEHOT: t71_leaf0 + t71_leaf1 + t71_leaf2 + t71_leaf3 + t71_leaf4
   + t71_leaf5 + t71_leaf6 + t71_leaf7 = 1
 T71_L0_f20_le: x_Gr_Liv_Area + 8078 t71_leaf0 <= 10000
 T71_L0_f10_le: const_Bsmt_Qual + 999998 t71_leaf0 <= 1e+06
 T71_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t71_leaf0 <= 1e+06
 T71_L1_f20_le: x_Gr_Liv_Area + 8078 t71_leaf1 <= 10000
 T71_L1_f10_le: const_Bsmt_Qual + 999998 t71_leaf1 <= 1e+06
 T71_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t71_leaf1
   >= -1e+06
 T71_L2_f20_le: x_Gr_Liv_Area + 8078 t71_leaf2 <= 10000
 T71_R2_f10_ge: const_Bsmt_Qual - 1.000002e+06 t71_leaf2 >= -1e+06
 T71_L2_f12_le: x_BsmtFin_SF_1 + 999708 t71_leaf2 <= 1e+06
 T71_L3_f20_le: x_Gr_Liv_Area + 8078 t71_leaf3 <= 10000
 T71_R3_f10_ge: const_Bsmt_Qual - 1.000002e+06 t71_leaf3 >= -1e+06
 T71_R3_f12_ge: x_BsmtFin_SF_1 - 292 t71_leaf3 >= 0
 T71_R4_f20_ge: x_Gr_Liv_Area - 1922 t71_leaf4 >= 0
 T71_L4_f18_le: const_2nd_Flr_SF + 998632 t71_leaf4 <= 1e+06
 T71_L4_f245_le: const_FunctioNo_aplical_Typ + 999999 t71_leaf4 <= 1e+06
 T71_R5_f20_ge: x_Gr_Liv_Area - 1922 t71_leaf5 >= 0
 T71_L5_f18_le: const_2nd_Flr_SF + 998632 t71_leaf5 <= 1e+06
 T71_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t71_leaf5
   >= -1e+06
 T71_R6_f20_ge: x_Gr_Liv_Area - 1922 t71_leaf6 >= 0
 T71_R6_f18_ge: const_2nd_Flr_SF - 1.001368e+06 t71_leaf6 >= -1e+06
 T71_L6_f1_le: const_Lot_Area + 987616 t71_leaf6 <= 1e+06
 T71_R7_f20_ge: x_Gr_Liv_Area - 1922 t71_leaf7 >= 0
 T71_R7_f18_ge: const_2nd_Flr_SF - 1.001368e+06 t71_leaf7 >= -1e+06
 T71_R7_f1_ge: const_Lot_Area - 1.012384e+06 t71_leaf7 >= -1e+06
 TREE_72_ONEHOT: t72_leaf0 + t72_leaf1 + t72_leaf2 + t72_leaf3 + t72_leaf4
   + t72_leaf5 + t72_leaf6 + t72_leaf7 = 1
 T72_L0_f44_le: const_Yr_Sold + 997992 t72_leaf0 <= 1e+06
 T72_L0_f17_le: x_1st_Flr_SF + 3680 t72_leaf0 <= 5000
 T72_L0_f58_le: const_MS_SubClass_160 + 999999 t72_leaf0 <= 1e+06
 T72_L1_f44_le: const_Yr_Sold + 997992 t72_leaf1 <= 1e+06
 T72_L1_f17_le: x_1st_Flr_SF + 3680 t72_leaf1 <= 5000
 T72_R1_f58_ge: const_MS_SubClass_160 - 1.000001e+06 t72_leaf1 >= -1e+06
 T72_L2_f44_le: const_Yr_Sold + 997992 t72_leaf2 <= 1e+06
 T72_R2_f17_ge: x_1st_Flr_SF - 1320 t72_leaf2 >= 0
 T72_L2_f6_le: const_Year_Remod_Add + 998009 t72_leaf2 <= 1e+06
 T72_L3_f44_le: const_Yr_Sold + 997992 t72_leaf3 <= 1e+06
 T72_R3_f17_ge: x_1st_Flr_SF - 1320 t72_leaf3 >= 0
 T72_R3_f6_ge: const_Year_Remod_Add - 1.001991e+06 t72_leaf3 >= -1e+06
 T72_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t72_leaf4 >= -1e+06
 T72_L4_f6_le: const_Year_Remod_Add + 998016 t72_leaf4 <= 1e+06
 T72_L4_f297_le: const_Sale_Condition_Normal + 999999 t72_leaf4 <= 1e+06
 T72_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t72_leaf5 >= -1e+06
 T72_L5_f6_le: const_Year_Remod_Add + 998016 t72_leaf5 <= 1e+06
 T72_R5_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t72_leaf5
   >= -1e+06
 T72_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t72_leaf6 >= -1e+06
 T72_R6_f6_ge: const_Year_Remod_Add - 1.001984e+06 t72_leaf6 >= -1e+06
 T72_L6_f20_le: x_Gr_Liv_Area + 7655 t72_leaf6 <= 10000
 T72_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t72_leaf7 >= -1e+06
 T72_R7_f6_ge: const_Year_Remod_Add - 1.001984e+06 t72_leaf7 >= -1e+06
 T72_R7_f20_ge: x_Gr_Liv_Area - 2345 t72_leaf7 >= 0
 TREE_73_ONEHOT: t73_leaf0 + t73_leaf1 + t73_leaf2 + t73_leaf3 + t73_leaf4
   + t73_leaf5 + t73_leaf6 = 1
 T73_L0_f4_le: const_Overall_Cond + 999996 t73_leaf0 <= 1e+06
 T73_L0_f14_le: x_Bsmt_Unf_SF + 99586 t73_leaf0 <= 100000
 T73_L1_f4_le: const_Overall_Cond + 999996 t73_leaf1 <= 1e+06
 T73_R1_f14_ge: x_Bsmt_Unf_SF - 414 t73_leaf1 >= 0
 T73_L1_f8_le: x_Exter_Qual + 2 t73_leaf1 <= 4
 T73_L2_f4_le: const_Overall_Cond + 999996 t73_leaf2 <= 1e+06
 T73_R2_f14_ge: x_Bsmt_Unf_SF - 414 t73_leaf2 >= 0
 T73_R2_f8_ge: x_Exter_Qual - 3 t73_leaf2 >= -1
 T73_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t73_leaf3 >= -1e+06
 T73_L3_f18_le: const_2nd_Flr_SF + 998704 t73_leaf3 <= 1e+06
 T73_L3_f95_le: const_Neighborhood_Crawfor + 999999 t73_leaf3 <= 1e+06
 T73_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t73_leaf4 >= -1e+06
 T73_L4_f18_le: const_2nd_Flr_SF + 998704 t73_leaf4 <= 1e+06
 T73_R4_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t73_leaf4
   >= -1e+06
 T73_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t73_leaf5 >= -1e+06
 T73_R5_f18_ge: const_2nd_Flr_SF - 1.001296e+06 t73_leaf5 >= -1e+06
 T73_L5_f14_le: x_Bsmt_Unf_SF + 99574 t73_leaf5 <= 100000
 T73_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t73_leaf6 >= -1e+06
 T73_R6_f18_ge: const_2nd_Flr_SF - 1.001296e+06 t73_leaf6 >= -1e+06
 T73_R6_f14_ge: x_Bsmt_Unf_SF - 426 t73_leaf6 >= 0
 TREE_74_ONEHOT: t74_leaf0 + t74_leaf1 + t74_leaf2 + t74_leaf3 + t74_leaf4
   + t74_leaf5 + t74_leaf6 = 1
 T74_L0_f3_le: Overall_Qual_calc + 999992 t74_leaf0 <= 1e+06
 T74_L0_f46_le: const_MS_SubClass_30 + 999999 t74_leaf0 <= 1e+06
 T74_L0_f44_le: const_Yr_Sold + 997992 t74_leaf0 <= 1e+06
 T74_L1_f3_le: Overall_Qual_calc + 999992 t74_leaf1 <= 1e+06
 T74_L1_f46_le: const_MS_SubClass_30 + 999999 t74_leaf1 <= 1e+06
 T74_R1_f44_ge: const_Yr_Sold - 1.002008e+06 t74_leaf1 >= -1e+06
 T74_L2_f3_le: Overall_Qual_calc + 999992 t74_leaf2 <= 1e+06
 T74_R2_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t74_leaf2 >= -1e+06
 T74_L2_f17_le: x_1st_Flr_SF + 4030 t74_leaf2 <= 5000
 T74_L3_f3_le: Overall_Qual_calc + 999992 t74_leaf3 <= 1e+06
 T74_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t74_leaf3 >= -1e+06
 T74_R3_f17_ge: x_1st_Flr_SF - 970 t74_leaf3 >= 0
 T74_R4_f3_ge: Overall_Qual_calc - 1.000008e+06 t74_leaf4 >= -1e+06
 T74_L4_f18_le: const_2nd_Flr_SF + 998632 t74_leaf4 <= 1e+06
 T74_L4_f17_le: x_1st_Flr_SF + 3536 t74_leaf4 <= 5000
 T74_R5_f3_ge: Overall_Qual_calc - 1.000008e+06 t74_leaf5 >= -1e+06
 T74_L5_f18_le: const_2nd_Flr_SF + 998632 t74_leaf5 <= 1e+06
 T74_R5_f17_ge: x_1st_Flr_SF - 1464 t74_leaf5 >= 0
 T74_R6_f3_ge: Overall_Qual_calc - 1.000008e+06 t74_leaf6 >= -1e+06
 T74_R6_f18_ge: const_2nd_Flr_SF - 1.001368e+06 t74_leaf6 >= -1e+06
 TREE_75_ONEHOT: t75_leaf0 + t75_leaf1 + t75_leaf2 + t75_leaf3 + t75_leaf4
   + t75_leaf5 + t75_leaf6 = 1
 T75_L0_f4_le: const_Overall_Cond + 999996 t75_leaf0 <= 1e+06
 T75_L0_f36_le: x_Wood_Deck_SF + 99978 t75_leaf0 <= 100000
 T75_L0_f297_le: const_Sale_Condition_Normal + 999999 t75_leaf0 <= 1e+06
 T75_L1_f4_le: const_Overall_Cond + 999996 t75_leaf1 <= 1e+06
 T75_L1_f36_le: x_Wood_Deck_SF + 99978 t75_leaf1 <= 100000
 T75_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t75_leaf1
   >= -1e+06
 T75_L2_f4_le: const_Overall_Cond + 999996 t75_leaf2 <= 1e+06
 T75_R2_f36_ge: x_Wood_Deck_SF - 22 t75_leaf2 >= 0
 T75_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t75_leaf3 >= -1e+06
 T75_L3_f20_le: x_Gr_Liv_Area + 8288 t75_leaf3 <= 10000
 T75_L3_f31_le: const_Garage_Yr_Blt + 998079 t75_leaf3 <= 1e+06
 T75_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t75_leaf4 >= -1e+06
 T75_L4_f20_le: x_Gr_Liv_Area + 8288 t75_leaf4 <= 10000
 T75_R4_f31_ge: const_Garage_Yr_Blt - 1.001921e+06 t75_leaf4 >= -1e+06
 T75_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t75_leaf5 >= -1e+06
 T75_R5_f20_ge: x_Gr_Liv_Area - 1712 t75_leaf5 >= 0
 T75_L5_f12_le: x_BsmtFin_SF_1 + 999812 t75_leaf5 <= 1e+06
 T75_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t75_leaf6 >= -1e+06
 T75_R6_f20_ge: x_Gr_Liv_Area - 1712 t75_leaf6 >= 0
 T75_R6_f12_ge: x_BsmtFin_SF_1 - 188 t75_leaf6 >= 0
 TREE_76_ONEHOT: t76_leaf0 + t76_leaf1 + t76_leaf2 + t76_leaf3 + t76_leaf4
   + t76_leaf5 + t76_leaf6 + t76_leaf7 = 1
 T76_L0_f6_le: const_Year_Remod_Add + 998016 t76_leaf0 <= 1e+06
 T76_L0_f297_le: const_Sale_Condition_Normal + 999999 t76_leaf0 <= 1e+06
 T76_L0_f44_le: const_Yr_Sold + 997992 t76_leaf0 <= 1e+06
 T76_L1_f6_le: const_Year_Remod_Add + 998016 t76_leaf1 <= 1e+06
 T76_L1_f297_le: const_Sale_Condition_Normal + 999999 t76_leaf1 <= 1e+06
 T76_R1_f44_ge: const_Yr_Sold - 1.002008e+06 t76_leaf1 >= -1e+06
 T76_L2_f6_le: const_Year_Remod_Add + 998016 t76_leaf2 <= 1e+06
 T76_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t76_leaf2
   >= -1e+06
 T76_L2_f0_le: const_Lot_Frontage + 999928 t76_leaf2 <= 1e+06
 T76_L3_f6_le: const_Year_Remod_Add + 998016 t76_leaf3 <= 1e+06
 T76_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t76_leaf3
   >= -1e+06
 T76_R3_f0_ge: const_Lot_Frontage - 1.000072e+06 t76_leaf3 >= -1e+06
 T76_R4_f6_ge: const_Year_Remod_Add - 1.001984e+06 t76_leaf4 >= -1e+06
 T76_L4_f44_le: const_Yr_Sold + 997993 t76_leaf4 <= 1e+06
 T76_L4_f36_le: x_Wood_Deck_SF + 99807 t76_leaf4 <= 100000
 T76_R5_f6_ge: const_Year_Remod_Add - 1.001984e+06 t76_leaf5 >= -1e+06
 T76_L5_f44_le: const_Yr_Sold + 997993 t76_leaf5 <= 1e+06
 T76_R5_f36_ge: x_Wood_Deck_SF - 193 t76_leaf5 >= 0
 T76_R6_f6_ge: const_Year_Remod_Add - 1.001984e+06 t76_leaf6 >= -1e+06
 T76_R6_f44_ge: const_Yr_Sold - 1.002007e+06 t76_leaf6 >= -1e+06
 T76_L6_f28_le: x_TotRms_AbvGrd + 3 t76_leaf6 <= 15
 T76_R7_f6_ge: const_Year_Remod_Add - 1.001984e+06 t76_leaf7 >= -1e+06
 T76_R7_f44_ge: const_Yr_Sold - 1.002007e+06 t76_leaf7 >= -1e+06
 T76_R7_f28_ge: x_TotRms_AbvGrd - 12 t76_leaf7 >= 0
 TREE_77_ONEHOT: t77_leaf0 + t77_leaf1 + t77_leaf2 + t77_leaf3 + t77_leaf4
   + t77_leaf5 + t77_leaf6 = 1
 T77_L0_f4_le: const_Overall_Cond + 999995 t77_leaf0 <= 1e+06
 T77_L0_f37_le: x_Open_Porch_SF + 99842 t77_leaf0 <= 100000
 T77_L0_f14_le: x_Bsmt_Unf_SF + 99598 t77_leaf0 <= 100000
 T77_L1_f4_le: const_Overall_Cond + 999995 t77_leaf1 <= 1e+06
 T77_L1_f37_le: x_Open_Porch_SF + 99842 t77_leaf1 <= 100000
 T77_R1_f14_ge: x_Bsmt_Unf_SF - 402 t77_leaf1 >= 0
 T77_L2_f4_le: const_Overall_Cond + 999995 t77_leaf2 <= 1e+06
 T77_R2_f37_ge: x_Open_Porch_SF - 158 t77_leaf2 >= 0
 T77_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t77_leaf3 >= -1e+06
 T77_L3_f44_le: const_Yr_Sold + 997992 t77_leaf3 <= 1e+06
 T77_L3_f1_le: const_Lot_Area + 990410 t77_leaf3 <= 1e+06
 T77_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t77_leaf4 >= -1e+06
 T77_L4_f44_le: const_Yr_Sold + 997992 t77_leaf4 <= 1e+06
 T77_R4_f1_ge: const_Lot_Area - 1.00959e+06 t77_leaf4 >= -1e+06
 T77_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t77_leaf5 >= -1e+06
 T77_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t77_leaf5 >= -1e+06
 T77_L5_f21_le: const_Bsmt_Full_Bath + 999999 t77_leaf5 <= 1e+06
 T77_R6_f4_ge: const_Overall_Cond - 1.000005e+06 t77_leaf6 >= -1e+06
 T77_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t77_leaf6 >= -1e+06
 T77_R6_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t77_leaf6 >= -1e+06
 TREE_78_ONEHOT: t78_leaf0 + t78_leaf1 + t78_leaf2 + t78_leaf3 + t78_leaf4
   + t78_leaf5 + t78_leaf6 + t78_leaf7 = 1
 T78_L0_f44_le: const_Yr_Sold + 997992 t78_leaf0 <= 1e+06
 T78_L0_f20_le: x_Gr_Liv_Area + 8266 t78_leaf0 <= 10000
 T78_L0_f0_le: const_Lot_Frontage + 999969 t78_leaf0 <= 1e+06
 T78_L1_f44_le: const_Yr_Sold + 997992 t78_leaf1 <= 1e+06
 T78_L1_f20_le: x_Gr_Liv_Area + 8266 t78_leaf1 <= 10000
 T78_R1_f0_ge: const_Lot_Frontage - 1.000031e+06 t78_leaf1 >= -1e+06
 T78_L2_f44_le: const_Yr_Sold + 997992 t78_leaf2 <= 1e+06
 T78_R2_f20_ge: x_Gr_Liv_Area - 1734 t78_leaf2 >= 0
 T78_L2_f5_le: const_Year_Built + 998099 t78_leaf2 <= 1e+06
 T78_L3_f44_le: const_Yr_Sold + 997992 t78_leaf3 <= 1e+06
 T78_R3_f20_ge: x_Gr_Liv_Area - 1734 t78_leaf3 >= 0
 T78_R3_f5_ge: const_Year_Built - 1.001901e+06 t78_leaf3 >= -1e+06
 T78_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t78_leaf4 >= -1e+06
 T78_L4_f293_le: const_Sale_Condition_Abnorml + 999999 t78_leaf4 <= 1e+06
 T78_L4_f20_le: x_Gr_Liv_Area + 8882 t78_leaf4 <= 10000
 T78_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t78_leaf5 >= -1e+06
 T78_L5_f293_le: const_Sale_Condition_Abnorml + 999999 t78_leaf5 <= 1e+06
 T78_R5_f20_ge: x_Gr_Liv_Area - 1118 t78_leaf5 >= 0
 T78_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t78_leaf6 >= -1e+06
 T78_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t78_leaf6
   >= -1e+06
 T78_L6_f3_le: Overall_Qual_calc + 999995 t78_leaf6 <= 1e+06
 T78_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t78_leaf7 >= -1e+06
 T78_R7_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t78_leaf7
   >= -1e+06
 T78_R7_f3_ge: Overall_Qual_calc - 1.000005e+06 t78_leaf7 >= -1e+06
 TREE_79_ONEHOT: t79_leaf0 + t79_leaf1 + t79_leaf2 + t79_leaf3 + t79_leaf4
   + t79_leaf5 + t79_leaf6 = 1
 T79_L0_f4_le: const_Overall_Cond + 999996 t79_leaf0 <= 1e+06
 T79_L0_f25_le: x_Bedroom_AbvGr + 2 t79_leaf0 <= 6
 T79_L0_f16_le: x_Heating_QC + t79_leaf0 <= 4
 T79_L1_f4_le: const_Overall_Cond + 999996 t79_leaf1 <= 1e+06
 T79_L1_f25_le: x_Bedroom_AbvGr + 2 t79_leaf1 <= 6
 T79_R1_f16_ge: x_Heating_QC - 4 t79_leaf1 >= -1
 T79_L2_f4_le: const_Overall_Cond + 999996 t79_leaf2 <= 1e+06
 T79_R2_f25_ge: x_Bedroom_AbvGr - 4 t79_leaf2 >= 0
 T79_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t79_leaf3 >= -1e+06
 T79_L3_f95_le: const_Neighborhood_Crawfor + 999999 t79_leaf3 <= 1e+06
 T79_L3_f15_le: const_Total_Bsmt_SF + 998954 t79_leaf3 <= 1e+06
 T79_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t79_leaf4 >= -1e+06
 T79_L4_f95_le: const_Neighborhood_Crawfor + 999999 t79_leaf4 <= 1e+06
 T79_R4_f15_ge: const_Total_Bsmt_SF - 1.001046e+06 t79_leaf4 >= -1e+06
 T79_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t79_leaf5 >= -1e+06
 T79_R5_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t79_leaf5
   >= -1e+06
 T79_L5_f5_le: const_Year_Built + 998081 t79_leaf5 <= 1e+06
 T79_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t79_leaf6 >= -1e+06
 T79_R6_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t79_leaf6
   >= -1e+06
 T79_R6_f5_ge: const_Year_Built - 1.001919e+06 t79_leaf6 >= -1e+06
 TREE_80_ONEHOT: t80_leaf0 + t80_leaf1 + t80_leaf2 + t80_leaf3 + t80_leaf4
   + t80_leaf5 + t80_leaf6 = 1
 T80_L0_f4_le: const_Overall_Cond + 999995 t80_leaf0 <= 1e+06
 T80_L0_f40_le: x_Screen_Porch + 99896 t80_leaf0 <= 100000
 T80_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t80_leaf0 <= 1e+06
 T80_L1_f4_le: const_Overall_Cond + 999995 t80_leaf1 <= 1e+06
 T80_L1_f40_le: x_Screen_Porch + 99896 t80_leaf1 <= 100000
 T80_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t80_leaf1
   >= -1e+06
 T80_L2_f4_le: const_Overall_Cond + 999995 t80_leaf2 <= 1e+06
 T80_R2_f40_ge: x_Screen_Porch - 104 t80_leaf2 >= 0
 T80_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t80_leaf3 >= -1e+06
 T80_L3_f1_le: const_Lot_Area + 988100 t80_leaf3 <= 1e+06
 T80_L3_f12_le: x_BsmtFin_SF_1 + 999830 t80_leaf3 <= 1e+06
 T80_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t80_leaf4 >= -1e+06
 T80_L4_f1_le: const_Lot_Area + 988100 t80_leaf4 <= 1e+06
 T80_R4_f12_ge: x_BsmtFin_SF_1 - 170 t80_leaf4 >= 0
 T80_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t80_leaf5 >= -1e+06
 T80_R5_f1_ge: const_Lot_Area - 1.0119e+06 t80_leaf5 >= -1e+06
 T80_L5_f17_le: x_1st_Flr_SF + 3492 t80_leaf5 <= 5000
 T80_R6_f4_ge: const_Overall_Cond - 1.000005e+06 t80_leaf6 >= -1e+06
 T80_R6_f1_ge: const_Lot_Area - 1.0119e+06 t80_leaf6 >= -1e+06
 T80_R6_f17_ge: x_1st_Flr_SF - 1508 t80_leaf6 >= 0
 TREE_81_ONEHOT: t81_leaf0 + t81_leaf1 + t81_leaf2 + t81_leaf3 + t81_leaf4
   = 1
 T81_L0_f4_le: const_Overall_Cond + 999997 t81_leaf0 <= 1e+06
 T81_R1_f4_ge: const_Overall_Cond - 1.000003e+06 t81_leaf1 >= -1e+06
 T81_L1_f30_le: x_Fireplace_Qu + 3 t81_leaf1 <= 4
 T81_L1_f1_le: const_Lot_Area + 994024 t81_leaf1 <= 1e+06
 T81_R2_f4_ge: const_Overall_Cond - 1.000003e+06 t81_leaf2 >= -1e+06
 T81_L2_f30_le: x_Fireplace_Qu + 3 t81_leaf2 <= 4
 T81_R2_f1_ge: const_Lot_Area - 1.005976e+06 t81_leaf2 >= -1e+06
 T81_R3_f4_ge: const_Overall_Cond - 1.000003e+06 t81_leaf3 >= -1e+06
 T81_R3_f30_ge: x_Fireplace_Qu - 2 t81_leaf3 >= -1
 T81_L3_f1_le: const_Lot_Area + 982773 t81_leaf3 <= 1e+06
 T81_R4_f4_ge: const_Overall_Cond - 1.000003e+06 t81_leaf4 >= -1e+06
 T81_R4_f30_ge: x_Fireplace_Qu - 2 t81_leaf4 >= -1
 T81_R4_f1_ge: const_Lot_Area - 1.017227e+06 t81_leaf4 >= -1e+06
 TREE_82_ONEHOT: t82_leaf0 + t82_leaf1 + t82_leaf2 + t82_leaf3 + t82_leaf4
   + t82_leaf5 + t82_leaf6 = 1
 T82_L0_f20_le: x_Gr_Liv_Area + 8200 t82_leaf0 <= 10000
 T82_L0_f46_le: const_MS_SubClass_30 + 999999 t82_leaf0 <= 1e+06
 T82_L0_f12_le: x_BsmtFin_SF_1 + 999640 t82_leaf0 <= 1e+06
 T82_L1_f20_le: x_Gr_Liv_Area + 8200 t82_leaf1 <= 10000
 T82_L1_f46_le: const_MS_SubClass_30 + 999999 t82_leaf1 <= 1e+06
 T82_R1_f12_ge: x_BsmtFin_SF_1 - 360 t82_leaf1 >= 0
 T82_L2_f20_le: x_Gr_Liv_Area + 8200 t82_leaf2 <= 10000
 T82_R2_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t82_leaf2 >= -1e+06
 T82_L2_f21_le: const_Bsmt_Full_Bath + 999999 t82_leaf2 <= 1e+06
 T82_L3_f20_le: x_Gr_Liv_Area + 8200 t82_leaf3 <= 10000
 T82_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t82_leaf3 >= -1e+06
 T82_R3_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t82_leaf3 >= -1e+06
 T82_R4_f20_ge: x_Gr_Liv_Area - 1800 t82_leaf4 >= 0
 T82_L4_f40_le: x_Screen_Porch + 99960 t82_leaf4 <= 100000
 T82_L4_f23_le: x_Full_Bath + t82_leaf4 <= 4
 T82_R5_f20_ge: x_Gr_Liv_Area - 1800 t82_leaf5 >= 0
 T82_L5_f40_le: x_Screen_Porch + 99960 t82_leaf5 <= 100000
 T82_R5_f23_ge: x_Full_Bath - 3 t82_leaf5 >= 0
 T82_R6_f20_ge: x_Gr_Liv_Area - 1800 t82_leaf6 >= 0
 T82_R6_f40_ge: x_Screen_Porch - 40 t82_leaf6 >= 0
 TREE_83_ONEHOT: t83_leaf0 + t83_leaf1 + t83_leaf2 + t83_leaf3 + t83_leaf4
   = 1
 T83_L0_f3_le: Overall_Qual_calc + 999997 t83_leaf0 <= 1e+06
 T83_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t83_leaf1 >= -1e+06
 T83_L1_f15_le: const_Total_Bsmt_SF + 998947 t83_leaf1 <= 1e+06
 T83_L1_f1_le: const_Lot_Area + 994024 t83_leaf1 <= 1e+06
 T83_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t83_leaf2 >= -1e+06
 T83_L2_f15_le: const_Total_Bsmt_SF + 998947 t83_leaf2 <= 1e+06
 T83_R2_f1_ge: const_Lot_Area - 1.005976e+06 t83_leaf2 >= -1e+06
 T83_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t83_leaf3 >= -1e+06
 T83_R3_f15_ge: const_Total_Bsmt_SF - 1.001053e+06 t83_leaf3 >= -1e+06
 T83_L3_f7_le: x_Mas_Vnr_Area + 1544 t83_leaf3 <= 2000
 T83_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t83_leaf4 >= -1e+06
 T83_R4_f15_ge: const_Total_Bsmt_SF - 1.001053e+06 t83_leaf4 >= -1e+06
 T83_R4_f7_ge: x_Mas_Vnr_Area - 456 t83_leaf4 >= 0
 TREE_84_ONEHOT: t84_leaf0 + t84_leaf1 + t84_leaf2 + t84_leaf3 + t84_leaf4
   + t84_leaf5 + t84_leaf6 = 1
 T84_L0_f4_le: const_Overall_Cond + 999995 t84_leaf0 <= 1e+06
 T84_L0_f37_le: x_Open_Porch_SF + 99928 t84_leaf0 <= 100000
 T84_L0_f40_le: x_Screen_Porch + 99896 t84_leaf0 <= 100000
 T84_L1_f4_le: const_Overall_Cond + 999995 t84_leaf1 <= 1e+06
 T84_L1_f37_le: x_Open_Porch_SF + 99928 t84_leaf1 <= 100000
 T84_R1_f40_ge: x_Screen_Porch - 104 t84_leaf1 >= 0
 T84_L2_f4_le: const_Overall_Cond + 999995 t84_leaf2 <= 1e+06
 T84_R2_f37_ge: x_Open_Porch_SF - 72 t84_leaf2 >= 0
 T84_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t84_leaf3 >= -1e+06
 T84_L3_f20_le: x_Gr_Liv_Area + 9012 t84_leaf3 <= 10000
 T84_L3_f261_le: x_fence_is_GdWo <= 1
 T84_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t84_leaf4 >= -1e+06
 T84_L4_f20_le: x_Gr_Liv_Area + 9012 t84_leaf4 <= 10000
 T84_R4_f261_ge: x_fence_is_GdWo - t84_leaf4 >= 0
 T84_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t84_leaf5 >= -1e+06
 T84_R5_f20_ge: x_Gr_Liv_Area - 988 t84_leaf5 >= 0
 T84_L5_f23_le: x_Full_Bath + t84_leaf5 <= 4
 T84_R6_f4_ge: const_Overall_Cond - 1.000005e+06 t84_leaf6 >= -1e+06
 T84_R6_f20_ge: x_Gr_Liv_Area - 988 t84_leaf6 >= 0
 T84_R6_f23_ge: x_Full_Bath - 3 t84_leaf6 >= 0
 TREE_85_ONEHOT: t85_leaf0 + t85_leaf1 + t85_leaf2 + t85_leaf3 + t85_leaf4
   + t85_leaf5 + t85_leaf6 = 1
 T85_L0_f207_le: const_Bsmt_Exposure_Gd + 999999 t85_leaf0 <= 1e+06
 T85_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t85_leaf0 <= 1e+06
 T85_L0_f15_le: const_Total_Bsmt_SF + 999337 t85_leaf0 <= 1e+06
 T85_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t85_leaf1 <= 1e+06
 T85_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t85_leaf1 <= 1e+06
 T85_R1_f15_ge: const_Total_Bsmt_SF - 1.000663e+06 t85_leaf1 >= -1e+06
 T85_L2_f207_le: const_Bsmt_Exposure_Gd + 999999 t85_leaf2 <= 1e+06
 T85_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t85_leaf2
   >= -1e+06
 T85_L2_f3_le: Overall_Qual_calc + 999995 t85_leaf2 <= 1e+06
 T85_L3_f207_le: const_Bsmt_Exposure_Gd + 999999 t85_leaf3 <= 1e+06
 T85_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t85_leaf3
   >= -1e+06
 T85_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t85_leaf3 >= -1e+06
 T85_R4_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t85_leaf4 >= -1e+06
 T85_L4_f15_le: const_Total_Bsmt_SF + 998562 t85_leaf4 <= 1e+06
 T85_R5_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t85_leaf5 >= -1e+06
 T85_R5_f15_ge: const_Total_Bsmt_SF - 1.001438e+06 t85_leaf5 >= -1e+06
 T85_L5_f14_le: x_Bsmt_Unf_SF + 99004 t85_leaf5 <= 100000
 T85_R6_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t85_leaf6 >= -1e+06
 T85_R6_f15_ge: const_Total_Bsmt_SF - 1.001438e+06 t85_leaf6 >= -1e+06
 T85_R6_f14_ge: x_Bsmt_Unf_SF - 996 t85_leaf6 >= 0
 TREE_86_ONEHOT: t86_leaf0 + t86_leaf1 + t86_leaf2 + t86_leaf3 + t86_leaf4
   + t86_leaf5 + t86_leaf6 = 1
 T86_L0_f46_le: const_MS_SubClass_30 + 999999 t86_leaf0 <= 1e+06
 T86_L0_f15_le: const_Total_Bsmt_SF + 999424 t86_leaf0 <= 1e+06
 T86_L0_f253_le: const_Garage_Finish_Fin + 999999 t86_leaf0 <= 1e+06
 T86_L1_f46_le: const_MS_SubClass_30 + 999999 t86_leaf1 <= 1e+06
 T86_L1_f15_le: const_Total_Bsmt_SF + 999424 t86_leaf1 <= 1e+06
 T86_R1_f253_ge: const_Garage_Finish_Fin - 1.000001e+06 t86_leaf1
   >= -1e+06
 T86_L2_f46_le: const_MS_SubClass_30 + 999999 t86_leaf2 <= 1e+06
 T86_R2_f15_ge: const_Total_Bsmt_SF - 1.000576e+06 t86_leaf2 >= -1e+06
 T86_L2_f1_le: const_Lot_Area + 986189 t86_leaf2 <= 1e+06
 T86_L3_f46_le: const_MS_SubClass_30 + 999999 t86_leaf3 <= 1e+06
 T86_R3_f15_ge: const_Total_Bsmt_SF - 1.000576e+06 t86_leaf3 >= -1e+06
 T86_R3_f1_ge: const_Lot_Area - 1.013811e+06 t86_leaf3 >= -1e+06
 T86_R4_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t86_leaf4 >= -1e+06
 T86_L4_f15_le: const_Total_Bsmt_SF + 999239 t86_leaf4 <= 1e+06
 T86_L4_f15_le: const_Total_Bsmt_SF + 999337 t86_leaf4 <= 1e+06
 T86_R5_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t86_leaf5 >= -1e+06
 T86_L5_f15_le: const_Total_Bsmt_SF + 999239 t86_leaf5 <= 1e+06
 T86_R5_f15_ge: const_Total_Bsmt_SF - 1.000663e+06 t86_leaf5 >= -1e+06
 T86_R6_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t86_leaf6 >= -1e+06
 T86_R6_f15_ge: const_Total_Bsmt_SF - 1.000761e+06 t86_leaf6 >= -1e+06
 TREE_87_ONEHOT: t87_leaf0 + t87_leaf1 + t87_leaf2 + t87_leaf3 + t87_leaf4
   + t87_leaf5 + t87_leaf6 = 1
 T87_L0_f4_le: const_Overall_Cond + 999996 t87_leaf0 <= 1e+06
 T87_L0_f73_le: const_Lot_Shape_IR1 + 999999 t87_leaf0 <= 1e+06
 T87_L0_f226_le: heat_is_GasA + 999999 t87_leaf0 <= 1e+06
 T87_L1_f4_le: const_Overall_Cond + 999996 t87_leaf1 <= 1e+06
 T87_L1_f73_le: const_Lot_Shape_IR1 + 999999 t87_leaf1 <= 1e+06
 T87_R1_f226_ge: heat_is_GasA - 1.000001e+06 t87_leaf1 >= -1e+06
 T87_L2_f4_le: const_Overall_Cond + 999996 t87_leaf2 <= 1e+06
 T87_R2_f73_ge: const_Lot_Shape_IR1 - 1.000001e+06 t87_leaf2 >= -1e+06
 T87_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t87_leaf3 >= -1e+06
 T87_L3_f44_le: const_Yr_Sold + 997992 t87_leaf3 <= 1e+06
 T87_L3_f18_le: const_2nd_Flr_SF + 998595 t87_leaf3 <= 1e+06
 T87_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t87_leaf4 >= -1e+06
 T87_L4_f44_le: const_Yr_Sold + 997992 t87_leaf4 <= 1e+06
 T87_R4_f18_ge: const_2nd_Flr_SF - 1.001405e+06 t87_leaf4 >= -1e+06
 T87_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t87_leaf5 >= -1e+06
 T87_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t87_leaf5 >= -1e+06
 T87_L5_f20_le: x_Gr_Liv_Area + 8854 t87_leaf5 <= 10000
 T87_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t87_leaf6 >= -1e+06
 T87_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t87_leaf6 >= -1e+06
 T87_R6_f20_ge: x_Gr_Liv_Area - 1146 t87_leaf6 >= 0
 TREE_88_ONEHOT: t88_leaf0 + t88_leaf1 + t88_leaf2 + t88_leaf3 + t88_leaf4
   + t88_leaf5 + t88_leaf6 = 1
 T88_L0_f4_le: const_Overall_Cond + 999996 t88_leaf0 <= 1e+06
 T88_L0_f15_le: const_Total_Bsmt_SF + 998600 t88_leaf0 <= 1e+06
 T88_L0_f297_le: const_Sale_Condition_Normal + 999999 t88_leaf0 <= 1e+06
 T88_L1_f4_le: const_Overall_Cond + 999996 t88_leaf1 <= 1e+06
 T88_L1_f15_le: const_Total_Bsmt_SF + 998600 t88_leaf1 <= 1e+06
 T88_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t88_leaf1
   >= -1e+06
 T88_L2_f4_le: const_Overall_Cond + 999996 t88_leaf2 <= 1e+06
 T88_R2_f15_ge: const_Total_Bsmt_SF - 1.0014e+06 t88_leaf2 >= -1e+06
 T88_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t88_leaf3 >= -1e+06
 T88_L3_f44_le: const_Yr_Sold + 997992 t88_leaf3 <= 1e+06
 T88_L3_f17_le: x_1st_Flr_SF + 3260 t88_leaf3 <= 5000
 T88_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t88_leaf4 >= -1e+06
 T88_L4_f44_le: const_Yr_Sold + 997992 t88_leaf4 <= 1e+06
 T88_R4_f17_ge: x_1st_Flr_SF - 1740 t88_leaf4 >= 0
 T88_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t88_leaf5 >= -1e+06
 T88_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t88_leaf5 >= -1e+06
 T88_L5_f15_le: const_Total_Bsmt_SF + 999235 t88_leaf5 <= 1e+06
 T88_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t88_leaf6 >= -1e+06
 T88_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t88_leaf6 >= -1e+06
 T88_R6_f15_ge: const_Total_Bsmt_SF - 1.000765e+06 t88_leaf6 >= -1e+06
 TREE_89_ONEHOT: t89_leaf0 + t89_leaf1 + t89_leaf2 + t89_leaf3 + t89_leaf4
   + t89_leaf5 + t89_leaf6 + t89_leaf7 = 1
 T89_L0_f11_le: x_Bsmt_Cond + 2 t89_leaf0 <= 4
 T89_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t89_leaf0 <= 1e+06
 T89_L0_f38_le: x_Enclosed_Porch + 99808 t89_leaf0 <= 100000
 T89_L1_f11_le: x_Bsmt_Cond + 2 t89_leaf1 <= 4
 T89_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t89_leaf1 <= 1e+06
 T89_R1_f38_ge: x_Enclosed_Porch - 192 t89_leaf1 >= 0
 T89_L2_f11_le: x_Bsmt_Cond + 2 t89_leaf2 <= 4
 T89_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t89_leaf2
   >= -1e+06
 T89_L2_f28_le: x_TotRms_AbvGrd + 9 t89_leaf2 <= 15
 T89_L3_f11_le: x_Bsmt_Cond + 2 t89_leaf3 <= 4
 T89_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t89_leaf3
   >= -1e+06
 T89_R3_f28_ge: x_TotRms_AbvGrd - 6 t89_leaf3 >= 0
 T89_R4_f11_ge: x_Bsmt_Cond - 3 t89_leaf4 >= -1
 T89_L4_f20_le: x_Gr_Liv_Area + 8251 t89_leaf4 <= 10000
 T89_L4_f117_le: const_Condition_1_Artery + 999999 t89_leaf4 <= 1e+06
 T89_R5_f11_ge: x_Bsmt_Cond - 3 t89_leaf5 >= -1
 T89_L5_f20_le: x_Gr_Liv_Area + 8251 t89_leaf5 <= 10000
 T89_R5_f117_ge: const_Condition_1_Artery - 1.000001e+06 t89_leaf5
   >= -1e+06
 T89_R6_f11_ge: x_Bsmt_Cond - 3 t89_leaf6 >= -1
 T89_R6_f20_ge: x_Gr_Liv_Area - 1749 t89_leaf6 >= 0
 T89_L6_f40_le: x_Screen_Porch + 99960 t89_leaf6 <= 100000
 T89_R7_f11_ge: x_Bsmt_Cond - 3 t89_leaf7 >= -1
 T89_R7_f20_ge: x_Gr_Liv_Area - 1749 t89_leaf7 >= 0
 T89_R7_f40_ge: x_Screen_Porch - 40 t89_leaf7 >= 0
 TREE_90_ONEHOT: t90_leaf0 + t90_leaf1 + t90_leaf2 + t90_leaf3 + t90_leaf4
   + t90_leaf5 + t90_leaf6 + t90_leaf7 = 1
 T90_L0_f44_le: const_Yr_Sold + 997992 t90_leaf0 <= 1e+06
 T90_L0_f134_le: const_Bldg_Type_1Fam + 999999 t90_leaf0 <= 1e+06
 T90_L0_f37_le: x_Open_Porch_SF + 99996 t90_leaf0 <= 100000
 T90_L1_f44_le: const_Yr_Sold + 997992 t90_leaf1 <= 1e+06
 T90_L1_f134_le: const_Bldg_Type_1Fam + 999999 t90_leaf1 <= 1e+06
 T90_R1_f37_ge: x_Open_Porch_SF - 4 t90_leaf1 >= 0
 T90_L2_f44_le: const_Yr_Sold + 997992 t90_leaf2 <= 1e+06
 T90_R2_f134_ge: const_Bldg_Type_1Fam - 1.000001e+06 t90_leaf2 >= -1e+06
 T90_L2_f15_le: const_Total_Bsmt_SF + 998311 t90_leaf2 <= 1e+06
 T90_L3_f44_le: const_Yr_Sold + 997992 t90_leaf3 <= 1e+06
 T90_R3_f134_ge: const_Bldg_Type_1Fam - 1.000001e+06 t90_leaf3 >= -1e+06
 T90_R3_f15_ge: const_Total_Bsmt_SF - 1.001689e+06 t90_leaf3 >= -1e+06
 T90_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t90_leaf4 >= -1e+06
 T90_L4_f293_le: const_Sale_Condition_Abnorml + 999999 t90_leaf4 <= 1e+06
 T90_L4_f37_le: x_Open_Porch_SF + 99969 t90_leaf4 <= 100000
 T90_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t90_leaf5 >= -1e+06
 T90_L5_f293_le: const_Sale_Condition_Abnorml + 999999 t90_leaf5 <= 1e+06
 T90_R5_f37_ge: x_Open_Porch_SF - 31 t90_leaf5 >= 0
 T90_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t90_leaf6 >= -1e+06
 T90_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t90_leaf6
   >= -1e+06
 T90_L6_f85_le: const_Lot_Config_Inside + 999999 t90_leaf6 <= 1e+06
 T90_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t90_leaf7 >= -1e+06
 T90_R7_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t90_leaf7
   >= -1e+06
 T90_R7_f85_ge: const_Lot_Config_Inside - 1.000001e+06 t90_leaf7 >= -1e+06
 TREE_91_ONEHOT: t91_leaf0 + t91_leaf1 + t91_leaf2 + t91_leaf3 + t91_leaf4
   + t91_leaf5 + t91_leaf6 + t91_leaf7 = 1
 T91_L0_f12_le: x_BsmtFin_SF_1 + 999708 t91_leaf0 <= 1e+06
 T91_L0_f16_le: x_Heating_QC + t91_leaf0 <= 4
 T91_L0_f66_le: const_MS_Zoning_RL + 999999 t91_leaf0 <= 1e+06
 T91_L1_f12_le: x_BsmtFin_SF_1 + 999708 t91_leaf1 <= 1e+06
 T91_L1_f16_le: x_Heating_QC + t91_leaf1 <= 4
 T91_R1_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t91_leaf1 >= -1e+06
 T91_L2_f12_le: x_BsmtFin_SF_1 + 999708 t91_leaf2 <= 1e+06
 T91_R2_f16_ge: x_Heating_QC - 4 t91_leaf2 >= -1
 T91_L2_f1_le: const_Lot_Area + 992560 t91_leaf2 <= 1e+06
 T91_L3_f12_le: x_BsmtFin_SF_1 + 999708 t91_leaf3 <= 1e+06
 T91_R3_f16_ge: x_Heating_QC - 4 t91_leaf3 >= -1
 T91_R3_f1_ge: const_Lot_Area - 1.00744e+06 t91_leaf3 >= -1e+06
 T91_R4_f12_ge: x_BsmtFin_SF_1 - 292 t91_leaf4 >= 0
 T91_L4_f15_le: const_Total_Bsmt_SF + 998581 t91_leaf4 <= 1e+06
 T91_L4_f1_le: const_Lot_Area + 993875 t91_leaf4 <= 1e+06
 T91_R5_f12_ge: x_BsmtFin_SF_1 - 292 t91_leaf5 >= 0
 T91_L5_f15_le: const_Total_Bsmt_SF + 998581 t91_leaf5 <= 1e+06
 T91_R5_f1_ge: const_Lot_Area - 1.006125e+06 t91_leaf5 >= -1e+06
 T91_R6_f12_ge: x_BsmtFin_SF_1 - 292 t91_leaf6 >= 0
 T91_R6_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t91_leaf6 >= -1e+06
 T91_L6_f118_le: const_Condition_1_Feedr + 999999 t91_leaf6 <= 1e+06
 T91_R7_f12_ge: x_BsmtFin_SF_1 - 292 t91_leaf7 >= 0
 T91_R7_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t91_leaf7 >= -1e+06
 T91_R7_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t91_leaf7
   >= -1e+06
 TREE_92_ONEHOT: t92_leaf0 + t92_leaf1 + t92_leaf2 + t92_leaf3 + t92_leaf4
   + t92_leaf5 + t92_leaf6 = 1
 T92_L0_f4_le: const_Overall_Cond + 999996 t92_leaf0 <= 1e+06
 T92_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t92_leaf0 <= 1e+06
 T92_L0_f231_le: central_air_no + 999999 t92_leaf0 <= 1e+06
 T92_L1_f4_le: const_Overall_Cond + 999996 t92_leaf1 <= 1e+06
 T92_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t92_leaf1 <= 1e+06
 T92_R1_f231_ge: central_air_no - 1.000001e+06 t92_leaf1 >= -1e+06
 T92_L2_f4_le: const_Overall_Cond + 999996 t92_leaf2 <= 1e+06
 T92_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t92_leaf2
   >= -1e+06
 T92_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t92_leaf3 >= -1e+06
 T92_L3_f23_le: x_Full_Bath + t92_leaf3 <= 4
 T92_L3_f44_le: const_Yr_Sold + 997992 t92_leaf3 <= 1e+06
 T92_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t92_leaf4 >= -1e+06
 T92_L4_f23_le: x_Full_Bath + t92_leaf4 <= 4
 T92_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t92_leaf4 >= -1e+06
 T92_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t92_leaf5 >= -1e+06
 T92_R5_f23_ge: x_Full_Bath - 3 t92_leaf5 >= 0
 T92_L5_f174_le: x_ex1_is_VinylSd <= 1
 T92_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t92_leaf6 >= -1e+06
 T92_R6_f23_ge: x_Full_Bath - 3 t92_leaf6 >= 0
 T92_R6_f174_ge: x_ex1_is_VinylSd - t92_leaf6 >= 0
 TREE_93_ONEHOT: t93_leaf0 + t93_leaf1 + t93_leaf2 + t93_leaf3 + t93_leaf4
   + t93_leaf5 + t93_leaf6 + t93_leaf7 = 1
 T93_L0_f44_le: const_Yr_Sold + 997992 t93_leaf0 <= 1e+06
 T93_L0_f20_le: x_Gr_Liv_Area + 8032 t93_leaf0 <= 10000
 T93_L0_f4_le: const_Overall_Cond + 999996 t93_leaf0 <= 1e+06
 T93_L1_f44_le: const_Yr_Sold + 997992 t93_leaf1 <= 1e+06
 T93_L1_f20_le: x_Gr_Liv_Area + 8032 t93_leaf1 <= 10000
 T93_R1_f4_ge: const_Overall_Cond - 1.000004e+06 t93_leaf1 >= -1e+06
 T93_L2_f44_le: const_Yr_Sold + 997992 t93_leaf2 <= 1e+06
 T93_R2_f20_ge: x_Gr_Liv_Area - 1968 t93_leaf2 >= 0
 T93_L2_f267_le: const_Misc_Feature_No_aplica + 999999 t93_leaf2 <= 1e+06
 T93_L3_f44_le: const_Yr_Sold + 997992 t93_leaf3 <= 1e+06
 T93_R3_f20_ge: x_Gr_Liv_Area - 1968 t93_leaf3 >= 0
 T93_R3_f267_ge: const_Misc_Feature_No_aplica - 1.000001e+06 t93_leaf3
   >= -1e+06
 T93_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t93_leaf4 >= -1e+06
 T93_L4_f77_le: const_Land_Contour_Bnk + 999999 t93_leaf4 <= 1e+06
 T93_L4_f33_le: x_Garage_Area + 99649 t93_leaf4 <= 100000
 T93_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t93_leaf5 >= -1e+06
 T93_L5_f77_le: const_Land_Contour_Bnk + 999999 t93_leaf5 <= 1e+06
 T93_R5_f33_ge: x_Garage_Area - 351 t93_leaf5 >= 0
 T93_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t93_leaf6 >= -1e+06
 T93_R6_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t93_leaf6 >= -1e+06
 T93_L6_f17_le: x_1st_Flr_SF + 3983 t93_leaf6 <= 5000
 T93_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t93_leaf7 >= -1e+06
 T93_R7_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t93_leaf7 >= -1e+06
 T93_R7_f17_ge: x_1st_Flr_SF - 1017 t93_leaf7 >= 0
 TREE_94_ONEHOT: t94_leaf0 + t94_leaf1 + t94_leaf2 + t94_leaf3 + t94_leaf4
   + t94_leaf5 + t94_leaf6 = 1
 T94_L0_f29_le: const_Fireplaces + 999998 t94_leaf0 <= 1e+06
 T94_L0_f298_le: const_Sale_Condition_Partial + 999999 t94_leaf0 <= 1e+06
 T94_L0_f297_le: const_Sale_Condition_Normal + 999999 t94_leaf0 <= 1e+06
 T94_L1_f29_le: const_Fireplaces + 999998 t94_leaf1 <= 1e+06
 T94_L1_f298_le: const_Sale_Condition_Partial + 999999 t94_leaf1 <= 1e+06
 T94_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t94_leaf1
   >= -1e+06
 T94_L2_f29_le: const_Fireplaces + 999998 t94_leaf2 <= 1e+06
 T94_R2_f298_ge: const_Sale_Condition_Partial - 1.000001e+06 t94_leaf2
   >= -1e+06
 T94_L2_f14_le: x_Bsmt_Unf_SF + 99362 t94_leaf2 <= 100000
 T94_L3_f29_le: const_Fireplaces + 999998 t94_leaf3 <= 1e+06
 T94_R3_f298_ge: const_Sale_Condition_Partial - 1.000001e+06 t94_leaf3
   >= -1e+06
 T94_R3_f14_ge: x_Bsmt_Unf_SF - 638 t94_leaf3 >= 0
 T94_R4_f29_ge: const_Fireplaces - 1.000002e+06 t94_leaf4 >= -1e+06
 T94_L4_f43_le: const_Misc_Val + 999946 t94_leaf4 <= 1e+06
 T94_L4_f20_le: x_Gr_Liv_Area + 7866 t94_leaf4 <= 10000
 T94_R5_f29_ge: const_Fireplaces - 1.000002e+06 t94_leaf5 >= -1e+06
 T94_L5_f43_le: const_Misc_Val + 999946 t94_leaf5 <= 1e+06
 T94_R5_f20_ge: x_Gr_Liv_Area - 2134 t94_leaf5 >= 0
 T94_R6_f29_ge: const_Fireplaces - 1.000002e+06 t94_leaf6 >= -1e+06
 T94_R6_f43_ge: const_Misc_Val - 1.000054e+06 t94_leaf6 >= -1e+06
 TREE_95_ONEHOT: t95_leaf0 + t95_leaf1 + t95_leaf2 + t95_leaf3 + t95_leaf4
   + t95_leaf5 + t95_leaf6 + t95_leaf7 = 1
 T95_L0_f31_le: const_Garage_Yr_Blt + 998071 t95_leaf0 <= 1e+06
 T95_L0_f20_le: x_Gr_Liv_Area + 8800 t95_leaf0 <= 10000
 T95_L0_f5_le: const_Year_Built + 998076 t95_leaf0 <= 1e+06
 T95_L1_f31_le: const_Garage_Yr_Blt + 998071 t95_leaf1 <= 1e+06
 T95_L1_f20_le: x_Gr_Liv_Area + 8800 t95_leaf1 <= 10000
 T95_R1_f5_ge: const_Year_Built - 1.001924e+06 t95_leaf1 >= -1e+06
 T95_L2_f31_le: const_Garage_Yr_Blt + 998071 t95_leaf2 <= 1e+06
 T95_R2_f20_ge: x_Gr_Liv_Area - 1200 t95_leaf2 >= 0
 T95_L2_f5_le: const_Year_Built + 998089 t95_leaf2 <= 1e+06
 T95_L3_f31_le: const_Garage_Yr_Blt + 998071 t95_leaf3 <= 1e+06
 T95_R3_f20_ge: x_Gr_Liv_Area - 1200 t95_leaf3 >= 0
 T95_R3_f5_ge: const_Year_Built - 1.001911e+06 t95_leaf3 >= -1e+06
 T95_R4_f31_ge: const_Garage_Yr_Blt - 1.001929e+06 t95_leaf4 >= -1e+06
 T95_L4_f6_le: const_Year_Remod_Add + 998005 t95_leaf4 <= 1e+06
 T95_L4_f13_le: x_BsmtFin_SF_2 + 99468 t95_leaf4 <= 100000
 T95_R5_f31_ge: const_Garage_Yr_Blt - 1.001929e+06 t95_leaf5 >= -1e+06
 T95_L5_f6_le: const_Year_Remod_Add + 998005 t95_leaf5 <= 1e+06
 T95_R5_f13_ge: x_BsmtFin_SF_2 - 532 t95_leaf5 >= 0
 T95_R6_f31_ge: const_Garage_Yr_Blt - 1.001929e+06 t95_leaf6 >= -1e+06
 T95_R6_f6_ge: const_Year_Remod_Add - 1.001995e+06 t95_leaf6 >= -1e+06
 T95_L6_f20_le: x_Gr_Liv_Area + 8032 t95_leaf6 <= 10000
 T95_R7_f31_ge: const_Garage_Yr_Blt - 1.001929e+06 t95_leaf7 >= -1e+06
 T95_R7_f6_ge: const_Year_Remod_Add - 1.001995e+06 t95_leaf7 >= -1e+06
 T95_R7_f20_ge: x_Gr_Liv_Area - 1968 t95_leaf7 >= 0
 TREE_96_ONEHOT: t96_leaf0 + t96_leaf1 + t96_leaf2 + t96_leaf3 + t96_leaf4
   + t96_leaf5 + t96_leaf6 = 1
 T96_L0_f9_le: x_Exter_Cond + 2 t96_leaf0 <= 4
 T96_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t96_leaf0 <= 1e+06
 T96_L0_f31_le: const_Garage_Yr_Blt + 998079 t96_leaf0 <= 1e+06
 T96_L1_f9_le: x_Exter_Cond + 2 t96_leaf1 <= 4
 T96_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t96_leaf1 <= 1e+06
 T96_R1_f31_ge: const_Garage_Yr_Blt - 1.001921e+06 t96_leaf1 >= -1e+06
 T96_L2_f9_le: x_Exter_Cond + 2 t96_leaf2 <= 4
 T96_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t96_leaf2
   >= -1e+06
 T96_R3_f9_ge: x_Exter_Cond - 3 t96_leaf3 >= -1
 T96_L3_f20_le: x_Gr_Liv_Area + 9012 t96_leaf3 <= 10000
 T96_L3_f4_le: const_Overall_Cond + 999993 t96_leaf3 <= 1e+06
 T96_R4_f9_ge: x_Exter_Cond - 3 t96_leaf4 >= -1
 T96_L4_f20_le: x_Gr_Liv_Area + 9012 t96_leaf4 <= 10000
 T96_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t96_leaf4 >= -1e+06
 T96_R5_f9_ge: x_Exter_Cond - 3 t96_leaf5 >= -1
 T96_R5_f20_ge: x_Gr_Liv_Area - 988 t96_leaf5 >= 0
 T96_L5_f44_le: const_Yr_Sold + 997993 t96_leaf5 <= 1e+06
 T96_R6_f9_ge: x_Exter_Cond - 3 t96_leaf6 >= -1
 T96_R6_f20_ge: x_Gr_Liv_Area - 988 t96_leaf6 >= 0
 T96_R6_f44_ge: const_Yr_Sold - 1.002007e+06 t96_leaf6 >= -1e+06
 TREE_97_ONEHOT: t97_leaf0 + t97_leaf1 + t97_leaf2 + t97_leaf3 + t97_leaf4
   + t97_leaf5 + t97_leaf6 + t97_leaf7 = 1
 T97_L0_f46_le: const_MS_SubClass_30 + 999999 t97_leaf0 <= 1e+06
 T97_L0_f95_le: const_Neighborhood_Crawfor + 999999 t97_leaf0 <= 1e+06
 T97_L0_f16_le: x_Heating_QC + t97_leaf0 <= 4
 T97_L1_f46_le: const_MS_SubClass_30 + 999999 t97_leaf1 <= 1e+06
 T97_L1_f95_le: const_Neighborhood_Crawfor + 999999 t97_leaf1 <= 1e+06
 T97_R1_f16_ge: x_Heating_QC - 4 t97_leaf1 >= -1
 T97_L2_f46_le: const_MS_SubClass_30 + 999999 t97_leaf2 <= 1e+06
 T97_R2_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t97_leaf2
   >= -1e+06
 T97_L2_f20_le: x_Gr_Liv_Area + 8496 t97_leaf2 <= 10000
 T97_L3_f46_le: const_MS_SubClass_30 + 999999 t97_leaf3 <= 1e+06
 T97_R3_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t97_leaf3
   >= -1e+06
 T97_R3_f20_ge: x_Gr_Liv_Area - 1504 t97_leaf3 >= 0
 T97_R4_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t97_leaf4 >= -1e+06
 T97_L4_f16_le: x_Heating_QC + t97_leaf4 <= 4
 T97_L4_f5_le: const_Year_Built + 998075 t97_leaf4 <= 1e+06
 T97_R5_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t97_leaf5 >= -1e+06
 T97_L5_f16_le: x_Heating_QC + t97_leaf5 <= 4
 T97_R5_f5_ge: const_Year_Built - 1.001925e+06 t97_leaf5 >= -1e+06
 T97_R6_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t97_leaf6 >= -1e+06
 T97_R6_f16_ge: x_Heating_QC - 4 t97_leaf6 >= -1
 T97_L6_f25_le: x_Bedroom_AbvGr + 3 t97_leaf6 <= 6
 T97_R7_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t97_leaf7 >= -1e+06
 T97_R7_f16_ge: x_Heating_QC - 4 t97_leaf7 >= -1
 T97_R7_f25_ge: x_Bedroom_AbvGr - 3 t97_leaf7 >= 0
 TREE_98_ONEHOT: t98_leaf0 + t98_leaf1 + t98_leaf2 + t98_leaf3 + t98_leaf4
   + t98_leaf5 + t98_leaf6 = 1
 T98_L0_f4_le: const_Overall_Cond + 999996 t98_leaf0 <= 1e+06
 T98_L0_f16_le: x_Heating_QC <= 4
 T98_L0_f44_le: const_Yr_Sold + 997993 t98_leaf0 <= 1e+06
 T98_L1_f4_le: const_Overall_Cond + 999996 t98_leaf1 <= 1e+06
 T98_L1_f16_le: x_Heating_QC <= 4
 T98_R1_f44_ge: const_Yr_Sold - 1.002007e+06 t98_leaf1 >= -1e+06
 T98_L2_f4_le: const_Overall_Cond + 999996 t98_leaf2 <= 1e+06
 T98_R2_f16_ge: x_Heating_QC - 5 t98_leaf2 >= -1
 T98_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t98_leaf3 >= -1e+06
 T98_L3_f20_le: x_Gr_Liv_Area + 7351 t98_leaf3 <= 10000
 T98_L3_f5_le: const_Year_Built + 997994 t98_leaf3 <= 1e+06
 T98_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t98_leaf4 >= -1e+06
 T98_L4_f20_le: x_Gr_Liv_Area + 7351 t98_leaf4 <= 10000
 T98_R4_f5_ge: const_Year_Built - 1.002006e+06 t98_leaf4 >= -1e+06
 T98_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t98_leaf5 >= -1e+06
 T98_R5_f20_ge: x_Gr_Liv_Area - 2649 t98_leaf5 >= 0
 T98_L5_f25_le: x_Bedroom_AbvGr + 2 t98_leaf5 <= 6
 T98_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t98_leaf6 >= -1e+06
 T98_R6_f20_ge: x_Gr_Liv_Area - 2649 t98_leaf6 >= 0
 T98_R6_f25_ge: x_Bedroom_AbvGr - 4 t98_leaf6 >= 0
 TREE_99_ONEHOT: t99_leaf0 + t99_leaf1 + t99_leaf2 + t99_leaf3 + t99_leaf4
   + t99_leaf5 + t99_leaf6 = 1
 T99_L0_f95_le: const_Neighborhood_Crawfor + 999999 t99_leaf0 <= 1e+06
 T99_L0_f12_le: x_BsmtFin_SF_1 + 999708 t99_leaf0 <= 1e+06
 T99_L0_f6_le: const_Year_Remod_Add + 998016 t99_leaf0 <= 1e+06
 T99_L1_f95_le: const_Neighborhood_Crawfor + 999999 t99_leaf1 <= 1e+06
 T99_L1_f12_le: x_BsmtFin_SF_1 + 999708 t99_leaf1 <= 1e+06
 T99_R1_f6_ge: const_Year_Remod_Add - 1.001984e+06 t99_leaf1 >= -1e+06
 T99_L2_f95_le: const_Neighborhood_Crawfor + 999999 t99_leaf2 <= 1e+06
 T99_R2_f12_ge: x_BsmtFin_SF_1 - 292 t99_leaf2 >= 0
 T99_L2_f6_le: const_Year_Remod_Add + 998006 t99_leaf2 <= 1e+06
 T99_L3_f95_le: const_Neighborhood_Crawfor + 999999 t99_leaf3 <= 1e+06
 T99_R3_f12_ge: x_BsmtFin_SF_1 - 292 t99_leaf3 >= 0
 T99_R3_f6_ge: const_Year_Remod_Add - 1.001994e+06 t99_leaf3 >= -1e+06
 T99_R4_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t99_leaf4
   >= -1e+06
 T99_L4_f14_le: x_Bsmt_Unf_SF + 99590 t99_leaf4 <= 100000
 T99_R5_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t99_leaf5
   >= -1e+06
 T99_R5_f14_ge: x_Bsmt_Unf_SF - 410 t99_leaf5 >= 0
 T99_L5_f4_le: const_Overall_Cond + 999994 t99_leaf5 <= 1e+06
 T99_R6_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t99_leaf6
   >= -1e+06
 T99_R6_f14_ge: x_Bsmt_Unf_SF - 410 t99_leaf6 >= 0
 T99_R6_f4_ge: const_Overall_Cond - 1.000006e+06 t99_leaf6 >= -1e+06
 TREE_100_ONEHOT: t100_leaf0 + t100_leaf1 + t100_leaf2 + t100_leaf3
   + t100_leaf4 + t100_leaf5 + t100_leaf6 + t100_leaf7 = 1
 T100_L0_f20_le: x_Gr_Liv_Area + 9167 t100_leaf0 <= 10000
 T100_L0_f100_le: const_Neighborhood_IDOTRR + 999999 t100_leaf0 <= 1e+06
 T100_L0_f15_le: const_Total_Bsmt_SF + 999352 t100_leaf0 <= 1e+06
 T100_L1_f20_le: x_Gr_Liv_Area + 9167 t100_leaf1 <= 10000
 T100_L1_f100_le: const_Neighborhood_IDOTRR + 999999 t100_leaf1 <= 1e+06
 T100_R1_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t100_leaf1 >= -1e+06
 T100_L2_f20_le: x_Gr_Liv_Area + 9167 t100_leaf2 <= 10000
 T100_R2_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t100_leaf2
   >= -1e+06
 T100_L2_f217_le: x_b1_is_Unf <= 1
 T100_L3_f20_le: x_Gr_Liv_Area + 9167 t100_leaf3 <= 10000
 T100_R3_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t100_leaf3
   >= -1e+06
 T100_R3_f217_ge: x_b1_is_Unf - t100_leaf3 >= 0
 T100_R4_f20_ge: x_Gr_Liv_Area - 833 t100_leaf4 >= 0
 T100_L4_f21_le: const_Bsmt_Full_Bath + 999999 t100_leaf4 <= 1e+06
 T100_L4_f245_le: const_FunctioNo_aplical_Typ + 999999 t100_leaf4 <= 1e+06
 T100_R5_f20_ge: x_Gr_Liv_Area - 833 t100_leaf5 >= 0
 T100_L5_f21_le: const_Bsmt_Full_Bath + 999999 t100_leaf5 <= 1e+06
 T100_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t100_leaf5
   >= -1e+06
 T100_R6_f20_ge: x_Gr_Liv_Area - 833 t100_leaf6 >= 0
 T100_R6_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t100_leaf6 >= -1e+06
 T100_L6_f33_le: x_Garage_Area + 98846 t100_leaf6 <= 100000
 T100_R7_f20_ge: x_Gr_Liv_Area - 833 t100_leaf7 >= 0
 T100_R7_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t100_leaf7 >= -1e+06
 T100_R7_f33_ge: x_Garage_Area - 1154 t100_leaf7 >= 0
 TREE_101_ONEHOT: t101_leaf0 + t101_leaf1 + t101_leaf2 + t101_leaf3
   + t101_leaf4 + t101_leaf5 + t101_leaf6 = 1
 T101_L0_f4_le: const_Overall_Cond + 999996 t101_leaf0 <= 1e+06
 T101_L0_f14_le: x_Bsmt_Unf_SF + 99630 t101_leaf0 <= 100000
 T101_L1_f4_le: const_Overall_Cond + 999996 t101_leaf1 <= 1e+06
 T101_R1_f14_ge: x_Bsmt_Unf_SF - 370 t101_leaf1 >= 0
 T101_L1_f275_le: const_Mo_Sold_5 + 999999 t101_leaf1 <= 1e+06
 T101_L2_f4_le: const_Overall_Cond + 999996 t101_leaf2 <= 1e+06
 T101_R2_f14_ge: x_Bsmt_Unf_SF - 370 t101_leaf2 >= 0
 T101_R2_f275_ge: const_Mo_Sold_5 - 1.000001e+06 t101_leaf2 >= -1e+06
 T101_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t101_leaf3 >= -1e+06
 T101_L3_f95_le: const_Neighborhood_Crawfor + 999999 t101_leaf3 <= 1e+06
 T101_L3_f33_le: x_Garage_Area + 98846 t101_leaf3 <= 100000
 T101_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t101_leaf4 >= -1e+06
 T101_L4_f95_le: const_Neighborhood_Crawfor + 999999 t101_leaf4 <= 1e+06
 T101_R4_f33_ge: x_Garage_Area - 1154 t101_leaf4 >= 0
 T101_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t101_leaf5 >= -1e+06
 T101_R5_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t101_leaf5
   >= -1e+06
 T101_L5_f20_le: x_Gr_Liv_Area + 8496 t101_leaf5 <= 10000
 T101_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t101_leaf6 >= -1e+06
 T101_R6_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t101_leaf6
   >= -1e+06
 T101_R6_f20_ge: x_Gr_Liv_Area - 1504 t101_leaf6 >= 0
 TREE_102_ONEHOT: t102_leaf0 + t102_leaf1 + t102_leaf2 + t102_leaf3
   + t102_leaf4 + t102_leaf5 + t102_leaf6 + t102_leaf7 = 1
 T102_L0_f3_le: Overall_Qual_calc + 999995 t102_leaf0 <= 1e+06
 T102_L0_f100_le: const_Neighborhood_IDOTRR + 999999 t102_leaf0 <= 1e+06
 T102_L0_f37_le: x_Open_Porch_SF + 99961 t102_leaf0 <= 100000
 T102_L1_f3_le: Overall_Qual_calc + 999995 t102_leaf1 <= 1e+06
 T102_L1_f100_le: const_Neighborhood_IDOTRR + 999999 t102_leaf1 <= 1e+06
 T102_R1_f37_ge: x_Open_Porch_SF - 39 t102_leaf1 >= 0
 T102_L2_f3_le: Overall_Qual_calc + 999995 t102_leaf2 <= 1e+06
 T102_R2_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t102_leaf2
   >= -1e+06
 T102_L2_f44_le: const_Yr_Sold + 997993 t102_leaf2 <= 1e+06
 T102_L3_f3_le: Overall_Qual_calc + 999995 t102_leaf3 <= 1e+06
 T102_R3_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t102_leaf3
   >= -1e+06
 T102_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t102_leaf3 >= -1e+06
 T102_R4_f3_ge: Overall_Qual_calc - 1.000005e+06 t102_leaf4 >= -1e+06
 T102_L4_f207_le: const_Bsmt_Exposure_Gd + 999999 t102_leaf4 <= 1e+06
 T102_L4_f23_le: x_Full_Bath + t102_leaf4 <= 4
 T102_R5_f3_ge: Overall_Qual_calc - 1.000005e+06 t102_leaf5 >= -1e+06
 T102_L5_f207_le: const_Bsmt_Exposure_Gd + 999999 t102_leaf5 <= 1e+06
 T102_R5_f23_ge: x_Full_Bath - 3 t102_leaf5 >= 0
 T102_R6_f3_ge: Overall_Qual_calc - 1.000005e+06 t102_leaf6 >= -1e+06
 T102_R6_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t102_leaf6
   >= -1e+06
 T102_L6_f6_le: const_Year_Remod_Add + 997991 t102_leaf6 <= 1e+06
 T102_R7_f3_ge: Overall_Qual_calc - 1.000005e+06 t102_leaf7 >= -1e+06
 T102_R7_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t102_leaf7
   >= -1e+06
 T102_R7_f6_ge: const_Year_Remod_Add - 1.002009e+06 t102_leaf7 >= -1e+06
 TREE_103_ONEHOT: t103_leaf0 + t103_leaf1 + t103_leaf2 + t103_leaf3
   + t103_leaf4 + t103_leaf5 = 1
 T103_L0_f4_le: const_Overall_Cond + 999995 t103_leaf0 <= 1e+06
 T103_L0_f15_le: const_Total_Bsmt_SF + 998638 t103_leaf0 <= 1e+06
 T103_L1_f4_le: const_Overall_Cond + 999995 t103_leaf1 <= 1e+06
 T103_R1_f15_ge: const_Total_Bsmt_SF - 1.001362e+06 t103_leaf1 >= -1e+06
 T103_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t103_leaf2 >= -1e+06
 T103_L2_f23_le: x_Full_Bath + t103_leaf2 <= 4
 T103_L2_f95_le: const_Neighborhood_Crawfor + 999999 t103_leaf2 <= 1e+06
 T103_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t103_leaf3 >= -1e+06
 T103_L3_f23_le: x_Full_Bath + t103_leaf3 <= 4
 T103_R3_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t103_leaf3
   >= -1e+06
 T103_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t103_leaf4 >= -1e+06
 T103_R4_f23_ge: x_Full_Bath - 3 t103_leaf4 >= 0
 T103_L4_f36_le: x_Wood_Deck_SF + 99884 t103_leaf4 <= 100000
 T103_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t103_leaf5 >= -1e+06
 T103_R5_f23_ge: x_Full_Bath - 3 t103_leaf5 >= 0
 T103_R5_f36_ge: x_Wood_Deck_SF - 116 t103_leaf5 >= 0
 TREE_104_ONEHOT: t104_leaf0 + t104_leaf1 + t104_leaf2 + t104_leaf3
   + t104_leaf4 + t104_leaf5 = 1
 T104_L0_f3_le: Overall_Qual_calc + 999997 t104_leaf0 <= 1e+06
 T104_L0_f33_le: x_Garage_Area + 99714 t104_leaf0 <= 100000
 T104_L1_f3_le: Overall_Qual_calc + 999997 t104_leaf1 <= 1e+06
 T104_R1_f33_ge: x_Garage_Area - 286 t104_leaf1 >= 0
 T104_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t104_leaf2 >= -1e+06
 T104_L2_f1_le: const_Lot_Area + 985320 t104_leaf2 <= 1e+06
 T104_L2_f31_le: const_Garage_Yr_Blt + 998100 t104_leaf2 <= 1e+06
 T104_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t104_leaf3 >= -1e+06
 T104_L3_f1_le: const_Lot_Area + 985320 t104_leaf3 <= 1e+06
 T104_R3_f31_ge: const_Garage_Yr_Blt - 1.0019e+06 t104_leaf3 >= -1e+06
 T104_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t104_leaf4 >= -1e+06
 T104_R4_f1_ge: const_Lot_Area - 1.01468e+06 t104_leaf4 >= -1e+06
 T104_L4_f18_le: const_2nd_Flr_SF + 998724 t104_leaf4 <= 1e+06
 T104_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t104_leaf5 >= -1e+06
 T104_R5_f1_ge: const_Lot_Area - 1.01468e+06 t104_leaf5 >= -1e+06
 T104_R5_f18_ge: const_2nd_Flr_SF - 1.001276e+06 t104_leaf5 >= -1e+06
 TREE_105_ONEHOT: t105_leaf0 + t105_leaf1 + t105_leaf2 + t105_leaf3
   + t105_leaf4 + t105_leaf5 + t105_leaf6 + t105_leaf7 = 1
 T105_L0_f11_le: x_Bsmt_Cond + 2 t105_leaf0 <= 4
 T105_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t105_leaf0
   <= 1e+06
 T105_L0_f17_le: x_1st_Flr_SF + 4020 t105_leaf0 <= 5000
 T105_L1_f11_le: x_Bsmt_Cond + 2 t105_leaf1 <= 4
 T105_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t105_leaf1
   <= 1e+06
 T105_R1_f17_ge: x_1st_Flr_SF - 980 t105_leaf1 >= 0
 T105_L2_f11_le: x_Bsmt_Cond + 2 t105_leaf2 <= 4
 T105_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t105_leaf2
   >= -1e+06
 T105_L2_f16_le: x_Heating_QC + t105_leaf2 <= 4
 T105_L3_f11_le: x_Bsmt_Cond + 2 t105_leaf3 <= 4
 T105_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t105_leaf3
   >= -1e+06
 T105_R3_f16_ge: x_Heating_QC - 4 t105_leaf3 >= -1
 T105_R4_f11_ge: x_Bsmt_Cond - 3 t105_leaf4 >= -1
 T105_L4_f4_le: const_Overall_Cond + 999993 t105_leaf4 <= 1e+06
 T105_L4_f5_le: const_Year_Built + 998022 t105_leaf4 <= 1e+06
 T105_R5_f11_ge: x_Bsmt_Cond - 3 t105_leaf5 >= -1
 T105_L5_f4_le: const_Overall_Cond + 999993 t105_leaf5 <= 1e+06
 T105_R5_f5_ge: const_Year_Built - 1.001978e+06 t105_leaf5 >= -1e+06
 T105_R6_f11_ge: x_Bsmt_Cond - 3 t105_leaf6 >= -1
 T105_R6_f4_ge: const_Overall_Cond - 1.000007e+06 t105_leaf6 >= -1e+06
 T105_L6_f17_le: x_1st_Flr_SF + 3900 t105_leaf6 <= 5000
 T105_R7_f11_ge: x_Bsmt_Cond - 3 t105_leaf7 >= -1
 T105_R7_f4_ge: const_Overall_Cond - 1.000007e+06 t105_leaf7 >= -1e+06
 T105_R7_f17_ge: x_1st_Flr_SF - 1100 t105_leaf7 >= 0
 TREE_106_ONEHOT: t106_leaf0 + t106_leaf1 + t106_leaf2 + t106_leaf3
   + t106_leaf4 + t106_leaf5 + t106_leaf6 = 1
 T106_L0_f21_le: const_Bsmt_Full_Bath + 999999 t106_leaf0 <= 1e+06
 T106_L0_f20_le: x_Gr_Liv_Area + 8854 t106_leaf0 <= 10000
 T106_L0_f297_le: const_Sale_Condition_Normal + 999999 t106_leaf0 <= 1e+06
 T106_L1_f21_le: const_Bsmt_Full_Bath + 999999 t106_leaf1 <= 1e+06
 T106_L1_f20_le: x_Gr_Liv_Area + 8854 t106_leaf1 <= 10000
 T106_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t106_leaf1
   >= -1e+06
 T106_L2_f21_le: const_Bsmt_Full_Bath + 999999 t106_leaf2 <= 1e+06
 T106_R2_f20_ge: x_Gr_Liv_Area - 1146 t106_leaf2 >= 0
 T106_L2_f4_le: const_Overall_Cond + 999993 t106_leaf2 <= 1e+06
 T106_L3_f21_le: const_Bsmt_Full_Bath + 999999 t106_leaf3 <= 1e+06
 T106_R3_f20_ge: x_Gr_Liv_Area - 1146 t106_leaf3 >= 0
 T106_R3_f4_ge: const_Overall_Cond - 1.000007e+06 t106_leaf3 >= -1e+06
 T106_R4_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t106_leaf4 >= -1e+06
 T106_L4_f28_le: x_TotRms_AbvGrd + 3 t106_leaf4 <= 15
 T106_L4_f1_le: const_Lot_Area + 981614 t106_leaf4 <= 1e+06
 T106_R5_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t106_leaf5 >= -1e+06
 T106_L5_f28_le: x_TotRms_AbvGrd + 3 t106_leaf5 <= 15
 T106_R5_f1_ge: const_Lot_Area - 1.018386e+06 t106_leaf5 >= -1e+06
 T106_R6_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t106_leaf6 >= -1e+06
 T106_R6_f28_ge: x_TotRms_AbvGrd - 12 t106_leaf6 >= 0
 TREE_107_ONEHOT: t107_leaf0 + t107_leaf1 + t107_leaf2 + t107_leaf3
   + t107_leaf4 + t107_leaf5 + t107_leaf6 + t107_leaf7 = 1
 T107_L0_f44_le: const_Yr_Sold + 997992 t107_leaf0 <= 1e+06
 T107_L0_f1_le: const_Lot_Area + 986189 t107_leaf0 <= 1e+06
 T107_L0_f12_le: x_BsmtFin_SF_1 + 999777 t107_leaf0 <= 1e+06
 T107_L1_f44_le: const_Yr_Sold + 997992 t107_leaf1 <= 1e+06
 T107_L1_f1_le: const_Lot_Area + 986189 t107_leaf1 <= 1e+06
 T107_R1_f12_ge: x_BsmtFin_SF_1 - 223 t107_leaf1 >= 0
 T107_L2_f44_le: const_Yr_Sold + 997992 t107_leaf2 <= 1e+06
 T107_R2_f1_ge: const_Lot_Area - 1.013811e+06 t107_leaf2 >= -1e+06
 T107_L2_f37_le: x_Open_Porch_SF + 99727 t107_leaf2 <= 100000
 T107_L3_f44_le: const_Yr_Sold + 997992 t107_leaf3 <= 1e+06
 T107_R3_f1_ge: const_Lot_Area - 1.013811e+06 t107_leaf3 >= -1e+06
 T107_R3_f37_ge: x_Open_Porch_SF - 273 t107_leaf3 >= 0
 T107_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t107_leaf4 >= -1e+06
 T107_L4_f100_le: const_Neighborhood_IDOTRR + 999999 t107_leaf4 <= 1e+06
 T107_L4_f118_le: const_Condition_1_Feedr + 999999 t107_leaf4 <= 1e+06
 T107_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t107_leaf5 >= -1e+06
 T107_L5_f100_le: const_Neighborhood_IDOTRR + 999999 t107_leaf5 <= 1e+06
 T107_R5_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t107_leaf5
   >= -1e+06
 T107_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t107_leaf6 >= -1e+06
 T107_R6_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t107_leaf6
   >= -1e+06
 T107_L6_f0_le: const_Lot_Frontage + 999937 t107_leaf6 <= 1e+06
 T107_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t107_leaf7 >= -1e+06
 T107_R7_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t107_leaf7
   >= -1e+06
 T107_R7_f0_ge: const_Lot_Frontage - 1.000063e+06 t107_leaf7 >= -1e+06
 TREE_108_ONEHOT: t108_leaf0 + t108_leaf1 + t108_leaf2 + t108_leaf3
   + t108_leaf4 + t108_leaf5 + t108_leaf6 + t108_leaf7 = 1
 T108_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t108_leaf0
   <= 1e+06
 T108_L0_f76_le: const_Lot_Shape_Reg + 999999 t108_leaf0 <= 1e+06
 T108_L0_f164_le: x_ex1_is_BrkFace <= 1
 T108_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t108_leaf1
   <= 1e+06
 T108_L1_f76_le: const_Lot_Shape_Reg + 999999 t108_leaf1 <= 1e+06
 T108_R1_f164_ge: x_ex1_is_BrkFace - t108_leaf1 >= 0
 T108_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t108_leaf2
   <= 1e+06
 T108_R2_f76_ge: const_Lot_Shape_Reg - 1.000001e+06 t108_leaf2 >= -1e+06
 T108_L2_f1_le: const_Lot_Area + 997355 t108_leaf2 <= 1e+06
 T108_L3_f293_le: const_Sale_Condition_Abnorml + 999999 t108_leaf3
   <= 1e+06
 T108_R3_f76_ge: const_Lot_Shape_Reg - 1.000001e+06 t108_leaf3 >= -1e+06
 T108_R3_f1_ge: const_Lot_Area - 1.002645e+06 t108_leaf3 >= -1e+06
 T108_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t108_leaf4
   >= -1e+06
 T108_L4_f3_le: Overall_Qual_calc + 999995 t108_leaf4 <= 1e+06
 T108_L4_f169_le: x_ex1_is_MetalSd <= 1
 T108_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t108_leaf5
   >= -1e+06
 T108_L5_f3_le: Overall_Qual_calc + 999995 t108_leaf5 <= 1e+06
 T108_R5_f169_ge: x_ex1_is_MetalSd - t108_leaf5 >= 0
 T108_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t108_leaf6
   >= -1e+06
 T108_R6_f3_ge: Overall_Qual_calc - 1.000005e+06 t108_leaf6 >= -1e+06
 T108_L6_f27_le: x_Kitchen_Qual <= 4
 T108_R7_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t108_leaf7
   >= -1e+06
 T108_R7_f3_ge: Overall_Qual_calc - 1.000005e+06 t108_leaf7 >= -1e+06
 T108_R7_f27_ge: x_Kitchen_Qual - 5 t108_leaf7 >= -1
 TREE_109_ONEHOT: t109_leaf0 + t109_leaf1 + t109_leaf2 + t109_leaf3
   + t109_leaf4 + t109_leaf5 + t109_leaf6 = 1
 T109_L0_f9_le: x_Exter_Cond + 2 t109_leaf0 <= 4
 T109_L0_f3_le: Overall_Qual_calc + 999997 t109_leaf0 <= 1e+06
 T109_L1_f9_le: x_Exter_Cond + 2 t109_leaf1 <= 4
 T109_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t109_leaf1 >= -1e+06
 T109_L1_f33_le: x_Garage_Area + 99649 t109_leaf1 <= 100000
 T109_L2_f9_le: x_Exter_Cond + 2 t109_leaf2 <= 4
 T109_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t109_leaf2 >= -1e+06
 T109_R2_f33_ge: x_Garage_Area - 351 t109_leaf2 >= 0
 T109_R3_f9_ge: x_Exter_Cond - 3 t109_leaf3 >= -1
 T109_L3_f20_le: x_Gr_Liv_Area + 7737 t109_leaf3 <= 10000
 T109_L3_f44_le: const_Yr_Sold + 997992 t109_leaf3 <= 1e+06
 T109_R4_f9_ge: x_Exter_Cond - 3 t109_leaf4 >= -1
 T109_L4_f20_le: x_Gr_Liv_Area + 7737 t109_leaf4 <= 10000
 T109_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t109_leaf4 >= -1e+06
 T109_R5_f9_ge: x_Exter_Cond - 3 t109_leaf5 >= -1
 T109_R5_f20_ge: x_Gr_Liv_Area - 2263 t109_leaf5 >= 0
 T109_L5_f17_le: x_1st_Flr_SF + 3759 t109_leaf5 <= 5000
 T109_R6_f9_ge: x_Exter_Cond - 3 t109_leaf6 >= -1
 T109_R6_f20_ge: x_Gr_Liv_Area - 2263 t109_leaf6 >= 0
 T109_R6_f17_ge: x_1st_Flr_SF - 1241 t109_leaf6 >= 0
 TREE_110_ONEHOT: t110_leaf0 + t110_leaf1 + t110_leaf2 + t110_leaf3
   + t110_leaf4 + t110_leaf5 = 1
 T110_L0_f40_le: x_Screen_Porch + 99896 t110_leaf0 <= 100000
 T110_L0_f12_le: x_BsmtFin_SF_1 + 999777 t110_leaf0 <= 1e+06
 T110_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t110_leaf0
   <= 1e+06
 T110_L1_f40_le: x_Screen_Porch + 99896 t110_leaf1 <= 100000
 T110_L1_f12_le: x_BsmtFin_SF_1 + 999777 t110_leaf1 <= 1e+06
 T110_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t110_leaf1
   >= -1e+06
 T110_L2_f40_le: x_Screen_Porch + 99896 t110_leaf2 <= 100000
 T110_R2_f12_ge: x_BsmtFin_SF_1 - 223 t110_leaf2 >= 0
 T110_L2_f18_le: const_2nd_Flr_SF + 998371 t110_leaf2 <= 1e+06
 T110_L3_f40_le: x_Screen_Porch + 99896 t110_leaf3 <= 100000
 T110_R3_f12_ge: x_BsmtFin_SF_1 - 223 t110_leaf3 >= 0
 T110_R3_f18_ge: const_2nd_Flr_SF - 1.001629e+06 t110_leaf3 >= -1e+06
 T110_R4_f40_ge: x_Screen_Porch - 104 t110_leaf4 >= 0
 T110_L4_f23_le: x_Full_Bath + 2 t110_leaf4 <= 4
 T110_R5_f40_ge: x_Screen_Porch - 104 t110_leaf5 >= 0
 T110_R5_f23_ge: x_Full_Bath - 2 t110_leaf5 >= 0
 TREE_111_ONEHOT: t111_leaf0 + t111_leaf1 + t111_leaf2 + t111_leaf3
   + t111_leaf4 + t111_leaf5 + t111_leaf6 + t111_leaf7 = 1
 T111_L0_f44_le: const_Yr_Sold + 997992 t111_leaf0 <= 1e+06
 T111_L0_f12_le: x_BsmtFin_SF_1 + 999799 t111_leaf0 <= 1e+06
 T111_L0_f36_le: x_Wood_Deck_SF + 99842 t111_leaf0 <= 100000
 T111_L1_f44_le: const_Yr_Sold + 997992 t111_leaf1 <= 1e+06
 T111_L1_f12_le: x_BsmtFin_SF_1 + 999799 t111_leaf1 <= 1e+06
 T111_R1_f36_ge: x_Wood_Deck_SF - 158 t111_leaf1 >= 0
 T111_L2_f44_le: const_Yr_Sold + 997992 t111_leaf2 <= 1e+06
 T111_R2_f12_ge: x_BsmtFin_SF_1 - 201 t111_leaf2 >= 0
 T111_L2_f4_le: const_Overall_Cond + 999994 t111_leaf2 <= 1e+06
 T111_L3_f44_le: const_Yr_Sold + 997992 t111_leaf3 <= 1e+06
 T111_R3_f12_ge: x_BsmtFin_SF_1 - 201 t111_leaf3 >= 0
 T111_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t111_leaf3 >= -1e+06
 T111_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t111_leaf4 >= -1e+06
 T111_L4_f100_le: const_Neighborhood_IDOTRR + 999999 t111_leaf4 <= 1e+06
 T111_L4_f4_le: const_Overall_Cond + 999993 t111_leaf4 <= 1e+06
 T111_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t111_leaf5 >= -1e+06
 T111_L5_f100_le: const_Neighborhood_IDOTRR + 999999 t111_leaf5 <= 1e+06
 T111_R5_f4_ge: const_Overall_Cond - 1.000007e+06 t111_leaf5 >= -1e+06
 T111_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t111_leaf6 >= -1e+06
 T111_R6_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t111_leaf6
   >= -1e+06
 T111_L6_f20_le: x_Gr_Liv_Area + 8866 t111_leaf6 <= 10000
 T111_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t111_leaf7 >= -1e+06
 T111_R7_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t111_leaf7
   >= -1e+06
 T111_R7_f20_ge: x_Gr_Liv_Area - 1134 t111_leaf7 >= 0
 TREE_112_ONEHOT: t112_leaf0 + t112_leaf1 + t112_leaf2 + t112_leaf3
   + t112_leaf4 + t112_leaf5 + t112_leaf6 + t112_leaf7 = 1
 T112_L0_f73_le: const_Lot_Shape_IR1 + 999999 t112_leaf0 <= 1e+06
 T112_L0_f18_le: const_2nd_Flr_SF + 999022 t112_leaf0 <= 1e+06
 T112_L0_f11_le: x_Bsmt_Cond + 2 t112_leaf0 <= 4
 T112_L1_f73_le: const_Lot_Shape_IR1 + 999999 t112_leaf1 <= 1e+06
 T112_L1_f18_le: const_2nd_Flr_SF + 999022 t112_leaf1 <= 1e+06
 T112_R1_f11_ge: x_Bsmt_Cond - 3 t112_leaf1 >= -1
 T112_L2_f73_le: const_Lot_Shape_IR1 + 999999 t112_leaf2 <= 1e+06
 T112_R2_f18_ge: const_2nd_Flr_SF - 1.000978e+06 t112_leaf2 >= -1e+06
 T112_L2_f4_le: const_Overall_Cond + 999994 t112_leaf2 <= 1e+06
 T112_L3_f73_le: const_Lot_Shape_IR1 + 999999 t112_leaf3 <= 1e+06
 T112_R3_f18_ge: const_2nd_Flr_SF - 1.000978e+06 t112_leaf3 >= -1e+06
 T112_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t112_leaf3 >= -1e+06
 T112_R4_f73_ge: const_Lot_Shape_IR1 - 1.000001e+06 t112_leaf4 >= -1e+06
 T112_L4_f44_le: const_Yr_Sold + 997992 t112_leaf4 <= 1e+06
 T112_L4_f1_le: const_Lot_Area + 984862 t112_leaf4 <= 1e+06
 T112_R5_f73_ge: const_Lot_Shape_IR1 - 1.000001e+06 t112_leaf5 >= -1e+06
 T112_L5_f44_le: const_Yr_Sold + 997992 t112_leaf5 <= 1e+06
 T112_R5_f1_ge: const_Lot_Area - 1.015138e+06 t112_leaf5 >= -1e+06
 T112_R6_f73_ge: const_Lot_Shape_IR1 - 1.000001e+06 t112_leaf6 >= -1e+06
 T112_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t112_leaf6 >= -1e+06
 T112_L6_f298_le: const_Sale_Condition_Partial + 999999 t112_leaf6
   <= 1e+06
 T112_R7_f73_ge: const_Lot_Shape_IR1 - 1.000001e+06 t112_leaf7 >= -1e+06
 T112_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t112_leaf7 >= -1e+06
 T112_R7_f298_ge: const_Sale_Condition_Partial - 1.000001e+06 t112_leaf7
   >= -1e+06
 TREE_113_ONEHOT: t113_leaf0 + t113_leaf1 + t113_leaf2 + t113_leaf3
   + t113_leaf4 + t113_leaf5 + t113_leaf6 = 1
 T113_L0_f20_le: x_Gr_Liv_Area + 9167 t113_leaf0 <= 10000
 T113_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t113_leaf0
   <= 1e+06
 T113_L0_f38_le: x_Enclosed_Porch + 99965 t113_leaf0 <= 100000
 T113_L1_f20_le: x_Gr_Liv_Area + 9167 t113_leaf1 <= 10000
 T113_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t113_leaf1
   <= 1e+06
 T113_R1_f38_ge: x_Enclosed_Porch - 35 t113_leaf1 >= 0
 T113_L2_f20_le: x_Gr_Liv_Area + 9167 t113_leaf2 <= 10000
 T113_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t113_leaf2
   >= -1e+06
 T113_R3_f20_ge: x_Gr_Liv_Area - 833 t113_leaf3 >= 0
 T113_L3_f15_le: const_Total_Bsmt_SF + 998562 t113_leaf3 <= 1e+06
 T113_L3_f95_le: const_Neighborhood_Crawfor + 999999 t113_leaf3 <= 1e+06
 T113_R4_f20_ge: x_Gr_Liv_Area - 833 t113_leaf4 >= 0
 T113_L4_f15_le: const_Total_Bsmt_SF + 998562 t113_leaf4 <= 1e+06
 T113_R4_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t113_leaf4
   >= -1e+06
 T113_R5_f20_ge: x_Gr_Liv_Area - 833 t113_leaf5 >= 0
 T113_R5_f15_ge: const_Total_Bsmt_SF - 1.001438e+06 t113_leaf5 >= -1e+06
 T113_L5_f37_le: x_Open_Porch_SF + 99765 t113_leaf5 <= 100000
 T113_R6_f20_ge: x_Gr_Liv_Area - 833 t113_leaf6 >= 0
 T113_R6_f15_ge: const_Total_Bsmt_SF - 1.001438e+06 t113_leaf6 >= -1e+06
 T113_R6_f37_ge: x_Open_Porch_SF - 235 t113_leaf6 >= 0
 TREE_114_ONEHOT: t114_leaf0 + t114_leaf1 + t114_leaf2 + t114_leaf3
   + t114_leaf4 + t114_leaf5 + t114_leaf6 + t114_leaf7 = 1
 T114_L0_f1_le: const_Lot_Area + 992736 t114_leaf0 <= 1e+06
 T114_L0_f12_le: x_BsmtFin_SF_1 + 999230 t114_leaf0 <= 1e+06
 T114_L0_f3_le: Overall_Qual_calc + 999995 t114_leaf0 <= 1e+06
 T114_L1_f1_le: const_Lot_Area + 992736 t114_leaf1 <= 1e+06
 T114_L1_f12_le: x_BsmtFin_SF_1 + 999230 t114_leaf1 <= 1e+06
 T114_R1_f3_ge: Overall_Qual_calc - 1.000005e+06 t114_leaf1 >= -1e+06
 T114_L2_f1_le: const_Lot_Area + 992736 t114_leaf2 <= 1e+06
 T114_R2_f12_ge: x_BsmtFin_SF_1 - 770 t114_leaf2 >= 0
 T114_L2_f5_le: const_Year_Built + 998075 t114_leaf2 <= 1e+06
 T114_L3_f1_le: const_Lot_Area + 992736 t114_leaf3 <= 1e+06
 T114_R3_f12_ge: x_BsmtFin_SF_1 - 770 t114_leaf3 >= 0
 T114_R3_f5_ge: const_Year_Built - 1.001925e+06 t114_leaf3 >= -1e+06
 T114_R4_f1_ge: const_Lot_Area - 1.007264e+06 t114_leaf4 >= -1e+06
 T114_L4_f36_le: x_Wood_Deck_SF + 99801 t114_leaf4 <= 100000
 T114_L4_f40_le: x_Screen_Porch + 99885 t114_leaf4 <= 100000
 T114_R5_f1_ge: const_Lot_Area - 1.007264e+06 t114_leaf5 >= -1e+06
 T114_L5_f36_le: x_Wood_Deck_SF + 99801 t114_leaf5 <= 100000
 T114_R5_f40_ge: x_Screen_Porch - 115 t114_leaf5 >= 0
 T114_R6_f1_ge: const_Lot_Area - 1.007264e+06 t114_leaf6 >= -1e+06
 T114_R6_f36_ge: x_Wood_Deck_SF - 199 t114_leaf6 >= 0
 T114_L6_f0_le: const_Lot_Frontage + 999850 t114_leaf6 <= 1e+06
 T114_R7_f1_ge: const_Lot_Area - 1.007264e+06 t114_leaf7 >= -1e+06
 T114_R7_f36_ge: x_Wood_Deck_SF - 199 t114_leaf7 >= 0
 T114_R7_f0_ge: const_Lot_Frontage - 1.00015e+06 t114_leaf7 >= -1e+06
 TREE_115_ONEHOT: t115_leaf0 + t115_leaf1 + t115_leaf2 + t115_leaf3
   + t115_leaf4 + t115_leaf5 = 1
 T115_L0_f4_le: const_Overall_Cond + 999995 t115_leaf0 <= 1e+06
 T115_L0_f40_le: x_Screen_Porch + 99912 t115_leaf0 <= 100000
 T115_L1_f4_le: const_Overall_Cond + 999995 t115_leaf1 <= 1e+06
 T115_R1_f40_ge: x_Screen_Porch - 88 t115_leaf1 >= 0
 T115_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t115_leaf2 >= -1e+06
 T115_L2_f36_le: x_Wood_Deck_SF + 99848 t115_leaf2 <= 100000
 T115_L2_f40_le: x_Screen_Porch + 99900 t115_leaf2 <= 100000
 T115_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t115_leaf3 >= -1e+06
 T115_L3_f36_le: x_Wood_Deck_SF + 99848 t115_leaf3 <= 100000
 T115_R3_f40_ge: x_Screen_Porch - 100 t115_leaf3 >= 0
 T115_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t115_leaf4 >= -1e+06
 T115_R4_f36_ge: x_Wood_Deck_SF - 152 t115_leaf4 >= 0
 T115_L4_f77_le: const_Land_Contour_Bnk + 999999 t115_leaf4 <= 1e+06
 T115_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t115_leaf5 >= -1e+06
 T115_R5_f36_ge: x_Wood_Deck_SF - 152 t115_leaf5 >= 0
 T115_R5_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t115_leaf5
   >= -1e+06
 TREE_116_ONEHOT: t116_leaf0 + t116_leaf1 + t116_leaf2 + t116_leaf3
   + t116_leaf4 + t116_leaf5 + t116_leaf6 = 1
 T116_L0_f44_le: const_Yr_Sold + 997992 t116_leaf0 <= 1e+06
 T116_L0_f20_le: x_Gr_Liv_Area + 6914 t116_leaf0 <= 10000
 T116_L0_f12_le: x_BsmtFin_SF_1 + 999119 t116_leaf0 <= 1e+06
 T116_L1_f44_le: const_Yr_Sold + 997992 t116_leaf1 <= 1e+06
 T116_L1_f20_le: x_Gr_Liv_Area + 6914 t116_leaf1 <= 10000
 T116_R1_f12_ge: x_BsmtFin_SF_1 - 881 t116_leaf1 >= 0
 T116_L2_f44_le: const_Yr_Sold + 997992 t116_leaf2 <= 1e+06
 T116_R2_f20_ge: x_Gr_Liv_Area - 3086 t116_leaf2 >= 0
 T116_R3_f44_ge: const_Yr_Sold - 1.002008e+06 t116_leaf3 >= -1e+06
 T116_L3_f293_le: const_Sale_Condition_Abnorml + 999999 t116_leaf3
   <= 1e+06
 T116_L3_f4_le: const_Overall_Cond + 999993 t116_leaf3 <= 1e+06
 T116_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t116_leaf4 >= -1e+06
 T116_L4_f293_le: const_Sale_Condition_Abnorml + 999999 t116_leaf4
   <= 1e+06
 T116_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t116_leaf4 >= -1e+06
 T116_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t116_leaf5 >= -1e+06
 T116_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t116_leaf5
   >= -1e+06
 T116_L5_f20_le: x_Gr_Liv_Area + 9024 t116_leaf5 <= 10000
 T116_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t116_leaf6 >= -1e+06
 T116_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t116_leaf6
   >= -1e+06
 T116_R6_f20_ge: x_Gr_Liv_Area - 976 t116_leaf6 >= 0
 TREE_117_ONEHOT: t117_leaf0 + t117_leaf1 + t117_leaf2 + t117_leaf3
   + t117_leaf4 = 1
 T117_L0_f20_le: x_Gr_Liv_Area + 6914 t117_leaf0 <= 10000
 T117_L0_f4_le: const_Overall_Cond + 999995 t117_leaf0 <= 1e+06
 T117_L0_f16_le: x_Heating_QC + t117_leaf0 <= 4
 T117_L1_f20_le: x_Gr_Liv_Area + 6914 t117_leaf1 <= 10000
 T117_L1_f4_le: const_Overall_Cond + 999995 t117_leaf1 <= 1e+06
 T117_R1_f16_ge: x_Heating_QC - 4 t117_leaf1 >= -1
 T117_L2_f20_le: x_Gr_Liv_Area + 6914 t117_leaf2 <= 10000
 T117_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t117_leaf2 >= -1e+06
 T117_L2_f21_le: const_Bsmt_Full_Bath + 999999 t117_leaf2 <= 1e+06
 T117_L3_f20_le: x_Gr_Liv_Area + 6914 t117_leaf3 <= 10000
 T117_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t117_leaf3 >= -1e+06
 T117_R3_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t117_leaf3 >= -1e+06
 T117_R4_f20_ge: x_Gr_Liv_Area - 3086 t117_leaf4 >= 0
 TREE_118_ONEHOT: t118_leaf0 + t118_leaf1 + t118_leaf2 + t118_leaf3
   + t118_leaf4 + t118_leaf5 = 1
 T118_L0_f161_le: x_ex1_is_AsbShng <= 1
 T118_L0_f44_le: const_Yr_Sold + 997993 t118_leaf0 <= 1e+06
 T118_L0_f36_le: x_Wood_Deck_SF + 99801 t118_leaf0 <= 100000
 T118_L1_f161_le: x_ex1_is_AsbShng <= 1
 T118_L1_f44_le: const_Yr_Sold + 997993 t118_leaf1 <= 1e+06
 T118_R1_f36_ge: x_Wood_Deck_SF - 199 t118_leaf1 >= 0
 T118_L2_f161_le: x_ex1_is_AsbShng <= 1
 T118_R2_f44_ge: const_Yr_Sold - 1.002007e+06 t118_leaf2 >= -1e+06
 T118_L2_f1_le: const_Lot_Area + 994024 t118_leaf2 <= 1e+06
 T118_L3_f161_le: x_ex1_is_AsbShng <= 1
 T118_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t118_leaf3 >= -1e+06
 T118_R3_f1_ge: const_Lot_Area - 1.005976e+06 t118_leaf3 >= -1e+06
 T118_R4_f161_ge: x_ex1_is_AsbShng - t118_leaf4 >= 0
 T118_L4_f25_le: x_Bedroom_AbvGr + 3 t118_leaf4 <= 6
 T118_R5_f161_ge: x_ex1_is_AsbShng - t118_leaf5 >= 0
 T118_R5_f25_ge: x_Bedroom_AbvGr - 3 t118_leaf5 >= 0
 TREE_119_ONEHOT: t119_leaf0 + t119_leaf1 + t119_leaf2 + t119_leaf3 = 1
 T119_L0_f4_le: const_Overall_Cond + 999997 t119_leaf0 <= 1e+06
 T119_R1_f4_ge: const_Overall_Cond - 1.000003e+06 t119_leaf1 >= -1e+06
 T119_L1_f18_le: const_2nd_Flr_SF + 998573 t119_leaf1 <= 1e+06
 T119_L1_f12_le: x_BsmtFin_SF_1 + 999925 t119_leaf1 <= 1e+06
 T119_R2_f4_ge: const_Overall_Cond - 1.000003e+06 t119_leaf2 >= -1e+06
 T119_L2_f18_le: const_2nd_Flr_SF + 998573 t119_leaf2 <= 1e+06
 T119_R2_f12_ge: x_BsmtFin_SF_1 - 75 t119_leaf2 >= 0
 T119_R3_f4_ge: const_Overall_Cond - 1.000003e+06 t119_leaf3 >= -1e+06
 T119_R3_f18_ge: const_2nd_Flr_SF - 1.001427e+06 t119_leaf3 >= -1e+06
 TREE_120_ONEHOT: t120_leaf0 + t120_leaf1 + t120_leaf2 + t120_leaf3
   + t120_leaf4 + t120_leaf5 = 1
 T120_L0_f3_le: Overall_Qual_calc + 999997 t120_leaf0 <= 1e+06
 T120_L0_f6_le: const_Year_Remod_Add + 998049 t120_leaf0 <= 1e+06
 T120_L1_f3_le: Overall_Qual_calc + 999997 t120_leaf1 <= 1e+06
 T120_R1_f6_ge: const_Year_Remod_Add - 1.001951e+06 t120_leaf1 >= -1e+06
 T120_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t120_leaf2 >= -1e+06
 T120_L2_f95_le: const_Neighborhood_Crawfor + 999999 t120_leaf2 <= 1e+06
 T120_L2_f33_le: x_Garage_Area + 98846 t120_leaf2 <= 100000
 T120_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t120_leaf3 >= -1e+06
 T120_L3_f95_le: const_Neighborhood_Crawfor + 999999 t120_leaf3 <= 1e+06
 T120_R3_f33_ge: x_Garage_Area - 1154 t120_leaf3 >= 0
 T120_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t120_leaf4 >= -1e+06
 T120_R4_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t120_leaf4
   >= -1e+06
 T120_L4_f1_le: const_Lot_Area + 995574 t120_leaf4 <= 1e+06
 T120_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t120_leaf5 >= -1e+06
 T120_R5_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t120_leaf5
   >= -1e+06
 T120_R5_f1_ge: const_Lot_Area - 1.004426e+06 t120_leaf5 >= -1e+06
 TREE_121_ONEHOT: t121_leaf0 + t121_leaf1 + t121_leaf2 + t121_leaf3
   + t121_leaf4 + t121_leaf5 + t121_leaf6 = 1
 T121_L0_f20_le: x_Gr_Liv_Area + 8845 t121_leaf0 <= 10000
 T121_L0_f297_le: const_Sale_Condition_Normal + 999999 t121_leaf0 <= 1e+06
 T121_L0_f259_le: x_paved_drive_is_Y <= 1
 T121_L1_f20_le: x_Gr_Liv_Area + 8845 t121_leaf1 <= 10000
 T121_L1_f297_le: const_Sale_Condition_Normal + 999999 t121_leaf1 <= 1e+06
 T121_R1_f259_ge: x_paved_drive_is_Y - t121_leaf1 >= 0
 T121_L2_f20_le: x_Gr_Liv_Area + 8845 t121_leaf2 <= 10000
 T121_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t121_leaf2
   >= -1e+06
 T121_L2_f31_le: const_Garage_Yr_Blt + 998069 t121_leaf2 <= 1e+06
 T121_L3_f20_le: x_Gr_Liv_Area + 8845 t121_leaf3 <= 10000
 T121_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t121_leaf3
   >= -1e+06
 T121_R3_f31_ge: const_Garage_Yr_Blt - 1.001931e+06 t121_leaf3 >= -1e+06
 T121_R4_f20_ge: x_Gr_Liv_Area - 1155 t121_leaf4 >= 0
 T121_L4_f7_le: x_Mas_Vnr_Area + 1496 t121_leaf4 <= 2000
 T121_L4_f12_le: x_BsmtFin_SF_1 + 999781 t121_leaf4 <= 1e+06
 T121_R5_f20_ge: x_Gr_Liv_Area - 1155 t121_leaf5 >= 0
 T121_L5_f7_le: x_Mas_Vnr_Area + 1496 t121_leaf5 <= 2000
 T121_R5_f12_ge: x_BsmtFin_SF_1 - 219 t121_leaf5 >= 0
 T121_R6_f20_ge: x_Gr_Liv_Area - 1155 t121_leaf6 >= 0
 T121_R6_f7_ge: x_Mas_Vnr_Area - 504 t121_leaf6 >= 0
 TREE_122_ONEHOT: t122_leaf0 + t122_leaf1 + t122_leaf2 + t122_leaf3
   + t122_leaf4 = 1
 T122_L0_f95_le: const_Neighborhood_Crawfor + 999999 t122_leaf0 <= 1e+06
 T122_L0_f1_le: const_Lot_Area + 994024 t122_leaf0 <= 1e+06
 T122_L0_f63_le: const_MS_Zoning_FV + 999999 t122_leaf0 <= 1e+06
 T122_L1_f95_le: const_Neighborhood_Crawfor + 999999 t122_leaf1 <= 1e+06
 T122_L1_f1_le: const_Lot_Area + 994024 t122_leaf1 <= 1e+06
 T122_R1_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t122_leaf1 >= -1e+06
 T122_L2_f95_le: const_Neighborhood_Crawfor + 999999 t122_leaf2 <= 1e+06
 T122_R2_f1_ge: const_Lot_Area - 1.005976e+06 t122_leaf2 >= -1e+06
 T122_L2_f207_le: const_Bsmt_Exposure_Gd + 999999 t122_leaf2 <= 1e+06
 T122_L3_f95_le: const_Neighborhood_Crawfor + 999999 t122_leaf3 <= 1e+06
 T122_R3_f1_ge: const_Lot_Area - 1.005976e+06 t122_leaf3 >= -1e+06
 T122_R3_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t122_leaf3
   >= -1e+06
 T122_R4_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t122_leaf4
   >= -1e+06
 TREE_123_ONEHOT: t123_leaf0 + t123_leaf1 + t123_leaf2 + t123_leaf3
   + t123_leaf4 + t123_leaf5 + t123_leaf6 + t123_leaf7 = 1
 T123_L0_f44_le: const_Yr_Sold + 997992 t123_leaf0 <= 1e+06
 T123_L0_f17_le: x_1st_Flr_SF + 3189 t123_leaf0 <= 5000
 T123_L0_f4_le: const_Overall_Cond + 999994 t123_leaf0 <= 1e+06
 T123_L1_f44_le: const_Yr_Sold + 997992 t123_leaf1 <= 1e+06
 T123_L1_f17_le: x_1st_Flr_SF + 3189 t123_leaf1 <= 5000
 T123_R1_f4_ge: const_Overall_Cond - 1.000006e+06 t123_leaf1 >= -1e+06
 T123_L2_f44_le: const_Yr_Sold + 997992 t123_leaf2 <= 1e+06
 T123_R2_f17_ge: x_1st_Flr_SF - 1811 t123_leaf2 >= 0
 T123_L2_f76_le: const_Lot_Shape_Reg + 999999 t123_leaf2 <= 1e+06
 T123_L3_f44_le: const_Yr_Sold + 997992 t123_leaf3 <= 1e+06
 T123_R3_f17_ge: x_1st_Flr_SF - 1811 t123_leaf3 >= 0
 T123_R3_f76_ge: const_Lot_Shape_Reg - 1.000001e+06 t123_leaf3 >= -1e+06
 T123_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t123_leaf4 >= -1e+06
 T123_L4_f6_le: const_Year_Remod_Add + 998046 t123_leaf4 <= 1e+06
 T123_L4_f87_le: const_Land_Slope_Mod + 999999 t123_leaf4 <= 1e+06
 T123_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t123_leaf5 >= -1e+06
 T123_L5_f6_le: const_Year_Remod_Add + 998046 t123_leaf5 <= 1e+06
 T123_R5_f87_ge: const_Land_Slope_Mod - 1.000001e+06 t123_leaf5 >= -1e+06
 T123_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t123_leaf6 >= -1e+06
 T123_R6_f6_ge: const_Year_Remod_Add - 1.001954e+06 t123_leaf6 >= -1e+06
 T123_L6_f164_le: x_ex1_is_BrkFace <= 1
 T123_R7_f44_ge: const_Yr_Sold - 1.002008e+06 t123_leaf7 >= -1e+06
 T123_R7_f6_ge: const_Year_Remod_Add - 1.001954e+06 t123_leaf7 >= -1e+06
 T123_R7_f164_ge: x_ex1_is_BrkFace - t123_leaf7 >= 0
 TREE_124_ONEHOT: t124_leaf0 + t124_leaf1 + t124_leaf2 + t124_leaf3
   + t124_leaf4 + t124_leaf5 = 1
 T124_L0_f36_le: x_Wood_Deck_SF + 99784 t124_leaf0 <= 100000
 T124_L0_f95_le: const_Neighborhood_Crawfor + 999999 t124_leaf0 <= 1e+06
 T124_L0_f6_le: const_Year_Remod_Add + 998007 t124_leaf0 <= 1e+06
 T124_L1_f36_le: x_Wood_Deck_SF + 99784 t124_leaf1 <= 100000
 T124_L1_f95_le: const_Neighborhood_Crawfor + 999999 t124_leaf1 <= 1e+06
 T124_R1_f6_ge: const_Year_Remod_Add - 1.001993e+06 t124_leaf1 >= -1e+06
 T124_L2_f36_le: x_Wood_Deck_SF + 99784 t124_leaf2 <= 100000
 T124_R2_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t124_leaf2
   >= -1e+06
 T124_L2_f33_le: x_Garage_Area + 99503 t124_leaf2 <= 100000
 T124_L3_f36_le: x_Wood_Deck_SF + 99784 t124_leaf3 <= 100000
 T124_R3_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t124_leaf3
   >= -1e+06
 T124_R3_f33_ge: x_Garage_Area - 497 t124_leaf3 >= 0
 T124_R4_f36_ge: x_Wood_Deck_SF - 216 t124_leaf4 >= 0
 T124_L4_f17_le: x_1st_Flr_SF + 3189 t124_leaf4 <= 5000
 T124_R5_f36_ge: x_Wood_Deck_SF - 216 t124_leaf5 >= 0
 T124_R5_f17_ge: x_1st_Flr_SF - 1811 t124_leaf5 >= 0
 TREE_125_ONEHOT: t125_leaf0 + t125_leaf1 + t125_leaf2 + t125_leaf3
   + t125_leaf4 + t125_leaf5 = 1
 T125_L0_f16_le: x_Heating_QC + 2 t125_leaf0 <= 4
 T125_L0_f191_le: x_ex2_is_VinylSd <= 1
 T125_L1_f16_le: x_Heating_QC + 2 t125_leaf1 <= 4
 T125_R1_f191_ge: x_ex2_is_VinylSd - t125_leaf1 >= 0
 T125_R2_f16_ge: x_Heating_QC - 3 t125_leaf2 >= -1
 T125_L2_f29_le: const_Fireplaces + 999998 t125_leaf2 <= 1e+06
 T125_L2_f289_le: const_Sale_Type_New + 999999 t125_leaf2 <= 1e+06
 T125_R3_f16_ge: x_Heating_QC - 3 t125_leaf3 >= -1
 T125_L3_f29_le: const_Fireplaces + 999998 t125_leaf3 <= 1e+06
 T125_R3_f289_ge: const_Sale_Type_New - 1.000001e+06 t125_leaf3 >= -1e+06
 T125_R4_f16_ge: x_Heating_QC - 3 t125_leaf4 >= -1
 T125_R4_f29_ge: const_Fireplaces - 1.000002e+06 t125_leaf4 >= -1e+06
 T125_L4_f0_le: const_Lot_Frontage + 999883 t125_leaf4 <= 1e+06
 T125_R5_f16_ge: x_Heating_QC - 3 t125_leaf5 >= -1
 T125_R5_f29_ge: const_Fireplaces - 1.000002e+06 t125_leaf5 >= -1e+06
 T125_R5_f0_ge: const_Lot_Frontage - 1.000117e+06 t125_leaf5 >= -1e+06
 TREE_126_ONEHOT: t126_leaf0 + t126_leaf1 + t126_leaf2 + t126_leaf3
   + t126_leaf4 = 1
 T126_L0_f17_le: x_1st_Flr_SF + 4424 t126_leaf0 <= 5000
 T126_R1_f17_ge: x_1st_Flr_SF - 576 t126_leaf1 >= 0
 T126_L1_f1_le: const_Lot_Area + 980478 t126_leaf1 <= 1e+06
 T126_L1_f46_le: const_MS_SubClass_30 + 999999 t126_leaf1 <= 1e+06
 T126_R2_f17_ge: x_1st_Flr_SF - 576 t126_leaf2 >= 0
 T126_L2_f1_le: const_Lot_Area + 980478 t126_leaf2 <= 1e+06
 T126_R2_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t126_leaf2 >= -1e+06
 T126_R3_f17_ge: x_1st_Flr_SF - 576 t126_leaf3 >= 0
 T126_R3_f1_ge: const_Lot_Area - 1.019522e+06 t126_leaf3 >= -1e+06
 T126_L3_f6_le: const_Year_Remod_Add + 998038 t126_leaf3 <= 1e+06
 T126_R4_f17_ge: x_1st_Flr_SF - 576 t126_leaf4 >= 0
 T126_R4_f1_ge: const_Lot_Area - 1.019522e+06 t126_leaf4 >= -1e+06
 T126_R4_f6_ge: const_Year_Remod_Add - 1.001962e+06 t126_leaf4 >= -1e+06
 TREE_127_ONEHOT: t127_leaf0 + t127_leaf1 + t127_leaf2 + t127_leaf3
   + t127_leaf4 + t127_leaf5 = 1
 T127_L0_f3_le: Overall_Qual_calc + 999997 t127_leaf0 <= 1e+06
 T127_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t127_leaf0
   <= 1e+06
 T127_L1_f3_le: Overall_Qual_calc + 999997 t127_leaf1 <= 1e+06
 T127_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t127_leaf1
   >= -1e+06
 T127_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t127_leaf2 >= -1e+06
 T127_L2_f5_le: const_Year_Built + 998087 t127_leaf2 <= 1e+06
 T127_L2_f16_le: x_Heating_QC + t127_leaf2 <= 4
 T127_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t127_leaf3 >= -1e+06
 T127_L3_f5_le: const_Year_Built + 998087 t127_leaf3 <= 1e+06
 T127_R3_f16_ge: x_Heating_QC - 4 t127_leaf3 >= -1
 T127_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t127_leaf4 >= -1e+06
 T127_R4_f5_ge: const_Year_Built - 1.001913e+06 t127_leaf4 >= -1e+06
 T127_L4_f1_le: const_Lot_Area + 996325 t127_leaf4 <= 1e+06
 T127_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t127_leaf5 >= -1e+06
 T127_R5_f5_ge: const_Year_Built - 1.001913e+06 t127_leaf5 >= -1e+06
 T127_R5_f1_ge: const_Lot_Area - 1.003675e+06 t127_leaf5 >= -1e+06
 TREE_128_ONEHOT: t128_leaf0 + t128_leaf1 + t128_leaf2 + t128_leaf3
   + t128_leaf4 = 1
 T128_L0_f3_le: Overall_Qual_calc + 999997 t128_leaf0 <= 1e+06
 T128_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t128_leaf1 >= -1e+06
 T128_L1_f4_le: const_Overall_Cond + 999993 t128_leaf1 <= 1e+06
 T128_L1_f31_le: const_Garage_Yr_Blt + 997995 t128_leaf1 <= 1e+06
 T128_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t128_leaf2 >= -1e+06
 T128_L2_f4_le: const_Overall_Cond + 999993 t128_leaf2 <= 1e+06
 T128_R2_f31_ge: const_Garage_Yr_Blt - 1.002005e+06 t128_leaf2 >= -1e+06
 T128_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t128_leaf3 >= -1e+06
 T128_R3_f4_ge: const_Overall_Cond - 1.000007e+06 t128_leaf3 >= -1e+06
 T128_L3_f17_le: x_1st_Flr_SF + 3680 t128_leaf3 <= 5000
 T128_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t128_leaf4 >= -1e+06
 T128_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t128_leaf4 >= -1e+06
 T128_R4_f17_ge: x_1st_Flr_SF - 1320 t128_leaf4 >= 0
 TREE_129_ONEHOT: t129_leaf0 + t129_leaf1 + t129_leaf2 + t129_leaf3
   + t129_leaf4 + t129_leaf5 = 1
 T129_L0_f8_le: x_Exter_Qual + 2 t129_leaf0 <= 4
 T129_L0_f297_le: const_Sale_Condition_Normal + 999999 t129_leaf0 <= 1e+06
 T129_L1_f8_le: x_Exter_Qual + 2 t129_leaf1 <= 4
 T129_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t129_leaf1
   >= -1e+06
 T129_R2_f8_ge: x_Exter_Qual - 3 t129_leaf2 >= -1
 T129_L2_f96_le: const_Neighborhood_Edwards + 999999 t129_leaf2 <= 1e+06
 T129_L2_f3_le: Overall_Qual_calc + 999990 t129_leaf2 <= 1e+06
 T129_R3_f8_ge: x_Exter_Qual - 3 t129_leaf3 >= -1
 T129_L3_f96_le: const_Neighborhood_Edwards + 999999 t129_leaf3 <= 1e+06
 T129_R3_f3_ge: Overall_Qual_calc - 1.00001e+06 t129_leaf3 >= -1e+06
 T129_R4_f8_ge: x_Exter_Qual - 3 t129_leaf4 >= -1
 T129_R4_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t129_leaf4
   >= -1e+06
 T129_L4_f0_le: const_Lot_Frontage + 999934 t129_leaf4 <= 1e+06
 T129_R5_f8_ge: x_Exter_Qual - 3 t129_leaf5 >= -1
 T129_R5_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t129_leaf5
   >= -1e+06
 T129_R5_f0_ge: const_Lot_Frontage - 1.000066e+06 t129_leaf5 >= -1e+06
 TREE_130_ONEHOT: t130_leaf0 + t130_leaf1 + t130_leaf2 + t130_leaf3
   + t130_leaf4 + t130_leaf5 + t130_leaf6 + t130_leaf7 = 1
 T130_L0_f36_le: x_Wood_Deck_SF + 99805 t130_leaf0 <= 100000
 T130_L0_f95_le: const_Neighborhood_Crawfor + 999999 t130_leaf0 <= 1e+06
 T130_L0_f6_le: const_Year_Remod_Add + 998005 t130_leaf0 <= 1e+06
 T130_L1_f36_le: x_Wood_Deck_SF + 99805 t130_leaf1 <= 100000
 T130_L1_f95_le: const_Neighborhood_Crawfor + 999999 t130_leaf1 <= 1e+06
 T130_R1_f6_ge: const_Year_Remod_Add - 1.001995e+06 t130_leaf1 >= -1e+06
 T130_L2_f36_le: x_Wood_Deck_SF + 99805 t130_leaf2 <= 100000
 T130_R2_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t130_leaf2
   >= -1e+06
 T130_L2_f0_le: const_Lot_Frontage + 999946 t130_leaf2 <= 1e+06
 T130_L3_f36_le: x_Wood_Deck_SF + 99805 t130_leaf3 <= 100000
 T130_R3_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t130_leaf3
   >= -1e+06
 T130_R3_f0_ge: const_Lot_Frontage - 1.000054e+06 t130_leaf3 >= -1e+06
 T130_R4_f36_ge: x_Wood_Deck_SF - 195 t130_leaf4 >= 0
 T130_L4_f7_le: x_Mas_Vnr_Area + 1700 t130_leaf4 <= 2000
 T130_L4_f14_le: x_Bsmt_Unf_SF + 98760 t130_leaf4 <= 100000
 T130_R5_f36_ge: x_Wood_Deck_SF - 195 t130_leaf5 >= 0
 T130_L5_f7_le: x_Mas_Vnr_Area + 1700 t130_leaf5 <= 2000
 T130_R5_f14_ge: x_Bsmt_Unf_SF - 1240 t130_leaf5 >= 0
 T130_R6_f36_ge: x_Wood_Deck_SF - 195 t130_leaf6 >= 0
 T130_R6_f7_ge: x_Mas_Vnr_Area - 300 t130_leaf6 >= 0
 T130_L6_f30_le: x_Fireplace_Qu + 2 t130_leaf6 <= 4
 T130_R7_f36_ge: x_Wood_Deck_SF - 195 t130_leaf7 >= 0
 T130_R7_f7_ge: x_Mas_Vnr_Area - 300 t130_leaf7 >= 0
 T130_R7_f30_ge: x_Fireplace_Qu - 3 t130_leaf7 >= -1
 TREE_131_ONEHOT: t131_leaf0 + t131_leaf1 + t131_leaf2 + t131_leaf3
   + t131_leaf4 + t131_leaf5 + t131_leaf6 + t131_leaf7 = 1
 T131_L0_f9_le: x_Exter_Cond + 2 t131_leaf0 <= 4
 T131_L0_f297_le: const_Sale_Condition_Normal + 999999 t131_leaf0 <= 1e+06
 T131_L0_f28_le: x_TotRms_AbvGrd + 9 t131_leaf0 <= 15
 T131_L1_f9_le: x_Exter_Cond + 2 t131_leaf1 <= 4
 T131_L1_f297_le: const_Sale_Condition_Normal + 999999 t131_leaf1 <= 1e+06
 T131_R1_f28_ge: x_TotRms_AbvGrd - 6 t131_leaf1 >= 0
 T131_L2_f9_le: x_Exter_Cond + 2 t131_leaf2 <= 4
 T131_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t131_leaf2
   >= -1e+06
 T131_L2_f17_le: x_1st_Flr_SF + 4049 t131_leaf2 <= 5000
 T131_L3_f9_le: x_Exter_Cond + 2 t131_leaf3 <= 4
 T131_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t131_leaf3
   >= -1e+06
 T131_R3_f17_ge: x_1st_Flr_SF - 951 t131_leaf3 >= 0
 T131_R4_f9_ge: x_Exter_Cond - 3 t131_leaf4 >= -1
 T131_L4_f40_le: x_Screen_Porch + 99858 t131_leaf4 <= 100000
 T131_L4_f44_le: const_Yr_Sold + 997993 t131_leaf4 <= 1e+06
 T131_R5_f9_ge: x_Exter_Cond - 3 t131_leaf5 >= -1
 T131_L5_f40_le: x_Screen_Porch + 99858 t131_leaf5 <= 100000
 T131_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t131_leaf5 >= -1e+06
 T131_R6_f9_ge: x_Exter_Cond - 3 t131_leaf6 >= -1
 T131_R6_f40_ge: x_Screen_Porch - 142 t131_leaf6 >= 0
 T131_L6_f1_le: const_Lot_Area + 986948 t131_leaf6 <= 1e+06
 T131_R7_f9_ge: x_Exter_Cond - 3 t131_leaf7 >= -1
 T131_R7_f40_ge: x_Screen_Porch - 142 t131_leaf7 >= 0
 T131_R7_f1_ge: const_Lot_Area - 1.013052e+06 t131_leaf7 >= -1e+06
 TREE_132_ONEHOT: t132_leaf0 + t132_leaf1 + t132_leaf2 + t132_leaf3
   + t132_leaf4 = 1
 T132_L0_f23_le: x_Full_Bath + t132_leaf0 <= 4
 T132_L0_f12_le: x_BsmtFin_SF_1 + 998268 t132_leaf0 <= 1e+06
 T132_L0_f12_le: x_BsmtFin_SF_1 + 999252 t132_leaf0 <= 1e+06
 T132_L1_f23_le: x_Full_Bath + t132_leaf1 <= 4
 T132_L1_f12_le: x_BsmtFin_SF_1 + 998268 t132_leaf1 <= 1e+06
 T132_R1_f12_ge: x_BsmtFin_SF_1 - 748 t132_leaf1 >= 0
 T132_L2_f23_le: x_Full_Bath + t132_leaf2 <= 4
 T132_R2_f12_ge: x_BsmtFin_SF_1 - 1732 t132_leaf2 >= 0
 T132_R3_f23_ge: x_Full_Bath - 3 t132_leaf3 >= 0
 T132_L3_f174_le: x_ex1_is_VinylSd <= 1
 T132_R4_f23_ge: x_Full_Bath - 3 t132_leaf4 >= 0
 T132_R4_f174_ge: x_ex1_is_VinylSd - t132_leaf4 >= 0
 TREE_133_ONEHOT: t133_leaf0 + t133_leaf1 + t133_leaf2 + t133_leaf3
   + t133_leaf4 = 1
 T133_L0_f62_le: const_MS_Zoning_C_(all) + 999999 t133_leaf0 <= 1e+06
 T133_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t133_leaf0 <= 1e+06
 T133_L0_f24_le: x_Half_Bath + 4 t133_leaf0 <= 5
 T133_L1_f62_le: const_MS_Zoning_C_(all) + 999999 t133_leaf1 <= 1e+06
 T133_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t133_leaf1 <= 1e+06
 T133_R1_f24_ge: x_Half_Bath - t133_leaf1 >= 0
 T133_L2_f62_le: const_MS_Zoning_C_(all) + 999999 t133_leaf2 <= 1e+06
 T133_R2_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t133_leaf2
   >= -1e+06
 T133_L2_f29_le: const_Fireplaces + 999998 t133_leaf2 <= 1e+06
 T133_L3_f62_le: const_MS_Zoning_C_(all) + 999999 t133_leaf3 <= 1e+06
 T133_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t133_leaf3
   >= -1e+06
 T133_R3_f29_ge: const_Fireplaces - 1.000002e+06 t133_leaf3 >= -1e+06
 T133_R4_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t133_leaf4
   >= -1e+06
 TREE_134_ONEHOT: t134_leaf0 + t134_leaf1 + t134_leaf2 + t134_leaf3
   + t134_leaf4 = 1
 T134_L0_f15_le: const_Total_Bsmt_SF + 997864 t134_leaf0 <= 1e+06
 T134_L0_f16_le: x_Heating_QC + t134_leaf0 <= 4
 T134_L0_f4_le: const_Overall_Cond + 999994 t134_leaf0 <= 1e+06
 T134_L1_f15_le: const_Total_Bsmt_SF + 997864 t134_leaf1 <= 1e+06
 T134_L1_f16_le: x_Heating_QC + t134_leaf1 <= 4
 T134_R1_f4_ge: const_Overall_Cond - 1.000006e+06 t134_leaf1 >= -1e+06
 T134_L2_f15_le: const_Total_Bsmt_SF + 997864 t134_leaf2 <= 1e+06
 T134_R2_f16_ge: x_Heating_QC - 4 t134_leaf2 >= -1
 T134_L2_f44_le: const_Yr_Sold + 997991 t134_leaf2 <= 1e+06
 T134_L3_f15_le: const_Total_Bsmt_SF + 997864 t134_leaf3 <= 1e+06
 T134_R3_f16_ge: x_Heating_QC - 4 t134_leaf3 >= -1
 T134_R3_f44_ge: const_Yr_Sold - 1.002009e+06 t134_leaf3 >= -1e+06
 T134_R4_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t134_leaf4 >= -1e+06
 TREE_135_ONEHOT: t135_leaf0 + t135_leaf1 + t135_leaf2 + t135_leaf3
   + t135_leaf4 + t135_leaf5 + t135_leaf6 = 1
 T135_L0_f44_le: const_Yr_Sold + 997993 t135_leaf0 <= 1e+06
 T135_L0_f0_le: const_Lot_Frontage + 999969 t135_leaf0 <= 1e+06
 T135_L1_f44_le: const_Yr_Sold + 997993 t135_leaf1 <= 1e+06
 T135_R1_f0_ge: const_Lot_Frontage - 1.000031e+06 t135_leaf1 >= -1e+06
 T135_L1_f117_le: const_Condition_1_Artery + 999999 t135_leaf1 <= 1e+06
 T135_L2_f44_le: const_Yr_Sold + 997993 t135_leaf2 <= 1e+06
 T135_R2_f0_ge: const_Lot_Frontage - 1.000031e+06 t135_leaf2 >= -1e+06
 T135_R2_f117_ge: const_Condition_1_Artery - 1.000001e+06 t135_leaf2
   >= -1e+06
 T135_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t135_leaf3 >= -1e+06
 T135_L3_f16_le: x_Heating_QC + t135_leaf3 <= 4
 T135_L3_f9_le: x_Exter_Cond + 2 t135_leaf3 <= 4
 T135_R4_f44_ge: const_Yr_Sold - 1.002007e+06 t135_leaf4 >= -1e+06
 T135_L4_f16_le: x_Heating_QC + t135_leaf4 <= 4
 T135_R4_f9_ge: x_Exter_Cond - 3 t135_leaf4 >= -1
 T135_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t135_leaf5 >= -1e+06
 T135_R5_f16_ge: x_Heating_QC - 4 t135_leaf5 >= -1
 T135_L5_f239_le: const_FunctioNo_aplical_Maj2 + 999999 t135_leaf5
   <= 1e+06
 T135_R6_f44_ge: const_Yr_Sold - 1.002007e+06 t135_leaf6 >= -1e+06
 T135_R6_f16_ge: x_Heating_QC - 4 t135_leaf6 >= -1
 T135_R6_f239_ge: const_FunctioNo_aplical_Maj2 - 1.000001e+06 t135_leaf6
   >= -1e+06
 TREE_136_ONEHOT: t136_leaf0 + t136_leaf1 + t136_leaf2 + t136_leaf3
   + t136_leaf4 = 1
 T136_L0_f23_le: x_Full_Bath + t136_leaf0 <= 4
 T136_L0_f33_le: x_Garage_Area + 98846 t136_leaf0 <= 100000
 T136_L0_f44_le: const_Yr_Sold + 997992 t136_leaf0 <= 1e+06
 T136_L1_f23_le: x_Full_Bath + t136_leaf1 <= 4
 T136_L1_f33_le: x_Garage_Area + 98846 t136_leaf1 <= 100000
 T136_R1_f44_ge: const_Yr_Sold - 1.002008e+06 t136_leaf1 >= -1e+06
 T136_L2_f23_le: x_Full_Bath + t136_leaf2 <= 4
 T136_R2_f33_ge: x_Garage_Area - 1154 t136_leaf2 >= 0
 T136_R3_f23_ge: x_Full_Bath - 3 t136_leaf3 >= 0
 T136_L3_f255_le: const_Garage_Finish_RFn + 999999 t136_leaf3 <= 1e+06
 T136_R4_f23_ge: x_Full_Bath - 3 t136_leaf4 >= 0
 T136_R4_f255_ge: const_Garage_Finish_RFn - 1.000001e+06 t136_leaf4
   >= -1e+06
 TREE_137_ONEHOT: t137_leaf0 + t137_leaf1 + t137_leaf2 + t137_leaf3
   + t137_leaf4 + t137_leaf5 + t137_leaf6 + t137_leaf7 = 1
 T137_L0_f1_le: const_Lot_Area + 994000 t137_leaf0 <= 1e+06
 T137_L0_f3_le: Overall_Qual_calc + 999992 t137_leaf0 <= 1e+06
 T137_L0_f33_le: x_Garage_Area + 99816 t137_leaf0 <= 100000
 T137_L1_f1_le: const_Lot_Area + 994000 t137_leaf1 <= 1e+06
 T137_L1_f3_le: Overall_Qual_calc + 999992 t137_leaf1 <= 1e+06
 T137_R1_f33_ge: x_Garage_Area - 184 t137_leaf1 >= 0
 T137_L2_f1_le: const_Lot_Area + 994000 t137_leaf2 <= 1e+06
 T137_R2_f3_ge: Overall_Qual_calc - 1.000008e+06 t137_leaf2 >= -1e+06
 T137_L2_f12_le: x_BsmtFin_SF_1 + 999676 t137_leaf2 <= 1e+06
 T137_L3_f1_le: const_Lot_Area + 994000 t137_leaf3 <= 1e+06
 T137_R3_f3_ge: Overall_Qual_calc - 1.000008e+06 t137_leaf3 >= -1e+06
 T137_R3_f12_ge: x_BsmtFin_SF_1 - 324 t137_leaf3 >= 0
 T137_R4_f1_ge: const_Lot_Area - 1.006e+06 t137_leaf4 >= -1e+06
 T137_L4_f18_le: const_2nd_Flr_SF + 998573 t137_leaf4 <= 1e+06
 T137_L4_f44_le: const_Yr_Sold + 997993 t137_leaf4 <= 1e+06
 T137_R5_f1_ge: const_Lot_Area - 1.006e+06 t137_leaf5 >= -1e+06
 T137_L5_f18_le: const_2nd_Flr_SF + 998573 t137_leaf5 <= 1e+06
 T137_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t137_leaf5 >= -1e+06
 T137_R6_f1_ge: const_Lot_Area - 1.006e+06 t137_leaf6 >= -1e+06
 T137_R6_f18_ge: const_2nd_Flr_SF - 1.001427e+06 t137_leaf6 >= -1e+06
 T137_L6_f1_le: const_Lot_Area + 990200 t137_leaf6 <= 1e+06
 T137_R7_f1_ge: const_Lot_Area - 1.006e+06 t137_leaf7 >= -1e+06
 T137_R7_f18_ge: const_2nd_Flr_SF - 1.001427e+06 t137_leaf7 >= -1e+06
 T137_R7_f1_ge: const_Lot_Area - 1.0098e+06 t137_leaf7 >= -1e+06
 TREE_138_ONEHOT: t138_leaf0 + t138_leaf1 + t138_leaf2 + t138_leaf3
   + t138_leaf4 + t138_leaf5 + t138_leaf6 = 1
 T138_L0_f9_le: x_Exter_Cond + 2 t138_leaf0 <= 4
 T138_L0_f3_le: Overall_Qual_calc + 999997 t138_leaf0 <= 1e+06
 T138_L1_f9_le: x_Exter_Cond + 2 t138_leaf1 <= 4
 T138_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t138_leaf1 >= -1e+06
 T138_L1_f14_le: x_Bsmt_Unf_SF + 99280 t138_leaf1 <= 100000
 T138_L2_f9_le: x_Exter_Cond + 2 t138_leaf2 <= 4
 T138_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t138_leaf2 >= -1e+06
 T138_R2_f14_ge: x_Bsmt_Unf_SF - 720 t138_leaf2 >= 0
 T138_R3_f9_ge: x_Exter_Cond - 3 t138_leaf3 >= -1
 T138_L3_f1_le: const_Lot_Area + 994024 t138_leaf3 <= 1e+06
 T138_L3_f33_le: x_Garage_Area + 99616 t138_leaf3 <= 100000
 T138_R4_f9_ge: x_Exter_Cond - 3 t138_leaf4 >= -1
 T138_L4_f1_le: const_Lot_Area + 994024 t138_leaf4 <= 1e+06
 T138_R4_f33_ge: x_Garage_Area - 384 t138_leaf4 >= 0
 T138_R5_f9_ge: x_Exter_Cond - 3 t138_leaf5 >= -1
 T138_R5_f1_ge: const_Lot_Area - 1.005976e+06 t138_leaf5 >= -1e+06
 T138_L5_f29_le: const_Fireplaces + 999997 t138_leaf5 <= 1e+06
 T138_R6_f9_ge: x_Exter_Cond - 3 t138_leaf6 >= -1
 T138_R6_f1_ge: const_Lot_Area - 1.005976e+06 t138_leaf6 >= -1e+06
 T138_R6_f29_ge: const_Fireplaces - 1.000003e+06 t138_leaf6 >= -1e+06
 TREE_139_ONEHOT: t139_leaf0 + t139_leaf1 + t139_leaf2 + t139_leaf3
   + t139_leaf4 + t139_leaf5 + t139_leaf6 + t139_leaf7 = 1
 T139_L0_f4_le: const_Overall_Cond + 999993 t139_leaf0 <= 1e+06
 T139_L0_f109_le: const_Neighborhood_OldTown + 999999 t139_leaf0 <= 1e+06
 T139_L0_f37_le: x_Open_Porch_SF + 99955 t139_leaf0 <= 100000
 T139_L1_f4_le: const_Overall_Cond + 999993 t139_leaf1 <= 1e+06
 T139_L1_f109_le: const_Neighborhood_OldTown + 999999 t139_leaf1 <= 1e+06
 T139_R1_f37_ge: x_Open_Porch_SF - 45 t139_leaf1 >= 0
 T139_L2_f4_le: const_Overall_Cond + 999993 t139_leaf2 <= 1e+06
 T139_R2_f109_ge: const_Neighborhood_OldTown - 1.000001e+06 t139_leaf2
   >= -1e+06
 T139_L2_f200_le: const_Foundation_BrkTil + 999999 t139_leaf2 <= 1e+06
 T139_L3_f4_le: const_Overall_Cond + 999993 t139_leaf3 <= 1e+06
 T139_R3_f109_ge: const_Neighborhood_OldTown - 1.000001e+06 t139_leaf3
   >= -1e+06
 T139_R3_f200_ge: const_Foundation_BrkTil - 1.000001e+06 t139_leaf3
   >= -1e+06
 T139_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t139_leaf4 >= -1e+06
 T139_L4_f119_le: const_Condition_1_Norm + 999999 t139_leaf4 <= 1e+06
 T139_L4_f15_le: const_Total_Bsmt_SF + 999096 t139_leaf4 <= 1e+06
 T139_R5_f4_ge: const_Overall_Cond - 1.000007e+06 t139_leaf5 >= -1e+06
 T139_L5_f119_le: const_Condition_1_Norm + 999999 t139_leaf5 <= 1e+06
 T139_R5_f15_ge: const_Total_Bsmt_SF - 1.000904e+06 t139_leaf5 >= -1e+06
 T139_R6_f4_ge: const_Overall_Cond - 1.000007e+06 t139_leaf6 >= -1e+06
 T139_R6_f119_ge: const_Condition_1_Norm - 1.000001e+06 t139_leaf6
   >= -1e+06
 T139_L6_f245_le: const_FunctioNo_aplical_Typ + 999999 t139_leaf6 <= 1e+06
 T139_R7_f4_ge: const_Overall_Cond - 1.000007e+06 t139_leaf7 >= -1e+06
 T139_R7_f119_ge: const_Condition_1_Norm - 1.000001e+06 t139_leaf7
   >= -1e+06
 T139_R7_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t139_leaf7
   >= -1e+06
 TREE_140_ONEHOT: t140_leaf0 + t140_leaf1 + t140_leaf2 + t140_leaf3
   + t140_leaf4 + t140_leaf5 + t140_leaf6 + t140_leaf7 = 1
 T140_L0_f11_le: x_Bsmt_Cond + 2 t140_leaf0 <= 4
 T140_L0_f9_le: x_Exter_Cond + 2 t140_leaf0 <= 4
 T140_L0_f0_le: const_Lot_Frontage + 999935 t140_leaf0 <= 1e+06
 T140_L1_f11_le: x_Bsmt_Cond + 2 t140_leaf1 <= 4
 T140_L1_f9_le: x_Exter_Cond + 2 t140_leaf1 <= 4
 T140_R1_f0_ge: const_Lot_Frontage - 1.000065e+06 t140_leaf1 >= -1e+06
 T140_L2_f11_le: x_Bsmt_Cond + 2 t140_leaf2 <= 4
 T140_R2_f9_ge: x_Exter_Cond - 3 t140_leaf2 >= -1
 T140_L2_f17_le: x_1st_Flr_SF + 3876 t140_leaf2 <= 5000
 T140_L3_f11_le: x_Bsmt_Cond + 2 t140_leaf3 <= 4
 T140_R3_f9_ge: x_Exter_Cond - 3 t140_leaf3 >= -1
 T140_R3_f17_ge: x_1st_Flr_SF - 1124 t140_leaf3 >= 0
 T140_R4_f11_ge: x_Bsmt_Cond - 3 t140_leaf4 >= -1
 T140_L4_f4_le: const_Overall_Cond + 999993 t140_leaf4 <= 1e+06
 T140_L4_f37_le: x_Open_Porch_SF + 99970 t140_leaf4 <= 100000
 T140_R5_f11_ge: x_Bsmt_Cond - 3 t140_leaf5 >= -1
 T140_L5_f4_le: const_Overall_Cond + 999993 t140_leaf5 <= 1e+06
 T140_R5_f37_ge: x_Open_Porch_SF - 30 t140_leaf5 >= 0
 T140_R6_f11_ge: x_Bsmt_Cond - 3 t140_leaf6 >= -1
 T140_R6_f4_ge: const_Overall_Cond - 1.000007e+06 t140_leaf6 >= -1e+06
 T140_L6_f1_le: const_Lot_Area + 981614 t140_leaf6 <= 1e+06
 T140_R7_f11_ge: x_Bsmt_Cond - 3 t140_leaf7 >= -1
 T140_R7_f4_ge: const_Overall_Cond - 1.000007e+06 t140_leaf7 >= -1e+06
 T140_R7_f1_ge: const_Lot_Area - 1.018386e+06 t140_leaf7 >= -1e+06
 TREE_141_ONEHOT: t141_leaf0 + t141_leaf1 + t141_leaf2 + t141_leaf3
   + t141_leaf4 + t141_leaf5 = 1
 T141_L0_f5_le: const_Year_Built + 998075 t141_leaf0 <= 1e+06
 T141_L0_f33_le: x_Garage_Area + 99375 t141_leaf0 <= 100000
 T141_L0_f119_le: const_Condition_1_Norm + 999999 t141_leaf0 <= 1e+06
 T141_L1_f5_le: const_Year_Built + 998075 t141_leaf1 <= 1e+06
 T141_L1_f33_le: x_Garage_Area + 99375 t141_leaf1 <= 100000
 T141_R1_f119_ge: const_Condition_1_Norm - 1.000001e+06 t141_leaf1
   >= -1e+06
 T141_L2_f5_le: const_Year_Built + 998075 t141_leaf2 <= 1e+06
 T141_R2_f33_ge: x_Garage_Area - 625 t141_leaf2 >= 0
 T141_R3_f5_ge: const_Year_Built - 1.001925e+06 t141_leaf3 >= -1e+06
 T141_L3_f18_le: const_2nd_Flr_SF + 998371 t141_leaf3 <= 1e+06
 T141_L3_f245_le: const_FunctioNo_aplical_Typ + 999999 t141_leaf3 <= 1e+06
 T141_R4_f5_ge: const_Year_Built - 1.001925e+06 t141_leaf4 >= -1e+06
 T141_L4_f18_le: const_2nd_Flr_SF + 998371 t141_leaf4 <= 1e+06
 T141_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t141_leaf4
   >= -1e+06
 T141_R5_f5_ge: const_Year_Built - 1.001925e+06 t141_leaf5 >= -1e+06
 T141_R5_f18_ge: const_2nd_Flr_SF - 1.001629e+06 t141_leaf5 >= -1e+06
 TREE_142_ONEHOT: t142_leaf0 + t142_leaf1 + t142_leaf2 + t142_leaf3
   + t142_leaf4 = 1
 T142_L0_f17_le: x_1st_Flr_SF + 2485 t142_leaf0 <= 5000
 T142_L0_f289_le: const_Sale_Type_New + 999999 t142_leaf0 <= 1e+06
 T142_L0_f20_le: x_Gr_Liv_Area + 7367 t142_leaf0 <= 10000
 T142_L1_f17_le: x_1st_Flr_SF + 2485 t142_leaf1 <= 5000
 T142_L1_f289_le: const_Sale_Type_New + 999999 t142_leaf1 <= 1e+06
 T142_R1_f20_ge: x_Gr_Liv_Area - 2633 t142_leaf1 >= 0
 T142_L2_f17_le: x_1st_Flr_SF + 2485 t142_leaf2 <= 5000
 T142_R2_f289_ge: const_Sale_Type_New - 1.000001e+06 t142_leaf2 >= -1e+06
 T142_L2_f20_le: x_Gr_Liv_Area + 8640 t142_leaf2 <= 10000
 T142_L3_f17_le: x_1st_Flr_SF + 2485 t142_leaf3 <= 5000
 T142_R3_f289_ge: const_Sale_Type_New - 1.000001e+06 t142_leaf3 >= -1e+06
 T142_R3_f20_ge: x_Gr_Liv_Area - 1360 t142_leaf3 >= 0
 T142_R4_f17_ge: x_1st_Flr_SF - 2515 t142_leaf4 >= 0
 TREE_143_ONEHOT: t143_leaf0 + t143_leaf1 + t143_leaf2 + t143_leaf3
   + t143_leaf4 + t143_leaf5 + t143_leaf6 = 1
 T143_L0_f20_le: x_Gr_Liv_Area + 9253 t143_leaf0 <= 10000
 T143_L0_f20_le: x_Gr_Liv_Area + 9280 t143_leaf0 <= 10000
 T143_L0_f28_le: x_TotRms_AbvGrd + 11 t143_leaf0 <= 15
 T143_L1_f20_le: x_Gr_Liv_Area + 9253 t143_leaf1 <= 10000
 T143_L1_f20_le: x_Gr_Liv_Area + 9280 t143_leaf1 <= 10000
 T143_R1_f28_ge: x_TotRms_AbvGrd - 4 t143_leaf1 >= 0
 T143_L2_f20_le: x_Gr_Liv_Area + 9253 t143_leaf2 <= 10000
 T143_R2_f20_ge: x_Gr_Liv_Area - 720 t143_leaf2 >= 0
 T143_R3_f20_ge: x_Gr_Liv_Area - 747 t143_leaf3 >= 0
 T143_L3_f164_le: x_ex1_is_BrkFace <= 1
 T143_L3_f4_le: const_Overall_Cond + 999994 t143_leaf3 <= 1e+06
 T143_R4_f20_ge: x_Gr_Liv_Area - 747 t143_leaf4 >= 0
 T143_L4_f164_le: x_ex1_is_BrkFace <= 1
 T143_R4_f4_ge: const_Overall_Cond - 1.000006e+06 t143_leaf4 >= -1e+06
 T143_R5_f20_ge: x_Gr_Liv_Area - 747 t143_leaf5 >= 0
 T143_R5_f164_ge: x_ex1_is_BrkFace - t143_leaf5 >= 0
 T143_L5_f245_le: const_FunctioNo_aplical_Typ + 999999 t143_leaf5 <= 1e+06
 T143_R6_f20_ge: x_Gr_Liv_Area - 747 t143_leaf6 >= 0
 T143_R6_f164_ge: x_ex1_is_BrkFace - t143_leaf6 >= 0
 T143_R6_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t143_leaf6
   >= -1e+06
 TREE_144_ONEHOT: t144_leaf0 + t144_leaf1 + t144_leaf2 + t144_leaf3
   + t144_leaf4 + t144_leaf5 = 1
 T144_L0_f95_le: const_Neighborhood_Crawfor + 999999 t144_leaf0 <= 1e+06
 T144_L0_f44_le: const_Yr_Sold + 997992 t144_leaf0 <= 1e+06
 T144_L0_f23_le: x_Full_Bath + t144_leaf0 <= 4
 T144_L1_f95_le: const_Neighborhood_Crawfor + 999999 t144_leaf1 <= 1e+06
 T144_L1_f44_le: const_Yr_Sold + 997992 t144_leaf1 <= 1e+06
 T144_R1_f23_ge: x_Full_Bath - 3 t144_leaf1 >= 0
 T144_L2_f95_le: const_Neighborhood_Crawfor + 999999 t144_leaf2 <= 1e+06
 T144_R2_f44_ge: const_Yr_Sold - 1.002008e+06 t144_leaf2 >= -1e+06
 T144_L2_f297_le: const_Sale_Condition_Normal + 999999 t144_leaf2 <= 1e+06
 T144_L3_f95_le: const_Neighborhood_Crawfor + 999999 t144_leaf3 <= 1e+06
 T144_R3_f44_ge: const_Yr_Sold - 1.002008e+06 t144_leaf3 >= -1e+06
 T144_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t144_leaf3
   >= -1e+06
 T144_R4_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t144_leaf4
   >= -1e+06
 T144_L4_f1_le: const_Lot_Area + 995165 t144_leaf4 <= 1e+06
 T144_R5_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t144_leaf5
   >= -1e+06
 T144_R5_f1_ge: const_Lot_Area - 1.004835e+06 t144_leaf5 >= -1e+06
 TREE_145_ONEHOT: t145_leaf0 + t145_leaf1 + t145_leaf2 + t145_leaf3
   + t145_leaf4 + t145_leaf5 = 1
 T145_L0_f161_le: x_ex1_is_AsbShng <= 1
 T145_L0_f21_le: const_Bsmt_Full_Bath + 999999 t145_leaf0 <= 1e+06
 T145_L0_f164_le: x_ex1_is_BrkFace <= 1
 T145_L1_f161_le: x_ex1_is_AsbShng <= 1
 T145_L1_f21_le: const_Bsmt_Full_Bath + 999999 t145_leaf1 <= 1e+06
 T145_R1_f164_ge: x_ex1_is_BrkFace - t145_leaf1 >= 0
 T145_L2_f161_le: x_ex1_is_AsbShng <= 1
 T145_R2_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t145_leaf2 >= -1e+06
 T145_L2_f190_le: x_ex2_is_Stucco <= 1
 T145_L3_f161_le: x_ex1_is_AsbShng <= 1
 T145_R3_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t145_leaf3 >= -1e+06
 T145_R3_f190_ge: x_ex2_is_Stucco - t145_leaf3 >= 0
 T145_R4_f161_ge: x_ex1_is_AsbShng - t145_leaf4 >= 0
 T145_L4_f1_le: const_Lot_Area + 991070 t145_leaf4 <= 1e+06
 T145_R5_f161_ge: x_ex1_is_AsbShng - t145_leaf5 >= 0
 T145_R5_f1_ge: const_Lot_Area - 1.00893e+06 t145_leaf5 >= -1e+06
 TREE_146_ONEHOT: t146_leaf0 + t146_leaf1 + t146_leaf2 + t146_leaf3
   + t146_leaf4 + t146_leaf5 + t146_leaf6 + t146_leaf7 = 1
 T146_L0_f12_le: x_BsmtFin_SF_1 + 999575 t146_leaf0 <= 1e+06
 T146_L0_f16_le: x_Heating_QC + t146_leaf0 <= 4
 T146_L0_f191_le: x_ex2_is_VinylSd <= 1
 T146_L1_f12_le: x_BsmtFin_SF_1 + 999575 t146_leaf1 <= 1e+06
 T146_L1_f16_le: x_Heating_QC + t146_leaf1 <= 4
 T146_R1_f191_ge: x_ex2_is_VinylSd - t146_leaf1 >= 0
 T146_L2_f12_le: x_BsmtFin_SF_1 + 999575 t146_leaf2 <= 1e+06
 T146_R2_f16_ge: x_Heating_QC - 4 t146_leaf2 >= -1
 T146_L2_f1_le: const_Lot_Area + 983221 t146_leaf2 <= 1e+06
 T146_L3_f12_le: x_BsmtFin_SF_1 + 999575 t146_leaf3 <= 1e+06
 T146_R3_f16_ge: x_Heating_QC - 4 t146_leaf3 >= -1
 T146_R3_f1_ge: const_Lot_Area - 1.016779e+06 t146_leaf3 >= -1e+06
 T146_R4_f12_ge: x_BsmtFin_SF_1 - 425 t146_leaf4 >= 0
 T146_L4_f15_le: const_Total_Bsmt_SF + 998502 t146_leaf4 <= 1e+06
 T146_L4_f11_le: x_Bsmt_Cond + t146_leaf4 <= 4
 T146_R5_f12_ge: x_BsmtFin_SF_1 - 425 t146_leaf5 >= 0
 T146_L5_f15_le: const_Total_Bsmt_SF + 998502 t146_leaf5 <= 1e+06
 T146_R5_f11_ge: x_Bsmt_Cond - 4 t146_leaf5 >= -1
 T146_R6_f12_ge: x_BsmtFin_SF_1 - 425 t146_leaf6 >= 0
 T146_R6_f15_ge: const_Total_Bsmt_SF - 1.001498e+06 t146_leaf6 >= -1e+06
 T146_L6_f208_le: const_Bsmt_Exposure_Mn + 999999 t146_leaf6 <= 1e+06
 T146_R7_f12_ge: x_BsmtFin_SF_1 - 425 t146_leaf7 >= 0
 T146_R7_f15_ge: const_Total_Bsmt_SF - 1.001498e+06 t146_leaf7 >= -1e+06
 T146_R7_f208_ge: const_Bsmt_Exposure_Mn - 1.000001e+06 t146_leaf7
   >= -1e+06
 TREE_147_ONEHOT: t147_leaf0 + t147_leaf1 + t147_leaf2 + t147_leaf3
   + t147_leaf4 + t147_leaf5 = 1
 T147_L0_f17_le: x_1st_Flr_SF + 4384 t147_leaf0 <= 5000
 T147_L0_f6_le: const_Year_Remod_Add + 998044 t147_leaf0 <= 1e+06
 T147_L1_f17_le: x_1st_Flr_SF + 4384 t147_leaf1 <= 5000
 T147_R1_f6_ge: const_Year_Remod_Add - 1.001956e+06 t147_leaf1 >= -1e+06
 T147_R2_f17_ge: x_1st_Flr_SF - 616 t147_leaf2 >= 0
 T147_L2_f118_le: const_Condition_1_Feedr + 999999 t147_leaf2 <= 1e+06
 T147_L2_f12_le: x_BsmtFin_SF_1 + 999300 t147_leaf2 <= 1e+06
 T147_R3_f17_ge: x_1st_Flr_SF - 616 t147_leaf3 >= 0
 T147_L3_f118_le: const_Condition_1_Feedr + 999999 t147_leaf3 <= 1e+06
 T147_R3_f12_ge: x_BsmtFin_SF_1 - 700 t147_leaf3 >= 0
 T147_R4_f17_ge: x_1st_Flr_SF - 616 t147_leaf4 >= 0
 T147_R4_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t147_leaf4
   >= -1e+06
 T147_L4_f271_le: const_Mo_Sold_1 + 999999 t147_leaf4 <= 1e+06
 T147_R5_f17_ge: x_1st_Flr_SF - 616 t147_leaf5 >= 0
 T147_R5_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t147_leaf5
   >= -1e+06
 T147_R5_f271_ge: const_Mo_Sold_1 - 1.000001e+06 t147_leaf5 >= -1e+06
 TREE_148_ONEHOT: t148_leaf0 + t148_leaf1 + t148_leaf2 + t148_leaf3
   + t148_leaf4 + t148_leaf5 = 1
 T148_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t148_leaf0
   <= 1e+06
 T148_L0_f164_le: x_ex1_is_BrkFace <= 1
 T148_L0_f15_le: const_Total_Bsmt_SF + 999415 t148_leaf0 <= 1e+06
 T148_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t148_leaf1
   <= 1e+06
 T148_L1_f164_le: x_ex1_is_BrkFace <= 1
 T148_R1_f15_ge: const_Total_Bsmt_SF - 1.000585e+06 t148_leaf1 >= -1e+06
 T148_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t148_leaf2
   <= 1e+06
 T148_R2_f164_ge: x_ex1_is_BrkFace - t148_leaf2 >= 0
 T148_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t148_leaf3
   >= -1e+06
 T148_L3_f3_le: Overall_Qual_calc + 999996 t148_leaf3 <= 1e+06
 T148_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t148_leaf4
   >= -1e+06
 T148_R4_f3_ge: Overall_Qual_calc - 1.000004e+06 t148_leaf4 >= -1e+06
 T148_L4_f29_le: const_Fireplaces + 999999 t148_leaf4 <= 1e+06
 T148_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t148_leaf5
   >= -1e+06
 T148_R5_f3_ge: Overall_Qual_calc - 1.000004e+06 t148_leaf5 >= -1e+06
 T148_R5_f29_ge: const_Fireplaces - 1.000001e+06 t148_leaf5 >= -1e+06
 TREE_149_ONEHOT: t149_leaf0 + t149_leaf1 + t149_leaf2 + t149_leaf3
   + t149_leaf4 + t149_leaf5 + t149_leaf6 = 1
 T149_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t149_leaf0 <= 1e+06
 T149_L0_f192_le: x_ex2_is_Wd_Sdng <= 1
 T149_L0_f12_le: x_BsmtFin_SF_1 + 999812 t149_leaf0 <= 1e+06
 T149_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t149_leaf1 <= 1e+06
 T149_L1_f192_le: x_ex2_is_Wd_Sdng <= 1
 T149_R1_f12_ge: x_BsmtFin_SF_1 - 188 t149_leaf1 >= 0
 T149_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t149_leaf2 <= 1e+06
 T149_R2_f192_ge: x_ex2_is_Wd_Sdng - t149_leaf2 >= 0
 T149_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t149_leaf3
   >= -1e+06
 T149_L3_f164_le: x_ex1_is_BrkFace <= 1
 T149_L3_f4_le: const_Overall_Cond + 999994 t149_leaf3 <= 1e+06
 T149_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t149_leaf4
   >= -1e+06
 T149_L4_f164_le: x_ex1_is_BrkFace <= 1
 T149_R4_f4_ge: const_Overall_Cond - 1.000006e+06 t149_leaf4 >= -1e+06
 T149_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t149_leaf5
   >= -1e+06
 T149_R5_f164_ge: x_ex1_is_BrkFace - t149_leaf5 >= 0
 T149_L5_f106_le: const_Neighborhood_No_aplicames + 999999 t149_leaf5
   <= 1e+06
 T149_R6_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t149_leaf6
   >= -1e+06
 T149_R6_f164_ge: x_ex1_is_BrkFace - t149_leaf6 >= 0
 T149_R6_f106_ge: const_Neighborhood_No_aplicames - 1.000001e+06 t149_leaf6
   >= -1e+06
 TREE_150_ONEHOT: t150_leaf0 + t150_leaf1 + t150_leaf2 + t150_leaf3
   + t150_leaf4 + t150_leaf5 + t150_leaf6 = 1
 T150_L0_f9_le: x_Exter_Cond + 2 t150_leaf0 <= 4
 T150_L0_f33_le: x_Garage_Area + 99703 t150_leaf0 <= 100000
 T150_L1_f9_le: x_Exter_Cond + 2 t150_leaf1 <= 4
 T150_R1_f33_ge: x_Garage_Area - 297 t150_leaf1 >= 0
 T150_L1_f15_le: const_Total_Bsmt_SF + 999183 t150_leaf1 <= 1e+06
 T150_L2_f9_le: x_Exter_Cond + 2 t150_leaf2 <= 4
 T150_R2_f33_ge: x_Garage_Area - 297 t150_leaf2 >= 0
 T150_R2_f15_ge: const_Total_Bsmt_SF - 1.000817e+06 t150_leaf2 >= -1e+06
 T150_R3_f9_ge: x_Exter_Cond - 3 t150_leaf3 >= -1
 T150_L3_f1_le: const_Lot_Area + 994024 t150_leaf3 <= 1e+06
 T150_L3_f20_le: x_Gr_Liv_Area + 9004 t150_leaf3 <= 10000
 T150_R4_f9_ge: x_Exter_Cond - 3 t150_leaf4 >= -1
 T150_L4_f1_le: const_Lot_Area + 994024 t150_leaf4 <= 1e+06
 T150_R4_f20_ge: x_Gr_Liv_Area - 996 t150_leaf4 >= 0
 T150_R5_f9_ge: x_Exter_Cond - 3 t150_leaf5 >= -1
 T150_R5_f1_ge: const_Lot_Area - 1.005976e+06 t150_leaf5 >= -1e+06
 T150_L5_f95_le: const_Neighborhood_Crawfor + 999999 t150_leaf5 <= 1e+06
 T150_R6_f9_ge: x_Exter_Cond - 3 t150_leaf6 >= -1
 T150_R6_f1_ge: const_Lot_Area - 1.005976e+06 t150_leaf6 >= -1e+06
 T150_R6_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t150_leaf6
   >= -1e+06
 TREE_151_ONEHOT: t151_leaf0 + t151_leaf1 + t151_leaf2 + t151_leaf3
   + t151_leaf4 = 1
 T151_L0_f33_le: x_Garage_Area + 99196 t151_leaf0 <= 100000
 T151_L0_f207_le: const_Bsmt_Exposure_Gd + 999999 t151_leaf0 <= 1e+06
 T151_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t151_leaf0 <= 1e+06
 T151_L1_f33_le: x_Garage_Area + 99196 t151_leaf1 <= 100000
 T151_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t151_leaf1 <= 1e+06
 T151_R1_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t151_leaf1
   >= -1e+06
 T151_L2_f33_le: x_Garage_Area + 99196 t151_leaf2 <= 100000
 T151_R2_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t151_leaf2
   >= -1e+06
 T151_L2_f1_le: const_Lot_Area + 983440 t151_leaf2 <= 1e+06
 T151_L3_f33_le: x_Garage_Area + 99196 t151_leaf3 <= 100000
 T151_R3_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t151_leaf3
   >= -1e+06
 T151_R3_f1_ge: const_Lot_Area - 1.01656e+06 t151_leaf3 >= -1e+06
 T151_R4_f33_ge: x_Garage_Area - 804 t151_leaf4 >= 0
 TREE_152_ONEHOT: t152_leaf0 + t152_leaf1 + t152_leaf2 + t152_leaf3
   + t152_leaf4 + t152_leaf5 = 1
 T152_L0_f40_le: x_Screen_Porch + 99850 t152_leaf0 <= 100000
 T152_L0_f9_le: x_Exter_Cond + 2 t152_leaf0 <= 4
 T152_L0_f1_le: const_Lot_Area + 990829 t152_leaf0 <= 1e+06
 T152_L1_f40_le: x_Screen_Porch + 99850 t152_leaf1 <= 100000
 T152_L1_f9_le: x_Exter_Cond + 2 t152_leaf1 <= 4
 T152_R1_f1_ge: const_Lot_Area - 1.009171e+06 t152_leaf1 >= -1e+06
 T152_L2_f40_le: x_Screen_Porch + 99850 t152_leaf2 <= 100000
 T152_R2_f9_ge: x_Exter_Cond - 3 t152_leaf2 >= -1
 T152_L2_f86_le: const_Land_Slope_Gtl + 999999 t152_leaf2 <= 1e+06
 T152_L3_f40_le: x_Screen_Porch + 99850 t152_leaf3 <= 100000
 T152_R3_f9_ge: x_Exter_Cond - 3 t152_leaf3 >= -1
 T152_R3_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t152_leaf3 >= -1e+06
 T152_R4_f40_ge: x_Screen_Porch - 150 t152_leaf4 >= 0
 T152_L4_f18_le: const_2nd_Flr_SF + 999022 t152_leaf4 <= 1e+06
 T152_R5_f40_ge: x_Screen_Porch - 150 t152_leaf5 >= 0
 T152_R5_f18_ge: const_2nd_Flr_SF - 1.000978e+06 t152_leaf5 >= -1e+06
 TREE_153_ONEHOT: t153_leaf0 + t153_leaf1 + t153_leaf2 + t153_leaf3
   + t153_leaf4 + t153_leaf5 + t153_leaf6 = 1
 T153_L0_f1_le: const_Lot_Area + 994000 t153_leaf0 <= 1e+06
 T153_L0_f31_le: const_Garage_Yr_Blt + 998105 t153_leaf0 <= 1e+06
 T153_L1_f1_le: const_Lot_Area + 994000 t153_leaf1 <= 1e+06
 T153_R1_f31_ge: const_Garage_Yr_Blt - 1.001895e+06 t153_leaf1 >= -1e+06
 T153_L1_f297_le: const_Sale_Condition_Normal + 999999 t153_leaf1 <= 1e+06
 T153_L2_f1_le: const_Lot_Area + 994000 t153_leaf2 <= 1e+06
 T153_R2_f31_ge: const_Garage_Yr_Blt - 1.001895e+06 t153_leaf2 >= -1e+06
 T153_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t153_leaf2
   >= -1e+06
 T153_R3_f1_ge: const_Lot_Area - 1.006e+06 t153_leaf3 >= -1e+06
 T153_L3_f40_le: x_Screen_Porch + 99858 t153_leaf3 <= 100000
 T153_L3_f5_le: const_Year_Built + 998087 t153_leaf3 <= 1e+06
 T153_R4_f1_ge: const_Lot_Area - 1.006e+06 t153_leaf4 >= -1e+06
 T153_L4_f40_le: x_Screen_Porch + 99858 t153_leaf4 <= 100000
 T153_R4_f5_ge: const_Year_Built - 1.001913e+06 t153_leaf4 >= -1e+06
 T153_R5_f1_ge: const_Lot_Area - 1.006e+06 t153_leaf5 >= -1e+06
 T153_R5_f40_ge: x_Screen_Porch - 142 t153_leaf5 >= 0
 T153_L5_f20_le: x_Gr_Liv_Area + 8070 t153_leaf5 <= 10000
 T153_R6_f1_ge: const_Lot_Area - 1.006e+06 t153_leaf6 >= -1e+06
 T153_R6_f40_ge: x_Screen_Porch - 142 t153_leaf6 >= 0
 T153_R6_f20_ge: x_Gr_Liv_Area - 1930 t153_leaf6 >= 0
 TREE_154_ONEHOT: t154_leaf0 + t154_leaf1 + t154_leaf2 + t154_leaf3
   + t154_leaf4 + t154_leaf5 + t154_leaf6 + t154_leaf7 = 1
 T154_L0_f16_le: x_Heating_QC + t154_leaf0 <= 4
 T154_L0_f297_le: const_Sale_Condition_Normal + 999999 t154_leaf0 <= 1e+06
 T154_L0_f20_le: x_Gr_Liv_Area + 9024 t154_leaf0 <= 10000
 T154_L1_f16_le: x_Heating_QC + t154_leaf1 <= 4
 T154_L1_f297_le: const_Sale_Condition_Normal + 999999 t154_leaf1 <= 1e+06
 T154_R1_f20_ge: x_Gr_Liv_Area - 976 t154_leaf1 >= 0
 T154_L2_f16_le: x_Heating_QC + t154_leaf2 <= 4
 T154_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t154_leaf2
   >= -1e+06
 T154_L2_f1_le: const_Lot_Area + 989104 t154_leaf2 <= 1e+06
 T154_L3_f16_le: x_Heating_QC + t154_leaf3 <= 4
 T154_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t154_leaf3
   >= -1e+06
 T154_R3_f1_ge: const_Lot_Area - 1.010896e+06 t154_leaf3 >= -1e+06
 T154_R4_f16_ge: x_Heating_QC - 4 t154_leaf4 >= -1
 T154_L4_f245_le: const_FunctioNo_aplical_Typ + 999999 t154_leaf4 <= 1e+06
 T154_L4_f139_le: const_House_Style_1.5Fin + 999999 t154_leaf4 <= 1e+06
 T154_R5_f16_ge: x_Heating_QC - 4 t154_leaf5 >= -1
 T154_L5_f245_le: const_FunctioNo_aplical_Typ + 999999 t154_leaf5 <= 1e+06
 T154_R5_f139_ge: const_House_Style_1.5Fin - 1.000001e+06 t154_leaf5
   >= -1e+06
 T154_R6_f16_ge: x_Heating_QC - 4 t154_leaf6 >= -1
 T154_R6_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t154_leaf6
   >= -1e+06
 T154_L6_f8_le: x_Exter_Qual + t154_leaf6 <= 4
 T154_R7_f16_ge: x_Heating_QC - 4 t154_leaf7 >= -1
 T154_R7_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t154_leaf7
   >= -1e+06
 T154_R7_f8_ge: x_Exter_Qual - 4 t154_leaf7 >= -1
 TREE_155_ONEHOT: t155_leaf0 + t155_leaf1 + t155_leaf2 + t155_leaf3
   + t155_leaf4 = 1
 T155_L0_f33_le: x_Garage_Area + 98846 t155_leaf0 <= 100000
 T155_L0_f1_le: const_Lot_Area + 982129 t155_leaf0 <= 1e+06
 T155_L0_f31_le: const_Garage_Yr_Blt + 997992 t155_leaf0 <= 1e+06
 T155_L1_f33_le: x_Garage_Area + 98846 t155_leaf1 <= 100000
 T155_L1_f1_le: const_Lot_Area + 982129 t155_leaf1 <= 1e+06
 T155_R1_f31_ge: const_Garage_Yr_Blt - 1.002008e+06 t155_leaf1 >= -1e+06
 T155_L2_f33_le: x_Garage_Area + 98846 t155_leaf2 <= 100000
 T155_R2_f1_ge: const_Lot_Area - 1.017871e+06 t155_leaf2 >= -1e+06
 T155_L2_f103_le: const_Neighborhood_Mitchel + 999999 t155_leaf2 <= 1e+06
 T155_L3_f33_le: x_Garage_Area + 98846 t155_leaf3 <= 100000
 T155_R3_f1_ge: const_Lot_Area - 1.017871e+06 t155_leaf3 >= -1e+06
 T155_R3_f103_ge: const_Neighborhood_Mitchel - 1.000001e+06 t155_leaf3
   >= -1e+06
 T155_R4_f33_ge: x_Garage_Area - 1154 t155_leaf4 >= 0
 TREE_156_ONEHOT: t156_leaf0 + t156_leaf1 + t156_leaf2 + t156_leaf3
   + t156_leaf4 = 1
 T156_L0_f40_le: x_Screen_Porch + 99960 t156_leaf0 <= 100000
 T156_L0_f5_le: const_Year_Built + 997993 t156_leaf0 <= 1e+06
 T156_L0_f12_le: x_BsmtFin_SF_1 + 999944 t156_leaf0 <= 1e+06
 T156_L1_f40_le: x_Screen_Porch + 99960 t156_leaf1 <= 100000
 T156_L1_f5_le: const_Year_Built + 997993 t156_leaf1 <= 1e+06
 T156_R1_f12_ge: x_BsmtFin_SF_1 - 56 t156_leaf1 >= 0
 T156_L2_f40_le: x_Screen_Porch + 99960 t156_leaf2 <= 100000
 T156_R2_f5_ge: const_Year_Built - 1.002007e+06 t156_leaf2 >= -1e+06
 T156_R3_f40_ge: x_Screen_Porch - 40 t156_leaf3 >= 0
 T156_L3_f20_le: x_Gr_Liv_Area + 8023 t156_leaf3 <= 10000
 T156_R4_f40_ge: x_Screen_Porch - 40 t156_leaf4 >= 0
 T156_R4_f20_ge: x_Gr_Liv_Area - 1977 t156_leaf4 >= 0
 TREE_157_ONEHOT: t157_leaf0 + t157_leaf1 + t157_leaf2 + t157_leaf3
   + t157_leaf4 + t157_leaf5 + t157_leaf6 = 1
 T157_L0_f1_le: const_Lot_Area + 992050 t157_leaf0 <= 1e+06
 T157_L0_f20_le: x_Gr_Liv_Area + 8872 t157_leaf0 <= 10000
 T157_L0_f12_le: x_BsmtFin_SF_1 + 999467 t157_leaf0 <= 1e+06
 T157_L1_f1_le: const_Lot_Area + 992050 t157_leaf1 <= 1e+06
 T157_L1_f20_le: x_Gr_Liv_Area + 8872 t157_leaf1 <= 10000
 T157_R1_f12_ge: x_BsmtFin_SF_1 - 533 t157_leaf1 >= 0
 T157_L2_f1_le: const_Lot_Area + 992050 t157_leaf2 <= 1e+06
 T157_R2_f20_ge: x_Gr_Liv_Area - 1128 t157_leaf2 >= 0
 T157_L2_f238_le: const_FunctioNo_aplical_Maj1 + 999999 t157_leaf2
   <= 1e+06
 T157_L3_f1_le: const_Lot_Area + 992050 t157_leaf3 <= 1e+06
 T157_R3_f20_ge: x_Gr_Liv_Area - 1128 t157_leaf3 >= 0
 T157_R3_f238_ge: const_FunctioNo_aplical_Maj1 - 1.000001e+06 t157_leaf3
   >= -1e+06
 T157_R4_f1_ge: const_Lot_Area - 1.00795e+06 t157_leaf4 >= -1e+06
 T157_L4_f18_le: const_2nd_Flr_SF + 998573 t157_leaf4 <= 1e+06
 T157_L4_f119_le: const_Condition_1_Norm + 999999 t157_leaf4 <= 1e+06
 T157_R5_f1_ge: const_Lot_Area - 1.00795e+06 t157_leaf5 >= -1e+06
 T157_L5_f18_le: const_2nd_Flr_SF + 998573 t157_leaf5 <= 1e+06
 T157_R5_f119_ge: const_Condition_1_Norm - 1.000001e+06 t157_leaf5
   >= -1e+06
 T157_R6_f1_ge: const_Lot_Area - 1.00795e+06 t157_leaf6 >= -1e+06
 T157_R6_f18_ge: const_2nd_Flr_SF - 1.001427e+06 t157_leaf6 >= -1e+06
 TREE_158_ONEHOT: t158_leaf0 + t158_leaf1 + t158_leaf2 + t158_leaf3
   + t158_leaf4 + t158_leaf5 + t158_leaf6 = 1
 T158_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t158_leaf0 <= 1e+06
 T158_L0_f40_le: x_Screen_Porch + 99879 t158_leaf0 <= 100000
 T158_L0_f1_le: const_Lot_Area + 986482 t158_leaf0 <= 1e+06
 T158_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t158_leaf1 <= 1e+06
 T158_L1_f40_le: x_Screen_Porch + 99879 t158_leaf1 <= 100000
 T158_R1_f1_ge: const_Lot_Area - 1.013518e+06 t158_leaf1 >= -1e+06
 T158_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t158_leaf2 <= 1e+06
 T158_R2_f40_ge: x_Screen_Porch - 121 t158_leaf2 >= 0
 T158_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t158_leaf3
   >= -1e+06
 T158_L3_f44_le: const_Yr_Sold + 997993 t158_leaf3 <= 1e+06
 T158_L3_f1_le: const_Lot_Area + 991873 t158_leaf3 <= 1e+06
 T158_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t158_leaf4
   >= -1e+06
 T158_L4_f44_le: const_Yr_Sold + 997993 t158_leaf4 <= 1e+06
 T158_R4_f1_ge: const_Lot_Area - 1.008127e+06 t158_leaf4 >= -1e+06
 T158_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t158_leaf5
   >= -1e+06
 T158_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t158_leaf5 >= -1e+06
 T158_L5_f20_le: x_Gr_Liv_Area + 8845 t158_leaf5 <= 10000
 T158_R6_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t158_leaf6
   >= -1e+06
 T158_R6_f44_ge: const_Yr_Sold - 1.002007e+06 t158_leaf6 >= -1e+06
 T158_R6_f20_ge: x_Gr_Liv_Area - 1155 t158_leaf6 >= 0
 TREE_159_ONEHOT: t159_leaf0 + t159_leaf1 + t159_leaf2 + t159_leaf3
   + t159_leaf4 = 1
 T159_L0_f3_le: Overall_Qual_calc + 999991 t159_leaf0 <= 1e+06
 T159_L0_f20_le: x_Gr_Liv_Area + 9253 t159_leaf0 <= 10000
 T159_L0_f16_le: x_Heating_QC + t159_leaf0 <= 4
 T159_L1_f3_le: Overall_Qual_calc + 999991 t159_leaf1 <= 1e+06
 T159_L1_f20_le: x_Gr_Liv_Area + 9253 t159_leaf1 <= 10000
 T159_R1_f16_ge: x_Heating_QC - 4 t159_leaf1 >= -1
 T159_L2_f3_le: Overall_Qual_calc + 999991 t159_leaf2 <= 1e+06
 T159_R2_f20_ge: x_Gr_Liv_Area - 747 t159_leaf2 >= 0
 T159_L2_f4_le: const_Overall_Cond + 999994 t159_leaf2 <= 1e+06
 T159_L3_f3_le: Overall_Qual_calc + 999991 t159_leaf3 <= 1e+06
 T159_R3_f20_ge: x_Gr_Liv_Area - 747 t159_leaf3 >= 0
 T159_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t159_leaf3 >= -1e+06
 T159_R4_f3_ge: Overall_Qual_calc - 1.000009e+06 t159_leaf4 >= -1e+06
 TREE_160_ONEHOT: t160_leaf0 + t160_leaf1 + t160_leaf2 + t160_leaf3
   + t160_leaf4 + t160_leaf5 + t160_leaf6 = 1
 T160_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t160_leaf0
   <= 1e+06
 T160_L0_f44_le: const_Yr_Sold + 997993 t160_leaf0 <= 1e+06
 T160_L0_f1_le: const_Lot_Area + 983221 t160_leaf0 <= 1e+06
 T160_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t160_leaf1
   <= 1e+06
 T160_L1_f44_le: const_Yr_Sold + 997993 t160_leaf1 <= 1e+06
 T160_R1_f1_ge: const_Lot_Area - 1.016779e+06 t160_leaf1 >= -1e+06
 T160_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t160_leaf2
   <= 1e+06
 T160_R2_f44_ge: const_Yr_Sold - 1.002007e+06 t160_leaf2 >= -1e+06
 T160_L2_f40_le: x_Screen_Porch + 99858 t160_leaf2 <= 100000
 T160_L3_f293_le: const_Sale_Condition_Abnorml + 999999 t160_leaf3
   <= 1e+06
 T160_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t160_leaf3 >= -1e+06
 T160_R3_f40_ge: x_Screen_Porch - 142 t160_leaf3 >= 0
 T160_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t160_leaf4
   >= -1e+06
 T160_L4_f3_le: Overall_Qual_calc + 999996 t160_leaf4 <= 1e+06
 T160_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t160_leaf5
   >= -1e+06
 T160_R5_f3_ge: Overall_Qual_calc - 1.000004e+06 t160_leaf5 >= -1e+06
 T160_L5_f16_le: x_Heating_QC + t160_leaf5 <= 4
 T160_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t160_leaf6
   >= -1e+06
 T160_R6_f3_ge: Overall_Qual_calc - 1.000004e+06 t160_leaf6 >= -1e+06
 T160_R6_f16_ge: x_Heating_QC - 4 t160_leaf6 >= -1
 TREE_161_ONEHOT: t161_leaf0 + t161_leaf1 + t161_leaf2 + t161_leaf3
   + t161_leaf4 + t161_leaf5 + t161_leaf6 = 1
 T161_L0_f77_le: const_Land_Contour_Bnk + 999999 t161_leaf0 <= 1e+06
 T161_L0_f86_le: const_Land_Slope_Gtl + 999999 t161_leaf0 <= 1e+06
 T161_L0_f1_le: const_Lot_Area + 989956 t161_leaf0 <= 1e+06
 T161_L1_f77_le: const_Land_Contour_Bnk + 999999 t161_leaf1 <= 1e+06
 T161_L1_f86_le: const_Land_Slope_Gtl + 999999 t161_leaf1 <= 1e+06
 T161_R1_f1_ge: const_Lot_Area - 1.010044e+06 t161_leaf1 >= -1e+06
 T161_L2_f77_le: const_Land_Contour_Bnk + 999999 t161_leaf2 <= 1e+06
 T161_R2_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t161_leaf2 >= -1e+06
 T161_L2_f17_le: x_1st_Flr_SF + 2879 t161_leaf2 <= 5000
 T161_L3_f77_le: const_Land_Contour_Bnk + 999999 t161_leaf3 <= 1e+06
 T161_R3_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t161_leaf3 >= -1e+06
 T161_R3_f17_ge: x_1st_Flr_SF - 2121 t161_leaf3 >= 0
 T161_R4_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t161_leaf4
   >= -1e+06
 T161_L4_f0_le: const_Lot_Frontage + 999889 t161_leaf4 <= 1e+06
 T161_L4_f0_le: const_Lot_Frontage + 999931 t161_leaf4 <= 1e+06
 T161_R5_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t161_leaf5
   >= -1e+06
 T161_L5_f0_le: const_Lot_Frontage + 999889 t161_leaf5 <= 1e+06
 T161_R5_f0_ge: const_Lot_Frontage - 1.000069e+06 t161_leaf5 >= -1e+06
 T161_R6_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t161_leaf6
   >= -1e+06
 T161_R6_f0_ge: const_Lot_Frontage - 1.000111e+06 t161_leaf6 >= -1e+06
 TREE_162_ONEHOT: t162_leaf0 + t162_leaf1 + t162_leaf2 + t162_leaf3
   + t162_leaf4 + t162_leaf5 + t162_leaf6 = 1
 T162_L0_f9_le: x_Exter_Cond + 2 t162_leaf0 <= 4
 T162_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t162_leaf0
   <= 1e+06
 T162_L0_f33_le: x_Garage_Area + 99655 t162_leaf0 <= 100000
 T162_L1_f9_le: x_Exter_Cond + 2 t162_leaf1 <= 4
 T162_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t162_leaf1
   <= 1e+06
 T162_R1_f33_ge: x_Garage_Area - 345 t162_leaf1 >= 0
 T162_L2_f9_le: x_Exter_Cond + 2 t162_leaf2 <= 4
 T162_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t162_leaf2
   >= -1e+06
 T162_R3_f9_ge: x_Exter_Cond - 3 t162_leaf3 >= -1
 T162_L3_f164_le: x_ex1_is_BrkFace <= 1
 T162_L3_f40_le: x_Screen_Porch + 99850 t162_leaf3 <= 100000
 T162_R4_f9_ge: x_Exter_Cond - 3 t162_leaf4 >= -1
 T162_L4_f164_le: x_ex1_is_BrkFace <= 1
 T162_R4_f40_ge: x_Screen_Porch - 150 t162_leaf4 >= 0
 T162_R5_f9_ge: x_Exter_Cond - 3 t162_leaf5 >= -1
 T162_R5_f164_ge: x_ex1_is_BrkFace - t162_leaf5 >= 0
 T162_L5_f20_le: x_Gr_Liv_Area + 8758 t162_leaf5 <= 10000
 T162_R6_f9_ge: x_Exter_Cond - 3 t162_leaf6 >= -1
 T162_R6_f164_ge: x_ex1_is_BrkFace - t162_leaf6 >= 0
 T162_R6_f20_ge: x_Gr_Liv_Area - 1242 t162_leaf6 >= 0
 TREE_163_ONEHOT: t163_leaf0 + t163_leaf1 + t163_leaf2 + t163_leaf3
   + t163_leaf4 + t163_leaf5 = 1
 T163_L0_f164_le: x_ex1_is_BrkFace <= 1
 T163_L0_f36_le: x_Wood_Deck_SF + 99801 t163_leaf0 <= 100000
 T163_L0_f31_le: const_Garage_Yr_Blt + 997992 t163_leaf0 <= 1e+06
 T163_L1_f164_le: x_ex1_is_BrkFace <= 1
 T163_L1_f36_le: x_Wood_Deck_SF + 99801 t163_leaf1 <= 100000
 T163_R1_f31_ge: const_Garage_Yr_Blt - 1.002008e+06 t163_leaf1 >= -1e+06
 T163_L2_f164_le: x_ex1_is_BrkFace <= 1
 T163_R2_f36_ge: x_Wood_Deck_SF - 199 t163_leaf2 >= 0
 T163_L2_f7_le: x_Mas_Vnr_Area + 1246 t163_leaf2 <= 2000
 T163_L3_f164_le: x_ex1_is_BrkFace <= 1
 T163_R3_f36_ge: x_Wood_Deck_SF - 199 t163_leaf3 >= 0
 T163_R3_f7_ge: x_Mas_Vnr_Area - 754 t163_leaf3 >= 0
 T163_R4_f164_ge: x_ex1_is_BrkFace - t163_leaf4 >= 0
 T163_L4_f1_le: const_Lot_Area + 990395 t163_leaf4 <= 1e+06
 T163_R5_f164_ge: x_ex1_is_BrkFace - t163_leaf5 >= 0
 T163_R5_f1_ge: const_Lot_Area - 1.009605e+06 t163_leaf5 >= -1e+06
 TREE_164_ONEHOT: t164_leaf0 + t164_leaf1 + t164_leaf2 + t164_leaf3
   + t164_leaf4 = 1
 T164_L0_f18_le: const_2nd_Flr_SF + 998573 t164_leaf0 <= 1e+06
 T164_L0_f11_le: x_Bsmt_Cond + 4 t164_leaf0 <= 4
 T164_L0_f33_le: x_Garage_Area + 99543 t164_leaf0 <= 100000
 T164_L1_f18_le: const_2nd_Flr_SF + 998573 t164_leaf1 <= 1e+06
 T164_L1_f11_le: x_Bsmt_Cond + 4 t164_leaf1 <= 4
 T164_R1_f33_ge: x_Garage_Area - 457 t164_leaf1 >= 0
 T164_L2_f18_le: const_2nd_Flr_SF + 998573 t164_leaf2 <= 1e+06
 T164_R2_f11_ge: x_Bsmt_Cond - t164_leaf2 >= -1
 T164_L2_f289_le: const_Sale_Type_New + 999999 t164_leaf2 <= 1e+06
 T164_L3_f18_le: const_2nd_Flr_SF + 998573 t164_leaf3 <= 1e+06
 T164_R3_f11_ge: x_Bsmt_Cond - t164_leaf3 >= -1
 T164_R3_f289_ge: const_Sale_Type_New - 1.000001e+06 t164_leaf3 >= -1e+06
 T164_R4_f18_ge: const_2nd_Flr_SF - 1.001427e+06 t164_leaf4 >= -1e+06
 TREE_165_ONEHOT: t165_leaf0 + t165_leaf1 + t165_leaf2 + t165_leaf3
   + t165_leaf4 = 1
 T165_L0_f17_le: x_1st_Flr_SF + 2485 t165_leaf0 <= 5000
 T165_L0_f15_le: const_Total_Bsmt_SF + 998176 t165_leaf0 <= 1e+06
 T165_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t165_leaf0
   <= 1e+06
 T165_L1_f17_le: x_1st_Flr_SF + 2485 t165_leaf1 <= 5000
 T165_L1_f15_le: const_Total_Bsmt_SF + 998176 t165_leaf1 <= 1e+06
 T165_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t165_leaf1
   >= -1e+06
 T165_L2_f17_le: x_1st_Flr_SF + 2485 t165_leaf2 <= 5000
 T165_R2_f15_ge: const_Total_Bsmt_SF - 1.001824e+06 t165_leaf2 >= -1e+06
 T165_L2_f108_le: const_Neighborhood_NridgHt + 999999 t165_leaf2 <= 1e+06
 T165_L3_f17_le: x_1st_Flr_SF + 2485 t165_leaf3 <= 5000
 T165_R3_f15_ge: const_Total_Bsmt_SF - 1.001824e+06 t165_leaf3 >= -1e+06
 T165_R3_f108_ge: const_Neighborhood_NridgHt - 1.000001e+06 t165_leaf3
   >= -1e+06
 T165_R4_f17_ge: x_1st_Flr_SF - 2515 t165_leaf4 >= 0
 TREE_166_ONEHOT: t166_leaf0 + t166_leaf1 + t166_leaf2 + t166_leaf3 = 1
 T166_L0_f18_le: const_2nd_Flr_SF + 998371 t166_leaf0 <= 1e+06
 T166_L0_f93_le: const_Neighborhood_ClearCr + 999999 t166_leaf0 <= 1e+06
 T166_L0_f298_le: const_Sale_Condition_Partial + 999999 t166_leaf0
   <= 1e+06
 T166_L1_f18_le: const_2nd_Flr_SF + 998371 t166_leaf1 <= 1e+06
 T166_L1_f93_le: const_Neighborhood_ClearCr + 999999 t166_leaf1 <= 1e+06
 T166_R1_f298_ge: const_Sale_Condition_Partial - 1.000001e+06 t166_leaf1
   >= -1e+06
 T166_L2_f18_le: const_2nd_Flr_SF + 998371 t166_leaf2 <= 1e+06
 T166_R2_f93_ge: const_Neighborhood_ClearCr - 1.000001e+06 t166_leaf2
   >= -1e+06
 T166_R3_f18_ge: const_2nd_Flr_SF - 1.001629e+06 t166_leaf3 >= -1e+06
 TREE_167_ONEHOT: t167_leaf0 + t167_leaf1 + t167_leaf2 + t167_leaf3
   + t167_leaf4 = 1
 T167_L0_f3_le: Overall_Qual_calc + 999997 t167_leaf0 <= 1e+06
 T167_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t167_leaf1 >= -1e+06
 T167_L1_f4_le: const_Overall_Cond + 999993 t167_leaf1 <= 1e+06
 T167_L1_f5_le: const_Year_Built + 998089 t167_leaf1 <= 1e+06
 T167_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t167_leaf2 >= -1e+06
 T167_L2_f4_le: const_Overall_Cond + 999993 t167_leaf2 <= 1e+06
 T167_R2_f5_ge: const_Year_Built - 1.001911e+06 t167_leaf2 >= -1e+06
 T167_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t167_leaf3 >= -1e+06
 T167_R3_f4_ge: const_Overall_Cond - 1.000007e+06 t167_leaf3 >= -1e+06
 T167_L3_f44_le: const_Yr_Sold + 997991 t167_leaf3 <= 1e+06
 T167_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t167_leaf4 >= -1e+06
 T167_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t167_leaf4 >= -1e+06
 T167_R4_f44_ge: const_Yr_Sold - 1.002009e+06 t167_leaf4 >= -1e+06
 TREE_168_ONEHOT: t168_leaf0 + t168_leaf1 + t168_leaf2 + t168_leaf3
   + t168_leaf4 = 1
 T168_L0_f117_le: const_Condition_1_Artery + 999999 t168_leaf0 <= 1e+06
 T168_L0_f95_le: const_Neighborhood_Crawfor + 999999 t168_leaf0 <= 1e+06
 T168_L0_f4_le: const_Overall_Cond + 999993 t168_leaf0 <= 1e+06
 T168_L1_f117_le: const_Condition_1_Artery + 999999 t168_leaf1 <= 1e+06
 T168_L1_f95_le: const_Neighborhood_Crawfor + 999999 t168_leaf1 <= 1e+06
 T168_R1_f4_ge: const_Overall_Cond - 1.000007e+06 t168_leaf1 >= -1e+06
 T168_L2_f117_le: const_Condition_1_Artery + 999999 t168_leaf2 <= 1e+06
 T168_R2_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t168_leaf2
   >= -1e+06
 T168_R3_f117_ge: const_Condition_1_Artery - 1.000001e+06 t168_leaf3
   >= -1e+06
 T168_L3_f33_le: x_Garage_Area + 99375 t168_leaf3 <= 100000
 T168_R4_f117_ge: const_Condition_1_Artery - 1.000001e+06 t168_leaf4
   >= -1e+06
 T168_R4_f33_ge: x_Garage_Area - 625 t168_leaf4 >= 0
 TREE_169_ONEHOT: t169_leaf0 + t169_leaf1 + t169_leaf2 + t169_leaf3
   + t169_leaf4 + t169_leaf5 + t169_leaf6 = 1
 T169_L0_f4_le: const_Overall_Cond + 999993 t169_leaf0 <= 1e+06
 T169_L0_f5_le: const_Year_Built + 998068 t169_leaf0 <= 1e+06
 T169_L0_f36_le: x_Wood_Deck_SF + 99832 t169_leaf0 <= 100000
 T169_L1_f4_le: const_Overall_Cond + 999993 t169_leaf1 <= 1e+06
 T169_L1_f5_le: const_Year_Built + 998068 t169_leaf1 <= 1e+06
 T169_R1_f36_ge: x_Wood_Deck_SF - 168 t169_leaf1 >= 0
 T169_L2_f4_le: const_Overall_Cond + 999993 t169_leaf2 <= 1e+06
 T169_R2_f5_ge: const_Year_Built - 1.001932e+06 t169_leaf2 >= -1e+06
 T169_L2_f6_le: const_Year_Remod_Add + 997993 t169_leaf2 <= 1e+06
 T169_L3_f4_le: const_Overall_Cond + 999993 t169_leaf3 <= 1e+06
 T169_R3_f5_ge: const_Year_Built - 1.001932e+06 t169_leaf3 >= -1e+06
 T169_R3_f6_ge: const_Year_Remod_Add - 1.002007e+06 t169_leaf3 >= -1e+06
 T169_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t169_leaf4 >= -1e+06
 T169_L4_f44_le: const_Yr_Sold + 997992 t169_leaf4 <= 1e+06
 T169_R5_f4_ge: const_Overall_Cond - 1.000007e+06 t169_leaf5 >= -1e+06
 T169_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t169_leaf5 >= -1e+06
 T169_L5_f119_le: const_Condition_1_Norm + 999999 t169_leaf5 <= 1e+06
 T169_R6_f4_ge: const_Overall_Cond - 1.000007e+06 t169_leaf6 >= -1e+06
 T169_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t169_leaf6 >= -1e+06
 T169_R6_f119_ge: const_Condition_1_Norm - 1.000001e+06 t169_leaf6
   >= -1e+06
 TREE_170_ONEHOT: t170_leaf0 + t170_leaf1 + t170_leaf2 + t170_leaf3
   + t170_leaf4 = 1
 T170_L0_f27_le: x_Kitchen_Qual <= 4
 T170_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t170_leaf0
   <= 1e+06
 T170_L0_f1_le: const_Lot_Area + 982773 t170_leaf0 <= 1e+06
 T170_L1_f27_le: x_Kitchen_Qual <= 4
 T170_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t170_leaf1
   <= 1e+06
 T170_R1_f1_ge: const_Lot_Area - 1.017227e+06 t170_leaf1 >= -1e+06
 T170_L2_f27_le: x_Kitchen_Qual <= 4
 T170_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t170_leaf2
   >= -1e+06
 T170_L2_f1_le: const_Lot_Area + 990100 t170_leaf2 <= 1e+06
 T170_L3_f27_le: x_Kitchen_Qual <= 4
 T170_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t170_leaf3
   >= -1e+06
 T170_R3_f1_ge: const_Lot_Area - 1.0099e+06 t170_leaf3 >= -1e+06
 T170_R4_f27_ge: x_Kitchen_Qual - 5 t170_leaf4 >= -1
 TREE_171_ONEHOT: t171_leaf0 + t171_leaf1 + t171_leaf2 + t171_leaf3
   + t171_leaf4 = 1
 T171_L0_f11_le: x_Bsmt_Cond + t171_leaf0 <= 4
 T171_L0_f96_le: const_Neighborhood_Edwards + 999999 t171_leaf0 <= 1e+06
 T171_L0_f40_le: x_Screen_Porch + 99790 t171_leaf0 <= 100000
 T171_L1_f11_le: x_Bsmt_Cond + t171_leaf1 <= 4
 T171_L1_f96_le: const_Neighborhood_Edwards + 999999 t171_leaf1 <= 1e+06
 T171_R1_f40_ge: x_Screen_Porch - 210 t171_leaf1 >= 0
 T171_L2_f11_le: x_Bsmt_Cond + t171_leaf2 <= 4
 T171_R2_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t171_leaf2
   >= -1e+06
 T171_L2_f81_le: const_Lot_Config_Corner + 999999 t171_leaf2 <= 1e+06
 T171_L3_f11_le: x_Bsmt_Cond + t171_leaf3 <= 4
 T171_R3_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t171_leaf3
   >= -1e+06
 T171_R3_f81_ge: const_Lot_Config_Corner - 1.000001e+06 t171_leaf3
   >= -1e+06
 T171_R4_f11_ge: x_Bsmt_Cond - 4 t171_leaf4 >= -1
 TREE_172_ONEHOT: t172_leaf0 + t172_leaf1 + t172_leaf2 + t172_leaf3
   + t172_leaf4 + t172_leaf5 + t172_leaf6 = 1
 T172_L0_f211_le: x_b1_is_ALQ <= 1
 T172_L0_f36_le: x_Wood_Deck_SF + 99944 t172_leaf0 <= 100000
 T172_L0_f24_le: x_Half_Bath + 4 t172_leaf0 <= 5
 T172_L1_f211_le: x_b1_is_ALQ <= 1
 T172_L1_f36_le: x_Wood_Deck_SF + 99944 t172_leaf1 <= 100000
 T172_R1_f24_ge: x_Half_Bath - t172_leaf1 >= 0
 T172_L2_f211_le: x_b1_is_ALQ <= 1
 T172_R2_f36_ge: x_Wood_Deck_SF - 56 t172_leaf2 >= 0
 T172_L2_f28_le: x_TotRms_AbvGrd + 4 t172_leaf2 <= 15
 T172_L3_f211_le: x_b1_is_ALQ <= 1
 T172_R3_f36_ge: x_Wood_Deck_SF - 56 t172_leaf3 >= 0
 T172_R3_f28_ge: x_TotRms_AbvGrd - 11 t172_leaf3 >= 0
 T172_R4_f211_ge: x_b1_is_ALQ - t172_leaf4 >= 0
 T172_L4_f245_le: const_FunctioNo_aplical_Typ + 999999 t172_leaf4 <= 1e+06
 T172_R5_f211_ge: x_b1_is_ALQ - t172_leaf5 >= 0
 T172_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t172_leaf5
   >= -1e+06
 T172_L5_f31_le: const_Garage_Yr_Blt + 998023 t172_leaf5 <= 1e+06
 T172_R6_f211_ge: x_b1_is_ALQ - t172_leaf6 >= 0
 T172_R6_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t172_leaf6
   >= -1e+06
 T172_R6_f31_ge: const_Garage_Yr_Blt - 1.001977e+06 t172_leaf6 >= -1e+06
 TREE_173_ONEHOT: t173_leaf0 + t173_leaf1 + t173_leaf2 + t173_leaf3
   + t173_leaf4 + t173_leaf5 + t173_leaf6 = 1
 T173_L0_f9_le: x_Exter_Cond + 2 t173_leaf0 <= 4
 T173_L0_f297_le: const_Sale_Condition_Normal + 999999 t173_leaf0 <= 1e+06
 T173_L1_f9_le: x_Exter_Cond + 2 t173_leaf1 <= 4
 T173_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t173_leaf1
   >= -1e+06
 T173_L1_f28_le: x_TotRms_AbvGrd + 10 t173_leaf1 <= 15
 T173_L2_f9_le: x_Exter_Cond + 2 t173_leaf2 <= 4
 T173_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t173_leaf2
   >= -1e+06
 T173_R2_f28_ge: x_TotRms_AbvGrd - 5 t173_leaf2 >= 0
 T173_R3_f9_ge: x_Exter_Cond - 3 t173_leaf3 >= -1
 T173_L3_f4_le: const_Overall_Cond + 999993 t173_leaf3 <= 1e+06
 T173_L3_f5_le: const_Year_Built + 998046 t173_leaf3 <= 1e+06
 T173_R4_f9_ge: x_Exter_Cond - 3 t173_leaf4 >= -1
 T173_L4_f4_le: const_Overall_Cond + 999993 t173_leaf4 <= 1e+06
 T173_R4_f5_ge: const_Year_Built - 1.001954e+06 t173_leaf4 >= -1e+06
 T173_R5_f9_ge: x_Exter_Cond - 3 t173_leaf5 >= -1
 T173_R5_f4_ge: const_Overall_Cond - 1.000007e+06 t173_leaf5 >= -1e+06
 T173_L5_f31_le: const_Garage_Yr_Blt + 998005 t173_leaf5 <= 1e+06
 T173_R6_f9_ge: x_Exter_Cond - 3 t173_leaf6 >= -1
 T173_R6_f4_ge: const_Overall_Cond - 1.000007e+06 t173_leaf6 >= -1e+06
 T173_R6_f31_ge: const_Garage_Yr_Blt - 1.001995e+06 t173_leaf6 >= -1e+06
 TREE_174_ONEHOT: t174_leaf0 + t174_leaf1 + t174_leaf2 + t174_leaf3 = 1
 T174_L0_f33_le: x_Garage_Area + 98846 t174_leaf0 <= 100000
 T174_L0_f207_le: const_Bsmt_Exposure_Gd + 999999 t174_leaf0 <= 1e+06
 T174_L0_f44_le: const_Yr_Sold + 997993 t174_leaf0 <= 1e+06
 T174_L1_f33_le: x_Garage_Area + 98846 t174_leaf1 <= 100000
 T174_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t174_leaf1 <= 1e+06
 T174_R1_f44_ge: const_Yr_Sold - 1.002007e+06 t174_leaf1 >= -1e+06
 T174_L2_f33_le: x_Garage_Area + 98846 t174_leaf2 <= 100000
 T174_R2_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t174_leaf2
   >= -1e+06
 T174_R3_f33_ge: x_Garage_Area - 1154 t174_leaf3 >= 0
 TREE_175_ONEHOT: t175_leaf0 + t175_leaf1 + t175_leaf2 + t175_leaf3
   + t175_leaf4 = 1
 T175_L0_f15_le: const_Total_Bsmt_SF + 997864 t175_leaf0 <= 1e+06
 T175_L0_f10_le: const_Bsmt_Qual + 999998 t175_leaf0 <= 1e+06
 T175_L0_f33_le: x_Garage_Area + 99513 t175_leaf0 <= 100000
 T175_L1_f15_le: const_Total_Bsmt_SF + 997864 t175_leaf1 <= 1e+06
 T175_L1_f10_le: const_Bsmt_Qual + 999998 t175_leaf1 <= 1e+06
 T175_R1_f33_ge: x_Garage_Area - 487 t175_leaf1 >= 0
 T175_L2_f15_le: const_Total_Bsmt_SF + 997864 t175_leaf2 <= 1e+06
 T175_R2_f10_ge: const_Bsmt_Qual - 1.000002e+06 t175_leaf2 >= -1e+06
 T175_L2_f117_le: const_Condition_1_Artery + 999999 t175_leaf2 <= 1e+06
 T175_L3_f15_le: const_Total_Bsmt_SF + 997864 t175_leaf3 <= 1e+06
 T175_R3_f10_ge: const_Bsmt_Qual - 1.000002e+06 t175_leaf3 >= -1e+06
 T175_R3_f117_ge: const_Condition_1_Artery - 1.000001e+06 t175_leaf3
   >= -1e+06
 T175_R4_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t175_leaf4 >= -1e+06
 TREE_176_ONEHOT: t176_leaf0 + t176_leaf1 + t176_leaf2 + t176_leaf3
   + t176_leaf4 + t176_leaf5 + t176_leaf6 = 1
 T176_L0_f44_le: const_Yr_Sold + 997993 t176_leaf0 <= 1e+06
 T176_L0_f231_le: central_air_no + 999999 t176_leaf0 <= 1e+06
 T176_L0_f4_le: const_Overall_Cond + 999992 t176_leaf0 <= 1e+06
 T176_L1_f44_le: const_Yr_Sold + 997993 t176_leaf1 <= 1e+06
 T176_L1_f231_le: central_air_no + 999999 t176_leaf1 <= 1e+06
 T176_R1_f4_ge: const_Overall_Cond - 1.000008e+06 t176_leaf1 >= -1e+06
 T176_L2_f44_le: const_Yr_Sold + 997993 t176_leaf2 <= 1e+06
 T176_R2_f231_ge: central_air_no - 1.000001e+06 t176_leaf2 >= -1e+06
 T176_L2_f1_le: const_Lot_Area + 993760 t176_leaf2 <= 1e+06
 T176_L3_f44_le: const_Yr_Sold + 997993 t176_leaf3 <= 1e+06
 T176_R3_f231_ge: central_air_no - 1.000001e+06 t176_leaf3 >= -1e+06
 T176_R3_f1_ge: const_Lot_Area - 1.00624e+06 t176_leaf3 >= -1e+06
 T176_R4_f44_ge: const_Yr_Sold - 1.002007e+06 t176_leaf4 >= -1e+06
 T176_L4_f33_le: x_Garage_Area + 98846 t176_leaf4 <= 100000
 T176_L4_f245_le: const_FunctioNo_aplical_Typ + 999999 t176_leaf4 <= 1e+06
 T176_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t176_leaf5 >= -1e+06
 T176_L5_f33_le: x_Garage_Area + 98846 t176_leaf5 <= 100000
 T176_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t176_leaf5
   >= -1e+06
 T176_R6_f44_ge: const_Yr_Sold - 1.002007e+06 t176_leaf6 >= -1e+06
 T176_R6_f33_ge: x_Garage_Area - 1154 t176_leaf6 >= 0
 TREE_177_ONEHOT: t177_leaf0 + t177_leaf1 + t177_leaf2 + t177_leaf3
   + t177_leaf4 + t177_leaf5 + t177_leaf6 = 1
 T177_L0_f16_le: x_Heating_QC + t177_leaf0 <= 4
 T177_L0_f36_le: x_Wood_Deck_SF + 99841 t177_leaf0 <= 100000
 T177_L0_f33_le: x_Garage_Area + 99216 t177_leaf0 <= 100000
 T177_L1_f16_le: x_Heating_QC + t177_leaf1 <= 4
 T177_L1_f36_le: x_Wood_Deck_SF + 99841 t177_leaf1 <= 100000
 T177_R1_f33_ge: x_Garage_Area - 784 t177_leaf1 >= 0
 T177_L2_f16_le: x_Heating_QC + t177_leaf2 <= 4
 T177_R2_f36_ge: x_Wood_Deck_SF - 159 t177_leaf2 >= 0
 T177_L2_f31_le: const_Garage_Yr_Blt + 998022 t177_leaf2 <= 1e+06
 T177_L3_f16_le: x_Heating_QC + t177_leaf3 <= 4
 T177_R3_f36_ge: x_Wood_Deck_SF - 159 t177_leaf3 >= 0
 T177_R3_f31_ge: const_Garage_Yr_Blt - 1.001978e+06 t177_leaf3 >= -1e+06
 T177_R4_f16_ge: x_Heating_QC - 4 t177_leaf4 >= -1
 T177_L4_f37_le: x_Open_Porch_SF + 99866 t177_leaf4 <= 100000
 T177_L4_f3_le: Overall_Qual_calc + 999993 t177_leaf4 <= 1e+06
 T177_R5_f16_ge: x_Heating_QC - 4 t177_leaf5 >= -1
 T177_L5_f37_le: x_Open_Porch_SF + 99866 t177_leaf5 <= 100000
 T177_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t177_leaf5 >= -1e+06
 T177_R6_f16_ge: x_Heating_QC - 4 t177_leaf6 >= -1
 T177_R6_f37_ge: x_Open_Porch_SF - 134 t177_leaf6 >= 0
 TREE_178_ONEHOT: t178_leaf0 + t178_leaf1 + t178_leaf2 + t178_leaf3
   + t178_leaf4 = 1
 T178_L0_f44_le: const_Yr_Sold + 997993 t178_leaf0 <= 1e+06
 T178_L0_f103_le: const_Neighborhood_Mitchel + 999999 t178_leaf0 <= 1e+06
 T178_L1_f44_le: const_Yr_Sold + 997993 t178_leaf1 <= 1e+06
 T178_R1_f103_ge: const_Neighborhood_Mitchel - 1.000001e+06 t178_leaf1
   >= -1e+06
 T178_R2_f44_ge: const_Yr_Sold - 1.002007e+06 t178_leaf2 >= -1e+06
 T178_L2_f63_le: const_MS_Zoning_FV + 999999 t178_leaf2 <= 1e+06
 T178_L2_f69_le: const_Street_Pave + 999999 t178_leaf2 <= 1e+06
 T178_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t178_leaf3 >= -1e+06
 T178_L3_f63_le: const_MS_Zoning_FV + 999999 t178_leaf3 <= 1e+06
 T178_R3_f69_ge: const_Street_Pave - 1.000001e+06 t178_leaf3 >= -1e+06
 T178_R4_f44_ge: const_Yr_Sold - 1.002007e+06 t178_leaf4 >= -1e+06
 T178_R4_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t178_leaf4 >= -1e+06
 TREE_179_ONEHOT: t179_leaf0 + t179_leaf1 + t179_leaf2 + t179_leaf3
   + t179_leaf4 + t179_leaf5 = 1
 T179_L0_f20_le: x_Gr_Liv_Area + 9232 t179_leaf0 <= 10000
 T179_L0_f17_le: x_1st_Flr_SF + 4283 t179_leaf0 <= 5000
 T179_L0_f192_le: x_ex2_is_Wd_Sdng <= 1
 T179_L1_f20_le: x_Gr_Liv_Area + 9232 t179_leaf1 <= 10000
 T179_L1_f17_le: x_1st_Flr_SF + 4283 t179_leaf1 <= 5000
 T179_R1_f192_ge: x_ex2_is_Wd_Sdng - t179_leaf1 >= 0
 T179_L2_f20_le: x_Gr_Liv_Area + 9232 t179_leaf2 <= 10000
 T179_R2_f17_ge: x_1st_Flr_SF - 717 t179_leaf2 >= 0
 T179_R3_f20_ge: x_Gr_Liv_Area - 768 t179_leaf3 >= 0
 T179_L3_f44_le: const_Yr_Sold + 997993 t179_leaf3 <= 1e+06
 T179_R4_f20_ge: x_Gr_Liv_Area - 768 t179_leaf4 >= 0
 T179_R4_f44_ge: const_Yr_Sold - 1.002007e+06 t179_leaf4 >= -1e+06
 T179_L4_f37_le: x_Open_Porch_SF + 99817 t179_leaf4 <= 100000
 T179_R5_f20_ge: x_Gr_Liv_Area - 768 t179_leaf5 >= 0
 T179_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t179_leaf5 >= -1e+06
 T179_R5_f37_ge: x_Open_Porch_SF - 183 t179_leaf5 >= 0
 TREE_180_ONEHOT: t180_leaf0 + t180_leaf1 + t180_leaf2 + t180_leaf3
   + t180_leaf4 = 1
 T180_L0_f3_le: Overall_Qual_calc + 999997 t180_leaf0 <= 1e+06
 T180_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t180_leaf1 >= -1e+06
 T180_L1_f95_le: const_Neighborhood_Crawfor + 999999 t180_leaf1 <= 1e+06
 T180_L1_f16_le: x_Heating_QC + t180_leaf1 <= 4
 T180_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t180_leaf2 >= -1e+06
 T180_L2_f95_le: const_Neighborhood_Crawfor + 999999 t180_leaf2 <= 1e+06
 T180_R2_f16_ge: x_Heating_QC - 4 t180_leaf2 >= -1
 T180_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t180_leaf3 >= -1e+06
 T180_R3_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t180_leaf3
   >= -1e+06
 T180_L3_f81_le: const_Lot_Config_Corner + 999999 t180_leaf3 <= 1e+06
 T180_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t180_leaf4 >= -1e+06
 T180_R4_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t180_leaf4
   >= -1e+06
 T180_R4_f81_ge: const_Lot_Config_Corner - 1.000001e+06 t180_leaf4
   >= -1e+06
 TREE_181_ONEHOT: t181_leaf0 + t181_leaf1 + t181_leaf2 + t181_leaf3
   + t181_leaf4 + t181_leaf5 + t181_leaf6 = 1
 T181_L0_f14_le: x_Bsmt_Unf_SF + 99887 t181_leaf0 <= 100000
 T181_L0_f11_le: x_Bsmt_Cond + t181_leaf0 <= 4
 T181_L0_f12_le: x_BsmtFin_SF_1 + 999375 t181_leaf0 <= 1e+06
 T181_L1_f14_le: x_Bsmt_Unf_SF + 99887 t181_leaf1 <= 100000
 T181_L1_f11_le: x_Bsmt_Cond + t181_leaf1 <= 4
 T181_R1_f12_ge: x_BsmtFin_SF_1 - 625 t181_leaf1 >= 0
 T181_L2_f14_le: x_Bsmt_Unf_SF + 99887 t181_leaf2 <= 100000
 T181_R2_f11_ge: x_Bsmt_Cond - 4 t181_leaf2 >= -1
 T181_R3_f14_ge: x_Bsmt_Unf_SF - 113 t181_leaf3 >= 0
 T181_L3_f20_le: x_Gr_Liv_Area + 9136 t181_leaf3 <= 10000
 T181_L3_f5_le: const_Year_Built + 998076 t181_leaf3 <= 1e+06
 T181_R4_f14_ge: x_Bsmt_Unf_SF - 113 t181_leaf4 >= 0
 T181_L4_f20_le: x_Gr_Liv_Area + 9136 t181_leaf4 <= 10000
 T181_R4_f5_ge: const_Year_Built - 1.001924e+06 t181_leaf4 >= -1e+06
 T181_R5_f14_ge: x_Bsmt_Unf_SF - 113 t181_leaf5 >= 0
 T181_R5_f20_ge: x_Gr_Liv_Area - 864 t181_leaf5 >= 0
 T181_L5_f18_le: const_2nd_Flr_SF + 998632 t181_leaf5 <= 1e+06
 T181_R6_f14_ge: x_Bsmt_Unf_SF - 113 t181_leaf6 >= 0
 T181_R6_f20_ge: x_Gr_Liv_Area - 864 t181_leaf6 >= 0
 T181_R6_f18_ge: const_2nd_Flr_SF - 1.001368e+06 t181_leaf6 >= -1e+06
 TREE_182_ONEHOT: t182_leaf0 + t182_leaf1 + t182_leaf2 + t182_leaf3
   + t182_leaf4 + t182_leaf5 + t182_leaf6 + t182_leaf7 = 1
 T182_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t182_leaf0
   <= 1e+06
 T182_L0_f1_le: const_Lot_Area + 985320 t182_leaf0 <= 1e+06
 T182_L0_f17_le: x_1st_Flr_SF + 4424 t182_leaf0 <= 5000
 T182_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t182_leaf1
   <= 1e+06
 T182_L1_f1_le: const_Lot_Area + 985320 t182_leaf1 <= 1e+06
 T182_R1_f17_ge: x_1st_Flr_SF - 576 t182_leaf1 >= 0
 T182_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t182_leaf2
   <= 1e+06
 T182_R2_f1_ge: const_Lot_Area - 1.01468e+06 t182_leaf2 >= -1e+06
 T182_L2_f37_le: x_Open_Porch_SF + 99726 t182_leaf2 <= 100000
 T182_L3_f293_le: const_Sale_Condition_Abnorml + 999999 t182_leaf3
   <= 1e+06
 T182_R3_f1_ge: const_Lot_Area - 1.01468e+06 t182_leaf3 >= -1e+06
 T182_R3_f37_ge: x_Open_Porch_SF - 274 t182_leaf3 >= 0
 T182_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t182_leaf4
   >= -1e+06
 T182_L4_f3_le: Overall_Qual_calc + 999995 t182_leaf4 <= 1e+06
 T182_L4_f4_le: const_Overall_Cond + 999994 t182_leaf4 <= 1e+06
 T182_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t182_leaf5
   >= -1e+06
 T182_L5_f3_le: Overall_Qual_calc + 999995 t182_leaf5 <= 1e+06
 T182_R5_f4_ge: const_Overall_Cond - 1.000006e+06 t182_leaf5 >= -1e+06
 T182_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t182_leaf6
   >= -1e+06
 T182_R6_f3_ge: Overall_Qual_calc - 1.000005e+06 t182_leaf6 >= -1e+06
 T182_L6_f237_le: x_elect_is_SBrkr <= 1
 T182_R7_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t182_leaf7
   >= -1e+06
 T182_R7_f3_ge: Overall_Qual_calc - 1.000005e+06 t182_leaf7 >= -1e+06
 T182_R7_f237_ge: x_elect_is_SBrkr - t182_leaf7 >= 0
 TREE_183_ONEHOT: t183_leaf0 + t183_leaf1 + t183_leaf2 + t183_leaf3
   + t183_leaf4 + t183_leaf5 = 1
 T183_L0_f118_le: const_Condition_1_Feedr + 999999 t183_leaf0 <= 1e+06
 T183_L0_f21_le: const_Bsmt_Full_Bath + 999998 t183_leaf0 <= 1e+06
 T183_L0_f4_le: const_Overall_Cond + 999993 t183_leaf0 <= 1e+06
 T183_L1_f118_le: const_Condition_1_Feedr + 999999 t183_leaf1 <= 1e+06
 T183_L1_f21_le: const_Bsmt_Full_Bath + 999998 t183_leaf1 <= 1e+06
 T183_R1_f4_ge: const_Overall_Cond - 1.000007e+06 t183_leaf1 >= -1e+06
 T183_L2_f118_le: const_Condition_1_Feedr + 999999 t183_leaf2 <= 1e+06
 T183_R2_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t183_leaf2 >= -1e+06
 T183_R3_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t183_leaf3
   >= -1e+06
 T183_L3_f80_le: const_Land_Contour_Lvl + 999999 t183_leaf3 <= 1e+06
 T183_L3_f175_le: x_ex1_is_Wd_Sdng <= 1
 T183_R4_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t183_leaf4
   >= -1e+06
 T183_L4_f80_le: const_Land_Contour_Lvl + 999999 t183_leaf4 <= 1e+06
 T183_R4_f175_ge: x_ex1_is_Wd_Sdng - t183_leaf4 >= 0
 T183_R5_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t183_leaf5
   >= -1e+06
 T183_R5_f80_ge: const_Land_Contour_Lvl - 1.000001e+06 t183_leaf5
   >= -1e+06
 TREE_184_ONEHOT: t184_leaf0 + t184_leaf1 + t184_leaf2 + t184_leaf3
   + t184_leaf4 + t184_leaf5 = 1
 T184_L0_f20_le: x_Gr_Liv_Area + 8039 t184_leaf0 <= 10000
 T184_L0_f113_le: const_Neighborhood_Somerst + 999999 t184_leaf0 <= 1e+06
 T184_L0_f211_le: x_b1_is_ALQ <= 1
 T184_L1_f20_le: x_Gr_Liv_Area + 8039 t184_leaf1 <= 10000
 T184_L1_f113_le: const_Neighborhood_Somerst + 999999 t184_leaf1 <= 1e+06
 T184_R1_f211_ge: x_b1_is_ALQ - t184_leaf1 >= 0
 T184_L2_f20_le: x_Gr_Liv_Area + 8039 t184_leaf2 <= 10000
 T184_R2_f113_ge: const_Neighborhood_Somerst - 1.000001e+06 t184_leaf2
   >= -1e+06
 T184_R3_f20_ge: x_Gr_Liv_Area - 1961 t184_leaf3 >= 0
 T184_L3_f96_le: const_Neighborhood_Edwards + 999999 t184_leaf3 <= 1e+06
 T184_L3_f15_le: const_Total_Bsmt_SF + 997864 t184_leaf3 <= 1e+06
 T184_R4_f20_ge: x_Gr_Liv_Area - 1961 t184_leaf4 >= 0
 T184_L4_f96_le: const_Neighborhood_Edwards + 999999 t184_leaf4 <= 1e+06
 T184_R4_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t184_leaf4 >= -1e+06
 T184_R5_f20_ge: x_Gr_Liv_Area - 1961 t184_leaf5 >= 0
 T184_R5_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t184_leaf5
   >= -1e+06
 TREE_185_ONEHOT: t185_leaf0 + t185_leaf1 + t185_leaf2 + t185_leaf3
   + t185_leaf4 + t185_leaf5 = 1
 T185_L0_f12_le: x_BsmtFin_SF_1 + 998268 t185_leaf0 <= 1e+06
 T185_L0_f29_le: const_Fireplaces + 999998 t185_leaf0 <= 1e+06
 T185_L0_f36_le: x_Wood_Deck_SF + 99791 t185_leaf0 <= 100000
 T185_L1_f12_le: x_BsmtFin_SF_1 + 998268 t185_leaf1 <= 1e+06
 T185_L1_f29_le: const_Fireplaces + 999998 t185_leaf1 <= 1e+06
 T185_R1_f36_ge: x_Wood_Deck_SF - 209 t185_leaf1 >= 0
 T185_L2_f12_le: x_BsmtFin_SF_1 + 998268 t185_leaf2 <= 1e+06
 T185_R2_f29_ge: const_Fireplaces - 1.000002e+06 t185_leaf2 >= -1e+06
 T185_L2_f15_le: const_Total_Bsmt_SF + 998518 t185_leaf2 <= 1e+06
 T185_L3_f12_le: x_BsmtFin_SF_1 + 998268 t185_leaf3 <= 1e+06
 T185_R3_f29_ge: const_Fireplaces - 1.000002e+06 t185_leaf3 >= -1e+06
 T185_R3_f15_ge: const_Total_Bsmt_SF - 1.001482e+06 t185_leaf3 >= -1e+06
 T185_R4_f12_ge: x_BsmtFin_SF_1 - 1732 t185_leaf4 >= 0
 T185_L4_f76_le: const_Lot_Shape_Reg + 999999 t185_leaf4 <= 1e+06
 T185_R5_f12_ge: x_BsmtFin_SF_1 - 1732 t185_leaf5 >= 0
 T185_R5_f76_ge: const_Lot_Shape_Reg - 1.000001e+06 t185_leaf5 >= -1e+06
 TREE_186_ONEHOT: t186_leaf0 + t186_leaf1 + t186_leaf2 + t186_leaf3
   + t186_leaf4 + t186_leaf5 + t186_leaf6 = 1
 T186_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t186_leaf0
   <= 1e+06
 T186_L0_f5_le: const_Year_Built + 997995 t186_leaf0 <= 1e+06
 T186_L0_f141_le: const_House_Style_1Story + 999999 t186_leaf0 <= 1e+06
 T186_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t186_leaf1
   <= 1e+06
 T186_L1_f5_le: const_Year_Built + 997995 t186_leaf1 <= 1e+06
 T186_R1_f141_ge: const_House_Style_1Story - 1.000001e+06 t186_leaf1
   >= -1e+06
 T186_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t186_leaf2
   <= 1e+06
 T186_R2_f5_ge: const_Year_Built - 1.002005e+06 t186_leaf2 >= -1e+06
 T186_L2_f66_le: const_MS_Zoning_RL + 999999 t186_leaf2 <= 1e+06
 T186_L3_f293_le: const_Sale_Condition_Abnorml + 999999 t186_leaf3
   <= 1e+06
 T186_R3_f5_ge: const_Year_Built - 1.002005e+06 t186_leaf3 >= -1e+06
 T186_R3_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t186_leaf3 >= -1e+06
 T186_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t186_leaf4
   >= -1e+06
 T186_L4_f3_le: Overall_Qual_calc + 999995 t186_leaf4 <= 1e+06
 T186_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t186_leaf5
   >= -1e+06
 T186_R5_f3_ge: Overall_Qual_calc - 1.000005e+06 t186_leaf5 >= -1e+06
 T186_L5_f38_le: x_Enclosed_Porch + 99968 t186_leaf5 <= 100000
 T186_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t186_leaf6
   >= -1e+06
 T186_R6_f3_ge: Overall_Qual_calc - 1.000005e+06 t186_leaf6 >= -1e+06
 T186_R6_f38_ge: x_Enclosed_Porch - 32 t186_leaf6 >= 0
 TREE_187_ONEHOT: t187_leaf0 + t187_leaf1 + t187_leaf2 + t187_leaf3
   + t187_leaf4 + t187_leaf5 + t187_leaf6 = 1
 T187_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t187_leaf0
   <= 1e+06
 T187_L0_f117_le: const_Condition_1_Artery + 999999 t187_leaf0 <= 1e+06
 T187_L0_f4_le: const_Overall_Cond + 999994 t187_leaf0 <= 1e+06
 T187_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t187_leaf1
   <= 1e+06
 T187_L1_f117_le: const_Condition_1_Artery + 999999 t187_leaf1 <= 1e+06
 T187_R1_f4_ge: const_Overall_Cond - 1.000006e+06 t187_leaf1 >= -1e+06
 T187_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t187_leaf2
   <= 1e+06
 T187_R2_f117_ge: const_Condition_1_Artery - 1.000001e+06 t187_leaf2
   >= -1e+06
 T187_L2_f16_le: x_Heating_QC + t187_leaf2 <= 4
 T187_L3_f293_le: const_Sale_Condition_Abnorml + 999999 t187_leaf3
   <= 1e+06
 T187_R3_f117_ge: const_Condition_1_Artery - 1.000001e+06 t187_leaf3
   >= -1e+06
 T187_R3_f16_ge: x_Heating_QC - 4 t187_leaf3 >= -1
 T187_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t187_leaf4
   >= -1e+06
 T187_L4_f4_le: const_Overall_Cond + 999996 t187_leaf4 <= 1e+06
 T187_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t187_leaf5
   >= -1e+06
 T187_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t187_leaf5 >= -1e+06
 T187_L5_f206_le: const_Bsmt_Exposure_Av + 999999 t187_leaf5 <= 1e+06
 T187_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t187_leaf6
   >= -1e+06
 T187_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t187_leaf6 >= -1e+06
 T187_R6_f206_ge: const_Bsmt_Exposure_Av - 1.000001e+06 t187_leaf6
   >= -1e+06
 TREE_188_ONEHOT: t188_leaf0 + t188_leaf1 + t188_leaf2 + t188_leaf3
   + t188_leaf4 + t188_leaf5 + t188_leaf6 + t188_leaf7 = 1
 T188_L0_f44_le: const_Yr_Sold + 997993 t188_leaf0 <= 1e+06
 T188_L0_f14_le: x_Bsmt_Unf_SF + 99033 t188_leaf0 <= 100000
 T188_L0_f6_le: const_Year_Remod_Add + 998049 t188_leaf0 <= 1e+06
 T188_L1_f44_le: const_Yr_Sold + 997993 t188_leaf1 <= 1e+06
 T188_L1_f14_le: x_Bsmt_Unf_SF + 99033 t188_leaf1 <= 100000
 T188_R1_f6_ge: const_Year_Remod_Add - 1.001951e+06 t188_leaf1 >= -1e+06
 T188_L2_f44_le: const_Yr_Sold + 997993 t188_leaf2 <= 1e+06
 T188_R2_f14_ge: x_Bsmt_Unf_SF - 967 t188_leaf2 >= 0
 T188_L2_f191_le: x_ex2_is_VinylSd <= 1
 T188_L3_f44_le: const_Yr_Sold + 997993 t188_leaf3 <= 1e+06
 T188_R3_f14_ge: x_Bsmt_Unf_SF - 967 t188_leaf3 >= 0
 T188_R3_f191_ge: x_ex2_is_VinylSd - t188_leaf3 >= 0
 T188_R4_f44_ge: const_Yr_Sold - 1.002007e+06 t188_leaf4 >= -1e+06
 T188_L4_f78_le: const_Land_Contour_HLS + 999999 t188_leaf4 <= 1e+06
 T188_L4_f218_le: x_b2_is_ALQ <= 1
 T188_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t188_leaf5 >= -1e+06
 T188_L5_f78_le: const_Land_Contour_HLS + 999999 t188_leaf5 <= 1e+06
 T188_R5_f218_ge: x_b2_is_ALQ - t188_leaf5 >= 0
 T188_R6_f44_ge: const_Yr_Sold - 1.002007e+06 t188_leaf6 >= -1e+06
 T188_R6_f78_ge: const_Land_Contour_HLS - 1.000001e+06 t188_leaf6
   >= -1e+06
 T188_L6_f12_le: x_BsmtFin_SF_1 + 999919 t188_leaf6 <= 1e+06
 T188_R7_f44_ge: const_Yr_Sold - 1.002007e+06 t188_leaf7 >= -1e+06
 T188_R7_f78_ge: const_Land_Contour_HLS - 1.000001e+06 t188_leaf7
   >= -1e+06
 T188_R7_f12_ge: x_BsmtFin_SF_1 - 81 t188_leaf7 >= 0
 TREE_189_ONEHOT: t189_leaf0 + t189_leaf1 + t189_leaf2 + t189_leaf3
   + t189_leaf4 = 1
 T189_L0_f37_le: x_Open_Porch_SF + 99556 t189_leaf0 <= 100000
 T189_L0_f207_le: const_Bsmt_Exposure_Gd + 999999 t189_leaf0 <= 1e+06
 T189_L0_f1_le: const_Lot_Area + 996124 t189_leaf0 <= 1e+06
 T189_L1_f37_le: x_Open_Porch_SF + 99556 t189_leaf1 <= 100000
 T189_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t189_leaf1 <= 1e+06
 T189_R1_f1_ge: const_Lot_Area - 1.003876e+06 t189_leaf1 >= -1e+06
 T189_L2_f37_le: x_Open_Porch_SF + 99556 t189_leaf2 <= 100000
 T189_R2_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t189_leaf2
   >= -1e+06
 T189_L2_f38_le: x_Enclosed_Porch + 99965 t189_leaf2 <= 100000
 T189_L3_f37_le: x_Open_Porch_SF + 99556 t189_leaf3 <= 100000
 T189_R3_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t189_leaf3
   >= -1e+06
 T189_R3_f38_ge: x_Enclosed_Porch - 35 t189_leaf3 >= 0
 T189_R4_f37_ge: x_Open_Porch_SF - 444 t189_leaf4 >= 0
 TREE_190_ONEHOT: t190_leaf0 + t190_leaf1 + t190_leaf2 + t190_leaf3
   + t190_leaf4 + t190_leaf5 = 1
 T190_L0_f31_le: const_Garage_Yr_Blt + 997995 t190_leaf0 <= 1e+06
 T190_L0_f20_le: x_Gr_Liv_Area + 9253 t190_leaf0 <= 10000
 T190_L0_f14_le: x_Bsmt_Unf_SF + 99727 t190_leaf0 <= 100000
 T190_L1_f31_le: const_Garage_Yr_Blt + 997995 t190_leaf1 <= 1e+06
 T190_L1_f20_le: x_Gr_Liv_Area + 9253 t190_leaf1 <= 10000
 T190_R1_f14_ge: x_Bsmt_Unf_SF - 273 t190_leaf1 >= 0
 T190_L2_f31_le: const_Garage_Yr_Blt + 997995 t190_leaf2 <= 1e+06
 T190_R2_f20_ge: x_Gr_Liv_Area - 747 t190_leaf2 >= 0
 T190_L2_f93_le: const_Neighborhood_ClearCr + 999999 t190_leaf2 <= 1e+06
 T190_L3_f31_le: const_Garage_Yr_Blt + 997995 t190_leaf3 <= 1e+06
 T190_R3_f20_ge: x_Gr_Liv_Area - 747 t190_leaf3 >= 0
 T190_R3_f93_ge: const_Neighborhood_ClearCr - 1.000001e+06 t190_leaf3
   >= -1e+06
 T190_R4_f31_ge: const_Garage_Yr_Blt - 1.002005e+06 t190_leaf4 >= -1e+06
 T190_L4_f40_le: x_Screen_Porch + 99960 t190_leaf4 <= 100000
 T190_R5_f31_ge: const_Garage_Yr_Blt - 1.002005e+06 t190_leaf5 >= -1e+06
 T190_R5_f40_ge: x_Screen_Porch - 40 t190_leaf5 >= 0
 TREE_191_ONEHOT: t191_leaf0 + t191_leaf1 + t191_leaf2 + t191_leaf3
   + t191_leaf4 + t191_leaf5 + t191_leaf6 = 1
 T191_L0_f1_le: const_Lot_Area + 976743 t191_leaf0 <= 1e+06
 T191_L0_f20_le: x_Gr_Liv_Area + 9253 t191_leaf0 <= 10000
 T191_L0_f66_le: const_MS_Zoning_RL + 999999 t191_leaf0 <= 1e+06
 T191_L1_f1_le: const_Lot_Area + 976743 t191_leaf1 <= 1e+06
 T191_L1_f20_le: x_Gr_Liv_Area + 9253 t191_leaf1 <= 10000
 T191_R1_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t191_leaf1 >= -1e+06
 T191_L2_f1_le: const_Lot_Area + 976743 t191_leaf2 <= 1e+06
 T191_R2_f20_ge: x_Gr_Liv_Area - 747 t191_leaf2 >= 0
 T191_L2_f20_le: x_Gr_Liv_Area + 6914 t191_leaf2 <= 10000
 T191_L3_f1_le: const_Lot_Area + 976743 t191_leaf3 <= 1e+06
 T191_R3_f20_ge: x_Gr_Liv_Area - 747 t191_leaf3 >= 0
 T191_R3_f20_ge: x_Gr_Liv_Area - 3086 t191_leaf3 >= 0
 T191_R4_f1_ge: const_Lot_Area - 1.023257e+06 t191_leaf4 >= -1e+06
 T191_L4_f297_le: const_Sale_Condition_Normal + 999999 t191_leaf4 <= 1e+06
 T191_R5_f1_ge: const_Lot_Area - 1.023257e+06 t191_leaf5 >= -1e+06
 T191_R5_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t191_leaf5
   >= -1e+06
 T191_L5_f36_le: x_Wood_Deck_SF + 99978 t191_leaf5 <= 100000
 T191_R6_f1_ge: const_Lot_Area - 1.023257e+06 t191_leaf6 >= -1e+06
 T191_R6_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t191_leaf6
   >= -1e+06
 T191_R6_f36_ge: x_Wood_Deck_SF - 22 t191_leaf6 >= 0
 TREE_192_ONEHOT: t192_leaf0 + t192_leaf1 + t192_leaf2 + t192_leaf3
   + t192_leaf4 = 1
 T192_L0_f211_le: x_b1_is_ALQ <= 1
 T192_L0_f82_le: const_Lot_Config_CulDSac + 999999 t192_leaf0 <= 1e+06
 T192_L0_f36_le: x_Wood_Deck_SF + 99944 t192_leaf0 <= 100000
 T192_L1_f211_le: x_b1_is_ALQ <= 1
 T192_L1_f82_le: const_Lot_Config_CulDSac + 999999 t192_leaf1 <= 1e+06
 T192_R1_f36_ge: x_Wood_Deck_SF - 56 t192_leaf1 >= 0
 T192_L2_f211_le: x_b1_is_ALQ <= 1
 T192_R2_f82_ge: const_Lot_Config_CulDSac - 1.000001e+06 t192_leaf2
   >= -1e+06
 T192_R3_f211_ge: x_b1_is_ALQ - t192_leaf3 >= 0
 T192_L3_f44_le: const_Yr_Sold + 997991 t192_leaf3 <= 1e+06
 T192_R4_f211_ge: x_b1_is_ALQ - t192_leaf4 >= 0
 T192_R4_f44_ge: const_Yr_Sold - 1.002009e+06 t192_leaf4 >= -1e+06
 TREE_193_ONEHOT: t193_leaf0 + t193_leaf1 + t193_leaf2 + t193_leaf3
   + t193_leaf4 + t193_leaf5 + t193_leaf6 + t193_leaf7 = 1
 T193_L0_f12_le: x_BsmtFin_SF_1 + 999935 t193_leaf0 <= 1e+06
 T193_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t193_leaf0
   <= 1e+06
 T193_L0_f33_le: x_Garage_Area + 99164 t193_leaf0 <= 100000
 T193_L1_f12_le: x_BsmtFin_SF_1 + 999935 t193_leaf1 <= 1e+06
 T193_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t193_leaf1
   <= 1e+06
 T193_R1_f33_ge: x_Garage_Area - 836 t193_leaf1 >= 0
 T193_L2_f12_le: x_BsmtFin_SF_1 + 999935 t193_leaf2 <= 1e+06
 T193_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t193_leaf2
   >= -1e+06
 T193_L2_f33_le: x_Garage_Area + 99549 t193_leaf2 <= 100000
 T193_L3_f12_le: x_BsmtFin_SF_1 + 999935 t193_leaf3 <= 1e+06
 T193_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t193_leaf3
   >= -1e+06
 T193_R3_f33_ge: x_Garage_Area - 451 t193_leaf3 >= 0
 T193_R4_f12_ge: x_BsmtFin_SF_1 - 65 t193_leaf4 >= 0
 T193_L4_f214_le: x_b1_is_LwQ <= 1
 T193_L4_f5_le: const_Year_Built + 998081 t193_leaf4 <= 1e+06
 T193_R5_f12_ge: x_BsmtFin_SF_1 - 65 t193_leaf5 >= 0
 T193_L5_f214_le: x_b1_is_LwQ <= 1
 T193_R5_f5_ge: const_Year_Built - 1.001919e+06 t193_leaf5 >= -1e+06
 T193_R6_f12_ge: x_BsmtFin_SF_1 - 65 t193_leaf6 >= 0
 T193_R6_f214_ge: x_b1_is_LwQ - t193_leaf6 >= 0
 T193_L6_f209_le: const_Bsmt_Exposure_No + 999999 t193_leaf6 <= 1e+06
 T193_R7_f12_ge: x_BsmtFin_SF_1 - 65 t193_leaf7 >= 0
 T193_R7_f214_ge: x_b1_is_LwQ - t193_leaf7 >= 0
 T193_R7_f209_ge: const_Bsmt_Exposure_No - 1.000001e+06 t193_leaf7
   >= -1e+06
 TREE_194_ONEHOT: t194_leaf0 + t194_leaf1 + t194_leaf2 + t194_leaf3 = 1
 T194_L0_f20_le: x_Gr_Liv_Area + 6914 t194_leaf0 <= 10000
 T194_L0_f21_le: const_Bsmt_Full_Bath + 999998 t194_leaf0 <= 1e+06
 T194_L0_f4_le: const_Overall_Cond + 999996 t194_leaf0 <= 1e+06
 T194_L1_f20_le: x_Gr_Liv_Area + 6914 t194_leaf1 <= 10000
 T194_L1_f21_le: const_Bsmt_Full_Bath + 999998 t194_leaf1 <= 1e+06
 T194_R1_f4_ge: const_Overall_Cond - 1.000004e+06 t194_leaf1 >= -1e+06
 T194_L2_f20_le: x_Gr_Liv_Area + 6914 t194_leaf2 <= 10000
 T194_R2_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t194_leaf2 >= -1e+06
 T194_R3_f20_ge: x_Gr_Liv_Area - 3086 t194_leaf3 >= 0
 TREE_195_ONEHOT: t195_leaf0 + t195_leaf1 + t195_leaf2 + t195_leaf3 = 1
 T195_L0_f20_le: x_Gr_Liv_Area + 6914 t195_leaf0 <= 10000
 T195_L0_f0_le: const_Lot_Frontage + 999972 t195_leaf0 <= 1e+06
 T195_L1_f20_le: x_Gr_Liv_Area + 6914 t195_leaf1 <= 10000
 T195_R1_f0_ge: const_Lot_Frontage - 1.000028e+06 t195_leaf1 >= -1e+06
 T195_L1_f3_le: Overall_Qual_calc + 999993 t195_leaf1 <= 1e+06
 T195_L2_f20_le: x_Gr_Liv_Area + 6914 t195_leaf2 <= 10000
 T195_R2_f0_ge: const_Lot_Frontage - 1.000028e+06 t195_leaf2 >= -1e+06
 T195_R2_f3_ge: Overall_Qual_calc - 1.000007e+06 t195_leaf2 >= -1e+06
 T195_R3_f20_ge: x_Gr_Liv_Area - 3086 t195_leaf3 >= 0
 TREE_196_ONEHOT: t196_leaf0 + t196_leaf1 + t196_leaf2 + t196_leaf3
   + t196_leaf4 + t196_leaf5 = 1
 T196_L0_f118_le: const_Condition_1_Feedr + 999999 t196_leaf0 <= 1e+06
 T196_L0_f164_le: x_ex1_is_BrkFace <= 1
 T196_L0_f27_le: x_Kitchen_Qual <= 4
 T196_L1_f118_le: const_Condition_1_Feedr + 999999 t196_leaf1 <= 1e+06
 T196_L1_f164_le: x_ex1_is_BrkFace <= 1
 T196_R1_f27_ge: x_Kitchen_Qual - 5 t196_leaf1 >= -1
 T196_L2_f118_le: const_Condition_1_Feedr + 999999 t196_leaf2 <= 1e+06
 T196_R2_f164_ge: x_ex1_is_BrkFace - t196_leaf2 >= 0
 T196_L2_f106_le: const_Neighborhood_No_aplicames + 999999 t196_leaf2
   <= 1e+06
 T196_L3_f118_le: const_Condition_1_Feedr + 999999 t196_leaf3 <= 1e+06
 T196_R3_f164_ge: x_ex1_is_BrkFace - t196_leaf3 >= 0
 T196_R3_f106_ge: const_Neighborhood_No_aplicames - 1.000001e+06 t196_leaf3
   >= -1e+06
 T196_R4_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t196_leaf4
   >= -1e+06
 T196_L4_f44_le: const_Yr_Sold + 997992 t196_leaf4 <= 1e+06
 T196_R5_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t196_leaf5
   >= -1e+06
 T196_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t196_leaf5 >= -1e+06
 TREE_197_ONEHOT: t197_leaf0 + t197_leaf1 + t197_leaf2 + t197_leaf3
   + t197_leaf4 + t197_leaf5 = 1
 T197_L0_f37_le: x_Open_Porch_SF + 99718 t197_leaf0 <= 100000
 T197_L0_f1_le: const_Lot_Area + 982773 t197_leaf0 <= 1e+06
 T197_L0_f3_le: Overall_Qual_calc + 999991 t197_leaf0 <= 1e+06
 T197_L1_f37_le: x_Open_Porch_SF + 99718 t197_leaf1 <= 100000
 T197_L1_f1_le: const_Lot_Area + 982773 t197_leaf1 <= 1e+06
 T197_R1_f3_ge: Overall_Qual_calc - 1.000009e+06 t197_leaf1 >= -1e+06
 T197_L2_f37_le: x_Open_Porch_SF + 99718 t197_leaf2 <= 100000
 T197_R2_f1_ge: const_Lot_Area - 1.017227e+06 t197_leaf2 >= -1e+06
 T197_L2_f283_le: const_Sale_Type_COD + 999999 t197_leaf2 <= 1e+06
 T197_L3_f37_le: x_Open_Porch_SF + 99718 t197_leaf3 <= 100000
 T197_R3_f1_ge: const_Lot_Area - 1.017227e+06 t197_leaf3 >= -1e+06
 T197_R3_f283_ge: const_Sale_Type_COD - 1.000001e+06 t197_leaf3 >= -1e+06
 T197_R4_f37_ge: x_Open_Porch_SF - 282 t197_leaf4 >= 0
 T197_L4_f297_le: const_Sale_Condition_Normal + 999999 t197_leaf4 <= 1e+06
 T197_R5_f37_ge: x_Open_Porch_SF - 282 t197_leaf5 >= 0
 T197_R5_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t197_leaf5
   >= -1e+06
 TREE_198_ONEHOT: t198_leaf0 + t198_leaf1 + t198_leaf2 + t198_leaf3
   + t198_leaf4 + t198_leaf5 = 1
 T198_L0_f3_le: Overall_Qual_calc + 999997 t198_leaf0 <= 1e+06
 T198_L0_f31_le: const_Garage_Yr_Blt + 998079 t198_leaf0 <= 1e+06
 T198_L1_f3_le: Overall_Qual_calc + 999997 t198_leaf1 <= 1e+06
 T198_R1_f31_ge: const_Garage_Yr_Blt - 1.001921e+06 t198_leaf1 >= -1e+06
 T198_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t198_leaf2 >= -1e+06
 T198_L2_f298_le: const_Sale_Condition_Partial + 999999 t198_leaf2
   <= 1e+06
 T198_L2_f3_le: Overall_Qual_calc + 999993 t198_leaf2 <= 1e+06
 T198_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t198_leaf3 >= -1e+06
 T198_L3_f298_le: const_Sale_Condition_Partial + 999999 t198_leaf3
   <= 1e+06
 T198_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t198_leaf3 >= -1e+06
 T198_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t198_leaf4 >= -1e+06
 T198_R4_f298_ge: const_Sale_Condition_Partial - 1.000001e+06 t198_leaf4
   >= -1e+06
 T198_L4_f37_le: x_Open_Porch_SF + 99756 t198_leaf4 <= 100000
 T198_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t198_leaf5 >= -1e+06
 T198_R5_f298_ge: const_Sale_Condition_Partial - 1.000001e+06 t198_leaf5
   >= -1e+06
 T198_R5_f37_ge: x_Open_Porch_SF - 244 t198_leaf5 >= 0
 TREE_199_ONEHOT: t199_leaf0 + t199_leaf1 + t199_leaf2 + t199_leaf3
   + t199_leaf4 = 1
 T199_L0_f40_le: x_Screen_Porch + 99850 t199_leaf0 <= 100000
 T199_L0_f5_le: const_Year_Built + 998089 t199_leaf0 <= 1e+06
 T199_L0_f161_le: x_ex1_is_AsbShng <= 1
 T199_L1_f40_le: x_Screen_Porch + 99850 t199_leaf1 <= 100000
 T199_L1_f5_le: const_Year_Built + 998089 t199_leaf1 <= 1e+06
 T199_R1_f161_ge: x_ex1_is_AsbShng - t199_leaf1 >= 0
 T199_L2_f40_le: x_Screen_Porch + 99850 t199_leaf2 <= 100000
 T199_R2_f5_ge: const_Year_Built - 1.001911e+06 t199_leaf2 >= -1e+06
 T199_L2_f18_le: const_2nd_Flr_SF + 999416 t199_leaf2 <= 1e+06
 T199_L3_f40_le: x_Screen_Porch + 99850 t199_leaf3 <= 100000
 T199_R3_f5_ge: const_Year_Built - 1.001911e+06 t199_leaf3 >= -1e+06
 T199_R3_f18_ge: const_2nd_Flr_SF - 1.000584e+06 t199_leaf3 >= -1e+06
 T199_R4_f40_ge: x_Screen_Porch - 150 t199_leaf4 >= 0
 TREE_200_ONEHOT: t200_leaf0 + t200_leaf1 + t200_leaf2 + t200_leaf3
   + t200_leaf4 = 1
 T200_L0_f164_le: x_ex1_is_BrkFace <= 1
 T200_L0_f216_le: x_b1_is_Rec <= 1
 T200_L0_f119_le: const_Condition_1_Norm + 999999 t200_leaf0 <= 1e+06
 T200_L1_f164_le: x_ex1_is_BrkFace <= 1
 T200_L1_f216_le: x_b1_is_Rec <= 1
 T200_R1_f119_ge: const_Condition_1_Norm - 1.000001e+06 t200_leaf1
   >= -1e+06
 T200_L2_f164_le: x_ex1_is_BrkFace <= 1
 T200_R2_f216_ge: x_b1_is_Rec - t200_leaf2 >= 0
 T200_R3_f164_ge: x_ex1_is_BrkFace - t200_leaf3 >= 0
 T200_L3_f245_le: const_FunctioNo_aplical_Typ + 999999 t200_leaf3 <= 1e+06
 T200_R4_f164_ge: x_ex1_is_BrkFace - t200_leaf4 >= 0
 T200_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t200_leaf4
   >= -1e+06
 TREE_201_ONEHOT: t201_leaf0 + t201_leaf1 + t201_leaf2 + t201_leaf3
   + t201_leaf4 = 1
 T201_L0_f21_le: const_Bsmt_Full_Bath + 999998 t201_leaf0 <= 1e+06
 T201_L0_f239_le: const_FunctioNo_aplical_Maj2 + 999999 t201_leaf0
   <= 1e+06
 T201_L0_f233_le: x_elect_is_FuseA <= 1
 T201_L1_f21_le: const_Bsmt_Full_Bath + 999998 t201_leaf1 <= 1e+06
 T201_L1_f239_le: const_FunctioNo_aplical_Maj2 + 999999 t201_leaf1
   <= 1e+06
 T201_R1_f233_ge: x_elect_is_FuseA - t201_leaf1 >= 0
 T201_L2_f21_le: const_Bsmt_Full_Bath + 999998 t201_leaf2 <= 1e+06
 T201_R2_f239_ge: const_FunctioNo_aplical_Maj2 - 1.000001e+06 t201_leaf2
   >= -1e+06
 T201_R3_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t201_leaf3 >= -1e+06
 T201_L3_f4_le: const_Overall_Cond + 999993 t201_leaf3 <= 1e+06
 T201_R4_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t201_leaf4 >= -1e+06
 T201_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t201_leaf4 >= -1e+06
 TREE_202_ONEHOT: t202_leaf0 + t202_leaf1 + t202_leaf2 + t202_leaf3
   + t202_leaf4 + t202_leaf5 + t202_leaf6 = 1
 T202_L0_f119_le: const_Condition_1_Norm + 999999 t202_leaf0 <= 1e+06
 T202_L0_f14_le: x_Bsmt_Unf_SF + 99004 t202_leaf0 <= 100000
 T202_L0_f32_le: const_Garage_Cars + 999997 t202_leaf0 <= 1e+06
 T202_L1_f119_le: const_Condition_1_Norm + 999999 t202_leaf1 <= 1e+06
 T202_L1_f14_le: x_Bsmt_Unf_SF + 99004 t202_leaf1 <= 100000
 T202_R1_f32_ge: const_Garage_Cars - 1.000003e+06 t202_leaf1 >= -1e+06
 T202_L2_f119_le: const_Condition_1_Norm + 999999 t202_leaf2 <= 1e+06
 T202_R2_f14_ge: x_Bsmt_Unf_SF - 996 t202_leaf2 >= 0
 T202_R3_f119_ge: const_Condition_1_Norm - 1.000001e+06 t202_leaf3
   >= -1e+06
 T202_L3_f44_le: const_Yr_Sold + 997991 t202_leaf3 <= 1e+06
 T202_L3_f4_le: const_Overall_Cond + 999993 t202_leaf3 <= 1e+06
 T202_R4_f119_ge: const_Condition_1_Norm - 1.000001e+06 t202_leaf4
   >= -1e+06
 T202_L4_f44_le: const_Yr_Sold + 997991 t202_leaf4 <= 1e+06
 T202_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t202_leaf4 >= -1e+06
 T202_R5_f119_ge: const_Condition_1_Norm - 1.000001e+06 t202_leaf5
   >= -1e+06
 T202_R5_f44_ge: const_Yr_Sold - 1.002009e+06 t202_leaf5 >= -1e+06
 T202_L5_f231_le: central_air_no + 999999 t202_leaf5 <= 1e+06
 T202_R6_f119_ge: const_Condition_1_Norm - 1.000001e+06 t202_leaf6
   >= -1e+06
 T202_R6_f44_ge: const_Yr_Sold - 1.002009e+06 t202_leaf6 >= -1e+06
 T202_R6_f231_ge: central_air_no - 1.000001e+06 t202_leaf6 >= -1e+06
 TREE_203_ONEHOT: t203_leaf0 + t203_leaf1 + t203_leaf2 + t203_leaf3
   + t203_leaf4 + t203_leaf5 + t203_leaf6 + t203_leaf7 = 1
 T203_L0_f44_le: const_Yr_Sold + 997991 t203_leaf0 <= 1e+06
 T203_L0_f29_le: const_Fireplaces + 999998 t203_leaf0 <= 1e+06
 T203_L0_f289_le: const_Sale_Type_New + 999999 t203_leaf0 <= 1e+06
 T203_L1_f44_le: const_Yr_Sold + 997991 t203_leaf1 <= 1e+06
 T203_L1_f29_le: const_Fireplaces + 999998 t203_leaf1 <= 1e+06
 T203_R1_f289_ge: const_Sale_Type_New - 1.000001e+06 t203_leaf1 >= -1e+06
 T203_L2_f44_le: const_Yr_Sold + 997991 t203_leaf2 <= 1e+06
 T203_R2_f29_ge: const_Fireplaces - 1.000002e+06 t203_leaf2 >= -1e+06
 T203_L2_f4_le: const_Overall_Cond + 999994 t203_leaf2 <= 1e+06
 T203_L3_f44_le: const_Yr_Sold + 997991 t203_leaf3 <= 1e+06
 T203_R3_f29_ge: const_Fireplaces - 1.000002e+06 t203_leaf3 >= -1e+06
 T203_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t203_leaf3 >= -1e+06
 T203_R4_f44_ge: const_Yr_Sold - 1.002009e+06 t203_leaf4 >= -1e+06
 T203_L4_f293_le: const_Sale_Condition_Abnorml + 999999 t203_leaf4
   <= 1e+06
 T203_L4_f148_le: const_Roof_Style_Gable + 999999 t203_leaf4 <= 1e+06
 T203_R5_f44_ge: const_Yr_Sold - 1.002009e+06 t203_leaf5 >= -1e+06
 T203_L5_f293_le: const_Sale_Condition_Abnorml + 999999 t203_leaf5
   <= 1e+06
 T203_R5_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t203_leaf5
   >= -1e+06
 T203_R6_f44_ge: const_Yr_Sold - 1.002009e+06 t203_leaf6 >= -1e+06
 T203_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t203_leaf6
   >= -1e+06
 T203_L6_f148_le: const_Roof_Style_Gable + 999999 t203_leaf6 <= 1e+06
 T203_R7_f44_ge: const_Yr_Sold - 1.002009e+06 t203_leaf7 >= -1e+06
 T203_R7_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t203_leaf7
   >= -1e+06
 T203_R7_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t203_leaf7
   >= -1e+06
 TREE_204_ONEHOT: t204_leaf0 + t204_leaf1 + t204_leaf2 + t204_leaf3
   + t204_leaf4 = 1
 T204_L0_f18_le: const_2nd_Flr_SF + 998704 t204_leaf0 <= 1e+06
 T204_L0_f9_le: x_Exter_Cond + 2 t204_leaf0 <= 4
 T204_L0_f20_le: x_Gr_Liv_Area + 8983 t204_leaf0 <= 10000
 T204_L1_f18_le: const_2nd_Flr_SF + 998704 t204_leaf1 <= 1e+06
 T204_L1_f9_le: x_Exter_Cond + 2 t204_leaf1 <= 4
 T204_R1_f20_ge: x_Gr_Liv_Area - 1017 t204_leaf1 >= 0
 T204_L2_f18_le: const_2nd_Flr_SF + 998704 t204_leaf2 <= 1e+06
 T204_R2_f9_ge: x_Exter_Cond - 3 t204_leaf2 >= -1
 T204_L2_f209_le: const_Bsmt_Exposure_No + 999999 t204_leaf2 <= 1e+06
 T204_L3_f18_le: const_2nd_Flr_SF + 998704 t204_leaf3 <= 1e+06
 T204_R3_f9_ge: x_Exter_Cond - 3 t204_leaf3 >= -1
 T204_R3_f209_ge: const_Bsmt_Exposure_No - 1.000001e+06 t204_leaf3
   >= -1e+06
 T204_R4_f18_ge: const_2nd_Flr_SF - 1.001296e+06 t204_leaf4 >= -1e+06
 TREE_205_ONEHOT: t205_leaf0 + t205_leaf1 + t205_leaf2 + t205_leaf3 = 1
 T205_L0_f113_le: const_Neighborhood_Somerst + 999999 t205_leaf0 <= 1e+06
 T205_L0_f1_le: const_Lot_Area + 995500 t205_leaf0 <= 1e+06
 T205_L1_f113_le: const_Neighborhood_Somerst + 999999 t205_leaf1 <= 1e+06
 T205_R1_f1_ge: const_Lot_Area - 1.0045e+06 t205_leaf1 >= -1e+06
 T205_L1_f40_le: x_Screen_Porch + 99857 t205_leaf1 <= 100000
 T205_L2_f113_le: const_Neighborhood_Somerst + 999999 t205_leaf2 <= 1e+06
 T205_R2_f1_ge: const_Lot_Area - 1.0045e+06 t205_leaf2 >= -1e+06
 T205_R2_f40_ge: x_Screen_Porch - 143 t205_leaf2 >= 0
 T205_R3_f113_ge: const_Neighborhood_Somerst - 1.000001e+06 t205_leaf3
   >= -1e+06
 TREE_206_ONEHOT: t206_leaf0 + t206_leaf1 + t206_leaf2 + t206_leaf3 = 1
 T206_L0_f20_le: x_Gr_Liv_Area + 6914 t206_leaf0 <= 10000
 T206_L0_f6_le: const_Year_Remod_Add + 997992 t206_leaf0 <= 1e+06
 T206_L0_f4_le: const_Overall_Cond + 999997 t206_leaf0 <= 1e+06
 T206_L1_f20_le: x_Gr_Liv_Area + 6914 t206_leaf1 <= 10000
 T206_L1_f6_le: const_Year_Remod_Add + 997992 t206_leaf1 <= 1e+06
 T206_R1_f4_ge: const_Overall_Cond - 1.000003e+06 t206_leaf1 >= -1e+06
 T206_L2_f20_le: x_Gr_Liv_Area + 6914 t206_leaf2 <= 10000
 T206_R2_f6_ge: const_Year_Remod_Add - 1.002008e+06 t206_leaf2 >= -1e+06
 T206_R3_f20_ge: x_Gr_Liv_Area - 3086 t206_leaf3 >= 0
 TREE_207_ONEHOT: t207_leaf0 + t207_leaf1 + t207_leaf2 + t207_leaf3
   + t207_leaf4 + t207_leaf5 = 1
 T207_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t207_leaf0
   <= 1e+06
 T207_L0_f164_le: x_ex1_is_BrkFace <= 1
 T207_L0_f77_le: const_Land_Contour_Bnk + 999999 t207_leaf0 <= 1e+06
 T207_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t207_leaf1
   <= 1e+06
 T207_L1_f164_le: x_ex1_is_BrkFace <= 1
 T207_R1_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t207_leaf1
   >= -1e+06
 T207_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t207_leaf2
   <= 1e+06
 T207_R2_f164_ge: x_ex1_is_BrkFace - t207_leaf2 >= 0
 T207_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t207_leaf3
   >= -1e+06
 T207_L3_f3_le: Overall_Qual_calc + 999996 t207_leaf3 <= 1e+06
 T207_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t207_leaf4
   >= -1e+06
 T207_R4_f3_ge: Overall_Qual_calc - 1.000004e+06 t207_leaf4 >= -1e+06
 T207_L4_f14_le: x_Bsmt_Unf_SF + 99328 t207_leaf4 <= 100000
 T207_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t207_leaf5
   >= -1e+06
 T207_R5_f3_ge: Overall_Qual_calc - 1.000004e+06 t207_leaf5 >= -1e+06
 T207_R5_f14_ge: x_Bsmt_Unf_SF - 672 t207_leaf5 >= 0
 TREE_208_ONEHOT: t208_leaf0 + t208_leaf1 + t208_leaf2 + t208_leaf3
   + t208_leaf4 = 1
 T208_L0_f37_le: x_Open_Porch_SF + 99516 t208_leaf0 <= 100000
 T208_L0_f7_le: x_Mas_Vnr_Area + 1527 t208_leaf0 <= 2000
 T208_L0_f211_le: x_b1_is_ALQ <= 1
 T208_L1_f37_le: x_Open_Porch_SF + 99516 t208_leaf1 <= 100000
 T208_L1_f7_le: x_Mas_Vnr_Area + 1527 t208_leaf1 <= 2000
 T208_R1_f211_ge: x_b1_is_ALQ - t208_leaf1 >= 0
 T208_L2_f37_le: x_Open_Porch_SF + 99516 t208_leaf2 <= 100000
 T208_R2_f7_ge: x_Mas_Vnr_Area - 473 t208_leaf2 >= 0
 T208_L2_f18_le: const_2nd_Flr_SF + 998694 t208_leaf2 <= 1e+06
 T208_L3_f37_le: x_Open_Porch_SF + 99516 t208_leaf3 <= 100000
 T208_R3_f7_ge: x_Mas_Vnr_Area - 473 t208_leaf3 >= 0
 T208_R3_f18_ge: const_2nd_Flr_SF - 1.001306e+06 t208_leaf3 >= -1e+06
 T208_R4_f37_ge: x_Open_Porch_SF - 484 t208_leaf4 >= 0
 TREE_209_ONEHOT: t209_leaf0 + t209_leaf1 + t209_leaf2 + t209_leaf3
   + t209_leaf4 = 1
 T209_L0_f31_le: const_Garage_Yr_Blt + 997992 t209_leaf0 <= 1e+06
 T209_L0_f1_le: const_Lot_Area + 980478 t209_leaf0 <= 1e+06
 T209_L0_f117_le: const_Condition_1_Artery + 999999 t209_leaf0 <= 1e+06
 T209_L1_f31_le: const_Garage_Yr_Blt + 997992 t209_leaf1 <= 1e+06
 T209_L1_f1_le: const_Lot_Area + 980478 t209_leaf1 <= 1e+06
 T209_R1_f117_ge: const_Condition_1_Artery - 1.000001e+06 t209_leaf1
   >= -1e+06
 T209_L2_f31_le: const_Garage_Yr_Blt + 997992 t209_leaf2 <= 1e+06
 T209_R2_f1_ge: const_Lot_Area - 1.019522e+06 t209_leaf2 >= -1e+06
 T209_L2_f0_le: const_Lot_Frontage + 999919.5 t209_leaf2 <= 1e+06
 T209_L3_f31_le: const_Garage_Yr_Blt + 997992 t209_leaf3 <= 1e+06
 T209_R3_f1_ge: const_Lot_Area - 1.019522e+06 t209_leaf3 >= -1e+06
 T209_R3_f0_ge: const_Lot_Frontage - 1.0000805e+06 t209_leaf3 >= -1e+06
 T209_R4_f31_ge: const_Garage_Yr_Blt - 1.002008e+06 t209_leaf4 >= -1e+06
 TREE_210_ONEHOT: t210_leaf0 + t210_leaf1 + t210_leaf2 + t210_leaf3
   + t210_leaf4 = 1
 T210_L0_f117_le: const_Condition_1_Artery + 999999 t210_leaf0 <= 1e+06
 T210_L0_f37_le: x_Open_Porch_SF + 99582 t210_leaf0 <= 100000
 T210_L0_f12_le: x_BsmtFin_SF_1 + 998035 t210_leaf0 <= 1e+06
 T210_L1_f117_le: const_Condition_1_Artery + 999999 t210_leaf1 <= 1e+06
 T210_L1_f37_le: x_Open_Porch_SF + 99582 t210_leaf1 <= 100000
 T210_R1_f12_ge: x_BsmtFin_SF_1 - 1965 t210_leaf1 >= 0
 T210_L2_f117_le: const_Condition_1_Artery + 999999 t210_leaf2 <= 1e+06
 T210_R2_f37_ge: x_Open_Porch_SF - 418 t210_leaf2 >= 0
 T210_R3_f117_ge: const_Condition_1_Artery - 1.000001e+06 t210_leaf3
   >= -1e+06
 T210_L3_f3_le: Overall_Qual_calc + 999996 t210_leaf3 <= 1e+06
 T210_R4_f117_ge: const_Condition_1_Artery - 1.000001e+06 t210_leaf4
   >= -1e+06
 T210_R4_f3_ge: Overall_Qual_calc - 1.000004e+06 t210_leaf4 >= -1e+06
 TREE_211_ONEHOT: t211_leaf0 + t211_leaf1 + t211_leaf2 + t211_leaf3
   + t211_leaf4 + t211_leaf5 + t211_leaf6 = 1
 T211_L0_f44_le: const_Yr_Sold + 997993 t211_leaf0 <= 1e+06
 T211_L0_f36_le: x_Wood_Deck_SF + 99850 t211_leaf0 <= 100000
 T211_L0_f33_le: x_Garage_Area + 99621 t211_leaf0 <= 100000
 T211_L1_f44_le: const_Yr_Sold + 997993 t211_leaf1 <= 1e+06
 T211_L1_f36_le: x_Wood_Deck_SF + 99850 t211_leaf1 <= 100000
 T211_R1_f33_ge: x_Garage_Area - 379 t211_leaf1 >= 0
 T211_L2_f44_le: const_Yr_Sold + 997993 t211_leaf2 <= 1e+06
 T211_R2_f36_ge: x_Wood_Deck_SF - 150 t211_leaf2 >= 0
 T211_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t211_leaf3 >= -1e+06
 T211_L3_f161_le: x_ex1_is_AsbShng <= 1
 T211_L3_f14_le: x_Bsmt_Unf_SF + 99524 t211_leaf3 <= 100000
 T211_R4_f44_ge: const_Yr_Sold - 1.002007e+06 t211_leaf4 >= -1e+06
 T211_L4_f161_le: x_ex1_is_AsbShng <= 1
 T211_R4_f14_ge: x_Bsmt_Unf_SF - 476 t211_leaf4 >= 0
 T211_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t211_leaf5 >= -1e+06
 T211_R5_f161_ge: x_ex1_is_AsbShng - t211_leaf5 >= 0
 T211_L5_f33_le: x_Garage_Area + 99634 t211_leaf5 <= 100000
 T211_R6_f44_ge: const_Yr_Sold - 1.002007e+06 t211_leaf6 >= -1e+06
 T211_R6_f161_ge: x_ex1_is_AsbShng - t211_leaf6 >= 0
 T211_R6_f33_ge: x_Garage_Area - 366 t211_leaf6 >= 0
 TREE_212_ONEHOT: t212_leaf0 + t212_leaf1 + t212_leaf2 + t212_leaf3
   + t212_leaf4 + t212_leaf5 = 1
 T212_L0_f9_le: x_Exter_Cond + 2 t212_leaf0 <= 4
 T212_L0_f0_le: const_Lot_Frontage + 999934 t212_leaf0 <= 1e+06
 T212_L1_f9_le: x_Exter_Cond + 2 t212_leaf1 <= 4
 T212_R1_f0_ge: const_Lot_Frontage - 1.000066e+06 t212_leaf1 >= -1e+06
 T212_L1_f37_le: x_Open_Porch_SF + 99996 t212_leaf1 <= 100000
 T212_L2_f9_le: x_Exter_Cond + 2 t212_leaf2 <= 4
 T212_R2_f0_ge: const_Lot_Frontage - 1.000066e+06 t212_leaf2 >= -1e+06
 T212_R2_f37_ge: x_Open_Porch_SF - 4 t212_leaf2 >= 0
 T212_R3_f9_ge: x_Exter_Cond - 3 t212_leaf3 >= -1
 T212_L3_f1_le: const_Lot_Area + 996325 t212_leaf3 <= 1e+06
 T212_R4_f9_ge: x_Exter_Cond - 3 t212_leaf4 >= -1
 T212_R4_f1_ge: const_Lot_Area - 1.003675e+06 t212_leaf4 >= -1e+06
 T212_L4_f31_le: const_Garage_Yr_Blt + 998059 t212_leaf4 <= 1e+06
 T212_R5_f9_ge: x_Exter_Cond - 3 t212_leaf5 >= -1
 T212_R5_f1_ge: const_Lot_Area - 1.003675e+06 t212_leaf5 >= -1e+06
 T212_R5_f31_ge: const_Garage_Yr_Blt - 1.001941e+06 t212_leaf5 >= -1e+06
 TREE_213_ONEHOT: t213_leaf0 + t213_leaf1 + t213_leaf2 + t213_leaf3 = 1
 T213_L0_f38_le: x_Enclosed_Porch + 99735 t213_leaf0 <= 100000
 T213_L0_f37_le: x_Open_Porch_SF + 99809 t213_leaf0 <= 100000
 T213_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t213_leaf0 <= 1e+06
 T213_L1_f38_le: x_Enclosed_Porch + 99735 t213_leaf1 <= 100000
 T213_L1_f37_le: x_Open_Porch_SF + 99809 t213_leaf1 <= 100000
 T213_R1_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t213_leaf1
   >= -1e+06
 T213_L2_f38_le: x_Enclosed_Porch + 99735 t213_leaf2 <= 100000
 T213_R2_f37_ge: x_Open_Porch_SF - 191 t213_leaf2 >= 0
 T213_R3_f38_ge: x_Enclosed_Porch - 265 t213_leaf3 >= 0
 TREE_214_ONEHOT: t214_leaf0 + t214_leaf1 + t214_leaf2 + t214_leaf3
   + t214_leaf4 + t214_leaf5 + t214_leaf6 = 1
 T214_L0_f117_le: const_Condition_1_Artery + 999999 t214_leaf0 <= 1e+06
 T214_L0_f4_le: const_Overall_Cond + 999993 t214_leaf0 <= 1e+06
 T214_L0_f31_le: const_Garage_Yr_Blt + 997995 t214_leaf0 <= 1e+06
 T214_L1_f117_le: const_Condition_1_Artery + 999999 t214_leaf1 <= 1e+06
 T214_L1_f4_le: const_Overall_Cond + 999993 t214_leaf1 <= 1e+06
 T214_R1_f31_ge: const_Garage_Yr_Blt - 1.002005e+06 t214_leaf1 >= -1e+06
 T214_L2_f117_le: const_Condition_1_Artery + 999999 t214_leaf2 <= 1e+06
 T214_R2_f4_ge: const_Overall_Cond - 1.000007e+06 t214_leaf2 >= -1e+06
 T214_L2_f31_le: const_Garage_Yr_Blt + 998092 t214_leaf2 <= 1e+06
 T214_L3_f117_le: const_Condition_1_Artery + 999999 t214_leaf3 <= 1e+06
 T214_R3_f4_ge: const_Overall_Cond - 1.000007e+06 t214_leaf3 >= -1e+06
 T214_R3_f31_ge: const_Garage_Yr_Blt - 1.001908e+06 t214_leaf3 >= -1e+06
 T214_R4_f117_ge: const_Condition_1_Artery - 1.000001e+06 t214_leaf4
   >= -1e+06
 T214_L4_f29_le: const_Fireplaces + 999999 t214_leaf4 <= 1e+06
 T214_L4_f1_le: const_Lot_Area + 989376 t214_leaf4 <= 1e+06
 T214_R5_f117_ge: const_Condition_1_Artery - 1.000001e+06 t214_leaf5
   >= -1e+06
 T214_L5_f29_le: const_Fireplaces + 999999 t214_leaf5 <= 1e+06
 T214_R5_f1_ge: const_Lot_Area - 1.010624e+06 t214_leaf5 >= -1e+06
 T214_R6_f117_ge: const_Condition_1_Artery - 1.000001e+06 t214_leaf6
   >= -1e+06
 T214_R6_f29_ge: const_Fireplaces - 1.000001e+06 t214_leaf6 >= -1e+06
 TREE_215_ONEHOT: t215_leaf0 + t215_leaf1 + t215_leaf2 + t215_leaf3
   + t215_leaf4 + t215_leaf5 = 1
 T215_L0_f3_le: Overall_Qual_calc + 999997 t215_leaf0 <= 1e+06
 T215_L0_f31_le: const_Garage_Yr_Blt + 998077 t215_leaf0 <= 1e+06
 T215_L1_f3_le: Overall_Qual_calc + 999997 t215_leaf1 <= 1e+06
 T215_R1_f31_ge: const_Garage_Yr_Blt - 1.001923e+06 t215_leaf1 >= -1e+06
 T215_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t215_leaf2 >= -1e+06
 T215_L2_f119_le: const_Condition_1_Norm + 999999 t215_leaf2 <= 1e+06
 T215_L2_f0_le: const_Lot_Frontage + 999879 t215_leaf2 <= 1e+06
 T215_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t215_leaf3 >= -1e+06
 T215_L3_f119_le: const_Condition_1_Norm + 999999 t215_leaf3 <= 1e+06
 T215_R3_f0_ge: const_Lot_Frontage - 1.000121e+06 t215_leaf3 >= -1e+06
 T215_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t215_leaf4 >= -1e+06
 T215_R4_f119_ge: const_Condition_1_Norm - 1.000001e+06 t215_leaf4
   >= -1e+06
 T215_L4_f4_le: const_Overall_Cond + 999994 t215_leaf4 <= 1e+06
 T215_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t215_leaf5 >= -1e+06
 T215_R5_f119_ge: const_Condition_1_Norm - 1.000001e+06 t215_leaf5
   >= -1e+06
 T215_R5_f4_ge: const_Overall_Cond - 1.000006e+06 t215_leaf5 >= -1e+06
 TREE_216_ONEHOT: t216_leaf0 + t216_leaf1 + t216_leaf2 + t216_leaf3
   + t216_leaf4 = 1
 T216_L0_f15_le: const_Total_Bsmt_SF + 997508 t216_leaf0 <= 1e+06
 T216_L0_f4_le: const_Overall_Cond + 999994 t216_leaf0 <= 1e+06
 T216_L0_f31_le: const_Garage_Yr_Blt + 997992 t216_leaf0 <= 1e+06
 T216_L1_f15_le: const_Total_Bsmt_SF + 997508 t216_leaf1 <= 1e+06
 T216_L1_f4_le: const_Overall_Cond + 999994 t216_leaf1 <= 1e+06
 T216_R1_f31_ge: const_Garage_Yr_Blt - 1.002008e+06 t216_leaf1 >= -1e+06
 T216_L2_f15_le: const_Total_Bsmt_SF + 997508 t216_leaf2 <= 1e+06
 T216_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t216_leaf2 >= -1e+06
 T216_L2_f31_le: const_Garage_Yr_Blt + 998092 t216_leaf2 <= 1e+06
 T216_L3_f15_le: const_Total_Bsmt_SF + 997508 t216_leaf3 <= 1e+06
 T216_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t216_leaf3 >= -1e+06
 T216_R3_f31_ge: const_Garage_Yr_Blt - 1.001908e+06 t216_leaf3 >= -1e+06
 T216_R4_f15_ge: const_Total_Bsmt_SF - 1.002492e+06 t216_leaf4 >= -1e+06
 TREE_217_ONEHOT: t217_leaf0 + t217_leaf1 + t217_leaf2 + t217_leaf3
   + t217_leaf4 = 1
 T217_L0_f33_le: x_Garage_Area + 98846 t217_leaf0 <= 100000
 T217_L0_f5_le: const_Year_Built + 998089 t217_leaf0 <= 1e+06
 T217_L0_f36_le: x_Wood_Deck_SF + 99912 t217_leaf0 <= 100000
 T217_L1_f33_le: x_Garage_Area + 98846 t217_leaf1 <= 100000
 T217_L1_f5_le: const_Year_Built + 998089 t217_leaf1 <= 1e+06
 T217_R1_f36_ge: x_Wood_Deck_SF - 88 t217_leaf1 >= 0
 T217_L2_f33_le: x_Garage_Area + 98846 t217_leaf2 <= 100000
 T217_R2_f5_ge: const_Year_Built - 1.001911e+06 t217_leaf2 >= -1e+06
 T217_L2_f6_le: const_Year_Remod_Add + 998035 t217_leaf2 <= 1e+06
 T217_L3_f33_le: x_Garage_Area + 98846 t217_leaf3 <= 100000
 T217_R3_f5_ge: const_Year_Built - 1.001911e+06 t217_leaf3 >= -1e+06
 T217_R3_f6_ge: const_Year_Remod_Add - 1.001965e+06 t217_leaf3 >= -1e+06
 T217_R4_f33_ge: x_Garage_Area - 1154 t217_leaf4 >= 0
 TREE_218_ONEHOT: t218_leaf0 + t218_leaf1 + t218_leaf2 + t218_leaf3 = 1
 T218_L0_f3_le: Overall_Qual_calc + 999990 t218_leaf0 <= 1e+06
 T218_L0_f37_le: x_Open_Porch_SF + 99806 t218_leaf0 <= 100000
 T218_L0_f289_le: const_Sale_Type_New + 999999 t218_leaf0 <= 1e+06
 T218_L1_f3_le: Overall_Qual_calc + 999990 t218_leaf1 <= 1e+06
 T218_L1_f37_le: x_Open_Porch_SF + 99806 t218_leaf1 <= 100000
 T218_R1_f289_ge: const_Sale_Type_New - 1.000001e+06 t218_leaf1 >= -1e+06
 T218_L2_f3_le: Overall_Qual_calc + 999990 t218_leaf2 <= 1e+06
 T218_R2_f37_ge: x_Open_Porch_SF - 194 t218_leaf2 >= 0
 T218_R3_f3_ge: Overall_Qual_calc - 1.00001e+06 t218_leaf3 >= -1e+06
 TREE_219_ONEHOT: t219_leaf0 + t219_leaf1 + t219_leaf2 + t219_leaf3
   + t219_leaf4 + t219_leaf5 + t219_leaf6 = 1
 T219_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t219_leaf0 <= 1e+06
 T219_L0_f86_le: const_Land_Slope_Gtl + 999999 t219_leaf0 <= 1e+06
 T219_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t219_leaf1 <= 1e+06
 T219_R1_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t219_leaf1 >= -1e+06
 T219_L1_f15_le: const_Total_Bsmt_SF + 999036 t219_leaf1 <= 1e+06
 T219_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t219_leaf2 <= 1e+06
 T219_R2_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t219_leaf2 >= -1e+06
 T219_R2_f15_ge: const_Total_Bsmt_SF - 1.000964e+06 t219_leaf2 >= -1e+06
 T219_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t219_leaf3
   >= -1e+06
 T219_L3_f1_le: const_Lot_Area + 984688 t219_leaf3 <= 1e+06
 T219_L3_f46_le: const_MS_SubClass_30 + 999999 t219_leaf3 <= 1e+06
 T219_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t219_leaf4
   >= -1e+06
 T219_L4_f1_le: const_Lot_Area + 984688 t219_leaf4 <= 1e+06
 T219_R4_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t219_leaf4 >= -1e+06
 T219_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t219_leaf5
   >= -1e+06
 T219_R5_f1_ge: const_Lot_Area - 1.015312e+06 t219_leaf5 >= -1e+06
 T219_L5_f12_le: x_BsmtFin_SF_1 + 998681 t219_leaf5 <= 1e+06
 T219_R6_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t219_leaf6
   >= -1e+06
 T219_R6_f1_ge: const_Lot_Area - 1.015312e+06 t219_leaf6 >= -1e+06
 T219_R6_f12_ge: x_BsmtFin_SF_1 - 1319 t219_leaf6 >= 0
 TREE_220_ONEHOT: t220_leaf0 + t220_leaf1 + t220_leaf2 + t220_leaf3
   + t220_leaf4 + t220_leaf5 + t220_leaf6 = 1
 T220_L0_f44_le: const_Yr_Sold + 997991 t220_leaf0 <= 1e+06
 T220_L0_f272_le: const_Mo_Sold_2 + 999999 t220_leaf0 <= 1e+06
 T220_L0_f250_le: const_Garage_Type_CarPort + 999999 t220_leaf0 <= 1e+06
 T220_L1_f44_le: const_Yr_Sold + 997991 t220_leaf1 <= 1e+06
 T220_L1_f272_le: const_Mo_Sold_2 + 999999 t220_leaf1 <= 1e+06
 T220_R1_f250_ge: const_Garage_Type_CarPort - 1.000001e+06 t220_leaf1
   >= -1e+06
 T220_L2_f44_le: const_Yr_Sold + 997991 t220_leaf2 <= 1e+06
 T220_R2_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t220_leaf2 >= -1e+06
 T220_L2_f15_le: const_Total_Bsmt_SF + 998511 t220_leaf2 <= 1e+06
 T220_L3_f44_le: const_Yr_Sold + 997991 t220_leaf3 <= 1e+06
 T220_R3_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t220_leaf3 >= -1e+06
 T220_R3_f15_ge: const_Total_Bsmt_SF - 1.001489e+06 t220_leaf3 >= -1e+06
 T220_R4_f44_ge: const_Yr_Sold - 1.002009e+06 t220_leaf4 >= -1e+06
 T220_L4_f293_le: const_Sale_Condition_Abnorml + 999999 t220_leaf4
   <= 1e+06
 T220_L4_f150_le: const_Roof_Style_Hip + 999999 t220_leaf4 <= 1e+06
 T220_R5_f44_ge: const_Yr_Sold - 1.002009e+06 t220_leaf5 >= -1e+06
 T220_L5_f293_le: const_Sale_Condition_Abnorml + 999999 t220_leaf5
   <= 1e+06
 T220_R5_f150_ge: const_Roof_Style_Hip - 1.000001e+06 t220_leaf5 >= -1e+06
 T220_R6_f44_ge: const_Yr_Sold - 1.002009e+06 t220_leaf6 >= -1e+06
 T220_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t220_leaf6
   >= -1e+06
 TREE_221_ONEHOT: t221_leaf0 + t221_leaf1 + t221_leaf2 + t221_leaf3
   + t221_leaf4 + t221_leaf5 = 1
 T221_L0_f44_le: const_Yr_Sold + 997993 t221_leaf0 <= 1e+06
 T221_L0_f231_le: central_air_no + 999999 t221_leaf0 <= 1e+06
 T221_L0_f4_le: const_Overall_Cond + 999994 t221_leaf0 <= 1e+06
 T221_L1_f44_le: const_Yr_Sold + 997993 t221_leaf1 <= 1e+06
 T221_L1_f231_le: central_air_no + 999999 t221_leaf1 <= 1e+06
 T221_R1_f4_ge: const_Overall_Cond - 1.000006e+06 t221_leaf1 >= -1e+06
 T221_L2_f44_le: const_Yr_Sold + 997993 t221_leaf2 <= 1e+06
 T221_R2_f231_ge: central_air_no - 1.000001e+06 t221_leaf2 >= -1e+06
 T221_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t221_leaf3 >= -1e+06
 T221_L3_f154_le: x_roof_matl_is_CompShg <= 1
 T221_R4_f44_ge: const_Yr_Sold - 1.002007e+06 t221_leaf4 >= -1e+06
 T221_R4_f154_ge: x_roof_matl_is_CompShg - t221_leaf4 >= 0
 T221_L4_f17_le: x_1st_Flr_SF + 2964 t221_leaf4 <= 5000
 T221_R5_f44_ge: const_Yr_Sold - 1.002007e+06 t221_leaf5 >= -1e+06
 T221_R5_f154_ge: x_roof_matl_is_CompShg - t221_leaf5 >= 0
 T221_R5_f17_ge: x_1st_Flr_SF - 2036 t221_leaf5 >= 0
 TREE_222_ONEHOT: t222_leaf0 + t222_leaf1 + t222_leaf2 + t222_leaf3
   + t222_leaf4 = 1
 T222_L0_f18_le: const_2nd_Flr_SF + 998371 t222_leaf0 <= 1e+06
 T222_L0_f44_le: const_Yr_Sold + 997993 t222_leaf0 <= 1e+06
 T222_L0_f0_le: const_Lot_Frontage + 999910 t222_leaf0 <= 1e+06
 T222_L1_f18_le: const_2nd_Flr_SF + 998371 t222_leaf1 <= 1e+06
 T222_L1_f44_le: const_Yr_Sold + 997993 t222_leaf1 <= 1e+06
 T222_R1_f0_ge: const_Lot_Frontage - 1.00009e+06 t222_leaf1 >= -1e+06
 T222_L2_f18_le: const_2nd_Flr_SF + 998371 t222_leaf2 <= 1e+06
 T222_R2_f44_ge: const_Yr_Sold - 1.002007e+06 t222_leaf2 >= -1e+06
 T222_L2_f6_le: const_Year_Remod_Add + 997993 t222_leaf2 <= 1e+06
 T222_L3_f18_le: const_2nd_Flr_SF + 998371 t222_leaf3 <= 1e+06
 T222_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t222_leaf3 >= -1e+06
 T222_R3_f6_ge: const_Year_Remod_Add - 1.002007e+06 t222_leaf3 >= -1e+06
 T222_R4_f18_ge: const_2nd_Flr_SF - 1.001629e+06 t222_leaf4 >= -1e+06
 TREE_223_ONEHOT: t223_leaf0 + t223_leaf1 + t223_leaf2 + t223_leaf3
   + t223_leaf4 + t223_leaf5 + t223_leaf6 = 1
 T223_L0_f119_le: const_Condition_1_Norm + 999999 t223_leaf0 <= 1e+06
 T223_L0_f0_le: const_Lot_Frontage + 999879 t223_leaf0 <= 1e+06
 T223_L0_f0_le: const_Lot_Frontage + 999920 t223_leaf0 <= 1e+06
 T223_L1_f119_le: const_Condition_1_Norm + 999999 t223_leaf1 <= 1e+06
 T223_L1_f0_le: const_Lot_Frontage + 999879 t223_leaf1 <= 1e+06
 T223_R1_f0_ge: const_Lot_Frontage - 1.00008e+06 t223_leaf1 >= -1e+06
 T223_L2_f119_le: const_Condition_1_Norm + 999999 t223_leaf2 <= 1e+06
 T223_R2_f0_ge: const_Lot_Frontage - 1.000121e+06 t223_leaf2 >= -1e+06
 T223_R3_f119_ge: const_Condition_1_Norm - 1.000001e+06 t223_leaf3
   >= -1e+06
 T223_L3_f29_le: const_Fireplaces + 999998 t223_leaf3 <= 1e+06
 T223_L3_f44_le: const_Yr_Sold + 997993 t223_leaf3 <= 1e+06
 T223_R4_f119_ge: const_Condition_1_Norm - 1.000001e+06 t223_leaf4
   >= -1e+06
 T223_L4_f29_le: const_Fireplaces + 999998 t223_leaf4 <= 1e+06
 T223_R4_f44_ge: const_Yr_Sold - 1.002007e+06 t223_leaf4 >= -1e+06
 T223_R5_f119_ge: const_Condition_1_Norm - 1.000001e+06 t223_leaf5
   >= -1e+06
 T223_R5_f29_ge: const_Fireplaces - 1.000002e+06 t223_leaf5 >= -1e+06
 T223_L5_f1_le: const_Lot_Area + 982129 t223_leaf5 <= 1e+06
 T223_R6_f119_ge: const_Condition_1_Norm - 1.000001e+06 t223_leaf6
   >= -1e+06
 T223_R6_f29_ge: const_Fireplaces - 1.000002e+06 t223_leaf6 >= -1e+06
 T223_R6_f1_ge: const_Lot_Area - 1.017871e+06 t223_leaf6 >= -1e+06
 TREE_224_ONEHOT: t224_leaf0 + t224_leaf1 + t224_leaf2 + t224_leaf3 = 1
 T224_L0_f33_le: x_Garage_Area + 98846 t224_leaf0 <= 100000
 T224_L0_f69_le: const_Street_Pave + 999999 t224_leaf0 <= 1e+06
 T224_L1_f33_le: x_Garage_Area + 98846 t224_leaf1 <= 100000
 T224_R1_f69_ge: const_Street_Pave - 1.000001e+06 t224_leaf1 >= -1e+06
 T224_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t224_leaf1 <= 1e+06
 T224_L2_f33_le: x_Garage_Area + 98846 t224_leaf2 <= 100000
 T224_R2_f69_ge: const_Street_Pave - 1.000001e+06 t224_leaf2 >= -1e+06
 T224_R2_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t224_leaf2
   >= -1e+06
 T224_R3_f33_ge: x_Garage_Area - 1154 t224_leaf3 >= 0
 TREE_225_ONEHOT: t225_leaf0 + t225_leaf1 + t225_leaf2 + t225_leaf3
   + t225_leaf4 = 1
 T225_L0_f21_le: const_Bsmt_Full_Bath + 999998 t225_leaf0 <= 1e+06
 T225_L0_f36_le: x_Wood_Deck_SF + 99788 t225_leaf0 <= 100000
 T225_L0_f96_le: const_Neighborhood_Edwards + 999999 t225_leaf0 <= 1e+06
 T225_L1_f21_le: const_Bsmt_Full_Bath + 999998 t225_leaf1 <= 1e+06
 T225_L1_f36_le: x_Wood_Deck_SF + 99788 t225_leaf1 <= 100000
 T225_R1_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t225_leaf1
   >= -1e+06
 T225_L2_f21_le: const_Bsmt_Full_Bath + 999998 t225_leaf2 <= 1e+06
 T225_R2_f36_ge: x_Wood_Deck_SF - 212 t225_leaf2 >= 0
 T225_L2_f0_le: const_Lot_Frontage + 999885 t225_leaf2 <= 1e+06
 T225_L3_f21_le: const_Bsmt_Full_Bath + 999998 t225_leaf3 <= 1e+06
 T225_R3_f36_ge: x_Wood_Deck_SF - 212 t225_leaf3 >= 0
 T225_R3_f0_ge: const_Lot_Frontage - 1.000115e+06 t225_leaf3 >= -1e+06
 T225_R4_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t225_leaf4 >= -1e+06
 TREE_226_ONEHOT: t226_leaf0 + t226_leaf1 + t226_leaf2 + t226_leaf3
   + t226_leaf4 + t226_leaf5 + t226_leaf6 + t226_leaf7 = 1
 T226_L0_f20_le: x_Gr_Liv_Area + 9012 t226_leaf0 <= 10000
 T226_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t226_leaf0
   <= 1e+06
 T226_L0_f231_le: central_air_no + 999999 t226_leaf0 <= 1e+06
 T226_L1_f20_le: x_Gr_Liv_Area + 9012 t226_leaf1 <= 10000
 T226_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t226_leaf1
   <= 1e+06
 T226_R1_f231_ge: central_air_no - 1.000001e+06 t226_leaf1 >= -1e+06
 T226_L2_f20_le: x_Gr_Liv_Area + 9012 t226_leaf2 <= 10000
 T226_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t226_leaf2
   >= -1e+06
 T226_L2_f44_le: const_Yr_Sold + 997992 t226_leaf2 <= 1e+06
 T226_L3_f20_le: x_Gr_Liv_Area + 9012 t226_leaf3 <= 10000
 T226_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t226_leaf3
   >= -1e+06
 T226_R3_f44_ge: const_Yr_Sold - 1.002008e+06 t226_leaf3 >= -1e+06
 T226_R4_f20_ge: x_Gr_Liv_Area - 988 t226_leaf4 >= 0
 T226_L4_f4_le: const_Overall_Cond + 999994 t226_leaf4 <= 1e+06
 T226_L4_f14_le: x_Bsmt_Unf_SF + 99534 t226_leaf4 <= 100000
 T226_R5_f20_ge: x_Gr_Liv_Area - 988 t226_leaf5 >= 0
 T226_L5_f4_le: const_Overall_Cond + 999994 t226_leaf5 <= 1e+06
 T226_R5_f14_ge: x_Bsmt_Unf_SF - 466 t226_leaf5 >= 0
 T226_R6_f20_ge: x_Gr_Liv_Area - 988 t226_leaf6 >= 0
 T226_R6_f4_ge: const_Overall_Cond - 1.000006e+06 t226_leaf6 >= -1e+06
 T226_L6_f5_le: const_Year_Built + 998108 t226_leaf6 <= 1e+06
 T226_R7_f20_ge: x_Gr_Liv_Area - 988 t226_leaf7 >= 0
 T226_R7_f4_ge: const_Overall_Cond - 1.000006e+06 t226_leaf7 >= -1e+06
 T226_R7_f5_ge: const_Year_Built - 1.001892e+06 t226_leaf7 >= -1e+06
 TREE_227_ONEHOT: t227_leaf0 + t227_leaf1 + t227_leaf2 + t227_leaf3
   + t227_leaf4 = 1
 T227_L0_f40_le: x_Screen_Porch + 99857 t227_leaf0 <= 100000
 T227_L0_f62_le: const_MS_Zoning_C_(all) + 999999 t227_leaf0 <= 1e+06
 T227_L0_f0_le: const_Lot_Frontage + 999870 t227_leaf0 <= 1e+06
 T227_L1_f40_le: x_Screen_Porch + 99857 t227_leaf1 <= 100000
 T227_L1_f62_le: const_MS_Zoning_C_(all) + 999999 t227_leaf1 <= 1e+06
 T227_R1_f0_ge: const_Lot_Frontage - 1.00013e+06 t227_leaf1 >= -1e+06
 T227_L2_f40_le: x_Screen_Porch + 99857 t227_leaf2 <= 100000
 T227_R2_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t227_leaf2
   >= -1e+06
 T227_L2_f1_le: const_Lot_Area + 993400 t227_leaf2 <= 1e+06
 T227_L3_f40_le: x_Screen_Porch + 99857 t227_leaf3 <= 100000
 T227_R3_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t227_leaf3
   >= -1e+06
 T227_R3_f1_ge: const_Lot_Area - 1.0066e+06 t227_leaf3 >= -1e+06
 T227_R4_f40_ge: x_Screen_Porch - 143 t227_leaf4 >= 0
 TREE_228_ONEHOT: t228_leaf0 + t228_leaf1 + t228_leaf2 + t228_leaf3
   + t228_leaf4 = 1
 T228_L0_f118_le: const_Condition_1_Feedr + 999999 t228_leaf0 <= 1e+06
 T228_L0_f23_le: x_Full_Bath + t228_leaf0 <= 4
 T228_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t228_leaf0 <= 1e+06
 T228_L1_f118_le: const_Condition_1_Feedr + 999999 t228_leaf1 <= 1e+06
 T228_L1_f23_le: x_Full_Bath + t228_leaf1 <= 4
 T228_R1_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t228_leaf1
   >= -1e+06
 T228_L2_f118_le: const_Condition_1_Feedr + 999999 t228_leaf2 <= 1e+06
 T228_R2_f23_ge: x_Full_Bath - 3 t228_leaf2 >= 0
 T228_R3_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t228_leaf3
   >= -1e+06
 T228_L3_f77_le: const_Land_Contour_Bnk + 999999 t228_leaf3 <= 1e+06
 T228_R4_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t228_leaf4
   >= -1e+06
 T228_R4_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t228_leaf4
   >= -1e+06
 TREE_229_ONEHOT: t229_leaf0 + t229_leaf1 + t229_leaf2 + t229_leaf3
   + t229_leaf4 = 1
 T229_L0_f29_le: const_Fireplaces + 999997 t229_leaf0 <= 1e+06
 T229_L0_f20_le: x_Gr_Liv_Area + 6914 t229_leaf0 <= 10000
 T229_L0_f1_le: const_Lot_Area + 995165 t229_leaf0 <= 1e+06
 T229_L1_f29_le: const_Fireplaces + 999997 t229_leaf1 <= 1e+06
 T229_L1_f20_le: x_Gr_Liv_Area + 6914 t229_leaf1 <= 10000
 T229_R1_f1_ge: const_Lot_Area - 1.004835e+06 t229_leaf1 >= -1e+06
 T229_L2_f29_le: const_Fireplaces + 999997 t229_leaf2 <= 1e+06
 T229_R2_f20_ge: x_Gr_Liv_Area - 3086 t229_leaf2 >= 0
 T229_L2_f37_le: x_Open_Porch_SF + 99965 t229_leaf2 <= 100000
 T229_L3_f29_le: const_Fireplaces + 999997 t229_leaf3 <= 1e+06
 T229_R3_f20_ge: x_Gr_Liv_Area - 3086 t229_leaf3 >= 0
 T229_R3_f37_ge: x_Open_Porch_SF - 35 t229_leaf3 >= 0
 T229_R4_f29_ge: const_Fireplaces - 1.000003e+06 t229_leaf4 >= -1e+06
 TREE_230_ONEHOT: t230_leaf0 + t230_leaf1 + t230_leaf2 + t230_leaf3
   + t230_leaf4 + t230_leaf5 + t230_leaf6 = 1
 T230_L0_f4_le: const_Overall_Cond + 999994 t230_leaf0 <= 1e+06
 T230_L0_f33_le: x_Garage_Area + 99747 t230_leaf0 <= 100000
 T230_L0_f297_le: const_Sale_Condition_Normal + 999999 t230_leaf0 <= 1e+06
 T230_L1_f4_le: const_Overall_Cond + 999994 t230_leaf1 <= 1e+06
 T230_L1_f33_le: x_Garage_Area + 99747 t230_leaf1 <= 100000
 T230_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t230_leaf1
   >= -1e+06
 T230_L2_f4_le: const_Overall_Cond + 999994 t230_leaf2 <= 1e+06
 T230_R2_f33_ge: x_Garage_Area - 253 t230_leaf2 >= 0
 T230_L2_f3_le: Overall_Qual_calc + 999997 t230_leaf2 <= 1e+06
 T230_L3_f4_le: const_Overall_Cond + 999994 t230_leaf3 <= 1e+06
 T230_R3_f33_ge: x_Garage_Area - 253 t230_leaf3 >= 0
 T230_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t230_leaf3 >= -1e+06
 T230_R4_f4_ge: const_Overall_Cond - 1.000006e+06 t230_leaf4 >= -1e+06
 T230_L4_f78_le: const_Land_Contour_HLS + 999999 t230_leaf4 <= 1e+06
 T230_L4_f9_le: x_Exter_Cond + t230_leaf4 <= 4
 T230_R5_f4_ge: const_Overall_Cond - 1.000006e+06 t230_leaf5 >= -1e+06
 T230_L5_f78_le: const_Land_Contour_HLS + 999999 t230_leaf5 <= 1e+06
 T230_R5_f9_ge: x_Exter_Cond - 4 t230_leaf5 >= -1
 T230_R6_f4_ge: const_Overall_Cond - 1.000006e+06 t230_leaf6 >= -1e+06
 T230_R6_f78_ge: const_Land_Contour_HLS - 1.000001e+06 t230_leaf6
   >= -1e+06
 TREE_231_ONEHOT: t231_leaf0 + t231_leaf1 + t231_leaf2 + t231_leaf3
   + t231_leaf4 + t231_leaf5 + t231_leaf6 + t231_leaf7 = 1
 T231_L0_f1_le: const_Lot_Area + 980478 t231_leaf0 <= 1e+06
 T231_L0_f4_le: const_Overall_Cond + 999994 t231_leaf0 <= 1e+06
 T231_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t231_leaf0
   <= 1e+06
 T231_L1_f1_le: const_Lot_Area + 980478 t231_leaf1 <= 1e+06
 T231_L1_f4_le: const_Overall_Cond + 999994 t231_leaf1 <= 1e+06
 T231_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t231_leaf1
   >= -1e+06
 T231_L2_f1_le: const_Lot_Area + 980478 t231_leaf2 <= 1e+06
 T231_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t231_leaf2 >= -1e+06
 T231_L2_f1_le: const_Lot_Area + 992600 t231_leaf2 <= 1e+06
 T231_L3_f1_le: const_Lot_Area + 980478 t231_leaf3 <= 1e+06
 T231_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t231_leaf3 >= -1e+06
 T231_R3_f1_ge: const_Lot_Area - 1.0074e+06 t231_leaf3 >= -1e+06
 T231_R4_f1_ge: const_Lot_Area - 1.019522e+06 t231_leaf4 >= -1e+06
 T231_L4_f213_le: x_b1_is_GLQ <= 1
 T231_L4_f197_le: x_mvt_is_No_aplica <= 1
 T231_R5_f1_ge: const_Lot_Area - 1.019522e+06 t231_leaf5 >= -1e+06
 T231_L5_f213_le: x_b1_is_GLQ <= 1
 T231_R5_f197_ge: x_mvt_is_No_aplica - t231_leaf5 >= 0
 T231_R6_f1_ge: const_Lot_Area - 1.019522e+06 t231_leaf6 >= -1e+06
 T231_R6_f213_ge: x_b1_is_GLQ - t231_leaf6 >= 0
 T231_L6_f37_le: x_Open_Porch_SF + 99928 t231_leaf6 <= 100000
 T231_R7_f1_ge: const_Lot_Area - 1.019522e+06 t231_leaf7 >= -1e+06
 T231_R7_f213_ge: x_b1_is_GLQ - t231_leaf7 >= 0
 T231_R7_f37_ge: x_Open_Porch_SF - 72 t231_leaf7 >= 0
 TREE_232_ONEHOT: t232_leaf0 + t232_leaf1 + t232_leaf2 + t232_leaf3
   + t232_leaf4 = 1
 T232_L0_f37_le: x_Open_Porch_SF + 99709 t232_leaf0 <= 100000
 T232_L0_f17_le: x_1st_Flr_SF + 2800 t232_leaf0 <= 5000
 T232_L0_f164_le: x_ex1_is_BrkFace <= 1
 T232_L1_f37_le: x_Open_Porch_SF + 99709 t232_leaf1 <= 100000
 T232_L1_f17_le: x_1st_Flr_SF + 2800 t232_leaf1 <= 5000
 T232_R1_f164_ge: x_ex1_is_BrkFace - t232_leaf1 >= 0
 T232_L2_f37_le: x_Open_Porch_SF + 99709 t232_leaf2 <= 100000
 T232_R2_f17_ge: x_1st_Flr_SF - 2200 t232_leaf2 >= 0
 T232_R3_f37_ge: x_Open_Porch_SF - 291 t232_leaf3 >= 0
 T232_L3_f12_le: x_BsmtFin_SF_1 + 999603 t232_leaf3 <= 1e+06
 T232_R4_f37_ge: x_Open_Porch_SF - 291 t232_leaf4 >= 0
 T232_R4_f12_ge: x_BsmtFin_SF_1 - 397 t232_leaf4 >= 0
 TREE_233_ONEHOT: t233_leaf0 + t233_leaf1 + t233_leaf2 + t233_leaf3
   + t233_leaf4 + t233_leaf5 + t233_leaf6 = 1
 T233_L0_f37_le: x_Open_Porch_SF + 99850 t233_leaf0 <= 100000
 T233_L0_f164_le: x_ex1_is_BrkFace <= 1
 T233_L0_f4_le: const_Overall_Cond + 999992 t233_leaf0 <= 1e+06
 T233_L1_f37_le: x_Open_Porch_SF + 99850 t233_leaf1 <= 100000
 T233_L1_f164_le: x_ex1_is_BrkFace <= 1
 T233_R1_f4_ge: const_Overall_Cond - 1.000008e+06 t233_leaf1 >= -1e+06
 T233_L2_f37_le: x_Open_Porch_SF + 99850 t233_leaf2 <= 100000
 T233_R2_f164_ge: x_ex1_is_BrkFace - t233_leaf2 >= 0
 T233_L2_f118_le: const_Condition_1_Feedr + 999999 t233_leaf2 <= 1e+06
 T233_L3_f37_le: x_Open_Porch_SF + 99850 t233_leaf3 <= 100000
 T233_R3_f164_ge: x_ex1_is_BrkFace - t233_leaf3 >= 0
 T233_R3_f118_ge: const_Condition_1_Feedr - 1.000001e+06 t233_leaf3
   >= -1e+06
 T233_R4_f37_ge: x_Open_Porch_SF - 150 t233_leaf4 >= 0
 T233_L4_f13_le: x_BsmtFin_SF_2 + 99509 t233_leaf4 <= 100000
 T233_L4_f20_le: x_Gr_Liv_Area + 7607 t233_leaf4 <= 10000
 T233_R5_f37_ge: x_Open_Porch_SF - 150 t233_leaf5 >= 0
 T233_L5_f13_le: x_BsmtFin_SF_2 + 99509 t233_leaf5 <= 100000
 T233_R5_f20_ge: x_Gr_Liv_Area - 2393 t233_leaf5 >= 0
 T233_R6_f37_ge: x_Open_Porch_SF - 150 t233_leaf6 >= 0
 T233_R6_f13_ge: x_BsmtFin_SF_2 - 491 t233_leaf6 >= 0
 TREE_234_ONEHOT: t234_leaf0 + t234_leaf1 + t234_leaf2 + t234_leaf3
   + t234_leaf4 + t234_leaf5 = 1
 T234_L0_f3_le: Overall_Qual_calc + 999996 t234_leaf0 <= 1e+06
 T234_L0_f15_le: const_Total_Bsmt_SF + 999284 t234_leaf0 <= 1e+06
 T234_L1_f3_le: Overall_Qual_calc + 999996 t234_leaf1 <= 1e+06
 T234_R1_f15_ge: const_Total_Bsmt_SF - 1.000716e+06 t234_leaf1 >= -1e+06
 T234_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t234_leaf2 >= -1e+06
 T234_L2_f12_le: x_BsmtFin_SF_1 + 998268 t234_leaf2 <= 1e+06
 T234_L2_f112_le: const_Neighborhood_SawyerW + 999999 t234_leaf2 <= 1e+06
 T234_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t234_leaf3 >= -1e+06
 T234_L3_f12_le: x_BsmtFin_SF_1 + 998268 t234_leaf3 <= 1e+06
 T234_R3_f112_ge: const_Neighborhood_SawyerW - 1.000001e+06 t234_leaf3
   >= -1e+06
 T234_R4_f3_ge: Overall_Qual_calc - 1.000004e+06 t234_leaf4 >= -1e+06
 T234_R4_f12_ge: x_BsmtFin_SF_1 - 1732 t234_leaf4 >= 0
 T234_L4_f36_le: x_Wood_Deck_SF + 99828 t234_leaf4 <= 100000
 T234_R5_f3_ge: Overall_Qual_calc - 1.000004e+06 t234_leaf5 >= -1e+06
 T234_R5_f12_ge: x_BsmtFin_SF_1 - 1732 t234_leaf5 >= 0
 T234_R5_f36_ge: x_Wood_Deck_SF - 172 t234_leaf5 >= 0
 TREE_235_ONEHOT: t235_leaf0 + t235_leaf1 + t235_leaf2 + t235_leaf3
   + t235_leaf4 = 1
 T235_L0_f17_le: x_1st_Flr_SF + 2485 t235_leaf0 <= 5000
 T235_L0_f20_le: x_Gr_Liv_Area + 9232 t235_leaf0 <= 10000
 T235_L0_f66_le: const_MS_Zoning_RL + 999999 t235_leaf0 <= 1e+06
 T235_L1_f17_le: x_1st_Flr_SF + 2485 t235_leaf1 <= 5000
 T235_L1_f20_le: x_Gr_Liv_Area + 9232 t235_leaf1 <= 10000
 T235_R1_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t235_leaf1 >= -1e+06
 T235_L2_f17_le: x_1st_Flr_SF + 2485 t235_leaf2 <= 5000
 T235_R2_f20_ge: x_Gr_Liv_Area - 768 t235_leaf2 >= 0
 T235_L2_f7_le: x_Mas_Vnr_Area + 1327 t235_leaf2 <= 2000
 T235_L3_f17_le: x_1st_Flr_SF + 2485 t235_leaf3 <= 5000
 T235_R3_f20_ge: x_Gr_Liv_Area - 768 t235_leaf3 >= 0
 T235_R3_f7_ge: x_Mas_Vnr_Area - 673 t235_leaf3 >= 0
 T235_R4_f17_ge: x_1st_Flr_SF - 2515 t235_leaf4 >= 0
 TREE_236_ONEHOT: t236_leaf0 + t236_leaf1 + t236_leaf2 + t236_leaf3
   + t236_leaf4 + t236_leaf5 = 1
 T236_L0_f4_le: const_Overall_Cond + 999993 t236_leaf0 <= 1e+06
 T236_L0_f82_le: const_Lot_Config_CulDSac + 999999 t236_leaf0 <= 1e+06
 T236_L0_f63_le: const_MS_Zoning_FV + 999999 t236_leaf0 <= 1e+06
 T236_L1_f4_le: const_Overall_Cond + 999993 t236_leaf1 <= 1e+06
 T236_L1_f82_le: const_Lot_Config_CulDSac + 999999 t236_leaf1 <= 1e+06
 T236_R1_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t236_leaf1 >= -1e+06
 T236_L2_f4_le: const_Overall_Cond + 999993 t236_leaf2 <= 1e+06
 T236_R2_f82_ge: const_Lot_Config_CulDSac - 1.000001e+06 t236_leaf2
   >= -1e+06
 T236_R3_f4_ge: const_Overall_Cond - 1.000007e+06 t236_leaf3 >= -1e+06
 T236_L3_f31_le: const_Garage_Yr_Blt + 998022 t236_leaf3 <= 1e+06
 T236_L3_f1_le: const_Lot_Area + 990262 t236_leaf3 <= 1e+06
 T236_R4_f4_ge: const_Overall_Cond - 1.000007e+06 t236_leaf4 >= -1e+06
 T236_L4_f31_le: const_Garage_Yr_Blt + 998022 t236_leaf4 <= 1e+06
 T236_R4_f1_ge: const_Lot_Area - 1.009738e+06 t236_leaf4 >= -1e+06
 T236_R5_f4_ge: const_Overall_Cond - 1.000007e+06 t236_leaf5 >= -1e+06
 T236_R5_f31_ge: const_Garage_Yr_Blt - 1.001978e+06 t236_leaf5 >= -1e+06
 TREE_237_ONEHOT: t237_leaf0 + t237_leaf1 + t237_leaf2 + t237_leaf3 = 1
 T237_L0_f40_le: x_Screen_Porch + 99896 t237_leaf0 <= 100000
 T237_L0_f3_le: Overall_Qual_calc + 999990 t237_leaf0 <= 1e+06
 T237_L0_f3_le: Overall_Qual_calc + 999996 t237_leaf0 <= 1e+06
 T237_L1_f40_le: x_Screen_Porch + 99896 t237_leaf1 <= 100000
 T237_L1_f3_le: Overall_Qual_calc + 999990 t237_leaf1 <= 1e+06
 T237_R1_f3_ge: Overall_Qual_calc - 1.000004e+06 t237_leaf1 >= -1e+06
 T237_L2_f40_le: x_Screen_Porch + 99896 t237_leaf2 <= 100000
 T237_R2_f3_ge: Overall_Qual_calc - 1.00001e+06 t237_leaf2 >= -1e+06
 T237_R3_f40_ge: x_Screen_Porch - 104 t237_leaf3 >= 0
 TREE_238_ONEHOT: t238_leaf0 + t238_leaf1 + t238_leaf2 + t238_leaf3
   + t238_leaf4 + t238_leaf5 = 1
 T238_L0_f0_le: const_Lot_Frontage + 999880 t238_leaf0 <= 1e+06
 T238_L0_f1_le: const_Lot_Area + 997240 t238_leaf0 <= 1e+06
 T238_L1_f0_le: const_Lot_Frontage + 999880 t238_leaf1 <= 1e+06
 T238_R1_f1_ge: const_Lot_Area - 1.00276e+06 t238_leaf1 >= -1e+06
 T238_L1_f233_le: x_elect_is_FuseA <= 1
 T238_L2_f0_le: const_Lot_Frontage + 999880 t238_leaf2 <= 1e+06
 T238_R2_f1_ge: const_Lot_Area - 1.00276e+06 t238_leaf2 >= -1e+06
 T238_R2_f233_ge: x_elect_is_FuseA - t238_leaf2 >= 0
 T238_R3_f0_ge: const_Lot_Frontage - 1.00012e+06 t238_leaf3 >= -1e+06
 T238_L3_f8_le: x_Exter_Qual + t238_leaf3 <= 4
 T238_R4_f0_ge: const_Lot_Frontage - 1.00012e+06 t238_leaf4 >= -1e+06
 T238_R4_f8_ge: x_Exter_Qual - 4 t238_leaf4 >= -1
 T238_L4_f36_le: x_Wood_Deck_SF + 99850 t238_leaf4 <= 100000
 T238_R5_f0_ge: const_Lot_Frontage - 1.00012e+06 t238_leaf5 >= -1e+06
 T238_R5_f8_ge: x_Exter_Qual - 4 t238_leaf5 >= -1
 T238_R5_f36_ge: x_Wood_Deck_SF - 150 t238_leaf5 >= 0
 TREE_239_ONEHOT: t239_leaf0 + t239_leaf1 + t239_leaf2 + t239_leaf3
   + t239_leaf4 + t239_leaf5 + t239_leaf6 = 1
 T239_L0_f207_le: const_Bsmt_Exposure_Gd + 999999 t239_leaf0 <= 1e+06
 T239_L0_f4_le: const_Overall_Cond + 999994 t239_leaf0 <= 1e+06
 T239_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t239_leaf0
   <= 1e+06
 T239_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t239_leaf1 <= 1e+06
 T239_L1_f4_le: const_Overall_Cond + 999994 t239_leaf1 <= 1e+06
 T239_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t239_leaf1
   >= -1e+06
 T239_L2_f207_le: const_Bsmt_Exposure_Gd + 999999 t239_leaf2 <= 1e+06
 T239_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t239_leaf2 >= -1e+06
 T239_L2_f119_le: const_Condition_1_Norm + 999999 t239_leaf2 <= 1e+06
 T239_L3_f207_le: const_Bsmt_Exposure_Gd + 999999 t239_leaf3 <= 1e+06
 T239_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t239_leaf3 >= -1e+06
 T239_R3_f119_ge: const_Condition_1_Norm - 1.000001e+06 t239_leaf3
   >= -1e+06
 T239_R4_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t239_leaf4
   >= -1e+06
 T239_L4_f33_le: x_Garage_Area + 99370 t239_leaf4 <= 100000
 T239_R5_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t239_leaf5
   >= -1e+06
 T239_R5_f33_ge: x_Garage_Area - 630 t239_leaf5 >= 0
 T239_L5_f0_le: const_Lot_Frontage + 999880 t239_leaf5 <= 1e+06
 T239_R6_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t239_leaf6
   >= -1e+06
 T239_R6_f33_ge: x_Garage_Area - 630 t239_leaf6 >= 0
 T239_R6_f0_ge: const_Lot_Frontage - 1.00012e+06 t239_leaf6 >= -1e+06
 TREE_240_ONEHOT: t240_leaf0 + t240_leaf1 + t240_leaf2 + t240_leaf3
   + t240_leaf4 + t240_leaf5 = 1
 T240_L0_f23_le: x_Full_Bath + t240_leaf0 <= 4
 T240_L0_f1_le: const_Lot_Area + 996325 t240_leaf0 <= 1e+06
 T240_L1_f23_le: x_Full_Bath + t240_leaf1 <= 4
 T240_R1_f1_ge: const_Lot_Area - 1.003675e+06 t240_leaf1 >= -1e+06
 T240_L1_f17_le: x_1st_Flr_SF + 4169 t240_leaf1 <= 5000
 T240_L2_f23_le: x_Full_Bath + t240_leaf2 <= 4
 T240_R2_f1_ge: const_Lot_Area - 1.003675e+06 t240_leaf2 >= -1e+06
 T240_R2_f17_ge: x_1st_Flr_SF - 831 t240_leaf2 >= 0
 T240_R3_f23_ge: x_Full_Bath - 3 t240_leaf3 >= 0
 T240_L3_f197_le: x_mvt_is_No_aplica <= 1
 T240_L3_f7_le: x_Mas_Vnr_Area + 1812 t240_leaf3 <= 2000
 T240_R4_f23_ge: x_Full_Bath - 3 t240_leaf4 >= 0
 T240_L4_f197_le: x_mvt_is_No_aplica <= 1
 T240_R4_f7_ge: x_Mas_Vnr_Area - 188 t240_leaf4 >= 0
 T240_R5_f23_ge: x_Full_Bath - 3 t240_leaf5 >= 0
 T240_R5_f197_ge: x_mvt_is_No_aplica - t240_leaf5 >= 0
 TREE_241_ONEHOT: t241_leaf0 + t241_leaf1 + t241_leaf2 + t241_leaf3
   + t241_leaf4 = 1
 T241_L0_f3_le: Overall_Qual_calc + 999997 t241_leaf0 <= 1e+06
 T241_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t241_leaf1 >= -1e+06
 T241_L1_f119_le: const_Condition_1_Norm + 999999 t241_leaf1 <= 1e+06
 T241_L1_f271_le: const_Mo_Sold_1 + 999999 t241_leaf1 <= 1e+06
 T241_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t241_leaf2 >= -1e+06
 T241_L2_f119_le: const_Condition_1_Norm + 999999 t241_leaf2 <= 1e+06
 T241_R2_f271_ge: const_Mo_Sold_1 - 1.000001e+06 t241_leaf2 >= -1e+06
 T241_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t241_leaf3 >= -1e+06
 T241_R3_f119_ge: const_Condition_1_Norm - 1.000001e+06 t241_leaf3
   >= -1e+06
 T241_L3_f21_le: const_Bsmt_Full_Bath + 999998 t241_leaf3 <= 1e+06
 T241_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t241_leaf4 >= -1e+06
 T241_R4_f119_ge: const_Condition_1_Norm - 1.000001e+06 t241_leaf4
   >= -1e+06
 T241_R4_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t241_leaf4 >= -1e+06
 TREE_242_ONEHOT: t242_leaf0 + t242_leaf1 + t242_leaf2 + t242_leaf3
   + t242_leaf4 = 1
 T242_L0_f37_le: x_Open_Porch_SF + 99805 t242_leaf0 <= 100000
 T242_L0_f44_le: const_Yr_Sold + 997993 t242_leaf0 <= 1e+06
 T242_L0_f5_le: const_Year_Built + 998050 t242_leaf0 <= 1e+06
 T242_L1_f37_le: x_Open_Porch_SF + 99805 t242_leaf1 <= 100000
 T242_L1_f44_le: const_Yr_Sold + 997993 t242_leaf1 <= 1e+06
 T242_R1_f5_ge: const_Year_Built - 1.00195e+06 t242_leaf1 >= -1e+06
 T242_L2_f37_le: x_Open_Porch_SF + 99805 t242_leaf2 <= 100000
 T242_R2_f44_ge: const_Yr_Sold - 1.002007e+06 t242_leaf2 >= -1e+06
 T242_L2_f1_le: const_Lot_Area + 995500 t242_leaf2 <= 1e+06
 T242_L3_f37_le: x_Open_Porch_SF + 99805 t242_leaf3 <= 100000
 T242_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t242_leaf3 >= -1e+06
 T242_R3_f1_ge: const_Lot_Area - 1.0045e+06 t242_leaf3 >= -1e+06
 T242_R4_f37_ge: x_Open_Porch_SF - 195 t242_leaf4 >= 0
 TREE_243_ONEHOT: t243_leaf0 + t243_leaf1 + t243_leaf2 + t243_leaf3
   + t243_leaf4 = 1
 T243_L0_f15_le: const_Total_Bsmt_SF + 997508 t243_leaf0 <= 1e+06
 T243_L0_f1_le: const_Lot_Area + 980478 t243_leaf0 <= 1e+06
 T243_L0_f272_le: const_Mo_Sold_2 + 999999 t243_leaf0 <= 1e+06
 T243_L1_f15_le: const_Total_Bsmt_SF + 997508 t243_leaf1 <= 1e+06
 T243_L1_f1_le: const_Lot_Area + 980478 t243_leaf1 <= 1e+06
 T243_R1_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t243_leaf1 >= -1e+06
 T243_L2_f15_le: const_Total_Bsmt_SF + 997508 t243_leaf2 <= 1e+06
 T243_R2_f1_ge: const_Lot_Area - 1.019522e+06 t243_leaf2 >= -1e+06
 T243_L2_f195_le: x_mvt_is_BrkFace <= 1
 T243_L3_f15_le: const_Total_Bsmt_SF + 997508 t243_leaf3 <= 1e+06
 T243_R3_f1_ge: const_Lot_Area - 1.019522e+06 t243_leaf3 >= -1e+06
 T243_R3_f195_ge: x_mvt_is_BrkFace - t243_leaf3 >= 0
 T243_R4_f15_ge: const_Total_Bsmt_SF - 1.002492e+06 t243_leaf4 >= -1e+06
 TREE_244_ONEHOT: t244_leaf0 + t244_leaf1 + t244_leaf2 + t244_leaf3 = 1
 T244_L0_f17_le: x_1st_Flr_SF + 2485 t244_leaf0 <= 5000
 T244_L0_f68_le: const_Street_Grvl + 999999 t244_leaf0 <= 1e+06
 T244_L0_f18_le: const_2nd_Flr_SF + 998371 t244_leaf0 <= 1e+06
 T244_L1_f17_le: x_1st_Flr_SF + 2485 t244_leaf1 <= 5000
 T244_L1_f68_le: const_Street_Grvl + 999999 t244_leaf1 <= 1e+06
 T244_R1_f18_ge: const_2nd_Flr_SF - 1.001629e+06 t244_leaf1 >= -1e+06
 T244_L2_f17_le: x_1st_Flr_SF + 2485 t244_leaf2 <= 5000
 T244_R2_f68_ge: const_Street_Grvl - 1.000001e+06 t244_leaf2 >= -1e+06
 T244_R3_f17_ge: x_1st_Flr_SF - 2515 t244_leaf3 >= 0
 TREE_245_ONEHOT: t245_leaf0 + t245_leaf1 + t245_leaf2 + t245_leaf3
   + t245_leaf4 + t245_leaf5 + t245_leaf6 = 1
 T245_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t245_leaf0 <= 1e+06
 T245_L0_f36_le: x_Wood_Deck_SF + 99978 t245_leaf0 <= 100000
 T245_L0_f38_le: x_Enclosed_Porch + 99909 t245_leaf0 <= 100000
 T245_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t245_leaf1 <= 1e+06
 T245_L1_f36_le: x_Wood_Deck_SF + 99978 t245_leaf1 <= 100000
 T245_R1_f38_ge: x_Enclosed_Porch - 91 t245_leaf1 >= 0
 T245_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t245_leaf2 <= 1e+06
 T245_R2_f36_ge: x_Wood_Deck_SF - 22 t245_leaf2 >= 0
 T245_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t245_leaf3
   >= -1e+06
 T245_L3_f4_le: const_Overall_Cond + 999994 t245_leaf3 <= 1e+06
 T245_L3_f63_le: const_MS_Zoning_FV + 999999 t245_leaf3 <= 1e+06
 T245_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t245_leaf4
   >= -1e+06
 T245_L4_f4_le: const_Overall_Cond + 999994 t245_leaf4 <= 1e+06
 T245_R4_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t245_leaf4 >= -1e+06
 T245_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t245_leaf5
   >= -1e+06
 T245_R5_f4_ge: const_Overall_Cond - 1.000006e+06 t245_leaf5 >= -1e+06
 T245_L5_f17_le: x_1st_Flr_SF + 3642 t245_leaf5 <= 5000
 T245_R6_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t245_leaf6
   >= -1e+06
 T245_R6_f4_ge: const_Overall_Cond - 1.000006e+06 t245_leaf6 >= -1e+06
 T245_R6_f17_ge: x_1st_Flr_SF - 1358 t245_leaf6 >= 0
 TREE_246_ONEHOT: t246_leaf0 + t246_leaf1 + t246_leaf2 + t246_leaf3 = 1
 T246_L0_f20_le: x_Gr_Liv_Area + 6914 t246_leaf0 <= 10000
 T246_L0_f1_le: const_Lot_Area + 996000 t246_leaf0 <= 1e+06
 T246_L1_f20_le: x_Gr_Liv_Area + 6914 t246_leaf1 <= 10000
 T246_R1_f1_ge: const_Lot_Area - 1.004e+06 t246_leaf1 >= -1e+06
 T246_L1_f29_le: const_Fireplaces + 999999 t246_leaf1 <= 1e+06
 T246_L2_f20_le: x_Gr_Liv_Area + 6914 t246_leaf2 <= 10000
 T246_R2_f1_ge: const_Lot_Area - 1.004e+06 t246_leaf2 >= -1e+06
 T246_R2_f29_ge: const_Fireplaces - 1.000001e+06 t246_leaf2 >= -1e+06
 T246_R3_f20_ge: x_Gr_Liv_Area - 3086 t246_leaf3 >= 0
 TREE_247_ONEHOT: t247_leaf0 + t247_leaf1 + t247_leaf2 + t247_leaf3
   + t247_leaf4 = 1
 T247_L0_f27_le: x_Kitchen_Qual + 2 t247_leaf0 <= 4
 T247_L0_f233_le: x_elect_is_FuseA <= 1
 T247_L1_f27_le: x_Kitchen_Qual + 2 t247_leaf1 <= 4
 T247_R1_f233_ge: x_elect_is_FuseA - t247_leaf1 >= 0
 T247_R2_f27_ge: x_Kitchen_Qual - 3 t247_leaf2 >= -1
 T247_L2_f33_le: x_Garage_Area + 99129 t247_leaf2 <= 100000
 T247_L2_f34_le: x_Garage_Qual + 2 t247_leaf2 <= 4
 T247_R3_f27_ge: x_Kitchen_Qual - 3 t247_leaf3 >= -1
 T247_L3_f33_le: x_Garage_Area + 99129 t247_leaf3 <= 100000
 T247_R3_f34_ge: x_Garage_Qual - 3 t247_leaf3 >= -1
 T247_R4_f27_ge: x_Kitchen_Qual - 3 t247_leaf4 >= -1
 T247_R4_f33_ge: x_Garage_Area - 871 t247_leaf4 >= 0
 TREE_248_ONEHOT: t248_leaf0 + t248_leaf1 + t248_leaf2 + t248_leaf3
   + t248_leaf4 + t248_leaf5 + t248_leaf6 + t248_leaf7 = 1
 T248_L0_f20_le: x_Gr_Liv_Area + 9012 t248_leaf0 <= 10000
 T248_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t248_leaf0
   <= 1e+06
 T248_L0_f38_le: x_Enclosed_Porch + 99879 t248_leaf0 <= 100000
 T248_L1_f20_le: x_Gr_Liv_Area + 9012 t248_leaf1 <= 10000
 T248_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t248_leaf1
   <= 1e+06
 T248_R1_f38_ge: x_Enclosed_Porch - 121 t248_leaf1 >= 0
 T248_L2_f20_le: x_Gr_Liv_Area + 9012 t248_leaf2 <= 10000
 T248_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t248_leaf2
   >= -1e+06
 T248_L2_f35_le: x_Garage_Cond + 2 t248_leaf2 <= 4
 T248_L3_f20_le: x_Gr_Liv_Area + 9012 t248_leaf3 <= 10000
 T248_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t248_leaf3
   >= -1e+06
 T248_R3_f35_ge: x_Garage_Cond - 3 t248_leaf3 >= -1
 T248_R4_f20_ge: x_Gr_Liv_Area - 988 t248_leaf4 >= 0
 T248_L4_f3_le: Overall_Qual_calc + 999993 t248_leaf4 <= 1e+06
 T248_L4_f18_le: const_2nd_Flr_SF + 999328 t248_leaf4 <= 1e+06
 T248_R5_f20_ge: x_Gr_Liv_Area - 988 t248_leaf5 >= 0
 T248_L5_f3_le: Overall_Qual_calc + 999993 t248_leaf5 <= 1e+06
 T248_R5_f18_ge: const_2nd_Flr_SF - 1.000672e+06 t248_leaf5 >= -1e+06
 T248_R6_f20_ge: x_Gr_Liv_Area - 988 t248_leaf6 >= 0
 T248_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t248_leaf6 >= -1e+06
 T248_L6_f3_le: Overall_Qual_calc + 999991 t248_leaf6 <= 1e+06
 T248_R7_f20_ge: x_Gr_Liv_Area - 988 t248_leaf7 >= 0
 T248_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t248_leaf7 >= -1e+06
 T248_R7_f3_ge: Overall_Qual_calc - 1.000009e+06 t248_leaf7 >= -1e+06
 TREE_249_ONEHOT: t249_leaf0 + t249_leaf1 + t249_leaf2 + t249_leaf3
   + t249_leaf4 = 1
 T249_L0_f7_le: x_Mas_Vnr_Area + 1078 t249_leaf0 <= 2000
 T249_L0_f44_le: const_Yr_Sold + 997992 t249_leaf0 <= 1e+06
 T249_L0_f20_le: x_Gr_Liv_Area + 7206 t249_leaf0 <= 10000
 T249_L1_f7_le: x_Mas_Vnr_Area + 1078 t249_leaf1 <= 2000
 T249_L1_f44_le: const_Yr_Sold + 997992 t249_leaf1 <= 1e+06
 T249_R1_f20_ge: x_Gr_Liv_Area - 2794 t249_leaf1 >= 0
 T249_L2_f7_le: x_Mas_Vnr_Area + 1078 t249_leaf2 <= 2000
 T249_R2_f44_ge: const_Yr_Sold - 1.002008e+06 t249_leaf2 >= -1e+06
 T249_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t249_leaf2
   <= 1e+06
 T249_L3_f7_le: x_Mas_Vnr_Area + 1078 t249_leaf3 <= 2000
 T249_R3_f44_ge: const_Yr_Sold - 1.002008e+06 t249_leaf3 >= -1e+06
 T249_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t249_leaf3
   >= -1e+06
 T249_R4_f7_ge: x_Mas_Vnr_Area - 922 t249_leaf4 >= 0
 TREE_250_ONEHOT: t250_leaf0 + t250_leaf1 + t250_leaf2 + t250_leaf3 = 1
 T250_L0_f14_le: x_Bsmt_Unf_SF + 99298 t250_leaf0 <= 100000
 T250_L0_f17_le: x_1st_Flr_SF + 2800 t250_leaf0 <= 5000
 T250_L0_f17_le: x_1st_Flr_SF + 3435 t250_leaf0 <= 5000
 T250_L1_f14_le: x_Bsmt_Unf_SF + 99298 t250_leaf1 <= 100000
 T250_L1_f17_le: x_1st_Flr_SF + 2800 t250_leaf1 <= 5000
 T250_R1_f17_ge: x_1st_Flr_SF - 1565 t250_leaf1 >= 0
 T250_L2_f14_le: x_Bsmt_Unf_SF + 99298 t250_leaf2 <= 100000
 T250_R2_f17_ge: x_1st_Flr_SF - 2200 t250_leaf2 >= 0
 T250_R3_f14_ge: x_Bsmt_Unf_SF - 702 t250_leaf3 >= 0
 TREE_251_ONEHOT: t251_leaf0 + t251_leaf1 + t251_leaf2 + t251_leaf3
   + t251_leaf4 = 1
 T251_L0_f62_le: const_MS_Zoning_C_(all) + 999999 t251_leaf0 <= 1e+06
 T251_L0_f51_le: const_MS_SubClass_70 + 999999 t251_leaf0 <= 1e+06
 T251_L0_f3_le: Overall_Qual_calc + 999995 t251_leaf0 <= 1e+06
 T251_L1_f62_le: const_MS_Zoning_C_(all) + 999999 t251_leaf1 <= 1e+06
 T251_L1_f51_le: const_MS_SubClass_70 + 999999 t251_leaf1 <= 1e+06
 T251_R1_f3_ge: Overall_Qual_calc - 1.000005e+06 t251_leaf1 >= -1e+06
 T251_L2_f62_le: const_MS_Zoning_C_(all) + 999999 t251_leaf2 <= 1e+06
 T251_R2_f51_ge: const_MS_SubClass_70 - 1.000001e+06 t251_leaf2 >= -1e+06
 T251_L2_f5_le: const_Year_Built + 998108 t251_leaf2 <= 1e+06
 T251_L3_f62_le: const_MS_Zoning_C_(all) + 999999 t251_leaf3 <= 1e+06
 T251_R3_f51_ge: const_MS_SubClass_70 - 1.000001e+06 t251_leaf3 >= -1e+06
 T251_R3_f5_ge: const_Year_Built - 1.001892e+06 t251_leaf3 >= -1e+06
 T251_R4_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t251_leaf4
   >= -1e+06
 TREE_252_ONEHOT: t252_leaf0 + t252_leaf1 + t252_leaf2 + t252_leaf3
   + t252_leaf4 + t252_leaf5 + t252_leaf6 + t252_leaf7 = 1
 T252_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t252_leaf0 <= 1e+06
 T252_L0_f38_le: x_Enclosed_Porch + 99909 t252_leaf0 <= 100000
 T252_L0_f36_le: x_Wood_Deck_SF + 99978 t252_leaf0 <= 100000
 T252_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t252_leaf1 <= 1e+06
 T252_L1_f38_le: x_Enclosed_Porch + 99909 t252_leaf1 <= 100000
 T252_R1_f36_ge: x_Wood_Deck_SF - 22 t252_leaf1 >= 0
 T252_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t252_leaf2 <= 1e+06
 T252_R2_f38_ge: x_Enclosed_Porch - 91 t252_leaf2 >= 0
 T252_L2_f169_le: x_ex1_is_MetalSd <= 1
 T252_L3_f245_le: const_FunctioNo_aplical_Typ + 999999 t252_leaf3 <= 1e+06
 T252_R3_f38_ge: x_Enclosed_Porch - 91 t252_leaf3 >= 0
 T252_R3_f169_ge: x_ex1_is_MetalSd - t252_leaf3 >= 0
 T252_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t252_leaf4
   >= -1e+06
 T252_L4_f296_le: const_Sale_Condition_Family + 999999 t252_leaf4 <= 1e+06
 T252_L4_f92_le: const_Neighborhood_BrkSide + 999999 t252_leaf4 <= 1e+06
 T252_R5_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t252_leaf5
   >= -1e+06
 T252_L5_f296_le: const_Sale_Condition_Family + 999999 t252_leaf5 <= 1e+06
 T252_R5_f92_ge: const_Neighborhood_BrkSide - 1.000001e+06 t252_leaf5
   >= -1e+06
 T252_R6_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t252_leaf6
   >= -1e+06
 T252_R6_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t252_leaf6
   >= -1e+06
 T252_L6_f33_le: x_Garage_Area + 99494 t252_leaf6 <= 100000
 T252_R7_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t252_leaf7
   >= -1e+06
 T252_R7_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t252_leaf7
   >= -1e+06
 T252_R7_f33_ge: x_Garage_Area - 506 t252_leaf7 >= 0
 TREE_253_ONEHOT: t253_leaf0 + t253_leaf1 + t253_leaf2 + t253_leaf3
   + t253_leaf4 + t253_leaf5 = 1
 T253_L0_f1_le: const_Lot_Area + 976743 t253_leaf0 <= 1e+06
 T253_L0_f1_le: const_Lot_Area + 981614 t253_leaf0 <= 1e+06
 T253_L0_f72_le: const_Alley_Pave + 999999 t253_leaf0 <= 1e+06
 T253_L1_f1_le: const_Lot_Area + 976743 t253_leaf1 <= 1e+06
 T253_L1_f1_le: const_Lot_Area + 981614 t253_leaf1 <= 1e+06
 T253_R1_f72_ge: const_Alley_Pave - 1.000001e+06 t253_leaf1 >= -1e+06
 T253_L2_f1_le: const_Lot_Area + 976743 t253_leaf2 <= 1e+06
 T253_R2_f1_ge: const_Lot_Area - 1.018386e+06 t253_leaf2 >= -1e+06
 T253_R3_f1_ge: const_Lot_Area - 1.023257e+06 t253_leaf3 >= -1e+06
 T253_L3_f20_le: x_Gr_Liv_Area + 7456 t253_leaf3 <= 10000
 T253_L3_f20_le: x_Gr_Liv_Area + 8509 t253_leaf3 <= 10000
 T253_R4_f1_ge: const_Lot_Area - 1.023257e+06 t253_leaf4 >= -1e+06
 T253_L4_f20_le: x_Gr_Liv_Area + 7456 t253_leaf4 <= 10000
 T253_R4_f20_ge: x_Gr_Liv_Area - 1491 t253_leaf4 >= 0
 T253_R5_f1_ge: const_Lot_Area - 1.023257e+06 t253_leaf5 >= -1e+06
 T253_R5_f20_ge: x_Gr_Liv_Area - 2544 t253_leaf5 >= 0
 TREE_254_ONEHOT: t254_leaf0 + t254_leaf1 + t254_leaf2 + t254_leaf3 = 1
 T254_L0_f3_le: Overall_Qual_calc + 999997 t254_leaf0 <= 1e+06
 T254_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t254_leaf1 >= -1e+06
 T254_L1_f63_le: const_MS_Zoning_FV + 999999 t254_leaf1 <= 1e+06
 T254_L1_f37_le: x_Open_Porch_SF + 99709 t254_leaf1 <= 100000
 T254_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t254_leaf2 >= -1e+06
 T254_L2_f63_le: const_MS_Zoning_FV + 999999 t254_leaf2 <= 1e+06
 T254_R2_f37_ge: x_Open_Porch_SF - 291 t254_leaf2 >= 0
 T254_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t254_leaf3 >= -1e+06
 T254_R3_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t254_leaf3 >= -1e+06
 TREE_255_ONEHOT: t255_leaf0 + t255_leaf1 + t255_leaf2 + t255_leaf3
   + t255_leaf4 + t255_leaf5 + t255_leaf6 = 1
 T255_L0_f161_le: x_ex1_is_AsbShng <= 1
 T255_L0_f25_le: x_Bedroom_AbvGr + 3 t255_leaf0 <= 6
 T255_L0_f14_le: x_Bsmt_Unf_SF + 98490 t255_leaf0 <= 100000
 T255_L1_f161_le: x_ex1_is_AsbShng <= 1
 T255_L1_f25_le: x_Bedroom_AbvGr + 3 t255_leaf1 <= 6
 T255_R1_f14_ge: x_Bsmt_Unf_SF - 1510 t255_leaf1 >= 0
 T255_L2_f161_le: x_ex1_is_AsbShng <= 1
 T255_R2_f25_ge: x_Bedroom_AbvGr - 3 t255_leaf2 >= 0
 T255_L2_f17_le: x_1st_Flr_SF + 3598 t255_leaf2 <= 5000
 T255_L3_f161_le: x_ex1_is_AsbShng <= 1
 T255_R3_f25_ge: x_Bedroom_AbvGr - 3 t255_leaf3 >= 0
 T255_R3_f17_ge: x_1st_Flr_SF - 1402 t255_leaf3 >= 0
 T255_R4_f161_ge: x_ex1_is_AsbShng - t255_leaf4 >= 0
 T255_L4_f31_le: const_Garage_Yr_Blt + 998077 t255_leaf4 <= 1e+06
 T255_R5_f161_ge: x_ex1_is_AsbShng - t255_leaf5 >= 0
 T255_R5_f31_ge: const_Garage_Yr_Blt - 1.001923e+06 t255_leaf5 >= -1e+06
 T255_L5_f1_le: const_Lot_Area + 990395 t255_leaf5 <= 1e+06
 T255_R6_f161_ge: x_ex1_is_AsbShng - t255_leaf6 >= 0
 T255_R6_f31_ge: const_Garage_Yr_Blt - 1.001923e+06 t255_leaf6 >= -1e+06
 T255_R6_f1_ge: const_Lot_Area - 1.009605e+06 t255_leaf6 >= -1e+06
 TREE_256_ONEHOT: t256_leaf0 + t256_leaf1 + t256_leaf2 + t256_leaf3
   + t256_leaf4 + t256_leaf5 + t256_leaf6 = 1
 T256_L0_f46_le: const_MS_SubClass_30 + 999999 t256_leaf0 <= 1e+06
 T256_L0_f86_le: const_Land_Slope_Gtl + 999999 t256_leaf0 <= 1e+06
 T256_L1_f46_le: const_MS_SubClass_30 + 999999 t256_leaf1 <= 1e+06
 T256_R1_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t256_leaf1 >= -1e+06
 T256_L1_f5_le: const_Year_Built + 998089 t256_leaf1 <= 1e+06
 T256_L2_f46_le: const_MS_SubClass_30 + 999999 t256_leaf2 <= 1e+06
 T256_R2_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t256_leaf2 >= -1e+06
 T256_R2_f5_ge: const_Year_Built - 1.001911e+06 t256_leaf2 >= -1e+06
 T256_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t256_leaf3 >= -1e+06
 T256_L3_f33_le: x_Garage_Area + 99503 t256_leaf3 <= 100000
 T256_L3_f1_le: const_Lot_Area + 991906 t256_leaf3 <= 1e+06
 T256_R4_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t256_leaf4 >= -1e+06
 T256_L4_f33_le: x_Garage_Area + 99503 t256_leaf4 <= 100000
 T256_R4_f1_ge: const_Lot_Area - 1.008094e+06 t256_leaf4 >= -1e+06
 T256_R5_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t256_leaf5 >= -1e+06
 T256_R5_f33_ge: x_Garage_Area - 497 t256_leaf5 >= 0
 T256_L5_f37_le: x_Open_Porch_SF + 99996 t256_leaf5 <= 100000
 T256_R6_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t256_leaf6 >= -1e+06
 T256_R6_f33_ge: x_Garage_Area - 497 t256_leaf6 >= 0
 T256_R6_f37_ge: x_Open_Porch_SF - 4 t256_leaf6 >= 0
 TREE_257_ONEHOT: t257_leaf0 + t257_leaf1 + t257_leaf2 + t257_leaf3
   + t257_leaf4 + t257_leaf5 + t257_leaf6 = 1
 T257_L0_f20_le: x_Gr_Liv_Area + 9012 t257_leaf0 <= 10000
 T257_L0_f33_le: x_Garage_Area + 99784 t257_leaf0 <= 100000
 T257_L0_f231_le: central_air_no + 999999 t257_leaf0 <= 1e+06
 T257_L1_f20_le: x_Gr_Liv_Area + 9012 t257_leaf1 <= 10000
 T257_L1_f33_le: x_Garage_Area + 99784 t257_leaf1 <= 100000
 T257_R1_f231_ge: central_air_no - 1.000001e+06 t257_leaf1 >= -1e+06
 T257_L2_f20_le: x_Gr_Liv_Area + 9012 t257_leaf2 <= 10000
 T257_R2_f33_ge: x_Garage_Area - 216 t257_leaf2 >= 0
 T257_L2_f9_le: x_Exter_Cond + 2 t257_leaf2 <= 4
 T257_L3_f20_le: x_Gr_Liv_Area + 9012 t257_leaf3 <= 10000
 T257_R3_f33_ge: x_Garage_Area - 216 t257_leaf3 >= 0
 T257_R3_f9_ge: x_Exter_Cond - 3 t257_leaf3 >= -1
 T257_R4_f20_ge: x_Gr_Liv_Area - 988 t257_leaf4 >= 0
 T257_L4_f4_le: const_Overall_Cond + 999996 t257_leaf4 <= 1e+06
 T257_R5_f20_ge: x_Gr_Liv_Area - 988 t257_leaf5 >= 0
 T257_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t257_leaf5 >= -1e+06
 T257_L5_f7_le: x_Mas_Vnr_Area + 1327 t257_leaf5 <= 2000
 T257_R6_f20_ge: x_Gr_Liv_Area - 988 t257_leaf6 >= 0
 T257_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t257_leaf6 >= -1e+06
 T257_R6_f7_ge: x_Mas_Vnr_Area - 673 t257_leaf6 >= 0
 TREE_258_ONEHOT: t258_leaf0 + t258_leaf1 + t258_leaf2 + t258_leaf3
   + t258_leaf4 = 1
 T258_L0_f5_le: const_Year_Built + 998086 t258_leaf0 <= 1e+06
 T258_R1_f5_ge: const_Year_Built - 1.001914e+06 t258_leaf1 >= -1e+06
 T258_L1_f231_le: central_air_no + 999999 t258_leaf1 <= 1e+06
 T258_L1_f63_le: const_MS_Zoning_FV + 999999 t258_leaf1 <= 1e+06
 T258_R2_f5_ge: const_Year_Built - 1.001914e+06 t258_leaf2 >= -1e+06
 T258_L2_f231_le: central_air_no + 999999 t258_leaf2 <= 1e+06
 T258_R2_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t258_leaf2 >= -1e+06
 T258_R3_f5_ge: const_Year_Built - 1.001914e+06 t258_leaf3 >= -1e+06
 T258_R3_f231_ge: central_air_no - 1.000001e+06 t258_leaf3 >= -1e+06
 T258_L3_f185_le: x_ex2_is_MetalSd <= 1
 T258_R4_f5_ge: const_Year_Built - 1.001914e+06 t258_leaf4 >= -1e+06
 T258_R4_f231_ge: central_air_no - 1.000001e+06 t258_leaf4 >= -1e+06
 T258_R4_f185_ge: x_ex2_is_MetalSd - t258_leaf4 >= 0
 TREE_259_ONEHOT: t259_leaf0 + t259_leaf1 + t259_leaf2 + t259_leaf3
   + t259_leaf4 = 1
 T259_L0_f18_le: const_2nd_Flr_SF + 998704 t259_leaf0 <= 1e+06
 T259_L0_f4_le: const_Overall_Cond + 999994 t259_leaf0 <= 1e+06
 T259_L0_f16_le: x_Heating_QC + t259_leaf0 <= 4
 T259_L1_f18_le: const_2nd_Flr_SF + 998704 t259_leaf1 <= 1e+06
 T259_L1_f4_le: const_Overall_Cond + 999994 t259_leaf1 <= 1e+06
 T259_R1_f16_ge: x_Heating_QC - 4 t259_leaf1 >= -1
 T259_L2_f18_le: const_2nd_Flr_SF + 998704 t259_leaf2 <= 1e+06
 T259_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t259_leaf2 >= -1e+06
 T259_L2_f18_le: const_2nd_Flr_SF + 999833 t259_leaf2 <= 1e+06
 T259_L3_f18_le: const_2nd_Flr_SF + 998704 t259_leaf3 <= 1e+06
 T259_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t259_leaf3 >= -1e+06
 T259_R3_f18_ge: const_2nd_Flr_SF - 1.000167e+06 t259_leaf3 >= -1e+06
 T259_R4_f18_ge: const_2nd_Flr_SF - 1.001296e+06 t259_leaf4 >= -1e+06
 TREE_260_ONEHOT: t260_leaf0 + t260_leaf1 + t260_leaf2 + t260_leaf3
   + t260_leaf4 + t260_leaf5 = 1
 T260_L0_f0_le: const_Lot_Frontage + 999879 t260_leaf0 <= 1e+06
 T260_L0_f20_le: x_Gr_Liv_Area + 9106 t260_leaf0 <= 10000
 T260_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t260_leaf0
   <= 1e+06
 T260_L1_f0_le: const_Lot_Frontage + 999879 t260_leaf1 <= 1e+06
 T260_L1_f20_le: x_Gr_Liv_Area + 9106 t260_leaf1 <= 10000
 T260_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t260_leaf1
   >= -1e+06
 T260_L2_f0_le: const_Lot_Frontage + 999879 t260_leaf2 <= 1e+06
 T260_R2_f20_ge: x_Gr_Liv_Area - 894 t260_leaf2 >= 0
 T260_L2_f194_le: x_mvt_is_BrkCmn <= 1
 T260_L3_f0_le: const_Lot_Frontage + 999879 t260_leaf3 <= 1e+06
 T260_R3_f20_ge: x_Gr_Liv_Area - 894 t260_leaf3 >= 0
 T260_R3_f194_ge: x_mvt_is_BrkCmn - t260_leaf3 >= 0
 T260_R4_f0_ge: const_Lot_Frontage - 1.000121e+06 t260_leaf4 >= -1e+06
 T260_L4_f14_le: x_Bsmt_Unf_SF + 99552 t260_leaf4 <= 100000
 T260_R5_f0_ge: const_Lot_Frontage - 1.000121e+06 t260_leaf5 >= -1e+06
 T260_R5_f14_ge: x_Bsmt_Unf_SF - 448 t260_leaf5 >= 0
 TREE_261_ONEHOT: t261_leaf0 + t261_leaf1 + t261_leaf2 + t261_leaf3 = 1
 T261_L0_f17_le: x_1st_Flr_SF + 2485 t261_leaf0 <= 5000
 T261_L0_f1_le: const_Lot_Area + 972303 t261_leaf0 <= 1e+06
 T261_L0_f46_le: const_MS_SubClass_30 + 999999 t261_leaf0 <= 1e+06
 T261_L1_f17_le: x_1st_Flr_SF + 2485 t261_leaf1 <= 5000
 T261_L1_f1_le: const_Lot_Area + 972303 t261_leaf1 <= 1e+06
 T261_R1_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t261_leaf1 >= -1e+06
 T261_L2_f17_le: x_1st_Flr_SF + 2485 t261_leaf2 <= 5000
 T261_R2_f1_ge: const_Lot_Area - 1.027697e+06 t261_leaf2 >= -1e+06
 T261_R3_f17_ge: x_1st_Flr_SF - 2515 t261_leaf3 >= 0
 TREE_262_ONEHOT: t262_leaf0 + t262_leaf1 + t262_leaf2 + t262_leaf3
   + t262_leaf4 = 1
 T262_L0_f3_le: Overall_Qual_calc + 999997 t262_leaf0 <= 1e+06
 T262_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t262_leaf1 >= -1e+06
 T262_L1_f26_le: x_Kitchen_AbvGr + t262_leaf1 <= 3
 T262_L1_f20_le: x_Gr_Liv_Area + 7351 t262_leaf1 <= 10000
 T262_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t262_leaf2 >= -1e+06
 T262_L2_f26_le: x_Kitchen_AbvGr + t262_leaf2 <= 3
 T262_R2_f20_ge: x_Gr_Liv_Area - 2649 t262_leaf2 >= 0
 T262_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t262_leaf3 >= -1e+06
 T262_R3_f26_ge: x_Kitchen_AbvGr - 2 t262_leaf3 >= 0
 T262_L3_f35_le: x_Garage_Cond + 3 t262_leaf3 <= 4
 T262_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t262_leaf4 >= -1e+06
 T262_R4_f26_ge: x_Kitchen_AbvGr - 2 t262_leaf4 >= 0
 T262_R4_f35_ge: x_Garage_Cond - 2 t262_leaf4 >= -1
 TREE_263_ONEHOT: t263_leaf0 + t263_leaf1 + t263_leaf2 + t263_leaf3
   + t263_leaf4 + t263_leaf5 = 1
 T263_L0_f272_le: const_Mo_Sold_2 + 999999 t263_leaf0 <= 1e+06
 T263_L0_f95_le: const_Neighborhood_Crawfor + 999999 t263_leaf0 <= 1e+06
 T263_L0_f20_le: x_Gr_Liv_Area + 9190 t263_leaf0 <= 10000
 T263_L1_f272_le: const_Mo_Sold_2 + 999999 t263_leaf1 <= 1e+06
 T263_L1_f95_le: const_Neighborhood_Crawfor + 999999 t263_leaf1 <= 1e+06
 T263_R1_f20_ge: x_Gr_Liv_Area - 810 t263_leaf1 >= 0
 T263_L2_f272_le: const_Mo_Sold_2 + 999999 t263_leaf2 <= 1e+06
 T263_R2_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t263_leaf2
   >= -1e+06
 T263_R3_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t263_leaf3 >= -1e+06
 T263_L3_f1_le: const_Lot_Area + 990035 t263_leaf3 <= 1e+06
 T263_R4_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t263_leaf4 >= -1e+06
 T263_R4_f1_ge: const_Lot_Area - 1.009965e+06 t263_leaf4 >= -1e+06
 T263_L4_f297_le: const_Sale_Condition_Normal + 999999 t263_leaf4 <= 1e+06
 T263_R5_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t263_leaf5 >= -1e+06
 T263_R5_f1_ge: const_Lot_Area - 1.009965e+06 t263_leaf5 >= -1e+06
 T263_R5_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t263_leaf5
   >= -1e+06
 TREE_264_ONEHOT: t264_leaf0 + t264_leaf1 + t264_leaf2 + t264_leaf3
   + t264_leaf4 = 1
 T264_L0_f3_le: Overall_Qual_calc + 999997 t264_leaf0 <= 1e+06
 T264_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t264_leaf1 >= -1e+06
 T264_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t264_leaf1 <= 1e+06
 T264_L1_f37_le: x_Open_Porch_SF + 99818 t264_leaf1 <= 100000
 T264_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t264_leaf2 >= -1e+06
 T264_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t264_leaf2 <= 1e+06
 T264_R2_f37_ge: x_Open_Porch_SF - 182 t264_leaf2 >= 0
 T264_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t264_leaf3 >= -1e+06
 T264_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t264_leaf3
   >= -1e+06
 T264_L3_f296_le: const_Sale_Condition_Family + 999999 t264_leaf3 <= 1e+06
 T264_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t264_leaf4 >= -1e+06
 T264_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t264_leaf4
   >= -1e+06
 T264_R4_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t264_leaf4
   >= -1e+06
 TREE_265_ONEHOT: t265_leaf0 + t265_leaf1 + t265_leaf2 + t265_leaf3
   + t265_leaf4 = 1
 T265_L0_f0_le: const_Lot_Frontage + 999848 t265_leaf0 <= 1e+06
 T265_L0_f3_le: Overall_Qual_calc + 999995 t265_leaf0 <= 1e+06
 T265_L0_f161_le: x_ex1_is_AsbShng <= 1
 T265_L1_f0_le: const_Lot_Frontage + 999848 t265_leaf1 <= 1e+06
 T265_L1_f3_le: Overall_Qual_calc + 999995 t265_leaf1 <= 1e+06
 T265_R1_f161_ge: x_ex1_is_AsbShng - t265_leaf1 >= 0
 T265_L2_f0_le: const_Lot_Frontage + 999848 t265_leaf2 <= 1e+06
 T265_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t265_leaf2 >= -1e+06
 T265_L2_f5_le: const_Year_Built + 998087 t265_leaf2 <= 1e+06
 T265_L3_f0_le: const_Lot_Frontage + 999848 t265_leaf3 <= 1e+06
 T265_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t265_leaf3 >= -1e+06
 T265_R3_f5_ge: const_Year_Built - 1.001913e+06 t265_leaf3 >= -1e+06
 T265_R4_f0_ge: const_Lot_Frontage - 1.000152e+06 t265_leaf4 >= -1e+06
 TREE_266_ONEHOT: t266_leaf0 + t266_leaf1 + t266_leaf2 + t266_leaf3
   + t266_leaf4 + t266_leaf5 + t266_leaf6 = 1
 T266_L0_f207_le: const_Bsmt_Exposure_Gd + 999999 t266_leaf0 <= 1e+06
 T266_L0_f161_le: x_ex1_is_AsbShng <= 1
 T266_L0_f33_le: x_Garage_Area + 99189 t266_leaf0 <= 100000
 T266_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t266_leaf1 <= 1e+06
 T266_L1_f161_le: x_ex1_is_AsbShng <= 1
 T266_R1_f33_ge: x_Garage_Area - 811 t266_leaf1 >= 0
 T266_L2_f207_le: const_Bsmt_Exposure_Gd + 999999 t266_leaf2 <= 1e+06
 T266_R2_f161_ge: x_ex1_is_AsbShng - t266_leaf2 >= 0
 T266_L2_f15_le: const_Total_Bsmt_SF + 999273 t266_leaf2 <= 1e+06
 T266_L3_f207_le: const_Bsmt_Exposure_Gd + 999999 t266_leaf3 <= 1e+06
 T266_R3_f161_ge: x_ex1_is_AsbShng - t266_leaf3 >= 0
 T266_R3_f15_ge: const_Total_Bsmt_SF - 1.000727e+06 t266_leaf3 >= -1e+06
 T266_R4_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t266_leaf4
   >= -1e+06
 T266_L4_f36_le: x_Wood_Deck_SF + 99489 t266_leaf4 <= 100000
 T266_L4_f29_le: const_Fireplaces + 999998 t266_leaf4 <= 1e+06
 T266_R5_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t266_leaf5
   >= -1e+06
 T266_L5_f36_le: x_Wood_Deck_SF + 99489 t266_leaf5 <= 100000
 T266_R5_f29_ge: const_Fireplaces - 1.000002e+06 t266_leaf5 >= -1e+06
 T266_R6_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t266_leaf6
   >= -1e+06
 T266_R6_f36_ge: x_Wood_Deck_SF - 511 t266_leaf6 >= 0
 TREE_267_ONEHOT: t267_leaf0 + t267_leaf1 + t267_leaf2 + t267_leaf3
   + t267_leaf4 + t267_leaf5 = 1
 T267_L0_f3_le: Overall_Qual_calc + 999993 t267_leaf0 <= 1e+06
 T267_L0_f18_le: const_2nd_Flr_SF + 998724 t267_leaf0 <= 1e+06
 T267_L0_f37_le: x_Open_Porch_SF + 99979 t267_leaf0 <= 100000
 T267_L1_f3_le: Overall_Qual_calc + 999993 t267_leaf1 <= 1e+06
 T267_L1_f18_le: const_2nd_Flr_SF + 998724 t267_leaf1 <= 1e+06
 T267_R1_f37_ge: x_Open_Porch_SF - 21 t267_leaf1 >= 0
 T267_L2_f3_le: Overall_Qual_calc + 999993 t267_leaf2 <= 1e+06
 T267_R2_f18_ge: const_2nd_Flr_SF - 1.001276e+06 t267_leaf2 >= -1e+06
 T267_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t267_leaf3 >= -1e+06
 T267_L3_f296_le: const_Sale_Condition_Family + 999999 t267_leaf3 <= 1e+06
 T267_L3_f0_le: const_Lot_Frontage + 999886 t267_leaf3 <= 1e+06
 T267_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t267_leaf4 >= -1e+06
 T267_L4_f296_le: const_Sale_Condition_Family + 999999 t267_leaf4 <= 1e+06
 T267_R4_f0_ge: const_Lot_Frontage - 1.000114e+06 t267_leaf4 >= -1e+06
 T267_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t267_leaf5 >= -1e+06
 T267_R5_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t267_leaf5
   >= -1e+06
 TREE_268_ONEHOT: t268_leaf0 + t268_leaf1 + t268_leaf2 + t268_leaf3
   + t268_leaf4 + t268_leaf5 = 1
 T268_L0_f37_le: x_Open_Porch_SF + 99718 t268_leaf0 <= 100000
 T268_L0_f29_le: const_Fireplaces + 999998 t268_leaf0 <= 1e+06
 T268_L0_f185_le: x_ex2_is_MetalSd <= 1
 T268_L1_f37_le: x_Open_Porch_SF + 99718 t268_leaf1 <= 100000
 T268_L1_f29_le: const_Fireplaces + 999998 t268_leaf1 <= 1e+06
 T268_R1_f185_ge: x_ex2_is_MetalSd - t268_leaf1 >= 0
 T268_L2_f37_le: x_Open_Porch_SF + 99718 t268_leaf2 <= 100000
 T268_R2_f29_ge: const_Fireplaces - 1.000002e+06 t268_leaf2 >= -1e+06
 T268_L2_f37_le: x_Open_Porch_SF + 99969 t268_leaf2 <= 100000
 T268_L3_f37_le: x_Open_Porch_SF + 99718 t268_leaf3 <= 100000
 T268_R3_f29_ge: const_Fireplaces - 1.000002e+06 t268_leaf3 >= -1e+06
 T268_R3_f37_ge: x_Open_Porch_SF - 31 t268_leaf3 >= 0
 T268_R4_f37_ge: x_Open_Porch_SF - 282 t268_leaf4 >= 0
 T268_L4_f256_le: const_Garage_Finish_Unf + 999999 t268_leaf4 <= 1e+06
 T268_R5_f37_ge: x_Open_Porch_SF - 282 t268_leaf5 >= 0
 T268_R5_f256_ge: const_Garage_Finish_Unf - 1.000001e+06 t268_leaf5
   >= -1e+06
 TREE_269_ONEHOT: t269_leaf0 + t269_leaf1 + t269_leaf2 + t269_leaf3
   + t269_leaf4 = 1
 T269_L0_f18_le: const_2nd_Flr_SF + 998389 t269_leaf0 <= 1e+06
 T269_L0_f0_le: const_Lot_Frontage + 999883 t269_leaf0 <= 1e+06
 T269_L0_f31_le: const_Garage_Yr_Blt + 997992 t269_leaf0 <= 1e+06
 T269_L1_f18_le: const_2nd_Flr_SF + 998389 t269_leaf1 <= 1e+06
 T269_L1_f0_le: const_Lot_Frontage + 999883 t269_leaf1 <= 1e+06
 T269_R1_f31_ge: const_Garage_Yr_Blt - 1.002008e+06 t269_leaf1 >= -1e+06
 T269_L2_f18_le: const_2nd_Flr_SF + 998389 t269_leaf2 <= 1e+06
 T269_R2_f0_ge: const_Lot_Frontage - 1.000117e+06 t269_leaf2 >= -1e+06
 T269_L2_f31_le: const_Garage_Yr_Blt + 998034 t269_leaf2 <= 1e+06
 T269_L3_f18_le: const_2nd_Flr_SF + 998389 t269_leaf3 <= 1e+06
 T269_R3_f0_ge: const_Lot_Frontage - 1.000117e+06 t269_leaf3 >= -1e+06
 T269_R3_f31_ge: const_Garage_Yr_Blt - 1.001966e+06 t269_leaf3 >= -1e+06
 T269_R4_f18_ge: const_2nd_Flr_SF - 1.001611e+06 t269_leaf4 >= -1e+06
 TREE_270_ONEHOT: t270_leaf0 + t270_leaf1 + t270_leaf2 + t270_leaf3
   + t270_leaf4 + t270_leaf5 = 1
 T270_L0_f12_le: x_BsmtFin_SF_1 + 998268 t270_leaf0 <= 1e+06
 T270_L0_f9_le: x_Exter_Cond + 2 t270_leaf0 <= 4
 T270_L0_f6_le: const_Year_Remod_Add + 998048 t270_leaf0 <= 1e+06
 T270_L1_f12_le: x_BsmtFin_SF_1 + 998268 t270_leaf1 <= 1e+06
 T270_L1_f9_le: x_Exter_Cond + 2 t270_leaf1 <= 4
 T270_R1_f6_ge: const_Year_Remod_Add - 1.001952e+06 t270_leaf1 >= -1e+06
 T270_L2_f12_le: x_BsmtFin_SF_1 + 998268 t270_leaf2 <= 1e+06
 T270_R2_f9_ge: x_Exter_Cond - 3 t270_leaf2 >= -1
 T270_L2_f250_le: const_Garage_Type_CarPort + 999999 t270_leaf2 <= 1e+06
 T270_L3_f12_le: x_BsmtFin_SF_1 + 998268 t270_leaf3 <= 1e+06
 T270_R3_f9_ge: x_Exter_Cond - 3 t270_leaf3 >= -1
 T270_R3_f250_ge: const_Garage_Type_CarPort - 1.000001e+06 t270_leaf3
   >= -1e+06
 T270_R4_f12_ge: x_BsmtFin_SF_1 - 1732 t270_leaf4 >= 0
 T270_L4_f7_le: x_Mas_Vnr_Area + 1433 t270_leaf4 <= 2000
 T270_R5_f12_ge: x_BsmtFin_SF_1 - 1732 t270_leaf5 >= 0
 T270_R5_f7_ge: x_Mas_Vnr_Area - 567 t270_leaf5 >= 0
 TREE_271_ONEHOT: t271_leaf0 + t271_leaf1 + t271_leaf2 + t271_leaf3
   + t271_leaf4 + t271_leaf5 = 1
 T271_L0_f7_le: x_Mas_Vnr_Area + 1212 t271_leaf0 <= 2000
 T271_L0_f3_le: Overall_Qual_calc + 999996 t271_leaf0 <= 1e+06
 T271_L0_f14_le: x_Bsmt_Unf_SF + 99734 t271_leaf0 <= 100000
 T271_L1_f7_le: x_Mas_Vnr_Area + 1212 t271_leaf1 <= 2000
 T271_L1_f3_le: Overall_Qual_calc + 999996 t271_leaf1 <= 1e+06
 T271_R1_f14_ge: x_Bsmt_Unf_SF - 266 t271_leaf1 >= 0
 T271_L2_f7_le: x_Mas_Vnr_Area + 1212 t271_leaf2 <= 2000
 T271_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t271_leaf2 >= -1e+06
 T271_L2_f298_le: const_Sale_Condition_Partial + 999999 t271_leaf2
   <= 1e+06
 T271_L3_f7_le: x_Mas_Vnr_Area + 1212 t271_leaf3 <= 2000
 T271_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t271_leaf3 >= -1e+06
 T271_R3_f298_ge: const_Sale_Condition_Partial - 1.000001e+06 t271_leaf3
   >= -1e+06
 T271_R4_f7_ge: x_Mas_Vnr_Area - 788 t271_leaf4 >= 0
 T271_L4_f20_le: x_Gr_Liv_Area + 7310 t271_leaf4 <= 10000
 T271_R5_f7_ge: x_Mas_Vnr_Area - 788 t271_leaf5 >= 0
 T271_R5_f20_ge: x_Gr_Liv_Area - 2690 t271_leaf5 >= 0
 TREE_272_ONEHOT: t272_leaf0 + t272_leaf1 + t272_leaf2 + t272_leaf3
   + t272_leaf4 + t272_leaf5 + t272_leaf6 + t272_leaf7 = 1
 T272_L0_f96_le: const_Neighborhood_Edwards + 999999 t272_leaf0 <= 1e+06
 T272_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t272_leaf0
   <= 1e+06
 T272_L0_f3_le: Overall_Qual_calc + 999997 t272_leaf0 <= 1e+06
 T272_L1_f96_le: const_Neighborhood_Edwards + 999999 t272_leaf1 <= 1e+06
 T272_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t272_leaf1
   <= 1e+06
 T272_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t272_leaf1 >= -1e+06
 T272_L2_f96_le: const_Neighborhood_Edwards + 999999 t272_leaf2 <= 1e+06
 T272_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t272_leaf2
   >= -1e+06
 T272_L2_f14_le: x_Bsmt_Unf_SF + 99298 t272_leaf2 <= 100000
 T272_L3_f96_le: const_Neighborhood_Edwards + 999999 t272_leaf3 <= 1e+06
 T272_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t272_leaf3
   >= -1e+06
 T272_R3_f14_ge: x_Bsmt_Unf_SF - 702 t272_leaf3 >= 0
 T272_R4_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t272_leaf4
   >= -1e+06
 T272_L4_f7_le: x_Mas_Vnr_Area + 1997 t272_leaf4 <= 2000
 T272_L4_f1_le: const_Lot_Area + 983715 t272_leaf4 <= 1e+06
 T272_R5_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t272_leaf5
   >= -1e+06
 T272_L5_f7_le: x_Mas_Vnr_Area + 1997 t272_leaf5 <= 2000
 T272_R5_f1_ge: const_Lot_Area - 1.016285e+06 t272_leaf5 >= -1e+06
 T272_R6_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t272_leaf6
   >= -1e+06
 T272_R6_f7_ge: x_Mas_Vnr_Area - 3 t272_leaf6 >= 0
 T272_L6_f202_le: const_Foundation_PConc + 999999 t272_leaf6 <= 1e+06
 T272_R7_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t272_leaf7
   >= -1e+06
 T272_R7_f7_ge: x_Mas_Vnr_Area - 3 t272_leaf7 >= 0
 T272_R7_f202_ge: const_Foundation_PConc - 1.000001e+06 t272_leaf7
   >= -1e+06
 TREE_273_ONEHOT: t273_leaf0 + t273_leaf1 + t273_leaf2 + t273_leaf3
   + t273_leaf4 = 1
 T273_L0_f33_le: x_Garage_Area + 98846 t273_leaf0 <= 100000
 T273_L0_f44_le: const_Yr_Sold + 997991 t273_leaf0 <= 1e+06
 T273_L0_f24_le: x_Half_Bath + 3 t273_leaf0 <= 5
 T273_L1_f33_le: x_Garage_Area + 98846 t273_leaf1 <= 100000
 T273_L1_f44_le: const_Yr_Sold + 997991 t273_leaf1 <= 1e+06
 T273_R1_f24_ge: x_Half_Bath - 2 t273_leaf1 >= 0
 T273_L2_f33_le: x_Garage_Area + 98846 t273_leaf2 <= 100000
 T273_R2_f44_ge: const_Yr_Sold - 1.002009e+06 t273_leaf2 >= -1e+06
 T273_L2_f20_le: x_Gr_Liv_Area + 9184 t273_leaf2 <= 10000
 T273_L3_f33_le: x_Garage_Area + 98846 t273_leaf3 <= 100000
 T273_R3_f44_ge: const_Yr_Sold - 1.002009e+06 t273_leaf3 >= -1e+06
 T273_R3_f20_ge: x_Gr_Liv_Area - 816 t273_leaf3 >= 0
 T273_R4_f33_ge: x_Garage_Area - 1154 t273_leaf4 >= 0
 TREE_274_ONEHOT: t274_leaf0 + t274_leaf1 + t274_leaf2 + t274_leaf3
   + t274_leaf4 = 1
 T274_L0_f15_le: const_Total_Bsmt_SF + 997864 t274_leaf0 <= 1e+06
 T274_L0_f31_le: const_Garage_Yr_Blt + 998047 t274_leaf0 <= 1e+06
 T274_L0_f17_le: x_1st_Flr_SF + 3608 t274_leaf0 <= 5000
 T274_L1_f15_le: const_Total_Bsmt_SF + 997864 t274_leaf1 <= 1e+06
 T274_L1_f31_le: const_Garage_Yr_Blt + 998047 t274_leaf1 <= 1e+06
 T274_R1_f17_ge: x_1st_Flr_SF - 1392 t274_leaf1 >= 0
 T274_L2_f15_le: const_Total_Bsmt_SF + 997864 t274_leaf2 <= 1e+06
 T274_R2_f31_ge: const_Garage_Yr_Blt - 1.001953e+06 t274_leaf2 >= -1e+06
 T274_L2_f14_le: x_Bsmt_Unf_SF + 99524 t274_leaf2 <= 100000
 T274_L3_f15_le: const_Total_Bsmt_SF + 997864 t274_leaf3 <= 1e+06
 T274_R3_f31_ge: const_Garage_Yr_Blt - 1.001953e+06 t274_leaf3 >= -1e+06
 T274_R3_f14_ge: x_Bsmt_Unf_SF - 476 t274_leaf3 >= 0
 T274_R4_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t274_leaf4 >= -1e+06
 TREE_275_ONEHOT: t275_leaf0 + t275_leaf1 + t275_leaf2 + t275_leaf3
   + t275_leaf4 + t275_leaf5 = 1
 T275_L0_f20_le: x_Gr_Liv_Area + 9253 t275_leaf0 <= 10000
 T275_L0_f38_le: x_Enclosed_Porch + 99984 t275_leaf0 <= 100000
 T275_L1_f20_le: x_Gr_Liv_Area + 9253 t275_leaf1 <= 10000
 T275_R1_f38_ge: x_Enclosed_Porch - 16 t275_leaf1 >= 0
 T275_R2_f20_ge: x_Gr_Liv_Area - 747 t275_leaf2 >= 0
 T275_L2_f43_le: const_Misc_Val + 999946 t275_leaf2 <= 1e+06
 T275_L2_f119_le: const_Condition_1_Norm + 999999 t275_leaf2 <= 1e+06
 T275_R3_f20_ge: x_Gr_Liv_Area - 747 t275_leaf3 >= 0
 T275_L3_f43_le: const_Misc_Val + 999946 t275_leaf3 <= 1e+06
 T275_R3_f119_ge: const_Condition_1_Norm - 1.000001e+06 t275_leaf3
   >= -1e+06
 T275_R4_f20_ge: x_Gr_Liv_Area - 747 t275_leaf4 >= 0
 T275_R4_f43_ge: const_Misc_Val - 1.000054e+06 t275_leaf4 >= -1e+06
 T275_L4_f25_le: x_Bedroom_AbvGr + 3 t275_leaf4 <= 6
 T275_R5_f20_ge: x_Gr_Liv_Area - 747 t275_leaf5 >= 0
 T275_R5_f43_ge: const_Misc_Val - 1.000054e+06 t275_leaf5 >= -1e+06
 T275_R5_f25_ge: x_Bedroom_AbvGr - 3 t275_leaf5 >= 0
 TREE_276_ONEHOT: t276_leaf0 + t276_leaf1 + t276_leaf2 + t276_leaf3
   + t276_leaf4 = 1
 T276_L0_f1_le: const_Lot_Area + 996818 t276_leaf0 <= 1e+06
 T276_R1_f1_ge: const_Lot_Area - 1.003182e+06 t276_leaf1 >= -1e+06
 T276_L1_f190_le: x_ex2_is_Stucco <= 1
 T276_L1_f17_le: x_1st_Flr_SF + 3189 t276_leaf1 <= 5000
 T276_R2_f1_ge: const_Lot_Area - 1.003182e+06 t276_leaf2 >= -1e+06
 T276_L2_f190_le: x_ex2_is_Stucco <= 1
 T276_R2_f17_ge: x_1st_Flr_SF - 1811 t276_leaf2 >= 0
 T276_R3_f1_ge: const_Lot_Area - 1.003182e+06 t276_leaf3 >= -1e+06
 T276_R3_f190_ge: x_ex2_is_Stucco - t276_leaf3 >= 0
 T276_L3_f12_le: x_BsmtFin_SF_1 + 999862 t276_leaf3 <= 1e+06
 T276_R4_f1_ge: const_Lot_Area - 1.003182e+06 t276_leaf4 >= -1e+06
 T276_R4_f190_ge: x_ex2_is_Stucco - t276_leaf4 >= 0
 T276_R4_f12_ge: x_BsmtFin_SF_1 - 138 t276_leaf4 >= 0
 TREE_277_ONEHOT: t277_leaf0 + t277_leaf1 + t277_leaf2 + t277_leaf3
   + t277_leaf4 + t277_leaf5 + t277_leaf6 = 1
 T277_L0_f4_le: const_Overall_Cond + 999995 t277_leaf0 <= 1e+06
 T277_L0_f297_le: const_Sale_Condition_Normal + 999999 t277_leaf0 <= 1e+06
 T277_L0_f209_le: const_Bsmt_Exposure_No + 999999 t277_leaf0 <= 1e+06
 T277_L1_f4_le: const_Overall_Cond + 999995 t277_leaf1 <= 1e+06
 T277_L1_f297_le: const_Sale_Condition_Normal + 999999 t277_leaf1 <= 1e+06
 T277_R1_f209_ge: const_Bsmt_Exposure_No - 1.000001e+06 t277_leaf1
   >= -1e+06
 T277_L2_f4_le: const_Overall_Cond + 999995 t277_leaf2 <= 1e+06
 T277_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t277_leaf2
   >= -1e+06
 T277_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t277_leaf2 <= 1e+06
 T277_L3_f4_le: const_Overall_Cond + 999995 t277_leaf3 <= 1e+06
 T277_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t277_leaf3
   >= -1e+06
 T277_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t277_leaf3
   >= -1e+06
 T277_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t277_leaf4 >= -1e+06
 T277_L4_f0_le: const_Lot_Frontage + 999848 t277_leaf4 <= 1e+06
 T277_L4_f37_le: x_Open_Porch_SF + 99724 t277_leaf4 <= 100000
 T277_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t277_leaf5 >= -1e+06
 T277_L5_f0_le: const_Lot_Frontage + 999848 t277_leaf5 <= 1e+06
 T277_R5_f37_ge: x_Open_Porch_SF - 276 t277_leaf5 >= 0
 T277_R6_f4_ge: const_Overall_Cond - 1.000005e+06 t277_leaf6 >= -1e+06
 T277_R6_f0_ge: const_Lot_Frontage - 1.000152e+06 t277_leaf6 >= -1e+06
 TREE_278_ONEHOT: t278_leaf0 + t278_leaf1 + t278_leaf2 + t278_leaf3 = 1
 T278_L0_f164_le: x_ex1_is_BrkFace <= 1
 T278_L0_f3_le: Overall_Qual_calc + 999997 t278_leaf0 <= 1e+06
 T278_L1_f164_le: x_ex1_is_BrkFace <= 1
 T278_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t278_leaf1 >= -1e+06
 T278_L1_f277_le: const_Mo_Sold_7 + 999999 t278_leaf1 <= 1e+06
 T278_L2_f164_le: x_ex1_is_BrkFace <= 1
 T278_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t278_leaf2 >= -1e+06
 T278_R2_f277_ge: const_Mo_Sold_7 - 1.000001e+06 t278_leaf2 >= -1e+06
 T278_R3_f164_ge: x_ex1_is_BrkFace - t278_leaf3 >= 0
 TREE_279_ONEHOT: t279_leaf0 + t279_leaf1 + t279_leaf2 + t279_leaf3
   + t279_leaf4 + t279_leaf5 = 1
 T279_L0_f3_le: Overall_Qual_calc + 999996 t279_leaf0 <= 1e+06
 T279_L0_f259_le: x_paved_drive_is_Y <= 1
 T279_L1_f3_le: Overall_Qual_calc + 999996 t279_leaf1 <= 1e+06
 T279_R1_f259_ge: x_paved_drive_is_Y - t279_leaf1 >= 0
 T279_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t279_leaf2 >= -1e+06
 T279_L2_f6_le: const_Year_Remod_Add + 997993 t279_leaf2 <= 1e+06
 T279_L2_f45_le: const_MS_SubClass_20 + 999999 t279_leaf2 <= 1e+06
 T279_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t279_leaf3 >= -1e+06
 T279_L3_f6_le: const_Year_Remod_Add + 997993 t279_leaf3 <= 1e+06
 T279_R3_f45_ge: const_MS_SubClass_20 - 1.000001e+06 t279_leaf3 >= -1e+06
 T279_R4_f3_ge: Overall_Qual_calc - 1.000004e+06 t279_leaf4 >= -1e+06
 T279_R4_f6_ge: const_Year_Remod_Add - 1.002007e+06 t279_leaf4 >= -1e+06
 T279_L4_f0_le: const_Lot_Frontage + 999905 t279_leaf4 <= 1e+06
 T279_R5_f3_ge: Overall_Qual_calc - 1.000004e+06 t279_leaf5 >= -1e+06
 T279_R5_f6_ge: const_Year_Remod_Add - 1.002007e+06 t279_leaf5 >= -1e+06
 T279_R5_f0_ge: const_Lot_Frontage - 1.000095e+06 t279_leaf5 >= -1e+06
 TREE_280_ONEHOT: t280_leaf0 + t280_leaf1 + t280_leaf2 + t280_leaf3
   + t280_leaf4 + t280_leaf5 + t280_leaf6 = 1
 T280_L0_f4_le: const_Overall_Cond + 999996 t280_leaf0 <= 1e+06
 T280_L0_f14_le: x_Bsmt_Unf_SF + 99664 t280_leaf0 <= 100000
 T280_L1_f4_le: const_Overall_Cond + 999996 t280_leaf1 <= 1e+06
 T280_R1_f14_ge: x_Bsmt_Unf_SF - 336 t280_leaf1 >= 0
 T280_L1_f6_le: const_Year_Remod_Add + 998048 t280_leaf1 <= 1e+06
 T280_L2_f4_le: const_Overall_Cond + 999996 t280_leaf2 <= 1e+06
 T280_R2_f14_ge: x_Bsmt_Unf_SF - 336 t280_leaf2 >= 0
 T280_R2_f6_ge: const_Year_Remod_Add - 1.001952e+06 t280_leaf2 >= -1e+06
 T280_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t280_leaf3 >= -1e+06
 T280_L3_f117_le: const_Condition_1_Artery + 999999 t280_leaf3 <= 1e+06
 T280_L3_f29_le: const_Fireplaces + 999997 t280_leaf3 <= 1e+06
 T280_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t280_leaf4 >= -1e+06
 T280_L4_f117_le: const_Condition_1_Artery + 999999 t280_leaf4 <= 1e+06
 T280_R4_f29_ge: const_Fireplaces - 1.000003e+06 t280_leaf4 >= -1e+06
 T280_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t280_leaf5 >= -1e+06
 T280_R5_f117_ge: const_Condition_1_Artery - 1.000001e+06 t280_leaf5
   >= -1e+06
 T280_L5_f33_le: x_Garage_Area + 99196 t280_leaf5 <= 100000
 T280_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t280_leaf6 >= -1e+06
 T280_R6_f117_ge: const_Condition_1_Artery - 1.000001e+06 t280_leaf6
   >= -1e+06
 T280_R6_f33_ge: x_Garage_Area - 804 t280_leaf6 >= 0
 TREE_281_ONEHOT: t281_leaf0 + t281_leaf1 + t281_leaf2 + t281_leaf3
   + t281_leaf4 = 1
 T281_L0_f3_le: Overall_Qual_calc + 999997 t281_leaf0 <= 1e+06
 T281_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t281_leaf1 >= -1e+06
 T281_L1_f3_le: Overall_Qual_calc + 999993 t281_leaf1 <= 1e+06
 T281_L1_f1_le: const_Lot_Area + 992000 t281_leaf1 <= 1e+06
 T281_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t281_leaf2 >= -1e+06
 T281_L2_f3_le: Overall_Qual_calc + 999993 t281_leaf2 <= 1e+06
 T281_R2_f1_ge: const_Lot_Area - 1.008e+06 t281_leaf2 >= -1e+06
 T281_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t281_leaf3 >= -1e+06
 T281_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t281_leaf3 >= -1e+06
 T281_L3_f77_le: const_Land_Contour_Bnk + 999999 t281_leaf3 <= 1e+06
 T281_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t281_leaf4 >= -1e+06
 T281_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t281_leaf4 >= -1e+06
 T281_R4_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t281_leaf4
   >= -1e+06
 TREE_282_ONEHOT: t282_leaf0 + t282_leaf1 + t282_leaf2 + t282_leaf3 = 1
 T282_L0_f37_le: x_Open_Porch_SF + 99582 t282_leaf0 <= 100000
 T282_L0_f7_le: x_Mas_Vnr_Area + 1295 t282_leaf0 <= 2000
 T282_L0_f44_le: const_Yr_Sold + 997993 t282_leaf0 <= 1e+06
 T282_L1_f37_le: x_Open_Porch_SF + 99582 t282_leaf1 <= 100000
 T282_L1_f7_le: x_Mas_Vnr_Area + 1295 t282_leaf1 <= 2000
 T282_R1_f44_ge: const_Yr_Sold - 1.002007e+06 t282_leaf1 >= -1e+06
 T282_L2_f37_le: x_Open_Porch_SF + 99582 t282_leaf2 <= 100000
 T282_R2_f7_ge: x_Mas_Vnr_Area - 705 t282_leaf2 >= 0
 T282_R3_f37_ge: x_Open_Porch_SF - 418 t282_leaf3 >= 0
 TREE_283_ONEHOT: t283_leaf0 + t283_leaf1 + t283_leaf2 + t283_leaf3
   + t283_leaf4 + t283_leaf5 = 1
 T283_L0_f10_le: const_Bsmt_Qual + 1e+06 t283_leaf0 <= 1e+06
 T283_L0_f251_le: const_Garage_Type_Detchd + 999999 t283_leaf0 <= 1e+06
 T283_L1_f10_le: const_Bsmt_Qual + 1e+06 t283_leaf1 <= 1e+06
 T283_R1_f251_ge: const_Garage_Type_Detchd - 1.000001e+06 t283_leaf1
   >= -1e+06
 T283_R2_f10_ge: const_Bsmt_Qual - 1e+06 t283_leaf2 >= -1e+06
 T283_L2_f46_le: const_MS_SubClass_30 + 999999 t283_leaf2 <= 1e+06
 T283_L2_f31_le: const_Garage_Yr_Blt + 998059 t283_leaf2 <= 1e+06
 T283_R3_f10_ge: const_Bsmt_Qual - 1e+06 t283_leaf3 >= -1e+06
 T283_L3_f46_le: const_MS_SubClass_30 + 999999 t283_leaf3 <= 1e+06
 T283_R3_f31_ge: const_Garage_Yr_Blt - 1.001941e+06 t283_leaf3 >= -1e+06
 T283_R4_f10_ge: const_Bsmt_Qual - 1e+06 t283_leaf4 >= -1e+06
 T283_R4_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t283_leaf4 >= -1e+06
 T283_L4_f6_le: const_Year_Remod_Add + 998049 t283_leaf4 <= 1e+06
 T283_R5_f10_ge: const_Bsmt_Qual - 1e+06 t283_leaf5 >= -1e+06
 T283_R5_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t283_leaf5 >= -1e+06
 T283_R5_f6_ge: const_Year_Remod_Add - 1.001951e+06 t283_leaf5 >= -1e+06
 TREE_284_ONEHOT: t284_leaf0 + t284_leaf1 + t284_leaf2 + t284_leaf3
   + t284_leaf4 = 1
 T284_L0_f38_le: x_Enclosed_Porch + 99752 t284_leaf0 <= 100000
 T284_L0_f33_le: x_Garage_Area + 98846 t284_leaf0 <= 100000
 T284_L0_f15_le: const_Total_Bsmt_SF + 997864 t284_leaf0 <= 1e+06
 T284_L1_f38_le: x_Enclosed_Porch + 99752 t284_leaf1 <= 100000
 T284_L1_f33_le: x_Garage_Area + 98846 t284_leaf1 <= 100000
 T284_R1_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t284_leaf1 >= -1e+06
 T284_L2_f38_le: x_Enclosed_Porch + 99752 t284_leaf2 <= 100000
 T284_R2_f33_ge: x_Garage_Area - 1154 t284_leaf2 >= 0
 T284_L2_f10_le: const_Bsmt_Qual + 999996 t284_leaf2 <= 1e+06
 T284_L3_f38_le: x_Enclosed_Porch + 99752 t284_leaf3 <= 100000
 T284_R3_f33_ge: x_Garage_Area - 1154 t284_leaf3 >= 0
 T284_R3_f10_ge: const_Bsmt_Qual - 1.000004e+06 t284_leaf3 >= -1e+06
 T284_R4_f38_ge: x_Enclosed_Porch - 248 t284_leaf4 >= 0
 TREE_285_ONEHOT: t285_leaf0 + t285_leaf1 + t285_leaf2 + t285_leaf3
   + t285_leaf4 + t285_leaf5 = 1
 T285_L0_f28_le: x_TotRms_AbvGrd + 3 t285_leaf0 <= 15
 T285_L0_f16_le: x_Heating_QC + t285_leaf0 <= 4
 T285_L0_f297_le: const_Sale_Condition_Normal + 999999 t285_leaf0 <= 1e+06
 T285_L1_f28_le: x_TotRms_AbvGrd + 3 t285_leaf1 <= 15
 T285_L1_f16_le: x_Heating_QC + t285_leaf1 <= 4
 T285_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t285_leaf1
   >= -1e+06
 T285_L2_f28_le: x_TotRms_AbvGrd + 3 t285_leaf2 <= 15
 T285_R2_f16_ge: x_Heating_QC - 4 t285_leaf2 >= -1
 T285_L2_f117_le: const_Condition_1_Artery + 999999 t285_leaf2 <= 1e+06
 T285_L3_f28_le: x_TotRms_AbvGrd + 3 t285_leaf3 <= 15
 T285_R3_f16_ge: x_Heating_QC - 4 t285_leaf3 >= -1
 T285_R3_f117_ge: const_Condition_1_Artery - 1.000001e+06 t285_leaf3
   >= -1e+06
 T285_R4_f28_ge: x_TotRms_AbvGrd - 12 t285_leaf4 >= 0
 T285_L4_f18_le: const_2nd_Flr_SF + 999025 t285_leaf4 <= 1e+06
 T285_R5_f28_ge: x_TotRms_AbvGrd - 12 t285_leaf5 >= 0
 T285_R5_f18_ge: const_2nd_Flr_SF - 1.000975e+06 t285_leaf5 >= -1e+06
 TREE_286_ONEHOT: t286_leaf0 + t286_leaf1 + t286_leaf2 + t286_leaf3
   + t286_leaf4 + t286_leaf5 = 1
 T286_L0_f3_le: Overall_Qual_calc + 999997 t286_leaf0 <= 1e+06
 T286_L0_f33_le: x_Garage_Area + 99714 t286_leaf0 <= 100000
 T286_L1_f3_le: Overall_Qual_calc + 999997 t286_leaf1 <= 1e+06
 T286_R1_f33_ge: x_Garage_Area - 286 t286_leaf1 >= 0
 T286_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t286_leaf2 >= -1e+06
 T286_L2_f1_le: const_Lot_Area + 979569 t286_leaf2 <= 1e+06
 T286_L2_f1_le: const_Lot_Area + 982459 t286_leaf2 <= 1e+06
 T286_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t286_leaf3 >= -1e+06
 T286_L3_f1_le: const_Lot_Area + 979569 t286_leaf3 <= 1e+06
 T286_R3_f1_ge: const_Lot_Area - 1.017541e+06 t286_leaf3 >= -1e+06
 T286_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t286_leaf4 >= -1e+06
 T286_R4_f1_ge: const_Lot_Area - 1.020431e+06 t286_leaf4 >= -1e+06
 T286_L4_f25_le: x_Bedroom_AbvGr + 2 t286_leaf4 <= 6
 T286_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t286_leaf5 >= -1e+06
 T286_R5_f1_ge: const_Lot_Area - 1.020431e+06 t286_leaf5 >= -1e+06
 T286_R5_f25_ge: x_Bedroom_AbvGr - 4 t286_leaf5 >= 0
 TREE_287_ONEHOT: t287_leaf0 + t287_leaf1 + t287_leaf2 + t287_leaf3
   + t287_leaf4 + t287_leaf5 = 1
 T287_L0_f3_le: Overall_Qual_calc + 999997 t287_leaf0 <= 1e+06
 T287_L0_f35_le: x_Garage_Cond + 2 t287_leaf0 <= 4
 T287_L1_f3_le: Overall_Qual_calc + 999997 t287_leaf1 <= 1e+06
 T287_R1_f35_ge: x_Garage_Cond - 3 t287_leaf1 >= -1
 T287_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t287_leaf2 >= -1e+06
 T287_L2_f21_le: const_Bsmt_Full_Bath + 999998 t287_leaf2 <= 1e+06
 T287_L2_f3_le: Overall_Qual_calc + 999993 t287_leaf2 <= 1e+06
 T287_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t287_leaf3 >= -1e+06
 T287_L3_f21_le: const_Bsmt_Full_Bath + 999998 t287_leaf3 <= 1e+06
 T287_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t287_leaf3 >= -1e+06
 T287_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t287_leaf4 >= -1e+06
 T287_R4_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t287_leaf4 >= -1e+06
 T287_L4_f18_le: const_2nd_Flr_SF + 999495 t287_leaf4 <= 1e+06
 T287_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t287_leaf5 >= -1e+06
 T287_R5_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t287_leaf5 >= -1e+06
 T287_R5_f18_ge: const_2nd_Flr_SF - 1.000505e+06 t287_leaf5 >= -1e+06
 TREE_288_ONEHOT: t288_leaf0 + t288_leaf1 + t288_leaf2 + t288_leaf3
   + t288_leaf4 + t288_leaf5 = 1
 T288_L0_f75_le: const_Lot_Shape_IR3 + 999999 t288_leaf0 <= 1e+06
 T288_L0_f14_le: x_Bsmt_Unf_SF + 99592 t288_leaf0 <= 100000
 T288_L0_f12_le: x_BsmtFin_SF_1 + 998427 t288_leaf0 <= 1e+06
 T288_L1_f75_le: const_Lot_Shape_IR3 + 999999 t288_leaf1 <= 1e+06
 T288_L1_f14_le: x_Bsmt_Unf_SF + 99592 t288_leaf1 <= 100000
 T288_R1_f12_ge: x_BsmtFin_SF_1 - 1573 t288_leaf1 >= 0
 T288_L2_f75_le: const_Lot_Shape_IR3 + 999999 t288_leaf2 <= 1e+06
 T288_R2_f14_ge: x_Bsmt_Unf_SF - 408 t288_leaf2 >= 0
 T288_L2_f25_le: x_Bedroom_AbvGr + 2 t288_leaf2 <= 6
 T288_L3_f75_le: const_Lot_Shape_IR3 + 999999 t288_leaf3 <= 1e+06
 T288_R3_f14_ge: x_Bsmt_Unf_SF - 408 t288_leaf3 >= 0
 T288_R3_f25_ge: x_Bedroom_AbvGr - 4 t288_leaf3 >= 0
 T288_R4_f75_ge: const_Lot_Shape_IR3 - 1.000001e+06 t288_leaf4 >= -1e+06
 T288_L4_f249_le: const_Garage_Type_BuiltIn + 999999 t288_leaf4 <= 1e+06
 T288_R5_f75_ge: const_Lot_Shape_IR3 - 1.000001e+06 t288_leaf5 >= -1e+06
 T288_R5_f249_ge: const_Garage_Type_BuiltIn - 1.000001e+06 t288_leaf5
   >= -1e+06
 TREE_289_ONEHOT: t289_leaf0 + t289_leaf1 + t289_leaf2 + t289_leaf3
   + t289_leaf4 + t289_leaf5 = 1
 T289_L0_f24_le: x_Half_Bath + 4 t289_leaf0 <= 5
 T289_L0_f21_le: const_Bsmt_Full_Bath + 999998 t289_leaf0 <= 1e+06
 T289_L0_f271_le: const_Mo_Sold_1 + 999999 t289_leaf0 <= 1e+06
 T289_L1_f24_le: x_Half_Bath + 4 t289_leaf1 <= 5
 T289_L1_f21_le: const_Bsmt_Full_Bath + 999998 t289_leaf1 <= 1e+06
 T289_R1_f271_ge: const_Mo_Sold_1 - 1.000001e+06 t289_leaf1 >= -1e+06
 T289_L2_f24_le: x_Half_Bath + 4 t289_leaf2 <= 5
 T289_R2_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t289_leaf2 >= -1e+06
 T289_R3_f24_ge: x_Half_Bath - t289_leaf3 >= 0
 T289_L3_f237_le: x_elect_is_SBrkr <= 1
 T289_L3_f119_le: const_Condition_1_Norm + 999999 t289_leaf3 <= 1e+06
 T289_R4_f24_ge: x_Half_Bath - t289_leaf4 >= 0
 T289_L4_f237_le: x_elect_is_SBrkr <= 1
 T289_R4_f119_ge: const_Condition_1_Norm - 1.000001e+06 t289_leaf4
   >= -1e+06
 T289_R5_f24_ge: x_Half_Bath - t289_leaf5 >= 0
 T289_R5_f237_ge: x_elect_is_SBrkr - t289_leaf5 >= 0
 TREE_290_ONEHOT: t290_leaf0 + t290_leaf1 + t290_leaf2 + t290_leaf3
   + t290_leaf4 + t290_leaf5 = 1
 T290_L0_f77_le: const_Land_Contour_Bnk + 999999 t290_leaf0 <= 1e+06
 T290_L0_f37_le: x_Open_Porch_SF + 99925 t290_leaf0 <= 100000
 T290_L0_f5_le: const_Year_Built + 997992 t290_leaf0 <= 1e+06
 T290_L1_f77_le: const_Land_Contour_Bnk + 999999 t290_leaf1 <= 1e+06
 T290_L1_f37_le: x_Open_Porch_SF + 99925 t290_leaf1 <= 100000
 T290_R1_f5_ge: const_Year_Built - 1.002008e+06 t290_leaf1 >= -1e+06
 T290_L2_f77_le: const_Land_Contour_Bnk + 999999 t290_leaf2 <= 1e+06
 T290_R2_f37_ge: x_Open_Porch_SF - 75 t290_leaf2 >= 0
 T290_L2_f1_le: const_Lot_Area + 985320 t290_leaf2 <= 1e+06
 T290_L3_f77_le: const_Land_Contour_Bnk + 999999 t290_leaf3 <= 1e+06
 T290_R3_f37_ge: x_Open_Porch_SF - 75 t290_leaf3 >= 0
 T290_R3_f1_ge: const_Lot_Area - 1.01468e+06 t290_leaf3 >= -1e+06
 T290_R4_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t290_leaf4
   >= -1e+06
 T290_L4_f8_le: x_Exter_Qual + t290_leaf4 <= 4
 T290_R5_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t290_leaf5
   >= -1e+06
 T290_R5_f8_ge: x_Exter_Qual - 4 t290_leaf5 >= -1
 TREE_291_ONEHOT: t291_leaf0 + t291_leaf1 + t291_leaf2 + t291_leaf3
   + t291_leaf4 = 1
 T291_L0_f15_le: const_Total_Bsmt_SF + 997508 t291_leaf0 <= 1e+06
 T291_L0_f6_le: const_Year_Remod_Add + 997993 t291_leaf0 <= 1e+06
 T291_L0_f0_le: const_Lot_Frontage + 999866 t291_leaf0 <= 1e+06
 T291_L1_f15_le: const_Total_Bsmt_SF + 997508 t291_leaf1 <= 1e+06
 T291_L1_f6_le: const_Year_Remod_Add + 997993 t291_leaf1 <= 1e+06
 T291_R1_f0_ge: const_Lot_Frontage - 1.000134e+06 t291_leaf1 >= -1e+06
 T291_L2_f15_le: const_Total_Bsmt_SF + 997508 t291_leaf2 <= 1e+06
 T291_R2_f6_ge: const_Year_Remod_Add - 1.002007e+06 t291_leaf2 >= -1e+06
 T291_L2_f280_le: const_Mo_Sold_10 + 999999 t291_leaf2 <= 1e+06
 T291_L3_f15_le: const_Total_Bsmt_SF + 997508 t291_leaf3 <= 1e+06
 T291_R3_f6_ge: const_Year_Remod_Add - 1.002007e+06 t291_leaf3 >= -1e+06
 T291_R3_f280_ge: const_Mo_Sold_10 - 1.000001e+06 t291_leaf3 >= -1e+06
 T291_R4_f15_ge: const_Total_Bsmt_SF - 1.002492e+06 t291_leaf4 >= -1e+06
 TREE_292_ONEHOT: t292_leaf0 + t292_leaf1 + t292_leaf2 + t292_leaf3
   + t292_leaf4 = 1
 T292_L0_f93_le: const_Neighborhood_ClearCr + 999999 t292_leaf0 <= 1e+06
 T292_L0_f9_le: x_Exter_Cond + 2 t292_leaf0 <= 4
 T292_L0_f37_le: x_Open_Porch_SF + 99963 t292_leaf0 <= 100000
 T292_L1_f93_le: const_Neighborhood_ClearCr + 999999 t292_leaf1 <= 1e+06
 T292_L1_f9_le: x_Exter_Cond + 2 t292_leaf1 <= 4
 T292_R1_f37_ge: x_Open_Porch_SF - 37 t292_leaf1 >= 0
 T292_L2_f93_le: const_Neighborhood_ClearCr + 999999 t292_leaf2 <= 1e+06
 T292_R2_f9_ge: x_Exter_Cond - 3 t292_leaf2 >= -1
 T292_L2_f41_le: x_Pool_Area + 99632 t292_leaf2 <= 100000
 T292_L3_f93_le: const_Neighborhood_ClearCr + 999999 t292_leaf3 <= 1e+06
 T292_R3_f9_ge: x_Exter_Cond - 3 t292_leaf3 >= -1
 T292_R3_f41_ge: x_Pool_Area - 368 t292_leaf3 >= 0
 T292_R4_f93_ge: const_Neighborhood_ClearCr - 1.000001e+06 t292_leaf4
   >= -1e+06
 TREE_293_ONEHOT: t293_leaf0 + t293_leaf1 + t293_leaf2 + t293_leaf3
   + t293_leaf4 = 1
 T293_L0_f20_le: x_Gr_Liv_Area + 7351 t293_leaf0 <= 10000
 T293_L0_f100_le: const_Neighborhood_IDOTRR + 999999 t293_leaf0 <= 1e+06
 T293_L0_f21_le: const_Bsmt_Full_Bath + 999999 t293_leaf0 <= 1e+06
 T293_L1_f20_le: x_Gr_Liv_Area + 7351 t293_leaf1 <= 10000
 T293_L1_f100_le: const_Neighborhood_IDOTRR + 999999 t293_leaf1 <= 1e+06
 T293_R1_f21_ge: const_Bsmt_Full_Bath - 1.000001e+06 t293_leaf1 >= -1e+06
 T293_L2_f20_le: x_Gr_Liv_Area + 7351 t293_leaf2 <= 10000
 T293_R2_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t293_leaf2
   >= -1e+06
 T293_R3_f20_ge: x_Gr_Liv_Area - 2649 t293_leaf3 >= 0
 T293_L3_f37_le: x_Open_Porch_SF + 99969 t293_leaf3 <= 100000
 T293_R4_f20_ge: x_Gr_Liv_Area - 2649 t293_leaf4 >= 0
 T293_R4_f37_ge: x_Open_Porch_SF - 31 t293_leaf4 >= 0
 TREE_294_ONEHOT: t294_leaf0 + t294_leaf1 + t294_leaf2 + t294_leaf3
   + t294_leaf4 = 1
 T294_L0_f296_le: const_Sale_Condition_Family + 999999 t294_leaf0 <= 1e+06
 T294_L0_f37_le: x_Open_Porch_SF + 99582 t294_leaf0 <= 100000
 T294_L0_f15_le: const_Total_Bsmt_SF + 997864 t294_leaf0 <= 1e+06
 T294_L1_f296_le: const_Sale_Condition_Family + 999999 t294_leaf1 <= 1e+06
 T294_L1_f37_le: x_Open_Porch_SF + 99582 t294_leaf1 <= 100000
 T294_R1_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t294_leaf1 >= -1e+06
 T294_L2_f296_le: const_Sale_Condition_Family + 999999 t294_leaf2 <= 1e+06
 T294_R2_f37_ge: x_Open_Porch_SF - 418 t294_leaf2 >= 0
 T294_R3_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t294_leaf3
   >= -1e+06
 T294_L3_f33_le: x_Garage_Area + 99559 t294_leaf3 <= 100000
 T294_R4_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t294_leaf4
   >= -1e+06
 T294_R4_f33_ge: x_Garage_Area - 441 t294_leaf4 >= 0
 TREE_295_ONEHOT: t295_leaf0 + t295_leaf1 + t295_leaf2 + t295_leaf3
   + t295_leaf4 + t295_leaf5 = 1
 T295_L0_f119_le: const_Condition_1_Norm + 999999 t295_leaf0 <= 1e+06
 T295_L0_f30_le: x_Fireplace_Qu + 2 t295_leaf0 <= 4
 T295_L0_f25_le: x_Bedroom_AbvGr + t295_leaf0 <= 6
 T295_L1_f119_le: const_Condition_1_Norm + 999999 t295_leaf1 <= 1e+06
 T295_L1_f30_le: x_Fireplace_Qu + 2 t295_leaf1 <= 4
 T295_R1_f25_ge: x_Bedroom_AbvGr - 5 t295_leaf1 >= 0
 T295_L2_f119_le: const_Condition_1_Norm + 999999 t295_leaf2 <= 1e+06
 T295_R2_f30_ge: x_Fireplace_Qu - 3 t295_leaf2 >= -1
 T295_R3_f119_ge: const_Condition_1_Norm - 1.000001e+06 t295_leaf3
   >= -1e+06
 T295_L3_f4_le: const_Overall_Cond + 999994 t295_leaf3 <= 1e+06
 T295_L3_f31_le: const_Garage_Yr_Blt + 998077 t295_leaf3 <= 1e+06
 T295_R4_f119_ge: const_Condition_1_Norm - 1.000001e+06 t295_leaf4
   >= -1e+06
 T295_L4_f4_le: const_Overall_Cond + 999994 t295_leaf4 <= 1e+06
 T295_R4_f31_ge: const_Garage_Yr_Blt - 1.001923e+06 t295_leaf4 >= -1e+06
 T295_R5_f119_ge: const_Condition_1_Norm - 1.000001e+06 t295_leaf5
   >= -1e+06
 T295_R5_f4_ge: const_Overall_Cond - 1.000006e+06 t295_leaf5 >= -1e+06
 TREE_296_ONEHOT: t296_leaf0 + t296_leaf1 + t296_leaf2 + t296_leaf3
   + t296_leaf4 + t296_leaf5 = 1
 T296_L0_f96_le: const_Neighborhood_Edwards + 999999 t296_leaf0 <= 1e+06
 T296_L0_f17_le: x_1st_Flr_SF + 2800 t296_leaf0 <= 5000
 T296_L0_f117_le: const_Condition_1_Artery + 999999 t296_leaf0 <= 1e+06
 T296_L1_f96_le: const_Neighborhood_Edwards + 999999 t296_leaf1 <= 1e+06
 T296_L1_f17_le: x_1st_Flr_SF + 2800 t296_leaf1 <= 5000
 T296_R1_f117_ge: const_Condition_1_Artery - 1.000001e+06 t296_leaf1
   >= -1e+06
 T296_L2_f96_le: const_Neighborhood_Edwards + 999999 t296_leaf2 <= 1e+06
 T296_R2_f17_ge: x_1st_Flr_SF - 2200 t296_leaf2 >= 0
 T296_R3_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t296_leaf3
   >= -1e+06
 T296_L3_f37_le: x_Open_Porch_SF + 99762 t296_leaf3 <= 100000
 T296_L3_f85_le: const_Lot_Config_Inside + 999999 t296_leaf3 <= 1e+06
 T296_R4_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t296_leaf4
   >= -1e+06
 T296_L4_f37_le: x_Open_Porch_SF + 99762 t296_leaf4 <= 100000
 T296_R4_f85_ge: const_Lot_Config_Inside - 1.000001e+06 t296_leaf4
   >= -1e+06
 T296_R5_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t296_leaf5
   >= -1e+06
 T296_R5_f37_ge: x_Open_Porch_SF - 238 t296_leaf5 >= 0
 TREE_297_ONEHOT: t297_leaf0 + t297_leaf1 + t297_leaf2 + t297_leaf3 = 1
 T297_L0_f95_le: const_Neighborhood_Crawfor + 999999 t297_leaf0 <= 1e+06
 T297_L0_f82_le: const_Lot_Config_CulDSac + 999999 t297_leaf0 <= 1e+06
 T297_L0_f120_le: const_Condition_1_PosA + 999999 t297_leaf0 <= 1e+06
 T297_L1_f95_le: const_Neighborhood_Crawfor + 999999 t297_leaf1 <= 1e+06
 T297_L1_f82_le: const_Lot_Config_CulDSac + 999999 t297_leaf1 <= 1e+06
 T297_R1_f120_ge: const_Condition_1_PosA - 1.000001e+06 t297_leaf1
   >= -1e+06
 T297_L2_f95_le: const_Neighborhood_Crawfor + 999999 t297_leaf2 <= 1e+06
 T297_R2_f82_ge: const_Lot_Config_CulDSac - 1.000001e+06 t297_leaf2
   >= -1e+06
 T297_R3_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t297_leaf3
   >= -1e+06
 TREE_298_ONEHOT: t298_leaf0 + t298_leaf1 + t298_leaf2 + t298_leaf3
   + t298_leaf4 = 1
 T298_L0_f7_le: x_Mas_Vnr_Area + 1078 t298_leaf0 <= 2000
 T298_L0_f45_le: const_MS_SubClass_20 + 999999 t298_leaf0 <= 1e+06
 T298_L0_f119_le: const_Condition_1_Norm + 999999 t298_leaf0 <= 1e+06
 T298_L1_f7_le: x_Mas_Vnr_Area + 1078 t298_leaf1 <= 2000
 T298_L1_f45_le: const_MS_SubClass_20 + 999999 t298_leaf1 <= 1e+06
 T298_R1_f119_ge: const_Condition_1_Norm - 1.000001e+06 t298_leaf1
   >= -1e+06
 T298_L2_f7_le: x_Mas_Vnr_Area + 1078 t298_leaf2 <= 2000
 T298_R2_f45_ge: const_MS_SubClass_20 - 1.000001e+06 t298_leaf2 >= -1e+06
 T298_L2_f37_le: x_Open_Porch_SF + 99909 t298_leaf2 <= 100000
 T298_L3_f7_le: x_Mas_Vnr_Area + 1078 t298_leaf3 <= 2000
 T298_R3_f45_ge: const_MS_SubClass_20 - 1.000001e+06 t298_leaf3 >= -1e+06
 T298_R3_f37_ge: x_Open_Porch_SF - 91 t298_leaf3 >= 0
 T298_R4_f7_ge: x_Mas_Vnr_Area - 922 t298_leaf4 >= 0
 TREE_299_ONEHOT: t299_leaf0 + t299_leaf1 + t299_leaf2 + t299_leaf3
   + t299_leaf4 + t299_leaf5 + t299_leaf6 = 1
 T299_L0_f0_le: const_Lot_Frontage + 999885 t299_leaf0 <= 1e+06
 T299_L0_f44_le: const_Yr_Sold + 997991 t299_leaf0 <= 1e+06
 T299_L0_f4_le: const_Overall_Cond + 999996 t299_leaf0 <= 1e+06
 T299_L1_f0_le: const_Lot_Frontage + 999885 t299_leaf1 <= 1e+06
 T299_L1_f44_le: const_Yr_Sold + 997991 t299_leaf1 <= 1e+06
 T299_R1_f4_ge: const_Overall_Cond - 1.000004e+06 t299_leaf1 >= -1e+06
 T299_L2_f0_le: const_Lot_Frontage + 999885 t299_leaf2 <= 1e+06
 T299_R2_f44_ge: const_Yr_Sold - 1.002009e+06 t299_leaf2 >= -1e+06
 T299_L2_f297_le: const_Sale_Condition_Normal + 999999 t299_leaf2 <= 1e+06
 T299_L3_f0_le: const_Lot_Frontage + 999885 t299_leaf3 <= 1e+06
 T299_R3_f44_ge: const_Yr_Sold - 1.002009e+06 t299_leaf3 >= -1e+06
 T299_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t299_leaf3
   >= -1e+06
 T299_R4_f0_ge: const_Lot_Frontage - 1.000115e+06 t299_leaf4 >= -1e+06
 T299_L4_f77_le: const_Land_Contour_Bnk + 999999 t299_leaf4 <= 1e+06
 T299_L4_f17_le: x_1st_Flr_SF + 3770 t299_leaf4 <= 5000
 T299_R5_f0_ge: const_Lot_Frontage - 1.000115e+06 t299_leaf5 >= -1e+06
 T299_L5_f77_le: const_Land_Contour_Bnk + 999999 t299_leaf5 <= 1e+06
 T299_R5_f17_ge: x_1st_Flr_SF - 1230 t299_leaf5 >= 0
 T299_R6_f0_ge: const_Lot_Frontage - 1.000115e+06 t299_leaf6 >= -1e+06
 T299_R6_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t299_leaf6
   >= -1e+06
 TREE_300_ONEHOT: t300_leaf0 + t300_leaf1 + t300_leaf2 + t300_leaf3
   + t300_leaf4 + t300_leaf5 = 1
 T300_L0_f1_le: const_Lot_Area + 982459 t300_leaf0 <= 1e+06
 T300_L0_f12_le: x_BsmtFin_SF_1 + 998427 t300_leaf0 <= 1e+06
 T300_L0_f63_le: const_MS_Zoning_FV + 999999 t300_leaf0 <= 1e+06
 T300_L1_f1_le: const_Lot_Area + 982459 t300_leaf1 <= 1e+06
 T300_L1_f12_le: x_BsmtFin_SF_1 + 998427 t300_leaf1 <= 1e+06
 T300_R1_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t300_leaf1 >= -1e+06
 T300_L2_f1_le: const_Lot_Area + 982459 t300_leaf2 <= 1e+06
 T300_R2_f12_ge: x_BsmtFin_SF_1 - 1573 t300_leaf2 >= 0
 T300_R3_f1_ge: const_Lot_Area - 1.017541e+06 t300_leaf3 >= -1e+06
 T300_L3_f247_le: const_Garage_Type_Attchd + 999999 t300_leaf3 <= 1e+06
 T300_R4_f1_ge: const_Lot_Area - 1.017541e+06 t300_leaf4 >= -1e+06
 T300_R4_f247_ge: const_Garage_Type_Attchd - 1.000001e+06 t300_leaf4
   >= -1e+06
 T300_L4_f6_le: const_Year_Remod_Add + 997995 t300_leaf4 <= 1e+06
 T300_R5_f1_ge: const_Lot_Area - 1.017541e+06 t300_leaf5 >= -1e+06
 T300_R5_f247_ge: const_Garage_Type_Attchd - 1.000001e+06 t300_leaf5
   >= -1e+06
 T300_R5_f6_ge: const_Year_Remod_Add - 1.002005e+06 t300_leaf5 >= -1e+06
 TREE_301_ONEHOT: t301_leaf0 + t301_leaf1 + t301_leaf2 + t301_leaf3
   + t301_leaf4 = 1
 T301_L0_f17_le: x_1st_Flr_SF + 2485 t301_leaf0 <= 5000
 T301_L0_f5_le: const_Year_Built + 998085 t301_leaf0 <= 1e+06
 T301_L0_f17_le: x_1st_Flr_SF + 4187 t301_leaf0 <= 5000
 T301_L1_f17_le: x_1st_Flr_SF + 2485 t301_leaf1 <= 5000
 T301_L1_f5_le: const_Year_Built + 998085 t301_leaf1 <= 1e+06
 T301_R1_f17_ge: x_1st_Flr_SF - 813 t301_leaf1 >= 0
 T301_L2_f17_le: x_1st_Flr_SF + 2485 t301_leaf2 <= 5000
 T301_R2_f5_ge: const_Year_Built - 1.001915e+06 t301_leaf2 >= -1e+06
 T301_L2_f44_le: const_Yr_Sold + 997993 t301_leaf2 <= 1e+06
 T301_L3_f17_le: x_1st_Flr_SF + 2485 t301_leaf3 <= 5000
 T301_R3_f5_ge: const_Year_Built - 1.001915e+06 t301_leaf3 >= -1e+06
 T301_R3_f44_ge: const_Yr_Sold - 1.002007e+06 t301_leaf3 >= -1e+06
 T301_R4_f17_ge: x_1st_Flr_SF - 2515 t301_leaf4 >= 0
 TREE_302_ONEHOT: t302_leaf0 + t302_leaf1 + t302_leaf2 + t302_leaf3 = 1
 T302_L0_f17_le: x_1st_Flr_SF + 2800 t302_leaf0 <= 5000
 T302_L0_f113_le: const_Neighborhood_Somerst + 999999 t302_leaf0 <= 1e+06
 T302_L0_f14_le: x_Bsmt_Unf_SF + 99892 t302_leaf0 <= 100000
 T302_L1_f17_le: x_1st_Flr_SF + 2800 t302_leaf1 <= 5000
 T302_L1_f113_le: const_Neighborhood_Somerst + 999999 t302_leaf1 <= 1e+06
 T302_R1_f14_ge: x_Bsmt_Unf_SF - 108 t302_leaf1 >= 0
 T302_L2_f17_le: x_1st_Flr_SF + 2800 t302_leaf2 <= 5000
 T302_R2_f113_ge: const_Neighborhood_Somerst - 1.000001e+06 t302_leaf2
   >= -1e+06
 T302_R3_f17_ge: x_1st_Flr_SF - 2200 t302_leaf3 >= 0
 TREE_303_ONEHOT: t303_leaf0 + t303_leaf1 + t303_leaf2 + t303_leaf3
   + t303_leaf4 = 1
 T303_L0_f7_le: x_Mas_Vnr_Area + 1212 t303_leaf0 <= 2000
 T303_L0_f3_le: Overall_Qual_calc + 999990 t303_leaf0 <= 1e+06
 T303_L0_f40_le: x_Screen_Porch + 99890 t303_leaf0 <= 100000
 T303_L1_f7_le: x_Mas_Vnr_Area + 1212 t303_leaf1 <= 2000
 T303_L1_f3_le: Overall_Qual_calc + 999990 t303_leaf1 <= 1e+06
 T303_R1_f40_ge: x_Screen_Porch - 110 t303_leaf1 >= 0
 T303_L2_f7_le: x_Mas_Vnr_Area + 1212 t303_leaf2 <= 2000
 T303_R2_f3_ge: Overall_Qual_calc - 1.00001e+06 t303_leaf2 >= -1e+06
 T303_R3_f7_ge: x_Mas_Vnr_Area - 788 t303_leaf3 >= 0
 T303_L3_f20_le: x_Gr_Liv_Area + 7272 t303_leaf3 <= 10000
 T303_R4_f7_ge: x_Mas_Vnr_Area - 788 t303_leaf4 >= 0
 T303_R4_f20_ge: x_Gr_Liv_Area - 2728 t303_leaf4 >= 0
 TREE_304_ONEHOT: t304_leaf0 + t304_leaf1 + t304_leaf2 + t304_leaf3
   + t304_leaf4 = 1
 T304_L0_f164_le: x_ex1_is_BrkFace <= 1
 T304_L0_f96_le: const_Neighborhood_Edwards + 999999 t304_leaf0 <= 1e+06
 T304_L0_f71_le: const_Alley_No_aplica + 999999 t304_leaf0 <= 1e+06
 T304_L1_f164_le: x_ex1_is_BrkFace <= 1
 T304_L1_f96_le: const_Neighborhood_Edwards + 999999 t304_leaf1 <= 1e+06
 T304_R1_f71_ge: const_Alley_No_aplica - 1.000001e+06 t304_leaf1 >= -1e+06
 T304_L2_f164_le: x_ex1_is_BrkFace <= 1
 T304_R2_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t304_leaf2
   >= -1e+06
 T304_L2_f14_le: x_Bsmt_Unf_SF + 99048 t304_leaf2 <= 100000
 T304_L3_f164_le: x_ex1_is_BrkFace <= 1
 T304_R3_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t304_leaf3
   >= -1e+06
 T304_R3_f14_ge: x_Bsmt_Unf_SF - 952 t304_leaf3 >= 0
 T304_R4_f164_ge: x_ex1_is_BrkFace - t304_leaf4 >= 0
 TREE_305_ONEHOT: t305_leaf0 + t305_leaf1 + t305_leaf2 + t305_leaf3
   + t305_leaf4 = 1
 T305_L0_f17_le: x_1st_Flr_SF + 2485 t305_leaf0 <= 5000
 T305_L0_f1_le: const_Lot_Area + 994000 t305_leaf0 <= 1e+06
 T305_L0_f67_le: const_MS_Zoning_RM + 999999 t305_leaf0 <= 1e+06
 T305_L1_f17_le: x_1st_Flr_SF + 2485 t305_leaf1 <= 5000
 T305_L1_f1_le: const_Lot_Area + 994000 t305_leaf1 <= 1e+06
 T305_R1_f67_ge: const_MS_Zoning_RM - 1.000001e+06 t305_leaf1 >= -1e+06
 T305_L2_f17_le: x_1st_Flr_SF + 2485 t305_leaf2 <= 5000
 T305_R2_f1_ge: const_Lot_Area - 1.006e+06 t305_leaf2 >= -1e+06
 T305_L2_f112_le: const_Neighborhood_SawyerW + 999999 t305_leaf2 <= 1e+06
 T305_L3_f17_le: x_1st_Flr_SF + 2485 t305_leaf3 <= 5000
 T305_R3_f1_ge: const_Lot_Area - 1.006e+06 t305_leaf3 >= -1e+06
 T305_R3_f112_ge: const_Neighborhood_SawyerW - 1.000001e+06 t305_leaf3
   >= -1e+06
 T305_R4_f17_ge: x_1st_Flr_SF - 2515 t305_leaf4 >= 0
 TREE_306_ONEHOT: t306_leaf0 + t306_leaf1 + t306_leaf2 + t306_leaf3
   + t306_leaf4 + t306_leaf5 = 1
 T306_L0_f3_le: Overall_Qual_calc + 999997 t306_leaf0 <= 1e+06
 T306_L0_f6_le: const_Year_Remod_Add + 998049 t306_leaf0 <= 1e+06
 T306_L1_f3_le: Overall_Qual_calc + 999997 t306_leaf1 <= 1e+06
 T306_R1_f6_ge: const_Year_Remod_Add - 1.001951e+06 t306_leaf1 >= -1e+06
 T306_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t306_leaf2 >= -1e+06
 T306_L2_f5_le: const_Year_Built + 998086 t306_leaf2 <= 1e+06
 T306_L2_f17_le: x_1st_Flr_SF + 4187 t306_leaf2 <= 5000
 T306_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t306_leaf3 >= -1e+06
 T306_L3_f5_le: const_Year_Built + 998086 t306_leaf3 <= 1e+06
 T306_R3_f17_ge: x_1st_Flr_SF - 813 t306_leaf3 >= 0
 T306_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t306_leaf4 >= -1e+06
 T306_R4_f5_ge: const_Year_Built - 1.001914e+06 t306_leaf4 >= -1e+06
 T306_L4_f232_le: central_air_yes + 999999 t306_leaf4 <= 1e+06
 T306_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t306_leaf5 >= -1e+06
 T306_R5_f5_ge: const_Year_Built - 1.001914e+06 t306_leaf5 >= -1e+06
 T306_R5_f232_ge: central_air_yes - 1.000001e+06 t306_leaf5 >= -1e+06
 TREE_307_ONEHOT: t307_leaf0 + t307_leaf1 + t307_leaf2 + t307_leaf3
   + t307_leaf4 + t307_leaf5 = 1
 T307_L0_f3_le: Overall_Qual_calc + 999996 t307_leaf0 <= 1e+06
 T307_L0_f0_le: const_Lot_Frontage + 999944 t307_leaf0 <= 1e+06
 T307_L1_f3_le: Overall_Qual_calc + 999996 t307_leaf1 <= 1e+06
 T307_R1_f0_ge: const_Lot_Frontage - 1.000056e+06 t307_leaf1 >= -1e+06
 T307_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t307_leaf2 >= -1e+06
 T307_L2_f206_le: const_Bsmt_Exposure_Av + 999999 t307_leaf2 <= 1e+06
 T307_L2_f5_le: const_Year_Built + 998085 t307_leaf2 <= 1e+06
 T307_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t307_leaf3 >= -1e+06
 T307_L3_f206_le: const_Bsmt_Exposure_Av + 999999 t307_leaf3 <= 1e+06
 T307_R3_f5_ge: const_Year_Built - 1.001915e+06 t307_leaf3 >= -1e+06
 T307_R4_f3_ge: Overall_Qual_calc - 1.000004e+06 t307_leaf4 >= -1e+06
 T307_R4_f206_ge: const_Bsmt_Exposure_Av - 1.000001e+06 t307_leaf4
   >= -1e+06
 T307_L4_f114_le: const_Neighborhood_StoneBr + 999999 t307_leaf4 <= 1e+06
 T307_R5_f3_ge: Overall_Qual_calc - 1.000004e+06 t307_leaf5 >= -1e+06
 T307_R5_f206_ge: const_Bsmt_Exposure_Av - 1.000001e+06 t307_leaf5
   >= -1e+06
 T307_R5_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t307_leaf5
   >= -1e+06
 TREE_308_ONEHOT: t308_leaf0 + t308_leaf1 + t308_leaf2 + t308_leaf3
   + t308_leaf4 = 1
 T308_L0_f63_le: const_MS_Zoning_FV + 999999 t308_leaf0 <= 1e+06
 T308_L0_f3_le: Overall_Qual_calc + 999993 t308_leaf0 <= 1e+06
 T308_L0_f23_le: x_Full_Bath + 2 t308_leaf0 <= 4
 T308_L1_f63_le: const_MS_Zoning_FV + 999999 t308_leaf1 <= 1e+06
 T308_L1_f3_le: Overall_Qual_calc + 999993 t308_leaf1 <= 1e+06
 T308_R1_f23_ge: x_Full_Bath - 2 t308_leaf1 >= 0
 T308_L2_f63_le: const_MS_Zoning_FV + 999999 t308_leaf2 <= 1e+06
 T308_R2_f3_ge: Overall_Qual_calc - 1.000007e+06 t308_leaf2 >= -1e+06
 T308_L2_f86_le: const_Land_Slope_Gtl + 999999 t308_leaf2 <= 1e+06
 T308_L3_f63_le: const_MS_Zoning_FV + 999999 t308_leaf3 <= 1e+06
 T308_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t308_leaf3 >= -1e+06
 T308_R3_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t308_leaf3 >= -1e+06
 T308_R4_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t308_leaf4 >= -1e+06
 TREE_309_ONEHOT: t309_leaf0 + t309_leaf1 + t309_leaf2 + t309_leaf3
   + t309_leaf4 + t309_leaf5 = 1
 T309_L0_f28_le: x_TotRms_AbvGrd + 3 t309_leaf0 <= 15
 T309_L0_f261_le: x_fence_is_GdWo <= 1
 T309_L0_f15_le: const_Total_Bsmt_SF + 997508 t309_leaf0 <= 1e+06
 T309_L1_f28_le: x_TotRms_AbvGrd + 3 t309_leaf1 <= 15
 T309_L1_f261_le: x_fence_is_GdWo <= 1
 T309_R1_f15_ge: const_Total_Bsmt_SF - 1.002492e+06 t309_leaf1 >= -1e+06
 T309_L2_f28_le: x_TotRms_AbvGrd + 3 t309_leaf2 <= 15
 T309_R2_f261_ge: x_fence_is_GdWo - t309_leaf2 >= 0
 T309_L2_f38_le: x_Enclosed_Porch + 99899 t309_leaf2 <= 100000
 T309_L3_f28_le: x_TotRms_AbvGrd + 3 t309_leaf3 <= 15
 T309_R3_f261_ge: x_fence_is_GdWo - t309_leaf3 >= 0
 T309_R3_f38_ge: x_Enclosed_Porch - 101 t309_leaf3 >= 0
 T309_R4_f28_ge: x_TotRms_AbvGrd - 12 t309_leaf4 >= 0
 T309_L4_f18_le: const_2nd_Flr_SF + 999025 t309_leaf4 <= 1e+06
 T309_R5_f28_ge: x_TotRms_AbvGrd - 12 t309_leaf5 >= 0
 T309_R5_f18_ge: const_2nd_Flr_SF - 1.000975e+06 t309_leaf5 >= -1e+06
 TREE_310_ONEHOT: t310_leaf0 + t310_leaf1 + t310_leaf2 + t310_leaf3
   + t310_leaf4 = 1
 T310_L0_f68_le: const_Street_Grvl + 999999 t310_leaf0 <= 1e+06
 T310_L0_f27_le: x_Kitchen_Qual + 2 t310_leaf0 <= 4
 T310_L0_f119_le: const_Condition_1_Norm + 999999 t310_leaf0 <= 1e+06
 T310_L1_f68_le: const_Street_Grvl + 999999 t310_leaf1 <= 1e+06
 T310_L1_f27_le: x_Kitchen_Qual + 2 t310_leaf1 <= 4
 T310_R1_f119_ge: const_Condition_1_Norm - 1.000001e+06 t310_leaf1
   >= -1e+06
 T310_L2_f68_le: const_Street_Grvl + 999999 t310_leaf2 <= 1e+06
 T310_R2_f27_ge: x_Kitchen_Qual - 3 t310_leaf2 >= -1
 T310_L2_f4_le: const_Overall_Cond + 999996 t310_leaf2 <= 1e+06
 T310_L3_f68_le: const_Street_Grvl + 999999 t310_leaf3 <= 1e+06
 T310_R3_f27_ge: x_Kitchen_Qual - 3 t310_leaf3 >= -1
 T310_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t310_leaf3 >= -1e+06
 T310_R4_f68_ge: const_Street_Grvl - 1.000001e+06 t310_leaf4 >= -1e+06
 TREE_311_ONEHOT: t311_leaf0 + t311_leaf1 + t311_leaf2 + t311_leaf3 = 1
 T311_L0_f6_le: const_Year_Remod_Add + 997992 t311_leaf0 <= 1e+06
 T311_L0_f4_le: const_Overall_Cond + 999994 t311_leaf0 <= 1e+06
 T311_L0_f17_le: x_1st_Flr_SF + 3060 t311_leaf0 <= 5000
 T311_L1_f6_le: const_Year_Remod_Add + 997992 t311_leaf1 <= 1e+06
 T311_L1_f4_le: const_Overall_Cond + 999994 t311_leaf1 <= 1e+06
 T311_R1_f17_ge: x_1st_Flr_SF - 1940 t311_leaf1 >= 0
 T311_L2_f6_le: const_Year_Remod_Add + 997992 t311_leaf2 <= 1e+06
 T311_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t311_leaf2 >= -1e+06
 T311_R3_f6_ge: const_Year_Remod_Add - 1.002008e+06 t311_leaf3 >= -1e+06
 TREE_312_ONEHOT: t312_leaf0 + t312_leaf1 + t312_leaf2 + t312_leaf3
   + t312_leaf4 = 1
 T312_L0_f33_le: x_Garage_Area + 98846 t312_leaf0 <= 100000
 T312_L0_f202_le: const_Foundation_PConc + 999999 t312_leaf0 <= 1e+06
 T312_L0_f5_le: const_Year_Built + 998075 t312_leaf0 <= 1e+06
 T312_L1_f33_le: x_Garage_Area + 98846 t312_leaf1 <= 100000
 T312_L1_f202_le: const_Foundation_PConc + 999999 t312_leaf1 <= 1e+06
 T312_R1_f5_ge: const_Year_Built - 1.001925e+06 t312_leaf1 >= -1e+06
 T312_L2_f33_le: x_Garage_Area + 98846 t312_leaf2 <= 100000
 T312_R2_f202_ge: const_Foundation_PConc - 1.000001e+06 t312_leaf2
   >= -1e+06
 T312_L2_f174_le: x_ex1_is_VinylSd <= 1
 T312_L3_f33_le: x_Garage_Area + 98846 t312_leaf3 <= 100000
 T312_R3_f202_ge: const_Foundation_PConc - 1.000001e+06 t312_leaf3
   >= -1e+06
 T312_R3_f174_ge: x_ex1_is_VinylSd - t312_leaf3 >= 0
 T312_R4_f33_ge: x_Garage_Area - 1154 t312_leaf4 >= 0
 TREE_313_ONEHOT: t313_leaf0 + t313_leaf1 + t313_leaf2 + t313_leaf3
   + t313_leaf4 + t313_leaf5 + t313_leaf6 = 1
 T313_L0_f9_le: x_Exter_Cond + 2 t313_leaf0 <= 4
 T313_L0_f1_le: const_Lot_Area + 990829 t313_leaf0 <= 1e+06
 T313_L0_f5_le: const_Year_Built + 998069 t313_leaf0 <= 1e+06
 T313_L1_f9_le: x_Exter_Cond + 2 t313_leaf1 <= 4
 T313_L1_f1_le: const_Lot_Area + 990829 t313_leaf1 <= 1e+06
 T313_R1_f5_ge: const_Year_Built - 1.001931e+06 t313_leaf1 >= -1e+06
 T313_L2_f9_le: x_Exter_Cond + 2 t313_leaf2 <= 4
 T313_R2_f1_ge: const_Lot_Area - 1.009171e+06 t313_leaf2 >= -1e+06
 T313_L2_f37_le: x_Open_Porch_SF + 99996 t313_leaf2 <= 100000
 T313_L3_f9_le: x_Exter_Cond + 2 t313_leaf3 <= 4
 T313_R3_f1_ge: const_Lot_Area - 1.009171e+06 t313_leaf3 >= -1e+06
 T313_R3_f37_ge: x_Open_Porch_SF - 4 t313_leaf3 >= 0
 T313_R4_f9_ge: x_Exter_Cond - 3 t313_leaf4 >= -1
 T313_L4_f86_le: const_Land_Slope_Gtl + 999999 t313_leaf4 <= 1e+06
 T313_R5_f9_ge: x_Exter_Cond - 3 t313_leaf5 >= -1
 T313_R5_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t313_leaf5 >= -1e+06
 T313_L5_f1_le: const_Lot_Area + 964867 t313_leaf5 <= 1e+06
 T313_R6_f9_ge: x_Exter_Cond - 3 t313_leaf6 >= -1
 T313_R6_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t313_leaf6 >= -1e+06
 T313_R6_f1_ge: const_Lot_Area - 1.035133e+06 t313_leaf6 >= -1e+06
 TREE_314_ONEHOT: t314_leaf0 + t314_leaf1 + t314_leaf2 + t314_leaf3
   + t314_leaf4 + t314_leaf5 + t314_leaf6 = 1
 T314_L0_f32_le: const_Garage_Cars + 999997 t314_leaf0 <= 1e+06
 T314_L0_f119_le: const_Condition_1_Norm + 999999 t314_leaf0 <= 1e+06
 T314_L0_f5_le: const_Year_Built + 998058 t314_leaf0 <= 1e+06
 T314_L1_f32_le: const_Garage_Cars + 999997 t314_leaf1 <= 1e+06
 T314_L1_f119_le: const_Condition_1_Norm + 999999 t314_leaf1 <= 1e+06
 T314_R1_f5_ge: const_Year_Built - 1.001942e+06 t314_leaf1 >= -1e+06
 T314_L2_f32_le: const_Garage_Cars + 999997 t314_leaf2 <= 1e+06
 T314_R2_f119_ge: const_Condition_1_Norm - 1.000001e+06 t314_leaf2
   >= -1e+06
 T314_L2_f45_le: const_MS_SubClass_20 + 999999 t314_leaf2 <= 1e+06
 T314_L3_f32_le: const_Garage_Cars + 999997 t314_leaf3 <= 1e+06
 T314_R3_f119_ge: const_Condition_1_Norm - 1.000001e+06 t314_leaf3
   >= -1e+06
 T314_R3_f45_ge: const_MS_SubClass_20 - 1.000001e+06 t314_leaf3 >= -1e+06
 T314_R4_f32_ge: const_Garage_Cars - 1.000003e+06 t314_leaf4 >= -1e+06
 T314_L4_f3_le: Overall_Qual_calc + 999990 t314_leaf4 <= 1e+06
 T314_L4_f272_le: const_Mo_Sold_2 + 999999 t314_leaf4 <= 1e+06
 T314_R5_f32_ge: const_Garage_Cars - 1.000003e+06 t314_leaf5 >= -1e+06
 T314_L5_f3_le: Overall_Qual_calc + 999990 t314_leaf5 <= 1e+06
 T314_R5_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t314_leaf5 >= -1e+06
 T314_R6_f32_ge: const_Garage_Cars - 1.000003e+06 t314_leaf6 >= -1e+06
 T314_R6_f3_ge: Overall_Qual_calc - 1.00001e+06 t314_leaf6 >= -1e+06
 TREE_315_ONEHOT: t315_leaf0 + t315_leaf1 + t315_leaf2 + t315_leaf3
   + t315_leaf4 = 1
 T315_L0_f86_le: const_Land_Slope_Gtl + 999999 t315_leaf0 <= 1e+06
 T315_L0_f255_le: const_Garage_Finish_RFn + 999999 t315_leaf0 <= 1e+06
 T315_L1_f86_le: const_Land_Slope_Gtl + 999999 t315_leaf1 <= 1e+06
 T315_R1_f255_ge: const_Garage_Finish_RFn - 1.000001e+06 t315_leaf1
   >= -1e+06
 T315_R2_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t315_leaf2 >= -1e+06
 T315_L2_f20_le: x_Gr_Liv_Area + 6914 t315_leaf2 <= 10000
 T315_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t315_leaf2 <= 1e+06
 T315_R3_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t315_leaf3 >= -1e+06
 T315_L3_f20_le: x_Gr_Liv_Area + 6914 t315_leaf3 <= 10000
 T315_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t315_leaf3
   >= -1e+06
 T315_R4_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t315_leaf4 >= -1e+06
 T315_R4_f20_ge: x_Gr_Liv_Area - 3086 t315_leaf4 >= 0
 TREE_316_ONEHOT: t316_leaf0 + t316_leaf1 + t316_leaf2 + t316_leaf3
   + t316_leaf4 = 1
 T316_L0_f206_le: const_Bsmt_Exposure_Av + 999999 t316_leaf0 <= 1e+06
 T316_L0_f296_le: const_Sale_Condition_Family + 999999 t316_leaf0 <= 1e+06
 T316_L0_f9_le: x_Exter_Cond + 2 t316_leaf0 <= 4
 T316_L1_f206_le: const_Bsmt_Exposure_Av + 999999 t316_leaf1 <= 1e+06
 T316_L1_f296_le: const_Sale_Condition_Family + 999999 t316_leaf1 <= 1e+06
 T316_R1_f9_ge: x_Exter_Cond - 3 t316_leaf1 >= -1
 T316_L2_f206_le: const_Bsmt_Exposure_Av + 999999 t316_leaf2 <= 1e+06
 T316_R2_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t316_leaf2
   >= -1e+06
 T316_L2_f7_le: x_Mas_Vnr_Area + 1808 t316_leaf2 <= 2000
 T316_L3_f206_le: const_Bsmt_Exposure_Av + 999999 t316_leaf3 <= 1e+06
 T316_R3_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t316_leaf3
   >= -1e+06
 T316_R3_f7_ge: x_Mas_Vnr_Area - 192 t316_leaf3 >= 0
 T316_R4_f206_ge: const_Bsmt_Exposure_Av - 1.000001e+06 t316_leaf4
   >= -1e+06
 TREE_317_ONEHOT: t317_leaf0 + t317_leaf1 + t317_leaf2 + t317_leaf3 = 1
 T317_L0_f43_le: const_Misc_Val + 999500 t317_leaf0 <= 1e+06
 T317_L0_f92_le: const_Neighborhood_BrkSide + 999999 t317_leaf0 <= 1e+06
 T317_L0_f209_le: const_Bsmt_Exposure_No + 999999 t317_leaf0 <= 1e+06
 T317_L1_f43_le: const_Misc_Val + 999500 t317_leaf1 <= 1e+06
 T317_L1_f92_le: const_Neighborhood_BrkSide + 999999 t317_leaf1 <= 1e+06
 T317_R1_f209_ge: const_Bsmt_Exposure_No - 1.000001e+06 t317_leaf1
   >= -1e+06
 T317_L2_f43_le: const_Misc_Val + 999500 t317_leaf2 <= 1e+06
 T317_R2_f92_ge: const_Neighborhood_BrkSide - 1.000001e+06 t317_leaf2
   >= -1e+06
 T317_R3_f43_ge: const_Misc_Val - 1.0005e+06 t317_leaf3 >= -1e+06
 TREE_318_ONEHOT: t318_leaf0 + t318_leaf1 + t318_leaf2 + t318_leaf3 = 1
 T318_L0_f69_le: const_Street_Pave + 999999 t318_leaf0 <= 1e+06
 T318_R1_f69_ge: const_Street_Pave - 1.000001e+06 t318_leaf1 >= -1e+06
 T318_L1_f36_le: x_Wood_Deck_SF + 99489 t318_leaf1 <= 100000
 T318_L1_f7_le: x_Mas_Vnr_Area + 1541 t318_leaf1 <= 2000
 T318_R2_f69_ge: const_Street_Pave - 1.000001e+06 t318_leaf2 >= -1e+06
 T318_L2_f36_le: x_Wood_Deck_SF + 99489 t318_leaf2 <= 100000
 T318_R2_f7_ge: x_Mas_Vnr_Area - 459 t318_leaf2 >= 0
 T318_R3_f69_ge: const_Street_Pave - 1.000001e+06 t318_leaf3 >= -1e+06
 T318_R3_f36_ge: x_Wood_Deck_SF - 511 t318_leaf3 >= 0
 TREE_319_ONEHOT: t319_leaf0 + t319_leaf1 + t319_leaf2 + t319_leaf3
   + t319_leaf4 = 1
 T319_L0_f44_le: const_Yr_Sold + 997991 t319_leaf0 <= 1e+06
 T319_L0_f36_le: x_Wood_Deck_SF + 99533 t319_leaf0 <= 100000
 T319_L1_f44_le: const_Yr_Sold + 997991 t319_leaf1 <= 1e+06
 T319_R1_f36_ge: x_Wood_Deck_SF - 467 t319_leaf1 >= 0
 T319_R2_f44_ge: const_Yr_Sold - 1.002009e+06 t319_leaf2 >= -1e+06
 T319_L2_f0_le: const_Lot_Frontage + 999915 t319_leaf2 <= 1e+06
 T319_L2_f261_le: x_fence_is_GdWo <= 1
 T319_R3_f44_ge: const_Yr_Sold - 1.002009e+06 t319_leaf3 >= -1e+06
 T319_L3_f0_le: const_Lot_Frontage + 999915 t319_leaf3 <= 1e+06
 T319_R3_f261_ge: x_fence_is_GdWo - t319_leaf3 >= 0
 T319_R4_f44_ge: const_Yr_Sold - 1.002009e+06 t319_leaf4 >= -1e+06
 T319_R4_f0_ge: const_Lot_Frontage - 1.000085e+06 t319_leaf4 >= -1e+06
 TREE_320_ONEHOT: t320_leaf0 + t320_leaf1 + t320_leaf2 + t320_leaf3
   + t320_leaf4 + t320_leaf5 + t320_leaf6 = 1
 T320_L0_f33_le: x_Garage_Area + 99370 t320_leaf0 <= 100000
 T320_L0_f20_le: x_Gr_Liv_Area + 9253 t320_leaf0 <= 10000
 T320_L0_f16_le: x_Heating_QC + t320_leaf0 <= 4
 T320_L1_f33_le: x_Garage_Area + 99370 t320_leaf1 <= 100000
 T320_L1_f20_le: x_Gr_Liv_Area + 9253 t320_leaf1 <= 10000
 T320_R1_f16_ge: x_Heating_QC - 4 t320_leaf1 >= -1
 T320_L2_f33_le: x_Garage_Area + 99370 t320_leaf2 <= 100000
 T320_R2_f20_ge: x_Gr_Liv_Area - 747 t320_leaf2 >= 0
 T320_L2_f11_le: x_Bsmt_Cond + t320_leaf2 <= 4
 T320_L3_f33_le: x_Garage_Area + 99370 t320_leaf3 <= 100000
 T320_R3_f20_ge: x_Gr_Liv_Area - 747 t320_leaf3 >= 0
 T320_R3_f11_ge: x_Bsmt_Cond - 4 t320_leaf3 >= -1
 T320_R4_f33_ge: x_Garage_Area - 630 t320_leaf4 >= 0
 T320_L4_f4_le: const_Overall_Cond + 999996 t320_leaf4 <= 1e+06
 T320_R5_f33_ge: x_Garage_Area - 630 t320_leaf5 >= 0
 T320_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t320_leaf5 >= -1e+06
 T320_L5_f33_le: x_Garage_Area + 98846 t320_leaf5 <= 100000
 T320_R6_f33_ge: x_Garage_Area - 630 t320_leaf6 >= 0
 T320_R6_f4_ge: const_Overall_Cond - 1.000004e+06 t320_leaf6 >= -1e+06
 T320_R6_f33_ge: x_Garage_Area - 1154 t320_leaf6 >= 0
 TREE_321_ONEHOT: t321_leaf0 + t321_leaf1 + t321_leaf2 + t321_leaf3
   + t321_leaf4 = 1
 T321_L0_f177_le: x_ex2_is_AsbShng <= 1
 T321_L0_f161_le: x_ex1_is_AsbShng <= 1
 T321_L0_f4_le: const_Overall_Cond + 999993 t321_leaf0 <= 1e+06
 T321_L1_f177_le: x_ex2_is_AsbShng <= 1
 T321_L1_f161_le: x_ex1_is_AsbShng <= 1
 T321_R1_f4_ge: const_Overall_Cond - 1.000007e+06 t321_leaf1 >= -1e+06
 T321_L2_f177_le: x_ex2_is_AsbShng <= 1
 T321_R2_f161_ge: x_ex1_is_AsbShng - t321_leaf2 >= 0
 T321_R3_f177_ge: x_ex2_is_AsbShng - t321_leaf3 >= 0
 T321_L3_f15_le: const_Total_Bsmt_SF + 999279 t321_leaf3 <= 1e+06
 T321_R4_f177_ge: x_ex2_is_AsbShng - t321_leaf4 >= 0
 T321_R4_f15_ge: const_Total_Bsmt_SF - 1.000721e+06 t321_leaf4 >= -1e+06
 TREE_322_ONEHOT: t322_leaf0 + t322_leaf1 + t322_leaf2 + t322_leaf3
   + t322_leaf4 + t322_leaf5 + t322_leaf6 = 1
 T322_L0_f232_le: central_air_yes + 999999 t322_leaf0 <= 1e+06
 T322_L0_f4_le: const_Overall_Cond + 999995 t322_leaf0 <= 1e+06
 T322_L0_f14_le: x_Bsmt_Unf_SF + 99273 t322_leaf0 <= 100000
 T322_L1_f232_le: central_air_yes + 999999 t322_leaf1 <= 1e+06
 T322_L1_f4_le: const_Overall_Cond + 999995 t322_leaf1 <= 1e+06
 T322_R1_f14_ge: x_Bsmt_Unf_SF - 727 t322_leaf1 >= 0
 T322_L2_f232_le: central_air_yes + 999999 t322_leaf2 <= 1e+06
 T322_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t322_leaf2 >= -1e+06
 T322_L2_f34_le: x_Garage_Qual + 2 t322_leaf2 <= 4
 T322_L3_f232_le: central_air_yes + 999999 t322_leaf3 <= 1e+06
 T322_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t322_leaf3 >= -1e+06
 T322_R3_f34_ge: x_Garage_Qual - 3 t322_leaf3 >= -1
 T322_R4_f232_ge: central_air_yes - 1.000001e+06 t322_leaf4 >= -1e+06
 T322_L4_f40_le: x_Screen_Porch + 99850 t322_leaf4 <= 100000
 T322_L4_f4_le: const_Overall_Cond + 999995 t322_leaf4 <= 1e+06
 T322_R5_f232_ge: central_air_yes - 1.000001e+06 t322_leaf5 >= -1e+06
 T322_L5_f40_le: x_Screen_Porch + 99850 t322_leaf5 <= 100000
 T322_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t322_leaf5 >= -1e+06
 T322_R6_f232_ge: central_air_yes - 1.000001e+06 t322_leaf6 >= -1e+06
 T322_R6_f40_ge: x_Screen_Porch - 150 t322_leaf6 >= 0
 TREE_323_ONEHOT: t323_leaf0 + t323_leaf1 + t323_leaf2 + t323_leaf3 = 1
 T323_L0_f51_le: const_MS_SubClass_70 + 999999 t323_leaf0 <= 1e+06
 T323_L0_f147_le: const_Roof_Style_Flat + 999999 t323_leaf0 <= 1e+06
 T323_L0_f213_le: x_b1_is_GLQ <= 1
 T323_L1_f51_le: const_MS_SubClass_70 + 999999 t323_leaf1 <= 1e+06
 T323_L1_f147_le: const_Roof_Style_Flat + 999999 t323_leaf1 <= 1e+06
 T323_R1_f213_ge: x_b1_is_GLQ - t323_leaf1 >= 0
 T323_L2_f51_le: const_MS_SubClass_70 + 999999 t323_leaf2 <= 1e+06
 T323_R2_f147_ge: const_Roof_Style_Flat - 1.000001e+06 t323_leaf2
   >= -1e+06
 T323_R3_f51_ge: const_MS_SubClass_70 - 1.000001e+06 t323_leaf3 >= -1e+06
 TREE_324_ONEHOT: t324_leaf0 + t324_leaf1 + t324_leaf2 + t324_leaf3
   + t324_leaf4 + t324_leaf5 + t324_leaf6 = 1
 T324_L0_f44_le: const_Yr_Sold + 997992 t324_leaf0 <= 1e+06
 T324_L0_f283_le: const_Sale_Type_COD + 999999 t324_leaf0 <= 1e+06
 T324_L0_f233_le: x_elect_is_FuseA <= 1
 T324_L1_f44_le: const_Yr_Sold + 997992 t324_leaf1 <= 1e+06
 T324_L1_f283_le: const_Sale_Type_COD + 999999 t324_leaf1 <= 1e+06
 T324_R1_f233_ge: x_elect_is_FuseA - t324_leaf1 >= 0
 T324_L2_f44_le: const_Yr_Sold + 997992 t324_leaf2 <= 1e+06
 T324_R2_f283_ge: const_Sale_Type_COD - 1.000001e+06 t324_leaf2 >= -1e+06
 T324_R3_f44_ge: const_Yr_Sold - 1.002008e+06 t324_leaf3 >= -1e+06
 T324_L3_f5_le: const_Year_Built + 998075 t324_leaf3 <= 1e+06
 T324_L3_f6_le: const_Year_Remod_Add + 998003 t324_leaf3 <= 1e+06
 T324_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t324_leaf4 >= -1e+06
 T324_L4_f5_le: const_Year_Built + 998075 t324_leaf4 <= 1e+06
 T324_R4_f6_ge: const_Year_Remod_Add - 1.001997e+06 t324_leaf4 >= -1e+06
 T324_R5_f44_ge: const_Yr_Sold - 1.002008e+06 t324_leaf5 >= -1e+06
 T324_R5_f5_ge: const_Year_Built - 1.001925e+06 t324_leaf5 >= -1e+06
 T324_L5_f219_le: x_b2_is_BLQ <= 1
 T324_R6_f44_ge: const_Yr_Sold - 1.002008e+06 t324_leaf6 >= -1e+06
 T324_R6_f5_ge: const_Year_Built - 1.001925e+06 t324_leaf6 >= -1e+06
 T324_R6_f219_ge: x_b2_is_BLQ - t324_leaf6 >= 0
 TREE_325_ONEHOT: t325_leaf0 + t325_leaf1 + t325_leaf2 + t325_leaf3
   + t325_leaf4 = 1
 T325_L0_f33_le: x_Garage_Area + 99105 t325_leaf0 <= 100000
 T325_L0_f161_le: x_ex1_is_AsbShng <= 1
 T325_L0_f14_le: x_Bsmt_Unf_SF + 99524 t325_leaf0 <= 100000
 T325_L1_f33_le: x_Garage_Area + 99105 t325_leaf1 <= 100000
 T325_L1_f161_le: x_ex1_is_AsbShng <= 1
 T325_R1_f14_ge: x_Bsmt_Unf_SF - 476 t325_leaf1 >= 0
 T325_L2_f33_le: x_Garage_Area + 99105 t325_leaf2 <= 100000
 T325_R2_f161_ge: x_ex1_is_AsbShng - t325_leaf2 >= 0
 T325_L2_f14_le: x_Bsmt_Unf_SF + 99493 t325_leaf2 <= 100000
 T325_L3_f33_le: x_Garage_Area + 99105 t325_leaf3 <= 100000
 T325_R3_f161_ge: x_ex1_is_AsbShng - t325_leaf3 >= 0
 T325_R3_f14_ge: x_Bsmt_Unf_SF - 507 t325_leaf3 >= 0
 T325_R4_f33_ge: x_Garage_Area - 895 t325_leaf4 >= 0
 TREE_326_ONEHOT: t326_leaf0 + t326_leaf1 + t326_leaf2 + t326_leaf3 = 1
 T326_L0_f121_le: const_Condition_1_PosN + 999999 t326_leaf0 <= 1e+06
 T326_L0_f164_le: x_ex1_is_BrkFace <= 1
 T326_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t326_leaf0 <= 1e+06
 T326_L1_f121_le: const_Condition_1_PosN + 999999 t326_leaf1 <= 1e+06
 T326_L1_f164_le: x_ex1_is_BrkFace <= 1
 T326_R1_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t326_leaf1
   >= -1e+06
 T326_L2_f121_le: const_Condition_1_PosN + 999999 t326_leaf2 <= 1e+06
 T326_R2_f164_ge: x_ex1_is_BrkFace - t326_leaf2 >= 0
 T326_R3_f121_ge: const_Condition_1_PosN - 1.000001e+06 t326_leaf3
   >= -1e+06
 TREE_327_ONEHOT: t327_leaf0 + t327_leaf1 + t327_leaf2 + t327_leaf3
   + t327_leaf4 = 1
 T327_L0_f113_le: const_Neighborhood_Somerst + 999999 t327_leaf0 <= 1e+06
 T327_L0_f4_le: const_Overall_Cond + 999994 t327_leaf0 <= 1e+06
 T327_L0_f31_le: const_Garage_Yr_Blt + 998034 t327_leaf0 <= 1e+06
 T327_L1_f113_le: const_Neighborhood_Somerst + 999999 t327_leaf1 <= 1e+06
 T327_L1_f4_le: const_Overall_Cond + 999994 t327_leaf1 <= 1e+06
 T327_R1_f31_ge: const_Garage_Yr_Blt - 1.001966e+06 t327_leaf1 >= -1e+06
 T327_L2_f113_le: const_Neighborhood_Somerst + 999999 t327_leaf2 <= 1e+06
 T327_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t327_leaf2 >= -1e+06
 T327_L2_f17_le: x_1st_Flr_SF + 3586 t327_leaf2 <= 5000
 T327_L3_f113_le: const_Neighborhood_Somerst + 999999 t327_leaf3 <= 1e+06
 T327_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t327_leaf3 >= -1e+06
 T327_R3_f17_ge: x_1st_Flr_SF - 1414 t327_leaf3 >= 0
 T327_R4_f113_ge: const_Neighborhood_Somerst - 1.000001e+06 t327_leaf4
   >= -1e+06
 TREE_328_ONEHOT: t328_leaf0 + t328_leaf1 + t328_leaf2 + t328_leaf3
   + t328_leaf4 + t328_leaf5 = 1
 T328_L0_f16_le: x_Heating_QC + 2 t328_leaf0 <= 4
 T328_L0_f191_le: x_ex2_is_VinylSd <= 1
 T328_L1_f16_le: x_Heating_QC + 2 t328_leaf1 <= 4
 T328_R1_f191_ge: x_ex2_is_VinylSd - t328_leaf1 >= 0
 T328_R2_f16_ge: x_Heating_QC - 3 t328_leaf2 >= -1
 T328_L2_f232_le: central_air_yes + 999999 t328_leaf2 <= 1e+06
 T328_L2_f217_le: x_b1_is_Unf <= 1
 T328_R3_f16_ge: x_Heating_QC - 3 t328_leaf3 >= -1
 T328_L3_f232_le: central_air_yes + 999999 t328_leaf3 <= 1e+06
 T328_R3_f217_ge: x_b1_is_Unf - t328_leaf3 >= 0
 T328_R4_f16_ge: x_Heating_QC - 3 t328_leaf4 >= -1
 T328_R4_f232_ge: central_air_yes - 1.000001e+06 t328_leaf4 >= -1e+06
 T328_L4_f193_le: const_Exterior_2nd_Wd_Shng + 999999 t328_leaf4 <= 1e+06
 T328_R5_f16_ge: x_Heating_QC - 3 t328_leaf5 >= -1
 T328_R5_f232_ge: central_air_yes - 1.000001e+06 t328_leaf5 >= -1e+06
 T328_R5_f193_ge: const_Exterior_2nd_Wd_Shng - 1.000001e+06 t328_leaf5
   >= -1e+06
 TREE_329_ONEHOT: t329_leaf0 + t329_leaf1 + t329_leaf2 + t329_leaf3
   + t329_leaf4 + t329_leaf5 + t329_leaf6 = 1
 T329_L0_f3_le: Overall_Qual_calc + 999993 t329_leaf0 <= 1e+06
 T329_L0_f23_le: x_Full_Bath + 2 t329_leaf0 <= 4
 T329_L0_f15_le: const_Total_Bsmt_SF + 998611 t329_leaf0 <= 1e+06
 T329_L1_f3_le: Overall_Qual_calc + 999993 t329_leaf1 <= 1e+06
 T329_L1_f23_le: x_Full_Bath + 2 t329_leaf1 <= 4
 T329_R1_f15_ge: const_Total_Bsmt_SF - 1.001389e+06 t329_leaf1 >= -1e+06
 T329_L2_f3_le: Overall_Qual_calc + 999993 t329_leaf2 <= 1e+06
 T329_R2_f23_ge: x_Full_Bath - 2 t329_leaf2 >= 0
 T329_L2_f20_le: x_Gr_Liv_Area + 8983 t329_leaf2 <= 10000
 T329_L3_f3_le: Overall_Qual_calc + 999993 t329_leaf3 <= 1e+06
 T329_R3_f23_ge: x_Full_Bath - 2 t329_leaf3 >= 0
 T329_R3_f20_ge: x_Gr_Liv_Area - 1017 t329_leaf3 >= 0
 T329_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t329_leaf4 >= -1e+06
 T329_L4_f56_le: const_MS_SubClass_120 + 999999 t329_leaf4 <= 1e+06
 T329_L4_f0_le: const_Lot_Frontage + 999889 t329_leaf4 <= 1e+06
 T329_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t329_leaf5 >= -1e+06
 T329_L5_f56_le: const_MS_SubClass_120 + 999999 t329_leaf5 <= 1e+06
 T329_R5_f0_ge: const_Lot_Frontage - 1.000111e+06 t329_leaf5 >= -1e+06
 T329_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t329_leaf6 >= -1e+06
 T329_R6_f56_ge: const_MS_SubClass_120 - 1.000001e+06 t329_leaf6 >= -1e+06
 TREE_330_ONEHOT: t330_leaf0 + t330_leaf1 + t330_leaf2 + t330_leaf3
   + t330_leaf4 + t330_leaf5 = 1
 T330_L0_f1_le: const_Lot_Area + 981614 t330_leaf0 <= 1e+06
 T330_L0_f27_le: x_Kitchen_Qual + 2 t330_leaf0 <= 4
 T330_L0_f1_le: const_Lot_Area + 991480 t330_leaf0 <= 1e+06
 T330_L1_f1_le: const_Lot_Area + 981614 t330_leaf1 <= 1e+06
 T330_L1_f27_le: x_Kitchen_Qual + 2 t330_leaf1 <= 4
 T330_R1_f1_ge: const_Lot_Area - 1.00852e+06 t330_leaf1 >= -1e+06
 T330_L2_f1_le: const_Lot_Area + 981614 t330_leaf2 <= 1e+06
 T330_R2_f27_ge: x_Kitchen_Qual - 3 t330_leaf2 >= -1
 T330_L2_f63_le: const_MS_Zoning_FV + 999999 t330_leaf2 <= 1e+06
 T330_L3_f1_le: const_Lot_Area + 981614 t330_leaf3 <= 1e+06
 T330_R3_f27_ge: x_Kitchen_Qual - 3 t330_leaf3 >= -1
 T330_R3_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t330_leaf3 >= -1e+06
 T330_R4_f1_ge: const_Lot_Area - 1.018386e+06 t330_leaf4 >= -1e+06
 T330_L4_f38_le: x_Enclosed_Porch + 99891 t330_leaf4 <= 100000
 T330_R5_f1_ge: const_Lot_Area - 1.018386e+06 t330_leaf5 >= -1e+06
 T330_R5_f38_ge: x_Enclosed_Porch - 109 t330_leaf5 >= 0
 TREE_331_ONEHOT: t331_leaf0 + t331_leaf1 + t331_leaf2 + t331_leaf3
   + t331_leaf4 = 1
 T331_L0_f6_le: const_Year_Remod_Add + 997993 t331_leaf0 <= 1e+06
 T331_L0_f0_le: const_Lot_Frontage + 999870 t331_leaf0 <= 1e+06
 T331_L1_f6_le: const_Year_Remod_Add + 997993 t331_leaf1 <= 1e+06
 T331_R1_f0_ge: const_Lot_Frontage - 1.00013e+06 t331_leaf1 >= -1e+06
 T331_R2_f6_ge: const_Year_Remod_Add - 1.002007e+06 t331_leaf2 >= -1e+06
 T331_L2_f0_le: const_Lot_Frontage + 999886 t331_leaf2 <= 1e+06
 T331_L2_f192_le: x_ex2_is_Wd_Sdng <= 1
 T331_R3_f6_ge: const_Year_Remod_Add - 1.002007e+06 t331_leaf3 >= -1e+06
 T331_L3_f0_le: const_Lot_Frontage + 999886 t331_leaf3 <= 1e+06
 T331_R3_f192_ge: x_ex2_is_Wd_Sdng - t331_leaf3 >= 0
 T331_R4_f6_ge: const_Year_Remod_Add - 1.002007e+06 t331_leaf4 >= -1e+06
 T331_R4_f0_ge: const_Lot_Frontage - 1.000114e+06 t331_leaf4 >= -1e+06
 TREE_332_ONEHOT: t332_leaf0 + t332_leaf1 + t332_leaf2 + t332_leaf3 = 1
 T332_L0_f33_le: x_Garage_Area + 98846 t332_leaf0 <= 100000
 T332_L0_f1_le: const_Lot_Area + 995565 t332_leaf0 <= 1e+06
 T332_L1_f33_le: x_Garage_Area + 98846 t332_leaf1 <= 100000
 T332_R1_f1_ge: const_Lot_Area - 1.004435e+06 t332_leaf1 >= -1e+06
 T332_L1_f6_le: const_Year_Remod_Add + 997993 t332_leaf1 <= 1e+06
 T332_L2_f33_le: x_Garage_Area + 98846 t332_leaf2 <= 100000
 T332_R2_f1_ge: const_Lot_Area - 1.004435e+06 t332_leaf2 >= -1e+06
 T332_R2_f6_ge: const_Year_Remod_Add - 1.002007e+06 t332_leaf2 >= -1e+06
 T332_R3_f33_ge: x_Garage_Area - 1154 t332_leaf3 >= 0
 TREE_333_ONEHOT: t333_leaf0 + t333_leaf1 + t333_leaf2 = 1
 T333_L0_f161_le: x_ex1_is_AsbShng <= 1
 T333_R1_f161_ge: x_ex1_is_AsbShng - t333_leaf1 >= 0
 T333_L1_f12_le: x_BsmtFin_SF_1 + 999808 t333_leaf1 <= 1e+06
 T333_R2_f161_ge: x_ex1_is_AsbShng - t333_leaf2 >= 0
 T333_R2_f12_ge: x_BsmtFin_SF_1 - 192 t333_leaf2 >= 0
 TREE_334_ONEHOT: t334_leaf0 + t334_leaf1 + t334_leaf2 + t334_leaf3
   + t334_leaf4 + t334_leaf5 + t334_leaf6 = 1
 T334_L0_f161_le: x_ex1_is_AsbShng <= 1
 T334_L0_f4_le: const_Overall_Cond + 999994 t334_leaf0 <= 1e+06
 T334_L0_f77_le: const_Land_Contour_Bnk + 999999 t334_leaf0 <= 1e+06
 T334_L1_f161_le: x_ex1_is_AsbShng <= 1
 T334_L1_f4_le: const_Overall_Cond + 999994 t334_leaf1 <= 1e+06
 T334_R1_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t334_leaf1
   >= -1e+06
 T334_L2_f161_le: x_ex1_is_AsbShng <= 1
 T334_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t334_leaf2 >= -1e+06
 T334_L2_f18_le: const_2nd_Flr_SF + 999624 t334_leaf2 <= 1e+06
 T334_L3_f161_le: x_ex1_is_AsbShng <= 1
 T334_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t334_leaf3 >= -1e+06
 T334_R3_f18_ge: const_2nd_Flr_SF - 1.000376e+06 t334_leaf3 >= -1e+06
 T334_R4_f161_ge: x_ex1_is_AsbShng - t334_leaf4 >= 0
 T334_L4_f31_le: const_Garage_Yr_Blt + 998079 t334_leaf4 <= 1e+06
 T334_R5_f161_ge: x_ex1_is_AsbShng - t334_leaf5 >= 0
 T334_R5_f31_ge: const_Garage_Yr_Blt - 1.001921e+06 t334_leaf5 >= -1e+06
 T334_L5_f1_le: const_Lot_Area + 990395 t334_leaf5 <= 1e+06
 T334_R6_f161_ge: x_ex1_is_AsbShng - t334_leaf6 >= 0
 T334_R6_f31_ge: const_Garage_Yr_Blt - 1.001921e+06 t334_leaf6 >= -1e+06
 T334_R6_f1_ge: const_Lot_Area - 1.009605e+06 t334_leaf6 >= -1e+06
 TREE_335_ONEHOT: t335_leaf0 + t335_leaf1 + t335_leaf2 + t335_leaf3
   + t335_leaf4 + t335_leaf5 = 1
 T335_L0_f277_le: const_Mo_Sold_7 + 999999 t335_leaf0 <= 1e+06
 T335_L0_f24_le: x_Half_Bath + 4 t335_leaf0 <= 5
 T335_L0_f11_le: x_Bsmt_Cond + t335_leaf0 <= 4
 T335_L1_f277_le: const_Mo_Sold_7 + 999999 t335_leaf1 <= 1e+06
 T335_L1_f24_le: x_Half_Bath + 4 t335_leaf1 <= 5
 T335_R1_f11_ge: x_Bsmt_Cond - 4 t335_leaf1 >= -1
 T335_L2_f277_le: const_Mo_Sold_7 + 999999 t335_leaf2 <= 1e+06
 T335_R2_f24_ge: x_Half_Bath - t335_leaf2 >= 0
 T335_L2_f3_le: Overall_Qual_calc + 999993 t335_leaf2 <= 1e+06
 T335_L3_f277_le: const_Mo_Sold_7 + 999999 t335_leaf3 <= 1e+06
 T335_R3_f24_ge: x_Half_Bath - t335_leaf3 >= 0
 T335_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t335_leaf3 >= -1e+06
 T335_R4_f277_ge: const_Mo_Sold_7 - 1.000001e+06 t335_leaf4 >= -1e+06
 T335_L4_f1_le: const_Lot_Area + 988572 t335_leaf4 <= 1e+06
 T335_R5_f277_ge: const_Mo_Sold_7 - 1.000001e+06 t335_leaf5 >= -1e+06
 T335_R5_f1_ge: const_Lot_Area - 1.011428e+06 t335_leaf5 >= -1e+06
 TREE_336_ONEHOT: t336_leaf0 + t336_leaf1 + t336_leaf2 + t336_leaf3
   + t336_leaf4 + t336_leaf5 + t336_leaf6 + t336_leaf7 = 1
 T336_L0_f0_le: const_Lot_Frontage + 999915 t336_leaf0 <= 1e+06
 T336_L0_f213_le: x_b1_is_GLQ <= 1
 T336_L0_f35_le: x_Garage_Cond + 3 t336_leaf0 <= 4
 T336_L1_f0_le: const_Lot_Frontage + 999915 t336_leaf1 <= 1e+06
 T336_L1_f213_le: x_b1_is_GLQ <= 1
 T336_R1_f35_ge: x_Garage_Cond - 2 t336_leaf1 >= -1
 T336_L2_f0_le: const_Lot_Frontage + 999915 t336_leaf2 <= 1e+06
 T336_R2_f213_ge: x_b1_is_GLQ - t336_leaf2 >= 0
 T336_L2_f1_le: const_Lot_Area + 987565 t336_leaf2 <= 1e+06
 T336_L3_f0_le: const_Lot_Frontage + 999915 t336_leaf3 <= 1e+06
 T336_R3_f213_ge: x_b1_is_GLQ - t336_leaf3 >= 0
 T336_R3_f1_ge: const_Lot_Area - 1.012435e+06 t336_leaf3 >= -1e+06
 T336_R4_f0_ge: const_Lot_Frontage - 1.000085e+06 t336_leaf4 >= -1e+06
 T336_L4_f148_le: const_Roof_Style_Gable + 999999 t336_leaf4 <= 1e+06
 T336_L4_f86_le: const_Land_Slope_Gtl + 999999 t336_leaf4 <= 1e+06
 T336_R5_f0_ge: const_Lot_Frontage - 1.000085e+06 t336_leaf5 >= -1e+06
 T336_L5_f148_le: const_Roof_Style_Gable + 999999 t336_leaf5 <= 1e+06
 T336_R5_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t336_leaf5 >= -1e+06
 T336_R6_f0_ge: const_Lot_Frontage - 1.000085e+06 t336_leaf6 >= -1e+06
 T336_R6_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t336_leaf6
   >= -1e+06
 T336_L6_f1_le: const_Lot_Area + 981614 t336_leaf6 <= 1e+06
 T336_R7_f0_ge: const_Lot_Frontage - 1.000085e+06 t336_leaf7 >= -1e+06
 T336_R7_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t336_leaf7
   >= -1e+06
 T336_R7_f1_ge: const_Lot_Area - 1.018386e+06 t336_leaf7 >= -1e+06
 TREE_337_ONEHOT: t337_leaf0 + t337_leaf1 + t337_leaf2 + t337_leaf3
   + t337_leaf4 + t337_leaf5 = 1
 T337_L0_f211_le: x_b1_is_ALQ <= 1
 T337_L0_f5_le: const_Year_Built + 998004 t337_leaf0 <= 1e+06
 T337_L0_f14_le: x_Bsmt_Unf_SF + 99183 t337_leaf0 <= 100000
 T337_L1_f211_le: x_b1_is_ALQ <= 1
 T337_L1_f5_le: const_Year_Built + 998004 t337_leaf1 <= 1e+06
 T337_R1_f14_ge: x_Bsmt_Unf_SF - 817 t337_leaf1 >= 0
 T337_L2_f211_le: x_b1_is_ALQ <= 1
 T337_R2_f5_ge: const_Year_Built - 1.001996e+06 t337_leaf2 >= -1e+06
 T337_L2_f8_le: x_Exter_Qual + t337_leaf2 <= 4
 T337_L3_f211_le: x_b1_is_ALQ <= 1
 T337_R3_f5_ge: const_Year_Built - 1.001996e+06 t337_leaf3 >= -1e+06
 T337_R3_f8_ge: x_Exter_Qual - 4 t337_leaf3 >= -1
 T337_R4_f211_ge: x_b1_is_ALQ - t337_leaf4 >= 0
 T337_L4_f85_le: const_Lot_Config_Inside + 999999 t337_leaf4 <= 1e+06
 T337_R5_f211_ge: x_b1_is_ALQ - t337_leaf5 >= 0
 T337_R5_f85_ge: const_Lot_Config_Inside - 1.000001e+06 t337_leaf5
   >= -1e+06
 TREE_338_ONEHOT: t338_leaf0 + t338_leaf1 + t338_leaf2 + t338_leaf3
   + t338_leaf4 + t338_leaf5 + t338_leaf6 = 1
 T338_L0_f30_le: x_Fireplace_Qu + 2 t338_leaf0 <= 4
 T338_L0_f9_le: x_Exter_Cond + 2 t338_leaf0 <= 4
 T338_L0_f175_le: x_ex1_is_Wd_Sdng <= 1
 T338_L1_f30_le: x_Fireplace_Qu + 2 t338_leaf1 <= 4
 T338_L1_f9_le: x_Exter_Cond + 2 t338_leaf1 <= 4
 T338_R1_f175_ge: x_ex1_is_Wd_Sdng - t338_leaf1 >= 0
 T338_L2_f30_le: x_Fireplace_Qu + 2 t338_leaf2 <= 4
 T338_R2_f9_ge: x_Exter_Cond - 3 t338_leaf2 >= -1
 T338_L2_f22_le: const_Bsmt_Half_Bath + 999999 t338_leaf2 <= 1e+06
 T338_L3_f30_le: x_Fireplace_Qu + 2 t338_leaf3 <= 4
 T338_R3_f9_ge: x_Exter_Cond - 3 t338_leaf3 >= -1
 T338_R3_f22_ge: const_Bsmt_Half_Bath - 1.000001e+06 t338_leaf3 >= -1e+06
 T338_R4_f30_ge: x_Fireplace_Qu - 3 t338_leaf4 >= -1
 T338_L4_f20_le: x_Gr_Liv_Area + 9124 t338_leaf4 <= 10000
 T338_R5_f30_ge: x_Fireplace_Qu - 3 t338_leaf5 >= -1
 T338_R5_f20_ge: x_Gr_Liv_Area - 876 t338_leaf5 >= 0
 T338_L5_f14_le: x_Bsmt_Unf_SF + 99652 t338_leaf5 <= 100000
 T338_R6_f30_ge: x_Fireplace_Qu - 3 t338_leaf6 >= -1
 T338_R6_f20_ge: x_Gr_Liv_Area - 876 t338_leaf6 >= 0
 T338_R6_f14_ge: x_Bsmt_Unf_SF - 348 t338_leaf6 >= 0
 TREE_339_ONEHOT: t339_leaf0 + t339_leaf1 + t339_leaf2 + t339_leaf3
   + t339_leaf4 + t339_leaf5 = 1
 T339_L0_f296_le: const_Sale_Condition_Family + 999999 t339_leaf0 <= 1e+06
 T339_L0_f1_le: const_Lot_Area + 980478 t339_leaf0 <= 1e+06
 T339_L0_f109_le: const_Neighborhood_OldTown + 999999 t339_leaf0 <= 1e+06
 T339_L1_f296_le: const_Sale_Condition_Family + 999999 t339_leaf1 <= 1e+06
 T339_L1_f1_le: const_Lot_Area + 980478 t339_leaf1 <= 1e+06
 T339_R1_f109_ge: const_Neighborhood_OldTown - 1.000001e+06 t339_leaf1
   >= -1e+06
 T339_L2_f296_le: const_Sale_Condition_Family + 999999 t339_leaf2 <= 1e+06
 T339_R2_f1_ge: const_Lot_Area - 1.019522e+06 t339_leaf2 >= -1e+06
 T339_L2_f77_le: const_Land_Contour_Bnk + 999999 t339_leaf2 <= 1e+06
 T339_L3_f296_le: const_Sale_Condition_Family + 999999 t339_leaf3 <= 1e+06
 T339_R3_f1_ge: const_Lot_Area - 1.019522e+06 t339_leaf3 >= -1e+06
 T339_R3_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t339_leaf3
   >= -1e+06
 T339_R4_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t339_leaf4
   >= -1e+06
 T339_L4_f7_le: x_Mas_Vnr_Area + 1978 t339_leaf4 <= 2000
 T339_R5_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t339_leaf5
   >= -1e+06
 T339_R5_f7_ge: x_Mas_Vnr_Area - 22 t339_leaf5 >= 0
 TREE_340_ONEHOT: t340_leaf0 + t340_leaf1 + t340_leaf2 + t340_leaf3
   + t340_leaf4 + t340_leaf5 + t340_leaf6 = 1
 T340_L0_f5_le: const_Year_Built + 998048 t340_leaf0 <= 1e+06
 T340_L0_f5_le: const_Year_Built + 998085 t340_leaf0 <= 1e+06
 T340_L0_f10_le: const_Bsmt_Qual + 999998 t340_leaf0 <= 1e+06
 T340_L1_f5_le: const_Year_Built + 998048 t340_leaf1 <= 1e+06
 T340_L1_f5_le: const_Year_Built + 998085 t340_leaf1 <= 1e+06
 T340_R1_f10_ge: const_Bsmt_Qual - 1.000002e+06 t340_leaf1 >= -1e+06
 T340_L2_f5_le: const_Year_Built + 998048 t340_leaf2 <= 1e+06
 T340_R2_f5_ge: const_Year_Built - 1.001915e+06 t340_leaf2 >= -1e+06
 T340_L2_f117_le: const_Condition_1_Artery + 999999 t340_leaf2 <= 1e+06
 T340_L3_f5_le: const_Year_Built + 998048 t340_leaf3 <= 1e+06
 T340_R3_f5_ge: const_Year_Built - 1.001915e+06 t340_leaf3 >= -1e+06
 T340_R3_f117_ge: const_Condition_1_Artery - 1.000001e+06 t340_leaf3
   >= -1e+06
 T340_R4_f5_ge: const_Year_Built - 1.001952e+06 t340_leaf4 >= -1e+06
 T340_L4_f6_le: const_Year_Remod_Add + 998047 t340_leaf4 <= 1e+06
 T340_R5_f5_ge: const_Year_Built - 1.001952e+06 t340_leaf5 >= -1e+06
 T340_R5_f6_ge: const_Year_Remod_Add - 1.001953e+06 t340_leaf5 >= -1e+06
 T340_L5_f296_le: const_Sale_Condition_Family + 999999 t340_leaf5 <= 1e+06
 T340_R6_f5_ge: const_Year_Built - 1.001952e+06 t340_leaf6 >= -1e+06
 T340_R6_f6_ge: const_Year_Remod_Add - 1.001953e+06 t340_leaf6 >= -1e+06
 T340_R6_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t340_leaf6
   >= -1e+06
 TREE_341_ONEHOT: t341_leaf0 + t341_leaf1 + t341_leaf2 = 1
 T341_L0_f296_le: const_Sale_Condition_Family + 999999 t341_leaf0 <= 1e+06
 T341_R1_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t341_leaf1
   >= -1e+06
 T341_L1_f33_le: x_Garage_Area + 99494 t341_leaf1 <= 100000
 T341_R2_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t341_leaf2
   >= -1e+06
 T341_R2_f33_ge: x_Garage_Area - 506 t341_leaf2 >= 0
 TREE_342_ONEHOT: t342_leaf0 + t342_leaf1 + t342_leaf2 + t342_leaf3
   + t342_leaf4 = 1
 T342_L0_f71_le: const_Alley_No_aplica + 999999 t342_leaf0 <= 1e+06
 T342_R1_f71_ge: const_Alley_No_aplica - 1.000001e+06 t342_leaf1 >= -1e+06
 T342_L1_f14_le: x_Bsmt_Unf_SF + 99571 t342_leaf1 <= 100000
 T342_L1_f161_le: x_ex1_is_AsbShng <= 1
 T342_R2_f71_ge: const_Alley_No_aplica - 1.000001e+06 t342_leaf2 >= -1e+06
 T342_L2_f14_le: x_Bsmt_Unf_SF + 99571 t342_leaf2 <= 100000
 T342_R2_f161_ge: x_ex1_is_AsbShng - t342_leaf2 >= 0
 T342_R3_f71_ge: const_Alley_No_aplica - 1.000001e+06 t342_leaf3 >= -1e+06
 T342_R3_f14_ge: x_Bsmt_Unf_SF - 429 t342_leaf3 >= 0
 T342_L3_f96_le: const_Neighborhood_Edwards + 999999 t342_leaf3 <= 1e+06
 T342_R4_f71_ge: const_Alley_No_aplica - 1.000001e+06 t342_leaf4 >= -1e+06
 T342_R4_f14_ge: x_Bsmt_Unf_SF - 429 t342_leaf4 >= 0
 T342_R4_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t342_leaf4
   >= -1e+06
 TREE_343_ONEHOT: t343_leaf0 + t343_leaf1 + t343_leaf2 + t343_leaf3
   + t343_leaf4 = 1
 T343_L0_f203_le: const_Foundation_Slab + 999999 t343_leaf0 <= 1e+06
 T343_L0_f12_le: x_BsmtFin_SF_1 + 998035 t343_leaf0 <= 1e+06
 T343_L0_f1_le: const_Lot_Area + 995000 t343_leaf0 <= 1e+06
 T343_L1_f203_le: const_Foundation_Slab + 999999 t343_leaf1 <= 1e+06
 T343_L1_f12_le: x_BsmtFin_SF_1 + 998035 t343_leaf1 <= 1e+06
 T343_R1_f1_ge: const_Lot_Area - 1.005e+06 t343_leaf1 >= -1e+06
 T343_L2_f203_le: const_Foundation_Slab + 999999 t343_leaf2 <= 1e+06
 T343_R2_f12_ge: x_BsmtFin_SF_1 - 1965 t343_leaf2 >= 0
 T343_R3_f203_ge: const_Foundation_Slab - 1.000001e+06 t343_leaf3
   >= -1e+06
 T343_L3_f0_le: const_Lot_Frontage + 999934 t343_leaf3 <= 1e+06
 T343_R4_f203_ge: const_Foundation_Slab - 1.000001e+06 t343_leaf4
   >= -1e+06
 T343_R4_f0_ge: const_Lot_Frontage - 1.000066e+06 t343_leaf4 >= -1e+06
 TREE_344_ONEHOT: t344_leaf0 + t344_leaf1 + t344_leaf2 + t344_leaf3
   + t344_leaf4 + t344_leaf5 + t344_leaf6 = 1
 T344_L0_f3_le: Overall_Qual_calc + 999995 t344_leaf0 <= 1e+06
 T344_L0_f272_le: const_Mo_Sold_2 + 999999 t344_leaf0 <= 1e+06
 T344_L0_f1_le: const_Lot_Area + 994024 t344_leaf0 <= 1e+06
 T344_L1_f3_le: Overall_Qual_calc + 999995 t344_leaf1 <= 1e+06
 T344_L1_f272_le: const_Mo_Sold_2 + 999999 t344_leaf1 <= 1e+06
 T344_R1_f1_ge: const_Lot_Area - 1.005976e+06 t344_leaf1 >= -1e+06
 T344_L2_f3_le: Overall_Qual_calc + 999995 t344_leaf2 <= 1e+06
 T344_R2_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t344_leaf2 >= -1e+06
 T344_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t344_leaf3 >= -1e+06
 T344_L3_f12_le: x_BsmtFin_SF_1 + 999064 t344_leaf3 <= 1e+06
 T344_L3_f1_le: const_Lot_Area + 991600 t344_leaf3 <= 1e+06
 T344_R4_f3_ge: Overall_Qual_calc - 1.000005e+06 t344_leaf4 >= -1e+06
 T344_L4_f12_le: x_BsmtFin_SF_1 + 999064 t344_leaf4 <= 1e+06
 T344_R4_f1_ge: const_Lot_Area - 1.0084e+06 t344_leaf4 >= -1e+06
 T344_R5_f3_ge: Overall_Qual_calc - 1.000005e+06 t344_leaf5 >= -1e+06
 T344_R5_f12_ge: x_BsmtFin_SF_1 - 936 t344_leaf5 >= 0
 T344_L5_f1_le: const_Lot_Area + 978630 t344_leaf5 <= 1e+06
 T344_R6_f3_ge: Overall_Qual_calc - 1.000005e+06 t344_leaf6 >= -1e+06
 T344_R6_f12_ge: x_BsmtFin_SF_1 - 936 t344_leaf6 >= 0
 T344_R6_f1_ge: const_Lot_Area - 1.02137e+06 t344_leaf6 >= -1e+06
 TREE_345_ONEHOT: t345_leaf0 + t345_leaf1 + t345_leaf2 + t345_leaf3
   + t345_leaf4 + t345_leaf5 + t345_leaf6 = 1
 T345_L0_f16_le: x_Heating_QC + t345_leaf0 <= 4
 T345_L0_f297_le: const_Sale_Condition_Normal + 999999 t345_leaf0 <= 1e+06
 T345_L0_f209_le: const_Bsmt_Exposure_No + 999999 t345_leaf0 <= 1e+06
 T345_L1_f16_le: x_Heating_QC + t345_leaf1 <= 4
 T345_L1_f297_le: const_Sale_Condition_Normal + 999999 t345_leaf1 <= 1e+06
 T345_R1_f209_ge: const_Bsmt_Exposure_No - 1.000001e+06 t345_leaf1
   >= -1e+06
 T345_L2_f16_le: x_Heating_QC + t345_leaf2 <= 4
 T345_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t345_leaf2
   >= -1e+06
 T345_L2_f3_le: Overall_Qual_calc + 999996 t345_leaf2 <= 1e+06
 T345_L3_f16_le: x_Heating_QC + t345_leaf3 <= 4
 T345_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t345_leaf3
   >= -1e+06
 T345_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t345_leaf3 >= -1e+06
 T345_R4_f16_ge: x_Heating_QC - 4 t345_leaf4 >= -1
 T345_L4_f3_le: Overall_Qual_calc + 999994 t345_leaf4 <= 1e+06
 T345_R5_f16_ge: x_Heating_QC - 4 t345_leaf5 >= -1
 T345_R5_f3_ge: Overall_Qual_calc - 1.000006e+06 t345_leaf5 >= -1e+06
 T345_L5_f96_le: const_Neighborhood_Edwards + 999999 t345_leaf5 <= 1e+06
 T345_R6_f16_ge: x_Heating_QC - 4 t345_leaf6 >= -1
 T345_R6_f3_ge: Overall_Qual_calc - 1.000006e+06 t345_leaf6 >= -1e+06
 T345_R6_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t345_leaf6
   >= -1e+06
 TREE_346_ONEHOT: t346_leaf0 + t346_leaf1 + t346_leaf2 + t346_leaf3
   + t346_leaf4 = 1
 T346_L0_f7_le: x_Mas_Vnr_Area + 1748 t346_leaf0 <= 2000
 T346_L0_f8_le: x_Exter_Qual + t346_leaf0 <= 4
 T346_L0_f16_le: x_Heating_QC + 2 t346_leaf0 <= 4
 T346_L1_f7_le: x_Mas_Vnr_Area + 1748 t346_leaf1 <= 2000
 T346_L1_f8_le: x_Exter_Qual + t346_leaf1 <= 4
 T346_R1_f16_ge: x_Heating_QC - 3 t346_leaf1 >= -1
 T346_L2_f7_le: x_Mas_Vnr_Area + 1748 t346_leaf2 <= 2000
 T346_R2_f8_ge: x_Exter_Qual - 4 t346_leaf2 >= -1
 T346_L2_f17_le: x_1st_Flr_SF + 3218 t346_leaf2 <= 5000
 T346_L3_f7_le: x_Mas_Vnr_Area + 1748 t346_leaf3 <= 2000
 T346_R3_f8_ge: x_Exter_Qual - 4 t346_leaf3 >= -1
 T346_R3_f17_ge: x_1st_Flr_SF - 1782 t346_leaf3 >= 0
 T346_R4_f7_ge: x_Mas_Vnr_Area - 252 t346_leaf4 >= 0
 TREE_347_ONEHOT: t347_leaf0 + t347_leaf1 + t347_leaf2 + t347_leaf3 = 1
 T347_L0_f38_le: x_Enclosed_Porch + 99710 t347_leaf0 <= 100000
 T347_L0_f0_le: const_Lot_Frontage + 999870 t347_leaf0 <= 1e+06
 T347_L0_f121_le: const_Condition_1_PosN + 999999 t347_leaf0 <= 1e+06
 T347_L1_f38_le: x_Enclosed_Porch + 99710 t347_leaf1 <= 100000
 T347_L1_f0_le: const_Lot_Frontage + 999870 t347_leaf1 <= 1e+06
 T347_R1_f121_ge: const_Condition_1_PosN - 1.000001e+06 t347_leaf1
   >= -1e+06
 T347_L2_f38_le: x_Enclosed_Porch + 99710 t347_leaf2 <= 100000
 T347_R2_f0_ge: const_Lot_Frontage - 1.00013e+06 t347_leaf2 >= -1e+06
 T347_R3_f38_ge: x_Enclosed_Porch - 290 t347_leaf3 >= 0
 TREE_348_ONEHOT: t348_leaf0 + t348_leaf1 + t348_leaf2 + t348_leaf3
   + t348_leaf4 = 1
 T348_L0_f40_le: x_Screen_Porch + 99892 t348_leaf0 <= 100000
 T348_L0_f7_le: x_Mas_Vnr_Area + 1541 t348_leaf0 <= 2000
 T348_L0_f44_le: const_Yr_Sold + 997993 t348_leaf0 <= 1e+06
 T348_L1_f40_le: x_Screen_Porch + 99892 t348_leaf1 <= 100000
 T348_L1_f7_le: x_Mas_Vnr_Area + 1541 t348_leaf1 <= 2000
 T348_R1_f44_ge: const_Yr_Sold - 1.002007e+06 t348_leaf1 >= -1e+06
 T348_L2_f40_le: x_Screen_Porch + 99892 t348_leaf2 <= 100000
 T348_R2_f7_ge: x_Mas_Vnr_Area - 459 t348_leaf2 >= 0
 T348_R3_f40_ge: x_Screen_Porch - 108 t348_leaf3 >= 0
 T348_L3_f217_le: x_b1_is_Unf <= 1
 T348_R4_f40_ge: x_Screen_Porch - 108 t348_leaf4 >= 0
 T348_R4_f217_ge: x_b1_is_Unf - t348_leaf4 >= 0
 TREE_349_ONEHOT: t349_leaf0 + t349_leaf1 + t349_leaf2 + t349_leaf3
   + t349_leaf4 = 1
 T349_L0_f0_le: const_Lot_Frontage + 999880 t349_leaf0 <= 1e+06
 T349_L0_f17_le: x_1st_Flr_SF + 2879 t349_leaf0 <= 5000
 T349_L0_f38_le: x_Enclosed_Porch + 99735 t349_leaf0 <= 100000
 T349_L1_f0_le: const_Lot_Frontage + 999880 t349_leaf1 <= 1e+06
 T349_L1_f17_le: x_1st_Flr_SF + 2879 t349_leaf1 <= 5000
 T349_R1_f38_ge: x_Enclosed_Porch - 265 t349_leaf1 >= 0
 T349_L2_f0_le: const_Lot_Frontage + 999880 t349_leaf2 <= 1e+06
 T349_R2_f17_ge: x_1st_Flr_SF - 2121 t349_leaf2 >= 0
 T349_R3_f0_ge: const_Lot_Frontage - 1.00012e+06 t349_leaf3 >= -1e+06
 T349_L3_f77_le: const_Land_Contour_Bnk + 999999 t349_leaf3 <= 1e+06
 T349_R4_f0_ge: const_Lot_Frontage - 1.00012e+06 t349_leaf4 >= -1e+06
 T349_R4_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t349_leaf4
   >= -1e+06
 TREE_350_ONEHOT: t350_leaf0 + t350_leaf1 + t350_leaf2 + t350_leaf3
   + t350_leaf4 + t350_leaf5 + t350_leaf6 = 1
 T350_L0_f9_le: x_Exter_Cond + 2 t350_leaf0 <= 4
 T350_L0_f0_le: const_Lot_Frontage + 999934 t350_leaf0 <= 1e+06
 T350_L1_f9_le: x_Exter_Cond + 2 t350_leaf1 <= 4
 T350_R1_f0_ge: const_Lot_Frontage - 1.000066e+06 t350_leaf1 >= -1e+06
 T350_L1_f18_le: const_2nd_Flr_SF + 999833 t350_leaf1 <= 1e+06
 T350_L2_f9_le: x_Exter_Cond + 2 t350_leaf2 <= 4
 T350_R2_f0_ge: const_Lot_Frontage - 1.000066e+06 t350_leaf2 >= -1e+06
 T350_R2_f18_ge: const_2nd_Flr_SF - 1.000167e+06 t350_leaf2 >= -1e+06
 T350_R3_f9_ge: x_Exter_Cond - 3 t350_leaf3 >= -1
 T350_L3_f31_le: const_Garage_Yr_Blt + 998059 t350_leaf3 <= 1e+06
 T350_L3_f14_le: x_Bsmt_Unf_SF + 99617 t350_leaf3 <= 100000
 T350_R4_f9_ge: x_Exter_Cond - 3 t350_leaf4 >= -1
 T350_L4_f31_le: const_Garage_Yr_Blt + 998059 t350_leaf4 <= 1e+06
 T350_R4_f14_ge: x_Bsmt_Unf_SF - 383 t350_leaf4 >= 0
 T350_R5_f9_ge: x_Exter_Cond - 3 t350_leaf5 >= -1
 T350_R5_f31_ge: const_Garage_Yr_Blt - 1.001941e+06 t350_leaf5 >= -1e+06
 T350_L5_f14_le: x_Bsmt_Unf_SF + 99524 t350_leaf5 <= 100000
 T350_R6_f9_ge: x_Exter_Cond - 3 t350_leaf6 >= -1
 T350_R6_f31_ge: const_Garage_Yr_Blt - 1.001941e+06 t350_leaf6 >= -1e+06
 T350_R6_f14_ge: x_Bsmt_Unf_SF - 476 t350_leaf6 >= 0
 TREE_351_ONEHOT: t351_leaf0 + t351_leaf1 + t351_leaf2 + t351_leaf3
   + t351_leaf4 + t351_leaf5 = 1
 T351_L0_f44_le: const_Yr_Sold + 997990 t351_leaf0 <= 1e+06
 T351_L0_f3_le: Overall_Qual_calc + 999995 t351_leaf0 <= 1e+06
 T351_L1_f44_le: const_Yr_Sold + 997990 t351_leaf1 <= 1e+06
 T351_R1_f3_ge: Overall_Qual_calc - 1.000005e+06 t351_leaf1 >= -1e+06
 T351_L1_f60_le: const_MS_SubClass_190 + 999999 t351_leaf1 <= 1e+06
 T351_L2_f44_le: const_Yr_Sold + 997990 t351_leaf2 <= 1e+06
 T351_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t351_leaf2 >= -1e+06
 T351_R2_f60_ge: const_MS_SubClass_190 - 1.000001e+06 t351_leaf2 >= -1e+06
 T351_R3_f44_ge: const_Yr_Sold - 1.00201e+06 t351_leaf3 >= -1e+06
 T351_L3_f7_le: x_Mas_Vnr_Area + 1676 t351_leaf3 <= 2000
 T351_L3_f15_le: const_Total_Bsmt_SF + 999470 t351_leaf3 <= 1e+06
 T351_R4_f44_ge: const_Yr_Sold - 1.00201e+06 t351_leaf4 >= -1e+06
 T351_L4_f7_le: x_Mas_Vnr_Area + 1676 t351_leaf4 <= 2000
 T351_R4_f15_ge: const_Total_Bsmt_SF - 1.00053e+06 t351_leaf4 >= -1e+06
 T351_R5_f44_ge: const_Yr_Sold - 1.00201e+06 t351_leaf5 >= -1e+06
 T351_R5_f7_ge: x_Mas_Vnr_Area - 324 t351_leaf5 >= 0
 TREE_352_ONEHOT: t352_leaf0 + t352_leaf1 = 1
 T352_L0_f37_le: x_Open_Porch_SF + 99811 t352_leaf0 <= 100000
 T352_R1_f37_ge: x_Open_Porch_SF - 189 t352_leaf1 >= 0
 TREE_353_ONEHOT: t353_leaf0 + t353_leaf1 + t353_leaf2 + t353_leaf3
   + t353_leaf4 = 1
 T353_L0_f37_le: x_Open_Porch_SF + 99713 t353_leaf0 <= 100000
 T353_L0_f20_le: x_Gr_Liv_Area + 6914 t353_leaf0 <= 10000
 T353_L0_f147_le: const_Roof_Style_Flat + 999999 t353_leaf0 <= 1e+06
 T353_L1_f37_le: x_Open_Porch_SF + 99713 t353_leaf1 <= 100000
 T353_L1_f20_le: x_Gr_Liv_Area + 6914 t353_leaf1 <= 10000
 T353_R1_f147_ge: const_Roof_Style_Flat - 1.000001e+06 t353_leaf1
   >= -1e+06
 T353_L2_f37_le: x_Open_Porch_SF + 99713 t353_leaf2 <= 100000
 T353_R2_f20_ge: x_Gr_Liv_Area - 3086 t353_leaf2 >= 0
 T353_R3_f37_ge: x_Open_Porch_SF - 287 t353_leaf3 >= 0
 T353_L3_f28_le: x_TotRms_AbvGrd + 5 t353_leaf3 <= 15
 T353_R4_f37_ge: x_Open_Porch_SF - 287 t353_leaf4 >= 0
 T353_R4_f28_ge: x_TotRms_AbvGrd - 10 t353_leaf4 >= 0
 TREE_354_ONEHOT: t354_leaf0 + t354_leaf1 + t354_leaf2 = 1
 T354_L0_f5_le: const_Year_Built + 998089 t354_leaf0 <= 1e+06
 T354_L0_f17_le: x_1st_Flr_SF + 4140 t354_leaf0 <= 5000
 T354_L1_f5_le: const_Year_Built + 998089 t354_leaf1 <= 1e+06
 T354_R1_f17_ge: x_1st_Flr_SF - 860 t354_leaf1 >= 0
 T354_R2_f5_ge: const_Year_Built - 1.001911e+06 t354_leaf2 >= -1e+06
 TREE_355_ONEHOT: t355_leaf0 + t355_leaf1 + t355_leaf2 + t355_leaf3
   + t355_leaf4 + t355_leaf5 + t355_leaf6 = 1
 T355_L0_f251_le: const_Garage_Type_Detchd + 999999 t355_leaf0 <= 1e+06
 T355_L0_f15_le: const_Total_Bsmt_SF + 999352 t355_leaf0 <= 1e+06
 T355_L0_f31_le: const_Garage_Yr_Blt + 998005 t355_leaf0 <= 1e+06
 T355_L1_f251_le: const_Garage_Type_Detchd + 999999 t355_leaf1 <= 1e+06
 T355_L1_f15_le: const_Total_Bsmt_SF + 999352 t355_leaf1 <= 1e+06
 T355_R1_f31_ge: const_Garage_Yr_Blt - 1.001995e+06 t355_leaf1 >= -1e+06
 T355_L2_f251_le: const_Garage_Type_Detchd + 999999 t355_leaf2 <= 1e+06
 T355_R2_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t355_leaf2 >= -1e+06
 T355_L2_f15_le: const_Total_Bsmt_SF + 999279 t355_leaf2 <= 1e+06
 T355_L3_f251_le: const_Garage_Type_Detchd + 999999 t355_leaf3 <= 1e+06
 T355_R3_f15_ge: const_Total_Bsmt_SF - 1.000648e+06 t355_leaf3 >= -1e+06
 T355_R3_f15_ge: const_Total_Bsmt_SF - 1.000721e+06 t355_leaf3 >= -1e+06
 T355_R4_f251_ge: const_Garage_Type_Detchd - 1.000001e+06 t355_leaf4
   >= -1e+06
 T355_L4_f1_le: const_Lot_Area + 988975 t355_leaf4 <= 1e+06
 T355_R5_f251_ge: const_Garage_Type_Detchd - 1.000001e+06 t355_leaf5
   >= -1e+06
 T355_R5_f1_ge: const_Lot_Area - 1.011025e+06 t355_leaf5 >= -1e+06
 T355_L5_f292_le: const_Sale_Type_WD_ + 999999 t355_leaf5 <= 1e+06
 T355_R6_f251_ge: const_Garage_Type_Detchd - 1.000001e+06 t355_leaf6
   >= -1e+06
 T355_R6_f1_ge: const_Lot_Area - 1.011025e+06 t355_leaf6 >= -1e+06
 T355_R6_f292_ge: const_Sale_Type_WD_ - 1.000001e+06 t355_leaf6 >= -1e+06
 TREE_356_ONEHOT: t356_leaf0 + t356_leaf1 + t356_leaf2 + t356_leaf3
   + t356_leaf4 + t356_leaf5 + t356_leaf6 = 1
 T356_L0_f4_le: const_Overall_Cond + 999995 t356_leaf0 <= 1e+06
 T356_L0_f15_le: const_Total_Bsmt_SF + 999273 t356_leaf0 <= 1e+06
 T356_L0_f17_le: x_1st_Flr_SF + 4384 t356_leaf0 <= 5000
 T356_L1_f4_le: const_Overall_Cond + 999995 t356_leaf1 <= 1e+06
 T356_L1_f15_le: const_Total_Bsmt_SF + 999273 t356_leaf1 <= 1e+06
 T356_R1_f17_ge: x_1st_Flr_SF - 616 t356_leaf1 >= 0
 T356_L2_f4_le: const_Overall_Cond + 999995 t356_leaf2 <= 1e+06
 T356_R2_f15_ge: const_Total_Bsmt_SF - 1.000727e+06 t356_leaf2 >= -1e+06
 T356_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t356_leaf3 >= -1e+06
 T356_L3_f18_le: const_2nd_Flr_SF + 999166 t356_leaf3 <= 1e+06
 T356_L3_f72_le: const_Alley_Pave + 999999 t356_leaf3 <= 1e+06
 T356_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t356_leaf4 >= -1e+06
 T356_L4_f18_le: const_2nd_Flr_SF + 999166 t356_leaf4 <= 1e+06
 T356_R4_f72_ge: const_Alley_Pave - 1.000001e+06 t356_leaf4 >= -1e+06
 T356_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t356_leaf5 >= -1e+06
 T356_R5_f18_ge: const_2nd_Flr_SF - 1.000834e+06 t356_leaf5 >= -1e+06
 T356_L5_f3_le: Overall_Qual_calc + 999993 t356_leaf5 <= 1e+06
 T356_R6_f4_ge: const_Overall_Cond - 1.000005e+06 t356_leaf6 >= -1e+06
 T356_R6_f18_ge: const_2nd_Flr_SF - 1.000834e+06 t356_leaf6 >= -1e+06
 T356_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t356_leaf6 >= -1e+06
 TREE_357_ONEHOT: t357_leaf0 + t357_leaf1 + t357_leaf2 + t357_leaf3 = 1
 T357_L0_f273_le: const_Mo_Sold_3 + 999999 t357_leaf0 <= 1e+06
 T357_L0_f95_le: const_Neighborhood_Crawfor + 999999 t357_leaf0 <= 1e+06
 T357_L0_f68_le: const_Street_Grvl + 999999 t357_leaf0 <= 1e+06
 T357_L1_f273_le: const_Mo_Sold_3 + 999999 t357_leaf1 <= 1e+06
 T357_L1_f95_le: const_Neighborhood_Crawfor + 999999 t357_leaf1 <= 1e+06
 T357_R1_f68_ge: const_Street_Grvl - 1.000001e+06 t357_leaf1 >= -1e+06
 T357_L2_f273_le: const_Mo_Sold_3 + 999999 t357_leaf2 <= 1e+06
 T357_R2_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t357_leaf2
   >= -1e+06
 T357_R3_f273_ge: const_Mo_Sold_3 - 1.000001e+06 t357_leaf3 >= -1e+06
 TREE_358_ONEHOT: t358_leaf0 + t358_leaf1 + t358_leaf2 + t358_leaf3
   + t358_leaf4 + t358_leaf5 + t358_leaf6 = 1
 T358_L0_f231_le: central_air_no + 999999 t358_leaf0 <= 1e+06
 T358_L0_f82_le: const_Lot_Config_CulDSac + 999999 t358_leaf0 <= 1e+06
 T358_L0_f17_le: x_1st_Flr_SF + 3547 t358_leaf0 <= 5000
 T358_L1_f231_le: central_air_no + 999999 t358_leaf1 <= 1e+06
 T358_L1_f82_le: const_Lot_Config_CulDSac + 999999 t358_leaf1 <= 1e+06
 T358_R1_f17_ge: x_1st_Flr_SF - 1453 t358_leaf1 >= 0
 T358_L2_f231_le: central_air_no + 999999 t358_leaf2 <= 1e+06
 T358_R2_f82_ge: const_Lot_Config_CulDSac - 1.000001e+06 t358_leaf2
   >= -1e+06
 T358_R3_f231_ge: central_air_no - 1.000001e+06 t358_leaf3 >= -1e+06
 T358_L3_f4_le: const_Overall_Cond + 999995 t358_leaf3 <= 1e+06
 T358_L3_f17_le: x_1st_Flr_SF + 4064 t358_leaf3 <= 5000
 T358_R4_f231_ge: central_air_no - 1.000001e+06 t358_leaf4 >= -1e+06
 T358_L4_f4_le: const_Overall_Cond + 999995 t358_leaf4 <= 1e+06
 T358_R4_f17_ge: x_1st_Flr_SF - 936 t358_leaf4 >= 0
 T358_R5_f231_ge: central_air_no - 1.000001e+06 t358_leaf5 >= -1e+06
 T358_R5_f4_ge: const_Overall_Cond - 1.000005e+06 t358_leaf5 >= -1e+06
 T358_L5_f92_le: const_Neighborhood_BrkSide + 999999 t358_leaf5 <= 1e+06
 T358_R6_f231_ge: central_air_no - 1.000001e+06 t358_leaf6 >= -1e+06
 T358_R6_f4_ge: const_Overall_Cond - 1.000005e+06 t358_leaf6 >= -1e+06
 T358_R6_f92_ge: const_Neighborhood_BrkSide - 1.000001e+06 t358_leaf6
   >= -1e+06
 TREE_359_ONEHOT: t359_leaf0 + t359_leaf1 + t359_leaf2 + t359_leaf3
   + t359_leaf4 + t359_leaf5 + t359_leaf6 = 1
 T359_L0_f161_le: x_ex1_is_AsbShng <= 1
 T359_L0_f114_le: const_Neighborhood_StoneBr + 999999 t359_leaf0 <= 1e+06
 T359_L0_f8_le: x_Exter_Qual + t359_leaf0 <= 4
 T359_L1_f161_le: x_ex1_is_AsbShng <= 1
 T359_L1_f114_le: const_Neighborhood_StoneBr + 999999 t359_leaf1 <= 1e+06
 T359_R1_f8_ge: x_Exter_Qual - 4 t359_leaf1 >= -1
 T359_L2_f161_le: x_ex1_is_AsbShng <= 1
 T359_R2_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t359_leaf2
   >= -1e+06
 T359_L2_f15_le: const_Total_Bsmt_SF + 998600 t359_leaf2 <= 1e+06
 T359_L3_f161_le: x_ex1_is_AsbShng <= 1
 T359_R3_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t359_leaf3
   >= -1e+06
 T359_R3_f15_ge: const_Total_Bsmt_SF - 1.0014e+06 t359_leaf3 >= -1e+06
 T359_R4_f161_ge: x_ex1_is_AsbShng - t359_leaf4 >= 0
 T359_L4_f33_le: x_Garage_Area + 99750 t359_leaf4 <= 100000
 T359_R5_f161_ge: x_ex1_is_AsbShng - t359_leaf5 >= 0
 T359_R5_f33_ge: x_Garage_Area - 250 t359_leaf5 >= 0
 T359_L5_f15_le: const_Total_Bsmt_SF + 999209 t359_leaf5 <= 1e+06
 T359_R6_f161_ge: x_ex1_is_AsbShng - t359_leaf6 >= 0
 T359_R6_f33_ge: x_Garage_Area - 250 t359_leaf6 >= 0
 T359_R6_f15_ge: const_Total_Bsmt_SF - 1.000791e+06 t359_leaf6 >= -1e+06
 TREE_360_ONEHOT: t360_leaf0 + t360_leaf1 = 1
 T360_L0_f3_le: Overall_Qual_calc + 999997 t360_leaf0 <= 1e+06
 T360_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t360_leaf1 >= -1e+06
 TREE_361_ONEHOT: t361_leaf0 + t361_leaf1 + t361_leaf2 + t361_leaf3 = 1
 T361_L0_f173_le: x_ex1_is_Stucco <= 1
 T361_L0_f17_le: x_1st_Flr_SF + 4354 t361_leaf0 <= 5000
 T361_L1_f173_le: x_ex1_is_Stucco <= 1
 T361_R1_f17_ge: x_1st_Flr_SF - 646 t361_leaf1 >= 0
 T361_L1_f33_le: x_Garage_Area + 99703 t361_leaf1 <= 100000
 T361_L2_f173_le: x_ex1_is_Stucco <= 1
 T361_R2_f17_ge: x_1st_Flr_SF - 646 t361_leaf2 >= 0
 T361_R2_f33_ge: x_Garage_Area - 297 t361_leaf2 >= 0
 T361_R3_f173_ge: x_ex1_is_Stucco - t361_leaf3 >= 0
 TREE_362_ONEHOT: t362_leaf0 + t362_leaf1 + t362_leaf2 + t362_leaf3
   + t362_leaf4 = 1
 T362_L0_f5_le: const_Year_Built + 998085 t362_leaf0 <= 1e+06
 T362_L0_f4_le: const_Overall_Cond + 999993 t362_leaf0 <= 1e+06
 T362_L1_f5_le: const_Year_Built + 998085 t362_leaf1 <= 1e+06
 T362_R1_f4_ge: const_Overall_Cond - 1.000007e+06 t362_leaf1 >= -1e+06
 T362_R2_f5_ge: const_Year_Built - 1.001915e+06 t362_leaf2 >= -1e+06
 T362_L2_f18_le: const_2nd_Flr_SF + 999642 t362_leaf2 <= 1e+06
 T362_R3_f5_ge: const_Year_Built - 1.001915e+06 t362_leaf3 >= -1e+06
 T362_R3_f18_ge: const_2nd_Flr_SF - 1.000358e+06 t362_leaf3 >= -1e+06
 T362_L3_f3_le: Overall_Qual_calc + 999993 t362_leaf3 <= 1e+06
 T362_R4_f5_ge: const_Year_Built - 1.001915e+06 t362_leaf4 >= -1e+06
 T362_R4_f18_ge: const_2nd_Flr_SF - 1.000358e+06 t362_leaf4 >= -1e+06
 T362_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t362_leaf4 >= -1e+06
 TREE_363_ONEHOT: t363_leaf0 + t363_leaf1 + t363_leaf2 + t363_leaf3
   + t363_leaf4 = 1
 T363_L0_f296_le: const_Sale_Condition_Family + 999999 t363_leaf0 <= 1e+06
 T363_L0_f14_le: x_Bsmt_Unf_SF + 98607 t363_leaf0 <= 100000
 T363_L0_f216_le: x_b1_is_Rec <= 1
 T363_L1_f296_le: const_Sale_Condition_Family + 999999 t363_leaf1 <= 1e+06
 T363_L1_f14_le: x_Bsmt_Unf_SF + 98607 t363_leaf1 <= 100000
 T363_R1_f216_ge: x_b1_is_Rec - t363_leaf1 >= 0
 T363_L2_f296_le: const_Sale_Condition_Family + 999999 t363_leaf2 <= 1e+06
 T363_R2_f14_ge: x_Bsmt_Unf_SF - 1393 t363_leaf2 >= 0
 T363_R3_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t363_leaf3
   >= -1e+06
 T363_L3_f14_le: x_Bsmt_Unf_SF + 99493 t363_leaf3 <= 100000
 T363_R4_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t363_leaf4
   >= -1e+06
 T363_R4_f14_ge: x_Bsmt_Unf_SF - 507 t363_leaf4 >= 0
 TREE_364_ONEHOT: t364_leaf0 + t364_leaf1 + t364_leaf2 + t364_leaf3
   + t364_leaf4 = 1
 T364_L0_f68_le: const_Street_Grvl + 999999 t364_leaf0 <= 1e+06
 T364_L0_f232_le: central_air_yes + 999999 t364_leaf0 <= 1e+06
 T364_L0_f1_le: const_Lot_Area + 992080 t364_leaf0 <= 1e+06
 T364_L1_f68_le: const_Street_Grvl + 999999 t364_leaf1 <= 1e+06
 T364_L1_f232_le: central_air_yes + 999999 t364_leaf1 <= 1e+06
 T364_R1_f1_ge: const_Lot_Area - 1.00792e+06 t364_leaf1 >= -1e+06
 T364_L2_f68_le: const_Street_Grvl + 999999 t364_leaf2 <= 1e+06
 T364_R2_f232_ge: central_air_yes - 1.000001e+06 t364_leaf2 >= -1e+06
 T364_L2_f206_le: const_Bsmt_Exposure_Av + 999999 t364_leaf2 <= 1e+06
 T364_L3_f68_le: const_Street_Grvl + 999999 t364_leaf3 <= 1e+06
 T364_R3_f232_ge: central_air_yes - 1.000001e+06 t364_leaf3 >= -1e+06
 T364_R3_f206_ge: const_Bsmt_Exposure_Av - 1.000001e+06 t364_leaf3
   >= -1e+06
 T364_R4_f68_ge: const_Street_Grvl - 1.000001e+06 t364_leaf4 >= -1e+06
 TREE_365_ONEHOT: t365_leaf0 + t365_leaf1 + t365_leaf2 + t365_leaf3
   + t365_leaf4 = 1
 T365_L0_f239_le: const_FunctioNo_aplical_Maj2 + 999999 t365_leaf0
   <= 1e+06
 T365_L0_f20_le: x_Gr_Liv_Area + 9184 t365_leaf0 <= 10000
 T365_L0_f257_le: x_paved_drive_is_N <= 1
 T365_L1_f239_le: const_FunctioNo_aplical_Maj2 + 999999 t365_leaf1
   <= 1e+06
 T365_L1_f20_le: x_Gr_Liv_Area + 9184 t365_leaf1 <= 10000
 T365_R1_f257_ge: x_paved_drive_is_N - t365_leaf1 >= 0
 T365_L2_f239_le: const_FunctioNo_aplical_Maj2 + 999999 t365_leaf2
   <= 1e+06
 T365_R2_f20_ge: x_Gr_Liv_Area - 816 t365_leaf2 >= 0
 T365_L2_f46_le: const_MS_SubClass_30 + 999999 t365_leaf2 <= 1e+06
 T365_L3_f239_le: const_FunctioNo_aplical_Maj2 + 999999 t365_leaf3
   <= 1e+06
 T365_R3_f20_ge: x_Gr_Liv_Area - 816 t365_leaf3 >= 0
 T365_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t365_leaf3 >= -1e+06
 T365_R4_f239_ge: const_FunctioNo_aplical_Maj2 - 1.000001e+06 t365_leaf4
   >= -1e+06
 TREE_366_ONEHOT: t366_leaf0 + t366_leaf1 + t366_leaf2 + t366_leaf3
   + t366_leaf4 = 1
 T366_L0_f30_le: x_Fireplace_Qu + 2 t366_leaf0 <= 4
 T366_L0_f1_le: const_Lot_Area + 995500 t366_leaf0 <= 1e+06
 T366_L1_f30_le: x_Fireplace_Qu + 2 t366_leaf1 <= 4
 T366_R1_f1_ge: const_Lot_Area - 1.0045e+06 t366_leaf1 >= -1e+06
 T366_R2_f30_ge: x_Fireplace_Qu - 3 t366_leaf2 >= -1
 T366_L2_f9_le: x_Exter_Cond + 2 t366_leaf2 <= 4
 T366_R3_f30_ge: x_Fireplace_Qu - 3 t366_leaf3 >= -1
 T366_R3_f9_ge: x_Exter_Cond - 3 t366_leaf3 >= -1
 T366_L3_f78_le: const_Land_Contour_HLS + 999999 t366_leaf3 <= 1e+06
 T366_R4_f30_ge: x_Fireplace_Qu - 3 t366_leaf4 >= -1
 T366_R4_f9_ge: x_Exter_Cond - 3 t366_leaf4 >= -1
 T366_R4_f78_ge: const_Land_Contour_HLS - 1.000001e+06 t366_leaf4
   >= -1e+06
 TREE_367_ONEHOT: t367_leaf0 + t367_leaf1 + t367_leaf2 + t367_leaf3 = 1
 T367_L0_f3_le: Overall_Qual_calc + 999996 t367_leaf0 <= 1e+06
 T367_R1_f3_ge: Overall_Qual_calc - 1.000004e+06 t367_leaf1 >= -1e+06
 T367_L1_f68_le: const_Street_Grvl + 999999 t367_leaf1 <= 1e+06
 T367_L1_f219_le: x_b2_is_BLQ <= 1
 T367_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t367_leaf2 >= -1e+06
 T367_L2_f68_le: const_Street_Grvl + 999999 t367_leaf2 <= 1e+06
 T367_R2_f219_ge: x_b2_is_BLQ - t367_leaf2 >= 0
 T367_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t367_leaf3 >= -1e+06
 T367_R3_f68_ge: const_Street_Grvl - 1.000001e+06 t367_leaf3 >= -1e+06
 TREE_368_ONEHOT: t368_leaf0 + t368_leaf1 + t368_leaf2 + t368_leaf3
   + t368_leaf4 = 1
 T368_L0_f96_le: const_Neighborhood_Edwards + 999999 t368_leaf0 <= 1e+06
 T368_R1_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t368_leaf1
   >= -1e+06
 T368_L1_f15_le: const_Total_Bsmt_SF + 999239 t368_leaf1 <= 1e+06
 T368_L1_f33_le: x_Garage_Area + 99775 t368_leaf1 <= 100000
 T368_R2_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t368_leaf2
   >= -1e+06
 T368_L2_f15_le: const_Total_Bsmt_SF + 999239 t368_leaf2 <= 1e+06
 T368_R2_f33_ge: x_Garage_Area - 225 t368_leaf2 >= 0
 T368_R3_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t368_leaf3
   >= -1e+06
 T368_R3_f15_ge: const_Total_Bsmt_SF - 1.000761e+06 t368_leaf3 >= -1e+06
 T368_L3_f33_le: x_Garage_Area + 99750 t368_leaf3 <= 100000
 T368_R4_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t368_leaf4
   >= -1e+06
 T368_R4_f15_ge: const_Total_Bsmt_SF - 1.000761e+06 t368_leaf4 >= -1e+06
 T368_R4_f33_ge: x_Garage_Area - 250 t368_leaf4 >= 0
 TREE_369_ONEHOT: t369_leaf0 + t369_leaf1 + t369_leaf2 + t369_leaf3
   + t369_leaf4 + t369_leaf5 = 1
 T369_L0_f72_le: const_Alley_Pave + 999999 t369_leaf0 <= 1e+06
 T369_L0_f211_le: x_b1_is_ALQ <= 1
 T369_L0_f5_le: const_Year_Built + 997995 t369_leaf0 <= 1e+06
 T369_L1_f72_le: const_Alley_Pave + 999999 t369_leaf1 <= 1e+06
 T369_L1_f211_le: x_b1_is_ALQ <= 1
 T369_R1_f5_ge: const_Year_Built - 1.002005e+06 t369_leaf1 >= -1e+06
 T369_L2_f72_le: const_Alley_Pave + 999999 t369_leaf2 <= 1e+06
 T369_R2_f211_ge: x_b1_is_ALQ - t369_leaf2 >= 0
 T369_L2_f3_le: Overall_Qual_calc + 999993 t369_leaf2 <= 1e+06
 T369_L3_f72_le: const_Alley_Pave + 999999 t369_leaf3 <= 1e+06
 T369_R3_f211_ge: x_b1_is_ALQ - t369_leaf3 >= 0
 T369_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t369_leaf3 >= -1e+06
 T369_R4_f72_ge: const_Alley_Pave - 1.000001e+06 t369_leaf4 >= -1e+06
 T369_L4_f3_le: Overall_Qual_calc + 999993 t369_leaf4 <= 1e+06
 T369_R5_f72_ge: const_Alley_Pave - 1.000001e+06 t369_leaf5 >= -1e+06
 T369_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t369_leaf5 >= -1e+06
 TREE_370_ONEHOT: t370_leaf0 + t370_leaf1 + t370_leaf2 = 1
 T370_L0_f75_le: const_Lot_Shape_IR3 + 999999 t370_leaf0 <= 1e+06
 T370_L0_f62_le: const_MS_Zoning_C_(all) + 999999 t370_leaf0 <= 1e+06
 T370_L1_f75_le: const_Lot_Shape_IR3 + 999999 t370_leaf1 <= 1e+06
 T370_R1_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t370_leaf1
   >= -1e+06
 T370_R2_f75_ge: const_Lot_Shape_IR3 - 1.000001e+06 t370_leaf2 >= -1e+06
 TREE_371_ONEHOT: t371_leaf0 + t371_leaf1 + t371_leaf2 + t371_leaf3
   + t371_leaf4 = 1
 T371_L0_f20_le: x_Gr_Liv_Area + 9064 t371_leaf0 <= 10000
 T371_R1_f20_ge: x_Gr_Liv_Area - 936 t371_leaf1 >= 0
 T371_L1_f20_le: x_Gr_Liv_Area + 8354 t371_leaf1 <= 10000
 T371_L1_f231_le: central_air_no + 999999 t371_leaf1 <= 1e+06
 T371_R2_f20_ge: x_Gr_Liv_Area - 936 t371_leaf2 >= 0
 T371_L2_f20_le: x_Gr_Liv_Area + 8354 t371_leaf2 <= 10000
 T371_R2_f231_ge: central_air_no - 1.000001e+06 t371_leaf2 >= -1e+06
 T371_R3_f20_ge: x_Gr_Liv_Area - 936 t371_leaf3 >= 0
 T371_R3_f20_ge: x_Gr_Liv_Area - 1646 t371_leaf3 >= 0
 T371_L3_f5_le: const_Year_Built + 998089 t371_leaf3 <= 1e+06
 T371_R4_f20_ge: x_Gr_Liv_Area - 936 t371_leaf4 >= 0
 T371_R4_f20_ge: x_Gr_Liv_Area - 1646 t371_leaf4 >= 0
 T371_R4_f5_ge: const_Year_Built - 1.001911e+06 t371_leaf4 >= -1e+06
 TREE_372_ONEHOT: t372_leaf0 + t372_leaf1 + t372_leaf2 = 1
 T372_L0_f40_le: x_Screen_Porch + 99892 t372_leaf0 <= 100000
 T372_L0_f0_le: const_Lot_Frontage + 999826 t372_leaf0 <= 1e+06
 T372_L1_f40_le: x_Screen_Porch + 99892 t372_leaf1 <= 100000
 T372_R1_f0_ge: const_Lot_Frontage - 1.000174e+06 t372_leaf1 >= -1e+06
 T372_R2_f40_ge: x_Screen_Porch - 108 t372_leaf2 >= 0
 TREE_373_ONEHOT: t373_leaf0 + t373_leaf1 + t373_leaf2 + t373_leaf3
   + t373_leaf4 = 1
 T373_L0_f177_le: x_ex2_is_AsbShng <= 1
 T373_L0_f15_le: const_Total_Bsmt_SF + 999494 t373_leaf0 <= 1e+06
 T373_L0_f35_le: x_Garage_Cond + 3 t373_leaf0 <= 4
 T373_L1_f177_le: x_ex2_is_AsbShng <= 1
 T373_L1_f15_le: const_Total_Bsmt_SF + 999494 t373_leaf1 <= 1e+06
 T373_R1_f35_ge: x_Garage_Cond - 2 t373_leaf1 >= -1
 T373_L2_f177_le: x_ex2_is_AsbShng <= 1
 T373_R2_f15_ge: const_Total_Bsmt_SF - 1.000506e+06 t373_leaf2 >= -1e+06
 T373_L2_f86_le: const_Land_Slope_Gtl + 999999 t373_leaf2 <= 1e+06
 T373_L3_f177_le: x_ex2_is_AsbShng <= 1
 T373_R3_f15_ge: const_Total_Bsmt_SF - 1.000506e+06 t373_leaf3 >= -1e+06
 T373_R3_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t373_leaf3 >= -1e+06
 T373_R4_f177_ge: x_ex2_is_AsbShng - t373_leaf4 >= 0
 TREE_374_ONEHOT: t374_leaf0 + t374_leaf1 + t374_leaf2 + t374_leaf3
   + t374_leaf4 + t374_leaf5 = 1
 T374_L0_f219_le: x_b2_is_BLQ <= 1
 T374_L0_f23_le: x_Full_Bath + t374_leaf0 <= 4
 T374_L0_f13_le: x_BsmtFin_SF_2 + 99288 t374_leaf0 <= 100000
 T374_L1_f219_le: x_b2_is_BLQ <= 1
 T374_L1_f23_le: x_Full_Bath + t374_leaf1 <= 4
 T374_R1_f13_ge: x_BsmtFin_SF_2 - 712 t374_leaf1 >= 0
 T374_L2_f219_le: x_b2_is_BLQ <= 1
 T374_R2_f23_ge: x_Full_Bath - 3 t374_leaf2 >= 0
 T374_R3_f219_ge: x_b2_is_BLQ - t374_leaf3 >= 0
 T374_L3_f33_le: x_Garage_Area + 99662 t374_leaf3 <= 100000
 T374_R4_f219_ge: x_b2_is_BLQ - t374_leaf4 >= 0
 T374_R4_f33_ge: x_Garage_Area - 338 t374_leaf4 >= 0
 T374_L4_f251_le: const_Garage_Type_Detchd + 999999 t374_leaf4 <= 1e+06
 T374_R5_f219_ge: x_b2_is_BLQ - t374_leaf5 >= 0
 T374_R5_f33_ge: x_Garage_Area - 338 t374_leaf5 >= 0
 T374_R5_f251_ge: const_Garage_Type_Detchd - 1.000001e+06 t374_leaf5
   >= -1e+06
 TREE_375_ONEHOT: t375_leaf0 + t375_leaf1 + t375_leaf2 + t375_leaf3 = 1
 T375_L0_f164_le: x_ex1_is_BrkFace <= 1
 T375_L0_f161_le: x_ex1_is_AsbShng <= 1
 T375_L1_f164_le: x_ex1_is_BrkFace <= 1
 T375_R1_f161_ge: x_ex1_is_AsbShng - t375_leaf1 >= 0
 T375_L1_f31_le: const_Garage_Yr_Blt + 998079 t375_leaf1 <= 1e+06
 T375_L2_f164_le: x_ex1_is_BrkFace <= 1
 T375_R2_f161_ge: x_ex1_is_AsbShng - t375_leaf2 >= 0
 T375_R2_f31_ge: const_Garage_Yr_Blt - 1.001921e+06 t375_leaf2 >= -1e+06
 T375_R3_f164_ge: x_ex1_is_BrkFace - t375_leaf3 >= 0
 TREE_376_ONEHOT: t376_leaf0 + t376_leaf1 + t376_leaf2 + t376_leaf3 = 1
 T376_L0_f5_le: const_Year_Built + 997995 t376_leaf0 <= 1e+06
 T376_L0_f96_le: const_Neighborhood_Edwards + 999999 t376_leaf0 <= 1e+06
 T376_L1_f5_le: const_Year_Built + 997995 t376_leaf1 <= 1e+06
 T376_R1_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t376_leaf1
   >= -1e+06
 T376_L1_f31_le: const_Garage_Yr_Blt + 998044 t376_leaf1 <= 1e+06
 T376_L2_f5_le: const_Year_Built + 997995 t376_leaf2 <= 1e+06
 T376_R2_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t376_leaf2
   >= -1e+06
 T376_R2_f31_ge: const_Garage_Yr_Blt - 1.001956e+06 t376_leaf2 >= -1e+06
 T376_R3_f5_ge: const_Year_Built - 1.002005e+06 t376_leaf3 >= -1e+06
 TREE_377_ONEHOT: t377_leaf0 = 1
 TREE_378_ONEHOT: t378_leaf0 + t378_leaf1 + t378_leaf2 = 1
 T378_L0_f3_le: Overall_Qual_calc + 999997 t378_leaf0 <= 1e+06
 T378_L0_f33_le: x_Garage_Area + 99727 t378_leaf0 <= 100000
 T378_L1_f3_le: Overall_Qual_calc + 999997 t378_leaf1 <= 1e+06
 T378_R1_f33_ge: x_Garage_Area - 273 t378_leaf1 >= 0
 T378_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t378_leaf2 >= -1e+06
 TREE_379_ONEHOT: t379_leaf0 + t379_leaf1 + t379_leaf2 + t379_leaf3
   + t379_leaf4 = 1
 T379_L0_f32_le: const_Garage_Cars + 999997 t379_leaf0 <= 1e+06
 T379_L0_f17_le: x_1st_Flr_SF + 4424 t379_leaf0 <= 5000
 T379_L1_f32_le: const_Garage_Cars + 999997 t379_leaf1 <= 1e+06
 T379_R1_f17_ge: x_1st_Flr_SF - 576 t379_leaf1 >= 0
 T379_R2_f32_ge: const_Garage_Cars - 1.000003e+06 t379_leaf2 >= -1e+06
 T379_L2_f30_le: x_Fireplace_Qu + 4 t379_leaf2 <= 4
 T379_R3_f32_ge: const_Garage_Cars - 1.000003e+06 t379_leaf3 >= -1e+06
 T379_R3_f30_ge: x_Fireplace_Qu - t379_leaf3 >= -1
 T379_L3_f40_le: x_Screen_Porch + 99960 t379_leaf3 <= 100000
 T379_R4_f32_ge: const_Garage_Cars - 1.000003e+06 t379_leaf4 >= -1e+06
 T379_R4_f30_ge: x_Fireplace_Qu - t379_leaf4 >= -1
 T379_R4_f40_ge: x_Screen_Porch - 40 t379_leaf4 >= 0
 TREE_380_ONEHOT: t380_leaf0 + t380_leaf1 = 1
 T380_L0_f40_le: x_Screen_Porch + 99844 t380_leaf0 <= 100000
 T380_R1_f40_ge: x_Screen_Porch - 156 t380_leaf1 >= 0
 TREE_381_ONEHOT: t381_leaf0 + t381_leaf1 + t381_leaf2 + t381_leaf3
   + t381_leaf4 = 1
 T381_L0_f3_le: Overall_Qual_calc + 999996 t381_leaf0 <= 1e+06
 T381_L0_f169_le: x_ex1_is_MetalSd <= 1
 T381_L1_f3_le: Overall_Qual_calc + 999996 t381_leaf1 <= 1e+06
 T381_R1_f169_ge: x_ex1_is_MetalSd - t381_leaf1 >= 0
 T381_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t381_leaf2 >= -1e+06
 T381_L2_f5_le: const_Year_Built + 998089 t381_leaf2 <= 1e+06
 T381_L2_f4_le: const_Overall_Cond + 999993 t381_leaf2 <= 1e+06
 T381_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t381_leaf3 >= -1e+06
 T381_L3_f5_le: const_Year_Built + 998089 t381_leaf3 <= 1e+06
 T381_R3_f4_ge: const_Overall_Cond - 1.000007e+06 t381_leaf3 >= -1e+06
 T381_R4_f3_ge: Overall_Qual_calc - 1.000004e+06 t381_leaf4 >= -1e+06
 T381_R4_f5_ge: const_Year_Built - 1.001911e+06 t381_leaf4 >= -1e+06
 TREE_382_ONEHOT: t382_leaf0 + t382_leaf1 + t382_leaf2 + t382_leaf3
   + t382_leaf4 + t382_leaf5 = 1
 T382_L0_f27_le: x_Kitchen_Qual + 2 t382_leaf0 <= 4
 T382_L0_f32_le: const_Garage_Cars + 999998 t382_leaf0 <= 1e+06
 T382_L1_f27_le: x_Kitchen_Qual + 2 t382_leaf1 <= 4
 T382_R1_f32_ge: const_Garage_Cars - 1.000002e+06 t382_leaf1 >= -1e+06
 T382_R2_f27_ge: x_Kitchen_Qual - 3 t382_leaf2 >= -1
 T382_L2_f6_le: const_Year_Remod_Add + 998035 t382_leaf2 <= 1e+06
 T382_L2_f17_le: x_1st_Flr_SF + 4084 t382_leaf2 <= 5000
 T382_R3_f27_ge: x_Kitchen_Qual - 3 t382_leaf3 >= -1
 T382_L3_f6_le: const_Year_Remod_Add + 998035 t382_leaf3 <= 1e+06
 T382_R3_f17_ge: x_1st_Flr_SF - 916 t382_leaf3 >= 0
 T382_R4_f27_ge: x_Kitchen_Qual - 3 t382_leaf4 >= -1
 T382_R4_f6_ge: const_Year_Remod_Add - 1.001965e+06 t382_leaf4 >= -1e+06
 T382_L4_f46_le: const_MS_SubClass_30 + 999999 t382_leaf4 <= 1e+06
 T382_R5_f27_ge: x_Kitchen_Qual - 3 t382_leaf5 >= -1
 T382_R5_f6_ge: const_Year_Remod_Add - 1.001965e+06 t382_leaf5 >= -1e+06
 T382_R5_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t382_leaf5 >= -1e+06
 TREE_383_ONEHOT: t383_leaf0 + t383_leaf1 + t383_leaf2 + t383_leaf3
   + t383_leaf4 + t383_leaf5 + t383_leaf6 = 1
 T383_L0_f20_le: x_Gr_Liv_Area + 9106 t383_leaf0 <= 10000
 T383_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t383_leaf0
   <= 1e+06
 T383_L0_f76_le: const_Lot_Shape_Reg + 999999 t383_leaf0 <= 1e+06
 T383_L1_f20_le: x_Gr_Liv_Area + 9106 t383_leaf1 <= 10000
 T383_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t383_leaf1
   <= 1e+06
 T383_R1_f76_ge: const_Lot_Shape_Reg - 1.000001e+06 t383_leaf1 >= -1e+06
 T383_L2_f20_le: x_Gr_Liv_Area + 9106 t383_leaf2 <= 10000
 T383_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t383_leaf2
   >= -1e+06
 T383_L2_f33_le: x_Garage_Area + 99629 t383_leaf2 <= 100000
 T383_L3_f20_le: x_Gr_Liv_Area + 9106 t383_leaf3 <= 10000
 T383_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t383_leaf3
   >= -1e+06
 T383_R3_f33_ge: x_Garage_Area - 371 t383_leaf3 >= 0
 T383_R4_f20_ge: x_Gr_Liv_Area - 894 t383_leaf4 >= 0
 T383_L4_f28_le: x_TotRms_AbvGrd + 3 t383_leaf4 <= 15
 T383_L4_f112_le: const_Neighborhood_SawyerW + 999999 t383_leaf4 <= 1e+06
 T383_R5_f20_ge: x_Gr_Liv_Area - 894 t383_leaf5 >= 0
 T383_L5_f28_le: x_TotRms_AbvGrd + 3 t383_leaf5 <= 15
 T383_R5_f112_ge: const_Neighborhood_SawyerW - 1.000001e+06 t383_leaf5
   >= -1e+06
 T383_R6_f20_ge: x_Gr_Liv_Area - 894 t383_leaf6 >= 0
 T383_R6_f28_ge: x_TotRms_AbvGrd - 12 t383_leaf6 >= 0
 TREE_384_ONEHOT: t384_leaf0 + t384_leaf1 + t384_leaf2 + t384_leaf3
   + t384_leaf4 = 1
 T384_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t384_leaf0 <= 1e+06
 T384_R1_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t384_leaf1
   >= -1e+06
 T384_L1_f14_le: x_Bsmt_Unf_SF + 99878 t384_leaf1 <= 100000
 T384_L1_f139_le: const_House_Style_1.5Fin + 999999 t384_leaf1 <= 1e+06
 T384_R2_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t384_leaf2
   >= -1e+06
 T384_L2_f14_le: x_Bsmt_Unf_SF + 99878 t384_leaf2 <= 100000
 T384_R2_f139_ge: const_House_Style_1.5Fin - 1.000001e+06 t384_leaf2
   >= -1e+06
 T384_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t384_leaf3
   >= -1e+06
 T384_R3_f14_ge: x_Bsmt_Unf_SF - 122 t384_leaf3 >= 0
 T384_L3_f33_le: x_Garage_Area + 99129 t384_leaf3 <= 100000
 T384_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t384_leaf4
   >= -1e+06
 T384_R4_f14_ge: x_Bsmt_Unf_SF - 122 t384_leaf4 >= 0
 T384_R4_f33_ge: x_Garage_Area - 871 t384_leaf4 >= 0
 TREE_385_ONEHOT: t385_leaf0 + t385_leaf1 + t385_leaf2 + t385_leaf3
   + t385_leaf4 = 1
 T385_L0_f161_le: x_ex1_is_AsbShng <= 1
 T385_L0_f62_le: const_MS_Zoning_C_(all) + 999999 t385_leaf0 <= 1e+06
 T385_L0_f3_le: Overall_Qual_calc + 999993 t385_leaf0 <= 1e+06
 T385_L1_f161_le: x_ex1_is_AsbShng <= 1
 T385_L1_f62_le: const_MS_Zoning_C_(all) + 999999 t385_leaf1 <= 1e+06
 T385_R1_f3_ge: Overall_Qual_calc - 1.000007e+06 t385_leaf1 >= -1e+06
 T385_L2_f161_le: x_ex1_is_AsbShng <= 1
 T385_R2_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t385_leaf2
   >= -1e+06
 T385_R3_f161_ge: x_ex1_is_AsbShng - t385_leaf3 >= 0
 T385_L3_f20_le: x_Gr_Liv_Area + 9117 t385_leaf3 <= 10000
 T385_R4_f161_ge: x_ex1_is_AsbShng - t385_leaf4 >= 0
 T385_R4_f20_ge: x_Gr_Liv_Area - 883 t385_leaf4 >= 0
 TREE_386_ONEHOT: t386_leaf0 + t386_leaf1 + t386_leaf2 + t386_leaf3 = 1
 T386_L0_f33_le: x_Garage_Area + 98846 t386_leaf0 <= 100000
 T386_L0_f5_le: const_Year_Built + 997994 t386_leaf0 <= 1e+06
 T386_L0_f3_le: Overall_Qual_calc + 999997 t386_leaf0 <= 1e+06
 T386_L1_f33_le: x_Garage_Area + 98846 t386_leaf1 <= 100000
 T386_L1_f5_le: const_Year_Built + 997994 t386_leaf1 <= 1e+06
 T386_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t386_leaf1 >= -1e+06
 T386_L2_f33_le: x_Garage_Area + 98846 t386_leaf2 <= 100000
 T386_R2_f5_ge: const_Year_Built - 1.002006e+06 t386_leaf2 >= -1e+06
 T386_R3_f33_ge: x_Garage_Area - 1154 t386_leaf3 >= 0
 TREE_387_ONEHOT: t387_leaf0 + t387_leaf1 + t387_leaf2 + t387_leaf3
   + t387_leaf4 = 1
 T387_L0_f0_le: const_Lot_Frontage + 999879 t387_leaf0 <= 1e+06
 T387_L0_f230_le: heat_is_Wall + 999999 t387_leaf0 <= 1e+06
 T387_L0_f231_le: central_air_no + 999999 t387_leaf0 <= 1e+06
 T387_L1_f0_le: const_Lot_Frontage + 999879 t387_leaf1 <= 1e+06
 T387_L1_f230_le: heat_is_Wall + 999999 t387_leaf1 <= 1e+06
 T387_R1_f231_ge: central_air_no - 1.000001e+06 t387_leaf1 >= -1e+06
 T387_L2_f0_le: const_Lot_Frontage + 999879 t387_leaf2 <= 1e+06
 T387_R2_f230_ge: heat_is_Wall - 1.000001e+06 t387_leaf2 >= -1e+06
 T387_R3_f0_ge: const_Lot_Frontage - 1.000121e+06 t387_leaf3 >= -1e+06
 T387_L3_f292_le: const_Sale_Type_WD_ + 999999 t387_leaf3 <= 1e+06
 T387_R4_f0_ge: const_Lot_Frontage - 1.000121e+06 t387_leaf4 >= -1e+06
 T387_R4_f292_ge: const_Sale_Type_WD_ - 1.000001e+06 t387_leaf4 >= -1e+06
 TREE_388_ONEHOT: t388_leaf0 + t388_leaf1 + t388_leaf2 + t388_leaf3 = 1
 T388_L0_f26_le: x_Kitchen_AbvGr + t388_leaf0 <= 3
 T388_L0_f4_le: const_Overall_Cond + 999997 t388_leaf0 <= 1e+06
 T388_L1_f26_le: x_Kitchen_AbvGr + t388_leaf1 <= 3
 T388_R1_f4_ge: const_Overall_Cond - 1.000003e+06 t388_leaf1 >= -1e+06
 T388_L1_f232_le: central_air_yes + 999999 t388_leaf1 <= 1e+06
 T388_L2_f26_le: x_Kitchen_AbvGr + t388_leaf2 <= 3
 T388_R2_f4_ge: const_Overall_Cond - 1.000003e+06 t388_leaf2 >= -1e+06
 T388_R2_f232_ge: central_air_yes - 1.000001e+06 t388_leaf2 >= -1e+06
 T388_R3_f26_ge: x_Kitchen_AbvGr - 2 t388_leaf3 >= 0
 TREE_389_ONEHOT: t389_leaf0 + t389_leaf1 + t389_leaf2 + t389_leaf3 = 1
 T389_L0_f92_le: const_Neighborhood_BrkSide + 999999 t389_leaf0 <= 1e+06
 T389_L0_f44_le: const_Yr_Sold + 997993 t389_leaf0 <= 1e+06
 T389_L0_f231_le: central_air_no + 999999 t389_leaf0 <= 1e+06
 T389_L1_f92_le: const_Neighborhood_BrkSide + 999999 t389_leaf1 <= 1e+06
 T389_L1_f44_le: const_Yr_Sold + 997993 t389_leaf1 <= 1e+06
 T389_R1_f231_ge: central_air_no - 1.000001e+06 t389_leaf1 >= -1e+06
 T389_L2_f92_le: const_Neighborhood_BrkSide + 999999 t389_leaf2 <= 1e+06
 T389_R2_f44_ge: const_Yr_Sold - 1.002007e+06 t389_leaf2 >= -1e+06
 T389_R3_f92_ge: const_Neighborhood_BrkSide - 1.000001e+06 t389_leaf3
   >= -1e+06
 TREE_390_ONEHOT: t390_leaf0 + t390_leaf1 + t390_leaf2 = 1
 T390_L0_f1_le: const_Lot_Area + 994024 t390_leaf0 <= 1e+06
 T390_R1_f1_ge: const_Lot_Area - 1.005976e+06 t390_leaf1 >= -1e+06
 T390_L1_f114_le: const_Neighborhood_StoneBr + 999999 t390_leaf1 <= 1e+06
 T390_R2_f1_ge: const_Lot_Area - 1.005976e+06 t390_leaf2 >= -1e+06
 T390_R2_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t390_leaf2
   >= -1e+06
 TREE_391_ONEHOT: t391_leaf0 + t391_leaf1 + t391_leaf2 + t391_leaf3 = 1
 T391_L0_f3_le: Overall_Qual_calc + 999997 t391_leaf0 <= 1e+06
 T391_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t391_leaf1 >= -1e+06
 T391_L1_f250_le: const_Garage_Type_CarPort + 999999 t391_leaf1 <= 1e+06
 T391_L1_f12_le: x_BsmtFin_SF_1 + 998268 t391_leaf1 <= 1e+06
 T391_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t391_leaf2 >= -1e+06
 T391_L2_f250_le: const_Garage_Type_CarPort + 999999 t391_leaf2 <= 1e+06
 T391_R2_f12_ge: x_BsmtFin_SF_1 - 1732 t391_leaf2 >= 0
 T391_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t391_leaf3 >= -1e+06
 T391_R3_f250_ge: const_Garage_Type_CarPort - 1.000001e+06 t391_leaf3
   >= -1e+06
 TREE_392_ONEHOT: t392_leaf0 + t392_leaf1 + t392_leaf2 + t392_leaf3 = 1
 T392_L0_f96_le: const_Neighborhood_Edwards + 999999 t392_leaf0 <= 1e+06
 T392_L0_f7_le: x_Mas_Vnr_Area + 1544 t392_leaf0 <= 2000
 T392_L1_f96_le: const_Neighborhood_Edwards + 999999 t392_leaf1 <= 1e+06
 T392_R1_f7_ge: x_Mas_Vnr_Area - 456 t392_leaf1 >= 0
 T392_R2_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t392_leaf2
   >= -1e+06
 T392_L2_f15_le: const_Total_Bsmt_SF + 999302 t392_leaf2 <= 1e+06
 T392_R3_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t392_leaf3
   >= -1e+06
 T392_R3_f15_ge: const_Total_Bsmt_SF - 1.000698e+06 t392_leaf3 >= -1e+06
 TREE_393_ONEHOT: t393_leaf0 + t393_leaf1 + t393_leaf2 + t393_leaf3
   + t393_leaf4 = 1
 T393_L0_f12_le: x_BsmtFin_SF_1 + 998268 t393_leaf0 <= 1e+06
 T393_L0_f8_le: x_Exter_Qual + t393_leaf0 <= 4
 T393_L0_f20_le: x_Gr_Liv_Area + 8284 t393_leaf0 <= 10000
 T393_L1_f12_le: x_BsmtFin_SF_1 + 998268 t393_leaf1 <= 1e+06
 T393_L1_f8_le: x_Exter_Qual + t393_leaf1 <= 4
 T393_R1_f20_ge: x_Gr_Liv_Area - 1716 t393_leaf1 >= 0
 T393_L2_f12_le: x_BsmtFin_SF_1 + 998268 t393_leaf2 <= 1e+06
 T393_R2_f8_ge: x_Exter_Qual - 4 t393_leaf2 >= -1
 T393_L2_f17_le: x_1st_Flr_SF + 3218 t393_leaf2 <= 5000
 T393_L3_f12_le: x_BsmtFin_SF_1 + 998268 t393_leaf3 <= 1e+06
 T393_R3_f8_ge: x_Exter_Qual - 4 t393_leaf3 >= -1
 T393_R3_f17_ge: x_1st_Flr_SF - 1782 t393_leaf3 >= 0
 T393_R4_f12_ge: x_BsmtFin_SF_1 - 1732 t393_leaf4 >= 0
 TREE_394_ONEHOT: t394_leaf0 + t394_leaf1 = 1
 T394_L0_f216_le: x_b1_is_Rec <= 1
 T394_R1_f216_ge: x_b1_is_Rec - t394_leaf1 >= 0
 TREE_395_ONEHOT: t395_leaf0 + t395_leaf1 + t395_leaf2 + t395_leaf3
   + t395_leaf4 + t395_leaf5 = 1
 T395_L0_f37_le: x_Open_Porch_SF + 99709 t395_leaf0 <= 100000
 T395_L0_f6_le: const_Year_Remod_Add + 997993 t395_leaf0 <= 1e+06
 T395_L0_f4_le: const_Overall_Cond + 999993 t395_leaf0 <= 1e+06
 T395_L1_f37_le: x_Open_Porch_SF + 99709 t395_leaf1 <= 100000
 T395_L1_f6_le: const_Year_Remod_Add + 997993 t395_leaf1 <= 1e+06
 T395_R1_f4_ge: const_Overall_Cond - 1.000007e+06 t395_leaf1 >= -1e+06
 T395_L2_f37_le: x_Open_Porch_SF + 99709 t395_leaf2 <= 100000
 T395_R2_f6_ge: const_Year_Remod_Add - 1.002007e+06 t395_leaf2 >= -1e+06
 T395_L2_f89_le: const_Neighborhood_Blmngtn + 999999 t395_leaf2 <= 1e+06
 T395_L3_f37_le: x_Open_Porch_SF + 99709 t395_leaf3 <= 100000
 T395_R3_f6_ge: const_Year_Remod_Add - 1.002007e+06 t395_leaf3 >= -1e+06
 T395_R3_f89_ge: const_Neighborhood_Blmngtn - 1.000001e+06 t395_leaf3
   >= -1e+06
 T395_R4_f37_ge: x_Open_Porch_SF - 291 t395_leaf4 >= 0
 T395_L4_f297_le: const_Sale_Condition_Normal + 999999 t395_leaf4 <= 1e+06
 T395_R5_f37_ge: x_Open_Porch_SF - 291 t395_leaf5 >= 0
 T395_R5_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t395_leaf5
   >= -1e+06
 TREE_396_ONEHOT: t396_leaf0 + t396_leaf1 + t396_leaf2 + t396_leaf3
   + t396_leaf4 = 1
 T396_L0_f3_le: Overall_Qual_calc + 999997 t396_leaf0 <= 1e+06
 T396_L0_f31_le: const_Garage_Yr_Blt + 998105 t396_leaf0 <= 1e+06
 T396_L1_f3_le: Overall_Qual_calc + 999997 t396_leaf1 <= 1e+06
 T396_R1_f31_ge: const_Garage_Yr_Blt - 1.001895e+06 t396_leaf1 >= -1e+06
 T396_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t396_leaf2 >= -1e+06
 T396_L2_f37_le: x_Open_Porch_SF + 99516 t396_leaf2 <= 100000
 T396_L2_f35_le: x_Garage_Cond + 2 t396_leaf2 <= 4
 T396_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t396_leaf3 >= -1e+06
 T396_L3_f37_le: x_Open_Porch_SF + 99516 t396_leaf3 <= 100000
 T396_R3_f35_ge: x_Garage_Cond - 3 t396_leaf3 >= -1
 T396_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t396_leaf4 >= -1e+06
 T396_R4_f37_ge: x_Open_Porch_SF - 484 t396_leaf4 >= 0
 TREE_397_ONEHOT: t397_leaf0 + t397_leaf1 + t397_leaf2 + t397_leaf3
   + t397_leaf4 = 1
 T397_L0_f296_le: const_Sale_Condition_Family + 999999 t397_leaf0 <= 1e+06
 T397_L0_f161_le: x_ex1_is_AsbShng <= 1
 T397_L1_f296_le: const_Sale_Condition_Family + 999999 t397_leaf1 <= 1e+06
 T397_R1_f161_ge: x_ex1_is_AsbShng - t397_leaf1 >= 0
 T397_L1_f31_le: const_Garage_Yr_Blt + 998077 t397_leaf1 <= 1e+06
 T397_L2_f296_le: const_Sale_Condition_Family + 999999 t397_leaf2 <= 1e+06
 T397_R2_f161_ge: x_ex1_is_AsbShng - t397_leaf2 >= 0
 T397_R2_f31_ge: const_Garage_Yr_Blt - 1.001923e+06 t397_leaf2 >= -1e+06
 T397_R3_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t397_leaf3
   >= -1e+06
 T397_L3_f12_le: x_BsmtFin_SF_1 + 999833 t397_leaf3 <= 1e+06
 T397_R4_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t397_leaf4
   >= -1e+06
 T397_R4_f12_ge: x_BsmtFin_SF_1 - 167 t397_leaf4 >= 0
 TREE_398_ONEHOT: t398_leaf0 + t398_leaf1 + t398_leaf2 + t398_leaf3 = 1
 T398_L0_f37_le: x_Open_Porch_SF + 99841 t398_leaf0 <= 100000
 T398_L0_f11_le: x_Bsmt_Cond + t398_leaf0 <= 4
 T398_L1_f37_le: x_Open_Porch_SF + 99841 t398_leaf1 <= 100000
 T398_R1_f11_ge: x_Bsmt_Cond - 4 t398_leaf1 >= -1
 T398_L1_f31_le: const_Garage_Yr_Blt + 998023 t398_leaf1 <= 1e+06
 T398_L2_f37_le: x_Open_Porch_SF + 99841 t398_leaf2 <= 100000
 T398_R2_f11_ge: x_Bsmt_Cond - 4 t398_leaf2 >= -1
 T398_R2_f31_ge: const_Garage_Yr_Blt - 1.001977e+06 t398_leaf2 >= -1e+06
 T398_R3_f37_ge: x_Open_Porch_SF - 159 t398_leaf3 >= 0
 TREE_399_ONEHOT: t399_leaf0 + t399_leaf1 + t399_leaf2 + t399_leaf3 = 1
 T399_L0_f86_le: const_Land_Slope_Gtl + 999999 t399_leaf0 <= 1e+06
 T399_R1_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t399_leaf1 >= -1e+06
 T399_L1_f1_le: const_Lot_Area + 974714 t399_leaf1 <= 1e+06
 T399_L1_f17_le: x_1st_Flr_SF + 4424 t399_leaf1 <= 5000
 T399_R2_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t399_leaf2 >= -1e+06
 T399_L2_f1_le: const_Lot_Area + 974714 t399_leaf2 <= 1e+06
 T399_R2_f17_ge: x_1st_Flr_SF - 576 t399_leaf2 >= 0
 T399_R3_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t399_leaf3 >= -1e+06
 T399_R3_f1_ge: const_Lot_Area - 1.025286e+06 t399_leaf3 >= -1e+06
 TREE_400_ONEHOT: t400_leaf0 + t400_leaf1 + t400_leaf2 = 1
 T400_L0_f250_le: const_Garage_Type_CarPort + 999999 t400_leaf0 <= 1e+06
 T400_L0_f4_le: const_Overall_Cond + 999993 t400_leaf0 <= 1e+06
 T400_L1_f250_le: const_Garage_Type_CarPort + 999999 t400_leaf1 <= 1e+06
 T400_R1_f4_ge: const_Overall_Cond - 1.000007e+06 t400_leaf1 >= -1e+06
 T400_R2_f250_ge: const_Garage_Type_CarPort - 1.000001e+06 t400_leaf2
   >= -1e+06
 TREE_401_ONEHOT: t401_leaf0 = 1
 TREE_402_ONEHOT: t402_leaf0 + t402_leaf1 + t402_leaf2 + t402_leaf3
   + t402_leaf4 = 1
 T402_L0_f112_le: const_Neighborhood_SawyerW + 999999 t402_leaf0 <= 1e+06
 T402_L0_f226_le: heat_is_GasA + 999999 t402_leaf0 <= 1e+06
 T402_L0_f17_le: x_1st_Flr_SF + 4068 t402_leaf0 <= 5000
 T402_L1_f112_le: const_Neighborhood_SawyerW + 999999 t402_leaf1 <= 1e+06
 T402_L1_f226_le: heat_is_GasA + 999999 t402_leaf1 <= 1e+06
 T402_R1_f17_ge: x_1st_Flr_SF - 932 t402_leaf1 >= 0
 T402_L2_f112_le: const_Neighborhood_SawyerW + 999999 t402_leaf2 <= 1e+06
 T402_R2_f226_ge: heat_is_GasA - 1.000001e+06 t402_leaf2 >= -1e+06
 T402_L2_f3_le: Overall_Qual_calc + 999996 t402_leaf2 <= 1e+06
 T402_L3_f112_le: const_Neighborhood_SawyerW + 999999 t402_leaf3 <= 1e+06
 T402_R3_f226_ge: heat_is_GasA - 1.000001e+06 t402_leaf3 >= -1e+06
 T402_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t402_leaf3 >= -1e+06
 T402_R4_f112_ge: const_Neighborhood_SawyerW - 1.000001e+06 t402_leaf4
   >= -1e+06
 TREE_403_ONEHOT: t403_leaf0 + t403_leaf1 + t403_leaf2 + t403_leaf3
   + t403_leaf4 = 1
 T403_L0_f68_le: const_Street_Grvl + 999999 t403_leaf0 <= 1e+06
 T403_L0_f33_le: x_Garage_Area + 99370 t403_leaf0 <= 100000
 T403_L0_f18_le: const_2nd_Flr_SF + 999060 t403_leaf0 <= 1e+06
 T403_L1_f68_le: const_Street_Grvl + 999999 t403_leaf1 <= 1e+06
 T403_L1_f33_le: x_Garage_Area + 99370 t403_leaf1 <= 100000
 T403_R1_f18_ge: const_2nd_Flr_SF - 1.00094e+06 t403_leaf1 >= -1e+06
 T403_L2_f68_le: const_Street_Grvl + 999999 t403_leaf2 <= 1e+06
 T403_R2_f33_ge: x_Garage_Area - 630 t403_leaf2 >= 0
 T403_L2_f4_le: const_Overall_Cond + 999995 t403_leaf2 <= 1e+06
 T403_L3_f68_le: const_Street_Grvl + 999999 t403_leaf3 <= 1e+06
 T403_R3_f33_ge: x_Garage_Area - 630 t403_leaf3 >= 0
 T403_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t403_leaf3 >= -1e+06
 T403_R4_f68_ge: const_Street_Grvl - 1.000001e+06 t403_leaf4 >= -1e+06
 TREE_404_ONEHOT: t404_leaf0 + t404_leaf1 + t404_leaf2 + t404_leaf3
   + t404_leaf4 + t404_leaf5 = 1
 T404_L0_f3_le: Overall_Qual_calc + 999995 t404_leaf0 <= 1e+06
 T404_L0_f6_le: const_Year_Remod_Add + 998047 t404_leaf0 <= 1e+06
 T404_L0_f0_le: const_Lot_Frontage + 999949 t404_leaf0 <= 1e+06
 T404_L1_f3_le: Overall_Qual_calc + 999995 t404_leaf1 <= 1e+06
 T404_L1_f6_le: const_Year_Remod_Add + 998047 t404_leaf1 <= 1e+06
 T404_R1_f0_ge: const_Lot_Frontage - 1.000051e+06 t404_leaf1 >= -1e+06
 T404_L2_f3_le: Overall_Qual_calc + 999995 t404_leaf2 <= 1e+06
 T404_R2_f6_ge: const_Year_Remod_Add - 1.001953e+06 t404_leaf2 >= -1e+06
 T404_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t404_leaf3 >= -1e+06
 T404_L3_f121_le: const_Condition_1_PosN + 999999 t404_leaf3 <= 1e+06
 T404_L3_f119_le: const_Condition_1_Norm + 999999 t404_leaf3 <= 1e+06
 T404_R4_f3_ge: Overall_Qual_calc - 1.000005e+06 t404_leaf4 >= -1e+06
 T404_L4_f121_le: const_Condition_1_PosN + 999999 t404_leaf4 <= 1e+06
 T404_R4_f119_ge: const_Condition_1_Norm - 1.000001e+06 t404_leaf4
   >= -1e+06
 T404_R5_f3_ge: Overall_Qual_calc - 1.000005e+06 t404_leaf5 >= -1e+06
 T404_R5_f121_ge: const_Condition_1_PosN - 1.000001e+06 t404_leaf5
   >= -1e+06
 TREE_405_ONEHOT: t405_leaf0 + t405_leaf1 + t405_leaf2 + t405_leaf3
   + t405_leaf4 = 1
 T405_L0_f20_le: x_Gr_Liv_Area + 9253 t405_leaf0 <= 10000
 T405_L0_f16_le: x_Heating_QC + t405_leaf0 <= 4
 T405_L1_f20_le: x_Gr_Liv_Area + 9253 t405_leaf1 <= 10000
 T405_R1_f16_ge: x_Heating_QC - 4 t405_leaf1 >= -1
 T405_R2_f20_ge: x_Gr_Liv_Area - 747 t405_leaf2 >= 0
 T405_L2_f3_le: Overall_Qual_calc + 999995 t405_leaf2 <= 1e+06
 T405_R3_f20_ge: x_Gr_Liv_Area - 747 t405_leaf3 >= 0
 T405_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t405_leaf3 >= -1e+06
 T405_L3_f37_le: x_Open_Porch_SF + 99740 t405_leaf3 <= 100000
 T405_R4_f20_ge: x_Gr_Liv_Area - 747 t405_leaf4 >= 0
 T405_R4_f3_ge: Overall_Qual_calc - 1.000005e+06 t405_leaf4 >= -1e+06
 T405_R4_f37_ge: x_Open_Porch_SF - 260 t405_leaf4 >= 0
 TREE_406_ONEHOT: t406_leaf0 = 1
 TREE_407_ONEHOT: t407_leaf0 = 1
 TREE_408_ONEHOT: t408_leaf0 + t408_leaf1 + t408_leaf2 + t408_leaf3
   + t408_leaf4 = 1
 T408_L0_f18_le: const_2nd_Flr_SF + 998389 t408_leaf0 <= 1e+06
 T408_L0_f20_le: x_Gr_Liv_Area + 7128 t408_leaf0 <= 10000
 T408_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t408_leaf0 <= 1e+06
 T408_L1_f18_le: const_2nd_Flr_SF + 998389 t408_leaf1 <= 1e+06
 T408_L1_f20_le: x_Gr_Liv_Area + 7128 t408_leaf1 <= 10000
 T408_R1_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t408_leaf1
   >= -1e+06
 T408_L2_f18_le: const_2nd_Flr_SF + 998389 t408_leaf2 <= 1e+06
 T408_R2_f20_ge: x_Gr_Liv_Area - 2872 t408_leaf2 >= 0
 T408_L2_f25_le: x_Bedroom_AbvGr + 2 t408_leaf2 <= 6
 T408_L3_f18_le: const_2nd_Flr_SF + 998389 t408_leaf3 <= 1e+06
 T408_R3_f20_ge: x_Gr_Liv_Area - 2872 t408_leaf3 >= 0
 T408_R3_f25_ge: x_Bedroom_AbvGr - 4 t408_leaf3 >= 0
 T408_R4_f18_ge: const_2nd_Flr_SF - 1.001611e+06 t408_leaf4 >= -1e+06
 TREE_409_ONEHOT: t409_leaf0 + t409_leaf1 + t409_leaf2 + t409_leaf3
   + t409_leaf4 + t409_leaf5 = 1
 T409_L0_f31_le: const_Garage_Yr_Blt + 998069 t409_leaf0 <= 1e+06
 T409_L0_f1_le: const_Lot_Area + 989663 t409_leaf0 <= 1e+06
 T409_L1_f31_le: const_Garage_Yr_Blt + 998069 t409_leaf1 <= 1e+06
 T409_R1_f1_ge: const_Lot_Area - 1.010337e+06 t409_leaf1 >= -1e+06
 T409_L1_f191_le: x_ex2_is_VinylSd <= 1
 T409_L2_f31_le: const_Garage_Yr_Blt + 998069 t409_leaf2 <= 1e+06
 T409_R2_f1_ge: const_Lot_Area - 1.010337e+06 t409_leaf2 >= -1e+06
 T409_R2_f191_ge: x_ex2_is_VinylSd - t409_leaf2 >= 0
 T409_R3_f31_ge: const_Garage_Yr_Blt - 1.001931e+06 t409_leaf3 >= -1e+06
 T409_L3_f293_le: const_Sale_Condition_Abnorml + 999999 t409_leaf3
   <= 1e+06
 T409_R4_f31_ge: const_Garage_Yr_Blt - 1.001931e+06 t409_leaf4 >= -1e+06
 T409_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t409_leaf4
   >= -1e+06
 T409_L4_f36_le: x_Wood_Deck_SF + 99819 t409_leaf4 <= 100000
 T409_R5_f31_ge: const_Garage_Yr_Blt - 1.001931e+06 t409_leaf5 >= -1e+06
 T409_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t409_leaf5
   >= -1e+06
 T409_R5_f36_ge: x_Wood_Deck_SF - 181 t409_leaf5 >= 0
 TREE_410_ONEHOT: t410_leaf0 + t410_leaf1 + t410_leaf2 + t410_leaf3 = 1
 T410_L0_f17_le: x_1st_Flr_SF + 2800 t410_leaf0 <= 5000
 T410_L0_f63_le: const_MS_Zoning_FV + 999999 t410_leaf0 <= 1e+06
 T410_L0_f8_le: x_Exter_Qual + t410_leaf0 <= 4
 T410_L1_f17_le: x_1st_Flr_SF + 2800 t410_leaf1 <= 5000
 T410_L1_f63_le: const_MS_Zoning_FV + 999999 t410_leaf1 <= 1e+06
 T410_R1_f8_ge: x_Exter_Qual - 4 t410_leaf1 >= -1
 T410_L2_f17_le: x_1st_Flr_SF + 2800 t410_leaf2 <= 5000
 T410_R2_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t410_leaf2 >= -1e+06
 T410_R3_f17_ge: x_1st_Flr_SF - 2200 t410_leaf3 >= 0
 TREE_411_ONEHOT: t411_leaf0 + t411_leaf1 + t411_leaf2 = 1
 T411_L0_f72_le: const_Alley_Pave + 999999 t411_leaf0 <= 1e+06
 T411_L0_f114_le: const_Neighborhood_StoneBr + 999999 t411_leaf0 <= 1e+06
 T411_L1_f72_le: const_Alley_Pave + 999999 t411_leaf1 <= 1e+06
 T411_R1_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t411_leaf1
   >= -1e+06
 T411_R2_f72_ge: const_Alley_Pave - 1.000001e+06 t411_leaf2 >= -1e+06
 TREE_412_ONEHOT: t412_leaf0 + t412_leaf1 + t412_leaf2 = 1
 T412_L0_f274_le: const_Mo_Sold_4 + 999999 t412_leaf0 <= 1e+06
 T412_R1_f274_ge: const_Mo_Sold_4 - 1.000001e+06 t412_leaf1 >= -1e+06
 T412_L1_f0_le: const_Lot_Frontage + 999923 t412_leaf1 <= 1e+06
 T412_R2_f274_ge: const_Mo_Sold_4 - 1.000001e+06 t412_leaf2 >= -1e+06
 T412_R2_f0_ge: const_Lot_Frontage - 1.000077e+06 t412_leaf2 >= -1e+06
 TREE_413_ONEHOT: t413_leaf0 + t413_leaf1 + t413_leaf2 + t413_leaf3
   + t413_leaf4 = 1
 T413_L0_f1_le: const_Lot_Area + 994000 t413_leaf0 <= 1e+06
 T413_R1_f1_ge: const_Lot_Area - 1.006e+06 t413_leaf1 >= -1e+06
 T413_L1_f46_le: const_MS_SubClass_30 + 999999 t413_leaf1 <= 1e+06
 T413_L1_f40_le: x_Screen_Porch + 99892 t413_leaf1 <= 100000
 T413_R2_f1_ge: const_Lot_Area - 1.006e+06 t413_leaf2 >= -1e+06
 T413_L2_f46_le: const_MS_SubClass_30 + 999999 t413_leaf2 <= 1e+06
 T413_R2_f40_ge: x_Screen_Porch - 108 t413_leaf2 >= 0
 T413_R3_f1_ge: const_Lot_Area - 1.006e+06 t413_leaf3 >= -1e+06
 T413_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t413_leaf3 >= -1e+06
 T413_L3_f38_le: x_Enclosed_Porch + 99923 t413_leaf3 <= 100000
 T413_R4_f1_ge: const_Lot_Area - 1.006e+06 t413_leaf4 >= -1e+06
 T413_R4_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t413_leaf4 >= -1e+06
 T413_R4_f38_ge: x_Enclosed_Porch - 77 t413_leaf4 >= 0
 TREE_414_ONEHOT: t414_leaf0 + t414_leaf1 + t414_leaf2 + t414_leaf3 = 1
 T414_L0_f63_le: const_MS_Zoning_FV + 999999 t414_leaf0 <= 1e+06
 T414_L0_f11_le: x_Bsmt_Cond + t414_leaf0 <= 4
 T414_L0_f96_le: const_Neighborhood_Edwards + 999999 t414_leaf0 <= 1e+06
 T414_L1_f63_le: const_MS_Zoning_FV + 999999 t414_leaf1 <= 1e+06
 T414_L1_f11_le: x_Bsmt_Cond + t414_leaf1 <= 4
 T414_R1_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t414_leaf1
   >= -1e+06
 T414_L2_f63_le: const_MS_Zoning_FV + 999999 t414_leaf2 <= 1e+06
 T414_R2_f11_ge: x_Bsmt_Cond - 4 t414_leaf2 >= -1
 T414_R3_f63_ge: const_MS_Zoning_FV - 1.000001e+06 t414_leaf3 >= -1e+06
 TREE_415_ONEHOT: t415_leaf0 + t415_leaf1 + t415_leaf2 + t415_leaf3
   + t415_leaf4 = 1
 T415_L0_f7_le: x_Mas_Vnr_Area + 1530 t415_leaf0 <= 2000
 T415_L0_f23_le: x_Full_Bath + 2 t415_leaf0 <= 4
 T415_L0_f297_le: const_Sale_Condition_Normal + 999999 t415_leaf0 <= 1e+06
 T415_L1_f7_le: x_Mas_Vnr_Area + 1530 t415_leaf1 <= 2000
 T415_L1_f23_le: x_Full_Bath + 2 t415_leaf1 <= 4
 T415_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t415_leaf1
   >= -1e+06
 T415_L2_f7_le: x_Mas_Vnr_Area + 1530 t415_leaf2 <= 2000
 T415_R2_f23_ge: x_Full_Bath - 2 t415_leaf2 >= 0
 T415_L2_f8_le: x_Exter_Qual + t415_leaf2 <= 4
 T415_L3_f7_le: x_Mas_Vnr_Area + 1530 t415_leaf3 <= 2000
 T415_R3_f23_ge: x_Full_Bath - 2 t415_leaf3 >= 0
 T415_R3_f8_ge: x_Exter_Qual - 4 t415_leaf3 >= -1
 T415_R4_f7_ge: x_Mas_Vnr_Area - 470 t415_leaf4 >= 0
 TREE_416_ONEHOT: t416_leaf0 + t416_leaf1 + t416_leaf2 + t416_leaf3
   + t416_leaf4 + t416_leaf5 + t416_leaf6 = 1
 T416_L0_f35_le: x_Garage_Cond + 2 t416_leaf0 <= 4
 T416_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t416_leaf0
   <= 1e+06
 T416_L0_f1_le: const_Lot_Area + 989010 t416_leaf0 <= 1e+06
 T416_L1_f35_le: x_Garage_Cond + 2 t416_leaf1 <= 4
 T416_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t416_leaf1
   <= 1e+06
 T416_R1_f1_ge: const_Lot_Area - 1.01099e+06 t416_leaf1 >= -1e+06
 T416_L2_f35_le: x_Garage_Cond + 2 t416_leaf2 <= 4
 T416_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t416_leaf2
   >= -1e+06
 T416_L2_f28_le: x_TotRms_AbvGrd + 8 t416_leaf2 <= 15
 T416_L3_f35_le: x_Garage_Cond + 2 t416_leaf3 <= 4
 T416_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t416_leaf3
   >= -1e+06
 T416_R3_f28_ge: x_TotRms_AbvGrd - 7 t416_leaf3 >= 0
 T416_R4_f35_ge: x_Garage_Cond - 3 t416_leaf4 >= -1
 T416_L4_f38_le: x_Enclosed_Porch + 99736 t416_leaf4 <= 100000
 T416_L4_f147_le: const_Roof_Style_Flat + 999999 t416_leaf4 <= 1e+06
 T416_R5_f35_ge: x_Garage_Cond - 3 t416_leaf5 >= -1
 T416_L5_f38_le: x_Enclosed_Porch + 99736 t416_leaf5 <= 100000
 T416_R5_f147_ge: const_Roof_Style_Flat - 1.000001e+06 t416_leaf5
   >= -1e+06
 T416_R6_f35_ge: x_Garage_Cond - 3 t416_leaf6 >= -1
 T416_R6_f38_ge: x_Enclosed_Porch - 264 t416_leaf6 >= 0
 TREE_417_ONEHOT: t417_leaf0 = 1
 TREE_418_ONEHOT: t418_leaf0 + t418_leaf1 = 1
 T418_L0_f12_le: x_BsmtFin_SF_1 + 998035 t418_leaf0 <= 1e+06
 T418_R1_f12_ge: x_BsmtFin_SF_1 - 1965 t418_leaf1 >= 0
 TREE_419_ONEHOT: t419_leaf0 + t419_leaf1 + t419_leaf2 + t419_leaf3
   + t419_leaf4 + t419_leaf5 + t419_leaf6 + t419_leaf7 = 1
 T419_L0_f3_le: Overall_Qual_calc + 999995 t419_leaf0 <= 1e+06
 T419_L0_f10_le: const_Bsmt_Qual + 1e+06 t419_leaf0 <= 1e+06
 T419_L0_f0_le: const_Lot_Frontage + 999934 t419_leaf0 <= 1e+06
 T419_L1_f3_le: Overall_Qual_calc + 999995 t419_leaf1 <= 1e+06
 T419_L1_f10_le: const_Bsmt_Qual + 1e+06 t419_leaf1 <= 1e+06
 T419_R1_f0_ge: const_Lot_Frontage - 1.000066e+06 t419_leaf1 >= -1e+06
 T419_L2_f3_le: Overall_Qual_calc + 999995 t419_leaf2 <= 1e+06
 T419_R2_f10_ge: const_Bsmt_Qual - 1e+06 t419_leaf2 >= -1e+06
 T419_L2_f0_le: const_Lot_Frontage + 999978 t419_leaf2 <= 1e+06
 T419_L3_f3_le: Overall_Qual_calc + 999995 t419_leaf3 <= 1e+06
 T419_R3_f10_ge: const_Bsmt_Qual - 1e+06 t419_leaf3 >= -1e+06
 T419_R3_f0_ge: const_Lot_Frontage - 1.000022e+06 t419_leaf3 >= -1e+06
 T419_R4_f3_ge: Overall_Qual_calc - 1.000005e+06 t419_leaf4 >= -1e+06
 T419_L4_f23_le: x_Full_Bath + 2 t419_leaf4 <= 4
 T419_L4_f119_le: const_Condition_1_Norm + 999999 t419_leaf4 <= 1e+06
 T419_R5_f3_ge: Overall_Qual_calc - 1.000005e+06 t419_leaf5 >= -1e+06
 T419_L5_f23_le: x_Full_Bath + 2 t419_leaf5 <= 4
 T419_R5_f119_ge: const_Condition_1_Norm - 1.000001e+06 t419_leaf5
   >= -1e+06
 T419_R6_f3_ge: Overall_Qual_calc - 1.000005e+06 t419_leaf6 >= -1e+06
 T419_R6_f23_ge: x_Full_Bath - 2 t419_leaf6 >= 0
 T419_L6_f3_le: Overall_Qual_calc + 999993 t419_leaf6 <= 1e+06
 T419_R7_f3_ge: Overall_Qual_calc - 1.000005e+06 t419_leaf7 >= -1e+06
 T419_R7_f23_ge: x_Full_Bath - 2 t419_leaf7 >= 0
 T419_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t419_leaf7 >= -1e+06
 TREE_420_ONEHOT: t420_leaf0 = 1
 TREE_421_ONEHOT: t421_leaf0 + t421_leaf1 + t421_leaf2 + t421_leaf3 = 1
 T421_L0_f9_le: x_Exter_Cond + 2 t421_leaf0 <= 4
 T421_L0_f23_le: x_Full_Bath + 2 t421_leaf0 <= 4
 T421_L0_f0_le: const_Lot_Frontage + 999939 t421_leaf0 <= 1e+06
 T421_L1_f9_le: x_Exter_Cond + 2 t421_leaf1 <= 4
 T421_L1_f23_le: x_Full_Bath + 2 t421_leaf1 <= 4
 T421_R1_f0_ge: const_Lot_Frontage - 1.000061e+06 t421_leaf1 >= -1e+06
 T421_L2_f9_le: x_Exter_Cond + 2 t421_leaf2 <= 4
 T421_R2_f23_ge: x_Full_Bath - 2 t421_leaf2 >= 0
 T421_R3_f9_ge: x_Exter_Cond - 3 t421_leaf3 >= -1
 TREE_422_ONEHOT: t422_leaf0 + t422_leaf1 + t422_leaf2 + t422_leaf3
   + t422_leaf4 = 1
 T422_L0_f37_le: x_Open_Porch_SF + 99742 t422_leaf0 <= 100000
 T422_L0_f11_le: x_Bsmt_Cond + 2 t422_leaf0 <= 4
 T422_L0_f297_le: const_Sale_Condition_Normal + 999999 t422_leaf0 <= 1e+06
 T422_L1_f37_le: x_Open_Porch_SF + 99742 t422_leaf1 <= 100000
 T422_L1_f11_le: x_Bsmt_Cond + 2 t422_leaf1 <= 4
 T422_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t422_leaf1
   >= -1e+06
 T422_L2_f37_le: x_Open_Porch_SF + 99742 t422_leaf2 <= 100000
 T422_R2_f11_ge: x_Bsmt_Cond - 3 t422_leaf2 >= -1
 T422_L2_f117_le: const_Condition_1_Artery + 999999 t422_leaf2 <= 1e+06
 T422_L3_f37_le: x_Open_Porch_SF + 99742 t422_leaf3 <= 100000
 T422_R3_f11_ge: x_Bsmt_Cond - 3 t422_leaf3 >= -1
 T422_R3_f117_ge: const_Condition_1_Artery - 1.000001e+06 t422_leaf3
   >= -1e+06
 T422_R4_f37_ge: x_Open_Porch_SF - 258 t422_leaf4 >= 0
 TREE_423_ONEHOT: t423_leaf0 + t423_leaf1 + t423_leaf2 + t423_leaf3 = 1
 T423_L0_f38_le: x_Enclosed_Porch + 99710 t423_leaf0 <= 100000
 T423_L0_f25_le: x_Bedroom_AbvGr + t423_leaf0 <= 6
 T423_L0_f11_le: x_Bsmt_Cond + 3 t423_leaf0 <= 4
 T423_L1_f38_le: x_Enclosed_Porch + 99710 t423_leaf1 <= 100000
 T423_L1_f25_le: x_Bedroom_AbvGr + t423_leaf1 <= 6
 T423_R1_f11_ge: x_Bsmt_Cond - 2 t423_leaf1 >= -1
 T423_L2_f38_le: x_Enclosed_Porch + 99710 t423_leaf2 <= 100000
 T423_R2_f25_ge: x_Bedroom_AbvGr - 5 t423_leaf2 >= 0
 T423_R3_f38_ge: x_Enclosed_Porch - 290 t423_leaf3 >= 0
 TREE_424_ONEHOT: t424_leaf0 + t424_leaf1 + t424_leaf2 + t424_leaf3
   + t424_leaf4 + t424_leaf5 = 1
 T424_L0_f3_le: Overall_Qual_calc + 999997 t424_leaf0 <= 1e+06
 T424_L0_f38_le: x_Enclosed_Porch + 99974 t424_leaf0 <= 100000
 T424_L1_f3_le: Overall_Qual_calc + 999997 t424_leaf1 <= 1e+06
 T424_R1_f38_ge: x_Enclosed_Porch - 26 t424_leaf1 >= 0
 T424_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t424_leaf2 >= -1e+06
 T424_L2_f119_le: const_Condition_1_Norm + 999999 t424_leaf2 <= 1e+06
 T424_L2_f14_le: x_Bsmt_Unf_SF + 99084 t424_leaf2 <= 100000
 T424_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t424_leaf3 >= -1e+06
 T424_L3_f119_le: const_Condition_1_Norm + 999999 t424_leaf3 <= 1e+06
 T424_R3_f14_ge: x_Bsmt_Unf_SF - 916 t424_leaf3 >= 0
 T424_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t424_leaf4 >= -1e+06
 T424_R4_f119_ge: const_Condition_1_Norm - 1.000001e+06 t424_leaf4
   >= -1e+06
 T424_L4_f164_le: x_ex1_is_BrkFace <= 1
 T424_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t424_leaf5 >= -1e+06
 T424_R5_f119_ge: const_Condition_1_Norm - 1.000001e+06 t424_leaf5
   >= -1e+06
 T424_R5_f164_ge: x_ex1_is_BrkFace - t424_leaf5 >= 0
 TREE_425_ONEHOT: t425_leaf0 + t425_leaf1 + t425_leaf2 = 1
 T425_L0_f92_le: const_Neighborhood_BrkSide + 999999 t425_leaf0 <= 1e+06
 T425_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t425_leaf0 <= 1e+06
 T425_L1_f92_le: const_Neighborhood_BrkSide + 999999 t425_leaf1 <= 1e+06
 T425_R1_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t425_leaf1
   >= -1e+06
 T425_R2_f92_ge: const_Neighborhood_BrkSide - 1.000001e+06 t425_leaf2
   >= -1e+06
 TREE_426_ONEHOT: t426_leaf0 + t426_leaf1 + t426_leaf2 + t426_leaf3
   + t426_leaf4 + t426_leaf5 + t426_leaf6 = 1
 T426_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t426_leaf0
   <= 1e+06
 T426_L0_f20_le: x_Gr_Liv_Area + 6605 t426_leaf0 <= 10000
 T426_L0_f35_le: x_Garage_Cond + 2 t426_leaf0 <= 4
 T426_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t426_leaf1
   <= 1e+06
 T426_L1_f20_le: x_Gr_Liv_Area + 6605 t426_leaf1 <= 10000
 T426_R1_f35_ge: x_Garage_Cond - 3 t426_leaf1 >= -1
 T426_L2_f293_le: const_Sale_Condition_Abnorml + 999999 t426_leaf2
   <= 1e+06
 T426_R2_f20_ge: x_Gr_Liv_Area - 3395 t426_leaf2 >= 0
 T426_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t426_leaf3
   >= -1e+06
 T426_L3_f81_le: const_Lot_Config_Corner + 999999 t426_leaf3 <= 1e+06
 T426_L3_f1_le: const_Lot_Area + 995329 t426_leaf3 <= 1e+06
 T426_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t426_leaf4
   >= -1e+06
 T426_L4_f81_le: const_Lot_Config_Corner + 999999 t426_leaf4 <= 1e+06
 T426_R4_f1_ge: const_Lot_Area - 1.004671e+06 t426_leaf4 >= -1e+06
 T426_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t426_leaf5
   >= -1e+06
 T426_R5_f81_ge: const_Lot_Config_Corner - 1.000001e+06 t426_leaf5
   >= -1e+06
 T426_L5_f20_le: x_Gr_Liv_Area + 8872 t426_leaf5 <= 10000
 T426_R6_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t426_leaf6
   >= -1e+06
 T426_R6_f81_ge: const_Lot_Config_Corner - 1.000001e+06 t426_leaf6
   >= -1e+06
 T426_R6_f20_ge: x_Gr_Liv_Area - 1128 t426_leaf6 >= 0
 TREE_427_ONEHOT: t427_leaf0 + t427_leaf1 + t427_leaf2 + t427_leaf3 = 1
 T427_L0_f1_le: const_Lot_Area + 972303 t427_leaf0 <= 1e+06
 T427_L0_f1_le: const_Lot_Area + 976743 t427_leaf0 <= 1e+06
 T427_L0_f194_le: x_mvt_is_BrkCmn <= 1
 T427_L1_f1_le: const_Lot_Area + 972303 t427_leaf1 <= 1e+06
 T427_L1_f1_le: const_Lot_Area + 976743 t427_leaf1 <= 1e+06
 T427_R1_f194_ge: x_mvt_is_BrkCmn - t427_leaf1 >= 0
 T427_L2_f1_le: const_Lot_Area + 972303 t427_leaf2 <= 1e+06
 T427_R2_f1_ge: const_Lot_Area - 1.023257e+06 t427_leaf2 >= -1e+06
 T427_R3_f1_ge: const_Lot_Area - 1.027697e+06 t427_leaf3 >= -1e+06
 TREE_428_ONEHOT: t428_leaf0 + t428_leaf1 + t428_leaf2 + t428_leaf3
   + t428_leaf4 = 1
 T428_L0_f38_le: x_Enclosed_Porch + 99732 t428_leaf0 <= 100000
 T428_L0_f5_le: const_Year_Built + 998054 t428_leaf0 <= 1e+06
 T428_L0_f95_le: const_Neighborhood_Crawfor + 999999 t428_leaf0 <= 1e+06
 T428_L1_f38_le: x_Enclosed_Porch + 99732 t428_leaf1 <= 100000
 T428_L1_f5_le: const_Year_Built + 998054 t428_leaf1 <= 1e+06
 T428_R1_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t428_leaf1
   >= -1e+06
 T428_L2_f38_le: x_Enclosed_Porch + 99732 t428_leaf2 <= 100000
 T428_R2_f5_ge: const_Year_Built - 1.001946e+06 t428_leaf2 >= -1e+06
 T428_L2_f5_le: const_Year_Built + 998050 t428_leaf2 <= 1e+06
 T428_L3_f38_le: x_Enclosed_Porch + 99732 t428_leaf3 <= 100000
 T428_R3_f5_ge: const_Year_Built - 1.001946e+06 t428_leaf3 >= -1e+06
 T428_R3_f5_ge: const_Year_Built - 1.00195e+06 t428_leaf3 >= -1e+06
 T428_R4_f38_ge: x_Enclosed_Porch - 268 t428_leaf4 >= 0
 TREE_429_ONEHOT: t429_leaf0 + t429_leaf1 + t429_leaf2 + t429_leaf3
   + t429_leaf4 + t429_leaf5 = 1
 T429_L0_f30_le: x_Fireplace_Qu + 2 t429_leaf0 <= 4
 T429_L0_f22_le: const_Bsmt_Half_Bath + 999999 t429_leaf0 <= 1e+06
 T429_L1_f30_le: x_Fireplace_Qu + 2 t429_leaf1 <= 4
 T429_R1_f22_ge: const_Bsmt_Half_Bath - 1.000001e+06 t429_leaf1 >= -1e+06
 T429_R2_f30_ge: x_Fireplace_Qu - 3 t429_leaf2 >= -1
 T429_L2_f96_le: const_Neighborhood_Edwards + 999999 t429_leaf2 <= 1e+06
 T429_L2_f296_le: const_Sale_Condition_Family + 999999 t429_leaf2 <= 1e+06
 T429_R3_f30_ge: x_Fireplace_Qu - 3 t429_leaf3 >= -1
 T429_L3_f96_le: const_Neighborhood_Edwards + 999999 t429_leaf3 <= 1e+06
 T429_R3_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t429_leaf3
   >= -1e+06
 T429_R4_f30_ge: x_Fireplace_Qu - 3 t429_leaf4 >= -1
 T429_R4_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t429_leaf4
   >= -1e+06
 T429_L4_f150_le: const_Roof_Style_Hip + 999999 t429_leaf4 <= 1e+06
 T429_R5_f30_ge: x_Fireplace_Qu - 3 t429_leaf5 >= -1
 T429_R5_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t429_leaf5
   >= -1e+06
 T429_R5_f150_ge: const_Roof_Style_Hip - 1.000001e+06 t429_leaf5 >= -1e+06
 TREE_430_ONEHOT: t430_leaf0 + t430_leaf1 + t430_leaf2 + t430_leaf3
   + t430_leaf4 + t430_leaf5 + t430_leaf6 = 1
 T430_L0_f14_le: x_Bsmt_Unf_SF + 99863 t430_leaf0 <= 100000
 T430_L0_f38_le: x_Enclosed_Porch + 99776 t430_leaf0 <= 100000
 T430_L0_f272_le: const_Mo_Sold_2 + 999999 t430_leaf0 <= 1e+06
 T430_L1_f14_le: x_Bsmt_Unf_SF + 99863 t430_leaf1 <= 100000
 T430_L1_f38_le: x_Enclosed_Porch + 99776 t430_leaf1 <= 100000
 T430_R1_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t430_leaf1 >= -1e+06
 T430_L2_f14_le: x_Bsmt_Unf_SF + 99863 t430_leaf2 <= 100000
 T430_R2_f38_ge: x_Enclosed_Porch - 224 t430_leaf2 >= 0
 T430_R3_f14_ge: x_Bsmt_Unf_SF - 137 t430_leaf3 >= 0
 T430_L3_f1_le: const_Lot_Area + 979569 t430_leaf3 <= 1e+06
 T430_L3_f18_le: const_2nd_Flr_SF + 999360 t430_leaf3 <= 1e+06
 T430_R4_f14_ge: x_Bsmt_Unf_SF - 137 t430_leaf4 >= 0
 T430_L4_f1_le: const_Lot_Area + 979569 t430_leaf4 <= 1e+06
 T430_R4_f18_ge: const_2nd_Flr_SF - 1.00064e+06 t430_leaf4 >= -1e+06
 T430_R5_f14_ge: x_Bsmt_Unf_SF - 137 t430_leaf5 >= 0
 T430_R5_f1_ge: const_Lot_Area - 1.020431e+06 t430_leaf5 >= -1e+06
 T430_L5_f38_le: x_Enclosed_Porch + 99806 t430_leaf5 <= 100000
 T430_R6_f14_ge: x_Bsmt_Unf_SF - 137 t430_leaf6 >= 0
 T430_R6_f1_ge: const_Lot_Area - 1.020431e+06 t430_leaf6 >= -1e+06
 T430_R6_f38_ge: x_Enclosed_Porch - 194 t430_leaf6 >= 0
 TREE_431_ONEHOT: t431_leaf0 + t431_leaf1 + t431_leaf2 + t431_leaf3 = 1
 T431_L0_f203_le: const_Foundation_Slab + 999999 t431_leaf0 <= 1e+06
 T431_L0_f37_le: x_Open_Porch_SF + 99713 t431_leaf0 <= 100000
 T431_L0_f20_le: x_Gr_Liv_Area + 7351 t431_leaf0 <= 10000
 T431_L1_f203_le: const_Foundation_Slab + 999999 t431_leaf1 <= 1e+06
 T431_L1_f37_le: x_Open_Porch_SF + 99713 t431_leaf1 <= 100000
 T431_R1_f20_ge: x_Gr_Liv_Area - 2649 t431_leaf1 >= 0
 T431_L2_f203_le: const_Foundation_Slab + 999999 t431_leaf2 <= 1e+06
 T431_R2_f37_ge: x_Open_Porch_SF - 287 t431_leaf2 >= 0
 T431_R3_f203_ge: const_Foundation_Slab - 1.000001e+06 t431_leaf3
   >= -1e+06
 TREE_432_ONEHOT: t432_leaf0 + t432_leaf1 + t432_leaf2 + t432_leaf3
   + t432_leaf4 + t432_leaf5 = 1
 T432_L0_f1_le: const_Lot_Area + 980478 t432_leaf0 <= 1e+06
 T432_L0_f15_le: const_Total_Bsmt_SF + 997864 t432_leaf0 <= 1e+06
 T432_L0_f38_le: x_Enclosed_Porch + 99776 t432_leaf0 <= 100000
 T432_L1_f1_le: const_Lot_Area + 980478 t432_leaf1 <= 1e+06
 T432_L1_f15_le: const_Total_Bsmt_SF + 997864 t432_leaf1 <= 1e+06
 T432_R1_f38_ge: x_Enclosed_Porch - 224 t432_leaf1 >= 0
 T432_L2_f1_le: const_Lot_Area + 980478 t432_leaf2 <= 1e+06
 T432_R2_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t432_leaf2 >= -1e+06
 T432_R3_f1_ge: const_Lot_Area - 1.019522e+06 t432_leaf3 >= -1e+06
 T432_L3_f25_le: x_Bedroom_AbvGr + 2 t432_leaf3 <= 6
 T432_L3_f297_le: const_Sale_Condition_Normal + 999999 t432_leaf3 <= 1e+06
 T432_R4_f1_ge: const_Lot_Area - 1.019522e+06 t432_leaf4 >= -1e+06
 T432_L4_f25_le: x_Bedroom_AbvGr + 2 t432_leaf4 <= 6
 T432_R4_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t432_leaf4
   >= -1e+06
 T432_R5_f1_ge: const_Lot_Area - 1.019522e+06 t432_leaf5 >= -1e+06
 T432_R5_f25_ge: x_Bedroom_AbvGr - 4 t432_leaf5 >= 0
 TREE_433_ONEHOT: t433_leaf0 + t433_leaf1 + t433_leaf2 = 1
 T433_L0_f37_le: x_Open_Porch_SF + 99792 t433_leaf0 <= 100000
 T433_L0_f218_le: x_b2_is_ALQ <= 1
 T433_L1_f37_le: x_Open_Porch_SF + 99792 t433_leaf1 <= 100000
 T433_R1_f218_ge: x_b2_is_ALQ - t433_leaf1 >= 0
 T433_R2_f37_ge: x_Open_Porch_SF - 208 t433_leaf2 >= 0
 TREE_434_ONEHOT: t434_leaf0 = 1
 TREE_435_ONEHOT: t435_leaf0 + t435_leaf1 + t435_leaf2 + t435_leaf3
   + t435_leaf4 = 1
 T435_L0_f38_le: x_Enclosed_Porch + 99806 t435_leaf0 <= 100000
 T435_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t435_leaf0 <= 1e+06
 T435_L0_f38_le: x_Enclosed_Porch + 99936 t435_leaf0 <= 100000
 T435_L1_f38_le: x_Enclosed_Porch + 99806 t435_leaf1 <= 100000
 T435_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t435_leaf1 <= 1e+06
 T435_R1_f38_ge: x_Enclosed_Porch - 64 t435_leaf1 >= 0
 T435_L2_f38_le: x_Enclosed_Porch + 99806 t435_leaf2 <= 100000
 T435_R2_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t435_leaf2
   >= -1e+06
 T435_L2_f4_le: const_Overall_Cond + 999993 t435_leaf2 <= 1e+06
 T435_L3_f38_le: x_Enclosed_Porch + 99806 t435_leaf3 <= 100000
 T435_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t435_leaf3
   >= -1e+06
 T435_R3_f4_ge: const_Overall_Cond - 1.000007e+06 t435_leaf3 >= -1e+06
 T435_R4_f38_ge: x_Enclosed_Porch - 194 t435_leaf4 >= 0
 TREE_436_ONEHOT: t436_leaf0 = 1
 TREE_437_ONEHOT: t437_leaf0 + t437_leaf1 + t437_leaf2 = 1
 T437_L0_f3_le: Overall_Qual_calc + 999996 t437_leaf0 <= 1e+06
 T437_L0_f80_le: const_Land_Contour_Lvl + 999999 t437_leaf0 <= 1e+06
 T437_L1_f3_le: Overall_Qual_calc + 999996 t437_leaf1 <= 1e+06
 T437_R1_f80_ge: const_Land_Contour_Lvl - 1.000001e+06 t437_leaf1
   >= -1e+06
 T437_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t437_leaf2 >= -1e+06
 TREE_438_ONEHOT: t438_leaf0 + t438_leaf1 = 1
 T438_L0_f119_le: const_Condition_1_Norm + 999999 t438_leaf0 <= 1e+06
 T438_R1_f119_ge: const_Condition_1_Norm - 1.000001e+06 t438_leaf1
   >= -1e+06
 TREE_439_ONEHOT: t439_leaf0 + t439_leaf1 + t439_leaf2 + t439_leaf3
   + t439_leaf4 = 1
 T439_L0_f231_le: central_air_no + 999999 t439_leaf0 <= 1e+06
 T439_R1_f231_ge: central_air_no - 1.000001e+06 t439_leaf1 >= -1e+06
 T439_L1_f12_le: x_BsmtFin_SF_1 + 999684 t439_leaf1 <= 1e+06
 T439_L1_f275_le: const_Mo_Sold_5 + 999999 t439_leaf1 <= 1e+06
 T439_R2_f231_ge: central_air_no - 1.000001e+06 t439_leaf2 >= -1e+06
 T439_L2_f12_le: x_BsmtFin_SF_1 + 999684 t439_leaf2 <= 1e+06
 T439_R2_f275_ge: const_Mo_Sold_5 - 1.000001e+06 t439_leaf2 >= -1e+06
 T439_R3_f231_ge: central_air_no - 1.000001e+06 t439_leaf3 >= -1e+06
 T439_R3_f12_ge: x_BsmtFin_SF_1 - 316 t439_leaf3 >= 0
 T439_L3_f12_le: x_BsmtFin_SF_1 + 999363 t439_leaf3 <= 1e+06
 T439_R4_f231_ge: central_air_no - 1.000001e+06 t439_leaf4 >= -1e+06
 T439_R4_f12_ge: x_BsmtFin_SF_1 - 316 t439_leaf4 >= 0
 T439_R4_f12_ge: x_BsmtFin_SF_1 - 637 t439_leaf4 >= 0
 TREE_440_ONEHOT: t440_leaf0 + t440_leaf1 + t440_leaf2 + t440_leaf3
   + t440_leaf4 + t440_leaf5 = 1
 T440_L0_f25_le: x_Bedroom_AbvGr + 3 t440_leaf0 <= 6
 T440_L0_f20_le: x_Gr_Liv_Area + 8205 t440_leaf0 <= 10000
 T440_L1_f25_le: x_Bedroom_AbvGr + 3 t440_leaf1 <= 6
 T440_R1_f20_ge: x_Gr_Liv_Area - 1795 t440_leaf1 >= 0
 T440_L1_f174_le: x_ex1_is_VinylSd <= 1
 T440_L2_f25_le: x_Bedroom_AbvGr + 3 t440_leaf2 <= 6
 T440_R2_f20_ge: x_Gr_Liv_Area - 1795 t440_leaf2 >= 0
 T440_R2_f174_ge: x_ex1_is_VinylSd - t440_leaf2 >= 0
 T440_R3_f25_ge: x_Bedroom_AbvGr - 3 t440_leaf3 >= 0
 T440_L3_f20_le: x_Gr_Liv_Area + 9024 t440_leaf3 <= 10000
 T440_L3_f31_le: const_Garage_Yr_Blt + 998033 t440_leaf3 <= 1e+06
 T440_R4_f25_ge: x_Bedroom_AbvGr - 3 t440_leaf4 >= 0
 T440_L4_f20_le: x_Gr_Liv_Area + 9024 t440_leaf4 <= 10000
 T440_R4_f31_ge: const_Garage_Yr_Blt - 1.001967e+06 t440_leaf4 >= -1e+06
 T440_R5_f25_ge: x_Bedroom_AbvGr - 3 t440_leaf5 >= 0
 T440_R5_f20_ge: x_Gr_Liv_Area - 976 t440_leaf5 >= 0
 TREE_441_ONEHOT: t441_leaf0 + t441_leaf1 + t441_leaf2 + t441_leaf3
   + t441_leaf4 = 1
 T441_L0_f34_le: x_Garage_Qual + 3 t441_leaf0 <= 4
 T441_L0_f1_le: const_Lot_Area + 991200 t441_leaf0 <= 1e+06
 T441_L1_f34_le: x_Garage_Qual + 3 t441_leaf1 <= 4
 T441_R1_f1_ge: const_Lot_Area - 1.0088e+06 t441_leaf1 >= -1e+06
 T441_R2_f34_ge: x_Garage_Qual - 2 t441_leaf2 >= -1
 T441_L2_f46_le: const_MS_SubClass_30 + 999999 t441_leaf2 <= 1e+06
 T441_R3_f34_ge: x_Garage_Qual - 2 t441_leaf3 >= -1
 T441_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t441_leaf3 >= -1e+06
 T441_L3_f17_le: x_1st_Flr_SF + 4180 t441_leaf3 <= 5000
 T441_R4_f34_ge: x_Garage_Qual - 2 t441_leaf4 >= -1
 T441_R4_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t441_leaf4 >= -1e+06
 T441_R4_f17_ge: x_1st_Flr_SF - 820 t441_leaf4 >= 0
 TREE_442_ONEHOT: t442_leaf0 + t442_leaf1 + t442_leaf2 = 1
 T442_L0_f147_le: const_Roof_Style_Flat + 999999 t442_leaf0 <= 1e+06
 T442_L0_f120_le: const_Condition_1_PosA + 999999 t442_leaf0 <= 1e+06
 T442_L1_f147_le: const_Roof_Style_Flat + 999999 t442_leaf1 <= 1e+06
 T442_R1_f120_ge: const_Condition_1_PosA - 1.000001e+06 t442_leaf1
   >= -1e+06
 T442_R2_f147_ge: const_Roof_Style_Flat - 1.000001e+06 t442_leaf2
   >= -1e+06
 TREE_443_ONEHOT: t443_leaf0 = 1
 TREE_444_ONEHOT: t444_leaf0 + t444_leaf1 + t444_leaf2 + t444_leaf3
   + t444_leaf4 = 1
 T444_L0_f30_le: x_Fireplace_Qu + 3 t444_leaf0 <= 4
 T444_L0_f18_le: const_2nd_Flr_SF + 999336 t444_leaf0 <= 1e+06
 T444_L0_f5_le: const_Year_Built + 998064 t444_leaf0 <= 1e+06
 T444_L1_f30_le: x_Fireplace_Qu + 3 t444_leaf1 <= 4
 T444_L1_f18_le: const_2nd_Flr_SF + 999336 t444_leaf1 <= 1e+06
 T444_R1_f5_ge: const_Year_Built - 1.001936e+06 t444_leaf1 >= -1e+06
 T444_L2_f30_le: x_Fireplace_Qu + 3 t444_leaf2 <= 4
 T444_R2_f18_ge: const_2nd_Flr_SF - 1.000664e+06 t444_leaf2 >= -1e+06
 T444_L2_f279_le: const_Mo_Sold_9 + 999999 t444_leaf2 <= 1e+06
 T444_L3_f30_le: x_Fireplace_Qu + 3 t444_leaf3 <= 4
 T444_R3_f18_ge: const_2nd_Flr_SF - 1.000664e+06 t444_leaf3 >= -1e+06
 T444_R3_f279_ge: const_Mo_Sold_9 - 1.000001e+06 t444_leaf3 >= -1e+06
 T444_R4_f30_ge: x_Fireplace_Qu - 2 t444_leaf4 >= -1
 TREE_445_ONEHOT: t445_leaf0 + t445_leaf1 + t445_leaf2 = 1
 T445_L0_f290_le: const_Sale_Type_Oth + 999999 t445_leaf0 <= 1e+06
 T445_L0_f14_le: x_Bsmt_Unf_SF + 98743 t445_leaf0 <= 100000
 T445_L1_f290_le: const_Sale_Type_Oth + 999999 t445_leaf1 <= 1e+06
 T445_R1_f14_ge: x_Bsmt_Unf_SF - 1257 t445_leaf1 >= 0
 T445_R2_f290_ge: const_Sale_Type_Oth - 1.000001e+06 t445_leaf2 >= -1e+06
 TREE_446_ONEHOT: t446_leaf0 + t446_leaf1 + t446_leaf2 + t446_leaf3 = 1
 T446_L0_f20_le: x_Gr_Liv_Area + 9013 t446_leaf0 <= 10000
 T446_L0_f6_le: const_Year_Remod_Add + 998047 t446_leaf0 <= 1e+06
 T446_L1_f20_le: x_Gr_Liv_Area + 9013 t446_leaf1 <= 10000
 T446_R1_f6_ge: const_Year_Remod_Add - 1.001953e+06 t446_leaf1 >= -1e+06
 T446_L1_f5_le: const_Year_Built + 998064 t446_leaf1 <= 1e+06
 T446_L2_f20_le: x_Gr_Liv_Area + 9013 t446_leaf2 <= 10000
 T446_R2_f6_ge: const_Year_Remod_Add - 1.001953e+06 t446_leaf2 >= -1e+06
 T446_R2_f5_ge: const_Year_Built - 1.001936e+06 t446_leaf2 >= -1e+06
 T446_R3_f20_ge: x_Gr_Liv_Area - 987 t446_leaf3 >= 0
 TREE_447_ONEHOT: t447_leaf0 + t447_leaf1 + t447_leaf2 = 1
 T447_L0_f5_le: const_Year_Built + 997993 t447_leaf0 <= 1e+06
 T447_R1_f5_ge: const_Year_Built - 1.002007e+06 t447_leaf1 >= -1e+06
 T447_L1_f0_le: const_Lot_Frontage + 999887 t447_leaf1 <= 1e+06
 T447_R2_f5_ge: const_Year_Built - 1.002007e+06 t447_leaf2 >= -1e+06
 T447_R2_f0_ge: const_Lot_Frontage - 1.000113e+06 t447_leaf2 >= -1e+06
 TREE_448_ONEHOT: t448_leaf0 + t448_leaf1 + t448_leaf2 = 1
 T448_L0_f17_le: x_1st_Flr_SF + 2485 t448_leaf0 <= 5000
 T448_L0_f1_le: const_Lot_Area + 980478 t448_leaf0 <= 1e+06
 T448_L1_f17_le: x_1st_Flr_SF + 2485 t448_leaf1 <= 5000
 T448_R1_f1_ge: const_Lot_Area - 1.019522e+06 t448_leaf1 >= -1e+06
 T448_R2_f17_ge: x_1st_Flr_SF - 2515 t448_leaf2 >= 0
 TREE_449_ONEHOT: t449_leaf0 + t449_leaf1 + t449_leaf2 + t449_leaf3
   + t449_leaf4 + t449_leaf5 = 1
 T449_L0_f33_le: x_Garage_Area + 99129 t449_leaf0 <= 100000
 T449_L0_f148_le: const_Roof_Style_Gable + 999999 t449_leaf0 <= 1e+06
 T449_L0_f44_le: const_Yr_Sold + 997992 t449_leaf0 <= 1e+06
 T449_L1_f33_le: x_Garage_Area + 99129 t449_leaf1 <= 100000
 T449_L1_f148_le: const_Roof_Style_Gable + 999999 t449_leaf1 <= 1e+06
 T449_R1_f44_ge: const_Yr_Sold - 1.002008e+06 t449_leaf1 >= -1e+06
 T449_L2_f33_le: x_Garage_Area + 99129 t449_leaf2 <= 100000
 T449_R2_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t449_leaf2
   >= -1e+06
 T449_L2_f78_le: const_Land_Contour_HLS + 999999 t449_leaf2 <= 1e+06
 T449_L3_f33_le: x_Garage_Area + 99129 t449_leaf3 <= 100000
 T449_R3_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t449_leaf3
   >= -1e+06
 T449_R3_f78_ge: const_Land_Contour_HLS - 1.000001e+06 t449_leaf3
   >= -1e+06
 T449_R4_f33_ge: x_Garage_Area - 871 t449_leaf4 >= 0
 T449_L4_f7_le: x_Mas_Vnr_Area + 1276 t449_leaf4 <= 2000
 T449_R5_f33_ge: x_Garage_Area - 871 t449_leaf5 >= 0
 T449_R5_f7_ge: x_Mas_Vnr_Area - 724 t449_leaf5 >= 0
 TREE_450_ONEHOT: t450_leaf0 = 1
 TREE_451_ONEHOT: t451_leaf0 + t451_leaf1 + t451_leaf2 + t451_leaf3
   + t451_leaf4 + t451_leaf5 = 1
 T451_L0_f272_le: const_Mo_Sold_2 + 999999 t451_leaf0 <= 1e+06
 T451_L0_f250_le: const_Garage_Type_CarPort + 999999 t451_leaf0 <= 1e+06
 T451_L0_f14_le: x_Bsmt_Unf_SF + 99887 t451_leaf0 <= 100000
 T451_L1_f272_le: const_Mo_Sold_2 + 999999 t451_leaf1 <= 1e+06
 T451_L1_f250_le: const_Garage_Type_CarPort + 999999 t451_leaf1 <= 1e+06
 T451_R1_f14_ge: x_Bsmt_Unf_SF - 113 t451_leaf1 >= 0
 T451_L2_f272_le: const_Mo_Sold_2 + 999999 t451_leaf2 <= 1e+06
 T451_R2_f250_ge: const_Garage_Type_CarPort - 1.000001e+06 t451_leaf2
   >= -1e+06
 T451_R3_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t451_leaf3 >= -1e+06
 T451_L3_f12_le: x_BsmtFin_SF_1 + 999781 t451_leaf3 <= 1e+06
 T451_L3_f18_le: const_2nd_Flr_SF + 999240 t451_leaf3 <= 1e+06
 T451_R4_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t451_leaf4 >= -1e+06
 T451_L4_f12_le: x_BsmtFin_SF_1 + 999781 t451_leaf4 <= 1e+06
 T451_R4_f18_ge: const_2nd_Flr_SF - 1.00076e+06 t451_leaf4 >= -1e+06
 T451_R5_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t451_leaf5 >= -1e+06
 T451_R5_f12_ge: x_BsmtFin_SF_1 - 219 t451_leaf5 >= 0
 TREE_452_ONEHOT: t452_leaf0 + t452_leaf1 + t452_leaf2 + t452_leaf3
   + t452_leaf4 = 1
 T452_L0_f177_le: x_ex2_is_AsbShng <= 1
 T452_L0_f86_le: const_Land_Slope_Gtl + 999999 t452_leaf0 <= 1e+06
 T452_L1_f177_le: x_ex2_is_AsbShng <= 1
 T452_R1_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t452_leaf1 >= -1e+06
 T452_L1_f5_le: const_Year_Built + 998089 t452_leaf1 <= 1e+06
 T452_L2_f177_le: x_ex2_is_AsbShng <= 1
 T452_R2_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t452_leaf2 >= -1e+06
 T452_R2_f5_ge: const_Year_Built - 1.001911e+06 t452_leaf2 >= -1e+06
 T452_R3_f177_ge: x_ex2_is_AsbShng - t452_leaf3 >= 0
 T452_L3_f12_le: x_BsmtFin_SF_1 + 999824 t452_leaf3 <= 1e+06
 T452_R4_f177_ge: x_ex2_is_AsbShng - t452_leaf4 >= 0
 T452_R4_f12_ge: x_BsmtFin_SF_1 - 176 t452_leaf4 >= 0
 TREE_453_ONEHOT: t453_leaf0 + t453_leaf1 + t453_leaf2 + t453_leaf3
   + t453_leaf4 + t453_leaf5 = 1
 T453_L0_f272_le: const_Mo_Sold_2 + 999999 t453_leaf0 <= 1e+06
 T453_L0_f1_le: const_Lot_Area + 994000 t453_leaf0 <= 1e+06
 T453_L0_f14_le: x_Bsmt_Unf_SF + 99405 t453_leaf0 <= 100000
 T453_L1_f272_le: const_Mo_Sold_2 + 999999 t453_leaf1 <= 1e+06
 T453_L1_f1_le: const_Lot_Area + 994000 t453_leaf1 <= 1e+06
 T453_R1_f14_ge: x_Bsmt_Unf_SF - 595 t453_leaf1 >= 0
 T453_L2_f272_le: const_Mo_Sold_2 + 999999 t453_leaf2 <= 1e+06
 T453_R2_f1_ge: const_Lot_Area - 1.006e+06 t453_leaf2 >= -1e+06
 T453_L2_f3_le: Overall_Qual_calc + 999995 t453_leaf2 <= 1e+06
 T453_L3_f272_le: const_Mo_Sold_2 + 999999 t453_leaf3 <= 1e+06
 T453_R3_f1_ge: const_Lot_Area - 1.006e+06 t453_leaf3 >= -1e+06
 T453_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t453_leaf3 >= -1e+06
 T453_R4_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t453_leaf4 >= -1e+06
 T453_L4_f297_le: const_Sale_Condition_Normal + 999999 t453_leaf4 <= 1e+06
 T453_R5_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t453_leaf5 >= -1e+06
 T453_R5_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t453_leaf5
   >= -1e+06
 TREE_454_ONEHOT: t454_leaf0 + t454_leaf1 = 1
 T454_L0_f194_le: x_mvt_is_BrkCmn <= 1
 T454_R1_f194_ge: x_mvt_is_BrkCmn - t454_leaf1 >= 0
 TREE_455_ONEHOT: t455_leaf0 = 1
 TREE_456_ONEHOT: t456_leaf0 = 1
 TREE_457_ONEHOT: t457_leaf0 + t457_leaf1 + t457_leaf2 = 1
 T457_L0_f296_le: const_Sale_Condition_Family + 999999 t457_leaf0 <= 1e+06
 T457_R1_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t457_leaf1
   >= -1e+06
 T457_L1_f81_le: const_Lot_Config_Corner + 999999 t457_leaf1 <= 1e+06
 T457_R2_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t457_leaf2
   >= -1e+06
 T457_R2_f81_ge: const_Lot_Config_Corner - 1.000001e+06 t457_leaf2
   >= -1e+06
 TREE_458_ONEHOT: t458_leaf0 + t458_leaf1 + t458_leaf2 + t458_leaf3 = 1
 T458_L0_f1_le: const_Lot_Area + 993558 t458_leaf0 <= 1e+06
 T458_R1_f1_ge: const_Lot_Area - 1.006442e+06 t458_leaf1 >= -1e+06
 T458_L1_f38_le: x_Enclosed_Porch + 99746 t458_leaf1 <= 100000
 T458_L1_f38_le: x_Enclosed_Porch + 99764 t458_leaf1 <= 100000
 T458_R2_f1_ge: const_Lot_Area - 1.006442e+06 t458_leaf2 >= -1e+06
 T458_L2_f38_le: x_Enclosed_Porch + 99746 t458_leaf2 <= 100000
 T458_R2_f38_ge: x_Enclosed_Porch - 236 t458_leaf2 >= 0
 T458_R3_f1_ge: const_Lot_Area - 1.006442e+06 t458_leaf3 >= -1e+06
 T458_R3_f38_ge: x_Enclosed_Porch - 254 t458_leaf3 >= 0
 TREE_459_ONEHOT: t459_leaf0 + t459_leaf1 = 1
 T459_L0_f44_le: const_Yr_Sold + 997993 t459_leaf0 <= 1e+06
 T459_R1_f44_ge: const_Yr_Sold - 1.002007e+06 t459_leaf1 >= -1e+06
 TREE_460_ONEHOT: t460_leaf0 = 1
 TREE_461_ONEHOT: t461_leaf0 + t461_leaf1 + t461_leaf2 + t461_leaf3
   + t461_leaf4 = 1
 T461_L0_f3_le: Overall_Qual_calc + 999997 t461_leaf0 <= 1e+06
 T461_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t461_leaf1 >= -1e+06
 T461_L1_f1_le: const_Lot_Area + 983221 t461_leaf1 <= 1e+06
 T461_L1_f38_le: x_Enclosed_Porch + 99769 t461_leaf1 <= 100000
 T461_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t461_leaf2 >= -1e+06
 T461_L2_f1_le: const_Lot_Area + 983221 t461_leaf2 <= 1e+06
 T461_R2_f38_ge: x_Enclosed_Porch - 231 t461_leaf2 >= 0
 T461_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t461_leaf3 >= -1e+06
 T461_R3_f1_ge: const_Lot_Area - 1.016779e+06 t461_leaf3 >= -1e+06
 T461_L3_f267_le: const_Misc_Feature_No_aplica + 999999 t461_leaf3
   <= 1e+06
 T461_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t461_leaf4 >= -1e+06
 T461_R4_f1_ge: const_Lot_Area - 1.016779e+06 t461_leaf4 >= -1e+06
 T461_R4_f267_ge: const_Misc_Feature_No_aplica - 1.000001e+06 t461_leaf4
   >= -1e+06
 TREE_462_ONEHOT: t462_leaf0 + t462_leaf1 = 1
 T462_L0_f0_le: const_Lot_Frontage + 999860 t462_leaf0 <= 1e+06
 T462_R1_f0_ge: const_Lot_Frontage - 1.00014e+06 t462_leaf1 >= -1e+06
 TREE_463_ONEHOT: t463_leaf0 + t463_leaf1 + t463_leaf2 + t463_leaf3
   + t463_leaf4 + t463_leaf5 + t463_leaf6 = 1
 T463_L0_f20_le: x_Gr_Liv_Area + 9106 t463_leaf0 <= 10000
 T463_L0_f1_le: const_Lot_Area + 989818 t463_leaf0 <= 1e+06
 T463_L0_f36_le: x_Wood_Deck_SF + 99978 t463_leaf0 <= 100000
 T463_L1_f20_le: x_Gr_Liv_Area + 9106 t463_leaf1 <= 10000
 T463_L1_f1_le: const_Lot_Area + 989818 t463_leaf1 <= 1e+06
 T463_R1_f36_ge: x_Wood_Deck_SF - 22 t463_leaf1 >= 0
 T463_L2_f20_le: x_Gr_Liv_Area + 9106 t463_leaf2 <= 10000
 T463_R2_f1_ge: const_Lot_Area - 1.010182e+06 t463_leaf2 >= -1e+06
 T463_R3_f20_ge: x_Gr_Liv_Area - 894 t463_leaf3 >= 0
 T463_L3_f1_le: const_Lot_Area + 978630 t463_leaf3 <= 1e+06
 T463_L3_f248_le: const_Garage_Type_Basment + 999999 t463_leaf3 <= 1e+06
 T463_R4_f20_ge: x_Gr_Liv_Area - 894 t463_leaf4 >= 0
 T463_L4_f1_le: const_Lot_Area + 978630 t463_leaf4 <= 1e+06
 T463_R4_f248_ge: const_Garage_Type_Basment - 1.000001e+06 t463_leaf4
   >= -1e+06
 T463_R5_f20_ge: x_Gr_Liv_Area - 894 t463_leaf5 >= 0
 T463_R5_f1_ge: const_Lot_Area - 1.02137e+06 t463_leaf5 >= -1e+06
 T463_L5_f33_le: x_Garage_Area + 99157 t463_leaf5 <= 100000
 T463_R6_f20_ge: x_Gr_Liv_Area - 894 t463_leaf6 >= 0
 T463_R6_f1_ge: const_Lot_Area - 1.02137e+06 t463_leaf6 >= -1e+06
 T463_R6_f33_ge: x_Garage_Area - 843 t463_leaf6 >= 0
 TREE_464_ONEHOT: t464_leaf0 + t464_leaf1 + t464_leaf2 = 1
 T464_L0_f202_le: const_Foundation_PConc + 999999 t464_leaf0 <= 1e+06
 T464_L0_f164_le: x_ex1_is_BrkFace <= 1
 T464_L1_f202_le: const_Foundation_PConc + 999999 t464_leaf1 <= 1e+06
 T464_R1_f164_ge: x_ex1_is_BrkFace - t464_leaf1 >= 0
 T464_R2_f202_ge: const_Foundation_PConc - 1.000001e+06 t464_leaf2
   >= -1e+06
 TREE_465_ONEHOT: t465_leaf0 + t465_leaf1 + t465_leaf2 + t465_leaf3 = 1
 T465_L0_f3_le: Overall_Qual_calc + 999996 t465_leaf0 <= 1e+06
 T465_L0_f17_le: x_1st_Flr_SF + 4187 t465_leaf0 <= 5000
 T465_L1_f3_le: Overall_Qual_calc + 999996 t465_leaf1 <= 1e+06
 T465_R1_f17_ge: x_1st_Flr_SF - 813 t465_leaf1 >= 0
 T465_R2_f3_ge: Overall_Qual_calc - 1.000004e+06 t465_leaf2 >= -1e+06
 T465_L2_f37_le: x_Open_Porch_SF + 99811 t465_leaf2 <= 100000
 T465_R3_f3_ge: Overall_Qual_calc - 1.000004e+06 t465_leaf3 >= -1e+06
 T465_R3_f37_ge: x_Open_Porch_SF - 189 t465_leaf3 >= 0
 TREE_466_ONEHOT: t466_leaf0 + t466_leaf1 + t466_leaf2 + t466_leaf3
   + t466_leaf4 = 1
 T466_L0_f3_le: Overall_Qual_calc + 999995 t466_leaf0 <= 1e+06
 T466_L0_f1_le: const_Lot_Area + 995730 t466_leaf0 <= 1e+06
 T466_L1_f3_le: Overall_Qual_calc + 999995 t466_leaf1 <= 1e+06
 T466_R1_f1_ge: const_Lot_Area - 1.00427e+06 t466_leaf1 >= -1e+06
 T466_L1_f33_le: x_Garage_Area + 99500 t466_leaf1 <= 100000
 T466_L2_f3_le: Overall_Qual_calc + 999995 t466_leaf2 <= 1e+06
 T466_R2_f1_ge: const_Lot_Area - 1.00427e+06 t466_leaf2 >= -1e+06
 T466_R2_f33_ge: x_Garage_Area - 500 t466_leaf2 >= 0
 T466_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t466_leaf3 >= -1e+06
 T466_L3_f218_le: x_b2_is_ALQ <= 1
 T466_R4_f3_ge: Overall_Qual_calc - 1.000005e+06 t466_leaf4 >= -1e+06
 T466_R4_f218_ge: x_b2_is_ALQ - t466_leaf4 >= 0
 TREE_467_ONEHOT: t467_leaf0 = 1
 TREE_468_ONEHOT: t468_leaf0 = 1
 TREE_469_ONEHOT: t469_leaf0 = 1
 TREE_470_ONEHOT: t470_leaf0 + t470_leaf1 + t470_leaf2 = 1
 T470_L0_f8_le: x_Exter_Qual + 2 t470_leaf0 <= 4
 T470_R1_f8_ge: x_Exter_Qual - 3 t470_leaf1 >= -1
 T470_L1_f17_le: x_1st_Flr_SF + 2800 t470_leaf1 <= 5000
 T470_R2_f8_ge: x_Exter_Qual - 3 t470_leaf2 >= -1
 T470_R2_f17_ge: x_1st_Flr_SF - 2200 t470_leaf2 >= 0
 TREE_471_ONEHOT: t471_leaf0 + t471_leaf1 + t471_leaf2 + t471_leaf3 = 1
 T471_L0_f96_le: const_Neighborhood_Edwards + 999999 t471_leaf0 <= 1e+06
 T471_R1_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t471_leaf1
   >= -1e+06
 T471_L1_f17_le: x_1st_Flr_SF + 4014 t471_leaf1 <= 5000
 T471_R2_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t471_leaf2
   >= -1e+06
 T471_R2_f17_ge: x_1st_Flr_SF - 986 t471_leaf2 >= 0
 T471_L2_f17_le: x_1st_Flr_SF + 3482 t471_leaf2 <= 5000
 T471_R3_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t471_leaf3
   >= -1e+06
 T471_R3_f17_ge: x_1st_Flr_SF - 986 t471_leaf3 >= 0
 T471_R3_f17_ge: x_1st_Flr_SF - 1518 t471_leaf3 >= 0
 TREE_472_ONEHOT: t472_leaf0 = 1
 TREE_473_ONEHOT: t473_leaf0 + t473_leaf1 + t473_leaf2 = 1
 T473_L0_f199_le: x_mvt_is_Stone <= 1
 T473_L0_f14_le: x_Bsmt_Unf_SF + 98607 t473_leaf0 <= 100000
 T473_L1_f199_le: x_mvt_is_Stone <= 1
 T473_R1_f14_ge: x_Bsmt_Unf_SF - 1393 t473_leaf1 >= 0
 T473_R2_f199_ge: x_mvt_is_Stone - t473_leaf2 >= 0
 TREE_474_ONEHOT: t474_leaf0 + t474_leaf1 + t474_leaf2 = 1
 T474_L0_f15_le: const_Total_Bsmt_SF + 998176 t474_leaf0 <= 1e+06
 T474_L0_f3_le: Overall_Qual_calc + 999996 t474_leaf0 <= 1e+06
 T474_L1_f15_le: const_Total_Bsmt_SF + 998176 t474_leaf1 <= 1e+06
 T474_R1_f3_ge: Overall_Qual_calc - 1.000004e+06 t474_leaf1 >= -1e+06
 T474_R2_f15_ge: const_Total_Bsmt_SF - 1.001824e+06 t474_leaf2 >= -1e+06
 TREE_475_ONEHOT: t475_leaf0 + t475_leaf1 + t475_leaf2 = 1
 T475_L0_f15_le: const_Total_Bsmt_SF + 999519 t475_leaf0 <= 1e+06
 T475_L0_f15_le: const_Total_Bsmt_SF + 999755 t475_leaf0 <= 1e+06
 T475_L1_f15_le: const_Total_Bsmt_SF + 999519 t475_leaf1 <= 1e+06
 T475_R1_f15_ge: const_Total_Bsmt_SF - 1.000245e+06 t475_leaf1 >= -1e+06
 T475_R2_f15_ge: const_Total_Bsmt_SF - 1.000481e+06 t475_leaf2 >= -1e+06
 TREE_476_ONEHOT: t476_leaf0 + t476_leaf1 + t476_leaf2 + t476_leaf3 = 1
 T476_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t476_leaf0 <= 1e+06
 T476_L0_f38_le: x_Enclosed_Porch + 99909 t476_leaf0 <= 100000
 T476_L0_f4_le: const_Overall_Cond + 999994 t476_leaf0 <= 1e+06
 T476_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t476_leaf1 <= 1e+06
 T476_L1_f38_le: x_Enclosed_Porch + 99909 t476_leaf1 <= 100000
 T476_R1_f4_ge: const_Overall_Cond - 1.000006e+06 t476_leaf1 >= -1e+06
 T476_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t476_leaf2 <= 1e+06
 T476_R2_f38_ge: x_Enclosed_Porch - 91 t476_leaf2 >= 0
 T476_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t476_leaf3
   >= -1e+06
 TREE_477_ONEHOT: t477_leaf0 + t477_leaf1 + t477_leaf2 + t477_leaf3 = 1
 T477_L0_f14_le: x_Bsmt_Unf_SF + 99424 t477_leaf0 <= 100000
 T477_L0_f237_le: x_elect_is_SBrkr <= 1
 T477_L1_f14_le: x_Bsmt_Unf_SF + 99424 t477_leaf1 <= 100000
 T477_R1_f237_ge: x_elect_is_SBrkr - t477_leaf1 >= 0
 T477_R2_f14_ge: x_Bsmt_Unf_SF - 576 t477_leaf2 >= 0
 T477_L2_f18_le: const_2nd_Flr_SF + 999419 t477_leaf2 <= 1e+06
 T477_R3_f14_ge: x_Bsmt_Unf_SF - 576 t477_leaf3 >= 0
 T477_R3_f18_ge: const_2nd_Flr_SF - 1.000581e+06 t477_leaf3 >= -1e+06
 TREE_478_ONEHOT: t478_leaf0 + t478_leaf1 + t478_leaf2 + t478_leaf3 = 1
 T478_L0_f38_le: x_Enclosed_Porch + 99846 t478_leaf0 <= 100000
 T478_R1_f38_ge: x_Enclosed_Porch - 154 t478_leaf1 >= 0
 T478_L1_f18_le: const_2nd_Flr_SF + 999692 t478_leaf1 <= 1e+06
 T478_L1_f14_le: x_Bsmt_Unf_SF + 99972 t478_leaf1 <= 100000
 T478_R2_f38_ge: x_Enclosed_Porch - 154 t478_leaf2 >= 0
 T478_L2_f18_le: const_2nd_Flr_SF + 999692 t478_leaf2 <= 1e+06
 T478_R2_f14_ge: x_Bsmt_Unf_SF - 28 t478_leaf2 >= 0
 T478_R3_f38_ge: x_Enclosed_Porch - 154 t478_leaf3 >= 0
 T478_R3_f18_ge: const_2nd_Flr_SF - 1.000308e+06 t478_leaf3 >= -1e+06
 TREE_479_ONEHOT: t479_leaf0 + t479_leaf1 + t479_leaf2 + t479_leaf3
   + t479_leaf4 + t479_leaf5 = 1
 T479_L0_f96_le: const_Neighborhood_Edwards + 999999 t479_leaf0 <= 1e+06
 T479_L0_f177_le: x_ex2_is_AsbShng <= 1
 T479_L0_f20_le: x_Gr_Liv_Area + 9106 t479_leaf0 <= 10000
 T479_L1_f96_le: const_Neighborhood_Edwards + 999999 t479_leaf1 <= 1e+06
 T479_L1_f177_le: x_ex2_is_AsbShng <= 1
 T479_R1_f20_ge: x_Gr_Liv_Area - 894 t479_leaf1 >= 0
 T479_L2_f96_le: const_Neighborhood_Edwards + 999999 t479_leaf2 <= 1e+06
 T479_R2_f177_ge: x_ex2_is_AsbShng - t479_leaf2 >= 0
 T479_R3_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t479_leaf3
   >= -1e+06
 T479_L3_f33_le: x_Garage_Area + 99390 t479_leaf3 <= 100000
 T479_L3_f73_le: const_Lot_Shape_IR1 + 999999 t479_leaf3 <= 1e+06
 T479_R4_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t479_leaf4
   >= -1e+06
 T479_L4_f33_le: x_Garage_Area + 99390 t479_leaf4 <= 100000
 T479_R4_f73_ge: const_Lot_Shape_IR1 - 1.000001e+06 t479_leaf4 >= -1e+06
 T479_R5_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t479_leaf5
   >= -1e+06
 T479_R5_f33_ge: x_Garage_Area - 610 t479_leaf5 >= 0
 TREE_480_ONEHOT: t480_leaf0 + t480_leaf1 + t480_leaf2 + t480_leaf3
   + t480_leaf4 = 1
 T480_L0_f114_le: const_Neighborhood_StoneBr + 999999 t480_leaf0 <= 1e+06
 T480_L0_f37_le: x_Open_Porch_SF + 99832 t480_leaf0 <= 100000
 T480_L0_f14_le: x_Bsmt_Unf_SF + 98627 t480_leaf0 <= 100000
 T480_L1_f114_le: const_Neighborhood_StoneBr + 999999 t480_leaf1 <= 1e+06
 T480_L1_f37_le: x_Open_Porch_SF + 99832 t480_leaf1 <= 100000
 T480_R1_f14_ge: x_Bsmt_Unf_SF - 1373 t480_leaf1 >= 0
 T480_L2_f114_le: const_Neighborhood_StoneBr + 999999 t480_leaf2 <= 1e+06
 T480_R2_f37_ge: x_Open_Porch_SF - 168 t480_leaf2 >= 0
 T480_L2_f28_le: x_TotRms_AbvGrd + 6 t480_leaf2 <= 15
 T480_L3_f114_le: const_Neighborhood_StoneBr + 999999 t480_leaf3 <= 1e+06
 T480_R3_f37_ge: x_Open_Porch_SF - 168 t480_leaf3 >= 0
 T480_R3_f28_ge: x_TotRms_AbvGrd - 9 t480_leaf3 >= 0
 T480_R4_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t480_leaf4
   >= -1e+06
 TREE_481_ONEHOT: t481_leaf0 + t481_leaf1 = 1
 T481_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t481_leaf0 <= 1e+06
 T481_R1_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t481_leaf1
   >= -1e+06
 TREE_482_ONEHOT: t482_leaf0 = 1
 TREE_483_ONEHOT: t483_leaf0 = 1
 TREE_484_ONEHOT: t484_leaf0 + t484_leaf1 + t484_leaf2 + t484_leaf3 = 1
 T484_L0_f15_le: const_Total_Bsmt_SF + 997508 t484_leaf0 <= 1e+06
 T484_L0_f250_le: const_Garage_Type_CarPort + 999999 t484_leaf0 <= 1e+06
 T484_L0_f21_le: const_Bsmt_Full_Bath + 999998 t484_leaf0 <= 1e+06
 T484_L1_f15_le: const_Total_Bsmt_SF + 997508 t484_leaf1 <= 1e+06
 T484_L1_f250_le: const_Garage_Type_CarPort + 999999 t484_leaf1 <= 1e+06
 T484_R1_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t484_leaf1 >= -1e+06
 T484_L2_f15_le: const_Total_Bsmt_SF + 997508 t484_leaf2 <= 1e+06
 T484_R2_f250_ge: const_Garage_Type_CarPort - 1.000001e+06 t484_leaf2
   >= -1e+06
 T484_R3_f15_ge: const_Total_Bsmt_SF - 1.002492e+06 t484_leaf3 >= -1e+06
 TREE_485_ONEHOT: t485_leaf0 = 1
 TREE_486_ONEHOT: t486_leaf0 = 1
 TREE_487_ONEHOT: t487_leaf0 = 1
 TREE_488_ONEHOT: t488_leaf0 + t488_leaf1 + t488_leaf2 + t488_leaf3 = 1
 T488_L0_f33_le: x_Garage_Area + 99692 t488_leaf0 <= 100000
 T488_L0_f297_le: const_Sale_Condition_Normal + 999999 t488_leaf0 <= 1e+06
 T488_L1_f33_le: x_Garage_Area + 99692 t488_leaf1 <= 100000
 T488_R1_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t488_leaf1
   >= -1e+06
 T488_L1_f3_le: Overall_Qual_calc + 999997 t488_leaf1 <= 1e+06
 T488_L2_f33_le: x_Garage_Area + 99692 t488_leaf2 <= 100000
 T488_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t488_leaf2
   >= -1e+06
 T488_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t488_leaf2 >= -1e+06
 T488_R3_f33_ge: x_Garage_Area - 308 t488_leaf3 >= 0
 TREE_489_ONEHOT: t489_leaf0 = 1
 TREE_490_ONEHOT: t490_leaf0 + t490_leaf1 + t490_leaf2 = 1
 T490_L0_f37_le: x_Open_Porch_SF + 99836 t490_leaf0 <= 100000
 T490_L0_f258_le: x_paved_drive_is_P <= 1
 T490_L1_f37_le: x_Open_Porch_SF + 99836 t490_leaf1 <= 100000
 T490_R1_f258_ge: x_paved_drive_is_P - t490_leaf1 >= 0
 T490_R2_f37_ge: x_Open_Porch_SF - 164 t490_leaf2 >= 0
 TREE_491_ONEHOT: t491_leaf0 + t491_leaf1 + t491_leaf2 + t491_leaf3 = 1
 T491_L0_f14_le: x_Bsmt_Unf_SF + 99524 t491_leaf0 <= 100000
 T491_L0_f25_le: x_Bedroom_AbvGr + 3 t491_leaf0 <= 6
 T491_L0_f1_le: const_Lot_Area + 994564 t491_leaf0 <= 1e+06
 T491_L1_f14_le: x_Bsmt_Unf_SF + 99524 t491_leaf1 <= 100000
 T491_L1_f25_le: x_Bedroom_AbvGr + 3 t491_leaf1 <= 6
 T491_R1_f1_ge: const_Lot_Area - 1.005436e+06 t491_leaf1 >= -1e+06
 T491_L2_f14_le: x_Bsmt_Unf_SF + 99524 t491_leaf2 <= 100000
 T491_R2_f25_ge: x_Bedroom_AbvGr - 3 t491_leaf2 >= 0
 T491_R3_f14_ge: x_Bsmt_Unf_SF - 476 t491_leaf3 >= 0
 TREE_492_ONEHOT: t492_leaf0 + t492_leaf1 + t492_leaf2 = 1
 T492_L0_f93_le: const_Neighborhood_ClearCr + 999999 t492_leaf0 <= 1e+06
 T492_L0_f38_le: x_Enclosed_Porch + 99735 t492_leaf0 <= 100000
 T492_L1_f93_le: const_Neighborhood_ClearCr + 999999 t492_leaf1 <= 1e+06
 T492_R1_f38_ge: x_Enclosed_Porch - 265 t492_leaf1 >= 0
 T492_R2_f93_ge: const_Neighborhood_ClearCr - 1.000001e+06 t492_leaf2
   >= -1e+06
 TREE_493_ONEHOT: t493_leaf0 + t493_leaf1 + t493_leaf2 + t493_leaf3
   + t493_leaf4 = 1
 T493_L0_f26_le: x_Kitchen_AbvGr + t493_leaf0 <= 3
 T493_L0_f33_le: x_Garage_Area + 99748 t493_leaf0 <= 100000
 T493_L0_f76_le: const_Lot_Shape_Reg + 999999 t493_leaf0 <= 1e+06
 T493_L1_f26_le: x_Kitchen_AbvGr + t493_leaf1 <= 3
 T493_L1_f33_le: x_Garage_Area + 99748 t493_leaf1 <= 100000
 T493_R1_f76_ge: const_Lot_Shape_Reg - 1.000001e+06 t493_leaf1 >= -1e+06
 T493_L2_f26_le: x_Kitchen_AbvGr + t493_leaf2 <= 3
 T493_R2_f33_ge: x_Garage_Area - 252 t493_leaf2 >= 0
 T493_L2_f95_le: const_Neighborhood_Crawfor + 999999 t493_leaf2 <= 1e+06
 T493_L3_f26_le: x_Kitchen_AbvGr + t493_leaf3 <= 3
 T493_R3_f33_ge: x_Garage_Area - 252 t493_leaf3 >= 0
 T493_R3_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t493_leaf3
   >= -1e+06
 T493_R4_f26_ge: x_Kitchen_AbvGr - 2 t493_leaf4 >= 0
 TREE_494_ONEHOT: t494_leaf0 + t494_leaf1 + t494_leaf2 = 1
 T494_L0_f150_le: const_Roof_Style_Hip + 999999 t494_leaf0 <= 1e+06
 T494_R1_f150_ge: const_Roof_Style_Hip - 1.000001e+06 t494_leaf1 >= -1e+06
 T494_L1_f77_le: const_Land_Contour_Bnk + 999999 t494_leaf1 <= 1e+06
 T494_R2_f150_ge: const_Roof_Style_Hip - 1.000001e+06 t494_leaf2 >= -1e+06
 T494_R2_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t494_leaf2
   >= -1e+06
 TREE_495_ONEHOT: t495_leaf0 + t495_leaf1 = 1
 T495_L0_f3_le: Overall_Qual_calc + 999997 t495_leaf0 <= 1e+06
 T495_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t495_leaf1 >= -1e+06
 TREE_496_ONEHOT: t496_leaf0 + t496_leaf1 = 1
 T496_L0_f3_le: Overall_Qual_calc + 999997 t496_leaf0 <= 1e+06
 T496_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t496_leaf1 >= -1e+06
 TREE_497_ONEHOT: t497_leaf0 + t497_leaf1 + t497_leaf2 + t497_leaf3 = 1
 T497_L0_f0_le: const_Lot_Frontage + 999883 t497_leaf0 <= 1e+06
 T497_L0_f11_le: x_Bsmt_Cond + 2 t497_leaf0 <= 4
 T497_L0_f38_le: x_Enclosed_Porch + 99895 t497_leaf0 <= 100000
 T497_L1_f0_le: const_Lot_Frontage + 999883 t497_leaf1 <= 1e+06
 T497_L1_f11_le: x_Bsmt_Cond + 2 t497_leaf1 <= 4
 T497_R1_f38_ge: x_Enclosed_Porch - 105 t497_leaf1 >= 0
 T497_L2_f0_le: const_Lot_Frontage + 999883 t497_leaf2 <= 1e+06
 T497_R2_f11_ge: x_Bsmt_Cond - 3 t497_leaf2 >= -1
 T497_R3_f0_ge: const_Lot_Frontage - 1.000117e+06 t497_leaf3 >= -1e+06
 TREE_498_ONEHOT: t498_leaf0 + t498_leaf1 + t498_leaf2 + t498_leaf3
   + t498_leaf4 = 1
 T498_L0_f5_le: const_Year_Built + 997995 t498_leaf0 <= 1e+06
 T498_L0_f3_le: Overall_Qual_calc + 999993 t498_leaf0 <= 1e+06
 T498_L0_f28_le: x_TotRms_AbvGrd + 7 t498_leaf0 <= 15
 T498_L1_f5_le: const_Year_Built + 997995 t498_leaf1 <= 1e+06
 T498_L1_f3_le: Overall_Qual_calc + 999993 t498_leaf1 <= 1e+06
 T498_R1_f28_ge: x_TotRms_AbvGrd - 8 t498_leaf1 >= 0
 T498_L2_f5_le: const_Year_Built + 997995 t498_leaf2 <= 1e+06
 T498_R2_f3_ge: Overall_Qual_calc - 1.000007e+06 t498_leaf2 >= -1e+06
 T498_L2_f1_le: const_Lot_Area + 974714 t498_leaf2 <= 1e+06
 T498_L3_f5_le: const_Year_Built + 997995 t498_leaf3 <= 1e+06
 T498_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t498_leaf3 >= -1e+06
 T498_R3_f1_ge: const_Lot_Area - 1.025286e+06 t498_leaf3 >= -1e+06
 T498_R4_f5_ge: const_Year_Built - 1.002005e+06 t498_leaf4 >= -1e+06
 TREE_499_ONEHOT: t499_leaf0 + t499_leaf1 = 1
 T499_L0_f92_le: const_Neighborhood_BrkSide + 999999 t499_leaf0 <= 1e+06
 T499_R1_f92_ge: const_Neighborhood_BrkSide - 1.000001e+06 t499_leaf1
   >= -1e+06
 TREE_500_ONEHOT: t500_leaf0 + t500_leaf1 + t500_leaf2 + t500_leaf3 = 1
 T500_L0_f33_le: x_Garage_Area + 99112 t500_leaf0 <= 100000
 T500_L0_f79_le: const_Land_Contour_Low + 999999 t500_leaf0 <= 1e+06
 T500_L1_f33_le: x_Garage_Area + 99112 t500_leaf1 <= 100000
 T500_R1_f79_ge: const_Land_Contour_Low - 1.000001e+06 t500_leaf1
   >= -1e+06
 T500_L1_f170_le: x_ex1_is_Plywood <= 1
 T500_L2_f33_le: x_Garage_Area + 99112 t500_leaf2 <= 100000
 T500_R2_f79_ge: const_Land_Contour_Low - 1.000001e+06 t500_leaf2
   >= -1e+06
 T500_R2_f170_ge: x_ex1_is_Plywood - t500_leaf2 >= 0
 T500_R3_f33_ge: x_Garage_Area - 888 t500_leaf3 >= 0
 TREE_501_ONEHOT: t501_leaf0 + t501_leaf1 + t501_leaf2 + t501_leaf3
   + t501_leaf4 + t501_leaf5 + t501_leaf6 = 1
 T501_L0_f20_le: x_Gr_Liv_Area + 9124 t501_leaf0 <= 10000
 T501_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t501_leaf0
   <= 1e+06
 T501_L0_f76_le: const_Lot_Shape_Reg + 999999 t501_leaf0 <= 1e+06
 T501_L1_f20_le: x_Gr_Liv_Area + 9124 t501_leaf1 <= 10000
 T501_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t501_leaf1
   <= 1e+06
 T501_R1_f76_ge: const_Lot_Shape_Reg - 1.000001e+06 t501_leaf1 >= -1e+06
 T501_L2_f20_le: x_Gr_Liv_Area + 9124 t501_leaf2 <= 10000
 T501_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t501_leaf2
   >= -1e+06
 T501_R3_f20_ge: x_Gr_Liv_Area - 876 t501_leaf3 >= 0
 T501_L3_f33_le: x_Garage_Area + 99356 t501_leaf3 <= 100000
 T501_L3_f250_le: const_Garage_Type_CarPort + 999999 t501_leaf3 <= 1e+06
 T501_R4_f20_ge: x_Gr_Liv_Area - 876 t501_leaf4 >= 0
 T501_L4_f33_le: x_Garage_Area + 99356 t501_leaf4 <= 100000
 T501_R4_f250_ge: const_Garage_Type_CarPort - 1.000001e+06 t501_leaf4
   >= -1e+06
 T501_R5_f20_ge: x_Gr_Liv_Area - 876 t501_leaf5 >= 0
 T501_R5_f33_ge: x_Garage_Area - 644 t501_leaf5 >= 0
 T501_L5_f20_le: x_Gr_Liv_Area + 8221 t501_leaf5 <= 10000
 T501_R6_f20_ge: x_Gr_Liv_Area - 876 t501_leaf6 >= 0
 T501_R6_f33_ge: x_Garage_Area - 644 t501_leaf6 >= 0
 T501_R6_f20_ge: x_Gr_Liv_Area - 1779 t501_leaf6 >= 0
 TREE_502_ONEHOT: t502_leaf0 + t502_leaf1 = 1
 T502_L0_f3_le: Overall_Qual_calc + 999997 t502_leaf0 <= 1e+06
 T502_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t502_leaf1 >= -1e+06
 TREE_503_ONEHOT: t503_leaf0 = 1
 TREE_504_ONEHOT: t504_leaf0 + t504_leaf1 + t504_leaf2 + t504_leaf3 = 1
 T504_L0_f10_le: const_Bsmt_Qual + 1e+06 t504_leaf0 <= 1e+06
 T504_L0_f0_le: const_Lot_Frontage + 999934 t504_leaf0 <= 1e+06
 T504_L1_f10_le: const_Bsmt_Qual + 1e+06 t504_leaf1 <= 1e+06
 T504_R1_f0_ge: const_Lot_Frontage - 1.000066e+06 t504_leaf1 >= -1e+06
 T504_L1_f276_le: const_Mo_Sold_6 + 999999 t504_leaf1 <= 1e+06
 T504_L2_f10_le: const_Bsmt_Qual + 1e+06 t504_leaf2 <= 1e+06
 T504_R2_f0_ge: const_Lot_Frontage - 1.000066e+06 t504_leaf2 >= -1e+06
 T504_R2_f276_ge: const_Mo_Sold_6 - 1.000001e+06 t504_leaf2 >= -1e+06
 T504_R3_f10_ge: const_Bsmt_Qual - 1e+06 t504_leaf3 >= -1e+06
 TREE_505_ONEHOT: t505_leaf0 + t505_leaf1 = 1
 T505_L0_f37_le: x_Open_Porch_SF + 99907 t505_leaf0 <= 100000
 T505_R1_f37_ge: x_Open_Porch_SF - 93 t505_leaf1 >= 0
 TREE_506_ONEHOT: t506_leaf0 + t506_leaf1 + t506_leaf2 + t506_leaf3
   + t506_leaf4 = 1
 T506_L0_f1_le: const_Lot_Area + 983221 t506_leaf0 <= 1e+06
 T506_L0_f6_le: const_Year_Remod_Add + 997992 t506_leaf0 <= 1e+06
 T506_L0_f20_le: x_Gr_Liv_Area + 7367 t506_leaf0 <= 10000
 T506_L1_f1_le: const_Lot_Area + 983221 t506_leaf1 <= 1e+06
 T506_L1_f6_le: const_Year_Remod_Add + 997992 t506_leaf1 <= 1e+06
 T506_R1_f20_ge: x_Gr_Liv_Area - 2633 t506_leaf1 >= 0
 T506_L2_f1_le: const_Lot_Area + 983221 t506_leaf2 <= 1e+06
 T506_R2_f6_ge: const_Year_Remod_Add - 1.002008e+06 t506_leaf2 >= -1e+06
 T506_R3_f1_ge: const_Lot_Area - 1.016779e+06 t506_leaf3 >= -1e+06
 T506_L3_f17_le: x_1st_Flr_SF + 3598 t506_leaf3 <= 5000
 T506_R4_f1_ge: const_Lot_Area - 1.016779e+06 t506_leaf4 >= -1e+06
 T506_R4_f17_ge: x_1st_Flr_SF - 1402 t506_leaf4 >= 0
 TREE_507_ONEHOT: t507_leaf0 + t507_leaf1 = 1
 T507_L0_f13_le: x_BsmtFin_SF_2 + 99940 t507_leaf0 <= 100000
 T507_R1_f13_ge: x_BsmtFin_SF_2 - 60 t507_leaf1 >= 0
 TREE_508_ONEHOT: t508_leaf0 + t508_leaf1 + t508_leaf2 + t508_leaf3
   + t508_leaf4 + t508_leaf5 = 1
 T508_L0_f161_le: x_ex1_is_AsbShng <= 1
 T508_L0_f148_le: const_Roof_Style_Gable + 999999 t508_leaf0 <= 1e+06
 T508_L0_f44_le: const_Yr_Sold + 997993 t508_leaf0 <= 1e+06
 T508_L1_f161_le: x_ex1_is_AsbShng <= 1
 T508_L1_f148_le: const_Roof_Style_Gable + 999999 t508_leaf1 <= 1e+06
 T508_R1_f44_ge: const_Yr_Sold - 1.002007e+06 t508_leaf1 >= -1e+06
 T508_L2_f161_le: x_ex1_is_AsbShng <= 1
 T508_R2_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t508_leaf2
   >= -1e+06
 T508_L2_f273_le: const_Mo_Sold_3 + 999999 t508_leaf2 <= 1e+06
 T508_L3_f161_le: x_ex1_is_AsbShng <= 1
 T508_R3_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t508_leaf3
   >= -1e+06
 T508_R3_f273_ge: const_Mo_Sold_3 - 1.000001e+06 t508_leaf3 >= -1e+06
 T508_R4_f161_ge: x_ex1_is_AsbShng - t508_leaf4 >= 0
 T508_L4_f33_le: x_Garage_Area + 99649 t508_leaf4 <= 100000
 T508_R5_f161_ge: x_ex1_is_AsbShng - t508_leaf5 >= 0
 T508_R5_f33_ge: x_Garage_Area - 351 t508_leaf5 >= 0
 TREE_509_ONEHOT: t509_leaf0 + t509_leaf1 + t509_leaf2 = 1
 T509_L0_f18_le: const_2nd_Flr_SF + 999203 t509_leaf0 <= 1e+06
 T509_L0_f44_le: const_Yr_Sold + 997990 t509_leaf0 <= 1e+06
 T509_L1_f18_le: const_2nd_Flr_SF + 999203 t509_leaf1 <= 1e+06
 T509_R1_f44_ge: const_Yr_Sold - 1.00201e+06 t509_leaf1 >= -1e+06
 T509_R2_f18_ge: const_2nd_Flr_SF - 1.000797e+06 t509_leaf2 >= -1e+06
 TREE_510_ONEHOT: t510_leaf0 = 1
 TREE_511_ONEHOT: t511_leaf0 = 1
 TREE_512_ONEHOT: t512_leaf0 = 1
 TREE_513_ONEHOT: t513_leaf0 + t513_leaf1 + t513_leaf2 + t513_leaf3
   + t513_leaf4 + t513_leaf5 = 1
 T513_L0_f7_le: x_Mas_Vnr_Area + 1212 t513_leaf0 <= 2000
 T513_L0_f4_le: const_Overall_Cond + 999995 t513_leaf0 <= 1e+06
 T513_L0_f30_le: x_Fireplace_Qu + 3 t513_leaf0 <= 4
 T513_L1_f7_le: x_Mas_Vnr_Area + 1212 t513_leaf1 <= 2000
 T513_L1_f4_le: const_Overall_Cond + 999995 t513_leaf1 <= 1e+06
 T513_R1_f30_ge: x_Fireplace_Qu - 2 t513_leaf1 >= -1
 T513_L2_f7_le: x_Mas_Vnr_Area + 1212 t513_leaf2 <= 2000
 T513_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t513_leaf2 >= -1e+06
 T513_L2_f164_le: x_ex1_is_BrkFace <= 1
 T513_L3_f7_le: x_Mas_Vnr_Area + 1212 t513_leaf3 <= 2000
 T513_R3_f4_ge: const_Overall_Cond - 1.000005e+06 t513_leaf3 >= -1e+06
 T513_R3_f164_ge: x_ex1_is_BrkFace - t513_leaf3 >= 0
 T513_R4_f7_ge: x_Mas_Vnr_Area - 788 t513_leaf4 >= 0
 T513_L4_f1_le: const_Lot_Area + 985885 t513_leaf4 <= 1e+06
 T513_R5_f7_ge: x_Mas_Vnr_Area - 788 t513_leaf5 >= 0
 T513_R5_f1_ge: const_Lot_Area - 1.014115e+06 t513_leaf5 >= -1e+06
 TREE_514_ONEHOT: t514_leaf0 = 1
 TREE_515_ONEHOT: t515_leaf0 + t515_leaf1 + t515_leaf2 + t515_leaf3 = 1
 T515_L0_f214_le: x_b1_is_LwQ <= 1
 T515_L0_f6_le: const_Year_Remod_Add + 998046 t515_leaf0 <= 1e+06
 T515_L1_f214_le: x_b1_is_LwQ <= 1
 T515_R1_f6_ge: const_Year_Remod_Add - 1.001954e+06 t515_leaf1 >= -1e+06
 T515_L1_f4_le: const_Overall_Cond + 999996 t515_leaf1 <= 1e+06
 T515_L2_f214_le: x_b1_is_LwQ <= 1
 T515_R2_f6_ge: const_Year_Remod_Add - 1.001954e+06 t515_leaf2 >= -1e+06
 T515_R2_f4_ge: const_Overall_Cond - 1.000004e+06 t515_leaf2 >= -1e+06
 T515_R3_f214_ge: x_b1_is_LwQ - t515_leaf3 >= 0
 TREE_516_ONEHOT: t516_leaf0 + t516_leaf1 + t516_leaf2 + t516_leaf3
   + t516_leaf4 = 1
 T516_L0_f295_le: const_Sale_Condition_Alloca + 999999 t516_leaf0 <= 1e+06
 T516_L0_f3_le: Overall_Qual_calc + 999993 t516_leaf0 <= 1e+06
 T516_L0_f214_le: x_b1_is_LwQ <= 1
 T516_L1_f295_le: const_Sale_Condition_Alloca + 999999 t516_leaf1 <= 1e+06
 T516_L1_f3_le: Overall_Qual_calc + 999993 t516_leaf1 <= 1e+06
 T516_R1_f214_ge: x_b1_is_LwQ - t516_leaf1 >= 0
 T516_L2_f295_le: const_Sale_Condition_Alloca + 999999 t516_leaf2 <= 1e+06
 T516_R2_f3_ge: Overall_Qual_calc - 1.000007e+06 t516_leaf2 >= -1e+06
 T516_L2_f296_le: const_Sale_Condition_Family + 999999 t516_leaf2 <= 1e+06
 T516_L3_f295_le: const_Sale_Condition_Alloca + 999999 t516_leaf3 <= 1e+06
 T516_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t516_leaf3 >= -1e+06
 T516_R3_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t516_leaf3
   >= -1e+06
 T516_R4_f295_ge: const_Sale_Condition_Alloca - 1.000001e+06 t516_leaf4
   >= -1e+06
 TREE_517_ONEHOT: t517_leaf0 + t517_leaf1 + t517_leaf2 + t517_leaf3
   + t517_leaf4 + t517_leaf5 = 1
 T517_L0_f33_le: x_Garage_Area + 99680 t517_leaf0 <= 100000
 T517_L0_f12_le: x_BsmtFin_SF_1 + 999024 t517_leaf0 <= 1e+06
 T517_L0_f177_le: x_ex2_is_AsbShng <= 1
 T517_L1_f33_le: x_Garage_Area + 99680 t517_leaf1 <= 100000
 T517_L1_f12_le: x_BsmtFin_SF_1 + 999024 t517_leaf1 <= 1e+06
 T517_R1_f177_ge: x_ex2_is_AsbShng - t517_leaf1 >= 0
 T517_L2_f33_le: x_Garage_Area + 99680 t517_leaf2 <= 100000
 T517_R2_f12_ge: x_BsmtFin_SF_1 - 976 t517_leaf2 >= 0
 T517_R3_f33_ge: x_Garage_Area - 320 t517_leaf3 >= 0
 T517_L3_f20_le: x_Gr_Liv_Area + 9117 t517_leaf3 <= 10000
 T517_R4_f33_ge: x_Garage_Area - 320 t517_leaf4 >= 0
 T517_R4_f20_ge: x_Gr_Liv_Area - 883 t517_leaf4 >= 0
 T517_L4_f33_le: x_Garage_Area + 99129 t517_leaf4 <= 100000
 T517_R5_f33_ge: x_Garage_Area - 320 t517_leaf5 >= 0
 T517_R5_f20_ge: x_Gr_Liv_Area - 883 t517_leaf5 >= 0
 T517_R5_f33_ge: x_Garage_Area - 871 t517_leaf5 >= 0
 TREE_518_ONEHOT: t518_leaf0 + t518_leaf1 + t518_leaf2 + t518_leaf3
   + t518_leaf4 + t518_leaf5 = 1
 T518_L0_f161_le: x_ex1_is_AsbShng <= 1
 T518_L0_f31_le: const_Garage_Yr_Blt + 998092 t518_leaf0 <= 1e+06
 T518_L0_f10_le: const_Bsmt_Qual + 999998 t518_leaf0 <= 1e+06
 T518_L1_f161_le: x_ex1_is_AsbShng <= 1
 T518_L1_f31_le: const_Garage_Yr_Blt + 998092 t518_leaf1 <= 1e+06
 T518_R1_f10_ge: const_Bsmt_Qual - 1.000002e+06 t518_leaf1 >= -1e+06
 T518_L2_f161_le: x_ex1_is_AsbShng <= 1
 T518_R2_f31_ge: const_Garage_Yr_Blt - 1.001908e+06 t518_leaf2 >= -1e+06
 T518_L2_f37_le: x_Open_Porch_SF + 99812 t518_leaf2 <= 100000
 T518_L3_f161_le: x_ex1_is_AsbShng <= 1
 T518_R3_f31_ge: const_Garage_Yr_Blt - 1.001908e+06 t518_leaf3 >= -1e+06
 T518_R3_f37_ge: x_Open_Porch_SF - 188 t518_leaf3 >= 0
 T518_R4_f161_ge: x_ex1_is_AsbShng - t518_leaf4 >= 0
 T518_L4_f15_le: const_Total_Bsmt_SF + 999273 t518_leaf4 <= 1e+06
 T518_R5_f161_ge: x_ex1_is_AsbShng - t518_leaf5 >= 0
 T518_R5_f15_ge: const_Total_Bsmt_SF - 1.000727e+06 t518_leaf5 >= -1e+06
 TREE_519_ONEHOT: t519_leaf0 + t519_leaf1 = 1
 T519_L0_f4_le: const_Overall_Cond + 999997 t519_leaf0 <= 1e+06
 T519_R1_f4_ge: const_Overall_Cond - 1.000003e+06 t519_leaf1 >= -1e+06
 TREE_520_ONEHOT: t520_leaf0 + t520_leaf1 + t520_leaf2 + t520_leaf3
   + t520_leaf4 + t520_leaf5 = 1
 T520_L0_f12_le: x_BsmtFin_SF_1 + 998741 t520_leaf0 <= 1e+06
 T520_L0_f31_le: const_Garage_Yr_Blt + 998040 t520_leaf0 <= 1e+06
 T520_L0_f17_le: x_1st_Flr_SF + 4202 t520_leaf0 <= 5000
 T520_L1_f12_le: x_BsmtFin_SF_1 + 998741 t520_leaf1 <= 1e+06
 T520_L1_f31_le: const_Garage_Yr_Blt + 998040 t520_leaf1 <= 1e+06
 T520_R1_f17_ge: x_1st_Flr_SF - 798 t520_leaf1 >= 0
 T520_L2_f12_le: x_BsmtFin_SF_1 + 998741 t520_leaf2 <= 1e+06
 T520_R2_f31_ge: const_Garage_Yr_Blt - 1.00196e+06 t520_leaf2 >= -1e+06
 T520_R3_f12_ge: x_BsmtFin_SF_1 - 1259 t520_leaf3 >= 0
 T520_L3_f1_le: const_Lot_Area + 984502 t520_leaf3 <= 1e+06
 T520_R4_f12_ge: x_BsmtFin_SF_1 - 1259 t520_leaf4 >= 0
 T520_R4_f1_ge: const_Lot_Area - 1.015498e+06 t520_leaf4 >= -1e+06
 T520_L4_f0_le: const_Lot_Frontage + 999940 t520_leaf4 <= 1e+06
 T520_R5_f12_ge: x_BsmtFin_SF_1 - 1259 t520_leaf5 >= 0
 T520_R5_f1_ge: const_Lot_Area - 1.015498e+06 t520_leaf5 >= -1e+06
 T520_R5_f0_ge: const_Lot_Frontage - 1.00006e+06 t520_leaf5 >= -1e+06
 TREE_521_ONEHOT: t521_leaf0 + t521_leaf1 + t521_leaf2 + t521_leaf3
   + t521_leaf4 = 1
 T521_L0_f3_le: Overall_Qual_calc + 999997 t521_leaf0 <= 1e+06
 T521_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t521_leaf0
   <= 1e+06
 T521_L1_f3_le: Overall_Qual_calc + 999997 t521_leaf1 <= 1e+06
 T521_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t521_leaf1
   >= -1e+06
 T521_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t521_leaf2 >= -1e+06
 T521_L2_f164_le: x_ex1_is_BrkFace <= 1
 T521_L2_f1_le: const_Lot_Area + 986948 t521_leaf2 <= 1e+06
 T521_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t521_leaf3 >= -1e+06
 T521_L3_f164_le: x_ex1_is_BrkFace <= 1
 T521_R3_f1_ge: const_Lot_Area - 1.013052e+06 t521_leaf3 >= -1e+06
 T521_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t521_leaf4 >= -1e+06
 T521_R4_f164_ge: x_ex1_is_BrkFace - t521_leaf4 >= 0
 TREE_522_ONEHOT: t522_leaf0 + t522_leaf1 + t522_leaf2 + t522_leaf3
   + t522_leaf4 + t522_leaf5 = 1
 T522_L0_f3_le: Overall_Qual_calc + 999997 t522_leaf0 <= 1e+06
 T522_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t522_leaf0
   <= 1e+06
 T522_L1_f3_le: Overall_Qual_calc + 999997 t522_leaf1 <= 1e+06
 T522_R1_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t522_leaf1
   >= -1e+06
 T522_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t522_leaf2 >= -1e+06
 T522_L2_f185_le: x_ex2_is_MetalSd <= 1
 T522_L2_f26_le: x_Kitchen_AbvGr + t522_leaf2 <= 3
 T522_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t522_leaf3 >= -1e+06
 T522_L3_f185_le: x_ex2_is_MetalSd <= 1
 T522_R3_f26_ge: x_Kitchen_AbvGr - 2 t522_leaf3 >= 0
 T522_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t522_leaf4 >= -1e+06
 T522_R4_f185_ge: x_ex2_is_MetalSd - t522_leaf4 >= 0
 T522_L4_f31_le: const_Garage_Yr_Blt + 997994 t522_leaf4 <= 1e+06
 T522_R5_f3_ge: Overall_Qual_calc - 1.000003e+06 t522_leaf5 >= -1e+06
 T522_R5_f185_ge: x_ex2_is_MetalSd - t522_leaf5 >= 0
 T522_R5_f31_ge: const_Garage_Yr_Blt - 1.002006e+06 t522_leaf5 >= -1e+06
 TREE_523_ONEHOT: t523_leaf0 + t523_leaf1 + t523_leaf2 + t523_leaf3
   + t523_leaf4 = 1
 T523_L0_f3_le: Overall_Qual_calc + 999995 t523_leaf0 <= 1e+06
 T523_L0_f177_le: x_ex2_is_AsbShng <= 1
 T523_L0_f197_le: x_mvt_is_No_aplica <= 1
 T523_L1_f3_le: Overall_Qual_calc + 999995 t523_leaf1 <= 1e+06
 T523_L1_f177_le: x_ex2_is_AsbShng <= 1
 T523_R1_f197_ge: x_mvt_is_No_aplica - t523_leaf1 >= 0
 T523_L2_f3_le: Overall_Qual_calc + 999995 t523_leaf2 <= 1e+06
 T523_R2_f177_ge: x_ex2_is_AsbShng - t523_leaf2 >= 0
 T523_L2_f251_le: const_Garage_Type_Detchd + 999999 t523_leaf2 <= 1e+06
 T523_L3_f3_le: Overall_Qual_calc + 999995 t523_leaf3 <= 1e+06
 T523_R3_f177_ge: x_ex2_is_AsbShng - t523_leaf3 >= 0
 T523_R3_f251_ge: const_Garage_Type_Detchd - 1.000001e+06 t523_leaf3
   >= -1e+06
 T523_R4_f3_ge: Overall_Qual_calc - 1.000005e+06 t523_leaf4 >= -1e+06
 TREE_524_ONEHOT: t524_leaf0 + t524_leaf1 + t524_leaf2 + t524_leaf3 = 1
 T524_L0_f11_le: x_Bsmt_Cond + 3 t524_leaf0 <= 4
 T524_L0_f276_le: const_Mo_Sold_6 + 999999 t524_leaf0 <= 1e+06
 T524_L1_f11_le: x_Bsmt_Cond + 3 t524_leaf1 <= 4
 T524_R1_f276_ge: const_Mo_Sold_6 - 1.000001e+06 t524_leaf1 >= -1e+06
 T524_R2_f11_ge: x_Bsmt_Cond - 2 t524_leaf2 >= -1
 T524_L2_f23_le: x_Full_Bath + 2 t524_leaf2 <= 4
 T524_R3_f11_ge: x_Bsmt_Cond - 2 t524_leaf3 >= -1
 T524_R3_f23_ge: x_Full_Bath - 2 t524_leaf3 >= 0
 TREE_525_ONEHOT: t525_leaf0 + t525_leaf1 = 1
 T525_L0_f60_le: const_MS_SubClass_190 + 999999 t525_leaf0 <= 1e+06
 T525_R1_f60_ge: const_MS_SubClass_190 - 1.000001e+06 t525_leaf1 >= -1e+06
 TREE_526_ONEHOT: t526_leaf0 + t526_leaf1 + t526_leaf2 + t526_leaf3
   + t526_leaf4 + t526_leaf5 = 1
 T526_L0_f161_le: x_ex1_is_AsbShng <= 1
 T526_L0_f25_le: x_Bedroom_AbvGr + 3 t526_leaf0 <= 6
 T526_L0_f209_le: const_Bsmt_Exposure_No + 999999 t526_leaf0 <= 1e+06
 T526_L1_f161_le: x_ex1_is_AsbShng <= 1
 T526_L1_f25_le: x_Bedroom_AbvGr + 3 t526_leaf1 <= 6
 T526_R1_f209_ge: const_Bsmt_Exposure_No - 1.000001e+06 t526_leaf1
   >= -1e+06
 T526_L2_f161_le: x_ex1_is_AsbShng <= 1
 T526_R2_f25_ge: x_Bedroom_AbvGr - 3 t526_leaf2 >= 0
 T526_L2_f11_le: x_Bsmt_Cond + t526_leaf2 <= 4
 T526_L3_f161_le: x_ex1_is_AsbShng <= 1
 T526_R3_f25_ge: x_Bedroom_AbvGr - 3 t526_leaf3 >= 0
 T526_R3_f11_ge: x_Bsmt_Cond - 4 t526_leaf3 >= -1
 T526_R4_f161_ge: x_ex1_is_AsbShng - t526_leaf4 >= 0
 T526_L4_f0_le: const_Lot_Frontage + 999934 t526_leaf4 <= 1e+06
 T526_R5_f161_ge: x_ex1_is_AsbShng - t526_leaf5 >= 0
 T526_R5_f0_ge: const_Lot_Frontage - 1.000066e+06 t526_leaf5 >= -1e+06
 TREE_527_ONEHOT: t527_leaf0 + t527_leaf1 + t527_leaf2 + t527_leaf3
   + t527_leaf4 + t527_leaf5 = 1
 T527_L0_f1_le: const_Lot_Area + 978220 t527_leaf0 <= 1e+06
 T527_L0_f231_le: central_air_no + 999999 t527_leaf0 <= 1e+06
 T527_L0_f213_le: x_b1_is_GLQ <= 1
 T527_L1_f1_le: const_Lot_Area + 978220 t527_leaf1 <= 1e+06
 T527_L1_f231_le: central_air_no + 999999 t527_leaf1 <= 1e+06
 T527_R1_f213_ge: x_b1_is_GLQ - t527_leaf1 >= 0
 T527_L2_f1_le: const_Lot_Area + 978220 t527_leaf2 <= 1e+06
 T527_R2_f231_ge: central_air_no - 1.000001e+06 t527_leaf2 >= -1e+06
 T527_L2_f66_le: const_MS_Zoning_RL + 999999 t527_leaf2 <= 1e+06
 T527_L3_f1_le: const_Lot_Area + 978220 t527_leaf3 <= 1e+06
 T527_R3_f231_ge: central_air_no - 1.000001e+06 t527_leaf3 >= -1e+06
 T527_R3_f66_ge: const_MS_Zoning_RL - 1.000001e+06 t527_leaf3 >= -1e+06
 T527_R4_f1_ge: const_Lot_Area - 1.02178e+06 t527_leaf4 >= -1e+06
 T527_L4_f213_le: x_b1_is_GLQ <= 1
 T527_R5_f1_ge: const_Lot_Area - 1.02178e+06 t527_leaf5 >= -1e+06
 T527_R5_f213_ge: x_b1_is_GLQ - t527_leaf5 >= 0
 TREE_528_ONEHOT: t528_leaf0 + t528_leaf1 + t528_leaf2 = 1
 T528_L0_f114_le: const_Neighborhood_StoneBr + 999999 t528_leaf0 <= 1e+06
 T528_R1_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t528_leaf1
   >= -1e+06
 T528_L1_f289_le: const_Sale_Type_New + 999999 t528_leaf1 <= 1e+06
 T528_R2_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t528_leaf2
   >= -1e+06
 T528_R2_f289_ge: const_Sale_Type_New - 1.000001e+06 t528_leaf2 >= -1e+06
 TREE_529_ONEHOT: t529_leaf0 + t529_leaf1 + t529_leaf2 + t529_leaf3
   + t529_leaf4 = 1
 T529_L0_f199_le: x_mvt_is_Stone <= 1
 T529_L0_f33_le: x_Garage_Area + 99680 t529_leaf0 <= 100000
 T529_L0_f232_le: central_air_yes + 999999 t529_leaf0 <= 1e+06
 T529_L1_f199_le: x_mvt_is_Stone <= 1
 T529_L1_f33_le: x_Garage_Area + 99680 t529_leaf1 <= 100000
 T529_R1_f232_ge: central_air_yes - 1.000001e+06 t529_leaf1 >= -1e+06
 T529_L2_f199_le: x_mvt_is_Stone <= 1
 T529_R2_f33_ge: x_Garage_Area - 320 t529_leaf2 >= 0
 T529_L2_f36_le: x_Wood_Deck_SF + 99726 t529_leaf2 <= 100000
 T529_L3_f199_le: x_mvt_is_Stone <= 1
 T529_R3_f33_ge: x_Garage_Area - 320 t529_leaf3 >= 0
 T529_R3_f36_ge: x_Wood_Deck_SF - 274 t529_leaf3 >= 0
 T529_R4_f199_ge: x_mvt_is_Stone - t529_leaf4 >= 0
 TREE_530_ONEHOT: t530_leaf0 = 1
 TREE_531_ONEHOT: t531_leaf0 + t531_leaf1 + t531_leaf2 = 1
 T531_L0_f5_le: const_Year_Built + 998086 t531_leaf0 <= 1e+06
 T531_L0_f10_le: const_Bsmt_Qual + 999998 t531_leaf0 <= 1e+06
 T531_L1_f5_le: const_Year_Built + 998086 t531_leaf1 <= 1e+06
 T531_R1_f10_ge: const_Bsmt_Qual - 1.000002e+06 t531_leaf1 >= -1e+06
 T531_R2_f5_ge: const_Year_Built - 1.001914e+06 t531_leaf2 >= -1e+06
 TREE_532_ONEHOT: t532_leaf0 = 1
 TREE_533_ONEHOT: t533_leaf0 + t533_leaf1 + t533_leaf2 = 1
 T533_L0_f62_le: const_MS_Zoning_C_(all) + 999999 t533_leaf0 <= 1e+06
 T533_R1_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t533_leaf1
   >= -1e+06
 T533_L1_f1_le: const_Lot_Area + 993400 t533_leaf1 <= 1e+06
 T533_R2_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t533_leaf2
   >= -1e+06
 T533_R2_f1_ge: const_Lot_Area - 1.0066e+06 t533_leaf2 >= -1e+06
 TREE_534_ONEHOT: t534_leaf0 = 1
 TREE_535_ONEHOT: t535_leaf0 + t535_leaf1 + t535_leaf2 + t535_leaf3
   + t535_leaf4 = 1
 T535_L0_f14_le: x_Bsmt_Unf_SF + 99524 t535_leaf0 <= 100000
 T535_L0_f1_le: const_Lot_Area + 978630 t535_leaf0 <= 1e+06
 T535_L0_f12_le: x_BsmtFin_SF_1 + 998035 t535_leaf0 <= 1e+06
 T535_L1_f14_le: x_Bsmt_Unf_SF + 99524 t535_leaf1 <= 100000
 T535_L1_f1_le: const_Lot_Area + 978630 t535_leaf1 <= 1e+06
 T535_R1_f12_ge: x_BsmtFin_SF_1 - 1965 t535_leaf1 >= 0
 T535_L2_f14_le: x_Bsmt_Unf_SF + 99524 t535_leaf2 <= 100000
 T535_R2_f1_ge: const_Lot_Area - 1.02137e+06 t535_leaf2 >= -1e+06
 T535_R3_f14_ge: x_Bsmt_Unf_SF - 476 t535_leaf3 >= 0
 T535_L3_f18_le: const_2nd_Flr_SF + 998629 t535_leaf3 <= 1e+06
 T535_R4_f14_ge: x_Bsmt_Unf_SF - 476 t535_leaf4 >= 0
 T535_R4_f18_ge: const_2nd_Flr_SF - 1.001371e+06 t535_leaf4 >= -1e+06
 TREE_536_ONEHOT: t536_leaf0 = 1
 TREE_537_ONEHOT: t537_leaf0 + t537_leaf1 = 1
 T537_L0_f75_le: const_Lot_Shape_IR3 + 999999 t537_leaf0 <= 1e+06
 T537_R1_f75_ge: const_Lot_Shape_IR3 - 1.000001e+06 t537_leaf1 >= -1e+06
 TREE_538_ONEHOT: t538_leaf0 + t538_leaf1 + t538_leaf2 + t538_leaf3
   + t538_leaf4 = 1
 T538_L0_f3_le: Overall_Qual_calc + 999993 t538_leaf0 <= 1e+06
 T538_L0_f12_le: x_BsmtFin_SF_1 + 999082 t538_leaf0 <= 1e+06
 T538_L1_f3_le: Overall_Qual_calc + 999993 t538_leaf1 <= 1e+06
 T538_R1_f12_ge: x_BsmtFin_SF_1 - 918 t538_leaf1 >= 0
 T538_R2_f3_ge: Overall_Qual_calc - 1.000007e+06 t538_leaf2 >= -1e+06
 T538_L2_f14_le: x_Bsmt_Unf_SF + 98205 t538_leaf2 <= 100000
 T538_L2_f206_le: const_Bsmt_Exposure_Av + 999999 t538_leaf2 <= 1e+06
 T538_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t538_leaf3 >= -1e+06
 T538_L3_f14_le: x_Bsmt_Unf_SF + 98205 t538_leaf3 <= 100000
 T538_R3_f206_ge: const_Bsmt_Exposure_Av - 1.000001e+06 t538_leaf3
   >= -1e+06
 T538_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t538_leaf4 >= -1e+06
 T538_R4_f14_ge: x_Bsmt_Unf_SF - 1795 t538_leaf4 >= 0
 TREE_539_ONEHOT: t539_leaf0 + t539_leaf1 = 1
 T539_L0_f44_le: const_Yr_Sold + 997993 t539_leaf0 <= 1e+06
 T539_R1_f44_ge: const_Yr_Sold - 1.002007e+06 t539_leaf1 >= -1e+06
 TREE_540_ONEHOT: t540_leaf0 + t540_leaf1 + t540_leaf2 + t540_leaf3
   + t540_leaf4 = 1
 T540_L0_f277_le: const_Mo_Sold_7 + 999999 t540_leaf0 <= 1e+06
 T540_L0_f35_le: x_Garage_Cond + 3 t540_leaf0 <= 4
 T540_L0_f1_le: const_Lot_Area + 991200 t540_leaf0 <= 1e+06
 T540_L1_f277_le: const_Mo_Sold_7 + 999999 t540_leaf1 <= 1e+06
 T540_L1_f35_le: x_Garage_Cond + 3 t540_leaf1 <= 4
 T540_R1_f1_ge: const_Lot_Area - 1.0088e+06 t540_leaf1 >= -1e+06
 T540_L2_f277_le: const_Mo_Sold_7 + 999999 t540_leaf2 <= 1e+06
 T540_R2_f35_ge: x_Garage_Cond - 2 t540_leaf2 >= -1
 T540_L2_f3_le: Overall_Qual_calc + 999997 t540_leaf2 <= 1e+06
 T540_L3_f277_le: const_Mo_Sold_7 + 999999 t540_leaf3 <= 1e+06
 T540_R3_f35_ge: x_Garage_Cond - 2 t540_leaf3 >= -1
 T540_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t540_leaf3 >= -1e+06
 T540_R4_f277_ge: const_Mo_Sold_7 - 1.000001e+06 t540_leaf4 >= -1e+06
 TREE_541_ONEHOT: t541_leaf0 + t541_leaf1 + t541_leaf2 = 1
 T541_L0_f8_le: x_Exter_Qual + 2 t541_leaf0 <= 4
 T541_L0_f33_le: x_Garage_Area + 99703 t541_leaf0 <= 100000
 T541_L1_f8_le: x_Exter_Qual + 2 t541_leaf1 <= 4
 T541_R1_f33_ge: x_Garage_Area - 297 t541_leaf1 >= 0
 T541_R2_f8_ge: x_Exter_Qual - 3 t541_leaf2 >= -1
 TREE_542_ONEHOT: t542_leaf0 = 1
 TREE_543_ONEHOT: t543_leaf0 + t543_leaf1 + t543_leaf2 + t543_leaf3
   + t543_leaf4 + t543_leaf5 + t543_leaf6 = 1
 T543_L0_f8_le: x_Exter_Qual + t543_leaf0 <= 4
 T543_L0_f20_le: x_Gr_Liv_Area + 8266 t543_leaf0 <= 10000
 T543_L0_f71_le: const_Alley_No_aplica + 999999 t543_leaf0 <= 1e+06
 T543_L1_f8_le: x_Exter_Qual + t543_leaf1 <= 4
 T543_L1_f20_le: x_Gr_Liv_Area + 8266 t543_leaf1 <= 10000
 T543_R1_f71_ge: const_Alley_No_aplica - 1.000001e+06 t543_leaf1 >= -1e+06
 T543_L2_f8_le: x_Exter_Qual + t543_leaf2 <= 4
 T543_R2_f20_ge: x_Gr_Liv_Area - 1734 t543_leaf2 >= 0
 T543_L2_f67_le: const_MS_Zoning_RM + 999999 t543_leaf2 <= 1e+06
 T543_L3_f8_le: x_Exter_Qual + t543_leaf3 <= 4
 T543_R3_f20_ge: x_Gr_Liv_Area - 1734 t543_leaf3 >= 0
 T543_R3_f67_ge: const_MS_Zoning_RM - 1.000001e+06 t543_leaf3 >= -1e+06
 T543_R4_f8_ge: x_Exter_Qual - 4 t543_leaf4 >= -1
 T543_L4_f77_le: const_Land_Contour_Bnk + 999999 t543_leaf4 <= 1e+06
 T543_L4_f31_le: const_Garage_Yr_Blt + 998007 t543_leaf4 <= 1e+06
 T543_R5_f8_ge: x_Exter_Qual - 4 t543_leaf5 >= -1
 T543_L5_f77_le: const_Land_Contour_Bnk + 999999 t543_leaf5 <= 1e+06
 T543_R5_f31_ge: const_Garage_Yr_Blt - 1.001993e+06 t543_leaf5 >= -1e+06
 T543_R6_f8_ge: x_Exter_Qual - 4 t543_leaf6 >= -1
 T543_R6_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t543_leaf6
   >= -1e+06
 TREE_544_ONEHOT: t544_leaf0 = 1
 TREE_545_ONEHOT: t545_leaf0 + t545_leaf1 + t545_leaf2 + t545_leaf3
   + t545_leaf4 = 1
 T545_L0_f0_le: const_Lot_Frontage + 999885 t545_leaf0 <= 1e+06
 T545_L0_f201_le: const_Foundation_CBlock + 999999 t545_leaf0 <= 1e+06
 T545_L1_f0_le: const_Lot_Frontage + 999885 t545_leaf1 <= 1e+06
 T545_R1_f201_ge: const_Foundation_CBlock - 1.000001e+06 t545_leaf1
   >= -1e+06
 T545_L1_f30_le: x_Fireplace_Qu + 4 t545_leaf1 <= 4
 T545_L2_f0_le: const_Lot_Frontage + 999885 t545_leaf2 <= 1e+06
 T545_R2_f201_ge: const_Foundation_CBlock - 1.000001e+06 t545_leaf2
   >= -1e+06
 T545_R2_f30_ge: x_Fireplace_Qu - t545_leaf2 >= -1
 T545_R3_f0_ge: const_Lot_Frontage - 1.000115e+06 t545_leaf3 >= -1e+06
 T545_L3_f8_le: x_Exter_Qual + t545_leaf3 <= 4
 T545_R4_f0_ge: const_Lot_Frontage - 1.000115e+06 t545_leaf4 >= -1e+06
 T545_R4_f8_ge: x_Exter_Qual - 4 t545_leaf4 >= -1
 TREE_546_ONEHOT: t546_leaf0 + t546_leaf1 = 1
 T546_L0_f120_le: const_Condition_1_PosA + 999999 t546_leaf0 <= 1e+06
 T546_R1_f120_ge: const_Condition_1_PosA - 1.000001e+06 t546_leaf1
   >= -1e+06
 TREE_547_ONEHOT: t547_leaf0 + t547_leaf1 = 1
 T547_L0_f0_le: const_Lot_Frontage + 999966 t547_leaf0 <= 1e+06
 T547_R1_f0_ge: const_Lot_Frontage - 1.000034e+06 t547_leaf1 >= -1e+06
 TREE_548_ONEHOT: t548_leaf0 + t548_leaf1 + t548_leaf2 + t548_leaf3 = 1
 T548_L0_f233_le: x_elect_is_FuseA <= 1
 T548_R1_f233_ge: x_elect_is_FuseA - t548_leaf1 >= 0
 T548_L1_f43_le: const_Misc_Val + 999946 t548_leaf1 <= 1e+06
 T548_L1_f177_le: x_ex2_is_AsbShng <= 1
 T548_R2_f233_ge: x_elect_is_FuseA - t548_leaf2 >= 0
 T548_L2_f43_le: const_Misc_Val + 999946 t548_leaf2 <= 1e+06
 T548_R2_f177_ge: x_ex2_is_AsbShng - t548_leaf2 >= 0
 T548_R3_f233_ge: x_elect_is_FuseA - t548_leaf3 >= 0
 T548_R3_f43_ge: const_Misc_Val - 1.000054e+06 t548_leaf3 >= -1e+06
 TREE_549_ONEHOT: t549_leaf0 + t549_leaf1 + t549_leaf2 + t549_leaf3
   + t549_leaf4 + t549_leaf5 = 1
 T549_L0_f26_le: x_Kitchen_AbvGr + t549_leaf0 <= 3
 T549_L0_f20_le: x_Gr_Liv_Area + 9253 t549_leaf0 <= 10000
 T549_L0_f17_le: x_1st_Flr_SF + 4288 t549_leaf0 <= 5000
 T549_L1_f26_le: x_Kitchen_AbvGr + t549_leaf1 <= 3
 T549_L1_f20_le: x_Gr_Liv_Area + 9253 t549_leaf1 <= 10000
 T549_R1_f17_ge: x_1st_Flr_SF - 712 t549_leaf1 >= 0
 T549_L2_f26_le: x_Kitchen_AbvGr + t549_leaf2 <= 3
 T549_R2_f20_ge: x_Gr_Liv_Area - 747 t549_leaf2 >= 0
 T549_L2_f3_le: Overall_Qual_calc + 999995 t549_leaf2 <= 1e+06
 T549_L3_f26_le: x_Kitchen_AbvGr + t549_leaf3 <= 3
 T549_R3_f20_ge: x_Gr_Liv_Area - 747 t549_leaf3 >= 0
 T549_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t549_leaf3 >= -1e+06
 T549_R4_f26_ge: x_Kitchen_AbvGr - 2 t549_leaf4 >= 0
 T549_L4_f9_le: x_Exter_Cond + 2 t549_leaf4 <= 4
 T549_R5_f26_ge: x_Kitchen_AbvGr - 2 t549_leaf5 >= 0
 T549_R5_f9_ge: x_Exter_Cond - 3 t549_leaf5 >= -1
 TREE_550_ONEHOT: t550_leaf0 + t550_leaf1 + t550_leaf2 + t550_leaf3 = 1
 T550_L0_f16_le: x_Heating_QC + 2 t550_leaf0 <= 4
 T550_R1_f16_ge: x_Heating_QC - 3 t550_leaf1 >= -1
 T550_L1_f7_le: x_Mas_Vnr_Area + 1078 t550_leaf1 <= 2000
 T550_L1_f0_le: const_Lot_Frontage + 999915 t550_leaf1 <= 1e+06
 T550_R2_f16_ge: x_Heating_QC - 3 t550_leaf2 >= -1
 T550_L2_f7_le: x_Mas_Vnr_Area + 1078 t550_leaf2 <= 2000
 T550_R2_f0_ge: const_Lot_Frontage - 1.000085e+06 t550_leaf2 >= -1e+06
 T550_R3_f16_ge: x_Heating_QC - 3 t550_leaf3 >= -1
 T550_R3_f7_ge: x_Mas_Vnr_Area - 922 t550_leaf3 >= 0
 TREE_551_ONEHOT: t551_leaf0 = 1
 TREE_552_ONEHOT: t552_leaf0 + t552_leaf1 + t552_leaf2 + t552_leaf3
   + t552_leaf4 = 1
 T552_L0_f277_le: const_Mo_Sold_7 + 999999 t552_leaf0 <= 1e+06
 T552_L0_f1_le: const_Lot_Area + 980478 t552_leaf0 <= 1e+06
 T552_L0_f15_le: const_Total_Bsmt_SF + 999755 t552_leaf0 <= 1e+06
 T552_L1_f277_le: const_Mo_Sold_7 + 999999 t552_leaf1 <= 1e+06
 T552_L1_f1_le: const_Lot_Area + 980478 t552_leaf1 <= 1e+06
 T552_R1_f15_ge: const_Total_Bsmt_SF - 1.000245e+06 t552_leaf1 >= -1e+06
 T552_L2_f277_le: const_Mo_Sold_7 + 999999 t552_leaf2 <= 1e+06
 T552_R2_f1_ge: const_Lot_Area - 1.019522e+06 t552_leaf2 >= -1e+06
 T552_L2_f0_le: const_Lot_Frontage + 999826 t552_leaf2 <= 1e+06
 T552_L3_f277_le: const_Mo_Sold_7 + 999999 t552_leaf3 <= 1e+06
 T552_R3_f1_ge: const_Lot_Area - 1.019522e+06 t552_leaf3 >= -1e+06
 T552_R3_f0_ge: const_Lot_Frontage - 1.000174e+06 t552_leaf3 >= -1e+06
 T552_R4_f277_ge: const_Mo_Sold_7 - 1.000001e+06 t552_leaf4 >= -1e+06
 TREE_553_ONEHOT: t553_leaf0 + t553_leaf1 + t553_leaf2 + t553_leaf3 = 1
 T553_L0_f4_le: const_Overall_Cond + 999994 t553_leaf0 <= 1e+06
 T553_L0_f5_le: const_Year_Built + 998047 t553_leaf0 <= 1e+06
 T553_L0_f5_le: const_Year_Built + 998098 t553_leaf0 <= 1e+06
 T553_L1_f4_le: const_Overall_Cond + 999994 t553_leaf1 <= 1e+06
 T553_L1_f5_le: const_Year_Built + 998047 t553_leaf1 <= 1e+06
 T553_R1_f5_ge: const_Year_Built - 1.001902e+06 t553_leaf1 >= -1e+06
 T553_L2_f4_le: const_Overall_Cond + 999994 t553_leaf2 <= 1e+06
 T553_R2_f5_ge: const_Year_Built - 1.001953e+06 t553_leaf2 >= -1e+06
 T553_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t553_leaf3 >= -1e+06
 TREE_554_ONEHOT: t554_leaf0 = 1
 TREE_555_ONEHOT: t555_leaf0 = 1
 TREE_556_ONEHOT: t556_leaf0 + t556_leaf1 = 1
 T556_L0_f20_le: x_Gr_Liv_Area + 9232 t556_leaf0 <= 10000
 T556_R1_f20_ge: x_Gr_Liv_Area - 768 t556_leaf1 >= 0
 TREE_557_ONEHOT: t557_leaf0 + t557_leaf1 + t557_leaf2 + t557_leaf3 = 1
 T557_L0_f1_le: const_Lot_Area + 994000 t557_leaf0 <= 1e+06
 T557_R1_f1_ge: const_Lot_Area - 1.006e+06 t557_leaf1 >= -1e+06
 T557_L1_f29_le: const_Fireplaces + 999997 t557_leaf1 <= 1e+06
 T557_L1_f28_le: x_TotRms_AbvGrd + 10 t557_leaf1 <= 15
 T557_R2_f1_ge: const_Lot_Area - 1.006e+06 t557_leaf2 >= -1e+06
 T557_L2_f29_le: const_Fireplaces + 999997 t557_leaf2 <= 1e+06
 T557_R2_f28_ge: x_TotRms_AbvGrd - 5 t557_leaf2 >= 0
 T557_R3_f1_ge: const_Lot_Area - 1.006e+06 t557_leaf3 >= -1e+06
 T557_R3_f29_ge: const_Fireplaces - 1.000003e+06 t557_leaf3 >= -1e+06
 TREE_558_ONEHOT: t558_leaf0 + t558_leaf1 + t558_leaf2 + t558_leaf3 = 1
 T558_L0_f33_le: x_Garage_Area + 99112 t558_leaf0 <= 100000
 T558_L0_f117_le: const_Condition_1_Artery + 999999 t558_leaf0 <= 1e+06
 T558_L0_f32_le: const_Garage_Cars + 999997 t558_leaf0 <= 1e+06
 T558_L1_f33_le: x_Garage_Area + 99112 t558_leaf1 <= 100000
 T558_L1_f117_le: const_Condition_1_Artery + 999999 t558_leaf1 <= 1e+06
 T558_R1_f32_ge: const_Garage_Cars - 1.000003e+06 t558_leaf1 >= -1e+06
 T558_L2_f33_le: x_Garage_Area + 99112 t558_leaf2 <= 100000
 T558_R2_f117_ge: const_Condition_1_Artery - 1.000001e+06 t558_leaf2
   >= -1e+06
 T558_R3_f33_ge: x_Garage_Area - 888 t558_leaf3 >= 0
 TREE_559_ONEHOT: t559_leaf0 + t559_leaf1 + t559_leaf2 + t559_leaf3
   + t559_leaf4 = 1
 T559_L0_f36_le: x_Wood_Deck_SF + 99784 t559_leaf0 <= 100000
 T559_L0_f4_le: const_Overall_Cond + 999994 t559_leaf0 <= 1e+06
 T559_L0_f216_le: x_b1_is_Rec <= 1
 T559_L1_f36_le: x_Wood_Deck_SF + 99784 t559_leaf1 <= 100000
 T559_L1_f4_le: const_Overall_Cond + 999994 t559_leaf1 <= 1e+06
 T559_R1_f216_ge: x_b1_is_Rec - t559_leaf1 >= 0
 T559_L2_f36_le: x_Wood_Deck_SF + 99784 t559_leaf2 <= 100000
 T559_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t559_leaf2 >= -1e+06
 T559_L2_f1_le: const_Lot_Area + 991374 t559_leaf2 <= 1e+06
 T559_L3_f36_le: x_Wood_Deck_SF + 99784 t559_leaf3 <= 100000
 T559_R3_f4_ge: const_Overall_Cond - 1.000006e+06 t559_leaf3 >= -1e+06
 T559_R3_f1_ge: const_Lot_Area - 1.008626e+06 t559_leaf3 >= -1e+06
 T559_R4_f36_ge: x_Wood_Deck_SF - 216 t559_leaf4 >= 0
 TREE_560_ONEHOT: t560_leaf0 + t560_leaf1 + t560_leaf2 + t560_leaf3 = 1
 T560_L0_f169_le: x_ex1_is_MetalSd <= 1
 T560_L0_f95_le: const_Neighborhood_Crawfor + 999999 t560_leaf0 <= 1e+06
 T560_L0_f1_le: const_Lot_Area + 987147 t560_leaf0 <= 1e+06
 T560_L1_f169_le: x_ex1_is_MetalSd <= 1
 T560_L1_f95_le: const_Neighborhood_Crawfor + 999999 t560_leaf1 <= 1e+06
 T560_R1_f1_ge: const_Lot_Area - 1.012853e+06 t560_leaf1 >= -1e+06
 T560_L2_f169_le: x_ex1_is_MetalSd <= 1
 T560_R2_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t560_leaf2
   >= -1e+06
 T560_R3_f169_ge: x_ex1_is_MetalSd - t560_leaf3 >= 0
 TREE_561_ONEHOT: t561_leaf0 + t561_leaf1 + t561_leaf2 = 1
 T561_L0_f20_le: x_Gr_Liv_Area + 9253 t561_leaf0 <= 10000
 T561_L0_f17_le: x_1st_Flr_SF + 4283 t561_leaf0 <= 5000
 T561_L1_f20_le: x_Gr_Liv_Area + 9253 t561_leaf1 <= 10000
 T561_R1_f17_ge: x_1st_Flr_SF - 717 t561_leaf1 >= 0
 T561_R2_f20_ge: x_Gr_Liv_Area - 747 t561_leaf2 >= 0
 TREE_562_ONEHOT: t562_leaf0 + t562_leaf1 + t562_leaf2 = 1
 T562_L0_f17_le: x_1st_Flr_SF + 3498 t562_leaf0 <= 5000
 T562_R1_f17_ge: x_1st_Flr_SF - 1502 t562_leaf1 >= 0
 T562_L1_f206_le: const_Bsmt_Exposure_Av + 999999 t562_leaf1 <= 1e+06
 T562_R2_f17_ge: x_1st_Flr_SF - 1502 t562_leaf2 >= 0
 T562_R2_f206_ge: const_Bsmt_Exposure_Av - 1.000001e+06 t562_leaf2
   >= -1e+06
 TREE_563_ONEHOT: t563_leaf0 = 1
 TREE_564_ONEHOT: t564_leaf0 + t564_leaf1 + t564_leaf2 = 1
 T564_L0_f177_le: x_ex2_is_AsbShng <= 1
 T564_R1_f177_ge: x_ex2_is_AsbShng - t564_leaf1 >= 0
 T564_L1_f15_le: const_Total_Bsmt_SF + 999273 t564_leaf1 <= 1e+06
 T564_R2_f177_ge: x_ex2_is_AsbShng - t564_leaf2 >= 0
 T564_R2_f15_ge: const_Total_Bsmt_SF - 1.000727e+06 t564_leaf2 >= -1e+06
 TREE_565_ONEHOT: t565_leaf0 = 1
 TREE_566_ONEHOT: t566_leaf0 + t566_leaf1 + t566_leaf2 + t566_leaf3
   + t566_leaf4 + t566_leaf5 = 1
 T566_L0_f6_le: const_Year_Remod_Add + 998042 t566_leaf0 <= 1e+06
 T566_L0_f15_le: const_Total_Bsmt_SF + 999288 t566_leaf0 <= 1e+06
 T566_L0_f1_le: const_Lot_Area + 991906 t566_leaf0 <= 1e+06
 T566_L1_f6_le: const_Year_Remod_Add + 998042 t566_leaf1 <= 1e+06
 T566_L1_f15_le: const_Total_Bsmt_SF + 999288 t566_leaf1 <= 1e+06
 T566_R1_f1_ge: const_Lot_Area - 1.008094e+06 t566_leaf1 >= -1e+06
 T566_L2_f6_le: const_Year_Remod_Add + 998042 t566_leaf2 <= 1e+06
 T566_R2_f15_ge: const_Total_Bsmt_SF - 1.000712e+06 t566_leaf2 >= -1e+06
 T566_L2_f24_le: x_Half_Bath + 4 t566_leaf2 <= 5
 T566_L3_f6_le: const_Year_Remod_Add + 998042 t566_leaf3 <= 1e+06
 T566_R3_f15_ge: const_Total_Bsmt_SF - 1.000712e+06 t566_leaf3 >= -1e+06
 T566_R3_f24_ge: x_Half_Bath - t566_leaf3 >= 0
 T566_R4_f6_ge: const_Year_Remod_Add - 1.001958e+06 t566_leaf4 >= -1e+06
 T566_L4_f4_le: const_Overall_Cond + 999996 t566_leaf4 <= 1e+06
 T566_R5_f6_ge: const_Year_Remod_Add - 1.001958e+06 t566_leaf5 >= -1e+06
 T566_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t566_leaf5 >= -1e+06
 TREE_567_ONEHOT: t567_leaf0 + t567_leaf1 + t567_leaf2 + t567_leaf3
   + t567_leaf4 + t567_leaf5 = 1
 T567_L0_f4_le: const_Overall_Cond + 999996 t567_leaf0 <= 1e+06
 T567_L0_f6_le: const_Year_Remod_Add + 998049 t567_leaf0 <= 1e+06
 T567_L1_f4_le: const_Overall_Cond + 999996 t567_leaf1 <= 1e+06
 T567_R1_f6_ge: const_Year_Remod_Add - 1.001951e+06 t567_leaf1 >= -1e+06
 T567_L1_f183_le: x_ex2_is_HdBoard <= 1
 T567_L2_f4_le: const_Overall_Cond + 999996 t567_leaf2 <= 1e+06
 T567_R2_f6_ge: const_Year_Remod_Add - 1.001951e+06 t567_leaf2 >= -1e+06
 T567_R2_f183_ge: x_ex2_is_HdBoard - t567_leaf2 >= 0
 T567_R3_f4_ge: const_Overall_Cond - 1.000004e+06 t567_leaf3 >= -1e+06
 T567_L3_f8_le: x_Exter_Qual + t567_leaf3 <= 4
 T567_L3_f72_le: const_Alley_Pave + 999999 t567_leaf3 <= 1e+06
 T567_R4_f4_ge: const_Overall_Cond - 1.000004e+06 t567_leaf4 >= -1e+06
 T567_L4_f8_le: x_Exter_Qual + t567_leaf4 <= 4
 T567_R4_f72_ge: const_Alley_Pave - 1.000001e+06 t567_leaf4 >= -1e+06
 T567_R5_f4_ge: const_Overall_Cond - 1.000004e+06 t567_leaf5 >= -1e+06
 T567_R5_f8_ge: x_Exter_Qual - 4 t567_leaf5 >= -1
 TREE_568_ONEHOT: t568_leaf0 + t568_leaf1 + t568_leaf2 + t568_leaf3 = 1
 T568_L0_f28_le: x_TotRms_AbvGrd + 4 t568_leaf0 <= 15
 T568_L0_f29_le: const_Fireplaces + 999998 t568_leaf0 <= 1e+06
 T568_L1_f28_le: x_TotRms_AbvGrd + 4 t568_leaf1 <= 15
 T568_R1_f29_ge: const_Fireplaces - 1.000002e+06 t568_leaf1 >= -1e+06
 T568_R2_f28_ge: x_TotRms_AbvGrd - 11 t568_leaf2 >= 0
 T568_L2_f14_le: x_Bsmt_Unf_SF + 98912 t568_leaf2 <= 100000
 T568_R3_f28_ge: x_TotRms_AbvGrd - 11 t568_leaf3 >= 0
 T568_R3_f14_ge: x_Bsmt_Unf_SF - 1088 t568_leaf3 >= 0
 TREE_569_ONEHOT: t569_leaf0 = 1
 TREE_570_ONEHOT: t570_leaf0 = 1
 TREE_571_ONEHOT: t571_leaf0 + t571_leaf1 + t571_leaf2 + t571_leaf3 = 1
 T571_L0_f15_le: const_Total_Bsmt_SF + 997864 t571_leaf0 <= 1e+06
 T571_L0_f272_le: const_Mo_Sold_2 + 999999 t571_leaf0 <= 1e+06
 T571_L1_f15_le: const_Total_Bsmt_SF + 997864 t571_leaf1 <= 1e+06
 T571_R1_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t571_leaf1 >= -1e+06
 T571_L1_f33_le: x_Garage_Area + 99526 t571_leaf1 <= 100000
 T571_L2_f15_le: const_Total_Bsmt_SF + 997864 t571_leaf2 <= 1e+06
 T571_R2_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t571_leaf2 >= -1e+06
 T571_R2_f33_ge: x_Garage_Area - 474 t571_leaf2 >= 0
 T571_R3_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t571_leaf3 >= -1e+06
 TREE_572_ONEHOT: t572_leaf0 + t572_leaf1 + t572_leaf2 = 1
 T572_L0_f5_le: const_Year_Built + 997992 t572_leaf0 <= 1e+06
 T572_L0_f1_le: const_Lot_Area + 972303 t572_leaf0 <= 1e+06
 T572_L1_f5_le: const_Year_Built + 997992 t572_leaf1 <= 1e+06
 T572_R1_f1_ge: const_Lot_Area - 1.027697e+06 t572_leaf1 >= -1e+06
 T572_R2_f5_ge: const_Year_Built - 1.002008e+06 t572_leaf2 >= -1e+06
 TREE_573_ONEHOT: t573_leaf0 = 1
 TREE_574_ONEHOT: t574_leaf0 + t574_leaf1 + t574_leaf2 + t574_leaf3 = 1
 T574_L0_f12_le: x_BsmtFin_SF_1 + 998741 t574_leaf0 <= 1e+06
 T574_R1_f12_ge: x_BsmtFin_SF_1 - 1259 t574_leaf1 >= 0
 T574_L1_f14_le: x_Bsmt_Unf_SF + 99552 t574_leaf1 <= 100000
 T574_L1_f12_le: x_BsmtFin_SF_1 + 998035 t574_leaf1 <= 1e+06
 T574_R2_f12_ge: x_BsmtFin_SF_1 - 1259 t574_leaf2 >= 0
 T574_L2_f14_le: x_Bsmt_Unf_SF + 99552 t574_leaf2 <= 100000
 T574_R2_f12_ge: x_BsmtFin_SF_1 - 1965 t574_leaf2 >= 0
 T574_R3_f12_ge: x_BsmtFin_SF_1 - 1259 t574_leaf3 >= 0
 T574_R3_f14_ge: x_Bsmt_Unf_SF - 448 t574_leaf3 >= 0
 TREE_575_ONEHOT: t575_leaf0 + t575_leaf1 = 1
 T575_L0_f214_le: x_b1_is_LwQ <= 1
 T575_R1_f214_ge: x_b1_is_LwQ - t575_leaf1 >= 0
 TREE_576_ONEHOT: t576_leaf0 + t576_leaf1 + t576_leaf2 + t576_leaf3
   + t576_leaf4 + t576_leaf5 = 1
 T576_L0_f3_le: Overall_Qual_calc + 999995 t576_leaf0 <= 1e+06
 T576_L0_f6_le: const_Year_Remod_Add + 998046 t576_leaf0 <= 1e+06
 T576_L1_f3_le: Overall_Qual_calc + 999995 t576_leaf1 <= 1e+06
 T576_R1_f6_ge: const_Year_Remod_Add - 1.001954e+06 t576_leaf1 >= -1e+06
 T576_L1_f37_le: x_Open_Porch_SF + 99965 t576_leaf1 <= 100000
 T576_L2_f3_le: Overall_Qual_calc + 999995 t576_leaf2 <= 1e+06
 T576_R2_f6_ge: const_Year_Remod_Add - 1.001954e+06 t576_leaf2 >= -1e+06
 T576_R2_f37_ge: x_Open_Porch_SF - 35 t576_leaf2 >= 0
 T576_R3_f3_ge: Overall_Qual_calc - 1.000005e+06 t576_leaf3 >= -1e+06
 T576_L3_f33_le: x_Garage_Area + 99456 t576_leaf3 <= 100000
 T576_R4_f3_ge: Overall_Qual_calc - 1.000005e+06 t576_leaf4 >= -1e+06
 T576_R4_f33_ge: x_Garage_Area - 544 t576_leaf4 >= 0
 T576_L4_f33_le: x_Garage_Area + 99364 t576_leaf4 <= 100000
 T576_R5_f3_ge: Overall_Qual_calc - 1.000005e+06 t576_leaf5 >= -1e+06
 T576_R5_f33_ge: x_Garage_Area - 544 t576_leaf5 >= 0
 T576_R5_f33_ge: x_Garage_Area - 636 t576_leaf5 >= 0
 TREE_577_ONEHOT: t577_leaf0 = 1
 TREE_578_ONEHOT: t578_leaf0 + t578_leaf1 + t578_leaf2 = 1
 T578_L0_f273_le: const_Mo_Sold_3 + 999999 t578_leaf0 <= 1e+06
 T578_L0_f40_le: x_Screen_Porch + 99846 t578_leaf0 <= 100000
 T578_L1_f273_le: const_Mo_Sold_3 + 999999 t578_leaf1 <= 1e+06
 T578_R1_f40_ge: x_Screen_Porch - 154 t578_leaf1 >= 0
 T578_R2_f273_ge: const_Mo_Sold_3 - 1.000001e+06 t578_leaf2 >= -1e+06
 TREE_579_ONEHOT: t579_leaf0 + t579_leaf1 = 1
 T579_L0_f18_le: const_2nd_Flr_SF + 999166 t579_leaf0 <= 1e+06
 T579_R1_f18_ge: const_2nd_Flr_SF - 1.000834e+06 t579_leaf1 >= -1e+06
 TREE_580_ONEHOT: t580_leaf0 + t580_leaf1 + t580_leaf2 + t580_leaf3
   + t580_leaf4 + t580_leaf5 = 1
 T580_L0_f216_le: x_b1_is_Rec <= 1
 T580_L0_f29_le: const_Fireplaces + 999997 t580_leaf0 <= 1e+06
 T580_L0_f7_le: x_Mas_Vnr_Area + 1433 t580_leaf0 <= 2000
 T580_L1_f216_le: x_b1_is_Rec <= 1
 T580_L1_f29_le: const_Fireplaces + 999997 t580_leaf1 <= 1e+06
 T580_R1_f7_ge: x_Mas_Vnr_Area - 567 t580_leaf1 >= 0
 T580_L2_f216_le: x_b1_is_Rec <= 1
 T580_R2_f29_ge: const_Fireplaces - 1.000003e+06 t580_leaf2 >= -1e+06
 T580_R3_f216_ge: x_b1_is_Rec - t580_leaf3 >= 0
 T580_L3_f12_le: x_BsmtFin_SF_1 + 999610 t580_leaf3 <= 1e+06
 T580_R4_f216_ge: x_b1_is_Rec - t580_leaf4 >= 0
 T580_R4_f12_ge: x_BsmtFin_SF_1 - 390 t580_leaf4 >= 0
 T580_L4_f12_le: x_BsmtFin_SF_1 + 999412 t580_leaf4 <= 1e+06
 T580_R5_f216_ge: x_b1_is_Rec - t580_leaf5 >= 0
 T580_R5_f12_ge: x_BsmtFin_SF_1 - 390 t580_leaf5 >= 0
 T580_R5_f12_ge: x_BsmtFin_SF_1 - 588 t580_leaf5 >= 0
 TREE_581_ONEHOT: t581_leaf0 + t581_leaf1 = 1
 T581_L0_f33_le: x_Garage_Area + 98846 t581_leaf0 <= 100000
 T581_R1_f33_ge: x_Garage_Area - 1154 t581_leaf1 >= 0
 TREE_582_ONEHOT: t582_leaf0 + t582_leaf1 + t582_leaf2 = 1
 T582_L0_f3_le: Overall_Qual_calc + 999990 t582_leaf0 <= 1e+06
 T582_L0_f16_le: x_Heating_QC + 2 t582_leaf0 <= 4
 T582_L1_f3_le: Overall_Qual_calc + 999990 t582_leaf1 <= 1e+06
 T582_R1_f16_ge: x_Heating_QC - 3 t582_leaf1 >= -1
 T582_R2_f3_ge: Overall_Qual_calc - 1.00001e+06 t582_leaf2 >= -1e+06
 TREE_583_ONEHOT: t583_leaf0 = 1
 TREE_584_ONEHOT: t584_leaf0 + t584_leaf1 + t584_leaf2 + t584_leaf3
   + t584_leaf4 = 1
 T584_L0_f5_le: const_Year_Built + 998086 t584_leaf0 <= 1e+06
 T584_R1_f5_ge: const_Year_Built - 1.001914e+06 t584_leaf1 >= -1e+06
 T584_L1_f18_le: const_2nd_Flr_SF + 999012 t584_leaf1 <= 1e+06
 T584_L1_f10_le: const_Bsmt_Qual + 999998 t584_leaf1 <= 1e+06
 T584_R2_f5_ge: const_Year_Built - 1.001914e+06 t584_leaf2 >= -1e+06
 T584_L2_f18_le: const_2nd_Flr_SF + 999012 t584_leaf2 <= 1e+06
 T584_R2_f10_ge: const_Bsmt_Qual - 1.000002e+06 t584_leaf2 >= -1e+06
 T584_R3_f5_ge: const_Year_Built - 1.001914e+06 t584_leaf3 >= -1e+06
 T584_R3_f18_ge: const_2nd_Flr_SF - 1.000988e+06 t584_leaf3 >= -1e+06
 T584_L3_f33_le: x_Garage_Area + 99378 t584_leaf3 <= 100000
 T584_R4_f5_ge: const_Year_Built - 1.001914e+06 t584_leaf4 >= -1e+06
 T584_R4_f18_ge: const_2nd_Flr_SF - 1.000988e+06 t584_leaf4 >= -1e+06
 T584_R4_f33_ge: x_Garage_Area - 622 t584_leaf4 >= 0
 TREE_585_ONEHOT: t585_leaf0 + t585_leaf1 + t585_leaf2 + t585_leaf3
   + t585_leaf4 = 1
 T585_L0_f0_le: const_Lot_Frontage + 999883 t585_leaf0 <= 1e+06
 T585_L0_f25_le: x_Bedroom_AbvGr + t585_leaf0 <= 6
 T585_L0_f272_le: const_Mo_Sold_2 + 999999 t585_leaf0 <= 1e+06
 T585_L1_f0_le: const_Lot_Frontage + 999883 t585_leaf1 <= 1e+06
 T585_L1_f25_le: x_Bedroom_AbvGr + t585_leaf1 <= 6
 T585_R1_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t585_leaf1 >= -1e+06
 T585_L2_f0_le: const_Lot_Frontage + 999883 t585_leaf2 <= 1e+06
 T585_R2_f25_ge: x_Bedroom_AbvGr - 5 t585_leaf2 >= 0
 T585_R3_f0_ge: const_Lot_Frontage - 1.000117e+06 t585_leaf3 >= -1e+06
 T585_L3_f297_le: const_Sale_Condition_Normal + 999999 t585_leaf3 <= 1e+06
 T585_R4_f0_ge: const_Lot_Frontage - 1.000117e+06 t585_leaf4 >= -1e+06
 T585_R4_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t585_leaf4
   >= -1e+06
 TREE_586_ONEHOT: t586_leaf0 + t586_leaf1 + t586_leaf2 + t586_leaf3
   + t586_leaf4 = 1
 T586_L0_f3_le: Overall_Qual_calc + 999997 t586_leaf0 <= 1e+06
 T586_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t586_leaf1 >= -1e+06
 T586_L1_f3_le: Overall_Qual_calc + 999993 t586_leaf1 <= 1e+06
 T586_L1_f20_le: x_Gr_Liv_Area + 8299 t586_leaf1 <= 10000
 T586_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t586_leaf2 >= -1e+06
 T586_L2_f3_le: Overall_Qual_calc + 999993 t586_leaf2 <= 1e+06
 T586_R2_f20_ge: x_Gr_Liv_Area - 1701 t586_leaf2 >= 0
 T586_R3_f3_ge: Overall_Qual_calc - 1.000003e+06 t586_leaf3 >= -1e+06
 T586_R3_f3_ge: Overall_Qual_calc - 1.000007e+06 t586_leaf3 >= -1e+06
 T586_L3_f105_le: const_Neighborhood_NWAmes + 999999 t586_leaf3 <= 1e+06
 T586_R4_f3_ge: Overall_Qual_calc - 1.000003e+06 t586_leaf4 >= -1e+06
 T586_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t586_leaf4 >= -1e+06
 T586_R4_f105_ge: const_Neighborhood_NWAmes - 1.000001e+06 t586_leaf4
   >= -1e+06
 TREE_587_ONEHOT: t587_leaf0 + t587_leaf1 + t587_leaf2 + t587_leaf3
   + t587_leaf4 + t587_leaf5 = 1
 T587_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t587_leaf0
   <= 1e+06
 T587_L0_f12_le: x_BsmtFin_SF_1 + 998526 t587_leaf0 <= 1e+06
 T587_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t587_leaf1
   <= 1e+06
 T587_R1_f12_ge: x_BsmtFin_SF_1 - 1474 t587_leaf1 >= 0
 T587_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t587_leaf2
   >= -1e+06
 T587_L2_f5_le: const_Year_Built + 998046 t587_leaf2 <= 1e+06
 T587_L2_f148_le: const_Roof_Style_Gable + 999999 t587_leaf2 <= 1e+06
 T587_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t587_leaf3
   >= -1e+06
 T587_L3_f5_le: const_Year_Built + 998046 t587_leaf3 <= 1e+06
 T587_R3_f148_ge: const_Roof_Style_Gable - 1.000001e+06 t587_leaf3
   >= -1e+06
 T587_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t587_leaf4
   >= -1e+06
 T587_R4_f5_ge: const_Year_Built - 1.001954e+06 t587_leaf4 >= -1e+06
 T587_L4_f44_le: const_Yr_Sold + 997991 t587_leaf4 <= 1e+06
 T587_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t587_leaf5
   >= -1e+06
 T587_R5_f5_ge: const_Year_Built - 1.001954e+06 t587_leaf5 >= -1e+06
 T587_R5_f44_ge: const_Yr_Sold - 1.002009e+06 t587_leaf5 >= -1e+06
 TREE_588_ONEHOT: t588_leaf0 + t588_leaf1 + t588_leaf2 + t588_leaf3
   + t588_leaf4 + t588_leaf5 + t588_leaf6 + t588_leaf7 = 1
 T588_L0_f3_le: Overall_Qual_calc + 999993 t588_leaf0 <= 1e+06
 T588_L0_f1_le: const_Lot_Area + 991480 t588_leaf0 <= 1e+06
 T588_L0_f33_le: x_Garage_Area + 99785 t588_leaf0 <= 100000
 T588_L1_f3_le: Overall_Qual_calc + 999993 t588_leaf1 <= 1e+06
 T588_L1_f1_le: const_Lot_Area + 991480 t588_leaf1 <= 1e+06
 T588_R1_f33_ge: x_Garage_Area - 215 t588_leaf1 >= 0
 T588_L2_f3_le: Overall_Qual_calc + 999993 t588_leaf2 <= 1e+06
 T588_R2_f1_ge: const_Lot_Area - 1.00852e+06 t588_leaf2 >= -1e+06
 T588_L2_f231_le: central_air_no + 999999 t588_leaf2 <= 1e+06
 T588_L3_f3_le: Overall_Qual_calc + 999993 t588_leaf3 <= 1e+06
 T588_R3_f1_ge: const_Lot_Area - 1.00852e+06 t588_leaf3 >= -1e+06
 T588_R3_f231_ge: central_air_no - 1.000001e+06 t588_leaf3 >= -1e+06
 T588_R4_f3_ge: Overall_Qual_calc - 1.000007e+06 t588_leaf4 >= -1e+06
 T588_L4_f6_le: const_Year_Remod_Add + 998007 t588_leaf4 <= 1e+06
 T588_L4_f297_le: const_Sale_Condition_Normal + 999999 t588_leaf4 <= 1e+06
 T588_R5_f3_ge: Overall_Qual_calc - 1.000007e+06 t588_leaf5 >= -1e+06
 T588_L5_f6_le: const_Year_Remod_Add + 998007 t588_leaf5 <= 1e+06
 T588_R5_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t588_leaf5
   >= -1e+06
 T588_R6_f3_ge: Overall_Qual_calc - 1.000007e+06 t588_leaf6 >= -1e+06
 T588_R6_f6_ge: const_Year_Remod_Add - 1.001993e+06 t588_leaf6 >= -1e+06
 T588_L6_f96_le: const_Neighborhood_Edwards + 999999 t588_leaf6 <= 1e+06
 T588_R7_f3_ge: Overall_Qual_calc - 1.000007e+06 t588_leaf7 >= -1e+06
 T588_R7_f6_ge: const_Year_Remod_Add - 1.001993e+06 t588_leaf7 >= -1e+06
 T588_R7_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t588_leaf7
   >= -1e+06
 TREE_589_ONEHOT: t589_leaf0 = 1
 TREE_590_ONEHOT: t590_leaf0 + t590_leaf1 = 1
 T590_L0_f20_le: x_Gr_Liv_Area + 7367 t590_leaf0 <= 10000
 T590_R1_f20_ge: x_Gr_Liv_Area - 2633 t590_leaf1 >= 0
 TREE_591_ONEHOT: t591_leaf0 = 1
 TREE_592_ONEHOT: t592_leaf0 = 1
 TREE_593_ONEHOT: t593_leaf0 = 1
 TREE_594_ONEHOT: t594_leaf0 = 1
 TREE_595_ONEHOT: t595_leaf0 + t595_leaf1 + t595_leaf2 + t595_leaf3
   + t595_leaf4 = 1
 T595_L0_f5_le: const_Year_Built + 998040 t595_leaf0 <= 1e+06
 T595_L0_f18_le: const_2nd_Flr_SF + 999528 t595_leaf0 <= 1e+06
 T595_L1_f5_le: const_Year_Built + 998040 t595_leaf1 <= 1e+06
 T595_R1_f18_ge: const_2nd_Flr_SF - 1.000472e+06 t595_leaf1 >= -1e+06
 T595_L1_f14_le: x_Bsmt_Unf_SF + 99810 t595_leaf1 <= 100000
 T595_L2_f5_le: const_Year_Built + 998040 t595_leaf2 <= 1e+06
 T595_R2_f18_ge: const_2nd_Flr_SF - 1.000472e+06 t595_leaf2 >= -1e+06
 T595_R2_f14_ge: x_Bsmt_Unf_SF - 190 t595_leaf2 >= 0
 T595_R3_f5_ge: const_Year_Built - 1.00196e+06 t595_leaf3 >= -1e+06
 T595_L3_f0_le: const_Lot_Frontage + 999885 t595_leaf3 <= 1e+06
 T595_R4_f5_ge: const_Year_Built - 1.00196e+06 t595_leaf4 >= -1e+06
 T595_R4_f0_ge: const_Lot_Frontage - 1.000115e+06 t595_leaf4 >= -1e+06
 TREE_596_ONEHOT: t596_leaf0 = 1
 TREE_597_ONEHOT: t597_leaf0 + t597_leaf1 + t597_leaf2 + t597_leaf3
   + t597_leaf4 = 1
 T597_L0_f194_le: x_mvt_is_BrkCmn <= 1
 T597_L0_f30_le: x_Fireplace_Qu + 2 t597_leaf0 <= 4
 T597_L0_f20_le: x_Gr_Liv_Area + 7763 t597_leaf0 <= 10000
 T597_L1_f194_le: x_mvt_is_BrkCmn <= 1
 T597_L1_f30_le: x_Fireplace_Qu + 2 t597_leaf1 <= 4
 T597_R1_f20_ge: x_Gr_Liv_Area - 2237 t597_leaf1 >= 0
 T597_L2_f194_le: x_mvt_is_BrkCmn <= 1
 T597_R2_f30_ge: x_Fireplace_Qu - 3 t597_leaf2 >= -1
 T597_L2_f21_le: const_Bsmt_Full_Bath + 999998 t597_leaf2 <= 1e+06
 T597_L3_f194_le: x_mvt_is_BrkCmn <= 1
 T597_R3_f30_ge: x_Fireplace_Qu - 3 t597_leaf3 >= -1
 T597_R3_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t597_leaf3 >= -1e+06
 T597_R4_f194_ge: x_mvt_is_BrkCmn - t597_leaf4 >= 0
 TREE_598_ONEHOT: t598_leaf0 + t598_leaf1 = 1
 T598_L0_f5_le: const_Year_Built + 998085 t598_leaf0 <= 1e+06
 T598_R1_f5_ge: const_Year_Built - 1.001915e+06 t598_leaf1 >= -1e+06
 TREE_599_ONEHOT: t599_leaf0 = 1
 TREE_600_ONEHOT: t600_leaf0 + t600_leaf1 + t600_leaf2 + t600_leaf3 = 1
 T600_L0_f216_le: x_b1_is_Rec <= 1
 T600_R1_f216_ge: x_b1_is_Rec - t600_leaf1 >= 0
 T600_L1_f192_le: x_ex2_is_Wd_Sdng <= 1
 T600_L1_f31_le: const_Garage_Yr_Blt + 998034 t600_leaf1 <= 1e+06
 T600_R2_f216_ge: x_b1_is_Rec - t600_leaf2 >= 0
 T600_L2_f192_le: x_ex2_is_Wd_Sdng <= 1
 T600_R2_f31_ge: const_Garage_Yr_Blt - 1.001966e+06 t600_leaf2 >= -1e+06
 T600_R3_f216_ge: x_b1_is_Rec - t600_leaf3 >= 0
 T600_R3_f192_ge: x_ex2_is_Wd_Sdng - t600_leaf3 >= 0
 TREE_601_ONEHOT: t601_leaf0 + t601_leaf1 + t601_leaf2 + t601_leaf3 = 1
 T601_L0_f216_le: x_b1_is_Rec <= 1
 T601_R1_f216_ge: x_b1_is_Rec - t601_leaf1 >= 0
 T601_L1_f12_le: x_BsmtFin_SF_1 + 999634 t601_leaf1 <= 1e+06
 T601_R2_f216_ge: x_b1_is_Rec - t601_leaf2 >= 0
 T601_R2_f12_ge: x_BsmtFin_SF_1 - 366 t601_leaf2 >= 0
 T601_L2_f20_le: x_Gr_Liv_Area + 9035 t601_leaf2 <= 10000
 T601_R3_f216_ge: x_b1_is_Rec - t601_leaf3 >= 0
 T601_R3_f12_ge: x_BsmtFin_SF_1 - 366 t601_leaf3 >= 0
 T601_R3_f20_ge: x_Gr_Liv_Area - 965 t601_leaf3 >= 0
 TREE_602_ONEHOT: t602_leaf0 = 1
 TREE_603_ONEHOT: t603_leaf0 + t603_leaf1 = 1
 T603_L0_f8_le: x_Exter_Qual + t603_leaf0 <= 4
 T603_R1_f8_ge: x_Exter_Qual - 4 t603_leaf1 >= -1
 TREE_604_ONEHOT: t604_leaf0 = 1
 TREE_605_ONEHOT: t605_leaf0 + t605_leaf1 + t605_leaf2 + t605_leaf3 = 1
 T605_L0_f17_le: x_1st_Flr_SF + 2485 t605_leaf0 <= 5000
 T605_L0_f0_le: const_Lot_Frontage + 999890 t605_leaf0 <= 1e+06
 T605_L0_f32_le: const_Garage_Cars + 999997 t605_leaf0 <= 1e+06
 T605_L1_f17_le: x_1st_Flr_SF + 2485 t605_leaf1 <= 5000
 T605_L1_f0_le: const_Lot_Frontage + 999890 t605_leaf1 <= 1e+06
 T605_R1_f32_ge: const_Garage_Cars - 1.000003e+06 t605_leaf1 >= -1e+06
 T605_L2_f17_le: x_1st_Flr_SF + 2485 t605_leaf2 <= 5000
 T605_R2_f0_ge: const_Lot_Frontage - 1.00011e+06 t605_leaf2 >= -1e+06
 T605_R3_f17_ge: x_1st_Flr_SF - 2515 t605_leaf3 >= 0
 TREE_606_ONEHOT: t606_leaf0 = 1
 TREE_607_ONEHOT: t607_leaf0 + t607_leaf1 = 1
 T607_L0_f158_le: x_roof_matl_is_Tar&Grv <= 1
 T607_R1_f158_ge: x_roof_matl_is_Tar&Grv - t607_leaf1 >= 0
 TREE_608_ONEHOT: t608_leaf0 + t608_leaf1 = 1
 T608_L0_f60_le: const_MS_SubClass_190 + 999999 t608_leaf0 <= 1e+06
 T608_R1_f60_ge: const_MS_SubClass_190 - 1.000001e+06 t608_leaf1 >= -1e+06
 TREE_609_ONEHOT: t609_leaf0 = 1
 TREE_610_ONEHOT: t610_leaf0 = 1
 TREE_611_ONEHOT: t611_leaf0 + t611_leaf1 = 1
 T611_L0_f274_le: const_Mo_Sold_4 + 999999 t611_leaf0 <= 1e+06
 T611_R1_f274_ge: const_Mo_Sold_4 - 1.000001e+06 t611_leaf1 >= -1e+06
 TREE_612_ONEHOT: t612_leaf0 = 1
 TREE_613_ONEHOT: t613_leaf0 + t613_leaf1 = 1
 T613_L0_f24_le: x_Half_Bath + 4 t613_leaf0 <= 5
 T613_R1_f24_ge: x_Half_Bath - t613_leaf1 >= 0
 TREE_614_ONEHOT: t614_leaf0 = 1
 TREE_615_ONEHOT: t615_leaf0 = 1
 TREE_616_ONEHOT: t616_leaf0 + t616_leaf1 + t616_leaf2 + t616_leaf3 = 1
 T616_L0_f85_le: const_Lot_Config_Inside + 999999 t616_leaf0 <= 1e+06
 T616_R1_f85_ge: const_Lot_Config_Inside - 1.000001e+06 t616_leaf1
   >= -1e+06
 T616_L1_f1_le: const_Lot_Area + 989200 t616_leaf1 <= 1e+06
 T616_R2_f85_ge: const_Lot_Config_Inside - 1.000001e+06 t616_leaf2
   >= -1e+06
 T616_R2_f1_ge: const_Lot_Area - 1.0108e+06 t616_leaf2 >= -1e+06
 T616_L2_f33_le: x_Garage_Area + 99634 t616_leaf2 <= 100000
 T616_R3_f85_ge: const_Lot_Config_Inside - 1.000001e+06 t616_leaf3
   >= -1e+06
 T616_R3_f1_ge: const_Lot_Area - 1.0108e+06 t616_leaf3 >= -1e+06
 T616_R3_f33_ge: x_Garage_Area - 366 t616_leaf3 >= 0
 TREE_617_ONEHOT: t617_leaf0 + t617_leaf1 + t617_leaf2 = 1
 T617_L0_f3_le: Overall_Qual_calc + 999995 t617_leaf0 <= 1e+06
 T617_L0_f80_le: const_Land_Contour_Lvl + 999999 t617_leaf0 <= 1e+06
 T617_L1_f3_le: Overall_Qual_calc + 999995 t617_leaf1 <= 1e+06
 T617_R1_f80_ge: const_Land_Contour_Lvl - 1.000001e+06 t617_leaf1
   >= -1e+06
 T617_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t617_leaf2 >= -1e+06
 TREE_618_ONEHOT: t618_leaf0 + t618_leaf1 = 1
 T618_L0_f238_le: const_FunctioNo_aplical_Maj1 + 999999 t618_leaf0
   <= 1e+06
 T618_R1_f238_ge: const_FunctioNo_aplical_Maj1 - 1.000001e+06 t618_leaf1
   >= -1e+06
 TREE_619_ONEHOT: t619_leaf0 + t619_leaf1 + t619_leaf2 + t619_leaf3
   + t619_leaf4 = 1
 T619_L0_f29_le: const_Fireplaces + 999999 t619_leaf0 <= 1e+06
 T619_L0_f33_le: x_Garage_Area + 99376 t619_leaf0 <= 100000
 T619_L1_f29_le: const_Fireplaces + 999999 t619_leaf1 <= 1e+06
 T619_R1_f33_ge: x_Garage_Area - 624 t619_leaf1 >= 0
 T619_R2_f29_ge: const_Fireplaces - 1.000001e+06 t619_leaf2 >= -1e+06
 T619_L2_f4_le: const_Overall_Cond + 999995 t619_leaf2 <= 1e+06
 T619_L2_f33_le: x_Garage_Area + 99356 t619_leaf2 <= 100000
 T619_R3_f29_ge: const_Fireplaces - 1.000001e+06 t619_leaf3 >= -1e+06
 T619_L3_f4_le: const_Overall_Cond + 999995 t619_leaf3 <= 1e+06
 T619_R3_f33_ge: x_Garage_Area - 644 t619_leaf3 >= 0
 T619_R4_f29_ge: const_Fireplaces - 1.000001e+06 t619_leaf4 >= -1e+06
 T619_R4_f4_ge: const_Overall_Cond - 1.000005e+06 t619_leaf4 >= -1e+06
 TREE_620_ONEHOT: t620_leaf0 + t620_leaf1 = 1
 T620_L0_f7_le: x_Mas_Vnr_Area + 1327 t620_leaf0 <= 2000
 T620_R1_f7_ge: x_Mas_Vnr_Area - 673 t620_leaf1 >= 0
 TREE_621_ONEHOT: t621_leaf0 + t621_leaf1 + t621_leaf2 + t621_leaf3 = 1
 T621_L0_f5_le: const_Year_Built + 997995 t621_leaf0 <= 1e+06
 T621_R1_f5_ge: const_Year_Built - 1.002005e+06 t621_leaf1 >= -1e+06
 T621_L1_f1_le: const_Lot_Area + 991842 t621_leaf1 <= 1e+06
 T621_L1_f0_le: const_Lot_Frontage + 999955 t621_leaf1 <= 1e+06
 T621_R2_f5_ge: const_Year_Built - 1.002005e+06 t621_leaf2 >= -1e+06
 T621_L2_f1_le: const_Lot_Area + 991842 t621_leaf2 <= 1e+06
 T621_R2_f0_ge: const_Lot_Frontage - 1.000045e+06 t621_leaf2 >= -1e+06
 T621_R3_f5_ge: const_Year_Built - 1.002005e+06 t621_leaf3 >= -1e+06
 T621_R3_f1_ge: const_Lot_Area - 1.008158e+06 t621_leaf3 >= -1e+06
 TREE_622_ONEHOT: t622_leaf0 + t622_leaf1 + t622_leaf2 + t622_leaf3 = 1
 T622_L0_f20_le: x_Gr_Liv_Area + 7351 t622_leaf0 <= 10000
 T622_L0_f29_le: const_Fireplaces + 999999 t622_leaf0 <= 1e+06
 T622_L1_f20_le: x_Gr_Liv_Area + 7351 t622_leaf1 <= 10000
 T622_R1_f29_ge: const_Fireplaces - 1.000001e+06 t622_leaf1 >= -1e+06
 T622_L1_f28_le: x_TotRms_AbvGrd + 10 t622_leaf1 <= 15
 T622_L2_f20_le: x_Gr_Liv_Area + 7351 t622_leaf2 <= 10000
 T622_R2_f29_ge: const_Fireplaces - 1.000001e+06 t622_leaf2 >= -1e+06
 T622_R2_f28_ge: x_TotRms_AbvGrd - 5 t622_leaf2 >= 0
 T622_R3_f20_ge: x_Gr_Liv_Area - 2649 t622_leaf3 >= 0
 TREE_623_ONEHOT: t623_leaf0 + t623_leaf1 + t623_leaf2 + t623_leaf3
   + t623_leaf4 + t623_leaf5 = 1
 T623_L0_f44_le: const_Yr_Sold + 997991 t623_leaf0 <= 1e+06
 T623_L0_f77_le: const_Land_Contour_Bnk + 999999 t623_leaf0 <= 1e+06
 T623_L1_f44_le: const_Yr_Sold + 997991 t623_leaf1 <= 1e+06
 T623_R1_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t623_leaf1
   >= -1e+06
 T623_L1_f33_le: x_Garage_Area + 99423 t623_leaf1 <= 100000
 T623_L2_f44_le: const_Yr_Sold + 997991 t623_leaf2 <= 1e+06
 T623_R2_f77_ge: const_Land_Contour_Bnk - 1.000001e+06 t623_leaf2
   >= -1e+06
 T623_R2_f33_ge: x_Garage_Area - 577 t623_leaf2 >= 0
 T623_R3_f44_ge: const_Yr_Sold - 1.002009e+06 t623_leaf3 >= -1e+06
 T623_L3_f293_le: const_Sale_Condition_Abnorml + 999999 t623_leaf3
   <= 1e+06
 T623_R4_f44_ge: const_Yr_Sold - 1.002009e+06 t623_leaf4 >= -1e+06
 T623_R4_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t623_leaf4
   >= -1e+06
 T623_L4_f81_le: const_Lot_Config_Corner + 999999 t623_leaf4 <= 1e+06
 T623_R5_f44_ge: const_Yr_Sold - 1.002009e+06 t623_leaf5 >= -1e+06
 T623_R5_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t623_leaf5
   >= -1e+06
 T623_R5_f81_ge: const_Lot_Config_Corner - 1.000001e+06 t623_leaf5
   >= -1e+06
 TREE_624_ONEHOT: t624_leaf0 + t624_leaf1 = 1
 T624_L0_f79_le: const_Land_Contour_Low + 999999 t624_leaf0 <= 1e+06
 T624_R1_f79_ge: const_Land_Contour_Low - 1.000001e+06 t624_leaf1
   >= -1e+06
 TREE_625_ONEHOT: t625_leaf0 + t625_leaf1 + t625_leaf2 = 1
 T625_L0_f15_le: const_Total_Bsmt_SF + 999755 t625_leaf0 <= 1e+06
 T625_L0_f0_le: const_Lot_Frontage + 999934 t625_leaf0 <= 1e+06
 T625_L1_f15_le: const_Total_Bsmt_SF + 999755 t625_leaf1 <= 1e+06
 T625_R1_f0_ge: const_Lot_Frontage - 1.000066e+06 t625_leaf1 >= -1e+06
 T625_R2_f15_ge: const_Total_Bsmt_SF - 1.000245e+06 t625_leaf2 >= -1e+06
 TREE_626_ONEHOT: t626_leaf0 + t626_leaf1 + t626_leaf2 + t626_leaf3 = 1
 T626_L0_f10_le: const_Bsmt_Qual + 999998 t626_leaf0 <= 1e+06
 T626_L0_f4_le: const_Overall_Cond + 999994 t626_leaf0 <= 1e+06
 T626_L0_f85_le: const_Lot_Config_Inside + 999999 t626_leaf0 <= 1e+06
 T626_L1_f10_le: const_Bsmt_Qual + 999998 t626_leaf1 <= 1e+06
 T626_L1_f4_le: const_Overall_Cond + 999994 t626_leaf1 <= 1e+06
 T626_R1_f85_ge: const_Lot_Config_Inside - 1.000001e+06 t626_leaf1
   >= -1e+06
 T626_L2_f10_le: const_Bsmt_Qual + 999998 t626_leaf2 <= 1e+06
 T626_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t626_leaf2 >= -1e+06
 T626_R3_f10_ge: const_Bsmt_Qual - 1.000002e+06 t626_leaf3 >= -1e+06
 TREE_627_ONEHOT: t627_leaf0 + t627_leaf1 = 1
 T627_L0_f15_le: const_Total_Bsmt_SF + 997958 t627_leaf0 <= 1e+06
 T627_R1_f15_ge: const_Total_Bsmt_SF - 1.002042e+06 t627_leaf1 >= -1e+06
 TREE_628_ONEHOT: t628_leaf0 = 1
 TREE_629_ONEHOT: t629_leaf0 = 1
 TREE_630_ONEHOT: t630_leaf0 = 1
 TREE_631_ONEHOT: t631_leaf0 = 1
 TREE_632_ONEHOT: t632_leaf0 + t632_leaf1 + t632_leaf2 + t632_leaf3 = 1
 T632_L0_f161_le: x_ex1_is_AsbShng <= 1
 T632_L0_f5_le: const_Year_Built + 998085 t632_leaf0 <= 1e+06
 T632_L1_f161_le: x_ex1_is_AsbShng <= 1
 T632_R1_f5_ge: const_Year_Built - 1.001915e+06 t632_leaf1 >= -1e+06
 T632_R2_f161_ge: x_ex1_is_AsbShng - t632_leaf2 >= 0
 T632_L2_f1_le: const_Lot_Area + 990395 t632_leaf2 <= 1e+06
 T632_R3_f161_ge: x_ex1_is_AsbShng - t632_leaf3 >= 0
 T632_R3_f1_ge: const_Lot_Area - 1.009605e+06 t632_leaf3 >= -1e+06
 TREE_633_ONEHOT: t633_leaf0 = 1
 TREE_634_ONEHOT: t634_leaf0 = 1
 TREE_635_ONEHOT: t635_leaf0 = 1
 TREE_636_ONEHOT: t636_leaf0 = 1
 TREE_637_ONEHOT: t637_leaf0 = 1
 TREE_638_ONEHOT: t638_leaf0 = 1
 TREE_639_ONEHOT: t639_leaf0 = 1
 TREE_640_ONEHOT: t640_leaf0 + t640_leaf1 + t640_leaf2 + t640_leaf3 = 1
 T640_L0_f161_le: x_ex1_is_AsbShng <= 1
 T640_L0_f4_le: const_Overall_Cond + 999996 t640_leaf0 <= 1e+06
 T640_L1_f161_le: x_ex1_is_AsbShng <= 1
 T640_R1_f4_ge: const_Overall_Cond - 1.000004e+06 t640_leaf1 >= -1e+06
 T640_R2_f161_ge: x_ex1_is_AsbShng - t640_leaf2 >= 0
 T640_L2_f20_le: x_Gr_Liv_Area + 9117 t640_leaf2 <= 10000
 T640_R3_f161_ge: x_ex1_is_AsbShng - t640_leaf3 >= 0
 T640_R3_f20_ge: x_Gr_Liv_Area - 883 t640_leaf3 >= 0
 TREE_641_ONEHOT: t641_leaf0 = 1
 TREE_642_ONEHOT: t642_leaf0 = 1
 TREE_643_ONEHOT: t643_leaf0 = 1
 TREE_644_ONEHOT: t644_leaf0 = 1
 TREE_645_ONEHOT: t645_leaf0 = 1
 TREE_646_ONEHOT: t646_leaf0 + t646_leaf1 + t646_leaf2 = 1
 T646_L0_f4_le: const_Overall_Cond + 999994 t646_leaf0 <= 1e+06
 T646_R1_f4_ge: const_Overall_Cond - 1.000006e+06 t646_leaf1 >= -1e+06
 T646_L1_f23_le: x_Full_Bath + 2 t646_leaf1 <= 4
 T646_R2_f4_ge: const_Overall_Cond - 1.000006e+06 t646_leaf2 >= -1e+06
 T646_R2_f23_ge: x_Full_Bath - 2 t646_leaf2 >= 0
 TREE_647_ONEHOT: t647_leaf0 + t647_leaf1 + t647_leaf2 = 1
 T647_L0_f207_le: const_Bsmt_Exposure_Gd + 999999 t647_leaf0 <= 1e+06
 T647_L0_f0_le: const_Lot_Frontage + 999924 t647_leaf0 <= 1e+06
 T647_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t647_leaf1 <= 1e+06
 T647_R1_f0_ge: const_Lot_Frontage - 1.000076e+06 t647_leaf1 >= -1e+06
 T647_R2_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t647_leaf2
   >= -1e+06
 TREE_648_ONEHOT: t648_leaf0 = 1
 TREE_649_ONEHOT: t649_leaf0 + t649_leaf1 = 1
 T649_L0_f219_le: x_b2_is_BLQ <= 1
 T649_R1_f219_ge: x_b2_is_BLQ - t649_leaf1 >= 0
 TREE_650_ONEHOT: t650_leaf0 = 1
 TREE_651_ONEHOT: t651_leaf0 = 1
 TREE_652_ONEHOT: t652_leaf0 + t652_leaf1 + t652_leaf2 = 1
 T652_L0_f37_le: x_Open_Porch_SF + 99712 t652_leaf0 <= 100000
 T652_L0_f0_le: const_Lot_Frontage + 999869 t652_leaf0 <= 1e+06
 T652_L1_f37_le: x_Open_Porch_SF + 99712 t652_leaf1 <= 100000
 T652_R1_f0_ge: const_Lot_Frontage - 1.000131e+06 t652_leaf1 >= -1e+06
 T652_R2_f37_ge: x_Open_Porch_SF - 288 t652_leaf2 >= 0
 TREE_653_ONEHOT: t653_leaf0 + t653_leaf1 = 1
 T653_L0_f12_le: x_BsmtFin_SF_1 + 998567 t653_leaf0 <= 1e+06
 T653_R1_f12_ge: x_BsmtFin_SF_1 - 1433 t653_leaf1 >= 0
 TREE_654_ONEHOT: t654_leaf0 = 1
 TREE_655_ONEHOT: t655_leaf0 + t655_leaf1 + t655_leaf2 = 1
 T655_L0_f62_le: const_MS_Zoning_C_(all) + 999999 t655_leaf0 <= 1e+06
 T655_R1_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t655_leaf1
   >= -1e+06
 T655_L1_f139_le: const_House_Style_1.5Fin + 999999 t655_leaf1 <= 1e+06
 T655_R2_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t655_leaf2
   >= -1e+06
 T655_R2_f139_ge: const_House_Style_1.5Fin - 1.000001e+06 t655_leaf2
   >= -1e+06
 TREE_656_ONEHOT: t656_leaf0 + t656_leaf1 + t656_leaf2 = 1
 T656_L0_f33_le: x_Garage_Area + 99170 t656_leaf0 <= 100000
 T656_L0_f19_le: const_Low_Qual_Fin_SF + 999892 t656_leaf0 <= 1e+06
 T656_L1_f33_le: x_Garage_Area + 99170 t656_leaf1 <= 100000
 T656_R1_f19_ge: const_Low_Qual_Fin_SF - 1.000108e+06 t656_leaf1 >= -1e+06
 T656_R2_f33_ge: x_Garage_Area - 830 t656_leaf2 >= 0
 TREE_657_ONEHOT: t657_leaf0 + t657_leaf1 = 1
 T657_L0_f290_le: const_Sale_Type_Oth + 999999 t657_leaf0 <= 1e+06
 T657_R1_f290_ge: const_Sale_Type_Oth - 1.000001e+06 t657_leaf1 >= -1e+06
 TREE_658_ONEHOT: t658_leaf0 = 1
 TREE_659_ONEHOT: t659_leaf0 + t659_leaf1 + t659_leaf2 + t659_leaf3
   + t659_leaf4 = 1
 T659_L0_f31_le: const_Garage_Yr_Blt + 998057 t659_leaf0 <= 1e+06
 T659_L0_f211_le: x_b1_is_ALQ <= 1
 T659_L0_f28_le: x_TotRms_AbvGrd + 7 t659_leaf0 <= 15
 T659_L1_f31_le: const_Garage_Yr_Blt + 998057 t659_leaf1 <= 1e+06
 T659_L1_f211_le: x_b1_is_ALQ <= 1
 T659_R1_f28_ge: x_TotRms_AbvGrd - 8 t659_leaf1 >= 0
 T659_L2_f31_le: const_Garage_Yr_Blt + 998057 t659_leaf2 <= 1e+06
 T659_R2_f211_ge: x_b1_is_ALQ - t659_leaf2 >= 0
 T659_R3_f31_ge: const_Garage_Yr_Blt - 1.001943e+06 t659_leaf3 >= -1e+06
 T659_L3_f34_le: x_Garage_Qual + 2 t659_leaf3 <= 4
 T659_R4_f31_ge: const_Garage_Yr_Blt - 1.001943e+06 t659_leaf4 >= -1e+06
 T659_R4_f34_ge: x_Garage_Qual - 3 t659_leaf4 >= -1
 TREE_660_ONEHOT: t660_leaf0 + t660_leaf1 + t660_leaf2 + t660_leaf3 = 1
 T660_L0_f0_le: const_Lot_Frontage + 999860 t660_leaf0 <= 1e+06
 T660_L0_f37_le: x_Open_Porch_SF + 99878 t660_leaf0 <= 100000
 T660_L0_f1_le: const_Lot_Area + 988235 t660_leaf0 <= 1e+06
 T660_L1_f0_le: const_Lot_Frontage + 999860 t660_leaf1 <= 1e+06
 T660_L1_f37_le: x_Open_Porch_SF + 99878 t660_leaf1 <= 100000
 T660_R1_f1_ge: const_Lot_Area - 1.011765e+06 t660_leaf1 >= -1e+06
 T660_L2_f0_le: const_Lot_Frontage + 999860 t660_leaf2 <= 1e+06
 T660_R2_f37_ge: x_Open_Porch_SF - 122 t660_leaf2 >= 0
 T660_R3_f0_ge: const_Lot_Frontage - 1.00014e+06 t660_leaf3 >= -1e+06
 TREE_661_ONEHOT: t661_leaf0 + t661_leaf1 + t661_leaf2 + t661_leaf3
   + t661_leaf4 = 1
 T661_L0_f17_le: x_1st_Flr_SF + 4424 t661_leaf0 <= 5000
 T661_R1_f17_ge: x_1st_Flr_SF - 576 t661_leaf1 >= 0
 T661_L1_f161_le: x_ex1_is_AsbShng <= 1
 T661_L1_f34_le: x_Garage_Qual + 2 t661_leaf1 <= 4
 T661_R2_f17_ge: x_1st_Flr_SF - 576 t661_leaf2 >= 0
 T661_L2_f161_le: x_ex1_is_AsbShng <= 1
 T661_R2_f34_ge: x_Garage_Qual - 3 t661_leaf2 >= -1
 T661_R3_f17_ge: x_1st_Flr_SF - 576 t661_leaf3 >= 0
 T661_R3_f161_ge: x_ex1_is_AsbShng - t661_leaf3 >= 0
 T661_L3_f14_le: x_Bsmt_Unf_SF + 99316 t661_leaf3 <= 100000
 T661_R4_f17_ge: x_1st_Flr_SF - 576 t661_leaf4 >= 0
 T661_R4_f161_ge: x_ex1_is_AsbShng - t661_leaf4 >= 0
 T661_R4_f14_ge: x_Bsmt_Unf_SF - 684 t661_leaf4 >= 0
 TREE_662_ONEHOT: t662_leaf0 = 1
 TREE_663_ONEHOT: t663_leaf0 + t663_leaf1 + t663_leaf2 + t663_leaf3
   + t663_leaf4 + t663_leaf5 + t663_leaf6 = 1
 T663_L0_f5_le: const_Year_Built + 998061 t663_leaf0 <= 1e+06
 T663_L0_f119_le: const_Condition_1_Norm + 999999 t663_leaf0 <= 1e+06
 T663_L1_f5_le: const_Year_Built + 998061 t663_leaf1 <= 1e+06
 T663_R1_f119_ge: const_Condition_1_Norm - 1.000001e+06 t663_leaf1
   >= -1e+06
 T663_L1_f5_le: const_Year_Built + 998085 t663_leaf1 <= 1e+06
 T663_L2_f5_le: const_Year_Built + 998061 t663_leaf2 <= 1e+06
 T663_R2_f119_ge: const_Condition_1_Norm - 1.000001e+06 t663_leaf2
   >= -1e+06
 T663_R2_f5_ge: const_Year_Built - 1.001915e+06 t663_leaf2 >= -1e+06
 T663_R3_f5_ge: const_Year_Built - 1.001939e+06 t663_leaf3 >= -1e+06
 T663_L3_f237_le: x_elect_is_SBrkr <= 1
 T663_L3_f86_le: const_Land_Slope_Gtl + 999999 t663_leaf3 <= 1e+06
 T663_R4_f5_ge: const_Year_Built - 1.001939e+06 t663_leaf4 >= -1e+06
 T663_L4_f237_le: x_elect_is_SBrkr <= 1
 T663_R4_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t663_leaf4 >= -1e+06
 T663_R5_f5_ge: const_Year_Built - 1.001939e+06 t663_leaf5 >= -1e+06
 T663_R5_f237_ge: x_elect_is_SBrkr - t663_leaf5 >= 0
 T663_L5_f231_le: central_air_no + 999999 t663_leaf5 <= 1e+06
 T663_R6_f5_ge: const_Year_Built - 1.001939e+06 t663_leaf6 >= -1e+06
 T663_R6_f237_ge: x_elect_is_SBrkr - t663_leaf6 >= 0
 T663_R6_f231_ge: central_air_no - 1.000001e+06 t663_leaf6 >= -1e+06
 TREE_664_ONEHOT: t664_leaf0 + t664_leaf1 + t664_leaf2 = 1
 T664_L0_f1_le: const_Lot_Area + 994000 t664_leaf0 <= 1e+06
 T664_R1_f1_ge: const_Lot_Area - 1.006e+06 t664_leaf1 >= -1e+06
 T664_L1_f3_le: Overall_Qual_calc + 999995 t664_leaf1 <= 1e+06
 T664_R2_f1_ge: const_Lot_Area - 1.006e+06 t664_leaf2 >= -1e+06
 T664_R2_f3_ge: Overall_Qual_calc - 1.000005e+06 t664_leaf2 >= -1e+06
 TREE_665_ONEHOT: t665_leaf0 + t665_leaf1 + t665_leaf2 = 1
 T665_L0_f213_le: x_b1_is_GLQ <= 1
 T665_L0_f1_le: const_Lot_Area + 980478 t665_leaf0 <= 1e+06
 T665_L1_f213_le: x_b1_is_GLQ <= 1
 T665_R1_f1_ge: const_Lot_Area - 1.019522e+06 t665_leaf1 >= -1e+06
 T665_R2_f213_ge: x_b1_is_GLQ - t665_leaf2 >= 0
 TREE_666_ONEHOT: t666_leaf0 + t666_leaf1 + t666_leaf2 = 1
 T666_L0_f114_le: const_Neighborhood_StoneBr + 999999 t666_leaf0 <= 1e+06
 T666_R1_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t666_leaf1
   >= -1e+06
 T666_L1_f15_le: const_Total_Bsmt_SF + 998658 t666_leaf1 <= 1e+06
 T666_R2_f114_ge: const_Neighborhood_StoneBr - 1.000001e+06 t666_leaf2
   >= -1e+06
 T666_R2_f15_ge: const_Total_Bsmt_SF - 1.001342e+06 t666_leaf2 >= -1e+06
 TREE_667_ONEHOT: t667_leaf0 + t667_leaf1 + t667_leaf2 + t667_leaf3 = 1
 T667_L0_f20_le: x_Gr_Liv_Area + 9117 t667_leaf0 <= 10000
 T667_L0_f33_le: x_Garage_Area + 99714 t667_leaf0 <= 100000
 T667_L1_f20_le: x_Gr_Liv_Area + 9117 t667_leaf1 <= 10000
 T667_R1_f33_ge: x_Garage_Area - 286 t667_leaf1 >= 0
 T667_L1_f161_le: x_ex1_is_AsbShng <= 1
 T667_L2_f20_le: x_Gr_Liv_Area + 9117 t667_leaf2 <= 10000
 T667_R2_f33_ge: x_Garage_Area - 286 t667_leaf2 >= 0
 T667_R2_f161_ge: x_ex1_is_AsbShng - t667_leaf2 >= 0
 T667_R3_f20_ge: x_Gr_Liv_Area - 883 t667_leaf3 >= 0
 TREE_668_ONEHOT: t668_leaf0 + t668_leaf1 + t668_leaf2 + t668_leaf3 = 1
 T668_L0_f33_le: x_Garage_Area + 99356 t668_leaf0 <= 100000
 T668_R1_f33_ge: x_Garage_Area - 644 t668_leaf1 >= 0
 T668_L1_f96_le: const_Neighborhood_Edwards + 999999 t668_leaf1 <= 1e+06
 T668_L1_f31_le: const_Garage_Yr_Blt + 998058 t668_leaf1 <= 1e+06
 T668_R2_f33_ge: x_Garage_Area - 644 t668_leaf2 >= 0
 T668_L2_f96_le: const_Neighborhood_Edwards + 999999 t668_leaf2 <= 1e+06
 T668_R2_f31_ge: const_Garage_Yr_Blt - 1.001942e+06 t668_leaf2 >= -1e+06
 T668_R3_f33_ge: x_Garage_Area - 644 t668_leaf3 >= 0
 T668_R3_f96_ge: const_Neighborhood_Edwards - 1.000001e+06 t668_leaf3
   >= -1e+06
 TREE_669_ONEHOT: t669_leaf0 = 1
 TREE_670_ONEHOT: t670_leaf0 = 1
 TREE_671_ONEHOT: t671_leaf0 + t671_leaf1 = 1
 T671_L0_f164_le: x_ex1_is_BrkFace <= 1
 T671_R1_f164_ge: x_ex1_is_BrkFace - t671_leaf1 >= 0
 TREE_672_ONEHOT: t672_leaf0 = 1
 TREE_673_ONEHOT: t673_leaf0 + t673_leaf1 + t673_leaf2 + t673_leaf3
   + t673_leaf4 = 1
 T673_L0_f11_le: x_Bsmt_Cond + 2 t673_leaf0 <= 4
 T673_L0_f31_le: const_Garage_Yr_Blt + 998026 t673_leaf0 <= 1e+06
 T673_L0_f214_le: x_b1_is_LwQ <= 1
 T673_L1_f11_le: x_Bsmt_Cond + 2 t673_leaf1 <= 4
 T673_L1_f31_le: const_Garage_Yr_Blt + 998026 t673_leaf1 <= 1e+06
 T673_R1_f214_ge: x_b1_is_LwQ - t673_leaf1 >= 0
 T673_L2_f11_le: x_Bsmt_Cond + 2 t673_leaf2 <= 4
 T673_R2_f31_ge: const_Garage_Yr_Blt - 1.001974e+06 t673_leaf2 >= -1e+06
 T673_R3_f11_ge: x_Bsmt_Cond - 3 t673_leaf3 >= -1
 T673_L3_f1_le: const_Lot_Area + 995500 t673_leaf3 <= 1e+06
 T673_R4_f11_ge: x_Bsmt_Cond - 3 t673_leaf4 >= -1
 T673_R4_f1_ge: const_Lot_Area - 1.0045e+06 t673_leaf4 >= -1e+06
 TREE_674_ONEHOT: t674_leaf0 + t674_leaf1 = 1
 T674_L0_f3_le: Overall_Qual_calc + 999990 t674_leaf0 <= 1e+06
 T674_R1_f3_ge: Overall_Qual_calc - 1.00001e+06 t674_leaf1 >= -1e+06
 TREE_675_ONEHOT: t675_leaf0 = 1
 TREE_676_ONEHOT: t676_leaf0 + t676_leaf1 = 1
 T676_L0_f3_le: Overall_Qual_calc + 999997 t676_leaf0 <= 1e+06
 T676_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t676_leaf1 >= -1e+06
 TREE_677_ONEHOT: t677_leaf0 = 1
 TREE_678_ONEHOT: t678_leaf0 + t678_leaf1 + t678_leaf2 + t678_leaf3
   + t678_leaf4 = 1
 T678_L0_f207_le: const_Bsmt_Exposure_Gd + 999999 t678_leaf0 <= 1e+06
 T678_L0_f38_le: x_Enclosed_Porch + 99735 t678_leaf0 <= 100000
 T678_L0_f1_le: const_Lot_Area + 988208 t678_leaf0 <= 1e+06
 T678_L1_f207_le: const_Bsmt_Exposure_Gd + 999999 t678_leaf1 <= 1e+06
 T678_L1_f38_le: x_Enclosed_Porch + 99735 t678_leaf1 <= 100000
 T678_R1_f1_ge: const_Lot_Area - 1.011792e+06 t678_leaf1 >= -1e+06
 T678_L2_f207_le: const_Bsmt_Exposure_Gd + 999999 t678_leaf2 <= 1e+06
 T678_R2_f38_ge: x_Enclosed_Porch - 265 t678_leaf2 >= 0
 T678_R3_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t678_leaf3
   >= -1e+06
 T678_L3_f31_le: const_Garage_Yr_Blt + 998039 t678_leaf3 <= 1e+06
 T678_R4_f207_ge: const_Bsmt_Exposure_Gd - 1.000001e+06 t678_leaf4
   >= -1e+06
 T678_R4_f31_ge: const_Garage_Yr_Blt - 1.001961e+06 t678_leaf4 >= -1e+06
 TREE_679_ONEHOT: t679_leaf0 = 1
 TREE_680_ONEHOT: t680_leaf0 + t680_leaf1 + t680_leaf2 + t680_leaf3
   + t680_leaf4 = 1
 T680_L0_f3_le: Overall_Qual_calc + 999990 t680_leaf0 <= 1e+06
 T680_L0_f29_le: const_Fireplaces + 999999 t680_leaf0 <= 1e+06
 T680_L0_f20_le: x_Gr_Liv_Area + 8205 t680_leaf0 <= 10000
 T680_L1_f3_le: Overall_Qual_calc + 999990 t680_leaf1 <= 1e+06
 T680_L1_f29_le: const_Fireplaces + 999999 t680_leaf1 <= 1e+06
 T680_R1_f20_ge: x_Gr_Liv_Area - 1795 t680_leaf1 >= 0
 T680_L2_f3_le: Overall_Qual_calc + 999990 t680_leaf2 <= 1e+06
 T680_R2_f29_ge: const_Fireplaces - 1.000001e+06 t680_leaf2 >= -1e+06
 T680_L2_f46_le: const_MS_SubClass_30 + 999999 t680_leaf2 <= 1e+06
 T680_L3_f3_le: Overall_Qual_calc + 999990 t680_leaf3 <= 1e+06
 T680_R3_f29_ge: const_Fireplaces - 1.000001e+06 t680_leaf3 >= -1e+06
 T680_R3_f46_ge: const_MS_SubClass_30 - 1.000001e+06 t680_leaf3 >= -1e+06
 T680_R4_f3_ge: Overall_Qual_calc - 1.00001e+06 t680_leaf4 >= -1e+06
 TREE_681_ONEHOT: t681_leaf0 + t681_leaf1 = 1
 T681_L0_f15_le: const_Total_Bsmt_SF + 997777 t681_leaf0 <= 1e+06
 T681_R1_f15_ge: const_Total_Bsmt_SF - 1.002223e+06 t681_leaf1 >= -1e+06
 TREE_682_ONEHOT: t682_leaf0 = 1
 TREE_683_ONEHOT: t683_leaf0 + t683_leaf1 + t683_leaf2 + t683_leaf3 = 1
 T683_L0_f17_le: x_1st_Flr_SF + 3566 t683_leaf0 <= 5000
 T683_R1_f17_ge: x_1st_Flr_SF - 1434 t683_leaf1 >= 0
 T683_L1_f33_le: x_Garage_Area + 99712 t683_leaf1 <= 100000
 T683_R2_f17_ge: x_1st_Flr_SF - 1434 t683_leaf2 >= 0
 T683_R2_f33_ge: x_Garage_Area - 288 t683_leaf2 >= 0
 T683_L2_f206_le: const_Bsmt_Exposure_Av + 999999 t683_leaf2 <= 1e+06
 T683_R3_f17_ge: x_1st_Flr_SF - 1434 t683_leaf3 >= 0
 T683_R3_f33_ge: x_Garage_Area - 288 t683_leaf3 >= 0
 T683_R3_f206_ge: const_Bsmt_Exposure_Av - 1.000001e+06 t683_leaf3
   >= -1e+06
 TREE_684_ONEHOT: t684_leaf0 + t684_leaf1 + t684_leaf2 = 1
 T684_L0_f20_le: x_Gr_Liv_Area + 9253 t684_leaf0 <= 10000
 T684_L0_f14_le: x_Bsmt_Unf_SF + 99727 t684_leaf0 <= 100000
 T684_L1_f20_le: x_Gr_Liv_Area + 9253 t684_leaf1 <= 10000
 T684_R1_f14_ge: x_Bsmt_Unf_SF - 273 t684_leaf1 >= 0
 T684_R2_f20_ge: x_Gr_Liv_Area - 747 t684_leaf2 >= 0
 TREE_685_ONEHOT: t685_leaf0 = 1
 TREE_686_ONEHOT: t686_leaf0 = 1
 TREE_687_ONEHOT: t687_leaf0 + t687_leaf1 + t687_leaf2 = 1
 T687_L0_f3_le: Overall_Qual_calc + 999997 t687_leaf0 <= 1e+06
 T687_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t687_leaf1 >= -1e+06
 T687_L1_f5_le: const_Year_Built + 997995 t687_leaf1 <= 1e+06
 T687_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t687_leaf2 >= -1e+06
 T687_R2_f5_ge: const_Year_Built - 1.002005e+06 t687_leaf2 >= -1e+06
 TREE_688_ONEHOT: t688_leaf0 = 1
 TREE_689_ONEHOT: t689_leaf0 = 1
 TREE_690_ONEHOT: t690_leaf0 = 1
 TREE_691_ONEHOT: t691_leaf0 = 1
 TREE_692_ONEHOT: t692_leaf0 + t692_leaf1 = 1
 T692_L0_f0_le: const_Lot_Frontage + 999848 t692_leaf0 <= 1e+06
 T692_R1_f0_ge: const_Lot_Frontage - 1.000152e+06 t692_leaf1 >= -1e+06
 TREE_693_ONEHOT: t693_leaf0 + t693_leaf1 = 1
 T693_L0_f20_le: x_Gr_Liv_Area + 9253 t693_leaf0 <= 10000
 T693_R1_f20_ge: x_Gr_Liv_Area - 747 t693_leaf1 >= 0
 TREE_694_ONEHOT: t694_leaf0 = 1
 TREE_695_ONEHOT: t695_leaf0 = 1
 TREE_696_ONEHOT: t696_leaf0 = 1
 TREE_697_ONEHOT: t697_leaf0 + t697_leaf1 = 1
 T697_L0_f25_le: x_Bedroom_AbvGr + 3 t697_leaf0 <= 6
 T697_R1_f25_ge: x_Bedroom_AbvGr - 3 t697_leaf1 >= 0
 TREE_698_ONEHOT: t698_leaf0 + t698_leaf1 = 1
 T698_L0_f33_le: x_Garage_Area + 98846 t698_leaf0 <= 100000
 T698_R1_f33_ge: x_Garage_Area - 1154 t698_leaf1 >= 0
 TREE_699_ONEHOT: t699_leaf0 = 1
 TREE_700_ONEHOT: t700_leaf0 = 1
 TREE_701_ONEHOT: t701_leaf0 = 1
 TREE_702_ONEHOT: t702_leaf0 + t702_leaf1 + t702_leaf2 + t702_leaf3 = 1
 T702_L0_f5_le: const_Year_Built + 998068 t702_leaf0 <= 1e+06
 T702_L0_f202_le: const_Foundation_PConc + 999999 t702_leaf0 <= 1e+06
 T702_L1_f5_le: const_Year_Built + 998068 t702_leaf1 <= 1e+06
 T702_R1_f202_ge: const_Foundation_PConc - 1.000001e+06 t702_leaf1
   >= -1e+06
 T702_R2_f5_ge: const_Year_Built - 1.001932e+06 t702_leaf2 >= -1e+06
 T702_L2_f21_le: const_Bsmt_Full_Bath + 999998 t702_leaf2 <= 1e+06
 T702_R3_f5_ge: const_Year_Built - 1.001932e+06 t702_leaf3 >= -1e+06
 T702_R3_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t702_leaf3 >= -1e+06
 TREE_703_ONEHOT: t703_leaf0 = 1
 TREE_704_ONEHOT: t704_leaf0 = 1
 TREE_705_ONEHOT: t705_leaf0 + t705_leaf1 + t705_leaf2 + t705_leaf3
   + t705_leaf4 = 1
 T705_L0_f15_le: const_Total_Bsmt_SF + 998581 t705_leaf0 <= 1e+06
 T705_L0_f15_le: const_Total_Bsmt_SF + 998620 t705_leaf0 <= 1e+06
 T705_L1_f15_le: const_Total_Bsmt_SF + 998581 t705_leaf1 <= 1e+06
 T705_R1_f15_ge: const_Total_Bsmt_SF - 1.00138e+06 t705_leaf1 >= -1e+06
 T705_R2_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t705_leaf2 >= -1e+06
 T705_L2_f17_le: x_1st_Flr_SF + 3578 t705_leaf2 <= 5000
 T705_R3_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t705_leaf3 >= -1e+06
 T705_R3_f17_ge: x_1st_Flr_SF - 1422 t705_leaf3 >= 0
 T705_L3_f33_le: x_Garage_Area + 99712 t705_leaf3 <= 100000
 T705_R4_f15_ge: const_Total_Bsmt_SF - 1.001419e+06 t705_leaf4 >= -1e+06
 T705_R4_f17_ge: x_1st_Flr_SF - 1422 t705_leaf4 >= 0
 T705_R4_f33_ge: x_Garage_Area - 288 t705_leaf4 >= 0
 TREE_706_ONEHOT: t706_leaf0 + t706_leaf1 + t706_leaf2 = 1
 T706_L0_f8_le: x_Exter_Qual + t706_leaf0 <= 4
 T706_R1_f8_ge: x_Exter_Qual - 4 t706_leaf1 >= -1
 T706_L1_f20_le: x_Gr_Liv_Area + 7206 t706_leaf1 <= 10000
 T706_R2_f8_ge: x_Exter_Qual - 4 t706_leaf2 >= -1
 T706_R2_f20_ge: x_Gr_Liv_Area - 2794 t706_leaf2 >= 0
 TREE_707_ONEHOT: t707_leaf0 = 1
 TREE_708_ONEHOT: t708_leaf0 = 1
 TREE_709_ONEHOT: t709_leaf0 + t709_leaf1 + t709_leaf2 = 1
 T709_L0_f274_le: const_Mo_Sold_4 + 999999 t709_leaf0 <= 1e+06
 T709_R1_f274_ge: const_Mo_Sold_4 - 1.000001e+06 t709_leaf1 >= -1e+06
 T709_L1_f29_le: const_Fireplaces + 999999 t709_leaf1 <= 1e+06
 T709_R2_f274_ge: const_Mo_Sold_4 - 1.000001e+06 t709_leaf2 >= -1e+06
 T709_R2_f29_ge: const_Fireplaces - 1.000001e+06 t709_leaf2 >= -1e+06
 TREE_710_ONEHOT: t710_leaf0 + t710_leaf1 + t710_leaf2 + t710_leaf3 = 1
 T710_L0_f38_le: x_Enclosed_Porch + 99982 t710_leaf0 <= 100000
 T710_R1_f38_ge: x_Enclosed_Porch - 18 t710_leaf1 >= 0
 T710_L1_f297_le: const_Sale_Condition_Normal + 999999 t710_leaf1 <= 1e+06
 T710_L1_f283_le: const_Sale_Type_COD + 999999 t710_leaf1 <= 1e+06
 T710_R2_f38_ge: x_Enclosed_Porch - 18 t710_leaf2 >= 0
 T710_L2_f297_le: const_Sale_Condition_Normal + 999999 t710_leaf2 <= 1e+06
 T710_R2_f283_ge: const_Sale_Type_COD - 1.000001e+06 t710_leaf2 >= -1e+06
 T710_R3_f38_ge: x_Enclosed_Porch - 18 t710_leaf3 >= 0
 T710_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t710_leaf3
   >= -1e+06
 TREE_711_ONEHOT: t711_leaf0 = 1
 TREE_712_ONEHOT: t712_leaf0 + t712_leaf1 + t712_leaf2 + t712_leaf3 = 1
 T712_L0_f76_le: const_Lot_Shape_Reg + 999999 t712_leaf0 <= 1e+06
 T712_L0_f28_le: x_TotRms_AbvGrd + 3 t712_leaf0 <= 15
 T712_L0_f86_le: const_Land_Slope_Gtl + 999999 t712_leaf0 <= 1e+06
 T712_L1_f76_le: const_Lot_Shape_Reg + 999999 t712_leaf1 <= 1e+06
 T712_L1_f28_le: x_TotRms_AbvGrd + 3 t712_leaf1 <= 15
 T712_R1_f86_ge: const_Land_Slope_Gtl - 1.000001e+06 t712_leaf1 >= -1e+06
 T712_L2_f76_le: const_Lot_Shape_Reg + 999999 t712_leaf2 <= 1e+06
 T712_R2_f28_ge: x_TotRms_AbvGrd - 12 t712_leaf2 >= 0
 T712_R3_f76_ge: const_Lot_Shape_Reg - 1.000001e+06 t712_leaf3 >= -1e+06
 TREE_713_ONEHOT: t713_leaf0 + t713_leaf1 = 1
 T713_L0_f258_le: x_paved_drive_is_P <= 1
 T713_R1_f258_ge: x_paved_drive_is_P - t713_leaf1 >= 0
 TREE_714_ONEHOT: t714_leaf0 = 1
 TREE_715_ONEHOT: t715_leaf0 + t715_leaf1 + t715_leaf2 + t715_leaf3 = 1
 T715_L0_f95_le: const_Neighborhood_Crawfor + 999999 t715_leaf0 <= 1e+06
 T715_L0_f1_le: const_Lot_Area + 988208 t715_leaf0 <= 1e+06
 T715_L1_f95_le: const_Neighborhood_Crawfor + 999999 t715_leaf1 <= 1e+06
 T715_R1_f1_ge: const_Lot_Area - 1.011792e+06 t715_leaf1 >= -1e+06
 T715_L1_f297_le: const_Sale_Condition_Normal + 999999 t715_leaf1 <= 1e+06
 T715_L2_f95_le: const_Neighborhood_Crawfor + 999999 t715_leaf2 <= 1e+06
 T715_R2_f1_ge: const_Lot_Area - 1.011792e+06 t715_leaf2 >= -1e+06
 T715_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t715_leaf2
   >= -1e+06
 T715_R3_f95_ge: const_Neighborhood_Crawfor - 1.000001e+06 t715_leaf3
   >= -1e+06
 TREE_716_ONEHOT: t716_leaf0 = 1
 TREE_717_ONEHOT: t717_leaf0 = 1
 TREE_718_ONEHOT: t718_leaf0 = 1
 TREE_719_ONEHOT: t719_leaf0 = 1
 TREE_720_ONEHOT: t720_leaf0 + t720_leaf1 = 1
 T720_L0_f3_le: Overall_Qual_calc + 999997 t720_leaf0 <= 1e+06
 T720_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t720_leaf1 >= -1e+06
 TREE_721_ONEHOT: t721_leaf0 = 1
 TREE_722_ONEHOT: t722_leaf0 = 1
 TREE_723_ONEHOT: t723_leaf0 = 1
 TREE_724_ONEHOT: t724_leaf0 = 1
 TREE_725_ONEHOT: t725_leaf0 = 1
 TREE_726_ONEHOT: t726_leaf0 + t726_leaf1 + t726_leaf2 = 1
 T726_L0_f154_le: x_roof_matl_is_CompShg <= 1
 T726_R1_f154_ge: x_roof_matl_is_CompShg - t726_leaf1 >= 0
 T726_L1_f31_le: const_Garage_Yr_Blt + 997992 t726_leaf1 <= 1e+06
 T726_R2_f154_ge: x_roof_matl_is_CompShg - t726_leaf2 >= 0
 T726_R2_f31_ge: const_Garage_Yr_Blt - 1.002008e+06 t726_leaf2 >= -1e+06
 TREE_727_ONEHOT: t727_leaf0 + t727_leaf1 + t727_leaf2 = 1
 T727_L0_f177_le: x_ex2_is_AsbShng <= 1
 T727_R1_f177_ge: x_ex2_is_AsbShng - t727_leaf1 >= 0
 T727_L1_f4_le: const_Overall_Cond + 999995 t727_leaf1 <= 1e+06
 T727_R2_f177_ge: x_ex2_is_AsbShng - t727_leaf2 >= 0
 T727_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t727_leaf2 >= -1e+06
 TREE_728_ONEHOT: t728_leaf0 + t728_leaf1 = 1
 T728_L0_f23_le: x_Full_Bath + t728_leaf0 <= 4
 T728_R1_f23_ge: x_Full_Bath - 3 t728_leaf1 >= 0
 TREE_729_ONEHOT: t729_leaf0 = 1
 TREE_730_ONEHOT: t730_leaf0 + t730_leaf1 + t730_leaf2 + t730_leaf3
   + t730_leaf4 = 1
 T730_L0_f20_le: x_Gr_Liv_Area + 9012 t730_leaf0 <= 10000
 T730_R1_f20_ge: x_Gr_Liv_Area - 988 t730_leaf1 >= 0
 T730_L1_f6_le: const_Year_Remod_Add + 998028 t730_leaf1 <= 1e+06
 T730_L1_f26_le: x_Kitchen_AbvGr + t730_leaf1 <= 3
 T730_R2_f20_ge: x_Gr_Liv_Area - 988 t730_leaf2 >= 0
 T730_L2_f6_le: const_Year_Remod_Add + 998028 t730_leaf2 <= 1e+06
 T730_R2_f26_ge: x_Kitchen_AbvGr - 2 t730_leaf2 >= 0
 T730_R3_f20_ge: x_Gr_Liv_Area - 988 t730_leaf3 >= 0
 T730_R3_f6_ge: const_Year_Remod_Add - 1.001972e+06 t730_leaf3 >= -1e+06
 T730_L3_f7_le: x_Mas_Vnr_Area + 1212 t730_leaf3 <= 2000
 T730_R4_f20_ge: x_Gr_Liv_Area - 988 t730_leaf4 >= 0
 T730_R4_f6_ge: const_Year_Remod_Add - 1.001972e+06 t730_leaf4 >= -1e+06
 T730_R4_f7_ge: x_Mas_Vnr_Area - 788 t730_leaf4 >= 0
 TREE_731_ONEHOT: t731_leaf0 = 1
 TREE_732_ONEHOT: t732_leaf0 = 1
 TREE_733_ONEHOT: t733_leaf0 = 1
 TREE_734_ONEHOT: t734_leaf0 = 1
 TREE_735_ONEHOT: t735_leaf0 = 1
 TREE_736_ONEHOT: t736_leaf0 = 1
 TREE_737_ONEHOT: t737_leaf0 = 1
 TREE_738_ONEHOT: t738_leaf0 = 1
 TREE_739_ONEHOT: t739_leaf0 + t739_leaf1 = 1
 T739_L0_f37_le: x_Open_Porch_SF + 99831 t739_leaf0 <= 100000
 T739_R1_f37_ge: x_Open_Porch_SF - 169 t739_leaf1 >= 0
 TREE_740_ONEHOT: t740_leaf0 = 1
 TREE_741_ONEHOT: t741_leaf0 + t741_leaf1 + t741_leaf2 = 1
 T741_L0_f1_le: const_Lot_Area + 976743 t741_leaf0 <= 1e+06
 T741_R1_f1_ge: const_Lot_Area - 1.023257e+06 t741_leaf1 >= -1e+06
 T741_L1_f297_le: const_Sale_Condition_Normal + 999999 t741_leaf1 <= 1e+06
 T741_R2_f1_ge: const_Lot_Area - 1.023257e+06 t741_leaf2 >= -1e+06
 T741_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t741_leaf2
   >= -1e+06
 TREE_742_ONEHOT: t742_leaf0 + t742_leaf1 + t742_leaf2 + t742_leaf3
   + t742_leaf4 + t742_leaf5 + t742_leaf6 = 1
 T742_L0_f272_le: const_Mo_Sold_2 + 999999 t742_leaf0 <= 1e+06
 T742_L0_f232_le: central_air_yes + 999999 t742_leaf0 <= 1e+06
 T742_L0_f40_le: x_Screen_Porch + 99947 t742_leaf0 <= 100000
 T742_L1_f272_le: const_Mo_Sold_2 + 999999 t742_leaf1 <= 1e+06
 T742_L1_f232_le: central_air_yes + 999999 t742_leaf1 <= 1e+06
 T742_R1_f40_ge: x_Screen_Porch - 53 t742_leaf1 >= 0
 T742_L2_f272_le: const_Mo_Sold_2 + 999999 t742_leaf2 <= 1e+06
 T742_R2_f232_ge: central_air_yes - 1.000001e+06 t742_leaf2 >= -1e+06
 T742_L2_f33_le: x_Garage_Area + 99730 t742_leaf2 <= 100000
 T742_L3_f272_le: const_Mo_Sold_2 + 999999 t742_leaf3 <= 1e+06
 T742_R3_f232_ge: central_air_yes - 1.000001e+06 t742_leaf3 >= -1e+06
 T742_R3_f33_ge: x_Garage_Area - 270 t742_leaf3 >= 0
 T742_R4_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t742_leaf4 >= -1e+06
 T742_L4_f297_le: const_Sale_Condition_Normal + 999999 t742_leaf4 <= 1e+06
 T742_L4_f37_le: x_Open_Porch_SF + 99950 t742_leaf4 <= 100000
 T742_R5_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t742_leaf5 >= -1e+06
 T742_L5_f297_le: const_Sale_Condition_Normal + 999999 t742_leaf5 <= 1e+06
 T742_R5_f37_ge: x_Open_Porch_SF - 50 t742_leaf5 >= 0
 T742_R6_f272_ge: const_Mo_Sold_2 - 1.000001e+06 t742_leaf6 >= -1e+06
 T742_R6_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t742_leaf6
   >= -1e+06
 TREE_743_ONEHOT: t743_leaf0 = 1
 TREE_744_ONEHOT: t744_leaf0 = 1
 TREE_745_ONEHOT: t745_leaf0 = 1
 TREE_746_ONEHOT: t746_leaf0 = 1
 TREE_747_ONEHOT: t747_leaf0 = 1
 TREE_748_ONEHOT: t748_leaf0 = 1
 TREE_749_ONEHOT: t749_leaf0 + t749_leaf1 + t749_leaf2 + t749_leaf3 = 1
 T749_L0_f259_le: x_paved_drive_is_Y <= 1
 T749_L0_f33_le: x_Garage_Area + 99597 t749_leaf0 <= 100000
 T749_L0_f1_le: const_Lot_Area + 989663 t749_leaf0 <= 1e+06
 T749_L1_f259_le: x_paved_drive_is_Y <= 1
 T749_L1_f33_le: x_Garage_Area + 99597 t749_leaf1 <= 100000
 T749_R1_f1_ge: const_Lot_Area - 1.010337e+06 t749_leaf1 >= -1e+06
 T749_L2_f259_le: x_paved_drive_is_Y <= 1
 T749_R2_f33_ge: x_Garage_Area - 403 t749_leaf2 >= 0
 T749_R3_f259_ge: x_paved_drive_is_Y - t749_leaf3 >= 0
 TREE_750_ONEHOT: t750_leaf0 = 1
 TREE_751_ONEHOT: t751_leaf0 = 1
 TREE_752_ONEHOT: t752_leaf0 = 1
 TREE_753_ONEHOT: t753_leaf0 = 1
 TREE_754_ONEHOT: t754_leaf0 = 1
 TREE_755_ONEHOT: t755_leaf0 = 1
 TREE_756_ONEHOT: t756_leaf0 = 1
 TREE_757_ONEHOT: t757_leaf0 + t757_leaf1 + t757_leaf2 + t757_leaf3 = 1
 T757_L0_f44_le: const_Yr_Sold + 997990 t757_leaf0 <= 1e+06
 T757_R1_f44_ge: const_Yr_Sold - 1.00201e+06 t757_leaf1 >= -1e+06
 T757_L1_f15_le: const_Total_Bsmt_SF + 999239 t757_leaf1 <= 1e+06
 T757_L1_f297_le: const_Sale_Condition_Normal + 999999 t757_leaf1 <= 1e+06
 T757_R2_f44_ge: const_Yr_Sold - 1.00201e+06 t757_leaf2 >= -1e+06
 T757_L2_f15_le: const_Total_Bsmt_SF + 999239 t757_leaf2 <= 1e+06
 T757_R2_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t757_leaf2
   >= -1e+06
 T757_R3_f44_ge: const_Yr_Sold - 1.00201e+06 t757_leaf3 >= -1e+06
 T757_R3_f15_ge: const_Total_Bsmt_SF - 1.000761e+06 t757_leaf3 >= -1e+06
 TREE_758_ONEHOT: t758_leaf0 = 1
 TREE_759_ONEHOT: t759_leaf0 + t759_leaf1 = 1
 T759_L0_f240_le: const_FunctioNo_aplical_Min1 + 999999 t759_leaf0
   <= 1e+06
 T759_R1_f240_ge: const_FunctioNo_aplical_Min1 - 1.000001e+06 t759_leaf1
   >= -1e+06
 TREE_760_ONEHOT: t760_leaf0 = 1
 TREE_761_ONEHOT: t761_leaf0 + t761_leaf1 + t761_leaf2 + t761_leaf3 = 1
 T761_L0_f10_le: const_Bsmt_Qual + 999996 t761_leaf0 <= 1e+06
 T761_L0_f33_le: x_Garage_Area + 99134 t761_leaf0 <= 100000
 T761_L1_f10_le: const_Bsmt_Qual + 999996 t761_leaf1 <= 1e+06
 T761_R1_f33_ge: x_Garage_Area - 866 t761_leaf1 >= 0
 T761_R2_f10_ge: const_Bsmt_Qual - 1.000004e+06 t761_leaf2 >= -1e+06
 T761_L2_f33_le: x_Garage_Area + 99356 t761_leaf2 <= 100000
 T761_R3_f10_ge: const_Bsmt_Qual - 1.000004e+06 t761_leaf3 >= -1e+06
 T761_R3_f33_ge: x_Garage_Area - 644 t761_leaf3 >= 0
 TREE_762_ONEHOT: t762_leaf0 = 1
 TREE_763_ONEHOT: t763_leaf0 + t763_leaf1 + t763_leaf2 + t763_leaf3
   + t763_leaf4 = 1
 T763_L0_f150_le: const_Roof_Style_Hip + 999999 t763_leaf0 <= 1e+06
 T763_L0_f1_le: const_Lot_Area + 980478 t763_leaf0 <= 1e+06
 T763_L1_f150_le: const_Roof_Style_Hip + 999999 t763_leaf1 <= 1e+06
 T763_R1_f1_ge: const_Lot_Area - 1.019522e+06 t763_leaf1 >= -1e+06
 T763_R2_f150_ge: const_Roof_Style_Hip - 1.000001e+06 t763_leaf2 >= -1e+06
 T763_L2_f44_le: const_Yr_Sold + 997992 t763_leaf2 <= 1e+06
 T763_R3_f150_ge: const_Roof_Style_Hip - 1.000001e+06 t763_leaf3 >= -1e+06
 T763_R3_f44_ge: const_Yr_Sold - 1.002008e+06 t763_leaf3 >= -1e+06
 T763_L3_f14_le: x_Bsmt_Unf_SF + 99534 t763_leaf3 <= 100000
 T763_R4_f150_ge: const_Roof_Style_Hip - 1.000001e+06 t763_leaf4 >= -1e+06
 T763_R4_f44_ge: const_Yr_Sold - 1.002008e+06 t763_leaf4 >= -1e+06
 T763_R4_f14_ge: x_Bsmt_Unf_SF - 466 t763_leaf4 >= 0
 TREE_764_ONEHOT: t764_leaf0 + t764_leaf1 + t764_leaf2 + t764_leaf3 = 1
 T764_L0_f14_le: x_Bsmt_Unf_SF + 99559 t764_leaf0 <= 100000
 T764_L0_f4_le: const_Overall_Cond + 999996 t764_leaf0 <= 1e+06
 T764_L1_f14_le: x_Bsmt_Unf_SF + 99559 t764_leaf1 <= 100000
 T764_R1_f4_ge: const_Overall_Cond - 1.000004e+06 t764_leaf1 >= -1e+06
 T764_L1_f79_le: const_Land_Contour_Low + 999999 t764_leaf1 <= 1e+06
 T764_L2_f14_le: x_Bsmt_Unf_SF + 99559 t764_leaf2 <= 100000
 T764_R2_f4_ge: const_Overall_Cond - 1.000004e+06 t764_leaf2 >= -1e+06
 T764_R2_f79_ge: const_Land_Contour_Low - 1.000001e+06 t764_leaf2
   >= -1e+06
 T764_R3_f14_ge: x_Bsmt_Unf_SF - 441 t764_leaf3 >= 0
 TREE_765_ONEHOT: t765_leaf0 = 1
 TREE_766_ONEHOT: t766_leaf0 + t766_leaf1 = 1
 T766_L0_f72_le: const_Alley_Pave + 999999 t766_leaf0 <= 1e+06
 T766_R1_f72_ge: const_Alley_Pave - 1.000001e+06 t766_leaf1 >= -1e+06
 TREE_767_ONEHOT: t767_leaf0 = 1
 TREE_768_ONEHOT: t768_leaf0 + t768_leaf1 + t768_leaf2 = 1
 T768_L0_f37_le: x_Open_Porch_SF + 99740 t768_leaf0 <= 100000
 T768_L0_f17_le: x_1st_Flr_SF + 3027 t768_leaf0 <= 5000
 T768_L1_f37_le: x_Open_Porch_SF + 99740 t768_leaf1 <= 100000
 T768_R1_f17_ge: x_1st_Flr_SF - 1973 t768_leaf1 >= 0
 T768_R2_f37_ge: x_Open_Porch_SF - 260 t768_leaf2 >= 0
 TREE_769_ONEHOT: t769_leaf0 = 1
 TREE_770_ONEHOT: t770_leaf0 = 1
 TREE_771_ONEHOT: t771_leaf0 = 1
 TREE_772_ONEHOT: t772_leaf0 + t772_leaf1 + t772_leaf2 = 1
 T772_L0_f1_le: const_Lot_Area + 989800 t772_leaf0 <= 1e+06
 T772_R1_f1_ge: const_Lot_Area - 1.0102e+06 t772_leaf1 >= -1e+06
 T772_L1_f15_le: const_Total_Bsmt_SF + 999519 t772_leaf1 <= 1e+06
 T772_R2_f1_ge: const_Lot_Area - 1.0102e+06 t772_leaf2 >= -1e+06
 T772_R2_f15_ge: const_Total_Bsmt_SF - 1.000481e+06 t772_leaf2 >= -1e+06
 TREE_773_ONEHOT: t773_leaf0 = 1
 TREE_774_ONEHOT: t774_leaf0 = 1
 TREE_775_ONEHOT: t775_leaf0 + t775_leaf1 = 1
 T775_L0_f100_le: const_Neighborhood_IDOTRR + 999999 t775_leaf0 <= 1e+06
 T775_R1_f100_ge: const_Neighborhood_IDOTRR - 1.000001e+06 t775_leaf1
   >= -1e+06
 TREE_776_ONEHOT: t776_leaf0 = 1
 TREE_777_ONEHOT: t777_leaf0 + t777_leaf1 = 1
 T777_L0_f33_le: x_Garage_Area + 99028 t777_leaf0 <= 100000
 T777_R1_f33_ge: x_Garage_Area - 972 t777_leaf1 >= 0
 TREE_778_ONEHOT: t778_leaf0 + t778_leaf1 + t778_leaf2 + t778_leaf3 = 1
 T778_L0_f62_le: const_MS_Zoning_C_(all) + 999999 t778_leaf0 <= 1e+06
 T778_L0_f177_le: x_ex2_is_AsbShng <= 1
 T778_L1_f62_le: const_MS_Zoning_C_(all) + 999999 t778_leaf1 <= 1e+06
 T778_R1_f177_ge: x_ex2_is_AsbShng - t778_leaf1 >= 0
 T778_L1_f4_le: const_Overall_Cond + 999995 t778_leaf1 <= 1e+06
 T778_L2_f62_le: const_MS_Zoning_C_(all) + 999999 t778_leaf2 <= 1e+06
 T778_R2_f177_ge: x_ex2_is_AsbShng - t778_leaf2 >= 0
 T778_R2_f4_ge: const_Overall_Cond - 1.000005e+06 t778_leaf2 >= -1e+06
 T778_R3_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t778_leaf3
   >= -1e+06
 TREE_779_ONEHOT: t779_leaf0 = 1
 TREE_780_ONEHOT: t780_leaf0 = 1
 TREE_781_ONEHOT: t781_leaf0 + t781_leaf1 + t781_leaf2 = 1
 T781_L0_f3_le: Overall_Qual_calc + 999997 t781_leaf0 <= 1e+06
 T781_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t781_leaf1 >= -1e+06
 T781_L1_f33_le: x_Garage_Area + 99196 t781_leaf1 <= 100000
 T781_R2_f3_ge: Overall_Qual_calc - 1.000003e+06 t781_leaf2 >= -1e+06
 T781_R2_f33_ge: x_Garage_Area - 804 t781_leaf2 >= 0
 TREE_782_ONEHOT: t782_leaf0 = 1
 TREE_783_ONEHOT: t783_leaf0 = 1
 TREE_784_ONEHOT: t784_leaf0 + t784_leaf1 + t784_leaf2 + t784_leaf3
   + t784_leaf4 = 1
 T784_L0_f15_le: const_Total_Bsmt_SF + 997864 t784_leaf0 <= 1e+06
 T784_L0_f33_le: x_Garage_Area + 99748 t784_leaf0 <= 100000
 T784_L0_f3_le: Overall_Qual_calc + 999994 t784_leaf0 <= 1e+06
 T784_L1_f15_le: const_Total_Bsmt_SF + 997864 t784_leaf1 <= 1e+06
 T784_L1_f33_le: x_Garage_Area + 99748 t784_leaf1 <= 100000
 T784_R1_f3_ge: Overall_Qual_calc - 1.000006e+06 t784_leaf1 >= -1e+06
 T784_L2_f15_le: const_Total_Bsmt_SF + 997864 t784_leaf2 <= 1e+06
 T784_R2_f33_ge: x_Garage_Area - 252 t784_leaf2 >= 0
 T784_L2_f62_le: const_MS_Zoning_C_(all) + 999999 t784_leaf2 <= 1e+06
 T784_L3_f15_le: const_Total_Bsmt_SF + 997864 t784_leaf3 <= 1e+06
 T784_R3_f33_ge: x_Garage_Area - 252 t784_leaf3 >= 0
 T784_R3_f62_ge: const_MS_Zoning_C_(all) - 1.000001e+06 t784_leaf3
   >= -1e+06
 T784_R4_f15_ge: const_Total_Bsmt_SF - 1.002136e+06 t784_leaf4 >= -1e+06
 TREE_785_ONEHOT: t785_leaf0 + t785_leaf1 = 1
 T785_L0_f17_le: x_1st_Flr_SF + 4424 t785_leaf0 <= 5000
 T785_R1_f17_ge: x_1st_Flr_SF - 576 t785_leaf1 >= 0
 TREE_786_ONEHOT: t786_leaf0 = 1
 TREE_787_ONEHOT: t787_leaf0 + t787_leaf1 = 1
 T787_L0_f12_le: x_BsmtFin_SF_1 + 998035 t787_leaf0 <= 1e+06
 T787_R1_f12_ge: x_BsmtFin_SF_1 - 1965 t787_leaf1 >= 0
 TREE_788_ONEHOT: t788_leaf0 = 1
 TREE_789_ONEHOT: t789_leaf0 = 1
 TREE_790_ONEHOT: t790_leaf0 = 1
 TREE_791_ONEHOT: t791_leaf0 = 1
 TREE_792_ONEHOT: t792_leaf0 = 1
 TREE_793_ONEHOT: t793_leaf0 + t793_leaf1 + t793_leaf2 + t793_leaf3 = 1
 T793_L0_f37_le: x_Open_Porch_SF + 99740 t793_leaf0 <= 100000
 T793_L0_f12_le: x_BsmtFin_SF_1 + 998035 t793_leaf0 <= 1e+06
 T793_L1_f37_le: x_Open_Porch_SF + 99740 t793_leaf1 <= 100000
 T793_R1_f12_ge: x_BsmtFin_SF_1 - 1965 t793_leaf1 >= 0
 T793_R2_f37_ge: x_Open_Porch_SF - 260 t793_leaf2 >= 0
 T793_L2_f297_le: const_Sale_Condition_Normal + 999999 t793_leaf2 <= 1e+06
 T793_R3_f37_ge: x_Open_Porch_SF - 260 t793_leaf3 >= 0
 T793_R3_f297_ge: const_Sale_Condition_Normal - 1.000001e+06 t793_leaf3
   >= -1e+06
 TREE_794_ONEHOT: t794_leaf0 = 1
 TREE_795_ONEHOT: t795_leaf0 + t795_leaf1 + t795_leaf2 + t795_leaf3
   + t795_leaf4 = 1
 T795_L0_f5_le: const_Year_Built + 998055 t795_leaf0 <= 1e+06
 T795_L0_f4_le: const_Overall_Cond + 999993 t795_leaf0 <= 1e+06
 T795_L0_f4_le: const_Overall_Cond + 999995 t795_leaf0 <= 1e+06
 T795_L1_f5_le: const_Year_Built + 998055 t795_leaf1 <= 1e+06
 T795_L1_f4_le: const_Overall_Cond + 999993 t795_leaf1 <= 1e+06
 T795_R1_f4_ge: const_Overall_Cond - 1.000005e+06 t795_leaf1 >= -1e+06
 T795_L2_f5_le: const_Year_Built + 998055 t795_leaf2 <= 1e+06
 T795_R2_f4_ge: const_Overall_Cond - 1.000007e+06 t795_leaf2 >= -1e+06
 T795_R3_f5_ge: const_Year_Built - 1.001945e+06 t795_leaf3 >= -1e+06
 T795_L3_f231_le: central_air_no + 999999 t795_leaf3 <= 1e+06
 T795_R4_f5_ge: const_Year_Built - 1.001945e+06 t795_leaf4 >= -1e+06
 T795_R4_f231_ge: central_air_no - 1.000001e+06 t795_leaf4 >= -1e+06
 TREE_796_ONEHOT: t796_leaf0 = 1
 TREE_797_ONEHOT: t797_leaf0 = 1
 TREE_798_ONEHOT: t798_leaf0 = 1
 TREE_799_ONEHOT: t799_leaf0 = 1
 TREE_800_ONEHOT: t800_leaf0 = 1
 TREE_801_ONEHOT: t801_leaf0 = 1
 TREE_802_ONEHOT: t802_leaf0 = 1
 TREE_803_ONEHOT: t803_leaf0 = 1
 TREE_804_ONEHOT: t804_leaf0 = 1
 TREE_805_ONEHOT: t805_leaf0 + t805_leaf1 = 1
 T805_L0_f9_le: x_Exter_Cond + 2 t805_leaf0 <= 4
 T805_R1_f9_ge: x_Exter_Cond - 3 t805_leaf1 >= -1
 TREE_806_ONEHOT: t806_leaf0 + t806_leaf1 = 1
 T806_L0_f20_le: x_Gr_Liv_Area + 9064 t806_leaf0 <= 10000
 T806_R1_f20_ge: x_Gr_Liv_Area - 936 t806_leaf1 >= 0
 TREE_807_ONEHOT: t807_leaf0 = 1
 TREE_808_ONEHOT: t808_leaf0 = 1
 TREE_809_ONEHOT: t809_leaf0 = 1
 TREE_810_ONEHOT: t810_leaf0 = 1
 TREE_811_ONEHOT: t811_leaf0 = 1
 TREE_812_ONEHOT: t812_leaf0 = 1
 TREE_813_ONEHOT: t813_leaf0 = 1
 TREE_814_ONEHOT: t814_leaf0 + t814_leaf1 + t814_leaf2 + t814_leaf3
   + t814_leaf4 = 1
 T814_L0_f38_le: x_Enclosed_Porch + 99764 t814_leaf0 <= 100000
 T814_L0_f258_le: x_paved_drive_is_P <= 1
 T814_L0_f106_le: const_Neighborhood_No_aplicames + 999999 t814_leaf0
   <= 1e+06
 T814_L1_f38_le: x_Enclosed_Porch + 99764 t814_leaf1 <= 100000
 T814_L1_f258_le: x_paved_drive_is_P <= 1
 T814_R1_f106_ge: const_Neighborhood_No_aplicames - 1.000001e+06 t814_leaf1
   >= -1e+06
 T814_L2_f38_le: x_Enclosed_Porch + 99764 t814_leaf2 <= 100000
 T814_R2_f258_ge: x_paved_drive_is_P - t814_leaf2 >= 0
 T814_R3_f38_ge: x_Enclosed_Porch - 236 t814_leaf3 >= 0
 T814_L3_f1_le: const_Lot_Area + 990250 t814_leaf3 <= 1e+06
 T814_R4_f38_ge: x_Enclosed_Porch - 236 t814_leaf4 >= 0
 T814_R4_f1_ge: const_Lot_Area - 1.00975e+06 t814_leaf4 >= -1e+06
 TREE_815_ONEHOT: t815_leaf0 = 1
 TREE_816_ONEHOT: t816_leaf0 = 1
 TREE_817_ONEHOT: t817_leaf0 + t817_leaf1 + t817_leaf2 = 1
 T817_L0_f161_le: x_ex1_is_AsbShng <= 1
 T817_R1_f161_ge: x_ex1_is_AsbShng - t817_leaf1 >= 0
 T817_L1_f20_le: x_Gr_Liv_Area + 8774 t817_leaf1 <= 10000
 T817_R2_f161_ge: x_ex1_is_AsbShng - t817_leaf2 >= 0
 T817_R2_f20_ge: x_Gr_Liv_Area - 1226 t817_leaf2 >= 0
 TREE_818_ONEHOT: t818_leaf0 = 1
 TREE_819_ONEHOT: t819_leaf0 = 1
 TREE_820_ONEHOT: t820_leaf0 = 1
 TREE_821_ONEHOT: t821_leaf0 = 1
 TREE_822_ONEHOT: t822_leaf0 = 1
 TREE_823_ONEHOT: t823_leaf0 = 1
 TREE_824_ONEHOT: t824_leaf0 = 1
 TREE_825_ONEHOT: t825_leaf0 + t825_leaf1 = 1
 T825_L0_f12_le: x_BsmtFin_SF_1 + 998613 t825_leaf0 <= 1e+06
 T825_R1_f12_ge: x_BsmtFin_SF_1 - 1387 t825_leaf1 >= 0
 TREE_826_ONEHOT: t826_leaf0 = 1
 TREE_827_ONEHOT: t827_leaf0 + t827_leaf1 + t827_leaf2 + t827_leaf3 = 1
 T827_L0_f5_le: const_Year_Built + 998048 t827_leaf0 <= 1e+06
 T827_L0_f5_le: const_Year_Built + 998053 t827_leaf0 <= 1e+06
 T827_L0_f44_le: const_Yr_Sold + 997990 t827_leaf0 <= 1e+06
 T827_L1_f5_le: const_Year_Built + 998048 t827_leaf1 <= 1e+06
 T827_L1_f5_le: const_Year_Built + 998053 t827_leaf1 <= 1e+06
 T827_R1_f44_ge: const_Yr_Sold - 1.00201e+06 t827_leaf1 >= -1e+06
 T827_L2_f5_le: const_Year_Built + 998048 t827_leaf2 <= 1e+06
 T827_R2_f5_ge: const_Year_Built - 1.001947e+06 t827_leaf2 >= -1e+06
 T827_R3_f5_ge: const_Year_Built - 1.001952e+06 t827_leaf3 >= -1e+06
 TREE_828_ONEHOT: t828_leaf0 = 1
 TREE_829_ONEHOT: t829_leaf0 + t829_leaf1 + t829_leaf2 + t829_leaf3
   + t829_leaf4 = 1
 T829_L0_f17_le: x_1st_Flr_SF + 3218 t829_leaf0 <= 5000
 T829_L0_f17_le: x_1st_Flr_SF + 4376 t829_leaf0 <= 5000
 T829_L1_f17_le: x_1st_Flr_SF + 3218 t829_leaf1 <= 5000
 T829_R1_f17_ge: x_1st_Flr_SF - 624 t829_leaf1 >= 0
 T829_R2_f17_ge: x_1st_Flr_SF - 1782 t829_leaf2 >= 0
 T829_L2_f18_le: const_2nd_Flr_SF + 999044 t829_leaf2 <= 1e+06
 T829_L2_f37_le: x_Open_Porch_SF + 99967 t829_leaf2 <= 100000
 T829_R3_f17_ge: x_1st_Flr_SF - 1782 t829_leaf3 >= 0
 T829_L3_f18_le: const_2nd_Flr_SF + 999044 t829_leaf3 <= 1e+06
 T829_R3_f37_ge: x_Open_Porch_SF - 33 t829_leaf3 >= 0
 T829_R4_f17_ge: x_1st_Flr_SF - 1782 t829_leaf4 >= 0
 T829_R4_f18_ge: const_2nd_Flr_SF - 1.000956e+06 t829_leaf4 >= -1e+06
 TREE_830_ONEHOT: t830_leaf0 = 1
 TREE_831_ONEHOT: t831_leaf0 + t831_leaf1 = 1
 T831_L0_f38_le: x_Enclosed_Porch + 99735 t831_leaf0 <= 100000
 T831_R1_f38_ge: x_Enclosed_Porch - 265 t831_leaf1 >= 0
 TREE_832_ONEHOT: t832_leaf0 + t832_leaf1 = 1
 T832_L0_f3_le: Overall_Qual_calc + 999997 t832_leaf0 <= 1e+06
 T832_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t832_leaf1 >= -1e+06
 TREE_833_ONEHOT: t833_leaf0 = 1
 TREE_834_ONEHOT: t834_leaf0 = 1
 TREE_835_ONEHOT: t835_leaf0 + t835_leaf1 + t835_leaf2 = 1
 T835_L0_f231_le: central_air_no + 999999 t835_leaf0 <= 1e+06
 T835_R1_f231_ge: central_air_no - 1.000001e+06 t835_leaf1 >= -1e+06
 T835_L1_f18_le: const_2nd_Flr_SF + 999188 t835_leaf1 <= 1e+06
 T835_R2_f231_ge: central_air_no - 1.000001e+06 t835_leaf2 >= -1e+06
 T835_R2_f18_ge: const_2nd_Flr_SF - 1.000812e+06 t835_leaf2 >= -1e+06
 TREE_836_ONEHOT: t836_leaf0 = 1
 TREE_837_ONEHOT: t837_leaf0 = 1
 TREE_838_ONEHOT: t838_leaf0 = 1
 TREE_839_ONEHOT: t839_leaf0 + t839_leaf1 = 1
 T839_L0_f147_le: const_Roof_Style_Flat + 999999 t839_leaf0 <= 1e+06
 T839_R1_f147_ge: const_Roof_Style_Flat - 1.000001e+06 t839_leaf1
   >= -1e+06
 TREE_840_ONEHOT: t840_leaf0 = 1
 TREE_841_ONEHOT: t841_leaf0 + t841_leaf1 = 1
 T841_L0_f214_le: x_b1_is_LwQ <= 1
 T841_R1_f214_ge: x_b1_is_LwQ - t841_leaf1 >= 0
 TREE_842_ONEHOT: t842_leaf0 + t842_leaf1 = 1
 T842_L0_f112_le: const_Neighborhood_SawyerW + 999999 t842_leaf0 <= 1e+06
 T842_R1_f112_ge: const_Neighborhood_SawyerW - 1.000001e+06 t842_leaf1
   >= -1e+06
 TREE_843_ONEHOT: t843_leaf0 = 1
 TREE_844_ONEHOT: t844_leaf0 + t844_leaf1 = 1
 T844_L0_f33_le: x_Garage_Area + 99760 t844_leaf0 <= 100000
 T844_R1_f33_ge: x_Garage_Area - 240 t844_leaf1 >= 0
 TREE_845_ONEHOT: t845_leaf0 + t845_leaf1 + t845_leaf2 = 1
 T845_L0_f33_le: x_Garage_Area + 99356 t845_leaf0 <= 100000
 T845_R1_f33_ge: x_Garage_Area - 644 t845_leaf1 >= 0
 T845_L1_f29_le: const_Fireplaces + 999999 t845_leaf1 <= 1e+06
 T845_R2_f33_ge: x_Garage_Area - 644 t845_leaf2 >= 0
 T845_R2_f29_ge: const_Fireplaces - 1.000001e+06 t845_leaf2 >= -1e+06
 TREE_846_ONEHOT: t846_leaf0 = 1
 TREE_847_ONEHOT: t847_leaf0 = 1
 TREE_848_ONEHOT: t848_leaf0 + t848_leaf1 + t848_leaf2 = 1
 T848_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t848_leaf0 <= 1e+06
 T848_L0_f38_le: x_Enclosed_Porch + 99909 t848_leaf0 <= 100000
 T848_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t848_leaf1 <= 1e+06
 T848_R1_f38_ge: x_Enclosed_Porch - 91 t848_leaf1 >= 0
 T848_R2_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t848_leaf2
   >= -1e+06
 TREE_849_ONEHOT: t849_leaf0 + t849_leaf1 + t849_leaf2 = 1
 T849_L0_f13_le: x_BsmtFin_SF_2 + 99940 t849_leaf0 <= 100000
 T849_L0_f37_le: x_Open_Porch_SF + 99811 t849_leaf0 <= 100000
 T849_L1_f13_le: x_BsmtFin_SF_2 + 99940 t849_leaf1 <= 100000
 T849_R1_f37_ge: x_Open_Porch_SF - 189 t849_leaf1 >= 0
 T849_R2_f13_ge: x_BsmtFin_SF_2 - 60 t849_leaf2 >= 0
 TREE_850_ONEHOT: t850_leaf0 = 1
 TREE_851_ONEHOT: t851_leaf0 = 1
 TREE_852_ONEHOT: t852_leaf0 = 1
 TREE_853_ONEHOT: t853_leaf0 = 1
 TREE_854_ONEHOT: t854_leaf0 + t854_leaf1 = 1
 T854_L0_f7_le: x_Mas_Vnr_Area + 1140 t854_leaf0 <= 2000
 T854_R1_f7_ge: x_Mas_Vnr_Area - 860 t854_leaf1 >= 0
 TREE_855_ONEHOT: t855_leaf0 = 1
 TREE_856_ONEHOT: t856_leaf0 + t856_leaf1 = 1
 T856_L0_f72_le: const_Alley_Pave + 999999 t856_leaf0 <= 1e+06
 T856_R1_f72_ge: const_Alley_Pave - 1.000001e+06 t856_leaf1 >= -1e+06
 TREE_857_ONEHOT: t857_leaf0 = 1
 TREE_858_ONEHOT: t858_leaf0 + t858_leaf1 + t858_leaf2 + t858_leaf3 = 1
 T858_L0_f293_le: const_Sale_Condition_Abnorml + 999999 t858_leaf0
   <= 1e+06
 T858_L0_f21_le: const_Bsmt_Full_Bath + 999998 t858_leaf0 <= 1e+06
 T858_L1_f293_le: const_Sale_Condition_Abnorml + 999999 t858_leaf1
   <= 1e+06
 T858_R1_f21_ge: const_Bsmt_Full_Bath - 1.000002e+06 t858_leaf1 >= -1e+06
 T858_R2_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t858_leaf2
   >= -1e+06
 T858_L2_f30_le: x_Fireplace_Qu + t858_leaf2 <= 4
 T858_R3_f293_ge: const_Sale_Condition_Abnorml - 1.000001e+06 t858_leaf3
   >= -1e+06
 T858_R3_f30_ge: x_Fireplace_Qu - 4 t858_leaf3 >= -1
 TREE_859_ONEHOT: t859_leaf0 = 1
 TREE_860_ONEHOT: t860_leaf0 + t860_leaf1 + t860_leaf2 = 1
 T860_L0_f20_le: x_Gr_Liv_Area + 9253 t860_leaf0 <= 10000
 T860_L0_f35_le: x_Garage_Cond + 2 t860_leaf0 <= 4
 T860_L1_f20_le: x_Gr_Liv_Area + 9253 t860_leaf1 <= 10000
 T860_R1_f35_ge: x_Garage_Cond - 3 t860_leaf1 >= -1
 T860_R2_f20_ge: x_Gr_Liv_Area - 747 t860_leaf2 >= 0
 TREE_861_ONEHOT: t861_leaf0 = 1
 TREE_862_ONEHOT: t862_leaf0 = 1
 TREE_863_ONEHOT: t863_leaf0 = 1
 TREE_864_ONEHOT: t864_leaf0 = 1
 TREE_865_ONEHOT: t865_leaf0 + t865_leaf1 + t865_leaf2 = 1
 T865_L0_f296_le: const_Sale_Condition_Family + 999999 t865_leaf0 <= 1e+06
 T865_R1_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t865_leaf1
   >= -1e+06
 T865_L1_f12_le: x_BsmtFin_SF_1 + 999833 t865_leaf1 <= 1e+06
 T865_R2_f296_ge: const_Sale_Condition_Family - 1.000001e+06 t865_leaf2
   >= -1e+06
 T865_R2_f12_ge: x_BsmtFin_SF_1 - 167 t865_leaf2 >= 0
 TREE_866_ONEHOT: t866_leaf0 = 1
 TREE_867_ONEHOT: t867_leaf0 = 1
 TREE_868_ONEHOT: t868_leaf0 = 1
 TREE_869_ONEHOT: t869_leaf0 = 1
 TREE_870_ONEHOT: t870_leaf0 = 1
 TREE_871_ONEHOT: t871_leaf0 = 1
 TREE_872_ONEHOT: t872_leaf0 = 1
 TREE_873_ONEHOT: t873_leaf0 = 1
 TREE_874_ONEHOT: t874_leaf0 = 1
 TREE_875_ONEHOT: t875_leaf0 + t875_leaf1 = 1
 T875_L0_f20_le: x_Gr_Liv_Area + 9184 t875_leaf0 <= 10000
 T875_R1_f20_ge: x_Gr_Liv_Area - 816 t875_leaf1 >= 0
 TREE_876_ONEHOT: t876_leaf0 + t876_leaf1 + t876_leaf2 + t876_leaf3 = 1
 T876_L0_f164_le: x_ex1_is_BrkFace <= 1
 T876_L0_f35_le: x_Garage_Cond + 2 t876_leaf0 <= 4
 T876_L0_f1_le: const_Lot_Area + 991875 t876_leaf0 <= 1e+06
 T876_L1_f164_le: x_ex1_is_BrkFace <= 1
 T876_L1_f35_le: x_Garage_Cond + 2 t876_leaf1 <= 4
 T876_R1_f1_ge: const_Lot_Area - 1.008125e+06 t876_leaf1 >= -1e+06
 T876_L2_f164_le: x_ex1_is_BrkFace <= 1
 T876_R2_f35_ge: x_Garage_Cond - 3 t876_leaf2 >= -1
 T876_R3_f164_ge: x_ex1_is_BrkFace - t876_leaf3 >= 0
 TREE_877_ONEHOT: t877_leaf0 + t877_leaf1 = 1
 T877_L0_f0_le: const_Lot_Frontage + 999891 t877_leaf0 <= 1e+06
 T877_R1_f0_ge: const_Lot_Frontage - 1.000109e+06 t877_leaf1 >= -1e+06
 TREE_878_ONEHOT: t878_leaf0 = 1
 TREE_879_ONEHOT: t879_leaf0 = 1
 TREE_880_ONEHOT: t880_leaf0 + t880_leaf1 + t880_leaf2 + t880_leaf3 = 1
 T880_L0_f1_le: const_Lot_Area + 979569 t880_leaf0 <= 1e+06
 T880_R1_f1_ge: const_Lot_Area - 1.020431e+06 t880_leaf1 >= -1e+06
 T880_L1_f17_le: x_1st_Flr_SF + 3759 t880_leaf1 <= 5000
 T880_R2_f1_ge: const_Lot_Area - 1.020431e+06 t880_leaf2 >= -1e+06
 T880_R2_f17_ge: x_1st_Flr_SF - 1241 t880_leaf2 >= 0
 T880_L2_f82_le: const_Lot_Config_CulDSac + 999999 t880_leaf2 <= 1e+06
 T880_R3_f1_ge: const_Lot_Area - 1.020431e+06 t880_leaf3 >= -1e+06
 T880_R3_f17_ge: x_1st_Flr_SF - 1241 t880_leaf3 >= 0
 T880_R3_f82_ge: const_Lot_Config_CulDSac - 1.000001e+06 t880_leaf3
   >= -1e+06
 TREE_881_ONEHOT: t881_leaf0 + t881_leaf1 + t881_leaf2 + t881_leaf3
   + t881_leaf4 = 1
 T881_L0_f245_le: const_FunctioNo_aplical_Typ + 999999 t881_leaf0 <= 1e+06
 T881_L0_f44_le: const_Yr_Sold + 997991 t881_leaf0 <= 1e+06
 T881_L0_f12_le: x_BsmtFin_SF_1 + 999812 t881_leaf0 <= 1e+06
 T881_L1_f245_le: const_FunctioNo_aplical_Typ + 999999 t881_leaf1 <= 1e+06
 T881_L1_f44_le: const_Yr_Sold + 997991 t881_leaf1 <= 1e+06
 T881_R1_f12_ge: x_BsmtFin_SF_1 - 188 t881_leaf1 >= 0
 T881_L2_f245_le: const_FunctioNo_aplical_Typ + 999999 t881_leaf2 <= 1e+06
 T881_R2_f44_ge: const_Yr_Sold - 1.002009e+06 t881_leaf2 >= -1e+06
 T881_R3_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t881_leaf3
   >= -1e+06
 T881_L3_f30_le: x_Fireplace_Qu + 4 t881_leaf3 <= 4
 T881_R4_f245_ge: const_FunctioNo_aplical_Typ - 1.000001e+06 t881_leaf4
   >= -1e+06
 T881_R4_f30_ge: x_Fireplace_Qu - t881_leaf4 >= -1
 TREE_882_ONEHOT: t882_leaf0 = 1
 TREE_883_ONEHOT: t883_leaf0 + t883_leaf1 = 1
 T883_L0_f292_le: const_Sale_Type_WD_ + 999999 t883_leaf0 <= 1e+06
 T883_R1_f292_ge: const_Sale_Type_WD_ - 1.000001e+06 t883_leaf1 >= -1e+06
 TREE_884_ONEHOT: t884_leaf0 = 1
 TREE_885_ONEHOT: t885_leaf0 = 1
 TREE_886_ONEHOT: t886_leaf0 + t886_leaf1 + t886_leaf2 + t886_leaf3
   + t886_leaf4 + t886_leaf5 = 1
 T886_L0_f38_le: x_Enclosed_Porch + 99922 t886_leaf0 <= 100000
 T886_L0_f38_le: x_Enclosed_Porch + 99944 t886_leaf0 <= 100000
 T886_L0_f28_le: x_TotRms_AbvGrd + 10 t886_leaf0 <= 15
 T886_L1_f38_le: x_Enclosed_Porch + 99922 t886_leaf1 <= 100000
 T886_L1_f38_le: x_Enclosed_Porch + 99944 t886_leaf1 <= 100000
 T886_R1_f28_ge: x_TotRms_AbvGrd - 5 t886_leaf1 >= 0
 T886_L2_f38_le: x_Enclosed_Porch + 99922 t886_leaf2 <= 100000
 T886_R2_f38_ge: x_Enclosed_Porch - 56 t886_leaf2 >= 0
 T886_R3_f38_ge: x_Enclosed_Porch - 78 t886_leaf3 >= 0
 T886_L3_f49_le: const_MS_SubClass_50 + 999999 t886_leaf3 <= 1e+06
 T886_L3_f275_le: const_Mo_Sold_5 + 999999 t886_leaf3 <= 1e+06
 T886_R4_f38_ge: x_Enclosed_Porch - 78 t886_leaf4 >= 0
 T886_L4_f49_le: const_MS_SubClass_50 + 999999 t886_leaf4 <= 1e+06
 T886_R4_f275_ge: const_Mo_Sold_5 - 1.000001e+06 t886_leaf4 >= -1e+06
 T886_R5_f38_ge: x_Enclosed_Porch - 78 t886_leaf5 >= 0
 T886_R5_f49_ge: const_MS_SubClass_50 - 1.000001e+06 t886_leaf5 >= -1e+06
 TREE_887_ONEHOT: t887_leaf0 = 1
 TREE_888_ONEHOT: t888_leaf0 + t888_leaf1 = 1
 T888_L0_f4_le: const_Overall_Cond + 999993 t888_leaf0 <= 1e+06
 T888_R1_f4_ge: const_Overall_Cond - 1.000007e+06 t888_leaf1 >= -1e+06
 TREE_889_ONEHOT: t889_leaf0 + t889_leaf1 + t889_leaf2 + t889_leaf3
   + t889_leaf4 = 1
 T889_L0_f20_le: x_Gr_Liv_Area + 9117 t889_leaf0 <= 10000
 T889_L0_f29_le: const_Fireplaces + 999999 t889_leaf0 <= 1e+06
 T889_L1_f20_le: x_Gr_Liv_Area + 9117 t889_leaf1 <= 10000
 T889_R1_f29_ge: const_Fireplaces - 1.000001e+06 t889_leaf1 >= -1e+06
 T889_R2_f20_ge: x_Gr_Liv_Area - 883 t889_leaf2 >= 0
 T889_L2_f15_le: const_Total_Bsmt_SF + 997777 t889_leaf2 <= 1e+06
 T889_L2_f15_le: const_Total_Bsmt_SF + 998864 t889_leaf2 <= 1e+06
 T889_R3_f20_ge: x_Gr_Liv_Area - 883 t889_leaf3 >= 0
 T889_L3_f15_le: const_Total_Bsmt_SF + 997777 t889_leaf3 <= 1e+06
 T889_R3_f15_ge: const_Total_Bsmt_SF - 1.001136e+06 t889_leaf3 >= -1e+06
 T889_R4_f20_ge: x_Gr_Liv_Area - 883 t889_leaf4 >= 0
 T889_R4_f15_ge: const_Total_Bsmt_SF - 1.002223e+06 t889_leaf4 >= -1e+06
 TREE_890_ONEHOT: t890_leaf0 = 1
 TREE_891_ONEHOT: t891_leaf0 = 1
 TREE_892_ONEHOT: t892_leaf0 = 1
 TREE_893_ONEHOT: t893_leaf0 = 1
 TREE_894_ONEHOT: t894_leaf0 + t894_leaf1 = 1
 T894_L0_f33_le: x_Garage_Area + 99129 t894_leaf0 <= 100000
 T894_R1_f33_ge: x_Garage_Area - 871 t894_leaf1 >= 0
 TREE_895_ONEHOT: t895_leaf0 = 1
 TREE_896_ONEHOT: t896_leaf0 + t896_leaf1 = 1
 T896_L0_f29_le: const_Fireplaces + 999998 t896_leaf0 <= 1e+06
 T896_R1_f29_ge: const_Fireplaces - 1.000002e+06 t896_leaf1 >= -1e+06
 TREE_897_ONEHOT: t897_leaf0 = 1
 TREE_898_ONEHOT: t898_leaf0 = 1
 TREE_899_ONEHOT: t899_leaf0 = 1
 TREE_900_ONEHOT: t900_leaf0 = 1
 TREE_901_ONEHOT: t901_leaf0 = 1
 TREE_902_ONEHOT: t902_leaf0 = 1
 TREE_903_ONEHOT: t903_leaf0 + t903_leaf1 + t903_leaf2 = 1
 T903_L0_f15_le: const_Total_Bsmt_SF + 999517 t903_leaf0 <= 1e+06
 T903_R1_f15_ge: const_Total_Bsmt_SF - 1.000483e+06 t903_leaf1 >= -1e+06
 T903_L1_f37_le: x_Open_Porch_SF + 99712 t903_leaf1 <= 100000
 T903_R2_f15_ge: const_Total_Bsmt_SF - 1.000483e+06 t903_leaf2 >= -1e+06
 T903_R2_f37_ge: x_Open_Porch_SF - 288 t903_leaf2 >= 0
 TREE_904_ONEHOT: t904_leaf0 + t904_leaf1 = 1
 T904_L0_f3_le: Overall_Qual_calc + 999997 t904_leaf0 <= 1e+06
 T904_R1_f3_ge: Overall_Qual_calc - 1.000003e+06 t904_leaf1 >= -1e+06
 TREE_905_ONEHOT: t905_leaf0 = 1
 TREE_906_ONEHOT: t906_leaf0 + t906_leaf1 + t906_leaf2 = 1
 T906_L0_f273_le: const_Mo_Sold_3 + 999999 t906_leaf0 <= 1e+06
 T906_L0_f32_le: const_Garage_Cars + 999997 t906_leaf0 <= 1e+06
 T906_L1_f273_le: const_Mo_Sold_3 + 999999 t906_leaf1 <= 1e+06
 T906_R1_f32_ge: const_Garage_Cars - 1.000003e+06 t906_leaf1 >= -1e+06
 T906_R2_f273_ge: const_Mo_Sold_3 - 1.000001e+06 t906_leaf2 >= -1e+06
 TREE_907_ONEHOT: t907_leaf0 = 1
 TREE_908_ONEHOT: t908_leaf0 = 1
 TREE_909_ONEHOT: t909_leaf0 + t909_leaf1 = 1
 T909_L0_f5_le: const_Year_Built + 998089 t909_leaf0 <= 1e+06
 T909_R1_f5_ge: const_Year_Built - 1.001911e+06 t909_leaf1 >= -1e+06
 TREE_910_ONEHOT: t910_leaf0 = 1
 TREE_911_ONEHOT: t911_leaf0 = 1
 TREE_912_ONEHOT: t912_leaf0 = 1
 TREE_913_ONEHOT: t913_leaf0 = 1
 YLOG_XGB_SUM: y_log_raw + 0.0227153655 t0_leaf0 + 0.0109316884 t0_leaf1
   + 0.00458499743 t0_leaf2 - 0.010366804 t0_leaf3 - 0.00269366521 t0_leaf4
   - 0.0123194279 t0_leaf5 - 0.0137853688 t0_leaf6 - 0.0280528124 t0_leaf7
   + 0.0257327575 t1_leaf0 + 0.012289213 t1_leaf1 + 0.0106788166 t1_leaf2
   + 0.00107296975 t1_leaf3 - 0.00125729048 t1_leaf4
   - 0.0100781415 t1_leaf5 - 0.0159680694 t1_leaf6 - 0.0299721193 t1_leaf7
   + 0.02208459 t2_leaf0 + 0.00984619744 t2_leaf1 + 0.00488707656 t2_leaf2
   - 0.00307397358 t2_leaf3 + 4.70025e-05 t2_leaf4 - 0.0104035893 t2_leaf5
   - 0.0102329962 t2_leaf6 - 0.0241005123 t2_leaf7 + 0.0244713817 t3_leaf0
   + 0.0123096956 t3_leaf1 + 0.00931277592 t3_leaf2 + 0.000628208 t3_leaf3
   - 0.00212428719 t3_leaf4 - 0.0101810889 t3_leaf5 - 0.0167293586 t3_leaf6
   - 0.0306091532 t3_leaf7 + 0.0118720299 t4_leaf0 + 0.0308137815 t4_leaf1
   + 0.00745445536 t4_leaf2 - 5.1375233999999997e-04 t4_leaf3
   - 0.00407241331 t4_leaf4 - 0.0137299579 t4_leaf5 - 0.0126338294 t4_leaf6
   - 0.0269263107 t4_leaf7 + 0.0214245245 t5_leaf0 + 0.006503365 t5_leaf1
   + 0.0094019305 t5_leaf2 + 0.00117268786 t5_leaf3
   - 8.9457724200000003e-04 t5_leaf4 - 0.00873220619 t5_leaf5
   - 0.0165350884 t5_leaf6 - 0.0269878004 t5_leaf7 + 0.0258362964 t6_leaf0
   + 0.00968578085 t6_leaf1 + 0.00945897959 t6_leaf2
   + 0.00201151217 t6_leaf3 - 2.7977881799999999e-04 t6_leaf4
   - 0.00826565176 t6_leaf5 - 0.0145302918 t6_leaf6 - 0.0263090581 t6_leaf7
   + 0.0316822976 t7_leaf0 + 0.0145802516 t7_leaf1 + 0.00734207034 t7_leaf2
   - 7.7541382100000005e-04 t7_leaf3 - 0.00437049242 t7_leaf4
   - 0.0136460699 t7_leaf5 - 0.0134710008 t7_leaf6 - 0.0235562176 t7_leaf7
   + 0.0089117866 t8_leaf0 + 0.020513352 t8_leaf1 + 0.00342838 t8_leaf2
   - 0.00765790418 t8_leaf3 - 0.000545633 t8_leaf4 - 0.00960193388 t8_leaf5
   - 0.0156558808 t8_leaf6 - 0.0258980691 t8_leaf7 + 0.00848270953 t9_leaf0
   + 0.0215684231 t9_leaf1 + 0.00273803 t9_leaf2 - 0.00590120209 t9_leaf3
   - 0.00314983702 t9_leaf4 - 0.0100015 t9_leaf5 - 0.0130673386 t9_leaf6
   - 0.0223948359 t9_leaf7 + 0.0277508087 t10_leaf0
   + 0.0141743738 t10_leaf1 + 0.00870797224 t10_leaf2
   + 0.00262025231 t10_leaf3 + 0.00703274645 t10_leaf4
   - 0.00118900684 t10_leaf5 - 0.00884083938 t10_leaf6
   - 0.0196685623 t10_leaf7 + 0.0221577026 t11_leaf0
   + 0.00698288763 t11_leaf1 + 0.00663986569 t11_leaf2
   - 1.16937277e-04 t11_leaf3 - 0.00215779711 t11_leaf4
   - 0.00982607622 t11_leaf5 - 0.00888452493 t11_leaf6
   - 0.0192282423 t11_leaf7 + 0.0192659479 t12_leaf0
   + 0.0064635952 t12_leaf1 + 0.00787145365 t12_leaf2
   + 0.00151863252 t12_leaf3 - 0.00502124149 t12_leaf4
   - 0.0146028595 t12_leaf5 - 0.0139762852 t12_leaf6
   - 0.0237245914 t12_leaf7 + 0.0225325394 t13_leaf0
   + 0.00886209123 t13_leaf1 + 0.00353924115 t13_leaf2
   - 0.0039666486 t13_leaf3 + 0.00126351533 t13_leaf4
   - 0.00731827179 t13_leaf5 - 0.00989489164 t13_leaf6
   - 0.0186696183 t13_leaf7 + 0.0274305008 t14_leaf0
   + 0.0118557578 t14_leaf1 + 0.00622334797 t14_leaf2
   - 0.00126592466 t14_leaf3 + 8.0659211400000005e-04 t14_leaf4
   - 0.00815387443 t14_leaf5 - 0.00930652022 t14_leaf6
   - 0.0186197665 t14_leaf7 + 0.00878201984 t15_leaf0
   + 0.0191700589 t15_leaf1 + 0.00606038515 t15_leaf2
   - 0.000511884 t15_leaf3 - 0.00604679575 t15_leaf4
   + 0.00104142074 t15_leaf5 - 0.0128101651 t15_leaf6
   - 0.0219680332 t15_leaf7 + 0.033626996 t16_leaf0
   + 0.0122924922 t16_leaf1 + 0.00504348194 t16_leaf2
   - 0.00162049884 t16_leaf3 - 0.00309862709 t16_leaf4
   - 0.0100407386 t16_leaf5 - 0.0130267804 t16_leaf6 - 0.02132928 t16_leaf7
   + 0.0325403064 t17_leaf0 + 0.0117814485 t17_leaf1
   + 0.00543196313 t17_leaf2 + 4.59072879e-04 t17_leaf3
   - 0.00167758879 t17_leaf4 - 0.00777553162 t17_leaf5
   - 0.0174500383 t17_leaf6 + 0.00633081142 t18_leaf0
   + 3.1245258300000001e-04 t18_leaf1 + 0.018252641 t18_leaf2
   + 0.00481385505 t18_leaf3 - 8.26768344e-04 t18_leaf4
   - 0.00854970608 t18_leaf5 - 0.0102690775 t18_leaf6
   - 0.0177027192 t18_leaf7 + 0.0195962638 t19_leaf0
   + 0.00677934196 t19_leaf1 + 0.00220389059 t19_leaf2
   - 0.00740419654 t19_leaf3 - 0.00260248478 t19_leaf4
   - 0.0102004157 t19_leaf5 - 0.00938276388 t19_leaf6
   - 0.0184724126 t19_leaf7 + 0.0188416783 t20_leaf0
   + 0.0104141226 t20_leaf1 + 0.00697504822 t20_leaf2
   + 0.00121598167 t20_leaf3 + 0.00344684091 t20_leaf4
   - 0.00370885315 t20_leaf5 - 0.00951877516 t20_leaf6
   - 0.0169056635 t20_leaf7 + 0.0196709931 t21_leaf0
   + 0.0081847189 t21_leaf1 + 0.0044476781 t21_leaf2
   - 0.00141097861 t21_leaf3 - 0.00209141336 t21_leaf4
   - 0.00728301657 t21_leaf5 - 0.0103179729 t21_leaf6
   - 0.0178762339 t21_leaf7 + 0.0130721973 t22_leaf0
   + 0.00586968102 t22_leaf1 + 0.00712332316 t22_leaf2
   - 4.61305e-05 t22_leaf3 + 0.00270687207 t22_leaf4
   - 0.00609189086 t22_leaf5 - 0.00470262952 t22_leaf6
   - 0.0117900008 t22_leaf7 + 0.0241041016 t23_leaf0
   + 0.00399510749 t23_leaf1 + 0.00963845942 t23_leaf2
   + 0.00364134507 t23_leaf3 - 0.00110748026 t23_leaf4
   - 0.00853515882 t23_leaf5 - 0.00907738693 t23_leaf6
   - 0.0155859552 t23_leaf7 + 0.0106579522 t24_leaf0
   + 0.00428464776 t24_leaf1 + 0.00870386045 t24_leaf2
   - 9.9255e-05 t24_leaf3 + 0.000860073 t24_leaf4 - 0.00440375926 t24_leaf5
   - 0.00689919759 t24_leaf6 - 0.0144275175 t24_leaf7
   + 0.0166552104 t25_leaf0 + 0.00561338896 t25_leaf1
   + 0.00518423039 t25_leaf2 - 3.2448841400000003e-04 t25_leaf3
   - 0.00109107571 t25_leaf4 - 0.00792970601 t25_leaf5
   - 0.0045954017 t25_leaf6 - 0.0124895899 t25_leaf7
   + 0.00617979141 t26_leaf0 + 0.00120182964 t26_leaf1
   + 0.0210638065 t26_leaf2 + 0.00887472183 t26_leaf3
   + 0.00159580633 t26_leaf4 - 0.00377549301 t26_leaf5
   - 0.00729638292 t26_leaf6 - 0.0145469727 t26_leaf7
   + 0.00489378301 t27_leaf0 - 4.9468432600000001e-04 t27_leaf1
   + 0.0157187432 t27_leaf2 + 0.00481182523 t27_leaf3
   + 0.00166690641 t27_leaf4 - 0.00311546191 t27_leaf5
   - 0.00613677455 t27_leaf6 - 0.0116045447 t27_leaf7
   + 0.0187104642 t28_leaf0 + 0.00741076889 t28_leaf1
   + 0.00565502374 t28_leaf2 + 3.3919763500000002e-04 t28_leaf3
   - 3.66133201e-04 t28_leaf4 - 0.0058907 t28_leaf5
   - 0.00767616974 t28_leaf6 - 0.0140224807 t28_leaf7
   + 0.0127238724 t29_leaf0 + 0.0300854705 t29_leaf1
   + 0.00493100937 t29_leaf2 - 7.28432e-05 t29_leaf3
   - 0.00213713478 t29_leaf4 - 0.00753479823 t29_leaf5
   - 0.00429897569 t29_leaf6 - 0.0106065618 t29_leaf7
   + 0.0130724311 t30_leaf0 + 0.000492796 t30_leaf1
   + 0.00796774589 t30_leaf2 + 0.00256570778 t30_leaf3
   + 0.00279190508 t30_leaf4 - 0.0026802551 t30_leaf5
   - 0.0071412297 t30_leaf6 - 0.0154138179 t30_leaf7
   + 0.0136442063 t31_leaf0 + 0.00502445176 t31_leaf1
   + 0.00457269605 t31_leaf2 + 2.11014863e-04 t31_leaf3
   - 0.00304262736 t31_leaf4 + 0.00594458589 t31_leaf5
   - 0.00728029804 t31_leaf6 - 0.0127771311 t31_leaf7
   + 0.0117484964 t32_leaf0 + 0.00496501196 t32_leaf1
   + 0.00283318176 t32_leaf2 + 0.0104719298 t32_leaf3
   + 3.7698636899999999e-04 t32_leaf4 - 0.00623146445 t32_leaf5
   - 0.00143284304 t32_leaf6 - 0.00855198875 t32_leaf7
   + 0.00918977335 t33_leaf0 + 0.00278635556 t33_leaf1
   - 0.002842488 t33_leaf2 + 0.00327161793 t33_leaf3
   - 0.005830999 t33_leaf4 - 0.00105623575 t33_leaf5
   - 0.0125997476 t33_leaf6 - 0.00234522973 t33_leaf7
   + 0.011665591 t34_leaf0 - 1.6888847999999999e-04 t34_leaf1
   + 0.00336432108 t34_leaf2 - 0.00126298319 t34_leaf3
   + 0.00412542606 t34_leaf4 - 0.00121779356 t34_leaf5
   - 0.00205302821 t34_leaf6 - 0.00703571457 t34_leaf7
   + 0.00844715722 t35_leaf0 + 0.00257450715 t35_leaf1
   + 0.0043825726 t35_leaf2 + 3.8339942700000001e-04 t35_leaf3
   - 6.98630814e-04 t35_leaf4 - 0.00462945038 t35_leaf5
   - 0.00643907581 t35_leaf6 - 0.0185931567 t35_leaf7
   + 0.0215764325 t36_leaf0 + 0.0040037795 t36_leaf1
   + 0.00556176435 t36_leaf2 + 8.1962760300000005e-04 t36_leaf3
   + 0.000127487 t36_leaf4 - 0.00550210755 t36_leaf5
   - 0.00624468969 t36_leaf6 - 0.0121824732 t36_leaf7
   + 0.0216833055 t37_leaf0 + 0.00404957728 t37_leaf1
   + 0.0045885928 t37_leaf2 + 0.000298595 t37_leaf3
   + 0.00412288681 t37_leaf4 - 0.00167904631 t37_leaf5 - 0.002872 t37_leaf6
   - 0.00819734484 t37_leaf7 + 0.0223066155 t38_leaf0
   + 0.00783404242 t38_leaf1 + 0.0017124014 t38_leaf2
   + 0.00551982131 t38_leaf3 + 0.00288653257 t38_leaf4
   - 0.00144843012 t38_leaf5 - 0.00463070674 t38_leaf6
   - 0.00932648312 t38_leaf7 + 0.0118375253 t39_leaf0
   + 0.00443984661 t39_leaf1 + 0.00377490488 t39_leaf2
   + 6.2661047600000004e-04 t39_leaf3 + 0.00727402512 t39_leaf4
   - 0.0015061337 t39_leaf5 - 0.00533777429 t39_leaf6
   - 0.0111084916 t39_leaf7 + 0.0114943217 t40_leaf0
   + 0.00330723636 t40_leaf1 + 0.00426313 t40_leaf2
   + 2.1022485500000001e-04 t40_leaf3 + 0.001674289 t40_leaf4
   - 0.0026396485 t40_leaf5 - 0.00464140065 t40_leaf6
   - 0.0132546667 t40_leaf7 + 0.00552448304 t41_leaf0
   + 6.59313169e-04 t41_leaf1 + 0.0129046207 t41_leaf2
   + 0.00264035817 t41_leaf3 - 0.0018652149 t41_leaf4
   - 0.00722075673 t41_leaf5 - 0.011348038 t41_leaf6
   + 0.00206290511 t41_leaf7 + 0.0167772565 t42_leaf0
   + 0.0066267834 t42_leaf1 + 0.00158261228 t42_leaf2
   - 0.00250198576 t42_leaf3 - 3.1887e-05 t42_leaf4
   - 0.00438540801 t42_leaf5 - 0.0101558948 t42_leaf6
   - 2.00129e-05 t42_leaf7 + 0.0113324579 t43_leaf0
   + 0.00410116185 t43_leaf1 + 0.00276523875 t43_leaf2
   - 0.00125339138 t43_leaf3 + 2.03384e-05 t43_leaf4
   - 0.0023150458 t43_leaf5 - 0.00568786589 t43_leaf6
   + 0.00415098574 t43_leaf7 + 0.00910151284 t44_leaf0
   + 0.00383293 t44_leaf1 + 0.00838313438 t44_leaf2
   + 0.00111966394 t44_leaf3 + 6.8685371700000003e-04 t44_leaf4
   - 0.0025440969 t44_leaf5 - 0.00495358044 t44_leaf6
   - 0.0103776949 t44_leaf7 + 0.00505575724 t45_leaf0
   + 0.0133783706 t45_leaf1 + 0.00235957187 t45_leaf2
   - 0.00308252987 t45_leaf3 + 0.0025718275 t45_leaf4
   - 0.0010436374 t45_leaf5 - 0.00292644696 t45_leaf6
   - 0.00666281069 t45_leaf7 + 0.00606718473 t46_leaf0
   + 0.0139334304 t46_leaf1 + 0.00274072331 t46_leaf2
   - 0.00140766613 t46_leaf3 - 1.23683421e-04 t46_leaf4
   - 0.00369302696 t46_leaf5 - 0.0123702735 t46_leaf6
   - 0.00301604951 t46_leaf7 + 0.0103797326 t47_leaf0
   + 0.00405490818 t47_leaf1 + 0.00373325124 t47_leaf2
   - 1.47079e-05 t47_leaf3 + 0.0017519109 t47_leaf4
   - 0.00119341386 t47_leaf5 - 0.00297605479 t47_leaf6
   - 0.00741331046 t47_leaf7 + 0.00300765014 t48_leaf0
   - 2.5583570799999998e-04 t48_leaf1 + 0.017837951 t48_leaf2
   + 0.0059100776 t48_leaf3 - 0.00412723 t48_leaf4 - 4.90748e-05 t48_leaf5
   - 0.00476859184 t48_leaf6 - 0.0105427271 t48_leaf7
   + 0.00625494123 t49_leaf0 + 0.0207809079 t49_leaf1
   + 0.00286215893 t49_leaf2 - 1.93393891e-04 t49_leaf3
   - 0.00456466852 t49_leaf4 - 0.00116436416 t49_leaf5
   + 0.00281661237 t49_leaf6 + 0.00884123892 t49_leaf7
   + 0.0128198247 t50_leaf0 + 0.00562385656 t50_leaf1
   + 0.00149687182 t50_leaf2 + 0.00145047752 t50_leaf3
   - 0.00191480585 t50_leaf4 - 0.00130254554 t50_leaf5
   - 0.00572970975 t50_leaf6 + 0.0133413933 t51_leaf0
   + 0.00468616839 t51_leaf1 + 6.5414927700000001e-04 t51_leaf2
   + 1.8289328699999999e-04 t51_leaf3 - 0.00304884836 t51_leaf4
   - 0.00831197295 t51_leaf5 + 0.00446023932 t51_leaf6
   + 0.0102805989 t52_leaf0 + 0.00449622981 t52_leaf1
   + 0.0010973023 t52_leaf2 - 0.00289466628 t52_leaf3
   - 0.00134074665 t52_leaf4 - 0.00471787807 t52_leaf5
   - 0.00390934804 t52_leaf6 - 0.0110397693 t52_leaf7
   + 0.0137361148 t53_leaf0 + 0.00302655576 t53_leaf1
   + 0.00305952691 t53_leaf2 - 2.29396552e-04 t53_leaf3
   + 0.0121354312 t53_leaf4 - 0.00170094613 t53_leaf5
   - 0.00812260713 t53_leaf6 + 0.0118717821 t54_leaf0
   + 3.7218356700000001e-04 t54_leaf1 + 0.00659488374 t54_leaf2
   + 0.0139810806 t54_leaf3 + 0.00348631665 t54_leaf4
   - 6.7772343799999996e-04 t54_leaf5 - 0.0063156914 t54_leaf6
   + 0.0167653784 t55_leaf0 + 0.00640684459 t55_leaf1
   + 0.00477095274 t55_leaf2 + 6.8369280799999998e-04 t55_leaf3
   - 6.7105662299999996e-04 t55_leaf4 - 0.00455359 t55_leaf5
   - 0.00631834986 t55_leaf6 + 2.00520837e-04 t55_leaf7
   + 0.0051939562 t56_leaf0 + 0.00217358186 t56_leaf1
   + 0.00191095902 t56_leaf2 - 0.00131825136 t56_leaf3
   - 0.000650858 t56_leaf4 - 0.00426714588 t56_leaf5
   - 0.00665138615 t56_leaf6 + 0.00171647721 t56_leaf7
   + 0.00746972952 t57_leaf0 + 0.00238190661 t57_leaf1
   + 0.002080312 t57_leaf2 - 4.1165237799999999e-04 t57_leaf3
   + 0.00140763901 t57_leaf4 - 0.00196782686 t57_leaf5
   - 0.00520120608 t57_leaf6 - 0.0133087216 t57_leaf7
   + 0.00952006876 t58_leaf0 + 0.00484250579 t58_leaf1
   + 0.00562144769 t58_leaf2 + 9.891671360000001e-04 t58_leaf3
   + 0.0013535422 t58_leaf4 - 0.00116737094 t58_leaf5
   - 0.00770844752 t58_leaf6 - 0.00276972773 t58_leaf7
   + 0.00428174948 t59_leaf0 + 0.00959212054 t59_leaf1
   + 0.00258154864 t59_leaf2 - 0.00209887489 t59_leaf3
   + 0.00171466765 t59_leaf4 - 0.00122465508 t59_leaf5
   - 0.00219453918 t59_leaf6 - 0.00580362277 t59_leaf7
   + 0.0137389749 t60_leaf0 - 9.12745e-05 t60_leaf1
   + 0.00242593791 t60_leaf2 + 5.5487582000000001e-04 t60_leaf3
   - 0.00191126345 t60_leaf4 - 0.00447982782 t60_leaf5
   - 0.0106911445 t60_leaf6 + 0.0166750643 t61_leaf0
   + 0.00588289043 t61_leaf1 - 7.68841e-05 t61_leaf2
   + 0.00399805512 t61_leaf3 + 0.000492941 t61_leaf4 - 0.00135926 t61_leaf5
   - 0.00517891441 t61_leaf6 + 0.00278435485 t62_leaf0
   + 8.05036e-06 t62_leaf1 + 4.8905145400000003e-04 t62_leaf2
   - 0.0020591733 t62_leaf3 - 0.00273880013 t62_leaf4
   - 0.00738351699 t62_leaf5 + 0.00123838743 t62_leaf6
   + 0.0139392773 t63_leaf0 + 0.00307171978 t63_leaf1
   - 5.6445144599999995e-04 t63_leaf2 + 0.00348932249 t63_leaf3
   - 0.00120525237 t63_leaf4 + 0.00358211459 t63_leaf5
   - 0.011650878 t63_leaf6 - 0.00175457832 t63_leaf7
   + 0.00906343479 t64_leaf0 + 0.00216957089 t64_leaf1
   + 0.00260281283 t64_leaf2 + 7.45388e-06 t64_leaf3
   + 6.01040665e-04 t64_leaf4 - 0.00208192132 t64_leaf5
   - 0.00478206389 t64_leaf6 - 0.0122264596 t64_leaf7
   + 0.00363004324 t65_leaf0 + 0.0149856992 t65_leaf1
   + 0.00504682679 t65_leaf2 + 0.00130402483 t65_leaf3
   + 0.00190944399 t65_leaf4 - 0.000804587 t65_leaf5
   - 0.000859871 t65_leaf6 - 0.00730189588 t65_leaf7
   - 0.00129302987 t66_leaf0 - 0.00543237804 t66_leaf1
   + 0.0030434099 t66_leaf2 - 2.58147833e-04 t66_leaf3
   + 0.00397123164 t66_leaf4 - 0.00184395711 t66_leaf5
   + 0.0130365873 t66_leaf6 + 0.00392097048 t66_leaf7
   + 0.0201685596 t67_leaf0 + 0.00411708979 t67_leaf1
   - 4.32555651e-04 t67_leaf2 + 0.00268459343 t67_leaf3
   - 8.7189522999999998e-04 t67_leaf4 - 0.00441960758 t67_leaf5
   + 0.00331276678 t68_leaf0 + 5.3622655099999997e-04 t68_leaf1
   + 3.8457987799999998e-04 t68_leaf2 + 0.00598658575 t68_leaf3
   - 0.00162188488 t68_leaf4 - 0.00564902 t68_leaf5
   + 0.00114679604 t68_leaf6 - 0.00125117938 t68_leaf7
   + 0.00617063232 t69_leaf0 + 4.0900977899999998e-04 t69_leaf1
   - 8.4713159599999999e-04 t69_leaf2 + 0.00203201384 t69_leaf3
   - 5.0446041900000004e-04 t69_leaf4 + 0.00140916137 t69_leaf5
   - 0.0043247398 t69_leaf6 + 0.0017048649 t70_leaf0
   + 0.0063395747 t70_leaf1 - 1.97213478e-04 t70_leaf2
   + 0.00210716901 t70_leaf3 - 0.00111249671 t70_leaf4
   - 0.00610058 t70_leaf5 + 0.000328411 t70_leaf6 + 0.00390894292 t70_leaf7
   + 0.00365128741 t71_leaf0 + 0.0166816153 t71_leaf1
   + 0.00154998933 t71_leaf2 - 5.61618421e-04 t71_leaf3
   + 0.0019131751 t71_leaf4 - 0.00305133243 t71_leaf5
   - 0.00337120914 t71_leaf6 - 0.0137052732 t71_leaf7
   - 0.000894219 t72_leaf0 + 0.00323132798 t72_leaf1
   - 9.9367741499999996e-04 t72_leaf2 - 0.00468653347 t72_leaf3
   + 0.00752118 t72_leaf4 + 0.00180251023 t72_leaf5
   + 2.16908869e-04 t72_leaf6 - 0.00418398902 t72_leaf7
   + 0.00169463211 t73_leaf0 + 0.00151673495 t73_leaf1
   + 0.0116114831 t73_leaf2 + 1.89168117e-04 t73_leaf3
   - 0.00488520134 t73_leaf4 - 0.00175018841 t73_leaf5
   - 0.00896445382 t73_leaf6 - 0.000846754 t74_leaf0
   + 0.0013139901 t74_leaf1 + 0.00843074173 t74_leaf2
   - 6.8333034900000002e-04 t74_leaf4 - 0.00364214228 t74_leaf5
   - 0.00901089888 t74_leaf6 + 0.0145026492 t75_leaf0
   + 0.00583441183 t75_leaf1 + 0.000311075 t75_leaf2
   + 0.00396964606 t75_leaf3 + 1.15244882e-04 t75_leaf4
   - 5.41288e-05 t75_leaf5 - 0.00320201 t75_leaf6
   + 3.71103437e-04 t76_leaf0 + 0.00804256927 t76_leaf1
   + 0.00229042652 t76_leaf2 - 0.000584338 t76_leaf3
   - 0.0019381796 t76_leaf4 - 0.0060423594 t76_leaf5
   - 6.4304471000000004e-04 t76_leaf6 + 0.00686827907 t76_leaf7
   + 0.00310458755 t77_leaf0 + 0.0084863361 t77_leaf1
   - 0.00107428979 t77_leaf2 - 5.64173e-05 t77_leaf3
   - 0.00262625772 t77_leaf4 + 0.00188998773 t77_leaf5
   - 6.0425448500000005e-04 t77_leaf6 + 0.0037126604 t78_leaf0
   - 0.00103696983 t78_leaf1 + 0.00254035578 t78_leaf2
   - 0.00393573428 t78_leaf3 + 0.00230369112 t78_leaf4
   - 2.1204933e-04 t78_leaf5 + 0.0131892404 t78_leaf6
   + 0.0032081469 t78_leaf7 + 0.0118674766 t79_leaf0
   + 0.00278372038 t79_leaf1 + 0.00196123403 t79_leaf2
   + 0.00104786688 t79_leaf3 - 0.000998914 t79_leaf4
   - 1.6466221099999999e-04 t79_leaf5 - 0.00537316827 t79_leaf6
   + 0.00408427278 t80_leaf0 + 0.0115598226 t80_leaf1
   - 0.00346721523 t80_leaf2 + 0.00155047537 t80_leaf3
   - 5.3645472500000004e-04 t80_leaf4 - 0.00145784079 t80_leaf5
   - 0.00435985625 t80_leaf6 + 0.0126114096 t81_leaf0
   + 0.00317987264 t81_leaf1 + 2.9505565200000001e-04 t81_leaf2
   - 8.1865757199999997e-04 t81_leaf3 - 0.00478470093 t81_leaf4
   + 0.00138328411 t82_leaf0 - 5.13060833e-04 t82_leaf1
   + 0.00596186705 t82_leaf2 - 3.4265813900000001e-04 t82_leaf3
   - 0.00139608781 t82_leaf4 - 0.00507387379 t82_leaf5
   - 0.00607924769 t82_leaf6 + 0.0161681436 t83_leaf0
   + 0.00318679377 t83_leaf1 + 2.7705565999999997e-04 t83_leaf2
   - 0.000813273 t83_leaf3 - 0.00479911407 t83_leaf4
   + 0.00569548365 t84_leaf0 - 6.8245083099999996e-04 t84_leaf1
   - 1.0185119800000001e-04 t84_leaf2 + 0.00200898363 t84_leaf3
   + 0.00741924066 t84_leaf4 - 4.3529251699999999e-04 t84_leaf5
   - 0.00559918536 t84_leaf6 + 0.0024415676 t85_leaf0
   - 1.9678450200000001e-04 t85_leaf1 + 0.00990337133 t85_leaf2
   + 0.00253942795 t85_leaf3 - 0.00175697659 t85_leaf4
   - 0.0054518478 t85_leaf5 + 0.000706311 t85_leaf6
   + 0.00417325553 t86_leaf0 - 7.7961053499999995e-04 t86_leaf1
   - 5.26916e-05 t86_leaf2 - 0.00265474245 t86_leaf3
   + 0.00421062391 t86_leaf4 + 0.0128922788 t86_leaf5
   + 0.0023644492 t86_leaf6 + 5.2344531300000004e-04 t87_leaf0
   + 0.0101360129 t87_leaf1 + 0.00182129955 t87_leaf2
   - 0.00143441337 t87_leaf3 - 0.0113042705 t87_leaf4
   + 0.00276995986 t87_leaf5 + 1.06550309e-04 t87_leaf6
   + 0.0104326783 t88_leaf0 + 0.00349468016 t88_leaf1
   - 6.0304388199999999e-04 t88_leaf2 - 9.9965173299999995e-04 t88_leaf3
   - 0.00575711764 t88_leaf4 + 0.00302681676 t88_leaf5
   + 5.54507e-05 t88_leaf6 + 0.0029954412 t89_leaf0
   - 0.00184305757 t89_leaf1 + 0.0197300445 t89_leaf2
   + 0.00545269577 t89_leaf3 - 4.6298e-06 t89_leaf4
   + 0.00453223055 t89_leaf5 - 0.00165447441 t89_leaf6
   - 0.00513537228 t89_leaf7 + 0.0023959598 t90_leaf0
   - 3.6118688900000002e-04 t90_leaf1 - 0.00135328935 t90_leaf2
   - 0.00525943935 t90_leaf3 + 0.00132536911 t90_leaf4
   - 4.8496131799999998e-04 t90_leaf5 + 9.7724120100000007e-04 t90_leaf6
   + 0.00664515 t90_leaf7 + 0.00519820862 t91_leaf0
   + 0.00132001971 t91_leaf1 + 0.00183331547 t91_leaf2
   - 3.82477243e-04 t91_leaf3 + 0.00139598083 t91_leaf4
   - 6.2231312000000005e-04 t91_leaf5 - 0.00298366975 t91_leaf6
   + 0.00321752625 t91_leaf7 + 0.00538262213 t92_leaf0
   + 5.6847970699999998e-04 t92_leaf1 + 0.0131767411 t92_leaf2
   - 0.00115153077 t92_leaf3 + 9.3280977999999997e-04 t92_leaf4
   - 0.00817385502 t92_leaf5 - 0.00200225553 t92_leaf6
   + 0.00434960192 t93_leaf0 - 4.6701548899999998e-04 t93_leaf1
   + 0.00272078183 t93_leaf2 - 0.00353141618 t93_leaf3
   + 0.00255698874 t93_leaf4 + 1.8024683199999999e-04 t93_leaf5
   + 0.00225737621 t93_leaf6 + 0.00874399487 t93_leaf7
   + 0.00269197836 t94_leaf0 + 3.02250788e-04 t94_leaf1
   - 0.0042550033 t94_leaf2 - 0.00142196496 t94_leaf3
   - 0.00210566702 t94_leaf4 - 0.00612526154 t94_leaf5
   + 0.00509647606 t94_leaf6 + 0.00824853778 t95_leaf0
   + 0.00248237513 t95_leaf1 + 0.00287000346 t95_leaf2
   + 7.9185835799999997e-04 t95_leaf4 - 0.00232427544 t95_leaf5
   - 6.0222053400000003e-04 t95_leaf6 - 0.00253305282 t95_leaf7
   + 0.00530554587 t96_leaf0 + 4.5696625600000001e-04 t96_leaf1
   + 0.0155092906 t96_leaf2 + 0.00247159973 t96_leaf3
   - 0.000173313 t96_leaf4 - 0.00207905 t96_leaf5 - 5.28007e-05 t96_leaf6
   + 0.0010239135 t97_leaf0 - 5.4656568700000002e-04 t97_leaf1
   - 0.00038952 t97_leaf2 - 0.00598687539 t97_leaf3
   + 0.0107287895 t97_leaf4 + 0.00213185046 t97_leaf5
   + 0.000275558 t97_leaf6 + 0.00430138642 t97_leaf7
   + 0.00250352104 t98_leaf0 + 0.0104057463 t98_leaf1
   + 9.97669e-05 t98_leaf2 + 9.07754e-05 t98_leaf3
   - 0.00222849217 t98_leaf4 + 0.00113806943 t98_leaf5
   - 0.00624600751 t98_leaf6 + 0.00241860095 t99_leaf0
   + 2.74929334e-04 t99_leaf1 + 0.000198231 t99_leaf2
   - 0.00134948047 t99_leaf3 - 0.00594925508 t99_leaf4
   + 7.8023423000000004e-04 t99_leaf5 - 0.00317678647 t99_leaf6
   + 0.00424996624 t100_leaf0 + 5.6870427199999995e-04 t100_leaf1
   + 0.0134092942 t100_leaf2 + 0.00213709287 t100_leaf3
   + 0.00314891571 t100_leaf4 + 3.8545057799999999e-04 t100_leaf5
   - 0.00103069737 t100_leaf6 + 0.00649810024 t100_leaf7
   + 0.00950837415 t101_leaf1 + 0.00077336 t101_leaf2
   + 4.20378e-05 t101_leaf3 + 0.00847122911 t101_leaf4
   - 0.000745398 t101_leaf5 - 0.00468973722 t101_leaf6
   + 0.00249119429 t102_leaf0 - 0.00117949373 t102_leaf1
   - 0.00203969935 t102_leaf2 + 0.00983485207 t102_leaf3
   - 6.62943e-05 t102_leaf4 - 0.00394211523 t102_leaf5
   - 0.00328013371 t102_leaf6 + 0.00288914074 t102_leaf7
   + 0.00384016265 t103_leaf0 - 2.32807113e-04 t103_leaf1
   - 7.96181e-05 t103_leaf2 - 0.00369127491 t103_leaf3
   - 0.00650277548 t103_leaf5 + 0.00154805754 t104_leaf0
   + 0.015939299 t104_leaf1 + 0.00336470571 t104_leaf2
   + 1.77849e-05 t104_leaf3 - 0.00219529774 t104_leaf4
   - 0.00881142356 t104_leaf5 - 7.23776e-05 t105_leaf0
   + 0.00287546078 t105_leaf1 + 0.0176807456 t105_leaf2
   + 0.00329083623 t105_leaf3 + 9.3594589299999999e-04 t105_leaf4
   - 0.00058116 t105_leaf5 - 0.000646476 t105_leaf6
   - 0.00355234812 t105_leaf7 + 0.00560936565 t106_leaf0
   + 0.00162962312 t106_leaf1 + 6.2943965900000005e-04 t106_leaf2
   - 0.00158329483 t106_leaf3 - 6.8990374000000004e-04 t106_leaf4
   - 0.00523410924 t106_leaf5 + 0.00725439098 t106_leaf6
   + 2.8199874200000001e-04 t107_leaf0 - 0.00136412017 t107_leaf1
   - 0.00377105712 t107_leaf2 + 0.00513667101 t107_leaf3
   + 1.2968918599999999e-04 t107_leaf4 + 0.00346692535 t107_leaf5
   + 0.00230826112 t107_leaf6 + 0.0111443391 t107_leaf7
   - 7.70105864e-04 t108_leaf0 - 0.00540410262 t108_leaf1
   + 0.00302288379 t108_leaf2 + 2.2209329499999999e-04 t108_leaf3
   + 0.0125533557 t108_leaf4 + 1.07306761e-04 t108_leaf5
   + 0.00201867265 t108_leaf6 - 0.00394007843 t108_leaf7
   + 0.0131585998 t109_leaf0 + 0.00525323441 t109_leaf1
   - 5.6794471999999996e-04 t109_leaf2 - 0.000836595 t109_leaf3
   + 7.2191387899999998e-04 t109_leaf4 - 8.0795417300000001e-04 t109_leaf5
   - 0.00391848106 t109_leaf6 + 6.8536069000000005e-04 t110_leaf0
   + 0.00591684505 t110_leaf1 - 0.000400256 t110_leaf2
   - 0.00840347726 t110_leaf3 - 0.00137121067 t110_leaf4
   - 0.0042779753 t110_leaf5 + 6.1652279699999995e-04 t111_leaf0
   - 0.00216934807 t111_leaf1 - 8.1799062899999997e-04 t111_leaf2
   - 0.00293535623 t111_leaf3 + 8.3927088399999997e-04 t111_leaf4
   - 0.00100846973 t111_leaf5 + 0.00890953839 t111_leaf6
   - 1.70183106e-04 t111_leaf7 + 0.00387369888 t112_leaf0
   + 5.3194788100000001e-04 t112_leaf1 - 0.00156509131 t112_leaf2
   - 0.00548292603 t112_leaf3 - 0.00155563245 t112_leaf4
   - 0.00444311183 t112_leaf5 - 8.32999e-05 t112_leaf6
   - 0.0033379 t112_leaf7 + 0.00389579497 t113_leaf0
   + 0.0129936766 t113_leaf2 + 4.6076640100000001e-04 t113_leaf3
   - 0.00320562324 t113_leaf4 - 0.00171589525 t113_leaf5
   + 0.00293459836 t113_leaf6 + 0.00343916262 t114_leaf0
   + 0.00140096608 t114_leaf1 + 0.0029606109 t114_leaf2
   - 0.00177769549 t114_leaf3 + 4.20475e-05 t114_leaf4
   - 0.00253912341 t114_leaf5 - 0.00222346955 t114_leaf6
   + 0.00464104488 t114_leaf7 + 0.0039138454 t115_leaf0
   - 4.3325129099999998e-04 t115_leaf1 + 6.8369769699999996e-04 t115_leaf2
   - 0.00145596988 t115_leaf3 - 0.00129996589 t115_leaf4
   + 0.00609500613 t115_leaf5 - 0.000587087 t116_leaf0
   - 0.00244218414 t116_leaf1 - 0.00813035481 t116_leaf2
   + 0.000477441 t116_leaf3 - 0.00100072054 t116_leaf4
   + 0.0140368547 t116_leaf5 + 0.00169876 t116_leaf6
   + 0.00469063083 t117_leaf0 + 3.7003177599999998e-04 t117_leaf2
   - 0.000823772 t117_leaf3 - 0.00668247836 t117_leaf4
   - 8.52337515e-04 t118_leaf0 - 0.00346091599 t118_leaf1
   + 0.00194121373 t118_leaf2 + 2.0217e-05 t118_leaf3
   + 0.0124354204 t118_leaf4 - 0.00113880471 t118_leaf5
   + 0.00813435297 t119_leaf0 + 0.000812568 t119_leaf1
   - 4.2833312199999998e-04 t119_leaf2 - 0.0072644027 t119_leaf3
   + 0.014609674 t120_leaf1 + 0.000148536 t120_leaf2
   + 0.00747601036 t120_leaf3 + 0.00106170145 t120_leaf4
   - 0.00397162698 t120_leaf5 + 0.0123893321 t121_leaf0
   + 0.00191624311 t121_leaf1 + 0.00268007908 t121_leaf2
   + 2.1115451800000001e-04 t121_leaf3 + 3.1029852100000001e-04 t121_leaf4
   - 8.76532751e-04 t121_leaf5 - 0.00303396909 t121_leaf6
   + 0.00243653287 t122_leaf0 - 6.7409244400000001e-04 t122_leaf1
   + 0.000143893 t122_leaf2 - 0.00192043663 t122_leaf3
   - 0.00321921869 t122_leaf4 + 1.76089423e-04 t123_leaf0
   - 0.00127760717 t123_leaf1 - 0.00550385797 t123_leaf2
   - 9.8484172499999989e-04 t123_leaf3 + 0.00171859399 t123_leaf4
   + 0.00801094715 t123_leaf5 + 4.8628283500000001e-04 t123_leaf6
   - 0.00287363213 t123_leaf7 + 0.00116190047 t124_leaf0
   - 1.80805218e-04 t124_leaf1 - 0.00382135366 t124_leaf2
   - 0.00116886781 t124_leaf4 - 0.00347349839 t124_leaf5
   + 0.00177458406 t125_leaf0 + 0.00960239209 t125_leaf1
   + 2.4653444399999998e-04 t125_leaf2 - 0.00219590357 t125_leaf3
   - 0.00281869085 t125_leaf4 + 0.00246447884 t125_leaf5
   + 0.0038692567 t126_leaf0 - 1.3273856900000001e-04 t126_leaf1
   + 0.00289387209 t126_leaf2 + 8.3683879299999999e-04 t126_leaf3
   - 0.00484327972 t126_leaf4 - 0.000827319 t127_leaf0
   + 0.0148607343 t127_leaf1 + 0.00430076662 t127_leaf2
   + 9.7087235200000001e-04 t127_leaf3 + 0.00212545763 t127_leaf4
   - 2.1923164600000001e-04 t127_leaf5 + 0.00807339326 t128_leaf0
   + 6.66278589e-04 t128_leaf1 - 0.00104274566 t128_leaf2
   - 5.7749310500000001e-04 t128_leaf3 - 0.00346494932 t128_leaf4
   + 0.0127552152 t129_leaf0 - 0.000351672 t129_leaf1
   - 0.000155821 t129_leaf2 - 0.00435514795 t129_leaf3
   + 5.8101856799999999e-04 t129_leaf4 + 0.00364471599 t129_leaf5
   + 9.8536640899999999e-04 t130_leaf0 - 4.2817671800000001e-04 t130_leaf1
   + 0.00155460928 t130_leaf2 - 0.00388227194 t130_leaf3
   - 0.000745447 t130_leaf4 - 0.00387294171 t130_leaf5
   + 4.92481e-05 t130_leaf6 - 0.00362609443 t130_leaf7
   + 0.0164996125 t131_leaf0 + 6.1366969000000002e-04 t131_leaf1
   - 0.00130882789 t131_leaf2 + 0.00341900345 t131_leaf3
   - 9.6837611600000004e-04 t131_leaf4 + 6.3013675400000002e-04 t131_leaf5
   - 0.00131754228 t131_leaf6 - 0.00589310424 t131_leaf7
   + 0.000501106 t132_leaf0 - 0.00103556376 t132_leaf1
   + 0.0067718057 t132_leaf2 - 0.00656205043 t132_leaf3
   - 6.0167774800000005e-04 t132_leaf4 + 0.00384620437 t133_leaf0
   - 0.00138416921 t133_leaf1 - 6.86593e-05 t133_leaf2
   - 0.00227153627 t133_leaf3 + 0.00617895694 t133_leaf4
   + 0.0022956226 t134_leaf0 - 6.2762631599999998e-04 t134_leaf1
   - 0.000735431 t134_leaf2 + 5.0324550799999996e-04 t134_leaf3
   - 0.00601193681 t134_leaf4 + 0.00223605451 t135_leaf0
   - 0.00155308784 t135_leaf1 + 0.00148947118 t135_leaf2
   + 0.00583571196 t135_leaf3 + 8.74444202e-04 t135_leaf4
   - 0.000375838 t135_leaf5 + 0.00445225183 t135_leaf6
   - 6.6615169599999996e-04 t136_leaf0 + 4.2879630900000002e-04 t136_leaf1
   + 0.00708312215 t136_leaf2 - 0.00471841451 t136_leaf3
   - 1.6588933099999999e-04 t136_leaf4 + 0.00440930529 t137_leaf0
   + 0.00149541919 t137_leaf1 + 0.00148240733 t137_leaf2
   - 0.0023806931 t137_leaf3 - 0.00117226562 t137_leaf4
   + 0.000152184 t137_leaf5 - 4.65111e-05 t137_leaf6
   - 0.0062578083 t137_leaf7 + 0.00992888212 t138_leaf0
   - 1.08826818e-04 t138_leaf1 + 0.0039262888 t138_leaf2
   + 0.00283156522 t138_leaf3 + 6.2334770300000002e-04 t138_leaf4
   - 4.5266345800000003e-04 t138_leaf5 + 0.00525639392 t138_leaf6
   + 8.50913872e-04 t139_leaf0 - 3.8085674200000002e-04 t139_leaf1
   + 0.00113696326 t139_leaf2 + 0.00554104149 t139_leaf3
   + 0.00246050628 t139_leaf4 - 0.00175151671 t139_leaf5
   + 3.1314615600000001e-04 t139_leaf6 - 0.00222654128 t139_leaf7
   + 0.00154242804 t140_leaf0 + 0.0124624632 t140_leaf1
   + 7.54184672e-04 t140_leaf2 + 0.00370411738 t140_leaf3
   + 0.000923204 t140_leaf4 - 4.3876149000000002e-04 t140_leaf5
   - 0.001051725 t140_leaf6 - 0.0061641084 t140_leaf7
   + 0.00404748367 t141_leaf0 + 4.16536233e-04 t141_leaf1
   + 0.0065574795 t141_leaf2 + 0.00249515404 t141_leaf3
   - 1.3307880699999999e-04 t141_leaf4 - 0.00717616407 t141_leaf5
   + 1.44768783e-04 t142_leaf0 - 0.00229809224 t142_leaf1
   + 1.8105332899999999e-04 t142_leaf2 - 0.00263043516 t142_leaf3
   + 0.00564735383 t142_leaf4 + 0.00383990351 t143_leaf0
   - 3.3651170100000002e-04 t143_leaf1 + 0.00904309284 t143_leaf2
   + 4.7012424299999999e-04 t143_leaf3 - 6.3238438500000003e-04 t143_leaf4
   + 0.00101561949 t143_leaf5 - 0.00377840595 t143_leaf6
   - 4.1747884800000002e-04 t144_leaf0 - 0.00421067467 t144_leaf1
   + 0.00271595479 t144_leaf2 + 2.5746374699999998e-04 t144_leaf3
   + 0.00167537411 t144_leaf4 - 0.00361242797 t144_leaf5
   + 4.67970327e-04 t145_leaf0 - 0.00331039634 t145_leaf1
   - 9.0656889399999996e-04 t145_leaf2 + 0.00333343935 t145_leaf3
   + 0.000228045 t145_leaf4 + 0.0117751723 t145_leaf5
   + 0.00125302398 t146_leaf0 + 0.00491212076 t146_leaf1
   + 2.53636885e-04 t146_leaf2 - 0.00295299874 t146_leaf3
   - 1.0158356599999999e-04 t146_leaf4 - 0.00248405733 t146_leaf5
   - 0.00221785437 t146_leaf6 + 0.00130692043 t146_leaf7
   + 0.00670928461 t147_leaf0 + 0.00143308355 t147_leaf1
   - 2.80722e-05 t147_leaf2 - 0.00108349102 t147_leaf3
   + 0.00128695776 t147_leaf4 + 0.0066680559 t147_leaf5
   + 0.00189711095 t148_leaf0 - 2.2969320700000001e-04 t148_leaf1
   - 0.00374605716 t148_leaf2 + 0.00977600832 t148_leaf3
   - 1.8778567e-04 t148_leaf4 + 0.00391730852 t148_leaf5
   + 0.00519103697 t149_leaf0 + 0.00105129613 t149_leaf1
   - 8.0310588199999997e-04 t149_leaf2 + 0.000287164 t149_leaf3
   - 6.5617461199999998e-04 t149_leaf4 - 0.00459218211 t149_leaf5
   - 0.000290911 t149_leaf6 + 0.0100056566 t150_leaf1
   + 0.000745856 t150_leaf2 + 0.0025771081 t150_leaf3
   + 6.6731940000000001e-04 t150_leaf4 - 0.000203765 t150_leaf5
   - 0.00303622219 t150_leaf6 + 0.00190708 t151_leaf0
   + 2.15112683e-04 t151_leaf1 - 0.00057727 t151_leaf2
   - 0.00464307098 t151_leaf3 - 0.00193217641 t151_leaf4
   - 0.000404638 t152_leaf0 + 0.00943173654 t152_leaf1
   - 0.0022599895 t152_leaf2 + 3.5593e-05 t152_leaf3
   - 0.00134276634 t152_leaf4 - 0.00549311098 t152_leaf5
   + 0.00475836545 t153_leaf0 - 0.00123234745 t153_leaf1
   + 0.00131855812 t153_leaf2 + 0.00307313376 t153_leaf3
   - 0.000167871 t153_leaf4 - 0.00109569973 t153_leaf5
   - 0.00541100558 t153_leaf6 + 0.0128901443 t154_leaf0
   + 0.00256861979 t154_leaf1 + 9.0384035100000005e-04 t154_leaf2
   - 8.6343852999999995e-04 t154_leaf3 + 0.00305730267 t154_leaf4
   - 0.0019004409 t154_leaf5 - 9.9845463400000003e-04 t154_leaf6
   + 8.13391e-05 t154_leaf7 + 2.0216322399999999e-04 t155_leaf0
   - 0.00289092213 t155_leaf1 - 0.00339708594 t155_leaf2
   + 0.00257968367 t155_leaf3 + 0.00782286096 t155_leaf4
   + 0.00103155756 t156_leaf0 - 0.00170895655 t156_leaf2
   - 0.000935968 t156_leaf3 - 0.00475443574 t156_leaf4
   + 0.00262764352 t157_leaf0 + 1.5708665899999999e-04 t157_leaf1
   - 8.96444e-06 t157_leaf2 + 0.00409145746 t157_leaf3
   + 9.0533064200000002e-04 t157_leaf4 - 6.45026797e-04 t157_leaf5
   - 0.00538952323 t157_leaf6 + 0.00128950353 t158_leaf0
   + 0.00497971522 t158_leaf1 - 0.00238245493 t158_leaf2
   + 2.32776e-05 t158_leaf3 - 0.00131484552 t158_leaf4
   + 8.5022358700000003e-04 t158_leaf5 - 0.000171738 t158_leaf6
   + 0.00685757771 t159_leaf0 + 1.48523803e-04 t159_leaf1
   + 0.000576754 t159_leaf2 - 5.52274694e-04 t159_leaf3
   - 0.00271712895 t159_leaf4 - 9.0906472200000005e-04 t160_leaf0
   - 0.00358924293 t160_leaf1 + 3.640184e-04 t160_leaf2
   - 0.00157650188 t160_leaf3 + 0.0123870578 t160_leaf4
   + 0.00301926606 t160_leaf5 - 6.48189e-05 t160_leaf6
   - 2.94476165e-04 t161_leaf0 - 0.0043304693 t161_leaf1
   + 2.19343e-05 t161_leaf2 - 0.00423778314 t161_leaf3
   + 0.00254894095 t161_leaf4 - 0.00142773858 t161_leaf5
   + 0.00995922927 t161_leaf6 + 0.00277962233 t162_leaf0
   - 0.0010286 t162_leaf1 + 0.00924222916 t162_leaf2
   + 1.9550281299999999e-04 t162_leaf3 - 0.00194780191 t162_leaf4
   - 1.1168448e-04 t162_leaf5 - 0.0040239715 t162_leaf6
   + 6.3215522100000001e-04 t163_leaf0 - 0.00231857831 t163_leaf1
   - 0.00110649515 t163_leaf2 + 0.00803782139 t163_leaf3
   - 7.7047327100000004e-04 t163_leaf4 - 0.004270704 t163_leaf5
   + 8.1543158699999996e-04 t164_leaf0 + 0.00594227761 t164_leaf1
   + 4.15493e-06 t164_leaf2 - 0.00151229242 t164_leaf3
   - 0.00441542594 t164_leaf4 + 4.72425e-05 t165_leaf0
   + 0.00250822585 t165_leaf1 - 0.00402288139 t165_leaf2
   - 1.1790121100000001e-04 t165_leaf3 + 0.00679614907 t165_leaf4
   + 1.99581875e-04 t166_leaf0 - 0.00118928764 t166_leaf1
   - 0.00316333212 t166_leaf2 - 0.00668793404 t166_leaf3
   + 0.00766464556 t167_leaf0 + 0.00420584157 t167_leaf1
   + 9.89351e-05 t167_leaf2 - 0.0018240317 t167_leaf3
   + 3.8032620799999999e-04 t167_leaf4 + 1.7208413899999999e-04 t168_leaf0
   - 9.0946885800000001e-04 t168_leaf1 - 0.00216228724 t168_leaf2
   + 0.00345744845 t168_leaf3 - 0.00236554514 t168_leaf4
   + 0.00364929414 t169_leaf0 - 0.00132622477 t169_leaf1
   + 4.6319028499999999e-04 t169_leaf2 - 0.00136351411 t169_leaf3
   - 0.00212590164 t169_leaf4 + 0.00151024479 t169_leaf5
   - 7.6797971299999998e-04 t169_leaf6 - 2.87673e-05 t170_leaf0
   - 0.00225284765 t170_leaf1 + 0.000315003 t170_leaf2
   + 0.00405591261 t170_leaf3 - 0.00202639261 t170_leaf4
   + 5.47664e-05 t171_leaf0 - 0.00262519903 t171_leaf1
   + 0.00253719394 t171_leaf2 - 0.0014689028 t171_leaf3
   - 0.00208207848 t171_leaf4 + 0.00113251607 t172_leaf0
   - 2.11031118e-04 t172_leaf1 - 0.000490008 t172_leaf2
   + 0.00244770618 t172_leaf3 + 0.00142520433 t172_leaf4
   - 0.00198743143 t172_leaf5 - 3.0158512499999998e-04 t172_leaf6
   + 0.0072251833 t173_leaf0 - 0.00204607402 t173_leaf1
   + 0.00241500279 t173_leaf2 + 0.00157694973 t173_leaf3
   - 6.17756e-05 t173_leaf4 - 0.00114715099 t173_leaf5
   + 0.00197835197 t173_leaf6 - 8.1820873299999995e-04 t174_leaf0
   + 2.3724029600000001e-04 t174_leaf1 - 0.00166332279 t174_leaf2
   + 0.00546094449 t174_leaf3 + 1.9622159e-04 t175_leaf0
   + 0.00540297572 t175_leaf1 + 2.16569e-05 t175_leaf2
   + 0.00286504603 t175_leaf3 - 0.00478268228 t175_leaf4
   - 7.3244056000000004e-04 t176_leaf0 - 0.00493534375 t176_leaf1
   + 8.5767684500000002e-04 t176_leaf2 - 0.00511272019 t176_leaf3
   + 0.00214686012 t176_leaf4 + 5.14896e-05 t176_leaf5
   + 0.0065221414 t176_leaf6 + 9.8397349899999992e-04 t177_leaf0
   - 0.00282414583 t177_leaf1 - 0.00186355936 t177_leaf2
   + 6.0287542900000002e-04 t177_leaf3 - 6.8606977600000002e-04 t177_leaf4
   + 2.3832917200000001e-04 t177_leaf5 - 0.00187534431 t177_leaf6
   - 0.00115839101 t178_leaf0 + 0.00196731486 t178_leaf1
   + 0.00514182774 t178_leaf2 + 2.2467914099999999e-04 t178_leaf3
   - 0.00207689381 t178_leaf4 + 0.0025010996 t179_leaf0
   - 0.00188148813 t179_leaf1 + 0.0075530149 t179_leaf2
   - 0.00114242511 t179_leaf3 + 1.6533964699999999e-04 t179_leaf4
   - 0.00189401931 t179_leaf5 + 0.00905951206 t180_leaf0
   + 0.000731928 t180_leaf1 - 0.000128799 t180_leaf2
   - 0.00292870682 t180_leaf3 + 6.6335080199999999e-04 t180_leaf4
   + 9.71544e-05 t181_leaf0 - 0.00143028074 t181_leaf1
   - 0.00304660876 t181_leaf2 + 0.00574098201 t181_leaf3
   + 0.00124600821 t181_leaf4 + 2.67431256e-04 t181_leaf5
   - 0.00320238806 t181_leaf6 + 0.00326526631 t182_leaf0
   + 2.61599e-05 t182_leaf1 - 0.00202956027 t182_leaf2
   + 0.00352951256 t182_leaf3 + 0.00961248111 t182_leaf4
   - 0.00224315352 t182_leaf6 + 0.00111086702 t182_leaf7
   + 0.000287012 t183_leaf0 - 9.6278119599999999e-04 t183_leaf1
   - 0.00483966758 t183_leaf2 + 0.00765790977 t183_leaf3
   + 1.9218541299999999e-04 t183_leaf4 + 0.00129547226 t183_leaf5
   + 0.000459992 t184_leaf0 - 7.9059024599999997e-04 t184_leaf1
   - 0.00159231713 t184_leaf2 - 9.04268178e-04 t184_leaf3
   - 0.00402761158 t184_leaf4 + 0.00337277 t184_leaf5
   + 2.07902878e-04 t185_leaf0 - 9.7563635799999997e-04 t185_leaf1
   - 8.3005614599999998e-04 t185_leaf2 - 0.00311585655 t185_leaf3
   + 0.00895266607 t185_leaf4 - 0.00164797914 t185_leaf5
   - 4.49474814e-04 t186_leaf0 + 5.4404145300000002e-04 t186_leaf1
   - 0.00297788437 t186_leaf2 - 0.000905289 t186_leaf3
   + 0.00635075895 t186_leaf4 + 0.00145009172 t186_leaf5
   - 0.00142081606 t186_leaf6 + 9.08691e-05 t187_leaf0
   - 8.3337322599999996e-04 t187_leaf1 + 0.0038740125 t187_leaf3
   + 0.00763477059 t187_leaf4 + 0.00193067873 t187_leaf5
   - 0.00215014885 t187_leaf6 - 0.00301195169 t188_leaf0
   - 8.6531275900000003e-04 t188_leaf1 - 0.00107223075 t188_leaf2
   + 0.00172887882 t188_leaf3 + 4.29223612e-04 t188_leaf4
   - 0.00225110166 t188_leaf5 + 6.7669572299999998e-04 t188_leaf6
   - 0.00299918419 t188_leaf7 + 0.00181068573 t189_leaf0
   + 1.10858316e-04 t189_leaf1 - 0.00149154337 t189_leaf2
   + 0.00204217 t189_leaf3 + 0.00593167311 t189_leaf4
   + 0.00549565488 t190_leaf0 + 1.9519748499999999e-04 t190_leaf1
   + 5.54534e-05 t190_leaf2 - 0.00258282013 t190_leaf3
   - 0.0011386812 t190_leaf4 - 0.00408697082 t190_leaf5
   + 0.005940089 t191_leaf0 - 8.06453e-05 t191_leaf2
   - 0.00509836851 t191_leaf3 + 0.00930203777 t191_leaf4
   + 0.00265856227 t191_leaf5 - 0.00259114569 t191_leaf6
   + 8.1273919200000003e-04 t192_leaf0 - 0.000115221 t192_leaf1
   - 0.0015672195 t192_leaf2 - 0.00165862311 t192_leaf3
   + 3.4220164500000001e-04 t192_leaf4 + 4.9974629700000003e-04 t193_leaf0
   - 0.00185153738 t193_leaf1 - 5.07829e-05 t193_leaf2
   + 0.00837193616 t193_leaf3 + 0.00290075387 t193_leaf4
   - 5.0091650299999995e-04 t193_leaf5 - 0.0015977869 t193_leaf6
   + 0.00216280879 t193_leaf7 + 0.00278056832 t194_leaf0
   + 1.1121e-05 t194_leaf1 - 0.00393475406 t194_leaf2
   - 0.00507996464 t194_leaf3 + 0.00199575373 t195_leaf0
   - 4.8313647899999999e-04 t195_leaf1 + 3.4862189199999998e-04 t195_leaf2
   - 0.0044179908 t195_leaf3 - 1.8382e-05 t196_leaf0
   - 0.00166235492 t196_leaf1 - 0.00447970582 t196_leaf2
   - 0.00100848381 t196_leaf3 + 0.000311528 t196_leaf4
   + 0.00330254831 t196_leaf5 + 7.38665e-05 t197_leaf0
   - 0.00194279698 t197_leaf1 - 0.00255087134 t197_leaf2
   + 0.00347314868 t197_leaf3 + 0.00895583071 t197_leaf4
   - 0.0011915277 t197_leaf5 - 0.00235328381 t198_leaf0
   + 0.00785529707 t198_leaf1 - 2.2192248400000001e-04 t198_leaf2
   + 0.000825873 t198_leaf3 - 0.00155206083 t198_leaf4
   + 0.00381226582 t198_leaf5 + 0.00230843481 t199_leaf0
   - 0.00318638654 t199_leaf1 + 2.1901678699999999e-04 t199_leaf2
   - 6.9844577199999996e-04 t199_leaf3 - 0.00185636024 t199_leaf4
   + 0.00106137909 t200_leaf0 - 7.76621e-05 t200_leaf1
   + 0.0014294571 t200_leaf2 + 0.00133561 t200_leaf3
   - 0.00268011773 t200_leaf4 - 2.43966e-05 t201_leaf0
   - 0.00136049569 t201_leaf1 + 0.00387548306 t201_leaf2
   - 0.00443078484 t201_leaf3 + 0.00166257948 t201_leaf4
   + 0.00185461331 t202_leaf0 - 0.00111788954 t202_leaf1
   - 0.00119412947 t202_leaf2 - 3.8466451200000002e-04 t202_leaf3
   - 0.00192728778 t202_leaf4 + 0.000387415 t202_leaf5
   + 0.00337574072 t202_leaf6 - 8.72986e-05 t203_leaf0
   - 0.00162577338 t203_leaf1 - 3.1842160400000002e-04 t203_leaf2
   - 0.00312998472 t203_leaf3 + 0.00169050321 t203_leaf4
   + 3.58721e-05 t203_leaf5 - 0.0002982 t203_leaf6
   + 0.00514168944 t203_leaf7 + 0.00862085633 t204_leaf0
   + 1.87115627e-04 t204_leaf1 - 0.000524832 t204_leaf2
   + 2.2800113899999999e-04 t204_leaf3 - 0.00282363733 t204_leaf4
   + 0.00190226838 t205_leaf0 + 5.46596e-05 t205_leaf1
   - 0.00214858679 t205_leaf2 - 0.00185900053 t205_leaf3
   + 0.00374794379 t206_leaf0 + 2.2710754999999999e-04 t206_leaf1
   - 0.00203852868 t206_leaf2 - 0.00450836169 t206_leaf3
   - 0.000120231 t207_leaf0 + 0.00177580479 t207_leaf1
   - 0.0025022414 t207_leaf2 + 0.0090105515 t207_leaf3
   - 7.5113872299999997e-04 t207_leaf4 + 0.00283448235 t207_leaf5
   + 4.7174433700000002e-04 t208_leaf0 - 7.0005719300000001e-04 t208_leaf1
   - 0.0022903448 t208_leaf2 + 0.00250066514 t208_leaf3
   + 0.00579069834 t208_leaf4 + 1.9255558400000001e-04 t209_leaf0
   + 0.00261323969 t209_leaf1 + 2.4063047e-04 t209_leaf2
   - 0.00325495284 t209_leaf3 - 0.00209606276 t209_leaf4
   - 4.92342e-05 t210_leaf0 - 0.00511628389 t210_leaf1
   + 0.00533114048 t210_leaf2 - 9.7858719500000003e-04 t210_leaf3
   + 0.00317438738 t210_leaf4 - 0.00174759305 t211_leaf0
   + 2.26691598e-04 t211_leaf1 - 0.00213410635 t211_leaf2
   - 2.3378941099999999e-04 t211_leaf3 + 6.6552014300000004e-04 t211_leaf4
   - 9.2880940099999997e-04 t211_leaf5 + 0.00954541843 t211_leaf6
   - 6.2729057399999995e-04 t212_leaf0 + 0.0105997743 t212_leaf1
   - 5.61784e-06 t212_leaf2 + 0.0014280465 t212_leaf3
   - 0.00124600285 t212_leaf4 + 6.76768e-05 t212_leaf5
   + 0.00145896478 t213_leaf0 + 2.84885e-05 t213_leaf1
   - 0.00157784426 t213_leaf2 - 0.00294638495 t213_leaf3
   + 5.1357131500000004e-04 t214_leaf0 - 6.1345967700000001e-04 t214_leaf1
   + 0.00172968546 t214_leaf2 - 0.000907142 t214_leaf3
   + 0.00159568107 t214_leaf4 - 0.00229590479 t214_leaf5
   + 0.00449209707 t214_leaf6 - 8.4975245399999998e-04 t215_leaf0
   + 0.00873776618 t215_leaf1 + 0.000858947 t215_leaf2
   + 0.0050591873 t215_leaf3 + 0.00024748 t215_leaf4
   - 7.4304960400000005e-04 t215_leaf5 + 3.9571285000000001e-04 t216_leaf0
   - 0.00166836358 t216_leaf1 + 0.00218727533 t216_leaf2
   - 7.4188149299999999e-04 t216_leaf3 + 0.00650743116 t216_leaf4
   + 0.00273647765 t217_leaf0 - 9.8437687800000009e-04 t217_leaf1
   - 0.000746225 t217_leaf2 + 1.03198799e-04 t217_leaf3
   + 0.00569834094 t217_leaf4 + 1.3599146e-04 t218_leaf0
   - 0.00106601778 t218_leaf1 - 0.00182674988 t218_leaf2
   - 0.00399381341 t218_leaf3 - 0.00208459538 t219_leaf0
   + 4.0899412099999999e-04 t219_leaf1 + 0.00313806557 t219_leaf2
   + 1.082e-05 t219_leaf3 + 0.00195136259 t219_leaf4
   - 0.00214015972 t219_leaf5 + 0.00132694491 t219_leaf6
   - 5.5559322900000001e-04 t220_leaf0 + 0.00294971443 t220_leaf1
   + 2.1886872100000001e-04 t220_leaf2 + 0.00475296704 t220_leaf3
   + 1.0583546800000001e-04 t220_leaf4 + 0.00165646453 t220_leaf5
   + 0.00464753713 t220_leaf6 + 4.32632e-05 t221_leaf0
   - 0.00177175936 t221_leaf1 - 0.00361449132 t221_leaf2
   + 0.00453913817 t221_leaf3 + 2.62517686e-04 t221_leaf4
   - 0.00248517911 t221_leaf5 - 0.00112314161 t222_leaf0
   + 0.00132448284 t222_leaf1 + 0.000299122 t222_leaf2
   - 0.00120631733 t222_leaf3 - 0.0046050013 t222_leaf4
   + 0.00145228324 t223_leaf0 - 0.00076052 t223_leaf1
   + 0.00489732297 t223_leaf2 - 7.7823438900000001e-04 t223_leaf3
   + 1.1775062000000001e-04 t223_leaf4 - 8.1301643499999995e-04 t223_leaf5
   - 0.00437305588 t223_leaf6 + 0.00506596453 t224_leaf0
   + 5.28848e-05 t224_leaf1 - 0.0013440731 t224_leaf2
   + 0.00567819877 t224_leaf3 + 2.7219415599999999e-04 t225_leaf0
   + 0.0019484068 t225_leaf1 - 9.9844683399999998e-04 t225_leaf2
   + 0.0037483694 t225_leaf3 - 0.00348433154 t225_leaf4
   + 0.00112925225 t226_leaf0 - 0.00265713152 t226_leaf1
   - 0.00175249623 t226_leaf2 + 0.0114988554 t226_leaf3
   - 4.6732911099999999e-04 t226_leaf4 + 0.000527025 t226_leaf5
   + 0.00225942791 t226_leaf6 - 0.000889518 t226_leaf7
   + 4.79336e-05 t227_leaf0 - 0.00312280934 t227_leaf1
   - 6.0751882799999998e-04 t227_leaf2 + 0.00566353789 t227_leaf3
   - 0.00178707391 t227_leaf4 + 0.00131278683 t228_leaf0
   - 1.9790265699999999e-04 t228_leaf1 - 0.00241944077 t228_leaf2
   + 0.00119158579 t228_leaf3 + 0.00522818789 t228_leaf4
   + 0.00132759777 t229_leaf0 + 0.000140202 t229_leaf1
   - 0.00498079043 t229_leaf3 + 0.00558081362 t229_leaf4
   - 0.00486052455 t230_leaf0 - 6.374e-05 t230_leaf1
   + 0.00917181652 t230_leaf2 + 4.9809686599999996e-04 t230_leaf3
   - 2.91e-05 t230_leaf4 - 0.00127094961 t230_leaf5
   - 0.00314872479 t230_leaf6 + 3.0519277800000001e-04 t231_leaf0
   + 0.00241700769 t231_leaf1 + 3.69585439e-04 t231_leaf2
   - 0.00103738694 t231_leaf3 - 0.00527172815 t231_leaf5
   + 0.00363997114 t231_leaf6 - 0.00187012309 t231_leaf7
   + 1.19347817e-04 t232_leaf0 - 0.00206339057 t232_leaf1
   - 0.00365805672 t232_leaf2 + 3.9213246799999998e-04 t232_leaf3
   + 0.00925439969 t232_leaf4 + 3.8949237199999999e-04 t233_leaf0
   - 9.8585244299999992e-04 t233_leaf1 - 0.00224351976 t233_leaf2
   + 0.00191873219 t233_leaf3 - 0.00107608573 t233_leaf4
   - 0.00382866967 t233_leaf5 + 0.00289950566 t233_leaf6
   - 0.00447130436 t234_leaf0 + 3.3622118600000001e-04 t234_leaf1
   - 1.39880154e-04 t234_leaf2 + 0.00181967823 t234_leaf3
   - 0.00210449332 t234_leaf4 + 0.0086479187 t234_leaf5
   + 0.00440281443 t235_leaf0 - 8.4987556299999996e-04 t235_leaf1
   - 1.00995509e-04 t235_leaf2 - 0.00273077027 t235_leaf3
   + 0.00544502027 t235_leaf4 + 3.4689163999999999e-04 t236_leaf0
   - 0.00124781742 t236_leaf1 - 0.00139640109 t236_leaf2
   - 3.0295952500000001e-04 t236_leaf3 - 0.00251525431 t236_leaf4
   + 2.2119721699999999e-04 t236_leaf5 - 0.00194301479 t237_leaf0
   + 2.1809065899999999e-04 t237_leaf1 - 0.00289399223 t237_leaf2
   - 0.00145072269 t237_leaf3 + 0.00148865231 t238_leaf0
   - 0.000135139 t238_leaf1 - 0.00138299179 t238_leaf2
   - 0.00114893075 t238_leaf3 + 0.0021455856 t238_leaf4
   + 0.00932715461 t238_leaf5 + 3.3013135499999999e-04 t239_leaf0
   + 0.0026459191 t239_leaf1 + 0.00109056965 t239_leaf2
   - 0.000585942 t239_leaf3 - 0.00176710845 t239_leaf4
   - 4.43362864e-04 t239_leaf5 + 0.00413649902 t239_leaf6
   + 0.00145143794 t240_leaf0 - 8.8383746300000005e-04 t240_leaf1
   + 7.22954e-05 t240_leaf2 + 0.00183427252 t240_leaf3
   - 0.00214131107 t240_leaf4 - 0.0044108904 t240_leaf5
   + 0.00514017139 t241_leaf0 + 6.4703787200000003e-04 t241_leaf1
   + 0.00512567908 t241_leaf2 - 3.49348e-05 t241_leaf3
   - 0.00348596415 t241_leaf4 - 0.00228291983 t242_leaf0
   - 3.2956415100000001e-04 t242_leaf1 + 0.00150711066 t242_leaf2
   + 5.88582e-05 t242_leaf3 - 0.00249068718 t242_leaf4
   - 2.23588e-05 t243_leaf0 + 0.0018569564 t243_leaf1
   - 0.00363136549 t243_leaf2 + 5.5037211899999999e-04 t243_leaf3
   + 0.00508007454 t243_leaf4 - 1.1657823800000001e-04 t244_leaf0
   - 0.0042176689 t244_leaf1 + 0.00409394363 t244_leaf2
   + 0.0065394952 t244_leaf3 + 0.00361436768 t245_leaf0
   - 8.0776674399999997e-04 t245_leaf1 - 2.9887745000000001e-04 t245_leaf2
   + 3.6450629700000002e-04 t245_leaf3 - 0.00120888627 t245_leaf4
   - 3.0983827299999999e-04 t245_leaf5 - 0.0021545 t245_leaf6
   + 0.00117985089 t246_leaf0 - 4.3443069400000002e-04 t246_leaf1
   + 2.2287925799999999e-04 t246_leaf2 - 0.00289490656 t246_leaf3
   + 0.000458228 t247_leaf0 + 0.00774591 t247_leaf1
   - 0.000959201 t247_leaf2 + 1.4756934299999999e-04 t247_leaf3
   - 0.00176002213 t247_leaf4 + 2.1451164499999999e-04 t248_leaf0
   + 0.00531575875 t248_leaf1 + 0.0112847826 t248_leaf2
   - 1.6135735500000001e-04 t248_leaf4 - 0.00177354785 t248_leaf5
   + 6.1185460099999998e-04 t248_leaf6 - 0.00113312027 t248_leaf7
   - 3.5636423899999999e-04 t249_leaf0 - 0.0033805382 t249_leaf1
   + 0.00014462 t249_leaf2 + 0.00265427376 t249_leaf3
   + 0.00472077494 t249_leaf4 - 0.000293176 t250_leaf0
   + 0.00113418349 t250_leaf1 - 0.00345080416 t250_leaf2
   + 0.000641144 t250_leaf3 - 0.00118860265 t251_leaf0
   + 7.17064e-05 t251_leaf1 + 0.00118327641 t251_leaf2
   - 0.00238022674 t251_leaf3 + 0.00396060292 t251_leaf4
   + 0.00335873431 t252_leaf0 - 0.00270171743 t252_leaf2
   + 0.00239010598 t252_leaf3 - 2.57805e-06 t252_leaf4
   - 0.00163466483 t252_leaf5 - 8.01882e-06 t252_leaf6
   + 0.00502102589 t252_leaf7 - 4.56738e-05 t253_leaf0
   - 0.00193453778 t253_leaf1 - 0.00242305128 t253_leaf2
   + 0.00371805113 t253_leaf3 - 0.00237564743 t253_leaf4
   + 0.0103309685 t253_leaf5 + 0.00549242 t254_leaf0
   - 7.11977e-05 t254_leaf1 + 0.00373166031 t254_leaf2
   - 0.00193886086 t254_leaf3 - 6.23599452e-04 t255_leaf0
   + 0.00361407129 t255_leaf1 + 1.0774e-05 t255_leaf2
   + 9.6435233699999998e-04 t255_leaf3 - 0.00104780344 t255_leaf4
   + 0.00139654393 t255_leaf5 + 0.00989516824 t255_leaf6
   - 0.00191599713 t256_leaf0 + 0.00181284233 t256_leaf1
   - 1.2294459199999999e-04 t256_leaf2 + 0.0020435038 t256_leaf3
   - 0.00151783053 t256_leaf4 + 0.00672136107 t256_leaf5
   + 6.9036911000000002e-04 t257_leaf0 - 0.00340480544 t257_leaf1
   + 0.00903323106 t257_leaf2 + 0.00112355198 t257_leaf3
   + 0.00250752852 t257_leaf4 - 1.58828392e-04 t257_leaf5
   - 0.00289178523 t257_leaf6 + 0.00164522417 t258_leaf0
   + 7.44413e-05 t258_leaf1 - 0.00129946624 t258_leaf2
   - 0.00243842648 t258_leaf3 + 6.0318387099999996e-04 t258_leaf4
   + 0.00117152603 t259_leaf0 - 9.8077e-05 t259_leaf1
   - 2.28508e-05 t259_leaf2 - 9.8764523900000002e-04 t259_leaf3
   - 0.00210561021 t259_leaf4 + 6.3973333500000004e-04 t260_leaf0
   + 0.00532145519 t260_leaf1 - 1.88202786e-04 t260_leaf2
   + 0.00267905765 t260_leaf3 - 3.2032e-05 t260_leaf4
   + 0.00604429469 t260_leaf5 - 3.73411e-05 t261_leaf0
   + 0.00154237507 t261_leaf1 - 0.00379871484 t261_leaf2
   + 0.00427913107 t261_leaf3 + 0.00419620611 t262_leaf0
   - 5.5934e-05 t262_leaf1 - 0.00194463949 t262_leaf2
   - 4.1574824700000001e-04 t262_leaf3 + 0.00261541316 t262_leaf4
   - 0.00200992543 t263_leaf0 - 4.8798e-05 t263_leaf1
   - 0.00213942723 t263_leaf2 + 7.56919e-05 t263_leaf3
   + 0.00904431287 t263_leaf4 + 0.00177308626 t263_leaf5
   + 0.00408255216 t264_leaf0 + 0.00186593214 t264_leaf1
   - 0.00300318166 t264_leaf2 - 1.5956540300000001e-04 t264_leaf3
   + 0.0024440689 t264_leaf4 - 5.1107222699999996e-04 t265_leaf0
   - 0.00538049592 t265_leaf1 + 0.00201728591 t265_leaf2
   + 9.50387e-05 t265_leaf3 + 0.00454309583 t265_leaf4
   + 0.000190981 t266_leaf0 - 0.00143905089 t266_leaf1
   - 0.00546398014 t266_leaf2 + 0.00130339409 t266_leaf3
   - 7.3680351499999996e-04 t266_leaf4 - 0.00364414044 t266_leaf5
   + 0.00233197 t266_leaf6 + 1.2354415999999999e-04 t267_leaf0
   - 7.3559622999999998e-04 t267_leaf1 - 0.00405685883 t267_leaf2
   + 0.000329477 t267_leaf3 + 0.00359073491 t267_leaf4
   + 0.00554084405 t267_leaf5 + 2.21941358e-04 t268_leaf0
   - 0.000819201 t268_leaf1 - 0.00242301845 t268_leaf3
   + 0.00757386303 t268_leaf4 - 4.2457255800000001e-04 t268_leaf5
   - 4.97551e-05 t269_leaf0 - 0.00175292813 t269_leaf1
   - 0.000707501 t269_leaf2 + 0.00326993829 t269_leaf3
   - 0.00375807681 t269_leaf4 - 2.4939863900000001e-04 t270_leaf0
   + 0.00639411481 t270_leaf1 + 4.71886e-05 t270_leaf2
   + 0.00334672257 t270_leaf3 - 6.2326487399999998e-04 t270_leaf4
   + 0.00716247 t270_leaf5 - 0.00442833919 t271_leaf0
   + 1.4622333400000001e-04 t271_leaf1 + 9.82355e-05 t271_leaf2
   - 9.3874812600000003e-04 t271_leaf3 - 0.00105954439 t271_leaf4
   + 0.00927618705 t271_leaf5 - 0.00433454383 t272_leaf0
   - 1.03450657e-04 t272_leaf1 + 0.0024835 t272_leaf2
   - 0.00135959743 t272_leaf3 + 0.00113346463 t272_leaf4
   - 0.00266743265 t272_leaf5 + 0.00201791618 t272_leaf6
   + 0.00742451521 t272_leaf7 - 3.4346629400000003e-04 t273_leaf0
   + 0.00359063689 t273_leaf1 - 0.00330649433 t273_leaf2
   + 6.6896458200000003e-04 t273_leaf3 + 0.00547504332 t273_leaf4
   - 4.00486373e-04 t274_leaf0 - 0.00366037036 t274_leaf1
   - 1.84193821e-04 t274_leaf2 + 6.7177915499999998e-04 t274_leaf3
   - 0.0028903354 t274_leaf4 + 0.00621055719 t275_leaf0
   - 6.3371221800000003e-04 t275_leaf1 + 7.4485671899999996e-04 t275_leaf2
   - 2.2396835300000001e-04 t275_leaf3 + 0.00468770042 t275_leaf4
   + 0.00147050933 t276_leaf0 + 1.62687e-06 t276_leaf1
   - 0.00136045087 t276_leaf2 + 0.00377469463 t276_leaf4
   - 2.33378e-05 t277_leaf0 + 0.00617275201 t277_leaf1
   - 0.00147706945 t277_leaf2 + 0.00146086968 t277_leaf3
   - 0.00259306352 t277_leaf5 + 0.00432141405 t277_leaf6
   - 0.00337299518 t278_leaf0 + 4.2686378599999999e-04 t278_leaf1
   - 0.00063616 t278_leaf2 - 0.0021892956 t278_leaf3
   - 0.00507613178 t279_leaf0 - 1.09629713e-04 t279_leaf1
   - 0.000168877 t279_leaf2 + 0.000606462 t279_leaf3
   - 0.00160181394 t279_leaf4 + 0.00150534371 t279_leaf5
   - 0.00207138457 t280_leaf0 + 0.00102000986 t280_leaf1
   + 0.0051493044 t280_leaf2 - 0.000170987 t280_leaf3
   + 0.00345125422 t280_leaf4 + 0.00247156643 t280_leaf5
   - 0.00201363 t280_leaf6 + 0.0038503136 t281_leaf0
   + 0.000510216 t281_leaf1 - 0.000483895 t281_leaf2
   + 3.9146453499999998e-04 t281_leaf3 + 0.00475307135 t281_leaf4
   - 6.4398453100000005e-04 t282_leaf0 + 1.74858811e-04 t282_leaf1
   - 0.00318361796 t282_leaf2 + 0.00484835077 t282_leaf3
   + 0.00386169809 t283_leaf0 - 6.49289e-05 t283_leaf1
   - 9.1451447200000002e-04 t283_leaf2 + 8.05617e-05 t283_leaf3
   + 2.2698544500000001e-04 t283_leaf4 + 0.00418088026 t283_leaf5
   + 1.4291050299999999e-04 t284_leaf0 - 0.00288824295 t284_leaf1
   - 0.00158694596 t284_leaf2 + 0.00739445258 t284_leaf3
   - 0.00380889582 t284_leaf4 + 0.00223277416 t285_leaf0
   + 3.35463e-05 t285_leaf1 - 4.9345928699999996e-04 t285_leaf2
   + 0.00218630931 t285_leaf3 + 0.00751385605 t285_leaf4
   - 6.4810167500000001e-04 t285_leaf5 - 0.000123455 t286_leaf0
   + 0.00700934045 t286_leaf1 + 1.55661e-05 t286_leaf2
   - 0.0022793063 t286_leaf3 + 0.00366440089 t286_leaf4
   - 7.9380057299999998e-04 t286_leaf5 + 0.00752031105 t287_leaf0
   - 0.00123848696 t287_leaf1 - 1.58184281e-04 t287_leaf2
   + 5.1618518799999996e-04 t287_leaf3 - 0.00377571257 t287_leaf4
   + 8.8647031200000004e-04 t287_leaf5 - 0.000256852 t288_leaf0
   - 0.00362383109 t288_leaf1 + 0.000524015 t288_leaf2
   - 0.000748236 t288_leaf3 + 0.00531673897 t288_leaf4
   - 2.49406e-05 t288_leaf5 + 0.000418141 t289_leaf0
   - 0.00140096224 t289_leaf1 - 0.00253972411 t289_leaf2
   + 0.00114682 t289_leaf3 - 0.00379769574 t289_leaf4
   - 0.000173191 t289_leaf5 + 1.47766215e-04 t290_leaf0
   - 0.00314578507 t290_leaf1 - 3.7835980800000001e-04 t290_leaf2
   - 0.00253154058 t290_leaf3 + 8.4055145299999996e-04 t290_leaf4
   + 0.00486761937 t290_leaf5 + 6.27366e-05 t291_leaf0
   - 0.00346763525 t291_leaf1 - 0.000826925 t291_leaf2
   - 0.00401142985 t291_leaf3 + 0.00395620847 t291_leaf4
   - 0.00352033926 t292_leaf0 + 0.000517997 t292_leaf1
   + 8.52992e-05 t292_leaf2 - 0.00401593931 t292_leaf3
   - 0.00239975704 t292_leaf4 + 3.9791179000000002e-04 t293_leaf0
   - 2.53668346e-04 t293_leaf1 - 0.00182697643 t293_leaf2
   + 8.9287053599999995e-04 t293_leaf3 - 0.00282241148 t293_leaf4
   - 1.1387186499999999e-04 t294_leaf0 - 0.00339871133 t294_leaf1
   + 0.00483922428 t294_leaf2 + 0.00464526471 t294_leaf4
   + 1.59116855e-04 t295_leaf0 - 0.00321378186 t295_leaf1
   + 0.00157341524 t295_leaf2 - 0.00182101491 t295_leaf3
   + 0.000188932 t295_leaf4 - 6.7284429700000003e-04 t295_leaf5
   - 0.000122089 t296_leaf0 + 0.00173406524 t296_leaf1
   - 0.00340278423 t296_leaf2 - 0.00151683472 t296_leaf3
   + 0.00193712453 t296_leaf4 + 0.00594010809 t296_leaf5
   + 2.0711979600000001e-04 t297_leaf0 - 0.00253592432 t297_leaf1
   - 0.0010065902 t297_leaf2 - 0.00152814423 t297_leaf3
   + 0.00102922658 t298_leaf0 - 3.7586656999999998e-04 t298_leaf1
   + 6.9690641200000005e-04 t298_leaf2 - 7.10505177e-04 t298_leaf3
   + 0.0053263274 t298_leaf4 + 0.00252569746 t299_leaf0
   - 4.4855123299999999e-04 t299_leaf1 + 0.00213559112 t299_leaf2
   + 0.000180841 t299_leaf3 - 0.00135933131 t299_leaf4
   + 0.0019838782 t299_leaf5 + 0.00720849726 t299_leaf6
   + 1.58873358e-04 t300_leaf0 - 0.00115301809 t300_leaf1
   - 0.00322986091 t300_leaf2 - 0.00398380356 t300_leaf3
   - 0.00120848475 t300_leaf4 + 0.00484970491 t300_leaf5
   - 3.4877139799999998e-04 t301_leaf0 + 0.00296033896 t301_leaf1
   - 6.1520020200000005e-04 t301_leaf2 + 1.71133317e-04 t301_leaf3
   + 0.00489132106 t301_leaf4 - 7.3663832199999998e-04 t302_leaf0
   + 1.9608880400000001e-04 t302_leaf1 - 0.00118370447 t302_leaf2
   - 0.00326862186 t302_leaf3 + 1.6021485499999999e-04 t303_leaf0
   - 9.9454948199999998e-04 t303_leaf1 - 0.00291492417 t303_leaf2
   - 0.00104555523 t303_leaf3 + 0.00812408794 t303_leaf4
   - 9.5099623999999998e-04 t304_leaf0 + 1.80124349e-04 t304_leaf1
   + 7.3888147000000001e-04 t304_leaf2 + 0.00383320893 t304_leaf3
   - 0.00166422955 t304_leaf4 + 0.00199022307 t305_leaf1
   - 1.23078702e-04 t305_leaf2 + 0.00148243934 t305_leaf3
   + 0.00520661892 t305_leaf4 - 9.0381765000000001e-04 t306_leaf0
   + 0.00788646191 t306_leaf1 - 0.000841667 t306_leaf2
   + 0.00276447274 t306_leaf3 - 0.0017813443 t306_leaf4
   - 1.71705e-05 t306_leaf5 + 1.28116e-05 t307_leaf0
   - 0.00395599427 t307_leaf1 + 0.00189880235 t307_leaf2
   + 1.77854847e-04 t307_leaf3 - 0.000747022 t307_leaf4
   - 0.00428922288 t307_leaf5 + 2.6954617400000002e-04 t308_leaf0
   - 8.0888014099999997e-04 t308_leaf1 - 0.00206334912 t308_leaf2
   + 8.1080343800000001e-04 t308_leaf3 - 0.00147733313 t308_leaf4
   - 1.24125378e-04 t309_leaf0 - 0.00386918196 t309_leaf1
   + 2.5943131200000002e-04 t309_leaf2 + 0.00634328229 t309_leaf3
   + 0.00678100064 t309_leaf4 - 7.8538153299999997e-04 t309_leaf5
   + 0.000536933 t310_leaf0 - 0.0036119679 t310_leaf1
   + 0.00215767254 t310_leaf2 - 3.6949e-05 t310_leaf3
   + 0.00365578989 t310_leaf4 + 3.9581954499999999e-04 t311_leaf0
   - 0.00208416721 t311_leaf1 - 4.1437486700000002e-04 t311_leaf2
   - 0.00160176645 t311_leaf3 + 0.00165890774 t312_leaf0
   + 2.6512972499999999e-04 t312_leaf1 - 0.00102446356 t312_leaf2
   - 3.22167e-05 t312_leaf3 + 0.00560738845 t312_leaf4
   + 0.00133982208 t313_leaf0 - 0.00257796329 t313_leaf1
   + 0.00833278615 t313_leaf2 + 5.7960068799999996e-04 t313_leaf3
   - 0.00151081663 t313_leaf4 + 1.28523e-05 t313_leaf5
   + 0.00454950612 t313_leaf6 + 0.00200938107 t314_leaf0
   - 5.40281e-05 t314_leaf1 - 5.7770241999999998e-04 t314_leaf2
   + 0.000150492 t314_leaf3 + 7.6886039499999995e-04 t314_leaf4
   + 0.00423714146 t314_leaf5 - 0.0017895808 t314_leaf6
   - 0.00257282355 t315_leaf0 + 8.5290061500000001e-04 t315_leaf1
   + 0.00119856338 t315_leaf2 - 4.15998e-05 t315_leaf3
   - 0.00303742406 t315_leaf4 + 0.00216715457 t316_leaf0
   + 9.87014e-05 t316_leaf1 + 3.4176014000000002e-04 t316_leaf2
   + 0.0073144543 t316_leaf3 - 0.00108855055 t316_leaf4
   - 5.17360168e-04 t317_leaf0 + 1.7728147200000001e-04 t317_leaf1
   - 0.00172652153 t317_leaf2 + 0.00171588711 t317_leaf3
   + 0.00354841375 t318_leaf0 - 6.97737e-05 t318_leaf1
   - 0.0016958334 t318_leaf2 + 0.00260068662 t318_leaf3
   - 2.6291116999999998e-04 t319_leaf0 + 0.00295241131 t319_leaf1
   + 8.23687e-06 t319_leaf2 + 0.00240364322 t319_leaf3
   + 0.00154703076 t319_leaf4 + 0.00477596652 t320_leaf0
   - 4.6765865400000002e-04 t320_leaf1 - 1.68398183e-04 t320_leaf2
   - 0.00200196262 t320_leaf3 + 0.00496319355 t320_leaf4
   + 0.000490511 t320_leaf5 + 0.00439156778 t320_leaf6
   + 1.8520858399999999e-04 t321_leaf0 - 6.1468756800000005e-04 t321_leaf1
   + 0.00425115647 t321_leaf2 - 0.00546862464 t321_leaf3
   - 0.00547229545 t322_leaf0 - 2.39806e-05 t322_leaf1
   - 0.00170430134 t322_leaf2 + 9.1492035400000005e-04 t322_leaf3
   + 0.00176575838 t322_leaf4 + 4.60345e-05 t322_leaf5
   - 0.00120398414 t322_leaf6 + 1.8011948900000001e-04 t323_leaf0
   - 4.5238624399999999e-04 t323_leaf1 + 0.00341132563 t323_leaf2
   - 0.00141889043 t323_leaf3 - 0.000321079 t324_leaf0
   - 0.0022959339 t324_leaf1 + 0.00267543341 t324_leaf2
   + 0.00258223596 t324_leaf3 - 0.000492911 t324_leaf4
   - 3.74294e-05 t324_leaf5 + 0.00250418065 t324_leaf6
   - 3.2058279599999999e-04 t325_leaf0 + 2.9336058699999998e-04 t325_leaf1
   + 0.00478067528 t325_leaf2 - 0.00189606089 t325_leaf4
   + 0.00127413636 t326_leaf0 - 1.75343e-05 t326_leaf1
   - 0.00161477295 t326_leaf2 - 0.0024579973 t326_leaf3
   + 0.00132498739 t327_leaf0 + 1.73690671e-04 t327_leaf1
   - 7.45741e-05 t327_leaf2 - 0.00146552688 t327_leaf3
   - 0.00120858278 t327_leaf4 + 2.2750486100000001e-04 t328_leaf0
   + 0.00507151894 t328_leaf1 + 2.04943717e-04 t328_leaf2
   - 0.00254847761 t328_leaf3 - 1.42290141e-04 t328_leaf4
   + 0.00169705914 t328_leaf5 + 0.00238882774 t329_leaf1
   + 0.00281935278 t329_leaf2 - 0.00112520216 t329_leaf3
   + 0.000571332 t329_leaf4 + 0.00366683095 t329_leaf5
   - 0.00142618536 t329_leaf6 + 1.2879574200000001e-04 t330_leaf0
   + 0.00585116027 t330_leaf1 + 0.00014276 t330_leaf2
   - 0.00132541044 t330_leaf3 - 0.00240255473 t330_leaf4
   + 6.8793044199999998e-04 t330_leaf5 - 8.38667e-05 t331_leaf0
   - 0.0024010248 t331_leaf1 - 0.00171459641 t331_leaf2
   + 0.00263206474 t331_leaf3 + 0.00419934792 t331_leaf4
   + 9.6414849400000004e-04 t332_leaf0 + 3.47737e-05 t332_leaf1
   - 0.00109788124 t332_leaf2 + 0.0062632584 t332_leaf3
   + 4.09268e-05 t333_leaf0 - 0.00394835416 t333_leaf1
   + 0.00231287722 t333_leaf2 + 1.6537464400000001e-04 t334_leaf0
   + 0.00309188012 t334_leaf1 - 2.53202e-05 t334_leaf2
   - 0.00109486771 t334_leaf3 - 8.8897277599999996e-04 t334_leaf4
   + 9.6699077399999998e-04 t334_leaf5 + 0.00886969082 t334_leaf6
   + 5.2639783799999997e-04 t335_leaf0 - 0.0016338682 t335_leaf1
   - 0.00104343833 t335_leaf2 + 4.48553183e-04 t335_leaf3
   - 1.5567545799999999e-04 t335_leaf4 - 0.00187155872 t335_leaf5
   - 0.00160752167 t336_leaf0 + 3.1359141600000002e-04 t336_leaf1
   - 9.6286024199999996e-04 t336_leaf2 + 5.9848232200000002e-04 t336_leaf3
   - 0.00189860119 t336_leaf4 + 0.00199509435 t336_leaf5
   + 0.000590066 t336_leaf6 - 0.00220652088 t336_leaf7
   + 0.000795731 t337_leaf0 - 2.9138824899999999e-04 t337_leaf1
   - 0.00142083864 t337_leaf2 + 3.64473e-05 t337_leaf3
   + 0.00032395 t337_leaf4 - 0.00113447732 t337_leaf5
   - 0.00367750041 t338_leaf0 + 0.00122654194 t338_leaf1
   - 2.2252574899999999e-04 t338_leaf2 - 0.00193526736 t338_leaf3
   + 0.00471969973 t338_leaf4 - 4.8723581099999999e-04 t338_leaf5
   + 5.0463870899999995e-04 t338_leaf6 - 1.17629337e-04 t339_leaf0
   + 9.6140347900000003e-04 t339_leaf1 - 0.00278302026 t339_leaf2
   + 0.00128122873 t339_leaf3 - 3.8778586900000003e-04 t339_leaf4
   + 0.00536413677 t339_leaf5 - 0.00132108782 t340_leaf0
   + 0.00147629 t340_leaf1 - 0.00117163546 t340_leaf2
   + 0.00104023027 t340_leaf3 + 0.00484620966 t340_leaf4
   + 1.2337842799999999e-04 t340_leaf5 + 0.00254626852 t340_leaf6
   - 2.86039e-05 t341_leaf0 - 0.00111355365 t341_leaf1
   + 0.00553426892 t341_leaf2 - 0.00123084709 t342_leaf0
   - 3.6714240599999999e-04 t342_leaf1 + 0.00451623881 t342_leaf2
   + 1.78188202e-04 t342_leaf3 + 0.00235758652 t342_leaf4
   + 8.9472380899999996e-04 t343_leaf0 - 1.01114449e-04 t343_leaf1
   - 0.00394161139 t343_leaf2 + 0.00475047017 t343_leaf4
   + 3.8082379599999999e-04 t344_leaf0 - 0.00262577645 t344_leaf1
   + 0.00469167205 t344_leaf2 + 0.00045507 t344_leaf3
   - 1.82502874e-04 t344_leaf4 + 5.7015364300000003e-04 t344_leaf5
   + 0.00414394168 t344_leaf6 - 5.4144015299999995e-04 t345_leaf0
   + 0.00437270664 t345_leaf1 - 0.00313811586 t345_leaf2
   + 2.6947187100000001e-04 t345_leaf3 - 9.11347859e-04 t345_leaf4
   - 9.3254e-05 t345_leaf5 + 0.00405573705 t345_leaf6
   + 0.00139602 t346_leaf0 - 3.6008350400000002e-04 t346_leaf1
   + 3.2259989500000002e-04 t346_leaf2 + 0.0024625957 t346_leaf3
   - 0.000830595 t346_leaf4 + 8.99141e-05 t347_leaf0
   - 0.00227420754 t347_leaf1 - 0.00256343302 t347_leaf2
   - 0.00367106148 t347_leaf3 - 3.2897887299999999e-04 t348_leaf0
   + 3.9366612299999998e-04 t348_leaf1 - 0.00101511332 t348_leaf2
   - 5.4611428600000005e-04 t348_leaf3 - 0.003374035 t348_leaf4
   + 9.19571e-05 t349_leaf0 - 0.00240908726 t349_leaf1
   - 0.00245076302 t349_leaf2 + 9.8818866500000003e-04 t349_leaf3
   + 0.00690844795 t349_leaf4 - 3.2475148299999999e-04 t350_leaf0
   + 0.00973814446 t350_leaf1 - 2.9134179899999998e-04 t350_leaf2
   + 6.0967245400000004e-04 t350_leaf3 - 0.00144730578 t350_leaf4
   - 2.1780614e-04 t350_leaf5 + 4.5981982799999998e-04 t350_leaf6
   - 0.00104373146 t351_leaf0 - 1.92786e-06 t351_leaf1
   + 0.00208109221 t351_leaf2 - 0.00127779273 t351_leaf3
   + 0.00186203606 t351_leaf4 - 0.00113667944 t351_leaf5
   + 7.73825e-05 t352_leaf0 - 0.00135260168 t352_leaf1
   + 6.2375e-05 t353_leaf0 - 0.00325105037 t353_leaf1
   - 0.00306166941 t353_leaf2 + 0.00588754145 t353_leaf4
   - 1.15324125e-04 t354_leaf0 + 0.00272124121 t354_leaf1
   - 3.3509e-07 t354_leaf2 + 0.00180337729 t355_leaf0
   - 0.00101348816 t355_leaf1 - 0.0018810227 t355_leaf2
   + 8.35823e-06 t355_leaf3 - 3.6670724500000002e-04 t355_leaf4
   + 0.00122590235 t355_leaf5 - 0.00281038252 t355_leaf6
   + 9.4813259799999996e-04 t356_leaf0 - 0.00461664191 t356_leaf1
   + 1.6723187599999999e-04 t356_leaf2 + 2.66287389e-04 t356_leaf3
   - 0.00154892378 t356_leaf4 - 0.00185120688 t356_leaf5
   - 1.93905711e-04 t356_leaf6 + 1.69247e-05 t357_leaf0
   + 0.00356943929 t357_leaf1 - 0.00181192113 t357_leaf2
   + 0.0013361465 t357_leaf3 + 6.07915e-06 t358_leaf0
   + 8.0617424100000001e-04 t358_leaf1 - 0.00104942918 t358_leaf2
   - 0.00565980794 t358_leaf3 - 0.000246672 t358_leaf4
   + 6.4163270900000004e-04 t358_leaf5 - 0.00307267508 t358_leaf6
   - 0.000198361 t359_leaf0 + 4.4679967700000001e-04 t359_leaf1
   + 7.1381719300000004e-04 t359_leaf2 - 0.00323616038 t359_leaf3
   - 9.7323471e-04 t359_leaf4 + 0.00727135781 t359_leaf5
   + 5.4081249999999995e-04 t359_leaf6 + 0.00438145734 t360_leaf0
   + 2.39074e-05 t360_leaf1 + 0.00131875603 t361_leaf0
   - 8.9658022599999998e-04 t361_leaf1 - 1.57023e-05 t361_leaf2
   + 0.00226311898 t361_leaf3 + 0.0030483352 t362_leaf0
   + 2.7732460900000002e-04 t362_leaf1 + 1.9260935399999999e-04 t362_leaf2
   - 0.00100800581 t362_leaf3 + 2.6348570799999998e-04 t362_leaf4
   - 1.1531873999999999e-04 t363_leaf0 + 8.7609561199999996e-04 t363_leaf1
   + 0.00123268459 t363_leaf2 - 0.00192875951 t363_leaf3
   + 0.0044528665 t363_leaf4 + 0.000244283 t364_leaf0
   - 0.0030041954 t364_leaf1 + 1.29988432e-04 t364_leaf2
   - 7.7662832300000003e-04 t364_leaf3 + 0.0029657376 t364_leaf4
   - 1.2490571000000001e-04 t365_leaf0 - 0.00339467661 t365_leaf1
   - 1.43663e-05 t365_leaf2 + 0.00242181052 t365_leaf3
   + 0.00363880442 t365_leaf4 + 7.0282735400000005e-04 t366_leaf0
   - 0.00060398 t366_leaf1 + 0.00358657353 t366_leaf2
   + 3.0503582100000002e-04 t366_leaf3 - 0.00132889487 t366_leaf4
   - 0.00200078031 t367_leaf0 - 3.3182e-05 t367_leaf1
   + 0.00164411799 t367_leaf2 + 0.00364097278 t367_leaf3
   + 0.00169384549 t368_leaf1 - 0.00221303455 t368_leaf2
   - 1.78963252e-04 t368_leaf3 + 0.00350425486 t368_leaf4
   + 3.6625811499999998e-04 t369_leaf0 - 5.9993780499999997e-04 t369_leaf1
   - 9.86127881e-04 t369_leaf2 + 0.00108041416 t369_leaf3
   - 0.00350803626 t369_leaf4 - 6.66822e-05 t370_leaf0
   + 0.00289346185 t370_leaf1 + 0.0029481696 t370_leaf2
   + 0.000934239 t371_leaf0 - 0.000146695 t371_leaf1
   - 0.00176510413 t371_leaf2 + 0.002228759 t371_leaf3
   + 0.000246258 t371_leaf4 - 9.53049e-06 t372_leaf0
   + 0.00344563974 t372_leaf1 - 0.00110441621 t372_leaf2
   + 0.00420444692 t373_leaf0 + 8.3548715300000003e-04 t373_leaf1
   - 0.00163631851 t373_leaf2 - 2.46095e-05 t373_leaf3
   - 0.00291510974 t373_leaf4 + 6.00542e-05 t374_leaf0
   - 0.0018150229 t374_leaf1 - 0.00175649358 t374_leaf2
   - 2.5747044100000001e-04 t374_leaf3 + 7.6758617099999997e-04 t374_leaf4
   + 0.00571370265 t374_leaf5 + 1.5558078300000001e-04 t375_leaf0
   - 0.00259509939 t375_leaf1 + 0.00570314098 t375_leaf2
   - 0.00162639795 t375_leaf3 + 2.23543e-05 t376_leaf0
   - 1.11490153e-04 t376_leaf1 + 0.00270636939 t376_leaf2
   - 0.000870607 t376_leaf3 - 4.10596e-05 t377_leaf0
   - 0.000119988 t378_leaf0 + 0.0062052994 t378_leaf1
   - 8.10791e-05 t378_leaf2 + 0.00154903985 t379_leaf0
   - 1.5084838399999999e-04 t379_leaf1 - 0.00115351565 t379_leaf2
   + 0.00167667412 t379_leaf3 - 0.00125871366 t379_leaf4
   + 9.66369e-05 t380_leaf0 - 0.00128897745 t380_leaf1
   - 0.00360955158 t381_leaf0 + 4.2576077999999999e-04 t381_leaf1
   + 0.00407692138 t381_leaf2 + 2.0025757800000001e-04 t381_leaf3
   - 7.91411e-05 t381_leaf4 + 0.00573411537 t382_leaf1
   - 0.00182594161 t382_leaf2 - 1.02225873e-04 t382_leaf3
   - 5.58846e-05 t382_leaf4 + 0.003298203 t382_leaf5
   + 0.00284459046 t383_leaf0 - 1.26051324e-04 t383_leaf1
   + 0.000514235 t383_leaf2 + 0.00816446356 t383_leaf3
   - 0.000126954 t383_leaf4 + 0.00122088904 t383_leaf5
   + 0.00349958916 t383_leaf6 + 0.00107319234 t384_leaf0
   - 0.00131920539 t384_leaf1 + 0.00109354616 t384_leaf2
   + 6.04226e-05 t384_leaf3 - 0.001343232 t384_leaf4
   - 0.00037517 t385_leaf0 + 0.000280224 t385_leaf1
   + 0.00290644704 t385_leaf2 + 0.00811601803 t385_leaf3
   + 4.1769247000000002e-04 t385_leaf4 - 0.0034200931 t386_leaf0
   - 3.64248e-05 t386_leaf1 - 0.00115289621 t386_leaf2
   + 0.00493885716 t386_leaf3 - 7.49526e-05 t387_leaf0
   - 0.00137458125 t387_leaf1 + 0.00326618855 t387_leaf2
   + 0.00619797874 t387_leaf3 + 4.5852363099999999e-04 t387_leaf4
   + 0.00339740049 t388_leaf0 - 0.00158210401 t388_leaf1
   - 0.000143282 t388_leaf2 + 0.00127880485 t388_leaf3
   - 0.000378296 t389_leaf0 - 0.00329512078 t389_leaf1
   + 0.000125669 t389_leaf2 - 0.00159044052 t389_leaf3
   + 8.4482930800000003e-04 t390_leaf0 - 9.58778e-06 t390_leaf1
   - 0.00288443617 t390_leaf2 - 0.00379512133 t391_leaf0
   + 8.15551e-05 t391_leaf1 + 0.00296520512 t391_leaf2
   + 0.00389235443 t391_leaf3 + 1.20911893e-04 t392_leaf0
   - 0.00109406374 t392_leaf1 - 6.9375260500000003e-04 t392_leaf2
   + 0.00197936734 t392_leaf3 + 3.77216e-05 t393_leaf0
   - 0.00111777405 t393_leaf1 + 3.0965116500000003e-04 t393_leaf2
   + 0.00172916299 t393_leaf3 + 0.00397514738 t393_leaf4
   - 3.50809e-05 t394_leaf0 + 8.9998776099999998e-04 t394_leaf1
   + 2.9985737600000001e-04 t395_leaf0 - 4.7913927100000001e-04 t395_leaf1
   - 0.00120976509 t395_leaf2 + 0.0027840538 t395_leaf3
   + 0.00522371242 t395_leaf4 - 0.00284729502 t396_leaf0
   + 0.00709749153 t396_leaf1 - 0.00123687042 t396_leaf2
   - 3.00129e-05 t396_leaf3 + 0.0036496541 t396_leaf4
   + 6.2671e-05 t397_leaf0 - 0.00157591247 t397_leaf1
   + 0.00552642578 t397_leaf2 + 0.00674045598 t397_leaf4
   + 3.96722e-05 t398_leaf0 - 0.00319606625 t398_leaf1
   - 1.01516111e-04 t398_leaf2 - 0.00150868483 t398_leaf3
   - 0.00152912701 t399_leaf0 + 0.00251390133 t399_leaf1
   + 1.1560982200000001e-04 t399_leaf2 + 0.00351572316 t399_leaf3
   + 1.5940518600000001e-04 t400_leaf0 - 5.99127263e-04 t400_leaf1
   + 0.00381097454 t400_leaf2 - 4.53919e-05 t401_leaf0
   + 0.00564953871 t402_leaf0 - 0.00271584233 t402_leaf2
   - 3.18203e-05 t402_leaf3 + 0.0015112285 t402_leaf4
   - 6.86742e-05 t403_leaf0 - 0.00143347366 t403_leaf1
   + 0.00622800551 t403_leaf2 + 3.8811678000000001e-04 t403_leaf3
   + 0.00357764028 t403_leaf4 + 1.4120044900000001e-04 t404_leaf0
   - 0.00316424388 t404_leaf1 - 9.43218e-05 t404_leaf2
   + 8.0756814000000003e-04 t404_leaf3 - 5.67092e-05 t404_leaf4
   - 0.0022607781 t404_leaf5 + 0.00405680668 t405_leaf0
   - 6.5469043299999999e-04 t405_leaf1 - 0.00115582917 t405_leaf2
   - 4.26802e-05 t405_leaf3 + 0.00216861605 t405_leaf4
   - 5.18724e-05 t406_leaf0 + 1.3569834100000001e-04 t407_leaf0
   + 0.00133405684 t408_leaf0 - 3.89816e-05 t408_leaf1
   + 0.00642907387 t408_leaf2 - 0.00327889412 t408_leaf4
   + 0.00154963485 t409_leaf0 - 0.00291140634 t409_leaf1
   + 0.00187251053 t409_leaf2 + 7.90314e-05 t409_leaf3
   - 0.00199257606 t409_leaf4 + 0.00182182936 t409_leaf5
   - 0.000123291 t410_leaf0 + 0.000493505 t410_leaf1
   - 0.00122180837 t410_leaf2 - 0.00296994438 t410_leaf3
   + 1.02676255e-04 t411_leaf0 - 0.00183412735 t411_leaf1
   - 0.00147866795 t411_leaf2 - 1.3781222500000001e-04 t412_leaf0
   + 1.657424e-04 t412_leaf1 + 0.00224904669 t412_leaf2
   + 7.0841325200000005e-04 t413_leaf0 - 1.7018542e-04 t413_leaf1
   - 0.00127953081 t413_leaf2 - 0.0031722053 t413_leaf3
   - 9.6181e-05 t413_leaf4 + 0.000149366 t414_leaf0
   + 0.00138157711 t414_leaf1 - 0.00148376951 t414_leaf2
   - 0.00133324484 t414_leaf3 + 0.00179355696 t415_leaf0
   + 1.8799344300000001e-04 t415_leaf1 - 0.000723069 t415_leaf2
   + 1.4499225700000001e-04 t415_leaf3 - 0.00152141228 t415_leaf4
   + 8.1245641899999998e-04 t416_leaf0 - 0.00248082401 t416_leaf1
   + 0.00874423329 t416_leaf2 + 0.000800903 t416_leaf3
   - 9.07759e-05 t416_leaf4 + 0.00321902148 t416_leaf5
   - 0.00285623595 t416_leaf6 + 2.53374e-05 t417_leaf0
   - 7.10422e-06 t418_leaf0 - 0.00395803805 t418_leaf1
   - 4.5501574600000002e-04 t419_leaf0 + 0.00510223862 t419_leaf1
   + 0.00102367788 t419_leaf2 - 0.0017822017 t419_leaf3
   + 0.00154519407 t419_leaf4 + 2.02048861e-04 t419_leaf5
   - 7.4598717000000001e-04 t419_leaf6 + 1.89153579e-04 t419_leaf7
   - 2.20677e-05 t420_leaf0 + 0.00704177702 t421_leaf1
   - 0.001020064 t421_leaf2 - 9.79082e-05 t421_leaf3
   + 0.00362774753 t422_leaf0 + 2.5098555500000002e-04 t422_leaf1
   - 2.6379738100000002e-04 t422_leaf2 + 0.00156404963 t422_leaf3
   + 0.00287776673 t422_leaf4 + 0.00159647933 t423_leaf0
   + 5.85356e-05 t423_leaf1 - 0.00161538564 t423_leaf2
   - 0.00311105628 t423_leaf3 + 0.00656476011 t424_leaf0
   - 0.00122000021 t424_leaf1 + 0.00125343387 t424_leaf2
   - 8.5437722699999998e-04 t424_leaf3 - 2.21706e-06 t424_leaf4
   - 0.00196333393 t424_leaf5 + 0.00112953514 t425_leaf0
   - 6.66121e-05 t425_leaf1 - 0.00167220412 t425_leaf2
   - 0.00111985102 t426_leaf0 + 4.23271e-05 t426_leaf1
   + 0.00418563979 t426_leaf2 - 0.00165348849 t426_leaf3
   + 0.00351644098 t426_leaf4 - 0.00367743196 t426_leaf5
   + 5.8089330600000005e-04 t426_leaf6 - 5.99333e-05 t427_leaf0
   + 0.00223484146 t427_leaf1 + 0.0037957069 t427_leaf2
   - 0.00265531242 t427_leaf3 + 0.00112425699 t428_leaf0
   - 0.00129858858 t428_leaf1 - 0.00202616863 t428_leaf2
   + 4.20229e-05 t428_leaf3 - 0.00239025755 t428_leaf4
   - 2.41433765e-04 t429_leaf0 - 0.00178461825 t429_leaf1
   + 1.2924273300000001e-04 t429_leaf2 + 0.00388094597 t429_leaf3
   + 5.4773938599999997e-04 t429_leaf4 + 0.00592315244 t429_leaf5
   - 9.4874214799999998e-04 t430_leaf0 + 0.00206570025 t430_leaf1
   + 0.00333172013 t430_leaf2 + 4.0651843299999998e-04 t430_leaf3
   - 2.6200636099999998e-04 t430_leaf4 + 0.00361612928 t430_leaf5
   - 0.00229765777 t430_leaf6 + 1.20824e-05 t431_leaf0
   - 0.00164907135 t431_leaf1 + 0.00275625475 t431_leaf2
   + 0.00231019733 t431_leaf3 + 7.57146e-05 t432_leaf0
   + 0.00198957277 t432_leaf1 - 0.00233209273 t432_leaf2
   + 0.0031641759 t432_leaf3 - 0.00101634313 t432_leaf4
   - 0.0036887941 t432_leaf5 + 2.15669e-05 t433_leaf0
   - 0.0018848211 t433_leaf1 - 0.00173518609 t433_leaf2
   + 1.14133e-05 t434_leaf0 + 0.00197329186 t435_leaf0
   - 0.00116715243 t435_leaf1 + 1.7193031099999999e-04 t435_leaf2
   - 6.3971267099999999e-04 t435_leaf3 - 0.00151438336 t435_leaf4
   - 5.75961e-05 t436_leaf0 + 0.00225907238 t437_leaf0
   - 0.00330276927 t437_leaf1 + 5.73463e-05 t437_leaf2
   + 6.7041115800000001e-04 t438_leaf0 - 1.2953938900000001e-04 t438_leaf1
   + 9.94534e-05 t439_leaf0 - 0.00250694621 t439_leaf1
   + 0.00289349793 t439_leaf3 - 0.00102203828 t439_leaf4
   - 6.5353076200000003e-04 t440_leaf0 - 5.3174415400000005e-04 t440_leaf1
   + 0.00432404736 t440_leaf2 + 0.00378699205 t440_leaf3
   + 4.6106247499999998e-04 t440_leaf4 + 7.34576e-05 t440_leaf5
   + 1.66527549e-04 t441_leaf0 - 0.00294450903 t441_leaf1
   - 2.827e-05 t441_leaf2 - 8.1279623599999999e-04 t441_leaf3
   + 0.00276922062 t441_leaf4 + 0.000136291 t442_leaf0
   - 0.00235576206 t442_leaf1 - 0.00293414062 t442_leaf2
   - 1.34487e-05 t443_leaf0 + 0.00102591398 t444_leaf0
   - 1.2149760700000001e-04 t444_leaf1 - 0.00137945823 t444_leaf2
   + 0.00161178317 t444_leaf3 + 3.8472726000000001e-04 t444_leaf4
   + 8.32588e-05 t445_leaf0 + 0.00105086854 t445_leaf1
   - 0.00349826342 t445_leaf2 - 6.2377279399999995e-04 t446_leaf0
   + 0.00429079775 t446_leaf1 + 6.7870889299999999e-04 t446_leaf2
   - 1.9123719500000001e-04 t446_leaf3 + 2.56255e-05 t447_leaf0
   - 0.00159084669 t447_leaf1 + 0.0024496105 t447_leaf2
   + 1.33694e-05 t448_leaf0 - 0.00188902114 t448_leaf1
   + 0.00372307422 t448_leaf2 - 2.8449261999999998e-04 t449_leaf0
   + 0.00147053401 t449_leaf1 + 2.24038e-05 t449_leaf2
   - 0.00169612793 t449_leaf3 - 0.00170219922 t449_leaf4
   + 0.00181391276 t449_leaf5 - 7.1119e-05 t450_leaf0
   - 9.4291230199999995e-04 t451_leaf0 + 4.75934e-05 t451_leaf1
   + 0.00346613443 t451_leaf2 + 0.00653374707 t451_leaf3
   - 0.00102886849 t451_leaf4 + 1.8928557999999999e-04 t451_leaf5
   - 0.001233013 t452_leaf0 + 0.0015077726 t452_leaf1
   + 2.04389e-05 t452_leaf2 - 0.0044702366 t452_leaf3
   + 0.00181529694 t452_leaf4 + 1.52129185e-04 t453_leaf0
   + 0.00203759572 t453_leaf1 - 0.00161340751 t453_leaf2
   - 1.0846368199999999e-04 t453_leaf3 + 0.00585953332 t453_leaf4
   + 3.3895132899999997e-04 t453_leaf5 + 3.29889e-05 t454_leaf0
   + 0.00272291969 t454_leaf1 - 1.73655e-05 t455_leaf0
   + 2.44113e-05 t456_leaf0 + 9.32154e-05 t457_leaf0
   + 0.00420576334 t457_leaf1 - 5.9827574299999996e-04 t457_leaf2
   + 7.3989847399999996e-04 t458_leaf0 - 1.2489067700000001e-04 t458_leaf1
   + 0.00260046916 t458_leaf2 - 0.00241280044 t458_leaf3
   - 4.89966187e-04 t459_leaf0 + 1.8072089000000001e-04 t459_leaf1
   + 2.68539e-05 t460_leaf0 - 0.00365525507 t461_leaf0
   + 1.11956e-05 t461_leaf1 + 0.00167014566 t461_leaf2
   + 0.00211073062 t461_leaf3 - 0.00182310585 t461_leaf4
   + 3.84204e-05 t462_leaf0 - 0.00330180908 t462_leaf1
   + 0.0026623155 t463_leaf0 - 0.00154995278 t463_leaf2
   - 0.000113712 t463_leaf3 + 0.00272678374 t463_leaf4
   + 1.98402515e-04 t463_leaf5 + 0.00614634017 t463_leaf6
   + 3.2911344900000001e-04 t464_leaf0 - 0.00144829368 t464_leaf1
   - 2.991418e-04 t464_leaf2 - 0.00351026165 t465_leaf0
   + 0.00177727162 t465_leaf1 + 6.13312e-05 t465_leaf2
   - 0.0013338984 t465_leaf3 + 0.00141628331 t466_leaf0
   - 0.00202802522 t466_leaf1 + 0.000123557 t466_leaf2
   + 1.19685e-05 t466_leaf3 - 0.00191104854 t466_leaf4
   - 7.3682e-05 t467_leaf0 - 2.89562e-05 t468_leaf0
   - 3.13081e-05 t469_leaf0 + 0.00376264658 t470_leaf0
   + 0.000159792 t470_leaf1 - 0.00231862627 t470_leaf2
   - 1.4720457000000001e-04 t471_leaf0 + 0.00247577345 t471_leaf1
   - 6.4353575000000002e-04 t471_leaf2 + 0.00312447967 t471_leaf3
   + 7.76921e-05 t472_leaf0 - 2.63081e-05 t473_leaf0
   + 0.00129485235 t473_leaf1 - 0.00110369059 t473_leaf2
   - 0.00197753659 t474_leaf0 + 7.7988e-05 t474_leaf1
   - 0.00148342282 t474_leaf2 - 0.00272118254 t475_leaf1
   + 0.000145236 t475_leaf2 + 0.00314609 t476_leaf0
   + 1.4161242899999999e-04 t476_leaf1 - 0.00178857194 t476_leaf2
   - 8.7832e-05 t476_leaf3 + 0.00122044783 t477_leaf0
   - 2.8364255600000002e-04 t477_leaf1 + 0.000864946 t477_leaf2
   - 8.26902e-05 t477_leaf3 + 8.57525e-05 t478_leaf0
   + 0.00348374667 t478_leaf1 - 0.000378845 t478_leaf2
   - 0.00267980504 t478_leaf3 + 8.7085965799999995e-04 t479_leaf0
   - 1.81842552e-04 t479_leaf1 - 0.00293440721 t479_leaf2
   + 0.00111641025 t479_leaf3 - 0.00176628842 t479_leaf4
   + 0.00552282017 t479_leaf5 - 1.29792e-05 t480_leaf0
   + 0.00113970845 t480_leaf1 - 0.00158561114 t480_leaf2
   + 0.000766751 t480_leaf3 - 0.00192926906 t480_leaf4
   + 0.00127546699 t481_leaf0 - 1.4352117400000001e-04 t481_leaf1
   - 1.2552221600000001e-04 t482_leaf0 - 3.2512e-05 t483_leaf0
   + 1.02750164e-04 t484_leaf0 - 0.00226877071 t484_leaf1
   + 0.00341905886 t484_leaf2 + 0.00394834392 t484_leaf3
   - 1.0345382899999999e-04 t485_leaf0 - 3.88308e-05 t487_leaf0
   - 0.00292010279 t488_leaf0 - 0.00347999088 t488_leaf1
   + 5.75834e-05 t488_leaf2 + 2.0868027099999999e-04 t488_leaf3
   - 4.66644e-05 t489_leaf0 + 0.0017986407 t490_leaf1
   - 0.00103469985 t490_leaf2 + 3.5199627700000001e-04 t491_leaf0
   - 0.00136651401 t491_leaf1 + 6.6548e-06 t491_leaf2
   + 3.73000832e-04 t491_leaf3 + 1.41125522e-04 t492_leaf0
   - 0.00266780611 t492_leaf1 - 0.00200483133 t492_leaf2
   + 9.8460342299999996e-04 t493_leaf0 - 0.00134595216 t493_leaf1
   - 2.0456e-05 t493_leaf2 - 0.00149051705 t493_leaf3
   + 0.0010529228 t493_leaf4 - 4.54372e-06 t494_leaf0
   + 5.6632346199999998e-04 t494_leaf1 + 0.00397269428 t494_leaf2
   - 0.00321686221 t495_leaf0 - 6.21794e-05 t495_leaf1
   - 0.00301175099 t496_leaf0 - 2.88821e-05 t496_leaf1
   + 0.00180305575 t497_leaf0 - 9.6271414100000002e-04 t497_leaf1
   - 5.18376e-05 t497_leaf2 + 0.002087828 t497_leaf3
   + 0.000159188 t498_leaf0 - 0.000818547 t498_leaf1
   + 8.6316186899999998e-04 t498_leaf2 - 0.00242004986 t498_leaf3
   - 5.5514072300000001e-04 t498_leaf4 + 1.2989759900000001e-04 t499_leaf0
   - 0.00142675801 t499_leaf1 + 4.07941e-05 t500_leaf0
   + 0.00300787762 t500_leaf1 - 0.00129570067 t500_leaf2
   - 0.00146840478 t500_leaf3 + 0.00254795491 t501_leaf0
   - 3.93916e-05 t501_leaf1 + 0.00644635316 t501_leaf2
   - 2.5085546100000002e-04 t501_leaf3 + 0.00360364304 t501_leaf4
   - 3.8084667199999998e-04 t501_leaf5 + 0.00160909793 t501_leaf6
   + 0.00377556635 t502_leaf0 - 1.9624820600000001e-04 t502_leaf1
   - 8.75945e-05 t503_leaf0 - 0.000511391 t504_leaf0 + 0.0038615 t504_leaf1
   - 3.7141461500000001e-04 t504_leaf2 + 2.67432e-06 t504_leaf3
   + 2.4011217500000001e-04 t505_leaf0 - 5.2660086700000001e-04 t505_leaf1
   + 2.33877363e-04 t506_leaf0 - 0.00147870381 t506_leaf1
   - 0.00134411163 t506_leaf2 - 0.00290927896 t506_leaf3
   + 1.3383560899999999e-04 t507_leaf0 - 7.1626342799999999e-04 t507_leaf1
   - 5.3725310100000004e-04 t508_leaf0 + 8.8796170900000003e-04 t508_leaf1
   - 0.000206968 t508_leaf2 + 0.00105185027 t508_leaf3
   - 0.0010938691 t508_leaf4 + 0.00482804701 t508_leaf5
   + 1.4307655499999999e-04 t509_leaf0 + 0.001189298 t509_leaf1
   - 4.4647930199999999e-04 t509_leaf2 - 6.60479e-05 t510_leaf0
   - 3.43176e-05 t512_leaf0 - 0.00199722941 t513_leaf0
   + 9.4344420400000004e-04 t513_leaf1 + 0.000176679 t513_leaf2
   - 0.00154657848 t513_leaf3 - 2.02906158e-04 t513_leaf4
   + 0.00518112211 t513_leaf5 + 6.07054e-05 t514_leaf0
   - 0.00107792637 t515_leaf0 + 0.00353061664 t515_leaf1
   - 9.84066e-05 t515_leaf2 + 0.001209234 t515_leaf3
   - 4.41536657e-04 t516_leaf0 + 0.00154177763 t516_leaf1
   + 1.7024911400000001e-04 t516_leaf2 + 0.00423171325 t516_leaf3
   + 0.00274650869 t516_leaf4 - 3.1581765500000001e-04 t517_leaf0
   - 0.00351536577 t517_leaf1 - 0.00408586068 t517_leaf2
   + 0.00178645935 t517_leaf3 + 1.9385012299999999e-04 t517_leaf4
   - 0.00115624163 t517_leaf5 - 0.00106289273 t518_leaf0
   + 0.00153160316 t518_leaf1 - 1.02088714e-04 t518_leaf2
   - 0.00156842894 t518_leaf3 - 0.00458377134 t518_leaf4
   + 7.4460613499999998e-04 t518_leaf5 + 0.00367111573 t519_leaf0
   - 3.83902e-05 t519_leaf1 - 8.9053367300000001e-04 t520_leaf0
   + 7.7219965199999999e-04 t520_leaf1 - 1.9234554199999999e-04 t520_leaf2
   + 2.1022613499999999e-04 t520_leaf3 - 3.49421403e-04 t520_leaf4
   + 0.00566709554 t520_leaf5 + 0.00620563 t521_leaf1
   - 6.86561e-05 t521_leaf2 + 6.8957672900000003e-04 t521_leaf3
   - 0.00141615234 t521_leaf4 - 4.8562241099999999e-04 t522_leaf0
   + 0.00607134681 t522_leaf1 - 3.30541e-05 t522_leaf2
   + 0.00167784735 t522_leaf3 - 4.91105253e-04 t522_leaf4
   - 0.00327840028 t522_leaf5 + 6.9879135100000005e-04 t523_leaf0
   - 0.00181738602 t523_leaf1 - 0.00172861025 t523_leaf2
   + 0.00421786495 t523_leaf3 + 1.55353e-05 t523_leaf4
   + 0.00317366421 t524_leaf0 - 6.5695674900000003e-04 t524_leaf1
   + 3.4179657800000003e-04 t524_leaf2 - 3.4004950399999997e-04 t524_leaf3
   + 3.93534e-05 t525_leaf0 + 0.00219327747 t525_leaf1
   - 0.00127791369 t526_leaf0 - 2.92764e-05 t526_leaf1
   + 2.2527154900000001e-04 t526_leaf2 - 0.00143341743 t526_leaf3
   + 0.00648676883 t526_leaf5 + 3.6766988299999999e-04 t527_leaf0
   - 0.000275858 t527_leaf1 + 0.000200251 t527_leaf2
   - 0.00213273405 t527_leaf3 + 1.7803646900000001e-04 t527_leaf4
   + 0.00472342456 t527_leaf5 + 0.000176469 t528_leaf0
   - 0.00442330306 t528_leaf2 - 0.00158713059 t529_leaf0
   - 4.99624e-05 t529_leaf1 + 5.4691475800000002e-04 t529_leaf2
   - 7.9812639199999997e-04 t529_leaf3 - 0.000896441 t529_leaf4
   + 4.60335e-06 t530_leaf0 - 0.000924574 t531_leaf0
   + 0.00253596297 t531_leaf1 - 7.68673e-05 t531_leaf2
   - 2.28338e-05 t532_leaf0 - 2.95257e-06 t533_leaf0
   - 6.2296568600000003e-04 t533_leaf1 + 0.00415514037 t533_leaf2
   - 1.6025e-05 t534_leaf0 - 3.0042944099999998e-04 t535_leaf0
   - 0.00392391114 t535_leaf1 + 0.00195275829 t535_leaf2
   + 4.78747534e-04 t535_leaf3 - 0.00248388969 t535_leaf4
   + 6.4586e-05 t536_leaf0 - 4.67767e-05 t537_leaf0
   + 0.00341072236 t537_leaf1 - 4.4578479700000003e-04 t538_leaf0
   + 7.2296452700000005e-04 t538_leaf1 + 4.8754861899999999e-04 t538_leaf2
   - 0.000606259 t538_leaf3 + 0.0031384984 t538_leaf4
   - 4.8732239499999999e-04 t539_leaf0 + 2.0426273100000001e-04 t539_leaf1
   + 1.68165803e-04 t540_leaf0 - 0.00337305199 t540_leaf1
   + 0.00388450897 t540_leaf2 + 1.3507531500000001e-04 t540_leaf3
   - 9.0746884200000003e-04 t540_leaf4 - 2.6875076599999998e-04 t541_leaf0
   + 0.0052269008 t541_leaf1 - 1.30752e-05 t541_leaf2
   - 6.90614e-05 t542_leaf0 - 0.00131369 t543_leaf0
   + 7.72315e-05 t543_leaf1 - 0.00173644105 t543_leaf2
   + 8.3307345599999995e-04 t543_leaf3 + 0.00134683051 t543_leaf4
   + 1.66499114e-04 t543_leaf5 + 0.004729365 t543_leaf6
   + 8.8539e-05 t544_leaf0 - 1.77745445e-04 t545_leaf0
   - 1.70226e-07 t545_leaf1 + 0.000948351 t545_leaf2
   - 4.03220591e-04 t545_leaf3 + 0.00361446152 t545_leaf4
   + 3.77067e-05 t546_leaf0 - 0.00304618152 t546_leaf1
   + 0.0012369107 t547_leaf0 - 1.06485764e-04 t547_leaf1
   - 4.41002e-05 t548_leaf0 - 0.00100922049 t548_leaf1
   - 0.00493960269 t548_leaf2 + 0.00284574181 t548_leaf3
   + 1.15695722e-04 t549_leaf0 + 0.00560277607 t549_leaf1
   - 0.00115313451 t549_leaf2 + 4.79214e-05 t549_leaf3
   - 0.00159952627 t549_leaf4 + 0.00193451648 t549_leaf5
   + 0.00164477609 t550_leaf0 - 0.000130768 t550_leaf1
   + 6.8304553899999998e-04 t550_leaf2 - 0.0028234187 t550_leaf3
   + 1.9745e-05 t551_leaf0 + 0.00205912162 t552_leaf0
   + 2.3069044999999999e-04 t552_leaf1 - 0.00159665616 t552_leaf2
   + 0.00279208645 t552_leaf3 - 5.9856072799999995e-04 t552_leaf4
   - 0.00314678508 t553_leaf0 + 0.00162353681 t553_leaf1
   + 4.17552e-05 t553_leaf2 - 4.1509248000000002e-04 t553_leaf3
   - 3.80871e-05 t554_leaf0 - 5.6915e-05 t555_leaf0
   + 0.00198085629 t556_leaf0 + 2.9309e-05 t556_leaf1
   + 6.2720844300000005e-04 t557_leaf0 - 0.00148981495 t557_leaf1
   - 1.7695018299999999e-04 t557_leaf2 + 0.00346032134 t557_leaf3
   - 1.8889833900000001e-04 t558_leaf0 + 9.1021781600000005e-04 t558_leaf1
   + 0.00167126919 t558_leaf2 - 0.00175395014 t558_leaf3
   + 0.000282303 t559_leaf0 + 0.00203023921 t559_leaf1
   + 2.9544832099999999e-04 t559_leaf2 - 0.000810566 t559_leaf3
   - 7.8644760699999997e-04 t559_leaf4 + 0.000142536 t560_leaf0
   + 0.00101987505 t560_leaf1 - 0.00159159 t560_leaf2
   - 5.5332051099999996e-04 t560_leaf3 + 7.72139e-05 t561_leaf0
   + 0.00542624481 t561_leaf1 - 6.34887e-05 t561_leaf2
   + 6.37661e-05 t562_leaf0 - 0.000311467 t562_leaf1
   - 0.00196128688 t562_leaf2 - 1.16428615e-04 t563_leaf0
   + 2.0515310500000001e-04 t564_leaf0 - 0.00482314359 t564_leaf1
   + 4.1603285399999999e-04 t564_leaf2 - 1.5195e-06 t565_leaf0
   - 0.00323019456 t566_leaf1 + 2.5759689700000001e-04 t566_leaf2
   - 0.00215753773 t566_leaf3 + 0.00398665061 t566_leaf4
   + 1.2567664100000001e-04 t566_leaf5 - 8.2781183300000005e-04 t567_leaf0
   + 0.00487138052 t567_leaf1 - 5.0017231800000004e-04 t567_leaf2
   - 2.1788768900000001e-04 t567_leaf3 - 0.00289756502 t567_leaf4
   + 3.2167922499999998e-04 t567_leaf5 - 1.19055323e-04 t568_leaf0
   - 0.00121519773 t568_leaf1 + 0.00260676257 t568_leaf2
   - 0.00157682376 t568_leaf3 + 4.23164e-05 t569_leaf0
   - 5.20453e-05 t570_leaf0 - 3.62942e-05 t571_leaf0
   - 8.7844137999999999e-04 t571_leaf1 + 0.00307952263 t571_leaf2
   - 0.00254928297 t571_leaf3 - 0.00253927941 t572_leaf1
   - 0.00199716701 t572_leaf2 - 6.67014e-05 t573_leaf0
   - 8.45809e-05 t574_leaf0 + 9.4545417200000005e-04 t574_leaf1
   - 0.00365613378 t574_leaf2 + 0.0031480852 t574_leaf3
   + 2.2494e-05 t575_leaf0 + 0.00162698282 t575_leaf1
   - 0.00219258294 t576_leaf0 + 0.000598109 t576_leaf1
   - 0.00207897346 t576_leaf2 + 2.2867335099999999e-04 t576_leaf3
   - 9.8742823999999996e-04 t576_leaf4 + 5.7297e-05 t576_leaf5
   + 1.27677253e-04 t577_leaf0 - 2.28782e-05 t578_leaf0
   - 0.00125513389 t578_leaf1 + 0.00101989054 t578_leaf2
   + 2.0294227599999999e-04 t579_leaf0 - 0.000583454 t579_leaf1
   - 1.2005634200000001e-04 t580_leaf0 - 0.00179917179 t580_leaf1
   + 0.00325733516 t580_leaf2 - 1.11995112e-04 t580_leaf3
   + 0.00312657072 t580_leaf4 + 5.2764930300000003e-04 t580_leaf5
   - 7.77127e-05 t581_leaf0 + 0.00298648793 t581_leaf1
   + 0.00151435041 t582_leaf0 - 4.53056e-05 t582_leaf1
   - 0.00259841839 t582_leaf2 - 6.19326e-07 t583_leaf0
   + 0.00159761496 t584_leaf0 + 0.0015324289 t584_leaf1
   + 2.25533e-05 t584_leaf2 - 0.00191564544 t584_leaf3
   - 1.21621408e-04 t585_leaf0 + 0.00129398273 t585_leaf1
   - 0.00209812913 t585_leaf2 + 0.00464958185 t585_leaf3
   + 0.000406847 t585_leaf4 + 0.0040965667 t586_leaf0
   - 4.71489e-05 t586_leaf1 - 0.00130873569 t586_leaf2
   + 2.6557187100000003e-04 t586_leaf3 + 0.00291670836 t586_leaf4
   - 6.99891e-05 t587_leaf0 + 0.00214311946 t587_leaf1
   - 0.00101804419 t587_leaf2 + 0.00436223717 t587_leaf3
   - 0.00124903326 t587_leaf4 + 0.00140586018 t587_leaf5
   + 0.00163603551 t588_leaf0 - 2.47692e-05 t588_leaf1
   - 3.3532275100000002e-04 t588_leaf2 - 0.00239840336 t588_leaf3
   + 0.00474706152 t588_leaf4 + 9.64133826e-04 t588_leaf5
   - 4.78732e-05 t588_leaf6 + 0.00343720964 t588_leaf7
   - 1.53807589e-04 t589_leaf0 - 2.9445e-05 t590_leaf0
   - 0.0017337529 t590_leaf1 - 3.39261e-05 t591_leaf0
   - 1.51668748e-04 t592_leaf0 - 1.52265e-05 t593_leaf0
   - 3.16768e-05 t594_leaf0 + 9.7461277599999996e-04 t595_leaf0
   + 0.00218158518 t595_leaf1 - 3.8273370600000001e-04 t595_leaf2
   - 6.51235e-05 t595_leaf3 + 0.00193780789 t595_leaf4
   + 6.25188e-05 t596_leaf0 - 3.2200469300000002e-04 t597_leaf0
   - 0.00320540881 t597_leaf1 + 1.25142513e-04 t597_leaf2
   + 0.00342430663 t597_leaf3 + 0.00239073788 t597_leaf4
   + 0.00147878507 t598_leaf0 - 1.2817417199999999e-04 t598_leaf1
   - 2.93934e-05 t599_leaf0 - 4.32474e-05 t600_leaf0
   + 0.00256722537 t600_leaf1 + 5.73431e-05 t600_leaf2
   - 0.000302245 t600_leaf3 - 1.35182854e-04 t601_leaf0
   - 0.000204421 t601_leaf1 + 0.00396685628 t601_leaf2
   + 8.3671067799999995e-04 t601_leaf3 + 1.68624e-05 t602_leaf0
   - 2.4055213699999999e-04 t603_leaf0 + 3.7220050599999998e-04 t603_leaf1
   + 8.9496e-06 t604_leaf0 + 9.57886e-06 t605_leaf0
   + 0.0010060796 t605_leaf1 - 0.00140665355 t605_leaf2
   + 0.00350159896 t605_leaf3 + 5.91163e-05 t606_leaf0
   - 2.13607e-05 t607_leaf0 - 0.00263399235 t607_leaf1
   - 1.9839710100000001e-04 t608_leaf0 + 0.00185567967 t608_leaf1
   - 1.41431e-05 t609_leaf0 - 1.66412e-05 t610_leaf0
   - 5.83672e-05 t611_leaf0 + 8.9526869100000001e-04 t611_leaf1
   + 1.20831e-05 t612_leaf0 + 0.000340316 t613_leaf0
   - 3.45610228e-04 t613_leaf1 + 2.24826e-05 t614_leaf0
   + 7.39887e-05 t615_leaf0 + 3.0573425500000001e-04 t616_leaf0
   - 8.70854e-05 t616_leaf1 - 0.00224715378 t616_leaf2
   - 0.000422832 t616_leaf3 + 0.00127459015 t617_leaf0
   - 0.00109007605 t617_leaf1 + 0.000116252 t617_leaf2
   + 5.42735e-08 t618_leaf0 + 0.00251163985 t618_leaf1
   - 1.27026e-05 t619_leaf0 - 0.0015222755 t619_leaf1
   + 9.6685037700000001e-04 t619_leaf2 + 0.00605672598 t619_leaf3
   + 2.4036846299999999e-04 t619_leaf4 + 7.3315e-05 t620_leaf0
   - 0.00213920418 t620_leaf1 + 1.3073127800000001e-04 t621_leaf0
   - 0.0029416082 t621_leaf2 - 1.7472318700000001e-04 t621_leaf3
   - 2.5530817200000001e-04 t622_leaf0 - 0.00156886294 t622_leaf1
   + 3.9943083500000001e-04 t622_leaf2 - 0.00186442037 t622_leaf3
   - 1.6841778400000001e-04 t623_leaf0 + 4.2586680499999999e-04 t623_leaf1
   + 0.00475928048 t623_leaf2 + 4.0905145599999998e-04 t623_leaf3
   + 0.00368787092 t623_leaf4 - 0.00186316215 t623_leaf5
   - 4.14198e-05 t624_leaf0 + 0.0020051985 t624_leaf1
   - 2.8728842200000001e-04 t625_leaf0 + 0.00315824826 t625_leaf1
   - 4.27315e-05 t625_leaf2 + 0.0035433867 t626_leaf1
   - 7.4106449000000004e-04 t626_leaf2 - 5.30406e-05 t626_leaf3
   - 5.01426e-05 t627_leaf0 - 0.00242259121 t627_leaf1
   - 4.87566e-05 t628_leaf0 - 5.10348e-05 t629_leaf0
   + 4.94083e-05 t630_leaf0 + 3.81261e-05 t631_leaf0
   + 0.00130104809 t632_leaf0 - 1.17781587e-04 t632_leaf1
   + 0.00565262558 t632_leaf3 - 8.86921e-06 t633_leaf0
   + 3.55426e-05 t634_leaf0 + 4.96217e-05 t635_leaf0
   + 5.66404e-06 t636_leaf0 + 4.27218e-05 t637_leaf0
   + 1.7494321199999999e-04 t638_leaf0 - 1.13147558e-04 t639_leaf0
   - 0.00185629714 t640_leaf0 + 1.33203e-05 t640_leaf1
   + 0.00613524625 t640_leaf2 - 3.83288e-05 t640_leaf3
   + 1.9781e-05 t641_leaf0 - 1.02989186e-04 t643_leaf0
   - 3.51006e-05 t644_leaf0 - 1.15778515e-04 t645_leaf0
   + 1.7900850800000001e-04 t646_leaf0 - 8.22664e-05 t646_leaf1
   - 0.00118274183 t646_leaf2 + 2.8086660299999998e-04 t647_leaf0
   - 3.39212333e-04 t647_leaf1 - 8.9756818499999998e-04 t647_leaf2
   + 1.95097e-05 t648_leaf0 - 3.03837e-05 t649_leaf0
   + 0.00182671263 t649_leaf1 - 2.0085256300000001e-04 t650_leaf0
   - 3.32028e-05 t651_leaf0 - 0.00230509043 t652_leaf1
   + 0.00236267853 t652_leaf2 + 1.68224e-05 t653_leaf0
   - 0.00166174909 t653_leaf1 + 3.70911e-06 t654_leaf0
   - 0.000121166 t655_leaf0 + 0.00397962844 t655_leaf1
   - 0.00103974401 t655_leaf2 + 3.03282e-05 t656_leaf0
   + 0.00268163951 t656_leaf1 - 0.00121936668 t656_leaf2
   + 7.6161e-05 t657_leaf0 - 0.00334497401 t657_leaf1
   + 3.09134e-05 t658_leaf0 + 0.00144463463 t659_leaf0
   - 3.6553133400000001e-04 t659_leaf1 - 0.00132835796 t659_leaf2
   - 0.00215905276 t659_leaf3 - 5.06413e-05 t659_leaf4
   + 3.5245e-05 t660_leaf0 + 7.8711414199999997e-04 t660_leaf1
   - 6.4149161300000005e-04 t660_leaf2 - 0.00286604767 t660_leaf3
   + 0.00172058155 t661_leaf0 - 0.00109011866 t661_leaf1
   - 7.8215e-05 t661_leaf2 + 0.00460131606 t661_leaf3
   - 9.5819658599999996e-04 t661_leaf4 + 2.33046e-06 t662_leaf0
   + 8.6160842299999997e-04 t663_leaf0 + 0.000143396 t663_leaf1
   - 0.00151998352 t663_leaf2 + 0.00543501507 t663_leaf3
   + 6.3389132300000004e-04 t663_leaf4 + 1.2378668200000001e-04 t663_leaf5
   - 0.00186408113 t663_leaf6 + 7.9294381399999995e-04 t664_leaf0
   - 0.00136463833 t664_leaf1 - 9.90472e-05 t664_leaf2
   + 3.67361819e-04 t665_leaf0 - 0.0016232183 t665_leaf1
   - 0.00033041 t665_leaf2 + 1.3996817899999999e-04 t666_leaf0
   + 4.0358543600000003e-04 t666_leaf1 - 0.00319703529 t666_leaf2
   - 4.4140065399999998e-04 t667_leaf0 + 0.00113000919 t667_leaf1
   + 0.00626973296 t667_leaf2 - 1.17209478e-04 t667_leaf3
   - 2.9240435100000002e-04 t668_leaf0 + 0.00383215095 t668_leaf1
   + 2.2067647699999999e-04 t668_leaf2 + 0.00543374242 t668_leaf3
   - 4.45153e-06 t669_leaf0 - 1.32121e-05 t670_leaf0
   + 2.78213e-05 t671_leaf0 - 0.00170903176 t671_leaf1
   + 6.89707e-05 t672_leaf0 + 0.00211846502 t673_leaf0
   - 9.3480525499999999e-04 t673_leaf1 - 0.00121623895 t673_leaf2
   + 8.4174156700000003e-04 t673_leaf3 - 0.000114582 t673_leaf4
   - 9.41094e-05 t674_leaf0 - 0.00270194747 t674_leaf1
   - 5.71125e-05 t675_leaf0 - 0.00303465081 t676_leaf0
   + 4.29067e-05 t676_leaf1 + 2.88531e-05 t677_leaf0
   - 3.25697e-05 t678_leaf0 + 0.000810209 t678_leaf1
   - 0.00225501112 t678_leaf2 - 0.00321772229 t678_leaf3
   - 0.000493416 t678_leaf4 - 4.57929e-05 t679_leaf0
   - 2.41172136e-04 t680_leaf0 - 0.00166071032 t680_leaf1
   + 2.9718561600000003e-04 t680_leaf2 - 0.00288230018 t680_leaf3
   - 0.00235753623 t680_leaf4 + 1.5446273e-04 t681_leaf0
   + 0.00329994829 t681_leaf1 - 1.39994e-05 t682_leaf0
   + 7.55557e-05 t683_leaf0 - 0.00354629103 t683_leaf1
   - 5.85615e-05 t683_leaf2 - 0.0017985818 t683_leaf3
   + 0.00401771395 t684_leaf0 - 0.000187505 t684_leaf1
   - 5.72283e-05 t684_leaf2 + 4.6655e-05 t685_leaf0
   - 3.94479e-05 t686_leaf0 + 0.00341447699 t687_leaf0
   + 1.02338498e-04 t687_leaf1 - 7.4143771800000001e-04 t687_leaf2
   + 1.59545e-05 t688_leaf0 + 7.5935e-05 t689_leaf0
   + 8.90341e-05 t690_leaf0 - 5.06326e-05 t691_leaf0
   + 1.12805414e-04 t692_leaf0 + 0.00327045657 t692_leaf1
   + 0.00245969556 t693_leaf0 - 1.3797616700000001e-04 t693_leaf1
   - 3.00627e-05 t694_leaf0 + 5.64036e-05 t695_leaf0
   - 1.3639789499999999e-04 t696_leaf0 - 3.1400666900000001e-04 t697_leaf0
   + 3.1261085000000002e-04 t697_leaf1 - 7.98502e-05 t698_leaf0
   + 0.00377371325 t698_leaf1 + 1.03841e-05 t699_leaf0
   + 1.91525e-05 t700_leaf0 + 5.08115e-05 t701_leaf0
   + 0.00135571905 t702_leaf0 - 5.9090985499999998e-04 t702_leaf1
   - 6.78367e-05 t702_leaf2 - 0.00289679901 t702_leaf3
   - 4.71973e-05 t703_leaf0 - 3.76481e-05 t704_leaf0
   - 1.35943e-05 t705_leaf0 + 0.00204694876 t705_leaf1
   - 0.00423646113 t705_leaf2 - 0.00364252692 t705_leaf3
   - 4.4542763499999999e-04 t705_leaf4 - 1.63703662e-04 t706_leaf0
   + 3.13693308e-04 t706_leaf1 + 0.00333039602 t706_leaf2
   + 2.41983e-05 t707_leaf0 - 1.68705e-05 t708_leaf0
   - 6.73504e-05 t709_leaf0 - 2.05909e-05 t709_leaf1
   + 0.00176071504 t709_leaf2 + 1.54333262e-04 t710_leaf0
   - 0.00332748494 t710_leaf1 + 4.0204724100000001e-04 t710_leaf2
   - 2.7995664300000002e-04 t710_leaf3 - 6.88676e-05 t711_leaf0
   - 0.00205248059 t712_leaf0 - 3.0062251600000001e-04 t712_leaf1
   + 0.00394420652 t712_leaf2 + 2.64586211e-04 t712_leaf3
   + 8.69266e-05 t713_leaf0 + 0.00198078365 t713_leaf1
   - 1.91327e-05 t714_leaf0 + 0.0019353847 t715_leaf1
   + 3.2512139299999999e-04 t715_leaf2 - 0.00133221119 t715_leaf3
   + 6.5844e-06 t716_leaf0 + 1.747153e-04 t717_leaf0
   + 1.63254e-05 t718_leaf0 - 1.31894e-05 t719_leaf0
   + 0.00303673791 t720_leaf0 + 3.98427e-05 t720_leaf1
   - 4.48759e-05 t721_leaf0 - 4.79159e-05 t722_leaf0
   + 1.5885065699999999e-04 t723_leaf0 - 4.81961e-05 t724_leaf0
   - 3.17488e-05 t725_leaf0 + 0.00244568195 t726_leaf0
   - 1.16781e-05 t726_leaf1 - 0.00163566624 t726_leaf2
   + 1.3390828100000001e-04 t727_leaf0 - 0.00513767684 t727_leaf1
   - 0.00164370693 t728_leaf1 - 1.93313521e-04 t729_leaf0
   + 8.3565863300000005e-04 t730_leaf0 - 8.2540471300000004e-04 t730_leaf1
   + 0.00161223742 t730_leaf2 + 1.4494759700000001e-04 t730_leaf3
   + 0.00345601351 t730_leaf4 - 5.82928e-05 t731_leaf0
   - 9.4503e-05 t732_leaf0 + 4.01409e-06 t733_leaf0
   + 7.71063e-05 t734_leaf0 + 7.2224e-05 t735_leaf0
   + 2.08127443e-04 t736_leaf0 - 6.67851e-05 t737_leaf0
   + 3.00794e-05 t738_leaf0 + 3.93748e-05 t739_leaf0
   - 0.00117612816 t739_leaf1 + 9.18681e-06 t740_leaf0
   - 5.65664e-05 t741_leaf0 + 0.00620048307 t741_leaf1
   - 4.4327616299999999e-04 t741_leaf2 - 0.00146190461 t742_leaf0
   + 0.00260374602 t742_leaf1 + 8.7459030300000005e-04 t742_leaf2
   - 3.09016e-05 t742_leaf3 + 0.00646517752 t742_leaf4
   + 0.000592359 t742_leaf6 - 8.91243e-06 t743_leaf0
   - 2.74526e-05 t744_leaf0 - 5.23045e-05 t745_leaf0
   + 1.00894e-05 t746_leaf0 - 2.31419e-05 t747_leaf0
   - 9.32162e-06 t748_leaf0 + 6.4837036199999996e-04 t749_leaf0
   - 0.00207467703 t749_leaf1 + 0.00306605967 t749_leaf2
   - 1.3687382999999999e-04 t749_leaf3 + 5.97513e-05 t750_leaf0
   - 3.40493e-05 t751_leaf0 - 1.4192868499999999e-04 t752_leaf0
   - 1.38671e-06 t753_leaf0 - 5.40337e-06 t754_leaf0
   + 1.08947e-05 t755_leaf0 + 1.72353e-05 t756_leaf0
   - 1.86329868e-04 t757_leaf0 + 0.00516315177 t757_leaf1
   + 9.00723157e-04 t757_leaf2 + 1.60954107e-04 t757_leaf3
   - 6.73389e-05 t758_leaf0 + 0.00015149 t759_leaf0
   - 0.0017015069 t759_leaf1 + 1.2294620700000001e-04 t760_leaf0
   - 6.67682e-05 t761_leaf0 - 0.00191103143 t761_leaf1
   - 5.7752965899999999e-04 t761_leaf2 + 0.00157087587 t761_leaf3
   + 8.58141e-05 t762_leaf0 - 3.07945e-05 t763_leaf0
   - 0.00197148137 t763_leaf1 - 1.2163622200000001e-04 t763_leaf2
   + 3.93870636e-04 t763_leaf3 + 0.00232828921 t763_leaf4
   - 0.00305568567 t764_leaf0 - 3.7642868099999998e-04 t764_leaf1
   + 0.00211131899 t764_leaf2 + 0.000226889 t764_leaf3
   - 4.9542e-05 t765_leaf0 + 1.9492457799999999e-04 t766_leaf0
   - 0.00173935061 t766_leaf1 - 2.03227e-05 t767_leaf0
   + 5.09111e-05 t768_leaf0 - 0.00189218577 t768_leaf1
   + 0.00236037141 t768_leaf2 - 3.1772e-05 t769_leaf0
   - 1.3172623600000001e-04 t770_leaf0 - 6.63884e-06 t771_leaf0
   + 1.79716852e-04 t772_leaf0 - 0.00314741069 t772_leaf1
   - 2.97736202e-04 t772_leaf2 + 1.5233812199999999e-04 t773_leaf0
   - 5.23566e-05 t774_leaf0 + 2.12635583e-04 t775_leaf0
   - 0.00143921957 t775_leaf1 + 2.67903e-05 t776_leaf0
   - 6.34962e-05 t777_leaf0 + 0.00239414 t777_leaf1
   - 2.05344e-05 t778_leaf0 - 0.00526554463 t778_leaf1
   + 1.59439e-05 t778_leaf2 + 0.00317724166 t778_leaf3
   - 6.49324e-05 t779_leaf0 - 3.595e-05 t780_leaf0
   + 0.00363019365 t781_leaf0 + 6.35494e-05 t781_leaf1
   - 0.00111368322 t781_leaf2 - 5.81681e-05 t782_leaf0
   - 1.92205e-05 t783_leaf0 - 0.00146728428 t784_leaf0
   + 6.3343520700000005e-04 t784_leaf1 + 1.1299551999999999e-04 t784_leaf2
   + 0.00331801292 t784_leaf3 - 0.00283829262 t784_leaf4
   + 0.00183699862 t785_leaf0 - 9.63388e-05 t785_leaf1
   - 8.14462e-05 t786_leaf0 + 5.78842e-05 t787_leaf0
   - 0.00339002116 t787_leaf1 + 6.38706e-06 t788_leaf0
   + 1.2692887699999999e-04 t789_leaf0 + 5.22696e-06 t790_leaf0
   + 1.4929116899999999e-04 t791_leaf0 - 1.48858e-05 t792_leaf0
   + 2.51086e-05 t793_leaf0 - 0.00331524736 t793_leaf1
   + 0.00442092028 t793_leaf2 + 3.74112e-05 t793_leaf3
   - 1.12779264e-04 t794_leaf0 + 0.00206198427 t795_leaf1
   - 3.7376178099999998e-04 t795_leaf2 + 1.80031e-05 t795_leaf3
   - 0.00176201877 t795_leaf4 - 6.21064e-05 t796_leaf0
   + 2.81242e-05 t797_leaf0 - 6.74817e-05 t798_leaf0
   + 1.2516773100000001e-04 t799_leaf0 - 1.62891447e-04 t800_leaf0
   - 3.00696e-05 t801_leaf0 + 7.3264e-05 t802_leaf0
   - 4.45435e-05 t803_leaf0 + 6.04631e-05 t804_leaf0
   - 0.00174679 t805_leaf0 + 1.79584e-05 t805_leaf1
   + 8.3087221699999997e-04 t806_leaf0 - 3.84749e-05 t806_leaf1
   + 1.04881474e-04 t807_leaf0 - 3.51333e-05 t808_leaf0
   + 9.56046e-06 t809_leaf0 - 5.1986e-05 t810_leaf0
   - 1.21441e-05 t811_leaf0 + 9.15348e-05 t812_leaf0
   + 0.000117883 t813_leaf0 - 2.1007812799999999e-04 t814_leaf0
   + 5.8581749900000004e-04 t814_leaf1 + 0.00186009903 t814_leaf2
   + 0.00355380168 t814_leaf3 - 4.4575668299999999e-04 t814_leaf4
   - 3.98216e-05 t815_leaf0 - 7.58876e-05 t816_leaf0
   - 7.58534e-05 t817_leaf0 + 0.00592026114 t817_leaf1
   + 2.4496673699999998e-04 t817_leaf2 + 2.5795e-05 t818_leaf0
   + 4.85721e-05 t819_leaf0 - 9.35011e-05 t820_leaf0
   + 5.67381e-06 t821_leaf0 + 5.19976e-05 t822_leaf0
   - 5.34172e-05 t823_leaf0 - 2.23862e-05 t824_leaf0
   - 3.01992e-05 t825_leaf0 - 0.00172661943 t825_leaf1
   + 8.59429e-05 t826_leaf0 - 3.9412928200000002e-04 t827_leaf0
   + 0.0015367259 t827_leaf1 - 0.0019078427 t827_leaf2
   + 1.7429774700000001e-04 t827_leaf3 + 4.18846e-05 t828_leaf0
   + 0.00125556637 t829_leaf0 - 9.90314e-05 t829_leaf1
   - 1.8633784200000001e-04 t829_leaf2 + 0.00217220676 t829_leaf3
   - 0.00154137623 t829_leaf4 + 3.21379e-05 t830_leaf0
   + 5.04298e-05 t831_leaf0 - 0.00215259194 t831_leaf1
   - 0.00369863166 t832_leaf0 - 4.28939e-05 t832_leaf1
   + 6.52045e-06 t833_leaf0 - 2.03854666e-04 t834_leaf0
   - 1.06950567e-04 t835_leaf0 - 0.0018394246 t835_leaf1
   + 0.00156592927 t835_leaf2 + 1.3048e-05 t836_leaf0
   + 2.09775e-05 t837_leaf0 + 3.89339e-05 t838_leaf0
   - 1.01981714e-04 t839_leaf0 + 0.00322737521 t839_leaf1
   - 6.40916e-08 t840_leaf0 - 6.03732e-05 t841_leaf0
   + 0.00112336886 t841_leaf1 - 5.491e-05 t842_leaf0
   + 0.00127638178 t842_leaf1 - 6.37554e-06 t843_leaf0
   + 0.00101418013 t844_leaf0 - 7.68308e-05 t844_leaf1
   - 2.02956173e-04 t845_leaf0 - 6.1968714000000004e-04 t845_leaf1
   + 9.3402277000000002e-04 t845_leaf2 - 1.67296e-05 t846_leaf0
   - 2.71912e-05 t847_leaf0 + 0.00175209437 t848_leaf0
   - 8.8514969700000004e-04 t848_leaf1 - 5.13455e-05 t848_leaf2
   + 9.53182e-05 t849_leaf0 - 0.00115381903 t849_leaf1
   - 8.83821223e-04 t849_leaf2 - 7.96345e-05 t850_leaf0
   - 7.88005e-06 t851_leaf0 + 2.35185e-05 t852_leaf0
   - 9.56815e-06 t853_leaf0 + 1.01649712e-04 t854_leaf0
   - 0.00261323177 t854_leaf1 - 7.56838e-05 t855_leaf0
   - 7.89397e-08 t856_leaf0 - 0.00186690688 t856_leaf1
   + 7.54399e-05 t857_leaf0 - 3.47688e-05 t858_leaf0
   - 0.0022359693 t858_leaf1 + 0.000603889 t858_leaf2
   + 0.00388221676 t858_leaf3 - 4.57757e-05 t859_leaf0
   + 0.00481479382 t860_leaf0 + 1.02130725e-04 t860_leaf2
   + 6.06652e-05 t861_leaf0 - 1.0676956299999999e-04 t862_leaf0
   + 1.3420476200000001e-04 t863_leaf0 + 4.7329e-05 t864_leaf0
   - 3.65058e-05 t865_leaf0 + 0.00540651614 t865_leaf2
   + 2.88547e-05 t866_leaf0 + 4.98076e-05 t867_leaf0
   + 4.69418e-05 t868_leaf0 - 3.77602e-05 t869_leaf0
   + 1.22322e-05 t870_leaf0 - 2.96809e-05 t871_leaf0
   + 1.49157e-05 t872_leaf0 + 7.27154e-05 t873_leaf0
   + 5.29768e-05 t874_leaf0 - 0.0014122345 t875_leaf0
   + 1.28257685e-04 t875_leaf1 + 4.5178542499999999e-04 t876_leaf0
   - 0.00186321908 t876_leaf1 + 1.74142391e-04 t876_leaf2
   - 0.00159601681 t876_leaf3 - 3.51588e-06 t877_leaf0
   - 0.00146434631 t877_leaf1 - 8.37409e-05 t878_leaf0
   + 3.44417e-05 t879_leaf0 + 2.45524e-05 t880_leaf0
   - 8.6292089000000001e-04 t880_leaf1 + 0.00399752706 t880_leaf2
   - 6.4143119400000002e-04 t880_leaf3 + 0.003140938 t881_leaf0
   - 0.000293376 t881_leaf2 - 4.1023155799999999e-04 t881_leaf3
   + 2.02451381e-04 t881_leaf4 + 1.4837e-05 t882_leaf0
   - 7.7574211200000004e-04 t883_leaf0 + 3.23797e-05 t883_leaf1
   + 2.24807e-06 t884_leaf0 + 3.82079e-05 t885_leaf0
   - 0.00118095614 t886_leaf0 + 1.69205e-05 t886_leaf1
   - 0.00339460955 t886_leaf2 + 0.00153407396 t886_leaf3
   - 8.8644446799999999e-04 t886_leaf4 - 0.000731332 t886_leaf5
   - 0.000104384 t887_leaf0 + 2.4452211800000002e-04 t888_leaf0
   - 4.4280648600000003e-04 t888_leaf1 + 0.00047766 t889_leaf0
   + 0.00487284176 t889_leaf1 - 2.5142284000000001e-04 t889_leaf2
   + 3.6270721499999999e-04 t889_leaf3 - 0.00309592066 t889_leaf4
   - 2.05314e-05 t890_leaf0 + 1.84283e-05 t891_leaf0
   + 4.44824e-05 t892_leaf0 + 3.05993e-05 t893_leaf0
   + 1.8142779300000001e-04 t894_leaf0 - 0.00136716268 t894_leaf1
   + 1.4151226800000001e-04 t895_leaf0 + 9.03398e-05 t896_leaf0
   - 9.7510457299999995e-04 t896_leaf1 + 9.31598e-06 t897_leaf0
   - 7.41732e-05 t898_leaf0 + 2.9616e-05 t899_leaf0 + 2.8436e-05 t900_leaf0
   + 2.14811e-05 t901_leaf0 - 2.54415e-05 t902_leaf0
   - 0.00147752825 t903_leaf0 + 0.00257980474 t903_leaf2
   + 0.00294329342 t904_leaf0 - 9.32581e-06 t904_leaf1
   - 4.59413e-05 t905_leaf0 - 0.000347855 t906_leaf0
   + 5.3914968200000004e-04 t906_leaf1 + 9.1764662600000004e-04 t906_leaf2
   + 3.6586e-05 t907_leaf0 - 1.6305962299999999e-04 t908_leaf0
   + 0.00130810367 t909_leaf0 - 1.22262e-06 t909_leaf1
   + 2.62518e-05 t910_leaf0 - 4.45145e-05 t911_leaf0 - 7.128e-05 t912_leaf0
   + 9.69016e-05 t913_leaf0 = 0
 YLOG_with_offset: - y_log_raw + y_log = 12.437748
 COST_eq_linexpr: - 17352 x_roof_matl_is_ClyTile
   - 8011.95 x_roof_matl_is_Membran - 11739 x_roof_matl_is_Metal
   - 7600 x_roof_matl_is_Roll - 8550 x_roof_matl_is_Tar&Grv
   - 22500 x_roof_matl_is_WdShake - 19500 x_roof_matl_is_WdShngl
   - 1712.5 x_elect_is_FuseA - 887.5 x_elect_is_FuseF
   - 62.5 x_elect_is_FuseP - 287.5 x_elect_is_Mix
   - 24038 x_garage_finish_is_Fin - 17500 x_garage_finish_is_Unf
   - 28000 x_AddFull - 11200 x_AddHalf - 16800 x_AddKitch - 25480 x_AddBed
   - 51659 x_garage_qual_is_Ex - 51659 x_garage_cond_is_Ex
   - 37849 x_garage_qual_is_Gd - 37849 x_garage_cond_is_Gd
   - 14113 x_garage_qual_is_Fa - 14113 x_garage_cond_is_Fa
   - 4188 x_garage_qual_is_Po - 4188 x_garage_cond_is_Po
   - 3354 x_paved_drive_is_P - 1800 x_paved_drive_is_N
   - 7260 x_fence_is_GdPrv - 5660 x_fence_is_MnPrv - 3232 x_fence_is_GdWo
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
   - 53500 x_b1_is_ALQ - 32000 x_b1_is_BLQ - 23500 x_b1_is_Rec
   - 15000 x_b1_is_LwQ - 75000 x_b2_is_GLQ - 53500 x_b2_is_ALQ
   - 32000 x_b2_is_BLQ - 23500 x_b2_is_Rec - 15000 x_b2_is_LwQ
   - 4423.2 x_z10_GarageArea - 11449.92 x_z20_GarageArea
   - 21078.72 x_z30_GarageArea - 1529.69 x_z10_OpenPorchSF
   - 3959.764 x_z20_OpenPorchSF - 7289.723999999999 x_z30_OpenPorchSF
   - 180000 kit_is_Ex - 83416.94 exterqual_is_Ex - 3949.31 extercond_is_Gd
   - 87366.25 extercond_is_Ex - 1.21 mv_area_mvt_BrkCmn
   - 15 mv_area_mvt_BrkFace - 22.5 mv_area_mvt_CBlock
   - 27.5 mv_area_mvt_Stone - 10850 roof_change_mat - 10304 x_Add1stFlr
   - 1773 heat_is_Floor - 8500 heat_is_GasW - 6300 heat_is_Grav
   - 5000 heat_is_OthW - 3700 heat_is_Wall - 3750 heat_qc_is_Po
   - 5125 heat_qc_is_Fa - 6500 heat_qc_is_TA - 8250 heat_qc_is_Gd
   - 1500 fireplace_is_Po - 2000 fireplace_is_Fa - 2500 fireplace_is_TA
   - 3525 fireplace_is_Gd - 4550 fireplace_is_Ex - 19200 bsmt_finish_all
   - 10750 bsmtcond_is_Gd - 21500 bsmtcond_is_Ex + cost_model = 0
 BUDGET: cost_model <= 18000
 NOCHG_cost_link: cost_model - 18000 z_change <= 0
 NOCHG_cost_lb: cost_model - z_change >= 0
 NOCHG_pos_Utilities: x_Utilities <= 3
 NOCHG_neg_Utilities: - x_Utilities - 3 z_change <= -3
 NOCHG_pos_Mas_Vnr_Area: x_Mas_Vnr_Area - 2000 z_change <= 0
 NOCHG_neg_Mas_Vnr_Area: - x_Mas_Vnr_Area <= 0
 NOCHG_pos_Exter_Qual: x_Exter_Qual - z_change <= 3
 NOCHG_neg_Exter_Qual: - x_Exter_Qual - 4 z_change <= -3
 NOCHG_pos_Exter_Cond: x_Exter_Cond - 2 z_change <= 2
 NOCHG_neg_Exter_Cond: - x_Exter_Cond - 3 z_change <= -2
 NOCHG_pos_Bsmt_Cond: x_Bsmt_Cond - 2 z_change <= 2
 NOCHG_neg_Bsmt_Cond: - x_Bsmt_Cond - 3 z_change <= -2
 NOCHG_pos_BsmtFin_SF_1: x_BsmtFin_SF_1 - 1e+06 z_change <= 0
 NOCHG_neg_BsmtFin_SF_1: - x_BsmtFin_SF_1 <= 0
 NOCHG_pos_BsmtFin_SF_2: x_BsmtFin_SF_2 - 100000 z_change <= 0
 NOCHG_neg_BsmtFin_SF_2: - x_BsmtFin_SF_2 <= 0
 NOCHG_pos_Bsmt_Unf_SF: x_Bsmt_Unf_SF - 99400 z_change <= 600
 NOCHG_neg_Bsmt_Unf_SF: - x_Bsmt_Unf_SF - 600 z_change <= -600
 NOCHG_pos_Heating_QC: x_Heating_QC <= 4
 NOCHG_neg_Heating_QC: - x_Heating_QC - 5 z_change <= -4
 NOCHG_pos_1st_Flr_SF: x_1st_Flr_SF - 4480 z_change <= 520
 NOCHG_neg_1st_Flr_SF: - x_1st_Flr_SF - 520 z_change <= -520
 NOCHG_pos_Gr_Liv_Area: x_Gr_Liv_Area - 8777 z_change <= 1223
 NOCHG_neg_Gr_Liv_Area: - x_Gr_Liv_Area - 1223 z_change <= -1223
 NOCHG_pos_Full_Bath: x_Full_Bath - 2 z_change <= 2
 NOCHG_neg_Full_Bath: - x_Full_Bath - 2 z_change <= -2
 NOCHG_pos_Half_Bath: x_Half_Bath - 4 z_change <= 1
 NOCHG_neg_Half_Bath: - x_Half_Bath - z_change <= -1
 NOCHG_pos_Bedroom_AbvGr: x_Bedroom_AbvGr - 4 z_change <= 2
 NOCHG_neg_Bedroom_AbvGr: - x_Bedroom_AbvGr - 2 z_change <= -2
 NOCHG_pos_Kitchen_AbvGr: x_Kitchen_AbvGr - 2 z_change <= 1
 NOCHG_neg_Kitchen_AbvGr: - x_Kitchen_AbvGr - z_change <= -1
 NOCHG_pos_Kitchen_Qual: x_Kitchen_Qual - z_change <= 3
 NOCHG_neg_Kitchen_Qual: - x_Kitchen_Qual - 4 z_change <= -3
 NOCHG_pos_TotRms_AbvGrd: x_TotRms_AbvGrd - 11 z_change <= 4
 NOCHG_neg_TotRms_AbvGrd: - x_TotRms_AbvGrd - 4 z_change <= -4
 NOCHG_pos_Fireplace_Qu: x_Fireplace_Qu - 5 z_change <= -1
 NOCHG_neg_Fireplace_Qu: - x_Fireplace_Qu <= 1
 NOCHG_pos_Garage_Area: x_Garage_Area - 99520 z_change <= 480
 NOCHG_neg_Garage_Area: - x_Garage_Area - 480 z_change <= -480
 NOCHG_pos_Garage_Qual: x_Garage_Qual - 2 z_change <= 2
 NOCHG_neg_Garage_Qual: - x_Garage_Qual - 3 z_change <= -2
 NOCHG_pos_Garage_Cond: x_Garage_Cond - 2 z_change <= 2
 NOCHG_neg_Garage_Cond: - x_Garage_Cond - 3 z_change <= -2
 NOCHG_pos_Wood_Deck_SF: x_Wood_Deck_SF - 100000 z_change <= 0
 NOCHG_neg_Wood_Deck_SF: - x_Wood_Deck_SF <= 0
 NOCHG_pos_Open_Porch_SF: x_Open_Porch_SF - 99834 z_change <= 166
 NOCHG_neg_Open_Porch_SF: - x_Open_Porch_SF - 166 z_change <= -166
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
 NOCHG_pos_Exterior_1st_MetalSd: x_ex1_is_MetalSd <= 1
 NOCHG_neg_Exterior_1st_MetalSd: - x_ex1_is_MetalSd - z_change <= -1
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
 NOCHG_pos_Exterior_1st_Wd_Sdng: x_ex1_is_Wd_Sdng - z_change <= 0
 NOCHG_neg_Exterior_1st_Wd_Sdng: - x_ex1_is_Wd_Sdng <= 0
 NOCHG_pos_Exterior_2nd_AsbShng: x_ex2_is_AsbShng - z_change <= 0
 NOCHG_neg_Exterior_2nd_AsbShng: - x_ex2_is_AsbShng <= 0
 NOCHG_pos_Exterior_2nd_AsphShn: x_ex2_is_AsphShn - z_change <= 0
 NOCHG_neg_Exterior_2nd_AsphShn: - x_ex2_is_AsphShn <= 0
 NOCHG_pos_Exterior_2nd_BrkFace: x_ex2_is_BrkFace - z_change <= 0
 NOCHG_neg_Exterior_2nd_BrkFace: - x_ex2_is_BrkFace <= 0
 NOCHG_pos_Exterior_2nd_CBlock: x_ex2_is_CBlock - z_change <= 0
 NOCHG_neg_Exterior_2nd_CBlock: - x_ex2_is_CBlock <= 0
 NOCHG_pos_Exterior_2nd_HdBoard: x_ex2_is_HdBoard - z_change <= 0
 NOCHG_neg_Exterior_2nd_HdBoard: - x_ex2_is_HdBoard <= 0
 NOCHG_pos_Exterior_2nd_ImStucc: x_ex2_is_ImStucc - z_change <= 0
 NOCHG_neg_Exterior_2nd_ImStucc: - x_ex2_is_ImStucc <= 0
 NOCHG_pos_Exterior_2nd_MetalSd: x_ex2_is_MetalSd <= 1
 NOCHG_neg_Exterior_2nd_MetalSd: - x_ex2_is_MetalSd - z_change <= -1
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
 NOCHG_pos_Exterior_2nd_Wd_Sdng: x_ex2_is_Wd_Sdng - z_change <= 0
 NOCHG_neg_Exterior_2nd_Wd_Sdng: - x_ex2_is_Wd_Sdng <= 0
 NOCHG_pos_Mas_Vnr_Type_BrkCmn: x_mvt_is_BrkCmn - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_BrkCmn: - x_mvt_is_BrkCmn <= 0
 NOCHG_pos_Mas_Vnr_Type_BrkFace: x_mvt_is_BrkFace - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_BrkFace: - x_mvt_is_BrkFace <= 0
 NOCHG_pos_Mas_Vnr_Type_CBlock: x_mvt_is_CBlock - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_CBlock: - x_mvt_is_CBlock <= 0
 NOCHG_pos_Mas_Vnr_Type_No_aplica: x_mvt_is_No_aplica <= 1
 NOCHG_neg_Mas_Vnr_Type_No_aplica: - x_mvt_is_No_aplica - z_change <= -1
 NOCHG_pos_Mas_Vnr_Type_NoAplica: x_mvt_is_No_aplica - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_NoAplica: - x_mvt_is_No_aplica <= 0
 NOCHG_pos_Mas_Vnr_Type_Stone: x_mvt_is_Stone - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_Stone: - x_mvt_is_Stone <= 0
 NOCHG_pos_BsmtFin_Type_1_ALQ: x_b1_is_ALQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_ALQ: - x_b1_is_ALQ <= 0
 NOCHG_pos_BsmtFin_Type_1_BLQ: x_b1_is_BLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_BLQ: - x_b1_is_BLQ <= 0
 NOCHG_pos_BsmtFin_Type_1_GLQ: x_b1_is_GLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_GLQ: - x_b1_is_GLQ <= 0
 NOCHG_pos_BsmtFin_Type_1_LwQ: x_b1_is_LwQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_LwQ: - x_b1_is_LwQ <= 0
 NOCHG_pos_BsmtFin_Type_1_Rec: x_b1_is_Rec - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_Rec: - x_b1_is_Rec <= 0
 NOCHG_pos_BsmtFin_Type_1_Unf: x_b1_is_Unf <= 1
 NOCHG_neg_BsmtFin_Type_1_Unf: - x_b1_is_Unf - z_change <= -1
 NOCHG_pos_BsmtFin_Type_2_ALQ: x_b2_is_ALQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_ALQ: - x_b2_is_ALQ <= 0
 NOCHG_pos_BsmtFin_Type_2_BLQ: x_b2_is_BLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_BLQ: - x_b2_is_BLQ <= 0
 NOCHG_pos_BsmtFin_Type_2_GLQ: x_b2_is_GLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_GLQ: - x_b2_is_GLQ <= 0
 NOCHG_pos_BsmtFin_Type_2_LwQ: x_b2_is_LwQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_LwQ: - x_b2_is_LwQ <= 0
 NOCHG_pos_BsmtFin_Type_2_Rec: x_b2_is_Rec - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_Rec: - x_b2_is_Rec <= 0
 NOCHG_pos_BsmtFin_Type_2_Unf: x_b2_is_Unf <= 1
 NOCHG_neg_BsmtFin_Type_2_Unf: - x_b2_is_Unf - z_change <= -1
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
 NOCHG_price_pos: y_price - 2e+06 z_change <= 237393.28125
 NOCHG_price_neg: - y_price - 2e+06 z_change <= -237393.28125
 MIN_PRICE_BASE: y_price >= 237393.281249
 NO_NEGATIVE_ROI: y_price - cost_model >= 237393.281249
Bounds
 x_Bedroom_AbvGr <= 6
 x_Full_Bath <= 4
 x_BsmtFin_SF_1 <= 1e+06
 x_BsmtFin_SF_2 <= 100000
 x_Bsmt_Unf_SF <= 100000
 x_Gr_Liv_Area <= 10000
 x_1st_Flr_SF <= 5000
 x_TotRms_AbvGrd <= 15
 x_Mas_Vnr_Area = 0
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
 exterqual_is_TA = 0
 extercond_is_Po = 0
 extercond_is_Fa = 0
 mv_area_mvt_BrkCmn <= 2000
 mv_area_mvt_BrkFace <= 2000
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
 heat_is_GasW = 0
 heat_is_Grav = 0
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
 B1_upg_flag = 1
 B2_upg_flag = 1
 util_ELO = 0
 util_NoSeWa = 0
 util_NoSewr = 0
 R8_other_rooms <= 15
 const_Lot_Frontage = 24
 const_Lot_Area = 2544
 const_Overall_Cond = 5
 const_Year_Built = 2005
 const_Year_Remod_Add = 2005
 const_Bsmt_Qual = 3
 const_Total_Bsmt_SF = 600
 const_2nd_Flr_SF = 623
 const_Low_Qual_Fin_SF = 80
 const_Bsmt_Full_Bath = 0
 const_Bsmt_Half_Bath = 0
 const_Fireplaces = 0
 const_Garage_Yr_Blt = 2005
 const_Garage_Cars = 2
 const_Misc_Val = 0
 const_Yr_Sold = 2006
 const_MS_SubClass_20 = 0
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
 const_MS_SubClass_160 = 1
 const_MS_SubClass_180 = 0
 const_MS_SubClass_190 = 0
 const_MS_Zoning_A_(agr) = 0
 const_MS_Zoning_C_(all) = 0
 const_MS_Zoning_FV = 1
 const_MS_Zoning_I_(all) = 0
 const_MS_Zoning_RH = 0
 const_MS_Zoning_RL = 0
 const_MS_Zoning_RM = 0
 const_Street_Grvl = 0
 const_Street_Pave = 1
 const_Alley_Grvl = 0
 const_Alley_No_aplica = 0
 const_Alley_Pave = 1
 const_Lot_Shape_IR1 = 0
 const_Lot_Shape_IR2 = 0
 const_Lot_Shape_IR3 = 0
 const_Lot_Shape_Reg = 1
 const_Land_Contour_Bnk = 0
 const_Land_Contour_HLS = 0
 const_Land_Contour_Low = 0
 const_Land_Contour_Lvl = 1
 const_Lot_Config_Corner = 0
 const_Lot_Config_CulDSac = 0
 const_Lot_Config_FR2 = 0
 const_Lot_Config_FR3 = 0
 const_Lot_Config_Inside = 1
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
 const_Neighborhood_No_aplicames = 0
 const_Neighborhood_NoRidge = 0
 const_Neighborhood_NridgHt = 0
 const_Neighborhood_OldTown = 0
 const_Neighborhood_SWISU = 0
 const_Neighborhood_Sawyer = 0
 const_Neighborhood_SawyerW = 0
 const_Neighborhood_Somerst = 1
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
 const_Bldg_Type_1Fam = 0
 const_Bldg_Type_2fmCon = 0
 const_Bldg_Type_Duplex = 0
 const_Bldg_Type_Twnhs = 1
 const_Bldg_Type_TwnhsE = 0
 const_House_Style_1.5Fin = 0
 const_House_Style_1.5Unf = 0
 const_House_Style_1Story = 0
 const_House_Style_2.5Fin = 0
 const_House_Style_2.5Unf = 0
 const_House_Style_2Story = 1
 const_House_Style_SFoyer = 0
 const_House_Style_SLvl = 0
 const_Roof_Style_Flat = 0
 const_Roof_Style_Gable = 1
 const_Roof_Style_Gambrel = 0
 const_Roof_Style_Hip = 0
 const_Roof_Style_Mansard = 0
 const_Roof_Style_Shed = 0
 const_Exterior_1st_WdShing = 0
 const_Exterior_2nd_Brk_Cmn = 0
 const_Exterior_2nd_CmentBd = 0
 const_Exterior_2nd_Wd_Shng = 0
 const_Foundation_BrkTil = 0
 const_Foundation_CBlock = 0
 const_Foundation_PConc = 1
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
 const_Garage_Type_Attchd = 0
 const_Garage_Type_Basment = 0
 const_Garage_Type_BuiltIn = 0
 const_Garage_Type_CarPort = 0
 const_Garage_Type_Detchd = 1
 const_Garage_Type_No_aplica = 0
 const_Garage_Finish_Fin = 0
 const_Garage_Finish_No_aplica = 0
 const_Garage_Finish_RFn = 1
 const_Garage_Finish_Unf = 0
 const_Misc_Feature_Elev = 0
 const_Misc_Feature_Gar2 = 0
 const_Misc_Feature_No_aplica = 1
 const_Misc_Feature_Othr = 0
 const_Misc_Feature_Shed = 0
 const_Misc_Feature_TenC = 0
 const_Mo_Sold_1 = 0
 const_Mo_Sold_2 = 0
 const_Mo_Sold_3 = 0
 const_Mo_Sold_4 = 0
 const_Mo_Sold_5 = 0
 const_Mo_Sold_6 = 0
 const_Mo_Sold_7 = 1
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
 heat_qc_is_Fa heat_qc_is_TA heat_qc_is_Gd heat_qc_is_Ex fireplace_is_Po
 fireplace_is_Fa fireplace_is_TA fireplace_is_Gd fireplace_is_Ex
 fireplace_is_No_aplica bsmt_finish_all bsmtcond_is_Po bsmtcond_is_Fa
 bsmtcond_is_TA bsmtcond_is_Gd bsmtcond_is_Ex B1_upg_flag B2_upg_flag
 util_ELO util_NoSeWa util_NoSewr util_AllPub t0_leaf0 t0_leaf1 t0_leaf2
 t0_leaf3 t0_leaf4 t0_leaf5 t0_leaf6 t0_leaf7 t1_leaf0 t1_leaf1 t1_leaf2
 t1_leaf3 t1_leaf4 t1_leaf5 t1_leaf6 t1_leaf7 t2_leaf0 t2_leaf1 t2_leaf2
 t2_leaf3 t2_leaf4 t2_leaf5 t2_leaf6 t2_leaf7 t3_leaf0 t3_leaf1 t3_leaf2
 t3_leaf3 t3_leaf4 t3_leaf5 t3_leaf6 t3_leaf7 t4_leaf0 t4_leaf1 t4_leaf2
 t4_leaf3 t4_leaf4 t4_leaf5 t4_leaf6 t4_leaf7 t5_leaf0 t5_leaf1 t5_leaf2
 t5_leaf3 t5_leaf4 t5_leaf5 t5_leaf6 t5_leaf7 t6_leaf0 t6_leaf1 t6_leaf2
 t6_leaf3 t6_leaf4 t6_leaf5 t6_leaf6 t6_leaf7 t7_leaf0 t7_leaf1 t7_leaf2
 t7_leaf3 t7_leaf4 t7_leaf5 t7_leaf6 t7_leaf7 t8_leaf0 t8_leaf1 t8_leaf2
 t8_leaf3 t8_leaf4 t8_leaf5 t8_leaf6 t8_leaf7 t9_leaf0 t9_leaf1 t9_leaf2
 t9_leaf3 t9_leaf4 t9_leaf5 t9_leaf6 t9_leaf7 t10_leaf0 t10_leaf1 t10_leaf2
 t10_leaf3 t10_leaf4 t10_leaf5 t10_leaf6 t10_leaf7 t11_leaf0 t11_leaf1
 t11_leaf2 t11_leaf3 t11_leaf4 t11_leaf5 t11_leaf6 t11_leaf7 t12_leaf0
 t12_leaf1 t12_leaf2 t12_leaf3 t12_leaf4 t12_leaf5 t12_leaf6 t12_leaf7
 t13_leaf0 t13_leaf1 t13_leaf2 t13_leaf3 t13_leaf4 t13_leaf5 t13_leaf6
 t13_leaf7 t14_leaf0 t14_leaf1 t14_leaf2 t14_leaf3 t14_leaf4 t14_leaf5
 t14_leaf6 t14_leaf7 t15_leaf0 t15_leaf1 t15_leaf2 t15_leaf3 t15_leaf4
 t15_leaf5 t15_leaf6 t15_leaf7 t16_leaf0 t16_leaf1 t16_leaf2 t16_leaf3
 t16_leaf4 t16_leaf5 t16_leaf6 t16_leaf7 t17_leaf0 t17_leaf1 t17_leaf2
 t17_leaf3 t17_leaf4 t17_leaf5 t17_leaf6 t18_leaf0 t18_leaf1 t18_leaf2
 t18_leaf3 t18_leaf4 t18_leaf5 t18_leaf6 t18_leaf7 t19_leaf0 t19_leaf1
 t19_leaf2 t19_leaf3 t19_leaf4 t19_leaf5 t19_leaf6 t19_leaf7 t20_leaf0
 t20_leaf1 t20_leaf2 t20_leaf3 t20_leaf4 t20_leaf5 t20_leaf6 t20_leaf7
 t21_leaf0 t21_leaf1 t21_leaf2 t21_leaf3 t21_leaf4 t21_leaf5 t21_leaf6
 t21_leaf7 t22_leaf0 t22_leaf1 t22_leaf2 t22_leaf3 t22_leaf4 t22_leaf5
 t22_leaf6 t22_leaf7 t23_leaf0 t23_leaf1 t23_leaf2 t23_leaf3 t23_leaf4
 t23_leaf5 t23_leaf6 t23_leaf7 t24_leaf0 t24_leaf1 t24_leaf2 t24_leaf3
 t24_leaf4 t24_leaf5 t24_leaf6 t24_leaf7 t25_leaf0 t25_leaf1 t25_leaf2
 t25_leaf3 t25_leaf4 t25_leaf5 t25_leaf6 t25_leaf7 t26_leaf0 t26_leaf1
 t26_leaf2 t26_leaf3 t26_leaf4 t26_leaf5 t26_leaf6 t26_leaf7 t27_leaf0
 t27_leaf1 t27_leaf2 t27_leaf3 t27_leaf4 t27_leaf5 t27_leaf6 t27_leaf7
 t28_leaf0 t28_leaf1 t28_leaf2 t28_leaf3 t28_leaf4 t28_leaf5 t28_leaf6
 t28_leaf7 t29_leaf0 t29_leaf1 t29_leaf2 t29_leaf3 t29_leaf4 t29_leaf5
 t29_leaf6 t29_leaf7 t30_leaf0 t30_leaf1 t30_leaf2 t30_leaf3 t30_leaf4
 t30_leaf5 t30_leaf6 t30_leaf7 t31_leaf0 t31_leaf1 t31_leaf2 t31_leaf3
 t31_leaf4 t31_leaf5 t31_leaf6 t31_leaf7 t32_leaf0 t32_leaf1 t32_leaf2
 t32_leaf3 t32_leaf4 t32_leaf5 t32_leaf6 t32_leaf7 t33_leaf0 t33_leaf1
 t33_leaf2 t33_leaf3 t33_leaf4 t33_leaf5 t33_leaf6 t33_leaf7 t34_leaf0
 t34_leaf1 t34_leaf2 t34_leaf3 t34_leaf4 t34_leaf5 t34_leaf6 t34_leaf7
 t35_leaf0 t35_leaf1 t35_leaf2 t35_leaf3 t35_leaf4 t35_leaf5 t35_leaf6
 t35_leaf7 t36_leaf0 t36_leaf1 t36_leaf2 t36_leaf3 t36_leaf4 t36_leaf5
 t36_leaf6 t36_leaf7 t37_leaf0 t37_leaf1 t37_leaf2 t37_leaf3 t37_leaf4
 t37_leaf5 t37_leaf6 t37_leaf7 t38_leaf0 t38_leaf1 t38_leaf2 t38_leaf3
 t38_leaf4 t38_leaf5 t38_leaf6 t38_leaf7 t39_leaf0 t39_leaf1 t39_leaf2
 t39_leaf3 t39_leaf4 t39_leaf5 t39_leaf6 t39_leaf7 t40_leaf0 t40_leaf1
 t40_leaf2 t40_leaf3 t40_leaf4 t40_leaf5 t40_leaf6 t40_leaf7 t41_leaf0
 t41_leaf1 t41_leaf2 t41_leaf3 t41_leaf4 t41_leaf5 t41_leaf6 t41_leaf7
 t42_leaf0 t42_leaf1 t42_leaf2 t42_leaf3 t42_leaf4 t42_leaf5 t42_leaf6
 t42_leaf7 t43_leaf0 t43_leaf1 t43_leaf2 t43_leaf3 t43_leaf4 t43_leaf5
 t43_leaf6 t43_leaf7 t44_leaf0 t44_leaf1 t44_leaf2 t44_leaf3 t44_leaf4
 t44_leaf5 t44_leaf6 t44_leaf7 t45_leaf0 t45_leaf1 t45_leaf2 t45_leaf3
 t45_leaf4 t45_leaf5 t45_leaf6 t45_leaf7 t46_leaf0 t46_leaf1 t46_leaf2
 t46_leaf3 t46_leaf4 t46_leaf5 t46_leaf6 t46_leaf7 t47_leaf0 t47_leaf1
 t47_leaf2 t47_leaf3 t47_leaf4 t47_leaf5 t47_leaf6 t47_leaf7 t48_leaf0
 t48_leaf1 t48_leaf2 t48_leaf3 t48_leaf4 t48_leaf5 t48_leaf6 t48_leaf7
 t49_leaf0 t49_leaf1 t49_leaf2 t49_leaf3 t49_leaf4 t49_leaf5 t49_leaf6
 t49_leaf7 t50_leaf0 t50_leaf1 t50_leaf2 t50_leaf3 t50_leaf4 t50_leaf5
 t50_leaf6 t51_leaf0 t51_leaf1 t51_leaf2 t51_leaf3 t51_leaf4 t51_leaf5
 t51_leaf6 t52_leaf0 t52_leaf1 t52_leaf2 t52_leaf3 t52_leaf4 t52_leaf5
 t52_leaf6 t52_leaf7 t53_leaf0 t53_leaf1 t53_leaf2 t53_leaf3 t53_leaf4
 t53_leaf5 t53_leaf6 t54_leaf0 t54_leaf1 t54_leaf2 t54_leaf3 t54_leaf4
 t54_leaf5 t54_leaf6 t55_leaf0 t55_leaf1 t55_leaf2 t55_leaf3 t55_leaf4
 t55_leaf5 t55_leaf6 t55_leaf7 t56_leaf0 t56_leaf1 t56_leaf2 t56_leaf3
 t56_leaf4 t56_leaf5 t56_leaf6 t56_leaf7 t57_leaf0 t57_leaf1 t57_leaf2
 t57_leaf3 t57_leaf4 t57_leaf5 t57_leaf6 t57_leaf7 t58_leaf0 t58_leaf1
 t58_leaf2 t58_leaf3 t58_leaf4 t58_leaf5 t58_leaf6 t58_leaf7 t59_leaf0
 t59_leaf1 t59_leaf2 t59_leaf3 t59_leaf4 t59_leaf5 t59_leaf6 t59_leaf7
 t60_leaf0 t60_leaf1 t60_leaf2 t60_leaf3 t60_leaf4 t60_leaf5 t60_leaf6
 t61_leaf0 t61_leaf1 t61_leaf2 t61_leaf3 t61_leaf4 t61_leaf5 t61_leaf6
 t62_leaf0 t62_leaf1 t62_leaf2 t62_leaf3 t62_leaf4 t62_leaf5 t62_leaf6
 t63_leaf0 t63_leaf1 t63_leaf2 t63_leaf3 t63_leaf4 t63_leaf5 t63_leaf6
 t63_leaf7 t64_leaf0 t64_leaf1 t64_leaf2 t64_leaf3 t64_leaf4 t64_leaf5
 t64_leaf6 t64_leaf7 t65_leaf0 t65_leaf1 t65_leaf2 t65_leaf3 t65_leaf4
 t65_leaf5 t65_leaf6 t65_leaf7 t66_leaf0 t66_leaf1 t66_leaf2 t66_leaf3
 t66_leaf4 t66_leaf5 t66_leaf6 t66_leaf7 t67_leaf0 t67_leaf1 t67_leaf2
 t67_leaf3 t67_leaf4 t67_leaf5 t68_leaf0 t68_leaf1 t68_leaf2 t68_leaf3
 t68_leaf4 t68_leaf5 t68_leaf6 t68_leaf7 t69_leaf0 t69_leaf1 t69_leaf2
 t69_leaf3 t69_leaf4 t69_leaf5 t69_leaf6 t70_leaf0 t70_leaf1 t70_leaf2
 t70_leaf3 t70_leaf4 t70_leaf5 t70_leaf6 t70_leaf7 t71_leaf0 t71_leaf1
 t71_leaf2 t71_leaf3 t71_leaf4 t71_leaf5 t71_leaf6 t71_leaf7 t72_leaf0
 t72_leaf1 t72_leaf2 t72_leaf3 t72_leaf4 t72_leaf5 t72_leaf6 t72_leaf7
 t73_leaf0 t73_leaf1 t73_leaf2 t73_leaf3 t73_leaf4 t73_leaf5 t73_leaf6
 t74_leaf0 t74_leaf1 t74_leaf2 t74_leaf3 t74_leaf4 t74_leaf5 t74_leaf6
 t75_leaf0 t75_leaf1 t75_leaf2 t75_leaf3 t75_leaf4 t75_leaf5 t75_leaf6
 t76_leaf0 t76_leaf1 t76_leaf2 t76_leaf3 t76_leaf4 t76_leaf5 t76_leaf6
 t76_leaf7 t77_leaf0 t77_leaf1 t77_leaf2 t77_leaf3 t77_leaf4 t77_leaf5
 t77_leaf6 t78_leaf0 t78_leaf1 t78_leaf2 t78_leaf3 t78_leaf4 t78_leaf5
 t78_leaf6 t78_leaf7 t79_leaf0 t79_leaf1 t79_leaf2 t79_leaf3 t79_leaf4
 t79_leaf5 t79_leaf6 t80_leaf0 t80_leaf1 t80_leaf2 t80_leaf3 t80_leaf4
 t80_leaf5 t80_leaf6 t81_leaf0 t81_leaf1 t81_leaf2 t81_leaf3 t81_leaf4
 t82_leaf0 t82_leaf1 t82_leaf2 t82_leaf3 t82_leaf4 t82_leaf5 t82_leaf6
 t83_leaf0 t83_leaf1 t83_leaf2 t83_leaf3 t83_leaf4 t84_leaf0 t84_leaf1
 t84_leaf2 t84_leaf3 t84_leaf4 t84_leaf5 t84_leaf6 t85_leaf0 t85_leaf1
 t85_leaf2 t85_leaf3 t85_leaf4 t85_leaf5 t85_leaf6 t86_leaf0 t86_leaf1
 t86_leaf2 t86_leaf3 t86_leaf4 t86_leaf5 t86_leaf6 t87_leaf0 t87_leaf1
 t87_leaf2 t87_leaf3 t87_leaf4 t87_leaf5 t87_leaf6 t88_leaf0 t88_leaf1
 t88_leaf2 t88_leaf3 t88_leaf4 t88_leaf5 t88_leaf6 t89_leaf0 t89_leaf1
 t89_leaf2 t89_leaf3 t89_leaf4 t89_leaf5 t89_leaf6 t89_leaf7 t90_leaf0
 t90_leaf1 t90_leaf2 t90_leaf3 t90_leaf4 t90_leaf5 t90_leaf6 t90_leaf7
 t91_leaf0 t91_leaf1 t91_leaf2 t91_leaf3 t91_leaf4 t91_leaf5 t91_leaf6
 t91_leaf7 t92_leaf0 t92_leaf1 t92_leaf2 t92_leaf3 t92_leaf4 t92_leaf5
 t92_leaf6 t93_leaf0 t93_leaf1 t93_leaf2 t93_leaf3 t93_leaf4 t93_leaf5
 t93_leaf6 t93_leaf7 t94_leaf0 t94_leaf1 t94_leaf2 t94_leaf3 t94_leaf4
 t94_leaf5 t94_leaf6 t95_leaf0 t95_leaf1 t95_leaf2 t95_leaf3 t95_leaf4
 t95_leaf5 t95_leaf6 t95_leaf7 t96_leaf0 t96_leaf1 t96_leaf2 t96_leaf3
 t96_leaf4 t96_leaf5 t96_leaf6 t97_leaf0 t97_leaf1 t97_leaf2 t97_leaf3
 t97_leaf4 t97_leaf5 t97_leaf6 t97_leaf7 t98_leaf0 t98_leaf1 t98_leaf2
 t98_leaf3 t98_leaf4 t98_leaf5 t98_leaf6 t99_leaf0 t99_leaf1 t99_leaf2
 t99_leaf3 t99_leaf4 t99_leaf5 t99_leaf6 t100_leaf0 t100_leaf1 t100_leaf2
 t100_leaf3 t100_leaf4 t100_leaf5 t100_leaf6 t100_leaf7 t101_leaf0
 t101_leaf1 t101_leaf2 t101_leaf3 t101_leaf4 t101_leaf5 t101_leaf6
 t102_leaf0 t102_leaf1 t102_leaf2 t102_leaf3 t102_leaf4 t102_leaf5
 t102_leaf6 t102_leaf7 t103_leaf0 t103_leaf1 t103_leaf2 t103_leaf3
 t103_leaf4 t103_leaf5 t104_leaf0 t104_leaf1 t104_leaf2 t104_leaf3
 t104_leaf4 t104_leaf5 t105_leaf0 t105_leaf1 t105_leaf2 t105_leaf3
 t105_leaf4 t105_leaf5 t105_leaf6 t105_leaf7 t106_leaf0 t106_leaf1
 t106_leaf2 t106_leaf3 t106_leaf4 t106_leaf5 t106_leaf6 t107_leaf0
 t107_leaf1 t107_leaf2 t107_leaf3 t107_leaf4 t107_leaf5 t107_leaf6
 t107_leaf7 t108_leaf0 t108_leaf1 t108_leaf2 t108_leaf3 t108_leaf4
 t108_leaf5 t108_leaf6 t108_leaf7 t109_leaf0 t109_leaf1 t109_leaf2
 t109_leaf3 t109_leaf4 t109_leaf5 t109_leaf6 t110_leaf0 t110_leaf1
 t110_leaf2 t110_leaf3 t110_leaf4 t110_leaf5 t111_leaf0 t111_leaf1
 t111_leaf2 t111_leaf3 t111_leaf4 t111_leaf5 t111_leaf6 t111_leaf7
 t112_leaf0 t112_leaf1 t112_leaf2 t112_leaf3 t112_leaf4 t112_leaf5
 t112_leaf6 t112_leaf7 t113_leaf0 t113_leaf1 t113_leaf2 t113_leaf3
 t113_leaf4 t113_leaf5 t113_leaf6 t114_leaf0 t114_leaf1 t114_leaf2
 t114_leaf3 t114_leaf4 t114_leaf5 t114_leaf6 t114_leaf7 t115_leaf0
 t115_leaf1 t115_leaf2 t115_leaf3 t115_leaf4 t115_leaf5 t116_leaf0
 t116_leaf1 t116_leaf2 t116_leaf3 t116_leaf4 t116_leaf5 t116_leaf6
 t117_leaf0 t117_leaf1 t117_leaf2 t117_leaf3 t117_leaf4 t118_leaf0
 t118_leaf1 t118_leaf2 t118_leaf3 t118_leaf4 t118_leaf5 t119_leaf0
 t119_leaf1 t119_leaf2 t119_leaf3 t120_leaf0 t120_leaf1 t120_leaf2
 t120_leaf3 t120_leaf4 t120_leaf5 t121_leaf0 t121_leaf1 t121_leaf2
 t121_leaf3 t121_leaf4 t121_leaf5 t121_leaf6 t122_leaf0 t122_leaf1
 t122_leaf2 t122_leaf3 t122_leaf4 t123_leaf0 t123_leaf1 t123_leaf2
 t123_leaf3 t123_leaf4 t123_leaf5 t123_leaf6 t123_leaf7 t124_leaf0
 t124_leaf1 t124_leaf2 t124_leaf3 t124_leaf4 t124_leaf5 t125_leaf0
 t125_leaf1 t125_leaf2 t125_leaf3 t125_leaf4 t125_leaf5 t126_leaf0
 t126_leaf1 t126_leaf2 t126_leaf3 t126_leaf4 t127_leaf0 t127_leaf1
 t127_leaf2 t127_leaf3 t127_leaf4 t127_leaf5 t128_leaf0 t128_leaf1
 t128_leaf2 t128_leaf3 t128_leaf4 t129_leaf0 t129_leaf1 t129_leaf2
 t129_leaf3 t129_leaf4 t129_leaf5 t130_leaf0 t130_leaf1 t130_leaf2
 t130_leaf3 t130_leaf4 t130_leaf5 t130_leaf6 t130_leaf7 t131_leaf0
 t131_leaf1 t131_leaf2 t131_leaf3 t131_leaf4 t131_leaf5 t131_leaf6
 t131_leaf7 t132_leaf0 t132_leaf1 t132_leaf2 t132_leaf3 t132_leaf4
 t133_leaf0 t133_leaf1 t133_leaf2 t133_leaf3 t133_leaf4 t134_leaf0
 t134_leaf1 t134_leaf2 t134_leaf3 t134_leaf4 t135_leaf0 t135_leaf1
 t135_leaf2 t135_leaf3 t135_leaf4 t135_leaf5 t135_leaf6 t136_leaf0
 t136_leaf1 t136_leaf2 t136_leaf3 t136_leaf4 t137_leaf0 t137_leaf1
 t137_leaf2 t137_leaf3 t137_leaf4 t137_leaf5 t137_leaf6 t137_leaf7
 t138_leaf0 t138_leaf1 t138_leaf2 t138_leaf3 t138_leaf4 t138_leaf5
 t138_leaf6 t139_leaf0 t139_leaf1 t139_leaf2 t139_leaf3 t139_leaf4
 t139_leaf5 t139_leaf6 t139_leaf7 t140_leaf0 t140_leaf1 t140_leaf2
 t140_leaf3 t140_leaf4 t140_leaf5 t140_leaf6 t140_leaf7 t141_leaf0
 t141_leaf1 t141_leaf2 t141_leaf3 t141_leaf4 t141_leaf5 t142_leaf0
 t142_leaf1 t142_leaf2 t142_leaf3 t142_leaf4 t143_leaf0 t143_leaf1
 t143_leaf2 t143_leaf3 t143_leaf4 t143_leaf5 t143_leaf6 t144_leaf0
 t144_leaf1 t144_leaf2 t144_leaf3 t144_leaf4 t144_leaf5 t145_leaf0
 t145_leaf1 t145_leaf2 t145_leaf3 t145_leaf4 t145_leaf5 t146_leaf0
 t146_leaf1 t146_leaf2 t146_leaf3 t146_leaf4 t146_leaf5 t146_leaf6
 t146_leaf7 t147_leaf0 t147_leaf1 t147_leaf2 t147_leaf3 t147_leaf4
 t147_leaf5 t148_leaf0 t148_leaf1 t148_leaf2 t148_leaf3 t148_leaf4
 t148_leaf5 t149_leaf0 t149_leaf1 t149_leaf2 t149_leaf3 t149_leaf4
 t149_leaf5 t149_leaf6 t150_leaf0 t150_leaf1 t150_leaf2 t150_leaf3
 t150_leaf4 t150_leaf5 t150_leaf6 t151_leaf0 t151_leaf1 t151_leaf2
 t151_leaf3 t151_leaf4 t152_leaf0 t152_leaf1 t152_leaf2 t152_leaf3
 t152_leaf4 t152_leaf5 t153_leaf0 t153_leaf1 t153_leaf2 t153_leaf3
 t153_leaf4 t153_leaf5 t153_leaf6 t154_leaf0 t154_leaf1 t154_leaf2
 t154_leaf3 t154_leaf4 t154_leaf5 t154_leaf6 t154_leaf7 t155_leaf0
 t155_leaf1 t155_leaf2 t155_leaf3 t155_leaf4 t156_leaf0 t156_leaf1
 t156_leaf2 t156_leaf3 t156_leaf4 t157_leaf0 t157_leaf1 t157_leaf2
 t157_leaf3 t157_leaf4 t157_leaf5 t157_leaf6 t158_leaf0 t158_leaf1
 t158_leaf2 t158_leaf3 t158_leaf4 t158_leaf5 t158_leaf6 t159_leaf0
 t159_leaf1 t159_leaf2 t159_leaf3 t159_leaf4 t160_leaf0 t160_leaf1
 t160_leaf2 t160_leaf3 t160_leaf4 t160_leaf5 t160_leaf6 t161_leaf0
 t161_leaf1 t161_leaf2 t161_leaf3 t161_leaf4 t161_leaf5 t161_leaf6
 t162_leaf0 t162_leaf1 t162_leaf2 t162_leaf3 t162_leaf4 t162_leaf5
 t162_leaf6 t163_leaf0 t163_leaf1 t163_leaf2 t163_leaf3 t163_leaf4
 t163_leaf5 t164_leaf0 t164_leaf1 t164_leaf2 t164_leaf3 t164_leaf4
 t165_leaf0 t165_leaf1 t165_leaf2 t165_leaf3 t165_leaf4 t166_leaf0
 t166_leaf1 t166_leaf2 t166_leaf3 t167_leaf0 t167_leaf1 t167_leaf2
 t167_leaf3 t167_leaf4 t168_leaf0 t168_leaf1 t168_leaf2 t168_leaf3
 t168_leaf4 t169_leaf0 t169_leaf1 t169_leaf2 t169_leaf3 t169_leaf4
 t169_leaf5 t169_leaf6 t170_leaf0 t170_leaf1 t170_leaf2 t170_leaf3
 t170_leaf4 t171_leaf0 t171_leaf1 t171_leaf2 t171_leaf3 t171_leaf4
 t172_leaf0 t172_leaf1 t172_leaf2 t172_leaf3 t172_leaf4 t172_leaf5
 t172_leaf6 t173_leaf0 t173_leaf1 t173_leaf2 t173_leaf3 t173_leaf4
 t173_leaf5 t173_leaf6 t174_leaf0 t174_leaf1 t174_leaf2 t174_leaf3
 t175_leaf0 t175_leaf1 t175_leaf2 t175_leaf3 t175_leaf4 t176_leaf0
 t176_leaf1 t176_leaf2 t176_leaf3 t176_leaf4 t176_leaf5 t176_leaf6
 t177_leaf0 t177_leaf1 t177_leaf2 t177_leaf3 t177_leaf4 t177_leaf5
 t177_leaf6 t178_leaf0 t178_leaf1 t178_leaf2 t178_leaf3 t178_leaf4
 t179_leaf0 t179_leaf1 t179_leaf2 t179_leaf3 t179_leaf4 t179_leaf5
 t180_leaf0 t180_leaf1 t180_leaf2 t180_leaf3 t180_leaf4 t181_leaf0
 t181_leaf1 t181_leaf2 t181_leaf3 t181_leaf4 t181_leaf5 t181_leaf6
 t182_leaf0 t182_leaf1 t182_leaf2 t182_leaf3 t182_leaf4 t182_leaf5
 t182_leaf6 t182_leaf7 t183_leaf0 t183_leaf1 t183_leaf2 t183_leaf3
 t183_leaf4 t183_leaf5 t184_leaf0 t184_leaf1 t184_leaf2 t184_leaf3
 t184_leaf4 t184_leaf5 t185_leaf0 t185_leaf1 t185_leaf2 t185_leaf3
 t185_leaf4 t185_leaf5 t186_leaf0 t186_leaf1 t186_leaf2 t186_leaf3
 t186_leaf4 t186_leaf5 t186_leaf6 t187_leaf0 t187_leaf1 t187_leaf2
 t187_leaf3 t187_leaf4 t187_leaf5 t187_leaf6 t188_leaf0 t188_leaf1
 t188_leaf2 t188_leaf3 t188_leaf4 t188_leaf5 t188_leaf6 t188_leaf7
 t189_leaf0 t189_leaf1 t189_leaf2 t189_leaf3 t189_leaf4 t190_leaf0
 t190_leaf1 t190_leaf2 t190_leaf3 t190_leaf4 t190_leaf5 t191_leaf0
 t191_leaf1 t191_leaf2 t191_leaf3 t191_leaf4 t191_leaf5 t191_leaf6
 t192_leaf0 t192_leaf1 t192_leaf2 t192_leaf3 t192_leaf4 t193_leaf0
 t193_leaf1 t193_leaf2 t193_leaf3 t193_leaf4 t193_leaf5 t193_leaf6
 t193_leaf7 t194_leaf0 t194_leaf1 t194_leaf2 t194_leaf3 t195_leaf0
 t195_leaf1 t195_leaf2 t195_leaf3 t196_leaf0 t196_leaf1 t196_leaf2
 t196_leaf3 t196_leaf4 t196_leaf5 t197_leaf0 t197_leaf1 t197_leaf2
 t197_leaf3 t197_leaf4 t197_leaf5 t198_leaf0 t198_leaf1 t198_leaf2
 t198_leaf3 t198_leaf4 t198_leaf5 t199_leaf0 t199_leaf1 t199_leaf2
 t199_leaf3 t199_leaf4 t200_leaf0 t200_leaf1 t200_leaf2 t200_leaf3
 t200_leaf4 t201_leaf0 t201_leaf1 t201_leaf2 t201_leaf3 t201_leaf4
 t202_leaf0 t202_leaf1 t202_leaf2 t202_leaf3 t202_leaf4 t202_leaf5
 t202_leaf6 t203_leaf0 t203_leaf1 t203_leaf2 t203_leaf3 t203_leaf4
 t203_leaf5 t203_leaf6 t203_leaf7 t204_leaf0 t204_leaf1 t204_leaf2
 t204_leaf3 t204_leaf4 t205_leaf0 t205_leaf1 t205_leaf2 t205_leaf3
 t206_leaf0 t206_leaf1 t206_leaf2 t206_leaf3 t207_leaf0 t207_leaf1
 t207_leaf2 t207_leaf3 t207_leaf4 t207_leaf5 t208_leaf0 t208_leaf1
 t208_leaf2 t208_leaf3 t208_leaf4 t209_leaf0 t209_leaf1 t209_leaf2
 t209_leaf3 t209_leaf4 t210_leaf0 t210_leaf1 t210_leaf2 t210_leaf3
 t210_leaf4 t211_leaf0 t211_leaf1 t211_leaf2 t211_leaf3 t211_leaf4
 t211_leaf5 t211_leaf6 t212_leaf0 t212_leaf1 t212_leaf2 t212_leaf3
 t212_leaf4 t212_leaf5 t213_leaf0 t213_leaf1 t213_leaf2 t213_leaf3
 t214_leaf0 t214_leaf1 t214_leaf2 t214_leaf3 t214_leaf4 t214_leaf5
 t214_leaf6 t215_leaf0 t215_leaf1 t215_leaf2 t215_leaf3 t215_leaf4
 t215_leaf5 t216_leaf0 t216_leaf1 t216_leaf2 t216_leaf3 t216_leaf4
 t217_leaf0 t217_leaf1 t217_leaf2 t217_leaf3 t217_leaf4 t218_leaf0
 t218_leaf1 t218_leaf2 t218_leaf3 t219_leaf0 t219_leaf1 t219_leaf2
 t219_leaf3 t219_leaf4 t219_leaf5 t219_leaf6 t220_leaf0 t220_leaf1
 t220_leaf2 t220_leaf3 t220_leaf4 t220_leaf5 t220_leaf6 t221_leaf0
 t221_leaf1 t221_leaf2 t221_leaf3 t221_leaf4 t221_leaf5 t222_leaf0
 t222_leaf1 t222_leaf2 t222_leaf3 t222_leaf4 t223_leaf0 t223_leaf1
 t223_leaf2 t223_leaf3 t223_leaf4 t223_leaf5 t223_leaf6 t224_leaf0
 t224_leaf1 t224_leaf2 t224_leaf3 t225_leaf0 t225_leaf1 t225_leaf2
 t225_leaf3 t225_leaf4 t226_leaf0 t226_leaf1 t226_leaf2 t226_leaf3
 t226_leaf4 t226_leaf5 t226_leaf6 t226_leaf7 t227_leaf0 t227_leaf1
 t227_leaf2 t227_leaf3 t227_leaf4 t228_leaf0 t228_leaf1 t228_leaf2
 t228_leaf3 t228_leaf4 t229_leaf0 t229_leaf1 t229_leaf2 t229_leaf3
 t229_leaf4 t230_leaf0 t230_leaf1 t230_leaf2 t230_leaf3 t230_leaf4
 t230_leaf5 t230_leaf6 t231_leaf0 t231_leaf1 t231_leaf2 t231_leaf3
 t231_leaf4 t231_leaf5 t231_leaf6 t231_leaf7 t232_leaf0 t232_leaf1
 t232_leaf2 t232_leaf3 t232_leaf4 t233_leaf0 t233_leaf1 t233_leaf2
 t233_leaf3 t233_leaf4 t233_leaf5 t233_leaf6 t234_leaf0 t234_leaf1
 t234_leaf2 t234_leaf3 t234_leaf4 t234_leaf5 t235_leaf0 t235_leaf1
 t235_leaf2 t235_leaf3 t235_leaf4 t236_leaf0 t236_leaf1 t236_leaf2
 t236_leaf3 t236_leaf4 t236_leaf5 t237_leaf0 t237_leaf1 t237_leaf2
 t237_leaf3 t238_leaf0 t238_leaf1 t238_leaf2 t238_leaf3 t238_leaf4
 t238_leaf5 t239_leaf0 t239_leaf1 t239_leaf2 t239_leaf3 t239_leaf4
 t239_leaf5 t239_leaf6 t240_leaf0 t240_leaf1 t240_leaf2 t240_leaf3
 t240_leaf4 t240_leaf5 t241_leaf0 t241_leaf1 t241_leaf2 t241_leaf3
 t241_leaf4 t242_leaf0 t242_leaf1 t242_leaf2 t242_leaf3 t242_leaf4
 t243_leaf0 t243_leaf1 t243_leaf2 t243_leaf3 t243_leaf4 t244_leaf0
 t244_leaf1 t244_leaf2 t244_leaf3 t245_leaf0 t245_leaf1 t245_leaf2
 t245_leaf3 t245_leaf4 t245_leaf5 t245_leaf6 t246_leaf0 t246_leaf1
 t246_leaf2 t246_leaf3 t247_leaf0 t247_leaf1 t247_leaf2 t247_leaf3
 t247_leaf4 t248_leaf0 t248_leaf1 t248_leaf2 t248_leaf3 t248_leaf4
 t248_leaf5 t248_leaf6 t248_leaf7 t249_leaf0 t249_leaf1 t249_leaf2
 t249_leaf3 t249_leaf4 t250_leaf0 t250_leaf1 t250_leaf2 t250_leaf3
 t251_leaf0 t251_leaf1 t251_leaf2 t251_leaf3 t251_leaf4 t252_leaf0
 t252_leaf1 t252_leaf2 t252_leaf3 t252_leaf4 t252_leaf5 t252_leaf6
 t252_leaf7 t253_leaf0 t253_leaf1 t253_leaf2 t253_leaf3 t253_leaf4
 t253_leaf5 t254_leaf0 t254_leaf1 t254_leaf2 t254_leaf3 t255_leaf0
 t255_leaf1 t255_leaf2 t255_leaf3 t255_leaf4 t255_leaf5 t255_leaf6
 t256_leaf0 t256_leaf1 t256_leaf2 t256_leaf3 t256_leaf4 t256_leaf5
 t256_leaf6 t257_leaf0 t257_leaf1 t257_leaf2 t257_leaf3 t257_leaf4
 t257_leaf5 t257_leaf6 t258_leaf0 t258_leaf1 t258_leaf2 t258_leaf3
 t258_leaf4 t259_leaf0 t259_leaf1 t259_leaf2 t259_leaf3 t259_leaf4
 t260_leaf0 t260_leaf1 t260_leaf2 t260_leaf3 t260_leaf4 t260_leaf5
 t261_leaf0 t261_leaf1 t261_leaf2 t261_leaf3 t262_leaf0 t262_leaf1
 t262_leaf2 t262_leaf3 t262_leaf4 t263_leaf0 t263_leaf1 t263_leaf2
 t263_leaf3 t263_leaf4 t263_leaf5 t264_leaf0 t264_leaf1 t264_leaf2
 t264_leaf3 t264_leaf4 t265_leaf0 t265_leaf1 t265_leaf2 t265_leaf3
 t265_leaf4 t266_leaf0 t266_leaf1 t266_leaf2 t266_leaf3 t266_leaf4
 t266_leaf5 t266_leaf6 t267_leaf0 t267_leaf1 t267_leaf2 t267_leaf3
 t267_leaf4 t267_leaf5 t268_leaf0 t268_leaf1 t268_leaf2 t268_leaf3
 t268_leaf4 t268_leaf5 t269_leaf0 t269_leaf1 t269_leaf2 t269_leaf3
 t269_leaf4 t270_leaf0 t270_leaf1 t270_leaf2 t270_leaf3 t270_leaf4
 t270_leaf5 t271_leaf0 t271_leaf1 t271_leaf2 t271_leaf3 t271_leaf4
 t271_leaf5 t272_leaf0 t272_leaf1 t272_leaf2 t272_leaf3 t272_leaf4
 t272_leaf5 t272_leaf6 t272_leaf7 t273_leaf0 t273_leaf1 t273_leaf2
 t273_leaf3 t273_leaf4 t274_leaf0 t274_leaf1 t274_leaf2 t274_leaf3
 t274_leaf4 t275_leaf0 t275_leaf1 t275_leaf2 t275_leaf3 t275_leaf4
 t275_leaf5 t276_leaf0 t276_leaf1 t276_leaf2 t276_leaf3 t276_leaf4
 t277_leaf0 t277_leaf1 t277_leaf2 t277_leaf3 t277_leaf4 t277_leaf5
 t277_leaf6 t278_leaf0 t278_leaf1 t278_leaf2 t278_leaf3 t279_leaf0
 t279_leaf1 t279_leaf2 t279_leaf3 t279_leaf4 t279_leaf5 t280_leaf0
 t280_leaf1 t280_leaf2 t280_leaf3 t280_leaf4 t280_leaf5 t280_leaf6
 t281_leaf0 t281_leaf1 t281_leaf2 t281_leaf3 t281_leaf4 t282_leaf0
 t282_leaf1 t282_leaf2 t282_leaf3 t283_leaf0 t283_leaf1 t283_leaf2
 t283_leaf3 t283_leaf4 t283_leaf5 t284_leaf0 t284_leaf1 t284_leaf2
 t284_leaf3 t284_leaf4 t285_leaf0 t285_leaf1 t285_leaf2 t285_leaf3
 t285_leaf4 t285_leaf5 t286_leaf0 t286_leaf1 t286_leaf2 t286_leaf3
 t286_leaf4 t286_leaf5 t287_leaf0 t287_leaf1 t287_leaf2 t287_leaf3
 t287_leaf4 t287_leaf5 t288_leaf0 t288_leaf1 t288_leaf2 t288_leaf3
 t288_leaf4 t288_leaf5 t289_leaf0 t289_leaf1 t289_leaf2 t289_leaf3
 t289_leaf4 t289_leaf5 t290_leaf0 t290_leaf1 t290_leaf2 t290_leaf3
 t290_leaf4 t290_leaf5 t291_leaf0 t291_leaf1 t291_leaf2 t291_leaf3
 t291_leaf4 t292_leaf0 t292_leaf1 t292_leaf2 t292_leaf3 t292_leaf4
 t293_leaf0 t293_leaf1 t293_leaf2 t293_leaf3 t293_leaf4 t294_leaf0
 t294_leaf1 t294_leaf2 t294_leaf3 t294_leaf4 t295_leaf0 t295_leaf1
 t295_leaf2 t295_leaf3 t295_leaf4 t295_leaf5 t296_leaf0 t296_leaf1
 t296_leaf2 t296_leaf3 t296_leaf4 t296_leaf5 t297_leaf0 t297_leaf1
 t297_leaf2 t297_leaf3 t298_leaf0 t298_leaf1 t298_leaf2 t298_leaf3
 t298_leaf4 t299_leaf0 t299_leaf1 t299_leaf2 t299_leaf3 t299_leaf4
 t299_leaf5 t299_leaf6 t300_leaf0 t300_leaf1 t300_leaf2 t300_leaf3
 t300_leaf4 t300_leaf5 t301_leaf0 t301_leaf1 t301_leaf2 t301_leaf3
 t301_leaf4 t302_leaf0 t302_leaf1 t302_leaf2 t302_leaf3 t303_leaf0
 t303_leaf1 t303_leaf2 t303_leaf3 t303_leaf4 t304_leaf0 t304_leaf1
 t304_leaf2 t304_leaf3 t304_leaf4 t305_leaf0 t305_leaf1 t305_leaf2
 t305_leaf3 t305_leaf4 t306_leaf0 t306_leaf1 t306_leaf2 t306_leaf3
 t306_leaf4 t306_leaf5 t307_leaf0 t307_leaf1 t307_leaf2 t307_leaf3
 t307_leaf4 t307_leaf5 t308_leaf0 t308_leaf1 t308_leaf2 t308_leaf3
 t308_leaf4 t309_leaf0 t309_leaf1 t309_leaf2 t309_leaf3 t309_leaf4
 t309_leaf5 t310_leaf0 t310_leaf1 t310_leaf2 t310_leaf3 t310_leaf4
 t311_leaf0 t311_leaf1 t311_leaf2 t311_leaf3 t312_leaf0 t312_leaf1
 t312_leaf2 t312_leaf3 t312_leaf4 t313_leaf0 t313_leaf1 t313_leaf2
 t313_leaf3 t313_leaf4 t313_leaf5 t313_leaf6 t314_leaf0 t314_leaf1
 t314_leaf2 t314_leaf3 t314_leaf4 t314_leaf5 t314_leaf6 t315_leaf0
 t315_leaf1 t315_leaf2 t315_leaf3 t315_leaf4 t316_leaf0 t316_leaf1
 t316_leaf2 t316_leaf3 t316_leaf4 t317_leaf0 t317_leaf1 t317_leaf2
 t317_leaf3 t318_leaf0 t318_leaf1 t318_leaf2 t318_leaf3 t319_leaf0
 t319_leaf1 t319_leaf2 t319_leaf3 t319_leaf4 t320_leaf0 t320_leaf1
 t320_leaf2 t320_leaf3 t320_leaf4 t320_leaf5 t320_leaf6 t321_leaf0
 t321_leaf1 t321_leaf2 t321_leaf3 t321_leaf4 t322_leaf0 t322_leaf1
 t322_leaf2 t322_leaf3 t322_leaf4 t322_leaf5 t322_leaf6 t323_leaf0
 t323_leaf1 t323_leaf2 t323_leaf3 t324_leaf0 t324_leaf1 t324_leaf2
 t324_leaf3 t324_leaf4 t324_leaf5 t324_leaf6 t325_leaf0 t325_leaf1
 t325_leaf2 t325_leaf3 t325_leaf4 t326_leaf0 t326_leaf1 t326_leaf2
 t326_leaf3 t327_leaf0 t327_leaf1 t327_leaf2 t327_leaf3 t327_leaf4
 t328_leaf0 t328_leaf1 t328_leaf2 t328_leaf3 t328_leaf4 t328_leaf5
 t329_leaf0 t329_leaf1 t329_leaf2 t329_leaf3 t329_leaf4 t329_leaf5
 t329_leaf6 t330_leaf0 t330_leaf1 t330_leaf2 t330_leaf3 t330_leaf4
 t330_leaf5 t331_leaf0 t331_leaf1 t331_leaf2 t331_leaf3 t331_leaf4
 t332_leaf0 t332_leaf1 t332_leaf2 t332_leaf3 t333_leaf0 t333_leaf1
 t333_leaf2 t334_leaf0 t334_leaf1 t334_leaf2 t334_leaf3 t334_leaf4
 t334_leaf5 t334_leaf6 t335_leaf0 t335_leaf1 t335_leaf2 t335_leaf3
 t335_leaf4 t335_leaf5 t336_leaf0 t336_leaf1 t336_leaf2 t336_leaf3
 t336_leaf4 t336_leaf5 t336_leaf6 t336_leaf7 t337_leaf0 t337_leaf1
 t337_leaf2 t337_leaf3 t337_leaf4 t337_leaf5 t338_leaf0 t338_leaf1
 t338_leaf2 t338_leaf3 t338_leaf4 t338_leaf5 t338_leaf6 t339_leaf0
 t339_leaf1 t339_leaf2 t339_leaf3 t339_leaf4 t339_leaf5 t340_leaf0
 t340_leaf1 t340_leaf2 t340_leaf3 t340_leaf4 t340_leaf5 t340_leaf6
 t341_leaf0 t341_leaf1 t341_leaf2 t342_leaf0 t342_leaf1 t342_leaf2
 t342_leaf3 t342_leaf4 t343_leaf0 t343_leaf1 t343_leaf2 t343_leaf3
 t343_leaf4 t344_leaf0 t344_leaf1 t344_leaf2 t344_leaf3 t344_leaf4
 t344_leaf5 t344_leaf6 t345_leaf0 t345_leaf1 t345_leaf2 t345_leaf3
 t345_leaf4 t345_leaf5 t345_leaf6 t346_leaf0 t346_leaf1 t346_leaf2
 t346_leaf3 t346_leaf4 t347_leaf0 t347_leaf1 t347_leaf2 t347_leaf3
 t348_leaf0 t348_leaf1 t348_leaf2 t348_leaf3 t348_leaf4 t349_leaf0
 t349_leaf1 t349_leaf2 t349_leaf3 t349_leaf4 t350_leaf0 t350_leaf1
 t350_leaf2 t350_leaf3 t350_leaf4 t350_leaf5 t350_leaf6 t351_leaf0
 t351_leaf1 t351_leaf2 t351_leaf3 t351_leaf4 t351_leaf5 t352_leaf0
 t352_leaf1 t353_leaf0 t353_leaf1 t353_leaf2 t353_leaf3 t353_leaf4
 t354_leaf0 t354_leaf1 t354_leaf2 t355_leaf0 t355_leaf1 t355_leaf2
 t355_leaf3 t355_leaf4 t355_leaf5 t355_leaf6 t356_leaf0 t356_leaf1
 t356_leaf2 t356_leaf3 t356_leaf4 t356_leaf5 t356_leaf6 t357_leaf0
 t357_leaf1 t357_leaf2 t357_leaf3 t358_leaf0 t358_leaf1 t358_leaf2
 t358_leaf3 t358_leaf4 t358_leaf5 t358_leaf6 t359_leaf0 t359_leaf1
 t359_leaf2 t359_leaf3 t359_leaf4 t359_leaf5 t359_leaf6 t360_leaf0
 t360_leaf1 t361_leaf0 t361_leaf1 t361_leaf2 t361_leaf3 t362_leaf0
 t362_leaf1 t362_leaf2 t362_leaf3 t362_leaf4 t363_leaf0 t363_leaf1
 t363_leaf2 t363_leaf3 t363_leaf4 t364_leaf0 t364_leaf1 t364_leaf2
 t364_leaf3 t364_leaf4 t365_leaf0 t365_leaf1 t365_leaf2 t365_leaf3
 t365_leaf4 t366_leaf0 t366_leaf1 t366_leaf2 t366_leaf3 t366_leaf4
 t367_leaf0 t367_leaf1 t367_leaf2 t367_leaf3 t368_leaf0 t368_leaf1
 t368_leaf2 t368_leaf3 t368_leaf4 t369_leaf0 t369_leaf1 t369_leaf2
 t369_leaf3 t369_leaf4 t369_leaf5 t370_leaf0 t370_leaf1 t370_leaf2
 t371_leaf0 t371_leaf1 t371_leaf2 t371_leaf3 t371_leaf4 t372_leaf0
 t372_leaf1 t372_leaf2 t373_leaf0 t373_leaf1 t373_leaf2 t373_leaf3
 t373_leaf4 t374_leaf0 t374_leaf1 t374_leaf2 t374_leaf3 t374_leaf4
 t374_leaf5 t375_leaf0 t375_leaf1 t375_leaf2 t375_leaf3 t376_leaf0
 t376_leaf1 t376_leaf2 t376_leaf3 t377_leaf0 t378_leaf0 t378_leaf1
 t378_leaf2 t379_leaf0 t379_leaf1 t379_leaf2 t379_leaf3 t379_leaf4
 t380_leaf0 t380_leaf1 t381_leaf0 t381_leaf1 t381_leaf2 t381_leaf3
 t381_leaf4 t382_leaf0 t382_leaf1 t382_leaf2 t382_leaf3 t382_leaf4
 t382_leaf5 t383_leaf0 t383_leaf1 t383_leaf2 t383_leaf3 t383_leaf4
 t383_leaf5 t383_leaf6 t384_leaf0 t384_leaf1 t384_leaf2 t384_leaf3
 t384_leaf4 t385_leaf0 t385_leaf1 t385_leaf2 t385_leaf3 t385_leaf4
 t386_leaf0 t386_leaf1 t386_leaf2 t386_leaf3 t387_leaf0 t387_leaf1
 t387_leaf2 t387_leaf3 t387_leaf4 t388_leaf0 t388_leaf1 t388_leaf2
 t388_leaf3 t389_leaf0 t389_leaf1 t389_leaf2 t389_leaf3 t390_leaf0
 t390_leaf1 t390_leaf2 t391_leaf0 t391_leaf1 t391_leaf2 t391_leaf3
 t392_leaf0 t392_leaf1 t392_leaf2 t392_leaf3 t393_leaf0 t393_leaf1
 t393_leaf2 t393_leaf3 t393_leaf4 t394_leaf0 t394_leaf1 t395_leaf0
 t395_leaf1 t395_leaf2 t395_leaf3 t395_leaf4 t395_leaf5 t396_leaf0
 t396_leaf1 t396_leaf2 t396_leaf3 t396_leaf4 t397_leaf0 t397_leaf1
 t397_leaf2 t397_leaf3 t397_leaf4 t398_leaf0 t398_leaf1 t398_leaf2
 t398_leaf3 t399_leaf0 t399_leaf1 t399_leaf2 t399_leaf3 t400_leaf0
 t400_leaf1 t400_leaf2 t401_leaf0 t402_leaf0 t402_leaf1 t402_leaf2
 t402_leaf3 t402_leaf4 t403_leaf0 t403_leaf1 t403_leaf2 t403_leaf3
 t403_leaf4 t404_leaf0 t404_leaf1 t404_leaf2 t404_leaf3 t404_leaf4
 t404_leaf5 t405_leaf0 t405_leaf1 t405_leaf2 t405_leaf3 t405_leaf4
 t406_leaf0 t407_leaf0 t408_leaf0 t408_leaf1 t408_leaf2 t408_leaf3
 t408_leaf4 t409_leaf0 t409_leaf1 t409_leaf2 t409_leaf3 t409_leaf4
 t409_leaf5 t410_leaf0 t410_leaf1 t410_leaf2 t410_leaf3 t411_leaf0
 t411_leaf1 t411_leaf2 t412_leaf0 t412_leaf1 t412_leaf2 t413_leaf0
 t413_leaf1 t413_leaf2 t413_leaf3 t413_leaf4 t414_leaf0 t414_leaf1
 t414_leaf2 t414_leaf3 t415_leaf0 t415_leaf1 t415_leaf2 t415_leaf3
 t415_leaf4 t416_leaf0 t416_leaf1 t416_leaf2 t416_leaf3 t416_leaf4
 t416_leaf5 t416_leaf6 t417_leaf0 t418_leaf0 t418_leaf1 t419_leaf0
 t419_leaf1 t419_leaf2 t419_leaf3 t419_leaf4 t419_leaf5 t419_leaf6
 t419_leaf7 t420_leaf0 t421_leaf0 t421_leaf1 t421_leaf2 t421_leaf3
 t422_leaf0 t422_leaf1 t422_leaf2 t422_leaf3 t422_leaf4 t423_leaf0
 t423_leaf1 t423_leaf2 t423_leaf3 t424_leaf0 t424_leaf1 t424_leaf2
 t424_leaf3 t424_leaf4 t424_leaf5 t425_leaf0 t425_leaf1 t425_leaf2
 t426_leaf0 t426_leaf1 t426_leaf2 t426_leaf3 t426_leaf4 t426_leaf5
 t426_leaf6 t427_leaf0 t427_leaf1 t427_leaf2 t427_leaf3 t428_leaf0
 t428_leaf1 t428_leaf2 t428_leaf3 t428_leaf4 t429_leaf0 t429_leaf1
 t429_leaf2 t429_leaf3 t429_leaf4 t429_leaf5 t430_leaf0 t430_leaf1
 t430_leaf2 t430_leaf3 t430_leaf4 t430_leaf5 t430_leaf6 t431_leaf0
 t431_leaf1 t431_leaf2 t431_leaf3 t432_leaf0 t432_leaf1 t432_leaf2
 t432_leaf3 t432_leaf4 t432_leaf5 t433_leaf0 t433_leaf1 t433_leaf2
 t434_leaf0 t435_leaf0 t435_leaf1 t435_leaf2 t435_leaf3 t435_leaf4
 t436_leaf0 t437_leaf0 t437_leaf1 t437_leaf2 t438_leaf0 t438_leaf1
 t439_leaf0 t439_leaf1 t439_leaf2 t439_leaf3 t439_leaf4 t440_leaf0
 t440_leaf1 t440_leaf2 t440_leaf3 t440_leaf4 t440_leaf5 t441_leaf0
 t441_leaf1 t441_leaf2 t441_leaf3 t441_leaf4 t442_leaf0 t442_leaf1
 t442_leaf2 t443_leaf0 t444_leaf0 t444_leaf1 t444_leaf2 t444_leaf3
 t444_leaf4 t445_leaf0 t445_leaf1 t445_leaf2 t446_leaf0 t446_leaf1
 t446_leaf2 t446_leaf3 t447_leaf0 t447_leaf1 t447_leaf2 t448_leaf0
 t448_leaf1 t448_leaf2 t449_leaf0 t449_leaf1 t449_leaf2 t449_leaf3
 t449_leaf4 t449_leaf5 t450_leaf0 t451_leaf0 t451_leaf1 t451_leaf2
 t451_leaf3 t451_leaf4 t451_leaf5 t452_leaf0 t452_leaf1 t452_leaf2
 t452_leaf3 t452_leaf4 t453_leaf0 t453_leaf1 t453_leaf2 t453_leaf3
 t453_leaf4 t453_leaf5 t454_leaf0 t454_leaf1 t455_leaf0 t456_leaf0
 t457_leaf0 t457_leaf1 t457_leaf2 t458_leaf0 t458_leaf1 t458_leaf2
 t458_leaf3 t459_leaf0 t459_leaf1 t460_leaf0 t461_leaf0 t461_leaf1
 t461_leaf2 t461_leaf3 t461_leaf4 t462_leaf0 t462_leaf1 t463_leaf0
 t463_leaf1 t463_leaf2 t463_leaf3 t463_leaf4 t463_leaf5 t463_leaf6
 t464_leaf0 t464_leaf1 t464_leaf2 t465_leaf0 t465_leaf1 t465_leaf2
 t465_leaf3 t466_leaf0 t466_leaf1 t466_leaf2 t466_leaf3 t466_leaf4
 t467_leaf0 t468_leaf0 t469_leaf0 t470_leaf0 t470_leaf1 t470_leaf2
 t471_leaf0 t471_leaf1 t471_leaf2 t471_leaf3 t472_leaf0 t473_leaf0
 t473_leaf1 t473_leaf2 t474_leaf0 t474_leaf1 t474_leaf2 t475_leaf0
 t475_leaf1 t475_leaf2 t476_leaf0 t476_leaf1 t476_leaf2 t476_leaf3
 t477_leaf0 t477_leaf1 t477_leaf2 t477_leaf3 t478_leaf0 t478_leaf1
 t478_leaf2 t478_leaf3 t479_leaf0 t479_leaf1 t479_leaf2 t479_leaf3
 t479_leaf4 t479_leaf5 t480_leaf0 t480_leaf1 t480_leaf2 t480_leaf3
 t480_leaf4 t481_leaf0 t481_leaf1 t482_leaf0 t483_leaf0 t484_leaf0
 t484_leaf1 t484_leaf2 t484_leaf3 t485_leaf0 t486_leaf0 t487_leaf0
 t488_leaf0 t488_leaf1 t488_leaf2 t488_leaf3 t489_leaf0 t490_leaf0
 t490_leaf1 t490_leaf2 t491_leaf0 t491_leaf1 t491_leaf2 t491_leaf3
 t492_leaf0 t492_leaf1 t492_leaf2 t493_leaf0 t493_leaf1 t493_leaf2
 t493_leaf3 t493_leaf4 t494_leaf0 t494_leaf1 t494_leaf2 t495_leaf0
 t495_leaf1 t496_leaf0 t496_leaf1 t497_leaf0 t497_leaf1 t497_leaf2
 t497_leaf3 t498_leaf0 t498_leaf1 t498_leaf2 t498_leaf3 t498_leaf4
 t499_leaf0 t499_leaf1 t500_leaf0 t500_leaf1 t500_leaf2 t500_leaf3
 t501_leaf0 t501_leaf1 t501_leaf2 t501_leaf3 t501_leaf4 t501_leaf5
 t501_leaf6 t502_leaf0 t502_leaf1 t503_leaf0 t504_leaf0 t504_leaf1
 t504_leaf2 t504_leaf3 t505_leaf0 t505_leaf1 t506_leaf0 t506_leaf1
 t506_leaf2 t506_leaf3 t506_leaf4 t507_leaf0 t507_leaf1 t508_leaf0
 t508_leaf1 t508_leaf2 t508_leaf3 t508_leaf4 t508_leaf5 t509_leaf0
 t509_leaf1 t509_leaf2 t510_leaf0 t511_leaf0 t512_leaf0 t513_leaf0
 t513_leaf1 t513_leaf2 t513_leaf3 t513_leaf4 t513_leaf5 t514_leaf0
 t515_leaf0 t515_leaf1 t515_leaf2 t515_leaf3 t516_leaf0 t516_leaf1
 t516_leaf2 t516_leaf3 t516_leaf4 t517_leaf0 t517_leaf1 t517_leaf2
 t517_leaf3 t517_leaf4 t517_leaf5 t518_leaf0 t518_leaf1 t518_leaf2
 t518_leaf3 t518_leaf4 t518_leaf5 t519_leaf0 t519_leaf1 t520_leaf0
 t520_leaf1 t520_leaf2 t520_leaf3 t520_leaf4 t520_leaf5 t521_leaf0
 t521_leaf1 t521_leaf2 t521_leaf3 t521_leaf4 t522_leaf0 t522_leaf1
 t522_leaf2 t522_leaf3 t522_leaf4 t522_leaf5 t523_leaf0 t523_leaf1
 t523_leaf2 t523_leaf3 t523_leaf4 t524_leaf0 t524_leaf1 t524_leaf2
 t524_leaf3 t525_leaf0 t525_leaf1 t526_leaf0 t526_leaf1 t526_leaf2
 t526_leaf3 t526_leaf4 t526_leaf5 t527_leaf0 t527_leaf1 t527_leaf2
 t527_leaf3 t527_leaf4 t527_leaf5 t528_leaf0 t528_leaf1 t528_leaf2
 t529_leaf0 t529_leaf1 t529_leaf2 t529_leaf3 t529_leaf4 t530_leaf0
 t531_leaf0 t531_leaf1 t531_leaf2 t532_leaf0 t533_leaf0 t533_leaf1
 t533_leaf2 t534_leaf0 t535_leaf0 t535_leaf1 t535_leaf2 t535_leaf3
 t535_leaf4 t536_leaf0 t537_leaf0 t537_leaf1 t538_leaf0 t538_leaf1
 t538_leaf2 t538_leaf3 t538_leaf4 t539_leaf0 t539_leaf1 t540_leaf0
 t540_leaf1 t540_leaf2 t540_leaf3 t540_leaf4 t541_leaf0 t541_leaf1
 t541_leaf2 t542_leaf0 t543_leaf0 t543_leaf1 t543_leaf2 t543_leaf3
 t543_leaf4 t543_leaf5 t543_leaf6 t544_leaf0 t545_leaf0 t545_leaf1
 t545_leaf2 t545_leaf3 t545_leaf4 t546_leaf0 t546_leaf1 t547_leaf0
 t547_leaf1 t548_leaf0 t548_leaf1 t548_leaf2 t548_leaf3 t549_leaf0
 t549_leaf1 t549_leaf2 t549_leaf3 t549_leaf4 t549_leaf5 t550_leaf0
 t550_leaf1 t550_leaf2 t550_leaf3 t551_leaf0 t552_leaf0 t552_leaf1
 t552_leaf2 t552_leaf3 t552_leaf4 t553_leaf0 t553_leaf1 t553_leaf2
 t553_leaf3 t554_leaf0 t555_leaf0 t556_leaf0 t556_leaf1 t557_leaf0
 t557_leaf1 t557_leaf2 t557_leaf3 t558_leaf0 t558_leaf1 t558_leaf2
 t558_leaf3 t559_leaf0 t559_leaf1 t559_leaf2 t559_leaf3 t559_leaf4
 t560_leaf0 t560_leaf1 t560_leaf2 t560_leaf3 t561_leaf0 t561_leaf1
 t561_leaf2 t562_leaf0 t562_leaf1 t562_leaf2 t563_leaf0 t564_leaf0
 t564_leaf1 t564_leaf2 t565_leaf0 t566_leaf0 t566_leaf1 t566_leaf2
 t566_leaf3 t566_leaf4 t566_leaf5 t567_leaf0 t567_leaf1 t567_leaf2
 t567_leaf3 t567_leaf4 t567_leaf5 t568_leaf0 t568_leaf1 t568_leaf2
 t568_leaf3 t569_leaf0 t570_leaf0 t571_leaf0 t571_leaf1 t571_leaf2
 t571_leaf3 t572_leaf0 t572_leaf1 t572_leaf2 t573_leaf0 t574_leaf0
 t574_leaf1 t574_leaf2 t574_leaf3 t575_leaf0 t575_leaf1 t576_leaf0
 t576_leaf1 t576_leaf2 t576_leaf3 t576_leaf4 t576_leaf5 t577_leaf0
 t578_leaf0 t578_leaf1 t578_leaf2 t579_leaf0 t579_leaf1 t580_leaf0
 t580_leaf1 t580_leaf2 t580_leaf3 t580_leaf4 t580_leaf5 t581_leaf0
 t581_leaf1 t582_leaf0 t582_leaf1 t582_leaf2 t583_leaf0 t584_leaf0
 t584_leaf1 t584_leaf2 t584_leaf3 t584_leaf4 t585_leaf0 t585_leaf1
 t585_leaf2 t585_leaf3 t585_leaf4 t586_leaf0 t586_leaf1 t586_leaf2
 t586_leaf3 t586_leaf4 t587_leaf0 t587_leaf1 t587_leaf2 t587_leaf3
 t587_leaf4 t587_leaf5 t588_leaf0 t588_leaf1 t588_leaf2 t588_leaf3
 t588_leaf4 t588_leaf5 t588_leaf6 t588_leaf7 t589_leaf0 t590_leaf0
 t590_leaf1 t591_leaf0 t592_leaf0 t593_leaf0 t594_leaf0 t595_leaf0
 t595_leaf1 t595_leaf2 t595_leaf3 t595_leaf4 t596_leaf0 t597_leaf0
 t597_leaf1 t597_leaf2 t597_leaf3 t597_leaf4 t598_leaf0 t598_leaf1
 t599_leaf0 t600_leaf0 t600_leaf1 t600_leaf2 t600_leaf3 t601_leaf0
 t601_leaf1 t601_leaf2 t601_leaf3 t602_leaf0 t603_leaf0 t603_leaf1
 t604_leaf0 t605_leaf0 t605_leaf1 t605_leaf2 t605_leaf3 t606_leaf0
 t607_leaf0 t607_leaf1 t608_leaf0 t608_leaf1 t609_leaf0 t610_leaf0
 t611_leaf0 t611_leaf1 t612_leaf0 t613_leaf0 t613_leaf1 t614_leaf0
 t615_leaf0 t616_leaf0 t616_leaf1 t616_leaf2 t616_leaf3 t617_leaf0
 t617_leaf1 t617_leaf2 t618_leaf0 t618_leaf1 t619_leaf0 t619_leaf1
 t619_leaf2 t619_leaf3 t619_leaf4 t620_leaf0 t620_leaf1 t621_leaf0
 t621_leaf1 t621_leaf2 t621_leaf3 t622_leaf0 t622_leaf1 t622_leaf2
 t622_leaf3 t623_leaf0 t623_leaf1 t623_leaf2 t623_leaf3 t623_leaf4
 t623_leaf5 t624_leaf0 t624_leaf1 t625_leaf0 t625_leaf1 t625_leaf2
 t626_leaf0 t626_leaf1 t626_leaf2 t626_leaf3 t627_leaf0 t627_leaf1
 t628_leaf0 t629_leaf0 t630_leaf0 t631_leaf0 t632_leaf0 t632_leaf1
 t632_leaf2 t632_leaf3 t633_leaf0 t634_leaf0 t635_leaf0 t636_leaf0
 t637_leaf0 t638_leaf0 t639_leaf0 t640_leaf0 t640_leaf1 t640_leaf2
 t640_leaf3 t641_leaf0 t642_leaf0 t643_leaf0 t644_leaf0 t645_leaf0
 t646_leaf0 t646_leaf1 t646_leaf2 t647_leaf0 t647_leaf1 t647_leaf2
 t648_leaf0 t649_leaf0 t649_leaf1 t650_leaf0 t651_leaf0 t652_leaf0
 t652_leaf1 t652_leaf2 t653_leaf0 t653_leaf1 t654_leaf0 t655_leaf0
 t655_leaf1 t655_leaf2 t656_leaf0 t656_leaf1 t656_leaf2 t657_leaf0
 t657_leaf1 t658_leaf0 t659_leaf0 t659_leaf1 t659_leaf2 t659_leaf3
 t659_leaf4 t660_leaf0 t660_leaf1 t660_leaf2 t660_leaf3 t661_leaf0
 t661_leaf1 t661_leaf2 t661_leaf3 t661_leaf4 t662_leaf0 t663_leaf0
 t663_leaf1 t663_leaf2 t663_leaf3 t663_leaf4 t663_leaf5 t663_leaf6
 t664_leaf0 t664_leaf1 t664_leaf2 t665_leaf0 t665_leaf1 t665_leaf2
 t666_leaf0 t666_leaf1 t666_leaf2 t667_leaf0 t667_leaf1 t667_leaf2
 t667_leaf3 t668_leaf0 t668_leaf1 t668_leaf2 t668_leaf3 t669_leaf0
 t670_leaf0 t671_leaf0 t671_leaf1 t672_leaf0 t673_leaf0 t673_leaf1
 t673_leaf2 t673_leaf3 t673_leaf4 t674_leaf0 t674_leaf1 t675_leaf0
 t676_leaf0 t676_leaf1 t677_leaf0 t678_leaf0 t678_leaf1 t678_leaf2
 t678_leaf3 t678_leaf4 t679_leaf0 t680_leaf0 t680_leaf1 t680_leaf2
 t680_leaf3 t680_leaf4 t681_leaf0 t681_leaf1 t682_leaf0 t683_leaf0
 t683_leaf1 t683_leaf2 t683_leaf3 t684_leaf0 t684_leaf1 t684_leaf2
 t685_leaf0 t686_leaf0 t687_leaf0 t687_leaf1 t687_leaf2 t688_leaf0
 t689_leaf0 t690_leaf0 t691_leaf0 t692_leaf0 t692_leaf1 t693_leaf0
 t693_leaf1 t694_leaf0 t695_leaf0 t696_leaf0 t697_leaf0 t697_leaf1
 t698_leaf0 t698_leaf1 t699_leaf0 t700_leaf0 t701_leaf0 t702_leaf0
 t702_leaf1 t702_leaf2 t702_leaf3 t703_leaf0 t704_leaf0 t705_leaf0
 t705_leaf1 t705_leaf2 t705_leaf3 t705_leaf4 t706_leaf0 t706_leaf1
 t706_leaf2 t707_leaf0 t708_leaf0 t709_leaf0 t709_leaf1 t709_leaf2
 t710_leaf0 t710_leaf1 t710_leaf2 t710_leaf3 t711_leaf0 t712_leaf0
 t712_leaf1 t712_leaf2 t712_leaf3 t713_leaf0 t713_leaf1 t714_leaf0
 t715_leaf0 t715_leaf1 t715_leaf2 t715_leaf3 t716_leaf0 t717_leaf0
 t718_leaf0 t719_leaf0 t720_leaf0 t720_leaf1 t721_leaf0 t722_leaf0
 t723_leaf0 t724_leaf0 t725_leaf0 t726_leaf0 t726_leaf1 t726_leaf2
 t727_leaf0 t727_leaf1 t727_leaf2 t728_leaf0 t728_leaf1 t729_leaf0
 t730_leaf0 t730_leaf1 t730_leaf2 t730_leaf3 t730_leaf4 t731_leaf0
 t732_leaf0 t733_leaf0 t734_leaf0 t735_leaf0 t736_leaf0 t737_leaf0
 t738_leaf0 t739_leaf0 t739_leaf1 t740_leaf0 t741_leaf0 t741_leaf1
 t741_leaf2 t742_leaf0 t742_leaf1 t742_leaf2 t742_leaf3 t742_leaf4
 t742_leaf5 t742_leaf6 t743_leaf0 t744_leaf0 t745_leaf0 t746_leaf0
 t747_leaf0 t748_leaf0 t749_leaf0 t749_leaf1 t749_leaf2 t749_leaf3
 t750_leaf0 t751_leaf0 t752_leaf0 t753_leaf0 t754_leaf0 t755_leaf0
 t756_leaf0 t757_leaf0 t757_leaf1 t757_leaf2 t757_leaf3 t758_leaf0
 t759_leaf0 t759_leaf1 t760_leaf0 t761_leaf0 t761_leaf1 t761_leaf2
 t761_leaf3 t762_leaf0 t763_leaf0 t763_leaf1 t763_leaf2 t763_leaf3
 t763_leaf4 t764_leaf0 t764_leaf1 t764_leaf2 t764_leaf3 t765_leaf0
 t766_leaf0 t766_leaf1 t767_leaf0 t768_leaf0 t768_leaf1 t768_leaf2
 t769_leaf0 t770_leaf0 t771_leaf0 t772_leaf0 t772_leaf1 t772_leaf2
 t773_leaf0 t774_leaf0 t775_leaf0 t775_leaf1 t776_leaf0 t777_leaf0
 t777_leaf1 t778_leaf0 t778_leaf1 t778_leaf2 t778_leaf3 t779_leaf0
 t780_leaf0 t781_leaf0 t781_leaf1 t781_leaf2 t782_leaf0 t783_leaf0
 t784_leaf0 t784_leaf1 t784_leaf2 t784_leaf3 t784_leaf4 t785_leaf0
 t785_leaf1 t786_leaf0 t787_leaf0 t787_leaf1 t788_leaf0 t789_leaf0
 t790_leaf0 t791_leaf0 t792_leaf0 t793_leaf0 t793_leaf1 t793_leaf2
 t793_leaf3 t794_leaf0 t795_leaf0 t795_leaf1 t795_leaf2 t795_leaf3
 t795_leaf4 t796_leaf0 t797_leaf0 t798_leaf0 t799_leaf0 t800_leaf0
 t801_leaf0 t802_leaf0 t803_leaf0 t804_leaf0 t805_leaf0 t805_leaf1
 t806_leaf0 t806_leaf1 t807_leaf0 t808_leaf0 t809_leaf0 t810_leaf0
 t811_leaf0 t812_leaf0 t813_leaf0 t814_leaf0 t814_leaf1 t814_leaf2
 t814_leaf3 t814_leaf4 t815_leaf0 t816_leaf0 t817_leaf0 t817_leaf1
 t817_leaf2 t818_leaf0 t819_leaf0 t820_leaf0 t821_leaf0 t822_leaf0
 t823_leaf0 t824_leaf0 t825_leaf0 t825_leaf1 t826_leaf0 t827_leaf0
 t827_leaf1 t827_leaf2 t827_leaf3 t828_leaf0 t829_leaf0 t829_leaf1
 t829_leaf2 t829_leaf3 t829_leaf4 t830_leaf0 t831_leaf0 t831_leaf1
 t832_leaf0 t832_leaf1 t833_leaf0 t834_leaf0 t835_leaf0 t835_leaf1
 t835_leaf2 t836_leaf0 t837_leaf0 t838_leaf0 t839_leaf0 t839_leaf1
 t840_leaf0 t841_leaf0 t841_leaf1 t842_leaf0 t842_leaf1 t843_leaf0
 t844_leaf0 t844_leaf1 t845_leaf0 t845_leaf1 t845_leaf2 t846_leaf0
 t847_leaf0 t848_leaf0 t848_leaf1 t848_leaf2 t849_leaf0 t849_leaf1
 t849_leaf2 t850_leaf0 t851_leaf0 t852_leaf0 t853_leaf0 t854_leaf0
 t854_leaf1 t855_leaf0 t856_leaf0 t856_leaf1 t857_leaf0 t858_leaf0
 t858_leaf1 t858_leaf2 t858_leaf3 t859_leaf0 t860_leaf0 t860_leaf1
 t860_leaf2 t861_leaf0 t862_leaf0 t863_leaf0 t864_leaf0 t865_leaf0
 t865_leaf1 t865_leaf2 t866_leaf0 t867_leaf0 t868_leaf0 t869_leaf0
 t870_leaf0 t871_leaf0 t872_leaf0 t873_leaf0 t874_leaf0 t875_leaf0
 t875_leaf1 t876_leaf0 t876_leaf1 t876_leaf2 t876_leaf3 t877_leaf0
 t877_leaf1 t878_leaf0 t879_leaf0 t880_leaf0 t880_leaf1 t880_leaf2
 t880_leaf3 t881_leaf0 t881_leaf1 t881_leaf2 t881_leaf3 t881_leaf4
 t882_leaf0 t883_leaf0 t883_leaf1 t884_leaf0 t885_leaf0 t886_leaf0
 t886_leaf1 t886_leaf2 t886_leaf3 t886_leaf4 t886_leaf5 t887_leaf0
 t888_leaf0 t888_leaf1 t889_leaf0 t889_leaf1 t889_leaf2 t889_leaf3
 t889_leaf4 t890_leaf0 t891_leaf0 t892_leaf0 t893_leaf0 t894_leaf0
 t894_leaf1 t895_leaf0 t896_leaf0 t896_leaf1 t897_leaf0 t898_leaf0
 t899_leaf0 t900_leaf0 t901_leaf0 t902_leaf0 t903_leaf0 t903_leaf1
 t903_leaf2 t904_leaf0 t904_leaf1 t905_leaf0 t906_leaf0 t906_leaf1
 t906_leaf2 t907_leaf0 t908_leaf0 t909_leaf0 t909_leaf1 t910_leaf0
 t911_leaf0 t912_leaf0 t913_leaf0 z_change
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
