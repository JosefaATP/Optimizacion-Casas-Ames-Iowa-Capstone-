\ Model remodel_embed
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xacc41a9af964a74b
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
   + 0 x_z20_PoolArea + 0 x_z30_PoolArea + 0 x_b1_is_No_aplica
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
   - 301951.4375 Constant
Subject To
 OverallQual_from_upgrades: - 0.0625 x_Kitchen_Qual - 0.0625 x_Exter_Qual
   - 0.0625 x_Exter_Cond - 0.0625 x_Heating_QC - 0.0625 x_Fireplace_Qu
   - 0.0625 x_Bsmt_Cond - 0.0625 x_Garage_Qual - 0.0625 x_Garage_Cond
   + Overall_Qual_calc = 4.9375
 Kitchen_Qual_upgrade_only: x_Kitchen_Qual >= 2
 Exter_Qual_upgrade_only: x_Exter_Qual >= 2
 Exter_Cond_upgrade_only: x_Exter_Cond >= 2
 Heating_QC_upgrade_only: x_Heating_QC >= 1
 Fireplace_Qu_upgrade_only: x_Fireplace_Qu >= 3
 Bsmt_Cond_upgrade_only: x_Bsmt_Cond >= 3
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
 MVT_pick_one: x_mvt_is_BrkCmn + x_mvt_is_BrkFace + x_mvt_is_CBlock
   + x_mvt_is_No_aplica + x_mvt_is_Stone = 1
 MVT_forbid_noaplica: x_mvt_is_No_aplica = 0
 MVT_forbid_cheaper_BrkCmn: x_mvt_is_BrkCmn = 0
 MVT_forbid_cheaper_BrkFace: x_mvt_is_BrkFace = 0
 MVT_forbid_cheaper_CBlock: x_mvt_is_CBlock = 0
 MVT_area_no_decrease: x_Mas_Vnr_Area >= 112
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
 UpgGa_off_for_Fin: x_UpgGarage = 0
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
   - 70 x_AddBed - 52.8 x_z10_GarageArea - 105.6 x_z20_GarageArea
   - 158.4 x_z30_GarageArea - 21 x_z10_WoodDeckSF - 42 x_z20_WoodDeckSF
   - 63 x_z30_WoodDeckSF - 6.2 x_z10_OpenPorchSF - 12.4 x_z20_OpenPorchSF
   - 18.6 x_z30_OpenPorchSF >= -29314
 AMPL_link_GarageArea: x_Garage_Area - 52.8 x_z10_GarageArea
   - 105.6 x_z20_GarageArea - 158.4 x_z30_GarageArea = 528
 AMPL_link_WoodDeckSF: x_Wood_Deck_SF - 21 x_z10_WoodDeckSF
   - 42 x_z20_WoodDeckSF - 63 x_z30_WoodDeckSF = 210
 AMPL_link_OpenPorchSF: x_Open_Porch_SF - 6.2 x_z10_OpenPorchSF
   - 12.4 x_z20_OpenPorchSF - 18.6 x_z30_OpenPorchSF = 62
 AMPL_link_EnclosedPorch: x_Enclosed_Porch = 0
 AMPL_link_3SsnPorch: x_3Ssn_Porch = 0
 AMPL_link_ScreenPorch: x_Screen_Porch = 0
 AMPL_link_PoolArea: x_Pool_Area = 0
 AMPL_link_1stflr: x_1st_Flr_SF - 40 x_AddFull - 20 x_AddHalf
   - 75 x_AddKitch - 70 x_AddBed = 1656
 AMPL_1stflr_upgrade: x_1st_Flr_SF - 40 x_Add1stFlr >= 1656
 COUNT_fullbath: x_Full_Bath - x_AddFull = 1
 COUNT_halfbath: - x_AddHalf + x_Half_Bath = 0
 COUNT_bedroom: x_Bedroom_AbvGr - x_AddBed = 3
 COUNT_kitchen: - x_AddKitch + x_Kitchen_AbvGr = 1
 COUNT_totrooms: x_TotRms_AbvGrd - x_AddKitch - x_AddBed = 7
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
 HEAT_paths_exclusive: x_heat_upg_type + x_heat_upg_qc <= 1
 HEAT_qc_upgrade_only: x_Heating_QC >= 1
 R70: - x_heat_upg_type + HEAT_any_rebuild >= 0
 R71: - x_heat_upg_qc + HEAT_any_rebuild >= 0
 R72: - x_heat_upg_type - x_heat_upg_qc + HEAT_any_rebuild <= 0
 HEAT_qc_min_if_any_rebuild: x_Heating_QC - 2 HEAT_any_rebuild >= 0
 HEAT_qc_onehot: heat_qc_is_Po + heat_qc_is_Fa + heat_qc_is_TA
   + heat_qc_is_Gd + heat_qc_is_Ex = 1
 HEAT_qc_link: x_Heating_QC - heat_qc_is_Fa - 2 heat_qc_is_TA
   - 3 heat_qc_is_Gd - 4 heat_qc_is_Ex = 0
 FQ_pick_one: fireplace_is_Po + fireplace_is_Fa + fireplace_is_TA
   + fireplace_is_Gd + fireplace_is_Ex + fireplace_is_No_aplica = 1
 FQ_level_match: x_Fireplace_Qu - fireplace_is_Fa - 2 fireplace_is_TA
   - 3 fireplace_is_Gd - 4 fireplace_is_Ex + fireplace_is_No_aplica = 0
 BSMT_all_or_nothing: - 441 bsmt_finish_all + bsmt_to_fin1 + bsmt_to_fin2
   = 0
 BSMT_link_fin1: x_BsmtFin_SF_1 - bsmt_to_fin1 = 639
 BSMT_link_fin2: x_BsmtFin_SF_2 - bsmt_to_fin2 = 0
 BSMT_link_unf: x_Bsmt_Unf_SF + 441 bsmt_finish_all = 441
 BSMT_conservation_sum: x_BsmtFin_SF_1 + x_BsmtFin_SF_2 + x_Bsmt_Unf_SF
   = 1080
 BSMTCOND_onehot: bsmtcond_is_Po + bsmtcond_is_Fa + bsmtcond_is_TA
   + bsmtcond_is_Gd + bsmtcond_is_Ex = 1
 BSMTCOND_link_int: x_Bsmt_Cond - bsmtcond_is_Fa - 2 bsmtcond_is_TA
   - 3 bsmtcond_is_Gd - 4 bsmtcond_is_Ex = 0
 UTIL_onehot: util_ELO + util_NoSeWa + util_NoSewr + util_AllPub = 1
 UTIL_link_int: x_Utilities - util_NoSeWa - 2 util_NoSewr - 3 util_AllPub
   = 0
 R2_grliv_le_lot: x_Gr_Liv_Area <= 31770
 R4_baths_le_bedrooms: - x_Bedroom_AbvGr + x_Full_Bath + x_Half_Bath <= 0
 R5_min_fullbath: x_Full_Bath >= 1
 R5_min_bedrooms: x_Bedroom_AbvGr >= 1
 R5_min_kitchen: x_Kitchen_AbvGr >= 1
 R7_gr_liv_equality: x_Gr_Liv_Area - x_1st_Flr_SF = 0
 R8_rooms_balance: - x_Bedroom_AbvGr + x_TotRms_AbvGrd - x_Kitchen_AbvGr
   - R8_other_rooms = 0
 T0_onehot: t0_leaf0 + t0_leaf1 + t0_leaf2 + t0_leaf3 + t0_leaf4 + t0_leaf5
   + t0_leaf6 + t0_leaf7 = 1
 T0_L_f5_k0: const_Year_Built + 1e+06 t0_leaf0 <= 1.001985999999e+06
 T0_L_f29_k0: const_Fireplaces + 1e+06 t0_leaf0 <= 1.000000999999e+06
 T0_L_f6_k0: const_Year_Remod_Add + 1e+06 t0_leaf0 <= 1.001953999999e+06
 T0_L_f5_k1: const_Year_Built + 1e+06 t0_leaf1 <= 1.001985999999e+06
 T0_L_f29_k1: const_Fireplaces + 1e+06 t0_leaf1 <= 1.000000999999e+06
 T0_R_f6_k1: const_Year_Remod_Add - 1e+06 t0_leaf1 >= -998046
 T0_L_f5_k2: const_Year_Built + 1e+06 t0_leaf2 <= 1.001985999999e+06
 T0_R_f29_k2: const_Fireplaces - 1e+06 t0_leaf2 >= -999999
 T0_L_f20_k2: x_Gr_Liv_Area + 1e+06 t0_leaf2 <= 1.001739999999e+06
 T0_L_f5_k3: const_Year_Built + 1e+06 t0_leaf3 <= 1.001985999999e+06
 T0_R_f29_k3: const_Fireplaces - 1e+06 t0_leaf3 >= -999999
 T0_R_f20_k3: x_Gr_Liv_Area - 1e+06 t0_leaf3 >= -998260
 T0_R_f5_k4: const_Year_Built - 1e+06 t0_leaf4 >= -998014
 T0_L_f33_k4: x_Garage_Area + 1e+06 t0_leaf4 <= 1.000649999999e+06
 T0_L_f20_k4: x_Gr_Liv_Area + 1e+06 t0_leaf4 <= 1.001483999999e+06
 T0_R_f5_k5: const_Year_Built - 1e+06 t0_leaf5 >= -998014
 T0_L_f33_k5: x_Garage_Area + 1e+06 t0_leaf5 <= 1.000649999999e+06
 T0_R_f20_k5: x_Gr_Liv_Area - 1e+06 t0_leaf5 >= -998516
 T0_R_f5_k6: const_Year_Built - 1e+06 t0_leaf6 >= -998014
 T0_R_f33_k6: x_Garage_Area - 1e+06 t0_leaf6 >= -999350
 T0_L_f20_k6: x_Gr_Liv_Area + 1e+06 t0_leaf6 <= 1.001660999999e+06
 T0_R_f5_k7: const_Year_Built - 1e+06 t0_leaf7 >= -998014
 T0_R_f33_k7: x_Garage_Area - 1e+06 t0_leaf7 >= -999350
 T0_R_f20_k7: x_Gr_Liv_Area - 1e+06 t0_leaf7 >= -998339
 T1_onehot: t1_leaf0 + t1_leaf1 + t1_leaf2 + t1_leaf3 + t1_leaf4 + t1_leaf5
   + t1_leaf6 + t1_leaf7 = 1
 T1_L_f3_k0: Overall_Qual_calc + 1e+06 t1_leaf0 <= 1.000006999999e+06
 T1_L_f3_k0: Overall_Qual_calc + 1e+06 t1_leaf0 <= 1.000004999999e+06
 T1_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t1_leaf0 <= 1.000863999999e+06
 T1_L_f3_k1: Overall_Qual_calc + 1e+06 t1_leaf1 <= 1.000006999999e+06
 T1_L_f3_k1: Overall_Qual_calc + 1e+06 t1_leaf1 <= 1.000004999999e+06
 T1_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t1_leaf1 >= -999136
 T1_L_f3_k2: Overall_Qual_calc + 1e+06 t1_leaf2 <= 1.000006999999e+06
 T1_R_f3_k2: Overall_Qual_calc - 1e+06 t1_leaf2 >= -999995
 T1_L_f32_k2: const_Garage_Cars + 1e+06 t1_leaf2 <= 1.000001999999e+06
 T1_L_f3_k3: Overall_Qual_calc + 1e+06 t1_leaf3 <= 1.000006999999e+06
 T1_R_f3_k3: Overall_Qual_calc - 1e+06 t1_leaf3 >= -999995
 T1_R_f32_k3: const_Garage_Cars - 1e+06 t1_leaf3 >= -999998
 T1_R_f3_k4: Overall_Qual_calc - 1e+06 t1_leaf4 >= -999993
 T1_L_f3_k4: Overall_Qual_calc + 1e+06 t1_leaf4 <= 1.000007999999e+06
 T1_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t1_leaf4 <= 1.000844999999e+06
 T1_R_f3_k5: Overall_Qual_calc - 1e+06 t1_leaf5 >= -999993
 T1_L_f3_k5: Overall_Qual_calc + 1e+06 t1_leaf5 <= 1.000007999999e+06
 T1_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t1_leaf5 >= -999155
 T1_R_f3_k6: Overall_Qual_calc - 1e+06 t1_leaf6 >= -999993
 T1_R_f3_k6: Overall_Qual_calc - 1e+06 t1_leaf6 >= -999992
 T1_L_f20_k6: x_Gr_Liv_Area + 1e+06 t1_leaf6 <= 1.001865999999e+06
 T1_R_f3_k7: Overall_Qual_calc - 1e+06 t1_leaf7 >= -999993
 T1_R_f3_k7: Overall_Qual_calc - 1e+06 t1_leaf7 >= -999992
 T1_R_f20_k7: x_Gr_Liv_Area - 1e+06 t1_leaf7 >= -998134
 T2_onehot: t2_leaf0 + t2_leaf1 + t2_leaf2 + t2_leaf3 + t2_leaf4 + t2_leaf5
   + t2_leaf6 + t2_leaf7 = 1
 T2_L_f8_k0: x_Exter_Qual + 1e+06 t2_leaf0 <= 1.000002999999e+06
 T2_L_f20_k0: x_Gr_Liv_Area + 1e+06 t2_leaf0 <= 1.001307999999e+06
 T2_L_f5_k0: const_Year_Built + 1e+06 t2_leaf0 <= 1.001952999999e+06
 T2_L_f8_k1: x_Exter_Qual + 1e+06 t2_leaf1 <= 1.000002999999e+06
 T2_L_f20_k1: x_Gr_Liv_Area + 1e+06 t2_leaf1 <= 1.001307999999e+06
 T2_R_f5_k1: const_Year_Built - 1e+06 t2_leaf1 >= -998047
 T2_L_f8_k2: x_Exter_Qual + 1e+06 t2_leaf2 <= 1.000002999999e+06
 T2_R_f20_k2: x_Gr_Liv_Area - 1e+06 t2_leaf2 >= -998692
 T2_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t2_leaf2 <= 1.000389999999e+06
 T2_L_f8_k3: x_Exter_Qual + 1e+06 t2_leaf3 <= 1.000002999999e+06
 T2_R_f20_k3: x_Gr_Liv_Area - 1e+06 t2_leaf3 >= -998692
 T2_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t2_leaf3 >= -999610
 T2_R_f8_k4: x_Exter_Qual - 1e+06 t2_leaf4 >= -999997
 T2_L_f20_k4: x_Gr_Liv_Area + 1e+06 t2_leaf4 <= 1.001626999999e+06
 T2_L_f17_k4: x_1st_Flr_SF + 1e+06 t2_leaf4 <= 1.001268999999e+06
 T2_R_f8_k5: x_Exter_Qual - 1e+06 t2_leaf5 >= -999997
 T2_L_f20_k5: x_Gr_Liv_Area + 1e+06 t2_leaf5 <= 1.001626999999e+06
 T2_R_f17_k5: x_1st_Flr_SF - 1e+06 t2_leaf5 >= -998731
 T2_R_f8_k6: x_Exter_Qual - 1e+06 t2_leaf6 >= -999997
 T2_R_f20_k6: x_Gr_Liv_Area - 1e+06 t2_leaf6 >= -998373
 T2_L_f17_k6: x_1st_Flr_SF + 1e+06 t2_leaf6 <= 1.001084999999e+06
 T2_R_f8_k7: x_Exter_Qual - 1e+06 t2_leaf7 >= -999997
 T2_R_f20_k7: x_Gr_Liv_Area - 1e+06 t2_leaf7 >= -998373
 T2_R_f17_k7: x_1st_Flr_SF - 1e+06 t2_leaf7 >= -998915
 T3_onehot: t3_leaf0 + t3_leaf1 + t3_leaf2 + t3_leaf3 + t3_leaf4 + t3_leaf5
   + t3_leaf6 + t3_leaf7 = 1
 T3_L_f3_k0: Overall_Qual_calc + 1e+06 t3_leaf0 <= 1.000006999999e+06
 T3_L_f3_k0: Overall_Qual_calc + 1e+06 t3_leaf0 <= 1.000004999999e+06
 T3_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t3_leaf0 <= 1.000799999999e+06
 T3_L_f3_k1: Overall_Qual_calc + 1e+06 t3_leaf1 <= 1.000006999999e+06
 T3_L_f3_k1: Overall_Qual_calc + 1e+06 t3_leaf1 <= 1.000004999999e+06
 T3_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t3_leaf1 >= -999200
 T3_L_f3_k2: Overall_Qual_calc + 1e+06 t3_leaf2 <= 1.000006999999e+06
 T3_R_f3_k2: Overall_Qual_calc - 1e+06 t3_leaf2 >= -999995
 T3_L_f20_k2: x_Gr_Liv_Area + 1e+06 t3_leaf2 <= 1.001361999999e+06
 T3_L_f3_k3: Overall_Qual_calc + 1e+06 t3_leaf3 <= 1.000006999999e+06
 T3_R_f3_k3: Overall_Qual_calc - 1e+06 t3_leaf3 >= -999995
 T3_R_f20_k3: x_Gr_Liv_Area - 1e+06 t3_leaf3 >= -998638
 T3_R_f3_k4: Overall_Qual_calc - 1e+06 t3_leaf4 >= -999993
 T3_L_f3_k4: Overall_Qual_calc + 1e+06 t3_leaf4 <= 1.000007999999e+06
 T3_L_f20_k4: x_Gr_Liv_Area + 1e+06 t3_leaf4 <= 1.001477999999e+06
 T3_R_f3_k5: Overall_Qual_calc - 1e+06 t3_leaf5 >= -999993
 T3_L_f3_k5: Overall_Qual_calc + 1e+06 t3_leaf5 <= 1.000007999999e+06
 T3_R_f20_k5: x_Gr_Liv_Area - 1e+06 t3_leaf5 >= -998522
 T3_R_f3_k6: Overall_Qual_calc - 1e+06 t3_leaf6 >= -999993
 T3_R_f3_k6: Overall_Qual_calc - 1e+06 t3_leaf6 >= -999992
 T3_L_f3_k6: Overall_Qual_calc + 1e+06 t3_leaf6 <= 1.000008999999e+06
 T3_R_f3_k7: Overall_Qual_calc - 1e+06 t3_leaf7 >= -999993
 T3_R_f3_k7: Overall_Qual_calc - 1e+06 t3_leaf7 >= -999992
 T3_R_f3_k7: Overall_Qual_calc - 1e+06 t3_leaf7 >= -999991
 T4_onehot: t4_leaf0 + t4_leaf1 + t4_leaf2 + t4_leaf3 + t4_leaf4 + t4_leaf5
   + t4_leaf6 + t4_leaf7 = 1
 T4_L_f8_k0: x_Exter_Qual + 1e+06 t4_leaf0 <= 1.000002999999e+06
 T4_L_f20_k0: x_Gr_Liv_Area + 1e+06 t4_leaf0 <= 1.001095999999e+06
 T4_L_f231_k0: central_air_no + 1e+06 t4_leaf0 <= 1.000000999999e+06
 T4_L_f8_k1: x_Exter_Qual + 1e+06 t4_leaf1 <= 1.000002999999e+06
 T4_L_f20_k1: x_Gr_Liv_Area + 1e+06 t4_leaf1 <= 1.001095999999e+06
 T4_R_f231_k1: central_air_no - 1e+06 t4_leaf1 >= -999999
 T4_L_f8_k2: x_Exter_Qual + 1e+06 t4_leaf2 <= 1.000002999999e+06
 T4_R_f20_k2: x_Gr_Liv_Area - 1e+06 t4_leaf2 >= -998904
 T4_L_f29_k2: const_Fireplaces + 1e+06 t4_leaf2 <= 1.000000999999e+06
 T4_L_f8_k3: x_Exter_Qual + 1e+06 t4_leaf3 <= 1.000002999999e+06
 T4_R_f20_k3: x_Gr_Liv_Area - 1e+06 t4_leaf3 >= -998904
 T4_R_f29_k3: const_Fireplaces - 1e+06 t4_leaf3 >= -999999
 T4_R_f8_k4: x_Exter_Qual - 1e+06 t4_leaf4 >= -999997
 T4_L_f32_k4: const_Garage_Cars + 1e+06 t4_leaf4 <= 1.000002999999e+06
 T4_L_f17_k4: x_1st_Flr_SF + 1e+06 t4_leaf4 <= 1.001413999999e+06
 T4_R_f8_k5: x_Exter_Qual - 1e+06 t4_leaf5 >= -999997
 T4_L_f32_k5: const_Garage_Cars + 1e+06 t4_leaf5 <= 1.000002999999e+06
 T4_R_f17_k5: x_1st_Flr_SF - 1e+06 t4_leaf5 >= -998586
 T4_R_f8_k6: x_Exter_Qual - 1e+06 t4_leaf6 >= -999997
 T4_R_f32_k6: const_Garage_Cars - 1e+06 t4_leaf6 >= -999997
 T4_L_f20_k6: x_Gr_Liv_Area + 1e+06 t4_leaf6 <= 1.001715999999e+06
 T4_R_f8_k7: x_Exter_Qual - 1e+06 t4_leaf7 >= -999997
 T4_R_f32_k7: const_Garage_Cars - 1e+06 t4_leaf7 >= -999997
 T4_R_f20_k7: x_Gr_Liv_Area - 1e+06 t4_leaf7 >= -998284
 T5_onehot: t5_leaf0 + t5_leaf1 + t5_leaf2 + t5_leaf3 + t5_leaf4 + t5_leaf5
   + t5_leaf6 + t5_leaf7 = 1
 T5_L_f3_k0: Overall_Qual_calc + 1e+06 t5_leaf0 <= 1.000006999999e+06
 T5_L_f3_k0: Overall_Qual_calc + 1e+06 t5_leaf0 <= 1.000004999999e+06
 T5_L_f17_k0: x_1st_Flr_SF + 1e+06 t5_leaf0 <= 1.001154999999e+06
 T5_L_f3_k1: Overall_Qual_calc + 1e+06 t5_leaf1 <= 1.000006999999e+06
 T5_L_f3_k1: Overall_Qual_calc + 1e+06 t5_leaf1 <= 1.000004999999e+06
 T5_R_f17_k1: x_1st_Flr_SF - 1e+06 t5_leaf1 >= -998845
 T5_L_f3_k2: Overall_Qual_calc + 1e+06 t5_leaf2 <= 1.000006999999e+06
 T5_R_f3_k2: Overall_Qual_calc - 1e+06 t5_leaf2 >= -999995
 T5_L_f32_k2: const_Garage_Cars + 1e+06 t5_leaf2 <= 1.000001999999e+06
 T5_L_f3_k3: Overall_Qual_calc + 1e+06 t5_leaf3 <= 1.000006999999e+06
 T5_R_f3_k3: Overall_Qual_calc - 1e+06 t5_leaf3 >= -999995
 T5_R_f32_k3: const_Garage_Cars - 1e+06 t5_leaf3 >= -999998
 T5_R_f3_k4: Overall_Qual_calc - 1e+06 t5_leaf4 >= -999993
 T5_L_f3_k4: Overall_Qual_calc + 1e+06 t5_leaf4 <= 1.000007999999e+06
 T5_L_f17_k4: x_1st_Flr_SF + 1e+06 t5_leaf4 <= 1.000931999999e+06
 T5_R_f3_k5: Overall_Qual_calc - 1e+06 t5_leaf5 >= -999993
 T5_L_f3_k5: Overall_Qual_calc + 1e+06 t5_leaf5 <= 1.000007999999e+06
 T5_R_f17_k5: x_1st_Flr_SF - 1e+06 t5_leaf5 >= -999068
 T5_R_f3_k6: Overall_Qual_calc - 1e+06 t5_leaf6 >= -999993
 T5_R_f3_k6: Overall_Qual_calc - 1e+06 t5_leaf6 >= -999992
 T5_L_f17_k6: x_1st_Flr_SF + 1e+06 t5_leaf6 <= 1.001765999999e+06
 T5_R_f3_k7: Overall_Qual_calc - 1e+06 t5_leaf7 >= -999993
 T5_R_f3_k7: Overall_Qual_calc - 1e+06 t5_leaf7 >= -999992
 T5_R_f17_k7: x_1st_Flr_SF - 1e+06 t5_leaf7 >= -998234
 T6_onehot: t6_leaf0 + t6_leaf1 + t6_leaf2 + t6_leaf3 + t6_leaf4 + t6_leaf5
   + t6_leaf6 + t6_leaf7 = 1
 T6_L_f10_k0: const_Bsmt_Qual + 1e+06 t6_leaf0 <= 1.000002999999e+06
 T6_L_f35_k0: x_Garage_Cond + 1e+06 t6_leaf0 <= 1.000001999999e+06
 T6_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t6_leaf0 <= 1.000006999999e+06
 T6_L_f10_k1: const_Bsmt_Qual + 1e+06 t6_leaf1 <= 1.000002999999e+06
 T6_L_f35_k1: x_Garage_Cond + 1e+06 t6_leaf1 <= 1.000001999999e+06
 T6_R_f28_k1: x_TotRms_AbvGrd - 1e+06 t6_leaf1 >= -999993
 T6_L_f10_k2: const_Bsmt_Qual + 1e+06 t6_leaf2 <= 1.000002999999e+06
 T6_R_f35_k2: x_Garage_Cond - 1e+06 t6_leaf2 >= -999998
 T6_L_f29_k2: const_Fireplaces + 1e+06 t6_leaf2 <= 1.000000999999e+06
 T6_L_f10_k3: const_Bsmt_Qual + 1e+06 t6_leaf3 <= 1.000002999999e+06
 T6_R_f35_k3: x_Garage_Cond - 1e+06 t6_leaf3 >= -999998
 T6_R_f29_k3: const_Fireplaces - 1e+06 t6_leaf3 >= -999999
 T6_R_f10_k4: const_Bsmt_Qual - 1e+06 t6_leaf4 >= -999997
 T6_L_f32_k4: const_Garage_Cars + 1e+06 t6_leaf4 <= 1.000002999999e+06
 T6_L_f29_k4: const_Fireplaces + 1e+06 t6_leaf4 <= 1.000000999999e+06
 T6_R_f10_k5: const_Bsmt_Qual - 1e+06 t6_leaf5 >= -999997
 T6_L_f32_k5: const_Garage_Cars + 1e+06 t6_leaf5 <= 1.000002999999e+06
 T6_R_f29_k5: const_Fireplaces - 1e+06 t6_leaf5 >= -999999
 T6_R_f10_k6: const_Bsmt_Qual - 1e+06 t6_leaf6 >= -999997
 T6_R_f32_k6: const_Garage_Cars - 1e+06 t6_leaf6 >= -999997
 T6_L_f27_k6: x_Kitchen_Qual + 1e+06 t6_leaf6 <= 1.000003999999e+06
 T6_R_f10_k7: const_Bsmt_Qual - 1e+06 t6_leaf7 >= -999997
 T6_R_f32_k7: const_Garage_Cars - 1e+06 t6_leaf7 >= -999997
 T6_R_f27_k7: x_Kitchen_Qual - 1e+06 t6_leaf7 >= -999996
 T7_onehot: t7_leaf0 + t7_leaf1 + t7_leaf2 + t7_leaf3 + t7_leaf4 + t7_leaf5
   + t7_leaf6 + t7_leaf7 = 1
 T7_L_f3_k0: Overall_Qual_calc + 1e+06 t7_leaf0 <= 1.000006999999e+06
 T7_L_f3_k0: Overall_Qual_calc + 1e+06 t7_leaf0 <= 1.000004999999e+06
 T7_L_f20_k0: x_Gr_Liv_Area + 1e+06 t7_leaf0 <= 1.000863999999e+06
 T7_L_f3_k1: Overall_Qual_calc + 1e+06 t7_leaf1 <= 1.000006999999e+06
 T7_L_f3_k1: Overall_Qual_calc + 1e+06 t7_leaf1 <= 1.000004999999e+06
 T7_R_f20_k1: x_Gr_Liv_Area - 1e+06 t7_leaf1 >= -999136
 T7_L_f3_k2: Overall_Qual_calc + 1e+06 t7_leaf2 <= 1.000006999999e+06
 T7_R_f3_k2: Overall_Qual_calc - 1e+06 t7_leaf2 >= -999995
 T7_L_f20_k2: x_Gr_Liv_Area + 1e+06 t7_leaf2 <= 1.001483999999e+06
 T7_L_f3_k3: Overall_Qual_calc + 1e+06 t7_leaf3 <= 1.000006999999e+06
 T7_R_f3_k3: Overall_Qual_calc - 1e+06 t7_leaf3 >= -999995
 T7_R_f20_k3: x_Gr_Liv_Area - 1e+06 t7_leaf3 >= -998516
 T7_R_f3_k4: Overall_Qual_calc - 1e+06 t7_leaf4 >= -999993
 T7_L_f17_k4: x_1st_Flr_SF + 1e+06 t7_leaf4 <= 1.001507999999e+06
 T7_L_f20_k4: x_Gr_Liv_Area + 1e+06 t7_leaf4 <= 1.001865999999e+06
 T7_R_f3_k5: Overall_Qual_calc - 1e+06 t7_leaf5 >= -999993
 T7_L_f17_k5: x_1st_Flr_SF + 1e+06 t7_leaf5 <= 1.001507999999e+06
 T7_R_f20_k5: x_Gr_Liv_Area - 1e+06 t7_leaf5 >= -998134
 T7_R_f3_k6: Overall_Qual_calc - 1e+06 t7_leaf6 >= -999993
 T7_R_f17_k6: x_1st_Flr_SF - 1e+06 t7_leaf6 >= -998492
 T7_L_f20_k6: x_Gr_Liv_Area + 1e+06 t7_leaf6 <= 1.001719999999e+06
 T7_R_f3_k7: Overall_Qual_calc - 1e+06 t7_leaf7 >= -999993
 T7_R_f17_k7: x_1st_Flr_SF - 1e+06 t7_leaf7 >= -998492
 T7_R_f20_k7: x_Gr_Liv_Area - 1e+06 t7_leaf7 >= -998280
 T8_onehot: t8_leaf0 + t8_leaf1 + t8_leaf2 + t8_leaf3 + t8_leaf4 + t8_leaf5
   + t8_leaf6 + t8_leaf7 = 1
 T8_L_f10_k0: const_Bsmt_Qual + 1e+06 t8_leaf0 <= 1.000002999999e+06
 T8_L_f29_k0: const_Fireplaces + 1e+06 t8_leaf0 <= 1.000000999999e+06
 T8_L_f231_k0: central_air_no + 1e+06 t8_leaf0 <= 1.000000999999e+06
 T8_L_f10_k1: const_Bsmt_Qual + 1e+06 t8_leaf1 <= 1.000002999999e+06
 T8_L_f29_k1: const_Fireplaces + 1e+06 t8_leaf1 <= 1.000000999999e+06
 T8_R_f231_k1: central_air_no - 1e+06 t8_leaf1 >= -999999
 T8_L_f10_k2: const_Bsmt_Qual + 1e+06 t8_leaf2 <= 1.000002999999e+06
 T8_R_f29_k2: const_Fireplaces - 1e+06 t8_leaf2 >= -999999
 T8_L_f28_k2: x_TotRms_AbvGrd + 1e+06 t8_leaf2 <= 1.000007999999e+06
 T8_L_f10_k3: const_Bsmt_Qual + 1e+06 t8_leaf3 <= 1.000002999999e+06
 T8_R_f29_k3: const_Fireplaces - 1e+06 t8_leaf3 >= -999999
 T8_R_f28_k3: x_TotRms_AbvGrd - 1e+06 t8_leaf3 >= -999992
 T8_R_f10_k4: const_Bsmt_Qual - 1e+06 t8_leaf4 >= -999997
 T8_L_f32_k4: const_Garage_Cars + 1e+06 t8_leaf4 <= 1.000002999999e+06
 T8_L_f17_k4: x_1st_Flr_SF + 1e+06 t8_leaf4 <= 1.001235999999e+06
 T8_R_f10_k5: const_Bsmt_Qual - 1e+06 t8_leaf5 >= -999997
 T8_L_f32_k5: const_Garage_Cars + 1e+06 t8_leaf5 <= 1.000002999999e+06
 T8_R_f17_k5: x_1st_Flr_SF - 1e+06 t8_leaf5 >= -998764
 T8_R_f10_k6: const_Bsmt_Qual - 1e+06 t8_leaf6 >= -999997
 T8_R_f32_k6: const_Garage_Cars - 1e+06 t8_leaf6 >= -999997
 T8_L_f17_k6: x_1st_Flr_SF + 1e+06 t8_leaf6 <= 1.001799999999e+06
 T8_R_f10_k7: const_Bsmt_Qual - 1e+06 t8_leaf7 >= -999997
 T8_R_f32_k7: const_Garage_Cars - 1e+06 t8_leaf7 >= -999997
 T8_R_f17_k7: x_1st_Flr_SF - 1e+06 t8_leaf7 >= -998200
 T9_onehot: t9_leaf0 + t9_leaf1 + t9_leaf2 + t9_leaf3 + t9_leaf4 + t9_leaf5
   + t9_leaf6 + t9_leaf7 = 1
 T9_L_f5_k0: const_Year_Built + 1e+06 t9_leaf0 <= 1.001980999999e+06
 T9_L_f17_k0: x_1st_Flr_SF + 1e+06 t9_leaf0 <= 1.001093999999e+06
 T9_L_f231_k0: central_air_no + 1e+06 t9_leaf0 <= 1.000000999999e+06
 T9_L_f5_k1: const_Year_Built + 1e+06 t9_leaf1 <= 1.001980999999e+06
 T9_L_f17_k1: x_1st_Flr_SF + 1e+06 t9_leaf1 <= 1.001093999999e+06
 T9_R_f231_k1: central_air_no - 1e+06 t9_leaf1 >= -999999
 T9_L_f5_k2: const_Year_Built + 1e+06 t9_leaf2 <= 1.001980999999e+06
 T9_R_f17_k2: x_1st_Flr_SF - 1e+06 t9_leaf2 >= -998906
 T9_L_f1_k2: const_Lot_Area + 1e+06 t9_leaf2 <= 1.012215999999e+06
 T9_L_f5_k3: const_Year_Built + 1e+06 t9_leaf3 <= 1.001980999999e+06
 T9_R_f17_k3: x_1st_Flr_SF - 1e+06 t9_leaf3 >= -998906
 T9_R_f1_k3: const_Lot_Area - 1e+06 t9_leaf3 >= -987784
 T9_R_f5_k4: const_Year_Built - 1e+06 t9_leaf4 >= -998019
 T9_L_f33_k4: x_Garage_Area + 1e+06 t9_leaf4 <= 1.000667999999e+06
 T9_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t9_leaf4 <= 1.001267999999e+06
 T9_R_f5_k5: const_Year_Built - 1e+06 t9_leaf5 >= -998019
 T9_L_f33_k5: x_Garage_Area + 1e+06 t9_leaf5 <= 1.000667999999e+06
 T9_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t9_leaf5 >= -998732
 T9_R_f5_k6: const_Year_Built - 1e+06 t9_leaf6 >= -998019
 T9_R_f33_k6: x_Garage_Area - 1e+06 t9_leaf6 >= -999332
 T9_L_f1_k6: const_Lot_Area + 1e+06 t9_leaf6 <= 1.011979999999e+06
 T9_R_f5_k7: const_Year_Built - 1e+06 t9_leaf7 >= -998019
 T9_R_f33_k7: x_Garage_Area - 1e+06 t9_leaf7 >= -999332
 T9_R_f1_k7: const_Lot_Area - 1e+06 t9_leaf7 >= -988020
 T10_onehot: t10_leaf0 + t10_leaf1 + t10_leaf2 + t10_leaf3 + t10_leaf4
   + t10_leaf5 + t10_leaf6 + t10_leaf7 = 1
 T10_L_f20_k0: x_Gr_Liv_Area + 1e+06 t10_leaf0 <= 1.001319999999e+06
 T10_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t10_leaf0 <= 1.001952999999e+06
 T10_L_f20_k0: x_Gr_Liv_Area + 1e+06 t10_leaf0 <= 1.000832999999e+06
 T10_L_f20_k1: x_Gr_Liv_Area + 1e+06 t10_leaf1 <= 1.001319999999e+06
 T10_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t10_leaf1 <= 1.001952999999e+06
 T10_R_f20_k1: x_Gr_Liv_Area - 1e+06 t10_leaf1 >= -999167
 T10_L_f20_k2: x_Gr_Liv_Area + 1e+06 t10_leaf2 <= 1.001319999999e+06
 T10_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t10_leaf2 >= -998047
 T10_L_f20_k2: x_Gr_Liv_Area + 1e+06 t10_leaf2 <= 1.001123999999e+06
 T10_L_f20_k3: x_Gr_Liv_Area + 1e+06 t10_leaf3 <= 1.001319999999e+06
 T10_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t10_leaf3 >= -998047
 T10_R_f20_k3: x_Gr_Liv_Area - 1e+06 t10_leaf3 >= -998876
 T10_R_f20_k4: x_Gr_Liv_Area - 1e+06 t10_leaf4 >= -998680
 T10_L_f8_k4: x_Exter_Qual + 1e+06 t10_leaf4 <= 1.000002999999e+06
 T10_L_f66_k4: const_MS_Zoning_RL + 1e+06 t10_leaf4 <= 1.000000999999e+06
 T10_R_f20_k5: x_Gr_Liv_Area - 1e+06 t10_leaf5 >= -998680
 T10_L_f8_k5: x_Exter_Qual + 1e+06 t10_leaf5 <= 1.000002999999e+06
 T10_R_f66_k5: const_MS_Zoning_RL - 1e+06 t10_leaf5 >= -999999
 T10_R_f20_k6: x_Gr_Liv_Area - 1e+06 t10_leaf6 >= -998680
 T10_R_f8_k6: x_Exter_Qual - 1e+06 t10_leaf6 >= -999997
 T10_L_f10_k6: const_Bsmt_Qual + 1e+06 t10_leaf6 <= 1.000003999999e+06
 T10_R_f20_k7: x_Gr_Liv_Area - 1e+06 t10_leaf7 >= -998680
 T10_R_f8_k7: x_Exter_Qual - 1e+06 t10_leaf7 >= -999997
 T10_R_f10_k7: const_Bsmt_Qual - 1e+06 t10_leaf7 >= -999996
 T11_onehot: t11_leaf0 + t11_leaf1 + t11_leaf2 + t11_leaf3 + t11_leaf4
   + t11_leaf5 + t11_leaf6 + t11_leaf7 = 1
 T11_L_f3_k0: Overall_Qual_calc + 1e+06 t11_leaf0 <= 1.000006999999e+06
 T11_L_f3_k0: Overall_Qual_calc + 1e+06 t11_leaf0 <= 1.000004999999e+06
 T11_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t11_leaf0 <= 1.000911999999e+06
 T11_L_f3_k1: Overall_Qual_calc + 1e+06 t11_leaf1 <= 1.000006999999e+06
 T11_L_f3_k1: Overall_Qual_calc + 1e+06 t11_leaf1 <= 1.000004999999e+06
 T11_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t11_leaf1 >= -999088
 T11_L_f3_k2: Overall_Qual_calc + 1e+06 t11_leaf2 <= 1.000006999999e+06
 T11_R_f3_k2: Overall_Qual_calc - 1e+06 t11_leaf2 >= -999995
 T11_L_f20_k2: x_Gr_Liv_Area + 1e+06 t11_leaf2 <= 1.001410999999e+06
 T11_L_f3_k3: Overall_Qual_calc + 1e+06 t11_leaf3 <= 1.000006999999e+06
 T11_R_f3_k3: Overall_Qual_calc - 1e+06 t11_leaf3 >= -999995
 T11_R_f20_k3: x_Gr_Liv_Area - 1e+06 t11_leaf3 >= -998589
 T11_R_f3_k4: Overall_Qual_calc - 1e+06 t11_leaf4 >= -999993
 T11_L_f3_k4: Overall_Qual_calc + 1e+06 t11_leaf4 <= 1.000007999999e+06
 T11_L_f20_k4: x_Gr_Liv_Area + 1e+06 t11_leaf4 <= 1.001725999999e+06
 T11_R_f3_k5: Overall_Qual_calc - 1e+06 t11_leaf5 >= -999993
 T11_L_f3_k5: Overall_Qual_calc + 1e+06 t11_leaf5 <= 1.000007999999e+06
 T11_R_f20_k5: x_Gr_Liv_Area - 1e+06 t11_leaf5 >= -998274
 T11_R_f3_k6: Overall_Qual_calc - 1e+06 t11_leaf6 >= -999993
 T11_R_f3_k6: Overall_Qual_calc - 1e+06 t11_leaf6 >= -999992
 T11_L_f20_k6: x_Gr_Liv_Area + 1e+06 t11_leaf6 <= 1.001669999999e+06
 T11_R_f3_k7: Overall_Qual_calc - 1e+06 t11_leaf7 >= -999993
 T11_R_f3_k7: Overall_Qual_calc - 1e+06 t11_leaf7 >= -999992
 T11_R_f20_k7: x_Gr_Liv_Area - 1e+06 t11_leaf7 >= -998330
 T12_onehot: t12_leaf0 + t12_leaf1 + t12_leaf2 + t12_leaf3 + t12_leaf4
   + t12_leaf5 + t12_leaf6 + t12_leaf7 = 1
 T12_L_f3_k0: Overall_Qual_calc + 1e+06 t12_leaf0 <= 1.000006999999e+06
 T12_L_f3_k0: Overall_Qual_calc + 1e+06 t12_leaf0 <= 1.000004999999e+06
 T12_L_f20_k0: x_Gr_Liv_Area + 1e+06 t12_leaf0 <= 1.001154999999e+06
 T12_L_f3_k1: Overall_Qual_calc + 1e+06 t12_leaf1 <= 1.000006999999e+06
 T12_L_f3_k1: Overall_Qual_calc + 1e+06 t12_leaf1 <= 1.000004999999e+06
 T12_R_f20_k1: x_Gr_Liv_Area - 1e+06 t12_leaf1 >= -998845
 T12_L_f3_k2: Overall_Qual_calc + 1e+06 t12_leaf2 <= 1.000006999999e+06
 T12_R_f3_k2: Overall_Qual_calc - 1e+06 t12_leaf2 >= -999995
 T12_L_f33_k2: x_Garage_Area + 1e+06 t12_leaf2 <= 1.000383999999e+06
 T12_L_f3_k3: Overall_Qual_calc + 1e+06 t12_leaf3 <= 1.000006999999e+06
 T12_R_f3_k3: Overall_Qual_calc - 1e+06 t12_leaf3 >= -999995
 T12_R_f33_k3: x_Garage_Area - 1e+06 t12_leaf3 >= -999616
 T12_R_f3_k4: Overall_Qual_calc - 1e+06 t12_leaf4 >= -999993
 T12_L_f10_k4: const_Bsmt_Qual + 1e+06 t12_leaf4 <= 1.000003999999e+06
 T12_L_f20_k4: x_Gr_Liv_Area + 1e+06 t12_leaf4 <= 1.002133999999e+06
 T12_R_f3_k5: Overall_Qual_calc - 1e+06 t12_leaf5 >= -999993
 T12_L_f10_k5: const_Bsmt_Qual + 1e+06 t12_leaf5 <= 1.000003999999e+06
 T12_R_f20_k5: x_Gr_Liv_Area - 1e+06 t12_leaf5 >= -997866
 T12_R_f3_k6: Overall_Qual_calc - 1e+06 t12_leaf6 >= -999993
 T12_R_f10_k6: const_Bsmt_Qual - 1e+06 t12_leaf6 >= -999996
 T12_L_f3_k6: Overall_Qual_calc + 1e+06 t12_leaf6 <= 1.000008999999e+06
 T12_R_f3_k7: Overall_Qual_calc - 1e+06 t12_leaf7 >= -999993
 T12_R_f10_k7: const_Bsmt_Qual - 1e+06 t12_leaf7 >= -999996
 T12_R_f3_k7: Overall_Qual_calc - 1e+06 t12_leaf7 >= -999991
 T13_onehot: t13_leaf0 + t13_leaf1 + t13_leaf2 + t13_leaf3 + t13_leaf4
   + t13_leaf5 + t13_leaf6 + t13_leaf7 = 1
 T13_L_f27_k0: x_Kitchen_Qual + 1e+06 t13_leaf0 <= 1.000002999999e+06
 T13_L_f32_k0: const_Garage_Cars + 1e+06 t13_leaf0 <= 1.000001999999e+06
 T13_L_f4_k0: const_Overall_Cond + 1e+06 t13_leaf0 <= 1.000003999999e+06
 T13_L_f27_k1: x_Kitchen_Qual + 1e+06 t13_leaf1 <= 1.000002999999e+06
 T13_L_f32_k1: const_Garage_Cars + 1e+06 t13_leaf1 <= 1.000001999999e+06
 T13_R_f4_k1: const_Overall_Cond - 1e+06 t13_leaf1 >= -999996
 T13_L_f27_k2: x_Kitchen_Qual + 1e+06 t13_leaf2 <= 1.000002999999e+06
 T13_R_f32_k2: const_Garage_Cars - 1e+06 t13_leaf2 >= -999998
 T13_L_f20_k2: x_Gr_Liv_Area + 1e+06 t13_leaf2 <= 1.001778999999e+06
 T13_L_f27_k3: x_Kitchen_Qual + 1e+06 t13_leaf3 <= 1.000002999999e+06
 T13_R_f32_k3: const_Garage_Cars - 1e+06 t13_leaf3 >= -999998
 T13_R_f20_k3: x_Gr_Liv_Area - 1e+06 t13_leaf3 >= -998221
 T13_R_f27_k4: x_Kitchen_Qual - 1e+06 t13_leaf4 >= -999997
 T13_L_f20_k4: x_Gr_Liv_Area + 1e+06 t13_leaf4 <= 1.001725999999e+06
 T13_L_f17_k4: x_1st_Flr_SF + 1e+06 t13_leaf4 <= 1.001281999999e+06
 T13_R_f27_k5: x_Kitchen_Qual - 1e+06 t13_leaf5 >= -999997
 T13_L_f20_k5: x_Gr_Liv_Area + 1e+06 t13_leaf5 <= 1.001725999999e+06
 T13_R_f17_k5: x_1st_Flr_SF - 1e+06 t13_leaf5 >= -998718
 T13_R_f27_k6: x_Kitchen_Qual - 1e+06 t13_leaf6 >= -999997
 T13_R_f20_k6: x_Gr_Liv_Area - 1e+06 t13_leaf6 >= -998274
 T13_L_f32_k6: const_Garage_Cars + 1e+06 t13_leaf6 <= 1.000002999999e+06
 T13_R_f27_k7: x_Kitchen_Qual - 1e+06 t13_leaf7 >= -999997
 T13_R_f20_k7: x_Gr_Liv_Area - 1e+06 t13_leaf7 >= -998274
 T13_R_f32_k7: const_Garage_Cars - 1e+06 t13_leaf7 >= -999997
 T14_onehot: t14_leaf0 + t14_leaf1 + t14_leaf2 + t14_leaf3 + t14_leaf4
   + t14_leaf5 + t14_leaf6 + t14_leaf7 = 1
 T14_L_f8_k0: x_Exter_Qual + 1e+06 t14_leaf0 <= 1.000002999999e+06
 T14_L_f232_k0: central_air_yes + 1e+06 t14_leaf0 <= 1.000000999999e+06
 T14_L_f20_k0: x_Gr_Liv_Area + 1e+06 t14_leaf0 <= 1.000875999999e+06
 T14_L_f8_k1: x_Exter_Qual + 1e+06 t14_leaf1 <= 1.000002999999e+06
 T14_L_f232_k1: central_air_yes + 1e+06 t14_leaf1 <= 1.000000999999e+06
 T14_R_f20_k1: x_Gr_Liv_Area - 1e+06 t14_leaf1 >= -999124
 T14_L_f8_k2: x_Exter_Qual + 1e+06 t14_leaf2 <= 1.000002999999e+06
 T14_R_f232_k2: central_air_yes - 1e+06 t14_leaf2 >= -999999
 T14_L_f20_k2: x_Gr_Liv_Area + 1e+06 t14_leaf2 <= 1.001415999999e+06
 T14_L_f8_k3: x_Exter_Qual + 1e+06 t14_leaf3 <= 1.000002999999e+06
 T14_R_f232_k3: central_air_yes - 1e+06 t14_leaf3 >= -999999
 T14_R_f20_k3: x_Gr_Liv_Area - 1e+06 t14_leaf3 >= -998584
 T14_R_f8_k4: x_Exter_Qual - 1e+06 t14_leaf4 >= -999997
 T14_L_f20_k4: x_Gr_Liv_Area + 1e+06 t14_leaf4 <= 1.001739999999e+06
 T14_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t14_leaf4 <= 1.001267999999e+06
 T14_R_f8_k5: x_Exter_Qual - 1e+06 t14_leaf5 >= -999997
 T14_L_f20_k5: x_Gr_Liv_Area + 1e+06 t14_leaf5 <= 1.001739999999e+06
 T14_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t14_leaf5 >= -998732
 T14_R_f8_k6: x_Exter_Qual - 1e+06 t14_leaf6 >= -999997
 T14_R_f20_k6: x_Gr_Liv_Area - 1e+06 t14_leaf6 >= -998260
 T14_L_f15_k6: const_Total_Bsmt_SF + 1e+06 t14_leaf6 <= 1.001205999999e+06
 T14_R_f8_k7: x_Exter_Qual - 1e+06 t14_leaf7 >= -999997
 T14_R_f20_k7: x_Gr_Liv_Area - 1e+06 t14_leaf7 >= -998260
 T14_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t14_leaf7 >= -998794
 T15_onehot: t15_leaf0 + t15_leaf1 + t15_leaf2 + t15_leaf3 + t15_leaf4
   + t15_leaf5 + t15_leaf6 + t15_leaf7 = 1
 T15_L_f20_k0: x_Gr_Liv_Area + 1e+06 t15_leaf0 <= 1.001410999999e+06
 T15_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t15_leaf0 <= 1.000969999999e+06
 T15_L_f231_k0: central_air_no + 1e+06 t15_leaf0 <= 1.000000999999e+06
 T15_L_f20_k1: x_Gr_Liv_Area + 1e+06 t15_leaf1 <= 1.001410999999e+06
 T15_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t15_leaf1 <= 1.000969999999e+06
 T15_R_f231_k1: central_air_no - 1e+06 t15_leaf1 >= -999999
 T15_L_f20_k2: x_Gr_Liv_Area + 1e+06 t15_leaf2 <= 1.001410999999e+06
 T15_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t15_leaf2 >= -999030
 T15_L_f247_k2: const_Garage_Type_Attchd + 1e+06 t15_leaf2
   <= 1.000000999999e+06
 T15_L_f20_k3: x_Gr_Liv_Area + 1e+06 t15_leaf3 <= 1.001410999999e+06
 T15_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t15_leaf3 >= -999030
 T15_R_f247_k3: const_Garage_Type_Attchd - 1e+06 t15_leaf3 >= -999999
 T15_R_f20_k4: x_Gr_Liv_Area - 1e+06 t15_leaf4 >= -998589
 T15_L_f10_k4: const_Bsmt_Qual + 1e+06 t15_leaf4 <= 1.000003999999e+06
 T15_L_f256_k4: const_Garage_Finish_Unf + 1e+06 t15_leaf4
   <= 1.000000999999e+06
 T15_R_f20_k5: x_Gr_Liv_Area - 1e+06 t15_leaf5 >= -998589
 T15_L_f10_k5: const_Bsmt_Qual + 1e+06 t15_leaf5 <= 1.000003999999e+06
 T15_R_f256_k5: const_Garage_Finish_Unf - 1e+06 t15_leaf5 >= -999999
 T15_R_f20_k6: x_Gr_Liv_Area - 1e+06 t15_leaf6 >= -998589
 T15_R_f10_k6: const_Bsmt_Qual - 1e+06 t15_leaf6 >= -999996
 T15_L_f20_k6: x_Gr_Liv_Area + 1e+06 t15_leaf6 <= 1.001994999999e+06
 T15_R_f20_k7: x_Gr_Liv_Area - 1e+06 t15_leaf7 >= -998589
 T15_R_f10_k7: const_Bsmt_Qual - 1e+06 t15_leaf7 >= -999996
 T15_R_f20_k7: x_Gr_Liv_Area - 1e+06 t15_leaf7 >= -998005
 T16_onehot: t16_leaf0 + t16_leaf1 + t16_leaf2 + t16_leaf3 + t16_leaf4
   + t16_leaf5 + t16_leaf6 + t16_leaf7 = 1
 T16_L_f3_k0: Overall_Qual_calc + 1e+06 t16_leaf0 <= 1.000006999999e+06
 T16_L_f3_k0: Overall_Qual_calc + 1e+06 t16_leaf0 <= 1.000004999999e+06
 T16_L_f3_k0: Overall_Qual_calc + 1e+06 t16_leaf0 <= 1.000002999999e+06
 T16_L_f3_k1: Overall_Qual_calc + 1e+06 t16_leaf1 <= 1.000006999999e+06
 T16_L_f3_k1: Overall_Qual_calc + 1e+06 t16_leaf1 <= 1.000004999999e+06
 T16_R_f3_k1: Overall_Qual_calc - 1e+06 t16_leaf1 >= -999997
 T16_L_f3_k2: Overall_Qual_calc + 1e+06 t16_leaf2 <= 1.000006999999e+06
 T16_R_f3_k2: Overall_Qual_calc - 1e+06 t16_leaf2 >= -999995
 T16_L_f1_k2: const_Lot_Area + 1e+06 t16_leaf2 <= 1.010799999999e+06
 T16_L_f3_k3: Overall_Qual_calc + 1e+06 t16_leaf3 <= 1.000006999999e+06
 T16_R_f3_k3: Overall_Qual_calc - 1e+06 t16_leaf3 >= -999995
 T16_R_f1_k3: const_Lot_Area - 1e+06 t16_leaf3 >= -989200
 T16_R_f3_k4: Overall_Qual_calc - 1e+06 t16_leaf4 >= -999993
 T16_L_f10_k4: const_Bsmt_Qual + 1e+06 t16_leaf4 <= 1.000003999999e+06
 T16_L_f20_k4: x_Gr_Liv_Area + 1e+06 t16_leaf4 <= 1.001794999999e+06
 T16_R_f3_k5: Overall_Qual_calc - 1e+06 t16_leaf5 >= -999993
 T16_L_f10_k5: const_Bsmt_Qual + 1e+06 t16_leaf5 <= 1.000003999999e+06
 T16_R_f20_k5: x_Gr_Liv_Area - 1e+06 t16_leaf5 >= -998205
 T16_R_f3_k6: Overall_Qual_calc - 1e+06 t16_leaf6 >= -999993
 T16_R_f10_k6: const_Bsmt_Qual - 1e+06 t16_leaf6 >= -999996
 T16_L_f20_k6: x_Gr_Liv_Area + 1e+06 t16_leaf6 <= 1.002197999999e+06
 T16_R_f3_k7: Overall_Qual_calc - 1e+06 t16_leaf7 >= -999993
 T16_R_f10_k7: const_Bsmt_Qual - 1e+06 t16_leaf7 >= -999996
 T16_R_f20_k7: x_Gr_Liv_Area - 1e+06 t16_leaf7 >= -997802
 T17_onehot: t17_leaf0 + t17_leaf1 + t17_leaf2 + t17_leaf3 + t17_leaf4
   + t17_leaf5 + t17_leaf6 = 1
 T17_L_f3_k0: Overall_Qual_calc + 1e+06 t17_leaf0 <= 1.000006999999e+06
 T17_L_f3_k0: Overall_Qual_calc + 1e+06 t17_leaf0 <= 1.000004999999e+06
 T17_L_f3_k0: Overall_Qual_calc + 1e+06 t17_leaf0 <= 1.000002999999e+06
 T17_L_f3_k1: Overall_Qual_calc + 1e+06 t17_leaf1 <= 1.000006999999e+06
 T17_L_f3_k1: Overall_Qual_calc + 1e+06 t17_leaf1 <= 1.000004999999e+06
 T17_R_f3_k1: Overall_Qual_calc - 1e+06 t17_leaf1 >= -999997
 T17_L_f3_k2: Overall_Qual_calc + 1e+06 t17_leaf2 <= 1.000006999999e+06
 T17_R_f3_k2: Overall_Qual_calc - 1e+06 t17_leaf2 >= -999995
 T17_L_f30_k2: x_Fireplace_Qu + 1e+06 t17_leaf2 <= 999999.999999
 T17_L_f3_k3: Overall_Qual_calc + 1e+06 t17_leaf3 <= 1.000006999999e+06
 T17_R_f3_k3: Overall_Qual_calc - 1e+06 t17_leaf3 >= -999995
 T17_R_f30_k3: x_Fireplace_Qu - 1e+06 t17_leaf3 >= -1e+06
 T17_R_f3_k4: Overall_Qual_calc - 1e+06 t17_leaf4 >= -999993
 T17_L_f27_k4: x_Kitchen_Qual + 1e+06 t17_leaf4 <= 1.000003999999e+06
 T17_L_f17_k4: x_1st_Flr_SF + 1e+06 t17_leaf4 <= 1.001063999999e+06
 T17_R_f3_k5: Overall_Qual_calc - 1e+06 t17_leaf5 >= -999993
 T17_L_f27_k5: x_Kitchen_Qual + 1e+06 t17_leaf5 <= 1.000003999999e+06
 T17_R_f17_k5: x_1st_Flr_SF - 1e+06 t17_leaf5 >= -998936
 T17_R_f3_k6: Overall_Qual_calc - 1e+06 t17_leaf6 >= -999993
 T17_R_f27_k6: x_Kitchen_Qual - 1e+06 t17_leaf6 >= -999996
 T18_onehot: t18_leaf0 + t18_leaf1 + t18_leaf2 + t18_leaf3 + t18_leaf4
   + t18_leaf5 + t18_leaf6 + t18_leaf7 = 1
 T18_L_f8_k0: x_Exter_Qual + 1e+06 t18_leaf0 <= 1.000002999999e+06
 T18_L_f231_k0: central_air_no + 1e+06 t18_leaf0 <= 1.000000999999e+06
 T18_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t18_leaf0 <= 1.000923999999e+06
 T18_L_f8_k1: x_Exter_Qual + 1e+06 t18_leaf1 <= 1.000002999999e+06
 T18_L_f231_k1: central_air_no + 1e+06 t18_leaf1 <= 1.000000999999e+06
 T18_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t18_leaf1 >= -999076
 T18_L_f8_k2: x_Exter_Qual + 1e+06 t18_leaf2 <= 1.000002999999e+06
 T18_R_f231_k2: central_air_no - 1e+06 t18_leaf2 >= -999999
 T18_L_f4_k2: const_Overall_Cond + 1e+06 t18_leaf2 <= 1.000005999999e+06
 T18_L_f8_k3: x_Exter_Qual + 1e+06 t18_leaf3 <= 1.000002999999e+06
 T18_R_f231_k3: central_air_no - 1e+06 t18_leaf3 >= -999999
 T18_R_f4_k3: const_Overall_Cond - 1e+06 t18_leaf3 >= -999994
 T18_R_f8_k4: x_Exter_Qual - 1e+06 t18_leaf4 >= -999997
 T18_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t18_leaf4 <= 1.001388999999e+06
 T18_L_f18_k4: const_2nd_Flr_SF + 1e+06 t18_leaf4 <= 1.000827999999e+06
 T18_R_f8_k5: x_Exter_Qual - 1e+06 t18_leaf5 >= -999997
 T18_L_f15_k5: const_Total_Bsmt_SF + 1e+06 t18_leaf5 <= 1.001388999999e+06
 T18_R_f18_k5: const_2nd_Flr_SF - 1e+06 t18_leaf5 >= -999172
 T18_R_f8_k6: x_Exter_Qual - 1e+06 t18_leaf6 >= -999997
 T18_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t18_leaf6 >= -998611
 T18_L_f1_k6: const_Lot_Area + 1e+06 t18_leaf6 <= 1.012098999999e+06
 T18_R_f8_k7: x_Exter_Qual - 1e+06 t18_leaf7 >= -999997
 T18_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t18_leaf7 >= -998611
 T18_R_f1_k7: const_Lot_Area - 1e+06 t18_leaf7 >= -987901
 T19_onehot: t19_leaf0 + t19_leaf1 + t19_leaf2 + t19_leaf3 + t19_leaf4
   + t19_leaf5 + t19_leaf6 + t19_leaf7 = 1
 T19_L_f5_k0: const_Year_Built + 1e+06 t19_leaf0 <= 1.001980999999e+06
 T19_L_f30_k0: x_Fireplace_Qu + 1e+06 t19_leaf0 <= 999999.999999
 T19_L_f20_k0: x_Gr_Liv_Area + 1e+06 t19_leaf0 <= 1.000809999999e+06
 T19_L_f5_k1: const_Year_Built + 1e+06 t19_leaf1 <= 1.001980999999e+06
 T19_L_f30_k1: x_Fireplace_Qu + 1e+06 t19_leaf1 <= 999999.999999
 T19_R_f20_k1: x_Gr_Liv_Area - 1e+06 t19_leaf1 >= -999190
 T19_L_f5_k2: const_Year_Built + 1e+06 t19_leaf2 <= 1.001980999999e+06
 T19_R_f30_k2: x_Fireplace_Qu - 1e+06 t19_leaf2 >= -1e+06
 T19_L_f20_k2: x_Gr_Liv_Area + 1e+06 t19_leaf2 <= 1.001778999999e+06
 T19_L_f5_k3: const_Year_Built + 1e+06 t19_leaf3 <= 1.001980999999e+06
 T19_R_f30_k3: x_Fireplace_Qu - 1e+06 t19_leaf3 >= -1e+06
 T19_R_f20_k3: x_Gr_Liv_Area - 1e+06 t19_leaf3 >= -998221
 T19_R_f5_k4: const_Year_Built - 1e+06 t19_leaf4 >= -998019
 T19_L_f20_k4: x_Gr_Liv_Area + 1e+06 t19_leaf4 <= 1.001960999999e+06
 T19_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t19_leaf4 <= 1.001493999999e+06
 T19_R_f5_k5: const_Year_Built - 1e+06 t19_leaf5 >= -998019
 T19_L_f20_k5: x_Gr_Liv_Area + 1e+06 t19_leaf5 <= 1.001960999999e+06
 T19_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t19_leaf5 >= -998506
 T19_R_f5_k6: const_Year_Built - 1e+06 t19_leaf6 >= -998019
 T19_R_f20_k6: x_Gr_Liv_Area - 1e+06 t19_leaf6 >= -998039
 T19_L_f15_k6: const_Total_Bsmt_SF + 1e+06 t19_leaf6 <= 1.001186999999e+06
 T19_R_f5_k7: const_Year_Built - 1e+06 t19_leaf7 >= -998019
 T19_R_f20_k7: x_Gr_Liv_Area - 1e+06 t19_leaf7 >= -998039
 T19_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t19_leaf7 >= -998813
 T20_onehot: t20_leaf0 + t20_leaf1 + t20_leaf2 + t20_leaf3 + t20_leaf4
   + t20_leaf5 + t20_leaf6 + t20_leaf7 = 1
 T20_L_f3_k0: Overall_Qual_calc + 1e+06 t20_leaf0 <= 1.000005999999e+06
 T20_L_f66_k0: const_MS_Zoning_RL + 1e+06 t20_leaf0 <= 1.000000999999e+06
 T20_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t20_leaf0 <= 1.000004999999e+06
 T20_L_f3_k1: Overall_Qual_calc + 1e+06 t20_leaf1 <= 1.000005999999e+06
 T20_L_f66_k1: const_MS_Zoning_RL + 1e+06 t20_leaf1 <= 1.000000999999e+06
 T20_R_f28_k1: x_TotRms_AbvGrd - 1e+06 t20_leaf1 >= -999995
 T20_L_f3_k2: Overall_Qual_calc + 1e+06 t20_leaf2 <= 1.000005999999e+06
 T20_R_f66_k2: const_MS_Zoning_RL - 1e+06 t20_leaf2 >= -999999
 T20_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t20_leaf2 <= 1.000556999999e+06
 T20_L_f3_k3: Overall_Qual_calc + 1e+06 t20_leaf3 <= 1.000005999999e+06
 T20_R_f66_k3: const_MS_Zoning_RL - 1e+06 t20_leaf3 >= -999999
 T20_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t20_leaf3 >= -999443
 T20_R_f3_k4: Overall_Qual_calc - 1e+06 t20_leaf4 >= -999994
 T20_L_f3_k4: Overall_Qual_calc + 1e+06 t20_leaf4 <= 1.000007999999e+06
 T20_L_f1_k4: const_Lot_Area + 1e+06 t20_leaf4 <= 1.007499999999e+06
 T20_R_f3_k5: Overall_Qual_calc - 1e+06 t20_leaf5 >= -999994
 T20_L_f3_k5: Overall_Qual_calc + 1e+06 t20_leaf5 <= 1.000007999999e+06
 T20_R_f1_k5: const_Lot_Area - 1e+06 t20_leaf5 >= -992500
 T20_R_f3_k6: Overall_Qual_calc - 1e+06 t20_leaf6 >= -999994
 T20_R_f3_k6: Overall_Qual_calc - 1e+06 t20_leaf6 >= -999992
 T20_L_f36_k6: x_Wood_Deck_SF + 1e+06 t20_leaf6 <= 1.000239999999e+06
 T20_R_f3_k7: Overall_Qual_calc - 1e+06 t20_leaf7 >= -999994
 T20_R_f3_k7: Overall_Qual_calc - 1e+06 t20_leaf7 >= -999992
 T20_R_f36_k7: x_Wood_Deck_SF - 1e+06 t20_leaf7 >= -999760
 T21_onehot: t21_leaf0 + t21_leaf1 + t21_leaf2 + t21_leaf3 + t21_leaf4
   + t21_leaf5 + t21_leaf6 + t21_leaf7 = 1
 T21_L_f3_k0: Overall_Qual_calc + 1e+06 t21_leaf0 <= 1.000006999999e+06
 T21_L_f66_k0: const_MS_Zoning_RL + 1e+06 t21_leaf0 <= 1.000000999999e+06
 T21_L_f4_k0: const_Overall_Cond + 1e+06 t21_leaf0 <= 1.000004999999e+06
 T21_L_f3_k1: Overall_Qual_calc + 1e+06 t21_leaf1 <= 1.000006999999e+06
 T21_L_f66_k1: const_MS_Zoning_RL + 1e+06 t21_leaf1 <= 1.000000999999e+06
 T21_R_f4_k1: const_Overall_Cond - 1e+06 t21_leaf1 >= -999995
 T21_L_f3_k2: Overall_Qual_calc + 1e+06 t21_leaf2 <= 1.000006999999e+06
 T21_R_f66_k2: const_MS_Zoning_RL - 1e+06 t21_leaf2 >= -999999
 T21_L_f3_k2: Overall_Qual_calc + 1e+06 t21_leaf2 <= 1.000005999999e+06
 T21_L_f3_k3: Overall_Qual_calc + 1e+06 t21_leaf3 <= 1.000006999999e+06
 T21_R_f66_k3: const_MS_Zoning_RL - 1e+06 t21_leaf3 >= -999999
 T21_R_f3_k3: Overall_Qual_calc - 1e+06 t21_leaf3 >= -999994
 T21_R_f3_k4: Overall_Qual_calc - 1e+06 t21_leaf4 >= -999993
 T21_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t21_leaf4 <= 1.001096999999e+06
 T21_L_f33_k4: x_Garage_Area + 1e+06 t21_leaf4 <= 1.000545999999e+06
 T21_R_f3_k5: Overall_Qual_calc - 1e+06 t21_leaf5 >= -999993
 T21_L_f12_k5: x_BsmtFin_SF_1 + 1e+06 t21_leaf5 <= 1.001096999999e+06
 T21_R_f33_k5: x_Garage_Area - 1e+06 t21_leaf5 >= -999454
 T21_R_f3_k6: Overall_Qual_calc - 1e+06 t21_leaf6 >= -999993
 T21_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t21_leaf6 >= -998903
 T21_L_f27_k6: x_Kitchen_Qual + 1e+06 t21_leaf6 <= 1.000003999999e+06
 T21_R_f3_k7: Overall_Qual_calc - 1e+06 t21_leaf7 >= -999993
 T21_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t21_leaf7 >= -998903
 T21_R_f27_k7: x_Kitchen_Qual - 1e+06 t21_leaf7 >= -999996
 T22_onehot: t22_leaf0 + t22_leaf1 + t22_leaf2 + t22_leaf3 + t22_leaf4
   + t22_leaf5 + t22_leaf6 + t22_leaf7 = 1
 T22_L_f20_k0: x_Gr_Liv_Area + 1e+06 t22_leaf0 <= 1.001490999999e+06
 T22_L_f17_k0: x_1st_Flr_SF + 1e+06 t22_leaf0 <= 1.001024999999e+06
 T22_L_f6_k0: const_Year_Remod_Add + 1e+06 t22_leaf0 <= 1.001950999999e+06
 T22_L_f20_k1: x_Gr_Liv_Area + 1e+06 t22_leaf1 <= 1.001490999999e+06
 T22_L_f17_k1: x_1st_Flr_SF + 1e+06 t22_leaf1 <= 1.001024999999e+06
 T22_R_f6_k1: const_Year_Remod_Add - 1e+06 t22_leaf1 >= -998049
 T22_L_f20_k2: x_Gr_Liv_Area + 1e+06 t22_leaf2 <= 1.001490999999e+06
 T22_R_f17_k2: x_1st_Flr_SF - 1e+06 t22_leaf2 >= -998975
 T22_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t22_leaf2 <= 1.001952999999e+06
 T22_L_f20_k3: x_Gr_Liv_Area + 1e+06 t22_leaf3 <= 1.001490999999e+06
 T22_R_f17_k3: x_1st_Flr_SF - 1e+06 t22_leaf3 >= -998975
 T22_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t22_leaf3 >= -998047
 T22_R_f20_k4: x_Gr_Liv_Area - 1e+06 t22_leaf4 >= -998509
 T22_L_f6_k4: const_Year_Remod_Add + 1e+06 t22_leaf4 <= 1.001982999999e+06
 T22_L_f29_k4: const_Fireplaces + 1e+06 t22_leaf4 <= 1.000001999999e+06
 T22_R_f20_k5: x_Gr_Liv_Area - 1e+06 t22_leaf5 >= -998509
 T22_L_f6_k5: const_Year_Remod_Add + 1e+06 t22_leaf5 <= 1.001982999999e+06
 T22_R_f29_k5: const_Fireplaces - 1e+06 t22_leaf5 >= -999998
 T22_R_f20_k6: x_Gr_Liv_Area - 1e+06 t22_leaf6 >= -998509
 T22_R_f6_k6: const_Year_Remod_Add - 1e+06 t22_leaf6 >= -998017
 T22_L_f20_k6: x_Gr_Liv_Area + 1e+06 t22_leaf6 <= 1.001901999999e+06
 T22_R_f20_k7: x_Gr_Liv_Area - 1e+06 t22_leaf7 >= -998509
 T22_R_f6_k7: const_Year_Remod_Add - 1e+06 t22_leaf7 >= -998017
 T22_R_f20_k7: x_Gr_Liv_Area - 1e+06 t22_leaf7 >= -998098
 T23_onehot: t23_leaf0 + t23_leaf1 + t23_leaf2 + t23_leaf3 + t23_leaf4
   + t23_leaf5 + t23_leaf6 + t23_leaf7 = 1
 T23_L_f3_k0: Overall_Qual_calc + 1e+06 t23_leaf0 <= 1.000005999999e+06
 T23_L_f3_k0: Overall_Qual_calc + 1e+06 t23_leaf0 <= 1.000003999999e+06
 T23_L_f5_k0: const_Year_Built + 1e+06 t23_leaf0 <= 1.001949999999e+06
 T23_L_f3_k1: Overall_Qual_calc + 1e+06 t23_leaf1 <= 1.000005999999e+06
 T23_L_f3_k1: Overall_Qual_calc + 1e+06 t23_leaf1 <= 1.000003999999e+06
 T23_R_f5_k1: const_Year_Built - 1e+06 t23_leaf1 >= -998050
 T23_L_f3_k2: Overall_Qual_calc + 1e+06 t23_leaf2 <= 1.000005999999e+06
 T23_R_f3_k2: Overall_Qual_calc - 1e+06 t23_leaf2 >= -999996
 T23_L_f66_k2: const_MS_Zoning_RL + 1e+06 t23_leaf2 <= 1.000000999999e+06
 T23_L_f3_k3: Overall_Qual_calc + 1e+06 t23_leaf3 <= 1.000005999999e+06
 T23_R_f3_k3: Overall_Qual_calc - 1e+06 t23_leaf3 >= -999996
 T23_R_f66_k3: const_MS_Zoning_RL - 1e+06 t23_leaf3 >= -999999
 T23_R_f3_k4: Overall_Qual_calc - 1e+06 t23_leaf4 >= -999994
 T23_L_f32_k4: const_Garage_Cars + 1e+06 t23_leaf4 <= 1.000002999999e+06
 T23_L_f1_k4: const_Lot_Area + 1e+06 t23_leaf4 <= 1.013261999999e+06
 T23_R_f3_k5: Overall_Qual_calc - 1e+06 t23_leaf5 >= -999994
 T23_L_f32_k5: const_Garage_Cars + 1e+06 t23_leaf5 <= 1.000002999999e+06
 T23_R_f1_k5: const_Lot_Area - 1e+06 t23_leaf5 >= -986738
 T23_R_f3_k6: Overall_Qual_calc - 1e+06 t23_leaf6 >= -999994
 T23_R_f32_k6: const_Garage_Cars - 1e+06 t23_leaf6 >= -999997
 T23_L_f3_k6: Overall_Qual_calc + 1e+06 t23_leaf6 <= 1.000008999999e+06
 T23_R_f3_k7: Overall_Qual_calc - 1e+06 t23_leaf7 >= -999994
 T23_R_f32_k7: const_Garage_Cars - 1e+06 t23_leaf7 >= -999997
 T23_R_f3_k7: Overall_Qual_calc - 1e+06 t23_leaf7 >= -999991
 T24_onehot: t24_leaf0 + t24_leaf1 + t24_leaf2 + t24_leaf3 + t24_leaf4
   + t24_leaf5 + t24_leaf6 + t24_leaf7 = 1
 T24_L_f3_k0: Overall_Qual_calc + 1e+06 t24_leaf0 <= 1.000006999999e+06
 T24_L_f32_k0: const_Garage_Cars + 1e+06 t24_leaf0 <= 1.000001999999e+06
 T24_L_f17_k0: x_1st_Flr_SF + 1e+06 t24_leaf0 <= 1.000893999999e+06
 T24_L_f3_k1: Overall_Qual_calc + 1e+06 t24_leaf1 <= 1.000006999999e+06
 T24_L_f32_k1: const_Garage_Cars + 1e+06 t24_leaf1 <= 1.000001999999e+06
 T24_R_f17_k1: x_1st_Flr_SF - 1e+06 t24_leaf1 >= -999106
 T24_L_f3_k2: Overall_Qual_calc + 1e+06 t24_leaf2 <= 1.000006999999e+06
 T24_R_f32_k2: const_Garage_Cars - 1e+06 t24_leaf2 >= -999998
 T24_L_f10_k2: const_Bsmt_Qual + 1e+06 t24_leaf2 <= 1.000001999999e+06
 T24_L_f3_k3: Overall_Qual_calc + 1e+06 t24_leaf3 <= 1.000006999999e+06
 T24_R_f32_k3: const_Garage_Cars - 1e+06 t24_leaf3 >= -999998
 T24_R_f10_k3: const_Bsmt_Qual - 1e+06 t24_leaf3 >= -999998
 T24_R_f3_k4: Overall_Qual_calc - 1e+06 t24_leaf4 >= -999993
 T24_L_f3_k4: Overall_Qual_calc + 1e+06 t24_leaf4 <= 1.000007999999e+06
 T24_L_f1_k4: const_Lot_Area + 1e+06 t24_leaf4 <= 1.007499999999e+06
 T24_R_f3_k5: Overall_Qual_calc - 1e+06 t24_leaf5 >= -999993
 T24_L_f3_k5: Overall_Qual_calc + 1e+06 t24_leaf5 <= 1.000007999999e+06
 T24_R_f1_k5: const_Lot_Area - 1e+06 t24_leaf5 >= -992500
 T24_R_f3_k6: Overall_Qual_calc - 1e+06 t24_leaf6 >= -999993
 T24_R_f3_k6: Overall_Qual_calc - 1e+06 t24_leaf6 >= -999992
 T24_L_f1_k6: const_Lot_Area + 1e+06 t24_leaf6 <= 1.012143999999e+06
 T24_R_f3_k7: Overall_Qual_calc - 1e+06 t24_leaf7 >= -999993
 T24_R_f3_k7: Overall_Qual_calc - 1e+06 t24_leaf7 >= -999992
 T24_R_f1_k7: const_Lot_Area - 1e+06 t24_leaf7 >= -987856
 T25_onehot: t25_leaf0 + t25_leaf1 + t25_leaf2 + t25_leaf3 + t25_leaf4
   + t25_leaf5 + t25_leaf6 + t25_leaf7 = 1
 T25_L_f8_k0: x_Exter_Qual + 1e+06 t25_leaf0 <= 1.000002999999e+06
 T25_L_f20_k0: x_Gr_Liv_Area + 1e+06 t25_leaf0 <= 1.001023999999e+06
 T25_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t25_leaf0 <= 1.000760999999e+06
 T25_L_f8_k1: x_Exter_Qual + 1e+06 t25_leaf1 <= 1.000002999999e+06
 T25_L_f20_k1: x_Gr_Liv_Area + 1e+06 t25_leaf1 <= 1.001023999999e+06
 T25_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t25_leaf1 >= -999239
 T25_L_f8_k2: x_Exter_Qual + 1e+06 t25_leaf2 <= 1.000002999999e+06
 T25_R_f20_k2: x_Gr_Liv_Area - 1e+06 t25_leaf2 >= -998976
 T25_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t25_leaf2 <= 1.000137999999e+06
 T25_L_f8_k3: x_Exter_Qual + 1e+06 t25_leaf3 <= 1.000002999999e+06
 T25_R_f20_k3: x_Gr_Liv_Area - 1e+06 t25_leaf3 >= -998976
 T25_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t25_leaf3 >= -999862
 T25_R_f8_k4: x_Exter_Qual - 1e+06 t25_leaf4 >= -999997
 T25_L_f20_k4: x_Gr_Liv_Area + 1e+06 t25_leaf4 <= 1.001960999999e+06
 T25_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t25_leaf4 <= 1.000747999999e+06
 T25_R_f8_k5: x_Exter_Qual - 1e+06 t25_leaf5 >= -999997
 T25_L_f20_k5: x_Gr_Liv_Area + 1e+06 t25_leaf5 <= 1.001960999999e+06
 T25_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t25_leaf5 >= -999252
 T25_R_f8_k6: x_Exter_Qual - 1e+06 t25_leaf6 >= -999997
 T25_R_f20_k6: x_Gr_Liv_Area - 1e+06 t25_leaf6 >= -998039
 T25_L_f15_k6: const_Total_Bsmt_SF + 1e+06 t25_leaf6 <= 1.001054999999e+06
 T25_R_f8_k7: x_Exter_Qual - 1e+06 t25_leaf7 >= -999997
 T25_R_f20_k7: x_Gr_Liv_Area - 1e+06 t25_leaf7 >= -998039
 T25_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t25_leaf7 >= -998945
 T26_onehot: t26_leaf0 + t26_leaf1 + t26_leaf2 + t26_leaf3 + t26_leaf4
   + t26_leaf5 + t26_leaf6 + t26_leaf7 = 1
 T26_L_f3_k0: Overall_Qual_calc + 1e+06 t26_leaf0 <= 1.000005999999e+06
 T26_L_f257_k0: x_paved_drive_is_N + 1e+06 t26_leaf0 <= 1.000000999999e+06
 T26_L_f17_k0: x_1st_Flr_SF + 1e+06 t26_leaf0 <= 1.001049999999e+06
 T26_L_f3_k1: Overall_Qual_calc + 1e+06 t26_leaf1 <= 1.000005999999e+06
 T26_L_f257_k1: x_paved_drive_is_N + 1e+06 t26_leaf1 <= 1.000000999999e+06
 T26_R_f17_k1: x_1st_Flr_SF - 1e+06 t26_leaf1 >= -998950
 T26_L_f3_k2: Overall_Qual_calc + 1e+06 t26_leaf2 <= 1.000005999999e+06
 T26_R_f257_k2: x_paved_drive_is_N - 1e+06 t26_leaf2 >= -999999
 T26_L_f20_k2: x_Gr_Liv_Area + 1e+06 t26_leaf2 <= 1.000959999999e+06
 T26_L_f3_k3: Overall_Qual_calc + 1e+06 t26_leaf3 <= 1.000005999999e+06
 T26_R_f257_k3: x_paved_drive_is_N - 1e+06 t26_leaf3 >= -999999
 T26_R_f20_k3: x_Gr_Liv_Area - 1e+06 t26_leaf3 >= -999040
 T26_R_f3_k4: Overall_Qual_calc - 1e+06 t26_leaf4 >= -999994
 T26_L_f3_k4: Overall_Qual_calc + 1e+06 t26_leaf4 <= 1.000007999999e+06
 T26_L_f20_k4: x_Gr_Liv_Area + 1e+06 t26_leaf4 <= 1.001487999999e+06
 T26_R_f3_k5: Overall_Qual_calc - 1e+06 t26_leaf5 >= -999994
 T26_L_f3_k5: Overall_Qual_calc + 1e+06 t26_leaf5 <= 1.000007999999e+06
 T26_R_f20_k5: x_Gr_Liv_Area - 1e+06 t26_leaf5 >= -998512
 T26_R_f3_k6: Overall_Qual_calc - 1e+06 t26_leaf6 >= -999994
 T26_R_f3_k6: Overall_Qual_calc - 1e+06 t26_leaf6 >= -999992
 T26_L_f20_k6: x_Gr_Liv_Area + 1e+06 t26_leaf6 <= 1.002223999999e+06
 T26_R_f3_k7: Overall_Qual_calc - 1e+06 t26_leaf7 >= -999994
 T26_R_f3_k7: Overall_Qual_calc - 1e+06 t26_leaf7 >= -999992
 T26_R_f20_k7: x_Gr_Liv_Area - 1e+06 t26_leaf7 >= -997776
 T27_onehot: t27_leaf0 + t27_leaf1 + t27_leaf2 + t27_leaf3 + t27_leaf4
   + t27_leaf5 + t27_leaf6 + t27_leaf7 = 1
 T27_L_f8_k0: x_Exter_Qual + 1e+06 t27_leaf0 <= 1.000002999999e+06
 T27_L_f231_k0: central_air_no + 1e+06 t27_leaf0 <= 1.000000999999e+06
 T27_L_f1_k0: const_Lot_Area + 1e+06 t27_leaf0 <= 1.009115999999e+06
 T27_L_f8_k1: x_Exter_Qual + 1e+06 t27_leaf1 <= 1.000002999999e+06
 T27_L_f231_k1: central_air_no + 1e+06 t27_leaf1 <= 1.000000999999e+06
 T27_R_f1_k1: const_Lot_Area - 1e+06 t27_leaf1 >= -990884
 T27_L_f8_k2: x_Exter_Qual + 1e+06 t27_leaf2 <= 1.000002999999e+06
 T27_R_f231_k2: central_air_no - 1e+06 t27_leaf2 >= -999999
 T27_L_f4_k2: const_Overall_Cond + 1e+06 t27_leaf2 <= 1.000005999999e+06
 T27_L_f8_k3: x_Exter_Qual + 1e+06 t27_leaf3 <= 1.000002999999e+06
 T27_R_f231_k3: central_air_no - 1e+06 t27_leaf3 >= -999999
 T27_R_f4_k3: const_Overall_Cond - 1e+06 t27_leaf3 >= -999994
 T27_R_f8_k4: x_Exter_Qual - 1e+06 t27_leaf4 >= -999997
 T27_L_f30_k4: x_Fireplace_Qu + 1e+06 t27_leaf4 <= 1.000000999999e+06
 T27_L_f33_k4: x_Garage_Area + 1e+06 t27_leaf4 <= 1.000496999999e+06
 T27_R_f8_k5: x_Exter_Qual - 1e+06 t27_leaf5 >= -999997
 T27_L_f30_k5: x_Fireplace_Qu + 1e+06 t27_leaf5 <= 1.000000999999e+06
 T27_R_f33_k5: x_Garage_Area - 1e+06 t27_leaf5 >= -999503
 T27_R_f8_k6: x_Exter_Qual - 1e+06 t27_leaf6 >= -999997
 T27_R_f30_k6: x_Fireplace_Qu - 1e+06 t27_leaf6 >= -999999
 T27_L_f33_k6: x_Garage_Area + 1e+06 t27_leaf6 <= 1.000705999999e+06
 T27_R_f8_k7: x_Exter_Qual - 1e+06 t27_leaf7 >= -999997
 T27_R_f30_k7: x_Fireplace_Qu - 1e+06 t27_leaf7 >= -999999
 T27_R_f33_k7: x_Garage_Area - 1e+06 t27_leaf7 >= -999294
 T28_onehot: t28_leaf0 + t28_leaf1 + t28_leaf2 + t28_leaf3 + t28_leaf4
   + t28_leaf5 + t28_leaf6 + t28_leaf7 = 1
 T28_L_f20_k0: x_Gr_Liv_Area + 1e+06 t28_leaf0 <= 1.001415999999e+06
 T28_L_f5_k0: const_Year_Built + 1e+06 t28_leaf0 <= 1.001952999999e+06
 T28_L_f20_k0: x_Gr_Liv_Area + 1e+06 t28_leaf0 <= 1.000832999999e+06
 T28_L_f20_k1: x_Gr_Liv_Area + 1e+06 t28_leaf1 <= 1.001415999999e+06
 T28_L_f5_k1: const_Year_Built + 1e+06 t28_leaf1 <= 1.001952999999e+06
 T28_R_f20_k1: x_Gr_Liv_Area - 1e+06 t28_leaf1 >= -999167
 T28_L_f20_k2: x_Gr_Liv_Area + 1e+06 t28_leaf2 <= 1.001415999999e+06
 T28_R_f5_k2: const_Year_Built - 1e+06 t28_leaf2 >= -998047
 T28_L_f17_k2: x_1st_Flr_SF + 1e+06 t28_leaf2 <= 1.000950999999e+06
 T28_L_f20_k3: x_Gr_Liv_Area + 1e+06 t28_leaf3 <= 1.001415999999e+06
 T28_R_f5_k3: const_Year_Built - 1e+06 t28_leaf3 >= -998047
 T28_R_f17_k3: x_1st_Flr_SF - 1e+06 t28_leaf3 >= -999049
 T28_R_f20_k4: x_Gr_Liv_Area - 1e+06 t28_leaf4 >= -998584
 T28_L_f20_k4: x_Gr_Liv_Area + 1e+06 t28_leaf4 <= 1.002223999999e+06
 T28_L_f21_k4: const_Bsmt_Full_Bath + 1e+06 t28_leaf4
   <= 1.000000999999e+06
 T28_R_f20_k5: x_Gr_Liv_Area - 1e+06 t28_leaf5 >= -998584
 T28_L_f20_k5: x_Gr_Liv_Area + 1e+06 t28_leaf5 <= 1.002223999999e+06
 T28_R_f21_k5: const_Bsmt_Full_Bath - 1e+06 t28_leaf5 >= -999999
 T28_R_f20_k6: x_Gr_Liv_Area - 1e+06 t28_leaf6 >= -998584
 T28_R_f20_k6: x_Gr_Liv_Area - 1e+06 t28_leaf6 >= -997776
 T28_L_f1_k6: const_Lot_Area + 1e+06 t28_leaf6 <= 1.011699999999e+06
 T28_R_f20_k7: x_Gr_Liv_Area - 1e+06 t28_leaf7 >= -998584
 T28_R_f20_k7: x_Gr_Liv_Area - 1e+06 t28_leaf7 >= -997776
 T28_R_f1_k7: const_Lot_Area - 1e+06 t28_leaf7 >= -988300
 T29_onehot: t29_leaf0 + t29_leaf1 + t29_leaf2 + t29_leaf3 + t29_leaf4
   + t29_leaf5 + t29_leaf6 + t29_leaf7 = 1
 T29_L_f3_k0: Overall_Qual_calc + 1e+06 t29_leaf0 <= 1.000006999999e+06
 T29_L_f3_k0: Overall_Qual_calc + 1e+06 t29_leaf0 <= 1.000003999999e+06
 T29_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t29_leaf0
   <= 1.000000999999e+06
 T29_L_f3_k1: Overall_Qual_calc + 1e+06 t29_leaf1 <= 1.000006999999e+06
 T29_L_f3_k1: Overall_Qual_calc + 1e+06 t29_leaf1 <= 1.000003999999e+06
 T29_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t29_leaf1 >= -999999
 T29_L_f3_k2: Overall_Qual_calc + 1e+06 t29_leaf2 <= 1.000006999999e+06
 T29_R_f3_k2: Overall_Qual_calc - 1e+06 t29_leaf2 >= -999996
 T29_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t29_leaf2 <= 1.000350999999e+06
 T29_L_f3_k3: Overall_Qual_calc + 1e+06 t29_leaf3 <= 1.000006999999e+06
 T29_R_f3_k3: Overall_Qual_calc - 1e+06 t29_leaf3 >= -999996
 T29_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t29_leaf3 >= -999649
 T29_R_f3_k4: Overall_Qual_calc - 1e+06 t29_leaf4 >= -999993
 T29_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t29_leaf4 <= 1.001474999999e+06
 T29_L_f1_k4: const_Lot_Area + 1e+06 t29_leaf4 <= 1.011979999999e+06
 T29_R_f3_k5: Overall_Qual_calc - 1e+06 t29_leaf5 >= -999993
 T29_L_f15_k5: const_Total_Bsmt_SF + 1e+06 t29_leaf5 <= 1.001474999999e+06
 T29_R_f1_k5: const_Lot_Area - 1e+06 t29_leaf5 >= -988020
 T29_R_f3_k6: Overall_Qual_calc - 1e+06 t29_leaf6 >= -999993
 T29_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t29_leaf6 >= -998525
 T29_L_f30_k6: x_Fireplace_Qu + 1e+06 t29_leaf6 <= 1.000001999999e+06
 T29_R_f3_k7: Overall_Qual_calc - 1e+06 t29_leaf7 >= -999993
 T29_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t29_leaf7 >= -998525
 T29_R_f30_k7: x_Fireplace_Qu - 1e+06 t29_leaf7 >= -999998
 T30_onehot: t30_leaf0 + t30_leaf1 + t30_leaf2 + t30_leaf3 + t30_leaf4
   + t30_leaf5 + t30_leaf6 + t30_leaf7 = 1
 T30_L_f3_k0: Overall_Qual_calc + 1e+06 t30_leaf0 <= 1.000005999999e+06
 T30_L_f11_k0: x_Bsmt_Cond + 1e+06 t30_leaf0 <= 1.000001999999e+06
 T30_L_f33_k0: x_Garage_Area + 1e+06 t30_leaf0 <= 1.000617999999e+06
 T30_L_f3_k1: Overall_Qual_calc + 1e+06 t30_leaf1 <= 1.000005999999e+06
 T30_L_f11_k1: x_Bsmt_Cond + 1e+06 t30_leaf1 <= 1.000001999999e+06
 T30_R_f33_k1: x_Garage_Area - 1e+06 t30_leaf1 >= -999382
 T30_L_f3_k2: Overall_Qual_calc + 1e+06 t30_leaf2 <= 1.000005999999e+06
 T30_R_f11_k2: x_Bsmt_Cond - 1e+06 t30_leaf2 >= -999998
 T30_L_f66_k2: const_MS_Zoning_RL + 1e+06 t30_leaf2 <= 1.000000999999e+06
 T30_L_f3_k3: Overall_Qual_calc + 1e+06 t30_leaf3 <= 1.000005999999e+06
 T30_R_f11_k3: x_Bsmt_Cond - 1e+06 t30_leaf3 >= -999998
 T30_R_f66_k3: const_MS_Zoning_RL - 1e+06 t30_leaf3 >= -999999
 T30_R_f3_k4: Overall_Qual_calc - 1e+06 t30_leaf4 >= -999994
 T30_L_f32_k4: const_Garage_Cars + 1e+06 t30_leaf4 <= 1.000002999999e+06
 T30_L_f1_k4: const_Lot_Area + 1e+06 t30_leaf4 <= 1.006372999999e+06
 T30_R_f3_k5: Overall_Qual_calc - 1e+06 t30_leaf5 >= -999994
 T30_L_f32_k5: const_Garage_Cars + 1e+06 t30_leaf5 <= 1.000002999999e+06
 T30_R_f1_k5: const_Lot_Area - 1e+06 t30_leaf5 >= -993627
 T30_R_f3_k6: Overall_Qual_calc - 1e+06 t30_leaf6 >= -999994
 T30_R_f32_k6: const_Garage_Cars - 1e+06 t30_leaf6 >= -999997
 T30_L_f1_k6: const_Lot_Area + 1e+06 t30_leaf6 <= 1.013517999999e+06
 T30_R_f3_k7: Overall_Qual_calc - 1e+06 t30_leaf7 >= -999994
 T30_R_f32_k7: const_Garage_Cars - 1e+06 t30_leaf7 >= -999997
 T30_R_f1_k7: const_Lot_Area - 1e+06 t30_leaf7 >= -986482
 T31_onehot: t31_leaf0 + t31_leaf1 + t31_leaf2 + t31_leaf3 + t31_leaf4
   + t31_leaf5 + t31_leaf6 + t31_leaf7 = 1
 T31_L_f20_k0: x_Gr_Liv_Area + 1e+06 t31_leaf0 <= 1.001487999999e+06
 T31_L_f35_k0: x_Garage_Cond + 1e+06 t31_leaf0 <= 1.000001999999e+06
 T31_L_f20_k0: x_Gr_Liv_Area + 1e+06 t31_leaf0 <= 1.001154999999e+06
 T31_L_f20_k1: x_Gr_Liv_Area + 1e+06 t31_leaf1 <= 1.001487999999e+06
 T31_L_f35_k1: x_Garage_Cond + 1e+06 t31_leaf1 <= 1.000001999999e+06
 T31_R_f20_k1: x_Gr_Liv_Area - 1e+06 t31_leaf1 >= -998845
 T31_L_f20_k2: x_Gr_Liv_Area + 1e+06 t31_leaf2 <= 1.001487999999e+06
 T31_R_f35_k2: x_Garage_Cond - 1e+06 t31_leaf2 >= -999998
 T31_L_f17_k2: x_1st_Flr_SF + 1e+06 t31_leaf2 <= 1.000987999999e+06
 T31_L_f20_k3: x_Gr_Liv_Area + 1e+06 t31_leaf3 <= 1.001487999999e+06
 T31_R_f35_k3: x_Garage_Cond - 1e+06 t31_leaf3 >= -999998
 T31_R_f17_k3: x_1st_Flr_SF - 1e+06 t31_leaf3 >= -999012
 T31_R_f20_k4: x_Gr_Liv_Area - 1e+06 t31_leaf4 >= -998512
 T31_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t31_leaf4 <= 1.000772999999e+06
 T31_L_f26_k4: x_Kitchen_AbvGr + 1e+06 t31_leaf4 <= 1.000001999999e+06
 T31_R_f20_k5: x_Gr_Liv_Area - 1e+06 t31_leaf5 >= -998512
 T31_L_f12_k5: x_BsmtFin_SF_1 + 1e+06 t31_leaf5 <= 1.000772999999e+06
 T31_R_f26_k5: x_Kitchen_AbvGr - 1e+06 t31_leaf5 >= -999998
 T31_R_f20_k6: x_Gr_Liv_Area - 1e+06 t31_leaf6 >= -998512
 T31_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t31_leaf6 >= -999227
 T31_L_f20_k6: x_Gr_Liv_Area + 1e+06 t31_leaf6 <= 1.001872999999e+06
 T31_R_f20_k7: x_Gr_Liv_Area - 1e+06 t31_leaf7 >= -998512
 T31_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t31_leaf7 >= -999227
 T31_R_f20_k7: x_Gr_Liv_Area - 1e+06 t31_leaf7 >= -998127
 T32_onehot: t32_leaf0 + t32_leaf1 + t32_leaf2 + t32_leaf3 + t32_leaf4
   + t32_leaf5 + t32_leaf6 + t32_leaf7 = 1
 T32_L_f32_k0: const_Garage_Cars + 1e+06 t32_leaf0 <= 1.000001999999e+06
 T32_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t32_leaf0 <= 1.000647999999e+06
 T32_L_f6_k0: const_Year_Remod_Add + 1e+06 t32_leaf0 <= 1.001977999999e+06
 T32_L_f32_k1: const_Garage_Cars + 1e+06 t32_leaf1 <= 1.000001999999e+06
 T32_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t32_leaf1 <= 1.000647999999e+06
 T32_R_f6_k1: const_Year_Remod_Add - 1e+06 t32_leaf1 >= -998022
 T32_L_f32_k2: const_Garage_Cars + 1e+06 t32_leaf2 <= 1.000001999999e+06
 T32_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t32_leaf2 >= -999352
 T32_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t32_leaf2
   <= 1.000000999999e+06
 T32_L_f32_k3: const_Garage_Cars + 1e+06 t32_leaf3 <= 1.000001999999e+06
 T32_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t32_leaf3 >= -999352
 T32_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t32_leaf3 >= -999999
 T32_R_f32_k4: const_Garage_Cars - 1e+06 t32_leaf4 >= -999998
 T32_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t32_leaf4 <= 1.001418999999e+06
 T32_L_f18_k4: const_2nd_Flr_SF + 1e+06 t32_leaf4 <= 1.000982999999e+06
 T32_R_f32_k5: const_Garage_Cars - 1e+06 t32_leaf5 >= -999998
 T32_L_f15_k5: const_Total_Bsmt_SF + 1e+06 t32_leaf5 <= 1.001418999999e+06
 T32_R_f18_k5: const_2nd_Flr_SF - 1e+06 t32_leaf5 >= -999017
 T32_R_f32_k6: const_Garage_Cars - 1e+06 t32_leaf6 >= -999998
 T32_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t32_leaf6 >= -998581
 T32_L_f29_k6: const_Fireplaces + 1e+06 t32_leaf6 <= 1.000000999999e+06
 T32_R_f32_k7: const_Garage_Cars - 1e+06 t32_leaf7 >= -999998
 T32_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t32_leaf7 >= -998581
 T32_R_f29_k7: const_Fireplaces - 1e+06 t32_leaf7 >= -999999
 T33_onehot: t33_leaf0 + t33_leaf1 + t33_leaf2 + t33_leaf3 + t33_leaf4
   + t33_leaf5 + t33_leaf6 + t33_leaf7 = 1
 T33_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t33_leaf0 <= 1.001239999999e+06
 T33_L_f18_k0: const_2nd_Flr_SF + 1e+06 t33_leaf0 <= 1.000603999999e+06
 T33_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t33_leaf0 <= 1.000773999999e+06
 T33_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t33_leaf1 <= 1.001239999999e+06
 T33_L_f18_k1: const_2nd_Flr_SF + 1e+06 t33_leaf1 <= 1.000603999999e+06
 T33_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t33_leaf1 >= -999226
 T33_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t33_leaf2 <= 1.001239999999e+06
 T33_R_f18_k2: const_2nd_Flr_SF - 1e+06 t33_leaf2 >= -999396
 T33_L_f67_k2: const_MS_Zoning_RM + 1e+06 t33_leaf2 <= 1.000000999999e+06
 T33_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t33_leaf3 <= 1.001239999999e+06
 T33_R_f18_k3: const_2nd_Flr_SF - 1e+06 t33_leaf3 >= -999396
 T33_R_f67_k3: const_MS_Zoning_RM - 1e+06 t33_leaf3 >= -999999
 T33_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t33_leaf4 >= -998760
 T33_L_f3_k4: Overall_Qual_calc + 1e+06 t33_leaf4 <= 1.000008999999e+06
 T33_L_f14_k4: x_Bsmt_Unf_SF + 1e+06 t33_leaf4 <= 1.001213999999e+06
 T33_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t33_leaf5 >= -998760
 T33_L_f3_k5: Overall_Qual_calc + 1e+06 t33_leaf5 <= 1.000008999999e+06
 T33_R_f14_k5: x_Bsmt_Unf_SF - 1e+06 t33_leaf5 >= -998786
 T33_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t33_leaf6 >= -998760
 T33_R_f3_k6: Overall_Qual_calc - 1e+06 t33_leaf6 >= -999991
 T33_L_f0_k6: const_Lot_Frontage + 1e+06 t33_leaf6 <= 1.000113999999e+06
 T33_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t33_leaf7 >= -998760
 T33_R_f3_k7: Overall_Qual_calc - 1e+06 t33_leaf7 >= -999991
 T33_R_f0_k7: const_Lot_Frontage - 1e+06 t33_leaf7 >= -999886
 T34_onehot: t34_leaf0 + t34_leaf1 + t34_leaf2 + t34_leaf3 + t34_leaf4
   + t34_leaf5 + t34_leaf6 + t34_leaf7 = 1
 T34_L_f27_k0: x_Kitchen_Qual + 1e+06 t34_leaf0 <= 1.000002999999e+06
 T34_L_f259_k0: x_paved_drive_is_Y + 1e+06 t34_leaf0 <= 1.000000999999e+06
 T34_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t34_leaf0 <= 1.001023999999e+06
 T34_L_f27_k1: x_Kitchen_Qual + 1e+06 t34_leaf1 <= 1.000002999999e+06
 T34_L_f259_k1: x_paved_drive_is_Y + 1e+06 t34_leaf1 <= 1.000000999999e+06
 T34_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t34_leaf1 >= -998976
 T34_L_f27_k2: x_Kitchen_Qual + 1e+06 t34_leaf2 <= 1.000002999999e+06
 T34_R_f259_k2: x_paved_drive_is_Y - 1e+06 t34_leaf2 >= -999999
 T34_L_f20_k2: x_Gr_Liv_Area + 1e+06 t34_leaf2 <= 1.001497999999e+06
 T34_L_f27_k3: x_Kitchen_Qual + 1e+06 t34_leaf3 <= 1.000002999999e+06
 T34_R_f259_k3: x_paved_drive_is_Y - 1e+06 t34_leaf3 >= -999999
 T34_R_f20_k3: x_Gr_Liv_Area - 1e+06 t34_leaf3 >= -998502
 T34_R_f27_k4: x_Kitchen_Qual - 1e+06 t34_leaf4 >= -999997
 T34_L_f17_k4: x_1st_Flr_SF + 1e+06 t34_leaf4 <= 1.001054999999e+06
 T34_L_f1_k4: const_Lot_Area + 1e+06 t34_leaf4 <= 1.006372999999e+06
 T34_R_f27_k5: x_Kitchen_Qual - 1e+06 t34_leaf5 >= -999997
 T34_L_f17_k5: x_1st_Flr_SF + 1e+06 t34_leaf5 <= 1.001054999999e+06
 T34_R_f1_k5: const_Lot_Area - 1e+06 t34_leaf5 >= -993627
 T34_R_f27_k6: x_Kitchen_Qual - 1e+06 t34_leaf6 >= -999997
 T34_R_f17_k6: x_1st_Flr_SF - 1e+06 t34_leaf6 >= -998945
 T34_L_f20_k6: x_Gr_Liv_Area + 1e+06 t34_leaf6 <= 1.001557999999e+06
 T34_R_f27_k7: x_Kitchen_Qual - 1e+06 t34_leaf7 >= -999997
 T34_R_f17_k7: x_1st_Flr_SF - 1e+06 t34_leaf7 >= -998945
 T34_R_f20_k7: x_Gr_Liv_Area - 1e+06 t34_leaf7 >= -998442
 T35_onehot: t35_leaf0 + t35_leaf1 + t35_leaf2 + t35_leaf3 + t35_leaf4
   + t35_leaf5 + t35_leaf6 + t35_leaf7 = 1
 T35_L_f27_k0: x_Kitchen_Qual + 1e+06 t35_leaf0 <= 1.000002999999e+06
 T35_L_f66_k0: const_MS_Zoning_RL + 1e+06 t35_leaf0 <= 1.000000999999e+06
 T35_L_f6_k0: const_Year_Remod_Add + 1e+06 t35_leaf0 <= 1.001992999999e+06
 T35_L_f27_k1: x_Kitchen_Qual + 1e+06 t35_leaf1 <= 1.000002999999e+06
 T35_L_f66_k1: const_MS_Zoning_RL + 1e+06 t35_leaf1 <= 1.000000999999e+06
 T35_R_f6_k1: const_Year_Remod_Add - 1e+06 t35_leaf1 >= -998007
 T35_L_f27_k2: x_Kitchen_Qual + 1e+06 t35_leaf2 <= 1.000002999999e+06
 T35_R_f66_k2: const_MS_Zoning_RL - 1e+06 t35_leaf2 >= -999999
 T35_L_f6_k2: const_Year_Remod_Add + 1e+06 t35_leaf2 <= 1.001958999999e+06
 T35_L_f27_k3: x_Kitchen_Qual + 1e+06 t35_leaf3 <= 1.000002999999e+06
 T35_R_f66_k3: const_MS_Zoning_RL - 1e+06 t35_leaf3 >= -999999
 T35_R_f6_k3: const_Year_Remod_Add - 1e+06 t35_leaf3 >= -998041
 T35_R_f27_k4: x_Kitchen_Qual - 1e+06 t35_leaf4 >= -999997
 T35_L_f1_k4: const_Lot_Area + 1e+06 t35_leaf4 <= 1.011764999999e+06
 T35_L_f7_k4: x_Mas_Vnr_Area + 1e+06 t35_leaf4 <= 1.000173999999e+06
 T35_R_f27_k5: x_Kitchen_Qual - 1e+06 t35_leaf5 >= -999997
 T35_L_f1_k5: const_Lot_Area + 1e+06 t35_leaf5 <= 1.011764999999e+06
 T35_R_f7_k5: x_Mas_Vnr_Area - 1e+06 t35_leaf5 >= -999826
 T35_R_f27_k6: x_Kitchen_Qual - 1e+06 t35_leaf6 >= -999997
 T35_R_f1_k6: const_Lot_Area - 1e+06 t35_leaf6 >= -988235
 T35_L_f18_k6: const_2nd_Flr_SF + 1e+06 t35_leaf6 <= 1.001367999999e+06
 T35_R_f27_k7: x_Kitchen_Qual - 1e+06 t35_leaf7 >= -999997
 T35_R_f1_k7: const_Lot_Area - 1e+06 t35_leaf7 >= -988235
 T35_R_f18_k7: const_2nd_Flr_SF - 1e+06 t35_leaf7 >= -998632
 T36_onehot: t36_leaf0 + t36_leaf1 + t36_leaf2 + t36_leaf3 + t36_leaf4
   + t36_leaf5 + t36_leaf6 + t36_leaf7 = 1
 T36_L_f20_k0: x_Gr_Liv_Area + 1e+06 t36_leaf0 <= 1.001281999999e+06
 T36_L_f3_k0: Overall_Qual_calc + 1e+06 t36_leaf0 <= 1.000003999999e+06
 T36_L_f17_k0: x_1st_Flr_SF + 1e+06 t36_leaf0 <= 1.000867999999e+06
 T36_L_f20_k1: x_Gr_Liv_Area + 1e+06 t36_leaf1 <= 1.001281999999e+06
 T36_L_f3_k1: Overall_Qual_calc + 1e+06 t36_leaf1 <= 1.000003999999e+06
 T36_R_f17_k1: x_1st_Flr_SF - 1e+06 t36_leaf1 >= -999132
 T36_L_f20_k2: x_Gr_Liv_Area + 1e+06 t36_leaf2 <= 1.001281999999e+06
 T36_R_f3_k2: Overall_Qual_calc - 1e+06 t36_leaf2 >= -999996
 T36_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t36_leaf2 <= 1.000556999999e+06
 T36_L_f20_k3: x_Gr_Liv_Area + 1e+06 t36_leaf3 <= 1.001281999999e+06
 T36_R_f3_k3: Overall_Qual_calc - 1e+06 t36_leaf3 >= -999996
 T36_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t36_leaf3 >= -999443
 T36_R_f20_k4: x_Gr_Liv_Area - 1e+06 t36_leaf4 >= -998718
 T36_L_f20_k4: x_Gr_Liv_Area + 1e+06 t36_leaf4 <= 1.002262999999e+06
 T36_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t36_leaf4 <= 1.000783999999e+06
 T36_R_f20_k5: x_Gr_Liv_Area - 1e+06 t36_leaf5 >= -998718
 T36_L_f20_k5: x_Gr_Liv_Area + 1e+06 t36_leaf5 <= 1.002262999999e+06
 T36_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t36_leaf5 >= -999216
 T36_R_f20_k6: x_Gr_Liv_Area - 1e+06 t36_leaf6 >= -998718
 T36_R_f20_k6: x_Gr_Liv_Area - 1e+06 t36_leaf6 >= -997737
 T36_L_f17_k6: x_1st_Flr_SF + 1e+06 t36_leaf6 <= 1.001501999999e+06
 T36_R_f20_k7: x_Gr_Liv_Area - 1e+06 t36_leaf7 >= -998718
 T36_R_f20_k7: x_Gr_Liv_Area - 1e+06 t36_leaf7 >= -997737
 T36_R_f17_k7: x_1st_Flr_SF - 1e+06 t36_leaf7 >= -998498
 T37_onehot: t37_leaf0 + t37_leaf1 + t37_leaf2 + t37_leaf3 + t37_leaf4
   + t37_leaf5 + t37_leaf6 + t37_leaf7 = 1
 T37_L_f20_k0: x_Gr_Liv_Area + 1e+06 t37_leaf0 <= 1.001493999999e+06
 T37_L_f3_k0: Overall_Qual_calc + 1e+06 t37_leaf0 <= 1.000003999999e+06
 T37_L_f17_k0: x_1st_Flr_SF + 1e+06 t37_leaf0 <= 1.000907999999e+06
 T37_L_f20_k1: x_Gr_Liv_Area + 1e+06 t37_leaf1 <= 1.001493999999e+06
 T37_L_f3_k1: Overall_Qual_calc + 1e+06 t37_leaf1 <= 1.000003999999e+06
 T37_R_f17_k1: x_1st_Flr_SF - 1e+06 t37_leaf1 >= -999092
 T37_L_f20_k2: x_Gr_Liv_Area + 1e+06 t37_leaf2 <= 1.001493999999e+06
 T37_R_f3_k2: Overall_Qual_calc - 1e+06 t37_leaf2 >= -999996
 T37_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t37_leaf2 <= 1.000974999999e+06
 T37_L_f20_k3: x_Gr_Liv_Area + 1e+06 t37_leaf3 <= 1.001493999999e+06
 T37_R_f3_k3: Overall_Qual_calc - 1e+06 t37_leaf3 >= -999996
 T37_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t37_leaf3 >= -999025
 T37_R_f20_k4: x_Gr_Liv_Area - 1e+06 t37_leaf4 >= -998506
 T37_L_f6_k4: const_Year_Remod_Add + 1e+06 t37_leaf4 <= 1.001982999999e+06
 T37_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t37_leaf4 <= 1.000191999999e+06
 T37_R_f20_k5: x_Gr_Liv_Area - 1e+06 t37_leaf5 >= -998506
 T37_L_f6_k5: const_Year_Remod_Add + 1e+06 t37_leaf5 <= 1.001982999999e+06
 T37_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t37_leaf5 >= -999808
 T37_R_f20_k6: x_Gr_Liv_Area - 1e+06 t37_leaf6 >= -998506
 T37_R_f6_k6: const_Year_Remod_Add - 1e+06 t37_leaf6 >= -998017
 T37_L_f20_k6: x_Gr_Liv_Area + 1e+06 t37_leaf6 <= 1.001967999999e+06
 T37_R_f20_k7: x_Gr_Liv_Area - 1e+06 t37_leaf7 >= -998506
 T37_R_f6_k7: const_Year_Remod_Add - 1e+06 t37_leaf7 >= -998017
 T37_R_f20_k7: x_Gr_Liv_Area - 1e+06 t37_leaf7 >= -998032
 T38_onehot: t38_leaf0 + t38_leaf1 + t38_leaf2 + t38_leaf3 + t38_leaf4
   + t38_leaf5 + t38_leaf6 + t38_leaf7 = 1
 T38_L_f3_k0: Overall_Qual_calc + 1e+06 t38_leaf0 <= 1.000005999999e+06
 T38_L_f232_k0: central_air_yes + 1e+06 t38_leaf0 <= 1.000000999999e+06
 T38_L_f3_k0: Overall_Qual_calc + 1e+06 t38_leaf0 <= 1.000002999999e+06
 T38_L_f3_k1: Overall_Qual_calc + 1e+06 t38_leaf1 <= 1.000005999999e+06
 T38_L_f232_k1: central_air_yes + 1e+06 t38_leaf1 <= 1.000000999999e+06
 T38_R_f3_k1: Overall_Qual_calc - 1e+06 t38_leaf1 >= -999997
 T38_L_f3_k2: Overall_Qual_calc + 1e+06 t38_leaf2 <= 1.000005999999e+06
 T38_R_f232_k2: central_air_yes - 1e+06 t38_leaf2 >= -999999
 T38_L_f217_k2: x_b1_is_Unf + 1e+06 t38_leaf2 <= 1.000000999999e+06
 T38_L_f3_k3: Overall_Qual_calc + 1e+06 t38_leaf3 <= 1.000005999999e+06
 T38_R_f232_k3: central_air_yes - 1e+06 t38_leaf3 >= -999999
 T38_R_f217_k3: x_b1_is_Unf - 1e+06 t38_leaf3 >= -999999
 T38_R_f3_k4: Overall_Qual_calc - 1e+06 t38_leaf4 >= -999994
 T38_L_f20_k4: x_Gr_Liv_Area + 1e+06 t38_leaf4 <= 1.001799999999e+06
 T38_L_f20_k4: x_Gr_Liv_Area + 1e+06 t38_leaf4 <= 1.001143999999e+06
 T38_R_f3_k5: Overall_Qual_calc - 1e+06 t38_leaf5 >= -999994
 T38_L_f20_k5: x_Gr_Liv_Area + 1e+06 t38_leaf5 <= 1.001799999999e+06
 T38_R_f20_k5: x_Gr_Liv_Area - 1e+06 t38_leaf5 >= -998856
 T38_R_f3_k6: Overall_Qual_calc - 1e+06 t38_leaf6 >= -999994
 T38_R_f20_k6: x_Gr_Liv_Area - 1e+06 t38_leaf6 >= -998200
 T38_L_f20_k6: x_Gr_Liv_Area + 1e+06 t38_leaf6 <= 1.002326999999e+06
 T38_R_f3_k7: Overall_Qual_calc - 1e+06 t38_leaf7 >= -999994
 T38_R_f20_k7: x_Gr_Liv_Area - 1e+06 t38_leaf7 >= -998200
 T38_R_f20_k7: x_Gr_Liv_Area - 1e+06 t38_leaf7 >= -997673
 T39_onehot: t39_leaf0 + t39_leaf1 + t39_leaf2 + t39_leaf3 + t39_leaf4
   + t39_leaf5 + t39_leaf6 + t39_leaf7 = 1
 T39_L_f20_k0: x_Gr_Liv_Area + 1e+06 t39_leaf0 <= 1.001307999999e+06
 T39_L_f5_k0: const_Year_Built + 1e+06 t39_leaf0 <= 1.001940999999e+06
 T39_L_f4_k0: const_Overall_Cond + 1e+06 t39_leaf0 <= 1.000006999999e+06
 T39_L_f20_k1: x_Gr_Liv_Area + 1e+06 t39_leaf1 <= 1.001307999999e+06
 T39_L_f5_k1: const_Year_Built + 1e+06 t39_leaf1 <= 1.001940999999e+06
 T39_R_f4_k1: const_Overall_Cond - 1e+06 t39_leaf1 >= -999993
 T39_L_f20_k2: x_Gr_Liv_Area + 1e+06 t39_leaf2 <= 1.001307999999e+06
 T39_R_f5_k2: const_Year_Built - 1e+06 t39_leaf2 >= -998059
 T39_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t39_leaf2
   <= 1.000000999999e+06
 T39_L_f20_k3: x_Gr_Liv_Area + 1e+06 t39_leaf3 <= 1.001307999999e+06
 T39_R_f5_k3: const_Year_Built - 1e+06 t39_leaf3 >= -998059
 T39_R_f21_k3: const_Bsmt_Full_Bath - 1e+06 t39_leaf3 >= -999999
 T39_R_f20_k4: x_Gr_Liv_Area - 1e+06 t39_leaf4 >= -998692
 T39_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t39_leaf4 <= 1.000855999999e+06
 T39_L_f4_k4: const_Overall_Cond + 1e+06 t39_leaf4 <= 1.000004999999e+06
 T39_R_f20_k5: x_Gr_Liv_Area - 1e+06 t39_leaf5 >= -998692
 T39_L_f12_k5: x_BsmtFin_SF_1 + 1e+06 t39_leaf5 <= 1.000855999999e+06
 T39_R_f4_k5: const_Overall_Cond - 1e+06 t39_leaf5 >= -999995
 T39_R_f20_k6: x_Gr_Liv_Area - 1e+06 t39_leaf6 >= -998692
 T39_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t39_leaf6 >= -999144
 T39_L_f3_k6: Overall_Qual_calc + 1e+06 t39_leaf6 <= 1.000008999999e+06
 T39_R_f20_k7: x_Gr_Liv_Area - 1e+06 t39_leaf7 >= -998692
 T39_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t39_leaf7 >= -999144
 T39_R_f3_k7: Overall_Qual_calc - 1e+06 t39_leaf7 >= -999991
 T40_onehot: t40_leaf0 + t40_leaf1 + t40_leaf2 + t40_leaf3 + t40_leaf4
   + t40_leaf5 + t40_leaf6 + t40_leaf7 = 1
 T40_L_f6_k0: const_Year_Remod_Add + 1e+06 t40_leaf0 <= 1.001984999999e+06
 T40_L_f35_k0: x_Garage_Cond + 1e+06 t40_leaf0 <= 1.000001999999e+06
 T40_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t40_leaf0 <= 1.000857999999e+06
 T40_L_f6_k1: const_Year_Remod_Add + 1e+06 t40_leaf1 <= 1.001984999999e+06
 T40_L_f35_k1: x_Garage_Cond + 1e+06 t40_leaf1 <= 1.000001999999e+06
 T40_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t40_leaf1 >= -999142
 T40_L_f6_k2: const_Year_Remod_Add + 1e+06 t40_leaf2 <= 1.001984999999e+06
 T40_R_f35_k2: x_Garage_Cond - 1e+06 t40_leaf2 >= -999998
 T40_L_f0_k2: const_Lot_Frontage + 1e+06 t40_leaf2 <= 1.000065999999e+06
 T40_L_f6_k3: const_Year_Remod_Add + 1e+06 t40_leaf3 <= 1.001984999999e+06
 T40_R_f35_k3: x_Garage_Cond - 1e+06 t40_leaf3 >= -999998
 T40_R_f0_k3: const_Lot_Frontage - 1e+06 t40_leaf3 >= -999934
 T40_R_f6_k4: const_Year_Remod_Add - 1e+06 t40_leaf4 >= -998015
 T40_L_f20_k4: x_Gr_Liv_Area + 1e+06 t40_leaf4 <= 1.001637999999e+06
 T40_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t40_leaf4 <= 1.001054999999e+06
 T40_R_f6_k5: const_Year_Remod_Add - 1e+06 t40_leaf5 >= -998015
 T40_L_f20_k5: x_Gr_Liv_Area + 1e+06 t40_leaf5 <= 1.001637999999e+06
 T40_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t40_leaf5 >= -998945
 T40_R_f6_k6: const_Year_Remod_Add - 1e+06 t40_leaf6 >= -998015
 T40_R_f20_k6: x_Gr_Liv_Area - 1e+06 t40_leaf6 >= -998362
 T40_L_f18_k6: const_2nd_Flr_SF + 1e+06 t40_leaf6 <= 1.001367999999e+06
 T40_R_f6_k7: const_Year_Remod_Add - 1e+06 t40_leaf7 >= -998015
 T40_R_f20_k7: x_Gr_Liv_Area - 1e+06 t40_leaf7 >= -998362
 T40_R_f18_k7: const_2nd_Flr_SF - 1e+06 t40_leaf7 >= -998632
 T41_onehot: t41_leaf0 + t41_leaf1 + t41_leaf2 + t41_leaf3 + t41_leaf4
   + t41_leaf5 + t41_leaf6 + t41_leaf7 = 1
 T41_L_f3_k0: Overall_Qual_calc + 1e+06 t41_leaf0 <= 1.000006999999e+06
 T41_L_f46_k0: const_MS_SubClass_30 + 1e+06 t41_leaf0
   <= 1.000000999999e+06
 T41_L_f3_k0: Overall_Qual_calc + 1e+06 t41_leaf0 <= 1.000004999999e+06
 T41_L_f3_k1: Overall_Qual_calc + 1e+06 t41_leaf1 <= 1.000006999999e+06
 T41_L_f46_k1: const_MS_SubClass_30 + 1e+06 t41_leaf1
   <= 1.000000999999e+06
 T41_R_f3_k1: Overall_Qual_calc - 1e+06 t41_leaf1 >= -999995
 T41_L_f3_k2: Overall_Qual_calc + 1e+06 t41_leaf2 <= 1.000006999999e+06
 T41_R_f46_k2: const_MS_SubClass_30 - 1e+06 t41_leaf2 >= -999999
 T41_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t41_leaf2 <= 1.000205999999e+06
 T41_L_f3_k3: Overall_Qual_calc + 1e+06 t41_leaf3 <= 1.000006999999e+06
 T41_R_f46_k3: const_MS_SubClass_30 - 1e+06 t41_leaf3 >= -999999
 T41_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t41_leaf3 >= -999794
 T41_R_f3_k4: Overall_Qual_calc - 1e+06 t41_leaf4 >= -999993
 T41_L_f3_k4: Overall_Qual_calc + 1e+06 t41_leaf4 <= 1.000008999999e+06
 T41_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t41_leaf4 <= 1.001096999999e+06
 T41_R_f3_k5: Overall_Qual_calc - 1e+06 t41_leaf5 >= -999993
 T41_L_f3_k5: Overall_Qual_calc + 1e+06 t41_leaf5 <= 1.000008999999e+06
 T41_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t41_leaf5 >= -998903
 T41_R_f3_k6: Overall_Qual_calc - 1e+06 t41_leaf6 >= -999993
 T41_R_f3_k6: Overall_Qual_calc - 1e+06 t41_leaf6 >= -999991
 T41_L_f37_k6: x_Open_Porch_SF + 1e+06 t41_leaf6 <= 1.000235999999e+06
 T41_R_f3_k7: Overall_Qual_calc - 1e+06 t41_leaf7 >= -999993
 T41_R_f3_k7: Overall_Qual_calc - 1e+06 t41_leaf7 >= -999991
 T41_R_f37_k7: x_Open_Porch_SF - 1e+06 t41_leaf7 >= -999764
 T42_onehot: t42_leaf0 + t42_leaf1 + t42_leaf2 + t42_leaf3 + t42_leaf4
   + t42_leaf5 + t42_leaf6 + t42_leaf7 = 1
 T42_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t42_leaf0 <= 1.001199999999e+06
 T42_L_f35_k0: x_Garage_Cond + 1e+06 t42_leaf0 <= 1.000001999999e+06
 T42_L_f9_k0: x_Exter_Cond + 1e+06 t42_leaf0 <= 1.000001999999e+06
 T42_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t42_leaf1 <= 1.001199999999e+06
 T42_L_f35_k1: x_Garage_Cond + 1e+06 t42_leaf1 <= 1.000001999999e+06
 T42_R_f9_k1: x_Exter_Cond - 1e+06 t42_leaf1 >= -999998
 T42_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t42_leaf2 <= 1.001199999999e+06
 T42_R_f35_k2: x_Garage_Cond - 1e+06 t42_leaf2 >= -999998
 T42_L_f20_k2: x_Gr_Liv_Area + 1e+06 t42_leaf2 <= 1.001765999999e+06
 T42_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t42_leaf3 <= 1.001199999999e+06
 T42_R_f35_k3: x_Garage_Cond - 1e+06 t42_leaf3 >= -999998
 T42_R_f20_k3: x_Gr_Liv_Area - 1e+06 t42_leaf3 >= -998234
 T42_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t42_leaf4 >= -998800
 T42_L_f20_k4: x_Gr_Liv_Area + 1e+06 t42_leaf4 <= 1.002051999999e+06
 T42_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t42_leaf4 <= 1.000323999999e+06
 T42_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t42_leaf5 >= -998800
 T42_L_f20_k5: x_Gr_Liv_Area + 1e+06 t42_leaf5 <= 1.002051999999e+06
 T42_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t42_leaf5 >= -999676
 T42_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t42_leaf6 >= -998800
 T42_R_f20_k6: x_Gr_Liv_Area - 1e+06 t42_leaf6 >= -997948
 T42_L_f37_k6: x_Open_Porch_SF + 1e+06 t42_leaf6 <= 1.000229999999e+06
 T42_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t42_leaf7 >= -998800
 T42_R_f20_k7: x_Gr_Liv_Area - 1e+06 t42_leaf7 >= -997948
 T42_R_f37_k7: x_Open_Porch_SF - 1e+06 t42_leaf7 >= -999770
 T43_onehot: t43_leaf0 + t43_leaf1 + t43_leaf2 + t43_leaf3 + t43_leaf4
   + t43_leaf5 + t43_leaf6 + t43_leaf7 = 1
 T43_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t43_leaf0 <= 1.001981999999e+06
 T43_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t43_leaf0 <= 1.000222999999e+06
 T43_L_f4_k0: const_Overall_Cond + 1e+06 t43_leaf0 <= 1.000004999999e+06
 T43_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t43_leaf1 <= 1.001981999999e+06
 T43_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t43_leaf1 <= 1.000222999999e+06
 T43_R_f4_k1: const_Overall_Cond - 1e+06 t43_leaf1 >= -999995
 T43_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t43_leaf2 <= 1.001981999999e+06
 T43_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t43_leaf2 >= -999777
 T43_L_f4_k2: const_Overall_Cond + 1e+06 t43_leaf2 <= 1.000005999999e+06
 T43_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t43_leaf3 <= 1.001981999999e+06
 T43_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t43_leaf3 >= -999777
 T43_R_f4_k3: const_Overall_Cond - 1e+06 t43_leaf3 >= -999994
 T43_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t43_leaf4 >= -998018
 T43_L_f3_k4: Overall_Qual_calc + 1e+06 t43_leaf4 <= 1.000007999999e+06
 T43_L_f37_k4: x_Open_Porch_SF + 1e+06 t43_leaf4 <= 1.000041999999e+06
 T43_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t43_leaf5 >= -998018
 T43_L_f3_k5: Overall_Qual_calc + 1e+06 t43_leaf5 <= 1.000007999999e+06
 T43_R_f37_k5: x_Open_Porch_SF - 1e+06 t43_leaf5 >= -999958
 T43_R_f31_k6: const_Garage_Yr_Blt - 1e+06 t43_leaf6 >= -998018
 T43_R_f3_k6: Overall_Qual_calc - 1e+06 t43_leaf6 >= -999992
 T43_L_f37_k6: x_Open_Porch_SF + 1e+06 t43_leaf6 <= 1.000273999999e+06
 T43_R_f31_k7: const_Garage_Yr_Blt - 1e+06 t43_leaf7 >= -998018
 T43_R_f3_k7: Overall_Qual_calc - 1e+06 t43_leaf7 >= -999992
 T43_R_f37_k7: x_Open_Porch_SF - 1e+06 t43_leaf7 >= -999726
 T44_onehot: t44_leaf0 + t44_leaf1 + t44_leaf2 + t44_leaf3 + t44_leaf4
   + t44_leaf5 + t44_leaf6 + t44_leaf7 = 1
 T44_L_f10_k0: const_Bsmt_Qual + 1e+06 t44_leaf0 <= 1.000002999999e+06
 T44_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t44_leaf0 <= 1.000629999999e+06
 T44_L_f20_k0: x_Gr_Liv_Area + 1e+06 t44_leaf0 <= 1.001210999999e+06
 T44_L_f10_k1: const_Bsmt_Qual + 1e+06 t44_leaf1 <= 1.000002999999e+06
 T44_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t44_leaf1 <= 1.000629999999e+06
 T44_R_f20_k1: x_Gr_Liv_Area - 1e+06 t44_leaf1 >= -998789
 T44_L_f10_k2: const_Bsmt_Qual + 1e+06 t44_leaf2 <= 1.000002999999e+06
 T44_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t44_leaf2 >= -999370
 T44_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t44_leaf2 <= 1.001920999999e+06
 T44_L_f10_k3: const_Bsmt_Qual + 1e+06 t44_leaf3 <= 1.000002999999e+06
 T44_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t44_leaf3 >= -999370
 T44_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t44_leaf3 >= -998079
 T44_R_f10_k4: const_Bsmt_Qual - 1e+06 t44_leaf4 >= -999997
 T44_L_f17_k4: x_1st_Flr_SF + 1e+06 t44_leaf4 <= 1.001651999999e+06
 T44_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t44_leaf4 <= 1.000925999999e+06
 T44_R_f10_k5: const_Bsmt_Qual - 1e+06 t44_leaf5 >= -999997
 T44_L_f17_k5: x_1st_Flr_SF + 1e+06 t44_leaf5 <= 1.001651999999e+06
 T44_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t44_leaf5 >= -999074
 T44_R_f10_k6: const_Bsmt_Qual - 1e+06 t44_leaf6 >= -999997
 T44_R_f17_k6: x_1st_Flr_SF - 1e+06 t44_leaf6 >= -998348
 T44_L_f20_k6: x_Gr_Liv_Area + 1e+06 t44_leaf6 <= 1.002027999999e+06
 T44_R_f10_k7: const_Bsmt_Qual - 1e+06 t44_leaf7 >= -999997
 T44_R_f17_k7: x_1st_Flr_SF - 1e+06 t44_leaf7 >= -998348
 T44_R_f20_k7: x_Gr_Liv_Area - 1e+06 t44_leaf7 >= -997972
 T45_onehot: t45_leaf0 + t45_leaf1 + t45_leaf2 + t45_leaf3 + t45_leaf4
   + t45_leaf5 + t45_leaf6 + t45_leaf7 = 1
 T45_L_f3_k0: Overall_Qual_calc + 1e+06 t45_leaf0 <= 1.000005999999e+06
 T45_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t45_leaf0 <= 1.000074999999e+06
 T45_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t45_leaf0
   <= 1.000000999999e+06
 T45_L_f3_k1: Overall_Qual_calc + 1e+06 t45_leaf1 <= 1.000005999999e+06
 T45_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t45_leaf1 <= 1.000074999999e+06
 T45_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t45_leaf1 >= -999999
 T45_L_f3_k2: Overall_Qual_calc + 1e+06 t45_leaf2 <= 1.000005999999e+06
 T45_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t45_leaf2 >= -999925
 T45_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t45_leaf2 <= 1.001355999999e+06
 T45_L_f3_k3: Overall_Qual_calc + 1e+06 t45_leaf3 <= 1.000005999999e+06
 T45_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t45_leaf3 >= -999925
 T45_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t45_leaf3 >= -998644
 T45_R_f3_k4: Overall_Qual_calc - 1e+06 t45_leaf4 >= -999994
 T45_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t45_leaf4 <= 1.000726999999e+06
 T45_L_f32_k4: const_Garage_Cars + 1e+06 t45_leaf4 <= 1.000001999999e+06
 T45_R_f3_k5: Overall_Qual_calc - 1e+06 t45_leaf5 >= -999994
 T45_L_f12_k5: x_BsmtFin_SF_1 + 1e+06 t45_leaf5 <= 1.000726999999e+06
 T45_R_f32_k5: const_Garage_Cars - 1e+06 t45_leaf5 >= -999998
 T45_R_f3_k6: Overall_Qual_calc - 1e+06 t45_leaf6 >= -999994
 T45_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t45_leaf6 >= -999273
 T45_L_f3_k6: Overall_Qual_calc + 1e+06 t45_leaf6 <= 1.000007999999e+06
 T45_R_f3_k7: Overall_Qual_calc - 1e+06 t45_leaf7 >= -999994
 T45_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t45_leaf7 >= -999273
 T45_R_f3_k7: Overall_Qual_calc - 1e+06 t45_leaf7 >= -999992
 T46_onehot: t46_leaf0 + t46_leaf1 + t46_leaf2 + t46_leaf3 + t46_leaf4
   + t46_leaf5 + t46_leaf6 + t46_leaf7 = 1
 T46_L_f17_k0: x_1st_Flr_SF + 1e+06 t46_leaf0 <= 1.001154999999e+06
 T46_L_f35_k0: x_Garage_Cond + 1e+06 t46_leaf0 <= 1.000001999999e+06
 T46_L_f46_k0: const_MS_SubClass_30 + 1e+06 t46_leaf0
   <= 1.000000999999e+06
 T46_L_f17_k1: x_1st_Flr_SF + 1e+06 t46_leaf1 <= 1.001154999999e+06
 T46_L_f35_k1: x_Garage_Cond + 1e+06 t46_leaf1 <= 1.000001999999e+06
 T46_R_f46_k1: const_MS_SubClass_30 - 1e+06 t46_leaf1 >= -999999
 T46_L_f17_k2: x_1st_Flr_SF + 1e+06 t46_leaf2 <= 1.001154999999e+06
 T46_R_f35_k2: x_Garage_Cond - 1e+06 t46_leaf2 >= -999998
 T46_L_f18_k2: const_2nd_Flr_SF + 1e+06 t46_leaf2 <= 1.000675999999e+06
 T46_L_f17_k3: x_1st_Flr_SF + 1e+06 t46_leaf3 <= 1.001154999999e+06
 T46_R_f35_k3: x_Garage_Cond - 1e+06 t46_leaf3 >= -999998
 T46_R_f18_k3: const_2nd_Flr_SF - 1e+06 t46_leaf3 >= -999324
 T46_R_f17_k4: x_1st_Flr_SF - 1e+06 t46_leaf4 >= -998845
 T46_L_f18_k4: const_2nd_Flr_SF + 1e+06 t46_leaf4 <= 1.001167999999e+06
 T46_L_f6_k4: const_Year_Remod_Add + 1e+06 t46_leaf4 <= 1.001992999999e+06
 T46_R_f17_k5: x_1st_Flr_SF - 1e+06 t46_leaf5 >= -998845
 T46_L_f18_k5: const_2nd_Flr_SF + 1e+06 t46_leaf5 <= 1.001167999999e+06
 T46_R_f6_k5: const_Year_Remod_Add - 1e+06 t46_leaf5 >= -998007
 T46_R_f17_k6: x_1st_Flr_SF - 1e+06 t46_leaf6 >= -998845
 T46_R_f18_k6: const_2nd_Flr_SF - 1e+06 t46_leaf6 >= -998832
 T46_L_f26_k6: x_Kitchen_AbvGr + 1e+06 t46_leaf6 <= 1.000001999999e+06
 T46_R_f17_k7: x_1st_Flr_SF - 1e+06 t46_leaf7 >= -998845
 T46_R_f18_k7: const_2nd_Flr_SF - 1e+06 t46_leaf7 >= -998832
 T46_R_f26_k7: x_Kitchen_AbvGr - 1e+06 t46_leaf7 >= -999998
 T47_onehot: t47_leaf0 + t47_leaf1 + t47_leaf2 + t47_leaf3 + t47_leaf4
   + t47_leaf5 + t47_leaf6 + t47_leaf7 = 1
 T47_L_f6_k0: const_Year_Remod_Add + 1e+06 t47_leaf0 <= 1.001980999999e+06
 T47_L_f4_k0: const_Overall_Cond + 1e+06 t47_leaf0 <= 1.000004999999e+06
 T47_L_f6_k0: const_Year_Remod_Add + 1e+06 t47_leaf0 <= 1.001958999999e+06
 T47_L_f6_k1: const_Year_Remod_Add + 1e+06 t47_leaf1 <= 1.001980999999e+06
 T47_L_f4_k1: const_Overall_Cond + 1e+06 t47_leaf1 <= 1.000004999999e+06
 T47_R_f6_k1: const_Year_Remod_Add - 1e+06 t47_leaf1 >= -998041
 T47_L_f6_k2: const_Year_Remod_Add + 1e+06 t47_leaf2 <= 1.001980999999e+06
 T47_R_f4_k2: const_Overall_Cond - 1e+06 t47_leaf2 >= -999995
 T47_L_f0_k2: const_Lot_Frontage + 1e+06 t47_leaf2 <= 1.000068999999e+06
 T47_L_f6_k3: const_Year_Remod_Add + 1e+06 t47_leaf3 <= 1.001980999999e+06
 T47_R_f4_k3: const_Overall_Cond - 1e+06 t47_leaf3 >= -999995
 T47_R_f0_k3: const_Lot_Frontage - 1e+06 t47_leaf3 >= -999931
 T47_R_f6_k4: const_Year_Remod_Add - 1e+06 t47_leaf4 >= -998019
 T47_L_f30_k4: x_Fireplace_Qu + 1e+06 t47_leaf4 <= 1.000001999999e+06
 T47_L_f0_k4: const_Lot_Frontage + 1e+06 t47_leaf4 <= 1.000060999999e+06
 T47_R_f6_k5: const_Year_Remod_Add - 1e+06 t47_leaf5 >= -998019
 T47_L_f30_k5: x_Fireplace_Qu + 1e+06 t47_leaf5 <= 1.000001999999e+06
 T47_R_f0_k5: const_Lot_Frontage - 1e+06 t47_leaf5 >= -999939
 T47_R_f6_k6: const_Year_Remod_Add - 1e+06 t47_leaf6 >= -998019
 T47_R_f30_k6: x_Fireplace_Qu - 1e+06 t47_leaf6 >= -999998
 T47_L_f3_k6: Overall_Qual_calc + 1e+06 t47_leaf6 <= 1.000008999999e+06
 T47_R_f6_k7: const_Year_Remod_Add - 1e+06 t47_leaf7 >= -998019
 T47_R_f30_k7: x_Fireplace_Qu - 1e+06 t47_leaf7 >= -999998
 T47_R_f3_k7: Overall_Qual_calc - 1e+06 t47_leaf7 >= -999991
 T48_onehot: t48_leaf0 + t48_leaf1 + t48_leaf2 + t48_leaf3 + t48_leaf4
   + t48_leaf5 + t48_leaf6 + t48_leaf7 = 1
 T48_L_f20_k0: x_Gr_Liv_Area + 1e+06 t48_leaf0 <= 1.001711999999e+06
 T48_L_f257_k0: x_paved_drive_is_N + 1e+06 t48_leaf0 <= 1.000000999999e+06
 T48_L_f33_k0: x_Garage_Area + 1e+06 t48_leaf0 <= 1.000383999999e+06
 T48_L_f20_k1: x_Gr_Liv_Area + 1e+06 t48_leaf1 <= 1.001711999999e+06
 T48_L_f257_k1: x_paved_drive_is_N + 1e+06 t48_leaf1 <= 1.000000999999e+06
 T48_R_f33_k1: x_Garage_Area - 1e+06 t48_leaf1 >= -999616
 T48_L_f20_k2: x_Gr_Liv_Area + 1e+06 t48_leaf2 <= 1.001711999999e+06
 T48_R_f257_k2: x_paved_drive_is_N - 1e+06 t48_leaf2 >= -999999
 T48_L_f297_k2: const_Sale_Condition_Normal + 1e+06 t48_leaf2
   <= 1.000000999999e+06
 T48_L_f20_k3: x_Gr_Liv_Area + 1e+06 t48_leaf3 <= 1.001711999999e+06
 T48_R_f257_k3: x_paved_drive_is_N - 1e+06 t48_leaf3 >= -999999
 T48_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t48_leaf3 >= -999999
 T48_R_f20_k4: x_Gr_Liv_Area - 1e+06 t48_leaf4 >= -998288
 T48_L_f213_k4: x_b1_is_GLQ + 1e+06 t48_leaf4 <= 1.000000999999e+06
 T48_L_f44_k4: const_Yr_Sold + 1e+06 t48_leaf4 <= 1.002007999999e+06
 T48_R_f20_k5: x_Gr_Liv_Area - 1e+06 t48_leaf5 >= -998288
 T48_L_f213_k5: x_b1_is_GLQ + 1e+06 t48_leaf5 <= 1.000000999999e+06
 T48_R_f44_k5: const_Yr_Sold - 1e+06 t48_leaf5 >= -997992
 T48_R_f20_k6: x_Gr_Liv_Area - 1e+06 t48_leaf6 >= -998288
 T48_R_f213_k6: x_b1_is_GLQ - 1e+06 t48_leaf6 >= -999999
 T48_L_f15_k6: const_Total_Bsmt_SF + 1e+06 t48_leaf6 <= 1.001849999999e+06
 T48_R_f20_k7: x_Gr_Liv_Area - 1e+06 t48_leaf7 >= -998288
 T48_R_f213_k7: x_b1_is_GLQ - 1e+06 t48_leaf7 >= -999999
 T48_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t48_leaf7 >= -998150
 T49_onehot: t49_leaf0 + t49_leaf1 + t49_leaf2 + t49_leaf3 + t49_leaf4
   + t49_leaf5 + t49_leaf6 + t49_leaf7 = 1
 T49_L_f20_k0: x_Gr_Liv_Area + 1e+06 t49_leaf0 <= 1.001493999999e+06
 T49_L_f259_k0: x_paved_drive_is_Y + 1e+06 t49_leaf0 <= 1.000000999999e+06
 T49_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t49_leaf0
   <= 1.000000999999e+06
 T49_L_f20_k1: x_Gr_Liv_Area + 1e+06 t49_leaf1 <= 1.001493999999e+06
 T49_L_f259_k1: x_paved_drive_is_Y + 1e+06 t49_leaf1 <= 1.000000999999e+06
 T49_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t49_leaf1 >= -999999
 T49_L_f20_k2: x_Gr_Liv_Area + 1e+06 t49_leaf2 <= 1.001493999999e+06
 T49_R_f259_k2: x_paved_drive_is_Y - 1e+06 t49_leaf2 >= -999999
 T49_L_f17_k2: x_1st_Flr_SF + 1e+06 t49_leaf2 <= 1.001063999999e+06
 T49_L_f20_k3: x_Gr_Liv_Area + 1e+06 t49_leaf3 <= 1.001493999999e+06
 T49_R_f259_k3: x_paved_drive_is_Y - 1e+06 t49_leaf3 >= -999999
 T49_R_f17_k3: x_1st_Flr_SF - 1e+06 t49_leaf3 >= -998936
 T49_R_f20_k4: x_Gr_Liv_Area - 1e+06 t49_leaf4 >= -998506
 T49_L_f109_k4: const_Neighborhood_OldTown + 1e+06 t49_leaf4
   <= 1.000000999999e+06
 T49_L_f44_k4: const_Yr_Sold + 1e+06 t49_leaf4 <= 1.002007999999e+06
 T49_R_f20_k5: x_Gr_Liv_Area - 1e+06 t49_leaf5 >= -998506
 T49_L_f109_k5: const_Neighborhood_OldTown + 1e+06 t49_leaf5
   <= 1.000000999999e+06
 T49_R_f44_k5: const_Yr_Sold - 1e+06 t49_leaf5 >= -997992
 T49_R_f20_k6: x_Gr_Liv_Area - 1e+06 t49_leaf6 >= -998506
 T49_R_f109_k6: const_Neighborhood_OldTown - 1e+06 t49_leaf6 >= -999999
 T49_L_f293_k6: const_Sale_Condition_Abnorml + 1e+06 t49_leaf6
   <= 1.000000999999e+06
 T49_R_f20_k7: x_Gr_Liv_Area - 1e+06 t49_leaf7 >= -998506
 T49_R_f109_k7: const_Neighborhood_OldTown - 1e+06 t49_leaf7 >= -999999
 T49_R_f293_k7: const_Sale_Condition_Abnorml - 1e+06 t49_leaf7 >= -999999
 T50_onehot: t50_leaf0 + t50_leaf1 + t50_leaf2 + t50_leaf3 + t50_leaf4
   + t50_leaf5 + t50_leaf6 = 1
 T50_L_f20_k0: x_Gr_Liv_Area + 1e+06 t50_leaf0 <= 1.001182999999e+06
 T50_L_f3_k0: Overall_Qual_calc + 1e+06 t50_leaf0 <= 1.000003999999e+06
 T50_L_f20_k1: x_Gr_Liv_Area + 1e+06 t50_leaf1 <= 1.001182999999e+06
 T50_R_f3_k1: Overall_Qual_calc - 1e+06 t50_leaf1 >= -999996
 T50_L_f66_k1: const_MS_Zoning_RL + 1e+06 t50_leaf1 <= 1.000000999999e+06
 T50_L_f20_k2: x_Gr_Liv_Area + 1e+06 t50_leaf2 <= 1.001182999999e+06
 T50_R_f3_k2: Overall_Qual_calc - 1e+06 t50_leaf2 >= -999996
 T50_R_f66_k2: const_MS_Zoning_RL - 1e+06 t50_leaf2 >= -999999
 T50_R_f20_k3: x_Gr_Liv_Area - 1e+06 t50_leaf3 >= -998817
 T50_L_f1_k3: const_Lot_Area + 1e+06 t50_leaf3 <= 1.011427999999e+06
 T50_L_f5_k3: const_Year_Built + 1e+06 t50_leaf3 <= 1.001985999999e+06
 T50_R_f20_k4: x_Gr_Liv_Area - 1e+06 t50_leaf4 >= -998817
 T50_L_f1_k4: const_Lot_Area + 1e+06 t50_leaf4 <= 1.011427999999e+06
 T50_R_f5_k4: const_Year_Built - 1e+06 t50_leaf4 >= -998014
 T50_R_f20_k5: x_Gr_Liv_Area - 1e+06 t50_leaf5 >= -998817
 T50_R_f1_k5: const_Lot_Area - 1e+06 t50_leaf5 >= -988572
 T50_L_f20_k5: x_Gr_Liv_Area + 1e+06 t50_leaf5 <= 1.001733999999e+06
 T50_R_f20_k6: x_Gr_Liv_Area - 1e+06 t50_leaf6 >= -998817
 T50_R_f1_k6: const_Lot_Area - 1e+06 t50_leaf6 >= -988572
 T50_R_f20_k6: x_Gr_Liv_Area - 1e+06 t50_leaf6 >= -998266
 T51_onehot: t51_leaf0 + t51_leaf1 + t51_leaf2 + t51_leaf3 + t51_leaf4
   + t51_leaf5 + t51_leaf6 = 1
 T51_L_f3_k0: Overall_Qual_calc + 1e+06 t51_leaf0 <= 1.000005999999e+06
 T51_L_f20_k0: x_Gr_Liv_Area + 1e+06 t51_leaf0 <= 1.000767999999e+06
 T51_L_f3_k1: Overall_Qual_calc + 1e+06 t51_leaf1 <= 1.000005999999e+06
 T51_R_f20_k1: x_Gr_Liv_Area - 1e+06 t51_leaf1 >= -999232
 T51_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t51_leaf1 <= 1.000026999999e+06
 T51_L_f3_k2: Overall_Qual_calc + 1e+06 t51_leaf2 <= 1.000005999999e+06
 T51_R_f20_k2: x_Gr_Liv_Area - 1e+06 t51_leaf2 >= -999232
 T51_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t51_leaf2 >= -999973
 T51_R_f3_k3: Overall_Qual_calc - 1e+06 t51_leaf3 >= -999994
 T51_L_f20_k3: x_Gr_Liv_Area + 1e+06 t51_leaf3 <= 1.002326999999e+06
 T51_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t51_leaf3 <= 1.000704999999e+06
 T51_R_f3_k4: Overall_Qual_calc - 1e+06 t51_leaf4 >= -999994
 T51_L_f20_k4: x_Gr_Liv_Area + 1e+06 t51_leaf4 <= 1.002326999999e+06
 T51_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t51_leaf4 >= -999295
 T51_R_f3_k5: Overall_Qual_calc - 1e+06 t51_leaf5 >= -999994
 T51_R_f20_k5: x_Gr_Liv_Area - 1e+06 t51_leaf5 >= -997673
 T51_L_f37_k5: x_Open_Porch_SF + 1e+06 t51_leaf5 <= 1.000273999999e+06
 T51_R_f3_k6: Overall_Qual_calc - 1e+06 t51_leaf6 >= -999994
 T51_R_f20_k6: x_Gr_Liv_Area - 1e+06 t51_leaf6 >= -997673
 T51_R_f37_k6: x_Open_Porch_SF - 1e+06 t51_leaf6 >= -999726
 T52_onehot: t52_leaf0 + t52_leaf1 + t52_leaf2 + t52_leaf3 + t52_leaf4
   + t52_leaf5 + t52_leaf6 + t52_leaf7 = 1
 T52_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t52_leaf0 <= 1.001418999999e+06
 T52_L_f4_k0: const_Overall_Cond + 1e+06 t52_leaf0 <= 1.000004999999e+06
 T52_L_f4_k0: const_Overall_Cond + 1e+06 t52_leaf0 <= 1.000003999999e+06
 T52_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t52_leaf1 <= 1.001418999999e+06
 T52_L_f4_k1: const_Overall_Cond + 1e+06 t52_leaf1 <= 1.000004999999e+06
 T52_R_f4_k1: const_Overall_Cond - 1e+06 t52_leaf1 >= -999996
 T52_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t52_leaf2 <= 1.001418999999e+06
 T52_R_f4_k2: const_Overall_Cond - 1e+06 t52_leaf2 >= -999995
 T52_L_f20_k2: x_Gr_Liv_Area + 1e+06 t52_leaf2 <= 1.001719999999e+06
 T52_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t52_leaf3 <= 1.001418999999e+06
 T52_R_f4_k3: const_Overall_Cond - 1e+06 t52_leaf3 >= -999995
 T52_R_f20_k3: x_Gr_Liv_Area - 1e+06 t52_leaf3 >= -998280
 T52_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t52_leaf4 >= -998581
 T52_L_f20_k4: x_Gr_Liv_Area + 1e+06 t52_leaf4 <= 1.002027999999e+06
 T52_L_f21_k4: const_Bsmt_Full_Bath + 1e+06 t52_leaf4
   <= 1.000000999999e+06
 T52_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t52_leaf5 >= -998581
 T52_L_f20_k5: x_Gr_Liv_Area + 1e+06 t52_leaf5 <= 1.002027999999e+06
 T52_R_f21_k5: const_Bsmt_Full_Bath - 1e+06 t52_leaf5 >= -999999
 T52_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t52_leaf6 >= -998581
 T52_R_f20_k6: x_Gr_Liv_Area - 1e+06 t52_leaf6 >= -997972
 T52_L_f1_k6: const_Lot_Area + 1e+06 t52_leaf6 <= 1.013687999999e+06
 T52_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t52_leaf7 >= -998581
 T52_R_f20_k7: x_Gr_Liv_Area - 1e+06 t52_leaf7 >= -997972
 T52_R_f1_k7: const_Lot_Area - 1e+06 t52_leaf7 >= -986312
 T53_onehot: t53_leaf0 + t53_leaf1 + t53_leaf2 + t53_leaf3 + t53_leaf4
   + t53_leaf5 + t53_leaf6 = 1
 T53_L_f30_k0: x_Fireplace_Qu + 1e+06 t53_leaf0 <= 1.000000999999e+06
 T53_L_f9_k0: x_Exter_Cond + 1e+06 t53_leaf0 <= 1.000001999999e+06
 T53_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t53_leaf0 <= 1.000003999999e+06
 T53_L_f30_k1: x_Fireplace_Qu + 1e+06 t53_leaf1 <= 1.000000999999e+06
 T53_L_f9_k1: x_Exter_Cond + 1e+06 t53_leaf1 <= 1.000001999999e+06
 T53_R_f25_k1: x_Bedroom_AbvGr - 1e+06 t53_leaf1 >= -999996
 T53_L_f30_k2: x_Fireplace_Qu + 1e+06 t53_leaf2 <= 1.000000999999e+06
 T53_R_f9_k2: x_Exter_Cond - 1e+06 t53_leaf2 >= -999998
 T53_L_f247_k2: const_Garage_Type_Attchd + 1e+06 t53_leaf2
   <= 1.000000999999e+06
 T53_L_f30_k3: x_Fireplace_Qu + 1e+06 t53_leaf3 <= 1.000000999999e+06
 T53_R_f9_k3: x_Exter_Cond - 1e+06 t53_leaf3 >= -999998
 T53_R_f247_k3: const_Garage_Type_Attchd - 1e+06 t53_leaf3 >= -999999
 T53_R_f30_k4: x_Fireplace_Qu - 1e+06 t53_leaf4 >= -999999
 T53_L_f4_k4: const_Overall_Cond + 1e+06 t53_leaf4 <= 1.000003999999e+06
 T53_R_f30_k5: x_Fireplace_Qu - 1e+06 t53_leaf5 >= -999999
 T53_R_f4_k5: const_Overall_Cond - 1e+06 t53_leaf5 >= -999996
 T53_L_f3_k5: Overall_Qual_calc + 1e+06 t53_leaf5 <= 1.000008999999e+06
 T53_R_f30_k6: x_Fireplace_Qu - 1e+06 t53_leaf6 >= -999999
 T53_R_f4_k6: const_Overall_Cond - 1e+06 t53_leaf6 >= -999996
 T53_R_f3_k6: Overall_Qual_calc - 1e+06 t53_leaf6 >= -999991
 T54_onehot: t54_leaf0 + t54_leaf1 + t54_leaf2 + t54_leaf3 + t54_leaf4
   + t54_leaf5 + t54_leaf6 = 1
 T54_L_f4_k0: const_Overall_Cond + 1e+06 t54_leaf0 <= 1.000004999999e+06
 T54_L_f4_k0: const_Overall_Cond + 1e+06 t54_leaf0 <= 1.000003999999e+06
 T54_L_f4_k1: const_Overall_Cond + 1e+06 t54_leaf1 <= 1.000004999999e+06
 T54_R_f4_k1: const_Overall_Cond - 1e+06 t54_leaf1 >= -999996
 T54_L_f44_k1: const_Yr_Sold + 1e+06 t54_leaf1 <= 1.002006999999e+06
 T54_L_f4_k2: const_Overall_Cond + 1e+06 t54_leaf2 <= 1.000004999999e+06
 T54_R_f4_k2: const_Overall_Cond - 1e+06 t54_leaf2 >= -999996
 T54_R_f44_k2: const_Yr_Sold - 1e+06 t54_leaf2 >= -997993
 T54_R_f4_k3: const_Overall_Cond - 1e+06 t54_leaf3 >= -999995
 T54_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t54_leaf3 <= 1.000647999999e+06
 T54_L_f9_k3: x_Exter_Cond + 1e+06 t54_leaf3 <= 1.000001999999e+06
 T54_R_f4_k4: const_Overall_Cond - 1e+06 t54_leaf4 >= -999995
 T54_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t54_leaf4 <= 1.000647999999e+06
 T54_R_f9_k4: x_Exter_Cond - 1e+06 t54_leaf4 >= -999998
 T54_R_f4_k5: const_Overall_Cond - 1e+06 t54_leaf5 >= -999995
 T54_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t54_leaf5 >= -999352
 T54_L_f27_k5: x_Kitchen_Qual + 1e+06 t54_leaf5 <= 1.000003999999e+06
 T54_R_f4_k6: const_Overall_Cond - 1e+06 t54_leaf6 >= -999995
 T54_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t54_leaf6 >= -999352
 T54_R_f27_k6: x_Kitchen_Qual - 1e+06 t54_leaf6 >= -999996
 T55_onehot: t55_leaf0 + t55_leaf1 + t55_leaf2 + t55_leaf3 + t55_leaf4
   + t55_leaf5 + t55_leaf6 + t55_leaf7 = 1
 T55_L_f29_k0: const_Fireplaces + 1e+06 t55_leaf0 <= 1.000000999999e+06
 T55_L_f9_k0: x_Exter_Cond + 1e+06 t55_leaf0 <= 1.000001999999e+06
 T55_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t55_leaf0
   <= 1.000000999999e+06
 T55_L_f29_k1: const_Fireplaces + 1e+06 t55_leaf1 <= 1.000000999999e+06
 T55_L_f9_k1: x_Exter_Cond + 1e+06 t55_leaf1 <= 1.000001999999e+06
 T55_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t55_leaf1 >= -999999
 T55_L_f29_k2: const_Fireplaces + 1e+06 t55_leaf2 <= 1.000000999999e+06
 T55_R_f9_k2: x_Exter_Cond - 1e+06 t55_leaf2 >= -999998
 T55_L_f33_k2: x_Garage_Area + 1e+06 t55_leaf2 <= 1.000269999999e+06
 T55_L_f29_k3: const_Fireplaces + 1e+06 t55_leaf3 <= 1.000000999999e+06
 T55_R_f9_k3: x_Exter_Cond - 1e+06 t55_leaf3 >= -999998
 T55_R_f33_k3: x_Garage_Area - 1e+06 t55_leaf3 >= -999730
 T55_R_f29_k4: const_Fireplaces - 1e+06 t55_leaf4 >= -999999
 T55_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t55_leaf4 <= 1.001096999999e+06
 T55_L_f18_k4: const_2nd_Flr_SF + 1e+06 t55_leaf4 <= 1.000932999999e+06
 T55_R_f29_k5: const_Fireplaces - 1e+06 t55_leaf5 >= -999999
 T55_L_f12_k5: x_BsmtFin_SF_1 + 1e+06 t55_leaf5 <= 1.001096999999e+06
 T55_R_f18_k5: const_2nd_Flr_SF - 1e+06 t55_leaf5 >= -999067
 T55_R_f29_k6: const_Fireplaces - 1e+06 t55_leaf6 >= -999999
 T55_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t55_leaf6 >= -998903
 T55_L_f36_k6: x_Wood_Deck_SF + 1e+06 t55_leaf6 <= 1.000370999999e+06
 T55_R_f29_k7: const_Fireplaces - 1e+06 t55_leaf7 >= -999999
 T55_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t55_leaf7 >= -998903
 T55_R_f36_k7: x_Wood_Deck_SF - 1e+06 t55_leaf7 >= -999629
 T56_onehot: t56_leaf0 + t56_leaf1 + t56_leaf2 + t56_leaf3 + t56_leaf4
   + t56_leaf5 + t56_leaf6 + t56_leaf7 = 1
 T56_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t56_leaf0 <= 1.001045999999e+06
 T56_L_f1_k0: const_Lot_Area + 1e+06 t56_leaf0 <= 1.006441999999e+06
 T56_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t56_leaf0 <= 1.000369999999e+06
 T56_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t56_leaf1 <= 1.001045999999e+06
 T56_L_f1_k1: const_Lot_Area + 1e+06 t56_leaf1 <= 1.006441999999e+06
 T56_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t56_leaf1 >= -999630
 T56_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t56_leaf2 <= 1.001045999999e+06
 T56_R_f1_k2: const_Lot_Area - 1e+06 t56_leaf2 >= -993558
 T56_L_f24_k2: x_Half_Bath + 1e+06 t56_leaf2 <= 1.000000999999e+06
 T56_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t56_leaf3 <= 1.001045999999e+06
 T56_R_f1_k3: const_Lot_Area - 1e+06 t56_leaf3 >= -993558
 T56_R_f24_k3: x_Half_Bath - 1e+06 t56_leaf3 >= -999999
 T56_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t56_leaf4 >= -998954
 T56_L_f20_k4: x_Gr_Liv_Area + 1e+06 t56_leaf4 <= 1.002236999999e+06
 T56_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t56_leaf4 <= 1.001096999999e+06
 T56_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t56_leaf5 >= -998954
 T56_L_f20_k5: x_Gr_Liv_Area + 1e+06 t56_leaf5 <= 1.002236999999e+06
 T56_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t56_leaf5 >= -998903
 T56_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t56_leaf6 >= -998954
 T56_R_f20_k6: x_Gr_Liv_Area - 1e+06 t56_leaf6 >= -997763
 T56_L_f43_k6: const_Misc_Val + 1e+06 t56_leaf6 <= 1.001199999999e+06
 T56_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t56_leaf7 >= -998954
 T56_R_f20_k7: x_Gr_Liv_Area - 1e+06 t56_leaf7 >= -997763
 T56_R_f43_k7: const_Misc_Val - 1e+06 t56_leaf7 >= -998800
 T57_onehot: t57_leaf0 + t57_leaf1 + t57_leaf2 + t57_leaf3 + t57_leaf4
   + t57_leaf5 + t57_leaf6 + t57_leaf7 = 1
 T57_L_f27_k0: x_Kitchen_Qual + 1e+06 t57_leaf0 <= 1.000002999999e+06
 T57_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t57_leaf0 <= 1.000055999999e+06
 T57_L_f20_k0: x_Gr_Liv_Area + 1e+06 t57_leaf0 <= 1.001145999999e+06
 T57_L_f27_k1: x_Kitchen_Qual + 1e+06 t57_leaf1 <= 1.000002999999e+06
 T57_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t57_leaf1 <= 1.000055999999e+06
 T57_R_f20_k1: x_Gr_Liv_Area - 1e+06 t57_leaf1 >= -998854
 T57_L_f27_k2: x_Kitchen_Qual + 1e+06 t57_leaf2 <= 1.000002999999e+06
 T57_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t57_leaf2 >= -999944
 T57_L_f1_k2: const_Lot_Area + 1e+06 t57_leaf2 <= 1.008625999999e+06
 T57_L_f27_k3: x_Kitchen_Qual + 1e+06 t57_leaf3 <= 1.000002999999e+06
 T57_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t57_leaf3 >= -999944
 T57_R_f1_k3: const_Lot_Area - 1e+06 t57_leaf3 >= -991374
 T57_R_f27_k4: x_Kitchen_Qual - 1e+06 t57_leaf4 >= -999997
 T57_L_f20_k4: x_Gr_Liv_Area + 1e+06 t57_leaf4 <= 1.002197999999e+06
 T57_L_f1_k4: const_Lot_Area + 1e+06 t57_leaf4 <= 1.006239999999e+06
 T57_R_f27_k5: x_Kitchen_Qual - 1e+06 t57_leaf5 >= -999997
 T57_L_f20_k5: x_Gr_Liv_Area + 1e+06 t57_leaf5 <= 1.002197999999e+06
 T57_R_f1_k5: const_Lot_Area - 1e+06 t57_leaf5 >= -993760
 T57_R_f27_k6: x_Kitchen_Qual - 1e+06 t57_leaf6 >= -999997
 T57_R_f20_k6: x_Gr_Liv_Area - 1e+06 t57_leaf6 >= -997802
 T57_L_f18_k6: const_2nd_Flr_SF + 1e+06 t57_leaf6 <= 1.001404999999e+06
 T57_R_f27_k7: x_Kitchen_Qual - 1e+06 t57_leaf7 >= -999997
 T57_R_f20_k7: x_Gr_Liv_Area - 1e+06 t57_leaf7 >= -997802
 T57_R_f18_k7: const_2nd_Flr_SF - 1e+06 t57_leaf7 >= -998595
 T58_onehot: t58_leaf0 + t58_leaf1 + t58_leaf2 + t58_leaf3 + t58_leaf4
   + t58_leaf5 + t58_leaf6 + t58_leaf7 = 1
 T58_L_f33_k0: x_Garage_Area + 1e+06 t58_leaf0 <= 1.000370999999e+06
 T58_L_f1_k0: const_Lot_Area + 1e+06 t58_leaf0 <= 1.005975999999e+06
 T58_L_f35_k0: x_Garage_Cond + 1e+06 t58_leaf0 <= 1.000001999999e+06
 T58_L_f33_k1: x_Garage_Area + 1e+06 t58_leaf1 <= 1.000370999999e+06
 T58_L_f1_k1: const_Lot_Area + 1e+06 t58_leaf1 <= 1.005975999999e+06
 T58_R_f35_k1: x_Garage_Cond - 1e+06 t58_leaf1 >= -999998
 T58_L_f33_k2: x_Garage_Area + 1e+06 t58_leaf2 <= 1.000370999999e+06
 T58_R_f1_k2: const_Lot_Area - 1e+06 t58_leaf2 >= -994024
 T58_L_f20_k2: x_Gr_Liv_Area + 1e+06 t58_leaf2 <= 1.000882999999e+06
 T58_L_f33_k3: x_Garage_Area + 1e+06 t58_leaf3 <= 1.000370999999e+06
 T58_R_f1_k3: const_Lot_Area - 1e+06 t58_leaf3 >= -994024
 T58_R_f20_k3: x_Gr_Liv_Area - 1e+06 t58_leaf3 >= -999117
 T58_R_f33_k4: x_Garage_Area - 1e+06 t58_leaf4 >= -999629
 T58_L_f17_k4: x_1st_Flr_SF + 1e+06 t58_leaf4 <= 1.001810999999e+06
 T58_L_f16_k4: x_Heating_QC + 1e+06 t58_leaf4 <= 1.000002999999e+06
 T58_R_f33_k5: x_Garage_Area - 1e+06 t58_leaf5 >= -999629
 T58_L_f17_k5: x_1st_Flr_SF + 1e+06 t58_leaf5 <= 1.001810999999e+06
 T58_R_f16_k5: x_Heating_QC - 1e+06 t58_leaf5 >= -999997
 T58_R_f33_k6: x_Garage_Area - 1e+06 t58_leaf6 >= -999629
 T58_R_f17_k6: x_1st_Flr_SF - 1e+06 t58_leaf6 >= -998189
 T58_L_f14_k6: x_Bsmt_Unf_SF + 1e+06 t58_leaf6 <= 1.000784999999e+06
 T58_R_f33_k7: x_Garage_Area - 1e+06 t58_leaf7 >= -999629
 T58_R_f17_k7: x_1st_Flr_SF - 1e+06 t58_leaf7 >= -998189
 T58_R_f14_k7: x_Bsmt_Unf_SF - 1e+06 t58_leaf7 >= -999215
 T59_onehot: t59_leaf0 + t59_leaf1 + t59_leaf2 + t59_leaf3 + t59_leaf4
   + t59_leaf5 + t59_leaf6 + t59_leaf7 = 1
 T59_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t59_leaf0 <= 1.001928999999e+06
 T59_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t59_leaf0 <= 1.000802999999e+06
 T59_L_f141_k0: const_House_Style_1Story + 1e+06 t59_leaf0
   <= 1.000000999999e+06
 T59_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t59_leaf1 <= 1.001928999999e+06
 T59_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t59_leaf1 <= 1.000802999999e+06
 T59_R_f141_k1: const_House_Style_1Story - 1e+06 t59_leaf1 >= -999999
 T59_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t59_leaf2 <= 1.001928999999e+06
 T59_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t59_leaf2 >= -999197
 T59_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t59_leaf2 <= 1.000080999999e+06
 T59_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t59_leaf3 <= 1.001928999999e+06
 T59_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t59_leaf3 >= -999197
 T59_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t59_leaf3 >= -999919
 T59_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t59_leaf4 >= -998071
 T59_L_f1_k4: const_Lot_Area + 1e+06 t59_leaf4 <= 1.011979999999e+06
 T59_L_f5_k4: const_Year_Built + 1e+06 t59_leaf4 <= 1.001979999999e+06
 T59_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t59_leaf5 >= -998071
 T59_L_f1_k5: const_Lot_Area + 1e+06 t59_leaf5 <= 1.011979999999e+06
 T59_R_f5_k5: const_Year_Built - 1e+06 t59_leaf5 >= -998020
 T59_R_f31_k6: const_Garage_Yr_Blt - 1e+06 t59_leaf6 >= -998071
 T59_R_f1_k6: const_Lot_Area - 1e+06 t59_leaf6 >= -988020
 T59_L_f207_k6: const_Bsmt_Exposure_Gd + 1e+06 t59_leaf6
   <= 1.000000999999e+06
 T59_R_f31_k7: const_Garage_Yr_Blt - 1e+06 t59_leaf7 >= -998071
 T59_R_f1_k7: const_Lot_Area - 1e+06 t59_leaf7 >= -988020
 T59_R_f207_k7: const_Bsmt_Exposure_Gd - 1e+06 t59_leaf7 >= -999999
 T60_onehot: t60_leaf0 + t60_leaf1 + t60_leaf2 + t60_leaf3 + t60_leaf4
   + t60_leaf5 + t60_leaf6 = 1
 T60_L_f30_k0: x_Fireplace_Qu + 1e+06 t60_leaf0 <= 1.000000999999e+06
 T60_L_f3_k0: Overall_Qual_calc + 1e+06 t60_leaf0 <= 1.000002999999e+06
 T60_L_f30_k1: x_Fireplace_Qu + 1e+06 t60_leaf1 <= 1.000000999999e+06
 T60_R_f3_k1: Overall_Qual_calc - 1e+06 t60_leaf1 >= -999997
 T60_L_f44_k1: const_Yr_Sold + 1e+06 t60_leaf1 <= 1.002007999999e+06
 T60_L_f30_k2: x_Fireplace_Qu + 1e+06 t60_leaf2 <= 1.000000999999e+06
 T60_R_f3_k2: Overall_Qual_calc - 1e+06 t60_leaf2 >= -999997
 T60_R_f44_k2: const_Yr_Sold - 1e+06 t60_leaf2 >= -997992
 T60_R_f30_k3: x_Fireplace_Qu - 1e+06 t60_leaf3 >= -999999
 T60_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t60_leaf3 <= 1.001096999999e+06
 T60_L_f0_k3: const_Lot_Frontage + 1e+06 t60_leaf3 <= 1.000064999999e+06
 T60_R_f30_k4: x_Fireplace_Qu - 1e+06 t60_leaf4 >= -999999
 T60_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t60_leaf4 <= 1.001096999999e+06
 T60_R_f0_k4: const_Lot_Frontage - 1e+06 t60_leaf4 >= -999935
 T60_R_f30_k5: x_Fireplace_Qu - 1e+06 t60_leaf5 >= -999999
 T60_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t60_leaf5 >= -998903
 T60_L_f28_k5: x_TotRms_AbvGrd + 1e+06 t60_leaf5 <= 1.000009999999e+06
 T60_R_f30_k6: x_Fireplace_Qu - 1e+06 t60_leaf6 >= -999999
 T60_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t60_leaf6 >= -998903
 T60_R_f28_k6: x_TotRms_AbvGrd - 1e+06 t60_leaf6 >= -999990
 T61_onehot: t61_leaf0 + t61_leaf1 + t61_leaf2 + t61_leaf3 + t61_leaf4
   + t61_leaf5 + t61_leaf6 = 1
 T61_L_f4_k0: const_Overall_Cond + 1e+06 t61_leaf0 <= 1.000003999999e+06
 T61_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t61_leaf0 <= 1.001399999999e+06
 T61_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t61_leaf0
   <= 1.000000999999e+06
 T61_L_f4_k1: const_Overall_Cond + 1e+06 t61_leaf1 <= 1.000003999999e+06
 T61_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t61_leaf1 <= 1.001399999999e+06
 T61_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t61_leaf1 >= -999999
 T61_L_f4_k2: const_Overall_Cond + 1e+06 t61_leaf2 <= 1.000003999999e+06
 T61_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t61_leaf2 >= -998600
 T61_R_f4_k3: const_Overall_Cond - 1e+06 t61_leaf3 >= -999996
 T61_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t61_leaf3 <= 1.001023999999e+06
 T61_L_f1_k3: const_Lot_Area + 1e+06 t61_leaf3 <= 1.006124999999e+06
 T61_R_f4_k4: const_Overall_Cond - 1e+06 t61_leaf4 >= -999996
 T61_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t61_leaf4 <= 1.001023999999e+06
 T61_R_f1_k4: const_Lot_Area - 1e+06 t61_leaf4 >= -993875
 T61_R_f4_k5: const_Overall_Cond - 1e+06 t61_leaf5 >= -999996
 T61_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t61_leaf5 >= -998976
 T61_L_f18_k5: const_2nd_Flr_SF + 1e+06 t61_leaf5 <= 1.000879999999e+06
 T61_R_f4_k6: const_Overall_Cond - 1e+06 t61_leaf6 >= -999996
 T61_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t61_leaf6 >= -998976
 T61_R_f18_k6: const_2nd_Flr_SF - 1e+06 t61_leaf6 >= -999120
 T62_onehot: t62_leaf0 + t62_leaf1 + t62_leaf2 + t62_leaf3 + t62_leaf4
   + t62_leaf5 + t62_leaf6 = 1
 T62_L_f20_k0: x_Gr_Liv_Area + 1e+06 t62_leaf0 <= 1.002197999999e+06
 T62_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t62_leaf0 <= 1.000328999999e+06
 T62_L_f5_k0: const_Year_Built + 1e+06 t62_leaf0 <= 1.001983999999e+06
 T62_L_f20_k1: x_Gr_Liv_Area + 1e+06 t62_leaf1 <= 1.002197999999e+06
 T62_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t62_leaf1 <= 1.000328999999e+06
 T62_R_f5_k1: const_Year_Built - 1e+06 t62_leaf1 >= -998016
 T62_L_f20_k2: x_Gr_Liv_Area + 1e+06 t62_leaf2 <= 1.002197999999e+06
 T62_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t62_leaf2 >= -999671
 T62_L_f20_k2: x_Gr_Liv_Area + 1e+06 t62_leaf2 <= 1.001415999999e+06
 T62_L_f20_k3: x_Gr_Liv_Area + 1e+06 t62_leaf3 <= 1.002197999999e+06
 T62_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t62_leaf3 >= -999671
 T62_R_f20_k3: x_Gr_Liv_Area - 1e+06 t62_leaf3 >= -998584
 T62_R_f20_k4: x_Gr_Liv_Area - 1e+06 t62_leaf4 >= -997802
 T62_L_f118_k4: const_Condition_1_Feedr + 1e+06 t62_leaf4
   <= 1.000000999999e+06
 T62_L_f1_k4: const_Lot_Area + 1e+06 t62_leaf4 <= 1.013517999999e+06
 T62_R_f20_k5: x_Gr_Liv_Area - 1e+06 t62_leaf5 >= -997802
 T62_L_f118_k5: const_Condition_1_Feedr + 1e+06 t62_leaf5
   <= 1.000000999999e+06
 T62_R_f1_k5: const_Lot_Area - 1e+06 t62_leaf5 >= -986482
 T62_R_f20_k6: x_Gr_Liv_Area - 1e+06 t62_leaf6 >= -997802
 T62_R_f118_k6: const_Condition_1_Feedr - 1e+06 t62_leaf6 >= -999999
 T63_onehot: t63_leaf0 + t63_leaf1 + t63_leaf2 + t63_leaf3 + t63_leaf4
   + t63_leaf5 + t63_leaf6 + t63_leaf7 = 1
 T63_L_f6_k0: const_Year_Remod_Add + 1e+06 t63_leaf0 <= 1.001986999999e+06
 T63_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t63_leaf0 <= 1.000449999999e+06
 T63_L_f3_k0: Overall_Qual_calc + 1e+06 t63_leaf0 <= 1.000002999999e+06
 T63_L_f6_k1: const_Year_Remod_Add + 1e+06 t63_leaf1 <= 1.001986999999e+06
 T63_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t63_leaf1 <= 1.000449999999e+06
 T63_R_f3_k1: Overall_Qual_calc - 1e+06 t63_leaf1 >= -999997
 T63_L_f6_k2: const_Year_Remod_Add + 1e+06 t63_leaf2 <= 1.001986999999e+06
 T63_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t63_leaf2 >= -999550
 T63_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t63_leaf2
   <= 1.000000999999e+06
 T63_L_f6_k3: const_Year_Remod_Add + 1e+06 t63_leaf3 <= 1.001986999999e+06
 T63_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t63_leaf3 >= -999550
 T63_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t63_leaf3 >= -999999
 T63_R_f6_k4: const_Year_Remod_Add - 1e+06 t63_leaf4 >= -998013
 T63_L_f18_k4: const_2nd_Flr_SF + 1e+06 t63_leaf4 <= 1.001367999999e+06
 T63_L_f96_k4: const_Neighborhood_Edwards + 1e+06 t63_leaf4
   <= 1.000000999999e+06
 T63_R_f6_k5: const_Year_Remod_Add - 1e+06 t63_leaf5 >= -998013
 T63_L_f18_k5: const_2nd_Flr_SF + 1e+06 t63_leaf5 <= 1.001367999999e+06
 T63_R_f96_k5: const_Neighborhood_Edwards - 1e+06 t63_leaf5 >= -999999
 T63_R_f6_k6: const_Year_Remod_Add - 1e+06 t63_leaf6 >= -998013
 T63_R_f18_k6: const_2nd_Flr_SF - 1e+06 t63_leaf6 >= -998632
 T63_L_f6_k6: const_Year_Remod_Add + 1e+06 t63_leaf6 <= 1.001998999999e+06
 T63_R_f6_k7: const_Year_Remod_Add - 1e+06 t63_leaf7 >= -998013
 T63_R_f18_k7: const_2nd_Flr_SF - 1e+06 t63_leaf7 >= -998632
 T63_R_f6_k7: const_Year_Remod_Add - 1e+06 t63_leaf7 >= -998001
 T64_onehot: t64_leaf0 + t64_leaf1 + t64_leaf2 + t64_leaf3 + t64_leaf4
   + t64_leaf5 + t64_leaf6 + t64_leaf7 = 1
 T64_L_f20_k0: x_Gr_Liv_Area + 1e+06 t64_leaf0 <= 1.001163999999e+06
 T64_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t64_leaf0 <= 1.000055999999e+06
 T64_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t64_leaf0 <= 1.000859999999e+06
 T64_L_f20_k1: x_Gr_Liv_Area + 1e+06 t64_leaf1 <= 1.001163999999e+06
 T64_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t64_leaf1 <= 1.000055999999e+06
 T64_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t64_leaf1 >= -999140
 T64_L_f20_k2: x_Gr_Liv_Area + 1e+06 t64_leaf2 <= 1.001163999999e+06
 T64_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t64_leaf2 >= -999944
 T64_L_f6_k2: const_Year_Remod_Add + 1e+06 t64_leaf2 <= 1.001976999999e+06
 T64_L_f20_k3: x_Gr_Liv_Area + 1e+06 t64_leaf3 <= 1.001163999999e+06
 T64_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t64_leaf3 >= -999944
 T64_R_f6_k3: const_Year_Remod_Add - 1e+06 t64_leaf3 >= -998023
 T64_R_f20_k4: x_Gr_Liv_Area - 1e+06 t64_leaf4 >= -998836
 T64_L_f20_k4: x_Gr_Liv_Area + 1e+06 t64_leaf4 <= 1.002543999999e+06
 T64_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t64_leaf4 <= 1.000291999999e+06
 T64_R_f20_k5: x_Gr_Liv_Area - 1e+06 t64_leaf5 >= -998836
 T64_L_f20_k5: x_Gr_Liv_Area + 1e+06 t64_leaf5 <= 1.002543999999e+06
 T64_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t64_leaf5 >= -999708
 T64_R_f20_k6: x_Gr_Liv_Area - 1e+06 t64_leaf6 >= -998836
 T64_R_f20_k6: x_Gr_Liv_Area - 1e+06 t64_leaf6 >= -997456
 T64_L_f20_k6: x_Gr_Liv_Area + 1e+06 t64_leaf6 <= 1.003085999999e+06
 T64_R_f20_k7: x_Gr_Liv_Area - 1e+06 t64_leaf7 >= -998836
 T64_R_f20_k7: x_Gr_Liv_Area - 1e+06 t64_leaf7 >= -997456
 T64_R_f20_k7: x_Gr_Liv_Area - 1e+06 t64_leaf7 >= -996914
 T65_onehot: t65_leaf0 + t65_leaf1 + t65_leaf2 + t65_leaf3 + t65_leaf4
   + t65_leaf5 + t65_leaf6 + t65_leaf7 = 1
 T65_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t65_leaf0 <= 1.000647999999e+06
 T65_L_f3_k0: Overall_Qual_calc + 1e+06 t65_leaf0 <= 1.000003999999e+06
 T65_L_f191_k0: x_ex2_is_VinylSd + 1e+06 t65_leaf0 <= 1.000000999999e+06
 T65_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t65_leaf1 <= 1.000647999999e+06
 T65_L_f3_k1: Overall_Qual_calc + 1e+06 t65_leaf1 <= 1.000003999999e+06
 T65_R_f191_k1: x_ex2_is_VinylSd - 1e+06 t65_leaf1 >= -999999
 T65_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t65_leaf2 <= 1.000647999999e+06
 T65_R_f3_k2: Overall_Qual_calc - 1e+06 t65_leaf2 >= -999996
 T65_L_f1_k2: const_Lot_Area + 1e+06 t65_leaf2 <= 1.006761999999e+06
 T65_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t65_leaf3 <= 1.000647999999e+06
 T65_R_f3_k3: Overall_Qual_calc - 1e+06 t65_leaf3 >= -999996
 T65_R_f1_k3: const_Lot_Area - 1e+06 t65_leaf3 >= -993238
 T65_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t65_leaf4 >= -999352
 T65_L_f17_k4: x_1st_Flr_SF + 1e+06 t65_leaf4 <= 1.001829999999e+06
 T65_L_f6_k4: const_Year_Remod_Add + 1e+06 t65_leaf4 <= 1.001962999999e+06
 T65_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t65_leaf5 >= -999352
 T65_L_f17_k5: x_1st_Flr_SF + 1e+06 t65_leaf5 <= 1.001829999999e+06
 T65_R_f6_k5: const_Year_Remod_Add - 1e+06 t65_leaf5 >= -998037
 T65_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t65_leaf6 >= -999352
 T65_R_f17_k6: x_1st_Flr_SF - 1e+06 t65_leaf6 >= -998170
 T65_L_f1_k6: const_Lot_Area + 1e+06 t65_leaf6 <= 1.011387999999e+06
 T65_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t65_leaf7 >= -999352
 T65_R_f17_k7: x_1st_Flr_SF - 1e+06 t65_leaf7 >= -998170
 T65_R_f1_k7: const_Lot_Area - 1e+06 t65_leaf7 >= -988612
 T66_onehot: t66_leaf0 + t66_leaf1 + t66_leaf2 + t66_leaf3 + t66_leaf4
   + t66_leaf5 + t66_leaf6 + t66_leaf7 = 1
 T66_L_f257_k0: x_paved_drive_is_N + 1e+06 t66_leaf0 <= 1.000000999999e+06
 T66_L_f44_k0: const_Yr_Sold + 1e+06 t66_leaf0 <= 1.002007999999e+06
 T66_L_f20_k0: x_Gr_Liv_Area + 1e+06 t66_leaf0 <= 1.001967999999e+06
 T66_L_f257_k1: x_paved_drive_is_N + 1e+06 t66_leaf1 <= 1.000000999999e+06
 T66_L_f44_k1: const_Yr_Sold + 1e+06 t66_leaf1 <= 1.002007999999e+06
 T66_R_f20_k1: x_Gr_Liv_Area - 1e+06 t66_leaf1 >= -998032
 T66_L_f257_k2: x_paved_drive_is_N + 1e+06 t66_leaf2 <= 1.000000999999e+06
 T66_R_f44_k2: const_Yr_Sold - 1e+06 t66_leaf2 >= -997992
 T66_L_f16_k2: x_Heating_QC + 1e+06 t66_leaf2 <= 1.000002999999e+06
 T66_L_f257_k3: x_paved_drive_is_N + 1e+06 t66_leaf3 <= 1.000000999999e+06
 T66_R_f44_k3: const_Yr_Sold - 1e+06 t66_leaf3 >= -997992
 T66_R_f16_k3: x_Heating_QC - 1e+06 t66_leaf3 >= -999997
 T66_R_f257_k4: x_paved_drive_is_N - 1e+06 t66_leaf4 >= -999999
 T66_L_f44_k4: const_Yr_Sold + 1e+06 t66_leaf4 <= 1.002007999999e+06
 T66_L_f20_k4: x_Gr_Liv_Area + 1e+06 t66_leaf4 <= 1.001487999999e+06
 T66_R_f257_k5: x_paved_drive_is_N - 1e+06 t66_leaf5 >= -999999
 T66_L_f44_k5: const_Yr_Sold + 1e+06 t66_leaf5 <= 1.002007999999e+06
 T66_R_f20_k5: x_Gr_Liv_Area - 1e+06 t66_leaf5 >= -998512
 T66_R_f257_k6: x_paved_drive_is_N - 1e+06 t66_leaf6 >= -999999
 T66_R_f44_k6: const_Yr_Sold - 1e+06 t66_leaf6 >= -997992
 T66_L_f20_k6: x_Gr_Liv_Area + 1e+06 t66_leaf6 <= 1.001104999999e+06
 T66_R_f257_k7: x_paved_drive_is_N - 1e+06 t66_leaf7 >= -999999
 T66_R_f44_k7: const_Yr_Sold - 1e+06 t66_leaf7 >= -997992
 T66_R_f20_k7: x_Gr_Liv_Area - 1e+06 t66_leaf7 >= -998895
 T67_onehot: t67_leaf0 + t67_leaf1 + t67_leaf2 + t67_leaf3 + t67_leaf4
   + t67_leaf5 = 1
 T67_L_f3_k0: Overall_Qual_calc + 1e+06 t67_leaf0 <= 1.000003999999e+06
 T67_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t67_leaf0
   <= 1.000000999999e+06
 T67_L_f3_k1: Overall_Qual_calc + 1e+06 t67_leaf1 <= 1.000003999999e+06
 T67_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t67_leaf1 >= -999999
 T67_R_f3_k2: Overall_Qual_calc - 1e+06 t67_leaf2 >= -999996
 T67_L_f6_k2: const_Year_Remod_Add + 1e+06 t67_leaf2 <= 1.001992999999e+06
 T67_L_f44_k2: const_Yr_Sold + 1e+06 t67_leaf2 <= 1.002007999999e+06
 T67_R_f3_k3: Overall_Qual_calc - 1e+06 t67_leaf3 >= -999996
 T67_L_f6_k3: const_Year_Remod_Add + 1e+06 t67_leaf3 <= 1.001992999999e+06
 T67_R_f44_k3: const_Yr_Sold - 1e+06 t67_leaf3 >= -997992
 T67_R_f3_k4: Overall_Qual_calc - 1e+06 t67_leaf4 >= -999996
 T67_R_f6_k4: const_Year_Remod_Add - 1e+06 t67_leaf4 >= -998007
 T67_L_f27_k4: x_Kitchen_Qual + 1e+06 t67_leaf4 <= 1.000003999999e+06
 T67_R_f3_k5: Overall_Qual_calc - 1e+06 t67_leaf5 >= -999996
 T67_R_f6_k5: const_Year_Remod_Add - 1e+06 t67_leaf5 >= -998007
 T67_R_f27_k5: x_Kitchen_Qual - 1e+06 t67_leaf5 >= -999996
 T68_onehot: t68_leaf0 + t68_leaf1 + t68_leaf2 + t68_leaf3 + t68_leaf4
   + t68_leaf5 + t68_leaf6 + t68_leaf7 = 1
 T68_L_f30_k0: x_Fireplace_Qu + 1e+06 t68_leaf0 <= 1.000000999999e+06
 T68_L_f46_k0: const_MS_SubClass_30 + 1e+06 t68_leaf0
   <= 1.000000999999e+06
 T68_L_f1_k0: const_Lot_Area + 1e+06 t68_leaf0 <= 1.006119999999e+06
 T68_L_f30_k1: x_Fireplace_Qu + 1e+06 t68_leaf1 <= 1.000000999999e+06
 T68_L_f46_k1: const_MS_SubClass_30 + 1e+06 t68_leaf1
   <= 1.000000999999e+06
 T68_R_f1_k1: const_Lot_Area - 1e+06 t68_leaf1 >= -993880
 T68_L_f30_k2: x_Fireplace_Qu + 1e+06 t68_leaf2 <= 1.000000999999e+06
 T68_R_f46_k2: const_MS_SubClass_30 - 1e+06 t68_leaf2 >= -999999
 T68_L_f264_k2: x_fence_is_No_aplica + 1e+06 t68_leaf2
   <= 1.000000999999e+06
 T68_L_f30_k3: x_Fireplace_Qu + 1e+06 t68_leaf3 <= 1.000000999999e+06
 T68_R_f46_k3: const_MS_SubClass_30 - 1e+06 t68_leaf3 >= -999999
 T68_R_f264_k3: x_fence_is_No_aplica - 1e+06 t68_leaf3 >= -999999
 T68_R_f30_k4: x_Fireplace_Qu - 1e+06 t68_leaf4 >= -999999
 T68_L_f44_k4: const_Yr_Sold + 1e+06 t68_leaf4 <= 1.002007999999e+06
 T68_L_f17_k4: x_1st_Flr_SF + 1e+06 t68_leaf4 <= 1.001488999999e+06
 T68_R_f30_k5: x_Fireplace_Qu - 1e+06 t68_leaf5 >= -999999
 T68_L_f44_k5: const_Yr_Sold + 1e+06 t68_leaf5 <= 1.002007999999e+06
 T68_R_f17_k5: x_1st_Flr_SF - 1e+06 t68_leaf5 >= -998511
 T68_R_f30_k6: x_Fireplace_Qu - 1e+06 t68_leaf6 >= -999999
 T68_R_f44_k6: const_Yr_Sold - 1e+06 t68_leaf6 >= -997992
 T68_L_f21_k6: const_Bsmt_Full_Bath + 1e+06 t68_leaf6
   <= 1.000000999999e+06
 T68_R_f30_k7: x_Fireplace_Qu - 1e+06 t68_leaf7 >= -999999
 T68_R_f44_k7: const_Yr_Sold - 1e+06 t68_leaf7 >= -997992
 T68_R_f21_k7: const_Bsmt_Full_Bath - 1e+06 t68_leaf7 >= -999999
 T69_onehot: t69_leaf0 + t69_leaf1 + t69_leaf2 + t69_leaf3 + t69_leaf4
   + t69_leaf5 + t69_leaf6 = 1
 T69_L_f4_k0: const_Overall_Cond + 1e+06 t69_leaf0 <= 1.000004999999e+06
 T69_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t69_leaf0 <= 1.001994999999e+06
 T69_L_f37_k0: x_Open_Porch_SF + 1e+06 t69_leaf0 <= 1.000071999999e+06
 T69_L_f4_k1: const_Overall_Cond + 1e+06 t69_leaf1 <= 1.000004999999e+06
 T69_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t69_leaf1 <= 1.001994999999e+06
 T69_R_f37_k1: x_Open_Porch_SF - 1e+06 t69_leaf1 >= -999928
 T69_L_f4_k2: const_Overall_Cond + 1e+06 t69_leaf2 <= 1.000004999999e+06
 T69_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t69_leaf2 >= -998005
 T69_R_f4_k3: const_Overall_Cond - 1e+06 t69_leaf3 >= -999995
 T69_L_f207_k3: const_Bsmt_Exposure_Gd + 1e+06 t69_leaf3
   <= 1.000000999999e+06
 T69_L_f1_k3: const_Lot_Area + 1e+06 t69_leaf3 <= 1.007199999999e+06
 T69_R_f4_k4: const_Overall_Cond - 1e+06 t69_leaf4 >= -999995
 T69_L_f207_k4: const_Bsmt_Exposure_Gd + 1e+06 t69_leaf4
   <= 1.000000999999e+06
 T69_R_f1_k4: const_Lot_Area - 1e+06 t69_leaf4 >= -992800
 T69_R_f4_k5: const_Overall_Cond - 1e+06 t69_leaf5 >= -999995
 T69_R_f207_k5: const_Bsmt_Exposure_Gd - 1e+06 t69_leaf5 >= -999999
 T69_L_f17_k5: x_1st_Flr_SF + 1e+06 t69_leaf5 <= 1.000947999999e+06
 T69_R_f4_k6: const_Overall_Cond - 1e+06 t69_leaf6 >= -999995
 T69_R_f207_k6: const_Bsmt_Exposure_Gd - 1e+06 t69_leaf6 >= -999999
 T69_R_f17_k6: x_1st_Flr_SF - 1e+06 t69_leaf6 >= -999052
 T70_onehot: t70_leaf0 + t70_leaf1 + t70_leaf2 + t70_leaf3 + t70_leaf4
   + t70_leaf5 + t70_leaf6 + t70_leaf7 = 1
 T70_L_f6_k0: const_Year_Remod_Add + 1e+06 t70_leaf0 <= 1.001983999999e+06
 T70_L_f66_k0: const_MS_Zoning_RL + 1e+06 t70_leaf0 <= 1.000000999999e+06
 T70_L_f44_k0: const_Yr_Sold + 1e+06 t70_leaf0 <= 1.002007999999e+06
 T70_L_f6_k1: const_Year_Remod_Add + 1e+06 t70_leaf1 <= 1.001983999999e+06
 T70_L_f66_k1: const_MS_Zoning_RL + 1e+06 t70_leaf1 <= 1.000000999999e+06
 T70_R_f44_k1: const_Yr_Sold - 1e+06 t70_leaf1 >= -997992
 T70_L_f6_k2: const_Year_Remod_Add + 1e+06 t70_leaf2 <= 1.001983999999e+06
 T70_R_f66_k2: const_MS_Zoning_RL - 1e+06 t70_leaf2 >= -999999
 T70_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t70_leaf2 <= 1.000519999999e+06
 T70_L_f6_k3: const_Year_Remod_Add + 1e+06 t70_leaf3 <= 1.001983999999e+06
 T70_R_f66_k3: const_MS_Zoning_RL - 1e+06 t70_leaf3 >= -999999
 T70_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t70_leaf3 >= -999480
 T70_R_f6_k4: const_Year_Remod_Add - 1e+06 t70_leaf4 >= -998016
 T70_L_f109_k4: const_Neighborhood_OldTown + 1e+06 t70_leaf4
   <= 1.000000999999e+06
 T70_L_f18_k4: const_2nd_Flr_SF + 1e+06 t70_leaf4 <= 1.001295999999e+06
 T70_R_f6_k5: const_Year_Remod_Add - 1e+06 t70_leaf5 >= -998016
 T70_L_f109_k5: const_Neighborhood_OldTown + 1e+06 t70_leaf5
   <= 1.000000999999e+06
 T70_R_f18_k5: const_2nd_Flr_SF - 1e+06 t70_leaf5 >= -998704
 T70_R_f6_k6: const_Year_Remod_Add - 1e+06 t70_leaf6 >= -998016
 T70_R_f109_k6: const_Neighborhood_OldTown - 1e+06 t70_leaf6 >= -999999
 T70_L_f200_k6: const_Foundation_BrkTil + 1e+06 t70_leaf6
   <= 1.000000999999e+06
 T70_R_f6_k7: const_Year_Remod_Add - 1e+06 t70_leaf7 >= -998016
 T70_R_f109_k7: const_Neighborhood_OldTown - 1e+06 t70_leaf7 >= -999999
 T70_R_f200_k7: const_Foundation_BrkTil - 1e+06 t70_leaf7 >= -999999
 T71_onehot: t71_leaf0 + t71_leaf1 + t71_leaf2 + t71_leaf3 + t71_leaf4
   + t71_leaf5 + t71_leaf6 + t71_leaf7 = 1
 T71_L_f20_k0: x_Gr_Liv_Area + 1e+06 t71_leaf0 <= 1.001921999999e+06
 T71_L_f10_k0: const_Bsmt_Qual + 1e+06 t71_leaf0 <= 1.000001999999e+06
 T71_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t71_leaf0
   <= 1.000000999999e+06
 T71_L_f20_k1: x_Gr_Liv_Area + 1e+06 t71_leaf1 <= 1.001921999999e+06
 T71_L_f10_k1: const_Bsmt_Qual + 1e+06 t71_leaf1 <= 1.000001999999e+06
 T71_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t71_leaf1 >= -999999
 T71_L_f20_k2: x_Gr_Liv_Area + 1e+06 t71_leaf2 <= 1.001921999999e+06
 T71_R_f10_k2: const_Bsmt_Qual - 1e+06 t71_leaf2 >= -999998
 T71_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t71_leaf2 <= 1.000291999999e+06
 T71_L_f20_k3: x_Gr_Liv_Area + 1e+06 t71_leaf3 <= 1.001921999999e+06
 T71_R_f10_k3: const_Bsmt_Qual - 1e+06 t71_leaf3 >= -999998
 T71_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t71_leaf3 >= -999708
 T71_R_f20_k4: x_Gr_Liv_Area - 1e+06 t71_leaf4 >= -998078
 T71_L_f18_k4: const_2nd_Flr_SF + 1e+06 t71_leaf4 <= 1.001367999999e+06
 T71_L_f245_k4: const_FunctioNo_aplical_Typ + 1e+06 t71_leaf4
   <= 1.000000999999e+06
 T71_R_f20_k5: x_Gr_Liv_Area - 1e+06 t71_leaf5 >= -998078
 T71_L_f18_k5: const_2nd_Flr_SF + 1e+06 t71_leaf5 <= 1.001367999999e+06
 T71_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t71_leaf5 >= -999999
 T71_R_f20_k6: x_Gr_Liv_Area - 1e+06 t71_leaf6 >= -998078
 T71_R_f18_k6: const_2nd_Flr_SF - 1e+06 t71_leaf6 >= -998632
 T71_L_f1_k6: const_Lot_Area + 1e+06 t71_leaf6 <= 1.012383999999e+06
 T71_R_f20_k7: x_Gr_Liv_Area - 1e+06 t71_leaf7 >= -998078
 T71_R_f18_k7: const_2nd_Flr_SF - 1e+06 t71_leaf7 >= -998632
 T71_R_f1_k7: const_Lot_Area - 1e+06 t71_leaf7 >= -987616
 T72_onehot: t72_leaf0 + t72_leaf1 + t72_leaf2 + t72_leaf3 + t72_leaf4
   + t72_leaf5 + t72_leaf6 + t72_leaf7 = 1
 T72_L_f44_k0: const_Yr_Sold + 1e+06 t72_leaf0 <= 1.002007999999e+06
 T72_L_f17_k0: x_1st_Flr_SF + 1e+06 t72_leaf0 <= 1.001319999999e+06
 T72_L_f58_k0: const_MS_SubClass_160 + 1e+06 t72_leaf0
   <= 1.000000999999e+06
 T72_L_f44_k1: const_Yr_Sold + 1e+06 t72_leaf1 <= 1.002007999999e+06
 T72_L_f17_k1: x_1st_Flr_SF + 1e+06 t72_leaf1 <= 1.001319999999e+06
 T72_R_f58_k1: const_MS_SubClass_160 - 1e+06 t72_leaf1 >= -999999
 T72_L_f44_k2: const_Yr_Sold + 1e+06 t72_leaf2 <= 1.002007999999e+06
 T72_R_f17_k2: x_1st_Flr_SF - 1e+06 t72_leaf2 >= -998680
 T72_L_f6_k2: const_Year_Remod_Add + 1e+06 t72_leaf2 <= 1.001990999999e+06
 T72_L_f44_k3: const_Yr_Sold + 1e+06 t72_leaf3 <= 1.002007999999e+06
 T72_R_f17_k3: x_1st_Flr_SF - 1e+06 t72_leaf3 >= -998680
 T72_R_f6_k3: const_Year_Remod_Add - 1e+06 t72_leaf3 >= -998009
 T72_R_f44_k4: const_Yr_Sold - 1e+06 t72_leaf4 >= -997992
 T72_L_f6_k4: const_Year_Remod_Add + 1e+06 t72_leaf4 <= 1.001983999999e+06
 T72_L_f297_k4: const_Sale_Condition_Normal + 1e+06 t72_leaf4
   <= 1.000000999999e+06
 T72_R_f44_k5: const_Yr_Sold - 1e+06 t72_leaf5 >= -997992
 T72_L_f6_k5: const_Year_Remod_Add + 1e+06 t72_leaf5 <= 1.001983999999e+06
 T72_R_f297_k5: const_Sale_Condition_Normal - 1e+06 t72_leaf5 >= -999999
 T72_R_f44_k6: const_Yr_Sold - 1e+06 t72_leaf6 >= -997992
 T72_R_f6_k6: const_Year_Remod_Add - 1e+06 t72_leaf6 >= -998016
 T72_L_f20_k6: x_Gr_Liv_Area + 1e+06 t72_leaf6 <= 1.002344999999e+06
 T72_R_f44_k7: const_Yr_Sold - 1e+06 t72_leaf7 >= -997992
 T72_R_f6_k7: const_Year_Remod_Add - 1e+06 t72_leaf7 >= -998016
 T72_R_f20_k7: x_Gr_Liv_Area - 1e+06 t72_leaf7 >= -997655
 T73_onehot: t73_leaf0 + t73_leaf1 + t73_leaf2 + t73_leaf3 + t73_leaf4
   + t73_leaf5 + t73_leaf6 = 1
 T73_L_f4_k0: const_Overall_Cond + 1e+06 t73_leaf0 <= 1.000003999999e+06
 T73_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t73_leaf0 <= 1.000413999999e+06
 T73_L_f4_k1: const_Overall_Cond + 1e+06 t73_leaf1 <= 1.000003999999e+06
 T73_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t73_leaf1 >= -999586
 T73_L_f8_k1: x_Exter_Qual + 1e+06 t73_leaf1 <= 1.000001999999e+06
 T73_L_f4_k2: const_Overall_Cond + 1e+06 t73_leaf2 <= 1.000003999999e+06
 T73_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t73_leaf2 >= -999586
 T73_R_f8_k2: x_Exter_Qual - 1e+06 t73_leaf2 >= -999998
 T73_R_f4_k3: const_Overall_Cond - 1e+06 t73_leaf3 >= -999996
 T73_L_f18_k3: const_2nd_Flr_SF + 1e+06 t73_leaf3 <= 1.001295999999e+06
 T73_L_f95_k3: const_Neighborhood_Crawfor + 1e+06 t73_leaf3
   <= 1.000000999999e+06
 T73_R_f4_k4: const_Overall_Cond - 1e+06 t73_leaf4 >= -999996
 T73_L_f18_k4: const_2nd_Flr_SF + 1e+06 t73_leaf4 <= 1.001295999999e+06
 T73_R_f95_k4: const_Neighborhood_Crawfor - 1e+06 t73_leaf4 >= -999999
 T73_R_f4_k5: const_Overall_Cond - 1e+06 t73_leaf5 >= -999996
 T73_R_f18_k5: const_2nd_Flr_SF - 1e+06 t73_leaf5 >= -998704
 T73_L_f14_k5: x_Bsmt_Unf_SF + 1e+06 t73_leaf5 <= 1.000425999999e+06
 T73_R_f4_k6: const_Overall_Cond - 1e+06 t73_leaf6 >= -999996
 T73_R_f18_k6: const_2nd_Flr_SF - 1e+06 t73_leaf6 >= -998704
 T73_R_f14_k6: x_Bsmt_Unf_SF - 1e+06 t73_leaf6 >= -999574
 T74_onehot: t74_leaf0 + t74_leaf1 + t74_leaf2 + t74_leaf3 + t74_leaf4
   + t74_leaf5 + t74_leaf6 = 1
 T74_L_f3_k0: Overall_Qual_calc + 1e+06 t74_leaf0 <= 1.000007999999e+06
 T74_L_f46_k0: const_MS_SubClass_30 + 1e+06 t74_leaf0
   <= 1.000000999999e+06
 T74_L_f44_k0: const_Yr_Sold + 1e+06 t74_leaf0 <= 1.002007999999e+06
 T74_L_f3_k1: Overall_Qual_calc + 1e+06 t74_leaf1 <= 1.000007999999e+06
 T74_L_f46_k1: const_MS_SubClass_30 + 1e+06 t74_leaf1
   <= 1.000000999999e+06
 T74_R_f44_k1: const_Yr_Sold - 1e+06 t74_leaf1 >= -997992
 T74_L_f3_k2: Overall_Qual_calc + 1e+06 t74_leaf2 <= 1.000007999999e+06
 T74_R_f46_k2: const_MS_SubClass_30 - 1e+06 t74_leaf2 >= -999999
 T74_L_f17_k2: x_1st_Flr_SF + 1e+06 t74_leaf2 <= 1.000969999999e+06
 T74_L_f3_k3: Overall_Qual_calc + 1e+06 t74_leaf3 <= 1.000007999999e+06
 T74_R_f46_k3: const_MS_SubClass_30 - 1e+06 t74_leaf3 >= -999999
 T74_R_f17_k3: x_1st_Flr_SF - 1e+06 t74_leaf3 >= -999030
 T74_R_f3_k4: Overall_Qual_calc - 1e+06 t74_leaf4 >= -999992
 T74_L_f18_k4: const_2nd_Flr_SF + 1e+06 t74_leaf4 <= 1.001367999999e+06
 T74_L_f17_k4: x_1st_Flr_SF + 1e+06 t74_leaf4 <= 1.001463999999e+06
 T74_R_f3_k5: Overall_Qual_calc - 1e+06 t74_leaf5 >= -999992
 T74_L_f18_k5: const_2nd_Flr_SF + 1e+06 t74_leaf5 <= 1.001367999999e+06
 T74_R_f17_k5: x_1st_Flr_SF - 1e+06 t74_leaf5 >= -998536
 T74_R_f3_k6: Overall_Qual_calc - 1e+06 t74_leaf6 >= -999992
 T74_R_f18_k6: const_2nd_Flr_SF - 1e+06 t74_leaf6 >= -998632
 T75_onehot: t75_leaf0 + t75_leaf1 + t75_leaf2 + t75_leaf3 + t75_leaf4
   + t75_leaf5 + t75_leaf6 = 1
 T75_L_f4_k0: const_Overall_Cond + 1e+06 t75_leaf0 <= 1.000003999999e+06
 T75_L_f36_k0: x_Wood_Deck_SF + 1e+06 t75_leaf0 <= 1.000021999999e+06
 T75_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t75_leaf0
   <= 1.000000999999e+06
 T75_L_f4_k1: const_Overall_Cond + 1e+06 t75_leaf1 <= 1.000003999999e+06
 T75_L_f36_k1: x_Wood_Deck_SF + 1e+06 t75_leaf1 <= 1.000021999999e+06
 T75_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t75_leaf1 >= -999999
 T75_L_f4_k2: const_Overall_Cond + 1e+06 t75_leaf2 <= 1.000003999999e+06
 T75_R_f36_k2: x_Wood_Deck_SF - 1e+06 t75_leaf2 >= -999978
 T75_R_f4_k3: const_Overall_Cond - 1e+06 t75_leaf3 >= -999996
 T75_L_f20_k3: x_Gr_Liv_Area + 1e+06 t75_leaf3 <= 1.001711999999e+06
 T75_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t75_leaf3 <= 1.001920999999e+06
 T75_R_f4_k4: const_Overall_Cond - 1e+06 t75_leaf4 >= -999996
 T75_L_f20_k4: x_Gr_Liv_Area + 1e+06 t75_leaf4 <= 1.001711999999e+06
 T75_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t75_leaf4 >= -998079
 T75_R_f4_k5: const_Overall_Cond - 1e+06 t75_leaf5 >= -999996
 T75_R_f20_k5: x_Gr_Liv_Area - 1e+06 t75_leaf5 >= -998288
 T75_L_f12_k5: x_BsmtFin_SF_1 + 1e+06 t75_leaf5 <= 1.000187999999e+06
 T75_R_f4_k6: const_Overall_Cond - 1e+06 t75_leaf6 >= -999996
 T75_R_f20_k6: x_Gr_Liv_Area - 1e+06 t75_leaf6 >= -998288
 T75_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t75_leaf6 >= -999812
 T76_onehot: t76_leaf0 + t76_leaf1 + t76_leaf2 + t76_leaf3 + t76_leaf4
   + t76_leaf5 + t76_leaf6 + t76_leaf7 = 1
 T76_L_f6_k0: const_Year_Remod_Add + 1e+06 t76_leaf0 <= 1.001983999999e+06
 T76_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t76_leaf0
   <= 1.000000999999e+06
 T76_L_f44_k0: const_Yr_Sold + 1e+06 t76_leaf0 <= 1.002007999999e+06
 T76_L_f6_k1: const_Year_Remod_Add + 1e+06 t76_leaf1 <= 1.001983999999e+06
 T76_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t76_leaf1
   <= 1.000000999999e+06
 T76_R_f44_k1: const_Yr_Sold - 1e+06 t76_leaf1 >= -997992
 T76_L_f6_k2: const_Year_Remod_Add + 1e+06 t76_leaf2 <= 1.001983999999e+06
 T76_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t76_leaf2 >= -999999
 T76_L_f0_k2: const_Lot_Frontage + 1e+06 t76_leaf2 <= 1.000071999999e+06
 T76_L_f6_k3: const_Year_Remod_Add + 1e+06 t76_leaf3 <= 1.001983999999e+06
 T76_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t76_leaf3 >= -999999
 T76_R_f0_k3: const_Lot_Frontage - 1e+06 t76_leaf3 >= -999928
 T76_R_f6_k4: const_Year_Remod_Add - 1e+06 t76_leaf4 >= -998016
 T76_L_f44_k4: const_Yr_Sold + 1e+06 t76_leaf4 <= 1.002006999999e+06
 T76_L_f36_k4: x_Wood_Deck_SF + 1e+06 t76_leaf4 <= 1.000192999999e+06
 T76_R_f6_k5: const_Year_Remod_Add - 1e+06 t76_leaf5 >= -998016
 T76_L_f44_k5: const_Yr_Sold + 1e+06 t76_leaf5 <= 1.002006999999e+06
 T76_R_f36_k5: x_Wood_Deck_SF - 1e+06 t76_leaf5 >= -999807
 T76_R_f6_k6: const_Year_Remod_Add - 1e+06 t76_leaf6 >= -998016
 T76_R_f44_k6: const_Yr_Sold - 1e+06 t76_leaf6 >= -997993
 T76_L_f28_k6: x_TotRms_AbvGrd + 1e+06 t76_leaf6 <= 1.000011999999e+06
 T76_R_f6_k7: const_Year_Remod_Add - 1e+06 t76_leaf7 >= -998016
 T76_R_f44_k7: const_Yr_Sold - 1e+06 t76_leaf7 >= -997993
 T76_R_f28_k7: x_TotRms_AbvGrd - 1e+06 t76_leaf7 >= -999988
 T77_onehot: t77_leaf0 + t77_leaf1 + t77_leaf2 + t77_leaf3 + t77_leaf4
   + t77_leaf5 + t77_leaf6 = 1
 T77_L_f4_k0: const_Overall_Cond + 1e+06 t77_leaf0 <= 1.000004999999e+06
 T77_L_f37_k0: x_Open_Porch_SF + 1e+06 t77_leaf0 <= 1.000157999999e+06
 T77_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t77_leaf0 <= 1.000401999999e+06
 T77_L_f4_k1: const_Overall_Cond + 1e+06 t77_leaf1 <= 1.000004999999e+06
 T77_L_f37_k1: x_Open_Porch_SF + 1e+06 t77_leaf1 <= 1.000157999999e+06
 T77_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t77_leaf1 >= -999598
 T77_L_f4_k2: const_Overall_Cond + 1e+06 t77_leaf2 <= 1.000004999999e+06
 T77_R_f37_k2: x_Open_Porch_SF - 1e+06 t77_leaf2 >= -999842
 T77_R_f4_k3: const_Overall_Cond - 1e+06 t77_leaf3 >= -999995
 T77_L_f44_k3: const_Yr_Sold + 1e+06 t77_leaf3 <= 1.002007999999e+06
 T77_L_f1_k3: const_Lot_Area + 1e+06 t77_leaf3 <= 1.009589999999e+06
 T77_R_f4_k4: const_Overall_Cond - 1e+06 t77_leaf4 >= -999995
 T77_L_f44_k4: const_Yr_Sold + 1e+06 t77_leaf4 <= 1.002007999999e+06
 T77_R_f1_k4: const_Lot_Area - 1e+06 t77_leaf4 >= -990410
 T77_R_f4_k5: const_Overall_Cond - 1e+06 t77_leaf5 >= -999995
 T77_R_f44_k5: const_Yr_Sold - 1e+06 t77_leaf5 >= -997992
 T77_L_f21_k5: const_Bsmt_Full_Bath + 1e+06 t77_leaf5
   <= 1.000000999999e+06
 T77_R_f4_k6: const_Overall_Cond - 1e+06 t77_leaf6 >= -999995
 T77_R_f44_k6: const_Yr_Sold - 1e+06 t77_leaf6 >= -997992
 T77_R_f21_k6: const_Bsmt_Full_Bath - 1e+06 t77_leaf6 >= -999999
 T78_onehot: t78_leaf0 + t78_leaf1 + t78_leaf2 + t78_leaf3 + t78_leaf4
   + t78_leaf5 + t78_leaf6 + t78_leaf7 = 1
 T78_L_f44_k0: const_Yr_Sold + 1e+06 t78_leaf0 <= 1.002007999999e+06
 T78_L_f20_k0: x_Gr_Liv_Area + 1e+06 t78_leaf0 <= 1.001733999999e+06
 T78_L_f0_k0: const_Lot_Frontage + 1e+06 t78_leaf0 <= 1.000030999999e+06
 T78_L_f44_k1: const_Yr_Sold + 1e+06 t78_leaf1 <= 1.002007999999e+06
 T78_L_f20_k1: x_Gr_Liv_Area + 1e+06 t78_leaf1 <= 1.001733999999e+06
 T78_R_f0_k1: const_Lot_Frontage - 1e+06 t78_leaf1 >= -999969
 T78_L_f44_k2: const_Yr_Sold + 1e+06 t78_leaf2 <= 1.002007999999e+06
 T78_R_f20_k2: x_Gr_Liv_Area - 1e+06 t78_leaf2 >= -998266
 T78_L_f5_k2: const_Year_Built + 1e+06 t78_leaf2 <= 1.001900999999e+06
 T78_L_f44_k3: const_Yr_Sold + 1e+06 t78_leaf3 <= 1.002007999999e+06
 T78_R_f20_k3: x_Gr_Liv_Area - 1e+06 t78_leaf3 >= -998266
 T78_R_f5_k3: const_Year_Built - 1e+06 t78_leaf3 >= -998099
 T78_R_f44_k4: const_Yr_Sold - 1e+06 t78_leaf4 >= -997992
 T78_L_f293_k4: const_Sale_Condition_Abnorml + 1e+06 t78_leaf4
   <= 1.000000999999e+06
 T78_L_f20_k4: x_Gr_Liv_Area + 1e+06 t78_leaf4 <= 1.001117999999e+06
 T78_R_f44_k5: const_Yr_Sold - 1e+06 t78_leaf5 >= -997992
 T78_L_f293_k5: const_Sale_Condition_Abnorml + 1e+06 t78_leaf5
   <= 1.000000999999e+06
 T78_R_f20_k5: x_Gr_Liv_Area - 1e+06 t78_leaf5 >= -998882
 T78_R_f44_k6: const_Yr_Sold - 1e+06 t78_leaf6 >= -997992
 T78_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t78_leaf6 >= -999999
 T78_L_f3_k6: Overall_Qual_calc + 1e+06 t78_leaf6 <= 1.000004999999e+06
 T78_R_f44_k7: const_Yr_Sold - 1e+06 t78_leaf7 >= -997992
 T78_R_f293_k7: const_Sale_Condition_Abnorml - 1e+06 t78_leaf7 >= -999999
 T78_R_f3_k7: Overall_Qual_calc - 1e+06 t78_leaf7 >= -999995
 T79_onehot: t79_leaf0 + t79_leaf1 + t79_leaf2 + t79_leaf3 + t79_leaf4
   + t79_leaf5 + t79_leaf6 = 1
 T79_L_f4_k0: const_Overall_Cond + 1e+06 t79_leaf0 <= 1.000003999999e+06
 T79_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t79_leaf0 <= 1.000003999999e+06
 T79_L_f16_k0: x_Heating_QC + 1e+06 t79_leaf0 <= 1.000002999999e+06
 T79_L_f4_k1: const_Overall_Cond + 1e+06 t79_leaf1 <= 1.000003999999e+06
 T79_L_f25_k1: x_Bedroom_AbvGr + 1e+06 t79_leaf1 <= 1.000003999999e+06
 T79_R_f16_k1: x_Heating_QC - 1e+06 t79_leaf1 >= -999997
 T79_L_f4_k2: const_Overall_Cond + 1e+06 t79_leaf2 <= 1.000003999999e+06
 T79_R_f25_k2: x_Bedroom_AbvGr - 1e+06 t79_leaf2 >= -999996
 T79_R_f4_k3: const_Overall_Cond - 1e+06 t79_leaf3 >= -999996
 T79_L_f95_k3: const_Neighborhood_Crawfor + 1e+06 t79_leaf3
   <= 1.000000999999e+06
 T79_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t79_leaf3 <= 1.001045999999e+06
 T79_R_f4_k4: const_Overall_Cond - 1e+06 t79_leaf4 >= -999996
 T79_L_f95_k4: const_Neighborhood_Crawfor + 1e+06 t79_leaf4
   <= 1.000000999999e+06
 T79_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t79_leaf4 >= -998954
 T79_R_f4_k5: const_Overall_Cond - 1e+06 t79_leaf5 >= -999996
 T79_R_f95_k5: const_Neighborhood_Crawfor - 1e+06 t79_leaf5 >= -999999
 T79_L_f5_k5: const_Year_Built + 1e+06 t79_leaf5 <= 1.001918999999e+06
 T79_R_f4_k6: const_Overall_Cond - 1e+06 t79_leaf6 >= -999996
 T79_R_f95_k6: const_Neighborhood_Crawfor - 1e+06 t79_leaf6 >= -999999
 T79_R_f5_k6: const_Year_Built - 1e+06 t79_leaf6 >= -998081
 T80_onehot: t80_leaf0 + t80_leaf1 + t80_leaf2 + t80_leaf3 + t80_leaf4
   + t80_leaf5 + t80_leaf6 = 1
 T80_L_f4_k0: const_Overall_Cond + 1e+06 t80_leaf0 <= 1.000004999999e+06
 T80_L_f40_k0: x_Screen_Porch + 1e+06 t80_leaf0 <= 1.000103999999e+06
 T80_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t80_leaf0
   <= 1.000000999999e+06
 T80_L_f4_k1: const_Overall_Cond + 1e+06 t80_leaf1 <= 1.000004999999e+06
 T80_L_f40_k1: x_Screen_Porch + 1e+06 t80_leaf1 <= 1.000103999999e+06
 T80_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t80_leaf1 >= -999999
 T80_L_f4_k2: const_Overall_Cond + 1e+06 t80_leaf2 <= 1.000004999999e+06
 T80_R_f40_k2: x_Screen_Porch - 1e+06 t80_leaf2 >= -999896
 T80_R_f4_k3: const_Overall_Cond - 1e+06 t80_leaf3 >= -999995
 T80_L_f1_k3: const_Lot_Area + 1e+06 t80_leaf3 <= 1.011899999999e+06
 T80_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t80_leaf3 <= 1.000169999999e+06
 T80_R_f4_k4: const_Overall_Cond - 1e+06 t80_leaf4 >= -999995
 T80_L_f1_k4: const_Lot_Area + 1e+06 t80_leaf4 <= 1.011899999999e+06
 T80_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t80_leaf4 >= -999830
 T80_R_f4_k5: const_Overall_Cond - 1e+06 t80_leaf5 >= -999995
 T80_R_f1_k5: const_Lot_Area - 1e+06 t80_leaf5 >= -988100
 T80_L_f17_k5: x_1st_Flr_SF + 1e+06 t80_leaf5 <= 1.001507999999e+06
 T80_R_f4_k6: const_Overall_Cond - 1e+06 t80_leaf6 >= -999995
 T80_R_f1_k6: const_Lot_Area - 1e+06 t80_leaf6 >= -988100
 T80_R_f17_k6: x_1st_Flr_SF - 1e+06 t80_leaf6 >= -998492
 T81_onehot: t81_leaf0 + t81_leaf1 + t81_leaf2 + t81_leaf3 + t81_leaf4 = 1
 T81_L_f4_k0: const_Overall_Cond + 1e+06 t81_leaf0 <= 1.000002999999e+06
 T81_R_f4_k1: const_Overall_Cond - 1e+06 t81_leaf1 >= -999997
 T81_L_f30_k1: x_Fireplace_Qu + 1e+06 t81_leaf1 <= 1.000000999999e+06
 T81_L_f1_k1: const_Lot_Area + 1e+06 t81_leaf1 <= 1.005975999999e+06
 T81_R_f4_k2: const_Overall_Cond - 1e+06 t81_leaf2 >= -999997
 T81_L_f30_k2: x_Fireplace_Qu + 1e+06 t81_leaf2 <= 1.000000999999e+06
 T81_R_f1_k2: const_Lot_Area - 1e+06 t81_leaf2 >= -994024
 T81_R_f4_k3: const_Overall_Cond - 1e+06 t81_leaf3 >= -999997
 T81_R_f30_k3: x_Fireplace_Qu - 1e+06 t81_leaf3 >= -999999
 T81_L_f1_k3: const_Lot_Area + 1e+06 t81_leaf3 <= 1.017226999999e+06
 T81_R_f4_k4: const_Overall_Cond - 1e+06 t81_leaf4 >= -999997
 T81_R_f30_k4: x_Fireplace_Qu - 1e+06 t81_leaf4 >= -999999
 T81_R_f1_k4: const_Lot_Area - 1e+06 t81_leaf4 >= -982773
 T82_onehot: t82_leaf0 + t82_leaf1 + t82_leaf2 + t82_leaf3 + t82_leaf4
   + t82_leaf5 + t82_leaf6 = 1
 T82_L_f20_k0: x_Gr_Liv_Area + 1e+06 t82_leaf0 <= 1.001799999999e+06
 T82_L_f46_k0: const_MS_SubClass_30 + 1e+06 t82_leaf0
   <= 1.000000999999e+06
 T82_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t82_leaf0 <= 1.000359999999e+06
 T82_L_f20_k1: x_Gr_Liv_Area + 1e+06 t82_leaf1 <= 1.001799999999e+06
 T82_L_f46_k1: const_MS_SubClass_30 + 1e+06 t82_leaf1
   <= 1.000000999999e+06
 T82_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t82_leaf1 >= -999640
 T82_L_f20_k2: x_Gr_Liv_Area + 1e+06 t82_leaf2 <= 1.001799999999e+06
 T82_R_f46_k2: const_MS_SubClass_30 - 1e+06 t82_leaf2 >= -999999
 T82_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t82_leaf2
   <= 1.000000999999e+06
 T82_L_f20_k3: x_Gr_Liv_Area + 1e+06 t82_leaf3 <= 1.001799999999e+06
 T82_R_f46_k3: const_MS_SubClass_30 - 1e+06 t82_leaf3 >= -999999
 T82_R_f21_k3: const_Bsmt_Full_Bath - 1e+06 t82_leaf3 >= -999999
 T82_R_f20_k4: x_Gr_Liv_Area - 1e+06 t82_leaf4 >= -998200
 T82_L_f40_k4: x_Screen_Porch + 1e+06 t82_leaf4 <= 1.000039999999e+06
 T82_L_f23_k4: x_Full_Bath + 1e+06 t82_leaf4 <= 1.000002999999e+06
 T82_R_f20_k5: x_Gr_Liv_Area - 1e+06 t82_leaf5 >= -998200
 T82_L_f40_k5: x_Screen_Porch + 1e+06 t82_leaf5 <= 1.000039999999e+06
 T82_R_f23_k5: x_Full_Bath - 1e+06 t82_leaf5 >= -999997
 T82_R_f20_k6: x_Gr_Liv_Area - 1e+06 t82_leaf6 >= -998200
 T82_R_f40_k6: x_Screen_Porch - 1e+06 t82_leaf6 >= -999960
 T83_onehot: t83_leaf0 + t83_leaf1 + t83_leaf2 + t83_leaf3 + t83_leaf4 = 1
 T83_L_f3_k0: Overall_Qual_calc + 1e+06 t83_leaf0 <= 1.000002999999e+06
 T83_R_f3_k1: Overall_Qual_calc - 1e+06 t83_leaf1 >= -999997
 T83_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t83_leaf1 <= 1.001052999999e+06
 T83_L_f1_k1: const_Lot_Area + 1e+06 t83_leaf1 <= 1.005975999999e+06
 T83_R_f3_k2: Overall_Qual_calc - 1e+06 t83_leaf2 >= -999997
 T83_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t83_leaf2 <= 1.001052999999e+06
 T83_R_f1_k2: const_Lot_Area - 1e+06 t83_leaf2 >= -994024
 T83_R_f3_k3: Overall_Qual_calc - 1e+06 t83_leaf3 >= -999997
 T83_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t83_leaf3 >= -998947
 T83_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t83_leaf3 <= 1.000455999999e+06
 T83_R_f3_k4: Overall_Qual_calc - 1e+06 t83_leaf4 >= -999997
 T83_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t83_leaf4 >= -998947
 T83_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t83_leaf4 >= -999544
 T84_onehot: t84_leaf0 + t84_leaf1 + t84_leaf2 + t84_leaf3 + t84_leaf4
   + t84_leaf5 + t84_leaf6 = 1
 T84_L_f4_k0: const_Overall_Cond + 1e+06 t84_leaf0 <= 1.000004999999e+06
 T84_L_f37_k0: x_Open_Porch_SF + 1e+06 t84_leaf0 <= 1.000071999999e+06
 T84_L_f40_k0: x_Screen_Porch + 1e+06 t84_leaf0 <= 1.000103999999e+06
 T84_L_f4_k1: const_Overall_Cond + 1e+06 t84_leaf1 <= 1.000004999999e+06
 T84_L_f37_k1: x_Open_Porch_SF + 1e+06 t84_leaf1 <= 1.000071999999e+06
 T84_R_f40_k1: x_Screen_Porch - 1e+06 t84_leaf1 >= -999896
 T84_L_f4_k2: const_Overall_Cond + 1e+06 t84_leaf2 <= 1.000004999999e+06
 T84_R_f37_k2: x_Open_Porch_SF - 1e+06 t84_leaf2 >= -999928
 T84_R_f4_k3: const_Overall_Cond - 1e+06 t84_leaf3 >= -999995
 T84_L_f20_k3: x_Gr_Liv_Area + 1e+06 t84_leaf3 <= 1.000987999999e+06
 T84_L_f261_k3: x_fence_is_GdWo + 1e+06 t84_leaf3 <= 1.000000999999e+06
 T84_R_f4_k4: const_Overall_Cond - 1e+06 t84_leaf4 >= -999995
 T84_L_f20_k4: x_Gr_Liv_Area + 1e+06 t84_leaf4 <= 1.000987999999e+06
 T84_R_f261_k4: x_fence_is_GdWo - 1e+06 t84_leaf4 >= -999999
 T84_R_f4_k5: const_Overall_Cond - 1e+06 t84_leaf5 >= -999995
 T84_R_f20_k5: x_Gr_Liv_Area - 1e+06 t84_leaf5 >= -999012
 T84_L_f23_k5: x_Full_Bath + 1e+06 t84_leaf5 <= 1.000002999999e+06
 T84_R_f4_k6: const_Overall_Cond - 1e+06 t84_leaf6 >= -999995
 T84_R_f20_k6: x_Gr_Liv_Area - 1e+06 t84_leaf6 >= -999012
 T84_R_f23_k6: x_Full_Bath - 1e+06 t84_leaf6 >= -999997
 T85_onehot: t85_leaf0 + t85_leaf1 + t85_leaf2 + t85_leaf3 + t85_leaf4
   + t85_leaf5 + t85_leaf6 = 1
 T85_L_f207_k0: const_Bsmt_Exposure_Gd + 1e+06 t85_leaf0
   <= 1.000000999999e+06
 T85_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t85_leaf0
   <= 1.000000999999e+06
 T85_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t85_leaf0 <= 1.000662999999e+06
 T85_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t85_leaf1
   <= 1.000000999999e+06
 T85_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t85_leaf1
   <= 1.000000999999e+06
 T85_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t85_leaf1 >= -999337
 T85_L_f207_k2: const_Bsmt_Exposure_Gd + 1e+06 t85_leaf2
   <= 1.000000999999e+06
 T85_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t85_leaf2 >= -999999
 T85_L_f3_k2: Overall_Qual_calc + 1e+06 t85_leaf2 <= 1.000004999999e+06
 T85_L_f207_k3: const_Bsmt_Exposure_Gd + 1e+06 t85_leaf3
   <= 1.000000999999e+06
 T85_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t85_leaf3 >= -999999
 T85_R_f3_k3: Overall_Qual_calc - 1e+06 t85_leaf3 >= -999995
 T85_R_f207_k4: const_Bsmt_Exposure_Gd - 1e+06 t85_leaf4 >= -999999
 T85_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t85_leaf4 <= 1.001437999999e+06
 T85_R_f207_k5: const_Bsmt_Exposure_Gd - 1e+06 t85_leaf5 >= -999999
 T85_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t85_leaf5 >= -998562
 T85_L_f14_k5: x_Bsmt_Unf_SF + 1e+06 t85_leaf5 <= 1.000995999999e+06
 T85_R_f207_k6: const_Bsmt_Exposure_Gd - 1e+06 t85_leaf6 >= -999999
 T85_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t85_leaf6 >= -998562
 T85_R_f14_k6: x_Bsmt_Unf_SF - 1e+06 t85_leaf6 >= -999004
 T86_onehot: t86_leaf0 + t86_leaf1 + t86_leaf2 + t86_leaf3 + t86_leaf4
   + t86_leaf5 + t86_leaf6 = 1
 T86_L_f46_k0: const_MS_SubClass_30 + 1e+06 t86_leaf0
   <= 1.000000999999e+06
 T86_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t86_leaf0 <= 1.000575999999e+06
 T86_L_f253_k0: const_Garage_Finish_Fin + 1e+06 t86_leaf0
   <= 1.000000999999e+06
 T86_L_f46_k1: const_MS_SubClass_30 + 1e+06 t86_leaf1
   <= 1.000000999999e+06
 T86_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t86_leaf1 <= 1.000575999999e+06
 T86_R_f253_k1: const_Garage_Finish_Fin - 1e+06 t86_leaf1 >= -999999
 T86_L_f46_k2: const_MS_SubClass_30 + 1e+06 t86_leaf2
   <= 1.000000999999e+06
 T86_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t86_leaf2 >= -999424
 T86_L_f1_k2: const_Lot_Area + 1e+06 t86_leaf2 <= 1.013810999999e+06
 T86_L_f46_k3: const_MS_SubClass_30 + 1e+06 t86_leaf3
   <= 1.000000999999e+06
 T86_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t86_leaf3 >= -999424
 T86_R_f1_k3: const_Lot_Area - 1e+06 t86_leaf3 >= -986189
 T86_R_f46_k4: const_MS_SubClass_30 - 1e+06 t86_leaf4 >= -999999
 T86_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t86_leaf4 <= 1.000760999999e+06
 T86_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t86_leaf4 <= 1.000662999999e+06
 T86_R_f46_k5: const_MS_SubClass_30 - 1e+06 t86_leaf5 >= -999999
 T86_L_f15_k5: const_Total_Bsmt_SF + 1e+06 t86_leaf5 <= 1.000760999999e+06
 T86_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t86_leaf5 >= -999337
 T86_R_f46_k6: const_MS_SubClass_30 - 1e+06 t86_leaf6 >= -999999
 T86_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t86_leaf6 >= -999239
 T87_onehot: t87_leaf0 + t87_leaf1 + t87_leaf2 + t87_leaf3 + t87_leaf4
   + t87_leaf5 + t87_leaf6 = 1
 T87_L_f4_k0: const_Overall_Cond + 1e+06 t87_leaf0 <= 1.000003999999e+06
 T87_L_f73_k0: const_Lot_Shape_IR1 + 1e+06 t87_leaf0 <= 1.000000999999e+06
 T87_L_f226_k0: x_Heating_GasA + 1e+06 t87_leaf0 <= 1.000000999999e+06
 T87_L_f4_k1: const_Overall_Cond + 1e+06 t87_leaf1 <= 1.000003999999e+06
 T87_L_f73_k1: const_Lot_Shape_IR1 + 1e+06 t87_leaf1 <= 1.000000999999e+06
 T87_R_f226_k1: x_Heating_GasA - 1e+06 t87_leaf1 >= -999999
 T87_L_f4_k2: const_Overall_Cond + 1e+06 t87_leaf2 <= 1.000003999999e+06
 T87_R_f73_k2: const_Lot_Shape_IR1 - 1e+06 t87_leaf2 >= -999999
 T87_R_f4_k3: const_Overall_Cond - 1e+06 t87_leaf3 >= -999996
 T87_L_f44_k3: const_Yr_Sold + 1e+06 t87_leaf3 <= 1.002007999999e+06
 T87_L_f18_k3: const_2nd_Flr_SF + 1e+06 t87_leaf3 <= 1.001404999999e+06
 T87_R_f4_k4: const_Overall_Cond - 1e+06 t87_leaf4 >= -999996
 T87_L_f44_k4: const_Yr_Sold + 1e+06 t87_leaf4 <= 1.002007999999e+06
 T87_R_f18_k4: const_2nd_Flr_SF - 1e+06 t87_leaf4 >= -998595
 T87_R_f4_k5: const_Overall_Cond - 1e+06 t87_leaf5 >= -999996
 T87_R_f44_k5: const_Yr_Sold - 1e+06 t87_leaf5 >= -997992
 T87_L_f20_k5: x_Gr_Liv_Area + 1e+06 t87_leaf5 <= 1.001145999999e+06
 T87_R_f4_k6: const_Overall_Cond - 1e+06 t87_leaf6 >= -999996
 T87_R_f44_k6: const_Yr_Sold - 1e+06 t87_leaf6 >= -997992
 T87_R_f20_k6: x_Gr_Liv_Area - 1e+06 t87_leaf6 >= -998854
 T88_onehot: t88_leaf0 + t88_leaf1 + t88_leaf2 + t88_leaf3 + t88_leaf4
   + t88_leaf5 + t88_leaf6 = 1
 T88_L_f4_k0: const_Overall_Cond + 1e+06 t88_leaf0 <= 1.000003999999e+06
 T88_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t88_leaf0 <= 1.001399999999e+06
 T88_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t88_leaf0
   <= 1.000000999999e+06
 T88_L_f4_k1: const_Overall_Cond + 1e+06 t88_leaf1 <= 1.000003999999e+06
 T88_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t88_leaf1 <= 1.001399999999e+06
 T88_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t88_leaf1 >= -999999
 T88_L_f4_k2: const_Overall_Cond + 1e+06 t88_leaf2 <= 1.000003999999e+06
 T88_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t88_leaf2 >= -998600
 T88_R_f4_k3: const_Overall_Cond - 1e+06 t88_leaf3 >= -999996
 T88_L_f44_k3: const_Yr_Sold + 1e+06 t88_leaf3 <= 1.002007999999e+06
 T88_L_f17_k3: x_1st_Flr_SF + 1e+06 t88_leaf3 <= 1.001739999999e+06
 T88_R_f4_k4: const_Overall_Cond - 1e+06 t88_leaf4 >= -999996
 T88_L_f44_k4: const_Yr_Sold + 1e+06 t88_leaf4 <= 1.002007999999e+06
 T88_R_f17_k4: x_1st_Flr_SF - 1e+06 t88_leaf4 >= -998260
 T88_R_f4_k5: const_Overall_Cond - 1e+06 t88_leaf5 >= -999996
 T88_R_f44_k5: const_Yr_Sold - 1e+06 t88_leaf5 >= -997992
 T88_L_f15_k5: const_Total_Bsmt_SF + 1e+06 t88_leaf5 <= 1.000764999999e+06
 T88_R_f4_k6: const_Overall_Cond - 1e+06 t88_leaf6 >= -999996
 T88_R_f44_k6: const_Yr_Sold - 1e+06 t88_leaf6 >= -997992
 T88_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t88_leaf6 >= -999235
 T89_onehot: t89_leaf0 + t89_leaf1 + t89_leaf2 + t89_leaf3 + t89_leaf4
   + t89_leaf5 + t89_leaf6 + t89_leaf7 = 1
 T89_L_f11_k0: x_Bsmt_Cond + 1e+06 t89_leaf0 <= 1.000001999999e+06
 T89_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t89_leaf0
   <= 1.000000999999e+06
 T89_L_f38_k0: x_Enclosed_Porch + 1e+06 t89_leaf0 <= 1.000191999999e+06
 T89_L_f11_k1: x_Bsmt_Cond + 1e+06 t89_leaf1 <= 1.000001999999e+06
 T89_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t89_leaf1
   <= 1.000000999999e+06
 T89_R_f38_k1: x_Enclosed_Porch - 1e+06 t89_leaf1 >= -999808
 T89_L_f11_k2: x_Bsmt_Cond + 1e+06 t89_leaf2 <= 1.000001999999e+06
 T89_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t89_leaf2 >= -999999
 T89_L_f28_k2: x_TotRms_AbvGrd + 1e+06 t89_leaf2 <= 1.000005999999e+06
 T89_L_f11_k3: x_Bsmt_Cond + 1e+06 t89_leaf3 <= 1.000001999999e+06
 T89_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t89_leaf3 >= -999999
 T89_R_f28_k3: x_TotRms_AbvGrd - 1e+06 t89_leaf3 >= -999994
 T89_R_f11_k4: x_Bsmt_Cond - 1e+06 t89_leaf4 >= -999998
 T89_L_f20_k4: x_Gr_Liv_Area + 1e+06 t89_leaf4 <= 1.001748999999e+06
 T89_L_f117_k4: const_Condition_1_Artery + 1e+06 t89_leaf4
   <= 1.000000999999e+06
 T89_R_f11_k5: x_Bsmt_Cond - 1e+06 t89_leaf5 >= -999998
 T89_L_f20_k5: x_Gr_Liv_Area + 1e+06 t89_leaf5 <= 1.001748999999e+06
 T89_R_f117_k5: const_Condition_1_Artery - 1e+06 t89_leaf5 >= -999999
 T89_R_f11_k6: x_Bsmt_Cond - 1e+06 t89_leaf6 >= -999998
 T89_R_f20_k6: x_Gr_Liv_Area - 1e+06 t89_leaf6 >= -998251
 T89_L_f40_k6: x_Screen_Porch + 1e+06 t89_leaf6 <= 1.000039999999e+06
 T89_R_f11_k7: x_Bsmt_Cond - 1e+06 t89_leaf7 >= -999998
 T89_R_f20_k7: x_Gr_Liv_Area - 1e+06 t89_leaf7 >= -998251
 T89_R_f40_k7: x_Screen_Porch - 1e+06 t89_leaf7 >= -999960
 T90_onehot: t90_leaf0 + t90_leaf1 + t90_leaf2 + t90_leaf3 + t90_leaf4
   + t90_leaf5 + t90_leaf6 + t90_leaf7 = 1
 T90_L_f44_k0: const_Yr_Sold + 1e+06 t90_leaf0 <= 1.002007999999e+06
 T90_L_f134_k0: const_Bldg_Type_1Fam + 1e+06 t90_leaf0
   <= 1.000000999999e+06
 T90_L_f37_k0: x_Open_Porch_SF + 1e+06 t90_leaf0 <= 1.000003999999e+06
 T90_L_f44_k1: const_Yr_Sold + 1e+06 t90_leaf1 <= 1.002007999999e+06
 T90_L_f134_k1: const_Bldg_Type_1Fam + 1e+06 t90_leaf1
   <= 1.000000999999e+06
 T90_R_f37_k1: x_Open_Porch_SF - 1e+06 t90_leaf1 >= -999996
 T90_L_f44_k2: const_Yr_Sold + 1e+06 t90_leaf2 <= 1.002007999999e+06
 T90_R_f134_k2: const_Bldg_Type_1Fam - 1e+06 t90_leaf2 >= -999999
 T90_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t90_leaf2 <= 1.001688999999e+06
 T90_L_f44_k3: const_Yr_Sold + 1e+06 t90_leaf3 <= 1.002007999999e+06
 T90_R_f134_k3: const_Bldg_Type_1Fam - 1e+06 t90_leaf3 >= -999999
 T90_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t90_leaf3 >= -998311
 T90_R_f44_k4: const_Yr_Sold - 1e+06 t90_leaf4 >= -997992
 T90_L_f293_k4: const_Sale_Condition_Abnorml + 1e+06 t90_leaf4
   <= 1.000000999999e+06
 T90_L_f37_k4: x_Open_Porch_SF + 1e+06 t90_leaf4 <= 1.000030999999e+06
 T90_R_f44_k5: const_Yr_Sold - 1e+06 t90_leaf5 >= -997992
 T90_L_f293_k5: const_Sale_Condition_Abnorml + 1e+06 t90_leaf5
   <= 1.000000999999e+06
 T90_R_f37_k5: x_Open_Porch_SF - 1e+06 t90_leaf5 >= -999969
 T90_R_f44_k6: const_Yr_Sold - 1e+06 t90_leaf6 >= -997992
 T90_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t90_leaf6 >= -999999
 T90_L_f85_k6: const_Lot_Config_Inside + 1e+06 t90_leaf6
   <= 1.000000999999e+06
 T90_R_f44_k7: const_Yr_Sold - 1e+06 t90_leaf7 >= -997992
 T90_R_f293_k7: const_Sale_Condition_Abnorml - 1e+06 t90_leaf7 >= -999999
 T90_R_f85_k7: const_Lot_Config_Inside - 1e+06 t90_leaf7 >= -999999
 T91_onehot: t91_leaf0 + t91_leaf1 + t91_leaf2 + t91_leaf3 + t91_leaf4
   + t91_leaf5 + t91_leaf6 + t91_leaf7 = 1
 T91_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t91_leaf0 <= 1.000291999999e+06
 T91_L_f16_k0: x_Heating_QC + 1e+06 t91_leaf0 <= 1.000002999999e+06
 T91_L_f66_k0: const_MS_Zoning_RL + 1e+06 t91_leaf0 <= 1.000000999999e+06
 T91_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t91_leaf1 <= 1.000291999999e+06
 T91_L_f16_k1: x_Heating_QC + 1e+06 t91_leaf1 <= 1.000002999999e+06
 T91_R_f66_k1: const_MS_Zoning_RL - 1e+06 t91_leaf1 >= -999999
 T91_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t91_leaf2 <= 1.000291999999e+06
 T91_R_f16_k2: x_Heating_QC - 1e+06 t91_leaf2 >= -999997
 T91_L_f1_k2: const_Lot_Area + 1e+06 t91_leaf2 <= 1.007439999999e+06
 T91_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t91_leaf3 <= 1.000291999999e+06
 T91_R_f16_k3: x_Heating_QC - 1e+06 t91_leaf3 >= -999997
 T91_R_f1_k3: const_Lot_Area - 1e+06 t91_leaf3 >= -992560
 T91_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t91_leaf4 >= -999708
 T91_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t91_leaf4 <= 1.001418999999e+06
 T91_L_f1_k4: const_Lot_Area + 1e+06 t91_leaf4 <= 1.006124999999e+06
 T91_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t91_leaf5 >= -999708
 T91_L_f15_k5: const_Total_Bsmt_SF + 1e+06 t91_leaf5 <= 1.001418999999e+06
 T91_R_f1_k5: const_Lot_Area - 1e+06 t91_leaf5 >= -993875
 T91_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t91_leaf6 >= -999708
 T91_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t91_leaf6 >= -998581
 T91_L_f118_k6: const_Condition_1_Feedr + 1e+06 t91_leaf6
   <= 1.000000999999e+06
 T91_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t91_leaf7 >= -999708
 T91_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t91_leaf7 >= -998581
 T91_R_f118_k7: const_Condition_1_Feedr - 1e+06 t91_leaf7 >= -999999
 T92_onehot: t92_leaf0 + t92_leaf1 + t92_leaf2 + t92_leaf3 + t92_leaf4
   + t92_leaf5 + t92_leaf6 = 1
 T92_L_f4_k0: const_Overall_Cond + 1e+06 t92_leaf0 <= 1.000003999999e+06
 T92_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t92_leaf0
   <= 1.000000999999e+06
 T92_L_f231_k0: central_air_no + 1e+06 t92_leaf0 <= 1.000000999999e+06
 T92_L_f4_k1: const_Overall_Cond + 1e+06 t92_leaf1 <= 1.000003999999e+06
 T92_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t92_leaf1
   <= 1.000000999999e+06
 T92_R_f231_k1: central_air_no - 1e+06 t92_leaf1 >= -999999
 T92_L_f4_k2: const_Overall_Cond + 1e+06 t92_leaf2 <= 1.000003999999e+06
 T92_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t92_leaf2 >= -999999
 T92_R_f4_k3: const_Overall_Cond - 1e+06 t92_leaf3 >= -999996
 T92_L_f23_k3: x_Full_Bath + 1e+06 t92_leaf3 <= 1.000002999999e+06
 T92_L_f44_k3: const_Yr_Sold + 1e+06 t92_leaf3 <= 1.002007999999e+06
 T92_R_f4_k4: const_Overall_Cond - 1e+06 t92_leaf4 >= -999996
 T92_L_f23_k4: x_Full_Bath + 1e+06 t92_leaf4 <= 1.000002999999e+06
 T92_R_f44_k4: const_Yr_Sold - 1e+06 t92_leaf4 >= -997992
 T92_R_f4_k5: const_Overall_Cond - 1e+06 t92_leaf5 >= -999996
 T92_R_f23_k5: x_Full_Bath - 1e+06 t92_leaf5 >= -999997
 T92_L_f174_k5: x_ex1_is_VinylSd + 1e+06 t92_leaf5 <= 1.000000999999e+06
 T92_R_f4_k6: const_Overall_Cond - 1e+06 t92_leaf6 >= -999996
 T92_R_f23_k6: x_Full_Bath - 1e+06 t92_leaf6 >= -999997
 T92_R_f174_k6: x_ex1_is_VinylSd - 1e+06 t92_leaf6 >= -999999
 T93_onehot: t93_leaf0 + t93_leaf1 + t93_leaf2 + t93_leaf3 + t93_leaf4
   + t93_leaf5 + t93_leaf6 + t93_leaf7 = 1
 T93_L_f44_k0: const_Yr_Sold + 1e+06 t93_leaf0 <= 1.002007999999e+06
 T93_L_f20_k0: x_Gr_Liv_Area + 1e+06 t93_leaf0 <= 1.001967999999e+06
 T93_L_f4_k0: const_Overall_Cond + 1e+06 t93_leaf0 <= 1.000003999999e+06
 T93_L_f44_k1: const_Yr_Sold + 1e+06 t93_leaf1 <= 1.002007999999e+06
 T93_L_f20_k1: x_Gr_Liv_Area + 1e+06 t93_leaf1 <= 1.001967999999e+06
 T93_R_f4_k1: const_Overall_Cond - 1e+06 t93_leaf1 >= -999996
 T93_L_f44_k2: const_Yr_Sold + 1e+06 t93_leaf2 <= 1.002007999999e+06
 T93_R_f20_k2: x_Gr_Liv_Area - 1e+06 t93_leaf2 >= -998032
 T93_L_f267_k2: const_Misc_Feature_No_aplica + 1e+06 t93_leaf2
   <= 1.000000999999e+06
 T93_L_f44_k3: const_Yr_Sold + 1e+06 t93_leaf3 <= 1.002007999999e+06
 T93_R_f20_k3: x_Gr_Liv_Area - 1e+06 t93_leaf3 >= -998032
 T93_R_f267_k3: const_Misc_Feature_No_aplica - 1e+06 t93_leaf3 >= -999999
 T93_R_f44_k4: const_Yr_Sold - 1e+06 t93_leaf4 >= -997992
 T93_L_f77_k4: const_Land_Contour_Bnk + 1e+06 t93_leaf4
   <= 1.000000999999e+06
 T93_L_f33_k4: x_Garage_Area + 1e+06 t93_leaf4 <= 1.000350999999e+06
 T93_R_f44_k5: const_Yr_Sold - 1e+06 t93_leaf5 >= -997992
 T93_L_f77_k5: const_Land_Contour_Bnk + 1e+06 t93_leaf5
   <= 1.000000999999e+06
 T93_R_f33_k5: x_Garage_Area - 1e+06 t93_leaf5 >= -999649
 T93_R_f44_k6: const_Yr_Sold - 1e+06 t93_leaf6 >= -997992
 T93_R_f77_k6: const_Land_Contour_Bnk - 1e+06 t93_leaf6 >= -999999
 T93_L_f17_k6: x_1st_Flr_SF + 1e+06 t93_leaf6 <= 1.001016999999e+06
 T93_R_f44_k7: const_Yr_Sold - 1e+06 t93_leaf7 >= -997992
 T93_R_f77_k7: const_Land_Contour_Bnk - 1e+06 t93_leaf7 >= -999999
 T93_R_f17_k7: x_1st_Flr_SF - 1e+06 t93_leaf7 >= -998983
 T94_onehot: t94_leaf0 + t94_leaf1 + t94_leaf2 + t94_leaf3 + t94_leaf4
   + t94_leaf5 + t94_leaf6 = 1
 T94_L_f29_k0: const_Fireplaces + 1e+06 t94_leaf0 <= 1.000001999999e+06
 T94_L_f298_k0: const_Sale_Condition_Partial + 1e+06 t94_leaf0
   <= 1.000000999999e+06
 T94_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t94_leaf0
   <= 1.000000999999e+06
 T94_L_f29_k1: const_Fireplaces + 1e+06 t94_leaf1 <= 1.000001999999e+06
 T94_L_f298_k1: const_Sale_Condition_Partial + 1e+06 t94_leaf1
   <= 1.000000999999e+06
 T94_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t94_leaf1 >= -999999
 T94_L_f29_k2: const_Fireplaces + 1e+06 t94_leaf2 <= 1.000001999999e+06
 T94_R_f298_k2: const_Sale_Condition_Partial - 1e+06 t94_leaf2 >= -999999
 T94_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t94_leaf2 <= 1.000637999999e+06
 T94_L_f29_k3: const_Fireplaces + 1e+06 t94_leaf3 <= 1.000001999999e+06
 T94_R_f298_k3: const_Sale_Condition_Partial - 1e+06 t94_leaf3 >= -999999
 T94_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t94_leaf3 >= -999362
 T94_R_f29_k4: const_Fireplaces - 1e+06 t94_leaf4 >= -999998
 T94_L_f43_k4: const_Misc_Val + 1e+06 t94_leaf4 <= 1.000053999999e+06
 T94_L_f20_k4: x_Gr_Liv_Area + 1e+06 t94_leaf4 <= 1.002133999999e+06
 T94_R_f29_k5: const_Fireplaces - 1e+06 t94_leaf5 >= -999998
 T94_L_f43_k5: const_Misc_Val + 1e+06 t94_leaf5 <= 1.000053999999e+06
 T94_R_f20_k5: x_Gr_Liv_Area - 1e+06 t94_leaf5 >= -997866
 T94_R_f29_k6: const_Fireplaces - 1e+06 t94_leaf6 >= -999998
 T94_R_f43_k6: const_Misc_Val - 1e+06 t94_leaf6 >= -999946
 T95_onehot: t95_leaf0 + t95_leaf1 + t95_leaf2 + t95_leaf3 + t95_leaf4
   + t95_leaf5 + t95_leaf6 + t95_leaf7 = 1
 T95_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t95_leaf0 <= 1.001928999999e+06
 T95_L_f20_k0: x_Gr_Liv_Area + 1e+06 t95_leaf0 <= 1.001199999999e+06
 T95_L_f5_k0: const_Year_Built + 1e+06 t95_leaf0 <= 1.001923999999e+06
 T95_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t95_leaf1 <= 1.001928999999e+06
 T95_L_f20_k1: x_Gr_Liv_Area + 1e+06 t95_leaf1 <= 1.001199999999e+06
 T95_R_f5_k1: const_Year_Built - 1e+06 t95_leaf1 >= -998076
 T95_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t95_leaf2 <= 1.001928999999e+06
 T95_R_f20_k2: x_Gr_Liv_Area - 1e+06 t95_leaf2 >= -998800
 T95_L_f5_k2: const_Year_Built + 1e+06 t95_leaf2 <= 1.001910999999e+06
 T95_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t95_leaf3 <= 1.001928999999e+06
 T95_R_f20_k3: x_Gr_Liv_Area - 1e+06 t95_leaf3 >= -998800
 T95_R_f5_k3: const_Year_Built - 1e+06 t95_leaf3 >= -998089
 T95_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t95_leaf4 >= -998071
 T95_L_f6_k4: const_Year_Remod_Add + 1e+06 t95_leaf4 <= 1.001994999999e+06
 T95_L_f13_k4: x_BsmtFin_SF_2 + 1e+06 t95_leaf4 <= 1.000531999999e+06
 T95_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t95_leaf5 >= -998071
 T95_L_f6_k5: const_Year_Remod_Add + 1e+06 t95_leaf5 <= 1.001994999999e+06
 T95_R_f13_k5: x_BsmtFin_SF_2 - 1e+06 t95_leaf5 >= -999468
 T95_R_f31_k6: const_Garage_Yr_Blt - 1e+06 t95_leaf6 >= -998071
 T95_R_f6_k6: const_Year_Remod_Add - 1e+06 t95_leaf6 >= -998005
 T95_L_f20_k6: x_Gr_Liv_Area + 1e+06 t95_leaf6 <= 1.001967999999e+06
 T95_R_f31_k7: const_Garage_Yr_Blt - 1e+06 t95_leaf7 >= -998071
 T95_R_f6_k7: const_Year_Remod_Add - 1e+06 t95_leaf7 >= -998005
 T95_R_f20_k7: x_Gr_Liv_Area - 1e+06 t95_leaf7 >= -998032
 T96_onehot: t96_leaf0 + t96_leaf1 + t96_leaf2 + t96_leaf3 + t96_leaf4
   + t96_leaf5 + t96_leaf6 = 1
 T96_L_f9_k0: x_Exter_Cond + 1e+06 t96_leaf0 <= 1.000001999999e+06
 T96_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t96_leaf0
   <= 1.000000999999e+06
 T96_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t96_leaf0 <= 1.001920999999e+06
 T96_L_f9_k1: x_Exter_Cond + 1e+06 t96_leaf1 <= 1.000001999999e+06
 T96_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t96_leaf1
   <= 1.000000999999e+06
 T96_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t96_leaf1 >= -998079
 T96_L_f9_k2: x_Exter_Cond + 1e+06 t96_leaf2 <= 1.000001999999e+06
 T96_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t96_leaf2 >= -999999
 T96_R_f9_k3: x_Exter_Cond - 1e+06 t96_leaf3 >= -999998
 T96_L_f20_k3: x_Gr_Liv_Area + 1e+06 t96_leaf3 <= 1.000987999999e+06
 T96_L_f4_k3: const_Overall_Cond + 1e+06 t96_leaf3 <= 1.000006999999e+06
 T96_R_f9_k4: x_Exter_Cond - 1e+06 t96_leaf4 >= -999998
 T96_L_f20_k4: x_Gr_Liv_Area + 1e+06 t96_leaf4 <= 1.000987999999e+06
 T96_R_f4_k4: const_Overall_Cond - 1e+06 t96_leaf4 >= -999993
 T96_R_f9_k5: x_Exter_Cond - 1e+06 t96_leaf5 >= -999998
 T96_R_f20_k5: x_Gr_Liv_Area - 1e+06 t96_leaf5 >= -999012
 T96_L_f44_k5: const_Yr_Sold + 1e+06 t96_leaf5 <= 1.002006999999e+06
 T96_R_f9_k6: x_Exter_Cond - 1e+06 t96_leaf6 >= -999998
 T96_R_f20_k6: x_Gr_Liv_Area - 1e+06 t96_leaf6 >= -999012
 T96_R_f44_k6: const_Yr_Sold - 1e+06 t96_leaf6 >= -997993
 T97_onehot: t97_leaf0 + t97_leaf1 + t97_leaf2 + t97_leaf3 + t97_leaf4
   + t97_leaf5 + t97_leaf6 + t97_leaf7 = 1
 T97_L_f46_k0: const_MS_SubClass_30 + 1e+06 t97_leaf0
   <= 1.000000999999e+06
 T97_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t97_leaf0
   <= 1.000000999999e+06
 T97_L_f16_k0: x_Heating_QC + 1e+06 t97_leaf0 <= 1.000002999999e+06
 T97_L_f46_k1: const_MS_SubClass_30 + 1e+06 t97_leaf1
   <= 1.000000999999e+06
 T97_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t97_leaf1
   <= 1.000000999999e+06
 T97_R_f16_k1: x_Heating_QC - 1e+06 t97_leaf1 >= -999997
 T97_L_f46_k2: const_MS_SubClass_30 + 1e+06 t97_leaf2
   <= 1.000000999999e+06
 T97_R_f95_k2: const_Neighborhood_Crawfor - 1e+06 t97_leaf2 >= -999999
 T97_L_f20_k2: x_Gr_Liv_Area + 1e+06 t97_leaf2 <= 1.001503999999e+06
 T97_L_f46_k3: const_MS_SubClass_30 + 1e+06 t97_leaf3
   <= 1.000000999999e+06
 T97_R_f95_k3: const_Neighborhood_Crawfor - 1e+06 t97_leaf3 >= -999999
 T97_R_f20_k3: x_Gr_Liv_Area - 1e+06 t97_leaf3 >= -998496
 T97_R_f46_k4: const_MS_SubClass_30 - 1e+06 t97_leaf4 >= -999999
 T97_L_f16_k4: x_Heating_QC + 1e+06 t97_leaf4 <= 1.000002999999e+06
 T97_L_f5_k4: const_Year_Built + 1e+06 t97_leaf4 <= 1.001924999999e+06
 T97_R_f46_k5: const_MS_SubClass_30 - 1e+06 t97_leaf5 >= -999999
 T97_L_f16_k5: x_Heating_QC + 1e+06 t97_leaf5 <= 1.000002999999e+06
 T97_R_f5_k5: const_Year_Built - 1e+06 t97_leaf5 >= -998075
 T97_R_f46_k6: const_MS_SubClass_30 - 1e+06 t97_leaf6 >= -999999
 T97_R_f16_k6: x_Heating_QC - 1e+06 t97_leaf6 >= -999997
 T97_L_f25_k6: x_Bedroom_AbvGr + 1e+06 t97_leaf6 <= 1.000002999999e+06
 T97_R_f46_k7: const_MS_SubClass_30 - 1e+06 t97_leaf7 >= -999999
 T97_R_f16_k7: x_Heating_QC - 1e+06 t97_leaf7 >= -999997
 T97_R_f25_k7: x_Bedroom_AbvGr - 1e+06 t97_leaf7 >= -999997
 T98_onehot: t98_leaf0 + t98_leaf1 + t98_leaf2 + t98_leaf3 + t98_leaf4
   + t98_leaf5 + t98_leaf6 = 1
 T98_L_f4_k0: const_Overall_Cond + 1e+06 t98_leaf0 <= 1.000003999999e+06
 T98_L_f16_k0: x_Heating_QC + 1e+06 t98_leaf0 <= 1.000003999999e+06
 T98_L_f44_k0: const_Yr_Sold + 1e+06 t98_leaf0 <= 1.002006999999e+06
 T98_L_f4_k1: const_Overall_Cond + 1e+06 t98_leaf1 <= 1.000003999999e+06
 T98_L_f16_k1: x_Heating_QC + 1e+06 t98_leaf1 <= 1.000003999999e+06
 T98_R_f44_k1: const_Yr_Sold - 1e+06 t98_leaf1 >= -997993
 T98_L_f4_k2: const_Overall_Cond + 1e+06 t98_leaf2 <= 1.000003999999e+06
 T98_R_f16_k2: x_Heating_QC - 1e+06 t98_leaf2 >= -999996
 T98_R_f4_k3: const_Overall_Cond - 1e+06 t98_leaf3 >= -999996
 T98_L_f20_k3: x_Gr_Liv_Area + 1e+06 t98_leaf3 <= 1.002648999999e+06
 T98_L_f5_k3: const_Year_Built + 1e+06 t98_leaf3 <= 1.002005999999e+06
 T98_R_f4_k4: const_Overall_Cond - 1e+06 t98_leaf4 >= -999996
 T98_L_f20_k4: x_Gr_Liv_Area + 1e+06 t98_leaf4 <= 1.002648999999e+06
 T98_R_f5_k4: const_Year_Built - 1e+06 t98_leaf4 >= -997994
 T98_R_f4_k5: const_Overall_Cond - 1e+06 t98_leaf5 >= -999996
 T98_R_f20_k5: x_Gr_Liv_Area - 1e+06 t98_leaf5 >= -997351
 T98_L_f25_k5: x_Bedroom_AbvGr + 1e+06 t98_leaf5 <= 1.000003999999e+06
 T98_R_f4_k6: const_Overall_Cond - 1e+06 t98_leaf6 >= -999996
 T98_R_f20_k6: x_Gr_Liv_Area - 1e+06 t98_leaf6 >= -997351
 T98_R_f25_k6: x_Bedroom_AbvGr - 1e+06 t98_leaf6 >= -999996
 T99_onehot: t99_leaf0 + t99_leaf1 + t99_leaf2 + t99_leaf3 + t99_leaf4
   + t99_leaf5 + t99_leaf6 = 1
 T99_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t99_leaf0
   <= 1.000000999999e+06
 T99_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t99_leaf0 <= 1.000291999999e+06
 T99_L_f6_k0: const_Year_Remod_Add + 1e+06 t99_leaf0 <= 1.001983999999e+06
 T99_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t99_leaf1
   <= 1.000000999999e+06
 T99_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t99_leaf1 <= 1.000291999999e+06
 T99_R_f6_k1: const_Year_Remod_Add - 1e+06 t99_leaf1 >= -998016
 T99_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t99_leaf2
   <= 1.000000999999e+06
 T99_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t99_leaf2 >= -999708
 T99_L_f6_k2: const_Year_Remod_Add + 1e+06 t99_leaf2 <= 1.001993999999e+06
 T99_L_f95_k3: const_Neighborhood_Crawfor + 1e+06 t99_leaf3
   <= 1.000000999999e+06
 T99_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t99_leaf3 >= -999708
 T99_R_f6_k3: const_Year_Remod_Add - 1e+06 t99_leaf3 >= -998006
 T99_R_f95_k4: const_Neighborhood_Crawfor - 1e+06 t99_leaf4 >= -999999
 T99_L_f14_k4: x_Bsmt_Unf_SF + 1e+06 t99_leaf4 <= 1.000409999999e+06
 T99_R_f95_k5: const_Neighborhood_Crawfor - 1e+06 t99_leaf5 >= -999999
 T99_R_f14_k5: x_Bsmt_Unf_SF - 1e+06 t99_leaf5 >= -999590
 T99_L_f4_k5: const_Overall_Cond + 1e+06 t99_leaf5 <= 1.000005999999e+06
 T99_R_f95_k6: const_Neighborhood_Crawfor - 1e+06 t99_leaf6 >= -999999
 T99_R_f14_k6: x_Bsmt_Unf_SF - 1e+06 t99_leaf6 >= -999590
 T99_R_f4_k6: const_Overall_Cond - 1e+06 t99_leaf6 >= -999994
 T100_onehot: t100_leaf0 + t100_leaf1 + t100_leaf2 + t100_leaf3
   + t100_leaf4 + t100_leaf5 + t100_leaf6 + t100_leaf7 = 1
 T100_L_f20_k0: x_Gr_Liv_Area + 1e+06 t100_leaf0 <= 1.000832999999e+06
 T100_L_f100_k0: const_Neighborhood_IDOTRR + 1e+06 t100_leaf0
   <= 1.000000999999e+06
 T100_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t100_leaf0
   <= 1.000647999999e+06
 T100_L_f20_k1: x_Gr_Liv_Area + 1e+06 t100_leaf1 <= 1.000832999999e+06
 T100_L_f100_k1: const_Neighborhood_IDOTRR + 1e+06 t100_leaf1
   <= 1.000000999999e+06
 T100_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t100_leaf1 >= -999352
 T100_L_f20_k2: x_Gr_Liv_Area + 1e+06 t100_leaf2 <= 1.000832999999e+06
 T100_R_f100_k2: const_Neighborhood_IDOTRR - 1e+06 t100_leaf2 >= -999999
 T100_L_f217_k2: x_b1_is_Unf + 1e+06 t100_leaf2 <= 1.000000999999e+06
 T100_L_f20_k3: x_Gr_Liv_Area + 1e+06 t100_leaf3 <= 1.000832999999e+06
 T100_R_f100_k3: const_Neighborhood_IDOTRR - 1e+06 t100_leaf3 >= -999999
 T100_R_f217_k3: x_b1_is_Unf - 1e+06 t100_leaf3 >= -999999
 T100_R_f20_k4: x_Gr_Liv_Area - 1e+06 t100_leaf4 >= -999167
 T100_L_f21_k4: const_Bsmt_Full_Bath + 1e+06 t100_leaf4
   <= 1.000000999999e+06
 T100_L_f245_k4: const_FunctioNo_aplical_Typ + 1e+06 t100_leaf4
   <= 1.000000999999e+06
 T100_R_f20_k5: x_Gr_Liv_Area - 1e+06 t100_leaf5 >= -999167
 T100_L_f21_k5: const_Bsmt_Full_Bath + 1e+06 t100_leaf5
   <= 1.000000999999e+06
 T100_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t100_leaf5 >= -999999
 T100_R_f20_k6: x_Gr_Liv_Area - 1e+06 t100_leaf6 >= -999167
 T100_R_f21_k6: const_Bsmt_Full_Bath - 1e+06 t100_leaf6 >= -999999
 T100_L_f33_k6: x_Garage_Area + 1e+06 t100_leaf6 <= 1.001153999999e+06
 T100_R_f20_k7: x_Gr_Liv_Area - 1e+06 t100_leaf7 >= -999167
 T100_R_f21_k7: const_Bsmt_Full_Bath - 1e+06 t100_leaf7 >= -999999
 T100_R_f33_k7: x_Garage_Area - 1e+06 t100_leaf7 >= -998846
 T101_onehot: t101_leaf0 + t101_leaf1 + t101_leaf2 + t101_leaf3
   + t101_leaf4 + t101_leaf5 + t101_leaf6 = 1
 T101_L_f4_k0: const_Overall_Cond + 1e+06 t101_leaf0 <= 1.000003999999e+06
 T101_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t101_leaf0 <= 1.000369999999e+06
 T101_L_f4_k1: const_Overall_Cond + 1e+06 t101_leaf1 <= 1.000003999999e+06
 T101_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t101_leaf1 >= -999630
 T101_L_f275_k1: const_Mo_Sold_5 + 1e+06 t101_leaf1 <= 1.000000999999e+06
 T101_L_f4_k2: const_Overall_Cond + 1e+06 t101_leaf2 <= 1.000003999999e+06
 T101_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t101_leaf2 >= -999630
 T101_R_f275_k2: const_Mo_Sold_5 - 1e+06 t101_leaf2 >= -999999
 T101_R_f4_k3: const_Overall_Cond - 1e+06 t101_leaf3 >= -999996
 T101_L_f95_k3: const_Neighborhood_Crawfor + 1e+06 t101_leaf3
   <= 1.000000999999e+06
 T101_L_f33_k3: x_Garage_Area + 1e+06 t101_leaf3 <= 1.001153999999e+06
 T101_R_f4_k4: const_Overall_Cond - 1e+06 t101_leaf4 >= -999996
 T101_L_f95_k4: const_Neighborhood_Crawfor + 1e+06 t101_leaf4
   <= 1.000000999999e+06
 T101_R_f33_k4: x_Garage_Area - 1e+06 t101_leaf4 >= -998846
 T101_R_f4_k5: const_Overall_Cond - 1e+06 t101_leaf5 >= -999996
 T101_R_f95_k5: const_Neighborhood_Crawfor - 1e+06 t101_leaf5 >= -999999
 T101_L_f20_k5: x_Gr_Liv_Area + 1e+06 t101_leaf5 <= 1.001503999999e+06
 T101_R_f4_k6: const_Overall_Cond - 1e+06 t101_leaf6 >= -999996
 T101_R_f95_k6: const_Neighborhood_Crawfor - 1e+06 t101_leaf6 >= -999999
 T101_R_f20_k6: x_Gr_Liv_Area - 1e+06 t101_leaf6 >= -998496
 T102_onehot: t102_leaf0 + t102_leaf1 + t102_leaf2 + t102_leaf3
   + t102_leaf4 + t102_leaf5 + t102_leaf6 + t102_leaf7 = 1
 T102_L_f3_k0: Overall_Qual_calc + 1e+06 t102_leaf0 <= 1.000004999999e+06
 T102_L_f100_k0: const_Neighborhood_IDOTRR + 1e+06 t102_leaf0
   <= 1.000000999999e+06
 T102_L_f37_k0: x_Open_Porch_SF + 1e+06 t102_leaf0 <= 1.000038999999e+06
 T102_L_f3_k1: Overall_Qual_calc + 1e+06 t102_leaf1 <= 1.000004999999e+06
 T102_L_f100_k1: const_Neighborhood_IDOTRR + 1e+06 t102_leaf1
   <= 1.000000999999e+06
 T102_R_f37_k1: x_Open_Porch_SF - 1e+06 t102_leaf1 >= -999961
 T102_L_f3_k2: Overall_Qual_calc + 1e+06 t102_leaf2 <= 1.000004999999e+06
 T102_R_f100_k2: const_Neighborhood_IDOTRR - 1e+06 t102_leaf2 >= -999999
 T102_L_f44_k2: const_Yr_Sold + 1e+06 t102_leaf2 <= 1.002006999999e+06
 T102_L_f3_k3: Overall_Qual_calc + 1e+06 t102_leaf3 <= 1.000004999999e+06
 T102_R_f100_k3: const_Neighborhood_IDOTRR - 1e+06 t102_leaf3 >= -999999
 T102_R_f44_k3: const_Yr_Sold - 1e+06 t102_leaf3 >= -997993
 T102_R_f3_k4: Overall_Qual_calc - 1e+06 t102_leaf4 >= -999995
 T102_L_f207_k4: const_Bsmt_Exposure_Gd + 1e+06 t102_leaf4
   <= 1.000000999999e+06
 T102_L_f23_k4: x_Full_Bath + 1e+06 t102_leaf4 <= 1.000002999999e+06
 T102_R_f3_k5: Overall_Qual_calc - 1e+06 t102_leaf5 >= -999995
 T102_L_f207_k5: const_Bsmt_Exposure_Gd + 1e+06 t102_leaf5
   <= 1.000000999999e+06
 T102_R_f23_k5: x_Full_Bath - 1e+06 t102_leaf5 >= -999997
 T102_R_f3_k6: Overall_Qual_calc - 1e+06 t102_leaf6 >= -999995
 T102_R_f207_k6: const_Bsmt_Exposure_Gd - 1e+06 t102_leaf6 >= -999999
 T102_L_f6_k6: const_Year_Remod_Add + 1e+06 t102_leaf6
   <= 1.002008999999e+06
 T102_R_f3_k7: Overall_Qual_calc - 1e+06 t102_leaf7 >= -999995
 T102_R_f207_k7: const_Bsmt_Exposure_Gd - 1e+06 t102_leaf7 >= -999999
 T102_R_f6_k7: const_Year_Remod_Add - 1e+06 t102_leaf7 >= -997991
 T103_onehot: t103_leaf0 + t103_leaf1 + t103_leaf2 + t103_leaf3
   + t103_leaf4 + t103_leaf5 = 1
 T103_L_f4_k0: const_Overall_Cond + 1e+06 t103_leaf0 <= 1.000004999999e+06
 T103_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t103_leaf0
   <= 1.001361999999e+06
 T103_L_f4_k1: const_Overall_Cond + 1e+06 t103_leaf1 <= 1.000004999999e+06
 T103_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t103_leaf1 >= -998638
 T103_R_f4_k2: const_Overall_Cond - 1e+06 t103_leaf2 >= -999995
 T103_L_f23_k2: x_Full_Bath + 1e+06 t103_leaf2 <= 1.000002999999e+06
 T103_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t103_leaf2
   <= 1.000000999999e+06
 T103_R_f4_k3: const_Overall_Cond - 1e+06 t103_leaf3 >= -999995
 T103_L_f23_k3: x_Full_Bath + 1e+06 t103_leaf3 <= 1.000002999999e+06
 T103_R_f95_k3: const_Neighborhood_Crawfor - 1e+06 t103_leaf3 >= -999999
 T103_R_f4_k4: const_Overall_Cond - 1e+06 t103_leaf4 >= -999995
 T103_R_f23_k4: x_Full_Bath - 1e+06 t103_leaf4 >= -999997
 T103_L_f36_k4: x_Wood_Deck_SF + 1e+06 t103_leaf4 <= 1.000115999999e+06
 T103_R_f4_k5: const_Overall_Cond - 1e+06 t103_leaf5 >= -999995
 T103_R_f23_k5: x_Full_Bath - 1e+06 t103_leaf5 >= -999997
 T103_R_f36_k5: x_Wood_Deck_SF - 1e+06 t103_leaf5 >= -999884
 T104_onehot: t104_leaf0 + t104_leaf1 + t104_leaf2 + t104_leaf3
   + t104_leaf4 + t104_leaf5 = 1
 T104_L_f3_k0: Overall_Qual_calc + 1e+06 t104_leaf0 <= 1.000002999999e+06
 T104_L_f33_k0: x_Garage_Area + 1e+06 t104_leaf0 <= 1.000285999999e+06
 T104_L_f3_k1: Overall_Qual_calc + 1e+06 t104_leaf1 <= 1.000002999999e+06
 T104_R_f33_k1: x_Garage_Area - 1e+06 t104_leaf1 >= -999714
 T104_R_f3_k2: Overall_Qual_calc - 1e+06 t104_leaf2 >= -999997
 T104_L_f1_k2: const_Lot_Area + 1e+06 t104_leaf2 <= 1.014679999999e+06
 T104_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t104_leaf2
   <= 1.001899999999e+06
 T104_R_f3_k3: Overall_Qual_calc - 1e+06 t104_leaf3 >= -999997
 T104_L_f1_k3: const_Lot_Area + 1e+06 t104_leaf3 <= 1.014679999999e+06
 T104_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t104_leaf3 >= -998100
 T104_R_f3_k4: Overall_Qual_calc - 1e+06 t104_leaf4 >= -999997
 T104_R_f1_k4: const_Lot_Area - 1e+06 t104_leaf4 >= -985320
 T104_L_f18_k4: const_2nd_Flr_SF + 1e+06 t104_leaf4 <= 1.001275999999e+06
 T104_R_f3_k5: Overall_Qual_calc - 1e+06 t104_leaf5 >= -999997
 T104_R_f1_k5: const_Lot_Area - 1e+06 t104_leaf5 >= -985320
 T104_R_f18_k5: const_2nd_Flr_SF - 1e+06 t104_leaf5 >= -998724
 T105_onehot: t105_leaf0 + t105_leaf1 + t105_leaf2 + t105_leaf3
   + t105_leaf4 + t105_leaf5 + t105_leaf6 + t105_leaf7 = 1
 T105_L_f11_k0: x_Bsmt_Cond + 1e+06 t105_leaf0 <= 1.000001999999e+06
 T105_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t105_leaf0
   <= 1.000000999999e+06
 T105_L_f17_k0: x_1st_Flr_SF + 1e+06 t105_leaf0 <= 1.000979999999e+06
 T105_L_f11_k1: x_Bsmt_Cond + 1e+06 t105_leaf1 <= 1.000001999999e+06
 T105_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t105_leaf1
   <= 1.000000999999e+06
 T105_R_f17_k1: x_1st_Flr_SF - 1e+06 t105_leaf1 >= -999020
 T105_L_f11_k2: x_Bsmt_Cond + 1e+06 t105_leaf2 <= 1.000001999999e+06
 T105_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t105_leaf2
   >= -999999
 T105_L_f16_k2: x_Heating_QC + 1e+06 t105_leaf2 <= 1.000002999999e+06
 T105_L_f11_k3: x_Bsmt_Cond + 1e+06 t105_leaf3 <= 1.000001999999e+06
 T105_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t105_leaf3
   >= -999999
 T105_R_f16_k3: x_Heating_QC - 1e+06 t105_leaf3 >= -999997
 T105_R_f11_k4: x_Bsmt_Cond - 1e+06 t105_leaf4 >= -999998
 T105_L_f4_k4: const_Overall_Cond + 1e+06 t105_leaf4 <= 1.000006999999e+06
 T105_L_f5_k4: const_Year_Built + 1e+06 t105_leaf4 <= 1.001977999999e+06
 T105_R_f11_k5: x_Bsmt_Cond - 1e+06 t105_leaf5 >= -999998
 T105_L_f4_k5: const_Overall_Cond + 1e+06 t105_leaf5 <= 1.000006999999e+06
 T105_R_f5_k5: const_Year_Built - 1e+06 t105_leaf5 >= -998022
 T105_R_f11_k6: x_Bsmt_Cond - 1e+06 t105_leaf6 >= -999998
 T105_R_f4_k6: const_Overall_Cond - 1e+06 t105_leaf6 >= -999993
 T105_L_f17_k6: x_1st_Flr_SF + 1e+06 t105_leaf6 <= 1.001099999999e+06
 T105_R_f11_k7: x_Bsmt_Cond - 1e+06 t105_leaf7 >= -999998
 T105_R_f4_k7: const_Overall_Cond - 1e+06 t105_leaf7 >= -999993
 T105_R_f17_k7: x_1st_Flr_SF - 1e+06 t105_leaf7 >= -998900
 T106_onehot: t106_leaf0 + t106_leaf1 + t106_leaf2 + t106_leaf3
   + t106_leaf4 + t106_leaf5 + t106_leaf6 = 1
 T106_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t106_leaf0
   <= 1.000000999999e+06
 T106_L_f20_k0: x_Gr_Liv_Area + 1e+06 t106_leaf0 <= 1.001145999999e+06
 T106_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t106_leaf0
   <= 1.000000999999e+06
 T106_L_f21_k1: const_Bsmt_Full_Bath + 1e+06 t106_leaf1
   <= 1.000000999999e+06
 T106_L_f20_k1: x_Gr_Liv_Area + 1e+06 t106_leaf1 <= 1.001145999999e+06
 T106_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t106_leaf1 >= -999999
 T106_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t106_leaf2
   <= 1.000000999999e+06
 T106_R_f20_k2: x_Gr_Liv_Area - 1e+06 t106_leaf2 >= -998854
 T106_L_f4_k2: const_Overall_Cond + 1e+06 t106_leaf2 <= 1.000006999999e+06
 T106_L_f21_k3: const_Bsmt_Full_Bath + 1e+06 t106_leaf3
   <= 1.000000999999e+06
 T106_R_f20_k3: x_Gr_Liv_Area - 1e+06 t106_leaf3 >= -998854
 T106_R_f4_k3: const_Overall_Cond - 1e+06 t106_leaf3 >= -999993
 T106_R_f21_k4: const_Bsmt_Full_Bath - 1e+06 t106_leaf4 >= -999999
 T106_L_f28_k4: x_TotRms_AbvGrd + 1e+06 t106_leaf4 <= 1.000011999999e+06
 T106_L_f1_k4: const_Lot_Area + 1e+06 t106_leaf4 <= 1.018385999999e+06
 T106_R_f21_k5: const_Bsmt_Full_Bath - 1e+06 t106_leaf5 >= -999999
 T106_L_f28_k5: x_TotRms_AbvGrd + 1e+06 t106_leaf5 <= 1.000011999999e+06
 T106_R_f1_k5: const_Lot_Area - 1e+06 t106_leaf5 >= -981614
 T106_R_f21_k6: const_Bsmt_Full_Bath - 1e+06 t106_leaf6 >= -999999
 T106_R_f28_k6: x_TotRms_AbvGrd - 1e+06 t106_leaf6 >= -999988
 T107_onehot: t107_leaf0 + t107_leaf1 + t107_leaf2 + t107_leaf3
   + t107_leaf4 + t107_leaf5 + t107_leaf6 + t107_leaf7 = 1
 T107_L_f44_k0: const_Yr_Sold + 1e+06 t107_leaf0 <= 1.002007999999e+06
 T107_L_f1_k0: const_Lot_Area + 1e+06 t107_leaf0 <= 1.013810999999e+06
 T107_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t107_leaf0 <= 1.000222999999e+06
 T107_L_f44_k1: const_Yr_Sold + 1e+06 t107_leaf1 <= 1.002007999999e+06
 T107_L_f1_k1: const_Lot_Area + 1e+06 t107_leaf1 <= 1.013810999999e+06
 T107_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t107_leaf1 >= -999777
 T107_L_f44_k2: const_Yr_Sold + 1e+06 t107_leaf2 <= 1.002007999999e+06
 T107_R_f1_k2: const_Lot_Area - 1e+06 t107_leaf2 >= -986189
 T107_L_f37_k2: x_Open_Porch_SF + 1e+06 t107_leaf2 <= 1.000272999999e+06
 T107_L_f44_k3: const_Yr_Sold + 1e+06 t107_leaf3 <= 1.002007999999e+06
 T107_R_f1_k3: const_Lot_Area - 1e+06 t107_leaf3 >= -986189
 T107_R_f37_k3: x_Open_Porch_SF - 1e+06 t107_leaf3 >= -999727
 T107_R_f44_k4: const_Yr_Sold - 1e+06 t107_leaf4 >= -997992
 T107_L_f100_k4: const_Neighborhood_IDOTRR + 1e+06 t107_leaf4
   <= 1.000000999999e+06
 T107_L_f118_k4: const_Condition_1_Feedr + 1e+06 t107_leaf4
   <= 1.000000999999e+06
 T107_R_f44_k5: const_Yr_Sold - 1e+06 t107_leaf5 >= -997992
 T107_L_f100_k5: const_Neighborhood_IDOTRR + 1e+06 t107_leaf5
   <= 1.000000999999e+06
 T107_R_f118_k5: const_Condition_1_Feedr - 1e+06 t107_leaf5 >= -999999
 T107_R_f44_k6: const_Yr_Sold - 1e+06 t107_leaf6 >= -997992
 T107_R_f100_k6: const_Neighborhood_IDOTRR - 1e+06 t107_leaf6 >= -999999
 T107_L_f0_k6: const_Lot_Frontage + 1e+06 t107_leaf6 <= 1.000062999999e+06
 T107_R_f44_k7: const_Yr_Sold - 1e+06 t107_leaf7 >= -997992
 T107_R_f100_k7: const_Neighborhood_IDOTRR - 1e+06 t107_leaf7 >= -999999
 T107_R_f0_k7: const_Lot_Frontage - 1e+06 t107_leaf7 >= -999937
 T108_onehot: t108_leaf0 + t108_leaf1 + t108_leaf2 + t108_leaf3
   + t108_leaf4 + t108_leaf5 + t108_leaf6 + t108_leaf7 = 1
 T108_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t108_leaf0
   <= 1.000000999999e+06
 T108_L_f76_k0: const_Lot_Shape_Reg + 1e+06 t108_leaf0
   <= 1.000000999999e+06
 T108_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t108_leaf0 <= 1.000000999999e+06
 T108_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t108_leaf1
   <= 1.000000999999e+06
 T108_L_f76_k1: const_Lot_Shape_Reg + 1e+06 t108_leaf1
   <= 1.000000999999e+06
 T108_R_f164_k1: x_ex1_is_BrkFace - 1e+06 t108_leaf1 >= -999999
 T108_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t108_leaf2
   <= 1.000000999999e+06
 T108_R_f76_k2: const_Lot_Shape_Reg - 1e+06 t108_leaf2 >= -999999
 T108_L_f1_k2: const_Lot_Area + 1e+06 t108_leaf2 <= 1.002644999999e+06
 T108_L_f293_k3: const_Sale_Condition_Abnorml + 1e+06 t108_leaf3
   <= 1.000000999999e+06
 T108_R_f76_k3: const_Lot_Shape_Reg - 1e+06 t108_leaf3 >= -999999
 T108_R_f1_k3: const_Lot_Area - 1e+06 t108_leaf3 >= -997355
 T108_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t108_leaf4
   >= -999999
 T108_L_f3_k4: Overall_Qual_calc + 1e+06 t108_leaf4 <= 1.000004999999e+06
 T108_L_f169_k4: x_ex1_is_MetalSd + 1e+06 t108_leaf4 <= 1.000000999999e+06
 T108_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t108_leaf5
   >= -999999
 T108_L_f3_k5: Overall_Qual_calc + 1e+06 t108_leaf5 <= 1.000004999999e+06
 T108_R_f169_k5: x_ex1_is_MetalSd - 1e+06 t108_leaf5 >= -999999
 T108_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t108_leaf6
   >= -999999
 T108_R_f3_k6: Overall_Qual_calc - 1e+06 t108_leaf6 >= -999995
 T108_L_f27_k6: x_Kitchen_Qual + 1e+06 t108_leaf6 <= 1.000003999999e+06
 T108_R_f293_k7: const_Sale_Condition_Abnorml - 1e+06 t108_leaf7
   >= -999999
 T108_R_f3_k7: Overall_Qual_calc - 1e+06 t108_leaf7 >= -999995
 T108_R_f27_k7: x_Kitchen_Qual - 1e+06 t108_leaf7 >= -999996
 T109_onehot: t109_leaf0 + t109_leaf1 + t109_leaf2 + t109_leaf3
   + t109_leaf4 + t109_leaf5 + t109_leaf6 = 1
 T109_L_f9_k0: x_Exter_Cond + 1e+06 t109_leaf0 <= 1.000001999999e+06
 T109_L_f3_k0: Overall_Qual_calc + 1e+06 t109_leaf0 <= 1.000002999999e+06
 T109_L_f9_k1: x_Exter_Cond + 1e+06 t109_leaf1 <= 1.000001999999e+06
 T109_R_f3_k1: Overall_Qual_calc - 1e+06 t109_leaf1 >= -999997
 T109_L_f33_k1: x_Garage_Area + 1e+06 t109_leaf1 <= 1.000350999999e+06
 T109_L_f9_k2: x_Exter_Cond + 1e+06 t109_leaf2 <= 1.000001999999e+06
 T109_R_f3_k2: Overall_Qual_calc - 1e+06 t109_leaf2 >= -999997
 T109_R_f33_k2: x_Garage_Area - 1e+06 t109_leaf2 >= -999649
 T109_R_f9_k3: x_Exter_Cond - 1e+06 t109_leaf3 >= -999998
 T109_L_f20_k3: x_Gr_Liv_Area + 1e+06 t109_leaf3 <= 1.002262999999e+06
 T109_L_f44_k3: const_Yr_Sold + 1e+06 t109_leaf3 <= 1.002007999999e+06
 T109_R_f9_k4: x_Exter_Cond - 1e+06 t109_leaf4 >= -999998
 T109_L_f20_k4: x_Gr_Liv_Area + 1e+06 t109_leaf4 <= 1.002262999999e+06
 T109_R_f44_k4: const_Yr_Sold - 1e+06 t109_leaf4 >= -997992
 T109_R_f9_k5: x_Exter_Cond - 1e+06 t109_leaf5 >= -999998
 T109_R_f20_k5: x_Gr_Liv_Area - 1e+06 t109_leaf5 >= -997737
 T109_L_f17_k5: x_1st_Flr_SF + 1e+06 t109_leaf5 <= 1.001240999999e+06
 T109_R_f9_k6: x_Exter_Cond - 1e+06 t109_leaf6 >= -999998
 T109_R_f20_k6: x_Gr_Liv_Area - 1e+06 t109_leaf6 >= -997737
 T109_R_f17_k6: x_1st_Flr_SF - 1e+06 t109_leaf6 >= -998759
 T110_onehot: t110_leaf0 + t110_leaf1 + t110_leaf2 + t110_leaf3
   + t110_leaf4 + t110_leaf5 = 1
 T110_L_f40_k0: x_Screen_Porch + 1e+06 t110_leaf0 <= 1.000103999999e+06
 T110_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t110_leaf0 <= 1.000222999999e+06
 T110_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t110_leaf0
   <= 1.000000999999e+06
 T110_L_f40_k1: x_Screen_Porch + 1e+06 t110_leaf1 <= 1.000103999999e+06
 T110_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t110_leaf1 <= 1.000222999999e+06
 T110_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t110_leaf1
   >= -999999
 T110_L_f40_k2: x_Screen_Porch + 1e+06 t110_leaf2 <= 1.000103999999e+06
 T110_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t110_leaf2 >= -999777
 T110_L_f18_k2: const_2nd_Flr_SF + 1e+06 t110_leaf2 <= 1.001628999999e+06
 T110_L_f40_k3: x_Screen_Porch + 1e+06 t110_leaf3 <= 1.000103999999e+06
 T110_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t110_leaf3 >= -999777
 T110_R_f18_k3: const_2nd_Flr_SF - 1e+06 t110_leaf3 >= -998371
 T110_R_f40_k4: x_Screen_Porch - 1e+06 t110_leaf4 >= -999896
 T110_L_f23_k4: x_Full_Bath + 1e+06 t110_leaf4 <= 1.000001999999e+06
 T110_R_f40_k5: x_Screen_Porch - 1e+06 t110_leaf5 >= -999896
 T110_R_f23_k5: x_Full_Bath - 1e+06 t110_leaf5 >= -999998
 T111_onehot: t111_leaf0 + t111_leaf1 + t111_leaf2 + t111_leaf3
   + t111_leaf4 + t111_leaf5 + t111_leaf6 + t111_leaf7 = 1
 T111_L_f44_k0: const_Yr_Sold + 1e+06 t111_leaf0 <= 1.002007999999e+06
 T111_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t111_leaf0 <= 1.000200999999e+06
 T111_L_f36_k0: x_Wood_Deck_SF + 1e+06 t111_leaf0 <= 1.000157999999e+06
 T111_L_f44_k1: const_Yr_Sold + 1e+06 t111_leaf1 <= 1.002007999999e+06
 T111_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t111_leaf1 <= 1.000200999999e+06
 T111_R_f36_k1: x_Wood_Deck_SF - 1e+06 t111_leaf1 >= -999842
 T111_L_f44_k2: const_Yr_Sold + 1e+06 t111_leaf2 <= 1.002007999999e+06
 T111_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t111_leaf2 >= -999799
 T111_L_f4_k2: const_Overall_Cond + 1e+06 t111_leaf2 <= 1.000005999999e+06
 T111_L_f44_k3: const_Yr_Sold + 1e+06 t111_leaf3 <= 1.002007999999e+06
 T111_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t111_leaf3 >= -999799
 T111_R_f4_k3: const_Overall_Cond - 1e+06 t111_leaf3 >= -999994
 T111_R_f44_k4: const_Yr_Sold - 1e+06 t111_leaf4 >= -997992
 T111_L_f100_k4: const_Neighborhood_IDOTRR + 1e+06 t111_leaf4
   <= 1.000000999999e+06
 T111_L_f4_k4: const_Overall_Cond + 1e+06 t111_leaf4 <= 1.000006999999e+06
 T111_R_f44_k5: const_Yr_Sold - 1e+06 t111_leaf5 >= -997992
 T111_L_f100_k5: const_Neighborhood_IDOTRR + 1e+06 t111_leaf5
   <= 1.000000999999e+06
 T111_R_f4_k5: const_Overall_Cond - 1e+06 t111_leaf5 >= -999993
 T111_R_f44_k6: const_Yr_Sold - 1e+06 t111_leaf6 >= -997992
 T111_R_f100_k6: const_Neighborhood_IDOTRR - 1e+06 t111_leaf6 >= -999999
 T111_L_f20_k6: x_Gr_Liv_Area + 1e+06 t111_leaf6 <= 1.001133999999e+06
 T111_R_f44_k7: const_Yr_Sold - 1e+06 t111_leaf7 >= -997992
 T111_R_f100_k7: const_Neighborhood_IDOTRR - 1e+06 t111_leaf7 >= -999999
 T111_R_f20_k7: x_Gr_Liv_Area - 1e+06 t111_leaf7 >= -998866
 T112_onehot: t112_leaf0 + t112_leaf1 + t112_leaf2 + t112_leaf3
   + t112_leaf4 + t112_leaf5 + t112_leaf6 + t112_leaf7 = 1
 T112_L_f73_k0: const_Lot_Shape_IR1 + 1e+06 t112_leaf0
   <= 1.000000999999e+06
 T112_L_f18_k0: const_2nd_Flr_SF + 1e+06 t112_leaf0 <= 1.000977999999e+06
 T112_L_f11_k0: x_Bsmt_Cond + 1e+06 t112_leaf0 <= 1.000001999999e+06
 T112_L_f73_k1: const_Lot_Shape_IR1 + 1e+06 t112_leaf1
   <= 1.000000999999e+06
 T112_L_f18_k1: const_2nd_Flr_SF + 1e+06 t112_leaf1 <= 1.000977999999e+06
 T112_R_f11_k1: x_Bsmt_Cond - 1e+06 t112_leaf1 >= -999998
 T112_L_f73_k2: const_Lot_Shape_IR1 + 1e+06 t112_leaf2
   <= 1.000000999999e+06
 T112_R_f18_k2: const_2nd_Flr_SF - 1e+06 t112_leaf2 >= -999022
 T112_L_f4_k2: const_Overall_Cond + 1e+06 t112_leaf2 <= 1.000005999999e+06
 T112_L_f73_k3: const_Lot_Shape_IR1 + 1e+06 t112_leaf3
   <= 1.000000999999e+06
 T112_R_f18_k3: const_2nd_Flr_SF - 1e+06 t112_leaf3 >= -999022
 T112_R_f4_k3: const_Overall_Cond - 1e+06 t112_leaf3 >= -999994
 T112_R_f73_k4: const_Lot_Shape_IR1 - 1e+06 t112_leaf4 >= -999999
 T112_L_f44_k4: const_Yr_Sold + 1e+06 t112_leaf4 <= 1.002007999999e+06
 T112_L_f1_k4: const_Lot_Area + 1e+06 t112_leaf4 <= 1.015137999999e+06
 T112_R_f73_k5: const_Lot_Shape_IR1 - 1e+06 t112_leaf5 >= -999999
 T112_L_f44_k5: const_Yr_Sold + 1e+06 t112_leaf5 <= 1.002007999999e+06
 T112_R_f1_k5: const_Lot_Area - 1e+06 t112_leaf5 >= -984862
 T112_R_f73_k6: const_Lot_Shape_IR1 - 1e+06 t112_leaf6 >= -999999
 T112_R_f44_k6: const_Yr_Sold - 1e+06 t112_leaf6 >= -997992
 T112_L_f298_k6: const_Sale_Condition_Partial + 1e+06 t112_leaf6
   <= 1.000000999999e+06
 T112_R_f73_k7: const_Lot_Shape_IR1 - 1e+06 t112_leaf7 >= -999999
 T112_R_f44_k7: const_Yr_Sold - 1e+06 t112_leaf7 >= -997992
 T112_R_f298_k7: const_Sale_Condition_Partial - 1e+06 t112_leaf7
   >= -999999
 T113_onehot: t113_leaf0 + t113_leaf1 + t113_leaf2 + t113_leaf3
   + t113_leaf4 + t113_leaf5 + t113_leaf6 = 1
 T113_L_f20_k0: x_Gr_Liv_Area + 1e+06 t113_leaf0 <= 1.000832999999e+06
 T113_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t113_leaf0
   <= 1.000000999999e+06
 T113_L_f38_k0: x_Enclosed_Porch + 1e+06 t113_leaf0 <= 1.000034999999e+06
 T113_L_f20_k1: x_Gr_Liv_Area + 1e+06 t113_leaf1 <= 1.000832999999e+06
 T113_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t113_leaf1
   <= 1.000000999999e+06
 T113_R_f38_k1: x_Enclosed_Porch - 1e+06 t113_leaf1 >= -999965
 T113_L_f20_k2: x_Gr_Liv_Area + 1e+06 t113_leaf2 <= 1.000832999999e+06
 T113_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t113_leaf2
   >= -999999
 T113_R_f20_k3: x_Gr_Liv_Area - 1e+06 t113_leaf3 >= -999167
 T113_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t113_leaf3
   <= 1.001437999999e+06
 T113_L_f95_k3: const_Neighborhood_Crawfor + 1e+06 t113_leaf3
   <= 1.000000999999e+06
 T113_R_f20_k4: x_Gr_Liv_Area - 1e+06 t113_leaf4 >= -999167
 T113_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t113_leaf4
   <= 1.001437999999e+06
 T113_R_f95_k4: const_Neighborhood_Crawfor - 1e+06 t113_leaf4 >= -999999
 T113_R_f20_k5: x_Gr_Liv_Area - 1e+06 t113_leaf5 >= -999167
 T113_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t113_leaf5 >= -998562
 T113_L_f37_k5: x_Open_Porch_SF + 1e+06 t113_leaf5 <= 1.000234999999e+06
 T113_R_f20_k6: x_Gr_Liv_Area - 1e+06 t113_leaf6 >= -999167
 T113_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t113_leaf6 >= -998562
 T113_R_f37_k6: x_Open_Porch_SF - 1e+06 t113_leaf6 >= -999765
 T114_onehot: t114_leaf0 + t114_leaf1 + t114_leaf2 + t114_leaf3
   + t114_leaf4 + t114_leaf5 + t114_leaf6 + t114_leaf7 = 1
 T114_L_f1_k0: const_Lot_Area + 1e+06 t114_leaf0 <= 1.007263999999e+06
 T114_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t114_leaf0 <= 1.000769999999e+06
 T114_L_f3_k0: Overall_Qual_calc + 1e+06 t114_leaf0 <= 1.000004999999e+06
 T114_L_f1_k1: const_Lot_Area + 1e+06 t114_leaf1 <= 1.007263999999e+06
 T114_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t114_leaf1 <= 1.000769999999e+06
 T114_R_f3_k1: Overall_Qual_calc - 1e+06 t114_leaf1 >= -999995
 T114_L_f1_k2: const_Lot_Area + 1e+06 t114_leaf2 <= 1.007263999999e+06
 T114_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t114_leaf2 >= -999230
 T114_L_f5_k2: const_Year_Built + 1e+06 t114_leaf2 <= 1.001924999999e+06
 T114_L_f1_k3: const_Lot_Area + 1e+06 t114_leaf3 <= 1.007263999999e+06
 T114_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t114_leaf3 >= -999230
 T114_R_f5_k3: const_Year_Built - 1e+06 t114_leaf3 >= -998075
 T114_R_f1_k4: const_Lot_Area - 1e+06 t114_leaf4 >= -992736
 T114_L_f36_k4: x_Wood_Deck_SF + 1e+06 t114_leaf4 <= 1.000198999999e+06
 T114_L_f40_k4: x_Screen_Porch + 1e+06 t114_leaf4 <= 1.000114999999e+06
 T114_R_f1_k5: const_Lot_Area - 1e+06 t114_leaf5 >= -992736
 T114_L_f36_k5: x_Wood_Deck_SF + 1e+06 t114_leaf5 <= 1.000198999999e+06
 T114_R_f40_k5: x_Screen_Porch - 1e+06 t114_leaf5 >= -999885
 T114_R_f1_k6: const_Lot_Area - 1e+06 t114_leaf6 >= -992736
 T114_R_f36_k6: x_Wood_Deck_SF - 1e+06 t114_leaf6 >= -999801
 T114_L_f0_k6: const_Lot_Frontage + 1e+06 t114_leaf6 <= 1.000149999999e+06
 T114_R_f1_k7: const_Lot_Area - 1e+06 t114_leaf7 >= -992736
 T114_R_f36_k7: x_Wood_Deck_SF - 1e+06 t114_leaf7 >= -999801
 T114_R_f0_k7: const_Lot_Frontage - 1e+06 t114_leaf7 >= -999850
 T115_onehot: t115_leaf0 + t115_leaf1 + t115_leaf2 + t115_leaf3
   + t115_leaf4 + t115_leaf5 = 1
 T115_L_f4_k0: const_Overall_Cond + 1e+06 t115_leaf0 <= 1.000004999999e+06
 T115_L_f40_k0: x_Screen_Porch + 1e+06 t115_leaf0 <= 1.000087999999e+06
 T115_L_f4_k1: const_Overall_Cond + 1e+06 t115_leaf1 <= 1.000004999999e+06
 T115_R_f40_k1: x_Screen_Porch - 1e+06 t115_leaf1 >= -999912
 T115_R_f4_k2: const_Overall_Cond - 1e+06 t115_leaf2 >= -999995
 T115_L_f36_k2: x_Wood_Deck_SF + 1e+06 t115_leaf2 <= 1.000151999999e+06
 T115_L_f40_k2: x_Screen_Porch + 1e+06 t115_leaf2 <= 1.000099999999e+06
 T115_R_f4_k3: const_Overall_Cond - 1e+06 t115_leaf3 >= -999995
 T115_L_f36_k3: x_Wood_Deck_SF + 1e+06 t115_leaf3 <= 1.000151999999e+06
 T115_R_f40_k3: x_Screen_Porch - 1e+06 t115_leaf3 >= -999900
 T115_R_f4_k4: const_Overall_Cond - 1e+06 t115_leaf4 >= -999995
 T115_R_f36_k4: x_Wood_Deck_SF - 1e+06 t115_leaf4 >= -999848
 T115_L_f77_k4: const_Land_Contour_Bnk + 1e+06 t115_leaf4
   <= 1.000000999999e+06
 T115_R_f4_k5: const_Overall_Cond - 1e+06 t115_leaf5 >= -999995
 T115_R_f36_k5: x_Wood_Deck_SF - 1e+06 t115_leaf5 >= -999848
 T115_R_f77_k5: const_Land_Contour_Bnk - 1e+06 t115_leaf5 >= -999999
 T116_onehot: t116_leaf0 + t116_leaf1 + t116_leaf2 + t116_leaf3
   + t116_leaf4 + t116_leaf5 + t116_leaf6 = 1
 T116_L_f44_k0: const_Yr_Sold + 1e+06 t116_leaf0 <= 1.002007999999e+06
 T116_L_f20_k0: x_Gr_Liv_Area + 1e+06 t116_leaf0 <= 1.003085999999e+06
 T116_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t116_leaf0 <= 1.000880999999e+06
 T116_L_f44_k1: const_Yr_Sold + 1e+06 t116_leaf1 <= 1.002007999999e+06
 T116_L_f20_k1: x_Gr_Liv_Area + 1e+06 t116_leaf1 <= 1.003085999999e+06
 T116_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t116_leaf1 >= -999119
 T116_L_f44_k2: const_Yr_Sold + 1e+06 t116_leaf2 <= 1.002007999999e+06
 T116_R_f20_k2: x_Gr_Liv_Area - 1e+06 t116_leaf2 >= -996914
 T116_R_f44_k3: const_Yr_Sold - 1e+06 t116_leaf3 >= -997992
 T116_L_f293_k3: const_Sale_Condition_Abnorml + 1e+06 t116_leaf3
   <= 1.000000999999e+06
 T116_L_f4_k3: const_Overall_Cond + 1e+06 t116_leaf3 <= 1.000006999999e+06
 T116_R_f44_k4: const_Yr_Sold - 1e+06 t116_leaf4 >= -997992
 T116_L_f293_k4: const_Sale_Condition_Abnorml + 1e+06 t116_leaf4
   <= 1.000000999999e+06
 T116_R_f4_k4: const_Overall_Cond - 1e+06 t116_leaf4 >= -999993
 T116_R_f44_k5: const_Yr_Sold - 1e+06 t116_leaf5 >= -997992
 T116_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t116_leaf5
   >= -999999
 T116_L_f20_k5: x_Gr_Liv_Area + 1e+06 t116_leaf5 <= 1.000975999999e+06
 T116_R_f44_k6: const_Yr_Sold - 1e+06 t116_leaf6 >= -997992
 T116_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t116_leaf6
   >= -999999
 T116_R_f20_k6: x_Gr_Liv_Area - 1e+06 t116_leaf6 >= -999024
 T117_onehot: t117_leaf0 + t117_leaf1 + t117_leaf2 + t117_leaf3
   + t117_leaf4 = 1
 T117_L_f20_k0: x_Gr_Liv_Area + 1e+06 t117_leaf0 <= 1.003085999999e+06
 T117_L_f4_k0: const_Overall_Cond + 1e+06 t117_leaf0 <= 1.000004999999e+06
 T117_L_f16_k0: x_Heating_QC + 1e+06 t117_leaf0 <= 1.000002999999e+06
 T117_L_f20_k1: x_Gr_Liv_Area + 1e+06 t117_leaf1 <= 1.003085999999e+06
 T117_L_f4_k1: const_Overall_Cond + 1e+06 t117_leaf1 <= 1.000004999999e+06
 T117_R_f16_k1: x_Heating_QC - 1e+06 t117_leaf1 >= -999997
 T117_L_f20_k2: x_Gr_Liv_Area + 1e+06 t117_leaf2 <= 1.003085999999e+06
 T117_R_f4_k2: const_Overall_Cond - 1e+06 t117_leaf2 >= -999995
 T117_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t117_leaf2
   <= 1.000000999999e+06
 T117_L_f20_k3: x_Gr_Liv_Area + 1e+06 t117_leaf3 <= 1.003085999999e+06
 T117_R_f4_k3: const_Overall_Cond - 1e+06 t117_leaf3 >= -999995
 T117_R_f21_k3: const_Bsmt_Full_Bath - 1e+06 t117_leaf3 >= -999999
 T117_R_f20_k4: x_Gr_Liv_Area - 1e+06 t117_leaf4 >= -996914
 T118_onehot: t118_leaf0 + t118_leaf1 + t118_leaf2 + t118_leaf3
   + t118_leaf4 + t118_leaf5 = 1
 T118_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t118_leaf0 <= 1.000000999999e+06
 T118_L_f44_k0: const_Yr_Sold + 1e+06 t118_leaf0 <= 1.002006999999e+06
 T118_L_f36_k0: x_Wood_Deck_SF + 1e+06 t118_leaf0 <= 1.000198999999e+06
 T118_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t118_leaf1 <= 1.000000999999e+06
 T118_L_f44_k1: const_Yr_Sold + 1e+06 t118_leaf1 <= 1.002006999999e+06
 T118_R_f36_k1: x_Wood_Deck_SF - 1e+06 t118_leaf1 >= -999801
 T118_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t118_leaf2 <= 1.000000999999e+06
 T118_R_f44_k2: const_Yr_Sold - 1e+06 t118_leaf2 >= -997993
 T118_L_f1_k2: const_Lot_Area + 1e+06 t118_leaf2 <= 1.005975999999e+06
 T118_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t118_leaf3 <= 1.000000999999e+06
 T118_R_f44_k3: const_Yr_Sold - 1e+06 t118_leaf3 >= -997993
 T118_R_f1_k3: const_Lot_Area - 1e+06 t118_leaf3 >= -994024
 T118_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t118_leaf4 >= -999999
 T118_L_f25_k4: x_Bedroom_AbvGr + 1e+06 t118_leaf4 <= 1.000002999999e+06
 T118_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t118_leaf5 >= -999999
 T118_R_f25_k5: x_Bedroom_AbvGr - 1e+06 t118_leaf5 >= -999997
 T119_onehot: t119_leaf0 + t119_leaf1 + t119_leaf2 + t119_leaf3 = 1
 T119_L_f4_k0: const_Overall_Cond + 1e+06 t119_leaf0 <= 1.000002999999e+06
 T119_R_f4_k1: const_Overall_Cond - 1e+06 t119_leaf1 >= -999997
 T119_L_f18_k1: const_2nd_Flr_SF + 1e+06 t119_leaf1 <= 1.001426999999e+06
 T119_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t119_leaf1 <= 1.000074999999e+06
 T119_R_f4_k2: const_Overall_Cond - 1e+06 t119_leaf2 >= -999997
 T119_L_f18_k2: const_2nd_Flr_SF + 1e+06 t119_leaf2 <= 1.001426999999e+06
 T119_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t119_leaf2 >= -999925
 T119_R_f4_k3: const_Overall_Cond - 1e+06 t119_leaf3 >= -999997
 T119_R_f18_k3: const_2nd_Flr_SF - 1e+06 t119_leaf3 >= -998573
 T120_onehot: t120_leaf0 + t120_leaf1 + t120_leaf2 + t120_leaf3
   + t120_leaf4 + t120_leaf5 = 1
 T120_L_f3_k0: Overall_Qual_calc + 1e+06 t120_leaf0 <= 1.000002999999e+06
 T120_L_f6_k0: const_Year_Remod_Add + 1e+06 t120_leaf0
   <= 1.001950999999e+06
 T120_L_f3_k1: Overall_Qual_calc + 1e+06 t120_leaf1 <= 1.000002999999e+06
 T120_R_f6_k1: const_Year_Remod_Add - 1e+06 t120_leaf1 >= -998049
 T120_R_f3_k2: Overall_Qual_calc - 1e+06 t120_leaf2 >= -999997
 T120_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t120_leaf2
   <= 1.000000999999e+06
 T120_L_f33_k2: x_Garage_Area + 1e+06 t120_leaf2 <= 1.001153999999e+06
 T120_R_f3_k3: Overall_Qual_calc - 1e+06 t120_leaf3 >= -999997
 T120_L_f95_k3: const_Neighborhood_Crawfor + 1e+06 t120_leaf3
   <= 1.000000999999e+06
 T120_R_f33_k3: x_Garage_Area - 1e+06 t120_leaf3 >= -998846
 T120_R_f3_k4: Overall_Qual_calc - 1e+06 t120_leaf4 >= -999997
 T120_R_f95_k4: const_Neighborhood_Crawfor - 1e+06 t120_leaf4 >= -999999
 T120_L_f1_k4: const_Lot_Area + 1e+06 t120_leaf4 <= 1.004425999999e+06
 T120_R_f3_k5: Overall_Qual_calc - 1e+06 t120_leaf5 >= -999997
 T120_R_f95_k5: const_Neighborhood_Crawfor - 1e+06 t120_leaf5 >= -999999
 T120_R_f1_k5: const_Lot_Area - 1e+06 t120_leaf5 >= -995574
 T121_onehot: t121_leaf0 + t121_leaf1 + t121_leaf2 + t121_leaf3
   + t121_leaf4 + t121_leaf5 + t121_leaf6 = 1
 T121_L_f20_k0: x_Gr_Liv_Area + 1e+06 t121_leaf0 <= 1.001154999999e+06
 T121_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t121_leaf0
   <= 1.000000999999e+06
 T121_L_f259_k0: x_paved_drive_is_Y + 1e+06 t121_leaf0
   <= 1.000000999999e+06
 T121_L_f20_k1: x_Gr_Liv_Area + 1e+06 t121_leaf1 <= 1.001154999999e+06
 T121_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t121_leaf1
   <= 1.000000999999e+06
 T121_R_f259_k1: x_paved_drive_is_Y - 1e+06 t121_leaf1 >= -999999
 T121_L_f20_k2: x_Gr_Liv_Area + 1e+06 t121_leaf2 <= 1.001154999999e+06
 T121_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t121_leaf2 >= -999999
 T121_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t121_leaf2
   <= 1.001930999999e+06
 T121_L_f20_k3: x_Gr_Liv_Area + 1e+06 t121_leaf3 <= 1.001154999999e+06
 T121_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t121_leaf3 >= -999999
 T121_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t121_leaf3 >= -998069
 T121_R_f20_k4: x_Gr_Liv_Area - 1e+06 t121_leaf4 >= -998845
 T121_L_f7_k4: x_Mas_Vnr_Area + 1e+06 t121_leaf4 <= 1.000503999999e+06
 T121_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t121_leaf4 <= 1.000218999999e+06
 T121_R_f20_k5: x_Gr_Liv_Area - 1e+06 t121_leaf5 >= -998845
 T121_L_f7_k5: x_Mas_Vnr_Area + 1e+06 t121_leaf5 <= 1.000503999999e+06
 T121_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t121_leaf5 >= -999781
 T121_R_f20_k6: x_Gr_Liv_Area - 1e+06 t121_leaf6 >= -998845
 T121_R_f7_k6: x_Mas_Vnr_Area - 1e+06 t121_leaf6 >= -999496
 T122_onehot: t122_leaf0 + t122_leaf1 + t122_leaf2 + t122_leaf3
   + t122_leaf4 = 1
 T122_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t122_leaf0
   <= 1.000000999999e+06
 T122_L_f1_k0: const_Lot_Area + 1e+06 t122_leaf0 <= 1.005975999999e+06
 T122_L_f63_k0: const_MS_Zoning_FV + 1e+06 t122_leaf0
   <= 1.000000999999e+06
 T122_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t122_leaf1
   <= 1.000000999999e+06
 T122_L_f1_k1: const_Lot_Area + 1e+06 t122_leaf1 <= 1.005975999999e+06
 T122_R_f63_k1: const_MS_Zoning_FV - 1e+06 t122_leaf1 >= -999999
 T122_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t122_leaf2
   <= 1.000000999999e+06
 T122_R_f1_k2: const_Lot_Area - 1e+06 t122_leaf2 >= -994024
 T122_L_f207_k2: const_Bsmt_Exposure_Gd + 1e+06 t122_leaf2
   <= 1.000000999999e+06
 T122_L_f95_k3: const_Neighborhood_Crawfor + 1e+06 t122_leaf3
   <= 1.000000999999e+06
 T122_R_f1_k3: const_Lot_Area - 1e+06 t122_leaf3 >= -994024
 T122_R_f207_k3: const_Bsmt_Exposure_Gd - 1e+06 t122_leaf3 >= -999999
 T122_R_f95_k4: const_Neighborhood_Crawfor - 1e+06 t122_leaf4 >= -999999
 T123_onehot: t123_leaf0 + t123_leaf1 + t123_leaf2 + t123_leaf3
   + t123_leaf4 + t123_leaf5 + t123_leaf6 + t123_leaf7 = 1
 T123_L_f44_k0: const_Yr_Sold + 1e+06 t123_leaf0 <= 1.002007999999e+06
 T123_L_f17_k0: x_1st_Flr_SF + 1e+06 t123_leaf0 <= 1.001810999999e+06
 T123_L_f4_k0: const_Overall_Cond + 1e+06 t123_leaf0 <= 1.000005999999e+06
 T123_L_f44_k1: const_Yr_Sold + 1e+06 t123_leaf1 <= 1.002007999999e+06
 T123_L_f17_k1: x_1st_Flr_SF + 1e+06 t123_leaf1 <= 1.001810999999e+06
 T123_R_f4_k1: const_Overall_Cond - 1e+06 t123_leaf1 >= -999994
 T123_L_f44_k2: const_Yr_Sold + 1e+06 t123_leaf2 <= 1.002007999999e+06
 T123_R_f17_k2: x_1st_Flr_SF - 1e+06 t123_leaf2 >= -998189
 T123_L_f76_k2: const_Lot_Shape_Reg + 1e+06 t123_leaf2
   <= 1.000000999999e+06
 T123_L_f44_k3: const_Yr_Sold + 1e+06 t123_leaf3 <= 1.002007999999e+06
 T123_R_f17_k3: x_1st_Flr_SF - 1e+06 t123_leaf3 >= -998189
 T123_R_f76_k3: const_Lot_Shape_Reg - 1e+06 t123_leaf3 >= -999999
 T123_R_f44_k4: const_Yr_Sold - 1e+06 t123_leaf4 >= -997992
 T123_L_f6_k4: const_Year_Remod_Add + 1e+06 t123_leaf4
   <= 1.001953999999e+06
 T123_L_f87_k4: const_Land_Slope_Mod + 1e+06 t123_leaf4
   <= 1.000000999999e+06
 T123_R_f44_k5: const_Yr_Sold - 1e+06 t123_leaf5 >= -997992
 T123_L_f6_k5: const_Year_Remod_Add + 1e+06 t123_leaf5
   <= 1.001953999999e+06
 T123_R_f87_k5: const_Land_Slope_Mod - 1e+06 t123_leaf5 >= -999999
 T123_R_f44_k6: const_Yr_Sold - 1e+06 t123_leaf6 >= -997992
 T123_R_f6_k6: const_Year_Remod_Add - 1e+06 t123_leaf6 >= -998046
 T123_L_f164_k6: x_ex1_is_BrkFace + 1e+06 t123_leaf6 <= 1.000000999999e+06
 T123_R_f44_k7: const_Yr_Sold - 1e+06 t123_leaf7 >= -997992
 T123_R_f6_k7: const_Year_Remod_Add - 1e+06 t123_leaf7 >= -998046
 T123_R_f164_k7: x_ex1_is_BrkFace - 1e+06 t123_leaf7 >= -999999
 T124_onehot: t124_leaf0 + t124_leaf1 + t124_leaf2 + t124_leaf3
   + t124_leaf4 + t124_leaf5 = 1
 T124_L_f36_k0: x_Wood_Deck_SF + 1e+06 t124_leaf0 <= 1.000215999999e+06
 T124_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t124_leaf0
   <= 1.000000999999e+06
 T124_L_f6_k0: const_Year_Remod_Add + 1e+06 t124_leaf0
   <= 1.001992999999e+06
 T124_L_f36_k1: x_Wood_Deck_SF + 1e+06 t124_leaf1 <= 1.000215999999e+06
 T124_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t124_leaf1
   <= 1.000000999999e+06
 T124_R_f6_k1: const_Year_Remod_Add - 1e+06 t124_leaf1 >= -998007
 T124_L_f36_k2: x_Wood_Deck_SF + 1e+06 t124_leaf2 <= 1.000215999999e+06
 T124_R_f95_k2: const_Neighborhood_Crawfor - 1e+06 t124_leaf2 >= -999999
 T124_L_f33_k2: x_Garage_Area + 1e+06 t124_leaf2 <= 1.000496999999e+06
 T124_L_f36_k3: x_Wood_Deck_SF + 1e+06 t124_leaf3 <= 1.000215999999e+06
 T124_R_f95_k3: const_Neighborhood_Crawfor - 1e+06 t124_leaf3 >= -999999
 T124_R_f33_k3: x_Garage_Area - 1e+06 t124_leaf3 >= -999503
 T124_R_f36_k4: x_Wood_Deck_SF - 1e+06 t124_leaf4 >= -999784
 T124_L_f17_k4: x_1st_Flr_SF + 1e+06 t124_leaf4 <= 1.001810999999e+06
 T124_R_f36_k5: x_Wood_Deck_SF - 1e+06 t124_leaf5 >= -999784
 T124_R_f17_k5: x_1st_Flr_SF - 1e+06 t124_leaf5 >= -998189
 T125_onehot: t125_leaf0 + t125_leaf1 + t125_leaf2 + t125_leaf3
   + t125_leaf4 + t125_leaf5 = 1
 T125_L_f16_k0: x_Heating_QC + 1e+06 t125_leaf0 <= 1.000001999999e+06
 T125_L_f191_k0: x_ex2_is_VinylSd + 1e+06 t125_leaf0 <= 1.000000999999e+06
 T125_L_f16_k1: x_Heating_QC + 1e+06 t125_leaf1 <= 1.000001999999e+06
 T125_R_f191_k1: x_ex2_is_VinylSd - 1e+06 t125_leaf1 >= -999999
 T125_R_f16_k2: x_Heating_QC - 1e+06 t125_leaf2 >= -999998
 T125_L_f29_k2: const_Fireplaces + 1e+06 t125_leaf2 <= 1.000001999999e+06
 T125_L_f289_k2: const_Sale_Type_New + 1e+06 t125_leaf2
   <= 1.000000999999e+06
 T125_R_f16_k3: x_Heating_QC - 1e+06 t125_leaf3 >= -999998
 T125_L_f29_k3: const_Fireplaces + 1e+06 t125_leaf3 <= 1.000001999999e+06
 T125_R_f289_k3: const_Sale_Type_New - 1e+06 t125_leaf3 >= -999999
 T125_R_f16_k4: x_Heating_QC - 1e+06 t125_leaf4 >= -999998
 T125_R_f29_k4: const_Fireplaces - 1e+06 t125_leaf4 >= -999998
 T125_L_f0_k4: const_Lot_Frontage + 1e+06 t125_leaf4 <= 1.000116999999e+06
 T125_R_f16_k5: x_Heating_QC - 1e+06 t125_leaf5 >= -999998
 T125_R_f29_k5: const_Fireplaces - 1e+06 t125_leaf5 >= -999998
 T125_R_f0_k5: const_Lot_Frontage - 1e+06 t125_leaf5 >= -999883
 T126_onehot: t126_leaf0 + t126_leaf1 + t126_leaf2 + t126_leaf3
   + t126_leaf4 = 1
 T126_L_f17_k0: x_1st_Flr_SF + 1e+06 t126_leaf0 <= 1.000575999999e+06
 T126_R_f17_k1: x_1st_Flr_SF - 1e+06 t126_leaf1 >= -999424
 T126_L_f1_k1: const_Lot_Area + 1e+06 t126_leaf1 <= 1.019521999999e+06
 T126_L_f46_k1: const_MS_SubClass_30 + 1e+06 t126_leaf1
   <= 1.000000999999e+06
 T126_R_f17_k2: x_1st_Flr_SF - 1e+06 t126_leaf2 >= -999424
 T126_L_f1_k2: const_Lot_Area + 1e+06 t126_leaf2 <= 1.019521999999e+06
 T126_R_f46_k2: const_MS_SubClass_30 - 1e+06 t126_leaf2 >= -999999
 T126_R_f17_k3: x_1st_Flr_SF - 1e+06 t126_leaf3 >= -999424
 T126_R_f1_k3: const_Lot_Area - 1e+06 t126_leaf3 >= -980478
 T126_L_f6_k3: const_Year_Remod_Add + 1e+06 t126_leaf3
   <= 1.001961999999e+06
 T126_R_f17_k4: x_1st_Flr_SF - 1e+06 t126_leaf4 >= -999424
 T126_R_f1_k4: const_Lot_Area - 1e+06 t126_leaf4 >= -980478
 T126_R_f6_k4: const_Year_Remod_Add - 1e+06 t126_leaf4 >= -998038
 T127_onehot: t127_leaf0 + t127_leaf1 + t127_leaf2 + t127_leaf3
   + t127_leaf4 + t127_leaf5 = 1
 T127_L_f3_k0: Overall_Qual_calc + 1e+06 t127_leaf0 <= 1.000002999999e+06
 T127_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t127_leaf0
   <= 1.000000999999e+06
 T127_L_f3_k1: Overall_Qual_calc + 1e+06 t127_leaf1 <= 1.000002999999e+06
 T127_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t127_leaf1
   >= -999999
 T127_R_f3_k2: Overall_Qual_calc - 1e+06 t127_leaf2 >= -999997
 T127_L_f5_k2: const_Year_Built + 1e+06 t127_leaf2 <= 1.001912999999e+06
 T127_L_f16_k2: x_Heating_QC + 1e+06 t127_leaf2 <= 1.000002999999e+06
 T127_R_f3_k3: Overall_Qual_calc - 1e+06 t127_leaf3 >= -999997
 T127_L_f5_k3: const_Year_Built + 1e+06 t127_leaf3 <= 1.001912999999e+06
 T127_R_f16_k3: x_Heating_QC - 1e+06 t127_leaf3 >= -999997
 T127_R_f3_k4: Overall_Qual_calc - 1e+06 t127_leaf4 >= -999997
 T127_R_f5_k4: const_Year_Built - 1e+06 t127_leaf4 >= -998087
 T127_L_f1_k4: const_Lot_Area + 1e+06 t127_leaf4 <= 1.003674999999e+06
 T127_R_f3_k5: Overall_Qual_calc - 1e+06 t127_leaf5 >= -999997
 T127_R_f5_k5: const_Year_Built - 1e+06 t127_leaf5 >= -998087
 T127_R_f1_k5: const_Lot_Area - 1e+06 t127_leaf5 >= -996325
 T128_onehot: t128_leaf0 + t128_leaf1 + t128_leaf2 + t128_leaf3
   + t128_leaf4 = 1
 T128_L_f3_k0: Overall_Qual_calc + 1e+06 t128_leaf0 <= 1.000002999999e+06
 T128_R_f3_k1: Overall_Qual_calc - 1e+06 t128_leaf1 >= -999997
 T128_L_f4_k1: const_Overall_Cond + 1e+06 t128_leaf1 <= 1.000006999999e+06
 T128_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t128_leaf1
   <= 1.002004999999e+06
 T128_R_f3_k2: Overall_Qual_calc - 1e+06 t128_leaf2 >= -999997
 T128_L_f4_k2: const_Overall_Cond + 1e+06 t128_leaf2 <= 1.000006999999e+06
 T128_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t128_leaf2 >= -997995
 T128_R_f3_k3: Overall_Qual_calc - 1e+06 t128_leaf3 >= -999997
 T128_R_f4_k3: const_Overall_Cond - 1e+06 t128_leaf3 >= -999993
 T128_L_f17_k3: x_1st_Flr_SF + 1e+06 t128_leaf3 <= 1.001319999999e+06
 T128_R_f3_k4: Overall_Qual_calc - 1e+06 t128_leaf4 >= -999997
 T128_R_f4_k4: const_Overall_Cond - 1e+06 t128_leaf4 >= -999993
 T128_R_f17_k4: x_1st_Flr_SF - 1e+06 t128_leaf4 >= -998680
 T129_onehot: t129_leaf0 + t129_leaf1 + t129_leaf2 + t129_leaf3
   + t129_leaf4 + t129_leaf5 = 1
 T129_L_f8_k0: x_Exter_Qual + 1e+06 t129_leaf0 <= 1.000001999999e+06
 T129_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t129_leaf0
   <= 1.000000999999e+06
 T129_L_f8_k1: x_Exter_Qual + 1e+06 t129_leaf1 <= 1.000001999999e+06
 T129_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t129_leaf1 >= -999999
 T129_R_f8_k2: x_Exter_Qual - 1e+06 t129_leaf2 >= -999998
 T129_L_f96_k2: const_Neighborhood_Edwards + 1e+06 t129_leaf2
   <= 1.000000999999e+06
 T129_L_f3_k2: Overall_Qual_calc + 1e+06 t129_leaf2 <= 1.000009999999e+06
 T129_R_f8_k3: x_Exter_Qual - 1e+06 t129_leaf3 >= -999998
 T129_L_f96_k3: const_Neighborhood_Edwards + 1e+06 t129_leaf3
   <= 1.000000999999e+06
 T129_R_f3_k3: Overall_Qual_calc - 1e+06 t129_leaf3 >= -999990
 T129_R_f8_k4: x_Exter_Qual - 1e+06 t129_leaf4 >= -999998
 T129_R_f96_k4: const_Neighborhood_Edwards - 1e+06 t129_leaf4 >= -999999
 T129_L_f0_k4: const_Lot_Frontage + 1e+06 t129_leaf4 <= 1.000065999999e+06
 T129_R_f8_k5: x_Exter_Qual - 1e+06 t129_leaf5 >= -999998
 T129_R_f96_k5: const_Neighborhood_Edwards - 1e+06 t129_leaf5 >= -999999
 T129_R_f0_k5: const_Lot_Frontage - 1e+06 t129_leaf5 >= -999934
 T130_onehot: t130_leaf0 + t130_leaf1 + t130_leaf2 + t130_leaf3
   + t130_leaf4 + t130_leaf5 + t130_leaf6 + t130_leaf7 = 1
 T130_L_f36_k0: x_Wood_Deck_SF + 1e+06 t130_leaf0 <= 1.000194999999e+06
 T130_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t130_leaf0
   <= 1.000000999999e+06
 T130_L_f6_k0: const_Year_Remod_Add + 1e+06 t130_leaf0
   <= 1.001994999999e+06
 T130_L_f36_k1: x_Wood_Deck_SF + 1e+06 t130_leaf1 <= 1.000194999999e+06
 T130_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t130_leaf1
   <= 1.000000999999e+06
 T130_R_f6_k1: const_Year_Remod_Add - 1e+06 t130_leaf1 >= -998005
 T130_L_f36_k2: x_Wood_Deck_SF + 1e+06 t130_leaf2 <= 1.000194999999e+06
 T130_R_f95_k2: const_Neighborhood_Crawfor - 1e+06 t130_leaf2 >= -999999
 T130_L_f0_k2: const_Lot_Frontage + 1e+06 t130_leaf2 <= 1.000053999999e+06
 T130_L_f36_k3: x_Wood_Deck_SF + 1e+06 t130_leaf3 <= 1.000194999999e+06
 T130_R_f95_k3: const_Neighborhood_Crawfor - 1e+06 t130_leaf3 >= -999999
 T130_R_f0_k3: const_Lot_Frontage - 1e+06 t130_leaf3 >= -999946
 T130_R_f36_k4: x_Wood_Deck_SF - 1e+06 t130_leaf4 >= -999805
 T130_L_f7_k4: x_Mas_Vnr_Area + 1e+06 t130_leaf4 <= 1.000299999999e+06
 T130_L_f14_k4: x_Bsmt_Unf_SF + 1e+06 t130_leaf4 <= 1.001239999999e+06
 T130_R_f36_k5: x_Wood_Deck_SF - 1e+06 t130_leaf5 >= -999805
 T130_L_f7_k5: x_Mas_Vnr_Area + 1e+06 t130_leaf5 <= 1.000299999999e+06
 T130_R_f14_k5: x_Bsmt_Unf_SF - 1e+06 t130_leaf5 >= -998760
 T130_R_f36_k6: x_Wood_Deck_SF - 1e+06 t130_leaf6 >= -999805
 T130_R_f7_k6: x_Mas_Vnr_Area - 1e+06 t130_leaf6 >= -999700
 T130_L_f30_k6: x_Fireplace_Qu + 1e+06 t130_leaf6 <= 1.000001999999e+06
 T130_R_f36_k7: x_Wood_Deck_SF - 1e+06 t130_leaf7 >= -999805
 T130_R_f7_k7: x_Mas_Vnr_Area - 1e+06 t130_leaf7 >= -999700
 T130_R_f30_k7: x_Fireplace_Qu - 1e+06 t130_leaf7 >= -999998
 T131_onehot: t131_leaf0 + t131_leaf1 + t131_leaf2 + t131_leaf3
   + t131_leaf4 + t131_leaf5 + t131_leaf6 + t131_leaf7 = 1
 T131_L_f9_k0: x_Exter_Cond + 1e+06 t131_leaf0 <= 1.000001999999e+06
 T131_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t131_leaf0
   <= 1.000000999999e+06
 T131_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t131_leaf0 <= 1.000005999999e+06
 T131_L_f9_k1: x_Exter_Cond + 1e+06 t131_leaf1 <= 1.000001999999e+06
 T131_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t131_leaf1
   <= 1.000000999999e+06
 T131_R_f28_k1: x_TotRms_AbvGrd - 1e+06 t131_leaf1 >= -999994
 T131_L_f9_k2: x_Exter_Cond + 1e+06 t131_leaf2 <= 1.000001999999e+06
 T131_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t131_leaf2 >= -999999
 T131_L_f17_k2: x_1st_Flr_SF + 1e+06 t131_leaf2 <= 1.000950999999e+06
 T131_L_f9_k3: x_Exter_Cond + 1e+06 t131_leaf3 <= 1.000001999999e+06
 T131_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t131_leaf3 >= -999999
 T131_R_f17_k3: x_1st_Flr_SF - 1e+06 t131_leaf3 >= -999049
 T131_R_f9_k4: x_Exter_Cond - 1e+06 t131_leaf4 >= -999998
 T131_L_f40_k4: x_Screen_Porch + 1e+06 t131_leaf4 <= 1.000141999999e+06
 T131_L_f44_k4: const_Yr_Sold + 1e+06 t131_leaf4 <= 1.002006999999e+06
 T131_R_f9_k5: x_Exter_Cond - 1e+06 t131_leaf5 >= -999998
 T131_L_f40_k5: x_Screen_Porch + 1e+06 t131_leaf5 <= 1.000141999999e+06
 T131_R_f44_k5: const_Yr_Sold - 1e+06 t131_leaf5 >= -997993
 T131_R_f9_k6: x_Exter_Cond - 1e+06 t131_leaf6 >= -999998
 T131_R_f40_k6: x_Screen_Porch - 1e+06 t131_leaf6 >= -999858
 T131_L_f1_k6: const_Lot_Area + 1e+06 t131_leaf6 <= 1.013051999999e+06
 T131_R_f9_k7: x_Exter_Cond - 1e+06 t131_leaf7 >= -999998
 T131_R_f40_k7: x_Screen_Porch - 1e+06 t131_leaf7 >= -999858
 T131_R_f1_k7: const_Lot_Area - 1e+06 t131_leaf7 >= -986948
 T132_onehot: t132_leaf0 + t132_leaf1 + t132_leaf2 + t132_leaf3
   + t132_leaf4 = 1
 T132_L_f23_k0: x_Full_Bath + 1e+06 t132_leaf0 <= 1.000002999999e+06
 T132_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t132_leaf0 <= 1.001731999999e+06
 T132_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t132_leaf0 <= 1.000747999999e+06
 T132_L_f23_k1: x_Full_Bath + 1e+06 t132_leaf1 <= 1.000002999999e+06
 T132_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t132_leaf1 <= 1.001731999999e+06
 T132_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t132_leaf1 >= -999252
 T132_L_f23_k2: x_Full_Bath + 1e+06 t132_leaf2 <= 1.000002999999e+06
 T132_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t132_leaf2 >= -998268
 T132_R_f23_k3: x_Full_Bath - 1e+06 t132_leaf3 >= -999997
 T132_L_f174_k3: x_ex1_is_VinylSd + 1e+06 t132_leaf3 <= 1.000000999999e+06
 T132_R_f23_k4: x_Full_Bath - 1e+06 t132_leaf4 >= -999997
 T132_R_f174_k4: x_ex1_is_VinylSd - 1e+06 t132_leaf4 >= -999999
 T133_onehot: t133_leaf0 + t133_leaf1 + t133_leaf2 + t133_leaf3
   + t133_leaf4 = 1
 T133_L_f62_k0: const_MS_Zoning_C_(all) + 1e+06 t133_leaf0
   <= 1.000000999999e+06
 T133_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t133_leaf0
   <= 1.000000999999e+06
 T133_L_f24_k0: x_Half_Bath + 1e+06 t133_leaf0 <= 1.000000999999e+06
 T133_L_f62_k1: const_MS_Zoning_C_(all) + 1e+06 t133_leaf1
   <= 1.000000999999e+06
 T133_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t133_leaf1
   <= 1.000000999999e+06
 T133_R_f24_k1: x_Half_Bath - 1e+06 t133_leaf1 >= -999999
 T133_L_f62_k2: const_MS_Zoning_C_(all) + 1e+06 t133_leaf2
   <= 1.000000999999e+06
 T133_R_f245_k2: const_FunctioNo_aplical_Typ - 1e+06 t133_leaf2 >= -999999
 T133_L_f29_k2: const_Fireplaces + 1e+06 t133_leaf2 <= 1.000001999999e+06
 T133_L_f62_k3: const_MS_Zoning_C_(all) + 1e+06 t133_leaf3
   <= 1.000000999999e+06
 T133_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t133_leaf3 >= -999999
 T133_R_f29_k3: const_Fireplaces - 1e+06 t133_leaf3 >= -999998
 T133_R_f62_k4: const_MS_Zoning_C_(all) - 1e+06 t133_leaf4 >= -999999
 T134_onehot: t134_leaf0 + t134_leaf1 + t134_leaf2 + t134_leaf3
   + t134_leaf4 = 1
 T134_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t134_leaf0
   <= 1.002135999999e+06
 T134_L_f16_k0: x_Heating_QC + 1e+06 t134_leaf0 <= 1.000002999999e+06
 T134_L_f4_k0: const_Overall_Cond + 1e+06 t134_leaf0 <= 1.000005999999e+06
 T134_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t134_leaf1
   <= 1.002135999999e+06
 T134_L_f16_k1: x_Heating_QC + 1e+06 t134_leaf1 <= 1.000002999999e+06
 T134_R_f4_k1: const_Overall_Cond - 1e+06 t134_leaf1 >= -999994
 T134_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t134_leaf2
   <= 1.002135999999e+06
 T134_R_f16_k2: x_Heating_QC - 1e+06 t134_leaf2 >= -999997
 T134_L_f44_k2: const_Yr_Sold + 1e+06 t134_leaf2 <= 1.002008999999e+06
 T134_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t134_leaf3
   <= 1.002135999999e+06
 T134_R_f16_k3: x_Heating_QC - 1e+06 t134_leaf3 >= -999997
 T134_R_f44_k3: const_Yr_Sold - 1e+06 t134_leaf3 >= -997991
 T134_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t134_leaf4 >= -997864
 T135_onehot: t135_leaf0 + t135_leaf1 + t135_leaf2 + t135_leaf3
   + t135_leaf4 + t135_leaf5 + t135_leaf6 = 1
 T135_L_f44_k0: const_Yr_Sold + 1e+06 t135_leaf0 <= 1.002006999999e+06
 T135_L_f0_k0: const_Lot_Frontage + 1e+06 t135_leaf0 <= 1.000030999999e+06
 T135_L_f44_k1: const_Yr_Sold + 1e+06 t135_leaf1 <= 1.002006999999e+06
 T135_R_f0_k1: const_Lot_Frontage - 1e+06 t135_leaf1 >= -999969
 T135_L_f117_k1: const_Condition_1_Artery + 1e+06 t135_leaf1
   <= 1.000000999999e+06
 T135_L_f44_k2: const_Yr_Sold + 1e+06 t135_leaf2 <= 1.002006999999e+06
 T135_R_f0_k2: const_Lot_Frontage - 1e+06 t135_leaf2 >= -999969
 T135_R_f117_k2: const_Condition_1_Artery - 1e+06 t135_leaf2 >= -999999
 T135_R_f44_k3: const_Yr_Sold - 1e+06 t135_leaf3 >= -997993
 T135_L_f16_k3: x_Heating_QC + 1e+06 t135_leaf3 <= 1.000002999999e+06
 T135_L_f9_k3: x_Exter_Cond + 1e+06 t135_leaf3 <= 1.000001999999e+06
 T135_R_f44_k4: const_Yr_Sold - 1e+06 t135_leaf4 >= -997993
 T135_L_f16_k4: x_Heating_QC + 1e+06 t135_leaf4 <= 1.000002999999e+06
 T135_R_f9_k4: x_Exter_Cond - 1e+06 t135_leaf4 >= -999998
 T135_R_f44_k5: const_Yr_Sold - 1e+06 t135_leaf5 >= -997993
 T135_R_f16_k5: x_Heating_QC - 1e+06 t135_leaf5 >= -999997
 T135_L_f239_k5: const_FunctioNo_aplical_Maj2 + 1e+06 t135_leaf5
   <= 1.000000999999e+06
 T135_R_f44_k6: const_Yr_Sold - 1e+06 t135_leaf6 >= -997993
 T135_R_f16_k6: x_Heating_QC - 1e+06 t135_leaf6 >= -999997
 T135_R_f239_k6: const_FunctioNo_aplical_Maj2 - 1e+06 t135_leaf6
   >= -999999
 T136_onehot: t136_leaf0 + t136_leaf1 + t136_leaf2 + t136_leaf3
   + t136_leaf4 = 1
 T136_L_f23_k0: x_Full_Bath + 1e+06 t136_leaf0 <= 1.000002999999e+06
 T136_L_f33_k0: x_Garage_Area + 1e+06 t136_leaf0 <= 1.001153999999e+06
 T136_L_f44_k0: const_Yr_Sold + 1e+06 t136_leaf0 <= 1.002007999999e+06
 T136_L_f23_k1: x_Full_Bath + 1e+06 t136_leaf1 <= 1.000002999999e+06
 T136_L_f33_k1: x_Garage_Area + 1e+06 t136_leaf1 <= 1.001153999999e+06
 T136_R_f44_k1: const_Yr_Sold - 1e+06 t136_leaf1 >= -997992
 T136_L_f23_k2: x_Full_Bath + 1e+06 t136_leaf2 <= 1.000002999999e+06
 T136_R_f33_k2: x_Garage_Area - 1e+06 t136_leaf2 >= -998846
 T136_R_f23_k3: x_Full_Bath - 1e+06 t136_leaf3 >= -999997
 T136_L_f255_k3: const_Garage_Finish_RFn + 1e+06 t136_leaf3
   <= 1.000000999999e+06
 T136_R_f23_k4: x_Full_Bath - 1e+06 t136_leaf4 >= -999997
 T136_R_f255_k4: const_Garage_Finish_RFn - 1e+06 t136_leaf4 >= -999999
 T137_onehot: t137_leaf0 + t137_leaf1 + t137_leaf2 + t137_leaf3
   + t137_leaf4 + t137_leaf5 + t137_leaf6 + t137_leaf7 = 1
 T137_L_f1_k0: const_Lot_Area + 1e+06 t137_leaf0 <= 1.005999999999e+06
 T137_L_f3_k0: Overall_Qual_calc + 1e+06 t137_leaf0 <= 1.000007999999e+06
 T137_L_f33_k0: x_Garage_Area + 1e+06 t137_leaf0 <= 1.000183999999e+06
 T137_L_f1_k1: const_Lot_Area + 1e+06 t137_leaf1 <= 1.005999999999e+06
 T137_L_f3_k1: Overall_Qual_calc + 1e+06 t137_leaf1 <= 1.000007999999e+06
 T137_R_f33_k1: x_Garage_Area - 1e+06 t137_leaf1 >= -999816
 T137_L_f1_k2: const_Lot_Area + 1e+06 t137_leaf2 <= 1.005999999999e+06
 T137_R_f3_k2: Overall_Qual_calc - 1e+06 t137_leaf2 >= -999992
 T137_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t137_leaf2 <= 1.000323999999e+06
 T137_L_f1_k3: const_Lot_Area + 1e+06 t137_leaf3 <= 1.005999999999e+06
 T137_R_f3_k3: Overall_Qual_calc - 1e+06 t137_leaf3 >= -999992
 T137_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t137_leaf3 >= -999676
 T137_R_f1_k4: const_Lot_Area - 1e+06 t137_leaf4 >= -994000
 T137_L_f18_k4: const_2nd_Flr_SF + 1e+06 t137_leaf4 <= 1.001426999999e+06
 T137_L_f44_k4: const_Yr_Sold + 1e+06 t137_leaf4 <= 1.002006999999e+06
 T137_R_f1_k5: const_Lot_Area - 1e+06 t137_leaf5 >= -994000
 T137_L_f18_k5: const_2nd_Flr_SF + 1e+06 t137_leaf5 <= 1.001426999999e+06
 T137_R_f44_k5: const_Yr_Sold - 1e+06 t137_leaf5 >= -997993
 T137_R_f1_k6: const_Lot_Area - 1e+06 t137_leaf6 >= -994000
 T137_R_f18_k6: const_2nd_Flr_SF - 1e+06 t137_leaf6 >= -998573
 T137_L_f1_k6: const_Lot_Area + 1e+06 t137_leaf6 <= 1.009799999999e+06
 T137_R_f1_k7: const_Lot_Area - 1e+06 t137_leaf7 >= -994000
 T137_R_f18_k7: const_2nd_Flr_SF - 1e+06 t137_leaf7 >= -998573
 T137_R_f1_k7: const_Lot_Area - 1e+06 t137_leaf7 >= -990200
 T138_onehot: t138_leaf0 + t138_leaf1 + t138_leaf2 + t138_leaf3
   + t138_leaf4 + t138_leaf5 + t138_leaf6 = 1
 T138_L_f9_k0: x_Exter_Cond + 1e+06 t138_leaf0 <= 1.000001999999e+06
 T138_L_f3_k0: Overall_Qual_calc + 1e+06 t138_leaf0 <= 1.000002999999e+06
 T138_L_f9_k1: x_Exter_Cond + 1e+06 t138_leaf1 <= 1.000001999999e+06
 T138_R_f3_k1: Overall_Qual_calc - 1e+06 t138_leaf1 >= -999997
 T138_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t138_leaf1 <= 1.000719999999e+06
 T138_L_f9_k2: x_Exter_Cond + 1e+06 t138_leaf2 <= 1.000001999999e+06
 T138_R_f3_k2: Overall_Qual_calc - 1e+06 t138_leaf2 >= -999997
 T138_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t138_leaf2 >= -999280
 T138_R_f9_k3: x_Exter_Cond - 1e+06 t138_leaf3 >= -999998
 T138_L_f1_k3: const_Lot_Area + 1e+06 t138_leaf3 <= 1.005975999999e+06
 T138_L_f33_k3: x_Garage_Area + 1e+06 t138_leaf3 <= 1.000383999999e+06
 T138_R_f9_k4: x_Exter_Cond - 1e+06 t138_leaf4 >= -999998
 T138_L_f1_k4: const_Lot_Area + 1e+06 t138_leaf4 <= 1.005975999999e+06
 T138_R_f33_k4: x_Garage_Area - 1e+06 t138_leaf4 >= -999616
 T138_R_f9_k5: x_Exter_Cond - 1e+06 t138_leaf5 >= -999998
 T138_R_f1_k5: const_Lot_Area - 1e+06 t138_leaf5 >= -994024
 T138_L_f29_k5: const_Fireplaces + 1e+06 t138_leaf5 <= 1.000002999999e+06
 T138_R_f9_k6: x_Exter_Cond - 1e+06 t138_leaf6 >= -999998
 T138_R_f1_k6: const_Lot_Area - 1e+06 t138_leaf6 >= -994024
 T138_R_f29_k6: const_Fireplaces - 1e+06 t138_leaf6 >= -999997
 T139_onehot: t139_leaf0 + t139_leaf1 + t139_leaf2 + t139_leaf3
   + t139_leaf4 + t139_leaf5 + t139_leaf6 + t139_leaf7 = 1
 T139_L_f4_k0: const_Overall_Cond + 1e+06 t139_leaf0 <= 1.000006999999e+06
 T139_L_f109_k0: const_Neighborhood_OldTown + 1e+06 t139_leaf0
   <= 1.000000999999e+06
 T139_L_f37_k0: x_Open_Porch_SF + 1e+06 t139_leaf0 <= 1.000044999999e+06
 T139_L_f4_k1: const_Overall_Cond + 1e+06 t139_leaf1 <= 1.000006999999e+06
 T139_L_f109_k1: const_Neighborhood_OldTown + 1e+06 t139_leaf1
   <= 1.000000999999e+06
 T139_R_f37_k1: x_Open_Porch_SF - 1e+06 t139_leaf1 >= -999955
 T139_L_f4_k2: const_Overall_Cond + 1e+06 t139_leaf2 <= 1.000006999999e+06
 T139_R_f109_k2: const_Neighborhood_OldTown - 1e+06 t139_leaf2 >= -999999
 T139_L_f200_k2: const_Foundation_BrkTil + 1e+06 t139_leaf2
   <= 1.000000999999e+06
 T139_L_f4_k3: const_Overall_Cond + 1e+06 t139_leaf3 <= 1.000006999999e+06
 T139_R_f109_k3: const_Neighborhood_OldTown - 1e+06 t139_leaf3 >= -999999
 T139_R_f200_k3: const_Foundation_BrkTil - 1e+06 t139_leaf3 >= -999999
 T139_R_f4_k4: const_Overall_Cond - 1e+06 t139_leaf4 >= -999993
 T139_L_f119_k4: const_Condition_1_Norm + 1e+06 t139_leaf4
   <= 1.000000999999e+06
 T139_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t139_leaf4
   <= 1.000903999999e+06
 T139_R_f4_k5: const_Overall_Cond - 1e+06 t139_leaf5 >= -999993
 T139_L_f119_k5: const_Condition_1_Norm + 1e+06 t139_leaf5
   <= 1.000000999999e+06
 T139_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t139_leaf5 >= -999096
 T139_R_f4_k6: const_Overall_Cond - 1e+06 t139_leaf6 >= -999993
 T139_R_f119_k6: const_Condition_1_Norm - 1e+06 t139_leaf6 >= -999999
 T139_L_f245_k6: const_FunctioNo_aplical_Typ + 1e+06 t139_leaf6
   <= 1.000000999999e+06
 T139_R_f4_k7: const_Overall_Cond - 1e+06 t139_leaf7 >= -999993
 T139_R_f119_k7: const_Condition_1_Norm - 1e+06 t139_leaf7 >= -999999
 T139_R_f245_k7: const_FunctioNo_aplical_Typ - 1e+06 t139_leaf7 >= -999999
 T140_onehot: t140_leaf0 + t140_leaf1 + t140_leaf2 + t140_leaf3
   + t140_leaf4 + t140_leaf5 + t140_leaf6 + t140_leaf7 = 1
 T140_L_f11_k0: x_Bsmt_Cond + 1e+06 t140_leaf0 <= 1.000001999999e+06
 T140_L_f9_k0: x_Exter_Cond + 1e+06 t140_leaf0 <= 1.000001999999e+06
 T140_L_f0_k0: const_Lot_Frontage + 1e+06 t140_leaf0 <= 1.000064999999e+06
 T140_L_f11_k1: x_Bsmt_Cond + 1e+06 t140_leaf1 <= 1.000001999999e+06
 T140_L_f9_k1: x_Exter_Cond + 1e+06 t140_leaf1 <= 1.000001999999e+06
 T140_R_f0_k1: const_Lot_Frontage - 1e+06 t140_leaf1 >= -999935
 T140_L_f11_k2: x_Bsmt_Cond + 1e+06 t140_leaf2 <= 1.000001999999e+06
 T140_R_f9_k2: x_Exter_Cond - 1e+06 t140_leaf2 >= -999998
 T140_L_f17_k2: x_1st_Flr_SF + 1e+06 t140_leaf2 <= 1.001123999999e+06
 T140_L_f11_k3: x_Bsmt_Cond + 1e+06 t140_leaf3 <= 1.000001999999e+06
 T140_R_f9_k3: x_Exter_Cond - 1e+06 t140_leaf3 >= -999998
 T140_R_f17_k3: x_1st_Flr_SF - 1e+06 t140_leaf3 >= -998876
 T140_R_f11_k4: x_Bsmt_Cond - 1e+06 t140_leaf4 >= -999998
 T140_L_f4_k4: const_Overall_Cond + 1e+06 t140_leaf4 <= 1.000006999999e+06
 T140_L_f37_k4: x_Open_Porch_SF + 1e+06 t140_leaf4 <= 1.000029999999e+06
 T140_R_f11_k5: x_Bsmt_Cond - 1e+06 t140_leaf5 >= -999998
 T140_L_f4_k5: const_Overall_Cond + 1e+06 t140_leaf5 <= 1.000006999999e+06
 T140_R_f37_k5: x_Open_Porch_SF - 1e+06 t140_leaf5 >= -999970
 T140_R_f11_k6: x_Bsmt_Cond - 1e+06 t140_leaf6 >= -999998
 T140_R_f4_k6: const_Overall_Cond - 1e+06 t140_leaf6 >= -999993
 T140_L_f1_k6: const_Lot_Area + 1e+06 t140_leaf6 <= 1.018385999999e+06
 T140_R_f11_k7: x_Bsmt_Cond - 1e+06 t140_leaf7 >= -999998
 T140_R_f4_k7: const_Overall_Cond - 1e+06 t140_leaf7 >= -999993
 T140_R_f1_k7: const_Lot_Area - 1e+06 t140_leaf7 >= -981614
 T141_onehot: t141_leaf0 + t141_leaf1 + t141_leaf2 + t141_leaf3
   + t141_leaf4 + t141_leaf5 = 1
 T141_L_f5_k0: const_Year_Built + 1e+06 t141_leaf0 <= 1.001924999999e+06
 T141_L_f33_k0: x_Garage_Area + 1e+06 t141_leaf0 <= 1.000624999999e+06
 T141_L_f119_k0: const_Condition_1_Norm + 1e+06 t141_leaf0
   <= 1.000000999999e+06
 T141_L_f5_k1: const_Year_Built + 1e+06 t141_leaf1 <= 1.001924999999e+06
 T141_L_f33_k1: x_Garage_Area + 1e+06 t141_leaf1 <= 1.000624999999e+06
 T141_R_f119_k1: const_Condition_1_Norm - 1e+06 t141_leaf1 >= -999999
 T141_L_f5_k2: const_Year_Built + 1e+06 t141_leaf2 <= 1.001924999999e+06
 T141_R_f33_k2: x_Garage_Area - 1e+06 t141_leaf2 >= -999375
 T141_R_f5_k3: const_Year_Built - 1e+06 t141_leaf3 >= -998075
 T141_L_f18_k3: const_2nd_Flr_SF + 1e+06 t141_leaf3 <= 1.001628999999e+06
 T141_L_f245_k3: const_FunctioNo_aplical_Typ + 1e+06 t141_leaf3
   <= 1.000000999999e+06
 T141_R_f5_k4: const_Year_Built - 1e+06 t141_leaf4 >= -998075
 T141_L_f18_k4: const_2nd_Flr_SF + 1e+06 t141_leaf4 <= 1.001628999999e+06
 T141_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t141_leaf4 >= -999999
 T141_R_f5_k5: const_Year_Built - 1e+06 t141_leaf5 >= -998075
 T141_R_f18_k5: const_2nd_Flr_SF - 1e+06 t141_leaf5 >= -998371
 T142_onehot: t142_leaf0 + t142_leaf1 + t142_leaf2 + t142_leaf3
   + t142_leaf4 = 1
 T142_L_f17_k0: x_1st_Flr_SF + 1e+06 t142_leaf0 <= 1.002514999999e+06
 T142_L_f289_k0: const_Sale_Type_New + 1e+06 t142_leaf0
   <= 1.000000999999e+06
 T142_L_f20_k0: x_Gr_Liv_Area + 1e+06 t142_leaf0 <= 1.002632999999e+06
 T142_L_f17_k1: x_1st_Flr_SF + 1e+06 t142_leaf1 <= 1.002514999999e+06
 T142_L_f289_k1: const_Sale_Type_New + 1e+06 t142_leaf1
   <= 1.000000999999e+06
 T142_R_f20_k1: x_Gr_Liv_Area - 1e+06 t142_leaf1 >= -997367
 T142_L_f17_k2: x_1st_Flr_SF + 1e+06 t142_leaf2 <= 1.002514999999e+06
 T142_R_f289_k2: const_Sale_Type_New - 1e+06 t142_leaf2 >= -999999
 T142_L_f20_k2: x_Gr_Liv_Area + 1e+06 t142_leaf2 <= 1.001359999999e+06
 T142_L_f17_k3: x_1st_Flr_SF + 1e+06 t142_leaf3 <= 1.002514999999e+06
 T142_R_f289_k3: const_Sale_Type_New - 1e+06 t142_leaf3 >= -999999
 T142_R_f20_k3: x_Gr_Liv_Area - 1e+06 t142_leaf3 >= -998640
 T142_R_f17_k4: x_1st_Flr_SF - 1e+06 t142_leaf4 >= -997485
 T143_onehot: t143_leaf0 + t143_leaf1 + t143_leaf2 + t143_leaf3
   + t143_leaf4 + t143_leaf5 + t143_leaf6 = 1
 T143_L_f20_k0: x_Gr_Liv_Area + 1e+06 t143_leaf0 <= 1.000746999999e+06
 T143_L_f20_k0: x_Gr_Liv_Area + 1e+06 t143_leaf0 <= 1.000719999999e+06
 T143_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t143_leaf0 <= 1.000003999999e+06
 T143_L_f20_k1: x_Gr_Liv_Area + 1e+06 t143_leaf1 <= 1.000746999999e+06
 T143_L_f20_k1: x_Gr_Liv_Area + 1e+06 t143_leaf1 <= 1.000719999999e+06
 T143_R_f28_k1: x_TotRms_AbvGrd - 1e+06 t143_leaf1 >= -999996
 T143_L_f20_k2: x_Gr_Liv_Area + 1e+06 t143_leaf2 <= 1.000746999999e+06
 T143_R_f20_k2: x_Gr_Liv_Area - 1e+06 t143_leaf2 >= -999280
 T143_R_f20_k3: x_Gr_Liv_Area - 1e+06 t143_leaf3 >= -999253
 T143_L_f164_k3: x_ex1_is_BrkFace + 1e+06 t143_leaf3 <= 1.000000999999e+06
 T143_L_f4_k3: const_Overall_Cond + 1e+06 t143_leaf3 <= 1.000005999999e+06
 T143_R_f20_k4: x_Gr_Liv_Area - 1e+06 t143_leaf4 >= -999253
 T143_L_f164_k4: x_ex1_is_BrkFace + 1e+06 t143_leaf4 <= 1.000000999999e+06
 T143_R_f4_k4: const_Overall_Cond - 1e+06 t143_leaf4 >= -999994
 T143_R_f20_k5: x_Gr_Liv_Area - 1e+06 t143_leaf5 >= -999253
 T143_R_f164_k5: x_ex1_is_BrkFace - 1e+06 t143_leaf5 >= -999999
 T143_L_f245_k5: const_FunctioNo_aplical_Typ + 1e+06 t143_leaf5
   <= 1.000000999999e+06
 T143_R_f20_k6: x_Gr_Liv_Area - 1e+06 t143_leaf6 >= -999253
 T143_R_f164_k6: x_ex1_is_BrkFace - 1e+06 t143_leaf6 >= -999999
 T143_R_f245_k6: const_FunctioNo_aplical_Typ - 1e+06 t143_leaf6 >= -999999
 T144_onehot: t144_leaf0 + t144_leaf1 + t144_leaf2 + t144_leaf3
   + t144_leaf4 + t144_leaf5 = 1
 T144_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t144_leaf0
   <= 1.000000999999e+06
 T144_L_f44_k0: const_Yr_Sold + 1e+06 t144_leaf0 <= 1.002007999999e+06
 T144_L_f23_k0: x_Full_Bath + 1e+06 t144_leaf0 <= 1.000002999999e+06
 T144_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t144_leaf1
   <= 1.000000999999e+06
 T144_L_f44_k1: const_Yr_Sold + 1e+06 t144_leaf1 <= 1.002007999999e+06
 T144_R_f23_k1: x_Full_Bath - 1e+06 t144_leaf1 >= -999997
 T144_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t144_leaf2
   <= 1.000000999999e+06
 T144_R_f44_k2: const_Yr_Sold - 1e+06 t144_leaf2 >= -997992
 T144_L_f297_k2: const_Sale_Condition_Normal + 1e+06 t144_leaf2
   <= 1.000000999999e+06
 T144_L_f95_k3: const_Neighborhood_Crawfor + 1e+06 t144_leaf3
   <= 1.000000999999e+06
 T144_R_f44_k3: const_Yr_Sold - 1e+06 t144_leaf3 >= -997992
 T144_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t144_leaf3 >= -999999
 T144_R_f95_k4: const_Neighborhood_Crawfor - 1e+06 t144_leaf4 >= -999999
 T144_L_f1_k4: const_Lot_Area + 1e+06 t144_leaf4 <= 1.004834999999e+06
 T144_R_f95_k5: const_Neighborhood_Crawfor - 1e+06 t144_leaf5 >= -999999
 T144_R_f1_k5: const_Lot_Area - 1e+06 t144_leaf5 >= -995165
 T145_onehot: t145_leaf0 + t145_leaf1 + t145_leaf2 + t145_leaf3
   + t145_leaf4 + t145_leaf5 = 1
 T145_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t145_leaf0 <= 1.000000999999e+06
 T145_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t145_leaf0
   <= 1.000000999999e+06
 T145_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t145_leaf0 <= 1.000000999999e+06
 T145_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t145_leaf1 <= 1.000000999999e+06
 T145_L_f21_k1: const_Bsmt_Full_Bath + 1e+06 t145_leaf1
   <= 1.000000999999e+06
 T145_R_f164_k1: x_ex1_is_BrkFace - 1e+06 t145_leaf1 >= -999999
 T145_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t145_leaf2 <= 1.000000999999e+06
 T145_R_f21_k2: const_Bsmt_Full_Bath - 1e+06 t145_leaf2 >= -999999
 T145_L_f190_k2: x_ex2_is_Stucco + 1e+06 t145_leaf2 <= 1.000000999999e+06
 T145_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t145_leaf3 <= 1.000000999999e+06
 T145_R_f21_k3: const_Bsmt_Full_Bath - 1e+06 t145_leaf3 >= -999999
 T145_R_f190_k3: x_ex2_is_Stucco - 1e+06 t145_leaf3 >= -999999
 T145_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t145_leaf4 >= -999999
 T145_L_f1_k4: const_Lot_Area + 1e+06 t145_leaf4 <= 1.008929999999e+06
 T145_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t145_leaf5 >= -999999
 T145_R_f1_k5: const_Lot_Area - 1e+06 t145_leaf5 >= -991070
 T146_onehot: t146_leaf0 + t146_leaf1 + t146_leaf2 + t146_leaf3
   + t146_leaf4 + t146_leaf5 + t146_leaf6 + t146_leaf7 = 1
 T146_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t146_leaf0 <= 1.000424999999e+06
 T146_L_f16_k0: x_Heating_QC + 1e+06 t146_leaf0 <= 1.000002999999e+06
 T146_L_f191_k0: x_ex2_is_VinylSd + 1e+06 t146_leaf0 <= 1.000000999999e+06
 T146_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t146_leaf1 <= 1.000424999999e+06
 T146_L_f16_k1: x_Heating_QC + 1e+06 t146_leaf1 <= 1.000002999999e+06
 T146_R_f191_k1: x_ex2_is_VinylSd - 1e+06 t146_leaf1 >= -999999
 T146_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t146_leaf2 <= 1.000424999999e+06
 T146_R_f16_k2: x_Heating_QC - 1e+06 t146_leaf2 >= -999997
 T146_L_f1_k2: const_Lot_Area + 1e+06 t146_leaf2 <= 1.016778999999e+06
 T146_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t146_leaf3 <= 1.000424999999e+06
 T146_R_f16_k3: x_Heating_QC - 1e+06 t146_leaf3 >= -999997
 T146_R_f1_k3: const_Lot_Area - 1e+06 t146_leaf3 >= -983221
 T146_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t146_leaf4 >= -999575
 T146_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t146_leaf4
   <= 1.001497999999e+06
 T146_L_f11_k4: x_Bsmt_Cond + 1e+06 t146_leaf4 <= 1.000002999999e+06
 T146_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t146_leaf5 >= -999575
 T146_L_f15_k5: const_Total_Bsmt_SF + 1e+06 t146_leaf5
   <= 1.001497999999e+06
 T146_R_f11_k5: x_Bsmt_Cond - 1e+06 t146_leaf5 >= -999997
 T146_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t146_leaf6 >= -999575
 T146_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t146_leaf6 >= -998502
 T146_L_f208_k6: const_Bsmt_Exposure_Mn + 1e+06 t146_leaf6
   <= 1.000000999999e+06
 T146_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t146_leaf7 >= -999575
 T146_R_f15_k7: const_Total_Bsmt_SF - 1e+06 t146_leaf7 >= -998502
 T146_R_f208_k7: const_Bsmt_Exposure_Mn - 1e+06 t146_leaf7 >= -999999
 T147_onehot: t147_leaf0 + t147_leaf1 + t147_leaf2 + t147_leaf3
   + t147_leaf4 + t147_leaf5 = 1
 T147_L_f17_k0: x_1st_Flr_SF + 1e+06 t147_leaf0 <= 1.000615999999e+06
 T147_L_f6_k0: const_Year_Remod_Add + 1e+06 t147_leaf0
   <= 1.001955999999e+06
 T147_L_f17_k1: x_1st_Flr_SF + 1e+06 t147_leaf1 <= 1.000615999999e+06
 T147_R_f6_k1: const_Year_Remod_Add - 1e+06 t147_leaf1 >= -998044
 T147_R_f17_k2: x_1st_Flr_SF - 1e+06 t147_leaf2 >= -999384
 T147_L_f118_k2: const_Condition_1_Feedr + 1e+06 t147_leaf2
   <= 1.000000999999e+06
 T147_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t147_leaf2 <= 1.000699999999e+06
 T147_R_f17_k3: x_1st_Flr_SF - 1e+06 t147_leaf3 >= -999384
 T147_L_f118_k3: const_Condition_1_Feedr + 1e+06 t147_leaf3
   <= 1.000000999999e+06
 T147_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t147_leaf3 >= -999300
 T147_R_f17_k4: x_1st_Flr_SF - 1e+06 t147_leaf4 >= -999384
 T147_R_f118_k4: const_Condition_1_Feedr - 1e+06 t147_leaf4 >= -999999
 T147_L_f271_k4: const_Mo_Sold_1 + 1e+06 t147_leaf4 <= 1.000000999999e+06
 T147_R_f17_k5: x_1st_Flr_SF - 1e+06 t147_leaf5 >= -999384
 T147_R_f118_k5: const_Condition_1_Feedr - 1e+06 t147_leaf5 >= -999999
 T147_R_f271_k5: const_Mo_Sold_1 - 1e+06 t147_leaf5 >= -999999
 T148_onehot: t148_leaf0 + t148_leaf1 + t148_leaf2 + t148_leaf3
   + t148_leaf4 + t148_leaf5 = 1
 T148_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t148_leaf0
   <= 1.000000999999e+06
 T148_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t148_leaf0 <= 1.000000999999e+06
 T148_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t148_leaf0
   <= 1.000584999999e+06
 T148_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t148_leaf1
   <= 1.000000999999e+06
 T148_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t148_leaf1 <= 1.000000999999e+06
 T148_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t148_leaf1 >= -999415
 T148_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t148_leaf2
   <= 1.000000999999e+06
 T148_R_f164_k2: x_ex1_is_BrkFace - 1e+06 t148_leaf2 >= -999999
 T148_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t148_leaf3
   >= -999999
 T148_L_f3_k3: Overall_Qual_calc + 1e+06 t148_leaf3 <= 1.000003999999e+06
 T148_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t148_leaf4
   >= -999999
 T148_R_f3_k4: Overall_Qual_calc - 1e+06 t148_leaf4 >= -999996
 T148_L_f29_k4: const_Fireplaces + 1e+06 t148_leaf4 <= 1.000000999999e+06
 T148_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t148_leaf5
   >= -999999
 T148_R_f3_k5: Overall_Qual_calc - 1e+06 t148_leaf5 >= -999996
 T148_R_f29_k5: const_Fireplaces - 1e+06 t148_leaf5 >= -999999
 T149_onehot: t149_leaf0 + t149_leaf1 + t149_leaf2 + t149_leaf3
   + t149_leaf4 + t149_leaf5 + t149_leaf6 = 1
 T149_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t149_leaf0
   <= 1.000000999999e+06
 T149_L_f192_k0: x_ex2_is_Wd_Sdng + 1e+06 t149_leaf0 <= 1.000000999999e+06
 T149_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t149_leaf0 <= 1.000187999999e+06
 T149_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t149_leaf1
   <= 1.000000999999e+06
 T149_L_f192_k1: x_ex2_is_Wd_Sdng + 1e+06 t149_leaf1 <= 1.000000999999e+06
 T149_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t149_leaf1 >= -999812
 T149_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t149_leaf2
   <= 1.000000999999e+06
 T149_R_f192_k2: x_ex2_is_Wd_Sdng - 1e+06 t149_leaf2 >= -999999
 T149_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t149_leaf3 >= -999999
 T149_L_f164_k3: x_ex1_is_BrkFace + 1e+06 t149_leaf3 <= 1.000000999999e+06
 T149_L_f4_k3: const_Overall_Cond + 1e+06 t149_leaf3 <= 1.000005999999e+06
 T149_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t149_leaf4 >= -999999
 T149_L_f164_k4: x_ex1_is_BrkFace + 1e+06 t149_leaf4 <= 1.000000999999e+06
 T149_R_f4_k4: const_Overall_Cond - 1e+06 t149_leaf4 >= -999994
 T149_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t149_leaf5 >= -999999
 T149_R_f164_k5: x_ex1_is_BrkFace - 1e+06 t149_leaf5 >= -999999
 T149_L_f106_k5: const_Neighborhood_No_aplicames + 1e+06 t149_leaf5
   <= 1.000000999999e+06
 T149_R_f245_k6: const_FunctioNo_aplical_Typ - 1e+06 t149_leaf6 >= -999999
 T149_R_f164_k6: x_ex1_is_BrkFace - 1e+06 t149_leaf6 >= -999999
 T149_R_f106_k6: const_Neighborhood_No_aplicames - 1e+06 t149_leaf6
   >= -999999
 T150_onehot: t150_leaf0 + t150_leaf1 + t150_leaf2 + t150_leaf3
   + t150_leaf4 + t150_leaf5 + t150_leaf6 = 1
 T150_L_f9_k0: x_Exter_Cond + 1e+06 t150_leaf0 <= 1.000001999999e+06
 T150_L_f33_k0: x_Garage_Area + 1e+06 t150_leaf0 <= 1.000296999999e+06
 T150_L_f9_k1: x_Exter_Cond + 1e+06 t150_leaf1 <= 1.000001999999e+06
 T150_R_f33_k1: x_Garage_Area - 1e+06 t150_leaf1 >= -999703
 T150_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t150_leaf1
   <= 1.000816999999e+06
 T150_L_f9_k2: x_Exter_Cond + 1e+06 t150_leaf2 <= 1.000001999999e+06
 T150_R_f33_k2: x_Garage_Area - 1e+06 t150_leaf2 >= -999703
 T150_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t150_leaf2 >= -999183
 T150_R_f9_k3: x_Exter_Cond - 1e+06 t150_leaf3 >= -999998
 T150_L_f1_k3: const_Lot_Area + 1e+06 t150_leaf3 <= 1.005975999999e+06
 T150_L_f20_k3: x_Gr_Liv_Area + 1e+06 t150_leaf3 <= 1.000995999999e+06
 T150_R_f9_k4: x_Exter_Cond - 1e+06 t150_leaf4 >= -999998
 T150_L_f1_k4: const_Lot_Area + 1e+06 t150_leaf4 <= 1.005975999999e+06
 T150_R_f20_k4: x_Gr_Liv_Area - 1e+06 t150_leaf4 >= -999004
 T150_R_f9_k5: x_Exter_Cond - 1e+06 t150_leaf5 >= -999998
 T150_R_f1_k5: const_Lot_Area - 1e+06 t150_leaf5 >= -994024
 T150_L_f95_k5: const_Neighborhood_Crawfor + 1e+06 t150_leaf5
   <= 1.000000999999e+06
 T150_R_f9_k6: x_Exter_Cond - 1e+06 t150_leaf6 >= -999998
 T150_R_f1_k6: const_Lot_Area - 1e+06 t150_leaf6 >= -994024
 T150_R_f95_k6: const_Neighborhood_Crawfor - 1e+06 t150_leaf6 >= -999999
 T151_onehot: t151_leaf0 + t151_leaf1 + t151_leaf2 + t151_leaf3
   + t151_leaf4 = 1
 T151_L_f33_k0: x_Garage_Area + 1e+06 t151_leaf0 <= 1.000803999999e+06
 T151_L_f207_k0: const_Bsmt_Exposure_Gd + 1e+06 t151_leaf0
   <= 1.000000999999e+06
 T151_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t151_leaf0
   <= 1.000000999999e+06
 T151_L_f33_k1: x_Garage_Area + 1e+06 t151_leaf1 <= 1.000803999999e+06
 T151_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t151_leaf1
   <= 1.000000999999e+06
 T151_R_f245_k1: const_FunctioNo_aplical_Typ - 1e+06 t151_leaf1 >= -999999
 T151_L_f33_k2: x_Garage_Area + 1e+06 t151_leaf2 <= 1.000803999999e+06
 T151_R_f207_k2: const_Bsmt_Exposure_Gd - 1e+06 t151_leaf2 >= -999999
 T151_L_f1_k2: const_Lot_Area + 1e+06 t151_leaf2 <= 1.016559999999e+06
 T151_L_f33_k3: x_Garage_Area + 1e+06 t151_leaf3 <= 1.000803999999e+06
 T151_R_f207_k3: const_Bsmt_Exposure_Gd - 1e+06 t151_leaf3 >= -999999
 T151_R_f1_k3: const_Lot_Area - 1e+06 t151_leaf3 >= -983440
 T151_R_f33_k4: x_Garage_Area - 1e+06 t151_leaf4 >= -999196
 T152_onehot: t152_leaf0 + t152_leaf1 + t152_leaf2 + t152_leaf3
   + t152_leaf4 + t152_leaf5 = 1
 T152_L_f40_k0: x_Screen_Porch + 1e+06 t152_leaf0 <= 1.000149999999e+06
 T152_L_f9_k0: x_Exter_Cond + 1e+06 t152_leaf0 <= 1.000001999999e+06
 T152_L_f1_k0: const_Lot_Area + 1e+06 t152_leaf0 <= 1.009170999999e+06
 T152_L_f40_k1: x_Screen_Porch + 1e+06 t152_leaf1 <= 1.000149999999e+06
 T152_L_f9_k1: x_Exter_Cond + 1e+06 t152_leaf1 <= 1.000001999999e+06
 T152_R_f1_k1: const_Lot_Area - 1e+06 t152_leaf1 >= -990829
 T152_L_f40_k2: x_Screen_Porch + 1e+06 t152_leaf2 <= 1.000149999999e+06
 T152_R_f9_k2: x_Exter_Cond - 1e+06 t152_leaf2 >= -999998
 T152_L_f86_k2: const_Land_Slope_Gtl + 1e+06 t152_leaf2
   <= 1.000000999999e+06
 T152_L_f40_k3: x_Screen_Porch + 1e+06 t152_leaf3 <= 1.000149999999e+06
 T152_R_f9_k3: x_Exter_Cond - 1e+06 t152_leaf3 >= -999998
 T152_R_f86_k3: const_Land_Slope_Gtl - 1e+06 t152_leaf3 >= -999999
 T152_R_f40_k4: x_Screen_Porch - 1e+06 t152_leaf4 >= -999850
 T152_L_f18_k4: const_2nd_Flr_SF + 1e+06 t152_leaf4 <= 1.000977999999e+06
 T152_R_f40_k5: x_Screen_Porch - 1e+06 t152_leaf5 >= -999850
 T152_R_f18_k5: const_2nd_Flr_SF - 1e+06 t152_leaf5 >= -999022
 T153_onehot: t153_leaf0 + t153_leaf1 + t153_leaf2 + t153_leaf3
   + t153_leaf4 + t153_leaf5 + t153_leaf6 = 1
 T153_L_f1_k0: const_Lot_Area + 1e+06 t153_leaf0 <= 1.005999999999e+06
 T153_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t153_leaf0
   <= 1.001894999999e+06
 T153_L_f1_k1: const_Lot_Area + 1e+06 t153_leaf1 <= 1.005999999999e+06
 T153_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t153_leaf1 >= -998105
 T153_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t153_leaf1
   <= 1.000000999999e+06
 T153_L_f1_k2: const_Lot_Area + 1e+06 t153_leaf2 <= 1.005999999999e+06
 T153_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t153_leaf2 >= -998105
 T153_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t153_leaf2 >= -999999
 T153_R_f1_k3: const_Lot_Area - 1e+06 t153_leaf3 >= -994000
 T153_L_f40_k3: x_Screen_Porch + 1e+06 t153_leaf3 <= 1.000141999999e+06
 T153_L_f5_k3: const_Year_Built + 1e+06 t153_leaf3 <= 1.001912999999e+06
 T153_R_f1_k4: const_Lot_Area - 1e+06 t153_leaf4 >= -994000
 T153_L_f40_k4: x_Screen_Porch + 1e+06 t153_leaf4 <= 1.000141999999e+06
 T153_R_f5_k4: const_Year_Built - 1e+06 t153_leaf4 >= -998087
 T153_R_f1_k5: const_Lot_Area - 1e+06 t153_leaf5 >= -994000
 T153_R_f40_k5: x_Screen_Porch - 1e+06 t153_leaf5 >= -999858
 T153_L_f20_k5: x_Gr_Liv_Area + 1e+06 t153_leaf5 <= 1.001929999999e+06
 T153_R_f1_k6: const_Lot_Area - 1e+06 t153_leaf6 >= -994000
 T153_R_f40_k6: x_Screen_Porch - 1e+06 t153_leaf6 >= -999858
 T153_R_f20_k6: x_Gr_Liv_Area - 1e+06 t153_leaf6 >= -998070
 T154_onehot: t154_leaf0 + t154_leaf1 + t154_leaf2 + t154_leaf3
   + t154_leaf4 + t154_leaf5 + t154_leaf6 + t154_leaf7 = 1
 T154_L_f16_k0: x_Heating_QC + 1e+06 t154_leaf0 <= 1.000002999999e+06
 T154_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t154_leaf0
   <= 1.000000999999e+06
 T154_L_f20_k0: x_Gr_Liv_Area + 1e+06 t154_leaf0 <= 1.000975999999e+06
 T154_L_f16_k1: x_Heating_QC + 1e+06 t154_leaf1 <= 1.000002999999e+06
 T154_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t154_leaf1
   <= 1.000000999999e+06
 T154_R_f20_k1: x_Gr_Liv_Area - 1e+06 t154_leaf1 >= -999024
 T154_L_f16_k2: x_Heating_QC + 1e+06 t154_leaf2 <= 1.000002999999e+06
 T154_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t154_leaf2 >= -999999
 T154_L_f1_k2: const_Lot_Area + 1e+06 t154_leaf2 <= 1.010895999999e+06
 T154_L_f16_k3: x_Heating_QC + 1e+06 t154_leaf3 <= 1.000002999999e+06
 T154_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t154_leaf3 >= -999999
 T154_R_f1_k3: const_Lot_Area - 1e+06 t154_leaf3 >= -989104
 T154_R_f16_k4: x_Heating_QC - 1e+06 t154_leaf4 >= -999997
 T154_L_f245_k4: const_FunctioNo_aplical_Typ + 1e+06 t154_leaf4
   <= 1.000000999999e+06
 T154_L_f139_k4: const_House_Style_1.5Fin + 1e+06 t154_leaf4
   <= 1.000000999999e+06
 T154_R_f16_k5: x_Heating_QC - 1e+06 t154_leaf5 >= -999997
 T154_L_f245_k5: const_FunctioNo_aplical_Typ + 1e+06 t154_leaf5
   <= 1.000000999999e+06
 T154_R_f139_k5: const_House_Style_1.5Fin - 1e+06 t154_leaf5 >= -999999
 T154_R_f16_k6: x_Heating_QC - 1e+06 t154_leaf6 >= -999997
 T154_R_f245_k6: const_FunctioNo_aplical_Typ - 1e+06 t154_leaf6 >= -999999
 T154_L_f8_k6: x_Exter_Qual + 1e+06 t154_leaf6 <= 1.000002999999e+06
 T154_R_f16_k7: x_Heating_QC - 1e+06 t154_leaf7 >= -999997
 T154_R_f245_k7: const_FunctioNo_aplical_Typ - 1e+06 t154_leaf7 >= -999999
 T154_R_f8_k7: x_Exter_Qual - 1e+06 t154_leaf7 >= -999997
 T155_onehot: t155_leaf0 + t155_leaf1 + t155_leaf2 + t155_leaf3
   + t155_leaf4 = 1
 T155_L_f33_k0: x_Garage_Area + 1e+06 t155_leaf0 <= 1.001153999999e+06
 T155_L_f1_k0: const_Lot_Area + 1e+06 t155_leaf0 <= 1.017870999999e+06
 T155_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t155_leaf0
   <= 1.002007999999e+06
 T155_L_f33_k1: x_Garage_Area + 1e+06 t155_leaf1 <= 1.001153999999e+06
 T155_L_f1_k1: const_Lot_Area + 1e+06 t155_leaf1 <= 1.017870999999e+06
 T155_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t155_leaf1 >= -997992
 T155_L_f33_k2: x_Garage_Area + 1e+06 t155_leaf2 <= 1.001153999999e+06
 T155_R_f1_k2: const_Lot_Area - 1e+06 t155_leaf2 >= -982129
 T155_L_f103_k2: const_Neighborhood_Mitchel + 1e+06 t155_leaf2
   <= 1.000000999999e+06
 T155_L_f33_k3: x_Garage_Area + 1e+06 t155_leaf3 <= 1.001153999999e+06
 T155_R_f1_k3: const_Lot_Area - 1e+06 t155_leaf3 >= -982129
 T155_R_f103_k3: const_Neighborhood_Mitchel - 1e+06 t155_leaf3 >= -999999
 T155_R_f33_k4: x_Garage_Area - 1e+06 t155_leaf4 >= -998846
 T156_onehot: t156_leaf0 + t156_leaf1 + t156_leaf2 + t156_leaf3
   + t156_leaf4 = 1
 T156_L_f40_k0: x_Screen_Porch + 1e+06 t156_leaf0 <= 1.000039999999e+06
 T156_L_f5_k0: const_Year_Built + 1e+06 t156_leaf0 <= 1.002006999999e+06
 T156_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t156_leaf0 <= 1.000055999999e+06
 T156_L_f40_k1: x_Screen_Porch + 1e+06 t156_leaf1 <= 1.000039999999e+06
 T156_L_f5_k1: const_Year_Built + 1e+06 t156_leaf1 <= 1.002006999999e+06
 T156_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t156_leaf1 >= -999944
 T156_L_f40_k2: x_Screen_Porch + 1e+06 t156_leaf2 <= 1.000039999999e+06
 T156_R_f5_k2: const_Year_Built - 1e+06 t156_leaf2 >= -997993
 T156_R_f40_k3: x_Screen_Porch - 1e+06 t156_leaf3 >= -999960
 T156_L_f20_k3: x_Gr_Liv_Area + 1e+06 t156_leaf3 <= 1.001976999999e+06
 T156_R_f40_k4: x_Screen_Porch - 1e+06 t156_leaf4 >= -999960
 T156_R_f20_k4: x_Gr_Liv_Area - 1e+06 t156_leaf4 >= -998023
 T157_onehot: t157_leaf0 + t157_leaf1 + t157_leaf2 + t157_leaf3
   + t157_leaf4 + t157_leaf5 + t157_leaf6 = 1
 T157_L_f1_k0: const_Lot_Area + 1e+06 t157_leaf0 <= 1.007949999999e+06
 T157_L_f20_k0: x_Gr_Liv_Area + 1e+06 t157_leaf0 <= 1.001127999999e+06
 T157_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t157_leaf0 <= 1.000532999999e+06
 T157_L_f1_k1: const_Lot_Area + 1e+06 t157_leaf1 <= 1.007949999999e+06
 T157_L_f20_k1: x_Gr_Liv_Area + 1e+06 t157_leaf1 <= 1.001127999999e+06
 T157_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t157_leaf1 >= -999467
 T157_L_f1_k2: const_Lot_Area + 1e+06 t157_leaf2 <= 1.007949999999e+06
 T157_R_f20_k2: x_Gr_Liv_Area - 1e+06 t157_leaf2 >= -998872
 T157_L_f238_k2: const_FunctioNo_aplical_Maj1 + 1e+06 t157_leaf2
   <= 1.000000999999e+06
 T157_L_f1_k3: const_Lot_Area + 1e+06 t157_leaf3 <= 1.007949999999e+06
 T157_R_f20_k3: x_Gr_Liv_Area - 1e+06 t157_leaf3 >= -998872
 T157_R_f238_k3: const_FunctioNo_aplical_Maj1 - 1e+06 t157_leaf3
   >= -999999
 T157_R_f1_k4: const_Lot_Area - 1e+06 t157_leaf4 >= -992050
 T157_L_f18_k4: const_2nd_Flr_SF + 1e+06 t157_leaf4 <= 1.001426999999e+06
 T157_L_f119_k4: const_Condition_1_Norm + 1e+06 t157_leaf4
   <= 1.000000999999e+06
 T157_R_f1_k5: const_Lot_Area - 1e+06 t157_leaf5 >= -992050
 T157_L_f18_k5: const_2nd_Flr_SF + 1e+06 t157_leaf5 <= 1.001426999999e+06
 T157_R_f119_k5: const_Condition_1_Norm - 1e+06 t157_leaf5 >= -999999
 T157_R_f1_k6: const_Lot_Area - 1e+06 t157_leaf6 >= -992050
 T157_R_f18_k6: const_2nd_Flr_SF - 1e+06 t157_leaf6 >= -998573
 T158_onehot: t158_leaf0 + t158_leaf1 + t158_leaf2 + t158_leaf3
   + t158_leaf4 + t158_leaf5 + t158_leaf6 = 1
 T158_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t158_leaf0
   <= 1.000000999999e+06
 T158_L_f40_k0: x_Screen_Porch + 1e+06 t158_leaf0 <= 1.000120999999e+06
 T158_L_f1_k0: const_Lot_Area + 1e+06 t158_leaf0 <= 1.013517999999e+06
 T158_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t158_leaf1
   <= 1.000000999999e+06
 T158_L_f40_k1: x_Screen_Porch + 1e+06 t158_leaf1 <= 1.000120999999e+06
 T158_R_f1_k1: const_Lot_Area - 1e+06 t158_leaf1 >= -986482
 T158_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t158_leaf2
   <= 1.000000999999e+06
 T158_R_f40_k2: x_Screen_Porch - 1e+06 t158_leaf2 >= -999879
 T158_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t158_leaf3 >= -999999
 T158_L_f44_k3: const_Yr_Sold + 1e+06 t158_leaf3 <= 1.002006999999e+06
 T158_L_f1_k3: const_Lot_Area + 1e+06 t158_leaf3 <= 1.008126999999e+06
 T158_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t158_leaf4 >= -999999
 T158_L_f44_k4: const_Yr_Sold + 1e+06 t158_leaf4 <= 1.002006999999e+06
 T158_R_f1_k4: const_Lot_Area - 1e+06 t158_leaf4 >= -991873
 T158_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t158_leaf5 >= -999999
 T158_R_f44_k5: const_Yr_Sold - 1e+06 t158_leaf5 >= -997993
 T158_L_f20_k5: x_Gr_Liv_Area + 1e+06 t158_leaf5 <= 1.001154999999e+06
 T158_R_f245_k6: const_FunctioNo_aplical_Typ - 1e+06 t158_leaf6 >= -999999
 T158_R_f44_k6: const_Yr_Sold - 1e+06 t158_leaf6 >= -997993
 T158_R_f20_k6: x_Gr_Liv_Area - 1e+06 t158_leaf6 >= -998845
 T159_onehot: t159_leaf0 + t159_leaf1 + t159_leaf2 + t159_leaf3
   + t159_leaf4 = 1
 T159_L_f3_k0: Overall_Qual_calc + 1e+06 t159_leaf0 <= 1.000008999999e+06
 T159_L_f20_k0: x_Gr_Liv_Area + 1e+06 t159_leaf0 <= 1.000746999999e+06
 T159_L_f16_k0: x_Heating_QC + 1e+06 t159_leaf0 <= 1.000002999999e+06
 T159_L_f3_k1: Overall_Qual_calc + 1e+06 t159_leaf1 <= 1.000008999999e+06
 T159_L_f20_k1: x_Gr_Liv_Area + 1e+06 t159_leaf1 <= 1.000746999999e+06
 T159_R_f16_k1: x_Heating_QC - 1e+06 t159_leaf1 >= -999997
 T159_L_f3_k2: Overall_Qual_calc + 1e+06 t159_leaf2 <= 1.000008999999e+06
 T159_R_f20_k2: x_Gr_Liv_Area - 1e+06 t159_leaf2 >= -999253
 T159_L_f4_k2: const_Overall_Cond + 1e+06 t159_leaf2 <= 1.000005999999e+06
 T159_L_f3_k3: Overall_Qual_calc + 1e+06 t159_leaf3 <= 1.000008999999e+06
 T159_R_f20_k3: x_Gr_Liv_Area - 1e+06 t159_leaf3 >= -999253
 T159_R_f4_k3: const_Overall_Cond - 1e+06 t159_leaf3 >= -999994
 T159_R_f3_k4: Overall_Qual_calc - 1e+06 t159_leaf4 >= -999991
 T160_onehot: t160_leaf0 + t160_leaf1 + t160_leaf2 + t160_leaf3
   + t160_leaf4 + t160_leaf5 + t160_leaf6 = 1
 T160_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t160_leaf0
   <= 1.000000999999e+06
 T160_L_f44_k0: const_Yr_Sold + 1e+06 t160_leaf0 <= 1.002006999999e+06
 T160_L_f1_k0: const_Lot_Area + 1e+06 t160_leaf0 <= 1.016778999999e+06
 T160_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t160_leaf1
   <= 1.000000999999e+06
 T160_L_f44_k1: const_Yr_Sold + 1e+06 t160_leaf1 <= 1.002006999999e+06
 T160_R_f1_k1: const_Lot_Area - 1e+06 t160_leaf1 >= -983221
 T160_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t160_leaf2
   <= 1.000000999999e+06
 T160_R_f44_k2: const_Yr_Sold - 1e+06 t160_leaf2 >= -997993
 T160_L_f40_k2: x_Screen_Porch + 1e+06 t160_leaf2 <= 1.000141999999e+06
 T160_L_f293_k3: const_Sale_Condition_Abnorml + 1e+06 t160_leaf3
   <= 1.000000999999e+06
 T160_R_f44_k3: const_Yr_Sold - 1e+06 t160_leaf3 >= -997993
 T160_R_f40_k3: x_Screen_Porch - 1e+06 t160_leaf3 >= -999858
 T160_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t160_leaf4
   >= -999999
 T160_L_f3_k4: Overall_Qual_calc + 1e+06 t160_leaf4 <= 1.000003999999e+06
 T160_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t160_leaf5
   >= -999999
 T160_R_f3_k5: Overall_Qual_calc - 1e+06 t160_leaf5 >= -999996
 T160_L_f16_k5: x_Heating_QC + 1e+06 t160_leaf5 <= 1.000002999999e+06
 T160_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t160_leaf6
   >= -999999
 T160_R_f3_k6: Overall_Qual_calc - 1e+06 t160_leaf6 >= -999996
 T160_R_f16_k6: x_Heating_QC - 1e+06 t160_leaf6 >= -999997
 T161_onehot: t161_leaf0 + t161_leaf1 + t161_leaf2 + t161_leaf3
   + t161_leaf4 + t161_leaf5 + t161_leaf6 = 1
 T161_L_f77_k0: const_Land_Contour_Bnk + 1e+06 t161_leaf0
   <= 1.000000999999e+06
 T161_L_f86_k0: const_Land_Slope_Gtl + 1e+06 t161_leaf0
   <= 1.000000999999e+06
 T161_L_f1_k0: const_Lot_Area + 1e+06 t161_leaf0 <= 1.010043999999e+06
 T161_L_f77_k1: const_Land_Contour_Bnk + 1e+06 t161_leaf1
   <= 1.000000999999e+06
 T161_L_f86_k1: const_Land_Slope_Gtl + 1e+06 t161_leaf1
   <= 1.000000999999e+06
 T161_R_f1_k1: const_Lot_Area - 1e+06 t161_leaf1 >= -989956
 T161_L_f77_k2: const_Land_Contour_Bnk + 1e+06 t161_leaf2
   <= 1.000000999999e+06
 T161_R_f86_k2: const_Land_Slope_Gtl - 1e+06 t161_leaf2 >= -999999
 T161_L_f17_k2: x_1st_Flr_SF + 1e+06 t161_leaf2 <= 1.002120999999e+06
 T161_L_f77_k3: const_Land_Contour_Bnk + 1e+06 t161_leaf3
   <= 1.000000999999e+06
 T161_R_f86_k3: const_Land_Slope_Gtl - 1e+06 t161_leaf3 >= -999999
 T161_R_f17_k3: x_1st_Flr_SF - 1e+06 t161_leaf3 >= -997879
 T161_R_f77_k4: const_Land_Contour_Bnk - 1e+06 t161_leaf4 >= -999999
 T161_L_f0_k4: const_Lot_Frontage + 1e+06 t161_leaf4 <= 1.000110999999e+06
 T161_L_f0_k4: const_Lot_Frontage + 1e+06 t161_leaf4 <= 1.000068999999e+06
 T161_R_f77_k5: const_Land_Contour_Bnk - 1e+06 t161_leaf5 >= -999999
 T161_L_f0_k5: const_Lot_Frontage + 1e+06 t161_leaf5 <= 1.000110999999e+06
 T161_R_f0_k5: const_Lot_Frontage - 1e+06 t161_leaf5 >= -999931
 T161_R_f77_k6: const_Land_Contour_Bnk - 1e+06 t161_leaf6 >= -999999
 T161_R_f0_k6: const_Lot_Frontage - 1e+06 t161_leaf6 >= -999889
 T162_onehot: t162_leaf0 + t162_leaf1 + t162_leaf2 + t162_leaf3
   + t162_leaf4 + t162_leaf5 + t162_leaf6 = 1
 T162_L_f9_k0: x_Exter_Cond + 1e+06 t162_leaf0 <= 1.000001999999e+06
 T162_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t162_leaf0
   <= 1.000000999999e+06
 T162_L_f33_k0: x_Garage_Area + 1e+06 t162_leaf0 <= 1.000344999999e+06
 T162_L_f9_k1: x_Exter_Cond + 1e+06 t162_leaf1 <= 1.000001999999e+06
 T162_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t162_leaf1
   <= 1.000000999999e+06
 T162_R_f33_k1: x_Garage_Area - 1e+06 t162_leaf1 >= -999655
 T162_L_f9_k2: x_Exter_Cond + 1e+06 t162_leaf2 <= 1.000001999999e+06
 T162_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t162_leaf2
   >= -999999
 T162_R_f9_k3: x_Exter_Cond - 1e+06 t162_leaf3 >= -999998
 T162_L_f164_k3: x_ex1_is_BrkFace + 1e+06 t162_leaf3 <= 1.000000999999e+06
 T162_L_f40_k3: x_Screen_Porch + 1e+06 t162_leaf3 <= 1.000149999999e+06
 T162_R_f9_k4: x_Exter_Cond - 1e+06 t162_leaf4 >= -999998
 T162_L_f164_k4: x_ex1_is_BrkFace + 1e+06 t162_leaf4 <= 1.000000999999e+06
 T162_R_f40_k4: x_Screen_Porch - 1e+06 t162_leaf4 >= -999850
 T162_R_f9_k5: x_Exter_Cond - 1e+06 t162_leaf5 >= -999998
 T162_R_f164_k5: x_ex1_is_BrkFace - 1e+06 t162_leaf5 >= -999999
 T162_L_f20_k5: x_Gr_Liv_Area + 1e+06 t162_leaf5 <= 1.001241999999e+06
 T162_R_f9_k6: x_Exter_Cond - 1e+06 t162_leaf6 >= -999998
 T162_R_f164_k6: x_ex1_is_BrkFace - 1e+06 t162_leaf6 >= -999999
 T162_R_f20_k6: x_Gr_Liv_Area - 1e+06 t162_leaf6 >= -998758
 T163_onehot: t163_leaf0 + t163_leaf1 + t163_leaf2 + t163_leaf3
   + t163_leaf4 + t163_leaf5 = 1
 T163_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t163_leaf0 <= 1.000000999999e+06
 T163_L_f36_k0: x_Wood_Deck_SF + 1e+06 t163_leaf0 <= 1.000198999999e+06
 T163_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t163_leaf0
   <= 1.002007999999e+06
 T163_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t163_leaf1 <= 1.000000999999e+06
 T163_L_f36_k1: x_Wood_Deck_SF + 1e+06 t163_leaf1 <= 1.000198999999e+06
 T163_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t163_leaf1 >= -997992
 T163_L_f164_k2: x_ex1_is_BrkFace + 1e+06 t163_leaf2 <= 1.000000999999e+06
 T163_R_f36_k2: x_Wood_Deck_SF - 1e+06 t163_leaf2 >= -999801
 T163_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t163_leaf2 <= 1.000753999999e+06
 T163_L_f164_k3: x_ex1_is_BrkFace + 1e+06 t163_leaf3 <= 1.000000999999e+06
 T163_R_f36_k3: x_Wood_Deck_SF - 1e+06 t163_leaf3 >= -999801
 T163_R_f7_k3: x_Mas_Vnr_Area - 1e+06 t163_leaf3 >= -999246
 T163_R_f164_k4: x_ex1_is_BrkFace - 1e+06 t163_leaf4 >= -999999
 T163_L_f1_k4: const_Lot_Area + 1e+06 t163_leaf4 <= 1.009604999999e+06
 T163_R_f164_k5: x_ex1_is_BrkFace - 1e+06 t163_leaf5 >= -999999
 T163_R_f1_k5: const_Lot_Area - 1e+06 t163_leaf5 >= -990395
 T164_onehot: t164_leaf0 + t164_leaf1 + t164_leaf2 + t164_leaf3
   + t164_leaf4 = 1
 T164_L_f18_k0: const_2nd_Flr_SF + 1e+06 t164_leaf0 <= 1.001426999999e+06
 T164_L_f11_k0: x_Bsmt_Cond + 1e+06 t164_leaf0 <= 999999.999999
 T164_L_f33_k0: x_Garage_Area + 1e+06 t164_leaf0 <= 1.000456999999e+06
 T164_L_f18_k1: const_2nd_Flr_SF + 1e+06 t164_leaf1 <= 1.001426999999e+06
 T164_L_f11_k1: x_Bsmt_Cond + 1e+06 t164_leaf1 <= 999999.999999
 T164_R_f33_k1: x_Garage_Area - 1e+06 t164_leaf1 >= -999543
 T164_L_f18_k2: const_2nd_Flr_SF + 1e+06 t164_leaf2 <= 1.001426999999e+06
 T164_R_f11_k2: x_Bsmt_Cond - 1e+06 t164_leaf2 >= -1e+06
 T164_L_f289_k2: const_Sale_Type_New + 1e+06 t164_leaf2
   <= 1.000000999999e+06
 T164_L_f18_k3: const_2nd_Flr_SF + 1e+06 t164_leaf3 <= 1.001426999999e+06
 T164_R_f11_k3: x_Bsmt_Cond - 1e+06 t164_leaf3 >= -1e+06
 T164_R_f289_k3: const_Sale_Type_New - 1e+06 t164_leaf3 >= -999999
 T164_R_f18_k4: const_2nd_Flr_SF - 1e+06 t164_leaf4 >= -998573
 T165_onehot: t165_leaf0 + t165_leaf1 + t165_leaf2 + t165_leaf3
   + t165_leaf4 = 1
 T165_L_f17_k0: x_1st_Flr_SF + 1e+06 t165_leaf0 <= 1.002514999999e+06
 T165_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t165_leaf0
   <= 1.001823999999e+06
 T165_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t165_leaf0
   <= 1.000000999999e+06
 T165_L_f17_k1: x_1st_Flr_SF + 1e+06 t165_leaf1 <= 1.002514999999e+06
 T165_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t165_leaf1
   <= 1.001823999999e+06
 T165_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t165_leaf1
   >= -999999
 T165_L_f17_k2: x_1st_Flr_SF + 1e+06 t165_leaf2 <= 1.002514999999e+06
 T165_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t165_leaf2 >= -998176
 T165_L_f108_k2: const_Neighborhood_NridgHt + 1e+06 t165_leaf2
   <= 1.000000999999e+06
 T165_L_f17_k3: x_1st_Flr_SF + 1e+06 t165_leaf3 <= 1.002514999999e+06
 T165_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t165_leaf3 >= -998176
 T165_R_f108_k3: const_Neighborhood_NridgHt - 1e+06 t165_leaf3 >= -999999
 T165_R_f17_k4: x_1st_Flr_SF - 1e+06 t165_leaf4 >= -997485
 T166_onehot: t166_leaf0 + t166_leaf1 + t166_leaf2 + t166_leaf3 = 1
 T166_L_f18_k0: const_2nd_Flr_SF + 1e+06 t166_leaf0 <= 1.001628999999e+06
 T166_L_f93_k0: const_Neighborhood_ClearCr + 1e+06 t166_leaf0
   <= 1.000000999999e+06
 T166_L_f298_k0: const_Sale_Condition_Partial + 1e+06 t166_leaf0
   <= 1.000000999999e+06
 T166_L_f18_k1: const_2nd_Flr_SF + 1e+06 t166_leaf1 <= 1.001628999999e+06
 T166_L_f93_k1: const_Neighborhood_ClearCr + 1e+06 t166_leaf1
   <= 1.000000999999e+06
 T166_R_f298_k1: const_Sale_Condition_Partial - 1e+06 t166_leaf1
   >= -999999
 T166_L_f18_k2: const_2nd_Flr_SF + 1e+06 t166_leaf2 <= 1.001628999999e+06
 T166_R_f93_k2: const_Neighborhood_ClearCr - 1e+06 t166_leaf2 >= -999999
 T166_R_f18_k3: const_2nd_Flr_SF - 1e+06 t166_leaf3 >= -998371
 T167_onehot: t167_leaf0 + t167_leaf1 + t167_leaf2 + t167_leaf3
   + t167_leaf4 = 1
 T167_L_f3_k0: Overall_Qual_calc + 1e+06 t167_leaf0 <= 1.000002999999e+06
 T167_R_f3_k1: Overall_Qual_calc - 1e+06 t167_leaf1 >= -999997
 T167_L_f4_k1: const_Overall_Cond + 1e+06 t167_leaf1 <= 1.000006999999e+06
 T167_L_f5_k1: const_Year_Built + 1e+06 t167_leaf1 <= 1.001910999999e+06
 T167_R_f3_k2: Overall_Qual_calc - 1e+06 t167_leaf2 >= -999997
 T167_L_f4_k2: const_Overall_Cond + 1e+06 t167_leaf2 <= 1.000006999999e+06
 T167_R_f5_k2: const_Year_Built - 1e+06 t167_leaf2 >= -998089
 T167_R_f3_k3: Overall_Qual_calc - 1e+06 t167_leaf3 >= -999997
 T167_R_f4_k3: const_Overall_Cond - 1e+06 t167_leaf3 >= -999993
 T167_L_f44_k3: const_Yr_Sold + 1e+06 t167_leaf3 <= 1.002008999999e+06
 T167_R_f3_k4: Overall_Qual_calc - 1e+06 t167_leaf4 >= -999997
 T167_R_f4_k4: const_Overall_Cond - 1e+06 t167_leaf4 >= -999993
 T167_R_f44_k4: const_Yr_Sold - 1e+06 t167_leaf4 >= -997991
 T168_onehot: t168_leaf0 + t168_leaf1 + t168_leaf2 + t168_leaf3
   + t168_leaf4 = 1
 T168_L_f117_k0: const_Condition_1_Artery + 1e+06 t168_leaf0
   <= 1.000000999999e+06
 T168_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t168_leaf0
   <= 1.000000999999e+06
 T168_L_f4_k0: const_Overall_Cond + 1e+06 t168_leaf0 <= 1.000006999999e+06
 T168_L_f117_k1: const_Condition_1_Artery + 1e+06 t168_leaf1
   <= 1.000000999999e+06
 T168_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t168_leaf1
   <= 1.000000999999e+06
 T168_R_f4_k1: const_Overall_Cond - 1e+06 t168_leaf1 >= -999993
 T168_L_f117_k2: const_Condition_1_Artery + 1e+06 t168_leaf2
   <= 1.000000999999e+06
 T168_R_f95_k2: const_Neighborhood_Crawfor - 1e+06 t168_leaf2 >= -999999
 T168_R_f117_k3: const_Condition_1_Artery - 1e+06 t168_leaf3 >= -999999
 T168_L_f33_k3: x_Garage_Area + 1e+06 t168_leaf3 <= 1.000624999999e+06
 T168_R_f117_k4: const_Condition_1_Artery - 1e+06 t168_leaf4 >= -999999
 T168_R_f33_k4: x_Garage_Area - 1e+06 t168_leaf4 >= -999375
 T169_onehot: t169_leaf0 + t169_leaf1 + t169_leaf2 + t169_leaf3
   + t169_leaf4 + t169_leaf5 + t169_leaf6 = 1
 T169_L_f4_k0: const_Overall_Cond + 1e+06 t169_leaf0 <= 1.000006999999e+06
 T169_L_f5_k0: const_Year_Built + 1e+06 t169_leaf0 <= 1.001931999999e+06
 T169_L_f36_k0: x_Wood_Deck_SF + 1e+06 t169_leaf0 <= 1.000167999999e+06
 T169_L_f4_k1: const_Overall_Cond + 1e+06 t169_leaf1 <= 1.000006999999e+06
 T169_L_f5_k1: const_Year_Built + 1e+06 t169_leaf1 <= 1.001931999999e+06
 T169_R_f36_k1: x_Wood_Deck_SF - 1e+06 t169_leaf1 >= -999832
 T169_L_f4_k2: const_Overall_Cond + 1e+06 t169_leaf2 <= 1.000006999999e+06
 T169_R_f5_k2: const_Year_Built - 1e+06 t169_leaf2 >= -998068
 T169_L_f6_k2: const_Year_Remod_Add + 1e+06 t169_leaf2
   <= 1.002006999999e+06
 T169_L_f4_k3: const_Overall_Cond + 1e+06 t169_leaf3 <= 1.000006999999e+06
 T169_R_f5_k3: const_Year_Built - 1e+06 t169_leaf3 >= -998068
 T169_R_f6_k3: const_Year_Remod_Add - 1e+06 t169_leaf3 >= -997993
 T169_R_f4_k4: const_Overall_Cond - 1e+06 t169_leaf4 >= -999993
 T169_L_f44_k4: const_Yr_Sold + 1e+06 t169_leaf4 <= 1.002007999999e+06
 T169_R_f4_k5: const_Overall_Cond - 1e+06 t169_leaf5 >= -999993
 T169_R_f44_k5: const_Yr_Sold - 1e+06 t169_leaf5 >= -997992
 T169_L_f119_k5: const_Condition_1_Norm + 1e+06 t169_leaf5
   <= 1.000000999999e+06
 T169_R_f4_k6: const_Overall_Cond - 1e+06 t169_leaf6 >= -999993
 T169_R_f44_k6: const_Yr_Sold - 1e+06 t169_leaf6 >= -997992
 T169_R_f119_k6: const_Condition_1_Norm - 1e+06 t169_leaf6 >= -999999
 T170_onehot: t170_leaf0 + t170_leaf1 + t170_leaf2 + t170_leaf3
   + t170_leaf4 = 1
 T170_L_f27_k0: x_Kitchen_Qual + 1e+06 t170_leaf0 <= 1.000003999999e+06
 T170_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t170_leaf0
   <= 1.000000999999e+06
 T170_L_f1_k0: const_Lot_Area + 1e+06 t170_leaf0 <= 1.017226999999e+06
 T170_L_f27_k1: x_Kitchen_Qual + 1e+06 t170_leaf1 <= 1.000003999999e+06
 T170_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t170_leaf1
   <= 1.000000999999e+06
 T170_R_f1_k1: const_Lot_Area - 1e+06 t170_leaf1 >= -982773
 T170_L_f27_k2: x_Kitchen_Qual + 1e+06 t170_leaf2 <= 1.000003999999e+06
 T170_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t170_leaf2
   >= -999999
 T170_L_f1_k2: const_Lot_Area + 1e+06 t170_leaf2 <= 1.009899999999e+06
 T170_L_f27_k3: x_Kitchen_Qual + 1e+06 t170_leaf3 <= 1.000003999999e+06
 T170_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t170_leaf3
   >= -999999
 T170_R_f1_k3: const_Lot_Area - 1e+06 t170_leaf3 >= -990100
 T170_R_f27_k4: x_Kitchen_Qual - 1e+06 t170_leaf4 >= -999996
 T171_onehot: t171_leaf0 + t171_leaf1 + t171_leaf2 + t171_leaf3
   + t171_leaf4 = 1
 T171_L_f11_k0: x_Bsmt_Cond + 1e+06 t171_leaf0 <= 1.000002999999e+06
 T171_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t171_leaf0
   <= 1.000000999999e+06
 T171_L_f40_k0: x_Screen_Porch + 1e+06 t171_leaf0 <= 1.000209999999e+06
 T171_L_f11_k1: x_Bsmt_Cond + 1e+06 t171_leaf1 <= 1.000002999999e+06
 T171_L_f96_k1: const_Neighborhood_Edwards + 1e+06 t171_leaf1
   <= 1.000000999999e+06
 T171_R_f40_k1: x_Screen_Porch - 1e+06 t171_leaf1 >= -999790
 T171_L_f11_k2: x_Bsmt_Cond + 1e+06 t171_leaf2 <= 1.000002999999e+06
 T171_R_f96_k2: const_Neighborhood_Edwards - 1e+06 t171_leaf2 >= -999999
 T171_L_f81_k2: const_Lot_Config_Corner + 1e+06 t171_leaf2
   <= 1.000000999999e+06
 T171_L_f11_k3: x_Bsmt_Cond + 1e+06 t171_leaf3 <= 1.000002999999e+06
 T171_R_f96_k3: const_Neighborhood_Edwards - 1e+06 t171_leaf3 >= -999999
 T171_R_f81_k3: const_Lot_Config_Corner - 1e+06 t171_leaf3 >= -999999
 T171_R_f11_k4: x_Bsmt_Cond - 1e+06 t171_leaf4 >= -999997
 T172_onehot: t172_leaf0 + t172_leaf1 + t172_leaf2 + t172_leaf3
   + t172_leaf4 + t172_leaf5 + t172_leaf6 = 1
 T172_L_f211_k0: x_b1_is_ALQ + 1e+06 t172_leaf0 <= 1.000000999999e+06
 T172_L_f36_k0: x_Wood_Deck_SF + 1e+06 t172_leaf0 <= 1.000055999999e+06
 T172_L_f24_k0: x_Half_Bath + 1e+06 t172_leaf0 <= 1.000000999999e+06
 T172_L_f211_k1: x_b1_is_ALQ + 1e+06 t172_leaf1 <= 1.000000999999e+06
 T172_L_f36_k1: x_Wood_Deck_SF + 1e+06 t172_leaf1 <= 1.000055999999e+06
 T172_R_f24_k1: x_Half_Bath - 1e+06 t172_leaf1 >= -999999
 T172_L_f211_k2: x_b1_is_ALQ + 1e+06 t172_leaf2 <= 1.000000999999e+06
 T172_R_f36_k2: x_Wood_Deck_SF - 1e+06 t172_leaf2 >= -999944
 T172_L_f28_k2: x_TotRms_AbvGrd + 1e+06 t172_leaf2 <= 1.000010999999e+06
 T172_L_f211_k3: x_b1_is_ALQ + 1e+06 t172_leaf3 <= 1.000000999999e+06
 T172_R_f36_k3: x_Wood_Deck_SF - 1e+06 t172_leaf3 >= -999944
 T172_R_f28_k3: x_TotRms_AbvGrd - 1e+06 t172_leaf3 >= -999989
 T172_R_f211_k4: x_b1_is_ALQ - 1e+06 t172_leaf4 >= -999999
 T172_L_f245_k4: const_FunctioNo_aplical_Typ + 1e+06 t172_leaf4
   <= 1.000000999999e+06
 T172_R_f211_k5: x_b1_is_ALQ - 1e+06 t172_leaf5 >= -999999
 T172_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t172_leaf5 >= -999999
 T172_L_f31_k5: const_Garage_Yr_Blt + 1e+06 t172_leaf5
   <= 1.001976999999e+06
 T172_R_f211_k6: x_b1_is_ALQ - 1e+06 t172_leaf6 >= -999999
 T172_R_f245_k6: const_FunctioNo_aplical_Typ - 1e+06 t172_leaf6 >= -999999
 T172_R_f31_k6: const_Garage_Yr_Blt - 1e+06 t172_leaf6 >= -998023
 T173_onehot: t173_leaf0 + t173_leaf1 + t173_leaf2 + t173_leaf3
   + t173_leaf4 + t173_leaf5 + t173_leaf6 = 1
 T173_L_f9_k0: x_Exter_Cond + 1e+06 t173_leaf0 <= 1.000001999999e+06
 T173_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t173_leaf0
   <= 1.000000999999e+06
 T173_L_f9_k1: x_Exter_Cond + 1e+06 t173_leaf1 <= 1.000001999999e+06
 T173_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t173_leaf1 >= -999999
 T173_L_f28_k1: x_TotRms_AbvGrd + 1e+06 t173_leaf1 <= 1.000004999999e+06
 T173_L_f9_k2: x_Exter_Cond + 1e+06 t173_leaf2 <= 1.000001999999e+06
 T173_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t173_leaf2 >= -999999
 T173_R_f28_k2: x_TotRms_AbvGrd - 1e+06 t173_leaf2 >= -999995
 T173_R_f9_k3: x_Exter_Cond - 1e+06 t173_leaf3 >= -999998
 T173_L_f4_k3: const_Overall_Cond + 1e+06 t173_leaf3 <= 1.000006999999e+06
 T173_L_f5_k3: const_Year_Built + 1e+06 t173_leaf3 <= 1.001953999999e+06
 T173_R_f9_k4: x_Exter_Cond - 1e+06 t173_leaf4 >= -999998
 T173_L_f4_k4: const_Overall_Cond + 1e+06 t173_leaf4 <= 1.000006999999e+06
 T173_R_f5_k4: const_Year_Built - 1e+06 t173_leaf4 >= -998046
 T173_R_f9_k5: x_Exter_Cond - 1e+06 t173_leaf5 >= -999998
 T173_R_f4_k5: const_Overall_Cond - 1e+06 t173_leaf5 >= -999993
 T173_L_f31_k5: const_Garage_Yr_Blt + 1e+06 t173_leaf5
   <= 1.001994999999e+06
 T173_R_f9_k6: x_Exter_Cond - 1e+06 t173_leaf6 >= -999998
 T173_R_f4_k6: const_Overall_Cond - 1e+06 t173_leaf6 >= -999993
 T173_R_f31_k6: const_Garage_Yr_Blt - 1e+06 t173_leaf6 >= -998005
 T174_onehot: t174_leaf0 + t174_leaf1 + t174_leaf2 + t174_leaf3 = 1
 T174_L_f33_k0: x_Garage_Area + 1e+06 t174_leaf0 <= 1.001153999999e+06
 T174_L_f207_k0: const_Bsmt_Exposure_Gd + 1e+06 t174_leaf0
   <= 1.000000999999e+06
 T174_L_f44_k0: const_Yr_Sold + 1e+06 t174_leaf0 <= 1.002006999999e+06
 T174_L_f33_k1: x_Garage_Area + 1e+06 t174_leaf1 <= 1.001153999999e+06
 T174_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t174_leaf1
   <= 1.000000999999e+06
 T174_R_f44_k1: const_Yr_Sold - 1e+06 t174_leaf1 >= -997993
 T174_L_f33_k2: x_Garage_Area + 1e+06 t174_leaf2 <= 1.001153999999e+06
 T174_R_f207_k2: const_Bsmt_Exposure_Gd - 1e+06 t174_leaf2 >= -999999
 T174_R_f33_k3: x_Garage_Area - 1e+06 t174_leaf3 >= -998846
 T175_onehot: t175_leaf0 + t175_leaf1 + t175_leaf2 + t175_leaf3
   + t175_leaf4 = 1
 T175_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t175_leaf0
   <= 1.002135999999e+06
 T175_L_f10_k0: const_Bsmt_Qual + 1e+06 t175_leaf0 <= 1.000001999999e+06
 T175_L_f33_k0: x_Garage_Area + 1e+06 t175_leaf0 <= 1.000486999999e+06
 T175_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t175_leaf1
   <= 1.002135999999e+06
 T175_L_f10_k1: const_Bsmt_Qual + 1e+06 t175_leaf1 <= 1.000001999999e+06
 T175_R_f33_k1: x_Garage_Area - 1e+06 t175_leaf1 >= -999513
 T175_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t175_leaf2
   <= 1.002135999999e+06
 T175_R_f10_k2: const_Bsmt_Qual - 1e+06 t175_leaf2 >= -999998
 T175_L_f117_k2: const_Condition_1_Artery + 1e+06 t175_leaf2
   <= 1.000000999999e+06
 T175_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t175_leaf3
   <= 1.002135999999e+06
 T175_R_f10_k3: const_Bsmt_Qual - 1e+06 t175_leaf3 >= -999998
 T175_R_f117_k3: const_Condition_1_Artery - 1e+06 t175_leaf3 >= -999999
 T175_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t175_leaf4 >= -997864
 T176_onehot: t176_leaf0 + t176_leaf1 + t176_leaf2 + t176_leaf3
   + t176_leaf4 + t176_leaf5 + t176_leaf6 = 1
 T176_L_f44_k0: const_Yr_Sold + 1e+06 t176_leaf0 <= 1.002006999999e+06
 T176_L_f231_k0: central_air_no + 1e+06 t176_leaf0 <= 1.000000999999e+06
 T176_L_f4_k0: const_Overall_Cond + 1e+06 t176_leaf0 <= 1.000007999999e+06
 T176_L_f44_k1: const_Yr_Sold + 1e+06 t176_leaf1 <= 1.002006999999e+06
 T176_L_f231_k1: central_air_no + 1e+06 t176_leaf1 <= 1.000000999999e+06
 T176_R_f4_k1: const_Overall_Cond - 1e+06 t176_leaf1 >= -999992
 T176_L_f44_k2: const_Yr_Sold + 1e+06 t176_leaf2 <= 1.002006999999e+06
 T176_R_f231_k2: central_air_no - 1e+06 t176_leaf2 >= -999999
 T176_L_f1_k2: const_Lot_Area + 1e+06 t176_leaf2 <= 1.006239999999e+06
 T176_L_f44_k3: const_Yr_Sold + 1e+06 t176_leaf3 <= 1.002006999999e+06
 T176_R_f231_k3: central_air_no - 1e+06 t176_leaf3 >= -999999
 T176_R_f1_k3: const_Lot_Area - 1e+06 t176_leaf3 >= -993760
 T176_R_f44_k4: const_Yr_Sold - 1e+06 t176_leaf4 >= -997993
 T176_L_f33_k4: x_Garage_Area + 1e+06 t176_leaf4 <= 1.001153999999e+06
 T176_L_f245_k4: const_FunctioNo_aplical_Typ + 1e+06 t176_leaf4
   <= 1.000000999999e+06
 T176_R_f44_k5: const_Yr_Sold - 1e+06 t176_leaf5 >= -997993
 T176_L_f33_k5: x_Garage_Area + 1e+06 t176_leaf5 <= 1.001153999999e+06
 T176_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t176_leaf5 >= -999999
 T176_R_f44_k6: const_Yr_Sold - 1e+06 t176_leaf6 >= -997993
 T176_R_f33_k6: x_Garage_Area - 1e+06 t176_leaf6 >= -998846
 T177_onehot: t177_leaf0 + t177_leaf1 + t177_leaf2 + t177_leaf3
   + t177_leaf4 + t177_leaf5 + t177_leaf6 = 1
 T177_L_f16_k0: x_Heating_QC + 1e+06 t177_leaf0 <= 1.000002999999e+06
 T177_L_f36_k0: x_Wood_Deck_SF + 1e+06 t177_leaf0 <= 1.000158999999e+06
 T177_L_f33_k0: x_Garage_Area + 1e+06 t177_leaf0 <= 1.000783999999e+06
 T177_L_f16_k1: x_Heating_QC + 1e+06 t177_leaf1 <= 1.000002999999e+06
 T177_L_f36_k1: x_Wood_Deck_SF + 1e+06 t177_leaf1 <= 1.000158999999e+06
 T177_R_f33_k1: x_Garage_Area - 1e+06 t177_leaf1 >= -999216
 T177_L_f16_k2: x_Heating_QC + 1e+06 t177_leaf2 <= 1.000002999999e+06
 T177_R_f36_k2: x_Wood_Deck_SF - 1e+06 t177_leaf2 >= -999841
 T177_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t177_leaf2
   <= 1.001977999999e+06
 T177_L_f16_k3: x_Heating_QC + 1e+06 t177_leaf3 <= 1.000002999999e+06
 T177_R_f36_k3: x_Wood_Deck_SF - 1e+06 t177_leaf3 >= -999841
 T177_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t177_leaf3 >= -998022
 T177_R_f16_k4: x_Heating_QC - 1e+06 t177_leaf4 >= -999997
 T177_L_f37_k4: x_Open_Porch_SF + 1e+06 t177_leaf4 <= 1.000133999999e+06
 T177_L_f3_k4: Overall_Qual_calc + 1e+06 t177_leaf4 <= 1.000006999999e+06
 T177_R_f16_k5: x_Heating_QC - 1e+06 t177_leaf5 >= -999997
 T177_L_f37_k5: x_Open_Porch_SF + 1e+06 t177_leaf5 <= 1.000133999999e+06
 T177_R_f3_k5: Overall_Qual_calc - 1e+06 t177_leaf5 >= -999993
 T177_R_f16_k6: x_Heating_QC - 1e+06 t177_leaf6 >= -999997
 T177_R_f37_k6: x_Open_Porch_SF - 1e+06 t177_leaf6 >= -999866
 T178_onehot: t178_leaf0 + t178_leaf1 + t178_leaf2 + t178_leaf3
   + t178_leaf4 = 1
 T178_L_f44_k0: const_Yr_Sold + 1e+06 t178_leaf0 <= 1.002006999999e+06
 T178_L_f103_k0: const_Neighborhood_Mitchel + 1e+06 t178_leaf0
   <= 1.000000999999e+06
 T178_L_f44_k1: const_Yr_Sold + 1e+06 t178_leaf1 <= 1.002006999999e+06
 T178_R_f103_k1: const_Neighborhood_Mitchel - 1e+06 t178_leaf1 >= -999999
 T178_R_f44_k2: const_Yr_Sold - 1e+06 t178_leaf2 >= -997993
 T178_L_f63_k2: const_MS_Zoning_FV + 1e+06 t178_leaf2
   <= 1.000000999999e+06
 T178_L_f69_k2: const_Street_Pave + 1e+06 t178_leaf2 <= 1.000000999999e+06
 T178_R_f44_k3: const_Yr_Sold - 1e+06 t178_leaf3 >= -997993
 T178_L_f63_k3: const_MS_Zoning_FV + 1e+06 t178_leaf3
   <= 1.000000999999e+06
 T178_R_f69_k3: const_Street_Pave - 1e+06 t178_leaf3 >= -999999
 T178_R_f44_k4: const_Yr_Sold - 1e+06 t178_leaf4 >= -997993
 T178_R_f63_k4: const_MS_Zoning_FV - 1e+06 t178_leaf4 >= -999999
 T179_onehot: t179_leaf0 + t179_leaf1 + t179_leaf2 + t179_leaf3
   + t179_leaf4 + t179_leaf5 = 1
 T179_L_f20_k0: x_Gr_Liv_Area + 1e+06 t179_leaf0 <= 1.000767999999e+06
 T179_L_f17_k0: x_1st_Flr_SF + 1e+06 t179_leaf0 <= 1.000716999999e+06
 T179_L_f192_k0: x_ex2_is_Wd_Sdng + 1e+06 t179_leaf0 <= 1.000000999999e+06
 T179_L_f20_k1: x_Gr_Liv_Area + 1e+06 t179_leaf1 <= 1.000767999999e+06
 T179_L_f17_k1: x_1st_Flr_SF + 1e+06 t179_leaf1 <= 1.000716999999e+06
 T179_R_f192_k1: x_ex2_is_Wd_Sdng - 1e+06 t179_leaf1 >= -999999
 T179_L_f20_k2: x_Gr_Liv_Area + 1e+06 t179_leaf2 <= 1.000767999999e+06
 T179_R_f17_k2: x_1st_Flr_SF - 1e+06 t179_leaf2 >= -999283
 T179_R_f20_k3: x_Gr_Liv_Area - 1e+06 t179_leaf3 >= -999232
 T179_L_f44_k3: const_Yr_Sold + 1e+06 t179_leaf3 <= 1.002006999999e+06
 T179_R_f20_k4: x_Gr_Liv_Area - 1e+06 t179_leaf4 >= -999232
 T179_R_f44_k4: const_Yr_Sold - 1e+06 t179_leaf4 >= -997993
 T179_L_f37_k4: x_Open_Porch_SF + 1e+06 t179_leaf4 <= 1.000182999999e+06
 T179_R_f20_k5: x_Gr_Liv_Area - 1e+06 t179_leaf5 >= -999232
 T179_R_f44_k5: const_Yr_Sold - 1e+06 t179_leaf5 >= -997993
 T179_R_f37_k5: x_Open_Porch_SF - 1e+06 t179_leaf5 >= -999817
 T180_onehot: t180_leaf0 + t180_leaf1 + t180_leaf2 + t180_leaf3
   + t180_leaf4 = 1
 T180_L_f3_k0: Overall_Qual_calc + 1e+06 t180_leaf0 <= 1.000002999999e+06
 T180_R_f3_k1: Overall_Qual_calc - 1e+06 t180_leaf1 >= -999997
 T180_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t180_leaf1
   <= 1.000000999999e+06
 T180_L_f16_k1: x_Heating_QC + 1e+06 t180_leaf1 <= 1.000002999999e+06
 T180_R_f3_k2: Overall_Qual_calc - 1e+06 t180_leaf2 >= -999997
 T180_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t180_leaf2
   <= 1.000000999999e+06
 T180_R_f16_k2: x_Heating_QC - 1e+06 t180_leaf2 >= -999997
 T180_R_f3_k3: Overall_Qual_calc - 1e+06 t180_leaf3 >= -999997
 T180_R_f95_k3: const_Neighborhood_Crawfor - 1e+06 t180_leaf3 >= -999999
 T180_L_f81_k3: const_Lot_Config_Corner + 1e+06 t180_leaf3
   <= 1.000000999999e+06
 T180_R_f3_k4: Overall_Qual_calc - 1e+06 t180_leaf4 >= -999997
 T180_R_f95_k4: const_Neighborhood_Crawfor - 1e+06 t180_leaf4 >= -999999
 T180_R_f81_k4: const_Lot_Config_Corner - 1e+06 t180_leaf4 >= -999999
 T181_onehot: t181_leaf0 + t181_leaf1 + t181_leaf2 + t181_leaf3
   + t181_leaf4 + t181_leaf5 + t181_leaf6 = 1
 T181_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t181_leaf0 <= 1.000112999999e+06
 T181_L_f11_k0: x_Bsmt_Cond + 1e+06 t181_leaf0 <= 1.000002999999e+06
 T181_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t181_leaf0 <= 1.000624999999e+06
 T181_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t181_leaf1 <= 1.000112999999e+06
 T181_L_f11_k1: x_Bsmt_Cond + 1e+06 t181_leaf1 <= 1.000002999999e+06
 T181_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t181_leaf1 >= -999375
 T181_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t181_leaf2 <= 1.000112999999e+06
 T181_R_f11_k2: x_Bsmt_Cond - 1e+06 t181_leaf2 >= -999997
 T181_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t181_leaf3 >= -999887
 T181_L_f20_k3: x_Gr_Liv_Area + 1e+06 t181_leaf3 <= 1.000863999999e+06
 T181_L_f5_k3: const_Year_Built + 1e+06 t181_leaf3 <= 1.001923999999e+06
 T181_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t181_leaf4 >= -999887
 T181_L_f20_k4: x_Gr_Liv_Area + 1e+06 t181_leaf4 <= 1.000863999999e+06
 T181_R_f5_k4: const_Year_Built - 1e+06 t181_leaf4 >= -998076
 T181_R_f14_k5: x_Bsmt_Unf_SF - 1e+06 t181_leaf5 >= -999887
 T181_R_f20_k5: x_Gr_Liv_Area - 1e+06 t181_leaf5 >= -999136
 T181_L_f18_k5: const_2nd_Flr_SF + 1e+06 t181_leaf5 <= 1.001367999999e+06
 T181_R_f14_k6: x_Bsmt_Unf_SF - 1e+06 t181_leaf6 >= -999887
 T181_R_f20_k6: x_Gr_Liv_Area - 1e+06 t181_leaf6 >= -999136
 T181_R_f18_k6: const_2nd_Flr_SF - 1e+06 t181_leaf6 >= -998632
 T182_onehot: t182_leaf0 + t182_leaf1 + t182_leaf2 + t182_leaf3
   + t182_leaf4 + t182_leaf5 + t182_leaf6 + t182_leaf7 = 1
 T182_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t182_leaf0
   <= 1.000000999999e+06
 T182_L_f1_k0: const_Lot_Area + 1e+06 t182_leaf0 <= 1.014679999999e+06
 T182_L_f17_k0: x_1st_Flr_SF + 1e+06 t182_leaf0 <= 1.000575999999e+06
 T182_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t182_leaf1
   <= 1.000000999999e+06
 T182_L_f1_k1: const_Lot_Area + 1e+06 t182_leaf1 <= 1.014679999999e+06
 T182_R_f17_k1: x_1st_Flr_SF - 1e+06 t182_leaf1 >= -999424
 T182_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t182_leaf2
   <= 1.000000999999e+06
 T182_R_f1_k2: const_Lot_Area - 1e+06 t182_leaf2 >= -985320
 T182_L_f37_k2: x_Open_Porch_SF + 1e+06 t182_leaf2 <= 1.000273999999e+06
 T182_L_f293_k3: const_Sale_Condition_Abnorml + 1e+06 t182_leaf3
   <= 1.000000999999e+06
 T182_R_f1_k3: const_Lot_Area - 1e+06 t182_leaf3 >= -985320
 T182_R_f37_k3: x_Open_Porch_SF - 1e+06 t182_leaf3 >= -999726
 T182_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t182_leaf4
   >= -999999
 T182_L_f3_k4: Overall_Qual_calc + 1e+06 t182_leaf4 <= 1.000004999999e+06
 T182_L_f4_k4: const_Overall_Cond + 1e+06 t182_leaf4 <= 1.000005999999e+06
 T182_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t182_leaf5
   >= -999999
 T182_L_f3_k5: Overall_Qual_calc + 1e+06 t182_leaf5 <= 1.000004999999e+06
 T182_R_f4_k5: const_Overall_Cond - 1e+06 t182_leaf5 >= -999994
 T182_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t182_leaf6
   >= -999999
 T182_R_f3_k6: Overall_Qual_calc - 1e+06 t182_leaf6 >= -999995
 T182_L_f237_k6: x_elect_is_SBrkr + 1e+06 t182_leaf6 <= 1.000000999999e+06
 T182_R_f293_k7: const_Sale_Condition_Abnorml - 1e+06 t182_leaf7
   >= -999999
 T182_R_f3_k7: Overall_Qual_calc - 1e+06 t182_leaf7 >= -999995
 T182_R_f237_k7: x_elect_is_SBrkr - 1e+06 t182_leaf7 >= -999999
 T183_onehot: t183_leaf0 + t183_leaf1 + t183_leaf2 + t183_leaf3
   + t183_leaf4 + t183_leaf5 = 1
 T183_L_f118_k0: const_Condition_1_Feedr + 1e+06 t183_leaf0
   <= 1.000000999999e+06
 T183_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t183_leaf0
   <= 1.000001999999e+06
 T183_L_f4_k0: const_Overall_Cond + 1e+06 t183_leaf0 <= 1.000006999999e+06
 T183_L_f118_k1: const_Condition_1_Feedr + 1e+06 t183_leaf1
   <= 1.000000999999e+06
 T183_L_f21_k1: const_Bsmt_Full_Bath + 1e+06 t183_leaf1
   <= 1.000001999999e+06
 T183_R_f4_k1: const_Overall_Cond - 1e+06 t183_leaf1 >= -999993
 T183_L_f118_k2: const_Condition_1_Feedr + 1e+06 t183_leaf2
   <= 1.000000999999e+06
 T183_R_f21_k2: const_Bsmt_Full_Bath - 1e+06 t183_leaf2 >= -999998
 T183_R_f118_k3: const_Condition_1_Feedr - 1e+06 t183_leaf3 >= -999999
 T183_L_f80_k3: const_Land_Contour_Lvl + 1e+06 t183_leaf3
   <= 1.000000999999e+06
 T183_L_f175_k3: x_ex1_is_Wd_Sdng + 1e+06 t183_leaf3 <= 1.000000999999e+06
 T183_R_f118_k4: const_Condition_1_Feedr - 1e+06 t183_leaf4 >= -999999
 T183_L_f80_k4: const_Land_Contour_Lvl + 1e+06 t183_leaf4
   <= 1.000000999999e+06
 T183_R_f175_k4: x_ex1_is_Wd_Sdng - 1e+06 t183_leaf4 >= -999999
 T183_R_f118_k5: const_Condition_1_Feedr - 1e+06 t183_leaf5 >= -999999
 T183_R_f80_k5: const_Land_Contour_Lvl - 1e+06 t183_leaf5 >= -999999
 T184_onehot: t184_leaf0 + t184_leaf1 + t184_leaf2 + t184_leaf3
   + t184_leaf4 + t184_leaf5 = 1
 T184_L_f20_k0: x_Gr_Liv_Area + 1e+06 t184_leaf0 <= 1.001960999999e+06
 T184_L_f113_k0: const_Neighborhood_Somerst + 1e+06 t184_leaf0
   <= 1.000000999999e+06
 T184_L_f211_k0: x_b1_is_ALQ + 1e+06 t184_leaf0 <= 1.000000999999e+06
 T184_L_f20_k1: x_Gr_Liv_Area + 1e+06 t184_leaf1 <= 1.001960999999e+06
 T184_L_f113_k1: const_Neighborhood_Somerst + 1e+06 t184_leaf1
   <= 1.000000999999e+06
 T184_R_f211_k1: x_b1_is_ALQ - 1e+06 t184_leaf1 >= -999999
 T184_L_f20_k2: x_Gr_Liv_Area + 1e+06 t184_leaf2 <= 1.001960999999e+06
 T184_R_f113_k2: const_Neighborhood_Somerst - 1e+06 t184_leaf2 >= -999999
 T184_R_f20_k3: x_Gr_Liv_Area - 1e+06 t184_leaf3 >= -998039
 T184_L_f96_k3: const_Neighborhood_Edwards + 1e+06 t184_leaf3
   <= 1.000000999999e+06
 T184_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t184_leaf3
   <= 1.002135999999e+06
 T184_R_f20_k4: x_Gr_Liv_Area - 1e+06 t184_leaf4 >= -998039
 T184_L_f96_k4: const_Neighborhood_Edwards + 1e+06 t184_leaf4
   <= 1.000000999999e+06
 T184_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t184_leaf4 >= -997864
 T184_R_f20_k5: x_Gr_Liv_Area - 1e+06 t184_leaf5 >= -998039
 T184_R_f96_k5: const_Neighborhood_Edwards - 1e+06 t184_leaf5 >= -999999
 T185_onehot: t185_leaf0 + t185_leaf1 + t185_leaf2 + t185_leaf3
   + t185_leaf4 + t185_leaf5 = 1
 T185_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t185_leaf0 <= 1.001731999999e+06
 T185_L_f29_k0: const_Fireplaces + 1e+06 t185_leaf0 <= 1.000001999999e+06
 T185_L_f36_k0: x_Wood_Deck_SF + 1e+06 t185_leaf0 <= 1.000208999999e+06
 T185_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t185_leaf1 <= 1.001731999999e+06
 T185_L_f29_k1: const_Fireplaces + 1e+06 t185_leaf1 <= 1.000001999999e+06
 T185_R_f36_k1: x_Wood_Deck_SF - 1e+06 t185_leaf1 >= -999791
 T185_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t185_leaf2 <= 1.001731999999e+06
 T185_R_f29_k2: const_Fireplaces - 1e+06 t185_leaf2 >= -999998
 T185_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t185_leaf2
   <= 1.001481999999e+06
 T185_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t185_leaf3 <= 1.001731999999e+06
 T185_R_f29_k3: const_Fireplaces - 1e+06 t185_leaf3 >= -999998
 T185_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t185_leaf3 >= -998518
 T185_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t185_leaf4 >= -998268
 T185_L_f76_k4: const_Lot_Shape_Reg + 1e+06 t185_leaf4
   <= 1.000000999999e+06
 T185_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t185_leaf5 >= -998268
 T185_R_f76_k5: const_Lot_Shape_Reg - 1e+06 t185_leaf5 >= -999999
 T186_onehot: t186_leaf0 + t186_leaf1 + t186_leaf2 + t186_leaf3
   + t186_leaf4 + t186_leaf5 + t186_leaf6 = 1
 T186_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t186_leaf0
   <= 1.000000999999e+06
 T186_L_f5_k0: const_Year_Built + 1e+06 t186_leaf0 <= 1.002004999999e+06
 T186_L_f141_k0: const_House_Style_1Story + 1e+06 t186_leaf0
   <= 1.000000999999e+06
 T186_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t186_leaf1
   <= 1.000000999999e+06
 T186_L_f5_k1: const_Year_Built + 1e+06 t186_leaf1 <= 1.002004999999e+06
 T186_R_f141_k1: const_House_Style_1Story - 1e+06 t186_leaf1 >= -999999
 T186_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t186_leaf2
   <= 1.000000999999e+06
 T186_R_f5_k2: const_Year_Built - 1e+06 t186_leaf2 >= -997995
 T186_L_f66_k2: const_MS_Zoning_RL + 1e+06 t186_leaf2
   <= 1.000000999999e+06
 T186_L_f293_k3: const_Sale_Condition_Abnorml + 1e+06 t186_leaf3
   <= 1.000000999999e+06
 T186_R_f5_k3: const_Year_Built - 1e+06 t186_leaf3 >= -997995
 T186_R_f66_k3: const_MS_Zoning_RL - 1e+06 t186_leaf3 >= -999999
 T186_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t186_leaf4
   >= -999999
 T186_L_f3_k4: Overall_Qual_calc + 1e+06 t186_leaf4 <= 1.000004999999e+06
 T186_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t186_leaf5
   >= -999999
 T186_R_f3_k5: Overall_Qual_calc - 1e+06 t186_leaf5 >= -999995
 T186_L_f38_k5: x_Enclosed_Porch + 1e+06 t186_leaf5 <= 1.000031999999e+06
 T186_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t186_leaf6
   >= -999999
 T186_R_f3_k6: Overall_Qual_calc - 1e+06 t186_leaf6 >= -999995
 T186_R_f38_k6: x_Enclosed_Porch - 1e+06 t186_leaf6 >= -999968
 T187_onehot: t187_leaf0 + t187_leaf1 + t187_leaf2 + t187_leaf3
   + t187_leaf4 + t187_leaf5 + t187_leaf6 = 1
 T187_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t187_leaf0
   <= 1.000000999999e+06
 T187_L_f117_k0: const_Condition_1_Artery + 1e+06 t187_leaf0
   <= 1.000000999999e+06
 T187_L_f4_k0: const_Overall_Cond + 1e+06 t187_leaf0 <= 1.000005999999e+06
 T187_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t187_leaf1
   <= 1.000000999999e+06
 T187_L_f117_k1: const_Condition_1_Artery + 1e+06 t187_leaf1
   <= 1.000000999999e+06
 T187_R_f4_k1: const_Overall_Cond - 1e+06 t187_leaf1 >= -999994
 T187_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t187_leaf2
   <= 1.000000999999e+06
 T187_R_f117_k2: const_Condition_1_Artery - 1e+06 t187_leaf2 >= -999999
 T187_L_f16_k2: x_Heating_QC + 1e+06 t187_leaf2 <= 1.000002999999e+06
 T187_L_f293_k3: const_Sale_Condition_Abnorml + 1e+06 t187_leaf3
   <= 1.000000999999e+06
 T187_R_f117_k3: const_Condition_1_Artery - 1e+06 t187_leaf3 >= -999999
 T187_R_f16_k3: x_Heating_QC - 1e+06 t187_leaf3 >= -999997
 T187_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t187_leaf4
   >= -999999
 T187_L_f4_k4: const_Overall_Cond + 1e+06 t187_leaf4 <= 1.000003999999e+06
 T187_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t187_leaf5
   >= -999999
 T187_R_f4_k5: const_Overall_Cond - 1e+06 t187_leaf5 >= -999996
 T187_L_f206_k5: const_Bsmt_Exposure_Av + 1e+06 t187_leaf5
   <= 1.000000999999e+06
 T187_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t187_leaf6
   >= -999999
 T187_R_f4_k6: const_Overall_Cond - 1e+06 t187_leaf6 >= -999996
 T187_R_f206_k6: const_Bsmt_Exposure_Av - 1e+06 t187_leaf6 >= -999999
 T188_onehot: t188_leaf0 + t188_leaf1 + t188_leaf2 + t188_leaf3
   + t188_leaf4 + t188_leaf5 + t188_leaf6 + t188_leaf7 = 1
 T188_L_f44_k0: const_Yr_Sold + 1e+06 t188_leaf0 <= 1.002006999999e+06
 T188_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t188_leaf0 <= 1.000966999999e+06
 T188_L_f6_k0: const_Year_Remod_Add + 1e+06 t188_leaf0
   <= 1.001950999999e+06
 T188_L_f44_k1: const_Yr_Sold + 1e+06 t188_leaf1 <= 1.002006999999e+06
 T188_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t188_leaf1 <= 1.000966999999e+06
 T188_R_f6_k1: const_Year_Remod_Add - 1e+06 t188_leaf1 >= -998049
 T188_L_f44_k2: const_Yr_Sold + 1e+06 t188_leaf2 <= 1.002006999999e+06
 T188_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t188_leaf2 >= -999033
 T188_L_f191_k2: x_ex2_is_VinylSd + 1e+06 t188_leaf2 <= 1.000000999999e+06
 T188_L_f44_k3: const_Yr_Sold + 1e+06 t188_leaf3 <= 1.002006999999e+06
 T188_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t188_leaf3 >= -999033
 T188_R_f191_k3: x_ex2_is_VinylSd - 1e+06 t188_leaf3 >= -999999
 T188_R_f44_k4: const_Yr_Sold - 1e+06 t188_leaf4 >= -997993
 T188_L_f78_k4: const_Land_Contour_HLS + 1e+06 t188_leaf4
   <= 1.000000999999e+06
 T188_L_f218_k4: x_b2_is_ALQ + 1e+06 t188_leaf4 <= 1.000000999999e+06
 T188_R_f44_k5: const_Yr_Sold - 1e+06 t188_leaf5 >= -997993
 T188_L_f78_k5: const_Land_Contour_HLS + 1e+06 t188_leaf5
   <= 1.000000999999e+06
 T188_R_f218_k5: x_b2_is_ALQ - 1e+06 t188_leaf5 >= -999999
 T188_R_f44_k6: const_Yr_Sold - 1e+06 t188_leaf6 >= -997993
 T188_R_f78_k6: const_Land_Contour_HLS - 1e+06 t188_leaf6 >= -999999
 T188_L_f12_k6: x_BsmtFin_SF_1 + 1e+06 t188_leaf6 <= 1.000080999999e+06
 T188_R_f44_k7: const_Yr_Sold - 1e+06 t188_leaf7 >= -997993
 T188_R_f78_k7: const_Land_Contour_HLS - 1e+06 t188_leaf7 >= -999999
 T188_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t188_leaf7 >= -999919
 T189_onehot: t189_leaf0 + t189_leaf1 + t189_leaf2 + t189_leaf3
   + t189_leaf4 = 1
 T189_L_f37_k0: x_Open_Porch_SF + 1e+06 t189_leaf0 <= 1.000443999999e+06
 T189_L_f207_k0: const_Bsmt_Exposure_Gd + 1e+06 t189_leaf0
   <= 1.000000999999e+06
 T189_L_f1_k0: const_Lot_Area + 1e+06 t189_leaf0 <= 1.003875999999e+06
 T189_L_f37_k1: x_Open_Porch_SF + 1e+06 t189_leaf1 <= 1.000443999999e+06
 T189_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t189_leaf1
   <= 1.000000999999e+06
 T189_R_f1_k1: const_Lot_Area - 1e+06 t189_leaf1 >= -996124
 T189_L_f37_k2: x_Open_Porch_SF + 1e+06 t189_leaf2 <= 1.000443999999e+06
 T189_R_f207_k2: const_Bsmt_Exposure_Gd - 1e+06 t189_leaf2 >= -999999
 T189_L_f38_k2: x_Enclosed_Porch + 1e+06 t189_leaf2 <= 1.000034999999e+06
 T189_L_f37_k3: x_Open_Porch_SF + 1e+06 t189_leaf3 <= 1.000443999999e+06
 T189_R_f207_k3: const_Bsmt_Exposure_Gd - 1e+06 t189_leaf3 >= -999999
 T189_R_f38_k3: x_Enclosed_Porch - 1e+06 t189_leaf3 >= -999965
 T189_R_f37_k4: x_Open_Porch_SF - 1e+06 t189_leaf4 >= -999556
 T190_onehot: t190_leaf0 + t190_leaf1 + t190_leaf2 + t190_leaf3
   + t190_leaf4 + t190_leaf5 = 1
 T190_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t190_leaf0
   <= 1.002004999999e+06
 T190_L_f20_k0: x_Gr_Liv_Area + 1e+06 t190_leaf0 <= 1.000746999999e+06
 T190_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t190_leaf0 <= 1.000272999999e+06
 T190_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t190_leaf1
   <= 1.002004999999e+06
 T190_L_f20_k1: x_Gr_Liv_Area + 1e+06 t190_leaf1 <= 1.000746999999e+06
 T190_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t190_leaf1 >= -999727
 T190_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t190_leaf2
   <= 1.002004999999e+06
 T190_R_f20_k2: x_Gr_Liv_Area - 1e+06 t190_leaf2 >= -999253
 T190_L_f93_k2: const_Neighborhood_ClearCr + 1e+06 t190_leaf2
   <= 1.000000999999e+06
 T190_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t190_leaf3
   <= 1.002004999999e+06
 T190_R_f20_k3: x_Gr_Liv_Area - 1e+06 t190_leaf3 >= -999253
 T190_R_f93_k3: const_Neighborhood_ClearCr - 1e+06 t190_leaf3 >= -999999
 T190_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t190_leaf4 >= -997995
 T190_L_f40_k4: x_Screen_Porch + 1e+06 t190_leaf4 <= 1.000039999999e+06
 T190_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t190_leaf5 >= -997995
 T190_R_f40_k5: x_Screen_Porch - 1e+06 t190_leaf5 >= -999960
 T191_onehot: t191_leaf0 + t191_leaf1 + t191_leaf2 + t191_leaf3
   + t191_leaf4 + t191_leaf5 + t191_leaf6 = 1
 T191_L_f1_k0: const_Lot_Area + 1e+06 t191_leaf0 <= 1.023256999999e+06
 T191_L_f20_k0: x_Gr_Liv_Area + 1e+06 t191_leaf0 <= 1.000746999999e+06
 T191_L_f66_k0: const_MS_Zoning_RL + 1e+06 t191_leaf0
   <= 1.000000999999e+06
 T191_L_f1_k1: const_Lot_Area + 1e+06 t191_leaf1 <= 1.023256999999e+06
 T191_L_f20_k1: x_Gr_Liv_Area + 1e+06 t191_leaf1 <= 1.000746999999e+06
 T191_R_f66_k1: const_MS_Zoning_RL - 1e+06 t191_leaf1 >= -999999
 T191_L_f1_k2: const_Lot_Area + 1e+06 t191_leaf2 <= 1.023256999999e+06
 T191_R_f20_k2: x_Gr_Liv_Area - 1e+06 t191_leaf2 >= -999253
 T191_L_f20_k2: x_Gr_Liv_Area + 1e+06 t191_leaf2 <= 1.003085999999e+06
 T191_L_f1_k3: const_Lot_Area + 1e+06 t191_leaf3 <= 1.023256999999e+06
 T191_R_f20_k3: x_Gr_Liv_Area - 1e+06 t191_leaf3 >= -999253
 T191_R_f20_k3: x_Gr_Liv_Area - 1e+06 t191_leaf3 >= -996914
 T191_R_f1_k4: const_Lot_Area - 1e+06 t191_leaf4 >= -976743
 T191_L_f297_k4: const_Sale_Condition_Normal + 1e+06 t191_leaf4
   <= 1.000000999999e+06
 T191_R_f1_k5: const_Lot_Area - 1e+06 t191_leaf5 >= -976743
 T191_R_f297_k5: const_Sale_Condition_Normal - 1e+06 t191_leaf5 >= -999999
 T191_L_f36_k5: x_Wood_Deck_SF + 1e+06 t191_leaf5 <= 1.000021999999e+06
 T191_R_f1_k6: const_Lot_Area - 1e+06 t191_leaf6 >= -976743
 T191_R_f297_k6: const_Sale_Condition_Normal - 1e+06 t191_leaf6 >= -999999
 T191_R_f36_k6: x_Wood_Deck_SF - 1e+06 t191_leaf6 >= -999978
 T192_onehot: t192_leaf0 + t192_leaf1 + t192_leaf2 + t192_leaf3
   + t192_leaf4 = 1
 T192_L_f211_k0: x_b1_is_ALQ + 1e+06 t192_leaf0 <= 1.000000999999e+06
 T192_L_f82_k0: const_Lot_Config_CulDSac + 1e+06 t192_leaf0
   <= 1.000000999999e+06
 T192_L_f36_k0: x_Wood_Deck_SF + 1e+06 t192_leaf0 <= 1.000055999999e+06
 T192_L_f211_k1: x_b1_is_ALQ + 1e+06 t192_leaf1 <= 1.000000999999e+06
 T192_L_f82_k1: const_Lot_Config_CulDSac + 1e+06 t192_leaf1
   <= 1.000000999999e+06
 T192_R_f36_k1: x_Wood_Deck_SF - 1e+06 t192_leaf1 >= -999944
 T192_L_f211_k2: x_b1_is_ALQ + 1e+06 t192_leaf2 <= 1.000000999999e+06
 T192_R_f82_k2: const_Lot_Config_CulDSac - 1e+06 t192_leaf2 >= -999999
 T192_R_f211_k3: x_b1_is_ALQ - 1e+06 t192_leaf3 >= -999999
 T192_L_f44_k3: const_Yr_Sold + 1e+06 t192_leaf3 <= 1.002008999999e+06
 T192_R_f211_k4: x_b1_is_ALQ - 1e+06 t192_leaf4 >= -999999
 T192_R_f44_k4: const_Yr_Sold - 1e+06 t192_leaf4 >= -997991
 T193_onehot: t193_leaf0 + t193_leaf1 + t193_leaf2 + t193_leaf3
   + t193_leaf4 + t193_leaf5 + t193_leaf6 + t193_leaf7 = 1
 T193_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t193_leaf0 <= 1.000064999999e+06
 T193_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t193_leaf0
   <= 1.000000999999e+06
 T193_L_f33_k0: x_Garage_Area + 1e+06 t193_leaf0 <= 1.000835999999e+06
 T193_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t193_leaf1 <= 1.000064999999e+06
 T193_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t193_leaf1
   <= 1.000000999999e+06
 T193_R_f33_k1: x_Garage_Area - 1e+06 t193_leaf1 >= -999164
 T193_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t193_leaf2 <= 1.000064999999e+06
 T193_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t193_leaf2
   >= -999999
 T193_L_f33_k2: x_Garage_Area + 1e+06 t193_leaf2 <= 1.000450999999e+06
 T193_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t193_leaf3 <= 1.000064999999e+06
 T193_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t193_leaf3
   >= -999999
 T193_R_f33_k3: x_Garage_Area - 1e+06 t193_leaf3 >= -999549
 T193_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t193_leaf4 >= -999935
 T193_L_f214_k4: x_b1_is_LwQ + 1e+06 t193_leaf4 <= 1.000000999999e+06
 T193_L_f5_k4: const_Year_Built + 1e+06 t193_leaf4 <= 1.001918999999e+06
 T193_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t193_leaf5 >= -999935
 T193_L_f214_k5: x_b1_is_LwQ + 1e+06 t193_leaf5 <= 1.000000999999e+06
 T193_R_f5_k5: const_Year_Built - 1e+06 t193_leaf5 >= -998081
 T193_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t193_leaf6 >= -999935
 T193_R_f214_k6: x_b1_is_LwQ - 1e+06 t193_leaf6 >= -999999
 T193_L_f209_k6: const_Bsmt_Exposure_No + 1e+06 t193_leaf6
   <= 1.000000999999e+06
 T193_R_f12_k7: x_BsmtFin_SF_1 - 1e+06 t193_leaf7 >= -999935
 T193_R_f214_k7: x_b1_is_LwQ - 1e+06 t193_leaf7 >= -999999
 T193_R_f209_k7: const_Bsmt_Exposure_No - 1e+06 t193_leaf7 >= -999999
 T194_onehot: t194_leaf0 + t194_leaf1 + t194_leaf2 + t194_leaf3 = 1
 T194_L_f20_k0: x_Gr_Liv_Area + 1e+06 t194_leaf0 <= 1.003085999999e+06
 T194_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t194_leaf0
   <= 1.000001999999e+06
 T194_L_f4_k0: const_Overall_Cond + 1e+06 t194_leaf0 <= 1.000003999999e+06
 T194_L_f20_k1: x_Gr_Liv_Area + 1e+06 t194_leaf1 <= 1.003085999999e+06
 T194_L_f21_k1: const_Bsmt_Full_Bath + 1e+06 t194_leaf1
   <= 1.000001999999e+06
 T194_R_f4_k1: const_Overall_Cond - 1e+06 t194_leaf1 >= -999996
 T194_L_f20_k2: x_Gr_Liv_Area + 1e+06 t194_leaf2 <= 1.003085999999e+06
 T194_R_f21_k2: const_Bsmt_Full_Bath - 1e+06 t194_leaf2 >= -999998
 T194_R_f20_k3: x_Gr_Liv_Area - 1e+06 t194_leaf3 >= -996914
 T195_onehot: t195_leaf0 + t195_leaf1 + t195_leaf2 + t195_leaf3 = 1
 T195_L_f20_k0: x_Gr_Liv_Area + 1e+06 t195_leaf0 <= 1.003085999999e+06
 T195_L_f0_k0: const_Lot_Frontage + 1e+06 t195_leaf0 <= 1.000027999999e+06
 T195_L_f20_k1: x_Gr_Liv_Area + 1e+06 t195_leaf1 <= 1.003085999999e+06
 T195_R_f0_k1: const_Lot_Frontage - 1e+06 t195_leaf1 >= -999972
 T195_L_f3_k1: Overall_Qual_calc + 1e+06 t195_leaf1 <= 1.000006999999e+06
 T195_L_f20_k2: x_Gr_Liv_Area + 1e+06 t195_leaf2 <= 1.003085999999e+06
 T195_R_f0_k2: const_Lot_Frontage - 1e+06 t195_leaf2 >= -999972
 T195_R_f3_k2: Overall_Qual_calc - 1e+06 t195_leaf2 >= -999993
 T195_R_f20_k3: x_Gr_Liv_Area - 1e+06 t195_leaf3 >= -996914
 T196_onehot: t196_leaf0 + t196_leaf1 + t196_leaf2 + t196_leaf3
   + t196_leaf4 + t196_leaf5 = 1
 T196_L_f118_k0: const_Condition_1_Feedr + 1e+06 t196_leaf0
   <= 1.000000999999e+06
 T196_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t196_leaf0 <= 1.000000999999e+06
 T196_L_f27_k0: x_Kitchen_Qual + 1e+06 t196_leaf0 <= 1.000003999999e+06
 T196_L_f118_k1: const_Condition_1_Feedr + 1e+06 t196_leaf1
   <= 1.000000999999e+06
 T196_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t196_leaf1 <= 1.000000999999e+06
 T196_R_f27_k1: x_Kitchen_Qual - 1e+06 t196_leaf1 >= -999996
 T196_L_f118_k2: const_Condition_1_Feedr + 1e+06 t196_leaf2
   <= 1.000000999999e+06
 T196_R_f164_k2: x_ex1_is_BrkFace - 1e+06 t196_leaf2 >= -999999
 T196_L_f106_k2: const_Neighborhood_No_aplicames + 1e+06 t196_leaf2
   <= 1.000000999999e+06
 T196_L_f118_k3: const_Condition_1_Feedr + 1e+06 t196_leaf3
   <= 1.000000999999e+06
 T196_R_f164_k3: x_ex1_is_BrkFace - 1e+06 t196_leaf3 >= -999999
 T196_R_f106_k3: const_Neighborhood_No_aplicames - 1e+06 t196_leaf3
   >= -999999
 T196_R_f118_k4: const_Condition_1_Feedr - 1e+06 t196_leaf4 >= -999999
 T196_L_f44_k4: const_Yr_Sold + 1e+06 t196_leaf4 <= 1.002007999999e+06
 T196_R_f118_k5: const_Condition_1_Feedr - 1e+06 t196_leaf5 >= -999999
 T196_R_f44_k5: const_Yr_Sold - 1e+06 t196_leaf5 >= -997992
 T197_onehot: t197_leaf0 + t197_leaf1 + t197_leaf2 + t197_leaf3
   + t197_leaf4 + t197_leaf5 = 1
 T197_L_f37_k0: x_Open_Porch_SF + 1e+06 t197_leaf0 <= 1.000281999999e+06
 T197_L_f1_k0: const_Lot_Area + 1e+06 t197_leaf0 <= 1.017226999999e+06
 T197_L_f3_k0: Overall_Qual_calc + 1e+06 t197_leaf0 <= 1.000008999999e+06
 T197_L_f37_k1: x_Open_Porch_SF + 1e+06 t197_leaf1 <= 1.000281999999e+06
 T197_L_f1_k1: const_Lot_Area + 1e+06 t197_leaf1 <= 1.017226999999e+06
 T197_R_f3_k1: Overall_Qual_calc - 1e+06 t197_leaf1 >= -999991
 T197_L_f37_k2: x_Open_Porch_SF + 1e+06 t197_leaf2 <= 1.000281999999e+06
 T197_R_f1_k2: const_Lot_Area - 1e+06 t197_leaf2 >= -982773
 T197_L_f283_k2: const_Sale_Type_COD + 1e+06 t197_leaf2
   <= 1.000000999999e+06
 T197_L_f37_k3: x_Open_Porch_SF + 1e+06 t197_leaf3 <= 1.000281999999e+06
 T197_R_f1_k3: const_Lot_Area - 1e+06 t197_leaf3 >= -982773
 T197_R_f283_k3: const_Sale_Type_COD - 1e+06 t197_leaf3 >= -999999
 T197_R_f37_k4: x_Open_Porch_SF - 1e+06 t197_leaf4 >= -999718
 T197_L_f297_k4: const_Sale_Condition_Normal + 1e+06 t197_leaf4
   <= 1.000000999999e+06
 T197_R_f37_k5: x_Open_Porch_SF - 1e+06 t197_leaf5 >= -999718
 T197_R_f297_k5: const_Sale_Condition_Normal - 1e+06 t197_leaf5 >= -999999
 T198_onehot: t198_leaf0 + t198_leaf1 + t198_leaf2 + t198_leaf3
   + t198_leaf4 + t198_leaf5 = 1
 T198_L_f3_k0: Overall_Qual_calc + 1e+06 t198_leaf0 <= 1.000002999999e+06
 T198_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t198_leaf0
   <= 1.001920999999e+06
 T198_L_f3_k1: Overall_Qual_calc + 1e+06 t198_leaf1 <= 1.000002999999e+06
 T198_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t198_leaf1 >= -998079
 T198_R_f3_k2: Overall_Qual_calc - 1e+06 t198_leaf2 >= -999997
 T198_L_f298_k2: const_Sale_Condition_Partial + 1e+06 t198_leaf2
   <= 1.000000999999e+06
 T198_L_f3_k2: Overall_Qual_calc + 1e+06 t198_leaf2 <= 1.000006999999e+06
 T198_R_f3_k3: Overall_Qual_calc - 1e+06 t198_leaf3 >= -999997
 T198_L_f298_k3: const_Sale_Condition_Partial + 1e+06 t198_leaf3
   <= 1.000000999999e+06
 T198_R_f3_k3: Overall_Qual_calc - 1e+06 t198_leaf3 >= -999993
 T198_R_f3_k4: Overall_Qual_calc - 1e+06 t198_leaf4 >= -999997
 T198_R_f298_k4: const_Sale_Condition_Partial - 1e+06 t198_leaf4
   >= -999999
 T198_L_f37_k4: x_Open_Porch_SF + 1e+06 t198_leaf4 <= 1.000243999999e+06
 T198_R_f3_k5: Overall_Qual_calc - 1e+06 t198_leaf5 >= -999997
 T198_R_f298_k5: const_Sale_Condition_Partial - 1e+06 t198_leaf5
   >= -999999
 T198_R_f37_k5: x_Open_Porch_SF - 1e+06 t198_leaf5 >= -999756
 T199_onehot: t199_leaf0 + t199_leaf1 + t199_leaf2 + t199_leaf3
   + t199_leaf4 = 1
 T199_L_f40_k0: x_Screen_Porch + 1e+06 t199_leaf0 <= 1.000149999999e+06
 T199_L_f5_k0: const_Year_Built + 1e+06 t199_leaf0 <= 1.001910999999e+06
 T199_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t199_leaf0 <= 1.000000999999e+06
 T199_L_f40_k1: x_Screen_Porch + 1e+06 t199_leaf1 <= 1.000149999999e+06
 T199_L_f5_k1: const_Year_Built + 1e+06 t199_leaf1 <= 1.001910999999e+06
 T199_R_f161_k1: x_ex1_is_AsbShng - 1e+06 t199_leaf1 >= -999999
 T199_L_f40_k2: x_Screen_Porch + 1e+06 t199_leaf2 <= 1.000149999999e+06
 T199_R_f5_k2: const_Year_Built - 1e+06 t199_leaf2 >= -998089
 T199_L_f18_k2: const_2nd_Flr_SF + 1e+06 t199_leaf2 <= 1.000583999999e+06
 T199_L_f40_k3: x_Screen_Porch + 1e+06 t199_leaf3 <= 1.000149999999e+06
 T199_R_f5_k3: const_Year_Built - 1e+06 t199_leaf3 >= -998089
 T199_R_f18_k3: const_2nd_Flr_SF - 1e+06 t199_leaf3 >= -999416
 T199_R_f40_k4: x_Screen_Porch - 1e+06 t199_leaf4 >= -999850
 T200_onehot: t200_leaf0 + t200_leaf1 + t200_leaf2 + t200_leaf3
   + t200_leaf4 = 1
 T200_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t200_leaf0 <= 1.000000999999e+06
 T200_L_f216_k0: x_b1_is_Rec + 1e+06 t200_leaf0 <= 1.000000999999e+06
 T200_L_f119_k0: const_Condition_1_Norm + 1e+06 t200_leaf0
   <= 1.000000999999e+06
 T200_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t200_leaf1 <= 1.000000999999e+06
 T200_L_f216_k1: x_b1_is_Rec + 1e+06 t200_leaf1 <= 1.000000999999e+06
 T200_R_f119_k1: const_Condition_1_Norm - 1e+06 t200_leaf1 >= -999999
 T200_L_f164_k2: x_ex1_is_BrkFace + 1e+06 t200_leaf2 <= 1.000000999999e+06
 T200_R_f216_k2: x_b1_is_Rec - 1e+06 t200_leaf2 >= -999999
 T200_R_f164_k3: x_ex1_is_BrkFace - 1e+06 t200_leaf3 >= -999999
 T200_L_f245_k3: const_FunctioNo_aplical_Typ + 1e+06 t200_leaf3
   <= 1.000000999999e+06
 T200_R_f164_k4: x_ex1_is_BrkFace - 1e+06 t200_leaf4 >= -999999
 T200_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t200_leaf4 >= -999999
 T201_onehot: t201_leaf0 + t201_leaf1 + t201_leaf2 + t201_leaf3
   + t201_leaf4 = 1
 T201_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t201_leaf0
   <= 1.000001999999e+06
 T201_L_f239_k0: const_FunctioNo_aplical_Maj2 + 1e+06 t201_leaf0
   <= 1.000000999999e+06
 T201_L_f233_k0: x_elect_is_FuseA + 1e+06 t201_leaf0 <= 1.000000999999e+06
 T201_L_f21_k1: const_Bsmt_Full_Bath + 1e+06 t201_leaf1
   <= 1.000001999999e+06
 T201_L_f239_k1: const_FunctioNo_aplical_Maj2 + 1e+06 t201_leaf1
   <= 1.000000999999e+06
 T201_R_f233_k1: x_elect_is_FuseA - 1e+06 t201_leaf1 >= -999999
 T201_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t201_leaf2
   <= 1.000001999999e+06
 T201_R_f239_k2: const_FunctioNo_aplical_Maj2 - 1e+06 t201_leaf2
   >= -999999
 T201_R_f21_k3: const_Bsmt_Full_Bath - 1e+06 t201_leaf3 >= -999998
 T201_L_f4_k3: const_Overall_Cond + 1e+06 t201_leaf3 <= 1.000006999999e+06
 T201_R_f21_k4: const_Bsmt_Full_Bath - 1e+06 t201_leaf4 >= -999998
 T201_R_f4_k4: const_Overall_Cond - 1e+06 t201_leaf4 >= -999993
 T202_onehot: t202_leaf0 + t202_leaf1 + t202_leaf2 + t202_leaf3
   + t202_leaf4 + t202_leaf5 + t202_leaf6 = 1
 T202_L_f119_k0: const_Condition_1_Norm + 1e+06 t202_leaf0
   <= 1.000000999999e+06
 T202_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t202_leaf0 <= 1.000995999999e+06
 T202_L_f32_k0: const_Garage_Cars + 1e+06 t202_leaf0 <= 1.000002999999e+06
 T202_L_f119_k1: const_Condition_1_Norm + 1e+06 t202_leaf1
   <= 1.000000999999e+06
 T202_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t202_leaf1 <= 1.000995999999e+06
 T202_R_f32_k1: const_Garage_Cars - 1e+06 t202_leaf1 >= -999997
 T202_L_f119_k2: const_Condition_1_Norm + 1e+06 t202_leaf2
   <= 1.000000999999e+06
 T202_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t202_leaf2 >= -999004
 T202_R_f119_k3: const_Condition_1_Norm - 1e+06 t202_leaf3 >= -999999
 T202_L_f44_k3: const_Yr_Sold + 1e+06 t202_leaf3 <= 1.002008999999e+06
 T202_L_f4_k3: const_Overall_Cond + 1e+06 t202_leaf3 <= 1.000006999999e+06
 T202_R_f119_k4: const_Condition_1_Norm - 1e+06 t202_leaf4 >= -999999
 T202_L_f44_k4: const_Yr_Sold + 1e+06 t202_leaf4 <= 1.002008999999e+06
 T202_R_f4_k4: const_Overall_Cond - 1e+06 t202_leaf4 >= -999993
 T202_R_f119_k5: const_Condition_1_Norm - 1e+06 t202_leaf5 >= -999999
 T202_R_f44_k5: const_Yr_Sold - 1e+06 t202_leaf5 >= -997991
 T202_L_f231_k5: central_air_no + 1e+06 t202_leaf5 <= 1.000000999999e+06
 T202_R_f119_k6: const_Condition_1_Norm - 1e+06 t202_leaf6 >= -999999
 T202_R_f44_k6: const_Yr_Sold - 1e+06 t202_leaf6 >= -997991
 T202_R_f231_k6: central_air_no - 1e+06 t202_leaf6 >= -999999
 T203_onehot: t203_leaf0 + t203_leaf1 + t203_leaf2 + t203_leaf3
   + t203_leaf4 + t203_leaf5 + t203_leaf6 + t203_leaf7 = 1
 T203_L_f44_k0: const_Yr_Sold + 1e+06 t203_leaf0 <= 1.002008999999e+06
 T203_L_f29_k0: const_Fireplaces + 1e+06 t203_leaf0 <= 1.000001999999e+06
 T203_L_f289_k0: const_Sale_Type_New + 1e+06 t203_leaf0
   <= 1.000000999999e+06
 T203_L_f44_k1: const_Yr_Sold + 1e+06 t203_leaf1 <= 1.002008999999e+06
 T203_L_f29_k1: const_Fireplaces + 1e+06 t203_leaf1 <= 1.000001999999e+06
 T203_R_f289_k1: const_Sale_Type_New - 1e+06 t203_leaf1 >= -999999
 T203_L_f44_k2: const_Yr_Sold + 1e+06 t203_leaf2 <= 1.002008999999e+06
 T203_R_f29_k2: const_Fireplaces - 1e+06 t203_leaf2 >= -999998
 T203_L_f4_k2: const_Overall_Cond + 1e+06 t203_leaf2 <= 1.000005999999e+06
 T203_L_f44_k3: const_Yr_Sold + 1e+06 t203_leaf3 <= 1.002008999999e+06
 T203_R_f29_k3: const_Fireplaces - 1e+06 t203_leaf3 >= -999998
 T203_R_f4_k3: const_Overall_Cond - 1e+06 t203_leaf3 >= -999994
 T203_R_f44_k4: const_Yr_Sold - 1e+06 t203_leaf4 >= -997991
 T203_L_f293_k4: const_Sale_Condition_Abnorml + 1e+06 t203_leaf4
   <= 1.000000999999e+06
 T203_L_f148_k4: const_Roof_Style_Gable + 1e+06 t203_leaf4
   <= 1.000000999999e+06
 T203_R_f44_k5: const_Yr_Sold - 1e+06 t203_leaf5 >= -997991
 T203_L_f293_k5: const_Sale_Condition_Abnorml + 1e+06 t203_leaf5
   <= 1.000000999999e+06
 T203_R_f148_k5: const_Roof_Style_Gable - 1e+06 t203_leaf5 >= -999999
 T203_R_f44_k6: const_Yr_Sold - 1e+06 t203_leaf6 >= -997991
 T203_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t203_leaf6
   >= -999999
 T203_L_f148_k6: const_Roof_Style_Gable + 1e+06 t203_leaf6
   <= 1.000000999999e+06
 T203_R_f44_k7: const_Yr_Sold - 1e+06 t203_leaf7 >= -997991
 T203_R_f293_k7: const_Sale_Condition_Abnorml - 1e+06 t203_leaf7
   >= -999999
 T203_R_f148_k7: const_Roof_Style_Gable - 1e+06 t203_leaf7 >= -999999
 T204_onehot: t204_leaf0 + t204_leaf1 + t204_leaf2 + t204_leaf3
   + t204_leaf4 = 1
 T204_L_f18_k0: const_2nd_Flr_SF + 1e+06 t204_leaf0 <= 1.001295999999e+06
 T204_L_f9_k0: x_Exter_Cond + 1e+06 t204_leaf0 <= 1.000001999999e+06
 T204_L_f20_k0: x_Gr_Liv_Area + 1e+06 t204_leaf0 <= 1.001016999999e+06
 T204_L_f18_k1: const_2nd_Flr_SF + 1e+06 t204_leaf1 <= 1.001295999999e+06
 T204_L_f9_k1: x_Exter_Cond + 1e+06 t204_leaf1 <= 1.000001999999e+06
 T204_R_f20_k1: x_Gr_Liv_Area - 1e+06 t204_leaf1 >= -998983
 T204_L_f18_k2: const_2nd_Flr_SF + 1e+06 t204_leaf2 <= 1.001295999999e+06
 T204_R_f9_k2: x_Exter_Cond - 1e+06 t204_leaf2 >= -999998
 T204_L_f209_k2: const_Bsmt_Exposure_No + 1e+06 t204_leaf2
   <= 1.000000999999e+06
 T204_L_f18_k3: const_2nd_Flr_SF + 1e+06 t204_leaf3 <= 1.001295999999e+06
 T204_R_f9_k3: x_Exter_Cond - 1e+06 t204_leaf3 >= -999998
 T204_R_f209_k3: const_Bsmt_Exposure_No - 1e+06 t204_leaf3 >= -999999
 T204_R_f18_k4: const_2nd_Flr_SF - 1e+06 t204_leaf4 >= -998704
 T205_onehot: t205_leaf0 + t205_leaf1 + t205_leaf2 + t205_leaf3 = 1
 T205_L_f113_k0: const_Neighborhood_Somerst + 1e+06 t205_leaf0
   <= 1.000000999999e+06
 T205_L_f1_k0: const_Lot_Area + 1e+06 t205_leaf0 <= 1.004499999999e+06
 T205_L_f113_k1: const_Neighborhood_Somerst + 1e+06 t205_leaf1
   <= 1.000000999999e+06
 T205_R_f1_k1: const_Lot_Area - 1e+06 t205_leaf1 >= -995500
 T205_L_f40_k1: x_Screen_Porch + 1e+06 t205_leaf1 <= 1.000142999999e+06
 T205_L_f113_k2: const_Neighborhood_Somerst + 1e+06 t205_leaf2
   <= 1.000000999999e+06
 T205_R_f1_k2: const_Lot_Area - 1e+06 t205_leaf2 >= -995500
 T205_R_f40_k2: x_Screen_Porch - 1e+06 t205_leaf2 >= -999857
 T205_R_f113_k3: const_Neighborhood_Somerst - 1e+06 t205_leaf3 >= -999999
 T206_onehot: t206_leaf0 + t206_leaf1 + t206_leaf2 + t206_leaf3 = 1
 T206_L_f20_k0: x_Gr_Liv_Area + 1e+06 t206_leaf0 <= 1.003085999999e+06
 T206_L_f6_k0: const_Year_Remod_Add + 1e+06 t206_leaf0
   <= 1.002007999999e+06
 T206_L_f4_k0: const_Overall_Cond + 1e+06 t206_leaf0 <= 1.000002999999e+06
 T206_L_f20_k1: x_Gr_Liv_Area + 1e+06 t206_leaf1 <= 1.003085999999e+06
 T206_L_f6_k1: const_Year_Remod_Add + 1e+06 t206_leaf1
   <= 1.002007999999e+06
 T206_R_f4_k1: const_Overall_Cond - 1e+06 t206_leaf1 >= -999997
 T206_L_f20_k2: x_Gr_Liv_Area + 1e+06 t206_leaf2 <= 1.003085999999e+06
 T206_R_f6_k2: const_Year_Remod_Add - 1e+06 t206_leaf2 >= -997992
 T206_R_f20_k3: x_Gr_Liv_Area - 1e+06 t206_leaf3 >= -996914
 T207_onehot: t207_leaf0 + t207_leaf1 + t207_leaf2 + t207_leaf3
   + t207_leaf4 + t207_leaf5 = 1
 T207_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t207_leaf0
   <= 1.000000999999e+06
 T207_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t207_leaf0 <= 1.000000999999e+06
 T207_L_f77_k0: const_Land_Contour_Bnk + 1e+06 t207_leaf0
   <= 1.000000999999e+06
 T207_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t207_leaf1
   <= 1.000000999999e+06
 T207_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t207_leaf1 <= 1.000000999999e+06
 T207_R_f77_k1: const_Land_Contour_Bnk - 1e+06 t207_leaf1 >= -999999
 T207_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t207_leaf2
   <= 1.000000999999e+06
 T207_R_f164_k2: x_ex1_is_BrkFace - 1e+06 t207_leaf2 >= -999999
 T207_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t207_leaf3
   >= -999999
 T207_L_f3_k3: Overall_Qual_calc + 1e+06 t207_leaf3 <= 1.000003999999e+06
 T207_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t207_leaf4
   >= -999999
 T207_R_f3_k4: Overall_Qual_calc - 1e+06 t207_leaf4 >= -999996
 T207_L_f14_k4: x_Bsmt_Unf_SF + 1e+06 t207_leaf4 <= 1.000671999999e+06
 T207_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t207_leaf5
   >= -999999
 T207_R_f3_k5: Overall_Qual_calc - 1e+06 t207_leaf5 >= -999996
 T207_R_f14_k5: x_Bsmt_Unf_SF - 1e+06 t207_leaf5 >= -999328
 T208_onehot: t208_leaf0 + t208_leaf1 + t208_leaf2 + t208_leaf3
   + t208_leaf4 = 1
 T208_L_f37_k0: x_Open_Porch_SF + 1e+06 t208_leaf0 <= 1.000483999999e+06
 T208_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t208_leaf0 <= 1.000472999999e+06
 T208_L_f211_k0: x_b1_is_ALQ + 1e+06 t208_leaf0 <= 1.000000999999e+06
 T208_L_f37_k1: x_Open_Porch_SF + 1e+06 t208_leaf1 <= 1.000483999999e+06
 T208_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t208_leaf1 <= 1.000472999999e+06
 T208_R_f211_k1: x_b1_is_ALQ - 1e+06 t208_leaf1 >= -999999
 T208_L_f37_k2: x_Open_Porch_SF + 1e+06 t208_leaf2 <= 1.000483999999e+06
 T208_R_f7_k2: x_Mas_Vnr_Area - 1e+06 t208_leaf2 >= -999527
 T208_L_f18_k2: const_2nd_Flr_SF + 1e+06 t208_leaf2 <= 1.001305999999e+06
 T208_L_f37_k3: x_Open_Porch_SF + 1e+06 t208_leaf3 <= 1.000483999999e+06
 T208_R_f7_k3: x_Mas_Vnr_Area - 1e+06 t208_leaf3 >= -999527
 T208_R_f18_k3: const_2nd_Flr_SF - 1e+06 t208_leaf3 >= -998694
 T208_R_f37_k4: x_Open_Porch_SF - 1e+06 t208_leaf4 >= -999516
 T209_onehot: t209_leaf0 + t209_leaf1 + t209_leaf2 + t209_leaf3
   + t209_leaf4 = 1
 T209_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t209_leaf0
   <= 1.002007999999e+06
 T209_L_f1_k0: const_Lot_Area + 1e+06 t209_leaf0 <= 1.019521999999e+06
 T209_L_f117_k0: const_Condition_1_Artery + 1e+06 t209_leaf0
   <= 1.000000999999e+06
 T209_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t209_leaf1
   <= 1.002007999999e+06
 T209_L_f1_k1: const_Lot_Area + 1e+06 t209_leaf1 <= 1.019521999999e+06
 T209_R_f117_k1: const_Condition_1_Artery - 1e+06 t209_leaf1 >= -999999
 T209_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t209_leaf2
   <= 1.002007999999e+06
 T209_R_f1_k2: const_Lot_Area - 1e+06 t209_leaf2 >= -980478
 T209_L_f0_k2: const_Lot_Frontage + 1e+06 t209_leaf2 <= 1.000080499999e+06
 T209_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t209_leaf3
   <= 1.002007999999e+06
 T209_R_f1_k3: const_Lot_Area - 1e+06 t209_leaf3 >= -980478
 T209_R_f0_k3: const_Lot_Frontage - 1e+06 t209_leaf3 >= -999919.5
 T209_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t209_leaf4 >= -997992
 T210_onehot: t210_leaf0 + t210_leaf1 + t210_leaf2 + t210_leaf3
   + t210_leaf4 = 1
 T210_L_f117_k0: const_Condition_1_Artery + 1e+06 t210_leaf0
   <= 1.000000999999e+06
 T210_L_f37_k0: x_Open_Porch_SF + 1e+06 t210_leaf0 <= 1.000417999999e+06
 T210_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t210_leaf0 <= 1.001964999999e+06
 T210_L_f117_k1: const_Condition_1_Artery + 1e+06 t210_leaf1
   <= 1.000000999999e+06
 T210_L_f37_k1: x_Open_Porch_SF + 1e+06 t210_leaf1 <= 1.000417999999e+06
 T210_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t210_leaf1 >= -998035
 T210_L_f117_k2: const_Condition_1_Artery + 1e+06 t210_leaf2
   <= 1.000000999999e+06
 T210_R_f37_k2: x_Open_Porch_SF - 1e+06 t210_leaf2 >= -999582
 T210_R_f117_k3: const_Condition_1_Artery - 1e+06 t210_leaf3 >= -999999
 T210_L_f3_k3: Overall_Qual_calc + 1e+06 t210_leaf3 <= 1.000003999999e+06
 T210_R_f117_k4: const_Condition_1_Artery - 1e+06 t210_leaf4 >= -999999
 T210_R_f3_k4: Overall_Qual_calc - 1e+06 t210_leaf4 >= -999996
 T211_onehot: t211_leaf0 + t211_leaf1 + t211_leaf2 + t211_leaf3
   + t211_leaf4 + t211_leaf5 + t211_leaf6 = 1
 T211_L_f44_k0: const_Yr_Sold + 1e+06 t211_leaf0 <= 1.002006999999e+06
 T211_L_f36_k0: x_Wood_Deck_SF + 1e+06 t211_leaf0 <= 1.000149999999e+06
 T211_L_f33_k0: x_Garage_Area + 1e+06 t211_leaf0 <= 1.000378999999e+06
 T211_L_f44_k1: const_Yr_Sold + 1e+06 t211_leaf1 <= 1.002006999999e+06
 T211_L_f36_k1: x_Wood_Deck_SF + 1e+06 t211_leaf1 <= 1.000149999999e+06
 T211_R_f33_k1: x_Garage_Area - 1e+06 t211_leaf1 >= -999621
 T211_L_f44_k2: const_Yr_Sold + 1e+06 t211_leaf2 <= 1.002006999999e+06
 T211_R_f36_k2: x_Wood_Deck_SF - 1e+06 t211_leaf2 >= -999850
 T211_R_f44_k3: const_Yr_Sold - 1e+06 t211_leaf3 >= -997993
 T211_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t211_leaf3 <= 1.000000999999e+06
 T211_L_f14_k3: x_Bsmt_Unf_SF + 1e+06 t211_leaf3 <= 1.000475999999e+06
 T211_R_f44_k4: const_Yr_Sold - 1e+06 t211_leaf4 >= -997993
 T211_L_f161_k4: x_ex1_is_AsbShng + 1e+06 t211_leaf4 <= 1.000000999999e+06
 T211_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t211_leaf4 >= -999524
 T211_R_f44_k5: const_Yr_Sold - 1e+06 t211_leaf5 >= -997993
 T211_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t211_leaf5 >= -999999
 T211_L_f33_k5: x_Garage_Area + 1e+06 t211_leaf5 <= 1.000365999999e+06
 T211_R_f44_k6: const_Yr_Sold - 1e+06 t211_leaf6 >= -997993
 T211_R_f161_k6: x_ex1_is_AsbShng - 1e+06 t211_leaf6 >= -999999
 T211_R_f33_k6: x_Garage_Area - 1e+06 t211_leaf6 >= -999634
 T212_onehot: t212_leaf0 + t212_leaf1 + t212_leaf2 + t212_leaf3
   + t212_leaf4 + t212_leaf5 = 1
 T212_L_f9_k0: x_Exter_Cond + 1e+06 t212_leaf0 <= 1.000001999999e+06
 T212_L_f0_k0: const_Lot_Frontage + 1e+06 t212_leaf0 <= 1.000065999999e+06
 T212_L_f9_k1: x_Exter_Cond + 1e+06 t212_leaf1 <= 1.000001999999e+06
 T212_R_f0_k1: const_Lot_Frontage - 1e+06 t212_leaf1 >= -999934
 T212_L_f37_k1: x_Open_Porch_SF + 1e+06 t212_leaf1 <= 1.000003999999e+06
 T212_L_f9_k2: x_Exter_Cond + 1e+06 t212_leaf2 <= 1.000001999999e+06
 T212_R_f0_k2: const_Lot_Frontage - 1e+06 t212_leaf2 >= -999934
 T212_R_f37_k2: x_Open_Porch_SF - 1e+06 t212_leaf2 >= -999996
 T212_R_f9_k3: x_Exter_Cond - 1e+06 t212_leaf3 >= -999998
 T212_L_f1_k3: const_Lot_Area + 1e+06 t212_leaf3 <= 1.003674999999e+06
 T212_R_f9_k4: x_Exter_Cond - 1e+06 t212_leaf4 >= -999998
 T212_R_f1_k4: const_Lot_Area - 1e+06 t212_leaf4 >= -996325
 T212_L_f31_k4: const_Garage_Yr_Blt + 1e+06 t212_leaf4
   <= 1.001940999999e+06
 T212_R_f9_k5: x_Exter_Cond - 1e+06 t212_leaf5 >= -999998
 T212_R_f1_k5: const_Lot_Area - 1e+06 t212_leaf5 >= -996325
 T212_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t212_leaf5 >= -998059
 T213_onehot: t213_leaf0 + t213_leaf1 + t213_leaf2 + t213_leaf3 = 1
 T213_L_f38_k0: x_Enclosed_Porch + 1e+06 t213_leaf0 <= 1.000264999999e+06
 T213_L_f37_k0: x_Open_Porch_SF + 1e+06 t213_leaf0 <= 1.000190999999e+06
 T213_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t213_leaf0
   <= 1.000000999999e+06
 T213_L_f38_k1: x_Enclosed_Porch + 1e+06 t213_leaf1 <= 1.000264999999e+06
 T213_L_f37_k1: x_Open_Porch_SF + 1e+06 t213_leaf1 <= 1.000190999999e+06
 T213_R_f245_k1: const_FunctioNo_aplical_Typ - 1e+06 t213_leaf1 >= -999999
 T213_L_f38_k2: x_Enclosed_Porch + 1e+06 t213_leaf2 <= 1.000264999999e+06
 T213_R_f37_k2: x_Open_Porch_SF - 1e+06 t213_leaf2 >= -999809
 T213_R_f38_k3: x_Enclosed_Porch - 1e+06 t213_leaf3 >= -999735
 T214_onehot: t214_leaf0 + t214_leaf1 + t214_leaf2 + t214_leaf3
   + t214_leaf4 + t214_leaf5 + t214_leaf6 = 1
 T214_L_f117_k0: const_Condition_1_Artery + 1e+06 t214_leaf0
   <= 1.000000999999e+06
 T214_L_f4_k0: const_Overall_Cond + 1e+06 t214_leaf0 <= 1.000006999999e+06
 T214_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t214_leaf0
   <= 1.002004999999e+06
 T214_L_f117_k1: const_Condition_1_Artery + 1e+06 t214_leaf1
   <= 1.000000999999e+06
 T214_L_f4_k1: const_Overall_Cond + 1e+06 t214_leaf1 <= 1.000006999999e+06
 T214_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t214_leaf1 >= -997995
 T214_L_f117_k2: const_Condition_1_Artery + 1e+06 t214_leaf2
   <= 1.000000999999e+06
 T214_R_f4_k2: const_Overall_Cond - 1e+06 t214_leaf2 >= -999993
 T214_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t214_leaf2
   <= 1.001907999999e+06
 T214_L_f117_k3: const_Condition_1_Artery + 1e+06 t214_leaf3
   <= 1.000000999999e+06
 T214_R_f4_k3: const_Overall_Cond - 1e+06 t214_leaf3 >= -999993
 T214_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t214_leaf3 >= -998092
 T214_R_f117_k4: const_Condition_1_Artery - 1e+06 t214_leaf4 >= -999999
 T214_L_f29_k4: const_Fireplaces + 1e+06 t214_leaf4 <= 1.000000999999e+06
 T214_L_f1_k4: const_Lot_Area + 1e+06 t214_leaf4 <= 1.010623999999e+06
 T214_R_f117_k5: const_Condition_1_Artery - 1e+06 t214_leaf5 >= -999999
 T214_L_f29_k5: const_Fireplaces + 1e+06 t214_leaf5 <= 1.000000999999e+06
 T214_R_f1_k5: const_Lot_Area - 1e+06 t214_leaf5 >= -989376
 T214_R_f117_k6: const_Condition_1_Artery - 1e+06 t214_leaf6 >= -999999
 T214_R_f29_k6: const_Fireplaces - 1e+06 t214_leaf6 >= -999999
 T215_onehot: t215_leaf0 + t215_leaf1 + t215_leaf2 + t215_leaf3
   + t215_leaf4 + t215_leaf5 = 1
 T215_L_f3_k0: Overall_Qual_calc + 1e+06 t215_leaf0 <= 1.000002999999e+06
 T215_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t215_leaf0
   <= 1.001922999999e+06
 T215_L_f3_k1: Overall_Qual_calc + 1e+06 t215_leaf1 <= 1.000002999999e+06
 T215_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t215_leaf1 >= -998077
 T215_R_f3_k2: Overall_Qual_calc - 1e+06 t215_leaf2 >= -999997
 T215_L_f119_k2: const_Condition_1_Norm + 1e+06 t215_leaf2
   <= 1.000000999999e+06
 T215_L_f0_k2: const_Lot_Frontage + 1e+06 t215_leaf2 <= 1.000120999999e+06
 T215_R_f3_k3: Overall_Qual_calc - 1e+06 t215_leaf3 >= -999997
 T215_L_f119_k3: const_Condition_1_Norm + 1e+06 t215_leaf3
   <= 1.000000999999e+06
 T215_R_f0_k3: const_Lot_Frontage - 1e+06 t215_leaf3 >= -999879
 T215_R_f3_k4: Overall_Qual_calc - 1e+06 t215_leaf4 >= -999997
 T215_R_f119_k4: const_Condition_1_Norm - 1e+06 t215_leaf4 >= -999999
 T215_L_f4_k4: const_Overall_Cond + 1e+06 t215_leaf4 <= 1.000005999999e+06
 T215_R_f3_k5: Overall_Qual_calc - 1e+06 t215_leaf5 >= -999997
 T215_R_f119_k5: const_Condition_1_Norm - 1e+06 t215_leaf5 >= -999999
 T215_R_f4_k5: const_Overall_Cond - 1e+06 t215_leaf5 >= -999994
 T216_onehot: t216_leaf0 + t216_leaf1 + t216_leaf2 + t216_leaf3
   + t216_leaf4 = 1
 T216_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t216_leaf0
   <= 1.002491999999e+06
 T216_L_f4_k0: const_Overall_Cond + 1e+06 t216_leaf0 <= 1.000005999999e+06
 T216_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t216_leaf0
   <= 1.002007999999e+06
 T216_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t216_leaf1
   <= 1.002491999999e+06
 T216_L_f4_k1: const_Overall_Cond + 1e+06 t216_leaf1 <= 1.000005999999e+06
 T216_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t216_leaf1 >= -997992
 T216_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t216_leaf2
   <= 1.002491999999e+06
 T216_R_f4_k2: const_Overall_Cond - 1e+06 t216_leaf2 >= -999994
 T216_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t216_leaf2
   <= 1.001907999999e+06
 T216_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t216_leaf3
   <= 1.002491999999e+06
 T216_R_f4_k3: const_Overall_Cond - 1e+06 t216_leaf3 >= -999994
 T216_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t216_leaf3 >= -998092
 T216_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t216_leaf4 >= -997508
 T217_onehot: t217_leaf0 + t217_leaf1 + t217_leaf2 + t217_leaf3
   + t217_leaf4 = 1
 T217_L_f33_k0: x_Garage_Area + 1e+06 t217_leaf0 <= 1.001153999999e+06
 T217_L_f5_k0: const_Year_Built + 1e+06 t217_leaf0 <= 1.001910999999e+06
 T217_L_f36_k0: x_Wood_Deck_SF + 1e+06 t217_leaf0 <= 1.000087999999e+06
 T217_L_f33_k1: x_Garage_Area + 1e+06 t217_leaf1 <= 1.001153999999e+06
 T217_L_f5_k1: const_Year_Built + 1e+06 t217_leaf1 <= 1.001910999999e+06
 T217_R_f36_k1: x_Wood_Deck_SF - 1e+06 t217_leaf1 >= -999912
 T217_L_f33_k2: x_Garage_Area + 1e+06 t217_leaf2 <= 1.001153999999e+06
 T217_R_f5_k2: const_Year_Built - 1e+06 t217_leaf2 >= -998089
 T217_L_f6_k2: const_Year_Remod_Add + 1e+06 t217_leaf2
   <= 1.001964999999e+06
 T217_L_f33_k3: x_Garage_Area + 1e+06 t217_leaf3 <= 1.001153999999e+06
 T217_R_f5_k3: const_Year_Built - 1e+06 t217_leaf3 >= -998089
 T217_R_f6_k3: const_Year_Remod_Add - 1e+06 t217_leaf3 >= -998035
 T217_R_f33_k4: x_Garage_Area - 1e+06 t217_leaf4 >= -998846
 T218_onehot: t218_leaf0 + t218_leaf1 + t218_leaf2 + t218_leaf3 = 1
 T218_L_f3_k0: Overall_Qual_calc + 1e+06 t218_leaf0 <= 1.000009999999e+06
 T218_L_f37_k0: x_Open_Porch_SF + 1e+06 t218_leaf0 <= 1.000193999999e+06
 T218_L_f289_k0: const_Sale_Type_New + 1e+06 t218_leaf0
   <= 1.000000999999e+06
 T218_L_f3_k1: Overall_Qual_calc + 1e+06 t218_leaf1 <= 1.000009999999e+06
 T218_L_f37_k1: x_Open_Porch_SF + 1e+06 t218_leaf1 <= 1.000193999999e+06
 T218_R_f289_k1: const_Sale_Type_New - 1e+06 t218_leaf1 >= -999999
 T218_L_f3_k2: Overall_Qual_calc + 1e+06 t218_leaf2 <= 1.000009999999e+06
 T218_R_f37_k2: x_Open_Porch_SF - 1e+06 t218_leaf2 >= -999806
 T218_R_f3_k3: Overall_Qual_calc - 1e+06 t218_leaf3 >= -999990
 T219_onehot: t219_leaf0 + t219_leaf1 + t219_leaf2 + t219_leaf3
   + t219_leaf4 + t219_leaf5 + t219_leaf6 = 1
 T219_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t219_leaf0
   <= 1.000000999999e+06
 T219_L_f86_k0: const_Land_Slope_Gtl + 1e+06 t219_leaf0
   <= 1.000000999999e+06
 T219_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t219_leaf1
   <= 1.000000999999e+06
 T219_R_f86_k1: const_Land_Slope_Gtl - 1e+06 t219_leaf1 >= -999999
 T219_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t219_leaf1
   <= 1.000963999999e+06
 T219_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t219_leaf2
   <= 1.000000999999e+06
 T219_R_f86_k2: const_Land_Slope_Gtl - 1e+06 t219_leaf2 >= -999999
 T219_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t219_leaf2 >= -999036
 T219_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t219_leaf3 >= -999999
 T219_L_f1_k3: const_Lot_Area + 1e+06 t219_leaf3 <= 1.015311999999e+06
 T219_L_f46_k3: const_MS_SubClass_30 + 1e+06 t219_leaf3
   <= 1.000000999999e+06
 T219_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t219_leaf4 >= -999999
 T219_L_f1_k4: const_Lot_Area + 1e+06 t219_leaf4 <= 1.015311999999e+06
 T219_R_f46_k4: const_MS_SubClass_30 - 1e+06 t219_leaf4 >= -999999
 T219_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t219_leaf5 >= -999999
 T219_R_f1_k5: const_Lot_Area - 1e+06 t219_leaf5 >= -984688
 T219_L_f12_k5: x_BsmtFin_SF_1 + 1e+06 t219_leaf5 <= 1.001318999999e+06
 T219_R_f245_k6: const_FunctioNo_aplical_Typ - 1e+06 t219_leaf6 >= -999999
 T219_R_f1_k6: const_Lot_Area - 1e+06 t219_leaf6 >= -984688
 T219_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t219_leaf6 >= -998681
 T220_onehot: t220_leaf0 + t220_leaf1 + t220_leaf2 + t220_leaf3
   + t220_leaf4 + t220_leaf5 + t220_leaf6 = 1
 T220_L_f44_k0: const_Yr_Sold + 1e+06 t220_leaf0 <= 1.002008999999e+06
 T220_L_f272_k0: const_Mo_Sold_2 + 1e+06 t220_leaf0 <= 1.000000999999e+06
 T220_L_f250_k0: const_Garage_Type_CarPort + 1e+06 t220_leaf0
   <= 1.000000999999e+06
 T220_L_f44_k1: const_Yr_Sold + 1e+06 t220_leaf1 <= 1.002008999999e+06
 T220_L_f272_k1: const_Mo_Sold_2 + 1e+06 t220_leaf1 <= 1.000000999999e+06
 T220_R_f250_k1: const_Garage_Type_CarPort - 1e+06 t220_leaf1 >= -999999
 T220_L_f44_k2: const_Yr_Sold + 1e+06 t220_leaf2 <= 1.002008999999e+06
 T220_R_f272_k2: const_Mo_Sold_2 - 1e+06 t220_leaf2 >= -999999
 T220_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t220_leaf2
   <= 1.001488999999e+06
 T220_L_f44_k3: const_Yr_Sold + 1e+06 t220_leaf3 <= 1.002008999999e+06
 T220_R_f272_k3: const_Mo_Sold_2 - 1e+06 t220_leaf3 >= -999999
 T220_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t220_leaf3 >= -998511
 T220_R_f44_k4: const_Yr_Sold - 1e+06 t220_leaf4 >= -997991
 T220_L_f293_k4: const_Sale_Condition_Abnorml + 1e+06 t220_leaf4
   <= 1.000000999999e+06
 T220_L_f150_k4: const_Roof_Style_Hip + 1e+06 t220_leaf4
   <= 1.000000999999e+06
 T220_R_f44_k5: const_Yr_Sold - 1e+06 t220_leaf5 >= -997991
 T220_L_f293_k5: const_Sale_Condition_Abnorml + 1e+06 t220_leaf5
   <= 1.000000999999e+06
 T220_R_f150_k5: const_Roof_Style_Hip - 1e+06 t220_leaf5 >= -999999
 T220_R_f44_k6: const_Yr_Sold - 1e+06 t220_leaf6 >= -997991
 T220_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t220_leaf6
   >= -999999
 T221_onehot: t221_leaf0 + t221_leaf1 + t221_leaf2 + t221_leaf3
   + t221_leaf4 + t221_leaf5 = 1
 T221_L_f44_k0: const_Yr_Sold + 1e+06 t221_leaf0 <= 1.002006999999e+06
 T221_L_f231_k0: central_air_no + 1e+06 t221_leaf0 <= 1.000000999999e+06
 T221_L_f4_k0: const_Overall_Cond + 1e+06 t221_leaf0 <= 1.000005999999e+06
 T221_L_f44_k1: const_Yr_Sold + 1e+06 t221_leaf1 <= 1.002006999999e+06
 T221_L_f231_k1: central_air_no + 1e+06 t221_leaf1 <= 1.000000999999e+06
 T221_R_f4_k1: const_Overall_Cond - 1e+06 t221_leaf1 >= -999994
 T221_L_f44_k2: const_Yr_Sold + 1e+06 t221_leaf2 <= 1.002006999999e+06
 T221_R_f231_k2: central_air_no - 1e+06 t221_leaf2 >= -999999
 T221_R_f44_k3: const_Yr_Sold - 1e+06 t221_leaf3 >= -997993
 T221_L_f154_k3: x_roof_matl_is_CompShg + 1e+06 t221_leaf3
   <= 1.000000999999e+06
 T221_R_f44_k4: const_Yr_Sold - 1e+06 t221_leaf4 >= -997993
 T221_R_f154_k4: x_roof_matl_is_CompShg - 1e+06 t221_leaf4 >= -999999
 T221_L_f17_k4: x_1st_Flr_SF + 1e+06 t221_leaf4 <= 1.002035999999e+06
 T221_R_f44_k5: const_Yr_Sold - 1e+06 t221_leaf5 >= -997993
 T221_R_f154_k5: x_roof_matl_is_CompShg - 1e+06 t221_leaf5 >= -999999
 T221_R_f17_k5: x_1st_Flr_SF - 1e+06 t221_leaf5 >= -997964
 T222_onehot: t222_leaf0 + t222_leaf1 + t222_leaf2 + t222_leaf3
   + t222_leaf4 = 1
 T222_L_f18_k0: const_2nd_Flr_SF + 1e+06 t222_leaf0 <= 1.001628999999e+06
 T222_L_f44_k0: const_Yr_Sold + 1e+06 t222_leaf0 <= 1.002006999999e+06
 T222_L_f0_k0: const_Lot_Frontage + 1e+06 t222_leaf0 <= 1.000089999999e+06
 T222_L_f18_k1: const_2nd_Flr_SF + 1e+06 t222_leaf1 <= 1.001628999999e+06
 T222_L_f44_k1: const_Yr_Sold + 1e+06 t222_leaf1 <= 1.002006999999e+06
 T222_R_f0_k1: const_Lot_Frontage - 1e+06 t222_leaf1 >= -999910
 T222_L_f18_k2: const_2nd_Flr_SF + 1e+06 t222_leaf2 <= 1.001628999999e+06
 T222_R_f44_k2: const_Yr_Sold - 1e+06 t222_leaf2 >= -997993
 T222_L_f6_k2: const_Year_Remod_Add + 1e+06 t222_leaf2
   <= 1.002006999999e+06
 T222_L_f18_k3: const_2nd_Flr_SF + 1e+06 t222_leaf3 <= 1.001628999999e+06
 T222_R_f44_k3: const_Yr_Sold - 1e+06 t222_leaf3 >= -997993
 T222_R_f6_k3: const_Year_Remod_Add - 1e+06 t222_leaf3 >= -997993
 T222_R_f18_k4: const_2nd_Flr_SF - 1e+06 t222_leaf4 >= -998371
 T223_onehot: t223_leaf0 + t223_leaf1 + t223_leaf2 + t223_leaf3
   + t223_leaf4 + t223_leaf5 + t223_leaf6 = 1
 T223_L_f119_k0: const_Condition_1_Norm + 1e+06 t223_leaf0
   <= 1.000000999999e+06
 T223_L_f0_k0: const_Lot_Frontage + 1e+06 t223_leaf0 <= 1.000120999999e+06
 T223_L_f0_k0: const_Lot_Frontage + 1e+06 t223_leaf0 <= 1.000079999999e+06
 T223_L_f119_k1: const_Condition_1_Norm + 1e+06 t223_leaf1
   <= 1.000000999999e+06
 T223_L_f0_k1: const_Lot_Frontage + 1e+06 t223_leaf1 <= 1.000120999999e+06
 T223_R_f0_k1: const_Lot_Frontage - 1e+06 t223_leaf1 >= -999920
 T223_L_f119_k2: const_Condition_1_Norm + 1e+06 t223_leaf2
   <= 1.000000999999e+06
 T223_R_f0_k2: const_Lot_Frontage - 1e+06 t223_leaf2 >= -999879
 T223_R_f119_k3: const_Condition_1_Norm - 1e+06 t223_leaf3 >= -999999
 T223_L_f29_k3: const_Fireplaces + 1e+06 t223_leaf3 <= 1.000001999999e+06
 T223_L_f44_k3: const_Yr_Sold + 1e+06 t223_leaf3 <= 1.002006999999e+06
 T223_R_f119_k4: const_Condition_1_Norm - 1e+06 t223_leaf4 >= -999999
 T223_L_f29_k4: const_Fireplaces + 1e+06 t223_leaf4 <= 1.000001999999e+06
 T223_R_f44_k4: const_Yr_Sold - 1e+06 t223_leaf4 >= -997993
 T223_R_f119_k5: const_Condition_1_Norm - 1e+06 t223_leaf5 >= -999999
 T223_R_f29_k5: const_Fireplaces - 1e+06 t223_leaf5 >= -999998
 T223_L_f1_k5: const_Lot_Area + 1e+06 t223_leaf5 <= 1.017870999999e+06
 T223_R_f119_k6: const_Condition_1_Norm - 1e+06 t223_leaf6 >= -999999
 T223_R_f29_k6: const_Fireplaces - 1e+06 t223_leaf6 >= -999998
 T223_R_f1_k6: const_Lot_Area - 1e+06 t223_leaf6 >= -982129
 T224_onehot: t224_leaf0 + t224_leaf1 + t224_leaf2 + t224_leaf3 = 1
 T224_L_f33_k0: x_Garage_Area + 1e+06 t224_leaf0 <= 1.001153999999e+06
 T224_L_f69_k0: const_Street_Pave + 1e+06 t224_leaf0 <= 1.000000999999e+06
 T224_L_f33_k1: x_Garage_Area + 1e+06 t224_leaf1 <= 1.001153999999e+06
 T224_R_f69_k1: const_Street_Pave - 1e+06 t224_leaf1 >= -999999
 T224_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t224_leaf1
   <= 1.000000999999e+06
 T224_L_f33_k2: x_Garage_Area + 1e+06 t224_leaf2 <= 1.001153999999e+06
 T224_R_f69_k2: const_Street_Pave - 1e+06 t224_leaf2 >= -999999
 T224_R_f207_k2: const_Bsmt_Exposure_Gd - 1e+06 t224_leaf2 >= -999999
 T224_R_f33_k3: x_Garage_Area - 1e+06 t224_leaf3 >= -998846
 T225_onehot: t225_leaf0 + t225_leaf1 + t225_leaf2 + t225_leaf3
   + t225_leaf4 = 1
 T225_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t225_leaf0
   <= 1.000001999999e+06
 T225_L_f36_k0: x_Wood_Deck_SF + 1e+06 t225_leaf0 <= 1.000211999999e+06
 T225_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t225_leaf0
   <= 1.000000999999e+06
 T225_L_f21_k1: const_Bsmt_Full_Bath + 1e+06 t225_leaf1
   <= 1.000001999999e+06
 T225_L_f36_k1: x_Wood_Deck_SF + 1e+06 t225_leaf1 <= 1.000211999999e+06
 T225_R_f96_k1: const_Neighborhood_Edwards - 1e+06 t225_leaf1 >= -999999
 T225_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t225_leaf2
   <= 1.000001999999e+06
 T225_R_f36_k2: x_Wood_Deck_SF - 1e+06 t225_leaf2 >= -999788
 T225_L_f0_k2: const_Lot_Frontage + 1e+06 t225_leaf2 <= 1.000114999999e+06
 T225_L_f21_k3: const_Bsmt_Full_Bath + 1e+06 t225_leaf3
   <= 1.000001999999e+06
 T225_R_f36_k3: x_Wood_Deck_SF - 1e+06 t225_leaf3 >= -999788
 T225_R_f0_k3: const_Lot_Frontage - 1e+06 t225_leaf3 >= -999885
 T225_R_f21_k4: const_Bsmt_Full_Bath - 1e+06 t225_leaf4 >= -999998
 T226_onehot: t226_leaf0 + t226_leaf1 + t226_leaf2 + t226_leaf3
   + t226_leaf4 + t226_leaf5 + t226_leaf6 + t226_leaf7 = 1
 T226_L_f20_k0: x_Gr_Liv_Area + 1e+06 t226_leaf0 <= 1.000987999999e+06
 T226_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t226_leaf0
   <= 1.000000999999e+06
 T226_L_f231_k0: central_air_no + 1e+06 t226_leaf0 <= 1.000000999999e+06
 T226_L_f20_k1: x_Gr_Liv_Area + 1e+06 t226_leaf1 <= 1.000987999999e+06
 T226_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t226_leaf1
   <= 1.000000999999e+06
 T226_R_f231_k1: central_air_no - 1e+06 t226_leaf1 >= -999999
 T226_L_f20_k2: x_Gr_Liv_Area + 1e+06 t226_leaf2 <= 1.000987999999e+06
 T226_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t226_leaf2
   >= -999999
 T226_L_f44_k2: const_Yr_Sold + 1e+06 t226_leaf2 <= 1.002007999999e+06
 T226_L_f20_k3: x_Gr_Liv_Area + 1e+06 t226_leaf3 <= 1.000987999999e+06
 T226_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t226_leaf3
   >= -999999
 T226_R_f44_k3: const_Yr_Sold - 1e+06 t226_leaf3 >= -997992
 T226_R_f20_k4: x_Gr_Liv_Area - 1e+06 t226_leaf4 >= -999012
 T226_L_f4_k4: const_Overall_Cond + 1e+06 t226_leaf4 <= 1.000005999999e+06
 T226_L_f14_k4: x_Bsmt_Unf_SF + 1e+06 t226_leaf4 <= 1.000465999999e+06
 T226_R_f20_k5: x_Gr_Liv_Area - 1e+06 t226_leaf5 >= -999012
 T226_L_f4_k5: const_Overall_Cond + 1e+06 t226_leaf5 <= 1.000005999999e+06
 T226_R_f14_k5: x_Bsmt_Unf_SF - 1e+06 t226_leaf5 >= -999534
 T226_R_f20_k6: x_Gr_Liv_Area - 1e+06 t226_leaf6 >= -999012
 T226_R_f4_k6: const_Overall_Cond - 1e+06 t226_leaf6 >= -999994
 T226_L_f5_k6: const_Year_Built + 1e+06 t226_leaf6 <= 1.001891999999e+06
 T226_R_f20_k7: x_Gr_Liv_Area - 1e+06 t226_leaf7 >= -999012
 T226_R_f4_k7: const_Overall_Cond - 1e+06 t226_leaf7 >= -999994
 T226_R_f5_k7: const_Year_Built - 1e+06 t226_leaf7 >= -998108
 T227_onehot: t227_leaf0 + t227_leaf1 + t227_leaf2 + t227_leaf3
   + t227_leaf4 = 1
 T227_L_f40_k0: x_Screen_Porch + 1e+06 t227_leaf0 <= 1.000142999999e+06
 T227_L_f62_k0: const_MS_Zoning_C_(all) + 1e+06 t227_leaf0
   <= 1.000000999999e+06
 T227_L_f0_k0: const_Lot_Frontage + 1e+06 t227_leaf0 <= 1.000129999999e+06
 T227_L_f40_k1: x_Screen_Porch + 1e+06 t227_leaf1 <= 1.000142999999e+06
 T227_L_f62_k1: const_MS_Zoning_C_(all) + 1e+06 t227_leaf1
   <= 1.000000999999e+06
 T227_R_f0_k1: const_Lot_Frontage - 1e+06 t227_leaf1 >= -999870
 T227_L_f40_k2: x_Screen_Porch + 1e+06 t227_leaf2 <= 1.000142999999e+06
 T227_R_f62_k2: const_MS_Zoning_C_(all) - 1e+06 t227_leaf2 >= -999999
 T227_L_f1_k2: const_Lot_Area + 1e+06 t227_leaf2 <= 1.006599999999e+06
 T227_L_f40_k3: x_Screen_Porch + 1e+06 t227_leaf3 <= 1.000142999999e+06
 T227_R_f62_k3: const_MS_Zoning_C_(all) - 1e+06 t227_leaf3 >= -999999
 T227_R_f1_k3: const_Lot_Area - 1e+06 t227_leaf3 >= -993400
 T227_R_f40_k4: x_Screen_Porch - 1e+06 t227_leaf4 >= -999857
 T228_onehot: t228_leaf0 + t228_leaf1 + t228_leaf2 + t228_leaf3
   + t228_leaf4 = 1
 T228_L_f118_k0: const_Condition_1_Feedr + 1e+06 t228_leaf0
   <= 1.000000999999e+06
 T228_L_f23_k0: x_Full_Bath + 1e+06 t228_leaf0 <= 1.000002999999e+06
 T228_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t228_leaf0
   <= 1.000000999999e+06
 T228_L_f118_k1: const_Condition_1_Feedr + 1e+06 t228_leaf1
   <= 1.000000999999e+06
 T228_L_f23_k1: x_Full_Bath + 1e+06 t228_leaf1 <= 1.000002999999e+06
 T228_R_f245_k1: const_FunctioNo_aplical_Typ - 1e+06 t228_leaf1 >= -999999
 T228_L_f118_k2: const_Condition_1_Feedr + 1e+06 t228_leaf2
   <= 1.000000999999e+06
 T228_R_f23_k2: x_Full_Bath - 1e+06 t228_leaf2 >= -999997
 T228_R_f118_k3: const_Condition_1_Feedr - 1e+06 t228_leaf3 >= -999999
 T228_L_f77_k3: const_Land_Contour_Bnk + 1e+06 t228_leaf3
   <= 1.000000999999e+06
 T228_R_f118_k4: const_Condition_1_Feedr - 1e+06 t228_leaf4 >= -999999
 T228_R_f77_k4: const_Land_Contour_Bnk - 1e+06 t228_leaf4 >= -999999
 T229_onehot: t229_leaf0 + t229_leaf1 + t229_leaf2 + t229_leaf3
   + t229_leaf4 = 1
 T229_L_f29_k0: const_Fireplaces + 1e+06 t229_leaf0 <= 1.000002999999e+06
 T229_L_f20_k0: x_Gr_Liv_Area + 1e+06 t229_leaf0 <= 1.003085999999e+06
 T229_L_f1_k0: const_Lot_Area + 1e+06 t229_leaf0 <= 1.004834999999e+06
 T229_L_f29_k1: const_Fireplaces + 1e+06 t229_leaf1 <= 1.000002999999e+06
 T229_L_f20_k1: x_Gr_Liv_Area + 1e+06 t229_leaf1 <= 1.003085999999e+06
 T229_R_f1_k1: const_Lot_Area - 1e+06 t229_leaf1 >= -995165
 T229_L_f29_k2: const_Fireplaces + 1e+06 t229_leaf2 <= 1.000002999999e+06
 T229_R_f20_k2: x_Gr_Liv_Area - 1e+06 t229_leaf2 >= -996914
 T229_L_f37_k2: x_Open_Porch_SF + 1e+06 t229_leaf2 <= 1.000034999999e+06
 T229_L_f29_k3: const_Fireplaces + 1e+06 t229_leaf3 <= 1.000002999999e+06
 T229_R_f20_k3: x_Gr_Liv_Area - 1e+06 t229_leaf3 >= -996914
 T229_R_f37_k3: x_Open_Porch_SF - 1e+06 t229_leaf3 >= -999965
 T229_R_f29_k4: const_Fireplaces - 1e+06 t229_leaf4 >= -999997
 T230_onehot: t230_leaf0 + t230_leaf1 + t230_leaf2 + t230_leaf3
   + t230_leaf4 + t230_leaf5 + t230_leaf6 = 1
 T230_L_f4_k0: const_Overall_Cond + 1e+06 t230_leaf0 <= 1.000005999999e+06
 T230_L_f33_k0: x_Garage_Area + 1e+06 t230_leaf0 <= 1.000252999999e+06
 T230_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t230_leaf0
   <= 1.000000999999e+06
 T230_L_f4_k1: const_Overall_Cond + 1e+06 t230_leaf1 <= 1.000005999999e+06
 T230_L_f33_k1: x_Garage_Area + 1e+06 t230_leaf1 <= 1.000252999999e+06
 T230_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t230_leaf1 >= -999999
 T230_L_f4_k2: const_Overall_Cond + 1e+06 t230_leaf2 <= 1.000005999999e+06
 T230_R_f33_k2: x_Garage_Area - 1e+06 t230_leaf2 >= -999747
 T230_L_f3_k2: Overall_Qual_calc + 1e+06 t230_leaf2 <= 1.000002999999e+06
 T230_L_f4_k3: const_Overall_Cond + 1e+06 t230_leaf3 <= 1.000005999999e+06
 T230_R_f33_k3: x_Garage_Area - 1e+06 t230_leaf3 >= -999747
 T230_R_f3_k3: Overall_Qual_calc - 1e+06 t230_leaf3 >= -999997
 T230_R_f4_k4: const_Overall_Cond - 1e+06 t230_leaf4 >= -999994
 T230_L_f78_k4: const_Land_Contour_HLS + 1e+06 t230_leaf4
   <= 1.000000999999e+06
 T230_L_f9_k4: x_Exter_Cond + 1e+06 t230_leaf4 <= 1.000002999999e+06
 T230_R_f4_k5: const_Overall_Cond - 1e+06 t230_leaf5 >= -999994
 T230_L_f78_k5: const_Land_Contour_HLS + 1e+06 t230_leaf5
   <= 1.000000999999e+06
 T230_R_f9_k5: x_Exter_Cond - 1e+06 t230_leaf5 >= -999997
 T230_R_f4_k6: const_Overall_Cond - 1e+06 t230_leaf6 >= -999994
 T230_R_f78_k6: const_Land_Contour_HLS - 1e+06 t230_leaf6 >= -999999
 T231_onehot: t231_leaf0 + t231_leaf1 + t231_leaf2 + t231_leaf3
   + t231_leaf4 + t231_leaf5 + t231_leaf6 + t231_leaf7 = 1
 T231_L_f1_k0: const_Lot_Area + 1e+06 t231_leaf0 <= 1.019521999999e+06
 T231_L_f4_k0: const_Overall_Cond + 1e+06 t231_leaf0 <= 1.000005999999e+06
 T231_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t231_leaf0
   <= 1.000000999999e+06
 T231_L_f1_k1: const_Lot_Area + 1e+06 t231_leaf1 <= 1.019521999999e+06
 T231_L_f4_k1: const_Overall_Cond + 1e+06 t231_leaf1 <= 1.000005999999e+06
 T231_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t231_leaf1
   >= -999999
 T231_L_f1_k2: const_Lot_Area + 1e+06 t231_leaf2 <= 1.019521999999e+06
 T231_R_f4_k2: const_Overall_Cond - 1e+06 t231_leaf2 >= -999994
 T231_L_f1_k2: const_Lot_Area + 1e+06 t231_leaf2 <= 1.007399999999e+06
 T231_L_f1_k3: const_Lot_Area + 1e+06 t231_leaf3 <= 1.019521999999e+06
 T231_R_f4_k3: const_Overall_Cond - 1e+06 t231_leaf3 >= -999994
 T231_R_f1_k3: const_Lot_Area - 1e+06 t231_leaf3 >= -992600
 T231_R_f1_k4: const_Lot_Area - 1e+06 t231_leaf4 >= -980478
 T231_L_f213_k4: x_b1_is_GLQ + 1e+06 t231_leaf4 <= 1.000000999999e+06
 T231_L_f197_k4: x_mvt_is_No_aplica + 1e+06 t231_leaf4
   <= 1.000000999999e+06
 T231_R_f1_k5: const_Lot_Area - 1e+06 t231_leaf5 >= -980478
 T231_L_f213_k5: x_b1_is_GLQ + 1e+06 t231_leaf5 <= 1.000000999999e+06
 T231_R_f197_k5: x_mvt_is_No_aplica - 1e+06 t231_leaf5 >= -999999
 T231_R_f1_k6: const_Lot_Area - 1e+06 t231_leaf6 >= -980478
 T231_R_f213_k6: x_b1_is_GLQ - 1e+06 t231_leaf6 >= -999999
 T231_L_f37_k6: x_Open_Porch_SF + 1e+06 t231_leaf6 <= 1.000071999999e+06
 T231_R_f1_k7: const_Lot_Area - 1e+06 t231_leaf7 >= -980478
 T231_R_f213_k7: x_b1_is_GLQ - 1e+06 t231_leaf7 >= -999999
 T231_R_f37_k7: x_Open_Porch_SF - 1e+06 t231_leaf7 >= -999928
 T232_onehot: t232_leaf0 + t232_leaf1 + t232_leaf2 + t232_leaf3
   + t232_leaf4 = 1
 T232_L_f37_k0: x_Open_Porch_SF + 1e+06 t232_leaf0 <= 1.000290999999e+06
 T232_L_f17_k0: x_1st_Flr_SF + 1e+06 t232_leaf0 <= 1.002199999999e+06
 T232_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t232_leaf0 <= 1.000000999999e+06
 T232_L_f37_k1: x_Open_Porch_SF + 1e+06 t232_leaf1 <= 1.000290999999e+06
 T232_L_f17_k1: x_1st_Flr_SF + 1e+06 t232_leaf1 <= 1.002199999999e+06
 T232_R_f164_k1: x_ex1_is_BrkFace - 1e+06 t232_leaf1 >= -999999
 T232_L_f37_k2: x_Open_Porch_SF + 1e+06 t232_leaf2 <= 1.000290999999e+06
 T232_R_f17_k2: x_1st_Flr_SF - 1e+06 t232_leaf2 >= -997800
 T232_R_f37_k3: x_Open_Porch_SF - 1e+06 t232_leaf3 >= -999709
 T232_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t232_leaf3 <= 1.000396999999e+06
 T232_R_f37_k4: x_Open_Porch_SF - 1e+06 t232_leaf4 >= -999709
 T232_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t232_leaf4 >= -999603
 T233_onehot: t233_leaf0 + t233_leaf1 + t233_leaf2 + t233_leaf3
   + t233_leaf4 + t233_leaf5 + t233_leaf6 = 1
 T233_L_f37_k0: x_Open_Porch_SF + 1e+06 t233_leaf0 <= 1.000149999999e+06
 T233_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t233_leaf0 <= 1.000000999999e+06
 T233_L_f4_k0: const_Overall_Cond + 1e+06 t233_leaf0 <= 1.000007999999e+06
 T233_L_f37_k1: x_Open_Porch_SF + 1e+06 t233_leaf1 <= 1.000149999999e+06
 T233_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t233_leaf1 <= 1.000000999999e+06
 T233_R_f4_k1: const_Overall_Cond - 1e+06 t233_leaf1 >= -999992
 T233_L_f37_k2: x_Open_Porch_SF + 1e+06 t233_leaf2 <= 1.000149999999e+06
 T233_R_f164_k2: x_ex1_is_BrkFace - 1e+06 t233_leaf2 >= -999999
 T233_L_f118_k2: const_Condition_1_Feedr + 1e+06 t233_leaf2
   <= 1.000000999999e+06
 T233_L_f37_k3: x_Open_Porch_SF + 1e+06 t233_leaf3 <= 1.000149999999e+06
 T233_R_f164_k3: x_ex1_is_BrkFace - 1e+06 t233_leaf3 >= -999999
 T233_R_f118_k3: const_Condition_1_Feedr - 1e+06 t233_leaf3 >= -999999
 T233_R_f37_k4: x_Open_Porch_SF - 1e+06 t233_leaf4 >= -999850
 T233_L_f13_k4: x_BsmtFin_SF_2 + 1e+06 t233_leaf4 <= 1.000490999999e+06
 T233_L_f20_k4: x_Gr_Liv_Area + 1e+06 t233_leaf4 <= 1.002392999999e+06
 T233_R_f37_k5: x_Open_Porch_SF - 1e+06 t233_leaf5 >= -999850
 T233_L_f13_k5: x_BsmtFin_SF_2 + 1e+06 t233_leaf5 <= 1.000490999999e+06
 T233_R_f20_k5: x_Gr_Liv_Area - 1e+06 t233_leaf5 >= -997607
 T233_R_f37_k6: x_Open_Porch_SF - 1e+06 t233_leaf6 >= -999850
 T233_R_f13_k6: x_BsmtFin_SF_2 - 1e+06 t233_leaf6 >= -999509
 T234_onehot: t234_leaf0 + t234_leaf1 + t234_leaf2 + t234_leaf3
   + t234_leaf4 + t234_leaf5 = 1
 T234_L_f3_k0: Overall_Qual_calc + 1e+06 t234_leaf0 <= 1.000003999999e+06
 T234_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t234_leaf0
   <= 1.000715999999e+06
 T234_L_f3_k1: Overall_Qual_calc + 1e+06 t234_leaf1 <= 1.000003999999e+06
 T234_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t234_leaf1 >= -999284
 T234_R_f3_k2: Overall_Qual_calc - 1e+06 t234_leaf2 >= -999996
 T234_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t234_leaf2 <= 1.001731999999e+06
 T234_L_f112_k2: const_Neighborhood_SawyerW + 1e+06 t234_leaf2
   <= 1.000000999999e+06
 T234_R_f3_k3: Overall_Qual_calc - 1e+06 t234_leaf3 >= -999996
 T234_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t234_leaf3 <= 1.001731999999e+06
 T234_R_f112_k3: const_Neighborhood_SawyerW - 1e+06 t234_leaf3 >= -999999
 T234_R_f3_k4: Overall_Qual_calc - 1e+06 t234_leaf4 >= -999996
 T234_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t234_leaf4 >= -998268
 T234_L_f36_k4: x_Wood_Deck_SF + 1e+06 t234_leaf4 <= 1.000171999999e+06
 T234_R_f3_k5: Overall_Qual_calc - 1e+06 t234_leaf5 >= -999996
 T234_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t234_leaf5 >= -998268
 T234_R_f36_k5: x_Wood_Deck_SF - 1e+06 t234_leaf5 >= -999828
 T235_onehot: t235_leaf0 + t235_leaf1 + t235_leaf2 + t235_leaf3
   + t235_leaf4 = 1
 T235_L_f17_k0: x_1st_Flr_SF + 1e+06 t235_leaf0 <= 1.002514999999e+06
 T235_L_f20_k0: x_Gr_Liv_Area + 1e+06 t235_leaf0 <= 1.000767999999e+06
 T235_L_f66_k0: const_MS_Zoning_RL + 1e+06 t235_leaf0
   <= 1.000000999999e+06
 T235_L_f17_k1: x_1st_Flr_SF + 1e+06 t235_leaf1 <= 1.002514999999e+06
 T235_L_f20_k1: x_Gr_Liv_Area + 1e+06 t235_leaf1 <= 1.000767999999e+06
 T235_R_f66_k1: const_MS_Zoning_RL - 1e+06 t235_leaf1 >= -999999
 T235_L_f17_k2: x_1st_Flr_SF + 1e+06 t235_leaf2 <= 1.002514999999e+06
 T235_R_f20_k2: x_Gr_Liv_Area - 1e+06 t235_leaf2 >= -999232
 T235_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t235_leaf2 <= 1.000672999999e+06
 T235_L_f17_k3: x_1st_Flr_SF + 1e+06 t235_leaf3 <= 1.002514999999e+06
 T235_R_f20_k3: x_Gr_Liv_Area - 1e+06 t235_leaf3 >= -999232
 T235_R_f7_k3: x_Mas_Vnr_Area - 1e+06 t235_leaf3 >= -999327
 T235_R_f17_k4: x_1st_Flr_SF - 1e+06 t235_leaf4 >= -997485
 T236_onehot: t236_leaf0 + t236_leaf1 + t236_leaf2 + t236_leaf3
   + t236_leaf4 + t236_leaf5 = 1
 T236_L_f4_k0: const_Overall_Cond + 1e+06 t236_leaf0 <= 1.000006999999e+06
 T236_L_f82_k0: const_Lot_Config_CulDSac + 1e+06 t236_leaf0
   <= 1.000000999999e+06
 T236_L_f63_k0: const_MS_Zoning_FV + 1e+06 t236_leaf0
   <= 1.000000999999e+06
 T236_L_f4_k1: const_Overall_Cond + 1e+06 t236_leaf1 <= 1.000006999999e+06
 T236_L_f82_k1: const_Lot_Config_CulDSac + 1e+06 t236_leaf1
   <= 1.000000999999e+06
 T236_R_f63_k1: const_MS_Zoning_FV - 1e+06 t236_leaf1 >= -999999
 T236_L_f4_k2: const_Overall_Cond + 1e+06 t236_leaf2 <= 1.000006999999e+06
 T236_R_f82_k2: const_Lot_Config_CulDSac - 1e+06 t236_leaf2 >= -999999
 T236_R_f4_k3: const_Overall_Cond - 1e+06 t236_leaf3 >= -999993
 T236_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t236_leaf3
   <= 1.001977999999e+06
 T236_L_f1_k3: const_Lot_Area + 1e+06 t236_leaf3 <= 1.009737999999e+06
 T236_R_f4_k4: const_Overall_Cond - 1e+06 t236_leaf4 >= -999993
 T236_L_f31_k4: const_Garage_Yr_Blt + 1e+06 t236_leaf4
   <= 1.001977999999e+06
 T236_R_f1_k4: const_Lot_Area - 1e+06 t236_leaf4 >= -990262
 T236_R_f4_k5: const_Overall_Cond - 1e+06 t236_leaf5 >= -999993
 T236_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t236_leaf5 >= -998022
 T237_onehot: t237_leaf0 + t237_leaf1 + t237_leaf2 + t237_leaf3 = 1
 T237_L_f40_k0: x_Screen_Porch + 1e+06 t237_leaf0 <= 1.000103999999e+06
 T237_L_f3_k0: Overall_Qual_calc + 1e+06 t237_leaf0 <= 1.000009999999e+06
 T237_L_f3_k0: Overall_Qual_calc + 1e+06 t237_leaf0 <= 1.000003999999e+06
 T237_L_f40_k1: x_Screen_Porch + 1e+06 t237_leaf1 <= 1.000103999999e+06
 T237_L_f3_k1: Overall_Qual_calc + 1e+06 t237_leaf1 <= 1.000009999999e+06
 T237_R_f3_k1: Overall_Qual_calc - 1e+06 t237_leaf1 >= -999996
 T237_L_f40_k2: x_Screen_Porch + 1e+06 t237_leaf2 <= 1.000103999999e+06
 T237_R_f3_k2: Overall_Qual_calc - 1e+06 t237_leaf2 >= -999990
 T237_R_f40_k3: x_Screen_Porch - 1e+06 t237_leaf3 >= -999896
 T238_onehot: t238_leaf0 + t238_leaf1 + t238_leaf2 + t238_leaf3
   + t238_leaf4 + t238_leaf5 = 1
 T238_L_f0_k0: const_Lot_Frontage + 1e+06 t238_leaf0 <= 1.000119999999e+06
 T238_L_f1_k0: const_Lot_Area + 1e+06 t238_leaf0 <= 1.002759999999e+06
 T238_L_f0_k1: const_Lot_Frontage + 1e+06 t238_leaf1 <= 1.000119999999e+06
 T238_R_f1_k1: const_Lot_Area - 1e+06 t238_leaf1 >= -997240
 T238_L_f233_k1: x_elect_is_FuseA + 1e+06 t238_leaf1 <= 1.000000999999e+06
 T238_L_f0_k2: const_Lot_Frontage + 1e+06 t238_leaf2 <= 1.000119999999e+06
 T238_R_f1_k2: const_Lot_Area - 1e+06 t238_leaf2 >= -997240
 T238_R_f233_k2: x_elect_is_FuseA - 1e+06 t238_leaf2 >= -999999
 T238_R_f0_k3: const_Lot_Frontage - 1e+06 t238_leaf3 >= -999880
 T238_L_f8_k3: x_Exter_Qual + 1e+06 t238_leaf3 <= 1.000002999999e+06
 T238_R_f0_k4: const_Lot_Frontage - 1e+06 t238_leaf4 >= -999880
 T238_R_f8_k4: x_Exter_Qual - 1e+06 t238_leaf4 >= -999997
 T238_L_f36_k4: x_Wood_Deck_SF + 1e+06 t238_leaf4 <= 1.000149999999e+06
 T238_R_f0_k5: const_Lot_Frontage - 1e+06 t238_leaf5 >= -999880
 T238_R_f8_k5: x_Exter_Qual - 1e+06 t238_leaf5 >= -999997
 T238_R_f36_k5: x_Wood_Deck_SF - 1e+06 t238_leaf5 >= -999850
 T239_onehot: t239_leaf0 + t239_leaf1 + t239_leaf2 + t239_leaf3
   + t239_leaf4 + t239_leaf5 + t239_leaf6 = 1
 T239_L_f207_k0: const_Bsmt_Exposure_Gd + 1e+06 t239_leaf0
   <= 1.000000999999e+06
 T239_L_f4_k0: const_Overall_Cond + 1e+06 t239_leaf0 <= 1.000005999999e+06
 T239_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t239_leaf0
   <= 1.000000999999e+06
 T239_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t239_leaf1
   <= 1.000000999999e+06
 T239_L_f4_k1: const_Overall_Cond + 1e+06 t239_leaf1 <= 1.000005999999e+06
 T239_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t239_leaf1
   >= -999999
 T239_L_f207_k2: const_Bsmt_Exposure_Gd + 1e+06 t239_leaf2
   <= 1.000000999999e+06
 T239_R_f4_k2: const_Overall_Cond - 1e+06 t239_leaf2 >= -999994
 T239_L_f119_k2: const_Condition_1_Norm + 1e+06 t239_leaf2
   <= 1.000000999999e+06
 T239_L_f207_k3: const_Bsmt_Exposure_Gd + 1e+06 t239_leaf3
   <= 1.000000999999e+06
 T239_R_f4_k3: const_Overall_Cond - 1e+06 t239_leaf3 >= -999994
 T239_R_f119_k3: const_Condition_1_Norm - 1e+06 t239_leaf3 >= -999999
 T239_R_f207_k4: const_Bsmt_Exposure_Gd - 1e+06 t239_leaf4 >= -999999
 T239_L_f33_k4: x_Garage_Area + 1e+06 t239_leaf4 <= 1.000629999999e+06
 T239_R_f207_k5: const_Bsmt_Exposure_Gd - 1e+06 t239_leaf5 >= -999999
 T239_R_f33_k5: x_Garage_Area - 1e+06 t239_leaf5 >= -999370
 T239_L_f0_k5: const_Lot_Frontage + 1e+06 t239_leaf5 <= 1.000119999999e+06
 T239_R_f207_k6: const_Bsmt_Exposure_Gd - 1e+06 t239_leaf6 >= -999999
 T239_R_f33_k6: x_Garage_Area - 1e+06 t239_leaf6 >= -999370
 T239_R_f0_k6: const_Lot_Frontage - 1e+06 t239_leaf6 >= -999880
 T240_onehot: t240_leaf0 + t240_leaf1 + t240_leaf2 + t240_leaf3
   + t240_leaf4 + t240_leaf5 = 1
 T240_L_f23_k0: x_Full_Bath + 1e+06 t240_leaf0 <= 1.000002999999e+06
 T240_L_f1_k0: const_Lot_Area + 1e+06 t240_leaf0 <= 1.003674999999e+06
 T240_L_f23_k1: x_Full_Bath + 1e+06 t240_leaf1 <= 1.000002999999e+06
 T240_R_f1_k1: const_Lot_Area - 1e+06 t240_leaf1 >= -996325
 T240_L_f17_k1: x_1st_Flr_SF + 1e+06 t240_leaf1 <= 1.000830999999e+06
 T240_L_f23_k2: x_Full_Bath + 1e+06 t240_leaf2 <= 1.000002999999e+06
 T240_R_f1_k2: const_Lot_Area - 1e+06 t240_leaf2 >= -996325
 T240_R_f17_k2: x_1st_Flr_SF - 1e+06 t240_leaf2 >= -999169
 T240_R_f23_k3: x_Full_Bath - 1e+06 t240_leaf3 >= -999997
 T240_L_f197_k3: x_mvt_is_No_aplica + 1e+06 t240_leaf3
   <= 1.000000999999e+06
 T240_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t240_leaf3 <= 1.000187999999e+06
 T240_R_f23_k4: x_Full_Bath - 1e+06 t240_leaf4 >= -999997
 T240_L_f197_k4: x_mvt_is_No_aplica + 1e+06 t240_leaf4
   <= 1.000000999999e+06
 T240_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t240_leaf4 >= -999812
 T240_R_f23_k5: x_Full_Bath - 1e+06 t240_leaf5 >= -999997
 T240_R_f197_k5: x_mvt_is_No_aplica - 1e+06 t240_leaf5 >= -999999
 T241_onehot: t241_leaf0 + t241_leaf1 + t241_leaf2 + t241_leaf3
   + t241_leaf4 = 1
 T241_L_f3_k0: Overall_Qual_calc + 1e+06 t241_leaf0 <= 1.000002999999e+06
 T241_R_f3_k1: Overall_Qual_calc - 1e+06 t241_leaf1 >= -999997
 T241_L_f119_k1: const_Condition_1_Norm + 1e+06 t241_leaf1
   <= 1.000000999999e+06
 T241_L_f271_k1: const_Mo_Sold_1 + 1e+06 t241_leaf1 <= 1.000000999999e+06
 T241_R_f3_k2: Overall_Qual_calc - 1e+06 t241_leaf2 >= -999997
 T241_L_f119_k2: const_Condition_1_Norm + 1e+06 t241_leaf2
   <= 1.000000999999e+06
 T241_R_f271_k2: const_Mo_Sold_1 - 1e+06 t241_leaf2 >= -999999
 T241_R_f3_k3: Overall_Qual_calc - 1e+06 t241_leaf3 >= -999997
 T241_R_f119_k3: const_Condition_1_Norm - 1e+06 t241_leaf3 >= -999999
 T241_L_f21_k3: const_Bsmt_Full_Bath + 1e+06 t241_leaf3
   <= 1.000001999999e+06
 T241_R_f3_k4: Overall_Qual_calc - 1e+06 t241_leaf4 >= -999997
 T241_R_f119_k4: const_Condition_1_Norm - 1e+06 t241_leaf4 >= -999999
 T241_R_f21_k4: const_Bsmt_Full_Bath - 1e+06 t241_leaf4 >= -999998
 T242_onehot: t242_leaf0 + t242_leaf1 + t242_leaf2 + t242_leaf3
   + t242_leaf4 = 1
 T242_L_f37_k0: x_Open_Porch_SF + 1e+06 t242_leaf0 <= 1.000194999999e+06
 T242_L_f44_k0: const_Yr_Sold + 1e+06 t242_leaf0 <= 1.002006999999e+06
 T242_L_f5_k0: const_Year_Built + 1e+06 t242_leaf0 <= 1.001949999999e+06
 T242_L_f37_k1: x_Open_Porch_SF + 1e+06 t242_leaf1 <= 1.000194999999e+06
 T242_L_f44_k1: const_Yr_Sold + 1e+06 t242_leaf1 <= 1.002006999999e+06
 T242_R_f5_k1: const_Year_Built - 1e+06 t242_leaf1 >= -998050
 T242_L_f37_k2: x_Open_Porch_SF + 1e+06 t242_leaf2 <= 1.000194999999e+06
 T242_R_f44_k2: const_Yr_Sold - 1e+06 t242_leaf2 >= -997993
 T242_L_f1_k2: const_Lot_Area + 1e+06 t242_leaf2 <= 1.004499999999e+06
 T242_L_f37_k3: x_Open_Porch_SF + 1e+06 t242_leaf3 <= 1.000194999999e+06
 T242_R_f44_k3: const_Yr_Sold - 1e+06 t242_leaf3 >= -997993
 T242_R_f1_k3: const_Lot_Area - 1e+06 t242_leaf3 >= -995500
 T242_R_f37_k4: x_Open_Porch_SF - 1e+06 t242_leaf4 >= -999805
 T243_onehot: t243_leaf0 + t243_leaf1 + t243_leaf2 + t243_leaf3
   + t243_leaf4 = 1
 T243_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t243_leaf0
   <= 1.002491999999e+06
 T243_L_f1_k0: const_Lot_Area + 1e+06 t243_leaf0 <= 1.019521999999e+06
 T243_L_f272_k0: const_Mo_Sold_2 + 1e+06 t243_leaf0 <= 1.000000999999e+06
 T243_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t243_leaf1
   <= 1.002491999999e+06
 T243_L_f1_k1: const_Lot_Area + 1e+06 t243_leaf1 <= 1.019521999999e+06
 T243_R_f272_k1: const_Mo_Sold_2 - 1e+06 t243_leaf1 >= -999999
 T243_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t243_leaf2
   <= 1.002491999999e+06
 T243_R_f1_k2: const_Lot_Area - 1e+06 t243_leaf2 >= -980478
 T243_L_f195_k2: x_mvt_is_BrkFace + 1e+06 t243_leaf2 <= 1.000000999999e+06
 T243_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t243_leaf3
   <= 1.002491999999e+06
 T243_R_f1_k3: const_Lot_Area - 1e+06 t243_leaf3 >= -980478
 T243_R_f195_k3: x_mvt_is_BrkFace - 1e+06 t243_leaf3 >= -999999
 T243_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t243_leaf4 >= -997508
 T244_onehot: t244_leaf0 + t244_leaf1 + t244_leaf2 + t244_leaf3 = 1
 T244_L_f17_k0: x_1st_Flr_SF + 1e+06 t244_leaf0 <= 1.002514999999e+06
 T244_L_f68_k0: const_Street_Grvl + 1e+06 t244_leaf0 <= 1.000000999999e+06
 T244_L_f18_k0: const_2nd_Flr_SF + 1e+06 t244_leaf0 <= 1.001628999999e+06
 T244_L_f17_k1: x_1st_Flr_SF + 1e+06 t244_leaf1 <= 1.002514999999e+06
 T244_L_f68_k1: const_Street_Grvl + 1e+06 t244_leaf1 <= 1.000000999999e+06
 T244_R_f18_k1: const_2nd_Flr_SF - 1e+06 t244_leaf1 >= -998371
 T244_L_f17_k2: x_1st_Flr_SF + 1e+06 t244_leaf2 <= 1.002514999999e+06
 T244_R_f68_k2: const_Street_Grvl - 1e+06 t244_leaf2 >= -999999
 T244_R_f17_k3: x_1st_Flr_SF - 1e+06 t244_leaf3 >= -997485
 T245_onehot: t245_leaf0 + t245_leaf1 + t245_leaf2 + t245_leaf3
   + t245_leaf4 + t245_leaf5 + t245_leaf6 = 1
 T245_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t245_leaf0
   <= 1.000000999999e+06
 T245_L_f36_k0: x_Wood_Deck_SF + 1e+06 t245_leaf0 <= 1.000021999999e+06
 T245_L_f38_k0: x_Enclosed_Porch + 1e+06 t245_leaf0 <= 1.000090999999e+06
 T245_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t245_leaf1
   <= 1.000000999999e+06
 T245_L_f36_k1: x_Wood_Deck_SF + 1e+06 t245_leaf1 <= 1.000021999999e+06
 T245_R_f38_k1: x_Enclosed_Porch - 1e+06 t245_leaf1 >= -999909
 T245_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t245_leaf2
   <= 1.000000999999e+06
 T245_R_f36_k2: x_Wood_Deck_SF - 1e+06 t245_leaf2 >= -999978
 T245_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t245_leaf3 >= -999999
 T245_L_f4_k3: const_Overall_Cond + 1e+06 t245_leaf3 <= 1.000005999999e+06
 T245_L_f63_k3: const_MS_Zoning_FV + 1e+06 t245_leaf3
   <= 1.000000999999e+06
 T245_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t245_leaf4 >= -999999
 T245_L_f4_k4: const_Overall_Cond + 1e+06 t245_leaf4 <= 1.000005999999e+06
 T245_R_f63_k4: const_MS_Zoning_FV - 1e+06 t245_leaf4 >= -999999
 T245_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t245_leaf5 >= -999999
 T245_R_f4_k5: const_Overall_Cond - 1e+06 t245_leaf5 >= -999994
 T245_L_f17_k5: x_1st_Flr_SF + 1e+06 t245_leaf5 <= 1.001357999999e+06
 T245_R_f245_k6: const_FunctioNo_aplical_Typ - 1e+06 t245_leaf6 >= -999999
 T245_R_f4_k6: const_Overall_Cond - 1e+06 t245_leaf6 >= -999994
 T245_R_f17_k6: x_1st_Flr_SF - 1e+06 t245_leaf6 >= -998642
 T246_onehot: t246_leaf0 + t246_leaf1 + t246_leaf2 + t246_leaf3 = 1
 T246_L_f20_k0: x_Gr_Liv_Area + 1e+06 t246_leaf0 <= 1.003085999999e+06
 T246_L_f1_k0: const_Lot_Area + 1e+06 t246_leaf0 <= 1.003999999999e+06
 T246_L_f20_k1: x_Gr_Liv_Area + 1e+06 t246_leaf1 <= 1.003085999999e+06
 T246_R_f1_k1: const_Lot_Area - 1e+06 t246_leaf1 >= -996000
 T246_L_f29_k1: const_Fireplaces + 1e+06 t246_leaf1 <= 1.000000999999e+06
 T246_L_f20_k2: x_Gr_Liv_Area + 1e+06 t246_leaf2 <= 1.003085999999e+06
 T246_R_f1_k2: const_Lot_Area - 1e+06 t246_leaf2 >= -996000
 T246_R_f29_k2: const_Fireplaces - 1e+06 t246_leaf2 >= -999999
 T246_R_f20_k3: x_Gr_Liv_Area - 1e+06 t246_leaf3 >= -996914
 T247_onehot: t247_leaf0 + t247_leaf1 + t247_leaf2 + t247_leaf3
   + t247_leaf4 = 1
 T247_L_f27_k0: x_Kitchen_Qual + 1e+06 t247_leaf0 <= 1.000001999999e+06
 T247_L_f233_k0: x_elect_is_FuseA + 1e+06 t247_leaf0 <= 1.000000999999e+06
 T247_L_f27_k1: x_Kitchen_Qual + 1e+06 t247_leaf1 <= 1.000001999999e+06
 T247_R_f233_k1: x_elect_is_FuseA - 1e+06 t247_leaf1 >= -999999
 T247_R_f27_k2: x_Kitchen_Qual - 1e+06 t247_leaf2 >= -999998
 T247_L_f33_k2: x_Garage_Area + 1e+06 t247_leaf2 <= 1.000870999999e+06
 T247_L_f34_k2: x_Garage_Qual + 1e+06 t247_leaf2 <= 1.000001999999e+06
 T247_R_f27_k3: x_Kitchen_Qual - 1e+06 t247_leaf3 >= -999998
 T247_L_f33_k3: x_Garage_Area + 1e+06 t247_leaf3 <= 1.000870999999e+06
 T247_R_f34_k3: x_Garage_Qual - 1e+06 t247_leaf3 >= -999998
 T247_R_f27_k4: x_Kitchen_Qual - 1e+06 t247_leaf4 >= -999998
 T247_R_f33_k4: x_Garage_Area - 1e+06 t247_leaf4 >= -999129
 T248_onehot: t248_leaf0 + t248_leaf1 + t248_leaf2 + t248_leaf3
   + t248_leaf4 + t248_leaf5 + t248_leaf6 + t248_leaf7 = 1
 T248_L_f20_k0: x_Gr_Liv_Area + 1e+06 t248_leaf0 <= 1.000987999999e+06
 T248_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t248_leaf0
   <= 1.000000999999e+06
 T248_L_f38_k0: x_Enclosed_Porch + 1e+06 t248_leaf0 <= 1.000120999999e+06
 T248_L_f20_k1: x_Gr_Liv_Area + 1e+06 t248_leaf1 <= 1.000987999999e+06
 T248_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t248_leaf1
   <= 1.000000999999e+06
 T248_R_f38_k1: x_Enclosed_Porch - 1e+06 t248_leaf1 >= -999879
 T248_L_f20_k2: x_Gr_Liv_Area + 1e+06 t248_leaf2 <= 1.000987999999e+06
 T248_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t248_leaf2
   >= -999999
 T248_L_f35_k2: x_Garage_Cond + 1e+06 t248_leaf2 <= 1.000001999999e+06
 T248_L_f20_k3: x_Gr_Liv_Area + 1e+06 t248_leaf3 <= 1.000987999999e+06
 T248_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t248_leaf3
   >= -999999
 T248_R_f35_k3: x_Garage_Cond - 1e+06 t248_leaf3 >= -999998
 T248_R_f20_k4: x_Gr_Liv_Area - 1e+06 t248_leaf4 >= -999012
 T248_L_f3_k4: Overall_Qual_calc + 1e+06 t248_leaf4 <= 1.000006999999e+06
 T248_L_f18_k4: const_2nd_Flr_SF + 1e+06 t248_leaf4 <= 1.000671999999e+06
 T248_R_f20_k5: x_Gr_Liv_Area - 1e+06 t248_leaf5 >= -999012
 T248_L_f3_k5: Overall_Qual_calc + 1e+06 t248_leaf5 <= 1.000006999999e+06
 T248_R_f18_k5: const_2nd_Flr_SF - 1e+06 t248_leaf5 >= -999328
 T248_R_f20_k6: x_Gr_Liv_Area - 1e+06 t248_leaf6 >= -999012
 T248_R_f3_k6: Overall_Qual_calc - 1e+06 t248_leaf6 >= -999993
 T248_L_f3_k6: Overall_Qual_calc + 1e+06 t248_leaf6 <= 1.000008999999e+06
 T248_R_f20_k7: x_Gr_Liv_Area - 1e+06 t248_leaf7 >= -999012
 T248_R_f3_k7: Overall_Qual_calc - 1e+06 t248_leaf7 >= -999993
 T248_R_f3_k7: Overall_Qual_calc - 1e+06 t248_leaf7 >= -999991
 T249_onehot: t249_leaf0 + t249_leaf1 + t249_leaf2 + t249_leaf3
   + t249_leaf4 = 1
 T249_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t249_leaf0 <= 1.000921999999e+06
 T249_L_f44_k0: const_Yr_Sold + 1e+06 t249_leaf0 <= 1.002007999999e+06
 T249_L_f20_k0: x_Gr_Liv_Area + 1e+06 t249_leaf0 <= 1.002793999999e+06
 T249_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t249_leaf1 <= 1.000921999999e+06
 T249_L_f44_k1: const_Yr_Sold + 1e+06 t249_leaf1 <= 1.002007999999e+06
 T249_R_f20_k1: x_Gr_Liv_Area - 1e+06 t249_leaf1 >= -997206
 T249_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t249_leaf2 <= 1.000921999999e+06
 T249_R_f44_k2: const_Yr_Sold - 1e+06 t249_leaf2 >= -997992
 T249_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t249_leaf2
   <= 1.000000999999e+06
 T249_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t249_leaf3 <= 1.000921999999e+06
 T249_R_f44_k3: const_Yr_Sold - 1e+06 t249_leaf3 >= -997992
 T249_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t249_leaf3
   >= -999999
 T249_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t249_leaf4 >= -999078
 T250_onehot: t250_leaf0 + t250_leaf1 + t250_leaf2 + t250_leaf3 = 1
 T250_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t250_leaf0 <= 1.000701999999e+06
 T250_L_f17_k0: x_1st_Flr_SF + 1e+06 t250_leaf0 <= 1.002199999999e+06
 T250_L_f17_k0: x_1st_Flr_SF + 1e+06 t250_leaf0 <= 1.001564999999e+06
 T250_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t250_leaf1 <= 1.000701999999e+06
 T250_L_f17_k1: x_1st_Flr_SF + 1e+06 t250_leaf1 <= 1.002199999999e+06
 T250_R_f17_k1: x_1st_Flr_SF - 1e+06 t250_leaf1 >= -998435
 T250_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t250_leaf2 <= 1.000701999999e+06
 T250_R_f17_k2: x_1st_Flr_SF - 1e+06 t250_leaf2 >= -997800
 T250_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t250_leaf3 >= -999298
 T251_onehot: t251_leaf0 + t251_leaf1 + t251_leaf2 + t251_leaf3
   + t251_leaf4 = 1
 T251_L_f62_k0: const_MS_Zoning_C_(all) + 1e+06 t251_leaf0
   <= 1.000000999999e+06
 T251_L_f51_k0: const_MS_SubClass_70 + 1e+06 t251_leaf0
   <= 1.000000999999e+06
 T251_L_f3_k0: Overall_Qual_calc + 1e+06 t251_leaf0 <= 1.000004999999e+06
 T251_L_f62_k1: const_MS_Zoning_C_(all) + 1e+06 t251_leaf1
   <= 1.000000999999e+06
 T251_L_f51_k1: const_MS_SubClass_70 + 1e+06 t251_leaf1
   <= 1.000000999999e+06
 T251_R_f3_k1: Overall_Qual_calc - 1e+06 t251_leaf1 >= -999995
 T251_L_f62_k2: const_MS_Zoning_C_(all) + 1e+06 t251_leaf2
   <= 1.000000999999e+06
 T251_R_f51_k2: const_MS_SubClass_70 - 1e+06 t251_leaf2 >= -999999
 T251_L_f5_k2: const_Year_Built + 1e+06 t251_leaf2 <= 1.001891999999e+06
 T251_L_f62_k3: const_MS_Zoning_C_(all) + 1e+06 t251_leaf3
   <= 1.000000999999e+06
 T251_R_f51_k3: const_MS_SubClass_70 - 1e+06 t251_leaf3 >= -999999
 T251_R_f5_k3: const_Year_Built - 1e+06 t251_leaf3 >= -998108
 T251_R_f62_k4: const_MS_Zoning_C_(all) - 1e+06 t251_leaf4 >= -999999
 T252_onehot: t252_leaf0 + t252_leaf1 + t252_leaf2 + t252_leaf3
   + t252_leaf4 + t252_leaf5 + t252_leaf6 + t252_leaf7 = 1
 T252_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t252_leaf0
   <= 1.000000999999e+06
 T252_L_f38_k0: x_Enclosed_Porch + 1e+06 t252_leaf0 <= 1.000090999999e+06
 T252_L_f36_k0: x_Wood_Deck_SF + 1e+06 t252_leaf0 <= 1.000021999999e+06
 T252_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t252_leaf1
   <= 1.000000999999e+06
 T252_L_f38_k1: x_Enclosed_Porch + 1e+06 t252_leaf1 <= 1.000090999999e+06
 T252_R_f36_k1: x_Wood_Deck_SF - 1e+06 t252_leaf1 >= -999978
 T252_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t252_leaf2
   <= 1.000000999999e+06
 T252_R_f38_k2: x_Enclosed_Porch - 1e+06 t252_leaf2 >= -999909
 T252_L_f169_k2: x_ex1_is_MetalSd + 1e+06 t252_leaf2 <= 1.000000999999e+06
 T252_L_f245_k3: const_FunctioNo_aplical_Typ + 1e+06 t252_leaf3
   <= 1.000000999999e+06
 T252_R_f38_k3: x_Enclosed_Porch - 1e+06 t252_leaf3 >= -999909
 T252_R_f169_k3: x_ex1_is_MetalSd - 1e+06 t252_leaf3 >= -999999
 T252_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t252_leaf4 >= -999999
 T252_L_f296_k4: const_Sale_Condition_Family + 1e+06 t252_leaf4
   <= 1.000000999999e+06
 T252_L_f92_k4: const_Neighborhood_BrkSide + 1e+06 t252_leaf4
   <= 1.000000999999e+06
 T252_R_f245_k5: const_FunctioNo_aplical_Typ - 1e+06 t252_leaf5 >= -999999
 T252_L_f296_k5: const_Sale_Condition_Family + 1e+06 t252_leaf5
   <= 1.000000999999e+06
 T252_R_f92_k5: const_Neighborhood_BrkSide - 1e+06 t252_leaf5 >= -999999
 T252_R_f245_k6: const_FunctioNo_aplical_Typ - 1e+06 t252_leaf6 >= -999999
 T252_R_f296_k6: const_Sale_Condition_Family - 1e+06 t252_leaf6 >= -999999
 T252_L_f33_k6: x_Garage_Area + 1e+06 t252_leaf6 <= 1.000505999999e+06
 T252_R_f245_k7: const_FunctioNo_aplical_Typ - 1e+06 t252_leaf7 >= -999999
 T252_R_f296_k7: const_Sale_Condition_Family - 1e+06 t252_leaf7 >= -999999
 T252_R_f33_k7: x_Garage_Area - 1e+06 t252_leaf7 >= -999494
 T253_onehot: t253_leaf0 + t253_leaf1 + t253_leaf2 + t253_leaf3
   + t253_leaf4 + t253_leaf5 = 1
 T253_L_f1_k0: const_Lot_Area + 1e+06 t253_leaf0 <= 1.023256999999e+06
 T253_L_f1_k0: const_Lot_Area + 1e+06 t253_leaf0 <= 1.018385999999e+06
 T253_L_f72_k0: const_Alley_Pave + 1e+06 t253_leaf0 <= 1.000000999999e+06
 T253_L_f1_k1: const_Lot_Area + 1e+06 t253_leaf1 <= 1.023256999999e+06
 T253_L_f1_k1: const_Lot_Area + 1e+06 t253_leaf1 <= 1.018385999999e+06
 T253_R_f72_k1: const_Alley_Pave - 1e+06 t253_leaf1 >= -999999
 T253_L_f1_k2: const_Lot_Area + 1e+06 t253_leaf2 <= 1.023256999999e+06
 T253_R_f1_k2: const_Lot_Area - 1e+06 t253_leaf2 >= -981614
 T253_R_f1_k3: const_Lot_Area - 1e+06 t253_leaf3 >= -976743
 T253_L_f20_k3: x_Gr_Liv_Area + 1e+06 t253_leaf3 <= 1.002543999999e+06
 T253_L_f20_k3: x_Gr_Liv_Area + 1e+06 t253_leaf3 <= 1.001490999999e+06
 T253_R_f1_k4: const_Lot_Area - 1e+06 t253_leaf4 >= -976743
 T253_L_f20_k4: x_Gr_Liv_Area + 1e+06 t253_leaf4 <= 1.002543999999e+06
 T253_R_f20_k4: x_Gr_Liv_Area - 1e+06 t253_leaf4 >= -998509
 T253_R_f1_k5: const_Lot_Area - 1e+06 t253_leaf5 >= -976743
 T253_R_f20_k5: x_Gr_Liv_Area - 1e+06 t253_leaf5 >= -997456
 T254_onehot: t254_leaf0 + t254_leaf1 + t254_leaf2 + t254_leaf3 = 1
 T254_L_f3_k0: Overall_Qual_calc + 1e+06 t254_leaf0 <= 1.000002999999e+06
 T254_R_f3_k1: Overall_Qual_calc - 1e+06 t254_leaf1 >= -999997
 T254_L_f63_k1: const_MS_Zoning_FV + 1e+06 t254_leaf1
   <= 1.000000999999e+06
 T254_L_f37_k1: x_Open_Porch_SF + 1e+06 t254_leaf1 <= 1.000290999999e+06
 T254_R_f3_k2: Overall_Qual_calc - 1e+06 t254_leaf2 >= -999997
 T254_L_f63_k2: const_MS_Zoning_FV + 1e+06 t254_leaf2
   <= 1.000000999999e+06
 T254_R_f37_k2: x_Open_Porch_SF - 1e+06 t254_leaf2 >= -999709
 T254_R_f3_k3: Overall_Qual_calc - 1e+06 t254_leaf3 >= -999997
 T254_R_f63_k3: const_MS_Zoning_FV - 1e+06 t254_leaf3 >= -999999
 T255_onehot: t255_leaf0 + t255_leaf1 + t255_leaf2 + t255_leaf3
   + t255_leaf4 + t255_leaf5 + t255_leaf6 = 1
 T255_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t255_leaf0 <= 1.000000999999e+06
 T255_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t255_leaf0 <= 1.000002999999e+06
 T255_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t255_leaf0 <= 1.001509999999e+06
 T255_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t255_leaf1 <= 1.000000999999e+06
 T255_L_f25_k1: x_Bedroom_AbvGr + 1e+06 t255_leaf1 <= 1.000002999999e+06
 T255_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t255_leaf1 >= -998490
 T255_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t255_leaf2 <= 1.000000999999e+06
 T255_R_f25_k2: x_Bedroom_AbvGr - 1e+06 t255_leaf2 >= -999997
 T255_L_f17_k2: x_1st_Flr_SF + 1e+06 t255_leaf2 <= 1.001401999999e+06
 T255_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t255_leaf3 <= 1.000000999999e+06
 T255_R_f25_k3: x_Bedroom_AbvGr - 1e+06 t255_leaf3 >= -999997
 T255_R_f17_k3: x_1st_Flr_SF - 1e+06 t255_leaf3 >= -998598
 T255_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t255_leaf4 >= -999999
 T255_L_f31_k4: const_Garage_Yr_Blt + 1e+06 t255_leaf4
   <= 1.001922999999e+06
 T255_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t255_leaf5 >= -999999
 T255_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t255_leaf5 >= -998077
 T255_L_f1_k5: const_Lot_Area + 1e+06 t255_leaf5 <= 1.009604999999e+06
 T255_R_f161_k6: x_ex1_is_AsbShng - 1e+06 t255_leaf6 >= -999999
 T255_R_f31_k6: const_Garage_Yr_Blt - 1e+06 t255_leaf6 >= -998077
 T255_R_f1_k6: const_Lot_Area - 1e+06 t255_leaf6 >= -990395
 T256_onehot: t256_leaf0 + t256_leaf1 + t256_leaf2 + t256_leaf3
   + t256_leaf4 + t256_leaf5 + t256_leaf6 = 1
 T256_L_f46_k0: const_MS_SubClass_30 + 1e+06 t256_leaf0
   <= 1.000000999999e+06
 T256_L_f86_k0: const_Land_Slope_Gtl + 1e+06 t256_leaf0
   <= 1.000000999999e+06
 T256_L_f46_k1: const_MS_SubClass_30 + 1e+06 t256_leaf1
   <= 1.000000999999e+06
 T256_R_f86_k1: const_Land_Slope_Gtl - 1e+06 t256_leaf1 >= -999999
 T256_L_f5_k1: const_Year_Built + 1e+06 t256_leaf1 <= 1.001910999999e+06
 T256_L_f46_k2: const_MS_SubClass_30 + 1e+06 t256_leaf2
   <= 1.000000999999e+06
 T256_R_f86_k2: const_Land_Slope_Gtl - 1e+06 t256_leaf2 >= -999999
 T256_R_f5_k2: const_Year_Built - 1e+06 t256_leaf2 >= -998089
 T256_R_f46_k3: const_MS_SubClass_30 - 1e+06 t256_leaf3 >= -999999
 T256_L_f33_k3: x_Garage_Area + 1e+06 t256_leaf3 <= 1.000496999999e+06
 T256_L_f1_k3: const_Lot_Area + 1e+06 t256_leaf3 <= 1.008093999999e+06
 T256_R_f46_k4: const_MS_SubClass_30 - 1e+06 t256_leaf4 >= -999999
 T256_L_f33_k4: x_Garage_Area + 1e+06 t256_leaf4 <= 1.000496999999e+06
 T256_R_f1_k4: const_Lot_Area - 1e+06 t256_leaf4 >= -991906
 T256_R_f46_k5: const_MS_SubClass_30 - 1e+06 t256_leaf5 >= -999999
 T256_R_f33_k5: x_Garage_Area - 1e+06 t256_leaf5 >= -999503
 T256_L_f37_k5: x_Open_Porch_SF + 1e+06 t256_leaf5 <= 1.000003999999e+06
 T256_R_f46_k6: const_MS_SubClass_30 - 1e+06 t256_leaf6 >= -999999
 T256_R_f33_k6: x_Garage_Area - 1e+06 t256_leaf6 >= -999503
 T256_R_f37_k6: x_Open_Porch_SF - 1e+06 t256_leaf6 >= -999996
 T257_onehot: t257_leaf0 + t257_leaf1 + t257_leaf2 + t257_leaf3
   + t257_leaf4 + t257_leaf5 + t257_leaf6 = 1
 T257_L_f20_k0: x_Gr_Liv_Area + 1e+06 t257_leaf0 <= 1.000987999999e+06
 T257_L_f33_k0: x_Garage_Area + 1e+06 t257_leaf0 <= 1.000215999999e+06
 T257_L_f231_k0: central_air_no + 1e+06 t257_leaf0 <= 1.000000999999e+06
 T257_L_f20_k1: x_Gr_Liv_Area + 1e+06 t257_leaf1 <= 1.000987999999e+06
 T257_L_f33_k1: x_Garage_Area + 1e+06 t257_leaf1 <= 1.000215999999e+06
 T257_R_f231_k1: central_air_no - 1e+06 t257_leaf1 >= -999999
 T257_L_f20_k2: x_Gr_Liv_Area + 1e+06 t257_leaf2 <= 1.000987999999e+06
 T257_R_f33_k2: x_Garage_Area - 1e+06 t257_leaf2 >= -999784
 T257_L_f9_k2: x_Exter_Cond + 1e+06 t257_leaf2 <= 1.000001999999e+06
 T257_L_f20_k3: x_Gr_Liv_Area + 1e+06 t257_leaf3 <= 1.000987999999e+06
 T257_R_f33_k3: x_Garage_Area - 1e+06 t257_leaf3 >= -999784
 T257_R_f9_k3: x_Exter_Cond - 1e+06 t257_leaf3 >= -999998
 T257_R_f20_k4: x_Gr_Liv_Area - 1e+06 t257_leaf4 >= -999012
 T257_L_f4_k4: const_Overall_Cond + 1e+06 t257_leaf4 <= 1.000003999999e+06
 T257_R_f20_k5: x_Gr_Liv_Area - 1e+06 t257_leaf5 >= -999012
 T257_R_f4_k5: const_Overall_Cond - 1e+06 t257_leaf5 >= -999996
 T257_L_f7_k5: x_Mas_Vnr_Area + 1e+06 t257_leaf5 <= 1.000672999999e+06
 T257_R_f20_k6: x_Gr_Liv_Area - 1e+06 t257_leaf6 >= -999012
 T257_R_f4_k6: const_Overall_Cond - 1e+06 t257_leaf6 >= -999996
 T257_R_f7_k6: x_Mas_Vnr_Area - 1e+06 t257_leaf6 >= -999327
 T258_onehot: t258_leaf0 + t258_leaf1 + t258_leaf2 + t258_leaf3
   + t258_leaf4 = 1
 T258_L_f5_k0: const_Year_Built + 1e+06 t258_leaf0 <= 1.001913999999e+06
 T258_R_f5_k1: const_Year_Built - 1e+06 t258_leaf1 >= -998086
 T258_L_f231_k1: central_air_no + 1e+06 t258_leaf1 <= 1.000000999999e+06
 T258_L_f63_k1: const_MS_Zoning_FV + 1e+06 t258_leaf1
   <= 1.000000999999e+06
 T258_R_f5_k2: const_Year_Built - 1e+06 t258_leaf2 >= -998086
 T258_L_f231_k2: central_air_no + 1e+06 t258_leaf2 <= 1.000000999999e+06
 T258_R_f63_k2: const_MS_Zoning_FV - 1e+06 t258_leaf2 >= -999999
 T258_R_f5_k3: const_Year_Built - 1e+06 t258_leaf3 >= -998086
 T258_R_f231_k3: central_air_no - 1e+06 t258_leaf3 >= -999999
 T258_L_f185_k3: x_ex2_is_MetalSd + 1e+06 t258_leaf3 <= 1.000000999999e+06
 T258_R_f5_k4: const_Year_Built - 1e+06 t258_leaf4 >= -998086
 T258_R_f231_k4: central_air_no - 1e+06 t258_leaf4 >= -999999
 T258_R_f185_k4: x_ex2_is_MetalSd - 1e+06 t258_leaf4 >= -999999
 T259_onehot: t259_leaf0 + t259_leaf1 + t259_leaf2 + t259_leaf3
   + t259_leaf4 = 1
 T259_L_f18_k0: const_2nd_Flr_SF + 1e+06 t259_leaf0 <= 1.001295999999e+06
 T259_L_f4_k0: const_Overall_Cond + 1e+06 t259_leaf0 <= 1.000005999999e+06
 T259_L_f16_k0: x_Heating_QC + 1e+06 t259_leaf0 <= 1.000002999999e+06
 T259_L_f18_k1: const_2nd_Flr_SF + 1e+06 t259_leaf1 <= 1.001295999999e+06
 T259_L_f4_k1: const_Overall_Cond + 1e+06 t259_leaf1 <= 1.000005999999e+06
 T259_R_f16_k1: x_Heating_QC - 1e+06 t259_leaf1 >= -999997
 T259_L_f18_k2: const_2nd_Flr_SF + 1e+06 t259_leaf2 <= 1.001295999999e+06
 T259_R_f4_k2: const_Overall_Cond - 1e+06 t259_leaf2 >= -999994
 T259_L_f18_k2: const_2nd_Flr_SF + 1e+06 t259_leaf2 <= 1.000166999999e+06
 T259_L_f18_k3: const_2nd_Flr_SF + 1e+06 t259_leaf3 <= 1.001295999999e+06
 T259_R_f4_k3: const_Overall_Cond - 1e+06 t259_leaf3 >= -999994
 T259_R_f18_k3: const_2nd_Flr_SF - 1e+06 t259_leaf3 >= -999833
 T259_R_f18_k4: const_2nd_Flr_SF - 1e+06 t259_leaf4 >= -998704
 T260_onehot: t260_leaf0 + t260_leaf1 + t260_leaf2 + t260_leaf3
   + t260_leaf4 + t260_leaf5 = 1
 T260_L_f0_k0: const_Lot_Frontage + 1e+06 t260_leaf0 <= 1.000120999999e+06
 T260_L_f20_k0: x_Gr_Liv_Area + 1e+06 t260_leaf0 <= 1.000893999999e+06
 T260_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t260_leaf0
   <= 1.000000999999e+06
 T260_L_f0_k1: const_Lot_Frontage + 1e+06 t260_leaf1 <= 1.000120999999e+06
 T260_L_f20_k1: x_Gr_Liv_Area + 1e+06 t260_leaf1 <= 1.000893999999e+06
 T260_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t260_leaf1
   >= -999999
 T260_L_f0_k2: const_Lot_Frontage + 1e+06 t260_leaf2 <= 1.000120999999e+06
 T260_R_f20_k2: x_Gr_Liv_Area - 1e+06 t260_leaf2 >= -999106
 T260_L_f194_k2: x_mvt_is_BrkCmn + 1e+06 t260_leaf2 <= 1.000000999999e+06
 T260_L_f0_k3: const_Lot_Frontage + 1e+06 t260_leaf3 <= 1.000120999999e+06
 T260_R_f20_k3: x_Gr_Liv_Area - 1e+06 t260_leaf3 >= -999106
 T260_R_f194_k3: x_mvt_is_BrkCmn - 1e+06 t260_leaf3 >= -999999
 T260_R_f0_k4: const_Lot_Frontage - 1e+06 t260_leaf4 >= -999879
 T260_L_f14_k4: x_Bsmt_Unf_SF + 1e+06 t260_leaf4 <= 1.000447999999e+06
 T260_R_f0_k5: const_Lot_Frontage - 1e+06 t260_leaf5 >= -999879
 T260_R_f14_k5: x_Bsmt_Unf_SF - 1e+06 t260_leaf5 >= -999552
 T261_onehot: t261_leaf0 + t261_leaf1 + t261_leaf2 + t261_leaf3 = 1
 T261_L_f17_k0: x_1st_Flr_SF + 1e+06 t261_leaf0 <= 1.002514999999e+06
 T261_L_f1_k0: const_Lot_Area + 1e+06 t261_leaf0 <= 1.027696999999e+06
 T261_L_f46_k0: const_MS_SubClass_30 + 1e+06 t261_leaf0
   <= 1.000000999999e+06
 T261_L_f17_k1: x_1st_Flr_SF + 1e+06 t261_leaf1 <= 1.002514999999e+06
 T261_L_f1_k1: const_Lot_Area + 1e+06 t261_leaf1 <= 1.027696999999e+06
 T261_R_f46_k1: const_MS_SubClass_30 - 1e+06 t261_leaf1 >= -999999
 T261_L_f17_k2: x_1st_Flr_SF + 1e+06 t261_leaf2 <= 1.002514999999e+06
 T261_R_f1_k2: const_Lot_Area - 1e+06 t261_leaf2 >= -972303
 T261_R_f17_k3: x_1st_Flr_SF - 1e+06 t261_leaf3 >= -997485
 T262_onehot: t262_leaf0 + t262_leaf1 + t262_leaf2 + t262_leaf3
   + t262_leaf4 = 1
 T262_L_f3_k0: Overall_Qual_calc + 1e+06 t262_leaf0 <= 1.000002999999e+06
 T262_R_f3_k1: Overall_Qual_calc - 1e+06 t262_leaf1 >= -999997
 T262_L_f26_k1: x_Kitchen_AbvGr + 1e+06 t262_leaf1 <= 1.000001999999e+06
 T262_L_f20_k1: x_Gr_Liv_Area + 1e+06 t262_leaf1 <= 1.002648999999e+06
 T262_R_f3_k2: Overall_Qual_calc - 1e+06 t262_leaf2 >= -999997
 T262_L_f26_k2: x_Kitchen_AbvGr + 1e+06 t262_leaf2 <= 1.000001999999e+06
 T262_R_f20_k2: x_Gr_Liv_Area - 1e+06 t262_leaf2 >= -997351
 T262_R_f3_k3: Overall_Qual_calc - 1e+06 t262_leaf3 >= -999997
 T262_R_f26_k3: x_Kitchen_AbvGr - 1e+06 t262_leaf3 >= -999998
 T262_L_f35_k3: x_Garage_Cond + 1e+06 t262_leaf3 <= 1.000000999999e+06
 T262_R_f3_k4: Overall_Qual_calc - 1e+06 t262_leaf4 >= -999997
 T262_R_f26_k4: x_Kitchen_AbvGr - 1e+06 t262_leaf4 >= -999998
 T262_R_f35_k4: x_Garage_Cond - 1e+06 t262_leaf4 >= -999999
 T263_onehot: t263_leaf0 + t263_leaf1 + t263_leaf2 + t263_leaf3
   + t263_leaf4 + t263_leaf5 = 1
 T263_L_f272_k0: const_Mo_Sold_2 + 1e+06 t263_leaf0 <= 1.000000999999e+06
 T263_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t263_leaf0
   <= 1.000000999999e+06
 T263_L_f20_k0: x_Gr_Liv_Area + 1e+06 t263_leaf0 <= 1.000809999999e+06
 T263_L_f272_k1: const_Mo_Sold_2 + 1e+06 t263_leaf1 <= 1.000000999999e+06
 T263_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t263_leaf1
   <= 1.000000999999e+06
 T263_R_f20_k1: x_Gr_Liv_Area - 1e+06 t263_leaf1 >= -999190
 T263_L_f272_k2: const_Mo_Sold_2 + 1e+06 t263_leaf2 <= 1.000000999999e+06
 T263_R_f95_k2: const_Neighborhood_Crawfor - 1e+06 t263_leaf2 >= -999999
 T263_R_f272_k3: const_Mo_Sold_2 - 1e+06 t263_leaf3 >= -999999
 T263_L_f1_k3: const_Lot_Area + 1e+06 t263_leaf3 <= 1.009964999999e+06
 T263_R_f272_k4: const_Mo_Sold_2 - 1e+06 t263_leaf4 >= -999999
 T263_R_f1_k4: const_Lot_Area - 1e+06 t263_leaf4 >= -990035
 T263_L_f297_k4: const_Sale_Condition_Normal + 1e+06 t263_leaf4
   <= 1.000000999999e+06
 T263_R_f272_k5: const_Mo_Sold_2 - 1e+06 t263_leaf5 >= -999999
 T263_R_f1_k5: const_Lot_Area - 1e+06 t263_leaf5 >= -990035
 T263_R_f297_k5: const_Sale_Condition_Normal - 1e+06 t263_leaf5 >= -999999
 T264_onehot: t264_leaf0 + t264_leaf1 + t264_leaf2 + t264_leaf3
   + t264_leaf4 = 1
 T264_L_f3_k0: Overall_Qual_calc + 1e+06 t264_leaf0 <= 1.000002999999e+06
 T264_R_f3_k1: Overall_Qual_calc - 1e+06 t264_leaf1 >= -999997
 T264_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t264_leaf1
   <= 1.000000999999e+06
 T264_L_f37_k1: x_Open_Porch_SF + 1e+06 t264_leaf1 <= 1.000181999999e+06
 T264_R_f3_k2: Overall_Qual_calc - 1e+06 t264_leaf2 >= -999997
 T264_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t264_leaf2
   <= 1.000000999999e+06
 T264_R_f37_k2: x_Open_Porch_SF - 1e+06 t264_leaf2 >= -999818
 T264_R_f3_k3: Overall_Qual_calc - 1e+06 t264_leaf3 >= -999997
 T264_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t264_leaf3 >= -999999
 T264_L_f296_k3: const_Sale_Condition_Family + 1e+06 t264_leaf3
   <= 1.000000999999e+06
 T264_R_f3_k4: Overall_Qual_calc - 1e+06 t264_leaf4 >= -999997
 T264_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t264_leaf4 >= -999999
 T264_R_f296_k4: const_Sale_Condition_Family - 1e+06 t264_leaf4 >= -999999
 T265_onehot: t265_leaf0 + t265_leaf1 + t265_leaf2 + t265_leaf3
   + t265_leaf4 = 1
 T265_L_f0_k0: const_Lot_Frontage + 1e+06 t265_leaf0 <= 1.000151999999e+06
 T265_L_f3_k0: Overall_Qual_calc + 1e+06 t265_leaf0 <= 1.000004999999e+06
 T265_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t265_leaf0 <= 1.000000999999e+06
 T265_L_f0_k1: const_Lot_Frontage + 1e+06 t265_leaf1 <= 1.000151999999e+06
 T265_L_f3_k1: Overall_Qual_calc + 1e+06 t265_leaf1 <= 1.000004999999e+06
 T265_R_f161_k1: x_ex1_is_AsbShng - 1e+06 t265_leaf1 >= -999999
 T265_L_f0_k2: const_Lot_Frontage + 1e+06 t265_leaf2 <= 1.000151999999e+06
 T265_R_f3_k2: Overall_Qual_calc - 1e+06 t265_leaf2 >= -999995
 T265_L_f5_k2: const_Year_Built + 1e+06 t265_leaf2 <= 1.001912999999e+06
 T265_L_f0_k3: const_Lot_Frontage + 1e+06 t265_leaf3 <= 1.000151999999e+06
 T265_R_f3_k3: Overall_Qual_calc - 1e+06 t265_leaf3 >= -999995
 T265_R_f5_k3: const_Year_Built - 1e+06 t265_leaf3 >= -998087
 T265_R_f0_k4: const_Lot_Frontage - 1e+06 t265_leaf4 >= -999848
 T266_onehot: t266_leaf0 + t266_leaf1 + t266_leaf2 + t266_leaf3
   + t266_leaf4 + t266_leaf5 + t266_leaf6 = 1
 T266_L_f207_k0: const_Bsmt_Exposure_Gd + 1e+06 t266_leaf0
   <= 1.000000999999e+06
 T266_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t266_leaf0 <= 1.000000999999e+06
 T266_L_f33_k0: x_Garage_Area + 1e+06 t266_leaf0 <= 1.000810999999e+06
 T266_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t266_leaf1
   <= 1.000000999999e+06
 T266_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t266_leaf1 <= 1.000000999999e+06
 T266_R_f33_k1: x_Garage_Area - 1e+06 t266_leaf1 >= -999189
 T266_L_f207_k2: const_Bsmt_Exposure_Gd + 1e+06 t266_leaf2
   <= 1.000000999999e+06
 T266_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t266_leaf2 >= -999999
 T266_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t266_leaf2
   <= 1.000726999999e+06
 T266_L_f207_k3: const_Bsmt_Exposure_Gd + 1e+06 t266_leaf3
   <= 1.000000999999e+06
 T266_R_f161_k3: x_ex1_is_AsbShng - 1e+06 t266_leaf3 >= -999999
 T266_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t266_leaf3 >= -999273
 T266_R_f207_k4: const_Bsmt_Exposure_Gd - 1e+06 t266_leaf4 >= -999999
 T266_L_f36_k4: x_Wood_Deck_SF + 1e+06 t266_leaf4 <= 1.000510999999e+06
 T266_L_f29_k4: const_Fireplaces + 1e+06 t266_leaf4 <= 1.000001999999e+06
 T266_R_f207_k5: const_Bsmt_Exposure_Gd - 1e+06 t266_leaf5 >= -999999
 T266_L_f36_k5: x_Wood_Deck_SF + 1e+06 t266_leaf5 <= 1.000510999999e+06
 T266_R_f29_k5: const_Fireplaces - 1e+06 t266_leaf5 >= -999998
 T266_R_f207_k6: const_Bsmt_Exposure_Gd - 1e+06 t266_leaf6 >= -999999
 T266_R_f36_k6: x_Wood_Deck_SF - 1e+06 t266_leaf6 >= -999489
 T267_onehot: t267_leaf0 + t267_leaf1 + t267_leaf2 + t267_leaf3
   + t267_leaf4 + t267_leaf5 = 1
 T267_L_f3_k0: Overall_Qual_calc + 1e+06 t267_leaf0 <= 1.000006999999e+06
 T267_L_f18_k0: const_2nd_Flr_SF + 1e+06 t267_leaf0 <= 1.001275999999e+06
 T267_L_f37_k0: x_Open_Porch_SF + 1e+06 t267_leaf0 <= 1.000020999999e+06
 T267_L_f3_k1: Overall_Qual_calc + 1e+06 t267_leaf1 <= 1.000006999999e+06
 T267_L_f18_k1: const_2nd_Flr_SF + 1e+06 t267_leaf1 <= 1.001275999999e+06
 T267_R_f37_k1: x_Open_Porch_SF - 1e+06 t267_leaf1 >= -999979
 T267_L_f3_k2: Overall_Qual_calc + 1e+06 t267_leaf2 <= 1.000006999999e+06
 T267_R_f18_k2: const_2nd_Flr_SF - 1e+06 t267_leaf2 >= -998724
 T267_R_f3_k3: Overall_Qual_calc - 1e+06 t267_leaf3 >= -999993
 T267_L_f296_k3: const_Sale_Condition_Family + 1e+06 t267_leaf3
   <= 1.000000999999e+06
 T267_L_f0_k3: const_Lot_Frontage + 1e+06 t267_leaf3 <= 1.000113999999e+06
 T267_R_f3_k4: Overall_Qual_calc - 1e+06 t267_leaf4 >= -999993
 T267_L_f296_k4: const_Sale_Condition_Family + 1e+06 t267_leaf4
   <= 1.000000999999e+06
 T267_R_f0_k4: const_Lot_Frontage - 1e+06 t267_leaf4 >= -999886
 T267_R_f3_k5: Overall_Qual_calc - 1e+06 t267_leaf5 >= -999993
 T267_R_f296_k5: const_Sale_Condition_Family - 1e+06 t267_leaf5 >= -999999
 T268_onehot: t268_leaf0 + t268_leaf1 + t268_leaf2 + t268_leaf3
   + t268_leaf4 + t268_leaf5 = 1
 T268_L_f37_k0: x_Open_Porch_SF + 1e+06 t268_leaf0 <= 1.000281999999e+06
 T268_L_f29_k0: const_Fireplaces + 1e+06 t268_leaf0 <= 1.000001999999e+06
 T268_L_f185_k0: x_ex2_is_MetalSd + 1e+06 t268_leaf0 <= 1.000000999999e+06
 T268_L_f37_k1: x_Open_Porch_SF + 1e+06 t268_leaf1 <= 1.000281999999e+06
 T268_L_f29_k1: const_Fireplaces + 1e+06 t268_leaf1 <= 1.000001999999e+06
 T268_R_f185_k1: x_ex2_is_MetalSd - 1e+06 t268_leaf1 >= -999999
 T268_L_f37_k2: x_Open_Porch_SF + 1e+06 t268_leaf2 <= 1.000281999999e+06
 T268_R_f29_k2: const_Fireplaces - 1e+06 t268_leaf2 >= -999998
 T268_L_f37_k2: x_Open_Porch_SF + 1e+06 t268_leaf2 <= 1.000030999999e+06
 T268_L_f37_k3: x_Open_Porch_SF + 1e+06 t268_leaf3 <= 1.000281999999e+06
 T268_R_f29_k3: const_Fireplaces - 1e+06 t268_leaf3 >= -999998
 T268_R_f37_k3: x_Open_Porch_SF - 1e+06 t268_leaf3 >= -999969
 T268_R_f37_k4: x_Open_Porch_SF - 1e+06 t268_leaf4 >= -999718
 T268_L_f256_k4: const_Garage_Finish_Unf + 1e+06 t268_leaf4
   <= 1.000000999999e+06
 T268_R_f37_k5: x_Open_Porch_SF - 1e+06 t268_leaf5 >= -999718
 T268_R_f256_k5: const_Garage_Finish_Unf - 1e+06 t268_leaf5 >= -999999
 T269_onehot: t269_leaf0 + t269_leaf1 + t269_leaf2 + t269_leaf3
   + t269_leaf4 = 1
 T269_L_f18_k0: const_2nd_Flr_SF + 1e+06 t269_leaf0 <= 1.001610999999e+06
 T269_L_f0_k0: const_Lot_Frontage + 1e+06 t269_leaf0 <= 1.000116999999e+06
 T269_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t269_leaf0
   <= 1.002007999999e+06
 T269_L_f18_k1: const_2nd_Flr_SF + 1e+06 t269_leaf1 <= 1.001610999999e+06
 T269_L_f0_k1: const_Lot_Frontage + 1e+06 t269_leaf1 <= 1.000116999999e+06
 T269_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t269_leaf1 >= -997992
 T269_L_f18_k2: const_2nd_Flr_SF + 1e+06 t269_leaf2 <= 1.001610999999e+06
 T269_R_f0_k2: const_Lot_Frontage - 1e+06 t269_leaf2 >= -999883
 T269_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t269_leaf2
   <= 1.001965999999e+06
 T269_L_f18_k3: const_2nd_Flr_SF + 1e+06 t269_leaf3 <= 1.001610999999e+06
 T269_R_f0_k3: const_Lot_Frontage - 1e+06 t269_leaf3 >= -999883
 T269_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t269_leaf3 >= -998034
 T269_R_f18_k4: const_2nd_Flr_SF - 1e+06 t269_leaf4 >= -998389
 T270_onehot: t270_leaf0 + t270_leaf1 + t270_leaf2 + t270_leaf3
   + t270_leaf4 + t270_leaf5 = 1
 T270_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t270_leaf0 <= 1.001731999999e+06
 T270_L_f9_k0: x_Exter_Cond + 1e+06 t270_leaf0 <= 1.000001999999e+06
 T270_L_f6_k0: const_Year_Remod_Add + 1e+06 t270_leaf0
   <= 1.001951999999e+06
 T270_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t270_leaf1 <= 1.001731999999e+06
 T270_L_f9_k1: x_Exter_Cond + 1e+06 t270_leaf1 <= 1.000001999999e+06
 T270_R_f6_k1: const_Year_Remod_Add - 1e+06 t270_leaf1 >= -998048
 T270_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t270_leaf2 <= 1.001731999999e+06
 T270_R_f9_k2: x_Exter_Cond - 1e+06 t270_leaf2 >= -999998
 T270_L_f250_k2: const_Garage_Type_CarPort + 1e+06 t270_leaf2
   <= 1.000000999999e+06
 T270_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t270_leaf3 <= 1.001731999999e+06
 T270_R_f9_k3: x_Exter_Cond - 1e+06 t270_leaf3 >= -999998
 T270_R_f250_k3: const_Garage_Type_CarPort - 1e+06 t270_leaf3 >= -999999
 T270_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t270_leaf4 >= -998268
 T270_L_f7_k4: x_Mas_Vnr_Area + 1e+06 t270_leaf4 <= 1.000566999999e+06
 T270_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t270_leaf5 >= -998268
 T270_R_f7_k5: x_Mas_Vnr_Area - 1e+06 t270_leaf5 >= -999433
 T271_onehot: t271_leaf0 + t271_leaf1 + t271_leaf2 + t271_leaf3
   + t271_leaf4 + t271_leaf5 = 1
 T271_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t271_leaf0 <= 1.000787999999e+06
 T271_L_f3_k0: Overall_Qual_calc + 1e+06 t271_leaf0 <= 1.000003999999e+06
 T271_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t271_leaf0 <= 1.000265999999e+06
 T271_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t271_leaf1 <= 1.000787999999e+06
 T271_L_f3_k1: Overall_Qual_calc + 1e+06 t271_leaf1 <= 1.000003999999e+06
 T271_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t271_leaf1 >= -999734
 T271_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t271_leaf2 <= 1.000787999999e+06
 T271_R_f3_k2: Overall_Qual_calc - 1e+06 t271_leaf2 >= -999996
 T271_L_f298_k2: const_Sale_Condition_Partial + 1e+06 t271_leaf2
   <= 1.000000999999e+06
 T271_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t271_leaf3 <= 1.000787999999e+06
 T271_R_f3_k3: Overall_Qual_calc - 1e+06 t271_leaf3 >= -999996
 T271_R_f298_k3: const_Sale_Condition_Partial - 1e+06 t271_leaf3
   >= -999999
 T271_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t271_leaf4 >= -999212
 T271_L_f20_k4: x_Gr_Liv_Area + 1e+06 t271_leaf4 <= 1.002689999999e+06
 T271_R_f7_k5: x_Mas_Vnr_Area - 1e+06 t271_leaf5 >= -999212
 T271_R_f20_k5: x_Gr_Liv_Area - 1e+06 t271_leaf5 >= -997310
 T272_onehot: t272_leaf0 + t272_leaf1 + t272_leaf2 + t272_leaf3
   + t272_leaf4 + t272_leaf5 + t272_leaf6 + t272_leaf7 = 1
 T272_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t272_leaf0
   <= 1.000000999999e+06
 T272_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t272_leaf0
   <= 1.000000999999e+06
 T272_L_f3_k0: Overall_Qual_calc + 1e+06 t272_leaf0 <= 1.000002999999e+06
 T272_L_f96_k1: const_Neighborhood_Edwards + 1e+06 t272_leaf1
   <= 1.000000999999e+06
 T272_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t272_leaf1
   <= 1.000000999999e+06
 T272_R_f3_k1: Overall_Qual_calc - 1e+06 t272_leaf1 >= -999997
 T272_L_f96_k2: const_Neighborhood_Edwards + 1e+06 t272_leaf2
   <= 1.000000999999e+06
 T272_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t272_leaf2
   >= -999999
 T272_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t272_leaf2 <= 1.000701999999e+06
 T272_L_f96_k3: const_Neighborhood_Edwards + 1e+06 t272_leaf3
   <= 1.000000999999e+06
 T272_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t272_leaf3
   >= -999999
 T272_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t272_leaf3 >= -999298
 T272_R_f96_k4: const_Neighborhood_Edwards - 1e+06 t272_leaf4 >= -999999
 T272_L_f7_k4: x_Mas_Vnr_Area + 1e+06 t272_leaf4 <= 1.000002999999e+06
 T272_L_f1_k4: const_Lot_Area + 1e+06 t272_leaf4 <= 1.016284999999e+06
 T272_R_f96_k5: const_Neighborhood_Edwards - 1e+06 t272_leaf5 >= -999999
 T272_L_f7_k5: x_Mas_Vnr_Area + 1e+06 t272_leaf5 <= 1.000002999999e+06
 T272_R_f1_k5: const_Lot_Area - 1e+06 t272_leaf5 >= -983715
 T272_R_f96_k6: const_Neighborhood_Edwards - 1e+06 t272_leaf6 >= -999999
 T272_R_f7_k6: x_Mas_Vnr_Area - 1e+06 t272_leaf6 >= -999997
 T272_L_f202_k6: const_Foundation_PConc + 1e+06 t272_leaf6
   <= 1.000000999999e+06
 T272_R_f96_k7: const_Neighborhood_Edwards - 1e+06 t272_leaf7 >= -999999
 T272_R_f7_k7: x_Mas_Vnr_Area - 1e+06 t272_leaf7 >= -999997
 T272_R_f202_k7: const_Foundation_PConc - 1e+06 t272_leaf7 >= -999999
 T273_onehot: t273_leaf0 + t273_leaf1 + t273_leaf2 + t273_leaf3
   + t273_leaf4 = 1
 T273_L_f33_k0: x_Garage_Area + 1e+06 t273_leaf0 <= 1.001153999999e+06
 T273_L_f44_k0: const_Yr_Sold + 1e+06 t273_leaf0 <= 1.002008999999e+06
 T273_L_f24_k0: x_Half_Bath + 1e+06 t273_leaf0 <= 1.000001999999e+06
 T273_L_f33_k1: x_Garage_Area + 1e+06 t273_leaf1 <= 1.001153999999e+06
 T273_L_f44_k1: const_Yr_Sold + 1e+06 t273_leaf1 <= 1.002008999999e+06
 T273_R_f24_k1: x_Half_Bath - 1e+06 t273_leaf1 >= -999998
 T273_L_f33_k2: x_Garage_Area + 1e+06 t273_leaf2 <= 1.001153999999e+06
 T273_R_f44_k2: const_Yr_Sold - 1e+06 t273_leaf2 >= -997991
 T273_L_f20_k2: x_Gr_Liv_Area + 1e+06 t273_leaf2 <= 1.000815999999e+06
 T273_L_f33_k3: x_Garage_Area + 1e+06 t273_leaf3 <= 1.001153999999e+06
 T273_R_f44_k3: const_Yr_Sold - 1e+06 t273_leaf3 >= -997991
 T273_R_f20_k3: x_Gr_Liv_Area - 1e+06 t273_leaf3 >= -999184
 T273_R_f33_k4: x_Garage_Area - 1e+06 t273_leaf4 >= -998846
 T274_onehot: t274_leaf0 + t274_leaf1 + t274_leaf2 + t274_leaf3
   + t274_leaf4 = 1
 T274_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t274_leaf0
   <= 1.002135999999e+06
 T274_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t274_leaf0
   <= 1.001952999999e+06
 T274_L_f17_k0: x_1st_Flr_SF + 1e+06 t274_leaf0 <= 1.001391999999e+06
 T274_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t274_leaf1
   <= 1.002135999999e+06
 T274_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t274_leaf1
   <= 1.001952999999e+06
 T274_R_f17_k1: x_1st_Flr_SF - 1e+06 t274_leaf1 >= -998608
 T274_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t274_leaf2
   <= 1.002135999999e+06
 T274_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t274_leaf2 >= -998047
 T274_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t274_leaf2 <= 1.000475999999e+06
 T274_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t274_leaf3
   <= 1.002135999999e+06
 T274_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t274_leaf3 >= -998047
 T274_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t274_leaf3 >= -999524
 T274_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t274_leaf4 >= -997864
 T275_onehot: t275_leaf0 + t275_leaf1 + t275_leaf2 + t275_leaf3
   + t275_leaf4 + t275_leaf5 = 1
 T275_L_f20_k0: x_Gr_Liv_Area + 1e+06 t275_leaf0 <= 1.000746999999e+06
 T275_L_f38_k0: x_Enclosed_Porch + 1e+06 t275_leaf0 <= 1.000015999999e+06
 T275_L_f20_k1: x_Gr_Liv_Area + 1e+06 t275_leaf1 <= 1.000746999999e+06
 T275_R_f38_k1: x_Enclosed_Porch - 1e+06 t275_leaf1 >= -999984
 T275_R_f20_k2: x_Gr_Liv_Area - 1e+06 t275_leaf2 >= -999253
 T275_L_f43_k2: const_Misc_Val + 1e+06 t275_leaf2 <= 1.000053999999e+06
 T275_L_f119_k2: const_Condition_1_Norm + 1e+06 t275_leaf2
   <= 1.000000999999e+06
 T275_R_f20_k3: x_Gr_Liv_Area - 1e+06 t275_leaf3 >= -999253
 T275_L_f43_k3: const_Misc_Val + 1e+06 t275_leaf3 <= 1.000053999999e+06
 T275_R_f119_k3: const_Condition_1_Norm - 1e+06 t275_leaf3 >= -999999
 T275_R_f20_k4: x_Gr_Liv_Area - 1e+06 t275_leaf4 >= -999253
 T275_R_f43_k4: const_Misc_Val - 1e+06 t275_leaf4 >= -999946
 T275_L_f25_k4: x_Bedroom_AbvGr + 1e+06 t275_leaf4 <= 1.000002999999e+06
 T275_R_f20_k5: x_Gr_Liv_Area - 1e+06 t275_leaf5 >= -999253
 T275_R_f43_k5: const_Misc_Val - 1e+06 t275_leaf5 >= -999946
 T275_R_f25_k5: x_Bedroom_AbvGr - 1e+06 t275_leaf5 >= -999997
 T276_onehot: t276_leaf0 + t276_leaf1 + t276_leaf2 + t276_leaf3
   + t276_leaf4 = 1
 T276_L_f1_k0: const_Lot_Area + 1e+06 t276_leaf0 <= 1.003181999999e+06
 T276_R_f1_k1: const_Lot_Area - 1e+06 t276_leaf1 >= -996818
 T276_L_f190_k1: x_ex2_is_Stucco + 1e+06 t276_leaf1 <= 1.000000999999e+06
 T276_L_f17_k1: x_1st_Flr_SF + 1e+06 t276_leaf1 <= 1.001810999999e+06
 T276_R_f1_k2: const_Lot_Area - 1e+06 t276_leaf2 >= -996818
 T276_L_f190_k2: x_ex2_is_Stucco + 1e+06 t276_leaf2 <= 1.000000999999e+06
 T276_R_f17_k2: x_1st_Flr_SF - 1e+06 t276_leaf2 >= -998189
 T276_R_f1_k3: const_Lot_Area - 1e+06 t276_leaf3 >= -996818
 T276_R_f190_k3: x_ex2_is_Stucco - 1e+06 t276_leaf3 >= -999999
 T276_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t276_leaf3 <= 1.000137999999e+06
 T276_R_f1_k4: const_Lot_Area - 1e+06 t276_leaf4 >= -996818
 T276_R_f190_k4: x_ex2_is_Stucco - 1e+06 t276_leaf4 >= -999999
 T276_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t276_leaf4 >= -999862
 T277_onehot: t277_leaf0 + t277_leaf1 + t277_leaf2 + t277_leaf3
   + t277_leaf4 + t277_leaf5 + t277_leaf6 = 1
 T277_L_f4_k0: const_Overall_Cond + 1e+06 t277_leaf0 <= 1.000004999999e+06
 T277_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t277_leaf0
   <= 1.000000999999e+06
 T277_L_f209_k0: const_Bsmt_Exposure_No + 1e+06 t277_leaf0
   <= 1.000000999999e+06
 T277_L_f4_k1: const_Overall_Cond + 1e+06 t277_leaf1 <= 1.000004999999e+06
 T277_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t277_leaf1
   <= 1.000000999999e+06
 T277_R_f209_k1: const_Bsmt_Exposure_No - 1e+06 t277_leaf1 >= -999999
 T277_L_f4_k2: const_Overall_Cond + 1e+06 t277_leaf2 <= 1.000004999999e+06
 T277_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t277_leaf2 >= -999999
 T277_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t277_leaf2
   <= 1.000000999999e+06
 T277_L_f4_k3: const_Overall_Cond + 1e+06 t277_leaf3 <= 1.000004999999e+06
 T277_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t277_leaf3 >= -999999
 T277_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t277_leaf3 >= -999999
 T277_R_f4_k4: const_Overall_Cond - 1e+06 t277_leaf4 >= -999995
 T277_L_f0_k4: const_Lot_Frontage + 1e+06 t277_leaf4 <= 1.000151999999e+06
 T277_L_f37_k4: x_Open_Porch_SF + 1e+06 t277_leaf4 <= 1.000275999999e+06
 T277_R_f4_k5: const_Overall_Cond - 1e+06 t277_leaf5 >= -999995
 T277_L_f0_k5: const_Lot_Frontage + 1e+06 t277_leaf5 <= 1.000151999999e+06
 T277_R_f37_k5: x_Open_Porch_SF - 1e+06 t277_leaf5 >= -999724
 T277_R_f4_k6: const_Overall_Cond - 1e+06 t277_leaf6 >= -999995
 T277_R_f0_k6: const_Lot_Frontage - 1e+06 t277_leaf6 >= -999848
 T278_onehot: t278_leaf0 + t278_leaf1 + t278_leaf2 + t278_leaf3 = 1
 T278_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t278_leaf0 <= 1.000000999999e+06
 T278_L_f3_k0: Overall_Qual_calc + 1e+06 t278_leaf0 <= 1.000002999999e+06
 T278_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t278_leaf1 <= 1.000000999999e+06
 T278_R_f3_k1: Overall_Qual_calc - 1e+06 t278_leaf1 >= -999997
 T278_L_f277_k1: const_Mo_Sold_7 + 1e+06 t278_leaf1 <= 1.000000999999e+06
 T278_L_f164_k2: x_ex1_is_BrkFace + 1e+06 t278_leaf2 <= 1.000000999999e+06
 T278_R_f3_k2: Overall_Qual_calc - 1e+06 t278_leaf2 >= -999997
 T278_R_f277_k2: const_Mo_Sold_7 - 1e+06 t278_leaf2 >= -999999
 T278_R_f164_k3: x_ex1_is_BrkFace - 1e+06 t278_leaf3 >= -999999
 T279_onehot: t279_leaf0 + t279_leaf1 + t279_leaf2 + t279_leaf3
   + t279_leaf4 + t279_leaf5 = 1
 T279_L_f3_k0: Overall_Qual_calc + 1e+06 t279_leaf0 <= 1.000003999999e+06
 T279_L_f259_k0: x_paved_drive_is_Y + 1e+06 t279_leaf0
   <= 1.000000999999e+06
 T279_L_f3_k1: Overall_Qual_calc + 1e+06 t279_leaf1 <= 1.000003999999e+06
 T279_R_f259_k1: x_paved_drive_is_Y - 1e+06 t279_leaf1 >= -999999
 T279_R_f3_k2: Overall_Qual_calc - 1e+06 t279_leaf2 >= -999996
 T279_L_f6_k2: const_Year_Remod_Add + 1e+06 t279_leaf2
   <= 1.002006999999e+06
 T279_L_f45_k2: const_MS_SubClass_20 + 1e+06 t279_leaf2
   <= 1.000000999999e+06
 T279_R_f3_k3: Overall_Qual_calc - 1e+06 t279_leaf3 >= -999996
 T279_L_f6_k3: const_Year_Remod_Add + 1e+06 t279_leaf3
   <= 1.002006999999e+06
 T279_R_f45_k3: const_MS_SubClass_20 - 1e+06 t279_leaf3 >= -999999
 T279_R_f3_k4: Overall_Qual_calc - 1e+06 t279_leaf4 >= -999996
 T279_R_f6_k4: const_Year_Remod_Add - 1e+06 t279_leaf4 >= -997993
 T279_L_f0_k4: const_Lot_Frontage + 1e+06 t279_leaf4 <= 1.000094999999e+06
 T279_R_f3_k5: Overall_Qual_calc - 1e+06 t279_leaf5 >= -999996
 T279_R_f6_k5: const_Year_Remod_Add - 1e+06 t279_leaf5 >= -997993
 T279_R_f0_k5: const_Lot_Frontage - 1e+06 t279_leaf5 >= -999905
 T280_onehot: t280_leaf0 + t280_leaf1 + t280_leaf2 + t280_leaf3
   + t280_leaf4 + t280_leaf5 + t280_leaf6 = 1
 T280_L_f4_k0: const_Overall_Cond + 1e+06 t280_leaf0 <= 1.000003999999e+06
 T280_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t280_leaf0 <= 1.000335999999e+06
 T280_L_f4_k1: const_Overall_Cond + 1e+06 t280_leaf1 <= 1.000003999999e+06
 T280_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t280_leaf1 >= -999664
 T280_L_f6_k1: const_Year_Remod_Add + 1e+06 t280_leaf1
   <= 1.001951999999e+06
 T280_L_f4_k2: const_Overall_Cond + 1e+06 t280_leaf2 <= 1.000003999999e+06
 T280_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t280_leaf2 >= -999664
 T280_R_f6_k2: const_Year_Remod_Add - 1e+06 t280_leaf2 >= -998048
 T280_R_f4_k3: const_Overall_Cond - 1e+06 t280_leaf3 >= -999996
 T280_L_f117_k3: const_Condition_1_Artery + 1e+06 t280_leaf3
   <= 1.000000999999e+06
 T280_L_f29_k3: const_Fireplaces + 1e+06 t280_leaf3 <= 1.000002999999e+06
 T280_R_f4_k4: const_Overall_Cond - 1e+06 t280_leaf4 >= -999996
 T280_L_f117_k4: const_Condition_1_Artery + 1e+06 t280_leaf4
   <= 1.000000999999e+06
 T280_R_f29_k4: const_Fireplaces - 1e+06 t280_leaf4 >= -999997
 T280_R_f4_k5: const_Overall_Cond - 1e+06 t280_leaf5 >= -999996
 T280_R_f117_k5: const_Condition_1_Artery - 1e+06 t280_leaf5 >= -999999
 T280_L_f33_k5: x_Garage_Area + 1e+06 t280_leaf5 <= 1.000803999999e+06
 T280_R_f4_k6: const_Overall_Cond - 1e+06 t280_leaf6 >= -999996
 T280_R_f117_k6: const_Condition_1_Artery - 1e+06 t280_leaf6 >= -999999
 T280_R_f33_k6: x_Garage_Area - 1e+06 t280_leaf6 >= -999196
 T281_onehot: t281_leaf0 + t281_leaf1 + t281_leaf2 + t281_leaf3
   + t281_leaf4 = 1
 T281_L_f3_k0: Overall_Qual_calc + 1e+06 t281_leaf0 <= 1.000002999999e+06
 T281_R_f3_k1: Overall_Qual_calc - 1e+06 t281_leaf1 >= -999997
 T281_L_f3_k1: Overall_Qual_calc + 1e+06 t281_leaf1 <= 1.000006999999e+06
 T281_L_f1_k1: const_Lot_Area + 1e+06 t281_leaf1 <= 1.007999999999e+06
 T281_R_f3_k2: Overall_Qual_calc - 1e+06 t281_leaf2 >= -999997
 T281_L_f3_k2: Overall_Qual_calc + 1e+06 t281_leaf2 <= 1.000006999999e+06
 T281_R_f1_k2: const_Lot_Area - 1e+06 t281_leaf2 >= -992000
 T281_R_f3_k3: Overall_Qual_calc - 1e+06 t281_leaf3 >= -999997
 T281_R_f3_k3: Overall_Qual_calc - 1e+06 t281_leaf3 >= -999993
 T281_L_f77_k3: const_Land_Contour_Bnk + 1e+06 t281_leaf3
   <= 1.000000999999e+06
 T281_R_f3_k4: Overall_Qual_calc - 1e+06 t281_leaf4 >= -999997
 T281_R_f3_k4: Overall_Qual_calc - 1e+06 t281_leaf4 >= -999993
 T281_R_f77_k4: const_Land_Contour_Bnk - 1e+06 t281_leaf4 >= -999999
 T282_onehot: t282_leaf0 + t282_leaf1 + t282_leaf2 + t282_leaf3 = 1
 T282_L_f37_k0: x_Open_Porch_SF + 1e+06 t282_leaf0 <= 1.000417999999e+06
 T282_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t282_leaf0 <= 1.000704999999e+06
 T282_L_f44_k0: const_Yr_Sold + 1e+06 t282_leaf0 <= 1.002006999999e+06
 T282_L_f37_k1: x_Open_Porch_SF + 1e+06 t282_leaf1 <= 1.000417999999e+06
 T282_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t282_leaf1 <= 1.000704999999e+06
 T282_R_f44_k1: const_Yr_Sold - 1e+06 t282_leaf1 >= -997993
 T282_L_f37_k2: x_Open_Porch_SF + 1e+06 t282_leaf2 <= 1.000417999999e+06
 T282_R_f7_k2: x_Mas_Vnr_Area - 1e+06 t282_leaf2 >= -999295
 T282_R_f37_k3: x_Open_Porch_SF - 1e+06 t282_leaf3 >= -999582
 T283_onehot: t283_leaf0 + t283_leaf1 + t283_leaf2 + t283_leaf3
   + t283_leaf4 + t283_leaf5 = 1
 T283_L_f10_k0: const_Bsmt_Qual + 1e+06 t283_leaf0 <= 999999.999999
 T283_L_f251_k0: const_Garage_Type_Detchd + 1e+06 t283_leaf0
   <= 1.000000999999e+06
 T283_L_f10_k1: const_Bsmt_Qual + 1e+06 t283_leaf1 <= 999999.999999
 T283_R_f251_k1: const_Garage_Type_Detchd - 1e+06 t283_leaf1 >= -999999
 T283_R_f10_k2: const_Bsmt_Qual - 1e+06 t283_leaf2 >= -1e+06
 T283_L_f46_k2: const_MS_SubClass_30 + 1e+06 t283_leaf2
   <= 1.000000999999e+06
 T283_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t283_leaf2
   <= 1.001940999999e+06
 T283_R_f10_k3: const_Bsmt_Qual - 1e+06 t283_leaf3 >= -1e+06
 T283_L_f46_k3: const_MS_SubClass_30 + 1e+06 t283_leaf3
   <= 1.000000999999e+06
 T283_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t283_leaf3 >= -998059
 T283_R_f10_k4: const_Bsmt_Qual - 1e+06 t283_leaf4 >= -1e+06
 T283_R_f46_k4: const_MS_SubClass_30 - 1e+06 t283_leaf4 >= -999999
 T283_L_f6_k4: const_Year_Remod_Add + 1e+06 t283_leaf4
   <= 1.001950999999e+06
 T283_R_f10_k5: const_Bsmt_Qual - 1e+06 t283_leaf5 >= -1e+06
 T283_R_f46_k5: const_MS_SubClass_30 - 1e+06 t283_leaf5 >= -999999
 T283_R_f6_k5: const_Year_Remod_Add - 1e+06 t283_leaf5 >= -998049
 T284_onehot: t284_leaf0 + t284_leaf1 + t284_leaf2 + t284_leaf3
   + t284_leaf4 = 1
 T284_L_f38_k0: x_Enclosed_Porch + 1e+06 t284_leaf0 <= 1.000247999999e+06
 T284_L_f33_k0: x_Garage_Area + 1e+06 t284_leaf0 <= 1.001153999999e+06
 T284_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t284_leaf0
   <= 1.002135999999e+06
 T284_L_f38_k1: x_Enclosed_Porch + 1e+06 t284_leaf1 <= 1.000247999999e+06
 T284_L_f33_k1: x_Garage_Area + 1e+06 t284_leaf1 <= 1.001153999999e+06
 T284_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t284_leaf1 >= -997864
 T284_L_f38_k2: x_Enclosed_Porch + 1e+06 t284_leaf2 <= 1.000247999999e+06
 T284_R_f33_k2: x_Garage_Area - 1e+06 t284_leaf2 >= -998846
 T284_L_f10_k2: const_Bsmt_Qual + 1e+06 t284_leaf2 <= 1.000003999999e+06
 T284_L_f38_k3: x_Enclosed_Porch + 1e+06 t284_leaf3 <= 1.000247999999e+06
 T284_R_f33_k3: x_Garage_Area - 1e+06 t284_leaf3 >= -998846
 T284_R_f10_k3: const_Bsmt_Qual - 1e+06 t284_leaf3 >= -999996
 T284_R_f38_k4: x_Enclosed_Porch - 1e+06 t284_leaf4 >= -999752
 T285_onehot: t285_leaf0 + t285_leaf1 + t285_leaf2 + t285_leaf3
   + t285_leaf4 + t285_leaf5 = 1
 T285_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t285_leaf0 <= 1.000011999999e+06
 T285_L_f16_k0: x_Heating_QC + 1e+06 t285_leaf0 <= 1.000002999999e+06
 T285_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t285_leaf0
   <= 1.000000999999e+06
 T285_L_f28_k1: x_TotRms_AbvGrd + 1e+06 t285_leaf1 <= 1.000011999999e+06
 T285_L_f16_k1: x_Heating_QC + 1e+06 t285_leaf1 <= 1.000002999999e+06
 T285_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t285_leaf1 >= -999999
 T285_L_f28_k2: x_TotRms_AbvGrd + 1e+06 t285_leaf2 <= 1.000011999999e+06
 T285_R_f16_k2: x_Heating_QC - 1e+06 t285_leaf2 >= -999997
 T285_L_f117_k2: const_Condition_1_Artery + 1e+06 t285_leaf2
   <= 1.000000999999e+06
 T285_L_f28_k3: x_TotRms_AbvGrd + 1e+06 t285_leaf3 <= 1.000011999999e+06
 T285_R_f16_k3: x_Heating_QC - 1e+06 t285_leaf3 >= -999997
 T285_R_f117_k3: const_Condition_1_Artery - 1e+06 t285_leaf3 >= -999999
 T285_R_f28_k4: x_TotRms_AbvGrd - 1e+06 t285_leaf4 >= -999988
 T285_L_f18_k4: const_2nd_Flr_SF + 1e+06 t285_leaf4 <= 1.000974999999e+06
 T285_R_f28_k5: x_TotRms_AbvGrd - 1e+06 t285_leaf5 >= -999988
 T285_R_f18_k5: const_2nd_Flr_SF - 1e+06 t285_leaf5 >= -999025
 T286_onehot: t286_leaf0 + t286_leaf1 + t286_leaf2 + t286_leaf3
   + t286_leaf4 + t286_leaf5 = 1
 T286_L_f3_k0: Overall_Qual_calc + 1e+06 t286_leaf0 <= 1.000002999999e+06
 T286_L_f33_k0: x_Garage_Area + 1e+06 t286_leaf0 <= 1.000285999999e+06
 T286_L_f3_k1: Overall_Qual_calc + 1e+06 t286_leaf1 <= 1.000002999999e+06
 T286_R_f33_k1: x_Garage_Area - 1e+06 t286_leaf1 >= -999714
 T286_R_f3_k2: Overall_Qual_calc - 1e+06 t286_leaf2 >= -999997
 T286_L_f1_k2: const_Lot_Area + 1e+06 t286_leaf2 <= 1.020430999999e+06
 T286_L_f1_k2: const_Lot_Area + 1e+06 t286_leaf2 <= 1.017540999999e+06
 T286_R_f3_k3: Overall_Qual_calc - 1e+06 t286_leaf3 >= -999997
 T286_L_f1_k3: const_Lot_Area + 1e+06 t286_leaf3 <= 1.020430999999e+06
 T286_R_f1_k3: const_Lot_Area - 1e+06 t286_leaf3 >= -982459
 T286_R_f3_k4: Overall_Qual_calc - 1e+06 t286_leaf4 >= -999997
 T286_R_f1_k4: const_Lot_Area - 1e+06 t286_leaf4 >= -979569
 T286_L_f25_k4: x_Bedroom_AbvGr + 1e+06 t286_leaf4 <= 1.000003999999e+06
 T286_R_f3_k5: Overall_Qual_calc - 1e+06 t286_leaf5 >= -999997
 T286_R_f1_k5: const_Lot_Area - 1e+06 t286_leaf5 >= -979569
 T286_R_f25_k5: x_Bedroom_AbvGr - 1e+06 t286_leaf5 >= -999996
 T287_onehot: t287_leaf0 + t287_leaf1 + t287_leaf2 + t287_leaf3
   + t287_leaf4 + t287_leaf5 = 1
 T287_L_f3_k0: Overall_Qual_calc + 1e+06 t287_leaf0 <= 1.000002999999e+06
 T287_L_f35_k0: x_Garage_Cond + 1e+06 t287_leaf0 <= 1.000001999999e+06
 T287_L_f3_k1: Overall_Qual_calc + 1e+06 t287_leaf1 <= 1.000002999999e+06
 T287_R_f35_k1: x_Garage_Cond - 1e+06 t287_leaf1 >= -999998
 T287_R_f3_k2: Overall_Qual_calc - 1e+06 t287_leaf2 >= -999997
 T287_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t287_leaf2
   <= 1.000001999999e+06
 T287_L_f3_k2: Overall_Qual_calc + 1e+06 t287_leaf2 <= 1.000006999999e+06
 T287_R_f3_k3: Overall_Qual_calc - 1e+06 t287_leaf3 >= -999997
 T287_L_f21_k3: const_Bsmt_Full_Bath + 1e+06 t287_leaf3
   <= 1.000001999999e+06
 T287_R_f3_k3: Overall_Qual_calc - 1e+06 t287_leaf3 >= -999993
 T287_R_f3_k4: Overall_Qual_calc - 1e+06 t287_leaf4 >= -999997
 T287_R_f21_k4: const_Bsmt_Full_Bath - 1e+06 t287_leaf4 >= -999998
 T287_L_f18_k4: const_2nd_Flr_SF + 1e+06 t287_leaf4 <= 1.000504999999e+06
 T287_R_f3_k5: Overall_Qual_calc - 1e+06 t287_leaf5 >= -999997
 T287_R_f21_k5: const_Bsmt_Full_Bath - 1e+06 t287_leaf5 >= -999998
 T287_R_f18_k5: const_2nd_Flr_SF - 1e+06 t287_leaf5 >= -999495
 T288_onehot: t288_leaf0 + t288_leaf1 + t288_leaf2 + t288_leaf3
   + t288_leaf4 + t288_leaf5 = 1
 T288_L_f75_k0: const_Lot_Shape_IR3 + 1e+06 t288_leaf0
   <= 1.000000999999e+06
 T288_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t288_leaf0 <= 1.000407999999e+06
 T288_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t288_leaf0 <= 1.001572999999e+06
 T288_L_f75_k1: const_Lot_Shape_IR3 + 1e+06 t288_leaf1
   <= 1.000000999999e+06
 T288_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t288_leaf1 <= 1.000407999999e+06
 T288_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t288_leaf1 >= -998427
 T288_L_f75_k2: const_Lot_Shape_IR3 + 1e+06 t288_leaf2
   <= 1.000000999999e+06
 T288_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t288_leaf2 >= -999592
 T288_L_f25_k2: x_Bedroom_AbvGr + 1e+06 t288_leaf2 <= 1.000003999999e+06
 T288_L_f75_k3: const_Lot_Shape_IR3 + 1e+06 t288_leaf3
   <= 1.000000999999e+06
 T288_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t288_leaf3 >= -999592
 T288_R_f25_k3: x_Bedroom_AbvGr - 1e+06 t288_leaf3 >= -999996
 T288_R_f75_k4: const_Lot_Shape_IR3 - 1e+06 t288_leaf4 >= -999999
 T288_L_f249_k4: const_Garage_Type_BuiltIn + 1e+06 t288_leaf4
   <= 1.000000999999e+06
 T288_R_f75_k5: const_Lot_Shape_IR3 - 1e+06 t288_leaf5 >= -999999
 T288_R_f249_k5: const_Garage_Type_BuiltIn - 1e+06 t288_leaf5 >= -999999
 T289_onehot: t289_leaf0 + t289_leaf1 + t289_leaf2 + t289_leaf3
   + t289_leaf4 + t289_leaf5 = 1
 T289_L_f24_k0: x_Half_Bath + 1e+06 t289_leaf0 <= 1.000000999999e+06
 T289_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t289_leaf0
   <= 1.000001999999e+06
 T289_L_f271_k0: const_Mo_Sold_1 + 1e+06 t289_leaf0 <= 1.000000999999e+06
 T289_L_f24_k1: x_Half_Bath + 1e+06 t289_leaf1 <= 1.000000999999e+06
 T289_L_f21_k1: const_Bsmt_Full_Bath + 1e+06 t289_leaf1
   <= 1.000001999999e+06
 T289_R_f271_k1: const_Mo_Sold_1 - 1e+06 t289_leaf1 >= -999999
 T289_L_f24_k2: x_Half_Bath + 1e+06 t289_leaf2 <= 1.000000999999e+06
 T289_R_f21_k2: const_Bsmt_Full_Bath - 1e+06 t289_leaf2 >= -999998
 T289_R_f24_k3: x_Half_Bath - 1e+06 t289_leaf3 >= -999999
 T289_L_f237_k3: x_elect_is_SBrkr + 1e+06 t289_leaf3 <= 1.000000999999e+06
 T289_L_f119_k3: const_Condition_1_Norm + 1e+06 t289_leaf3
   <= 1.000000999999e+06
 T289_R_f24_k4: x_Half_Bath - 1e+06 t289_leaf4 >= -999999
 T289_L_f237_k4: x_elect_is_SBrkr + 1e+06 t289_leaf4 <= 1.000000999999e+06
 T289_R_f119_k4: const_Condition_1_Norm - 1e+06 t289_leaf4 >= -999999
 T289_R_f24_k5: x_Half_Bath - 1e+06 t289_leaf5 >= -999999
 T289_R_f237_k5: x_elect_is_SBrkr - 1e+06 t289_leaf5 >= -999999
 T290_onehot: t290_leaf0 + t290_leaf1 + t290_leaf2 + t290_leaf3
   + t290_leaf4 + t290_leaf5 = 1
 T290_L_f77_k0: const_Land_Contour_Bnk + 1e+06 t290_leaf0
   <= 1.000000999999e+06
 T290_L_f37_k0: x_Open_Porch_SF + 1e+06 t290_leaf0 <= 1.000074999999e+06
 T290_L_f5_k0: const_Year_Built + 1e+06 t290_leaf0 <= 1.002007999999e+06
 T290_L_f77_k1: const_Land_Contour_Bnk + 1e+06 t290_leaf1
   <= 1.000000999999e+06
 T290_L_f37_k1: x_Open_Porch_SF + 1e+06 t290_leaf1 <= 1.000074999999e+06
 T290_R_f5_k1: const_Year_Built - 1e+06 t290_leaf1 >= -997992
 T290_L_f77_k2: const_Land_Contour_Bnk + 1e+06 t290_leaf2
   <= 1.000000999999e+06
 T290_R_f37_k2: x_Open_Porch_SF - 1e+06 t290_leaf2 >= -999925
 T290_L_f1_k2: const_Lot_Area + 1e+06 t290_leaf2 <= 1.014679999999e+06
 T290_L_f77_k3: const_Land_Contour_Bnk + 1e+06 t290_leaf3
   <= 1.000000999999e+06
 T290_R_f37_k3: x_Open_Porch_SF - 1e+06 t290_leaf3 >= -999925
 T290_R_f1_k3: const_Lot_Area - 1e+06 t290_leaf3 >= -985320
 T290_R_f77_k4: const_Land_Contour_Bnk - 1e+06 t290_leaf4 >= -999999
 T290_L_f8_k4: x_Exter_Qual + 1e+06 t290_leaf4 <= 1.000002999999e+06
 T290_R_f77_k5: const_Land_Contour_Bnk - 1e+06 t290_leaf5 >= -999999
 T290_R_f8_k5: x_Exter_Qual - 1e+06 t290_leaf5 >= -999997
 T291_onehot: t291_leaf0 + t291_leaf1 + t291_leaf2 + t291_leaf3
   + t291_leaf4 = 1
 T291_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t291_leaf0
   <= 1.002491999999e+06
 T291_L_f6_k0: const_Year_Remod_Add + 1e+06 t291_leaf0
   <= 1.002006999999e+06
 T291_L_f0_k0: const_Lot_Frontage + 1e+06 t291_leaf0 <= 1.000133999999e+06
 T291_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t291_leaf1
   <= 1.002491999999e+06
 T291_L_f6_k1: const_Year_Remod_Add + 1e+06 t291_leaf1
   <= 1.002006999999e+06
 T291_R_f0_k1: const_Lot_Frontage - 1e+06 t291_leaf1 >= -999866
 T291_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t291_leaf2
   <= 1.002491999999e+06
 T291_R_f6_k2: const_Year_Remod_Add - 1e+06 t291_leaf2 >= -997993
 T291_L_f280_k2: const_Mo_Sold_10 + 1e+06 t291_leaf2 <= 1.000000999999e+06
 T291_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t291_leaf3
   <= 1.002491999999e+06
 T291_R_f6_k3: const_Year_Remod_Add - 1e+06 t291_leaf3 >= -997993
 T291_R_f280_k3: const_Mo_Sold_10 - 1e+06 t291_leaf3 >= -999999
 T291_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t291_leaf4 >= -997508
 T292_onehot: t292_leaf0 + t292_leaf1 + t292_leaf2 + t292_leaf3
   + t292_leaf4 = 1
 T292_L_f93_k0: const_Neighborhood_ClearCr + 1e+06 t292_leaf0
   <= 1.000000999999e+06
 T292_L_f9_k0: x_Exter_Cond + 1e+06 t292_leaf0 <= 1.000001999999e+06
 T292_L_f37_k0: x_Open_Porch_SF + 1e+06 t292_leaf0 <= 1.000036999999e+06
 T292_L_f93_k1: const_Neighborhood_ClearCr + 1e+06 t292_leaf1
   <= 1.000000999999e+06
 T292_L_f9_k1: x_Exter_Cond + 1e+06 t292_leaf1 <= 1.000001999999e+06
 T292_R_f37_k1: x_Open_Porch_SF - 1e+06 t292_leaf1 >= -999963
 T292_L_f93_k2: const_Neighborhood_ClearCr + 1e+06 t292_leaf2
   <= 1.000000999999e+06
 T292_R_f9_k2: x_Exter_Cond - 1e+06 t292_leaf2 >= -999998
 T292_L_f41_k2: x_Pool_Area + 1e+06 t292_leaf2 <= 1.000367999999e+06
 T292_L_f93_k3: const_Neighborhood_ClearCr + 1e+06 t292_leaf3
   <= 1.000000999999e+06
 T292_R_f9_k3: x_Exter_Cond - 1e+06 t292_leaf3 >= -999998
 T292_R_f41_k3: x_Pool_Area - 1e+06 t292_leaf3 >= -999632
 T292_R_f93_k4: const_Neighborhood_ClearCr - 1e+06 t292_leaf4 >= -999999
 T293_onehot: t293_leaf0 + t293_leaf1 + t293_leaf2 + t293_leaf3
   + t293_leaf4 = 1
 T293_L_f20_k0: x_Gr_Liv_Area + 1e+06 t293_leaf0 <= 1.002648999999e+06
 T293_L_f100_k0: const_Neighborhood_IDOTRR + 1e+06 t293_leaf0
   <= 1.000000999999e+06
 T293_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t293_leaf0
   <= 1.000000999999e+06
 T293_L_f20_k1: x_Gr_Liv_Area + 1e+06 t293_leaf1 <= 1.002648999999e+06
 T293_L_f100_k1: const_Neighborhood_IDOTRR + 1e+06 t293_leaf1
   <= 1.000000999999e+06
 T293_R_f21_k1: const_Bsmt_Full_Bath - 1e+06 t293_leaf1 >= -999999
 T293_L_f20_k2: x_Gr_Liv_Area + 1e+06 t293_leaf2 <= 1.002648999999e+06
 T293_R_f100_k2: const_Neighborhood_IDOTRR - 1e+06 t293_leaf2 >= -999999
 T293_R_f20_k3: x_Gr_Liv_Area - 1e+06 t293_leaf3 >= -997351
 T293_L_f37_k3: x_Open_Porch_SF + 1e+06 t293_leaf3 <= 1.000030999999e+06
 T293_R_f20_k4: x_Gr_Liv_Area - 1e+06 t293_leaf4 >= -997351
 T293_R_f37_k4: x_Open_Porch_SF - 1e+06 t293_leaf4 >= -999969
 T294_onehot: t294_leaf0 + t294_leaf1 + t294_leaf2 + t294_leaf3
   + t294_leaf4 = 1
 T294_L_f296_k0: const_Sale_Condition_Family + 1e+06 t294_leaf0
   <= 1.000000999999e+06
 T294_L_f37_k0: x_Open_Porch_SF + 1e+06 t294_leaf0 <= 1.000417999999e+06
 T294_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t294_leaf0
   <= 1.002135999999e+06
 T294_L_f296_k1: const_Sale_Condition_Family + 1e+06 t294_leaf1
   <= 1.000000999999e+06
 T294_L_f37_k1: x_Open_Porch_SF + 1e+06 t294_leaf1 <= 1.000417999999e+06
 T294_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t294_leaf1 >= -997864
 T294_L_f296_k2: const_Sale_Condition_Family + 1e+06 t294_leaf2
   <= 1.000000999999e+06
 T294_R_f37_k2: x_Open_Porch_SF - 1e+06 t294_leaf2 >= -999582
 T294_R_f296_k3: const_Sale_Condition_Family - 1e+06 t294_leaf3 >= -999999
 T294_L_f33_k3: x_Garage_Area + 1e+06 t294_leaf3 <= 1.000440999999e+06
 T294_R_f296_k4: const_Sale_Condition_Family - 1e+06 t294_leaf4 >= -999999
 T294_R_f33_k4: x_Garage_Area - 1e+06 t294_leaf4 >= -999559
 T295_onehot: t295_leaf0 + t295_leaf1 + t295_leaf2 + t295_leaf3
   + t295_leaf4 + t295_leaf5 = 1
 T295_L_f119_k0: const_Condition_1_Norm + 1e+06 t295_leaf0
   <= 1.000000999999e+06
 T295_L_f30_k0: x_Fireplace_Qu + 1e+06 t295_leaf0 <= 1.000001999999e+06
 T295_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t295_leaf0 <= 1.000004999999e+06
 T295_L_f119_k1: const_Condition_1_Norm + 1e+06 t295_leaf1
   <= 1.000000999999e+06
 T295_L_f30_k1: x_Fireplace_Qu + 1e+06 t295_leaf1 <= 1.000001999999e+06
 T295_R_f25_k1: x_Bedroom_AbvGr - 1e+06 t295_leaf1 >= -999995
 T295_L_f119_k2: const_Condition_1_Norm + 1e+06 t295_leaf2
   <= 1.000000999999e+06
 T295_R_f30_k2: x_Fireplace_Qu - 1e+06 t295_leaf2 >= -999998
 T295_R_f119_k3: const_Condition_1_Norm - 1e+06 t295_leaf3 >= -999999
 T295_L_f4_k3: const_Overall_Cond + 1e+06 t295_leaf3 <= 1.000005999999e+06
 T295_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t295_leaf3
   <= 1.001922999999e+06
 T295_R_f119_k4: const_Condition_1_Norm - 1e+06 t295_leaf4 >= -999999
 T295_L_f4_k4: const_Overall_Cond + 1e+06 t295_leaf4 <= 1.000005999999e+06
 T295_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t295_leaf4 >= -998077
 T295_R_f119_k5: const_Condition_1_Norm - 1e+06 t295_leaf5 >= -999999
 T295_R_f4_k5: const_Overall_Cond - 1e+06 t295_leaf5 >= -999994
 T296_onehot: t296_leaf0 + t296_leaf1 + t296_leaf2 + t296_leaf3
   + t296_leaf4 + t296_leaf5 = 1
 T296_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t296_leaf0
   <= 1.000000999999e+06
 T296_L_f17_k0: x_1st_Flr_SF + 1e+06 t296_leaf0 <= 1.002199999999e+06
 T296_L_f117_k0: const_Condition_1_Artery + 1e+06 t296_leaf0
   <= 1.000000999999e+06
 T296_L_f96_k1: const_Neighborhood_Edwards + 1e+06 t296_leaf1
   <= 1.000000999999e+06
 T296_L_f17_k1: x_1st_Flr_SF + 1e+06 t296_leaf1 <= 1.002199999999e+06
 T296_R_f117_k1: const_Condition_1_Artery - 1e+06 t296_leaf1 >= -999999
 T296_L_f96_k2: const_Neighborhood_Edwards + 1e+06 t296_leaf2
   <= 1.000000999999e+06
 T296_R_f17_k2: x_1st_Flr_SF - 1e+06 t296_leaf2 >= -997800
 T296_R_f96_k3: const_Neighborhood_Edwards - 1e+06 t296_leaf3 >= -999999
 T296_L_f37_k3: x_Open_Porch_SF + 1e+06 t296_leaf3 <= 1.000237999999e+06
 T296_L_f85_k3: const_Lot_Config_Inside + 1e+06 t296_leaf3
   <= 1.000000999999e+06
 T296_R_f96_k4: const_Neighborhood_Edwards - 1e+06 t296_leaf4 >= -999999
 T296_L_f37_k4: x_Open_Porch_SF + 1e+06 t296_leaf4 <= 1.000237999999e+06
 T296_R_f85_k4: const_Lot_Config_Inside - 1e+06 t296_leaf4 >= -999999
 T296_R_f96_k5: const_Neighborhood_Edwards - 1e+06 t296_leaf5 >= -999999
 T296_R_f37_k5: x_Open_Porch_SF - 1e+06 t296_leaf5 >= -999762
 T297_onehot: t297_leaf0 + t297_leaf1 + t297_leaf2 + t297_leaf3 = 1
 T297_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t297_leaf0
   <= 1.000000999999e+06
 T297_L_f82_k0: const_Lot_Config_CulDSac + 1e+06 t297_leaf0
   <= 1.000000999999e+06
 T297_L_f120_k0: const_Condition_1_PosA + 1e+06 t297_leaf0
   <= 1.000000999999e+06
 T297_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t297_leaf1
   <= 1.000000999999e+06
 T297_L_f82_k1: const_Lot_Config_CulDSac + 1e+06 t297_leaf1
   <= 1.000000999999e+06
 T297_R_f120_k1: const_Condition_1_PosA - 1e+06 t297_leaf1 >= -999999
 T297_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t297_leaf2
   <= 1.000000999999e+06
 T297_R_f82_k2: const_Lot_Config_CulDSac - 1e+06 t297_leaf2 >= -999999
 T297_R_f95_k3: const_Neighborhood_Crawfor - 1e+06 t297_leaf3 >= -999999
 T298_onehot: t298_leaf0 + t298_leaf1 + t298_leaf2 + t298_leaf3
   + t298_leaf4 = 1
 T298_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t298_leaf0 <= 1.000921999999e+06
 T298_L_f45_k0: const_MS_SubClass_20 + 1e+06 t298_leaf0
   <= 1.000000999999e+06
 T298_L_f119_k0: const_Condition_1_Norm + 1e+06 t298_leaf0
   <= 1.000000999999e+06
 T298_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t298_leaf1 <= 1.000921999999e+06
 T298_L_f45_k1: const_MS_SubClass_20 + 1e+06 t298_leaf1
   <= 1.000000999999e+06
 T298_R_f119_k1: const_Condition_1_Norm - 1e+06 t298_leaf1 >= -999999
 T298_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t298_leaf2 <= 1.000921999999e+06
 T298_R_f45_k2: const_MS_SubClass_20 - 1e+06 t298_leaf2 >= -999999
 T298_L_f37_k2: x_Open_Porch_SF + 1e+06 t298_leaf2 <= 1.000090999999e+06
 T298_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t298_leaf3 <= 1.000921999999e+06
 T298_R_f45_k3: const_MS_SubClass_20 - 1e+06 t298_leaf3 >= -999999
 T298_R_f37_k3: x_Open_Porch_SF - 1e+06 t298_leaf3 >= -999909
 T298_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t298_leaf4 >= -999078
 T299_onehot: t299_leaf0 + t299_leaf1 + t299_leaf2 + t299_leaf3
   + t299_leaf4 + t299_leaf5 + t299_leaf6 = 1
 T299_L_f0_k0: const_Lot_Frontage + 1e+06 t299_leaf0 <= 1.000114999999e+06
 T299_L_f44_k0: const_Yr_Sold + 1e+06 t299_leaf0 <= 1.002008999999e+06
 T299_L_f4_k0: const_Overall_Cond + 1e+06 t299_leaf0 <= 1.000003999999e+06
 T299_L_f0_k1: const_Lot_Frontage + 1e+06 t299_leaf1 <= 1.000114999999e+06
 T299_L_f44_k1: const_Yr_Sold + 1e+06 t299_leaf1 <= 1.002008999999e+06
 T299_R_f4_k1: const_Overall_Cond - 1e+06 t299_leaf1 >= -999996
 T299_L_f0_k2: const_Lot_Frontage + 1e+06 t299_leaf2 <= 1.000114999999e+06
 T299_R_f44_k2: const_Yr_Sold - 1e+06 t299_leaf2 >= -997991
 T299_L_f297_k2: const_Sale_Condition_Normal + 1e+06 t299_leaf2
   <= 1.000000999999e+06
 T299_L_f0_k3: const_Lot_Frontage + 1e+06 t299_leaf3 <= 1.000114999999e+06
 T299_R_f44_k3: const_Yr_Sold - 1e+06 t299_leaf3 >= -997991
 T299_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t299_leaf3 >= -999999
 T299_R_f0_k4: const_Lot_Frontage - 1e+06 t299_leaf4 >= -999885
 T299_L_f77_k4: const_Land_Contour_Bnk + 1e+06 t299_leaf4
   <= 1.000000999999e+06
 T299_L_f17_k4: x_1st_Flr_SF + 1e+06 t299_leaf4 <= 1.001229999999e+06
 T299_R_f0_k5: const_Lot_Frontage - 1e+06 t299_leaf5 >= -999885
 T299_L_f77_k5: const_Land_Contour_Bnk + 1e+06 t299_leaf5
   <= 1.000000999999e+06
 T299_R_f17_k5: x_1st_Flr_SF - 1e+06 t299_leaf5 >= -998770
 T299_R_f0_k6: const_Lot_Frontage - 1e+06 t299_leaf6 >= -999885
 T299_R_f77_k6: const_Land_Contour_Bnk - 1e+06 t299_leaf6 >= -999999
 T300_onehot: t300_leaf0 + t300_leaf1 + t300_leaf2 + t300_leaf3
   + t300_leaf4 + t300_leaf5 = 1
 T300_L_f1_k0: const_Lot_Area + 1e+06 t300_leaf0 <= 1.017540999999e+06
 T300_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t300_leaf0 <= 1.001572999999e+06
 T300_L_f63_k0: const_MS_Zoning_FV + 1e+06 t300_leaf0
   <= 1.000000999999e+06
 T300_L_f1_k1: const_Lot_Area + 1e+06 t300_leaf1 <= 1.017540999999e+06
 T300_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t300_leaf1 <= 1.001572999999e+06
 T300_R_f63_k1: const_MS_Zoning_FV - 1e+06 t300_leaf1 >= -999999
 T300_L_f1_k2: const_Lot_Area + 1e+06 t300_leaf2 <= 1.017540999999e+06
 T300_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t300_leaf2 >= -998427
 T300_R_f1_k3: const_Lot_Area - 1e+06 t300_leaf3 >= -982459
 T300_L_f247_k3: const_Garage_Type_Attchd + 1e+06 t300_leaf3
   <= 1.000000999999e+06
 T300_R_f1_k4: const_Lot_Area - 1e+06 t300_leaf4 >= -982459
 T300_R_f247_k4: const_Garage_Type_Attchd - 1e+06 t300_leaf4 >= -999999
 T300_L_f6_k4: const_Year_Remod_Add + 1e+06 t300_leaf4
   <= 1.002004999999e+06
 T300_R_f1_k5: const_Lot_Area - 1e+06 t300_leaf5 >= -982459
 T300_R_f247_k5: const_Garage_Type_Attchd - 1e+06 t300_leaf5 >= -999999
 T300_R_f6_k5: const_Year_Remod_Add - 1e+06 t300_leaf5 >= -997995
 T301_onehot: t301_leaf0 + t301_leaf1 + t301_leaf2 + t301_leaf3
   + t301_leaf4 = 1
 T301_L_f17_k0: x_1st_Flr_SF + 1e+06 t301_leaf0 <= 1.002514999999e+06
 T301_L_f5_k0: const_Year_Built + 1e+06 t301_leaf0 <= 1.001914999999e+06
 T301_L_f17_k0: x_1st_Flr_SF + 1e+06 t301_leaf0 <= 1.000812999999e+06
 T301_L_f17_k1: x_1st_Flr_SF + 1e+06 t301_leaf1 <= 1.002514999999e+06
 T301_L_f5_k1: const_Year_Built + 1e+06 t301_leaf1 <= 1.001914999999e+06
 T301_R_f17_k1: x_1st_Flr_SF - 1e+06 t301_leaf1 >= -999187
 T301_L_f17_k2: x_1st_Flr_SF + 1e+06 t301_leaf2 <= 1.002514999999e+06
 T301_R_f5_k2: const_Year_Built - 1e+06 t301_leaf2 >= -998085
 T301_L_f44_k2: const_Yr_Sold + 1e+06 t301_leaf2 <= 1.002006999999e+06
 T301_L_f17_k3: x_1st_Flr_SF + 1e+06 t301_leaf3 <= 1.002514999999e+06
 T301_R_f5_k3: const_Year_Built - 1e+06 t301_leaf3 >= -998085
 T301_R_f44_k3: const_Yr_Sold - 1e+06 t301_leaf3 >= -997993
 T301_R_f17_k4: x_1st_Flr_SF - 1e+06 t301_leaf4 >= -997485
 T302_onehot: t302_leaf0 + t302_leaf1 + t302_leaf2 + t302_leaf3 = 1
 T302_L_f17_k0: x_1st_Flr_SF + 1e+06 t302_leaf0 <= 1.002199999999e+06
 T302_L_f113_k0: const_Neighborhood_Somerst + 1e+06 t302_leaf0
   <= 1.000000999999e+06
 T302_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t302_leaf0 <= 1.000107999999e+06
 T302_L_f17_k1: x_1st_Flr_SF + 1e+06 t302_leaf1 <= 1.002199999999e+06
 T302_L_f113_k1: const_Neighborhood_Somerst + 1e+06 t302_leaf1
   <= 1.000000999999e+06
 T302_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t302_leaf1 >= -999892
 T302_L_f17_k2: x_1st_Flr_SF + 1e+06 t302_leaf2 <= 1.002199999999e+06
 T302_R_f113_k2: const_Neighborhood_Somerst - 1e+06 t302_leaf2 >= -999999
 T302_R_f17_k3: x_1st_Flr_SF - 1e+06 t302_leaf3 >= -997800
 T303_onehot: t303_leaf0 + t303_leaf1 + t303_leaf2 + t303_leaf3
   + t303_leaf4 = 1
 T303_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t303_leaf0 <= 1.000787999999e+06
 T303_L_f3_k0: Overall_Qual_calc + 1e+06 t303_leaf0 <= 1.000009999999e+06
 T303_L_f40_k0: x_Screen_Porch + 1e+06 t303_leaf0 <= 1.000109999999e+06
 T303_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t303_leaf1 <= 1.000787999999e+06
 T303_L_f3_k1: Overall_Qual_calc + 1e+06 t303_leaf1 <= 1.000009999999e+06
 T303_R_f40_k1: x_Screen_Porch - 1e+06 t303_leaf1 >= -999890
 T303_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t303_leaf2 <= 1.000787999999e+06
 T303_R_f3_k2: Overall_Qual_calc - 1e+06 t303_leaf2 >= -999990
 T303_R_f7_k3: x_Mas_Vnr_Area - 1e+06 t303_leaf3 >= -999212
 T303_L_f20_k3: x_Gr_Liv_Area + 1e+06 t303_leaf3 <= 1.002727999999e+06
 T303_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t303_leaf4 >= -999212
 T303_R_f20_k4: x_Gr_Liv_Area - 1e+06 t303_leaf4 >= -997272
 T304_onehot: t304_leaf0 + t304_leaf1 + t304_leaf2 + t304_leaf3
   + t304_leaf4 = 1
 T304_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t304_leaf0 <= 1.000000999999e+06
 T304_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t304_leaf0
   <= 1.000000999999e+06
 T304_L_f71_k0: const_Alley_No_aplica + 1e+06 t304_leaf0
   <= 1.000000999999e+06
 T304_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t304_leaf1 <= 1.000000999999e+06
 T304_L_f96_k1: const_Neighborhood_Edwards + 1e+06 t304_leaf1
   <= 1.000000999999e+06
 T304_R_f71_k1: const_Alley_No_aplica - 1e+06 t304_leaf1 >= -999999
 T304_L_f164_k2: x_ex1_is_BrkFace + 1e+06 t304_leaf2 <= 1.000000999999e+06
 T304_R_f96_k2: const_Neighborhood_Edwards - 1e+06 t304_leaf2 >= -999999
 T304_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t304_leaf2 <= 1.000951999999e+06
 T304_L_f164_k3: x_ex1_is_BrkFace + 1e+06 t304_leaf3 <= 1.000000999999e+06
 T304_R_f96_k3: const_Neighborhood_Edwards - 1e+06 t304_leaf3 >= -999999
 T304_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t304_leaf3 >= -999048
 T304_R_f164_k4: x_ex1_is_BrkFace - 1e+06 t304_leaf4 >= -999999
 T305_onehot: t305_leaf0 + t305_leaf1 + t305_leaf2 + t305_leaf3
   + t305_leaf4 = 1
 T305_L_f17_k0: x_1st_Flr_SF + 1e+06 t305_leaf0 <= 1.002514999999e+06
 T305_L_f1_k0: const_Lot_Area + 1e+06 t305_leaf0 <= 1.005999999999e+06
 T305_L_f67_k0: const_MS_Zoning_RM + 1e+06 t305_leaf0
   <= 1.000000999999e+06
 T305_L_f17_k1: x_1st_Flr_SF + 1e+06 t305_leaf1 <= 1.002514999999e+06
 T305_L_f1_k1: const_Lot_Area + 1e+06 t305_leaf1 <= 1.005999999999e+06
 T305_R_f67_k1: const_MS_Zoning_RM - 1e+06 t305_leaf1 >= -999999
 T305_L_f17_k2: x_1st_Flr_SF + 1e+06 t305_leaf2 <= 1.002514999999e+06
 T305_R_f1_k2: const_Lot_Area - 1e+06 t305_leaf2 >= -994000
 T305_L_f112_k2: const_Neighborhood_SawyerW + 1e+06 t305_leaf2
   <= 1.000000999999e+06
 T305_L_f17_k3: x_1st_Flr_SF + 1e+06 t305_leaf3 <= 1.002514999999e+06
 T305_R_f1_k3: const_Lot_Area - 1e+06 t305_leaf3 >= -994000
 T305_R_f112_k3: const_Neighborhood_SawyerW - 1e+06 t305_leaf3 >= -999999
 T305_R_f17_k4: x_1st_Flr_SF - 1e+06 t305_leaf4 >= -997485
 T306_onehot: t306_leaf0 + t306_leaf1 + t306_leaf2 + t306_leaf3
   + t306_leaf4 + t306_leaf5 = 1
 T306_L_f3_k0: Overall_Qual_calc + 1e+06 t306_leaf0 <= 1.000002999999e+06
 T306_L_f6_k0: const_Year_Remod_Add + 1e+06 t306_leaf0
   <= 1.001950999999e+06
 T306_L_f3_k1: Overall_Qual_calc + 1e+06 t306_leaf1 <= 1.000002999999e+06
 T306_R_f6_k1: const_Year_Remod_Add - 1e+06 t306_leaf1 >= -998049
 T306_R_f3_k2: Overall_Qual_calc - 1e+06 t306_leaf2 >= -999997
 T306_L_f5_k2: const_Year_Built + 1e+06 t306_leaf2 <= 1.001913999999e+06
 T306_L_f17_k2: x_1st_Flr_SF + 1e+06 t306_leaf2 <= 1.000812999999e+06
 T306_R_f3_k3: Overall_Qual_calc - 1e+06 t306_leaf3 >= -999997
 T306_L_f5_k3: const_Year_Built + 1e+06 t306_leaf3 <= 1.001913999999e+06
 T306_R_f17_k3: x_1st_Flr_SF - 1e+06 t306_leaf3 >= -999187
 T306_R_f3_k4: Overall_Qual_calc - 1e+06 t306_leaf4 >= -999997
 T306_R_f5_k4: const_Year_Built - 1e+06 t306_leaf4 >= -998086
 T306_L_f232_k4: central_air_yes + 1e+06 t306_leaf4 <= 1.000000999999e+06
 T306_R_f3_k5: Overall_Qual_calc - 1e+06 t306_leaf5 >= -999997
 T306_R_f5_k5: const_Year_Built - 1e+06 t306_leaf5 >= -998086
 T306_R_f232_k5: central_air_yes - 1e+06 t306_leaf5 >= -999999
 T307_onehot: t307_leaf0 + t307_leaf1 + t307_leaf2 + t307_leaf3
   + t307_leaf4 + t307_leaf5 = 1
 T307_L_f3_k0: Overall_Qual_calc + 1e+06 t307_leaf0 <= 1.000003999999e+06
 T307_L_f0_k0: const_Lot_Frontage + 1e+06 t307_leaf0 <= 1.000055999999e+06
 T307_L_f3_k1: Overall_Qual_calc + 1e+06 t307_leaf1 <= 1.000003999999e+06
 T307_R_f0_k1: const_Lot_Frontage - 1e+06 t307_leaf1 >= -999944
 T307_R_f3_k2: Overall_Qual_calc - 1e+06 t307_leaf2 >= -999996
 T307_L_f206_k2: const_Bsmt_Exposure_Av + 1e+06 t307_leaf2
   <= 1.000000999999e+06
 T307_L_f5_k2: const_Year_Built + 1e+06 t307_leaf2 <= 1.001914999999e+06
 T307_R_f3_k3: Overall_Qual_calc - 1e+06 t307_leaf3 >= -999996
 T307_L_f206_k3: const_Bsmt_Exposure_Av + 1e+06 t307_leaf3
   <= 1.000000999999e+06
 T307_R_f5_k3: const_Year_Built - 1e+06 t307_leaf3 >= -998085
 T307_R_f3_k4: Overall_Qual_calc - 1e+06 t307_leaf4 >= -999996
 T307_R_f206_k4: const_Bsmt_Exposure_Av - 1e+06 t307_leaf4 >= -999999
 T307_L_f114_k4: const_Neighborhood_StoneBr + 1e+06 t307_leaf4
   <= 1.000000999999e+06
 T307_R_f3_k5: Overall_Qual_calc - 1e+06 t307_leaf5 >= -999996
 T307_R_f206_k5: const_Bsmt_Exposure_Av - 1e+06 t307_leaf5 >= -999999
 T307_R_f114_k5: const_Neighborhood_StoneBr - 1e+06 t307_leaf5 >= -999999
 T308_onehot: t308_leaf0 + t308_leaf1 + t308_leaf2 + t308_leaf3
   + t308_leaf4 = 1
 T308_L_f63_k0: const_MS_Zoning_FV + 1e+06 t308_leaf0
   <= 1.000000999999e+06
 T308_L_f3_k0: Overall_Qual_calc + 1e+06 t308_leaf0 <= 1.000006999999e+06
 T308_L_f23_k0: x_Full_Bath + 1e+06 t308_leaf0 <= 1.000001999999e+06
 T308_L_f63_k1: const_MS_Zoning_FV + 1e+06 t308_leaf1
   <= 1.000000999999e+06
 T308_L_f3_k1: Overall_Qual_calc + 1e+06 t308_leaf1 <= 1.000006999999e+06
 T308_R_f23_k1: x_Full_Bath - 1e+06 t308_leaf1 >= -999998
 T308_L_f63_k2: const_MS_Zoning_FV + 1e+06 t308_leaf2
   <= 1.000000999999e+06
 T308_R_f3_k2: Overall_Qual_calc - 1e+06 t308_leaf2 >= -999993
 T308_L_f86_k2: const_Land_Slope_Gtl + 1e+06 t308_leaf2
   <= 1.000000999999e+06
 T308_L_f63_k3: const_MS_Zoning_FV + 1e+06 t308_leaf3
   <= 1.000000999999e+06
 T308_R_f3_k3: Overall_Qual_calc - 1e+06 t308_leaf3 >= -999993
 T308_R_f86_k3: const_Land_Slope_Gtl - 1e+06 t308_leaf3 >= -999999
 T308_R_f63_k4: const_MS_Zoning_FV - 1e+06 t308_leaf4 >= -999999
 T309_onehot: t309_leaf0 + t309_leaf1 + t309_leaf2 + t309_leaf3
   + t309_leaf4 + t309_leaf5 = 1
 T309_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t309_leaf0 <= 1.000011999999e+06
 T309_L_f261_k0: x_fence_is_GdWo + 1e+06 t309_leaf0 <= 1.000000999999e+06
 T309_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t309_leaf0
   <= 1.002491999999e+06
 T309_L_f28_k1: x_TotRms_AbvGrd + 1e+06 t309_leaf1 <= 1.000011999999e+06
 T309_L_f261_k1: x_fence_is_GdWo + 1e+06 t309_leaf1 <= 1.000000999999e+06
 T309_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t309_leaf1 >= -997508
 T309_L_f28_k2: x_TotRms_AbvGrd + 1e+06 t309_leaf2 <= 1.000011999999e+06
 T309_R_f261_k2: x_fence_is_GdWo - 1e+06 t309_leaf2 >= -999999
 T309_L_f38_k2: x_Enclosed_Porch + 1e+06 t309_leaf2 <= 1.000100999999e+06
 T309_L_f28_k3: x_TotRms_AbvGrd + 1e+06 t309_leaf3 <= 1.000011999999e+06
 T309_R_f261_k3: x_fence_is_GdWo - 1e+06 t309_leaf3 >= -999999
 T309_R_f38_k3: x_Enclosed_Porch - 1e+06 t309_leaf3 >= -999899
 T309_R_f28_k4: x_TotRms_AbvGrd - 1e+06 t309_leaf4 >= -999988
 T309_L_f18_k4: const_2nd_Flr_SF + 1e+06 t309_leaf4 <= 1.000974999999e+06
 T309_R_f28_k5: x_TotRms_AbvGrd - 1e+06 t309_leaf5 >= -999988
 T309_R_f18_k5: const_2nd_Flr_SF - 1e+06 t309_leaf5 >= -999025
 T310_onehot: t310_leaf0 + t310_leaf1 + t310_leaf2 + t310_leaf3
   + t310_leaf4 = 1
 T310_L_f68_k0: const_Street_Grvl + 1e+06 t310_leaf0 <= 1.000000999999e+06
 T310_L_f27_k0: x_Kitchen_Qual + 1e+06 t310_leaf0 <= 1.000001999999e+06
 T310_L_f119_k0: const_Condition_1_Norm + 1e+06 t310_leaf0
   <= 1.000000999999e+06
 T310_L_f68_k1: const_Street_Grvl + 1e+06 t310_leaf1 <= 1.000000999999e+06
 T310_L_f27_k1: x_Kitchen_Qual + 1e+06 t310_leaf1 <= 1.000001999999e+06
 T310_R_f119_k1: const_Condition_1_Norm - 1e+06 t310_leaf1 >= -999999
 T310_L_f68_k2: const_Street_Grvl + 1e+06 t310_leaf2 <= 1.000000999999e+06
 T310_R_f27_k2: x_Kitchen_Qual - 1e+06 t310_leaf2 >= -999998
 T310_L_f4_k2: const_Overall_Cond + 1e+06 t310_leaf2 <= 1.000003999999e+06
 T310_L_f68_k3: const_Street_Grvl + 1e+06 t310_leaf3 <= 1.000000999999e+06
 T310_R_f27_k3: x_Kitchen_Qual - 1e+06 t310_leaf3 >= -999998
 T310_R_f4_k3: const_Overall_Cond - 1e+06 t310_leaf3 >= -999996
 T310_R_f68_k4: const_Street_Grvl - 1e+06 t310_leaf4 >= -999999
 T311_onehot: t311_leaf0 + t311_leaf1 + t311_leaf2 + t311_leaf3 = 1
 T311_L_f6_k0: const_Year_Remod_Add + 1e+06 t311_leaf0
   <= 1.002007999999e+06
 T311_L_f4_k0: const_Overall_Cond + 1e+06 t311_leaf0 <= 1.000005999999e+06
 T311_L_f17_k0: x_1st_Flr_SF + 1e+06 t311_leaf0 <= 1.001939999999e+06
 T311_L_f6_k1: const_Year_Remod_Add + 1e+06 t311_leaf1
   <= 1.002007999999e+06
 T311_L_f4_k1: const_Overall_Cond + 1e+06 t311_leaf1 <= 1.000005999999e+06
 T311_R_f17_k1: x_1st_Flr_SF - 1e+06 t311_leaf1 >= -998060
 T311_L_f6_k2: const_Year_Remod_Add + 1e+06 t311_leaf2
   <= 1.002007999999e+06
 T311_R_f4_k2: const_Overall_Cond - 1e+06 t311_leaf2 >= -999994
 T311_R_f6_k3: const_Year_Remod_Add - 1e+06 t311_leaf3 >= -997992
 T312_onehot: t312_leaf0 + t312_leaf1 + t312_leaf2 + t312_leaf3
   + t312_leaf4 = 1
 T312_L_f33_k0: x_Garage_Area + 1e+06 t312_leaf0 <= 1.001153999999e+06
 T312_L_f202_k0: const_Foundation_PConc + 1e+06 t312_leaf0
   <= 1.000000999999e+06
 T312_L_f5_k0: const_Year_Built + 1e+06 t312_leaf0 <= 1.001924999999e+06
 T312_L_f33_k1: x_Garage_Area + 1e+06 t312_leaf1 <= 1.001153999999e+06
 T312_L_f202_k1: const_Foundation_PConc + 1e+06 t312_leaf1
   <= 1.000000999999e+06
 T312_R_f5_k1: const_Year_Built - 1e+06 t312_leaf1 >= -998075
 T312_L_f33_k2: x_Garage_Area + 1e+06 t312_leaf2 <= 1.001153999999e+06
 T312_R_f202_k2: const_Foundation_PConc - 1e+06 t312_leaf2 >= -999999
 T312_L_f174_k2: x_ex1_is_VinylSd + 1e+06 t312_leaf2 <= 1.000000999999e+06
 T312_L_f33_k3: x_Garage_Area + 1e+06 t312_leaf3 <= 1.001153999999e+06
 T312_R_f202_k3: const_Foundation_PConc - 1e+06 t312_leaf3 >= -999999
 T312_R_f174_k3: x_ex1_is_VinylSd - 1e+06 t312_leaf3 >= -999999
 T312_R_f33_k4: x_Garage_Area - 1e+06 t312_leaf4 >= -998846
 T313_onehot: t313_leaf0 + t313_leaf1 + t313_leaf2 + t313_leaf3
   + t313_leaf4 + t313_leaf5 + t313_leaf6 = 1
 T313_L_f9_k0: x_Exter_Cond + 1e+06 t313_leaf0 <= 1.000001999999e+06
 T313_L_f1_k0: const_Lot_Area + 1e+06 t313_leaf0 <= 1.009170999999e+06
 T313_L_f5_k0: const_Year_Built + 1e+06 t313_leaf0 <= 1.001930999999e+06
 T313_L_f9_k1: x_Exter_Cond + 1e+06 t313_leaf1 <= 1.000001999999e+06
 T313_L_f1_k1: const_Lot_Area + 1e+06 t313_leaf1 <= 1.009170999999e+06
 T313_R_f5_k1: const_Year_Built - 1e+06 t313_leaf1 >= -998069
 T313_L_f9_k2: x_Exter_Cond + 1e+06 t313_leaf2 <= 1.000001999999e+06
 T313_R_f1_k2: const_Lot_Area - 1e+06 t313_leaf2 >= -990829
 T313_L_f37_k2: x_Open_Porch_SF + 1e+06 t313_leaf2 <= 1.000003999999e+06
 T313_L_f9_k3: x_Exter_Cond + 1e+06 t313_leaf3 <= 1.000001999999e+06
 T313_R_f1_k3: const_Lot_Area - 1e+06 t313_leaf3 >= -990829
 T313_R_f37_k3: x_Open_Porch_SF - 1e+06 t313_leaf3 >= -999996
 T313_R_f9_k4: x_Exter_Cond - 1e+06 t313_leaf4 >= -999998
 T313_L_f86_k4: const_Land_Slope_Gtl + 1e+06 t313_leaf4
   <= 1.000000999999e+06
 T313_R_f9_k5: x_Exter_Cond - 1e+06 t313_leaf5 >= -999998
 T313_R_f86_k5: const_Land_Slope_Gtl - 1e+06 t313_leaf5 >= -999999
 T313_L_f1_k5: const_Lot_Area + 1e+06 t313_leaf5 <= 1.035132999999e+06
 T313_R_f9_k6: x_Exter_Cond - 1e+06 t313_leaf6 >= -999998
 T313_R_f86_k6: const_Land_Slope_Gtl - 1e+06 t313_leaf6 >= -999999
 T313_R_f1_k6: const_Lot_Area - 1e+06 t313_leaf6 >= -964867
 T314_onehot: t314_leaf0 + t314_leaf1 + t314_leaf2 + t314_leaf3
   + t314_leaf4 + t314_leaf5 + t314_leaf6 = 1
 T314_L_f32_k0: const_Garage_Cars + 1e+06 t314_leaf0 <= 1.000002999999e+06
 T314_L_f119_k0: const_Condition_1_Norm + 1e+06 t314_leaf0
   <= 1.000000999999e+06
 T314_L_f5_k0: const_Year_Built + 1e+06 t314_leaf0 <= 1.001941999999e+06
 T314_L_f32_k1: const_Garage_Cars + 1e+06 t314_leaf1 <= 1.000002999999e+06
 T314_L_f119_k1: const_Condition_1_Norm + 1e+06 t314_leaf1
   <= 1.000000999999e+06
 T314_R_f5_k1: const_Year_Built - 1e+06 t314_leaf1 >= -998058
 T314_L_f32_k2: const_Garage_Cars + 1e+06 t314_leaf2 <= 1.000002999999e+06
 T314_R_f119_k2: const_Condition_1_Norm - 1e+06 t314_leaf2 >= -999999
 T314_L_f45_k2: const_MS_SubClass_20 + 1e+06 t314_leaf2
   <= 1.000000999999e+06
 T314_L_f32_k3: const_Garage_Cars + 1e+06 t314_leaf3 <= 1.000002999999e+06
 T314_R_f119_k3: const_Condition_1_Norm - 1e+06 t314_leaf3 >= -999999
 T314_R_f45_k3: const_MS_SubClass_20 - 1e+06 t314_leaf3 >= -999999
 T314_R_f32_k4: const_Garage_Cars - 1e+06 t314_leaf4 >= -999997
 T314_L_f3_k4: Overall_Qual_calc + 1e+06 t314_leaf4 <= 1.000009999999e+06
 T314_L_f272_k4: const_Mo_Sold_2 + 1e+06 t314_leaf4 <= 1.000000999999e+06
 T314_R_f32_k5: const_Garage_Cars - 1e+06 t314_leaf5 >= -999997
 T314_L_f3_k5: Overall_Qual_calc + 1e+06 t314_leaf5 <= 1.000009999999e+06
 T314_R_f272_k5: const_Mo_Sold_2 - 1e+06 t314_leaf5 >= -999999
 T314_R_f32_k6: const_Garage_Cars - 1e+06 t314_leaf6 >= -999997
 T314_R_f3_k6: Overall_Qual_calc - 1e+06 t314_leaf6 >= -999990
 T315_onehot: t315_leaf0 + t315_leaf1 + t315_leaf2 + t315_leaf3
   + t315_leaf4 = 1
 T315_L_f86_k0: const_Land_Slope_Gtl + 1e+06 t315_leaf0
   <= 1.000000999999e+06
 T315_L_f255_k0: const_Garage_Finish_RFn + 1e+06 t315_leaf0
   <= 1.000000999999e+06
 T315_L_f86_k1: const_Land_Slope_Gtl + 1e+06 t315_leaf1
   <= 1.000000999999e+06
 T315_R_f255_k1: const_Garage_Finish_RFn - 1e+06 t315_leaf1 >= -999999
 T315_R_f86_k2: const_Land_Slope_Gtl - 1e+06 t315_leaf2 >= -999999
 T315_L_f20_k2: x_Gr_Liv_Area + 1e+06 t315_leaf2 <= 1.003085999999e+06
 T315_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t315_leaf2
   <= 1.000000999999e+06
 T315_R_f86_k3: const_Land_Slope_Gtl - 1e+06 t315_leaf3 >= -999999
 T315_L_f20_k3: x_Gr_Liv_Area + 1e+06 t315_leaf3 <= 1.003085999999e+06
 T315_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t315_leaf3 >= -999999
 T315_R_f86_k4: const_Land_Slope_Gtl - 1e+06 t315_leaf4 >= -999999
 T315_R_f20_k4: x_Gr_Liv_Area - 1e+06 t315_leaf4 >= -996914
 T316_onehot: t316_leaf0 + t316_leaf1 + t316_leaf2 + t316_leaf3
   + t316_leaf4 = 1
 T316_L_f206_k0: const_Bsmt_Exposure_Av + 1e+06 t316_leaf0
   <= 1.000000999999e+06
 T316_L_f296_k0: const_Sale_Condition_Family + 1e+06 t316_leaf0
   <= 1.000000999999e+06
 T316_L_f9_k0: x_Exter_Cond + 1e+06 t316_leaf0 <= 1.000001999999e+06
 T316_L_f206_k1: const_Bsmt_Exposure_Av + 1e+06 t316_leaf1
   <= 1.000000999999e+06
 T316_L_f296_k1: const_Sale_Condition_Family + 1e+06 t316_leaf1
   <= 1.000000999999e+06
 T316_R_f9_k1: x_Exter_Cond - 1e+06 t316_leaf1 >= -999998
 T316_L_f206_k2: const_Bsmt_Exposure_Av + 1e+06 t316_leaf2
   <= 1.000000999999e+06
 T316_R_f296_k2: const_Sale_Condition_Family - 1e+06 t316_leaf2 >= -999999
 T316_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t316_leaf2 <= 1.000191999999e+06
 T316_L_f206_k3: const_Bsmt_Exposure_Av + 1e+06 t316_leaf3
   <= 1.000000999999e+06
 T316_R_f296_k3: const_Sale_Condition_Family - 1e+06 t316_leaf3 >= -999999
 T316_R_f7_k3: x_Mas_Vnr_Area - 1e+06 t316_leaf3 >= -999808
 T316_R_f206_k4: const_Bsmt_Exposure_Av - 1e+06 t316_leaf4 >= -999999
 T317_onehot: t317_leaf0 + t317_leaf1 + t317_leaf2 + t317_leaf3 = 1
 T317_L_f43_k0: const_Misc_Val + 1e+06 t317_leaf0 <= 1.000499999999e+06
 T317_L_f92_k0: const_Neighborhood_BrkSide + 1e+06 t317_leaf0
   <= 1.000000999999e+06
 T317_L_f209_k0: const_Bsmt_Exposure_No + 1e+06 t317_leaf0
   <= 1.000000999999e+06
 T317_L_f43_k1: const_Misc_Val + 1e+06 t317_leaf1 <= 1.000499999999e+06
 T317_L_f92_k1: const_Neighborhood_BrkSide + 1e+06 t317_leaf1
   <= 1.000000999999e+06
 T317_R_f209_k1: const_Bsmt_Exposure_No - 1e+06 t317_leaf1 >= -999999
 T317_L_f43_k2: const_Misc_Val + 1e+06 t317_leaf2 <= 1.000499999999e+06
 T317_R_f92_k2: const_Neighborhood_BrkSide - 1e+06 t317_leaf2 >= -999999
 T317_R_f43_k3: const_Misc_Val - 1e+06 t317_leaf3 >= -999500
 T318_onehot: t318_leaf0 + t318_leaf1 + t318_leaf2 + t318_leaf3 = 1
 T318_L_f69_k0: const_Street_Pave + 1e+06 t318_leaf0 <= 1.000000999999e+06
 T318_R_f69_k1: const_Street_Pave - 1e+06 t318_leaf1 >= -999999
 T318_L_f36_k1: x_Wood_Deck_SF + 1e+06 t318_leaf1 <= 1.000510999999e+06
 T318_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t318_leaf1 <= 1.000458999999e+06
 T318_R_f69_k2: const_Street_Pave - 1e+06 t318_leaf2 >= -999999
 T318_L_f36_k2: x_Wood_Deck_SF + 1e+06 t318_leaf2 <= 1.000510999999e+06
 T318_R_f7_k2: x_Mas_Vnr_Area - 1e+06 t318_leaf2 >= -999541
 T318_R_f69_k3: const_Street_Pave - 1e+06 t318_leaf3 >= -999999
 T318_R_f36_k3: x_Wood_Deck_SF - 1e+06 t318_leaf3 >= -999489
 T319_onehot: t319_leaf0 + t319_leaf1 + t319_leaf2 + t319_leaf3
   + t319_leaf4 = 1
 T319_L_f44_k0: const_Yr_Sold + 1e+06 t319_leaf0 <= 1.002008999999e+06
 T319_L_f36_k0: x_Wood_Deck_SF + 1e+06 t319_leaf0 <= 1.000466999999e+06
 T319_L_f44_k1: const_Yr_Sold + 1e+06 t319_leaf1 <= 1.002008999999e+06
 T319_R_f36_k1: x_Wood_Deck_SF - 1e+06 t319_leaf1 >= -999533
 T319_R_f44_k2: const_Yr_Sold - 1e+06 t319_leaf2 >= -997991
 T319_L_f0_k2: const_Lot_Frontage + 1e+06 t319_leaf2 <= 1.000084999999e+06
 T319_L_f261_k2: x_fence_is_GdWo + 1e+06 t319_leaf2 <= 1.000000999999e+06
 T319_R_f44_k3: const_Yr_Sold - 1e+06 t319_leaf3 >= -997991
 T319_L_f0_k3: const_Lot_Frontage + 1e+06 t319_leaf3 <= 1.000084999999e+06
 T319_R_f261_k3: x_fence_is_GdWo - 1e+06 t319_leaf3 >= -999999
 T319_R_f44_k4: const_Yr_Sold - 1e+06 t319_leaf4 >= -997991
 T319_R_f0_k4: const_Lot_Frontage - 1e+06 t319_leaf4 >= -999915
 T320_onehot: t320_leaf0 + t320_leaf1 + t320_leaf2 + t320_leaf3
   + t320_leaf4 + t320_leaf5 + t320_leaf6 = 1
 T320_L_f33_k0: x_Garage_Area + 1e+06 t320_leaf0 <= 1.000629999999e+06
 T320_L_f20_k0: x_Gr_Liv_Area + 1e+06 t320_leaf0 <= 1.000746999999e+06
 T320_L_f16_k0: x_Heating_QC + 1e+06 t320_leaf0 <= 1.000002999999e+06
 T320_L_f33_k1: x_Garage_Area + 1e+06 t320_leaf1 <= 1.000629999999e+06
 T320_L_f20_k1: x_Gr_Liv_Area + 1e+06 t320_leaf1 <= 1.000746999999e+06
 T320_R_f16_k1: x_Heating_QC - 1e+06 t320_leaf1 >= -999997
 T320_L_f33_k2: x_Garage_Area + 1e+06 t320_leaf2 <= 1.000629999999e+06
 T320_R_f20_k2: x_Gr_Liv_Area - 1e+06 t320_leaf2 >= -999253
 T320_L_f11_k2: x_Bsmt_Cond + 1e+06 t320_leaf2 <= 1.000002999999e+06
 T320_L_f33_k3: x_Garage_Area + 1e+06 t320_leaf3 <= 1.000629999999e+06
 T320_R_f20_k3: x_Gr_Liv_Area - 1e+06 t320_leaf3 >= -999253
 T320_R_f11_k3: x_Bsmt_Cond - 1e+06 t320_leaf3 >= -999997
 T320_R_f33_k4: x_Garage_Area - 1e+06 t320_leaf4 >= -999370
 T320_L_f4_k4: const_Overall_Cond + 1e+06 t320_leaf4 <= 1.000003999999e+06
 T320_R_f33_k5: x_Garage_Area - 1e+06 t320_leaf5 >= -999370
 T320_R_f4_k5: const_Overall_Cond - 1e+06 t320_leaf5 >= -999996
 T320_L_f33_k5: x_Garage_Area + 1e+06 t320_leaf5 <= 1.001153999999e+06
 T320_R_f33_k6: x_Garage_Area - 1e+06 t320_leaf6 >= -999370
 T320_R_f4_k6: const_Overall_Cond - 1e+06 t320_leaf6 >= -999996
 T320_R_f33_k6: x_Garage_Area - 1e+06 t320_leaf6 >= -998846
 T321_onehot: t321_leaf0 + t321_leaf1 + t321_leaf2 + t321_leaf3
   + t321_leaf4 = 1
 T321_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t321_leaf0 <= 1.000000999999e+06
 T321_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t321_leaf0 <= 1.000000999999e+06
 T321_L_f4_k0: const_Overall_Cond + 1e+06 t321_leaf0 <= 1.000006999999e+06
 T321_L_f177_k1: x_ex2_is_AsbShng + 1e+06 t321_leaf1 <= 1.000000999999e+06
 T321_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t321_leaf1 <= 1.000000999999e+06
 T321_R_f4_k1: const_Overall_Cond - 1e+06 t321_leaf1 >= -999993
 T321_L_f177_k2: x_ex2_is_AsbShng + 1e+06 t321_leaf2 <= 1.000000999999e+06
 T321_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t321_leaf2 >= -999999
 T321_R_f177_k3: x_ex2_is_AsbShng - 1e+06 t321_leaf3 >= -999999
 T321_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t321_leaf3
   <= 1.000720999999e+06
 T321_R_f177_k4: x_ex2_is_AsbShng - 1e+06 t321_leaf4 >= -999999
 T321_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t321_leaf4 >= -999279
 T322_onehot: t322_leaf0 + t322_leaf1 + t322_leaf2 + t322_leaf3
   + t322_leaf4 + t322_leaf5 + t322_leaf6 = 1
 T322_L_f232_k0: central_air_yes + 1e+06 t322_leaf0 <= 1.000000999999e+06
 T322_L_f4_k0: const_Overall_Cond + 1e+06 t322_leaf0 <= 1.000004999999e+06
 T322_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t322_leaf0 <= 1.000726999999e+06
 T322_L_f232_k1: central_air_yes + 1e+06 t322_leaf1 <= 1.000000999999e+06
 T322_L_f4_k1: const_Overall_Cond + 1e+06 t322_leaf1 <= 1.000004999999e+06
 T322_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t322_leaf1 >= -999273
 T322_L_f232_k2: central_air_yes + 1e+06 t322_leaf2 <= 1.000000999999e+06
 T322_R_f4_k2: const_Overall_Cond - 1e+06 t322_leaf2 >= -999995
 T322_L_f34_k2: x_Garage_Qual + 1e+06 t322_leaf2 <= 1.000001999999e+06
 T322_L_f232_k3: central_air_yes + 1e+06 t322_leaf3 <= 1.000000999999e+06
 T322_R_f4_k3: const_Overall_Cond - 1e+06 t322_leaf3 >= -999995
 T322_R_f34_k3: x_Garage_Qual - 1e+06 t322_leaf3 >= -999998
 T322_R_f232_k4: central_air_yes - 1e+06 t322_leaf4 >= -999999
 T322_L_f40_k4: x_Screen_Porch + 1e+06 t322_leaf4 <= 1.000149999999e+06
 T322_L_f4_k4: const_Overall_Cond + 1e+06 t322_leaf4 <= 1.000004999999e+06
 T322_R_f232_k5: central_air_yes - 1e+06 t322_leaf5 >= -999999
 T322_L_f40_k5: x_Screen_Porch + 1e+06 t322_leaf5 <= 1.000149999999e+06
 T322_R_f4_k5: const_Overall_Cond - 1e+06 t322_leaf5 >= -999995
 T322_R_f232_k6: central_air_yes - 1e+06 t322_leaf6 >= -999999
 T322_R_f40_k6: x_Screen_Porch - 1e+06 t322_leaf6 >= -999850
 T323_onehot: t323_leaf0 + t323_leaf1 + t323_leaf2 + t323_leaf3 = 1
 T323_L_f51_k0: const_MS_SubClass_70 + 1e+06 t323_leaf0
   <= 1.000000999999e+06
 T323_L_f147_k0: const_Roof_Style_Flat + 1e+06 t323_leaf0
   <= 1.000000999999e+06
 T323_L_f213_k0: x_b1_is_GLQ + 1e+06 t323_leaf0 <= 1.000000999999e+06
 T323_L_f51_k1: const_MS_SubClass_70 + 1e+06 t323_leaf1
   <= 1.000000999999e+06
 T323_L_f147_k1: const_Roof_Style_Flat + 1e+06 t323_leaf1
   <= 1.000000999999e+06
 T323_R_f213_k1: x_b1_is_GLQ - 1e+06 t323_leaf1 >= -999999
 T323_L_f51_k2: const_MS_SubClass_70 + 1e+06 t323_leaf2
   <= 1.000000999999e+06
 T323_R_f147_k2: const_Roof_Style_Flat - 1e+06 t323_leaf2 >= -999999
 T323_R_f51_k3: const_MS_SubClass_70 - 1e+06 t323_leaf3 >= -999999
 T324_onehot: t324_leaf0 + t324_leaf1 + t324_leaf2 + t324_leaf3
   + t324_leaf4 + t324_leaf5 + t324_leaf6 = 1
 T324_L_f44_k0: const_Yr_Sold + 1e+06 t324_leaf0 <= 1.002007999999e+06
 T324_L_f283_k0: const_Sale_Type_COD + 1e+06 t324_leaf0
   <= 1.000000999999e+06
 T324_L_f233_k0: x_elect_is_FuseA + 1e+06 t324_leaf0 <= 1.000000999999e+06
 T324_L_f44_k1: const_Yr_Sold + 1e+06 t324_leaf1 <= 1.002007999999e+06
 T324_L_f283_k1: const_Sale_Type_COD + 1e+06 t324_leaf1
   <= 1.000000999999e+06
 T324_R_f233_k1: x_elect_is_FuseA - 1e+06 t324_leaf1 >= -999999
 T324_L_f44_k2: const_Yr_Sold + 1e+06 t324_leaf2 <= 1.002007999999e+06
 T324_R_f283_k2: const_Sale_Type_COD - 1e+06 t324_leaf2 >= -999999
 T324_R_f44_k3: const_Yr_Sold - 1e+06 t324_leaf3 >= -997992
 T324_L_f5_k3: const_Year_Built + 1e+06 t324_leaf3 <= 1.001924999999e+06
 T324_L_f6_k3: const_Year_Remod_Add + 1e+06 t324_leaf3
   <= 1.001996999999e+06
 T324_R_f44_k4: const_Yr_Sold - 1e+06 t324_leaf4 >= -997992
 T324_L_f5_k4: const_Year_Built + 1e+06 t324_leaf4 <= 1.001924999999e+06
 T324_R_f6_k4: const_Year_Remod_Add - 1e+06 t324_leaf4 >= -998003
 T324_R_f44_k5: const_Yr_Sold - 1e+06 t324_leaf5 >= -997992
 T324_R_f5_k5: const_Year_Built - 1e+06 t324_leaf5 >= -998075
 T324_L_f219_k5: x_b2_is_BLQ + 1e+06 t324_leaf5 <= 1.000000999999e+06
 T324_R_f44_k6: const_Yr_Sold - 1e+06 t324_leaf6 >= -997992
 T324_R_f5_k6: const_Year_Built - 1e+06 t324_leaf6 >= -998075
 T324_R_f219_k6: x_b2_is_BLQ - 1e+06 t324_leaf6 >= -999999
 T325_onehot: t325_leaf0 + t325_leaf1 + t325_leaf2 + t325_leaf3
   + t325_leaf4 = 1
 T325_L_f33_k0: x_Garage_Area + 1e+06 t325_leaf0 <= 1.000894999999e+06
 T325_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t325_leaf0 <= 1.000000999999e+06
 T325_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t325_leaf0 <= 1.000475999999e+06
 T325_L_f33_k1: x_Garage_Area + 1e+06 t325_leaf1 <= 1.000894999999e+06
 T325_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t325_leaf1 <= 1.000000999999e+06
 T325_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t325_leaf1 >= -999524
 T325_L_f33_k2: x_Garage_Area + 1e+06 t325_leaf2 <= 1.000894999999e+06
 T325_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t325_leaf2 >= -999999
 T325_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t325_leaf2 <= 1.000506999999e+06
 T325_L_f33_k3: x_Garage_Area + 1e+06 t325_leaf3 <= 1.000894999999e+06
 T325_R_f161_k3: x_ex1_is_AsbShng - 1e+06 t325_leaf3 >= -999999
 T325_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t325_leaf3 >= -999493
 T325_R_f33_k4: x_Garage_Area - 1e+06 t325_leaf4 >= -999105
 T326_onehot: t326_leaf0 + t326_leaf1 + t326_leaf2 + t326_leaf3 = 1
 T326_L_f121_k0: const_Condition_1_PosN + 1e+06 t326_leaf0
   <= 1.000000999999e+06
 T326_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t326_leaf0 <= 1.000000999999e+06
 T326_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t326_leaf0
   <= 1.000000999999e+06
 T326_L_f121_k1: const_Condition_1_PosN + 1e+06 t326_leaf1
   <= 1.000000999999e+06
 T326_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t326_leaf1 <= 1.000000999999e+06
 T326_R_f245_k1: const_FunctioNo_aplical_Typ - 1e+06 t326_leaf1 >= -999999
 T326_L_f121_k2: const_Condition_1_PosN + 1e+06 t326_leaf2
   <= 1.000000999999e+06
 T326_R_f164_k2: x_ex1_is_BrkFace - 1e+06 t326_leaf2 >= -999999
 T326_R_f121_k3: const_Condition_1_PosN - 1e+06 t326_leaf3 >= -999999
 T327_onehot: t327_leaf0 + t327_leaf1 + t327_leaf2 + t327_leaf3
   + t327_leaf4 = 1
 T327_L_f113_k0: const_Neighborhood_Somerst + 1e+06 t327_leaf0
   <= 1.000000999999e+06
 T327_L_f4_k0: const_Overall_Cond + 1e+06 t327_leaf0 <= 1.000005999999e+06
 T327_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t327_leaf0
   <= 1.001965999999e+06
 T327_L_f113_k1: const_Neighborhood_Somerst + 1e+06 t327_leaf1
   <= 1.000000999999e+06
 T327_L_f4_k1: const_Overall_Cond + 1e+06 t327_leaf1 <= 1.000005999999e+06
 T327_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t327_leaf1 >= -998034
 T327_L_f113_k2: const_Neighborhood_Somerst + 1e+06 t327_leaf2
   <= 1.000000999999e+06
 T327_R_f4_k2: const_Overall_Cond - 1e+06 t327_leaf2 >= -999994
 T327_L_f17_k2: x_1st_Flr_SF + 1e+06 t327_leaf2 <= 1.001413999999e+06
 T327_L_f113_k3: const_Neighborhood_Somerst + 1e+06 t327_leaf3
   <= 1.000000999999e+06
 T327_R_f4_k3: const_Overall_Cond - 1e+06 t327_leaf3 >= -999994
 T327_R_f17_k3: x_1st_Flr_SF - 1e+06 t327_leaf3 >= -998586
 T327_R_f113_k4: const_Neighborhood_Somerst - 1e+06 t327_leaf4 >= -999999
 T328_onehot: t328_leaf0 + t328_leaf1 + t328_leaf2 + t328_leaf3
   + t328_leaf4 + t328_leaf5 = 1
 T328_L_f16_k0: x_Heating_QC + 1e+06 t328_leaf0 <= 1.000001999999e+06
 T328_L_f191_k0: x_ex2_is_VinylSd + 1e+06 t328_leaf0 <= 1.000000999999e+06
 T328_L_f16_k1: x_Heating_QC + 1e+06 t328_leaf1 <= 1.000001999999e+06
 T328_R_f191_k1: x_ex2_is_VinylSd - 1e+06 t328_leaf1 >= -999999
 T328_R_f16_k2: x_Heating_QC - 1e+06 t328_leaf2 >= -999998
 T328_L_f232_k2: central_air_yes + 1e+06 t328_leaf2 <= 1.000000999999e+06
 T328_L_f217_k2: x_b1_is_Unf + 1e+06 t328_leaf2 <= 1.000000999999e+06
 T328_R_f16_k3: x_Heating_QC - 1e+06 t328_leaf3 >= -999998
 T328_L_f232_k3: central_air_yes + 1e+06 t328_leaf3 <= 1.000000999999e+06
 T328_R_f217_k3: x_b1_is_Unf - 1e+06 t328_leaf3 >= -999999
 T328_R_f16_k4: x_Heating_QC - 1e+06 t328_leaf4 >= -999998
 T328_R_f232_k4: central_air_yes - 1e+06 t328_leaf4 >= -999999
 T328_L_f193_k4: const_Exterior_2nd_Wd_Shng + 1e+06 t328_leaf4
   <= 1.000000999999e+06
 T328_R_f16_k5: x_Heating_QC - 1e+06 t328_leaf5 >= -999998
 T328_R_f232_k5: central_air_yes - 1e+06 t328_leaf5 >= -999999
 T328_R_f193_k5: const_Exterior_2nd_Wd_Shng - 1e+06 t328_leaf5 >= -999999
 T329_onehot: t329_leaf0 + t329_leaf1 + t329_leaf2 + t329_leaf3
   + t329_leaf4 + t329_leaf5 + t329_leaf6 = 1
 T329_L_f3_k0: Overall_Qual_calc + 1e+06 t329_leaf0 <= 1.000006999999e+06
 T329_L_f23_k0: x_Full_Bath + 1e+06 t329_leaf0 <= 1.000001999999e+06
 T329_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t329_leaf0
   <= 1.001388999999e+06
 T329_L_f3_k1: Overall_Qual_calc + 1e+06 t329_leaf1 <= 1.000006999999e+06
 T329_L_f23_k1: x_Full_Bath + 1e+06 t329_leaf1 <= 1.000001999999e+06
 T329_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t329_leaf1 >= -998611
 T329_L_f3_k2: Overall_Qual_calc + 1e+06 t329_leaf2 <= 1.000006999999e+06
 T329_R_f23_k2: x_Full_Bath - 1e+06 t329_leaf2 >= -999998
 T329_L_f20_k2: x_Gr_Liv_Area + 1e+06 t329_leaf2 <= 1.001016999999e+06
 T329_L_f3_k3: Overall_Qual_calc + 1e+06 t329_leaf3 <= 1.000006999999e+06
 T329_R_f23_k3: x_Full_Bath - 1e+06 t329_leaf3 >= -999998
 T329_R_f20_k3: x_Gr_Liv_Area - 1e+06 t329_leaf3 >= -998983
 T329_R_f3_k4: Overall_Qual_calc - 1e+06 t329_leaf4 >= -999993
 T329_L_f56_k4: const_MS_SubClass_120 + 1e+06 t329_leaf4
   <= 1.000000999999e+06
 T329_L_f0_k4: const_Lot_Frontage + 1e+06 t329_leaf4 <= 1.000110999999e+06
 T329_R_f3_k5: Overall_Qual_calc - 1e+06 t329_leaf5 >= -999993
 T329_L_f56_k5: const_MS_SubClass_120 + 1e+06 t329_leaf5
   <= 1.000000999999e+06
 T329_R_f0_k5: const_Lot_Frontage - 1e+06 t329_leaf5 >= -999889
 T329_R_f3_k6: Overall_Qual_calc - 1e+06 t329_leaf6 >= -999993
 T329_R_f56_k6: const_MS_SubClass_120 - 1e+06 t329_leaf6 >= -999999
 T330_onehot: t330_leaf0 + t330_leaf1 + t330_leaf2 + t330_leaf3
   + t330_leaf4 + t330_leaf5 = 1
 T330_L_f1_k0: const_Lot_Area + 1e+06 t330_leaf0 <= 1.018385999999e+06
 T330_L_f27_k0: x_Kitchen_Qual + 1e+06 t330_leaf0 <= 1.000001999999e+06
 T330_L_f1_k0: const_Lot_Area + 1e+06 t330_leaf0 <= 1.008519999999e+06
 T330_L_f1_k1: const_Lot_Area + 1e+06 t330_leaf1 <= 1.018385999999e+06
 T330_L_f27_k1: x_Kitchen_Qual + 1e+06 t330_leaf1 <= 1.000001999999e+06
 T330_R_f1_k1: const_Lot_Area - 1e+06 t330_leaf1 >= -991480
 T330_L_f1_k2: const_Lot_Area + 1e+06 t330_leaf2 <= 1.018385999999e+06
 T330_R_f27_k2: x_Kitchen_Qual - 1e+06 t330_leaf2 >= -999998
 T330_L_f63_k2: const_MS_Zoning_FV + 1e+06 t330_leaf2
   <= 1.000000999999e+06
 T330_L_f1_k3: const_Lot_Area + 1e+06 t330_leaf3 <= 1.018385999999e+06
 T330_R_f27_k3: x_Kitchen_Qual - 1e+06 t330_leaf3 >= -999998
 T330_R_f63_k3: const_MS_Zoning_FV - 1e+06 t330_leaf3 >= -999999
 T330_R_f1_k4: const_Lot_Area - 1e+06 t330_leaf4 >= -981614
 T330_L_f38_k4: x_Enclosed_Porch + 1e+06 t330_leaf4 <= 1.000108999999e+06
 T330_R_f1_k5: const_Lot_Area - 1e+06 t330_leaf5 >= -981614
 T330_R_f38_k5: x_Enclosed_Porch - 1e+06 t330_leaf5 >= -999891
 T331_onehot: t331_leaf0 + t331_leaf1 + t331_leaf2 + t331_leaf3
   + t331_leaf4 = 1
 T331_L_f6_k0: const_Year_Remod_Add + 1e+06 t331_leaf0
   <= 1.002006999999e+06
 T331_L_f0_k0: const_Lot_Frontage + 1e+06 t331_leaf0 <= 1.000129999999e+06
 T331_L_f6_k1: const_Year_Remod_Add + 1e+06 t331_leaf1
   <= 1.002006999999e+06
 T331_R_f0_k1: const_Lot_Frontage - 1e+06 t331_leaf1 >= -999870
 T331_R_f6_k2: const_Year_Remod_Add - 1e+06 t331_leaf2 >= -997993
 T331_L_f0_k2: const_Lot_Frontage + 1e+06 t331_leaf2 <= 1.000113999999e+06
 T331_L_f192_k2: x_ex2_is_Wd_Sdng + 1e+06 t331_leaf2 <= 1.000000999999e+06
 T331_R_f6_k3: const_Year_Remod_Add - 1e+06 t331_leaf3 >= -997993
 T331_L_f0_k3: const_Lot_Frontage + 1e+06 t331_leaf3 <= 1.000113999999e+06
 T331_R_f192_k3: x_ex2_is_Wd_Sdng - 1e+06 t331_leaf3 >= -999999
 T331_R_f6_k4: const_Year_Remod_Add - 1e+06 t331_leaf4 >= -997993
 T331_R_f0_k4: const_Lot_Frontage - 1e+06 t331_leaf4 >= -999886
 T332_onehot: t332_leaf0 + t332_leaf1 + t332_leaf2 + t332_leaf3 = 1
 T332_L_f33_k0: x_Garage_Area + 1e+06 t332_leaf0 <= 1.001153999999e+06
 T332_L_f1_k0: const_Lot_Area + 1e+06 t332_leaf0 <= 1.004434999999e+06
 T332_L_f33_k1: x_Garage_Area + 1e+06 t332_leaf1 <= 1.001153999999e+06
 T332_R_f1_k1: const_Lot_Area - 1e+06 t332_leaf1 >= -995565
 T332_L_f6_k1: const_Year_Remod_Add + 1e+06 t332_leaf1
   <= 1.002006999999e+06
 T332_L_f33_k2: x_Garage_Area + 1e+06 t332_leaf2 <= 1.001153999999e+06
 T332_R_f1_k2: const_Lot_Area - 1e+06 t332_leaf2 >= -995565
 T332_R_f6_k2: const_Year_Remod_Add - 1e+06 t332_leaf2 >= -997993
 T332_R_f33_k3: x_Garage_Area - 1e+06 t332_leaf3 >= -998846
 T333_onehot: t333_leaf0 + t333_leaf1 + t333_leaf2 = 1
 T333_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t333_leaf0 <= 1.000000999999e+06
 T333_R_f161_k1: x_ex1_is_AsbShng - 1e+06 t333_leaf1 >= -999999
 T333_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t333_leaf1 <= 1.000191999999e+06
 T333_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t333_leaf2 >= -999999
 T333_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t333_leaf2 >= -999808
 T334_onehot: t334_leaf0 + t334_leaf1 + t334_leaf2 + t334_leaf3
   + t334_leaf4 + t334_leaf5 + t334_leaf6 = 1
 T334_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t334_leaf0 <= 1.000000999999e+06
 T334_L_f4_k0: const_Overall_Cond + 1e+06 t334_leaf0 <= 1.000005999999e+06
 T334_L_f77_k0: const_Land_Contour_Bnk + 1e+06 t334_leaf0
   <= 1.000000999999e+06
 T334_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t334_leaf1 <= 1.000000999999e+06
 T334_L_f4_k1: const_Overall_Cond + 1e+06 t334_leaf1 <= 1.000005999999e+06
 T334_R_f77_k1: const_Land_Contour_Bnk - 1e+06 t334_leaf1 >= -999999
 T334_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t334_leaf2 <= 1.000000999999e+06
 T334_R_f4_k2: const_Overall_Cond - 1e+06 t334_leaf2 >= -999994
 T334_L_f18_k2: const_2nd_Flr_SF + 1e+06 t334_leaf2 <= 1.000375999999e+06
 T334_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t334_leaf3 <= 1.000000999999e+06
 T334_R_f4_k3: const_Overall_Cond - 1e+06 t334_leaf3 >= -999994
 T334_R_f18_k3: const_2nd_Flr_SF - 1e+06 t334_leaf3 >= -999624
 T334_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t334_leaf4 >= -999999
 T334_L_f31_k4: const_Garage_Yr_Blt + 1e+06 t334_leaf4
   <= 1.001920999999e+06
 T334_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t334_leaf5 >= -999999
 T334_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t334_leaf5 >= -998079
 T334_L_f1_k5: const_Lot_Area + 1e+06 t334_leaf5 <= 1.009604999999e+06
 T334_R_f161_k6: x_ex1_is_AsbShng - 1e+06 t334_leaf6 >= -999999
 T334_R_f31_k6: const_Garage_Yr_Blt - 1e+06 t334_leaf6 >= -998079
 T334_R_f1_k6: const_Lot_Area - 1e+06 t334_leaf6 >= -990395
 T335_onehot: t335_leaf0 + t335_leaf1 + t335_leaf2 + t335_leaf3
   + t335_leaf4 + t335_leaf5 = 1
 T335_L_f277_k0: const_Mo_Sold_7 + 1e+06 t335_leaf0 <= 1.000000999999e+06
 T335_L_f24_k0: x_Half_Bath + 1e+06 t335_leaf0 <= 1.000000999999e+06
 T335_L_f11_k0: x_Bsmt_Cond + 1e+06 t335_leaf0 <= 1.000002999999e+06
 T335_L_f277_k1: const_Mo_Sold_7 + 1e+06 t335_leaf1 <= 1.000000999999e+06
 T335_L_f24_k1: x_Half_Bath + 1e+06 t335_leaf1 <= 1.000000999999e+06
 T335_R_f11_k1: x_Bsmt_Cond - 1e+06 t335_leaf1 >= -999997
 T335_L_f277_k2: const_Mo_Sold_7 + 1e+06 t335_leaf2 <= 1.000000999999e+06
 T335_R_f24_k2: x_Half_Bath - 1e+06 t335_leaf2 >= -999999
 T335_L_f3_k2: Overall_Qual_calc + 1e+06 t335_leaf2 <= 1.000006999999e+06
 T335_L_f277_k3: const_Mo_Sold_7 + 1e+06 t335_leaf3 <= 1.000000999999e+06
 T335_R_f24_k3: x_Half_Bath - 1e+06 t335_leaf3 >= -999999
 T335_R_f3_k3: Overall_Qual_calc - 1e+06 t335_leaf3 >= -999993
 T335_R_f277_k4: const_Mo_Sold_7 - 1e+06 t335_leaf4 >= -999999
 T335_L_f1_k4: const_Lot_Area + 1e+06 t335_leaf4 <= 1.011427999999e+06
 T335_R_f277_k5: const_Mo_Sold_7 - 1e+06 t335_leaf5 >= -999999
 T335_R_f1_k5: const_Lot_Area - 1e+06 t335_leaf5 >= -988572
 T336_onehot: t336_leaf0 + t336_leaf1 + t336_leaf2 + t336_leaf3
   + t336_leaf4 + t336_leaf5 + t336_leaf6 + t336_leaf7 = 1
 T336_L_f0_k0: const_Lot_Frontage + 1e+06 t336_leaf0 <= 1.000084999999e+06
 T336_L_f213_k0: x_b1_is_GLQ + 1e+06 t336_leaf0 <= 1.000000999999e+06
 T336_L_f35_k0: x_Garage_Cond + 1e+06 t336_leaf0 <= 1.000000999999e+06
 T336_L_f0_k1: const_Lot_Frontage + 1e+06 t336_leaf1 <= 1.000084999999e+06
 T336_L_f213_k1: x_b1_is_GLQ + 1e+06 t336_leaf1 <= 1.000000999999e+06
 T336_R_f35_k1: x_Garage_Cond - 1e+06 t336_leaf1 >= -999999
 T336_L_f0_k2: const_Lot_Frontage + 1e+06 t336_leaf2 <= 1.000084999999e+06
 T336_R_f213_k2: x_b1_is_GLQ - 1e+06 t336_leaf2 >= -999999
 T336_L_f1_k2: const_Lot_Area + 1e+06 t336_leaf2 <= 1.012434999999e+06
 T336_L_f0_k3: const_Lot_Frontage + 1e+06 t336_leaf3 <= 1.000084999999e+06
 T336_R_f213_k3: x_b1_is_GLQ - 1e+06 t336_leaf3 >= -999999
 T336_R_f1_k3: const_Lot_Area - 1e+06 t336_leaf3 >= -987565
 T336_R_f0_k4: const_Lot_Frontage - 1e+06 t336_leaf4 >= -999915
 T336_L_f148_k4: const_Roof_Style_Gable + 1e+06 t336_leaf4
   <= 1.000000999999e+06
 T336_L_f86_k4: const_Land_Slope_Gtl + 1e+06 t336_leaf4
   <= 1.000000999999e+06
 T336_R_f0_k5: const_Lot_Frontage - 1e+06 t336_leaf5 >= -999915
 T336_L_f148_k5: const_Roof_Style_Gable + 1e+06 t336_leaf5
   <= 1.000000999999e+06
 T336_R_f86_k5: const_Land_Slope_Gtl - 1e+06 t336_leaf5 >= -999999
 T336_R_f0_k6: const_Lot_Frontage - 1e+06 t336_leaf6 >= -999915
 T336_R_f148_k6: const_Roof_Style_Gable - 1e+06 t336_leaf6 >= -999999
 T336_L_f1_k6: const_Lot_Area + 1e+06 t336_leaf6 <= 1.018385999999e+06
 T336_R_f0_k7: const_Lot_Frontage - 1e+06 t336_leaf7 >= -999915
 T336_R_f148_k7: const_Roof_Style_Gable - 1e+06 t336_leaf7 >= -999999
 T336_R_f1_k7: const_Lot_Area - 1e+06 t336_leaf7 >= -981614
 T337_onehot: t337_leaf0 + t337_leaf1 + t337_leaf2 + t337_leaf3
   + t337_leaf4 + t337_leaf5 = 1
 T337_L_f211_k0: x_b1_is_ALQ + 1e+06 t337_leaf0 <= 1.000000999999e+06
 T337_L_f5_k0: const_Year_Built + 1e+06 t337_leaf0 <= 1.001995999999e+06
 T337_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t337_leaf0 <= 1.000816999999e+06
 T337_L_f211_k1: x_b1_is_ALQ + 1e+06 t337_leaf1 <= 1.000000999999e+06
 T337_L_f5_k1: const_Year_Built + 1e+06 t337_leaf1 <= 1.001995999999e+06
 T337_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t337_leaf1 >= -999183
 T337_L_f211_k2: x_b1_is_ALQ + 1e+06 t337_leaf2 <= 1.000000999999e+06
 T337_R_f5_k2: const_Year_Built - 1e+06 t337_leaf2 >= -998004
 T337_L_f8_k2: x_Exter_Qual + 1e+06 t337_leaf2 <= 1.000002999999e+06
 T337_L_f211_k3: x_b1_is_ALQ + 1e+06 t337_leaf3 <= 1.000000999999e+06
 T337_R_f5_k3: const_Year_Built - 1e+06 t337_leaf3 >= -998004
 T337_R_f8_k3: x_Exter_Qual - 1e+06 t337_leaf3 >= -999997
 T337_R_f211_k4: x_b1_is_ALQ - 1e+06 t337_leaf4 >= -999999
 T337_L_f85_k4: const_Lot_Config_Inside + 1e+06 t337_leaf4
   <= 1.000000999999e+06
 T337_R_f211_k5: x_b1_is_ALQ - 1e+06 t337_leaf5 >= -999999
 T337_R_f85_k5: const_Lot_Config_Inside - 1e+06 t337_leaf5 >= -999999
 T338_onehot: t338_leaf0 + t338_leaf1 + t338_leaf2 + t338_leaf3
   + t338_leaf4 + t338_leaf5 + t338_leaf6 = 1
 T338_L_f30_k0: x_Fireplace_Qu + 1e+06 t338_leaf0 <= 1.000001999999e+06
 T338_L_f9_k0: x_Exter_Cond + 1e+06 t338_leaf0 <= 1.000001999999e+06
 T338_L_f175_k0: x_ex1_is_Wd_Sdng + 1e+06 t338_leaf0 <= 1.000000999999e+06
 T338_L_f30_k1: x_Fireplace_Qu + 1e+06 t338_leaf1 <= 1.000001999999e+06
 T338_L_f9_k1: x_Exter_Cond + 1e+06 t338_leaf1 <= 1.000001999999e+06
 T338_R_f175_k1: x_ex1_is_Wd_Sdng - 1e+06 t338_leaf1 >= -999999
 T338_L_f30_k2: x_Fireplace_Qu + 1e+06 t338_leaf2 <= 1.000001999999e+06
 T338_R_f9_k2: x_Exter_Cond - 1e+06 t338_leaf2 >= -999998
 T338_L_f22_k2: const_Bsmt_Half_Bath + 1e+06 t338_leaf2
   <= 1.000000999999e+06
 T338_L_f30_k3: x_Fireplace_Qu + 1e+06 t338_leaf3 <= 1.000001999999e+06
 T338_R_f9_k3: x_Exter_Cond - 1e+06 t338_leaf3 >= -999998
 T338_R_f22_k3: const_Bsmt_Half_Bath - 1e+06 t338_leaf3 >= -999999
 T338_R_f30_k4: x_Fireplace_Qu - 1e+06 t338_leaf4 >= -999998
 T338_L_f20_k4: x_Gr_Liv_Area + 1e+06 t338_leaf4 <= 1.000875999999e+06
 T338_R_f30_k5: x_Fireplace_Qu - 1e+06 t338_leaf5 >= -999998
 T338_R_f20_k5: x_Gr_Liv_Area - 1e+06 t338_leaf5 >= -999124
 T338_L_f14_k5: x_Bsmt_Unf_SF + 1e+06 t338_leaf5 <= 1.000347999999e+06
 T338_R_f30_k6: x_Fireplace_Qu - 1e+06 t338_leaf6 >= -999998
 T338_R_f20_k6: x_Gr_Liv_Area - 1e+06 t338_leaf6 >= -999124
 T338_R_f14_k6: x_Bsmt_Unf_SF - 1e+06 t338_leaf6 >= -999652
 T339_onehot: t339_leaf0 + t339_leaf1 + t339_leaf2 + t339_leaf3
   + t339_leaf4 + t339_leaf5 = 1
 T339_L_f296_k0: const_Sale_Condition_Family + 1e+06 t339_leaf0
   <= 1.000000999999e+06
 T339_L_f1_k0: const_Lot_Area + 1e+06 t339_leaf0 <= 1.019521999999e+06
 T339_L_f109_k0: const_Neighborhood_OldTown + 1e+06 t339_leaf0
   <= 1.000000999999e+06
 T339_L_f296_k1: const_Sale_Condition_Family + 1e+06 t339_leaf1
   <= 1.000000999999e+06
 T339_L_f1_k1: const_Lot_Area + 1e+06 t339_leaf1 <= 1.019521999999e+06
 T339_R_f109_k1: const_Neighborhood_OldTown - 1e+06 t339_leaf1 >= -999999
 T339_L_f296_k2: const_Sale_Condition_Family + 1e+06 t339_leaf2
   <= 1.000000999999e+06
 T339_R_f1_k2: const_Lot_Area - 1e+06 t339_leaf2 >= -980478
 T339_L_f77_k2: const_Land_Contour_Bnk + 1e+06 t339_leaf2
   <= 1.000000999999e+06
 T339_L_f296_k3: const_Sale_Condition_Family + 1e+06 t339_leaf3
   <= 1.000000999999e+06
 T339_R_f1_k3: const_Lot_Area - 1e+06 t339_leaf3 >= -980478
 T339_R_f77_k3: const_Land_Contour_Bnk - 1e+06 t339_leaf3 >= -999999
 T339_R_f296_k4: const_Sale_Condition_Family - 1e+06 t339_leaf4 >= -999999
 T339_L_f7_k4: x_Mas_Vnr_Area + 1e+06 t339_leaf4 <= 1.000021999999e+06
 T339_R_f296_k5: const_Sale_Condition_Family - 1e+06 t339_leaf5 >= -999999
 T339_R_f7_k5: x_Mas_Vnr_Area - 1e+06 t339_leaf5 >= -999978
 T340_onehot: t340_leaf0 + t340_leaf1 + t340_leaf2 + t340_leaf3
   + t340_leaf4 + t340_leaf5 + t340_leaf6 = 1
 T340_L_f5_k0: const_Year_Built + 1e+06 t340_leaf0 <= 1.001951999999e+06
 T340_L_f5_k0: const_Year_Built + 1e+06 t340_leaf0 <= 1.001914999999e+06
 T340_L_f10_k0: const_Bsmt_Qual + 1e+06 t340_leaf0 <= 1.000001999999e+06
 T340_L_f5_k1: const_Year_Built + 1e+06 t340_leaf1 <= 1.001951999999e+06
 T340_L_f5_k1: const_Year_Built + 1e+06 t340_leaf1 <= 1.001914999999e+06
 T340_R_f10_k1: const_Bsmt_Qual - 1e+06 t340_leaf1 >= -999998
 T340_L_f5_k2: const_Year_Built + 1e+06 t340_leaf2 <= 1.001951999999e+06
 T340_R_f5_k2: const_Year_Built - 1e+06 t340_leaf2 >= -998085
 T340_L_f117_k2: const_Condition_1_Artery + 1e+06 t340_leaf2
   <= 1.000000999999e+06
 T340_L_f5_k3: const_Year_Built + 1e+06 t340_leaf3 <= 1.001951999999e+06
 T340_R_f5_k3: const_Year_Built - 1e+06 t340_leaf3 >= -998085
 T340_R_f117_k3: const_Condition_1_Artery - 1e+06 t340_leaf3 >= -999999
 T340_R_f5_k4: const_Year_Built - 1e+06 t340_leaf4 >= -998048
 T340_L_f6_k4: const_Year_Remod_Add + 1e+06 t340_leaf4
   <= 1.001952999999e+06
 T340_R_f5_k5: const_Year_Built - 1e+06 t340_leaf5 >= -998048
 T340_R_f6_k5: const_Year_Remod_Add - 1e+06 t340_leaf5 >= -998047
 T340_L_f296_k5: const_Sale_Condition_Family + 1e+06 t340_leaf5
   <= 1.000000999999e+06
 T340_R_f5_k6: const_Year_Built - 1e+06 t340_leaf6 >= -998048
 T340_R_f6_k6: const_Year_Remod_Add - 1e+06 t340_leaf6 >= -998047
 T340_R_f296_k6: const_Sale_Condition_Family - 1e+06 t340_leaf6 >= -999999
 T341_onehot: t341_leaf0 + t341_leaf1 + t341_leaf2 = 1
 T341_L_f296_k0: const_Sale_Condition_Family + 1e+06 t341_leaf0
   <= 1.000000999999e+06
 T341_R_f296_k1: const_Sale_Condition_Family - 1e+06 t341_leaf1 >= -999999
 T341_L_f33_k1: x_Garage_Area + 1e+06 t341_leaf1 <= 1.000505999999e+06
 T341_R_f296_k2: const_Sale_Condition_Family - 1e+06 t341_leaf2 >= -999999
 T341_R_f33_k2: x_Garage_Area - 1e+06 t341_leaf2 >= -999494
 T342_onehot: t342_leaf0 + t342_leaf1 + t342_leaf2 + t342_leaf3
   + t342_leaf4 = 1
 T342_L_f71_k0: const_Alley_No_aplica + 1e+06 t342_leaf0
   <= 1.000000999999e+06
 T342_R_f71_k1: const_Alley_No_aplica - 1e+06 t342_leaf1 >= -999999
 T342_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t342_leaf1 <= 1.000428999999e+06
 T342_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t342_leaf1 <= 1.000000999999e+06
 T342_R_f71_k2: const_Alley_No_aplica - 1e+06 t342_leaf2 >= -999999
 T342_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t342_leaf2 <= 1.000428999999e+06
 T342_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t342_leaf2 >= -999999
 T342_R_f71_k3: const_Alley_No_aplica - 1e+06 t342_leaf3 >= -999999
 T342_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t342_leaf3 >= -999571
 T342_L_f96_k3: const_Neighborhood_Edwards + 1e+06 t342_leaf3
   <= 1.000000999999e+06
 T342_R_f71_k4: const_Alley_No_aplica - 1e+06 t342_leaf4 >= -999999
 T342_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t342_leaf4 >= -999571
 T342_R_f96_k4: const_Neighborhood_Edwards - 1e+06 t342_leaf4 >= -999999
 T343_onehot: t343_leaf0 + t343_leaf1 + t343_leaf2 + t343_leaf3
   + t343_leaf4 = 1
 T343_L_f203_k0: const_Foundation_Slab + 1e+06 t343_leaf0
   <= 1.000000999999e+06
 T343_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t343_leaf0 <= 1.001964999999e+06
 T343_L_f1_k0: const_Lot_Area + 1e+06 t343_leaf0 <= 1.004999999999e+06
 T343_L_f203_k1: const_Foundation_Slab + 1e+06 t343_leaf1
   <= 1.000000999999e+06
 T343_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t343_leaf1 <= 1.001964999999e+06
 T343_R_f1_k1: const_Lot_Area - 1e+06 t343_leaf1 >= -995000
 T343_L_f203_k2: const_Foundation_Slab + 1e+06 t343_leaf2
   <= 1.000000999999e+06
 T343_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t343_leaf2 >= -998035
 T343_R_f203_k3: const_Foundation_Slab - 1e+06 t343_leaf3 >= -999999
 T343_L_f0_k3: const_Lot_Frontage + 1e+06 t343_leaf3 <= 1.000065999999e+06
 T343_R_f203_k4: const_Foundation_Slab - 1e+06 t343_leaf4 >= -999999
 T343_R_f0_k4: const_Lot_Frontage - 1e+06 t343_leaf4 >= -999934
 T344_onehot: t344_leaf0 + t344_leaf1 + t344_leaf2 + t344_leaf3
   + t344_leaf4 + t344_leaf5 + t344_leaf6 = 1
 T344_L_f3_k0: Overall_Qual_calc + 1e+06 t344_leaf0 <= 1.000004999999e+06
 T344_L_f272_k0: const_Mo_Sold_2 + 1e+06 t344_leaf0 <= 1.000000999999e+06
 T344_L_f1_k0: const_Lot_Area + 1e+06 t344_leaf0 <= 1.005975999999e+06
 T344_L_f3_k1: Overall_Qual_calc + 1e+06 t344_leaf1 <= 1.000004999999e+06
 T344_L_f272_k1: const_Mo_Sold_2 + 1e+06 t344_leaf1 <= 1.000000999999e+06
 T344_R_f1_k1: const_Lot_Area - 1e+06 t344_leaf1 >= -994024
 T344_L_f3_k2: Overall_Qual_calc + 1e+06 t344_leaf2 <= 1.000004999999e+06
 T344_R_f272_k2: const_Mo_Sold_2 - 1e+06 t344_leaf2 >= -999999
 T344_R_f3_k3: Overall_Qual_calc - 1e+06 t344_leaf3 >= -999995
 T344_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t344_leaf3 <= 1.000935999999e+06
 T344_L_f1_k3: const_Lot_Area + 1e+06 t344_leaf3 <= 1.008399999999e+06
 T344_R_f3_k4: Overall_Qual_calc - 1e+06 t344_leaf4 >= -999995
 T344_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t344_leaf4 <= 1.000935999999e+06
 T344_R_f1_k4: const_Lot_Area - 1e+06 t344_leaf4 >= -991600
 T344_R_f3_k5: Overall_Qual_calc - 1e+06 t344_leaf5 >= -999995
 T344_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t344_leaf5 >= -999064
 T344_L_f1_k5: const_Lot_Area + 1e+06 t344_leaf5 <= 1.021369999999e+06
 T344_R_f3_k6: Overall_Qual_calc - 1e+06 t344_leaf6 >= -999995
 T344_R_f12_k6: x_BsmtFin_SF_1 - 1e+06 t344_leaf6 >= -999064
 T344_R_f1_k6: const_Lot_Area - 1e+06 t344_leaf6 >= -978630
 T345_onehot: t345_leaf0 + t345_leaf1 + t345_leaf2 + t345_leaf3
   + t345_leaf4 + t345_leaf5 + t345_leaf6 = 1
 T345_L_f16_k0: x_Heating_QC + 1e+06 t345_leaf0 <= 1.000002999999e+06
 T345_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t345_leaf0
   <= 1.000000999999e+06
 T345_L_f209_k0: const_Bsmt_Exposure_No + 1e+06 t345_leaf0
   <= 1.000000999999e+06
 T345_L_f16_k1: x_Heating_QC + 1e+06 t345_leaf1 <= 1.000002999999e+06
 T345_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t345_leaf1
   <= 1.000000999999e+06
 T345_R_f209_k1: const_Bsmt_Exposure_No - 1e+06 t345_leaf1 >= -999999
 T345_L_f16_k2: x_Heating_QC + 1e+06 t345_leaf2 <= 1.000002999999e+06
 T345_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t345_leaf2 >= -999999
 T345_L_f3_k2: Overall_Qual_calc + 1e+06 t345_leaf2 <= 1.000003999999e+06
 T345_L_f16_k3: x_Heating_QC + 1e+06 t345_leaf3 <= 1.000002999999e+06
 T345_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t345_leaf3 >= -999999
 T345_R_f3_k3: Overall_Qual_calc - 1e+06 t345_leaf3 >= -999996
 T345_R_f16_k4: x_Heating_QC - 1e+06 t345_leaf4 >= -999997
 T345_L_f3_k4: Overall_Qual_calc + 1e+06 t345_leaf4 <= 1.000005999999e+06
 T345_R_f16_k5: x_Heating_QC - 1e+06 t345_leaf5 >= -999997
 T345_R_f3_k5: Overall_Qual_calc - 1e+06 t345_leaf5 >= -999994
 T345_L_f96_k5: const_Neighborhood_Edwards + 1e+06 t345_leaf5
   <= 1.000000999999e+06
 T345_R_f16_k6: x_Heating_QC - 1e+06 t345_leaf6 >= -999997
 T345_R_f3_k6: Overall_Qual_calc - 1e+06 t345_leaf6 >= -999994
 T345_R_f96_k6: const_Neighborhood_Edwards - 1e+06 t345_leaf6 >= -999999
 T346_onehot: t346_leaf0 + t346_leaf1 + t346_leaf2 + t346_leaf3
   + t346_leaf4 = 1
 T346_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t346_leaf0 <= 1.000251999999e+06
 T346_L_f8_k0: x_Exter_Qual + 1e+06 t346_leaf0 <= 1.000002999999e+06
 T346_L_f16_k0: x_Heating_QC + 1e+06 t346_leaf0 <= 1.000001999999e+06
 T346_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t346_leaf1 <= 1.000251999999e+06
 T346_L_f8_k1: x_Exter_Qual + 1e+06 t346_leaf1 <= 1.000002999999e+06
 T346_R_f16_k1: x_Heating_QC - 1e+06 t346_leaf1 >= -999998
 T346_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t346_leaf2 <= 1.000251999999e+06
 T346_R_f8_k2: x_Exter_Qual - 1e+06 t346_leaf2 >= -999997
 T346_L_f17_k2: x_1st_Flr_SF + 1e+06 t346_leaf2 <= 1.001781999999e+06
 T346_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t346_leaf3 <= 1.000251999999e+06
 T346_R_f8_k3: x_Exter_Qual - 1e+06 t346_leaf3 >= -999997
 T346_R_f17_k3: x_1st_Flr_SF - 1e+06 t346_leaf3 >= -998218
 T346_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t346_leaf4 >= -999748
 T347_onehot: t347_leaf0 + t347_leaf1 + t347_leaf2 + t347_leaf3 = 1
 T347_L_f38_k0: x_Enclosed_Porch + 1e+06 t347_leaf0 <= 1.000289999999e+06
 T347_L_f0_k0: const_Lot_Frontage + 1e+06 t347_leaf0 <= 1.000129999999e+06
 T347_L_f121_k0: const_Condition_1_PosN + 1e+06 t347_leaf0
   <= 1.000000999999e+06
 T347_L_f38_k1: x_Enclosed_Porch + 1e+06 t347_leaf1 <= 1.000289999999e+06
 T347_L_f0_k1: const_Lot_Frontage + 1e+06 t347_leaf1 <= 1.000129999999e+06
 T347_R_f121_k1: const_Condition_1_PosN - 1e+06 t347_leaf1 >= -999999
 T347_L_f38_k2: x_Enclosed_Porch + 1e+06 t347_leaf2 <= 1.000289999999e+06
 T347_R_f0_k2: const_Lot_Frontage - 1e+06 t347_leaf2 >= -999870
 T347_R_f38_k3: x_Enclosed_Porch - 1e+06 t347_leaf3 >= -999710
 T348_onehot: t348_leaf0 + t348_leaf1 + t348_leaf2 + t348_leaf3
   + t348_leaf4 = 1
 T348_L_f40_k0: x_Screen_Porch + 1e+06 t348_leaf0 <= 1.000107999999e+06
 T348_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t348_leaf0 <= 1.000458999999e+06
 T348_L_f44_k0: const_Yr_Sold + 1e+06 t348_leaf0 <= 1.002006999999e+06
 T348_L_f40_k1: x_Screen_Porch + 1e+06 t348_leaf1 <= 1.000107999999e+06
 T348_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t348_leaf1 <= 1.000458999999e+06
 T348_R_f44_k1: const_Yr_Sold - 1e+06 t348_leaf1 >= -997993
 T348_L_f40_k2: x_Screen_Porch + 1e+06 t348_leaf2 <= 1.000107999999e+06
 T348_R_f7_k2: x_Mas_Vnr_Area - 1e+06 t348_leaf2 >= -999541
 T348_R_f40_k3: x_Screen_Porch - 1e+06 t348_leaf3 >= -999892
 T348_L_f217_k3: x_b1_is_Unf + 1e+06 t348_leaf3 <= 1.000000999999e+06
 T348_R_f40_k4: x_Screen_Porch - 1e+06 t348_leaf4 >= -999892
 T348_R_f217_k4: x_b1_is_Unf - 1e+06 t348_leaf4 >= -999999
 T349_onehot: t349_leaf0 + t349_leaf1 + t349_leaf2 + t349_leaf3
   + t349_leaf4 = 1
 T349_L_f0_k0: const_Lot_Frontage + 1e+06 t349_leaf0 <= 1.000119999999e+06
 T349_L_f17_k0: x_1st_Flr_SF + 1e+06 t349_leaf0 <= 1.002120999999e+06
 T349_L_f38_k0: x_Enclosed_Porch + 1e+06 t349_leaf0 <= 1.000264999999e+06
 T349_L_f0_k1: const_Lot_Frontage + 1e+06 t349_leaf1 <= 1.000119999999e+06
 T349_L_f17_k1: x_1st_Flr_SF + 1e+06 t349_leaf1 <= 1.002120999999e+06
 T349_R_f38_k1: x_Enclosed_Porch - 1e+06 t349_leaf1 >= -999735
 T349_L_f0_k2: const_Lot_Frontage + 1e+06 t349_leaf2 <= 1.000119999999e+06
 T349_R_f17_k2: x_1st_Flr_SF - 1e+06 t349_leaf2 >= -997879
 T349_R_f0_k3: const_Lot_Frontage - 1e+06 t349_leaf3 >= -999880
 T349_L_f77_k3: const_Land_Contour_Bnk + 1e+06 t349_leaf3
   <= 1.000000999999e+06
 T349_R_f0_k4: const_Lot_Frontage - 1e+06 t349_leaf4 >= -999880
 T349_R_f77_k4: const_Land_Contour_Bnk - 1e+06 t349_leaf4 >= -999999
 T350_onehot: t350_leaf0 + t350_leaf1 + t350_leaf2 + t350_leaf3
   + t350_leaf4 + t350_leaf5 + t350_leaf6 = 1
 T350_L_f9_k0: x_Exter_Cond + 1e+06 t350_leaf0 <= 1.000001999999e+06
 T350_L_f0_k0: const_Lot_Frontage + 1e+06 t350_leaf0 <= 1.000065999999e+06
 T350_L_f9_k1: x_Exter_Cond + 1e+06 t350_leaf1 <= 1.000001999999e+06
 T350_R_f0_k1: const_Lot_Frontage - 1e+06 t350_leaf1 >= -999934
 T350_L_f18_k1: const_2nd_Flr_SF + 1e+06 t350_leaf1 <= 1.000166999999e+06
 T350_L_f9_k2: x_Exter_Cond + 1e+06 t350_leaf2 <= 1.000001999999e+06
 T350_R_f0_k2: const_Lot_Frontage - 1e+06 t350_leaf2 >= -999934
 T350_R_f18_k2: const_2nd_Flr_SF - 1e+06 t350_leaf2 >= -999833
 T350_R_f9_k3: x_Exter_Cond - 1e+06 t350_leaf3 >= -999998
 T350_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t350_leaf3
   <= 1.001940999999e+06
 T350_L_f14_k3: x_Bsmt_Unf_SF + 1e+06 t350_leaf3 <= 1.000382999999e+06
 T350_R_f9_k4: x_Exter_Cond - 1e+06 t350_leaf4 >= -999998
 T350_L_f31_k4: const_Garage_Yr_Blt + 1e+06 t350_leaf4
   <= 1.001940999999e+06
 T350_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t350_leaf4 >= -999617
 T350_R_f9_k5: x_Exter_Cond - 1e+06 t350_leaf5 >= -999998
 T350_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t350_leaf5 >= -998059
 T350_L_f14_k5: x_Bsmt_Unf_SF + 1e+06 t350_leaf5 <= 1.000475999999e+06
 T350_R_f9_k6: x_Exter_Cond - 1e+06 t350_leaf6 >= -999998
 T350_R_f31_k6: const_Garage_Yr_Blt - 1e+06 t350_leaf6 >= -998059
 T350_R_f14_k6: x_Bsmt_Unf_SF - 1e+06 t350_leaf6 >= -999524
 T351_onehot: t351_leaf0 + t351_leaf1 + t351_leaf2 + t351_leaf3
   + t351_leaf4 + t351_leaf5 = 1
 T351_L_f44_k0: const_Yr_Sold + 1e+06 t351_leaf0 <= 1.002009999999e+06
 T351_L_f3_k0: Overall_Qual_calc + 1e+06 t351_leaf0 <= 1.000004999999e+06
 T351_L_f44_k1: const_Yr_Sold + 1e+06 t351_leaf1 <= 1.002009999999e+06
 T351_R_f3_k1: Overall_Qual_calc - 1e+06 t351_leaf1 >= -999995
 T351_L_f60_k1: const_MS_SubClass_190 + 1e+06 t351_leaf1
   <= 1.000000999999e+06
 T351_L_f44_k2: const_Yr_Sold + 1e+06 t351_leaf2 <= 1.002009999999e+06
 T351_R_f3_k2: Overall_Qual_calc - 1e+06 t351_leaf2 >= -999995
 T351_R_f60_k2: const_MS_SubClass_190 - 1e+06 t351_leaf2 >= -999999
 T351_R_f44_k3: const_Yr_Sold - 1e+06 t351_leaf3 >= -997990
 T351_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t351_leaf3 <= 1.000323999999e+06
 T351_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t351_leaf3
   <= 1.000529999999e+06
 T351_R_f44_k4: const_Yr_Sold - 1e+06 t351_leaf4 >= -997990
 T351_L_f7_k4: x_Mas_Vnr_Area + 1e+06 t351_leaf4 <= 1.000323999999e+06
 T351_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t351_leaf4 >= -999470
 T351_R_f44_k5: const_Yr_Sold - 1e+06 t351_leaf5 >= -997990
 T351_R_f7_k5: x_Mas_Vnr_Area - 1e+06 t351_leaf5 >= -999676
 T352_onehot: t352_leaf0 + t352_leaf1 = 1
 T352_L_f37_k0: x_Open_Porch_SF + 1e+06 t352_leaf0 <= 1.000188999999e+06
 T352_R_f37_k1: x_Open_Porch_SF - 1e+06 t352_leaf1 >= -999811
 T353_onehot: t353_leaf0 + t353_leaf1 + t353_leaf2 + t353_leaf3
   + t353_leaf4 = 1
 T353_L_f37_k0: x_Open_Porch_SF + 1e+06 t353_leaf0 <= 1.000286999999e+06
 T353_L_f20_k0: x_Gr_Liv_Area + 1e+06 t353_leaf0 <= 1.003085999999e+06
 T353_L_f147_k0: const_Roof_Style_Flat + 1e+06 t353_leaf0
   <= 1.000000999999e+06
 T353_L_f37_k1: x_Open_Porch_SF + 1e+06 t353_leaf1 <= 1.000286999999e+06
 T353_L_f20_k1: x_Gr_Liv_Area + 1e+06 t353_leaf1 <= 1.003085999999e+06
 T353_R_f147_k1: const_Roof_Style_Flat - 1e+06 t353_leaf1 >= -999999
 T353_L_f37_k2: x_Open_Porch_SF + 1e+06 t353_leaf2 <= 1.000286999999e+06
 T353_R_f20_k2: x_Gr_Liv_Area - 1e+06 t353_leaf2 >= -996914
 T353_R_f37_k3: x_Open_Porch_SF - 1e+06 t353_leaf3 >= -999713
 T353_L_f28_k3: x_TotRms_AbvGrd + 1e+06 t353_leaf3 <= 1.000009999999e+06
 T353_R_f37_k4: x_Open_Porch_SF - 1e+06 t353_leaf4 >= -999713
 T353_R_f28_k4: x_TotRms_AbvGrd - 1e+06 t353_leaf4 >= -999990
 T354_onehot: t354_leaf0 + t354_leaf1 + t354_leaf2 = 1
 T354_L_f5_k0: const_Year_Built + 1e+06 t354_leaf0 <= 1.001910999999e+06
 T354_L_f17_k0: x_1st_Flr_SF + 1e+06 t354_leaf0 <= 1.000859999999e+06
 T354_L_f5_k1: const_Year_Built + 1e+06 t354_leaf1 <= 1.001910999999e+06
 T354_R_f17_k1: x_1st_Flr_SF - 1e+06 t354_leaf1 >= -999140
 T354_R_f5_k2: const_Year_Built - 1e+06 t354_leaf2 >= -998089
 T355_onehot: t355_leaf0 + t355_leaf1 + t355_leaf2 + t355_leaf3
   + t355_leaf4 + t355_leaf5 + t355_leaf6 = 1
 T355_L_f251_k0: const_Garage_Type_Detchd + 1e+06 t355_leaf0
   <= 1.000000999999e+06
 T355_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t355_leaf0
   <= 1.000647999999e+06
 T355_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t355_leaf0
   <= 1.001994999999e+06
 T355_L_f251_k1: const_Garage_Type_Detchd + 1e+06 t355_leaf1
   <= 1.000000999999e+06
 T355_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t355_leaf1
   <= 1.000647999999e+06
 T355_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t355_leaf1 >= -998005
 T355_L_f251_k2: const_Garage_Type_Detchd + 1e+06 t355_leaf2
   <= 1.000000999999e+06
 T355_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t355_leaf2 >= -999352
 T355_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t355_leaf2
   <= 1.000720999999e+06
 T355_L_f251_k3: const_Garage_Type_Detchd + 1e+06 t355_leaf3
   <= 1.000000999999e+06
 T355_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t355_leaf3 >= -999352
 T355_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t355_leaf3 >= -999279
 T355_R_f251_k4: const_Garage_Type_Detchd - 1e+06 t355_leaf4 >= -999999
 T355_L_f1_k4: const_Lot_Area + 1e+06 t355_leaf4 <= 1.011024999999e+06
 T355_R_f251_k5: const_Garage_Type_Detchd - 1e+06 t355_leaf5 >= -999999
 T355_R_f1_k5: const_Lot_Area - 1e+06 t355_leaf5 >= -988975
 T355_L_f292_k5: const_Sale_Type_WD_ + 1e+06 t355_leaf5
   <= 1.000000999999e+06
 T355_R_f251_k6: const_Garage_Type_Detchd - 1e+06 t355_leaf6 >= -999999
 T355_R_f1_k6: const_Lot_Area - 1e+06 t355_leaf6 >= -988975
 T355_R_f292_k6: const_Sale_Type_WD_ - 1e+06 t355_leaf6 >= -999999
 T356_onehot: t356_leaf0 + t356_leaf1 + t356_leaf2 + t356_leaf3
   + t356_leaf4 + t356_leaf5 + t356_leaf6 = 1
 T356_L_f4_k0: const_Overall_Cond + 1e+06 t356_leaf0 <= 1.000004999999e+06
 T356_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t356_leaf0
   <= 1.000726999999e+06
 T356_L_f17_k0: x_1st_Flr_SF + 1e+06 t356_leaf0 <= 1.000615999999e+06
 T356_L_f4_k1: const_Overall_Cond + 1e+06 t356_leaf1 <= 1.000004999999e+06
 T356_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t356_leaf1
   <= 1.000726999999e+06
 T356_R_f17_k1: x_1st_Flr_SF - 1e+06 t356_leaf1 >= -999384
 T356_L_f4_k2: const_Overall_Cond + 1e+06 t356_leaf2 <= 1.000004999999e+06
 T356_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t356_leaf2 >= -999273
 T356_R_f4_k3: const_Overall_Cond - 1e+06 t356_leaf3 >= -999995
 T356_L_f18_k3: const_2nd_Flr_SF + 1e+06 t356_leaf3 <= 1.000833999999e+06
 T356_L_f72_k3: const_Alley_Pave + 1e+06 t356_leaf3 <= 1.000000999999e+06
 T356_R_f4_k4: const_Overall_Cond - 1e+06 t356_leaf4 >= -999995
 T356_L_f18_k4: const_2nd_Flr_SF + 1e+06 t356_leaf4 <= 1.000833999999e+06
 T356_R_f72_k4: const_Alley_Pave - 1e+06 t356_leaf4 >= -999999
 T356_R_f4_k5: const_Overall_Cond - 1e+06 t356_leaf5 >= -999995
 T356_R_f18_k5: const_2nd_Flr_SF - 1e+06 t356_leaf5 >= -999166
 T356_L_f3_k5: Overall_Qual_calc + 1e+06 t356_leaf5 <= 1.000006999999e+06
 T356_R_f4_k6: const_Overall_Cond - 1e+06 t356_leaf6 >= -999995
 T356_R_f18_k6: const_2nd_Flr_SF - 1e+06 t356_leaf6 >= -999166
 T356_R_f3_k6: Overall_Qual_calc - 1e+06 t356_leaf6 >= -999993
 T357_onehot: t357_leaf0 + t357_leaf1 + t357_leaf2 + t357_leaf3 = 1
 T357_L_f273_k0: const_Mo_Sold_3 + 1e+06 t357_leaf0 <= 1.000000999999e+06
 T357_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t357_leaf0
   <= 1.000000999999e+06
 T357_L_f68_k0: const_Street_Grvl + 1e+06 t357_leaf0 <= 1.000000999999e+06
 T357_L_f273_k1: const_Mo_Sold_3 + 1e+06 t357_leaf1 <= 1.000000999999e+06
 T357_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t357_leaf1
   <= 1.000000999999e+06
 T357_R_f68_k1: const_Street_Grvl - 1e+06 t357_leaf1 >= -999999
 T357_L_f273_k2: const_Mo_Sold_3 + 1e+06 t357_leaf2 <= 1.000000999999e+06
 T357_R_f95_k2: const_Neighborhood_Crawfor - 1e+06 t357_leaf2 >= -999999
 T357_R_f273_k3: const_Mo_Sold_3 - 1e+06 t357_leaf3 >= -999999
 T358_onehot: t358_leaf0 + t358_leaf1 + t358_leaf2 + t358_leaf3
   + t358_leaf4 + t358_leaf5 + t358_leaf6 = 1
 T358_L_f231_k0: central_air_no + 1e+06 t358_leaf0 <= 1.000000999999e+06
 T358_L_f82_k0: const_Lot_Config_CulDSac + 1e+06 t358_leaf0
   <= 1.000000999999e+06
 T358_L_f17_k0: x_1st_Flr_SF + 1e+06 t358_leaf0 <= 1.001452999999e+06
 T358_L_f231_k1: central_air_no + 1e+06 t358_leaf1 <= 1.000000999999e+06
 T358_L_f82_k1: const_Lot_Config_CulDSac + 1e+06 t358_leaf1
   <= 1.000000999999e+06
 T358_R_f17_k1: x_1st_Flr_SF - 1e+06 t358_leaf1 >= -998547
 T358_L_f231_k2: central_air_no + 1e+06 t358_leaf2 <= 1.000000999999e+06
 T358_R_f82_k2: const_Lot_Config_CulDSac - 1e+06 t358_leaf2 >= -999999
 T358_R_f231_k3: central_air_no - 1e+06 t358_leaf3 >= -999999
 T358_L_f4_k3: const_Overall_Cond + 1e+06 t358_leaf3 <= 1.000004999999e+06
 T358_L_f17_k3: x_1st_Flr_SF + 1e+06 t358_leaf3 <= 1.000935999999e+06
 T358_R_f231_k4: central_air_no - 1e+06 t358_leaf4 >= -999999
 T358_L_f4_k4: const_Overall_Cond + 1e+06 t358_leaf4 <= 1.000004999999e+06
 T358_R_f17_k4: x_1st_Flr_SF - 1e+06 t358_leaf4 >= -999064
 T358_R_f231_k5: central_air_no - 1e+06 t358_leaf5 >= -999999
 T358_R_f4_k5: const_Overall_Cond - 1e+06 t358_leaf5 >= -999995
 T358_L_f92_k5: const_Neighborhood_BrkSide + 1e+06 t358_leaf5
   <= 1.000000999999e+06
 T358_R_f231_k6: central_air_no - 1e+06 t358_leaf6 >= -999999
 T358_R_f4_k6: const_Overall_Cond - 1e+06 t358_leaf6 >= -999995
 T358_R_f92_k6: const_Neighborhood_BrkSide - 1e+06 t358_leaf6 >= -999999
 T359_onehot: t359_leaf0 + t359_leaf1 + t359_leaf2 + t359_leaf3
   + t359_leaf4 + t359_leaf5 + t359_leaf6 = 1
 T359_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t359_leaf0 <= 1.000000999999e+06
 T359_L_f114_k0: const_Neighborhood_StoneBr + 1e+06 t359_leaf0
   <= 1.000000999999e+06
 T359_L_f8_k0: x_Exter_Qual + 1e+06 t359_leaf0 <= 1.000002999999e+06
 T359_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t359_leaf1 <= 1.000000999999e+06
 T359_L_f114_k1: const_Neighborhood_StoneBr + 1e+06 t359_leaf1
   <= 1.000000999999e+06
 T359_R_f8_k1: x_Exter_Qual - 1e+06 t359_leaf1 >= -999997
 T359_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t359_leaf2 <= 1.000000999999e+06
 T359_R_f114_k2: const_Neighborhood_StoneBr - 1e+06 t359_leaf2 >= -999999
 T359_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t359_leaf2
   <= 1.001399999999e+06
 T359_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t359_leaf3 <= 1.000000999999e+06
 T359_R_f114_k3: const_Neighborhood_StoneBr - 1e+06 t359_leaf3 >= -999999
 T359_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t359_leaf3 >= -998600
 T359_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t359_leaf4 >= -999999
 T359_L_f33_k4: x_Garage_Area + 1e+06 t359_leaf4 <= 1.000249999999e+06
 T359_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t359_leaf5 >= -999999
 T359_R_f33_k5: x_Garage_Area - 1e+06 t359_leaf5 >= -999750
 T359_L_f15_k5: const_Total_Bsmt_SF + 1e+06 t359_leaf5
   <= 1.000790999999e+06
 T359_R_f161_k6: x_ex1_is_AsbShng - 1e+06 t359_leaf6 >= -999999
 T359_R_f33_k6: x_Garage_Area - 1e+06 t359_leaf6 >= -999750
 T359_R_f15_k6: const_Total_Bsmt_SF - 1e+06 t359_leaf6 >= -999209
 T360_onehot: t360_leaf0 + t360_leaf1 = 1
 T360_L_f3_k0: Overall_Qual_calc + 1e+06 t360_leaf0 <= 1.000002999999e+06
 T360_R_f3_k1: Overall_Qual_calc - 1e+06 t360_leaf1 >= -999997
 T361_onehot: t361_leaf0 + t361_leaf1 + t361_leaf2 + t361_leaf3 = 1
 T361_L_f173_k0: x_ex1_is_Stucco + 1e+06 t361_leaf0 <= 1.000000999999e+06
 T361_L_f17_k0: x_1st_Flr_SF + 1e+06 t361_leaf0 <= 1.000645999999e+06
 T361_L_f173_k1: x_ex1_is_Stucco + 1e+06 t361_leaf1 <= 1.000000999999e+06
 T361_R_f17_k1: x_1st_Flr_SF - 1e+06 t361_leaf1 >= -999354
 T361_L_f33_k1: x_Garage_Area + 1e+06 t361_leaf1 <= 1.000296999999e+06
 T361_L_f173_k2: x_ex1_is_Stucco + 1e+06 t361_leaf2 <= 1.000000999999e+06
 T361_R_f17_k2: x_1st_Flr_SF - 1e+06 t361_leaf2 >= -999354
 T361_R_f33_k2: x_Garage_Area - 1e+06 t361_leaf2 >= -999703
 T361_R_f173_k3: x_ex1_is_Stucco - 1e+06 t361_leaf3 >= -999999
 T362_onehot: t362_leaf0 + t362_leaf1 + t362_leaf2 + t362_leaf3
   + t362_leaf4 = 1
 T362_L_f5_k0: const_Year_Built + 1e+06 t362_leaf0 <= 1.001914999999e+06
 T362_L_f4_k0: const_Overall_Cond + 1e+06 t362_leaf0 <= 1.000006999999e+06
 T362_L_f5_k1: const_Year_Built + 1e+06 t362_leaf1 <= 1.001914999999e+06
 T362_R_f4_k1: const_Overall_Cond - 1e+06 t362_leaf1 >= -999993
 T362_R_f5_k2: const_Year_Built - 1e+06 t362_leaf2 >= -998085
 T362_L_f18_k2: const_2nd_Flr_SF + 1e+06 t362_leaf2 <= 1.000357999999e+06
 T362_R_f5_k3: const_Year_Built - 1e+06 t362_leaf3 >= -998085
 T362_R_f18_k3: const_2nd_Flr_SF - 1e+06 t362_leaf3 >= -999642
 T362_L_f3_k3: Overall_Qual_calc + 1e+06 t362_leaf3 <= 1.000006999999e+06
 T362_R_f5_k4: const_Year_Built - 1e+06 t362_leaf4 >= -998085
 T362_R_f18_k4: const_2nd_Flr_SF - 1e+06 t362_leaf4 >= -999642
 T362_R_f3_k4: Overall_Qual_calc - 1e+06 t362_leaf4 >= -999993
 T363_onehot: t363_leaf0 + t363_leaf1 + t363_leaf2 + t363_leaf3
   + t363_leaf4 = 1
 T363_L_f296_k0: const_Sale_Condition_Family + 1e+06 t363_leaf0
   <= 1.000000999999e+06
 T363_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t363_leaf0 <= 1.001392999999e+06
 T363_L_f216_k0: x_b1_is_Rec + 1e+06 t363_leaf0 <= 1.000000999999e+06
 T363_L_f296_k1: const_Sale_Condition_Family + 1e+06 t363_leaf1
   <= 1.000000999999e+06
 T363_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t363_leaf1 <= 1.001392999999e+06
 T363_R_f216_k1: x_b1_is_Rec - 1e+06 t363_leaf1 >= -999999
 T363_L_f296_k2: const_Sale_Condition_Family + 1e+06 t363_leaf2
   <= 1.000000999999e+06
 T363_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t363_leaf2 >= -998607
 T363_R_f296_k3: const_Sale_Condition_Family - 1e+06 t363_leaf3 >= -999999
 T363_L_f14_k3: x_Bsmt_Unf_SF + 1e+06 t363_leaf3 <= 1.000506999999e+06
 T363_R_f296_k4: const_Sale_Condition_Family - 1e+06 t363_leaf4 >= -999999
 T363_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t363_leaf4 >= -999493
 T364_onehot: t364_leaf0 + t364_leaf1 + t364_leaf2 + t364_leaf3
   + t364_leaf4 = 1
 T364_L_f68_k0: const_Street_Grvl + 1e+06 t364_leaf0 <= 1.000000999999e+06
 T364_L_f232_k0: central_air_yes + 1e+06 t364_leaf0 <= 1.000000999999e+06
 T364_L_f1_k0: const_Lot_Area + 1e+06 t364_leaf0 <= 1.007919999999e+06
 T364_L_f68_k1: const_Street_Grvl + 1e+06 t364_leaf1 <= 1.000000999999e+06
 T364_L_f232_k1: central_air_yes + 1e+06 t364_leaf1 <= 1.000000999999e+06
 T364_R_f1_k1: const_Lot_Area - 1e+06 t364_leaf1 >= -992080
 T364_L_f68_k2: const_Street_Grvl + 1e+06 t364_leaf2 <= 1.000000999999e+06
 T364_R_f232_k2: central_air_yes - 1e+06 t364_leaf2 >= -999999
 T364_L_f206_k2: const_Bsmt_Exposure_Av + 1e+06 t364_leaf2
   <= 1.000000999999e+06
 T364_L_f68_k3: const_Street_Grvl + 1e+06 t364_leaf3 <= 1.000000999999e+06
 T364_R_f232_k3: central_air_yes - 1e+06 t364_leaf3 >= -999999
 T364_R_f206_k3: const_Bsmt_Exposure_Av - 1e+06 t364_leaf3 >= -999999
 T364_R_f68_k4: const_Street_Grvl - 1e+06 t364_leaf4 >= -999999
 T365_onehot: t365_leaf0 + t365_leaf1 + t365_leaf2 + t365_leaf3
   + t365_leaf4 = 1
 T365_L_f239_k0: const_FunctioNo_aplical_Maj2 + 1e+06 t365_leaf0
   <= 1.000000999999e+06
 T365_L_f20_k0: x_Gr_Liv_Area + 1e+06 t365_leaf0 <= 1.000815999999e+06
 T365_L_f257_k0: x_paved_drive_is_N + 1e+06 t365_leaf0
   <= 1.000000999999e+06
 T365_L_f239_k1: const_FunctioNo_aplical_Maj2 + 1e+06 t365_leaf1
   <= 1.000000999999e+06
 T365_L_f20_k1: x_Gr_Liv_Area + 1e+06 t365_leaf1 <= 1.000815999999e+06
 T365_R_f257_k1: x_paved_drive_is_N - 1e+06 t365_leaf1 >= -999999
 T365_L_f239_k2: const_FunctioNo_aplical_Maj2 + 1e+06 t365_leaf2
   <= 1.000000999999e+06
 T365_R_f20_k2: x_Gr_Liv_Area - 1e+06 t365_leaf2 >= -999184
 T365_L_f46_k2: const_MS_SubClass_30 + 1e+06 t365_leaf2
   <= 1.000000999999e+06
 T365_L_f239_k3: const_FunctioNo_aplical_Maj2 + 1e+06 t365_leaf3
   <= 1.000000999999e+06
 T365_R_f20_k3: x_Gr_Liv_Area - 1e+06 t365_leaf3 >= -999184
 T365_R_f46_k3: const_MS_SubClass_30 - 1e+06 t365_leaf3 >= -999999
 T365_R_f239_k4: const_FunctioNo_aplical_Maj2 - 1e+06 t365_leaf4
   >= -999999
 T366_onehot: t366_leaf0 + t366_leaf1 + t366_leaf2 + t366_leaf3
   + t366_leaf4 = 1
 T366_L_f30_k0: x_Fireplace_Qu + 1e+06 t366_leaf0 <= 1.000001999999e+06
 T366_L_f1_k0: const_Lot_Area + 1e+06 t366_leaf0 <= 1.004499999999e+06
 T366_L_f30_k1: x_Fireplace_Qu + 1e+06 t366_leaf1 <= 1.000001999999e+06
 T366_R_f1_k1: const_Lot_Area - 1e+06 t366_leaf1 >= -995500
 T366_R_f30_k2: x_Fireplace_Qu - 1e+06 t366_leaf2 >= -999998
 T366_L_f9_k2: x_Exter_Cond + 1e+06 t366_leaf2 <= 1.000001999999e+06
 T366_R_f30_k3: x_Fireplace_Qu - 1e+06 t366_leaf3 >= -999998
 T366_R_f9_k3: x_Exter_Cond - 1e+06 t366_leaf3 >= -999998
 T366_L_f78_k3: const_Land_Contour_HLS + 1e+06 t366_leaf3
   <= 1.000000999999e+06
 T366_R_f30_k4: x_Fireplace_Qu - 1e+06 t366_leaf4 >= -999998
 T366_R_f9_k4: x_Exter_Cond - 1e+06 t366_leaf4 >= -999998
 T366_R_f78_k4: const_Land_Contour_HLS - 1e+06 t366_leaf4 >= -999999
 T367_onehot: t367_leaf0 + t367_leaf1 + t367_leaf2 + t367_leaf3 = 1
 T367_L_f3_k0: Overall_Qual_calc + 1e+06 t367_leaf0 <= 1.000003999999e+06
 T367_R_f3_k1: Overall_Qual_calc - 1e+06 t367_leaf1 >= -999996
 T367_L_f68_k1: const_Street_Grvl + 1e+06 t367_leaf1 <= 1.000000999999e+06
 T367_L_f219_k1: x_b2_is_BLQ + 1e+06 t367_leaf1 <= 1.000000999999e+06
 T367_R_f3_k2: Overall_Qual_calc - 1e+06 t367_leaf2 >= -999996
 T367_L_f68_k2: const_Street_Grvl + 1e+06 t367_leaf2 <= 1.000000999999e+06
 T367_R_f219_k2: x_b2_is_BLQ - 1e+06 t367_leaf2 >= -999999
 T367_R_f3_k3: Overall_Qual_calc - 1e+06 t367_leaf3 >= -999996
 T367_R_f68_k3: const_Street_Grvl - 1e+06 t367_leaf3 >= -999999
 T368_onehot: t368_leaf0 + t368_leaf1 + t368_leaf2 + t368_leaf3
   + t368_leaf4 = 1
 T368_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t368_leaf0
   <= 1.000000999999e+06
 T368_R_f96_k1: const_Neighborhood_Edwards - 1e+06 t368_leaf1 >= -999999
 T368_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t368_leaf1
   <= 1.000760999999e+06
 T368_L_f33_k1: x_Garage_Area + 1e+06 t368_leaf1 <= 1.000224999999e+06
 T368_R_f96_k2: const_Neighborhood_Edwards - 1e+06 t368_leaf2 >= -999999
 T368_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t368_leaf2
   <= 1.000760999999e+06
 T368_R_f33_k2: x_Garage_Area - 1e+06 t368_leaf2 >= -999775
 T368_R_f96_k3: const_Neighborhood_Edwards - 1e+06 t368_leaf3 >= -999999
 T368_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t368_leaf3 >= -999239
 T368_L_f33_k3: x_Garage_Area + 1e+06 t368_leaf3 <= 1.000249999999e+06
 T368_R_f96_k4: const_Neighborhood_Edwards - 1e+06 t368_leaf4 >= -999999
 T368_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t368_leaf4 >= -999239
 T368_R_f33_k4: x_Garage_Area - 1e+06 t368_leaf4 >= -999750
 T369_onehot: t369_leaf0 + t369_leaf1 + t369_leaf2 + t369_leaf3
   + t369_leaf4 + t369_leaf5 = 1
 T369_L_f72_k0: const_Alley_Pave + 1e+06 t369_leaf0 <= 1.000000999999e+06
 T369_L_f211_k0: x_b1_is_ALQ + 1e+06 t369_leaf0 <= 1.000000999999e+06
 T369_L_f5_k0: const_Year_Built + 1e+06 t369_leaf0 <= 1.002004999999e+06
 T369_L_f72_k1: const_Alley_Pave + 1e+06 t369_leaf1 <= 1.000000999999e+06
 T369_L_f211_k1: x_b1_is_ALQ + 1e+06 t369_leaf1 <= 1.000000999999e+06
 T369_R_f5_k1: const_Year_Built - 1e+06 t369_leaf1 >= -997995
 T369_L_f72_k2: const_Alley_Pave + 1e+06 t369_leaf2 <= 1.000000999999e+06
 T369_R_f211_k2: x_b1_is_ALQ - 1e+06 t369_leaf2 >= -999999
 T369_L_f3_k2: Overall_Qual_calc + 1e+06 t369_leaf2 <= 1.000006999999e+06
 T369_L_f72_k3: const_Alley_Pave + 1e+06 t369_leaf3 <= 1.000000999999e+06
 T369_R_f211_k3: x_b1_is_ALQ - 1e+06 t369_leaf3 >= -999999
 T369_R_f3_k3: Overall_Qual_calc - 1e+06 t369_leaf3 >= -999993
 T369_R_f72_k4: const_Alley_Pave - 1e+06 t369_leaf4 >= -999999
 T369_L_f3_k4: Overall_Qual_calc + 1e+06 t369_leaf4 <= 1.000006999999e+06
 T369_R_f72_k5: const_Alley_Pave - 1e+06 t369_leaf5 >= -999999
 T369_R_f3_k5: Overall_Qual_calc - 1e+06 t369_leaf5 >= -999993
 T370_onehot: t370_leaf0 + t370_leaf1 + t370_leaf2 = 1
 T370_L_f75_k0: const_Lot_Shape_IR3 + 1e+06 t370_leaf0
   <= 1.000000999999e+06
 T370_L_f62_k0: const_MS_Zoning_C_(all) + 1e+06 t370_leaf0
   <= 1.000000999999e+06
 T370_L_f75_k1: const_Lot_Shape_IR3 + 1e+06 t370_leaf1
   <= 1.000000999999e+06
 T370_R_f62_k1: const_MS_Zoning_C_(all) - 1e+06 t370_leaf1 >= -999999
 T370_R_f75_k2: const_Lot_Shape_IR3 - 1e+06 t370_leaf2 >= -999999
 T371_onehot: t371_leaf0 + t371_leaf1 + t371_leaf2 + t371_leaf3
   + t371_leaf4 = 1
 T371_L_f20_k0: x_Gr_Liv_Area + 1e+06 t371_leaf0 <= 1.000935999999e+06
 T371_R_f20_k1: x_Gr_Liv_Area - 1e+06 t371_leaf1 >= -999064
 T371_L_f20_k1: x_Gr_Liv_Area + 1e+06 t371_leaf1 <= 1.001645999999e+06
 T371_L_f231_k1: central_air_no + 1e+06 t371_leaf1 <= 1.000000999999e+06
 T371_R_f20_k2: x_Gr_Liv_Area - 1e+06 t371_leaf2 >= -999064
 T371_L_f20_k2: x_Gr_Liv_Area + 1e+06 t371_leaf2 <= 1.001645999999e+06
 T371_R_f231_k2: central_air_no - 1e+06 t371_leaf2 >= -999999
 T371_R_f20_k3: x_Gr_Liv_Area - 1e+06 t371_leaf3 >= -999064
 T371_R_f20_k3: x_Gr_Liv_Area - 1e+06 t371_leaf3 >= -998354
 T371_L_f5_k3: const_Year_Built + 1e+06 t371_leaf3 <= 1.001910999999e+06
 T371_R_f20_k4: x_Gr_Liv_Area - 1e+06 t371_leaf4 >= -999064
 T371_R_f20_k4: x_Gr_Liv_Area - 1e+06 t371_leaf4 >= -998354
 T371_R_f5_k4: const_Year_Built - 1e+06 t371_leaf4 >= -998089
 T372_onehot: t372_leaf0 + t372_leaf1 + t372_leaf2 = 1
 T372_L_f40_k0: x_Screen_Porch + 1e+06 t372_leaf0 <= 1.000107999999e+06
 T372_L_f0_k0: const_Lot_Frontage + 1e+06 t372_leaf0 <= 1.000173999999e+06
 T372_L_f40_k1: x_Screen_Porch + 1e+06 t372_leaf1 <= 1.000107999999e+06
 T372_R_f0_k1: const_Lot_Frontage - 1e+06 t372_leaf1 >= -999826
 T372_R_f40_k2: x_Screen_Porch - 1e+06 t372_leaf2 >= -999892
 T373_onehot: t373_leaf0 + t373_leaf1 + t373_leaf2 + t373_leaf3
   + t373_leaf4 = 1
 T373_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t373_leaf0 <= 1.000000999999e+06
 T373_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t373_leaf0
   <= 1.000505999999e+06
 T373_L_f35_k0: x_Garage_Cond + 1e+06 t373_leaf0 <= 1.000000999999e+06
 T373_L_f177_k1: x_ex2_is_AsbShng + 1e+06 t373_leaf1 <= 1.000000999999e+06
 T373_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t373_leaf1
   <= 1.000505999999e+06
 T373_R_f35_k1: x_Garage_Cond - 1e+06 t373_leaf1 >= -999999
 T373_L_f177_k2: x_ex2_is_AsbShng + 1e+06 t373_leaf2 <= 1.000000999999e+06
 T373_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t373_leaf2 >= -999494
 T373_L_f86_k2: const_Land_Slope_Gtl + 1e+06 t373_leaf2
   <= 1.000000999999e+06
 T373_L_f177_k3: x_ex2_is_AsbShng + 1e+06 t373_leaf3 <= 1.000000999999e+06
 T373_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t373_leaf3 >= -999494
 T373_R_f86_k3: const_Land_Slope_Gtl - 1e+06 t373_leaf3 >= -999999
 T373_R_f177_k4: x_ex2_is_AsbShng - 1e+06 t373_leaf4 >= -999999
 T374_onehot: t374_leaf0 + t374_leaf1 + t374_leaf2 + t374_leaf3
   + t374_leaf4 + t374_leaf5 = 1
 T374_L_f219_k0: x_b2_is_BLQ + 1e+06 t374_leaf0 <= 1.000000999999e+06
 T374_L_f23_k0: x_Full_Bath + 1e+06 t374_leaf0 <= 1.000002999999e+06
 T374_L_f13_k0: x_BsmtFin_SF_2 + 1e+06 t374_leaf0 <= 1.000711999999e+06
 T374_L_f219_k1: x_b2_is_BLQ + 1e+06 t374_leaf1 <= 1.000000999999e+06
 T374_L_f23_k1: x_Full_Bath + 1e+06 t374_leaf1 <= 1.000002999999e+06
 T374_R_f13_k1: x_BsmtFin_SF_2 - 1e+06 t374_leaf1 >= -999288
 T374_L_f219_k2: x_b2_is_BLQ + 1e+06 t374_leaf2 <= 1.000000999999e+06
 T374_R_f23_k2: x_Full_Bath - 1e+06 t374_leaf2 >= -999997
 T374_R_f219_k3: x_b2_is_BLQ - 1e+06 t374_leaf3 >= -999999
 T374_L_f33_k3: x_Garage_Area + 1e+06 t374_leaf3 <= 1.000337999999e+06
 T374_R_f219_k4: x_b2_is_BLQ - 1e+06 t374_leaf4 >= -999999
 T374_R_f33_k4: x_Garage_Area - 1e+06 t374_leaf4 >= -999662
 T374_L_f251_k4: const_Garage_Type_Detchd + 1e+06 t374_leaf4
   <= 1.000000999999e+06
 T374_R_f219_k5: x_b2_is_BLQ - 1e+06 t374_leaf5 >= -999999
 T374_R_f33_k5: x_Garage_Area - 1e+06 t374_leaf5 >= -999662
 T374_R_f251_k5: const_Garage_Type_Detchd - 1e+06 t374_leaf5 >= -999999
 T375_onehot: t375_leaf0 + t375_leaf1 + t375_leaf2 + t375_leaf3 = 1
 T375_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t375_leaf0 <= 1.000000999999e+06
 T375_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t375_leaf0 <= 1.000000999999e+06
 T375_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t375_leaf1 <= 1.000000999999e+06
 T375_R_f161_k1: x_ex1_is_AsbShng - 1e+06 t375_leaf1 >= -999999
 T375_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t375_leaf1
   <= 1.001920999999e+06
 T375_L_f164_k2: x_ex1_is_BrkFace + 1e+06 t375_leaf2 <= 1.000000999999e+06
 T375_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t375_leaf2 >= -999999
 T375_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t375_leaf2 >= -998079
 T375_R_f164_k3: x_ex1_is_BrkFace - 1e+06 t375_leaf3 >= -999999
 T376_onehot: t376_leaf0 + t376_leaf1 + t376_leaf2 + t376_leaf3 = 1
 T376_L_f5_k0: const_Year_Built + 1e+06 t376_leaf0 <= 1.002004999999e+06
 T376_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t376_leaf0
   <= 1.000000999999e+06
 T376_L_f5_k1: const_Year_Built + 1e+06 t376_leaf1 <= 1.002004999999e+06
 T376_R_f96_k1: const_Neighborhood_Edwards - 1e+06 t376_leaf1 >= -999999
 T376_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t376_leaf1
   <= 1.001955999999e+06
 T376_L_f5_k2: const_Year_Built + 1e+06 t376_leaf2 <= 1.002004999999e+06
 T376_R_f96_k2: const_Neighborhood_Edwards - 1e+06 t376_leaf2 >= -999999
 T376_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t376_leaf2 >= -998044
 T376_R_f5_k3: const_Year_Built - 1e+06 t376_leaf3 >= -997995
 T377_onehot: t377_leaf0 = 1
 T378_onehot: t378_leaf0 + t378_leaf1 + t378_leaf2 = 1
 T378_L_f3_k0: Overall_Qual_calc + 1e+06 t378_leaf0 <= 1.000002999999e+06
 T378_L_f33_k0: x_Garage_Area + 1e+06 t378_leaf0 <= 1.000272999999e+06
 T378_L_f3_k1: Overall_Qual_calc + 1e+06 t378_leaf1 <= 1.000002999999e+06
 T378_R_f33_k1: x_Garage_Area - 1e+06 t378_leaf1 >= -999727
 T378_R_f3_k2: Overall_Qual_calc - 1e+06 t378_leaf2 >= -999997
 T379_onehot: t379_leaf0 + t379_leaf1 + t379_leaf2 + t379_leaf3
   + t379_leaf4 = 1
 T379_L_f32_k0: const_Garage_Cars + 1e+06 t379_leaf0 <= 1.000002999999e+06
 T379_L_f17_k0: x_1st_Flr_SF + 1e+06 t379_leaf0 <= 1.000575999999e+06
 T379_L_f32_k1: const_Garage_Cars + 1e+06 t379_leaf1 <= 1.000002999999e+06
 T379_R_f17_k1: x_1st_Flr_SF - 1e+06 t379_leaf1 >= -999424
 T379_R_f32_k2: const_Garage_Cars - 1e+06 t379_leaf2 >= -999997
 T379_L_f30_k2: x_Fireplace_Qu + 1e+06 t379_leaf2 <= 999999.999999
 T379_R_f32_k3: const_Garage_Cars - 1e+06 t379_leaf3 >= -999997
 T379_R_f30_k3: x_Fireplace_Qu - 1e+06 t379_leaf3 >= -1e+06
 T379_L_f40_k3: x_Screen_Porch + 1e+06 t379_leaf3 <= 1.000039999999e+06
 T379_R_f32_k4: const_Garage_Cars - 1e+06 t379_leaf4 >= -999997
 T379_R_f30_k4: x_Fireplace_Qu - 1e+06 t379_leaf4 >= -1e+06
 T379_R_f40_k4: x_Screen_Porch - 1e+06 t379_leaf4 >= -999960
 T380_onehot: t380_leaf0 + t380_leaf1 = 1
 T380_L_f40_k0: x_Screen_Porch + 1e+06 t380_leaf0 <= 1.000155999999e+06
 T380_R_f40_k1: x_Screen_Porch - 1e+06 t380_leaf1 >= -999844
 T381_onehot: t381_leaf0 + t381_leaf1 + t381_leaf2 + t381_leaf3
   + t381_leaf4 = 1
 T381_L_f3_k0: Overall_Qual_calc + 1e+06 t381_leaf0 <= 1.000003999999e+06
 T381_L_f169_k0: x_ex1_is_MetalSd + 1e+06 t381_leaf0 <= 1.000000999999e+06
 T381_L_f3_k1: Overall_Qual_calc + 1e+06 t381_leaf1 <= 1.000003999999e+06
 T381_R_f169_k1: x_ex1_is_MetalSd - 1e+06 t381_leaf1 >= -999999
 T381_R_f3_k2: Overall_Qual_calc - 1e+06 t381_leaf2 >= -999996
 T381_L_f5_k2: const_Year_Built + 1e+06 t381_leaf2 <= 1.001910999999e+06
 T381_L_f4_k2: const_Overall_Cond + 1e+06 t381_leaf2 <= 1.000006999999e+06
 T381_R_f3_k3: Overall_Qual_calc - 1e+06 t381_leaf3 >= -999996
 T381_L_f5_k3: const_Year_Built + 1e+06 t381_leaf3 <= 1.001910999999e+06
 T381_R_f4_k3: const_Overall_Cond - 1e+06 t381_leaf3 >= -999993
 T381_R_f3_k4: Overall_Qual_calc - 1e+06 t381_leaf4 >= -999996
 T381_R_f5_k4: const_Year_Built - 1e+06 t381_leaf4 >= -998089
 T382_onehot: t382_leaf0 + t382_leaf1 + t382_leaf2 + t382_leaf3
   + t382_leaf4 + t382_leaf5 = 1
 T382_L_f27_k0: x_Kitchen_Qual + 1e+06 t382_leaf0 <= 1.000001999999e+06
 T382_L_f32_k0: const_Garage_Cars + 1e+06 t382_leaf0 <= 1.000001999999e+06
 T382_L_f27_k1: x_Kitchen_Qual + 1e+06 t382_leaf1 <= 1.000001999999e+06
 T382_R_f32_k1: const_Garage_Cars - 1e+06 t382_leaf1 >= -999998
 T382_R_f27_k2: x_Kitchen_Qual - 1e+06 t382_leaf2 >= -999998
 T382_L_f6_k2: const_Year_Remod_Add + 1e+06 t382_leaf2
   <= 1.001964999999e+06
 T382_L_f17_k2: x_1st_Flr_SF + 1e+06 t382_leaf2 <= 1.000915999999e+06
 T382_R_f27_k3: x_Kitchen_Qual - 1e+06 t382_leaf3 >= -999998
 T382_L_f6_k3: const_Year_Remod_Add + 1e+06 t382_leaf3
   <= 1.001964999999e+06
 T382_R_f17_k3: x_1st_Flr_SF - 1e+06 t382_leaf3 >= -999084
 T382_R_f27_k4: x_Kitchen_Qual - 1e+06 t382_leaf4 >= -999998
 T382_R_f6_k4: const_Year_Remod_Add - 1e+06 t382_leaf4 >= -998035
 T382_L_f46_k4: const_MS_SubClass_30 + 1e+06 t382_leaf4
   <= 1.000000999999e+06
 T382_R_f27_k5: x_Kitchen_Qual - 1e+06 t382_leaf5 >= -999998
 T382_R_f6_k5: const_Year_Remod_Add - 1e+06 t382_leaf5 >= -998035
 T382_R_f46_k5: const_MS_SubClass_30 - 1e+06 t382_leaf5 >= -999999
 T383_onehot: t383_leaf0 + t383_leaf1 + t383_leaf2 + t383_leaf3
   + t383_leaf4 + t383_leaf5 + t383_leaf6 = 1
 T383_L_f20_k0: x_Gr_Liv_Area + 1e+06 t383_leaf0 <= 1.000893999999e+06
 T383_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t383_leaf0
   <= 1.000000999999e+06
 T383_L_f76_k0: const_Lot_Shape_Reg + 1e+06 t383_leaf0
   <= 1.000000999999e+06
 T383_L_f20_k1: x_Gr_Liv_Area + 1e+06 t383_leaf1 <= 1.000893999999e+06
 T383_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t383_leaf1
   <= 1.000000999999e+06
 T383_R_f76_k1: const_Lot_Shape_Reg - 1e+06 t383_leaf1 >= -999999
 T383_L_f20_k2: x_Gr_Liv_Area + 1e+06 t383_leaf2 <= 1.000893999999e+06
 T383_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t383_leaf2
   >= -999999
 T383_L_f33_k2: x_Garage_Area + 1e+06 t383_leaf2 <= 1.000370999999e+06
 T383_L_f20_k3: x_Gr_Liv_Area + 1e+06 t383_leaf3 <= 1.000893999999e+06
 T383_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t383_leaf3
   >= -999999
 T383_R_f33_k3: x_Garage_Area - 1e+06 t383_leaf3 >= -999629
 T383_R_f20_k4: x_Gr_Liv_Area - 1e+06 t383_leaf4 >= -999106
 T383_L_f28_k4: x_TotRms_AbvGrd + 1e+06 t383_leaf4 <= 1.000011999999e+06
 T383_L_f112_k4: const_Neighborhood_SawyerW + 1e+06 t383_leaf4
   <= 1.000000999999e+06
 T383_R_f20_k5: x_Gr_Liv_Area - 1e+06 t383_leaf5 >= -999106
 T383_L_f28_k5: x_TotRms_AbvGrd + 1e+06 t383_leaf5 <= 1.000011999999e+06
 T383_R_f112_k5: const_Neighborhood_SawyerW - 1e+06 t383_leaf5 >= -999999
 T383_R_f20_k6: x_Gr_Liv_Area - 1e+06 t383_leaf6 >= -999106
 T383_R_f28_k6: x_TotRms_AbvGrd - 1e+06 t383_leaf6 >= -999988
 T384_onehot: t384_leaf0 + t384_leaf1 + t384_leaf2 + t384_leaf3
   + t384_leaf4 = 1
 T384_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t384_leaf0
   <= 1.000000999999e+06
 T384_R_f245_k1: const_FunctioNo_aplical_Typ - 1e+06 t384_leaf1 >= -999999
 T384_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t384_leaf1 <= 1.000121999999e+06
 T384_L_f139_k1: const_House_Style_1.5Fin + 1e+06 t384_leaf1
   <= 1.000000999999e+06
 T384_R_f245_k2: const_FunctioNo_aplical_Typ - 1e+06 t384_leaf2 >= -999999
 T384_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t384_leaf2 <= 1.000121999999e+06
 T384_R_f139_k2: const_House_Style_1.5Fin - 1e+06 t384_leaf2 >= -999999
 T384_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t384_leaf3 >= -999999
 T384_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t384_leaf3 >= -999878
 T384_L_f33_k3: x_Garage_Area + 1e+06 t384_leaf3 <= 1.000870999999e+06
 T384_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t384_leaf4 >= -999999
 T384_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t384_leaf4 >= -999878
 T384_R_f33_k4: x_Garage_Area - 1e+06 t384_leaf4 >= -999129
 T385_onehot: t385_leaf0 + t385_leaf1 + t385_leaf2 + t385_leaf3
   + t385_leaf4 = 1
 T385_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t385_leaf0 <= 1.000000999999e+06
 T385_L_f62_k0: const_MS_Zoning_C_(all) + 1e+06 t385_leaf0
   <= 1.000000999999e+06
 T385_L_f3_k0: Overall_Qual_calc + 1e+06 t385_leaf0 <= 1.000006999999e+06
 T385_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t385_leaf1 <= 1.000000999999e+06
 T385_L_f62_k1: const_MS_Zoning_C_(all) + 1e+06 t385_leaf1
   <= 1.000000999999e+06
 T385_R_f3_k1: Overall_Qual_calc - 1e+06 t385_leaf1 >= -999993
 T385_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t385_leaf2 <= 1.000000999999e+06
 T385_R_f62_k2: const_MS_Zoning_C_(all) - 1e+06 t385_leaf2 >= -999999
 T385_R_f161_k3: x_ex1_is_AsbShng - 1e+06 t385_leaf3 >= -999999
 T385_L_f20_k3: x_Gr_Liv_Area + 1e+06 t385_leaf3 <= 1.000882999999e+06
 T385_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t385_leaf4 >= -999999
 T385_R_f20_k4: x_Gr_Liv_Area - 1e+06 t385_leaf4 >= -999117
 T386_onehot: t386_leaf0 + t386_leaf1 + t386_leaf2 + t386_leaf3 = 1
 T386_L_f33_k0: x_Garage_Area + 1e+06 t386_leaf0 <= 1.001153999999e+06
 T386_L_f5_k0: const_Year_Built + 1e+06 t386_leaf0 <= 1.002005999999e+06
 T386_L_f3_k0: Overall_Qual_calc + 1e+06 t386_leaf0 <= 1.000002999999e+06
 T386_L_f33_k1: x_Garage_Area + 1e+06 t386_leaf1 <= 1.001153999999e+06
 T386_L_f5_k1: const_Year_Built + 1e+06 t386_leaf1 <= 1.002005999999e+06
 T386_R_f3_k1: Overall_Qual_calc - 1e+06 t386_leaf1 >= -999997
 T386_L_f33_k2: x_Garage_Area + 1e+06 t386_leaf2 <= 1.001153999999e+06
 T386_R_f5_k2: const_Year_Built - 1e+06 t386_leaf2 >= -997994
 T386_R_f33_k3: x_Garage_Area - 1e+06 t386_leaf3 >= -998846
 T387_onehot: t387_leaf0 + t387_leaf1 + t387_leaf2 + t387_leaf3
   + t387_leaf4 = 1
 T387_L_f0_k0: const_Lot_Frontage + 1e+06 t387_leaf0 <= 1.000120999999e+06
 T387_L_f230_k0: x_Heating_Wall + 1e+06 t387_leaf0 <= 1.000000999999e+06
 T387_L_f231_k0: central_air_no + 1e+06 t387_leaf0 <= 1.000000999999e+06
 T387_L_f0_k1: const_Lot_Frontage + 1e+06 t387_leaf1 <= 1.000120999999e+06
 T387_L_f230_k1: x_Heating_Wall + 1e+06 t387_leaf1 <= 1.000000999999e+06
 T387_R_f231_k1: central_air_no - 1e+06 t387_leaf1 >= -999999
 T387_L_f0_k2: const_Lot_Frontage + 1e+06 t387_leaf2 <= 1.000120999999e+06
 T387_R_f230_k2: x_Heating_Wall - 1e+06 t387_leaf2 >= -999999
 T387_R_f0_k3: const_Lot_Frontage - 1e+06 t387_leaf3 >= -999879
 T387_L_f292_k3: const_Sale_Type_WD_ + 1e+06 t387_leaf3
   <= 1.000000999999e+06
 T387_R_f0_k4: const_Lot_Frontage - 1e+06 t387_leaf4 >= -999879
 T387_R_f292_k4: const_Sale_Type_WD_ - 1e+06 t387_leaf4 >= -999999
 T388_onehot: t388_leaf0 + t388_leaf1 + t388_leaf2 + t388_leaf3 = 1
 T388_L_f26_k0: x_Kitchen_AbvGr + 1e+06 t388_leaf0 <= 1.000001999999e+06
 T388_L_f4_k0: const_Overall_Cond + 1e+06 t388_leaf0 <= 1.000002999999e+06
 T388_L_f26_k1: x_Kitchen_AbvGr + 1e+06 t388_leaf1 <= 1.000001999999e+06
 T388_R_f4_k1: const_Overall_Cond - 1e+06 t388_leaf1 >= -999997
 T388_L_f232_k1: central_air_yes + 1e+06 t388_leaf1 <= 1.000000999999e+06
 T388_L_f26_k2: x_Kitchen_AbvGr + 1e+06 t388_leaf2 <= 1.000001999999e+06
 T388_R_f4_k2: const_Overall_Cond - 1e+06 t388_leaf2 >= -999997
 T388_R_f232_k2: central_air_yes - 1e+06 t388_leaf2 >= -999999
 T388_R_f26_k3: x_Kitchen_AbvGr - 1e+06 t388_leaf3 >= -999998
 T389_onehot: t389_leaf0 + t389_leaf1 + t389_leaf2 + t389_leaf3 = 1
 T389_L_f92_k0: const_Neighborhood_BrkSide + 1e+06 t389_leaf0
   <= 1.000000999999e+06
 T389_L_f44_k0: const_Yr_Sold + 1e+06 t389_leaf0 <= 1.002006999999e+06
 T389_L_f231_k0: central_air_no + 1e+06 t389_leaf0 <= 1.000000999999e+06
 T389_L_f92_k1: const_Neighborhood_BrkSide + 1e+06 t389_leaf1
   <= 1.000000999999e+06
 T389_L_f44_k1: const_Yr_Sold + 1e+06 t389_leaf1 <= 1.002006999999e+06
 T389_R_f231_k1: central_air_no - 1e+06 t389_leaf1 >= -999999
 T389_L_f92_k2: const_Neighborhood_BrkSide + 1e+06 t389_leaf2
   <= 1.000000999999e+06
 T389_R_f44_k2: const_Yr_Sold - 1e+06 t389_leaf2 >= -997993
 T389_R_f92_k3: const_Neighborhood_BrkSide - 1e+06 t389_leaf3 >= -999999
 T390_onehot: t390_leaf0 + t390_leaf1 + t390_leaf2 = 1
 T390_L_f1_k0: const_Lot_Area + 1e+06 t390_leaf0 <= 1.005975999999e+06
 T390_R_f1_k1: const_Lot_Area - 1e+06 t390_leaf1 >= -994024
 T390_L_f114_k1: const_Neighborhood_StoneBr + 1e+06 t390_leaf1
   <= 1.000000999999e+06
 T390_R_f1_k2: const_Lot_Area - 1e+06 t390_leaf2 >= -994024
 T390_R_f114_k2: const_Neighborhood_StoneBr - 1e+06 t390_leaf2 >= -999999
 T391_onehot: t391_leaf0 + t391_leaf1 + t391_leaf2 + t391_leaf3 = 1
 T391_L_f3_k0: Overall_Qual_calc + 1e+06 t391_leaf0 <= 1.000002999999e+06
 T391_R_f3_k1: Overall_Qual_calc - 1e+06 t391_leaf1 >= -999997
 T391_L_f250_k1: const_Garage_Type_CarPort + 1e+06 t391_leaf1
   <= 1.000000999999e+06
 T391_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t391_leaf1 <= 1.001731999999e+06
 T391_R_f3_k2: Overall_Qual_calc - 1e+06 t391_leaf2 >= -999997
 T391_L_f250_k2: const_Garage_Type_CarPort + 1e+06 t391_leaf2
   <= 1.000000999999e+06
 T391_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t391_leaf2 >= -998268
 T391_R_f3_k3: Overall_Qual_calc - 1e+06 t391_leaf3 >= -999997
 T391_R_f250_k3: const_Garage_Type_CarPort - 1e+06 t391_leaf3 >= -999999
 T392_onehot: t392_leaf0 + t392_leaf1 + t392_leaf2 + t392_leaf3 = 1
 T392_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t392_leaf0
   <= 1.000000999999e+06
 T392_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t392_leaf0 <= 1.000455999999e+06
 T392_L_f96_k1: const_Neighborhood_Edwards + 1e+06 t392_leaf1
   <= 1.000000999999e+06
 T392_R_f7_k1: x_Mas_Vnr_Area - 1e+06 t392_leaf1 >= -999544
 T392_R_f96_k2: const_Neighborhood_Edwards - 1e+06 t392_leaf2 >= -999999
 T392_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t392_leaf2
   <= 1.000697999999e+06
 T392_R_f96_k3: const_Neighborhood_Edwards - 1e+06 t392_leaf3 >= -999999
 T392_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t392_leaf3 >= -999302
 T393_onehot: t393_leaf0 + t393_leaf1 + t393_leaf2 + t393_leaf3
   + t393_leaf4 = 1
 T393_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t393_leaf0 <= 1.001731999999e+06
 T393_L_f8_k0: x_Exter_Qual + 1e+06 t393_leaf0 <= 1.000002999999e+06
 T393_L_f20_k0: x_Gr_Liv_Area + 1e+06 t393_leaf0 <= 1.001715999999e+06
 T393_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t393_leaf1 <= 1.001731999999e+06
 T393_L_f8_k1: x_Exter_Qual + 1e+06 t393_leaf1 <= 1.000002999999e+06
 T393_R_f20_k1: x_Gr_Liv_Area - 1e+06 t393_leaf1 >= -998284
 T393_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t393_leaf2 <= 1.001731999999e+06
 T393_R_f8_k2: x_Exter_Qual - 1e+06 t393_leaf2 >= -999997
 T393_L_f17_k2: x_1st_Flr_SF + 1e+06 t393_leaf2 <= 1.001781999999e+06
 T393_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t393_leaf3 <= 1.001731999999e+06
 T393_R_f8_k3: x_Exter_Qual - 1e+06 t393_leaf3 >= -999997
 T393_R_f17_k3: x_1st_Flr_SF - 1e+06 t393_leaf3 >= -998218
 T393_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t393_leaf4 >= -998268
 T394_onehot: t394_leaf0 + t394_leaf1 = 1
 T394_L_f216_k0: x_b1_is_Rec + 1e+06 t394_leaf0 <= 1.000000999999e+06
 T394_R_f216_k1: x_b1_is_Rec - 1e+06 t394_leaf1 >= -999999
 T395_onehot: t395_leaf0 + t395_leaf1 + t395_leaf2 + t395_leaf3
   + t395_leaf4 + t395_leaf5 = 1
 T395_L_f37_k0: x_Open_Porch_SF + 1e+06 t395_leaf0 <= 1.000290999999e+06
 T395_L_f6_k0: const_Year_Remod_Add + 1e+06 t395_leaf0
   <= 1.002006999999e+06
 T395_L_f4_k0: const_Overall_Cond + 1e+06 t395_leaf0 <= 1.000006999999e+06
 T395_L_f37_k1: x_Open_Porch_SF + 1e+06 t395_leaf1 <= 1.000290999999e+06
 T395_L_f6_k1: const_Year_Remod_Add + 1e+06 t395_leaf1
   <= 1.002006999999e+06
 T395_R_f4_k1: const_Overall_Cond - 1e+06 t395_leaf1 >= -999993
 T395_L_f37_k2: x_Open_Porch_SF + 1e+06 t395_leaf2 <= 1.000290999999e+06
 T395_R_f6_k2: const_Year_Remod_Add - 1e+06 t395_leaf2 >= -997993
 T395_L_f89_k2: const_Neighborhood_Blmngtn + 1e+06 t395_leaf2
   <= 1.000000999999e+06
 T395_L_f37_k3: x_Open_Porch_SF + 1e+06 t395_leaf3 <= 1.000290999999e+06
 T395_R_f6_k3: const_Year_Remod_Add - 1e+06 t395_leaf3 >= -997993
 T395_R_f89_k3: const_Neighborhood_Blmngtn - 1e+06 t395_leaf3 >= -999999
 T395_R_f37_k4: x_Open_Porch_SF - 1e+06 t395_leaf4 >= -999709
 T395_L_f297_k4: const_Sale_Condition_Normal + 1e+06 t395_leaf4
   <= 1.000000999999e+06
 T395_R_f37_k5: x_Open_Porch_SF - 1e+06 t395_leaf5 >= -999709
 T395_R_f297_k5: const_Sale_Condition_Normal - 1e+06 t395_leaf5 >= -999999
 T396_onehot: t396_leaf0 + t396_leaf1 + t396_leaf2 + t396_leaf3
   + t396_leaf4 = 1
 T396_L_f3_k0: Overall_Qual_calc + 1e+06 t396_leaf0 <= 1.000002999999e+06
 T396_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t396_leaf0
   <= 1.001894999999e+06
 T396_L_f3_k1: Overall_Qual_calc + 1e+06 t396_leaf1 <= 1.000002999999e+06
 T396_R_f31_k1: const_Garage_Yr_Blt - 1e+06 t396_leaf1 >= -998105
 T396_R_f3_k2: Overall_Qual_calc - 1e+06 t396_leaf2 >= -999997
 T396_L_f37_k2: x_Open_Porch_SF + 1e+06 t396_leaf2 <= 1.000483999999e+06
 T396_L_f35_k2: x_Garage_Cond + 1e+06 t396_leaf2 <= 1.000001999999e+06
 T396_R_f3_k3: Overall_Qual_calc - 1e+06 t396_leaf3 >= -999997
 T396_L_f37_k3: x_Open_Porch_SF + 1e+06 t396_leaf3 <= 1.000483999999e+06
 T396_R_f35_k3: x_Garage_Cond - 1e+06 t396_leaf3 >= -999998
 T396_R_f3_k4: Overall_Qual_calc - 1e+06 t396_leaf4 >= -999997
 T396_R_f37_k4: x_Open_Porch_SF - 1e+06 t396_leaf4 >= -999516
 T397_onehot: t397_leaf0 + t397_leaf1 + t397_leaf2 + t397_leaf3
   + t397_leaf4 = 1
 T397_L_f296_k0: const_Sale_Condition_Family + 1e+06 t397_leaf0
   <= 1.000000999999e+06
 T397_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t397_leaf0 <= 1.000000999999e+06
 T397_L_f296_k1: const_Sale_Condition_Family + 1e+06 t397_leaf1
   <= 1.000000999999e+06
 T397_R_f161_k1: x_ex1_is_AsbShng - 1e+06 t397_leaf1 >= -999999
 T397_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t397_leaf1
   <= 1.001922999999e+06
 T397_L_f296_k2: const_Sale_Condition_Family + 1e+06 t397_leaf2
   <= 1.000000999999e+06
 T397_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t397_leaf2 >= -999999
 T397_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t397_leaf2 >= -998077
 T397_R_f296_k3: const_Sale_Condition_Family - 1e+06 t397_leaf3 >= -999999
 T397_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t397_leaf3 <= 1.000166999999e+06
 T397_R_f296_k4: const_Sale_Condition_Family - 1e+06 t397_leaf4 >= -999999
 T397_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t397_leaf4 >= -999833
 T398_onehot: t398_leaf0 + t398_leaf1 + t398_leaf2 + t398_leaf3 = 1
 T398_L_f37_k0: x_Open_Porch_SF + 1e+06 t398_leaf0 <= 1.000158999999e+06
 T398_L_f11_k0: x_Bsmt_Cond + 1e+06 t398_leaf0 <= 1.000002999999e+06
 T398_L_f37_k1: x_Open_Porch_SF + 1e+06 t398_leaf1 <= 1.000158999999e+06
 T398_R_f11_k1: x_Bsmt_Cond - 1e+06 t398_leaf1 >= -999997
 T398_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t398_leaf1
   <= 1.001976999999e+06
 T398_L_f37_k2: x_Open_Porch_SF + 1e+06 t398_leaf2 <= 1.000158999999e+06
 T398_R_f11_k2: x_Bsmt_Cond - 1e+06 t398_leaf2 >= -999997
 T398_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t398_leaf2 >= -998023
 T398_R_f37_k3: x_Open_Porch_SF - 1e+06 t398_leaf3 >= -999841
 T399_onehot: t399_leaf0 + t399_leaf1 + t399_leaf2 + t399_leaf3 = 1
 T399_L_f86_k0: const_Land_Slope_Gtl + 1e+06 t399_leaf0
   <= 1.000000999999e+06
 T399_R_f86_k1: const_Land_Slope_Gtl - 1e+06 t399_leaf1 >= -999999
 T399_L_f1_k1: const_Lot_Area + 1e+06 t399_leaf1 <= 1.025285999999e+06
 T399_L_f17_k1: x_1st_Flr_SF + 1e+06 t399_leaf1 <= 1.000575999999e+06
 T399_R_f86_k2: const_Land_Slope_Gtl - 1e+06 t399_leaf2 >= -999999
 T399_L_f1_k2: const_Lot_Area + 1e+06 t399_leaf2 <= 1.025285999999e+06
 T399_R_f17_k2: x_1st_Flr_SF - 1e+06 t399_leaf2 >= -999424
 T399_R_f86_k3: const_Land_Slope_Gtl - 1e+06 t399_leaf3 >= -999999
 T399_R_f1_k3: const_Lot_Area - 1e+06 t399_leaf3 >= -974714
 T400_onehot: t400_leaf0 + t400_leaf1 + t400_leaf2 = 1
 T400_L_f250_k0: const_Garage_Type_CarPort + 1e+06 t400_leaf0
   <= 1.000000999999e+06
 T400_L_f4_k0: const_Overall_Cond + 1e+06 t400_leaf0 <= 1.000006999999e+06
 T400_L_f250_k1: const_Garage_Type_CarPort + 1e+06 t400_leaf1
   <= 1.000000999999e+06
 T400_R_f4_k1: const_Overall_Cond - 1e+06 t400_leaf1 >= -999993
 T400_R_f250_k2: const_Garage_Type_CarPort - 1e+06 t400_leaf2 >= -999999
 T401_onehot: t401_leaf0 = 1
 T402_onehot: t402_leaf0 + t402_leaf1 + t402_leaf2 + t402_leaf3
   + t402_leaf4 = 1
 T402_L_f112_k0: const_Neighborhood_SawyerW + 1e+06 t402_leaf0
   <= 1.000000999999e+06
 T402_L_f226_k0: x_Heating_GasA + 1e+06 t402_leaf0 <= 1.000000999999e+06
 T402_L_f17_k0: x_1st_Flr_SF + 1e+06 t402_leaf0 <= 1.000931999999e+06
 T402_L_f112_k1: const_Neighborhood_SawyerW + 1e+06 t402_leaf1
   <= 1.000000999999e+06
 T402_L_f226_k1: x_Heating_GasA + 1e+06 t402_leaf1 <= 1.000000999999e+06
 T402_R_f17_k1: x_1st_Flr_SF - 1e+06 t402_leaf1 >= -999068
 T402_L_f112_k2: const_Neighborhood_SawyerW + 1e+06 t402_leaf2
   <= 1.000000999999e+06
 T402_R_f226_k2: x_Heating_GasA - 1e+06 t402_leaf2 >= -999999
 T402_L_f3_k2: Overall_Qual_calc + 1e+06 t402_leaf2 <= 1.000003999999e+06
 T402_L_f112_k3: const_Neighborhood_SawyerW + 1e+06 t402_leaf3
   <= 1.000000999999e+06
 T402_R_f226_k3: x_Heating_GasA - 1e+06 t402_leaf3 >= -999999
 T402_R_f3_k3: Overall_Qual_calc - 1e+06 t402_leaf3 >= -999996
 T402_R_f112_k4: const_Neighborhood_SawyerW - 1e+06 t402_leaf4 >= -999999
 T403_onehot: t403_leaf0 + t403_leaf1 + t403_leaf2 + t403_leaf3
   + t403_leaf4 = 1
 T403_L_f68_k0: const_Street_Grvl + 1e+06 t403_leaf0 <= 1.000000999999e+06
 T403_L_f33_k0: x_Garage_Area + 1e+06 t403_leaf0 <= 1.000629999999e+06
 T403_L_f18_k0: const_2nd_Flr_SF + 1e+06 t403_leaf0 <= 1.000939999999e+06
 T403_L_f68_k1: const_Street_Grvl + 1e+06 t403_leaf1 <= 1.000000999999e+06
 T403_L_f33_k1: x_Garage_Area + 1e+06 t403_leaf1 <= 1.000629999999e+06
 T403_R_f18_k1: const_2nd_Flr_SF - 1e+06 t403_leaf1 >= -999060
 T403_L_f68_k2: const_Street_Grvl + 1e+06 t403_leaf2 <= 1.000000999999e+06
 T403_R_f33_k2: x_Garage_Area - 1e+06 t403_leaf2 >= -999370
 T403_L_f4_k2: const_Overall_Cond + 1e+06 t403_leaf2 <= 1.000004999999e+06
 T403_L_f68_k3: const_Street_Grvl + 1e+06 t403_leaf3 <= 1.000000999999e+06
 T403_R_f33_k3: x_Garage_Area - 1e+06 t403_leaf3 >= -999370
 T403_R_f4_k3: const_Overall_Cond - 1e+06 t403_leaf3 >= -999995
 T403_R_f68_k4: const_Street_Grvl - 1e+06 t403_leaf4 >= -999999
 T404_onehot: t404_leaf0 + t404_leaf1 + t404_leaf2 + t404_leaf3
   + t404_leaf4 + t404_leaf5 = 1
 T404_L_f3_k0: Overall_Qual_calc + 1e+06 t404_leaf0 <= 1.000004999999e+06
 T404_L_f6_k0: const_Year_Remod_Add + 1e+06 t404_leaf0
   <= 1.001952999999e+06
 T404_L_f0_k0: const_Lot_Frontage + 1e+06 t404_leaf0 <= 1.000050999999e+06
 T404_L_f3_k1: Overall_Qual_calc + 1e+06 t404_leaf1 <= 1.000004999999e+06
 T404_L_f6_k1: const_Year_Remod_Add + 1e+06 t404_leaf1
   <= 1.001952999999e+06
 T404_R_f0_k1: const_Lot_Frontage - 1e+06 t404_leaf1 >= -999949
 T404_L_f3_k2: Overall_Qual_calc + 1e+06 t404_leaf2 <= 1.000004999999e+06
 T404_R_f6_k2: const_Year_Remod_Add - 1e+06 t404_leaf2 >= -998047
 T404_R_f3_k3: Overall_Qual_calc - 1e+06 t404_leaf3 >= -999995
 T404_L_f121_k3: const_Condition_1_PosN + 1e+06 t404_leaf3
   <= 1.000000999999e+06
 T404_L_f119_k3: const_Condition_1_Norm + 1e+06 t404_leaf3
   <= 1.000000999999e+06
 T404_R_f3_k4: Overall_Qual_calc - 1e+06 t404_leaf4 >= -999995
 T404_L_f121_k4: const_Condition_1_PosN + 1e+06 t404_leaf4
   <= 1.000000999999e+06
 T404_R_f119_k4: const_Condition_1_Norm - 1e+06 t404_leaf4 >= -999999
 T404_R_f3_k5: Overall_Qual_calc - 1e+06 t404_leaf5 >= -999995
 T404_R_f121_k5: const_Condition_1_PosN - 1e+06 t404_leaf5 >= -999999
 T405_onehot: t405_leaf0 + t405_leaf1 + t405_leaf2 + t405_leaf3
   + t405_leaf4 = 1
 T405_L_f20_k0: x_Gr_Liv_Area + 1e+06 t405_leaf0 <= 1.000746999999e+06
 T405_L_f16_k0: x_Heating_QC + 1e+06 t405_leaf0 <= 1.000002999999e+06
 T405_L_f20_k1: x_Gr_Liv_Area + 1e+06 t405_leaf1 <= 1.000746999999e+06
 T405_R_f16_k1: x_Heating_QC - 1e+06 t405_leaf1 >= -999997
 T405_R_f20_k2: x_Gr_Liv_Area - 1e+06 t405_leaf2 >= -999253
 T405_L_f3_k2: Overall_Qual_calc + 1e+06 t405_leaf2 <= 1.000004999999e+06
 T405_R_f20_k3: x_Gr_Liv_Area - 1e+06 t405_leaf3 >= -999253
 T405_R_f3_k3: Overall_Qual_calc - 1e+06 t405_leaf3 >= -999995
 T405_L_f37_k3: x_Open_Porch_SF + 1e+06 t405_leaf3 <= 1.000259999999e+06
 T405_R_f20_k4: x_Gr_Liv_Area - 1e+06 t405_leaf4 >= -999253
 T405_R_f3_k4: Overall_Qual_calc - 1e+06 t405_leaf4 >= -999995
 T405_R_f37_k4: x_Open_Porch_SF - 1e+06 t405_leaf4 >= -999740
 T406_onehot: t406_leaf0 = 1
 T407_onehot: t407_leaf0 = 1
 T408_onehot: t408_leaf0 + t408_leaf1 + t408_leaf2 + t408_leaf3
   + t408_leaf4 = 1
 T408_L_f18_k0: const_2nd_Flr_SF + 1e+06 t408_leaf0 <= 1.001610999999e+06
 T408_L_f20_k0: x_Gr_Liv_Area + 1e+06 t408_leaf0 <= 1.002871999999e+06
 T408_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t408_leaf0
   <= 1.000000999999e+06
 T408_L_f18_k1: const_2nd_Flr_SF + 1e+06 t408_leaf1 <= 1.001610999999e+06
 T408_L_f20_k1: x_Gr_Liv_Area + 1e+06 t408_leaf1 <= 1.002871999999e+06
 T408_R_f245_k1: const_FunctioNo_aplical_Typ - 1e+06 t408_leaf1 >= -999999
 T408_L_f18_k2: const_2nd_Flr_SF + 1e+06 t408_leaf2 <= 1.001610999999e+06
 T408_R_f20_k2: x_Gr_Liv_Area - 1e+06 t408_leaf2 >= -997128
 T408_L_f25_k2: x_Bedroom_AbvGr + 1e+06 t408_leaf2 <= 1.000003999999e+06
 T408_L_f18_k3: const_2nd_Flr_SF + 1e+06 t408_leaf3 <= 1.001610999999e+06
 T408_R_f20_k3: x_Gr_Liv_Area - 1e+06 t408_leaf3 >= -997128
 T408_R_f25_k3: x_Bedroom_AbvGr - 1e+06 t408_leaf3 >= -999996
 T408_R_f18_k4: const_2nd_Flr_SF - 1e+06 t408_leaf4 >= -998389
 T409_onehot: t409_leaf0 + t409_leaf1 + t409_leaf2 + t409_leaf3
   + t409_leaf4 + t409_leaf5 = 1
 T409_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t409_leaf0
   <= 1.001930999999e+06
 T409_L_f1_k0: const_Lot_Area + 1e+06 t409_leaf0 <= 1.010336999999e+06
 T409_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t409_leaf1
   <= 1.001930999999e+06
 T409_R_f1_k1: const_Lot_Area - 1e+06 t409_leaf1 >= -989663
 T409_L_f191_k1: x_ex2_is_VinylSd + 1e+06 t409_leaf1 <= 1.000000999999e+06
 T409_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t409_leaf2
   <= 1.001930999999e+06
 T409_R_f1_k2: const_Lot_Area - 1e+06 t409_leaf2 >= -989663
 T409_R_f191_k2: x_ex2_is_VinylSd - 1e+06 t409_leaf2 >= -999999
 T409_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t409_leaf3 >= -998069
 T409_L_f293_k3: const_Sale_Condition_Abnorml + 1e+06 t409_leaf3
   <= 1.000000999999e+06
 T409_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t409_leaf4 >= -998069
 T409_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t409_leaf4
   >= -999999
 T409_L_f36_k4: x_Wood_Deck_SF + 1e+06 t409_leaf4 <= 1.000180999999e+06
 T409_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t409_leaf5 >= -998069
 T409_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t409_leaf5
   >= -999999
 T409_R_f36_k5: x_Wood_Deck_SF - 1e+06 t409_leaf5 >= -999819
 T410_onehot: t410_leaf0 + t410_leaf1 + t410_leaf2 + t410_leaf3 = 1
 T410_L_f17_k0: x_1st_Flr_SF + 1e+06 t410_leaf0 <= 1.002199999999e+06
 T410_L_f63_k0: const_MS_Zoning_FV + 1e+06 t410_leaf0
   <= 1.000000999999e+06
 T410_L_f8_k0: x_Exter_Qual + 1e+06 t410_leaf0 <= 1.000002999999e+06
 T410_L_f17_k1: x_1st_Flr_SF + 1e+06 t410_leaf1 <= 1.002199999999e+06
 T410_L_f63_k1: const_MS_Zoning_FV + 1e+06 t410_leaf1
   <= 1.000000999999e+06
 T410_R_f8_k1: x_Exter_Qual - 1e+06 t410_leaf1 >= -999997
 T410_L_f17_k2: x_1st_Flr_SF + 1e+06 t410_leaf2 <= 1.002199999999e+06
 T410_R_f63_k2: const_MS_Zoning_FV - 1e+06 t410_leaf2 >= -999999
 T410_R_f17_k3: x_1st_Flr_SF - 1e+06 t410_leaf3 >= -997800
 T411_onehot: t411_leaf0 + t411_leaf1 + t411_leaf2 = 1
 T411_L_f72_k0: const_Alley_Pave + 1e+06 t411_leaf0 <= 1.000000999999e+06
 T411_L_f114_k0: const_Neighborhood_StoneBr + 1e+06 t411_leaf0
   <= 1.000000999999e+06
 T411_L_f72_k1: const_Alley_Pave + 1e+06 t411_leaf1 <= 1.000000999999e+06
 T411_R_f114_k1: const_Neighborhood_StoneBr - 1e+06 t411_leaf1 >= -999999
 T411_R_f72_k2: const_Alley_Pave - 1e+06 t411_leaf2 >= -999999
 T412_onehot: t412_leaf0 + t412_leaf1 + t412_leaf2 = 1
 T412_L_f274_k0: const_Mo_Sold_4 + 1e+06 t412_leaf0 <= 1.000000999999e+06
 T412_R_f274_k1: const_Mo_Sold_4 - 1e+06 t412_leaf1 >= -999999
 T412_L_f0_k1: const_Lot_Frontage + 1e+06 t412_leaf1 <= 1.000076999999e+06
 T412_R_f274_k2: const_Mo_Sold_4 - 1e+06 t412_leaf2 >= -999999
 T412_R_f0_k2: const_Lot_Frontage - 1e+06 t412_leaf2 >= -999923
 T413_onehot: t413_leaf0 + t413_leaf1 + t413_leaf2 + t413_leaf3
   + t413_leaf4 = 1
 T413_L_f1_k0: const_Lot_Area + 1e+06 t413_leaf0 <= 1.005999999999e+06
 T413_R_f1_k1: const_Lot_Area - 1e+06 t413_leaf1 >= -994000
 T413_L_f46_k1: const_MS_SubClass_30 + 1e+06 t413_leaf1
   <= 1.000000999999e+06
 T413_L_f40_k1: x_Screen_Porch + 1e+06 t413_leaf1 <= 1.000107999999e+06
 T413_R_f1_k2: const_Lot_Area - 1e+06 t413_leaf2 >= -994000
 T413_L_f46_k2: const_MS_SubClass_30 + 1e+06 t413_leaf2
   <= 1.000000999999e+06
 T413_R_f40_k2: x_Screen_Porch - 1e+06 t413_leaf2 >= -999892
 T413_R_f1_k3: const_Lot_Area - 1e+06 t413_leaf3 >= -994000
 T413_R_f46_k3: const_MS_SubClass_30 - 1e+06 t413_leaf3 >= -999999
 T413_L_f38_k3: x_Enclosed_Porch + 1e+06 t413_leaf3 <= 1.000076999999e+06
 T413_R_f1_k4: const_Lot_Area - 1e+06 t413_leaf4 >= -994000
 T413_R_f46_k4: const_MS_SubClass_30 - 1e+06 t413_leaf4 >= -999999
 T413_R_f38_k4: x_Enclosed_Porch - 1e+06 t413_leaf4 >= -999923
 T414_onehot: t414_leaf0 + t414_leaf1 + t414_leaf2 + t414_leaf3 = 1
 T414_L_f63_k0: const_MS_Zoning_FV + 1e+06 t414_leaf0
   <= 1.000000999999e+06
 T414_L_f11_k0: x_Bsmt_Cond + 1e+06 t414_leaf0 <= 1.000002999999e+06
 T414_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t414_leaf0
   <= 1.000000999999e+06
 T414_L_f63_k1: const_MS_Zoning_FV + 1e+06 t414_leaf1
   <= 1.000000999999e+06
 T414_L_f11_k1: x_Bsmt_Cond + 1e+06 t414_leaf1 <= 1.000002999999e+06
 T414_R_f96_k1: const_Neighborhood_Edwards - 1e+06 t414_leaf1 >= -999999
 T414_L_f63_k2: const_MS_Zoning_FV + 1e+06 t414_leaf2
   <= 1.000000999999e+06
 T414_R_f11_k2: x_Bsmt_Cond - 1e+06 t414_leaf2 >= -999997
 T414_R_f63_k3: const_MS_Zoning_FV - 1e+06 t414_leaf3 >= -999999
 T415_onehot: t415_leaf0 + t415_leaf1 + t415_leaf2 + t415_leaf3
   + t415_leaf4 = 1
 T415_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t415_leaf0 <= 1.000469999999e+06
 T415_L_f23_k0: x_Full_Bath + 1e+06 t415_leaf0 <= 1.000001999999e+06
 T415_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t415_leaf0
   <= 1.000000999999e+06
 T415_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t415_leaf1 <= 1.000469999999e+06
 T415_L_f23_k1: x_Full_Bath + 1e+06 t415_leaf1 <= 1.000001999999e+06
 T415_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t415_leaf1 >= -999999
 T415_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t415_leaf2 <= 1.000469999999e+06
 T415_R_f23_k2: x_Full_Bath - 1e+06 t415_leaf2 >= -999998
 T415_L_f8_k2: x_Exter_Qual + 1e+06 t415_leaf2 <= 1.000002999999e+06
 T415_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t415_leaf3 <= 1.000469999999e+06
 T415_R_f23_k3: x_Full_Bath - 1e+06 t415_leaf3 >= -999998
 T415_R_f8_k3: x_Exter_Qual - 1e+06 t415_leaf3 >= -999997
 T415_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t415_leaf4 >= -999530
 T416_onehot: t416_leaf0 + t416_leaf1 + t416_leaf2 + t416_leaf3
   + t416_leaf4 + t416_leaf5 + t416_leaf6 = 1
 T416_L_f35_k0: x_Garage_Cond + 1e+06 t416_leaf0 <= 1.000001999999e+06
 T416_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t416_leaf0
   <= 1.000000999999e+06
 T416_L_f1_k0: const_Lot_Area + 1e+06 t416_leaf0 <= 1.010989999999e+06
 T416_L_f35_k1: x_Garage_Cond + 1e+06 t416_leaf1 <= 1.000001999999e+06
 T416_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t416_leaf1
   <= 1.000000999999e+06
 T416_R_f1_k1: const_Lot_Area - 1e+06 t416_leaf1 >= -989010
 T416_L_f35_k2: x_Garage_Cond + 1e+06 t416_leaf2 <= 1.000001999999e+06
 T416_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t416_leaf2
   >= -999999
 T416_L_f28_k2: x_TotRms_AbvGrd + 1e+06 t416_leaf2 <= 1.000006999999e+06
 T416_L_f35_k3: x_Garage_Cond + 1e+06 t416_leaf3 <= 1.000001999999e+06
 T416_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t416_leaf3
   >= -999999
 T416_R_f28_k3: x_TotRms_AbvGrd - 1e+06 t416_leaf3 >= -999993
 T416_R_f35_k4: x_Garage_Cond - 1e+06 t416_leaf4 >= -999998
 T416_L_f38_k4: x_Enclosed_Porch + 1e+06 t416_leaf4 <= 1.000263999999e+06
 T416_L_f147_k4: const_Roof_Style_Flat + 1e+06 t416_leaf4
   <= 1.000000999999e+06
 T416_R_f35_k5: x_Garage_Cond - 1e+06 t416_leaf5 >= -999998
 T416_L_f38_k5: x_Enclosed_Porch + 1e+06 t416_leaf5 <= 1.000263999999e+06
 T416_R_f147_k5: const_Roof_Style_Flat - 1e+06 t416_leaf5 >= -999999
 T416_R_f35_k6: x_Garage_Cond - 1e+06 t416_leaf6 >= -999998
 T416_R_f38_k6: x_Enclosed_Porch - 1e+06 t416_leaf6 >= -999736
 T417_onehot: t417_leaf0 = 1
 T418_onehot: t418_leaf0 + t418_leaf1 = 1
 T418_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t418_leaf0 <= 1.001964999999e+06
 T418_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t418_leaf1 >= -998035
 T419_onehot: t419_leaf0 + t419_leaf1 + t419_leaf2 + t419_leaf3
   + t419_leaf4 + t419_leaf5 + t419_leaf6 + t419_leaf7 = 1
 T419_L_f3_k0: Overall_Qual_calc + 1e+06 t419_leaf0 <= 1.000004999999e+06
 T419_L_f10_k0: const_Bsmt_Qual + 1e+06 t419_leaf0 <= 999999.999999
 T419_L_f0_k0: const_Lot_Frontage + 1e+06 t419_leaf0 <= 1.000065999999e+06
 T419_L_f3_k1: Overall_Qual_calc + 1e+06 t419_leaf1 <= 1.000004999999e+06
 T419_L_f10_k1: const_Bsmt_Qual + 1e+06 t419_leaf1 <= 999999.999999
 T419_R_f0_k1: const_Lot_Frontage - 1e+06 t419_leaf1 >= -999934
 T419_L_f3_k2: Overall_Qual_calc + 1e+06 t419_leaf2 <= 1.000004999999e+06
 T419_R_f10_k2: const_Bsmt_Qual - 1e+06 t419_leaf2 >= -1e+06
 T419_L_f0_k2: const_Lot_Frontage + 1e+06 t419_leaf2 <= 1.000021999999e+06
 T419_L_f3_k3: Overall_Qual_calc + 1e+06 t419_leaf3 <= 1.000004999999e+06
 T419_R_f10_k3: const_Bsmt_Qual - 1e+06 t419_leaf3 >= -1e+06
 T419_R_f0_k3: const_Lot_Frontage - 1e+06 t419_leaf3 >= -999978
 T419_R_f3_k4: Overall_Qual_calc - 1e+06 t419_leaf4 >= -999995
 T419_L_f23_k4: x_Full_Bath + 1e+06 t419_leaf4 <= 1.000001999999e+06
 T419_L_f119_k4: const_Condition_1_Norm + 1e+06 t419_leaf4
   <= 1.000000999999e+06
 T419_R_f3_k5: Overall_Qual_calc - 1e+06 t419_leaf5 >= -999995
 T419_L_f23_k5: x_Full_Bath + 1e+06 t419_leaf5 <= 1.000001999999e+06
 T419_R_f119_k5: const_Condition_1_Norm - 1e+06 t419_leaf5 >= -999999
 T419_R_f3_k6: Overall_Qual_calc - 1e+06 t419_leaf6 >= -999995
 T419_R_f23_k6: x_Full_Bath - 1e+06 t419_leaf6 >= -999998
 T419_L_f3_k6: Overall_Qual_calc + 1e+06 t419_leaf6 <= 1.000006999999e+06
 T419_R_f3_k7: Overall_Qual_calc - 1e+06 t419_leaf7 >= -999995
 T419_R_f23_k7: x_Full_Bath - 1e+06 t419_leaf7 >= -999998
 T419_R_f3_k7: Overall_Qual_calc - 1e+06 t419_leaf7 >= -999993
 T420_onehot: t420_leaf0 = 1
 T421_onehot: t421_leaf0 + t421_leaf1 + t421_leaf2 + t421_leaf3 = 1
 T421_L_f9_k0: x_Exter_Cond + 1e+06 t421_leaf0 <= 1.000001999999e+06
 T421_L_f23_k0: x_Full_Bath + 1e+06 t421_leaf0 <= 1.000001999999e+06
 T421_L_f0_k0: const_Lot_Frontage + 1e+06 t421_leaf0 <= 1.000060999999e+06
 T421_L_f9_k1: x_Exter_Cond + 1e+06 t421_leaf1 <= 1.000001999999e+06
 T421_L_f23_k1: x_Full_Bath + 1e+06 t421_leaf1 <= 1.000001999999e+06
 T421_R_f0_k1: const_Lot_Frontage - 1e+06 t421_leaf1 >= -999939
 T421_L_f9_k2: x_Exter_Cond + 1e+06 t421_leaf2 <= 1.000001999999e+06
 T421_R_f23_k2: x_Full_Bath - 1e+06 t421_leaf2 >= -999998
 T421_R_f9_k3: x_Exter_Cond - 1e+06 t421_leaf3 >= -999998
 T422_onehot: t422_leaf0 + t422_leaf1 + t422_leaf2 + t422_leaf3
   + t422_leaf4 = 1
 T422_L_f37_k0: x_Open_Porch_SF + 1e+06 t422_leaf0 <= 1.000257999999e+06
 T422_L_f11_k0: x_Bsmt_Cond + 1e+06 t422_leaf0 <= 1.000001999999e+06
 T422_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t422_leaf0
   <= 1.000000999999e+06
 T422_L_f37_k1: x_Open_Porch_SF + 1e+06 t422_leaf1 <= 1.000257999999e+06
 T422_L_f11_k1: x_Bsmt_Cond + 1e+06 t422_leaf1 <= 1.000001999999e+06
 T422_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t422_leaf1 >= -999999
 T422_L_f37_k2: x_Open_Porch_SF + 1e+06 t422_leaf2 <= 1.000257999999e+06
 T422_R_f11_k2: x_Bsmt_Cond - 1e+06 t422_leaf2 >= -999998
 T422_L_f117_k2: const_Condition_1_Artery + 1e+06 t422_leaf2
   <= 1.000000999999e+06
 T422_L_f37_k3: x_Open_Porch_SF + 1e+06 t422_leaf3 <= 1.000257999999e+06
 T422_R_f11_k3: x_Bsmt_Cond - 1e+06 t422_leaf3 >= -999998
 T422_R_f117_k3: const_Condition_1_Artery - 1e+06 t422_leaf3 >= -999999
 T422_R_f37_k4: x_Open_Porch_SF - 1e+06 t422_leaf4 >= -999742
 T423_onehot: t423_leaf0 + t423_leaf1 + t423_leaf2 + t423_leaf3 = 1
 T423_L_f38_k0: x_Enclosed_Porch + 1e+06 t423_leaf0 <= 1.000289999999e+06
 T423_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t423_leaf0 <= 1.000004999999e+06
 T423_L_f11_k0: x_Bsmt_Cond + 1e+06 t423_leaf0 <= 1.000000999999e+06
 T423_L_f38_k1: x_Enclosed_Porch + 1e+06 t423_leaf1 <= 1.000289999999e+06
 T423_L_f25_k1: x_Bedroom_AbvGr + 1e+06 t423_leaf1 <= 1.000004999999e+06
 T423_R_f11_k1: x_Bsmt_Cond - 1e+06 t423_leaf1 >= -999999
 T423_L_f38_k2: x_Enclosed_Porch + 1e+06 t423_leaf2 <= 1.000289999999e+06
 T423_R_f25_k2: x_Bedroom_AbvGr - 1e+06 t423_leaf2 >= -999995
 T423_R_f38_k3: x_Enclosed_Porch - 1e+06 t423_leaf3 >= -999710
 T424_onehot: t424_leaf0 + t424_leaf1 + t424_leaf2 + t424_leaf3
   + t424_leaf4 + t424_leaf5 = 1
 T424_L_f3_k0: Overall_Qual_calc + 1e+06 t424_leaf0 <= 1.000002999999e+06
 T424_L_f38_k0: x_Enclosed_Porch + 1e+06 t424_leaf0 <= 1.000025999999e+06
 T424_L_f3_k1: Overall_Qual_calc + 1e+06 t424_leaf1 <= 1.000002999999e+06
 T424_R_f38_k1: x_Enclosed_Porch - 1e+06 t424_leaf1 >= -999974
 T424_R_f3_k2: Overall_Qual_calc - 1e+06 t424_leaf2 >= -999997
 T424_L_f119_k2: const_Condition_1_Norm + 1e+06 t424_leaf2
   <= 1.000000999999e+06
 T424_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t424_leaf2 <= 1.000915999999e+06
 T424_R_f3_k3: Overall_Qual_calc - 1e+06 t424_leaf3 >= -999997
 T424_L_f119_k3: const_Condition_1_Norm + 1e+06 t424_leaf3
   <= 1.000000999999e+06
 T424_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t424_leaf3 >= -999084
 T424_R_f3_k4: Overall_Qual_calc - 1e+06 t424_leaf4 >= -999997
 T424_R_f119_k4: const_Condition_1_Norm - 1e+06 t424_leaf4 >= -999999
 T424_L_f164_k4: x_ex1_is_BrkFace + 1e+06 t424_leaf4 <= 1.000000999999e+06
 T424_R_f3_k5: Overall_Qual_calc - 1e+06 t424_leaf5 >= -999997
 T424_R_f119_k5: const_Condition_1_Norm - 1e+06 t424_leaf5 >= -999999
 T424_R_f164_k5: x_ex1_is_BrkFace - 1e+06 t424_leaf5 >= -999999
 T425_onehot: t425_leaf0 + t425_leaf1 + t425_leaf2 = 1
 T425_L_f92_k0: const_Neighborhood_BrkSide + 1e+06 t425_leaf0
   <= 1.000000999999e+06
 T425_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t425_leaf0
   <= 1.000000999999e+06
 T425_L_f92_k1: const_Neighborhood_BrkSide + 1e+06 t425_leaf1
   <= 1.000000999999e+06
 T425_R_f245_k1: const_FunctioNo_aplical_Typ - 1e+06 t425_leaf1 >= -999999
 T425_R_f92_k2: const_Neighborhood_BrkSide - 1e+06 t425_leaf2 >= -999999
 T426_onehot: t426_leaf0 + t426_leaf1 + t426_leaf2 + t426_leaf3
   + t426_leaf4 + t426_leaf5 + t426_leaf6 = 1
 T426_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t426_leaf0
   <= 1.000000999999e+06
 T426_L_f20_k0: x_Gr_Liv_Area + 1e+06 t426_leaf0 <= 1.003394999999e+06
 T426_L_f35_k0: x_Garage_Cond + 1e+06 t426_leaf0 <= 1.000001999999e+06
 T426_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t426_leaf1
   <= 1.000000999999e+06
 T426_L_f20_k1: x_Gr_Liv_Area + 1e+06 t426_leaf1 <= 1.003394999999e+06
 T426_R_f35_k1: x_Garage_Cond - 1e+06 t426_leaf1 >= -999998
 T426_L_f293_k2: const_Sale_Condition_Abnorml + 1e+06 t426_leaf2
   <= 1.000000999999e+06
 T426_R_f20_k2: x_Gr_Liv_Area - 1e+06 t426_leaf2 >= -996605
 T426_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t426_leaf3
   >= -999999
 T426_L_f81_k3: const_Lot_Config_Corner + 1e+06 t426_leaf3
   <= 1.000000999999e+06
 T426_L_f1_k3: const_Lot_Area + 1e+06 t426_leaf3 <= 1.004670999999e+06
 T426_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t426_leaf4
   >= -999999
 T426_L_f81_k4: const_Lot_Config_Corner + 1e+06 t426_leaf4
   <= 1.000000999999e+06
 T426_R_f1_k4: const_Lot_Area - 1e+06 t426_leaf4 >= -995329
 T426_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t426_leaf5
   >= -999999
 T426_R_f81_k5: const_Lot_Config_Corner - 1e+06 t426_leaf5 >= -999999
 T426_L_f20_k5: x_Gr_Liv_Area + 1e+06 t426_leaf5 <= 1.001127999999e+06
 T426_R_f293_k6: const_Sale_Condition_Abnorml - 1e+06 t426_leaf6
   >= -999999
 T426_R_f81_k6: const_Lot_Config_Corner - 1e+06 t426_leaf6 >= -999999
 T426_R_f20_k6: x_Gr_Liv_Area - 1e+06 t426_leaf6 >= -998872
 T427_onehot: t427_leaf0 + t427_leaf1 + t427_leaf2 + t427_leaf3 = 1
 T427_L_f1_k0: const_Lot_Area + 1e+06 t427_leaf0 <= 1.027696999999e+06
 T427_L_f1_k0: const_Lot_Area + 1e+06 t427_leaf0 <= 1.023256999999e+06
 T427_L_f194_k0: x_mvt_is_BrkCmn + 1e+06 t427_leaf0 <= 1.000000999999e+06
 T427_L_f1_k1: const_Lot_Area + 1e+06 t427_leaf1 <= 1.027696999999e+06
 T427_L_f1_k1: const_Lot_Area + 1e+06 t427_leaf1 <= 1.023256999999e+06
 T427_R_f194_k1: x_mvt_is_BrkCmn - 1e+06 t427_leaf1 >= -999999
 T427_L_f1_k2: const_Lot_Area + 1e+06 t427_leaf2 <= 1.027696999999e+06
 T427_R_f1_k2: const_Lot_Area - 1e+06 t427_leaf2 >= -976743
 T427_R_f1_k3: const_Lot_Area - 1e+06 t427_leaf3 >= -972303
 T428_onehot: t428_leaf0 + t428_leaf1 + t428_leaf2 + t428_leaf3
   + t428_leaf4 = 1
 T428_L_f38_k0: x_Enclosed_Porch + 1e+06 t428_leaf0 <= 1.000267999999e+06
 T428_L_f5_k0: const_Year_Built + 1e+06 t428_leaf0 <= 1.001945999999e+06
 T428_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t428_leaf0
   <= 1.000000999999e+06
 T428_L_f38_k1: x_Enclosed_Porch + 1e+06 t428_leaf1 <= 1.000267999999e+06
 T428_L_f5_k1: const_Year_Built + 1e+06 t428_leaf1 <= 1.001945999999e+06
 T428_R_f95_k1: const_Neighborhood_Crawfor - 1e+06 t428_leaf1 >= -999999
 T428_L_f38_k2: x_Enclosed_Porch + 1e+06 t428_leaf2 <= 1.000267999999e+06
 T428_R_f5_k2: const_Year_Built - 1e+06 t428_leaf2 >= -998054
 T428_L_f5_k2: const_Year_Built + 1e+06 t428_leaf2 <= 1.001949999999e+06
 T428_L_f38_k3: x_Enclosed_Porch + 1e+06 t428_leaf3 <= 1.000267999999e+06
 T428_R_f5_k3: const_Year_Built - 1e+06 t428_leaf3 >= -998054
 T428_R_f5_k3: const_Year_Built - 1e+06 t428_leaf3 >= -998050
 T428_R_f38_k4: x_Enclosed_Porch - 1e+06 t428_leaf4 >= -999732
 T429_onehot: t429_leaf0 + t429_leaf1 + t429_leaf2 + t429_leaf3
   + t429_leaf4 + t429_leaf5 = 1
 T429_L_f30_k0: x_Fireplace_Qu + 1e+06 t429_leaf0 <= 1.000001999999e+06
 T429_L_f22_k0: const_Bsmt_Half_Bath + 1e+06 t429_leaf0
   <= 1.000000999999e+06
 T429_L_f30_k1: x_Fireplace_Qu + 1e+06 t429_leaf1 <= 1.000001999999e+06
 T429_R_f22_k1: const_Bsmt_Half_Bath - 1e+06 t429_leaf1 >= -999999
 T429_R_f30_k2: x_Fireplace_Qu - 1e+06 t429_leaf2 >= -999998
 T429_L_f96_k2: const_Neighborhood_Edwards + 1e+06 t429_leaf2
   <= 1.000000999999e+06
 T429_L_f296_k2: const_Sale_Condition_Family + 1e+06 t429_leaf2
   <= 1.000000999999e+06
 T429_R_f30_k3: x_Fireplace_Qu - 1e+06 t429_leaf3 >= -999998
 T429_L_f96_k3: const_Neighborhood_Edwards + 1e+06 t429_leaf3
   <= 1.000000999999e+06
 T429_R_f296_k3: const_Sale_Condition_Family - 1e+06 t429_leaf3 >= -999999
 T429_R_f30_k4: x_Fireplace_Qu - 1e+06 t429_leaf4 >= -999998
 T429_R_f96_k4: const_Neighborhood_Edwards - 1e+06 t429_leaf4 >= -999999
 T429_L_f150_k4: const_Roof_Style_Hip + 1e+06 t429_leaf4
   <= 1.000000999999e+06
 T429_R_f30_k5: x_Fireplace_Qu - 1e+06 t429_leaf5 >= -999998
 T429_R_f96_k5: const_Neighborhood_Edwards - 1e+06 t429_leaf5 >= -999999
 T429_R_f150_k5: const_Roof_Style_Hip - 1e+06 t429_leaf5 >= -999999
 T430_onehot: t430_leaf0 + t430_leaf1 + t430_leaf2 + t430_leaf3
   + t430_leaf4 + t430_leaf5 + t430_leaf6 = 1
 T430_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t430_leaf0 <= 1.000136999999e+06
 T430_L_f38_k0: x_Enclosed_Porch + 1e+06 t430_leaf0 <= 1.000223999999e+06
 T430_L_f272_k0: const_Mo_Sold_2 + 1e+06 t430_leaf0 <= 1.000000999999e+06
 T430_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t430_leaf1 <= 1.000136999999e+06
 T430_L_f38_k1: x_Enclosed_Porch + 1e+06 t430_leaf1 <= 1.000223999999e+06
 T430_R_f272_k1: const_Mo_Sold_2 - 1e+06 t430_leaf1 >= -999999
 T430_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t430_leaf2 <= 1.000136999999e+06
 T430_R_f38_k2: x_Enclosed_Porch - 1e+06 t430_leaf2 >= -999776
 T430_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t430_leaf3 >= -999863
 T430_L_f1_k3: const_Lot_Area + 1e+06 t430_leaf3 <= 1.020430999999e+06
 T430_L_f18_k3: const_2nd_Flr_SF + 1e+06 t430_leaf3 <= 1.000639999999e+06
 T430_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t430_leaf4 >= -999863
 T430_L_f1_k4: const_Lot_Area + 1e+06 t430_leaf4 <= 1.020430999999e+06
 T430_R_f18_k4: const_2nd_Flr_SF - 1e+06 t430_leaf4 >= -999360
 T430_R_f14_k5: x_Bsmt_Unf_SF - 1e+06 t430_leaf5 >= -999863
 T430_R_f1_k5: const_Lot_Area - 1e+06 t430_leaf5 >= -979569
 T430_L_f38_k5: x_Enclosed_Porch + 1e+06 t430_leaf5 <= 1.000193999999e+06
 T430_R_f14_k6: x_Bsmt_Unf_SF - 1e+06 t430_leaf6 >= -999863
 T430_R_f1_k6: const_Lot_Area - 1e+06 t430_leaf6 >= -979569
 T430_R_f38_k6: x_Enclosed_Porch - 1e+06 t430_leaf6 >= -999806
 T431_onehot: t431_leaf0 + t431_leaf1 + t431_leaf2 + t431_leaf3 = 1
 T431_L_f203_k0: const_Foundation_Slab + 1e+06 t431_leaf0
   <= 1.000000999999e+06
 T431_L_f37_k0: x_Open_Porch_SF + 1e+06 t431_leaf0 <= 1.000286999999e+06
 T431_L_f20_k0: x_Gr_Liv_Area + 1e+06 t431_leaf0 <= 1.002648999999e+06
 T431_L_f203_k1: const_Foundation_Slab + 1e+06 t431_leaf1
   <= 1.000000999999e+06
 T431_L_f37_k1: x_Open_Porch_SF + 1e+06 t431_leaf1 <= 1.000286999999e+06
 T431_R_f20_k1: x_Gr_Liv_Area - 1e+06 t431_leaf1 >= -997351
 T431_L_f203_k2: const_Foundation_Slab + 1e+06 t431_leaf2
   <= 1.000000999999e+06
 T431_R_f37_k2: x_Open_Porch_SF - 1e+06 t431_leaf2 >= -999713
 T431_R_f203_k3: const_Foundation_Slab - 1e+06 t431_leaf3 >= -999999
 T432_onehot: t432_leaf0 + t432_leaf1 + t432_leaf2 + t432_leaf3
   + t432_leaf4 + t432_leaf5 = 1
 T432_L_f1_k0: const_Lot_Area + 1e+06 t432_leaf0 <= 1.019521999999e+06
 T432_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t432_leaf0
   <= 1.002135999999e+06
 T432_L_f38_k0: x_Enclosed_Porch + 1e+06 t432_leaf0 <= 1.000223999999e+06
 T432_L_f1_k1: const_Lot_Area + 1e+06 t432_leaf1 <= 1.019521999999e+06
 T432_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t432_leaf1
   <= 1.002135999999e+06
 T432_R_f38_k1: x_Enclosed_Porch - 1e+06 t432_leaf1 >= -999776
 T432_L_f1_k2: const_Lot_Area + 1e+06 t432_leaf2 <= 1.019521999999e+06
 T432_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t432_leaf2 >= -997864
 T432_R_f1_k3: const_Lot_Area - 1e+06 t432_leaf3 >= -980478
 T432_L_f25_k3: x_Bedroom_AbvGr + 1e+06 t432_leaf3 <= 1.000003999999e+06
 T432_L_f297_k3: const_Sale_Condition_Normal + 1e+06 t432_leaf3
   <= 1.000000999999e+06
 T432_R_f1_k4: const_Lot_Area - 1e+06 t432_leaf4 >= -980478
 T432_L_f25_k4: x_Bedroom_AbvGr + 1e+06 t432_leaf4 <= 1.000003999999e+06
 T432_R_f297_k4: const_Sale_Condition_Normal - 1e+06 t432_leaf4 >= -999999
 T432_R_f1_k5: const_Lot_Area - 1e+06 t432_leaf5 >= -980478
 T432_R_f25_k5: x_Bedroom_AbvGr - 1e+06 t432_leaf5 >= -999996
 T433_onehot: t433_leaf0 + t433_leaf1 + t433_leaf2 = 1
 T433_L_f37_k0: x_Open_Porch_SF + 1e+06 t433_leaf0 <= 1.000207999999e+06
 T433_L_f218_k0: x_b2_is_ALQ + 1e+06 t433_leaf0 <= 1.000000999999e+06
 T433_L_f37_k1: x_Open_Porch_SF + 1e+06 t433_leaf1 <= 1.000207999999e+06
 T433_R_f218_k1: x_b2_is_ALQ - 1e+06 t433_leaf1 >= -999999
 T433_R_f37_k2: x_Open_Porch_SF - 1e+06 t433_leaf2 >= -999792
 T434_onehot: t434_leaf0 = 1
 T435_onehot: t435_leaf0 + t435_leaf1 + t435_leaf2 + t435_leaf3
   + t435_leaf4 = 1
 T435_L_f38_k0: x_Enclosed_Porch + 1e+06 t435_leaf0 <= 1.000193999999e+06
 T435_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t435_leaf0
   <= 1.000000999999e+06
 T435_L_f38_k0: x_Enclosed_Porch + 1e+06 t435_leaf0 <= 1.000063999999e+06
 T435_L_f38_k1: x_Enclosed_Porch + 1e+06 t435_leaf1 <= 1.000193999999e+06
 T435_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t435_leaf1
   <= 1.000000999999e+06
 T435_R_f38_k1: x_Enclosed_Porch - 1e+06 t435_leaf1 >= -999936
 T435_L_f38_k2: x_Enclosed_Porch + 1e+06 t435_leaf2 <= 1.000193999999e+06
 T435_R_f245_k2: const_FunctioNo_aplical_Typ - 1e+06 t435_leaf2 >= -999999
 T435_L_f4_k2: const_Overall_Cond + 1e+06 t435_leaf2 <= 1.000006999999e+06
 T435_L_f38_k3: x_Enclosed_Porch + 1e+06 t435_leaf3 <= 1.000193999999e+06
 T435_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t435_leaf3 >= -999999
 T435_R_f4_k3: const_Overall_Cond - 1e+06 t435_leaf3 >= -999993
 T435_R_f38_k4: x_Enclosed_Porch - 1e+06 t435_leaf4 >= -999806
 T436_onehot: t436_leaf0 = 1
 T437_onehot: t437_leaf0 + t437_leaf1 + t437_leaf2 = 1
 T437_L_f3_k0: Overall_Qual_calc + 1e+06 t437_leaf0 <= 1.000003999999e+06
 T437_L_f80_k0: const_Land_Contour_Lvl + 1e+06 t437_leaf0
   <= 1.000000999999e+06
 T437_L_f3_k1: Overall_Qual_calc + 1e+06 t437_leaf1 <= 1.000003999999e+06
 T437_R_f80_k1: const_Land_Contour_Lvl - 1e+06 t437_leaf1 >= -999999
 T437_R_f3_k2: Overall_Qual_calc - 1e+06 t437_leaf2 >= -999996
 T438_onehot: t438_leaf0 + t438_leaf1 = 1
 T438_L_f119_k0: const_Condition_1_Norm + 1e+06 t438_leaf0
   <= 1.000000999999e+06
 T438_R_f119_k1: const_Condition_1_Norm - 1e+06 t438_leaf1 >= -999999
 T439_onehot: t439_leaf0 + t439_leaf1 + t439_leaf2 + t439_leaf3
   + t439_leaf4 = 1
 T439_L_f231_k0: central_air_no + 1e+06 t439_leaf0 <= 1.000000999999e+06
 T439_R_f231_k1: central_air_no - 1e+06 t439_leaf1 >= -999999
 T439_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t439_leaf1 <= 1.000315999999e+06
 T439_L_f275_k1: const_Mo_Sold_5 + 1e+06 t439_leaf1 <= 1.000000999999e+06
 T439_R_f231_k2: central_air_no - 1e+06 t439_leaf2 >= -999999
 T439_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t439_leaf2 <= 1.000315999999e+06
 T439_R_f275_k2: const_Mo_Sold_5 - 1e+06 t439_leaf2 >= -999999
 T439_R_f231_k3: central_air_no - 1e+06 t439_leaf3 >= -999999
 T439_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t439_leaf3 >= -999684
 T439_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t439_leaf3 <= 1.000636999999e+06
 T439_R_f231_k4: central_air_no - 1e+06 t439_leaf4 >= -999999
 T439_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t439_leaf4 >= -999684
 T439_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t439_leaf4 >= -999363
 T440_onehot: t440_leaf0 + t440_leaf1 + t440_leaf2 + t440_leaf3
   + t440_leaf4 + t440_leaf5 = 1
 T440_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t440_leaf0 <= 1.000002999999e+06
 T440_L_f20_k0: x_Gr_Liv_Area + 1e+06 t440_leaf0 <= 1.001794999999e+06
 T440_L_f25_k1: x_Bedroom_AbvGr + 1e+06 t440_leaf1 <= 1.000002999999e+06
 T440_R_f20_k1: x_Gr_Liv_Area - 1e+06 t440_leaf1 >= -998205
 T440_L_f174_k1: x_ex1_is_VinylSd + 1e+06 t440_leaf1 <= 1.000000999999e+06
 T440_L_f25_k2: x_Bedroom_AbvGr + 1e+06 t440_leaf2 <= 1.000002999999e+06
 T440_R_f20_k2: x_Gr_Liv_Area - 1e+06 t440_leaf2 >= -998205
 T440_R_f174_k2: x_ex1_is_VinylSd - 1e+06 t440_leaf2 >= -999999
 T440_R_f25_k3: x_Bedroom_AbvGr - 1e+06 t440_leaf3 >= -999997
 T440_L_f20_k3: x_Gr_Liv_Area + 1e+06 t440_leaf3 <= 1.000975999999e+06
 T440_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t440_leaf3
   <= 1.001966999999e+06
 T440_R_f25_k4: x_Bedroom_AbvGr - 1e+06 t440_leaf4 >= -999997
 T440_L_f20_k4: x_Gr_Liv_Area + 1e+06 t440_leaf4 <= 1.000975999999e+06
 T440_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t440_leaf4 >= -998033
 T440_R_f25_k5: x_Bedroom_AbvGr - 1e+06 t440_leaf5 >= -999997
 T440_R_f20_k5: x_Gr_Liv_Area - 1e+06 t440_leaf5 >= -999024
 T441_onehot: t441_leaf0 + t441_leaf1 + t441_leaf2 + t441_leaf3
   + t441_leaf4 = 1
 T441_L_f34_k0: x_Garage_Qual + 1e+06 t441_leaf0 <= 1.000000999999e+06
 T441_L_f1_k0: const_Lot_Area + 1e+06 t441_leaf0 <= 1.008799999999e+06
 T441_L_f34_k1: x_Garage_Qual + 1e+06 t441_leaf1 <= 1.000000999999e+06
 T441_R_f1_k1: const_Lot_Area - 1e+06 t441_leaf1 >= -991200
 T441_R_f34_k2: x_Garage_Qual - 1e+06 t441_leaf2 >= -999999
 T441_L_f46_k2: const_MS_SubClass_30 + 1e+06 t441_leaf2
   <= 1.000000999999e+06
 T441_R_f34_k3: x_Garage_Qual - 1e+06 t441_leaf3 >= -999999
 T441_R_f46_k3: const_MS_SubClass_30 - 1e+06 t441_leaf3 >= -999999
 T441_L_f17_k3: x_1st_Flr_SF + 1e+06 t441_leaf3 <= 1.000819999999e+06
 T441_R_f34_k4: x_Garage_Qual - 1e+06 t441_leaf4 >= -999999
 T441_R_f46_k4: const_MS_SubClass_30 - 1e+06 t441_leaf4 >= -999999
 T441_R_f17_k4: x_1st_Flr_SF - 1e+06 t441_leaf4 >= -999180
 T442_onehot: t442_leaf0 + t442_leaf1 + t442_leaf2 = 1
 T442_L_f147_k0: const_Roof_Style_Flat + 1e+06 t442_leaf0
   <= 1.000000999999e+06
 T442_L_f120_k0: const_Condition_1_PosA + 1e+06 t442_leaf0
   <= 1.000000999999e+06
 T442_L_f147_k1: const_Roof_Style_Flat + 1e+06 t442_leaf1
   <= 1.000000999999e+06
 T442_R_f120_k1: const_Condition_1_PosA - 1e+06 t442_leaf1 >= -999999
 T442_R_f147_k2: const_Roof_Style_Flat - 1e+06 t442_leaf2 >= -999999
 T443_onehot: t443_leaf0 = 1
 T444_onehot: t444_leaf0 + t444_leaf1 + t444_leaf2 + t444_leaf3
   + t444_leaf4 = 1
 T444_L_f30_k0: x_Fireplace_Qu + 1e+06 t444_leaf0 <= 1.000000999999e+06
 T444_L_f18_k0: const_2nd_Flr_SF + 1e+06 t444_leaf0 <= 1.000663999999e+06
 T444_L_f5_k0: const_Year_Built + 1e+06 t444_leaf0 <= 1.001935999999e+06
 T444_L_f30_k1: x_Fireplace_Qu + 1e+06 t444_leaf1 <= 1.000000999999e+06
 T444_L_f18_k1: const_2nd_Flr_SF + 1e+06 t444_leaf1 <= 1.000663999999e+06
 T444_R_f5_k1: const_Year_Built - 1e+06 t444_leaf1 >= -998064
 T444_L_f30_k2: x_Fireplace_Qu + 1e+06 t444_leaf2 <= 1.000000999999e+06
 T444_R_f18_k2: const_2nd_Flr_SF - 1e+06 t444_leaf2 >= -999336
 T444_L_f279_k2: const_Mo_Sold_9 + 1e+06 t444_leaf2 <= 1.000000999999e+06
 T444_L_f30_k3: x_Fireplace_Qu + 1e+06 t444_leaf3 <= 1.000000999999e+06
 T444_R_f18_k3: const_2nd_Flr_SF - 1e+06 t444_leaf3 >= -999336
 T444_R_f279_k3: const_Mo_Sold_9 - 1e+06 t444_leaf3 >= -999999
 T444_R_f30_k4: x_Fireplace_Qu - 1e+06 t444_leaf4 >= -999999
 T445_onehot: t445_leaf0 + t445_leaf1 + t445_leaf2 = 1
 T445_L_f290_k0: const_Sale_Type_Oth + 1e+06 t445_leaf0
   <= 1.000000999999e+06
 T445_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t445_leaf0 <= 1.001256999999e+06
 T445_L_f290_k1: const_Sale_Type_Oth + 1e+06 t445_leaf1
   <= 1.000000999999e+06
 T445_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t445_leaf1 >= -998743
 T445_R_f290_k2: const_Sale_Type_Oth - 1e+06 t445_leaf2 >= -999999
 T446_onehot: t446_leaf0 + t446_leaf1 + t446_leaf2 + t446_leaf3 = 1
 T446_L_f20_k0: x_Gr_Liv_Area + 1e+06 t446_leaf0 <= 1.000986999999e+06
 T446_L_f6_k0: const_Year_Remod_Add + 1e+06 t446_leaf0
   <= 1.001952999999e+06
 T446_L_f20_k1: x_Gr_Liv_Area + 1e+06 t446_leaf1 <= 1.000986999999e+06
 T446_R_f6_k1: const_Year_Remod_Add - 1e+06 t446_leaf1 >= -998047
 T446_L_f5_k1: const_Year_Built + 1e+06 t446_leaf1 <= 1.001935999999e+06
 T446_L_f20_k2: x_Gr_Liv_Area + 1e+06 t446_leaf2 <= 1.000986999999e+06
 T446_R_f6_k2: const_Year_Remod_Add - 1e+06 t446_leaf2 >= -998047
 T446_R_f5_k2: const_Year_Built - 1e+06 t446_leaf2 >= -998064
 T446_R_f20_k3: x_Gr_Liv_Area - 1e+06 t446_leaf3 >= -999013
 T447_onehot: t447_leaf0 + t447_leaf1 + t447_leaf2 = 1
 T447_L_f5_k0: const_Year_Built + 1e+06 t447_leaf0 <= 1.002006999999e+06
 T447_R_f5_k1: const_Year_Built - 1e+06 t447_leaf1 >= -997993
 T447_L_f0_k1: const_Lot_Frontage + 1e+06 t447_leaf1 <= 1.000112999999e+06
 T447_R_f5_k2: const_Year_Built - 1e+06 t447_leaf2 >= -997993
 T447_R_f0_k2: const_Lot_Frontage - 1e+06 t447_leaf2 >= -999887
 T448_onehot: t448_leaf0 + t448_leaf1 + t448_leaf2 = 1
 T448_L_f17_k0: x_1st_Flr_SF + 1e+06 t448_leaf0 <= 1.002514999999e+06
 T448_L_f1_k0: const_Lot_Area + 1e+06 t448_leaf0 <= 1.019521999999e+06
 T448_L_f17_k1: x_1st_Flr_SF + 1e+06 t448_leaf1 <= 1.002514999999e+06
 T448_R_f1_k1: const_Lot_Area - 1e+06 t448_leaf1 >= -980478
 T448_R_f17_k2: x_1st_Flr_SF - 1e+06 t448_leaf2 >= -997485
 T449_onehot: t449_leaf0 + t449_leaf1 + t449_leaf2 + t449_leaf3
   + t449_leaf4 + t449_leaf5 = 1
 T449_L_f33_k0: x_Garage_Area + 1e+06 t449_leaf0 <= 1.000870999999e+06
 T449_L_f148_k0: const_Roof_Style_Gable + 1e+06 t449_leaf0
   <= 1.000000999999e+06
 T449_L_f44_k0: const_Yr_Sold + 1e+06 t449_leaf0 <= 1.002007999999e+06
 T449_L_f33_k1: x_Garage_Area + 1e+06 t449_leaf1 <= 1.000870999999e+06
 T449_L_f148_k1: const_Roof_Style_Gable + 1e+06 t449_leaf1
   <= 1.000000999999e+06
 T449_R_f44_k1: const_Yr_Sold - 1e+06 t449_leaf1 >= -997992
 T449_L_f33_k2: x_Garage_Area + 1e+06 t449_leaf2 <= 1.000870999999e+06
 T449_R_f148_k2: const_Roof_Style_Gable - 1e+06 t449_leaf2 >= -999999
 T449_L_f78_k2: const_Land_Contour_HLS + 1e+06 t449_leaf2
   <= 1.000000999999e+06
 T449_L_f33_k3: x_Garage_Area + 1e+06 t449_leaf3 <= 1.000870999999e+06
 T449_R_f148_k3: const_Roof_Style_Gable - 1e+06 t449_leaf3 >= -999999
 T449_R_f78_k3: const_Land_Contour_HLS - 1e+06 t449_leaf3 >= -999999
 T449_R_f33_k4: x_Garage_Area - 1e+06 t449_leaf4 >= -999129
 T449_L_f7_k4: x_Mas_Vnr_Area + 1e+06 t449_leaf4 <= 1.000723999999e+06
 T449_R_f33_k5: x_Garage_Area - 1e+06 t449_leaf5 >= -999129
 T449_R_f7_k5: x_Mas_Vnr_Area - 1e+06 t449_leaf5 >= -999276
 T450_onehot: t450_leaf0 = 1
 T451_onehot: t451_leaf0 + t451_leaf1 + t451_leaf2 + t451_leaf3
   + t451_leaf4 + t451_leaf5 = 1
 T451_L_f272_k0: const_Mo_Sold_2 + 1e+06 t451_leaf0 <= 1.000000999999e+06
 T451_L_f250_k0: const_Garage_Type_CarPort + 1e+06 t451_leaf0
   <= 1.000000999999e+06
 T451_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t451_leaf0 <= 1.000112999999e+06
 T451_L_f272_k1: const_Mo_Sold_2 + 1e+06 t451_leaf1 <= 1.000000999999e+06
 T451_L_f250_k1: const_Garage_Type_CarPort + 1e+06 t451_leaf1
   <= 1.000000999999e+06
 T451_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t451_leaf1 >= -999887
 T451_L_f272_k2: const_Mo_Sold_2 + 1e+06 t451_leaf2 <= 1.000000999999e+06
 T451_R_f250_k2: const_Garage_Type_CarPort - 1e+06 t451_leaf2 >= -999999
 T451_R_f272_k3: const_Mo_Sold_2 - 1e+06 t451_leaf3 >= -999999
 T451_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t451_leaf3 <= 1.000218999999e+06
 T451_L_f18_k3: const_2nd_Flr_SF + 1e+06 t451_leaf3 <= 1.000759999999e+06
 T451_R_f272_k4: const_Mo_Sold_2 - 1e+06 t451_leaf4 >= -999999
 T451_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t451_leaf4 <= 1.000218999999e+06
 T451_R_f18_k4: const_2nd_Flr_SF - 1e+06 t451_leaf4 >= -999240
 T451_R_f272_k5: const_Mo_Sold_2 - 1e+06 t451_leaf5 >= -999999
 T451_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t451_leaf5 >= -999781
 T452_onehot: t452_leaf0 + t452_leaf1 + t452_leaf2 + t452_leaf3
   + t452_leaf4 = 1
 T452_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t452_leaf0 <= 1.000000999999e+06
 T452_L_f86_k0: const_Land_Slope_Gtl + 1e+06 t452_leaf0
   <= 1.000000999999e+06
 T452_L_f177_k1: x_ex2_is_AsbShng + 1e+06 t452_leaf1 <= 1.000000999999e+06
 T452_R_f86_k1: const_Land_Slope_Gtl - 1e+06 t452_leaf1 >= -999999
 T452_L_f5_k1: const_Year_Built + 1e+06 t452_leaf1 <= 1.001910999999e+06
 T452_L_f177_k2: x_ex2_is_AsbShng + 1e+06 t452_leaf2 <= 1.000000999999e+06
 T452_R_f86_k2: const_Land_Slope_Gtl - 1e+06 t452_leaf2 >= -999999
 T452_R_f5_k2: const_Year_Built - 1e+06 t452_leaf2 >= -998089
 T452_R_f177_k3: x_ex2_is_AsbShng - 1e+06 t452_leaf3 >= -999999
 T452_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t452_leaf3 <= 1.000175999999e+06
 T452_R_f177_k4: x_ex2_is_AsbShng - 1e+06 t452_leaf4 >= -999999
 T452_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t452_leaf4 >= -999824
 T453_onehot: t453_leaf0 + t453_leaf1 + t453_leaf2 + t453_leaf3
   + t453_leaf4 + t453_leaf5 = 1
 T453_L_f272_k0: const_Mo_Sold_2 + 1e+06 t453_leaf0 <= 1.000000999999e+06
 T453_L_f1_k0: const_Lot_Area + 1e+06 t453_leaf0 <= 1.005999999999e+06
 T453_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t453_leaf0 <= 1.000594999999e+06
 T453_L_f272_k1: const_Mo_Sold_2 + 1e+06 t453_leaf1 <= 1.000000999999e+06
 T453_L_f1_k1: const_Lot_Area + 1e+06 t453_leaf1 <= 1.005999999999e+06
 T453_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t453_leaf1 >= -999405
 T453_L_f272_k2: const_Mo_Sold_2 + 1e+06 t453_leaf2 <= 1.000000999999e+06
 T453_R_f1_k2: const_Lot_Area - 1e+06 t453_leaf2 >= -994000
 T453_L_f3_k2: Overall_Qual_calc + 1e+06 t453_leaf2 <= 1.000004999999e+06
 T453_L_f272_k3: const_Mo_Sold_2 + 1e+06 t453_leaf3 <= 1.000000999999e+06
 T453_R_f1_k3: const_Lot_Area - 1e+06 t453_leaf3 >= -994000
 T453_R_f3_k3: Overall_Qual_calc - 1e+06 t453_leaf3 >= -999995
 T453_R_f272_k4: const_Mo_Sold_2 - 1e+06 t453_leaf4 >= -999999
 T453_L_f297_k4: const_Sale_Condition_Normal + 1e+06 t453_leaf4
   <= 1.000000999999e+06
 T453_R_f272_k5: const_Mo_Sold_2 - 1e+06 t453_leaf5 >= -999999
 T453_R_f297_k5: const_Sale_Condition_Normal - 1e+06 t453_leaf5 >= -999999
 T454_onehot: t454_leaf0 + t454_leaf1 = 1
 T454_L_f194_k0: x_mvt_is_BrkCmn + 1e+06 t454_leaf0 <= 1.000000999999e+06
 T454_R_f194_k1: x_mvt_is_BrkCmn - 1e+06 t454_leaf1 >= -999999
 T455_onehot: t455_leaf0 = 1
 T456_onehot: t456_leaf0 = 1
 T457_onehot: t457_leaf0 + t457_leaf1 + t457_leaf2 = 1
 T457_L_f296_k0: const_Sale_Condition_Family + 1e+06 t457_leaf0
   <= 1.000000999999e+06
 T457_R_f296_k1: const_Sale_Condition_Family - 1e+06 t457_leaf1 >= -999999
 T457_L_f81_k1: const_Lot_Config_Corner + 1e+06 t457_leaf1
   <= 1.000000999999e+06
 T457_R_f296_k2: const_Sale_Condition_Family - 1e+06 t457_leaf2 >= -999999
 T457_R_f81_k2: const_Lot_Config_Corner - 1e+06 t457_leaf2 >= -999999
 T458_onehot: t458_leaf0 + t458_leaf1 + t458_leaf2 + t458_leaf3 = 1
 T458_L_f1_k0: const_Lot_Area + 1e+06 t458_leaf0 <= 1.006441999999e+06
 T458_R_f1_k1: const_Lot_Area - 1e+06 t458_leaf1 >= -993558
 T458_L_f38_k1: x_Enclosed_Porch + 1e+06 t458_leaf1 <= 1.000253999999e+06
 T458_L_f38_k1: x_Enclosed_Porch + 1e+06 t458_leaf1 <= 1.000235999999e+06
 T458_R_f1_k2: const_Lot_Area - 1e+06 t458_leaf2 >= -993558
 T458_L_f38_k2: x_Enclosed_Porch + 1e+06 t458_leaf2 <= 1.000253999999e+06
 T458_R_f38_k2: x_Enclosed_Porch - 1e+06 t458_leaf2 >= -999764
 T458_R_f1_k3: const_Lot_Area - 1e+06 t458_leaf3 >= -993558
 T458_R_f38_k3: x_Enclosed_Porch - 1e+06 t458_leaf3 >= -999746
 T459_onehot: t459_leaf0 + t459_leaf1 = 1
 T459_L_f44_k0: const_Yr_Sold + 1e+06 t459_leaf0 <= 1.002006999999e+06
 T459_R_f44_k1: const_Yr_Sold - 1e+06 t459_leaf1 >= -997993
 T460_onehot: t460_leaf0 = 1
 T461_onehot: t461_leaf0 + t461_leaf1 + t461_leaf2 + t461_leaf3
   + t461_leaf4 = 1
 T461_L_f3_k0: Overall_Qual_calc + 1e+06 t461_leaf0 <= 1.000002999999e+06
 T461_R_f3_k1: Overall_Qual_calc - 1e+06 t461_leaf1 >= -999997
 T461_L_f1_k1: const_Lot_Area + 1e+06 t461_leaf1 <= 1.016778999999e+06
 T461_L_f38_k1: x_Enclosed_Porch + 1e+06 t461_leaf1 <= 1.000230999999e+06
 T461_R_f3_k2: Overall_Qual_calc - 1e+06 t461_leaf2 >= -999997
 T461_L_f1_k2: const_Lot_Area + 1e+06 t461_leaf2 <= 1.016778999999e+06
 T461_R_f38_k2: x_Enclosed_Porch - 1e+06 t461_leaf2 >= -999769
 T461_R_f3_k3: Overall_Qual_calc - 1e+06 t461_leaf3 >= -999997
 T461_R_f1_k3: const_Lot_Area - 1e+06 t461_leaf3 >= -983221
 T461_L_f267_k3: const_Misc_Feature_No_aplica + 1e+06 t461_leaf3
   <= 1.000000999999e+06
 T461_R_f3_k4: Overall_Qual_calc - 1e+06 t461_leaf4 >= -999997
 T461_R_f1_k4: const_Lot_Area - 1e+06 t461_leaf4 >= -983221
 T461_R_f267_k4: const_Misc_Feature_No_aplica - 1e+06 t461_leaf4
   >= -999999
 T462_onehot: t462_leaf0 + t462_leaf1 = 1
 T462_L_f0_k0: const_Lot_Frontage + 1e+06 t462_leaf0 <= 1.000139999999e+06
 T462_R_f0_k1: const_Lot_Frontage - 1e+06 t462_leaf1 >= -999860
 T463_onehot: t463_leaf0 + t463_leaf1 + t463_leaf2 + t463_leaf3
   + t463_leaf4 + t463_leaf5 + t463_leaf6 = 1
 T463_L_f20_k0: x_Gr_Liv_Area + 1e+06 t463_leaf0 <= 1.000893999999e+06
 T463_L_f1_k0: const_Lot_Area + 1e+06 t463_leaf0 <= 1.010181999999e+06
 T463_L_f36_k0: x_Wood_Deck_SF + 1e+06 t463_leaf0 <= 1.000021999999e+06
 T463_L_f20_k1: x_Gr_Liv_Area + 1e+06 t463_leaf1 <= 1.000893999999e+06
 T463_L_f1_k1: const_Lot_Area + 1e+06 t463_leaf1 <= 1.010181999999e+06
 T463_R_f36_k1: x_Wood_Deck_SF - 1e+06 t463_leaf1 >= -999978
 T463_L_f20_k2: x_Gr_Liv_Area + 1e+06 t463_leaf2 <= 1.000893999999e+06
 T463_R_f1_k2: const_Lot_Area - 1e+06 t463_leaf2 >= -989818
 T463_R_f20_k3: x_Gr_Liv_Area - 1e+06 t463_leaf3 >= -999106
 T463_L_f1_k3: const_Lot_Area + 1e+06 t463_leaf3 <= 1.021369999999e+06
 T463_L_f248_k3: const_Garage_Type_Basment + 1e+06 t463_leaf3
   <= 1.000000999999e+06
 T463_R_f20_k4: x_Gr_Liv_Area - 1e+06 t463_leaf4 >= -999106
 T463_L_f1_k4: const_Lot_Area + 1e+06 t463_leaf4 <= 1.021369999999e+06
 T463_R_f248_k4: const_Garage_Type_Basment - 1e+06 t463_leaf4 >= -999999
 T463_R_f20_k5: x_Gr_Liv_Area - 1e+06 t463_leaf5 >= -999106
 T463_R_f1_k5: const_Lot_Area - 1e+06 t463_leaf5 >= -978630
 T463_L_f33_k5: x_Garage_Area + 1e+06 t463_leaf5 <= 1.000842999999e+06
 T463_R_f20_k6: x_Gr_Liv_Area - 1e+06 t463_leaf6 >= -999106
 T463_R_f1_k6: const_Lot_Area - 1e+06 t463_leaf6 >= -978630
 T463_R_f33_k6: x_Garage_Area - 1e+06 t463_leaf6 >= -999157
 T464_onehot: t464_leaf0 + t464_leaf1 + t464_leaf2 = 1
 T464_L_f202_k0: const_Foundation_PConc + 1e+06 t464_leaf0
   <= 1.000000999999e+06
 T464_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t464_leaf0 <= 1.000000999999e+06
 T464_L_f202_k1: const_Foundation_PConc + 1e+06 t464_leaf1
   <= 1.000000999999e+06
 T464_R_f164_k1: x_ex1_is_BrkFace - 1e+06 t464_leaf1 >= -999999
 T464_R_f202_k2: const_Foundation_PConc - 1e+06 t464_leaf2 >= -999999
 T465_onehot: t465_leaf0 + t465_leaf1 + t465_leaf2 + t465_leaf3 = 1
 T465_L_f3_k0: Overall_Qual_calc + 1e+06 t465_leaf0 <= 1.000003999999e+06
 T465_L_f17_k0: x_1st_Flr_SF + 1e+06 t465_leaf0 <= 1.000812999999e+06
 T465_L_f3_k1: Overall_Qual_calc + 1e+06 t465_leaf1 <= 1.000003999999e+06
 T465_R_f17_k1: x_1st_Flr_SF - 1e+06 t465_leaf1 >= -999187
 T465_R_f3_k2: Overall_Qual_calc - 1e+06 t465_leaf2 >= -999996
 T465_L_f37_k2: x_Open_Porch_SF + 1e+06 t465_leaf2 <= 1.000188999999e+06
 T465_R_f3_k3: Overall_Qual_calc - 1e+06 t465_leaf3 >= -999996
 T465_R_f37_k3: x_Open_Porch_SF - 1e+06 t465_leaf3 >= -999811
 T466_onehot: t466_leaf0 + t466_leaf1 + t466_leaf2 + t466_leaf3
   + t466_leaf4 = 1
 T466_L_f3_k0: Overall_Qual_calc + 1e+06 t466_leaf0 <= 1.000004999999e+06
 T466_L_f1_k0: const_Lot_Area + 1e+06 t466_leaf0 <= 1.004269999999e+06
 T466_L_f3_k1: Overall_Qual_calc + 1e+06 t466_leaf1 <= 1.000004999999e+06
 T466_R_f1_k1: const_Lot_Area - 1e+06 t466_leaf1 >= -995730
 T466_L_f33_k1: x_Garage_Area + 1e+06 t466_leaf1 <= 1.000499999999e+06
 T466_L_f3_k2: Overall_Qual_calc + 1e+06 t466_leaf2 <= 1.000004999999e+06
 T466_R_f1_k2: const_Lot_Area - 1e+06 t466_leaf2 >= -995730
 T466_R_f33_k2: x_Garage_Area - 1e+06 t466_leaf2 >= -999500
 T466_R_f3_k3: Overall_Qual_calc - 1e+06 t466_leaf3 >= -999995
 T466_L_f218_k3: x_b2_is_ALQ + 1e+06 t466_leaf3 <= 1.000000999999e+06
 T466_R_f3_k4: Overall_Qual_calc - 1e+06 t466_leaf4 >= -999995
 T466_R_f218_k4: x_b2_is_ALQ - 1e+06 t466_leaf4 >= -999999
 T467_onehot: t467_leaf0 = 1
 T468_onehot: t468_leaf0 = 1
 T469_onehot: t469_leaf0 = 1
 T470_onehot: t470_leaf0 + t470_leaf1 + t470_leaf2 = 1
 T470_L_f8_k0: x_Exter_Qual + 1e+06 t470_leaf0 <= 1.000001999999e+06
 T470_R_f8_k1: x_Exter_Qual - 1e+06 t470_leaf1 >= -999998
 T470_L_f17_k1: x_1st_Flr_SF + 1e+06 t470_leaf1 <= 1.002199999999e+06
 T470_R_f8_k2: x_Exter_Qual - 1e+06 t470_leaf2 >= -999998
 T470_R_f17_k2: x_1st_Flr_SF - 1e+06 t470_leaf2 >= -997800
 T471_onehot: t471_leaf0 + t471_leaf1 + t471_leaf2 + t471_leaf3 = 1
 T471_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t471_leaf0
   <= 1.000000999999e+06
 T471_R_f96_k1: const_Neighborhood_Edwards - 1e+06 t471_leaf1 >= -999999
 T471_L_f17_k1: x_1st_Flr_SF + 1e+06 t471_leaf1 <= 1.000985999999e+06
 T471_R_f96_k2: const_Neighborhood_Edwards - 1e+06 t471_leaf2 >= -999999
 T471_R_f17_k2: x_1st_Flr_SF - 1e+06 t471_leaf2 >= -999014
 T471_L_f17_k2: x_1st_Flr_SF + 1e+06 t471_leaf2 <= 1.001517999999e+06
 T471_R_f96_k3: const_Neighborhood_Edwards - 1e+06 t471_leaf3 >= -999999
 T471_R_f17_k3: x_1st_Flr_SF - 1e+06 t471_leaf3 >= -999014
 T471_R_f17_k3: x_1st_Flr_SF - 1e+06 t471_leaf3 >= -998482
 T472_onehot: t472_leaf0 = 1
 T473_onehot: t473_leaf0 + t473_leaf1 + t473_leaf2 = 1
 T473_L_f199_k0: x_mvt_is_Stone + 1e+06 t473_leaf0 <= 1.000000999999e+06
 T473_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t473_leaf0 <= 1.001392999999e+06
 T473_L_f199_k1: x_mvt_is_Stone + 1e+06 t473_leaf1 <= 1.000000999999e+06
 T473_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t473_leaf1 >= -998607
 T473_R_f199_k2: x_mvt_is_Stone - 1e+06 t473_leaf2 >= -999999
 T474_onehot: t474_leaf0 + t474_leaf1 + t474_leaf2 = 1
 T474_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t474_leaf0
   <= 1.001823999999e+06
 T474_L_f3_k0: Overall_Qual_calc + 1e+06 t474_leaf0 <= 1.000003999999e+06
 T474_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t474_leaf1
   <= 1.001823999999e+06
 T474_R_f3_k1: Overall_Qual_calc - 1e+06 t474_leaf1 >= -999996
 T474_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t474_leaf2 >= -998176
 T475_onehot: t475_leaf0 + t475_leaf1 + t475_leaf2 = 1
 T475_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t475_leaf0
   <= 1.000480999999e+06
 T475_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t475_leaf0
   <= 1.000244999999e+06
 T475_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t475_leaf1
   <= 1.000480999999e+06
 T475_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t475_leaf1 >= -999755
 T475_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t475_leaf2 >= -999519
 T476_onehot: t476_leaf0 + t476_leaf1 + t476_leaf2 + t476_leaf3 = 1
 T476_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t476_leaf0
   <= 1.000000999999e+06
 T476_L_f38_k0: x_Enclosed_Porch + 1e+06 t476_leaf0 <= 1.000090999999e+06
 T476_L_f4_k0: const_Overall_Cond + 1e+06 t476_leaf0 <= 1.000005999999e+06
 T476_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t476_leaf1
   <= 1.000000999999e+06
 T476_L_f38_k1: x_Enclosed_Porch + 1e+06 t476_leaf1 <= 1.000090999999e+06
 T476_R_f4_k1: const_Overall_Cond - 1e+06 t476_leaf1 >= -999994
 T476_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t476_leaf2
   <= 1.000000999999e+06
 T476_R_f38_k2: x_Enclosed_Porch - 1e+06 t476_leaf2 >= -999909
 T476_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t476_leaf3 >= -999999
 T477_onehot: t477_leaf0 + t477_leaf1 + t477_leaf2 + t477_leaf3 = 1
 T477_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t477_leaf0 <= 1.000575999999e+06
 T477_L_f237_k0: x_elect_is_SBrkr + 1e+06 t477_leaf0 <= 1.000000999999e+06
 T477_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t477_leaf1 <= 1.000575999999e+06
 T477_R_f237_k1: x_elect_is_SBrkr - 1e+06 t477_leaf1 >= -999999
 T477_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t477_leaf2 >= -999424
 T477_L_f18_k2: const_2nd_Flr_SF + 1e+06 t477_leaf2 <= 1.000580999999e+06
 T477_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t477_leaf3 >= -999424
 T477_R_f18_k3: const_2nd_Flr_SF - 1e+06 t477_leaf3 >= -999419
 T478_onehot: t478_leaf0 + t478_leaf1 + t478_leaf2 + t478_leaf3 = 1
 T478_L_f38_k0: x_Enclosed_Porch + 1e+06 t478_leaf0 <= 1.000153999999e+06
 T478_R_f38_k1: x_Enclosed_Porch - 1e+06 t478_leaf1 >= -999846
 T478_L_f18_k1: const_2nd_Flr_SF + 1e+06 t478_leaf1 <= 1.000307999999e+06
 T478_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t478_leaf1 <= 1.000027999999e+06
 T478_R_f38_k2: x_Enclosed_Porch - 1e+06 t478_leaf2 >= -999846
 T478_L_f18_k2: const_2nd_Flr_SF + 1e+06 t478_leaf2 <= 1.000307999999e+06
 T478_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t478_leaf2 >= -999972
 T478_R_f38_k3: x_Enclosed_Porch - 1e+06 t478_leaf3 >= -999846
 T478_R_f18_k3: const_2nd_Flr_SF - 1e+06 t478_leaf3 >= -999692
 T479_onehot: t479_leaf0 + t479_leaf1 + t479_leaf2 + t479_leaf3
   + t479_leaf4 + t479_leaf5 = 1
 T479_L_f96_k0: const_Neighborhood_Edwards + 1e+06 t479_leaf0
   <= 1.000000999999e+06
 T479_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t479_leaf0 <= 1.000000999999e+06
 T479_L_f20_k0: x_Gr_Liv_Area + 1e+06 t479_leaf0 <= 1.000893999999e+06
 T479_L_f96_k1: const_Neighborhood_Edwards + 1e+06 t479_leaf1
   <= 1.000000999999e+06
 T479_L_f177_k1: x_ex2_is_AsbShng + 1e+06 t479_leaf1 <= 1.000000999999e+06
 T479_R_f20_k1: x_Gr_Liv_Area - 1e+06 t479_leaf1 >= -999106
 T479_L_f96_k2: const_Neighborhood_Edwards + 1e+06 t479_leaf2
   <= 1.000000999999e+06
 T479_R_f177_k2: x_ex2_is_AsbShng - 1e+06 t479_leaf2 >= -999999
 T479_R_f96_k3: const_Neighborhood_Edwards - 1e+06 t479_leaf3 >= -999999
 T479_L_f33_k3: x_Garage_Area + 1e+06 t479_leaf3 <= 1.000609999999e+06
 T479_L_f73_k3: const_Lot_Shape_IR1 + 1e+06 t479_leaf3
   <= 1.000000999999e+06
 T479_R_f96_k4: const_Neighborhood_Edwards - 1e+06 t479_leaf4 >= -999999
 T479_L_f33_k4: x_Garage_Area + 1e+06 t479_leaf4 <= 1.000609999999e+06
 T479_R_f73_k4: const_Lot_Shape_IR1 - 1e+06 t479_leaf4 >= -999999
 T479_R_f96_k5: const_Neighborhood_Edwards - 1e+06 t479_leaf5 >= -999999
 T479_R_f33_k5: x_Garage_Area - 1e+06 t479_leaf5 >= -999390
 T480_onehot: t480_leaf0 + t480_leaf1 + t480_leaf2 + t480_leaf3
   + t480_leaf4 = 1
 T480_L_f114_k0: const_Neighborhood_StoneBr + 1e+06 t480_leaf0
   <= 1.000000999999e+06
 T480_L_f37_k0: x_Open_Porch_SF + 1e+06 t480_leaf0 <= 1.000167999999e+06
 T480_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t480_leaf0 <= 1.001372999999e+06
 T480_L_f114_k1: const_Neighborhood_StoneBr + 1e+06 t480_leaf1
   <= 1.000000999999e+06
 T480_L_f37_k1: x_Open_Porch_SF + 1e+06 t480_leaf1 <= 1.000167999999e+06
 T480_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t480_leaf1 >= -998627
 T480_L_f114_k2: const_Neighborhood_StoneBr + 1e+06 t480_leaf2
   <= 1.000000999999e+06
 T480_R_f37_k2: x_Open_Porch_SF - 1e+06 t480_leaf2 >= -999832
 T480_L_f28_k2: x_TotRms_AbvGrd + 1e+06 t480_leaf2 <= 1.000008999999e+06
 T480_L_f114_k3: const_Neighborhood_StoneBr + 1e+06 t480_leaf3
   <= 1.000000999999e+06
 T480_R_f37_k3: x_Open_Porch_SF - 1e+06 t480_leaf3 >= -999832
 T480_R_f28_k3: x_TotRms_AbvGrd - 1e+06 t480_leaf3 >= -999991
 T480_R_f114_k4: const_Neighborhood_StoneBr - 1e+06 t480_leaf4 >= -999999
 T481_onehot: t481_leaf0 + t481_leaf1 = 1
 T481_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t481_leaf0
   <= 1.000000999999e+06
 T481_R_f245_k1: const_FunctioNo_aplical_Typ - 1e+06 t481_leaf1 >= -999999
 T482_onehot: t482_leaf0 = 1
 T483_onehot: t483_leaf0 = 1
 T484_onehot: t484_leaf0 + t484_leaf1 + t484_leaf2 + t484_leaf3 = 1
 T484_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t484_leaf0
   <= 1.002491999999e+06
 T484_L_f250_k0: const_Garage_Type_CarPort + 1e+06 t484_leaf0
   <= 1.000000999999e+06
 T484_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t484_leaf0
   <= 1.000001999999e+06
 T484_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t484_leaf1
   <= 1.002491999999e+06
 T484_L_f250_k1: const_Garage_Type_CarPort + 1e+06 t484_leaf1
   <= 1.000000999999e+06
 T484_R_f21_k1: const_Bsmt_Full_Bath - 1e+06 t484_leaf1 >= -999998
 T484_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t484_leaf2
   <= 1.002491999999e+06
 T484_R_f250_k2: const_Garage_Type_CarPort - 1e+06 t484_leaf2 >= -999999
 T484_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t484_leaf3 >= -997508
 T485_onehot: t485_leaf0 = 1
 T486_onehot: t486_leaf0 = 1
 T487_onehot: t487_leaf0 = 1
 T488_onehot: t488_leaf0 + t488_leaf1 + t488_leaf2 + t488_leaf3 = 1
 T488_L_f33_k0: x_Garage_Area + 1e+06 t488_leaf0 <= 1.000307999999e+06
 T488_L_f297_k0: const_Sale_Condition_Normal + 1e+06 t488_leaf0
   <= 1.000000999999e+06
 T488_L_f33_k1: x_Garage_Area + 1e+06 t488_leaf1 <= 1.000307999999e+06
 T488_R_f297_k1: const_Sale_Condition_Normal - 1e+06 t488_leaf1 >= -999999
 T488_L_f3_k1: Overall_Qual_calc + 1e+06 t488_leaf1 <= 1.000002999999e+06
 T488_L_f33_k2: x_Garage_Area + 1e+06 t488_leaf2 <= 1.000307999999e+06
 T488_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t488_leaf2 >= -999999
 T488_R_f3_k2: Overall_Qual_calc - 1e+06 t488_leaf2 >= -999997
 T488_R_f33_k3: x_Garage_Area - 1e+06 t488_leaf3 >= -999692
 T489_onehot: t489_leaf0 = 1
 T490_onehot: t490_leaf0 + t490_leaf1 + t490_leaf2 = 1
 T490_L_f37_k0: x_Open_Porch_SF + 1e+06 t490_leaf0 <= 1.000163999999e+06
 T490_L_f258_k0: x_paved_drive_is_P + 1e+06 t490_leaf0
   <= 1.000000999999e+06
 T490_L_f37_k1: x_Open_Porch_SF + 1e+06 t490_leaf1 <= 1.000163999999e+06
 T490_R_f258_k1: x_paved_drive_is_P - 1e+06 t490_leaf1 >= -999999
 T490_R_f37_k2: x_Open_Porch_SF - 1e+06 t490_leaf2 >= -999836
 T491_onehot: t491_leaf0 + t491_leaf1 + t491_leaf2 + t491_leaf3 = 1
 T491_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t491_leaf0 <= 1.000475999999e+06
 T491_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t491_leaf0 <= 1.000002999999e+06
 T491_L_f1_k0: const_Lot_Area + 1e+06 t491_leaf0 <= 1.005435999999e+06
 T491_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t491_leaf1 <= 1.000475999999e+06
 T491_L_f25_k1: x_Bedroom_AbvGr + 1e+06 t491_leaf1 <= 1.000002999999e+06
 T491_R_f1_k1: const_Lot_Area - 1e+06 t491_leaf1 >= -994564
 T491_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t491_leaf2 <= 1.000475999999e+06
 T491_R_f25_k2: x_Bedroom_AbvGr - 1e+06 t491_leaf2 >= -999997
 T491_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t491_leaf3 >= -999524
 T492_onehot: t492_leaf0 + t492_leaf1 + t492_leaf2 = 1
 T492_L_f93_k0: const_Neighborhood_ClearCr + 1e+06 t492_leaf0
   <= 1.000000999999e+06
 T492_L_f38_k0: x_Enclosed_Porch + 1e+06 t492_leaf0 <= 1.000264999999e+06
 T492_L_f93_k1: const_Neighborhood_ClearCr + 1e+06 t492_leaf1
   <= 1.000000999999e+06
 T492_R_f38_k1: x_Enclosed_Porch - 1e+06 t492_leaf1 >= -999735
 T492_R_f93_k2: const_Neighborhood_ClearCr - 1e+06 t492_leaf2 >= -999999
 T493_onehot: t493_leaf0 + t493_leaf1 + t493_leaf2 + t493_leaf3
   + t493_leaf4 = 1
 T493_L_f26_k0: x_Kitchen_AbvGr + 1e+06 t493_leaf0 <= 1.000001999999e+06
 T493_L_f33_k0: x_Garage_Area + 1e+06 t493_leaf0 <= 1.000251999999e+06
 T493_L_f76_k0: const_Lot_Shape_Reg + 1e+06 t493_leaf0
   <= 1.000000999999e+06
 T493_L_f26_k1: x_Kitchen_AbvGr + 1e+06 t493_leaf1 <= 1.000001999999e+06
 T493_L_f33_k1: x_Garage_Area + 1e+06 t493_leaf1 <= 1.000251999999e+06
 T493_R_f76_k1: const_Lot_Shape_Reg - 1e+06 t493_leaf1 >= -999999
 T493_L_f26_k2: x_Kitchen_AbvGr + 1e+06 t493_leaf2 <= 1.000001999999e+06
 T493_R_f33_k2: x_Garage_Area - 1e+06 t493_leaf2 >= -999748
 T493_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t493_leaf2
   <= 1.000000999999e+06
 T493_L_f26_k3: x_Kitchen_AbvGr + 1e+06 t493_leaf3 <= 1.000001999999e+06
 T493_R_f33_k3: x_Garage_Area - 1e+06 t493_leaf3 >= -999748
 T493_R_f95_k3: const_Neighborhood_Crawfor - 1e+06 t493_leaf3 >= -999999
 T493_R_f26_k4: x_Kitchen_AbvGr - 1e+06 t493_leaf4 >= -999998
 T494_onehot: t494_leaf0 + t494_leaf1 + t494_leaf2 = 1
 T494_L_f150_k0: const_Roof_Style_Hip + 1e+06 t494_leaf0
   <= 1.000000999999e+06
 T494_R_f150_k1: const_Roof_Style_Hip - 1e+06 t494_leaf1 >= -999999
 T494_L_f77_k1: const_Land_Contour_Bnk + 1e+06 t494_leaf1
   <= 1.000000999999e+06
 T494_R_f150_k2: const_Roof_Style_Hip - 1e+06 t494_leaf2 >= -999999
 T494_R_f77_k2: const_Land_Contour_Bnk - 1e+06 t494_leaf2 >= -999999
 T495_onehot: t495_leaf0 + t495_leaf1 = 1
 T495_L_f3_k0: Overall_Qual_calc + 1e+06 t495_leaf0 <= 1.000002999999e+06
 T495_R_f3_k1: Overall_Qual_calc - 1e+06 t495_leaf1 >= -999997
 T496_onehot: t496_leaf0 + t496_leaf1 = 1
 T496_L_f3_k0: Overall_Qual_calc + 1e+06 t496_leaf0 <= 1.000002999999e+06
 T496_R_f3_k1: Overall_Qual_calc - 1e+06 t496_leaf1 >= -999997
 T497_onehot: t497_leaf0 + t497_leaf1 + t497_leaf2 + t497_leaf3 = 1
 T497_L_f0_k0: const_Lot_Frontage + 1e+06 t497_leaf0 <= 1.000116999999e+06
 T497_L_f11_k0: x_Bsmt_Cond + 1e+06 t497_leaf0 <= 1.000001999999e+06
 T497_L_f38_k0: x_Enclosed_Porch + 1e+06 t497_leaf0 <= 1.000104999999e+06
 T497_L_f0_k1: const_Lot_Frontage + 1e+06 t497_leaf1 <= 1.000116999999e+06
 T497_L_f11_k1: x_Bsmt_Cond + 1e+06 t497_leaf1 <= 1.000001999999e+06
 T497_R_f38_k1: x_Enclosed_Porch - 1e+06 t497_leaf1 >= -999895
 T497_L_f0_k2: const_Lot_Frontage + 1e+06 t497_leaf2 <= 1.000116999999e+06
 T497_R_f11_k2: x_Bsmt_Cond - 1e+06 t497_leaf2 >= -999998
 T497_R_f0_k3: const_Lot_Frontage - 1e+06 t497_leaf3 >= -999883
 T498_onehot: t498_leaf0 + t498_leaf1 + t498_leaf2 + t498_leaf3
   + t498_leaf4 = 1
 T498_L_f5_k0: const_Year_Built + 1e+06 t498_leaf0 <= 1.002004999999e+06
 T498_L_f3_k0: Overall_Qual_calc + 1e+06 t498_leaf0 <= 1.000006999999e+06
 T498_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t498_leaf0 <= 1.000007999999e+06
 T498_L_f5_k1: const_Year_Built + 1e+06 t498_leaf1 <= 1.002004999999e+06
 T498_L_f3_k1: Overall_Qual_calc + 1e+06 t498_leaf1 <= 1.000006999999e+06
 T498_R_f28_k1: x_TotRms_AbvGrd - 1e+06 t498_leaf1 >= -999992
 T498_L_f5_k2: const_Year_Built + 1e+06 t498_leaf2 <= 1.002004999999e+06
 T498_R_f3_k2: Overall_Qual_calc - 1e+06 t498_leaf2 >= -999993
 T498_L_f1_k2: const_Lot_Area + 1e+06 t498_leaf2 <= 1.025285999999e+06
 T498_L_f5_k3: const_Year_Built + 1e+06 t498_leaf3 <= 1.002004999999e+06
 T498_R_f3_k3: Overall_Qual_calc - 1e+06 t498_leaf3 >= -999993
 T498_R_f1_k3: const_Lot_Area - 1e+06 t498_leaf3 >= -974714
 T498_R_f5_k4: const_Year_Built - 1e+06 t498_leaf4 >= -997995
 T499_onehot: t499_leaf0 + t499_leaf1 = 1
 T499_L_f92_k0: const_Neighborhood_BrkSide + 1e+06 t499_leaf0
   <= 1.000000999999e+06
 T499_R_f92_k1: const_Neighborhood_BrkSide - 1e+06 t499_leaf1 >= -999999
 T500_onehot: t500_leaf0 + t500_leaf1 + t500_leaf2 + t500_leaf3 = 1
 T500_L_f33_k0: x_Garage_Area + 1e+06 t500_leaf0 <= 1.000887999999e+06
 T500_L_f79_k0: const_Land_Contour_Low + 1e+06 t500_leaf0
   <= 1.000000999999e+06
 T500_L_f33_k1: x_Garage_Area + 1e+06 t500_leaf1 <= 1.000887999999e+06
 T500_R_f79_k1: const_Land_Contour_Low - 1e+06 t500_leaf1 >= -999999
 T500_L_f170_k1: x_ex1_is_Plywood + 1e+06 t500_leaf1 <= 1.000000999999e+06
 T500_L_f33_k2: x_Garage_Area + 1e+06 t500_leaf2 <= 1.000887999999e+06
 T500_R_f79_k2: const_Land_Contour_Low - 1e+06 t500_leaf2 >= -999999
 T500_R_f170_k2: x_ex1_is_Plywood - 1e+06 t500_leaf2 >= -999999
 T500_R_f33_k3: x_Garage_Area - 1e+06 t500_leaf3 >= -999112
 T501_onehot: t501_leaf0 + t501_leaf1 + t501_leaf2 + t501_leaf3
   + t501_leaf4 + t501_leaf5 + t501_leaf6 = 1
 T501_L_f20_k0: x_Gr_Liv_Area + 1e+06 t501_leaf0 <= 1.000875999999e+06
 T501_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t501_leaf0
   <= 1.000000999999e+06
 T501_L_f76_k0: const_Lot_Shape_Reg + 1e+06 t501_leaf0
   <= 1.000000999999e+06
 T501_L_f20_k1: x_Gr_Liv_Area + 1e+06 t501_leaf1 <= 1.000875999999e+06
 T501_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t501_leaf1
   <= 1.000000999999e+06
 T501_R_f76_k1: const_Lot_Shape_Reg - 1e+06 t501_leaf1 >= -999999
 T501_L_f20_k2: x_Gr_Liv_Area + 1e+06 t501_leaf2 <= 1.000875999999e+06
 T501_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t501_leaf2
   >= -999999
 T501_R_f20_k3: x_Gr_Liv_Area - 1e+06 t501_leaf3 >= -999124
 T501_L_f33_k3: x_Garage_Area + 1e+06 t501_leaf3 <= 1.000643999999e+06
 T501_L_f250_k3: const_Garage_Type_CarPort + 1e+06 t501_leaf3
   <= 1.000000999999e+06
 T501_R_f20_k4: x_Gr_Liv_Area - 1e+06 t501_leaf4 >= -999124
 T501_L_f33_k4: x_Garage_Area + 1e+06 t501_leaf4 <= 1.000643999999e+06
 T501_R_f250_k4: const_Garage_Type_CarPort - 1e+06 t501_leaf4 >= -999999
 T501_R_f20_k5: x_Gr_Liv_Area - 1e+06 t501_leaf5 >= -999124
 T501_R_f33_k5: x_Garage_Area - 1e+06 t501_leaf5 >= -999356
 T501_L_f20_k5: x_Gr_Liv_Area + 1e+06 t501_leaf5 <= 1.001778999999e+06
 T501_R_f20_k6: x_Gr_Liv_Area - 1e+06 t501_leaf6 >= -999124
 T501_R_f33_k6: x_Garage_Area - 1e+06 t501_leaf6 >= -999356
 T501_R_f20_k6: x_Gr_Liv_Area - 1e+06 t501_leaf6 >= -998221
 T502_onehot: t502_leaf0 + t502_leaf1 = 1
 T502_L_f3_k0: Overall_Qual_calc + 1e+06 t502_leaf0 <= 1.000002999999e+06
 T502_R_f3_k1: Overall_Qual_calc - 1e+06 t502_leaf1 >= -999997
 T503_onehot: t503_leaf0 = 1
 T504_onehot: t504_leaf0 + t504_leaf1 + t504_leaf2 + t504_leaf3 = 1
 T504_L_f10_k0: const_Bsmt_Qual + 1e+06 t504_leaf0 <= 999999.999999
 T504_L_f0_k0: const_Lot_Frontage + 1e+06 t504_leaf0 <= 1.000065999999e+06
 T504_L_f10_k1: const_Bsmt_Qual + 1e+06 t504_leaf1 <= 999999.999999
 T504_R_f0_k1: const_Lot_Frontage - 1e+06 t504_leaf1 >= -999934
 T504_L_f276_k1: const_Mo_Sold_6 + 1e+06 t504_leaf1 <= 1.000000999999e+06
 T504_L_f10_k2: const_Bsmt_Qual + 1e+06 t504_leaf2 <= 999999.999999
 T504_R_f0_k2: const_Lot_Frontage - 1e+06 t504_leaf2 >= -999934
 T504_R_f276_k2: const_Mo_Sold_6 - 1e+06 t504_leaf2 >= -999999
 T504_R_f10_k3: const_Bsmt_Qual - 1e+06 t504_leaf3 >= -1e+06
 T505_onehot: t505_leaf0 + t505_leaf1 = 1
 T505_L_f37_k0: x_Open_Porch_SF + 1e+06 t505_leaf0 <= 1.000092999999e+06
 T505_R_f37_k1: x_Open_Porch_SF - 1e+06 t505_leaf1 >= -999907
 T506_onehot: t506_leaf0 + t506_leaf1 + t506_leaf2 + t506_leaf3
   + t506_leaf4 = 1
 T506_L_f1_k0: const_Lot_Area + 1e+06 t506_leaf0 <= 1.016778999999e+06
 T506_L_f6_k0: const_Year_Remod_Add + 1e+06 t506_leaf0
   <= 1.002007999999e+06
 T506_L_f20_k0: x_Gr_Liv_Area + 1e+06 t506_leaf0 <= 1.002632999999e+06
 T506_L_f1_k1: const_Lot_Area + 1e+06 t506_leaf1 <= 1.016778999999e+06
 T506_L_f6_k1: const_Year_Remod_Add + 1e+06 t506_leaf1
   <= 1.002007999999e+06
 T506_R_f20_k1: x_Gr_Liv_Area - 1e+06 t506_leaf1 >= -997367
 T506_L_f1_k2: const_Lot_Area + 1e+06 t506_leaf2 <= 1.016778999999e+06
 T506_R_f6_k2: const_Year_Remod_Add - 1e+06 t506_leaf2 >= -997992
 T506_R_f1_k3: const_Lot_Area - 1e+06 t506_leaf3 >= -983221
 T506_L_f17_k3: x_1st_Flr_SF + 1e+06 t506_leaf3 <= 1.001401999999e+06
 T506_R_f1_k4: const_Lot_Area - 1e+06 t506_leaf4 >= -983221
 T506_R_f17_k4: x_1st_Flr_SF - 1e+06 t506_leaf4 >= -998598
 T507_onehot: t507_leaf0 + t507_leaf1 = 1
 T507_L_f13_k0: x_BsmtFin_SF_2 + 1e+06 t507_leaf0 <= 1.000059999999e+06
 T507_R_f13_k1: x_BsmtFin_SF_2 - 1e+06 t507_leaf1 >= -999940
 T508_onehot: t508_leaf0 + t508_leaf1 + t508_leaf2 + t508_leaf3
   + t508_leaf4 + t508_leaf5 = 1
 T508_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t508_leaf0 <= 1.000000999999e+06
 T508_L_f148_k0: const_Roof_Style_Gable + 1e+06 t508_leaf0
   <= 1.000000999999e+06
 T508_L_f44_k0: const_Yr_Sold + 1e+06 t508_leaf0 <= 1.002006999999e+06
 T508_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t508_leaf1 <= 1.000000999999e+06
 T508_L_f148_k1: const_Roof_Style_Gable + 1e+06 t508_leaf1
   <= 1.000000999999e+06
 T508_R_f44_k1: const_Yr_Sold - 1e+06 t508_leaf1 >= -997993
 T508_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t508_leaf2 <= 1.000000999999e+06
 T508_R_f148_k2: const_Roof_Style_Gable - 1e+06 t508_leaf2 >= -999999
 T508_L_f273_k2: const_Mo_Sold_3 + 1e+06 t508_leaf2 <= 1.000000999999e+06
 T508_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t508_leaf3 <= 1.000000999999e+06
 T508_R_f148_k3: const_Roof_Style_Gable - 1e+06 t508_leaf3 >= -999999
 T508_R_f273_k3: const_Mo_Sold_3 - 1e+06 t508_leaf3 >= -999999
 T508_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t508_leaf4 >= -999999
 T508_L_f33_k4: x_Garage_Area + 1e+06 t508_leaf4 <= 1.000350999999e+06
 T508_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t508_leaf5 >= -999999
 T508_R_f33_k5: x_Garage_Area - 1e+06 t508_leaf5 >= -999649
 T509_onehot: t509_leaf0 + t509_leaf1 + t509_leaf2 = 1
 T509_L_f18_k0: const_2nd_Flr_SF + 1e+06 t509_leaf0 <= 1.000796999999e+06
 T509_L_f44_k0: const_Yr_Sold + 1e+06 t509_leaf0 <= 1.002009999999e+06
 T509_L_f18_k1: const_2nd_Flr_SF + 1e+06 t509_leaf1 <= 1.000796999999e+06
 T509_R_f44_k1: const_Yr_Sold - 1e+06 t509_leaf1 >= -997990
 T509_R_f18_k2: const_2nd_Flr_SF - 1e+06 t509_leaf2 >= -999203
 T510_onehot: t510_leaf0 = 1
 T511_onehot: t511_leaf0 = 1
 T512_onehot: t512_leaf0 = 1
 T513_onehot: t513_leaf0 + t513_leaf1 + t513_leaf2 + t513_leaf3
   + t513_leaf4 + t513_leaf5 = 1
 T513_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t513_leaf0 <= 1.000787999999e+06
 T513_L_f4_k0: const_Overall_Cond + 1e+06 t513_leaf0 <= 1.000004999999e+06
 T513_L_f30_k0: x_Fireplace_Qu + 1e+06 t513_leaf0 <= 1.000000999999e+06
 T513_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t513_leaf1 <= 1.000787999999e+06
 T513_L_f4_k1: const_Overall_Cond + 1e+06 t513_leaf1 <= 1.000004999999e+06
 T513_R_f30_k1: x_Fireplace_Qu - 1e+06 t513_leaf1 >= -999999
 T513_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t513_leaf2 <= 1.000787999999e+06
 T513_R_f4_k2: const_Overall_Cond - 1e+06 t513_leaf2 >= -999995
 T513_L_f164_k2: x_ex1_is_BrkFace + 1e+06 t513_leaf2 <= 1.000000999999e+06
 T513_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t513_leaf3 <= 1.000787999999e+06
 T513_R_f4_k3: const_Overall_Cond - 1e+06 t513_leaf3 >= -999995
 T513_R_f164_k3: x_ex1_is_BrkFace - 1e+06 t513_leaf3 >= -999999
 T513_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t513_leaf4 >= -999212
 T513_L_f1_k4: const_Lot_Area + 1e+06 t513_leaf4 <= 1.014114999999e+06
 T513_R_f7_k5: x_Mas_Vnr_Area - 1e+06 t513_leaf5 >= -999212
 T513_R_f1_k5: const_Lot_Area - 1e+06 t513_leaf5 >= -985885
 T514_onehot: t514_leaf0 = 1
 T515_onehot: t515_leaf0 + t515_leaf1 + t515_leaf2 + t515_leaf3 = 1
 T515_L_f214_k0: x_b1_is_LwQ + 1e+06 t515_leaf0 <= 1.000000999999e+06
 T515_L_f6_k0: const_Year_Remod_Add + 1e+06 t515_leaf0
   <= 1.001953999999e+06
 T515_L_f214_k1: x_b1_is_LwQ + 1e+06 t515_leaf1 <= 1.000000999999e+06
 T515_R_f6_k1: const_Year_Remod_Add - 1e+06 t515_leaf1 >= -998046
 T515_L_f4_k1: const_Overall_Cond + 1e+06 t515_leaf1 <= 1.000003999999e+06
 T515_L_f214_k2: x_b1_is_LwQ + 1e+06 t515_leaf2 <= 1.000000999999e+06
 T515_R_f6_k2: const_Year_Remod_Add - 1e+06 t515_leaf2 >= -998046
 T515_R_f4_k2: const_Overall_Cond - 1e+06 t515_leaf2 >= -999996
 T515_R_f214_k3: x_b1_is_LwQ - 1e+06 t515_leaf3 >= -999999
 T516_onehot: t516_leaf0 + t516_leaf1 + t516_leaf2 + t516_leaf3
   + t516_leaf4 = 1
 T516_L_f295_k0: const_Sale_Condition_Alloca + 1e+06 t516_leaf0
   <= 1.000000999999e+06
 T516_L_f3_k0: Overall_Qual_calc + 1e+06 t516_leaf0 <= 1.000006999999e+06
 T516_L_f214_k0: x_b1_is_LwQ + 1e+06 t516_leaf0 <= 1.000000999999e+06
 T516_L_f295_k1: const_Sale_Condition_Alloca + 1e+06 t516_leaf1
   <= 1.000000999999e+06
 T516_L_f3_k1: Overall_Qual_calc + 1e+06 t516_leaf1 <= 1.000006999999e+06
 T516_R_f214_k1: x_b1_is_LwQ - 1e+06 t516_leaf1 >= -999999
 T516_L_f295_k2: const_Sale_Condition_Alloca + 1e+06 t516_leaf2
   <= 1.000000999999e+06
 T516_R_f3_k2: Overall_Qual_calc - 1e+06 t516_leaf2 >= -999993
 T516_L_f296_k2: const_Sale_Condition_Family + 1e+06 t516_leaf2
   <= 1.000000999999e+06
 T516_L_f295_k3: const_Sale_Condition_Alloca + 1e+06 t516_leaf3
   <= 1.000000999999e+06
 T516_R_f3_k3: Overall_Qual_calc - 1e+06 t516_leaf3 >= -999993
 T516_R_f296_k3: const_Sale_Condition_Family - 1e+06 t516_leaf3 >= -999999
 T516_R_f295_k4: const_Sale_Condition_Alloca - 1e+06 t516_leaf4 >= -999999
 T517_onehot: t517_leaf0 + t517_leaf1 + t517_leaf2 + t517_leaf3
   + t517_leaf4 + t517_leaf5 = 1
 T517_L_f33_k0: x_Garage_Area + 1e+06 t517_leaf0 <= 1.000319999999e+06
 T517_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t517_leaf0 <= 1.000975999999e+06
 T517_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t517_leaf0 <= 1.000000999999e+06
 T517_L_f33_k1: x_Garage_Area + 1e+06 t517_leaf1 <= 1.000319999999e+06
 T517_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t517_leaf1 <= 1.000975999999e+06
 T517_R_f177_k1: x_ex2_is_AsbShng - 1e+06 t517_leaf1 >= -999999
 T517_L_f33_k2: x_Garage_Area + 1e+06 t517_leaf2 <= 1.000319999999e+06
 T517_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t517_leaf2 >= -999024
 T517_R_f33_k3: x_Garage_Area - 1e+06 t517_leaf3 >= -999680
 T517_L_f20_k3: x_Gr_Liv_Area + 1e+06 t517_leaf3 <= 1.000882999999e+06
 T517_R_f33_k4: x_Garage_Area - 1e+06 t517_leaf4 >= -999680
 T517_R_f20_k4: x_Gr_Liv_Area - 1e+06 t517_leaf4 >= -999117
 T517_L_f33_k4: x_Garage_Area + 1e+06 t517_leaf4 <= 1.000870999999e+06
 T517_R_f33_k5: x_Garage_Area - 1e+06 t517_leaf5 >= -999680
 T517_R_f20_k5: x_Gr_Liv_Area - 1e+06 t517_leaf5 >= -999117
 T517_R_f33_k5: x_Garage_Area - 1e+06 t517_leaf5 >= -999129
 T518_onehot: t518_leaf0 + t518_leaf1 + t518_leaf2 + t518_leaf3
   + t518_leaf4 + t518_leaf5 = 1
 T518_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t518_leaf0 <= 1.000000999999e+06
 T518_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t518_leaf0
   <= 1.001907999999e+06
 T518_L_f10_k0: const_Bsmt_Qual + 1e+06 t518_leaf0 <= 1.000001999999e+06
 T518_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t518_leaf1 <= 1.000000999999e+06
 T518_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t518_leaf1
   <= 1.001907999999e+06
 T518_R_f10_k1: const_Bsmt_Qual - 1e+06 t518_leaf1 >= -999998
 T518_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t518_leaf2 <= 1.000000999999e+06
 T518_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t518_leaf2 >= -998092
 T518_L_f37_k2: x_Open_Porch_SF + 1e+06 t518_leaf2 <= 1.000187999999e+06
 T518_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t518_leaf3 <= 1.000000999999e+06
 T518_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t518_leaf3 >= -998092
 T518_R_f37_k3: x_Open_Porch_SF - 1e+06 t518_leaf3 >= -999812
 T518_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t518_leaf4 >= -999999
 T518_L_f15_k4: const_Total_Bsmt_SF + 1e+06 t518_leaf4
   <= 1.000726999999e+06
 T518_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t518_leaf5 >= -999999
 T518_R_f15_k5: const_Total_Bsmt_SF - 1e+06 t518_leaf5 >= -999273
 T519_onehot: t519_leaf0 + t519_leaf1 = 1
 T519_L_f4_k0: const_Overall_Cond + 1e+06 t519_leaf0 <= 1.000002999999e+06
 T519_R_f4_k1: const_Overall_Cond - 1e+06 t519_leaf1 >= -999997
 T520_onehot: t520_leaf0 + t520_leaf1 + t520_leaf2 + t520_leaf3
   + t520_leaf4 + t520_leaf5 = 1
 T520_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t520_leaf0 <= 1.001258999999e+06
 T520_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t520_leaf0
   <= 1.001959999999e+06
 T520_L_f17_k0: x_1st_Flr_SF + 1e+06 t520_leaf0 <= 1.000797999999e+06
 T520_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t520_leaf1 <= 1.001258999999e+06
 T520_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t520_leaf1
   <= 1.001959999999e+06
 T520_R_f17_k1: x_1st_Flr_SF - 1e+06 t520_leaf1 >= -999202
 T520_L_f12_k2: x_BsmtFin_SF_1 + 1e+06 t520_leaf2 <= 1.001258999999e+06
 T520_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t520_leaf2 >= -998040
 T520_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t520_leaf3 >= -998741
 T520_L_f1_k3: const_Lot_Area + 1e+06 t520_leaf3 <= 1.015497999999e+06
 T520_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t520_leaf4 >= -998741
 T520_R_f1_k4: const_Lot_Area - 1e+06 t520_leaf4 >= -984502
 T520_L_f0_k4: const_Lot_Frontage + 1e+06 t520_leaf4 <= 1.000059999999e+06
 T520_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t520_leaf5 >= -998741
 T520_R_f1_k5: const_Lot_Area - 1e+06 t520_leaf5 >= -984502
 T520_R_f0_k5: const_Lot_Frontage - 1e+06 t520_leaf5 >= -999940
 T521_onehot: t521_leaf0 + t521_leaf1 + t521_leaf2 + t521_leaf3
   + t521_leaf4 = 1
 T521_L_f3_k0: Overall_Qual_calc + 1e+06 t521_leaf0 <= 1.000002999999e+06
 T521_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t521_leaf0
   <= 1.000000999999e+06
 T521_L_f3_k1: Overall_Qual_calc + 1e+06 t521_leaf1 <= 1.000002999999e+06
 T521_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t521_leaf1
   >= -999999
 T521_R_f3_k2: Overall_Qual_calc - 1e+06 t521_leaf2 >= -999997
 T521_L_f164_k2: x_ex1_is_BrkFace + 1e+06 t521_leaf2 <= 1.000000999999e+06
 T521_L_f1_k2: const_Lot_Area + 1e+06 t521_leaf2 <= 1.013051999999e+06
 T521_R_f3_k3: Overall_Qual_calc - 1e+06 t521_leaf3 >= -999997
 T521_L_f164_k3: x_ex1_is_BrkFace + 1e+06 t521_leaf3 <= 1.000000999999e+06
 T521_R_f1_k3: const_Lot_Area - 1e+06 t521_leaf3 >= -986948
 T521_R_f3_k4: Overall_Qual_calc - 1e+06 t521_leaf4 >= -999997
 T521_R_f164_k4: x_ex1_is_BrkFace - 1e+06 t521_leaf4 >= -999999
 T522_onehot: t522_leaf0 + t522_leaf1 + t522_leaf2 + t522_leaf3
   + t522_leaf4 + t522_leaf5 = 1
 T522_L_f3_k0: Overall_Qual_calc + 1e+06 t522_leaf0 <= 1.000002999999e+06
 T522_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t522_leaf0
   <= 1.000000999999e+06
 T522_L_f3_k1: Overall_Qual_calc + 1e+06 t522_leaf1 <= 1.000002999999e+06
 T522_R_f293_k1: const_Sale_Condition_Abnorml - 1e+06 t522_leaf1
   >= -999999
 T522_R_f3_k2: Overall_Qual_calc - 1e+06 t522_leaf2 >= -999997
 T522_L_f185_k2: x_ex2_is_MetalSd + 1e+06 t522_leaf2 <= 1.000000999999e+06
 T522_L_f26_k2: x_Kitchen_AbvGr + 1e+06 t522_leaf2 <= 1.000001999999e+06
 T522_R_f3_k3: Overall_Qual_calc - 1e+06 t522_leaf3 >= -999997
 T522_L_f185_k3: x_ex2_is_MetalSd + 1e+06 t522_leaf3 <= 1.000000999999e+06
 T522_R_f26_k3: x_Kitchen_AbvGr - 1e+06 t522_leaf3 >= -999998
 T522_R_f3_k4: Overall_Qual_calc - 1e+06 t522_leaf4 >= -999997
 T522_R_f185_k4: x_ex2_is_MetalSd - 1e+06 t522_leaf4 >= -999999
 T522_L_f31_k4: const_Garage_Yr_Blt + 1e+06 t522_leaf4
   <= 1.002005999999e+06
 T522_R_f3_k5: Overall_Qual_calc - 1e+06 t522_leaf5 >= -999997
 T522_R_f185_k5: x_ex2_is_MetalSd - 1e+06 t522_leaf5 >= -999999
 T522_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t522_leaf5 >= -997994
 T523_onehot: t523_leaf0 + t523_leaf1 + t523_leaf2 + t523_leaf3
   + t523_leaf4 = 1
 T523_L_f3_k0: Overall_Qual_calc + 1e+06 t523_leaf0 <= 1.000004999999e+06
 T523_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t523_leaf0 <= 1.000000999999e+06
 T523_L_f197_k0: x_mvt_is_No_aplica + 1e+06 t523_leaf0
   <= 1.000000999999e+06
 T523_L_f3_k1: Overall_Qual_calc + 1e+06 t523_leaf1 <= 1.000004999999e+06
 T523_L_f177_k1: x_ex2_is_AsbShng + 1e+06 t523_leaf1 <= 1.000000999999e+06
 T523_R_f197_k1: x_mvt_is_No_aplica - 1e+06 t523_leaf1 >= -999999
 T523_L_f3_k2: Overall_Qual_calc + 1e+06 t523_leaf2 <= 1.000004999999e+06
 T523_R_f177_k2: x_ex2_is_AsbShng - 1e+06 t523_leaf2 >= -999999
 T523_L_f251_k2: const_Garage_Type_Detchd + 1e+06 t523_leaf2
   <= 1.000000999999e+06
 T523_L_f3_k3: Overall_Qual_calc + 1e+06 t523_leaf3 <= 1.000004999999e+06
 T523_R_f177_k3: x_ex2_is_AsbShng - 1e+06 t523_leaf3 >= -999999
 T523_R_f251_k3: const_Garage_Type_Detchd - 1e+06 t523_leaf3 >= -999999
 T523_R_f3_k4: Overall_Qual_calc - 1e+06 t523_leaf4 >= -999995
 T524_onehot: t524_leaf0 + t524_leaf1 + t524_leaf2 + t524_leaf3 = 1
 T524_L_f11_k0: x_Bsmt_Cond + 1e+06 t524_leaf0 <= 1.000000999999e+06
 T524_L_f276_k0: const_Mo_Sold_6 + 1e+06 t524_leaf0 <= 1.000000999999e+06
 T524_L_f11_k1: x_Bsmt_Cond + 1e+06 t524_leaf1 <= 1.000000999999e+06
 T524_R_f276_k1: const_Mo_Sold_6 - 1e+06 t524_leaf1 >= -999999
 T524_R_f11_k2: x_Bsmt_Cond - 1e+06 t524_leaf2 >= -999999
 T524_L_f23_k2: x_Full_Bath + 1e+06 t524_leaf2 <= 1.000001999999e+06
 T524_R_f11_k3: x_Bsmt_Cond - 1e+06 t524_leaf3 >= -999999
 T524_R_f23_k3: x_Full_Bath - 1e+06 t524_leaf3 >= -999998
 T525_onehot: t525_leaf0 + t525_leaf1 = 1
 T525_L_f60_k0: const_MS_SubClass_190 + 1e+06 t525_leaf0
   <= 1.000000999999e+06
 T525_R_f60_k1: const_MS_SubClass_190 - 1e+06 t525_leaf1 >= -999999
 T526_onehot: t526_leaf0 + t526_leaf1 + t526_leaf2 + t526_leaf3
   + t526_leaf4 + t526_leaf5 = 1
 T526_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t526_leaf0 <= 1.000000999999e+06
 T526_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t526_leaf0 <= 1.000002999999e+06
 T526_L_f209_k0: const_Bsmt_Exposure_No + 1e+06 t526_leaf0
   <= 1.000000999999e+06
 T526_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t526_leaf1 <= 1.000000999999e+06
 T526_L_f25_k1: x_Bedroom_AbvGr + 1e+06 t526_leaf1 <= 1.000002999999e+06
 T526_R_f209_k1: const_Bsmt_Exposure_No - 1e+06 t526_leaf1 >= -999999
 T526_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t526_leaf2 <= 1.000000999999e+06
 T526_R_f25_k2: x_Bedroom_AbvGr - 1e+06 t526_leaf2 >= -999997
 T526_L_f11_k2: x_Bsmt_Cond + 1e+06 t526_leaf2 <= 1.000002999999e+06
 T526_L_f161_k3: x_ex1_is_AsbShng + 1e+06 t526_leaf3 <= 1.000000999999e+06
 T526_R_f25_k3: x_Bedroom_AbvGr - 1e+06 t526_leaf3 >= -999997
 T526_R_f11_k3: x_Bsmt_Cond - 1e+06 t526_leaf3 >= -999997
 T526_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t526_leaf4 >= -999999
 T526_L_f0_k4: const_Lot_Frontage + 1e+06 t526_leaf4 <= 1.000065999999e+06
 T526_R_f161_k5: x_ex1_is_AsbShng - 1e+06 t526_leaf5 >= -999999
 T526_R_f0_k5: const_Lot_Frontage - 1e+06 t526_leaf5 >= -999934
 T527_onehot: t527_leaf0 + t527_leaf1 + t527_leaf2 + t527_leaf3
   + t527_leaf4 + t527_leaf5 = 1
 T527_L_f1_k0: const_Lot_Area + 1e+06 t527_leaf0 <= 1.021779999999e+06
 T527_L_f231_k0: central_air_no + 1e+06 t527_leaf0 <= 1.000000999999e+06
 T527_L_f213_k0: x_b1_is_GLQ + 1e+06 t527_leaf0 <= 1.000000999999e+06
 T527_L_f1_k1: const_Lot_Area + 1e+06 t527_leaf1 <= 1.021779999999e+06
 T527_L_f231_k1: central_air_no + 1e+06 t527_leaf1 <= 1.000000999999e+06
 T527_R_f213_k1: x_b1_is_GLQ - 1e+06 t527_leaf1 >= -999999
 T527_L_f1_k2: const_Lot_Area + 1e+06 t527_leaf2 <= 1.021779999999e+06
 T527_R_f231_k2: central_air_no - 1e+06 t527_leaf2 >= -999999
 T527_L_f66_k2: const_MS_Zoning_RL + 1e+06 t527_leaf2
   <= 1.000000999999e+06
 T527_L_f1_k3: const_Lot_Area + 1e+06 t527_leaf3 <= 1.021779999999e+06
 T527_R_f231_k3: central_air_no - 1e+06 t527_leaf3 >= -999999
 T527_R_f66_k3: const_MS_Zoning_RL - 1e+06 t527_leaf3 >= -999999
 T527_R_f1_k4: const_Lot_Area - 1e+06 t527_leaf4 >= -978220
 T527_L_f213_k4: x_b1_is_GLQ + 1e+06 t527_leaf4 <= 1.000000999999e+06
 T527_R_f1_k5: const_Lot_Area - 1e+06 t527_leaf5 >= -978220
 T527_R_f213_k5: x_b1_is_GLQ - 1e+06 t527_leaf5 >= -999999
 T528_onehot: t528_leaf0 + t528_leaf1 + t528_leaf2 = 1
 T528_L_f114_k0: const_Neighborhood_StoneBr + 1e+06 t528_leaf0
   <= 1.000000999999e+06
 T528_R_f114_k1: const_Neighborhood_StoneBr - 1e+06 t528_leaf1 >= -999999
 T528_L_f289_k1: const_Sale_Type_New + 1e+06 t528_leaf1
   <= 1.000000999999e+06
 T528_R_f114_k2: const_Neighborhood_StoneBr - 1e+06 t528_leaf2 >= -999999
 T528_R_f289_k2: const_Sale_Type_New - 1e+06 t528_leaf2 >= -999999
 T529_onehot: t529_leaf0 + t529_leaf1 + t529_leaf2 + t529_leaf3
   + t529_leaf4 = 1
 T529_L_f199_k0: x_mvt_is_Stone + 1e+06 t529_leaf0 <= 1.000000999999e+06
 T529_L_f33_k0: x_Garage_Area + 1e+06 t529_leaf0 <= 1.000319999999e+06
 T529_L_f232_k0: central_air_yes + 1e+06 t529_leaf0 <= 1.000000999999e+06
 T529_L_f199_k1: x_mvt_is_Stone + 1e+06 t529_leaf1 <= 1.000000999999e+06
 T529_L_f33_k1: x_Garage_Area + 1e+06 t529_leaf1 <= 1.000319999999e+06
 T529_R_f232_k1: central_air_yes - 1e+06 t529_leaf1 >= -999999
 T529_L_f199_k2: x_mvt_is_Stone + 1e+06 t529_leaf2 <= 1.000000999999e+06
 T529_R_f33_k2: x_Garage_Area - 1e+06 t529_leaf2 >= -999680
 T529_L_f36_k2: x_Wood_Deck_SF + 1e+06 t529_leaf2 <= 1.000273999999e+06
 T529_L_f199_k3: x_mvt_is_Stone + 1e+06 t529_leaf3 <= 1.000000999999e+06
 T529_R_f33_k3: x_Garage_Area - 1e+06 t529_leaf3 >= -999680
 T529_R_f36_k3: x_Wood_Deck_SF - 1e+06 t529_leaf3 >= -999726
 T529_R_f199_k4: x_mvt_is_Stone - 1e+06 t529_leaf4 >= -999999
 T530_onehot: t530_leaf0 = 1
 T531_onehot: t531_leaf0 + t531_leaf1 + t531_leaf2 = 1
 T531_L_f5_k0: const_Year_Built + 1e+06 t531_leaf0 <= 1.001913999999e+06
 T531_L_f10_k0: const_Bsmt_Qual + 1e+06 t531_leaf0 <= 1.000001999999e+06
 T531_L_f5_k1: const_Year_Built + 1e+06 t531_leaf1 <= 1.001913999999e+06
 T531_R_f10_k1: const_Bsmt_Qual - 1e+06 t531_leaf1 >= -999998
 T531_R_f5_k2: const_Year_Built - 1e+06 t531_leaf2 >= -998086
 T532_onehot: t532_leaf0 = 1
 T533_onehot: t533_leaf0 + t533_leaf1 + t533_leaf2 = 1
 T533_L_f62_k0: const_MS_Zoning_C_(all) + 1e+06 t533_leaf0
   <= 1.000000999999e+06
 T533_R_f62_k1: const_MS_Zoning_C_(all) - 1e+06 t533_leaf1 >= -999999
 T533_L_f1_k1: const_Lot_Area + 1e+06 t533_leaf1 <= 1.006599999999e+06
 T533_R_f62_k2: const_MS_Zoning_C_(all) - 1e+06 t533_leaf2 >= -999999
 T533_R_f1_k2: const_Lot_Area - 1e+06 t533_leaf2 >= -993400
 T534_onehot: t534_leaf0 = 1
 T535_onehot: t535_leaf0 + t535_leaf1 + t535_leaf2 + t535_leaf3
   + t535_leaf4 = 1
 T535_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t535_leaf0 <= 1.000475999999e+06
 T535_L_f1_k0: const_Lot_Area + 1e+06 t535_leaf0 <= 1.021369999999e+06
 T535_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t535_leaf0 <= 1.001964999999e+06
 T535_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t535_leaf1 <= 1.000475999999e+06
 T535_L_f1_k1: const_Lot_Area + 1e+06 t535_leaf1 <= 1.021369999999e+06
 T535_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t535_leaf1 >= -998035
 T535_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t535_leaf2 <= 1.000475999999e+06
 T535_R_f1_k2: const_Lot_Area - 1e+06 t535_leaf2 >= -978630
 T535_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t535_leaf3 >= -999524
 T535_L_f18_k3: const_2nd_Flr_SF + 1e+06 t535_leaf3 <= 1.001370999999e+06
 T535_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t535_leaf4 >= -999524
 T535_R_f18_k4: const_2nd_Flr_SF - 1e+06 t535_leaf4 >= -998629
 T536_onehot: t536_leaf0 = 1
 T537_onehot: t537_leaf0 + t537_leaf1 = 1
 T537_L_f75_k0: const_Lot_Shape_IR3 + 1e+06 t537_leaf0
   <= 1.000000999999e+06
 T537_R_f75_k1: const_Lot_Shape_IR3 - 1e+06 t537_leaf1 >= -999999
 T538_onehot: t538_leaf0 + t538_leaf1 + t538_leaf2 + t538_leaf3
   + t538_leaf4 = 1
 T538_L_f3_k0: Overall_Qual_calc + 1e+06 t538_leaf0 <= 1.000006999999e+06
 T538_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t538_leaf0 <= 1.000917999999e+06
 T538_L_f3_k1: Overall_Qual_calc + 1e+06 t538_leaf1 <= 1.000006999999e+06
 T538_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t538_leaf1 >= -999082
 T538_R_f3_k2: Overall_Qual_calc - 1e+06 t538_leaf2 >= -999993
 T538_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t538_leaf2 <= 1.001794999999e+06
 T538_L_f206_k2: const_Bsmt_Exposure_Av + 1e+06 t538_leaf2
   <= 1.000000999999e+06
 T538_R_f3_k3: Overall_Qual_calc - 1e+06 t538_leaf3 >= -999993
 T538_L_f14_k3: x_Bsmt_Unf_SF + 1e+06 t538_leaf3 <= 1.001794999999e+06
 T538_R_f206_k3: const_Bsmt_Exposure_Av - 1e+06 t538_leaf3 >= -999999
 T538_R_f3_k4: Overall_Qual_calc - 1e+06 t538_leaf4 >= -999993
 T538_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t538_leaf4 >= -998205
 T539_onehot: t539_leaf0 + t539_leaf1 = 1
 T539_L_f44_k0: const_Yr_Sold + 1e+06 t539_leaf0 <= 1.002006999999e+06
 T539_R_f44_k1: const_Yr_Sold - 1e+06 t539_leaf1 >= -997993
 T540_onehot: t540_leaf0 + t540_leaf1 + t540_leaf2 + t540_leaf3
   + t540_leaf4 = 1
 T540_L_f277_k0: const_Mo_Sold_7 + 1e+06 t540_leaf0 <= 1.000000999999e+06
 T540_L_f35_k0: x_Garage_Cond + 1e+06 t540_leaf0 <= 1.000000999999e+06
 T540_L_f1_k0: const_Lot_Area + 1e+06 t540_leaf0 <= 1.008799999999e+06
 T540_L_f277_k1: const_Mo_Sold_7 + 1e+06 t540_leaf1 <= 1.000000999999e+06
 T540_L_f35_k1: x_Garage_Cond + 1e+06 t540_leaf1 <= 1.000000999999e+06
 T540_R_f1_k1: const_Lot_Area - 1e+06 t540_leaf1 >= -991200
 T540_L_f277_k2: const_Mo_Sold_7 + 1e+06 t540_leaf2 <= 1.000000999999e+06
 T540_R_f35_k2: x_Garage_Cond - 1e+06 t540_leaf2 >= -999999
 T540_L_f3_k2: Overall_Qual_calc + 1e+06 t540_leaf2 <= 1.000002999999e+06
 T540_L_f277_k3: const_Mo_Sold_7 + 1e+06 t540_leaf3 <= 1.000000999999e+06
 T540_R_f35_k3: x_Garage_Cond - 1e+06 t540_leaf3 >= -999999
 T540_R_f3_k3: Overall_Qual_calc - 1e+06 t540_leaf3 >= -999997
 T540_R_f277_k4: const_Mo_Sold_7 - 1e+06 t540_leaf4 >= -999999
 T541_onehot: t541_leaf0 + t541_leaf1 + t541_leaf2 = 1
 T541_L_f8_k0: x_Exter_Qual + 1e+06 t541_leaf0 <= 1.000001999999e+06
 T541_L_f33_k0: x_Garage_Area + 1e+06 t541_leaf0 <= 1.000296999999e+06
 T541_L_f8_k1: x_Exter_Qual + 1e+06 t541_leaf1 <= 1.000001999999e+06
 T541_R_f33_k1: x_Garage_Area - 1e+06 t541_leaf1 >= -999703
 T541_R_f8_k2: x_Exter_Qual - 1e+06 t541_leaf2 >= -999998
 T542_onehot: t542_leaf0 = 1
 T543_onehot: t543_leaf0 + t543_leaf1 + t543_leaf2 + t543_leaf3
   + t543_leaf4 + t543_leaf5 + t543_leaf6 = 1
 T543_L_f8_k0: x_Exter_Qual + 1e+06 t543_leaf0 <= 1.000002999999e+06
 T543_L_f20_k0: x_Gr_Liv_Area + 1e+06 t543_leaf0 <= 1.001733999999e+06
 T543_L_f71_k0: const_Alley_No_aplica + 1e+06 t543_leaf0
   <= 1.000000999999e+06
 T543_L_f8_k1: x_Exter_Qual + 1e+06 t543_leaf1 <= 1.000002999999e+06
 T543_L_f20_k1: x_Gr_Liv_Area + 1e+06 t543_leaf1 <= 1.001733999999e+06
 T543_R_f71_k1: const_Alley_No_aplica - 1e+06 t543_leaf1 >= -999999
 T543_L_f8_k2: x_Exter_Qual + 1e+06 t543_leaf2 <= 1.000002999999e+06
 T543_R_f20_k2: x_Gr_Liv_Area - 1e+06 t543_leaf2 >= -998266
 T543_L_f67_k2: const_MS_Zoning_RM + 1e+06 t543_leaf2
   <= 1.000000999999e+06
 T543_L_f8_k3: x_Exter_Qual + 1e+06 t543_leaf3 <= 1.000002999999e+06
 T543_R_f20_k3: x_Gr_Liv_Area - 1e+06 t543_leaf3 >= -998266
 T543_R_f67_k3: const_MS_Zoning_RM - 1e+06 t543_leaf3 >= -999999
 T543_R_f8_k4: x_Exter_Qual - 1e+06 t543_leaf4 >= -999997
 T543_L_f77_k4: const_Land_Contour_Bnk + 1e+06 t543_leaf4
   <= 1.000000999999e+06
 T543_L_f31_k4: const_Garage_Yr_Blt + 1e+06 t543_leaf4
   <= 1.001992999999e+06
 T543_R_f8_k5: x_Exter_Qual - 1e+06 t543_leaf5 >= -999997
 T543_L_f77_k5: const_Land_Contour_Bnk + 1e+06 t543_leaf5
   <= 1.000000999999e+06
 T543_R_f31_k5: const_Garage_Yr_Blt - 1e+06 t543_leaf5 >= -998007
 T543_R_f8_k6: x_Exter_Qual - 1e+06 t543_leaf6 >= -999997
 T543_R_f77_k6: const_Land_Contour_Bnk - 1e+06 t543_leaf6 >= -999999
 T544_onehot: t544_leaf0 = 1
 T545_onehot: t545_leaf0 + t545_leaf1 + t545_leaf2 + t545_leaf3
   + t545_leaf4 = 1
 T545_L_f0_k0: const_Lot_Frontage + 1e+06 t545_leaf0 <= 1.000114999999e+06
 T545_L_f201_k0: const_Foundation_CBlock + 1e+06 t545_leaf0
   <= 1.000000999999e+06
 T545_L_f0_k1: const_Lot_Frontage + 1e+06 t545_leaf1 <= 1.000114999999e+06
 T545_R_f201_k1: const_Foundation_CBlock - 1e+06 t545_leaf1 >= -999999
 T545_L_f30_k1: x_Fireplace_Qu + 1e+06 t545_leaf1 <= 999999.999999
 T545_L_f0_k2: const_Lot_Frontage + 1e+06 t545_leaf2 <= 1.000114999999e+06
 T545_R_f201_k2: const_Foundation_CBlock - 1e+06 t545_leaf2 >= -999999
 T545_R_f30_k2: x_Fireplace_Qu - 1e+06 t545_leaf2 >= -1e+06
 T545_R_f0_k3: const_Lot_Frontage - 1e+06 t545_leaf3 >= -999885
 T545_L_f8_k3: x_Exter_Qual + 1e+06 t545_leaf3 <= 1.000002999999e+06
 T545_R_f0_k4: const_Lot_Frontage - 1e+06 t545_leaf4 >= -999885
 T545_R_f8_k4: x_Exter_Qual - 1e+06 t545_leaf4 >= -999997
 T546_onehot: t546_leaf0 + t546_leaf1 = 1
 T546_L_f120_k0: const_Condition_1_PosA + 1e+06 t546_leaf0
   <= 1.000000999999e+06
 T546_R_f120_k1: const_Condition_1_PosA - 1e+06 t546_leaf1 >= -999999
 T547_onehot: t547_leaf0 + t547_leaf1 = 1
 T547_L_f0_k0: const_Lot_Frontage + 1e+06 t547_leaf0 <= 1.000033999999e+06
 T547_R_f0_k1: const_Lot_Frontage - 1e+06 t547_leaf1 >= -999966
 T548_onehot: t548_leaf0 + t548_leaf1 + t548_leaf2 + t548_leaf3 = 1
 T548_L_f233_k0: x_elect_is_FuseA + 1e+06 t548_leaf0 <= 1.000000999999e+06
 T548_R_f233_k1: x_elect_is_FuseA - 1e+06 t548_leaf1 >= -999999
 T548_L_f43_k1: const_Misc_Val + 1e+06 t548_leaf1 <= 1.000053999999e+06
 T548_L_f177_k1: x_ex2_is_AsbShng + 1e+06 t548_leaf1 <= 1.000000999999e+06
 T548_R_f233_k2: x_elect_is_FuseA - 1e+06 t548_leaf2 >= -999999
 T548_L_f43_k2: const_Misc_Val + 1e+06 t548_leaf2 <= 1.000053999999e+06
 T548_R_f177_k2: x_ex2_is_AsbShng - 1e+06 t548_leaf2 >= -999999
 T548_R_f233_k3: x_elect_is_FuseA - 1e+06 t548_leaf3 >= -999999
 T548_R_f43_k3: const_Misc_Val - 1e+06 t548_leaf3 >= -999946
 T549_onehot: t549_leaf0 + t549_leaf1 + t549_leaf2 + t549_leaf3
   + t549_leaf4 + t549_leaf5 = 1
 T549_L_f26_k0: x_Kitchen_AbvGr + 1e+06 t549_leaf0 <= 1.000001999999e+06
 T549_L_f20_k0: x_Gr_Liv_Area + 1e+06 t549_leaf0 <= 1.000746999999e+06
 T549_L_f17_k0: x_1st_Flr_SF + 1e+06 t549_leaf0 <= 1.000711999999e+06
 T549_L_f26_k1: x_Kitchen_AbvGr + 1e+06 t549_leaf1 <= 1.000001999999e+06
 T549_L_f20_k1: x_Gr_Liv_Area + 1e+06 t549_leaf1 <= 1.000746999999e+06
 T549_R_f17_k1: x_1st_Flr_SF - 1e+06 t549_leaf1 >= -999288
 T549_L_f26_k2: x_Kitchen_AbvGr + 1e+06 t549_leaf2 <= 1.000001999999e+06
 T549_R_f20_k2: x_Gr_Liv_Area - 1e+06 t549_leaf2 >= -999253
 T549_L_f3_k2: Overall_Qual_calc + 1e+06 t549_leaf2 <= 1.000004999999e+06
 T549_L_f26_k3: x_Kitchen_AbvGr + 1e+06 t549_leaf3 <= 1.000001999999e+06
 T549_R_f20_k3: x_Gr_Liv_Area - 1e+06 t549_leaf3 >= -999253
 T549_R_f3_k3: Overall_Qual_calc - 1e+06 t549_leaf3 >= -999995
 T549_R_f26_k4: x_Kitchen_AbvGr - 1e+06 t549_leaf4 >= -999998
 T549_L_f9_k4: x_Exter_Cond + 1e+06 t549_leaf4 <= 1.000001999999e+06
 T549_R_f26_k5: x_Kitchen_AbvGr - 1e+06 t549_leaf5 >= -999998
 T549_R_f9_k5: x_Exter_Cond - 1e+06 t549_leaf5 >= -999998
 T550_onehot: t550_leaf0 + t550_leaf1 + t550_leaf2 + t550_leaf3 = 1
 T550_L_f16_k0: x_Heating_QC + 1e+06 t550_leaf0 <= 1.000001999999e+06
 T550_R_f16_k1: x_Heating_QC - 1e+06 t550_leaf1 >= -999998
 T550_L_f7_k1: x_Mas_Vnr_Area + 1e+06 t550_leaf1 <= 1.000921999999e+06
 T550_L_f0_k1: const_Lot_Frontage + 1e+06 t550_leaf1 <= 1.000084999999e+06
 T550_R_f16_k2: x_Heating_QC - 1e+06 t550_leaf2 >= -999998
 T550_L_f7_k2: x_Mas_Vnr_Area + 1e+06 t550_leaf2 <= 1.000921999999e+06
 T550_R_f0_k2: const_Lot_Frontage - 1e+06 t550_leaf2 >= -999915
 T550_R_f16_k3: x_Heating_QC - 1e+06 t550_leaf3 >= -999998
 T550_R_f7_k3: x_Mas_Vnr_Area - 1e+06 t550_leaf3 >= -999078
 T551_onehot: t551_leaf0 = 1
 T552_onehot: t552_leaf0 + t552_leaf1 + t552_leaf2 + t552_leaf3
   + t552_leaf4 = 1
 T552_L_f277_k0: const_Mo_Sold_7 + 1e+06 t552_leaf0 <= 1.000000999999e+06
 T552_L_f1_k0: const_Lot_Area + 1e+06 t552_leaf0 <= 1.019521999999e+06
 T552_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t552_leaf0
   <= 1.000244999999e+06
 T552_L_f277_k1: const_Mo_Sold_7 + 1e+06 t552_leaf1 <= 1.000000999999e+06
 T552_L_f1_k1: const_Lot_Area + 1e+06 t552_leaf1 <= 1.019521999999e+06
 T552_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t552_leaf1 >= -999755
 T552_L_f277_k2: const_Mo_Sold_7 + 1e+06 t552_leaf2 <= 1.000000999999e+06
 T552_R_f1_k2: const_Lot_Area - 1e+06 t552_leaf2 >= -980478
 T552_L_f0_k2: const_Lot_Frontage + 1e+06 t552_leaf2 <= 1.000173999999e+06
 T552_L_f277_k3: const_Mo_Sold_7 + 1e+06 t552_leaf3 <= 1.000000999999e+06
 T552_R_f1_k3: const_Lot_Area - 1e+06 t552_leaf3 >= -980478
 T552_R_f0_k3: const_Lot_Frontage - 1e+06 t552_leaf3 >= -999826
 T552_R_f277_k4: const_Mo_Sold_7 - 1e+06 t552_leaf4 >= -999999
 T553_onehot: t553_leaf0 + t553_leaf1 + t553_leaf2 + t553_leaf3 = 1
 T553_L_f4_k0: const_Overall_Cond + 1e+06 t553_leaf0 <= 1.000005999999e+06
 T553_L_f5_k0: const_Year_Built + 1e+06 t553_leaf0 <= 1.001952999999e+06
 T553_L_f5_k0: const_Year_Built + 1e+06 t553_leaf0 <= 1.001901999999e+06
 T553_L_f4_k1: const_Overall_Cond + 1e+06 t553_leaf1 <= 1.000005999999e+06
 T553_L_f5_k1: const_Year_Built + 1e+06 t553_leaf1 <= 1.001952999999e+06
 T553_R_f5_k1: const_Year_Built - 1e+06 t553_leaf1 >= -998098
 T553_L_f4_k2: const_Overall_Cond + 1e+06 t553_leaf2 <= 1.000005999999e+06
 T553_R_f5_k2: const_Year_Built - 1e+06 t553_leaf2 >= -998047
 T553_R_f4_k3: const_Overall_Cond - 1e+06 t553_leaf3 >= -999994
 T554_onehot: t554_leaf0 = 1
 T555_onehot: t555_leaf0 = 1
 T556_onehot: t556_leaf0 + t556_leaf1 = 1
 T556_L_f20_k0: x_Gr_Liv_Area + 1e+06 t556_leaf0 <= 1.000767999999e+06
 T556_R_f20_k1: x_Gr_Liv_Area - 1e+06 t556_leaf1 >= -999232
 T557_onehot: t557_leaf0 + t557_leaf1 + t557_leaf2 + t557_leaf3 = 1
 T557_L_f1_k0: const_Lot_Area + 1e+06 t557_leaf0 <= 1.005999999999e+06
 T557_R_f1_k1: const_Lot_Area - 1e+06 t557_leaf1 >= -994000
 T557_L_f29_k1: const_Fireplaces + 1e+06 t557_leaf1 <= 1.000002999999e+06
 T557_L_f28_k1: x_TotRms_AbvGrd + 1e+06 t557_leaf1 <= 1.000004999999e+06
 T557_R_f1_k2: const_Lot_Area - 1e+06 t557_leaf2 >= -994000
 T557_L_f29_k2: const_Fireplaces + 1e+06 t557_leaf2 <= 1.000002999999e+06
 T557_R_f28_k2: x_TotRms_AbvGrd - 1e+06 t557_leaf2 >= -999995
 T557_R_f1_k3: const_Lot_Area - 1e+06 t557_leaf3 >= -994000
 T557_R_f29_k3: const_Fireplaces - 1e+06 t557_leaf3 >= -999997
 T558_onehot: t558_leaf0 + t558_leaf1 + t558_leaf2 + t558_leaf3 = 1
 T558_L_f33_k0: x_Garage_Area + 1e+06 t558_leaf0 <= 1.000887999999e+06
 T558_L_f117_k0: const_Condition_1_Artery + 1e+06 t558_leaf0
   <= 1.000000999999e+06
 T558_L_f32_k0: const_Garage_Cars + 1e+06 t558_leaf0 <= 1.000002999999e+06
 T558_L_f33_k1: x_Garage_Area + 1e+06 t558_leaf1 <= 1.000887999999e+06
 T558_L_f117_k1: const_Condition_1_Artery + 1e+06 t558_leaf1
   <= 1.000000999999e+06
 T558_R_f32_k1: const_Garage_Cars - 1e+06 t558_leaf1 >= -999997
 T558_L_f33_k2: x_Garage_Area + 1e+06 t558_leaf2 <= 1.000887999999e+06
 T558_R_f117_k2: const_Condition_1_Artery - 1e+06 t558_leaf2 >= -999999
 T558_R_f33_k3: x_Garage_Area - 1e+06 t558_leaf3 >= -999112
 T559_onehot: t559_leaf0 + t559_leaf1 + t559_leaf2 + t559_leaf3
   + t559_leaf4 = 1
 T559_L_f36_k0: x_Wood_Deck_SF + 1e+06 t559_leaf0 <= 1.000215999999e+06
 T559_L_f4_k0: const_Overall_Cond + 1e+06 t559_leaf0 <= 1.000005999999e+06
 T559_L_f216_k0: x_b1_is_Rec + 1e+06 t559_leaf0 <= 1.000000999999e+06
 T559_L_f36_k1: x_Wood_Deck_SF + 1e+06 t559_leaf1 <= 1.000215999999e+06
 T559_L_f4_k1: const_Overall_Cond + 1e+06 t559_leaf1 <= 1.000005999999e+06
 T559_R_f216_k1: x_b1_is_Rec - 1e+06 t559_leaf1 >= -999999
 T559_L_f36_k2: x_Wood_Deck_SF + 1e+06 t559_leaf2 <= 1.000215999999e+06
 T559_R_f4_k2: const_Overall_Cond - 1e+06 t559_leaf2 >= -999994
 T559_L_f1_k2: const_Lot_Area + 1e+06 t559_leaf2 <= 1.008625999999e+06
 T559_L_f36_k3: x_Wood_Deck_SF + 1e+06 t559_leaf3 <= 1.000215999999e+06
 T559_R_f4_k3: const_Overall_Cond - 1e+06 t559_leaf3 >= -999994
 T559_R_f1_k3: const_Lot_Area - 1e+06 t559_leaf3 >= -991374
 T559_R_f36_k4: x_Wood_Deck_SF - 1e+06 t559_leaf4 >= -999784
 T560_onehot: t560_leaf0 + t560_leaf1 + t560_leaf2 + t560_leaf3 = 1
 T560_L_f169_k0: x_ex1_is_MetalSd + 1e+06 t560_leaf0 <= 1.000000999999e+06
 T560_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t560_leaf0
   <= 1.000000999999e+06
 T560_L_f1_k0: const_Lot_Area + 1e+06 t560_leaf0 <= 1.012852999999e+06
 T560_L_f169_k1: x_ex1_is_MetalSd + 1e+06 t560_leaf1 <= 1.000000999999e+06
 T560_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t560_leaf1
   <= 1.000000999999e+06
 T560_R_f1_k1: const_Lot_Area - 1e+06 t560_leaf1 >= -987147
 T560_L_f169_k2: x_ex1_is_MetalSd + 1e+06 t560_leaf2 <= 1.000000999999e+06
 T560_R_f95_k2: const_Neighborhood_Crawfor - 1e+06 t560_leaf2 >= -999999
 T560_R_f169_k3: x_ex1_is_MetalSd - 1e+06 t560_leaf3 >= -999999
 T561_onehot: t561_leaf0 + t561_leaf1 + t561_leaf2 = 1
 T561_L_f20_k0: x_Gr_Liv_Area + 1e+06 t561_leaf0 <= 1.000746999999e+06
 T561_L_f17_k0: x_1st_Flr_SF + 1e+06 t561_leaf0 <= 1.000716999999e+06
 T561_L_f20_k1: x_Gr_Liv_Area + 1e+06 t561_leaf1 <= 1.000746999999e+06
 T561_R_f17_k1: x_1st_Flr_SF - 1e+06 t561_leaf1 >= -999283
 T561_R_f20_k2: x_Gr_Liv_Area - 1e+06 t561_leaf2 >= -999253
 T562_onehot: t562_leaf0 + t562_leaf1 + t562_leaf2 = 1
 T562_L_f17_k0: x_1st_Flr_SF + 1e+06 t562_leaf0 <= 1.001501999999e+06
 T562_R_f17_k1: x_1st_Flr_SF - 1e+06 t562_leaf1 >= -998498
 T562_L_f206_k1: const_Bsmt_Exposure_Av + 1e+06 t562_leaf1
   <= 1.000000999999e+06
 T562_R_f17_k2: x_1st_Flr_SF - 1e+06 t562_leaf2 >= -998498
 T562_R_f206_k2: const_Bsmt_Exposure_Av - 1e+06 t562_leaf2 >= -999999
 T563_onehot: t563_leaf0 = 1
 T564_onehot: t564_leaf0 + t564_leaf1 + t564_leaf2 = 1
 T564_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t564_leaf0 <= 1.000000999999e+06
 T564_R_f177_k1: x_ex2_is_AsbShng - 1e+06 t564_leaf1 >= -999999
 T564_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t564_leaf1
   <= 1.000726999999e+06
 T564_R_f177_k2: x_ex2_is_AsbShng - 1e+06 t564_leaf2 >= -999999
 T564_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t564_leaf2 >= -999273
 T565_onehot: t565_leaf0 = 1
 T566_onehot: t566_leaf0 + t566_leaf1 + t566_leaf2 + t566_leaf3
   + t566_leaf4 + t566_leaf5 = 1
 T566_L_f6_k0: const_Year_Remod_Add + 1e+06 t566_leaf0
   <= 1.001957999999e+06
 T566_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t566_leaf0
   <= 1.000711999999e+06
 T566_L_f1_k0: const_Lot_Area + 1e+06 t566_leaf0 <= 1.008093999999e+06
 T566_L_f6_k1: const_Year_Remod_Add + 1e+06 t566_leaf1
   <= 1.001957999999e+06
 T566_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t566_leaf1
   <= 1.000711999999e+06
 T566_R_f1_k1: const_Lot_Area - 1e+06 t566_leaf1 >= -991906
 T566_L_f6_k2: const_Year_Remod_Add + 1e+06 t566_leaf2
   <= 1.001957999999e+06
 T566_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t566_leaf2 >= -999288
 T566_L_f24_k2: x_Half_Bath + 1e+06 t566_leaf2 <= 1.000000999999e+06
 T566_L_f6_k3: const_Year_Remod_Add + 1e+06 t566_leaf3
   <= 1.001957999999e+06
 T566_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t566_leaf3 >= -999288
 T566_R_f24_k3: x_Half_Bath - 1e+06 t566_leaf3 >= -999999
 T566_R_f6_k4: const_Year_Remod_Add - 1e+06 t566_leaf4 >= -998042
 T566_L_f4_k4: const_Overall_Cond + 1e+06 t566_leaf4 <= 1.000003999999e+06
 T566_R_f6_k5: const_Year_Remod_Add - 1e+06 t566_leaf5 >= -998042
 T566_R_f4_k5: const_Overall_Cond - 1e+06 t566_leaf5 >= -999996
 T567_onehot: t567_leaf0 + t567_leaf1 + t567_leaf2 + t567_leaf3
   + t567_leaf4 + t567_leaf5 = 1
 T567_L_f4_k0: const_Overall_Cond + 1e+06 t567_leaf0 <= 1.000003999999e+06
 T567_L_f6_k0: const_Year_Remod_Add + 1e+06 t567_leaf0
   <= 1.001950999999e+06
 T567_L_f4_k1: const_Overall_Cond + 1e+06 t567_leaf1 <= 1.000003999999e+06
 T567_R_f6_k1: const_Year_Remod_Add - 1e+06 t567_leaf1 >= -998049
 T567_L_f183_k1: x_ex2_is_HdBoard + 1e+06 t567_leaf1 <= 1.000000999999e+06
 T567_L_f4_k2: const_Overall_Cond + 1e+06 t567_leaf2 <= 1.000003999999e+06
 T567_R_f6_k2: const_Year_Remod_Add - 1e+06 t567_leaf2 >= -998049
 T567_R_f183_k2: x_ex2_is_HdBoard - 1e+06 t567_leaf2 >= -999999
 T567_R_f4_k3: const_Overall_Cond - 1e+06 t567_leaf3 >= -999996
 T567_L_f8_k3: x_Exter_Qual + 1e+06 t567_leaf3 <= 1.000002999999e+06
 T567_L_f72_k3: const_Alley_Pave + 1e+06 t567_leaf3 <= 1.000000999999e+06
 T567_R_f4_k4: const_Overall_Cond - 1e+06 t567_leaf4 >= -999996
 T567_L_f8_k4: x_Exter_Qual + 1e+06 t567_leaf4 <= 1.000002999999e+06
 T567_R_f72_k4: const_Alley_Pave - 1e+06 t567_leaf4 >= -999999
 T567_R_f4_k5: const_Overall_Cond - 1e+06 t567_leaf5 >= -999996
 T567_R_f8_k5: x_Exter_Qual - 1e+06 t567_leaf5 >= -999997
 T568_onehot: t568_leaf0 + t568_leaf1 + t568_leaf2 + t568_leaf3 = 1
 T568_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t568_leaf0 <= 1.000010999999e+06
 T568_L_f29_k0: const_Fireplaces + 1e+06 t568_leaf0 <= 1.000001999999e+06
 T568_L_f28_k1: x_TotRms_AbvGrd + 1e+06 t568_leaf1 <= 1.000010999999e+06
 T568_R_f29_k1: const_Fireplaces - 1e+06 t568_leaf1 >= -999998
 T568_R_f28_k2: x_TotRms_AbvGrd - 1e+06 t568_leaf2 >= -999989
 T568_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t568_leaf2 <= 1.001087999999e+06
 T568_R_f28_k3: x_TotRms_AbvGrd - 1e+06 t568_leaf3 >= -999989
 T568_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t568_leaf3 >= -998912
 T569_onehot: t569_leaf0 = 1
 T570_onehot: t570_leaf0 = 1
 T571_onehot: t571_leaf0 + t571_leaf1 + t571_leaf2 + t571_leaf3 = 1
 T571_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t571_leaf0
   <= 1.002135999999e+06
 T571_L_f272_k0: const_Mo_Sold_2 + 1e+06 t571_leaf0 <= 1.000000999999e+06
 T571_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t571_leaf1
   <= 1.002135999999e+06
 T571_R_f272_k1: const_Mo_Sold_2 - 1e+06 t571_leaf1 >= -999999
 T571_L_f33_k1: x_Garage_Area + 1e+06 t571_leaf1 <= 1.000473999999e+06
 T571_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t571_leaf2
   <= 1.002135999999e+06
 T571_R_f272_k2: const_Mo_Sold_2 - 1e+06 t571_leaf2 >= -999999
 T571_R_f33_k2: x_Garage_Area - 1e+06 t571_leaf2 >= -999526
 T571_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t571_leaf3 >= -997864
 T572_onehot: t572_leaf0 + t572_leaf1 + t572_leaf2 = 1
 T572_L_f5_k0: const_Year_Built + 1e+06 t572_leaf0 <= 1.002007999999e+06
 T572_L_f1_k0: const_Lot_Area + 1e+06 t572_leaf0 <= 1.027696999999e+06
 T572_L_f5_k1: const_Year_Built + 1e+06 t572_leaf1 <= 1.002007999999e+06
 T572_R_f1_k1: const_Lot_Area - 1e+06 t572_leaf1 >= -972303
 T572_R_f5_k2: const_Year_Built - 1e+06 t572_leaf2 >= -997992
 T573_onehot: t573_leaf0 = 1
 T574_onehot: t574_leaf0 + t574_leaf1 + t574_leaf2 + t574_leaf3 = 1
 T574_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t574_leaf0 <= 1.001258999999e+06
 T574_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t574_leaf1 >= -998741
 T574_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t574_leaf1 <= 1.000447999999e+06
 T574_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t574_leaf1 <= 1.001964999999e+06
 T574_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t574_leaf2 >= -998741
 T574_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t574_leaf2 <= 1.000447999999e+06
 T574_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t574_leaf2 >= -998035
 T574_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t574_leaf3 >= -998741
 T574_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t574_leaf3 >= -999552
 T575_onehot: t575_leaf0 + t575_leaf1 = 1
 T575_L_f214_k0: x_b1_is_LwQ + 1e+06 t575_leaf0 <= 1.000000999999e+06
 T575_R_f214_k1: x_b1_is_LwQ - 1e+06 t575_leaf1 >= -999999
 T576_onehot: t576_leaf0 + t576_leaf1 + t576_leaf2 + t576_leaf3
   + t576_leaf4 + t576_leaf5 = 1
 T576_L_f3_k0: Overall_Qual_calc + 1e+06 t576_leaf0 <= 1.000004999999e+06
 T576_L_f6_k0: const_Year_Remod_Add + 1e+06 t576_leaf0
   <= 1.001953999999e+06
 T576_L_f3_k1: Overall_Qual_calc + 1e+06 t576_leaf1 <= 1.000004999999e+06
 T576_R_f6_k1: const_Year_Remod_Add - 1e+06 t576_leaf1 >= -998046
 T576_L_f37_k1: x_Open_Porch_SF + 1e+06 t576_leaf1 <= 1.000034999999e+06
 T576_L_f3_k2: Overall_Qual_calc + 1e+06 t576_leaf2 <= 1.000004999999e+06
 T576_R_f6_k2: const_Year_Remod_Add - 1e+06 t576_leaf2 >= -998046
 T576_R_f37_k2: x_Open_Porch_SF - 1e+06 t576_leaf2 >= -999965
 T576_R_f3_k3: Overall_Qual_calc - 1e+06 t576_leaf3 >= -999995
 T576_L_f33_k3: x_Garage_Area + 1e+06 t576_leaf3 <= 1.000543999999e+06
 T576_R_f3_k4: Overall_Qual_calc - 1e+06 t576_leaf4 >= -999995
 T576_R_f33_k4: x_Garage_Area - 1e+06 t576_leaf4 >= -999456
 T576_L_f33_k4: x_Garage_Area + 1e+06 t576_leaf4 <= 1.000635999999e+06
 T576_R_f3_k5: Overall_Qual_calc - 1e+06 t576_leaf5 >= -999995
 T576_R_f33_k5: x_Garage_Area - 1e+06 t576_leaf5 >= -999456
 T576_R_f33_k5: x_Garage_Area - 1e+06 t576_leaf5 >= -999364
 T577_onehot: t577_leaf0 = 1
 T578_onehot: t578_leaf0 + t578_leaf1 + t578_leaf2 = 1
 T578_L_f273_k0: const_Mo_Sold_3 + 1e+06 t578_leaf0 <= 1.000000999999e+06
 T578_L_f40_k0: x_Screen_Porch + 1e+06 t578_leaf0 <= 1.000153999999e+06
 T578_L_f273_k1: const_Mo_Sold_3 + 1e+06 t578_leaf1 <= 1.000000999999e+06
 T578_R_f40_k1: x_Screen_Porch - 1e+06 t578_leaf1 >= -999846
 T578_R_f273_k2: const_Mo_Sold_3 - 1e+06 t578_leaf2 >= -999999
 T579_onehot: t579_leaf0 + t579_leaf1 = 1
 T579_L_f18_k0: const_2nd_Flr_SF + 1e+06 t579_leaf0 <= 1.000833999999e+06
 T579_R_f18_k1: const_2nd_Flr_SF - 1e+06 t579_leaf1 >= -999166
 T580_onehot: t580_leaf0 + t580_leaf1 + t580_leaf2 + t580_leaf3
   + t580_leaf4 + t580_leaf5 = 1
 T580_L_f216_k0: x_b1_is_Rec + 1e+06 t580_leaf0 <= 1.000000999999e+06
 T580_L_f29_k0: const_Fireplaces + 1e+06 t580_leaf0 <= 1.000002999999e+06
 T580_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t580_leaf0 <= 1.000566999999e+06
 T580_L_f216_k1: x_b1_is_Rec + 1e+06 t580_leaf1 <= 1.000000999999e+06
 T580_L_f29_k1: const_Fireplaces + 1e+06 t580_leaf1 <= 1.000002999999e+06
 T580_R_f7_k1: x_Mas_Vnr_Area - 1e+06 t580_leaf1 >= -999433
 T580_L_f216_k2: x_b1_is_Rec + 1e+06 t580_leaf2 <= 1.000000999999e+06
 T580_R_f29_k2: const_Fireplaces - 1e+06 t580_leaf2 >= -999997
 T580_R_f216_k3: x_b1_is_Rec - 1e+06 t580_leaf3 >= -999999
 T580_L_f12_k3: x_BsmtFin_SF_1 + 1e+06 t580_leaf3 <= 1.000389999999e+06
 T580_R_f216_k4: x_b1_is_Rec - 1e+06 t580_leaf4 >= -999999
 T580_R_f12_k4: x_BsmtFin_SF_1 - 1e+06 t580_leaf4 >= -999610
 T580_L_f12_k4: x_BsmtFin_SF_1 + 1e+06 t580_leaf4 <= 1.000587999999e+06
 T580_R_f216_k5: x_b1_is_Rec - 1e+06 t580_leaf5 >= -999999
 T580_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t580_leaf5 >= -999610
 T580_R_f12_k5: x_BsmtFin_SF_1 - 1e+06 t580_leaf5 >= -999412
 T581_onehot: t581_leaf0 + t581_leaf1 = 1
 T581_L_f33_k0: x_Garage_Area + 1e+06 t581_leaf0 <= 1.001153999999e+06
 T581_R_f33_k1: x_Garage_Area - 1e+06 t581_leaf1 >= -998846
 T582_onehot: t582_leaf0 + t582_leaf1 + t582_leaf2 = 1
 T582_L_f3_k0: Overall_Qual_calc + 1e+06 t582_leaf0 <= 1.000009999999e+06
 T582_L_f16_k0: x_Heating_QC + 1e+06 t582_leaf0 <= 1.000001999999e+06
 T582_L_f3_k1: Overall_Qual_calc + 1e+06 t582_leaf1 <= 1.000009999999e+06
 T582_R_f16_k1: x_Heating_QC - 1e+06 t582_leaf1 >= -999998
 T582_R_f3_k2: Overall_Qual_calc - 1e+06 t582_leaf2 >= -999990
 T583_onehot: t583_leaf0 = 1
 T584_onehot: t584_leaf0 + t584_leaf1 + t584_leaf2 + t584_leaf3
   + t584_leaf4 = 1
 T584_L_f5_k0: const_Year_Built + 1e+06 t584_leaf0 <= 1.001913999999e+06
 T584_R_f5_k1: const_Year_Built - 1e+06 t584_leaf1 >= -998086
 T584_L_f18_k1: const_2nd_Flr_SF + 1e+06 t584_leaf1 <= 1.000987999999e+06
 T584_L_f10_k1: const_Bsmt_Qual + 1e+06 t584_leaf1 <= 1.000001999999e+06
 T584_R_f5_k2: const_Year_Built - 1e+06 t584_leaf2 >= -998086
 T584_L_f18_k2: const_2nd_Flr_SF + 1e+06 t584_leaf2 <= 1.000987999999e+06
 T584_R_f10_k2: const_Bsmt_Qual - 1e+06 t584_leaf2 >= -999998
 T584_R_f5_k3: const_Year_Built - 1e+06 t584_leaf3 >= -998086
 T584_R_f18_k3: const_2nd_Flr_SF - 1e+06 t584_leaf3 >= -999012
 T584_L_f33_k3: x_Garage_Area + 1e+06 t584_leaf3 <= 1.000621999999e+06
 T584_R_f5_k4: const_Year_Built - 1e+06 t584_leaf4 >= -998086
 T584_R_f18_k4: const_2nd_Flr_SF - 1e+06 t584_leaf4 >= -999012
 T584_R_f33_k4: x_Garage_Area - 1e+06 t584_leaf4 >= -999378
 T585_onehot: t585_leaf0 + t585_leaf1 + t585_leaf2 + t585_leaf3
   + t585_leaf4 = 1
 T585_L_f0_k0: const_Lot_Frontage + 1e+06 t585_leaf0 <= 1.000116999999e+06
 T585_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t585_leaf0 <= 1.000004999999e+06
 T585_L_f272_k0: const_Mo_Sold_2 + 1e+06 t585_leaf0 <= 1.000000999999e+06
 T585_L_f0_k1: const_Lot_Frontage + 1e+06 t585_leaf1 <= 1.000116999999e+06
 T585_L_f25_k1: x_Bedroom_AbvGr + 1e+06 t585_leaf1 <= 1.000004999999e+06
 T585_R_f272_k1: const_Mo_Sold_2 - 1e+06 t585_leaf1 >= -999999
 T585_L_f0_k2: const_Lot_Frontage + 1e+06 t585_leaf2 <= 1.000116999999e+06
 T585_R_f25_k2: x_Bedroom_AbvGr - 1e+06 t585_leaf2 >= -999995
 T585_R_f0_k3: const_Lot_Frontage - 1e+06 t585_leaf3 >= -999883
 T585_L_f297_k3: const_Sale_Condition_Normal + 1e+06 t585_leaf3
   <= 1.000000999999e+06
 T585_R_f0_k4: const_Lot_Frontage - 1e+06 t585_leaf4 >= -999883
 T585_R_f297_k4: const_Sale_Condition_Normal - 1e+06 t585_leaf4 >= -999999
 T586_onehot: t586_leaf0 + t586_leaf1 + t586_leaf2 + t586_leaf3
   + t586_leaf4 = 1
 T586_L_f3_k0: Overall_Qual_calc + 1e+06 t586_leaf0 <= 1.000002999999e+06
 T586_R_f3_k1: Overall_Qual_calc - 1e+06 t586_leaf1 >= -999997
 T586_L_f3_k1: Overall_Qual_calc + 1e+06 t586_leaf1 <= 1.000006999999e+06
 T586_L_f20_k1: x_Gr_Liv_Area + 1e+06 t586_leaf1 <= 1.001700999999e+06
 T586_R_f3_k2: Overall_Qual_calc - 1e+06 t586_leaf2 >= -999997
 T586_L_f3_k2: Overall_Qual_calc + 1e+06 t586_leaf2 <= 1.000006999999e+06
 T586_R_f20_k2: x_Gr_Liv_Area - 1e+06 t586_leaf2 >= -998299
 T586_R_f3_k3: Overall_Qual_calc - 1e+06 t586_leaf3 >= -999997
 T586_R_f3_k3: Overall_Qual_calc - 1e+06 t586_leaf3 >= -999993
 T586_L_f105_k3: const_Neighborhood_NWAmes + 1e+06 t586_leaf3
   <= 1.000000999999e+06
 T586_R_f3_k4: Overall_Qual_calc - 1e+06 t586_leaf4 >= -999997
 T586_R_f3_k4: Overall_Qual_calc - 1e+06 t586_leaf4 >= -999993
 T586_R_f105_k4: const_Neighborhood_NWAmes - 1e+06 t586_leaf4 >= -999999
 T587_onehot: t587_leaf0 + t587_leaf1 + t587_leaf2 + t587_leaf3
   + t587_leaf4 + t587_leaf5 = 1
 T587_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t587_leaf0
   <= 1.000000999999e+06
 T587_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t587_leaf0 <= 1.001473999999e+06
 T587_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t587_leaf1
   <= 1.000000999999e+06
 T587_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t587_leaf1 >= -998526
 T587_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t587_leaf2
   >= -999999
 T587_L_f5_k2: const_Year_Built + 1e+06 t587_leaf2 <= 1.001953999999e+06
 T587_L_f148_k2: const_Roof_Style_Gable + 1e+06 t587_leaf2
   <= 1.000000999999e+06
 T587_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t587_leaf3
   >= -999999
 T587_L_f5_k3: const_Year_Built + 1e+06 t587_leaf3 <= 1.001953999999e+06
 T587_R_f148_k3: const_Roof_Style_Gable - 1e+06 t587_leaf3 >= -999999
 T587_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t587_leaf4
   >= -999999
 T587_R_f5_k4: const_Year_Built - 1e+06 t587_leaf4 >= -998046
 T587_L_f44_k4: const_Yr_Sold + 1e+06 t587_leaf4 <= 1.002008999999e+06
 T587_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t587_leaf5
   >= -999999
 T587_R_f5_k5: const_Year_Built - 1e+06 t587_leaf5 >= -998046
 T587_R_f44_k5: const_Yr_Sold - 1e+06 t587_leaf5 >= -997991
 T588_onehot: t588_leaf0 + t588_leaf1 + t588_leaf2 + t588_leaf3
   + t588_leaf4 + t588_leaf5 + t588_leaf6 + t588_leaf7 = 1
 T588_L_f3_k0: Overall_Qual_calc + 1e+06 t588_leaf0 <= 1.000006999999e+06
 T588_L_f1_k0: const_Lot_Area + 1e+06 t588_leaf0 <= 1.008519999999e+06
 T588_L_f33_k0: x_Garage_Area + 1e+06 t588_leaf0 <= 1.000214999999e+06
 T588_L_f3_k1: Overall_Qual_calc + 1e+06 t588_leaf1 <= 1.000006999999e+06
 T588_L_f1_k1: const_Lot_Area + 1e+06 t588_leaf1 <= 1.008519999999e+06
 T588_R_f33_k1: x_Garage_Area - 1e+06 t588_leaf1 >= -999785
 T588_L_f3_k2: Overall_Qual_calc + 1e+06 t588_leaf2 <= 1.000006999999e+06
 T588_R_f1_k2: const_Lot_Area - 1e+06 t588_leaf2 >= -991480
 T588_L_f231_k2: central_air_no + 1e+06 t588_leaf2 <= 1.000000999999e+06
 T588_L_f3_k3: Overall_Qual_calc + 1e+06 t588_leaf3 <= 1.000006999999e+06
 T588_R_f1_k3: const_Lot_Area - 1e+06 t588_leaf3 >= -991480
 T588_R_f231_k3: central_air_no - 1e+06 t588_leaf3 >= -999999
 T588_R_f3_k4: Overall_Qual_calc - 1e+06 t588_leaf4 >= -999993
 T588_L_f6_k4: const_Year_Remod_Add + 1e+06 t588_leaf4
   <= 1.001992999999e+06
 T588_L_f297_k4: const_Sale_Condition_Normal + 1e+06 t588_leaf4
   <= 1.000000999999e+06
 T588_R_f3_k5: Overall_Qual_calc - 1e+06 t588_leaf5 >= -999993
 T588_L_f6_k5: const_Year_Remod_Add + 1e+06 t588_leaf5
   <= 1.001992999999e+06
 T588_R_f297_k5: const_Sale_Condition_Normal - 1e+06 t588_leaf5 >= -999999
 T588_R_f3_k6: Overall_Qual_calc - 1e+06 t588_leaf6 >= -999993
 T588_R_f6_k6: const_Year_Remod_Add - 1e+06 t588_leaf6 >= -998007
 T588_L_f96_k6: const_Neighborhood_Edwards + 1e+06 t588_leaf6
   <= 1.000000999999e+06
 T588_R_f3_k7: Overall_Qual_calc - 1e+06 t588_leaf7 >= -999993
 T588_R_f6_k7: const_Year_Remod_Add - 1e+06 t588_leaf7 >= -998007
 T588_R_f96_k7: const_Neighborhood_Edwards - 1e+06 t588_leaf7 >= -999999
 T589_onehot: t589_leaf0 = 1
 T590_onehot: t590_leaf0 + t590_leaf1 = 1
 T590_L_f20_k0: x_Gr_Liv_Area + 1e+06 t590_leaf0 <= 1.002632999999e+06
 T590_R_f20_k1: x_Gr_Liv_Area - 1e+06 t590_leaf1 >= -997367
 T591_onehot: t591_leaf0 = 1
 T592_onehot: t592_leaf0 = 1
 T593_onehot: t593_leaf0 = 1
 T594_onehot: t594_leaf0 = 1
 T595_onehot: t595_leaf0 + t595_leaf1 + t595_leaf2 + t595_leaf3
   + t595_leaf4 = 1
 T595_L_f5_k0: const_Year_Built + 1e+06 t595_leaf0 <= 1.001959999999e+06
 T595_L_f18_k0: const_2nd_Flr_SF + 1e+06 t595_leaf0 <= 1.000471999999e+06
 T595_L_f5_k1: const_Year_Built + 1e+06 t595_leaf1 <= 1.001959999999e+06
 T595_R_f18_k1: const_2nd_Flr_SF - 1e+06 t595_leaf1 >= -999528
 T595_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t595_leaf1 <= 1.000189999999e+06
 T595_L_f5_k2: const_Year_Built + 1e+06 t595_leaf2 <= 1.001959999999e+06
 T595_R_f18_k2: const_2nd_Flr_SF - 1e+06 t595_leaf2 >= -999528
 T595_R_f14_k2: x_Bsmt_Unf_SF - 1e+06 t595_leaf2 >= -999810
 T595_R_f5_k3: const_Year_Built - 1e+06 t595_leaf3 >= -998040
 T595_L_f0_k3: const_Lot_Frontage + 1e+06 t595_leaf3 <= 1.000114999999e+06
 T595_R_f5_k4: const_Year_Built - 1e+06 t595_leaf4 >= -998040
 T595_R_f0_k4: const_Lot_Frontage - 1e+06 t595_leaf4 >= -999885
 T596_onehot: t596_leaf0 = 1
 T597_onehot: t597_leaf0 + t597_leaf1 + t597_leaf2 + t597_leaf3
   + t597_leaf4 = 1
 T597_L_f194_k0: x_mvt_is_BrkCmn + 1e+06 t597_leaf0 <= 1.000000999999e+06
 T597_L_f30_k0: x_Fireplace_Qu + 1e+06 t597_leaf0 <= 1.000001999999e+06
 T597_L_f20_k0: x_Gr_Liv_Area + 1e+06 t597_leaf0 <= 1.002236999999e+06
 T597_L_f194_k1: x_mvt_is_BrkCmn + 1e+06 t597_leaf1 <= 1.000000999999e+06
 T597_L_f30_k1: x_Fireplace_Qu + 1e+06 t597_leaf1 <= 1.000001999999e+06
 T597_R_f20_k1: x_Gr_Liv_Area - 1e+06 t597_leaf1 >= -997763
 T597_L_f194_k2: x_mvt_is_BrkCmn + 1e+06 t597_leaf2 <= 1.000000999999e+06
 T597_R_f30_k2: x_Fireplace_Qu - 1e+06 t597_leaf2 >= -999998
 T597_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t597_leaf2
   <= 1.000001999999e+06
 T597_L_f194_k3: x_mvt_is_BrkCmn + 1e+06 t597_leaf3 <= 1.000000999999e+06
 T597_R_f30_k3: x_Fireplace_Qu - 1e+06 t597_leaf3 >= -999998
 T597_R_f21_k3: const_Bsmt_Full_Bath - 1e+06 t597_leaf3 >= -999998
 T597_R_f194_k4: x_mvt_is_BrkCmn - 1e+06 t597_leaf4 >= -999999
 T598_onehot: t598_leaf0 + t598_leaf1 = 1
 T598_L_f5_k0: const_Year_Built + 1e+06 t598_leaf0 <= 1.001914999999e+06
 T598_R_f5_k1: const_Year_Built - 1e+06 t598_leaf1 >= -998085
 T599_onehot: t599_leaf0 = 1
 T600_onehot: t600_leaf0 + t600_leaf1 + t600_leaf2 + t600_leaf3 = 1
 T600_L_f216_k0: x_b1_is_Rec + 1e+06 t600_leaf0 <= 1.000000999999e+06
 T600_R_f216_k1: x_b1_is_Rec - 1e+06 t600_leaf1 >= -999999
 T600_L_f192_k1: x_ex2_is_Wd_Sdng + 1e+06 t600_leaf1 <= 1.000000999999e+06
 T600_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t600_leaf1
   <= 1.001965999999e+06
 T600_R_f216_k2: x_b1_is_Rec - 1e+06 t600_leaf2 >= -999999
 T600_L_f192_k2: x_ex2_is_Wd_Sdng + 1e+06 t600_leaf2 <= 1.000000999999e+06
 T600_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t600_leaf2 >= -998034
 T600_R_f216_k3: x_b1_is_Rec - 1e+06 t600_leaf3 >= -999999
 T600_R_f192_k3: x_ex2_is_Wd_Sdng - 1e+06 t600_leaf3 >= -999999
 T601_onehot: t601_leaf0 + t601_leaf1 + t601_leaf2 + t601_leaf3 = 1
 T601_L_f216_k0: x_b1_is_Rec + 1e+06 t601_leaf0 <= 1.000000999999e+06
 T601_R_f216_k1: x_b1_is_Rec - 1e+06 t601_leaf1 >= -999999
 T601_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t601_leaf1 <= 1.000365999999e+06
 T601_R_f216_k2: x_b1_is_Rec - 1e+06 t601_leaf2 >= -999999
 T601_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t601_leaf2 >= -999634
 T601_L_f20_k2: x_Gr_Liv_Area + 1e+06 t601_leaf2 <= 1.000964999999e+06
 T601_R_f216_k3: x_b1_is_Rec - 1e+06 t601_leaf3 >= -999999
 T601_R_f12_k3: x_BsmtFin_SF_1 - 1e+06 t601_leaf3 >= -999634
 T601_R_f20_k3: x_Gr_Liv_Area - 1e+06 t601_leaf3 >= -999035
 T602_onehot: t602_leaf0 = 1
 T603_onehot: t603_leaf0 + t603_leaf1 = 1
 T603_L_f8_k0: x_Exter_Qual + 1e+06 t603_leaf0 <= 1.000002999999e+06
 T603_R_f8_k1: x_Exter_Qual - 1e+06 t603_leaf1 >= -999997
 T604_onehot: t604_leaf0 = 1
 T605_onehot: t605_leaf0 + t605_leaf1 + t605_leaf2 + t605_leaf3 = 1
 T605_L_f17_k0: x_1st_Flr_SF + 1e+06 t605_leaf0 <= 1.002514999999e+06
 T605_L_f0_k0: const_Lot_Frontage + 1e+06 t605_leaf0 <= 1.000109999999e+06
 T605_L_f32_k0: const_Garage_Cars + 1e+06 t605_leaf0 <= 1.000002999999e+06
 T605_L_f17_k1: x_1st_Flr_SF + 1e+06 t605_leaf1 <= 1.002514999999e+06
 T605_L_f0_k1: const_Lot_Frontage + 1e+06 t605_leaf1 <= 1.000109999999e+06
 T605_R_f32_k1: const_Garage_Cars - 1e+06 t605_leaf1 >= -999997
 T605_L_f17_k2: x_1st_Flr_SF + 1e+06 t605_leaf2 <= 1.002514999999e+06
 T605_R_f0_k2: const_Lot_Frontage - 1e+06 t605_leaf2 >= -999890
 T605_R_f17_k3: x_1st_Flr_SF - 1e+06 t605_leaf3 >= -997485
 T606_onehot: t606_leaf0 = 1
 T607_onehot: t607_leaf0 + t607_leaf1 = 1
 T607_L_f158_k0: x_roof_matl_is_Tar&Grv + 1e+06 t607_leaf0
   <= 1.000000999999e+06
 T607_R_f158_k1: x_roof_matl_is_Tar&Grv - 1e+06 t607_leaf1 >= -999999
 T608_onehot: t608_leaf0 + t608_leaf1 = 1
 T608_L_f60_k0: const_MS_SubClass_190 + 1e+06 t608_leaf0
   <= 1.000000999999e+06
 T608_R_f60_k1: const_MS_SubClass_190 - 1e+06 t608_leaf1 >= -999999
 T609_onehot: t609_leaf0 = 1
 T610_onehot: t610_leaf0 = 1
 T611_onehot: t611_leaf0 + t611_leaf1 = 1
 T611_L_f274_k0: const_Mo_Sold_4 + 1e+06 t611_leaf0 <= 1.000000999999e+06
 T611_R_f274_k1: const_Mo_Sold_4 - 1e+06 t611_leaf1 >= -999999
 T612_onehot: t612_leaf0 = 1
 T613_onehot: t613_leaf0 + t613_leaf1 = 1
 T613_L_f24_k0: x_Half_Bath + 1e+06 t613_leaf0 <= 1.000000999999e+06
 T613_R_f24_k1: x_Half_Bath - 1e+06 t613_leaf1 >= -999999
 T614_onehot: t614_leaf0 = 1
 T615_onehot: t615_leaf0 = 1
 T616_onehot: t616_leaf0 + t616_leaf1 + t616_leaf2 + t616_leaf3 = 1
 T616_L_f85_k0: const_Lot_Config_Inside + 1e+06 t616_leaf0
   <= 1.000000999999e+06
 T616_R_f85_k1: const_Lot_Config_Inside - 1e+06 t616_leaf1 >= -999999
 T616_L_f1_k1: const_Lot_Area + 1e+06 t616_leaf1 <= 1.010799999999e+06
 T616_R_f85_k2: const_Lot_Config_Inside - 1e+06 t616_leaf2 >= -999999
 T616_R_f1_k2: const_Lot_Area - 1e+06 t616_leaf2 >= -989200
 T616_L_f33_k2: x_Garage_Area + 1e+06 t616_leaf2 <= 1.000365999999e+06
 T616_R_f85_k3: const_Lot_Config_Inside - 1e+06 t616_leaf3 >= -999999
 T616_R_f1_k3: const_Lot_Area - 1e+06 t616_leaf3 >= -989200
 T616_R_f33_k3: x_Garage_Area - 1e+06 t616_leaf3 >= -999634
 T617_onehot: t617_leaf0 + t617_leaf1 + t617_leaf2 = 1
 T617_L_f3_k0: Overall_Qual_calc + 1e+06 t617_leaf0 <= 1.000004999999e+06
 T617_L_f80_k0: const_Land_Contour_Lvl + 1e+06 t617_leaf0
   <= 1.000000999999e+06
 T617_L_f3_k1: Overall_Qual_calc + 1e+06 t617_leaf1 <= 1.000004999999e+06
 T617_R_f80_k1: const_Land_Contour_Lvl - 1e+06 t617_leaf1 >= -999999
 T617_R_f3_k2: Overall_Qual_calc - 1e+06 t617_leaf2 >= -999995
 T618_onehot: t618_leaf0 + t618_leaf1 = 1
 T618_L_f238_k0: const_FunctioNo_aplical_Maj1 + 1e+06 t618_leaf0
   <= 1.000000999999e+06
 T618_R_f238_k1: const_FunctioNo_aplical_Maj1 - 1e+06 t618_leaf1
   >= -999999
 T619_onehot: t619_leaf0 + t619_leaf1 + t619_leaf2 + t619_leaf3
   + t619_leaf4 = 1
 T619_L_f29_k0: const_Fireplaces + 1e+06 t619_leaf0 <= 1.000000999999e+06
 T619_L_f33_k0: x_Garage_Area + 1e+06 t619_leaf0 <= 1.000623999999e+06
 T619_L_f29_k1: const_Fireplaces + 1e+06 t619_leaf1 <= 1.000000999999e+06
 T619_R_f33_k1: x_Garage_Area - 1e+06 t619_leaf1 >= -999376
 T619_R_f29_k2: const_Fireplaces - 1e+06 t619_leaf2 >= -999999
 T619_L_f4_k2: const_Overall_Cond + 1e+06 t619_leaf2 <= 1.000004999999e+06
 T619_L_f33_k2: x_Garage_Area + 1e+06 t619_leaf2 <= 1.000643999999e+06
 T619_R_f29_k3: const_Fireplaces - 1e+06 t619_leaf3 >= -999999
 T619_L_f4_k3: const_Overall_Cond + 1e+06 t619_leaf3 <= 1.000004999999e+06
 T619_R_f33_k3: x_Garage_Area - 1e+06 t619_leaf3 >= -999356
 T619_R_f29_k4: const_Fireplaces - 1e+06 t619_leaf4 >= -999999
 T619_R_f4_k4: const_Overall_Cond - 1e+06 t619_leaf4 >= -999995
 T620_onehot: t620_leaf0 + t620_leaf1 = 1
 T620_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t620_leaf0 <= 1.000672999999e+06
 T620_R_f7_k1: x_Mas_Vnr_Area - 1e+06 t620_leaf1 >= -999327
 T621_onehot: t621_leaf0 + t621_leaf1 + t621_leaf2 + t621_leaf3 = 1
 T621_L_f5_k0: const_Year_Built + 1e+06 t621_leaf0 <= 1.002004999999e+06
 T621_R_f5_k1: const_Year_Built - 1e+06 t621_leaf1 >= -997995
 T621_L_f1_k1: const_Lot_Area + 1e+06 t621_leaf1 <= 1.008157999999e+06
 T621_L_f0_k1: const_Lot_Frontage + 1e+06 t621_leaf1 <= 1.000044999999e+06
 T621_R_f5_k2: const_Year_Built - 1e+06 t621_leaf2 >= -997995
 T621_L_f1_k2: const_Lot_Area + 1e+06 t621_leaf2 <= 1.008157999999e+06
 T621_R_f0_k2: const_Lot_Frontage - 1e+06 t621_leaf2 >= -999955
 T621_R_f5_k3: const_Year_Built - 1e+06 t621_leaf3 >= -997995
 T621_R_f1_k3: const_Lot_Area - 1e+06 t621_leaf3 >= -991842
 T622_onehot: t622_leaf0 + t622_leaf1 + t622_leaf2 + t622_leaf3 = 1
 T622_L_f20_k0: x_Gr_Liv_Area + 1e+06 t622_leaf0 <= 1.002648999999e+06
 T622_L_f29_k0: const_Fireplaces + 1e+06 t622_leaf0 <= 1.000000999999e+06
 T622_L_f20_k1: x_Gr_Liv_Area + 1e+06 t622_leaf1 <= 1.002648999999e+06
 T622_R_f29_k1: const_Fireplaces - 1e+06 t622_leaf1 >= -999999
 T622_L_f28_k1: x_TotRms_AbvGrd + 1e+06 t622_leaf1 <= 1.000004999999e+06
 T622_L_f20_k2: x_Gr_Liv_Area + 1e+06 t622_leaf2 <= 1.002648999999e+06
 T622_R_f29_k2: const_Fireplaces - 1e+06 t622_leaf2 >= -999999
 T622_R_f28_k2: x_TotRms_AbvGrd - 1e+06 t622_leaf2 >= -999995
 T622_R_f20_k3: x_Gr_Liv_Area - 1e+06 t622_leaf3 >= -997351
 T623_onehot: t623_leaf0 + t623_leaf1 + t623_leaf2 + t623_leaf3
   + t623_leaf4 + t623_leaf5 = 1
 T623_L_f44_k0: const_Yr_Sold + 1e+06 t623_leaf0 <= 1.002008999999e+06
 T623_L_f77_k0: const_Land_Contour_Bnk + 1e+06 t623_leaf0
   <= 1.000000999999e+06
 T623_L_f44_k1: const_Yr_Sold + 1e+06 t623_leaf1 <= 1.002008999999e+06
 T623_R_f77_k1: const_Land_Contour_Bnk - 1e+06 t623_leaf1 >= -999999
 T623_L_f33_k1: x_Garage_Area + 1e+06 t623_leaf1 <= 1.000576999999e+06
 T623_L_f44_k2: const_Yr_Sold + 1e+06 t623_leaf2 <= 1.002008999999e+06
 T623_R_f77_k2: const_Land_Contour_Bnk - 1e+06 t623_leaf2 >= -999999
 T623_R_f33_k2: x_Garage_Area - 1e+06 t623_leaf2 >= -999423
 T623_R_f44_k3: const_Yr_Sold - 1e+06 t623_leaf3 >= -997991
 T623_L_f293_k3: const_Sale_Condition_Abnorml + 1e+06 t623_leaf3
   <= 1.000000999999e+06
 T623_R_f44_k4: const_Yr_Sold - 1e+06 t623_leaf4 >= -997991
 T623_R_f293_k4: const_Sale_Condition_Abnorml - 1e+06 t623_leaf4
   >= -999999
 T623_L_f81_k4: const_Lot_Config_Corner + 1e+06 t623_leaf4
   <= 1.000000999999e+06
 T623_R_f44_k5: const_Yr_Sold - 1e+06 t623_leaf5 >= -997991
 T623_R_f293_k5: const_Sale_Condition_Abnorml - 1e+06 t623_leaf5
   >= -999999
 T623_R_f81_k5: const_Lot_Config_Corner - 1e+06 t623_leaf5 >= -999999
 T624_onehot: t624_leaf0 + t624_leaf1 = 1
 T624_L_f79_k0: const_Land_Contour_Low + 1e+06 t624_leaf0
   <= 1.000000999999e+06
 T624_R_f79_k1: const_Land_Contour_Low - 1e+06 t624_leaf1 >= -999999
 T625_onehot: t625_leaf0 + t625_leaf1 + t625_leaf2 = 1
 T625_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t625_leaf0
   <= 1.000244999999e+06
 T625_L_f0_k0: const_Lot_Frontage + 1e+06 t625_leaf0 <= 1.000065999999e+06
 T625_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t625_leaf1
   <= 1.000244999999e+06
 T625_R_f0_k1: const_Lot_Frontage - 1e+06 t625_leaf1 >= -999934
 T625_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t625_leaf2 >= -999755
 T626_onehot: t626_leaf0 + t626_leaf1 + t626_leaf2 + t626_leaf3 = 1
 T626_L_f10_k0: const_Bsmt_Qual + 1e+06 t626_leaf0 <= 1.000001999999e+06
 T626_L_f4_k0: const_Overall_Cond + 1e+06 t626_leaf0 <= 1.000005999999e+06
 T626_L_f85_k0: const_Lot_Config_Inside + 1e+06 t626_leaf0
   <= 1.000000999999e+06
 T626_L_f10_k1: const_Bsmt_Qual + 1e+06 t626_leaf1 <= 1.000001999999e+06
 T626_L_f4_k1: const_Overall_Cond + 1e+06 t626_leaf1 <= 1.000005999999e+06
 T626_R_f85_k1: const_Lot_Config_Inside - 1e+06 t626_leaf1 >= -999999
 T626_L_f10_k2: const_Bsmt_Qual + 1e+06 t626_leaf2 <= 1.000001999999e+06
 T626_R_f4_k2: const_Overall_Cond - 1e+06 t626_leaf2 >= -999994
 T626_R_f10_k3: const_Bsmt_Qual - 1e+06 t626_leaf3 >= -999998
 T627_onehot: t627_leaf0 + t627_leaf1 = 1
 T627_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t627_leaf0
   <= 1.002041999999e+06
 T627_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t627_leaf1 >= -997958
 T628_onehot: t628_leaf0 = 1
 T629_onehot: t629_leaf0 = 1
 T630_onehot: t630_leaf0 = 1
 T631_onehot: t631_leaf0 = 1
 T632_onehot: t632_leaf0 + t632_leaf1 + t632_leaf2 + t632_leaf3 = 1
 T632_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t632_leaf0 <= 1.000000999999e+06
 T632_L_f5_k0: const_Year_Built + 1e+06 t632_leaf0 <= 1.001914999999e+06
 T632_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t632_leaf1 <= 1.000000999999e+06
 T632_R_f5_k1: const_Year_Built - 1e+06 t632_leaf1 >= -998085
 T632_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t632_leaf2 >= -999999
 T632_L_f1_k2: const_Lot_Area + 1e+06 t632_leaf2 <= 1.009604999999e+06
 T632_R_f161_k3: x_ex1_is_AsbShng - 1e+06 t632_leaf3 >= -999999
 T632_R_f1_k3: const_Lot_Area - 1e+06 t632_leaf3 >= -990395
 T633_onehot: t633_leaf0 = 1
 T634_onehot: t634_leaf0 = 1
 T635_onehot: t635_leaf0 = 1
 T636_onehot: t636_leaf0 = 1
 T637_onehot: t637_leaf0 = 1
 T638_onehot: t638_leaf0 = 1
 T639_onehot: t639_leaf0 = 1
 T640_onehot: t640_leaf0 + t640_leaf1 + t640_leaf2 + t640_leaf3 = 1
 T640_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t640_leaf0 <= 1.000000999999e+06
 T640_L_f4_k0: const_Overall_Cond + 1e+06 t640_leaf0 <= 1.000003999999e+06
 T640_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t640_leaf1 <= 1.000000999999e+06
 T640_R_f4_k1: const_Overall_Cond - 1e+06 t640_leaf1 >= -999996
 T640_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t640_leaf2 >= -999999
 T640_L_f20_k2: x_Gr_Liv_Area + 1e+06 t640_leaf2 <= 1.000882999999e+06
 T640_R_f161_k3: x_ex1_is_AsbShng - 1e+06 t640_leaf3 >= -999999
 T640_R_f20_k3: x_Gr_Liv_Area - 1e+06 t640_leaf3 >= -999117
 T641_onehot: t641_leaf0 = 1
 T642_onehot: t642_leaf0 = 1
 T643_onehot: t643_leaf0 = 1
 T644_onehot: t644_leaf0 = 1
 T645_onehot: t645_leaf0 = 1
 T646_onehot: t646_leaf0 + t646_leaf1 + t646_leaf2 = 1
 T646_L_f4_k0: const_Overall_Cond + 1e+06 t646_leaf0 <= 1.000005999999e+06
 T646_R_f4_k1: const_Overall_Cond - 1e+06 t646_leaf1 >= -999994
 T646_L_f23_k1: x_Full_Bath + 1e+06 t646_leaf1 <= 1.000001999999e+06
 T646_R_f4_k2: const_Overall_Cond - 1e+06 t646_leaf2 >= -999994
 T646_R_f23_k2: x_Full_Bath - 1e+06 t646_leaf2 >= -999998
 T647_onehot: t647_leaf0 + t647_leaf1 + t647_leaf2 = 1
 T647_L_f207_k0: const_Bsmt_Exposure_Gd + 1e+06 t647_leaf0
   <= 1.000000999999e+06
 T647_L_f0_k0: const_Lot_Frontage + 1e+06 t647_leaf0 <= 1.000075999999e+06
 T647_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t647_leaf1
   <= 1.000000999999e+06
 T647_R_f0_k1: const_Lot_Frontage - 1e+06 t647_leaf1 >= -999924
 T647_R_f207_k2: const_Bsmt_Exposure_Gd - 1e+06 t647_leaf2 >= -999999
 T648_onehot: t648_leaf0 = 1
 T649_onehot: t649_leaf0 + t649_leaf1 = 1
 T649_L_f219_k0: x_b2_is_BLQ + 1e+06 t649_leaf0 <= 1.000000999999e+06
 T649_R_f219_k1: x_b2_is_BLQ - 1e+06 t649_leaf1 >= -999999
 T650_onehot: t650_leaf0 = 1
 T651_onehot: t651_leaf0 = 1
 T652_onehot: t652_leaf0 + t652_leaf1 + t652_leaf2 = 1
 T652_L_f37_k0: x_Open_Porch_SF + 1e+06 t652_leaf0 <= 1.000287999999e+06
 T652_L_f0_k0: const_Lot_Frontage + 1e+06 t652_leaf0 <= 1.000130999999e+06
 T652_L_f37_k1: x_Open_Porch_SF + 1e+06 t652_leaf1 <= 1.000287999999e+06
 T652_R_f0_k1: const_Lot_Frontage - 1e+06 t652_leaf1 >= -999869
 T652_R_f37_k2: x_Open_Porch_SF - 1e+06 t652_leaf2 >= -999712
 T653_onehot: t653_leaf0 + t653_leaf1 = 1
 T653_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t653_leaf0 <= 1.001432999999e+06
 T653_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t653_leaf1 >= -998567
 T654_onehot: t654_leaf0 = 1
 T655_onehot: t655_leaf0 + t655_leaf1 + t655_leaf2 = 1
 T655_L_f62_k0: const_MS_Zoning_C_(all) + 1e+06 t655_leaf0
   <= 1.000000999999e+06
 T655_R_f62_k1: const_MS_Zoning_C_(all) - 1e+06 t655_leaf1 >= -999999
 T655_L_f139_k1: const_House_Style_1.5Fin + 1e+06 t655_leaf1
   <= 1.000000999999e+06
 T655_R_f62_k2: const_MS_Zoning_C_(all) - 1e+06 t655_leaf2 >= -999999
 T655_R_f139_k2: const_House_Style_1.5Fin - 1e+06 t655_leaf2 >= -999999
 T656_onehot: t656_leaf0 + t656_leaf1 + t656_leaf2 = 1
 T656_L_f33_k0: x_Garage_Area + 1e+06 t656_leaf0 <= 1.000829999999e+06
 T656_L_f19_k0: const_Low_Qual_Fin_SF + 1e+06 t656_leaf0
   <= 1.000107999999e+06
 T656_L_f33_k1: x_Garage_Area + 1e+06 t656_leaf1 <= 1.000829999999e+06
 T656_R_f19_k1: const_Low_Qual_Fin_SF - 1e+06 t656_leaf1 >= -999892
 T656_R_f33_k2: x_Garage_Area - 1e+06 t656_leaf2 >= -999170
 T657_onehot: t657_leaf0 + t657_leaf1 = 1
 T657_L_f290_k0: const_Sale_Type_Oth + 1e+06 t657_leaf0
   <= 1.000000999999e+06
 T657_R_f290_k1: const_Sale_Type_Oth - 1e+06 t657_leaf1 >= -999999
 T658_onehot: t658_leaf0 = 1
 T659_onehot: t659_leaf0 + t659_leaf1 + t659_leaf2 + t659_leaf3
   + t659_leaf4 = 1
 T659_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t659_leaf0
   <= 1.001942999999e+06
 T659_L_f211_k0: x_b1_is_ALQ + 1e+06 t659_leaf0 <= 1.000000999999e+06
 T659_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t659_leaf0 <= 1.000007999999e+06
 T659_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t659_leaf1
   <= 1.001942999999e+06
 T659_L_f211_k1: x_b1_is_ALQ + 1e+06 t659_leaf1 <= 1.000000999999e+06
 T659_R_f28_k1: x_TotRms_AbvGrd - 1e+06 t659_leaf1 >= -999992
 T659_L_f31_k2: const_Garage_Yr_Blt + 1e+06 t659_leaf2
   <= 1.001942999999e+06
 T659_R_f211_k2: x_b1_is_ALQ - 1e+06 t659_leaf2 >= -999999
 T659_R_f31_k3: const_Garage_Yr_Blt - 1e+06 t659_leaf3 >= -998057
 T659_L_f34_k3: x_Garage_Qual + 1e+06 t659_leaf3 <= 1.000001999999e+06
 T659_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t659_leaf4 >= -998057
 T659_R_f34_k4: x_Garage_Qual - 1e+06 t659_leaf4 >= -999998
 T660_onehot: t660_leaf0 + t660_leaf1 + t660_leaf2 + t660_leaf3 = 1
 T660_L_f0_k0: const_Lot_Frontage + 1e+06 t660_leaf0 <= 1.000139999999e+06
 T660_L_f37_k0: x_Open_Porch_SF + 1e+06 t660_leaf0 <= 1.000121999999e+06
 T660_L_f1_k0: const_Lot_Area + 1e+06 t660_leaf0 <= 1.011764999999e+06
 T660_L_f0_k1: const_Lot_Frontage + 1e+06 t660_leaf1 <= 1.000139999999e+06
 T660_L_f37_k1: x_Open_Porch_SF + 1e+06 t660_leaf1 <= 1.000121999999e+06
 T660_R_f1_k1: const_Lot_Area - 1e+06 t660_leaf1 >= -988235
 T660_L_f0_k2: const_Lot_Frontage + 1e+06 t660_leaf2 <= 1.000139999999e+06
 T660_R_f37_k2: x_Open_Porch_SF - 1e+06 t660_leaf2 >= -999878
 T660_R_f0_k3: const_Lot_Frontage - 1e+06 t660_leaf3 >= -999860
 T661_onehot: t661_leaf0 + t661_leaf1 + t661_leaf2 + t661_leaf3
   + t661_leaf4 = 1
 T661_L_f17_k0: x_1st_Flr_SF + 1e+06 t661_leaf0 <= 1.000575999999e+06
 T661_R_f17_k1: x_1st_Flr_SF - 1e+06 t661_leaf1 >= -999424
 T661_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t661_leaf1 <= 1.000000999999e+06
 T661_L_f34_k1: x_Garage_Qual + 1e+06 t661_leaf1 <= 1.000001999999e+06
 T661_R_f17_k2: x_1st_Flr_SF - 1e+06 t661_leaf2 >= -999424
 T661_L_f161_k2: x_ex1_is_AsbShng + 1e+06 t661_leaf2 <= 1.000000999999e+06
 T661_R_f34_k2: x_Garage_Qual - 1e+06 t661_leaf2 >= -999998
 T661_R_f17_k3: x_1st_Flr_SF - 1e+06 t661_leaf3 >= -999424
 T661_R_f161_k3: x_ex1_is_AsbShng - 1e+06 t661_leaf3 >= -999999
 T661_L_f14_k3: x_Bsmt_Unf_SF + 1e+06 t661_leaf3 <= 1.000683999999e+06
 T661_R_f17_k4: x_1st_Flr_SF - 1e+06 t661_leaf4 >= -999424
 T661_R_f161_k4: x_ex1_is_AsbShng - 1e+06 t661_leaf4 >= -999999
 T661_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t661_leaf4 >= -999316
 T662_onehot: t662_leaf0 = 1
 T663_onehot: t663_leaf0 + t663_leaf1 + t663_leaf2 + t663_leaf3
   + t663_leaf4 + t663_leaf5 + t663_leaf6 = 1
 T663_L_f5_k0: const_Year_Built + 1e+06 t663_leaf0 <= 1.001938999999e+06
 T663_L_f119_k0: const_Condition_1_Norm + 1e+06 t663_leaf0
   <= 1.000000999999e+06
 T663_L_f5_k1: const_Year_Built + 1e+06 t663_leaf1 <= 1.001938999999e+06
 T663_R_f119_k1: const_Condition_1_Norm - 1e+06 t663_leaf1 >= -999999
 T663_L_f5_k1: const_Year_Built + 1e+06 t663_leaf1 <= 1.001914999999e+06
 T663_L_f5_k2: const_Year_Built + 1e+06 t663_leaf2 <= 1.001938999999e+06
 T663_R_f119_k2: const_Condition_1_Norm - 1e+06 t663_leaf2 >= -999999
 T663_R_f5_k2: const_Year_Built - 1e+06 t663_leaf2 >= -998085
 T663_R_f5_k3: const_Year_Built - 1e+06 t663_leaf3 >= -998061
 T663_L_f237_k3: x_elect_is_SBrkr + 1e+06 t663_leaf3 <= 1.000000999999e+06
 T663_L_f86_k3: const_Land_Slope_Gtl + 1e+06 t663_leaf3
   <= 1.000000999999e+06
 T663_R_f5_k4: const_Year_Built - 1e+06 t663_leaf4 >= -998061
 T663_L_f237_k4: x_elect_is_SBrkr + 1e+06 t663_leaf4 <= 1.000000999999e+06
 T663_R_f86_k4: const_Land_Slope_Gtl - 1e+06 t663_leaf4 >= -999999
 T663_R_f5_k5: const_Year_Built - 1e+06 t663_leaf5 >= -998061
 T663_R_f237_k5: x_elect_is_SBrkr - 1e+06 t663_leaf5 >= -999999
 T663_L_f231_k5: central_air_no + 1e+06 t663_leaf5 <= 1.000000999999e+06
 T663_R_f5_k6: const_Year_Built - 1e+06 t663_leaf6 >= -998061
 T663_R_f237_k6: x_elect_is_SBrkr - 1e+06 t663_leaf6 >= -999999
 T663_R_f231_k6: central_air_no - 1e+06 t663_leaf6 >= -999999
 T664_onehot: t664_leaf0 + t664_leaf1 + t664_leaf2 = 1
 T664_L_f1_k0: const_Lot_Area + 1e+06 t664_leaf0 <= 1.005999999999e+06
 T664_R_f1_k1: const_Lot_Area - 1e+06 t664_leaf1 >= -994000
 T664_L_f3_k1: Overall_Qual_calc + 1e+06 t664_leaf1 <= 1.000004999999e+06
 T664_R_f1_k2: const_Lot_Area - 1e+06 t664_leaf2 >= -994000
 T664_R_f3_k2: Overall_Qual_calc - 1e+06 t664_leaf2 >= -999995
 T665_onehot: t665_leaf0 + t665_leaf1 + t665_leaf2 = 1
 T665_L_f213_k0: x_b1_is_GLQ + 1e+06 t665_leaf0 <= 1.000000999999e+06
 T665_L_f1_k0: const_Lot_Area + 1e+06 t665_leaf0 <= 1.019521999999e+06
 T665_L_f213_k1: x_b1_is_GLQ + 1e+06 t665_leaf1 <= 1.000000999999e+06
 T665_R_f1_k1: const_Lot_Area - 1e+06 t665_leaf1 >= -980478
 T665_R_f213_k2: x_b1_is_GLQ - 1e+06 t665_leaf2 >= -999999
 T666_onehot: t666_leaf0 + t666_leaf1 + t666_leaf2 = 1
 T666_L_f114_k0: const_Neighborhood_StoneBr + 1e+06 t666_leaf0
   <= 1.000000999999e+06
 T666_R_f114_k1: const_Neighborhood_StoneBr - 1e+06 t666_leaf1 >= -999999
 T666_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t666_leaf1
   <= 1.001341999999e+06
 T666_R_f114_k2: const_Neighborhood_StoneBr - 1e+06 t666_leaf2 >= -999999
 T666_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t666_leaf2 >= -998658
 T667_onehot: t667_leaf0 + t667_leaf1 + t667_leaf2 + t667_leaf3 = 1
 T667_L_f20_k0: x_Gr_Liv_Area + 1e+06 t667_leaf0 <= 1.000882999999e+06
 T667_L_f33_k0: x_Garage_Area + 1e+06 t667_leaf0 <= 1.000285999999e+06
 T667_L_f20_k1: x_Gr_Liv_Area + 1e+06 t667_leaf1 <= 1.000882999999e+06
 T667_R_f33_k1: x_Garage_Area - 1e+06 t667_leaf1 >= -999714
 T667_L_f161_k1: x_ex1_is_AsbShng + 1e+06 t667_leaf1 <= 1.000000999999e+06
 T667_L_f20_k2: x_Gr_Liv_Area + 1e+06 t667_leaf2 <= 1.000882999999e+06
 T667_R_f33_k2: x_Garage_Area - 1e+06 t667_leaf2 >= -999714
 T667_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t667_leaf2 >= -999999
 T667_R_f20_k3: x_Gr_Liv_Area - 1e+06 t667_leaf3 >= -999117
 T668_onehot: t668_leaf0 + t668_leaf1 + t668_leaf2 + t668_leaf3 = 1
 T668_L_f33_k0: x_Garage_Area + 1e+06 t668_leaf0 <= 1.000643999999e+06
 T668_R_f33_k1: x_Garage_Area - 1e+06 t668_leaf1 >= -999356
 T668_L_f96_k1: const_Neighborhood_Edwards + 1e+06 t668_leaf1
   <= 1.000000999999e+06
 T668_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t668_leaf1
   <= 1.001941999999e+06
 T668_R_f33_k2: x_Garage_Area - 1e+06 t668_leaf2 >= -999356
 T668_L_f96_k2: const_Neighborhood_Edwards + 1e+06 t668_leaf2
   <= 1.000000999999e+06
 T668_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t668_leaf2 >= -998058
 T668_R_f33_k3: x_Garage_Area - 1e+06 t668_leaf3 >= -999356
 T668_R_f96_k3: const_Neighborhood_Edwards - 1e+06 t668_leaf3 >= -999999
 T669_onehot: t669_leaf0 = 1
 T670_onehot: t670_leaf0 = 1
 T671_onehot: t671_leaf0 + t671_leaf1 = 1
 T671_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t671_leaf0 <= 1.000000999999e+06
 T671_R_f164_k1: x_ex1_is_BrkFace - 1e+06 t671_leaf1 >= -999999
 T672_onehot: t672_leaf0 = 1
 T673_onehot: t673_leaf0 + t673_leaf1 + t673_leaf2 + t673_leaf3
   + t673_leaf4 = 1
 T673_L_f11_k0: x_Bsmt_Cond + 1e+06 t673_leaf0 <= 1.000001999999e+06
 T673_L_f31_k0: const_Garage_Yr_Blt + 1e+06 t673_leaf0
   <= 1.001973999999e+06
 T673_L_f214_k0: x_b1_is_LwQ + 1e+06 t673_leaf0 <= 1.000000999999e+06
 T673_L_f11_k1: x_Bsmt_Cond + 1e+06 t673_leaf1 <= 1.000001999999e+06
 T673_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t673_leaf1
   <= 1.001973999999e+06
 T673_R_f214_k1: x_b1_is_LwQ - 1e+06 t673_leaf1 >= -999999
 T673_L_f11_k2: x_Bsmt_Cond + 1e+06 t673_leaf2 <= 1.000001999999e+06
 T673_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t673_leaf2 >= -998026
 T673_R_f11_k3: x_Bsmt_Cond - 1e+06 t673_leaf3 >= -999998
 T673_L_f1_k3: const_Lot_Area + 1e+06 t673_leaf3 <= 1.004499999999e+06
 T673_R_f11_k4: x_Bsmt_Cond - 1e+06 t673_leaf4 >= -999998
 T673_R_f1_k4: const_Lot_Area - 1e+06 t673_leaf4 >= -995500
 T674_onehot: t674_leaf0 + t674_leaf1 = 1
 T674_L_f3_k0: Overall_Qual_calc + 1e+06 t674_leaf0 <= 1.000009999999e+06
 T674_R_f3_k1: Overall_Qual_calc - 1e+06 t674_leaf1 >= -999990
 T675_onehot: t675_leaf0 = 1
 T676_onehot: t676_leaf0 + t676_leaf1 = 1
 T676_L_f3_k0: Overall_Qual_calc + 1e+06 t676_leaf0 <= 1.000002999999e+06
 T676_R_f3_k1: Overall_Qual_calc - 1e+06 t676_leaf1 >= -999997
 T677_onehot: t677_leaf0 = 1
 T678_onehot: t678_leaf0 + t678_leaf1 + t678_leaf2 + t678_leaf3
   + t678_leaf4 = 1
 T678_L_f207_k0: const_Bsmt_Exposure_Gd + 1e+06 t678_leaf0
   <= 1.000000999999e+06
 T678_L_f38_k0: x_Enclosed_Porch + 1e+06 t678_leaf0 <= 1.000264999999e+06
 T678_L_f1_k0: const_Lot_Area + 1e+06 t678_leaf0 <= 1.011791999999e+06
 T678_L_f207_k1: const_Bsmt_Exposure_Gd + 1e+06 t678_leaf1
   <= 1.000000999999e+06
 T678_L_f38_k1: x_Enclosed_Porch + 1e+06 t678_leaf1 <= 1.000264999999e+06
 T678_R_f1_k1: const_Lot_Area - 1e+06 t678_leaf1 >= -988208
 T678_L_f207_k2: const_Bsmt_Exposure_Gd + 1e+06 t678_leaf2
   <= 1.000000999999e+06
 T678_R_f38_k2: x_Enclosed_Porch - 1e+06 t678_leaf2 >= -999735
 T678_R_f207_k3: const_Bsmt_Exposure_Gd - 1e+06 t678_leaf3 >= -999999
 T678_L_f31_k3: const_Garage_Yr_Blt + 1e+06 t678_leaf3
   <= 1.001960999999e+06
 T678_R_f207_k4: const_Bsmt_Exposure_Gd - 1e+06 t678_leaf4 >= -999999
 T678_R_f31_k4: const_Garage_Yr_Blt - 1e+06 t678_leaf4 >= -998039
 T679_onehot: t679_leaf0 = 1
 T680_onehot: t680_leaf0 + t680_leaf1 + t680_leaf2 + t680_leaf3
   + t680_leaf4 = 1
 T680_L_f3_k0: Overall_Qual_calc + 1e+06 t680_leaf0 <= 1.000009999999e+06
 T680_L_f29_k0: const_Fireplaces + 1e+06 t680_leaf0 <= 1.000000999999e+06
 T680_L_f20_k0: x_Gr_Liv_Area + 1e+06 t680_leaf0 <= 1.001794999999e+06
 T680_L_f3_k1: Overall_Qual_calc + 1e+06 t680_leaf1 <= 1.000009999999e+06
 T680_L_f29_k1: const_Fireplaces + 1e+06 t680_leaf1 <= 1.000000999999e+06
 T680_R_f20_k1: x_Gr_Liv_Area - 1e+06 t680_leaf1 >= -998205
 T680_L_f3_k2: Overall_Qual_calc + 1e+06 t680_leaf2 <= 1.000009999999e+06
 T680_R_f29_k2: const_Fireplaces - 1e+06 t680_leaf2 >= -999999
 T680_L_f46_k2: const_MS_SubClass_30 + 1e+06 t680_leaf2
   <= 1.000000999999e+06
 T680_L_f3_k3: Overall_Qual_calc + 1e+06 t680_leaf3 <= 1.000009999999e+06
 T680_R_f29_k3: const_Fireplaces - 1e+06 t680_leaf3 >= -999999
 T680_R_f46_k3: const_MS_SubClass_30 - 1e+06 t680_leaf3 >= -999999
 T680_R_f3_k4: Overall_Qual_calc - 1e+06 t680_leaf4 >= -999990
 T681_onehot: t681_leaf0 + t681_leaf1 = 1
 T681_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t681_leaf0
   <= 1.002222999999e+06
 T681_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t681_leaf1 >= -997777
 T682_onehot: t682_leaf0 = 1
 T683_onehot: t683_leaf0 + t683_leaf1 + t683_leaf2 + t683_leaf3 = 1
 T683_L_f17_k0: x_1st_Flr_SF + 1e+06 t683_leaf0 <= 1.001433999999e+06
 T683_R_f17_k1: x_1st_Flr_SF - 1e+06 t683_leaf1 >= -998566
 T683_L_f33_k1: x_Garage_Area + 1e+06 t683_leaf1 <= 1.000287999999e+06
 T683_R_f17_k2: x_1st_Flr_SF - 1e+06 t683_leaf2 >= -998566
 T683_R_f33_k2: x_Garage_Area - 1e+06 t683_leaf2 >= -999712
 T683_L_f206_k2: const_Bsmt_Exposure_Av + 1e+06 t683_leaf2
   <= 1.000000999999e+06
 T683_R_f17_k3: x_1st_Flr_SF - 1e+06 t683_leaf3 >= -998566
 T683_R_f33_k3: x_Garage_Area - 1e+06 t683_leaf3 >= -999712
 T683_R_f206_k3: const_Bsmt_Exposure_Av - 1e+06 t683_leaf3 >= -999999
 T684_onehot: t684_leaf0 + t684_leaf1 + t684_leaf2 = 1
 T684_L_f20_k0: x_Gr_Liv_Area + 1e+06 t684_leaf0 <= 1.000746999999e+06
 T684_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t684_leaf0 <= 1.000272999999e+06
 T684_L_f20_k1: x_Gr_Liv_Area + 1e+06 t684_leaf1 <= 1.000746999999e+06
 T684_R_f14_k1: x_Bsmt_Unf_SF - 1e+06 t684_leaf1 >= -999727
 T684_R_f20_k2: x_Gr_Liv_Area - 1e+06 t684_leaf2 >= -999253
 T685_onehot: t685_leaf0 = 1
 T686_onehot: t686_leaf0 = 1
 T687_onehot: t687_leaf0 + t687_leaf1 + t687_leaf2 = 1
 T687_L_f3_k0: Overall_Qual_calc + 1e+06 t687_leaf0 <= 1.000002999999e+06
 T687_R_f3_k1: Overall_Qual_calc - 1e+06 t687_leaf1 >= -999997
 T687_L_f5_k1: const_Year_Built + 1e+06 t687_leaf1 <= 1.002004999999e+06
 T687_R_f3_k2: Overall_Qual_calc - 1e+06 t687_leaf2 >= -999997
 T687_R_f5_k2: const_Year_Built - 1e+06 t687_leaf2 >= -997995
 T688_onehot: t688_leaf0 = 1
 T689_onehot: t689_leaf0 = 1
 T690_onehot: t690_leaf0 = 1
 T691_onehot: t691_leaf0 = 1
 T692_onehot: t692_leaf0 + t692_leaf1 = 1
 T692_L_f0_k0: const_Lot_Frontage + 1e+06 t692_leaf0 <= 1.000151999999e+06
 T692_R_f0_k1: const_Lot_Frontage - 1e+06 t692_leaf1 >= -999848
 T693_onehot: t693_leaf0 + t693_leaf1 = 1
 T693_L_f20_k0: x_Gr_Liv_Area + 1e+06 t693_leaf0 <= 1.000746999999e+06
 T693_R_f20_k1: x_Gr_Liv_Area - 1e+06 t693_leaf1 >= -999253
 T694_onehot: t694_leaf0 = 1
 T695_onehot: t695_leaf0 = 1
 T696_onehot: t696_leaf0 = 1
 T697_onehot: t697_leaf0 + t697_leaf1 = 1
 T697_L_f25_k0: x_Bedroom_AbvGr + 1e+06 t697_leaf0 <= 1.000002999999e+06
 T697_R_f25_k1: x_Bedroom_AbvGr - 1e+06 t697_leaf1 >= -999997
 T698_onehot: t698_leaf0 + t698_leaf1 = 1
 T698_L_f33_k0: x_Garage_Area + 1e+06 t698_leaf0 <= 1.001153999999e+06
 T698_R_f33_k1: x_Garage_Area - 1e+06 t698_leaf1 >= -998846
 T699_onehot: t699_leaf0 = 1
 T700_onehot: t700_leaf0 = 1
 T701_onehot: t701_leaf0 = 1
 T702_onehot: t702_leaf0 + t702_leaf1 + t702_leaf2 + t702_leaf3 = 1
 T702_L_f5_k0: const_Year_Built + 1e+06 t702_leaf0 <= 1.001931999999e+06
 T702_L_f202_k0: const_Foundation_PConc + 1e+06 t702_leaf0
   <= 1.000000999999e+06
 T702_L_f5_k1: const_Year_Built + 1e+06 t702_leaf1 <= 1.001931999999e+06
 T702_R_f202_k1: const_Foundation_PConc - 1e+06 t702_leaf1 >= -999999
 T702_R_f5_k2: const_Year_Built - 1e+06 t702_leaf2 >= -998068
 T702_L_f21_k2: const_Bsmt_Full_Bath + 1e+06 t702_leaf2
   <= 1.000001999999e+06
 T702_R_f5_k3: const_Year_Built - 1e+06 t702_leaf3 >= -998068
 T702_R_f21_k3: const_Bsmt_Full_Bath - 1e+06 t702_leaf3 >= -999998
 T703_onehot: t703_leaf0 = 1
 T704_onehot: t704_leaf0 = 1
 T705_onehot: t705_leaf0 + t705_leaf1 + t705_leaf2 + t705_leaf3
   + t705_leaf4 = 1
 T705_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t705_leaf0
   <= 1.001418999999e+06
 T705_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t705_leaf0
   <= 1.001379999999e+06
 T705_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t705_leaf1
   <= 1.001418999999e+06
 T705_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t705_leaf1 >= -998620
 T705_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t705_leaf2 >= -998581
 T705_L_f17_k2: x_1st_Flr_SF + 1e+06 t705_leaf2 <= 1.001421999999e+06
 T705_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t705_leaf3 >= -998581
 T705_R_f17_k3: x_1st_Flr_SF - 1e+06 t705_leaf3 >= -998578
 T705_L_f33_k3: x_Garage_Area + 1e+06 t705_leaf3 <= 1.000287999999e+06
 T705_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t705_leaf4 >= -998581
 T705_R_f17_k4: x_1st_Flr_SF - 1e+06 t705_leaf4 >= -998578
 T705_R_f33_k4: x_Garage_Area - 1e+06 t705_leaf4 >= -999712
 T706_onehot: t706_leaf0 + t706_leaf1 + t706_leaf2 = 1
 T706_L_f8_k0: x_Exter_Qual + 1e+06 t706_leaf0 <= 1.000002999999e+06
 T706_R_f8_k1: x_Exter_Qual - 1e+06 t706_leaf1 >= -999997
 T706_L_f20_k1: x_Gr_Liv_Area + 1e+06 t706_leaf1 <= 1.002793999999e+06
 T706_R_f8_k2: x_Exter_Qual - 1e+06 t706_leaf2 >= -999997
 T706_R_f20_k2: x_Gr_Liv_Area - 1e+06 t706_leaf2 >= -997206
 T707_onehot: t707_leaf0 = 1
 T708_onehot: t708_leaf0 = 1
 T709_onehot: t709_leaf0 + t709_leaf1 + t709_leaf2 = 1
 T709_L_f274_k0: const_Mo_Sold_4 + 1e+06 t709_leaf0 <= 1.000000999999e+06
 T709_R_f274_k1: const_Mo_Sold_4 - 1e+06 t709_leaf1 >= -999999
 T709_L_f29_k1: const_Fireplaces + 1e+06 t709_leaf1 <= 1.000000999999e+06
 T709_R_f274_k2: const_Mo_Sold_4 - 1e+06 t709_leaf2 >= -999999
 T709_R_f29_k2: const_Fireplaces - 1e+06 t709_leaf2 >= -999999
 T710_onehot: t710_leaf0 + t710_leaf1 + t710_leaf2 + t710_leaf3 = 1
 T710_L_f38_k0: x_Enclosed_Porch + 1e+06 t710_leaf0 <= 1.000017999999e+06
 T710_R_f38_k1: x_Enclosed_Porch - 1e+06 t710_leaf1 >= -999982
 T710_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t710_leaf1
   <= 1.000000999999e+06
 T710_L_f283_k1: const_Sale_Type_COD + 1e+06 t710_leaf1
   <= 1.000000999999e+06
 T710_R_f38_k2: x_Enclosed_Porch - 1e+06 t710_leaf2 >= -999982
 T710_L_f297_k2: const_Sale_Condition_Normal + 1e+06 t710_leaf2
   <= 1.000000999999e+06
 T710_R_f283_k2: const_Sale_Type_COD - 1e+06 t710_leaf2 >= -999999
 T710_R_f38_k3: x_Enclosed_Porch - 1e+06 t710_leaf3 >= -999982
 T710_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t710_leaf3 >= -999999
 T711_onehot: t711_leaf0 = 1
 T712_onehot: t712_leaf0 + t712_leaf1 + t712_leaf2 + t712_leaf3 = 1
 T712_L_f76_k0: const_Lot_Shape_Reg + 1e+06 t712_leaf0
   <= 1.000000999999e+06
 T712_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t712_leaf0 <= 1.000011999999e+06
 T712_L_f86_k0: const_Land_Slope_Gtl + 1e+06 t712_leaf0
   <= 1.000000999999e+06
 T712_L_f76_k1: const_Lot_Shape_Reg + 1e+06 t712_leaf1
   <= 1.000000999999e+06
 T712_L_f28_k1: x_TotRms_AbvGrd + 1e+06 t712_leaf1 <= 1.000011999999e+06
 T712_R_f86_k1: const_Land_Slope_Gtl - 1e+06 t712_leaf1 >= -999999
 T712_L_f76_k2: const_Lot_Shape_Reg + 1e+06 t712_leaf2
   <= 1.000000999999e+06
 T712_R_f28_k2: x_TotRms_AbvGrd - 1e+06 t712_leaf2 >= -999988
 T712_R_f76_k3: const_Lot_Shape_Reg - 1e+06 t712_leaf3 >= -999999
 T713_onehot: t713_leaf0 + t713_leaf1 = 1
 T713_L_f258_k0: x_paved_drive_is_P + 1e+06 t713_leaf0
   <= 1.000000999999e+06
 T713_R_f258_k1: x_paved_drive_is_P - 1e+06 t713_leaf1 >= -999999
 T714_onehot: t714_leaf0 = 1
 T715_onehot: t715_leaf0 + t715_leaf1 + t715_leaf2 + t715_leaf3 = 1
 T715_L_f95_k0: const_Neighborhood_Crawfor + 1e+06 t715_leaf0
   <= 1.000000999999e+06
 T715_L_f1_k0: const_Lot_Area + 1e+06 t715_leaf0 <= 1.011791999999e+06
 T715_L_f95_k1: const_Neighborhood_Crawfor + 1e+06 t715_leaf1
   <= 1.000000999999e+06
 T715_R_f1_k1: const_Lot_Area - 1e+06 t715_leaf1 >= -988208
 T715_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t715_leaf1
   <= 1.000000999999e+06
 T715_L_f95_k2: const_Neighborhood_Crawfor + 1e+06 t715_leaf2
   <= 1.000000999999e+06
 T715_R_f1_k2: const_Lot_Area - 1e+06 t715_leaf2 >= -988208
 T715_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t715_leaf2 >= -999999
 T715_R_f95_k3: const_Neighborhood_Crawfor - 1e+06 t715_leaf3 >= -999999
 T716_onehot: t716_leaf0 = 1
 T717_onehot: t717_leaf0 = 1
 T718_onehot: t718_leaf0 = 1
 T719_onehot: t719_leaf0 = 1
 T720_onehot: t720_leaf0 + t720_leaf1 = 1
 T720_L_f3_k0: Overall_Qual_calc + 1e+06 t720_leaf0 <= 1.000002999999e+06
 T720_R_f3_k1: Overall_Qual_calc - 1e+06 t720_leaf1 >= -999997
 T721_onehot: t721_leaf0 = 1
 T722_onehot: t722_leaf0 = 1
 T723_onehot: t723_leaf0 = 1
 T724_onehot: t724_leaf0 = 1
 T725_onehot: t725_leaf0 = 1
 T726_onehot: t726_leaf0 + t726_leaf1 + t726_leaf2 = 1
 T726_L_f154_k0: x_roof_matl_is_CompShg + 1e+06 t726_leaf0
   <= 1.000000999999e+06
 T726_R_f154_k1: x_roof_matl_is_CompShg - 1e+06 t726_leaf1 >= -999999
 T726_L_f31_k1: const_Garage_Yr_Blt + 1e+06 t726_leaf1
   <= 1.002007999999e+06
 T726_R_f154_k2: x_roof_matl_is_CompShg - 1e+06 t726_leaf2 >= -999999
 T726_R_f31_k2: const_Garage_Yr_Blt - 1e+06 t726_leaf2 >= -997992
 T727_onehot: t727_leaf0 + t727_leaf1 + t727_leaf2 = 1
 T727_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t727_leaf0 <= 1.000000999999e+06
 T727_R_f177_k1: x_ex2_is_AsbShng - 1e+06 t727_leaf1 >= -999999
 T727_L_f4_k1: const_Overall_Cond + 1e+06 t727_leaf1 <= 1.000004999999e+06
 T727_R_f177_k2: x_ex2_is_AsbShng - 1e+06 t727_leaf2 >= -999999
 T727_R_f4_k2: const_Overall_Cond - 1e+06 t727_leaf2 >= -999995
 T728_onehot: t728_leaf0 + t728_leaf1 = 1
 T728_L_f23_k0: x_Full_Bath + 1e+06 t728_leaf0 <= 1.000002999999e+06
 T728_R_f23_k1: x_Full_Bath - 1e+06 t728_leaf1 >= -999997
 T729_onehot: t729_leaf0 = 1
 T730_onehot: t730_leaf0 + t730_leaf1 + t730_leaf2 + t730_leaf3
   + t730_leaf4 = 1
 T730_L_f20_k0: x_Gr_Liv_Area + 1e+06 t730_leaf0 <= 1.000987999999e+06
 T730_R_f20_k1: x_Gr_Liv_Area - 1e+06 t730_leaf1 >= -999012
 T730_L_f6_k1: const_Year_Remod_Add + 1e+06 t730_leaf1
   <= 1.001971999999e+06
 T730_L_f26_k1: x_Kitchen_AbvGr + 1e+06 t730_leaf1 <= 1.000001999999e+06
 T730_R_f20_k2: x_Gr_Liv_Area - 1e+06 t730_leaf2 >= -999012
 T730_L_f6_k2: const_Year_Remod_Add + 1e+06 t730_leaf2
   <= 1.001971999999e+06
 T730_R_f26_k2: x_Kitchen_AbvGr - 1e+06 t730_leaf2 >= -999998
 T730_R_f20_k3: x_Gr_Liv_Area - 1e+06 t730_leaf3 >= -999012
 T730_R_f6_k3: const_Year_Remod_Add - 1e+06 t730_leaf3 >= -998028
 T730_L_f7_k3: x_Mas_Vnr_Area + 1e+06 t730_leaf3 <= 1.000787999999e+06
 T730_R_f20_k4: x_Gr_Liv_Area - 1e+06 t730_leaf4 >= -999012
 T730_R_f6_k4: const_Year_Remod_Add - 1e+06 t730_leaf4 >= -998028
 T730_R_f7_k4: x_Mas_Vnr_Area - 1e+06 t730_leaf4 >= -999212
 T731_onehot: t731_leaf0 = 1
 T732_onehot: t732_leaf0 = 1
 T733_onehot: t733_leaf0 = 1
 T734_onehot: t734_leaf0 = 1
 T735_onehot: t735_leaf0 = 1
 T736_onehot: t736_leaf0 = 1
 T737_onehot: t737_leaf0 = 1
 T738_onehot: t738_leaf0 = 1
 T739_onehot: t739_leaf0 + t739_leaf1 = 1
 T739_L_f37_k0: x_Open_Porch_SF + 1e+06 t739_leaf0 <= 1.000168999999e+06
 T739_R_f37_k1: x_Open_Porch_SF - 1e+06 t739_leaf1 >= -999831
 T740_onehot: t740_leaf0 = 1
 T741_onehot: t741_leaf0 + t741_leaf1 + t741_leaf2 = 1
 T741_L_f1_k0: const_Lot_Area + 1e+06 t741_leaf0 <= 1.023256999999e+06
 T741_R_f1_k1: const_Lot_Area - 1e+06 t741_leaf1 >= -976743
 T741_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t741_leaf1
   <= 1.000000999999e+06
 T741_R_f1_k2: const_Lot_Area - 1e+06 t741_leaf2 >= -976743
 T741_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t741_leaf2 >= -999999
 T742_onehot: t742_leaf0 + t742_leaf1 + t742_leaf2 + t742_leaf3
   + t742_leaf4 + t742_leaf5 + t742_leaf6 = 1
 T742_L_f272_k0: const_Mo_Sold_2 + 1e+06 t742_leaf0 <= 1.000000999999e+06
 T742_L_f232_k0: central_air_yes + 1e+06 t742_leaf0 <= 1.000000999999e+06
 T742_L_f40_k0: x_Screen_Porch + 1e+06 t742_leaf0 <= 1.000052999999e+06
 T742_L_f272_k1: const_Mo_Sold_2 + 1e+06 t742_leaf1 <= 1.000000999999e+06
 T742_L_f232_k1: central_air_yes + 1e+06 t742_leaf1 <= 1.000000999999e+06
 T742_R_f40_k1: x_Screen_Porch - 1e+06 t742_leaf1 >= -999947
 T742_L_f272_k2: const_Mo_Sold_2 + 1e+06 t742_leaf2 <= 1.000000999999e+06
 T742_R_f232_k2: central_air_yes - 1e+06 t742_leaf2 >= -999999
 T742_L_f33_k2: x_Garage_Area + 1e+06 t742_leaf2 <= 1.000269999999e+06
 T742_L_f272_k3: const_Mo_Sold_2 + 1e+06 t742_leaf3 <= 1.000000999999e+06
 T742_R_f232_k3: central_air_yes - 1e+06 t742_leaf3 >= -999999
 T742_R_f33_k3: x_Garage_Area - 1e+06 t742_leaf3 >= -999730
 T742_R_f272_k4: const_Mo_Sold_2 - 1e+06 t742_leaf4 >= -999999
 T742_L_f297_k4: const_Sale_Condition_Normal + 1e+06 t742_leaf4
   <= 1.000000999999e+06
 T742_L_f37_k4: x_Open_Porch_SF + 1e+06 t742_leaf4 <= 1.000049999999e+06
 T742_R_f272_k5: const_Mo_Sold_2 - 1e+06 t742_leaf5 >= -999999
 T742_L_f297_k5: const_Sale_Condition_Normal + 1e+06 t742_leaf5
   <= 1.000000999999e+06
 T742_R_f37_k5: x_Open_Porch_SF - 1e+06 t742_leaf5 >= -999950
 T742_R_f272_k6: const_Mo_Sold_2 - 1e+06 t742_leaf6 >= -999999
 T742_R_f297_k6: const_Sale_Condition_Normal - 1e+06 t742_leaf6 >= -999999
 T743_onehot: t743_leaf0 = 1
 T744_onehot: t744_leaf0 = 1
 T745_onehot: t745_leaf0 = 1
 T746_onehot: t746_leaf0 = 1
 T747_onehot: t747_leaf0 = 1
 T748_onehot: t748_leaf0 = 1
 T749_onehot: t749_leaf0 + t749_leaf1 + t749_leaf2 + t749_leaf3 = 1
 T749_L_f259_k0: x_paved_drive_is_Y + 1e+06 t749_leaf0
   <= 1.000000999999e+06
 T749_L_f33_k0: x_Garage_Area + 1e+06 t749_leaf0 <= 1.000402999999e+06
 T749_L_f1_k0: const_Lot_Area + 1e+06 t749_leaf0 <= 1.010336999999e+06
 T749_L_f259_k1: x_paved_drive_is_Y + 1e+06 t749_leaf1
   <= 1.000000999999e+06
 T749_L_f33_k1: x_Garage_Area + 1e+06 t749_leaf1 <= 1.000402999999e+06
 T749_R_f1_k1: const_Lot_Area - 1e+06 t749_leaf1 >= -989663
 T749_L_f259_k2: x_paved_drive_is_Y + 1e+06 t749_leaf2
   <= 1.000000999999e+06
 T749_R_f33_k2: x_Garage_Area - 1e+06 t749_leaf2 >= -999597
 T749_R_f259_k3: x_paved_drive_is_Y - 1e+06 t749_leaf3 >= -999999
 T750_onehot: t750_leaf0 = 1
 T751_onehot: t751_leaf0 = 1
 T752_onehot: t752_leaf0 = 1
 T753_onehot: t753_leaf0 = 1
 T754_onehot: t754_leaf0 = 1
 T755_onehot: t755_leaf0 = 1
 T756_onehot: t756_leaf0 = 1
 T757_onehot: t757_leaf0 + t757_leaf1 + t757_leaf2 + t757_leaf3 = 1
 T757_L_f44_k0: const_Yr_Sold + 1e+06 t757_leaf0 <= 1.002009999999e+06
 T757_R_f44_k1: const_Yr_Sold - 1e+06 t757_leaf1 >= -997990
 T757_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t757_leaf1
   <= 1.000760999999e+06
 T757_L_f297_k1: const_Sale_Condition_Normal + 1e+06 t757_leaf1
   <= 1.000000999999e+06
 T757_R_f44_k2: const_Yr_Sold - 1e+06 t757_leaf2 >= -997990
 T757_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t757_leaf2
   <= 1.000760999999e+06
 T757_R_f297_k2: const_Sale_Condition_Normal - 1e+06 t757_leaf2 >= -999999
 T757_R_f44_k3: const_Yr_Sold - 1e+06 t757_leaf3 >= -997990
 T757_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t757_leaf3 >= -999239
 T758_onehot: t758_leaf0 = 1
 T759_onehot: t759_leaf0 + t759_leaf1 = 1
 T759_L_f240_k0: const_FunctioNo_aplical_Min1 + 1e+06 t759_leaf0
   <= 1.000000999999e+06
 T759_R_f240_k1: const_FunctioNo_aplical_Min1 - 1e+06 t759_leaf1
   >= -999999
 T760_onehot: t760_leaf0 = 1
 T761_onehot: t761_leaf0 + t761_leaf1 + t761_leaf2 + t761_leaf3 = 1
 T761_L_f10_k0: const_Bsmt_Qual + 1e+06 t761_leaf0 <= 1.000003999999e+06
 T761_L_f33_k0: x_Garage_Area + 1e+06 t761_leaf0 <= 1.000865999999e+06
 T761_L_f10_k1: const_Bsmt_Qual + 1e+06 t761_leaf1 <= 1.000003999999e+06
 T761_R_f33_k1: x_Garage_Area - 1e+06 t761_leaf1 >= -999134
 T761_R_f10_k2: const_Bsmt_Qual - 1e+06 t761_leaf2 >= -999996
 T761_L_f33_k2: x_Garage_Area + 1e+06 t761_leaf2 <= 1.000643999999e+06
 T761_R_f10_k3: const_Bsmt_Qual - 1e+06 t761_leaf3 >= -999996
 T761_R_f33_k3: x_Garage_Area - 1e+06 t761_leaf3 >= -999356
 T762_onehot: t762_leaf0 = 1
 T763_onehot: t763_leaf0 + t763_leaf1 + t763_leaf2 + t763_leaf3
   + t763_leaf4 = 1
 T763_L_f150_k0: const_Roof_Style_Hip + 1e+06 t763_leaf0
   <= 1.000000999999e+06
 T763_L_f1_k0: const_Lot_Area + 1e+06 t763_leaf0 <= 1.019521999999e+06
 T763_L_f150_k1: const_Roof_Style_Hip + 1e+06 t763_leaf1
   <= 1.000000999999e+06
 T763_R_f1_k1: const_Lot_Area - 1e+06 t763_leaf1 >= -980478
 T763_R_f150_k2: const_Roof_Style_Hip - 1e+06 t763_leaf2 >= -999999
 T763_L_f44_k2: const_Yr_Sold + 1e+06 t763_leaf2 <= 1.002007999999e+06
 T763_R_f150_k3: const_Roof_Style_Hip - 1e+06 t763_leaf3 >= -999999
 T763_R_f44_k3: const_Yr_Sold - 1e+06 t763_leaf3 >= -997992
 T763_L_f14_k3: x_Bsmt_Unf_SF + 1e+06 t763_leaf3 <= 1.000465999999e+06
 T763_R_f150_k4: const_Roof_Style_Hip - 1e+06 t763_leaf4 >= -999999
 T763_R_f44_k4: const_Yr_Sold - 1e+06 t763_leaf4 >= -997992
 T763_R_f14_k4: x_Bsmt_Unf_SF - 1e+06 t763_leaf4 >= -999534
 T764_onehot: t764_leaf0 + t764_leaf1 + t764_leaf2 + t764_leaf3 = 1
 T764_L_f14_k0: x_Bsmt_Unf_SF + 1e+06 t764_leaf0 <= 1.000440999999e+06
 T764_L_f4_k0: const_Overall_Cond + 1e+06 t764_leaf0 <= 1.000003999999e+06
 T764_L_f14_k1: x_Bsmt_Unf_SF + 1e+06 t764_leaf1 <= 1.000440999999e+06
 T764_R_f4_k1: const_Overall_Cond - 1e+06 t764_leaf1 >= -999996
 T764_L_f79_k1: const_Land_Contour_Low + 1e+06 t764_leaf1
   <= 1.000000999999e+06
 T764_L_f14_k2: x_Bsmt_Unf_SF + 1e+06 t764_leaf2 <= 1.000440999999e+06
 T764_R_f4_k2: const_Overall_Cond - 1e+06 t764_leaf2 >= -999996
 T764_R_f79_k2: const_Land_Contour_Low - 1e+06 t764_leaf2 >= -999999
 T764_R_f14_k3: x_Bsmt_Unf_SF - 1e+06 t764_leaf3 >= -999559
 T765_onehot: t765_leaf0 = 1
 T766_onehot: t766_leaf0 + t766_leaf1 = 1
 T766_L_f72_k0: const_Alley_Pave + 1e+06 t766_leaf0 <= 1.000000999999e+06
 T766_R_f72_k1: const_Alley_Pave - 1e+06 t766_leaf1 >= -999999
 T767_onehot: t767_leaf0 = 1
 T768_onehot: t768_leaf0 + t768_leaf1 + t768_leaf2 = 1
 T768_L_f37_k0: x_Open_Porch_SF + 1e+06 t768_leaf0 <= 1.000259999999e+06
 T768_L_f17_k0: x_1st_Flr_SF + 1e+06 t768_leaf0 <= 1.001972999999e+06
 T768_L_f37_k1: x_Open_Porch_SF + 1e+06 t768_leaf1 <= 1.000259999999e+06
 T768_R_f17_k1: x_1st_Flr_SF - 1e+06 t768_leaf1 >= -998027
 T768_R_f37_k2: x_Open_Porch_SF - 1e+06 t768_leaf2 >= -999740
 T769_onehot: t769_leaf0 = 1
 T770_onehot: t770_leaf0 = 1
 T771_onehot: t771_leaf0 = 1
 T772_onehot: t772_leaf0 + t772_leaf1 + t772_leaf2 = 1
 T772_L_f1_k0: const_Lot_Area + 1e+06 t772_leaf0 <= 1.010199999999e+06
 T772_R_f1_k1: const_Lot_Area - 1e+06 t772_leaf1 >= -989800
 T772_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t772_leaf1
   <= 1.000480999999e+06
 T772_R_f1_k2: const_Lot_Area - 1e+06 t772_leaf2 >= -989800
 T772_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t772_leaf2 >= -999519
 T773_onehot: t773_leaf0 = 1
 T774_onehot: t774_leaf0 = 1
 T775_onehot: t775_leaf0 + t775_leaf1 = 1
 T775_L_f100_k0: const_Neighborhood_IDOTRR + 1e+06 t775_leaf0
   <= 1.000000999999e+06
 T775_R_f100_k1: const_Neighborhood_IDOTRR - 1e+06 t775_leaf1 >= -999999
 T776_onehot: t776_leaf0 = 1
 T777_onehot: t777_leaf0 + t777_leaf1 = 1
 T777_L_f33_k0: x_Garage_Area + 1e+06 t777_leaf0 <= 1.000971999999e+06
 T777_R_f33_k1: x_Garage_Area - 1e+06 t777_leaf1 >= -999028
 T778_onehot: t778_leaf0 + t778_leaf1 + t778_leaf2 + t778_leaf3 = 1
 T778_L_f62_k0: const_MS_Zoning_C_(all) + 1e+06 t778_leaf0
   <= 1.000000999999e+06
 T778_L_f177_k0: x_ex2_is_AsbShng + 1e+06 t778_leaf0 <= 1.000000999999e+06
 T778_L_f62_k1: const_MS_Zoning_C_(all) + 1e+06 t778_leaf1
   <= 1.000000999999e+06
 T778_R_f177_k1: x_ex2_is_AsbShng - 1e+06 t778_leaf1 >= -999999
 T778_L_f4_k1: const_Overall_Cond + 1e+06 t778_leaf1 <= 1.000004999999e+06
 T778_L_f62_k2: const_MS_Zoning_C_(all) + 1e+06 t778_leaf2
   <= 1.000000999999e+06
 T778_R_f177_k2: x_ex2_is_AsbShng - 1e+06 t778_leaf2 >= -999999
 T778_R_f4_k2: const_Overall_Cond - 1e+06 t778_leaf2 >= -999995
 T778_R_f62_k3: const_MS_Zoning_C_(all) - 1e+06 t778_leaf3 >= -999999
 T779_onehot: t779_leaf0 = 1
 T780_onehot: t780_leaf0 = 1
 T781_onehot: t781_leaf0 + t781_leaf1 + t781_leaf2 = 1
 T781_L_f3_k0: Overall_Qual_calc + 1e+06 t781_leaf0 <= 1.000002999999e+06
 T781_R_f3_k1: Overall_Qual_calc - 1e+06 t781_leaf1 >= -999997
 T781_L_f33_k1: x_Garage_Area + 1e+06 t781_leaf1 <= 1.000803999999e+06
 T781_R_f3_k2: Overall_Qual_calc - 1e+06 t781_leaf2 >= -999997
 T781_R_f33_k2: x_Garage_Area - 1e+06 t781_leaf2 >= -999196
 T782_onehot: t782_leaf0 = 1
 T783_onehot: t783_leaf0 = 1
 T784_onehot: t784_leaf0 + t784_leaf1 + t784_leaf2 + t784_leaf3
   + t784_leaf4 = 1
 T784_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t784_leaf0
   <= 1.002135999999e+06
 T784_L_f33_k0: x_Garage_Area + 1e+06 t784_leaf0 <= 1.000251999999e+06
 T784_L_f3_k0: Overall_Qual_calc + 1e+06 t784_leaf0 <= 1.000005999999e+06
 T784_L_f15_k1: const_Total_Bsmt_SF + 1e+06 t784_leaf1
   <= 1.002135999999e+06
 T784_L_f33_k1: x_Garage_Area + 1e+06 t784_leaf1 <= 1.000251999999e+06
 T784_R_f3_k1: Overall_Qual_calc - 1e+06 t784_leaf1 >= -999994
 T784_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t784_leaf2
   <= 1.002135999999e+06
 T784_R_f33_k2: x_Garage_Area - 1e+06 t784_leaf2 >= -999748
 T784_L_f62_k2: const_MS_Zoning_C_(all) + 1e+06 t784_leaf2
   <= 1.000000999999e+06
 T784_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t784_leaf3
   <= 1.002135999999e+06
 T784_R_f33_k3: x_Garage_Area - 1e+06 t784_leaf3 >= -999748
 T784_R_f62_k3: const_MS_Zoning_C_(all) - 1e+06 t784_leaf3 >= -999999
 T784_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t784_leaf4 >= -997864
 T785_onehot: t785_leaf0 + t785_leaf1 = 1
 T785_L_f17_k0: x_1st_Flr_SF + 1e+06 t785_leaf0 <= 1.000575999999e+06
 T785_R_f17_k1: x_1st_Flr_SF - 1e+06 t785_leaf1 >= -999424
 T786_onehot: t786_leaf0 = 1
 T787_onehot: t787_leaf0 + t787_leaf1 = 1
 T787_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t787_leaf0 <= 1.001964999999e+06
 T787_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t787_leaf1 >= -998035
 T788_onehot: t788_leaf0 = 1
 T789_onehot: t789_leaf0 = 1
 T790_onehot: t790_leaf0 = 1
 T791_onehot: t791_leaf0 = 1
 T792_onehot: t792_leaf0 = 1
 T793_onehot: t793_leaf0 + t793_leaf1 + t793_leaf2 + t793_leaf3 = 1
 T793_L_f37_k0: x_Open_Porch_SF + 1e+06 t793_leaf0 <= 1.000259999999e+06
 T793_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t793_leaf0 <= 1.001964999999e+06
 T793_L_f37_k1: x_Open_Porch_SF + 1e+06 t793_leaf1 <= 1.000259999999e+06
 T793_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t793_leaf1 >= -998035
 T793_R_f37_k2: x_Open_Porch_SF - 1e+06 t793_leaf2 >= -999740
 T793_L_f297_k2: const_Sale_Condition_Normal + 1e+06 t793_leaf2
   <= 1.000000999999e+06
 T793_R_f37_k3: x_Open_Porch_SF - 1e+06 t793_leaf3 >= -999740
 T793_R_f297_k3: const_Sale_Condition_Normal - 1e+06 t793_leaf3 >= -999999
 T794_onehot: t794_leaf0 = 1
 T795_onehot: t795_leaf0 + t795_leaf1 + t795_leaf2 + t795_leaf3
   + t795_leaf4 = 1
 T795_L_f5_k0: const_Year_Built + 1e+06 t795_leaf0 <= 1.001944999999e+06
 T795_L_f4_k0: const_Overall_Cond + 1e+06 t795_leaf0 <= 1.000006999999e+06
 T795_L_f4_k0: const_Overall_Cond + 1e+06 t795_leaf0 <= 1.000004999999e+06
 T795_L_f5_k1: const_Year_Built + 1e+06 t795_leaf1 <= 1.001944999999e+06
 T795_L_f4_k1: const_Overall_Cond + 1e+06 t795_leaf1 <= 1.000006999999e+06
 T795_R_f4_k1: const_Overall_Cond - 1e+06 t795_leaf1 >= -999995
 T795_L_f5_k2: const_Year_Built + 1e+06 t795_leaf2 <= 1.001944999999e+06
 T795_R_f4_k2: const_Overall_Cond - 1e+06 t795_leaf2 >= -999993
 T795_R_f5_k3: const_Year_Built - 1e+06 t795_leaf3 >= -998055
 T795_L_f231_k3: central_air_no + 1e+06 t795_leaf3 <= 1.000000999999e+06
 T795_R_f5_k4: const_Year_Built - 1e+06 t795_leaf4 >= -998055
 T795_R_f231_k4: central_air_no - 1e+06 t795_leaf4 >= -999999
 T796_onehot: t796_leaf0 = 1
 T797_onehot: t797_leaf0 = 1
 T798_onehot: t798_leaf0 = 1
 T799_onehot: t799_leaf0 = 1
 T800_onehot: t800_leaf0 = 1
 T801_onehot: t801_leaf0 = 1
 T802_onehot: t802_leaf0 = 1
 T803_onehot: t803_leaf0 = 1
 T804_onehot: t804_leaf0 = 1
 T805_onehot: t805_leaf0 + t805_leaf1 = 1
 T805_L_f9_k0: x_Exter_Cond + 1e+06 t805_leaf0 <= 1.000001999999e+06
 T805_R_f9_k1: x_Exter_Cond - 1e+06 t805_leaf1 >= -999998
 T806_onehot: t806_leaf0 + t806_leaf1 = 1
 T806_L_f20_k0: x_Gr_Liv_Area + 1e+06 t806_leaf0 <= 1.000935999999e+06
 T806_R_f20_k1: x_Gr_Liv_Area - 1e+06 t806_leaf1 >= -999064
 T807_onehot: t807_leaf0 = 1
 T808_onehot: t808_leaf0 = 1
 T809_onehot: t809_leaf0 = 1
 T810_onehot: t810_leaf0 = 1
 T811_onehot: t811_leaf0 = 1
 T812_onehot: t812_leaf0 = 1
 T813_onehot: t813_leaf0 = 1
 T814_onehot: t814_leaf0 + t814_leaf1 + t814_leaf2 + t814_leaf3
   + t814_leaf4 = 1
 T814_L_f38_k0: x_Enclosed_Porch + 1e+06 t814_leaf0 <= 1.000235999999e+06
 T814_L_f258_k0: x_paved_drive_is_P + 1e+06 t814_leaf0
   <= 1.000000999999e+06
 T814_L_f106_k0: const_Neighborhood_No_aplicames + 1e+06 t814_leaf0
   <= 1.000000999999e+06
 T814_L_f38_k1: x_Enclosed_Porch + 1e+06 t814_leaf1 <= 1.000235999999e+06
 T814_L_f258_k1: x_paved_drive_is_P + 1e+06 t814_leaf1
   <= 1.000000999999e+06
 T814_R_f106_k1: const_Neighborhood_No_aplicames - 1e+06 t814_leaf1
   >= -999999
 T814_L_f38_k2: x_Enclosed_Porch + 1e+06 t814_leaf2 <= 1.000235999999e+06
 T814_R_f258_k2: x_paved_drive_is_P - 1e+06 t814_leaf2 >= -999999
 T814_R_f38_k3: x_Enclosed_Porch - 1e+06 t814_leaf3 >= -999764
 T814_L_f1_k3: const_Lot_Area + 1e+06 t814_leaf3 <= 1.009749999999e+06
 T814_R_f38_k4: x_Enclosed_Porch - 1e+06 t814_leaf4 >= -999764
 T814_R_f1_k4: const_Lot_Area - 1e+06 t814_leaf4 >= -990250
 T815_onehot: t815_leaf0 = 1
 T816_onehot: t816_leaf0 = 1
 T817_onehot: t817_leaf0 + t817_leaf1 + t817_leaf2 = 1
 T817_L_f161_k0: x_ex1_is_AsbShng + 1e+06 t817_leaf0 <= 1.000000999999e+06
 T817_R_f161_k1: x_ex1_is_AsbShng - 1e+06 t817_leaf1 >= -999999
 T817_L_f20_k1: x_Gr_Liv_Area + 1e+06 t817_leaf1 <= 1.001225999999e+06
 T817_R_f161_k2: x_ex1_is_AsbShng - 1e+06 t817_leaf2 >= -999999
 T817_R_f20_k2: x_Gr_Liv_Area - 1e+06 t817_leaf2 >= -998774
 T818_onehot: t818_leaf0 = 1
 T819_onehot: t819_leaf0 = 1
 T820_onehot: t820_leaf0 = 1
 T821_onehot: t821_leaf0 = 1
 T822_onehot: t822_leaf0 = 1
 T823_onehot: t823_leaf0 = 1
 T824_onehot: t824_leaf0 = 1
 T825_onehot: t825_leaf0 + t825_leaf1 = 1
 T825_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t825_leaf0 <= 1.001386999999e+06
 T825_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t825_leaf1 >= -998613
 T826_onehot: t826_leaf0 = 1
 T827_onehot: t827_leaf0 + t827_leaf1 + t827_leaf2 + t827_leaf3 = 1
 T827_L_f5_k0: const_Year_Built + 1e+06 t827_leaf0 <= 1.001951999999e+06
 T827_L_f5_k0: const_Year_Built + 1e+06 t827_leaf0 <= 1.001946999999e+06
 T827_L_f44_k0: const_Yr_Sold + 1e+06 t827_leaf0 <= 1.002009999999e+06
 T827_L_f5_k1: const_Year_Built + 1e+06 t827_leaf1 <= 1.001951999999e+06
 T827_L_f5_k1: const_Year_Built + 1e+06 t827_leaf1 <= 1.001946999999e+06
 T827_R_f44_k1: const_Yr_Sold - 1e+06 t827_leaf1 >= -997990
 T827_L_f5_k2: const_Year_Built + 1e+06 t827_leaf2 <= 1.001951999999e+06
 T827_R_f5_k2: const_Year_Built - 1e+06 t827_leaf2 >= -998053
 T827_R_f5_k3: const_Year_Built - 1e+06 t827_leaf3 >= -998048
 T828_onehot: t828_leaf0 = 1
 T829_onehot: t829_leaf0 + t829_leaf1 + t829_leaf2 + t829_leaf3
   + t829_leaf4 = 1
 T829_L_f17_k0: x_1st_Flr_SF + 1e+06 t829_leaf0 <= 1.001781999999e+06
 T829_L_f17_k0: x_1st_Flr_SF + 1e+06 t829_leaf0 <= 1.000623999999e+06
 T829_L_f17_k1: x_1st_Flr_SF + 1e+06 t829_leaf1 <= 1.001781999999e+06
 T829_R_f17_k1: x_1st_Flr_SF - 1e+06 t829_leaf1 >= -999376
 T829_R_f17_k2: x_1st_Flr_SF - 1e+06 t829_leaf2 >= -998218
 T829_L_f18_k2: const_2nd_Flr_SF + 1e+06 t829_leaf2 <= 1.000955999999e+06
 T829_L_f37_k2: x_Open_Porch_SF + 1e+06 t829_leaf2 <= 1.000032999999e+06
 T829_R_f17_k3: x_1st_Flr_SF - 1e+06 t829_leaf3 >= -998218
 T829_L_f18_k3: const_2nd_Flr_SF + 1e+06 t829_leaf3 <= 1.000955999999e+06
 T829_R_f37_k3: x_Open_Porch_SF - 1e+06 t829_leaf3 >= -999967
 T829_R_f17_k4: x_1st_Flr_SF - 1e+06 t829_leaf4 >= -998218
 T829_R_f18_k4: const_2nd_Flr_SF - 1e+06 t829_leaf4 >= -999044
 T830_onehot: t830_leaf0 = 1
 T831_onehot: t831_leaf0 + t831_leaf1 = 1
 T831_L_f38_k0: x_Enclosed_Porch + 1e+06 t831_leaf0 <= 1.000264999999e+06
 T831_R_f38_k1: x_Enclosed_Porch - 1e+06 t831_leaf1 >= -999735
 T832_onehot: t832_leaf0 + t832_leaf1 = 1
 T832_L_f3_k0: Overall_Qual_calc + 1e+06 t832_leaf0 <= 1.000002999999e+06
 T832_R_f3_k1: Overall_Qual_calc - 1e+06 t832_leaf1 >= -999997
 T833_onehot: t833_leaf0 = 1
 T834_onehot: t834_leaf0 = 1
 T835_onehot: t835_leaf0 + t835_leaf1 + t835_leaf2 = 1
 T835_L_f231_k0: central_air_no + 1e+06 t835_leaf0 <= 1.000000999999e+06
 T835_R_f231_k1: central_air_no - 1e+06 t835_leaf1 >= -999999
 T835_L_f18_k1: const_2nd_Flr_SF + 1e+06 t835_leaf1 <= 1.000811999999e+06
 T835_R_f231_k2: central_air_no - 1e+06 t835_leaf2 >= -999999
 T835_R_f18_k2: const_2nd_Flr_SF - 1e+06 t835_leaf2 >= -999188
 T836_onehot: t836_leaf0 = 1
 T837_onehot: t837_leaf0 = 1
 T838_onehot: t838_leaf0 = 1
 T839_onehot: t839_leaf0 + t839_leaf1 = 1
 T839_L_f147_k0: const_Roof_Style_Flat + 1e+06 t839_leaf0
   <= 1.000000999999e+06
 T839_R_f147_k1: const_Roof_Style_Flat - 1e+06 t839_leaf1 >= -999999
 T840_onehot: t840_leaf0 = 1
 T841_onehot: t841_leaf0 + t841_leaf1 = 1
 T841_L_f214_k0: x_b1_is_LwQ + 1e+06 t841_leaf0 <= 1.000000999999e+06
 T841_R_f214_k1: x_b1_is_LwQ - 1e+06 t841_leaf1 >= -999999
 T842_onehot: t842_leaf0 + t842_leaf1 = 1
 T842_L_f112_k0: const_Neighborhood_SawyerW + 1e+06 t842_leaf0
   <= 1.000000999999e+06
 T842_R_f112_k1: const_Neighborhood_SawyerW - 1e+06 t842_leaf1 >= -999999
 T843_onehot: t843_leaf0 = 1
 T844_onehot: t844_leaf0 + t844_leaf1 = 1
 T844_L_f33_k0: x_Garage_Area + 1e+06 t844_leaf0 <= 1.000239999999e+06
 T844_R_f33_k1: x_Garage_Area - 1e+06 t844_leaf1 >= -999760
 T845_onehot: t845_leaf0 + t845_leaf1 + t845_leaf2 = 1
 T845_L_f33_k0: x_Garage_Area + 1e+06 t845_leaf0 <= 1.000643999999e+06
 T845_R_f33_k1: x_Garage_Area - 1e+06 t845_leaf1 >= -999356
 T845_L_f29_k1: const_Fireplaces + 1e+06 t845_leaf1 <= 1.000000999999e+06
 T845_R_f33_k2: x_Garage_Area - 1e+06 t845_leaf2 >= -999356
 T845_R_f29_k2: const_Fireplaces - 1e+06 t845_leaf2 >= -999999
 T846_onehot: t846_leaf0 = 1
 T847_onehot: t847_leaf0 = 1
 T848_onehot: t848_leaf0 + t848_leaf1 + t848_leaf2 = 1
 T848_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t848_leaf0
   <= 1.000000999999e+06
 T848_L_f38_k0: x_Enclosed_Porch + 1e+06 t848_leaf0 <= 1.000090999999e+06
 T848_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t848_leaf1
   <= 1.000000999999e+06
 T848_R_f38_k1: x_Enclosed_Porch - 1e+06 t848_leaf1 >= -999909
 T848_R_f245_k2: const_FunctioNo_aplical_Typ - 1e+06 t848_leaf2 >= -999999
 T849_onehot: t849_leaf0 + t849_leaf1 + t849_leaf2 = 1
 T849_L_f13_k0: x_BsmtFin_SF_2 + 1e+06 t849_leaf0 <= 1.000059999999e+06
 T849_L_f37_k0: x_Open_Porch_SF + 1e+06 t849_leaf0 <= 1.000188999999e+06
 T849_L_f13_k1: x_BsmtFin_SF_2 + 1e+06 t849_leaf1 <= 1.000059999999e+06
 T849_R_f37_k1: x_Open_Porch_SF - 1e+06 t849_leaf1 >= -999811
 T849_R_f13_k2: x_BsmtFin_SF_2 - 1e+06 t849_leaf2 >= -999940
 T850_onehot: t850_leaf0 = 1
 T851_onehot: t851_leaf0 = 1
 T852_onehot: t852_leaf0 = 1
 T853_onehot: t853_leaf0 = 1
 T854_onehot: t854_leaf0 + t854_leaf1 = 1
 T854_L_f7_k0: x_Mas_Vnr_Area + 1e+06 t854_leaf0 <= 1.000859999999e+06
 T854_R_f7_k1: x_Mas_Vnr_Area - 1e+06 t854_leaf1 >= -999140
 T855_onehot: t855_leaf0 = 1
 T856_onehot: t856_leaf0 + t856_leaf1 = 1
 T856_L_f72_k0: const_Alley_Pave + 1e+06 t856_leaf0 <= 1.000000999999e+06
 T856_R_f72_k1: const_Alley_Pave - 1e+06 t856_leaf1 >= -999999
 T857_onehot: t857_leaf0 = 1
 T858_onehot: t858_leaf0 + t858_leaf1 + t858_leaf2 + t858_leaf3 = 1
 T858_L_f293_k0: const_Sale_Condition_Abnorml + 1e+06 t858_leaf0
   <= 1.000000999999e+06
 T858_L_f21_k0: const_Bsmt_Full_Bath + 1e+06 t858_leaf0
   <= 1.000001999999e+06
 T858_L_f293_k1: const_Sale_Condition_Abnorml + 1e+06 t858_leaf1
   <= 1.000000999999e+06
 T858_R_f21_k1: const_Bsmt_Full_Bath - 1e+06 t858_leaf1 >= -999998
 T858_R_f293_k2: const_Sale_Condition_Abnorml - 1e+06 t858_leaf2
   >= -999999
 T858_L_f30_k2: x_Fireplace_Qu + 1e+06 t858_leaf2 <= 1.000002999999e+06
 T858_R_f293_k3: const_Sale_Condition_Abnorml - 1e+06 t858_leaf3
   >= -999999
 T858_R_f30_k3: x_Fireplace_Qu - 1e+06 t858_leaf3 >= -999997
 T859_onehot: t859_leaf0 = 1
 T860_onehot: t860_leaf0 + t860_leaf1 + t860_leaf2 = 1
 T860_L_f20_k0: x_Gr_Liv_Area + 1e+06 t860_leaf0 <= 1.000746999999e+06
 T860_L_f35_k0: x_Garage_Cond + 1e+06 t860_leaf0 <= 1.000001999999e+06
 T860_L_f20_k1: x_Gr_Liv_Area + 1e+06 t860_leaf1 <= 1.000746999999e+06
 T860_R_f35_k1: x_Garage_Cond - 1e+06 t860_leaf1 >= -999998
 T860_R_f20_k2: x_Gr_Liv_Area - 1e+06 t860_leaf2 >= -999253
 T861_onehot: t861_leaf0 = 1
 T862_onehot: t862_leaf0 = 1
 T863_onehot: t863_leaf0 = 1
 T864_onehot: t864_leaf0 = 1
 T865_onehot: t865_leaf0 + t865_leaf1 + t865_leaf2 = 1
 T865_L_f296_k0: const_Sale_Condition_Family + 1e+06 t865_leaf0
   <= 1.000000999999e+06
 T865_R_f296_k1: const_Sale_Condition_Family - 1e+06 t865_leaf1 >= -999999
 T865_L_f12_k1: x_BsmtFin_SF_1 + 1e+06 t865_leaf1 <= 1.000166999999e+06
 T865_R_f296_k2: const_Sale_Condition_Family - 1e+06 t865_leaf2 >= -999999
 T865_R_f12_k2: x_BsmtFin_SF_1 - 1e+06 t865_leaf2 >= -999833
 T866_onehot: t866_leaf0 = 1
 T867_onehot: t867_leaf0 = 1
 T868_onehot: t868_leaf0 = 1
 T869_onehot: t869_leaf0 = 1
 T870_onehot: t870_leaf0 = 1
 T871_onehot: t871_leaf0 = 1
 T872_onehot: t872_leaf0 = 1
 T873_onehot: t873_leaf0 = 1
 T874_onehot: t874_leaf0 = 1
 T875_onehot: t875_leaf0 + t875_leaf1 = 1
 T875_L_f20_k0: x_Gr_Liv_Area + 1e+06 t875_leaf0 <= 1.000815999999e+06
 T875_R_f20_k1: x_Gr_Liv_Area - 1e+06 t875_leaf1 >= -999184
 T876_onehot: t876_leaf0 + t876_leaf1 + t876_leaf2 + t876_leaf3 = 1
 T876_L_f164_k0: x_ex1_is_BrkFace + 1e+06 t876_leaf0 <= 1.000000999999e+06
 T876_L_f35_k0: x_Garage_Cond + 1e+06 t876_leaf0 <= 1.000001999999e+06
 T876_L_f1_k0: const_Lot_Area + 1e+06 t876_leaf0 <= 1.008124999999e+06
 T876_L_f164_k1: x_ex1_is_BrkFace + 1e+06 t876_leaf1 <= 1.000000999999e+06
 T876_L_f35_k1: x_Garage_Cond + 1e+06 t876_leaf1 <= 1.000001999999e+06
 T876_R_f1_k1: const_Lot_Area - 1e+06 t876_leaf1 >= -991875
 T876_L_f164_k2: x_ex1_is_BrkFace + 1e+06 t876_leaf2 <= 1.000000999999e+06
 T876_R_f35_k2: x_Garage_Cond - 1e+06 t876_leaf2 >= -999998
 T876_R_f164_k3: x_ex1_is_BrkFace - 1e+06 t876_leaf3 >= -999999
 T877_onehot: t877_leaf0 + t877_leaf1 = 1
 T877_L_f0_k0: const_Lot_Frontage + 1e+06 t877_leaf0 <= 1.000108999999e+06
 T877_R_f0_k1: const_Lot_Frontage - 1e+06 t877_leaf1 >= -999891
 T878_onehot: t878_leaf0 = 1
 T879_onehot: t879_leaf0 = 1
 T880_onehot: t880_leaf0 + t880_leaf1 + t880_leaf2 + t880_leaf3 = 1
 T880_L_f1_k0: const_Lot_Area + 1e+06 t880_leaf0 <= 1.020430999999e+06
 T880_R_f1_k1: const_Lot_Area - 1e+06 t880_leaf1 >= -979569
 T880_L_f17_k1: x_1st_Flr_SF + 1e+06 t880_leaf1 <= 1.001240999999e+06
 T880_R_f1_k2: const_Lot_Area - 1e+06 t880_leaf2 >= -979569
 T880_R_f17_k2: x_1st_Flr_SF - 1e+06 t880_leaf2 >= -998759
 T880_L_f82_k2: const_Lot_Config_CulDSac + 1e+06 t880_leaf2
   <= 1.000000999999e+06
 T880_R_f1_k3: const_Lot_Area - 1e+06 t880_leaf3 >= -979569
 T880_R_f17_k3: x_1st_Flr_SF - 1e+06 t880_leaf3 >= -998759
 T880_R_f82_k3: const_Lot_Config_CulDSac - 1e+06 t880_leaf3 >= -999999
 T881_onehot: t881_leaf0 + t881_leaf1 + t881_leaf2 + t881_leaf3
   + t881_leaf4 = 1
 T881_L_f245_k0: const_FunctioNo_aplical_Typ + 1e+06 t881_leaf0
   <= 1.000000999999e+06
 T881_L_f44_k0: const_Yr_Sold + 1e+06 t881_leaf0 <= 1.002008999999e+06
 T881_L_f12_k0: x_BsmtFin_SF_1 + 1e+06 t881_leaf0 <= 1.000187999999e+06
 T881_L_f245_k1: const_FunctioNo_aplical_Typ + 1e+06 t881_leaf1
   <= 1.000000999999e+06
 T881_L_f44_k1: const_Yr_Sold + 1e+06 t881_leaf1 <= 1.002008999999e+06
 T881_R_f12_k1: x_BsmtFin_SF_1 - 1e+06 t881_leaf1 >= -999812
 T881_L_f245_k2: const_FunctioNo_aplical_Typ + 1e+06 t881_leaf2
   <= 1.000000999999e+06
 T881_R_f44_k2: const_Yr_Sold - 1e+06 t881_leaf2 >= -997991
 T881_R_f245_k3: const_FunctioNo_aplical_Typ - 1e+06 t881_leaf3 >= -999999
 T881_L_f30_k3: x_Fireplace_Qu + 1e+06 t881_leaf3 <= 999999.999999
 T881_R_f245_k4: const_FunctioNo_aplical_Typ - 1e+06 t881_leaf4 >= -999999
 T881_R_f30_k4: x_Fireplace_Qu - 1e+06 t881_leaf4 >= -1e+06
 T882_onehot: t882_leaf0 = 1
 T883_onehot: t883_leaf0 + t883_leaf1 = 1
 T883_L_f292_k0: const_Sale_Type_WD_ + 1e+06 t883_leaf0
   <= 1.000000999999e+06
 T883_R_f292_k1: const_Sale_Type_WD_ - 1e+06 t883_leaf1 >= -999999
 T884_onehot: t884_leaf0 = 1
 T885_onehot: t885_leaf0 = 1
 T886_onehot: t886_leaf0 + t886_leaf1 + t886_leaf2 + t886_leaf3
   + t886_leaf4 + t886_leaf5 = 1
 T886_L_f38_k0: x_Enclosed_Porch + 1e+06 t886_leaf0 <= 1.000077999999e+06
 T886_L_f38_k0: x_Enclosed_Porch + 1e+06 t886_leaf0 <= 1.000055999999e+06
 T886_L_f28_k0: x_TotRms_AbvGrd + 1e+06 t886_leaf0 <= 1.000004999999e+06
 T886_L_f38_k1: x_Enclosed_Porch + 1e+06 t886_leaf1 <= 1.000077999999e+06
 T886_L_f38_k1: x_Enclosed_Porch + 1e+06 t886_leaf1 <= 1.000055999999e+06
 T886_R_f28_k1: x_TotRms_AbvGrd - 1e+06 t886_leaf1 >= -999995
 T886_L_f38_k2: x_Enclosed_Porch + 1e+06 t886_leaf2 <= 1.000077999999e+06
 T886_R_f38_k2: x_Enclosed_Porch - 1e+06 t886_leaf2 >= -999944
 T886_R_f38_k3: x_Enclosed_Porch - 1e+06 t886_leaf3 >= -999922
 T886_L_f49_k3: const_MS_SubClass_50 + 1e+06 t886_leaf3
   <= 1.000000999999e+06
 T886_L_f275_k3: const_Mo_Sold_5 + 1e+06 t886_leaf3 <= 1.000000999999e+06
 T886_R_f38_k4: x_Enclosed_Porch - 1e+06 t886_leaf4 >= -999922
 T886_L_f49_k4: const_MS_SubClass_50 + 1e+06 t886_leaf4
   <= 1.000000999999e+06
 T886_R_f275_k4: const_Mo_Sold_5 - 1e+06 t886_leaf4 >= -999999
 T886_R_f38_k5: x_Enclosed_Porch - 1e+06 t886_leaf5 >= -999922
 T886_R_f49_k5: const_MS_SubClass_50 - 1e+06 t886_leaf5 >= -999999
 T887_onehot: t887_leaf0 = 1
 T888_onehot: t888_leaf0 + t888_leaf1 = 1
 T888_L_f4_k0: const_Overall_Cond + 1e+06 t888_leaf0 <= 1.000006999999e+06
 T888_R_f4_k1: const_Overall_Cond - 1e+06 t888_leaf1 >= -999993
 T889_onehot: t889_leaf0 + t889_leaf1 + t889_leaf2 + t889_leaf3
   + t889_leaf4 = 1
 T889_L_f20_k0: x_Gr_Liv_Area + 1e+06 t889_leaf0 <= 1.000882999999e+06
 T889_L_f29_k0: const_Fireplaces + 1e+06 t889_leaf0 <= 1.000000999999e+06
 T889_L_f20_k1: x_Gr_Liv_Area + 1e+06 t889_leaf1 <= 1.000882999999e+06
 T889_R_f29_k1: const_Fireplaces - 1e+06 t889_leaf1 >= -999999
 T889_R_f20_k2: x_Gr_Liv_Area - 1e+06 t889_leaf2 >= -999117
 T889_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t889_leaf2
   <= 1.002222999999e+06
 T889_L_f15_k2: const_Total_Bsmt_SF + 1e+06 t889_leaf2
   <= 1.001135999999e+06
 T889_R_f20_k3: x_Gr_Liv_Area - 1e+06 t889_leaf3 >= -999117
 T889_L_f15_k3: const_Total_Bsmt_SF + 1e+06 t889_leaf3
   <= 1.002222999999e+06
 T889_R_f15_k3: const_Total_Bsmt_SF - 1e+06 t889_leaf3 >= -998864
 T889_R_f20_k4: x_Gr_Liv_Area - 1e+06 t889_leaf4 >= -999117
 T889_R_f15_k4: const_Total_Bsmt_SF - 1e+06 t889_leaf4 >= -997777
 T890_onehot: t890_leaf0 = 1
 T891_onehot: t891_leaf0 = 1
 T892_onehot: t892_leaf0 = 1
 T893_onehot: t893_leaf0 = 1
 T894_onehot: t894_leaf0 + t894_leaf1 = 1
 T894_L_f33_k0: x_Garage_Area + 1e+06 t894_leaf0 <= 1.000870999999e+06
 T894_R_f33_k1: x_Garage_Area - 1e+06 t894_leaf1 >= -999129
 T895_onehot: t895_leaf0 = 1
 T896_onehot: t896_leaf0 + t896_leaf1 = 1
 T896_L_f29_k0: const_Fireplaces + 1e+06 t896_leaf0 <= 1.000001999999e+06
 T896_R_f29_k1: const_Fireplaces - 1e+06 t896_leaf1 >= -999998
 T897_onehot: t897_leaf0 = 1
 T898_onehot: t898_leaf0 = 1
 T899_onehot: t899_leaf0 = 1
 T900_onehot: t900_leaf0 = 1
 T901_onehot: t901_leaf0 = 1
 T902_onehot: t902_leaf0 = 1
 T903_onehot: t903_leaf0 + t903_leaf1 + t903_leaf2 = 1
 T903_L_f15_k0: const_Total_Bsmt_SF + 1e+06 t903_leaf0
   <= 1.000482999999e+06
 T903_R_f15_k1: const_Total_Bsmt_SF - 1e+06 t903_leaf1 >= -999517
 T903_L_f37_k1: x_Open_Porch_SF + 1e+06 t903_leaf1 <= 1.000287999999e+06
 T903_R_f15_k2: const_Total_Bsmt_SF - 1e+06 t903_leaf2 >= -999517
 T903_R_f37_k2: x_Open_Porch_SF - 1e+06 t903_leaf2 >= -999712
 T904_onehot: t904_leaf0 + t904_leaf1 = 1
 T904_L_f3_k0: Overall_Qual_calc + 1e+06 t904_leaf0 <= 1.000002999999e+06
 T904_R_f3_k1: Overall_Qual_calc - 1e+06 t904_leaf1 >= -999997
 T905_onehot: t905_leaf0 = 1
 T906_onehot: t906_leaf0 + t906_leaf1 + t906_leaf2 = 1
 T906_L_f273_k0: const_Mo_Sold_3 + 1e+06 t906_leaf0 <= 1.000000999999e+06
 T906_L_f32_k0: const_Garage_Cars + 1e+06 t906_leaf0 <= 1.000002999999e+06
 T906_L_f273_k1: const_Mo_Sold_3 + 1e+06 t906_leaf1 <= 1.000000999999e+06
 T906_R_f32_k1: const_Garage_Cars - 1e+06 t906_leaf1 >= -999997
 T906_R_f273_k2: const_Mo_Sold_3 - 1e+06 t906_leaf2 >= -999999
 T907_onehot: t907_leaf0 = 1
 T908_onehot: t908_leaf0 = 1
 T909_onehot: t909_leaf0 + t909_leaf1 = 1
 T909_L_f5_k0: const_Year_Built + 1e+06 t909_leaf0 <= 1.001910999999e+06
 T909_R_f5_k1: const_Year_Built - 1e+06 t909_leaf1 >= -998089
 T910_onehot: t910_leaf0 = 1
 T911_onehot: t911_leaf0 = 1
 T912_onehot: t912_leaf0 = 1
 T913_onehot: t913_leaf0 = 1
 YLOG_sum_leaves: y_log_raw + 0.0227153655 t0_leaf0 + 0.0109316884 t0_leaf1
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
   + 9.69016e-05 t913_leaf0 = 12.437748
 YLOG_with_offset: - y_log_raw + y_log = 0
 COST_eq_linexpr: - 17352 x_roof_matl_is_ClyTile
   - 8011.95 x_roof_matl_is_Membran - 11739 x_roof_matl_is_Metal
   - 7600 x_roof_matl_is_Roll - 8550 x_roof_matl_is_Tar&Grv
   - 22500 x_roof_matl_is_WdShake - 19500 x_roof_matl_is_WdShngl
   - 3300 x_elect_is_FuseA - 2475 x_elect_is_FuseF - 1650 x_elect_is_FuseP
   - 1875 x_elect_is_Mix - 20769 x_garage_finish_is_RFn
   - 17500 x_garage_finish_is_Unf - 28000 x_AddFull - 11200 x_AddHalf
   - 16800 x_AddKitch - 25480 x_AddBed - 51659 x_garage_qual_is_Ex
   - 51659 x_garage_cond_is_Ex - 37849 x_garage_qual_is_Gd
   - 37849 x_garage_cond_is_Gd - 14113 x_garage_qual_is_Fa
   - 14113 x_garage_cond_is_Fa - 4188 x_garage_qual_is_Po
   - 4188 x_garage_cond_is_Po - 4908 x_paved_drive_is_Y
   - 1800 x_paved_drive_is_N - 11940 x_fence_is_GdPrv
   - 10340 x_fence_is_MnPrv - 3232 x_fence_is_GdWo - 2400 x_fence_is_MnWw
   - 19000 x_ex1_is_AsbShng - 22500 x_ex1_is_AsphShn
   - 26000 x_ex1_is_BrkComm - 10300 x_ex1_is_CBlock
   - 14674 x_ex1_is_CemntBd - 21300 x_ex1_is_HdBoard
   - 16500 x_ex1_is_ImStucc - 11196 x_ex1_is_MetalSd
   - 21765.3125 x_ex1_is_Other - 3461.81 x_ex1_is_Plywood
   - 17625 x_ex1_is_PreCast - 106250 x_ex1_is_Stone - 5629 x_ex1_is_Stucco
   - 17410 x_ex1_is_VinylSd - 12500 x_ex1_is_Wd_Sdng
   - 21900 x_ex1_is_WdShngl - 19000 x_ex2_is_AsbShng
   - 22500 x_ex2_is_AsphShn - 26000 x_ex2_is_BrkComm
   - 22000 x_ex2_is_BrkFace - 10300 x_ex2_is_CBlock
   - 14674 x_ex2_is_CemntBd - 21300 x_ex2_is_HdBoard
   - 16500 x_ex2_is_ImStucc - 11196 x_ex2_is_MetalSd
   - 21765.3125 x_ex2_is_Other - 17625 x_ex2_is_PreCast
   - 106250 x_ex2_is_Stone - 5629 x_ex2_is_Stucco - 17410 x_ex2_is_VinylSd
   - 12500 x_ex2_is_Wd_Sdng - 21900 x_ex2_is_WdShngl - 75000 x_b1_is_GLQ
   - 53500 x_b1_is_ALQ - 23500 x_b1_is_Rec - 15000 x_b1_is_LwQ
   - 11250 x_b1_is_Unf - 75000 x_b2_is_GLQ - 53500 x_b2_is_ALQ
   - 32000 x_b2_is_BLQ - 23500 x_b2_is_Rec - 15000 x_b2_is_LwQ
   - 13601.28 x_z10_GarageArea - 27202.56 x_z20_GarageArea
   - 40803.84 x_z30_GarageArea - 1050 x_z10_WoodDeckSF
   - 2100 x_z20_WoodDeckSF - 3150 x_z30_WoodDeckSF
   - 480.5 x_z10_OpenPorchSF - 961 x_z20_OpenPorchSF
   - 1441.5 x_z30_OpenPorchSF - 42500 kit_is_Gd - 180000 kit_is_Ex
   - 7646.7 exterqual_is_Po - 14558 exterqual_is_Fa
   - 22833.06 exterqual_is_Gd - 106250 exterqual_is_Ex
   - 7646.7 extercond_is_Po - 14558 extercond_is_Fa
   - 22833.06 extercond_is_Gd - 106250 extercond_is_Ex
   + 26.29 mv_area_mvt_BrkCmn + 12.5 mv_area_mvt_BrkFace
   + 5 mv_area_mvt_CBlock - 10850 roof_change_mat - 10304 x_Add1stFlr
   - 3750 heat_qc_is_Po - 6500 heat_qc_is_TA - 8250 heat_qc_is_Gd
   - 10000 heat_qc_is_Ex + 2025 fireplace_is_Po + 1525 fireplace_is_Fa
   + 1025 fireplace_is_TA - 1025 fireplace_is_Ex
   + 3525 fireplace_is_No_aplica - 14112 bsmt_finish_all
   - 10750 bsmtcond_is_Ex + cost_model = 0
 BUDGET: cost_model <= 60000
 NOCHG_cost_link: - 60000 z_change + cost_model <= 0
 NOCHG_cost_lb: - z_change + cost_model >= 0
 MIN_SPEND_IF_CHANGE: - 8000 z_change + cost_model >= 0
 NOCHG_pos_Utilities: x_Utilities <= 3
 NOCHG_neg_Utilities: - x_Utilities - 3 z_change <= -3
 NOCHG_pos_Mas_Vnr_Area: x_Mas_Vnr_Area - 1888 z_change <= 112
 NOCHG_neg_Mas_Vnr_Area: - x_Mas_Vnr_Area - 112 z_change <= -112
 NOCHG_pos_Exter_Qual: x_Exter_Qual - 2 z_change <= 2
 NOCHG_neg_Exter_Qual: - x_Exter_Qual - 3 z_change <= -2
 NOCHG_pos_Exter_Cond: x_Exter_Cond - 2 z_change <= 2
 NOCHG_neg_Exter_Cond: - x_Exter_Cond - 3 z_change <= -2
 NOCHG_pos_Bsmt_Cond: x_Bsmt_Cond - z_change <= 3
 NOCHG_neg_Bsmt_Cond: - x_Bsmt_Cond - 4 z_change <= -3
 NOCHG_pos_BsmtFin_SF_1: x_BsmtFin_SF_1 - 999361 z_change <= 639
 NOCHG_neg_BsmtFin_SF_1: - x_BsmtFin_SF_1 - 639 z_change <= -639
 NOCHG_pos_BsmtFin_SF_2: x_BsmtFin_SF_2 - 100000 z_change <= 0
 NOCHG_neg_BsmtFin_SF_2: - x_BsmtFin_SF_2 <= 0
 NOCHG_pos_Bsmt_Unf_SF: x_Bsmt_Unf_SF - 99559 z_change <= 441
 NOCHG_neg_Bsmt_Unf_SF: - x_Bsmt_Unf_SF - 441 z_change <= -441
 NOCHG_pos_Heating_QC: x_Heating_QC - 3 z_change <= 1
 NOCHG_neg_Heating_QC: - x_Heating_QC - 2 z_change <= -1
 NOCHG_pos_1st_Flr_SF: x_1st_Flr_SF - 3344 z_change <= 1656
 NOCHG_neg_1st_Flr_SF: - x_1st_Flr_SF - 1656 z_change <= -1656
 NOCHG_pos_Gr_Liv_Area: x_Gr_Liv_Area - 8344 z_change <= 1656
 NOCHG_neg_Gr_Liv_Area: - x_Gr_Liv_Area - 1656 z_change <= -1656
 NOCHG_pos_Full_Bath: x_Full_Bath - 3 z_change <= 1
 NOCHG_neg_Full_Bath: - x_Full_Bath - z_change <= -1
 NOCHG_pos_Half_Bath: x_Half_Bath - 5 z_change <= 0
 NOCHG_neg_Half_Bath: - x_Half_Bath <= 0
 NOCHG_pos_Bedroom_AbvGr: x_Bedroom_AbvGr - 3 z_change <= 3
 NOCHG_neg_Bedroom_AbvGr: - x_Bedroom_AbvGr - 3 z_change <= -3
 NOCHG_pos_Kitchen_AbvGr: x_Kitchen_AbvGr - 2 z_change <= 1
 NOCHG_neg_Kitchen_AbvGr: - x_Kitchen_AbvGr - z_change <= -1
 NOCHG_pos_Kitchen_Qual: x_Kitchen_Qual - 2 z_change <= 2
 NOCHG_neg_Kitchen_Qual: - x_Kitchen_Qual - 3 z_change <= -2
 NOCHG_pos_TotRms_AbvGrd: x_TotRms_AbvGrd - 8 z_change <= 7
 NOCHG_neg_TotRms_AbvGrd: - x_TotRms_AbvGrd - 7 z_change <= -7
 NOCHG_pos_Fireplace_Qu: x_Fireplace_Qu - z_change <= 3
 NOCHG_neg_Fireplace_Qu: - x_Fireplace_Qu - 4 z_change <= -3
 NOCHG_pos_Garage_Area: x_Garage_Area - 99472 z_change <= 528
 NOCHG_neg_Garage_Area: - x_Garage_Area - 528 z_change <= -528
 NOCHG_pos_Garage_Qual: x_Garage_Qual - 2 z_change <= 2
 NOCHG_neg_Garage_Qual: - x_Garage_Qual - 3 z_change <= -2
 NOCHG_pos_Garage_Cond: x_Garage_Cond - 2 z_change <= 2
 NOCHG_neg_Garage_Cond: - x_Garage_Cond - 3 z_change <= -2
 NOCHG_pos_Wood_Deck_SF: x_Wood_Deck_SF - 99790 z_change <= 210
 NOCHG_neg_Wood_Deck_SF: - x_Wood_Deck_SF - 210 z_change <= -210
 NOCHG_pos_Open_Porch_SF: x_Open_Porch_SF - 99938 z_change <= 62
 NOCHG_neg_Open_Porch_SF: - x_Open_Porch_SF - 62 z_change <= -62
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
 NOCHG_pos_Exterior_1st_BrkFace: x_ex1_is_BrkFace <= 1
 NOCHG_neg_Exterior_1st_BrkFace: - x_ex1_is_BrkFace - z_change <= -1
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
 NOCHG_pos_Exterior_2nd_MetalSd: x_ex2_is_MetalSd - z_change <= 0
 NOCHG_neg_Exterior_2nd_MetalSd: - x_ex2_is_MetalSd <= 0
 NOCHG_pos_Exterior_2nd_Other: x_ex2_is_Other - z_change <= 0
 NOCHG_neg_Exterior_2nd_Other: - x_ex2_is_Other <= 0
 NOCHG_pos_Exterior_2nd_Plywood: x_ex2_is_Plywood <= 1
 NOCHG_neg_Exterior_2nd_Plywood: - x_ex2_is_Plywood - z_change <= -1
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
 NOCHG_pos_Mas_Vnr_Type_No_aplica: x_mvt_is_No_aplica - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_No_aplica: - x_mvt_is_No_aplica <= 0
 NOCHG_pos_Mas_Vnr_Type_NoAplica: x_mvt_is_No_aplica - z_change <= 0
 NOCHG_neg_Mas_Vnr_Type_NoAplica: - x_mvt_is_No_aplica <= 0
 NOCHG_pos_Mas_Vnr_Type_Stone: x_mvt_is_Stone <= 1
 NOCHG_neg_Mas_Vnr_Type_Stone: - x_mvt_is_Stone - z_change <= -1
 NOCHG_pos_BsmtFin_Type_1_ALQ: x_b1_is_ALQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_ALQ: - x_b1_is_ALQ <= 0
 NOCHG_pos_BsmtFin_Type_1_BLQ: x_b1_is_BLQ <= 1
 NOCHG_neg_BsmtFin_Type_1_BLQ: - x_b1_is_BLQ - z_change <= -1
 NOCHG_pos_BsmtFin_Type_1_GLQ: x_b1_is_GLQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_GLQ: - x_b1_is_GLQ <= 0
 NOCHG_pos_BsmtFin_Type_1_LwQ: x_b1_is_LwQ - z_change <= 0
 NOCHG_neg_BsmtFin_Type_1_LwQ: - x_b1_is_LwQ <= 0
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
 NOCHG_pos_BsmtFin_Type_2_Rec: x_b2_is_Rec - z_change <= 0
 NOCHG_neg_BsmtFin_Type_2_Rec: - x_b2_is_Rec <= 0
 NOCHG_pos_BsmtFin_Type_2_Unf: x_b2_is_Unf <= 1
 NOCHG_neg_BsmtFin_Type_2_Unf: - x_b2_is_Unf - z_change <= -1
 NOCHG_pos_Heating_Floor: x_Heating_Floor - z_change <= 0
 NOCHG_neg_Heating_Floor: - x_Heating_Floor <= 0
 NOCHG_pos_Heating_GasA: x_Heating_GasA <= 1
 NOCHG_neg_Heating_GasA: - x_Heating_GasA - z_change <= -1
 NOCHG_pos_Heating_GasW: x_Heating_GasW - z_change <= 0
 NOCHG_neg_Heating_GasW: - x_Heating_GasW <= 0
 NOCHG_pos_Heating_Grav: x_Heating_Grav - z_change <= 0
 NOCHG_neg_Heating_Grav: - x_Heating_Grav <= 0
 NOCHG_pos_Heating_OthW: x_Heating_OthW - z_change <= 0
 NOCHG_neg_Heating_OthW: - x_Heating_OthW <= 0
 NOCHG_pos_Heating_Wall: x_Heating_Wall - z_change <= 0
 NOCHG_neg_Heating_Wall: - x_Heating_Wall <= 0
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
 NOCHG_pos_Paved_Drive_P: x_paved_drive_is_P <= 1
 NOCHG_neg_Paved_Drive_P: - x_paved_drive_is_P - z_change <= -1
 NOCHG_pos_Paved_Drive_Y: x_paved_drive_is_Y - z_change <= 0
 NOCHG_neg_Paved_Drive_Y: - x_paved_drive_is_Y <= 0
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
 NOCHG_price_pos: y_price - 2e+06 z_change <= 301951.4375
 NOCHG_price_neg: - y_price - 2e+06 z_change <= -301951.4375
 MIN_PRICE_BASE: y_price >= 301951.437499
 NO_NEGATIVE_ROI: y_price - cost_model >= 301951.437499
Bounds
 x_Bedroom_AbvGr <= 6
 x_Full_Bath <= 4
 x_BsmtFin_SF_1 <= 1e+06
 x_BsmtFin_SF_2 <= 100000
 x_Bsmt_Unf_SF <= 100000
 x_Gr_Liv_Area <= 10000
 x_1st_Flr_SF <= 5000
 x_TotRms_AbvGrd <= 15
 x_Mas_Vnr_Area = 112
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
 x_garage_finish_is_RFn = 0
 x_garage_finish_is_Unf = 0
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
 1 <= Overall_Qual_calc <= 10
 kit_is_Po = 0
 kit_is_Fa = 0
 exterqual_is_Po = 0
 exterqual_is_Fa = 0
 extercond_is_Po = 0
 extercond_is_Fa = 0
 mv_area_mvt_BrkCmn <= 2000
 mv_area_mvt_BrkFace <= 2000
 mv_area_mvt_CBlock <= 2000
 central_air_yes = 1
 poolqc_is_Po = 0
 poolqc_is_Fa = 0
 poolqc_is_TA = 0
 poolqc_is_Gd = 0
 poolqc_is_Ex = 0
 poolqc_is_No_aplica = 1
 UpgPool = 0
 fireplace_is_Po = 0
 fireplace_is_Fa = 0
 fireplace_is_TA = 0
 fireplace_is_No_aplica = 0
 bsmtcond_is_Po = 0
 bsmtcond_is_Fa = 0
 bsmtcond_is_TA = 0
 B1_upg_flag = 0
 B2_upg_flag = 1
 util_ELO = 0
 util_NoSeWa = 0
 util_NoSewr = 0
 R8_other_rooms <= 15
 const_Lot_Frontage = 141
 const_Lot_Area = 31770
 const_Overall_Cond = 5
 const_Year_Built = 1960
 const_Year_Remod_Add = 1960
 const_Bsmt_Qual = 2
 const_Total_Bsmt_SF = 1080
 const_2nd_Flr_SF = 0
 const_Low_Qual_Fin_SF = 0
 const_Bsmt_Full_Bath = 1
 const_Bsmt_Half_Bath = 0
 const_Fireplaces = 2
 const_Garage_Yr_Blt = 1960
 const_Garage_Cars = 2
 const_Misc_Val = 0
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
 const_Bsmt_Exposure_Gd = 1
 const_Bsmt_Exposure_Mn = 0
 const_Bsmt_Exposure_No = 0
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
 const_Garage_Finish_Fin = 1
 const_Garage_Finish_No_aplica = 0
 const_Garage_Finish_RFn = 0
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
 const_Mo_Sold_5 = 1
 const_Mo_Sold_6 = 0
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
 extercond_is_TA extercond_is_Gd extercond_is_Ex roof_change_mat
 central_air_yes central_air_no poolqc_is_Po poolqc_is_Fa poolqc_is_TA
 poolqc_is_Gd poolqc_is_Ex poolqc_is_No_aplica UpgPool x_Add1stFlr
 HEAT_any_rebuild heat_qc_is_Po heat_qc_is_Fa heat_qc_is_TA heat_qc_is_Gd
 heat_qc_is_Ex fireplace_is_Po fireplace_is_Fa fireplace_is_TA
 fireplace_is_Gd fireplace_is_Ex fireplace_is_No_aplica bsmt_finish_all
 bsmtcond_is_Po bsmtcond_is_Fa bsmtcond_is_TA bsmtcond_is_Gd bsmtcond_is_Ex
 B1_upg_flag B2_upg_flag util_ELO util_NoSeWa util_NoSewr util_AllPub
 t0_leaf0 t0_leaf1 t0_leaf2 t0_leaf3 t0_leaf4 t0_leaf5 t0_leaf6 t0_leaf7
 t1_leaf0 t1_leaf1 t1_leaf2 t1_leaf3 t1_leaf4 t1_leaf5 t1_leaf6 t1_leaf7
 t2_leaf0 t2_leaf1 t2_leaf2 t2_leaf3 t2_leaf4 t2_leaf5 t2_leaf6 t2_leaf7
 t3_leaf0 t3_leaf1 t3_leaf2 t3_leaf3 t3_leaf4 t3_leaf5 t3_leaf6 t3_leaf7
 t4_leaf0 t4_leaf1 t4_leaf2 t4_leaf3 t4_leaf4 t4_leaf5 t4_leaf6 t4_leaf7
 t5_leaf0 t5_leaf1 t5_leaf2 t5_leaf3 t5_leaf4 t5_leaf5 t5_leaf6 t5_leaf7
 t6_leaf0 t6_leaf1 t6_leaf2 t6_leaf3 t6_leaf4 t6_leaf5 t6_leaf6 t6_leaf7
 t7_leaf0 t7_leaf1 t7_leaf2 t7_leaf3 t7_leaf4 t7_leaf5 t7_leaf6 t7_leaf7
 t8_leaf0 t8_leaf1 t8_leaf2 t8_leaf3 t8_leaf4 t8_leaf5 t8_leaf6 t8_leaf7
 t9_leaf0 t9_leaf1 t9_leaf2 t9_leaf3 t9_leaf4 t9_leaf5 t9_leaf6 t9_leaf7
 t10_leaf0 t10_leaf1 t10_leaf2 t10_leaf3 t10_leaf4 t10_leaf5 t10_leaf6
 t10_leaf7 t11_leaf0 t11_leaf1 t11_leaf2 t11_leaf3 t11_leaf4 t11_leaf5
 t11_leaf6 t11_leaf7 t12_leaf0 t12_leaf1 t12_leaf2 t12_leaf3 t12_leaf4
 t12_leaf5 t12_leaf6 t12_leaf7 t13_leaf0 t13_leaf1 t13_leaf2 t13_leaf3
 t13_leaf4 t13_leaf5 t13_leaf6 t13_leaf7 t14_leaf0 t14_leaf1 t14_leaf2
 t14_leaf3 t14_leaf4 t14_leaf5 t14_leaf6 t14_leaf7 t15_leaf0 t15_leaf1
 t15_leaf2 t15_leaf3 t15_leaf4 t15_leaf5 t15_leaf6 t15_leaf7 t16_leaf0
 t16_leaf1 t16_leaf2 t16_leaf3 t16_leaf4 t16_leaf5 t16_leaf6 t16_leaf7
 t17_leaf0 t17_leaf1 t17_leaf2 t17_leaf3 t17_leaf4 t17_leaf5 t17_leaf6
 t18_leaf0 t18_leaf1 t18_leaf2 t18_leaf3 t18_leaf4 t18_leaf5 t18_leaf6
 t18_leaf7 t19_leaf0 t19_leaf1 t19_leaf2 t19_leaf3 t19_leaf4 t19_leaf5
 t19_leaf6 t19_leaf7 t20_leaf0 t20_leaf1 t20_leaf2 t20_leaf3 t20_leaf4
 t20_leaf5 t20_leaf6 t20_leaf7 t21_leaf0 t21_leaf1 t21_leaf2 t21_leaf3
 t21_leaf4 t21_leaf5 t21_leaf6 t21_leaf7 t22_leaf0 t22_leaf1 t22_leaf2
 t22_leaf3 t22_leaf4 t22_leaf5 t22_leaf6 t22_leaf7 t23_leaf0 t23_leaf1
 t23_leaf2 t23_leaf3 t23_leaf4 t23_leaf5 t23_leaf6 t23_leaf7 t24_leaf0
 t24_leaf1 t24_leaf2 t24_leaf3 t24_leaf4 t24_leaf5 t24_leaf6 t24_leaf7
 t25_leaf0 t25_leaf1 t25_leaf2 t25_leaf3 t25_leaf4 t25_leaf5 t25_leaf6
 t25_leaf7 t26_leaf0 t26_leaf1 t26_leaf2 t26_leaf3 t26_leaf4 t26_leaf5
 t26_leaf6 t26_leaf7 t27_leaf0 t27_leaf1 t27_leaf2 t27_leaf3 t27_leaf4
 t27_leaf5 t27_leaf6 t27_leaf7 t28_leaf0 t28_leaf1 t28_leaf2 t28_leaf3
 t28_leaf4 t28_leaf5 t28_leaf6 t28_leaf7 t29_leaf0 t29_leaf1 t29_leaf2
 t29_leaf3 t29_leaf4 t29_leaf5 t29_leaf6 t29_leaf7 t30_leaf0 t30_leaf1
 t30_leaf2 t30_leaf3 t30_leaf4 t30_leaf5 t30_leaf6 t30_leaf7 t31_leaf0
 t31_leaf1 t31_leaf2 t31_leaf3 t31_leaf4 t31_leaf5 t31_leaf6 t31_leaf7
 t32_leaf0 t32_leaf1 t32_leaf2 t32_leaf3 t32_leaf4 t32_leaf5 t32_leaf6
 t32_leaf7 t33_leaf0 t33_leaf1 t33_leaf2 t33_leaf3 t33_leaf4 t33_leaf5
 t33_leaf6 t33_leaf7 t34_leaf0 t34_leaf1 t34_leaf2 t34_leaf3 t34_leaf4
 t34_leaf5 t34_leaf6 t34_leaf7 t35_leaf0 t35_leaf1 t35_leaf2 t35_leaf3
 t35_leaf4 t35_leaf5 t35_leaf6 t35_leaf7 t36_leaf0 t36_leaf1 t36_leaf2
 t36_leaf3 t36_leaf4 t36_leaf5 t36_leaf6 t36_leaf7 t37_leaf0 t37_leaf1
 t37_leaf2 t37_leaf3 t37_leaf4 t37_leaf5 t37_leaf6 t37_leaf7 t38_leaf0
 t38_leaf1 t38_leaf2 t38_leaf3 t38_leaf4 t38_leaf5 t38_leaf6 t38_leaf7
 t39_leaf0 t39_leaf1 t39_leaf2 t39_leaf3 t39_leaf4 t39_leaf5 t39_leaf6
 t39_leaf7 t40_leaf0 t40_leaf1 t40_leaf2 t40_leaf3 t40_leaf4 t40_leaf5
 t40_leaf6 t40_leaf7 t41_leaf0 t41_leaf1 t41_leaf2 t41_leaf3 t41_leaf4
 t41_leaf5 t41_leaf6 t41_leaf7 t42_leaf0 t42_leaf1 t42_leaf2 t42_leaf3
 t42_leaf4 t42_leaf5 t42_leaf6 t42_leaf7 t43_leaf0 t43_leaf1 t43_leaf2
 t43_leaf3 t43_leaf4 t43_leaf5 t43_leaf6 t43_leaf7 t44_leaf0 t44_leaf1
 t44_leaf2 t44_leaf3 t44_leaf4 t44_leaf5 t44_leaf6 t44_leaf7 t45_leaf0
 t45_leaf1 t45_leaf2 t45_leaf3 t45_leaf4 t45_leaf5 t45_leaf6 t45_leaf7
 t46_leaf0 t46_leaf1 t46_leaf2 t46_leaf3 t46_leaf4 t46_leaf5 t46_leaf6
 t46_leaf7 t47_leaf0 t47_leaf1 t47_leaf2 t47_leaf3 t47_leaf4 t47_leaf5
 t47_leaf6 t47_leaf7 t48_leaf0 t48_leaf1 t48_leaf2 t48_leaf3 t48_leaf4
 t48_leaf5 t48_leaf6 t48_leaf7 t49_leaf0 t49_leaf1 t49_leaf2 t49_leaf3
 t49_leaf4 t49_leaf5 t49_leaf6 t49_leaf7 t50_leaf0 t50_leaf1 t50_leaf2
 t50_leaf3 t50_leaf4 t50_leaf5 t50_leaf6 t51_leaf0 t51_leaf1 t51_leaf2
 t51_leaf3 t51_leaf4 t51_leaf5 t51_leaf6 t52_leaf0 t52_leaf1 t52_leaf2
 t52_leaf3 t52_leaf4 t52_leaf5 t52_leaf6 t52_leaf7 t53_leaf0 t53_leaf1
 t53_leaf2 t53_leaf3 t53_leaf4 t53_leaf5 t53_leaf6 t54_leaf0 t54_leaf1
 t54_leaf2 t54_leaf3 t54_leaf4 t54_leaf5 t54_leaf6 t55_leaf0 t55_leaf1
 t55_leaf2 t55_leaf3 t55_leaf4 t55_leaf5 t55_leaf6 t55_leaf7 t56_leaf0
 t56_leaf1 t56_leaf2 t56_leaf3 t56_leaf4 t56_leaf5 t56_leaf6 t56_leaf7
 t57_leaf0 t57_leaf1 t57_leaf2 t57_leaf3 t57_leaf4 t57_leaf5 t57_leaf6
 t57_leaf7 t58_leaf0 t58_leaf1 t58_leaf2 t58_leaf3 t58_leaf4 t58_leaf5
 t58_leaf6 t58_leaf7 t59_leaf0 t59_leaf1 t59_leaf2 t59_leaf3 t59_leaf4
 t59_leaf5 t59_leaf6 t59_leaf7 t60_leaf0 t60_leaf1 t60_leaf2 t60_leaf3
 t60_leaf4 t60_leaf5 t60_leaf6 t61_leaf0 t61_leaf1 t61_leaf2 t61_leaf3
 t61_leaf4 t61_leaf5 t61_leaf6 t62_leaf0 t62_leaf1 t62_leaf2 t62_leaf3
 t62_leaf4 t62_leaf5 t62_leaf6 t63_leaf0 t63_leaf1 t63_leaf2 t63_leaf3
 t63_leaf4 t63_leaf5 t63_leaf6 t63_leaf7 t64_leaf0 t64_leaf1 t64_leaf2
 t64_leaf3 t64_leaf4 t64_leaf5 t64_leaf6 t64_leaf7 t65_leaf0 t65_leaf1
 t65_leaf2 t65_leaf3 t65_leaf4 t65_leaf5 t65_leaf6 t65_leaf7 t66_leaf0
 t66_leaf1 t66_leaf2 t66_leaf3 t66_leaf4 t66_leaf5 t66_leaf6 t66_leaf7
 t67_leaf0 t67_leaf1 t67_leaf2 t67_leaf3 t67_leaf4 t67_leaf5 t68_leaf0
 t68_leaf1 t68_leaf2 t68_leaf3 t68_leaf4 t68_leaf5 t68_leaf6 t68_leaf7
 t69_leaf0 t69_leaf1 t69_leaf2 t69_leaf3 t69_leaf4 t69_leaf5 t69_leaf6
 t70_leaf0 t70_leaf1 t70_leaf2 t70_leaf3 t70_leaf4 t70_leaf5 t70_leaf6
 t70_leaf7 t71_leaf0 t71_leaf1 t71_leaf2 t71_leaf3 t71_leaf4 t71_leaf5
 t71_leaf6 t71_leaf7 t72_leaf0 t72_leaf1 t72_leaf2 t72_leaf3 t72_leaf4
 t72_leaf5 t72_leaf6 t72_leaf7 t73_leaf0 t73_leaf1 t73_leaf2 t73_leaf3
 t73_leaf4 t73_leaf5 t73_leaf6 t74_leaf0 t74_leaf1 t74_leaf2 t74_leaf3
 t74_leaf4 t74_leaf5 t74_leaf6 t75_leaf0 t75_leaf1 t75_leaf2 t75_leaf3
 t75_leaf4 t75_leaf5 t75_leaf6 t76_leaf0 t76_leaf1 t76_leaf2 t76_leaf3
 t76_leaf4 t76_leaf5 t76_leaf6 t76_leaf7 t77_leaf0 t77_leaf1 t77_leaf2
 t77_leaf3 t77_leaf4 t77_leaf5 t77_leaf6 t78_leaf0 t78_leaf1 t78_leaf2
 t78_leaf3 t78_leaf4 t78_leaf5 t78_leaf6 t78_leaf7 t79_leaf0 t79_leaf1
 t79_leaf2 t79_leaf3 t79_leaf4 t79_leaf5 t79_leaf6 t80_leaf0 t80_leaf1
 t80_leaf2 t80_leaf3 t80_leaf4 t80_leaf5 t80_leaf6 t81_leaf0 t81_leaf1
 t81_leaf2 t81_leaf3 t81_leaf4 t82_leaf0 t82_leaf1 t82_leaf2 t82_leaf3
 t82_leaf4 t82_leaf5 t82_leaf6 t83_leaf0 t83_leaf1 t83_leaf2 t83_leaf3
 t83_leaf4 t84_leaf0 t84_leaf1 t84_leaf2 t84_leaf3 t84_leaf4 t84_leaf5
 t84_leaf6 t85_leaf0 t85_leaf1 t85_leaf2 t85_leaf3 t85_leaf4 t85_leaf5
 t85_leaf6 t86_leaf0 t86_leaf1 t86_leaf2 t86_leaf3 t86_leaf4 t86_leaf5
 t86_leaf6 t87_leaf0 t87_leaf1 t87_leaf2 t87_leaf3 t87_leaf4 t87_leaf5
 t87_leaf6 t88_leaf0 t88_leaf1 t88_leaf2 t88_leaf3 t88_leaf4 t88_leaf5
 t88_leaf6 t89_leaf0 t89_leaf1 t89_leaf2 t89_leaf3 t89_leaf4 t89_leaf5
 t89_leaf6 t89_leaf7 t90_leaf0 t90_leaf1 t90_leaf2 t90_leaf3 t90_leaf4
 t90_leaf5 t90_leaf6 t90_leaf7 t91_leaf0 t91_leaf1 t91_leaf2 t91_leaf3
 t91_leaf4 t91_leaf5 t91_leaf6 t91_leaf7 t92_leaf0 t92_leaf1 t92_leaf2
 t92_leaf3 t92_leaf4 t92_leaf5 t92_leaf6 t93_leaf0 t93_leaf1 t93_leaf2
 t93_leaf3 t93_leaf4 t93_leaf5 t93_leaf6 t93_leaf7 t94_leaf0 t94_leaf1
 t94_leaf2 t94_leaf3 t94_leaf4 t94_leaf5 t94_leaf6 t95_leaf0 t95_leaf1
 t95_leaf2 t95_leaf3 t95_leaf4 t95_leaf5 t95_leaf6 t95_leaf7 t96_leaf0
 t96_leaf1 t96_leaf2 t96_leaf3 t96_leaf4 t96_leaf5 t96_leaf6 t97_leaf0
 t97_leaf1 t97_leaf2 t97_leaf3 t97_leaf4 t97_leaf5 t97_leaf6 t97_leaf7
 t98_leaf0 t98_leaf1 t98_leaf2 t98_leaf3 t98_leaf4 t98_leaf5 t98_leaf6
 t99_leaf0 t99_leaf1 t99_leaf2 t99_leaf3 t99_leaf4 t99_leaf5 t99_leaf6
 t100_leaf0 t100_leaf1 t100_leaf2 t100_leaf3 t100_leaf4 t100_leaf5
 t100_leaf6 t100_leaf7 t101_leaf0 t101_leaf1 t101_leaf2 t101_leaf3
 t101_leaf4 t101_leaf5 t101_leaf6 t102_leaf0 t102_leaf1 t102_leaf2
 t102_leaf3 t102_leaf4 t102_leaf5 t102_leaf6 t102_leaf7 t103_leaf0
 t103_leaf1 t103_leaf2 t103_leaf3 t103_leaf4 t103_leaf5 t104_leaf0
 t104_leaf1 t104_leaf2 t104_leaf3 t104_leaf4 t104_leaf5 t105_leaf0
 t105_leaf1 t105_leaf2 t105_leaf3 t105_leaf4 t105_leaf5 t105_leaf6
 t105_leaf7 t106_leaf0 t106_leaf1 t106_leaf2 t106_leaf3 t106_leaf4
 t106_leaf5 t106_leaf6 t107_leaf0 t107_leaf1 t107_leaf2 t107_leaf3
 t107_leaf4 t107_leaf5 t107_leaf6 t107_leaf7 t108_leaf0 t108_leaf1
 t108_leaf2 t108_leaf3 t108_leaf4 t108_leaf5 t108_leaf6 t108_leaf7
 t109_leaf0 t109_leaf1 t109_leaf2 t109_leaf3 t109_leaf4 t109_leaf5
 t109_leaf6 t110_leaf0 t110_leaf1 t110_leaf2 t110_leaf3 t110_leaf4
 t110_leaf5 t111_leaf0 t111_leaf1 t111_leaf2 t111_leaf3 t111_leaf4
 t111_leaf5 t111_leaf6 t111_leaf7 t112_leaf0 t112_leaf1 t112_leaf2
 t112_leaf3 t112_leaf4 t112_leaf5 t112_leaf6 t112_leaf7 t113_leaf0
 t113_leaf1 t113_leaf2 t113_leaf3 t113_leaf4 t113_leaf5 t113_leaf6
 t114_leaf0 t114_leaf1 t114_leaf2 t114_leaf3 t114_leaf4 t114_leaf5
 t114_leaf6 t114_leaf7 t115_leaf0 t115_leaf1 t115_leaf2 t115_leaf3
 t115_leaf4 t115_leaf5 t116_leaf0 t116_leaf1 t116_leaf2 t116_leaf3
 t116_leaf4 t116_leaf5 t116_leaf6 t117_leaf0 t117_leaf1 t117_leaf2
 t117_leaf3 t117_leaf4 t118_leaf0 t118_leaf1 t118_leaf2 t118_leaf3
 t118_leaf4 t118_leaf5 t119_leaf0 t119_leaf1 t119_leaf2 t119_leaf3
 t120_leaf0 t120_leaf1 t120_leaf2 t120_leaf3 t120_leaf4 t120_leaf5
 t121_leaf0 t121_leaf1 t121_leaf2 t121_leaf3 t121_leaf4 t121_leaf5
 t121_leaf6 t122_leaf0 t122_leaf1 t122_leaf2 t122_leaf3 t122_leaf4
 t123_leaf0 t123_leaf1 t123_leaf2 t123_leaf3 t123_leaf4 t123_leaf5
 t123_leaf6 t123_leaf7 t124_leaf0 t124_leaf1 t124_leaf2 t124_leaf3
 t124_leaf4 t124_leaf5 t125_leaf0 t125_leaf1 t125_leaf2 t125_leaf3
 t125_leaf4 t125_leaf5 t126_leaf0 t126_leaf1 t126_leaf2 t126_leaf3
 t126_leaf4 t127_leaf0 t127_leaf1 t127_leaf2 t127_leaf3 t127_leaf4
 t127_leaf5 t128_leaf0 t128_leaf1 t128_leaf2 t128_leaf3 t128_leaf4
 t129_leaf0 t129_leaf1 t129_leaf2 t129_leaf3 t129_leaf4 t129_leaf5
 t130_leaf0 t130_leaf1 t130_leaf2 t130_leaf3 t130_leaf4 t130_leaf5
 t130_leaf6 t130_leaf7 t131_leaf0 t131_leaf1 t131_leaf2 t131_leaf3
 t131_leaf4 t131_leaf5 t131_leaf6 t131_leaf7 t132_leaf0 t132_leaf1
 t132_leaf2 t132_leaf3 t132_leaf4 t133_leaf0 t133_leaf1 t133_leaf2
 t133_leaf3 t133_leaf4 t134_leaf0 t134_leaf1 t134_leaf2 t134_leaf3
 t134_leaf4 t135_leaf0 t135_leaf1 t135_leaf2 t135_leaf3 t135_leaf4
 t135_leaf5 t135_leaf6 t136_leaf0 t136_leaf1 t136_leaf2 t136_leaf3
 t136_leaf4 t137_leaf0 t137_leaf1 t137_leaf2 t137_leaf3 t137_leaf4
 t137_leaf5 t137_leaf6 t137_leaf7 t138_leaf0 t138_leaf1 t138_leaf2
 t138_leaf3 t138_leaf4 t138_leaf5 t138_leaf6 t139_leaf0 t139_leaf1
 t139_leaf2 t139_leaf3 t139_leaf4 t139_leaf5 t139_leaf6 t139_leaf7
 t140_leaf0 t140_leaf1 t140_leaf2 t140_leaf3 t140_leaf4 t140_leaf5
 t140_leaf6 t140_leaf7 t141_leaf0 t141_leaf1 t141_leaf2 t141_leaf3
 t141_leaf4 t141_leaf5 t142_leaf0 t142_leaf1 t142_leaf2 t142_leaf3
 t142_leaf4 t143_leaf0 t143_leaf1 t143_leaf2 t143_leaf3 t143_leaf4
 t143_leaf5 t143_leaf6 t144_leaf0 t144_leaf1 t144_leaf2 t144_leaf3
 t144_leaf4 t144_leaf5 t145_leaf0 t145_leaf1 t145_leaf2 t145_leaf3
 t145_leaf4 t145_leaf5 t146_leaf0 t146_leaf1 t146_leaf2 t146_leaf3
 t146_leaf4 t146_leaf5 t146_leaf6 t146_leaf7 t147_leaf0 t147_leaf1
 t147_leaf2 t147_leaf3 t147_leaf4 t147_leaf5 t148_leaf0 t148_leaf1
 t148_leaf2 t148_leaf3 t148_leaf4 t148_leaf5 t149_leaf0 t149_leaf1
 t149_leaf2 t149_leaf3 t149_leaf4 t149_leaf5 t149_leaf6 t150_leaf0
 t150_leaf1 t150_leaf2 t150_leaf3 t150_leaf4 t150_leaf5 t150_leaf6
 t151_leaf0 t151_leaf1 t151_leaf2 t151_leaf3 t151_leaf4 t152_leaf0
 t152_leaf1 t152_leaf2 t152_leaf3 t152_leaf4 t152_leaf5 t153_leaf0
 t153_leaf1 t153_leaf2 t153_leaf3 t153_leaf4 t153_leaf5 t153_leaf6
 t154_leaf0 t154_leaf1 t154_leaf2 t154_leaf3 t154_leaf4 t154_leaf5
 t154_leaf6 t154_leaf7 t155_leaf0 t155_leaf1 t155_leaf2 t155_leaf3
 t155_leaf4 t156_leaf0 t156_leaf1 t156_leaf2 t156_leaf3 t156_leaf4
 t157_leaf0 t157_leaf1 t157_leaf2 t157_leaf3 t157_leaf4 t157_leaf5
 t157_leaf6 t158_leaf0 t158_leaf1 t158_leaf2 t158_leaf3 t158_leaf4
 t158_leaf5 t158_leaf6 t159_leaf0 t159_leaf1 t159_leaf2 t159_leaf3
 t159_leaf4 t160_leaf0 t160_leaf1 t160_leaf2 t160_leaf3 t160_leaf4
 t160_leaf5 t160_leaf6 t161_leaf0 t161_leaf1 t161_leaf2 t161_leaf3
 t161_leaf4 t161_leaf5 t161_leaf6 t162_leaf0 t162_leaf1 t162_leaf2
 t162_leaf3 t162_leaf4 t162_leaf5 t162_leaf6 t163_leaf0 t163_leaf1
 t163_leaf2 t163_leaf3 t163_leaf4 t163_leaf5 t164_leaf0 t164_leaf1
 t164_leaf2 t164_leaf3 t164_leaf4 t165_leaf0 t165_leaf1 t165_leaf2
 t165_leaf3 t165_leaf4 t166_leaf0 t166_leaf1 t166_leaf2 t166_leaf3
 t167_leaf0 t167_leaf1 t167_leaf2 t167_leaf3 t167_leaf4 t168_leaf0
 t168_leaf1 t168_leaf2 t168_leaf3 t168_leaf4 t169_leaf0 t169_leaf1
 t169_leaf2 t169_leaf3 t169_leaf4 t169_leaf5 t169_leaf6 t170_leaf0
 t170_leaf1 t170_leaf2 t170_leaf3 t170_leaf4 t171_leaf0 t171_leaf1
 t171_leaf2 t171_leaf3 t171_leaf4 t172_leaf0 t172_leaf1 t172_leaf2
 t172_leaf3 t172_leaf4 t172_leaf5 t172_leaf6 t173_leaf0 t173_leaf1
 t173_leaf2 t173_leaf3 t173_leaf4 t173_leaf5 t173_leaf6 t174_leaf0
 t174_leaf1 t174_leaf2 t174_leaf3 t175_leaf0 t175_leaf1 t175_leaf2
 t175_leaf3 t175_leaf4 t176_leaf0 t176_leaf1 t176_leaf2 t176_leaf3
 t176_leaf4 t176_leaf5 t176_leaf6 t177_leaf0 t177_leaf1 t177_leaf2
 t177_leaf3 t177_leaf4 t177_leaf5 t177_leaf6 t178_leaf0 t178_leaf1
 t178_leaf2 t178_leaf3 t178_leaf4 t179_leaf0 t179_leaf1 t179_leaf2
 t179_leaf3 t179_leaf4 t179_leaf5 t180_leaf0 t180_leaf1 t180_leaf2
 t180_leaf3 t180_leaf4 t181_leaf0 t181_leaf1 t181_leaf2 t181_leaf3
 t181_leaf4 t181_leaf5 t181_leaf6 t182_leaf0 t182_leaf1 t182_leaf2
 t182_leaf3 t182_leaf4 t182_leaf5 t182_leaf6 t182_leaf7 t183_leaf0
 t183_leaf1 t183_leaf2 t183_leaf3 t183_leaf4 t183_leaf5 t184_leaf0
 t184_leaf1 t184_leaf2 t184_leaf3 t184_leaf4 t184_leaf5 t185_leaf0
 t185_leaf1 t185_leaf2 t185_leaf3 t185_leaf4 t185_leaf5 t186_leaf0
 t186_leaf1 t186_leaf2 t186_leaf3 t186_leaf4 t186_leaf5 t186_leaf6
 t187_leaf0 t187_leaf1 t187_leaf2 t187_leaf3 t187_leaf4 t187_leaf5
 t187_leaf6 t188_leaf0 t188_leaf1 t188_leaf2 t188_leaf3 t188_leaf4
 t188_leaf5 t188_leaf6 t188_leaf7 t189_leaf0 t189_leaf1 t189_leaf2
 t189_leaf3 t189_leaf4 t190_leaf0 t190_leaf1 t190_leaf2 t190_leaf3
 t190_leaf4 t190_leaf5 t191_leaf0 t191_leaf1 t191_leaf2 t191_leaf3
 t191_leaf4 t191_leaf5 t191_leaf6 t192_leaf0 t192_leaf1 t192_leaf2
 t192_leaf3 t192_leaf4 t193_leaf0 t193_leaf1 t193_leaf2 t193_leaf3
 t193_leaf4 t193_leaf5 t193_leaf6 t193_leaf7 t194_leaf0 t194_leaf1
 t194_leaf2 t194_leaf3 t195_leaf0 t195_leaf1 t195_leaf2 t195_leaf3
 t196_leaf0 t196_leaf1 t196_leaf2 t196_leaf3 t196_leaf4 t196_leaf5
 t197_leaf0 t197_leaf1 t197_leaf2 t197_leaf3 t197_leaf4 t197_leaf5
 t198_leaf0 t198_leaf1 t198_leaf2 t198_leaf3 t198_leaf4 t198_leaf5
 t199_leaf0 t199_leaf1 t199_leaf2 t199_leaf3 t199_leaf4 t200_leaf0
 t200_leaf1 t200_leaf2 t200_leaf3 t200_leaf4 t201_leaf0 t201_leaf1
 t201_leaf2 t201_leaf3 t201_leaf4 t202_leaf0 t202_leaf1 t202_leaf2
 t202_leaf3 t202_leaf4 t202_leaf5 t202_leaf6 t203_leaf0 t203_leaf1
 t203_leaf2 t203_leaf3 t203_leaf4 t203_leaf5 t203_leaf6 t203_leaf7
 t204_leaf0 t204_leaf1 t204_leaf2 t204_leaf3 t204_leaf4 t205_leaf0
 t205_leaf1 t205_leaf2 t205_leaf3 t206_leaf0 t206_leaf1 t206_leaf2
 t206_leaf3 t207_leaf0 t207_leaf1 t207_leaf2 t207_leaf3 t207_leaf4
 t207_leaf5 t208_leaf0 t208_leaf1 t208_leaf2 t208_leaf3 t208_leaf4
 t209_leaf0 t209_leaf1 t209_leaf2 t209_leaf3 t209_leaf4 t210_leaf0
 t210_leaf1 t210_leaf2 t210_leaf3 t210_leaf4 t211_leaf0 t211_leaf1
 t211_leaf2 t211_leaf3 t211_leaf4 t211_leaf5 t211_leaf6 t212_leaf0
 t212_leaf1 t212_leaf2 t212_leaf3 t212_leaf4 t212_leaf5 t213_leaf0
 t213_leaf1 t213_leaf2 t213_leaf3 t214_leaf0 t214_leaf1 t214_leaf2
 t214_leaf3 t214_leaf4 t214_leaf5 t214_leaf6 t215_leaf0 t215_leaf1
 t215_leaf2 t215_leaf3 t215_leaf4 t215_leaf5 t216_leaf0 t216_leaf1
 t216_leaf2 t216_leaf3 t216_leaf4 t217_leaf0 t217_leaf1 t217_leaf2
 t217_leaf3 t217_leaf4 t218_leaf0 t218_leaf1 t218_leaf2 t218_leaf3
 t219_leaf0 t219_leaf1 t219_leaf2 t219_leaf3 t219_leaf4 t219_leaf5
 t219_leaf6 t220_leaf0 t220_leaf1 t220_leaf2 t220_leaf3 t220_leaf4
 t220_leaf5 t220_leaf6 t221_leaf0 t221_leaf1 t221_leaf2 t221_leaf3
 t221_leaf4 t221_leaf5 t222_leaf0 t222_leaf1 t222_leaf2 t222_leaf3
 t222_leaf4 t223_leaf0 t223_leaf1 t223_leaf2 t223_leaf3 t223_leaf4
 t223_leaf5 t223_leaf6 t224_leaf0 t224_leaf1 t224_leaf2 t224_leaf3
 t225_leaf0 t225_leaf1 t225_leaf2 t225_leaf3 t225_leaf4 t226_leaf0
 t226_leaf1 t226_leaf2 t226_leaf3 t226_leaf4 t226_leaf5 t226_leaf6
 t226_leaf7 t227_leaf0 t227_leaf1 t227_leaf2 t227_leaf3 t227_leaf4
 t228_leaf0 t228_leaf1 t228_leaf2 t228_leaf3 t228_leaf4 t229_leaf0
 t229_leaf1 t229_leaf2 t229_leaf3 t229_leaf4 t230_leaf0 t230_leaf1
 t230_leaf2 t230_leaf3 t230_leaf4 t230_leaf5 t230_leaf6 t231_leaf0
 t231_leaf1 t231_leaf2 t231_leaf3 t231_leaf4 t231_leaf5 t231_leaf6
 t231_leaf7 t232_leaf0 t232_leaf1 t232_leaf2 t232_leaf3 t232_leaf4
 t233_leaf0 t233_leaf1 t233_leaf2 t233_leaf3 t233_leaf4 t233_leaf5
 t233_leaf6 t234_leaf0 t234_leaf1 t234_leaf2 t234_leaf3 t234_leaf4
 t234_leaf5 t235_leaf0 t235_leaf1 t235_leaf2 t235_leaf3 t235_leaf4
 t236_leaf0 t236_leaf1 t236_leaf2 t236_leaf3 t236_leaf4 t236_leaf5
 t237_leaf0 t237_leaf1 t237_leaf2 t237_leaf3 t238_leaf0 t238_leaf1
 t238_leaf2 t238_leaf3 t238_leaf4 t238_leaf5 t239_leaf0 t239_leaf1
 t239_leaf2 t239_leaf3 t239_leaf4 t239_leaf5 t239_leaf6 t240_leaf0
 t240_leaf1 t240_leaf2 t240_leaf3 t240_leaf4 t240_leaf5 t241_leaf0
 t241_leaf1 t241_leaf2 t241_leaf3 t241_leaf4 t242_leaf0 t242_leaf1
 t242_leaf2 t242_leaf3 t242_leaf4 t243_leaf0 t243_leaf1 t243_leaf2
 t243_leaf3 t243_leaf4 t244_leaf0 t244_leaf1 t244_leaf2 t244_leaf3
 t245_leaf0 t245_leaf1 t245_leaf2 t245_leaf3 t245_leaf4 t245_leaf5
 t245_leaf6 t246_leaf0 t246_leaf1 t246_leaf2 t246_leaf3 t247_leaf0
 t247_leaf1 t247_leaf2 t247_leaf3 t247_leaf4 t248_leaf0 t248_leaf1
 t248_leaf2 t248_leaf3 t248_leaf4 t248_leaf5 t248_leaf6 t248_leaf7
 t249_leaf0 t249_leaf1 t249_leaf2 t249_leaf3 t249_leaf4 t250_leaf0
 t250_leaf1 t250_leaf2 t250_leaf3 t251_leaf0 t251_leaf1 t251_leaf2
 t251_leaf3 t251_leaf4 t252_leaf0 t252_leaf1 t252_leaf2 t252_leaf3
 t252_leaf4 t252_leaf5 t252_leaf6 t252_leaf7 t253_leaf0 t253_leaf1
 t253_leaf2 t253_leaf3 t253_leaf4 t253_leaf5 t254_leaf0 t254_leaf1
 t254_leaf2 t254_leaf3 t255_leaf0 t255_leaf1 t255_leaf2 t255_leaf3
 t255_leaf4 t255_leaf5 t255_leaf6 t256_leaf0 t256_leaf1 t256_leaf2
 t256_leaf3 t256_leaf4 t256_leaf5 t256_leaf6 t257_leaf0 t257_leaf1
 t257_leaf2 t257_leaf3 t257_leaf4 t257_leaf5 t257_leaf6 t258_leaf0
 t258_leaf1 t258_leaf2 t258_leaf3 t258_leaf4 t259_leaf0 t259_leaf1
 t259_leaf2 t259_leaf3 t259_leaf4 t260_leaf0 t260_leaf1 t260_leaf2
 t260_leaf3 t260_leaf4 t260_leaf5 t261_leaf0 t261_leaf1 t261_leaf2
 t261_leaf3 t262_leaf0 t262_leaf1 t262_leaf2 t262_leaf3 t262_leaf4
 t263_leaf0 t263_leaf1 t263_leaf2 t263_leaf3 t263_leaf4 t263_leaf5
 t264_leaf0 t264_leaf1 t264_leaf2 t264_leaf3 t264_leaf4 t265_leaf0
 t265_leaf1 t265_leaf2 t265_leaf3 t265_leaf4 t266_leaf0 t266_leaf1
 t266_leaf2 t266_leaf3 t266_leaf4 t266_leaf5 t266_leaf6 t267_leaf0
 t267_leaf1 t267_leaf2 t267_leaf3 t267_leaf4 t267_leaf5 t268_leaf0
 t268_leaf1 t268_leaf2 t268_leaf3 t268_leaf4 t268_leaf5 t269_leaf0
 t269_leaf1 t269_leaf2 t269_leaf3 t269_leaf4 t270_leaf0 t270_leaf1
 t270_leaf2 t270_leaf3 t270_leaf4 t270_leaf5 t271_leaf0 t271_leaf1
 t271_leaf2 t271_leaf3 t271_leaf4 t271_leaf5 t272_leaf0 t272_leaf1
 t272_leaf2 t272_leaf3 t272_leaf4 t272_leaf5 t272_leaf6 t272_leaf7
 t273_leaf0 t273_leaf1 t273_leaf2 t273_leaf3 t273_leaf4 t274_leaf0
 t274_leaf1 t274_leaf2 t274_leaf3 t274_leaf4 t275_leaf0 t275_leaf1
 t275_leaf2 t275_leaf3 t275_leaf4 t275_leaf5 t276_leaf0 t276_leaf1
 t276_leaf2 t276_leaf3 t276_leaf4 t277_leaf0 t277_leaf1 t277_leaf2
 t277_leaf3 t277_leaf4 t277_leaf5 t277_leaf6 t278_leaf0 t278_leaf1
 t278_leaf2 t278_leaf3 t279_leaf0 t279_leaf1 t279_leaf2 t279_leaf3
 t279_leaf4 t279_leaf5 t280_leaf0 t280_leaf1 t280_leaf2 t280_leaf3
 t280_leaf4 t280_leaf5 t280_leaf6 t281_leaf0 t281_leaf1 t281_leaf2
 t281_leaf3 t281_leaf4 t282_leaf0 t282_leaf1 t282_leaf2 t282_leaf3
 t283_leaf0 t283_leaf1 t283_leaf2 t283_leaf3 t283_leaf4 t283_leaf5
 t284_leaf0 t284_leaf1 t284_leaf2 t284_leaf3 t284_leaf4 t285_leaf0
 t285_leaf1 t285_leaf2 t285_leaf3 t285_leaf4 t285_leaf5 t286_leaf0
 t286_leaf1 t286_leaf2 t286_leaf3 t286_leaf4 t286_leaf5 t287_leaf0
 t287_leaf1 t287_leaf2 t287_leaf3 t287_leaf4 t287_leaf5 t288_leaf0
 t288_leaf1 t288_leaf2 t288_leaf3 t288_leaf4 t288_leaf5 t289_leaf0
 t289_leaf1 t289_leaf2 t289_leaf3 t289_leaf4 t289_leaf5 t290_leaf0
 t290_leaf1 t290_leaf2 t290_leaf3 t290_leaf4 t290_leaf5 t291_leaf0
 t291_leaf1 t291_leaf2 t291_leaf3 t291_leaf4 t292_leaf0 t292_leaf1
 t292_leaf2 t292_leaf3 t292_leaf4 t293_leaf0 t293_leaf1 t293_leaf2
 t293_leaf3 t293_leaf4 t294_leaf0 t294_leaf1 t294_leaf2 t294_leaf3
 t294_leaf4 t295_leaf0 t295_leaf1 t295_leaf2 t295_leaf3 t295_leaf4
 t295_leaf5 t296_leaf0 t296_leaf1 t296_leaf2 t296_leaf3 t296_leaf4
 t296_leaf5 t297_leaf0 t297_leaf1 t297_leaf2 t297_leaf3 t298_leaf0
 t298_leaf1 t298_leaf2 t298_leaf3 t298_leaf4 t299_leaf0 t299_leaf1
 t299_leaf2 t299_leaf3 t299_leaf4 t299_leaf5 t299_leaf6 t300_leaf0
 t300_leaf1 t300_leaf2 t300_leaf3 t300_leaf4 t300_leaf5 t301_leaf0
 t301_leaf1 t301_leaf2 t301_leaf3 t301_leaf4 t302_leaf0 t302_leaf1
 t302_leaf2 t302_leaf3 t303_leaf0 t303_leaf1 t303_leaf2 t303_leaf3
 t303_leaf4 t304_leaf0 t304_leaf1 t304_leaf2 t304_leaf3 t304_leaf4
 t305_leaf0 t305_leaf1 t305_leaf2 t305_leaf3 t305_leaf4 t306_leaf0
 t306_leaf1 t306_leaf2 t306_leaf3 t306_leaf4 t306_leaf5 t307_leaf0
 t307_leaf1 t307_leaf2 t307_leaf3 t307_leaf4 t307_leaf5 t308_leaf0
 t308_leaf1 t308_leaf2 t308_leaf3 t308_leaf4 t309_leaf0 t309_leaf1
 t309_leaf2 t309_leaf3 t309_leaf4 t309_leaf5 t310_leaf0 t310_leaf1
 t310_leaf2 t310_leaf3 t310_leaf4 t311_leaf0 t311_leaf1 t311_leaf2
 t311_leaf3 t312_leaf0 t312_leaf1 t312_leaf2 t312_leaf3 t312_leaf4
 t313_leaf0 t313_leaf1 t313_leaf2 t313_leaf3 t313_leaf4 t313_leaf5
 t313_leaf6 t314_leaf0 t314_leaf1 t314_leaf2 t314_leaf3 t314_leaf4
 t314_leaf5 t314_leaf6 t315_leaf0 t315_leaf1 t315_leaf2 t315_leaf3
 t315_leaf4 t316_leaf0 t316_leaf1 t316_leaf2 t316_leaf3 t316_leaf4
 t317_leaf0 t317_leaf1 t317_leaf2 t317_leaf3 t318_leaf0 t318_leaf1
 t318_leaf2 t318_leaf3 t319_leaf0 t319_leaf1 t319_leaf2 t319_leaf3
 t319_leaf4 t320_leaf0 t320_leaf1 t320_leaf2 t320_leaf3 t320_leaf4
 t320_leaf5 t320_leaf6 t321_leaf0 t321_leaf1 t321_leaf2 t321_leaf3
 t321_leaf4 t322_leaf0 t322_leaf1 t322_leaf2 t322_leaf3 t322_leaf4
 t322_leaf5 t322_leaf6 t323_leaf0 t323_leaf1 t323_leaf2 t323_leaf3
 t324_leaf0 t324_leaf1 t324_leaf2 t324_leaf3 t324_leaf4 t324_leaf5
 t324_leaf6 t325_leaf0 t325_leaf1 t325_leaf2 t325_leaf3 t325_leaf4
 t326_leaf0 t326_leaf1 t326_leaf2 t326_leaf3 t327_leaf0 t327_leaf1
 t327_leaf2 t327_leaf3 t327_leaf4 t328_leaf0 t328_leaf1 t328_leaf2
 t328_leaf3 t328_leaf4 t328_leaf5 t329_leaf0 t329_leaf1 t329_leaf2
 t329_leaf3 t329_leaf4 t329_leaf5 t329_leaf6 t330_leaf0 t330_leaf1
 t330_leaf2 t330_leaf3 t330_leaf4 t330_leaf5 t331_leaf0 t331_leaf1
 t331_leaf2 t331_leaf3 t331_leaf4 t332_leaf0 t332_leaf1 t332_leaf2
 t332_leaf3 t333_leaf0 t333_leaf1 t333_leaf2 t334_leaf0 t334_leaf1
 t334_leaf2 t334_leaf3 t334_leaf4 t334_leaf5 t334_leaf6 t335_leaf0
 t335_leaf1 t335_leaf2 t335_leaf3 t335_leaf4 t335_leaf5 t336_leaf0
 t336_leaf1 t336_leaf2 t336_leaf3 t336_leaf4 t336_leaf5 t336_leaf6
 t336_leaf7 t337_leaf0 t337_leaf1 t337_leaf2 t337_leaf3 t337_leaf4
 t337_leaf5 t338_leaf0 t338_leaf1 t338_leaf2 t338_leaf3 t338_leaf4
 t338_leaf5 t338_leaf6 t339_leaf0 t339_leaf1 t339_leaf2 t339_leaf3
 t339_leaf4 t339_leaf5 t340_leaf0 t340_leaf1 t340_leaf2 t340_leaf3
 t340_leaf4 t340_leaf5 t340_leaf6 t341_leaf0 t341_leaf1 t341_leaf2
 t342_leaf0 t342_leaf1 t342_leaf2 t342_leaf3 t342_leaf4 t343_leaf0
 t343_leaf1 t343_leaf2 t343_leaf3 t343_leaf4 t344_leaf0 t344_leaf1
 t344_leaf2 t344_leaf3 t344_leaf4 t344_leaf5 t344_leaf6 t345_leaf0
 t345_leaf1 t345_leaf2 t345_leaf3 t345_leaf4 t345_leaf5 t345_leaf6
 t346_leaf0 t346_leaf1 t346_leaf2 t346_leaf3 t346_leaf4 t347_leaf0
 t347_leaf1 t347_leaf2 t347_leaf3 t348_leaf0 t348_leaf1 t348_leaf2
 t348_leaf3 t348_leaf4 t349_leaf0 t349_leaf1 t349_leaf2 t349_leaf3
 t349_leaf4 t350_leaf0 t350_leaf1 t350_leaf2 t350_leaf3 t350_leaf4
 t350_leaf5 t350_leaf6 t351_leaf0 t351_leaf1 t351_leaf2 t351_leaf3
 t351_leaf4 t351_leaf5 t352_leaf0 t352_leaf1 t353_leaf0 t353_leaf1
 t353_leaf2 t353_leaf3 t353_leaf4 t354_leaf0 t354_leaf1 t354_leaf2
 t355_leaf0 t355_leaf1 t355_leaf2 t355_leaf3 t355_leaf4 t355_leaf5
 t355_leaf6 t356_leaf0 t356_leaf1 t356_leaf2 t356_leaf3 t356_leaf4
 t356_leaf5 t356_leaf6 t357_leaf0 t357_leaf1 t357_leaf2 t357_leaf3
 t358_leaf0 t358_leaf1 t358_leaf2 t358_leaf3 t358_leaf4 t358_leaf5
 t358_leaf6 t359_leaf0 t359_leaf1 t359_leaf2 t359_leaf3 t359_leaf4
 t359_leaf5 t359_leaf6 t360_leaf0 t360_leaf1 t361_leaf0 t361_leaf1
 t361_leaf2 t361_leaf3 t362_leaf0 t362_leaf1 t362_leaf2 t362_leaf3
 t362_leaf4 t363_leaf0 t363_leaf1 t363_leaf2 t363_leaf3 t363_leaf4
 t364_leaf0 t364_leaf1 t364_leaf2 t364_leaf3 t364_leaf4 t365_leaf0
 t365_leaf1 t365_leaf2 t365_leaf3 t365_leaf4 t366_leaf0 t366_leaf1
 t366_leaf2 t366_leaf3 t366_leaf4 t367_leaf0 t367_leaf1 t367_leaf2
 t367_leaf3 t368_leaf0 t368_leaf1 t368_leaf2 t368_leaf3 t368_leaf4
 t369_leaf0 t369_leaf1 t369_leaf2 t369_leaf3 t369_leaf4 t369_leaf5
 t370_leaf0 t370_leaf1 t370_leaf2 t371_leaf0 t371_leaf1 t371_leaf2
 t371_leaf3 t371_leaf4 t372_leaf0 t372_leaf1 t372_leaf2 t373_leaf0
 t373_leaf1 t373_leaf2 t373_leaf3 t373_leaf4 t374_leaf0 t374_leaf1
 t374_leaf2 t374_leaf3 t374_leaf4 t374_leaf5 t375_leaf0 t375_leaf1
 t375_leaf2 t375_leaf3 t376_leaf0 t376_leaf1 t376_leaf2 t376_leaf3
 t377_leaf0 t378_leaf0 t378_leaf1 t378_leaf2 t379_leaf0 t379_leaf1
 t379_leaf2 t379_leaf3 t379_leaf4 t380_leaf0 t380_leaf1 t381_leaf0
 t381_leaf1 t381_leaf2 t381_leaf3 t381_leaf4 t382_leaf0 t382_leaf1
 t382_leaf2 t382_leaf3 t382_leaf4 t382_leaf5 t383_leaf0 t383_leaf1
 t383_leaf2 t383_leaf3 t383_leaf4 t383_leaf5 t383_leaf6 t384_leaf0
 t384_leaf1 t384_leaf2 t384_leaf3 t384_leaf4 t385_leaf0 t385_leaf1
 t385_leaf2 t385_leaf3 t385_leaf4 t386_leaf0 t386_leaf1 t386_leaf2
 t386_leaf3 t387_leaf0 t387_leaf1 t387_leaf2 t387_leaf3 t387_leaf4
 t388_leaf0 t388_leaf1 t388_leaf2 t388_leaf3 t389_leaf0 t389_leaf1
 t389_leaf2 t389_leaf3 t390_leaf0 t390_leaf1 t390_leaf2 t391_leaf0
 t391_leaf1 t391_leaf2 t391_leaf3 t392_leaf0 t392_leaf1 t392_leaf2
 t392_leaf3 t393_leaf0 t393_leaf1 t393_leaf2 t393_leaf3 t393_leaf4
 t394_leaf0 t394_leaf1 t395_leaf0 t395_leaf1 t395_leaf2 t395_leaf3
 t395_leaf4 t395_leaf5 t396_leaf0 t396_leaf1 t396_leaf2 t396_leaf3
 t396_leaf4 t397_leaf0 t397_leaf1 t397_leaf2 t397_leaf3 t397_leaf4
 t398_leaf0 t398_leaf1 t398_leaf2 t398_leaf3 t399_leaf0 t399_leaf1
 t399_leaf2 t399_leaf3 t400_leaf0 t400_leaf1 t400_leaf2 t401_leaf0
 t402_leaf0 t402_leaf1 t402_leaf2 t402_leaf3 t402_leaf4 t403_leaf0
 t403_leaf1 t403_leaf2 t403_leaf3 t403_leaf4 t404_leaf0 t404_leaf1
 t404_leaf2 t404_leaf3 t404_leaf4 t404_leaf5 t405_leaf0 t405_leaf1
 t405_leaf2 t405_leaf3 t405_leaf4 t406_leaf0 t407_leaf0 t408_leaf0
 t408_leaf1 t408_leaf2 t408_leaf3 t408_leaf4 t409_leaf0 t409_leaf1
 t409_leaf2 t409_leaf3 t409_leaf4 t409_leaf5 t410_leaf0 t410_leaf1
 t410_leaf2 t410_leaf3 t411_leaf0 t411_leaf1 t411_leaf2 t412_leaf0
 t412_leaf1 t412_leaf2 t413_leaf0 t413_leaf1 t413_leaf2 t413_leaf3
 t413_leaf4 t414_leaf0 t414_leaf1 t414_leaf2 t414_leaf3 t415_leaf0
 t415_leaf1 t415_leaf2 t415_leaf3 t415_leaf4 t416_leaf0 t416_leaf1
 t416_leaf2 t416_leaf3 t416_leaf4 t416_leaf5 t416_leaf6 t417_leaf0
 t418_leaf0 t418_leaf1 t419_leaf0 t419_leaf1 t419_leaf2 t419_leaf3
 t419_leaf4 t419_leaf5 t419_leaf6 t419_leaf7 t420_leaf0 t421_leaf0
 t421_leaf1 t421_leaf2 t421_leaf3 t422_leaf0 t422_leaf1 t422_leaf2
 t422_leaf3 t422_leaf4 t423_leaf0 t423_leaf1 t423_leaf2 t423_leaf3
 t424_leaf0 t424_leaf1 t424_leaf2 t424_leaf3 t424_leaf4 t424_leaf5
 t425_leaf0 t425_leaf1 t425_leaf2 t426_leaf0 t426_leaf1 t426_leaf2
 t426_leaf3 t426_leaf4 t426_leaf5 t426_leaf6 t427_leaf0 t427_leaf1
 t427_leaf2 t427_leaf3 t428_leaf0 t428_leaf1 t428_leaf2 t428_leaf3
 t428_leaf4 t429_leaf0 t429_leaf1 t429_leaf2 t429_leaf3 t429_leaf4
 t429_leaf5 t430_leaf0 t430_leaf1 t430_leaf2 t430_leaf3 t430_leaf4
 t430_leaf5 t430_leaf6 t431_leaf0 t431_leaf1 t431_leaf2 t431_leaf3
 t432_leaf0 t432_leaf1 t432_leaf2 t432_leaf3 t432_leaf4 t432_leaf5
 t433_leaf0 t433_leaf1 t433_leaf2 t434_leaf0 t435_leaf0 t435_leaf1
 t435_leaf2 t435_leaf3 t435_leaf4 t436_leaf0 t437_leaf0 t437_leaf1
 t437_leaf2 t438_leaf0 t438_leaf1 t439_leaf0 t439_leaf1 t439_leaf2
 t439_leaf3 t439_leaf4 t440_leaf0 t440_leaf1 t440_leaf2 t440_leaf3
 t440_leaf4 t440_leaf5 t441_leaf0 t441_leaf1 t441_leaf2 t441_leaf3
 t441_leaf4 t442_leaf0 t442_leaf1 t442_leaf2 t443_leaf0 t444_leaf0
 t444_leaf1 t444_leaf2 t444_leaf3 t444_leaf4 t445_leaf0 t445_leaf1
 t445_leaf2 t446_leaf0 t446_leaf1 t446_leaf2 t446_leaf3 t447_leaf0
 t447_leaf1 t447_leaf2 t448_leaf0 t448_leaf1 t448_leaf2 t449_leaf0
 t449_leaf1 t449_leaf2 t449_leaf3 t449_leaf4 t449_leaf5 t450_leaf0
 t451_leaf0 t451_leaf1 t451_leaf2 t451_leaf3 t451_leaf4 t451_leaf5
 t452_leaf0 t452_leaf1 t452_leaf2 t452_leaf3 t452_leaf4 t453_leaf0
 t453_leaf1 t453_leaf2 t453_leaf3 t453_leaf4 t453_leaf5 t454_leaf0
 t454_leaf1 t455_leaf0 t456_leaf0 t457_leaf0 t457_leaf1 t457_leaf2
 t458_leaf0 t458_leaf1 t458_leaf2 t458_leaf3 t459_leaf0 t459_leaf1
 t460_leaf0 t461_leaf0 t461_leaf1 t461_leaf2 t461_leaf3 t461_leaf4
 t462_leaf0 t462_leaf1 t463_leaf0 t463_leaf1 t463_leaf2 t463_leaf3
 t463_leaf4 t463_leaf5 t463_leaf6 t464_leaf0 t464_leaf1 t464_leaf2
 t465_leaf0 t465_leaf1 t465_leaf2 t465_leaf3 t466_leaf0 t466_leaf1
 t466_leaf2 t466_leaf3 t466_leaf4 t467_leaf0 t468_leaf0 t469_leaf0
 t470_leaf0 t470_leaf1 t470_leaf2 t471_leaf0 t471_leaf1 t471_leaf2
 t471_leaf3 t472_leaf0 t473_leaf0 t473_leaf1 t473_leaf2 t474_leaf0
 t474_leaf1 t474_leaf2 t475_leaf0 t475_leaf1 t475_leaf2 t476_leaf0
 t476_leaf1 t476_leaf2 t476_leaf3 t477_leaf0 t477_leaf1 t477_leaf2
 t477_leaf3 t478_leaf0 t478_leaf1 t478_leaf2 t478_leaf3 t479_leaf0
 t479_leaf1 t479_leaf2 t479_leaf3 t479_leaf4 t479_leaf5 t480_leaf0
 t480_leaf1 t480_leaf2 t480_leaf3 t480_leaf4 t481_leaf0 t481_leaf1
 t482_leaf0 t483_leaf0 t484_leaf0 t484_leaf1 t484_leaf2 t484_leaf3
 t485_leaf0 t486_leaf0 t487_leaf0 t488_leaf0 t488_leaf1 t488_leaf2
 t488_leaf3 t489_leaf0 t490_leaf0 t490_leaf1 t490_leaf2 t491_leaf0
 t491_leaf1 t491_leaf2 t491_leaf3 t492_leaf0 t492_leaf1 t492_leaf2
 t493_leaf0 t493_leaf1 t493_leaf2 t493_leaf3 t493_leaf4 t494_leaf0
 t494_leaf1 t494_leaf2 t495_leaf0 t495_leaf1 t496_leaf0 t496_leaf1
 t497_leaf0 t497_leaf1 t497_leaf2 t497_leaf3 t498_leaf0 t498_leaf1
 t498_leaf2 t498_leaf3 t498_leaf4 t499_leaf0 t499_leaf1 t500_leaf0
 t500_leaf1 t500_leaf2 t500_leaf3 t501_leaf0 t501_leaf1 t501_leaf2
 t501_leaf3 t501_leaf4 t501_leaf5 t501_leaf6 t502_leaf0 t502_leaf1
 t503_leaf0 t504_leaf0 t504_leaf1 t504_leaf2 t504_leaf3 t505_leaf0
 t505_leaf1 t506_leaf0 t506_leaf1 t506_leaf2 t506_leaf3 t506_leaf4
 t507_leaf0 t507_leaf1 t508_leaf0 t508_leaf1 t508_leaf2 t508_leaf3
 t508_leaf4 t508_leaf5 t509_leaf0 t509_leaf1 t509_leaf2 t510_leaf0
 t511_leaf0 t512_leaf0 t513_leaf0 t513_leaf1 t513_leaf2 t513_leaf3
 t513_leaf4 t513_leaf5 t514_leaf0 t515_leaf0 t515_leaf1 t515_leaf2
 t515_leaf3 t516_leaf0 t516_leaf1 t516_leaf2 t516_leaf3 t516_leaf4
 t517_leaf0 t517_leaf1 t517_leaf2 t517_leaf3 t517_leaf4 t517_leaf5
 t518_leaf0 t518_leaf1 t518_leaf2 t518_leaf3 t518_leaf4 t518_leaf5
 t519_leaf0 t519_leaf1 t520_leaf0 t520_leaf1 t520_leaf2 t520_leaf3
 t520_leaf4 t520_leaf5 t521_leaf0 t521_leaf1 t521_leaf2 t521_leaf3
 t521_leaf4 t522_leaf0 t522_leaf1 t522_leaf2 t522_leaf3 t522_leaf4
 t522_leaf5 t523_leaf0 t523_leaf1 t523_leaf2 t523_leaf3 t523_leaf4
 t524_leaf0 t524_leaf1 t524_leaf2 t524_leaf3 t525_leaf0 t525_leaf1
 t526_leaf0 t526_leaf1 t526_leaf2 t526_leaf3 t526_leaf4 t526_leaf5
 t527_leaf0 t527_leaf1 t527_leaf2 t527_leaf3 t527_leaf4 t527_leaf5
 t528_leaf0 t528_leaf1 t528_leaf2 t529_leaf0 t529_leaf1 t529_leaf2
 t529_leaf3 t529_leaf4 t530_leaf0 t531_leaf0 t531_leaf1 t531_leaf2
 t532_leaf0 t533_leaf0 t533_leaf1 t533_leaf2 t534_leaf0 t535_leaf0
 t535_leaf1 t535_leaf2 t535_leaf3 t535_leaf4 t536_leaf0 t537_leaf0
 t537_leaf1 t538_leaf0 t538_leaf1 t538_leaf2 t538_leaf3 t538_leaf4
 t539_leaf0 t539_leaf1 t540_leaf0 t540_leaf1 t540_leaf2 t540_leaf3
 t540_leaf4 t541_leaf0 t541_leaf1 t541_leaf2 t542_leaf0 t543_leaf0
 t543_leaf1 t543_leaf2 t543_leaf3 t543_leaf4 t543_leaf5 t543_leaf6
 t544_leaf0 t545_leaf0 t545_leaf1 t545_leaf2 t545_leaf3 t545_leaf4
 t546_leaf0 t546_leaf1 t547_leaf0 t547_leaf1 t548_leaf0 t548_leaf1
 t548_leaf2 t548_leaf3 t549_leaf0 t549_leaf1 t549_leaf2 t549_leaf3
 t549_leaf4 t549_leaf5 t550_leaf0 t550_leaf1 t550_leaf2 t550_leaf3
 t551_leaf0 t552_leaf0 t552_leaf1 t552_leaf2 t552_leaf3 t552_leaf4
 t553_leaf0 t553_leaf1 t553_leaf2 t553_leaf3 t554_leaf0 t555_leaf0
 t556_leaf0 t556_leaf1 t557_leaf0 t557_leaf1 t557_leaf2 t557_leaf3
 t558_leaf0 t558_leaf1 t558_leaf2 t558_leaf3 t559_leaf0 t559_leaf1
 t559_leaf2 t559_leaf3 t559_leaf4 t560_leaf0 t560_leaf1 t560_leaf2
 t560_leaf3 t561_leaf0 t561_leaf1 t561_leaf2 t562_leaf0 t562_leaf1
 t562_leaf2 t563_leaf0 t564_leaf0 t564_leaf1 t564_leaf2 t565_leaf0
 t566_leaf0 t566_leaf1 t566_leaf2 t566_leaf3 t566_leaf4 t566_leaf5
 t567_leaf0 t567_leaf1 t567_leaf2 t567_leaf3 t567_leaf4 t567_leaf5
 t568_leaf0 t568_leaf1 t568_leaf2 t568_leaf3 t569_leaf0 t570_leaf0
 t571_leaf0 t571_leaf1 t571_leaf2 t571_leaf3 t572_leaf0 t572_leaf1
 t572_leaf2 t573_leaf0 t574_leaf0 t574_leaf1 t574_leaf2 t574_leaf3
 t575_leaf0 t575_leaf1 t576_leaf0 t576_leaf1 t576_leaf2 t576_leaf3
 t576_leaf4 t576_leaf5 t577_leaf0 t578_leaf0 t578_leaf1 t578_leaf2
 t579_leaf0 t579_leaf1 t580_leaf0 t580_leaf1 t580_leaf2 t580_leaf3
 t580_leaf4 t580_leaf5 t581_leaf0 t581_leaf1 t582_leaf0 t582_leaf1
 t582_leaf2 t583_leaf0 t584_leaf0 t584_leaf1 t584_leaf2 t584_leaf3
 t584_leaf4 t585_leaf0 t585_leaf1 t585_leaf2 t585_leaf3 t585_leaf4
 t586_leaf0 t586_leaf1 t586_leaf2 t586_leaf3 t586_leaf4 t587_leaf0
 t587_leaf1 t587_leaf2 t587_leaf3 t587_leaf4 t587_leaf5 t588_leaf0
 t588_leaf1 t588_leaf2 t588_leaf3 t588_leaf4 t588_leaf5 t588_leaf6
 t588_leaf7 t589_leaf0 t590_leaf0 t590_leaf1 t591_leaf0 t592_leaf0
 t593_leaf0 t594_leaf0 t595_leaf0 t595_leaf1 t595_leaf2 t595_leaf3
 t595_leaf4 t596_leaf0 t597_leaf0 t597_leaf1 t597_leaf2 t597_leaf3
 t597_leaf4 t598_leaf0 t598_leaf1 t599_leaf0 t600_leaf0 t600_leaf1
 t600_leaf2 t600_leaf3 t601_leaf0 t601_leaf1 t601_leaf2 t601_leaf3
 t602_leaf0 t603_leaf0 t603_leaf1 t604_leaf0 t605_leaf0 t605_leaf1
 t605_leaf2 t605_leaf3 t606_leaf0 t607_leaf0 t607_leaf1 t608_leaf0
 t608_leaf1 t609_leaf0 t610_leaf0 t611_leaf0 t611_leaf1 t612_leaf0
 t613_leaf0 t613_leaf1 t614_leaf0 t615_leaf0 t616_leaf0 t616_leaf1
 t616_leaf2 t616_leaf3 t617_leaf0 t617_leaf1 t617_leaf2 t618_leaf0
 t618_leaf1 t619_leaf0 t619_leaf1 t619_leaf2 t619_leaf3 t619_leaf4
 t620_leaf0 t620_leaf1 t621_leaf0 t621_leaf1 t621_leaf2 t621_leaf3
 t622_leaf0 t622_leaf1 t622_leaf2 t622_leaf3 t623_leaf0 t623_leaf1
 t623_leaf2 t623_leaf3 t623_leaf4 t623_leaf5 t624_leaf0 t624_leaf1
 t625_leaf0 t625_leaf1 t625_leaf2 t626_leaf0 t626_leaf1 t626_leaf2
 t626_leaf3 t627_leaf0 t627_leaf1 t628_leaf0 t629_leaf0 t630_leaf0
 t631_leaf0 t632_leaf0 t632_leaf1 t632_leaf2 t632_leaf3 t633_leaf0
 t634_leaf0 t635_leaf0 t636_leaf0 t637_leaf0 t638_leaf0 t639_leaf0
 t640_leaf0 t640_leaf1 t640_leaf2 t640_leaf3 t641_leaf0 t642_leaf0
 t643_leaf0 t644_leaf0 t645_leaf0 t646_leaf0 t646_leaf1 t646_leaf2
 t647_leaf0 t647_leaf1 t647_leaf2 t648_leaf0 t649_leaf0 t649_leaf1
 t650_leaf0 t651_leaf0 t652_leaf0 t652_leaf1 t652_leaf2 t653_leaf0
 t653_leaf1 t654_leaf0 t655_leaf0 t655_leaf1 t655_leaf2 t656_leaf0
 t656_leaf1 t656_leaf2 t657_leaf0 t657_leaf1 t658_leaf0 t659_leaf0
 t659_leaf1 t659_leaf2 t659_leaf3 t659_leaf4 t660_leaf0 t660_leaf1
 t660_leaf2 t660_leaf3 t661_leaf0 t661_leaf1 t661_leaf2 t661_leaf3
 t661_leaf4 t662_leaf0 t663_leaf0 t663_leaf1 t663_leaf2 t663_leaf3
 t663_leaf4 t663_leaf5 t663_leaf6 t664_leaf0 t664_leaf1 t664_leaf2
 t665_leaf0 t665_leaf1 t665_leaf2 t666_leaf0 t666_leaf1 t666_leaf2
 t667_leaf0 t667_leaf1 t667_leaf2 t667_leaf3 t668_leaf0 t668_leaf1
 t668_leaf2 t668_leaf3 t669_leaf0 t670_leaf0 t671_leaf0 t671_leaf1
 t672_leaf0 t673_leaf0 t673_leaf1 t673_leaf2 t673_leaf3 t673_leaf4
 t674_leaf0 t674_leaf1 t675_leaf0 t676_leaf0 t676_leaf1 t677_leaf0
 t678_leaf0 t678_leaf1 t678_leaf2 t678_leaf3 t678_leaf4 t679_leaf0
 t680_leaf0 t680_leaf1 t680_leaf2 t680_leaf3 t680_leaf4 t681_leaf0
 t681_leaf1 t682_leaf0 t683_leaf0 t683_leaf1 t683_leaf2 t683_leaf3
 t684_leaf0 t684_leaf1 t684_leaf2 t685_leaf0 t686_leaf0 t687_leaf0
 t687_leaf1 t687_leaf2 t688_leaf0 t689_leaf0 t690_leaf0 t691_leaf0
 t692_leaf0 t692_leaf1 t693_leaf0 t693_leaf1 t694_leaf0 t695_leaf0
 t696_leaf0 t697_leaf0 t697_leaf1 t698_leaf0 t698_leaf1 t699_leaf0
 t700_leaf0 t701_leaf0 t702_leaf0 t702_leaf1 t702_leaf2 t702_leaf3
 t703_leaf0 t704_leaf0 t705_leaf0 t705_leaf1 t705_leaf2 t705_leaf3
 t705_leaf4 t706_leaf0 t706_leaf1 t706_leaf2 t707_leaf0 t708_leaf0
 t709_leaf0 t709_leaf1 t709_leaf2 t710_leaf0 t710_leaf1 t710_leaf2
 t710_leaf3 t711_leaf0 t712_leaf0 t712_leaf1 t712_leaf2 t712_leaf3
 t713_leaf0 t713_leaf1 t714_leaf0 t715_leaf0 t715_leaf1 t715_leaf2
 t715_leaf3 t716_leaf0 t717_leaf0 t718_leaf0 t719_leaf0 t720_leaf0
 t720_leaf1 t721_leaf0 t722_leaf0 t723_leaf0 t724_leaf0 t725_leaf0
 t726_leaf0 t726_leaf1 t726_leaf2 t727_leaf0 t727_leaf1 t727_leaf2
 t728_leaf0 t728_leaf1 t729_leaf0 t730_leaf0 t730_leaf1 t730_leaf2
 t730_leaf3 t730_leaf4 t731_leaf0 t732_leaf0 t733_leaf0 t734_leaf0
 t735_leaf0 t736_leaf0 t737_leaf0 t738_leaf0 t739_leaf0 t739_leaf1
 t740_leaf0 t741_leaf0 t741_leaf1 t741_leaf2 t742_leaf0 t742_leaf1
 t742_leaf2 t742_leaf3 t742_leaf4 t742_leaf5 t742_leaf6 t743_leaf0
 t744_leaf0 t745_leaf0 t746_leaf0 t747_leaf0 t748_leaf0 t749_leaf0
 t749_leaf1 t749_leaf2 t749_leaf3 t750_leaf0 t751_leaf0 t752_leaf0
 t753_leaf0 t754_leaf0 t755_leaf0 t756_leaf0 t757_leaf0 t757_leaf1
 t757_leaf2 t757_leaf3 t758_leaf0 t759_leaf0 t759_leaf1 t760_leaf0
 t761_leaf0 t761_leaf1 t761_leaf2 t761_leaf3 t762_leaf0 t763_leaf0
 t763_leaf1 t763_leaf2 t763_leaf3 t763_leaf4 t764_leaf0 t764_leaf1
 t764_leaf2 t764_leaf3 t765_leaf0 t766_leaf0 t766_leaf1 t767_leaf0
 t768_leaf0 t768_leaf1 t768_leaf2 t769_leaf0 t770_leaf0 t771_leaf0
 t772_leaf0 t772_leaf1 t772_leaf2 t773_leaf0 t774_leaf0 t775_leaf0
 t775_leaf1 t776_leaf0 t777_leaf0 t777_leaf1 t778_leaf0 t778_leaf1
 t778_leaf2 t778_leaf3 t779_leaf0 t780_leaf0 t781_leaf0 t781_leaf1
 t781_leaf2 t782_leaf0 t783_leaf0 t784_leaf0 t784_leaf1 t784_leaf2
 t784_leaf3 t784_leaf4 t785_leaf0 t785_leaf1 t786_leaf0 t787_leaf0
 t787_leaf1 t788_leaf0 t789_leaf0 t790_leaf0 t791_leaf0 t792_leaf0
 t793_leaf0 t793_leaf1 t793_leaf2 t793_leaf3 t794_leaf0 t795_leaf0
 t795_leaf1 t795_leaf2 t795_leaf3 t795_leaf4 t796_leaf0 t797_leaf0
 t798_leaf0 t799_leaf0 t800_leaf0 t801_leaf0 t802_leaf0 t803_leaf0
 t804_leaf0 t805_leaf0 t805_leaf1 t806_leaf0 t806_leaf1 t807_leaf0
 t808_leaf0 t809_leaf0 t810_leaf0 t811_leaf0 t812_leaf0 t813_leaf0
 t814_leaf0 t814_leaf1 t814_leaf2 t814_leaf3 t814_leaf4 t815_leaf0
 t816_leaf0 t817_leaf0 t817_leaf1 t817_leaf2 t818_leaf0 t819_leaf0
 t820_leaf0 t821_leaf0 t822_leaf0 t823_leaf0 t824_leaf0 t825_leaf0
 t825_leaf1 t826_leaf0 t827_leaf0 t827_leaf1 t827_leaf2 t827_leaf3
 t828_leaf0 t829_leaf0 t829_leaf1 t829_leaf2 t829_leaf3 t829_leaf4
 t830_leaf0 t831_leaf0 t831_leaf1 t832_leaf0 t832_leaf1 t833_leaf0
 t834_leaf0 t835_leaf0 t835_leaf1 t835_leaf2 t836_leaf0 t837_leaf0
 t838_leaf0 t839_leaf0 t839_leaf1 t840_leaf0 t841_leaf0 t841_leaf1
 t842_leaf0 t842_leaf1 t843_leaf0 t844_leaf0 t844_leaf1 t845_leaf0
 t845_leaf1 t845_leaf2 t846_leaf0 t847_leaf0 t848_leaf0 t848_leaf1
 t848_leaf2 t849_leaf0 t849_leaf1 t849_leaf2 t850_leaf0 t851_leaf0
 t852_leaf0 t853_leaf0 t854_leaf0 t854_leaf1 t855_leaf0 t856_leaf0
 t856_leaf1 t857_leaf0 t858_leaf0 t858_leaf1 t858_leaf2 t858_leaf3
 t859_leaf0 t860_leaf0 t860_leaf1 t860_leaf2 t861_leaf0 t862_leaf0
 t863_leaf0 t864_leaf0 t865_leaf0 t865_leaf1 t865_leaf2 t866_leaf0
 t867_leaf0 t868_leaf0 t869_leaf0 t870_leaf0 t871_leaf0 t872_leaf0
 t873_leaf0 t874_leaf0 t875_leaf0 t875_leaf1 t876_leaf0 t876_leaf1
 t876_leaf2 t876_leaf3 t877_leaf0 t877_leaf1 t878_leaf0 t879_leaf0
 t880_leaf0 t880_leaf1 t880_leaf2 t880_leaf3 t881_leaf0 t881_leaf1
 t881_leaf2 t881_leaf3 t881_leaf4 t882_leaf0 t883_leaf0 t883_leaf1
 t884_leaf0 t885_leaf0 t886_leaf0 t886_leaf1 t886_leaf2 t886_leaf3
 t886_leaf4 t886_leaf5 t887_leaf0 t888_leaf0 t888_leaf1 t889_leaf0
 t889_leaf1 t889_leaf2 t889_leaf3 t889_leaf4 t890_leaf0 t891_leaf0
 t892_leaf0 t893_leaf0 t894_leaf0 t894_leaf1 t895_leaf0 t896_leaf0
 t896_leaf1 t897_leaf0 t898_leaf0 t899_leaf0 t900_leaf0 t901_leaf0
 t902_leaf0 t903_leaf0 t903_leaf1 t903_leaf2 t904_leaf0 t904_leaf1
 t905_leaf0 t906_leaf0 t906_leaf1 t906_leaf2 t907_leaf0 t908_leaf0
 t909_leaf0 t909_leaf1 t910_leaf0 t911_leaf0 t912_leaf0 t913_leaf0 z_change
Generals
 x_Bedroom_AbvGr x_Full_Bath x_TotRms_AbvGrd x_Kitchen_Qual x_Exter_Qual
 x_Exter_Cond x_Heating_QC x_Fireplace_Qu x_Bsmt_Cond x_Garage_Qual
 x_Garage_Cond x_Pool_QC x_Utilities x_Roof_Matl x_Half_Bath
 x_Kitchen_AbvGr R8_other_rooms
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
