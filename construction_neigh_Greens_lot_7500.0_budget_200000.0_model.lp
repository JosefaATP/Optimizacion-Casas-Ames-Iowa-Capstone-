\ Model construction_embed
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 x_House_Style_1.5Unf + 0 x_House_Style_2.5Fin + 0 x_House_Style_2.5Unf
   + 0 x_House_Style_SFoyer + 0 x_House_Style_SLvl
   + 0 x_Mas_Vnr_Type_NoAplica + 0 CONST_ONE - cost_model + y_price
Subject To
 link__lot_area_fix: x_Lot_Area = 7000
 link__Neighborhood_Blmngtn__fix: x_Neighborhood_Blmngtn = 0
 link__Neighborhood_Blueste__fix: x_Neighborhood_Blueste = 0
 link__Neighborhood_BrDale__fix: x_Neighborhood_BrDale = 0
 link__Neighborhood_BrkSide__fix: x_Neighborhood_BrkSide = 0
 link__Neighborhood_ClearCr__fix: x_Neighborhood_ClearCr = 0
 link__Neighborhood_CollgCr__fix: x_Neighborhood_CollgCr = 0
 link__Neighborhood_Crawfor__fix: x_Neighborhood_Crawfor = 0
 link__Neighborhood_Edwards__fix: x_Neighborhood_Edwards = 0
 link__Neighborhood_Gilbert__fix: x_Neighborhood_Gilbert = 0
 link__Neighborhood_Greens__fix: x_Neighborhood_Greens = 1
 link__Neighborhood_GrnHill__fix: x_Neighborhood_GrnHill = 0
 link__Neighborhood_IDOTRR__fix: x_Neighborhood_IDOTRR = 0
 link__Neighborhood_Landmrk__fix: x_Neighborhood_Landmrk = 0
 link__Neighborhood_MeadowV__fix: x_Neighborhood_MeadowV = 0
 link__Neighborhood_Mitchel__fix: x_Neighborhood_Mitchel = 0
 link__Neighborhood_NPkVill__fix: x_Neighborhood_NPkVill = 0
 link__Neighborhood_NWAmes__fix: x_Neighborhood_NWAmes = 0
 link__Neighborhood_No_aplicames__fix: x_Neighborhood_No_aplicames = 0
 link__Neighborhood_NoRidge__fix: x_Neighborhood_NoRidge = 0
 link__Neighborhood_NridgHt__fix: x_Neighborhood_NridgHt = 0
 link__Neighborhood_OldTown__fix: x_Neighborhood_OldTown = 0
 link__Neighborhood_SWISU__fix: x_Neighborhood_SWISU = 0
 link__Neighborhood_Sawyer__fix: x_Neighborhood_Sawyer = 0
 link__Neighborhood_SawyerW__fix: x_Neighborhood_SawyerW = 0
 link__Neighborhood_Somerst__fix: x_Neighborhood_Somerst = 0
 link__Neighborhood_StoneBr__fix: x_Neighborhood_StoneBr = 0
 link__Neighborhood_Timber__fix: x_Neighborhood_Timber = 0
 link__Neighborhood_Veenker__fix: x_Neighborhood_Veenker = 0
 ex__x__mssubclass__onehot: x_MS_SubClass_20 + x_MS_SubClass_30
   + x_MS_SubClass_40 + x_MS_SubClass_45 + x_MS_SubClass_50
   + x_MS_SubClass_60 + x_MS_SubClass_70 + x_MS_SubClass_75
   + x_MS_SubClass_80 + x_MS_SubClass_85 + x_MS_SubClass_90
   + x_MS_SubClass_120 + x_MS_SubClass_150 + x_MS_SubClass_160
   + x_MS_SubClass_180 + x_MS_SubClass_190 = 1
 link__fence__fix__Fence_GdPrv: x_Fence_GdPrv = 0
 link__fence__fix__Fence_GdWo: x_Fence_GdWo = 0
 link__fence__fix__Fence_MnPrv: x_Fence_MnPrv = 0
 link__fence__fix__Fence_MnWw: x_Fence_MnWw = 0
 link__fence__fix__Fence_No_aplica: x_Fence_No_aplica = 1
 fix__yearbuilt: x_Year_Built = 2025
 fix__yearremodadd: x_Year_Remod/Add = 2025
 fix__yrsold: x_Yr_Sold = 2025
 link__fix_ohe__mosold__mosold1: x_Mo_Sold_1 = 0
 link__fix_ohe__mosold__mosold2: x_Mo_Sold_2 = 0
 link__fix_ohe__mosold__mosold3: x_Mo_Sold_3 = 0
 link__fix_ohe__mosold__mosold4: x_Mo_Sold_4 = 0
 link__fix_ohe__mosold__mosold5: x_Mo_Sold_5 = 0
 link__fix_ohe__mosold__mosold6: x_Mo_Sold_6 = 1
 link__fix_ohe__mosold__mosold7: x_Mo_Sold_7 = 0
 link__fix_ohe__mosold__mosold8: x_Mo_Sold_8 = 0
 link__fix_ohe__mosold__mosold9: x_Mo_Sold_9 = 0
 link__fix_ohe__mosold__mosold10: x_Mo_Sold_10 = 0
 link__fix_ohe__mosold__mosold11: x_Mo_Sold_11 = 0
 link__fix_ohe__mosold__mosold12: x_Mo_Sold_12 = 0
 link__fix_ohe__mszoning__mszoningaagr: x_MS_Zoning_A_(agr) = 0
 link__fix_ohe__mszoning__mszoningcall: x_MS_Zoning_C_(all) = 0
 link__fix_ohe__mszoning__mszoningfv: x_MS_Zoning_FV = 0
 link__fix_ohe__mszoning__mszoningiall: x_MS_Zoning_I_(all) = 0
 link__fix_ohe__mszoning__mszoningrh: x_MS_Zoning_RH = 0
 link__fix_ohe__mszoning__mszoningrl: x_MS_Zoning_RL = 1
 link__fix_ohe__mszoning__mszoningrm: x_MS_Zoning_RM = 0
 link__fix_ohe__street__streetgrvl: x_Street_Grvl = 0
 link__fix_ohe__street__streetpave: x_Street_Pave = 1
 link__fix_ohe__alley__alleygrvl: x_Alley_Grvl = 0
 link__fix_ohe__alley__alleynoaplica: x_Alley_No_aplica = 1
 link__fix_ohe__alley__alleypave: x_Alley_Pave = 0
 link__fix_ohe__lotshape__lotshapeir1: x_Lot_Shape_IR1 = 0
 link__fix_ohe__lotshape__lotshapeir2: x_Lot_Shape_IR2 = 0
 link__fix_ohe__lotshape__lotshapeir3: x_Lot_Shape_IR3 = 0
 link__fix_ohe__lotshape__lotshapereg: x_Lot_Shape_Reg = 1
 link__fix_ohe__landcontour__landcontourbnk: x_Land_Contour_Bnk = 0
 link__fix_ohe__landcontour__landcontourhls: x_Land_Contour_HLS = 0
 link__fix_ohe__landcontour__landcontourlow: x_Land_Contour_Low = 0
 link__fix_ohe__landcontour__landcontourlvl: x_Land_Contour_Lvl = 1
 link__fix_ohe__lotconfig__lotconfigcorner: x_Lot_Config_Corner = 0
 link__fix_ohe__lotconfig__lotconfigculdsac: x_Lot_Config_CulDSac = 0
 link__fix_ohe__lotconfig__lotconfigfr2: x_Lot_Config_FR2 = 0
 link__fix_ohe__lotconfig__lotconfigfr3: x_Lot_Config_FR3 = 0
 link__fix_ohe__lotconfig__lotconfiginside: x_Lot_Config_Inside = 1
 link__fix_ohe__landslope__landslopegtl: x_Land_Slope_Gtl = 1
 link__fix_ohe__landslope__landslopemod: x_Land_Slope_Mod = 0
 link__fix_ohe__landslope__landslopesev: x_Land_Slope_Sev = 0
 link__fix_ohe__condition1__condition1artery: x_Condition_1_Artery = 0
 link__fix_ohe__condition1__condition1feedr: x_Condition_1_Feedr = 0
 link__fix_ohe__condition1__condition1norm: x_Condition_1_Norm = 1
 link__fix_ohe__condition1__condition1posa: x_Condition_1_PosA = 0
 link__fix_ohe__condition1__condition1posn: x_Condition_1_PosN = 0
 link__fix_ohe__condition1__condition1rrae: x_Condition_1_RRAe = 0
 link__fix_ohe__condition1__condition1rran: x_Condition_1_RRAn = 0
 link__fix_ohe__condition1__condition1rrne: x_Condition_1_RRNe = 0
 link__fix_ohe__condition1__condition1rrnn: x_Condition_1_RRNn = 0
 link__fix_ohe__condition2__condition2artery: x_Condition_2_Artery = 0
 link__fix_ohe__condition2__condition2feedr: x_Condition_2_Feedr = 0
 link__fix_ohe__condition2__condition2norm: x_Condition_2_Norm = 1
 link__fix_ohe__condition2__condition2posa: x_Condition_2_PosA = 0
 link__fix_ohe__condition2__condition2posn: x_Condition_2_PosN = 0
 link__fix_ohe__condition2__condition2rrae: x_Condition_2_RRAe = 0
 link__fix_ohe__condition2__condition2rran: x_Condition_2_RRAn = 0
 link__fix_ohe__condition2__condition2rrnn: x_Condition_2_RRNn = 0
 link__fix_ohe__functional__functionoaplicalmaj1: x_FunctioNo_aplical_Maj1
   = 0
 link__fix_ohe__functional__functionoaplicalmaj2: x_FunctioNo_aplical_Maj2
   = 0
 link__fix_ohe__functional__functionoaplicalmin1: x_FunctioNo_aplical_Min1
   = 0
 link__fix_ohe__functional__functionoaplicalmin2: x_FunctioNo_aplical_Min2
   = 0
 link__fix_ohe__functional__functionoaplicalmod: x_FunctioNo_aplical_Mod
   = 0
 link__fix_ohe__functional__functionoaplicalsal: x_FunctioNo_aplical_Sal
   = 0
 link__fix_ohe__functional__functionoaplicalsev: x_FunctioNo_aplical_Sev
   = 0
 link__fix_ohe__functional__functionoaplicaltyp: x_FunctioNo_aplical_Typ
   = 1
 link__fix_ohe__saletype__saletypecod: x_Sale_Type_COD = 0
 link__fix_ohe__saletype__saletypecwd: x_Sale_Type_CWD = 0
 link__fix_ohe__saletype__saletypecon: x_Sale_Type_Con = 0
 link__fix_ohe__saletype__saletypeconld: x_Sale_Type_ConLD = 0
 link__fix_ohe__saletype__saletypeconli: x_Sale_Type_ConLI = 0
 link__fix_ohe__saletype__saletypeconlw: x_Sale_Type_ConLw = 0
 link__fix_ohe__saletype__saletypenew: x_Sale_Type_New = 0
 link__fix_ohe__saletype__saletypeoth: x_Sale_Type_Oth = 0
 link__fix_ohe__saletype__saletypevwd: x_Sale_Type_VWD = 0
 link__fix_ohe__saletype__saletypewd: x_Sale_Type_WD_ = 1
 link__fix_ohe__salecondition__saleconditionabnorml:
   x_Sale_Condition_Abnorml = 0
 link__fix_ohe__salecondition__saleconditionadjland:
   x_Sale_Condition_AdjLand = 0
 link__fix_ohe__salecondition__saleconditionalloca: x_Sale_Condition_Alloca
   = 0
 link__fix_ohe__salecondition__saleconditionfamily: x_Sale_Condition_Family
   = 0
 link__fix_ohe__salecondition__saleconditionnormal: x_Sale_Condition_Normal
   = 1
 link__fix_ohe__salecondition__saleconditionpartial:
   x_Sale_Condition_Partial = 0
 link__fix_ohe__bldgtype__bldgtype1fam: x_Bldg_Type_1Fam = 1
 link__fix_ohe__bldgtype__bldgtype2fmcon: x_Bldg_Type_2fmCon = 0
 link__fix_ohe__bldgtype__bldgtypeduplex: x_Bldg_Type_Duplex = 0
 link__fix_ohe__bldgtype__bldgtypetwnhs: x_Bldg_Type_Twnhs = 0
 link__fix_ohe__bldgtype__bldgtypetwnhse: x_Bldg_Type_TwnhsE = 0
 ex__x__mszoning__onehot: x_MS_Zoning_A_(agr) + x_MS_Zoning_C_(all)
   + x_MS_Zoning_FV + x_MS_Zoning_I_(all) + x_MS_Zoning_RH + x_MS_Zoning_RL
   + x_MS_Zoning_RM = 1
 ex__x__street__onehot: x_Street_Grvl + x_Street_Pave = 1
 ex__x__alley__onehot: x_Alley_Grvl + x_Alley_No_aplica + x_Alley_Pave = 1
 ex__x__lotshape__onehot: x_Lot_Shape_IR1 + x_Lot_Shape_IR2
   + x_Lot_Shape_IR3 + x_Lot_Shape_Reg = 1
 ex__x__landcontour__onehot: x_Land_Contour_Bnk + x_Land_Contour_HLS
   + x_Land_Contour_Low + x_Land_Contour_Lvl = 1
 ex__x__lotconfig__onehot: x_Lot_Config_Corner + x_Lot_Config_CulDSac
   + x_Lot_Config_FR2 + x_Lot_Config_FR3 + x_Lot_Config_Inside = 1
 ex__x__landslope__onehot: x_Land_Slope_Gtl + x_Land_Slope_Mod
   + x_Land_Slope_Sev = 1
 ex__x__condition1__onehot: x_Condition_1_Artery + x_Condition_1_Feedr
   + x_Condition_1_Norm + x_Condition_1_PosA + x_Condition_1_PosN
   + x_Condition_1_RRAe + x_Condition_1_RRAn + x_Condition_1_RRNe
   + x_Condition_1_RRNn = 1
 ex__x__condition2__onehot: x_Condition_2_Artery + x_Condition_2_Feedr
   + x_Condition_2_Norm + x_Condition_2_PosA + x_Condition_2_PosN
   + x_Condition_2_RRAe + x_Condition_2_RRAn + x_Condition_2_RRNn = 1
 ex__x__saletype__onehot: x_Sale_Type_COD + x_Sale_Type_CWD
   + x_Sale_Type_Con + x_Sale_Type_ConLD + x_Sale_Type_ConLI
   + x_Sale_Type_ConLw + x_Sale_Type_New + x_Sale_Type_Oth
   + x_Sale_Type_VWD + x_Sale_Type_WD_ = 1
 ex__x__salecondition__onehot: x_Sale_Condition_Abnorml
   + x_Sale_Condition_AdjLand + x_Sale_Condition_Alloca
   + x_Sale_Condition_Family + x_Sale_Condition_Normal
   + x_Sale_Condition_Partial = 1
 ex__x__neighborhood__onehot: x_Neighborhood_Blmngtn
   + x_Neighborhood_Blueste + x_Neighborhood_BrDale
   + x_Neighborhood_BrkSide + x_Neighborhood_ClearCr
   + x_Neighborhood_CollgCr + x_Neighborhood_Crawfor
   + x_Neighborhood_Edwards + x_Neighborhood_Gilbert
   + x_Neighborhood_Greens + x_Neighborhood_GrnHill + x_Neighborhood_IDOTRR
   + x_Neighborhood_Landmrk + x_Neighborhood_MeadowV
   + x_Neighborhood_Mitchel + x_Neighborhood_NPkVill
   + x_Neighborhood_NWAmes + x_Neighborhood_No_aplicames
   + x_Neighborhood_NoRidge + x_Neighborhood_NridgHt
   + x_Neighborhood_OldTown + x_Neighborhood_SWISU + x_Neighborhood_Sawyer
   + x_Neighborhood_SawyerW + x_Neighborhood_Somerst
   + x_Neighborhood_StoneBr + x_Neighborhood_Timber
   + x_Neighborhood_Veenker = 1
 ex__x__exterior1st__onehot: x_Exterior_1st_AsbShng
   + x_Exterior_1st_AsphShn + x_Exterior_1st_BrkComm
   + x_Exterior_1st_BrkFace + x_Exterior_1st_CBlock
   + x_Exterior_1st_CemntBd + x_Exterior_1st_HdBoard
   + x_Exterior_1st_ImStucc + x_Exterior_1st_MetalSd
   + x_Exterior_1st_Plywood + x_Exterior_1st_PreCast + x_Exterior_1st_Stone
   + x_Exterior_1st_Stucco + x_Exterior_1st_VinylSd
   + x_Exterior_1st_Wd_Sdng + x_Exterior_1st_WdShing = 1
 ex__x__exterior2nd__onehot: x_Exterior_2nd_AsbShng
   + x_Exterior_2nd_AsphShn + x_Exterior_2nd_Brk_Cmn
   + x_Exterior_2nd_BrkFace + x_Exterior_2nd_CBlock
   + x_Exterior_2nd_CmentBd + x_Exterior_2nd_HdBoard
   + x_Exterior_2nd_ImStucc + x_Exterior_2nd_MetalSd + x_Exterior_2nd_Other
   + x_Exterior_2nd_Plywood + x_Exterior_2nd_PreCast + x_Exterior_2nd_Stone
   + x_Exterior_2nd_Stucco + x_Exterior_2nd_VinylSd
   + x_Exterior_2nd_Wd_Sdng + x_Exterior_2nd_Wd_Shng = 1
 fix__overallcond: x_Overall_Cond = 10
 fix__extercond__ex: x_Exter_Cond = 4
 fix__bsmtcond__ex: x_Bsmt_Cond = 4
 fix__utilities: x_Utilities = 3
 fix__lowqualfinsf: x_Low_Qual_Fin_SF = 0
 7.4__first_floor_always: Floor1 = 1
 7.4__second_floor_if_first: - Floor1 + Floor2 <= 0
 7.4__one_or_two_story: IsOneStory + IsTwoStory = 1
 7.4__two_story_link: Floor2 - IsTwoStory = 0
 7.21.2__bed2_onoff: - 8 Floor2 + Bedroom2 <= 0
 7.21.2__fullbath2_onoff: - 4 Floor2 + FullBath2 <= 0
 7.21.2__halfbath2_onoff: - 3 Floor2 + HalfBath2 <= 0
 7.21.2__kitchen2_onoff: - 2 Floor2 + Kitchen2 <= 0
 7.22.3__other_cnt_2nd_if_on: - 8 Floor2 + OtherRooms2 <= 0
 link__fullbath: x_Full_Bath - FullBath1 - FullBath2 = 0
 link__halfbath: x_Half_Bath - HalfBath1 - HalfBath2 = 0
 link__kitchenabvgr: x_Kitchen_AbvGr - Kitchen1 - Kitchen2 = 0
 link__bedroomabvgr: x_Bedroom_AbvGr - Bedroom1 - Bedroom2 = 0
 7.1.1__lot_cap: x_1st_Flr_SF + x_Pool_Area + TotalPorchSF <= 7000
 7.1.2__2nd_leq_1st: - x_1st_Flr_SF + x_2nd_Flr_SF <= 0
 7.1.3__GrLivArea: - x_1st_Flr_SF - x_2nd_Flr_SF + x_Gr_Liv_Area = 0
 7.1.4__AFullBath: - AreaFullBath1 - AreaFullBath2 + AreaFullBath = 0
 7.1.5__AHalfBath: - AreaHalfBath1 - AreaHalfBath2 + AreaHalfBath = 0
 11.3__AKitchen: - AreaKitchen1 - AreaKitchen2 + AreaKitchen = 0
 11.3__ABedroom: - AreaBedroom1 - AreaBedroom2 + AreaBedroom = 0
 7.22.1__AreaOther_sum: - AreaOther1 - AreaOther2 + AreaOther = 0
 7.1.6__fullbath_1min: FullBath1 >= 1
 7.1.7__kitchen_1min: Kitchen1 >= 1
 7.1.8__2nd_max: x_2nd_Flr_SF - 3500 Floor2 <= 0
 7.1.8__2nd_min_if_on: x_2nd_Flr_SF - 350 Floor2 >= 0
 7.1.8__1st_min_if_on: x_1st_Flr_SF - 450 Floor1 >= 0
 7.2.1__FullBath_count: - FullBath1 - FullBath2 + FullBath = 0
 7.2.2__HalfBath_count: - HalfBath1 - HalfBath2 + HalfBath = 0
 7.2.3__Kitchen_count: - Kitchen1 - Kitchen2 + Kitchen = 0
 link__fullbath: x_Full_Bath - FullBath = 0
 link__halfbath: x_Half_Bath - HalfBath = 0
 link__kitchenabvgr: x_Kitchen_AbvGr - Kitchen = 0
 11.11__half_leq_full: - FullBath + HalfBath <= 0
 11.xx__has_kitch_leq_cnt: - Kitchen + HasKitchen <= 0
 11.xx__has_kitch_ge_frac: - 0.5 Kitchen + HasKitchen >= 0
 ex__BldgType: BldgType__1Fam + BldgType__TwnhsE + BldgType__TwnhsI
   + BldgType__Duplex + BldgType__2FmCon = 1
 link__bldgtype__1fam: x_Bldg_Type_1Fam - BldgType__1Fam = 0
 link__bldgtype__twnhse: x_Bldg_Type_TwnhsE - BldgType__TwnhsE = 0
 link__bldgtype__duplex: x_Bldg_Type_Duplex - BldgType__Duplex = 0
 link__bldgtype__2fmcon: x_Bldg_Type_2fmCon - BldgType__2FmCon = 0
 7.3.1__bedrooms_sum: - Bedroom1 - Bedroom2 + Bedrooms = 0
 11.xx__kitch_allow_if_area_2: x_Gr_Liv_Area - 5600 zKitchen_2 >= -2400
 11.xx__kitch_allow_if_area_3: x_Gr_Liv_Area - 5600 zKitchen_3 >= -600
 7.3.4__kitchen_by_area_1fam: Kitchen + 10 BldgType__1Fam - zKitchen_2
   - zKitchen_3 <= 11
 link__bedroomabvgr: x_Bedroom_AbvGr - Bedrooms = 0
 11.xx__has_fire_ge_frac: - 0.25 x_Fireplaces + HasFireplace >= 0
 11.xx__fire_leq_cap: x_Fireplaces - 4 HasFireplace <= 0
 onehot__fq: - HasFireplace + zFQ_TA + zFQ_Gd + zFQ_Ex = 0
 part__fplaces_sum: - x_Fireplaces + Fireplaces_TA + Fireplaces_Gd
   + Fireplaces_Ex = 0
 part__fplaces_ta: - 3 zFQ_TA + Fireplaces_TA <= 0
 part__fplaces_gd: - 3 zFQ_Gd + Fireplaces_Gd <= 0
 part__fplaces_ex: - 3 zFQ_Ex + Fireplaces_Ex <= 0
 7.5.1__gar_area_lb: 150 x_Garage_Cars - x_Garage_Area <= 0
 7.5.1__gar_area_ub: - 250 x_Garage_Cars + x_Garage_Area <= 0
 ex__GarageType: GarageType__NA + GarageType__Attchd + GarageType__Detchd
   + GarageType__BuiltIn + GarageType__Basment + GarageType__CarPort
   + GarageType__2Types = 1
 link__garagetype__na: x_Garage_Type_No_aplica - GarageType__NA = 0
 link__garagetype__attchd: x_Garage_Type_Attchd - GarageType__Attchd = 0
 link__garagetype__detchd: x_Garage_Type_Detchd - GarageType__Detchd = 0
 link__garagetype__builtin: x_Garage_Type_BuiltIn - GarageType__BuiltIn
   = 0
 link__garagetype__basment: x_Garage_Type_Basment - GarageType__Basment
   = 0
 link__garagetype__carport: x_Garage_Type_CarPort - GarageType__CarPort
   = 0
 link__garagetype__2types: x_Garage_Type_2Types - GarageType__2Types = 0
 ex__GarageFinish: GarageFinish__NA + GarageFinish__Fin + GarageFinish__RFn
   + GarageFinish__Unf = 1
 link__garagefinish__na: x_Garage_Finish_No_aplica - GarageFinish__NA = 0
 link__garagefinish__fin: x_Garage_Finish_Fin - GarageFinish__Fin = 0
 link__garagefinish__rfn: x_Garage_Finish_RFn - GarageFinish__RFn = 0
 link__garagefinish__unf: x_Garage_Finish_Unf - GarageFinish__Unf = 0
 7.5.3__finish_na_eq_type_na: - GarageType__NA + GarageFinish__NA = 0
 7.5.3__finish_choices_if_has: GarageType__NA + GarageFinish__Fin
   + GarageFinish__RFn = 1
 7.5.4__min_cars_if_has: x_Garage_Cars + GarageType__NA >= 1
 7.5.2__cap_cars_if_has: x_Garage_Cars + 4 GarageType__NA <= 4
 7.5.2__cap_area_if_has: x_Garage_Area + 1400 GarageType__NA <= 1400
 11.xx__garage_cond_ex_or_na: x_Garage_Cond + 5 GarageType__NA = 4
 link__x__garagetype2types: x_Garage_Type_2Types - GarageType__2Types = 0
 link__x__garagetypeattchd: x_Garage_Type_Attchd - GarageType__Attchd = 0
 link__x__garagetypebasment: x_Garage_Type_Basment - GarageType__Basment
   = 0
 link__x__garagetypebuiltin: x_Garage_Type_BuiltIn - GarageType__BuiltIn
   = 0
 link__x__garagetypecarport: x_Garage_Type_CarPort - GarageType__CarPort
   = 0
 link__x__garagetypedetchd: x_Garage_Type_Detchd - GarageType__Detchd = 0
 link__x__garagetypenoaplica: x_Garage_Type_No_aplica - GarageType__NA = 0
 link__x__garagefinishfin: x_Garage_Finish_Fin - GarageFinish__Fin = 0
 link__x__garagefinishrfn: x_Garage_Finish_RFn - GarageFinish__RFn = 0
 link__x__garagefinishunf: x_Garage_Finish_Unf - GarageFinish__Unf = 0
 link__x__garagefinishnoaplica: x_Garage_Finish_No_aplica
   - GarageFinish__NA = 0
 11.xx__gar_yr_zero_if_na: x_Garage_Yr_Blt + 2100 GarageType__NA <= 2100
 11.xx__gar_yr_ge_yb_if_has: - x_Year_Built + x_Garage_Yr_Blt
   + 2100 GarageType__NA >= 0
 11.xx__gar_yr_le_yb_if_has: - x_Year_Built + x_Garage_Yr_Blt
   - 2100 GarageType__NA <= 0
 onehot__gq: GarageType__NA + zGQ_TA + zGQ_Gd + zGQ_Ex = 1
 part__agarage_sum: - x_Garage_Area + AGarage_TA + AGarage_Gd + AGarage_Ex
   = 0
 part__agarage_ta: - 1400 zGQ_TA + AGarage_TA <= 0
 part__agarage_gd: - 1400 zGQ_Gd + AGarage_Gd <= 0
 part__agarage_ex: - 1400 zGQ_Ex + AGarage_Ex <= 0
 onehot__hq: zHQ_TA + zHQ_Gd + zHQ_Ex = 1
 link__hq_ord: x_Heating_QC - 2 zHQ_TA - 3 zHQ_Gd - 4 zHQ_Ex = 0
 onehot__eq: zEQ_TA + zEQ_Gd + zEQ_Ex = 1
 link__eq_ord: x_Exter_Qual - 2 zEQ_TA - 3 zEQ_Gd - 4 zEQ_Ex = 0
 ex__HouseStyle: HouseStyle__1Story + HouseStyle__2Story = 1
 11.1__floor2_only_if_2story: Floor2 - HouseStyle__2Story = 0
 7.4__hs_one_story: IsOneStory - HouseStyle__1Story = 0
 7.4__hs_two_story: IsTwoStory - HouseStyle__2Story = 0
 link__mssubclass__20_if_1story: x_MS_SubClass_20 - HouseStyle__1Story = 0
 link__mssubclass__60_if_2story: x_MS_SubClass_60 - HouseStyle__2Story = 0
 link__mssubclass__off__MS_SubClass_30: x_MS_SubClass_30 = 0
 link__mssubclass__off__MS_SubClass_40: x_MS_SubClass_40 = 0
 link__mssubclass__off__MS_SubClass_45: x_MS_SubClass_45 = 0
 link__mssubclass__off__MS_SubClass_50: x_MS_SubClass_50 = 0
 link__mssubclass__off__MS_SubClass_70: x_MS_SubClass_70 = 0
 link__mssubclass__off__MS_SubClass_75: x_MS_SubClass_75 = 0
 link__mssubclass__off__MS_SubClass_80: x_MS_SubClass_80 = 0
 link__mssubclass__off__MS_SubClass_85: x_MS_SubClass_85 = 0
 link__mssubclass__off__MS_SubClass_90: x_MS_SubClass_90 = 0
 link__mssubclass__off__MS_SubClass_120: x_MS_SubClass_120 = 0
 link__mssubclass__off__MS_SubClass_150: x_MS_SubClass_150 = 0
 link__mssubclass__off__MS_SubClass_160: x_MS_SubClass_160 = 0
 link__mssubclass__off__MS_SubClass_180: x_MS_SubClass_180 = 0
 link__mssubclass__off__MS_SubClass_190: x_MS_SubClass_190 = 0
 ex__Heating: Heating__Floor + Heating__GasA + Heating__GasW
   + Heating__Grav + Heating__OthW + Heating__Wall = 1
 link__heating__floor: x_Heating_Floor - Heating__Floor = 0
 link__heating__gasa: x_Heating_GasA - Heating__GasA = 0
 link__heating__gasw: x_Heating_GasW - Heating__GasW = 0
 link__heating__grav: x_Heating_Grav - Heating__Grav = 0
 link__heating__othw: x_Heating_OthW - Heating__OthW = 0
 link__heating__wall: x_Heating_Wall - Heating__Wall = 0
 ex__CentralAir: CentralAir__N + CentralAir__Y = 1
 link__centralair__n: x_Central_Air_N - CentralAir__N = 0
 link__centralair__y: x_Central_Air_Y - CentralAir__Y = 0
 ex__Electrical: Electrical__SBrkr + Electrical__FuseA + Electrical__FuseF
   + Electrical__FuseP + Electrical__Mix = 1
 link__electrical__sbrkr: x_Electrical_SBrkr - Electrical__SBrkr = 0
 link__electrical__fusea: x_Electrical_FuseA - Electrical__FuseA = 0
 link__electrical__fusef: x_Electrical_FuseF - Electrical__FuseF = 0
 link__electrical__fusep: x_Electrical_FuseP - Electrical__FuseP = 0
 link__electrical__mix: x_Electrical_Mix - Electrical__Mix = 0
 ex__PavedDrive: PavedDrive__Y + PavedDrive__P + PavedDrive__N = 1
 link__paveddrive__y: x_Paved_Drive_Y - PavedDrive__Y = 0
 link__paveddrive__p: x_Paved_Drive_N - PavedDrive__P = 0
 link__paveddrive__n: x_Paved_Drive_N - PavedDrive__N = 0
 ex__MiscFeature: MiscFeature__Elev + MiscFeature__Gar2 + MiscFeature__Othr
   + MiscFeature__Shed + MiscFeature__TenC + MiscFeature__No_aplica = 1
 link__miscfeature__elev: x_Misc_Feature_Elev - MiscFeature__Elev = 0
 link__miscfeature__gar2: x_Misc_Feature_Gar2 - MiscFeature__Gar2 = 0
 link__miscfeature__othr: x_Misc_Feature_Othr - MiscFeature__Othr = 0
 link__miscfeature__shed: x_Misc_Feature_Shed - MiscFeature__Shed = 0
 link__miscfeature__tenc: x_Misc_Feature_TenC - MiscFeature__TenC = 0
 link__miscfeature__noaplica: x_Misc_Feature_No_aplica
   - MiscFeature__No_aplica = 0
 link__overallqual_var_int: x_Overall_Qual - OverallQualInt = 0
 def__overallqual_score: - 0.4 x_Exter_Qual - 0.4 x_Bsmt_Qual
   - 0.4 x_Heating_QC - 0.4 x_Kitchen_Qual - 0.4 x_Garage_Qual
   + OverallQualScore = 2
 overallqual_floor_le: OverallQualInt - OverallQualScore <= 0
 overallqual_floor_ge: OverallQualInt - OverallQualScore >= -0.999
 7.7.2__PlanRoof_sum: - PR1 - PR2 + PlanRoofArea = 0
 7.7.1__pr1_leq_1st: - x_1st_Flr_SF + PR1 <= 0
 7.7.1__pr1_if_floor1: - 4200 Floor1 + PR1 <= 0
 7.7.1__pr1_linear: - x_1st_Flr_SF - 4200 Floor1 + PR1 >= -4200
 7.7.1__pr2_leq_2nd: - x_2nd_Flr_SF + PR2 <= 0
 7.7.1__pr2_if_floor2: - 3500 Floor2 + PR2 <= 0
 7.7.1__pr2_linear: - x_2nd_Flr_SF - 3500 Floor2 + PR2 >= -3500
 7.7.1__planroof_ub: PlanRoofArea <= 4200
 ex__RoofStyle: RoofStyle__Flat + RoofStyle__Gable + RoofStyle__Gambrel
   + RoofStyle__Hip + RoofStyle__Mansard + RoofStyle__Shed = 1
 ex__RoofMatl: RoofMatl__ClyTile + RoofMatl__CompShg + RoofMatl__Membran
   + RoofMatl__Metal + RoofMatl__Roll + RoofMatl__Tar&Grv
   + RoofMatl__WdShake + RoofMatl__WdShngl = 1
 link__roofstyle__flat: x_Roof_Style_Flat - RoofStyle__Flat = 0
 link__roofstyle__gable: x_Roof_Style_Gable - RoofStyle__Gable = 0
 link__roofstyle__gambrel: x_Roof_Style_Gambrel - RoofStyle__Gambrel = 0
 link__roofstyle__hip: x_Roof_Style_Hip - RoofStyle__Hip = 0
 link__roofstyle__mansard: x_Roof_Style_Mansard - RoofStyle__Mansard = 0
 link__roofstyle__shed: x_Roof_Style_Shed - RoofStyle__Shed = 0
 link__roofmatl__clytile: x_Roof_Matl_ClyTile - RoofMatl__ClyTile = 0
 link__roofmatl__compshg: x_Roof_Matl_CompShg - RoofMatl__CompShg = 0
 link__roofmatl__membran: x_Roof_Matl_Membran - RoofMatl__Membran = 0
 link__roofmatl__metal: x_Roof_Matl_Metal - RoofMatl__Metal = 0
 link__roofmatl__roll: x_Roof_Matl_Roll - RoofMatl__Roll = 0
 link__roofmatl__targrv: x_Roof_Matl_Tar&Grv - RoofMatl__Tar&Grv = 0
 link__roofmatl__wdshake: x_Roof_Matl_WdShake - RoofMatl__WdShake = 0
 link__roofmatl__wdshngl: x_Roof_Matl_WdShngl - RoofMatl__WdShngl = 0
 7.7.4__y_leq_rs__Flat_ClyTile: - RoofStyle__Flat + Y__Flat__ClyTile <= 0
 7.7.4__y_leq_rm__Flat_ClyTile: - RoofMatl__ClyTile + Y__Flat__ClyTile
   <= 0
 7.7.4__y_ge_and__Flat_ClyTile: - RoofStyle__Flat - RoofMatl__ClyTile
   + Y__Flat__ClyTile >= -1
 7.7.5__z_leq_plan__Flat_ClyTile: - PlanRoofArea + Z__Flat__ClyTile <= 0
 7.7.5__z_leq_u_y__Flat_ClyTile: - 4200 Y__Flat__ClyTile + Z__Flat__ClyTile
   <= 0
 7.7.5__z_ge_plan_u__Flat_ClyTile: - PlanRoofArea - 4200 Y__Flat__ClyTile
   + Z__Flat__ClyTile >= -4200
 7.7.4__y_leq_rs__Flat_CompShg: - RoofStyle__Flat + Y__Flat__CompShg <= 0
 7.7.4__y_leq_rm__Flat_CompShg: - RoofMatl__CompShg + Y__Flat__CompShg
   <= 0
 7.7.4__y_ge_and__Flat_CompShg: - RoofStyle__Flat - RoofMatl__CompShg
   + Y__Flat__CompShg >= -1
 7.7.5__z_leq_plan__Flat_CompShg: - PlanRoofArea + Z__Flat__CompShg <= 0
 7.7.5__z_leq_u_y__Flat_CompShg: - 4200 Y__Flat__CompShg + Z__Flat__CompShg
   <= 0
 7.7.5__z_ge_plan_u__Flat_CompShg: - PlanRoofArea - 4200 Y__Flat__CompShg
   + Z__Flat__CompShg >= -4200
 7.7.4__y_leq_rs__Flat_Membran: - RoofStyle__Flat + Y__Flat__Membran <= 0
 7.7.4__y_leq_rm__Flat_Membran: - RoofMatl__Membran + Y__Flat__Membran
   <= 0
 7.7.4__y_ge_and__Flat_Membran: - RoofStyle__Flat - RoofMatl__Membran
   + Y__Flat__Membran >= -1
 7.7.5__z_leq_plan__Flat_Membran: - PlanRoofArea + Z__Flat__Membran <= 0
 7.7.5__z_leq_u_y__Flat_Membran: - 4200 Y__Flat__Membran + Z__Flat__Membran
   <= 0
 7.7.5__z_ge_plan_u__Flat_Membran: - PlanRoofArea - 4200 Y__Flat__Membran
   + Z__Flat__Membran >= -4200
 7.7.4__y_leq_rs__Flat_Metal: - RoofStyle__Flat + Y__Flat__Metal <= 0
 7.7.4__y_leq_rm__Flat_Metal: - RoofMatl__Metal + Y__Flat__Metal <= 0
 7.7.4__y_ge_and__Flat_Metal: - RoofStyle__Flat - RoofMatl__Metal
   + Y__Flat__Metal >= -1
 7.7.5__z_leq_plan__Flat_Metal: - PlanRoofArea + Z__Flat__Metal <= 0
 7.7.5__z_leq_u_y__Flat_Metal: - 4200 Y__Flat__Metal + Z__Flat__Metal <= 0
 7.7.5__z_ge_plan_u__Flat_Metal: - PlanRoofArea - 4200 Y__Flat__Metal
   + Z__Flat__Metal >= -4200
 7.7.4__y_leq_rs__Flat_Roll: - RoofStyle__Flat + Y__Flat__Roll <= 0
 7.7.4__y_leq_rm__Flat_Roll: - RoofMatl__Roll + Y__Flat__Roll <= 0
 7.7.4__y_ge_and__Flat_Roll: - RoofStyle__Flat - RoofMatl__Roll
   + Y__Flat__Roll >= -1
 7.7.5__z_leq_plan__Flat_Roll: - PlanRoofArea + Z__Flat__Roll <= 0
 7.7.5__z_leq_u_y__Flat_Roll: - 4200 Y__Flat__Roll + Z__Flat__Roll <= 0
 7.7.5__z_ge_plan_u__Flat_Roll: - PlanRoofArea - 4200 Y__Flat__Roll
   + Z__Flat__Roll >= -4200
 7.7.4__y_leq_rs__Flat_Tar&Grv: - RoofStyle__Flat + Y__Flat__Tar&Grv <= 0
 7.7.4__y_leq_rm__Flat_Tar&Grv: - RoofMatl__Tar&Grv + Y__Flat__Tar&Grv
   <= 0
 7.7.4__y_ge_and__Flat_Tar&Grv: - RoofStyle__Flat - RoofMatl__Tar&Grv
   + Y__Flat__Tar&Grv >= -1
 7.7.5__z_leq_plan__Flat_Tar&Grv: - PlanRoofArea + Z__Flat__Tar&Grv <= 0
 7.7.5__z_leq_u_y__Flat_Tar&Grv: - 4200 Y__Flat__Tar&Grv + Z__Flat__Tar&Grv
   <= 0
 7.7.5__z_ge_plan_u__Flat_Tar&Grv: - PlanRoofArea - 4200 Y__Flat__Tar&Grv
   + Z__Flat__Tar&Grv >= -4200
 7.7.4__y_leq_rs__Flat_WdShake: - RoofStyle__Flat + Y__Flat__WdShake <= 0
 7.7.4__y_leq_rm__Flat_WdShake: - RoofMatl__WdShake + Y__Flat__WdShake
   <= 0
 7.7.4__y_ge_and__Flat_WdShake: - RoofStyle__Flat - RoofMatl__WdShake
   + Y__Flat__WdShake >= -1
 7.7.5__z_leq_plan__Flat_WdShake: - PlanRoofArea + Z__Flat__WdShake <= 0
 7.7.5__z_leq_u_y__Flat_WdShake: - 4200 Y__Flat__WdShake + Z__Flat__WdShake
   <= 0
 7.7.5__z_ge_plan_u__Flat_WdShake: - PlanRoofArea - 4200 Y__Flat__WdShake
   + Z__Flat__WdShake >= -4200
 7.7.4__y_leq_rs__Flat_WdShngl: - RoofStyle__Flat + Y__Flat__WdShngl <= 0
 7.7.4__y_leq_rm__Flat_WdShngl: - RoofMatl__WdShngl + Y__Flat__WdShngl
   <= 0
 7.7.4__y_ge_and__Flat_WdShngl: - RoofStyle__Flat - RoofMatl__WdShngl
   + Y__Flat__WdShngl >= -1
 7.7.5__z_leq_plan__Flat_WdShngl: - PlanRoofArea + Z__Flat__WdShngl <= 0
 7.7.5__z_leq_u_y__Flat_WdShngl: - 4200 Y__Flat__WdShngl + Z__Flat__WdShngl
   <= 0
 7.7.5__z_ge_plan_u__Flat_WdShngl: - PlanRoofArea - 4200 Y__Flat__WdShngl
   + Z__Flat__WdShngl >= -4200
 7.7.4__y_leq_rs__Gable_ClyTile: - RoofStyle__Gable + Y__Gable__ClyTile
   <= 0
 7.7.4__y_leq_rm__Gable_ClyTile: - RoofMatl__ClyTile + Y__Gable__ClyTile
   <= 0
 7.7.4__y_ge_and__Gable_ClyTile: - RoofStyle__Gable - RoofMatl__ClyTile
   + Y__Gable__ClyTile >= -1
 7.7.5__z_leq_plan__Gable_ClyTile: - PlanRoofArea + Z__Gable__ClyTile <= 0
 7.7.5__z_leq_u_y__Gable_ClyTile: - 4200 Y__Gable__ClyTile
   + Z__Gable__ClyTile <= 0
 7.7.5__z_ge_plan_u__Gable_ClyTile: - PlanRoofArea - 4200 Y__Gable__ClyTile
   + Z__Gable__ClyTile >= -4200
 7.7.4__y_leq_rs__Gable_CompShg: - RoofStyle__Gable + Y__Gable__CompShg
   <= 0
 7.7.4__y_leq_rm__Gable_CompShg: - RoofMatl__CompShg + Y__Gable__CompShg
   <= 0
 7.7.4__y_ge_and__Gable_CompShg: - RoofStyle__Gable - RoofMatl__CompShg
   + Y__Gable__CompShg >= -1
 7.7.5__z_leq_plan__Gable_CompShg: - PlanRoofArea + Z__Gable__CompShg <= 0
 7.7.5__z_leq_u_y__Gable_CompShg: - 4200 Y__Gable__CompShg
   + Z__Gable__CompShg <= 0
 7.7.5__z_ge_plan_u__Gable_CompShg: - PlanRoofArea - 4200 Y__Gable__CompShg
   + Z__Gable__CompShg >= -4200
 7.7.4__y_leq_rs__Gable_Membran: - RoofStyle__Gable + Y__Gable__Membran
   <= 0
 7.7.4__y_leq_rm__Gable_Membran: - RoofMatl__Membran + Y__Gable__Membran
   <= 0
 7.7.4__y_ge_and__Gable_Membran: - RoofStyle__Gable - RoofMatl__Membran
   + Y__Gable__Membran >= -1
 7.7.5__z_leq_plan__Gable_Membran: - PlanRoofArea + Z__Gable__Membran <= 0
 7.7.5__z_leq_u_y__Gable_Membran: - 4200 Y__Gable__Membran
   + Z__Gable__Membran <= 0
 7.7.5__z_ge_plan_u__Gable_Membran: - PlanRoofArea - 4200 Y__Gable__Membran
   + Z__Gable__Membran >= -4200
 7.7.4__y_leq_rs__Gable_Metal: - RoofStyle__Gable + Y__Gable__Metal <= 0
 7.7.4__y_leq_rm__Gable_Metal: - RoofMatl__Metal + Y__Gable__Metal <= 0
 7.7.4__y_ge_and__Gable_Metal: - RoofStyle__Gable - RoofMatl__Metal
   + Y__Gable__Metal >= -1
 7.7.5__z_leq_plan__Gable_Metal: - PlanRoofArea + Z__Gable__Metal <= 0
 7.7.5__z_leq_u_y__Gable_Metal: - 4200 Y__Gable__Metal + Z__Gable__Metal
   <= 0
 7.7.5__z_ge_plan_u__Gable_Metal: - PlanRoofArea - 4200 Y__Gable__Metal
   + Z__Gable__Metal >= -4200
 7.7.4__y_leq_rs__Gable_Roll: - RoofStyle__Gable + Y__Gable__Roll <= 0
 7.7.4__y_leq_rm__Gable_Roll: - RoofMatl__Roll + Y__Gable__Roll <= 0
 7.7.4__y_ge_and__Gable_Roll: - RoofStyle__Gable - RoofMatl__Roll
   + Y__Gable__Roll >= -1
 7.7.5__z_leq_plan__Gable_Roll: - PlanRoofArea + Z__Gable__Roll <= 0
 7.7.5__z_leq_u_y__Gable_Roll: - 4200 Y__Gable__Roll + Z__Gable__Roll <= 0
 7.7.5__z_ge_plan_u__Gable_Roll: - PlanRoofArea - 4200 Y__Gable__Roll
   + Z__Gable__Roll >= -4200
 7.7.4__y_leq_rs__Gable_Tar&Grv: - RoofStyle__Gable + Y__Gable__Tar&Grv
   <= 0
 7.7.4__y_leq_rm__Gable_Tar&Grv: - RoofMatl__Tar&Grv + Y__Gable__Tar&Grv
   <= 0
 7.7.4__y_ge_and__Gable_Tar&Grv: - RoofStyle__Gable - RoofMatl__Tar&Grv
   + Y__Gable__Tar&Grv >= -1
 7.7.5__z_leq_plan__Gable_Tar&Grv: - PlanRoofArea + Z__Gable__Tar&Grv <= 0
 7.7.5__z_leq_u_y__Gable_Tar&Grv: - 4200 Y__Gable__Tar&Grv
   + Z__Gable__Tar&Grv <= 0
 7.7.5__z_ge_plan_u__Gable_Tar&Grv: - PlanRoofArea - 4200 Y__Gable__Tar&Grv
   + Z__Gable__Tar&Grv >= -4200
 7.7.4__y_leq_rs__Gable_WdShake: - RoofStyle__Gable + Y__Gable__WdShake
   <= 0
 7.7.4__y_leq_rm__Gable_WdShake: - RoofMatl__WdShake + Y__Gable__WdShake
   <= 0
 7.7.4__y_ge_and__Gable_WdShake: - RoofStyle__Gable - RoofMatl__WdShake
   + Y__Gable__WdShake >= -1
 7.7.5__z_leq_plan__Gable_WdShake: - PlanRoofArea + Z__Gable__WdShake <= 0
 7.7.5__z_leq_u_y__Gable_WdShake: - 4200 Y__Gable__WdShake
   + Z__Gable__WdShake <= 0
 7.7.5__z_ge_plan_u__Gable_WdShake: - PlanRoofArea - 4200 Y__Gable__WdShake
   + Z__Gable__WdShake >= -4200
 7.7.4__y_leq_rs__Gable_WdShngl: - RoofStyle__Gable + Y__Gable__WdShngl
   <= 0
 7.7.4__y_leq_rm__Gable_WdShngl: - RoofMatl__WdShngl + Y__Gable__WdShngl
   <= 0
 7.7.4__y_ge_and__Gable_WdShngl: - RoofStyle__Gable - RoofMatl__WdShngl
   + Y__Gable__WdShngl >= -1
 7.7.5__z_leq_plan__Gable_WdShngl: - PlanRoofArea + Z__Gable__WdShngl <= 0
 7.7.5__z_leq_u_y__Gable_WdShngl: - 4200 Y__Gable__WdShngl
   + Z__Gable__WdShngl <= 0
 7.7.5__z_ge_plan_u__Gable_WdShngl: - PlanRoofArea - 4200 Y__Gable__WdShngl
   + Z__Gable__WdShngl >= -4200
 7.7.4__y_leq_rs__Gambrel_ClyTile: - RoofStyle__Gambrel
   + Y__Gambrel__ClyTile <= 0
 7.7.4__y_leq_rm__Gambrel_ClyTile: - RoofMatl__ClyTile
   + Y__Gambrel__ClyTile <= 0
 7.7.4__y_ge_and__Gambrel_ClyTile: - RoofStyle__Gambrel - RoofMatl__ClyTile
   + Y__Gambrel__ClyTile >= -1
 7.7.5__z_leq_plan__Gambrel_ClyTile: - PlanRoofArea + Z__Gambrel__ClyTile
   <= 0
 7.7.5__z_leq_u_y__Gambrel_ClyTile: - 4200 Y__Gambrel__ClyTile
   + Z__Gambrel__ClyTile <= 0
 7.7.5__z_ge_plan_u__Gambrel_ClyTile: - PlanRoofArea
   - 4200 Y__Gambrel__ClyTile + Z__Gambrel__ClyTile >= -4200
 7.7.4__y_leq_rs__Gambrel_CompShg: - RoofStyle__Gambrel
   + Y__Gambrel__CompShg <= 0
 7.7.4__y_leq_rm__Gambrel_CompShg: - RoofMatl__CompShg
   + Y__Gambrel__CompShg <= 0
 7.7.4__y_ge_and__Gambrel_CompShg: - RoofStyle__Gambrel - RoofMatl__CompShg
   + Y__Gambrel__CompShg >= -1
 7.7.5__z_leq_plan__Gambrel_CompShg: - PlanRoofArea + Z__Gambrel__CompShg
   <= 0
 7.7.5__z_leq_u_y__Gambrel_CompShg: - 4200 Y__Gambrel__CompShg
   + Z__Gambrel__CompShg <= 0
 7.7.5__z_ge_plan_u__Gambrel_CompShg: - PlanRoofArea
   - 4200 Y__Gambrel__CompShg + Z__Gambrel__CompShg >= -4200
 7.7.4__y_leq_rs__Gambrel_Membran: - RoofStyle__Gambrel
   + Y__Gambrel__Membran <= 0
 7.7.4__y_leq_rm__Gambrel_Membran: - RoofMatl__Membran
   + Y__Gambrel__Membran <= 0
 7.7.4__y_ge_and__Gambrel_Membran: - RoofStyle__Gambrel - RoofMatl__Membran
   + Y__Gambrel__Membran >= -1
 7.7.5__z_leq_plan__Gambrel_Membran: - PlanRoofArea + Z__Gambrel__Membran
   <= 0
 7.7.5__z_leq_u_y__Gambrel_Membran: - 4200 Y__Gambrel__Membran
   + Z__Gambrel__Membran <= 0
 7.7.5__z_ge_plan_u__Gambrel_Membran: - PlanRoofArea
   - 4200 Y__Gambrel__Membran + Z__Gambrel__Membran >= -4200
 7.7.4__y_leq_rs__Gambrel_Metal: - RoofStyle__Gambrel + Y__Gambrel__Metal
   <= 0
 7.7.4__y_leq_rm__Gambrel_Metal: - RoofMatl__Metal + Y__Gambrel__Metal
   <= 0
 7.7.4__y_ge_and__Gambrel_Metal: - RoofStyle__Gambrel - RoofMatl__Metal
   + Y__Gambrel__Metal >= -1
 7.7.5__z_leq_plan__Gambrel_Metal: - PlanRoofArea + Z__Gambrel__Metal <= 0
 7.7.5__z_leq_u_y__Gambrel_Metal: - 4200 Y__Gambrel__Metal
   + Z__Gambrel__Metal <= 0
 7.7.5__z_ge_plan_u__Gambrel_Metal: - PlanRoofArea - 4200 Y__Gambrel__Metal
   + Z__Gambrel__Metal >= -4200
 7.7.4__y_leq_rs__Gambrel_Roll: - RoofStyle__Gambrel + Y__Gambrel__Roll
   <= 0
 7.7.4__y_leq_rm__Gambrel_Roll: - RoofMatl__Roll + Y__Gambrel__Roll <= 0
 7.7.4__y_ge_and__Gambrel_Roll: - RoofStyle__Gambrel - RoofMatl__Roll
   + Y__Gambrel__Roll >= -1
 7.7.5__z_leq_plan__Gambrel_Roll: - PlanRoofArea + Z__Gambrel__Roll <= 0
 7.7.5__z_leq_u_y__Gambrel_Roll: - 4200 Y__Gambrel__Roll + Z__Gambrel__Roll
   <= 0
 7.7.5__z_ge_plan_u__Gambrel_Roll: - PlanRoofArea - 4200 Y__Gambrel__Roll
   + Z__Gambrel__Roll >= -4200
 7.7.4__y_leq_rs__Gambrel_Tar&Grv: - RoofStyle__Gambrel
   + Y__Gambrel__Tar&Grv <= 0
 7.7.4__y_leq_rm__Gambrel_Tar&Grv: - RoofMatl__Tar&Grv
   + Y__Gambrel__Tar&Grv <= 0
 7.7.4__y_ge_and__Gambrel_Tar&Grv: - RoofStyle__Gambrel - RoofMatl__Tar&Grv
   + Y__Gambrel__Tar&Grv >= -1
 7.7.5__z_leq_plan__Gambrel_Tar&Grv: - PlanRoofArea + Z__Gambrel__Tar&Grv
   <= 0
 7.7.5__z_leq_u_y__Gambrel_Tar&Grv: - 4200 Y__Gambrel__Tar&Grv
   + Z__Gambrel__Tar&Grv <= 0
 7.7.5__z_ge_plan_u__Gambrel_Tar&Grv: - PlanRoofArea
   - 4200 Y__Gambrel__Tar&Grv + Z__Gambrel__Tar&Grv >= -4200
 7.7.4__y_leq_rs__Gambrel_WdShake: - RoofStyle__Gambrel
   + Y__Gambrel__WdShake <= 0
 7.7.4__y_leq_rm__Gambrel_WdShake: - RoofMatl__WdShake
   + Y__Gambrel__WdShake <= 0
 7.7.4__y_ge_and__Gambrel_WdShake: - RoofStyle__Gambrel - RoofMatl__WdShake
   + Y__Gambrel__WdShake >= -1
 7.7.5__z_leq_plan__Gambrel_WdShake: - PlanRoofArea + Z__Gambrel__WdShake
   <= 0
 7.7.5__z_leq_u_y__Gambrel_WdShake: - 4200 Y__Gambrel__WdShake
   + Z__Gambrel__WdShake <= 0
 7.7.5__z_ge_plan_u__Gambrel_WdShake: - PlanRoofArea
   - 4200 Y__Gambrel__WdShake + Z__Gambrel__WdShake >= -4200
 7.7.4__y_leq_rs__Gambrel_WdShngl: - RoofStyle__Gambrel
   + Y__Gambrel__WdShngl <= 0
 7.7.4__y_leq_rm__Gambrel_WdShngl: - RoofMatl__WdShngl
   + Y__Gambrel__WdShngl <= 0
 7.7.4__y_ge_and__Gambrel_WdShngl: - RoofStyle__Gambrel - RoofMatl__WdShngl
   + Y__Gambrel__WdShngl >= -1
 7.7.5__z_leq_plan__Gambrel_WdShngl: - PlanRoofArea + Z__Gambrel__WdShngl
   <= 0
 7.7.5__z_leq_u_y__Gambrel_WdShngl: - 4200 Y__Gambrel__WdShngl
   + Z__Gambrel__WdShngl <= 0
 7.7.5__z_ge_plan_u__Gambrel_WdShngl: - PlanRoofArea
   - 4200 Y__Gambrel__WdShngl + Z__Gambrel__WdShngl >= -4200
 7.7.4__y_leq_rs__Hip_ClyTile: - RoofStyle__Hip + Y__Hip__ClyTile <= 0
 7.7.4__y_leq_rm__Hip_ClyTile: - RoofMatl__ClyTile + Y__Hip__ClyTile <= 0
 7.7.4__y_ge_and__Hip_ClyTile: - RoofStyle__Hip - RoofMatl__ClyTile
   + Y__Hip__ClyTile >= -1
 7.7.5__z_leq_plan__Hip_ClyTile: - PlanRoofArea + Z__Hip__ClyTile <= 0
 7.7.5__z_leq_u_y__Hip_ClyTile: - 4200 Y__Hip__ClyTile + Z__Hip__ClyTile
   <= 0
 7.7.5__z_ge_plan_u__Hip_ClyTile: - PlanRoofArea - 4200 Y__Hip__ClyTile
   + Z__Hip__ClyTile >= -4200
 7.7.4__y_leq_rs__Hip_CompShg: - RoofStyle__Hip + Y__Hip__CompShg <= 0
 7.7.4__y_leq_rm__Hip_CompShg: - RoofMatl__CompShg + Y__Hip__CompShg <= 0
 7.7.4__y_ge_and__Hip_CompShg: - RoofStyle__Hip - RoofMatl__CompShg
   + Y__Hip__CompShg >= -1
 7.7.5__z_leq_plan__Hip_CompShg: - PlanRoofArea + Z__Hip__CompShg <= 0
 7.7.5__z_leq_u_y__Hip_CompShg: - 4200 Y__Hip__CompShg + Z__Hip__CompShg
   <= 0
 7.7.5__z_ge_plan_u__Hip_CompShg: - PlanRoofArea - 4200 Y__Hip__CompShg
   + Z__Hip__CompShg >= -4200
 7.7.4__y_leq_rs__Hip_Membran: - RoofStyle__Hip + Y__Hip__Membran <= 0
 7.7.4__y_leq_rm__Hip_Membran: - RoofMatl__Membran + Y__Hip__Membran <= 0
 7.7.4__y_ge_and__Hip_Membran: - RoofStyle__Hip - RoofMatl__Membran
   + Y__Hip__Membran >= -1
 7.7.5__z_leq_plan__Hip_Membran: - PlanRoofArea + Z__Hip__Membran <= 0
 7.7.5__z_leq_u_y__Hip_Membran: - 4200 Y__Hip__Membran + Z__Hip__Membran
   <= 0
 7.7.5__z_ge_plan_u__Hip_Membran: - PlanRoofArea - 4200 Y__Hip__Membran
   + Z__Hip__Membran >= -4200
 7.7.4__y_leq_rs__Hip_Metal: - RoofStyle__Hip + Y__Hip__Metal <= 0
 7.7.4__y_leq_rm__Hip_Metal: - RoofMatl__Metal + Y__Hip__Metal <= 0
 7.7.4__y_ge_and__Hip_Metal: - RoofStyle__Hip - RoofMatl__Metal
   + Y__Hip__Metal >= -1
 7.7.5__z_leq_plan__Hip_Metal: - PlanRoofArea + Z__Hip__Metal <= 0
 7.7.5__z_leq_u_y__Hip_Metal: - 4200 Y__Hip__Metal + Z__Hip__Metal <= 0
 7.7.5__z_ge_plan_u__Hip_Metal: - PlanRoofArea - 4200 Y__Hip__Metal
   + Z__Hip__Metal >= -4200
 7.7.4__y_leq_rs__Hip_Roll: - RoofStyle__Hip + Y__Hip__Roll <= 0
 7.7.4__y_leq_rm__Hip_Roll: - RoofMatl__Roll + Y__Hip__Roll <= 0
 7.7.4__y_ge_and__Hip_Roll: - RoofStyle__Hip - RoofMatl__Roll
   + Y__Hip__Roll >= -1
 7.7.5__z_leq_plan__Hip_Roll: - PlanRoofArea + Z__Hip__Roll <= 0
 7.7.5__z_leq_u_y__Hip_Roll: - 4200 Y__Hip__Roll + Z__Hip__Roll <= 0
 7.7.5__z_ge_plan_u__Hip_Roll: - PlanRoofArea - 4200 Y__Hip__Roll
   + Z__Hip__Roll >= -4200
 7.7.4__y_leq_rs__Hip_Tar&Grv: - RoofStyle__Hip + Y__Hip__Tar&Grv <= 0
 7.7.4__y_leq_rm__Hip_Tar&Grv: - RoofMatl__Tar&Grv + Y__Hip__Tar&Grv <= 0
 7.7.4__y_ge_and__Hip_Tar&Grv: - RoofStyle__Hip - RoofMatl__Tar&Grv
   + Y__Hip__Tar&Grv >= -1
 7.7.5__z_leq_plan__Hip_Tar&Grv: - PlanRoofArea + Z__Hip__Tar&Grv <= 0
 7.7.5__z_leq_u_y__Hip_Tar&Grv: - 4200 Y__Hip__Tar&Grv + Z__Hip__Tar&Grv
   <= 0
 7.7.5__z_ge_plan_u__Hip_Tar&Grv: - PlanRoofArea - 4200 Y__Hip__Tar&Grv
   + Z__Hip__Tar&Grv >= -4200
 7.7.4__y_leq_rs__Hip_WdShake: - RoofStyle__Hip + Y__Hip__WdShake <= 0
 7.7.4__y_leq_rm__Hip_WdShake: - RoofMatl__WdShake + Y__Hip__WdShake <= 0
 7.7.4__y_ge_and__Hip_WdShake: - RoofStyle__Hip - RoofMatl__WdShake
   + Y__Hip__WdShake >= -1
 7.7.5__z_leq_plan__Hip_WdShake: - PlanRoofArea + Z__Hip__WdShake <= 0
 7.7.5__z_leq_u_y__Hip_WdShake: - 4200 Y__Hip__WdShake + Z__Hip__WdShake
   <= 0
 7.7.5__z_ge_plan_u__Hip_WdShake: - PlanRoofArea - 4200 Y__Hip__WdShake
   + Z__Hip__WdShake >= -4200
 7.7.4__y_leq_rs__Hip_WdShngl: - RoofStyle__Hip + Y__Hip__WdShngl <= 0
 7.7.4__y_leq_rm__Hip_WdShngl: - RoofMatl__WdShngl + Y__Hip__WdShngl <= 0
 7.7.4__y_ge_and__Hip_WdShngl: - RoofStyle__Hip - RoofMatl__WdShngl
   + Y__Hip__WdShngl >= -1
 7.7.5__z_leq_plan__Hip_WdShngl: - PlanRoofArea + Z__Hip__WdShngl <= 0
 7.7.5__z_leq_u_y__Hip_WdShngl: - 4200 Y__Hip__WdShngl + Z__Hip__WdShngl
   <= 0
 7.7.5__z_ge_plan_u__Hip_WdShngl: - PlanRoofArea - 4200 Y__Hip__WdShngl
   + Z__Hip__WdShngl >= -4200
 7.7.4__y_leq_rs__Mansard_ClyTile: - RoofStyle__Mansard
   + Y__Mansard__ClyTile <= 0
 7.7.4__y_leq_rm__Mansard_ClyTile: - RoofMatl__ClyTile
   + Y__Mansard__ClyTile <= 0
 7.7.4__y_ge_and__Mansard_ClyTile: - RoofStyle__Mansard - RoofMatl__ClyTile
   + Y__Mansard__ClyTile >= -1
 7.7.5__z_leq_plan__Mansard_ClyTile: - PlanRoofArea + Z__Mansard__ClyTile
   <= 0
 7.7.5__z_leq_u_y__Mansard_ClyTile: - 4200 Y__Mansard__ClyTile
   + Z__Mansard__ClyTile <= 0
 7.7.5__z_ge_plan_u__Mansard_ClyTile: - PlanRoofArea
   - 4200 Y__Mansard__ClyTile + Z__Mansard__ClyTile >= -4200
 7.7.4__y_leq_rs__Mansard_CompShg: - RoofStyle__Mansard
   + Y__Mansard__CompShg <= 0
 7.7.4__y_leq_rm__Mansard_CompShg: - RoofMatl__CompShg
   + Y__Mansard__CompShg <= 0
 7.7.4__y_ge_and__Mansard_CompShg: - RoofStyle__Mansard - RoofMatl__CompShg
   + Y__Mansard__CompShg >= -1
 7.7.5__z_leq_plan__Mansard_CompShg: - PlanRoofArea + Z__Mansard__CompShg
   <= 0
 7.7.5__z_leq_u_y__Mansard_CompShg: - 4200 Y__Mansard__CompShg
   + Z__Mansard__CompShg <= 0
 7.7.5__z_ge_plan_u__Mansard_CompShg: - PlanRoofArea
   - 4200 Y__Mansard__CompShg + Z__Mansard__CompShg >= -4200
 7.7.4__y_leq_rs__Mansard_Membran: - RoofStyle__Mansard
   + Y__Mansard__Membran <= 0
 7.7.4__y_leq_rm__Mansard_Membran: - RoofMatl__Membran
   + Y__Mansard__Membran <= 0
 7.7.4__y_ge_and__Mansard_Membran: - RoofStyle__Mansard - RoofMatl__Membran
   + Y__Mansard__Membran >= -1
 7.7.5__z_leq_plan__Mansard_Membran: - PlanRoofArea + Z__Mansard__Membran
   <= 0
 7.7.5__z_leq_u_y__Mansard_Membran: - 4200 Y__Mansard__Membran
   + Z__Mansard__Membran <= 0
 7.7.5__z_ge_plan_u__Mansard_Membran: - PlanRoofArea
   - 4200 Y__Mansard__Membran + Z__Mansard__Membran >= -4200
 7.7.4__y_leq_rs__Mansard_Metal: - RoofStyle__Mansard + Y__Mansard__Metal
   <= 0
 7.7.4__y_leq_rm__Mansard_Metal: - RoofMatl__Metal + Y__Mansard__Metal
   <= 0
 7.7.4__y_ge_and__Mansard_Metal: - RoofStyle__Mansard - RoofMatl__Metal
   + Y__Mansard__Metal >= -1
 7.7.5__z_leq_plan__Mansard_Metal: - PlanRoofArea + Z__Mansard__Metal <= 0
 7.7.5__z_leq_u_y__Mansard_Metal: - 4200 Y__Mansard__Metal
   + Z__Mansard__Metal <= 0
 7.7.5__z_ge_plan_u__Mansard_Metal: - PlanRoofArea - 4200 Y__Mansard__Metal
   + Z__Mansard__Metal >= -4200
 7.7.4__y_leq_rs__Mansard_Roll: - RoofStyle__Mansard + Y__Mansard__Roll
   <= 0
 7.7.4__y_leq_rm__Mansard_Roll: - RoofMatl__Roll + Y__Mansard__Roll <= 0
 7.7.4__y_ge_and__Mansard_Roll: - RoofStyle__Mansard - RoofMatl__Roll
   + Y__Mansard__Roll >= -1
 7.7.5__z_leq_plan__Mansard_Roll: - PlanRoofArea + Z__Mansard__Roll <= 0
 7.7.5__z_leq_u_y__Mansard_Roll: - 4200 Y__Mansard__Roll + Z__Mansard__Roll
   <= 0
 7.7.5__z_ge_plan_u__Mansard_Roll: - PlanRoofArea - 4200 Y__Mansard__Roll
   + Z__Mansard__Roll >= -4200
 7.7.4__y_leq_rs__Mansard_Tar&Grv: - RoofStyle__Mansard
   + Y__Mansard__Tar&Grv <= 0
 7.7.4__y_leq_rm__Mansard_Tar&Grv: - RoofMatl__Tar&Grv
   + Y__Mansard__Tar&Grv <= 0
 7.7.4__y_ge_and__Mansard_Tar&Grv: - RoofStyle__Mansard - RoofMatl__Tar&Grv
   + Y__Mansard__Tar&Grv >= -1
 7.7.5__z_leq_plan__Mansard_Tar&Grv: - PlanRoofArea + Z__Mansard__Tar&Grv
   <= 0
 7.7.5__z_leq_u_y__Mansard_Tar&Grv: - 4200 Y__Mansard__Tar&Grv
   + Z__Mansard__Tar&Grv <= 0
 7.7.5__z_ge_plan_u__Mansard_Tar&Grv: - PlanRoofArea
   - 4200 Y__Mansard__Tar&Grv + Z__Mansard__Tar&Grv >= -4200
 7.7.4__y_leq_rs__Mansard_WdShake: - RoofStyle__Mansard
   + Y__Mansard__WdShake <= 0
 7.7.4__y_leq_rm__Mansard_WdShake: - RoofMatl__WdShake
   + Y__Mansard__WdShake <= 0
 7.7.4__y_ge_and__Mansard_WdShake: - RoofStyle__Mansard - RoofMatl__WdShake
   + Y__Mansard__WdShake >= -1
 7.7.5__z_leq_plan__Mansard_WdShake: - PlanRoofArea + Z__Mansard__WdShake
   <= 0
 7.7.5__z_leq_u_y__Mansard_WdShake: - 4200 Y__Mansard__WdShake
   + Z__Mansard__WdShake <= 0
 7.7.5__z_ge_plan_u__Mansard_WdShake: - PlanRoofArea
   - 4200 Y__Mansard__WdShake + Z__Mansard__WdShake >= -4200
 7.7.4__y_leq_rs__Mansard_WdShngl: - RoofStyle__Mansard
   + Y__Mansard__WdShngl <= 0
 7.7.4__y_leq_rm__Mansard_WdShngl: - RoofMatl__WdShngl
   + Y__Mansard__WdShngl <= 0
 7.7.4__y_ge_and__Mansard_WdShngl: - RoofStyle__Mansard - RoofMatl__WdShngl
   + Y__Mansard__WdShngl >= -1
 7.7.5__z_leq_plan__Mansard_WdShngl: - PlanRoofArea + Z__Mansard__WdShngl
   <= 0
 7.7.5__z_leq_u_y__Mansard_WdShngl: - 4200 Y__Mansard__WdShngl
   + Z__Mansard__WdShngl <= 0
 7.7.5__z_ge_plan_u__Mansard_WdShngl: - PlanRoofArea
   - 4200 Y__Mansard__WdShngl + Z__Mansard__WdShngl >= -4200
 7.7.4__y_leq_rs__Shed_ClyTile: - RoofStyle__Shed + Y__Shed__ClyTile <= 0
 7.7.4__y_leq_rm__Shed_ClyTile: - RoofMatl__ClyTile + Y__Shed__ClyTile
   <= 0
 7.7.4__y_ge_and__Shed_ClyTile: - RoofStyle__Shed - RoofMatl__ClyTile
   + Y__Shed__ClyTile >= -1
 7.7.5__z_leq_plan__Shed_ClyTile: - PlanRoofArea + Z__Shed__ClyTile <= 0
 7.7.5__z_leq_u_y__Shed_ClyTile: - 4200 Y__Shed__ClyTile + Z__Shed__ClyTile
   <= 0
 7.7.5__z_ge_plan_u__Shed_ClyTile: - PlanRoofArea - 4200 Y__Shed__ClyTile
   + Z__Shed__ClyTile >= -4200
 7.7.4__y_leq_rs__Shed_CompShg: - RoofStyle__Shed + Y__Shed__CompShg <= 0
 7.7.4__y_leq_rm__Shed_CompShg: - RoofMatl__CompShg + Y__Shed__CompShg
   <= 0
 7.7.4__y_ge_and__Shed_CompShg: - RoofStyle__Shed - RoofMatl__CompShg
   + Y__Shed__CompShg >= -1
 7.7.5__z_leq_plan__Shed_CompShg: - PlanRoofArea + Z__Shed__CompShg <= 0
 7.7.5__z_leq_u_y__Shed_CompShg: - 4200 Y__Shed__CompShg + Z__Shed__CompShg
   <= 0
 7.7.5__z_ge_plan_u__Shed_CompShg: - PlanRoofArea - 4200 Y__Shed__CompShg
   + Z__Shed__CompShg >= -4200
 7.7.4__y_leq_rs__Shed_Membran: - RoofStyle__Shed + Y__Shed__Membran <= 0
 7.7.4__y_leq_rm__Shed_Membran: - RoofMatl__Membran + Y__Shed__Membran
   <= 0
 7.7.4__y_ge_and__Shed_Membran: - RoofStyle__Shed - RoofMatl__Membran
   + Y__Shed__Membran >= -1
 7.7.5__z_leq_plan__Shed_Membran: - PlanRoofArea + Z__Shed__Membran <= 0
 7.7.5__z_leq_u_y__Shed_Membran: - 4200 Y__Shed__Membran + Z__Shed__Membran
   <= 0
 7.7.5__z_ge_plan_u__Shed_Membran: - PlanRoofArea - 4200 Y__Shed__Membran
   + Z__Shed__Membran >= -4200
 7.7.4__y_leq_rs__Shed_Metal: - RoofStyle__Shed + Y__Shed__Metal <= 0
 7.7.4__y_leq_rm__Shed_Metal: - RoofMatl__Metal + Y__Shed__Metal <= 0
 7.7.4__y_ge_and__Shed_Metal: - RoofStyle__Shed - RoofMatl__Metal
   + Y__Shed__Metal >= -1
 7.7.5__z_leq_plan__Shed_Metal: - PlanRoofArea + Z__Shed__Metal <= 0
 7.7.5__z_leq_u_y__Shed_Metal: - 4200 Y__Shed__Metal + Z__Shed__Metal <= 0
 7.7.5__z_ge_plan_u__Shed_Metal: - PlanRoofArea - 4200 Y__Shed__Metal
   + Z__Shed__Metal >= -4200
 7.7.4__y_leq_rs__Shed_Roll: - RoofStyle__Shed + Y__Shed__Roll <= 0
 7.7.4__y_leq_rm__Shed_Roll: - RoofMatl__Roll + Y__Shed__Roll <= 0
 7.7.4__y_ge_and__Shed_Roll: - RoofStyle__Shed - RoofMatl__Roll
   + Y__Shed__Roll >= -1
 7.7.5__z_leq_plan__Shed_Roll: - PlanRoofArea + Z__Shed__Roll <= 0
 7.7.5__z_leq_u_y__Shed_Roll: - 4200 Y__Shed__Roll + Z__Shed__Roll <= 0
 7.7.5__z_ge_plan_u__Shed_Roll: - PlanRoofArea - 4200 Y__Shed__Roll
   + Z__Shed__Roll >= -4200
 7.7.4__y_leq_rs__Shed_Tar&Grv: - RoofStyle__Shed + Y__Shed__Tar&Grv <= 0
 7.7.4__y_leq_rm__Shed_Tar&Grv: - RoofMatl__Tar&Grv + Y__Shed__Tar&Grv
   <= 0
 7.7.4__y_ge_and__Shed_Tar&Grv: - RoofStyle__Shed - RoofMatl__Tar&Grv
   + Y__Shed__Tar&Grv >= -1
 7.7.5__z_leq_plan__Shed_Tar&Grv: - PlanRoofArea + Z__Shed__Tar&Grv <= 0
 7.7.5__z_leq_u_y__Shed_Tar&Grv: - 4200 Y__Shed__Tar&Grv + Z__Shed__Tar&Grv
   <= 0
 7.7.5__z_ge_plan_u__Shed_Tar&Grv: - PlanRoofArea - 4200 Y__Shed__Tar&Grv
   + Z__Shed__Tar&Grv >= -4200
 7.7.4__y_leq_rs__Shed_WdShake: - RoofStyle__Shed + Y__Shed__WdShake <= 0
 7.7.4__y_leq_rm__Shed_WdShake: - RoofMatl__WdShake + Y__Shed__WdShake
   <= 0
 7.7.4__y_ge_and__Shed_WdShake: - RoofStyle__Shed - RoofMatl__WdShake
   + Y__Shed__WdShake >= -1
 7.7.5__z_leq_plan__Shed_WdShake: - PlanRoofArea + Z__Shed__WdShake <= 0
 7.7.5__z_leq_u_y__Shed_WdShake: - 4200 Y__Shed__WdShake + Z__Shed__WdShake
   <= 0
 7.7.5__z_ge_plan_u__Shed_WdShake: - PlanRoofArea - 4200 Y__Shed__WdShake
   + Z__Shed__WdShake >= -4200
 7.7.4__y_leq_rs__Shed_WdShngl: - RoofStyle__Shed + Y__Shed__WdShngl <= 0
 7.7.4__y_leq_rm__Shed_WdShngl: - RoofMatl__WdShngl + Y__Shed__WdShngl
   <= 0
 7.7.4__y_ge_and__Shed_WdShngl: - RoofStyle__Shed - RoofMatl__WdShngl
   + Y__Shed__WdShngl >= -1
 7.7.5__z_leq_plan__Shed_WdShngl: - PlanRoofArea + Z__Shed__WdShngl <= 0
 7.7.5__z_leq_u_y__Shed_WdShngl: - 4200 Y__Shed__WdShngl + Z__Shed__WdShngl
   <= 0
 7.7.5__z_ge_plan_u__Shed_WdShngl: - PlanRoofArea - 4200 Y__Shed__WdShngl
   + Z__Shed__WdShngl >= -4200
 7.7.4__one_y: Y__Flat__ClyTile + Y__Flat__CompShg + Y__Flat__Membran
   + Y__Flat__Metal + Y__Flat__Roll + Y__Flat__Tar&Grv + Y__Flat__WdShake
   + Y__Flat__WdShngl + Y__Gable__ClyTile + Y__Gable__CompShg
   + Y__Gable__Membran + Y__Gable__Metal + Y__Gable__Roll
   + Y__Gable__Tar&Grv + Y__Gable__WdShake + Y__Gable__WdShngl
   + Y__Gambrel__ClyTile + Y__Gambrel__CompShg + Y__Gambrel__Membran
   + Y__Gambrel__Metal + Y__Gambrel__Roll + Y__Gambrel__Tar&Grv
   + Y__Gambrel__WdShake + Y__Gambrel__WdShngl + Y__Hip__ClyTile
   + Y__Hip__CompShg + Y__Hip__Membran + Y__Hip__Metal + Y__Hip__Roll
   + Y__Hip__Tar&Grv + Y__Hip__WdShake + Y__Hip__WdShngl
   + Y__Mansard__ClyTile + Y__Mansard__CompShg + Y__Mansard__Membran
   + Y__Mansard__Metal + Y__Mansard__Roll + Y__Mansard__Tar&Grv
   + Y__Mansard__WdShake + Y__Mansard__WdShngl + Y__Shed__ClyTile
   + Y__Shed__CompShg + Y__Shed__Membran + Y__Shed__Metal + Y__Shed__Roll
   + Y__Shed__Tar&Grv + Y__Shed__WdShake + Y__Shed__WdShngl = 1
 7.7.3__actual_roof_area: - 1.1 Z__Flat__ClyTile - 1.1 Z__Flat__CompShg
   - 1.1 Z__Flat__Membran - 1.1 Z__Flat__Metal - 1.1 Z__Flat__Roll
   - 1.1 Z__Flat__Tar&Grv - 1.1 Z__Flat__WdShake - 1.1 Z__Flat__WdShngl
   - 1.1 Z__Gable__ClyTile - 1.1 Z__Gable__CompShg - 1.1 Z__Gable__Membran
   - 1.1 Z__Gable__Metal - 1.1 Z__Gable__Roll - 1.1 Z__Gable__Tar&Grv
   - 1.1 Z__Gable__WdShake - 1.1 Z__Gable__WdShngl
   - 1.1 Z__Gambrel__ClyTile - 1.1 Z__Gambrel__CompShg
   - 1.1 Z__Gambrel__Membran - 1.1 Z__Gambrel__Metal - 1.1 Z__Gambrel__Roll
   - 1.1 Z__Gambrel__Tar&Grv - 1.1 Z__Gambrel__WdShake
   - 1.1 Z__Gambrel__WdShngl - 1.1 Z__Hip__ClyTile - 1.1 Z__Hip__CompShg
   - 1.1 Z__Hip__Membran - 1.1 Z__Hip__Metal - 1.1 Z__Hip__Roll
   - 1.1 Z__Hip__Tar&Grv - 1.1 Z__Hip__WdShake - 1.1 Z__Hip__WdShngl
   - 1.1 Z__Mansard__ClyTile - 1.1 Z__Mansard__CompShg
   - 1.1 Z__Mansard__Membran - 1.1 Z__Mansard__Metal - 1.1 Z__Mansard__Roll
   - 1.1 Z__Mansard__Tar&Grv - 1.1 Z__Mansard__WdShake
   - 1.1 Z__Mansard__WdShngl - 1.1 Z__Shed__ClyTile - 1.1 Z__Shed__CompShg
   - 1.1 Z__Shed__Membran - 1.1 Z__Shed__Metal - 1.1 Z__Shed__Roll
   - 1.1 Z__Shed__Tar&Grv - 1.1 Z__Shed__WdShake - 1.1 Z__Shed__WdShngl
   + ActualRoofArea = 0
 7.8__total_area: - x_Total_Bsmt_SF - x_1st_Flr_SF - x_2nd_Flr_SF
   + TotalArea = 0
 7.9__cap_1st: x_1st_Flr_SF <= 4200
 7.9__cap_2nd: x_2nd_Flr_SF <= 3500
 7.9__cap_bsmt: x_Total_Bsmt_SF <= 3500
 11.10__GrLiv_leq_0.8lot: x_Gr_Liv_Area <= 5600
 7.9__cap_garage: x_Garage_Area <= 1400
 7.10__baths_vs_bed: 3 FullBath - 2 Bedrooms <= 0
 11.xx__fullbath_min1: FullBath >= 1
 11.xx__bedrooms_min1: Bedrooms >= 1
 7.11.1__pool_max: x_Pool_Area - 700 HasPool <= 0
 7.11.1__pool_min: x_Pool_Area - 160 HasPool >= 0
 7.11.1__pool_nonneg: x_Pool_Area >= 0
 onehot__pqc: - HasPool + zPQC_TA + zPQC_Gd + zPQC_Ex = 0
 part__apool_sum: - x_Pool_Area + APool_TA + APool_Gd + APool_Ex = 0
 part__apool_ta: - 700 zPQC_TA + APool_TA <= 0
 part__apool_gd: - 700 zPQC_Gd + APool_Gd <= 0
 part__apool_ex: - 700 zPQC_Ex + APool_Ex <= 0
 7.12.1__porch_sum: - x_Open_Porch_SF - x_Enclosed_Porch - x_3Ssn_Porch
   - x_Screen_Porch + TotalPorchSF = 0
 7.12.1__porch_cap_total: TotalPorchSF <= 1750
 7.12.1__porch_leq_1st: - x_1st_Flr_SF + TotalPorchSF <= 0
 7.12.2__open_min: x_Open_Porch_SF - 40 HasOpenPorch >= 0
 7.12.2__open_max: x_Open_Porch_SF - 700 HasOpenPorch <= 0
 7.12.2__encl_min: x_Enclosed_Porch - 60 HasEnclosedPorch >= 0
 7.12.2__encl_max: x_Enclosed_Porch - 700 HasEnclosedPorch <= 0
 7.12.2__screen_min: x_Screen_Porch - 40 HasScreenPorch >= 0
 7.12.2__screen_max: x_Screen_Porch - 350 HasScreenPorch <= 0
 7.12.2__3ssn_min: x_3Ssn_Porch - 80 Has3SsnPorch >= 0
 7.12.2__3ssn_max: x_3Ssn_Porch - 700 Has3SsnPorch <= 0
 7.12.3__ext_cap: x_Wood_Deck_SF + x_Pool_Area + TotalPorchSF <= 2450
 7.12.3__deck_open_cap: x_Wood_Deck_SF + x_Open_Porch_SF <= 1400
 7.13__deck_min: x_Wood_Deck_SF - 40 HasWoodDeck >= 0
 7.13__deck_max: x_Wood_Deck_SF - 1050 HasWoodDeck <= 0
 ex__BsmtExposure: BsmtExposure__Gd + BsmtExposure__Av + BsmtExposure__Mn
   + BsmtExposure__No + BsmtExposure__NA = 1
 link__bsmtexposure__gd: x_Bsmt_Exposure_Gd - BsmtExposure__Gd = 0
 link__bsmtexposure__av: x_Bsmt_Exposure_Av - BsmtExposure__Av = 0
 link__bsmtexposure__mn: x_Bsmt_Exposure_Mn - BsmtExposure__Mn = 0
 link__bsmtexposure__no: x_Bsmt_Exposure_No - BsmtExposure__No = 0
 link__bsmtexposure__na: x_Bsmt_Exposure_No_aplica - BsmtExposure__NA = 0
 ex__BsmtFinType1: BsmtFinType1__GLQ + BsmtFinType1__ALQ
   + BsmtFinType1__BLQ + BsmtFinType1__Rec + BsmtFinType1__LwQ
   + BsmtFinType1__Unf + BsmtFinType1__NA = 1
 ex__BsmtFinType2: BsmtFinType2__GLQ + BsmtFinType2__ALQ
   + BsmtFinType2__BLQ + BsmtFinType2__Rec + BsmtFinType2__LwQ
   + BsmtFinType2__Unf + BsmtFinType2__NA = 1
 link__bsmtfintype1__glq: x_BsmtFin_Type_1_GLQ - BsmtFinType1__GLQ = 0
 link__bsmtfintype1__alq: x_BsmtFin_Type_1_ALQ - BsmtFinType1__ALQ = 0
 link__bsmtfintype1__blq: x_BsmtFin_Type_1_BLQ - BsmtFinType1__BLQ = 0
 link__bsmtfintype1__rec: x_BsmtFin_Type_1_Rec - BsmtFinType1__Rec = 0
 link__bsmtfintype1__lwq: x_BsmtFin_Type_1_LwQ - BsmtFinType1__LwQ = 0
 link__bsmtfintype1__unf: x_BsmtFin_Type_1_Unf - BsmtFinType1__Unf = 0
 link__bsmtfintype1__na: x_BsmtFin_Type_1_No_aplica - BsmtFinType1__NA = 0
 link__bsmtfintype2__glq: x_BsmtFin_Type_2_GLQ - BsmtFinType2__GLQ = 0
 link__bsmtfintype2__alq: x_BsmtFin_Type_2_ALQ - BsmtFinType2__ALQ = 0
 link__bsmtfintype2__blq: x_BsmtFin_Type_2_BLQ - BsmtFinType2__BLQ = 0
 link__bsmtfintype2__rec: x_BsmtFin_Type_2_Rec - BsmtFinType2__Rec = 0
 link__bsmtfintype2__lwq: x_BsmtFin_Type_2_LwQ - BsmtFinType2__LwQ = 0
 link__bsmtfintype2__unf: x_BsmtFin_Type_2_Unf - BsmtFinType2__Unf = 0
 link__bsmtfintype2__na: x_BsmtFin_Type_2_No_aplica - BsmtFinType2__NA = 0
 7.0__bsmt_unf_zero: x_Bsmt_Unf_SF = 0
 link__bsmtunfsf: = 0
 link__bsmtfinsf1: x_BsmtFin_SF_1 - BsmtFinSF1 = 0
 link__bsmtfinsf2: x_BsmtFin_SF_2 - BsmtFinSF2 = 0
 7.8__bsmt_parts_sum: - x_Bsmt_Unf_SF + x_Total_Bsmt_SF - BsmtFinSF1
   - BsmtFinSF2 = 0
 link__totalbsmtsf: = 0
 7.18__bsmt_cap_if_not_na: x_Total_Bsmt_SF + 3500 BsmtExposure__NA <= 3500
 7.18__sf1_cap_by_type: - 3500 BsmtFinType1__GLQ - 3500 BsmtFinType1__ALQ
   - 3500 BsmtFinType1__BLQ - 3500 BsmtFinType1__Rec
   - 3500 BsmtFinType1__LwQ - 3500 BsmtFinType1__Unf + BsmtFinSF1 <= 0
 7.18__sf2_cap_by_type: - 3500 BsmtFinType2__GLQ - 3500 BsmtFinType2__ALQ
   - 3500 BsmtFinType2__BLQ - 3500 BsmtFinType2__Rec
   - 3500 BsmtFinType2__LwQ - 3500 BsmtFinType2__Unf + BsmtFinSF2 <= 0
 7.18__sf1_min_if_real: - 100 BsmtFinType1__GLQ - 100 BsmtFinType1__ALQ
   - 100 BsmtFinType1__BLQ - 100 BsmtFinType1__Rec - 100 BsmtFinType1__LwQ
   + BsmtFinSF1 >= 0
 7.18__sf2_min_if_real: - 100 BsmtFinType2__GLQ - 100 BsmtFinType2__ALQ
   - 100 BsmtFinType2__BLQ - 100 BsmtFinType2__Rec - 100 BsmtFinType2__LwQ
   + BsmtFinSF2 >= 0
 7.18__bsmt_fullbath_if_real: - 2 BsmtFinType1__GLQ - 2 BsmtFinType1__ALQ
   - 2 BsmtFinType1__BLQ - 2 BsmtFinType1__Rec - 2 BsmtFinType1__LwQ
   - 2 BsmtFinType2__GLQ - 2 BsmtFinType2__ALQ - 2 BsmtFinType2__BLQ
   - 2 BsmtFinType2__Rec - 2 BsmtFinType2__LwQ + BsmtFullBath <= 0
 7.18__bsmt_halfbath_if_real: - BsmtFinType1__GLQ - BsmtFinType1__ALQ
   - BsmtFinType1__BLQ - BsmtFinType1__Rec - BsmtFinType1__LwQ
   - BsmtFinType2__GLQ - BsmtFinType2__ALQ - BsmtFinType2__BLQ
   - BsmtFinType2__Rec - BsmtFinType2__LwQ + BsmtHalfBath <= 0
 7.18__sf1_off_if_na: 3500 BsmtExposure__NA + BsmtFinSF1 <= 3500
 7.18__sf2_off_if_na: 3500 BsmtExposure__NA + BsmtFinSF2 <= 3500
 7.18__bf_off_if_na: 2 BsmtExposure__NA + BsmtFullBath <= 2
 7.18__bh_off_if_na: BsmtExposure__NA + BsmtHalfBath <= 1
 link__bsmtfullbath: x_Bsmt_Full_Bath - BsmtFullBath = 0
 link__bsmthalfbath: x_Bsmt_Half_Bath - BsmtHalfBath = 0
 7.22.1__totrms_def: x_TotRms_AbvGrd - OtherRooms - FullBath - HalfBath
   - Bedrooms = 0
 link__totrmsabvgrd: = 0
 11.20__min_area_full_bath: AreaFullBath - 40 FullBath >= 0
 11.20__min_area_half_bath: AreaHalfBath - 20 HalfBath >= 0
 11.20__min_area_kitchen: AreaKitchen - 75 Kitchen >= 0
 11.20__min_area_bedroom: AreaBedroom - 70 Bedrooms >= 0
 7.xx__areas_leq_floors: - x_1st_Flr_SF - x_2nd_Flr_SF + AreaOther1
   + AreaOther2 + AreaKitchen + AreaFullBath + AreaHalfBath + AreaBedroom
   <= 0
 ex__Exterior1st: Exterior1st__VinylSd + Exterior1st__MetalSd
   + Exterior1st__Wd_Sdng + Exterior1st__HdBoard + Exterior1st__Stucco
   + Exterior1st__Plywood + Exterior1st__CemntBd + Exterior1st__BrkFace
   + Exterior1st__BrkComm + Exterior1st__WdShngl + Exterior1st__AsbShng
   + Exterior1st__Stone + Exterior1st__ImStucc + Exterior1st__AsphShn
   + Exterior1st__CBlock = 1
 ex__Exterior2nd: Exterior2nd__VinylSd + Exterior2nd__MetalSd
   + Exterior2nd__Wd_Sdng + Exterior2nd__HdBoard + Exterior2nd__Stucco
   + Exterior2nd__Plywood + Exterior2nd__CemntBd + Exterior2nd__BrkFace
   + Exterior2nd__BrkComm + Exterior2nd__WdShngl + Exterior2nd__AsbShng
   + Exterior2nd__Stone + Exterior2nd__ImStucc + Exterior2nd__AsphShn
   + Exterior2nd__CBlock = 1
 link__exterior1st__vinylsd: x_Exterior_1st_VinylSd - Exterior1st__VinylSd
   = 0
 link__exterior1st__metalsd: x_Exterior_1st_MetalSd - Exterior1st__MetalSd
   = 0
 link__exterior1st__wdsdng: x_Exterior_1st_Wd_Sdng - Exterior1st__Wd_Sdng
   = 0
 link__exterior1st__hdboard: x_Exterior_1st_HdBoard - Exterior1st__HdBoard
   = 0
 link__exterior1st__stucco: x_Exterior_1st_Stucco - Exterior1st__Stucco
   = 0
 link__exterior1st__plywood: x_Exterior_1st_Plywood - Exterior1st__Plywood
   = 0
 link__exterior1st__cemntbd: x_Exterior_1st_CemntBd - Exterior1st__CemntBd
   = 0
 link__exterior1st__brkface: x_Exterior_1st_BrkFace - Exterior1st__BrkFace
   = 0
 link__exterior1st__brkcomm: x_Exterior_1st_BrkComm - Exterior1st__BrkComm
   = 0
 link__exterior1st__asbshng: x_Exterior_1st_AsbShng - Exterior1st__AsbShng
   = 0
 link__exterior1st__stone: x_Exterior_1st_Stone - Exterior1st__Stone = 0
 link__exterior1st__imstucc: x_Exterior_1st_ImStucc - Exterior1st__ImStucc
   = 0
 link__exterior1st__asphshn: x_Exterior_1st_AsphShn - Exterior1st__AsphShn
   = 0
 link__exterior1st__cblock: x_Exterior_1st_CBlock - Exterior1st__CBlock
   = 0
 link__exterior2nd__vinylsd: x_Exterior_2nd_VinylSd - Exterior2nd__VinylSd
   = 0
 link__exterior2nd__metalsd: x_Exterior_2nd_MetalSd - Exterior2nd__MetalSd
   = 0
 link__exterior2nd__wdsdng: x_Exterior_2nd_Wd_Sdng - Exterior2nd__Wd_Sdng
   = 0
 link__exterior2nd__hdboard: x_Exterior_2nd_HdBoard - Exterior2nd__HdBoard
   = 0
 link__exterior2nd__stucco: x_Exterior_2nd_Stucco - Exterior2nd__Stucco
   = 0
 link__exterior2nd__plywood: x_Exterior_2nd_Plywood - Exterior2nd__Plywood
   = 0
 link__exterior2nd__brkface: x_Exterior_2nd_BrkFace - Exterior2nd__BrkFace
   = 0
 link__exterior2nd__asbshng: x_Exterior_2nd_AsbShng - Exterior2nd__AsbShng
   = 0
 link__exterior2nd__stone: x_Exterior_2nd_Stone - Exterior2nd__Stone = 0
 link__exterior2nd__imstucc: x_Exterior_2nd_ImStucc - Exterior2nd__ImStucc
   = 0
 link__exterior2nd__asphshn: x_Exterior_2nd_AsphShn - Exterior2nd__AsphShn
   = 0
 link__exterior2nd__cblock: x_Exterior_2nd_CBlock - Exterior2nd__CBlock
   = 0
 7.15__same_mat__VinylSd: SameMaterial - Exterior1st__VinylSd
   - Exterior2nd__VinylSd >= -1
 7.15__same_mat__MetalSd: SameMaterial - Exterior1st__MetalSd
   - Exterior2nd__MetalSd >= -1
 7.15__same_mat__Wd_Sdng: SameMaterial - Exterior1st__Wd_Sdng
   - Exterior2nd__Wd_Sdng >= -1
 7.15__same_mat__HdBoard: SameMaterial - Exterior1st__HdBoard
   - Exterior2nd__HdBoard >= -1
 7.15__same_mat__Stucco: SameMaterial - Exterior1st__Stucco
   - Exterior2nd__Stucco >= -1
 7.15__same_mat__Plywood: SameMaterial - Exterior1st__Plywood
   - Exterior2nd__Plywood >= -1
 7.15__same_mat__CemntBd: SameMaterial - Exterior1st__CemntBd
   - Exterior2nd__CemntBd >= -1
 7.15__same_mat__BrkFace: SameMaterial - Exterior1st__BrkFace
   - Exterior2nd__BrkFace >= -1
 7.15__same_mat__BrkComm: SameMaterial - Exterior1st__BrkComm
   - Exterior2nd__BrkComm >= -1
 7.15__same_mat__WdShngl: SameMaterial - Exterior1st__WdShngl
   - Exterior2nd__WdShngl >= -1
 7.15__same_mat__AsbShng: SameMaterial - Exterior1st__AsbShng
   - Exterior2nd__AsbShng >= -1
 7.15__same_mat__Stone: SameMaterial - Exterior1st__Stone
   - Exterior2nd__Stone >= -1
 7.15__same_mat__ImStucc: SameMaterial - Exterior1st__ImStucc
   - Exterior2nd__ImStucc >= -1
 7.15__same_mat__AsphShn: SameMaterial - Exterior1st__AsphShn
   - Exterior2nd__AsphShn >= -1
 7.15__same_mat__CBlock: SameMaterial - Exterior1st__CBlock
   - Exterior2nd__CBlock >= -1
 ex__MasVnrType: MasVnrType__BrkCmn + MasVnrType__BrkFace
   + MasVnrType__CBlock + MasVnrType__None + MasVnrType__Stone = 1
 link__masvnrtype__brkcmn: x_Mas_Vnr_Type_BrkCmn - MasVnrType__BrkCmn = 0
 link__masvnrtype__brkface: x_Mas_Vnr_Type_BrkFace - MasVnrType__BrkFace
   = 0
 link__masvnrtype__cblock: x_Mas_Vnr_Type_CBlock - MasVnrType__CBlock = 0
 link__masvnrtype__none: x_Mas_Vnr_Type_No_aplica - MasVnrType__None = 0
 link__masvnrtype__stone: x_Mas_Vnr_Type_Stone - MasVnrType__Stone = 0
 7.16.1__mas_cap: x_Mas_Vnr_Area - 0.4 AreaExterior1st <= 0
 7.16__mas_min_if_used: x_Mas_Vnr_Area + 20 MasVnrType__None >= 20
 7.16__mv_leq_area__BrkCmn: - x_Mas_Vnr_Area + MvProd__BrkCmn <= 0
 7.16__mv_leq_area__BrkFace: - x_Mas_Vnr_Area + MvProd__BrkFace <= 0
 7.16__mv_leq_area__CBlock: - x_Mas_Vnr_Area + MvProd__CBlock <= 0
 7.16__mv_leq_area__None: - x_Mas_Vnr_Area + MvProd__None <= 0
 7.16__mv_leq_area__Stone: - x_Mas_Vnr_Area + MvProd__Stone <= 0
 7.16__mas_leq_total_area: x_Mas_Vnr_Area - TotalArea <= 0
 ex__Foundation: Foundation__BrkTil + Foundation__CBlock
   + Foundation__PConc + Foundation__Slab + Foundation__Stone
   + Foundation__Wood = 1
 link__foundation__brktil: x_Foundation_BrkTil - Foundation__BrkTil = 0
 link__foundation__cblock: x_Foundation_CBlock - Foundation__CBlock = 0
 link__foundation__pconc: x_Foundation_PConc - Foundation__PConc = 0
 link__foundation__slab: x_Foundation_Slab - Foundation__Slab = 0
 link__foundation__stone: x_Foundation_Stone - Foundation__Stone = 0
 link__foundation__wood: x_Foundation_Wood - Foundation__Wood = 0
 7.20__AreaFoundation_eq_1st: - x_1st_Flr_SF + AreaFoundation = 0
 7.18__bsmt_leq_found_base: x_Total_Bsmt_SF - AreaFoundation <= 0
 7.20__fa_leq_area__BrkTil: - AreaFoundation + FA__BrkTil <= 0
 7.20__fa_leq_u_f__BrkTil: - 4200 Foundation__BrkTil + FA__BrkTil <= 0
 7.20__fa_ge_lin__BrkTil: - 4200 Foundation__BrkTil - AreaFoundation
   + FA__BrkTil >= -4200
 7.20__fa_leq_area__CBlock: - AreaFoundation + FA__CBlock <= 0
 7.20__fa_leq_u_f__CBlock: - 4200 Foundation__CBlock + FA__CBlock <= 0
 7.20__fa_ge_lin__CBlock: - 4200 Foundation__CBlock - AreaFoundation
   + FA__CBlock >= -4200
 7.20__fa_leq_area__PConc: - AreaFoundation + FA__PConc <= 0
 7.20__fa_leq_u_f__PConc: - 4200 Foundation__PConc + FA__PConc <= 0
 7.20__fa_ge_lin__PConc: - 4200 Foundation__PConc - AreaFoundation
   + FA__PConc >= -4200
 7.20__fa_leq_area__Slab: - AreaFoundation + FA__Slab <= 0
 7.20__fa_leq_u_f__Slab: - 4200 Foundation__Slab + FA__Slab <= 0
 7.20__fa_ge_lin__Slab: - 4200 Foundation__Slab - AreaFoundation + FA__Slab
   >= -4200
 7.20__fa_leq_area__Stone: - AreaFoundation + FA__Stone <= 0
 7.20__fa_leq_u_f__Stone: - 4200 Foundation__Stone + FA__Stone <= 0
 7.20__fa_ge_lin__Stone: - 4200 Foundation__Stone - AreaFoundation
   + FA__Stone >= -4200
 7.20__fa_leq_area__Wood: - AreaFoundation + FA__Wood <= 0
 7.20__fa_leq_u_f__Wood: - 4200 Foundation__Wood + FA__Wood <= 0
 7.20__fa_ge_lin__Wood: - 4200 Foundation__Wood - AreaFoundation + FA__Wood
   >= -4200
 7.20__slab_implies_na: - BsmtExposure__NA + Foundation__Slab <= 0
 7.20__wood_implies_na: - BsmtExposure__NA + Foundation__Wood <= 0
 7.20__na_implies_slab_or_wood: BsmtExposure__NA - Foundation__Slab
   - Foundation__Wood <= 0
 7.21.1__abed1_cap: - 3500 Floor1 + AreaBedroom1 <= 0
 7.21.1__afb1_cap: - 200 Floor1 + AreaFullBath1 <= 0
 7.21.1__ahb1_cap: - 80 Floor1 + AreaHalfBath1 <= 0
 7.21.1__ak1_cap: - 300 Floor1 + AreaKitchen1 <= 0
 7.21.1__aoth1_cap: - 3500 Floor1 + AreaOther1 <= 0
 7.21.2__abed2_cap: - 200 Floor2 + AreaBedroom2 <= 0
 7.21.2__afb2_cap: - 60 Floor2 + AreaFullBath2 <= 0
 7.21.2__ahb2_cap: - 20 Floor2 + AreaHalfBath2 <= 0
 7.21.2__ak2_cap: - 200 Floor2 + AreaKitchen2 <= 0
 7.21.2__aoth2_cap: - 3500 Floor2 + AreaOther2 <= 0
 7.21.1__ak1_cap_by_cnt: - 350 Kitchen1 + AreaKitchen1 <= 0
 7.21.2__ak2_cap_by_cnt: - 275 Kitchen2 + AreaKitchen2 <= 0
 7.21.1__afb1_cap_by_cnt: - 140 FullBath1 + AreaFullBath1 <= 0
 7.21.2__afb2_cap_by_cnt: - 110 FullBath2 + AreaFullBath2 <= 0
 7.21.1__ahb1_cap_by_cnt: - 80 HalfBath1 + AreaHalfBath1 <= 0
 7.21.2__ahb2_cap_by_cnt: - 60 HalfBath2 + AreaHalfBath2 <= 0
 7.21.1__abed1_cap_by_cnt: - 450 Bedroom1 + AreaBedroom1 <= 0
 7.21.2__abed2_cap_by_cnt: - 350 Bedroom2 + AreaBedroom2 <= 0
 7.21.1__partition_1st: - x_1st_Flr_SF + AreaFullBath1 + AreaHalfBath1
   + AreaKitchen1 + AreaBedroom1 + AreaOther1 + Remainder1 = 0
 7.21.2__partition_2nd: - x_2nd_Flr_SF + AreaFullBath2 + AreaHalfBath2
   + AreaKitchen2 + AreaBedroom2 + AreaOther2 + Remainder2 = 0
 def__total_ag: - x_1st_Flr_SF - x_2nd_Flr_SF + TotalAboveGradeSF = 0
 share__total_ag_scale: TotalAboveGradeSF - 7000 TotalAboveGradeShare = 0
 share__total_ag_ub: TotalAboveGradeShare <= 1.1
 def__remainder_sum: - Remainder1 - Remainder2 + RemainderSum = 0
 share__remainder_scale: RemainderSum - 7000 RemainderShare = 0
 share__remainder_ge: - 0.15 TotalAboveGradeShare + RemainderShare >= 0
 share__remainder_le: - 0.25 TotalAboveGradeShare + RemainderShare <= 0
 share__other_scale: AreaOther - 7000 AreaOtherShare = 0
 share__other_ge: - 0.2 TotalAboveGradeShare + AreaOtherShare >= 0
 share__other_le: - 0.3 TotalAboveGradeShare + AreaOtherShare <= 0
 7.22.1__other_cnt_sum: - OtherRooms1 - OtherRooms2 + OtherRooms = 0
 7.22.4__other1_min: AreaOther1 - 100 OtherRooms1 >= 0
 7.22.4__other2_min: AreaOther2 - 100 OtherRooms2 >= 0
 7.22.2__other_min_on_1st: OtherRooms1 >= 1
 7.23__p1_ub_linear: - 0.1 x_1st_Flr_SF + P1 <= 40
 7.23__p1_lb_linear: - 80 Floor1 + P1 >= 0
 7.23__p1_onoff: - 460 Floor1 + P1 <= 0
 7.23__p2_lb_linear: - 80 Floor2 + P2 >= 0
 7.23__p1_ge_p2_if_2nd: - 390 Floor2 + P1 - P2 >= -390
 7.23.1__area_ext: - 7 P1 - 7 P2 + AreaExterior1st = 0
 7.23.1__w_leq_area__VinylSd: - AreaExterior1st + W__VinylSd <= 0
 7.23.1__w_leq_u__VinylSd: - 5950 Exterior1st__VinylSd + W__VinylSd <= 0
 7.23.1__w_ge_lin__VinylSd: - 5950 Exterior1st__VinylSd - AreaExterior1st
   + W__VinylSd >= -5950
 7.23.1__w_leq_area__MetalSd: - AreaExterior1st + W__MetalSd <= 0
 7.23.1__w_leq_u__MetalSd: - 5950 Exterior1st__MetalSd + W__MetalSd <= 0
 7.23.1__w_ge_lin__MetalSd: - 5950 Exterior1st__MetalSd - AreaExterior1st
   + W__MetalSd >= -5950
 7.23.1__w_leq_area__Wd_Sdng: - AreaExterior1st + W__Wd_Sdng <= 0
 7.23.1__w_leq_u__Wd_Sdng: - 5950 Exterior1st__Wd_Sdng + W__Wd_Sdng <= 0
 7.23.1__w_ge_lin__Wd_Sdng: - 5950 Exterior1st__Wd_Sdng - AreaExterior1st
   + W__Wd_Sdng >= -5950
 7.23.1__w_leq_area__HdBoard: - AreaExterior1st + W__HdBoard <= 0
 7.23.1__w_leq_u__HdBoard: - 5950 Exterior1st__HdBoard + W__HdBoard <= 0
 7.23.1__w_ge_lin__HdBoard: - 5950 Exterior1st__HdBoard - AreaExterior1st
   + W__HdBoard >= -5950
 7.23.1__w_leq_area__Stucco: - AreaExterior1st + W__Stucco <= 0
 7.23.1__w_leq_u__Stucco: - 5950 Exterior1st__Stucco + W__Stucco <= 0
 7.23.1__w_ge_lin__Stucco: - 5950 Exterior1st__Stucco - AreaExterior1st
   + W__Stucco >= -5950
 7.23.1__w_leq_area__Plywood: - AreaExterior1st + W__Plywood <= 0
 7.23.1__w_leq_u__Plywood: - 5950 Exterior1st__Plywood + W__Plywood <= 0
 7.23.1__w_ge_lin__Plywood: - 5950 Exterior1st__Plywood - AreaExterior1st
   + W__Plywood >= -5950
 7.23.1__w_leq_area__CemntBd: - AreaExterior1st + W__CemntBd <= 0
 7.23.1__w_leq_u__CemntBd: - 5950 Exterior1st__CemntBd + W__CemntBd <= 0
 7.23.1__w_ge_lin__CemntBd: - 5950 Exterior1st__CemntBd - AreaExterior1st
   + W__CemntBd >= -5950
 7.23.1__w_leq_area__BrkFace: - AreaExterior1st + W__BrkFace <= 0
 7.23.1__w_leq_u__BrkFace: - 5950 Exterior1st__BrkFace + W__BrkFace <= 0
 7.23.1__w_ge_lin__BrkFace: - 5950 Exterior1st__BrkFace - AreaExterior1st
   + W__BrkFace >= -5950
 7.23.1__w_leq_area__BrkComm: - AreaExterior1st + W__BrkComm <= 0
 7.23.1__w_leq_u__BrkComm: - 5950 Exterior1st__BrkComm + W__BrkComm <= 0
 7.23.1__w_ge_lin__BrkComm: - 5950 Exterior1st__BrkComm - AreaExterior1st
   + W__BrkComm >= -5950
 7.23.1__w_leq_area__WdShngl: - AreaExterior1st + W__WdShngl <= 0
 7.23.1__w_leq_u__WdShngl: - 5950 Exterior1st__WdShngl + W__WdShngl <= 0
 7.23.1__w_ge_lin__WdShngl: - 5950 Exterior1st__WdShngl - AreaExterior1st
   + W__WdShngl >= -5950
 7.23.1__w_leq_area__AsbShng: - AreaExterior1st + W__AsbShng <= 0
 7.23.1__w_leq_u__AsbShng: - 5950 Exterior1st__AsbShng + W__AsbShng <= 0
 7.23.1__w_ge_lin__AsbShng: - 5950 Exterior1st__AsbShng - AreaExterior1st
   + W__AsbShng >= -5950
 7.23.1__w_leq_area__Stone: - AreaExterior1st + W__Stone <= 0
 7.23.1__w_leq_u__Stone: - 5950 Exterior1st__Stone + W__Stone <= 0
 7.23.1__w_ge_lin__Stone: - 5950 Exterior1st__Stone - AreaExterior1st
   + W__Stone >= -5950
 7.23.1__w_leq_area__ImStucc: - AreaExterior1st + W__ImStucc <= 0
 7.23.1__w_leq_u__ImStucc: - 5950 Exterior1st__ImStucc + W__ImStucc <= 0
 7.23.1__w_ge_lin__ImStucc: - 5950 Exterior1st__ImStucc - AreaExterior1st
   + W__ImStucc >= -5950
 7.23.1__w_leq_area__AsphShn: - AreaExterior1st + W__AsphShn <= 0
 7.23.1__w_leq_u__AsphShn: - 5950 Exterior1st__AsphShn + W__AsphShn <= 0
 7.23.1__w_ge_lin__AsphShn: - 5950 Exterior1st__AsphShn - AreaExterior1st
   + W__AsphShn >= -5950
 7.23.1__w_leq_area__CBlock: - AreaExterior1st + W__CBlock <= 0
 7.23.1__w_leq_u__CBlock: - 5950 Exterior1st__CBlock + W__CBlock <= 0
 7.23.1__w_ge_lin__CBlock: - 5950 Exterior1st__CBlock - AreaExterior1st
   + W__CBlock >= -5950
 7.23.1__w_sum: - AreaExterior1st + W__VinylSd + W__MetalSd + W__Wd_Sdng
   + W__HdBoard + W__Stucco + W__Plywood + W__CemntBd + W__BrkFace
   + W__BrkComm + W__WdShngl + W__AsbShng + W__Stone + W__ImStucc
   + W__AsphShn + W__CBlock = 0
 ex__Fence: Fence__GdPrv + Fence__MnPrv + Fence__GdWo + Fence__MnWw
   + Fence__No_aplica = 1
 link__fence__gdprv: x_Fence_GdPrv - Fence__GdPrv = 0
 link__fence__mnprv: x_Fence_MnPrv - Fence__MnPrv = 0
 link__fence__gdwo: x_Fence_GdWo - Fence__GdWo = 0
 link__fence__mnww: x_Fence_MnWw - Fence__MnWw = 0
 link__fence__noaplica: x_Fence_No_aplica - Fence__No_aplica = 0
 11.xx__hasreja_from_fence: HasReja + Fence__No_aplica = 1
 ex__MiscFeature: MiscFeature__Elev + MiscFeature__Gar2 + MiscFeature__Othr
   + MiscFeature__Shed + MiscFeature__TenC + MiscFeature__No_aplica = 1
 link__miscfeature__elev: x_Misc_Feature_Elev - MiscFeature__Elev = 0
 link__miscfeature__gar2: x_Misc_Feature_Gar2 - MiscFeature__Gar2 = 0
 link__miscfeature__othr: x_Misc_Feature_Othr - MiscFeature__Othr = 0
 link__miscfeature__shed: x_Misc_Feature_Shed - MiscFeature__Shed = 0
 link__miscfeature__tenc: x_Misc_Feature_TenC - MiscFeature__TenC = 0
 link__miscfeature__noaplica: x_Misc_Feature_No_aplica
   - MiscFeature__No_aplica = 0
 def_cost_model: - 230 x_Bsmt_Unf_SF - 50 x_Wood_Deck_SF
   - 77.5 x_Open_Porch_SF - 80 x_Enclosed_Porch - 157.5 x_3Ssn_Porch
   - 72.5 x_Screen_Porch - 31750 Floor1 - 230 AreaOther1 - 230 AreaOther2
   - 200 AreaKitchen - 500 AreaFullBath - 500 AreaHalfBath
   - 325 AreaBedroom - 2500 Fireplaces_TA - 3525 Fireplaces_Gd
   - 4550 Fireplaces_Ex - 24038 GarageFinish__Fin - 20769 GarageFinish__RFn
   - 17500 GarageFinish__Unf - 54 AGarage_TA - 57.6 AGarage_Gd
   - 60 AGarage_Ex - 6500 zHQ_TA - 8250 zHQ_Gd - 10000 zHQ_Ex
   - 18883.75 zEQ_TA - 22833.06 zEQ_Gd - 106250 zEQ_Ex
   - 1773 Heating__Floor - 5750 Heating__GasA - 8500 Heating__GasW
   - 6300 Heating__Grav - 5000 Heating__OthW - 3700 Heating__Wall
   - 5362 CentralAir__Y - 1587.5 Electrical__SBrkr - 2500 Electrical__FuseA
   - 1675 Electrical__FuseF - 850 Electrical__FuseP - 1075 Electrical__Mix
   - 4 PavedDrive__Y - 5 PavedDrive__P - PavedDrive__N
   - 48000 MiscFeature__Elev - 32100 MiscFeature__Gar2
   - 50000 MiscFeature__Othr - 5631 MiscFeature__Shed
   - 15774 MiscFeature__TenC - 17352 RoofMatl__ClyTile
   - 20000 RoofMatl__CompShg - 8011.95 RoofMatl__Membran
   - 11739 RoofMatl__Metal - 7600 RoofMatl__Roll - 8550 RoofMatl__Tar&Grv
   - 22500 RoofMatl__WdShake - 19500 RoofMatl__WdShngl - 70.4 APool_TA
   - 80.96 APool_Gd - 88 APool_Ex - 15 BsmtFinSF1 - 15 BsmtFinSF2
   - 12500 Exterior1st__VinylSd - 9600 Exterior1st__MetalSd
   - 12500 Exterior1st__Wd_Sdng - 21300 Exterior1st__HdBoard
   - 5629 Exterior1st__Stucco - 8461.81 Exterior1st__Plywood
   - 14674 Exterior1st__CemntBd - 22000 Exterior1st__BrkFace
   - 26000 Exterior1st__BrkComm - 21900 Exterior1st__WdShngl
   - 19000 Exterior1st__AsbShng - 50216.5 Exterior1st__Stone
   - 16500 Exterior1st__ImStucc - 22500 Exterior1st__AsphShn
   - 10300 Exterior1st__CBlock - 12500 Exterior2nd__VinylSd
   - 9600 Exterior2nd__MetalSd - 12500 Exterior2nd__Wd_Sdng
   - 21300 Exterior2nd__HdBoard - 5629 Exterior2nd__Stucco
   - 8461.81 Exterior2nd__Plywood - 14674 Exterior2nd__CemntBd
   - 22000 Exterior2nd__BrkFace - 26000 Exterior2nd__BrkComm
   - 21900 Exterior2nd__WdShngl - 19000 Exterior2nd__AsbShng
   - 50216.5 Exterior2nd__Stone - 16500 Exterior2nd__ImStucc
   - 22500 Exterior2nd__AsphShn - 10300 Exterior2nd__CBlock
   - 1.21 MvProd__BrkCmn - 15 MvProd__BrkFace - 22.5 MvProd__CBlock
   - 27.5 MvProd__Stone - 22 FA__BrkTil - 12 FA__CBlock - 10 FA__PConc
   - 10 FA__Slab - 23.5 FA__Stone - 40 FA__Wood - 230 Remainder1
   - 230 Remainder2 + cost_model = 0
 TREE_0_ONEHOT: t0_leaf0 = 1
 TREE_1_ONEHOT: t1_leaf0 = 1
 TREE_2_ONEHOT: t2_leaf0 = 1
 TREE_3_ONEHOT: t3_leaf0 = 1
 TREE_4_ONEHOT: t4_leaf0 = 1
 TREE_5_ONEHOT: t5_leaf0 = 1
 TREE_6_ONEHOT: t6_leaf0 = 1
 TREE_7_ONEHOT: t7_leaf0 = 1
 TREE_8_ONEHOT: t8_leaf0 = 1
 TREE_9_ONEHOT: t9_leaf0 = 1
 TREE_10_ONEHOT: t10_leaf0 = 1
 TREE_11_ONEHOT: t11_leaf0 = 1
 TREE_12_ONEHOT: t12_leaf0 = 1
 TREE_13_ONEHOT: t13_leaf0 = 1
 TREE_14_ONEHOT: t14_leaf0 = 1
 TREE_15_ONEHOT: t15_leaf0 = 1
 TREE_16_ONEHOT: t16_leaf0 = 1
 TREE_17_ONEHOT: t17_leaf0 = 1
 TREE_18_ONEHOT: t18_leaf0 = 1
 TREE_19_ONEHOT: t19_leaf0 = 1
 TREE_20_ONEHOT: t20_leaf0 = 1
 TREE_21_ONEHOT: t21_leaf0 = 1
 TREE_22_ONEHOT: t22_leaf0 = 1
 TREE_23_ONEHOT: t23_leaf0 + t23_leaf1 = 1
 T23_L0_f3_le: x_Overall_Qual + 999993.000001 t23_leaf0 <= 1e+06
 T23_R1_f3_ge: x_Overall_Qual - 7 t23_leaf1 >= 0
 TREE_24_ONEHOT: t24_leaf0 + t24_leaf1 = 1
 T24_L0_f3_le: x_Overall_Qual + 999993.000001 t24_leaf0 <= 1e+06
 T24_R1_f3_ge: x_Overall_Qual - 7 t24_leaf1 >= 0
 TREE_25_ONEHOT: t25_leaf0 + t25_leaf1 = 1
 T25_L0_f3_le: x_Overall_Qual + 999993.000001 t25_leaf0 <= 1e+06
 T25_R1_f3_ge: x_Overall_Qual - 7 t25_leaf1 >= 0
 TREE_26_ONEHOT: t26_leaf0 + t26_leaf1 = 1
 T26_L0_f3_le: x_Overall_Qual + 999994.000001 t26_leaf0 <= 1e+06
 T26_R1_f3_ge: x_Overall_Qual - 6 t26_leaf1 >= 0
 TREE_27_ONEHOT: t27_leaf0 + t27_leaf1 = 1
 T27_L0_f8_le: x_Exter_Qual + 1.000001 t27_leaf0 <= 4
 T27_R1_f8_ge: x_Exter_Qual - t27_leaf1 >= 2
 TREE_28_ONEHOT: t28_leaf0 + t28_leaf1 = 1
 T28_L0_f31_le: x_Garage_Yr_Blt + 998017.000001 t28_leaf0 <= 1e+06
 T28_R1_f31_ge: x_Garage_Yr_Blt - 1983 t28_leaf1 >= 0
 TREE_29_ONEHOT: t29_leaf0 + t29_leaf1 = 1
 T29_L0_f5_le: x_Year_Built + 998019.000001 t29_leaf0 <= 1e+06
 T29_R1_f5_ge: x_Year_Built - 1981 t29_leaf1 >= 0
 TREE_30_ONEHOT: t30_leaf0 + t30_leaf1 = 1
 T30_L0_f3_le: x_Overall_Qual + 999993.000001 t30_leaf0 <= 1e+06
 T30_R1_f3_ge: x_Overall_Qual - 7 t30_leaf1 >= 0
 TREE_31_ONEHOT: t31_leaf0 + t31_leaf1 = 1
 T31_L0_f20_le: x_Gr_Liv_Area + 2005.760001 t31_leaf0 <= 3421.76
 T31_R1_f20_ge: x_Gr_Liv_Area - 1416 t31_leaf1 >= 0
 TREE_32_ONEHOT: t32_leaf0 + t32_leaf1 = 1
 T32_L0_f33_le: x_Garage_Area + 773.750001 t32_leaf0 <= 1163.75
 T32_R1_f33_ge: x_Garage_Area - 390 t32_leaf1 >= 0
 TREE_33_ONEHOT: t33_leaf0 + t33_leaf1 = 1
 T33_L0_f20_le: x_Gr_Liv_Area + 1933.760001 t33_leaf0 <= 3421.76
 T33_R1_f20_ge: x_Gr_Liv_Area - 1488 t33_leaf1 >= 0
 TREE_34_ONEHOT: t34_leaf0 + t34_leaf1 + t34_leaf2 = 1
 T34_L0_f3_le: x_Overall_Qual + 999994.000001 t34_leaf0 <= 1e+06
 T34_L0_f3_le: x_Overall_Qual + 999997.000001 t34_leaf0 <= 1e+06
 T34_L1_f3_le: x_Overall_Qual + 999994.000001 t34_leaf1 <= 1e+06
 T34_R1_f3_ge: x_Overall_Qual - 3 t34_leaf1 >= 0
 T34_R2_f3_ge: x_Overall_Qual - 6 t34_leaf2 >= 0
 TREE_35_ONEHOT: t35_leaf0 + t35_leaf1 = 1
 T35_L0_f3_le: x_Overall_Qual + 999993.000001 t35_leaf0 <= 1e+06
 T35_R1_f3_ge: x_Overall_Qual - 7 t35_leaf1 >= 0
 TREE_36_ONEHOT: t36_leaf0 + t36_leaf1 = 1
 T36_L0_f20_le: x_Gr_Liv_Area + 2043.760001 t36_leaf0 <= 3421.76
 T36_R1_f20_ge: x_Gr_Liv_Area - 1378 t36_leaf1 >= 0
 TREE_37_ONEHOT: t37_leaf0 + t37_leaf1 + t37_leaf2 = 1
 T37_L0_f20_le: x_Gr_Liv_Area + 1933.760001 t37_leaf0 <= 3421.76
 T37_L0_f231_le: x_Central_Air_N + 1e-06 t37_leaf0 <= 1
 T37_L1_f20_le: x_Gr_Liv_Area + 1933.760001 t37_leaf1 <= 3421.76
 T37_R1_f231_ge: x_Central_Air_N - t37_leaf1 >= 0
 T37_R2_f20_ge: x_Gr_Liv_Area - 1488 t37_leaf2 >= 0
 TREE_38_ONEHOT: t38_leaf0 + t38_leaf1 + t38_leaf2 = 1
 T38_L0_f3_le: x_Overall_Qual + 999994.000001 t38_leaf0 <= 1e+06
 T38_L0_f3_le: x_Overall_Qual + 999997.000001 t38_leaf0 <= 1e+06
 T38_L1_f3_le: x_Overall_Qual + 999994.000001 t38_leaf1 <= 1e+06
 T38_R1_f3_ge: x_Overall_Qual - 3 t38_leaf1 >= 0
 T38_R2_f3_ge: x_Overall_Qual - 6 t38_leaf2 >= 0
 TREE_39_ONEHOT: t39_leaf0 + t39_leaf1 + t39_leaf2 = 1
 T39_L0_f3_le: x_Overall_Qual + 999993.000001 t39_leaf0 <= 1e+06
 T39_L0_f3_le: x_Overall_Qual + 999997.000001 t39_leaf0 <= 1e+06
 T39_L1_f3_le: x_Overall_Qual + 999993.000001 t39_leaf1 <= 1e+06
 T39_R1_f3_ge: x_Overall_Qual - 3 t39_leaf1 >= 0
 T39_R2_f3_ge: x_Overall_Qual - 7 t39_leaf2 >= 0
 TREE_40_ONEHOT: t40_leaf0 + t40_leaf1 + t40_leaf2 = 1
 T40_L0_f3_le: x_Overall_Qual + 999993.000001 t40_leaf0 <= 1e+06
 T40_L0_f3_le: x_Overall_Qual + 999997.000001 t40_leaf0 <= 1e+06
 T40_L1_f3_le: x_Overall_Qual + 999993.000001 t40_leaf1 <= 1e+06
 T40_R1_f3_ge: x_Overall_Qual - 3 t40_leaf1 >= 0
 T40_R2_f3_ge: x_Overall_Qual - 7 t40_leaf2 >= 0
 TREE_41_ONEHOT: t41_leaf0 + t41_leaf1 + t41_leaf2 = 1
 T41_L0_f20_le: x_Gr_Liv_Area + 1933.760001 t41_leaf0 <= 3421.76
 T41_L0_f4_le: x_Overall_Cond + 999997.000001 t41_leaf0 <= 1e+06
 T41_L1_f20_le: x_Gr_Liv_Area + 1933.760001 t41_leaf1 <= 3421.76
 T41_R1_f4_ge: x_Overall_Cond - 3 t41_leaf1 >= 0
 T41_R2_f20_ge: x_Gr_Liv_Area - 1488 t41_leaf2 >= 0
 TREE_42_ONEHOT: t42_leaf0 + t42_leaf1 = 1
 T42_L0_f27_le: x_Kitchen_Qual + 1.000001 t42_leaf0 <= 4
 T42_R1_f27_ge: x_Kitchen_Qual - t42_leaf1 >= 2
 TREE_43_ONEHOT: t43_leaf0 + t43_leaf1 + t43_leaf2 = 1
 T43_L0_f3_le: x_Overall_Qual + 999992.000001 t43_leaf0 <= 1e+06
 T43_L0_f3_le: x_Overall_Qual + 999995.000001 t43_leaf0 <= 1e+06
 T43_L1_f3_le: x_Overall_Qual + 999992.000001 t43_leaf1 <= 1e+06
 T43_R1_f3_ge: x_Overall_Qual - 5 t43_leaf1 >= 0
 T43_R2_f3_ge: x_Overall_Qual - 8 t43_leaf2 >= 0
 TREE_44_ONEHOT: t44_leaf0 + t44_leaf1 + t44_leaf2 = 1
 T44_L0_f15_le: x_Total_Bsmt_SF + 1461.480001 t44_leaf0 <= 2514.48
 T44_L0_f3_le: x_Overall_Qual + 999995.000001 t44_leaf0 <= 1e+06
 T44_L1_f15_le: x_Total_Bsmt_SF + 1461.480001 t44_leaf1 <= 2514.48
 T44_R1_f3_ge: x_Overall_Qual - 5 t44_leaf1 >= 0
 T44_R2_f15_ge: x_Total_Bsmt_SF - 1053 t44_leaf2 >= 0
 TREE_45_ONEHOT: t45_leaf0 + t45_leaf1 + t45_leaf2 = 1
 T45_L0_f32_le: x_Garage_Cars + 2.000001 t45_leaf0 <= 4
 T45_R1_f32_ge: x_Garage_Cars - 2 t45_leaf1 >= 0
 T45_L1_f8_le: x_Exter_Qual + 2.000001 t45_leaf1 <= 4
 T45_R2_f32_ge: x_Garage_Cars - 2 t45_leaf2 >= 0
 T45_R2_f8_ge: x_Exter_Qual >= 2
 TREE_46_ONEHOT: t46_leaf0 + t46_leaf1 + t46_leaf2 + t46_leaf3 = 1
 T46_L0_f3_le: x_Overall_Qual + 999992.000001 t46_leaf0 <= 1e+06
 T46_L0_f3_le: x_Overall_Qual + 999995.000001 t46_leaf0 <= 1e+06
 T46_L0_f4_le: x_Overall_Cond + 999997.000001 t46_leaf0 <= 1e+06
 T46_L1_f3_le: x_Overall_Qual + 999992.000001 t46_leaf1 <= 1e+06
 T46_L1_f3_le: x_Overall_Qual + 999995.000001 t46_leaf1 <= 1e+06
 T46_R1_f4_ge: x_Overall_Cond - 3 t46_leaf1 >= 0
 T46_L2_f3_le: x_Overall_Qual + 999992.000001 t46_leaf2 <= 1e+06
 T46_R2_f3_ge: x_Overall_Qual - 5 t46_leaf2 >= 0
 T46_R3_f3_ge: x_Overall_Qual - 8 t46_leaf3 >= 0
 TREE_47_ONEHOT: t47_leaf0 + t47_leaf1 + t47_leaf2 = 1
 T47_L0_f3_le: x_Overall_Qual + 999992.000001 t47_leaf0 <= 1e+06
 T47_L0_f3_le: x_Overall_Qual + 999995.000001 t47_leaf0 <= 1e+06
 T47_L1_f3_le: x_Overall_Qual + 999992.000001 t47_leaf1 <= 1e+06
 T47_R1_f3_ge: x_Overall_Qual - 5 t47_leaf1 >= 0
 T47_R2_f3_ge: x_Overall_Qual - 8 t47_leaf2 >= 0
 TREE_48_ONEHOT: t48_leaf0 + t48_leaf1 + t48_leaf2 + t48_leaf3 + t48_leaf4
   + t48_leaf5 = 1
 T48_L0_f20_le: x_Gr_Liv_Area + 2221.760001000001 t48_leaf0 <= 3421.76
 T48_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t48_leaf0 <= 1
 T48_L0_f6_le: x_Year_Remod/Add + 998047.000001 t48_leaf0 <= 1e+06
 T48_L1_f20_le: x_Gr_Liv_Area + 2221.760001000001 t48_leaf1 <= 3421.76
 T48_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t48_leaf1 <= 1
 T48_R1_f6_ge: x_Year_Remod/Add - 1953 t48_leaf1 >= 0
 T48_L2_f20_le: x_Gr_Liv_Area + 2221.760001000001 t48_leaf2 <= 3421.76
 T48_R2_f161_ge: x_Exterior_1st_AsbShng - t48_leaf2 >= 0
 T48_L2_f20_le: x_Gr_Liv_Area + 2588.760001 t48_leaf2 <= 3421.76
 T48_L3_f20_le: x_Gr_Liv_Area + 2221.760001000001 t48_leaf3 <= 3421.76
 T48_R3_f161_ge: x_Exterior_1st_AsbShng - t48_leaf3 >= 0
 T48_R3_f20_ge: x_Gr_Liv_Area - 833 t48_leaf3 >= 0
 T48_R4_f20_ge: x_Gr_Liv_Area - 1200 t48_leaf4 >= 0
 T48_L4_f10_le: x_Bsmt_Qual + 1e-06 t48_leaf4 <= 4
 T48_R5_f20_ge: x_Gr_Liv_Area - 1200 t48_leaf5 >= 0
 T48_R5_f10_ge: x_Bsmt_Qual - 2 t48_leaf5 >= 2
 TREE_49_ONEHOT: t49_leaf0 + t49_leaf1 + t49_leaf2 + t49_leaf3 + t49_leaf4
   = 1
 T49_L0_f30_le: x_Fireplace_Qu + 3.000001 t49_leaf0 <= 4
 T49_L0_f231_le: x_Central_Air_N + 1e-06 t49_leaf0 <= 1
 T49_L1_f30_le: x_Fireplace_Qu + 3.000001 t49_leaf1 <= 4
 T49_R1_f231_ge: x_Central_Air_N - t49_leaf1 >= 0
 T49_L1_f20_le: x_Gr_Liv_Area + 2674.760001 t49_leaf1 <= 3421.76
 T49_L2_f30_le: x_Fireplace_Qu + 3.000001 t49_leaf2 <= 4
 T49_R2_f231_ge: x_Central_Air_N - t49_leaf2 >= 0
 T49_R2_f20_ge: x_Gr_Liv_Area - 747 t49_leaf2 >= 0
 T49_R3_f30_ge: x_Fireplace_Qu >= 1
 T49_L3_f20_le: x_Gr_Liv_Area + 2588.760001 t49_leaf3 <= 3421.76
 T49_R4_f30_ge: x_Fireplace_Qu >= 1
 T49_R4_f20_ge: x_Gr_Liv_Area - 833 t49_leaf4 >= 0
 TREE_50_ONEHOT: t50_leaf0 + t50_leaf1 + t50_leaf2 + t50_leaf3 = 1
 T50_L0_f15_le: x_Total_Bsmt_SF + 1468.480001 t50_leaf0 <= 2514.48
 T50_L0_f35_le: x_Garage_Cond + 2.000001 t50_leaf0 <= 4
 T50_L0_f226_le: x_Heating_GasA + 1e-06 t50_leaf0 <= 1
 T50_L1_f15_le: x_Total_Bsmt_SF + 1468.480001 t50_leaf1 <= 2514.48
 T50_L1_f35_le: x_Garage_Cond + 2.000001 t50_leaf1 <= 4
 T50_R1_f226_ge: x_Heating_GasA - t50_leaf1 >= 0
 T50_L2_f15_le: x_Total_Bsmt_SF + 1468.480001 t50_leaf2 <= 2514.48
 T50_R2_f35_ge: x_Garage_Cond - 3 t50_leaf2 >= -1
 T50_R3_f15_ge: x_Total_Bsmt_SF - 1046 t50_leaf3 >= 0
 TREE_51_ONEHOT: t51_leaf0 + t51_leaf1 + t51_leaf2 + t51_leaf3 + t51_leaf4
   = 1
 T51_L0_f30_le: x_Fireplace_Qu + 3.000001 t51_leaf0 <= 4
 T51_L0_f20_le: x_Gr_Liv_Area + 2674.760001 t51_leaf0 <= 3421.76
 T51_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t51_leaf0 <= 1
 T51_L1_f30_le: x_Fireplace_Qu + 3.000001 t51_leaf1 <= 4
 T51_L1_f20_le: x_Gr_Liv_Area + 2674.760001 t51_leaf1 <= 3421.76
 T51_R1_f297_ge: x_Sale_Condition_Normal - t51_leaf1 >= 0
 T51_L2_f30_le: x_Fireplace_Qu + 3.000001 t51_leaf2 <= 4
 T51_R2_f20_ge: x_Gr_Liv_Area - 747 t51_leaf2 >= 0
 T51_R3_f30_ge: x_Fireplace_Qu >= 1
 T51_L3_f20_le: x_Gr_Liv_Area + 1655.760001 t51_leaf3 <= 3421.76
 T51_R4_f30_ge: x_Fireplace_Qu >= 1
 T51_R4_f20_ge: x_Gr_Liv_Area - 1766 t51_leaf4 >= 0
 TREE_52_ONEHOT: t52_leaf0 + t52_leaf1 + t52_leaf2 + t52_leaf3 = 1
 T52_L0_f3_le: x_Overall_Qual + 999992.000001 t52_leaf0 <= 1e+06
 T52_L0_f3_le: x_Overall_Qual + 999995.000001 t52_leaf0 <= 1e+06
 T52_L0_f62_le: x_MS_Zoning_C_(all) + 1e-06 t52_leaf0 <= 1
 T52_L1_f3_le: x_Overall_Qual + 999992.000001 t52_leaf1 <= 1e+06
 T52_L1_f3_le: x_Overall_Qual + 999995.000001 t52_leaf1 <= 1e+06
 T52_R1_f62_ge: x_MS_Zoning_C_(all) - t52_leaf1 >= 0
 T52_L2_f3_le: x_Overall_Qual + 999992.000001 t52_leaf2 <= 1e+06
 T52_R2_f3_ge: x_Overall_Qual - 5 t52_leaf2 >= 0
 T52_R3_f3_ge: x_Overall_Qual - 8 t52_leaf3 >= 0
 TREE_53_ONEHOT: t53_leaf0 + t53_leaf1 + t53_leaf2 + t53_leaf3 + t53_leaf4
   = 1
 T53_L0_f15_le: x_Total_Bsmt_SF + 1451.480001 t53_leaf0 <= 2514.48
 T53_L0_f3_le: x_Overall_Qual + 999996.000001 t53_leaf0 <= 1e+06
 T53_L1_f15_le: x_Total_Bsmt_SF + 1451.480001 t53_leaf1 <= 2514.48
 T53_R1_f3_ge: x_Overall_Qual - 4 t53_leaf1 >= 0
 T53_L1_f62_le: x_MS_Zoning_C_(all) + 1e-06 t53_leaf1 <= 1
 T53_L2_f15_le: x_Total_Bsmt_SF + 1451.480001 t53_leaf2 <= 2514.48
 T53_R2_f3_ge: x_Overall_Qual - 4 t53_leaf2 >= 0
 T53_R2_f62_ge: x_MS_Zoning_C_(all) - t53_leaf2 >= 0
 T53_R3_f15_ge: x_Total_Bsmt_SF - 1063 t53_leaf3 >= 0
 T53_L3_f3_le: x_Overall_Qual + 999992.000001 t53_leaf3 <= 1e+06
 T53_R4_f15_ge: x_Total_Bsmt_SF - 1063 t53_leaf4 >= 0
 T53_R4_f3_ge: x_Overall_Qual - 8 t53_leaf4 >= 0
 TREE_54_ONEHOT: t54_leaf0 + t54_leaf1 + t54_leaf2 + t54_leaf3 + t54_leaf4
   = 1
 T54_L0_f3_le: x_Overall_Qual + 999992.000001 t54_leaf0 <= 1e+06
 T54_L0_f3_le: x_Overall_Qual + 999996.000001 t54_leaf0 <= 1e+06
 T54_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t54_leaf0 <= 1
 T54_L1_f3_le: x_Overall_Qual + 999992.000001 t54_leaf1 <= 1e+06
 T54_L1_f3_le: x_Overall_Qual + 999996.000001 t54_leaf1 <= 1e+06
 T54_R1_f293_ge: x_Sale_Condition_Abnorml - t54_leaf1 >= 0
 T54_L2_f3_le: x_Overall_Qual + 999992.000001 t54_leaf2 <= 1e+06
 T54_R2_f3_ge: x_Overall_Qual - 4 t54_leaf2 >= 0
 T54_L2_f15_le: x_Total_Bsmt_SF + 1825.480001 t54_leaf2 <= 2514.48
 T54_L3_f3_le: x_Overall_Qual + 999992.000001 t54_leaf3 <= 1e+06
 T54_R3_f3_ge: x_Overall_Qual - 4 t54_leaf3 >= 0
 T54_R3_f15_ge: x_Total_Bsmt_SF - 689 t54_leaf3 >= 0
 T54_R4_f3_ge: x_Overall_Qual - 8 t54_leaf4 >= 0
 TREE_55_ONEHOT: t55_leaf0 + t55_leaf1 + t55_leaf2 + t55_leaf3 + t55_leaf4
   + t55_leaf5 = 1
 T55_L0_f30_le: x_Fireplace_Qu + 3.000001 t55_leaf0 <= 4
 T55_L0_f20_le: x_Gr_Liv_Area + 2611.760001 t55_leaf0 <= 3421.76
 T55_L0_f15_le: x_Total_Bsmt_SF + 2195.480001 t55_leaf0 <= 2514.48
 T55_L1_f30_le: x_Fireplace_Qu + 3.000001 t55_leaf1 <= 4
 T55_L1_f20_le: x_Gr_Liv_Area + 2611.760001 t55_leaf1 <= 3421.76
 T55_R1_f15_ge: x_Total_Bsmt_SF - 319 t55_leaf1 >= 0
 T55_L2_f30_le: x_Fireplace_Qu + 3.000001 t55_leaf2 <= 4
 T55_R2_f20_ge: x_Gr_Liv_Area - 810 t55_leaf2 >= 0
 T55_L2_f9_le: x_Exter_Cond + 2.000001 t55_leaf2 <= 4
 T55_L3_f30_le: x_Fireplace_Qu + 3.000001 t55_leaf3 <= 4
 T55_R3_f20_ge: x_Gr_Liv_Area - 810 t55_leaf3 >= 0
 T55_R3_f9_ge: x_Exter_Cond - 3 t55_leaf3 >= -1
 T55_R4_f30_ge: x_Fireplace_Qu >= 1
 T55_L4_f20_le: x_Gr_Liv_Area + 1655.760001 t55_leaf4 <= 3421.76
 T55_R5_f30_ge: x_Fireplace_Qu >= 1
 T55_R5_f20_ge: x_Gr_Liv_Area - 1766 t55_leaf5 >= 0
 TREE_56_ONEHOT: t56_leaf0 + t56_leaf1 + t56_leaf2 + t56_leaf3 = 1
 T56_L0_f33_le: x_Garage_Area + 722.750001 t56_leaf0 <= 1163.75
 T56_L0_f4_le: x_Overall_Cond + 999996.000001 t56_leaf0 <= 1e+06
 T56_L1_f33_le: x_Garage_Area + 722.750001 t56_leaf1 <= 1163.75
 T56_R1_f4_ge: x_Overall_Cond - 4 t56_leaf1 >= 0
 T56_R2_f33_ge: x_Garage_Area - 441 t56_leaf2 >= 0
 T56_L2_f20_le: x_Gr_Liv_Area + 1453.760001 t56_leaf2 <= 3421.76
 T56_R3_f33_ge: x_Garage_Area - 441 t56_leaf3 >= 0
 T56_R3_f20_ge: x_Gr_Liv_Area - 1968 t56_leaf3 >= 0
 TREE_57_ONEHOT: t57_leaf0 + t57_leaf1 + t57_leaf2 + t57_leaf3 = 1
 T57_L0_f3_le: x_Overall_Qual + 999992.000001 t57_leaf0 <= 1e+06
 T57_L0_f15_le: x_Total_Bsmt_SF + 1851.480001 t57_leaf0 <= 2514.48
 T57_L1_f3_le: x_Overall_Qual + 999992.000001 t57_leaf1 <= 1e+06
 T57_R1_f15_ge: x_Total_Bsmt_SF - 663 t57_leaf1 >= 0
 T57_L1_f4_le: x_Overall_Cond + 999996.000001 t57_leaf1 <= 1e+06
 T57_L2_f3_le: x_Overall_Qual + 999992.000001 t57_leaf2 <= 1e+06
 T57_R2_f15_ge: x_Total_Bsmt_SF - 663 t57_leaf2 >= 0
 T57_R2_f4_ge: x_Overall_Cond - 4 t57_leaf2 >= 0
 T57_R3_f3_ge: x_Overall_Qual - 8 t57_leaf3 >= 0
 TREE_58_ONEHOT: t58_leaf0 + t58_leaf1 + t58_leaf2 + t58_leaf3 + t58_leaf4
   = 1
 T58_L0_f30_le: x_Fireplace_Qu + 3.000001 t58_leaf0 <= 4
 T58_L0_f20_le: x_Gr_Liv_Area + 2633.760001 t58_leaf0 <= 3421.76
 T58_L1_f30_le: x_Fireplace_Qu + 3.000001 t58_leaf1 <= 4
 T58_R1_f20_ge: x_Gr_Liv_Area - 788 t58_leaf1 >= 0
 T58_L1_f4_le: x_Overall_Cond + 999996.000001 t58_leaf1 <= 1e+06
 T58_L2_f30_le: x_Fireplace_Qu + 3.000001 t58_leaf2 <= 4
 T58_R2_f20_ge: x_Gr_Liv_Area - 788 t58_leaf2 >= 0
 T58_R2_f4_ge: x_Overall_Cond - 4 t58_leaf2 >= 0
 T58_R3_f30_ge: x_Fireplace_Qu >= 1
 T58_L3_f20_le: x_Gr_Liv_Area + 1751.760001 t58_leaf3 <= 3421.76
 T58_R4_f30_ge: x_Fireplace_Qu >= 1
 T58_R4_f20_ge: x_Gr_Liv_Area - 1670 t58_leaf4 >= 0
 TREE_59_ONEHOT: t59_leaf0 + t59_leaf1 + t59_leaf2 + t59_leaf3 + t59_leaf4
   = 1
 T59_L0_f15_le: x_Total_Bsmt_SF + 1365.480001 t59_leaf0 <= 2514.48
 T59_L0_f35_le: x_Garage_Cond + 2.000001 t59_leaf0 <= 4
 T59_L1_f15_le: x_Total_Bsmt_SF + 1365.480001 t59_leaf1 <= 2514.48
 T59_R1_f35_ge: x_Garage_Cond - 3 t59_leaf1 >= -1
 T59_L1_f67_le: x_MS_Zoning_RM + 1e-06 t59_leaf1 <= 1
 T59_L2_f15_le: x_Total_Bsmt_SF + 1365.480001 t59_leaf2 <= 2514.48
 T59_R2_f35_ge: x_Garage_Cond - 3 t59_leaf2 >= -1
 T59_R2_f67_ge: x_MS_Zoning_RM - t59_leaf2 >= 0
 T59_R3_f15_ge: x_Total_Bsmt_SF - 1149 t59_leaf3 >= 0
 T59_L3_f32_le: x_Garage_Cars + 1.000001 t59_leaf3 <= 4
 T59_R4_f15_ge: x_Total_Bsmt_SF - 1149 t59_leaf4 >= 0
 T59_R4_f32_ge: x_Garage_Cars - 3 t59_leaf4 >= 0
 TREE_60_ONEHOT: t60_leaf0 + t60_leaf1 + t60_leaf2 + t60_leaf3 + t60_leaf4
   = 1
 T60_L0_f32_le: x_Garage_Cars + 1.000001 t60_leaf0 <= 4
 T60_L0_f20_le: x_Gr_Liv_Area + 2605.760001 t60_leaf0 <= 3421.76
 T60_L0_f10_le: x_Bsmt_Qual + 4.000001 t60_leaf0 <= 4
 T60_L1_f32_le: x_Garage_Cars + 1.000001 t60_leaf1 <= 4
 T60_L1_f20_le: x_Gr_Liv_Area + 2605.760001 t60_leaf1 <= 3421.76
 T60_R1_f10_ge: x_Bsmt_Qual >= 0
 T60_L2_f32_le: x_Garage_Cars + 1.000001 t60_leaf2 <= 4
 T60_R2_f20_ge: x_Gr_Liv_Area - 816 t60_leaf2 >= 0
 T60_L2_f247_le: x_Garage_Type_Attchd + 1e-06 t60_leaf2 <= 1
 T60_L3_f32_le: x_Garage_Cars + 1.000001 t60_leaf3 <= 4
 T60_R3_f20_ge: x_Gr_Liv_Area - 816 t60_leaf3 >= 0
 T60_R3_f247_ge: x_Garage_Type_Attchd - t60_leaf3 >= 0
 T60_R4_f32_ge: x_Garage_Cars - 3 t60_leaf4 >= 0
 TREE_61_ONEHOT: t61_leaf0 + t61_leaf1 + t61_leaf2 + t61_leaf3 + t61_leaf4
   + t61_leaf5 = 1
 T61_L0_f10_le: x_Bsmt_Qual + 1.000001 t61_leaf0 <= 4
 T61_L0_f4_le: x_Overall_Cond + 999996.000001 t61_leaf0 <= 1e+06
 T61_L0_f109_le: x_Neighborhood_OldTown + 1e-06 t61_leaf0 <= 1
 T61_L1_f10_le: x_Bsmt_Qual + 1.000001 t61_leaf1 <= 4
 T61_L1_f4_le: x_Overall_Cond + 999996.000001 t61_leaf1 <= 1e+06
 T61_R1_f109_ge: x_Neighborhood_OldTown - t61_leaf1 >= 0
 T61_L2_f10_le: x_Bsmt_Qual + 1.000001 t61_leaf2 <= 4
 T61_R2_f4_ge: x_Overall_Cond - 4 t61_leaf2 >= 0
 T61_L2_f1_le: x_Lot_Area + 238.0000010000003 t61_leaf2 <= 7000
 T61_L3_f10_le: x_Bsmt_Qual + 1.000001 t61_leaf3 <= 4
 T61_R3_f4_ge: x_Overall_Cond - 4 t61_leaf3 >= 0
 T61_R3_f1_ge: x_Lot_Area >= 6762
 T61_R4_f10_ge: x_Bsmt_Qual - t61_leaf4 >= 2
 T61_L4_f32_le: x_Garage_Cars + 1.000001 t61_leaf4 <= 4
 T61_R5_f10_ge: x_Bsmt_Qual - t61_leaf5 >= 2
 T61_R5_f32_ge: x_Garage_Cars - 3 t61_leaf5 >= 0
 TREE_62_ONEHOT: t62_leaf0 + t62_leaf1 + t62_leaf2 + t62_leaf3 + t62_leaf4
   = 1
 T62_L0_f20_le: x_Gr_Liv_Area + 1621.760001 t62_leaf0 <= 3421.76
 T62_L0_f6_le: x_Year_Remod/Add + 998049.000001 t62_leaf0 <= 1e+06
 T62_L1_f20_le: x_Gr_Liv_Area + 1621.760001 t62_leaf1 <= 3421.76
 T62_R1_f6_ge: x_Year_Remod/Add - 1951 t62_leaf1 >= 0
 T62_L1_f0_le: x_Lot_Frontage + 153.530001 t62_leaf1 <= 175.53
 T62_L2_f20_le: x_Gr_Liv_Area + 1621.760001 t62_leaf2 <= 3421.76
 T62_R2_f6_ge: x_Year_Remod/Add - 1951 t62_leaf2 >= 0
 T62_R2_f0_ge: x_Lot_Frontage - 22 t62_leaf2 >= 0
 T62_R3_f20_ge: x_Gr_Liv_Area - 1800 t62_leaf3 >= 0
 T62_L3_f232_le: x_Central_Air_Y + 1e-06 t62_leaf3 <= 1
 T62_R4_f20_ge: x_Gr_Liv_Area - 1800 t62_leaf4 >= 0
 T62_R4_f232_ge: x_Central_Air_Y - t62_leaf4 >= 0
 TREE_63_ONEHOT: t63_leaf0 + t63_leaf1 + t63_leaf2 + t63_leaf3 + t63_leaf4
   + t63_leaf5 = 1
 T63_L0_f20_le: x_Gr_Liv_Area + 1877.760001 t63_leaf0 <= 3421.76
 T63_L0_f5_le: x_Year_Built + 998047.000001 t63_leaf0 <= 1e+06
 T63_L0_f8_le: x_Exter_Qual + 2.000001 t63_leaf0 <= 4
 T63_L1_f20_le: x_Gr_Liv_Area + 1877.760001 t63_leaf1 <= 3421.76
 T63_L1_f5_le: x_Year_Built + 998047.000001 t63_leaf1 <= 1e+06
 T63_R1_f8_ge: x_Exter_Qual >= 2
 T63_L2_f20_le: x_Gr_Liv_Area + 1877.760001 t63_leaf2 <= 3421.76
 T63_R2_f5_ge: x_Year_Built - 1953 t63_leaf2 >= 0
 T63_L2_f17_le: x_1st_Flr_SF + 1958.900001 t63_leaf2 <= 2534.9
 T63_L3_f20_le: x_Gr_Liv_Area + 1877.760001 t63_leaf3 <= 3421.76
 T63_R3_f5_ge: x_Year_Built - 1953 t63_leaf3 >= 0
 T63_R3_f17_ge: x_1st_Flr_SF - 576 t63_leaf3 >= 0
 T63_R4_f20_ge: x_Gr_Liv_Area - 1544 t63_leaf4 >= 0
 T63_L4_f3_le: x_Overall_Qual + 999992.000001 t63_leaf4 <= 1e+06
 T63_R5_f20_ge: x_Gr_Liv_Area - 1544 t63_leaf5 >= 0
 T63_R5_f3_ge: x_Overall_Qual - 8 t63_leaf5 >= 0
 TREE_64_ONEHOT: t64_leaf0 + t64_leaf1 + t64_leaf2 + t64_leaf3 + t64_leaf4
   + t64_leaf5 + t64_leaf6 = 1
 T64_L0_f20_le: x_Gr_Liv_Area + 2257.760001000001 t64_leaf0 <= 3421.76
 T64_L0_f4_le: x_Overall_Cond + 999995.000001 t64_leaf0 <= 1e+06
 T64_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t64_leaf0 <= 1
 T64_L1_f20_le: x_Gr_Liv_Area + 2257.760001000001 t64_leaf1 <= 3421.76
 T64_L1_f4_le: x_Overall_Cond + 999995.000001 t64_leaf1 <= 1e+06
 T64_R1_f297_ge: x_Sale_Condition_Normal - t64_leaf1 >= 0
 T64_L2_f20_le: x_Gr_Liv_Area + 2257.760001000001 t64_leaf2 <= 3421.76
 T64_R2_f4_ge: x_Overall_Cond - 5 t64_leaf2 >= 0
 T64_L2_f3_le: x_Overall_Qual + 999996.000001 t64_leaf2 <= 1e+06
 T64_L3_f20_le: x_Gr_Liv_Area + 2257.760001000001 t64_leaf3 <= 3421.76
 T64_R3_f4_ge: x_Overall_Cond - 5 t64_leaf3 >= 0
 T64_R3_f3_ge: x_Overall_Qual - 4 t64_leaf3 >= 0
 T64_R4_f20_ge: x_Gr_Liv_Area - 1164 t64_leaf4 >= 0
 T64_L4_f20_le: x_Gr_Liv_Area + 1466.760001 t64_leaf4 <= 3421.76
 T64_R5_f20_ge: x_Gr_Liv_Area - 1164 t64_leaf5 >= 0
 T64_R5_f20_ge: x_Gr_Liv_Area - 1955 t64_leaf5 >= 0
 T64_L5_f60_le: x_MS_SubClass_190 + 1e-06 t64_leaf5 <= 1
 T64_R6_f20_ge: x_Gr_Liv_Area - 1164 t64_leaf6 >= 0
 T64_R6_f20_ge: x_Gr_Liv_Area - 1955 t64_leaf6 >= 0
 T64_R6_f60_ge: x_MS_SubClass_190 - t64_leaf6 >= 0
 TREE_65_ONEHOT: t65_leaf0 + t65_leaf1 + t65_leaf2 + t65_leaf3 + t65_leaf4
   + t65_leaf5 = 1
 T65_L0_f15_le: x_Total_Bsmt_SF + 1327.480001 t65_leaf0 <= 2514.48
 T65_L0_f3_le: x_Overall_Qual + 999995.000001 t65_leaf0 <= 1e+06
 T65_L0_f8_le: x_Exter_Qual + 2.000001 t65_leaf0 <= 4
 T65_L1_f15_le: x_Total_Bsmt_SF + 1327.480001 t65_leaf1 <= 2514.48
 T65_L1_f3_le: x_Overall_Qual + 999995.000001 t65_leaf1 <= 1e+06
 T65_R1_f8_ge: x_Exter_Qual >= 2
 T65_L2_f15_le: x_Total_Bsmt_SF + 1327.480001 t65_leaf2 <= 2514.48
 T65_R2_f3_ge: x_Overall_Qual - 5 t65_leaf2 >= 0
 T65_L2_f67_le: x_MS_Zoning_RM + 1e-06 t65_leaf2 <= 1
 T65_L3_f15_le: x_Total_Bsmt_SF + 1327.480001 t65_leaf3 <= 2514.48
 T65_R3_f3_ge: x_Overall_Qual - 5 t65_leaf3 >= 0
 T65_R3_f67_ge: x_MS_Zoning_RM - t65_leaf3 >= 0
 T65_R4_f15_ge: x_Total_Bsmt_SF - 1187 t65_leaf4 >= 0
 T65_L4_f20_le: x_Gr_Liv_Area + 1597.760001 t65_leaf4 <= 3421.76
 T65_R5_f15_ge: x_Total_Bsmt_SF - 1187 t65_leaf5 >= 0
 T65_R5_f20_ge: x_Gr_Liv_Area - 1824 t65_leaf5 >= 0
 TREE_66_ONEHOT: t66_leaf0 + t66_leaf1 + t66_leaf2 + t66_leaf3 + t66_leaf4
   + t66_leaf5 + t66_leaf6 = 1
 T66_L0_f17_le: x_1st_Flr_SF + 1326.900001 t66_leaf0 <= 2534.9
 T66_L0_f46_le: x_MS_SubClass_30 + 1e-06 t66_leaf0 <= 1
 T66_L0_f35_le: x_Garage_Cond + 2.000001 t66_leaf0 <= 4
 T66_L1_f17_le: x_1st_Flr_SF + 1326.900001 t66_leaf1 <= 2534.9
 T66_L1_f46_le: x_MS_SubClass_30 + 1e-06 t66_leaf1 <= 1
 T66_R1_f35_ge: x_Garage_Cond - 3 t66_leaf1 >= -1
 T66_L2_f17_le: x_1st_Flr_SF + 1326.900001 t66_leaf2 <= 2534.9
 T66_R2_f46_ge: x_MS_SubClass_30 - t66_leaf2 >= 0
 T66_L2_f4_le: x_Overall_Cond + 999996.000001 t66_leaf2 <= 1e+06
 T66_L3_f17_le: x_1st_Flr_SF + 1326.900001 t66_leaf3 <= 2534.9
 T66_R3_f46_ge: x_MS_SubClass_30 - t66_leaf3 >= 0
 T66_R3_f4_ge: x_Overall_Cond - 4 t66_leaf3 >= 0
 T66_R4_f17_ge: x_1st_Flr_SF - 1208 t66_leaf4 >= 0
 T66_L4_f27_le: x_Kitchen_Qual + 1e-06 t66_leaf4 <= 4
 T66_L4_f4_le: x_Overall_Cond + 999996.000001 t66_leaf4 <= 1e+06
 T66_R5_f17_ge: x_1st_Flr_SF - 1208 t66_leaf5 >= 0
 T66_L5_f27_le: x_Kitchen_Qual + 1e-06 t66_leaf5 <= 4
 T66_R5_f4_ge: x_Overall_Cond - 4 t66_leaf5 >= 0
 T66_R6_f17_ge: x_1st_Flr_SF - 1208 t66_leaf6 >= 0
 T66_R6_f27_ge: x_Kitchen_Qual - 2 t66_leaf6 >= 2
 TREE_67_ONEHOT: t67_leaf0 + t67_leaf1 + t67_leaf2 + t67_leaf3 + t67_leaf4
   + t67_leaf5 + t67_leaf6 = 1
 T67_L0_f20_le: x_Gr_Liv_Area + 1681.760001 t67_leaf0 <= 3421.76
 T67_L0_f259_le: x_Paved_Drive_Y + 1e-06 t67_leaf0 <= 1
 T67_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t67_leaf0 <= 1
 T67_L1_f20_le: x_Gr_Liv_Area + 1681.760001 t67_leaf1 <= 3421.76
 T67_L1_f259_le: x_Paved_Drive_Y + 1e-06 t67_leaf1 <= 1
 T67_R1_f297_ge: x_Sale_Condition_Normal - t67_leaf1 >= 0
 T67_L2_f20_le: x_Gr_Liv_Area + 1681.760001 t67_leaf2 <= 3421.76
 T67_R2_f259_ge: x_Paved_Drive_Y - t67_leaf2 >= 0
 T67_L2_f247_le: x_Garage_Type_Attchd + 1e-06 t67_leaf2 <= 1
 T67_L3_f20_le: x_Gr_Liv_Area + 1681.760001 t67_leaf3 <= 3421.76
 T67_R3_f259_ge: x_Paved_Drive_Y - t67_leaf3 >= 0
 T67_R3_f247_ge: x_Garage_Type_Attchd - t67_leaf3 >= 0
 T67_R4_f20_ge: x_Gr_Liv_Area - 1740 t67_leaf4 >= 0
 T67_L4_f109_le: x_Neighborhood_OldTown + 1e-06 t67_leaf4 <= 1
 T67_R5_f20_ge: x_Gr_Liv_Area - 1740 t67_leaf5 >= 0
 T67_R5_f109_ge: x_Neighborhood_OldTown - t67_leaf5 >= 0
 T67_L5_f4_le: x_Overall_Cond + 999995.000001 t67_leaf5 <= 1e+06
 T67_R6_f20_ge: x_Gr_Liv_Area - 1740 t67_leaf6 >= 0
 T67_R6_f109_ge: x_Neighborhood_OldTown - t67_leaf6 >= 0
 T67_R6_f4_ge: x_Overall_Cond - 5 t67_leaf6 >= 0
 TREE_68_ONEHOT: t68_leaf0 + t68_leaf1 + t68_leaf2 + t68_leaf3 + t68_leaf4
   + t68_leaf5 = 1
 T68_L0_f29_le: x_Fireplaces + 2.000001 t68_leaf0 <= 3
 T68_L0_f5_le: x_Year_Built + 998050.000001 t68_leaf0 <= 1e+06
 T68_L0_f0_le: x_Lot_Frontage + 74.530001 t68_leaf0 <= 175.53
 T68_L1_f29_le: x_Fireplaces + 2.000001 t68_leaf1 <= 3
 T68_L1_f5_le: x_Year_Built + 998050.000001 t68_leaf1 <= 1e+06
 T68_R1_f0_ge: x_Lot_Frontage - 101 t68_leaf1 >= 0
 T68_L2_f29_le: x_Fireplaces + 2.000001 t68_leaf2 <= 3
 T68_R2_f5_ge: x_Year_Built - 1950 t68_leaf2 >= 0
 T68_R3_f29_ge: x_Fireplaces - t68_leaf3 >= 0
 T68_L3_f3_le: x_Overall_Qual + 999991.000001 t68_leaf3 <= 1e+06
 T68_L3_f15_le: x_Total_Bsmt_SF + 1563.480001 t68_leaf3 <= 2514.48
 T68_R4_f29_ge: x_Fireplaces - t68_leaf4 >= 0
 T68_L4_f3_le: x_Overall_Qual + 999991.000001 t68_leaf4 <= 1e+06
 T68_R4_f15_ge: x_Total_Bsmt_SF - 951 t68_leaf4 >= 0
 T68_R5_f29_ge: x_Fireplaces - t68_leaf5 >= 0
 T68_R5_f3_ge: x_Overall_Qual - 9 t68_leaf5 >= 0
 TREE_69_ONEHOT: t69_leaf0 + t69_leaf1 + t69_leaf2 + t69_leaf3 + t69_leaf4
   + t69_leaf5 = 1
 T69_L0_f15_le: x_Total_Bsmt_SF + 1274.480001 t69_leaf0 <= 2514.48
 T69_L0_f46_le: x_MS_SubClass_30 + 1e-06 t69_leaf0 <= 1
 T69_L0_f1_le: x_Lot_Area <= 7668.999999
 T69_L1_f15_le: x_Total_Bsmt_SF + 1274.480001 t69_leaf1 <= 2514.48
 T69_L1_f46_le: x_MS_SubClass_30 + 1e-06 t69_leaf1 <= 1
 T69_R1_f1_ge: x_Lot_Area - 669 t69_leaf1 >= 7000
 T69_L2_f15_le: x_Total_Bsmt_SF + 1274.480001 t69_leaf2 <= 2514.48
 T69_R2_f46_ge: x_MS_SubClass_30 - t69_leaf2 >= 0
 T69_L2_f1_le: x_Lot_Area + 2730.000001 t69_leaf2 <= 7000
 T69_L3_f15_le: x_Total_Bsmt_SF + 1274.480001 t69_leaf3 <= 2514.48
 T69_R3_f46_ge: x_MS_SubClass_30 - t69_leaf3 >= 0
 T69_R3_f1_ge: x_Lot_Area >= 4270
 T69_R4_f15_ge: x_Total_Bsmt_SF - 1240 t69_leaf4 >= 0
 T69_L4_f27_le: x_Kitchen_Qual + 1e-06 t69_leaf4 <= 4
 T69_R5_f15_ge: x_Total_Bsmt_SF - 1240 t69_leaf5 >= 0
 T69_R5_f27_ge: x_Kitchen_Qual - 2 t69_leaf5 >= 2
 TREE_70_ONEHOT: t70_leaf0 + t70_leaf1 + t70_leaf2 + t70_leaf3 + t70_leaf4
   + t70_leaf5 = 1
 T70_L0_f30_le: x_Fireplace_Qu + 3.000001 t70_leaf0 <= 4
 T70_L0_f259_le: x_Paved_Drive_Y + 1e-06 t70_leaf0 <= 1
 T70_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t70_leaf0 <= 1
 T70_L1_f30_le: x_Fireplace_Qu + 3.000001 t70_leaf1 <= 4
 T70_L1_f259_le: x_Paved_Drive_Y + 1e-06 t70_leaf1 <= 1
 T70_R1_f293_ge: x_Sale_Condition_Abnorml - t70_leaf1 >= 0
 T70_L2_f30_le: x_Fireplace_Qu + 3.000001 t70_leaf2 <= 4
 T70_R2_f259_ge: x_Paved_Drive_Y - t70_leaf2 >= 0
 T70_L2_f3_le: x_Overall_Qual + 999996.000001 t70_leaf2 <= 1e+06
 T70_L3_f30_le: x_Fireplace_Qu + 3.000001 t70_leaf3 <= 4
 T70_R3_f259_ge: x_Paved_Drive_Y - t70_leaf3 >= 0
 T70_R3_f3_ge: x_Overall_Qual - 4 t70_leaf3 >= 0
 T70_R4_f30_ge: x_Fireplace_Qu >= 1
 T70_L4_f10_le: x_Bsmt_Qual + 1e-06 t70_leaf4 <= 4
 T70_R5_f30_ge: x_Fireplace_Qu >= 1
 T70_R5_f10_ge: x_Bsmt_Qual - 2 t70_leaf5 >= 2
 TREE_71_ONEHOT: t71_leaf0 + t71_leaf1 + t71_leaf2 + t71_leaf3 + t71_leaf4
   + t71_leaf5 + t71_leaf6 = 1
 T71_L0_f20_le: x_Gr_Liv_Area + 1453.760001 t71_leaf0 <= 3421.76
 T71_L0_f5_le: x_Year_Built + 998044.000001 t71_leaf0 <= 1e+06
 T71_L0_f20_le: x_Gr_Liv_Area + 2573.760001 t71_leaf0 <= 3421.76
 T71_L1_f20_le: x_Gr_Liv_Area + 1453.760001 t71_leaf1 <= 3421.76
 T71_L1_f5_le: x_Year_Built + 998044.000001 t71_leaf1 <= 1e+06
 T71_R1_f20_ge: x_Gr_Liv_Area - 848 t71_leaf1 >= 0
 T71_L2_f20_le: x_Gr_Liv_Area + 1453.760001 t71_leaf2 <= 3421.76
 T71_R2_f5_ge: x_Year_Built - 1956 t71_leaf2 >= 0
 T71_L2_f1_le: x_Lot_Area + 4355.000001 t71_leaf2 <= 7000
 T71_L3_f20_le: x_Gr_Liv_Area + 1453.760001 t71_leaf3 <= 3421.76
 T71_R3_f5_ge: x_Year_Built - 1956 t71_leaf3 >= 0
 T71_R3_f1_ge: x_Lot_Area >= 2645
 T71_R4_f20_ge: x_Gr_Liv_Area - 1968 t71_leaf4 >= 0
 T71_L4_f1_le: x_Lot_Area <= 7063.999999
 T71_R5_f20_ge: x_Gr_Liv_Area - 1968 t71_leaf5 >= 0
 T71_R5_f1_ge: x_Lot_Area - 64 t71_leaf5 >= 7000
 T71_L5_f70_le: x_Alley_Grvl + 1e-06 t71_leaf5 <= 1
 T71_R6_f20_ge: x_Gr_Liv_Area - 1968 t71_leaf6 >= 0
 T71_R6_f1_ge: x_Lot_Area - 64 t71_leaf6 >= 7000
 T71_R6_f70_ge: x_Alley_Grvl - t71_leaf6 >= 0
 TREE_72_ONEHOT: t72_leaf0 + t72_leaf1 + t72_leaf2 + t72_leaf3 + t72_leaf4
   + t72_leaf5 + t72_leaf6 = 1
 T72_L0_f20_le: x_Gr_Liv_Area + 1655.760001 t72_leaf0 <= 3421.76
 T72_L0_f4_le: x_Overall_Cond + 999995.000001 t72_leaf0 <= 1e+06
 T72_L0_f20_le: x_Gr_Liv_Area + 2545.760001 t72_leaf0 <= 3421.76
 T72_L1_f20_le: x_Gr_Liv_Area + 1655.760001 t72_leaf1 <= 3421.76
 T72_L1_f4_le: x_Overall_Cond + 999995.000001 t72_leaf1 <= 1e+06
 T72_R1_f20_ge: x_Gr_Liv_Area - 876 t72_leaf1 >= 0
 T72_L2_f20_le: x_Gr_Liv_Area + 1655.760001 t72_leaf2 <= 3421.76
 T72_R2_f4_ge: x_Overall_Cond - 5 t72_leaf2 >= 0
 T72_L2_f21_le: x_Bsmt_Full_Bath + 1.020001 t72_leaf2 <= 2.02
 T72_L3_f20_le: x_Gr_Liv_Area + 1655.760001 t72_leaf3 <= 3421.76
 T72_R3_f4_ge: x_Overall_Cond - 5 t72_leaf3 >= 0
 T72_R3_f21_ge: x_Bsmt_Full_Bath - t72_leaf3 >= 0
 T72_R4_f20_ge: x_Gr_Liv_Area - 1766 t72_leaf4 >= 0
 T72_L4_f4_le: x_Overall_Cond + 999996.000001 t72_leaf4 <= 1e+06
 T72_R5_f20_ge: x_Gr_Liv_Area - 1766 t72_leaf5 >= 0
 T72_R5_f4_ge: x_Overall_Cond - 4 t72_leaf5 >= 0
 T72_L5_f1_le: x_Lot_Area <= 11387.999999
 T72_R6_f20_ge: x_Gr_Liv_Area - 1766 t72_leaf6 >= 0
 T72_R6_f4_ge: x_Overall_Cond - 4 t72_leaf6 >= 0
 T72_R6_f1_ge: x_Lot_Area - 4388 t72_leaf6 >= 7000
 TREE_73_ONEHOT: t73_leaf0 + t73_leaf1 + t73_leaf2 + t73_leaf3 + t73_leaf4
   + t73_leaf5 = 1
 T73_L0_f12_le: x_BsmtFin_SF_1 + 1305.500001 t73_leaf0 <= 1984.5
 T73_L0_f46_le: x_MS_SubClass_30 + 1e-06 t73_leaf0 <= 1
 T73_L0_f27_le: x_Kitchen_Qual + 1.000001 t73_leaf0 <= 4
 T73_L1_f12_le: x_BsmtFin_SF_1 + 1305.500001 t73_leaf1 <= 1984.5
 T73_L1_f46_le: x_MS_SubClass_30 + 1e-06 t73_leaf1 <= 1
 T73_R1_f27_ge: x_Kitchen_Qual - t73_leaf1 >= 2
 T73_L2_f12_le: x_BsmtFin_SF_1 + 1305.500001 t73_leaf2 <= 1984.5
 T73_R2_f46_ge: x_MS_SubClass_30 - t73_leaf2 >= 0
 T73_L2_f257_le: x_Paved_Drive_N + 1e-06 t73_leaf2 <= 1
 T73_L3_f12_le: x_BsmtFin_SF_1 + 1305.500001 t73_leaf3 <= 1984.5
 T73_R3_f46_ge: x_MS_SubClass_30 - t73_leaf3 >= 0
 T73_R3_f257_ge: x_Paved_Drive_N - t73_leaf3 >= 0
 T73_R4_f12_ge: x_BsmtFin_SF_1 - 679 t73_leaf4 >= 0
 T73_L4_f12_le: x_BsmtFin_SF_1 + 874.5000010000001 t73_leaf4 <= 1984.5
 T73_R5_f12_ge: x_BsmtFin_SF_1 - 679 t73_leaf5 >= 0
 T73_R5_f12_ge: x_BsmtFin_SF_1 - 1110 t73_leaf5 >= 0
 TREE_74_ONEHOT: t74_leaf0 + t74_leaf1 + t74_leaf2 + t74_leaf3 + t74_leaf4
   + t74_leaf5 = 1
 T74_L0_f3_le: x_Overall_Qual + 999992.000001 t74_leaf0 <= 1e+06
 T74_L0_f231_le: x_Central_Air_N + 1e-06 t74_leaf0 <= 1
 T74_L0_f17_le: x_1st_Flr_SF + 1558.900001 t74_leaf0 <= 2534.9
 T74_L1_f3_le: x_Overall_Qual + 999992.000001 t74_leaf1 <= 1e+06
 T74_L1_f231_le: x_Central_Air_N + 1e-06 t74_leaf1 <= 1
 T74_R1_f17_ge: x_1st_Flr_SF - 976 t74_leaf1 >= 0
 T74_L2_f3_le: x_Overall_Qual + 999992.000001 t74_leaf2 <= 1e+06
 T74_R2_f231_ge: x_Central_Air_N - t74_leaf2 >= 0
 T74_L2_f4_le: x_Overall_Cond + 999996.000001 t74_leaf2 <= 1e+06
 T74_L3_f3_le: x_Overall_Qual + 999992.000001 t74_leaf3 <= 1e+06
 T74_R3_f231_ge: x_Central_Air_N - t74_leaf3 >= 0
 T74_R3_f4_ge: x_Overall_Cond - 4 t74_leaf3 >= 0
 T74_R4_f3_ge: x_Overall_Qual - 8 t74_leaf4 >= 0
 T74_L4_f17_le: x_1st_Flr_SF + 1152.900001 t74_leaf4 <= 2534.9
 T74_R5_f3_ge: x_Overall_Qual - 8 t74_leaf5 >= 0
 T74_R5_f17_ge: x_1st_Flr_SF - 1382 t74_leaf5 >= 0
 TREE_75_ONEHOT: t75_leaf0 + t75_leaf1 + t75_leaf2 + t75_leaf3 + t75_leaf4
   + t75_leaf5 + t75_leaf6 = 1
 T75_L0_f20_le: x_Gr_Liv_Area + 1709.760001 t75_leaf0 <= 3421.76
 T75_L0_f257_le: x_Paved_Drive_N + 1e-06 t75_leaf0 <= 1
 T75_L0_f15_le: x_Total_Bsmt_SF + 1866.480001 t75_leaf0 <= 2514.48
 T75_L1_f20_le: x_Gr_Liv_Area + 1709.760001 t75_leaf1 <= 3421.76
 T75_L1_f257_le: x_Paved_Drive_N + 1e-06 t75_leaf1 <= 1
 T75_R1_f15_ge: x_Total_Bsmt_SF - 648 t75_leaf1 >= 0
 T75_L2_f20_le: x_Gr_Liv_Area + 1709.760001 t75_leaf2 <= 3421.76
 T75_R2_f257_ge: x_Paved_Drive_N - t75_leaf2 >= 0
 T75_L2_f9_le: x_Exter_Cond + 2.000001 t75_leaf2 <= 4
 T75_L3_f20_le: x_Gr_Liv_Area + 1709.760001 t75_leaf3 <= 3421.76
 T75_R3_f257_ge: x_Paved_Drive_N - t75_leaf3 >= 0
 T75_R3_f9_ge: x_Exter_Cond - 3 t75_leaf3 >= -1
 T75_R4_f20_ge: x_Gr_Liv_Area - 1712 t75_leaf4 >= 0
 T75_L4_f6_le: x_Year_Remod/Add + 998017.000001 t75_leaf4 <= 1e+06
 T75_L4_f5_le: x_Year_Built + 998089.000001 t75_leaf4 <= 1e+06
 T75_R5_f20_ge: x_Gr_Liv_Area - 1712 t75_leaf5 >= 0
 T75_L5_f6_le: x_Year_Remod/Add + 998017.000001 t75_leaf5 <= 1e+06
 T75_R5_f5_ge: x_Year_Built - 1911 t75_leaf5 >= 0
 T75_R6_f20_ge: x_Gr_Liv_Area - 1712 t75_leaf6 >= 0
 T75_R6_f6_ge: x_Year_Remod/Add - 1983 t75_leaf6 >= 0
 TREE_76_ONEHOT: t76_leaf0 + t76_leaf1 + t76_leaf2 + t76_leaf3 + t76_leaf4
   + t76_leaf5 = 1
 T76_L0_f1_le: x_Lot_Area <= 11979.999999
 T76_L0_f5_le: x_Year_Built + 998015.000001 t76_leaf0 <= 1e+06
 T76_L0_f217_le: x_BsmtFin_Type_1_Unf + 1e-06 t76_leaf0 <= 1
 T76_L1_f1_le: x_Lot_Area <= 11979.999999
 T76_L1_f5_le: x_Year_Built + 998015.000001 t76_leaf1 <= 1e+06
 T76_R1_f217_ge: x_BsmtFin_Type_1_Unf - t76_leaf1 >= 0
 T76_L2_f1_le: x_Lot_Area <= 11979.999999
 T76_R2_f5_ge: x_Year_Built - 1985 t76_leaf2 >= 0
 T76_R3_f1_ge: x_Lot_Area - 4980 t76_leaf3 >= 7000
 T76_L3_f11_le: x_Bsmt_Cond + 4.000001 t76_leaf3 <= 4
 T76_R4_f1_ge: x_Lot_Area - 4980 t76_leaf4 >= 7000
 T76_R4_f11_ge: x_Bsmt_Cond - t76_leaf4 >= -1
 T76_L4_f24_le: x_Half_Bath + 3.000001 t76_leaf4 <= 5
 T76_R5_f1_ge: x_Lot_Area - 4980 t76_leaf5 >= 7000
 T76_R5_f11_ge: x_Bsmt_Cond - t76_leaf5 >= -1
 T76_R5_f24_ge: x_Half_Bath - 2 t76_leaf5 >= 0
 TREE_77_ONEHOT: t77_leaf0 + t77_leaf1 + t77_leaf2 + t77_leaf3 + t77_leaf4
   + t77_leaf5 + t77_leaf6 + t77_leaf7 = 1
 T77_L0_f6_le: x_Year_Remod/Add + 998016.000001 t77_leaf0 <= 1e+06
 T77_L0_f20_le: x_Gr_Liv_Area + 2538.760001 t77_leaf0 <= 3421.76
 T77_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t77_leaf0 <= 1
 T77_L1_f6_le: x_Year_Remod/Add + 998016.000001 t77_leaf1 <= 1e+06
 T77_L1_f20_le: x_Gr_Liv_Area + 2538.760001 t77_leaf1 <= 3421.76
 T77_R1_f161_ge: x_Exterior_1st_AsbShng - t77_leaf1 >= 0
 T77_L2_f6_le: x_Year_Remod/Add + 998016.000001 t77_leaf2 <= 1e+06
 T77_R2_f20_ge: x_Gr_Liv_Area - 883 t77_leaf2 >= 0
 T77_L2_f12_le: x_BsmtFin_SF_1 + 1542.500001 t77_leaf2 <= 1984.5
 T77_L3_f6_le: x_Year_Remod/Add + 998016.000001 t77_leaf3 <= 1e+06
 T77_R3_f20_ge: x_Gr_Liv_Area - 883 t77_leaf3 >= 0
 T77_R3_f12_ge: x_BsmtFin_SF_1 - 442 t77_leaf3 >= 0
 T77_R4_f6_ge: x_Year_Remod/Add - 1984 t77_leaf4 >= 0
 T77_L4_f20_le: x_Gr_Liv_Area + 1695.760001 t77_leaf4 <= 3421.76
 T77_L4_f293_le: x_Sale_Condition_Abnorml + 1e-06 t77_leaf4 <= 1
 T77_R5_f6_ge: x_Year_Remod/Add - 1984 t77_leaf5 >= 0
 T77_L5_f20_le: x_Gr_Liv_Area + 1695.760001 t77_leaf5 <= 3421.76
 T77_R5_f293_ge: x_Sale_Condition_Abnorml - t77_leaf5 >= 0
 T77_R6_f6_ge: x_Year_Remod/Add - 1984 t77_leaf6 >= 0
 T77_R6_f20_ge: x_Gr_Liv_Area - 1726 t77_leaf6 >= 0
 T77_L6_f259_le: x_Paved_Drive_Y + 1e-06 t77_leaf6 <= 1
 T77_R7_f6_ge: x_Year_Remod/Add - 1984 t77_leaf7 >= 0
 T77_R7_f20_ge: x_Gr_Liv_Area - 1726 t77_leaf7 >= 0
 T77_R7_f259_ge: x_Paved_Drive_Y - t77_leaf7 >= 0
 TREE_78_ONEHOT: t78_leaf0 + t78_leaf1 + t78_leaf2 + t78_leaf3 + t78_leaf4
   + t78_leaf5 = 1
 T78_L0_f15_le: x_Total_Bsmt_SF + 1237.480001 t78_leaf0 <= 2514.48
 T78_L0_f66_le: x_MS_Zoning_RL + 1e-06 t78_leaf0 <= 1
 T78_L0_f226_le: x_Heating_GasA + 1e-06 t78_leaf0 <= 1
 T78_L1_f15_le: x_Total_Bsmt_SF + 1237.480001 t78_leaf1 <= 2514.48
 T78_L1_f66_le: x_MS_Zoning_RL + 1e-06 t78_leaf1 <= 1
 T78_R1_f226_ge: x_Heating_GasA - t78_leaf1 >= 0
 T78_L2_f15_le: x_Total_Bsmt_SF + 1237.480001 t78_leaf2 <= 2514.48
 T78_R2_f66_ge: x_MS_Zoning_RL - t78_leaf2 >= 0
 T78_L2_f11_le: x_Bsmt_Cond + 2.000001 t78_leaf2 <= 4
 T78_L3_f15_le: x_Total_Bsmt_SF + 1237.480001 t78_leaf3 <= 2514.48
 T78_R3_f66_ge: x_MS_Zoning_RL - t78_leaf3 >= 0
 T78_R3_f11_ge: x_Bsmt_Cond - 3 t78_leaf3 >= -1
 T78_R4_f15_ge: x_Total_Bsmt_SF - 1277 t78_leaf4 >= 0
 T78_L4_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t78_leaf4 <= 1984.5
 T78_R5_f15_ge: x_Total_Bsmt_SF - 1277 t78_leaf5 >= 0
 T78_R5_f12_ge: x_BsmtFin_SF_1 - 1087 t78_leaf5 >= 0
 TREE_79_ONEHOT: t79_leaf0 + t79_leaf1 + t79_leaf2 + t79_leaf3 + t79_leaf4
   + t79_leaf5 + t79_leaf6 = 1
 T79_L0_f6_le: x_Year_Remod/Add + 998016.000001 t79_leaf0 <= 1e+06
 T79_L0_f66_le: x_MS_Zoning_RL + 1e-06 t79_leaf0 <= 1
 T79_L0_f4_le: x_Overall_Cond + 999996.000001 t79_leaf0 <= 1e+06
 T79_L1_f6_le: x_Year_Remod/Add + 998016.000001 t79_leaf1 <= 1e+06
 T79_L1_f66_le: x_MS_Zoning_RL + 1e-06 t79_leaf1 <= 1
 T79_R1_f4_ge: x_Overall_Cond - 4 t79_leaf1 >= 0
 T79_L2_f6_le: x_Year_Remod/Add + 998016.000001 t79_leaf2 <= 1e+06
 T79_R2_f66_ge: x_MS_Zoning_RL - t79_leaf2 >= 0
 T79_L2_f217_le: x_BsmtFin_Type_1_Unf + 1e-06 t79_leaf2 <= 1
 T79_L3_f6_le: x_Year_Remod/Add + 998016.000001 t79_leaf3 <= 1e+06
 T79_R3_f66_ge: x_MS_Zoning_RL - t79_leaf3 >= 0
 T79_R3_f217_ge: x_BsmtFin_Type_1_Unf - t79_leaf3 >= 0
 T79_R4_f6_ge: x_Year_Remod/Add - 1984 t79_leaf4 >= 0
 T79_L4_f32_le: x_Garage_Cars + 1.000001 t79_leaf4 <= 4
 T79_L4_f10_le: x_Bsmt_Qual + 4.000001 t79_leaf4 <= 4
 T79_R5_f6_ge: x_Year_Remod/Add - 1984 t79_leaf5 >= 0
 T79_L5_f32_le: x_Garage_Cars + 1.000001 t79_leaf5 <= 4
 T79_R5_f10_ge: x_Bsmt_Qual >= 0
 T79_R6_f6_ge: x_Year_Remod/Add - 1984 t79_leaf6 >= 0
 T79_R6_f32_ge: x_Garage_Cars - 3 t79_leaf6 >= 0
 TREE_80_ONEHOT: t80_leaf0 + t80_leaf1 + t80_leaf2 + t80_leaf3 + t80_leaf4
   + t80_leaf5 + t80_leaf6 = 1
 T80_L0_f15_le: x_Total_Bsmt_SF + 1095.480001 t80_leaf0 <= 2514.48
 T80_L0_f4_le: x_Overall_Cond + 999995.000001 t80_leaf0 <= 1e+06
 T80_L0_f109_le: x_Neighborhood_OldTown + 1e-06 t80_leaf0 <= 1
 T80_L1_f15_le: x_Total_Bsmt_SF + 1095.480001 t80_leaf1 <= 2514.48
 T80_L1_f4_le: x_Overall_Cond + 999995.000001 t80_leaf1 <= 1e+06
 T80_R1_f109_ge: x_Neighborhood_OldTown - t80_leaf1 >= 0
 T80_L2_f15_le: x_Total_Bsmt_SF + 1095.480001 t80_leaf2 <= 2514.48
 T80_R2_f4_ge: x_Overall_Cond - 5 t80_leaf2 >= 0
 T80_L2_f20_le: x_Gr_Liv_Area + 1621.760001 t80_leaf2 <= 3421.76
 T80_L3_f15_le: x_Total_Bsmt_SF + 1095.480001 t80_leaf3 <= 2514.48
 T80_R3_f4_ge: x_Overall_Cond - 5 t80_leaf3 >= 0
 T80_R3_f20_ge: x_Gr_Liv_Area - 1800 t80_leaf3 >= 0
 T80_R4_f15_ge: x_Total_Bsmt_SF - 1419 t80_leaf4 >= 0
 T80_L4_f20_le: x_Gr_Liv_Area + 1650.760001 t80_leaf4 <= 3421.76
 T80_L4_f5_le: x_Year_Built + 998043.000001 t80_leaf4 <= 1e+06
 T80_R5_f15_ge: x_Total_Bsmt_SF - 1419 t80_leaf5 >= 0
 T80_L5_f20_le: x_Gr_Liv_Area + 1650.760001 t80_leaf5 <= 3421.76
 T80_R5_f5_ge: x_Year_Built - 1957 t80_leaf5 >= 0
 T80_R6_f15_ge: x_Total_Bsmt_SF - 1419 t80_leaf6 >= 0
 T80_R6_f20_ge: x_Gr_Liv_Area - 1771 t80_leaf6 >= 0
 TREE_81_ONEHOT: t81_leaf0 + t81_leaf1 + t81_leaf2 + t81_leaf3 + t81_leaf4
   + t81_leaf5 + t81_leaf6 = 1
 T81_L0_f15_le: x_Total_Bsmt_SF + 1274.480001 t81_leaf0 <= 2514.48
 T81_L0_f0_le: x_Lot_Frontage + 114.530001 t81_leaf0 <= 175.53
 T81_L0_f20_le: x_Gr_Liv_Area + 2433.760001 t81_leaf0 <= 3421.76
 T81_L1_f15_le: x_Total_Bsmt_SF + 1274.480001 t81_leaf1 <= 2514.48
 T81_L1_f0_le: x_Lot_Frontage + 114.530001 t81_leaf1 <= 175.53
 T81_R1_f20_ge: x_Gr_Liv_Area - 988 t81_leaf1 >= 0
 T81_L2_f15_le: x_Total_Bsmt_SF + 1274.480001 t81_leaf2 <= 2514.48
 T81_R2_f0_ge: x_Lot_Frontage - 61 t81_leaf2 >= 0
 T81_L2_f231_le: x_Central_Air_N + 1e-06 t81_leaf2 <= 1
 T81_L3_f15_le: x_Total_Bsmt_SF + 1274.480001 t81_leaf3 <= 2514.48
 T81_R3_f0_ge: x_Lot_Frontage - 61 t81_leaf3 >= 0
 T81_R3_f231_ge: x_Central_Air_N - t81_leaf3 >= 0
 T81_R4_f15_ge: x_Total_Bsmt_SF - 1240 t81_leaf4 >= 0
 T81_L4_f20_le: x_Gr_Liv_Area + 1555.760001 t81_leaf4 <= 3421.76
 T81_L4_f12_le: x_BsmtFin_SF_1 + 1612.500001 t81_leaf4 <= 1984.5
 T81_R5_f15_ge: x_Total_Bsmt_SF - 1240 t81_leaf5 >= 0
 T81_L5_f20_le: x_Gr_Liv_Area + 1555.760001 t81_leaf5 <= 3421.76
 T81_R5_f12_ge: x_BsmtFin_SF_1 - 372 t81_leaf5 >= 0
 T81_R6_f15_ge: x_Total_Bsmt_SF - 1240 t81_leaf6 >= 0
 T81_R6_f20_ge: x_Gr_Liv_Area - 1866 t81_leaf6 >= 0
 TREE_82_ONEHOT: t82_leaf0 + t82_leaf1 + t82_leaf2 + t82_leaf3 + t82_leaf4
   + t82_leaf5 = 1
 T82_L0_f20_le: x_Gr_Liv_Area + 1499.760001 t82_leaf0 <= 3421.76
 T82_L0_f247_le: x_Garage_Type_Attchd + 1e-06 t82_leaf0 <= 1
 T82_L0_f20_le: x_Gr_Liv_Area + 2653.760001 t82_leaf0 <= 3421.76
 T82_L1_f20_le: x_Gr_Liv_Area + 1499.760001 t82_leaf1 <= 3421.76
 T82_L1_f247_le: x_Garage_Type_Attchd + 1e-06 t82_leaf1 <= 1
 T82_R1_f20_ge: x_Gr_Liv_Area - 768 t82_leaf1 >= 0
 T82_L2_f20_le: x_Gr_Liv_Area + 1499.760001 t82_leaf2 <= 3421.76
 T82_R2_f247_ge: x_Garage_Type_Attchd - t82_leaf2 >= 0
 T82_L2_f15_le: x_Total_Bsmt_SF + 1839.480001 t82_leaf2 <= 2514.48
 T82_L3_f20_le: x_Gr_Liv_Area + 1499.760001 t82_leaf3 <= 3421.76
 T82_R3_f247_ge: x_Garage_Type_Attchd - t82_leaf3 >= 0
 T82_R3_f15_ge: x_Total_Bsmt_SF - 675 t82_leaf3 >= 0
 T82_R4_f20_ge: x_Gr_Liv_Area - 1922 t82_leaf4 >= 0
 T82_L4_f35_le: x_Garage_Cond + 3.000001 t82_leaf4 <= 4
 T82_R5_f20_ge: x_Gr_Liv_Area - 1922 t82_leaf5 >= 0
 T82_R5_f35_ge: x_Garage_Cond - 2 t82_leaf5 >= -1
 TREE_83_ONEHOT: t83_leaf0 + t83_leaf1 + t83_leaf2 + t83_leaf3 + t83_leaf4
   + t83_leaf5 + t83_leaf6 = 1
 T83_L0_f17_le: x_1st_Flr_SF + 1434.900001 t83_leaf0 <= 2534.9
 T83_L0_f9_le: x_Exter_Cond + 2.000001 t83_leaf0 <= 4
 T83_L0_f0_le: x_Lot_Frontage + 111.530001 t83_leaf0 <= 175.53
 T83_L1_f17_le: x_1st_Flr_SF + 1434.900001 t83_leaf1 <= 2534.9
 T83_L1_f9_le: x_Exter_Cond + 2.000001 t83_leaf1 <= 4
 T83_R1_f0_ge: x_Lot_Frontage - 64 t83_leaf1 >= 0
 T83_L2_f17_le: x_1st_Flr_SF + 1434.900001 t83_leaf2 <= 2534.9
 T83_R2_f9_ge: x_Exter_Cond - 3 t83_leaf2 >= -1
 T83_L2_f1_le: x_Lot_Area + 1000.000001 t83_leaf2 <= 7000
 T83_L3_f17_le: x_1st_Flr_SF + 1434.900001 t83_leaf3 <= 2534.9
 T83_R3_f9_ge: x_Exter_Cond - 3 t83_leaf3 >= -1
 T83_R3_f1_ge: x_Lot_Area >= 6000
 T83_R4_f17_ge: x_1st_Flr_SF - 1100 t83_leaf4 >= 0
 T83_L4_f27_le: x_Kitchen_Qual + 1e-06 t83_leaf4 <= 4
 T83_L4_f10_le: x_Bsmt_Qual + 2.000001 t83_leaf4 <= 4
 T83_R5_f17_ge: x_1st_Flr_SF - 1100 t83_leaf5 >= 0
 T83_L5_f27_le: x_Kitchen_Qual + 1e-06 t83_leaf5 <= 4
 T83_R5_f10_ge: x_Bsmt_Qual >= 2
 T83_R6_f17_ge: x_1st_Flr_SF - 1100 t83_leaf6 >= 0
 T83_R6_f27_ge: x_Kitchen_Qual - 2 t83_leaf6 >= 2
 TREE_84_ONEHOT: t84_leaf0 + t84_leaf1 + t84_leaf2 + t84_leaf3 + t84_leaf4
   + t84_leaf5 + t84_leaf6 + t84_leaf7 = 1
 T84_L0_f27_le: x_Kitchen_Qual + 1.000001 t84_leaf0 <= 4
 T84_L0_f15_le: x_Total_Bsmt_SF + 1903.480001 t84_leaf0 <= 2514.48
 T84_L0_f17_le: x_1st_Flr_SF + 2008.900001 t84_leaf0 <= 2534.9
 T84_L1_f27_le: x_Kitchen_Qual + 1.000001 t84_leaf1 <= 4
 T84_L1_f15_le: x_Total_Bsmt_SF + 1903.480001 t84_leaf1 <= 2514.48
 T84_R1_f17_ge: x_1st_Flr_SF - 526 t84_leaf1 >= 0
 T84_L2_f27_le: x_Kitchen_Qual + 1.000001 t84_leaf2 <= 4
 T84_R2_f15_ge: x_Total_Bsmt_SF - 611 t84_leaf2 >= 0
 T84_L2_f4_le: x_Overall_Cond + 999996.000001 t84_leaf2 <= 1e+06
 T84_L3_f27_le: x_Kitchen_Qual + 1.000001 t84_leaf3 <= 4
 T84_R3_f15_ge: x_Total_Bsmt_SF - 611 t84_leaf3 >= 0
 T84_R3_f4_ge: x_Overall_Cond - 4 t84_leaf3 >= 0
 T84_R4_f27_ge: x_Kitchen_Qual - t84_leaf4 >= 2
 T84_L4_f17_le: x_1st_Flr_SF + 894.9000010000002 t84_leaf4 <= 2534.9
 T84_L4_f1_le: x_Lot_Area + 760.0000010000003 t84_leaf4 <= 7000
 T84_R5_f27_ge: x_Kitchen_Qual - t84_leaf5 >= 2
 T84_L5_f17_le: x_1st_Flr_SF + 894.9000010000002 t84_leaf5 <= 2534.9
 T84_R5_f1_ge: x_Lot_Area >= 6240
 T84_R6_f27_ge: x_Kitchen_Qual - t84_leaf6 >= 2
 T84_R6_f17_ge: x_1st_Flr_SF - 1640 t84_leaf6 >= 0
 T84_L6_f0_le: x_Lot_Frontage + 55.530001 t84_leaf6 <= 175.53
 T84_R7_f27_ge: x_Kitchen_Qual - t84_leaf7 >= 2
 T84_R7_f17_ge: x_1st_Flr_SF - 1640 t84_leaf7 >= 0
 T84_R7_f0_ge: x_Lot_Frontage - 120 t84_leaf7 >= 0
 TREE_85_ONEHOT: t85_leaf0 + t85_leaf1 + t85_leaf2 + t85_leaf3 + t85_leaf4
   + t85_leaf5 + t85_leaf6 + t85_leaf7 = 1
 T85_L0_f30_le: x_Fireplace_Qu + 3.000001 t85_leaf0 <= 4
 T85_L0_f6_le: x_Year_Remod/Add + 998038.000001 t85_leaf0 <= 1e+06
 T85_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t85_leaf0 <= 1
 T85_L1_f30_le: x_Fireplace_Qu + 3.000001 t85_leaf1 <= 4
 T85_L1_f6_le: x_Year_Remod/Add + 998038.000001 t85_leaf1 <= 1e+06
 T85_R1_f297_ge: x_Sale_Condition_Normal - t85_leaf1 >= 0
 T85_L2_f30_le: x_Fireplace_Qu + 3.000001 t85_leaf2 <= 4
 T85_R2_f6_ge: x_Year_Remod/Add - 1962 t85_leaf2 >= 0
 T85_L2_f0_le: x_Lot_Frontage + 143.530001 t85_leaf2 <= 175.53
 T85_L3_f30_le: x_Fireplace_Qu + 3.000001 t85_leaf3 <= 4
 T85_R3_f6_ge: x_Year_Remod/Add - 1962 t85_leaf3 >= 0
 T85_R3_f0_ge: x_Lot_Frontage - 32 t85_leaf3 >= 0
 T85_R4_f30_ge: x_Fireplace_Qu >= 1
 T85_L4_f17_le: x_1st_Flr_SF + 1026.900001 t85_leaf4 <= 2534.9
 T85_L4_f4_le: x_Overall_Cond + 999995.000001 t85_leaf4 <= 1e+06
 T85_R5_f30_ge: x_Fireplace_Qu >= 1
 T85_L5_f17_le: x_1st_Flr_SF + 1026.900001 t85_leaf5 <= 2534.9
 T85_R5_f4_ge: x_Overall_Cond - 5 t85_leaf5 >= 0
 T85_R6_f30_ge: x_Fireplace_Qu >= 1
 T85_R6_f17_ge: x_1st_Flr_SF - 1508 t85_leaf6 >= 0
 T85_L6_f32_le: x_Garage_Cars + 2.000001 t85_leaf6 <= 4
 T85_R7_f30_ge: x_Fireplace_Qu >= 1
 T85_R7_f17_ge: x_1st_Flr_SF - 1508 t85_leaf7 >= 0
 T85_R7_f32_ge: x_Garage_Cars - 2 t85_leaf7 >= 0
 TREE_86_ONEHOT: t86_leaf0 + t86_leaf1 + t86_leaf2 + t86_leaf3 + t86_leaf4
   + t86_leaf5 + t86_leaf6 + t86_leaf7 = 1
 T86_L0_f12_le: x_BsmtFin_SF_1 + 1226.500001 t86_leaf0 <= 1984.5
 T86_L0_f31_le: x_Garage_Yr_Blt + 998071.000001 t86_leaf0 <= 1e+06
 T86_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t86_leaf0 <= 1
 T86_L1_f12_le: x_BsmtFin_SF_1 + 1226.500001 t86_leaf1 <= 1984.5
 T86_L1_f31_le: x_Garage_Yr_Blt + 998071.000001 t86_leaf1 <= 1e+06
 T86_R1_f293_ge: x_Sale_Condition_Abnorml - t86_leaf1 >= 0
 T86_L2_f12_le: x_BsmtFin_SF_1 + 1226.500001 t86_leaf2 <= 1984.5
 T86_R2_f31_ge: x_Garage_Yr_Blt - 1929 t86_leaf2 >= 0
 T86_L2_f1_le: x_Lot_Area <= 12179.999999
 T86_L3_f12_le: x_BsmtFin_SF_1 + 1226.500001 t86_leaf3 <= 1984.5
 T86_R3_f31_ge: x_Garage_Yr_Blt - 1929 t86_leaf3 >= 0
 T86_R3_f1_ge: x_Lot_Area - 5180 t86_leaf3 >= 7000
 T86_R4_f12_ge: x_BsmtFin_SF_1 - 758 t86_leaf4 >= 0
 T86_L4_f28_le: x_TotRms_AbvGrd + 5.080001 t86_leaf4 <= 12.08
 T86_L4_f4_le: x_Overall_Cond + 999995.000001 t86_leaf4 <= 1e+06
 T86_R5_f12_ge: x_BsmtFin_SF_1 - 758 t86_leaf5 >= 0
 T86_L5_f28_le: x_TotRms_AbvGrd + 5.080001 t86_leaf5 <= 12.08
 T86_R5_f4_ge: x_Overall_Cond - 5 t86_leaf5 >= 0
 T86_R6_f12_ge: x_BsmtFin_SF_1 - 758 t86_leaf6 >= 0
 T86_R6_f28_ge: x_TotRms_AbvGrd - 7 t86_leaf6 >= 0
 T86_L6_f1_le: x_Lot_Area <= 27696.999999
 T86_R7_f12_ge: x_BsmtFin_SF_1 - 758 t86_leaf7 >= 0
 T86_R7_f28_ge: x_TotRms_AbvGrd - 7 t86_leaf7 >= 0
 T86_R7_f1_ge: x_Lot_Area - 20697 t86_leaf7 >= 7000
 TREE_87_ONEHOT: t87_leaf0 + t87_leaf1 + t87_leaf2 + t87_leaf3 + t87_leaf4
   + t87_leaf5 + t87_leaf6 + t87_leaf7 = 1
 T87_L0_f12_le: x_BsmtFin_SF_1 + 1257.500001 t87_leaf0 <= 1984.5
 T87_L0_f4_le: x_Overall_Cond + 999996.000001 t87_leaf0 <= 1e+06
 T87_L0_f109_le: x_Neighborhood_OldTown + 1e-06 t87_leaf0 <= 1
 T87_L1_f12_le: x_BsmtFin_SF_1 + 1257.500001 t87_leaf1 <= 1984.5
 T87_L1_f4_le: x_Overall_Cond + 999996.000001 t87_leaf1 <= 1e+06
 T87_R1_f109_ge: x_Neighborhood_OldTown - t87_leaf1 >= 0
 T87_L2_f12_le: x_BsmtFin_SF_1 + 1257.500001 t87_leaf2 <= 1984.5
 T87_R2_f4_ge: x_Overall_Cond - 4 t87_leaf2 >= 0
 T87_L2_f1_le: x_Lot_Area <= 11979.999999
 T87_L3_f12_le: x_BsmtFin_SF_1 + 1257.500001 t87_leaf3 <= 1984.5
 T87_R3_f4_ge: x_Overall_Cond - 4 t87_leaf3 >= 0
 T87_R3_f1_ge: x_Lot_Area - 4980 t87_leaf3 >= 7000
 T87_R4_f12_ge: x_BsmtFin_SF_1 - 727 t87_leaf4 >= 0
 T87_L4_f15_le: x_Total_Bsmt_SF + 1095.480001 t87_leaf4 <= 2514.48
 T87_L4_f4_le: x_Overall_Cond + 999995.000001 t87_leaf4 <= 1e+06
 T87_R5_f12_ge: x_BsmtFin_SF_1 - 727 t87_leaf5 >= 0
 T87_L5_f15_le: x_Total_Bsmt_SF + 1095.480001 t87_leaf5 <= 2514.48
 T87_R5_f4_ge: x_Overall_Cond - 5 t87_leaf5 >= 0
 T87_R6_f12_ge: x_BsmtFin_SF_1 - 727 t87_leaf6 >= 0
 T87_R6_f15_ge: x_Total_Bsmt_SF - 1419 t87_leaf6 >= 0
 T87_L6_f37_le: x_Open_Porch_SF + 215.810001 t87_leaf6 <= 488.81
 T87_R7_f12_ge: x_BsmtFin_SF_1 - 727 t87_leaf7 >= 0
 T87_R7_f15_ge: x_Total_Bsmt_SF - 1419 t87_leaf7 >= 0
 T87_R7_f37_ge: x_Open_Porch_SF - 273 t87_leaf7 >= 0
 TREE_88_ONEHOT: t88_leaf0 + t88_leaf1 + t88_leaf2 + t88_leaf3 + t88_leaf4
   + t88_leaf5 + t88_leaf6 = 1
 T88_L0_f20_le: x_Gr_Liv_Area + 1158.760001 t88_leaf0 <= 3421.76
 T88_L0_f12_le: x_BsmtFin_SF_1 + 1692.500001 t88_leaf0 <= 1984.5
 T88_L0_f20_le: x_Gr_Liv_Area + 2445.760001 t88_leaf0 <= 3421.76
 T88_L1_f20_le: x_Gr_Liv_Area + 1158.760001 t88_leaf1 <= 3421.76
 T88_L1_f12_le: x_BsmtFin_SF_1 + 1692.500001 t88_leaf1 <= 1984.5
 T88_R1_f20_ge: x_Gr_Liv_Area - 976 t88_leaf1 >= 0
 T88_L2_f20_le: x_Gr_Liv_Area + 1158.760001 t88_leaf2 <= 3421.76
 T88_R2_f12_ge: x_BsmtFin_SF_1 - 292 t88_leaf2 >= 0
 T88_L2_f20_le: x_Gr_Liv_Area + 2523.760001 t88_leaf2 <= 3421.76
 T88_L3_f20_le: x_Gr_Liv_Area + 1158.760001 t88_leaf3 <= 3421.76
 T88_R3_f12_ge: x_BsmtFin_SF_1 - 292 t88_leaf3 >= 0
 T88_R3_f20_ge: x_Gr_Liv_Area - 898 t88_leaf3 >= 0
 T88_R4_f20_ge: x_Gr_Liv_Area - 2263 t88_leaf4 >= 0
 T88_L4_f6_le: x_Year_Remod/Add + 998019.000001 t88_leaf4 <= 1e+06
 T88_R5_f20_ge: x_Gr_Liv_Area - 2263 t88_leaf5 >= 0
 T88_R5_f6_ge: x_Year_Remod/Add - 1981 t88_leaf5 >= 0
 T88_L5_f0_le: x_Lot_Frontage + 55.530001 t88_leaf5 <= 175.53
 T88_R6_f20_ge: x_Gr_Liv_Area - 2263 t88_leaf6 >= 0
 T88_R6_f6_ge: x_Year_Remod/Add - 1981 t88_leaf6 >= 0
 T88_R6_f0_ge: x_Lot_Frontage - 120 t88_leaf6 >= 0
 TREE_89_ONEHOT: t89_leaf0 + t89_leaf1 + t89_leaf2 + t89_leaf3 + t89_leaf4
   + t89_leaf5 + t89_leaf6 + t89_leaf7 = 1
 T89_L0_f17_le: x_1st_Flr_SF + 1232.900001 t89_leaf0 <= 2534.9
 T89_L0_f11_le: x_Bsmt_Cond + 2.000001 t89_leaf0 <= 4
 T89_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t89_leaf0 <= 1
 T89_L1_f17_le: x_1st_Flr_SF + 1232.900001 t89_leaf1 <= 2534.9
 T89_L1_f11_le: x_Bsmt_Cond + 2.000001 t89_leaf1 <= 4
 T89_R1_f293_ge: x_Sale_Condition_Abnorml - t89_leaf1 >= 0
 T89_L2_f17_le: x_1st_Flr_SF + 1232.900001 t89_leaf2 <= 2534.9
 T89_R2_f11_ge: x_Bsmt_Cond - 3 t89_leaf2 >= -1
 T89_L2_f1_le: x_Lot_Area + 558.0000010000003 t89_leaf2 <= 7000
 T89_L3_f17_le: x_1st_Flr_SF + 1232.900001 t89_leaf3 <= 2534.9
 T89_R3_f11_ge: x_Bsmt_Cond - 3 t89_leaf3 >= -1
 T89_R3_f1_ge: x_Lot_Area >= 6442
 T89_R4_f17_ge: x_1st_Flr_SF - 1302 t89_leaf4 >= 0
 T89_L4_f6_le: x_Year_Remod/Add + 998010.000001 t89_leaf4 <= 1e+06
 T89_L4_f14_le: x_Bsmt_Unf_SF + 482.6800010000002 t89_leaf4 <= 1812.68
 T89_R5_f17_ge: x_1st_Flr_SF - 1302 t89_leaf5 >= 0
 T89_L5_f6_le: x_Year_Remod/Add + 998010.000001 t89_leaf5 <= 1e+06
 T89_R5_f14_ge: x_Bsmt_Unf_SF - 1330 t89_leaf5 >= 0
 T89_R6_f17_ge: x_1st_Flr_SF - 1302 t89_leaf6 >= 0
 T89_R6_f6_ge: x_Year_Remod/Add - 1990 t89_leaf6 >= 0
 T89_L6_f17_le: x_1st_Flr_SF + 734.9000010000002 t89_leaf6 <= 2534.9
 T89_R7_f17_ge: x_1st_Flr_SF - 1302 t89_leaf7 >= 0
 T89_R7_f6_ge: x_Year_Remod/Add - 1990 t89_leaf7 >= 0
 T89_R7_f17_ge: x_1st_Flr_SF - 1800 t89_leaf7 >= 0
 TREE_90_ONEHOT: t90_leaf0 + t90_leaf1 + t90_leaf2 + t90_leaf3 + t90_leaf4
   + t90_leaf5 + t90_leaf6 = 1
 T90_L0_f33_le: x_Garage_Area + 488.750001 t90_leaf0 <= 1163.75
 T90_L0_f12_le: x_BsmtFin_SF_1 + 1692.500001 t90_leaf0 <= 1984.5
 T90_L0_f16_le: x_Heating_QC + 1.000001 t90_leaf0 <= 4
 T90_L1_f33_le: x_Garage_Area + 488.750001 t90_leaf1 <= 1163.75
 T90_L1_f12_le: x_BsmtFin_SF_1 + 1692.500001 t90_leaf1 <= 1984.5
 T90_R1_f16_ge: x_Heating_QC - t90_leaf1 >= 2
 T90_L2_f33_le: x_Garage_Area + 488.750001 t90_leaf2 <= 1163.75
 T90_R2_f12_ge: x_BsmtFin_SF_1 - 292 t90_leaf2 >= 0
 T90_L2_f5_le: x_Year_Built + 998086.000001 t90_leaf2 <= 1e+06
 T90_L3_f33_le: x_Garage_Area + 488.750001 t90_leaf3 <= 1163.75
 T90_R3_f12_ge: x_BsmtFin_SF_1 - 292 t90_leaf3 >= 0
 T90_R3_f5_ge: x_Year_Built - 1914 t90_leaf3 >= 0
 T90_R4_f33_ge: x_Garage_Area - 675 t90_leaf4 >= 0
 T90_L4_f33_le: x_Garage_Area + 9.750001000000111 t90_leaf4 <= 1163.75
 T90_L4_f27_le: x_Kitchen_Qual + 1.000001 t90_leaf4 <= 4
 T90_R5_f33_ge: x_Garage_Area - 675 t90_leaf5 >= 0
 T90_L5_f33_le: x_Garage_Area + 9.750001000000111 t90_leaf5 <= 1163.75
 T90_R5_f27_ge: x_Kitchen_Qual - t90_leaf5 >= 2
 T90_R6_f33_ge: x_Garage_Area - 675 t90_leaf6 >= 0
 T90_R6_f33_ge: x_Garage_Area - 1154 t90_leaf6 >= 0
 TREE_91_ONEHOT: t91_leaf0 + t91_leaf1 + t91_leaf2 + t91_leaf3 + t91_leaf4
   + t91_leaf5 + t91_leaf6 + t91_leaf7 = 1
 T91_L0_f20_le: x_Gr_Liv_Area + 2159.760001000001 t91_leaf0 <= 3421.76
 T91_L0_f5_le: x_Year_Built + 998062.000001 t91_leaf0 <= 1e+06
 T91_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t91_leaf0 <= 1
 T91_L1_f20_le: x_Gr_Liv_Area + 2159.760001000001 t91_leaf1 <= 3421.76
 T91_L1_f5_le: x_Year_Built + 998062.000001 t91_leaf1 <= 1e+06
 T91_R1_f293_ge: x_Sale_Condition_Abnorml - t91_leaf1 >= 0
 T91_L2_f20_le: x_Gr_Liv_Area + 2159.760001000001 t91_leaf2 <= 3421.76
 T91_R2_f5_ge: x_Year_Built - 1938 t91_leaf2 >= 0
 T91_L2_f15_le: x_Total_Bsmt_SF + 1964.480001 t91_leaf2 <= 2514.48
 T91_L3_f20_le: x_Gr_Liv_Area + 2159.760001000001 t91_leaf3 <= 3421.76
 T91_R3_f5_ge: x_Year_Built - 1938 t91_leaf3 >= 0
 T91_R3_f15_ge: x_Total_Bsmt_SF - 550 t91_leaf3 >= 0
 T91_R4_f20_ge: x_Gr_Liv_Area - 1262 t91_leaf4 >= 0
 T91_L4_f20_le: x_Gr_Liv_Area + 1158.760001 t91_leaf4 <= 3421.76
 T91_L4_f21_le: x_Bsmt_Full_Bath + 1.020001 t91_leaf4 <= 2.02
 T91_R5_f20_ge: x_Gr_Liv_Area - 1262 t91_leaf5 >= 0
 T91_L5_f20_le: x_Gr_Liv_Area + 1158.760001 t91_leaf5 <= 3421.76
 T91_R5_f21_ge: x_Bsmt_Full_Bath - t91_leaf5 >= 0
 T91_R6_f20_ge: x_Gr_Liv_Area - 1262 t91_leaf6 >= 0
 T91_R6_f20_ge: x_Gr_Liv_Area - 2263 t91_leaf6 >= 0
 T91_L6_f28_le: x_TotRms_AbvGrd + 0.080001 t91_leaf6 <= 12.08
 T91_R7_f20_ge: x_Gr_Liv_Area - 1262 t91_leaf7 >= 0
 T91_R7_f20_ge: x_Gr_Liv_Area - 2263 t91_leaf7 >= 0
 T91_R7_f28_ge: x_TotRms_AbvGrd - 12 t91_leaf7 >= 0
 TREE_92_ONEHOT: t92_leaf0 + t92_leaf1 + t92_leaf2 + t92_leaf3 + t92_leaf4
   + t92_leaf5 + t92_leaf6 = 1
 T92_L0_f20_le: x_Gr_Liv_Area + 1352.760001 t92_leaf0 <= 3421.76
 T92_L0_f9_le: x_Exter_Cond + 2.000001 t92_leaf0 <= 4
 T92_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t92_leaf0 <= 1
 T92_L1_f20_le: x_Gr_Liv_Area + 1352.760001 t92_leaf1 <= 3421.76
 T92_L1_f9_le: x_Exter_Cond + 2.000001 t92_leaf1 <= 4
 T92_R1_f297_ge: x_Sale_Condition_Normal - t92_leaf1 >= 0
 T92_L2_f20_le: x_Gr_Liv_Area + 1352.760001 t92_leaf2 <= 3421.76
 T92_R2_f9_ge: x_Exter_Cond - 3 t92_leaf2 >= -1
 T92_L2_f20_le: x_Gr_Liv_Area + 2485.760001 t92_leaf2 <= 3421.76
 T92_L3_f20_le: x_Gr_Liv_Area + 1352.760001 t92_leaf3 <= 3421.76
 T92_R3_f9_ge: x_Exter_Cond - 3 t92_leaf3 >= -1
 T92_R3_f20_ge: x_Gr_Liv_Area - 936 t92_leaf3 >= 0
 T92_R4_f20_ge: x_Gr_Liv_Area - 2069 t92_leaf4 >= 0
 T92_L4_f27_le: x_Kitchen_Qual + 1.000001 t92_leaf4 <= 4
 T92_L4_f5_le: x_Year_Built + 998099.000001 t92_leaf4 <= 1e+06
 T92_R5_f20_ge: x_Gr_Liv_Area - 2069 t92_leaf5 >= 0
 T92_L5_f27_le: x_Kitchen_Qual + 1.000001 t92_leaf5 <= 4
 T92_R5_f5_ge: x_Year_Built - 1901 t92_leaf5 >= 0
 T92_R6_f20_ge: x_Gr_Liv_Area - 2069 t92_leaf6 >= 0
 T92_R6_f27_ge: x_Kitchen_Qual - t92_leaf6 >= 2
 TREE_93_ONEHOT: t93_leaf0 + t93_leaf1 + t93_leaf2 + t93_leaf3 + t93_leaf4
   + t93_leaf5 + t93_leaf6 = 1
 T93_L0_f32_le: x_Garage_Cars + 1.000001 t93_leaf0 <= 4
 T93_L0_f12_le: x_BsmtFin_SF_1 + 1957.500001 t93_leaf0 <= 1984.5
 T93_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t93_leaf0 <= 1
 T93_L1_f32_le: x_Garage_Cars + 1.000001 t93_leaf1 <= 4
 T93_L1_f12_le: x_BsmtFin_SF_1 + 1957.500001 t93_leaf1 <= 1984.5
 T93_R1_f293_ge: x_Sale_Condition_Abnorml - t93_leaf1 >= 0
 T93_L2_f32_le: x_Garage_Cars + 1.000001 t93_leaf2 <= 4
 T93_R2_f12_ge: x_BsmtFin_SF_1 - 27 t93_leaf2 >= 0
 T93_L2_f44_le: x_Yr_Sold + 997991.000001 t93_leaf2 <= 1e+06
 T93_L3_f32_le: x_Garage_Cars + 1.000001 t93_leaf3 <= 4
 T93_R3_f12_ge: x_BsmtFin_SF_1 - 27 t93_leaf3 >= 0
 T93_R3_f44_ge: x_Yr_Sold - 2009 t93_leaf3 >= 0
 T93_R4_f32_ge: x_Garage_Cars - 3 t93_leaf4 >= 0
 T93_L4_f16_le: x_Heating_QC + 1.000001 t93_leaf4 <= 4
 T93_L4_f4_le: x_Overall_Cond + 999994.000001 t93_leaf4 <= 1e+06
 T93_R5_f32_ge: x_Garage_Cars - 3 t93_leaf5 >= 0
 T93_L5_f16_le: x_Heating_QC + 1.000001 t93_leaf5 <= 4
 T93_R5_f4_ge: x_Overall_Cond - 6 t93_leaf5 >= 0
 T93_R6_f32_ge: x_Garage_Cars - 3 t93_leaf6 >= 0
 T93_R6_f16_ge: x_Heating_QC - t93_leaf6 >= 2
 TREE_94_ONEHOT: t94_leaf0 + t94_leaf1 + t94_leaf2 + t94_leaf3 + t94_leaf4
   + t94_leaf5 = 1
 T94_L0_f27_le: x_Kitchen_Qual + 1e-06 t94_leaf0 <= 4
 T94_L0_f10_le: x_Bsmt_Qual + 1.000001 t94_leaf0 <= 4
 T94_L0_f1_le: x_Lot_Area <= 9115.999999
 T94_L1_f27_le: x_Kitchen_Qual + 1e-06 t94_leaf1 <= 4
 T94_L1_f10_le: x_Bsmt_Qual + 1.000001 t94_leaf1 <= 4
 T94_R1_f1_ge: x_Lot_Area - 2116 t94_leaf1 >= 7000
 T94_L2_f27_le: x_Kitchen_Qual + 1e-06 t94_leaf2 <= 4
 T94_R2_f10_ge: x_Bsmt_Qual - t94_leaf2 >= 2
 T94_L2_f1_le: x_Lot_Area + 4240.000001 t94_leaf2 <= 7000
 T94_L3_f27_le: x_Kitchen_Qual + 1e-06 t94_leaf3 <= 4
 T94_R3_f10_ge: x_Bsmt_Qual - t94_leaf3 >= 2
 T94_R3_f1_ge: x_Lot_Area >= 2760
 T94_R4_f27_ge: x_Kitchen_Qual - 2 t94_leaf4 >= 2
 T94_L4_f257_le: x_Paved_Drive_N + 1e-06 t94_leaf4 <= 1
 T94_R5_f27_ge: x_Kitchen_Qual - 2 t94_leaf5 >= 2
 T94_R5_f257_ge: x_Paved_Drive_N - t94_leaf5 >= 0
 TREE_95_ONEHOT: t95_leaf0 + t95_leaf1 + t95_leaf2 + t95_leaf3 + t95_leaf4
   + t95_leaf5 + t95_leaf6 + t95_leaf7 = 1
 T95_L0_f6_le: x_Year_Remod/Add + 998007.000001 t95_leaf0 <= 1e+06
 T95_L0_f4_le: x_Overall_Cond + 999996.000001 t95_leaf0 <= 1e+06
 T95_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t95_leaf0 <= 1
 T95_L1_f6_le: x_Year_Remod/Add + 998007.000001 t95_leaf1 <= 1e+06
 T95_L1_f4_le: x_Overall_Cond + 999996.000001 t95_leaf1 <= 1e+06
 T95_R1_f297_ge: x_Sale_Condition_Normal - t95_leaf1 >= 0
 T95_L2_f6_le: x_Year_Remod/Add + 998007.000001 t95_leaf2 <= 1e+06
 T95_R2_f4_ge: x_Overall_Cond - 4 t95_leaf2 >= 0
 T95_L2_f31_le: x_Garage_Yr_Blt + 998086.000001 t95_leaf2 <= 1e+06
 T95_L3_f6_le: x_Year_Remod/Add + 998007.000001 t95_leaf3 <= 1e+06
 T95_R3_f4_ge: x_Overall_Cond - 4 t95_leaf3 >= 0
 T95_R3_f31_ge: x_Garage_Yr_Blt - 1914 t95_leaf3 >= 0
 T95_R4_f6_ge: x_Year_Remod/Add - 1993 t95_leaf4 >= 0
 T95_L4_f15_le: x_Total_Bsmt_SF + 664.4800010000001 t95_leaf4 <= 2514.48
 T95_L4_f1_le: x_Lot_Area <= 11699.999999
 T95_R5_f6_ge: x_Year_Remod/Add - 1993 t95_leaf5 >= 0
 T95_L5_f15_le: x_Total_Bsmt_SF + 664.4800010000001 t95_leaf5 <= 2514.48
 T95_R5_f1_ge: x_Lot_Area - 4700 t95_leaf5 >= 7000
 T95_R6_f6_ge: x_Year_Remod/Add - 1993 t95_leaf6 >= 0
 T95_R6_f15_ge: x_Total_Bsmt_SF - 1850 t95_leaf6 >= 0
 T95_L6_f6_le: x_Year_Remod/Add + 997991.000001 t95_leaf6 <= 1e+06
 T95_R7_f6_ge: x_Year_Remod/Add - 1993 t95_leaf7 >= 0
 T95_R7_f15_ge: x_Total_Bsmt_SF - 1850 t95_leaf7 >= 0
 T95_R7_f6_ge: x_Year_Remod/Add - 2009 t95_leaf7 >= 0
 TREE_96_ONEHOT: t96_leaf0 + t96_leaf1 + t96_leaf2 + t96_leaf3 + t96_leaf4
   + t96_leaf5 + t96_leaf6 + t96_leaf7 = 1
 T96_L0_f20_le: x_Gr_Liv_Area + 2433.760001 t96_leaf0 <= 3421.76
 T96_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t96_leaf0 <= 1
 T96_L0_f3_le: x_Overall_Qual + 999995.000001 t96_leaf0 <= 1e+06
 T96_L1_f20_le: x_Gr_Liv_Area + 2433.760001 t96_leaf1 <= 3421.76
 T96_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t96_leaf1 <= 1
 T96_R1_f3_ge: x_Overall_Qual - 5 t96_leaf1 >= 0
 T96_L2_f20_le: x_Gr_Liv_Area + 2433.760001 t96_leaf2 <= 3421.76
 T96_R2_f297_ge: x_Sale_Condition_Normal - t96_leaf2 >= 0
 T96_L2_f217_le: x_BsmtFin_Type_1_Unf + 1e-06 t96_leaf2 <= 1
 T96_L3_f20_le: x_Gr_Liv_Area + 2433.760001 t96_leaf3 <= 3421.76
 T96_R3_f297_ge: x_Sale_Condition_Normal - t96_leaf3 >= 0
 T96_R3_f217_ge: x_BsmtFin_Type_1_Unf - t96_leaf3 >= 0
 T96_R4_f20_ge: x_Gr_Liv_Area - 988 t96_leaf4 >= 0
 T96_L4_f20_le: x_Gr_Liv_Area + 1028.760001 t96_leaf4 <= 3421.76
 T96_L4_f4_le: x_Overall_Cond + 999996.000001 t96_leaf4 <= 1e+06
 T96_R5_f20_ge: x_Gr_Liv_Area - 988 t96_leaf5 >= 0
 T96_L5_f20_le: x_Gr_Liv_Area + 1028.760001 t96_leaf5 <= 3421.76
 T96_R5_f4_ge: x_Overall_Cond - 4 t96_leaf5 >= 0
 T96_R6_f20_ge: x_Gr_Liv_Area - 988 t96_leaf6 >= 0
 T96_R6_f20_ge: x_Gr_Liv_Area - 2393 t96_leaf6 >= 0
 T96_L6_f118_le: x_Condition_1_Feedr + 1e-06 t96_leaf6 <= 1
 T96_R7_f20_ge: x_Gr_Liv_Area - 988 t96_leaf7 >= 0
 T96_R7_f20_ge: x_Gr_Liv_Area - 2393 t96_leaf7 >= 0
 T96_R7_f118_ge: x_Condition_1_Feedr - t96_leaf7 >= 0
 TREE_97_ONEHOT: t97_leaf0 + t97_leaf1 + t97_leaf2 + t97_leaf3 + t97_leaf4
   + t97_leaf5 + t97_leaf6 = 1
 T97_L0_f1_le: x_Lot_Area <= 11979.999999
 T97_L0_f16_le: x_Heating_QC + 1.000001 t97_leaf0 <= 4
 T97_L0_f66_le: x_MS_Zoning_RL + 1e-06 t97_leaf0 <= 1
 T97_L1_f1_le: x_Lot_Area <= 11979.999999
 T97_L1_f16_le: x_Heating_QC + 1.000001 t97_leaf1 <= 4
 T97_R1_f66_ge: x_MS_Zoning_RL - t97_leaf1 >= 0
 T97_L2_f1_le: x_Lot_Area <= 11979.999999
 T97_R2_f16_ge: x_Heating_QC - t97_leaf2 >= 2
 T97_L2_f12_le: x_BsmtFin_SF_1 + 1802.500001 t97_leaf2 <= 1984.5
 T97_L3_f1_le: x_Lot_Area <= 11979.999999
 T97_R3_f16_ge: x_Heating_QC - t97_leaf3 >= 2
 T97_R3_f12_ge: x_BsmtFin_SF_1 - 182 t97_leaf3 >= 0
 T97_R4_f1_ge: x_Lot_Area - 4980 t97_leaf4 >= 7000
 T97_L4_f60_le: x_MS_SubClass_190 + 1e-06 t97_leaf4 <= 1
 T97_L4_f96_le: x_Neighborhood_Edwards + 1e-06 t97_leaf4 <= 1
 T97_R5_f1_ge: x_Lot_Area - 4980 t97_leaf5 >= 7000
 T97_L5_f60_le: x_MS_SubClass_190 + 1e-06 t97_leaf5 <= 1
 T97_R5_f96_ge: x_Neighborhood_Edwards - t97_leaf5 >= 0
 T97_R6_f1_ge: x_Lot_Area - 4980 t97_leaf6 >= 7000
 T97_R6_f60_ge: x_MS_SubClass_190 - t97_leaf6 >= 0
 TREE_98_ONEHOT: t98_leaf0 + t98_leaf1 + t98_leaf2 + t98_leaf3 + t98_leaf4
   + t98_leaf5 + t98_leaf6 = 1
 T98_L0_f20_le: x_Gr_Liv_Area + 1126.760001 t98_leaf0 <= 3421.76
 T98_L0_f4_le: x_Overall_Cond + 999995.000001 t98_leaf0 <= 1e+06
 T98_L0_f11_le: x_Bsmt_Cond + 2.000001 t98_leaf0 <= 4
 T98_L1_f20_le: x_Gr_Liv_Area + 1126.760001 t98_leaf1 <= 3421.76
 T98_L1_f4_le: x_Overall_Cond + 999995.000001 t98_leaf1 <= 1e+06
 T98_R1_f11_ge: x_Bsmt_Cond - 3 t98_leaf1 >= -1
 T98_L2_f20_le: x_Gr_Liv_Area + 1126.760001 t98_leaf2 <= 3421.76
 T98_R2_f4_ge: x_Overall_Cond - 5 t98_leaf2 >= 0
 T98_L2_f20_le: x_Gr_Liv_Area + 2325.760001000001 t98_leaf2 <= 3421.76
 T98_L3_f20_le: x_Gr_Liv_Area + 1126.760001 t98_leaf3 <= 3421.76
 T98_R3_f4_ge: x_Overall_Cond - 5 t98_leaf3 >= 0
 T98_R3_f20_ge: x_Gr_Liv_Area - 1096 t98_leaf3 >= 0
 T98_R4_f20_ge: x_Gr_Liv_Area - 2295 t98_leaf4 >= 0
 T98_L4_f36_le: x_Wood_Deck_SF + 4.900000999999975 t98_leaf4 <= 525.9
 T98_L4_f28_le: x_TotRms_AbvGrd + 4.080001 t98_leaf4 <= 12.08
 T98_R5_f20_ge: x_Gr_Liv_Area - 2295 t98_leaf5 >= 0
 T98_L5_f36_le: x_Wood_Deck_SF + 4.900000999999975 t98_leaf5 <= 525.9
 T98_R5_f28_ge: x_TotRms_AbvGrd - 8 t98_leaf5 >= 0
 T98_R6_f20_ge: x_Gr_Liv_Area - 2295 t98_leaf6 >= 0
 T98_R6_f36_ge: x_Wood_Deck_SF - 521 t98_leaf6 >= 0
 TREE_99_ONEHOT: t99_leaf0 + t99_leaf1 + t99_leaf2 + t99_leaf3 + t99_leaf4
   + t99_leaf5 + t99_leaf6 = 1
 T99_L0_f12_le: x_BsmtFin_SF_1 + 1257.500001 t99_leaf0 <= 1984.5
 T99_L0_f6_le: x_Year_Remod/Add + 998016.000001 t99_leaf0 <= 1e+06
 T99_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t99_leaf0 <= 1
 T99_L1_f12_le: x_BsmtFin_SF_1 + 1257.500001 t99_leaf1 <= 1984.5
 T99_L1_f6_le: x_Year_Remod/Add + 998016.000001 t99_leaf1 <= 1e+06
 T99_R1_f293_ge: x_Sale_Condition_Abnorml - t99_leaf1 >= 0
 T99_L2_f12_le: x_BsmtFin_SF_1 + 1257.500001 t99_leaf2 <= 1984.5
 T99_R2_f6_ge: x_Year_Remod/Add - 1984 t99_leaf2 >= 0
 T99_L2_f20_le: x_Gr_Liv_Area + 1477.760001 t99_leaf2 <= 3421.76
 T99_L3_f12_le: x_BsmtFin_SF_1 + 1257.500001 t99_leaf3 <= 1984.5
 T99_R3_f6_ge: x_Year_Remod/Add - 1984 t99_leaf3 >= 0
 T99_R3_f20_ge: x_Gr_Liv_Area - 1944 t99_leaf3 >= 0
 T99_R4_f12_ge: x_BsmtFin_SF_1 - 727 t99_leaf4 >= 0
 T99_L4_f12_le: x_BsmtFin_SF_1 + 784.5000010000001 t99_leaf4 <= 1984.5
 T99_L4_f20_le: x_Gr_Liv_Area + 2527.760001 t99_leaf4 <= 3421.76
 T99_R5_f12_ge: x_BsmtFin_SF_1 - 727 t99_leaf5 >= 0
 T99_L5_f12_le: x_BsmtFin_SF_1 + 784.5000010000001 t99_leaf5 <= 1984.5
 T99_R5_f20_ge: x_Gr_Liv_Area - 894 t99_leaf5 >= 0
 T99_R6_f12_ge: x_BsmtFin_SF_1 - 727 t99_leaf6 >= 0
 T99_R6_f12_ge: x_BsmtFin_SF_1 - 1200 t99_leaf6 >= 0
 TREE_100_ONEHOT: t100_leaf0 + t100_leaf1 + t100_leaf2 + t100_leaf3
   + t100_leaf4 + t100_leaf5 + t100_leaf6 = 1
 T100_L0_f10_le: x_Bsmt_Qual + 1e-06 t100_leaf0 <= 4
 T100_L0_f4_le: x_Overall_Cond + 999995.000001 t100_leaf0 <= 1e+06
 T100_L0_f76_le: x_Lot_Shape_Reg + 1e-06 t100_leaf0 <= 1
 T100_L1_f10_le: x_Bsmt_Qual + 1e-06 t100_leaf1 <= 4
 T100_L1_f4_le: x_Overall_Cond + 999995.000001 t100_leaf1 <= 1e+06
 T100_R1_f76_ge: x_Lot_Shape_Reg - t100_leaf1 >= 0
 T100_L2_f10_le: x_Bsmt_Qual + 1e-06 t100_leaf2 <= 4
 T100_R2_f4_ge: x_Overall_Cond - 5 t100_leaf2 >= 0
 T100_L2_f0_le: x_Lot_Frontage + 111.530001 t100_leaf2 <= 175.53
 T100_L3_f10_le: x_Bsmt_Qual + 1e-06 t100_leaf3 <= 4
 T100_R3_f4_ge: x_Overall_Cond - 5 t100_leaf3 >= 0
 T100_R3_f0_ge: x_Lot_Frontage - 64 t100_leaf3 >= 0
 T100_R4_f10_ge: x_Bsmt_Qual - 2 t100_leaf4 >= 2
 T100_L4_f96_le: x_Neighborhood_Edwards + 1e-06 t100_leaf4 <= 1
 T100_L4_f12_le: x_BsmtFin_SF_1 + 1934.500001 t100_leaf4 <= 1984.5
 T100_R5_f10_ge: x_Bsmt_Qual - 2 t100_leaf5 >= 2
 T100_L5_f96_le: x_Neighborhood_Edwards + 1e-06 t100_leaf5 <= 1
 T100_R5_f12_ge: x_BsmtFin_SF_1 - 50 t100_leaf5 >= 0
 T100_R6_f10_ge: x_Bsmt_Qual - 2 t100_leaf6 >= 2
 T100_R6_f96_ge: x_Neighborhood_Edwards - t100_leaf6 >= 0
 TREE_101_ONEHOT: t101_leaf0 + t101_leaf1 + t101_leaf2 + t101_leaf3
   + t101_leaf4 + t101_leaf5 + t101_leaf6 + t101_leaf7 = 1
 T101_L0_f6_le: x_Year_Remod/Add + 998007.000001 t101_leaf0 <= 1e+06
 T101_L0_f66_le: x_MS_Zoning_RL + 1e-06 t101_leaf0 <= 1
 T101_L0_f20_le: x_Gr_Liv_Area + 2485.760001 t101_leaf0 <= 3421.76
 T101_L1_f6_le: x_Year_Remod/Add + 998007.000001 t101_leaf1 <= 1e+06
 T101_L1_f66_le: x_MS_Zoning_RL + 1e-06 t101_leaf1 <= 1
 T101_R1_f20_ge: x_Gr_Liv_Area - 936 t101_leaf1 >= 0
 T101_L2_f6_le: x_Year_Remod/Add + 998007.000001 t101_leaf2 <= 1e+06
 T101_R2_f66_ge: x_MS_Zoning_RL - t101_leaf2 >= 0
 T101_L2_f1_le: x_Lot_Area <= 10799.999999
 T101_L3_f6_le: x_Year_Remod/Add + 998007.000001 t101_leaf3 <= 1e+06
 T101_R3_f66_ge: x_MS_Zoning_RL - t101_leaf3 >= 0
 T101_R3_f1_ge: x_Lot_Area - 3800 t101_leaf3 >= 7000
 T101_R4_f6_ge: x_Year_Remod/Add - 1993 t101_leaf4 >= 0
 T101_L4_f17_le: x_1st_Flr_SF + 1164.900001 t101_leaf4 <= 2534.9
 T101_L4_f67_le: x_MS_Zoning_RM + 1e-06 t101_leaf4 <= 1
 T101_R5_f6_ge: x_Year_Remod/Add - 1993 t101_leaf5 >= 0
 T101_L5_f17_le: x_1st_Flr_SF + 1164.900001 t101_leaf5 <= 2534.9
 T101_R5_f67_ge: x_MS_Zoning_RM - t101_leaf5 >= 0
 T101_R6_f6_ge: x_Year_Remod/Add - 1993 t101_leaf6 >= 0
 T101_R6_f17_ge: x_1st_Flr_SF - 1370 t101_leaf6 >= 0
 T101_L6_f33_le: x_Garage_Area + 9.750001000000111 t101_leaf6 <= 1163.75
 T101_R7_f6_ge: x_Year_Remod/Add - 1993 t101_leaf7 >= 0
 T101_R7_f17_ge: x_1st_Flr_SF - 1370 t101_leaf7 >= 0
 T101_R7_f33_ge: x_Garage_Area - 1154 t101_leaf7 >= 0
 TREE_102_ONEHOT: t102_leaf0 + t102_leaf1 + t102_leaf2 + t102_leaf3
   + t102_leaf4 + t102_leaf5 + t102_leaf6 + t102_leaf7 = 1
 T102_L0_f12_le: x_BsmtFin_SF_1 + 1214.500001 t102_leaf0 <= 1984.5
 T102_L0_f44_le: x_Yr_Sold + 997992.000001 t102_leaf0 <= 1e+06
 T102_L0_f20_le: x_Gr_Liv_Area + 1382.760001 t102_leaf0 <= 3421.76
 T102_L1_f12_le: x_BsmtFin_SF_1 + 1214.500001 t102_leaf1 <= 1984.5
 T102_L1_f44_le: x_Yr_Sold + 997992.000001 t102_leaf1 <= 1e+06
 T102_R1_f20_ge: x_Gr_Liv_Area - 2039 t102_leaf1 >= 0
 T102_L2_f12_le: x_BsmtFin_SF_1 + 1214.500001 t102_leaf2 <= 1984.5
 T102_R2_f44_ge: x_Yr_Sold - 2008 t102_leaf2 >= 0
 T102_L2_f6_le: x_Year_Remod/Add + 998045.000001 t102_leaf2 <= 1e+06
 T102_L3_f12_le: x_BsmtFin_SF_1 + 1214.500001 t102_leaf3 <= 1984.5
 T102_R3_f44_ge: x_Yr_Sold - 2008 t102_leaf3 >= 0
 T102_R3_f6_ge: x_Year_Remod/Add - 1955 t102_leaf3 >= 0
 T102_R4_f12_ge: x_BsmtFin_SF_1 - 770 t102_leaf4 >= 0
 T102_L4_f1_le: x_Lot_Area <= 13439.999999
 T102_L4_f5_le: x_Year_Built + 998030.000001 t102_leaf4 <= 1e+06
 T102_R5_f12_ge: x_BsmtFin_SF_1 - 770 t102_leaf5 >= 0
 T102_L5_f1_le: x_Lot_Area <= 13439.999999
 T102_R5_f5_ge: x_Year_Built - 1970 t102_leaf5 >= 0
 T102_R6_f12_ge: x_BsmtFin_SF_1 - 770 t102_leaf6 >= 0
 T102_R6_f1_ge: x_Lot_Area - 6440 t102_leaf6 >= 7000
 T102_L6_f4_le: x_Overall_Cond + 999995.000001 t102_leaf6 <= 1e+06
 T102_R7_f12_ge: x_BsmtFin_SF_1 - 770 t102_leaf7 >= 0
 T102_R7_f1_ge: x_Lot_Area - 6440 t102_leaf7 >= 7000
 T102_R7_f4_ge: x_Overall_Cond - 5 t102_leaf7 >= 0
 TREE_103_ONEHOT: t103_leaf0 + t103_leaf1 + t103_leaf2 + t103_leaf3
   + t103_leaf4 + t103_leaf5 + t103_leaf6 + t103_leaf7 = 1
 T103_L0_f1_le: x_Lot_Area <= 13439.999999
 T103_L0_f4_le: x_Overall_Cond + 999995.000001 t103_leaf0 <= 1e+06
 T103_L0_f109_le: x_Neighborhood_OldTown + 1e-06 t103_leaf0 <= 1
 T103_L1_f1_le: x_Lot_Area <= 13439.999999
 T103_L1_f4_le: x_Overall_Cond + 999995.000001 t103_leaf1 <= 1e+06
 T103_R1_f109_ge: x_Neighborhood_OldTown - t103_leaf1 >= 0
 T103_L2_f1_le: x_Lot_Area <= 13439.999999
 T103_R2_f4_ge: x_Overall_Cond - 5 t103_leaf2 >= 0
 T103_L2_f6_le: x_Year_Remod/Add + 998014.000001 t103_leaf2 <= 1e+06
 T103_L3_f1_le: x_Lot_Area <= 13439.999999
 T103_R3_f4_ge: x_Overall_Cond - 5 t103_leaf3 >= 0
 T103_R3_f6_ge: x_Year_Remod/Add - 1986 t103_leaf3 >= 0
 T103_R4_f1_ge: x_Lot_Area - 6440 t103_leaf4 >= 7000
 T103_L4_f15_le: x_Total_Bsmt_SF + 1009.480001 t103_leaf4 <= 2514.48
 T103_L4_f15_le: x_Total_Bsmt_SF + 2269.480001 t103_leaf4 <= 2514.48
 T103_R5_f1_ge: x_Lot_Area - 6440 t103_leaf5 >= 7000
 T103_L5_f15_le: x_Total_Bsmt_SF + 1009.480001 t103_leaf5 <= 2514.48
 T103_R5_f15_ge: x_Total_Bsmt_SF - 245 t103_leaf5 >= 0
 T103_R6_f1_ge: x_Lot_Area - 6440 t103_leaf6 >= 7000
 T103_R6_f15_ge: x_Total_Bsmt_SF - 1505 t103_leaf6 >= 0
 T103_L6_f33_le: x_Garage_Area + 191.750001 t103_leaf6 <= 1163.75
 T103_R7_f1_ge: x_Lot_Area - 6440 t103_leaf7 >= 7000
 T103_R7_f15_ge: x_Total_Bsmt_SF - 1505 t103_leaf7 >= 0
 T103_R7_f33_ge: x_Garage_Area - 972 t103_leaf7 >= 0
 TREE_104_ONEHOT: t104_leaf0 + t104_leaf1 + t104_leaf2 + t104_leaf3
   + t104_leaf4 + t104_leaf5 + t104_leaf6 = 1
 T104_L0_f20_le: x_Gr_Liv_Area + 1076.760001 t104_leaf0 <= 3421.76
 T104_L0_f10_le: x_Bsmt_Qual + 2.000001 t104_leaf0 <= 4
 T104_L0_f44_le: x_Yr_Sold + 997993.000001 t104_leaf0 <= 1e+06
 T104_L1_f20_le: x_Gr_Liv_Area + 1076.760001 t104_leaf1 <= 3421.76
 T104_L1_f10_le: x_Bsmt_Qual + 2.000001 t104_leaf1 <= 4
 T104_R1_f44_ge: x_Yr_Sold - 2007 t104_leaf1 >= 0
 T104_L2_f20_le: x_Gr_Liv_Area + 1076.760001 t104_leaf2 <= 3421.76
 T104_R2_f10_ge: x_Bsmt_Qual >= 2
 T104_L2_f20_le: x_Gr_Liv_Area + 2113.760001000001 t104_leaf2 <= 3421.76
 T104_L3_f20_le: x_Gr_Liv_Area + 1076.760001 t104_leaf3 <= 3421.76
 T104_R3_f10_ge: x_Bsmt_Qual >= 2
 T104_R3_f20_ge: x_Gr_Liv_Area - 1308 t104_leaf3 >= 0
 T104_R4_f20_ge: x_Gr_Liv_Area - 2345 t104_leaf4 >= 0
 T104_L4_f20_le: x_Gr_Liv_Area + 335.7600010000001 t104_leaf4 <= 3421.76
 T104_L4_f15_le: x_Total_Bsmt_SF + 1689.480001 t104_leaf4 <= 2514.48
 T104_R5_f20_ge: x_Gr_Liv_Area - 2345 t104_leaf5 >= 0
 T104_L5_f20_le: x_Gr_Liv_Area + 335.7600010000001 t104_leaf5 <= 3421.76
 T104_R5_f15_ge: x_Total_Bsmt_SF - 825 t104_leaf5 >= 0
 T104_R6_f20_ge: x_Gr_Liv_Area - 2345 t104_leaf6 >= 0
 T104_R6_f20_ge: x_Gr_Liv_Area - 3086 t104_leaf6 >= 0
 TREE_105_ONEHOT: t105_leaf0 + t105_leaf1 + t105_leaf2 + t105_leaf3
   + t105_leaf4 + t105_leaf5 = 1
 T105_L0_f3_le: x_Overall_Qual + 999991.000001 t105_leaf0 <= 1e+06
 T105_L0_f44_le: x_Yr_Sold + 997992.000001 t105_leaf0 <= 1e+06
 T105_L0_f4_le: x_Overall_Cond + 999996.000001 t105_leaf0 <= 1e+06
 T105_L1_f3_le: x_Overall_Qual + 999991.000001 t105_leaf1 <= 1e+06
 T105_L1_f44_le: x_Yr_Sold + 997992.000001 t105_leaf1 <= 1e+06
 T105_R1_f4_ge: x_Overall_Cond - 4 t105_leaf1 >= 0
 T105_L2_f3_le: x_Overall_Qual + 999991.000001 t105_leaf2 <= 1e+06
 T105_R2_f44_ge: x_Yr_Sold - 2008 t105_leaf2 >= 0
 T105_L2_f11_le: x_Bsmt_Cond + 2.000001 t105_leaf2 <= 4
 T105_L3_f3_le: x_Overall_Qual + 999991.000001 t105_leaf3 <= 1e+06
 T105_R3_f44_ge: x_Yr_Sold - 2008 t105_leaf3 >= 0
 T105_R3_f11_ge: x_Bsmt_Cond - 3 t105_leaf3 >= -1
 T105_R4_f3_ge: x_Overall_Qual - 9 t105_leaf4 >= 0
 T105_L4_f33_le: x_Garage_Area + 122.7500010000001 t105_leaf4 <= 1163.75
 T105_R5_f3_ge: x_Overall_Qual - 9 t105_leaf5 >= 0
 T105_R5_f33_ge: x_Garage_Area - 1041 t105_leaf5 >= 0
 TREE_106_ONEHOT: t106_leaf0 + t106_leaf1 + t106_leaf2 + t106_leaf3
   + t106_leaf4 + t106_leaf5 + t106_leaf6 + t106_leaf7 = 1
 T106_L0_f1_le: x_Lot_Area <= 11183.999999
 T106_L0_f6_le: x_Year_Remod/Add + 998023.000001 t106_leaf0 <= 1e+06
 T106_L0_f5_le: x_Year_Built + 998075.000001 t106_leaf0 <= 1e+06
 T106_L1_f1_le: x_Lot_Area <= 11183.999999
 T106_L1_f6_le: x_Year_Remod/Add + 998023.000001 t106_leaf1 <= 1e+06
 T106_R1_f5_ge: x_Year_Built - 1925 t106_leaf1 >= 0
 T106_L2_f1_le: x_Lot_Area <= 11183.999999
 T106_R2_f6_ge: x_Year_Remod/Add - 1977 t106_leaf2 >= 0
 T106_L2_f1_le: x_Lot_Area + 3325.000001 t106_leaf2 <= 7000
 T106_L3_f1_le: x_Lot_Area <= 11183.999999
 T106_R3_f6_ge: x_Year_Remod/Add - 1977 t106_leaf3 >= 0
 T106_R3_f1_ge: x_Lot_Area >= 3675
 T106_R4_f1_ge: x_Lot_Area - 4184 t106_leaf4 >= 7000
 T106_L4_f20_le: x_Gr_Liv_Area + 1393.760001 t106_leaf4 <= 3421.76
 T106_L4_f25_le: x_Bedroom_AbvGr + 3.000001 t106_leaf4 <= 7
 T106_R5_f1_ge: x_Lot_Area - 4184 t106_leaf5 >= 7000
 T106_L5_f20_le: x_Gr_Liv_Area + 1393.760001 t106_leaf5 <= 3421.76
 T106_R5_f25_ge: x_Bedroom_AbvGr - 4 t106_leaf5 >= 0
 T106_R6_f1_ge: x_Lot_Area - 4184 t106_leaf6 >= 7000
 T106_R6_f20_ge: x_Gr_Liv_Area - 2028 t106_leaf6 >= 0
 T106_L6_f77_le: x_Land_Contour_Bnk + 1e-06 t106_leaf6 <= 1
 T106_R7_f1_ge: x_Lot_Area - 4184 t106_leaf7 >= 7000
 T106_R7_f20_ge: x_Gr_Liv_Area - 2028 t106_leaf7 >= 0
 T106_R7_f77_ge: x_Land_Contour_Bnk - t106_leaf7 >= 0
 TREE_107_ONEHOT: t107_leaf0 + t107_leaf1 + t107_leaf2 + t107_leaf3
   + t107_leaf4 + t107_leaf5 + t107_leaf6 + t107_leaf7 = 1
 T107_L0_f20_le: x_Gr_Liv_Area + 2275.760001000001 t107_leaf0 <= 3421.76
 T107_L0_f100_le: x_Neighborhood_IDOTRR + 1e-06 t107_leaf0 <= 1
 T107_L0_f0_le: x_Lot_Frontage + 144.530001 t107_leaf0 <= 175.53
 T107_L1_f20_le: x_Gr_Liv_Area + 2275.760001000001 t107_leaf1 <= 3421.76
 T107_L1_f100_le: x_Neighborhood_IDOTRR + 1e-06 t107_leaf1 <= 1
 T107_R1_f0_ge: x_Lot_Frontage - 31 t107_leaf1 >= 0
 T107_L2_f20_le: x_Gr_Liv_Area + 2275.760001000001 t107_leaf2 <= 3421.76
 T107_R2_f100_ge: x_Neighborhood_IDOTRR - t107_leaf2 >= 0
 T107_L2_f0_le: x_Lot_Frontage + 109.530001 t107_leaf2 <= 175.53
 T107_L3_f20_le: x_Gr_Liv_Area + 2275.760001000001 t107_leaf3 <= 3421.76
 T107_R3_f100_ge: x_Neighborhood_IDOTRR - t107_leaf3 >= 0
 T107_R3_f0_ge: x_Lot_Frontage - 66 t107_leaf3 >= 0
 T107_R4_f20_ge: x_Gr_Liv_Area - 1146 t107_leaf4 >= 0
 T107_L4_f12_le: x_BsmtFin_SF_1 + 1696.500001 t107_leaf4 <= 1984.5
 T107_L4_f44_le: x_Yr_Sold + 997992.000001 t107_leaf4 <= 1e+06
 T107_R5_f20_ge: x_Gr_Liv_Area - 1146 t107_leaf5 >= 0
 T107_L5_f12_le: x_BsmtFin_SF_1 + 1696.500001 t107_leaf5 <= 1984.5
 T107_R5_f44_ge: x_Yr_Sold - 2008 t107_leaf5 >= 0
 T107_R6_f20_ge: x_Gr_Liv_Area - 1146 t107_leaf6 >= 0
 T107_R6_f12_ge: x_BsmtFin_SF_1 - 288 t107_leaf6 >= 0
 T107_L6_f44_le: x_Yr_Sold + 997992.000001 t107_leaf6 <= 1e+06
 T107_R7_f20_ge: x_Gr_Liv_Area - 1146 t107_leaf7 >= 0
 T107_R7_f12_ge: x_BsmtFin_SF_1 - 288 t107_leaf7 >= 0
 T107_R7_f44_ge: x_Yr_Sold - 2008 t107_leaf7 >= 0
 TREE_108_ONEHOT: t108_leaf0 + t108_leaf1 + t108_leaf2 + t108_leaf3
   + t108_leaf4 + t108_leaf5 + t108_leaf6 = 1
 T108_L0_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t108_leaf0 <= 1984.5
 T108_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t108_leaf0 <= 1
 T108_L0_f12_le: x_BsmtFin_SF_1 + 1692.500001 t108_leaf0 <= 1984.5
 T108_L1_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t108_leaf1 <= 1984.5
 T108_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t108_leaf1 <= 1
 T108_R1_f12_ge: x_BsmtFin_SF_1 - 292 t108_leaf1 >= 0
 T108_L2_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t108_leaf2 <= 1984.5
 T108_R2_f293_ge: x_Sale_Condition_Abnorml - t108_leaf2 >= 0
 T108_L2_f3_le: x_Overall_Qual + 999995.000001 t108_leaf2 <= 1e+06
 T108_L3_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t108_leaf3 <= 1984.5
 T108_R3_f293_ge: x_Sale_Condition_Abnorml - t108_leaf3 >= 0
 T108_R3_f3_ge: x_Overall_Qual - 5 t108_leaf3 >= 0
 T108_R4_f12_ge: x_BsmtFin_SF_1 - 1087 t108_leaf4 >= 0
 T108_L4_f103_le: x_Neighborhood_Mitchel + 1e-06 t108_leaf4 <= 1
 T108_L4_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t108_leaf4 <= 1
 T108_R5_f12_ge: x_BsmtFin_SF_1 - 1087 t108_leaf5 >= 0
 T108_L5_f103_le: x_Neighborhood_Mitchel + 1e-06 t108_leaf5 <= 1
 T108_R5_f207_ge: x_Bsmt_Exposure_Gd - t108_leaf5 >= 0
 T108_R6_f12_ge: x_BsmtFin_SF_1 - 1087 t108_leaf6 >= 0
 T108_R6_f103_ge: x_Neighborhood_Mitchel - t108_leaf6 >= 0
 TREE_109_ONEHOT: t109_leaf0 + t109_leaf1 + t109_leaf2 + t109_leaf3
   + t109_leaf4 + t109_leaf5 = 1
 T109_L0_f20_le: x_Gr_Liv_Area + 1158.760001 t109_leaf0 <= 3421.76
 T109_L0_f44_le: x_Yr_Sold + 997992.000001 t109_leaf0 <= 1e+06
 T109_L0_f12_le: x_BsmtFin_SF_1 + 1699.500001 t109_leaf0 <= 1984.5
 T109_L1_f20_le: x_Gr_Liv_Area + 1158.760001 t109_leaf1 <= 3421.76
 T109_L1_f44_le: x_Yr_Sold + 997992.000001 t109_leaf1 <= 1e+06
 T109_R1_f12_ge: x_BsmtFin_SF_1 - 285 t109_leaf1 >= 0
 T109_L2_f20_le: x_Gr_Liv_Area + 1158.760001 t109_leaf2 <= 3421.76
 T109_R2_f44_ge: x_Yr_Sold - 2008 t109_leaf2 >= 0
 T109_L2_f293_le: x_Sale_Condition_Abnorml + 1e-06 t109_leaf2 <= 1
 T109_L3_f20_le: x_Gr_Liv_Area + 1158.760001 t109_leaf3 <= 3421.76
 T109_R3_f44_ge: x_Yr_Sold - 2008 t109_leaf3 >= 0
 T109_R3_f293_ge: x_Sale_Condition_Abnorml - t109_leaf3 >= 0
 T109_R4_f20_ge: x_Gr_Liv_Area - 2263 t109_leaf4 >= 0
 T109_L4_f3_le: x_Overall_Qual + 999991.000001 t109_leaf4 <= 1e+06
 T109_R5_f20_ge: x_Gr_Liv_Area - 2263 t109_leaf5 >= 0
 T109_R5_f3_ge: x_Overall_Qual - 9 t109_leaf5 >= 0
 TREE_110_ONEHOT: t110_leaf0 + t110_leaf1 + t110_leaf2 + t110_leaf3
   + t110_leaf4 + t110_leaf5 = 1
 T110_L0_f3_le: x_Overall_Qual + 999991.000001 t110_leaf0 <= 1e+06
 T110_L0_f20_le: x_Gr_Liv_Area + 2517.760001 t110_leaf0 <= 3421.76
 T110_L0_f86_le: x_Land_Slope_Gtl + 1e-06 t110_leaf0 <= 1
 T110_L1_f3_le: x_Overall_Qual + 999991.000001 t110_leaf1 <= 1e+06
 T110_L1_f20_le: x_Gr_Liv_Area + 2517.760001 t110_leaf1 <= 3421.76
 T110_R1_f86_ge: x_Land_Slope_Gtl - t110_leaf1 >= 0
 T110_L2_f3_le: x_Overall_Qual + 999991.000001 t110_leaf2 <= 1e+06
 T110_R2_f20_ge: x_Gr_Liv_Area - 904 t110_leaf2 >= 0
 T110_L2_f33_le: x_Garage_Area + 983.750001 t110_leaf2 <= 1163.75
 T110_L3_f3_le: x_Overall_Qual + 999991.000001 t110_leaf3 <= 1e+06
 T110_R3_f20_ge: x_Gr_Liv_Area - 904 t110_leaf3 >= 0
 T110_R3_f33_ge: x_Garage_Area - 180 t110_leaf3 >= 0
 T110_R4_f3_ge: x_Overall_Qual - 9 t110_leaf4 >= 0
 T110_L4_f37_le: x_Open_Porch_SF + 252.810001 t110_leaf4 <= 488.81
 T110_R5_f3_ge: x_Overall_Qual - 9 t110_leaf5 >= 0
 T110_R5_f37_ge: x_Open_Porch_SF - 236 t110_leaf5 >= 0
 TREE_111_ONEHOT: t111_leaf0 + t111_leaf1 + t111_leaf2 + t111_leaf3
   + t111_leaf4 + t111_leaf5 + t111_leaf6 + t111_leaf7 = 1
 T111_L0_f44_le: x_Yr_Sold + 997992.000001 t111_leaf0 <= 1e+06
 T111_L0_f30_le: x_Fireplace_Qu + 4.000001 t111_leaf0 <= 4
 T111_L0_f4_le: x_Overall_Cond + 999996.000001 t111_leaf0 <= 1e+06
 T111_L1_f44_le: x_Yr_Sold + 997992.000001 t111_leaf1 <= 1e+06
 T111_L1_f30_le: x_Fireplace_Qu + 4.000001 t111_leaf1 <= 4
 T111_R1_f4_ge: x_Overall_Cond - 4 t111_leaf1 >= 0
 T111_L2_f44_le: x_Yr_Sold + 997992.000001 t111_leaf2 <= 1e+06
 T111_R2_f30_ge: x_Fireplace_Qu >= 0
 T111_L2_f5_le: x_Year_Built + 998086.000001 t111_leaf2 <= 1e+06
 T111_L3_f44_le: x_Yr_Sold + 997992.000001 t111_leaf3 <= 1e+06
 T111_R3_f30_ge: x_Fireplace_Qu >= 0
 T111_R3_f5_ge: x_Year_Built - 1914 t111_leaf3 >= 0
 T111_R4_f44_ge: x_Yr_Sold - 2008 t111_leaf4 >= 0
 T111_L4_f21_le: x_Bsmt_Full_Bath + 1.020001 t111_leaf4 <= 2.02
 T111_L4_f100_le: x_Neighborhood_IDOTRR + 1e-06 t111_leaf4 <= 1
 T111_R5_f44_ge: x_Yr_Sold - 2008 t111_leaf5 >= 0
 T111_L5_f21_le: x_Bsmt_Full_Bath + 1.020001 t111_leaf5 <= 2.02
 T111_R5_f100_ge: x_Neighborhood_IDOTRR - t111_leaf5 >= 0
 T111_R6_f44_ge: x_Yr_Sold - 2008 t111_leaf6 >= 0
 T111_R6_f21_ge: x_Bsmt_Full_Bath - t111_leaf6 >= 0
 T111_L6_f5_le: x_Year_Built + 998079.000001 t111_leaf6 <= 1e+06
 T111_R7_f44_ge: x_Yr_Sold - 2008 t111_leaf7 >= 0
 T111_R7_f21_ge: x_Bsmt_Full_Bath - t111_leaf7 >= 0
 T111_R7_f5_ge: x_Year_Built - 1921 t111_leaf7 >= 0
 TREE_112_ONEHOT: t112_leaf0 + t112_leaf1 + t112_leaf2 + t112_leaf3
   + t112_leaf4 + t112_leaf5 = 1
 T112_L0_f18_le: x_2nd_Flr_SF + 658.540001 t112_leaf0 <= 1641.54
 T112_L0_f73_le: x_Lot_Shape_IR1 + 1e-06 t112_leaf0 <= 1
 T112_L0_f213_le: x_BsmtFin_Type_1_GLQ + 1e-06 t112_leaf0 <= 1
 T112_L1_f18_le: x_2nd_Flr_SF + 658.540001 t112_leaf1 <= 1641.54
 T112_L1_f73_le: x_Lot_Shape_IR1 + 1e-06 t112_leaf1 <= 1
 T112_R1_f213_ge: x_BsmtFin_Type_1_GLQ - t112_leaf1 >= 0
 T112_L2_f18_le: x_2nd_Flr_SF + 658.540001 t112_leaf2 <= 1641.54
 T112_R2_f73_ge: x_Lot_Shape_IR1 - t112_leaf2 >= 0
 T112_L2_f17_le: x_1st_Flr_SF + 1252.900001 t112_leaf2 <= 2534.9
 T112_L3_f18_le: x_2nd_Flr_SF + 658.540001 t112_leaf3 <= 1641.54
 T112_R3_f73_ge: x_Lot_Shape_IR1 - t112_leaf3 >= 0
 T112_R3_f17_ge: x_1st_Flr_SF - 1282 t112_leaf3 >= 0
 T112_R4_f18_ge: x_2nd_Flr_SF - 983 t112_leaf4 >= 0
 T112_L4_f17_le: x_1st_Flr_SF + 1169.900001 t112_leaf4 <= 2534.9
 T112_R5_f18_ge: x_2nd_Flr_SF - 983 t112_leaf5 >= 0
 T112_R5_f17_ge: x_1st_Flr_SF - 1365 t112_leaf5 >= 0
 TREE_113_ONEHOT: t113_leaf0 + t113_leaf1 + t113_leaf2 + t113_leaf3
   + t113_leaf4 + t113_leaf5 + t113_leaf6 + t113_leaf7 = 1
 T113_L0_f20_le: x_Gr_Liv_Area + 1681.760001 t113_leaf0 <= 3421.76
 T113_L0_f20_le: x_Gr_Liv_Area + 2573.760001 t113_leaf0 <= 3421.76
 T113_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t113_leaf0 <= 1
 T113_L1_f20_le: x_Gr_Liv_Area + 1681.760001 t113_leaf1 <= 3421.76
 T113_L1_f20_le: x_Gr_Liv_Area + 2573.760001 t113_leaf1 <= 3421.76
 T113_R1_f293_ge: x_Sale_Condition_Abnorml - t113_leaf1 >= 0
 T113_L2_f20_le: x_Gr_Liv_Area + 1681.760001 t113_leaf2 <= 3421.76
 T113_R2_f20_ge: x_Gr_Liv_Area - 848 t113_leaf2 >= 0
 T113_L2_f31_le: x_Garage_Yr_Blt + 998040.000001 t113_leaf2 <= 1e+06
 T113_L3_f20_le: x_Gr_Liv_Area + 1681.760001 t113_leaf3 <= 3421.76
 T113_R3_f20_ge: x_Gr_Liv_Area - 848 t113_leaf3 >= 0
 T113_R3_f31_ge: x_Garage_Yr_Blt - 1960 t113_leaf3 >= 0
 T113_R4_f20_ge: x_Gr_Liv_Area - 1740 t113_leaf4 >= 0
 T113_L4_f5_le: x_Year_Built + 998089.000001 t113_leaf4 <= 1e+06
 T113_L4_f17_le: x_1st_Flr_SF + 1551.900001 t113_leaf4 <= 2534.9
 T113_R5_f20_ge: x_Gr_Liv_Area - 1740 t113_leaf5 >= 0
 T113_L5_f5_le: x_Year_Built + 998089.000001 t113_leaf5 <= 1e+06
 T113_R5_f17_ge: x_1st_Flr_SF - 983 t113_leaf5 >= 0
 T113_R6_f20_ge: x_Gr_Liv_Area - 1740 t113_leaf6 >= 0
 T113_R6_f5_ge: x_Year_Built - 1911 t113_leaf6 >= 0
 T113_L6_f15_le: x_Total_Bsmt_SF + 664.4800010000001 t113_leaf6 <= 2514.48
 T113_R7_f20_ge: x_Gr_Liv_Area - 1740 t113_leaf7 >= 0
 T113_R7_f5_ge: x_Year_Built - 1911 t113_leaf7 >= 0
 T113_R7_f15_ge: x_Total_Bsmt_SF - 1850 t113_leaf7 >= 0
 TREE_114_ONEHOT: t114_leaf0 + t114_leaf1 + t114_leaf2 + t114_leaf3
   + t114_leaf4 + t114_leaf5 + t114_leaf6 = 1
 T114_L0_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t114_leaf0 <= 1984.5
 T114_L0_f36_le: x_Wood_Deck_SF + 326.900001 t114_leaf0 <= 525.9
 T114_L0_f44_le: x_Yr_Sold + 997992.000001 t114_leaf0 <= 1e+06
 T114_L1_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t114_leaf1 <= 1984.5
 T114_L1_f36_le: x_Wood_Deck_SF + 326.900001 t114_leaf1 <= 525.9
 T114_R1_f44_ge: x_Yr_Sold - 2008 t114_leaf1 >= 0
 T114_L2_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t114_leaf2 <= 1984.5
 T114_R2_f36_ge: x_Wood_Deck_SF - 199 t114_leaf2 >= 0
 T114_L2_f17_le: x_1st_Flr_SF + 1440.900001 t114_leaf2 <= 2534.9
 T114_L3_f12_le: x_BsmtFin_SF_1 + 897.5000010000001 t114_leaf3 <= 1984.5
 T114_R3_f36_ge: x_Wood_Deck_SF - 199 t114_leaf3 >= 0
 T114_R3_f17_ge: x_1st_Flr_SF - 1094 t114_leaf3 >= 0
 T114_R4_f12_ge: x_BsmtFin_SF_1 - 1087 t114_leaf4 >= 0
 T114_L4_f111_le: x_Neighborhood_Sawyer + 1e-06 t114_leaf4 <= 1
 T114_L4_f183_le: x_Exterior_2nd_HdBoard + 1e-06 t114_leaf4 <= 1
 T114_R5_f12_ge: x_BsmtFin_SF_1 - 1087 t114_leaf5 >= 0
 T114_L5_f111_le: x_Neighborhood_Sawyer + 1e-06 t114_leaf5 <= 1
 T114_R5_f183_ge: x_Exterior_2nd_HdBoard - t114_leaf5 >= 0
 T114_R6_f12_ge: x_BsmtFin_SF_1 - 1087 t114_leaf6 >= 0
 T114_R6_f111_ge: x_Neighborhood_Sawyer - t114_leaf6 >= 0
 TREE_115_ONEHOT: t115_leaf0 + t115_leaf1 + t115_leaf2 + t115_leaf3
   + t115_leaf4 = 1
 T115_L0_f3_le: x_Overall_Qual + 999991.000001 t115_leaf0 <= 1e+06
 T115_L0_f12_le: x_BsmtFin_SF_1 + 1512.500001 t115_leaf0 <= 1984.5
 T115_L0_f16_le: x_Heating_QC + 1.000001 t115_leaf0 <= 4
 T115_L1_f3_le: x_Overall_Qual + 999991.000001 t115_leaf1 <= 1e+06
 T115_L1_f12_le: x_BsmtFin_SF_1 + 1512.500001 t115_leaf1 <= 1984.5
 T115_R1_f16_ge: x_Heating_QC - t115_leaf1 >= 2
 T115_L2_f3_le: x_Overall_Qual + 999991.000001 t115_leaf2 <= 1e+06
 T115_R2_f12_ge: x_BsmtFin_SF_1 - 472 t115_leaf2 >= 0
 T115_L2_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t115_leaf2 <= 1
 T115_L3_f3_le: x_Overall_Qual + 999991.000001 t115_leaf3 <= 1e+06
 T115_R3_f12_ge: x_BsmtFin_SF_1 - 472 t115_leaf3 >= 0
 T115_R3_f216_ge: x_BsmtFin_Type_1_Rec - t115_leaf3 >= 0
 T115_R4_f3_ge: x_Overall_Qual - 9 t115_leaf4 >= 0
 TREE_116_ONEHOT: t116_leaf0 + t116_leaf1 + t116_leaf2 + t116_leaf3
   + t116_leaf4 + t116_leaf5 + t116_leaf6 = 1
 T116_L0_f20_le: x_Gr_Liv_Area + 1094.760001 t116_leaf0 <= 3421.76
 T116_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t116_leaf0 <= 1
 T116_L0_f15_le: x_Total_Bsmt_SF + 1903.480001 t116_leaf0 <= 2514.48
 T116_L1_f20_le: x_Gr_Liv_Area + 1094.760001 t116_leaf1 <= 3421.76
 T116_L1_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t116_leaf1 <= 1
 T116_R1_f15_ge: x_Total_Bsmt_SF - 611 t116_leaf1 >= 0
 T116_L2_f20_le: x_Gr_Liv_Area + 1094.760001 t116_leaf2 <= 3421.76
 T116_R2_f207_ge: x_Bsmt_Exposure_Gd - t116_leaf2 >= 0
 T116_L2_f0_le: x_Lot_Frontage + 134.530001 t116_leaf2 <= 175.53
 T116_L3_f20_le: x_Gr_Liv_Area + 1094.760001 t116_leaf3 <= 3421.76
 T116_R3_f207_ge: x_Bsmt_Exposure_Gd - t116_leaf3 >= 0
 T116_R3_f0_ge: x_Lot_Frontage - 41 t116_leaf3 >= 0
 T116_R4_f20_ge: x_Gr_Liv_Area - 2327 t116_leaf4 >= 0
 T116_L4_f24_le: x_Half_Bath + 3.000001 t116_leaf4 <= 5
 T116_L4_f18_le: x_2nd_Flr_SF + 236.5400010000001 t116_leaf4 <= 1641.54
 T116_R5_f20_ge: x_Gr_Liv_Area - 2327 t116_leaf5 >= 0
 T116_L5_f24_le: x_Half_Bath + 3.000001 t116_leaf5 <= 5
 T116_R5_f18_ge: x_2nd_Flr_SF - 1405 t116_leaf5 >= 0
 T116_R6_f20_ge: x_Gr_Liv_Area - 2327 t116_leaf6 >= 0
 T116_R6_f24_ge: x_Half_Bath - 2 t116_leaf6 >= 0
 TREE_117_ONEHOT: t117_leaf0 + t117_leaf1 + t117_leaf2 + t117_leaf3
   + t117_leaf4 = 1
 T117_L0_f27_le: x_Kitchen_Qual + 1e-06 t117_leaf0 <= 4
 T117_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t117_leaf0 <= 1
 T117_L0_f1_le: x_Lot_Area <= 13906.999999
 T117_L1_f27_le: x_Kitchen_Qual + 1e-06 t117_leaf1 <= 4
 T117_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t117_leaf1 <= 1
 T117_R1_f1_ge: x_Lot_Area - 6907 t117_leaf1 >= 7000
 T117_L2_f27_le: x_Kitchen_Qual + 1e-06 t117_leaf2 <= 4
 T117_R2_f293_ge: x_Sale_Condition_Abnorml - t117_leaf2 >= 0
 T117_L2_f12_le: x_BsmtFin_SF_1 + 1968.500001 t117_leaf2 <= 1984.5
 T117_L3_f27_le: x_Kitchen_Qual + 1e-06 t117_leaf3 <= 4
 T117_R3_f293_ge: x_Sale_Condition_Abnorml - t117_leaf3 >= 0
 T117_R3_f12_ge: x_BsmtFin_SF_1 - 16 t117_leaf3 >= 0
 T117_R4_f27_ge: x_Kitchen_Qual - 2 t117_leaf4 >= 2
 TREE_118_ONEHOT: t118_leaf0 + t118_leaf1 + t118_leaf2 + t118_leaf3
   + t118_leaf4 + t118_leaf5 + t118_leaf6 = 1
 T118_L0_f20_le: x_Gr_Liv_Area + 1076.760001 t118_leaf0 <= 3421.76
 T118_L0_f12_le: x_BsmtFin_SF_1 + 1637.500001 t118_leaf0 <= 1984.5
 T118_L0_f16_le: x_Heating_QC + 1.000001 t118_leaf0 <= 4
 T118_L1_f20_le: x_Gr_Liv_Area + 1076.760001 t118_leaf1 <= 3421.76
 T118_L1_f12_le: x_BsmtFin_SF_1 + 1637.500001 t118_leaf1 <= 1984.5
 T118_R1_f16_ge: x_Heating_QC - t118_leaf1 >= 2
 T118_L2_f20_le: x_Gr_Liv_Area + 1076.760001 t118_leaf2 <= 3421.76
 T118_R2_f12_ge: x_BsmtFin_SF_1 - 347 t118_leaf2 >= 0
 T118_L2_f44_le: x_Yr_Sold + 997992.000001 t118_leaf2 <= 1e+06
 T118_L3_f20_le: x_Gr_Liv_Area + 1076.760001 t118_leaf3 <= 3421.76
 T118_R3_f12_ge: x_BsmtFin_SF_1 - 347 t118_leaf3 >= 0
 T118_R3_f44_ge: x_Yr_Sold - 2008 t118_leaf3 >= 0
 T118_R4_f20_ge: x_Gr_Liv_Area - 2345 t118_leaf4 >= 0
 T118_L4_f21_le: x_Bsmt_Full_Bath + 0.020001 t118_leaf4 <= 2.02
 T118_L4_f20_le: x_Gr_Liv_Area + 335.7600010000001 t118_leaf4 <= 3421.76
 T118_R5_f20_ge: x_Gr_Liv_Area - 2345 t118_leaf5 >= 0
 T118_L5_f21_le: x_Bsmt_Full_Bath + 0.020001 t118_leaf5 <= 2.02
 T118_R5_f20_ge: x_Gr_Liv_Area - 3086 t118_leaf5 >= 0
 T118_R6_f20_ge: x_Gr_Liv_Area - 2345 t118_leaf6 >= 0
 T118_R6_f21_ge: x_Bsmt_Full_Bath - 2 t118_leaf6 >= 0
 TREE_119_ONEHOT: t119_leaf0 + t119_leaf1 + t119_leaf2 + t119_leaf3
   + t119_leaf4 + t119_leaf5 + t119_leaf6 = 1
 T119_L0_f20_le: x_Gr_Liv_Area + 1028.760001 t119_leaf0 <= 3421.76
 T119_L0_f5_le: x_Year_Built + 997995.000001 t119_leaf0 <= 1e+06
 T119_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t119_leaf0 <= 1
 T119_L1_f20_le: x_Gr_Liv_Area + 1028.760001 t119_leaf1 <= 3421.76
 T119_L1_f5_le: x_Year_Built + 997995.000001 t119_leaf1 <= 1e+06
 T119_R1_f297_ge: x_Sale_Condition_Normal - t119_leaf1 >= 0
 T119_L2_f20_le: x_Gr_Liv_Area + 1028.760001 t119_leaf2 <= 3421.76
 T119_R2_f5_ge: x_Year_Built - 2005 t119_leaf2 >= 0
 T119_L2_f20_le: x_Gr_Liv_Area + 1813.760001 t119_leaf2 <= 3421.76
 T119_L3_f20_le: x_Gr_Liv_Area + 1028.760001 t119_leaf3 <= 3421.76
 T119_R3_f5_ge: x_Year_Built - 2005 t119_leaf3 >= 0
 T119_R3_f20_ge: x_Gr_Liv_Area - 1608 t119_leaf3 >= 0
 T119_R4_f20_ge: x_Gr_Liv_Area - 2393 t119_leaf4 >= 0
 T119_L4_f37_le: x_Open_Porch_SF + 201.810001 t119_leaf4 <= 488.81
 T119_L4_f18_le: x_2nd_Flr_SF + 270.5400010000001 t119_leaf4 <= 1641.54
 T119_R5_f20_ge: x_Gr_Liv_Area - 2393 t119_leaf5 >= 0
 T119_L5_f37_le: x_Open_Porch_SF + 201.810001 t119_leaf5 <= 488.81
 T119_R5_f18_ge: x_2nd_Flr_SF - 1371 t119_leaf5 >= 0
 T119_R6_f20_ge: x_Gr_Liv_Area - 2393 t119_leaf6 >= 0
 T119_R6_f37_ge: x_Open_Porch_SF - 287 t119_leaf6 >= 0
 TREE_120_ONEHOT: t120_leaf0 + t120_leaf1 + t120_leaf2 + t120_leaf3
   + t120_leaf4 + t120_leaf5 + t120_leaf6 = 1
 T120_L0_f20_le: x_Gr_Liv_Area + 2485.760001 t120_leaf0 <= 3421.76
 T120_L0_f9_le: x_Exter_Cond + 2.000001 t120_leaf0 <= 4
 T120_L1_f20_le: x_Gr_Liv_Area + 2485.760001 t120_leaf1 <= 3421.76
 T120_R1_f9_ge: x_Exter_Cond - 3 t120_leaf1 >= -1
 T120_L1_f5_le: x_Year_Built + 998077.000001 t120_leaf1 <= 1e+06
 T120_L2_f20_le: x_Gr_Liv_Area + 2485.760001 t120_leaf2 <= 3421.76
 T120_R2_f9_ge: x_Exter_Cond - 3 t120_leaf2 >= -1
 T120_R2_f5_ge: x_Year_Built - 1923 t120_leaf2 >= 0
 T120_R3_f20_ge: x_Gr_Liv_Area - 936 t120_leaf3 >= 0
 T120_L3_f1_le: x_Lot_Area <= 12098.999999
 T120_L3_f44_le: x_Yr_Sold + 997992.000001 t120_leaf3 <= 1e+06
 T120_R4_f20_ge: x_Gr_Liv_Area - 936 t120_leaf4 >= 0
 T120_L4_f1_le: x_Lot_Area <= 12098.999999
 T120_R4_f44_ge: x_Yr_Sold - 2008 t120_leaf4 >= 0
 T120_R5_f20_ge: x_Gr_Liv_Area - 936 t120_leaf5 >= 0
 T120_R5_f1_ge: x_Lot_Area - 5099 t120_leaf5 >= 7000
 T120_L5_f33_le: x_Garage_Area + 9.750001000000111 t120_leaf5 <= 1163.75
 T120_R6_f20_ge: x_Gr_Liv_Area - 936 t120_leaf6 >= 0
 T120_R6_f1_ge: x_Lot_Area - 5099 t120_leaf6 >= 7000
 T120_R6_f33_ge: x_Garage_Area - 1154 t120_leaf6 >= 0
 TREE_121_ONEHOT: t121_leaf0 + t121_leaf1 + t121_leaf2 + t121_leaf3
   + t121_leaf4 + t121_leaf5 + t121_leaf6 = 1
 T121_L0_f20_le: x_Gr_Liv_Area + 1158.760001 t121_leaf0 <= 3421.76
 T121_L0_f12_le: x_BsmtFin_SF_1 + 1957.500001 t121_leaf0 <= 1984.5
 T121_L0_f20_le: x_Gr_Liv_Area + 2266.760001000001 t121_leaf0 <= 3421.76
 T121_L1_f20_le: x_Gr_Liv_Area + 1158.760001 t121_leaf1 <= 3421.76
 T121_L1_f12_le: x_BsmtFin_SF_1 + 1957.500001 t121_leaf1 <= 1984.5
 T121_R1_f20_ge: x_Gr_Liv_Area - 1155 t121_leaf1 >= 0
 T121_L2_f20_le: x_Gr_Liv_Area + 1158.760001 t121_leaf2 <= 3421.76
 T121_R2_f12_ge: x_BsmtFin_SF_1 - 27 t121_leaf2 >= 0
 T121_L2_f20_le: x_Gr_Liv_Area + 2269.760001000001 t121_leaf2 <= 3421.76
 T121_L3_f20_le: x_Gr_Liv_Area + 1158.760001 t121_leaf3 <= 3421.76
 T121_R3_f12_ge: x_BsmtFin_SF_1 - 27 t121_leaf3 >= 0
 T121_R3_f20_ge: x_Gr_Liv_Area - 1152 t121_leaf3 >= 0
 T121_R4_f20_ge: x_Gr_Liv_Area - 2263 t121_leaf4 >= 0
 T121_L4_f37_le: x_Open_Porch_SF + 222.810001 t121_leaf4 <= 488.81
 T121_L4_f15_le: x_Total_Bsmt_SF + 762.4800010000001 t121_leaf4 <= 2514.48
 T121_R5_f20_ge: x_Gr_Liv_Area - 2263 t121_leaf5 >= 0
 T121_L5_f37_le: x_Open_Porch_SF + 222.810001 t121_leaf5 <= 488.81
 T121_R5_f15_ge: x_Total_Bsmt_SF - 1752 t121_leaf5 >= 0
 T121_R6_f20_ge: x_Gr_Liv_Area - 2263 t121_leaf6 >= 0
 T121_R6_f37_ge: x_Open_Porch_SF - 266 t121_leaf6 >= 0
 TREE_122_ONEHOT: t122_leaf0 + t122_leaf1 + t122_leaf2 + t122_leaf3
   + t122_leaf4 + t122_leaf5 + t122_leaf6 + t122_leaf7 = 1
 T122_L0_f44_le: x_Yr_Sold + 997992.000001 t122_leaf0 <= 1e+06
 T122_L0_f1_le: x_Lot_Area <= 9589.999999
 T122_L0_f46_le: x_MS_SubClass_30 + 1e-06 t122_leaf0 <= 1
 T122_L1_f44_le: x_Yr_Sold + 997992.000001 t122_leaf1 <= 1e+06
 T122_L1_f1_le: x_Lot_Area <= 9589.999999
 T122_R1_f46_ge: x_MS_SubClass_30 - t122_leaf1 >= 0
 T122_L2_f44_le: x_Yr_Sold + 997992.000001 t122_leaf2 <= 1e+06
 T122_R2_f1_ge: x_Lot_Area - 2590 t122_leaf2 >= 7000
 T122_L2_f272_le: x_Mo_Sold_2 + 1e-06 t122_leaf2 <= 1
 T122_L3_f44_le: x_Yr_Sold + 997992.000001 t122_leaf3 <= 1e+06
 T122_R3_f1_ge: x_Lot_Area - 2590 t122_leaf3 >= 7000
 T122_R3_f272_ge: x_Mo_Sold_2 - t122_leaf3 >= 0
 T122_R4_f44_ge: x_Yr_Sold - 2008 t122_leaf4 >= 0
 T122_L4_f31_le: x_Garage_Yr_Blt + 998043.000001 t122_leaf4 <= 1e+06
 T122_L4_f297_le: x_Sale_Condition_Normal + 1e-06 t122_leaf4 <= 1
 T122_R5_f44_ge: x_Yr_Sold - 2008 t122_leaf5 >= 0
 T122_L5_f31_le: x_Garage_Yr_Blt + 998043.000001 t122_leaf5 <= 1e+06
 T122_R5_f297_ge: x_Sale_Condition_Normal - t122_leaf5 >= 0
 T122_R6_f44_ge: x_Yr_Sold - 2008 t122_leaf6 >= 0
 T122_R6_f31_ge: x_Garage_Yr_Blt - 1957 t122_leaf6 >= 0
 T122_L6_f21_le: x_Bsmt_Full_Bath + 1.020001 t122_leaf6 <= 2.02
 T122_R7_f44_ge: x_Yr_Sold - 2008 t122_leaf7 >= 0
 T122_R7_f31_ge: x_Garage_Yr_Blt - 1957 t122_leaf7 >= 0
 T122_R7_f21_ge: x_Bsmt_Full_Bath - t122_leaf7 >= 0
 TREE_123_ONEHOT: t123_leaf0 + t123_leaf1 + t123_leaf2 + t123_leaf3
   + t123_leaf4 = 1
 T123_L0_f18_le: x_2nd_Flr_SF + 236.5400010000001 t123_leaf0 <= 1641.54
 T123_L0_f1_le: x_Lot_Area + 1000.000001 t123_leaf0 <= 7000
 T123_L0_f33_le: x_Garage_Area + 958.750001 t123_leaf0 <= 1163.75
 T123_L1_f18_le: x_2nd_Flr_SF + 236.5400010000001 t123_leaf1 <= 1641.54
 T123_L1_f1_le: x_Lot_Area + 1000.000001 t123_leaf1 <= 7000
 T123_R1_f33_ge: x_Garage_Area - 205 t123_leaf1 >= 0
 T123_L2_f18_le: x_2nd_Flr_SF + 236.5400010000001 t123_leaf2 <= 1641.54
 T123_R2_f1_ge: x_Lot_Area >= 6000
 T123_L2_f21_le: x_Bsmt_Full_Bath + 1.020001 t123_leaf2 <= 2.02
 T123_L3_f18_le: x_2nd_Flr_SF + 236.5400010000001 t123_leaf3 <= 1641.54
 T123_R3_f1_ge: x_Lot_Area >= 6000
 T123_R3_f21_ge: x_Bsmt_Full_Bath - t123_leaf3 >= 0
 T123_R4_f18_ge: x_2nd_Flr_SF - 1405 t123_leaf4 >= 0
 TREE_124_ONEHOT: t124_leaf0 + t124_leaf1 + t124_leaf2 + t124_leaf3
   + t124_leaf4 + t124_leaf5 + t124_leaf6 = 1
 T124_L0_f44_le: x_Yr_Sold + 997992.000001 t124_leaf0 <= 1e+06
 T124_L0_f3_le: x_Overall_Qual + 999991.000001 t124_leaf0 <= 1e+06
 T124_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t124_leaf0 <= 1
 T124_L1_f44_le: x_Yr_Sold + 997992.000001 t124_leaf1 <= 1e+06
 T124_L1_f3_le: x_Overall_Qual + 999991.000001 t124_leaf1 <= 1e+06
 T124_R1_f95_ge: x_Neighborhood_Crawfor - t124_leaf1 >= 0
 T124_L2_f44_le: x_Yr_Sold + 997992.000001 t124_leaf2 <= 1e+06
 T124_R2_f3_ge: x_Overall_Qual - 9 t124_leaf2 >= 0
 T124_R3_f44_ge: x_Yr_Sold - 2008 t124_leaf3 >= 0
 T124_L3_f6_le: x_Year_Remod/Add + 998006.000001 t124_leaf3 <= 1e+06
 T124_L3_f29_le: x_Fireplaces + 1.000001 t124_leaf3 <= 3
 T124_R4_f44_ge: x_Yr_Sold - 2008 t124_leaf4 >= 0
 T124_L4_f6_le: x_Year_Remod/Add + 998006.000001 t124_leaf4 <= 1e+06
 T124_R4_f29_ge: x_Fireplaces - 2 t124_leaf4 >= 0
 T124_R5_f44_ge: x_Yr_Sold - 2008 t124_leaf5 >= 0
 T124_R5_f6_ge: x_Year_Remod/Add - 1994 t124_leaf5 >= 0
 T124_L5_f85_le: x_Lot_Config_Inside + 1e-06 t124_leaf5 <= 1
 T124_R6_f44_ge: x_Yr_Sold - 2008 t124_leaf6 >= 0
 T124_R6_f6_ge: x_Year_Remod/Add - 1994 t124_leaf6 >= 0
 T124_R6_f85_ge: x_Lot_Config_Inside - t124_leaf6 >= 0
 TREE_125_ONEHOT: t125_leaf0 + t125_leaf1 + t125_leaf2 + t125_leaf3
   + t125_leaf4 + t125_leaf5 + t125_leaf6 + t125_leaf7 = 1
 T125_L0_f15_le: x_Total_Bsmt_SF + 1039.480001 t125_leaf0 <= 2514.48
 T125_L0_f16_le: x_Heating_QC + 2.000001 t125_leaf0 <= 4
 T125_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t125_leaf0 <= 1
 T125_L1_f15_le: x_Total_Bsmt_SF + 1039.480001 t125_leaf1 <= 2514.48
 T125_L1_f16_le: x_Heating_QC + 2.000001 t125_leaf1 <= 4
 T125_R1_f297_ge: x_Sale_Condition_Normal - t125_leaf1 >= 0
 T125_L2_f15_le: x_Total_Bsmt_SF + 1039.480001 t125_leaf2 <= 2514.48
 T125_R2_f16_ge: x_Heating_QC >= 2
 T125_L2_f44_le: x_Yr_Sold + 997993.000001 t125_leaf2 <= 1e+06
 T125_L3_f15_le: x_Total_Bsmt_SF + 1039.480001 t125_leaf3 <= 2514.48
 T125_R3_f16_ge: x_Heating_QC >= 2
 T125_R3_f44_ge: x_Yr_Sold - 2007 t125_leaf3 >= 0
 T125_R4_f15_ge: x_Total_Bsmt_SF - 1475 t125_leaf4 >= 0
 T125_L4_f29_le: x_Fireplaces + 1.000001 t125_leaf4 <= 3
 T125_L4_f289_le: x_Sale_Type_New + 1e-06 t125_leaf4 <= 1
 T125_R5_f15_ge: x_Total_Bsmt_SF - 1475 t125_leaf5 >= 0
 T125_L5_f29_le: x_Fireplaces + 1.000001 t125_leaf5 <= 3
 T125_R5_f289_ge: x_Sale_Type_New - t125_leaf5 >= 0
 T125_R6_f15_ge: x_Total_Bsmt_SF - 1475 t125_leaf6 >= 0
 T125_R6_f29_ge: x_Fireplaces - 2 t125_leaf6 >= 0
 T125_L6_f77_le: x_Land_Contour_Bnk + 1e-06 t125_leaf6 <= 1
 T125_R7_f15_ge: x_Total_Bsmt_SF - 1475 t125_leaf7 >= 0
 T125_R7_f29_ge: x_Fireplaces - 2 t125_leaf7 >= 0
 T125_R7_f77_ge: x_Land_Contour_Bnk - t125_leaf7 >= 0
 TREE_126_ONEHOT: t126_leaf0 + t126_leaf1 + t126_leaf2 + t126_leaf3
   + t126_leaf4 + t126_leaf5 + t126_leaf6 = 1
 T126_L0_f12_le: x_BsmtFin_SF_1 + 1796.500001 t126_leaf0 <= 1984.5
 T126_L0_f36_le: x_Wood_Deck_SF + 367.900001 t126_leaf0 <= 525.9
 T126_L0_f16_le: x_Heating_QC + 1.000001 t126_leaf0 <= 4
 T126_L1_f12_le: x_BsmtFin_SF_1 + 1796.500001 t126_leaf1 <= 1984.5
 T126_L1_f36_le: x_Wood_Deck_SF + 367.900001 t126_leaf1 <= 525.9
 T126_R1_f16_ge: x_Heating_QC - t126_leaf1 >= 2
 T126_L2_f12_le: x_BsmtFin_SF_1 + 1796.500001 t126_leaf2 <= 1984.5
 T126_R2_f36_ge: x_Wood_Deck_SF - 158 t126_leaf2 >= 0
 T126_R3_f12_ge: x_BsmtFin_SF_1 - 188 t126_leaf3 >= 0
 T126_L3_f15_le: x_Total_Bsmt_SF + 1095.480001 t126_leaf3 <= 2514.48
 T126_L3_f0_le: x_Lot_Frontage + 133.530001 t126_leaf3 <= 175.53
 T126_R4_f12_ge: x_BsmtFin_SF_1 - 188 t126_leaf4 >= 0
 T126_L4_f15_le: x_Total_Bsmt_SF + 1095.480001 t126_leaf4 <= 2514.48
 T126_R4_f0_ge: x_Lot_Frontage - 42 t126_leaf4 >= 0
 T126_R5_f12_ge: x_BsmtFin_SF_1 - 188 t126_leaf5 >= 0
 T126_R5_f15_ge: x_Total_Bsmt_SF - 1419 t126_leaf5 >= 0
 T126_L5_f106_le: x_Neighborhood_No_aplicames + 1e-06 t126_leaf5 <= 1
 T126_R6_f12_ge: x_BsmtFin_SF_1 - 188 t126_leaf6 >= 0
 T126_R6_f15_ge: x_Total_Bsmt_SF - 1419 t126_leaf6 >= 0
 T126_R6_f106_ge: x_Neighborhood_No_aplicames - t126_leaf6 >= 0
 TREE_127_ONEHOT: t127_leaf0 + t127_leaf1 + t127_leaf2 + t127_leaf3
   + t127_leaf4 + t127_leaf5 = 1
 T127_L0_f18_le: x_2nd_Flr_SF + 273.5400010000001 t127_leaf0 <= 1641.54
 T127_L0_f67_le: x_MS_Zoning_RM + 1e-06 t127_leaf0 <= 1
 T127_L0_f44_le: x_Yr_Sold + 997992.000001 t127_leaf0 <= 1e+06
 T127_L1_f18_le: x_2nd_Flr_SF + 273.5400010000001 t127_leaf1 <= 1641.54
 T127_L1_f67_le: x_MS_Zoning_RM + 1e-06 t127_leaf1 <= 1
 T127_R1_f44_ge: x_Yr_Sold - 2008 t127_leaf1 >= 0
 T127_L2_f18_le: x_2nd_Flr_SF + 273.5400010000001 t127_leaf2 <= 1641.54
 T127_R2_f67_ge: x_MS_Zoning_RM - t127_leaf2 >= 0
 T127_L2_f293_le: x_Sale_Condition_Abnorml + 1e-06 t127_leaf2 <= 1
 T127_L3_f18_le: x_2nd_Flr_SF + 273.5400010000001 t127_leaf3 <= 1641.54
 T127_R3_f67_ge: x_MS_Zoning_RM - t127_leaf3 >= 0
 T127_R3_f293_ge: x_Sale_Condition_Abnorml - t127_leaf3 >= 0
 T127_R4_f18_ge: x_2nd_Flr_SF - 1368 t127_leaf4 >= 0
 T127_L4_f37_le: x_Open_Porch_SF + 484.810001 t127_leaf4 <= 488.81
 T127_R5_f18_ge: x_2nd_Flr_SF - 1368 t127_leaf5 >= 0
 T127_R5_f37_ge: x_Open_Porch_SF - 4 t127_leaf5 >= 0
 TREE_128_ONEHOT: t128_leaf0 + t128_leaf1 + t128_leaf2 + t128_leaf3
   + t128_leaf4 + t128_leaf5 + t128_leaf6 + t128_leaf7 = 1
 T128_L0_f44_le: x_Yr_Sold + 997992.000001 t128_leaf0 <= 1e+06
 T128_L0_f36_le: x_Wood_Deck_SF + 317.900001 t128_leaf0 <= 525.9
 T128_L0_f16_le: x_Heating_QC + 1e-06 t128_leaf0 <= 4
 T128_L1_f44_le: x_Yr_Sold + 997992.000001 t128_leaf1 <= 1e+06
 T128_L1_f36_le: x_Wood_Deck_SF + 317.900001 t128_leaf1 <= 525.9
 T128_R1_f16_ge: x_Heating_QC - 2 t128_leaf1 >= 2
 T128_L2_f44_le: x_Yr_Sold + 997992.000001 t128_leaf2 <= 1e+06
 T128_R2_f36_ge: x_Wood_Deck_SF - 208 t128_leaf2 >= 0
 T128_L2_f119_le: x_Condition_1_Norm + 1e-06 t128_leaf2 <= 1
 T128_L3_f44_le: x_Yr_Sold + 997992.000001 t128_leaf3 <= 1e+06
 T128_R3_f36_ge: x_Wood_Deck_SF - 208 t128_leaf3 >= 0
 T128_R3_f119_ge: x_Condition_1_Norm - t128_leaf3 >= 0
 T128_R4_f44_ge: x_Yr_Sold - 2008 t128_leaf4 >= 0
 T128_L4_f293_le: x_Sale_Condition_Abnorml + 1e-06 t128_leaf4 <= 1
 T128_L4_f17_le: x_1st_Flr_SF + 1434.900001 t128_leaf4 <= 2534.9
 T128_R5_f44_ge: x_Yr_Sold - 2008 t128_leaf5 >= 0
 T128_L5_f293_le: x_Sale_Condition_Abnorml + 1e-06 t128_leaf5 <= 1
 T128_R5_f17_ge: x_1st_Flr_SF - 1100 t128_leaf5 >= 0
 T128_R6_f44_ge: x_Yr_Sold - 2008 t128_leaf6 >= 0
 T128_R6_f293_ge: x_Sale_Condition_Abnorml - t128_leaf6 >= 0
 T128_L6_f11_le: x_Bsmt_Cond + 2.000001 t128_leaf6 <= 4
 T128_R7_f44_ge: x_Yr_Sold - 2008 t128_leaf7 >= 0
 T128_R7_f293_ge: x_Sale_Condition_Abnorml - t128_leaf7 >= 0
 T128_R7_f11_ge: x_Bsmt_Cond - 3 t128_leaf7 >= -1
 TREE_129_ONEHOT: t129_leaf0 + t129_leaf1 + t129_leaf2 + t129_leaf3
   + t129_leaf4 + t129_leaf5 + t129_leaf6 = 1
 T129_L0_f12_le: x_BsmtFin_SF_1 + 1092.500001 t129_leaf0 <= 1984.5
 T129_L0_f1_le: x_Lot_Area + 1000.000001 t129_leaf0 <= 7000
 T129_L0_f20_le: x_Gr_Liv_Area + 2445.760001 t129_leaf0 <= 3421.76
 T129_L1_f12_le: x_BsmtFin_SF_1 + 1092.500001 t129_leaf1 <= 1984.5
 T129_L1_f1_le: x_Lot_Area + 1000.000001 t129_leaf1 <= 7000
 T129_R1_f20_ge: x_Gr_Liv_Area - 976 t129_leaf1 >= 0
 T129_L2_f12_le: x_BsmtFin_SF_1 + 1092.500001 t129_leaf2 <= 1984.5
 T129_R2_f1_ge: x_Lot_Area >= 6000
 T129_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t129_leaf2 <= 1
 T129_L3_f12_le: x_BsmtFin_SF_1 + 1092.500001 t129_leaf3 <= 1984.5
 T129_R3_f1_ge: x_Lot_Area >= 6000
 T129_R3_f95_ge: x_Neighborhood_Crawfor - t129_leaf3 >= 0
 T129_R4_f12_ge: x_BsmtFin_SF_1 - 892 t129_leaf4 >= 0
 T129_L4_f118_le: x_Condition_1_Feedr + 1e-06 t129_leaf4 <= 1
 T129_L4_f20_le: x_Gr_Liv_Area + 772.7600010000001 t129_leaf4 <= 3421.76
 T129_R5_f12_ge: x_BsmtFin_SF_1 - 892 t129_leaf5 >= 0
 T129_L5_f118_le: x_Condition_1_Feedr + 1e-06 t129_leaf5 <= 1
 T129_R5_f20_ge: x_Gr_Liv_Area - 2649 t129_leaf5 >= 0
 T129_R6_f12_ge: x_BsmtFin_SF_1 - 892 t129_leaf6 >= 0
 T129_R6_f118_ge: x_Condition_1_Feedr - t129_leaf6 >= 0
 TREE_130_ONEHOT: t130_leaf0 + t130_leaf1 + t130_leaf2 + t130_leaf3
   + t130_leaf4 + t130_leaf5 + t130_leaf6 = 1
 T130_L0_f20_le: x_Gr_Liv_Area + 1028.760001 t130_leaf0 <= 3421.76
 T130_L0_f1_le: x_Lot_Area + 3325.000001 t130_leaf0 <= 7000
 T130_L0_f12_le: x_BsmtFin_SF_1 + 1614.500001 t130_leaf0 <= 1984.5
 T130_L1_f20_le: x_Gr_Liv_Area + 1028.760001 t130_leaf1 <= 3421.76
 T130_L1_f1_le: x_Lot_Area + 3325.000001 t130_leaf1 <= 7000
 T130_R1_f12_ge: x_BsmtFin_SF_1 - 370 t130_leaf1 >= 0
 T130_L2_f20_le: x_Gr_Liv_Area + 1028.760001 t130_leaf2 <= 3421.76
 T130_R2_f1_ge: x_Lot_Area >= 3675
 T130_L2_f4_le: x_Overall_Cond + 999993.000001 t130_leaf2 <= 1e+06
 T130_L3_f20_le: x_Gr_Liv_Area + 1028.760001 t130_leaf3 <= 3421.76
 T130_R3_f1_ge: x_Lot_Area >= 3675
 T130_R3_f4_ge: x_Overall_Cond - 7 t130_leaf3 >= 0
 T130_R4_f20_ge: x_Gr_Liv_Area - 2393 t130_leaf4 >= 0
 T130_L4_f4_le: x_Overall_Cond + 999995.000001 t130_leaf4 <= 1e+06
 T130_R5_f20_ge: x_Gr_Liv_Area - 2393 t130_leaf5 >= 0
 T130_R5_f4_ge: x_Overall_Cond - 5 t130_leaf5 >= 0
 T130_L5_f17_le: x_1st_Flr_SF + 752.9000010000002 t130_leaf5 <= 2534.9
 T130_R6_f20_ge: x_Gr_Liv_Area - 2393 t130_leaf6 >= 0
 T130_R6_f4_ge: x_Overall_Cond - 5 t130_leaf6 >= 0
 T130_R6_f17_ge: x_1st_Flr_SF - 1782 t130_leaf6 >= 0
 TREE_131_ONEHOT: t131_leaf0 + t131_leaf1 + t131_leaf2 + t131_leaf3
   + t131_leaf4 = 1
 T131_L0_f18_le: x_2nd_Flr_SF + 270.5400010000001 t131_leaf0 <= 1641.54
 T131_L0_f46_le: x_MS_SubClass_30 + 1e-06 t131_leaf0 <= 1
 T131_L0_f298_le: x_Sale_Condition_Partial + 1e-06 t131_leaf0 <= 1
 T131_L1_f18_le: x_2nd_Flr_SF + 270.5400010000001 t131_leaf1 <= 1641.54
 T131_L1_f46_le: x_MS_SubClass_30 + 1e-06 t131_leaf1 <= 1
 T131_R1_f298_ge: x_Sale_Condition_Partial - t131_leaf1 >= 0
 T131_L2_f18_le: x_2nd_Flr_SF + 270.5400010000001 t131_leaf2 <= 1641.54
 T131_R2_f46_ge: x_MS_SubClass_30 - t131_leaf2 >= 0
 T131_L2_f6_le: x_Year_Remod/Add + 998044.000001 t131_leaf2 <= 1e+06
 T131_L3_f18_le: x_2nd_Flr_SF + 270.5400010000001 t131_leaf3 <= 1641.54
 T131_R3_f46_ge: x_MS_SubClass_30 - t131_leaf3 >= 0
 T131_R3_f6_ge: x_Year_Remod/Add - 1956 t131_leaf3 >= 0
 T131_R4_f18_ge: x_2nd_Flr_SF - 1371 t131_leaf4 >= 0
 TREE_132_ONEHOT: t132_leaf0 + t132_leaf1 + t132_leaf2 + t132_leaf3
   + t132_leaf4 + t132_leaf5 = 1
 T132_L0_f23_le: x_Full_Bath + 3.000001 t132_leaf0 <= 6
 T132_L0_f20_le: x_Gr_Liv_Area + 2397.760001 t132_leaf0 <= 3421.76
 T132_L0_f12_le: x_BsmtFin_SF_1 + 1928.500001 t132_leaf0 <= 1984.5
 T132_L1_f23_le: x_Full_Bath + 3.000001 t132_leaf1 <= 6
 T132_L1_f20_le: x_Gr_Liv_Area + 2397.760001 t132_leaf1 <= 3421.76
 T132_R1_f12_ge: x_BsmtFin_SF_1 - 56 t132_leaf1 >= 0
 T132_L2_f23_le: x_Full_Bath + 3.000001 t132_leaf2 <= 6
 T132_R2_f20_ge: x_Gr_Liv_Area - 1024 t132_leaf2 >= 0
 T132_L2_f12_le: x_BsmtFin_SF_1 + 1556.500001 t132_leaf2 <= 1984.5
 T132_L3_f23_le: x_Full_Bath + 3.000001 t132_leaf3 <= 6
 T132_R3_f20_ge: x_Gr_Liv_Area - 1024 t132_leaf3 >= 0
 T132_R3_f12_ge: x_BsmtFin_SF_1 - 428 t132_leaf3 >= 0
 T132_R4_f23_ge: x_Full_Bath - 3 t132_leaf4 >= 0
 T132_L4_f20_le: x_Gr_Liv_Area + 627.7600010000001 t132_leaf4 <= 3421.76
 T132_R5_f23_ge: x_Full_Bath - 3 t132_leaf5 >= 0
 T132_R5_f20_ge: x_Gr_Liv_Area - 2794 t132_leaf5 >= 0
 TREE_133_ONEHOT: t133_leaf0 + t133_leaf1 + t133_leaf2 + t133_leaf3
   + t133_leaf4 + t133_leaf5 + t133_leaf6 = 1
 T133_L0_f20_le: x_Gr_Liv_Area + 1028.760001 t133_leaf0 <= 3421.76
 T133_L0_f62_le: x_MS_Zoning_C_(all) + 1e-06 t133_leaf0 <= 1
 T133_L0_f203_le: x_Foundation_Slab + 1e-06 t133_leaf0 <= 1
 T133_L1_f20_le: x_Gr_Liv_Area + 1028.760001 t133_leaf1 <= 3421.76
 T133_L1_f62_le: x_MS_Zoning_C_(all) + 1e-06 t133_leaf1 <= 1
 T133_R1_f203_ge: x_Foundation_Slab - t133_leaf1 >= 0
 T133_L2_f20_le: x_Gr_Liv_Area + 1028.760001 t133_leaf2 <= 3421.76
 T133_R2_f62_ge: x_MS_Zoning_C_(all) - t133_leaf2 >= 0
 T133_L2_f44_le: x_Yr_Sold + 997992.000001 t133_leaf2 <= 1e+06
 T133_L3_f20_le: x_Gr_Liv_Area + 1028.760001 t133_leaf3 <= 3421.76
 T133_R3_f62_ge: x_MS_Zoning_C_(all) - t133_leaf3 >= 0
 T133_R3_f44_ge: x_Yr_Sold - 2008 t133_leaf3 >= 0
 T133_R4_f20_ge: x_Gr_Liv_Area - 2393 t133_leaf4 >= 0
 T133_L4_f118_le: x_Condition_1_Feedr + 1e-06 t133_leaf4 <= 1
 T133_L4_f20_le: x_Gr_Liv_Area + 335.7600010000001 t133_leaf4 <= 3421.76
 T133_R5_f20_ge: x_Gr_Liv_Area - 2393 t133_leaf5 >= 0
 T133_L5_f118_le: x_Condition_1_Feedr + 1e-06 t133_leaf5 <= 1
 T133_R5_f20_ge: x_Gr_Liv_Area - 3086 t133_leaf5 >= 0
 T133_R6_f20_ge: x_Gr_Liv_Area - 2393 t133_leaf6 >= 0
 T133_R6_f118_ge: x_Condition_1_Feedr - t133_leaf6 >= 0
 TREE_134_ONEHOT: t134_leaf0 + t134_leaf1 + t134_leaf2 + t134_leaf3
   + t134_leaf4 + t134_leaf5 + t134_leaf6 = 1
 T134_L0_f15_le: x_Total_Bsmt_SF + 690.4800010000001 t134_leaf0 <= 2514.48
 T134_L0_f6_le: x_Year_Remod/Add + 998007.000001 t134_leaf0 <= 1e+06
 T134_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t134_leaf0 <= 1
 T134_L1_f15_le: x_Total_Bsmt_SF + 690.4800010000001 t134_leaf1 <= 2514.48
 T134_L1_f6_le: x_Year_Remod/Add + 998007.000001 t134_leaf1 <= 1e+06
 T134_R1_f297_ge: x_Sale_Condition_Normal - t134_leaf1 >= 0
 T134_L2_f15_le: x_Total_Bsmt_SF + 690.4800010000001 t134_leaf2 <= 2514.48
 T134_R2_f6_ge: x_Year_Remod/Add - 1993 t134_leaf2 >= 0
 T134_L2_f20_le: x_Gr_Liv_Area + 1184.760001 t134_leaf2 <= 3421.76
 T134_L3_f15_le: x_Total_Bsmt_SF + 690.4800010000001 t134_leaf3 <= 2514.48
 T134_R3_f6_ge: x_Year_Remod/Add - 1993 t134_leaf3 >= 0
 T134_R3_f20_ge: x_Gr_Liv_Area - 2237 t134_leaf3 >= 0
 T134_R4_f15_ge: x_Total_Bsmt_SF - 1824 t134_leaf4 >= 0
 T134_L4_f20_le: x_Gr_Liv_Area + 1533.760001 t134_leaf4 <= 3421.76
 T134_R5_f15_ge: x_Total_Bsmt_SF - 1824 t134_leaf5 >= 0
 T134_R5_f20_ge: x_Gr_Liv_Area - 1888 t134_leaf5 >= 0
 T134_L5_f273_le: x_Mo_Sold_3 + 1e-06 t134_leaf5 <= 1
 T134_R6_f15_ge: x_Total_Bsmt_SF - 1824 t134_leaf6 >= 0
 T134_R6_f20_ge: x_Gr_Liv_Area - 1888 t134_leaf6 >= 0
 T134_R6_f273_ge: x_Mo_Sold_3 - t134_leaf6 >= 0
 TREE_135_ONEHOT: t135_leaf0 + t135_leaf1 + t135_leaf2 + t135_leaf3
   + t135_leaf4 + t135_leaf5 + t135_leaf6 = 1
 T135_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t135_leaf0 <= 1
 T135_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t135_leaf0 <= 1
 T135_L0_f37_le: x_Open_Porch_SF + 466.810001 t135_leaf0 <= 488.81
 T135_L1_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t135_leaf1 <= 1
 T135_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t135_leaf1 <= 1
 T135_R1_f37_ge: x_Open_Porch_SF - 22 t135_leaf1 >= 0
 T135_L2_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t135_leaf2 <= 1
 T135_R2_f95_ge: x_Neighborhood_Crawfor - t135_leaf2 >= 0
 T135_L2_f1_le: x_Lot_Area <= 7263.999999
 T135_L3_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t135_leaf3 <= 1
 T135_R3_f95_ge: x_Neighborhood_Crawfor - t135_leaf3 >= 0
 T135_R3_f1_ge: x_Lot_Area - 264 t135_leaf3 >= 7000
 T135_R4_f207_ge: x_Bsmt_Exposure_Gd - t135_leaf4 >= 0
 T135_L4_f1_le: x_Lot_Area <= 13687.999999
 T135_L4_f1_le: x_Lot_Area <= 7017.999999
 T135_R5_f207_ge: x_Bsmt_Exposure_Gd - t135_leaf5 >= 0
 T135_L5_f1_le: x_Lot_Area <= 13687.999999
 T135_R5_f1_ge: x_Lot_Area - 18 t135_leaf5 >= 7000
 T135_R6_f207_ge: x_Bsmt_Exposure_Gd - t135_leaf6 >= 0
 T135_R6_f1_ge: x_Lot_Area - 6688 t135_leaf6 >= 7000
 TREE_136_ONEHOT: t136_leaf0 + t136_leaf1 + t136_leaf2 + t136_leaf3
   + t136_leaf4 + t136_leaf5 + t136_leaf6 + t136_leaf7 = 1
 T136_L0_f44_le: x_Yr_Sold + 997992.000001 t136_leaf0 <= 1e+06
 T136_L0_f1_le: x_Lot_Area <= 11387.999999
 T136_L0_f21_le: x_Bsmt_Full_Bath + 1.020001 t136_leaf0 <= 2.02
 T136_L1_f44_le: x_Yr_Sold + 997992.000001 t136_leaf1 <= 1e+06
 T136_L1_f1_le: x_Lot_Area <= 11387.999999
 T136_R1_f21_ge: x_Bsmt_Full_Bath - t136_leaf1 >= 0
 T136_L2_f44_le: x_Yr_Sold + 997992.000001 t136_leaf2 <= 1e+06
 T136_R2_f1_ge: x_Lot_Area - 4388 t136_leaf2 >= 7000
 T136_L2_f37_le: x_Open_Porch_SF + 222.810001 t136_leaf2 <= 488.81
 T136_L3_f44_le: x_Yr_Sold + 997992.000001 t136_leaf3 <= 1e+06
 T136_R3_f1_ge: x_Lot_Area - 4388 t136_leaf3 >= 7000
 T136_R3_f37_ge: x_Open_Porch_SF - 266 t136_leaf3 >= 0
 T136_R4_f44_ge: x_Yr_Sold - 2008 t136_leaf4 >= 0
 T136_L4_f6_le: x_Year_Remod/Add + 998007.000001 t136_leaf4 <= 1e+06
 T136_L4_f4_le: x_Overall_Cond + 999994.000001 t136_leaf4 <= 1e+06
 T136_R5_f44_ge: x_Yr_Sold - 2008 t136_leaf5 >= 0
 T136_L5_f6_le: x_Year_Remod/Add + 998007.000001 t136_leaf5 <= 1e+06
 T136_R5_f4_ge: x_Overall_Cond - 6 t136_leaf5 >= 0
 T136_R6_f44_ge: x_Yr_Sold - 2008 t136_leaf6 >= 0
 T136_R6_f6_ge: x_Year_Remod/Add - 1993 t136_leaf6 >= 0
 T136_L6_f86_le: x_Land_Slope_Gtl + 1e-06 t136_leaf6 <= 1
 T136_R7_f44_ge: x_Yr_Sold - 2008 t136_leaf7 >= 0
 T136_R7_f6_ge: x_Year_Remod/Add - 1993 t136_leaf7 >= 0
 T136_R7_f86_ge: x_Land_Slope_Gtl - t136_leaf7 >= 0
 TREE_137_ONEHOT: t137_leaf0 + t137_leaf1 + t137_leaf2 + t137_leaf3
   + t137_leaf4 + t137_leaf5 + t137_leaf6 + t137_leaf7 = 1
 T137_L0_f12_le: x_BsmtFin_SF_1 + 887.5000010000001 t137_leaf0 <= 1984.5
 T137_L0_f67_le: x_MS_Zoning_RM + 1e-06 t137_leaf0 <= 1
 T137_L0_f4_le: x_Overall_Cond + 999993.000001 t137_leaf0 <= 1e+06
 T137_L1_f12_le: x_BsmtFin_SF_1 + 887.5000010000001 t137_leaf1 <= 1984.5
 T137_L1_f67_le: x_MS_Zoning_RM + 1e-06 t137_leaf1 <= 1
 T137_R1_f4_ge: x_Overall_Cond - 7 t137_leaf1 >= 0
 T137_L2_f12_le: x_BsmtFin_SF_1 + 887.5000010000001 t137_leaf2 <= 1984.5
 T137_R2_f67_ge: x_MS_Zoning_RM - t137_leaf2 >= 0
 T137_L2_f4_le: x_Overall_Cond + 999995.000001 t137_leaf2 <= 1e+06
 T137_L3_f12_le: x_BsmtFin_SF_1 + 887.5000010000001 t137_leaf3 <= 1984.5
 T137_R3_f67_ge: x_MS_Zoning_RM - t137_leaf3 >= 0
 T137_R3_f4_ge: x_Overall_Cond - 5 t137_leaf3 >= 0
 T137_R4_f12_ge: x_BsmtFin_SF_1 - 1097 t137_leaf4 >= 0
 T137_L4_f10_le: x_Bsmt_Qual + 1.000001 t137_leaf4 <= 4
 T137_L4_f37_le: x_Open_Porch_SF + 420.810001 t137_leaf4 <= 488.81
 T137_R5_f12_ge: x_BsmtFin_SF_1 - 1097 t137_leaf5 >= 0
 T137_L5_f10_le: x_Bsmt_Qual + 1.000001 t137_leaf5 <= 4
 T137_R5_f37_ge: x_Open_Porch_SF - 68 t137_leaf5 >= 0
 T137_R6_f12_ge: x_BsmtFin_SF_1 - 1097 t137_leaf6 >= 0
 T137_R6_f10_ge: x_Bsmt_Qual - t137_leaf6 >= 2
 T137_L6_f174_le: x_Exterior_1st_VinylSd + 1e-06 t137_leaf6 <= 1
 T137_R7_f12_ge: x_BsmtFin_SF_1 - 1097 t137_leaf7 >= 0
 T137_R7_f10_ge: x_Bsmt_Qual - t137_leaf7 >= 2
 T137_R7_f174_ge: x_Exterior_1st_VinylSd - t137_leaf7 >= 0
 TREE_138_ONEHOT: t138_leaf0 + t138_leaf1 + t138_leaf2 + t138_leaf3
   + t138_leaf4 + t138_leaf5 + t138_leaf6 + t138_leaf7 = 1
 T138_L0_f20_le: x_Gr_Liv_Area + 2139.760001000001 t138_leaf0 <= 3421.76
 T138_L0_f12_le: x_BsmtFin_SF_1 + 1964.500001 t138_leaf0 <= 1984.5
 T138_L0_f9_le: x_Exter_Cond + 2.000001 t138_leaf0 <= 4
 T138_L1_f20_le: x_Gr_Liv_Area + 2139.760001000001 t138_leaf1 <= 3421.76
 T138_L1_f12_le: x_BsmtFin_SF_1 + 1964.500001 t138_leaf1 <= 1984.5
 T138_R1_f9_ge: x_Exter_Cond - 3 t138_leaf1 >= -1
 T138_L2_f20_le: x_Gr_Liv_Area + 2139.760001000001 t138_leaf2 <= 3421.76
 T138_R2_f12_ge: x_BsmtFin_SF_1 - 20 t138_leaf2 >= 0
 T138_L2_f17_le: x_1st_Flr_SF + 1640.900001 t138_leaf2 <= 2534.9
 T138_L3_f20_le: x_Gr_Liv_Area + 2139.760001000001 t138_leaf3 <= 3421.76
 T138_R3_f12_ge: x_BsmtFin_SF_1 - 20 t138_leaf3 >= 0
 T138_R3_f17_ge: x_1st_Flr_SF - 894 t138_leaf3 >= 0
 T138_R4_f20_ge: x_Gr_Liv_Area - 1282 t138_leaf4 >= 0
 T138_L4_f4_le: x_Overall_Cond + 999995.000001 t138_leaf4 <= 1e+06
 T138_L4_f14_le: x_Bsmt_Unf_SF + 1398.680001 t138_leaf4 <= 1812.68
 T138_R5_f20_ge: x_Gr_Liv_Area - 1282 t138_leaf5 >= 0
 T138_L5_f4_le: x_Overall_Cond + 999995.000001 t138_leaf5 <= 1e+06
 T138_R5_f14_ge: x_Bsmt_Unf_SF - 414 t138_leaf5 >= 0
 T138_R6_f20_ge: x_Gr_Liv_Area - 1282 t138_leaf6 >= 0
 T138_R6_f4_ge: x_Overall_Cond - 5 t138_leaf6 >= 0
 T138_L6_f20_le: x_Gr_Liv_Area + 1299.760001 t138_leaf6 <= 3421.76
 T138_R7_f20_ge: x_Gr_Liv_Area - 1282 t138_leaf7 >= 0
 T138_R7_f4_ge: x_Overall_Cond - 5 t138_leaf7 >= 0
 T138_R7_f20_ge: x_Gr_Liv_Area - 2122 t138_leaf7 >= 0
 TREE_139_ONEHOT: t139_leaf0 + t139_leaf1 + t139_leaf2 + t139_leaf3
   + t139_leaf4 + t139_leaf5 + t139_leaf6 + t139_leaf7 = 1
 T139_L0_f12_le: x_BsmtFin_SF_1 + 1765.500001 t139_leaf0 <= 1984.5
 T139_L0_f16_le: x_Heating_QC + 1.000001 t139_leaf0 <= 4
 T139_L0_f44_le: x_Yr_Sold + 997990.000001 t139_leaf0 <= 1e+06
 T139_L1_f12_le: x_BsmtFin_SF_1 + 1765.500001 t139_leaf1 <= 1984.5
 T139_L1_f16_le: x_Heating_QC + 1.000001 t139_leaf1 <= 4
 T139_R1_f44_ge: x_Yr_Sold - 2010 t139_leaf1 >= 0
 T139_L2_f12_le: x_BsmtFin_SF_1 + 1765.500001 t139_leaf2 <= 1984.5
 T139_R2_f16_ge: x_Heating_QC - t139_leaf2 >= 2
 T139_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t139_leaf2 <= 1
 T139_L3_f12_le: x_BsmtFin_SF_1 + 1765.500001 t139_leaf3 <= 1984.5
 T139_R3_f16_ge: x_Heating_QC - t139_leaf3 >= 2
 T139_R3_f245_ge: x_FunctioNo_aplical_Typ - t139_leaf3 >= 0
 T139_R4_f12_ge: x_BsmtFin_SF_1 - 219 t139_leaf4 >= 0
 T139_L4_f20_le: x_Gr_Liv_Area + 2119.760001000001 t139_leaf4 <= 3421.76
 T139_L4_f17_le: x_1st_Flr_SF + 1605.900001 t139_leaf4 <= 2534.9
 T139_R5_f12_ge: x_BsmtFin_SF_1 - 219 t139_leaf5 >= 0
 T139_L5_f20_le: x_Gr_Liv_Area + 2119.760001000001 t139_leaf5 <= 3421.76
 T139_R5_f17_ge: x_1st_Flr_SF - 929 t139_leaf5 >= 0
 T139_R6_f12_ge: x_BsmtFin_SF_1 - 219 t139_leaf6 >= 0
 T139_R6_f20_ge: x_Gr_Liv_Area - 1302 t139_leaf6 >= 0
 T139_L6_f5_le: x_Year_Built + 998089.000001 t139_leaf6 <= 1e+06
 T139_R7_f12_ge: x_BsmtFin_SF_1 - 219 t139_leaf7 >= 0
 T139_R7_f20_ge: x_Gr_Liv_Area - 1302 t139_leaf7 >= 0
 T139_R7_f5_ge: x_Year_Built - 1911 t139_leaf7 >= 0
 TREE_140_ONEHOT: t140_leaf0 + t140_leaf1 + t140_leaf2 + t140_leaf3
   + t140_leaf4 + t140_leaf5 + t140_leaf6 = 1
 T140_L0_f11_le: x_Bsmt_Cond + 2.000001 t140_leaf0 <= 4
 T140_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t140_leaf0 <= 1
 T140_L0_f28_le: x_TotRms_AbvGrd + 6.080001 t140_leaf0 <= 12.08
 T140_L1_f11_le: x_Bsmt_Cond + 2.000001 t140_leaf1 <= 4
 T140_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t140_leaf1 <= 1
 T140_R1_f28_ge: x_TotRms_AbvGrd - 6 t140_leaf1 >= 0
 T140_L2_f11_le: x_Bsmt_Cond + 2.000001 t140_leaf2 <= 4
 T140_R2_f297_ge: x_Sale_Condition_Normal - t140_leaf2 >= 0
 T140_L2_f3_le: x_Overall_Qual + 999996.000001 t140_leaf2 <= 1e+06
 T140_L3_f11_le: x_Bsmt_Cond + 2.000001 t140_leaf3 <= 4
 T140_R3_f297_ge: x_Sale_Condition_Normal - t140_leaf3 >= 0
 T140_R3_f3_ge: x_Overall_Qual - 4 t140_leaf3 >= 0
 T140_R4_f11_ge: x_Bsmt_Cond - 3 t140_leaf4 >= -1
 T140_L4_f3_le: x_Overall_Qual + 999991.000001 t140_leaf4 <= 1e+06
 T140_L4_f4_le: x_Overall_Cond + 999993.000001 t140_leaf4 <= 1e+06
 T140_R5_f11_ge: x_Bsmt_Cond - 3 t140_leaf5 >= -1
 T140_L5_f3_le: x_Overall_Qual + 999991.000001 t140_leaf5 <= 1e+06
 T140_R5_f4_ge: x_Overall_Cond - 7 t140_leaf5 >= 0
 T140_R6_f11_ge: x_Bsmt_Cond - 3 t140_leaf6 >= -1
 T140_R6_f3_ge: x_Overall_Qual - 9 t140_leaf6 >= 0
 TREE_141_ONEHOT: t141_leaf0 + t141_leaf1 + t141_leaf2 + t141_leaf3
   + t141_leaf4 + t141_leaf5 + t141_leaf6 = 1
 T141_L0_f23_le: x_Full_Bath + 3.000001 t141_leaf0 <= 6
 T141_L0_f31_le: x_Garage_Yr_Blt + 998071.000001 t141_leaf0 <= 1e+06
 T141_L0_f20_le: x_Gr_Liv_Area + 2397.760001 t141_leaf0 <= 3421.76
 T141_L1_f23_le: x_Full_Bath + 3.000001 t141_leaf1 <= 6
 T141_L1_f31_le: x_Garage_Yr_Blt + 998071.000001 t141_leaf1 <= 1e+06
 T141_R1_f20_ge: x_Gr_Liv_Area - 1024 t141_leaf1 >= 0
 T141_L2_f23_le: x_Full_Bath + 3.000001 t141_leaf2 <= 6
 T141_R2_f31_ge: x_Garage_Yr_Blt - 1929 t141_leaf2 >= 0
 T141_L2_f20_le: x_Gr_Liv_Area + 549.7600010000001 t141_leaf2 <= 3421.76
 T141_L3_f23_le: x_Full_Bath + 3.000001 t141_leaf3 <= 6
 T141_R3_f31_ge: x_Garage_Yr_Blt - 1929 t141_leaf3 >= 0
 T141_R3_f20_ge: x_Gr_Liv_Area - 2872 t141_leaf3 >= 0
 T141_R4_f23_ge: x_Full_Bath - 3 t141_leaf4 >= 0
 T141_L4_f20_le: x_Gr_Liv_Area + 549.7600010000001 t141_leaf4 <= 3421.76
 T141_L4_f26_le: x_Kitchen_AbvGr + 1.000001 t141_leaf4 <= 3
 T141_R5_f23_ge: x_Full_Bath - 3 t141_leaf5 >= 0
 T141_L5_f20_le: x_Gr_Liv_Area + 549.7600010000001 t141_leaf5 <= 3421.76
 T141_R5_f26_ge: x_Kitchen_AbvGr - 2 t141_leaf5 >= 0
 T141_R6_f23_ge: x_Full_Bath - 3 t141_leaf6 >= 0
 T141_R6_f20_ge: x_Gr_Liv_Area - 2872 t141_leaf6 >= 0
 TREE_142_ONEHOT: t142_leaf0 + t142_leaf1 + t142_leaf2 + t142_leaf3
   + t142_leaf4 + t142_leaf5 + t142_leaf6 = 1
 T142_L0_f44_le: x_Yr_Sold + 997992.000001 t142_leaf0 <= 1e+06
 T142_L0_f1_le: x_Lot_Area + 1000.000001 t142_leaf0 <= 7000
 T142_L1_f44_le: x_Yr_Sold + 997992.000001 t142_leaf1 <= 1e+06
 T142_R1_f1_ge: x_Lot_Area >= 6000
 T142_L1_f283_le: x_Sale_Type_COD + 1e-06 t142_leaf1 <= 1
 T142_L2_f44_le: x_Yr_Sold + 997992.000001 t142_leaf2 <= 1e+06
 T142_R2_f1_ge: x_Lot_Area >= 6000
 T142_R2_f283_ge: x_Sale_Type_COD - t142_leaf2 >= 0
 T142_R3_f44_ge: x_Yr_Sold - 2008 t142_leaf3 >= 0
 T142_L3_f118_le: x_Condition_1_Feedr + 1e-06 t142_leaf3 <= 1
 T142_L3_f6_le: x_Year_Remod/Add + 997992.000001 t142_leaf3 <= 1e+06
 T142_R4_f44_ge: x_Yr_Sold - 2008 t142_leaf4 >= 0
 T142_L4_f118_le: x_Condition_1_Feedr + 1e-06 t142_leaf4 <= 1
 T142_R4_f6_ge: x_Year_Remod/Add - 2008 t142_leaf4 >= 0
 T142_R5_f44_ge: x_Yr_Sold - 2008 t142_leaf5 >= 0
 T142_R5_f118_ge: x_Condition_1_Feedr - t142_leaf5 >= 0
 T142_L5_f17_le: x_1st_Flr_SF + 1578.900001 t142_leaf5 <= 2534.9
 T142_R6_f44_ge: x_Yr_Sold - 2008 t142_leaf6 >= 0
 T142_R6_f118_ge: x_Condition_1_Feedr - t142_leaf6 >= 0
 T142_R6_f17_ge: x_1st_Flr_SF - 956 t142_leaf6 >= 0
 TREE_143_ONEHOT: t143_leaf0 + t143_leaf1 + t143_leaf2 + t143_leaf3
   + t143_leaf4 + t143_leaf5 + t143_leaf6 + t143_leaf7 = 1
 T143_L0_f12_le: x_BsmtFin_SF_1 + 1692.500001 t143_leaf0 <= 1984.5
 T143_L0_f1_le: x_Lot_Area + 1024.000001 t143_leaf0 <= 7000
 T143_L0_f71_le: x_Alley_No_aplica + 1e-06 t143_leaf0 <= 1
 T143_L1_f12_le: x_BsmtFin_SF_1 + 1692.500001 t143_leaf1 <= 1984.5
 T143_L1_f1_le: x_Lot_Area + 1024.000001 t143_leaf1 <= 7000
 T143_R1_f71_ge: x_Alley_No_aplica - t143_leaf1 >= 0
 T143_L2_f12_le: x_BsmtFin_SF_1 + 1692.500001 t143_leaf2 <= 1984.5
 T143_R2_f1_ge: x_Lot_Area >= 5976
 T143_L2_f272_le: x_Mo_Sold_2 + 1e-06 t143_leaf2 <= 1
 T143_L3_f12_le: x_BsmtFin_SF_1 + 1692.500001 t143_leaf3 <= 1984.5
 T143_R3_f1_ge: x_Lot_Area >= 5976
 T143_R3_f272_ge: x_Mo_Sold_2 - t143_leaf3 >= 0
 T143_R4_f12_ge: x_BsmtFin_SF_1 - 292 t143_leaf4 >= 0
 T143_L4_f17_le: x_1st_Flr_SF + 882.9000010000002 t143_leaf4 <= 2534.9
 T143_L4_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t143_leaf4 <= 1
 T143_R5_f12_ge: x_BsmtFin_SF_1 - 292 t143_leaf5 >= 0
 T143_L5_f17_le: x_1st_Flr_SF + 882.9000010000002 t143_leaf5 <= 2534.9
 T143_R5_f216_ge: x_BsmtFin_Type_1_Rec - t143_leaf5 >= 0
 T143_R6_f12_ge: x_BsmtFin_SF_1 - 292 t143_leaf6 >= 0
 T143_R6_f17_ge: x_1st_Flr_SF - 1652 t143_leaf6 >= 0
 T143_L6_f81_le: x_Lot_Config_Corner + 1e-06 t143_leaf6 <= 1
 T143_R7_f12_ge: x_BsmtFin_SF_1 - 292 t143_leaf7 >= 0
 T143_R7_f17_ge: x_1st_Flr_SF - 1652 t143_leaf7 >= 0
 T143_R7_f81_ge: x_Lot_Config_Corner - t143_leaf7 >= 0
 TREE_144_ONEHOT: t144_leaf0 + t144_leaf1 + t144_leaf2 + t144_leaf3
   + t144_leaf4 + t144_leaf5 + t144_leaf6 = 1
 T144_L0_f44_le: x_Yr_Sold + 997992.000001 t144_leaf0 <= 1e+06
 T144_L0_f18_le: x_2nd_Flr_SF + 273.5400010000001 t144_leaf0 <= 1641.54
 T144_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t144_leaf0 <= 1
 T144_L1_f44_le: x_Yr_Sold + 997992.000001 t144_leaf1 <= 1e+06
 T144_L1_f18_le: x_2nd_Flr_SF + 273.5400010000001 t144_leaf1 <= 1641.54
 T144_R1_f95_ge: x_Neighborhood_Crawfor - t144_leaf1 >= 0
 T144_L2_f44_le: x_Yr_Sold + 997992.000001 t144_leaf2 <= 1e+06
 T144_R2_f18_ge: x_2nd_Flr_SF - 1368 t144_leaf2 >= 0
 T144_R3_f44_ge: x_Yr_Sold - 2008 t144_leaf3 >= 0
 T144_L3_f293_le: x_Sale_Condition_Abnorml + 1e-06 t144_leaf3 <= 1
 T144_L3_f21_le: x_Bsmt_Full_Bath + 1.020001 t144_leaf3 <= 2.02
 T144_R4_f44_ge: x_Yr_Sold - 2008 t144_leaf4 >= 0
 T144_L4_f293_le: x_Sale_Condition_Abnorml + 1e-06 t144_leaf4 <= 1
 T144_R4_f21_ge: x_Bsmt_Full_Bath - t144_leaf4 >= 0
 T144_R5_f44_ge: x_Yr_Sold - 2008 t144_leaf5 >= 0
 T144_R5_f293_ge: x_Sale_Condition_Abnorml - t144_leaf5 >= 0
 T144_L5_f202_le: x_Foundation_PConc + 1e-06 t144_leaf5 <= 1
 T144_R6_f44_ge: x_Yr_Sold - 2008 t144_leaf6 >= 0
 T144_R6_f293_ge: x_Sale_Condition_Abnorml - t144_leaf6 >= 0
 T144_R6_f202_ge: x_Foundation_PConc - t144_leaf6 >= 0
 TREE_145_ONEHOT: t145_leaf0 + t145_leaf1 + t145_leaf2 + t145_leaf3
   + t145_leaf4 + t145_leaf5 = 1
 T145_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t145_leaf0 <= 1
 T145_L0_f18_le: x_2nd_Flr_SF + 12.54000100000007 t145_leaf0 <= 1641.54
 T145_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t145_leaf0 <= 1
 T145_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t145_leaf1 <= 1
 T145_L1_f18_le: x_2nd_Flr_SF + 12.54000100000007 t145_leaf1 <= 1641.54
 T145_R1_f95_ge: x_Neighborhood_Crawfor - t145_leaf1 >= 0
 T145_L2_f161_le: x_Exterior_1st_AsbShng + 1e-06 t145_leaf2 <= 1
 T145_R2_f18_ge: x_2nd_Flr_SF - 1629 t145_leaf2 >= 0
 T145_R3_f161_ge: x_Exterior_1st_AsbShng - t145_leaf3 >= 0
 T145_L3_f33_le: x_Garage_Area + 760.750001 t145_leaf3 <= 1163.75
 T145_L3_f217_le: x_BsmtFin_Type_1_Unf + 1e-06 t145_leaf3 <= 1
 T145_R4_f161_ge: x_Exterior_1st_AsbShng - t145_leaf4 >= 0
 T145_L4_f33_le: x_Garage_Area + 760.750001 t145_leaf4 <= 1163.75
 T145_R4_f217_ge: x_BsmtFin_Type_1_Unf - t145_leaf4 >= 0
 T145_R5_f161_ge: x_Exterior_1st_AsbShng - t145_leaf5 >= 0
 T145_R5_f33_ge: x_Garage_Area - 403 t145_leaf5 >= 0
 TREE_146_ONEHOT: t146_leaf0 + t146_leaf1 + t146_leaf2 + t146_leaf3
   + t146_leaf4 + t146_leaf5 + t146_leaf6 + t146_leaf7 = 1
 T146_L0_f12_le: x_BsmtFin_SF_1 + 1796.500001 t146_leaf0 <= 1984.5
 T146_L0_f1_le: x_Lot_Area + 1024.000001 t146_leaf0 <= 7000
 T146_L0_f63_le: x_MS_Zoning_FV + 1e-06 t146_leaf0 <= 1
 T146_L1_f12_le: x_BsmtFin_SF_1 + 1796.500001 t146_leaf1 <= 1984.5
 T146_L1_f1_le: x_Lot_Area + 1024.000001 t146_leaf1 <= 7000
 T146_R1_f63_ge: x_MS_Zoning_FV - t146_leaf1 >= 0
 T146_L2_f12_le: x_BsmtFin_SF_1 + 1796.500001 t146_leaf2 <= 1984.5
 T146_R2_f1_ge: x_Lot_Area >= 5976
 T146_L2_f4_le: x_Overall_Cond + 999993.000001 t146_leaf2 <= 1e+06
 T146_L3_f12_le: x_BsmtFin_SF_1 + 1796.500001 t146_leaf3 <= 1984.5
 T146_R3_f1_ge: x_Lot_Area >= 5976
 T146_R3_f4_ge: x_Overall_Cond - 7 t146_leaf3 >= 0
 T146_R4_f12_ge: x_BsmtFin_SF_1 - 188 t146_leaf4 >= 0
 T146_L4_f17_le: x_1st_Flr_SF + 813.9000010000002 t146_leaf4 <= 2534.9
 T146_L4_f4_le: x_Overall_Cond + 999994.000001 t146_leaf4 <= 1e+06
 T146_R5_f12_ge: x_BsmtFin_SF_1 - 188 t146_leaf5 >= 0
 T146_L5_f17_le: x_1st_Flr_SF + 813.9000010000002 t146_leaf5 <= 2534.9
 T146_R5_f4_ge: x_Overall_Cond - 6 t146_leaf5 >= 0
 T146_R6_f12_ge: x_BsmtFin_SF_1 - 188 t146_leaf6 >= 0
 T146_R6_f17_ge: x_1st_Flr_SF - 1721 t146_leaf6 >= 0
 T146_L6_f20_le: x_Gr_Liv_Area + 902.7600010000001 t146_leaf6 <= 3421.76
 T146_R7_f12_ge: x_BsmtFin_SF_1 - 188 t146_leaf7 >= 0
 T146_R7_f17_ge: x_1st_Flr_SF - 1721 t146_leaf7 >= 0
 T146_R7_f20_ge: x_Gr_Liv_Area - 2519 t146_leaf7 >= 0
 TREE_147_ONEHOT: t147_leaf0 + t147_leaf1 + t147_leaf2 + t147_leaf3
   + t147_leaf4 + t147_leaf5 + t147_leaf6 + t147_leaf7 = 1
 T147_L0_f6_le: x_Year_Remod/Add + 998006.000001 t147_leaf0 <= 1e+06
 T147_L0_f4_le: x_Overall_Cond + 999994.000001 t147_leaf0 <= 1e+06
 T147_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t147_leaf0 <= 1
 T147_L1_f6_le: x_Year_Remod/Add + 998006.000001 t147_leaf1 <= 1e+06
 T147_L1_f4_le: x_Overall_Cond + 999994.000001 t147_leaf1 <= 1e+06
 T147_R1_f207_ge: x_Bsmt_Exposure_Gd - t147_leaf1 >= 0
 T147_L2_f6_le: x_Year_Remod/Add + 998006.000001 t147_leaf2 <= 1e+06
 T147_R2_f4_ge: x_Overall_Cond - 6 t147_leaf2 >= 0
 T147_L2_f20_le: x_Gr_Liv_Area + 2373.760001000001 t147_leaf2 <= 3421.76
 T147_L3_f6_le: x_Year_Remod/Add + 998006.000001 t147_leaf3 <= 1e+06
 T147_R3_f4_ge: x_Overall_Cond - 6 t147_leaf3 >= 0
 T147_R3_f20_ge: x_Gr_Liv_Area - 1048 t147_leaf3 >= 0
 T147_R4_f6_ge: x_Year_Remod/Add - 1994 t147_leaf4 >= 0
 T147_L4_f20_le: x_Gr_Liv_Area + 1499.760001 t147_leaf4 <= 3421.76
 T147_L4_f12_le: x_BsmtFin_SF_1 + 1814.500001 t147_leaf4 <= 1984.5
 T147_R5_f6_ge: x_Year_Remod/Add - 1994 t147_leaf5 >= 0
 T147_L5_f20_le: x_Gr_Liv_Area + 1499.760001 t147_leaf5 <= 3421.76
 T147_R5_f12_ge: x_BsmtFin_SF_1 - 170 t147_leaf5 >= 0
 T147_R6_f6_ge: x_Year_Remod/Add - 1994 t147_leaf6 >= 0
 T147_R6_f20_ge: x_Gr_Liv_Area - 1922 t147_leaf6 >= 0
 T147_L6_f96_le: x_Neighborhood_Edwards + 1e-06 t147_leaf6 <= 1
 T147_R7_f6_ge: x_Year_Remod/Add - 1994 t147_leaf7 >= 0
 T147_R7_f20_ge: x_Gr_Liv_Area - 1922 t147_leaf7 >= 0
 T147_R7_f96_ge: x_Neighborhood_Edwards - t147_leaf7 >= 0
 TREE_148_ONEHOT: t148_leaf0 + t148_leaf1 + t148_leaf2 + t148_leaf3
   + t148_leaf4 + t148_leaf5 + t148_leaf6 + t148_leaf7 = 1
 T148_L0_f44_le: x_Yr_Sold + 997992.000001 t148_leaf0 <= 1e+06
 T148_L0_f20_le: x_Gr_Liv_Area + 1453.760001 t148_leaf0 <= 3421.76
 T148_L0_f137_le: x_Bldg_Type_Twnhs + 1e-06 t148_leaf0 <= 1
 T148_L1_f44_le: x_Yr_Sold + 997992.000001 t148_leaf1 <= 1e+06
 T148_L1_f20_le: x_Gr_Liv_Area + 1453.760001 t148_leaf1 <= 3421.76
 T148_R1_f137_ge: x_Bldg_Type_Twnhs - t148_leaf1 >= 0
 T148_L2_f44_le: x_Yr_Sold + 997992.000001 t148_leaf2 <= 1e+06
 T148_R2_f20_ge: x_Gr_Liv_Area - 1968 t148_leaf2 >= 0
 T148_L2_f37_le: x_Open_Porch_SF + 226.810001 t148_leaf2 <= 488.81
 T148_L3_f44_le: x_Yr_Sold + 997992.000001 t148_leaf3 <= 1e+06
 T148_R3_f20_ge: x_Gr_Liv_Area - 1968 t148_leaf3 >= 0
 T148_R3_f37_ge: x_Open_Porch_SF - 262 t148_leaf3 >= 0
 T148_R4_f44_ge: x_Yr_Sold - 2008 t148_leaf4 >= 0
 T148_L4_f232_le: x_Central_Air_Y + 1e-06 t148_leaf4 <= 1
 T148_L4_f36_le: x_Wood_Deck_SF + 469.900001 t148_leaf4 <= 525.9
 T148_R5_f44_ge: x_Yr_Sold - 2008 t148_leaf5 >= 0
 T148_L5_f232_le: x_Central_Air_Y + 1e-06 t148_leaf5 <= 1
 T148_R5_f36_ge: x_Wood_Deck_SF - 56 t148_leaf5 >= 0
 T148_R6_f44_ge: x_Yr_Sold - 2008 t148_leaf6 >= 0
 T148_R6_f232_ge: x_Central_Air_Y - t148_leaf6 >= 0
 T148_L6_f6_le: x_Year_Remod/Add + 997993.000001 t148_leaf6 <= 1e+06
 T148_R7_f44_ge: x_Yr_Sold - 2008 t148_leaf7 >= 0
 T148_R7_f232_ge: x_Central_Air_Y - t148_leaf7 >= 0
 T148_R7_f6_ge: x_Year_Remod/Add - 2007 t148_leaf7 >= 0
 TREE_149_ONEHOT: t149_leaf0 + t149_leaf1 + t149_leaf2 + t149_leaf3
   + t149_leaf4 + t149_leaf5 + t149_leaf6 + t149_leaf7 = 1
 T149_L0_f1_le: x_Lot_Area <= 12852.999999
 T149_L0_f1_le: x_Lot_Area + 1000.000001 t149_leaf0 <= 7000
 T149_L0_f33_le: x_Garage_Area + 963.750001 t149_leaf0 <= 1163.75
 T149_L1_f1_le: x_Lot_Area <= 12852.999999
 T149_L1_f1_le: x_Lot_Area + 1000.000001 t149_leaf1 <= 7000
 T149_R1_f33_ge: x_Garage_Area - 200 t149_leaf1 >= 0
 T149_L2_f1_le: x_Lot_Area <= 12852.999999
 T149_R2_f1_ge: x_Lot_Area >= 6000
 T149_L2_f36_le: x_Wood_Deck_SF + 377.900001 t149_leaf2 <= 525.9
 T149_L3_f1_le: x_Lot_Area <= 12852.999999
 T149_R3_f1_ge: x_Lot_Area >= 6000
 T149_R3_f36_ge: x_Wood_Deck_SF - 148 t149_leaf3 >= 0
 T149_R4_f1_ge: x_Lot_Area - 5853 t149_leaf4 >= 7000
 T149_L4_f209_le: x_Bsmt_Exposure_No + 1e-06 t149_leaf4 <= 1
 T149_L4_f12_le: x_BsmtFin_SF_1 + 574.5000010000001 t149_leaf4 <= 1984.5
 T149_R5_f1_ge: x_Lot_Area - 5853 t149_leaf5 >= 7000
 T149_L5_f209_le: x_Bsmt_Exposure_No + 1e-06 t149_leaf5 <= 1
 T149_R5_f12_ge: x_BsmtFin_SF_1 - 1410 t149_leaf5 >= 0
 T149_R6_f1_ge: x_Lot_Area - 5853 t149_leaf6 >= 7000
 T149_R6_f209_ge: x_Bsmt_Exposure_No - t149_leaf6 >= 0
 T149_L6_f297_le: x_Sale_Condition_Normal + 1e-06 t149_leaf6 <= 1
 T149_R7_f1_ge: x_Lot_Area - 5853 t149_leaf7 >= 7000
 T149_R7_f209_ge: x_Bsmt_Exposure_No - t149_leaf7 >= 0
 T149_R7_f297_ge: x_Sale_Condition_Normal - t149_leaf7 >= 0
 TREE_150_ONEHOT: t150_leaf0 + t150_leaf1 + t150_leaf2 + t150_leaf3
   + t150_leaf4 + t150_leaf5 = 1
 T150_L0_f18_le: x_2nd_Flr_SF + 236.5400010000001 t150_leaf0 <= 1641.54
 T150_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t150_leaf0 <= 1
 T150_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t150_leaf0 <= 1
 T150_L1_f18_le: x_2nd_Flr_SF + 236.5400010000001 t150_leaf1 <= 1641.54
 T150_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t150_leaf1 <= 1
 T150_R1_f207_ge: x_Bsmt_Exposure_Gd - t150_leaf1 >= 0
 T150_L2_f18_le: x_2nd_Flr_SF + 236.5400010000001 t150_leaf2 <= 1641.54
 T150_R2_f95_ge: x_Neighborhood_Crawfor - t150_leaf2 >= 0
 T150_L2_f22_le: x_Bsmt_Half_Bath + 0.010001 t150_leaf2 <= 1.01
 T150_L3_f18_le: x_2nd_Flr_SF + 236.5400010000001 t150_leaf3 <= 1641.54
 T150_R3_f95_ge: x_Neighborhood_Crawfor - t150_leaf3 >= 0
 T150_R3_f22_ge: x_Bsmt_Half_Bath - t150_leaf3 >= 0
 T150_R4_f18_ge: x_2nd_Flr_SF - 1405 t150_leaf4 >= 0
 T150_L4_f44_le: x_Yr_Sold + 997992.000001 t150_leaf4 <= 1e+06
 T150_R5_f18_ge: x_2nd_Flr_SF - 1405 t150_leaf5 >= 0
 T150_R5_f44_ge: x_Yr_Sold - 2008 t150_leaf5 >= 0
 TREE_151_ONEHOT: t151_leaf0 + t151_leaf1 + t151_leaf2 + t151_leaf3
   + t151_leaf4 + t151_leaf5 + t151_leaf6 = 1
 T151_L0_f29_le: x_Fireplaces + 1.000001 t151_leaf0 <= 3
 T151_L0_f256_le: x_Garage_Finish_Unf + 1e-06 t151_leaf0 <= 1
 T151_L0_f33_le: x_Garage_Area + 663.750001 t151_leaf0 <= 1163.75
 T151_L1_f29_le: x_Fireplaces + 1.000001 t151_leaf1 <= 3
 T151_L1_f256_le: x_Garage_Finish_Unf + 1e-06 t151_leaf1 <= 1
 T151_R1_f33_ge: x_Garage_Area - 500 t151_leaf1 >= 0
 T151_L2_f29_le: x_Fireplaces + 1.000001 t151_leaf2 <= 3
 T151_R2_f256_ge: x_Garage_Finish_Unf - t151_leaf2 >= 0
 T151_L2_f4_le: x_Overall_Cond + 999994.000001 t151_leaf2 <= 1e+06
 T151_L3_f29_le: x_Fireplaces + 1.000001 t151_leaf3 <= 3
 T151_R3_f256_ge: x_Garage_Finish_Unf - t151_leaf3 >= 0
 T151_R3_f4_ge: x_Overall_Cond - 6 t151_leaf3 >= 0
 T151_R4_f29_ge: x_Fireplaces - 2 t151_leaf4 >= 0
 T151_L4_f17_le: x_1st_Flr_SF + 1341.900001 t151_leaf4 <= 2534.9
 T151_R5_f29_ge: x_Fireplaces - 2 t151_leaf5 >= 0
 T151_R5_f17_ge: x_1st_Flr_SF - 1193 t151_leaf5 >= 0
 T151_L5_f180_le: x_Exterior_2nd_BrkFace + 1e-06 t151_leaf5 <= 1
 T151_R6_f29_ge: x_Fireplaces - 2 t151_leaf6 >= 0
 T151_R6_f17_ge: x_1st_Flr_SF - 1193 t151_leaf6 >= 0
 T151_R6_f180_ge: x_Exterior_2nd_BrkFace - t151_leaf6 >= 0
 TREE_152_ONEHOT: t152_leaf0 + t152_leaf1 + t152_leaf2 + t152_leaf3
   + t152_leaf4 + t152_leaf5 = 1
 T152_L0_f40_le: x_Screen_Porch + 118.270001 t152_leaf0 <= 268.27
 T152_L0_f20_le: x_Gr_Liv_Area + 2485.760001 t152_leaf0 <= 3421.76
 T152_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t152_leaf0 <= 1
 T152_L1_f40_le: x_Screen_Porch + 118.270001 t152_leaf1 <= 268.27
 T152_L1_f20_le: x_Gr_Liv_Area + 2485.760001 t152_leaf1 <= 3421.76
 T152_R1_f297_ge: x_Sale_Condition_Normal - t152_leaf1 >= 0
 T152_L2_f40_le: x_Screen_Porch + 118.270001 t152_leaf2 <= 268.27
 T152_R2_f20_ge: x_Gr_Liv_Area - 936 t152_leaf2 >= 0
 T152_L2_f4_le: x_Overall_Cond + 999994.000001 t152_leaf2 <= 1e+06
 T152_L3_f40_le: x_Screen_Porch + 118.270001 t152_leaf3 <= 268.27
 T152_R3_f20_ge: x_Gr_Liv_Area - 936 t152_leaf3 >= 0
 T152_R3_f4_ge: x_Overall_Cond - 6 t152_leaf3 >= 0
 T152_R4_f40_ge: x_Screen_Porch - 150 t152_leaf4 >= 0
 T152_L4_f20_le: x_Gr_Liv_Area + 1444.760001 t152_leaf4 <= 3421.76
 T152_R5_f40_ge: x_Screen_Porch - 150 t152_leaf5 >= 0
 T152_R5_f20_ge: x_Gr_Liv_Area - 1977 t152_leaf5 >= 0
 TREE_153_ONEHOT: t153_leaf0 + t153_leaf1 + t153_leaf2 + t153_leaf3
   + t153_leaf4 + t153_leaf5 + t153_leaf6 + t153_leaf7 = 1
 T153_L0_f1_le: x_Lot_Area <= 13614.999999
 T153_L0_f1_le: x_Lot_Area + 1000.000001 t153_leaf0 <= 7000
 T153_L0_f35_le: x_Garage_Cond + 2.000001 t153_leaf0 <= 4
 T153_L1_f1_le: x_Lot_Area <= 13614.999999
 T153_L1_f1_le: x_Lot_Area + 1000.000001 t153_leaf1 <= 7000
 T153_R1_f35_ge: x_Garage_Cond - 3 t153_leaf1 >= -1
 T153_L2_f1_le: x_Lot_Area <= 13614.999999
 T153_R2_f1_ge: x_Lot_Area >= 6000
 T153_L2_f119_le: x_Condition_1_Norm + 1e-06 t153_leaf2 <= 1
 T153_L3_f1_le: x_Lot_Area <= 13614.999999
 T153_R3_f1_ge: x_Lot_Area >= 6000
 T153_R3_f119_ge: x_Condition_1_Norm - t153_leaf3 >= 0
 T153_R4_f1_ge: x_Lot_Area - 6615 t153_leaf4 >= 7000
 T153_L4_f25_le: x_Bedroom_AbvGr + 3.000001 t153_leaf4 <= 7
 T153_L4_f20_le: x_Gr_Liv_Area + 693.7600010000001 t153_leaf4 <= 3421.76
 T153_R5_f1_ge: x_Lot_Area - 6615 t153_leaf5 >= 7000
 T153_L5_f25_le: x_Bedroom_AbvGr + 3.000001 t153_leaf5 <= 7
 T153_R5_f20_ge: x_Gr_Liv_Area - 2728 t153_leaf5 >= 0
 T153_R6_f1_ge: x_Lot_Area - 6615 t153_leaf6 >= 7000
 T153_R6_f25_ge: x_Bedroom_AbvGr - 4 t153_leaf6 >= 0
 T153_L6_f20_le: x_Gr_Liv_Area + 335.7600010000001 t153_leaf6 <= 3421.76
 T153_R7_f1_ge: x_Lot_Area - 6615 t153_leaf7 >= 7000
 T153_R7_f25_ge: x_Bedroom_AbvGr - 4 t153_leaf7 >= 0
 T153_R7_f20_ge: x_Gr_Liv_Area - 3086 t153_leaf7 >= 0
 TREE_154_ONEHOT: t154_leaf0 + t154_leaf1 + t154_leaf2 + t154_leaf3
   + t154_leaf4 + t154_leaf5 + t154_leaf6 + t154_leaf7 = 1
 T154_L0_f16_le: x_Heating_QC + 1.000001 t154_leaf0 <= 4
 T154_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t154_leaf0 <= 1
 T154_L0_f44_le: x_Yr_Sold + 997993.000001 t154_leaf0 <= 1e+06
 T154_L1_f16_le: x_Heating_QC + 1.000001 t154_leaf1 <= 4
 T154_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t154_leaf1 <= 1
 T154_R1_f44_ge: x_Yr_Sold - 2007 t154_leaf1 >= 0
 T154_L2_f16_le: x_Heating_QC + 1.000001 t154_leaf2 <= 4
 T154_R2_f297_ge: x_Sale_Condition_Normal - t154_leaf2 >= 0
 T154_L2_f4_le: x_Overall_Cond + 999993.000001 t154_leaf2 <= 1e+06
 T154_L3_f16_le: x_Heating_QC + 1.000001 t154_leaf3 <= 4
 T154_R3_f297_ge: x_Sale_Condition_Normal - t154_leaf3 >= 0
 T154_R3_f4_ge: x_Overall_Cond - 7 t154_leaf3 >= 0
 T154_R4_f16_ge: x_Heating_QC - t154_leaf4 >= 2
 T154_L4_f8_le: x_Exter_Qual + 1e-06 t154_leaf4 <= 4
 T154_L4_f44_le: x_Yr_Sold + 997993.000001 t154_leaf4 <= 1e+06
 T154_R5_f16_ge: x_Heating_QC - t154_leaf5 >= 2
 T154_L5_f8_le: x_Exter_Qual + 1e-06 t154_leaf5 <= 4
 T154_R5_f44_ge: x_Yr_Sold - 2007 t154_leaf5 >= 0
 T154_R6_f16_ge: x_Heating_QC - t154_leaf6 >= 2
 T154_R6_f8_ge: x_Exter_Qual - 2 t154_leaf6 >= 2
 T154_L6_f31_le: x_Garage_Yr_Blt + 997992.000001 t154_leaf6 <= 1e+06
 T154_R7_f16_ge: x_Heating_QC - t154_leaf7 >= 2
 T154_R7_f8_ge: x_Exter_Qual - 2 t154_leaf7 >= 2
 T154_R7_f31_ge: x_Garage_Yr_Blt - 2008 t154_leaf7 >= 0
 TREE_155_ONEHOT: t155_leaf0 + t155_leaf1 + t155_leaf2 + t155_leaf3
   + t155_leaf4 + t155_leaf5 = 1
 T155_L0_f44_le: x_Yr_Sold + 997992.000001 t155_leaf0 <= 1e+06
 T155_L0_f18_le: x_2nd_Flr_SF + 214.5400010000001 t155_leaf0 <= 1641.54
 T155_L0_f4_le: x_Overall_Cond + 999995.000001 t155_leaf0 <= 1e+06
 T155_L1_f44_le: x_Yr_Sold + 997992.000001 t155_leaf1 <= 1e+06
 T155_L1_f18_le: x_2nd_Flr_SF + 214.5400010000001 t155_leaf1 <= 1641.54
 T155_R1_f4_ge: x_Overall_Cond - 5 t155_leaf1 >= 0
 T155_L2_f44_le: x_Yr_Sold + 997992.000001 t155_leaf2 <= 1e+06
 T155_R2_f18_ge: x_2nd_Flr_SF - 1427 t155_leaf2 >= 0
 T155_R3_f44_ge: x_Yr_Sold - 2008 t155_leaf3 >= 0
 T155_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t155_leaf3 <= 1
 T155_L3_f202_le: x_Foundation_PConc + 1e-06 t155_leaf3 <= 1
 T155_R4_f44_ge: x_Yr_Sold - 2008 t155_leaf4 >= 0
 T155_L4_f95_le: x_Neighborhood_Crawfor + 1e-06 t155_leaf4 <= 1
 T155_R4_f202_ge: x_Foundation_PConc - t155_leaf4 >= 0
 T155_R5_f44_ge: x_Yr_Sold - 2008 t155_leaf5 >= 0
 T155_R5_f95_ge: x_Neighborhood_Crawfor - t155_leaf5 >= 0
 TREE_156_ONEHOT: t156_leaf0 + t156_leaf1 + t156_leaf2 + t156_leaf3
   + t156_leaf4 = 1
 T156_L0_f27_le: x_Kitchen_Qual + 1e-06 t156_leaf0 <= 4
 T156_L0_f12_le: x_BsmtFin_SF_1 + 1957.500001 t156_leaf0 <= 1984.5
 T156_L0_f17_le: x_1st_Flr_SF + 1918.900001 t156_leaf0 <= 2534.9
 T156_L1_f27_le: x_Kitchen_Qual + 1e-06 t156_leaf1 <= 4
 T156_L1_f12_le: x_BsmtFin_SF_1 + 1957.500001 t156_leaf1 <= 1984.5
 T156_R1_f17_ge: x_1st_Flr_SF - 616 t156_leaf1 >= 0
 T156_L2_f27_le: x_Kitchen_Qual + 1e-06 t156_leaf2 <= 4
 T156_R2_f12_ge: x_BsmtFin_SF_1 - 27 t156_leaf2 >= 0
 T156_L2_f31_le: x_Garage_Yr_Blt + 997995.000001 t156_leaf2 <= 1e+06
 T156_L3_f27_le: x_Kitchen_Qual + 1e-06 t156_leaf3 <= 4
 T156_R3_f12_ge: x_BsmtFin_SF_1 - 27 t156_leaf3 >= 0
 T156_R3_f31_ge: x_Garage_Yr_Blt - 2005 t156_leaf3 >= 0
 T156_R4_f27_ge: x_Kitchen_Qual - 2 t156_leaf4 >= 2
 TREE_157_ONEHOT: t157_leaf0 + t157_leaf1 + t157_leaf2 + t157_leaf3
   + t157_leaf4 + t157_leaf5 + t157_leaf6 = 1
 T157_L0_f1_le: x_Lot_Area <= 7949.999999
 T157_L0_f298_le: x_Sale_Condition_Partial + 1e-06 t157_leaf0 <= 1
 T157_L0_f12_le: x_BsmtFin_SF_1 + 1964.500001 t157_leaf0 <= 1984.5
 T157_L1_f1_le: x_Lot_Area <= 7949.999999
 T157_L1_f298_le: x_Sale_Condition_Partial + 1e-06 t157_leaf1 <= 1
 T157_R1_f12_ge: x_BsmtFin_SF_1 - 20 t157_leaf1 >= 0
 T157_L2_f1_le: x_Lot_Area <= 7949.999999
 T157_R2_f298_ge: x_Sale_Condition_Partial - t157_leaf2 >= 0
 T157_R3_f1_ge: x_Lot_Area - 950 t157_leaf3 >= 7000
 T157_L3_f86_le: x_Land_Slope_Gtl + 1e-06 t157_leaf3 <= 1
 T157_L3_f1_le: x_Lot_Area <= 9532.999999
 T157_R4_f1_ge: x_Lot_Area - 950 t157_leaf4 >= 7000
 T157_L4_f86_le: x_Land_Slope_Gtl + 1e-06 t157_leaf4 <= 1
 T157_R4_f1_ge: x_Lot_Area - 2533 t157_leaf4 >= 7000
 T157_R5_f1_ge: x_Lot_Area - 950 t157_leaf5 >= 7000
 T157_R5_f86_ge: x_Land_Slope_Gtl - t157_leaf5 >= 0
 T157_L5_f4_le: x_Overall_Cond + 999994.000001 t157_leaf5 <= 1e+06
 T157_R6_f1_ge: x_Lot_Area - 950 t157_leaf6 >= 7000
 T157_R6_f86_ge: x_Land_Slope_Gtl - t157_leaf6 >= 0
 T157_R6_f4_ge: x_Overall_Cond - 6 t157_leaf6 >= 0
 TREE_158_ONEHOT: t158_leaf0 + t158_leaf1 + t158_leaf2 + t158_leaf3
   + t158_leaf4 + t158_leaf5 = 1
 T158_L0_f3_le: x_Overall_Qual + 999991.000001 t158_leaf0 <= 1e+06
 T158_L0_f4_le: x_Overall_Cond + 999993.000001 t158_leaf0 <= 1e+06
 T158_L0_f5_le: x_Year_Built + 998089.000001 t158_leaf0 <= 1e+06
 T158_L1_f3_le: x_Overall_Qual + 999991.000001 t158_leaf1 <= 1e+06
 T158_L1_f4_le: x_Overall_Cond + 999993.000001 t158_leaf1 <= 1e+06
 T158_R1_f5_ge: x_Year_Built - 1911 t158_leaf1 >= 0
 T158_L2_f3_le: x_Overall_Qual + 999991.000001 t158_leaf2 <= 1e+06
 T158_R2_f4_ge: x_Overall_Cond - 7 t158_leaf2 >= 0
 T158_L2_f30_le: x_Fireplace_Qu + 2.000001 t158_leaf2 <= 4
 T158_L3_f3_le: x_Overall_Qual + 999991.000001 t158_leaf3 <= 1e+06
 T158_R3_f4_ge: x_Overall_Cond - 7 t158_leaf3 >= 0
 T158_R3_f30_ge: x_Fireplace_Qu >= 2
 T158_R4_f3_ge: x_Overall_Qual - 9 t158_leaf4 >= 0
 T158_L4_f0_le: x_Lot_Frontage + 60.530001 t158_leaf4 <= 175.53
 T158_R5_f3_ge: x_Overall_Qual - 9 t158_leaf5 >= 0
 T158_R5_f0_ge: x_Lot_Frontage - 115 t158_leaf5 >= 0
 TREE_159_ONEHOT: t159_leaf0 + t159_leaf1 + t159_leaf2 + t159_leaf3
   + t159_leaf4 + t159_leaf5 = 1
 T159_L0_f3_le: x_Overall_Qual + 999991.000001 t159_leaf0 <= 1e+06
 T159_L0_f44_le: x_Yr_Sold + 997993.000001 t159_leaf0 <= 1e+06
 T159_L0_f12_le: x_BsmtFin_SF_1 + 1559.500001 t159_leaf0 <= 1984.5
 T159_L1_f3_le: x_Overall_Qual + 999991.000001 t159_leaf1 <= 1e+06
 T159_L1_f44_le: x_Yr_Sold + 997993.000001 t159_leaf1 <= 1e+06
 T159_R1_f12_ge: x_BsmtFin_SF_1 - 425 t159_leaf1 >= 0
 T159_L2_f3_le: x_Overall_Qual + 999991.000001 t159_leaf2 <= 1e+06
 T159_R2_f44_ge: x_Yr_Sold - 2007 t159_leaf2 >= 0
 T159_L2_f231_le: x_Central_Air_N + 1e-06 t159_leaf2 <= 1
 T159_L3_f3_le: x_Overall_Qual + 999991.000001 t159_leaf3 <= 1e+06
 T159_R3_f44_ge: x_Yr_Sold - 2007 t159_leaf3 >= 0
 T159_R3_f231_ge: x_Central_Air_N - t159_leaf3 >= 0
 T159_R4_f3_ge: x_Overall_Qual - 9 t159_leaf4 >= 0
 T159_L4_f17_le: x_1st_Flr_SF + 723.9000010000002 t159_leaf4 <= 2534.9
 T159_R5_f3_ge: x_Overall_Qual - 9 t159_leaf5 >= 0
 T159_R5_f17_ge: x_1st_Flr_SF - 1811 t159_leaf5 >= 0
 TREE_160_ONEHOT: t160_leaf0 + t160_leaf1 + t160_leaf2 + t160_leaf3
   + t160_leaf4 + t160_leaf5 + t160_leaf6 = 1
 T160_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t160_leaf0 <= 1
 T160_L0_f119_le: x_Condition_1_Norm + 1e-06 t160_leaf0 <= 1
 T160_L0_f17_le: x_1st_Flr_SF + 1202.900001 t160_leaf0 <= 2534.9
 T160_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t160_leaf1 <= 1
 T160_L1_f119_le: x_Condition_1_Norm + 1e-06 t160_leaf1 <= 1
 T160_R1_f17_ge: x_1st_Flr_SF - 1332 t160_leaf1 >= 0
 T160_L2_f161_le: x_Exterior_1st_AsbShng + 1e-06 t160_leaf2 <= 1
 T160_R2_f119_ge: x_Condition_1_Norm - t160_leaf2 >= 0
 T160_L2_f76_le: x_Lot_Shape_Reg + 1e-06 t160_leaf2 <= 1
 T160_L3_f161_le: x_Exterior_1st_AsbShng + 1e-06 t160_leaf3 <= 1
 T160_R3_f119_ge: x_Condition_1_Norm - t160_leaf3 >= 0
 T160_R3_f76_ge: x_Lot_Shape_Reg - t160_leaf3 >= 0
 T160_R4_f161_ge: x_Exterior_1st_AsbShng - t160_leaf4 >= 0
 T160_L4_f17_le: x_1st_Flr_SF + 1699.900001 t160_leaf4 <= 2534.9
 T160_R5_f161_ge: x_Exterior_1st_AsbShng - t160_leaf5 >= 0
 T160_R5_f17_ge: x_1st_Flr_SF - 835 t160_leaf5 >= 0
 T160_L5_f233_le: x_Electrical_FuseA + 1e-06 t160_leaf5 <= 1
 T160_R6_f161_ge: x_Exterior_1st_AsbShng - t160_leaf6 >= 0
 T160_R6_f17_ge: x_1st_Flr_SF - 835 t160_leaf6 >= 0
 T160_R6_f233_ge: x_Electrical_FuseA - t160_leaf6 >= 0
 TREE_161_ONEHOT: t161_leaf0 + t161_leaf1 + t161_leaf2 + t161_leaf3
   + t161_leaf4 + t161_leaf5 + t161_leaf6 + t161_leaf7 = 1
 T161_L0_f44_le: x_Yr_Sold + 997992.000001 t161_leaf0 <= 1e+06
 T161_L0_f36_le: x_Wood_Deck_SF + 402.900001 t161_leaf0 <= 525.9
 T161_L0_f4_le: x_Overall_Cond + 999994.000001 t161_leaf0 <= 1e+06
 T161_L1_f44_le: x_Yr_Sold + 997992.000001 t161_leaf1 <= 1e+06
 T161_L1_f36_le: x_Wood_Deck_SF + 402.900001 t161_leaf1 <= 525.9
 T161_R1_f4_ge: x_Overall_Cond - 6 t161_leaf1 >= 0
 T161_L2_f44_le: x_Yr_Sold + 997992.000001 t161_leaf2 <= 1e+06
 T161_R2_f36_ge: x_Wood_Deck_SF - 123 t161_leaf2 >= 0
 T161_L2_f17_le: x_1st_Flr_SF + 1120.900001 t161_leaf2 <= 2534.9
 T161_L3_f44_le: x_Yr_Sold + 997992.000001 t161_leaf3 <= 1e+06
 T161_R3_f36_ge: x_Wood_Deck_SF - 123 t161_leaf3 >= 0
 T161_R3_f17_ge: x_1st_Flr_SF - 1414 t161_leaf3 >= 0
 T161_R4_f44_ge: x_Yr_Sold - 2008 t161_leaf4 >= 0
 T161_L4_f119_le: x_Condition_1_Norm + 1e-06 t161_leaf4 <= 1
 T161_L4_f33_le: x_Garage_Area + 619.750001 t161_leaf4 <= 1163.75
 T161_R5_f44_ge: x_Yr_Sold - 2008 t161_leaf5 >= 0
 T161_L5_f119_le: x_Condition_1_Norm + 1e-06 t161_leaf5 <= 1
 T161_R5_f33_ge: x_Garage_Area - 544 t161_leaf5 >= 0
 T161_R6_f44_ge: x_Yr_Sold - 2008 t161_leaf6 >= 0
 T161_R6_f119_ge: x_Condition_1_Norm - t161_leaf6 >= 0
 T161_L6_f76_le: x_Lot_Shape_Reg + 1e-06 t161_leaf6 <= 1
 T161_R7_f44_ge: x_Yr_Sold - 2008 t161_leaf7 >= 0
 T161_R7_f119_ge: x_Condition_1_Norm - t161_leaf7 >= 0
 T161_R7_f76_ge: x_Lot_Shape_Reg - t161_leaf7 >= 0
 TREE_162_ONEHOT: t162_leaf0 + t162_leaf1 + t162_leaf2 + t162_leaf3
   + t162_leaf4 + t162_leaf5 + t162_leaf6 + t162_leaf7 = 1
 T162_L0_f9_le: x_Exter_Cond + 2.000001 t162_leaf0 <= 4
 T162_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t162_leaf0 <= 1
 T162_L0_f28_le: x_TotRms_AbvGrd + 6.080001 t162_leaf0 <= 12.08
 T162_L1_f9_le: x_Exter_Cond + 2.000001 t162_leaf1 <= 4
 T162_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t162_leaf1 <= 1
 T162_R1_f28_ge: x_TotRms_AbvGrd - 6 t162_leaf1 >= 0
 T162_L2_f9_le: x_Exter_Cond + 2.000001 t162_leaf2 <= 4
 T162_R2_f297_ge: x_Sale_Condition_Normal - t162_leaf2 >= 0
 T162_L2_f251_le: x_Garage_Type_Detchd + 1e-06 t162_leaf2 <= 1
 T162_L3_f9_le: x_Exter_Cond + 2.000001 t162_leaf3 <= 4
 T162_R3_f297_ge: x_Sale_Condition_Normal - t162_leaf3 >= 0
 T162_R3_f251_ge: x_Garage_Type_Detchd - t162_leaf3 >= 0
 T162_R4_f9_ge: x_Exter_Cond - 3 t162_leaf4 >= -1
 T162_L4_f78_le: x_Land_Contour_HLS + 1e-06 t162_leaf4 <= 1
 T162_L4_f20_le: x_Gr_Liv_Area + 788.7600010000001 t162_leaf4 <= 3421.76
 T162_R5_f9_ge: x_Exter_Cond - 3 t162_leaf5 >= -1
 T162_L5_f78_le: x_Land_Contour_HLS + 1e-06 t162_leaf5 <= 1
 T162_R5_f20_ge: x_Gr_Liv_Area - 2633 t162_leaf5 >= 0
 T162_R6_f9_ge: x_Exter_Cond - 3 t162_leaf6 >= -1
 T162_R6_f78_ge: x_Land_Contour_HLS - t162_leaf6 >= 0
 T162_L6_f167_le: x_Exterior_1st_HdBoard + 1e-06 t162_leaf6 <= 1
 T162_R7_f9_ge: x_Exter_Cond - 3 t162_leaf7 >= -1
 T162_R7_f78_ge: x_Land_Contour_HLS - t162_leaf7 >= 0
 T162_R7_f167_ge: x_Exterior_1st_HdBoard - t162_leaf7 >= 0
 TREE_163_ONEHOT: t163_leaf0 + t163_leaf1 + t163_leaf2 + t163_leaf3
   + t163_leaf4 + t163_leaf5 + t163_leaf6 + t163_leaf7 = 1
 T163_L0_f1_le: x_Lot_Area + 1024.000001 t163_leaf0 <= 7000
 T163_L0_f12_le: x_BsmtFin_SF_1 + 1802.500001 t163_leaf0 <= 1984.5
 T163_L0_f34_le: x_Garage_Qual + 4.000001 t163_leaf0 <= 4
 T163_L1_f1_le: x_Lot_Area + 1024.000001 t163_leaf1 <= 7000
 T163_L1_f12_le: x_BsmtFin_SF_1 + 1802.500001 t163_leaf1 <= 1984.5
 T163_R1_f34_ge: x_Garage_Qual >= 0
 T163_L2_f1_le: x_Lot_Area + 1024.000001 t163_leaf2 <= 7000
 T163_R2_f12_ge: x_BsmtFin_SF_1 - 182 t163_leaf2 >= 0
 T163_L2_f5_le: x_Year_Built + 997995.000001 t163_leaf2 <= 1e+06
 T163_L3_f1_le: x_Lot_Area + 1024.000001 t163_leaf3 <= 7000
 T163_R3_f12_ge: x_BsmtFin_SF_1 - 182 t163_leaf3 >= 0
 T163_R3_f5_ge: x_Year_Built - 2005 t163_leaf3 >= 0
 T163_R4_f1_ge: x_Lot_Area >= 5976
 T163_L4_f40_le: x_Screen_Porch + 158.270001 t163_leaf4 <= 268.27
 T163_L4_f4_le: x_Overall_Cond + 999994.000001 t163_leaf4 <= 1e+06
 T163_R5_f1_ge: x_Lot_Area >= 5976
 T163_L5_f40_le: x_Screen_Porch + 158.270001 t163_leaf5 <= 268.27
 T163_R5_f4_ge: x_Overall_Cond - 6 t163_leaf5 >= 0
 T163_R6_f1_ge: x_Lot_Area >= 5976
 T163_R6_f40_ge: x_Screen_Porch - 110 t163_leaf6 >= 0
 T163_L6_f20_le: x_Gr_Liv_Area + 1268.760001 t163_leaf6 <= 3421.76
 T163_R7_f1_ge: x_Lot_Area >= 5976
 T163_R7_f40_ge: x_Screen_Porch - 110 t163_leaf7 >= 0
 T163_R7_f20_ge: x_Gr_Liv_Area - 2153 t163_leaf7 >= 0
 TREE_164_ONEHOT: t164_leaf0 + t164_leaf1 + t164_leaf2 + t164_leaf3
   + t164_leaf4 + t164_leaf5 + t164_leaf6 = 1
 T164_L0_f20_le: x_Gr_Liv_Area + 772.7600010000001 t164_leaf0 <= 3421.76
 T164_L0_f5_le: x_Year_Built + 997994.000001 t164_leaf0 <= 1e+06
 T164_L0_f11_le: x_Bsmt_Cond + 4.000001 t164_leaf0 <= 4
 T164_L1_f20_le: x_Gr_Liv_Area + 772.7600010000001 t164_leaf1 <= 3421.76
 T164_L1_f5_le: x_Year_Built + 997994.000001 t164_leaf1 <= 1e+06
 T164_R1_f11_ge: x_Bsmt_Cond - t164_leaf1 >= -1
 T164_L2_f20_le: x_Gr_Liv_Area + 772.7600010000001 t164_leaf2 <= 3421.76
 T164_R2_f5_ge: x_Year_Built - 2006 t164_leaf2 >= 0
 T164_L2_f20_le: x_Gr_Liv_Area + 1796.760001 t164_leaf2 <= 3421.76
 T164_L3_f20_le: x_Gr_Liv_Area + 772.7600010000001 t164_leaf3 <= 3421.76
 T164_R3_f5_ge: x_Year_Built - 2006 t164_leaf3 >= 0
 T164_R3_f20_ge: x_Gr_Liv_Area - 1625 t164_leaf3 >= 0
 T164_R4_f20_ge: x_Gr_Liv_Area - 2649 t164_leaf4 >= 0
 T164_L4_f37_le: x_Open_Porch_SF + 457.810001 t164_leaf4 <= 488.81
 T164_R5_f20_ge: x_Gr_Liv_Area - 2649 t164_leaf5 >= 0
 T164_R5_f37_ge: x_Open_Porch_SF - 31 t164_leaf5 >= 0
 T164_L5_f255_le: x_Garage_Finish_RFn + 1e-06 t164_leaf5 <= 1
 T164_R6_f20_ge: x_Gr_Liv_Area - 2649 t164_leaf6 >= 0
 T164_R6_f37_ge: x_Open_Porch_SF - 31 t164_leaf6 >= 0
 T164_R6_f255_ge: x_Garage_Finish_RFn - t164_leaf6 >= 0
 TREE_165_ONEHOT: t165_leaf0 + t165_leaf1 + t165_leaf2 + t165_leaf3
   + t165_leaf4 + t165_leaf5 = 1
 T165_L0_f18_le: x_2nd_Flr_SF + 214.5400010000001 t165_leaf0 <= 1641.54
 T165_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t165_leaf0 <= 1
 T165_L0_f1_le: x_Lot_Area <= 12215.999999
 T165_L1_f18_le: x_2nd_Flr_SF + 214.5400010000001 t165_leaf1 <= 1641.54
 T165_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t165_leaf1 <= 1
 T165_R1_f1_ge: x_Lot_Area - 5216 t165_leaf1 >= 7000
 T165_L2_f18_le: x_2nd_Flr_SF + 214.5400010000001 t165_leaf2 <= 1641.54
 T165_R2_f293_ge: x_Sale_Condition_Abnorml - t165_leaf2 >= 0
 T165_L2_f3_le: x_Overall_Qual + 999995.000001 t165_leaf2 <= 1e+06
 T165_L3_f18_le: x_2nd_Flr_SF + 214.5400010000001 t165_leaf3 <= 1641.54
 T165_R3_f293_ge: x_Sale_Condition_Abnorml - t165_leaf3 >= 0
 T165_R3_f3_ge: x_Overall_Qual - 5 t165_leaf3 >= 0
 T165_R4_f18_ge: x_2nd_Flr_SF - 1427 t165_leaf4 >= 0
 T165_L4_f37_le: x_Open_Porch_SF + 455.810001 t165_leaf4 <= 488.81
 T165_R5_f18_ge: x_2nd_Flr_SF - 1427 t165_leaf5 >= 0
 T165_R5_f37_ge: x_Open_Porch_SF - 33 t165_leaf5 >= 0
 TREE_166_ONEHOT: t166_leaf0 + t166_leaf1 + t166_leaf2 + t166_leaf3
   + t166_leaf4 + t166_leaf5 + t166_leaf6 = 1
 T166_L0_f4_le: x_Overall_Cond + 999993.000001 t166_leaf0 <= 1e+06
 T166_L0_f33_le: x_Garage_Area + 359.750001 t166_leaf0 <= 1163.75
 T166_L0_f5_le: x_Year_Built + 998064.000001 t166_leaf0 <= 1e+06
 T166_L1_f4_le: x_Overall_Cond + 999993.000001 t166_leaf1 <= 1e+06
 T166_L1_f33_le: x_Garage_Area + 359.750001 t166_leaf1 <= 1163.75
 T166_R1_f5_ge: x_Year_Built - 1936 t166_leaf1 >= 0
 T166_L2_f4_le: x_Overall_Cond + 999993.000001 t166_leaf2 <= 1e+06
 T166_R2_f33_ge: x_Garage_Area - 804 t166_leaf2 >= 0
 T166_L2_f23_le: x_Full_Bath + 3.000001 t166_leaf2 <= 6
 T166_L3_f4_le: x_Overall_Cond + 999993.000001 t166_leaf3 <= 1e+06
 T166_R3_f33_ge: x_Garage_Area - 804 t166_leaf3 >= 0
 T166_R3_f23_ge: x_Full_Bath - 3 t166_leaf3 >= 0
 T166_R4_f4_ge: x_Overall_Cond - 7 t166_leaf4 >= 0
 T166_L4_f20_le: x_Gr_Liv_Area + 2297.760001000001 t166_leaf4 <= 3421.76
 T166_L4_f5_le: x_Year_Built + 998102.000001 t166_leaf4 <= 1e+06
 T166_R5_f4_ge: x_Overall_Cond - 7 t166_leaf5 >= 0
 T166_L5_f20_le: x_Gr_Liv_Area + 2297.760001000001 t166_leaf5 <= 3421.76
 T166_R5_f5_ge: x_Year_Built - 1898 t166_leaf5 >= 0
 T166_R6_f4_ge: x_Overall_Cond - 7 t166_leaf6 >= 0
 T166_R6_f20_ge: x_Gr_Liv_Area - 1124 t166_leaf6 >= 0
 TREE_167_ONEHOT: t167_leaf0 + t167_leaf1 + t167_leaf2 + t167_leaf3
   + t167_leaf4 + t167_leaf5 + t167_leaf6 = 1
 T167_L0_f4_le: x_Overall_Cond + 999995.000001 t167_leaf0 <= 1e+06
 T167_L0_f35_le: x_Garage_Cond + 3.000001 t167_leaf0 <= 4
 T167_L1_f4_le: x_Overall_Cond + 999995.000001 t167_leaf1 <= 1e+06
 T167_R1_f35_ge: x_Garage_Cond - 2 t167_leaf1 >= -1
 T167_L1_f31_le: x_Garage_Yr_Blt + 998014.000001 t167_leaf1 <= 1e+06
 T167_L2_f4_le: x_Overall_Cond + 999995.000001 t167_leaf2 <= 1e+06
 T167_R2_f35_ge: x_Garage_Cond - 2 t167_leaf2 >= -1
 T167_R2_f31_ge: x_Garage_Yr_Blt - 1986 t167_leaf2 >= 0
 T167_R3_f4_ge: x_Overall_Cond - 5 t167_leaf3 >= 0
 T167_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t167_leaf3 <= 1
 T167_L3_f36_le: x_Wood_Deck_SF + 321.900001 t167_leaf3 <= 525.9
 T167_R4_f4_ge: x_Overall_Cond - 5 t167_leaf4 >= 0
 T167_L4_f95_le: x_Neighborhood_Crawfor + 1e-06 t167_leaf4 <= 1
 T167_R4_f36_ge: x_Wood_Deck_SF - 204 t167_leaf4 >= 0
 T167_R5_f4_ge: x_Overall_Cond - 5 t167_leaf5 >= 0
 T167_R5_f95_ge: x_Neighborhood_Crawfor - t167_leaf5 >= 0
 T167_L5_f231_le: x_Central_Air_N + 1e-06 t167_leaf5 <= 1
 T167_R6_f4_ge: x_Overall_Cond - 5 t167_leaf6 >= 0
 T167_R6_f95_ge: x_Neighborhood_Crawfor - t167_leaf6 >= 0
 T167_R6_f231_ge: x_Central_Air_N - t167_leaf6 >= 0
 TREE_168_ONEHOT: t168_leaf0 + t168_leaf1 + t168_leaf2 + t168_leaf3
   + t168_leaf4 + t168_leaf5 + t168_leaf6 + t168_leaf7 = 1
 T168_L0_f44_le: x_Yr_Sold + 997993.000001 t168_leaf0 <= 1e+06
 T168_L0_f14_le: x_Bsmt_Unf_SF + 919.6800010000001 t168_leaf0 <= 1812.68
 T168_L0_f4_le: x_Overall_Cond + 999996.000001 t168_leaf0 <= 1e+06
 T168_L1_f44_le: x_Yr_Sold + 997993.000001 t168_leaf1 <= 1e+06
 T168_L1_f14_le: x_Bsmt_Unf_SF + 919.6800010000001 t168_leaf1 <= 1812.68
 T168_R1_f4_ge: x_Overall_Cond - 4 t168_leaf1 >= 0
 T168_L2_f44_le: x_Yr_Sold + 997993.000001 t168_leaf2 <= 1e+06
 T168_R2_f14_ge: x_Bsmt_Unf_SF - 893 t168_leaf2 >= 0
 T168_L2_f36_le: x_Wood_Deck_SF + 375.900001 t168_leaf2 <= 525.9
 T168_L3_f44_le: x_Yr_Sold + 997993.000001 t168_leaf3 <= 1e+06
 T168_R3_f14_ge: x_Bsmt_Unf_SF - 893 t168_leaf3 >= 0
 T168_R3_f36_ge: x_Wood_Deck_SF - 150 t168_leaf3 >= 0
 T168_R4_f44_ge: x_Yr_Sold - 2007 t168_leaf4 >= 0
 T168_L4_f293_le: x_Sale_Condition_Abnorml + 1e-06 t168_leaf4 <= 1
 T168_L4_f289_le: x_Sale_Type_New + 1e-06 t168_leaf4 <= 1
 T168_R5_f44_ge: x_Yr_Sold - 2007 t168_leaf5 >= 0
 T168_L5_f293_le: x_Sale_Condition_Abnorml + 1e-06 t168_leaf5 <= 1
 T168_R5_f289_ge: x_Sale_Type_New - t168_leaf5 >= 0
 T168_R6_f44_ge: x_Yr_Sold - 2007 t168_leaf6 >= 0
 T168_R6_f293_ge: x_Sale_Condition_Abnorml - t168_leaf6 >= 0
 T168_L6_f35_le: x_Garage_Cond + 2.000001 t168_leaf6 <= 4
 T168_R7_f44_ge: x_Yr_Sold - 2007 t168_leaf7 >= 0
 T168_R7_f293_ge: x_Sale_Condition_Abnorml - t168_leaf7 >= 0
 T168_R7_f35_ge: x_Garage_Cond - 3 t168_leaf7 >= -1
 TREE_169_ONEHOT: t169_leaf0 + t169_leaf1 + t169_leaf2 + t169_leaf3
   + t169_leaf4 + t169_leaf5 + t169_leaf6 = 1
 T169_L0_f3_le: x_Overall_Qual + 999991.000001 t169_leaf0 <= 1e+06
 T169_L0_f4_le: x_Overall_Cond + 999993.000001 t169_leaf0 <= 1e+06
 T169_L0_f5_le: x_Year_Built + 998068.000001 t169_leaf0 <= 1e+06
 T169_L1_f3_le: x_Overall_Qual + 999991.000001 t169_leaf1 <= 1e+06
 T169_L1_f4_le: x_Overall_Cond + 999993.000001 t169_leaf1 <= 1e+06
 T169_R1_f5_ge: x_Year_Built - 1932 t169_leaf1 >= 0
 T169_L2_f3_le: x_Overall_Qual + 999991.000001 t169_leaf2 <= 1e+06
 T169_R2_f4_ge: x_Overall_Cond - 7 t169_leaf2 >= 0
 T169_L2_f40_le: x_Screen_Porch + 228.270001 t169_leaf2 <= 268.27
 T169_L3_f3_le: x_Overall_Qual + 999991.000001 t169_leaf3 <= 1e+06
 T169_R3_f4_ge: x_Overall_Cond - 7 t169_leaf3 >= 0
 T169_R3_f40_ge: x_Screen_Porch - 40 t169_leaf3 >= 0
 T169_R4_f3_ge: x_Overall_Qual - 9 t169_leaf4 >= 0
 T169_L4_f37_le: x_Open_Porch_SF + 357.810001 t169_leaf4 <= 488.81
 T169_R5_f3_ge: x_Overall_Qual - 9 t169_leaf5 >= 0
 T169_R5_f37_ge: x_Open_Porch_SF - 131 t169_leaf5 >= 0
 T169_L5_f50_le: x_MS_SubClass_60 + 1e-06 t169_leaf5 <= 1
 T169_R6_f3_ge: x_Overall_Qual - 9 t169_leaf6 >= 0
 T169_R6_f37_ge: x_Open_Porch_SF - 131 t169_leaf6 >= 0
 T169_R6_f50_ge: x_MS_SubClass_60 - t169_leaf6 >= 0
 TREE_170_ONEHOT: t170_leaf0 + t170_leaf1 + t170_leaf2 + t170_leaf3
   + t170_leaf4 + t170_leaf5 = 1
 T170_L0_f20_le: x_Gr_Liv_Area + 335.7600010000001 t170_leaf0 <= 3421.76
 T170_L0_f5_le: x_Year_Built + 997995.000001 t170_leaf0 <= 1e+06
 T170_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t170_leaf0 <= 1
 T170_L1_f20_le: x_Gr_Liv_Area + 335.7600010000001 t170_leaf1 <= 3421.76
 T170_L1_f5_le: x_Year_Built + 997995.000001 t170_leaf1 <= 1e+06
 T170_R1_f164_ge: x_Exterior_1st_BrkFace - t170_leaf1 >= 0
 T170_L2_f20_le: x_Gr_Liv_Area + 335.7600010000001 t170_leaf2 <= 3421.76
 T170_R2_f5_ge: x_Year_Built - 2005 t170_leaf2 >= 0
 T170_L2_f12_le: x_BsmtFin_SF_1 + 1868.500001 t170_leaf2 <= 1984.5
 T170_L3_f20_le: x_Gr_Liv_Area + 335.7600010000001 t170_leaf3 <= 3421.76
 T170_R3_f5_ge: x_Year_Built - 2005 t170_leaf3 >= 0
 T170_R3_f12_ge: x_BsmtFin_SF_1 - 116 t170_leaf3 >= 0
 T170_R4_f20_ge: x_Gr_Liv_Area - 3086 t170_leaf4 >= 0
 T170_L4_f37_le: x_Open_Porch_SF + 484.810001 t170_leaf4 <= 488.81
 T170_R5_f20_ge: x_Gr_Liv_Area - 3086 t170_leaf5 >= 0
 T170_R5_f37_ge: x_Open_Porch_SF - 4 t170_leaf5 >= 0
 TREE_171_ONEHOT: t171_leaf0 + t171_leaf1 + t171_leaf2 + t171_leaf3
   + t171_leaf4 + t171_leaf5 + t171_leaf6 = 1
 T171_L0_f16_le: x_Heating_QC + 1.000001 t171_leaf0 <= 4
 T171_L0_f296_le: x_Sale_Condition_Family + 1e-06 t171_leaf0 <= 1
 T171_L0_f36_le: x_Wood_Deck_SF + 371.900001 t171_leaf0 <= 525.9
 T171_L1_f16_le: x_Heating_QC + 1.000001 t171_leaf1 <= 4
 T171_L1_f296_le: x_Sale_Condition_Family + 1e-06 t171_leaf1 <= 1
 T171_R1_f36_ge: x_Wood_Deck_SF - 154 t171_leaf1 >= 0
 T171_L2_f16_le: x_Heating_QC + 1.000001 t171_leaf2 <= 4
 T171_R2_f296_ge: x_Sale_Condition_Family - t171_leaf2 >= 0
 T171_L2_f5_le: x_Year_Built + 998073.000001 t171_leaf2 <= 1e+06
 T171_L3_f16_le: x_Heating_QC + 1.000001 t171_leaf3 <= 4
 T171_R3_f296_ge: x_Sale_Condition_Family - t171_leaf3 >= 0
 T171_R3_f5_ge: x_Year_Built - 1927 t171_leaf3 >= 0
 T171_R4_f16_ge: x_Heating_QC - t171_leaf4 >= 2
 T171_L4_f18_le: x_2nd_Flr_SF + 273.5400010000001 t171_leaf4 <= 1641.54
 T171_L4_f0_le: x_Lot_Frontage + 57.530001 t171_leaf4 <= 175.53
 T171_R5_f16_ge: x_Heating_QC - t171_leaf5 >= 2
 T171_L5_f18_le: x_2nd_Flr_SF + 273.5400010000001 t171_leaf5 <= 1641.54
 T171_R5_f0_ge: x_Lot_Frontage - 118 t171_leaf5 >= 0
 T171_R6_f16_ge: x_Heating_QC - t171_leaf6 >= 2
 T171_R6_f18_ge: x_2nd_Flr_SF - 1368 t171_leaf6 >= 0
 TREE_172_ONEHOT: t172_leaf0 + t172_leaf1 + t172_leaf2 + t172_leaf3
   + t172_leaf4 + t172_leaf5 = 1
 T172_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t172_leaf0 <= 1
 T172_L0_f36_le: x_Wood_Deck_SF + 357.900001 t172_leaf0 <= 525.9
 T172_L0_f44_le: x_Yr_Sold + 997993.000001 t172_leaf0 <= 1e+06
 T172_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t172_leaf1 <= 1
 T172_L1_f36_le: x_Wood_Deck_SF + 357.900001 t172_leaf1 <= 525.9
 T172_R1_f44_ge: x_Yr_Sold - 2007 t172_leaf1 >= 0
 T172_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t172_leaf2 <= 1
 T172_R2_f36_ge: x_Wood_Deck_SF - 168 t172_leaf2 >= 0
 T172_L2_f0_le: x_Lot_Frontage + 55.530001 t172_leaf2 <= 175.53
 T172_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t172_leaf3 <= 1
 T172_R3_f36_ge: x_Wood_Deck_SF - 168 t172_leaf3 >= 0
 T172_R3_f0_ge: x_Lot_Frontage - 120 t172_leaf3 >= 0
 T172_R4_f95_ge: x_Neighborhood_Crawfor - t172_leaf4 >= 0
 T172_L4_f0_le: x_Lot_Frontage + 121.530001 t172_leaf4 <= 175.53
 T172_R5_f95_ge: x_Neighborhood_Crawfor - t172_leaf5 >= 0
 T172_R5_f0_ge: x_Lot_Frontage - 54 t172_leaf5 >= 0
 TREE_173_ONEHOT: t173_leaf0 + t173_leaf1 + t173_leaf2 + t173_leaf3
   + t173_leaf4 + t173_leaf5 + t173_leaf6 = 1
 T173_L0_f29_le: x_Fireplaces + 1.000001 t173_leaf0 <= 3
 T173_L0_f44_le: x_Yr_Sold + 997993.000001 t173_leaf0 <= 1e+06
 T173_L0_f17_le: x_1st_Flr_SF + 1888.900001 t173_leaf0 <= 2534.9
 T173_L1_f29_le: x_Fireplaces + 1.000001 t173_leaf1 <= 3
 T173_L1_f44_le: x_Yr_Sold + 997993.000001 t173_leaf1 <= 1e+06
 T173_R1_f17_ge: x_1st_Flr_SF - 646 t173_leaf1 >= 0
 T173_L2_f29_le: x_Fireplaces + 1.000001 t173_leaf2 <= 3
 T173_R2_f44_ge: x_Yr_Sold - 2007 t173_leaf2 >= 0
 T173_L2_f21_le: x_Bsmt_Full_Bath + 1.020001 t173_leaf2 <= 2.02
 T173_L3_f29_le: x_Fireplaces + 1.000001 t173_leaf3 <= 3
 T173_R3_f44_ge: x_Yr_Sold - 2007 t173_leaf3 >= 0
 T173_R3_f21_ge: x_Bsmt_Full_Bath - t173_leaf3 >= 0
 T173_R4_f29_ge: x_Fireplaces - 2 t173_leaf4 >= 0
 T173_L4_f37_le: x_Open_Porch_SF + 449.810001 t173_leaf4 <= 488.81
 T173_L4_f95_le: x_Neighborhood_Crawfor + 1e-06 t173_leaf4 <= 1
 T173_R5_f29_ge: x_Fireplaces - 2 t173_leaf5 >= 0
 T173_L5_f37_le: x_Open_Porch_SF + 449.810001 t173_leaf5 <= 488.81
 T173_R5_f95_ge: x_Neighborhood_Crawfor - t173_leaf5 >= 0
 T173_R6_f29_ge: x_Fireplaces - 2 t173_leaf6 >= 0
 T173_R6_f37_ge: x_Open_Porch_SF - 39 t173_leaf6 >= 0
 TREE_174_ONEHOT: t174_leaf0 + t174_leaf1 + t174_leaf2 + t174_leaf3
   + t174_leaf4 + t174_leaf5 + t174_leaf6 = 1
 T174_L0_f44_le: x_Yr_Sold + 997993.000001 t174_leaf0 <= 1e+06
 T174_L0_f1_le: x_Lot_Area + 4240.000001 t174_leaf0 <= 7000
 T174_L1_f44_le: x_Yr_Sold + 997993.000001 t174_leaf1 <= 1e+06
 T174_R1_f1_ge: x_Lot_Area >= 2760
 T174_L1_f1_le: x_Lot_Area <= 17226.999999
 T174_L2_f44_le: x_Yr_Sold + 997993.000001 t174_leaf2 <= 1e+06
 T174_R2_f1_ge: x_Lot_Area >= 2760
 T174_R2_f1_ge: x_Lot_Area - 10227 t174_leaf2 >= 7000
 T174_R3_f44_ge: x_Yr_Sold - 2007 t174_leaf3 >= 0
 T174_L3_f78_le: x_Land_Contour_HLS + 1e-06 t174_leaf3 <= 1
 T174_L3_f37_le: x_Open_Porch_SF + 201.810001 t174_leaf3 <= 488.81
 T174_R4_f44_ge: x_Yr_Sold - 2007 t174_leaf4 >= 0
 T174_L4_f78_le: x_Land_Contour_HLS + 1e-06 t174_leaf4 <= 1
 T174_R4_f37_ge: x_Open_Porch_SF - 287 t174_leaf4 >= 0
 T174_R5_f44_ge: x_Yr_Sold - 2007 t174_leaf5 >= 0
 T174_R5_f78_ge: x_Land_Contour_HLS - t174_leaf5 >= 0
 T174_L5_f167_le: x_Exterior_1st_HdBoard + 1e-06 t174_leaf5 <= 1
 T174_R6_f44_ge: x_Yr_Sold - 2007 t174_leaf6 >= 0
 T174_R6_f78_ge: x_Land_Contour_HLS - t174_leaf6 >= 0
 T174_R6_f167_ge: x_Exterior_1st_HdBoard - t174_leaf6 >= 0
 TREE_175_ONEHOT: t175_leaf0 + t175_leaf1 + t175_leaf2 + t175_leaf3
   + t175_leaf4 = 1
 T175_L0_f3_le: x_Overall_Qual + 999990.000001 t175_leaf0 <= 1e+06
 T175_L0_f44_le: x_Yr_Sold + 997992.000001 t175_leaf0 <= 1e+06
 T175_L0_f12_le: x_BsmtFin_SF_1 + 1761.500001 t175_leaf0 <= 1984.5
 T175_L1_f3_le: x_Overall_Qual + 999990.000001 t175_leaf1 <= 1e+06
 T175_L1_f44_le: x_Yr_Sold + 997992.000001 t175_leaf1 <= 1e+06
 T175_R1_f12_ge: x_BsmtFin_SF_1 - 223 t175_leaf1 >= 0
 T175_L2_f3_le: x_Overall_Qual + 999990.000001 t175_leaf2 <= 1e+06
 T175_R2_f44_ge: x_Yr_Sold - 2008 t175_leaf2 >= 0
 T175_L2_f293_le: x_Sale_Condition_Abnorml + 1e-06 t175_leaf2 <= 1
 T175_L3_f3_le: x_Overall_Qual + 999990.000001 t175_leaf3 <= 1e+06
 T175_R3_f44_ge: x_Yr_Sold - 2008 t175_leaf3 >= 0
 T175_R3_f293_ge: x_Sale_Condition_Abnorml - t175_leaf3 >= 0
 T175_R4_f3_ge: x_Overall_Qual - 10 t175_leaf4 >= 0
 TREE_176_ONEHOT: t176_leaf0 + t176_leaf1 + t176_leaf2 + t176_leaf3
   + t176_leaf4 = 1
 T176_L0_f23_le: x_Full_Bath + 3.000001 t176_leaf0 <= 6
 T176_L0_f12_le: x_BsmtFin_SF_1 + 19.50000100000011 t176_leaf0 <= 1984.5
 T176_L0_f29_le: x_Fireplaces + 1.000001 t176_leaf0 <= 3
 T176_L1_f23_le: x_Full_Bath + 3.000001 t176_leaf1 <= 6
 T176_L1_f12_le: x_BsmtFin_SF_1 + 19.50000100000011 t176_leaf1 <= 1984.5
 T176_R1_f29_ge: x_Fireplaces - 2 t176_leaf1 >= 0
 T176_L2_f23_le: x_Full_Bath + 3.000001 t176_leaf2 <= 6
 T176_R2_f12_ge: x_BsmtFin_SF_1 - 1965 t176_leaf2 >= 0
 T176_R3_f23_ge: x_Full_Bath - 3 t176_leaf3 >= 0
 T176_L3_f26_le: x_Kitchen_AbvGr + 1.000001 t176_leaf3 <= 3
 T176_R4_f23_ge: x_Full_Bath - 3 t176_leaf4 >= 0
 T176_R4_f26_ge: x_Kitchen_AbvGr - 2 t176_leaf4 >= 0
 TREE_177_ONEHOT: t177_leaf0 + t177_leaf1 + t177_leaf2 + t177_leaf3
   + t177_leaf4 + t177_leaf5 + t177_leaf6 = 1
 T177_L0_f16_le: x_Heating_QC + 1.000001 t177_leaf0 <= 4
 T177_L0_f5_le: x_Year_Built + 998075.000001 t177_leaf0 <= 1e+06
 T177_L0_f25_le: x_Bedroom_AbvGr + 4.000001 t177_leaf0 <= 7
 T177_L1_f16_le: x_Heating_QC + 1.000001 t177_leaf1 <= 4
 T177_L1_f5_le: x_Year_Built + 998075.000001 t177_leaf1 <= 1e+06
 T177_R1_f25_ge: x_Bedroom_AbvGr - 3 t177_leaf1 >= 0
 T177_L2_f16_le: x_Heating_QC + 1.000001 t177_leaf2 <= 4
 T177_R2_f5_ge: x_Year_Built - 1925 t177_leaf2 >= 0
 T177_L2_f134_le: x_Bldg_Type_1Fam + 1e-06 t177_leaf2 <= 1
 T177_L3_f16_le: x_Heating_QC + 1.000001 t177_leaf3 <= 4
 T177_R3_f5_ge: x_Year_Built - 1925 t177_leaf3 >= 0
 T177_R3_f134_ge: x_Bldg_Type_1Fam - t177_leaf3 >= 0
 T177_R4_f16_ge: x_Heating_QC - t177_leaf4 >= 2
 T177_L4_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t177_leaf4 <= 2514.48
 T177_L4_f117_le: x_Condition_1_Artery + 1e-06 t177_leaf4 <= 1
 T177_R5_f16_ge: x_Heating_QC - t177_leaf5 >= 2
 T177_L5_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t177_leaf5 <= 2514.48
 T177_R5_f117_ge: x_Condition_1_Artery - t177_leaf5 >= 0
 T177_R6_f16_ge: x_Heating_QC - t177_leaf6 >= 2
 T177_R6_f15_ge: x_Total_Bsmt_SF - 2223 t177_leaf6 >= 0
 TREE_178_ONEHOT: t178_leaf0 + t178_leaf1 + t178_leaf2 + t178_leaf3
   + t178_leaf4 + t178_leaf5 + t178_leaf6 = 1
 T178_L0_f36_le: x_Wood_Deck_SF + 309.900001 t178_leaf0 <= 525.9
 T178_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t178_leaf0 <= 1
 T178_L0_f3_le: x_Overall_Qual + 999991.000001 t178_leaf0 <= 1e+06
 T178_L1_f36_le: x_Wood_Deck_SF + 309.900001 t178_leaf1 <= 525.9
 T178_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t178_leaf1 <= 1
 T178_R1_f3_ge: x_Overall_Qual - 9 t178_leaf1 >= 0
 T178_L2_f36_le: x_Wood_Deck_SF + 309.900001 t178_leaf2 <= 525.9
 T178_R2_f95_ge: x_Neighborhood_Crawfor - t178_leaf2 >= 0
 T178_L2_f1_le: x_Lot_Area <= 9554.999999
 T178_L3_f36_le: x_Wood_Deck_SF + 309.900001 t178_leaf3 <= 525.9
 T178_R3_f95_ge: x_Neighborhood_Crawfor - t178_leaf3 >= 0
 T178_R3_f1_ge: x_Lot_Area - 2555 t178_leaf3 >= 7000
 T178_R4_f36_ge: x_Wood_Deck_SF - 216 t178_leaf4 >= 0
 T178_L4_f297_le: x_Sale_Condition_Normal + 1e-06 t178_leaf4 <= 1
 T178_R5_f36_ge: x_Wood_Deck_SF - 216 t178_leaf5 >= 0
 T178_R5_f297_ge: x_Sale_Condition_Normal - t178_leaf5 >= 0
 T178_L5_f31_le: x_Garage_Yr_Blt + 998002.000001 t178_leaf5 <= 1e+06
 T178_R6_f36_ge: x_Wood_Deck_SF - 216 t178_leaf6 >= 0
 T178_R6_f297_ge: x_Sale_Condition_Normal - t178_leaf6 >= 0
 T178_R6_f31_ge: x_Garage_Yr_Blt - 1998 t178_leaf6 >= 0
 TREE_179_ONEHOT: t179_leaf0 + t179_leaf1 + t179_leaf2 + t179_leaf3
   + t179_leaf4 + t179_leaf5 = 1
 T179_L0_f40_le: x_Screen_Porch + 120.270001 t179_leaf0 <= 268.27
 T179_L0_f20_le: x_Gr_Liv_Area + 2653.760001 t179_leaf0 <= 3421.76
 T179_L0_f17_le: x_1st_Flr_SF + 1817.900001 t179_leaf0 <= 2534.9
 T179_L1_f40_le: x_Screen_Porch + 120.270001 t179_leaf1 <= 268.27
 T179_L1_f20_le: x_Gr_Liv_Area + 2653.760001 t179_leaf1 <= 3421.76
 T179_R1_f17_ge: x_1st_Flr_SF - 717 t179_leaf1 >= 0
 T179_L2_f40_le: x_Screen_Porch + 120.270001 t179_leaf2 <= 268.27
 T179_R2_f20_ge: x_Gr_Liv_Area - 768 t179_leaf2 >= 0
 T179_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t179_leaf2 <= 1
 T179_L3_f40_le: x_Screen_Porch + 120.270001 t179_leaf3 <= 268.27
 T179_R3_f20_ge: x_Gr_Liv_Area - 768 t179_leaf3 >= 0
 T179_R3_f95_ge: x_Neighborhood_Crawfor - t179_leaf3 >= 0
 T179_R4_f40_ge: x_Screen_Porch - 148 t179_leaf4 >= 0
 T179_L4_f20_le: x_Gr_Liv_Area + 1672.760001 t179_leaf4 <= 3421.76
 T179_R5_f40_ge: x_Screen_Porch - 148 t179_leaf5 >= 0
 T179_R5_f20_ge: x_Gr_Liv_Area - 1749 t179_leaf5 >= 0
 TREE_180_ONEHOT: t180_leaf0 + t180_leaf1 + t180_leaf2 + t180_leaf3
   + t180_leaf4 = 1
 T180_L0_f40_le: x_Screen_Porch + 126.270001 t180_leaf0 <= 268.27
 T180_L0_f11_le: x_Bsmt_Cond + 2.000001 t180_leaf0 <= 4
 T180_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t180_leaf0 <= 1
 T180_L1_f40_le: x_Screen_Porch + 126.270001 t180_leaf1 <= 268.27
 T180_L1_f11_le: x_Bsmt_Cond + 2.000001 t180_leaf1 <= 4
 T180_R1_f293_ge: x_Sale_Condition_Abnorml - t180_leaf1 >= 0
 T180_L2_f40_le: x_Screen_Porch + 126.270001 t180_leaf2 <= 268.27
 T180_R2_f11_ge: x_Bsmt_Cond - 3 t180_leaf2 >= -1
 T180_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t180_leaf2 <= 1
 T180_L3_f40_le: x_Screen_Porch + 126.270001 t180_leaf3 <= 268.27
 T180_R3_f11_ge: x_Bsmt_Cond - 3 t180_leaf3 >= -1
 T180_R3_f95_ge: x_Neighborhood_Crawfor - t180_leaf3 >= 0
 T180_R4_f40_ge: x_Screen_Porch - 142 t180_leaf4 >= 0
 TREE_181_ONEHOT: t181_leaf0 + t181_leaf1 + t181_leaf2 + t181_leaf3
   + t181_leaf4 + t181_leaf5 + t181_leaf6 + t181_leaf7 = 1
 T181_L0_f6_le: x_Year_Remod/Add + 998007.000001 t181_leaf0 <= 1e+06
 T181_L0_f14_le: x_Bsmt_Unf_SF + 1383.680001 t181_leaf0 <= 1812.68
 T181_L0_f31_le: x_Garage_Yr_Blt + 998074.000001 t181_leaf0 <= 1e+06
 T181_L1_f6_le: x_Year_Remod/Add + 998007.000001 t181_leaf1 <= 1e+06
 T181_L1_f14_le: x_Bsmt_Unf_SF + 1383.680001 t181_leaf1 <= 1812.68
 T181_R1_f31_ge: x_Garage_Yr_Blt - 1926 t181_leaf1 >= 0
 T181_L2_f6_le: x_Year_Remod/Add + 998007.000001 t181_leaf2 <= 1e+06
 T181_R2_f14_ge: x_Bsmt_Unf_SF - 429 t181_leaf2 >= 0
 T181_L2_f33_le: x_Garage_Area + 676.750001 t181_leaf2 <= 1163.75
 T181_L3_f6_le: x_Year_Remod/Add + 998007.000001 t181_leaf3 <= 1e+06
 T181_R3_f14_ge: x_Bsmt_Unf_SF - 429 t181_leaf3 >= 0
 T181_R3_f33_ge: x_Garage_Area - 487 t181_leaf3 >= 0
 T181_R4_f6_ge: x_Year_Remod/Add - 1993 t181_leaf4 >= 0
 T181_L4_f17_le: x_1st_Flr_SF + 465.900001 t181_leaf4 <= 2534.9
 T181_L4_f76_le: x_Lot_Shape_Reg + 1e-06 t181_leaf4 <= 1
 T181_R5_f6_ge: x_Year_Remod/Add - 1993 t181_leaf5 >= 0
 T181_L5_f17_le: x_1st_Flr_SF + 465.900001 t181_leaf5 <= 2534.9
 T181_R5_f76_ge: x_Lot_Shape_Reg - t181_leaf5 >= 0
 T181_R6_f6_ge: x_Year_Remod/Add - 1993 t181_leaf6 >= 0
 T181_R6_f17_ge: x_1st_Flr_SF - 2069 t181_leaf6 >= 0
 T181_L6_f0_le: x_Lot_Frontage + 67.530001 t181_leaf6 <= 175.53
 T181_R7_f6_ge: x_Year_Remod/Add - 1993 t181_leaf7 >= 0
 T181_R7_f17_ge: x_1st_Flr_SF - 2069 t181_leaf7 >= 0
 T181_R7_f0_ge: x_Lot_Frontage - 108 t181_leaf7 >= 0
 TREE_182_ONEHOT: t182_leaf0 + t182_leaf1 + t182_leaf2 + t182_leaf3
   + t182_leaf4 + t182_leaf5 + t182_leaf6 = 1
 T182_L0_f1_le: x_Lot_Area <= 14679.999999
 T182_L0_f20_le: x_Gr_Liv_Area + 2527.760001 t182_leaf0 <= 3421.76
 T182_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t182_leaf0 <= 1
 T182_L1_f1_le: x_Lot_Area <= 14679.999999
 T182_L1_f20_le: x_Gr_Liv_Area + 2527.760001 t182_leaf1 <= 3421.76
 T182_R1_f293_ge: x_Sale_Condition_Abnorml - t182_leaf1 >= 0
 T182_L2_f1_le: x_Lot_Area <= 14679.999999
 T182_R2_f20_ge: x_Gr_Liv_Area - 894 t182_leaf2 >= 0
 T182_L2_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t182_leaf2 <= 1
 T182_L3_f1_le: x_Lot_Area <= 14679.999999
 T182_R3_f20_ge: x_Gr_Liv_Area - 894 t182_leaf3 >= 0
 T182_R3_f207_ge: x_Bsmt_Exposure_Gd - t182_leaf3 >= 0
 T182_R4_f1_ge: x_Lot_Area - 7680 t182_leaf4 >= 7000
 T182_L4_f293_le: x_Sale_Condition_Abnorml + 1e-06 t182_leaf4 <= 1
 T182_L4_f25_le: x_Bedroom_AbvGr + 3.000001 t182_leaf4 <= 7
 T182_R5_f1_ge: x_Lot_Area - 7680 t182_leaf5 >= 7000
 T182_L5_f293_le: x_Sale_Condition_Abnorml + 1e-06 t182_leaf5 <= 1
 T182_R5_f25_ge: x_Bedroom_AbvGr - 4 t182_leaf5 >= 0
 T182_R6_f1_ge: x_Lot_Area - 7680 t182_leaf6 >= 7000
 T182_R6_f293_ge: x_Sale_Condition_Abnorml - t182_leaf6 >= 0
 TREE_183_ONEHOT: t183_leaf0 + t183_leaf1 + t183_leaf2 + t183_leaf3
   + t183_leaf4 + t183_leaf5 + t183_leaf6 = 1
 T183_L0_f40_le: x_Screen_Porch + 130.270001 t183_leaf0 <= 268.27
 T183_L0_f44_le: x_Yr_Sold + 997993.000001 t183_leaf0 <= 1e+06
 T183_L0_f1_le: x_Lot_Area <= 7699.999999
 T183_L1_f40_le: x_Screen_Porch + 130.270001 t183_leaf1 <= 268.27
 T183_L1_f44_le: x_Yr_Sold + 997993.000001 t183_leaf1 <= 1e+06
 T183_R1_f1_ge: x_Lot_Area - 700 t183_leaf1 >= 7000
 T183_L2_f40_le: x_Screen_Porch + 130.270001 t183_leaf2 <= 268.27
 T183_R2_f44_ge: x_Yr_Sold - 2007 t183_leaf2 >= 0
 T183_L2_f1_le: x_Lot_Area <= 23256.999999
 T183_L3_f40_le: x_Screen_Porch + 130.270001 t183_leaf3 <= 268.27
 T183_R3_f44_ge: x_Yr_Sold - 2007 t183_leaf3 >= 0
 T183_R3_f1_ge: x_Lot_Area - 16257 t183_leaf3 >= 7000
 T183_R4_f40_ge: x_Screen_Porch - 138 t183_leaf4 >= 0
 T183_L4_f1_le: x_Lot_Area + 1638.000001 t183_leaf4 <= 7000
 T183_R5_f40_ge: x_Screen_Porch - 138 t183_leaf5 >= 0
 T183_R5_f1_ge: x_Lot_Area >= 5362
 T183_L5_f273_le: x_Mo_Sold_3 + 1e-06 t183_leaf5 <= 1
 T183_R6_f40_ge: x_Screen_Porch - 138 t183_leaf6 >= 0
 T183_R6_f1_ge: x_Lot_Area >= 5362
 T183_R6_f273_ge: x_Mo_Sold_3 - t183_leaf6 >= 0
 TREE_184_ONEHOT: t184_leaf0 + t184_leaf1 + t184_leaf2 + t184_leaf3
   + t184_leaf4 + t184_leaf5 + t184_leaf6 + t184_leaf7 = 1
 T184_L0_f4_le: x_Overall_Cond + 999996.000001 t184_leaf0 <= 1e+06
 T184_L0_f6_le: x_Year_Remod/Add + 998042.000001 t184_leaf0 <= 1e+06
 T184_L0_f231_le: x_Central_Air_N + 1e-06 t184_leaf0 <= 1
 T184_L1_f4_le: x_Overall_Cond + 999996.000001 t184_leaf1 <= 1e+06
 T184_L1_f6_le: x_Year_Remod/Add + 998042.000001 t184_leaf1 <= 1e+06
 T184_R1_f231_ge: x_Central_Air_N - t184_leaf1 >= 0
 T184_L2_f4_le: x_Overall_Cond + 999996.000001 t184_leaf2 <= 1e+06
 T184_R2_f6_ge: x_Year_Remod/Add - 1958 t184_leaf2 >= 0
 T184_L2_f25_le: x_Bedroom_AbvGr + 3.000001 t184_leaf2 <= 7
 T184_L3_f4_le: x_Overall_Cond + 999996.000001 t184_leaf3 <= 1e+06
 T184_R3_f6_ge: x_Year_Remod/Add - 1958 t184_leaf3 >= 0
 T184_R3_f25_ge: x_Bedroom_AbvGr - 4 t184_leaf3 >= 0
 T184_R4_f4_ge: x_Overall_Cond - 4 t184_leaf4 >= 0
 T184_L4_f33_le: x_Garage_Area + 292.750001 t184_leaf4 <= 1163.75
 T184_L4_f0_le: x_Lot_Frontage + 65.530001 t184_leaf4 <= 175.53
 T184_R5_f4_ge: x_Overall_Cond - 4 t184_leaf5 >= 0
 T184_L5_f33_le: x_Garage_Area + 292.750001 t184_leaf5 <= 1163.75
 T184_R5_f0_ge: x_Lot_Frontage - 110 t184_leaf5 >= 0
 T184_R6_f4_ge: x_Overall_Cond - 4 t184_leaf6 >= 0
 T184_R6_f33_ge: x_Garage_Area - 871 t184_leaf6 >= 0
 T184_L6_f0_le: x_Lot_Frontage + 61.530001 t184_leaf6 <= 175.53
 T184_R7_f4_ge: x_Overall_Cond - 4 t184_leaf7 >= 0
 T184_R7_f33_ge: x_Garage_Area - 871 t184_leaf7 >= 0
 T184_R7_f0_ge: x_Lot_Frontage - 114 t184_leaf7 >= 0
 TREE_185_ONEHOT: t185_leaf0 + t185_leaf1 + t185_leaf2 + t185_leaf3
   + t185_leaf4 + t185_leaf5 = 1
 T185_L0_f1_le: x_Lot_Area <= 17226.999999
 T185_L0_f20_le: x_Gr_Liv_Area + 772.7600010000001 t185_leaf0 <= 3421.76
 T185_L0_f3_le: x_Overall_Qual + 999993.000001 t185_leaf0 <= 1e+06
 T185_L1_f1_le: x_Lot_Area <= 17226.999999
 T185_L1_f20_le: x_Gr_Liv_Area + 772.7600010000001 t185_leaf1 <= 3421.76
 T185_R1_f3_ge: x_Overall_Qual - 7 t185_leaf1 >= 0
 T185_L2_f1_le: x_Lot_Area <= 17226.999999
 T185_R2_f20_ge: x_Gr_Liv_Area - 2649 t185_leaf2 >= 0
 T185_R3_f1_ge: x_Lot_Area - 10227 t185_leaf3 >= 7000
 T185_L3_f17_le: x_1st_Flr_SF + 169.900001 t185_leaf3 <= 2534.9
 T185_L3_f20_le: x_Gr_Liv_Area + 1930.760001 t185_leaf3 <= 3421.76
 T185_R4_f1_ge: x_Lot_Area - 10227 t185_leaf4 >= 7000
 T185_L4_f17_le: x_1st_Flr_SF + 169.900001 t185_leaf4 <= 2534.9
 T185_R4_f20_ge: x_Gr_Liv_Area - 1491 t185_leaf4 >= 0
 T185_R5_f1_ge: x_Lot_Area - 10227 t185_leaf5 >= 7000
 T185_R5_f17_ge: x_1st_Flr_SF - 2365 t185_leaf5 >= 0
 TREE_186_ONEHOT: t186_leaf0 + t186_leaf1 + t186_leaf2 + t186_leaf3
   + t186_leaf4 = 1
 T186_L0_f289_le: x_Sale_Type_New + 1e-06 t186_leaf0 <= 1
 T186_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t186_leaf0 <= 1
 T186_L0_f119_le: x_Condition_1_Norm + 1e-06 t186_leaf0 <= 1
 T186_L1_f289_le: x_Sale_Type_New + 1e-06 t186_leaf1 <= 1
 T186_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t186_leaf1 <= 1
 T186_R1_f119_ge: x_Condition_1_Norm - t186_leaf1 >= 0
 T186_L2_f289_le: x_Sale_Type_New + 1e-06 t186_leaf2 <= 1
 T186_R2_f293_ge: x_Sale_Condition_Abnorml - t186_leaf2 >= 0
 T186_L2_f35_le: x_Garage_Cond + 2.000001 t186_leaf2 <= 4
 T186_L3_f289_le: x_Sale_Type_New + 1e-06 t186_leaf3 <= 1
 T186_R3_f293_ge: x_Sale_Condition_Abnorml - t186_leaf3 >= 0
 T186_R3_f35_ge: x_Garage_Cond - 3 t186_leaf3 >= -1
 T186_R4_f289_ge: x_Sale_Type_New - t186_leaf4 >= 0
 TREE_187_ONEHOT: t187_leaf0 + t187_leaf1 + t187_leaf2 + t187_leaf3
   + t187_leaf4 = 1
 T187_L0_f18_le: x_2nd_Flr_SF + 12.54000100000007 t187_leaf0 <= 1641.54
 T187_L0_f4_le: x_Overall_Cond + 999994.000001 t187_leaf0 <= 1e+06
 T187_L0_f31_le: x_Garage_Yr_Blt + 998038.000001 t187_leaf0 <= 1e+06
 T187_L1_f18_le: x_2nd_Flr_SF + 12.54000100000007 t187_leaf1 <= 1641.54
 T187_L1_f4_le: x_Overall_Cond + 999994.000001 t187_leaf1 <= 1e+06
 T187_R1_f31_ge: x_Garage_Yr_Blt - 1962 t187_leaf1 >= 0
 T187_L2_f18_le: x_2nd_Flr_SF + 12.54000100000007 t187_leaf2 <= 1641.54
 T187_R2_f4_ge: x_Overall_Cond - 6 t187_leaf2 >= 0
 T187_L2_f20_le: x_Gr_Liv_Area + 2153.760001000001 t187_leaf2 <= 3421.76
 T187_L3_f18_le: x_2nd_Flr_SF + 12.54000100000007 t187_leaf3 <= 1641.54
 T187_R3_f4_ge: x_Overall_Cond - 6 t187_leaf3 >= 0
 T187_R3_f20_ge: x_Gr_Liv_Area - 1268 t187_leaf3 >= 0
 T187_R4_f18_ge: x_2nd_Flr_SF - 1629 t187_leaf4 >= 0
 TREE_188_ONEHOT: t188_leaf0 + t188_leaf1 + t188_leaf2 + t188_leaf3
   + t188_leaf4 + t188_leaf5 + t188_leaf6 = 1
 T188_L0_f1_le: x_Lot_Area <= 14827.999999
 T188_L0_f5_le: x_Year_Built + 998089.000001 t188_leaf0 <= 1e+06
 T188_L0_f36_le: x_Wood_Deck_SF + 402.900001 t188_leaf0 <= 525.9
 T188_L1_f1_le: x_Lot_Area <= 14827.999999
 T188_L1_f5_le: x_Year_Built + 998089.000001 t188_leaf1 <= 1e+06
 T188_R1_f36_ge: x_Wood_Deck_SF - 123 t188_leaf1 >= 0
 T188_L2_f1_le: x_Lot_Area <= 14827.999999
 T188_R2_f5_ge: x_Year_Built - 1911 t188_leaf2 >= 0
 T188_L2_f78_le: x_Land_Contour_HLS + 1e-06 t188_leaf2 <= 1
 T188_L3_f1_le: x_Lot_Area <= 14827.999999
 T188_R3_f5_ge: x_Year_Built - 1911 t188_leaf3 >= 0
 T188_R3_f78_ge: x_Land_Contour_HLS - t188_leaf3 >= 0
 T188_R4_f1_ge: x_Lot_Area - 7828 t188_leaf4 >= 7000
 T188_L4_f37_le: x_Open_Porch_SF + 223.810001 t188_leaf4 <= 488.81
 T188_L4_f29_le: x_Fireplaces + 1.000001 t188_leaf4 <= 3
 T188_R5_f1_ge: x_Lot_Area - 7828 t188_leaf5 >= 7000
 T188_L5_f37_le: x_Open_Porch_SF + 223.810001 t188_leaf5 <= 488.81
 T188_R5_f29_ge: x_Fireplaces - 2 t188_leaf5 >= 0
 T188_R6_f1_ge: x_Lot_Area - 7828 t188_leaf6 >= 7000
 T188_R6_f37_ge: x_Open_Porch_SF - 265 t188_leaf6 >= 0
 TREE_189_ONEHOT: t189_leaf0 + t189_leaf1 + t189_leaf2 + t189_leaf3
   + t189_leaf4 + t189_leaf5 = 1
 T189_L0_f62_le: x_MS_Zoning_C_(all) + 1e-06 t189_leaf0 <= 1
 T189_L0_f29_le: x_Fireplaces + 1.000001 t189_leaf0 <= 3
 T189_L0_f6_le: x_Year_Remod/Add + 997992.000001 t189_leaf0 <= 1e+06
 T189_L1_f62_le: x_MS_Zoning_C_(all) + 1e-06 t189_leaf1 <= 1
 T189_L1_f29_le: x_Fireplaces + 1.000001 t189_leaf1 <= 3
 T189_R1_f6_ge: x_Year_Remod/Add - 2008 t189_leaf1 >= 0
 T189_L2_f62_le: x_MS_Zoning_C_(all) + 1e-06 t189_leaf2 <= 1
 T189_R2_f29_ge: x_Fireplaces - 2 t189_leaf2 >= 0
 T189_L2_f86_le: x_Land_Slope_Gtl + 1e-06 t189_leaf2 <= 1
 T189_L3_f62_le: x_MS_Zoning_C_(all) + 1e-06 t189_leaf3 <= 1
 T189_R3_f29_ge: x_Fireplaces - 2 t189_leaf3 >= 0
 T189_R3_f86_ge: x_Land_Slope_Gtl - t189_leaf3 >= 0
 T189_R4_f62_ge: x_MS_Zoning_C_(all) - t189_leaf4 >= 0
 T189_L4_f4_le: x_Overall_Cond + 999996.000001 t189_leaf4 <= 1e+06
 T189_R5_f62_ge: x_MS_Zoning_C_(all) - t189_leaf5 >= 0
 T189_R5_f4_ge: x_Overall_Cond - 4 t189_leaf5 >= 0
 TREE_190_ONEHOT: t190_leaf0 + t190_leaf1 + t190_leaf2 + t190_leaf3
   + t190_leaf4 + t190_leaf5 = 1
 T190_L0_f17_le: x_1st_Flr_SF + 465.900001 t190_leaf0 <= 2534.9
 T190_L0_f40_le: x_Screen_Porch + 120.270001 t190_leaf0 <= 268.27
 T190_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t190_leaf0 <= 1
 T190_L1_f17_le: x_1st_Flr_SF + 465.900001 t190_leaf1 <= 2534.9
 T190_L1_f40_le: x_Screen_Porch + 120.270001 t190_leaf1 <= 268.27
 T190_R1_f95_ge: x_Neighborhood_Crawfor - t190_leaf1 >= 0
 T190_L2_f17_le: x_1st_Flr_SF + 465.900001 t190_leaf2 <= 2534.9
 T190_R2_f40_ge: x_Screen_Porch - 148 t190_leaf2 >= 0
 T190_L2_f0_le: x_Lot_Frontage + 121.530001 t190_leaf2 <= 175.53
 T190_L3_f17_le: x_1st_Flr_SF + 465.900001 t190_leaf3 <= 2534.9
 T190_R3_f40_ge: x_Screen_Porch - 148 t190_leaf3 >= 0
 T190_R3_f0_ge: x_Lot_Frontage - 54 t190_leaf3 >= 0
 T190_R4_f17_ge: x_1st_Flr_SF - 2069 t190_leaf4 >= 0
 T190_L4_f6_le: x_Year_Remod/Add + 998018.000001 t190_leaf4 <= 1e+06
 T190_R5_f17_ge: x_1st_Flr_SF - 2069 t190_leaf5 >= 0
 T190_R5_f6_ge: x_Year_Remod/Add - 1982 t190_leaf5 >= 0
 TREE_191_ONEHOT: t191_leaf0 + t191_leaf1 + t191_leaf2 + t191_leaf3
   + t191_leaf4 + t191_leaf5 = 1
 T191_L0_f37_le: x_Open_Porch_SF + 232.810001 t191_leaf0 <= 488.81
 T191_L0_f40_le: x_Screen_Porch + 106.270001 t191_leaf0 <= 268.27
 T191_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t191_leaf0 <= 1
 T191_L1_f37_le: x_Open_Porch_SF + 232.810001 t191_leaf1 <= 488.81
 T191_L1_f40_le: x_Screen_Porch + 106.270001 t191_leaf1 <= 268.27
 T191_R1_f245_ge: x_FunctioNo_aplical_Typ - t191_leaf1 >= 0
 T191_L2_f37_le: x_Open_Porch_SF + 232.810001 t191_leaf2 <= 488.81
 T191_R2_f40_ge: x_Screen_Porch - 162 t191_leaf2 >= 0
 T191_R3_f37_ge: x_Open_Porch_SF - 256 t191_leaf3 >= 0
 T191_L3_f1_le: x_Lot_Area <= 18385.999999
 T191_L3_f31_le: x_Garage_Yr_Blt + 998047.000001 t191_leaf3 <= 1e+06
 T191_R4_f37_ge: x_Open_Porch_SF - 256 t191_leaf4 >= 0
 T191_L4_f1_le: x_Lot_Area <= 18385.999999
 T191_R4_f31_ge: x_Garage_Yr_Blt - 1953 t191_leaf4 >= 0
 T191_R5_f37_ge: x_Open_Porch_SF - 256 t191_leaf5 >= 0
 T191_R5_f1_ge: x_Lot_Area - 11386 t191_leaf5 >= 7000
 TREE_192_ONEHOT: t192_leaf0 + t192_leaf1 + t192_leaf2 + t192_leaf3
   + t192_leaf4 + t192_leaf5 = 1
 T192_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t192_leaf0 <= 1
 T192_L0_f20_le: x_Gr_Liv_Area + 2538.760001 t192_leaf0 <= 3421.76
 T192_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t192_leaf0 <= 1
 T192_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t192_leaf1 <= 1
 T192_L1_f20_le: x_Gr_Liv_Area + 2538.760001 t192_leaf1 <= 3421.76
 T192_R1_f297_ge: x_Sale_Condition_Normal - t192_leaf1 >= 0
 T192_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t192_leaf2 <= 1
 T192_R2_f20_ge: x_Gr_Liv_Area - 883 t192_leaf2 >= 0
 T192_L2_f3_le: x_Overall_Qual + 999991.000001 t192_leaf2 <= 1e+06
 T192_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t192_leaf3 <= 1
 T192_R3_f20_ge: x_Gr_Liv_Area - 883 t192_leaf3 >= 0
 T192_R3_f3_ge: x_Overall_Qual - 9 t192_leaf3 >= 0
 T192_R4_f95_ge: x_Neighborhood_Crawfor - t192_leaf4 >= 0
 T192_L4_f200_le: x_Foundation_BrkTil + 1e-06 t192_leaf4 <= 1
 T192_R5_f95_ge: x_Neighborhood_Crawfor - t192_leaf5 >= 0
 T192_R5_f200_ge: x_Foundation_BrkTil - t192_leaf5 >= 0
 TREE_193_ONEHOT: t193_leaf0 + t193_leaf1 + t193_leaf2 + t193_leaf3
   + t193_leaf4 + t193_leaf5 = 1
 T193_L0_f1_le: x_Lot_Area <= 19521.999999
 T193_L0_f210_le: x_Bsmt_Exposure_No_aplica + 1e-06 t193_leaf0 <= 1
 T193_L0_f1_le: x_Lot_Area + 1024.000001 t193_leaf0 <= 7000
 T193_L1_f1_le: x_Lot_Area <= 19521.999999
 T193_L1_f210_le: x_Bsmt_Exposure_No_aplica + 1e-06 t193_leaf1 <= 1
 T193_R1_f1_ge: x_Lot_Area >= 5976
 T193_L2_f1_le: x_Lot_Area <= 19521.999999
 T193_R2_f210_ge: x_Bsmt_Exposure_No_aplica - t193_leaf2 >= 0
 T193_L2_f192_le: x_Exterior_2nd_Wd_Sdng + 1e-06 t193_leaf2 <= 1
 T193_L3_f1_le: x_Lot_Area <= 19521.999999
 T193_R3_f210_ge: x_Bsmt_Exposure_No_aplica - t193_leaf3 >= 0
 T193_R3_f192_ge: x_Exterior_2nd_Wd_Sdng - t193_leaf3 >= 0
 T193_R4_f1_ge: x_Lot_Area - 12522 t193_leaf4 >= 7000
 T193_L4_f293_le: x_Sale_Condition_Abnorml + 1e-06 t193_leaf4 <= 1
 T193_R5_f1_ge: x_Lot_Area - 12522 t193_leaf5 >= 7000
 T193_R5_f293_ge: x_Sale_Condition_Abnorml - t193_leaf5 >= 0
 TREE_194_ONEHOT: t194_leaf0 + t194_leaf1 + t194_leaf2 + t194_leaf3
   + t194_leaf4 + t194_leaf5 = 1
 T194_L0_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t194_leaf0 <= 2514.48
 T194_L0_f4_le: x_Overall_Cond + 999994.000001 t194_leaf0 <= 1e+06
 T194_L0_f12_le: x_BsmtFin_SF_1 + 1570.500001 t194_leaf0 <= 1984.5
 T194_L1_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t194_leaf1 <= 2514.48
 T194_L1_f4_le: x_Overall_Cond + 999994.000001 t194_leaf1 <= 1e+06
 T194_R1_f12_ge: x_BsmtFin_SF_1 - 414 t194_leaf1 >= 0
 T194_L2_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t194_leaf2 <= 2514.48
 T194_R2_f4_ge: x_Overall_Cond - 6 t194_leaf2 >= 0
 T194_L2_f17_le: x_1st_Flr_SF + 1214.900001 t194_leaf2 <= 2534.9
 T194_L3_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t194_leaf3 <= 2514.48
 T194_R3_f4_ge: x_Overall_Cond - 6 t194_leaf3 >= 0
 T194_R3_f17_ge: x_1st_Flr_SF - 1320 t194_leaf3 >= 0
 T194_R4_f15_ge: x_Total_Bsmt_SF - 2136 t194_leaf4 >= 0
 T194_L4_f12_le: x_BsmtFin_SF_1 + 874.5000010000001 t194_leaf4 <= 1984.5
 T194_R5_f15_ge: x_Total_Bsmt_SF - 2136 t194_leaf5 >= 0
 T194_R5_f12_ge: x_BsmtFin_SF_1 - 1110 t194_leaf5 >= 0
 TREE_195_ONEHOT: t195_leaf0 + t195_leaf1 + t195_leaf2 + t195_leaf3
   + t195_leaf4 = 1
 T195_L0_f18_le: x_2nd_Flr_SF + 118.5400010000001 t195_leaf0 <= 1641.54
 T195_L0_f3_le: x_Overall_Qual + 999993.000001 t195_leaf0 <= 1e+06
 T195_L0_f18_le: x_2nd_Flr_SF + 1057.540001 t195_leaf0 <= 1641.54
 T195_L1_f18_le: x_2nd_Flr_SF + 118.5400010000001 t195_leaf1 <= 1641.54
 T195_L1_f3_le: x_Overall_Qual + 999993.000001 t195_leaf1 <= 1e+06
 T195_R1_f18_ge: x_2nd_Flr_SF - 584 t195_leaf1 >= 0
 T195_L2_f18_le: x_2nd_Flr_SF + 118.5400010000001 t195_leaf2 <= 1641.54
 T195_R2_f3_ge: x_Overall_Qual - 7 t195_leaf2 >= 0
 T195_L2_f289_le: x_Sale_Type_New + 1e-06 t195_leaf2 <= 1
 T195_L3_f18_le: x_2nd_Flr_SF + 118.5400010000001 t195_leaf3 <= 1641.54
 T195_R3_f3_ge: x_Overall_Qual - 7 t195_leaf3 >= 0
 T195_R3_f289_ge: x_Sale_Type_New - t195_leaf3 >= 0
 T195_R4_f18_ge: x_2nd_Flr_SF - 1523 t195_leaf4 >= 0
 TREE_196_ONEHOT: t196_leaf0 + t196_leaf1 + t196_leaf2 + t196_leaf3
   + t196_leaf4 + t196_leaf5 = 1
 T196_L0_f23_le: x_Full_Bath + 3.000001 t196_leaf0 <= 6
 T196_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t196_leaf0 <= 1
 T196_L0_f10_le: x_Bsmt_Qual + 4.000001 t196_leaf0 <= 4
 T196_L1_f23_le: x_Full_Bath + 3.000001 t196_leaf1 <= 6
 T196_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t196_leaf1 <= 1
 T196_R1_f10_ge: x_Bsmt_Qual >= 0
 T196_L2_f23_le: x_Full_Bath + 3.000001 t196_leaf2 <= 6
 T196_R2_f164_ge: x_Exterior_1st_BrkFace - t196_leaf2 >= 0
 T196_L2_f217_le: x_BsmtFin_Type_1_Unf + 1e-06 t196_leaf2 <= 1
 T196_L3_f23_le: x_Full_Bath + 3.000001 t196_leaf3 <= 6
 T196_R3_f164_ge: x_Exterior_1st_BrkFace - t196_leaf3 >= 0
 T196_R3_f217_ge: x_BsmtFin_Type_1_Unf - t196_leaf3 >= 0
 T196_R4_f23_ge: x_Full_Bath - 3 t196_leaf4 >= 0
 T196_L4_f26_le: x_Kitchen_AbvGr + 1.000001 t196_leaf4 <= 3
 T196_R5_f23_ge: x_Full_Bath - 3 t196_leaf5 >= 0
 T196_R5_f26_ge: x_Kitchen_AbvGr - 2 t196_leaf5 >= 0
 TREE_197_ONEHOT: t197_leaf0 + t197_leaf1 + t197_leaf2 + t197_leaf3
   + t197_leaf4 + t197_leaf5 = 1
 T197_L0_f40_le: x_Screen_Porch + 118.270001 t197_leaf0 <= 268.27
 T197_L0_f37_le: x_Open_Porch_SF + 210.810001 t197_leaf0 <= 488.81
 T197_L0_f3_le: x_Overall_Qual + 999990.000001 t197_leaf0 <= 1e+06
 T197_L1_f40_le: x_Screen_Porch + 118.270001 t197_leaf1 <= 268.27
 T197_L1_f37_le: x_Open_Porch_SF + 210.810001 t197_leaf1 <= 488.81
 T197_R1_f3_ge: x_Overall_Qual - 10 t197_leaf1 >= 0
 T197_L2_f40_le: x_Screen_Porch + 118.270001 t197_leaf2 <= 268.27
 T197_R2_f37_ge: x_Open_Porch_SF - 278 t197_leaf2 >= 0
 T197_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t197_leaf2 <= 1
 T197_L3_f40_le: x_Screen_Porch + 118.270001 t197_leaf3 <= 268.27
 T197_R3_f37_ge: x_Open_Porch_SF - 278 t197_leaf3 >= 0
 T197_R3_f297_ge: x_Sale_Condition_Normal - t197_leaf3 >= 0
 T197_R4_f40_ge: x_Screen_Porch - 150 t197_leaf4 >= 0
 T197_L4_f174_le: x_Exterior_1st_VinylSd + 1e-06 t197_leaf4 <= 1
 T197_R5_f40_ge: x_Screen_Porch - 150 t197_leaf5 >= 0
 T197_R5_f174_ge: x_Exterior_1st_VinylSd - t197_leaf5 >= 0
 TREE_198_ONEHOT: t198_leaf0 + t198_leaf1 + t198_leaf2 + t198_leaf3
   + t198_leaf4 + t198_leaf5 + t198_leaf6 = 1
 T198_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t198_leaf0 <= 1
 T198_L0_f4_le: x_Overall_Cond + 999996.000001 t198_leaf0 <= 1e+06
 T198_L0_f6_le: x_Year_Remod/Add + 998049.000001 t198_leaf0 <= 1e+06
 T198_L1_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t198_leaf1 <= 1
 T198_L1_f4_le: x_Overall_Cond + 999996.000001 t198_leaf1 <= 1e+06
 T198_R1_f6_ge: x_Year_Remod/Add - 1951 t198_leaf1 >= 0
 T198_L2_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t198_leaf2 <= 1
 T198_R2_f4_ge: x_Overall_Cond - 4 t198_leaf2 >= 0
 T198_L2_f1_le: x_Lot_Area + 3000.000001 t198_leaf2 <= 7000
 T198_L3_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t198_leaf3 <= 1
 T198_R3_f4_ge: x_Overall_Cond - 4 t198_leaf3 >= 0
 T198_R3_f1_ge: x_Lot_Area >= 4000
 T198_R4_f207_ge: x_Bsmt_Exposure_Gd - t198_leaf4 >= 0
 T198_L4_f6_le: x_Year_Remod/Add + 997991.000001 t198_leaf4 <= 1e+06
 T198_L4_f1_le: x_Lot_Area <= 17540.999999
 T198_R5_f207_ge: x_Bsmt_Exposure_Gd - t198_leaf5 >= 0
 T198_L5_f6_le: x_Year_Remod/Add + 997991.000001 t198_leaf5 <= 1e+06
 T198_R5_f1_ge: x_Lot_Area - 10541 t198_leaf5 >= 7000
 T198_R6_f207_ge: x_Bsmt_Exposure_Gd - t198_leaf6 >= 0
 T198_R6_f6_ge: x_Year_Remod/Add - 2009 t198_leaf6 >= 0
 TREE_199_ONEHOT: t199_leaf0 + t199_leaf1 + t199_leaf2 + t199_leaf3
   + t199_leaf4 + t199_leaf5 + t199_leaf6 + t199_leaf7 = 1
 T199_L0_f4_le: x_Overall_Cond + 999994.000001 t199_leaf0 <= 1e+06
 T199_L0_f10_le: x_Bsmt_Qual + 1.000001 t199_leaf0 <= 4
 T199_L0_f3_le: x_Overall_Qual + 999993.000001 t199_leaf0 <= 1e+06
 T199_L1_f4_le: x_Overall_Cond + 999994.000001 t199_leaf1 <= 1e+06
 T199_L1_f10_le: x_Bsmt_Qual + 1.000001 t199_leaf1 <= 4
 T199_R1_f3_ge: x_Overall_Qual - 7 t199_leaf1 >= 0
 T199_L2_f4_le: x_Overall_Cond + 999994.000001 t199_leaf2 <= 1e+06
 T199_R2_f10_ge: x_Bsmt_Qual - t199_leaf2 >= 2
 T199_L2_f3_le: x_Overall_Qual + 999993.000001 t199_leaf2 <= 1e+06
 T199_L3_f4_le: x_Overall_Cond + 999994.000001 t199_leaf3 <= 1e+06
 T199_R3_f10_ge: x_Bsmt_Qual - t199_leaf3 >= 2
 T199_R3_f3_ge: x_Overall_Qual - 7 t199_leaf3 >= 0
 T199_R4_f4_ge: x_Overall_Cond - 6 t199_leaf4 >= 0
 T199_L4_f119_le: x_Condition_1_Norm + 1e-06 t199_leaf4 <= 1
 T199_L4_f23_le: x_Full_Bath + 4.000001 t199_leaf4 <= 6
 T199_R5_f4_ge: x_Overall_Cond - 6 t199_leaf5 >= 0
 T199_L5_f119_le: x_Condition_1_Norm + 1e-06 t199_leaf5 <= 1
 T199_R5_f23_ge: x_Full_Bath - 2 t199_leaf5 >= 0
 T199_R6_f4_ge: x_Overall_Cond - 6 t199_leaf6 >= 0
 T199_R6_f119_ge: x_Condition_1_Norm - t199_leaf6 >= 0
 T199_L6_f17_le: x_1st_Flr_SF + 1888.900001 t199_leaf6 <= 2534.9
 T199_R7_f4_ge: x_Overall_Cond - 6 t199_leaf7 >= 0
 T199_R7_f119_ge: x_Condition_1_Norm - t199_leaf7 >= 0
 T199_R7_f17_ge: x_1st_Flr_SF - 646 t199_leaf7 >= 0
 TREE_200_ONEHOT: t200_leaf0 + t200_leaf1 + t200_leaf2 + t200_leaf3
   + t200_leaf4 + t200_leaf5 + t200_leaf6 = 1
 T200_L0_f44_le: x_Yr_Sold + 997992.000001 t200_leaf0 <= 1e+06
 T200_L0_f36_le: x_Wood_Deck_SF + 367.900001 t200_leaf0 <= 525.9
 T200_L0_f17_le: x_1st_Flr_SF + 1120.900001 t200_leaf0 <= 2534.9
 T200_L1_f44_le: x_Yr_Sold + 997992.000001 t200_leaf1 <= 1e+06
 T200_L1_f36_le: x_Wood_Deck_SF + 367.900001 t200_leaf1 <= 525.9
 T200_R1_f17_ge: x_1st_Flr_SF - 1414 t200_leaf1 >= 0
 T200_L2_f44_le: x_Yr_Sold + 997992.000001 t200_leaf2 <= 1e+06
 T200_R2_f36_ge: x_Wood_Deck_SF - 158 t200_leaf2 >= 0
 T200_L2_f6_le: x_Year_Remod/Add + 998012.000001 t200_leaf2 <= 1e+06
 T200_L3_f44_le: x_Yr_Sold + 997992.000001 t200_leaf3 <= 1e+06
 T200_R3_f36_ge: x_Wood_Deck_SF - 158 t200_leaf3 >= 0
 T200_R3_f6_ge: x_Year_Remod/Add - 1988 t200_leaf3 >= 0
 T200_R4_f44_ge: x_Yr_Sold - 2008 t200_leaf4 >= 0
 T200_L4_f3_le: x_Overall_Qual + 999997.000001 t200_leaf4 <= 1e+06
 T200_R5_f44_ge: x_Yr_Sold - 2008 t200_leaf5 >= 0
 T200_R5_f3_ge: x_Overall_Qual - 3 t200_leaf5 >= 0
 T200_L5_f0_le: x_Lot_Frontage + 104.530001 t200_leaf5 <= 175.53
 T200_R6_f44_ge: x_Yr_Sold - 2008 t200_leaf6 >= 0
 T200_R6_f3_ge: x_Overall_Qual - 3 t200_leaf6 >= 0
 T200_R6_f0_ge: x_Lot_Frontage - 71 t200_leaf6 >= 0
 TREE_201_ONEHOT: t201_leaf0 + t201_leaf1 + t201_leaf2 + t201_leaf3
   + t201_leaf4 + t201_leaf5 = 1
 T201_L0_f289_le: x_Sale_Type_New + 1e-06 t201_leaf0 <= 1
 T201_L0_f36_le: x_Wood_Deck_SF + 301.900001 t201_leaf0 <= 525.9
 T201_L0_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t201_leaf0 <= 1
 T201_L1_f289_le: x_Sale_Type_New + 1e-06 t201_leaf1 <= 1
 T201_L1_f36_le: x_Wood_Deck_SF + 301.900001 t201_leaf1 <= 525.9
 T201_R1_f211_ge: x_BsmtFin_Type_1_ALQ - t201_leaf1 >= 0
 T201_L2_f289_le: x_Sale_Type_New + 1e-06 t201_leaf2 <= 1
 T201_R2_f36_ge: x_Wood_Deck_SF - 224 t201_leaf2 >= 0
 T201_R3_f289_ge: x_Sale_Type_New - t201_leaf3 >= 0
 T201_L3_f217_le: x_BsmtFin_Type_1_Unf + 1e-06 t201_leaf3 <= 1
 T201_R4_f289_ge: x_Sale_Type_New - t201_leaf4 >= 0
 T201_R4_f217_ge: x_BsmtFin_Type_1_Unf - t201_leaf4 >= 0
 T201_L4_f195_le: x_Mas_Vnr_Type_BrkFace + 1e-06 t201_leaf4 <= 1
 T201_R5_f289_ge: x_Sale_Type_New - t201_leaf5 >= 0
 T201_R5_f217_ge: x_BsmtFin_Type_1_Unf - t201_leaf5 >= 0
 T201_R5_f195_ge: x_Mas_Vnr_Type_BrkFace - t201_leaf5 >= 0
 TREE_202_ONEHOT: t202_leaf0 + t202_leaf1 + t202_leaf2 + t202_leaf3
   + t202_leaf4 + t202_leaf5 + t202_leaf6 = 1
 T202_L0_f44_le: x_Yr_Sold + 997992.000001 t202_leaf0 <= 1e+06
 T202_L0_f119_le: x_Condition_1_Norm + 1e-06 t202_leaf0 <= 1
 T202_L0_f16_le: x_Heating_QC + 1e-06 t202_leaf0 <= 4
 T202_L1_f44_le: x_Yr_Sold + 997992.000001 t202_leaf1 <= 1e+06
 T202_L1_f119_le: x_Condition_1_Norm + 1e-06 t202_leaf1 <= 1
 T202_R1_f16_ge: x_Heating_QC - 2 t202_leaf1 >= 2
 T202_L2_f44_le: x_Yr_Sold + 997992.000001 t202_leaf2 <= 1e+06
 T202_R2_f119_ge: x_Condition_1_Norm - t202_leaf2 >= 0
 T202_L2_f14_le: x_Bsmt_Unf_SF + 905.6800010000001 t202_leaf2 <= 1812.68
 T202_L3_f44_le: x_Yr_Sold + 997992.000001 t202_leaf3 <= 1e+06
 T202_R3_f119_ge: x_Condition_1_Norm - t202_leaf3 >= 0
 T202_R3_f14_ge: x_Bsmt_Unf_SF - 907 t202_leaf3 >= 0
 T202_R4_f44_ge: x_Yr_Sold - 2008 t202_leaf4 >= 0
 T202_L4_f77_le: x_Land_Contour_Bnk + 1e-06 t202_leaf4 <= 1
 T202_L4_f261_le: x_Fence_GdWo + 1e-06 t202_leaf4 <= 1
 T202_R5_f44_ge: x_Yr_Sold - 2008 t202_leaf5 >= 0
 T202_L5_f77_le: x_Land_Contour_Bnk + 1e-06 t202_leaf5 <= 1
 T202_R5_f261_ge: x_Fence_GdWo - t202_leaf5 >= 0
 T202_R6_f44_ge: x_Yr_Sold - 2008 t202_leaf6 >= 0
 T202_R6_f77_ge: x_Land_Contour_Bnk - t202_leaf6 >= 0
 TREE_203_ONEHOT: t203_leaf0 + t203_leaf1 + t203_leaf2 + t203_leaf3
   + t203_leaf4 = 1
 T203_L0_f289_le: x_Sale_Type_New + 1e-06 t203_leaf0 <= 1
 T203_L0_f4_le: x_Overall_Cond + 999994.000001 t203_leaf0 <= 1e+06
 T203_L0_f5_le: x_Year_Built + 998040.000001 t203_leaf0 <= 1e+06
 T203_L1_f289_le: x_Sale_Type_New + 1e-06 t203_leaf1 <= 1
 T203_L1_f4_le: x_Overall_Cond + 999994.000001 t203_leaf1 <= 1e+06
 T203_R1_f5_ge: x_Year_Built - 1960 t203_leaf1 >= 0
 T203_L2_f289_le: x_Sale_Type_New + 1e-06 t203_leaf2 <= 1
 T203_R2_f4_ge: x_Overall_Cond - 6 t203_leaf2 >= 0
 T203_L2_f1_le: x_Lot_Area <= 7263.999999
 T203_L3_f289_le: x_Sale_Type_New + 1e-06 t203_leaf3 <= 1
 T203_R3_f4_ge: x_Overall_Cond - 6 t203_leaf3 >= 0
 T203_R3_f1_ge: x_Lot_Area - 264 t203_leaf3 >= 7000
 T203_R4_f289_ge: x_Sale_Type_New - t203_leaf4 >= 0
 TREE_204_ONEHOT: t204_leaf0 + t204_leaf1 + t204_leaf2 + t204_leaf3
   + t204_leaf4 = 1
 T204_L0_f18_le: x_2nd_Flr_SF + 345.5400010000001 t204_leaf0 <= 1641.54
 T204_L0_f44_le: x_Yr_Sold + 997993.000001 t204_leaf0 <= 1e+06
 T204_L0_f209_le: x_Bsmt_Exposure_No + 1e-06 t204_leaf0 <= 1
 T204_L1_f18_le: x_2nd_Flr_SF + 345.5400010000001 t204_leaf1 <= 1641.54
 T204_L1_f44_le: x_Yr_Sold + 997993.000001 t204_leaf1 <= 1e+06
 T204_R1_f209_ge: x_Bsmt_Exposure_No - t204_leaf1 >= 0
 T204_L2_f18_le: x_2nd_Flr_SF + 345.5400010000001 t204_leaf2 <= 1641.54
 T204_R2_f44_ge: x_Yr_Sold - 2007 t204_leaf2 >= 0
 T204_L2_f9_le: x_Exter_Cond + 2.000001 t204_leaf2 <= 4
 T204_L3_f18_le: x_2nd_Flr_SF + 345.5400010000001 t204_leaf3 <= 1641.54
 T204_R3_f44_ge: x_Yr_Sold - 2007 t204_leaf3 >= 0
 T204_R3_f9_ge: x_Exter_Cond - 3 t204_leaf3 >= -1
 T204_R4_f18_ge: x_2nd_Flr_SF - 1296 t204_leaf4 >= 0
 TREE_205_ONEHOT: t205_leaf0 + t205_leaf1 + t205_leaf2 + t205_leaf3
   + t205_leaf4 = 1
 T205_L0_f1_le: x_Lot_Area + 2500.000001 t205_leaf0 <= 7000
 T205_L0_f63_le: x_MS_Zoning_FV + 1e-06 t205_leaf0 <= 1
 T205_L1_f1_le: x_Lot_Area + 2500.000001 t205_leaf1 <= 7000
 T205_R1_f63_ge: x_MS_Zoning_FV - t205_leaf1 >= 0
 T205_R2_f1_ge: x_Lot_Area >= 4500
 T205_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t205_leaf2 <= 1
 T205_L2_f23_le: x_Full_Bath + 3.000001 t205_leaf2 <= 6
 T205_R3_f1_ge: x_Lot_Area >= 4500
 T205_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t205_leaf3 <= 1
 T205_R3_f23_ge: x_Full_Bath - 3 t205_leaf3 >= 0
 T205_R4_f1_ge: x_Lot_Area >= 4500
 T205_R4_f95_ge: x_Neighborhood_Crawfor - t205_leaf4 >= 0
 TREE_206_ONEHOT: t206_leaf0 + t206_leaf1 + t206_leaf2 + t206_leaf3
   + t206_leaf4 = 1
 T206_L0_f3_le: x_Overall_Qual + 999990.000001 t206_leaf0 <= 1e+06
 T206_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t206_leaf0 <= 1
 T206_L0_f233_le: x_Electrical_FuseA + 1e-06 t206_leaf0 <= 1
 T206_L1_f3_le: x_Overall_Qual + 999990.000001 t206_leaf1 <= 1e+06
 T206_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t206_leaf1 <= 1
 T206_R1_f233_ge: x_Electrical_FuseA - t206_leaf1 >= 0
 T206_L2_f3_le: x_Overall_Qual + 999990.000001 t206_leaf2 <= 1e+06
 T206_R2_f245_ge: x_FunctioNo_aplical_Typ - t206_leaf2 >= 0
 T206_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t206_leaf2 <= 1
 T206_L3_f3_le: x_Overall_Qual + 999990.000001 t206_leaf3 <= 1e+06
 T206_R3_f245_ge: x_FunctioNo_aplical_Typ - t206_leaf3 >= 0
 T206_R3_f164_ge: x_Exterior_1st_BrkFace - t206_leaf3 >= 0
 T206_R4_f3_ge: x_Overall_Qual - 10 t206_leaf4 >= 0
 TREE_207_ONEHOT: t207_leaf0 + t207_leaf1 + t207_leaf2 + t207_leaf3
   + t207_leaf4 + t207_leaf5 = 1
 T207_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t207_leaf0 <= 1
 T207_L0_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t207_leaf0 <= 1
 T207_L0_f202_le: x_Foundation_PConc + 1e-06 t207_leaf0 <= 1
 T207_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t207_leaf1 <= 1
 T207_L1_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t207_leaf1 <= 1
 T207_R1_f202_ge: x_Foundation_PConc - t207_leaf1 >= 0
 T207_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t207_leaf2 <= 1
 T207_R2_f211_ge: x_BsmtFin_Type_1_ALQ - t207_leaf2 >= 0
 T207_L2_f5_le: x_Year_Built + 998049.000001 t207_leaf2 <= 1e+06
 T207_L3_f164_le: x_Exterior_1st_BrkFace + 1e-06 t207_leaf3 <= 1
 T207_R3_f211_ge: x_BsmtFin_Type_1_ALQ - t207_leaf3 >= 0
 T207_R3_f5_ge: x_Year_Built - 1951 t207_leaf3 >= 0
 T207_R4_f164_ge: x_Exterior_1st_BrkFace - t207_leaf4 >= 0
 T207_L4_f106_le: x_Neighborhood_No_aplicames + 1e-06 t207_leaf4 <= 1
 T207_R5_f164_ge: x_Exterior_1st_BrkFace - t207_leaf5 >= 0
 T207_R5_f106_ge: x_Neighborhood_No_aplicames - t207_leaf5 >= 0
 TREE_208_ONEHOT: t208_leaf0 + t208_leaf1 + t208_leaf2 + t208_leaf3
   + t208_leaf4 + t208_leaf5 = 1
 T208_L0_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t208_leaf0 <= 1
 T208_L0_f17_le: x_1st_Flr_SF + 594.9000010000002 t208_leaf0 <= 2534.9
 T208_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t208_leaf0 <= 1
 T208_L1_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t208_leaf1 <= 1
 T208_L1_f17_le: x_1st_Flr_SF + 594.9000010000002 t208_leaf1 <= 2534.9
 T208_R1_f293_ge: x_Sale_Condition_Abnorml - t208_leaf1 >= 0
 T208_L2_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t208_leaf2 <= 1
 T208_R2_f17_ge: x_1st_Flr_SF - 1940 t208_leaf2 >= 0
 T208_L2_f33_le: x_Garage_Area + 9.750001000000111 t208_leaf2 <= 1163.75
 T208_L3_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t208_leaf3 <= 1
 T208_R3_f17_ge: x_1st_Flr_SF - 1940 t208_leaf3 >= 0
 T208_R3_f33_ge: x_Garage_Area - 1154 t208_leaf3 >= 0
 T208_R4_f211_ge: x_BsmtFin_Type_1_ALQ - t208_leaf4 >= 0
 T208_L4_f33_le: x_Garage_Area + 611.750001 t208_leaf4 <= 1163.75
 T208_R5_f211_ge: x_BsmtFin_Type_1_ALQ - t208_leaf5 >= 0
 T208_R5_f33_ge: x_Garage_Area - 552 t208_leaf5 >= 0
 TREE_209_ONEHOT: t209_leaf0 + t209_leaf1 + t209_leaf2 + t209_leaf3
   + t209_leaf4 + t209_leaf5 + t209_leaf6 + t209_leaf7 = 1
 T209_L0_f3_le: x_Overall_Qual + 999993.000001 t209_leaf0 <= 1e+06
 T209_L0_f18_le: x_2nd_Flr_SF + 981.540001 t209_leaf0 <= 1641.54
 T209_L0_f213_le: x_BsmtFin_Type_1_GLQ + 1e-06 t209_leaf0 <= 1
 T209_L1_f3_le: x_Overall_Qual + 999993.000001 t209_leaf1 <= 1e+06
 T209_L1_f18_le: x_2nd_Flr_SF + 981.540001 t209_leaf1 <= 1641.54
 T209_R1_f213_ge: x_BsmtFin_Type_1_GLQ - t209_leaf1 >= 0
 T209_L2_f3_le: x_Overall_Qual + 999993.000001 t209_leaf2 <= 1e+06
 T209_R2_f18_ge: x_2nd_Flr_SF - 660 t209_leaf2 >= 0
 T209_L2_f55_le: x_MS_SubClass_90 + 1e-06 t209_leaf2 <= 1
 T209_L3_f3_le: x_Overall_Qual + 999993.000001 t209_leaf3 <= 1e+06
 T209_R3_f18_ge: x_2nd_Flr_SF - 660 t209_leaf3 >= 0
 T209_R3_f55_ge: x_MS_SubClass_90 - t209_leaf3 >= 0
 T209_R4_f3_ge: x_Overall_Qual - 7 t209_leaf4 >= 0
 T209_L4_f6_le: x_Year_Remod/Add + 998007.000001 t209_leaf4 <= 1e+06
 T209_L4_f86_le: x_Land_Slope_Gtl + 1e-06 t209_leaf4 <= 1
 T209_R5_f3_ge: x_Overall_Qual - 7 t209_leaf5 >= 0
 T209_L5_f6_le: x_Year_Remod/Add + 998007.000001 t209_leaf5 <= 1e+06
 T209_R5_f86_ge: x_Land_Slope_Gtl - t209_leaf5 >= 0
 T209_R6_f3_ge: x_Overall_Qual - 7 t209_leaf6 >= 0
 T209_R6_f6_ge: x_Year_Remod/Add - 1993 t209_leaf6 >= 0
 T209_L6_f3_le: x_Overall_Qual + 999991.000001 t209_leaf6 <= 1e+06
 T209_R7_f3_ge: x_Overall_Qual - 7 t209_leaf7 >= 0
 T209_R7_f6_ge: x_Year_Remod/Add - 1993 t209_leaf7 >= 0
 T209_R7_f3_ge: x_Overall_Qual - 9 t209_leaf7 >= 0
 TREE_210_ONEHOT: t210_leaf0 + t210_leaf1 + t210_leaf2 + t210_leaf3
   + t210_leaf4 + t210_leaf5 + t210_leaf6 = 1
 T210_L0_f12_le: x_BsmtFin_SF_1 + 1761.500001 t210_leaf0 <= 1984.5
 T210_L0_f18_le: x_2nd_Flr_SF + 1265.540001 t210_leaf0 <= 1641.54
 T210_L0_f139_le: x_House_Style_1.5Fin + 1e-06 t210_leaf0 <= 1
 T210_L1_f12_le: x_BsmtFin_SF_1 + 1761.500001 t210_leaf1 <= 1984.5
 T210_L1_f18_le: x_2nd_Flr_SF + 1265.540001 t210_leaf1 <= 1641.54
 T210_R1_f139_ge: x_House_Style_1.5Fin - t210_leaf1 >= 0
 T210_L2_f12_le: x_BsmtFin_SF_1 + 1761.500001 t210_leaf2 <= 1984.5
 T210_R2_f18_ge: x_2nd_Flr_SF - 376 t210_leaf2 >= 0
 T210_L2_f119_le: x_Condition_1_Norm + 1e-06 t210_leaf2 <= 1
 T210_L3_f12_le: x_BsmtFin_SF_1 + 1761.500001 t210_leaf3 <= 1984.5
 T210_R3_f18_ge: x_2nd_Flr_SF - 376 t210_leaf3 >= 0
 T210_R3_f119_ge: x_Condition_1_Norm - t210_leaf3 >= 0
 T210_R4_f12_ge: x_BsmtFin_SF_1 - 223 t210_leaf4 >= 0
 T210_L4_f37_le: x_Open_Porch_SF + 124.810001 t210_leaf4 <= 488.81
 T210_L4_f5_le: x_Year_Built + 997995.000001 t210_leaf4 <= 1e+06
 T210_R5_f12_ge: x_BsmtFin_SF_1 - 223 t210_leaf5 >= 0
 T210_L5_f37_le: x_Open_Porch_SF + 124.810001 t210_leaf5 <= 488.81
 T210_R5_f5_ge: x_Year_Built - 2005 t210_leaf5 >= 0
 T210_R6_f12_ge: x_BsmtFin_SF_1 - 223 t210_leaf6 >= 0
 T210_R6_f37_ge: x_Open_Porch_SF - 364 t210_leaf6 >= 0
 TREE_211_ONEHOT: t211_leaf0 + t211_leaf1 + t211_leaf2 + t211_leaf3
   + t211_leaf4 + t211_leaf5 + t211_leaf6 = 1
 T211_L0_f44_le: x_Yr_Sold + 997993.000001 t211_leaf0 <= 1e+06
 T211_L0_f36_le: x_Wood_Deck_SF + 375.900001 t211_leaf0 <= 525.9
 T211_L0_f33_le: x_Garage_Area + 784.750001 t211_leaf0 <= 1163.75
 T211_L1_f44_le: x_Yr_Sold + 997993.000001 t211_leaf1 <= 1e+06
 T211_L1_f36_le: x_Wood_Deck_SF + 375.900001 t211_leaf1 <= 525.9
 T211_R1_f33_ge: x_Garage_Area - 379 t211_leaf1 >= 0
 T211_L2_f44_le: x_Yr_Sold + 997993.000001 t211_leaf2 <= 1e+06
 T211_R2_f36_ge: x_Wood_Deck_SF - 150 t211_leaf2 >= 0
 T211_R3_f44_ge: x_Yr_Sold - 2007 t211_leaf3 >= 0
 T211_L3_f17_le: x_1st_Flr_SF + 634.9000010000002 t211_leaf3 <= 2534.9
 T211_L3_f11_le: x_Bsmt_Cond + 1.000001 t211_leaf3 <= 4
 T211_R4_f44_ge: x_Yr_Sold - 2007 t211_leaf4 >= 0
 T211_L4_f17_le: x_1st_Flr_SF + 634.9000010000002 t211_leaf4 <= 2534.9
 T211_R4_f11_ge: x_Bsmt_Cond - 4 t211_leaf4 >= -1
 T211_R5_f44_ge: x_Yr_Sold - 2007 t211_leaf5 >= 0
 T211_R5_f17_ge: x_1st_Flr_SF - 1900 t211_leaf5 >= 0
 T211_L5_f0_le: x_Lot_Frontage + 83.530001 t211_leaf5 <= 175.53
 T211_R6_f44_ge: x_Yr_Sold - 2007 t211_leaf6 >= 0
 T211_R6_f17_ge: x_1st_Flr_SF - 1900 t211_leaf6 >= 0
 T211_R6_f0_ge: x_Lot_Frontage - 92 t211_leaf6 >= 0
 TREE_212_ONEHOT: t212_leaf0 + t212_leaf1 + t212_leaf2 + t212_leaf3
   + t212_leaf4 = 1
 T212_L0_f13_le: x_BsmtFin_SF_2 + 35.05000099999995 t212_leaf0 <= 757.05
 T212_L0_f44_le: x_Yr_Sold + 997993.000001 t212_leaf0 <= 1e+06
 T212_L0_f1_le: x_Lot_Area <= 8196.999999
 T212_L1_f13_le: x_BsmtFin_SF_2 + 35.05000099999995 t212_leaf1 <= 757.05
 T212_L1_f44_le: x_Yr_Sold + 997993.000001 t212_leaf1 <= 1e+06
 T212_R1_f1_ge: x_Lot_Area - 1197 t212_leaf1 >= 7000
 T212_L2_f13_le: x_BsmtFin_SF_2 + 35.05000099999995 t212_leaf2 <= 757.05
 T212_R2_f44_ge: x_Yr_Sold - 2007 t212_leaf2 >= 0
 T212_L2_f161_le: x_Exterior_1st_AsbShng + 1e-06 t212_leaf2 <= 1
 T212_L3_f13_le: x_BsmtFin_SF_2 + 35.05000099999995 t212_leaf3 <= 757.05
 T212_R3_f44_ge: x_Yr_Sold - 2007 t212_leaf3 >= 0
 T212_R3_f161_ge: x_Exterior_1st_AsbShng - t212_leaf3 >= 0
 T212_R4_f13_ge: x_BsmtFin_SF_2 - 722 t212_leaf4 >= 0
 TREE_213_ONEHOT: t213_leaf0 + t213_leaf1 + t213_leaf2 + t213_leaf3
   + t213_leaf4 + t213_leaf5 = 1
 T213_L0_f117_le: x_Condition_1_Artery + 1e-06 t213_leaf0 <= 1
 T213_L0_f38_le: x_Enclosed_Porch + 100.750001 t213_leaf0 <= 292.75
 T213_L0_f289_le: x_Sale_Type_New + 1e-06 t213_leaf0 <= 1
 T213_L1_f117_le: x_Condition_1_Artery + 1e-06 t213_leaf1 <= 1
 T213_L1_f38_le: x_Enclosed_Porch + 100.750001 t213_leaf1 <= 292.75
 T213_R1_f289_ge: x_Sale_Type_New - t213_leaf1 >= 0
 T213_L2_f117_le: x_Condition_1_Artery + 1e-06 t213_leaf2 <= 1
 T213_R2_f38_ge: x_Enclosed_Porch - 192 t213_leaf2 >= 0
 T213_R3_f117_ge: x_Condition_1_Artery - t213_leaf3 >= 0
 T213_L3_f0_le: x_Lot_Frontage + 95.530001 t213_leaf3 <= 175.53
 T213_L3_f3_le: x_Overall_Qual + 999995.000001 t213_leaf3 <= 1e+06
 T213_R4_f117_ge: x_Condition_1_Artery - t213_leaf4 >= 0
 T213_L4_f0_le: x_Lot_Frontage + 95.530001 t213_leaf4 <= 175.53
 T213_R4_f3_ge: x_Overall_Qual - 5 t213_leaf4 >= 0
 T213_R5_f117_ge: x_Condition_1_Artery - t213_leaf5 >= 0
 T213_R5_f0_ge: x_Lot_Frontage - 80 t213_leaf5 >= 0
 TREE_214_ONEHOT: t214_leaf0 + t214_leaf1 + t214_leaf2 + t214_leaf3
   + t214_leaf4 + t214_leaf5 = 1
 T214_L0_f33_le: x_Garage_Area + 292.750001 t214_leaf0 <= 1163.75
 T214_L0_f5_le: x_Year_Built + 998089.000001 t214_leaf0 <= 1e+06
 T214_L0_f17_le: x_1st_Flr_SF + 1622.900001 t214_leaf0 <= 2534.9
 T214_L1_f33_le: x_Garage_Area + 292.750001 t214_leaf1 <= 1163.75
 T214_L1_f5_le: x_Year_Built + 998089.000001 t214_leaf1 <= 1e+06
 T214_R1_f17_ge: x_1st_Flr_SF - 912 t214_leaf1 >= 0
 T214_L2_f33_le: x_Garage_Area + 292.750001 t214_leaf2 <= 1163.75
 T214_R2_f5_ge: x_Year_Built - 1911 t214_leaf2 >= 0
 T214_L2_f37_le: x_Open_Porch_SF + 330.810001 t214_leaf2 <= 488.81
 T214_L3_f33_le: x_Garage_Area + 292.750001 t214_leaf3 <= 1163.75
 T214_R3_f5_ge: x_Year_Built - 1911 t214_leaf3 >= 0
 T214_R3_f37_ge: x_Open_Porch_SF - 158 t214_leaf3 >= 0
 T214_R4_f33_ge: x_Garage_Area - 871 t214_leaf4 >= 0
 T214_L4_f6_le: x_Year_Remod/Add + 997992.000001 t214_leaf4 <= 1e+06
 T214_R5_f33_ge: x_Garage_Area - 871 t214_leaf5 >= 0
 T214_R5_f6_ge: x_Year_Remod/Add - 2008 t214_leaf5 >= 0
 TREE_215_ONEHOT: t215_leaf0 + t215_leaf1 + t215_leaf2 + t215_leaf3
   + t215_leaf4 + t215_leaf5 + t215_leaf6 + t215_leaf7 = 1
 T215_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t215_leaf0 <= 1
 T215_L0_f1_le: x_Lot_Area <= 13517.999999
 T215_L0_f1_le: x_Lot_Area <= 10439.999999
 T215_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t215_leaf1 <= 1
 T215_L1_f1_le: x_Lot_Area <= 13517.999999
 T215_R1_f1_ge: x_Lot_Area - 3440 t215_leaf1 >= 7000
 T215_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t215_leaf2 <= 1
 T215_R2_f1_ge: x_Lot_Area - 6518 t215_leaf2 >= 7000
 T215_L2_f32_le: x_Garage_Cars + 2.000001 t215_leaf2 <= 4
 T215_L3_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t215_leaf3 <= 1
 T215_R3_f1_ge: x_Lot_Area - 6518 t215_leaf3 >= 7000
 T215_R3_f32_ge: x_Garage_Cars - 2 t215_leaf3 >= 0
 T215_R4_f245_ge: x_FunctioNo_aplical_Typ - t215_leaf4 >= 0
 T215_L4_f259_le: x_Paved_Drive_Y + 1e-06 t215_leaf4 <= 1
 T215_L4_f32_le: x_Garage_Cars + 2.000001 t215_leaf4 <= 4
 T215_R5_f245_ge: x_FunctioNo_aplical_Typ - t215_leaf5 >= 0
 T215_L5_f259_le: x_Paved_Drive_Y + 1e-06 t215_leaf5 <= 1
 T215_R5_f32_ge: x_Garage_Cars - 2 t215_leaf5 >= 0
 T215_R6_f245_ge: x_FunctioNo_aplical_Typ - t215_leaf6 >= 0
 T215_R6_f259_ge: x_Paved_Drive_Y - t215_leaf6 >= 0
 T215_L6_f3_le: x_Overall_Qual + 999993.000001 t215_leaf6 <= 1e+06
 T215_R7_f245_ge: x_FunctioNo_aplical_Typ - t215_leaf7 >= 0
 T215_R7_f259_ge: x_Paved_Drive_Y - t215_leaf7 >= 0
 T215_R7_f3_ge: x_Overall_Qual - 7 t215_leaf7 >= 0
 TREE_216_ONEHOT: t216_leaf0 + t216_leaf1 + t216_leaf2 + t216_leaf3
   + t216_leaf4 + t216_leaf5 + t216_leaf6 = 1
 T216_L0_f0_le: x_Lot_Frontage + 55.530001 t216_leaf0 <= 175.53
 T216_L0_f1_le: x_Lot_Area <= 19521.999999
 T216_L0_f44_le: x_Yr_Sold + 997991.000001 t216_leaf0 <= 1e+06
 T216_L1_f0_le: x_Lot_Frontage + 55.530001 t216_leaf1 <= 175.53
 T216_L1_f1_le: x_Lot_Area <= 19521.999999
 T216_R1_f44_ge: x_Yr_Sold - 2009 t216_leaf1 >= 0
 T216_L2_f0_le: x_Lot_Frontage + 55.530001 t216_leaf2 <= 175.53
 T216_R2_f1_ge: x_Lot_Area - 12522 t216_leaf2 >= 7000
 T216_L2_f25_le: x_Bedroom_AbvGr + 3.000001 t216_leaf2 <= 7
 T216_L3_f0_le: x_Lot_Frontage + 55.530001 t216_leaf3 <= 175.53
 T216_R3_f1_ge: x_Lot_Area - 12522 t216_leaf3 >= 7000
 T216_R3_f25_ge: x_Bedroom_AbvGr - 4 t216_leaf3 >= 0
 T216_R4_f0_ge: x_Lot_Frontage - 120 t216_leaf4 >= 0
 T216_L4_f12_le: x_BsmtFin_SF_1 + 597.5000010000001 t216_leaf4 <= 1984.5
 T216_L4_f195_le: x_Mas_Vnr_Type_BrkFace + 1e-06 t216_leaf4 <= 1
 T216_R5_f0_ge: x_Lot_Frontage - 120 t216_leaf5 >= 0
 T216_L5_f12_le: x_BsmtFin_SF_1 + 597.5000010000001 t216_leaf5 <= 1984.5
 T216_R5_f195_ge: x_Mas_Vnr_Type_BrkFace - t216_leaf5 >= 0
 T216_R6_f0_ge: x_Lot_Frontage - 120 t216_leaf6 >= 0
 T216_R6_f12_ge: x_BsmtFin_SF_1 - 1387 t216_leaf6 >= 0
 TREE_217_ONEHOT: t217_leaf0 + t217_leaf1 + t217_leaf2 + t217_leaf3
   + t217_leaf4 + t217_leaf5 + t217_leaf6 = 1
 T217_L0_f3_le: x_Overall_Qual + 999993.000001 t217_leaf0 <= 1e+06
 T217_L0_f134_le: x_Bldg_Type_1Fam + 1e-06 t217_leaf0 <= 1
 T217_L0_f6_le: x_Year_Remod/Add + 998001.000001 t217_leaf0 <= 1e+06
 T217_L1_f3_le: x_Overall_Qual + 999993.000001 t217_leaf1 <= 1e+06
 T217_L1_f134_le: x_Bldg_Type_1Fam + 1e-06 t217_leaf1 <= 1
 T217_R1_f6_ge: x_Year_Remod/Add - 1999 t217_leaf1 >= 0
 T217_L2_f3_le: x_Overall_Qual + 999993.000001 t217_leaf2 <= 1e+06
 T217_R2_f134_ge: x_Bldg_Type_1Fam - t217_leaf2 >= 0
 T217_L2_f144_le: x_House_Style_2Story + 1e-06 t217_leaf2 <= 1
 T217_L3_f3_le: x_Overall_Qual + 999993.000001 t217_leaf3 <= 1e+06
 T217_R3_f134_ge: x_Bldg_Type_1Fam - t217_leaf3 >= 0
 T217_R3_f144_ge: x_House_Style_2Story - t217_leaf3 >= 0
 T217_R4_f3_ge: x_Overall_Qual - 7 t217_leaf4 >= 0
 T217_L4_f77_le: x_Land_Contour_Bnk + 1e-06 t217_leaf4 <= 1
 T217_L4_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t217_leaf4 <= 1
 T217_R5_f3_ge: x_Overall_Qual - 7 t217_leaf5 >= 0
 T217_L5_f77_le: x_Land_Contour_Bnk + 1e-06 t217_leaf5 <= 1
 T217_R5_f207_ge: x_Bsmt_Exposure_Gd - t217_leaf5 >= 0
 T217_R6_f3_ge: x_Overall_Qual - 7 t217_leaf6 >= 0
 T217_R6_f77_ge: x_Land_Contour_Bnk - t217_leaf6 >= 0
 TREE_218_ONEHOT: t218_leaf0 + t218_leaf1 + t218_leaf2 + t218_leaf3
   + t218_leaf4 = 1
 T218_L0_f298_le: x_Sale_Condition_Partial + 1e-06 t218_leaf0 <= 1
 T218_L0_f4_le: x_Overall_Cond + 999993.000001 t218_leaf0 <= 1e+06
 T218_L0_f17_le: x_1st_Flr_SF + 334.900001 t218_leaf0 <= 2534.9
 T218_L1_f298_le: x_Sale_Condition_Partial + 1e-06 t218_leaf1 <= 1
 T218_L1_f4_le: x_Overall_Cond + 999993.000001 t218_leaf1 <= 1e+06
 T218_R1_f17_ge: x_1st_Flr_SF - 2200 t218_leaf1 >= 0
 T218_L2_f298_le: x_Sale_Condition_Partial + 1e-06 t218_leaf2 <= 1
 T218_R2_f4_ge: x_Overall_Cond - 7 t218_leaf2 >= 0
 T218_L2_f44_le: x_Yr_Sold + 997990.000001 t218_leaf2 <= 1e+06
 T218_L3_f298_le: x_Sale_Condition_Partial + 1e-06 t218_leaf3 <= 1
 T218_R3_f4_ge: x_Overall_Cond - 7 t218_leaf3 >= 0
 T218_R3_f44_ge: x_Yr_Sold - 2010 t218_leaf3 >= 0
 T218_R4_f298_ge: x_Sale_Condition_Partial - t218_leaf4 >= 0
 TREE_219_ONEHOT: t219_leaf0 + t219_leaf1 + t219_leaf2 + t219_leaf3
   + t219_leaf4 + t219_leaf5 + t219_leaf6 = 1
 T219_L0_f1_le: x_Lot_Area <= 15311.999999
 T219_L0_f46_le: x_MS_SubClass_30 + 1e-06 t219_leaf0 <= 1
 T219_L0_f4_le: x_Overall_Cond + 999994.000001 t219_leaf0 <= 1e+06
 T219_L1_f1_le: x_Lot_Area <= 15311.999999
 T219_L1_f46_le: x_MS_SubClass_30 + 1e-06 t219_leaf1 <= 1
 T219_R1_f4_ge: x_Overall_Cond - 6 t219_leaf1 >= 0
 T219_L2_f1_le: x_Lot_Area <= 15311.999999
 T219_R2_f46_ge: x_MS_SubClass_30 - t219_leaf2 >= 0
 T219_L2_f0_le: x_Lot_Frontage + 104.530001 t219_leaf2 <= 175.53
 T219_L3_f1_le: x_Lot_Area <= 15311.999999
 T219_R3_f46_ge: x_MS_SubClass_30 - t219_leaf3 >= 0
 T219_R3_f0_ge: x_Lot_Frontage - 71 t219_leaf3 >= 0
 T219_R4_f1_ge: x_Lot_Area - 8312 t219_leaf4 >= 7000
 T219_L4_f37_le: x_Open_Porch_SF + 226.810001 t219_leaf4 <= 488.81
 T219_L4_f20_le: x_Gr_Liv_Area + 1382.760001 t219_leaf4 <= 3421.76
 T219_R5_f1_ge: x_Lot_Area - 8312 t219_leaf5 >= 7000
 T219_L5_f37_le: x_Open_Porch_SF + 226.810001 t219_leaf5 <= 488.81
 T219_R5_f20_ge: x_Gr_Liv_Area - 2039 t219_leaf5 >= 0
 T219_R6_f1_ge: x_Lot_Area - 8312 t219_leaf6 >= 7000
 T219_R6_f37_ge: x_Open_Porch_SF - 262 t219_leaf6 >= 0
 TREE_220_ONEHOT: t220_leaf0 + t220_leaf1 + t220_leaf2 + t220_leaf3
   + t220_leaf4 + t220_leaf5 + t220_leaf6 + t220_leaf7 = 1
 T220_L0_f44_le: x_Yr_Sold + 997991.000001 t220_leaf0 <= 1e+06
 T220_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t220_leaf0 <= 1
 T220_L0_f40_le: x_Screen_Porch + 147.270001 t220_leaf0 <= 268.27
 T220_L1_f44_le: x_Yr_Sold + 997991.000001 t220_leaf1 <= 1e+06
 T220_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t220_leaf1 <= 1
 T220_R1_f40_ge: x_Screen_Porch - 121 t220_leaf1 >= 0
 T220_L2_f44_le: x_Yr_Sold + 997991.000001 t220_leaf2 <= 1e+06
 T220_R2_f245_ge: x_FunctioNo_aplical_Typ - t220_leaf2 >= 0
 T220_L2_f1_le: x_Lot_Area + 1638.000001 t220_leaf2 <= 7000
 T220_L3_f44_le: x_Yr_Sold + 997991.000001 t220_leaf3 <= 1e+06
 T220_R3_f245_ge: x_FunctioNo_aplical_Typ - t220_leaf3 >= 0
 T220_R3_f1_ge: x_Lot_Area >= 5362
 T220_R4_f44_ge: x_Yr_Sold - 2009 t220_leaf4 >= 0
 T220_L4_f9_le: x_Exter_Cond + 2.000001 t220_leaf4 <= 4
 T220_L4_f44_le: x_Yr_Sold + 997990.000001 t220_leaf4 <= 1e+06
 T220_R5_f44_ge: x_Yr_Sold - 2009 t220_leaf5 >= 0
 T220_L5_f9_le: x_Exter_Cond + 2.000001 t220_leaf5 <= 4
 T220_R5_f44_ge: x_Yr_Sold - 2010 t220_leaf5 >= 0
 T220_R6_f44_ge: x_Yr_Sold - 2009 t220_leaf6 >= 0
 T220_R6_f9_ge: x_Exter_Cond - 3 t220_leaf6 >= -1
 T220_L6_f3_le: x_Overall_Qual + 999996.000001 t220_leaf6 <= 1e+06
 T220_R7_f44_ge: x_Yr_Sold - 2009 t220_leaf7 >= 0
 T220_R7_f9_ge: x_Exter_Cond - 3 t220_leaf7 >= -1
 T220_R7_f3_ge: x_Overall_Qual - 4 t220_leaf7 >= 0
 TREE_221_ONEHOT: t221_leaf0 + t221_leaf1 + t221_leaf2 + t221_leaf3
   + t221_leaf4 + t221_leaf5 + t221_leaf6 = 1
 T221_L0_f44_le: x_Yr_Sold + 997993.000001 t221_leaf0 <= 1e+06
 T221_L0_f36_le: x_Wood_Deck_SF + 326.900001 t221_leaf0 <= 525.9
 T221_L0_f232_le: x_Central_Air_Y + 1e-06 t221_leaf0 <= 1
 T221_L1_f44_le: x_Yr_Sold + 997993.000001 t221_leaf1 <= 1e+06
 T221_L1_f36_le: x_Wood_Deck_SF + 326.900001 t221_leaf1 <= 525.9
 T221_R1_f232_ge: x_Central_Air_Y - t221_leaf1 >= 0
 T221_L2_f44_le: x_Yr_Sold + 997993.000001 t221_leaf2 <= 1e+06
 T221_R2_f36_ge: x_Wood_Deck_SF - 199 t221_leaf2 >= 0
 T221_R3_f44_ge: x_Yr_Sold - 2007 t221_leaf3 >= 0
 T221_L3_f14_le: x_Bsmt_Unf_SF + 1357.680001 t221_leaf3 <= 1812.68
 T221_L3_f282_le: x_Mo_Sold_12 + 1e-06 t221_leaf3 <= 1
 T221_R4_f44_ge: x_Yr_Sold - 2007 t221_leaf4 >= 0
 T221_L4_f14_le: x_Bsmt_Unf_SF + 1357.680001 t221_leaf4 <= 1812.68
 T221_R4_f282_ge: x_Mo_Sold_12 - t221_leaf4 >= 0
 T221_R5_f44_ge: x_Yr_Sold - 2007 t221_leaf5 >= 0
 T221_R5_f14_ge: x_Bsmt_Unf_SF - 455 t221_leaf5 >= 0
 T221_L5_f16_le: x_Heating_QC + 1.000001 t221_leaf5 <= 4
 T221_R6_f44_ge: x_Yr_Sold - 2007 t221_leaf6 >= 0
 T221_R6_f14_ge: x_Bsmt_Unf_SF - 455 t221_leaf6 >= 0
 T221_R6_f16_ge: x_Heating_QC - t221_leaf6 >= 2
 TREE_222_ONEHOT: t222_leaf0 + t222_leaf1 + t222_leaf2 + t222_leaf3
   + t222_leaf4 = 1
 T222_L0_f18_le: x_2nd_Flr_SF + 12.54000100000007 t222_leaf0 <= 1641.54
 T222_L0_f4_le: x_Overall_Cond + 999993.000001 t222_leaf0 <= 1e+06
 T222_L0_f37_le: x_Open_Porch_SF + 413.810001 t222_leaf0 <= 488.81
 T222_L1_f18_le: x_2nd_Flr_SF + 12.54000100000007 t222_leaf1 <= 1641.54
 T222_L1_f4_le: x_Overall_Cond + 999993.000001 t222_leaf1 <= 1e+06
 T222_R1_f37_ge: x_Open_Porch_SF - 75 t222_leaf1 >= 0
 T222_L2_f18_le: x_2nd_Flr_SF + 12.54000100000007 t222_leaf2 <= 1641.54
 T222_R2_f4_ge: x_Overall_Cond - 7 t222_leaf2 >= 0
 T222_L2_f18_le: x_2nd_Flr_SF + 807.540001 t222_leaf2 <= 1641.54
 T222_L3_f18_le: x_2nd_Flr_SF + 12.54000100000007 t222_leaf3 <= 1641.54
 T222_R3_f4_ge: x_Overall_Cond - 7 t222_leaf3 >= 0
 T222_R3_f18_ge: x_2nd_Flr_SF - 834 t222_leaf3 >= 0
 T222_R4_f18_ge: x_2nd_Flr_SF - 1629 t222_leaf4 >= 0
 TREE_223_ONEHOT: t223_leaf0 + t223_leaf1 + t223_leaf2 + t223_leaf3
   + t223_leaf4 = 1
 T223_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t223_leaf0 <= 1
 T223_L0_f3_le: x_Overall_Qual + 999993.000001 t223_leaf0 <= 1e+06
 T223_L0_f31_le: x_Garage_Yr_Blt + 998003.000001 t223_leaf0 <= 1e+06
 T223_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t223_leaf1 <= 1
 T223_L1_f3_le: x_Overall_Qual + 999993.000001 t223_leaf1 <= 1e+06
 T223_R1_f31_ge: x_Garage_Yr_Blt - 1997 t223_leaf1 >= 0
 T223_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t223_leaf2 <= 1
 T223_R2_f3_ge: x_Overall_Qual - 7 t223_leaf2 >= 0
 T223_L2_f105_le: x_Neighborhood_NWAmes + 1e-06 t223_leaf2 <= 1
 T223_L3_f164_le: x_Exterior_1st_BrkFace + 1e-06 t223_leaf3 <= 1
 T223_R3_f3_ge: x_Overall_Qual - 7 t223_leaf3 >= 0
 T223_R3_f105_ge: x_Neighborhood_NWAmes - t223_leaf3 >= 0
 T223_R4_f164_ge: x_Exterior_1st_BrkFace - t223_leaf4 >= 0
 TREE_224_ONEHOT: t224_leaf0 + t224_leaf1 + t224_leaf2 + t224_leaf3
   + t224_leaf4 + t224_leaf5 = 1
 T224_L0_f4_le: x_Overall_Cond + 999994.000001 t224_leaf0 <= 1e+06
 T224_L0_f37_le: x_Open_Porch_SF + 197.810001 t224_leaf0 <= 488.81
 T224_L0_f5_le: x_Year_Built + 998058.000001 t224_leaf0 <= 1e+06
 T224_L1_f4_le: x_Overall_Cond + 999994.000001 t224_leaf1 <= 1e+06
 T224_L1_f37_le: x_Open_Porch_SF + 197.810001 t224_leaf1 <= 488.81
 T224_R1_f5_ge: x_Year_Built - 1942 t224_leaf1 >= 0
 T224_L2_f4_le: x_Overall_Cond + 999994.000001 t224_leaf2 <= 1e+06
 T224_R2_f37_ge: x_Open_Porch_SF - 291 t224_leaf2 >= 0
 T224_R3_f4_ge: x_Overall_Cond - 6 t224_leaf3 >= 0
 T224_L3_f78_le: x_Land_Contour_HLS + 1e-06 t224_leaf3 <= 1
 T224_L3_f119_le: x_Condition_1_Norm + 1e-06 t224_leaf3 <= 1
 T224_R4_f4_ge: x_Overall_Cond - 6 t224_leaf4 >= 0
 T224_L4_f78_le: x_Land_Contour_HLS + 1e-06 t224_leaf4 <= 1
 T224_R4_f119_ge: x_Condition_1_Norm - t224_leaf4 >= 0
 T224_R5_f4_ge: x_Overall_Cond - 6 t224_leaf5 >= 0
 T224_R5_f78_ge: x_Land_Contour_HLS - t224_leaf5 >= 0
 TREE_225_ONEHOT: t225_leaf0 + t225_leaf1 + t225_leaf2 + t225_leaf3
   + t225_leaf4 + t225_leaf5 = 1
 T225_L0_f21_le: x_Bsmt_Full_Bath + 0.020001 t225_leaf0 <= 2.02
 T225_L0_f76_le: x_Lot_Shape_Reg + 1e-06 t225_leaf0 <= 1
 T225_L0_f3_le: x_Overall_Qual + 999993.000001 t225_leaf0 <= 1e+06
 T225_L1_f21_le: x_Bsmt_Full_Bath + 0.020001 t225_leaf1 <= 2.02
 T225_L1_f76_le: x_Lot_Shape_Reg + 1e-06 t225_leaf1 <= 1
 T225_R1_f3_ge: x_Overall_Qual - 7 t225_leaf1 >= 0
 T225_L2_f21_le: x_Bsmt_Full_Bath + 0.020001 t225_leaf2 <= 2.02
 T225_R2_f76_ge: x_Lot_Shape_Reg - t225_leaf2 >= 0
 T225_L2_f4_le: x_Overall_Cond + 999993.000001 t225_leaf2 <= 1e+06
 T225_L3_f21_le: x_Bsmt_Full_Bath + 0.020001 t225_leaf3 <= 2.02
 T225_R3_f76_ge: x_Lot_Shape_Reg - t225_leaf3 >= 0
 T225_R3_f4_ge: x_Overall_Cond - 7 t225_leaf3 >= 0
 T225_R4_f21_ge: x_Bsmt_Full_Bath - 2 t225_leaf4 >= 0
 T225_L4_f12_le: x_BsmtFin_SF_1 + 925.5000010000001 t225_leaf4 <= 1984.5
 T225_R5_f21_ge: x_Bsmt_Full_Bath - 2 t225_leaf5 >= 0
 T225_R5_f12_ge: x_BsmtFin_SF_1 - 1059 t225_leaf5 >= 0
 TREE_226_ONEHOT: t226_leaf0 + t226_leaf1 + t226_leaf2 + t226_leaf3
   + t226_leaf4 + t226_leaf5 + t226_leaf6 = 1
 T226_L0_f20_le: x_Gr_Liv_Area + 2433.760001 t226_leaf0 <= 3421.76
 T226_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t226_leaf0 <= 1
 T226_L0_f3_le: x_Overall_Qual + 999997.000001 t226_leaf0 <= 1e+06
 T226_L1_f20_le: x_Gr_Liv_Area + 2433.760001 t226_leaf1 <= 3421.76
 T226_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t226_leaf1 <= 1
 T226_R1_f3_ge: x_Overall_Qual - 3 t226_leaf1 >= 0
 T226_L2_f20_le: x_Gr_Liv_Area + 2433.760001 t226_leaf2 <= 3421.76
 T226_R2_f161_ge: x_Exterior_1st_AsbShng - t226_leaf2 >= 0
 T226_R3_f20_ge: x_Gr_Liv_Area - 988 t226_leaf3 >= 0
 T226_L3_f3_le: x_Overall_Qual + 999993.000001 t226_leaf3 <= 1e+06
 T226_L3_f10_le: x_Bsmt_Qual + 4.000001 t226_leaf3 <= 4
 T226_R4_f20_ge: x_Gr_Liv_Area - 988 t226_leaf4 >= 0
 T226_L4_f3_le: x_Overall_Qual + 999993.000001 t226_leaf4 <= 1e+06
 T226_R4_f10_ge: x_Bsmt_Qual >= 0
 T226_R5_f20_ge: x_Gr_Liv_Area - 988 t226_leaf5 >= 0
 T226_R5_f3_ge: x_Overall_Qual - 7 t226_leaf5 >= 0
 T226_L5_f166_le: x_Exterior_1st_CemntBd + 1e-06 t226_leaf5 <= 1
 T226_R6_f20_ge: x_Gr_Liv_Area - 988 t226_leaf6 >= 0
 T226_R6_f3_ge: x_Overall_Qual - 7 t226_leaf6 >= 0
 T226_R6_f166_ge: x_Exterior_1st_CemntBd - t226_leaf6 >= 0
 TREE_227_ONEHOT: t227_leaf0 + t227_leaf1 + t227_leaf2 + t227_leaf3
   + t227_leaf4 = 1
 T227_L0_f40_le: x_Screen_Porch + 126.270001 t227_leaf0 <= 268.27
 T227_L0_f18_le: x_2nd_Flr_SF + 12.54000100000007 t227_leaf0 <= 1641.54
 T227_L0_f5_le: x_Year_Built + 998089.000001 t227_leaf0 <= 1e+06
 T227_L1_f40_le: x_Screen_Porch + 126.270001 t227_leaf1 <= 268.27
 T227_L1_f18_le: x_2nd_Flr_SF + 12.54000100000007 t227_leaf1 <= 1641.54
 T227_R1_f5_ge: x_Year_Built - 1911 t227_leaf1 >= 0
 T227_L2_f40_le: x_Screen_Porch + 126.270001 t227_leaf2 <= 268.27
 T227_R2_f18_ge: x_2nd_Flr_SF - 1629 t227_leaf2 >= 0
 T227_R3_f40_ge: x_Screen_Porch - 142 t227_leaf3 >= 0
 T227_L3_f261_le: x_Fence_GdWo + 1e-06 t227_leaf3 <= 1
 T227_R4_f40_ge: x_Screen_Porch - 142 t227_leaf4 >= 0
 T227_R4_f261_ge: x_Fence_GdWo - t227_leaf4 >= 0
 TREE_228_ONEHOT: t228_leaf0 + t228_leaf1 + t228_leaf2 + t228_leaf3
   + t228_leaf4 + t228_leaf5 + t228_leaf6 + t228_leaf7 = 1
 T228_L0_f44_le: x_Yr_Sold + 997992.000001 t228_leaf0 <= 1e+06
 T228_L0_f76_le: x_Lot_Shape_Reg + 1e-06 t228_leaf0 <= 1
 T228_L0_f37_le: x_Open_Porch_SF + 226.810001 t228_leaf0 <= 488.81
 T228_L1_f44_le: x_Yr_Sold + 997992.000001 t228_leaf1 <= 1e+06
 T228_L1_f76_le: x_Lot_Shape_Reg + 1e-06 t228_leaf1 <= 1
 T228_R1_f37_ge: x_Open_Porch_SF - 262 t228_leaf1 >= 0
 T228_L2_f44_le: x_Yr_Sold + 997992.000001 t228_leaf2 <= 1e+06
 T228_R2_f76_ge: x_Lot_Shape_Reg - t228_leaf2 >= 0
 T228_L2_f282_le: x_Mo_Sold_12 + 1e-06 t228_leaf2 <= 1
 T228_L3_f44_le: x_Yr_Sold + 997992.000001 t228_leaf3 <= 1e+06
 T228_R3_f76_ge: x_Lot_Shape_Reg - t228_leaf3 >= 0
 T228_R3_f282_ge: x_Mo_Sold_12 - t228_leaf3 >= 0
 T228_R4_f44_ge: x_Yr_Sold - 2008 t228_leaf4 >= 0
 T228_L4_f237_le: x_Electrical_SBrkr + 1e-06 t228_leaf4 <= 1
 T228_L4_f4_le: x_Overall_Cond + 999994.000001 t228_leaf4 <= 1e+06
 T228_R5_f44_ge: x_Yr_Sold - 2008 t228_leaf5 >= 0
 T228_L5_f237_le: x_Electrical_SBrkr + 1e-06 t228_leaf5 <= 1
 T228_R5_f4_ge: x_Overall_Cond - 6 t228_leaf5 >= 0
 T228_R6_f44_ge: x_Yr_Sold - 2008 t228_leaf6 >= 0
 T228_R6_f237_ge: x_Electrical_SBrkr - t228_leaf6 >= 0
 T228_L6_f150_le: x_Roof_Style_Hip + 1e-06 t228_leaf6 <= 1
 T228_R7_f44_ge: x_Yr_Sold - 2008 t228_leaf7 >= 0
 T228_R7_f237_ge: x_Electrical_SBrkr - t228_leaf7 >= 0
 T228_R7_f150_ge: x_Roof_Style_Hip - t228_leaf7 >= 0
 TREE_229_ONEHOT: t229_leaf0 + t229_leaf1 + t229_leaf2 + t229_leaf3
   + t229_leaf4 = 1
 T229_L0_f18_le: x_2nd_Flr_SF + 273.5400010000001 t229_leaf0 <= 1641.54
 T229_L0_f117_le: x_Condition_1_Artery + 1e-06 t229_leaf0 <= 1
 T229_L0_f0_le: x_Lot_Frontage + 26.530001 t229_leaf0 <= 175.53
 T229_L1_f18_le: x_2nd_Flr_SF + 273.5400010000001 t229_leaf1 <= 1641.54
 T229_L1_f117_le: x_Condition_1_Artery + 1e-06 t229_leaf1 <= 1
 T229_R1_f0_ge: x_Lot_Frontage - 149 t229_leaf1 >= 0
 T229_L2_f18_le: x_2nd_Flr_SF + 273.5400010000001 t229_leaf2 <= 1641.54
 T229_R2_f117_ge: x_Condition_1_Artery - t229_leaf2 >= 0
 T229_R3_f18_ge: x_2nd_Flr_SF - 1368 t229_leaf3 >= 0
 T229_L3_f37_le: x_Open_Porch_SF + 448.810001 t229_leaf3 <= 488.81
 T229_R4_f18_ge: x_2nd_Flr_SF - 1368 t229_leaf4 >= 0
 T229_R4_f37_ge: x_Open_Porch_SF - 40 t229_leaf4 >= 0
 TREE_230_ONEHOT: t230_leaf0 + t230_leaf1 + t230_leaf2 + t230_leaf3
   + t230_leaf4 + t230_leaf5 + t230_leaf6 + t230_leaf7 = 1
 T230_L0_f4_le: x_Overall_Cond + 999994.000001 t230_leaf0 <= 1e+06
 T230_L0_f96_le: x_Neighborhood_Edwards + 1e-06 t230_leaf0 <= 1
 T230_L0_f33_le: x_Garage_Area + 910.750001 t230_leaf0 <= 1163.75
 T230_L1_f4_le: x_Overall_Cond + 999994.000001 t230_leaf1 <= 1e+06
 T230_L1_f96_le: x_Neighborhood_Edwards + 1e-06 t230_leaf1 <= 1
 T230_R1_f33_ge: x_Garage_Area - 253 t230_leaf1 >= 0
 T230_L2_f4_le: x_Overall_Cond + 999994.000001 t230_leaf2 <= 1e+06
 T230_R2_f96_ge: x_Neighborhood_Edwards - t230_leaf2 >= 0
 T230_L2_f187_le: x_Exterior_2nd_Plywood + 1e-06 t230_leaf2 <= 1
 T230_L3_f4_le: x_Overall_Cond + 999994.000001 t230_leaf3 <= 1e+06
 T230_R3_f96_ge: x_Neighborhood_Edwards - t230_leaf3 >= 0
 T230_R3_f187_ge: x_Exterior_2nd_Plywood - t230_leaf3 >= 0
 T230_R4_f4_ge: x_Overall_Cond - 6 t230_leaf4 >= 0
 T230_L4_f251_le: x_Garage_Type_Detchd + 1e-06 t230_leaf4 <= 1
 T230_L4_f259_le: x_Paved_Drive_Y + 1e-06 t230_leaf4 <= 1
 T230_R5_f4_ge: x_Overall_Cond - 6 t230_leaf5 >= 0
 T230_L5_f251_le: x_Garage_Type_Detchd + 1e-06 t230_leaf5 <= 1
 T230_R5_f259_ge: x_Paved_Drive_Y - t230_leaf5 >= 0
 T230_R6_f4_ge: x_Overall_Cond - 6 t230_leaf6 >= 0
 T230_R6_f251_ge: x_Garage_Type_Detchd - t230_leaf6 >= 0
 T230_L6_f12_le: x_BsmtFin_SF_1 + 988.500001 t230_leaf6 <= 1984.5
 T230_R7_f4_ge: x_Overall_Cond - 6 t230_leaf7 >= 0
 T230_R7_f251_ge: x_Garage_Type_Detchd - t230_leaf7 >= 0
 T230_R7_f12_ge: x_BsmtFin_SF_1 - 996 t230_leaf7 >= 0
 TREE_231_ONEHOT: t231_leaf0 + t231_leaf1 + t231_leaf2 + t231_leaf3
   + t231_leaf4 + t231_leaf5 + t231_leaf6 = 1
 T231_L0_f18_le: x_2nd_Flr_SF + 1060.540001 t231_leaf0 <= 1641.54
 T231_L0_f13_le: x_BsmtFin_SF_2 + 274.050001 t231_leaf0 <= 757.05
 T231_L0_f12_le: x_BsmtFin_SF_1 + 1419.500001 t231_leaf0 <= 1984.5
 T231_L1_f18_le: x_2nd_Flr_SF + 1060.540001 t231_leaf1 <= 1641.54
 T231_L1_f13_le: x_BsmtFin_SF_2 + 274.050001 t231_leaf1 <= 757.05
 T231_R1_f12_ge: x_BsmtFin_SF_1 - 565 t231_leaf1 >= 0
 T231_L2_f18_le: x_2nd_Flr_SF + 1060.540001 t231_leaf2 <= 1641.54
 T231_R2_f13_ge: x_BsmtFin_SF_2 - 483 t231_leaf2 >= 0
 T231_R3_f18_ge: x_2nd_Flr_SF - 581 t231_leaf3 >= 0
 T231_L3_f40_le: x_Screen_Porch + 120.270001 t231_leaf3 <= 268.27
 T231_L3_f5_le: x_Year_Built + 998089.000001 t231_leaf3 <= 1e+06
 T231_R4_f18_ge: x_2nd_Flr_SF - 581 t231_leaf4 >= 0
 T231_L4_f40_le: x_Screen_Porch + 120.270001 t231_leaf4 <= 268.27
 T231_R4_f5_ge: x_Year_Built - 1911 t231_leaf4 >= 0
 T231_R5_f18_ge: x_2nd_Flr_SF - 581 t231_leaf5 >= 0
 T231_R5_f40_ge: x_Screen_Porch - 148 t231_leaf5 >= 0
 T231_L5_f28_le: x_TotRms_AbvGrd + 4.080001 t231_leaf5 <= 12.08
 T231_R6_f18_ge: x_2nd_Flr_SF - 581 t231_leaf6 >= 0
 T231_R6_f40_ge: x_Screen_Porch - 148 t231_leaf6 >= 0
 T231_R6_f28_ge: x_TotRms_AbvGrd - 8 t231_leaf6 >= 0
 TREE_232_ONEHOT: t232_leaf0 + t232_leaf1 + t232_leaf2 + t232_leaf3
   + t232_leaf4 = 1
 T232_L0_f1_le: x_Lot_Area <= 23256.999999
 T232_L0_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t232_leaf0 <= 2514.48
 T232_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t232_leaf0 <= 1
 T232_L1_f1_le: x_Lot_Area <= 23256.999999
 T232_L1_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t232_leaf1 <= 2514.48
 T232_R1_f293_ge: x_Sale_Condition_Abnorml - t232_leaf1 >= 0
 T232_L2_f1_le: x_Lot_Area <= 23256.999999
 T232_R2_f15_ge: x_Total_Bsmt_SF - 2223 t232_leaf2 >= 0
 T232_R3_f1_ge: x_Lot_Area - 16257 t232_leaf3 >= 7000
 T232_L3_f297_le: x_Sale_Condition_Normal + 1e-06 t232_leaf3 <= 1
 T232_R4_f1_ge: x_Lot_Area - 16257 t232_leaf4 >= 7000
 T232_R4_f297_ge: x_Sale_Condition_Normal - t232_leaf4 >= 0
 TREE_233_ONEHOT: t233_leaf0 + t233_leaf1 + t233_leaf2 + t233_leaf3
   + t233_leaf4 = 1
 T233_L0_f18_le: x_2nd_Flr_SF + 345.5400010000001 t233_leaf0 <= 1641.54
 T233_L0_f3_le: x_Overall_Qual + 999993.000001 t233_leaf0 <= 1e+06
 T233_L0_f18_le: x_2nd_Flr_SF + 1053.540001 t233_leaf0 <= 1641.54
 T233_L1_f18_le: x_2nd_Flr_SF + 345.5400010000001 t233_leaf1 <= 1641.54
 T233_L1_f3_le: x_Overall_Qual + 999993.000001 t233_leaf1 <= 1e+06
 T233_R1_f18_ge: x_2nd_Flr_SF - 588 t233_leaf1 >= 0
 T233_L2_f18_le: x_2nd_Flr_SF + 345.5400010000001 t233_leaf2 <= 1641.54
 T233_R2_f3_ge: x_Overall_Qual - 7 t233_leaf2 >= 0
 T233_L2_f6_le: x_Year_Remod/Add + 998021.000001 t233_leaf2 <= 1e+06
 T233_L3_f18_le: x_2nd_Flr_SF + 345.5400010000001 t233_leaf3 <= 1641.54
 T233_R3_f3_ge: x_Overall_Qual - 7 t233_leaf3 >= 0
 T233_R3_f6_ge: x_Year_Remod/Add - 1979 t233_leaf3 >= 0
 T233_R4_f18_ge: x_2nd_Flr_SF - 1296 t233_leaf4 >= 0
 TREE_234_ONEHOT: t234_leaf0 + t234_leaf1 + t234_leaf2 + t234_leaf3
   + t234_leaf4 + t234_leaf5 = 1
 T234_L0_f92_le: x_Neighborhood_BrkSide + 1e-06 t234_leaf0 <= 1
 T234_L0_f119_le: x_Condition_1_Norm + 1e-06 t234_leaf0 <= 1
 T234_L0_f0_le: x_Lot_Frontage + 54.530001 t234_leaf0 <= 175.53
 T234_L1_f92_le: x_Neighborhood_BrkSide + 1e-06 t234_leaf1 <= 1
 T234_L1_f119_le: x_Condition_1_Norm + 1e-06 t234_leaf1 <= 1
 T234_R1_f0_ge: x_Lot_Frontage - 121 t234_leaf1 >= 0
 T234_L2_f92_le: x_Neighborhood_BrkSide + 1e-06 t234_leaf2 <= 1
 T234_R2_f119_ge: x_Condition_1_Norm - t234_leaf2 >= 0
 T234_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t234_leaf2 <= 1
 T234_L3_f92_le: x_Neighborhood_BrkSide + 1e-06 t234_leaf3 <= 1
 T234_R3_f119_ge: x_Condition_1_Norm - t234_leaf3 >= 0
 T234_R3_f95_ge: x_Neighborhood_Crawfor - t234_leaf3 >= 0
 T234_R4_f92_ge: x_Neighborhood_BrkSide - t234_leaf4 >= 0
 T234_L4_f85_le: x_Lot_Config_Inside + 1e-06 t234_leaf4 <= 1
 T234_R5_f92_ge: x_Neighborhood_BrkSide - t234_leaf5 >= 0
 T234_R5_f85_ge: x_Lot_Config_Inside - t234_leaf5 >= 0
 TREE_235_ONEHOT: t235_leaf0 + t235_leaf1 + t235_leaf2 + t235_leaf3
   + t235_leaf4 + t235_leaf5 = 1
 T235_L0_f10_le: x_Bsmt_Qual + 2.000001 t235_leaf0 <= 4
 T235_L0_f33_le: x_Garage_Area + 702.750001 t235_leaf0 <= 1163.75
 T235_L1_f10_le: x_Bsmt_Qual + 2.000001 t235_leaf1 <= 4
 T235_R1_f33_ge: x_Garage_Area - 461 t235_leaf1 >= 0
 T235_L1_f31_le: x_Garage_Yr_Blt + 998047.000001 t235_leaf1 <= 1e+06
 T235_L2_f10_le: x_Bsmt_Qual + 2.000001 t235_leaf2 <= 4
 T235_R2_f33_ge: x_Garage_Area - 461 t235_leaf2 >= 0
 T235_R2_f31_ge: x_Garage_Yr_Blt - 1953 t235_leaf2 >= 0
 T235_R3_f10_ge: x_Bsmt_Qual >= 2
 T235_L3_f15_le: x_Total_Bsmt_SF + 22.4800009999999 t235_leaf3 <= 2514.48
 T235_L3_f7_le: x_Mas_Vnr_Area + 258.070001 t235_leaf3 <= 931.07
 T235_R4_f10_ge: x_Bsmt_Qual >= 2
 T235_L4_f15_le: x_Total_Bsmt_SF + 22.4800009999999 t235_leaf4 <= 2514.48
 T235_R4_f7_ge: x_Mas_Vnr_Area - 673 t235_leaf4 >= 0
 T235_R5_f10_ge: x_Bsmt_Qual >= 2
 T235_R5_f15_ge: x_Total_Bsmt_SF - 2492 t235_leaf5 >= 0
 TREE_236_ONEHOT: t236_leaf0 + t236_leaf1 + t236_leaf2 + t236_leaf3
   + t236_leaf4 + t236_leaf5 = 1
 T236_L0_f1_le: x_Lot_Area <= 14679.999999
 T236_L0_f1_le: x_Lot_Area <= 14229.999999
 T236_L0_f3_le: x_Overall_Qual + 999993.000001 t236_leaf0 <= 1e+06
 T236_L1_f1_le: x_Lot_Area <= 14679.999999
 T236_L1_f1_le: x_Lot_Area <= 14229.999999
 T236_R1_f3_ge: x_Overall_Qual - 7 t236_leaf1 >= 0
 T236_L2_f1_le: x_Lot_Area <= 14679.999999
 T236_R2_f1_ge: x_Lot_Area - 7230 t236_leaf2 >= 7000
 T236_R3_f1_ge: x_Lot_Area - 7680 t236_leaf3 >= 7000
 T236_L3_f37_le: x_Open_Porch_SF + 228.810001 t236_leaf3 <= 488.81
 T236_L3_f283_le: x_Sale_Type_COD + 1e-06 t236_leaf3 <= 1
 T236_R4_f1_ge: x_Lot_Area - 7680 t236_leaf4 >= 7000
 T236_L4_f37_le: x_Open_Porch_SF + 228.810001 t236_leaf4 <= 488.81
 T236_R4_f283_ge: x_Sale_Type_COD - t236_leaf4 >= 0
 T236_R5_f1_ge: x_Lot_Area - 7680 t236_leaf5 >= 7000
 T236_R5_f37_ge: x_Open_Porch_SF - 260 t236_leaf5 >= 0
 TREE_237_ONEHOT: t237_leaf0 + t237_leaf1 + t237_leaf2 + t237_leaf3
   + t237_leaf4 + t237_leaf5 + t237_leaf6 = 1
 T237_L0_f4_le: x_Overall_Cond + 999994.000001 t237_leaf0 <= 1e+06
 T237_L0_f12_le: x_BsmtFin_SF_1 + 466.5000010000001 t237_leaf0 <= 1984.5
 T237_L0_f50_le: x_MS_SubClass_60 + 1e-06 t237_leaf0 <= 1
 T237_L1_f4_le: x_Overall_Cond + 999994.000001 t237_leaf1 <= 1e+06
 T237_L1_f12_le: x_BsmtFin_SF_1 + 466.5000010000001 t237_leaf1 <= 1984.5
 T237_R1_f50_ge: x_MS_SubClass_60 - t237_leaf1 >= 0
 T237_L2_f4_le: x_Overall_Cond + 999994.000001 t237_leaf2 <= 1e+06
 T237_R2_f12_ge: x_BsmtFin_SF_1 - 1518 t237_leaf2 >= 0
 T237_R3_f4_ge: x_Overall_Cond - 6 t237_leaf3 >= 0
 T237_L3_f117_le: x_Condition_1_Artery + 1e-06 t237_leaf3 <= 1
 T237_L3_f25_le: x_Bedroom_AbvGr + 3.000001 t237_leaf3 <= 7
 T237_R4_f4_ge: x_Overall_Cond - 6 t237_leaf4 >= 0
 T237_L4_f117_le: x_Condition_1_Artery + 1e-06 t237_leaf4 <= 1
 T237_R4_f25_ge: x_Bedroom_AbvGr - 4 t237_leaf4 >= 0
 T237_R5_f4_ge: x_Overall_Cond - 6 t237_leaf5 >= 0
 T237_R5_f117_ge: x_Condition_1_Artery - t237_leaf5 >= 0
 T237_L5_f14_le: x_Bsmt_Unf_SF + 995.6800010000001 t237_leaf5 <= 1812.68
 T237_R6_f4_ge: x_Overall_Cond - 6 t237_leaf6 >= 0
 T237_R6_f117_ge: x_Condition_1_Artery - t237_leaf6 >= 0
 T237_R6_f14_ge: x_Bsmt_Unf_SF - 817 t237_leaf6 >= 0
 TREE_238_ONEHOT: t238_leaf0 + t238_leaf1 + t238_leaf2 + t238_leaf3
   + t238_leaf4 = 1
 T238_L0_f1_le: x_Lot_Area + 3124.000001 t238_leaf0 <= 7000
 T238_R1_f1_ge: x_Lot_Area >= 3876
 T238_L1_f119_le: x_Condition_1_Norm + 1e-06 t238_leaf1 <= 1
 T238_L1_f120_le: x_Condition_1_PosA + 1e-06 t238_leaf1 <= 1
 T238_R2_f1_ge: x_Lot_Area >= 3876
 T238_L2_f119_le: x_Condition_1_Norm + 1e-06 t238_leaf2 <= 1
 T238_R2_f120_ge: x_Condition_1_PosA - t238_leaf2 >= 0
 T238_R3_f1_ge: x_Lot_Area >= 3876
 T238_R3_f119_ge: x_Condition_1_Norm - t238_leaf3 >= 0
 T238_L3_f31_le: x_Garage_Yr_Blt + 998076.000001 t238_leaf3 <= 1e+06
 T238_R4_f1_ge: x_Lot_Area >= 3876
 T238_R4_f119_ge: x_Condition_1_Norm - t238_leaf4 >= 0
 T238_R4_f31_ge: x_Garage_Yr_Blt - 1924 t238_leaf4 >= 0
 TREE_239_ONEHOT: t239_leaf0 + t239_leaf1 + t239_leaf2 + t239_leaf3
   + t239_leaf4 = 1
 T239_L0_f40_le: x_Screen_Porch + 164.270001 t239_leaf0 <= 268.27
 T239_L0_f36_le: x_Wood_Deck_SF + 366.900001 t239_leaf0 <= 525.9
 T239_L0_f4_le: x_Overall_Cond + 999994.000001 t239_leaf0 <= 1e+06
 T239_L1_f40_le: x_Screen_Porch + 164.270001 t239_leaf1 <= 268.27
 T239_L1_f36_le: x_Wood_Deck_SF + 366.900001 t239_leaf1 <= 525.9
 T239_R1_f4_ge: x_Overall_Cond - 6 t239_leaf1 >= 0
 T239_L2_f40_le: x_Screen_Porch + 164.270001 t239_leaf2 <= 268.27
 T239_R2_f36_ge: x_Wood_Deck_SF - 159 t239_leaf2 >= 0
 T239_L2_f77_le: x_Land_Contour_Bnk + 1e-06 t239_leaf2 <= 1
 T239_L3_f40_le: x_Screen_Porch + 164.270001 t239_leaf3 <= 268.27
 T239_R3_f36_ge: x_Wood_Deck_SF - 159 t239_leaf3 >= 0
 T239_R3_f77_ge: x_Land_Contour_Bnk - t239_leaf3 >= 0
 T239_R4_f40_ge: x_Screen_Porch - 104 t239_leaf4 >= 0
 TREE_240_ONEHOT: t240_leaf0 + t240_leaf1 + t240_leaf2 + t240_leaf3
   + t240_leaf4 + t240_leaf5 + t240_leaf6 = 1
 T240_L0_f4_le: x_Overall_Cond + 999996.000001 t240_leaf0 <= 1e+06
 T240_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t240_leaf0 <= 1
 T240_L0_f30_le: x_Fireplace_Qu + 3.000001 t240_leaf0 <= 4
 T240_L1_f4_le: x_Overall_Cond + 999996.000001 t240_leaf1 <= 1e+06
 T240_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t240_leaf1 <= 1
 T240_R1_f30_ge: x_Fireplace_Qu >= 1
 T240_L2_f4_le: x_Overall_Cond + 999996.000001 t240_leaf2 <= 1e+06
 T240_R2_f293_ge: x_Sale_Condition_Abnorml - t240_leaf2 >= 0
 T240_R3_f4_ge: x_Overall_Cond - 4 t240_leaf3 >= 0
 T240_L3_f3_le: x_Overall_Qual + 999993.000001 t240_leaf3 <= 1e+06
 T240_L3_f18_le: x_2nd_Flr_SF + 1060.540001 t240_leaf3 <= 1641.54
 T240_R4_f4_ge: x_Overall_Cond - 4 t240_leaf4 >= 0
 T240_L4_f3_le: x_Overall_Qual + 999993.000001 t240_leaf4 <= 1e+06
 T240_R4_f18_ge: x_2nd_Flr_SF - 581 t240_leaf4 >= 0
 T240_R5_f4_ge: x_Overall_Cond - 4 t240_leaf5 >= 0
 T240_R5_f3_ge: x_Overall_Qual - 7 t240_leaf5 >= 0
 T240_L5_f15_le: x_Total_Bsmt_SF + 1733.480001 t240_leaf5 <= 2514.48
 T240_R6_f4_ge: x_Overall_Cond - 4 t240_leaf6 >= 0
 T240_R6_f3_ge: x_Overall_Qual - 7 t240_leaf6 >= 0
 T240_R6_f15_ge: x_Total_Bsmt_SF - 781 t240_leaf6 >= 0
 TREE_241_ONEHOT: t241_leaf0 + t241_leaf1 + t241_leaf2 + t241_leaf3
   + t241_leaf4 + t241_leaf5 = 1
 T241_L0_f6_le: x_Year_Remod/Add + 998016.000001 t241_leaf0 <= 1e+06
 T241_L0_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t241_leaf0 <= 1
 T241_L0_f33_le: x_Garage_Area + 676.750001 t241_leaf0 <= 1163.75
 T241_L1_f6_le: x_Year_Remod/Add + 998016.000001 t241_leaf1 <= 1e+06
 T241_L1_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t241_leaf1 <= 1
 T241_R1_f33_ge: x_Garage_Area - 487 t241_leaf1 >= 0
 T241_L2_f6_le: x_Year_Remod/Add + 998016.000001 t241_leaf2 <= 1e+06
 T241_R2_f211_ge: x_BsmtFin_Type_1_ALQ - t241_leaf2 >= 0
 T241_R3_f6_ge: x_Year_Remod/Add - 1984 t241_leaf3 >= 0
 T241_L3_f18_le: x_2nd_Flr_SF + 663.540001 t241_leaf3 <= 1641.54
 T241_L3_f114_le: x_Neighborhood_StoneBr + 1e-06 t241_leaf3 <= 1
 T241_R4_f6_ge: x_Year_Remod/Add - 1984 t241_leaf4 >= 0
 T241_L4_f18_le: x_2nd_Flr_SF + 663.540001 t241_leaf4 <= 1641.54
 T241_R4_f114_ge: x_Neighborhood_StoneBr - t241_leaf4 >= 0
 T241_R5_f6_ge: x_Year_Remod/Add - 1984 t241_leaf5 >= 0
 T241_R5_f18_ge: x_2nd_Flr_SF - 978 t241_leaf5 >= 0
 TREE_242_ONEHOT: t242_leaf0 + t242_leaf1 + t242_leaf2 + t242_leaf3
   + t242_leaf4 = 1
 T242_L0_f37_le: x_Open_Porch_SF + 293.810001 t242_leaf0 <= 488.81
 T242_L0_f44_le: x_Yr_Sold + 997993.000001 t242_leaf0 <= 1e+06
 T242_L0_f6_le: x_Year_Remod/Add + 998049.000001 t242_leaf0 <= 1e+06
 T242_L1_f37_le: x_Open_Porch_SF + 293.810001 t242_leaf1 <= 488.81
 T242_L1_f44_le: x_Yr_Sold + 997993.000001 t242_leaf1 <= 1e+06
 T242_R1_f6_ge: x_Year_Remod/Add - 1951 t242_leaf1 >= 0
 T242_L2_f37_le: x_Open_Porch_SF + 293.810001 t242_leaf2 <= 488.81
 T242_R2_f44_ge: x_Yr_Sold - 2007 t242_leaf2 >= 0
 T242_L2_f117_le: x_Condition_1_Artery + 1e-06 t242_leaf2 <= 1
 T242_L3_f37_le: x_Open_Porch_SF + 293.810001 t242_leaf3 <= 488.81
 T242_R3_f44_ge: x_Yr_Sold - 2007 t242_leaf3 >= 0
 T242_R3_f117_ge: x_Condition_1_Artery - t242_leaf3 >= 0
 T242_R4_f37_ge: x_Open_Porch_SF - 195 t242_leaf4 >= 0
 TREE_243_ONEHOT: t243_leaf0 + t243_leaf1 + t243_leaf2 + t243_leaf3
   + t243_leaf4 = 1
 T243_L0_f272_le: x_Mo_Sold_2 + 1e-06 t243_leaf0 <= 1
 T243_L0_f3_le: x_Overall_Qual + 999997.000001 t243_leaf0 <= 1e+06
 T243_L1_f272_le: x_Mo_Sold_2 + 1e-06 t243_leaf1 <= 1
 T243_R1_f3_ge: x_Overall_Qual - 3 t243_leaf1 >= 0
 T243_L1_f17_le: x_1st_Flr_SF + 19.90000099999997 t243_leaf1 <= 2534.9
 T243_L2_f272_le: x_Mo_Sold_2 + 1e-06 t243_leaf2 <= 1
 T243_R2_f3_ge: x_Overall_Qual - 3 t243_leaf2 >= 0
 T243_R2_f17_ge: x_1st_Flr_SF - 2515 t243_leaf2 >= 0
 T243_R3_f272_ge: x_Mo_Sold_2 - t243_leaf3 >= 0
 T243_L3_f297_le: x_Sale_Condition_Normal + 1e-06 t243_leaf3 <= 1
 T243_R4_f272_ge: x_Mo_Sold_2 - t243_leaf4 >= 0
 T243_R4_f297_ge: x_Sale_Condition_Normal - t243_leaf4 >= 0
 TREE_244_ONEHOT: t244_leaf0 + t244_leaf1 + t244_leaf2 + t244_leaf3
   + t244_leaf4 = 1
 T244_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t244_leaf0 <= 1
 T244_L0_f17_le: x_1st_Flr_SF + 19.90000099999997 t244_leaf0 <= 2534.9
 T244_L0_f3_le: x_Overall_Qual + 999990.000001 t244_leaf0 <= 1e+06
 T244_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t244_leaf1 <= 1
 T244_L1_f17_le: x_1st_Flr_SF + 19.90000099999997 t244_leaf1 <= 2534.9
 T244_R1_f3_ge: x_Overall_Qual - 10 t244_leaf1 >= 0
 T244_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t244_leaf2 <= 1
 T244_R2_f17_ge: x_1st_Flr_SF - 2515 t244_leaf2 >= 0
 T244_R3_f164_ge: x_Exterior_1st_BrkFace - t244_leaf3 >= 0
 T244_L3_f20_le: x_Gr_Liv_Area + 2173.760001000001 t244_leaf3 <= 3421.76
 T244_R4_f164_ge: x_Exterior_1st_BrkFace - t244_leaf4 >= 0
 T244_R4_f20_ge: x_Gr_Liv_Area - 1248 t244_leaf4 >= 0
 TREE_245_ONEHOT: t245_leaf0 + t245_leaf1 + t245_leaf2 + t245_leaf3
   + t245_leaf4 + t245_leaf5 + t245_leaf6 = 1
 T245_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t245_leaf0 <= 1
 T245_L0_f38_le: x_Enclosed_Porch + 201.750001 t245_leaf0 <= 292.75
 T245_L0_f36_le: x_Wood_Deck_SF + 493.900001 t245_leaf0 <= 525.9
 T245_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t245_leaf1 <= 1
 T245_L1_f38_le: x_Enclosed_Porch + 201.750001 t245_leaf1 <= 292.75
 T245_R1_f36_ge: x_Wood_Deck_SF - 32 t245_leaf1 >= 0
 T245_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t245_leaf2 <= 1
 T245_R2_f38_ge: x_Enclosed_Porch - 91 t245_leaf2 >= 0
 T245_L2_f185_le: x_Exterior_2nd_MetalSd + 1e-06 t245_leaf2 <= 1
 T245_L3_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t245_leaf3 <= 1
 T245_R3_f38_ge: x_Enclosed_Porch - 91 t245_leaf3 >= 0
 T245_R3_f185_ge: x_Exterior_2nd_MetalSd - t245_leaf3 >= 0
 T245_R4_f245_ge: x_FunctioNo_aplical_Typ - t245_leaf4 >= 0
 T245_L4_f36_le: x_Wood_Deck_SF + 251.900001 t245_leaf4 <= 525.9
 T245_L4_f164_le: x_Exterior_1st_BrkFace + 1e-06 t245_leaf4 <= 1
 T245_R5_f245_ge: x_FunctioNo_aplical_Typ - t245_leaf5 >= 0
 T245_L5_f36_le: x_Wood_Deck_SF + 251.900001 t245_leaf5 <= 525.9
 T245_R5_f164_ge: x_Exterior_1st_BrkFace - t245_leaf5 >= 0
 T245_R6_f245_ge: x_FunctioNo_aplical_Typ - t245_leaf6 >= 0
 T245_R6_f36_ge: x_Wood_Deck_SF - 274 t245_leaf6 >= 0
 TREE_246_ONEHOT: t246_leaf0 + t246_leaf1 + t246_leaf2 + t246_leaf3
   + t246_leaf4 = 1
 T246_L0_f1_le: x_Lot_Area + 3000.000001 t246_leaf0 <= 7000
 T246_R1_f1_ge: x_Lot_Area >= 4000
 T246_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t246_leaf1 <= 1
 T246_L1_f3_le: x_Overall_Qual + 999997.000001 t246_leaf1 <= 1e+06
 T246_R2_f1_ge: x_Lot_Area >= 4000
 T246_L2_f293_le: x_Sale_Condition_Abnorml + 1e-06 t246_leaf2 <= 1
 T246_R2_f3_ge: x_Overall_Qual - 3 t246_leaf2 >= 0
 T246_R3_f1_ge: x_Lot_Area >= 4000
 T246_R3_f293_ge: x_Sale_Condition_Abnorml - t246_leaf3 >= 0
 T246_L3_f20_le: x_Gr_Liv_Area + 2588.760001 t246_leaf3 <= 3421.76
 T246_R4_f1_ge: x_Lot_Area >= 4000
 T246_R4_f293_ge: x_Sale_Condition_Abnorml - t246_leaf4 >= 0
 T246_R4_f20_ge: x_Gr_Liv_Area - 833 t246_leaf4 >= 0
 TREE_247_ONEHOT: t247_leaf0 + t247_leaf1 + t247_leaf2 + t247_leaf3
   + t247_leaf4 + t247_leaf5 = 1
 T247_L0_f27_le: x_Kitchen_Qual + 2.000001 t247_leaf0 <= 4
 T247_L0_f233_le: x_Electrical_FuseA + 1e-06 t247_leaf0 <= 1
 T247_L1_f27_le: x_Kitchen_Qual + 2.000001 t247_leaf1 <= 4
 T247_R1_f233_ge: x_Electrical_FuseA - t247_leaf1 >= 0
 T247_R2_f27_ge: x_Kitchen_Qual >= 2
 T247_L2_f33_le: x_Garage_Area + 359.750001 t247_leaf2 <= 1163.75
 T247_L2_f119_le: x_Condition_1_Norm + 1e-06 t247_leaf2 <= 1
 T247_R3_f27_ge: x_Kitchen_Qual >= 2
 T247_L3_f33_le: x_Garage_Area + 359.750001 t247_leaf3 <= 1163.75
 T247_R3_f119_ge: x_Condition_1_Norm - t247_leaf3 >= 0
 T247_R4_f27_ge: x_Kitchen_Qual >= 2
 T247_R4_f33_ge: x_Garage_Area - 804 t247_leaf4 >= 0
 T247_L4_f297_le: x_Sale_Condition_Normal + 1e-06 t247_leaf4 <= 1
 T247_R5_f27_ge: x_Kitchen_Qual >= 2
 T247_R5_f33_ge: x_Garage_Area - 804 t247_leaf5 >= 0
 T247_R5_f297_ge: x_Sale_Condition_Normal - t247_leaf5 >= 0
 TREE_248_ONEHOT: t248_leaf0 + t248_leaf1 + t248_leaf2 + t248_leaf3
   + t248_leaf4 = 1
 T248_L0_f44_le: x_Yr_Sold + 997993.000001 t248_leaf0 <= 1e+06
 T248_R1_f44_ge: x_Yr_Sold - 2007 t248_leaf1 >= 0
 T248_L1_f12_le: x_BsmtFin_SF_1 + 1802.500001 t248_leaf1 <= 1984.5
 T248_L1_f6_le: x_Year_Remod/Add + 997992.000001 t248_leaf1 <= 1e+06
 T248_R2_f44_ge: x_Yr_Sold - 2007 t248_leaf2 >= 0
 T248_L2_f12_le: x_BsmtFin_SF_1 + 1802.500001 t248_leaf2 <= 1984.5
 T248_R2_f6_ge: x_Year_Remod/Add - 2008 t248_leaf2 >= 0
 T248_R3_f44_ge: x_Yr_Sold - 2007 t248_leaf3 >= 0
 T248_R3_f12_ge: x_BsmtFin_SF_1 - 182 t248_leaf3 >= 0
 T248_L3_f29_le: x_Fireplaces + 1.000001 t248_leaf3 <= 3
 T248_R4_f44_ge: x_Yr_Sold - 2007 t248_leaf4 >= 0
 T248_R4_f12_ge: x_BsmtFin_SF_1 - 182 t248_leaf4 >= 0
 T248_R4_f29_ge: x_Fireplaces - 2 t248_leaf4 >= 0
 TREE_249_ONEHOT: t249_leaf0 + t249_leaf1 + t249_leaf2 + t249_leaf3
   + t249_leaf4 + t249_leaf5 = 1
 T249_L0_f5_le: x_Year_Built + 998086.000001 t249_leaf0 <= 1e+06
 T249_L0_f1_le: x_Lot_Area + 1225.000001 t249_leaf0 <= 7000
 T249_L1_f5_le: x_Year_Built + 998086.000001 t249_leaf1 <= 1e+06
 T249_R1_f1_ge: x_Lot_Area >= 5775
 T249_L1_f6_le: x_Year_Remod/Add + 998013.000001 t249_leaf1 <= 1e+06
 T249_L2_f5_le: x_Year_Built + 998086.000001 t249_leaf2 <= 1e+06
 T249_R2_f1_ge: x_Lot_Area >= 5775
 T249_R2_f6_ge: x_Year_Remod/Add - 1987 t249_leaf2 >= 0
 T249_R3_f5_ge: x_Year_Built - 1914 t249_leaf3 >= 0
 T249_L3_f37_le: x_Open_Porch_SF + 4.810001 t249_leaf3 <= 488.81
 T249_L3_f20_le: x_Gr_Liv_Area + 335.7600010000001 t249_leaf3 <= 3421.76
 T249_R4_f5_ge: x_Year_Built - 1914 t249_leaf4 >= 0
 T249_L4_f37_le: x_Open_Porch_SF + 4.810001 t249_leaf4 <= 488.81
 T249_R4_f20_ge: x_Gr_Liv_Area - 3086 t249_leaf4 >= 0
 T249_R5_f5_ge: x_Year_Built - 1914 t249_leaf5 >= 0
 T249_R5_f37_ge: x_Open_Porch_SF - 484 t249_leaf5 >= 0
 TREE_250_ONEHOT: t250_leaf0 + t250_leaf1 + t250_leaf2 + t250_leaf3
   + t250_leaf4 + t250_leaf5 = 1
 T250_L0_f18_le: x_2nd_Flr_SF + 812.540001 t250_leaf0 <= 1641.54
 T250_L0_f20_le: x_Gr_Liv_Area + 1861.760001 t250_leaf0 <= 3421.76
 T250_L0_f117_le: x_Condition_1_Artery + 1e-06 t250_leaf0 <= 1
 T250_L1_f18_le: x_2nd_Flr_SF + 812.540001 t250_leaf1 <= 1641.54
 T250_L1_f20_le: x_Gr_Liv_Area + 1861.760001 t250_leaf1 <= 3421.76
 T250_R1_f117_ge: x_Condition_1_Artery - t250_leaf1 >= 0
 T250_L2_f18_le: x_2nd_Flr_SF + 812.540001 t250_leaf2 <= 1641.54
 T250_R2_f20_ge: x_Gr_Liv_Area - 1560 t250_leaf2 >= 0
 T250_L2_f37_le: x_Open_Porch_SF + 467.810001 t250_leaf2 <= 488.81
 T250_L3_f18_le: x_2nd_Flr_SF + 812.540001 t250_leaf3 <= 1641.54
 T250_R3_f20_ge: x_Gr_Liv_Area - 1560 t250_leaf3 >= 0
 T250_R3_f37_ge: x_Open_Porch_SF - 21 t250_leaf3 >= 0
 T250_R4_f18_ge: x_2nd_Flr_SF - 829 t250_leaf4 >= 0
 T250_L4_f14_le: x_Bsmt_Unf_SF + 1672.680001 t250_leaf4 <= 1812.68
 T250_R5_f18_ge: x_2nd_Flr_SF - 829 t250_leaf5 >= 0
 T250_R5_f14_ge: x_Bsmt_Unf_SF - 140 t250_leaf5 >= 0
 TREE_251_ONEHOT: t251_leaf0 + t251_leaf1 + t251_leaf2 + t251_leaf3
   + t251_leaf4 = 1
 T251_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t251_leaf0 <= 1
 T251_L0_f119_le: x_Condition_1_Norm + 1e-06 t251_leaf0 <= 1
 T251_L0_f121_le: x_Condition_1_PosN + 1e-06 t251_leaf0 <= 1
 T251_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t251_leaf1 <= 1
 T251_L1_f119_le: x_Condition_1_Norm + 1e-06 t251_leaf1 <= 1
 T251_R1_f121_ge: x_Condition_1_PosN - t251_leaf1 >= 0
 T251_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t251_leaf2 <= 1
 T251_R2_f119_ge: x_Condition_1_Norm - t251_leaf2 >= 0
 T251_L2_f68_le: x_Street_Grvl + 1e-06 t251_leaf2 <= 1
 T251_L3_f164_le: x_Exterior_1st_BrkFace + 1e-06 t251_leaf3 <= 1
 T251_R3_f119_ge: x_Condition_1_Norm - t251_leaf3 >= 0
 T251_R3_f68_ge: x_Street_Grvl - t251_leaf3 >= 0
 T251_R4_f164_ge: x_Exterior_1st_BrkFace - t251_leaf4 >= 0
 TREE_252_ONEHOT: t252_leaf0 + t252_leaf1 + t252_leaf2 + t252_leaf3
   + t252_leaf4 + t252_leaf5 + t252_leaf6 + t252_leaf7 = 1
 T252_L0_f16_le: x_Heating_QC + 1.000001 t252_leaf0 <= 4
 T252_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t252_leaf0 <= 1
 T252_L0_f33_le: x_Garage_Area + 681.750001 t252_leaf0 <= 1163.75
 T252_L1_f16_le: x_Heating_QC + 1.000001 t252_leaf1 <= 4
 T252_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t252_leaf1 <= 1
 T252_R1_f33_ge: x_Garage_Area - 482 t252_leaf1 >= 0
 T252_L2_f16_le: x_Heating_QC + 1.000001 t252_leaf2 <= 4
 T252_R2_f297_ge: x_Sale_Condition_Normal - t252_leaf2 >= 0
 T252_L2_f3_le: x_Overall_Qual + 999997.000001 t252_leaf2 <= 1e+06
 T252_L3_f16_le: x_Heating_QC + 1.000001 t252_leaf3 <= 4
 T252_R3_f297_ge: x_Sale_Condition_Normal - t252_leaf3 >= 0
 T252_R3_f3_ge: x_Overall_Qual - 3 t252_leaf3 >= 0
 T252_R4_f16_ge: x_Heating_QC - t252_leaf4 >= 2
 T252_L4_f3_le: x_Overall_Qual + 999993.000001 t252_leaf4 <= 1e+06
 T252_L4_f18_le: x_2nd_Flr_SF + 1040.540001 t252_leaf4 <= 1641.54
 T252_R5_f16_ge: x_Heating_QC - t252_leaf5 >= 2
 T252_L5_f3_le: x_Overall_Qual + 999993.000001 t252_leaf5 <= 1e+06
 T252_R5_f18_ge: x_2nd_Flr_SF - 601 t252_leaf5 >= 0
 T252_R6_f16_ge: x_Heating_QC - t252_leaf6 >= 2
 T252_R6_f3_ge: x_Overall_Qual - 7 t252_leaf6 >= 0
 T252_L6_f292_le: x_Sale_Type_WD_ + 1e-06 t252_leaf6 <= 1
 T252_R7_f16_ge: x_Heating_QC - t252_leaf7 >= 2
 T252_R7_f3_ge: x_Overall_Qual - 7 t252_leaf7 >= 0
 T252_R7_f292_ge: x_Sale_Type_WD_ - t252_leaf7 >= 0
 TREE_253_ONEHOT: t253_leaf0 + t253_leaf1 + t253_leaf2 + t253_leaf3
   + t253_leaf4 + t253_leaf5 + t253_leaf6 + t253_leaf7 = 1
 T253_L0_f4_le: x_Overall_Cond + 999993.000001 t253_leaf0 <= 1e+06
 T253_L0_f96_le: x_Neighborhood_Edwards + 1e-06 t253_leaf0 <= 1
 T253_L0_f1_le: x_Lot_Area <= 14679.999999
 T253_L1_f4_le: x_Overall_Cond + 999993.000001 t253_leaf1 <= 1e+06
 T253_L1_f96_le: x_Neighborhood_Edwards + 1e-06 t253_leaf1 <= 1
 T253_R1_f1_ge: x_Lot_Area - 7680 t253_leaf1 >= 7000
 T253_L2_f4_le: x_Overall_Cond + 999993.000001 t253_leaf2 <= 1e+06
 T253_R2_f96_ge: x_Neighborhood_Edwards - t253_leaf2 >= 0
 T253_L2_f14_le: x_Bsmt_Unf_SF + 1548.680001 t253_leaf2 <= 1812.68
 T253_L3_f4_le: x_Overall_Cond + 999993.000001 t253_leaf3 <= 1e+06
 T253_R3_f96_ge: x_Neighborhood_Edwards - t253_leaf3 >= 0
 T253_R3_f14_ge: x_Bsmt_Unf_SF - 264 t253_leaf3 >= 0
 T253_R4_f4_ge: x_Overall_Cond - 7 t253_leaf4 >= 0
 T253_L4_f247_le: x_Garage_Type_Attchd + 1e-06 t253_leaf4 <= 1
 T253_L4_f1_le: x_Lot_Area <= 8625.999999
 T253_R5_f4_ge: x_Overall_Cond - 7 t253_leaf5 >= 0
 T253_L5_f247_le: x_Garage_Type_Attchd + 1e-06 t253_leaf5 <= 1
 T253_R5_f1_ge: x_Lot_Area - 1626 t253_leaf5 >= 7000
 T253_R6_f4_ge: x_Overall_Cond - 7 t253_leaf6 >= 0
 T253_R6_f247_ge: x_Garage_Type_Attchd - t253_leaf6 >= 0
 T253_L6_f5_le: x_Year_Built + 998054.000001 t253_leaf6 <= 1e+06
 T253_R7_f4_ge: x_Overall_Cond - 7 t253_leaf7 >= 0
 T253_R7_f247_ge: x_Garage_Type_Attchd - t253_leaf7 >= 0
 T253_R7_f5_ge: x_Year_Built - 1946 t253_leaf7 >= 0
 TREE_254_ONEHOT: t254_leaf0 + t254_leaf1 + t254_leaf2 + t254_leaf3
   + t254_leaf4 = 1
 T254_L0_f63_le: x_MS_Zoning_FV + 1e-06 t254_leaf0 <= 1
 T254_L0_f37_le: x_Open_Porch_SF + 197.810001 t254_leaf0 <= 488.81
 T254_L0_f12_le: x_BsmtFin_SF_1 + 420.5000010000001 t254_leaf0 <= 1984.5
 T254_L1_f63_le: x_MS_Zoning_FV + 1e-06 t254_leaf1 <= 1
 T254_L1_f37_le: x_Open_Porch_SF + 197.810001 t254_leaf1 <= 488.81
 T254_R1_f12_ge: x_BsmtFin_SF_1 - 1564 t254_leaf1 >= 0
 T254_L2_f63_le: x_MS_Zoning_FV + 1e-06 t254_leaf2 <= 1
 T254_R2_f37_ge: x_Open_Porch_SF - 291 t254_leaf2 >= 0
 T254_L2_f4_le: x_Overall_Cond + 999994.000001 t254_leaf2 <= 1e+06
 T254_L3_f63_le: x_MS_Zoning_FV + 1e-06 t254_leaf3 <= 1
 T254_R3_f37_ge: x_Open_Porch_SF - 291 t254_leaf3 >= 0
 T254_R3_f4_ge: x_Overall_Cond - 6 t254_leaf3 >= 0
 T254_R4_f63_ge: x_MS_Zoning_FV - t254_leaf4 >= 0
 TREE_255_ONEHOT: t255_leaf0 + t255_leaf1 + t255_leaf2 + t255_leaf3
   + t255_leaf4 + t255_leaf5 + t255_leaf6 = 1
 T255_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t255_leaf0 <= 1
 T255_L0_f14_le: x_Bsmt_Unf_SF + 439.6800010000002 t255_leaf0 <= 1812.68
 T255_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t255_leaf0 <= 1
 T255_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t255_leaf1 <= 1
 T255_L1_f14_le: x_Bsmt_Unf_SF + 439.6800010000002 t255_leaf1 <= 1812.68
 T255_R1_f164_ge: x_Exterior_1st_BrkFace - t255_leaf1 >= 0
 T255_L2_f161_le: x_Exterior_1st_AsbShng + 1e-06 t255_leaf2 <= 1
 T255_R2_f14_ge: x_Bsmt_Unf_SF - 1373 t255_leaf2 >= 0
 T255_L2_f20_le: x_Gr_Liv_Area + 1369.760001 t255_leaf2 <= 3421.76
 T255_L3_f161_le: x_Exterior_1st_AsbShng + 1e-06 t255_leaf3 <= 1
 T255_R3_f14_ge: x_Bsmt_Unf_SF - 1373 t255_leaf3 >= 0
 T255_R3_f20_ge: x_Gr_Liv_Area - 2052 t255_leaf3 >= 0
 T255_R4_f161_ge: x_Exterior_1st_AsbShng - t255_leaf4 >= 0
 T255_L4_f31_le: x_Garage_Yr_Blt + 998079.000001 t255_leaf4 <= 1e+06
 T255_R5_f161_ge: x_Exterior_1st_AsbShng - t255_leaf5 >= 0
 T255_R5_f31_ge: x_Garage_Yr_Blt - 1921 t255_leaf5 >= 0
 T255_L5_f1_le: x_Lot_Area <= 9604.999999
 T255_R6_f161_ge: x_Exterior_1st_AsbShng - t255_leaf6 >= 0
 T255_R6_f31_ge: x_Garage_Yr_Blt - 1921 t255_leaf6 >= 0
 T255_R6_f1_ge: x_Lot_Area - 2605 t255_leaf6 >= 7000
 TREE_256_ONEHOT: t256_leaf0 + t256_leaf1 + t256_leaf2 + t256_leaf3
   + t256_leaf4 + t256_leaf5 + t256_leaf6 = 1
 T256_L0_f1_le: x_Lot_Area + 1000.000001 t256_leaf0 <= 7000
 T256_L0_f20_le: x_Gr_Liv_Area + 2433.760001 t256_leaf0 <= 3421.76
 T256_L0_f16_le: x_Heating_QC + 1e-06 t256_leaf0 <= 4
 T256_L1_f1_le: x_Lot_Area + 1000.000001 t256_leaf1 <= 7000
 T256_L1_f20_le: x_Gr_Liv_Area + 2433.760001 t256_leaf1 <= 3421.76
 T256_R1_f16_ge: x_Heating_QC - 2 t256_leaf1 >= 2
 T256_L2_f1_le: x_Lot_Area + 1000.000001 t256_leaf2 <= 7000
 T256_R2_f20_ge: x_Gr_Liv_Area - 988 t256_leaf2 >= 0
 T256_R3_f1_ge: x_Lot_Area >= 6000
 T256_L3_f92_le: x_Neighborhood_BrkSide + 1e-06 t256_leaf3 <= 1
 T256_L3_f86_le: x_Land_Slope_Gtl + 1e-06 t256_leaf3 <= 1
 T256_R4_f1_ge: x_Lot_Area >= 6000
 T256_L4_f92_le: x_Neighborhood_BrkSide + 1e-06 t256_leaf4 <= 1
 T256_R4_f86_ge: x_Land_Slope_Gtl - t256_leaf4 >= 0
 T256_R5_f1_ge: x_Lot_Area >= 6000
 T256_R5_f92_ge: x_Neighborhood_BrkSide - t256_leaf5 >= 0
 T256_L5_f5_le: x_Year_Built + 998062.000001 t256_leaf5 <= 1e+06
 T256_R6_f1_ge: x_Lot_Area >= 6000
 T256_R6_f92_ge: x_Neighborhood_BrkSide - t256_leaf6 >= 0
 T256_R6_f5_ge: x_Year_Built - 1938 t256_leaf6 >= 0
 TREE_257_ONEHOT: t257_leaf0 + t257_leaf1 + t257_leaf2 + t257_leaf3
   + t257_leaf4 + t257_leaf5 + t257_leaf6 = 1
 T257_L0_f46_le: x_MS_SubClass_30 + 1e-06 t257_leaf0 <= 1
 T257_L0_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t257_leaf0 <= 2514.48
 T257_L0_f6_le: x_Year_Remod/Add + 997993.000001 t257_leaf0 <= 1e+06
 T257_L1_f46_le: x_MS_SubClass_30 + 1e-06 t257_leaf1 <= 1
 T257_L1_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t257_leaf1 <= 2514.48
 T257_R1_f6_ge: x_Year_Remod/Add - 2007 t257_leaf1 >= 0
 T257_L2_f46_le: x_MS_SubClass_30 + 1e-06 t257_leaf2 <= 1
 T257_R2_f15_ge: x_Total_Bsmt_SF - 2136 t257_leaf2 >= 0
 T257_R3_f46_ge: x_MS_SubClass_30 - t257_leaf3 >= 0
 T257_L3_f6_le: x_Year_Remod/Add + 998049.000001 t257_leaf3 <= 1e+06
 T257_L3_f29_le: x_Fireplaces + 2.000001 t257_leaf3 <= 3
 T257_R4_f46_ge: x_MS_SubClass_30 - t257_leaf4 >= 0
 T257_L4_f6_le: x_Year_Remod/Add + 998049.000001 t257_leaf4 <= 1e+06
 T257_R4_f29_ge: x_Fireplaces - t257_leaf4 >= 0
 T257_R5_f46_ge: x_MS_SubClass_30 - t257_leaf5 >= 0
 T257_R5_f6_ge: x_Year_Remod/Add - 1951 t257_leaf5 >= 0
 T257_L5_f31_le: x_Garage_Yr_Blt + 998071.000001 t257_leaf5 <= 1e+06
 T257_R6_f46_ge: x_MS_SubClass_30 - t257_leaf6 >= 0
 T257_R6_f6_ge: x_Year_Remod/Add - 1951 t257_leaf6 >= 0
 T257_R6_f31_ge: x_Garage_Yr_Blt - 1929 t257_leaf6 >= 0
 TREE_258_ONEHOT: t258_leaf0 + t258_leaf1 + t258_leaf2 + t258_leaf3
   + t258_leaf4 + t258_leaf5 = 1
 T258_L0_f209_le: x_Bsmt_Exposure_No + 1e-06 t258_leaf0 <= 1
 T258_L0_f82_le: x_Lot_Config_CulDSac + 1e-06 t258_leaf0 <= 1
 T258_L0_f0_le: x_Lot_Frontage + 15.530001 t258_leaf0 <= 175.53
 T258_L1_f209_le: x_Bsmt_Exposure_No + 1e-06 t258_leaf1 <= 1
 T258_L1_f82_le: x_Lot_Config_CulDSac + 1e-06 t258_leaf1 <= 1
 T258_R1_f0_ge: x_Lot_Frontage - 160 t258_leaf1 >= 0
 T258_L2_f209_le: x_Bsmt_Exposure_No + 1e-06 t258_leaf2 <= 1
 T258_R2_f82_ge: x_Lot_Config_CulDSac - t258_leaf2 >= 0
 T258_R3_f209_ge: x_Bsmt_Exposure_No - t258_leaf3 >= 0
 T258_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t258_leaf3 <= 1
 T258_L3_f16_le: x_Heating_QC + 1.000001 t258_leaf3 <= 4
 T258_R4_f209_ge: x_Bsmt_Exposure_No - t258_leaf4 >= 0
 T258_L4_f95_le: x_Neighborhood_Crawfor + 1e-06 t258_leaf4 <= 1
 T258_R4_f16_ge: x_Heating_QC - t258_leaf4 >= 2
 T258_R5_f209_ge: x_Bsmt_Exposure_No - t258_leaf5 >= 0
 T258_R5_f95_ge: x_Neighborhood_Crawfor - t258_leaf5 >= 0
 TREE_259_ONEHOT: t259_leaf0 + t259_leaf1 + t259_leaf2 + t259_leaf3
   + t259_leaf4 + t259_leaf5 + t259_leaf6 = 1
 T259_L0_f33_le: x_Garage_Area + 352.750001 t259_leaf0 <= 1163.75
 T259_L0_f3_le: x_Overall_Qual + 999993.000001 t259_leaf0 <= 1e+06
 T259_L0_f24_le: x_Half_Bath + 4.000001 t259_leaf0 <= 5
 T259_L1_f33_le: x_Garage_Area + 352.750001 t259_leaf1 <= 1163.75
 T259_L1_f3_le: x_Overall_Qual + 999993.000001 t259_leaf1 <= 1e+06
 T259_R1_f24_ge: x_Half_Bath - t259_leaf1 >= 0
 T259_L2_f33_le: x_Garage_Area + 352.750001 t259_leaf2 <= 1163.75
 T259_R2_f3_ge: x_Overall_Qual - 7 t259_leaf2 >= 0
 T259_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t259_leaf2 <= 1
 T259_L3_f33_le: x_Garage_Area + 352.750001 t259_leaf3 <= 1163.75
 T259_R3_f3_ge: x_Overall_Qual - 7 t259_leaf3 >= 0
 T259_R3_f164_ge: x_Exterior_1st_BrkFace - t259_leaf3 >= 0
 T259_R4_f33_ge: x_Garage_Area - 811 t259_leaf4 >= 0
 T259_L4_f25_le: x_Bedroom_AbvGr + 3.000001 t259_leaf4 <= 7
 T259_L4_f33_le: x_Garage_Area + 9.750001000000111 t259_leaf4 <= 1163.75
 T259_R5_f33_ge: x_Garage_Area - 811 t259_leaf5 >= 0
 T259_L5_f25_le: x_Bedroom_AbvGr + 3.000001 t259_leaf5 <= 7
 T259_R5_f33_ge: x_Garage_Area - 1154 t259_leaf5 >= 0
 T259_R6_f33_ge: x_Garage_Area - 811 t259_leaf6 >= 0
 T259_R6_f25_ge: x_Bedroom_AbvGr - 4 t259_leaf6 >= 0
 TREE_260_ONEHOT: t260_leaf0 + t260_leaf1 + t260_leaf2 + t260_leaf3
   + t260_leaf4 + t260_leaf5 + t260_leaf6 + t260_leaf7 = 1
 T260_L0_f44_le: x_Yr_Sold + 997992.000001 t260_leaf0 <= 1e+06
 T260_L0_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t260_leaf0 <= 1
 T260_L0_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t260_leaf0 <= 1
 T260_L1_f44_le: x_Yr_Sold + 997992.000001 t260_leaf1 <= 1e+06
 T260_L1_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t260_leaf1 <= 1
 T260_R1_f214_ge: x_BsmtFin_Type_1_LwQ - t260_leaf1 >= 0
 T260_L2_f44_le: x_Yr_Sold + 997992.000001 t260_leaf2 <= 1e+06
 T260_R2_f216_ge: x_BsmtFin_Type_1_Rec - t260_leaf2 >= 0
 T260_L2_f24_le: x_Half_Bath + 4.000001 t260_leaf2 <= 5
 T260_L3_f44_le: x_Yr_Sold + 997992.000001 t260_leaf3 <= 1e+06
 T260_R3_f216_ge: x_BsmtFin_Type_1_Rec - t260_leaf3 >= 0
 T260_R3_f24_ge: x_Half_Bath - t260_leaf3 >= 0
 T260_R4_f44_ge: x_Yr_Sold - 2008 t260_leaf4 >= 0
 T260_L4_f293_le: x_Sale_Condition_Abnorml + 1e-06 t260_leaf4 <= 1
 T260_L4_f150_le: x_Roof_Style_Hip + 1e-06 t260_leaf4 <= 1
 T260_R5_f44_ge: x_Yr_Sold - 2008 t260_leaf5 >= 0
 T260_L5_f293_le: x_Sale_Condition_Abnorml + 1e-06 t260_leaf5 <= 1
 T260_R5_f150_ge: x_Roof_Style_Hip - t260_leaf5 >= 0
 T260_R6_f44_ge: x_Yr_Sold - 2008 t260_leaf6 >= 0
 T260_R6_f293_ge: x_Sale_Condition_Abnorml - t260_leaf6 >= 0
 T260_L6_f3_le: x_Overall_Qual + 999995.000001 t260_leaf6 <= 1e+06
 T260_R7_f44_ge: x_Yr_Sold - 2008 t260_leaf7 >= 0
 T260_R7_f293_ge: x_Sale_Condition_Abnorml - t260_leaf7 >= 0
 T260_R7_f3_ge: x_Overall_Qual - 5 t260_leaf7 >= 0
 TREE_261_ONEHOT: t261_leaf0 + t261_leaf1 + t261_leaf2 + t261_leaf3
   + t261_leaf4 = 1
 T261_L0_f37_le: x_Open_Porch_SF + 210.810001 t261_leaf0 <= 488.81
 T261_L0_f88_le: x_Land_Slope_Sev + 1e-06 t261_leaf0 <= 1
 T261_L0_f18_le: x_2nd_Flr_SF + 653.540001 t261_leaf0 <= 1641.54
 T261_L1_f37_le: x_Open_Porch_SF + 210.810001 t261_leaf1 <= 488.81
 T261_L1_f88_le: x_Land_Slope_Sev + 1e-06 t261_leaf1 <= 1
 T261_R1_f18_ge: x_2nd_Flr_SF - 988 t261_leaf1 >= 0
 T261_L2_f37_le: x_Open_Porch_SF + 210.810001 t261_leaf2 <= 488.81
 T261_R2_f88_ge: x_Land_Slope_Sev - t261_leaf2 >= 0
 T261_R3_f37_ge: x_Open_Porch_SF - 278 t261_leaf3 >= 0
 T261_L3_f256_le: x_Garage_Finish_Unf + 1e-06 t261_leaf3 <= 1
 T261_R4_f37_ge: x_Open_Porch_SF - 278 t261_leaf4 >= 0
 T261_R4_f256_ge: x_Garage_Finish_Unf - t261_leaf4 >= 0
 TREE_262_ONEHOT: t262_leaf0 + t262_leaf1 + t262_leaf2 + t262_leaf3
   + t262_leaf4 = 1
 T262_L0_f3_le: x_Overall_Qual + 999997.000001 t262_leaf0 <= 1e+06
 T262_L0_f6_le: x_Year_Remod/Add + 998049.000001 t262_leaf0 <= 1e+06
 T262_L1_f3_le: x_Overall_Qual + 999997.000001 t262_leaf1 <= 1e+06
 T262_R1_f6_ge: x_Year_Remod/Add - 1951 t262_leaf1 >= 0
 T262_R2_f3_ge: x_Overall_Qual - 3 t262_leaf2 >= 0
 T262_L2_f23_le: x_Full_Bath + 3.000001 t262_leaf2 <= 6
 T262_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t262_leaf2 <= 1
 T262_R3_f3_ge: x_Overall_Qual - 3 t262_leaf3 >= 0
 T262_L3_f23_le: x_Full_Bath + 3.000001 t262_leaf3 <= 6
 T262_R3_f245_ge: x_FunctioNo_aplical_Typ - t262_leaf3 >= 0
 T262_R4_f3_ge: x_Overall_Qual - 3 t262_leaf4 >= 0
 T262_R4_f23_ge: x_Full_Bath - 3 t262_leaf4 >= 0
 TREE_263_ONEHOT: t263_leaf0 + t263_leaf1 + t263_leaf2 + t263_leaf3
   + t263_leaf4 + t263_leaf5 + t263_leaf6 = 1
 T263_L0_f272_le: x_Mo_Sold_2 + 1e-06 t263_leaf0 <= 1
 T263_L0_f40_le: x_Screen_Porch + 108.270001 t263_leaf0 <= 268.27
 T263_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t263_leaf0 <= 1
 T263_L1_f272_le: x_Mo_Sold_2 + 1e-06 t263_leaf1 <= 1
 T263_L1_f40_le: x_Screen_Porch + 108.270001 t263_leaf1 <= 268.27
 T263_R1_f95_ge: x_Neighborhood_Crawfor - t263_leaf1 >= 0
 T263_L2_f272_le: x_Mo_Sold_2 + 1e-06 t263_leaf2 <= 1
 T263_R2_f40_ge: x_Screen_Porch - 160 t263_leaf2 >= 0
 T263_R3_f272_ge: x_Mo_Sold_2 - t263_leaf3 >= 0
 T263_L3_f297_le: x_Sale_Condition_Normal + 1e-06 t263_leaf3 <= 1
 T263_L3_f37_le: x_Open_Porch_SF + 438.810001 t263_leaf3 <= 488.81
 T263_R4_f272_ge: x_Mo_Sold_2 - t263_leaf4 >= 0
 T263_L4_f297_le: x_Sale_Condition_Normal + 1e-06 t263_leaf4 <= 1
 T263_R4_f37_ge: x_Open_Porch_SF - 50 t263_leaf4 >= 0
 T263_R5_f272_ge: x_Mo_Sold_2 - t263_leaf5 >= 0
 T263_R5_f297_ge: x_Sale_Condition_Normal - t263_leaf5 >= 0
 T263_L5_f18_le: x_2nd_Flr_SF + 1057.540001 t263_leaf5 <= 1641.54
 T263_R6_f272_ge: x_Mo_Sold_2 - t263_leaf6 >= 0
 T263_R6_f297_ge: x_Sale_Condition_Normal - t263_leaf6 >= 0
 T263_R6_f18_ge: x_2nd_Flr_SF - 584 t263_leaf6 >= 0
 TREE_264_ONEHOT: t264_leaf0 + t264_leaf1 + t264_leaf2 + t264_leaf3
   + t264_leaf4 + t264_leaf5 + t264_leaf6 = 1
 T264_L0_f1_le: x_Lot_Area + 1000.000001 t264_leaf0 <= 7000
 T264_L0_f12_le: x_BsmtFin_SF_1 + 1614.500001 t264_leaf0 <= 1984.5
 T264_L0_f233_le: x_Electrical_FuseA + 1e-06 t264_leaf0 <= 1
 T264_L1_f1_le: x_Lot_Area + 1000.000001 t264_leaf1 <= 7000
 T264_L1_f12_le: x_BsmtFin_SF_1 + 1614.500001 t264_leaf1 <= 1984.5
 T264_R1_f233_ge: x_Electrical_FuseA - t264_leaf1 >= 0
 T264_L2_f1_le: x_Lot_Area + 1000.000001 t264_leaf2 <= 7000
 T264_R2_f12_ge: x_BsmtFin_SF_1 - 370 t264_leaf2 >= 0
 T264_R3_f1_ge: x_Lot_Area >= 6000
 T264_L3_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t264_leaf3 <= 1
 T264_L3_f96_le: x_Neighborhood_Edwards + 1e-06 t264_leaf3 <= 1
 T264_R4_f1_ge: x_Lot_Area >= 6000
 T264_L4_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t264_leaf4 <= 1
 T264_R4_f96_ge: x_Neighborhood_Edwards - t264_leaf4 >= 0
 T264_R5_f1_ge: x_Lot_Area >= 6000
 T264_R5_f207_ge: x_Bsmt_Exposure_Gd - t264_leaf5 >= 0
 T264_L5_f119_le: x_Condition_1_Norm + 1e-06 t264_leaf5 <= 1
 T264_R6_f1_ge: x_Lot_Area >= 6000
 T264_R6_f207_ge: x_Bsmt_Exposure_Gd - t264_leaf6 >= 0
 T264_R6_f119_ge: x_Condition_1_Norm - t264_leaf6 >= 0
 TREE_265_ONEHOT: t265_leaf0 + t265_leaf1 + t265_leaf2 + t265_leaf3
   + t265_leaf4 + t265_leaf5 = 1
 T265_L0_f8_le: x_Exter_Qual + 2.000001 t265_leaf0 <= 4
 T265_L0_f119_le: x_Condition_1_Norm + 1e-06 t265_leaf0 <= 1
 T265_L1_f8_le: x_Exter_Qual + 2.000001 t265_leaf1 <= 4
 T265_R1_f119_ge: x_Condition_1_Norm - t265_leaf1 >= 0
 T265_R2_f8_ge: x_Exter_Qual >= 2
 T265_L2_f1_le: x_Lot_Area + 1024.000001 t265_leaf2 <= 7000
 T265_L2_f33_le: x_Garage_Area + 983.750001 t265_leaf2 <= 1163.75
 T265_R3_f8_ge: x_Exter_Qual >= 2
 T265_L3_f1_le: x_Lot_Area + 1024.000001 t265_leaf3 <= 7000
 T265_R3_f33_ge: x_Garage_Area - 180 t265_leaf3 >= 0
 T265_R4_f8_ge: x_Exter_Qual >= 2
 T265_R4_f1_ge: x_Lot_Area >= 5976
 T265_L4_f93_le: x_Neighborhood_ClearCr + 1e-06 t265_leaf4 <= 1
 T265_R5_f8_ge: x_Exter_Qual >= 2
 T265_R5_f1_ge: x_Lot_Area >= 5976
 T265_R5_f93_ge: x_Neighborhood_ClearCr - t265_leaf5 >= 0
 TREE_266_ONEHOT: t266_leaf0 + t266_leaf1 + t266_leaf2 + t266_leaf3
   + t266_leaf4 + t266_leaf5 + t266_leaf6 = 1
 T266_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t266_leaf0 <= 1
 T266_L0_f15_le: x_Total_Bsmt_SF + 588.4800010000001 t266_leaf0 <= 2514.48
 T266_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t266_leaf0 <= 1
 T266_L1_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t266_leaf1 <= 1
 T266_L1_f15_le: x_Total_Bsmt_SF + 588.4800010000001 t266_leaf1 <= 2514.48
 T266_R1_f177_ge: x_Exterior_2nd_AsbShng - t266_leaf1 >= 0
 T266_L2_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t266_leaf2 <= 1
 T266_R2_f15_ge: x_Total_Bsmt_SF - 1926 t266_leaf2 >= 0
 T266_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t266_leaf2 <= 1
 T266_L3_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t266_leaf3 <= 1
 T266_R3_f15_ge: x_Total_Bsmt_SF - 1926 t266_leaf3 >= 0
 T266_R3_f297_ge: x_Sale_Condition_Normal - t266_leaf3 >= 0
 T266_R4_f207_ge: x_Bsmt_Exposure_Gd - t266_leaf4 >= 0
 T266_L4_f21_le: x_Bsmt_Full_Bath + 0.020001 t266_leaf4 <= 2.02
 T266_L4_f36_le: x_Wood_Deck_SF + 14.90000099999997 t266_leaf4 <= 525.9
 T266_R5_f207_ge: x_Bsmt_Exposure_Gd - t266_leaf5 >= 0
 T266_L5_f21_le: x_Bsmt_Full_Bath + 0.020001 t266_leaf5 <= 2.02
 T266_R5_f36_ge: x_Wood_Deck_SF - 511 t266_leaf5 >= 0
 T266_R6_f207_ge: x_Bsmt_Exposure_Gd - t266_leaf6 >= 0
 T266_R6_f21_ge: x_Bsmt_Full_Bath - 2 t266_leaf6 >= 0
 TREE_267_ONEHOT: t267_leaf0 + t267_leaf1 + t267_leaf2 + t267_leaf3
   + t267_leaf4 + t267_leaf5 + t267_leaf6 = 1
 T267_L0_f33_le: x_Garage_Area + 948.750001 t267_leaf0 <= 1163.75
 T267_L0_f1_le: x_Lot_Area <= 8126.999999
 T267_L1_f33_le: x_Garage_Area + 948.750001 t267_leaf1 <= 1163.75
 T267_R1_f1_ge: x_Lot_Area - 1127 t267_leaf1 >= 7000
 T267_L1_f38_le: x_Enclosed_Porch + 179.750001 t267_leaf1 <= 292.75
 T267_L2_f33_le: x_Garage_Area + 948.750001 t267_leaf2 <= 1163.75
 T267_R2_f1_ge: x_Lot_Area - 1127 t267_leaf2 >= 7000
 T267_R2_f38_ge: x_Enclosed_Porch - 113 t267_leaf2 >= 0
 T267_R3_f33_ge: x_Garage_Area - 215 t267_leaf3 >= 0
 T267_L3_f4_le: x_Overall_Cond + 999994.000001 t267_leaf3 <= 1e+06
 T267_L3_f37_le: x_Open_Porch_SF + 470.810001 t267_leaf3 <= 488.81
 T267_R4_f33_ge: x_Garage_Area - 215 t267_leaf4 >= 0
 T267_L4_f4_le: x_Overall_Cond + 999994.000001 t267_leaf4 <= 1e+06
 T267_R4_f37_ge: x_Open_Porch_SF - 18 t267_leaf4 >= 0
 T267_R5_f33_ge: x_Garage_Area - 215 t267_leaf5 >= 0
 T267_R5_f4_ge: x_Overall_Cond - 6 t267_leaf5 >= 0
 T267_L5_f258_le: x_Paved_Drive_P + 1e-06 t267_leaf5 <= 1
 T267_R6_f33_ge: x_Garage_Area - 215 t267_leaf6 >= 0
 T267_R6_f4_ge: x_Overall_Cond - 6 t267_leaf6 >= 0
 T267_R6_f258_ge: x_Paved_Drive_P - t267_leaf6 >= 0
 TREE_268_ONEHOT: t268_leaf0 + t268_leaf1 + t268_leaf2 + t268_leaf3
   + t268_leaf4 + t268_leaf5 = 1
 T268_L0_f4_le: x_Overall_Cond + 999993.000001 t268_leaf0 <= 1e+06
 T268_L0_f113_le: x_Neighborhood_Somerst + 1e-06 t268_leaf0 <= 1
 T268_L0_f37_le: x_Open_Porch_SF + 196.810001 t268_leaf0 <= 488.81
 T268_L1_f4_le: x_Overall_Cond + 999993.000001 t268_leaf1 <= 1e+06
 T268_L1_f113_le: x_Neighborhood_Somerst + 1e-06 t268_leaf1 <= 1
 T268_R1_f37_ge: x_Open_Porch_SF - 292 t268_leaf1 >= 0
 T268_L2_f4_le: x_Overall_Cond + 999993.000001 t268_leaf2 <= 1e+06
 T268_R2_f113_ge: x_Neighborhood_Somerst - t268_leaf2 >= 0
 T268_R3_f4_ge: x_Overall_Cond - 7 t268_leaf3 >= 0
 T268_L3_f17_le: x_1st_Flr_SF + 1581.900001 t268_leaf3 <= 2534.9
 T268_R4_f4_ge: x_Overall_Cond - 7 t268_leaf4 >= 0
 T268_R4_f17_ge: x_1st_Flr_SF - 953 t268_leaf4 >= 0
 T268_L4_f3_le: x_Overall_Qual + 999993.000001 t268_leaf4 <= 1e+06
 T268_R5_f4_ge: x_Overall_Cond - 7 t268_leaf5 >= 0
 T268_R5_f17_ge: x_1st_Flr_SF - 953 t268_leaf5 >= 0
 T268_R5_f3_ge: x_Overall_Qual - 7 t268_leaf5 >= 0
 TREE_269_ONEHOT: t269_leaf0 + t269_leaf1 + t269_leaf2 + t269_leaf3
   + t269_leaf4 = 1
 T269_L0_f21_le: x_Bsmt_Full_Bath + 0.020001 t269_leaf0 <= 2.02
 T269_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t269_leaf0 <= 1
 T269_L0_f18_le: x_2nd_Flr_SF + 214.5400010000001 t269_leaf0 <= 1641.54
 T269_L1_f21_le: x_Bsmt_Full_Bath + 0.020001 t269_leaf1 <= 2.02
 T269_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t269_leaf1 <= 1
 T269_R1_f18_ge: x_2nd_Flr_SF - 1427 t269_leaf1 >= 0
 T269_L2_f21_le: x_Bsmt_Full_Bath + 0.020001 t269_leaf2 <= 2.02
 T269_R2_f161_ge: x_Exterior_1st_AsbShng - t269_leaf2 >= 0
 T269_L2_f217_le: x_BsmtFin_Type_1_Unf + 1e-06 t269_leaf2 <= 1
 T269_L3_f21_le: x_Bsmt_Full_Bath + 0.020001 t269_leaf3 <= 2.02
 T269_R3_f161_ge: x_Exterior_1st_AsbShng - t269_leaf3 >= 0
 T269_R3_f217_ge: x_BsmtFin_Type_1_Unf - t269_leaf3 >= 0
 T269_R4_f21_ge: x_Bsmt_Full_Bath - 2 t269_leaf4 >= 0
 TREE_270_ONEHOT: t270_leaf0 + t270_leaf1 + t270_leaf2 + t270_leaf3
   + t270_leaf4 + t270_leaf5 + t270_leaf6 = 1
 T270_L0_f9_le: x_Exter_Cond + 2.000001 t270_leaf0 <= 4
 T270_L0_f6_le: x_Year_Remod/Add + 998048.000001 t270_leaf0 <= 1e+06
 T270_L1_f9_le: x_Exter_Cond + 2.000001 t270_leaf1 <= 4
 T270_R1_f6_ge: x_Year_Remod/Add - 1952 t270_leaf1 >= 0
 T270_L1_f85_le: x_Lot_Config_Inside + 1e-06 t270_leaf1 <= 1
 T270_L2_f9_le: x_Exter_Cond + 2.000001 t270_leaf2 <= 4
 T270_R2_f6_ge: x_Year_Remod/Add - 1952 t270_leaf2 >= 0
 T270_R2_f85_ge: x_Lot_Config_Inside - t270_leaf2 >= 0
 T270_R3_f9_ge: x_Exter_Cond - 3 t270_leaf3 >= -1
 T270_L3_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t270_leaf3 <= 1984.5
 T270_L3_f4_le: x_Overall_Cond + 999994.000001 t270_leaf3 <= 1e+06
 T270_R4_f9_ge: x_Exter_Cond - 3 t270_leaf4 >= -1
 T270_L4_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t270_leaf4 <= 1984.5
 T270_R4_f4_ge: x_Overall_Cond - 6 t270_leaf4 >= 0
 T270_R5_f9_ge: x_Exter_Cond - 3 t270_leaf5 >= -1
 T270_R5_f12_ge: x_BsmtFin_SF_1 - 1732 t270_leaf5 >= 0
 T270_L5_f80_le: x_Land_Contour_Lvl + 1e-06 t270_leaf5 <= 1
 T270_R6_f9_ge: x_Exter_Cond - 3 t270_leaf6 >= -1
 T270_R6_f12_ge: x_BsmtFin_SF_1 - 1732 t270_leaf6 >= 0
 T270_R6_f80_ge: x_Land_Contour_Lvl - t270_leaf6 >= 0
 TREE_271_ONEHOT: t271_leaf0 + t271_leaf1 + t271_leaf2 + t271_leaf3
   + t271_leaf4 = 1
 T271_L0_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t271_leaf0 <= 1984.5
 T271_L0_f3_le: x_Overall_Qual + 999997.000001 t271_leaf0 <= 1e+06
 T271_L1_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t271_leaf1 <= 1984.5
 T271_R1_f3_ge: x_Overall_Qual - 3 t271_leaf1 >= 0
 T271_L1_f298_le: x_Sale_Condition_Partial + 1e-06 t271_leaf1 <= 1
 T271_L2_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t271_leaf2 <= 1984.5
 T271_R2_f3_ge: x_Overall_Qual - 3 t271_leaf2 >= 0
 T271_R2_f298_ge: x_Sale_Condition_Partial - t271_leaf2 >= 0
 T271_R3_f12_ge: x_BsmtFin_SF_1 - 1732 t271_leaf3 >= 0
 T271_L3_f80_le: x_Land_Contour_Lvl + 1e-06 t271_leaf3 <= 1
 T271_R4_f12_ge: x_BsmtFin_SF_1 - 1732 t271_leaf4 >= 0
 T271_R4_f80_ge: x_Land_Contour_Lvl - t271_leaf4 >= 0
 TREE_272_ONEHOT: t272_leaf0 + t272_leaf1 + t272_leaf2 + t272_leaf3
   + t272_leaf4 = 1
 T272_L0_f33_le: x_Garage_Area + 9.750001000000111 t272_leaf0 <= 1163.75
 T272_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t272_leaf0 <= 1
 T272_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t272_leaf0 <= 1
 T272_L1_f33_le: x_Garage_Area + 9.750001000000111 t272_leaf1 <= 1163.75
 T272_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t272_leaf1 <= 1
 T272_R1_f293_ge: x_Sale_Condition_Abnorml - t272_leaf1 >= 0
 T272_L2_f33_le: x_Garage_Area + 9.750001000000111 t272_leaf2 <= 1163.75
 T272_R2_f177_ge: x_Exterior_2nd_AsbShng - t272_leaf2 >= 0
 T272_L2_f233_le: x_Electrical_FuseA + 1e-06 t272_leaf2 <= 1
 T272_L3_f33_le: x_Garage_Area + 9.750001000000111 t272_leaf3 <= 1163.75
 T272_R3_f177_ge: x_Exterior_2nd_AsbShng - t272_leaf3 >= 0
 T272_R3_f233_ge: x_Electrical_FuseA - t272_leaf3 >= 0
 T272_R4_f33_ge: x_Garage_Area - 1154 t272_leaf4 >= 0
 TREE_273_ONEHOT: t273_leaf0 + t273_leaf1 + t273_leaf2 + t273_leaf3
   + t273_leaf4 + t273_leaf5 + t273_leaf6 = 1
 T273_L0_f44_le: x_Yr_Sold + 997992.000001 t273_leaf0 <= 1e+06
 T273_L0_f3_le: x_Overall_Qual + 999993.000001 t273_leaf0 <= 1e+06
 T273_L0_f15_le: x_Total_Bsmt_SF + 1964.480001 t273_leaf0 <= 2514.48
 T273_L1_f44_le: x_Yr_Sold + 997992.000001 t273_leaf1 <= 1e+06
 T273_L1_f3_le: x_Overall_Qual + 999993.000001 t273_leaf1 <= 1e+06
 T273_R1_f15_ge: x_Total_Bsmt_SF - 550 t273_leaf1 >= 0
 T273_L2_f44_le: x_Yr_Sold + 997992.000001 t273_leaf2 <= 1e+06
 T273_R2_f3_ge: x_Overall_Qual - 7 t273_leaf2 >= 0
 T273_L2_f12_le: x_BsmtFin_SF_1 + 1055.500001 t273_leaf2 <= 1984.5
 T273_L3_f44_le: x_Yr_Sold + 997992.000001 t273_leaf3 <= 1e+06
 T273_R3_f3_ge: x_Overall_Qual - 7 t273_leaf3 >= 0
 T273_R3_f12_ge: x_BsmtFin_SF_1 - 929 t273_leaf3 >= 0
 T273_R4_f44_ge: x_Yr_Sold - 2008 t273_leaf4 >= 0
 T273_L4_f20_le: x_Gr_Liv_Area + 549.7600010000001 t273_leaf4 <= 3421.76
 T273_L4_f38_le: x_Enclosed_Porch + 58.750001 t273_leaf4 <= 292.75
 T273_R5_f44_ge: x_Yr_Sold - 2008 t273_leaf5 >= 0
 T273_L5_f20_le: x_Gr_Liv_Area + 549.7600010000001 t273_leaf5 <= 3421.76
 T273_R5_f38_ge: x_Enclosed_Porch - 234 t273_leaf5 >= 0
 T273_R6_f44_ge: x_Yr_Sold - 2008 t273_leaf6 >= 0
 T273_R6_f20_ge: x_Gr_Liv_Area - 2872 t273_leaf6 >= 0
 TREE_274_ONEHOT: t274_leaf0 + t274_leaf1 + t274_leaf2 + t274_leaf3
   + t274_leaf4 + t274_leaf5 + t274_leaf6 = 1
 T274_L0_f1_le: x_Lot_Area + 2500.000001 t274_leaf0 <= 7000
 T274_L0_f20_le: x_Gr_Liv_Area + 1863.760001 t274_leaf0 <= 3421.76
 T274_L0_f20_le: x_Gr_Liv_Area + 1976.760001 t274_leaf0 <= 3421.76
 T274_L1_f1_le: x_Lot_Area + 2500.000001 t274_leaf1 <= 7000
 T274_L1_f20_le: x_Gr_Liv_Area + 1863.760001 t274_leaf1 <= 3421.76
 T274_R1_f20_ge: x_Gr_Liv_Area - 1445 t274_leaf1 >= 0
 T274_L2_f1_le: x_Lot_Area + 2500.000001 t274_leaf2 <= 7000
 T274_R2_f20_ge: x_Gr_Liv_Area - 1558 t274_leaf2 >= 0
 T274_R3_f1_ge: x_Lot_Area >= 4500
 T274_L3_f6_le: x_Year_Remod/Add + 998006.000001 t274_leaf3 <= 1e+06
 T274_L3_f296_le: x_Sale_Condition_Family + 1e-06 t274_leaf3 <= 1
 T274_R4_f1_ge: x_Lot_Area >= 4500
 T274_L4_f6_le: x_Year_Remod/Add + 998006.000001 t274_leaf4 <= 1e+06
 T274_R4_f296_ge: x_Sale_Condition_Family - t274_leaf4 >= 0
 T274_R5_f1_ge: x_Lot_Area >= 4500
 T274_R5_f6_ge: x_Year_Remod/Add - 1994 t274_leaf5 >= 0
 T274_L5_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t274_leaf5 <= 2514.48
 T274_R6_f1_ge: x_Lot_Area >= 4500
 T274_R6_f6_ge: x_Year_Remod/Add - 1994 t274_leaf6 >= 0
 T274_R6_f15_ge: x_Total_Bsmt_SF - 2136 t274_leaf6 >= 0
 TREE_275_ONEHOT: t275_leaf0 + t275_leaf1 + t275_leaf2 + t275_leaf3
   + t275_leaf4 = 1
 T275_L0_f20_le: x_Gr_Liv_Area + 2653.760001 t275_leaf0 <= 3421.76
 T275_L0_f38_le: x_Enclosed_Porch + 258.750001 t275_leaf0 <= 292.75
 T275_L1_f20_le: x_Gr_Liv_Area + 2653.760001 t275_leaf1 <= 3421.76
 T275_R1_f38_ge: x_Enclosed_Porch - 34 t275_leaf1 >= 0
 T275_R2_f20_ge: x_Gr_Liv_Area - 768 t275_leaf2 >= 0
 T275_L2_f0_le: x_Lot_Frontage + 45.530001 t275_leaf2 <= 175.53
 T275_L2_f277_le: x_Mo_Sold_7 + 1e-06 t275_leaf2 <= 1
 T275_R3_f20_ge: x_Gr_Liv_Area - 768 t275_leaf3 >= 0
 T275_L3_f0_le: x_Lot_Frontage + 45.530001 t275_leaf3 <= 175.53
 T275_R3_f277_ge: x_Mo_Sold_7 - t275_leaf3 >= 0
 T275_R4_f20_ge: x_Gr_Liv_Area - 768 t275_leaf4 >= 0
 T275_R4_f0_ge: x_Lot_Frontage - 130 t275_leaf4 >= 0
 TREE_276_ONEHOT: t276_leaf0 + t276_leaf1 + t276_leaf2 + t276_leaf3
   + t276_leaf4 + t276_leaf5 = 1
 T276_L0_f118_le: x_Condition_1_Feedr + 1e-06 t276_leaf0 <= 1
 T276_L0_f1_le: x_Lot_Area + 1000.000001 t276_leaf0 <= 7000
 T276_L0_f209_le: x_Bsmt_Exposure_No + 1e-06 t276_leaf0 <= 1
 T276_L1_f118_le: x_Condition_1_Feedr + 1e-06 t276_leaf1 <= 1
 T276_L1_f1_le: x_Lot_Area + 1000.000001 t276_leaf1 <= 7000
 T276_R1_f209_ge: x_Bsmt_Exposure_No - t276_leaf1 >= 0
 T276_L2_f118_le: x_Condition_1_Feedr + 1e-06 t276_leaf2 <= 1
 T276_R2_f1_ge: x_Lot_Area >= 6000
 T276_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t276_leaf2 <= 1
 T276_L3_f118_le: x_Condition_1_Feedr + 1e-06 t276_leaf3 <= 1
 T276_R3_f1_ge: x_Lot_Area >= 6000
 T276_R3_f164_ge: x_Exterior_1st_BrkFace - t276_leaf3 >= 0
 T276_R4_f118_ge: x_Condition_1_Feedr - t276_leaf4 >= 0
 T276_L4_f38_le: x_Enclosed_Porch + 100.750001 t276_leaf4 <= 292.75
 T276_R5_f118_ge: x_Condition_1_Feedr - t276_leaf5 >= 0
 T276_R5_f38_ge: x_Enclosed_Porch - 192 t276_leaf5 >= 0
 TREE_277_ONEHOT: t277_leaf0 + t277_leaf1 + t277_leaf2 + t277_leaf3
   + t277_leaf4 + t277_leaf5 = 1
 T277_L0_f4_le: x_Overall_Cond + 999995.000001 t277_leaf0 <= 1e+06
 T277_L0_f14_le: x_Bsmt_Unf_SF + 1338.680001 t277_leaf0 <= 1812.68
 T277_L1_f4_le: x_Overall_Cond + 999995.000001 t277_leaf1 <= 1e+06
 T277_R1_f14_ge: x_Bsmt_Unf_SF - 474 t277_leaf1 >= 0
 T277_L1_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t277_leaf1 <= 1
 T277_L2_f4_le: x_Overall_Cond + 999995.000001 t277_leaf2 <= 1e+06
 T277_R2_f14_ge: x_Bsmt_Unf_SF - 474 t277_leaf2 >= 0
 T277_R2_f211_ge: x_BsmtFin_Type_1_ALQ - t277_leaf2 >= 0
 T277_R3_f4_ge: x_Overall_Cond - 5 t277_leaf3 >= 0
 T277_L3_f0_le: x_Lot_Frontage + 23.530001 t277_leaf3 <= 175.53
 T277_L3_f1_le: x_Lot_Area <= 17226.999999
 T277_R4_f4_ge: x_Overall_Cond - 5 t277_leaf4 >= 0
 T277_L4_f0_le: x_Lot_Frontage + 23.530001 t277_leaf4 <= 175.53
 T277_R4_f1_ge: x_Lot_Area - 10227 t277_leaf4 >= 7000
 T277_R5_f4_ge: x_Overall_Cond - 5 t277_leaf5 >= 0
 T277_R5_f0_ge: x_Lot_Frontage - 152 t277_leaf5 >= 0
 TREE_278_ONEHOT: t278_leaf0 + t278_leaf1 + t278_leaf2 + t278_leaf3 = 1
 T278_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t278_leaf0 <= 1
 T278_L0_f3_le: x_Overall_Qual + 999997.000001 t278_leaf0 <= 1e+06
 T278_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t278_leaf1 <= 1
 T278_R1_f3_ge: x_Overall_Qual - 3 t278_leaf1 >= 0
 T278_L1_f277_le: x_Mo_Sold_7 + 1e-06 t278_leaf1 <= 1
 T278_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t278_leaf2 <= 1
 T278_R2_f3_ge: x_Overall_Qual - 3 t278_leaf2 >= 0
 T278_R2_f277_ge: x_Mo_Sold_7 - t278_leaf2 >= 0
 T278_R3_f164_ge: x_Exterior_1st_BrkFace - t278_leaf3 >= 0
 TREE_279_ONEHOT: t279_leaf0 + t279_leaf1 + t279_leaf2 + t279_leaf3
   + t279_leaf4 + t279_leaf5 = 1
 T279_L0_f3_le: x_Overall_Qual + 999996.000001 t279_leaf0 <= 1e+06
 T279_L0_f259_le: x_Paved_Drive_Y + 1e-06 t279_leaf0 <= 1
 T279_L1_f3_le: x_Overall_Qual + 999996.000001 t279_leaf1 <= 1e+06
 T279_R1_f259_ge: x_Paved_Drive_Y - t279_leaf1 >= 0
 T279_R2_f3_ge: x_Overall_Qual - 4 t279_leaf2 >= 0
 T279_L2_f296_le: x_Sale_Condition_Family + 1e-06 t279_leaf2 <= 1
 T279_L2_f6_le: x_Year_Remod/Add + 997993.000001 t279_leaf2 <= 1e+06
 T279_R3_f3_ge: x_Overall_Qual - 4 t279_leaf3 >= 0
 T279_L3_f296_le: x_Sale_Condition_Family + 1e-06 t279_leaf3 <= 1
 T279_R3_f6_ge: x_Year_Remod/Add - 2007 t279_leaf3 >= 0
 T279_R4_f3_ge: x_Overall_Qual - 4 t279_leaf4 >= 0
 T279_R4_f296_ge: x_Sale_Condition_Family - t279_leaf4 >= 0
 T279_L4_f195_le: x_Mas_Vnr_Type_BrkFace + 1e-06 t279_leaf4 <= 1
 T279_R5_f3_ge: x_Overall_Qual - 4 t279_leaf5 >= 0
 T279_R5_f296_ge: x_Sale_Condition_Family - t279_leaf5 >= 0
 T279_R5_f195_ge: x_Mas_Vnr_Type_BrkFace - t279_leaf5 >= 0
 TREE_280_ONEHOT: t280_leaf0 + t280_leaf1 + t280_leaf2 + t280_leaf3
   + t280_leaf4 + t280_leaf5 + t280_leaf6 = 1
 T280_L0_f119_le: x_Condition_1_Norm + 1e-06 t280_leaf0 <= 1
 T280_L0_f32_le: x_Garage_Cars + 1.000001 t280_leaf0 <= 4
 T280_L0_f3_le: x_Overall_Qual + 999993.000001 t280_leaf0 <= 1e+06
 T280_L1_f119_le: x_Condition_1_Norm + 1e-06 t280_leaf1 <= 1
 T280_L1_f32_le: x_Garage_Cars + 1.000001 t280_leaf1 <= 4
 T280_R1_f3_ge: x_Overall_Qual - 7 t280_leaf1 >= 0
 T280_L2_f119_le: x_Condition_1_Norm + 1e-06 t280_leaf2 <= 1
 T280_R2_f32_ge: x_Garage_Cars - 3 t280_leaf2 >= 0
 T280_R3_f119_ge: x_Condition_1_Norm - t280_leaf3 >= 0
 T280_L3_f31_le: x_Garage_Yr_Blt + 998072.000001 t280_leaf3 <= 1e+06
 T280_L3_f208_le: x_Bsmt_Exposure_Mn + 1e-06 t280_leaf3 <= 1
 T280_R4_f119_ge: x_Condition_1_Norm - t280_leaf4 >= 0
 T280_L4_f31_le: x_Garage_Yr_Blt + 998072.000001 t280_leaf4 <= 1e+06
 T280_R4_f208_ge: x_Bsmt_Exposure_Mn - t280_leaf4 >= 0
 T280_R5_f119_ge: x_Condition_1_Norm - t280_leaf5 >= 0
 T280_R5_f31_ge: x_Garage_Yr_Blt - 1928 t280_leaf5 >= 0
 T280_L5_f3_le: x_Overall_Qual + 999997.000001 t280_leaf5 <= 1e+06
 T280_R6_f119_ge: x_Condition_1_Norm - t280_leaf6 >= 0
 T280_R6_f31_ge: x_Garage_Yr_Blt - 1928 t280_leaf6 >= 0
 T280_R6_f3_ge: x_Overall_Qual - 3 t280_leaf6 >= 0
 TREE_281_ONEHOT: t281_leaf0 + t281_leaf1 + t281_leaf2 + t281_leaf3
   + t281_leaf4 + t281_leaf5 + t281_leaf6 = 1
 T281_L0_f3_le: x_Overall_Qual + 999993.000001 t281_leaf0 <= 1e+06
 T281_L0_f20_le: x_Gr_Liv_Area + 2275.760001000001 t281_leaf0 <= 3421.76
 T281_L0_f292_le: x_Sale_Type_WD_ + 1e-06 t281_leaf0 <= 1
 T281_L1_f3_le: x_Overall_Qual + 999993.000001 t281_leaf1 <= 1e+06
 T281_L1_f20_le: x_Gr_Liv_Area + 2275.760001000001 t281_leaf1 <= 3421.76
 T281_R1_f292_ge: x_Sale_Type_WD_ - t281_leaf1 >= 0
 T281_L2_f3_le: x_Overall_Qual + 999993.000001 t281_leaf2 <= 1e+06
 T281_R2_f20_ge: x_Gr_Liv_Area - 1146 t281_leaf2 >= 0
 T281_L2_f11_le: x_Bsmt_Cond + 1.000001 t281_leaf2 <= 4
 T281_L3_f3_le: x_Overall_Qual + 999993.000001 t281_leaf3 <= 1e+06
 T281_R3_f20_ge: x_Gr_Liv_Area - 1146 t281_leaf3 >= 0
 T281_R3_f11_ge: x_Bsmt_Cond - 4 t281_leaf3 >= -1
 T281_R4_f3_ge: x_Overall_Qual - 7 t281_leaf4 >= 0
 T281_L4_f96_le: x_Neighborhood_Edwards + 1e-06 t281_leaf4 <= 1
 T281_L4_f3_le: x_Overall_Qual + 999991.000001 t281_leaf4 <= 1e+06
 T281_R5_f3_ge: x_Overall_Qual - 7 t281_leaf5 >= 0
 T281_L5_f96_le: x_Neighborhood_Edwards + 1e-06 t281_leaf5 <= 1
 T281_R5_f3_ge: x_Overall_Qual - 9 t281_leaf5 >= 0
 T281_R6_f3_ge: x_Overall_Qual - 7 t281_leaf6 >= 0
 T281_R6_f96_ge: x_Neighborhood_Edwards - t281_leaf6 >= 0
 TREE_282_ONEHOT: t282_leaf0 + t282_leaf1 + t282_leaf2 + t282_leaf3
   + t282_leaf4 = 1
 T282_L0_f44_le: x_Yr_Sold + 997993.000001 t282_leaf0 <= 1e+06
 T282_R1_f44_ge: x_Yr_Sold - 2007 t282_leaf1 >= 0
 T282_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t282_leaf1 <= 1
 T282_L1_f63_le: x_MS_Zoning_FV + 1e-06 t282_leaf1 <= 1
 T282_R2_f44_ge: x_Yr_Sold - 2007 t282_leaf2 >= 0
 T282_L2_f293_le: x_Sale_Condition_Abnorml + 1e-06 t282_leaf2 <= 1
 T282_R2_f63_ge: x_MS_Zoning_FV - t282_leaf2 >= 0
 T282_R3_f44_ge: x_Yr_Sold - 2007 t282_leaf3 >= 0
 T282_R3_f293_ge: x_Sale_Condition_Abnorml - t282_leaf3 >= 0
 T282_L3_f9_le: x_Exter_Cond + 2.000001 t282_leaf3 <= 4
 T282_R4_f44_ge: x_Yr_Sold - 2007 t282_leaf4 >= 0
 T282_R4_f293_ge: x_Sale_Condition_Abnorml - t282_leaf4 >= 0
 T282_R4_f9_ge: x_Exter_Cond - 3 t282_leaf4 >= -1
 TREE_283_ONEHOT: t283_leaf0 + t283_leaf1 + t283_leaf2 + t283_leaf3
   + t283_leaf4 = 1
 T283_L0_f3_le: x_Overall_Qual + 999997.000001 t283_leaf0 <= 1e+06
 T283_R1_f3_ge: x_Overall_Qual - 3 t283_leaf1 >= 0
 T283_L1_f3_le: x_Overall_Qual + 999993.000001 t283_leaf1 <= 1e+06
 T283_L1_f18_le: x_2nd_Flr_SF + 970.540001 t283_leaf1 <= 1641.54
 T283_R2_f3_ge: x_Overall_Qual - 3 t283_leaf2 >= 0
 T283_L2_f3_le: x_Overall_Qual + 999993.000001 t283_leaf2 <= 1e+06
 T283_R2_f18_ge: x_2nd_Flr_SF - 671 t283_leaf2 >= 0
 T283_R3_f3_ge: x_Overall_Qual - 3 t283_leaf3 >= 0
 T283_R3_f3_ge: x_Overall_Qual - 7 t283_leaf3 >= 0
 T283_L3_f77_le: x_Land_Contour_Bnk + 1e-06 t283_leaf3 <= 1
 T283_R4_f3_ge: x_Overall_Qual - 3 t283_leaf4 >= 0
 T283_R4_f3_ge: x_Overall_Qual - 7 t283_leaf4 >= 0
 T283_R4_f77_ge: x_Land_Contour_Bnk - t283_leaf4 >= 0
 TREE_284_ONEHOT: t284_leaf0 + t284_leaf1 + t284_leaf2 + t284_leaf3
   + t284_leaf4 = 1
 T284_L0_f38_le: x_Enclosed_Porch + 44.750001 t284_leaf0 <= 292.75
 T284_L0_f17_le: x_1st_Flr_SF + 19.90000099999997 t284_leaf0 <= 2534.9
 T284_L0_f20_le: x_Gr_Liv_Area + 335.7600010000001 t284_leaf0 <= 3421.76
 T284_L1_f38_le: x_Enclosed_Porch + 44.750001 t284_leaf1 <= 292.75
 T284_L1_f17_le: x_1st_Flr_SF + 19.90000099999997 t284_leaf1 <= 2534.9
 T284_R1_f20_ge: x_Gr_Liv_Area - 3086 t284_leaf1 >= 0
 T284_L2_f38_le: x_Enclosed_Porch + 44.750001 t284_leaf2 <= 292.75
 T284_R2_f17_ge: x_1st_Flr_SF - 2515 t284_leaf2 >= 0
 T284_L2_f37_le: x_Open_Porch_SF + 415.810001 t284_leaf2 <= 488.81
 T284_L3_f38_le: x_Enclosed_Porch + 44.750001 t284_leaf3 <= 292.75
 T284_R3_f17_ge: x_1st_Flr_SF - 2515 t284_leaf3 >= 0
 T284_R3_f37_ge: x_Open_Porch_SF - 73 t284_leaf3 >= 0
 T284_R4_f38_ge: x_Enclosed_Porch - 248 t284_leaf4 >= 0
 TREE_285_ONEHOT: t285_leaf0 + t285_leaf1 + t285_leaf2 + t285_leaf3
   + t285_leaf4 + t285_leaf5 + t285_leaf6 = 1
 T285_L0_f23_le: x_Full_Bath + 4.000001 t285_leaf0 <= 6
 T285_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t285_leaf0 <= 1
 T285_L0_f33_le: x_Garage_Area + 771.750001 t285_leaf0 <= 1163.75
 T285_L1_f23_le: x_Full_Bath + 4.000001 t285_leaf1 <= 6
 T285_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t285_leaf1 <= 1
 T285_R1_f33_ge: x_Garage_Area - 392 t285_leaf1 >= 0
 T285_L2_f23_le: x_Full_Bath + 4.000001 t285_leaf2 <= 6
 T285_R2_f297_ge: x_Sale_Condition_Normal - t285_leaf2 >= 0
 T285_L2_f4_le: x_Overall_Cond + 999996.000001 t285_leaf2 <= 1e+06
 T285_L3_f23_le: x_Full_Bath + 4.000001 t285_leaf3 <= 6
 T285_R3_f297_ge: x_Sale_Condition_Normal - t285_leaf3 >= 0
 T285_R3_f4_ge: x_Overall_Cond - 4 t285_leaf3 >= 0
 T285_R4_f23_ge: x_Full_Bath - 2 t285_leaf4 >= 0
 T285_L4_f33_le: x_Garage_Area + 9.750001000000111 t285_leaf4 <= 1163.75
 T285_L4_f296_le: x_Sale_Condition_Family + 1e-06 t285_leaf4 <= 1
 T285_R5_f23_ge: x_Full_Bath - 2 t285_leaf5 >= 0
 T285_L5_f33_le: x_Garage_Area + 9.750001000000111 t285_leaf5 <= 1163.75
 T285_R5_f296_ge: x_Sale_Condition_Family - t285_leaf5 >= 0
 T285_R6_f23_ge: x_Full_Bath - 2 t285_leaf6 >= 0
 T285_R6_f33_ge: x_Garage_Area - 1154 t285_leaf6 >= 0
 TREE_286_ONEHOT: t286_leaf0 + t286_leaf1 + t286_leaf2 + t286_leaf3
   + t286_leaf4 + t286_leaf5 = 1
 T286_L0_f3_le: x_Overall_Qual + 999997.000001 t286_leaf0 <= 1e+06
 T286_L0_f31_le: x_Garage_Yr_Blt + 998077.000001 t286_leaf0 <= 1e+06
 T286_L1_f3_le: x_Overall_Qual + 999997.000001 t286_leaf1 <= 1e+06
 T286_R1_f31_ge: x_Garage_Yr_Blt - 1923 t286_leaf1 >= 0
 T286_R2_f3_ge: x_Overall_Qual - 3 t286_leaf2 >= 0
 T286_L2_f3_le: x_Overall_Qual + 999993.000001 t286_leaf2 <= 1e+06
 T286_L2_f144_le: x_House_Style_2Story + 1e-06 t286_leaf2 <= 1
 T286_R3_f3_ge: x_Overall_Qual - 3 t286_leaf3 >= 0
 T286_L3_f3_le: x_Overall_Qual + 999993.000001 t286_leaf3 <= 1e+06
 T286_R3_f144_ge: x_House_Style_2Story - t286_leaf3 >= 0
 T286_R4_f3_ge: x_Overall_Qual - 3 t286_leaf4 >= 0
 T286_R4_f3_ge: x_Overall_Qual - 7 t286_leaf4 >= 0
 T286_L4_f31_le: x_Garage_Yr_Blt + 998014.000001 t286_leaf4 <= 1e+06
 T286_R5_f3_ge: x_Overall_Qual - 3 t286_leaf5 >= 0
 T286_R5_f3_ge: x_Overall_Qual - 7 t286_leaf5 >= 0
 T286_R5_f31_ge: x_Garage_Yr_Blt - 1986 t286_leaf5 >= 0
 TREE_287_ONEHOT: t287_leaf0 + t287_leaf1 + t287_leaf2 + t287_leaf3
   + t287_leaf4 + t287_leaf5 = 1
 T287_L0_f3_le: x_Overall_Qual + 999997.000001 t287_leaf0 <= 1e+06
 T287_L0_f6_le: x_Year_Remod/Add + 998049.000001 t287_leaf0 <= 1e+06
 T287_L1_f3_le: x_Overall_Qual + 999997.000001 t287_leaf1 <= 1e+06
 T287_R1_f6_ge: x_Year_Remod/Add - 1951 t287_leaf1 >= 0
 T287_R2_f3_ge: x_Overall_Qual - 3 t287_leaf2 >= 0
 T287_L2_f3_le: x_Overall_Qual + 999993.000001 t287_leaf2 <= 1e+06
 T287_L2_f23_le: x_Full_Bath + 4.000001 t287_leaf2 <= 6
 T287_R3_f3_ge: x_Overall_Qual - 3 t287_leaf3 >= 0
 T287_L3_f3_le: x_Overall_Qual + 999993.000001 t287_leaf3 <= 1e+06
 T287_R3_f23_ge: x_Full_Bath - 2 t287_leaf3 >= 0
 T287_R4_f3_ge: x_Overall_Qual - 3 t287_leaf4 >= 0
 T287_R4_f3_ge: x_Overall_Qual - 7 t287_leaf4 >= 0
 T287_L4_f6_le: x_Year_Remod/Add + 998020.000001 t287_leaf4 <= 1e+06
 T287_R5_f3_ge: x_Overall_Qual - 3 t287_leaf5 >= 0
 T287_R5_f3_ge: x_Overall_Qual - 7 t287_leaf5 >= 0
 T287_R5_f6_ge: x_Year_Remod/Add - 1980 t287_leaf5 >= 0
 TREE_288_ONEHOT: t288_leaf0 + t288_leaf1 + t288_leaf2 + t288_leaf3
   + t288_leaf4 = 1
 T288_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t288_leaf0 <= 1
 T288_L0_f20_le: x_Gr_Liv_Area + 1184.760001 t288_leaf0 <= 3421.76
 T288_L0_f3_le: x_Overall_Qual + 999993.000001 t288_leaf0 <= 1e+06
 T288_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t288_leaf1 <= 1
 T288_L1_f20_le: x_Gr_Liv_Area + 1184.760001 t288_leaf1 <= 3421.76
 T288_R1_f3_ge: x_Overall_Qual - 7 t288_leaf1 >= 0
 T288_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t288_leaf2 <= 1
 T288_R2_f20_ge: x_Gr_Liv_Area - 2237 t288_leaf2 >= 0
 T288_L2_f77_le: x_Land_Contour_Bnk + 1e-06 t288_leaf2 <= 1
 T288_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t288_leaf3 <= 1
 T288_R3_f20_ge: x_Gr_Liv_Area - 2237 t288_leaf3 >= 0
 T288_R3_f77_ge: x_Land_Contour_Bnk - t288_leaf3 >= 0
 T288_R4_f95_ge: x_Neighborhood_Crawfor - t288_leaf4 >= 0
 TREE_289_ONEHOT: t289_leaf0 + t289_leaf1 + t289_leaf2 + t289_leaf3
   + t289_leaf4 = 1
 T289_L0_f33_le: x_Garage_Area + 292.750001 t289_leaf0 <= 1163.75
 T289_L0_f26_le: x_Kitchen_AbvGr + 1.000001 t289_leaf0 <= 3
 T289_L0_f1_le: x_Lot_Area <= 17226.999999
 T289_L1_f33_le: x_Garage_Area + 292.750001 t289_leaf1 <= 1163.75
 T289_L1_f26_le: x_Kitchen_AbvGr + 1.000001 t289_leaf1 <= 3
 T289_R1_f1_ge: x_Lot_Area - 10227 t289_leaf1 >= 7000
 T289_L2_f33_le: x_Garage_Area + 292.750001 t289_leaf2 <= 1163.75
 T289_R2_f26_ge: x_Kitchen_AbvGr - 2 t289_leaf2 >= 0
 T289_L2_f34_le: x_Garage_Qual + 3.000001 t289_leaf2 <= 4
 T289_L3_f33_le: x_Garage_Area + 292.750001 t289_leaf3 <= 1163.75
 T289_R3_f26_ge: x_Kitchen_AbvGr - 2 t289_leaf3 >= 0
 T289_R3_f34_ge: x_Garage_Qual >= 1
 T289_R4_f33_ge: x_Garage_Area - 871 t289_leaf4 >= 0
 TREE_290_ONEHOT: t290_leaf0 + t290_leaf1 + t290_leaf2 + t290_leaf3
   + t290_leaf4 + t290_leaf5 = 1
 T290_L0_f37_le: x_Open_Porch_SF + 413.810001 t290_leaf0 <= 488.81
 T290_L0_f295_le: x_Sale_Condition_Alloca + 1e-06 t290_leaf0 <= 1
 T290_L1_f37_le: x_Open_Porch_SF + 413.810001 t290_leaf1 <= 488.81
 T290_R1_f295_ge: x_Sale_Condition_Alloca - t290_leaf1 >= 0
 T290_L1_f23_le: x_Full_Bath + 4.000001 t290_leaf1 <= 6
 T290_L2_f37_le: x_Open_Porch_SF + 413.810001 t290_leaf2 <= 488.81
 T290_R2_f295_ge: x_Sale_Condition_Alloca - t290_leaf2 >= 0
 T290_R2_f23_ge: x_Full_Bath - 2 t290_leaf2 >= 0
 T290_R3_f37_ge: x_Open_Porch_SF - 75 t290_leaf3 >= 0
 T290_L3_f18_le: x_2nd_Flr_SF + 273.5400010000001 t290_leaf3 <= 1641.54
 T290_L3_f200_le: x_Foundation_BrkTil + 1e-06 t290_leaf3 <= 1
 T290_R4_f37_ge: x_Open_Porch_SF - 75 t290_leaf4 >= 0
 T290_L4_f18_le: x_2nd_Flr_SF + 273.5400010000001 t290_leaf4 <= 1641.54
 T290_R4_f200_ge: x_Foundation_BrkTil - t290_leaf4 >= 0
 T290_R5_f37_ge: x_Open_Porch_SF - 75 t290_leaf5 >= 0
 T290_R5_f18_ge: x_2nd_Flr_SF - 1368 t290_leaf5 >= 0
 TREE_291_ONEHOT: t291_leaf0 + t291_leaf1 + t291_leaf2 + t291_leaf3 = 1
 T291_L0_f17_le: x_1st_Flr_SF + 19.90000099999997 t291_leaf0 <= 2534.9
 T291_L0_f0_le: x_Lot_Frontage + 40.530001 t291_leaf0 <= 175.53
 T291_L0_f6_le: x_Year_Remod/Add + 997993.000001 t291_leaf0 <= 1e+06
 T291_L1_f17_le: x_1st_Flr_SF + 19.90000099999997 t291_leaf1 <= 2534.9
 T291_L1_f0_le: x_Lot_Frontage + 40.530001 t291_leaf1 <= 175.53
 T291_R1_f6_ge: x_Year_Remod/Add - 2007 t291_leaf1 >= 0
 T291_L2_f17_le: x_1st_Flr_SF + 19.90000099999997 t291_leaf2 <= 2534.9
 T291_R2_f0_ge: x_Lot_Frontage - 135 t291_leaf2 >= 0
 T291_R3_f17_ge: x_1st_Flr_SF - 2515 t291_leaf3 >= 0
 TREE_292_ONEHOT: t292_leaf0 + t292_leaf1 + t292_leaf2 + t292_leaf3
   + t292_leaf4 + t292_leaf5 + t292_leaf6 = 1
 T292_L0_f31_le: x_Garage_Yr_Blt + 998076.000001 t292_leaf0 <= 1e+06
 T292_L0_f20_le: x_Gr_Liv_Area + 2266.760001000001 t292_leaf0 <= 3421.76
 T292_L0_f8_le: x_Exter_Qual + 2.000001 t292_leaf0 <= 4
 T292_L1_f31_le: x_Garage_Yr_Blt + 998076.000001 t292_leaf1 <= 1e+06
 T292_L1_f20_le: x_Gr_Liv_Area + 2266.760001000001 t292_leaf1 <= 3421.76
 T292_R1_f8_ge: x_Exter_Qual >= 2
 T292_L2_f31_le: x_Garage_Yr_Blt + 998076.000001 t292_leaf2 <= 1e+06
 T292_R2_f20_ge: x_Gr_Liv_Area - 1155 t292_leaf2 >= 0
 T292_L2_f85_le: x_Lot_Config_Inside + 1e-06 t292_leaf2 <= 1
 T292_L3_f31_le: x_Garage_Yr_Blt + 998076.000001 t292_leaf3 <= 1e+06
 T292_R3_f20_ge: x_Gr_Liv_Area - 1155 t292_leaf3 >= 0
 T292_R3_f85_ge: x_Lot_Config_Inside - t292_leaf3 >= 0
 T292_R4_f31_ge: x_Garage_Yr_Blt - 1924 t292_leaf4 >= 0
 T292_L4_f277_le: x_Mo_Sold_7 + 1e-06 t292_leaf4 <= 1
 T292_L4_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t292_leaf4 <= 1984.5
 T292_R5_f31_ge: x_Garage_Yr_Blt - 1924 t292_leaf5 >= 0
 T292_L5_f277_le: x_Mo_Sold_7 + 1e-06 t292_leaf5 <= 1
 T292_R5_f12_ge: x_BsmtFin_SF_1 - 1732 t292_leaf5 >= 0
 T292_R6_f31_ge: x_Garage_Yr_Blt - 1924 t292_leaf6 >= 0
 T292_R6_f277_ge: x_Mo_Sold_7 - t292_leaf6 >= 0
 TREE_293_ONEHOT: t293_leaf0 + t293_leaf1 + t293_leaf2 + t293_leaf3 = 1
 T293_L0_f7_le: x_Mas_Vnr_Area + 458.070001 t293_leaf0 <= 931.07
 T293_L0_f37_le: x_Open_Porch_SF + 289.810001 t293_leaf0 <= 488.81
 T293_L0_f14_le: x_Bsmt_Unf_SF + 1392.680001 t293_leaf0 <= 1812.68
 T293_L1_f7_le: x_Mas_Vnr_Area + 458.070001 t293_leaf1 <= 931.07
 T293_L1_f37_le: x_Open_Porch_SF + 289.810001 t293_leaf1 <= 488.81
 T293_R1_f14_ge: x_Bsmt_Unf_SF - 420 t293_leaf1 >= 0
 T293_L2_f7_le: x_Mas_Vnr_Area + 458.070001 t293_leaf2 <= 931.07
 T293_R2_f37_ge: x_Open_Porch_SF - 199 t293_leaf2 >= 0
 T293_R3_f7_ge: x_Mas_Vnr_Area - 473 t293_leaf3 >= 0
 TREE_294_ONEHOT: t294_leaf0 + t294_leaf1 + t294_leaf2 + t294_leaf3
   + t294_leaf4 + t294_leaf5 = 1
 T294_L0_f3_le: x_Overall_Qual + 999993.000001 t294_leaf0 <= 1e+06
 T294_L0_f16_le: x_Heating_QC + 1.000001 t294_leaf0 <= 4
 T294_L0_f36_le: x_Wood_Deck_SF + 245.900001 t294_leaf0 <= 525.9
 T294_L1_f3_le: x_Overall_Qual + 999993.000001 t294_leaf1 <= 1e+06
 T294_L1_f16_le: x_Heating_QC + 1.000001 t294_leaf1 <= 4
 T294_R1_f36_ge: x_Wood_Deck_SF - 280 t294_leaf1 >= 0
 T294_L2_f3_le: x_Overall_Qual + 999993.000001 t294_leaf2 <= 1e+06
 T294_R2_f16_ge: x_Heating_QC - t294_leaf2 >= 2
 T294_R3_f3_ge: x_Overall_Qual - 7 t294_leaf3 >= 0
 T294_L3_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t294_leaf3 <= 2514.48
 T294_L3_f105_le: x_Neighborhood_NWAmes + 1e-06 t294_leaf3 <= 1
 T294_R4_f3_ge: x_Overall_Qual - 7 t294_leaf4 >= 0
 T294_L4_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t294_leaf4 <= 2514.48
 T294_R4_f105_ge: x_Neighborhood_NWAmes - t294_leaf4 >= 0
 T294_R5_f3_ge: x_Overall_Qual - 7 t294_leaf5 >= 0
 T294_R5_f15_ge: x_Total_Bsmt_SF - 2136 t294_leaf5 >= 0
 TREE_295_ONEHOT: t295_leaf0 + t295_leaf1 + t295_leaf2 + t295_leaf3
   + t295_leaf4 = 1
 T295_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t295_leaf0 <= 1
 T295_L0_f119_le: x_Condition_1_Norm + 1e-06 t295_leaf0 <= 1
 T295_L0_f30_le: x_Fireplace_Qu + 1.000001 t295_leaf0 <= 4
 T295_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t295_leaf1 <= 1
 T295_L1_f119_le: x_Condition_1_Norm + 1e-06 t295_leaf1 <= 1
 T295_R1_f30_ge: x_Fireplace_Qu - t295_leaf1 >= 2
 T295_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t295_leaf2 <= 1
 T295_R2_f119_ge: x_Condition_1_Norm - t295_leaf2 >= 0
 T295_L2_f31_le: x_Garage_Yr_Blt + 998075.000001 t295_leaf2 <= 1e+06
 T295_L3_f164_le: x_Exterior_1st_BrkFace + 1e-06 t295_leaf3 <= 1
 T295_R3_f119_ge: x_Condition_1_Norm - t295_leaf3 >= 0
 T295_R3_f31_ge: x_Garage_Yr_Blt - 1925 t295_leaf3 >= 0
 T295_R4_f164_ge: x_Exterior_1st_BrkFace - t295_leaf4 >= 0
 TREE_296_ONEHOT: t296_leaf0 + t296_leaf1 + t296_leaf2 + t296_leaf3
   + t296_leaf4 + t296_leaf5 = 1
 T296_L0_f119_le: x_Condition_1_Norm + 1e-06 t296_leaf0 <= 1
 T296_L0_f45_le: x_MS_SubClass_20 + 1e-06 t296_leaf0 <= 1
 T296_L0_f30_le: x_Fireplace_Qu + 1.000001 t296_leaf0 <= 4
 T296_L1_f119_le: x_Condition_1_Norm + 1e-06 t296_leaf1 <= 1
 T296_L1_f45_le: x_MS_SubClass_20 + 1e-06 t296_leaf1 <= 1
 T296_R1_f30_ge: x_Fireplace_Qu - t296_leaf1 >= 2
 T296_L2_f119_le: x_Condition_1_Norm + 1e-06 t296_leaf2 <= 1
 T296_R2_f45_ge: x_MS_SubClass_20 - t296_leaf2 >= 0
 T296_R3_f119_ge: x_Condition_1_Norm - t296_leaf3 >= 0
 T296_L3_f92_le: x_Neighborhood_BrkSide + 1e-06 t296_leaf3 <= 1
 T296_L3_f0_le: x_Lot_Frontage + 40.530001 t296_leaf3 <= 175.53
 T296_R4_f119_ge: x_Condition_1_Norm - t296_leaf4 >= 0
 T296_L4_f92_le: x_Neighborhood_BrkSide + 1e-06 t296_leaf4 <= 1
 T296_R4_f0_ge: x_Lot_Frontage - 135 t296_leaf4 >= 0
 T296_R5_f119_ge: x_Condition_1_Norm - t296_leaf5 >= 0
 T296_R5_f92_ge: x_Neighborhood_BrkSide - t296_leaf5 >= 0
 TREE_297_ONEHOT: t297_leaf0 + t297_leaf1 + t297_leaf2 + t297_leaf3
   + t297_leaf4 + t297_leaf5 = 1
 T297_L0_f82_le: x_Lot_Config_CulDSac + 1e-06 t297_leaf0 <= 1
 T297_L0_f3_le: x_Overall_Qual + 999993.000001 t297_leaf0 <= 1e+06
 T297_L0_f18_le: x_2nd_Flr_SF + 1057.540001 t297_leaf0 <= 1641.54
 T297_L1_f82_le: x_Lot_Config_CulDSac + 1e-06 t297_leaf1 <= 1
 T297_L1_f3_le: x_Overall_Qual + 999993.000001 t297_leaf1 <= 1e+06
 T297_R1_f18_ge: x_2nd_Flr_SF - 584 t297_leaf1 >= 0
 T297_L2_f82_le: x_Lot_Config_CulDSac + 1e-06 t297_leaf2 <= 1
 T297_R2_f3_ge: x_Overall_Qual - 7 t297_leaf2 >= 0
 T297_L2_f31_le: x_Garage_Yr_Blt + 997995.000001 t297_leaf2 <= 1e+06
 T297_L3_f82_le: x_Lot_Config_CulDSac + 1e-06 t297_leaf3 <= 1
 T297_R3_f3_ge: x_Overall_Qual - 7 t297_leaf3 >= 0
 T297_R3_f31_ge: x_Garage_Yr_Blt - 2005 t297_leaf3 >= 0
 T297_R4_f82_ge: x_Lot_Config_CulDSac - t297_leaf4 >= 0
 T297_L4_f1_le: x_Lot_Area <= 16132.999999
 T297_R5_f82_ge: x_Lot_Config_CulDSac - t297_leaf5 >= 0
 T297_R5_f1_ge: x_Lot_Area - 9133 t297_leaf5 >= 7000
 TREE_298_ONEHOT: t298_leaf0 + t298_leaf1 + t298_leaf2 + t298_leaf3 = 1
 T298_L0_f7_le: x_Mas_Vnr_Area + 9.070001000000047 t298_leaf0 <= 931.07
 T298_L0_f63_le: x_MS_Zoning_FV + 1e-06 t298_leaf0 <= 1
 T298_L0_f20_le: x_Gr_Liv_Area + 788.7600010000001 t298_leaf0 <= 3421.76
 T298_L1_f7_le: x_Mas_Vnr_Area + 9.070001000000047 t298_leaf1 <= 931.07
 T298_L1_f63_le: x_MS_Zoning_FV + 1e-06 t298_leaf1 <= 1
 T298_R1_f20_ge: x_Gr_Liv_Area - 2633 t298_leaf1 >= 0
 T298_L2_f7_le: x_Mas_Vnr_Area + 9.070001000000047 t298_leaf2 <= 931.07
 T298_R2_f63_ge: x_MS_Zoning_FV - t298_leaf2 >= 0
 T298_R3_f7_ge: x_Mas_Vnr_Area - 922 t298_leaf3 >= 0
 TREE_299_ONEHOT: t299_leaf0 + t299_leaf1 + t299_leaf2 + t299_leaf3 = 1
 T299_L0_f118_le: x_Condition_1_Feedr + 1e-06 t299_leaf0 <= 1
 T299_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t299_leaf0 <= 1
 T299_L0_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t299_leaf0 <= 1
 T299_L1_f118_le: x_Condition_1_Feedr + 1e-06 t299_leaf1 <= 1
 T299_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t299_leaf1 <= 1
 T299_R1_f218_ge: x_BsmtFin_Type_2_ALQ - t299_leaf1 >= 0
 T299_L2_f118_le: x_Condition_1_Feedr + 1e-06 t299_leaf2 <= 1
 T299_R2_f95_ge: x_Neighborhood_Crawfor - t299_leaf2 >= 0
 T299_R3_f118_ge: x_Condition_1_Feedr - t299_leaf3 >= 0
 TREE_300_ONEHOT: t300_leaf0 + t300_leaf1 + t300_leaf2 + t300_leaf3
   + t300_leaf4 + t300_leaf5 = 1
 T300_L0_f20_le: x_Gr_Liv_Area + 2433.760001 t300_leaf0 <= 3421.76
 T300_L0_f20_le: x_Gr_Liv_Area + 2456.760001 t300_leaf0 <= 3421.76
 T300_L1_f20_le: x_Gr_Liv_Area + 2433.760001 t300_leaf1 <= 3421.76
 T300_R1_f20_ge: x_Gr_Liv_Area - 965 t300_leaf1 >= 0
 T300_R2_f20_ge: x_Gr_Liv_Area - 988 t300_leaf2 >= 0
 T300_L2_f4_le: x_Overall_Cond + 999994.000001 t300_leaf2 <= 1e+06
 T300_L2_f5_le: x_Year_Built + 998051.000001 t300_leaf2 <= 1e+06
 T300_R3_f20_ge: x_Gr_Liv_Area - 988 t300_leaf3 >= 0
 T300_L3_f4_le: x_Overall_Cond + 999994.000001 t300_leaf3 <= 1e+06
 T300_R3_f5_ge: x_Year_Built - 1949 t300_leaf3 >= 0
 T300_R4_f20_ge: x_Gr_Liv_Area - 988 t300_leaf4 >= 0
 T300_R4_f4_ge: x_Overall_Cond - 6 t300_leaf4 >= 0
 T300_L4_f251_le: x_Garage_Type_Detchd + 1e-06 t300_leaf4 <= 1
 T300_R5_f20_ge: x_Gr_Liv_Area - 988 t300_leaf5 >= 0
 T300_R5_f4_ge: x_Overall_Cond - 6 t300_leaf5 >= 0
 T300_R5_f251_ge: x_Garage_Type_Detchd - t300_leaf5 >= 0
 TREE_301_ONEHOT: t301_leaf0 + t301_leaf1 + t301_leaf2 + t301_leaf3 = 1
 T301_L0_f44_le: x_Yr_Sold + 997993.000001 t301_leaf0 <= 1e+06
 T301_R1_f44_ge: x_Yr_Sold - 2007 t301_leaf1 >= 0
 T301_L1_f20_le: x_Gr_Liv_Area + 26.7600010000001 t301_leaf1 <= 3421.76
 T301_L1_f11_le: x_Bsmt_Cond + 2.000001 t301_leaf1 <= 4
 T301_R2_f44_ge: x_Yr_Sold - 2007 t301_leaf2 >= 0
 T301_L2_f20_le: x_Gr_Liv_Area + 26.7600010000001 t301_leaf2 <= 3421.76
 T301_R2_f11_ge: x_Bsmt_Cond - 3 t301_leaf2 >= -1
 T301_R3_f44_ge: x_Yr_Sold - 2007 t301_leaf3 >= 0
 T301_R3_f20_ge: x_Gr_Liv_Area - 3395 t301_leaf3 >= 0
 TREE_302_ONEHOT: t302_leaf0 + t302_leaf1 + t302_leaf2 + t302_leaf3 = 1
 T302_L0_f113_le: x_Neighborhood_Somerst + 1e-06 t302_leaf0 <= 1
 T302_L0_f17_le: x_1st_Flr_SF + 334.900001 t302_leaf0 <= 2534.9
 T302_L0_f51_le: x_MS_SubClass_70 + 1e-06 t302_leaf0 <= 1
 T302_L1_f113_le: x_Neighborhood_Somerst + 1e-06 t302_leaf1 <= 1
 T302_L1_f17_le: x_1st_Flr_SF + 334.900001 t302_leaf1 <= 2534.9
 T302_R1_f51_ge: x_MS_SubClass_70 - t302_leaf1 >= 0
 T302_L2_f113_le: x_Neighborhood_Somerst + 1e-06 t302_leaf2 <= 1
 T302_R2_f17_ge: x_1st_Flr_SF - 2200 t302_leaf2 >= 0
 T302_R3_f113_ge: x_Neighborhood_Somerst - t302_leaf3 >= 0
 TREE_303_ONEHOT: t303_leaf0 + t303_leaf1 + t303_leaf2 + t303_leaf3
   + t303_leaf4 = 1
 T303_L0_f7_le: x_Mas_Vnr_Area + 143.070001 t303_leaf0 <= 931.07
 T303_L0_f3_le: x_Overall_Qual + 999990.000001 t303_leaf0 <= 1e+06
 T303_L0_f3_le: x_Overall_Qual + 999993.000001 t303_leaf0 <= 1e+06
 T303_L1_f7_le: x_Mas_Vnr_Area + 143.070001 t303_leaf1 <= 931.07
 T303_L1_f3_le: x_Overall_Qual + 999990.000001 t303_leaf1 <= 1e+06
 T303_R1_f3_ge: x_Overall_Qual - 7 t303_leaf1 >= 0
 T303_L2_f7_le: x_Mas_Vnr_Area + 143.070001 t303_leaf2 <= 931.07
 T303_R2_f3_ge: x_Overall_Qual - 10 t303_leaf2 >= 0
 T303_R3_f7_ge: x_Mas_Vnr_Area - 788 t303_leaf3 >= 0
 T303_L3_f76_le: x_Lot_Shape_Reg + 1e-06 t303_leaf3 <= 1
 T303_R4_f7_ge: x_Mas_Vnr_Area - 788 t303_leaf4 >= 0
 T303_R4_f76_ge: x_Lot_Shape_Reg - t303_leaf4 >= 0
 TREE_304_ONEHOT: t304_leaf0 + t304_leaf1 + t304_leaf2 + t304_leaf3
   + t304_leaf4 + t304_leaf5 + t304_leaf6 + t304_leaf7 = 1
 T304_L0_f16_le: x_Heating_QC + 1.000001 t304_leaf0 <= 4
 T304_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t304_leaf0 <= 1
 T304_L0_f294_le: x_Sale_Condition_AdjLand + 1e-06 t304_leaf0 <= 1
 T304_L1_f16_le: x_Heating_QC + 1.000001 t304_leaf1 <= 4
 T304_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t304_leaf1 <= 1
 T304_R1_f294_ge: x_Sale_Condition_AdjLand - t304_leaf1 >= 0
 T304_L2_f16_le: x_Heating_QC + 1.000001 t304_leaf2 <= 4
 T304_R2_f297_ge: x_Sale_Condition_Normal - t304_leaf2 >= 0
 T304_L2_f1_le: x_Lot_Area <= 9919.999999
 T304_L3_f16_le: x_Heating_QC + 1.000001 t304_leaf3 <= 4
 T304_R3_f297_ge: x_Sale_Condition_Normal - t304_leaf3 >= 0
 T304_R3_f1_ge: x_Lot_Area - 2920 t304_leaf3 >= 7000
 T304_R4_f16_ge: x_Heating_QC - t304_leaf4 >= 2
 T304_L4_f44_le: x_Yr_Sold + 997992.000001 t304_leaf4 <= 1e+06
 T304_L4_f1_le: x_Lot_Area <= 25285.999999
 T304_R5_f16_ge: x_Heating_QC - t304_leaf5 >= 2
 T304_L5_f44_le: x_Yr_Sold + 997992.000001 t304_leaf5 <= 1e+06
 T304_R5_f1_ge: x_Lot_Area - 18286 t304_leaf5 >= 7000
 T304_R6_f16_ge: x_Heating_QC - t304_leaf6 >= 2
 T304_R6_f44_ge: x_Yr_Sold - 2008 t304_leaf6 >= 0
 T304_L6_f5_le: x_Year_Built + 997992.000001 t304_leaf6 <= 1e+06
 T304_R7_f16_ge: x_Heating_QC - t304_leaf7 >= 2
 T304_R7_f44_ge: x_Yr_Sold - 2008 t304_leaf7 >= 0
 T304_R7_f5_ge: x_Year_Built - 2008 t304_leaf7 >= 0
 TREE_305_ONEHOT: t305_leaf0 + t305_leaf1 + t305_leaf2 + t305_leaf3
   + t305_leaf4 + t305_leaf5 + t305_leaf6 = 1
 T305_L0_f31_le: x_Garage_Yr_Blt + 998071.000001 t305_leaf0 <= 1e+06
 T305_L0_f20_le: x_Gr_Liv_Area + 2277.760001000001 t305_leaf0 <= 3421.76
 T305_L0_f92_le: x_Neighborhood_BrkSide + 1e-06 t305_leaf0 <= 1
 T305_L1_f31_le: x_Garage_Yr_Blt + 998071.000001 t305_leaf1 <= 1e+06
 T305_L1_f20_le: x_Gr_Liv_Area + 2277.760001000001 t305_leaf1 <= 3421.76
 T305_R1_f92_ge: x_Neighborhood_BrkSide - t305_leaf1 >= 0
 T305_L2_f31_le: x_Garage_Yr_Blt + 998071.000001 t305_leaf2 <= 1e+06
 T305_R2_f20_ge: x_Gr_Liv_Area - 1144 t305_leaf2 >= 0
 T305_L2_f3_le: x_Overall_Qual + 999995.000001 t305_leaf2 <= 1e+06
 T305_L3_f31_le: x_Garage_Yr_Blt + 998071.000001 t305_leaf3 <= 1e+06
 T305_R3_f20_ge: x_Gr_Liv_Area - 1144 t305_leaf3 >= 0
 T305_R3_f3_ge: x_Overall_Qual - 5 t305_leaf3 >= 0
 T305_R4_f31_ge: x_Garage_Yr_Blt - 1929 t305_leaf4 >= 0
 T305_L4_f37_le: x_Open_Porch_SF + 197.810001 t305_leaf4 <= 488.81
 T305_L4_f20_le: x_Gr_Liv_Area + 335.7600010000001 t305_leaf4 <= 3421.76
 T305_R5_f31_ge: x_Garage_Yr_Blt - 1929 t305_leaf5 >= 0
 T305_L5_f37_le: x_Open_Porch_SF + 197.810001 t305_leaf5 <= 488.81
 T305_R5_f20_ge: x_Gr_Liv_Area - 3086 t305_leaf5 >= 0
 T305_R6_f31_ge: x_Garage_Yr_Blt - 1929 t305_leaf6 >= 0
 T305_R6_f37_ge: x_Open_Porch_SF - 291 t305_leaf6 >= 0
 TREE_306_ONEHOT: t306_leaf0 + t306_leaf1 + t306_leaf2 + t306_leaf3
   + t306_leaf4 + t306_leaf5 = 1
 T306_L0_f12_le: x_BsmtFin_SF_1 + 1802.500001 t306_leaf0 <= 1984.5
 T306_L0_f40_le: x_Screen_Porch + 228.270001 t306_leaf0 <= 268.27
 T306_L0_f20_le: x_Gr_Liv_Area + 1665.760001 t306_leaf0 <= 3421.76
 T306_L1_f12_le: x_BsmtFin_SF_1 + 1802.500001 t306_leaf1 <= 1984.5
 T306_L1_f40_le: x_Screen_Porch + 228.270001 t306_leaf1 <= 268.27
 T306_R1_f20_ge: x_Gr_Liv_Area - 1756 t306_leaf1 >= 0
 T306_L2_f12_le: x_BsmtFin_SF_1 + 1802.500001 t306_leaf2 <= 1984.5
 T306_R2_f40_ge: x_Screen_Porch - 40 t306_leaf2 >= 0
 T306_R3_f12_ge: x_BsmtFin_SF_1 - 182 t306_leaf3 >= 0
 T306_L3_f14_le: x_Bsmt_Unf_SF + 1784.680001 t306_leaf3 <= 1812.68
 T306_R4_f12_ge: x_BsmtFin_SF_1 - 182 t306_leaf4 >= 0
 T306_R4_f14_ge: x_Bsmt_Unf_SF - 28 t306_leaf4 >= 0
 T306_L4_f44_le: x_Yr_Sold + 997991.000001 t306_leaf4 <= 1e+06
 T306_R5_f12_ge: x_BsmtFin_SF_1 - 182 t306_leaf5 >= 0
 T306_R5_f14_ge: x_Bsmt_Unf_SF - 28 t306_leaf5 >= 0
 T306_R5_f44_ge: x_Yr_Sold - 2009 t306_leaf5 >= 0
 TREE_307_ONEHOT: t307_leaf0 + t307_leaf1 + t307_leaf2 + t307_leaf3
   + t307_leaf4 + t307_leaf5 = 1
 T307_L0_f5_le: x_Year_Built + 998085.000001 t307_leaf0 <= 1e+06
 T307_L0_f20_le: x_Gr_Liv_Area + 2051.760001000001 t307_leaf0 <= 3421.76
 T307_L1_f5_le: x_Year_Built + 998085.000001 t307_leaf1 <= 1e+06
 T307_R1_f20_ge: x_Gr_Liv_Area - 1370 t307_leaf1 >= 0
 T307_R2_f5_ge: x_Year_Built - 1915 t307_leaf2 >= 0
 T307_L2_f3_le: x_Overall_Qual + 999996.000001 t307_leaf2 <= 1e+06
 T307_L2_f293_le: x_Sale_Condition_Abnorml + 1e-06 t307_leaf2 <= 1
 T307_R3_f5_ge: x_Year_Built - 1915 t307_leaf3 >= 0
 T307_L3_f3_le: x_Overall_Qual + 999996.000001 t307_leaf3 <= 1e+06
 T307_R3_f293_ge: x_Sale_Condition_Abnorml - t307_leaf3 >= 0
 T307_R4_f5_ge: x_Year_Built - 1915 t307_leaf4 >= 0
 T307_R4_f3_ge: x_Overall_Qual - 4 t307_leaf4 >= 0
 T307_L4_f206_le: x_Bsmt_Exposure_Av + 1e-06 t307_leaf4 <= 1
 T307_R5_f5_ge: x_Year_Built - 1915 t307_leaf5 >= 0
 T307_R5_f3_ge: x_Overall_Qual - 4 t307_leaf5 >= 0
 T307_R5_f206_ge: x_Bsmt_Exposure_Av - t307_leaf5 >= 0
 TREE_308_ONEHOT: t308_leaf0 + t308_leaf1 + t308_leaf2 + t308_leaf3 = 1
 T308_L0_f40_le: x_Screen_Porch + 120.270001 t308_leaf0 <= 268.27
 T308_L0_f63_le: x_MS_Zoning_FV + 1e-06 t308_leaf0 <= 1
 T308_L0_f4_le: x_Overall_Cond + 999996.000001 t308_leaf0 <= 1e+06
 T308_L1_f40_le: x_Screen_Porch + 120.270001 t308_leaf1 <= 268.27
 T308_L1_f63_le: x_MS_Zoning_FV + 1e-06 t308_leaf1 <= 1
 T308_R1_f4_ge: x_Overall_Cond - 4 t308_leaf1 >= 0
 T308_L2_f40_le: x_Screen_Porch + 120.270001 t308_leaf2 <= 268.27
 T308_R2_f63_ge: x_MS_Zoning_FV - t308_leaf2 >= 0
 T308_R3_f40_ge: x_Screen_Porch - 148 t308_leaf3 >= 0
 TREE_309_ONEHOT: t309_leaf0 + t309_leaf1 + t309_leaf2 + t309_leaf3
   + t309_leaf4 + t309_leaf5 + t309_leaf6 + t309_leaf7 = 1
 T309_L0_f3_le: x_Overall_Qual + 999993.000001 t309_leaf0 <= 1e+06
 T309_L0_f23_le: x_Full_Bath + 4.000001 t309_leaf0 <= 6
 T309_L0_f62_le: x_MS_Zoning_C_(all) + 1e-06 t309_leaf0 <= 1
 T309_L1_f3_le: x_Overall_Qual + 999993.000001 t309_leaf1 <= 1e+06
 T309_L1_f23_le: x_Full_Bath + 4.000001 t309_leaf1 <= 6
 T309_R1_f62_ge: x_MS_Zoning_C_(all) - t309_leaf1 >= 0
 T309_L2_f3_le: x_Overall_Qual + 999993.000001 t309_leaf2 <= 1e+06
 T309_R2_f23_ge: x_Full_Bath - 2 t309_leaf2 >= 0
 T309_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t309_leaf2 <= 1
 T309_L3_f3_le: x_Overall_Qual + 999993.000001 t309_leaf3 <= 1e+06
 T309_R3_f23_ge: x_Full_Bath - 2 t309_leaf3 >= 0
 T309_R3_f245_ge: x_FunctioNo_aplical_Typ - t309_leaf3 >= 0
 T309_R4_f3_ge: x_Overall_Qual - 7 t309_leaf4 >= 0
 T309_L4_f6_le: x_Year_Remod/Add + 998006.000001 t309_leaf4 <= 1e+06
 T309_L4_f297_le: x_Sale_Condition_Normal + 1e-06 t309_leaf4 <= 1
 T309_R5_f3_ge: x_Overall_Qual - 7 t309_leaf5 >= 0
 T309_L5_f6_le: x_Year_Remod/Add + 998006.000001 t309_leaf5 <= 1e+06
 T309_R5_f297_ge: x_Sale_Condition_Normal - t309_leaf5 >= 0
 T309_R6_f3_ge: x_Overall_Qual - 7 t309_leaf6 >= 0
 T309_R6_f6_ge: x_Year_Remod/Add - 1994 t309_leaf6 >= 0
 T309_L6_f20_le: x_Gr_Liv_Area + 26.7600010000001 t309_leaf6 <= 3421.76
 T309_R7_f3_ge: x_Overall_Qual - 7 t309_leaf7 >= 0
 T309_R7_f6_ge: x_Year_Remod/Add - 1994 t309_leaf7 >= 0
 T309_R7_f20_ge: x_Gr_Liv_Area - 3395 t309_leaf7 >= 0
 TREE_310_ONEHOT: t310_leaf0 + t310_leaf1 + t310_leaf2 + t310_leaf3 = 1
 T310_L0_f1_le: x_Lot_Area <= 27696.999999
 T310_L0_f69_le: x_Street_Pave + 1e-06 t310_leaf0 <= 1
 T310_L1_f1_le: x_Lot_Area <= 27696.999999
 T310_R1_f69_ge: x_Street_Pave - t310_leaf1 >= 0
 T310_L1_f33_le: x_Garage_Area + 268.750001 t310_leaf1 <= 1163.75
 T310_L2_f1_le: x_Lot_Area <= 27696.999999
 T310_R2_f69_ge: x_Street_Pave - t310_leaf2 >= 0
 T310_R2_f33_ge: x_Garage_Area - 895 t310_leaf2 >= 0
 T310_R3_f1_ge: x_Lot_Area - 20697 t310_leaf3 >= 7000
 TREE_311_ONEHOT: t311_leaf0 + t311_leaf1 + t311_leaf2 + t311_leaf3
   + t311_leaf4 = 1
 T311_L0_f21_le: x_Bsmt_Full_Bath + 0.020001 t311_leaf0 <= 2.02
 T311_L0_f6_le: x_Year_Remod/Add + 997992.000001 t311_leaf0 <= 1e+06
 T311_L0_f0_le: x_Lot_Frontage + 129.530001 t311_leaf0 <= 175.53
 T311_L1_f21_le: x_Bsmt_Full_Bath + 0.020001 t311_leaf1 <= 2.02
 T311_L1_f6_le: x_Year_Remod/Add + 997992.000001 t311_leaf1 <= 1e+06
 T311_R1_f0_ge: x_Lot_Frontage - 46 t311_leaf1 >= 0
 T311_L2_f21_le: x_Bsmt_Full_Bath + 0.020001 t311_leaf2 <= 2.02
 T311_R2_f6_ge: x_Year_Remod/Add - 2008 t311_leaf2 >= 0
 T311_R3_f21_ge: x_Bsmt_Full_Bath - 2 t311_leaf3 >= 0
 T311_L3_f18_le: x_2nd_Flr_SF + 1136.540001 t311_leaf3 <= 1641.54
 T311_R4_f21_ge: x_Bsmt_Full_Bath - 2 t311_leaf4 >= 0
 T311_R4_f18_ge: x_2nd_Flr_SF - 505 t311_leaf4 >= 0
 TREE_312_ONEHOT: t312_leaf0 + t312_leaf1 + t312_leaf2 + t312_leaf3
   + t312_leaf4 + t312_leaf5 + t312_leaf6 = 1
 T312_L0_f202_le: x_Foundation_PConc + 1e-06 t312_leaf0 <= 1
 T312_L0_f31_le: x_Garage_Yr_Blt + 998037.000001 t312_leaf0 <= 1e+06
 T312_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t312_leaf0 <= 1
 T312_L1_f202_le: x_Foundation_PConc + 1e-06 t312_leaf1 <= 1
 T312_L1_f31_le: x_Garage_Yr_Blt + 998037.000001 t312_leaf1 <= 1e+06
 T312_R1_f95_ge: x_Neighborhood_Crawfor - t312_leaf1 >= 0
 T312_L2_f202_le: x_Foundation_PConc + 1e-06 t312_leaf2 <= 1
 T312_R2_f31_ge: x_Garage_Yr_Blt - 1963 t312_leaf2 >= 0
 T312_L2_f296_le: x_Sale_Condition_Family + 1e-06 t312_leaf2 <= 1
 T312_L3_f202_le: x_Foundation_PConc + 1e-06 t312_leaf3 <= 1
 T312_R3_f31_ge: x_Garage_Yr_Blt - 1963 t312_leaf3 >= 0
 T312_R3_f296_ge: x_Sale_Condition_Family - t312_leaf3 >= 0
 T312_R4_f202_ge: x_Foundation_PConc - t312_leaf4 >= 0
 T312_L4_f33_le: x_Garage_Area + 9.750001000000111 t312_leaf4 <= 1163.75
 T312_L4_f4_le: x_Overall_Cond + 999994.000001 t312_leaf4 <= 1e+06
 T312_R5_f202_ge: x_Foundation_PConc - t312_leaf5 >= 0
 T312_L5_f33_le: x_Garage_Area + 9.750001000000111 t312_leaf5 <= 1163.75
 T312_R5_f4_ge: x_Overall_Cond - 6 t312_leaf5 >= 0
 T312_R6_f202_ge: x_Foundation_PConc - t312_leaf6 >= 0
 T312_R6_f33_ge: x_Garage_Area - 1154 t312_leaf6 >= 0
 TREE_313_ONEHOT: t313_leaf0 + t313_leaf1 + t313_leaf2 + t313_leaf3 = 1
 T313_L0_f33_le: x_Garage_Area + 9.750001000000111 t313_leaf0 <= 1163.75
 T313_L0_f41_le: x_Pool_Area + 76.77000099999998 t313_leaf0 <= 444.77
 T313_L0_f213_le: x_BsmtFin_Type_1_GLQ + 1e-06 t313_leaf0 <= 1
 T313_L1_f33_le: x_Garage_Area + 9.750001000000111 t313_leaf1 <= 1163.75
 T313_L1_f41_le: x_Pool_Area + 76.77000099999998 t313_leaf1 <= 444.77
 T313_R1_f213_ge: x_BsmtFin_Type_1_GLQ - t313_leaf1 >= 0
 T313_L2_f33_le: x_Garage_Area + 9.750001000000111 t313_leaf2 <= 1163.75
 T313_R2_f41_ge: x_Pool_Area - 368 t313_leaf2 >= 0
 T313_R3_f33_ge: x_Garage_Area - 1154 t313_leaf3 >= 0
 TREE_314_ONEHOT: t314_leaf0 + t314_leaf1 + t314_leaf2 + t314_leaf3
   + t314_leaf4 = 1
 T314_L0_f14_le: x_Bsmt_Unf_SF + 395.6800010000002 t314_leaf0 <= 1812.68
 T314_L0_f44_le: x_Yr_Sold + 997992.000001 t314_leaf0 <= 1e+06
 T314_L0_f20_le: x_Gr_Liv_Area + 26.7600010000001 t314_leaf0 <= 3421.76
 T314_L1_f14_le: x_Bsmt_Unf_SF + 395.6800010000002 t314_leaf1 <= 1812.68
 T314_L1_f44_le: x_Yr_Sold + 997992.000001 t314_leaf1 <= 1e+06
 T314_R1_f20_ge: x_Gr_Liv_Area - 3395 t314_leaf1 >= 0
 T314_L2_f14_le: x_Bsmt_Unf_SF + 395.6800010000002 t314_leaf2 <= 1812.68
 T314_R2_f44_ge: x_Yr_Sold - 2008 t314_leaf2 >= 0
 T314_L2_f20_le: x_Gr_Liv_Area + 2269.760001000001 t314_leaf2 <= 3421.76
 T314_L3_f14_le: x_Bsmt_Unf_SF + 395.6800010000002 t314_leaf3 <= 1812.68
 T314_R3_f44_ge: x_Yr_Sold - 2008 t314_leaf3 >= 0
 T314_R3_f20_ge: x_Gr_Liv_Area - 1152 t314_leaf3 >= 0
 T314_R4_f14_ge: x_Bsmt_Unf_SF - 1417 t314_leaf4 >= 0
 TREE_315_ONEHOT: t315_leaf0 + t315_leaf1 + t315_leaf2 + t315_leaf3 = 1
 T315_L0_f86_le: x_Land_Slope_Gtl + 1e-06 t315_leaf0 <= 1
 T315_R1_f86_ge: x_Land_Slope_Gtl - t315_leaf1 >= 0
 T315_L1_f6_le: x_Year_Remod/Add + 997993.000001 t315_leaf1 <= 1e+06
 T315_L1_f290_le: x_Sale_Type_Oth + 1e-06 t315_leaf1 <= 1
 T315_R2_f86_ge: x_Land_Slope_Gtl - t315_leaf2 >= 0
 T315_L2_f6_le: x_Year_Remod/Add + 997993.000001 t315_leaf2 <= 1e+06
 T315_R2_f290_ge: x_Sale_Type_Oth - t315_leaf2 >= 0
 T315_R3_f86_ge: x_Land_Slope_Gtl - t315_leaf3 >= 0
 T315_R3_f6_ge: x_Year_Remod/Add - 2007 t315_leaf3 >= 0
 TREE_316_ONEHOT: t316_leaf0 + t316_leaf1 + t316_leaf2 + t316_leaf3 = 1
 T316_L0_f3_le: x_Overall_Qual + 999997.000001 t316_leaf0 <= 1e+06
 T316_R1_f3_ge: x_Overall_Qual - 3 t316_leaf1 >= 0
 T316_L1_f206_le: x_Bsmt_Exposure_Av + 1e-06 t316_leaf1 <= 1
 T316_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t316_leaf1 <= 1
 T316_R2_f3_ge: x_Overall_Qual - 3 t316_leaf2 >= 0
 T316_L2_f206_le: x_Bsmt_Exposure_Av + 1e-06 t316_leaf2 <= 1
 T316_R2_f245_ge: x_FunctioNo_aplical_Typ - t316_leaf2 >= 0
 T316_R3_f3_ge: x_Overall_Qual - 3 t316_leaf3 >= 0
 T316_R3_f206_ge: x_Bsmt_Exposure_Av - t316_leaf3 >= 0
 TREE_317_ONEHOT: t317_leaf0 + t317_leaf1 + t317_leaf2 + t317_leaf3
   + t317_leaf4 + t317_leaf5 + t317_leaf6 = 1
 T317_L0_f3_le: x_Overall_Qual + 999993.000001 t317_leaf0 <= 1e+06
 T317_L0_f18_le: x_2nd_Flr_SF + 1060.540001 t317_leaf0 <= 1641.54
 T317_L0_f209_le: x_Bsmt_Exposure_No + 1e-06 t317_leaf0 <= 1
 T317_L1_f3_le: x_Overall_Qual + 999993.000001 t317_leaf1 <= 1e+06
 T317_L1_f18_le: x_2nd_Flr_SF + 1060.540001 t317_leaf1 <= 1641.54
 T317_R1_f209_ge: x_Bsmt_Exposure_No - t317_leaf1 >= 0
 T317_L2_f3_le: x_Overall_Qual + 999993.000001 t317_leaf2 <= 1e+06
 T317_R2_f18_ge: x_2nd_Flr_SF - 581 t317_leaf2 >= 0
 T317_R3_f3_ge: x_Overall_Qual - 7 t317_leaf3 >= 0
 T317_L3_f15_le: x_Total_Bsmt_SF + 1039.480001 t317_leaf3 <= 2514.48
 T317_L3_f18_le: x_2nd_Flr_SF + 753.540001 t317_leaf3 <= 1641.54
 T317_R4_f3_ge: x_Overall_Qual - 7 t317_leaf4 >= 0
 T317_L4_f15_le: x_Total_Bsmt_SF + 1039.480001 t317_leaf4 <= 2514.48
 T317_R4_f18_ge: x_2nd_Flr_SF - 888 t317_leaf4 >= 0
 T317_R5_f3_ge: x_Overall_Qual - 7 t317_leaf5 >= 0
 T317_R5_f15_ge: x_Total_Bsmt_SF - 1475 t317_leaf5 >= 0
 T317_L5_f37_le: x_Open_Porch_SF + 372.810001 t317_leaf5 <= 488.81
 T317_R6_f3_ge: x_Overall_Qual - 7 t317_leaf6 >= 0
 T317_R6_f15_ge: x_Total_Bsmt_SF - 1475 t317_leaf6 >= 0
 T317_R6_f37_ge: x_Open_Porch_SF - 116 t317_leaf6 >= 0
 TREE_318_ONEHOT: t318_leaf0 + t318_leaf1 + t318_leaf2 + t318_leaf3
   + t318_leaf4 = 1
 T318_L0_f7_le: x_Mas_Vnr_Area + 472.070001 t318_leaf0 <= 931.07
 T318_L0_f68_le: x_Street_Grvl + 1e-06 t318_leaf0 <= 1
 T318_L0_f134_le: x_Bldg_Type_1Fam + 1e-06 t318_leaf0 <= 1
 T318_L1_f7_le: x_Mas_Vnr_Area + 472.070001 t318_leaf1 <= 931.07
 T318_L1_f68_le: x_Street_Grvl + 1e-06 t318_leaf1 <= 1
 T318_R1_f134_ge: x_Bldg_Type_1Fam - t318_leaf1 >= 0
 T318_L2_f7_le: x_Mas_Vnr_Area + 472.070001 t318_leaf2 <= 931.07
 T318_R2_f68_ge: x_Street_Grvl - t318_leaf2 >= 0
 T318_R3_f7_ge: x_Mas_Vnr_Area - 459 t318_leaf3 >= 0
 T318_L3_f36_le: x_Wood_Deck_SF + 125.900001 t318_leaf3 <= 525.9
 T318_R4_f7_ge: x_Mas_Vnr_Area - 459 t318_leaf4 >= 0
 T318_R4_f36_ge: x_Wood_Deck_SF - 400 t318_leaf4 >= 0
 TREE_319_ONEHOT: t319_leaf0 + t319_leaf1 + t319_leaf2 + t319_leaf3
   + t319_leaf4 + t319_leaf5 + t319_leaf6 = 1
 T319_L0_f44_le: x_Yr_Sold + 997991.000001 t319_leaf0 <= 1e+06
 T319_L0_f3_le: x_Overall_Qual + 999993.000001 t319_leaf0 <= 1e+06
 T319_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t319_leaf0 <= 1
 T319_L1_f44_le: x_Yr_Sold + 997991.000001 t319_leaf1 <= 1e+06
 T319_L1_f3_le: x_Overall_Qual + 999993.000001 t319_leaf1 <= 1e+06
 T319_R1_f297_ge: x_Sale_Condition_Normal - t319_leaf1 >= 0
 T319_L2_f44_le: x_Yr_Sold + 997991.000001 t319_leaf2 <= 1e+06
 T319_R2_f3_ge: x_Overall_Qual - 7 t319_leaf2 >= 0
 T319_L2_f40_le: x_Screen_Porch + 160.270001 t319_leaf2 <= 268.27
 T319_L3_f44_le: x_Yr_Sold + 997991.000001 t319_leaf3 <= 1e+06
 T319_R3_f3_ge: x_Overall_Qual - 7 t319_leaf3 >= 0
 T319_R3_f40_ge: x_Screen_Porch - 108 t319_leaf3 >= 0
 T319_R4_f44_ge: x_Yr_Sold - 2009 t319_leaf4 >= 0
 T319_L4_f5_le: x_Year_Built + 998085.000001 t319_leaf4 <= 1e+06
 T319_R5_f44_ge: x_Yr_Sold - 2009 t319_leaf5 >= 0
 T319_R5_f5_ge: x_Year_Built - 1915 t319_leaf5 >= 0
 T319_L5_f148_le: x_Roof_Style_Gable + 1e-06 t319_leaf5 <= 1
 T319_R6_f44_ge: x_Yr_Sold - 2009 t319_leaf6 >= 0
 T319_R6_f5_ge: x_Year_Built - 1915 t319_leaf6 >= 0
 T319_R6_f148_ge: x_Roof_Style_Gable - t319_leaf6 >= 0
 TREE_320_ONEHOT: t320_leaf0 + t320_leaf1 + t320_leaf2 + t320_leaf3
   + t320_leaf4 + t320_leaf5 = 1
 T320_L0_f37_le: x_Open_Porch_SF + 196.810001 t320_leaf0 <= 488.81
 T320_L0_f37_le: x_Open_Porch_SF + 320.810001 t320_leaf0 <= 488.81
 T320_L0_f272_le: x_Mo_Sold_2 + 1e-06 t320_leaf0 <= 1
 T320_L1_f37_le: x_Open_Porch_SF + 196.810001 t320_leaf1 <= 488.81
 T320_L1_f37_le: x_Open_Porch_SF + 320.810001 t320_leaf1 <= 488.81
 T320_R1_f272_ge: x_Mo_Sold_2 - t320_leaf1 >= 0
 T320_L2_f37_le: x_Open_Porch_SF + 196.810001 t320_leaf2 <= 488.81
 T320_R2_f37_ge: x_Open_Porch_SF - 168 t320_leaf2 >= 0
 T320_L2_f150_le: x_Roof_Style_Hip + 1e-06 t320_leaf2 <= 1
 T320_L3_f37_le: x_Open_Porch_SF + 196.810001 t320_leaf3 <= 488.81
 T320_R3_f37_ge: x_Open_Porch_SF - 168 t320_leaf3 >= 0
 T320_R3_f150_ge: x_Roof_Style_Hip - t320_leaf3 >= 0
 T320_R4_f37_ge: x_Open_Porch_SF - 292 t320_leaf4 >= 0
 T320_L4_f12_le: x_BsmtFin_SF_1 + 1612.500001 t320_leaf4 <= 1984.5
 T320_R5_f37_ge: x_Open_Porch_SF - 292 t320_leaf5 >= 0
 T320_R5_f12_ge: x_BsmtFin_SF_1 - 372 t320_leaf5 >= 0
 TREE_321_ONEHOT: t321_leaf0 + t321_leaf1 + t321_leaf2 + t321_leaf3 = 1
 T321_L0_f4_le: x_Overall_Cond + 999993.000001 t321_leaf0 <= 1e+06
 T321_L0_f37_le: x_Open_Porch_SF + 44.810001 t321_leaf0 <= 488.81
 T321_L0_f219_le: x_BsmtFin_Type_2_BLQ + 1e-06 t321_leaf0 <= 1
 T321_L1_f4_le: x_Overall_Cond + 999993.000001 t321_leaf1 <= 1e+06
 T321_L1_f37_le: x_Open_Porch_SF + 44.810001 t321_leaf1 <= 488.81
 T321_R1_f219_ge: x_BsmtFin_Type_2_BLQ - t321_leaf1 >= 0
 T321_L2_f4_le: x_Overall_Cond + 999993.000001 t321_leaf2 <= 1e+06
 T321_R2_f37_ge: x_Open_Porch_SF - 444 t321_leaf2 >= 0
 T321_R3_f4_ge: x_Overall_Cond - 7 t321_leaf3 >= 0
 TREE_322_ONEHOT: t322_leaf0 + t322_leaf1 + t322_leaf2 + t322_leaf3
   + t322_leaf4 + t322_leaf5 = 1
 T322_L0_f3_le: x_Overall_Qual + 999996.000001 t322_leaf0 <= 1e+06
 T322_L0_f169_le: x_Exterior_1st_MetalSd + 1e-06 t322_leaf0 <= 1
 T322_L1_f3_le: x_Overall_Qual + 999996.000001 t322_leaf1 <= 1e+06
 T322_R1_f169_ge: x_Exterior_1st_MetalSd - t322_leaf1 >= 0
 T322_R2_f3_ge: x_Overall_Qual - 4 t322_leaf2 >= 0
 T322_L2_f117_le: x_Condition_1_Artery + 1e-06 t322_leaf2 <= 1
 T322_L2_f154_le: x_Roof_Matl_CompShg + 1e-06 t322_leaf2 <= 1
 T322_R3_f3_ge: x_Overall_Qual - 4 t322_leaf3 >= 0
 T322_L3_f117_le: x_Condition_1_Artery + 1e-06 t322_leaf3 <= 1
 T322_R3_f154_ge: x_Roof_Matl_CompShg - t322_leaf3 >= 0
 T322_R4_f3_ge: x_Overall_Qual - 4 t322_leaf4 >= 0
 T322_R4_f117_ge: x_Condition_1_Artery - t322_leaf4 >= 0
 T322_L4_f17_le: x_1st_Flr_SF + 1236.900001 t322_leaf4 <= 2534.9
 T322_R5_f3_ge: x_Overall_Qual - 4 t322_leaf5 >= 0
 T322_R5_f117_ge: x_Condition_1_Artery - t322_leaf5 >= 0
 T322_R5_f17_ge: x_1st_Flr_SF - 1298 t322_leaf5 >= 0
 TREE_323_ONEHOT: t323_leaf0 + t323_leaf1 + t323_leaf2 + t323_leaf3
   + t323_leaf4 = 1
 T323_L0_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t323_leaf0 <= 1
 T323_L0_f51_le: x_MS_SubClass_70 + 1e-06 t323_leaf0 <= 1
 T323_L0_f5_le: x_Year_Built + 998068.000001 t323_leaf0 <= 1e+06
 T323_L1_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t323_leaf1 <= 1
 T323_L1_f51_le: x_MS_SubClass_70 + 1e-06 t323_leaf1 <= 1
 T323_R1_f5_ge: x_Year_Built - 1932 t323_leaf1 >= 0
 T323_L2_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t323_leaf2 <= 1
 T323_R2_f51_ge: x_MS_SubClass_70 - t323_leaf2 >= 0
 T323_L2_f80_le: x_Land_Contour_Lvl + 1e-06 t323_leaf2 <= 1
 T323_L3_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t323_leaf3 <= 1
 T323_R3_f51_ge: x_MS_SubClass_70 - t323_leaf3 >= 0
 T323_R3_f80_ge: x_Land_Contour_Lvl - t323_leaf3 >= 0
 T323_R4_f239_ge: x_FunctioNo_aplical_Maj2 - t323_leaf4 >= 0
 TREE_324_ONEHOT: t324_leaf0 + t324_leaf1 + t324_leaf2 + t324_leaf3
   + t324_leaf4 + t324_leaf5 + t324_leaf6 = 1
 T324_L0_f5_le: x_Year_Built + 998075.000001 t324_leaf0 <= 1e+06
 T324_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t324_leaf0 <= 1
 T324_L0_f44_le: x_Yr_Sold + 997992.000001 t324_leaf0 <= 1e+06
 T324_L1_f5_le: x_Year_Built + 998075.000001 t324_leaf1 <= 1e+06
 T324_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t324_leaf1 <= 1
 T324_R1_f44_ge: x_Yr_Sold - 2008 t324_leaf1 >= 0
 T324_L2_f5_le: x_Year_Built + 998075.000001 t324_leaf2 <= 1e+06
 T324_R2_f293_ge: x_Sale_Condition_Abnorml - t324_leaf2 >= 0
 T324_L2_f0_le: x_Lot_Frontage + 106.530001 t324_leaf2 <= 175.53
 T324_L3_f5_le: x_Year_Built + 998075.000001 t324_leaf3 <= 1e+06
 T324_R3_f293_ge: x_Sale_Condition_Abnorml - t324_leaf3 >= 0
 T324_R3_f0_ge: x_Lot_Frontage - 69 t324_leaf3 >= 0
 T324_R4_f5_ge: x_Year_Built - 1925 t324_leaf4 >= 0
 T324_L4_f37_le: x_Open_Porch_SF + 293.810001 t324_leaf4 <= 488.81
 T324_L4_f21_le: x_Bsmt_Full_Bath + 0.020001 t324_leaf4 <= 2.02
 T324_R5_f5_ge: x_Year_Built - 1925 t324_leaf5 >= 0
 T324_L5_f37_le: x_Open_Porch_SF + 293.810001 t324_leaf5 <= 488.81
 T324_R5_f21_ge: x_Bsmt_Full_Bath - 2 t324_leaf5 >= 0
 T324_R6_f5_ge: x_Year_Built - 1925 t324_leaf6 >= 0
 T324_R6_f37_ge: x_Open_Porch_SF - 195 t324_leaf6 >= 0
 TREE_325_ONEHOT: t325_leaf0 + t325_leaf1 + t325_leaf2 + t325_leaf3
   + t325_leaf4 + t325_leaf5 = 1
 T325_L0_f1_le: x_Lot_Area <= 16778.999999
 T325_L0_f26_le: x_Kitchen_AbvGr + 1.000001 t325_leaf0 <= 3
 T325_L0_f14_le: x_Bsmt_Unf_SF + 1336.680001 t325_leaf0 <= 1812.68
 T325_L1_f1_le: x_Lot_Area <= 16778.999999
 T325_L1_f26_le: x_Kitchen_AbvGr + 1.000001 t325_leaf1 <= 3
 T325_R1_f14_ge: x_Bsmt_Unf_SF - 476 t325_leaf1 >= 0
 T325_L2_f1_le: x_Lot_Area <= 16778.999999
 T325_R2_f26_ge: x_Kitchen_AbvGr - 2 t325_leaf2 >= 0
 T325_L2_f32_le: x_Garage_Cars + 3.000001 t325_leaf2 <= 4
 T325_L3_f1_le: x_Lot_Area <= 16778.999999
 T325_R3_f26_ge: x_Kitchen_AbvGr - 2 t325_leaf3 >= 0
 T325_R3_f32_ge: x_Garage_Cars - t325_leaf3 >= 0
 T325_R4_f1_ge: x_Lot_Area - 9779 t325_leaf4 >= 7000
 T325_L4_f77_le: x_Land_Contour_Bnk + 1e-06 t325_leaf4 <= 1
 T325_R5_f1_ge: x_Lot_Area - 9779 t325_leaf5 >= 7000
 T325_R5_f77_ge: x_Land_Contour_Bnk - t325_leaf5 >= 0
 TREE_326_ONEHOT: t326_leaf0 + t326_leaf1 + t326_leaf2 + t326_leaf3
   + t326_leaf4 + t326_leaf5 = 1
 T326_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t326_leaf0 <= 1
 T326_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t326_leaf0 <= 1
 T326_L0_f34_le: x_Garage_Qual + 3.000001 t326_leaf0 <= 4
 T326_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t326_leaf1 <= 1
 T326_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t326_leaf1 <= 1
 T326_R1_f34_ge: x_Garage_Qual >= 1
 T326_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t326_leaf2 <= 1
 T326_R2_f245_ge: x_FunctioNo_aplical_Typ - t326_leaf2 >= 0
 T326_L2_f121_le: x_Condition_1_PosN + 1e-06 t326_leaf2 <= 1
 T326_L3_f164_le: x_Exterior_1st_BrkFace + 1e-06 t326_leaf3 <= 1
 T326_R3_f245_ge: x_FunctioNo_aplical_Typ - t326_leaf3 >= 0
 T326_R3_f121_ge: x_Condition_1_PosN - t326_leaf3 >= 0
 T326_R4_f164_ge: x_Exterior_1st_BrkFace - t326_leaf4 >= 0
 T326_L4_f20_le: x_Gr_Liv_Area + 2126.760001000001 t326_leaf4 <= 3421.76
 T326_R5_f164_ge: x_Exterior_1st_BrkFace - t326_leaf5 >= 0
 T326_R5_f20_ge: x_Gr_Liv_Area - 1295 t326_leaf5 >= 0
 TREE_327_ONEHOT: t327_leaf0 + t327_leaf1 + t327_leaf2 + t327_leaf3
   + t327_leaf4 = 1
 T327_L0_f113_le: x_Neighborhood_Somerst + 1e-06 t327_leaf0 <= 1
 T327_L0_f4_le: x_Overall_Cond + 999993.000001 t327_leaf0 <= 1e+06
 T327_L0_f213_le: x_BsmtFin_Type_1_GLQ + 1e-06 t327_leaf0 <= 1
 T327_L1_f113_le: x_Neighborhood_Somerst + 1e-06 t327_leaf1 <= 1
 T327_L1_f4_le: x_Overall_Cond + 999993.000001 t327_leaf1 <= 1e+06
 T327_R1_f213_ge: x_BsmtFin_Type_1_GLQ - t327_leaf1 >= 0
 T327_L2_f113_le: x_Neighborhood_Somerst + 1e-06 t327_leaf2 <= 1
 T327_R2_f4_ge: x_Overall_Cond - 7 t327_leaf2 >= 0
 T327_L2_f44_le: x_Yr_Sold + 997991.000001 t327_leaf2 <= 1e+06
 T327_L3_f113_le: x_Neighborhood_Somerst + 1e-06 t327_leaf3 <= 1
 T327_R3_f4_ge: x_Overall_Cond - 7 t327_leaf3 >= 0
 T327_R3_f44_ge: x_Yr_Sold - 2009 t327_leaf3 >= 0
 T327_R4_f113_ge: x_Neighborhood_Somerst - t327_leaf4 >= 0
 TREE_328_ONEHOT: t328_leaf0 + t328_leaf1 + t328_leaf2 + t328_leaf3
   + t328_leaf4 = 1
 T328_L0_f44_le: x_Yr_Sold + 997993.000001 t328_leaf0 <= 1e+06
 T328_R1_f44_ge: x_Yr_Sold - 2007 t328_leaf1 >= 0
 T328_L1_f15_le: x_Total_Bsmt_SF + 1759.480001 t328_leaf1 <= 2514.48
 T328_L1_f86_le: x_Land_Slope_Gtl + 1e-06 t328_leaf1 <= 1
 T328_R2_f44_ge: x_Yr_Sold - 2007 t328_leaf2 >= 0
 T328_L2_f15_le: x_Total_Bsmt_SF + 1759.480001 t328_leaf2 <= 2514.48
 T328_R2_f86_ge: x_Land_Slope_Gtl - t328_leaf2 >= 0
 T328_R3_f44_ge: x_Yr_Sold - 2007 t328_leaf3 >= 0
 T328_R3_f15_ge: x_Total_Bsmt_SF - 755 t328_leaf3 >= 0
 T328_L3_f258_le: x_Paved_Drive_P + 1e-06 t328_leaf3 <= 1
 T328_R4_f44_ge: x_Yr_Sold - 2007 t328_leaf4 >= 0
 T328_R4_f15_ge: x_Total_Bsmt_SF - 755 t328_leaf4 >= 0
 T328_R4_f258_ge: x_Paved_Drive_P - t328_leaf4 >= 0
 TREE_329_ONEHOT: t329_leaf0 + t329_leaf1 + t329_leaf2 + t329_leaf3
   + t329_leaf4 + t329_leaf5 = 1
 T329_L0_f14_le: x_Bsmt_Unf_SF + 1699.680001 t329_leaf0 <= 1812.68
 T329_L0_f14_le: x_Bsmt_Unf_SF + 1707.680001 t329_leaf0 <= 1812.68
 T329_L1_f14_le: x_Bsmt_Unf_SF + 1699.680001 t329_leaf1 <= 1812.68
 T329_R1_f14_ge: x_Bsmt_Unf_SF - 105 t329_leaf1 >= 0
 T329_R2_f14_ge: x_Bsmt_Unf_SF - 113 t329_leaf2 >= 0
 T329_L2_f40_le: x_Screen_Porch + 160.270001 t329_leaf2 <= 268.27
 T329_L2_f37_le: x_Open_Porch_SF + 123.810001 t329_leaf2 <= 488.81
 T329_R3_f14_ge: x_Bsmt_Unf_SF - 113 t329_leaf3 >= 0
 T329_L3_f40_le: x_Screen_Porch + 160.270001 t329_leaf3 <= 268.27
 T329_R3_f37_ge: x_Open_Porch_SF - 365 t329_leaf3 >= 0
 T329_R4_f14_ge: x_Bsmt_Unf_SF - 113 t329_leaf4 >= 0
 T329_R4_f40_ge: x_Screen_Porch - 108 t329_leaf4 >= 0
 T329_L4_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t329_leaf4 <= 1
 T329_R5_f14_ge: x_Bsmt_Unf_SF - 113 t329_leaf5 >= 0
 T329_R5_f40_ge: x_Screen_Porch - 108 t329_leaf5 >= 0
 T329_R5_f216_ge: x_BsmtFin_Type_1_Rec - t329_leaf5 >= 0
 TREE_330_ONEHOT: t330_leaf0 + t330_leaf1 + t330_leaf2 + t330_leaf3
   + t330_leaf4 + t330_leaf5 = 1
 T330_L0_f1_le: x_Lot_Area <= 18385.999999
 T330_L0_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t330_leaf0 <= 1
 T330_L0_f92_le: x_Neighborhood_BrkSide + 1e-06 t330_leaf0 <= 1
 T330_L1_f1_le: x_Lot_Area <= 18385.999999
 T330_L1_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t330_leaf1 <= 1
 T330_R1_f92_ge: x_Neighborhood_BrkSide - t330_leaf1 >= 0
 T330_L2_f1_le: x_Lot_Area <= 18385.999999
 T330_R2_f239_ge: x_FunctioNo_aplical_Maj2 - t330_leaf2 >= 0
 T330_R3_f1_ge: x_Lot_Area - 11386 t330_leaf3 >= 7000
 T330_L3_f209_le: x_Bsmt_Exposure_No + 1e-06 t330_leaf3 <= 1
 T330_R4_f1_ge: x_Lot_Area - 11386 t330_leaf4 >= 7000
 T330_R4_f209_ge: x_Bsmt_Exposure_No - t330_leaf4 >= 0
 T330_L4_f247_le: x_Garage_Type_Attchd + 1e-06 t330_leaf4 <= 1
 T330_R5_f1_ge: x_Lot_Area - 11386 t330_leaf5 >= 7000
 T330_R5_f209_ge: x_Bsmt_Exposure_No - t330_leaf5 >= 0
 T330_R5_f247_ge: x_Garage_Type_Attchd - t330_leaf5 >= 0
 TREE_331_ONEHOT: t331_leaf0 + t331_leaf1 + t331_leaf2 + t331_leaf3
   + t331_leaf4 + t331_leaf5 = 1
 T331_L0_f37_le: x_Open_Porch_SF + 436.810001 t331_leaf0 <= 488.81
 T331_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t331_leaf0 <= 1
 T331_L0_f33_le: x_Garage_Area + 716.750001 t331_leaf0 <= 1163.75
 T331_L1_f37_le: x_Open_Porch_SF + 436.810001 t331_leaf1 <= 488.81
 T331_L1_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t331_leaf1 <= 1
 T331_R1_f33_ge: x_Garage_Area - 447 t331_leaf1 >= 0
 T331_L2_f37_le: x_Open_Porch_SF + 436.810001 t331_leaf2 <= 488.81
 T331_R2_f207_ge: x_Bsmt_Exposure_Gd - t331_leaf2 >= 0
 T331_R3_f37_ge: x_Open_Porch_SF - 52 t331_leaf3 >= 0
 T331_L3_f92_le: x_Neighborhood_BrkSide + 1e-06 t331_leaf3 <= 1
 T331_L3_f37_le: x_Open_Porch_SF + 44.810001 t331_leaf3 <= 488.81
 T331_R4_f37_ge: x_Open_Porch_SF - 52 t331_leaf4 >= 0
 T331_L4_f92_le: x_Neighborhood_BrkSide + 1e-06 t331_leaf4 <= 1
 T331_R4_f37_ge: x_Open_Porch_SF - 444 t331_leaf4 >= 0
 T331_R5_f37_ge: x_Open_Porch_SF - 52 t331_leaf5 >= 0
 T331_R5_f92_ge: x_Neighborhood_BrkSide - t331_leaf5 >= 0
 TREE_332_ONEHOT: t332_leaf0 + t332_leaf1 + t332_leaf2 + t332_leaf3
   + t332_leaf4 + t332_leaf5 = 1
 T332_L0_f4_le: x_Overall_Cond + 999996.000001 t332_leaf0 <= 1e+06
 T332_L0_f257_le: x_Paved_Drive_N + 1e-06 t332_leaf0 <= 1
 T332_L1_f4_le: x_Overall_Cond + 999996.000001 t332_leaf1 <= 1e+06
 T332_R1_f257_ge: x_Paved_Drive_N - t332_leaf1 >= 0
 T332_R2_f4_ge: x_Overall_Cond - 4 t332_leaf2 >= 0
 T332_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t332_leaf2 <= 1
 T332_L2_f35_le: x_Garage_Cond + 3.000001 t332_leaf2 <= 4
 T332_R3_f4_ge: x_Overall_Cond - 4 t332_leaf3 >= 0
 T332_L3_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t332_leaf3 <= 1
 T332_R3_f35_ge: x_Garage_Cond - 2 t332_leaf3 >= -1
 T332_R4_f4_ge: x_Overall_Cond - 4 t332_leaf4 >= 0
 T332_R4_f245_ge: x_FunctioNo_aplical_Typ - t332_leaf4 >= 0
 T332_L4_f4_le: x_Overall_Cond + 999992.000001 t332_leaf4 <= 1e+06
 T332_R5_f4_ge: x_Overall_Cond - 4 t332_leaf5 >= 0
 T332_R5_f245_ge: x_FunctioNo_aplical_Typ - t332_leaf5 >= 0
 T332_R5_f4_ge: x_Overall_Cond - 8 t332_leaf5 >= 0
 TREE_333_ONEHOT: t333_leaf0 + t333_leaf1 + t333_leaf2 + t333_leaf3 = 1
 T333_L0_f58_le: x_MS_SubClass_160 + 1e-06 t333_leaf0 <= 1
 T333_L0_f3_le: x_Overall_Qual + 999997.000001 t333_leaf0 <= 1e+06
 T333_L1_f58_le: x_MS_SubClass_160 + 1e-06 t333_leaf1 <= 1
 T333_R1_f3_ge: x_Overall_Qual - 3 t333_leaf1 >= 0
 T333_L1_f113_le: x_Neighborhood_Somerst + 1e-06 t333_leaf1 <= 1
 T333_L2_f58_le: x_MS_SubClass_160 + 1e-06 t333_leaf2 <= 1
 T333_R2_f3_ge: x_Overall_Qual - 3 t333_leaf2 >= 0
 T333_R2_f113_ge: x_Neighborhood_Somerst - t333_leaf2 >= 0
 T333_R3_f58_ge: x_MS_SubClass_160 - t333_leaf3 >= 0
 TREE_334_ONEHOT: t334_leaf0 + t334_leaf1 + t334_leaf2 + t334_leaf3
   + t334_leaf4 + t334_leaf5 = 1
 T334_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t334_leaf0 <= 1
 T334_L0_f4_le: x_Overall_Cond + 999994.000001 t334_leaf0 <= 1e+06
 T334_L0_f8_le: x_Exter_Qual + 2.000001 t334_leaf0 <= 4
 T334_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t334_leaf1 <= 1
 T334_L1_f4_le: x_Overall_Cond + 999994.000001 t334_leaf1 <= 1e+06
 T334_R1_f8_ge: x_Exter_Qual >= 2
 T334_L2_f161_le: x_Exterior_1st_AsbShng + 1e-06 t334_leaf2 <= 1
 T334_R2_f4_ge: x_Overall_Cond - 6 t334_leaf2 >= 0
 T334_L2_f20_le: x_Gr_Liv_Area + 2436.760001 t334_leaf2 <= 3421.76
 T334_L3_f161_le: x_Exterior_1st_AsbShng + 1e-06 t334_leaf3 <= 1
 T334_R3_f4_ge: x_Overall_Cond - 6 t334_leaf3 >= 0
 T334_R3_f20_ge: x_Gr_Liv_Area - 985 t334_leaf3 >= 0
 T334_R4_f161_ge: x_Exterior_1st_AsbShng - t334_leaf4 >= 0
 T334_L4_f33_le: x_Garage_Area + 722.750001 t334_leaf4 <= 1163.75
 T334_R5_f161_ge: x_Exterior_1st_AsbShng - t334_leaf5 >= 0
 T334_R5_f33_ge: x_Garage_Area - 441 t334_leaf5 >= 0
 TREE_335_ONEHOT: t335_leaf0 + t335_leaf1 + t335_leaf2 + t335_leaf3
   + t335_leaf4 + t335_leaf5 + t335_leaf6 + t335_leaf7 = 1
 T335_L0_f109_le: x_Neighborhood_OldTown + 1e-06 t335_leaf0 <= 1
 T335_L0_f3_le: x_Overall_Qual + 999993.000001 t335_leaf0 <= 1e+06
 T335_L0_f20_le: x_Gr_Liv_Area + 1672.760001 t335_leaf0 <= 3421.76
 T335_L1_f109_le: x_Neighborhood_OldTown + 1e-06 t335_leaf1 <= 1
 T335_L1_f3_le: x_Overall_Qual + 999993.000001 t335_leaf1 <= 1e+06
 T335_R1_f20_ge: x_Gr_Liv_Area - 1749 t335_leaf1 >= 0
 T335_L2_f109_le: x_Neighborhood_OldTown + 1e-06 t335_leaf2 <= 1
 T335_R2_f3_ge: x_Overall_Qual - 7 t335_leaf2 >= 0
 T335_L2_f6_le: x_Year_Remod/Add + 997994.000001 t335_leaf2 <= 1e+06
 T335_L3_f109_le: x_Neighborhood_OldTown + 1e-06 t335_leaf3 <= 1
 T335_R3_f3_ge: x_Overall_Qual - 7 t335_leaf3 >= 0
 T335_R3_f6_ge: x_Year_Remod/Add - 2006 t335_leaf3 >= 0
 T335_R4_f109_ge: x_Neighborhood_OldTown - t335_leaf4 >= 0
 T335_L4_f15_le: x_Total_Bsmt_SF + 1834.480001 t335_leaf4 <= 2514.48
 T335_L4_f16_le: x_Heating_QC + 1.000001 t335_leaf4 <= 4
 T335_R5_f109_ge: x_Neighborhood_OldTown - t335_leaf5 >= 0
 T335_L5_f15_le: x_Total_Bsmt_SF + 1834.480001 t335_leaf5 <= 2514.48
 T335_R5_f16_ge: x_Heating_QC - t335_leaf5 >= 2
 T335_R6_f109_ge: x_Neighborhood_OldTown - t335_leaf6 >= 0
 T335_R6_f15_ge: x_Total_Bsmt_SF - 680 t335_leaf6 >= 0
 T335_L6_f9_le: x_Exter_Cond + 2.000001 t335_leaf6 <= 4
 T335_R7_f109_ge: x_Neighborhood_OldTown - t335_leaf7 >= 0
 T335_R7_f15_ge: x_Total_Bsmt_SF - 680 t335_leaf7 >= 0
 T335_R7_f9_ge: x_Exter_Cond - 3 t335_leaf7 >= -1
 TREE_336_ONEHOT: t336_leaf0 + t336_leaf1 + t336_leaf2 + t336_leaf3
   + t336_leaf4 = 1
 T336_L0_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t336_leaf0 <= 1e+06
 T336_L0_f37_le: x_Open_Porch_SF + 363.810001 t336_leaf0 <= 488.81
 T336_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t336_leaf0 <= 1
 T336_L1_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t336_leaf1 <= 1e+06
 T336_L1_f37_le: x_Open_Porch_SF + 363.810001 t336_leaf1 <= 488.81
 T336_R1_f297_ge: x_Sale_Condition_Normal - t336_leaf1 >= 0
 T336_L2_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t336_leaf2 <= 1e+06
 T336_R2_f37_ge: x_Open_Porch_SF - 125 t336_leaf2 >= 0
 T336_L2_f81_le: x_Lot_Config_Corner + 1e-06 t336_leaf2 <= 1
 T336_L3_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t336_leaf3 <= 1e+06
 T336_R3_f37_ge: x_Open_Porch_SF - 125 t336_leaf3 >= 0
 T336_R3_f81_ge: x_Lot_Config_Corner - t336_leaf3 >= 0
 T336_R4_f19_ge: x_Low_Qual_Fin_SF - 108 t336_leaf4 >= 0
 TREE_337_ONEHOT: t337_leaf0 + t337_leaf1 + t337_leaf2 + t337_leaf3
   + t337_leaf4 = 1
 T337_L0_f5_le: x_Year_Built + 998085.000001 t337_leaf0 <= 1e+06
 T337_L0_f15_le: x_Total_Bsmt_SF + 1866.480001 t337_leaf0 <= 2514.48
 T337_L1_f5_le: x_Year_Built + 998085.000001 t337_leaf1 <= 1e+06
 T337_R1_f15_ge: x_Total_Bsmt_SF - 648 t337_leaf1 >= 0
 T337_R2_f5_ge: x_Year_Built - 1915 t337_leaf2 >= 0
 T337_L2_f1_le: x_Lot_Area <= 27696.999999
 T337_L2_f5_le: x_Year_Built + 997995.000001 t337_leaf2 <= 1e+06
 T337_R3_f5_ge: x_Year_Built - 1915 t337_leaf3 >= 0
 T337_L3_f1_le: x_Lot_Area <= 27696.999999
 T337_R3_f5_ge: x_Year_Built - 2005 t337_leaf3 >= 0
 T337_R4_f5_ge: x_Year_Built - 1915 t337_leaf4 >= 0
 T337_R4_f1_ge: x_Lot_Area - 20697 t337_leaf4 >= 7000
 TREE_338_ONEHOT: t338_leaf0 + t338_leaf1 + t338_leaf2 + t338_leaf3
   + t338_leaf4 + t338_leaf5 = 1
 T338_L0_f3_le: x_Overall_Qual + 999993.000001 t338_leaf0 <= 1e+06
 T338_L0_f18_le: x_2nd_Flr_SF + 969.540001 t338_leaf0 <= 1641.54
 T338_L0_f20_le: x_Gr_Liv_Area + 1937.760001 t338_leaf0 <= 3421.76
 T338_L1_f3_le: x_Overall_Qual + 999993.000001 t338_leaf1 <= 1e+06
 T338_L1_f18_le: x_2nd_Flr_SF + 969.540001 t338_leaf1 <= 1641.54
 T338_R1_f20_ge: x_Gr_Liv_Area - 1484 t338_leaf1 >= 0
 T338_L2_f3_le: x_Overall_Qual + 999993.000001 t338_leaf2 <= 1e+06
 T338_R2_f18_ge: x_2nd_Flr_SF - 672 t338_leaf2 >= 0
 T338_R3_f3_ge: x_Overall_Qual - 7 t338_leaf3 >= 0
 T338_L3_f33_le: x_Garage_Area + 9.750001000000111 t338_leaf3 <= 1163.75
 T338_L3_f3_le: x_Overall_Qual + 999990.000001 t338_leaf3 <= 1e+06
 T338_R4_f3_ge: x_Overall_Qual - 7 t338_leaf4 >= 0
 T338_L4_f33_le: x_Garage_Area + 9.750001000000111 t338_leaf4 <= 1163.75
 T338_R4_f3_ge: x_Overall_Qual - 10 t338_leaf4 >= 0
 T338_R5_f3_ge: x_Overall_Qual - 7 t338_leaf5 >= 0
 T338_R5_f33_ge: x_Garage_Area - 1154 t338_leaf5 >= 0
 TREE_339_ONEHOT: t339_leaf0 + t339_leaf1 + t339_leaf2 + t339_leaf3
   + t339_leaf4 = 1
 T339_L0_f296_le: x_Sale_Condition_Family + 1e-06 t339_leaf0 <= 1
 T339_L0_f1_le: x_Lot_Area <= 19521.999999
 T339_L0_f109_le: x_Neighborhood_OldTown + 1e-06 t339_leaf0 <= 1
 T339_L1_f296_le: x_Sale_Condition_Family + 1e-06 t339_leaf1 <= 1
 T339_L1_f1_le: x_Lot_Area <= 19521.999999
 T339_R1_f109_ge: x_Neighborhood_OldTown - t339_leaf1 >= 0
 T339_L2_f296_le: x_Sale_Condition_Family + 1e-06 t339_leaf2 <= 1
 T339_R2_f1_ge: x_Lot_Area - 12522 t339_leaf2 >= 7000
 T339_R3_f296_ge: x_Sale_Condition_Family - t339_leaf3 >= 0
 T339_L3_f33_le: x_Garage_Area + 681.750001 t339_leaf3 <= 1163.75
 T339_R4_f296_ge: x_Sale_Condition_Family - t339_leaf4 >= 0
 T339_R4_f33_ge: x_Garage_Area - 482 t339_leaf4 >= 0
 TREE_340_ONEHOT: t340_leaf0 + t340_leaf1 + t340_leaf2 + t340_leaf3
   + t340_leaf4 + t340_leaf5 + t340_leaf6 = 1
 T340_L0_f31_le: x_Garage_Yr_Blt + 998059.000001 t340_leaf0 <= 1e+06
 T340_L0_f1_le: x_Lot_Area <= 8815.999999
 T340_L1_f31_le: x_Garage_Yr_Blt + 998059.000001 t340_leaf1 <= 1e+06
 T340_R1_f1_ge: x_Lot_Area - 1816 t340_leaf1 >= 7000
 T340_L1_f117_le: x_Condition_1_Artery + 1e-06 t340_leaf1 <= 1
 T340_L2_f31_le: x_Garage_Yr_Blt + 998059.000001 t340_leaf2 <= 1e+06
 T340_R2_f1_ge: x_Lot_Area - 1816 t340_leaf2 >= 7000
 T340_R2_f117_ge: x_Condition_1_Artery - t340_leaf2 >= 0
 T340_R3_f31_ge: x_Garage_Yr_Blt - 1941 t340_leaf3 >= 0
 T340_L3_f18_le: x_2nd_Flr_SF + 647.540001 t340_leaf3 <= 1641.54
 T340_L3_f154_le: x_Roof_Matl_CompShg + 1e-06 t340_leaf3 <= 1
 T340_R4_f31_ge: x_Garage_Yr_Blt - 1941 t340_leaf4 >= 0
 T340_L4_f18_le: x_2nd_Flr_SF + 647.540001 t340_leaf4 <= 1641.54
 T340_R4_f154_ge: x_Roof_Matl_CompShg - t340_leaf4 >= 0
 T340_R5_f31_ge: x_Garage_Yr_Blt - 1941 t340_leaf5 >= 0
 T340_R5_f18_ge: x_2nd_Flr_SF - 994 t340_leaf5 >= 0
 T340_L5_f15_le: x_Total_Bsmt_SF + 1516.480001 t340_leaf5 <= 2514.48
 T340_R6_f31_ge: x_Garage_Yr_Blt - 1941 t340_leaf6 >= 0
 T340_R6_f18_ge: x_2nd_Flr_SF - 994 t340_leaf6 >= 0
 T340_R6_f15_ge: x_Total_Bsmt_SF - 998 t340_leaf6 >= 0
 TREE_341_ONEHOT: t341_leaf0 + t341_leaf1 + t341_leaf2 + t341_leaf3
   + t341_leaf4 = 1
 T341_L0_f3_le: x_Overall_Qual + 999990.000001 t341_leaf0 <= 1e+06
 T341_L0_f20_le: x_Gr_Liv_Area + 2527.760001 t341_leaf0 <= 3421.76
 T341_L0_f6_le: x_Year_Remod/Add + 998048.000001 t341_leaf0 <= 1e+06
 T341_L1_f3_le: x_Overall_Qual + 999990.000001 t341_leaf1 <= 1e+06
 T341_L1_f20_le: x_Gr_Liv_Area + 2527.760001 t341_leaf1 <= 3421.76
 T341_R1_f6_ge: x_Year_Remod/Add - 1952 t341_leaf1 >= 0
 T341_L2_f3_le: x_Overall_Qual + 999990.000001 t341_leaf2 <= 1e+06
 T341_R2_f20_ge: x_Gr_Liv_Area - 894 t341_leaf2 >= 0
 T341_L2_f30_le: x_Fireplace_Qu + 3.000001 t341_leaf2 <= 4
 T341_L3_f3_le: x_Overall_Qual + 999990.000001 t341_leaf3 <= 1e+06
 T341_R3_f20_ge: x_Gr_Liv_Area - 894 t341_leaf3 >= 0
 T341_R3_f30_ge: x_Fireplace_Qu >= 1
 T341_R4_f3_ge: x_Overall_Qual - 10 t341_leaf4 >= 0
 TREE_342_ONEHOT: t342_leaf0 + t342_leaf1 + t342_leaf2 + t342_leaf3
   + t342_leaf4 + t342_leaf5 = 1
 T342_L0_f26_le: x_Kitchen_AbvGr + 1.000001 t342_leaf0 <= 3
 T342_L0_f71_le: x_Alley_No_aplica + 1e-06 t342_leaf0 <= 1
 T342_L0_f251_le: x_Garage_Type_Detchd + 1e-06 t342_leaf0 <= 1
 T342_L1_f26_le: x_Kitchen_AbvGr + 1.000001 t342_leaf1 <= 3
 T342_L1_f71_le: x_Alley_No_aplica + 1e-06 t342_leaf1 <= 1
 T342_R1_f251_ge: x_Garage_Type_Detchd - t342_leaf1 >= 0
 T342_L2_f26_le: x_Kitchen_AbvGr + 1.000001 t342_leaf2 <= 3
 T342_R2_f71_ge: x_Alley_No_aplica - t342_leaf2 >= 0
 T342_L2_f9_le: x_Exter_Cond + 2.000001 t342_leaf2 <= 4
 T342_L3_f26_le: x_Kitchen_AbvGr + 1.000001 t342_leaf3 <= 3
 T342_R3_f71_ge: x_Alley_No_aplica - t342_leaf3 >= 0
 T342_R3_f9_ge: x_Exter_Cond - 3 t342_leaf3 >= -1
 T342_R4_f26_ge: x_Kitchen_AbvGr - 2 t342_leaf4 >= 0
 T342_L4_f9_le: x_Exter_Cond + 2.000001 t342_leaf4 <= 4
 T342_R5_f26_ge: x_Kitchen_AbvGr - 2 t342_leaf5 >= 0
 T342_R5_f9_ge: x_Exter_Cond - 3 t342_leaf5 >= -1
 TREE_343_ONEHOT: t343_leaf0 + t343_leaf1 + t343_leaf2 + t343_leaf3
   + t343_leaf4 + t343_leaf5 = 1
 T343_L0_f203_le: x_Foundation_Slab + 1e-06 t343_leaf0 <= 1
 T343_L0_f5_le: x_Year_Built + 998092.000001 t343_leaf0 <= 1e+06
 T343_L0_f15_le: x_Total_Bsmt_SF + 1866.480001 t343_leaf0 <= 2514.48
 T343_L1_f203_le: x_Foundation_Slab + 1e-06 t343_leaf1 <= 1
 T343_L1_f5_le: x_Year_Built + 998092.000001 t343_leaf1 <= 1e+06
 T343_R1_f15_ge: x_Total_Bsmt_SF - 648 t343_leaf1 >= 0
 T343_L2_f203_le: x_Foundation_Slab + 1e-06 t343_leaf2 <= 1
 T343_R2_f5_ge: x_Year_Built - 1908 t343_leaf2 >= 0
 T343_L2_f1_le: x_Lot_Area <= 17226.999999
 T343_L3_f203_le: x_Foundation_Slab + 1e-06 t343_leaf3 <= 1
 T343_R3_f5_ge: x_Year_Built - 1908 t343_leaf3 >= 0
 T343_R3_f1_ge: x_Lot_Area - 10227 t343_leaf3 >= 7000
 T343_R4_f203_ge: x_Foundation_Slab - t343_leaf4 >= 0
 T343_L4_f0_le: x_Lot_Frontage + 109.530001 t343_leaf4 <= 175.53
 T343_R5_f203_ge: x_Foundation_Slab - t343_leaf5 >= 0
 T343_R5_f0_ge: x_Lot_Frontage - 66 t343_leaf5 >= 0
 TREE_344_ONEHOT: t344_leaf0 + t344_leaf1 + t344_leaf2 + t344_leaf3
   + t344_leaf4 + t344_leaf5 + t344_leaf6 + t344_leaf7 = 1
 T344_L0_f7_le: x_Mas_Vnr_Area + 915.070001 t344_leaf0 <= 931.07
 T344_L0_f44_le: x_Yr_Sold + 997990.000001 t344_leaf0 <= 1e+06
 T344_L0_f203_le: x_Foundation_Slab + 1e-06 t344_leaf0 <= 1
 T344_L1_f7_le: x_Mas_Vnr_Area + 915.070001 t344_leaf1 <= 931.07
 T344_L1_f44_le: x_Yr_Sold + 997990.000001 t344_leaf1 <= 1e+06
 T344_R1_f203_ge: x_Foundation_Slab - t344_leaf1 >= 0
 T344_L2_f7_le: x_Mas_Vnr_Area + 915.070001 t344_leaf2 <= 931.07
 T344_R2_f44_ge: x_Yr_Sold - 2010 t344_leaf2 >= 0
 T344_L2_f206_le: x_Bsmt_Exposure_Av + 1e-06 t344_leaf2 <= 1
 T344_L3_f7_le: x_Mas_Vnr_Area + 915.070001 t344_leaf3 <= 931.07
 T344_R3_f44_ge: x_Yr_Sold - 2010 t344_leaf3 >= 0
 T344_R3_f206_ge: x_Bsmt_Exposure_Av - t344_leaf3 >= 0
 T344_R4_f7_ge: x_Mas_Vnr_Area - 16 t344_leaf4 >= 0
 T344_L4_f96_le: x_Neighborhood_Edwards + 1e-06 t344_leaf4 <= 1
 T344_L4_f33_le: x_Garage_Area + 292.750001 t344_leaf4 <= 1163.75
 T344_R5_f7_ge: x_Mas_Vnr_Area - 16 t344_leaf5 >= 0
 T344_L5_f96_le: x_Neighborhood_Edwards + 1e-06 t344_leaf5 <= 1
 T344_R5_f33_ge: x_Garage_Area - 871 t344_leaf5 >= 0
 T344_R6_f7_ge: x_Mas_Vnr_Area - 16 t344_leaf6 >= 0
 T344_R6_f96_ge: x_Neighborhood_Edwards - t344_leaf6 >= 0
 T344_L6_f14_le: x_Bsmt_Unf_SF + 1438.680001 t344_leaf6 <= 1812.68
 T344_R7_f7_ge: x_Mas_Vnr_Area - 16 t344_leaf7 >= 0
 T344_R7_f96_ge: x_Neighborhood_Edwards - t344_leaf7 >= 0
 T344_R7_f14_ge: x_Bsmt_Unf_SF - 374 t344_leaf7 >= 0
 TREE_345_ONEHOT: t345_leaf0 + t345_leaf1 + t345_leaf2 + t345_leaf3
   + t345_leaf4 + t345_leaf5 + t345_leaf6 = 1
 T345_L0_f16_le: x_Heating_QC + 1.000001 t345_leaf0 <= 4
 T345_L0_f209_le: x_Bsmt_Exposure_No + 1e-06 t345_leaf0 <= 1
 T345_L1_f16_le: x_Heating_QC + 1.000001 t345_leaf1 <= 4
 T345_R1_f209_ge: x_Bsmt_Exposure_No - t345_leaf1 >= 0
 T345_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t345_leaf1 <= 1
 T345_L2_f16_le: x_Heating_QC + 1.000001 t345_leaf2 <= 4
 T345_R2_f209_ge: x_Bsmt_Exposure_No - t345_leaf2 >= 0
 T345_R2_f293_ge: x_Sale_Condition_Abnorml - t345_leaf2 >= 0
 T345_R3_f16_ge: x_Heating_QC - t345_leaf3 >= 2
 T345_L3_f3_le: x_Overall_Qual + 999993.000001 t345_leaf3 <= 1e+06
 T345_L3_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t345_leaf3 <= 1
 T345_R4_f16_ge: x_Heating_QC - t345_leaf4 >= 2
 T345_L4_f3_le: x_Overall_Qual + 999993.000001 t345_leaf4 <= 1e+06
 T345_R4_f214_ge: x_BsmtFin_Type_1_LwQ - t345_leaf4 >= 0
 T345_R5_f16_ge: x_Heating_QC - t345_leaf5 >= 2
 T345_R5_f3_ge: x_Overall_Qual - 7 t345_leaf5 >= 0
 T345_L5_f96_le: x_Neighborhood_Edwards + 1e-06 t345_leaf5 <= 1
 T345_R6_f16_ge: x_Heating_QC - t345_leaf6 >= 2
 T345_R6_f3_ge: x_Overall_Qual - 7 t345_leaf6 >= 0
 T345_R6_f96_ge: x_Neighborhood_Edwards - t345_leaf6 >= 0
 TREE_346_ONEHOT: t346_leaf0 + t346_leaf1 + t346_leaf2 + t346_leaf3 = 1
 T346_L0_f3_le: x_Overall_Qual + 999997.000001 t346_leaf0 <= 1e+06
 T346_R1_f3_ge: x_Overall_Qual - 3 t346_leaf1 >= 0
 T346_L1_f3_le: x_Overall_Qual + 999993.000001 t346_leaf1 <= 1e+06
 T346_R2_f3_ge: x_Overall_Qual - 3 t346_leaf2 >= 0
 T346_R2_f3_ge: x_Overall_Qual - 7 t346_leaf2 >= 0
 T346_L2_f86_le: x_Land_Slope_Gtl + 1e-06 t346_leaf2 <= 1
 T346_R3_f3_ge: x_Overall_Qual - 3 t346_leaf3 >= 0
 T346_R3_f3_ge: x_Overall_Qual - 7 t346_leaf3 >= 0
 T346_R3_f86_ge: x_Land_Slope_Gtl - t346_leaf3 >= 0
 TREE_347_ONEHOT: t347_leaf0 + t347_leaf1 + t347_leaf2 + t347_leaf3
   + t347_leaf4 = 1
 T347_L0_f292_le: x_Sale_Type_WD_ + 1e-06 t347_leaf0 <= 1
 T347_L0_f4_le: x_Overall_Cond + 999996.000001 t347_leaf0 <= 1e+06
 T347_L1_f292_le: x_Sale_Type_WD_ + 1e-06 t347_leaf1 <= 1
 T347_R1_f4_ge: x_Overall_Cond - 4 t347_leaf1 >= 0
 T347_R2_f292_ge: x_Sale_Type_WD_ - t347_leaf2 >= 0
 T347_L2_f38_le: x_Enclosed_Porch + 2.750000999999997 t347_leaf2 <= 292.75
 T347_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t347_leaf2 <= 1
 T347_R3_f292_ge: x_Sale_Type_WD_ - t347_leaf3 >= 0
 T347_L3_f38_le: x_Enclosed_Porch + 2.750000999999997 t347_leaf3 <= 292.75
 T347_R3_f297_ge: x_Sale_Condition_Normal - t347_leaf3 >= 0
 T347_R4_f292_ge: x_Sale_Type_WD_ - t347_leaf4 >= 0
 T347_R4_f38_ge: x_Enclosed_Porch - 290 t347_leaf4 >= 0
 TREE_348_ONEHOT: t348_leaf0 + t348_leaf1 + t348_leaf2 + t348_leaf3
   + t348_leaf4 = 1
 T348_L0_f7_le: x_Mas_Vnr_Area + 472.070001 t348_leaf0 <= 931.07
 T348_L0_f14_le: x_Bsmt_Unf_SF + 598.6800010000002 t348_leaf0 <= 1812.68
 T348_L0_f23_le: x_Full_Bath + 4.000001 t348_leaf0 <= 6
 T348_L1_f7_le: x_Mas_Vnr_Area + 472.070001 t348_leaf1 <= 931.07
 T348_L1_f14_le: x_Bsmt_Unf_SF + 598.6800010000002 t348_leaf1 <= 1812.68
 T348_R1_f23_ge: x_Full_Bath - 2 t348_leaf1 >= 0
 T348_L2_f7_le: x_Mas_Vnr_Area + 472.070001 t348_leaf2 <= 931.07
 T348_R2_f14_ge: x_Bsmt_Unf_SF - 1214 t348_leaf2 >= 0
 T348_L2_f36_le: x_Wood_Deck_SF + 440.900001 t348_leaf2 <= 525.9
 T348_L3_f7_le: x_Mas_Vnr_Area + 472.070001 t348_leaf3 <= 931.07
 T348_R3_f14_ge: x_Bsmt_Unf_SF - 1214 t348_leaf3 >= 0
 T348_R3_f36_ge: x_Wood_Deck_SF - 85 t348_leaf3 >= 0
 T348_R4_f7_ge: x_Mas_Vnr_Area - 459 t348_leaf4 >= 0
 TREE_349_ONEHOT: t349_leaf0 + t349_leaf1 + t349_leaf2 + t349_leaf3
   + t349_leaf4 + t349_leaf5 = 1
 T349_L0_f0_le: x_Lot_Frontage + 55.530001 t349_leaf0 <= 175.53
 T349_L0_f1_le: x_Lot_Area <= 19521.999999
 T349_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t349_leaf0 <= 1
 T349_L1_f0_le: x_Lot_Frontage + 55.530001 t349_leaf1 <= 175.53
 T349_L1_f1_le: x_Lot_Area <= 19521.999999
 T349_R1_f164_ge: x_Exterior_1st_BrkFace - t349_leaf1 >= 0
 T349_L2_f0_le: x_Lot_Frontage + 55.530001 t349_leaf2 <= 175.53
 T349_R2_f1_ge: x_Lot_Area - 12522 t349_leaf2 >= 7000
 T349_R3_f0_ge: x_Lot_Frontage - 120 t349_leaf3 >= 0
 T349_L3_f77_le: x_Land_Contour_Bnk + 1e-06 t349_leaf3 <= 1
 T349_L3_f0_le: x_Lot_Frontage + 38.530001 t349_leaf3 <= 175.53
 T349_R4_f0_ge: x_Lot_Frontage - 120 t349_leaf4 >= 0
 T349_L4_f77_le: x_Land_Contour_Bnk + 1e-06 t349_leaf4 <= 1
 T349_R4_f0_ge: x_Lot_Frontage - 137 t349_leaf4 >= 0
 T349_R5_f0_ge: x_Lot_Frontage - 120 t349_leaf5 >= 0
 T349_R5_f77_ge: x_Land_Contour_Bnk - t349_leaf5 >= 0
 TREE_350_ONEHOT: t350_leaf0 + t350_leaf1 + t350_leaf2 + t350_leaf3
   + t350_leaf4 + t350_leaf5 = 1
 T350_L0_f9_le: x_Exter_Cond + 2.000001 t350_leaf0 <= 4
 T350_L0_f20_le: x_Gr_Liv_Area + 2404.760001 t350_leaf0 <= 3421.76
 T350_L1_f9_le: x_Exter_Cond + 2.000001 t350_leaf1 <= 4
 T350_R1_f20_ge: x_Gr_Liv_Area - 1017 t350_leaf1 >= 0
 T350_R2_f9_ge: x_Exter_Cond - 3 t350_leaf2 >= -1
 T350_L2_f134_le: x_Bldg_Type_1Fam + 1e-06 t350_leaf2 <= 1
 T350_L2_f12_le: x_BsmtFin_SF_1 + 784.5000010000001 t350_leaf2 <= 1984.5
 T350_R3_f9_ge: x_Exter_Cond - 3 t350_leaf3 >= -1
 T350_L3_f134_le: x_Bldg_Type_1Fam + 1e-06 t350_leaf3 <= 1
 T350_R3_f12_ge: x_BsmtFin_SF_1 - 1200 t350_leaf3 >= 0
 T350_R4_f9_ge: x_Exter_Cond - 3 t350_leaf4 >= -1
 T350_R4_f134_ge: x_Bldg_Type_1Fam - t350_leaf4 >= 0
 T350_L4_f273_le: x_Mo_Sold_3 + 1e-06 t350_leaf4 <= 1
 T350_R5_f9_ge: x_Exter_Cond - 3 t350_leaf5 >= -1
 T350_R5_f134_ge: x_Bldg_Type_1Fam - t350_leaf5 >= 0
 T350_R5_f273_ge: x_Mo_Sold_3 - t350_leaf5 >= 0
 TREE_351_ONEHOT: t351_leaf0 + t351_leaf1 + t351_leaf2 + t351_leaf3
   + t351_leaf4 + t351_leaf5 + t351_leaf6 = 1
 T351_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t351_leaf0 <= 1
 T351_L0_f3_le: x_Overall_Qual + 999997.000001 t351_leaf0 <= 1e+06
 T351_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t351_leaf1 <= 1
 T351_R1_f3_ge: x_Overall_Qual - 3 t351_leaf1 >= 0
 T351_L1_f62_le: x_MS_Zoning_C_(all) + 1e-06 t351_leaf1 <= 1
 T351_L2_f293_le: x_Sale_Condition_Abnorml + 1e-06 t351_leaf2 <= 1
 T351_R2_f3_ge: x_Overall_Qual - 3 t351_leaf2 >= 0
 T351_R2_f62_ge: x_MS_Zoning_C_(all) - t351_leaf2 >= 0
 T351_R3_f293_ge: x_Sale_Condition_Abnorml - t351_leaf3 >= 0
 T351_L3_f33_le: x_Garage_Area + 722.750001 t351_leaf3 <= 1163.75
 T351_L3_f14_le: x_Bsmt_Unf_SF + 944.6800010000001 t351_leaf3 <= 1812.68
 T351_R4_f293_ge: x_Sale_Condition_Abnorml - t351_leaf4 >= 0
 T351_L4_f33_le: x_Garage_Area + 722.750001 t351_leaf4 <= 1163.75
 T351_R4_f14_ge: x_Bsmt_Unf_SF - 868 t351_leaf4 >= 0
 T351_R5_f293_ge: x_Sale_Condition_Abnorml - t351_leaf5 >= 0
 T351_R5_f33_ge: x_Garage_Area - 441 t351_leaf5 >= 0
 T351_L5_f12_le: x_BsmtFin_SF_1 + 999.500001 t351_leaf5 <= 1984.5
 T351_R6_f293_ge: x_Sale_Condition_Abnorml - t351_leaf6 >= 0
 T351_R6_f33_ge: x_Garage_Area - 441 t351_leaf6 >= 0
 T351_R6_f12_ge: x_BsmtFin_SF_1 - 985 t351_leaf6 >= 0
 TREE_352_ONEHOT: t352_leaf0 + t352_leaf1 + t352_leaf2 + t352_leaf3
   + t352_leaf4 + t352_leaf5 + t352_leaf6 = 1
 T352_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t352_leaf0 <= 1
 T352_L0_f12_le: x_BsmtFin_SF_1 + 1735.500001 t352_leaf0 <= 1984.5
 T352_L0_f33_le: x_Garage_Area + 706.750001 t352_leaf0 <= 1163.75
 T352_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t352_leaf1 <= 1
 T352_L1_f12_le: x_BsmtFin_SF_1 + 1735.500001 t352_leaf1 <= 1984.5
 T352_R1_f33_ge: x_Garage_Area - 457 t352_leaf1 >= 0
 T352_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t352_leaf2 <= 1
 T352_R2_f12_ge: x_BsmtFin_SF_1 - 249 t352_leaf2 >= 0
 T352_R3_f245_ge: x_FunctioNo_aplical_Typ - t352_leaf3 >= 0
 T352_L3_f14_le: x_Bsmt_Unf_SF + 1704.680001 t352_leaf3 <= 1812.68
 T352_L3_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t352_leaf3 <= 1
 T352_R4_f245_ge: x_FunctioNo_aplical_Typ - t352_leaf4 >= 0
 T352_L4_f14_le: x_Bsmt_Unf_SF + 1704.680001 t352_leaf4 <= 1812.68
 T352_R4_f207_ge: x_Bsmt_Exposure_Gd - t352_leaf4 >= 0
 T352_R5_f245_ge: x_FunctioNo_aplical_Typ - t352_leaf5 >= 0
 T352_R5_f14_ge: x_Bsmt_Unf_SF - 108 t352_leaf5 >= 0
 T352_L5_f175_le: x_Exterior_1st_Wd_Sdng + 1e-06 t352_leaf5 <= 1
 T352_R6_f245_ge: x_FunctioNo_aplical_Typ - t352_leaf6 >= 0
 T352_R6_f14_ge: x_Bsmt_Unf_SF - 108 t352_leaf6 >= 0
 T352_R6_f175_ge: x_Exterior_1st_Wd_Sdng - t352_leaf6 >= 0
 TREE_353_ONEHOT: t353_leaf0 + t353_leaf1 + t353_leaf2 + t353_leaf3
   + t353_leaf4 + t353_leaf5 + t353_leaf6 = 1
 T353_L0_f3_le: x_Overall_Qual + 999992.000001 t353_leaf0 <= 1e+06
 T353_L0_f11_le: x_Bsmt_Cond + 1.000001 t353_leaf0 <= 4
 T353_L0_f147_le: x_Roof_Style_Flat + 1e-06 t353_leaf0 <= 1
 T353_L1_f3_le: x_Overall_Qual + 999992.000001 t353_leaf1 <= 1e+06
 T353_L1_f11_le: x_Bsmt_Cond + 1.000001 t353_leaf1 <= 4
 T353_R1_f147_ge: x_Roof_Style_Flat - t353_leaf1 >= 0
 T353_L2_f3_le: x_Overall_Qual + 999992.000001 t353_leaf2 <= 1e+06
 T353_R2_f11_ge: x_Bsmt_Cond - 4 t353_leaf2 >= -1
 T353_L2_f96_le: x_Neighborhood_Edwards + 1e-06 t353_leaf2 <= 1
 T353_L3_f3_le: x_Overall_Qual + 999992.000001 t353_leaf3 <= 1e+06
 T353_R3_f11_ge: x_Bsmt_Cond - 4 t353_leaf3 >= -1
 T353_R3_f96_ge: x_Neighborhood_Edwards - t353_leaf3 >= 0
 T353_R4_f3_ge: x_Overall_Qual - 8 t353_leaf4 >= 0
 T353_L4_f77_le: x_Land_Contour_Bnk + 1e-06 t353_leaf4 <= 1
 T353_L4_f255_le: x_Garage_Finish_RFn + 1e-06 t353_leaf4 <= 1
 T353_R5_f3_ge: x_Overall_Qual - 8 t353_leaf5 >= 0
 T353_L5_f77_le: x_Land_Contour_Bnk + 1e-06 t353_leaf5 <= 1
 T353_R5_f255_ge: x_Garage_Finish_RFn - t353_leaf5 >= 0
 T353_R6_f3_ge: x_Overall_Qual - 8 t353_leaf6 >= 0
 T353_R6_f77_ge: x_Land_Contour_Bnk - t353_leaf6 >= 0
 TREE_354_ONEHOT: t354_leaf0 + t354_leaf1 + t354_leaf2 + t354_leaf3
   + t354_leaf4 + t354_leaf5 = 1
 T354_L0_f5_le: x_Year_Built + 998089.000001 t354_leaf0 <= 1e+06
 T354_L0_f6_le: x_Year_Remod/Add + 998000.000001 t354_leaf0 <= 1e+06
 T354_L1_f5_le: x_Year_Built + 998089.000001 t354_leaf1 <= 1e+06
 T354_R1_f6_ge: x_Year_Remod/Add - 2000 t354_leaf1 >= 0
 T354_R2_f5_ge: x_Year_Built - 1911 t354_leaf2 >= 0
 T354_L2_f33_le: x_Garage_Area + 848.750001 t354_leaf2 <= 1163.75
 T354_L2_f5_le: x_Year_Built + 998015.000001 t354_leaf2 <= 1e+06
 T354_R3_f5_ge: x_Year_Built - 1911 t354_leaf3 >= 0
 T354_L3_f33_le: x_Garage_Area + 848.750001 t354_leaf3 <= 1163.75
 T354_R3_f5_ge: x_Year_Built - 1985 t354_leaf3 >= 0
 T354_R4_f5_ge: x_Year_Built - 1911 t354_leaf4 >= 0
 T354_R4_f33_ge: x_Garage_Area - 315 t354_leaf4 >= 0
 T354_L4_f231_le: x_Central_Air_N + 1e-06 t354_leaf4 <= 1
 T354_R5_f5_ge: x_Year_Built - 1911 t354_leaf5 >= 0
 T354_R5_f33_ge: x_Garage_Area - 315 t354_leaf5 >= 0
 T354_R5_f231_ge: x_Central_Air_N - t354_leaf5 >= 0
 TREE_355_ONEHOT: t355_leaf0 + t355_leaf1 + t355_leaf2 + t355_leaf3
   + t355_leaf4 + t355_leaf5 = 1
 T355_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t355_leaf0 <= 1
 T355_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t355_leaf0 <= 1
 T355_L0_f232_le: x_Central_Air_Y + 1e-06 t355_leaf0 <= 1
 T355_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t355_leaf1 <= 1
 T355_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t355_leaf1 <= 1
 T355_R1_f232_ge: x_Central_Air_Y - t355_leaf1 >= 0
 T355_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t355_leaf2 <= 1
 T355_R2_f245_ge: x_FunctioNo_aplical_Typ - t355_leaf2 >= 0
 T355_L2_f295_le: x_Sale_Condition_Alloca + 1e-06 t355_leaf2 <= 1
 T355_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t355_leaf3 <= 1
 T355_R3_f245_ge: x_FunctioNo_aplical_Typ - t355_leaf3 >= 0
 T355_R3_f295_ge: x_Sale_Condition_Alloca - t355_leaf3 >= 0
 T355_R4_f95_ge: x_Neighborhood_Crawfor - t355_leaf4 >= 0
 T355_L4_f33_le: x_Garage_Area + 634.750001 t355_leaf4 <= 1163.75
 T355_R5_f95_ge: x_Neighborhood_Crawfor - t355_leaf5 >= 0
 T355_R5_f33_ge: x_Garage_Area - 529 t355_leaf5 >= 0
 TREE_356_ONEHOT: t356_leaf0 + t356_leaf1 + t356_leaf2 + t356_leaf3
   + t356_leaf4 = 1
 T356_L0_f40_le: x_Screen_Porch + 120.270001 t356_leaf0 <= 268.27
 T356_L0_f3_le: x_Overall_Qual + 999997.000001 t356_leaf0 <= 1e+06
 T356_L1_f40_le: x_Screen_Porch + 120.270001 t356_leaf1 <= 268.27
 T356_R1_f3_ge: x_Overall_Qual - 3 t356_leaf1 >= 0
 T356_L1_f1_le: x_Lot_Area <= 10932.999999
 T356_L2_f40_le: x_Screen_Porch + 120.270001 t356_leaf2 <= 268.27
 T356_R2_f3_ge: x_Overall_Qual - 3 t356_leaf2 >= 0
 T356_R2_f1_ge: x_Lot_Area - 3933 t356_leaf2 >= 7000
 T356_R3_f40_ge: x_Screen_Porch - 148 t356_leaf3 >= 0
 T356_L3_f16_le: x_Heating_QC + 1.000001 t356_leaf3 <= 4
 T356_R4_f40_ge: x_Screen_Porch - 148 t356_leaf4 >= 0
 T356_R4_f16_ge: x_Heating_QC - t356_leaf4 >= 2
 TREE_357_ONEHOT: t357_leaf0 + t357_leaf1 + t357_leaf2 + t357_leaf3 = 1
 T357_L0_f273_le: x_Mo_Sold_3 + 1e-06 t357_leaf0 <= 1
 T357_L0_f38_le: x_Enclosed_Porch + 12.750001 t357_leaf0 <= 292.75
 T357_L0_f63_le: x_MS_Zoning_FV + 1e-06 t357_leaf0 <= 1
 T357_L1_f273_le: x_Mo_Sold_3 + 1e-06 t357_leaf1 <= 1
 T357_L1_f38_le: x_Enclosed_Porch + 12.750001 t357_leaf1 <= 292.75
 T357_R1_f63_ge: x_MS_Zoning_FV - t357_leaf1 >= 0
 T357_L2_f273_le: x_Mo_Sold_3 + 1e-06 t357_leaf2 <= 1
 T357_R2_f38_ge: x_Enclosed_Porch - 280 t357_leaf2 >= 0
 T357_R3_f273_ge: x_Mo_Sold_3 - t357_leaf3 >= 0
 TREE_358_ONEHOT: t358_leaf0 + t358_leaf1 + t358_leaf2 + t358_leaf3 = 1
 T358_L0_f3_le: x_Overall_Qual + 999997.000001 t358_leaf0 <= 1e+06
 T358_R1_f3_ge: x_Overall_Qual - 3 t358_leaf1 >= 0
 T358_L1_f44_le: x_Yr_Sold + 997993.000001 t358_leaf1 <= 1e+06
 T358_R2_f3_ge: x_Overall_Qual - 3 t358_leaf2 >= 0
 T358_R2_f44_ge: x_Yr_Sold - 2007 t358_leaf2 >= 0
 T358_L2_f18_le: x_2nd_Flr_SF + 581.5400010000001 t358_leaf2 <= 1641.54
 T358_R3_f3_ge: x_Overall_Qual - 3 t358_leaf3 >= 0
 T358_R3_f44_ge: x_Yr_Sold - 2007 t358_leaf3 >= 0
 T358_R3_f18_ge: x_2nd_Flr_SF - 1060 t358_leaf3 >= 0
 TREE_359_ONEHOT: t359_leaf0 + t359_leaf1 + t359_leaf2 = 1
 T359_L0_f43_le: x_Misc_Val + 29.47000099999968 t359_leaf0 <= 3029.47
 T359_L0_f1_le: x_Lot_Area <= 27696.999999
 T359_L1_f43_le: x_Misc_Val + 29.47000099999968 t359_leaf1 <= 3029.47
 T359_R1_f1_ge: x_Lot_Area - 20697 t359_leaf1 >= 7000
 T359_R2_f43_ge: x_Misc_Val - 3000 t359_leaf2 >= 0
 TREE_360_ONEHOT: t360_leaf0 + t360_leaf1 + t360_leaf2 + t360_leaf3
   + t360_leaf4 + t360_leaf5 + t360_leaf6 = 1
 T360_L0_f20_le: x_Gr_Liv_Area + 2653.760001 t360_leaf0 <= 3421.76
 T360_L0_f20_le: x_Gr_Liv_Area + 2701.760001 t360_leaf0 <= 3421.76
 T360_L0_f175_le: x_Exterior_1st_Wd_Sdng + 1e-06 t360_leaf0 <= 1
 T360_L1_f20_le: x_Gr_Liv_Area + 2653.760001 t360_leaf1 <= 3421.76
 T360_L1_f20_le: x_Gr_Liv_Area + 2701.760001 t360_leaf1 <= 3421.76
 T360_R1_f175_ge: x_Exterior_1st_Wd_Sdng - t360_leaf1 >= 0
 T360_L2_f20_le: x_Gr_Liv_Area + 2653.760001 t360_leaf2 <= 3421.76
 T360_R2_f20_ge: x_Gr_Liv_Area - 720 t360_leaf2 >= 0
 T360_R3_f20_ge: x_Gr_Liv_Area - 768 t360_leaf3 >= 0
 T360_L3_f44_le: x_Yr_Sold + 997990.000001 t360_leaf3 <= 1e+06
 T360_L3_f15_le: x_Total_Bsmt_SF + 1387.480001 t360_leaf3 <= 2514.48
 T360_R4_f20_ge: x_Gr_Liv_Area - 768 t360_leaf4 >= 0
 T360_L4_f44_le: x_Yr_Sold + 997990.000001 t360_leaf4 <= 1e+06
 T360_R4_f15_ge: x_Total_Bsmt_SF - 1127 t360_leaf4 >= 0
 T360_R5_f20_ge: x_Gr_Liv_Area - 768 t360_leaf5 >= 0
 T360_R5_f44_ge: x_Yr_Sold - 2010 t360_leaf5 >= 0
 T360_L5_f293_le: x_Sale_Condition_Abnorml + 1e-06 t360_leaf5 <= 1
 T360_R6_f20_ge: x_Gr_Liv_Area - 768 t360_leaf6 >= 0
 T360_R6_f44_ge: x_Yr_Sold - 2010 t360_leaf6 >= 0
 T360_R6_f293_ge: x_Sale_Condition_Abnorml - t360_leaf6 >= 0
 TREE_361_ONEHOT: t361_leaf0 + t361_leaf1 + t361_leaf2 + t361_leaf3
   + t361_leaf4 = 1
 T361_L0_f92_le: x_Neighborhood_BrkSide + 1e-06 t361_leaf0 <= 1
 T361_L0_f44_le: x_Yr_Sold + 997993.000001 t361_leaf0 <= 1e+06
 T361_L0_f272_le: x_Mo_Sold_2 + 1e-06 t361_leaf0 <= 1
 T361_L1_f92_le: x_Neighborhood_BrkSide + 1e-06 t361_leaf1 <= 1
 T361_L1_f44_le: x_Yr_Sold + 997993.000001 t361_leaf1 <= 1e+06
 T361_R1_f272_ge: x_Mo_Sold_2 - t361_leaf1 >= 0
 T361_L2_f92_le: x_Neighborhood_BrkSide + 1e-06 t361_leaf2 <= 1
 T361_R2_f44_ge: x_Yr_Sold - 2007 t361_leaf2 >= 0
 T361_L2_f259_le: x_Paved_Drive_Y + 1e-06 t361_leaf2 <= 1
 T361_L3_f92_le: x_Neighborhood_BrkSide + 1e-06 t361_leaf3 <= 1
 T361_R3_f44_ge: x_Yr_Sold - 2007 t361_leaf3 >= 0
 T361_R3_f259_ge: x_Paved_Drive_Y - t361_leaf3 >= 0
 T361_R4_f92_ge: x_Neighborhood_BrkSide - t361_leaf4 >= 0
 TREE_362_ONEHOT: t362_leaf0 + t362_leaf1 + t362_leaf2 + t362_leaf3
   + t362_leaf4 = 1
 T362_L0_f33_le: x_Garage_Area + 9.750001000000111 t362_leaf0 <= 1163.75
 T362_L0_f261_le: x_Fence_GdWo + 1e-06 t362_leaf0 <= 1
 T362_L0_f33_le: x_Garage_Area + 721.750001 t362_leaf0 <= 1163.75
 T362_L1_f33_le: x_Garage_Area + 9.750001000000111 t362_leaf1 <= 1163.75
 T362_L1_f261_le: x_Fence_GdWo + 1e-06 t362_leaf1 <= 1
 T362_R1_f33_ge: x_Garage_Area - 442 t362_leaf1 >= 0
 T362_L2_f33_le: x_Garage_Area + 9.750001000000111 t362_leaf2 <= 1163.75
 T362_R2_f261_ge: x_Fence_GdWo - t362_leaf2 >= 0
 T362_L2_f38_le: x_Enclosed_Porch + 179.750001 t362_leaf2 <= 292.75
 T362_L3_f33_le: x_Garage_Area + 9.750001000000111 t362_leaf3 <= 1163.75
 T362_R3_f261_ge: x_Fence_GdWo - t362_leaf3 >= 0
 T362_R3_f38_ge: x_Enclosed_Porch - 113 t362_leaf3 >= 0
 T362_R4_f33_ge: x_Garage_Area - 1154 t362_leaf4 >= 0
 TREE_363_ONEHOT: t363_leaf0 + t363_leaf1 + t363_leaf2 + t363_leaf3
   + t363_leaf4 + t363_leaf5 = 1
 T363_L0_f14_le: x_Bsmt_Unf_SF + 419.6800010000002 t363_leaf0 <= 1812.68
 T363_L0_f41_le: x_Pool_Area + 76.77000099999998 t363_leaf0 <= 444.77
 T363_L0_f148_le: x_Roof_Style_Gable + 1e-06 t363_leaf0 <= 1
 T363_L1_f14_le: x_Bsmt_Unf_SF + 419.6800010000002 t363_leaf1 <= 1812.68
 T363_L1_f41_le: x_Pool_Area + 76.77000099999998 t363_leaf1 <= 444.77
 T363_R1_f148_ge: x_Roof_Style_Gable - t363_leaf1 >= 0
 T363_L2_f14_le: x_Bsmt_Unf_SF + 419.6800010000002 t363_leaf2 <= 1812.68
 T363_R2_f41_ge: x_Pool_Area - 368 t363_leaf2 >= 0
 T363_R3_f14_ge: x_Bsmt_Unf_SF - 1393 t363_leaf3 >= 0
 T363_L3_f8_le: x_Exter_Qual + 1e-06 t363_leaf3 <= 4
 T363_L3_f199_le: x_Mas_Vnr_Type_Stone + 1e-06 t363_leaf3 <= 1
 T363_R4_f14_ge: x_Bsmt_Unf_SF - 1393 t363_leaf4 >= 0
 T363_L4_f8_le: x_Exter_Qual + 1e-06 t363_leaf4 <= 4
 T363_R4_f199_ge: x_Mas_Vnr_Type_Stone - t363_leaf4 >= 0
 T363_R5_f14_ge: x_Bsmt_Unf_SF - 1393 t363_leaf5 >= 0
 T363_R5_f8_ge: x_Exter_Qual - 2 t363_leaf5 >= 2
 TREE_364_ONEHOT: t364_leaf0 + t364_leaf1 + t364_leaf2 + t364_leaf3
   + t364_leaf4 + t364_leaf5 = 1
 T364_L0_f231_le: x_Central_Air_N + 1e-06 t364_leaf0 <= 1
 T364_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t364_leaf0 <= 1
 T364_L1_f231_le: x_Central_Air_N + 1e-06 t364_leaf1 <= 1
 T364_R1_f245_ge: x_FunctioNo_aplical_Typ - t364_leaf1 >= 0
 T364_L1_f7_le: x_Mas_Vnr_Area + 475.070001 t364_leaf1 <= 931.07
 T364_L2_f231_le: x_Central_Air_N + 1e-06 t364_leaf2 <= 1
 T364_R2_f245_ge: x_FunctioNo_aplical_Typ - t364_leaf2 >= 0
 T364_R2_f7_ge: x_Mas_Vnr_Area - 456 t364_leaf2 >= 0
 T364_R3_f231_ge: x_Central_Air_N - t364_leaf3 >= 0
 T364_L3_f12_le: x_BsmtFin_SF_1 + 1633.500001 t364_leaf3 <= 1984.5
 T364_L3_f148_le: x_Roof_Style_Gable + 1e-06 t364_leaf3 <= 1
 T364_R4_f231_ge: x_Central_Air_N - t364_leaf4 >= 0
 T364_L4_f12_le: x_BsmtFin_SF_1 + 1633.500001 t364_leaf4 <= 1984.5
 T364_R4_f148_ge: x_Roof_Style_Gable - t364_leaf4 >= 0
 T364_R5_f231_ge: x_Central_Air_N - t364_leaf5 >= 0
 T364_R5_f12_ge: x_BsmtFin_SF_1 - 351 t364_leaf5 >= 0
 TREE_365_ONEHOT: t365_leaf0 + t365_leaf1 + t365_leaf2 + t365_leaf3
   + t365_leaf4 + t365_leaf5 = 1
 T365_L0_f119_le: x_Condition_1_Norm + 1e-06 t365_leaf0 <= 1
 T365_L0_f5_le: x_Year_Built + 998051.000001 t365_leaf0 <= 1e+06
 T365_L1_f119_le: x_Condition_1_Norm + 1e-06 t365_leaf1 <= 1
 T365_R1_f5_ge: x_Year_Built - 1949 t365_leaf1 >= 0
 T365_R2_f119_ge: x_Condition_1_Norm - t365_leaf2 >= 0
 T365_L2_f26_le: x_Kitchen_AbvGr + 1.000001 t365_leaf2 <= 3
 T365_L2_f231_le: x_Central_Air_N + 1e-06 t365_leaf2 <= 1
 T365_R3_f119_ge: x_Condition_1_Norm - t365_leaf3 >= 0
 T365_L3_f26_le: x_Kitchen_AbvGr + 1.000001 t365_leaf3 <= 3
 T365_R3_f231_ge: x_Central_Air_N - t365_leaf3 >= 0
 T365_R4_f119_ge: x_Condition_1_Norm - t365_leaf4 >= 0
 T365_R4_f26_ge: x_Kitchen_AbvGr - 2 t365_leaf4 >= 0
 T365_L4_f5_le: x_Year_Built + 998032.000001 t365_leaf4 <= 1e+06
 T365_R5_f119_ge: x_Condition_1_Norm - t365_leaf5 >= 0
 T365_R5_f26_ge: x_Kitchen_AbvGr - 2 t365_leaf5 >= 0
 T365_R5_f5_ge: x_Year_Built - 1968 t365_leaf5 >= 0
 TREE_366_ONEHOT: t366_leaf0 + t366_leaf1 + t366_leaf2 + t366_leaf3
   + t366_leaf4 + t366_leaf5 = 1
 T366_L0_f3_le: x_Overall_Qual + 999993.000001 t366_leaf0 <= 1e+06
 T366_L0_f8_le: x_Exter_Qual + 2.000001 t366_leaf0 <= 4
 T366_L1_f3_le: x_Overall_Qual + 999993.000001 t366_leaf1 <= 1e+06
 T366_R1_f8_ge: x_Exter_Qual >= 2
 T366_L1_f4_le: x_Overall_Cond + 999992.000001 t366_leaf1 <= 1e+06
 T366_L2_f3_le: x_Overall_Qual + 999993.000001 t366_leaf2 <= 1e+06
 T366_R2_f8_ge: x_Exter_Qual >= 2
 T366_R2_f4_ge: x_Overall_Cond - 8 t366_leaf2 >= 0
 T366_R3_f3_ge: x_Overall_Qual - 7 t366_leaf3 >= 0
 T366_L3_f105_le: x_Neighborhood_NWAmes + 1e-06 t366_leaf3 <= 1
 T366_L3_f164_le: x_Exterior_1st_BrkFace + 1e-06 t366_leaf3 <= 1
 T366_R4_f3_ge: x_Overall_Qual - 7 t366_leaf4 >= 0
 T366_L4_f105_le: x_Neighborhood_NWAmes + 1e-06 t366_leaf4 <= 1
 T366_R4_f164_ge: x_Exterior_1st_BrkFace - t366_leaf4 >= 0
 T366_R5_f3_ge: x_Overall_Qual - 7 t366_leaf5 >= 0
 T366_R5_f105_ge: x_Neighborhood_NWAmes - t366_leaf5 >= 0
 TREE_367_ONEHOT: t367_leaf0 + t367_leaf1 + t367_leaf2 + t367_leaf3
   + t367_leaf4 = 1
 T367_L0_f18_le: x_2nd_Flr_SF + 786.540001 t367_leaf0 <= 1641.54
 T367_L0_f267_le: x_Misc_Feature_No_aplica + 1e-06 t367_leaf0 <= 1
 T367_L0_f183_le: x_Exterior_2nd_HdBoard + 1e-06 t367_leaf0 <= 1
 T367_L1_f18_le: x_2nd_Flr_SF + 786.540001 t367_leaf1 <= 1641.54
 T367_L1_f267_le: x_Misc_Feature_No_aplica + 1e-06 t367_leaf1 <= 1
 T367_R1_f183_ge: x_Exterior_2nd_HdBoard - t367_leaf1 >= 0
 T367_L2_f18_le: x_2nd_Flr_SF + 786.540001 t367_leaf2 <= 1641.54
 T367_R2_f267_ge: x_Misc_Feature_No_aplica - t367_leaf2 >= 0
 T367_L2_f296_le: x_Sale_Condition_Family + 1e-06 t367_leaf2 <= 1
 T367_L3_f18_le: x_2nd_Flr_SF + 786.540001 t367_leaf3 <= 1641.54
 T367_R3_f267_ge: x_Misc_Feature_No_aplica - t367_leaf3 >= 0
 T367_R3_f296_ge: x_Sale_Condition_Family - t367_leaf3 >= 0
 T367_R4_f18_ge: x_2nd_Flr_SF - 855 t367_leaf4 >= 0
 TREE_368_ONEHOT: t368_leaf0 + t368_leaf1 + t368_leaf2 + t368_leaf3
   + t368_leaf4 = 1
 T368_L0_f14_le: x_Bsmt_Unf_SF + 555.6800010000002 t368_leaf0 <= 1812.68
 T368_L0_f219_le: x_BsmtFin_Type_2_BLQ + 1e-06 t368_leaf0 <= 1
 T368_L0_f13_le: x_BsmtFin_SF_2 + 7.050000999999952 t368_leaf0 <= 757.05
 T368_L1_f14_le: x_Bsmt_Unf_SF + 555.6800010000002 t368_leaf1 <= 1812.68
 T368_L1_f219_le: x_BsmtFin_Type_2_BLQ + 1e-06 t368_leaf1 <= 1
 T368_R1_f13_ge: x_BsmtFin_SF_2 - 750 t368_leaf1 >= 0
 T368_L2_f14_le: x_Bsmt_Unf_SF + 555.6800010000002 t368_leaf2 <= 1812.68
 T368_R2_f219_ge: x_BsmtFin_Type_2_BLQ - t368_leaf2 >= 0
 T368_R3_f14_ge: x_Bsmt_Unf_SF - 1257 t368_leaf3 >= 0
 T368_L3_f0_le: x_Lot_Frontage + 98.530001 t368_leaf3 <= 175.53
 T368_R4_f14_ge: x_Bsmt_Unf_SF - 1257 t368_leaf4 >= 0
 T368_R4_f0_ge: x_Lot_Frontage - 77 t368_leaf4 >= 0
 TREE_369_ONEHOT: t369_leaf0 + t369_leaf1 + t369_leaf2 + t369_leaf3
   + t369_leaf4 = 1
 T369_L0_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t369_leaf0 <= 1
 T369_L0_f3_le: x_Overall_Qual + 999995.000001 t369_leaf0 <= 1e+06
 T369_L0_f20_le: x_Gr_Liv_Area + 2325.760001000001 t369_leaf0 <= 3421.76
 T369_L1_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t369_leaf1 <= 1
 T369_L1_f3_le: x_Overall_Qual + 999995.000001 t369_leaf1 <= 1e+06
 T369_R1_f20_ge: x_Gr_Liv_Area - 1096 t369_leaf1 >= 0
 T369_L2_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t369_leaf2 <= 1
 T369_R2_f3_ge: x_Overall_Qual - 5 t369_leaf2 >= 0
 T369_L2_f5_le: x_Year_Built + 998086.000001 t369_leaf2 <= 1e+06
 T369_L3_f239_le: x_FunctioNo_aplical_Maj2 + 1e-06 t369_leaf3 <= 1
 T369_R3_f3_ge: x_Overall_Qual - 5 t369_leaf3 >= 0
 T369_R3_f5_ge: x_Year_Built - 1914 t369_leaf3 >= 0
 T369_R4_f239_ge: x_FunctioNo_aplical_Maj2 - t369_leaf4 >= 0
 TREE_370_ONEHOT: t370_leaf0 + t370_leaf1 + t370_leaf2 + t370_leaf3
   + t370_leaf4 = 1
 T370_L0_f62_le: x_MS_Zoning_C_(all) + 1e-06 t370_leaf0 <= 1
 T370_L0_f118_le: x_Condition_1_Feedr + 1e-06 t370_leaf0 <= 1
 T370_L0_f1_le: x_Lot_Area + 2500.000001 t370_leaf0 <= 7000
 T370_L1_f62_le: x_MS_Zoning_C_(all) + 1e-06 t370_leaf1 <= 1
 T370_L1_f118_le: x_Condition_1_Feedr + 1e-06 t370_leaf1 <= 1
 T370_R1_f1_ge: x_Lot_Area >= 4500
 T370_L2_f62_le: x_MS_Zoning_C_(all) + 1e-06 t370_leaf2 <= 1
 T370_R2_f118_ge: x_Condition_1_Feedr - t370_leaf2 >= 0
 T370_L2_f44_le: x_Yr_Sold + 997992.000001 t370_leaf2 <= 1e+06
 T370_L3_f62_le: x_MS_Zoning_C_(all) + 1e-06 t370_leaf3 <= 1
 T370_R3_f118_ge: x_Condition_1_Feedr - t370_leaf3 >= 0
 T370_R3_f44_ge: x_Yr_Sold - 2008 t370_leaf3 >= 0
 T370_R4_f62_ge: x_MS_Zoning_C_(all) - t370_leaf4 >= 0
 TREE_371_ONEHOT: t371_leaf0 + t371_leaf1 + t371_leaf2 + t371_leaf3
   + t371_leaf4 = 1
 T371_L0_f93_le: x_Neighborhood_ClearCr + 1e-06 t371_leaf0 <= 1
 T371_L0_f226_le: x_Heating_GasA + 1e-06 t371_leaf0 <= 1
 T371_L0_f5_le: x_Year_Built + 998089.000001 t371_leaf0 <= 1e+06
 T371_L1_f93_le: x_Neighborhood_ClearCr + 1e-06 t371_leaf1 <= 1
 T371_L1_f226_le: x_Heating_GasA + 1e-06 t371_leaf1 <= 1
 T371_R1_f5_ge: x_Year_Built - 1911 t371_leaf1 >= 0
 T371_L2_f93_le: x_Neighborhood_ClearCr + 1e-06 t371_leaf2 <= 1
 T371_R2_f226_ge: x_Heating_GasA - t371_leaf2 >= 0
 T371_L2_f1_le: x_Lot_Area <= 23256.999999
 T371_L3_f93_le: x_Neighborhood_ClearCr + 1e-06 t371_leaf3 <= 1
 T371_R3_f226_ge: x_Heating_GasA - t371_leaf3 >= 0
 T371_R3_f1_ge: x_Lot_Area - 16257 t371_leaf3 >= 7000
 T371_R4_f93_ge: x_Neighborhood_ClearCr - t371_leaf4 >= 0
 TREE_372_ONEHOT: t372_leaf0 + t372_leaf1 + t372_leaf2 + t372_leaf3 = 1
 T372_L0_f8_le: x_Exter_Qual + 2.000001 t372_leaf0 <= 4
 T372_R1_f8_ge: x_Exter_Qual >= 2
 T372_L1_f36_le: x_Wood_Deck_SF + 299.900001 t372_leaf1 <= 525.9
 T372_L1_f21_le: x_Bsmt_Full_Bath + 0.020001 t372_leaf1 <= 2.02
 T372_R2_f8_ge: x_Exter_Qual >= 2
 T372_L2_f36_le: x_Wood_Deck_SF + 299.900001 t372_leaf2 <= 525.9
 T372_R2_f21_ge: x_Bsmt_Full_Bath - 2 t372_leaf2 >= 0
 T372_R3_f8_ge: x_Exter_Qual >= 2
 T372_R3_f36_ge: x_Wood_Deck_SF - 226 t372_leaf3 >= 0
 TREE_373_ONEHOT: t373_leaf0 + t373_leaf1 + t373_leaf2 + t373_leaf3
   + t373_leaf4 = 1
 T373_L0_f203_le: x_Foundation_Slab + 1e-06 t373_leaf0 <= 1
 T373_L0_f40_le: x_Screen_Porch + 160.270001 t373_leaf0 <= 268.27
 T373_L0_f86_le: x_Land_Slope_Gtl + 1e-06 t373_leaf0 <= 1
 T373_L1_f203_le: x_Foundation_Slab + 1e-06 t373_leaf1 <= 1
 T373_L1_f40_le: x_Screen_Porch + 160.270001 t373_leaf1 <= 268.27
 T373_R1_f86_ge: x_Land_Slope_Gtl - t373_leaf1 >= 0
 T373_L2_f203_le: x_Foundation_Slab + 1e-06 t373_leaf2 <= 1
 T373_R2_f40_ge: x_Screen_Porch - 108 t373_leaf2 >= 0
 T373_R3_f203_ge: x_Foundation_Slab - t373_leaf3 >= 0
 T373_L3_f0_le: x_Lot_Frontage + 109.530001 t373_leaf3 <= 175.53
 T373_R4_f203_ge: x_Foundation_Slab - t373_leaf4 >= 0
 T373_R4_f0_ge: x_Lot_Frontage - 66 t373_leaf4 >= 0
 TREE_374_ONEHOT: t374_leaf0 + t374_leaf1 + t374_leaf2 + t374_leaf3
   + t374_leaf4 = 1
 T374_L0_f20_le: x_Gr_Liv_Area + 2653.760001 t374_leaf0 <= 3421.76
 T374_L0_f20_le: x_Gr_Liv_Area + 2701.760001 t374_leaf0 <= 3421.76
 T374_L1_f20_le: x_Gr_Liv_Area + 2653.760001 t374_leaf1 <= 3421.76
 T374_R1_f20_ge: x_Gr_Liv_Area - 720 t374_leaf1 >= 0
 T374_R2_f20_ge: x_Gr_Liv_Area - 768 t374_leaf2 >= 0
 T374_L2_f63_le: x_MS_Zoning_FV + 1e-06 t374_leaf2 <= 1
 T374_L2_f44_le: x_Yr_Sold + 997993.000001 t374_leaf2 <= 1e+06
 T374_R3_f20_ge: x_Gr_Liv_Area - 768 t374_leaf3 >= 0
 T374_L3_f63_le: x_MS_Zoning_FV + 1e-06 t374_leaf3 <= 1
 T374_R3_f44_ge: x_Yr_Sold - 2007 t374_leaf3 >= 0
 T374_R4_f20_ge: x_Gr_Liv_Area - 768 t374_leaf4 >= 0
 T374_R4_f63_ge: x_MS_Zoning_FV - t374_leaf4 >= 0
 TREE_375_ONEHOT: t375_leaf0 + t375_leaf1 + t375_leaf2 + t375_leaf3
   + t375_leaf4 + t375_leaf5 = 1
 T375_L0_f15_le: x_Total_Bsmt_SF + 588.4800010000001 t375_leaf0 <= 2514.48
 T375_L0_f3_le: x_Overall_Qual + 999992.000001 t375_leaf0 <= 1e+06
 T375_L0_f6_le: x_Year_Remod/Add + 997992.000001 t375_leaf0 <= 1e+06
 T375_L1_f15_le: x_Total_Bsmt_SF + 588.4800010000001 t375_leaf1 <= 2514.48
 T375_L1_f3_le: x_Overall_Qual + 999992.000001 t375_leaf1 <= 1e+06
 T375_R1_f6_ge: x_Year_Remod/Add - 2008 t375_leaf1 >= 0
 T375_L2_f15_le: x_Total_Bsmt_SF + 588.4800010000001 t375_leaf2 <= 2514.48
 T375_R2_f3_ge: x_Overall_Qual - 8 t375_leaf2 >= 0
 T375_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t375_leaf2 <= 1
 T375_L3_f15_le: x_Total_Bsmt_SF + 588.4800010000001 t375_leaf3 <= 2514.48
 T375_R3_f3_ge: x_Overall_Qual - 8 t375_leaf3 >= 0
 T375_R3_f95_ge: x_Neighborhood_Crawfor - t375_leaf3 >= 0
 T375_R4_f15_ge: x_Total_Bsmt_SF - 1926 t375_leaf4 >= 0
 T375_L4_f36_le: x_Wood_Deck_SF + 125.900001 t375_leaf4 <= 525.9
 T375_R5_f15_ge: x_Total_Bsmt_SF - 1926 t375_leaf5 >= 0
 T375_R5_f36_ge: x_Wood_Deck_SF - 400 t375_leaf5 >= 0
 TREE_376_ONEHOT: t376_leaf0 + t376_leaf1 + t376_leaf2 + t376_leaf3
   + t376_leaf4 + t376_leaf5 = 1
 T376_L0_f96_le: x_Neighborhood_Edwards + 1e-06 t376_leaf0 <= 1
 T376_L0_f85_le: x_Lot_Config_Inside + 1e-06 t376_leaf0 <= 1
 T376_L1_f96_le: x_Neighborhood_Edwards + 1e-06 t376_leaf1 <= 1
 T376_R1_f85_ge: x_Lot_Config_Inside - t376_leaf1 >= 0
 T376_L1_f1_le: x_Lot_Area + 1638.000001 t376_leaf1 <= 7000
 T376_L2_f96_le: x_Neighborhood_Edwards + 1e-06 t376_leaf2 <= 1
 T376_R2_f85_ge: x_Lot_Config_Inside - t376_leaf2 >= 0
 T376_R2_f1_ge: x_Lot_Area >= 5362
 T376_R3_f96_ge: x_Neighborhood_Edwards - t376_leaf3 >= 0
 T376_L3_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t376_leaf3 <= 1
 T376_L3_f81_le: x_Lot_Config_Corner + 1e-06 t376_leaf3 <= 1
 T376_R4_f96_ge: x_Neighborhood_Edwards - t376_leaf4 >= 0
 T376_L4_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t376_leaf4 <= 1
 T376_R4_f81_ge: x_Lot_Config_Corner - t376_leaf4 >= 0
 T376_R5_f96_ge: x_Neighborhood_Edwards - t376_leaf5 >= 0
 T376_R5_f207_ge: x_Bsmt_Exposure_Gd - t376_leaf5 >= 0
 TREE_377_ONEHOT: t377_leaf0 + t377_leaf1 + t377_leaf2 + t377_leaf3
   + t377_leaf4 + t377_leaf5 = 1
 T377_L0_f6_le: x_Year_Remod/Add + 998048.000001 t377_leaf0 <= 1e+06
 T377_L0_f17_le: x_1st_Flr_SF + 1674.900001 t377_leaf0 <= 2534.9
 T377_L1_f6_le: x_Year_Remod/Add + 998048.000001 t377_leaf1 <= 1e+06
 T377_R1_f17_ge: x_1st_Flr_SF - 860 t377_leaf1 >= 0
 T377_R2_f6_ge: x_Year_Remod/Add - 1952 t377_leaf2 >= 0
 T377_L2_f3_le: x_Overall_Qual + 999996.000001 t377_leaf2 <= 1e+06
 T377_L2_f259_le: x_Paved_Drive_Y + 1e-06 t377_leaf2 <= 1
 T377_R3_f6_ge: x_Year_Remod/Add - 1952 t377_leaf3 >= 0
 T377_L3_f3_le: x_Overall_Qual + 999996.000001 t377_leaf3 <= 1e+06
 T377_R3_f259_ge: x_Paved_Drive_Y - t377_leaf3 >= 0
 T377_R4_f6_ge: x_Year_Remod/Add - 1952 t377_leaf4 >= 0
 T377_R4_f3_ge: x_Overall_Qual - 4 t377_leaf4 >= 0
 T377_L4_f0_le: x_Lot_Frontage + 57.530001 t377_leaf4 <= 175.53
 T377_R5_f6_ge: x_Year_Remod/Add - 1952 t377_leaf5 >= 0
 T377_R5_f3_ge: x_Overall_Qual - 4 t377_leaf5 >= 0
 T377_R5_f0_ge: x_Lot_Frontage - 118 t377_leaf5 >= 0
 TREE_378_ONEHOT: t378_leaf0 + t378_leaf1 + t378_leaf2 + t378_leaf3
   + t378_leaf4 = 1
 T378_L0_f112_le: x_Neighborhood_SawyerW + 1e-06 t378_leaf0 <= 1
 T378_L0_f3_le: x_Overall_Qual + 999997.000001 t378_leaf0 <= 1e+06
 T378_L0_f31_le: x_Garage_Yr_Blt + 998077.000001 t378_leaf0 <= 1e+06
 T378_L1_f112_le: x_Neighborhood_SawyerW + 1e-06 t378_leaf1 <= 1
 T378_L1_f3_le: x_Overall_Qual + 999997.000001 t378_leaf1 <= 1e+06
 T378_R1_f31_ge: x_Garage_Yr_Blt - 1923 t378_leaf1 >= 0
 T378_L2_f112_le: x_Neighborhood_SawyerW + 1e-06 t378_leaf2 <= 1
 T378_R2_f3_ge: x_Overall_Qual - 3 t378_leaf2 >= 0
 T378_L2_f55_le: x_MS_SubClass_90 + 1e-06 t378_leaf2 <= 1
 T378_L3_f112_le: x_Neighborhood_SawyerW + 1e-06 t378_leaf3 <= 1
 T378_R3_f3_ge: x_Overall_Qual - 3 t378_leaf3 >= 0
 T378_R3_f55_ge: x_MS_SubClass_90 - t378_leaf3 >= 0
 T378_R4_f112_ge: x_Neighborhood_SawyerW - t378_leaf4 >= 0
 TREE_379_ONEHOT: t379_leaf0 + t379_leaf1 + t379_leaf2 + t379_leaf3
   + t379_leaf4 + t379_leaf5 + t379_leaf6 + t379_leaf7 = 1
 T379_L0_f3_le: x_Overall_Qual + 999993.000001 t379_leaf0 <= 1e+06
 T379_L0_f20_le: x_Gr_Liv_Area + 2674.760001 t379_leaf0 <= 3421.76
 T379_L0_f20_le: x_Gr_Liv_Area + 2701.760001 t379_leaf0 <= 3421.76
 T379_L1_f3_le: x_Overall_Qual + 999993.000001 t379_leaf1 <= 1e+06
 T379_L1_f20_le: x_Gr_Liv_Area + 2674.760001 t379_leaf1 <= 3421.76
 T379_R1_f20_ge: x_Gr_Liv_Area - 720 t379_leaf1 >= 0
 T379_L2_f3_le: x_Overall_Qual + 999993.000001 t379_leaf2 <= 1e+06
 T379_R2_f20_ge: x_Gr_Liv_Area - 747 t379_leaf2 >= 0
 T379_L2_f72_le: x_Alley_Pave + 1e-06 t379_leaf2 <= 1
 T379_L3_f3_le: x_Overall_Qual + 999993.000001 t379_leaf3 <= 1e+06
 T379_R3_f20_ge: x_Gr_Liv_Area - 747 t379_leaf3 >= 0
 T379_R3_f72_ge: x_Alley_Pave - t379_leaf3 >= 0
 T379_R4_f3_ge: x_Overall_Qual - 7 t379_leaf4 >= 0
 T379_L4_f296_le: x_Sale_Condition_Family + 1e-06 t379_leaf4 <= 1
 T379_L4_f14_le: x_Bsmt_Unf_SF + 1649.680001 t379_leaf4 <= 1812.68
 T379_R5_f3_ge: x_Overall_Qual - 7 t379_leaf5 >= 0
 T379_L5_f296_le: x_Sale_Condition_Family + 1e-06 t379_leaf5 <= 1
 T379_R5_f14_ge: x_Bsmt_Unf_SF - 163 t379_leaf5 >= 0
 T379_R6_f3_ge: x_Overall_Qual - 7 t379_leaf6 >= 0
 T379_R6_f296_ge: x_Sale_Condition_Family - t379_leaf6 >= 0
 T379_L6_f7_le: x_Mas_Vnr_Area + 849.070001 t379_leaf6 <= 931.07
 T379_R7_f3_ge: x_Overall_Qual - 7 t379_leaf7 >= 0
 T379_R7_f296_ge: x_Sale_Condition_Family - t379_leaf7 >= 0
 T379_R7_f7_ge: x_Mas_Vnr_Area - 82 t379_leaf7 >= 0
 TREE_380_ONEHOT: t380_leaf0 + t380_leaf1 + t380_leaf2 + t380_leaf3 = 1
 T380_L0_f40_le: x_Screen_Porch + 112.270001 t380_leaf0 <= 268.27
 T380_L0_f33_le: x_Garage_Area + 963.750001 t380_leaf0 <= 1163.75
 T380_L0_f15_le: x_Total_Bsmt_SF + 1237.480001 t380_leaf0 <= 2514.48
 T380_L1_f40_le: x_Screen_Porch + 112.270001 t380_leaf1 <= 268.27
 T380_L1_f33_le: x_Garage_Area + 963.750001 t380_leaf1 <= 1163.75
 T380_R1_f15_ge: x_Total_Bsmt_SF - 1277 t380_leaf1 >= 0
 T380_L2_f40_le: x_Screen_Porch + 112.270001 t380_leaf2 <= 268.27
 T380_R2_f33_ge: x_Garage_Area - 200 t380_leaf2 >= 0
 T380_R3_f40_ge: x_Screen_Porch - 156 t380_leaf3 >= 0
 TREE_381_ONEHOT: t381_leaf0 + t381_leaf1 + t381_leaf2 + t381_leaf3
   + t381_leaf4 + t381_leaf5 + t381_leaf6 = 1
 T381_L0_f15_le: x_Total_Bsmt_SF + 2033.480001 t381_leaf0 <= 2514.48
 T381_L0_f192_le: x_Exterior_2nd_Wd_Sdng + 1e-06 t381_leaf0 <= 1
 T381_L0_f17_le: x_1st_Flr_SF + 1379.900001 t381_leaf0 <= 2534.9
 T381_L1_f15_le: x_Total_Bsmt_SF + 2033.480001 t381_leaf1 <= 2514.48
 T381_L1_f192_le: x_Exterior_2nd_Wd_Sdng + 1e-06 t381_leaf1 <= 1
 T381_R1_f17_ge: x_1st_Flr_SF - 1155 t381_leaf1 >= 0
 T381_L2_f15_le: x_Total_Bsmt_SF + 2033.480001 t381_leaf2 <= 2514.48
 T381_R2_f192_ge: x_Exterior_2nd_Wd_Sdng - t381_leaf2 >= 0
 T381_R3_f15_ge: x_Total_Bsmt_SF - 481 t381_leaf3 >= 0
 T381_L3_f5_le: x_Year_Built + 998089.000001 t381_leaf3 <= 1e+06
 T381_L3_f18_le: x_2nd_Flr_SF + 658.540001 t381_leaf3 <= 1641.54
 T381_R4_f15_ge: x_Total_Bsmt_SF - 481 t381_leaf4 >= 0
 T381_L4_f5_le: x_Year_Built + 998089.000001 t381_leaf4 <= 1e+06
 T381_R4_f18_ge: x_2nd_Flr_SF - 983 t381_leaf4 >= 0
 T381_R5_f15_ge: x_Total_Bsmt_SF - 481 t381_leaf5 >= 0
 T381_R5_f5_ge: x_Year_Built - 1911 t381_leaf5 >= 0
 T381_L5_f272_le: x_Mo_Sold_2 + 1e-06 t381_leaf5 <= 1
 T381_R6_f15_ge: x_Total_Bsmt_SF - 481 t381_leaf6 >= 0
 T381_R6_f5_ge: x_Year_Built - 1911 t381_leaf6 >= 0
 T381_R6_f272_ge: x_Mo_Sold_2 - t381_leaf6 >= 0
 TREE_382_ONEHOT: t382_leaf0 + t382_leaf1 + t382_leaf2 + t382_leaf3
   + t382_leaf4 + t382_leaf5 = 1
 T382_L0_f27_le: x_Kitchen_Qual + 2.000001 t382_leaf0 <= 4
 T382_L0_f209_le: x_Bsmt_Exposure_No + 1e-06 t382_leaf0 <= 1
 T382_L1_f27_le: x_Kitchen_Qual + 2.000001 t382_leaf1 <= 4
 T382_R1_f209_ge: x_Bsmt_Exposure_No - t382_leaf1 >= 0
 T382_L1_f49_le: x_MS_SubClass_50 + 1e-06 t382_leaf1 <= 1
 T382_L2_f27_le: x_Kitchen_Qual + 2.000001 t382_leaf2 <= 4
 T382_R2_f209_ge: x_Bsmt_Exposure_No - t382_leaf2 >= 0
 T382_R2_f49_ge: x_MS_SubClass_50 - t382_leaf2 >= 0
 T382_R3_f27_ge: x_Kitchen_Qual >= 2
 T382_L3_f279_le: x_Mo_Sold_9 + 1e-06 t382_leaf3 <= 1
 T382_L3_f14_le: x_Bsmt_Unf_SF + 1672.680001 t382_leaf3 <= 1812.68
 T382_R4_f27_ge: x_Kitchen_Qual >= 2
 T382_L4_f279_le: x_Mo_Sold_9 + 1e-06 t382_leaf4 <= 1
 T382_R4_f14_ge: x_Bsmt_Unf_SF - 140 t382_leaf4 >= 0
 T382_R5_f27_ge: x_Kitchen_Qual >= 2
 T382_R5_f279_ge: x_Mo_Sold_9 - t382_leaf5 >= 0
 TREE_383_ONEHOT: t383_leaf0 + t383_leaf1 + t383_leaf2 + t383_leaf3 = 1
 T383_L0_f28_le: x_TotRms_AbvGrd + 0.080001 t383_leaf0 <= 12.08
 T383_L0_f20_le: x_Gr_Liv_Area + 2404.760001 t383_leaf0 <= 3421.76
 T383_L0_f9_le: x_Exter_Cond + 2.000001 t383_leaf0 <= 4
 T383_L1_f28_le: x_TotRms_AbvGrd + 0.080001 t383_leaf1 <= 12.08
 T383_L1_f20_le: x_Gr_Liv_Area + 2404.760001 t383_leaf1 <= 3421.76
 T383_R1_f9_ge: x_Exter_Cond - 3 t383_leaf1 >= -1
 T383_L2_f28_le: x_TotRms_AbvGrd + 0.080001 t383_leaf2 <= 12.08
 T383_R2_f20_ge: x_Gr_Liv_Area - 1017 t383_leaf2 >= 0
 T383_R3_f28_ge: x_TotRms_AbvGrd - 12 t383_leaf3 >= 0
 TREE_384_ONEHOT: t384_leaf0 + t384_leaf1 + t384_leaf2 + t384_leaf3 = 1
 T384_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t384_leaf0 <= 1
 T384_R1_f245_ge: x_FunctioNo_aplical_Typ - t384_leaf1 >= 0
 T384_L1_f20_le: x_Gr_Liv_Area + 2404.760001 t384_leaf1 <= 3421.76
 T384_L1_f5_le: x_Year_Built + 998076.000001 t384_leaf1 <= 1e+06
 T384_R2_f245_ge: x_FunctioNo_aplical_Typ - t384_leaf2 >= 0
 T384_L2_f20_le: x_Gr_Liv_Area + 2404.760001 t384_leaf2 <= 3421.76
 T384_R2_f5_ge: x_Year_Built - 1924 t384_leaf2 >= 0
 T384_R3_f245_ge: x_FunctioNo_aplical_Typ - t384_leaf3 >= 0
 T384_R3_f20_ge: x_Gr_Liv_Area - 1017 t384_leaf3 >= 0
 TREE_385_ONEHOT: t385_leaf0 + t385_leaf1 + t385_leaf2 + t385_leaf3 = 1
 T385_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t385_leaf0 <= 1
 T385_L0_f154_le: x_Roof_Matl_CompShg + 1e-06 t385_leaf0 <= 1
 T385_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t385_leaf1 <= 1
 T385_R1_f154_ge: x_Roof_Matl_CompShg - t385_leaf1 >= 0
 T385_L1_f3_le: x_Overall_Qual + 999993.000001 t385_leaf1 <= 1e+06
 T385_L2_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t385_leaf2 <= 1
 T385_R2_f154_ge: x_Roof_Matl_CompShg - t385_leaf2 >= 0
 T385_R2_f3_ge: x_Overall_Qual - 7 t385_leaf2 >= 0
 T385_R3_f177_ge: x_Exterior_2nd_AsbShng - t385_leaf3 >= 0
 TREE_386_ONEHOT: t386_leaf0 + t386_leaf1 + t386_leaf2 + t386_leaf3 = 1
 T386_L0_f33_le: x_Garage_Area + 9.750001000000111 t386_leaf0 <= 1163.75
 T386_L0_f5_le: x_Year_Built + 997995.000001 t386_leaf0 <= 1e+06
 T386_L0_f35_le: x_Garage_Cond + 3.000001 t386_leaf0 <= 4
 T386_L1_f33_le: x_Garage_Area + 9.750001000000111 t386_leaf1 <= 1163.75
 T386_L1_f5_le: x_Year_Built + 997995.000001 t386_leaf1 <= 1e+06
 T386_R1_f35_ge: x_Garage_Cond - 2 t386_leaf1 >= -1
 T386_L2_f33_le: x_Garage_Area + 9.750001000000111 t386_leaf2 <= 1163.75
 T386_R2_f5_ge: x_Year_Built - 2005 t386_leaf2 >= 0
 T386_R3_f33_ge: x_Garage_Area - 1154 t386_leaf3 >= 0
 TREE_387_ONEHOT: t387_leaf0 + t387_leaf1 + t387_leaf2 + t387_leaf3 = 1
 T387_L0_f0_le: x_Lot_Frontage + 54.530001 t387_leaf0 <= 175.53
 T387_L0_f71_le: x_Alley_No_aplica + 1e-06 t387_leaf0 <= 1
 T387_L1_f0_le: x_Lot_Frontage + 54.530001 t387_leaf1 <= 175.53
 T387_R1_f71_ge: x_Alley_No_aplica - t387_leaf1 >= 0
 T387_R2_f0_ge: x_Lot_Frontage - 121 t387_leaf2 >= 0
 T387_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t387_leaf2 <= 1
 T387_R3_f0_ge: x_Lot_Frontage - 121 t387_leaf3 >= 0
 T387_R3_f297_ge: x_Sale_Condition_Normal - t387_leaf3 >= 0
 TREE_388_ONEHOT: t388_leaf0 + t388_leaf1 + t388_leaf2 + t388_leaf3 = 1
 T388_L0_f26_le: x_Kitchen_AbvGr + 1.000001 t388_leaf0 <= 3
 T388_L0_f4_le: x_Overall_Cond + 999997.000001 t388_leaf0 <= 1e+06
 T388_L1_f26_le: x_Kitchen_AbvGr + 1.000001 t388_leaf1 <= 3
 T388_R1_f4_ge: x_Overall_Cond - 3 t388_leaf1 >= 0
 T388_L1_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t388_leaf1 <= 1
 T388_L2_f26_le: x_Kitchen_AbvGr + 1.000001 t388_leaf2 <= 3
 T388_R2_f4_ge: x_Overall_Cond - 3 t388_leaf2 >= 0
 T388_R2_f216_ge: x_BsmtFin_Type_1_Rec - t388_leaf2 >= 0
 T388_R3_f26_ge: x_Kitchen_AbvGr - 2 t388_leaf3 >= 0
 TREE_389_ONEHOT: t389_leaf0 = 1
 TREE_390_ONEHOT: t390_leaf0 + t390_leaf1 + t390_leaf2 + t390_leaf3
   + t390_leaf4 = 1
 T390_L0_f40_le: x_Screen_Porch + 40.27000099999998 t390_leaf0 <= 268.27
 T390_L0_f3_le: x_Overall_Qual + 999993.000001 t390_leaf0 <= 1e+06
 T390_L0_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t390_leaf0 <= 1
 T390_L1_f40_le: x_Screen_Porch + 40.27000099999998 t390_leaf1 <= 268.27
 T390_L1_f3_le: x_Overall_Qual + 999993.000001 t390_leaf1 <= 1e+06
 T390_R1_f214_ge: x_BsmtFin_Type_1_LwQ - t390_leaf1 >= 0
 T390_L2_f40_le: x_Screen_Porch + 40.27000099999998 t390_leaf2 <= 268.27
 T390_R2_f3_ge: x_Overall_Qual - 7 t390_leaf2 >= 0
 T390_L2_f37_le: x_Open_Porch_SF + 210.810001 t390_leaf2 <= 488.81
 T390_L3_f40_le: x_Screen_Porch + 40.27000099999998 t390_leaf3 <= 268.27
 T390_R3_f3_ge: x_Overall_Qual - 7 t390_leaf3 >= 0
 T390_R3_f37_ge: x_Open_Porch_SF - 278 t390_leaf3 >= 0
 T390_R4_f40_ge: x_Screen_Porch - 228 t390_leaf4 >= 0
 TREE_391_ONEHOT: t391_leaf0 + t391_leaf1 + t391_leaf2 + t391_leaf3 = 1
 T391_L0_f3_le: x_Overall_Qual + 999997.000001 t391_leaf0 <= 1e+06
 T391_R1_f3_ge: x_Overall_Qual - 3 t391_leaf1 >= 0
 T391_L1_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t391_leaf1 <= 1
 T391_L1_f96_le: x_Neighborhood_Edwards + 1e-06 t391_leaf1 <= 1
 T391_R2_f3_ge: x_Overall_Qual - 3 t391_leaf2 >= 0
 T391_L2_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t391_leaf2 <= 1
 T391_R2_f96_ge: x_Neighborhood_Edwards - t391_leaf2 >= 0
 T391_R3_f3_ge: x_Overall_Qual - 3 t391_leaf3 >= 0
 T391_R3_f218_ge: x_BsmtFin_Type_2_ALQ - t391_leaf3 >= 0
 TREE_392_ONEHOT: t392_leaf0 + t392_leaf1 + t392_leaf2 + t392_leaf3 = 1
 T392_L0_f31_le: x_Garage_Yr_Blt + 998092.000001 t392_leaf0 <= 1e+06
 T392_L0_f27_le: x_Kitchen_Qual + 2.000001 t392_leaf0 <= 4
 T392_L1_f31_le: x_Garage_Yr_Blt + 998092.000001 t392_leaf1 <= 1e+06
 T392_R1_f27_ge: x_Kitchen_Qual >= 2
 T392_L1_f10_le: x_Bsmt_Qual + 2.000001 t392_leaf1 <= 4
 T392_L2_f31_le: x_Garage_Yr_Blt + 998092.000001 t392_leaf2 <= 1e+06
 T392_R2_f27_ge: x_Kitchen_Qual >= 2
 T392_R2_f10_ge: x_Bsmt_Qual >= 2
 T392_R3_f31_ge: x_Garage_Yr_Blt - 1908 t392_leaf3 >= 0
 TREE_393_ONEHOT: t393_leaf0 + t393_leaf1 + t393_leaf2 + t393_leaf3
   + t393_leaf4 + t393_leaf5 = 1
 T393_L0_f3_le: x_Overall_Qual + 999992.000001 t393_leaf0 <= 1e+06
 T393_L0_f31_le: x_Garage_Yr_Blt + 997995.000001 t393_leaf0 <= 1e+06
 T393_L1_f3_le: x_Overall_Qual + 999992.000001 t393_leaf1 <= 1e+06
 T393_R1_f31_ge: x_Garage_Yr_Blt - 2005 t393_leaf1 >= 0
 T393_L1_f37_le: x_Open_Porch_SF + 455.810001 t393_leaf1 <= 488.81
 T393_L2_f3_le: x_Overall_Qual + 999992.000001 t393_leaf2 <= 1e+06
 T393_R2_f31_ge: x_Garage_Yr_Blt - 2005 t393_leaf2 >= 0
 T393_R2_f37_ge: x_Open_Porch_SF - 33 t393_leaf2 >= 0
 T393_R3_f3_ge: x_Overall_Qual - 8 t393_leaf3 >= 0
 T393_L3_f23_le: x_Full_Bath + 3.000001 t393_leaf3 <= 6
 T393_L3_f20_le: x_Gr_Liv_Area + 627.7600010000001 t393_leaf3 <= 3421.76
 T393_R4_f3_ge: x_Overall_Qual - 8 t393_leaf4 >= 0
 T393_L4_f23_le: x_Full_Bath + 3.000001 t393_leaf4 <= 6
 T393_R4_f20_ge: x_Gr_Liv_Area - 2794 t393_leaf4 >= 0
 T393_R5_f3_ge: x_Overall_Qual - 8 t393_leaf5 >= 0
 T393_R5_f23_ge: x_Full_Bath - 3 t393_leaf5 >= 0
 TREE_394_ONEHOT: t394_leaf0 + t394_leaf1 + t394_leaf2 + t394_leaf3 = 1
 T394_L0_f63_le: x_MS_Zoning_FV + 1e-06 t394_leaf0 <= 1
 T394_L0_f72_le: x_Alley_Pave + 1e-06 t394_leaf0 <= 1
 T394_L0_f35_le: x_Garage_Cond + 3.000001 t394_leaf0 <= 4
 T394_L1_f63_le: x_MS_Zoning_FV + 1e-06 t394_leaf1 <= 1
 T394_L1_f72_le: x_Alley_Pave + 1e-06 t394_leaf1 <= 1
 T394_R1_f35_ge: x_Garage_Cond - 2 t394_leaf1 >= -1
 T394_L2_f63_le: x_MS_Zoning_FV + 1e-06 t394_leaf2 <= 1
 T394_R2_f72_ge: x_Alley_Pave - t394_leaf2 >= 0
 T394_R3_f63_ge: x_MS_Zoning_FV - t394_leaf3 >= 0
 TREE_395_ONEHOT: t395_leaf0 + t395_leaf1 + t395_leaf2 + t395_leaf3
   + t395_leaf4 + t395_leaf5 + t395_leaf6 = 1
 T395_L0_f15_le: x_Total_Bsmt_SF + 1964.480001 t395_leaf0 <= 2514.48
 T395_L0_f31_le: x_Garage_Yr_Blt + 998021.000001 t395_leaf0 <= 1e+06
 T395_L0_f6_le: x_Year_Remod/Add + 998048.000001 t395_leaf0 <= 1e+06
 T395_L1_f15_le: x_Total_Bsmt_SF + 1964.480001 t395_leaf1 <= 2514.48
 T395_L1_f31_le: x_Garage_Yr_Blt + 998021.000001 t395_leaf1 <= 1e+06
 T395_R1_f6_ge: x_Year_Remod/Add - 1952 t395_leaf1 >= 0
 T395_L2_f15_le: x_Total_Bsmt_SF + 1964.480001 t395_leaf2 <= 2514.48
 T395_R2_f31_ge: x_Garage_Yr_Blt - 1979 t395_leaf2 >= 0
 T395_R3_f15_ge: x_Total_Bsmt_SF - 550 t395_leaf3 >= 0
 T395_L3_f37_le: x_Open_Porch_SF + 197.810001 t395_leaf3 <= 488.81
 T395_L3_f297_le: x_Sale_Condition_Normal + 1e-06 t395_leaf3 <= 1
 T395_R4_f15_ge: x_Total_Bsmt_SF - 550 t395_leaf4 >= 0
 T395_L4_f37_le: x_Open_Porch_SF + 197.810001 t395_leaf4 <= 488.81
 T395_R4_f297_ge: x_Sale_Condition_Normal - t395_leaf4 >= 0
 T395_R5_f15_ge: x_Total_Bsmt_SF - 550 t395_leaf5 >= 0
 T395_R5_f37_ge: x_Open_Porch_SF - 291 t395_leaf5 >= 0
 T395_L5_f297_le: x_Sale_Condition_Normal + 1e-06 t395_leaf5 <= 1
 T395_R6_f15_ge: x_Total_Bsmt_SF - 550 t395_leaf6 >= 0
 T395_R6_f37_ge: x_Open_Porch_SF - 291 t395_leaf6 >= 0
 T395_R6_f297_ge: x_Sale_Condition_Normal - t395_leaf6 >= 0
 TREE_396_ONEHOT: t396_leaf0 + t396_leaf1 + t396_leaf2 + t396_leaf3
   + t396_leaf4 = 1
 T396_L0_f92_le: x_Neighborhood_BrkSide + 1e-06 t396_leaf0 <= 1
 T396_L0_f3_le: x_Overall_Qual + 999997.000001 t396_leaf0 <= 1e+06
 T396_L0_f31_le: x_Garage_Yr_Blt + 998077.000001 t396_leaf0 <= 1e+06
 T396_L1_f92_le: x_Neighborhood_BrkSide + 1e-06 t396_leaf1 <= 1
 T396_L1_f3_le: x_Overall_Qual + 999997.000001 t396_leaf1 <= 1e+06
 T396_R1_f31_ge: x_Garage_Yr_Blt - 1923 t396_leaf1 >= 0
 T396_L2_f92_le: x_Neighborhood_BrkSide + 1e-06 t396_leaf2 <= 1
 T396_R2_f3_ge: x_Overall_Qual - 3 t396_leaf2 >= 0
 T396_L2_f8_le: x_Exter_Qual + 2.000001 t396_leaf2 <= 4
 T396_L3_f92_le: x_Neighborhood_BrkSide + 1e-06 t396_leaf3 <= 1
 T396_R3_f3_ge: x_Overall_Qual - 3 t396_leaf3 >= 0
 T396_R3_f8_ge: x_Exter_Qual >= 2
 T396_R4_f92_ge: x_Neighborhood_BrkSide - t396_leaf4 >= 0
 TREE_397_ONEHOT: t397_leaf0 + t397_leaf1 + t397_leaf2 + t397_leaf3
   + t397_leaf4 = 1
 T397_L0_f3_le: x_Overall_Qual + 999997.000001 t397_leaf0 <= 1e+06
 T397_R1_f3_ge: x_Overall_Qual - 3 t397_leaf1 >= 0
 T397_L1_f3_le: x_Overall_Qual + 999993.000001 t397_leaf1 <= 1e+06
 T397_L1_f28_le: x_TotRms_AbvGrd + 6.080001 t397_leaf1 <= 12.08
 T397_R2_f3_ge: x_Overall_Qual - 3 t397_leaf2 >= 0
 T397_L2_f3_le: x_Overall_Qual + 999993.000001 t397_leaf2 <= 1e+06
 T397_R2_f28_ge: x_TotRms_AbvGrd - 6 t397_leaf2 >= 0
 T397_R3_f3_ge: x_Overall_Qual - 3 t397_leaf3 >= 0
 T397_R3_f3_ge: x_Overall_Qual - 7 t397_leaf3 >= 0
 T397_L3_f5_le: x_Year_Built + 997995.000001 t397_leaf3 <= 1e+06
 T397_R4_f3_ge: x_Overall_Qual - 3 t397_leaf4 >= 0
 T397_R4_f3_ge: x_Overall_Qual - 7 t397_leaf4 >= 0
 T397_R4_f5_ge: x_Year_Built - 2005 t397_leaf4 >= 0
 TREE_398_ONEHOT: t398_leaf0 + t398_leaf1 + t398_leaf2 + t398_leaf3
   + t398_leaf4 = 1
 T398_L0_f18_le: x_2nd_Flr_SF + 653.540001 t398_leaf0 <= 1641.54
 T398_L0_f3_le: x_Overall_Qual + 999993.000001 t398_leaf0 <= 1e+06
 T398_L0_f11_le: x_Bsmt_Cond + 1.000001 t398_leaf0 <= 4
 T398_L1_f18_le: x_2nd_Flr_SF + 653.540001 t398_leaf1 <= 1641.54
 T398_L1_f3_le: x_Overall_Qual + 999993.000001 t398_leaf1 <= 1e+06
 T398_R1_f11_ge: x_Bsmt_Cond - 4 t398_leaf1 >= -1
 T398_L2_f18_le: x_2nd_Flr_SF + 653.540001 t398_leaf2 <= 1641.54
 T398_R2_f3_ge: x_Overall_Qual - 7 t398_leaf2 >= 0
 T398_L2_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t398_leaf2 <= 1
 T398_L3_f18_le: x_2nd_Flr_SF + 653.540001 t398_leaf3 <= 1641.54
 T398_R3_f3_ge: x_Overall_Qual - 7 t398_leaf3 >= 0
 T398_R3_f211_ge: x_BsmtFin_Type_1_ALQ - t398_leaf3 >= 0
 T398_R4_f18_ge: x_2nd_Flr_SF - 988 t398_leaf4 >= 0
 TREE_399_ONEHOT: t399_leaf0 + t399_leaf1 + t399_leaf2 + t399_leaf3
   + t399_leaf4 = 1
 T399_L0_f17_le: x_1st_Flr_SF + 1958.900001 t399_leaf0 <= 2534.9
 T399_R1_f17_ge: x_1st_Flr_SF - 576 t399_leaf1 >= 0
 T399_L1_f27_le: x_Kitchen_Qual + 2.000001 t399_leaf1 <= 4
 T399_L1_f31_le: x_Garage_Yr_Blt + 998065.000001 t399_leaf1 <= 1e+06
 T399_R2_f17_ge: x_1st_Flr_SF - 576 t399_leaf2 >= 0
 T399_L2_f27_le: x_Kitchen_Qual + 2.000001 t399_leaf2 <= 4
 T399_R2_f31_ge: x_Garage_Yr_Blt - 1935 t399_leaf2 >= 0
 T399_R3_f17_ge: x_1st_Flr_SF - 576 t399_leaf3 >= 0
 T399_R3_f27_ge: x_Kitchen_Qual >= 2
 T399_L3_f164_le: x_Exterior_1st_BrkFace + 1e-06 t399_leaf3 <= 1
 T399_R4_f17_ge: x_1st_Flr_SF - 576 t399_leaf4 >= 0
 T399_R4_f27_ge: x_Kitchen_Qual >= 2
 T399_R4_f164_ge: x_Exterior_1st_BrkFace - t399_leaf4 >= 0
 TREE_400_ONEHOT: t400_leaf0 + t400_leaf1 = 1
 T400_L0_f4_le: x_Overall_Cond + 999993.000001 t400_leaf0 <= 1e+06
 T400_R1_f4_ge: x_Overall_Cond - 7 t400_leaf1 >= 0
 TREE_401_ONEHOT: t401_leaf0 + t401_leaf1 = 1
 T401_L0_f3_le: x_Overall_Qual + 999997.000001 t401_leaf0 <= 1e+06
 T401_R1_f3_ge: x_Overall_Qual - 3 t401_leaf1 >= 0
 TREE_402_ONEHOT: t402_leaf0 + t402_leaf1 + t402_leaf2 + t402_leaf3 = 1
 T402_L0_f112_le: x_Neighborhood_SawyerW + 1e-06 t402_leaf0 <= 1
 T402_L0_f19_le: x_Low_Qual_Fin_SF + 999886.000001 t402_leaf0 <= 1e+06
 T402_L0_f20_le: x_Gr_Liv_Area + 1755.760001 t402_leaf0 <= 3421.76
 T402_L1_f112_le: x_Neighborhood_SawyerW + 1e-06 t402_leaf1 <= 1
 T402_L1_f19_le: x_Low_Qual_Fin_SF + 999886.000001 t402_leaf1 <= 1e+06
 T402_R1_f20_ge: x_Gr_Liv_Area - 1666 t402_leaf1 >= 0
 T402_L2_f112_le: x_Neighborhood_SawyerW + 1e-06 t402_leaf2 <= 1
 T402_R2_f19_ge: x_Low_Qual_Fin_SF - 114 t402_leaf2 >= 0
 T402_R3_f112_ge: x_Neighborhood_SawyerW - t402_leaf3 >= 0
 TREE_403_ONEHOT: t403_leaf0 + t403_leaf1 + t403_leaf2 + t403_leaf3
   + t403_leaf4 + t403_leaf5 = 1
 T403_L0_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t403_leaf0 <= 1
 T403_L0_f20_le: x_Gr_Liv_Area + 2269.760001000001 t403_leaf0 <= 3421.76
 T403_L0_f38_le: x_Enclosed_Porch + 134.750001 t403_leaf0 <= 292.75
 T403_L1_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t403_leaf1 <= 1
 T403_L1_f20_le: x_Gr_Liv_Area + 2269.760001000001 t403_leaf1 <= 3421.76
 T403_R1_f38_ge: x_Enclosed_Porch - 158 t403_leaf1 >= 0
 T403_L2_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t403_leaf2 <= 1
 T403_R2_f20_ge: x_Gr_Liv_Area - 1152 t403_leaf2 >= 0
 T403_L2_f15_le: x_Total_Bsmt_SF + 1530.480001 t403_leaf2 <= 2514.48
 T403_L3_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t403_leaf3 <= 1
 T403_R3_f20_ge: x_Gr_Liv_Area - 1152 t403_leaf3 >= 0
 T403_R3_f15_ge: x_Total_Bsmt_SF - 984 t403_leaf3 >= 0
 T403_R4_f211_ge: x_BsmtFin_Type_1_ALQ - t403_leaf4 >= 0
 T403_L4_f6_le: x_Year_Remod/Add + 998027.000001 t403_leaf4 <= 1e+06
 T403_R5_f211_ge: x_BsmtFin_Type_1_ALQ - t403_leaf5 >= 0
 T403_R5_f6_ge: x_Year_Remod/Add - 1973 t403_leaf5 >= 0
 TREE_404_ONEHOT: t404_leaf0 + t404_leaf1 + t404_leaf2 + t404_leaf3 = 1
 T404_L0_f8_le: x_Exter_Qual + 2.000001 t404_leaf0 <= 4
 T404_R1_f8_ge: x_Exter_Qual >= 2
 T404_L1_f1_le: x_Lot_Area + 760.0000010000003 t404_leaf1 <= 7000
 T404_R2_f8_ge: x_Exter_Qual >= 2
 T404_R2_f1_ge: x_Lot_Area >= 6240
 T404_L2_f3_le: x_Overall_Qual + 999995.000001 t404_leaf2 <= 1e+06
 T404_R3_f8_ge: x_Exter_Qual >= 2
 T404_R3_f1_ge: x_Lot_Area >= 6240
 T404_R3_f3_ge: x_Overall_Qual - 5 t404_leaf3 >= 0
 TREE_405_ONEHOT: t405_leaf0 + t405_leaf1 + t405_leaf2 + t405_leaf3
   + t405_leaf4 = 1
 T405_L0_f33_le: x_Garage_Area + 9.750001000000111 t405_leaf0 <= 1163.75
 T405_L0_f24_le: x_Half_Bath + 4.000001 t405_leaf0 <= 5
 T405_L0_f272_le: x_Mo_Sold_2 + 1e-06 t405_leaf0 <= 1
 T405_L1_f33_le: x_Garage_Area + 9.750001000000111 t405_leaf1 <= 1163.75
 T405_L1_f24_le: x_Half_Bath + 4.000001 t405_leaf1 <= 5
 T405_R1_f272_ge: x_Mo_Sold_2 - t405_leaf1 >= 0
 T405_L2_f33_le: x_Garage_Area + 9.750001000000111 t405_leaf2 <= 1163.75
 T405_R2_f24_ge: x_Half_Bath - t405_leaf2 >= 0
 T405_L2_f49_le: x_MS_SubClass_50 + 1e-06 t405_leaf2 <= 1
 T405_L3_f33_le: x_Garage_Area + 9.750001000000111 t405_leaf3 <= 1163.75
 T405_R3_f24_ge: x_Half_Bath - t405_leaf3 >= 0
 T405_R3_f49_ge: x_MS_SubClass_50 - t405_leaf3 >= 0
 T405_R4_f33_ge: x_Garage_Area - 1154 t405_leaf4 >= 0
 TREE_406_ONEHOT: t406_leaf0 + t406_leaf1 + t406_leaf2 + t406_leaf3 = 1
 T406_L0_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t406_leaf0 <= 1
 T406_L0_f226_le: x_Heating_GasA + 1e-06 t406_leaf0 <= 1
 T406_L1_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t406_leaf1 <= 1
 T406_R1_f226_ge: x_Heating_GasA - t406_leaf1 >= 0
 T406_L1_f37_le: x_Open_Porch_SF + 120.810001 t406_leaf1 <= 488.81
 T406_L2_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t406_leaf2 <= 1
 T406_R2_f226_ge: x_Heating_GasA - t406_leaf2 >= 0
 T406_R2_f37_ge: x_Open_Porch_SF - 368 t406_leaf2 >= 0
 T406_R3_f218_ge: x_BsmtFin_Type_2_ALQ - t406_leaf3 >= 0
 TREE_407_ONEHOT: t407_leaf0 + t407_leaf1 + t407_leaf2 + t407_leaf3
   + t407_leaf4 = 1
 T407_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t407_leaf0 <= 1
 T407_L0_f5_le: x_Year_Built + 998089.000001 t407_leaf0 <= 1e+06
 T407_L0_f6_le: x_Year_Remod/Add + 998001.000001 t407_leaf0 <= 1e+06
 T407_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t407_leaf1 <= 1
 T407_L1_f5_le: x_Year_Built + 998089.000001 t407_leaf1 <= 1e+06
 T407_R1_f6_ge: x_Year_Remod/Add - 1999 t407_leaf1 >= 0
 T407_L2_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t407_leaf2 <= 1
 T407_R2_f5_ge: x_Year_Built - 1911 t407_leaf2 >= 0
 T407_R3_f177_ge: x_Exterior_2nd_AsbShng - t407_leaf3 >= 0
 T407_L3_f31_le: x_Garage_Yr_Blt + 998079.000001 t407_leaf3 <= 1e+06
 T407_R4_f177_ge: x_Exterior_2nd_AsbShng - t407_leaf4 >= 0
 T407_R4_f31_ge: x_Garage_Yr_Blt - 1921 t407_leaf4 >= 0
 TREE_408_ONEHOT: t408_leaf0 + t408_leaf1 + t408_leaf2 + t408_leaf3
   + t408_leaf4 = 1
 T408_L0_f23_le: x_Full_Bath + 3.000001 t408_leaf0 <= 6
 T408_L0_f37_le: x_Open_Porch_SF + 206.810001 t408_leaf0 <= 488.81
 T408_L0_f8_le: x_Exter_Qual + 1e-06 t408_leaf0 <= 4
 T408_L1_f23_le: x_Full_Bath + 3.000001 t408_leaf1 <= 6
 T408_L1_f37_le: x_Open_Porch_SF + 206.810001 t408_leaf1 <= 488.81
 T408_R1_f8_ge: x_Exter_Qual - 2 t408_leaf1 >= 2
 T408_L2_f23_le: x_Full_Bath + 3.000001 t408_leaf2 <= 6
 T408_R2_f37_ge: x_Open_Porch_SF - 282 t408_leaf2 >= 0
 T408_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t408_leaf2 <= 1
 T408_L3_f23_le: x_Full_Bath + 3.000001 t408_leaf3 <= 6
 T408_R3_f37_ge: x_Open_Porch_SF - 282 t408_leaf3 >= 0
 T408_R3_f297_ge: x_Sale_Condition_Normal - t408_leaf3 >= 0
 T408_R4_f23_ge: x_Full_Bath - 3 t408_leaf4 >= 0
 TREE_409_ONEHOT: t409_leaf0 + t409_leaf1 + t409_leaf2 + t409_leaf3
   + t409_leaf4 + t409_leaf5 + t409_leaf6 = 1
 T409_L0_f119_le: x_Condition_1_Norm + 1e-06 t409_leaf0 <= 1
 T409_L0_f14_le: x_Bsmt_Unf_SF + 905.6800010000001 t409_leaf0 <= 1812.68
 T409_L0_f14_le: x_Bsmt_Unf_SF + 1636.680001 t409_leaf0 <= 1812.68
 T409_L1_f119_le: x_Condition_1_Norm + 1e-06 t409_leaf1 <= 1
 T409_L1_f14_le: x_Bsmt_Unf_SF + 905.6800010000001 t409_leaf1 <= 1812.68
 T409_R1_f14_ge: x_Bsmt_Unf_SF - 176 t409_leaf1 >= 0
 T409_L2_f119_le: x_Condition_1_Norm + 1e-06 t409_leaf2 <= 1
 T409_R2_f14_ge: x_Bsmt_Unf_SF - 907 t409_leaf2 >= 0
 T409_R3_f119_ge: x_Condition_1_Norm - t409_leaf3 >= 0
 T409_L3_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t409_leaf3 <= 1
 T409_L3_f12_le: x_BsmtFin_SF_1 + 1702.500001 t409_leaf3 <= 1984.5
 T409_R4_f119_ge: x_Condition_1_Norm - t409_leaf4 >= 0
 T409_L4_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t409_leaf4 <= 1
 T409_R4_f12_ge: x_BsmtFin_SF_1 - 282 t409_leaf4 >= 0
 T409_R5_f119_ge: x_Condition_1_Norm - t409_leaf5 >= 0
 T409_R5_f216_ge: x_BsmtFin_Type_1_Rec - t409_leaf5 >= 0
 T409_L5_f12_le: x_BsmtFin_SF_1 + 1512.500001 t409_leaf5 <= 1984.5
 T409_R6_f119_ge: x_Condition_1_Norm - t409_leaf6 >= 0
 T409_R6_f216_ge: x_BsmtFin_Type_1_Rec - t409_leaf6 >= 0
 T409_R6_f12_ge: x_BsmtFin_SF_1 - 472 t409_leaf6 >= 0
 TREE_410_ONEHOT: t410_leaf0 + t410_leaf1 + t410_leaf2 + t410_leaf3
   + t410_leaf4 = 1
 T410_L0_f17_le: x_1st_Flr_SF + 334.900001 t410_leaf0 <= 2534.9
 T410_L0_f14_le: x_Bsmt_Unf_SF + 555.6800010000002 t410_leaf0 <= 1812.68
 T410_L0_f113_le: x_Neighborhood_Somerst + 1e-06 t410_leaf0 <= 1
 T410_L1_f17_le: x_1st_Flr_SF + 334.900001 t410_leaf1 <= 2534.9
 T410_L1_f14_le: x_Bsmt_Unf_SF + 555.6800010000002 t410_leaf1 <= 1812.68
 T410_R1_f113_ge: x_Neighborhood_Somerst - t410_leaf1 >= 0
 T410_L2_f17_le: x_1st_Flr_SF + 334.900001 t410_leaf2 <= 2534.9
 T410_R2_f14_ge: x_Bsmt_Unf_SF - 1257 t410_leaf2 >= 0
 T410_L2_f199_le: x_Mas_Vnr_Type_Stone + 1e-06 t410_leaf2 <= 1
 T410_L3_f17_le: x_1st_Flr_SF + 334.900001 t410_leaf3 <= 2534.9
 T410_R3_f14_ge: x_Bsmt_Unf_SF - 1257 t410_leaf3 >= 0
 T410_R3_f199_ge: x_Mas_Vnr_Type_Stone - t410_leaf3 >= 0
 T410_R4_f17_ge: x_1st_Flr_SF - 2200 t410_leaf4 >= 0
 TREE_411_ONEHOT: t411_leaf0 + t411_leaf1 = 1
 T411_L0_f37_le: x_Open_Porch_SF + 413.810001 t411_leaf0 <= 488.81
 T411_R1_f37_ge: x_Open_Porch_SF - 75 t411_leaf1 >= 0
 TREE_412_ONEHOT: t412_leaf0 + t412_leaf1 + t412_leaf2 + t412_leaf3
   + t412_leaf4 + t412_leaf5 + t412_leaf6 + t412_leaf7 = 1
 T412_L0_f3_le: x_Overall_Qual + 999993.000001 t412_leaf0 <= 1e+06
 T412_L0_f1_le: x_Lot_Area <= 8399.999999
 T412_L0_f16_le: x_Heating_QC + 1e-06 t412_leaf0 <= 4
 T412_L1_f3_le: x_Overall_Qual + 999993.000001 t412_leaf1 <= 1e+06
 T412_L1_f1_le: x_Lot_Area <= 8399.999999
 T412_R1_f16_ge: x_Heating_QC - 2 t412_leaf1 >= 2
 T412_L2_f3_le: x_Overall_Qual + 999993.000001 t412_leaf2 <= 1e+06
 T412_R2_f1_ge: x_Lot_Area - 1400 t412_leaf2 >= 7000
 T412_L2_f231_le: x_Central_Air_N + 1e-06 t412_leaf2 <= 1
 T412_L3_f3_le: x_Overall_Qual + 999993.000001 t412_leaf3 <= 1e+06
 T412_R3_f1_ge: x_Lot_Area - 1400 t412_leaf3 >= 7000
 T412_R3_f231_ge: x_Central_Air_N - t412_leaf3 >= 0
 T412_R4_f3_ge: x_Overall_Qual - 7 t412_leaf4 >= 0
 T412_L4_f6_le: x_Year_Remod/Add + 998006.000001 t412_leaf4 <= 1e+06
 T412_L4_f297_le: x_Sale_Condition_Normal + 1e-06 t412_leaf4 <= 1
 T412_R5_f3_ge: x_Overall_Qual - 7 t412_leaf5 >= 0
 T412_L5_f6_le: x_Year_Remod/Add + 998006.000001 t412_leaf5 <= 1e+06
 T412_R5_f297_ge: x_Sale_Condition_Normal - t412_leaf5 >= 0
 T412_R6_f3_ge: x_Overall_Qual - 7 t412_leaf6 >= 0
 T412_R6_f6_ge: x_Year_Remod/Add - 1994 t412_leaf6 >= 0
 T412_L6_f77_le: x_Land_Contour_Bnk + 1e-06 t412_leaf6 <= 1
 T412_R7_f3_ge: x_Overall_Qual - 7 t412_leaf7 >= 0
 T412_R7_f6_ge: x_Year_Remod/Add - 1994 t412_leaf7 >= 0
 T412_R7_f77_ge: x_Land_Contour_Bnk - t412_leaf7 >= 0
 TREE_413_ONEHOT: t413_leaf0 + t413_leaf1 + t413_leaf2 + t413_leaf3 = 1
 T413_L0_f17_le: x_1st_Flr_SF + 1888.900001 t413_leaf0 <= 2534.9
 T413_R1_f17_ge: x_1st_Flr_SF - 646 t413_leaf1 >= 0
 T413_L1_f3_le: x_Overall_Qual + 999995.000001 t413_leaf1 <= 1e+06
 T413_L1_f86_le: x_Land_Slope_Gtl + 1e-06 t413_leaf1 <= 1
 T413_R2_f17_ge: x_1st_Flr_SF - 646 t413_leaf2 >= 0
 T413_L2_f3_le: x_Overall_Qual + 999995.000001 t413_leaf2 <= 1e+06
 T413_R2_f86_ge: x_Land_Slope_Gtl - t413_leaf2 >= 0
 T413_R3_f17_ge: x_1st_Flr_SF - 646 t413_leaf3 >= 0
 T413_R3_f3_ge: x_Overall_Qual - 5 t413_leaf3 >= 0
 TREE_414_ONEHOT: t414_leaf0 + t414_leaf1 + t414_leaf2 + t414_leaf3 = 1
 T414_L0_f280_le: x_Mo_Sold_10 + 1e-06 t414_leaf0 <= 1
 T414_L0_f11_le: x_Bsmt_Cond + 1.000001 t414_leaf0 <= 4
 T414_L0_f31_le: x_Garage_Yr_Blt + 998041.000001 t414_leaf0 <= 1e+06
 T414_L1_f280_le: x_Mo_Sold_10 + 1e-06 t414_leaf1 <= 1
 T414_L1_f11_le: x_Bsmt_Cond + 1.000001 t414_leaf1 <= 4
 T414_R1_f31_ge: x_Garage_Yr_Blt - 1959 t414_leaf1 >= 0
 T414_L2_f280_le: x_Mo_Sold_10 + 1e-06 t414_leaf2 <= 1
 T414_R2_f11_ge: x_Bsmt_Cond - 4 t414_leaf2 >= -1
 T414_R3_f280_ge: x_Mo_Sold_10 - t414_leaf3 >= 0
 TREE_415_ONEHOT: t415_leaf0 + t415_leaf1 + t415_leaf2 + t415_leaf3
   + t415_leaf4 = 1
 T415_L0_f7_le: x_Mas_Vnr_Area + 461.070001 t415_leaf0 <= 931.07
 T415_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t415_leaf0 <= 1
 T415_L0_f23_le: x_Full_Bath + 4.000001 t415_leaf0 <= 6
 T415_L1_f7_le: x_Mas_Vnr_Area + 461.070001 t415_leaf1 <= 931.07
 T415_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t415_leaf1 <= 1
 T415_R1_f23_ge: x_Full_Bath - 2 t415_leaf1 >= 0
 T415_L2_f7_le: x_Mas_Vnr_Area + 461.070001 t415_leaf2 <= 931.07
 T415_R2_f164_ge: x_Exterior_1st_BrkFace - t415_leaf2 >= 0
 T415_L2_f17_le: x_1st_Flr_SF + 1293.900001 t415_leaf2 <= 2534.9
 T415_L3_f7_le: x_Mas_Vnr_Area + 461.070001 t415_leaf3 <= 931.07
 T415_R3_f164_ge: x_Exterior_1st_BrkFace - t415_leaf3 >= 0
 T415_R3_f17_ge: x_1st_Flr_SF - 1241 t415_leaf3 >= 0
 T415_R4_f7_ge: x_Mas_Vnr_Area - 470 t415_leaf4 >= 0
 TREE_416_ONEHOT: t416_leaf0 + t416_leaf1 + t416_leaf2 + t416_leaf3
   + t416_leaf4 = 1
 T416_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t416_leaf0 <= 1
 T416_L0_f106_le: x_Neighborhood_No_aplicames + 1e-06 t416_leaf0 <= 1
 T416_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t416_leaf1 <= 1
 T416_R1_f106_ge: x_Neighborhood_No_aplicames - t416_leaf1 >= 0
 T416_R2_f293_ge: x_Sale_Condition_Abnorml - t416_leaf2 >= 0
 T416_L2_f9_le: x_Exter_Cond + 2.000001 t416_leaf2 <= 4
 T416_R3_f293_ge: x_Sale_Condition_Abnorml - t416_leaf3 >= 0
 T416_R3_f9_ge: x_Exter_Cond - 3 t416_leaf3 >= -1
 T416_L3_f103_le: x_Neighborhood_Mitchel + 1e-06 t416_leaf3 <= 1
 T416_R4_f293_ge: x_Sale_Condition_Abnorml - t416_leaf4 >= 0
 T416_R4_f9_ge: x_Exter_Cond - 3 t416_leaf4 >= -1
 T416_R4_f103_ge: x_Neighborhood_Mitchel - t416_leaf4 >= 0
 TREE_417_ONEHOT: t417_leaf0 + t417_leaf1 + t417_leaf2 + t417_leaf3 = 1
 T417_L0_f18_le: x_2nd_Flr_SF + 658.540001 t417_leaf0 <= 1641.54
 T417_L0_f7_le: x_Mas_Vnr_Area + 143.070001 t417_leaf0 <= 931.07
 T417_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t417_leaf0 <= 1
 T417_L1_f18_le: x_2nd_Flr_SF + 658.540001 t417_leaf1 <= 1641.54
 T417_L1_f7_le: x_Mas_Vnr_Area + 143.070001 t417_leaf1 <= 931.07
 T417_R1_f177_ge: x_Exterior_2nd_AsbShng - t417_leaf1 >= 0
 T417_L2_f18_le: x_2nd_Flr_SF + 658.540001 t417_leaf2 <= 1641.54
 T417_R2_f7_ge: x_Mas_Vnr_Area - 788 t417_leaf2 >= 0
 T417_R3_f18_ge: x_2nd_Flr_SF - 983 t417_leaf3 >= 0
 TREE_418_ONEHOT: t418_leaf0 + t418_leaf1 + t418_leaf2 + t418_leaf3
   + t418_leaf4 + t418_leaf5 = 1
 T418_L0_f272_le: x_Mo_Sold_2 + 1e-06 t418_leaf0 <= 1
 T418_L0_f120_le: x_Condition_1_PosA + 1e-06 t418_leaf0 <= 1
 T418_L0_f1_le: x_Lot_Area <= 11791.999999
 T418_L1_f272_le: x_Mo_Sold_2 + 1e-06 t418_leaf1 <= 1
 T418_L1_f120_le: x_Condition_1_PosA + 1e-06 t418_leaf1 <= 1
 T418_R1_f1_ge: x_Lot_Area - 4792 t418_leaf1 >= 7000
 T418_L2_f272_le: x_Mo_Sold_2 + 1e-06 t418_leaf2 <= 1
 T418_R2_f120_ge: x_Condition_1_PosA - t418_leaf2 >= 0
 T418_R3_f272_ge: x_Mo_Sold_2 - t418_leaf3 >= 0
 T418_L3_f297_le: x_Sale_Condition_Normal + 1e-06 t418_leaf3 <= 1
 T418_L3_f73_le: x_Lot_Shape_IR1 + 1e-06 t418_leaf3 <= 1
 T418_R4_f272_ge: x_Mo_Sold_2 - t418_leaf4 >= 0
 T418_L4_f297_le: x_Sale_Condition_Normal + 1e-06 t418_leaf4 <= 1
 T418_R4_f73_ge: x_Lot_Shape_IR1 - t418_leaf4 >= 0
 T418_R5_f272_ge: x_Mo_Sold_2 - t418_leaf5 >= 0
 T418_R5_f297_ge: x_Sale_Condition_Normal - t418_leaf5 >= 0
 TREE_419_ONEHOT: t419_leaf0 + t419_leaf1 + t419_leaf2 + t419_leaf3
   + t419_leaf4 = 1
 T419_L0_f7_le: x_Mas_Vnr_Area + 461.070001 t419_leaf0 <= 931.07
 T419_L0_f7_le: x_Mas_Vnr_Area + 534.070001 t419_leaf0 <= 931.07
 T419_L0_f33_le: x_Garage_Area + 292.750001 t419_leaf0 <= 1163.75
 T419_L1_f7_le: x_Mas_Vnr_Area + 461.070001 t419_leaf1 <= 931.07
 T419_L1_f7_le: x_Mas_Vnr_Area + 534.070001 t419_leaf1 <= 931.07
 T419_R1_f33_ge: x_Garage_Area - 871 t419_leaf1 >= 0
 T419_L2_f7_le: x_Mas_Vnr_Area + 461.070001 t419_leaf2 <= 931.07
 T419_R2_f7_ge: x_Mas_Vnr_Area - 397 t419_leaf2 >= 0
 T419_L2_f1_le: x_Lot_Area <= 10236.999999
 T419_L3_f7_le: x_Mas_Vnr_Area + 461.070001 t419_leaf3 <= 931.07
 T419_R3_f7_ge: x_Mas_Vnr_Area - 397 t419_leaf3 >= 0
 T419_R3_f1_ge: x_Lot_Area - 3237 t419_leaf3 >= 7000
 T419_R4_f7_ge: x_Mas_Vnr_Area - 470 t419_leaf4 >= 0
 TREE_420_ONEHOT: t420_leaf0 + t420_leaf1 + t420_leaf2 + t420_leaf3
   + t420_leaf4 + t420_leaf5 = 1
 T420_L0_f20_le: x_Gr_Liv_Area + 2611.760001 t420_leaf0 <= 3421.76
 T420_L0_f1_le: x_Lot_Area + 627.0000010000003 t420_leaf0 <= 7000
 T420_L1_f20_le: x_Gr_Liv_Area + 2611.760001 t420_leaf1 <= 3421.76
 T420_R1_f1_ge: x_Lot_Area >= 6373
 T420_R2_f20_ge: x_Gr_Liv_Area - 810 t420_leaf2 >= 0
 T420_L2_f46_le: x_MS_SubClass_30 + 1e-06 t420_leaf2 <= 1
 T420_L2_f96_le: x_Neighborhood_Edwards + 1e-06 t420_leaf2 <= 1
 T420_R3_f20_ge: x_Gr_Liv_Area - 810 t420_leaf3 >= 0
 T420_L3_f46_le: x_MS_SubClass_30 + 1e-06 t420_leaf3 <= 1
 T420_R3_f96_ge: x_Neighborhood_Edwards - t420_leaf3 >= 0
 T420_R4_f20_ge: x_Gr_Liv_Area - 810 t420_leaf4 >= 0
 T420_R4_f46_ge: x_MS_SubClass_30 - t420_leaf4 >= 0
 T420_L4_f33_le: x_Garage_Area + 730.750001 t420_leaf4 <= 1163.75
 T420_R5_f20_ge: x_Gr_Liv_Area - 810 t420_leaf5 >= 0
 T420_R5_f46_ge: x_MS_SubClass_30 - t420_leaf5 >= 0
 T420_R5_f33_ge: x_Garage_Area - 433 t420_leaf5 >= 0
 TREE_421_ONEHOT: t421_leaf0 + t421_leaf1 + t421_leaf2 + t421_leaf3
   + t421_leaf4 = 1
 T421_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t421_leaf0 <= 1
 T421_L0_f14_le: x_Bsmt_Unf_SF + 1699.680001 t421_leaf0 <= 1812.68
 T421_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t421_leaf1 <= 1
 T421_R1_f14_ge: x_Bsmt_Unf_SF - 113 t421_leaf1 >= 0
 T421_L1_f68_le: x_Street_Grvl + 1e-06 t421_leaf1 <= 1
 T421_L2_f161_le: x_Exterior_1st_AsbShng + 1e-06 t421_leaf2 <= 1
 T421_R2_f14_ge: x_Bsmt_Unf_SF - 113 t421_leaf2 >= 0
 T421_R2_f68_ge: x_Street_Grvl - t421_leaf2 >= 0
 T421_R3_f161_ge: x_Exterior_1st_AsbShng - t421_leaf3 >= 0
 T421_L3_f1_le: x_Lot_Area <= 9170.999999
 T421_R4_f161_ge: x_Exterior_1st_AsbShng - t421_leaf4 >= 0
 T421_R4_f1_ge: x_Lot_Area - 2171 t421_leaf4 >= 7000
 TREE_422_ONEHOT: t422_leaf0 + t422_leaf1 + t422_leaf2 + t422_leaf3 = 1
 T422_L0_f37_le: x_Open_Porch_SF + 196.810001 t422_leaf0 <= 488.81
 T422_L0_f4_le: x_Overall_Cond + 999997.000001 t422_leaf0 <= 1e+06
 T422_L1_f37_le: x_Open_Porch_SF + 196.810001 t422_leaf1 <= 488.81
 T422_R1_f4_ge: x_Overall_Cond - 3 t422_leaf1 >= 0
 T422_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t422_leaf1 <= 1
 T422_L2_f37_le: x_Open_Porch_SF + 196.810001 t422_leaf2 <= 488.81
 T422_R2_f4_ge: x_Overall_Cond - 3 t422_leaf2 >= 0
 T422_R2_f177_ge: x_Exterior_2nd_AsbShng - t422_leaf2 >= 0
 T422_R3_f37_ge: x_Open_Porch_SF - 292 t422_leaf3 >= 0
 TREE_423_ONEHOT: t423_leaf0 + t423_leaf1 + t423_leaf2 + t423_leaf3
   + t423_leaf4 + t423_leaf5 = 1
 T423_L0_f17_le: x_1st_Flr_SF + 1793.900001 t423_leaf0 <= 2534.9
 T423_L0_f106_le: x_Neighborhood_No_aplicames + 1e-06 t423_leaf0 <= 1
 T423_L0_f38_le: x_Enclosed_Porch + 204.750001 t423_leaf0 <= 292.75
 T423_L1_f17_le: x_1st_Flr_SF + 1793.900001 t423_leaf1 <= 2534.9
 T423_L1_f106_le: x_Neighborhood_No_aplicames + 1e-06 t423_leaf1 <= 1
 T423_R1_f38_ge: x_Enclosed_Porch - 88 t423_leaf1 >= 0
 T423_L2_f17_le: x_1st_Flr_SF + 1793.900001 t423_leaf2 <= 2534.9
 T423_R2_f106_ge: x_Neighborhood_No_aplicames - t423_leaf2 >= 0
 T423_R3_f17_ge: x_1st_Flr_SF - 741 t423_leaf3 >= 0
 T423_L3_f113_le: x_Neighborhood_Somerst + 1e-06 t423_leaf3 <= 1
 T423_L3_f86_le: x_Land_Slope_Gtl + 1e-06 t423_leaf3 <= 1
 T423_R4_f17_ge: x_1st_Flr_SF - 741 t423_leaf4 >= 0
 T423_L4_f113_le: x_Neighborhood_Somerst + 1e-06 t423_leaf4 <= 1
 T423_R4_f86_ge: x_Land_Slope_Gtl - t423_leaf4 >= 0
 T423_R5_f17_ge: x_1st_Flr_SF - 741 t423_leaf5 >= 0
 T423_R5_f113_ge: x_Neighborhood_Somerst - t423_leaf5 >= 0
 TREE_424_ONEHOT: t424_leaf0 + t424_leaf1 + t424_leaf2 + t424_leaf3
   + t424_leaf4 + t424_leaf5 = 1
 T424_L0_f119_le: x_Condition_1_Norm + 1e-06 t424_leaf0 <= 1
 T424_L0_f139_le: x_House_Style_1.5Fin + 1e-06 t424_leaf0 <= 1
 T424_L0_f3_le: x_Overall_Qual + 999993.000001 t424_leaf0 <= 1e+06
 T424_L1_f119_le: x_Condition_1_Norm + 1e-06 t424_leaf1 <= 1
 T424_L1_f139_le: x_House_Style_1.5Fin + 1e-06 t424_leaf1 <= 1
 T424_R1_f3_ge: x_Overall_Qual - 7 t424_leaf1 >= 0
 T424_L2_f119_le: x_Condition_1_Norm + 1e-06 t424_leaf2 <= 1
 T424_R2_f139_ge: x_House_Style_1.5Fin - t424_leaf2 >= 0
 T424_R3_f119_ge: x_Condition_1_Norm - t424_leaf3 >= 0
 T424_L3_f164_le: x_Exterior_1st_BrkFace + 1e-06 t424_leaf3 <= 1
 T424_L3_f3_le: x_Overall_Qual + 999997.000001 t424_leaf3 <= 1e+06
 T424_R4_f119_ge: x_Condition_1_Norm - t424_leaf4 >= 0
 T424_L4_f164_le: x_Exterior_1st_BrkFace + 1e-06 t424_leaf4 <= 1
 T424_R4_f3_ge: x_Overall_Qual - 3 t424_leaf4 >= 0
 T424_R5_f119_ge: x_Condition_1_Norm - t424_leaf5 >= 0
 T424_R5_f164_ge: x_Exterior_1st_BrkFace - t424_leaf5 >= 0
 TREE_425_ONEHOT: t425_leaf0 + t425_leaf1 = 1
 T425_L0_f68_le: x_Street_Grvl + 1e-06 t425_leaf0 <= 1
 T425_R1_f68_ge: x_Street_Grvl - t425_leaf1 >= 0
 TREE_426_ONEHOT: t426_leaf0 + t426_leaf1 = 1
 T426_L0_f29_le: x_Fireplaces + 1e-06 t426_leaf0 <= 3
 T426_R1_f29_ge: x_Fireplaces - 3 t426_leaf1 >= 0
 TREE_427_ONEHOT: t427_leaf0 + t427_leaf1 + t427_leaf2 + t427_leaf3
   + t427_leaf4 + t427_leaf5 = 1
 T427_L0_f34_le: x_Garage_Qual + 2.000001 t427_leaf0 <= 4
 T427_L0_f1_le: x_Lot_Area <= 8519.999999
 T427_L1_f34_le: x_Garage_Qual + 2.000001 t427_leaf1 <= 4
 T427_R1_f1_ge: x_Lot_Area - 1520 t427_leaf1 >= 7000
 T427_L1_f14_le: x_Bsmt_Unf_SF + 784.6800010000002 t427_leaf1 <= 1812.68
 T427_L2_f34_le: x_Garage_Qual + 2.000001 t427_leaf2 <= 4
 T427_R2_f1_ge: x_Lot_Area - 1520 t427_leaf2 >= 7000
 T427_R2_f14_ge: x_Bsmt_Unf_SF - 1028 t427_leaf2 >= 0
 T427_R3_f34_ge: x_Garage_Qual >= 2
 T427_L3_f1_le: x_Lot_Area <= 35132.999999
 T427_L3_f1_le: x_Lot_Area <= 23256.999999
 T427_R4_f34_ge: x_Garage_Qual >= 2
 T427_L4_f1_le: x_Lot_Area <= 35132.999999
 T427_R4_f1_ge: x_Lot_Area - 16257 t427_leaf4 >= 7000
 T427_R5_f34_ge: x_Garage_Qual >= 2
 T427_R5_f1_ge: x_Lot_Area - 28133 t427_leaf5 >= 7000
 TREE_428_ONEHOT: t428_leaf0 + t428_leaf1 + t428_leaf2 + t428_leaf3
   + t428_leaf4 + t428_leaf5 + t428_leaf6 = 1
 T428_L0_f46_le: x_MS_SubClass_30 + 1e-06 t428_leaf0 <= 1
 T428_L0_f27_le: x_Kitchen_Qual + 2.000001 t428_leaf0 <= 4
 T428_L0_f17_le: x_1st_Flr_SF + 1574.900001 t428_leaf0 <= 2534.9
 T428_L1_f46_le: x_MS_SubClass_30 + 1e-06 t428_leaf1 <= 1
 T428_L1_f27_le: x_Kitchen_Qual + 2.000001 t428_leaf1 <= 4
 T428_R1_f17_ge: x_1st_Flr_SF - 960 t428_leaf1 >= 0
 T428_L2_f46_le: x_MS_SubClass_30 + 1e-06 t428_leaf2 <= 1
 T428_R2_f27_ge: x_Kitchen_Qual >= 2
 T428_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t428_leaf2 <= 1
 T428_L3_f46_le: x_MS_SubClass_30 + 1e-06 t428_leaf3 <= 1
 T428_R3_f27_ge: x_Kitchen_Qual >= 2
 T428_R3_f164_ge: x_Exterior_1st_BrkFace - t428_leaf3 >= 0
 T428_R4_f46_ge: x_MS_SubClass_30 - t428_leaf4 >= 0
 T428_L4_f27_le: x_Kitchen_Qual + 2.000001 t428_leaf4 <= 4
 T428_R5_f46_ge: x_MS_SubClass_30 - t428_leaf5 >= 0
 T428_R5_f27_ge: x_Kitchen_Qual >= 2
 T428_L5_f31_le: x_Garage_Yr_Blt + 998071.000001 t428_leaf5 <= 1e+06
 T428_R6_f46_ge: x_MS_SubClass_30 - t428_leaf6 >= 0
 T428_R6_f27_ge: x_Kitchen_Qual >= 2
 T428_R6_f31_ge: x_Garage_Yr_Blt - 1929 t428_leaf6 >= 0
 TREE_429_ONEHOT: t429_leaf0 + t429_leaf1 + t429_leaf2 + t429_leaf3 = 1
 T429_L0_f92_le: x_Neighborhood_BrkSide + 1e-06 t429_leaf0 <= 1
 T429_L0_f33_le: x_Garage_Area + 9.750001000000111 t429_leaf0 <= 1163.75
 T429_L0_f154_le: x_Roof_Matl_CompShg + 1e-06 t429_leaf0 <= 1
 T429_L1_f92_le: x_Neighborhood_BrkSide + 1e-06 t429_leaf1 <= 1
 T429_L1_f33_le: x_Garage_Area + 9.750001000000111 t429_leaf1 <= 1163.75
 T429_R1_f154_ge: x_Roof_Matl_CompShg - t429_leaf1 >= 0
 T429_L2_f92_le: x_Neighborhood_BrkSide + 1e-06 t429_leaf2 <= 1
 T429_R2_f33_ge: x_Garage_Area - 1154 t429_leaf2 >= 0
 T429_R3_f92_ge: x_Neighborhood_BrkSide - t429_leaf3 >= 0
 TREE_430_ONEHOT: t430_leaf0 + t430_leaf1 + t430_leaf2 + t430_leaf3
   + t430_leaf4 + t430_leaf5 = 1
 T430_L0_f3_le: x_Overall_Qual + 999992.000001 t430_leaf0 <= 1e+06
 T430_L0_f23_le: x_Full_Bath + 4.000001 t430_leaf0 <= 6
 T430_L0_f3_le: x_Overall_Qual + 999997.000001 t430_leaf0 <= 1e+06
 T430_L1_f3_le: x_Overall_Qual + 999992.000001 t430_leaf1 <= 1e+06
 T430_L1_f23_le: x_Full_Bath + 4.000001 t430_leaf1 <= 6
 T430_R1_f3_ge: x_Overall_Qual - 3 t430_leaf1 >= 0
 T430_L2_f3_le: x_Overall_Qual + 999992.000001 t430_leaf2 <= 1e+06
 T430_R2_f23_ge: x_Full_Bath - 2 t430_leaf2 >= 0
 T430_L2_f106_le: x_Neighborhood_No_aplicames + 1e-06 t430_leaf2 <= 1
 T430_L3_f3_le: x_Overall_Qual + 999992.000001 t430_leaf3 <= 1e+06
 T430_R3_f23_ge: x_Full_Bath - 2 t430_leaf3 >= 0
 T430_R3_f106_ge: x_Neighborhood_No_aplicames - t430_leaf3 >= 0
 T430_R4_f3_ge: x_Overall_Qual - 8 t430_leaf4 >= 0
 T430_L4_f7_le: x_Mas_Vnr_Area + 143.070001 t430_leaf4 <= 931.07
 T430_R5_f3_ge: x_Overall_Qual - 8 t430_leaf5 >= 0
 T430_R5_f7_ge: x_Mas_Vnr_Area - 788 t430_leaf5 >= 0
 TREE_431_ONEHOT: t431_leaf0 + t431_leaf1 + t431_leaf2 = 1
 T431_L0_f9_le: x_Exter_Cond + 2.000001 t431_leaf0 <= 4
 T431_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t431_leaf0 <= 1
 T431_L1_f9_le: x_Exter_Cond + 2.000001 t431_leaf1 <= 4
 T431_R1_f293_ge: x_Sale_Condition_Abnorml - t431_leaf1 >= 0
 T431_R2_f9_ge: x_Exter_Cond - 3 t431_leaf2 >= -1
 TREE_432_ONEHOT: t432_leaf0 + t432_leaf1 + t432_leaf2 + t432_leaf3 = 1
 T432_L0_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t432_leaf0 <= 2514.48
 T432_L0_f296_le: x_Sale_Condition_Family + 1e-06 t432_leaf0 <= 1
 T432_L0_f4_le: x_Overall_Cond + 999993.000001 t432_leaf0 <= 1e+06
 T432_L1_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t432_leaf1 <= 2514.48
 T432_L1_f296_le: x_Sale_Condition_Family + 1e-06 t432_leaf1 <= 1
 T432_R1_f4_ge: x_Overall_Cond - 7 t432_leaf1 >= 0
 T432_L2_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t432_leaf2 <= 2514.48
 T432_R2_f296_ge: x_Sale_Condition_Family - t432_leaf2 >= 0
 T432_R3_f15_ge: x_Total_Bsmt_SF - 2136 t432_leaf3 >= 0
 TREE_433_ONEHOT: t433_leaf0 + t433_leaf1 + t433_leaf2 + t433_leaf3
   + t433_leaf4 = 1
 T433_L0_f203_le: x_Foundation_Slab + 1e-06 t433_leaf0 <= 1
 T433_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t433_leaf0 <= 1
 T433_L0_f37_le: x_Open_Porch_SF + 280.810001 t433_leaf0 <= 488.81
 T433_L1_f203_le: x_Foundation_Slab + 1e-06 t433_leaf1 <= 1
 T433_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t433_leaf1 <= 1
 T433_R1_f37_ge: x_Open_Porch_SF - 208 t433_leaf1 >= 0
 T433_L2_f203_le: x_Foundation_Slab + 1e-06 t433_leaf2 <= 1
 T433_R2_f161_ge: x_Exterior_1st_AsbShng - t433_leaf2 >= 0
 T433_L2_f4_le: x_Overall_Cond + 999995.000001 t433_leaf2 <= 1e+06
 T433_L3_f203_le: x_Foundation_Slab + 1e-06 t433_leaf3 <= 1
 T433_R3_f161_ge: x_Exterior_1st_AsbShng - t433_leaf3 >= 0
 T433_R3_f4_ge: x_Overall_Cond - 5 t433_leaf3 >= 0
 T433_R4_f203_ge: x_Foundation_Slab - t433_leaf4 >= 0
 TREE_434_ONEHOT: t434_leaf0 + t434_leaf1 + t434_leaf2 + t434_leaf3 = 1
 T434_L0_f18_le: x_2nd_Flr_SF + 214.5400010000001 t434_leaf0 <= 1641.54
 T434_L0_f20_le: x_Gr_Liv_Area + 549.7600010000001 t434_leaf0 <= 3421.76
 T434_L0_f7_le: x_Mas_Vnr_Area + 483.070001 t434_leaf0 <= 931.07
 T434_L1_f18_le: x_2nd_Flr_SF + 214.5400010000001 t434_leaf1 <= 1641.54
 T434_L1_f20_le: x_Gr_Liv_Area + 549.7600010000001 t434_leaf1 <= 3421.76
 T434_R1_f7_ge: x_Mas_Vnr_Area - 448 t434_leaf1 >= 0
 T434_L2_f18_le: x_2nd_Flr_SF + 214.5400010000001 t434_leaf2 <= 1641.54
 T434_R2_f20_ge: x_Gr_Liv_Area - 2872 t434_leaf2 >= 0
 T434_R3_f18_ge: x_2nd_Flr_SF - 1427 t434_leaf3 >= 0
 TREE_435_ONEHOT: t435_leaf0 + t435_leaf1 + t435_leaf2 + t435_leaf3
   + t435_leaf4 = 1
 T435_L0_f38_le: x_Enclosed_Porch + 98.750001 t435_leaf0 <= 292.75
 T435_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t435_leaf0 <= 1
 T435_L0_f38_le: x_Enclosed_Porch + 228.750001 t435_leaf0 <= 292.75
 T435_L1_f38_le: x_Enclosed_Porch + 98.750001 t435_leaf1 <= 292.75
 T435_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t435_leaf1 <= 1
 T435_R1_f38_ge: x_Enclosed_Porch - 64 t435_leaf1 >= 0
 T435_L2_f38_le: x_Enclosed_Porch + 98.750001 t435_leaf2 <= 292.75
 T435_R2_f245_ge: x_FunctioNo_aplical_Typ - t435_leaf2 >= 0
 T435_L2_f296_le: x_Sale_Condition_Family + 1e-06 t435_leaf2 <= 1
 T435_L3_f38_le: x_Enclosed_Porch + 98.750001 t435_leaf3 <= 292.75
 T435_R3_f245_ge: x_FunctioNo_aplical_Typ - t435_leaf3 >= 0
 T435_R3_f296_ge: x_Sale_Condition_Family - t435_leaf3 >= 0
 T435_R4_f38_ge: x_Enclosed_Porch - 194 t435_leaf4 >= 0
 TREE_436_ONEHOT: t436_leaf0 + t436_leaf1 + t436_leaf2 = 1
 T436_L0_f9_le: x_Exter_Cond + 2.000001 t436_leaf0 <= 4
 T436_L0_f276_le: x_Mo_Sold_6 + 1e-06 t436_leaf0 <= 1
 T436_L1_f9_le: x_Exter_Cond + 2.000001 t436_leaf1 <= 4
 T436_R1_f276_ge: x_Mo_Sold_6 - t436_leaf1 >= 0
 T436_R2_f9_ge: x_Exter_Cond - 3 t436_leaf2 >= -1
 TREE_437_ONEHOT: t437_leaf0 + t437_leaf1 + t437_leaf2 = 1
 T437_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t437_leaf0 <= 1
 T437_R1_f177_ge: x_Exterior_2nd_AsbShng - t437_leaf1 >= 0
 T437_L1_f4_le: x_Overall_Cond + 999995.000001 t437_leaf1 <= 1e+06
 T437_R2_f177_ge: x_Exterior_2nd_AsbShng - t437_leaf2 >= 0
 T437_R2_f4_ge: x_Overall_Cond - 5 t437_leaf2 >= 0
 TREE_438_ONEHOT: t438_leaf0 + t438_leaf1 + t438_leaf2 + t438_leaf3
   + t438_leaf4 = 1
 T438_L0_f119_le: x_Condition_1_Norm + 1e-06 t438_leaf0 <= 1
 T438_R1_f119_ge: x_Condition_1_Norm - t438_leaf1 >= 0
 T438_L1_f9_le: x_Exter_Cond + 2.000001 t438_leaf1 <= 4
 T438_L1_f10_le: x_Bsmt_Qual + 4.000001 t438_leaf1 <= 4
 T438_R2_f119_ge: x_Condition_1_Norm - t438_leaf2 >= 0
 T438_L2_f9_le: x_Exter_Cond + 2.000001 t438_leaf2 <= 4
 T438_R2_f10_ge: x_Bsmt_Qual >= 0
 T438_R3_f119_ge: x_Condition_1_Norm - t438_leaf3 >= 0
 T438_R3_f9_ge: x_Exter_Cond - 3 t438_leaf3 >= -1
 T438_L3_f18_le: x_2nd_Flr_SF + 807.540001 t438_leaf3 <= 1641.54
 T438_R4_f119_ge: x_Condition_1_Norm - t438_leaf4 >= 0
 T438_R4_f9_ge: x_Exter_Cond - 3 t438_leaf4 >= -1
 T438_R4_f18_ge: x_2nd_Flr_SF - 834 t438_leaf4 >= 0
 TREE_439_ONEHOT: t439_leaf0 + t439_leaf1 + t439_leaf2 + t439_leaf3
   + t439_leaf4 = 1
 T439_L0_f3_le: x_Overall_Qual + 999995.000001 t439_leaf0 <= 1e+06
 T439_L0_f1_le: x_Lot_Area <= 8898.999999
 T439_L0_f5_le: x_Year_Built + 998077.000001 t439_leaf0 <= 1e+06
 T439_L1_f3_le: x_Overall_Qual + 999995.000001 t439_leaf1 <= 1e+06
 T439_L1_f1_le: x_Lot_Area <= 8898.999999
 T439_R1_f5_ge: x_Year_Built - 1923 t439_leaf1 >= 0
 T439_L2_f3_le: x_Overall_Qual + 999995.000001 t439_leaf2 <= 1e+06
 T439_R2_f1_ge: x_Lot_Area - 1899 t439_leaf2 >= 7000
 T439_L2_f6_le: x_Year_Remod/Add + 998035.000001 t439_leaf2 <= 1e+06
 T439_L3_f3_le: x_Overall_Qual + 999995.000001 t439_leaf3 <= 1e+06
 T439_R3_f1_ge: x_Lot_Area - 1899 t439_leaf3 >= 7000
 T439_R3_f6_ge: x_Year_Remod/Add - 1965 t439_leaf3 >= 0
 T439_R4_f3_ge: x_Overall_Qual - 5 t439_leaf4 >= 0
 TREE_440_ONEHOT: t440_leaf0 + t440_leaf1 + t440_leaf2 + t440_leaf3 = 1
 T440_L0_f3_le: x_Overall_Qual + 999996.000001 t440_leaf0 <= 1e+06
 T440_R1_f3_ge: x_Overall_Qual - 4 t440_leaf1 >= 0
 T440_L1_f93_le: x_Neighborhood_ClearCr + 1e-06 t440_leaf1 <= 1
 T440_L1_f92_le: x_Neighborhood_BrkSide + 1e-06 t440_leaf1 <= 1
 T440_R2_f3_ge: x_Overall_Qual - 4 t440_leaf2 >= 0
 T440_L2_f93_le: x_Neighborhood_ClearCr + 1e-06 t440_leaf2 <= 1
 T440_R2_f92_ge: x_Neighborhood_BrkSide - t440_leaf2 >= 0
 T440_R3_f3_ge: x_Overall_Qual - 4 t440_leaf3 >= 0
 T440_R3_f93_ge: x_Neighborhood_ClearCr - t440_leaf3 >= 0
 TREE_441_ONEHOT: t441_leaf0 + t441_leaf1 + t441_leaf2 + t441_leaf3
   + t441_leaf4 + t441_leaf5 + t441_leaf6 + t441_leaf7 = 1
 T441_L0_f33_le: x_Garage_Area + 843.750001 t441_leaf0 <= 1163.75
 T441_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t441_leaf0 <= 1
 T441_L0_f11_le: x_Bsmt_Cond + 2.000001 t441_leaf0 <= 4
 T441_L1_f33_le: x_Garage_Area + 843.750001 t441_leaf1 <= 1163.75
 T441_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t441_leaf1 <= 1
 T441_R1_f11_ge: x_Bsmt_Cond - 3 t441_leaf1 >= -1
 T441_L2_f33_le: x_Garage_Area + 843.750001 t441_leaf2 <= 1163.75
 T441_R2_f297_ge: x_Sale_Condition_Normal - t441_leaf2 >= 0
 T441_L2_f3_le: x_Overall_Qual + 999997.000001 t441_leaf2 <= 1e+06
 T441_L3_f33_le: x_Garage_Area + 843.750001 t441_leaf3 <= 1163.75
 T441_R3_f297_ge: x_Sale_Condition_Normal - t441_leaf3 >= 0
 T441_R3_f3_ge: x_Overall_Qual - 3 t441_leaf3 >= 0
 T441_R4_f33_ge: x_Garage_Area - 320 t441_leaf4 >= 0
 T441_L4_f161_le: x_Exterior_1st_AsbShng + 1e-06 t441_leaf4 <= 1
 T441_L4_f5_le: x_Year_Built + 998085.000001 t441_leaf4 <= 1e+06
 T441_R5_f33_ge: x_Garage_Area - 320 t441_leaf5 >= 0
 T441_L5_f161_le: x_Exterior_1st_AsbShng + 1e-06 t441_leaf5 <= 1
 T441_R5_f5_ge: x_Year_Built - 1915 t441_leaf5 >= 0
 T441_R6_f33_ge: x_Garage_Area - 320 t441_leaf6 >= 0
 T441_R6_f161_ge: x_Exterior_1st_AsbShng - t441_leaf6 >= 0
 T441_L6_f5_le: x_Year_Built + 998079.000001 t441_leaf6 <= 1e+06
 T441_R7_f33_ge: x_Garage_Area - 320 t441_leaf7 >= 0
 T441_R7_f161_ge: x_Exterior_1st_AsbShng - t441_leaf7 >= 0
 T441_R7_f5_ge: x_Year_Built - 1921 t441_leaf7 >= 0
 TREE_442_ONEHOT: t442_leaf0 + t442_leaf1 + t442_leaf2 + t442_leaf3
   + t442_leaf4 = 1
 T442_L0_f147_le: x_Roof_Style_Flat + 1e-06 t442_leaf0 <= 1
 T442_L0_f92_le: x_Neighborhood_BrkSide + 1e-06 t442_leaf0 <= 1
 T442_L0_f24_le: x_Half_Bath + 4.000001 t442_leaf0 <= 5
 T442_L1_f147_le: x_Roof_Style_Flat + 1e-06 t442_leaf1 <= 1
 T442_L1_f92_le: x_Neighborhood_BrkSide + 1e-06 t442_leaf1 <= 1
 T442_R1_f24_ge: x_Half_Bath - t442_leaf1 >= 0
 T442_L2_f147_le: x_Roof_Style_Flat + 1e-06 t442_leaf2 <= 1
 T442_R2_f92_ge: x_Neighborhood_BrkSide - t442_leaf2 >= 0
 T442_L2_f33_le: x_Garage_Area + 923.750001 t442_leaf2 <= 1163.75
 T442_L3_f147_le: x_Roof_Style_Flat + 1e-06 t442_leaf3 <= 1
 T442_R3_f92_ge: x_Neighborhood_BrkSide - t442_leaf3 >= 0
 T442_R3_f33_ge: x_Garage_Area - 240 t442_leaf3 >= 0
 T442_R4_f147_ge: x_Roof_Style_Flat - t442_leaf4 >= 0
 TREE_443_ONEHOT: t443_leaf0 + t443_leaf1 + t443_leaf2 + t443_leaf3 = 1
 T443_L0_f68_le: x_Street_Grvl + 1e-06 t443_leaf0 <= 1
 T443_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t443_leaf0 <= 1
 T443_L1_f68_le: x_Street_Grvl + 1e-06 t443_leaf1 <= 1
 T443_R1_f293_ge: x_Sale_Condition_Abnorml - t443_leaf1 >= 0
 T443_L1_f38_le: x_Enclosed_Porch + 276.750001 t443_leaf1 <= 292.75
 T443_L2_f68_le: x_Street_Grvl + 1e-06 t443_leaf2 <= 1
 T443_R2_f293_ge: x_Sale_Condition_Abnorml - t443_leaf2 >= 0
 T443_R2_f38_ge: x_Enclosed_Porch - 16 t443_leaf2 >= 0
 T443_R3_f68_ge: x_Street_Grvl - t443_leaf3 >= 0
 TREE_444_ONEHOT: t444_leaf0 + t444_leaf1 + t444_leaf2 + t444_leaf3 = 1
 T444_L0_f30_le: x_Fireplace_Qu + 3.000001 t444_leaf0 <= 4
 T444_L0_f1_le: x_Lot_Area + 1024.000001 t444_leaf0 <= 7000
 T444_L1_f30_le: x_Fireplace_Qu + 3.000001 t444_leaf1 <= 4
 T444_R1_f1_ge: x_Lot_Area >= 5976
 T444_L1_f19_le: x_Low_Qual_Fin_SF + 999920.000001 t444_leaf1 <= 1e+06
 T444_L2_f30_le: x_Fireplace_Qu + 3.000001 t444_leaf2 <= 4
 T444_R2_f1_ge: x_Lot_Area >= 5976
 T444_R2_f19_ge: x_Low_Qual_Fin_SF - 80 t444_leaf2 >= 0
 T444_R3_f30_ge: x_Fireplace_Qu >= 1
 TREE_445_ONEHOT: t445_leaf0 + t445_leaf1 + t445_leaf2 + t445_leaf3
   + t445_leaf4 = 1
 T445_L0_f3_le: x_Overall_Qual + 999993.000001 t445_leaf0 <= 1e+06
 T445_L0_f24_le: x_Half_Bath + 4.000001 t445_leaf0 <= 5
 T445_L0_f10_le: x_Bsmt_Qual + 1.000001 t445_leaf0 <= 4
 T445_L1_f3_le: x_Overall_Qual + 999993.000001 t445_leaf1 <= 1e+06
 T445_L1_f24_le: x_Half_Bath + 4.000001 t445_leaf1 <= 5
 T445_R1_f10_ge: x_Bsmt_Qual - t445_leaf1 >= 2
 T445_L2_f3_le: x_Overall_Qual + 999993.000001 t445_leaf2 <= 1e+06
 T445_R2_f24_ge: x_Half_Bath - t445_leaf2 >= 0
 T445_R3_f3_ge: x_Overall_Qual - 7 t445_leaf3 >= 0
 T445_L3_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t445_leaf3 <= 1
 T445_R4_f3_ge: x_Overall_Qual - 7 t445_leaf4 >= 0
 T445_R4_f211_ge: x_BsmtFin_Type_1_ALQ - t445_leaf4 >= 0
 TREE_446_ONEHOT: t446_leaf0 + t446_leaf1 + t446_leaf2 + t446_leaf3
   + t446_leaf4 = 1
 T446_L0_f219_le: x_BsmtFin_Type_2_BLQ + 1e-06 t446_leaf0 <= 1
 T446_L0_f5_le: x_Year_Built + 998068.000001 t446_leaf0 <= 1e+06
 T446_L0_f237_le: x_Electrical_SBrkr + 1e-06 t446_leaf0 <= 1
 T446_L1_f219_le: x_BsmtFin_Type_2_BLQ + 1e-06 t446_leaf1 <= 1
 T446_L1_f5_le: x_Year_Built + 998068.000001 t446_leaf1 <= 1e+06
 T446_R1_f237_ge: x_Electrical_SBrkr - t446_leaf1 >= 0
 T446_L2_f219_le: x_BsmtFin_Type_2_BLQ + 1e-06 t446_leaf2 <= 1
 T446_R2_f5_ge: x_Year_Built - 1932 t446_leaf2 >= 0
 T446_L2_f41_le: x_Pool_Area + 76.77000099999998 t446_leaf2 <= 444.77
 T446_L3_f219_le: x_BsmtFin_Type_2_BLQ + 1e-06 t446_leaf3 <= 1
 T446_R3_f5_ge: x_Year_Built - 1932 t446_leaf3 >= 0
 T446_R3_f41_ge: x_Pool_Area - 368 t446_leaf3 >= 0
 T446_R4_f219_ge: x_BsmtFin_Type_2_BLQ - t446_leaf4 >= 0
 TREE_447_ONEHOT: t447_leaf0 + t447_leaf1 + t447_leaf2 + t447_leaf3
   + t447_leaf4 = 1
 T447_L0_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t447_leaf0 <= 1
 T447_L0_f5_le: x_Year_Built + 997994.000001 t447_leaf0 <= 1e+06
 T447_L0_f17_le: x_1st_Flr_SF + 1150.900001 t447_leaf0 <= 2534.9
 T447_L1_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t447_leaf1 <= 1
 T447_L1_f5_le: x_Year_Built + 997994.000001 t447_leaf1 <= 1e+06
 T447_R1_f17_ge: x_1st_Flr_SF - 1384 t447_leaf1 >= 0
 T447_L2_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t447_leaf2 <= 1
 T447_R2_f5_ge: x_Year_Built - 2006 t447_leaf2 >= 0
 T447_L2_f272_le: x_Mo_Sold_2 + 1e-06 t447_leaf2 <= 1
 T447_L3_f211_le: x_BsmtFin_Type_1_ALQ + 1e-06 t447_leaf3 <= 1
 T447_R3_f5_ge: x_Year_Built - 2006 t447_leaf3 >= 0
 T447_R3_f272_ge: x_Mo_Sold_2 - t447_leaf3 >= 0
 T447_R4_f211_ge: x_BsmtFin_Type_1_ALQ - t447_leaf4 >= 0
 TREE_448_ONEHOT: t448_leaf0 + t448_leaf1 + t448_leaf2 + t448_leaf3
   + t448_leaf4 + t448_leaf5 = 1
 T448_L0_f3_le: x_Overall_Qual + 999993.000001 t448_leaf0 <= 1e+06
 T448_L0_f18_le: x_2nd_Flr_SF + 1184.540001 t448_leaf0 <= 1641.54
 T448_L0_f14_le: x_Bsmt_Unf_SF + 1214.680001 t448_leaf0 <= 1812.68
 T448_L1_f3_le: x_Overall_Qual + 999993.000001 t448_leaf1 <= 1e+06
 T448_L1_f18_le: x_2nd_Flr_SF + 1184.540001 t448_leaf1 <= 1641.54
 T448_R1_f14_ge: x_Bsmt_Unf_SF - 598 t448_leaf1 >= 0
 T448_L2_f3_le: x_Overall_Qual + 999993.000001 t448_leaf2 <= 1e+06
 T448_R2_f18_ge: x_2nd_Flr_SF - 457 t448_leaf2 >= 0
 T448_R3_f3_ge: x_Overall_Qual - 7 t448_leaf3 >= 0
 T448_L3_f114_le: x_Neighborhood_StoneBr + 1e-06 t448_leaf3 <= 1
 T448_R4_f3_ge: x_Overall_Qual - 7 t448_leaf4 >= 0
 T448_R4_f114_ge: x_Neighborhood_StoneBr - t448_leaf4 >= 0
 T448_L4_f1_le: x_Lot_Area + 400.0000010000003 t448_leaf4 <= 7000
 T448_R5_f3_ge: x_Overall_Qual - 7 t448_leaf5 >= 0
 T448_R5_f114_ge: x_Neighborhood_StoneBr - t448_leaf5 >= 0
 T448_R5_f1_ge: x_Lot_Area >= 6600
 TREE_449_ONEHOT: t449_leaf0 = 1
 TREE_450_ONEHOT: t450_leaf0 = 1
 TREE_451_ONEHOT: t451_leaf0 + t451_leaf1 + t451_leaf2 + t451_leaf3 = 1
 T451_L0_f7_le: x_Mas_Vnr_Area + 258.070001 t451_leaf0 <= 931.07
 T451_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t451_leaf0 <= 1
 T451_L1_f7_le: x_Mas_Vnr_Area + 258.070001 t451_leaf1 <= 931.07
 T451_R1_f161_ge: x_Exterior_1st_AsbShng - t451_leaf1 >= 0
 T451_L1_f1_le: x_Lot_Area <= 9604.999999
 T451_L2_f7_le: x_Mas_Vnr_Area + 258.070001 t451_leaf2 <= 931.07
 T451_R2_f161_ge: x_Exterior_1st_AsbShng - t451_leaf2 >= 0
 T451_R2_f1_ge: x_Lot_Area - 2605 t451_leaf2 >= 7000
 T451_R3_f7_ge: x_Mas_Vnr_Area - 673 t451_leaf3 >= 0
 TREE_452_ONEHOT: t452_leaf0 + t452_leaf1 + t452_leaf2 + t452_leaf3 = 1
 T452_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t452_leaf0 <= 1
 T452_L0_f86_le: x_Land_Slope_Gtl + 1e-06 t452_leaf0 <= 1
 T452_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t452_leaf1 <= 1
 T452_R1_f86_ge: x_Land_Slope_Gtl - t452_leaf1 >= 0
 T452_R2_f161_ge: x_Exterior_1st_AsbShng - t452_leaf2 >= 0
 T452_L2_f12_le: x_BsmtFin_SF_1 + 1808.500001 t452_leaf2 <= 1984.5
 T452_R3_f161_ge: x_Exterior_1st_AsbShng - t452_leaf3 >= 0
 T452_R3_f12_ge: x_BsmtFin_SF_1 - 176 t452_leaf3 >= 0
 TREE_453_ONEHOT: t453_leaf0 + t453_leaf1 + t453_leaf2 + t453_leaf3
   + t453_leaf4 = 1
 T453_L0_f272_le: x_Mo_Sold_2 + 1e-06 t453_leaf0 <= 1
 T453_L0_f71_le: x_Alley_No_aplica + 1e-06 t453_leaf0 <= 1
 T453_L1_f272_le: x_Mo_Sold_2 + 1e-06 t453_leaf1 <= 1
 T453_R1_f71_ge: x_Alley_No_aplica - t453_leaf1 >= 0
 T453_L1_f296_le: x_Sale_Condition_Family + 1e-06 t453_leaf1 <= 1
 T453_L2_f272_le: x_Mo_Sold_2 + 1e-06 t453_leaf2 <= 1
 T453_R2_f71_ge: x_Alley_No_aplica - t453_leaf2 >= 0
 T453_R2_f296_ge: x_Sale_Condition_Family - t453_leaf2 >= 0
 T453_R3_f272_ge: x_Mo_Sold_2 - t453_leaf3 >= 0
 T453_L3_f297_le: x_Sale_Condition_Normal + 1e-06 t453_leaf3 <= 1
 T453_R4_f272_ge: x_Mo_Sold_2 - t453_leaf4 >= 0
 T453_R4_f297_ge: x_Sale_Condition_Normal - t453_leaf4 >= 0
 TREE_454_ONEHOT: t454_leaf0 + t454_leaf1 + t454_leaf2 + t454_leaf3
   + t454_leaf4 + t454_leaf5 = 1
 T454_L0_f6_le: x_Year_Remod/Add + 997994.000001 t454_leaf0 <= 1e+06
 T454_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t454_leaf0 <= 1
 T454_L0_f54_le: x_MS_SubClass_85 + 1e-06 t454_leaf0 <= 1
 T454_L1_f6_le: x_Year_Remod/Add + 997994.000001 t454_leaf1 <= 1e+06
 T454_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t454_leaf1 <= 1
 T454_R1_f54_ge: x_MS_SubClass_85 - t454_leaf1 >= 0
 T454_L2_f6_le: x_Year_Remod/Add + 997994.000001 t454_leaf2 <= 1e+06
 T454_R2_f177_ge: x_Exterior_2nd_AsbShng - t454_leaf2 >= 0
 T454_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t454_leaf2 <= 1
 T454_L3_f6_le: x_Year_Remod/Add + 997994.000001 t454_leaf3 <= 1e+06
 T454_R3_f177_ge: x_Exterior_2nd_AsbShng - t454_leaf3 >= 0
 T454_R3_f297_ge: x_Sale_Condition_Normal - t454_leaf3 >= 0
 T454_R4_f6_ge: x_Year_Remod/Add - 2006 t454_leaf4 >= 0
 T454_L4_f16_le: x_Heating_QC + 1e-06 t454_leaf4 <= 4
 T454_R5_f6_ge: x_Year_Remod/Add - 2006 t454_leaf5 >= 0
 T454_R5_f16_ge: x_Heating_QC - 2 t454_leaf5 >= 2
 TREE_455_ONEHOT: t455_leaf0 + t455_leaf1 + t455_leaf2 + t455_leaf3 = 1
 T455_L0_f3_le: x_Overall_Qual + 999990.000001 t455_leaf0 <= 1e+06
 T455_L0_f3_le: x_Overall_Qual + 999993.000001 t455_leaf0 <= 1e+06
 T455_L0_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t455_leaf0 <= 1
 T455_L1_f3_le: x_Overall_Qual + 999990.000001 t455_leaf1 <= 1e+06
 T455_L1_f3_le: x_Overall_Qual + 999993.000001 t455_leaf1 <= 1e+06
 T455_R1_f216_ge: x_BsmtFin_Type_1_Rec - t455_leaf1 >= 0
 T455_L2_f3_le: x_Overall_Qual + 999990.000001 t455_leaf2 <= 1e+06
 T455_R2_f3_ge: x_Overall_Qual - 7 t455_leaf2 >= 0
 T455_R3_f3_ge: x_Overall_Qual - 10 t455_leaf3 >= 0
 TREE_456_ONEHOT: t456_leaf0 = 1
 TREE_457_ONEHOT: t457_leaf0 + t457_leaf1 = 1
 T457_L0_f41_le: x_Pool_Area + 0.770001 t457_leaf0 <= 444.77
 T457_R1_f41_ge: x_Pool_Area - 444 t457_leaf1 >= 0
 TREE_458_ONEHOT: t458_leaf0 + t458_leaf1 + t458_leaf2 + t458_leaf3
   + t458_leaf4 + t458_leaf5 + t458_leaf6 = 1
 T458_L0_f21_le: x_Bsmt_Full_Bath + 1.020001 t458_leaf0 <= 2.02
 T458_L0_f12_le: x_BsmtFin_SF_1 + 1168.500001 t458_leaf0 <= 1984.5
 T458_L0_f20_le: x_Gr_Liv_Area + 2404.760001 t458_leaf0 <= 3421.76
 T458_L1_f21_le: x_Bsmt_Full_Bath + 1.020001 t458_leaf1 <= 2.02
 T458_L1_f12_le: x_BsmtFin_SF_1 + 1168.500001 t458_leaf1 <= 1984.5
 T458_R1_f20_ge: x_Gr_Liv_Area - 1017 t458_leaf1 >= 0
 T458_L2_f21_le: x_Bsmt_Full_Bath + 1.020001 t458_leaf2 <= 2.02
 T458_R2_f12_ge: x_BsmtFin_SF_1 - 816 t458_leaf2 >= 0
 T458_R3_f21_ge: x_Bsmt_Full_Bath - t458_leaf3 >= 0
 T458_L3_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t458_leaf3 <= 1984.5
 T458_L3_f1_le: x_Lot_Area + 2165.000001 t458_leaf3 <= 7000
 T458_R4_f21_ge: x_Bsmt_Full_Bath - t458_leaf4 >= 0
 T458_L4_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t458_leaf4 <= 1984.5
 T458_R4_f1_ge: x_Lot_Area >= 4835
 T458_R5_f21_ge: x_Bsmt_Full_Bath - t458_leaf5 >= 0
 T458_R5_f12_ge: x_BsmtFin_SF_1 - 1732 t458_leaf5 >= 0
 T458_L5_f36_le: x_Wood_Deck_SF + 353.900001 t458_leaf5 <= 525.9
 T458_R6_f21_ge: x_Bsmt_Full_Bath - t458_leaf6 >= 0
 T458_R6_f12_ge: x_BsmtFin_SF_1 - 1732 t458_leaf6 >= 0
 T458_R6_f36_ge: x_Wood_Deck_SF - 172 t458_leaf6 >= 0
 TREE_459_ONEHOT: t459_leaf0 + t459_leaf1 + t459_leaf2 + t459_leaf3
   + t459_leaf4 = 1
 T459_L0_f26_le: x_Kitchen_AbvGr + 1.000001 t459_leaf0 <= 3
 T459_L0_f3_le: x_Overall_Qual + 999993.000001 t459_leaf0 <= 1e+06
 T459_L0_f37_le: x_Open_Porch_SF + 336.810001 t459_leaf0 <= 488.81
 T459_L1_f26_le: x_Kitchen_AbvGr + 1.000001 t459_leaf1 <= 3
 T459_L1_f3_le: x_Overall_Qual + 999993.000001 t459_leaf1 <= 1e+06
 T459_R1_f37_ge: x_Open_Porch_SF - 152 t459_leaf1 >= 0
 T459_L2_f26_le: x_Kitchen_AbvGr + 1.000001 t459_leaf2 <= 3
 T459_R2_f3_ge: x_Overall_Qual - 7 t459_leaf2 >= 0
 T459_L2_f87_le: x_Land_Slope_Mod + 1e-06 t459_leaf2 <= 1
 T459_L3_f26_le: x_Kitchen_AbvGr + 1.000001 t459_leaf3 <= 3
 T459_R3_f3_ge: x_Overall_Qual - 7 t459_leaf3 >= 0
 T459_R3_f87_ge: x_Land_Slope_Mod - t459_leaf3 >= 0
 T459_R4_f26_ge: x_Kitchen_AbvGr - 2 t459_leaf4 >= 0
 TREE_460_ONEHOT: t460_leaf0 = 1
 TREE_461_ONEHOT: t461_leaf0 + t461_leaf1 + t461_leaf2 + t461_leaf3
   + t461_leaf4 + t461_leaf5 = 1
 T461_L0_f1_le: x_Lot_Area <= 16778.999999
 T461_L0_f4_le: x_Overall_Cond + 999996.000001 t461_leaf0 <= 1e+06
 T461_L0_f37_le: x_Open_Porch_SF + 484.810001 t461_leaf0 <= 488.81
 T461_L1_f1_le: x_Lot_Area <= 16778.999999
 T461_L1_f4_le: x_Overall_Cond + 999996.000001 t461_leaf1 <= 1e+06
 T461_R1_f37_ge: x_Open_Porch_SF - 4 t461_leaf1 >= 0
 T461_L2_f1_le: x_Lot_Area <= 16778.999999
 T461_R2_f4_ge: x_Overall_Cond - 4 t461_leaf2 >= 0
 T461_L2_f62_le: x_MS_Zoning_C_(all) + 1e-06 t461_leaf2 <= 1
 T461_L3_f1_le: x_Lot_Area <= 16778.999999
 T461_R3_f4_ge: x_Overall_Cond - 4 t461_leaf3 >= 0
 T461_R3_f62_ge: x_MS_Zoning_C_(all) - t461_leaf3 >= 0
 T461_R4_f1_ge: x_Lot_Area - 9779 t461_leaf4 >= 7000
 T461_L4_f150_le: x_Roof_Style_Hip + 1e-06 t461_leaf4 <= 1
 T461_R5_f1_ge: x_Lot_Area - 9779 t461_leaf5 >= 7000
 T461_R5_f150_ge: x_Roof_Style_Hip - t461_leaf5 >= 0
 TREE_462_ONEHOT: t462_leaf0 + t462_leaf1 = 1
 T462_L0_f0_le: x_Lot_Frontage + 35.530001 t462_leaf0 <= 175.53
 T462_R1_f0_ge: x_Lot_Frontage - 140 t462_leaf1 >= 0
 TREE_463_ONEHOT: t463_leaf0 + t463_leaf1 + t463_leaf2 + t463_leaf3
   + t463_leaf4 + t463_leaf5 + t463_leaf6 = 1
 T463_L0_f1_le: x_Lot_Area <= 21369.999999
 T463_L0_f20_le: x_Gr_Liv_Area + 2527.760001 t463_leaf0 <= 3421.76
 T463_L0_f174_le: x_Exterior_1st_VinylSd + 1e-06 t463_leaf0 <= 1
 T463_L1_f1_le: x_Lot_Area <= 21369.999999
 T463_L1_f20_le: x_Gr_Liv_Area + 2527.760001 t463_leaf1 <= 3421.76
 T463_R1_f174_ge: x_Exterior_1st_VinylSd - t463_leaf1 >= 0
 T463_L2_f1_le: x_Lot_Area <= 21369.999999
 T463_R2_f20_ge: x_Gr_Liv_Area - 894 t463_leaf2 >= 0
 T463_L2_f7_le: x_Mas_Vnr_Area + 458.070001 t463_leaf2 <= 931.07
 T463_L3_f1_le: x_Lot_Area <= 21369.999999
 T463_R3_f20_ge: x_Gr_Liv_Area - 894 t463_leaf3 >= 0
 T463_R3_f7_ge: x_Mas_Vnr_Area - 473 t463_leaf3 >= 0
 T463_R4_f1_ge: x_Lot_Area - 14370 t463_leaf4 >= 7000
 T463_L4_f148_le: x_Roof_Style_Gable + 1e-06 t463_leaf4 <= 1
 T463_L4_f36_le: x_Wood_Deck_SF + 309.900001 t463_leaf4 <= 525.9
 T463_R5_f1_ge: x_Lot_Area - 14370 t463_leaf5 >= 7000
 T463_L5_f148_le: x_Roof_Style_Gable + 1e-06 t463_leaf5 <= 1
 T463_R5_f36_ge: x_Wood_Deck_SF - 216 t463_leaf5 >= 0
 T463_R6_f1_ge: x_Lot_Area - 14370 t463_leaf6 >= 7000
 T463_R6_f148_ge: x_Roof_Style_Gable - t463_leaf6 >= 0
 TREE_464_ONEHOT: t464_leaf0 + t464_leaf1 + t464_leaf2 = 1
 T464_L0_f5_le: x_Year_Built + 997995.000001 t464_leaf0 <= 1e+06
 T464_L0_f0_le: x_Lot_Frontage + 66.530001 t464_leaf0 <= 175.53
 T464_L1_f5_le: x_Year_Built + 997995.000001 t464_leaf1 <= 1e+06
 T464_R1_f0_ge: x_Lot_Frontage - 109 t464_leaf1 >= 0
 T464_R2_f5_ge: x_Year_Built - 2005 t464_leaf2 >= 0
 TREE_465_ONEHOT: t465_leaf0 + t465_leaf1 + t465_leaf2 + t465_leaf3
   + t465_leaf4 = 1
 T465_L0_f25_le: x_Bedroom_AbvGr + 4.000001 t465_leaf0 <= 7
 T465_L0_f44_le: x_Yr_Sold + 997990.000001 t465_leaf0 <= 1e+06
 T465_L0_f1_le: x_Lot_Area <= 8849.999999
 T465_L1_f25_le: x_Bedroom_AbvGr + 4.000001 t465_leaf1 <= 7
 T465_L1_f44_le: x_Yr_Sold + 997990.000001 t465_leaf1 <= 1e+06
 T465_R1_f1_ge: x_Lot_Area - 1850 t465_leaf1 >= 7000
 T465_L2_f25_le: x_Bedroom_AbvGr + 4.000001 t465_leaf2 <= 7
 T465_R2_f44_ge: x_Yr_Sold - 2010 t465_leaf2 >= 0
 T465_L2_f14_le: x_Bsmt_Unf_SF + 1619.680001 t465_leaf2 <= 1812.68
 T465_L3_f25_le: x_Bedroom_AbvGr + 4.000001 t465_leaf3 <= 7
 T465_R3_f44_ge: x_Yr_Sold - 2010 t465_leaf3 >= 0
 T465_R3_f14_ge: x_Bsmt_Unf_SF - 193 t465_leaf3 >= 0
 T465_R4_f25_ge: x_Bedroom_AbvGr - 3 t465_leaf4 >= 0
 TREE_466_ONEHOT: t466_leaf0 + t466_leaf1 + t466_leaf2 + t466_leaf3 = 1
 T466_L0_f31_le: x_Garage_Yr_Blt + 997992.000001 t466_leaf0 <= 1e+06
 T466_L0_f12_le: x_BsmtFin_SF_1 + 411.5000010000001 t466_leaf0 <= 1984.5
 T466_L0_f31_le: x_Garage_Yr_Blt + 998075.000001 t466_leaf0 <= 1e+06
 T466_L1_f31_le: x_Garage_Yr_Blt + 997992.000001 t466_leaf1 <= 1e+06
 T466_L1_f12_le: x_BsmtFin_SF_1 + 411.5000010000001 t466_leaf1 <= 1984.5
 T466_R1_f31_ge: x_Garage_Yr_Blt - 1925 t466_leaf1 >= 0
 T466_L2_f31_le: x_Garage_Yr_Blt + 997992.000001 t466_leaf2 <= 1e+06
 T466_R2_f12_ge: x_BsmtFin_SF_1 - 1573 t466_leaf2 >= 0
 T466_R3_f31_ge: x_Garage_Yr_Blt - 2008 t466_leaf3 >= 0
 TREE_467_ONEHOT: t467_leaf0 = 1
 TREE_468_ONEHOT: t468_leaf0 + t468_leaf1 = 1
 T468_L0_f7_le: x_Mas_Vnr_Area + 278.070001 t468_leaf0 <= 931.07
 T468_R1_f7_ge: x_Mas_Vnr_Area - 653 t468_leaf1 >= 0
 TREE_469_ONEHOT: t469_leaf0 + t469_leaf1 + t469_leaf2 + t469_leaf3
   + t469_leaf4 = 1
 T469_L0_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t469_leaf0 <= 1
 T469_L0_f0_le: x_Lot_Frontage + 58.530001 t469_leaf0 <= 175.53
 T469_L0_f21_le: x_Bsmt_Full_Bath + 0.020001 t469_leaf0 <= 2.02
 T469_L1_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t469_leaf1 <= 1
 T469_L1_f0_le: x_Lot_Frontage + 58.530001 t469_leaf1 <= 175.53
 T469_R1_f21_ge: x_Bsmt_Full_Bath - 2 t469_leaf1 >= 0
 T469_L2_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t469_leaf2 <= 1
 T469_R2_f0_ge: x_Lot_Frontage - 117 t469_leaf2 >= 0
 T469_L2_f15_le: x_Total_Bsmt_SF + 1032.480001 t469_leaf2 <= 2514.48
 T469_L3_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t469_leaf3 <= 1
 T469_R3_f0_ge: x_Lot_Frontage - 117 t469_leaf3 >= 0
 T469_R3_f15_ge: x_Total_Bsmt_SF - 1482 t469_leaf3 >= 0
 T469_R4_f216_ge: x_BsmtFin_Type_1_Rec - t469_leaf4 >= 0
 TREE_470_ONEHOT: t470_leaf0 + t470_leaf1 + t470_leaf2 + t470_leaf3 = 1
 T470_L0_f40_le: x_Screen_Porch + 16.27000099999998 t470_leaf0 <= 268.27
 T470_L0_f5_le: x_Year_Built + 998068.000001 t470_leaf0 <= 1e+06
 T470_L0_f16_le: x_Heating_QC + 1.000001 t470_leaf0 <= 4
 T470_L1_f40_le: x_Screen_Porch + 16.27000099999998 t470_leaf1 <= 268.27
 T470_L1_f5_le: x_Year_Built + 998068.000001 t470_leaf1 <= 1e+06
 T470_R1_f16_ge: x_Heating_QC - t470_leaf1 >= 2
 T470_L2_f40_le: x_Screen_Porch + 16.27000099999998 t470_leaf2 <= 268.27
 T470_R2_f5_ge: x_Year_Built - 1932 t470_leaf2 >= 0
 T470_R3_f40_ge: x_Screen_Porch - 252 t470_leaf3 >= 0
 TREE_471_ONEHOT: t471_leaf0 + t471_leaf1 + t471_leaf2 + t471_leaf3 = 1
 T471_L0_f96_le: x_Neighborhood_Edwards + 1e-06 t471_leaf0 <= 1
 T471_R1_f96_ge: x_Neighborhood_Edwards - t471_leaf1 >= 0
 T471_L1_f17_le: x_1st_Flr_SF + 1548.900001 t471_leaf1 <= 2534.9
 T471_R2_f96_ge: x_Neighborhood_Edwards - t471_leaf2 >= 0
 T471_R2_f17_ge: x_1st_Flr_SF - 986 t471_leaf2 >= 0
 T471_L2_f174_le: x_Exterior_1st_VinylSd + 1e-06 t471_leaf2 <= 1
 T471_R3_f96_ge: x_Neighborhood_Edwards - t471_leaf3 >= 0
 T471_R3_f17_ge: x_1st_Flr_SF - 986 t471_leaf3 >= 0
 T471_R3_f174_ge: x_Exterior_1st_VinylSd - t471_leaf3 >= 0
 TREE_472_ONEHOT: t472_leaf0 + t472_leaf1 + t472_leaf2 + t472_leaf3 = 1
 T472_L0_f206_le: x_Bsmt_Exposure_Av + 1e-06 t472_leaf0 <= 1
 T472_L0_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t472_leaf0 <= 1
 T472_L0_f18_le: x_2nd_Flr_SF + 760.540001 t472_leaf0 <= 1641.54
 T472_L1_f206_le: x_Bsmt_Exposure_Av + 1e-06 t472_leaf1 <= 1
 T472_L1_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t472_leaf1 <= 1
 T472_R1_f18_ge: x_2nd_Flr_SF - 881 t472_leaf1 >= 0
 T472_L2_f206_le: x_Bsmt_Exposure_Av + 1e-06 t472_leaf2 <= 1
 T472_R2_f218_ge: x_BsmtFin_Type_2_ALQ - t472_leaf2 >= 0
 T472_R3_f206_ge: x_Bsmt_Exposure_Av - t472_leaf3 >= 0
 TREE_473_ONEHOT: t473_leaf0 + t473_leaf1 + t473_leaf2 + t473_leaf3 = 1
 T473_L0_f62_le: x_MS_Zoning_C_(all) + 1e-06 t473_leaf0 <= 1
 T473_L0_f199_le: x_Mas_Vnr_Type_Stone + 1e-06 t473_leaf0 <= 1
 T473_L0_f272_le: x_Mo_Sold_2 + 1e-06 t473_leaf0 <= 1
 T473_L1_f62_le: x_MS_Zoning_C_(all) + 1e-06 t473_leaf1 <= 1
 T473_L1_f199_le: x_Mas_Vnr_Type_Stone + 1e-06 t473_leaf1 <= 1
 T473_R1_f272_ge: x_Mo_Sold_2 - t473_leaf1 >= 0
 T473_L2_f62_le: x_MS_Zoning_C_(all) + 1e-06 t473_leaf2 <= 1
 T473_R2_f199_ge: x_Mas_Vnr_Type_Stone - t473_leaf2 >= 0
 T473_R3_f62_ge: x_MS_Zoning_C_(all) - t473_leaf3 >= 0
 TREE_474_ONEHOT: t474_leaf0 + t474_leaf1 + t474_leaf2 = 1
 T474_L0_f29_le: x_Fireplaces + 1e-06 t474_leaf0 <= 3
 T474_L0_f15_le: x_Total_Bsmt_SF + 588.4800010000001 t474_leaf0 <= 2514.48
 T474_L1_f29_le: x_Fireplaces + 1e-06 t474_leaf1 <= 3
 T474_R1_f15_ge: x_Total_Bsmt_SF - 1926 t474_leaf1 >= 0
 T474_R2_f29_ge: x_Fireplaces - 3 t474_leaf2 >= 0
 TREE_475_ONEHOT: t475_leaf0 + t475_leaf1 + t475_leaf2 + t475_leaf3
   + t475_leaf4 = 1
 T475_L0_f3_le: x_Overall_Qual + 999994.000001 t475_leaf0 <= 1e+06
 T475_L0_f100_le: x_Neighborhood_IDOTRR + 1e-06 t475_leaf0 <= 1
 T475_L0_f5_le: x_Year_Built + 998068.000001 t475_leaf0 <= 1e+06
 T475_L1_f3_le: x_Overall_Qual + 999994.000001 t475_leaf1 <= 1e+06
 T475_L1_f100_le: x_Neighborhood_IDOTRR + 1e-06 t475_leaf1 <= 1
 T475_R1_f5_ge: x_Year_Built - 1932 t475_leaf1 >= 0
 T475_L2_f3_le: x_Overall_Qual + 999994.000001 t475_leaf2 <= 1e+06
 T475_R2_f100_ge: x_Neighborhood_IDOTRR - t475_leaf2 >= 0
 T475_R3_f3_ge: x_Overall_Qual - 6 t475_leaf3 >= 0
 T475_L3_f33_le: x_Garage_Area + 877.750001 t475_leaf3 <= 1163.75
 T475_R4_f3_ge: x_Overall_Qual - 6 t475_leaf4 >= 0
 T475_R4_f33_ge: x_Garage_Area - 286 t475_leaf4 >= 0
 TREE_476_ONEHOT: t476_leaf0 = 1
 TREE_477_ONEHOT: t477_leaf0 + t477_leaf1 + t477_leaf2 + t477_leaf3
   + t477_leaf4 = 1
 T477_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t477_leaf0 <= 1
 T477_L0_f14_le: x_Bsmt_Unf_SF + 1240.680001 t477_leaf0 <= 1812.68
 T477_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t477_leaf1 <= 1
 T477_R1_f14_ge: x_Bsmt_Unf_SF - 572 t477_leaf1 >= 0
 T477_L1_f14_le: x_Bsmt_Unf_SF + 1201.680001 t477_leaf1 <= 1812.68
 T477_L2_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t477_leaf2 <= 1
 T477_R2_f14_ge: x_Bsmt_Unf_SF - 572 t477_leaf2 >= 0
 T477_R2_f14_ge: x_Bsmt_Unf_SF - 611 t477_leaf2 >= 0
 T477_R3_f177_ge: x_Exterior_2nd_AsbShng - t477_leaf3 >= 0
 T477_L3_f0_le: x_Lot_Frontage + 106.530001 t477_leaf3 <= 175.53
 T477_R4_f177_ge: x_Exterior_2nd_AsbShng - t477_leaf4 >= 0
 T477_R4_f0_ge: x_Lot_Frontage - 69 t477_leaf4 >= 0
 TREE_478_ONEHOT: t478_leaf0 + t478_leaf1 + t478_leaf2 + t478_leaf3 = 1
 T478_L0_f38_le: x_Enclosed_Porch + 46.750001 t478_leaf0 <= 292.75
 T478_L0_f38_le: x_Enclosed_Porch + 56.750001 t478_leaf0 <= 292.75
 T478_L0_f38_le: x_Enclosed_Porch + 100.750001 t478_leaf0 <= 292.75
 T478_L1_f38_le: x_Enclosed_Porch + 46.750001 t478_leaf1 <= 292.75
 T478_L1_f38_le: x_Enclosed_Porch + 56.750001 t478_leaf1 <= 292.75
 T478_R1_f38_ge: x_Enclosed_Porch - 192 t478_leaf1 >= 0
 T478_L2_f38_le: x_Enclosed_Porch + 46.750001 t478_leaf2 <= 292.75
 T478_R2_f38_ge: x_Enclosed_Porch - 236 t478_leaf2 >= 0
 T478_R3_f38_ge: x_Enclosed_Porch - 246 t478_leaf3 >= 0
 TREE_479_ONEHOT: t479_leaf0 + t479_leaf1 + t479_leaf2 + t479_leaf3 = 1
 T479_L0_f96_le: x_Neighborhood_Edwards + 1e-06 t479_leaf0 <= 1
 T479_R1_f96_ge: x_Neighborhood_Edwards - t479_leaf1 >= 0
 T479_L1_f33_le: x_Garage_Area + 553.750001 t479_leaf1 <= 1163.75
 T479_L1_f6_le: x_Year_Remod/Add + 997995.000001 t479_leaf1 <= 1e+06
 T479_R2_f96_ge: x_Neighborhood_Edwards - t479_leaf2 >= 0
 T479_L2_f33_le: x_Garage_Area + 553.750001 t479_leaf2 <= 1163.75
 T479_R2_f6_ge: x_Year_Remod/Add - 2005 t479_leaf2 >= 0
 T479_R3_f96_ge: x_Neighborhood_Edwards - t479_leaf3 >= 0
 T479_R3_f33_ge: x_Garage_Area - 610 t479_leaf3 >= 0
 TREE_480_ONEHOT: t480_leaf0 + t480_leaf1 = 1
 T480_L0_f38_le: x_Enclosed_Porch + 4.750000999999997 t480_leaf0 <= 292.75
 T480_R1_f38_ge: x_Enclosed_Porch - 288 t480_leaf1 >= 0
 TREE_481_ONEHOT: t481_leaf0 + t481_leaf1 + t481_leaf2 + t481_leaf3 = 1
 T481_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t481_leaf0 <= 1
 T481_L0_f17_le: x_1st_Flr_SF + 1738.900001 t481_leaf0 <= 2534.9
 T481_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t481_leaf1 <= 1
 T481_R1_f17_ge: x_1st_Flr_SF - 796 t481_leaf1 >= 0
 T481_R2_f245_ge: x_FunctioNo_aplical_Typ - t481_leaf2 >= 0
 T481_L2_f14_le: x_Bsmt_Unf_SF + 395.6800010000002 t481_leaf2 <= 1812.68
 T481_R3_f245_ge: x_FunctioNo_aplical_Typ - t481_leaf3 >= 0
 T481_R3_f14_ge: x_Bsmt_Unf_SF - 1417 t481_leaf3 >= 0
 TREE_482_ONEHOT: t482_leaf0 + t482_leaf1 + t482_leaf2 = 1
 T482_L0_f0_le: x_Lot_Frontage + 54.530001 t482_leaf0 <= 175.53
 T482_R1_f0_ge: x_Lot_Frontage - 121 t482_leaf1 >= 0
 T482_L1_f36_le: x_Wood_Deck_SF + 421.900001 t482_leaf1 <= 525.9
 T482_R2_f0_ge: x_Lot_Frontage - 121 t482_leaf2 >= 0
 T482_R2_f36_ge: x_Wood_Deck_SF - 104 t482_leaf2 >= 0
 TREE_483_ONEHOT: t483_leaf0 = 1
 TREE_484_ONEHOT: t484_leaf0 + t484_leaf1 = 1
 T484_L0_f38_le: x_Enclosed_Porch + 27.750001 t484_leaf0 <= 292.75
 T484_R1_f38_ge: x_Enclosed_Porch - 265 t484_leaf1 >= 0
 TREE_485_ONEHOT: t485_leaf0 = 1
 TREE_486_ONEHOT: t486_leaf0 + t486_leaf1 = 1
 T486_L0_f20_le: x_Gr_Liv_Area + 2445.760001 t486_leaf0 <= 3421.76
 T486_R1_f20_ge: x_Gr_Liv_Area - 976 t486_leaf1 >= 0
 TREE_487_ONEHOT: t487_leaf0 + t487_leaf1 = 1
 T487_L0_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t487_leaf0 <= 1
 T487_R1_f218_ge: x_BsmtFin_Type_2_ALQ - t487_leaf1 >= 0
 TREE_488_ONEHOT: t488_leaf0 + t488_leaf1 + t488_leaf2 + t488_leaf3 = 1
 T488_L0_f231_le: x_Central_Air_N + 1e-06 t488_leaf0 <= 1
 T488_R1_f231_ge: x_Central_Air_N - t488_leaf1 >= 0
 T488_L1_f31_le: x_Garage_Yr_Blt + 998049.000001 t488_leaf1 <= 1e+06
 T488_L1_f18_le: x_2nd_Flr_SF + 829.540001 t488_leaf1 <= 1641.54
 T488_R2_f231_ge: x_Central_Air_N - t488_leaf2 >= 0
 T488_L2_f31_le: x_Garage_Yr_Blt + 998049.000001 t488_leaf2 <= 1e+06
 T488_R2_f18_ge: x_2nd_Flr_SF - 812 t488_leaf2 >= 0
 T488_R3_f231_ge: x_Central_Air_N - t488_leaf3 >= 0
 T488_R3_f31_ge: x_Garage_Yr_Blt - 1951 t488_leaf3 >= 0
 TREE_489_ONEHOT: t489_leaf0 + t489_leaf1 + t489_leaf2 + t489_leaf3 = 1
 T489_L0_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t489_leaf0 <= 2514.48
 T489_L0_f14_le: x_Bsmt_Unf_SF + 17.68000100000017 t489_leaf0 <= 1812.68
 T489_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t489_leaf0 <= 1
 T489_L1_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t489_leaf1 <= 2514.48
 T489_L1_f14_le: x_Bsmt_Unf_SF + 17.68000100000017 t489_leaf1 <= 1812.68
 T489_R1_f177_ge: x_Exterior_2nd_AsbShng - t489_leaf1 >= 0
 T489_L2_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t489_leaf2 <= 2514.48
 T489_R2_f14_ge: x_Bsmt_Unf_SF - 1795 t489_leaf2 >= 0
 T489_R3_f15_ge: x_Total_Bsmt_SF - 2136 t489_leaf3 >= 0
 TREE_490_ONEHOT: t490_leaf0 + t490_leaf1 = 1
 T490_L0_f37_le: x_Open_Porch_SF + 326.810001 t490_leaf0 <= 488.81
 T490_R1_f37_ge: x_Open_Porch_SF - 162 t490_leaf1 >= 0
 TREE_491_ONEHOT: t491_leaf0 + t491_leaf1 + t491_leaf2 + t491_leaf3
   + t491_leaf4 + t491_leaf5 + t491_leaf6 = 1
 T491_L0_f14_le: x_Bsmt_Unf_SF + 1558.680001 t491_leaf0 <= 1812.68
 T491_L0_f21_le: x_Bsmt_Full_Bath + 0.020001 t491_leaf0 <= 2.02
 T491_L0_f1_le: x_Lot_Area <= 14462.999999
 T491_L1_f14_le: x_Bsmt_Unf_SF + 1558.680001 t491_leaf1 <= 1812.68
 T491_L1_f21_le: x_Bsmt_Full_Bath + 0.020001 t491_leaf1 <= 2.02
 T491_R1_f1_ge: x_Lot_Area - 7463 t491_leaf1 >= 7000
 T491_L2_f14_le: x_Bsmt_Unf_SF + 1558.680001 t491_leaf2 <= 1812.68
 T491_R2_f21_ge: x_Bsmt_Full_Bath - 2 t491_leaf2 >= 0
 T491_R3_f14_ge: x_Bsmt_Unf_SF - 254 t491_leaf3 >= 0
 T491_L3_f96_le: x_Neighborhood_Edwards + 1e-06 t491_leaf3 <= 1
 T491_L3_f7_le: x_Mas_Vnr_Area + 258.070001 t491_leaf3 <= 931.07
 T491_R4_f14_ge: x_Bsmt_Unf_SF - 254 t491_leaf4 >= 0
 T491_L4_f96_le: x_Neighborhood_Edwards + 1e-06 t491_leaf4 <= 1
 T491_R4_f7_ge: x_Mas_Vnr_Area - 673 t491_leaf4 >= 0
 T491_R5_f14_ge: x_Bsmt_Unf_SF - 254 t491_leaf5 >= 0
 T491_R5_f96_ge: x_Neighborhood_Edwards - t491_leaf5 >= 0
 T491_L5_f4_le: x_Overall_Cond + 999994.000001 t491_leaf5 <= 1e+06
 T491_R6_f14_ge: x_Bsmt_Unf_SF - 254 t491_leaf6 >= 0
 T491_R6_f96_ge: x_Neighborhood_Edwards - t491_leaf6 >= 0
 T491_R6_f4_ge: x_Overall_Cond - 6 t491_leaf6 >= 0
 TREE_492_ONEHOT: t492_leaf0 + t492_leaf1 + t492_leaf2 = 1
 T492_L0_f3_le: x_Overall_Qual + 999992.000001 t492_leaf0 <= 1e+06
 T492_L0_f113_le: x_Neighborhood_Somerst + 1e-06 t492_leaf0 <= 1
 T492_L1_f3_le: x_Overall_Qual + 999992.000001 t492_leaf1 <= 1e+06
 T492_R1_f113_ge: x_Neighborhood_Somerst - t492_leaf1 >= 0
 T492_R2_f3_ge: x_Overall_Qual - 8 t492_leaf2 >= 0
 TREE_493_ONEHOT: t493_leaf0 + t493_leaf1 + t493_leaf2 + t493_leaf3
   + t493_leaf4 + t493_leaf5 = 1
 T493_L0_f33_le: x_Garage_Area + 910.750001 t493_leaf0 <= 1163.75
 T493_L0_f1_le: x_Lot_Area <= 10181.999999
 T493_L1_f33_le: x_Garage_Area + 910.750001 t493_leaf1 <= 1163.75
 T493_R1_f1_ge: x_Lot_Area - 3182 t493_leaf1 >= 7000
 T493_R2_f33_ge: x_Garage_Area - 253 t493_leaf2 >= 0
 T493_L2_f21_le: x_Bsmt_Full_Bath + 1.020001 t493_leaf2 <= 2.02
 T493_L2_f20_le: x_Gr_Liv_Area + 2139.760001000001 t493_leaf2 <= 3421.76
 T493_R3_f33_ge: x_Garage_Area - 253 t493_leaf3 >= 0
 T493_L3_f21_le: x_Bsmt_Full_Bath + 1.020001 t493_leaf3 <= 2.02
 T493_R3_f20_ge: x_Gr_Liv_Area - 1282 t493_leaf3 >= 0
 T493_R4_f33_ge: x_Garage_Area - 253 t493_leaf4 >= 0
 T493_R4_f21_ge: x_Bsmt_Full_Bath - t493_leaf4 >= 0
 T493_L4_f20_le: x_Gr_Liv_Area + 335.7600010000001 t493_leaf4 <= 3421.76
 T493_R5_f33_ge: x_Garage_Area - 253 t493_leaf5 >= 0
 T493_R5_f21_ge: x_Bsmt_Full_Bath - t493_leaf5 >= 0
 T493_R5_f20_ge: x_Gr_Liv_Area - 3086 t493_leaf5 >= 0
 TREE_494_ONEHOT: t494_leaf0 + t494_leaf1 + t494_leaf2 + t494_leaf3
   + t494_leaf4 = 1
 T494_L0_f18_le: x_2nd_Flr_SF + 214.5400010000001 t494_leaf0 <= 1641.54
 T494_L0_f3_le: x_Overall_Qual + 999993.000001 t494_leaf0 <= 1e+06
 T494_L0_f18_le: x_2nd_Flr_SF + 1176.540001 t494_leaf0 <= 1641.54
 T494_L1_f18_le: x_2nd_Flr_SF + 214.5400010000001 t494_leaf1 <= 1641.54
 T494_L1_f3_le: x_Overall_Qual + 999993.000001 t494_leaf1 <= 1e+06
 T494_R1_f18_ge: x_2nd_Flr_SF - 465 t494_leaf1 >= 0
 T494_L2_f18_le: x_2nd_Flr_SF + 214.5400010000001 t494_leaf2 <= 1641.54
 T494_R2_f3_ge: x_Overall_Qual - 7 t494_leaf2 >= 0
 T494_L2_f1_le: x_Lot_Area <= 21369.999999
 T494_L3_f18_le: x_2nd_Flr_SF + 214.5400010000001 t494_leaf3 <= 1641.54
 T494_R3_f3_ge: x_Overall_Qual - 7 t494_leaf3 >= 0
 T494_R3_f1_ge: x_Lot_Area - 14370 t494_leaf3 >= 7000
 T494_R4_f18_ge: x_2nd_Flr_SF - 1427 t494_leaf4 >= 0
 TREE_495_ONEHOT: t495_leaf0 + t495_leaf1 = 1
 T495_L0_f3_le: x_Overall_Qual + 999997.000001 t495_leaf0 <= 1e+06
 T495_R1_f3_ge: x_Overall_Qual - 3 t495_leaf1 >= 0
 TREE_496_ONEHOT: t496_leaf0 + t496_leaf1 + t496_leaf2 + t496_leaf3 = 1
 T496_L0_f17_le: x_1st_Flr_SF + 19.90000099999997 t496_leaf0 <= 2534.9
 T496_L0_f9_le: x_Exter_Cond + 2.000001 t496_leaf0 <= 4
 T496_L1_f17_le: x_1st_Flr_SF + 19.90000099999997 t496_leaf1 <= 2534.9
 T496_R1_f9_ge: x_Exter_Cond - 3 t496_leaf1 >= -1
 T496_L1_f26_le: x_Kitchen_AbvGr + 1.000001 t496_leaf1 <= 3
 T496_L2_f17_le: x_1st_Flr_SF + 19.90000099999997 t496_leaf2 <= 2534.9
 T496_R2_f9_ge: x_Exter_Cond - 3 t496_leaf2 >= -1
 T496_R2_f26_ge: x_Kitchen_AbvGr - 2 t496_leaf2 >= 0
 T496_R3_f17_ge: x_1st_Flr_SF - 2515 t496_leaf3 >= 0
 TREE_497_ONEHOT: t497_leaf0 + t497_leaf1 + t497_leaf2 + t497_leaf3 = 1
 T497_L0_f33_le: x_Garage_Area + 938.750001 t497_leaf0 <= 1163.75
 T497_R1_f33_ge: x_Garage_Area - 225 t497_leaf1 >= 0
 T497_L1_f158_le: x_Roof_Matl_Tar&Grv + 1e-06 t497_leaf1 <= 1
 T497_L1_f3_le: x_Overall_Qual + 999992.000001 t497_leaf1 <= 1e+06
 T497_R2_f33_ge: x_Garage_Area - 225 t497_leaf2 >= 0
 T497_L2_f158_le: x_Roof_Matl_Tar&Grv + 1e-06 t497_leaf2 <= 1
 T497_R2_f3_ge: x_Overall_Qual - 8 t497_leaf2 >= 0
 T497_R3_f33_ge: x_Garage_Area - 225 t497_leaf3 >= 0
 T497_R3_f158_ge: x_Roof_Matl_Tar&Grv - t497_leaf3 >= 0
 TREE_498_ONEHOT: t498_leaf0 + t498_leaf1 + t498_leaf2 = 1
 T498_L0_f5_le: x_Year_Built + 997995.000001 t498_leaf0 <= 1e+06
 T498_L0_f72_le: x_Alley_Pave + 1e-06 t498_leaf0 <= 1
 T498_L1_f5_le: x_Year_Built + 997995.000001 t498_leaf1 <= 1e+06
 T498_R1_f72_ge: x_Alley_Pave - t498_leaf1 >= 0
 T498_R2_f5_ge: x_Year_Built - 2005 t498_leaf2 >= 0
 TREE_499_ONEHOT: t499_leaf0 + t499_leaf1 + t499_leaf2 + t499_leaf3 = 1
 T499_L0_f36_le: x_Wood_Deck_SF + 299.900001 t499_leaf0 <= 525.9
 T499_L0_f148_le: x_Roof_Style_Gable + 1e-06 t499_leaf0 <= 1
 T499_L1_f36_le: x_Wood_Deck_SF + 299.900001 t499_leaf1 <= 525.9
 T499_R1_f148_ge: x_Roof_Style_Gable - t499_leaf1 >= 0
 T499_L1_f111_le: x_Neighborhood_Sawyer + 1e-06 t499_leaf1 <= 1
 T499_L2_f36_le: x_Wood_Deck_SF + 299.900001 t499_leaf2 <= 525.9
 T499_R2_f148_ge: x_Roof_Style_Gable - t499_leaf2 >= 0
 T499_R2_f111_ge: x_Neighborhood_Sawyer - t499_leaf2 >= 0
 T499_R3_f36_ge: x_Wood_Deck_SF - 226 t499_leaf3 >= 0
 TREE_500_ONEHOT: t500_leaf0 + t500_leaf1 + t500_leaf2 + t500_leaf3
   + t500_leaf4 = 1
 T500_L0_f203_le: x_Foundation_Slab + 1e-06 t500_leaf0 <= 1
 T500_L0_f31_le: x_Garage_Yr_Blt + 998051.000001 t500_leaf0 <= 1e+06
 T500_L0_f14_le: x_Bsmt_Unf_SF + 1148.680001 t500_leaf0 <= 1812.68
 T500_L1_f203_le: x_Foundation_Slab + 1e-06 t500_leaf1 <= 1
 T500_L1_f31_le: x_Garage_Yr_Blt + 998051.000001 t500_leaf1 <= 1e+06
 T500_R1_f14_ge: x_Bsmt_Unf_SF - 664 t500_leaf1 >= 0
 T500_L2_f203_le: x_Foundation_Slab + 1e-06 t500_leaf2 <= 1
 T500_R2_f31_ge: x_Garage_Yr_Blt - 1949 t500_leaf2 >= 0
 T500_R3_f203_ge: x_Foundation_Slab - t500_leaf3 >= 0
 T500_L3_f0_le: x_Lot_Frontage + 109.530001 t500_leaf3 <= 175.53
 T500_R4_f203_ge: x_Foundation_Slab - t500_leaf4 >= 0
 T500_R4_f0_ge: x_Lot_Frontage - 66 t500_leaf4 >= 0
 TREE_501_ONEHOT: t501_leaf0 + t501_leaf1 + t501_leaf2 + t501_leaf3 = 1
 T501_L0_f3_le: x_Overall_Qual + 999997.000001 t501_leaf0 <= 1e+06
 T501_L0_f33_le: x_Garage_Area + 877.750001 t501_leaf0 <= 1163.75
 T501_L1_f3_le: x_Overall_Qual + 999997.000001 t501_leaf1 <= 1e+06
 T501_R1_f33_ge: x_Garage_Area - 286 t501_leaf1 >= 0
 T501_R2_f3_ge: x_Overall_Qual - 3 t501_leaf2 >= 0
 T501_L2_f33_le: x_Garage_Area + 9.750001000000111 t501_leaf2 <= 1163.75
 T501_R3_f3_ge: x_Overall_Qual - 3 t501_leaf3 >= 0
 T501_R3_f33_ge: x_Garage_Area - 1154 t501_leaf3 >= 0
 TREE_502_ONEHOT: t502_leaf0 + t502_leaf1 + t502_leaf2 = 1
 T502_L0_f271_le: x_Mo_Sold_1 + 1e-06 t502_leaf0 <= 1
 T502_R1_f271_ge: x_Mo_Sold_1 - t502_leaf1 >= 0
 T502_L1_f5_le: x_Year_Built + 998044.000001 t502_leaf1 <= 1e+06
 T502_R2_f271_ge: x_Mo_Sold_1 - t502_leaf2 >= 0
 T502_R2_f5_ge: x_Year_Built - 1956 t502_leaf2 >= 0
 TREE_503_ONEHOT: t503_leaf0 + t503_leaf1 + t503_leaf2 + t503_leaf3
   + t503_leaf4 = 1
 T503_L0_f4_le: x_Overall_Cond + 999993.000001 t503_leaf0 <= 1e+06
 T503_L0_f5_le: x_Year_Built + 998068.000001 t503_leaf0 <= 1e+06
 T503_L0_f0_le: x_Lot_Frontage + 104.530001 t503_leaf0 <= 175.53
 T503_L1_f4_le: x_Overall_Cond + 999993.000001 t503_leaf1 <= 1e+06
 T503_L1_f5_le: x_Year_Built + 998068.000001 t503_leaf1 <= 1e+06
 T503_R1_f0_ge: x_Lot_Frontage - 71 t503_leaf1 >= 0
 T503_L2_f4_le: x_Overall_Cond + 999993.000001 t503_leaf2 <= 1e+06
 T503_R2_f5_ge: x_Year_Built - 1932 t503_leaf2 >= 0
 T503_L2_f3_le: x_Overall_Qual + 999996.000001 t503_leaf2 <= 1e+06
 T503_L3_f4_le: x_Overall_Cond + 999993.000001 t503_leaf3 <= 1e+06
 T503_R3_f5_ge: x_Year_Built - 1932 t503_leaf3 >= 0
 T503_R3_f3_ge: x_Overall_Qual - 4 t503_leaf3 >= 0
 T503_R4_f4_ge: x_Overall_Cond - 7 t503_leaf4 >= 0
 TREE_504_ONEHOT: t504_leaf0 + t504_leaf1 + t504_leaf2 = 1
 T504_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t504_leaf0 <= 1
 T504_R1_f161_ge: x_Exterior_1st_AsbShng - t504_leaf1 >= 0
 T504_L1_f20_le: x_Gr_Liv_Area + 2538.760001 t504_leaf1 <= 3421.76
 T504_R2_f161_ge: x_Exterior_1st_AsbShng - t504_leaf2 >= 0
 T504_R2_f20_ge: x_Gr_Liv_Area - 883 t504_leaf2 >= 0
 TREE_505_ONEHOT: t505_leaf0 + t505_leaf1 = 1
 T505_L0_f4_le: x_Overall_Cond + 999997.000001 t505_leaf0 <= 1e+06
 T505_R1_f4_ge: x_Overall_Cond - 3 t505_leaf1 >= 0
 TREE_506_ONEHOT: t506_leaf0 + t506_leaf1 + t506_leaf2 = 1
 T506_L0_f8_le: x_Exter_Qual + 2.000001 t506_leaf0 <= 4
 T506_R1_f8_ge: x_Exter_Qual >= 2
 T506_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t506_leaf1 <= 1
 T506_R2_f8_ge: x_Exter_Qual >= 2
 T506_R2_f177_ge: x_Exterior_2nd_AsbShng - t506_leaf2 >= 0
 TREE_507_ONEHOT: t507_leaf0 + t507_leaf1 = 1
 T507_L0_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t507_leaf0 <= 1
 T507_R1_f218_ge: x_BsmtFin_Type_2_ALQ - t507_leaf1 >= 0
 TREE_508_ONEHOT: t508_leaf0 + t508_leaf1 + t508_leaf2 = 1
 T508_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t508_leaf0 <= 1
 T508_R1_f161_ge: x_Exterior_1st_AsbShng - t508_leaf1 >= 0
 T508_L1_f33_le: x_Garage_Area + 784.750001 t508_leaf1 <= 1163.75
 T508_R2_f161_ge: x_Exterior_1st_AsbShng - t508_leaf2 >= 0
 T508_R2_f33_ge: x_Garage_Area - 379 t508_leaf2 >= 0
 TREE_509_ONEHOT: t509_leaf0 + t509_leaf1 + t509_leaf2 + t509_leaf3 = 1
 T509_L0_f18_le: x_2nd_Flr_SF + 807.540001 t509_leaf0 <= 1641.54
 T509_L0_f3_le: x_Overall_Qual + 999995.000001 t509_leaf0 <= 1e+06
 T509_L0_f1_le: x_Lot_Area + 2730.000001 t509_leaf0 <= 7000
 T509_L1_f18_le: x_2nd_Flr_SF + 807.540001 t509_leaf1 <= 1641.54
 T509_L1_f3_le: x_Overall_Qual + 999995.000001 t509_leaf1 <= 1e+06
 T509_R1_f1_ge: x_Lot_Area >= 4270
 T509_L2_f18_le: x_2nd_Flr_SF + 807.540001 t509_leaf2 <= 1641.54
 T509_R2_f3_ge: x_Overall_Qual - 5 t509_leaf2 >= 0
 T509_R3_f18_ge: x_2nd_Flr_SF - 834 t509_leaf3 >= 0
 TREE_510_ONEHOT: t510_leaf0 = 1
 TREE_511_ONEHOT: t511_leaf0 = 1
 TREE_512_ONEHOT: t512_leaf0 = 1
 TREE_513_ONEHOT: t513_leaf0 + t513_leaf1 + t513_leaf2 + t513_leaf3
   + t513_leaf4 + t513_leaf5 = 1
 T513_L0_f1_le: x_Lot_Area <= 21369.999999
 T513_L0_f9_le: x_Exter_Cond + 2.000001 t513_leaf0 <= 4
 T513_L1_f1_le: x_Lot_Area <= 21369.999999
 T513_R1_f9_ge: x_Exter_Cond - 3 t513_leaf1 >= -1
 T513_L1_f1_le: x_Lot_Area <= 18385.999999
 T513_L2_f1_le: x_Lot_Area <= 21369.999999
 T513_R2_f9_ge: x_Exter_Cond - 3 t513_leaf2 >= -1
 T513_R2_f1_ge: x_Lot_Area - 11386 t513_leaf2 >= 7000
 T513_R3_f1_ge: x_Lot_Area - 14370 t513_leaf3 >= 7000
 T513_L3_f213_le: x_BsmtFin_Type_1_GLQ + 1e-06 t513_leaf3 <= 1
 T513_L3_f20_le: x_Gr_Liv_Area + 1766.760001 t513_leaf3 <= 3421.76
 T513_R4_f1_ge: x_Lot_Area - 14370 t513_leaf4 >= 7000
 T513_L4_f213_le: x_BsmtFin_Type_1_GLQ + 1e-06 t513_leaf4 <= 1
 T513_R4_f20_ge: x_Gr_Liv_Area - 1655 t513_leaf4 >= 0
 T513_R5_f1_ge: x_Lot_Area - 14370 t513_leaf5 >= 7000
 T513_R5_f213_ge: x_BsmtFin_Type_1_GLQ - t513_leaf5 >= 0
 TREE_514_ONEHOT: t514_leaf0 + t514_leaf1 = 1
 T514_L0_f0_le: x_Lot_Frontage + 58.530001 t514_leaf0 <= 175.53
 T514_R1_f0_ge: x_Lot_Frontage - 117 t514_leaf1 >= 0
 TREE_515_ONEHOT: t515_leaf0 + t515_leaf1 + t515_leaf2 + t515_leaf3 = 1
 T515_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t515_leaf0 <= 1
 T515_L0_f273_le: x_Mo_Sold_3 + 1e-06 t515_leaf0 <= 1
 T515_L1_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t515_leaf1 <= 1
 T515_R1_f273_ge: x_Mo_Sold_3 - t515_leaf1 >= 0
 T515_R2_f207_ge: x_Bsmt_Exposure_Gd - t515_leaf2 >= 0
 T515_L2_f81_le: x_Lot_Config_Corner + 1e-06 t515_leaf2 <= 1
 T515_R3_f207_ge: x_Bsmt_Exposure_Gd - t515_leaf3 >= 0
 T515_R3_f81_ge: x_Lot_Config_Corner - t515_leaf3 >= 0
 TREE_516_ONEHOT: t516_leaf0 + t516_leaf1 + t516_leaf2 + t516_leaf3
   + t516_leaf4 = 1
 T516_L0_f295_le: x_Sale_Condition_Alloca + 1e-06 t516_leaf0 <= 1
 T516_L0_f206_le: x_Bsmt_Exposure_Av + 1e-06 t516_leaf0 <= 1
 T516_L0_f12_le: x_BsmtFin_SF_1 + 1264.500001 t516_leaf0 <= 1984.5
 T516_L1_f295_le: x_Sale_Condition_Alloca + 1e-06 t516_leaf1 <= 1
 T516_L1_f206_le: x_Bsmt_Exposure_Av + 1e-06 t516_leaf1 <= 1
 T516_R1_f12_ge: x_BsmtFin_SF_1 - 720 t516_leaf1 >= 0
 T516_L2_f295_le: x_Sale_Condition_Alloca + 1e-06 t516_leaf2 <= 1
 T516_R2_f206_ge: x_Bsmt_Exposure_Av - t516_leaf2 >= 0
 T516_L2_f15_le: x_Total_Bsmt_SF + 664.4800010000001 t516_leaf2 <= 2514.48
 T516_L3_f295_le: x_Sale_Condition_Alloca + 1e-06 t516_leaf3 <= 1
 T516_R3_f206_ge: x_Bsmt_Exposure_Av - t516_leaf3 >= 0
 T516_R3_f15_ge: x_Total_Bsmt_SF - 1850 t516_leaf3 >= 0
 T516_R4_f295_ge: x_Sale_Condition_Alloca - t516_leaf4 >= 0
 TREE_517_ONEHOT: t517_leaf0 + t517_leaf1 = 1
 T517_L0_f3_le: x_Overall_Qual + 999997.000001 t517_leaf0 <= 1e+06
 T517_R1_f3_ge: x_Overall_Qual - 3 t517_leaf1 >= 0
 TREE_518_ONEHOT: t518_leaf0 + t518_leaf1 + t518_leaf2 + t518_leaf3 = 1
 T518_L0_f3_le: x_Overall_Qual + 999997.000001 t518_leaf0 <= 1e+06
 T518_R1_f3_ge: x_Overall_Qual - 3 t518_leaf1 >= 0
 T518_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t518_leaf1 <= 1
 T518_R2_f3_ge: x_Overall_Qual - 3 t518_leaf2 >= 0
 T518_R2_f161_ge: x_Exterior_1st_AsbShng - t518_leaf2 >= 0
 T518_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t518_leaf2 <= 1
 T518_R3_f3_ge: x_Overall_Qual - 3 t518_leaf3 >= 0
 T518_R3_f161_ge: x_Exterior_1st_AsbShng - t518_leaf3 >= 0
 T518_R3_f297_ge: x_Sale_Condition_Normal - t518_leaf3 >= 0
 TREE_519_ONEHOT: t519_leaf0 + t519_leaf1 + t519_leaf2 + t519_leaf3 = 1
 T519_L0_f114_le: x_Neighborhood_StoneBr + 1e-06 t519_leaf0 <= 1
 T519_L0_f30_le: x_Fireplace_Qu + 2.000001 t519_leaf0 <= 4
 T519_L1_f114_le: x_Neighborhood_StoneBr + 1e-06 t519_leaf1 <= 1
 T519_R1_f30_ge: x_Fireplace_Qu >= 2
 T519_L1_f14_le: x_Bsmt_Unf_SF + 1346.680001 t519_leaf1 <= 1812.68
 T519_L2_f114_le: x_Neighborhood_StoneBr + 1e-06 t519_leaf2 <= 1
 T519_R2_f30_ge: x_Fireplace_Qu >= 2
 T519_R2_f14_ge: x_Bsmt_Unf_SF - 466 t519_leaf2 >= 0
 T519_R3_f114_ge: x_Neighborhood_StoneBr - t519_leaf3 >= 0
 TREE_520_ONEHOT: t520_leaf0 + t520_leaf1 + t520_leaf2 + t520_leaf3
   + t520_leaf4 + t520_leaf5 = 1
 T520_L0_f12_le: x_BsmtFin_SF_1 + 725.5000010000001 t520_leaf0 <= 1984.5
 T520_L0_f134_le: x_Bldg_Type_1Fam + 1e-06 t520_leaf0 <= 1
 T520_L1_f12_le: x_BsmtFin_SF_1 + 725.5000010000001 t520_leaf1 <= 1984.5
 T520_R1_f134_ge: x_Bldg_Type_1Fam - t520_leaf1 >= 0
 T520_L1_f31_le: x_Garage_Yr_Blt + 998033.000001 t520_leaf1 <= 1e+06
 T520_L2_f12_le: x_BsmtFin_SF_1 + 725.5000010000001 t520_leaf2 <= 1984.5
 T520_R2_f134_ge: x_Bldg_Type_1Fam - t520_leaf2 >= 0
 T520_R2_f31_ge: x_Garage_Yr_Blt - 1967 t520_leaf2 >= 0
 T520_R3_f12_ge: x_BsmtFin_SF_1 - 1259 t520_leaf3 >= 0
 T520_L3_f1_le: x_Lot_Area <= 15497.999999
 T520_R4_f12_ge: x_BsmtFin_SF_1 - 1259 t520_leaf4 >= 0
 T520_R4_f1_ge: x_Lot_Area - 8498 t520_leaf4 >= 7000
 T520_L4_f0_le: x_Lot_Frontage + 115.530001 t520_leaf4 <= 175.53
 T520_R5_f12_ge: x_BsmtFin_SF_1 - 1259 t520_leaf5 >= 0
 T520_R5_f1_ge: x_Lot_Area - 8498 t520_leaf5 >= 7000
 T520_R5_f0_ge: x_Lot_Frontage - 60 t520_leaf5 >= 0
 TREE_521_ONEHOT: t521_leaf0 + t521_leaf1 + t521_leaf2 + t521_leaf3
   + t521_leaf4 = 1
 T521_L0_f15_le: x_Total_Bsmt_SF + 1733.480001 t521_leaf0 <= 2514.48
 T521_L0_f20_le: x_Gr_Liv_Area + 2297.760001000001 t521_leaf0 <= 3421.76
 T521_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t521_leaf0 <= 1
 T521_L1_f15_le: x_Total_Bsmt_SF + 1733.480001 t521_leaf1 <= 2514.48
 T521_L1_f20_le: x_Gr_Liv_Area + 2297.760001000001 t521_leaf1 <= 3421.76
 T521_R1_f297_ge: x_Sale_Condition_Normal - t521_leaf1 >= 0
 T521_L2_f15_le: x_Total_Bsmt_SF + 1733.480001 t521_leaf2 <= 2514.48
 T521_R2_f20_ge: x_Gr_Liv_Area - 1124 t521_leaf2 >= 0
 T521_L2_f18_le: x_2nd_Flr_SF + 916.540001 t521_leaf2 <= 1641.54
 T521_L3_f15_le: x_Total_Bsmt_SF + 1733.480001 t521_leaf3 <= 2514.48
 T521_R3_f20_ge: x_Gr_Liv_Area - 1124 t521_leaf3 >= 0
 T521_R3_f18_ge: x_2nd_Flr_SF - 725 t521_leaf3 >= 0
 T521_R4_f15_ge: x_Total_Bsmt_SF - 781 t521_leaf4 >= 0
 TREE_522_ONEHOT: t522_leaf0 + t522_leaf1 + t522_leaf2 + t522_leaf3
   + t522_leaf4 = 1
 T522_L0_f3_le: x_Overall_Qual + 999997.000001 t522_leaf0 <= 1e+06
 T522_L0_f33_le: x_Garage_Area + 864.750001 t522_leaf0 <= 1163.75
 T522_L1_f3_le: x_Overall_Qual + 999997.000001 t522_leaf1 <= 1e+06
 T522_R1_f33_ge: x_Garage_Area - 299 t522_leaf1 >= 0
 T522_R2_f3_ge: x_Overall_Qual - 3 t522_leaf2 >= 0
 T522_L2_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t522_leaf2 <= 2514.48
 T522_L2_f169_le: x_Exterior_1st_MetalSd + 1e-06 t522_leaf2 <= 1
 T522_R3_f3_ge: x_Overall_Qual - 3 t522_leaf3 >= 0
 T522_L3_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t522_leaf3 <= 2514.48
 T522_R3_f169_ge: x_Exterior_1st_MetalSd - t522_leaf3 >= 0
 T522_R4_f3_ge: x_Overall_Qual - 3 t522_leaf4 >= 0
 T522_R4_f15_ge: x_Total_Bsmt_SF - 2136 t522_leaf4 >= 0
 TREE_523_ONEHOT: t523_leaf0 + t523_leaf1 + t523_leaf2 + t523_leaf3
   + t523_leaf4 + t523_leaf5 = 1
 T523_L0_f6_le: x_Year_Remod/Add + 998049.000001 t523_leaf0 <= 1e+06
 T523_L0_f44_le: x_Yr_Sold + 997991.000001 t523_leaf0 <= 1e+06
 T523_L1_f6_le: x_Year_Remod/Add + 998049.000001 t523_leaf1 <= 1e+06
 T523_R1_f44_ge: x_Yr_Sold - 2009 t523_leaf1 >= 0
 T523_R2_f6_ge: x_Year_Remod/Add - 1951 t523_leaf2 >= 0
 T523_L2_f46_le: x_MS_SubClass_30 + 1e-06 t523_leaf2 <= 1
 T523_L2_f43_le: x_Misc_Val + 2975.470001 t523_leaf2 <= 3029.47
 T523_R3_f6_ge: x_Year_Remod/Add - 1951 t523_leaf3 >= 0
 T523_L3_f46_le: x_MS_SubClass_30 + 1e-06 t523_leaf3 <= 1
 T523_R3_f43_ge: x_Misc_Val - 54 t523_leaf3 >= 0
 T523_R4_f6_ge: x_Year_Remod/Add - 1951 t523_leaf4 >= 0
 T523_R4_f46_ge: x_MS_SubClass_30 - t523_leaf4 >= 0
 T523_L4_f33_le: x_Garage_Area + 676.750001 t523_leaf4 <= 1163.75
 T523_R5_f6_ge: x_Year_Remod/Add - 1951 t523_leaf5 >= 0
 T523_R5_f46_ge: x_MS_SubClass_30 - t523_leaf5 >= 0
 T523_R5_f33_ge: x_Garage_Area - 487 t523_leaf5 >= 0
 TREE_524_ONEHOT: t524_leaf0 + t524_leaf1 + t524_leaf2 + t524_leaf3
   + t524_leaf4 + t524_leaf5 = 1
 T524_L0_f23_le: x_Full_Bath + 4.000001 t524_leaf0 <= 6
 T524_L0_f38_le: x_Enclosed_Porch + 58.750001 t524_leaf0 <= 292.75
 T524_L1_f23_le: x_Full_Bath + 4.000001 t524_leaf1 <= 6
 T524_R1_f38_ge: x_Enclosed_Porch - 234 t524_leaf1 >= 0
 T524_L1_f44_le: x_Yr_Sold + 997991.000001 t524_leaf1 <= 1e+06
 T524_L2_f23_le: x_Full_Bath + 4.000001 t524_leaf2 <= 6
 T524_R2_f38_ge: x_Enclosed_Porch - 234 t524_leaf2 >= 0
 T524_R2_f44_ge: x_Yr_Sold - 2009 t524_leaf2 >= 0
 T524_R3_f23_ge: x_Full_Bath - 2 t524_leaf3 >= 0
 T524_L3_f4_le: x_Overall_Cond + 999994.000001 t524_leaf3 <= 1e+06
 T524_L3_f9_le: x_Exter_Cond + 1.000001 t524_leaf3 <= 4
 T524_R4_f23_ge: x_Full_Bath - 2 t524_leaf4 >= 0
 T524_L4_f4_le: x_Overall_Cond + 999994.000001 t524_leaf4 <= 1e+06
 T524_R4_f9_ge: x_Exter_Cond - 4 t524_leaf4 >= -1
 T524_R5_f23_ge: x_Full_Bath - 2 t524_leaf5 >= 0
 T524_R5_f4_ge: x_Overall_Cond - 6 t524_leaf5 >= 0
 TREE_525_ONEHOT: t525_leaf0 = 1
 TREE_526_ONEHOT: t526_leaf0 + t526_leaf1 + t526_leaf2 = 1
 T526_L0_f3_le: x_Overall_Qual + 999997.000001 t526_leaf0 <= 1e+06
 T526_R1_f3_ge: x_Overall_Qual - 3 t526_leaf1 >= 0
 T526_L1_f28_le: x_TotRms_AbvGrd + 7.080001 t526_leaf1 <= 12.08
 T526_R2_f3_ge: x_Overall_Qual - 3 t526_leaf2 >= 0
 T526_R2_f28_ge: x_TotRms_AbvGrd - 5 t526_leaf2 >= 0
 TREE_527_ONEHOT: t527_leaf0 + t527_leaf1 = 1
 T527_L0_f3_le: x_Overall_Qual + 999997.000001 t527_leaf0 <= 1e+06
 T527_R1_f3_ge: x_Overall_Qual - 3 t527_leaf1 >= 0
 TREE_528_ONEHOT: t528_leaf0 + t528_leaf1 + t528_leaf2 + t528_leaf3
   + t528_leaf4 + t528_leaf5 = 1
 T528_L0_f114_le: x_Neighborhood_StoneBr + 1e-06 t528_leaf0 <= 1
 T528_L0_f1_le: x_Lot_Area <= 23256.999999
 T528_L0_f30_le: x_Fireplace_Qu + 2.000001 t528_leaf0 <= 4
 T528_L1_f114_le: x_Neighborhood_StoneBr + 1e-06 t528_leaf1 <= 1
 T528_L1_f1_le: x_Lot_Area <= 23256.999999
 T528_R1_f30_ge: x_Fireplace_Qu >= 2
 T528_L2_f114_le: x_Neighborhood_StoneBr + 1e-06 t528_leaf2 <= 1
 T528_R2_f1_ge: x_Lot_Area - 16257 t528_leaf2 >= 7000
 T528_L2_f82_le: x_Lot_Config_CulDSac + 1e-06 t528_leaf2 <= 1
 T528_L3_f114_le: x_Neighborhood_StoneBr + 1e-06 t528_leaf3 <= 1
 T528_R3_f1_ge: x_Lot_Area - 16257 t528_leaf3 >= 7000
 T528_R3_f82_ge: x_Lot_Config_CulDSac - t528_leaf3 >= 0
 T528_R4_f114_ge: x_Neighborhood_StoneBr - t528_leaf4 >= 0
 T528_L4_f289_le: x_Sale_Type_New + 1e-06 t528_leaf4 <= 1
 T528_R5_f114_ge: x_Neighborhood_StoneBr - t528_leaf5 >= 0
 T528_R5_f289_ge: x_Sale_Type_New - t528_leaf5 >= 0
 TREE_529_ONEHOT: t529_leaf0 = 1
 TREE_530_ONEHOT: t530_leaf0 = 1
 TREE_531_ONEHOT: t531_leaf0 + t531_leaf1 = 1
 T531_L0_f5_le: x_Year_Built + 998086.000001 t531_leaf0 <= 1e+06
 T531_R1_f5_ge: x_Year_Built - 1914 t531_leaf1 >= 0
 TREE_532_ONEHOT: t532_leaf0 + t532_leaf1 + t532_leaf2 + t532_leaf3
   + t532_leaf4 + t532_leaf5 = 1
 T532_L0_f4_le: x_Overall_Cond + 999994.000001 t532_leaf0 <= 1e+06
 T532_L0_f16_le: x_Heating_QC + 1.000001 t532_leaf0 <= 4
 T532_L0_f33_le: x_Garage_Area + 681.750001 t532_leaf0 <= 1163.75
 T532_L1_f4_le: x_Overall_Cond + 999994.000001 t532_leaf1 <= 1e+06
 T532_L1_f16_le: x_Heating_QC + 1.000001 t532_leaf1 <= 4
 T532_R1_f33_ge: x_Garage_Area - 482 t532_leaf1 >= 0
 T532_L2_f4_le: x_Overall_Cond + 999994.000001 t532_leaf2 <= 1e+06
 T532_R2_f16_ge: x_Heating_QC - t532_leaf2 >= 2
 T532_L2_f11_le: x_Bsmt_Cond + 2.000001 t532_leaf2 <= 4
 T532_L3_f4_le: x_Overall_Cond + 999994.000001 t532_leaf3 <= 1e+06
 T532_R3_f16_ge: x_Heating_QC - t532_leaf3 >= 2
 T532_R3_f11_ge: x_Bsmt_Cond - 3 t532_leaf3 >= -1
 T532_R4_f4_ge: x_Overall_Cond - 6 t532_leaf4 >= 0
 T532_L4_f15_le: x_Total_Bsmt_SF + 1903.480001 t532_leaf4 <= 2514.48
 T532_R5_f4_ge: x_Overall_Cond - 6 t532_leaf5 >= 0
 T532_R5_f15_ge: x_Total_Bsmt_SF - 611 t532_leaf5 >= 0
 TREE_533_ONEHOT: t533_leaf0 = 1
 TREE_534_ONEHOT: t534_leaf0 = 1
 TREE_535_ONEHOT: t535_leaf0 + t535_leaf1 + t535_leaf2 + t535_leaf3
   + t535_leaf4 + t535_leaf5 = 1
 T535_L0_f14_le: x_Bsmt_Unf_SF + 1420.680001 t535_leaf0 <= 1812.68
 T535_L0_f15_le: x_Total_Bsmt_SF + 472.4800010000001 t535_leaf0 <= 2514.48
 T535_L1_f14_le: x_Bsmt_Unf_SF + 1420.680001 t535_leaf1 <= 1812.68
 T535_R1_f15_ge: x_Total_Bsmt_SF - 2042 t535_leaf1 >= 0
 T535_R2_f14_ge: x_Bsmt_Unf_SF - 392 t535_leaf2 >= 0
 T535_L2_f96_le: x_Neighborhood_Edwards + 1e-06 t535_leaf2 <= 1
 T535_L2_f15_le: x_Total_Bsmt_SF + 1753.480001 t535_leaf2 <= 2514.48
 T535_R3_f14_ge: x_Bsmt_Unf_SF - 392 t535_leaf3 >= 0
 T535_L3_f96_le: x_Neighborhood_Edwards + 1e-06 t535_leaf3 <= 1
 T535_R3_f15_ge: x_Total_Bsmt_SF - 761 t535_leaf3 >= 0
 T535_R4_f14_ge: x_Bsmt_Unf_SF - 392 t535_leaf4 >= 0
 T535_R4_f96_ge: x_Neighborhood_Edwards - t535_leaf4 >= 0
 T535_L4_f33_le: x_Garage_Area + 855.750001 t535_leaf4 <= 1163.75
 T535_R5_f14_ge: x_Bsmt_Unf_SF - 392 t535_leaf5 >= 0
 T535_R5_f96_ge: x_Neighborhood_Edwards - t535_leaf5 >= 0
 T535_R5_f33_ge: x_Garage_Area - 308 t535_leaf5 >= 0
 TREE_536_ONEHOT: t536_leaf0 = 1
 TREE_537_ONEHOT: t537_leaf0 + t537_leaf1 + t537_leaf2 + t537_leaf3 = 1
 T537_L0_f3_le: x_Overall_Qual + 999993.000001 t537_leaf0 <= 1e+06
 T537_R1_f3_ge: x_Overall_Qual - 7 t537_leaf1 >= 0
 T537_L1_f77_le: x_Land_Contour_Bnk + 1e-06 t537_leaf1 <= 1
 T537_L1_f105_le: x_Neighborhood_NWAmes + 1e-06 t537_leaf1 <= 1
 T537_R2_f3_ge: x_Overall_Qual - 7 t537_leaf2 >= 0
 T537_L2_f77_le: x_Land_Contour_Bnk + 1e-06 t537_leaf2 <= 1
 T537_R2_f105_ge: x_Neighborhood_NWAmes - t537_leaf2 >= 0
 T537_R3_f3_ge: x_Overall_Qual - 7 t537_leaf3 >= 0
 T537_R3_f77_ge: x_Land_Contour_Bnk - t537_leaf3 >= 0
 TREE_538_ONEHOT: t538_leaf0 + t538_leaf1 + t538_leaf2 = 1
 T538_L0_f3_le: x_Overall_Qual + 999993.000001 t538_leaf0 <= 1e+06
 T538_L0_f5_le: x_Year_Built + 997995.000001 t538_leaf0 <= 1e+06
 T538_L1_f3_le: x_Overall_Qual + 999993.000001 t538_leaf1 <= 1e+06
 T538_R1_f5_ge: x_Year_Built - 2005 t538_leaf1 >= 0
 T538_R2_f3_ge: x_Overall_Qual - 7 t538_leaf2 >= 0
 TREE_539_ONEHOT: t539_leaf0 + t539_leaf1 + t539_leaf2 + t539_leaf3
   + t539_leaf4 = 1
 T539_L0_f44_le: x_Yr_Sold + 997993.000001 t539_leaf0 <= 1e+06
 T539_L0_f1_le: x_Lot_Area <= 16284.999999
 T539_L1_f44_le: x_Yr_Sold + 997993.000001 t539_leaf1 <= 1e+06
 T539_R1_f1_ge: x_Lot_Area - 9285 t539_leaf1 >= 7000
 T539_R2_f44_ge: x_Yr_Sold - 2007 t539_leaf2 >= 0
 T539_L2_f7_le: x_Mas_Vnr_Area + 143.070001 t539_leaf2 <= 931.07
 T539_L2_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t539_leaf2 <= 2514.48
 T539_R3_f44_ge: x_Yr_Sold - 2007 t539_leaf3 >= 0
 T539_L3_f7_le: x_Mas_Vnr_Area + 143.070001 t539_leaf3 <= 931.07
 T539_R3_f15_ge: x_Total_Bsmt_SF - 2136 t539_leaf3 >= 0
 T539_R4_f44_ge: x_Yr_Sold - 2007 t539_leaf4 >= 0
 T539_R4_f7_ge: x_Mas_Vnr_Area - 788 t539_leaf4 >= 0
 TREE_540_ONEHOT: t540_leaf0 + t540_leaf1 + t540_leaf2 + t540_leaf3
   + t540_leaf4 = 1
 T540_L0_f277_le: x_Mo_Sold_7 + 1e-06 t540_leaf0 <= 1
 T540_L0_f35_le: x_Garage_Cond + 3.000001 t540_leaf0 <= 4
 T540_L0_f232_le: x_Central_Air_Y + 1e-06 t540_leaf0 <= 1
 T540_L1_f277_le: x_Mo_Sold_7 + 1e-06 t540_leaf1 <= 1
 T540_L1_f35_le: x_Garage_Cond + 3.000001 t540_leaf1 <= 4
 T540_R1_f232_ge: x_Central_Air_Y - t540_leaf1 >= 0
 T540_L2_f277_le: x_Mo_Sold_7 + 1e-06 t540_leaf2 <= 1
 T540_R2_f35_ge: x_Garage_Cond - 2 t540_leaf2 >= -1
 T540_L2_f4_le: x_Overall_Cond + 999995.000001 t540_leaf2 <= 1e+06
 T540_L3_f277_le: x_Mo_Sold_7 + 1e-06 t540_leaf3 <= 1
 T540_R3_f35_ge: x_Garage_Cond - 2 t540_leaf3 >= -1
 T540_R3_f4_ge: x_Overall_Cond - 5 t540_leaf3 >= 0
 T540_R4_f277_ge: x_Mo_Sold_7 - t540_leaf4 >= 0
 TREE_541_ONEHOT: t541_leaf0 + t541_leaf1 = 1
 T541_L0_f290_le: x_Sale_Type_Oth + 1e-06 t541_leaf0 <= 1
 T541_R1_f290_ge: x_Sale_Type_Oth - t541_leaf1 >= 0
 TREE_542_ONEHOT: t542_leaf0 = 1
 TREE_543_ONEHOT: t543_leaf0 + t543_leaf1 + t543_leaf2 + t543_leaf3
   + t543_leaf4 = 1
 T543_L0_f3_le: x_Overall_Qual + 999993.000001 t543_leaf0 <= 1e+06
 T543_L0_f72_le: x_Alley_Pave + 1e-06 t543_leaf0 <= 1
 T543_L0_f44_le: x_Yr_Sold + 997990.000001 t543_leaf0 <= 1e+06
 T543_L1_f3_le: x_Overall_Qual + 999993.000001 t543_leaf1 <= 1e+06
 T543_L1_f72_le: x_Alley_Pave + 1e-06 t543_leaf1 <= 1
 T543_R1_f44_ge: x_Yr_Sold - 2010 t543_leaf1 >= 0
 T543_L2_f3_le: x_Overall_Qual + 999993.000001 t543_leaf2 <= 1e+06
 T543_R2_f72_ge: x_Alley_Pave - t543_leaf2 >= 0
 T543_R3_f3_ge: x_Overall_Qual - 7 t543_leaf3 >= 0
 T543_L3_f77_le: x_Land_Contour_Bnk + 1e-06 t543_leaf3 <= 1
 T543_R4_f3_ge: x_Overall_Qual - 7 t543_leaf4 >= 0
 T543_R4_f77_ge: x_Land_Contour_Bnk - t543_leaf4 >= 0
 TREE_544_ONEHOT: t544_leaf0 + t544_leaf1 + t544_leaf2 + t544_leaf3
   + t544_leaf4 + t544_leaf5 = 1
 T544_L0_f3_le: x_Overall_Qual + 999993.000001 t544_leaf0 <= 1e+06
 T544_L0_f37_le: x_Open_Porch_SF + 319.810001 t544_leaf0 <= 488.81
 T544_L1_f3_le: x_Overall_Qual + 999993.000001 t544_leaf1 <= 1e+06
 T544_R1_f37_ge: x_Open_Porch_SF - 169 t544_leaf1 >= 0
 T544_R2_f3_ge: x_Overall_Qual - 7 t544_leaf2 >= 0
 T544_L2_f6_le: x_Year_Remod/Add + 998006.000001 t544_leaf2 <= 1e+06
 T544_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t544_leaf2 <= 1
 T544_R3_f3_ge: x_Overall_Qual - 7 t544_leaf3 >= 0
 T544_L3_f6_le: x_Year_Remod/Add + 998006.000001 t544_leaf3 <= 1e+06
 T544_R3_f297_ge: x_Sale_Condition_Normal - t544_leaf3 >= 0
 T544_R4_f3_ge: x_Overall_Qual - 7 t544_leaf4 >= 0
 T544_R4_f6_ge: x_Year_Remod/Add - 1994 t544_leaf4 >= 0
 T544_L4_f77_le: x_Land_Contour_Bnk + 1e-06 t544_leaf4 <= 1
 T544_R5_f3_ge: x_Overall_Qual - 7 t544_leaf5 >= 0
 T544_R5_f6_ge: x_Year_Remod/Add - 1994 t544_leaf5 >= 0
 T544_R5_f77_ge: x_Land_Contour_Bnk - t544_leaf5 >= 0
 TREE_545_ONEHOT: t545_leaf0 + t545_leaf1 + t545_leaf2 = 1
 T545_L0_f96_le: x_Neighborhood_Edwards + 1e-06 t545_leaf0 <= 1
 T545_R1_f96_ge: x_Neighborhood_Edwards - t545_leaf1 >= 0
 T545_L1_f80_le: x_Land_Contour_Lvl + 1e-06 t545_leaf1 <= 1
 T545_R2_f96_ge: x_Neighborhood_Edwards - t545_leaf2 >= 0
 T545_R2_f80_ge: x_Land_Contour_Lvl - t545_leaf2 >= 0
 TREE_546_ONEHOT: t546_leaf0 + t546_leaf1 + t546_leaf2 = 1
 T546_L0_f120_le: x_Condition_1_PosA + 1e-06 t546_leaf0 <= 1
 T546_L0_f210_le: x_Bsmt_Exposure_No_aplica + 1e-06 t546_leaf0 <= 1
 T546_L1_f120_le: x_Condition_1_PosA + 1e-06 t546_leaf1 <= 1
 T546_R1_f210_ge: x_Bsmt_Exposure_No_aplica - t546_leaf1 >= 0
 T546_R2_f120_ge: x_Condition_1_PosA - t546_leaf2 >= 0
 TREE_547_ONEHOT: t547_leaf0 + t547_leaf1 + t547_leaf2 + t547_leaf3
   + t547_leaf4 = 1
 T547_L0_f17_le: x_1st_Flr_SF + 1888.900001 t547_leaf0 <= 2534.9
 T547_R1_f17_ge: x_1st_Flr_SF - 646 t547_leaf1 >= 0
 T547_L1_f15_le: x_Total_Bsmt_SF + 2269.480001 t547_leaf1 <= 2514.48
 T547_L1_f5_le: x_Year_Built + 998048.000001 t547_leaf1 <= 1e+06
 T547_R2_f17_ge: x_1st_Flr_SF - 646 t547_leaf2 >= 0
 T547_L2_f15_le: x_Total_Bsmt_SF + 2269.480001 t547_leaf2 <= 2514.48
 T547_R2_f5_ge: x_Year_Built - 1952 t547_leaf2 >= 0
 T547_R3_f17_ge: x_1st_Flr_SF - 646 t547_leaf3 >= 0
 T547_R3_f15_ge: x_Total_Bsmt_SF - 245 t547_leaf3 >= 0
 T547_L3_f15_le: x_Total_Bsmt_SF + 1823.480001 t547_leaf3 <= 2514.48
 T547_R4_f17_ge: x_1st_Flr_SF - 646 t547_leaf4 >= 0
 T547_R4_f15_ge: x_Total_Bsmt_SF - 245 t547_leaf4 >= 0
 T547_R4_f15_ge: x_Total_Bsmt_SF - 691 t547_leaf4 >= 0
 TREE_548_ONEHOT: t548_leaf0 = 1
 TREE_549_ONEHOT: t549_leaf0 + t549_leaf1 + t549_leaf2 + t549_leaf3 = 1
 T549_L0_f26_le: x_Kitchen_AbvGr + 1.000001 t549_leaf0 <= 3
 T549_L0_f117_le: x_Condition_1_Artery + 1e-06 t549_leaf0 <= 1
 T549_L0_f3_le: x_Overall_Qual + 999992.000001 t549_leaf0 <= 1e+06
 T549_L1_f26_le: x_Kitchen_AbvGr + 1.000001 t549_leaf1 <= 3
 T549_L1_f117_le: x_Condition_1_Artery + 1e-06 t549_leaf1 <= 1
 T549_R1_f3_ge: x_Overall_Qual - 8 t549_leaf1 >= 0
 T549_L2_f26_le: x_Kitchen_AbvGr + 1.000001 t549_leaf2 <= 3
 T549_R2_f117_ge: x_Condition_1_Artery - t549_leaf2 >= 0
 T549_R3_f26_ge: x_Kitchen_AbvGr - 2 t549_leaf3 >= 0
 TREE_550_ONEHOT: t550_leaf0 + t550_leaf1 + t550_leaf2 + t550_leaf3 = 1
 T550_L0_f9_le: x_Exter_Cond + 2.000001 t550_leaf0 <= 4
 T550_L0_f28_le: x_TotRms_AbvGrd + 6.080001 t550_leaf0 <= 12.08
 T550_L1_f9_le: x_Exter_Cond + 2.000001 t550_leaf1 <= 4
 T550_R1_f28_ge: x_TotRms_AbvGrd - 6 t550_leaf1 >= 0
 T550_R2_f9_ge: x_Exter_Cond - 3 t550_leaf2 >= -1
 T550_L2_f185_le: x_Exterior_2nd_MetalSd + 1e-06 t550_leaf2 <= 1
 T550_R3_f9_ge: x_Exter_Cond - 3 t550_leaf3 >= -1
 T550_R3_f185_ge: x_Exterior_2nd_MetalSd - t550_leaf3 >= 0
 TREE_551_ONEHOT: t551_leaf0 = 1
 TREE_552_ONEHOT: t552_leaf0 + t552_leaf1 + t552_leaf2 + t552_leaf3 = 1
 T552_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t552_leaf0 <= 1
 T552_L0_f1_le: x_Lot_Area <= 17226.999999
 T552_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t552_leaf1 <= 1
 T552_R1_f1_ge: x_Lot_Area - 10227 t552_leaf1 >= 7000
 T552_L1_f37_le: x_Open_Porch_SF + 226.810001 t552_leaf1 <= 488.81
 T552_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t552_leaf2 <= 1
 T552_R2_f1_ge: x_Lot_Area - 10227 t552_leaf2 >= 7000
 T552_R2_f37_ge: x_Open_Porch_SF - 262 t552_leaf2 >= 0
 T552_R3_f95_ge: x_Neighborhood_Crawfor - t552_leaf3 >= 0
 TREE_553_ONEHOT: t553_leaf0 = 1
 TREE_554_ONEHOT: t554_leaf0 = 1
 TREE_555_ONEHOT: t555_leaf0 = 1
 TREE_556_ONEHOT: t556_leaf0 = 1
 TREE_557_ONEHOT: t557_leaf0 + t557_leaf1 = 1
 T557_L0_f23_le: x_Full_Bath + 3.000001 t557_leaf0 <= 6
 T557_R1_f23_ge: x_Full_Bath - 3 t557_leaf1 >= 0
 TREE_558_ONEHOT: t558_leaf0 + t558_leaf1 + t558_leaf2 + t558_leaf3 = 1
 T558_L0_f33_le: x_Garage_Area + 268.750001 t558_leaf0 <= 1163.75
 T558_L0_f117_le: x_Condition_1_Artery + 1e-06 t558_leaf0 <= 1
 T558_L0_f17_le: x_1st_Flr_SF + 1793.900001 t558_leaf0 <= 2534.9
 T558_L1_f33_le: x_Garage_Area + 268.750001 t558_leaf1 <= 1163.75
 T558_L1_f117_le: x_Condition_1_Artery + 1e-06 t558_leaf1 <= 1
 T558_R1_f17_ge: x_1st_Flr_SF - 741 t558_leaf1 >= 0
 T558_L2_f33_le: x_Garage_Area + 268.750001 t558_leaf2 <= 1163.75
 T558_R2_f117_ge: x_Condition_1_Artery - t558_leaf2 >= 0
 T558_R3_f33_ge: x_Garage_Area - 895 t558_leaf3 >= 0
 TREE_559_ONEHOT: t559_leaf0 + t559_leaf1 = 1
 T559_L0_f36_le: x_Wood_Deck_SF + 309.900001 t559_leaf0 <= 525.9
 T559_R1_f36_ge: x_Wood_Deck_SF - 216 t559_leaf1 >= 0
 TREE_560_ONEHOT: t560_leaf0 + t560_leaf1 + t560_leaf2 + t560_leaf3
   + t560_leaf4 = 1
 T560_L0_f1_le: x_Lot_Area <= 11764.999999
 T560_L0_f19_le: x_Low_Qual_Fin_SF + 999886.000001 t560_leaf0 <= 1e+06
 T560_L1_f1_le: x_Lot_Area <= 11764.999999
 T560_R1_f19_ge: x_Low_Qual_Fin_SF - 114 t560_leaf1 >= 0
 T560_R2_f1_ge: x_Lot_Area - 4765 t560_leaf2 >= 7000
 T560_L2_f35_le: x_Garage_Cond + 2.000001 t560_leaf2 <= 4
 T560_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t560_leaf2 <= 1
 T560_R3_f1_ge: x_Lot_Area - 4765 t560_leaf3 >= 7000
 T560_L3_f35_le: x_Garage_Cond + 2.000001 t560_leaf3 <= 4
 T560_R3_f297_ge: x_Sale_Condition_Normal - t560_leaf3 >= 0
 T560_R4_f1_ge: x_Lot_Area - 4765 t560_leaf4 >= 7000
 T560_R4_f35_ge: x_Garage_Cond - 3 t560_leaf4 >= -1
 TREE_561_ONEHOT: t561_leaf0 + t561_leaf1 + t561_leaf2 = 1
 T561_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t561_leaf0 <= 1
 T561_L0_f25_le: x_Bedroom_AbvGr + 5.000001 t561_leaf0 <= 7
 T561_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t561_leaf1 <= 1
 T561_R1_f25_ge: x_Bedroom_AbvGr - 2 t561_leaf1 >= 0
 T561_R2_f245_ge: x_FunctioNo_aplical_Typ - t561_leaf2 >= 0
 TREE_562_ONEHOT: t562_leaf0 + t562_leaf1 + t562_leaf2 = 1
 T562_L0_f17_le: x_1st_Flr_SF + 1793.900001 t562_leaf0 <= 2534.9
 T562_L0_f20_le: x_Gr_Liv_Area + 2445.760001 t562_leaf0 <= 3421.76
 T562_L1_f17_le: x_1st_Flr_SF + 1793.900001 t562_leaf1 <= 2534.9
 T562_R1_f20_ge: x_Gr_Liv_Area - 976 t562_leaf1 >= 0
 T562_R2_f17_ge: x_1st_Flr_SF - 741 t562_leaf2 >= 0
 TREE_563_ONEHOT: t563_leaf0 = 1
 TREE_564_ONEHOT: t564_leaf0 = 1
 TREE_565_ONEHOT: t565_leaf0 + t565_leaf1 = 1
 T565_L0_f40_le: x_Screen_Porch + 176.270001 t565_leaf0 <= 268.27
 T565_R1_f40_ge: x_Screen_Porch - 92 t565_leaf1 >= 0
 TREE_566_ONEHOT: t566_leaf0 + t566_leaf1 + t566_leaf2 + t566_leaf3
   + t566_leaf4 = 1
 T566_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t566_leaf0 <= 1
 T566_L0_f15_le: x_Total_Bsmt_SF + 1839.480001 t566_leaf0 <= 2514.48
 T566_L0_f6_le: x_Year_Remod/Add + 998043.000001 t566_leaf0 <= 1e+06
 T566_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t566_leaf1 <= 1
 T566_L1_f15_le: x_Total_Bsmt_SF + 1839.480001 t566_leaf1 <= 2514.48
 T566_R1_f6_ge: x_Year_Remod/Add - 1957 t566_leaf1 >= 0
 T566_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t566_leaf2 <= 1
 T566_R2_f15_ge: x_Total_Bsmt_SF - 675 t566_leaf2 >= 0
 T566_L2_f15_le: x_Total_Bsmt_SF + 1749.480001 t566_leaf2 <= 2514.48
 T566_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t566_leaf3 <= 1
 T566_R3_f15_ge: x_Total_Bsmt_SF - 675 t566_leaf3 >= 0
 T566_R3_f15_ge: x_Total_Bsmt_SF - 765 t566_leaf3 >= 0
 T566_R4_f95_ge: x_Neighborhood_Crawfor - t566_leaf4 >= 0
 TREE_567_ONEHOT: t567_leaf0 + t567_leaf1 + t567_leaf2 + t567_leaf3 = 1
 T567_L0_f4_le: x_Overall_Cond + 999996.000001 t567_leaf0 <= 1e+06
 T567_L0_f6_le: x_Year_Remod/Add + 998049.000001 t567_leaf0 <= 1e+06
 T567_L1_f4_le: x_Overall_Cond + 999996.000001 t567_leaf1 <= 1e+06
 T567_R1_f6_ge: x_Year_Remod/Add - 1951 t567_leaf1 >= 0
 T567_L1_f167_le: x_Exterior_1st_HdBoard + 1e-06 t567_leaf1 <= 1
 T567_L2_f4_le: x_Overall_Cond + 999996.000001 t567_leaf2 <= 1e+06
 T567_R2_f6_ge: x_Year_Remod/Add - 1951 t567_leaf2 >= 0
 T567_R2_f167_ge: x_Exterior_1st_HdBoard - t567_leaf2 >= 0
 T567_R3_f4_ge: x_Overall_Cond - 4 t567_leaf3 >= 0
 TREE_568_ONEHOT: t568_leaf0 + t568_leaf1 + t568_leaf2 + t568_leaf3 = 1
 T568_L0_f3_le: x_Overall_Qual + 999996.000001 t568_leaf0 <= 1e+06
 T568_R1_f3_ge: x_Overall_Qual - 4 t568_leaf1 >= 0
 T568_L1_f5_le: x_Year_Built + 998077.000001 t568_leaf1 <= 1e+06
 T568_L1_f10_le: x_Bsmt_Qual + 2.000001 t568_leaf1 <= 4
 T568_R2_f3_ge: x_Overall_Qual - 4 t568_leaf2 >= 0
 T568_L2_f5_le: x_Year_Built + 998077.000001 t568_leaf2 <= 1e+06
 T568_R2_f10_ge: x_Bsmt_Qual >= 2
 T568_R3_f3_ge: x_Overall_Qual - 4 t568_leaf3 >= 0
 T568_R3_f5_ge: x_Year_Built - 1923 t568_leaf3 >= 0
 TREE_569_ONEHOT: t569_leaf0 + t569_leaf1 + t569_leaf2 = 1
 T569_L0_f37_le: x_Open_Porch_SF + 197.810001 t569_leaf0 <= 488.81
 T569_L0_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t569_leaf0 <= 2514.48
 T569_L1_f37_le: x_Open_Porch_SF + 197.810001 t569_leaf1 <= 488.81
 T569_R1_f15_ge: x_Total_Bsmt_SF - 2223 t569_leaf1 >= 0
 T569_R2_f37_ge: x_Open_Porch_SF - 291 t569_leaf2 >= 0
 TREE_570_ONEHOT: t570_leaf0 = 1
 TREE_571_ONEHOT: t571_leaf0 + t571_leaf1 + t571_leaf2 + t571_leaf3
   + t571_leaf4 = 1
 T571_L0_f296_le: x_Sale_Condition_Family + 1e-06 t571_leaf0 <= 1
 T571_L0_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t571_leaf0 <= 1
 T571_L0_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t571_leaf0 <= 2514.48
 T571_L1_f296_le: x_Sale_Condition_Family + 1e-06 t571_leaf1 <= 1
 T571_L1_f218_le: x_BsmtFin_Type_2_ALQ + 1e-06 t571_leaf1 <= 1
 T571_R1_f15_ge: x_Total_Bsmt_SF - 2136 t571_leaf1 >= 0
 T571_L2_f296_le: x_Sale_Condition_Family + 1e-06 t571_leaf2 <= 1
 T571_R2_f218_ge: x_BsmtFin_Type_2_ALQ - t571_leaf2 >= 0
 T571_R3_f296_ge: x_Sale_Condition_Family - t571_leaf3 >= 0
 T571_L3_f7_le: x_Mas_Vnr_Area + 739.070001 t571_leaf3 <= 931.07
 T571_R4_f296_ge: x_Sale_Condition_Family - t571_leaf4 >= 0
 T571_R4_f7_ge: x_Mas_Vnr_Area - 192 t571_leaf4 >= 0
 TREE_572_ONEHOT: t572_leaf0 + t572_leaf1 + t572_leaf2 + t572_leaf3 = 1
 T572_L0_f5_le: x_Year_Built + 997992.000001 t572_leaf0 <= 1e+06
 T572_L0_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t572_leaf0 <= 2514.48
 T572_L0_f33_le: x_Garage_Area + 533.750001 t572_leaf0 <= 1163.75
 T572_L1_f5_le: x_Year_Built + 997992.000001 t572_leaf1 <= 1e+06
 T572_L1_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t572_leaf1 <= 2514.48
 T572_R1_f33_ge: x_Garage_Area - 630 t572_leaf1 >= 0
 T572_L2_f5_le: x_Year_Built + 997992.000001 t572_leaf2 <= 1e+06
 T572_R2_f15_ge: x_Total_Bsmt_SF - 2136 t572_leaf2 >= 0
 T572_R3_f5_ge: x_Year_Built - 2008 t572_leaf3 >= 0
 TREE_573_ONEHOT: t573_leaf0 + t573_leaf1 = 1
 T573_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t573_leaf0 <= 1
 T573_R1_f164_ge: x_Exterior_1st_BrkFace - t573_leaf1 >= 0
 TREE_574_ONEHOT: t574_leaf0 = 1
 TREE_575_ONEHOT: t575_leaf0 + t575_leaf1 + t575_leaf2 + t575_leaf3
   + t575_leaf4 = 1
 T575_L0_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t575_leaf0 <= 1
 T575_L0_f6_le: x_Year_Remod/Add + 998049.000001 t575_leaf0 <= 1e+06
 T575_L0_f3_le: x_Overall_Qual + 999995.000001 t575_leaf0 <= 1e+06
 T575_L1_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t575_leaf1 <= 1
 T575_L1_f6_le: x_Year_Remod/Add + 998049.000001 t575_leaf1 <= 1e+06
 T575_R1_f3_ge: x_Overall_Qual - 5 t575_leaf1 >= 0
 T575_L2_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t575_leaf2 <= 1
 T575_R2_f6_ge: x_Year_Remod/Add - 1951 t575_leaf2 >= 0
 T575_L2_f161_le: x_Exterior_1st_AsbShng + 1e-06 t575_leaf2 <= 1
 T575_L3_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t575_leaf3 <= 1
 T575_R3_f6_ge: x_Year_Remod/Add - 1951 t575_leaf3 >= 0
 T575_R3_f161_ge: x_Exterior_1st_AsbShng - t575_leaf3 >= 0
 T575_R4_f214_ge: x_BsmtFin_Type_1_LwQ - t575_leaf4 >= 0
 TREE_576_ONEHOT: t576_leaf0 + t576_leaf1 + t576_leaf2 + t576_leaf3 = 1
 T576_L0_f28_le: x_TotRms_AbvGrd + 7.080001 t576_leaf0 <= 12.08
 T576_L0_f0_le: x_Lot_Frontage + 124.530001 t576_leaf0 <= 175.53
 T576_L1_f28_le: x_TotRms_AbvGrd + 7.080001 t576_leaf1 <= 12.08
 T576_R1_f0_ge: x_Lot_Frontage - 51 t576_leaf1 >= 0
 T576_R2_f28_ge: x_TotRms_AbvGrd - 5 t576_leaf2 >= 0
 T576_L2_f154_le: x_Roof_Matl_CompShg + 1e-06 t576_leaf2 <= 1
 T576_R3_f28_ge: x_TotRms_AbvGrd - 5 t576_leaf3 >= 0
 T576_R3_f154_ge: x_Roof_Matl_CompShg - t576_leaf3 >= 0
 TREE_577_ONEHOT: t577_leaf0 = 1
 TREE_578_ONEHOT: t578_leaf0 = 1
 TREE_579_ONEHOT: t579_leaf0 + t579_leaf1 + t579_leaf2 + t579_leaf3 = 1
 T579_L0_f15_le: x_Total_Bsmt_SF + 1945.480001 t579_leaf0 <= 2514.48
 T579_L0_f86_le: x_Land_Slope_Gtl + 1e-06 t579_leaf0 <= 1
 T579_L1_f15_le: x_Total_Bsmt_SF + 1945.480001 t579_leaf1 <= 2514.48
 T579_R1_f86_ge: x_Land_Slope_Gtl - t579_leaf1 >= 0
 T579_R2_f15_ge: x_Total_Bsmt_SF - 569 t579_leaf2 >= 0
 T579_L2_f35_le: x_Garage_Cond + 3.000001 t579_leaf2 <= 4
 T579_R3_f15_ge: x_Total_Bsmt_SF - 569 t579_leaf3 >= 0
 T579_R3_f35_ge: x_Garage_Cond - 2 t579_leaf3 >= -1
 TREE_580_ONEHOT: t580_leaf0 + t580_leaf1 + t580_leaf2 + t580_leaf3
   + t580_leaf4 = 1
 T580_L0_f7_le: x_Mas_Vnr_Area + 71.07000100000005 t580_leaf0 <= 931.07
 T580_L0_f5_le: x_Year_Built + 998089.000001 t580_leaf0 <= 1e+06
 T580_L0_f15_le: x_Total_Bsmt_SF + 1854.480001 t580_leaf0 <= 2514.48
 T580_L1_f7_le: x_Mas_Vnr_Area + 71.07000100000005 t580_leaf1 <= 931.07
 T580_L1_f5_le: x_Year_Built + 998089.000001 t580_leaf1 <= 1e+06
 T580_R1_f15_ge: x_Total_Bsmt_SF - 660 t580_leaf1 >= 0
 T580_L2_f7_le: x_Mas_Vnr_Area + 71.07000100000005 t580_leaf2 <= 931.07
 T580_R2_f5_ge: x_Year_Built - 1911 t580_leaf2 >= 0
 T580_L2_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t580_leaf2 <= 1
 T580_L3_f7_le: x_Mas_Vnr_Area + 71.07000100000005 t580_leaf3 <= 931.07
 T580_R3_f5_ge: x_Year_Built - 1911 t580_leaf3 >= 0
 T580_R3_f216_ge: x_BsmtFin_Type_1_Rec - t580_leaf3 >= 0
 T580_R4_f7_ge: x_Mas_Vnr_Area - 860 t580_leaf4 >= 0
 TREE_581_ONEHOT: t581_leaf0 = 1
 TREE_582_ONEHOT: t582_leaf0 + t582_leaf1 + t582_leaf2 + t582_leaf3
   + t582_leaf4 = 1
 T582_L0_f296_le: x_Sale_Condition_Family + 1e-06 t582_leaf0 <= 1
 T582_L0_f16_le: x_Heating_QC + 2.000001 t582_leaf0 <= 4
 T582_L0_f17_le: x_1st_Flr_SF + 1798.900001 t582_leaf0 <= 2534.9
 T582_L1_f296_le: x_Sale_Condition_Family + 1e-06 t582_leaf1 <= 1
 T582_L1_f16_le: x_Heating_QC + 2.000001 t582_leaf1 <= 4
 T582_R1_f17_ge: x_1st_Flr_SF - 736 t582_leaf1 >= 0
 T582_L2_f296_le: x_Sale_Condition_Family + 1e-06 t582_leaf2 <= 1
 T582_R2_f16_ge: x_Heating_QC >= 2
 T582_R3_f296_ge: x_Sale_Condition_Family - t582_leaf3 >= 0
 T582_L3_f17_le: x_1st_Flr_SF + 1220.900001 t582_leaf3 <= 2534.9
 T582_R4_f296_ge: x_Sale_Condition_Family - t582_leaf4 >= 0
 T582_R4_f17_ge: x_1st_Flr_SF - 1314 t582_leaf4 >= 0
 TREE_583_ONEHOT: t583_leaf0 = 1
 TREE_584_ONEHOT: t584_leaf0 + t584_leaf1 = 1
 T584_L0_f21_le: x_Bsmt_Full_Bath + 0.020001 t584_leaf0 <= 2.02
 T584_R1_f21_ge: x_Bsmt_Full_Bath - 2 t584_leaf1 >= 0
 TREE_585_ONEHOT: t585_leaf0 + t585_leaf1 + t585_leaf2 + t585_leaf3
   + t585_leaf4 = 1
 T585_L0_f0_le: x_Lot_Frontage + 58.530001 t585_leaf0 <= 175.53
 T585_L0_f1_le: x_Lot_Area <= 17226.999999
 T585_L0_f15_le: x_Total_Bsmt_SF + 1441.480001 t585_leaf0 <= 2514.48
 T585_L1_f0_le: x_Lot_Frontage + 58.530001 t585_leaf1 <= 175.53
 T585_L1_f1_le: x_Lot_Area <= 17226.999999
 T585_R1_f15_ge: x_Total_Bsmt_SF - 1073 t585_leaf1 >= 0
 T585_L2_f0_le: x_Lot_Frontage + 58.530001 t585_leaf2 <= 175.53
 T585_R2_f1_ge: x_Lot_Area - 10227 t585_leaf2 >= 7000
 T585_R3_f0_ge: x_Lot_Frontage - 117 t585_leaf3 >= 0
 T585_L3_f14_le: x_Bsmt_Unf_SF + 1724.680001 t585_leaf3 <= 1812.68
 T585_R4_f0_ge: x_Lot_Frontage - 117 t585_leaf4 >= 0
 T585_R4_f14_ge: x_Bsmt_Unf_SF - 88 t585_leaf4 >= 0
 TREE_586_ONEHOT: t586_leaf0 + t586_leaf1 = 1
 T586_L0_f147_le: x_Roof_Style_Flat + 1e-06 t586_leaf0 <= 1
 T586_R1_f147_ge: x_Roof_Style_Flat - t586_leaf1 >= 0
 TREE_587_ONEHOT: t587_leaf0 = 1
 TREE_588_ONEHOT: t588_leaf0 + t588_leaf1 + t588_leaf2 + t588_leaf3 = 1
 T588_L0_f3_le: x_Overall_Qual + 999993.000001 t588_leaf0 <= 1e+06
 T588_R1_f3_ge: x_Overall_Qual - 7 t588_leaf1 >= 0
 T588_L1_f6_le: x_Year_Remod/Add + 998007.000001 t588_leaf1 <= 1e+06
 T588_L1_f9_le: x_Exter_Cond + 1.000001 t588_leaf1 <= 4
 T588_R2_f3_ge: x_Overall_Qual - 7 t588_leaf2 >= 0
 T588_L2_f6_le: x_Year_Remod/Add + 998007.000001 t588_leaf2 <= 1e+06
 T588_R2_f9_ge: x_Exter_Cond - 4 t588_leaf2 >= -1
 T588_R3_f3_ge: x_Overall_Qual - 7 t588_leaf3 >= 0
 T588_R3_f6_ge: x_Year_Remod/Add - 1993 t588_leaf3 >= 0
 TREE_589_ONEHOT: t589_leaf0 + t589_leaf1 = 1
 T589_L0_f3_le: x_Overall_Qual + 999997.000001 t589_leaf0 <= 1e+06
 T589_R1_f3_ge: x_Overall_Qual - 3 t589_leaf1 >= 0
 TREE_590_ONEHOT: t590_leaf0 = 1
 TREE_591_ONEHOT: t591_leaf0 = 1
 TREE_592_ONEHOT: t592_leaf0 + t592_leaf1 + t592_leaf2 = 1
 T592_L0_f79_le: x_Land_Contour_Low + 1e-06 t592_leaf0 <= 1
 T592_R1_f79_ge: x_Land_Contour_Low - t592_leaf1 >= 0
 T592_L1_f14_le: x_Bsmt_Unf_SF + 1546.680001 t592_leaf1 <= 1812.68
 T592_R2_f79_ge: x_Land_Contour_Low - t592_leaf2 >= 0
 T592_R2_f14_ge: x_Bsmt_Unf_SF - 266 t592_leaf2 >= 0
 TREE_593_ONEHOT: t593_leaf0 = 1
 TREE_594_ONEHOT: t594_leaf0 + t594_leaf1 = 1
 T594_L0_f292_le: x_Sale_Type_WD_ + 1e-06 t594_leaf0 <= 1
 T594_R1_f292_ge: x_Sale_Type_WD_ - t594_leaf1 >= 0
 TREE_595_ONEHOT: t595_leaf0 = 1
 TREE_596_ONEHOT: t596_leaf0 = 1
 TREE_597_ONEHOT: t597_leaf0 + t597_leaf1 = 1
 T597_L0_f7_le: x_Mas_Vnr_Area + 887.070001 t597_leaf0 <= 931.07
 T597_R1_f7_ge: x_Mas_Vnr_Area - 44 t597_leaf1 >= 0
 TREE_598_ONEHOT: t598_leaf0 + t598_leaf1 = 1
 T598_L0_f5_le: x_Year_Built + 998085.000001 t598_leaf0 <= 1e+06
 T598_R1_f5_ge: x_Year_Built - 1915 t598_leaf1 >= 0
 TREE_599_ONEHOT: t599_leaf0 + t599_leaf1 + t599_leaf2 + t599_leaf3
   + t599_leaf4 = 1
 T599_L0_f31_le: x_Garage_Yr_Blt + 997992.000001 t599_leaf0 <= 1e+06
 T599_L0_f3_le: x_Overall_Qual + 999993.000001 t599_leaf0 <= 1e+06
 T599_L0_f20_le: x_Gr_Liv_Area + 2433.760001 t599_leaf0 <= 3421.76
 T599_L1_f31_le: x_Garage_Yr_Blt + 997992.000001 t599_leaf1 <= 1e+06
 T599_L1_f3_le: x_Overall_Qual + 999993.000001 t599_leaf1 <= 1e+06
 T599_R1_f20_ge: x_Gr_Liv_Area - 988 t599_leaf1 >= 0
 T599_L2_f31_le: x_Garage_Yr_Blt + 997992.000001 t599_leaf2 <= 1e+06
 T599_R2_f3_ge: x_Overall_Qual - 7 t599_leaf2 >= 0
 T599_L2_f8_le: x_Exter_Qual + 1e-06 t599_leaf2 <= 4
 T599_L3_f31_le: x_Garage_Yr_Blt + 997992.000001 t599_leaf3 <= 1e+06
 T599_R3_f3_ge: x_Overall_Qual - 7 t599_leaf3 >= 0
 T599_R3_f8_ge: x_Exter_Qual - 2 t599_leaf3 >= 2
 T599_R4_f31_ge: x_Garage_Yr_Blt - 2008 t599_leaf4 >= 0
 TREE_600_ONEHOT: t600_leaf0 + t600_leaf1 + t600_leaf2 + t600_leaf3 = 1
 T600_L0_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t600_leaf0 <= 1
 T600_L0_f1_le: x_Lot_Area <= 11024.999999
 T600_L1_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t600_leaf1 <= 1
 T600_R1_f1_ge: x_Lot_Area - 4025 t600_leaf1 >= 7000
 T600_L1_f15_le: x_Total_Bsmt_SF + 1016.480001 t600_leaf1 <= 2514.48
 T600_L2_f216_le: x_BsmtFin_Type_1_Rec + 1e-06 t600_leaf2 <= 1
 T600_R2_f1_ge: x_Lot_Area - 4025 t600_leaf2 >= 7000
 T600_R2_f15_ge: x_Total_Bsmt_SF - 1498 t600_leaf2 >= 0
 T600_R3_f216_ge: x_BsmtFin_Type_1_Rec - t600_leaf3 >= 0
 TREE_601_ONEHOT: t601_leaf0 = 1
 TREE_602_ONEHOT: t602_leaf0 + t602_leaf1 + t602_leaf2 + t602_leaf3
   + t602_leaf4 = 1
 T602_L0_f114_le: x_Neighborhood_StoneBr + 1e-06 t602_leaf0 <= 1
 T602_L0_f30_le: x_Fireplace_Qu + 2.000001 t602_leaf0 <= 4
 T602_L0_f134_le: x_Bldg_Type_1Fam + 1e-06 t602_leaf0 <= 1
 T602_L1_f114_le: x_Neighborhood_StoneBr + 1e-06 t602_leaf1 <= 1
 T602_L1_f30_le: x_Fireplace_Qu + 2.000001 t602_leaf1 <= 4
 T602_R1_f134_ge: x_Bldg_Type_1Fam - t602_leaf1 >= 0
 T602_L2_f114_le: x_Neighborhood_StoneBr + 1e-06 t602_leaf2 <= 1
 T602_R2_f30_ge: x_Fireplace_Qu >= 2
 T602_L2_f117_le: x_Condition_1_Artery + 1e-06 t602_leaf2 <= 1
 T602_L3_f114_le: x_Neighborhood_StoneBr + 1e-06 t602_leaf3 <= 1
 T602_R3_f30_ge: x_Fireplace_Qu >= 2
 T602_R3_f117_ge: x_Condition_1_Artery - t602_leaf3 >= 0
 T602_R4_f114_ge: x_Neighborhood_StoneBr - t602_leaf4 >= 0
 TREE_603_ONEHOT: t603_leaf0 + t603_leaf1 + t603_leaf2 + t603_leaf3 = 1
 T603_L0_f251_le: x_Garage_Type_Detchd + 1e-06 t603_leaf0 <= 1
 T603_L0_f6_le: x_Year_Remod/Add + 998045.000001 t603_leaf0 <= 1e+06
 T603_L0_f87_le: x_Land_Slope_Mod + 1e-06 t603_leaf0 <= 1
 T603_L1_f251_le: x_Garage_Type_Detchd + 1e-06 t603_leaf1 <= 1
 T603_L1_f6_le: x_Year_Remod/Add + 998045.000001 t603_leaf1 <= 1e+06
 T603_R1_f87_ge: x_Land_Slope_Mod - t603_leaf1 >= 0
 T603_L2_f251_le: x_Garage_Type_Detchd + 1e-06 t603_leaf2 <= 1
 T603_R2_f6_ge: x_Year_Remod/Add - 1955 t603_leaf2 >= 0
 T603_R3_f251_ge: x_Garage_Type_Detchd - t603_leaf3 >= 0
 TREE_604_ONEHOT: t604_leaf0 + t604_leaf1 + t604_leaf2 = 1
 T604_L0_f38_le: x_Enclosed_Porch + 44.750001 t604_leaf0 <= 292.75
 T604_L0_f38_le: x_Enclosed_Porch + 56.750001 t604_leaf0 <= 292.75
 T604_L1_f38_le: x_Enclosed_Porch + 44.750001 t604_leaf1 <= 292.75
 T604_R1_f38_ge: x_Enclosed_Porch - 236 t604_leaf1 >= 0
 T604_R2_f38_ge: x_Enclosed_Porch - 248 t604_leaf2 >= 0
 TREE_605_ONEHOT: t605_leaf0 + t605_leaf1 = 1
 T605_L0_f17_le: x_1st_Flr_SF + 19.90000099999997 t605_leaf0 <= 2534.9
 T605_R1_f17_ge: x_1st_Flr_SF - 2515 t605_leaf1 >= 0
 TREE_606_ONEHOT: t606_leaf0 = 1
 TREE_607_ONEHOT: t607_leaf0 + t607_leaf1 + t607_leaf2 = 1
 T607_L0_f147_le: x_Roof_Style_Flat + 1e-06 t607_leaf0 <= 1
 T607_L0_f38_le: x_Enclosed_Porch + 33.750001 t607_leaf0 <= 292.75
 T607_L1_f147_le: x_Roof_Style_Flat + 1e-06 t607_leaf1 <= 1
 T607_R1_f38_ge: x_Enclosed_Porch - 259 t607_leaf1 >= 0
 T607_R2_f147_ge: x_Roof_Style_Flat - t607_leaf2 >= 0
 TREE_608_ONEHOT: t608_leaf0 + t608_leaf1 + t608_leaf2 + t608_leaf3 = 1
 T608_L0_f60_le: x_MS_SubClass_190 + 1e-06 t608_leaf0 <= 1
 T608_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t608_leaf0 <= 1
 T608_L0_f18_le: x_2nd_Flr_SF + 1116.540001 t608_leaf0 <= 1641.54
 T608_L1_f60_le: x_MS_SubClass_190 + 1e-06 t608_leaf1 <= 1
 T608_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t608_leaf1 <= 1
 T608_R1_f18_ge: x_2nd_Flr_SF - 525 t608_leaf1 >= 0
 T608_L2_f60_le: x_MS_SubClass_190 + 1e-06 t608_leaf2 <= 1
 T608_R2_f297_ge: x_Sale_Condition_Normal - t608_leaf2 >= 0
 T608_R3_f60_ge: x_MS_SubClass_190 - t608_leaf3 >= 0
 TREE_609_ONEHOT: t609_leaf0 + t609_leaf1 + t609_leaf2 + t609_leaf3 = 1
 T609_L0_f38_le: x_Enclosed_Porch + 260.750001 t609_leaf0 <= 292.75
 T609_R1_f38_ge: x_Enclosed_Porch - 32 t609_leaf1 >= 0
 T609_L1_f261_le: x_Fence_GdWo + 1e-06 t609_leaf1 <= 1
 T609_L1_f15_le: x_Total_Bsmt_SF + 1816.480001 t609_leaf1 <= 2514.48
 T609_R2_f38_ge: x_Enclosed_Porch - 32 t609_leaf2 >= 0
 T609_L2_f261_le: x_Fence_GdWo + 1e-06 t609_leaf2 <= 1
 T609_R2_f15_ge: x_Total_Bsmt_SF - 698 t609_leaf2 >= 0
 T609_R3_f38_ge: x_Enclosed_Porch - 32 t609_leaf3 >= 0
 T609_R3_f261_ge: x_Fence_GdWo - t609_leaf3 >= 0
 TREE_610_ONEHOT: t610_leaf0 = 1
 TREE_611_ONEHOT: t611_leaf0 + t611_leaf1 = 1
 T611_L0_f8_le: x_Exter_Qual + 2.000001 t611_leaf0 <= 4
 T611_R1_f8_ge: x_Exter_Qual >= 2
 TREE_612_ONEHOT: t612_leaf0 = 1
 TREE_613_ONEHOT: t613_leaf0 + t613_leaf1 + t613_leaf2 + t613_leaf3
   + t613_leaf4 = 1
 T613_L0_f24_le: x_Half_Bath + 4.000001 t613_leaf0 <= 5
 T613_L0_f3_le: x_Overall_Qual + 999997.000001 t613_leaf0 <= 1e+06
 T613_L1_f24_le: x_Half_Bath + 4.000001 t613_leaf1 <= 5
 T613_R1_f3_ge: x_Overall_Qual - 3 t613_leaf1 >= 0
 T613_L1_f150_le: x_Roof_Style_Hip + 1e-06 t613_leaf1 <= 1
 T613_L2_f24_le: x_Half_Bath + 4.000001 t613_leaf2 <= 5
 T613_R2_f3_ge: x_Overall_Qual - 3 t613_leaf2 >= 0
 T613_R2_f150_ge: x_Roof_Style_Hip - t613_leaf2 >= 0
 T613_R3_f24_ge: x_Half_Bath - t613_leaf3 >= 0
 T613_L3_f192_le: x_Exterior_2nd_Wd_Sdng + 1e-06 t613_leaf3 <= 1
 T613_R4_f24_ge: x_Half_Bath - t613_leaf4 >= 0
 T613_R4_f192_ge: x_Exterior_2nd_Wd_Sdng - t613_leaf4 >= 0
 TREE_614_ONEHOT: t614_leaf0 + t614_leaf1 = 1
 T614_L0_f0_le: x_Lot_Frontage + 45.530001 t614_leaf0 <= 175.53
 T614_R1_f0_ge: x_Lot_Frontage - 130 t614_leaf1 >= 0
 TREE_615_ONEHOT: t615_leaf0 + t615_leaf1 = 1
 T615_L0_f5_le: x_Year_Built + 997995.000001 t615_leaf0 <= 1e+06
 T615_R1_f5_ge: x_Year_Built - 2005 t615_leaf1 >= 0
 TREE_616_ONEHOT: t616_leaf0 + t616_leaf1 = 1
 T616_L0_f8_le: x_Exter_Qual + 2.000001 t616_leaf0 <= 4
 T616_R1_f8_ge: x_Exter_Qual >= 2
 TREE_617_ONEHOT: t617_leaf0 + t617_leaf1 + t617_leaf2 + t617_leaf3 = 1
 T617_L0_f1_le: x_Lot_Area <= 21779.999999
 T617_L0_f232_le: x_Central_Air_Y + 1e-06 t617_leaf0 <= 1
 T617_L1_f1_le: x_Lot_Area <= 21779.999999
 T617_R1_f232_ge: x_Central_Air_Y - t617_leaf1 >= 0
 T617_R2_f1_ge: x_Lot_Area - 14780 t617_leaf2 >= 7000
 T617_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t617_leaf2 <= 1
 T617_R3_f1_ge: x_Lot_Area - 14780 t617_leaf3 >= 7000
 T617_R3_f297_ge: x_Sale_Condition_Normal - t617_leaf3 >= 0
 TREE_618_ONEHOT: t618_leaf0 + t618_leaf1 = 1
 T618_L0_f17_le: x_1st_Flr_SF + 696.9000010000002 t618_leaf0 <= 2534.9
 T618_R1_f17_ge: x_1st_Flr_SF - 1838 t618_leaf1 >= 0
 TREE_619_ONEHOT: t619_leaf0 + t619_leaf1 + t619_leaf2 = 1
 T619_L0_f1_le: x_Lot_Area + 1000.000001 t619_leaf0 <= 7000
 T619_R1_f1_ge: x_Lot_Area >= 6000
 T619_L1_f114_le: x_Neighborhood_StoneBr + 1e-06 t619_leaf1 <= 1
 T619_R2_f1_ge: x_Lot_Area >= 6000
 T619_R2_f114_ge: x_Neighborhood_StoneBr - t619_leaf2 >= 0
 TREE_620_ONEHOT: t620_leaf0 + t620_leaf1 + t620_leaf2 + t620_leaf3 = 1
 T620_L0_f7_le: x_Mas_Vnr_Area + 427.070001 t620_leaf0 <= 931.07
 T620_L0_f6_le: x_Year_Remod/Add + 998003.000001 t620_leaf0 <= 1e+06
 T620_L1_f7_le: x_Mas_Vnr_Area + 427.070001 t620_leaf1 <= 931.07
 T620_R1_f6_ge: x_Year_Remod/Add - 1997 t620_leaf1 >= 0
 T620_L1_f1_le: x_Lot_Area <= 20430.999999
 T620_L2_f7_le: x_Mas_Vnr_Area + 427.070001 t620_leaf2 <= 931.07
 T620_R2_f6_ge: x_Year_Remod/Add - 1997 t620_leaf2 >= 0
 T620_R2_f1_ge: x_Lot_Area - 13431 t620_leaf2 >= 7000
 T620_R3_f7_ge: x_Mas_Vnr_Area - 504 t620_leaf3 >= 0
 TREE_621_ONEHOT: t621_leaf0 + t621_leaf1 = 1
 T621_L0_f5_le: x_Year_Built + 997995.000001 t621_leaf0 <= 1e+06
 T621_R1_f5_ge: x_Year_Built - 2005 t621_leaf1 >= 0
 TREE_622_ONEHOT: t622_leaf0 + t622_leaf1 + t622_leaf2 + t622_leaf3 = 1
 T622_L0_f20_le: x_Gr_Liv_Area + 772.7600010000001 t622_leaf0 <= 3421.76
 T622_L0_f30_le: x_Fireplace_Qu + 2.000001 t622_leaf0 <= 4
 T622_L1_f20_le: x_Gr_Liv_Area + 772.7600010000001 t622_leaf1 <= 3421.76
 T622_R1_f30_ge: x_Fireplace_Qu >= 2
 T622_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t622_leaf1 <= 1
 T622_L2_f20_le: x_Gr_Liv_Area + 772.7600010000001 t622_leaf2 <= 3421.76
 T622_R2_f30_ge: x_Fireplace_Qu >= 2
 T622_R2_f293_ge: x_Sale_Condition_Abnorml - t622_leaf2 >= 0
 T622_R3_f20_ge: x_Gr_Liv_Area - 2649 t622_leaf3 >= 0
 TREE_623_ONEHOT: t623_leaf0 + t623_leaf1 + t623_leaf2 + t623_leaf3 = 1
 T623_L0_f44_le: x_Yr_Sold + 997991.000001 t623_leaf0 <= 1e+06
 T623_R1_f44_ge: x_Yr_Sold - 2009 t623_leaf1 >= 0
 T623_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t623_leaf1 <= 1
 T623_L1_f292_le: x_Sale_Type_WD_ + 1e-06 t623_leaf1 <= 1
 T623_R2_f44_ge: x_Yr_Sold - 2009 t623_leaf2 >= 0
 T623_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t623_leaf2 <= 1
 T623_R2_f292_ge: x_Sale_Type_WD_ - t623_leaf2 >= 0
 T623_R3_f44_ge: x_Yr_Sold - 2009 t623_leaf3 >= 0
 T623_R3_f297_ge: x_Sale_Condition_Normal - t623_leaf3 >= 0
 TREE_624_ONEHOT: t624_leaf0 + t624_leaf1 = 1
 T624_L0_f18_le: x_2nd_Flr_SF + 760.540001 t624_leaf0 <= 1641.54
 T624_R1_f18_ge: x_2nd_Flr_SF - 881 t624_leaf1 >= 0
 TREE_625_ONEHOT: t625_leaf0 + t625_leaf1 = 1
 T625_L0_f10_le: x_Bsmt_Qual + 4.000001 t625_leaf0 <= 4
 T625_R1_f10_ge: x_Bsmt_Qual >= 0
 TREE_626_ONEHOT: t626_leaf0 + t626_leaf1 + t626_leaf2 + t626_leaf3 = 1
 T626_L0_f9_le: x_Exter_Cond + 2.000001 t626_leaf0 <= 4
 T626_L0_f1_le: x_Lot_Area <= 9041.999999
 T626_L1_f9_le: x_Exter_Cond + 2.000001 t626_leaf1 <= 4
 T626_R1_f1_ge: x_Lot_Area - 2042 t626_leaf1 >= 7000
 T626_L1_f20_le: x_Gr_Liv_Area + 2139.760001000001 t626_leaf1 <= 3421.76
 T626_L2_f9_le: x_Exter_Cond + 2.000001 t626_leaf2 <= 4
 T626_R2_f1_ge: x_Lot_Area - 2042 t626_leaf2 >= 7000
 T626_R2_f20_ge: x_Gr_Liv_Area - 1282 t626_leaf2 >= 0
 T626_R3_f9_ge: x_Exter_Cond - 3 t626_leaf3 >= -1
 TREE_627_ONEHOT: t627_leaf0 + t627_leaf1 + t627_leaf2 + t627_leaf3 = 1
 T627_L0_f20_le: x_Gr_Liv_Area + 2485.760001 t627_leaf0 <= 3421.76
 T627_L0_f20_le: x_Gr_Liv_Area + 2611.760001 t627_leaf0 <= 3421.76
 T627_L0_f1_le: x_Lot_Area + 1638.000001 t627_leaf0 <= 7000
 T627_L1_f20_le: x_Gr_Liv_Area + 2485.760001 t627_leaf1 <= 3421.76
 T627_L1_f20_le: x_Gr_Liv_Area + 2611.760001 t627_leaf1 <= 3421.76
 T627_R1_f1_ge: x_Lot_Area >= 5362
 T627_L2_f20_le: x_Gr_Liv_Area + 2485.760001 t627_leaf2 <= 3421.76
 T627_R2_f20_ge: x_Gr_Liv_Area - 810 t627_leaf2 >= 0
 T627_R3_f20_ge: x_Gr_Liv_Area - 936 t627_leaf3 >= 0
 TREE_628_ONEHOT: t628_leaf0 + t628_leaf1 + t628_leaf2 + t628_leaf3 = 1
 T628_L0_f139_le: x_House_Style_1.5Fin + 1e-06 t628_leaf0 <= 1
 T628_L0_f3_le: x_Overall_Qual + 999996.000001 t628_leaf0 <= 1e+06
 T628_L1_f139_le: x_House_Style_1.5Fin + 1e-06 t628_leaf1 <= 1
 T628_R1_f3_ge: x_Overall_Qual - 4 t628_leaf1 >= 0
 T628_R2_f139_ge: x_House_Style_1.5Fin - t628_leaf2 >= 0
 T628_L2_f251_le: x_Garage_Type_Detchd + 1e-06 t628_leaf2 <= 1
 T628_R3_f139_ge: x_House_Style_1.5Fin - t628_leaf3 >= 0
 T628_R3_f251_ge: x_Garage_Type_Detchd - t628_leaf3 >= 0
 TREE_629_ONEHOT: t629_leaf0 = 1
 TREE_630_ONEHOT: t630_leaf0 = 1
 TREE_631_ONEHOT: t631_leaf0 + t631_leaf1 + t631_leaf2 + t631_leaf3 = 1
 T631_L0_f63_le: x_MS_Zoning_FV + 1e-06 t631_leaf0 <= 1
 T631_L0_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t631_leaf0 <= 1
 T631_L0_f38_le: x_Enclosed_Porch + 253.750001 t631_leaf0 <= 292.75
 T631_L1_f63_le: x_MS_Zoning_FV + 1e-06 t631_leaf1 <= 1
 T631_L1_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t631_leaf1 <= 1
 T631_R1_f38_ge: x_Enclosed_Porch - 39 t631_leaf1 >= 0
 T631_L2_f63_le: x_MS_Zoning_FV + 1e-06 t631_leaf2 <= 1
 T631_R2_f207_ge: x_Bsmt_Exposure_Gd - t631_leaf2 >= 0
 T631_R3_f63_ge: x_MS_Zoning_FV - t631_leaf3 >= 0
 TREE_632_ONEHOT: t632_leaf0 + t632_leaf1 + t632_leaf2 + t632_leaf3
   + t632_leaf4 = 1
 T632_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t632_leaf0 <= 1
 T632_L0_f1_le: x_Lot_Area + 880.0000010000003 t632_leaf0 <= 7000
 T632_L1_f161_le: x_Exterior_1st_AsbShng + 1e-06 t632_leaf1 <= 1
 T632_R1_f1_ge: x_Lot_Area >= 6120
 T632_L1_f114_le: x_Neighborhood_StoneBr + 1e-06 t632_leaf1 <= 1
 T632_L2_f161_le: x_Exterior_1st_AsbShng + 1e-06 t632_leaf2 <= 1
 T632_R2_f1_ge: x_Lot_Area >= 6120
 T632_R2_f114_ge: x_Neighborhood_StoneBr - t632_leaf2 >= 0
 T632_R3_f161_ge: x_Exterior_1st_AsbShng - t632_leaf3 >= 0
 T632_L3_f1_le: x_Lot_Area <= 9604.999999
 T632_R4_f161_ge: x_Exterior_1st_AsbShng - t632_leaf4 >= 0
 T632_R4_f1_ge: x_Lot_Area - 2605 t632_leaf4 >= 7000
 TREE_633_ONEHOT: t633_leaf0 = 1
 TREE_634_ONEHOT: t634_leaf0 + t634_leaf1 + t634_leaf2 = 1
 T634_L0_f30_le: x_Fireplace_Qu + 1.000001 t634_leaf0 <= 4
 T634_R1_f30_ge: x_Fireplace_Qu - t634_leaf1 >= 2
 T634_L1_f31_le: x_Garage_Yr_Blt + 998079.000001 t634_leaf1 <= 1e+06
 T634_R2_f30_ge: x_Fireplace_Qu - t634_leaf2 >= 2
 T634_R2_f31_ge: x_Garage_Yr_Blt - 1921 t634_leaf2 >= 0
 TREE_635_ONEHOT: t635_leaf0 + t635_leaf1 + t635_leaf2 = 1
 T635_L0_f37_le: x_Open_Porch_SF + 459.810001 t635_leaf0 <= 488.81
 T635_R1_f37_ge: x_Open_Porch_SF - 29 t635_leaf1 >= 0
 T635_L1_f3_le: x_Overall_Qual + 999995.000001 t635_leaf1 <= 1e+06
 T635_R2_f37_ge: x_Open_Porch_SF - 29 t635_leaf2 >= 0
 T635_R2_f3_ge: x_Overall_Qual - 5 t635_leaf2 >= 0
 TREE_636_ONEHOT: t636_leaf0 = 1
 TREE_637_ONEHOT: t637_leaf0 + t637_leaf1 = 1
 T637_L0_f7_le: x_Mas_Vnr_Area + 364.070001 t637_leaf0 <= 931.07
 T637_R1_f7_ge: x_Mas_Vnr_Area - 567 t637_leaf1 >= 0
 TREE_638_ONEHOT: t638_leaf0 + t638_leaf1 = 1
 T638_L0_f0_le: x_Lot_Frontage + 76.530001 t638_leaf0 <= 175.53
 T638_R1_f0_ge: x_Lot_Frontage - 99 t638_leaf1 >= 0
 TREE_639_ONEHOT: t639_leaf0 = 1
 TREE_640_ONEHOT: t640_leaf0 + t640_leaf1 + t640_leaf2 = 1
 T640_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t640_leaf0 <= 1
 T640_R1_f161_ge: x_Exterior_1st_AsbShng - t640_leaf1 >= 0
 T640_L1_f20_le: x_Gr_Liv_Area + 2538.760001 t640_leaf1 <= 3421.76
 T640_R2_f161_ge: x_Exterior_1st_AsbShng - t640_leaf2 >= 0
 T640_R2_f20_ge: x_Gr_Liv_Area - 883 t640_leaf2 >= 0
 TREE_641_ONEHOT: t641_leaf0 = 1
 TREE_642_ONEHOT: t642_leaf0 = 1
 TREE_643_ONEHOT: t643_leaf0 + t643_leaf1 = 1
 T643_L0_f4_le: x_Overall_Cond + 999997.000001 t643_leaf0 <= 1e+06
 T643_R1_f4_ge: x_Overall_Cond - 3 t643_leaf1 >= 0
 TREE_644_ONEHOT: t644_leaf0 = 1
 TREE_645_ONEHOT: t645_leaf0 = 1
 TREE_646_ONEHOT: t646_leaf0 + t646_leaf1 + t646_leaf2 + t646_leaf3 = 1
 T646_L0_f4_le: x_Overall_Cond + 999994.000001 t646_leaf0 <= 1e+06
 T646_R1_f4_ge: x_Overall_Cond - 6 t646_leaf1 >= 0
 T646_L1_f23_le: x_Full_Bath + 4.000001 t646_leaf1 <= 6
 T646_L1_f25_le: x_Bedroom_AbvGr + 4.000001 t646_leaf1 <= 7
 T646_R2_f4_ge: x_Overall_Cond - 6 t646_leaf2 >= 0
 T646_L2_f23_le: x_Full_Bath + 4.000001 t646_leaf2 <= 6
 T646_R2_f25_ge: x_Bedroom_AbvGr - 3 t646_leaf2 >= 0
 T646_R3_f4_ge: x_Overall_Cond - 6 t646_leaf3 >= 0
 T646_R3_f23_ge: x_Full_Bath - 2 t646_leaf3 >= 0
 TREE_647_ONEHOT: t647_leaf0 + t647_leaf1 + t647_leaf2 + t647_leaf3
   + t647_leaf4 + t647_leaf5 = 1
 T647_L0_f20_le: x_Gr_Liv_Area + 2545.760001 t647_leaf0 <= 3421.76
 T647_L0_f148_le: x_Roof_Style_Gable + 1e-06 t647_leaf0 <= 1
 T647_L1_f20_le: x_Gr_Liv_Area + 2545.760001 t647_leaf1 <= 3421.76
 T647_R1_f148_ge: x_Roof_Style_Gable - t647_leaf1 >= 0
 T647_L1_f73_le: x_Lot_Shape_IR1 + 1e-06 t647_leaf1 <= 1
 T647_L2_f20_le: x_Gr_Liv_Area + 2545.760001 t647_leaf2 <= 3421.76
 T647_R2_f148_ge: x_Roof_Style_Gable - t647_leaf2 >= 0
 T647_R2_f73_ge: x_Lot_Shape_IR1 - t647_leaf2 >= 0
 T647_R3_f20_ge: x_Gr_Liv_Area - 876 t647_leaf3 >= 0
 T647_L3_f207_le: x_Bsmt_Exposure_Gd + 1e-06 t647_leaf3 <= 1
 T647_R4_f20_ge: x_Gr_Liv_Area - 876 t647_leaf4 >= 0
 T647_R4_f207_ge: x_Bsmt_Exposure_Gd - t647_leaf4 >= 0
 T647_L4_f7_le: x_Mas_Vnr_Area + 749.070001 t647_leaf4 <= 931.07
 T647_R5_f20_ge: x_Gr_Liv_Area - 876 t647_leaf5 >= 0
 T647_R5_f207_ge: x_Bsmt_Exposure_Gd - t647_leaf5 >= 0
 T647_R5_f7_ge: x_Mas_Vnr_Area - 182 t647_leaf5 >= 0
 TREE_648_ONEHOT: t648_leaf0 = 1
 TREE_649_ONEHOT: t649_leaf0 + t649_leaf1 = 1
 T649_L0_f219_le: x_BsmtFin_Type_2_BLQ + 1e-06 t649_leaf0 <= 1
 T649_R1_f219_ge: x_BsmtFin_Type_2_BLQ - t649_leaf1 >= 0
 TREE_650_ONEHOT: t650_leaf0 + t650_leaf1 + t650_leaf2 = 1
 T650_L0_f4_le: x_Overall_Cond + 999996.000001 t650_leaf0 <= 1e+06
 T650_R1_f4_ge: x_Overall_Cond - 4 t650_leaf1 >= 0
 T650_L1_f5_le: x_Year_Built + 998089.000001 t650_leaf1 <= 1e+06
 T650_R2_f4_ge: x_Overall_Cond - 4 t650_leaf2 >= 0
 T650_R2_f5_ge: x_Year_Built - 1911 t650_leaf2 >= 0
 TREE_651_ONEHOT: t651_leaf0 = 1
 TREE_652_ONEHOT: t652_leaf0 = 1
 TREE_653_ONEHOT: t653_leaf0 + t653_leaf1 = 1
 T653_L0_f0_le: x_Lot_Frontage + 45.530001 t653_leaf0 <= 175.53
 T653_R1_f0_ge: x_Lot_Frontage - 130 t653_leaf1 >= 0
 TREE_654_ONEHOT: t654_leaf0 = 1
 TREE_655_ONEHOT: t655_leaf0 + t655_leaf1 + t655_leaf2 = 1
 T655_L0_f38_le: x_Enclosed_Porch + 56.750001 t655_leaf0 <= 292.75
 T655_R1_f38_ge: x_Enclosed_Porch - 236 t655_leaf1 >= 0
 T655_L1_f209_le: x_Bsmt_Exposure_No + 1e-06 t655_leaf1 <= 1
 T655_R2_f38_ge: x_Enclosed_Porch - 236 t655_leaf2 >= 0
 T655_R2_f209_ge: x_Bsmt_Exposure_No - t655_leaf2 >= 0
 TREE_656_ONEHOT: t656_leaf0 + t656_leaf1 = 1
 T656_L0_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t656_leaf0 <= 1e+06
 T656_R1_f19_ge: x_Low_Qual_Fin_SF - 108 t656_leaf1 >= 0
 TREE_657_ONEHOT: t657_leaf0 + t657_leaf1 = 1
 T657_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t657_leaf0 <= 1
 T657_R1_f164_ge: x_Exterior_1st_BrkFace - t657_leaf1 >= 0
 TREE_658_ONEHOT: t658_leaf0 = 1
 TREE_659_ONEHOT: t659_leaf0 + t659_leaf1 + t659_leaf2 + t659_leaf3
   + t659_leaf4 = 1
 T659_L0_f1_le: x_Lot_Area <= 17540.999999
 T659_L0_f33_le: x_Garage_Area + 890.750001 t659_leaf0 <= 1163.75
 T659_L0_f20_le: x_Gr_Liv_Area + 1712.760001 t659_leaf0 <= 3421.76
 T659_L1_f1_le: x_Lot_Area <= 17540.999999
 T659_L1_f33_le: x_Garage_Area + 890.750001 t659_leaf1 <= 1163.75
 T659_R1_f20_ge: x_Gr_Liv_Area - 1709 t659_leaf1 >= 0
 T659_L2_f1_le: x_Lot_Area <= 17540.999999
 T659_R2_f33_ge: x_Garage_Area - 273 t659_leaf2 >= 0
 T659_R3_f1_ge: x_Lot_Area - 10541 t659_leaf3 >= 7000
 T659_L3_f166_le: x_Exterior_1st_CemntBd + 1e-06 t659_leaf3 <= 1
 T659_R4_f1_ge: x_Lot_Area - 10541 t659_leaf4 >= 7000
 T659_R4_f166_ge: x_Exterior_1st_CemntBd - t659_leaf4 >= 0
 TREE_660_ONEHOT: t660_leaf0 + t660_leaf1 + t660_leaf2 + t660_leaf3 = 1
 T660_L0_f0_le: x_Lot_Frontage + 35.530001 t660_leaf0 <= 175.53
 T660_L0_f37_le: x_Open_Porch_SF + 363.810001 t660_leaf0 <= 488.81
 T660_L0_f1_le: x_Lot_Area <= 11764.999999
 T660_L1_f0_le: x_Lot_Frontage + 35.530001 t660_leaf1 <= 175.53
 T660_L1_f37_le: x_Open_Porch_SF + 363.810001 t660_leaf1 <= 488.81
 T660_R1_f1_ge: x_Lot_Area - 4765 t660_leaf1 >= 7000
 T660_L2_f0_le: x_Lot_Frontage + 35.530001 t660_leaf2 <= 175.53
 T660_R2_f37_ge: x_Open_Porch_SF - 125 t660_leaf2 >= 0
 T660_R3_f0_ge: x_Lot_Frontage - 140 t660_leaf3 >= 0
 TREE_661_ONEHOT: t661_leaf0 + t661_leaf1 + t661_leaf2 = 1
 T661_L0_f17_le: x_1st_Flr_SF + 1958.900001 t661_leaf0 <= 2534.9
 T661_R1_f17_ge: x_1st_Flr_SF - 576 t661_leaf1 >= 0
 T661_L1_f31_le: x_Garage_Yr_Blt + 998076.000001 t661_leaf1 <= 1e+06
 T661_R2_f17_ge: x_1st_Flr_SF - 576 t661_leaf2 >= 0
 T661_R2_f31_ge: x_Garage_Yr_Blt - 1924 t661_leaf2 >= 0
 TREE_662_ONEHOT: t662_leaf0 = 1
 TREE_663_ONEHOT: t663_leaf0 + t663_leaf1 + t663_leaf2 + t663_leaf3
   + t663_leaf4 = 1
 T663_L0_f5_le: x_Year_Built + 998059.000001 t663_leaf0 <= 1e+06
 T663_L0_f119_le: x_Condition_1_Norm + 1e-06 t663_leaf0 <= 1
 T663_L1_f5_le: x_Year_Built + 998059.000001 t663_leaf1 <= 1e+06
 T663_R1_f119_ge: x_Condition_1_Norm - t663_leaf1 >= 0
 T663_L1_f109_le: x_Neighborhood_OldTown + 1e-06 t663_leaf1 <= 1
 T663_L2_f5_le: x_Year_Built + 998059.000001 t663_leaf2 <= 1e+06
 T663_R2_f119_ge: x_Condition_1_Norm - t663_leaf2 >= 0
 T663_R2_f109_ge: x_Neighborhood_OldTown - t663_leaf2 >= 0
 T663_R3_f5_ge: x_Year_Built - 1941 t663_leaf3 >= 0
 T663_L3_f27_le: x_Kitchen_Qual + 2.000001 t663_leaf3 <= 4
 T663_R4_f5_ge: x_Year_Built - 1941 t663_leaf4 >= 0
 T663_R4_f27_ge: x_Kitchen_Qual >= 2
 TREE_664_ONEHOT: t664_leaf0 = 1
 TREE_665_ONEHOT: t665_leaf0 + t665_leaf1 + t665_leaf2 + t665_leaf3
   + t665_leaf4 = 1
 T665_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t665_leaf0 <= 1
 T665_L0_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t665_leaf0 <= 1
 T665_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t665_leaf1 <= 1
 T665_R1_f214_ge: x_BsmtFin_Type_1_LwQ - t665_leaf1 >= 0
 T665_R2_f293_ge: x_Sale_Condition_Abnorml - t665_leaf2 >= 0
 T665_L2_f20_le: x_Gr_Liv_Area + 2545.760001 t665_leaf2 <= 3421.76
 T665_R3_f293_ge: x_Sale_Condition_Abnorml - t665_leaf3 >= 0
 T665_R3_f20_ge: x_Gr_Liv_Area - 876 t665_leaf3 >= 0
 T665_L3_f1_le: x_Lot_Area <= 9919.999999
 T665_R4_f293_ge: x_Sale_Condition_Abnorml - t665_leaf4 >= 0
 T665_R4_f20_ge: x_Gr_Liv_Area - 876 t665_leaf4 >= 0
 T665_R4_f1_ge: x_Lot_Area - 2920 t665_leaf4 >= 7000
 TREE_666_ONEHOT: t666_leaf0 = 1
 TREE_667_ONEHOT: t667_leaf0 = 1
 TREE_668_ONEHOT: t668_leaf0 + t668_leaf1 + t668_leaf2 = 1
 T668_L0_f17_le: x_1st_Flr_SF + 752.9000010000002 t668_leaf0 <= 2534.9
 T668_R1_f17_ge: x_1st_Flr_SF - 1782 t668_leaf1 >= 0
 T668_L1_f18_le: x_2nd_Flr_SF + 685.540001 t668_leaf1 <= 1641.54
 T668_R2_f17_ge: x_1st_Flr_SF - 1782 t668_leaf2 >= 0
 T668_R2_f18_ge: x_2nd_Flr_SF - 956 t668_leaf2 >= 0
 TREE_669_ONEHOT: t669_leaf0 = 1
 TREE_670_ONEHOT: t670_leaf0 + t670_leaf1 + t670_leaf2 + t670_leaf3 = 1
 T670_L0_f96_le: x_Neighborhood_Edwards + 1e-06 t670_leaf0 <= 1
 T670_R1_f96_ge: x_Neighborhood_Edwards - t670_leaf1 >= 0
 T670_L1_f7_le: x_Mas_Vnr_Area + 847.070001 t670_leaf1 <= 931.07
 T670_L1_f11_le: x_Bsmt_Cond + 1.000001 t670_leaf1 <= 4
 T670_R2_f96_ge: x_Neighborhood_Edwards - t670_leaf2 >= 0
 T670_L2_f7_le: x_Mas_Vnr_Area + 847.070001 t670_leaf2 <= 931.07
 T670_R2_f11_ge: x_Bsmt_Cond - 4 t670_leaf2 >= -1
 T670_R3_f96_ge: x_Neighborhood_Edwards - t670_leaf3 >= 0
 T670_R3_f7_ge: x_Mas_Vnr_Area - 84 t670_leaf3 >= 0
 TREE_671_ONEHOT: t671_leaf0 + t671_leaf1 + t671_leaf2 = 1
 T671_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t671_leaf0 <= 1
 T671_L0_f30_le: x_Fireplace_Qu + 2.000001 t671_leaf0 <= 4
 T671_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t671_leaf1 <= 1
 T671_R1_f30_ge: x_Fireplace_Qu >= 2
 T671_R2_f164_ge: x_Exterior_1st_BrkFace - t671_leaf2 >= 0
 TREE_672_ONEHOT: t672_leaf0 = 1
 TREE_673_ONEHOT: t673_leaf0 + t673_leaf1 + t673_leaf2 + t673_leaf3
   + t673_leaf4 = 1
 T673_L0_f4_le: x_Overall_Cond + 999996.000001 t673_leaf0 <= 1e+06
 T673_L0_f14_le: x_Bsmt_Unf_SF + 1398.680001 t673_leaf0 <= 1812.68
 T673_L1_f4_le: x_Overall_Cond + 999996.000001 t673_leaf1 <= 1e+06
 T673_R1_f14_ge: x_Bsmt_Unf_SF - 414 t673_leaf1 >= 0
 T673_R2_f4_ge: x_Overall_Cond - 4 t673_leaf2 >= 0
 T673_L2_f15_le: x_Total_Bsmt_SF + 1938.480001 t673_leaf2 <= 2514.48
 T673_L2_f9_le: x_Exter_Cond + 2.000001 t673_leaf2 <= 4
 T673_R3_f4_ge: x_Overall_Cond - 4 t673_leaf3 >= 0
 T673_L3_f15_le: x_Total_Bsmt_SF + 1938.480001 t673_leaf3 <= 2514.48
 T673_R3_f9_ge: x_Exter_Cond - 3 t673_leaf3 >= -1
 T673_R4_f4_ge: x_Overall_Cond - 4 t673_leaf4 >= 0
 T673_R4_f15_ge: x_Total_Bsmt_SF - 576 t673_leaf4 >= 0
 TREE_674_ONEHOT: t674_leaf0 = 1
 TREE_675_ONEHOT: t675_leaf0 = 1
 TREE_676_ONEHOT: t676_leaf0 = 1
 TREE_677_ONEHOT: t677_leaf0 = 1
 TREE_678_ONEHOT: t678_leaf0 + t678_leaf1 = 1
 T678_L0_f38_le: x_Enclosed_Porch + 27.750001 t678_leaf0 <= 292.75
 T678_R1_f38_ge: x_Enclosed_Porch - 265 t678_leaf1 >= 0
 TREE_679_ONEHOT: t679_leaf0 = 1
 TREE_680_ONEHOT: t680_leaf0 + t680_leaf1 + t680_leaf2 + t680_leaf3 = 1
 T680_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t680_leaf0 <= 1
 T680_L0_f3_le: x_Overall_Qual + 999990.000001 t680_leaf0 <= 1e+06
 T680_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t680_leaf1 <= 1
 T680_R1_f3_ge: x_Overall_Qual - 10 t680_leaf1 >= 0
 T680_R2_f177_ge: x_Exterior_2nd_AsbShng - t680_leaf2 >= 0
 T680_L2_f15_le: x_Total_Bsmt_SF + 1787.480001 t680_leaf2 <= 2514.48
 T680_R3_f177_ge: x_Exterior_2nd_AsbShng - t680_leaf3 >= 0
 T680_R3_f15_ge: x_Total_Bsmt_SF - 727 t680_leaf3 >= 0
 TREE_681_ONEHOT: t681_leaf0 + t681_leaf1 + t681_leaf2 + t681_leaf3 = 1
 T681_L0_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t681_leaf0 <= 2514.48
 T681_L0_f261_le: x_Fence_GdWo + 1e-06 t681_leaf0 <= 1
 T681_L1_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t681_leaf1 <= 2514.48
 T681_R1_f261_ge: x_Fence_GdWo - t681_leaf1 >= 0
 T681_L1_f15_le: x_Total_Bsmt_SF + 1620.480001 t681_leaf1 <= 2514.48
 T681_L2_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t681_leaf2 <= 2514.48
 T681_R2_f261_ge: x_Fence_GdWo - t681_leaf2 >= 0
 T681_R2_f15_ge: x_Total_Bsmt_SF - 894 t681_leaf2 >= 0
 T681_R3_f15_ge: x_Total_Bsmt_SF - 2223 t681_leaf3 >= 0
 TREE_682_ONEHOT: t682_leaf0 + t682_leaf1 + t682_leaf2 = 1
 T682_L0_f5_le: x_Year_Built + 998089.000001 t682_leaf0 <= 1e+06
 T682_L0_f10_le: x_Bsmt_Qual + 2.000001 t682_leaf0 <= 4
 T682_L1_f5_le: x_Year_Built + 998089.000001 t682_leaf1 <= 1e+06
 T682_R1_f10_ge: x_Bsmt_Qual >= 2
 T682_R2_f5_ge: x_Year_Built - 1911 t682_leaf2 >= 0
 TREE_683_ONEHOT: t683_leaf0 + t683_leaf1 = 1
 T683_L0_f15_le: x_Total_Bsmt_SF + 1201.480001 t683_leaf0 <= 2514.48
 T683_R1_f15_ge: x_Total_Bsmt_SF - 1313 t683_leaf1 >= 0
 TREE_684_ONEHOT: t684_leaf0 = 1
 TREE_685_ONEHOT: t685_leaf0 = 1
 TREE_686_ONEHOT: t686_leaf0 = 1
 TREE_687_ONEHOT: t687_leaf0 + t687_leaf1 + t687_leaf2 + t687_leaf3
   + t687_leaf4 + t687_leaf5 = 1
 T687_L0_f5_le: x_Year_Built + 997994.000001 t687_leaf0 <= 1e+06
 T687_L0_f3_le: x_Overall_Qual + 999997.000001 t687_leaf0 <= 1e+06
 T687_L1_f5_le: x_Year_Built + 997994.000001 t687_leaf1 <= 1e+06
 T687_R1_f3_ge: x_Overall_Qual - 3 t687_leaf1 >= 0
 T687_L1_f3_le: x_Overall_Qual + 999995.000001 t687_leaf1 <= 1e+06
 T687_L2_f5_le: x_Year_Built + 997994.000001 t687_leaf2 <= 1e+06
 T687_R2_f3_ge: x_Overall_Qual - 3 t687_leaf2 >= 0
 T687_R2_f3_ge: x_Overall_Qual - 5 t687_leaf2 >= 0
 T687_R3_f5_ge: x_Year_Built - 2006 t687_leaf3 >= 0
 T687_L3_f14_le: x_Bsmt_Unf_SF + 1329.680001 t687_leaf3 <= 1812.68
 T687_L3_f275_le: x_Mo_Sold_5 + 1e-06 t687_leaf3 <= 1
 T687_R4_f5_ge: x_Year_Built - 2006 t687_leaf4 >= 0
 T687_L4_f14_le: x_Bsmt_Unf_SF + 1329.680001 t687_leaf4 <= 1812.68
 T687_R4_f275_ge: x_Mo_Sold_5 - t687_leaf4 >= 0
 T687_R5_f5_ge: x_Year_Built - 2006 t687_leaf5 >= 0
 T687_R5_f14_ge: x_Bsmt_Unf_SF - 483 t687_leaf5 >= 0
 TREE_688_ONEHOT: t688_leaf0 + t688_leaf1 + t688_leaf2 + t688_leaf3 = 1
 T688_L0_f164_le: x_Exterior_1st_BrkFace + 1e-06 t688_leaf0 <= 1
 T688_L0_f3_le: x_Overall_Qual + 999992.000001 t688_leaf0 <= 1e+06
 T688_L0_f15_le: x_Total_Bsmt_SF + 1740.480001 t688_leaf0 <= 2514.48
 T688_L1_f164_le: x_Exterior_1st_BrkFace + 1e-06 t688_leaf1 <= 1
 T688_L1_f3_le: x_Overall_Qual + 999992.000001 t688_leaf1 <= 1e+06
 T688_R1_f15_ge: x_Total_Bsmt_SF - 774 t688_leaf1 >= 0
 T688_L2_f164_le: x_Exterior_1st_BrkFace + 1e-06 t688_leaf2 <= 1
 T688_R2_f3_ge: x_Overall_Qual - 8 t688_leaf2 >= 0
 T688_R3_f164_ge: x_Exterior_1st_BrkFace - t688_leaf3 >= 0
 TREE_689_ONEHOT: t689_leaf0 + t689_leaf1 = 1
 T689_L0_f5_le: x_Year_Built + 998086.000001 t689_leaf0 <= 1e+06
 T689_R1_f5_ge: x_Year_Built - 1914 t689_leaf1 >= 0
 TREE_690_ONEHOT: t690_leaf0 = 1
 TREE_691_ONEHOT: t691_leaf0 = 1
 TREE_692_ONEHOT: t692_leaf0 + t692_leaf1 = 1
 T692_L0_f0_le: x_Lot_Frontage + 23.530001 t692_leaf0 <= 175.53
 T692_R1_f0_ge: x_Lot_Frontage - 152 t692_leaf1 >= 0
 TREE_693_ONEHOT: t693_leaf0 + t693_leaf1 = 1
 T693_L0_f40_le: x_Screen_Porch + 72.27000099999998 t693_leaf0 <= 268.27
 T693_R1_f40_ge: x_Screen_Porch - 196 t693_leaf1 >= 0
 TREE_694_ONEHOT: t694_leaf0 = 1
 TREE_695_ONEHOT: t695_leaf0 = 1
 TREE_696_ONEHOT: t696_leaf0 + t696_leaf1 = 1
 T696_L0_f1_le: x_Lot_Area <= 18385.999999
 T696_R1_f1_ge: x_Lot_Area - 11386 t696_leaf1 >= 7000
 TREE_697_ONEHOT: t697_leaf0 = 1
 TREE_698_ONEHOT: t698_leaf0 + t698_leaf1 = 1
 T698_L0_f33_le: x_Garage_Area + 9.750001000000111 t698_leaf0 <= 1163.75
 T698_R1_f33_ge: x_Garage_Area - 1154 t698_leaf1 >= 0
 TREE_699_ONEHOT: t699_leaf0 + t699_leaf1 + t699_leaf2 + t699_leaf3
   + t699_leaf4 = 1
 T699_L0_f5_le: x_Year_Built + 998076.000001 t699_leaf0 <= 1e+06
 T699_L0_f4_le: x_Overall_Cond + 999992.000001 t699_leaf0 <= 1e+06
 T699_L1_f5_le: x_Year_Built + 998076.000001 t699_leaf1 <= 1e+06
 T699_R1_f4_ge: x_Overall_Cond - 8 t699_leaf1 >= 0
 T699_R2_f5_ge: x_Year_Built - 1924 t699_leaf2 >= 0
 T699_L2_f232_le: x_Central_Air_Y + 1e-06 t699_leaf2 <= 1
 T699_L2_f86_le: x_Land_Slope_Gtl + 1e-06 t699_leaf2 <= 1
 T699_R3_f5_ge: x_Year_Built - 1924 t699_leaf3 >= 0
 T699_L3_f232_le: x_Central_Air_Y + 1e-06 t699_leaf3 <= 1
 T699_R3_f86_ge: x_Land_Slope_Gtl - t699_leaf3 >= 0
 T699_R4_f5_ge: x_Year_Built - 1924 t699_leaf4 >= 0
 T699_R4_f232_ge: x_Central_Air_Y - t699_leaf4 >= 0
 TREE_700_ONEHOT: t700_leaf0 + t700_leaf1 + t700_leaf2 = 1
 T700_L0_f0_le: x_Lot_Frontage + 54.530001 t700_leaf0 <= 175.53
 T700_R1_f0_ge: x_Lot_Frontage - 121 t700_leaf1 >= 0
 T700_L1_f3_le: x_Overall_Qual + 999992.000001 t700_leaf1 <= 1e+06
 T700_R2_f0_ge: x_Lot_Frontage - 121 t700_leaf2 >= 0
 T700_R2_f3_ge: x_Overall_Qual - 8 t700_leaf2 >= 0
 TREE_701_ONEHOT: t701_leaf0 + t701_leaf1 = 1
 T701_L0_f3_le: x_Overall_Qual + 999997.000001 t701_leaf0 <= 1e+06
 T701_R1_f3_ge: x_Overall_Qual - 3 t701_leaf1 >= 0
 TREE_702_ONEHOT: t702_leaf0 + t702_leaf1 + t702_leaf2 + t702_leaf3
   + t702_leaf4 + t702_leaf5 = 1
 T702_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t702_leaf0 <= 1
 T702_L0_f5_le: x_Year_Built + 998068.000001 t702_leaf0 <= 1e+06
 T702_L0_f20_le: x_Gr_Liv_Area + 2153.760001000001 t702_leaf0 <= 3421.76
 T702_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t702_leaf1 <= 1
 T702_L1_f5_le: x_Year_Built + 998068.000001 t702_leaf1 <= 1e+06
 T702_R1_f20_ge: x_Gr_Liv_Area - 1268 t702_leaf1 >= 0
 T702_L2_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t702_leaf2 <= 1
 T702_R2_f5_ge: x_Year_Built - 1932 t702_leaf2 >= 0
 T702_L2_f21_le: x_Bsmt_Full_Bath + 0.020001 t702_leaf2 <= 2.02
 T702_L3_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t702_leaf3 <= 1
 T702_R3_f5_ge: x_Year_Built - 1932 t702_leaf3 >= 0
 T702_R3_f21_ge: x_Bsmt_Full_Bath - 2 t702_leaf3 >= 0
 T702_R4_f177_ge: x_Exterior_2nd_AsbShng - t702_leaf4 >= 0
 T702_L4_f4_le: x_Overall_Cond + 999995.000001 t702_leaf4 <= 1e+06
 T702_R5_f177_ge: x_Exterior_2nd_AsbShng - t702_leaf5 >= 0
 T702_R5_f4_ge: x_Overall_Cond - 5 t702_leaf5 >= 0
 TREE_703_ONEHOT: t703_leaf0 + t703_leaf1 + t703_leaf2 + t703_leaf3
   + t703_leaf4 = 1
 T703_L0_f15_le: x_Total_Bsmt_SF + 2033.480001 t703_leaf0 <= 2514.48
 T703_L0_f192_le: x_Exterior_2nd_Wd_Sdng + 1e-06 t703_leaf0 <= 1
 T703_L1_f15_le: x_Total_Bsmt_SF + 2033.480001 t703_leaf1 <= 2514.48
 T703_R1_f192_ge: x_Exterior_2nd_Wd_Sdng - t703_leaf1 >= 0
 T703_R2_f15_ge: x_Total_Bsmt_SF - 481 t703_leaf2 >= 0
 T703_L2_f96_le: x_Neighborhood_Edwards + 1e-06 t703_leaf2 <= 1
 T703_L2_f33_le: x_Garage_Area + 275.750001 t703_leaf2 <= 1163.75
 T703_R3_f15_ge: x_Total_Bsmt_SF - 481 t703_leaf3 >= 0
 T703_L3_f96_le: x_Neighborhood_Edwards + 1e-06 t703_leaf3 <= 1
 T703_R3_f33_ge: x_Garage_Area - 888 t703_leaf3 >= 0
 T703_R4_f15_ge: x_Total_Bsmt_SF - 481 t703_leaf4 >= 0
 T703_R4_f96_ge: x_Neighborhood_Edwards - t703_leaf4 >= 0
 TREE_704_ONEHOT: t704_leaf0 = 1
 TREE_705_ONEHOT: t705_leaf0 + t705_leaf1 + t705_leaf2 = 1
 T705_L0_f1_le: x_Lot_Area <= 9199.999999
 T705_R1_f1_ge: x_Lot_Area - 2200 t705_leaf1 >= 7000
 T705_L1_f33_le: x_Garage_Area + 948.750001 t705_leaf1 <= 1163.75
 T705_R2_f1_ge: x_Lot_Area - 2200 t705_leaf2 >= 7000
 T705_R2_f33_ge: x_Garage_Area - 215 t705_leaf2 >= 0
 TREE_706_ONEHOT: t706_leaf0 + t706_leaf1 + t706_leaf2 + t706_leaf3
   + t706_leaf4 = 1
 T706_L0_f3_le: x_Overall_Qual + 999992.000001 t706_leaf0 <= 1e+06
 T706_L0_f11_le: x_Bsmt_Cond + 1.000001 t706_leaf0 <= 4
 T706_L1_f3_le: x_Overall_Qual + 999992.000001 t706_leaf1 <= 1e+06
 T706_R1_f11_ge: x_Bsmt_Cond - 4 t706_leaf1 >= -1
 T706_R2_f3_ge: x_Overall_Qual - 8 t706_leaf2 >= 0
 T706_L2_f37_le: x_Open_Porch_SF + 423.810001 t706_leaf2 <= 488.81
 T706_L2_f113_le: x_Neighborhood_Somerst + 1e-06 t706_leaf2 <= 1
 T706_R3_f3_ge: x_Overall_Qual - 8 t706_leaf3 >= 0
 T706_L3_f37_le: x_Open_Porch_SF + 423.810001 t706_leaf3 <= 488.81
 T706_R3_f113_ge: x_Neighborhood_Somerst - t706_leaf3 >= 0
 T706_R4_f3_ge: x_Overall_Qual - 8 t706_leaf4 >= 0
 T706_R4_f37_ge: x_Open_Porch_SF - 65 t706_leaf4 >= 0
 TREE_707_ONEHOT: t707_leaf0 = 1
 TREE_708_ONEHOT: t708_leaf0 + t708_leaf1 + t708_leaf2 + t708_leaf3
   + t708_leaf4 = 1
 T708_L0_f114_le: x_Neighborhood_StoneBr + 1e-06 t708_leaf0 <= 1
 T708_L0_f33_le: x_Garage_Area + 760.750001 t708_leaf0 <= 1163.75
 T708_L0_f119_le: x_Condition_1_Norm + 1e-06 t708_leaf0 <= 1
 T708_L1_f114_le: x_Neighborhood_StoneBr + 1e-06 t708_leaf1 <= 1
 T708_L1_f33_le: x_Garage_Area + 760.750001 t708_leaf1 <= 1163.75
 T708_R1_f119_ge: x_Condition_1_Norm - t708_leaf1 >= 0
 T708_L2_f114_le: x_Neighborhood_StoneBr + 1e-06 t708_leaf2 <= 1
 T708_R2_f33_ge: x_Garage_Area - 403 t708_leaf2 >= 0
 T708_L2_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t708_leaf2 <= 1
 T708_L3_f114_le: x_Neighborhood_StoneBr + 1e-06 t708_leaf3 <= 1
 T708_R3_f33_ge: x_Garage_Area - 403 t708_leaf3 >= 0
 T708_R3_f177_ge: x_Exterior_2nd_AsbShng - t708_leaf3 >= 0
 T708_R4_f114_ge: x_Neighborhood_StoneBr - t708_leaf4 >= 0
 TREE_709_ONEHOT: t709_leaf0 = 1
 TREE_710_ONEHOT: t710_leaf0 + t710_leaf1 + t710_leaf2 = 1
 T710_L0_f38_le: x_Enclosed_Porch + 274.750001 t710_leaf0 <= 292.75
 T710_R1_f38_ge: x_Enclosed_Porch - 18 t710_leaf1 >= 0
 T710_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t710_leaf1 <= 1
 T710_R2_f38_ge: x_Enclosed_Porch - 18 t710_leaf2 >= 0
 T710_R2_f297_ge: x_Sale_Condition_Normal - t710_leaf2 >= 0
 TREE_711_ONEHOT: t711_leaf0 = 1
 TREE_712_ONEHOT: t712_leaf0 + t712_leaf1 + t712_leaf2 + t712_leaf3
   + t712_leaf4 = 1
 T712_L0_f95_le: x_Neighborhood_Crawfor + 1e-06 t712_leaf0 <= 1
 T712_L0_f33_le: x_Garage_Area + 890.750001 t712_leaf0 <= 1163.75
 T712_L0_f1_le: x_Lot_Area <= 8126.999999
 T712_L1_f95_le: x_Neighborhood_Crawfor + 1e-06 t712_leaf1 <= 1
 T712_L1_f33_le: x_Garage_Area + 890.750001 t712_leaf1 <= 1163.75
 T712_R1_f1_ge: x_Lot_Area - 1127 t712_leaf1 >= 7000
 T712_L2_f95_le: x_Neighborhood_Crawfor + 1e-06 t712_leaf2 <= 1
 T712_R2_f33_ge: x_Garage_Area - 273 t712_leaf2 >= 0
 T712_L2_f17_le: x_1st_Flr_SF + 19.90000099999997 t712_leaf2 <= 2534.9
 T712_L3_f95_le: x_Neighborhood_Crawfor + 1e-06 t712_leaf3 <= 1
 T712_R3_f33_ge: x_Garage_Area - 273 t712_leaf3 >= 0
 T712_R3_f17_ge: x_1st_Flr_SF - 2515 t712_leaf3 >= 0
 T712_R4_f95_ge: x_Neighborhood_Crawfor - t712_leaf4 >= 0
 TREE_713_ONEHOT: t713_leaf0 + t713_leaf1 + t713_leaf2 + t713_leaf3
   + t713_leaf4 = 1
 T713_L0_f8_le: x_Exter_Qual + 1e-06 t713_leaf0 <= 4
 T713_L0_f3_le: x_Overall_Qual + 999996.000001 t713_leaf0 <= 1e+06
 T713_L0_f44_le: x_Yr_Sold + 997993.000001 t713_leaf0 <= 1e+06
 T713_L1_f8_le: x_Exter_Qual + 1e-06 t713_leaf1 <= 4
 T713_L1_f3_le: x_Overall_Qual + 999996.000001 t713_leaf1 <= 1e+06
 T713_R1_f44_ge: x_Yr_Sold - 2007 t713_leaf1 >= 0
 T713_L2_f8_le: x_Exter_Qual + 1e-06 t713_leaf2 <= 4
 T713_R2_f3_ge: x_Overall_Qual - 4 t713_leaf2 >= 0
 T713_L2_f5_le: x_Year_Built + 997992.000001 t713_leaf2 <= 1e+06
 T713_L3_f8_le: x_Exter_Qual + 1e-06 t713_leaf3 <= 4
 T713_R3_f3_ge: x_Overall_Qual - 4 t713_leaf3 >= 0
 T713_R3_f5_ge: x_Year_Built - 2008 t713_leaf3 >= 0
 T713_R4_f8_ge: x_Exter_Qual - 2 t713_leaf4 >= 2
 TREE_714_ONEHOT: t714_leaf0 + t714_leaf1 = 1
 T714_L0_f0_le: x_Lot_Frontage + 45.530001 t714_leaf0 <= 175.53
 T714_R1_f0_ge: x_Lot_Frontage - 130 t714_leaf1 >= 0
 TREE_715_ONEHOT: t715_leaf0 + t715_leaf1 + t715_leaf2 + t715_leaf3
   + t715_leaf4 = 1
 T715_L0_f1_le: x_Lot_Area <= 11791.999999
 T715_R1_f1_ge: x_Lot_Area - 4792 t715_leaf1 >= 7000
 T715_L1_f35_le: x_Garage_Cond + 2.000001 t715_leaf1 <= 4
 T715_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t715_leaf1 <= 1
 T715_R2_f1_ge: x_Lot_Area - 4792 t715_leaf2 >= 7000
 T715_L2_f35_le: x_Garage_Cond + 2.000001 t715_leaf2 <= 4
 T715_R2_f297_ge: x_Sale_Condition_Normal - t715_leaf2 >= 0
 T715_R3_f1_ge: x_Lot_Area - 4792 t715_leaf3 >= 7000
 T715_R3_f35_ge: x_Garage_Cond - 3 t715_leaf3 >= -1
 T715_L3_f6_le: x_Year_Remod/Add + 998031.000001 t715_leaf3 <= 1e+06
 T715_R4_f1_ge: x_Lot_Area - 4792 t715_leaf4 >= 7000
 T715_R4_f35_ge: x_Garage_Cond - 3 t715_leaf4 >= -1
 T715_R4_f6_ge: x_Year_Remod/Add - 1969 t715_leaf4 >= 0
 TREE_716_ONEHOT: t716_leaf0 = 1
 TREE_717_ONEHOT: t717_leaf0 = 1
 TREE_718_ONEHOT: t718_leaf0 + t718_leaf1 + t718_leaf2 + t718_leaf3
   + t718_leaf4 = 1
 T718_L0_f4_le: x_Overall_Cond + 999994.000001 t718_leaf0 <= 1e+06
 T718_L0_f5_le: x_Year_Built + 998058.000001 t718_leaf0 <= 1e+06
 T718_L0_f17_le: x_1st_Flr_SF + 1703.900001 t718_leaf0 <= 2534.9
 T718_L1_f4_le: x_Overall_Cond + 999994.000001 t718_leaf1 <= 1e+06
 T718_L1_f5_le: x_Year_Built + 998058.000001 t718_leaf1 <= 1e+06
 T718_R1_f17_ge: x_1st_Flr_SF - 831 t718_leaf1 >= 0
 T718_L2_f4_le: x_Overall_Cond + 999994.000001 t718_leaf2 <= 1e+06
 T718_R2_f5_ge: x_Year_Built - 1942 t718_leaf2 >= 0
 T718_L2_f27_le: x_Kitchen_Qual + 2.000001 t718_leaf2 <= 4
 T718_L3_f4_le: x_Overall_Cond + 999994.000001 t718_leaf3 <= 1e+06
 T718_R3_f5_ge: x_Year_Built - 1942 t718_leaf3 >= 0
 T718_R3_f27_ge: x_Kitchen_Qual >= 2
 T718_R4_f4_ge: x_Overall_Cond - 6 t718_leaf4 >= 0
 TREE_719_ONEHOT: t719_leaf0 = 1
 TREE_720_ONEHOT: t720_leaf0 = 1
 TREE_721_ONEHOT: t721_leaf0 + t721_leaf1 + t721_leaf2 + t721_leaf3
   + t721_leaf4 = 1
 T721_L0_f37_le: x_Open_Porch_SF + 324.810001 t721_leaf0 <= 488.81
 T721_L0_f121_le: x_Condition_1_PosN + 1e-06 t721_leaf0 <= 1
 T721_L0_f141_le: x_House_Style_1Story + 1e-06 t721_leaf0 <= 1
 T721_L1_f37_le: x_Open_Porch_SF + 324.810001 t721_leaf1 <= 488.81
 T721_L1_f121_le: x_Condition_1_PosN + 1e-06 t721_leaf1 <= 1
 T721_R1_f141_ge: x_House_Style_1Story - t721_leaf1 >= 0
 T721_L2_f37_le: x_Open_Porch_SF + 324.810001 t721_leaf2 <= 488.81
 T721_R2_f121_ge: x_Condition_1_PosN - t721_leaf2 >= 0
 T721_R3_f37_ge: x_Open_Porch_SF - 164 t721_leaf3 >= 0
 T721_L3_f0_le: x_Lot_Frontage + 74.530001 t721_leaf3 <= 175.53
 T721_R4_f37_ge: x_Open_Porch_SF - 164 t721_leaf4 >= 0
 T721_R4_f0_ge: x_Lot_Frontage - 101 t721_leaf4 >= 0
 TREE_722_ONEHOT: t722_leaf0 = 1
 TREE_723_ONEHOT: t723_leaf0 + t723_leaf1 + t723_leaf2 + t723_leaf3
   + t723_leaf4 = 1
 T723_L0_f251_le: x_Garage_Type_Detchd + 1e-06 t723_leaf0 <= 1
 T723_L0_f13_le: x_BsmtFin_SF_2 + 711.050001 t723_leaf0 <= 757.05
 T723_L1_f251_le: x_Garage_Type_Detchd + 1e-06 t723_leaf1 <= 1
 T723_R1_f13_ge: x_BsmtFin_SF_2 - 46 t723_leaf1 >= 0
 T723_R2_f251_ge: x_Garage_Type_Detchd - t723_leaf2 >= 0
 T723_L2_f15_le: x_Total_Bsmt_SF + 1459.480001 t723_leaf2 <= 2514.48
 T723_L2_f1_le: x_Lot_Area <= 8625.999999
 T723_R3_f251_ge: x_Garage_Type_Detchd - t723_leaf3 >= 0
 T723_L3_f15_le: x_Total_Bsmt_SF + 1459.480001 t723_leaf3 <= 2514.48
 T723_R3_f1_ge: x_Lot_Area - 1626 t723_leaf3 >= 7000
 T723_R4_f251_ge: x_Garage_Type_Detchd - t723_leaf4 >= 0
 T723_R4_f15_ge: x_Total_Bsmt_SF - 1055 t723_leaf4 >= 0
 TREE_724_ONEHOT: t724_leaf0 + t724_leaf1 + t724_leaf2 = 1
 T724_L0_f17_le: x_1st_Flr_SF + 1010.900001 t724_leaf0 <= 2534.9
 T724_L0_f7_le: x_Mas_Vnr_Area + 451.070001 t724_leaf0 <= 931.07
 T724_L1_f17_le: x_1st_Flr_SF + 1010.900001 t724_leaf1 <= 2534.9
 T724_R1_f7_ge: x_Mas_Vnr_Area - 480 t724_leaf1 >= 0
 T724_R2_f17_ge: x_1st_Flr_SF - 1524 t724_leaf2 >= 0
 TREE_725_ONEHOT: t725_leaf0 = 1
 TREE_726_ONEHOT: t726_leaf0 + t726_leaf1 = 1
 T726_L0_f158_le: x_Roof_Matl_Tar&Grv + 1e-06 t726_leaf0 <= 1
 T726_R1_f158_ge: x_Roof_Matl_Tar&Grv - t726_leaf1 >= 0
 TREE_727_ONEHOT: t727_leaf0 + t727_leaf1 + t727_leaf2 = 1
 T727_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t727_leaf0 <= 1
 T727_R1_f177_ge: x_Exterior_2nd_AsbShng - t727_leaf1 >= 0
 T727_L1_f3_le: x_Overall_Qual + 999995.000001 t727_leaf1 <= 1e+06
 T727_R2_f177_ge: x_Exterior_2nd_AsbShng - t727_leaf2 >= 0
 T727_R2_f3_ge: x_Overall_Qual - 5 t727_leaf2 >= 0
 TREE_728_ONEHOT: t728_leaf0 = 1
 TREE_729_ONEHOT: t729_leaf0 = 1
 TREE_730_ONEHOT: t730_leaf0 = 1
 TREE_731_ONEHOT: t731_leaf0 = 1
 TREE_732_ONEHOT: t732_leaf0 = 1
 TREE_733_ONEHOT: t733_leaf0 = 1
 TREE_734_ONEHOT: t734_leaf0 = 1
 TREE_735_ONEHOT: t735_leaf0 + t735_leaf1 + t735_leaf2 + t735_leaf3 = 1
 T735_L0_f1_le: x_Lot_Area <= 11791.999999
 T735_L0_f6_le: x_Year_Remod/Add + 997992.000001 t735_leaf0 <= 1e+06
 T735_L1_f1_le: x_Lot_Area <= 11791.999999
 T735_R1_f6_ge: x_Year_Remod/Add - 2008 t735_leaf1 >= 0
 T735_R2_f1_ge: x_Lot_Area - 4792 t735_leaf2 >= 7000
 T735_L2_f170_le: x_Exterior_1st_Plywood + 1e-06 t735_leaf2 <= 1
 T735_R3_f1_ge: x_Lot_Area - 4792 t735_leaf3 >= 7000
 T735_R3_f170_ge: x_Exterior_1st_Plywood - t735_leaf3 >= 0
 TREE_736_ONEHOT: t736_leaf0 = 1
 TREE_737_ONEHOT: t737_leaf0 = 1
 TREE_738_ONEHOT: t738_leaf0 + t738_leaf1 + t738_leaf2 + t738_leaf3
   + t738_leaf4 = 1
 T738_L0_f44_le: x_Yr_Sold + 997992.000001 t738_leaf0 <= 1e+06
 T738_L0_f86_le: x_Land_Slope_Gtl + 1e-06 t738_leaf0 <= 1
 T738_L1_f44_le: x_Yr_Sold + 997992.000001 t738_leaf1 <= 1e+06
 T738_R1_f86_ge: x_Land_Slope_Gtl - t738_leaf1 >= 0
 T738_R2_f44_ge: x_Yr_Sold - 2008 t738_leaf2 >= 0
 T738_L2_f148_le: x_Roof_Style_Gable + 1e-06 t738_leaf2 <= 1
 T738_R3_f44_ge: x_Yr_Sold - 2008 t738_leaf3 >= 0
 T738_R3_f148_ge: x_Roof_Style_Gable - t738_leaf3 >= 0
 T738_L3_f35_le: x_Garage_Cond + 2.000001 t738_leaf3 <= 4
 T738_R4_f44_ge: x_Yr_Sold - 2008 t738_leaf4 >= 0
 T738_R4_f148_ge: x_Roof_Style_Gable - t738_leaf4 >= 0
 T738_R4_f35_ge: x_Garage_Cond - 3 t738_leaf4 >= -1
 TREE_739_ONEHOT: t739_leaf0 = 1
 TREE_740_ONEHOT: t740_leaf0 = 1
 TREE_741_ONEHOT: t741_leaf0 + t741_leaf1 + t741_leaf2 + t741_leaf3
   + t741_leaf4 = 1
 T741_L0_f1_le: x_Lot_Area <= 12333.999999
 T741_R1_f1_ge: x_Lot_Area - 5334 t741_leaf1 >= 7000
 T741_L1_f175_le: x_Exterior_1st_Wd_Sdng + 1e-06 t741_leaf1 <= 1
 T741_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t741_leaf1 <= 1
 T741_R2_f1_ge: x_Lot_Area - 5334 t741_leaf2 >= 7000
 T741_L2_f175_le: x_Exterior_1st_Wd_Sdng + 1e-06 t741_leaf2 <= 1
 T741_R2_f293_ge: x_Sale_Condition_Abnorml - t741_leaf2 >= 0
 T741_R3_f1_ge: x_Lot_Area - 5334 t741_leaf3 >= 7000
 T741_R3_f175_ge: x_Exterior_1st_Wd_Sdng - t741_leaf3 >= 0
 T741_L3_f17_le: x_1st_Flr_SF + 1150.900001 t741_leaf3 <= 2534.9
 T741_R4_f1_ge: x_Lot_Area - 5334 t741_leaf4 >= 7000
 T741_R4_f175_ge: x_Exterior_1st_Wd_Sdng - t741_leaf4 >= 0
 T741_R4_f17_ge: x_1st_Flr_SF - 1384 t741_leaf4 >= 0
 TREE_742_ONEHOT: t742_leaf0 + t742_leaf1 + t742_leaf2 = 1
 T742_L0_f272_le: x_Mo_Sold_2 + 1e-06 t742_leaf0 <= 1
 T742_R1_f272_ge: x_Mo_Sold_2 - t742_leaf1 >= 0
 T742_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t742_leaf1 <= 1
 T742_R2_f272_ge: x_Mo_Sold_2 - t742_leaf2 >= 0
 T742_R2_f297_ge: x_Sale_Condition_Normal - t742_leaf2 >= 0
 TREE_743_ONEHOT: t743_leaf0 = 1
 TREE_744_ONEHOT: t744_leaf0 = 1
 TREE_745_ONEHOT: t745_leaf0 = 1
 TREE_746_ONEHOT: t746_leaf0 = 1
 TREE_747_ONEHOT: t747_leaf0 = 1
 TREE_748_ONEHOT: t748_leaf0 = 1
 TREE_749_ONEHOT: t749_leaf0 + t749_leaf1 + t749_leaf2 = 1
 T749_L0_f259_le: x_Paved_Drive_Y + 1e-06 t749_leaf0 <= 1
 T749_L0_f33_le: x_Garage_Area + 689.750001 t749_leaf0 <= 1163.75
 T749_L1_f259_le: x_Paved_Drive_Y + 1e-06 t749_leaf1 <= 1
 T749_R1_f33_ge: x_Garage_Area - 474 t749_leaf1 >= 0
 T749_R2_f259_ge: x_Paved_Drive_Y - t749_leaf2 >= 0
 TREE_750_ONEHOT: t750_leaf0 = 1
 TREE_751_ONEHOT: t751_leaf0 + t751_leaf1 + t751_leaf2 + t751_leaf3
   + t751_leaf4 + t751_leaf5 = 1
 T751_L0_f0_le: x_Lot_Frontage + 97.530001 t751_leaf0 <= 175.53
 T751_L0_f261_le: x_Fence_GdWo + 1e-06 t751_leaf0 <= 1
 T751_L1_f0_le: x_Lot_Frontage + 97.530001 t751_leaf1 <= 175.53
 T751_R1_f261_ge: x_Fence_GdWo - t751_leaf1 >= 0
 T751_L1_f1_le: x_Lot_Area <= 8404.999999
 T751_L2_f0_le: x_Lot_Frontage + 97.530001 t751_leaf2 <= 175.53
 T751_R2_f261_ge: x_Fence_GdWo - t751_leaf2 >= 0
 T751_R2_f1_ge: x_Lot_Area - 1405 t751_leaf2 >= 7000
 T751_R3_f0_ge: x_Lot_Frontage - 78 t751_leaf3 >= 0
 T751_L3_f103_le: x_Neighborhood_Mitchel + 1e-06 t751_leaf3 <= 1
 T751_L3_f3_le: x_Overall_Qual + 999994.000001 t751_leaf3 <= 1e+06
 T751_R4_f0_ge: x_Lot_Frontage - 78 t751_leaf4 >= 0
 T751_L4_f103_le: x_Neighborhood_Mitchel + 1e-06 t751_leaf4 <= 1
 T751_R4_f3_ge: x_Overall_Qual - 6 t751_leaf4 >= 0
 T751_R5_f0_ge: x_Lot_Frontage - 78 t751_leaf5 >= 0
 T751_R5_f103_ge: x_Neighborhood_Mitchel - t751_leaf5 >= 0
 TREE_752_ONEHOT: t752_leaf0 = 1
 TREE_753_ONEHOT: t753_leaf0 = 1
 TREE_754_ONEHOT: t754_leaf0 = 1
 TREE_755_ONEHOT: t755_leaf0 = 1
 TREE_756_ONEHOT: t756_leaf0 + t756_leaf1 + t756_leaf2 + t756_leaf3 = 1
 T756_L0_f4_le: x_Overall_Cond + 999994.000001 t756_leaf0 <= 1e+06
 T756_L0_f36_le: x_Wood_Deck_SF + 305.900001 t756_leaf0 <= 525.9
 T756_L0_f31_le: x_Garage_Yr_Blt + 998003.000001 t756_leaf0 <= 1e+06
 T756_L1_f4_le: x_Overall_Cond + 999994.000001 t756_leaf1 <= 1e+06
 T756_L1_f36_le: x_Wood_Deck_SF + 305.900001 t756_leaf1 <= 525.9
 T756_R1_f31_ge: x_Garage_Yr_Blt - 1997 t756_leaf1 >= 0
 T756_L2_f4_le: x_Overall_Cond + 999994.000001 t756_leaf2 <= 1e+06
 T756_R2_f36_ge: x_Wood_Deck_SF - 220 t756_leaf2 >= 0
 T756_R3_f4_ge: x_Overall_Cond - 6 t756_leaf3 >= 0
 TREE_757_ONEHOT: t757_leaf0 + t757_leaf1 + t757_leaf2 = 1
 T757_L0_f44_le: x_Yr_Sold + 997993.000001 t757_leaf0 <= 1e+06
 T757_L0_f56_le: x_MS_SubClass_120 + 1e-06 t757_leaf0 <= 1
 T757_L1_f44_le: x_Yr_Sold + 997993.000001 t757_leaf1 <= 1e+06
 T757_R1_f56_ge: x_MS_SubClass_120 - t757_leaf1 >= 0
 T757_R2_f44_ge: x_Yr_Sold - 2007 t757_leaf2 >= 0
 TREE_758_ONEHOT: t758_leaf0 + t758_leaf1 + t758_leaf2 + t758_leaf3
   + t758_leaf4 + t758_leaf5 = 1
 T758_L0_f3_le: x_Overall_Qual + 999997.000001 t758_leaf0 <= 1e+06
 T758_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t758_leaf0 <= 1
 T758_L1_f3_le: x_Overall_Qual + 999997.000001 t758_leaf1 <= 1e+06
 T758_R1_f161_ge: x_Exterior_1st_AsbShng - t758_leaf1 >= 0
 T758_R2_f3_ge: x_Overall_Qual - 3 t758_leaf2 >= 0
 T758_L2_f1_le: x_Lot_Area <= 7949.999999
 T758_L2_f31_le: x_Garage_Yr_Blt + 998081.000001 t758_leaf2 <= 1e+06
 T758_R3_f3_ge: x_Overall_Qual - 3 t758_leaf3 >= 0
 T758_L3_f1_le: x_Lot_Area <= 7949.999999
 T758_R3_f31_ge: x_Garage_Yr_Blt - 1919 t758_leaf3 >= 0
 T758_R4_f3_ge: x_Overall_Qual - 3 t758_leaf4 >= 0
 T758_R4_f1_ge: x_Lot_Area - 950 t758_leaf4 >= 7000
 T758_L4_f36_le: x_Wood_Deck_SF + 433.900001 t758_leaf4 <= 525.9
 T758_R5_f3_ge: x_Overall_Qual - 3 t758_leaf5 >= 0
 T758_R5_f1_ge: x_Lot_Area - 950 t758_leaf5 >= 7000
 T758_R5_f36_ge: x_Wood_Deck_SF - 92 t758_leaf5 >= 0
 TREE_759_ONEHOT: t759_leaf0 = 1
 TREE_760_ONEHOT: t760_leaf0 = 1
 TREE_761_ONEHOT: t761_leaf0 + t761_leaf1 + t761_leaf2 = 1
 T761_L0_f7_le: x_Mas_Vnr_Area + 177.070001 t761_leaf0 <= 931.07
 T761_R1_f7_ge: x_Mas_Vnr_Area - 754 t761_leaf1 >= 0
 T761_L1_f33_le: x_Garage_Area + 352.750001 t761_leaf1 <= 1163.75
 T761_R2_f7_ge: x_Mas_Vnr_Area - 754 t761_leaf2 >= 0
 T761_R2_f33_ge: x_Garage_Area - 811 t761_leaf2 >= 0
 TREE_762_ONEHOT: t762_leaf0 = 1
 TREE_763_ONEHOT: t763_leaf0 = 1
 TREE_764_ONEHOT: t764_leaf0 + t764_leaf1 + t764_leaf2 = 1
 T764_L0_f14_le: x_Bsmt_Unf_SF + 1371.680001 t764_leaf0 <= 1812.68
 T764_L0_f6_le: x_Year_Remod/Add + 998013.000001 t764_leaf0 <= 1e+06
 T764_L1_f14_le: x_Bsmt_Unf_SF + 1371.680001 t764_leaf1 <= 1812.68
 T764_R1_f6_ge: x_Year_Remod/Add - 1987 t764_leaf1 >= 0
 T764_R2_f14_ge: x_Bsmt_Unf_SF - 441 t764_leaf2 >= 0
 TREE_765_ONEHOT: t765_leaf0 = 1
 TREE_766_ONEHOT: t766_leaf0 + t766_leaf1 + t766_leaf2 + t766_leaf3
   + t766_leaf4 = 1
 T766_L0_f14_le: x_Bsmt_Unf_SF + 1009.680001 t766_leaf0 <= 1812.68
 T766_L0_f5_le: x_Year_Built + 998068.000001 t766_leaf0 <= 1e+06
 T766_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t766_leaf0 <= 1
 T766_L1_f14_le: x_Bsmt_Unf_SF + 1009.680001 t766_leaf1 <= 1812.68
 T766_L1_f5_le: x_Year_Built + 998068.000001 t766_leaf1 <= 1e+06
 T766_R1_f297_ge: x_Sale_Condition_Normal - t766_leaf1 >= 0
 T766_L2_f14_le: x_Bsmt_Unf_SF + 1009.680001 t766_leaf2 <= 1812.68
 T766_R2_f5_ge: x_Year_Built - 1932 t766_leaf2 >= 0
 T766_L2_f17_le: x_1st_Flr_SF + 1454.900001 t766_leaf2 <= 2534.9
 T766_L3_f14_le: x_Bsmt_Unf_SF + 1009.680001 t766_leaf3 <= 1812.68
 T766_R3_f5_ge: x_Year_Built - 1932 t766_leaf3 >= 0
 T766_R3_f17_ge: x_1st_Flr_SF - 1080 t766_leaf3 >= 0
 T766_R4_f14_ge: x_Bsmt_Unf_SF - 803 t766_leaf4 >= 0
 TREE_767_ONEHOT: t767_leaf0 = 1
 TREE_768_ONEHOT: t768_leaf0 + t768_leaf1 + t768_leaf2 = 1
 T768_L0_f37_le: x_Open_Porch_SF + 228.810001 t768_leaf0 <= 488.81
 T768_L0_f15_le: x_Total_Bsmt_SF + 548.4800010000001 t768_leaf0 <= 2514.48
 T768_L1_f37_le: x_Open_Porch_SF + 228.810001 t768_leaf1 <= 488.81
 T768_R1_f15_ge: x_Total_Bsmt_SF - 1966 t768_leaf1 >= 0
 T768_R2_f37_ge: x_Open_Porch_SF - 260 t768_leaf2 >= 0
 TREE_769_ONEHOT: t769_leaf0 = 1
 TREE_770_ONEHOT: t770_leaf0 = 1
 TREE_771_ONEHOT: t771_leaf0 = 1
 TREE_772_ONEHOT: t772_leaf0 = 1
 TREE_773_ONEHOT: t773_leaf0 = 1
 TREE_774_ONEHOT: t774_leaf0 + t774_leaf1 = 1
 T774_L0_f3_le: x_Overall_Qual + 999997.000001 t774_leaf0 <= 1e+06
 T774_R1_f3_ge: x_Overall_Qual - 3 t774_leaf1 >= 0
 TREE_775_ONEHOT: t775_leaf0 + t775_leaf1 = 1
 T775_L0_f283_le: x_Sale_Type_COD + 1e-06 t775_leaf0 <= 1
 T775_R1_f283_ge: x_Sale_Type_COD - t775_leaf1 >= 0
 TREE_776_ONEHOT: t776_leaf0 = 1
 TREE_777_ONEHOT: t777_leaf0 + t777_leaf1 = 1
 T777_L0_f12_le: x_BsmtFin_SF_1 + 252.5000010000001 t777_leaf0 <= 1984.5
 T777_R1_f12_ge: x_BsmtFin_SF_1 - 1732 t777_leaf1 >= 0
 TREE_778_ONEHOT: t778_leaf0 + t778_leaf1 + t778_leaf2 + t778_leaf3 = 1
 T778_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t778_leaf0 <= 1
 T778_L0_f62_le: x_MS_Zoning_C_(all) + 1e-06 t778_leaf0 <= 1
 T778_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t778_leaf1 <= 1
 T778_R1_f62_ge: x_MS_Zoning_C_(all) - t778_leaf1 >= 0
 T778_R2_f177_ge: x_Exterior_2nd_AsbShng - t778_leaf2 >= 0
 T778_L2_f233_le: x_Electrical_FuseA + 1e-06 t778_leaf2 <= 1
 T778_R3_f177_ge: x_Exterior_2nd_AsbShng - t778_leaf3 >= 0
 T778_R3_f233_ge: x_Electrical_FuseA - t778_leaf3 >= 0
 TREE_779_ONEHOT: t779_leaf0 = 1
 TREE_780_ONEHOT: t780_leaf0 = 1
 TREE_781_ONEHOT: t781_leaf0 + t781_leaf1 + t781_leaf2 + t781_leaf3 = 1
 T781_L0_f3_le: x_Overall_Qual + 999997.000001 t781_leaf0 <= 1e+06
 T781_R1_f3_ge: x_Overall_Qual - 3 t781_leaf1 >= 0
 T781_L1_f42_le: x_Pool_QC + 2.000001 t781_leaf1 <= 4
 T781_L1_f7_le: x_Mas_Vnr_Area + 786.070001 t781_leaf1 <= 931.07
 T781_R2_f3_ge: x_Overall_Qual - 3 t781_leaf2 >= 0
 T781_L2_f42_le: x_Pool_QC + 2.000001 t781_leaf2 <= 4
 T781_R2_f7_ge: x_Mas_Vnr_Area - 145 t781_leaf2 >= 0
 T781_R3_f3_ge: x_Overall_Qual - 3 t781_leaf3 >= 0
 T781_R3_f42_ge: x_Pool_QC - 3 t781_leaf3 >= -1
 TREE_782_ONEHOT: t782_leaf0 = 1
 TREE_783_ONEHOT: t783_leaf0 = 1
 TREE_784_ONEHOT: t784_leaf0 + t784_leaf1 + t784_leaf2 + t784_leaf3 = 1
 T784_L0_f7_le: x_Mas_Vnr_Area + 258.070001 t784_leaf0 <= 931.07
 T784_L0_f34_le: x_Garage_Qual + 2.000001 t784_leaf0 <= 4
 T784_L0_f3_le: x_Overall_Qual + 999994.000001 t784_leaf0 <= 1e+06
 T784_L1_f7_le: x_Mas_Vnr_Area + 258.070001 t784_leaf1 <= 931.07
 T784_L1_f34_le: x_Garage_Qual + 2.000001 t784_leaf1 <= 4
 T784_R1_f3_ge: x_Overall_Qual - 6 t784_leaf1 >= 0
 T784_L2_f7_le: x_Mas_Vnr_Area + 258.070001 t784_leaf2 <= 931.07
 T784_R2_f34_ge: x_Garage_Qual >= 2
 T784_R3_f7_ge: x_Mas_Vnr_Area - 673 t784_leaf3 >= 0
 TREE_785_ONEHOT: t785_leaf0 = 1
 TREE_786_ONEHOT: t786_leaf0 + t786_leaf1 + t786_leaf2 = 1
 T786_L0_f93_le: x_Neighborhood_ClearCr + 1e-06 t786_leaf0 <= 1
 T786_L0_f4_le: x_Overall_Cond + 999997.000001 t786_leaf0 <= 1e+06
 T786_L1_f93_le: x_Neighborhood_ClearCr + 1e-06 t786_leaf1 <= 1
 T786_R1_f4_ge: x_Overall_Cond - 3 t786_leaf1 >= 0
 T786_R2_f93_ge: x_Neighborhood_ClearCr - t786_leaf2 >= 0
 TREE_787_ONEHOT: t787_leaf0 = 1
 TREE_788_ONEHOT: t788_leaf0 + t788_leaf1 = 1
 T788_L0_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t788_leaf0 <= 2514.48
 T788_R1_f15_ge: x_Total_Bsmt_SF - 2136 t788_leaf1 >= 0
 TREE_789_ONEHOT: t789_leaf0 = 1
 TREE_790_ONEHOT: t790_leaf0 + t790_leaf1 = 1
 T790_L0_f203_le: x_Foundation_Slab + 1e-06 t790_leaf0 <= 1
 T790_R1_f203_ge: x_Foundation_Slab - t790_leaf1 >= 0
 TREE_791_ONEHOT: t791_leaf0 = 1
 TREE_792_ONEHOT: t792_leaf0 = 1
 TREE_793_ONEHOT: t793_leaf0 = 1
 TREE_794_ONEHOT: t794_leaf0 + t794_leaf1 = 1
 T794_L0_f119_le: x_Condition_1_Norm + 1e-06 t794_leaf0 <= 1
 T794_R1_f119_ge: x_Condition_1_Norm - t794_leaf1 >= 0
 TREE_795_ONEHOT: t795_leaf0 + t795_leaf1 + t795_leaf2 + t795_leaf3
   + t795_leaf4 + t795_leaf5 = 1
 T795_L0_f5_le: x_Year_Built + 998058.000001 t795_leaf0 <= 1e+06
 T795_L0_f4_le: x_Overall_Cond + 999993.000001 t795_leaf0 <= 1e+06
 T795_L0_f14_le: x_Bsmt_Unf_SF + 995.6800010000001 t795_leaf0 <= 1812.68
 T795_L1_f5_le: x_Year_Built + 998058.000001 t795_leaf1 <= 1e+06
 T795_L1_f4_le: x_Overall_Cond + 999993.000001 t795_leaf1 <= 1e+06
 T795_R1_f14_ge: x_Bsmt_Unf_SF - 817 t795_leaf1 >= 0
 T795_L2_f5_le: x_Year_Built + 998058.000001 t795_leaf2 <= 1e+06
 T795_R2_f4_ge: x_Overall_Cond - 7 t795_leaf2 >= 0
 T795_R3_f5_ge: x_Year_Built - 1942 t795_leaf3 >= 0
 T795_L3_f3_le: x_Overall_Qual + 999995.000001 t795_leaf3 <= 1e+06
 T795_L3_f36_le: x_Wood_Deck_SF + 429.900001 t795_leaf3 <= 525.9
 T795_R4_f5_ge: x_Year_Built - 1942 t795_leaf4 >= 0
 T795_L4_f3_le: x_Overall_Qual + 999995.000001 t795_leaf4 <= 1e+06
 T795_R4_f36_ge: x_Wood_Deck_SF - 96 t795_leaf4 >= 0
 T795_R5_f5_ge: x_Year_Built - 1942 t795_leaf5 >= 0
 T795_R5_f3_ge: x_Overall_Qual - 5 t795_leaf5 >= 0
 TREE_796_ONEHOT: t796_leaf0 + t796_leaf1 + t796_leaf2 + t796_leaf3
   + t796_leaf4 = 1
 T796_L0_f209_le: x_Bsmt_Exposure_No + 1e-06 t796_leaf0 <= 1
 T796_L0_f43_le: x_Misc_Val + 2975.470001 t796_leaf0 <= 3029.47
 T796_L1_f209_le: x_Bsmt_Exposure_No + 1e-06 t796_leaf1 <= 1
 T796_R1_f43_ge: x_Misc_Val - 54 t796_leaf1 >= 0
 T796_R2_f209_ge: x_Bsmt_Exposure_No - t796_leaf2 >= 0
 T796_L2_f17_le: x_1st_Flr_SF + 1142.900001 t796_leaf2 <= 2534.9
 T796_R3_f209_ge: x_Bsmt_Exposure_No - t796_leaf3 >= 0
 T796_R3_f17_ge: x_1st_Flr_SF - 1392 t796_leaf3 >= 0
 T796_L3_f15_le: x_Total_Bsmt_SF + 1114.480001 t796_leaf3 <= 2514.48
 T796_R4_f209_ge: x_Bsmt_Exposure_No - t796_leaf4 >= 0
 T796_R4_f17_ge: x_1st_Flr_SF - 1392 t796_leaf4 >= 0
 T796_R4_f15_ge: x_Total_Bsmt_SF - 1400 t796_leaf4 >= 0
 TREE_797_ONEHOT: t797_leaf0 = 1
 TREE_798_ONEHOT: t798_leaf0 + t798_leaf1 = 1
 T798_L0_f3_le: x_Overall_Qual + 999997.000001 t798_leaf0 <= 1e+06
 T798_R1_f3_ge: x_Overall_Qual - 3 t798_leaf1 >= 0
 TREE_799_ONEHOT: t799_leaf0 + t799_leaf1 + t799_leaf2 + t799_leaf3 = 1
 T799_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t799_leaf0 <= 1
 T799_R1_f161_ge: x_Exterior_1st_AsbShng - t799_leaf1 >= 0
 T799_L1_f35_le: x_Garage_Cond + 4.000001 t799_leaf1 <= 4
 T799_R2_f161_ge: x_Exterior_1st_AsbShng - t799_leaf2 >= 0
 T799_R2_f35_ge: x_Garage_Cond - t799_leaf2 >= -1
 T799_L2_f20_le: x_Gr_Liv_Area + 2195.760001000001 t799_leaf2 <= 3421.76
 T799_R3_f161_ge: x_Exterior_1st_AsbShng - t799_leaf3 >= 0
 T799_R3_f35_ge: x_Garage_Cond - t799_leaf3 >= -1
 T799_R3_f20_ge: x_Gr_Liv_Area - 1226 t799_leaf3 >= 0
 TREE_800_ONEHOT: t800_leaf0 = 1
 TREE_801_ONEHOT: t801_leaf0 = 1
 TREE_802_ONEHOT: t802_leaf0 = 1
 TREE_803_ONEHOT: t803_leaf0 = 1
 TREE_804_ONEHOT: t804_leaf0 = 1
 TREE_805_ONEHOT: t805_leaf0 = 1
 TREE_806_ONEHOT: t806_leaf0 + t806_leaf1 + t806_leaf2 + t806_leaf3 = 1
 T806_L0_f20_le: x_Gr_Liv_Area + 2485.760001 t806_leaf0 <= 3421.76
 T806_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t806_leaf0 <= 1
 T806_L1_f20_le: x_Gr_Liv_Area + 2485.760001 t806_leaf1 <= 3421.76
 T806_R1_f293_ge: x_Sale_Condition_Abnorml - t806_leaf1 >= 0
 T806_L1_f85_le: x_Lot_Config_Inside + 1e-06 t806_leaf1 <= 1
 T806_L2_f20_le: x_Gr_Liv_Area + 2485.760001 t806_leaf2 <= 3421.76
 T806_R2_f293_ge: x_Sale_Condition_Abnorml - t806_leaf2 >= 0
 T806_R2_f85_ge: x_Lot_Config_Inside - t806_leaf2 >= 0
 T806_R3_f20_ge: x_Gr_Liv_Area - 936 t806_leaf3 >= 0
 TREE_807_ONEHOT: t807_leaf0 + t807_leaf1 = 1
 T807_L0_f68_le: x_Street_Grvl + 1e-06 t807_leaf0 <= 1
 T807_R1_f68_ge: x_Street_Grvl - t807_leaf1 >= 0
 TREE_808_ONEHOT: t808_leaf0 + t808_leaf1 + t808_leaf2 + t808_leaf3 = 1
 T808_L0_f8_le: x_Exter_Qual + 2.000001 t808_leaf0 <= 4
 T808_R1_f8_ge: x_Exter_Qual >= 2
 T808_L1_f20_le: x_Gr_Liv_Area + 2611.760001 t808_leaf1 <= 3421.76
 T808_L1_f1_le: x_Lot_Area + 1638.000001 t808_leaf1 <= 7000
 T808_R2_f8_ge: x_Exter_Qual >= 2
 T808_L2_f20_le: x_Gr_Liv_Area + 2611.760001 t808_leaf2 <= 3421.76
 T808_R2_f1_ge: x_Lot_Area >= 5362
 T808_R3_f8_ge: x_Exter_Qual >= 2
 T808_R3_f20_ge: x_Gr_Liv_Area - 810 t808_leaf3 >= 0
 TREE_809_ONEHOT: t809_leaf0 + t809_leaf1 = 1
 T809_L0_f4_le: x_Overall_Cond + 999997.000001 t809_leaf0 <= 1e+06
 T809_R1_f4_ge: x_Overall_Cond - 3 t809_leaf1 >= 0
 TREE_810_ONEHOT: t810_leaf0 = 1
 TREE_811_ONEHOT: t811_leaf0 = 1
 TREE_812_ONEHOT: t812_leaf0 = 1
 TREE_813_ONEHOT: t813_leaf0 = 1
 TREE_814_ONEHOT: t814_leaf0 + t814_leaf1 + t814_leaf2 + t814_leaf3
   + t814_leaf4 = 1
 T814_L0_f3_le: x_Overall_Qual + 999997.000001 t814_leaf0 <= 1e+06
 T814_R1_f3_ge: x_Overall_Qual - 3 t814_leaf1 >= 0
 T814_L1_f38_le: x_Enclosed_Porch + 56.750001 t814_leaf1 <= 292.75
 T814_L1_f106_le: x_Neighborhood_No_aplicames + 1e-06 t814_leaf1 <= 1
 T814_R2_f3_ge: x_Overall_Qual - 3 t814_leaf2 >= 0
 T814_L2_f38_le: x_Enclosed_Porch + 56.750001 t814_leaf2 <= 292.75
 T814_R2_f106_ge: x_Neighborhood_No_aplicames - t814_leaf2 >= 0
 T814_R3_f3_ge: x_Overall_Qual - 3 t814_leaf3 >= 0
 T814_R3_f38_ge: x_Enclosed_Porch - 236 t814_leaf3 >= 0
 T814_L3_f1_le: x_Lot_Area <= 9749.999999
 T814_R4_f3_ge: x_Overall_Qual - 3 t814_leaf4 >= 0
 T814_R4_f38_ge: x_Enclosed_Porch - 236 t814_leaf4 >= 0
 T814_R4_f1_ge: x_Lot_Area - 2750 t814_leaf4 >= 7000
 TREE_815_ONEHOT: t815_leaf0 + t815_leaf1 = 1
 T815_L0_f40_le: x_Screen_Porch + 158.270001 t815_leaf0 <= 268.27
 T815_R1_f40_ge: x_Screen_Porch - 110 t815_leaf1 >= 0
 TREE_816_ONEHOT: t816_leaf0 = 1
 TREE_817_ONEHOT: t817_leaf0 = 1
 TREE_818_ONEHOT: t818_leaf0 + t818_leaf1 = 1
 T818_L0_f96_le: x_Neighborhood_Edwards + 1e-06 t818_leaf0 <= 1
 T818_R1_f96_ge: x_Neighborhood_Edwards - t818_leaf1 >= 0
 TREE_819_ONEHOT: t819_leaf0 + t819_leaf1 + t819_leaf2 + t819_leaf3 = 1
 T819_L0_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t819_leaf0 <= 1e+06
 T819_L0_f75_le: x_Lot_Shape_IR3 + 1e-06 t819_leaf0 <= 1
 T819_L0_f33_le: x_Garage_Area + 292.750001 t819_leaf0 <= 1163.75
 T819_L1_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t819_leaf1 <= 1e+06
 T819_L1_f75_le: x_Lot_Shape_IR3 + 1e-06 t819_leaf1 <= 1
 T819_R1_f33_ge: x_Garage_Area - 871 t819_leaf1 >= 0
 T819_L2_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t819_leaf2 <= 1e+06
 T819_R2_f75_ge: x_Lot_Shape_IR3 - t819_leaf2 >= 0
 T819_R3_f19_ge: x_Low_Qual_Fin_SF - 108 t819_leaf3 >= 0
 TREE_820_ONEHOT: t820_leaf0 = 1
 TREE_821_ONEHOT: t821_leaf0 = 1
 TREE_822_ONEHOT: t822_leaf0 + t822_leaf1 + t822_leaf2 + t822_leaf3 = 1
 T822_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t822_leaf0 <= 1
 T822_L0_f35_le: x_Garage_Cond + 3.000001 t822_leaf0 <= 4
 T822_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t822_leaf1 <= 1
 T822_R1_f35_ge: x_Garage_Cond - 2 t822_leaf1 >= -1
 T822_L1_f17_le: x_1st_Flr_SF + 1554.900001 t822_leaf1 <= 2534.9
 T822_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t822_leaf2 <= 1
 T822_R2_f35_ge: x_Garage_Cond - 2 t822_leaf2 >= -1
 T822_R2_f17_ge: x_1st_Flr_SF - 980 t822_leaf2 >= 0
 T822_R3_f245_ge: x_FunctioNo_aplical_Typ - t822_leaf3 >= 0
 TREE_823_ONEHOT: t823_leaf0 = 1
 TREE_824_ONEHOT: t824_leaf0 + t824_leaf1 + t824_leaf2 = 1
 T824_L0_f106_le: x_Neighborhood_No_aplicames + 1e-06 t824_leaf0 <= 1
 T824_R1_f106_ge: x_Neighborhood_No_aplicames - t824_leaf1 >= 0
 T824_L1_f17_le: x_1st_Flr_SF + 1738.900001 t824_leaf1 <= 2534.9
 T824_R2_f106_ge: x_Neighborhood_No_aplicames - t824_leaf2 >= 0
 T824_R2_f17_ge: x_1st_Flr_SF - 796 t824_leaf2 >= 0
 TREE_825_ONEHOT: t825_leaf0 = 1
 TREE_826_ONEHOT: t826_leaf0 + t826_leaf1 + t826_leaf2 = 1
 T826_L0_f87_le: x_Land_Slope_Mod + 1e-06 t826_leaf0 <= 1
 T826_L0_f28_le: x_TotRms_AbvGrd + 7.080001 t826_leaf0 <= 12.08
 T826_L1_f87_le: x_Land_Slope_Mod + 1e-06 t826_leaf1 <= 1
 T826_R1_f28_ge: x_TotRms_AbvGrd - 5 t826_leaf1 >= 0
 T826_R2_f87_ge: x_Land_Slope_Mod - t826_leaf2 >= 0
 TREE_827_ONEHOT: t827_leaf0 + t827_leaf1 + t827_leaf2 = 1
 T827_L0_f5_le: x_Year_Built + 998048.000001 t827_leaf0 <= 1e+06
 T827_L0_f9_le: x_Exter_Cond + 1.000001 t827_leaf0 <= 4
 T827_L1_f5_le: x_Year_Built + 998048.000001 t827_leaf1 <= 1e+06
 T827_R1_f9_ge: x_Exter_Cond - 4 t827_leaf1 >= -1
 T827_R2_f5_ge: x_Year_Built - 1952 t827_leaf2 >= 0
 TREE_828_ONEHOT: t828_leaf0 + t828_leaf1 + t828_leaf2 + t828_leaf3 = 1
 T828_L0_f3_le: x_Overall_Qual + 999993.000001 t828_leaf0 <= 1e+06
 T828_L0_f5_le: x_Year_Built + 998003.000001 t828_leaf0 <= 1e+06
 T828_L1_f3_le: x_Overall_Qual + 999993.000001 t828_leaf1 <= 1e+06
 T828_R1_f5_ge: x_Year_Built - 1997 t828_leaf1 >= 0
 T828_R2_f3_ge: x_Overall_Qual - 7 t828_leaf2 >= 0
 T828_L2_f10_le: x_Bsmt_Qual + 2.000001 t828_leaf2 <= 4
 T828_R3_f3_ge: x_Overall_Qual - 7 t828_leaf3 >= 0
 T828_R3_f10_ge: x_Bsmt_Qual >= 2
 TREE_829_ONEHOT: t829_leaf0 + t829_leaf1 + t829_leaf2 + t829_leaf3
   + t829_leaf4 + t829_leaf5 + t829_leaf6 + t829_leaf7 = 1
 T829_L0_f20_le: x_Gr_Liv_Area + 2153.760001000001 t829_leaf0 <= 3421.76
 T829_L0_f15_le: x_Total_Bsmt_SF + 1767.480001 t829_leaf0 <= 2514.48
 T829_L0_f9_le: x_Exter_Cond + 1.000001 t829_leaf0 <= 4
 T829_L1_f20_le: x_Gr_Liv_Area + 2153.760001000001 t829_leaf1 <= 3421.76
 T829_L1_f15_le: x_Total_Bsmt_SF + 1767.480001 t829_leaf1 <= 2514.48
 T829_R1_f9_ge: x_Exter_Cond - 4 t829_leaf1 >= -1
 T829_L2_f20_le: x_Gr_Liv_Area + 2153.760001000001 t829_leaf2 <= 3421.76
 T829_R2_f15_ge: x_Total_Bsmt_SF - 747 t829_leaf2 >= 0
 T829_L2_f17_le: x_1st_Flr_SF + 1731.900001 t829_leaf2 <= 2534.9
 T829_L3_f20_le: x_Gr_Liv_Area + 2153.760001000001 t829_leaf3 <= 3421.76
 T829_R3_f15_ge: x_Total_Bsmt_SF - 747 t829_leaf3 >= 0
 T829_R3_f17_ge: x_1st_Flr_SF - 803 t829_leaf3 >= 0
 T829_R4_f20_ge: x_Gr_Liv_Area - 1268 t829_leaf4 >= 0
 T829_L4_f17_le: x_1st_Flr_SF + 1010.900001 t829_leaf4 <= 2534.9
 T829_L4_f147_le: x_Roof_Style_Flat + 1e-06 t829_leaf4 <= 1
 T829_R5_f20_ge: x_Gr_Liv_Area - 1268 t829_leaf5 >= 0
 T829_L5_f17_le: x_1st_Flr_SF + 1010.900001 t829_leaf5 <= 2534.9
 T829_R5_f147_ge: x_Roof_Style_Flat - t829_leaf5 >= 0
 T829_R6_f20_ge: x_Gr_Liv_Area - 1268 t829_leaf6 >= 0
 T829_R6_f17_ge: x_1st_Flr_SF - 1524 t829_leaf6 >= 0
 T829_L6_f14_le: x_Bsmt_Unf_SF + 1346.680001 t829_leaf6 <= 1812.68
 T829_R7_f20_ge: x_Gr_Liv_Area - 1268 t829_leaf7 >= 0
 T829_R7_f17_ge: x_1st_Flr_SF - 1524 t829_leaf7 >= 0
 T829_R7_f14_ge: x_Bsmt_Unf_SF - 466 t829_leaf7 >= 0
 TREE_830_ONEHOT: t830_leaf0 = 1
 TREE_831_ONEHOT: t831_leaf0 = 1
 TREE_832_ONEHOT: t832_leaf0 + t832_leaf1 = 1
 T832_L0_f3_le: x_Overall_Qual + 999997.000001 t832_leaf0 <= 1e+06
 T832_R1_f3_ge: x_Overall_Qual - 3 t832_leaf1 >= 0
 TREE_833_ONEHOT: t833_leaf0 + t833_leaf1 = 1
 T833_L0_f3_le: x_Overall_Qual + 999997.000001 t833_leaf0 <= 1e+06
 T833_R1_f3_ge: x_Overall_Qual - 3 t833_leaf1 >= 0
 TREE_834_ONEHOT: t834_leaf0 + t834_leaf1 + t834_leaf2 + t834_leaf3 = 1
 T834_L0_f117_le: x_Condition_1_Artery + 1e-06 t834_leaf0 <= 1
 T834_L0_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t834_leaf0 <= 1e+06
 T834_L0_f6_le: x_Year_Remod/Add + 998044.000001 t834_leaf0 <= 1e+06
 T834_L1_f117_le: x_Condition_1_Artery + 1e-06 t834_leaf1 <= 1
 T834_L1_f19_le: x_Low_Qual_Fin_SF + 999892.000001 t834_leaf1 <= 1e+06
 T834_R1_f6_ge: x_Year_Remod/Add - 1956 t834_leaf1 >= 0
 T834_L2_f117_le: x_Condition_1_Artery + 1e-06 t834_leaf2 <= 1
 T834_R2_f19_ge: x_Low_Qual_Fin_SF - 108 t834_leaf2 >= 0
 T834_R3_f117_ge: x_Condition_1_Artery - t834_leaf3 >= 0
 TREE_835_ONEHOT: t835_leaf0 + t835_leaf1 + t835_leaf2 = 1
 T835_L0_f232_le: x_Central_Air_Y + 1e-06 t835_leaf0 <= 1
 T835_L0_f1_le: x_Lot_Area <= 8749.999999
 T835_L1_f232_le: x_Central_Air_Y + 1e-06 t835_leaf1 <= 1
 T835_R1_f1_ge: x_Lot_Area - 1750 t835_leaf1 >= 7000
 T835_R2_f232_ge: x_Central_Air_Y - t835_leaf2 >= 0
 TREE_836_ONEHOT: t836_leaf0 + t836_leaf1 = 1
 T836_L0_f18_le: x_2nd_Flr_SF + 777.540001 t836_leaf0 <= 1641.54
 T836_R1_f18_ge: x_2nd_Flr_SF - 864 t836_leaf1 >= 0
 TREE_837_ONEHOT: t837_leaf0 = 1
 TREE_838_ONEHOT: t838_leaf0 = 1
 TREE_839_ONEHOT: t839_leaf0 + t839_leaf1 = 1
 T839_L0_f147_le: x_Roof_Style_Flat + 1e-06 t839_leaf0 <= 1
 T839_R1_f147_ge: x_Roof_Style_Flat - t839_leaf1 >= 0
 TREE_840_ONEHOT: t840_leaf0 = 1
 TREE_841_ONEHOT: t841_leaf0 + t841_leaf1 = 1
 T841_L0_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t841_leaf0 <= 1
 T841_R1_f214_ge: x_BsmtFin_Type_1_LwQ - t841_leaf1 >= 0
 TREE_842_ONEHOT: t842_leaf0 = 1
 TREE_843_ONEHOT: t843_leaf0 = 1
 TREE_844_ONEHOT: t844_leaf0 + t844_leaf1 = 1
 T844_L0_f14_le: x_Bsmt_Unf_SF + 127.6800010000002 t844_leaf0 <= 1812.68
 T844_R1_f14_ge: x_Bsmt_Unf_SF - 1685 t844_leaf1 >= 0
 TREE_845_ONEHOT: t845_leaf0 = 1
 TREE_846_ONEHOT: t846_leaf0 = 1
 TREE_847_ONEHOT: t847_leaf0 + t847_leaf1 = 1
 T847_L0_f112_le: x_Neighborhood_SawyerW + 1e-06 t847_leaf0 <= 1
 T847_R1_f112_ge: x_Neighborhood_SawyerW - t847_leaf1 >= 0
 TREE_848_ONEHOT: t848_leaf0 + t848_leaf1 + t848_leaf2 = 1
 T848_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t848_leaf0 <= 1
 T848_L0_f175_le: x_Exterior_1st_Wd_Sdng + 1e-06 t848_leaf0 <= 1
 T848_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t848_leaf1 <= 1
 T848_R1_f175_ge: x_Exterior_1st_Wd_Sdng - t848_leaf1 >= 0
 T848_R2_f245_ge: x_FunctioNo_aplical_Typ - t848_leaf2 >= 0
 TREE_849_ONEHOT: t849_leaf0 = 1
 TREE_850_ONEHOT: t850_leaf0 = 1
 TREE_851_ONEHOT: t851_leaf0 = 1
 TREE_852_ONEHOT: t852_leaf0 = 1
 TREE_853_ONEHOT: t853_leaf0 = 1
 TREE_854_ONEHOT: t854_leaf0 = 1
 TREE_855_ONEHOT: t855_leaf0 + t855_leaf1 = 1
 T855_L0_f21_le: x_Bsmt_Full_Bath + 0.020001 t855_leaf0 <= 2.02
 T855_R1_f21_ge: x_Bsmt_Full_Bath - 2 t855_leaf1 >= 0
 TREE_856_ONEHOT: t856_leaf0 = 1
 TREE_857_ONEHOT: t857_leaf0 = 1
 TREE_858_ONEHOT: t858_leaf0 + t858_leaf1 = 1
 T858_L0_f40_le: x_Screen_Porch + 2.270000999999979 t858_leaf0 <= 268.27
 T858_R1_f40_ge: x_Screen_Porch - 266 t858_leaf1 >= 0
 TREE_859_ONEHOT: t859_leaf0 = 1
 TREE_860_ONEHOT: t860_leaf0 + t860_leaf1 = 1
 T860_L0_f33_le: x_Garage_Area + 983.750001 t860_leaf0 <= 1163.75
 T860_R1_f33_ge: x_Garage_Area - 180 t860_leaf1 >= 0
 TREE_861_ONEHOT: t861_leaf0 = 1
 TREE_862_ONEHOT: t862_leaf0 + t862_leaf1 + t862_leaf2 = 1
 T862_L0_f9_le: x_Exter_Cond + 2.000001 t862_leaf0 <= 4
 T862_R1_f9_ge: x_Exter_Cond - 3 t862_leaf1 >= -1
 T862_L1_f279_le: x_Mo_Sold_9 + 1e-06 t862_leaf1 <= 1
 T862_R2_f9_ge: x_Exter_Cond - 3 t862_leaf2 >= -1
 T862_R2_f279_ge: x_Mo_Sold_9 - t862_leaf2 >= 0
 TREE_863_ONEHOT: t863_leaf0 + t863_leaf1 + t863_leaf2 + t863_leaf3 = 1
 T863_L0_f31_le: x_Garage_Yr_Blt + 998051.000001 t863_leaf0 <= 1e+06
 T863_R1_f31_ge: x_Garage_Yr_Blt - 1949 t863_leaf1 >= 0
 T863_L1_f27_le: x_Kitchen_Qual + 2.000001 t863_leaf1 <= 4
 T863_L1_f32_le: x_Garage_Cars + 2.000001 t863_leaf1 <= 4
 T863_R2_f31_ge: x_Garage_Yr_Blt - 1949 t863_leaf2 >= 0
 T863_L2_f27_le: x_Kitchen_Qual + 2.000001 t863_leaf2 <= 4
 T863_R2_f32_ge: x_Garage_Cars - 2 t863_leaf2 >= 0
 T863_R3_f31_ge: x_Garage_Yr_Blt - 1949 t863_leaf3 >= 0
 T863_R3_f27_ge: x_Kitchen_Qual >= 2
 TREE_864_ONEHOT: t864_leaf0 = 1
 TREE_865_ONEHOT: t865_leaf0 + t865_leaf1 + t865_leaf2 + t865_leaf3 = 1
 T865_L0_f297_le: x_Sale_Condition_Normal + 1e-06 t865_leaf0 <= 1
 T865_L0_f0_le: x_Lot_Frontage + 115.530001 t865_leaf0 <= 175.53
 T865_L1_f297_le: x_Sale_Condition_Normal + 1e-06 t865_leaf1 <= 1
 T865_R1_f0_ge: x_Lot_Frontage - 60 t865_leaf1 >= 0
 T865_L1_f1_le: x_Lot_Area <= 16778.999999
 T865_L2_f297_le: x_Sale_Condition_Normal + 1e-06 t865_leaf2 <= 1
 T865_R2_f0_ge: x_Lot_Frontage - 60 t865_leaf2 >= 0
 T865_R2_f1_ge: x_Lot_Area - 9779 t865_leaf2 >= 7000
 T865_R3_f297_ge: x_Sale_Condition_Normal - t865_leaf3 >= 0
 TREE_866_ONEHOT: t866_leaf0 + t866_leaf1 + t866_leaf2 + t866_leaf3 = 1
 T866_L0_f5_le: x_Year_Built + 998068.000001 t866_leaf0 <= 1e+06
 T866_L0_f17_le: x_1st_Flr_SF + 1309.900001 t866_leaf0 <= 2534.9
 T866_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t866_leaf0 <= 1
 T866_L1_f5_le: x_Year_Built + 998068.000001 t866_leaf1 <= 1e+06
 T866_L1_f17_le: x_1st_Flr_SF + 1309.900001 t866_leaf1 <= 2534.9
 T866_R1_f161_ge: x_Exterior_1st_AsbShng - t866_leaf1 >= 0
 T866_L2_f5_le: x_Year_Built + 998068.000001 t866_leaf2 <= 1e+06
 T866_R2_f17_ge: x_1st_Flr_SF - 1225 t866_leaf2 >= 0
 T866_R3_f5_ge: x_Year_Built - 1932 t866_leaf3 >= 0
 TREE_867_ONEHOT: t867_leaf0 + t867_leaf1 + t867_leaf2 = 1
 T867_L0_f113_le: x_Neighborhood_Somerst + 1e-06 t867_leaf0 <= 1
 T867_L0_f33_le: x_Garage_Area + 191.750001 t867_leaf0 <= 1163.75
 T867_L1_f113_le: x_Neighborhood_Somerst + 1e-06 t867_leaf1 <= 1
 T867_R1_f33_ge: x_Garage_Area - 972 t867_leaf1 >= 0
 T867_R2_f113_ge: x_Neighborhood_Somerst - t867_leaf2 >= 0
 TREE_868_ONEHOT: t868_leaf0 = 1
 TREE_869_ONEHOT: t869_leaf0 = 1
 TREE_870_ONEHOT: t870_leaf0 = 1
 TREE_871_ONEHOT: t871_leaf0 = 1
 TREE_872_ONEHOT: t872_leaf0 = 1
 TREE_873_ONEHOT: t873_leaf0 + t873_leaf1 = 1
 T873_L0_f8_le: x_Exter_Qual + 2.000001 t873_leaf0 <= 4
 T873_R1_f8_ge: x_Exter_Qual >= 2
 TREE_874_ONEHOT: t874_leaf0 = 1
 TREE_875_ONEHOT: t875_leaf0 = 1
 TREE_876_ONEHOT: t876_leaf0 + t876_leaf1 + t876_leaf2 + t876_leaf3 = 1
 T876_L0_f7_le: x_Mas_Vnr_Area + 909.070001 t876_leaf0 <= 931.07
 T876_L0_f37_le: x_Open_Porch_SF + 300.810001 t876_leaf0 <= 488.81
 T876_L0_f35_le: x_Garage_Cond + 2.000001 t876_leaf0 <= 4
 T876_L1_f7_le: x_Mas_Vnr_Area + 909.070001 t876_leaf1 <= 931.07
 T876_L1_f37_le: x_Open_Porch_SF + 300.810001 t876_leaf1 <= 488.81
 T876_R1_f35_ge: x_Garage_Cond - 3 t876_leaf1 >= -1
 T876_L2_f7_le: x_Mas_Vnr_Area + 909.070001 t876_leaf2 <= 931.07
 T876_R2_f37_ge: x_Open_Porch_SF - 188 t876_leaf2 >= 0
 T876_R3_f7_ge: x_Mas_Vnr_Area - 22 t876_leaf3 >= 0
 TREE_877_ONEHOT: t877_leaf0 + t877_leaf1 + t877_leaf2 = 1
 T877_L0_f0_le: x_Lot_Frontage + 67.530001 t877_leaf0 <= 175.53
 T877_R1_f0_ge: x_Lot_Frontage - 108 t877_leaf1 >= 0
 T877_L1_f119_le: x_Condition_1_Norm + 1e-06 t877_leaf1 <= 1
 T877_R2_f0_ge: x_Lot_Frontage - 108 t877_leaf2 >= 0
 T877_R2_f119_ge: x_Condition_1_Norm - t877_leaf2 >= 0
 TREE_878_ONEHOT: t878_leaf0 = 1
 TREE_879_ONEHOT: t879_leaf0 + t879_leaf1 = 1
 T879_L0_f71_le: x_Alley_No_aplica + 1e-06 t879_leaf0 <= 1
 T879_R1_f71_ge: x_Alley_No_aplica - t879_leaf1 >= 0
 TREE_880_ONEHOT: t880_leaf0 + t880_leaf1 + t880_leaf2 + t880_leaf3
   + t880_leaf4 + t880_leaf5 = 1
 T880_L0_f1_le: x_Lot_Area <= 20430.999999
 T880_L0_f4_le: x_Overall_Cond + 999993.000001 t880_leaf0 <= 1e+06
 T880_L0_f213_le: x_BsmtFin_Type_1_GLQ + 1e-06 t880_leaf0 <= 1
 T880_L1_f1_le: x_Lot_Area <= 20430.999999
 T880_L1_f4_le: x_Overall_Cond + 999993.000001 t880_leaf1 <= 1e+06
 T880_R1_f213_ge: x_BsmtFin_Type_1_GLQ - t880_leaf1 >= 0
 T880_L2_f1_le: x_Lot_Area <= 20430.999999
 T880_R2_f4_ge: x_Overall_Cond - 7 t880_leaf2 >= 0
 T880_R3_f1_ge: x_Lot_Area - 13431 t880_leaf3 >= 7000
 T880_L3_f17_le: x_1st_Flr_SF + 1293.900001 t880_leaf3 <= 2534.9
 T880_R4_f1_ge: x_Lot_Area - 13431 t880_leaf4 >= 7000
 T880_R4_f17_ge: x_1st_Flr_SF - 1241 t880_leaf4 >= 0
 T880_L4_f74_le: x_Lot_Shape_IR2 + 1e-06 t880_leaf4 <= 1
 T880_R5_f1_ge: x_Lot_Area - 13431 t880_leaf5 >= 7000
 T880_R5_f17_ge: x_1st_Flr_SF - 1241 t880_leaf5 >= 0
 T880_R5_f74_ge: x_Lot_Shape_IR2 - t880_leaf5 >= 0
 TREE_881_ONEHOT: t881_leaf0 + t881_leaf1 + t881_leaf2 + t881_leaf3
   + t881_leaf4 = 1
 T881_L0_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t881_leaf0 <= 1
 T881_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t881_leaf0 <= 1
 T881_L0_f44_le: x_Yr_Sold + 997991.000001 t881_leaf0 <= 1e+06
 T881_L1_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t881_leaf1 <= 1
 T881_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t881_leaf1 <= 1
 T881_R1_f44_ge: x_Yr_Sold - 2009 t881_leaf1 >= 0
 T881_L2_f177_le: x_Exterior_2nd_AsbShng + 1e-06 t881_leaf2 <= 1
 T881_R2_f245_ge: x_FunctioNo_aplical_Typ - t881_leaf2 >= 0
 T881_R3_f177_ge: x_Exterior_2nd_AsbShng - t881_leaf3 >= 0
 T881_L3_f217_le: x_BsmtFin_Type_1_Unf + 1e-06 t881_leaf3 <= 1
 T881_R4_f177_ge: x_Exterior_2nd_AsbShng - t881_leaf4 >= 0
 T881_R4_f217_ge: x_BsmtFin_Type_1_Unf - t881_leaf4 >= 0
 TREE_882_ONEHOT: t882_leaf0 = 1
 TREE_883_ONEHOT: t883_leaf0 = 1
 TREE_884_ONEHOT: t884_leaf0 + t884_leaf1 + t884_leaf2 + t884_leaf3
   + t884_leaf4 = 1
 T884_L0_f231_le: x_Central_Air_N + 1e-06 t884_leaf0 <= 1
 T884_L0_f18_le: x_2nd_Flr_SF + 685.540001 t884_leaf0 <= 1641.54
 T884_L1_f231_le: x_Central_Air_N + 1e-06 t884_leaf1 <= 1
 T884_R1_f18_ge: x_2nd_Flr_SF - 956 t884_leaf1 >= 0
 T884_R2_f231_ge: x_Central_Air_N - t884_leaf2 >= 0
 T884_L2_f44_le: x_Yr_Sold + 997990.000001 t884_leaf2 <= 1e+06
 T884_L2_f18_le: x_2nd_Flr_SF + 841.540001 t884_leaf2 <= 1641.54
 T884_R3_f231_ge: x_Central_Air_N - t884_leaf3 >= 0
 T884_L3_f44_le: x_Yr_Sold + 997990.000001 t884_leaf3 <= 1e+06
 T884_R3_f18_ge: x_2nd_Flr_SF - 800 t884_leaf3 >= 0
 T884_R4_f231_ge: x_Central_Air_N - t884_leaf4 >= 0
 T884_R4_f44_ge: x_Yr_Sold - 2010 t884_leaf4 >= 0
 TREE_885_ONEHOT: t885_leaf0 = 1
 TREE_886_ONEHOT: t886_leaf0 + t886_leaf1 + t886_leaf2 + t886_leaf3
   + t886_leaf4 + t886_leaf5 = 1
 T886_L0_f38_le: x_Enclosed_Porch + 214.750001 t886_leaf0 <= 292.75
 T886_L0_f38_le: x_Enclosed_Porch + 236.750001 t886_leaf0 <= 292.75
 T886_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t886_leaf0 <= 1
 T886_L1_f38_le: x_Enclosed_Porch + 214.750001 t886_leaf1 <= 292.75
 T886_L1_f38_le: x_Enclosed_Porch + 236.750001 t886_leaf1 <= 292.75
 T886_R1_f293_ge: x_Sale_Condition_Abnorml - t886_leaf1 >= 0
 T886_L2_f38_le: x_Enclosed_Porch + 214.750001 t886_leaf2 <= 292.75
 T886_R2_f38_ge: x_Enclosed_Porch - 56 t886_leaf2 >= 0
 T886_R3_f38_ge: x_Enclosed_Porch - 78 t886_leaf3 >= 0
 T886_L3_f20_le: x_Gr_Liv_Area + 2404.760001 t886_leaf3 <= 3421.76
 T886_L3_f67_le: x_MS_Zoning_RM + 1e-06 t886_leaf3 <= 1
 T886_R4_f38_ge: x_Enclosed_Porch - 78 t886_leaf4 >= 0
 T886_L4_f20_le: x_Gr_Liv_Area + 2404.760001 t886_leaf4 <= 3421.76
 T886_R4_f67_ge: x_MS_Zoning_RM - t886_leaf4 >= 0
 T886_R5_f38_ge: x_Enclosed_Porch - 78 t886_leaf5 >= 0
 T886_R5_f20_ge: x_Gr_Liv_Area - 1017 t886_leaf5 >= 0
 TREE_887_ONEHOT: t887_leaf0 + t887_leaf1 = 1
 T887_L0_f75_le: x_Lot_Shape_IR3 + 1e-06 t887_leaf0 <= 1
 T887_R1_f75_ge: x_Lot_Shape_IR3 - t887_leaf1 >= 0
 TREE_888_ONEHOT: t888_leaf0 + t888_leaf1 + t888_leaf2 + t888_leaf3
   + t888_leaf4 + t888_leaf5 = 1
 T888_L0_f233_le: x_Electrical_FuseA + 1e-06 t888_leaf0 <= 1
 T888_L0_f18_le: x_2nd_Flr_SF + 236.5400010000001 t888_leaf0 <= 1641.54
 T888_L0_f86_le: x_Land_Slope_Gtl + 1e-06 t888_leaf0 <= 1
 T888_L1_f233_le: x_Electrical_FuseA + 1e-06 t888_leaf1 <= 1
 T888_L1_f18_le: x_2nd_Flr_SF + 236.5400010000001 t888_leaf1 <= 1641.54
 T888_R1_f86_ge: x_Land_Slope_Gtl - t888_leaf1 >= 0
 T888_L2_f233_le: x_Electrical_FuseA + 1e-06 t888_leaf2 <= 1
 T888_R2_f18_ge: x_2nd_Flr_SF - 1405 t888_leaf2 >= 0
 T888_R3_f233_ge: x_Electrical_FuseA - t888_leaf3 >= 0
 T888_L3_f35_le: x_Garage_Cond + 2.000001 t888_leaf3 <= 4
 T888_R4_f233_ge: x_Electrical_FuseA - t888_leaf4 >= 0
 T888_R4_f35_ge: x_Garage_Cond - 3 t888_leaf4 >= -1
 T888_L4_f43_le: x_Misc_Val + 2975.470001 t888_leaf4 <= 3029.47
 T888_R5_f233_ge: x_Electrical_FuseA - t888_leaf5 >= 0
 T888_R5_f35_ge: x_Garage_Cond - 3 t888_leaf5 >= -1
 T888_R5_f43_ge: x_Misc_Val - 54 t888_leaf5 >= 0
 TREE_889_ONEHOT: t889_leaf0 + t889_leaf1 + t889_leaf2 + t889_leaf3
   + t889_leaf4 + t889_leaf5 = 1
 T889_L0_f20_le: x_Gr_Liv_Area + 2538.760001 t889_leaf0 <= 3421.76
 T889_L0_f30_le: x_Fireplace_Qu + 4.000001 t889_leaf0 <= 4
 T889_L1_f20_le: x_Gr_Liv_Area + 2538.760001 t889_leaf1 <= 3421.76
 T889_R1_f30_ge: x_Fireplace_Qu >= 0
 T889_R2_f20_ge: x_Gr_Liv_Area - 883 t889_leaf2 >= 0
 T889_L2_f15_le: x_Total_Bsmt_SF + 1378.480001 t889_leaf2 <= 2514.48
 T889_L2_f259_le: x_Paved_Drive_Y + 1e-06 t889_leaf2 <= 1
 T889_R3_f20_ge: x_Gr_Liv_Area - 883 t889_leaf3 >= 0
 T889_L3_f15_le: x_Total_Bsmt_SF + 1378.480001 t889_leaf3 <= 2514.48
 T889_R3_f259_ge: x_Paved_Drive_Y - t889_leaf3 >= 0
 T889_R4_f20_ge: x_Gr_Liv_Area - 883 t889_leaf4 >= 0
 T889_R4_f15_ge: x_Total_Bsmt_SF - 1136 t889_leaf4 >= 0
 T889_L4_f296_le: x_Sale_Condition_Family + 1e-06 t889_leaf4 <= 1
 T889_R5_f20_ge: x_Gr_Liv_Area - 883 t889_leaf5 >= 0
 T889_R5_f15_ge: x_Total_Bsmt_SF - 1136 t889_leaf5 >= 0
 T889_R5_f296_ge: x_Sale_Condition_Family - t889_leaf5 >= 0
 TREE_890_ONEHOT: t890_leaf0 + t890_leaf1 + t890_leaf2 = 1
 T890_L0_f214_le: x_BsmtFin_Type_1_LwQ + 1e-06 t890_leaf0 <= 1
 T890_R1_f214_ge: x_BsmtFin_Type_1_LwQ - t890_leaf1 >= 0
 T890_L1_f12_le: x_BsmtFin_SF_1 + 1722.500001 t890_leaf1 <= 1984.5
 T890_R2_f214_ge: x_BsmtFin_Type_1_LwQ - t890_leaf2 >= 0
 T890_R2_f12_ge: x_BsmtFin_SF_1 - 262 t890_leaf2 >= 0
 TREE_891_ONEHOT: t891_leaf0 + t891_leaf1 = 1
 T891_L0_f7_le: x_Mas_Vnr_Area + 143.070001 t891_leaf0 <= 931.07
 T891_R1_f7_ge: x_Mas_Vnr_Area - 788 t891_leaf1 >= 0
 TREE_892_ONEHOT: t892_leaf0 = 1
 TREE_893_ONEHOT: t893_leaf0 = 1
 TREE_894_ONEHOT: t894_leaf0 + t894_leaf1 = 1
 T894_L0_f33_le: x_Garage_Area + 292.750001 t894_leaf0 <= 1163.75
 T894_R1_f33_ge: x_Garage_Area - 871 t894_leaf1 >= 0
 TREE_895_ONEHOT: t895_leaf0 = 1
 TREE_896_ONEHOT: t896_leaf0 + t896_leaf1 = 1
 T896_L0_f1_le: x_Lot_Area + 2500.000001 t896_leaf0 <= 7000
 T896_R1_f1_ge: x_Lot_Area >= 4500
 TREE_897_ONEHOT: t897_leaf0 = 1
 TREE_898_ONEHOT: t898_leaf0 = 1
 TREE_899_ONEHOT: t899_leaf0 = 1
 TREE_900_ONEHOT: t900_leaf0 = 1
 TREE_901_ONEHOT: t901_leaf0 = 1
 TREE_902_ONEHOT: t902_leaf0 = 1
 TREE_903_ONEHOT: t903_leaf0 + t903_leaf1 = 1
 T903_L0_f69_le: x_Street_Pave + 1e-06 t903_leaf0 <= 1
 T903_R1_f69_ge: x_Street_Pave - t903_leaf1 >= 0
 TREE_904_ONEHOT: t904_leaf0 + t904_leaf1 + t904_leaf2 = 1
 T904_L0_f3_le: x_Overall_Qual + 999992.000001 t904_leaf0 <= 1e+06
 T904_L0_f3_le: x_Overall_Qual + 999997.000001 t904_leaf0 <= 1e+06
 T904_L1_f3_le: x_Overall_Qual + 999992.000001 t904_leaf1 <= 1e+06
 T904_R1_f3_ge: x_Overall_Qual - 3 t904_leaf1 >= 0
 T904_R2_f3_ge: x_Overall_Qual - 8 t904_leaf2 >= 0
 TREE_905_ONEHOT: t905_leaf0 + t905_leaf1 + t905_leaf2 = 1
 T905_L0_f3_le: x_Overall_Qual + 999995.000001 t905_leaf0 <= 1e+06
 T905_L0_f8_le: x_Exter_Qual + 1.000001 t905_leaf0 <= 4
 T905_L1_f3_le: x_Overall_Qual + 999995.000001 t905_leaf1 <= 1e+06
 T905_R1_f8_ge: x_Exter_Qual - t905_leaf1 >= 2
 T905_R2_f3_ge: x_Overall_Qual - 5 t905_leaf2 >= 0
 TREE_906_ONEHOT: t906_leaf0 + t906_leaf1 + t906_leaf2 + t906_leaf3
   + t906_leaf4 = 1
 T906_L0_f273_le: x_Mo_Sold_3 + 1e-06 t906_leaf0 <= 1
 T906_L0_f3_le: x_Overall_Qual + 999992.000001 t906_leaf0 <= 1e+06
 T906_L0_f256_le: x_Garage_Finish_Unf + 1e-06 t906_leaf0 <= 1
 T906_L1_f273_le: x_Mo_Sold_3 + 1e-06 t906_leaf1 <= 1
 T906_L1_f3_le: x_Overall_Qual + 999992.000001 t906_leaf1 <= 1e+06
 T906_R1_f256_ge: x_Garage_Finish_Unf - t906_leaf1 >= 0
 T906_L2_f273_le: x_Mo_Sold_3 + 1e-06 t906_leaf2 <= 1
 T906_R2_f3_ge: x_Overall_Qual - 8 t906_leaf2 >= 0
 T906_L2_f0_le: x_Lot_Frontage + 110.530001 t906_leaf2 <= 175.53
 T906_L3_f273_le: x_Mo_Sold_3 + 1e-06 t906_leaf3 <= 1
 T906_R3_f3_ge: x_Overall_Qual - 8 t906_leaf3 >= 0
 T906_R3_f0_ge: x_Lot_Frontage - 65 t906_leaf3 >= 0
 T906_R4_f273_ge: x_Mo_Sold_3 - t906_leaf4 >= 0
 TREE_907_ONEHOT: t907_leaf0 + t907_leaf1 + t907_leaf2 = 1
 T907_L0_f3_le: x_Overall_Qual + 999997.000001 t907_leaf0 <= 1e+06
 T907_L0_f35_le: x_Garage_Cond + 2.000001 t907_leaf0 <= 4
 T907_L1_f3_le: x_Overall_Qual + 999997.000001 t907_leaf1 <= 1e+06
 T907_R1_f35_ge: x_Garage_Cond - 3 t907_leaf1 >= -1
 T907_R2_f3_ge: x_Overall_Qual - 3 t907_leaf2 >= 0
 TREE_908_ONEHOT: t908_leaf0 = 1
 TREE_909_ONEHOT: t909_leaf0 + t909_leaf1 + t909_leaf2 + t909_leaf3 = 1
 T909_L0_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t909_leaf0 <= 1
 T909_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t909_leaf0 <= 1
 T909_L0_f23_le: x_Full_Bath + 4.000001 t909_leaf0 <= 6
 T909_L1_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t909_leaf1 <= 1
 T909_L1_f293_le: x_Sale_Condition_Abnorml + 1e-06 t909_leaf1 <= 1
 T909_R1_f23_ge: x_Full_Bath - 2 t909_leaf1 >= 0
 T909_L2_f245_le: x_FunctioNo_aplical_Typ + 1e-06 t909_leaf2 <= 1
 T909_R2_f293_ge: x_Sale_Condition_Abnorml - t909_leaf2 >= 0
 T909_R3_f245_ge: x_FunctioNo_aplical_Typ - t909_leaf3 >= 0
 TREE_910_ONEHOT: t910_leaf0 = 1
 TREE_911_ONEHOT: t911_leaf0 + t911_leaf1 + t911_leaf2 = 1
 T911_L0_f3_le: x_Overall_Qual + 999993.000001 t911_leaf0 <= 1e+06
 T911_L0_f1_le: x_Lot_Area + 1024.000001 t911_leaf0 <= 7000
 T911_L1_f3_le: x_Overall_Qual + 999993.000001 t911_leaf1 <= 1e+06
 T911_R1_f1_ge: x_Lot_Area >= 5976
 T911_R2_f3_ge: x_Overall_Qual - 7 t911_leaf2 >= 0
 TREE_912_ONEHOT: t912_leaf0 = 1
 TREE_913_ONEHOT: t913_leaf0 = 1
 TREE_914_ONEHOT: t914_leaf0 = 1
 TREE_915_ONEHOT: t915_leaf0 + t915_leaf1 + t915_leaf2 = 1
 T915_L0_f15_le: x_Total_Bsmt_SF + 834.4800010000001 t915_leaf0 <= 2514.48
 T915_L0_f63_le: x_MS_Zoning_FV + 1e-06 t915_leaf0 <= 1
 T915_L1_f15_le: x_Total_Bsmt_SF + 834.4800010000001 t915_leaf1 <= 2514.48
 T915_R1_f63_ge: x_MS_Zoning_FV - t915_leaf1 >= 0
 T915_R2_f15_ge: x_Total_Bsmt_SF - 1680 t915_leaf2 >= 0
 TREE_916_ONEHOT: t916_leaf0 + t916_leaf1 + t916_leaf2 = 1
 T916_L0_f20_le: x_Gr_Liv_Area + 2538.760001 t916_leaf0 <= 3421.76
 T916_L0_f161_le: x_Exterior_1st_AsbShng + 1e-06 t916_leaf0 <= 1
 T916_L1_f20_le: x_Gr_Liv_Area + 2538.760001 t916_leaf1 <= 3421.76
 T916_R1_f161_ge: x_Exterior_1st_AsbShng - t916_leaf1 >= 0
 T916_R2_f20_ge: x_Gr_Liv_Area - 883 t916_leaf2 >= 0
 TREE_917_ONEHOT: t917_leaf0 = 1
 TREE_918_ONEHOT: t918_leaf0 = 1
 TREE_919_ONEHOT: t919_leaf0 = 1
 TREE_920_ONEHOT: t920_leaf0 = 1
 TREE_921_ONEHOT: t921_leaf0 = 1
 TREE_922_ONEHOT: t922_leaf0 + t922_leaf1 + t922_leaf2 = 1
 T922_L0_f261_le: x_Fence_GdWo + 1e-06 t922_leaf0 <= 1
 T922_R1_f261_ge: x_Fence_GdWo - t922_leaf1 >= 0
 T922_L1_f1_le: x_Lot_Area <= 8519.999999
 T922_R2_f261_ge: x_Fence_GdWo - t922_leaf2 >= 0
 T922_R2_f1_ge: x_Lot_Area - 1520 t922_leaf2 >= 7000
 TREE_923_ONEHOT: t923_leaf0 + t923_leaf1 + t923_leaf2 + t923_leaf3
   + t923_leaf4 + t923_leaf5 = 1
 T923_L0_f20_le: x_Gr_Liv_Area + 2433.760001 t923_leaf0 <= 3421.76
 T923_L0_f293_le: x_Sale_Condition_Abnorml + 1e-06 t923_leaf0 <= 1
 T923_L1_f20_le: x_Gr_Liv_Area + 2433.760001 t923_leaf1 <= 3421.76
 T923_R1_f293_ge: x_Sale_Condition_Abnorml - t923_leaf1 >= 0
 T923_L1_f17_le: x_1st_Flr_SF + 1654.900001 t923_leaf1 <= 2534.9
 T923_L2_f20_le: x_Gr_Liv_Area + 2433.760001 t923_leaf2 <= 3421.76
 T923_R2_f293_ge: x_Sale_Condition_Abnorml - t923_leaf2 >= 0
 T923_R2_f17_ge: x_1st_Flr_SF - 880 t923_leaf2 >= 0
 T923_R3_f20_ge: x_Gr_Liv_Area - 988 t923_leaf3 >= 0
 T923_L3_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t923_leaf3 <= 2514.48
 T923_L3_f20_le: x_Gr_Liv_Area + 1976.760001 t923_leaf3 <= 3421.76
 T923_R4_f20_ge: x_Gr_Liv_Area - 988 t923_leaf4 >= 0
 T923_L4_f15_le: x_Total_Bsmt_SF + 378.4800009999999 t923_leaf4 <= 2514.48
 T923_R4_f20_ge: x_Gr_Liv_Area - 1445 t923_leaf4 >= 0
 T923_R5_f20_ge: x_Gr_Liv_Area - 988 t923_leaf5 >= 0
 T923_R5_f15_ge: x_Total_Bsmt_SF - 2136 t923_leaf5 >= 0
 TREE_924_ONEHOT: t924_leaf0 + t924_leaf1 + t924_leaf2 + t924_leaf3 = 1
 T924_L0_f20_le: x_Gr_Liv_Area + 2485.760001 t924_leaf0 <= 3421.76
 T924_L0_f29_le: x_Fireplaces + 2.000001 t924_leaf0 <= 3
 T924_L1_f20_le: x_Gr_Liv_Area + 2485.760001 t924_leaf1 <= 3421.76
 T924_R1_f29_ge: x_Fireplaces - t924_leaf1 >= 0
 T924_L1_f1_le: x_Lot_Area <= 8012.999999
 T924_L2_f20_le: x_Gr_Liv_Area + 2485.760001 t924_leaf2 <= 3421.76
 T924_R2_f29_ge: x_Fireplaces - t924_leaf2 >= 0
 T924_R2_f1_ge: x_Lot_Area - 1013 t924_leaf2 >= 7000
 T924_R3_f20_ge: x_Gr_Liv_Area - 936 t924_leaf3 >= 0
 TREE_925_ONEHOT: t925_leaf0 + t925_leaf1 + t925_leaf2 = 1
 T925_L0_f1_le: x_Lot_Area <= 20430.999999
 T925_L0_f27_le: x_Kitchen_Qual + 1.000001 t925_leaf0 <= 4
 T925_L1_f1_le: x_Lot_Area <= 20430.999999
 T925_R1_f27_ge: x_Kitchen_Qual - t925_leaf1 >= 2
 T925_R2_f1_ge: x_Lot_Area - 13431 t925_leaf2 >= 7000
 TREE_926_ONEHOT: t926_leaf0 = 1
 TREE_927_ONEHOT: t927_leaf0 = 1
 TREE_928_ONEHOT: t928_leaf0 = 1
 TREE_929_ONEHOT: t929_leaf0 = 1
 TREE_930_ONEHOT: t930_leaf0 = 1
 TREE_931_ONEHOT: t931_leaf0 + t931_leaf1 + t931_leaf2 = 1
 T931_L0_f0_le: x_Lot_Frontage + 76.530001 t931_leaf0 <= 175.53
 T931_R1_f0_ge: x_Lot_Frontage - 99 t931_leaf1 >= 0
 T931_L1_f0_le: x_Lot_Frontage + 1.530000999999999 t931_leaf1 <= 175.53
 T931_R2_f0_ge: x_Lot_Frontage - 99 t931_leaf2 >= 0
 T931_R2_f0_ge: x_Lot_Frontage - 174 t931_leaf2 >= 0
 TREE_932_ONEHOT: t932_leaf0 + t932_leaf1 = 1
 T932_L0_f250_le: x_Garage_Type_CarPort + 1e-06 t932_leaf0 <= 1
 T932_R1_f250_ge: x_Garage_Type_CarPort - t932_leaf1 >= 0
 TREE_933_ONEHOT: t933_leaf0 = 1
 TREE_934_ONEHOT: t934_leaf0 = 1
 TREE_935_ONEHOT: t935_leaf0 + t935_leaf1 + t935_leaf2 + t935_leaf3 = 1
 T935_L0_f41_le: x_Pool_Area + 76.77000099999998 t935_leaf0 <= 444.77
 T935_L0_f4_le: x_Overall_Cond + 999996.000001 t935_leaf0 <= 1e+06
 T935_L1_f41_le: x_Pool_Area + 76.77000099999998 t935_leaf1 <= 444.77
 T935_R1_f4_ge: x_Overall_Cond - 4 t935_leaf1 >= 0
 T935_L1_f5_le: x_Year_Built + 998068.000001 t935_leaf1 <= 1e+06
 T935_L2_f41_le: x_Pool_Area + 76.77000099999998 t935_leaf2 <= 444.77
 T935_R2_f4_ge: x_Overall_Cond - 4 t935_leaf2 >= 0
 T935_R2_f5_ge: x_Year_Built - 1932 t935_leaf2 >= 0
 T935_R3_f41_ge: x_Pool_Area - 368 t935_leaf3 >= 0
 TREE_936_ONEHOT: t936_leaf0 = 1
 TREE_937_ONEHOT: t937_leaf0 + t937_leaf1 + t937_leaf2 = 1
 T937_L0_f9_le: x_Exter_Cond + 2.000001 t937_leaf0 <= 4
 T937_R1_f9_ge: x_Exter_Cond - 3 t937_leaf1 >= -1
 T937_L1_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t937_leaf1 <= 2514.48
 T937_R2_f9_ge: x_Exter_Cond - 3 t937_leaf2 >= -1
 T937_R2_f15_ge: x_Total_Bsmt_SF - 2223 t937_leaf2 >= 0
 TREE_938_ONEHOT: t938_leaf0 = 1
 TREE_939_ONEHOT: t939_leaf0 + t939_leaf1 = 1
 T939_L0_f15_le: x_Total_Bsmt_SF + 291.4800009999999 t939_leaf0 <= 2514.48
 T939_R1_f15_ge: x_Total_Bsmt_SF - 2223 t939_leaf1 >= 0
 TREE_940_ONEHOT: t940_leaf0 + t940_leaf1 + t940_leaf2 + t940_leaf3 = 1
 T940_L0_f3_le: x_Overall_Qual + 999994.000001 t940_leaf0 <= 1e+06
 T940_L0_f247_le: x_Garage_Type_Attchd + 1e-06 t940_leaf0 <= 1
 T940_L0_f15_le: x_Total_Bsmt_SF + 1214.480001 t940_leaf0 <= 2514.48
 T940_L1_f3_le: x_Overall_Qual + 999994.000001 t940_leaf1 <= 1e+06
 T940_L1_f247_le: x_Garage_Type_Attchd + 1e-06 t940_leaf1 <= 1
 T940_R1_f15_ge: x_Total_Bsmt_SF - 1300 t940_leaf1 >= 0
 T940_L2_f3_le: x_Overall_Qual + 999994.000001 t940_leaf2 <= 1e+06
 T940_R2_f247_ge: x_Garage_Type_Attchd - t940_leaf2 >= 0
 T940_R3_f3_ge: x_Overall_Qual - 6 t940_leaf3 >= 0
 TREE_941_ONEHOT: t941_leaf0 + t941_leaf1 = 1
 T941_L0_f3_le: x_Overall_Qual + 999991.000001 t941_leaf0 <= 1e+06
 T941_R1_f3_ge: x_Overall_Qual - 9 t941_leaf1 >= 0
 TREE_942_ONEHOT: t942_leaf0 = 1
 TREE_943_ONEHOT: t943_leaf0 = 1
 TREE_944_ONEHOT: t944_leaf0 = 1
 TREE_945_ONEHOT: t945_leaf0 = 1
 TREE_946_ONEHOT: t946_leaf0 = 1
 TREE_947_ONEHOT: t947_leaf0 + t947_leaf1 + t947_leaf2 + t947_leaf3
   + t947_leaf4 = 1
 T947_L0_f3_le: x_Overall_Qual + 999995.000001 t947_leaf0 <= 1e+06
 T947_L0_f6_le: x_Year_Remod/Add + 998038.000001 t947_leaf0 <= 1e+06
 T947_L0_f1_le: x_Lot_Area <= 9532.999999
 T947_L1_f3_le: x_Overall_Qual + 999995.000001 t947_leaf1 <= 1e+06
 T947_L1_f6_le: x_Year_Remod/Add + 998038.000001 t947_leaf1 <= 1e+06
 T947_R1_f1_ge: x_Lot_Area - 2533 t947_leaf1 >= 7000
 T947_L2_f3_le: x_Overall_Qual + 999995.000001 t947_leaf2 <= 1e+06
 T947_R2_f6_ge: x_Year_Remod/Add - 1962 t947_leaf2 >= 0
 T947_L2_f16_le: x_Heating_QC + 1.000001 t947_leaf2 <= 4
 T947_L3_f3_le: x_Overall_Qual + 999995.000001 t947_leaf3 <= 1e+06
 T947_R3_f6_ge: x_Year_Remod/Add - 1962 t947_leaf3 >= 0
 T947_R3_f16_ge: x_Heating_QC - t947_leaf3 >= 2
 T947_R4_f3_ge: x_Overall_Qual - 5 t947_leaf4 >= 0
 TREE_948_ONEHOT: t948_leaf0 = 1
 TREE_949_ONEHOT: t949_leaf0 = 1
 TREE_950_ONEHOT: t950_leaf0 = 1
 TREE_951_ONEHOT: t951_leaf0 + t951_leaf1 + t951_leaf2 = 1
 T951_L0_f296_le: x_Sale_Condition_Family + 1e-06 t951_leaf0 <= 1
 T951_R1_f296_ge: x_Sale_Condition_Family - t951_leaf1 >= 0
 T951_L1_f7_le: x_Mas_Vnr_Area + 909.070001 t951_leaf1 <= 931.07
 T951_R2_f296_ge: x_Sale_Condition_Family - t951_leaf2 >= 0
 T951_R2_f7_ge: x_Mas_Vnr_Area - 22 t951_leaf2 >= 0
 TREE_952_ONEHOT: t952_leaf0 = 1
 TREE_953_ONEHOT: t953_leaf0 = 1
 TREE_954_ONEHOT: t954_leaf0 + t954_leaf1 + t954_leaf2 + t954_leaf3
   + t954_leaf4 = 1
 T954_L0_f3_le: x_Overall_Qual + 999992.000001 t954_leaf0 <= 1e+06
 T954_L0_f15_le: x_Total_Bsmt_SF + 1076.480001 t954_leaf0 <= 2514.48
 T954_L1_f3_le: x_Overall_Qual + 999992.000001 t954_leaf1 <= 1e+06
 T954_R1_f15_ge: x_Total_Bsmt_SF - 1438 t954_leaf1 >= 0
 T954_L1_f28_le: x_TotRms_AbvGrd + 6.080001 t954_leaf1 <= 12.08
 T954_L2_f3_le: x_Overall_Qual + 999992.000001 t954_leaf2 <= 1e+06
 T954_R2_f15_ge: x_Total_Bsmt_SF - 1438 t954_leaf2 >= 0
 T954_R2_f28_ge: x_TotRms_AbvGrd - 6 t954_leaf2 >= 0
 T954_R3_f3_ge: x_Overall_Qual - 8 t954_leaf3 >= 0
 T954_L3_f18_le: x_2nd_Flr_SF + 236.5400010000001 t954_leaf3 <= 1641.54
 T954_R4_f3_ge: x_Overall_Qual - 8 t954_leaf4 >= 0
 T954_R4_f18_ge: x_2nd_Flr_SF - 1405 t954_leaf4 >= 0
 TREE_955_ONEHOT: t955_leaf0 = 1
 TREE_956_ONEHOT: t956_leaf0 + t956_leaf1 = 1
 T956_L0_f3_le: x_Overall_Qual + 999992.000001 t956_leaf0 <= 1e+06
 T956_R1_f3_ge: x_Overall_Qual - 8 t956_leaf1 >= 0
 TREE_957_ONEHOT: t957_leaf0 + t957_leaf1 + t957_leaf2 + t957_leaf3
   + t957_leaf4 = 1
 T957_L0_f35_le: x_Garage_Cond + 2.000001 t957_leaf0 <= 4
 T957_L0_f1_le: x_Lot_Area <= 13100.999999
 T957_L0_f5_le: x_Year_Built + 998064.000001 t957_leaf0 <= 1e+06
 T957_L1_f35_le: x_Garage_Cond + 2.000001 t957_leaf1 <= 4
 T957_L1_f1_le: x_Lot_Area <= 13100.999999
 T957_R1_f5_ge: x_Year_Built - 1936 t957_leaf1 >= 0
 T957_L2_f35_le: x_Garage_Cond + 2.000001 t957_leaf2 <= 4
 T957_R2_f1_ge: x_Lot_Area - 6101 t957_leaf2 >= 7000
 T957_R3_f35_ge: x_Garage_Cond - 3 t957_leaf3 >= -1
 T957_L3_f106_le: x_Neighborhood_No_aplicames + 1e-06 t957_leaf3 <= 1
 T957_R4_f35_ge: x_Garage_Cond - 3 t957_leaf4 >= -1
 T957_R4_f106_ge: x_Neighborhood_No_aplicames - t957_leaf4 >= 0
 YLOG_XGB_SUM_STRICT: y_log - 0.504045188 t0_leaf0 - 0.482881427 t1_leaf0
   - 0.462346554 t2_leaf0 - 0.442416757 t3_leaf0 - 0.423827738 t4_leaf0
   - 0.406718105 t5_leaf0 - 0.388666034 t6_leaf0 - 0.372950315 t7_leaf0
   - 0.356958956 t8_leaf0 - 0.341672182 t9_leaf0 - 0.327231765 t10_leaf0
   - 0.313321412 t11_leaf0 - 0.299923956 t12_leaf0 - 0.287695199 t13_leaf0
   - 0.275456518 t14_leaf0 - 0.263657898 t15_leaf0 - 0.252950877 t16_leaf0
   - 0.241458759 t17_leaf0 - 0.232026309 t18_leaf0 - 0.222016707 t19_leaf0
   - 0.212877616 t20_leaf0 - 0.203600854 t21_leaf0 - 0.195218012 t22_leaf0
   - 0.177936837 t23_leaf0 - 0.200780556 t23_leaf1 - 0.170204267 t24_leaf0
   - 0.192655176 t24_leaf1 - 0.162858665 t25_leaf0 - 0.184863031 t25_leaf1
   - 0.150811613 t26_leaf0 - 0.172091812 t26_leaf1 - 0.149724171 t27_leaf0
   - 0.170104459 t27_leaf1 - 0.142522916 t28_leaf0 - 0.16037108 t28_leaf1
   - 0.136761278 t29_leaf0 - 0.155523077 t29_leaf1 - 0.130828112 t30_leaf0
   - 0.149716258 t30_leaf1 - 0.123164117 t31_leaf0 - 0.141133875 t31_leaf1
   - 0.11487522 t32_leaf0 - 0.131934717 t32_leaf1 - 0.113312773 t33_leaf0
   - 0.130847722 t33_leaf1 - 0.0444914959 t34_leaf0 - 0.106975965 t34_leaf1
   - 0.1217338 t34_leaf2 - 0.105519354 t35_leaf0 - 0.121380992 t35_leaf1
   - 0.0976278484 t36_leaf0 - 0.113471948 t36_leaf1
   - 0.0963234082 t37_leaf0 - 0.0716662481 t37_leaf1
   - 0.109765545 t37_leaf2 - 0.0348871835 t38_leaf0
   - 0.0891734511 t38_leaf1 - 0.103439152 t38_leaf2
   - 0.0319422334 t39_leaf0 - 0.0881977007 t39_leaf1
   - 0.102707252 t39_leaf2 - 0.0329293534 t40_leaf0
   - 0.0850960538 t40_leaf1 - 0.0989284366 t40_leaf2
   - 0.0233326722 t41_leaf0 - 0.0797935426 t41_leaf1
   - 0.0935123488 t41_leaf2 - 0.076474987 t42_leaf0
   - 0.0896939039 t42_leaf1 - 0.0608726405 t43_leaf0
   - 0.0772765502 t43_leaf1 - 0.0932453573 t43_leaf2
   - 0.0557299443 t44_leaf0 - 0.0717785582 t44_leaf1
   - 0.0830021277 t44_leaf2 - 0.0630232841 t45_leaf0
   - 0.0195275787 t45_leaf1 - 0.0763520375 t45_leaf2
   - 0.016954178 t46_leaf0 - 0.0519875549 t46_leaf1
   - 0.0680868477 t46_leaf2 - 0.0836376175 t46_leaf3
   - 0.0503634736 t47_leaf0 - 0.0652702078 t47_leaf1
   - 0.0796747804 t47_leaf2 - 0.0424284749 t48_leaf0
   - 0.0577201061 t48_leaf1 + 5.1891553400000004e-04 t48_leaf2
   - 0.0231980141 t48_leaf3 - 0.0652166605 t48_leaf4
   - 0.0803370774 t48_leaf5 - 0.0564215928 t49_leaf0
   - 0.0173283275 t49_leaf1 - 0.0434801579 t49_leaf2
   - 0.0196449589 t49_leaf3 - 0.0668516308 t49_leaf4
   - 0.00975957047 t50_leaf0 - 0.0414430089 t50_leaf1
   - 0.0550913773 t50_leaf2 - 0.0648489967 t50_leaf3
   - 0.00415315293 t51_leaf0 - 0.0253665876 t51_leaf1
   - 0.0511952192 t51_leaf2 - 0.0565945022 t51_leaf3
   - 0.0671763122 t51_leaf4 - 0.0409850255 t52_leaf0
   - 0.0148272431 t52_leaf1 - 0.0528117977 t52_leaf2
   - 0.0667790771 t52_leaf3 - 0.0252225678 t53_leaf0
   - 0.0474746116 t53_leaf1 - 0.0167559776 t53_leaf2
   - 0.0537164658 t53_leaf3 - 0.0650293976 t53_leaf4
   - 0.0264631864 t54_leaf0 - 0.00372877368 t54_leaf1
   - 0.0397829674 t54_leaf2 - 0.0488000587 t54_leaf3
   - 0.060823869 t54_leaf4 - 0.0061627361 t55_leaf0
   - 0.0276274793 t55_leaf1 - 0.0258089472 t55_leaf2
   - 0.043406941 t55_leaf3 - 0.0481569543 t55_leaf4
   - 0.0583192147 t55_leaf5 - 0.021248145 t56_leaf0
   - 0.0406489931 t56_leaf1 - 0.0465154499 t56_leaf2
   - 0.0567740053 t56_leaf3 - 0.0326307304 t57_leaf0
   - 0.0243052673 t57_leaf1 - 0.0429625027 t57_leaf2
   - 0.0542507209 t57_leaf3 - 0.0196800884 t58_leaf0
   - 0.0211032741 t58_leaf1 - 0.0381814167 t58_leaf2
   - 0.0414327346 t58_leaf3 - 0.0504817553 t58_leaf4
   - 0.025769379 t59_leaf0 - 0.0386953615 t59_leaf1
   - 0.0313899294 t59_leaf2 - 0.0425709039 t59_leaf3
   - 0.0525971651 t59_leaf4 - 0.00407658564 t60_leaf0
   - 0.021161925 t60_leaf1 - 0.0333084166 t60_leaf2
   - 0.0392833985 t60_leaf3 - 0.0501793884 t60_leaf4
   - 0.0204185713 t61_leaf0 + 0.000250128 t61_leaf1
   - 0.0266987849 t61_leaf2 - 0.0345652886 t61_leaf3
   - 0.038058307 t61_leaf4 - 0.0479928143 t61_leaf5
   - 0.0251765978 t62_leaf0 - 0.020271644 t62_leaf1
   - 0.0343681835 t62_leaf2 - 0.0213184208 t62_leaf3
   - 0.0427309908 t62_leaf4 - 0.00415697508 t63_leaf0
   - 0.0247189272 t63_leaf1 - 0.0187882483 t63_leaf2
   - 0.0324079432 t63_leaf3 - 0.0349347219 t63_leaf4
   - 0.0428471267 t63_leaf5 - 0.0154002653 t64_leaf1
   - 0.00729787536 t64_leaf2 - 0.0267092567 t64_leaf3
   - 0.032701049 t64_leaf4 - 0.0422185622 t64_leaf5
   - 0.0118723726 t64_leaf6 - 0.00609612186 t65_leaf0
   - 0.0213032141 t65_leaf1 - 0.0302484147 t65_leaf2
   - 0.0233618524 t65_leaf3 - 0.0331194028 t65_leaf4
   - 0.0424578376 t65_leaf5 - 0.0191396326 t66_leaf0
   - 0.0281995293 t66_leaf1 + 0.00214746897 t66_leaf2
   - 0.0151098622 t66_leaf3 - 0.0145820118 t66_leaf4
   - 0.033007279 t66_leaf5 - 0.0428004563 t66_leaf6
   - 0.00473429821 t67_leaf0 - 0.019739829 t67_leaf1
   - 0.0241102073 t67_leaf2 - 0.0285594948 t67_leaf3
   - 0.0356545448 t67_leaf4 - 0.00551293604 t67_leaf5
   - 0.0235828888 t67_leaf6 - 0.019688271 t68_leaf0 - 0.004791677 t68_leaf1
   - 0.025106499 t68_leaf2 - 0.0257904138 t68_leaf3
   - 0.0310217552 t68_leaf4 - 0.0412647091 t68_leaf5
   - 0.020358203 t69_leaf0 - 0.0257746074 t69_leaf1
   - 0.00253723399 t69_leaf2 - 0.0142500317 t69_leaf3
   - 0.0293344222 t69_leaf4 - 0.0380199701 t69_leaf5
   - 0.0162033122 t70_leaf0 - 8.7198545199999996e-04 t70_leaf1
   - 0.00811454281 t70_leaf2 - 0.0225780103 t70_leaf3
   - 0.0265256073 t70_leaf4 - 0.0351583362 t70_leaf5
   - 0.00906986743 t71_leaf0 - 0.0193533953 t71_leaf1
   - 0.0141508076 t71_leaf2 - 0.0245789159 t71_leaf3
   - 0.0119888429 t71_leaf4 - 0.0332998857 t71_leaf5
   - 0.00933770277 t71_leaf6 + 5.7640677600000005e-04 t72_leaf0
   - 0.0134196393 t72_leaf1 - 0.0198835731 t72_leaf2
   - 0.0237594806 t72_leaf3 - 0.00512431702 t72_leaf4
   - 0.0246840585 t72_leaf5 - 0.0318650715 t72_leaf6
   - 0.0187579747 t73_leaf0 - 0.0227293391 t73_leaf1
   - 0.0131972991 t73_leaf2 - 0.00442610169 t73_leaf3
   - 0.0241716783 t73_leaf4 - 0.0307730082 t73_leaf5
   - 0.0175719392 t74_leaf0 - 0.0217686482 t74_leaf1
   - 0.00271528261 t74_leaf2 - 0.0131595274 t74_leaf3
   - 0.0221441258 t74_leaf4 - 0.0310537871 t74_leaf5
   - 0.0137176737 t75_leaf0 - 0.0196195878 t75_leaf1
   - 8.6810422400000005e-04 t75_leaf2 - 0.0120080626 t75_leaf3
   - 0.00630182074 t75_leaf4 - 0.0192503557 t75_leaf5
   - 0.0267161001 t75_leaf6 - 0.016596457 t76_leaf0
   - 0.0124323918 t76_leaf1 - 0.020672448 t76_leaf2 - 0.000670038 t76_leaf3
   - 0.0247736406 t76_leaf4 - 0.00652049901 t76_leaf5
   - 0.00910373777 t77_leaf0 + 0.00634037424 t77_leaf1
   - 0.013681761 t77_leaf2 - 0.0186844785 t77_leaf3
   - 0.0185299609 t77_leaf4 - 0.00726342946 t77_leaf5
   - 0.00907226 t77_leaf6 - 0.0248629879 t77_leaf7
   + 0.00299106329 t78_leaf0 - 0.0131193865 t78_leaf1
   - 0.0110545801 t78_leaf2 - 0.0178815033 t78_leaf3
   - 0.0206496082 t78_leaf4 - 0.0271329582 t78_leaf5
   + 7.4096355800000002e-04 t79_leaf0 - 0.0106013687 t79_leaf1
   - 0.0161031056 t79_leaf2 - 0.0110247359 t79_leaf3
   - 0.00473181717 t79_leaf4 - 0.017786663 t79_leaf5
   - 0.0241272412 t79_leaf6 - 0.0078419894 t80_leaf0
   - 0.000327232 t80_leaf1 - 0.0145052634 t80_leaf2
   - 0.0191348698 t80_leaf3 - 0.00399155729 t80_leaf4
   - 0.0193261206 t80_leaf5 - 0.0261181165 t80_leaf6
   - 0.00843105372 t81_leaf0 - 0.0124185281 t81_leaf1
   - 0.0159948934 t81_leaf2 - 0.00564126298 t81_leaf3
   - 0.0143589387 t81_leaf4 - 0.0194627047 t81_leaf5
   - 0.024183156 t81_leaf6 - 0.00337483454 t82_leaf0
   - 0.0116664004 t82_leaf1 - 0.00867162272 t82_leaf2
   - 0.0157372262 t82_leaf3 - 0.00519546168 t82_leaf4
   - 0.0212729052 t82_leaf5 - 0.0063619893 t83_leaf0
   + 0.00794325117 t83_leaf1 - 0.00817265455 t83_leaf2
   - 0.0131498342 t83_leaf3 - 0.00792641379 t83_leaf4
   - 0.0159723256 t83_leaf5 - 0.0234265849 t83_leaf6
   - 0.000700692 t84_leaf0 - 0.00764670782 t84_leaf1
   - 0.00482547842 t84_leaf2 - 0.0123941349 t84_leaf3
   - 0.0103833713 t84_leaf4 - 0.0156060811 t84_leaf5
   - 0.0222006012 t84_leaf6 - 0.00276647136 t84_leaf7
   - 0.00138441974 t85_leaf0 - 0.00885044411 t85_leaf1
   - 0.00613652589 t85_leaf2 - 0.0124990288 t85_leaf3
   - 0.00469885394 t85_leaf4 - 0.0141670974 t85_leaf5
   - 0.00651978282 t85_leaf6 - 0.0195905529 t85_leaf7
   - 0.00680208392 t86_leaf0 + 5.4188323e-04 t86_leaf1
   - 0.0108207567 t86_leaf2 - 0.0156599302 t86_leaf3
   - 0.004065393 t86_leaf4 - 0.0143128475 t86_leaf5
   - 0.0195821039 t86_leaf6 - 0.00389526156 t86_leaf7
   - 0.00352557539 t87_leaf0 + 0.00484274561 t87_leaf1
   - 0.0103782117 t87_leaf2 - 0.0147414235 t87_leaf3
   - 0.00183658779 t87_leaf4 - 0.0139799528 t87_leaf5
   - 0.0188312437 t87_leaf6 - 0.00212286 t87_leaf7 - 0.00370575 t88_leaf0
   - 0.00959117711 t88_leaf1 - 0.00736988522 t88_leaf2
   - 0.0128926821 t88_leaf3 - 0.00917080324 t88_leaf4
   - 0.0207466073 t88_leaf5 - 0.00398657378 t88_leaf6
   - 0.00618345384 t89_leaf0 + 0.00927254744 t89_leaf1
   - 0.0068139052 t89_leaf2 - 0.0111118136 t89_leaf3
   - 0.0120520303 t89_leaf4 - 0.00583824236 t89_leaf5
   - 0.0144291446 t89_leaf6 - 0.0206480455 t89_leaf7
   - 0.00533346832 t90_leaf0 - 0.00920430105 t90_leaf1
   - 4.05877e-05 t90_leaf2 - 0.0114673199 t90_leaf3
   - 0.00995715242 t90_leaf4 - 0.0176861212 t90_leaf5
   - 1.95580127e-04 t90_leaf6 - 0.00431571295 t91_leaf0
   + 0.0070476057 t91_leaf1 - 0.00434309524 t91_leaf2
   - 0.00893101282 t91_leaf3 - 0.00960993953 t91_leaf4
   - 0.0129600773 t91_leaf5 - 0.0177655928 t91_leaf6
   - 0.00400163606 t91_leaf7 + 0.00947574433 t92_leaf0
   - 0.00332197151 t92_leaf1 - 0.00524238963 t92_leaf2
   - 0.00968233496 t92_leaf3 - 1.6719954099999999e-04 t92_leaf4
   - 0.0105591854 t92_leaf5 - 0.0174847394 t92_leaf6
   - 0.00669482304 t93_leaf0 - 0.000328741 t93_leaf1
   - 0.0102892518 t93_leaf2 - 0.00736386981 t93_leaf3
   - 9.4074523100000005e-04 t93_leaf4 - 0.00743235182 t93_leaf5
   - 0.0147158541 t93_leaf6 - 0.00537325162 t94_leaf0
   - 0.00847260747 t94_leaf1 - 0.00415192917 t94_leaf2
   - 0.00999782886 t94_leaf3 - 0.0163903665 t94_leaf4
   - 0.00356069976 t94_leaf5 + 0.0062719211 t95_leaf0
   - 0.00290997257 t95_leaf1 - 0.00130383216 t95_leaf2
   - 0.00735426601 t95_leaf3 - 0.00902896561 t95_leaf4
   - 0.011917362 t95_leaf5 - 0.0187763292 t95_leaf6 - 0.00513004 t95_leaf7
   + 0.0133001776 t96_leaf0 - 0.00418375246 t96_leaf1
   - 0.00598910078 t96_leaf2 - 0.00168784079 t96_leaf3
   - 2.9545580000000002e-04 t96_leaf4 - 0.00866527576 t96_leaf5
   - 0.015390167 t96_leaf6 - 0.00214421586 t96_leaf7
   - 0.00127589866 t97_leaf0 - 0.00613600295 t97_leaf1
   - 0.00661460264 t97_leaf2 - 0.00915512908 t97_leaf3
   - 0.0114935804 t97_leaf4 - 0.00280999648 t97_leaf5
   + 1.5714949399999999e-04 t97_leaf6 + 0.00352967042 t98_leaf0
   - 0.002612588 t98_leaf1 - 0.00532040838 t98_leaf2
   - 0.00835536607 t98_leaf3 - 0.00549140666 t98_leaf4
   - 0.0153406924 t98_leaf5 - 0.00283310656 t98_leaf6
   - 0.00514997263 t99_leaf0 + 1.12850306e-04 t99_leaf1
   - 0.00678095501 t99_leaf2 - 0.0105158417 t99_leaf3
   - 1.7788047300000001e-04 t99_leaf4 - 0.00907611847 t99_leaf5
   - 0.0132063469 t99_leaf6 - 0.00414740387 t100_leaf0
   - 0.00010705 t100_leaf1 - 0.00505239749 t100_leaf2
   - 0.00760924723 t100_leaf3 - 0.00697017321 t100_leaf4
   - 0.0153234927 t100_leaf5 + 0.00183943065 t100_leaf6
   + 0.00284667825 t101_leaf0 - 0.00305897463 t101_leaf1
   - 0.0048371288 t101_leaf2 - 0.00777276279 t101_leaf3
   - 0.00729740504 t101_leaf4 - 0.00437276252 t101_leaf5
   - 0.0108348839 t101_leaf6 + 0.00289578084 t101_leaf7
   - 0.00675748289 t102_leaf0 - 0.0115915183 t102_leaf1
   - 6.6559913100000003e-04 t102_leaf2 - 0.00500206323 t102_leaf3
   - 0.00534481555 t102_leaf4 - 0.00935193244 t102_leaf5
   - 0.0022198041 t102_leaf6 - 0.0130237425 t102_leaf7
   - 0.00120564026 t103_leaf0 + 0.00263698096 t103_leaf1
   - 0.00467364257 t103_leaf2 - 0.00690822443 t103_leaf3
   - 0.00921351276 t103_leaf5 - 0.01483675 t103_leaf6
   - 0.00229868945 t103_leaf7 - 0.00513685215 t104_leaf0
   + 0.00169342884 t104_leaf1 - 0.00427744305 t104_leaf2
   - 0.00666809734 t104_leaf3 - 0.00179732742 t104_leaf4
   - 0.0108102178 t104_leaf5 - 0.018251922 t104_leaf6
   - 0.000246453 t105_leaf0 - 0.00681228098 t105_leaf1
   + 6.3724827499999997e-04 t105_leaf2 - 0.0046210885 t105_leaf3
   - 0.0115818521 t105_leaf4 - 0.00216724421 t105_leaf5
   + 7.4043386799999998e-04 t106_leaf0 - 0.00338555011 t106_leaf1
   - 0.00214867038 t106_leaf2 - 0.00580755202 t106_leaf3
   - 0.00713532 t106_leaf4 - 0.00211179908 t106_leaf5
   - 0.0111399107 t106_leaf6 - 5.5035139699999996e-04 t106_leaf7
   + 0.0010329925 t107_leaf0 - 0.00365947047 t107_leaf1
   - 5.3566e-05 t107_leaf2 + 0.0113096824 t107_leaf3
   - 0.00589204207 t107_leaf4 - 0.00318539189 t107_leaf5
   - 0.00922325905 t107_leaf6 - 0.00619193958 t107_leaf7
   - 0.00352432532 t108_leaf0 - 0.00574970292 t108_leaf1
   + 0.00838363823 t108_leaf2 - 0.00194446882 t108_leaf3
   - 0.00764171919 t108_leaf4 - 0.0120905032 t108_leaf5
   - 0.00166998513 t108_leaf6 - 0.00443243841 t109_leaf0
   - 0.00702655362 t109_leaf1 - 0.0034102404 t109_leaf2
   + 0.00230510882 t109_leaf3 - 0.00874213129 t109_leaf4
   - 0.0143644214 t109_leaf5 + 0.0051437458 t110_leaf0
   - 0.00170058198 t110_leaf1 - 1.61280142e-04 t110_leaf2
   - 0.00486472622 t110_leaf3 - 0.0123152053 t110_leaf4
   + 0.00196756376 t110_leaf5 + 5.68332151e-04 t111_leaf0
   - 0.00468563288 t111_leaf1 + 0.00125141058 t111_leaf2
   - 0.00765820034 t111_leaf3 - 0.00229961448 t111_leaf4
   + 0.00360079738 t111_leaf5 + 8.1404787500000001e-04 t111_leaf6
   - 0.00513554132 t111_leaf7 - 0.00224021426 t112_leaf0
   - 0.00498528965 t112_leaf1 - 0.0046739527 t112_leaf2
   - 0.0068448619 t112_leaf3 - 0.00702207303 t112_leaf4
   - 0.0125588859 t112_leaf5 + 2.1023828600000001e-04 t113_leaf0
   + 0.00948146824 t113_leaf1 - 0.0018581975 t113_leaf2
   - 0.00388801051 t113_leaf3 + 0.00373724801 t113_leaf4
   - 5.5686448500000003e-04 t113_leaf5 - 0.00630543 t113_leaf6
   - 0.0107451202 t113_leaf7 - 0.00415974855 t114_leaf0
   - 0.00214565592 t114_leaf1 - 0.00388128962 t114_leaf2
   - 0.00764545053 t114_leaf3 - 0.00897111651 t114_leaf4
   - 0.00134543 t114_leaf5 + 9.1905042099999998e-04 t114_leaf6
   - 4.1737567500000001e-04 t115_leaf0 - 0.00304683275 t115_leaf1
   - 0.00472104643 t115_leaf2 - 0.00132744131 t115_leaf3
   - 0.0101572676 t115_leaf4 + 0.000234428 t116_leaf0
   - 0.00339615578 t116_leaf1 - 0.00123265712 t116_leaf2
   - 0.00718968967 t116_leaf3 - 0.00746422494 t116_leaf4
   - 0.0145522766 t116_leaf5 - 0.00295641716 t117_leaf0
   - 0.00644140737 t117_leaf1 + 0.00414867699 t117_leaf2
   - 0.00118234206 t117_leaf3 - 0.00835503172 t117_leaf4
   + 0.000375675 t118_leaf0 - 0.00281060208 t118_leaf1
   - 0.00535129476 t118_leaf2 - 0.00301881251 t118_leaf3
   - 0.0068481355 t118_leaf4 - 0.0138126938 t118_leaf5
   + 0.000187739 t118_leaf6 + 0.000670078 t119_leaf0
   - 0.00270990725 t119_leaf1 - 0.00354381348 t119_leaf2
   - 0.00677919155 t119_leaf3 - 0.0070757349 t119_leaf4
   - 0.0149871279 t119_leaf5 + 0.00301929633 t119_leaf6
   + 0.0092006363 t120_leaf0 + 0.00368129625 t120_leaf1
   - 8.8703923399999999e-04 t120_leaf2 - 0.00341508933 t120_leaf3
   - 0.00187396456 t120_leaf4 - 0.00556694623 t120_leaf5
   + 0.00521240756 t120_leaf6 + 0.00175072288 t121_leaf0
   - 0.00187985448 t121_leaf1 - 0.00183300825 t121_leaf2
   - 0.00388476159 t121_leaf3 - 0.00576996757 t121_leaf4
   - 0.00990400556 t121_leaf5 - 0.0030907637 t122_leaf0
   + 6.0926680500000005e-04 t122_leaf1 - 0.00540610729 t122_leaf2
   + 0.00392843271 t122_leaf4 - 0.000350896 t122_leaf5
   - 0.00138391531 t122_leaf6 - 0.00316014444 t122_leaf7
   + 0.00341548584 t123_leaf0 - 4.63284232e-04 t123_leaf1
   - 0.00196430855 t123_leaf2 - 0.00389647717 t123_leaf3
   - 0.0129520874 t123_leaf4 - 0.00298682018 t124_leaf0
   - 0.0072060558 t124_leaf1 - 0.00925311353 t124_leaf2
   - 2.1091527e-04 t124_leaf3 - 0.00361799449 t124_leaf4
   - 0.000760529 t124_leaf5 - 0.00310696685 t124_leaf6
   + 0.00793284643 t125_leaf0 + 6.5253023099999996e-04 t125_leaf1
   - 0.00384577201 t125_leaf2 - 0.00167289877 t125_leaf3
   - 0.00302136573 t125_leaf4 - 0.00661001028 t125_leaf5
   - 0.00880378764 t125_leaf6 + 6.5708329199999996e-04 t125_leaf7
   + 0.00123799674 t126_leaf0 - 0.00128760771 t126_leaf1
   - 0.00351751363 t126_leaf2 - 3.00595e-05 t126_leaf3
   - 0.00278982939 t126_leaf4 - 0.00547107356 t126_leaf5
   - 8.0037873700000001e-04 t126_leaf6 - 0.00315009989 t127_leaf0
   - 0.00161822 t127_leaf1 - 5.2178348399999996e-04 t127_leaf2
   + 0.00459717261 t127_leaf3 - 0.00142896723 t127_leaf4
   - 0.0110801412 t127_leaf5 - 0.00166141393 t128_leaf0
   - 0.00338720507 t128_leaf1 - 9.1255619200000001e-04 t128_leaf2
   - 0.00624578586 t128_leaf3 - 6.2041438699999998e-04 t128_leaf4
   - 0.00250076712 t128_leaf5 + 0.0105986595 t128_leaf6
   + 0.00120998104 t128_leaf7 + 0.0031223658 t129_leaf0
   - 6.10348e-05 t129_leaf1 - 0.00190355245 t129_leaf2
   - 0.00600432558 t129_leaf3 - 0.00393860042 t129_leaf4
   - 0.00891969632 t129_leaf5 + 0.00235834601 t129_leaf6
   + 0.00269612228 t130_leaf0 - 0.00103835168 t130_leaf1
   - 0.00164001447 t130_leaf2 - 0.00360439275 t130_leaf3
   + 0.002517418 t130_leaf4 - 0.00474372879 t130_leaf5
   - 0.0104129575 t130_leaf6 - 0.00146162 t131_leaf0
   - 0.00438929303 t131_leaf1 + 4.7499930999999999e-04 t131_leaf2
   + 0.00550517254 t131_leaf3 - 0.0106911147 t131_leaf4
   + 0.00350799435 t132_leaf0 - 6.1333418100000003e-04 t132_leaf1
   - 0.00128906046 t132_leaf2 - 0.002917859 t132_leaf3
   - 0.00413561892 t132_leaf4 - 0.0123126982 t132_leaf5
   - 0.0016735869 t133_leaf0 + 0.00332947075 t133_leaf1
   + 6.0350529399999997e-04 t133_leaf2 + 0.00787132047 t133_leaf3
   - 0.00464145839 t133_leaf4 - 0.0109810308 t133_leaf5
   + 4.4565537200000002e-04 t133_leaf6 + 0.00245819101 t134_leaf0
   - 0.00118089619 t134_leaf1 - 0.00199113134 t134_leaf2
   - 0.00452749431 t134_leaf3 - 0.00149182265 t134_leaf4
   - 0.00903459638 t134_leaf5 - 0.00125746755 t134_leaf6
   - 4.33887268e-04 t135_leaf0 - 0.00195536041 t135_leaf1
   - 6.33826538e-04 t135_leaf2 - 0.00614282349 t135_leaf3
   - 6.99916e-05 t135_leaf4 - 0.0034769706 t135_leaf5
   - 0.00713034905 t135_leaf6 - 0.00126560079 t136_leaf0
   - 0.00313505251 t136_leaf1 - 0.00472367601 t136_leaf2
   + 0.0027630562 t136_leaf3 + 0.00115344406 t136_leaf4
   - 0.0010887438 t136_leaf5 - 0.00571710989 t136_leaf6
   - 0.00141036406 t136_leaf7 - 0.00102377287 t137_leaf0
   - 0.00361687923 t137_leaf1 + 0.00369414338 t137_leaf2
   + 0.000271843 t137_leaf3 + 0.00165812019 t137_leaf4
   - 0.0031480589 t137_leaf5 - 0.0062198597 t137_leaf6
   - 0.00310941227 t137_leaf7 + 0.00866525527 t138_leaf0
   + 0.00145104481 t138_leaf1 + 4.8963486899999996e-04 t138_leaf2
   - 0.00180146808 t138_leaf3 - 6.0332630500000003e-04 t138_leaf4
   + 0.00349245407 t138_leaf5 - 0.00193340622 t138_leaf6
   - 0.00431627687 t138_leaf7 + 6.0520943999999998e-04 t139_leaf0
   + 0.0050661792 t139_leaf1 + 0.00249463134 t139_leaf2
   - 0.0013231734 t139_leaf3 + 0.000123414 t139_leaf4
   - 0.00157989725 t139_leaf5 + 0.00420985511 t139_leaf6
   - 0.00273193768 t139_leaf7 + 0.014113903 t140_leaf0
   + 0.00220107683 t140_leaf1 - 0.00286302925 t140_leaf2
   + 0.00207335805 t140_leaf3 - 8.2270987300000001e-04 t140_leaf4
   - 0.0028552888 t140_leaf5 - 0.0049281586 t140_leaf6
   + 0.00438196026 t141_leaf0 + 1.3969614499999999e-04 t141_leaf1
   - 0.00126900186 t141_leaf2 + 0.00727035664 t141_leaf3
   - 0.00384615315 t141_leaf4 + 7.8522157899999997e-04 t141_leaf5
   - 0.0114654331 t141_leaf6 + 2.6989116999999999e-04 t142_leaf0
   - 0.00256690383 t142_leaf1 + 0.00219823141 t142_leaf2
   - 0.000475779 t142_leaf3 - 0.00356981577 t142_leaf4
   + 5.0028937400000003e-04 t142_leaf5 + 0.00362525927 t142_leaf6
   - 2.46718526e-04 t143_leaf0 + 0.00257481425 t143_leaf1
   - 7.5940834399999997e-04 t143_leaf2 + 0.00261578755 t143_leaf3
   - 0.00177594414 t143_leaf4 + 0.000577442 t143_leaf5
   - 0.00481217261 t143_leaf6 - 5.5899820299999995e-04 t143_leaf7
   - 0.00177115458 t144_leaf0 - 0.00517518399 t144_leaf1
   - 0.00994426664 t144_leaf2 + 1.6999419299999999e-04 t144_leaf3
   - 0.00151265401 t144_leaf4 + 0.00478542363 t144_leaf5
   - 0.00199135649 t144_leaf6 - 9.72747337e-04 t145_leaf0
   - 0.00408588862 t145_leaf1 - 0.00984219182 t145_leaf2
   + 0.00285486551 t145_leaf3 - 0.00190919789 t145_leaf4
   + 0.0103002079 t145_leaf5 + 0.00333934254 t146_leaf0
   + 3.3707875999999999e-04 t146_leaf2 - 0.00223647384 t146_leaf3
   - 5.65356226e-04 t146_leaf4 - 0.00233995379 t146_leaf5
   - 0.00299203279 t146_leaf6 - 0.00786591601 t146_leaf7
   + 9.5216784399999995e-04 t147_leaf0 - 0.0030378683 t147_leaf1
   + 2.8588136799999998e-04 t147_leaf2 - 0.00185165042 t147_leaf3
   - 0.000254479 t147_leaf4 - 0.00210082834 t147_leaf5
   - 0.00364395836 t147_leaf6 + 0.00316612469 t147_leaf7
   - 0.00161546981 t148_leaf0 + 0.00150665012 t148_leaf1
   - 0.00481108017 t148_leaf2 + 0.00287955301 t148_leaf3
   + 0.00402135123 t148_leaf4 - 0.00146889244 t148_leaf5
   - 0.00255940109 t148_leaf7 + 0.00353058078 t149_leaf0
   + 0.000461057 t149_leaf1 - 4.4287549100000001e-04 t149_leaf2
   - 0.00168172 t149_leaf3 - 0.00476092938 t149_leaf4
   + 0.000805794 t149_leaf5 + 0.00222717132 t149_leaf6
   - 0.002077132 t149_leaf7 - 3.6975258299999999e-04 t150_leaf0
   - 0.00246478594 t150_leaf1 - 0.00491816457 t150_leaf2
   + 4.4698212899999998e-04 t150_leaf3 - 0.0101479162 t150_leaf4
   - 3.0434445900000002e-04 t150_leaf5 - 3.9463146800000002e-04 t151_leaf0
   - 0.001897195 t151_leaf1 + 0.00143086328 t151_leaf2
   - 6.4413686100000004e-04 t151_leaf3 - 0.00115852582 t151_leaf4
   - 0.00483317534 t151_leaf5 + 6.9896539200000002e-04 t151_leaf6
   + 0.00677890144 t152_leaf0 + 4.74842789e-04 t152_leaf1
   - 3.9518237500000002e-04 t152_leaf2 - 0.001781406 t152_leaf3
   - 0.00187484198 t152_leaf4 - 0.00778567372 t152_leaf5
   + 0.00475745695 t153_leaf0 + 6.1603059299999996e-04 t153_leaf1
   + 8.3270628199999996e-04 t153_leaf2 - 9.9689071100000004e-04 t153_leaf3
   - 0.00248301518 t153_leaf4 + 0.0057357843 t153_leaf5
   - 0.00349321263 t153_leaf6 - 0.00985586923 t153_leaf7
   - 8.9347455699999995e-04 t154_leaf0 + 0.00677845534 t154_leaf1
   + 5.5634451599999996e-04 t154_leaf2 - 0.0021021862 t154_leaf3
   - 0.00225682114 t154_leaf4 - 6.2316120600000005e-04 t154_leaf5
   - 0.00532807643 t154_leaf6 - 2.6362267100000001e-04 t154_leaf7
   + 0.00180967094 t155_leaf0 - 0.00167607469 t155_leaf1
   - 0.00968632 t155_leaf2 + 9.5605512599999996e-04 t155_leaf3
   - 7.4609048900000003e-04 t155_leaf4 - 0.0040524588 t155_leaf5
   + 0.00292519806 t156_leaf0 + 0.00029486 t156_leaf1
   - 7.3783955299999997e-04 t156_leaf2 - 0.00292024761 t156_leaf3
   - 0.00331424479 t156_leaf4 + 0.00185830705 t157_leaf0
   + 2.5938218500000002e-04 t157_leaf1 - 0.00241702958 t157_leaf2
   + 0.000256894 t157_leaf3 - 0.00522483373 t157_leaf4
   - 0.000641406 t157_leaf5 - 0.00203255 t157_leaf6
   + 0.00423221616 t158_leaf0 - 1.92309337e-04 t158_leaf1
   - 8.2118174799999998e-04 t158_leaf2 - 0.00332002738 t158_leaf3
   - 0.00398281729 t158_leaf4 + 0.00134900515 t158_leaf5
   - 6.4541341300000004e-04 t159_leaf0 - 0.00260172714 t159_leaf1
   - 0.000198277 t159_leaf2 + 0.0026484991 t159_leaf3
   - 0.00240153749 t159_leaf4 - 0.00593572715 t159_leaf5
   + 0.00153787481 t160_leaf0 - 0.00161949429 t160_leaf1
   - 0.00151795091 t160_leaf2 - 3.3289464799999999e-04 t160_leaf3
   + 0.0106090568 t160_leaf4 + 0.00205192622 t160_leaf5
   - 0.00220296904 t160_leaf6 + 1.0223380399999999e-04 t161_leaf0
   - 0.00188005227 t161_leaf1 - 0.00129824982 t161_leaf2
   - 0.0041435794 t161_leaf3 + 0.00294768414 t161_leaf4
   - 0.00126280019 t161_leaf6 + 1.9385827200000001e-04 t161_leaf7
   + 0.012919019 t162_leaf0 + 6.4368912699999995e-04 t162_leaf1
   + 0.00239005685 t162_leaf2 - 0.00191149896 t162_leaf3
   - 0.000346316 t162_leaf4 - 0.00311703724 t162_leaf5
   - 0.00454606162 t162_leaf6 + 0.00159543846 t162_leaf7
   + 0.00542457867 t163_leaf0 + 0.00220030197 t163_leaf1
   + 8.0613547499999998e-04 t163_leaf2 - 0.00240981113 t163_leaf3
   + 1.5896665e-04 t163_leaf4 - 0.00115035 t163_leaf5
   - 0.00210031029 t163_leaf6 - 0.00685018115 t163_leaf7
   + 0.00310196145 t164_leaf0 - 2.77511659e-04 t164_leaf1
   - 0.00100230367 t164_leaf2 - 0.00328903459 t164_leaf3
   - 3.6127137699999999e-04 t164_leaf4 - 0.00730696833 t164_leaf5
   - 5.7028816099999997e-04 t164_leaf6 - 1.24636412e-04 t165_leaf0
   - 0.00167174893 t165_leaf1 + 0.0064482512 t165_leaf2
   + 7.3355645900000002e-04 t165_leaf3 - 0.000384105 t165_leaf4
   - 0.0100042745 t165_leaf5 + 0.00212549279 t166_leaf0
   - 8.11523e-05 t166_leaf1 - 0.00157165574 t166_leaf2
   - 0.00696214288 t166_leaf3 + 0.00297262659 t166_leaf4
   - 8.9815742099999996e-04 t166_leaf5 - 0.00258600037 t166_leaf6
   - 0.00221450138 t167_leaf0 + 0.00436712336 t167_leaf1
   - 0.00169801584 t167_leaf2 - 1.0727458e-04 t167_leaf3
   - 0.00191551482 t167_leaf4 - 0.00437737722 t167_leaf5
   + 0.00124677108 t167_leaf6 + 0.00250804448 t168_leaf0
   - 0.0026088194 t168_leaf1 + 0.00107148825 t168_leaf2
   - 0.00286741694 t168_leaf3 - 1.58232477e-04 t168_leaf4
   - 0.00233726809 t168_leaf5 + 0.00625416031 t168_leaf6
   + 0.00173844979 t168_leaf7 + 0.00317717576 t169_leaf0
   + 7.86291e-05 t169_leaf1 - 0.00114482734 t169_leaf2
   - 0.00381708145 t169_leaf3 - 0.00441873679 t169_leaf4
   - 0.00248940312 t169_leaf5 + 0.00266726571 t169_leaf6
   - 1.71102e-05 t170_leaf0 - 0.00311422232 t170_leaf1
   - 7.20411655e-04 t170_leaf2 - 0.00317858788 t170_leaf3
   - 0.000612085 t170_leaf4 - 0.00871978886 t170_leaf5
   + 7.5681641499999997e-04 t171_leaf0 - 9.1353867899999998e-04 t171_leaf1
   + 0.000280402 t171_leaf2 + 0.00757943466 t171_leaf3
   - 7.4793538100000002e-04 t171_leaf4 + 0.00278970599 t171_leaf5
   - 0.0037071018 t171_leaf6 - 0.00112119666 t172_leaf0
   + 0.000504369 t172_leaf1 - 0.00123380951 t172_leaf2
   + 0.00312549644 t172_leaf3 + 7.2426663199999996e-04 t172_leaf4
   - 0.00438592304 t172_leaf5 + 0.00170896074 t173_leaf0
   - 0.00160445133 t173_leaf1 + 7.8341521999999997e-04 t173_leaf2
   - 5.3278054099999996e-04 t173_leaf3 + 3.7611887099999998e-04 t173_leaf4
   - 0.00382088474 t173_leaf5 - 0.00357194059 t173_leaf6
   + 0.00259555969 t174_leaf0 - 0.0015318552 t174_leaf1
   - 0.0058158855 t174_leaf2 - 4.12015e-05 t174_leaf3
   + 0.0044690799 t174_leaf4 - 0.00415706169 t174_leaf5
   + 0.00230730907 t174_leaf6 + 1.2511118299999999e-04 t175_leaf0
   - 0.00139111653 t175_leaf1 + 0.000255091 t175_leaf2
   + 0.00342954299 t175_leaf3 - 0.00599129638 t175_leaf4
   - 1.2479267000000001e-04 t176_leaf0 - 0.00249303458 t176_leaf1
   + 0.00578994071 t176_leaf2 - 0.00485068699 t176_leaf3
   + 0.00115515816 t176_leaf4 + 0.00587215228 t177_leaf0
   + 3.8991554200000001e-04 t177_leaf1 + 0.00157746114 t177_leaf2
   - 3.9974474899999999e-04 t177_leaf3 - 8.31670477e-04 t177_leaf4
   + 0.00184705551 t177_leaf5 - 0.00540041598 t177_leaf6
   + 8.48346e-05 t178_leaf0 - 0.00255363388 t178_leaf1
   - 6.5386731899999995e-04 t178_leaf2 - 0.00471301377 t178_leaf3
   - 0.00342146028 t178_leaf4 - 0.00169367739 t178_leaf5
   + 1.42457e-05 t178_leaf6 + 2.2742379199999999e-04 t179_leaf0
   + 0.00790399313 t179_leaf1 - 1.7950926799999999e-04 t179_leaf2
   - 0.00307233515 t179_leaf3 - 0.00152951153 t179_leaf4
   - 0.00491567748 t179_leaf5 + 0.00122118287 t180_leaf0
   + 0.00820975285 t180_leaf1 - 4.08191e-05 t180_leaf2
   - 0.00273068831 t180_leaf3 - 0.00288064149 t180_leaf4
   + 0.0020478568 t181_leaf0 - 6.0647324399999995e-04 t181_leaf1
   + 2.9211639800000002e-04 t181_leaf2 + 0.00291145244 t181_leaf3
   - 0.00118049176 t181_leaf4 + 6.13291e-05 t181_leaf5
   - 0.00558166159 t181_leaf6 + 0.00109954388 t181_leaf7
   + 9.2028413199999996e-04 t182_leaf0 + 0.00966390129 t182_leaf1
   + 5.39777e-05 t182_leaf2 - 0.00161121285 t182_leaf3
   - 0.00147120038 t182_leaf4 - 0.0047468082 t182_leaf5
   + 0.0028024863 t182_leaf6 + 7.1964622499999997e-04 t183_leaf0
   - 0.00157024188 t183_leaf1 + 2.5297846900000003e-04 t183_leaf2
   + 0.00536873937 t183_leaf3 + 0.00229754183 t183_leaf4
   - 0.00292849424 t183_leaf5 + 6.3965824699999997e-04 t183_leaf6
   + 0.00186171045 t184_leaf0 - 0.00260638748 t184_leaf1
   + 0.00884117372 t184_leaf2 - 5.73763e-05 t184_leaf3
   - 1.7431328900000001e-04 t184_leaf4 - 0.00310243783 t184_leaf5
   - 0.00351742841 t184_leaf6 + 0.00225837831 t184_leaf7
   - 4.4683078899999999e-04 t185_leaf0 + 0.000627598 t185_leaf1
   - 0.00313549838 t185_leaf2 - 0.0046074288 t185_leaf4
   + 0.00619214866 t185_leaf5 + 0.00107866491 t186_leaf0
   - 3.9067847000000001e-04 t186_leaf1 + 0.0070162192 t186_leaf2
   + 5.5501231700000005e-04 t186_leaf3 - 0.00250983262 t186_leaf4
   + 0.00180838082 t187_leaf0 - 1.4190023600000001e-04 t187_leaf1
   - 0.00179364404 t187_leaf3 - 0.006685514 t187_leaf4
   + 0.00315739377 t188_leaf0 - 2.5762684499999999e-04 t188_leaf1
   + 1.04602e-05 t188_leaf2 - 0.00235443469 t188_leaf3
   - 0.00104418932 t188_leaf4 - 0.00432574702 t188_leaf5
   + 0.0029007718 t188_leaf6 + 1.4389079300000001e-04 t189_leaf0
   - 0.00204840861 t189_leaf1 - 0.00441745576 t189_leaf2
   - 0.0010761203 t189_leaf3 + 0.00558339572 t189_leaf5
   - 1.93742e-05 t190_leaf0 - 0.00290150847 t190_leaf1
   + 6.2799727300000003e-04 t190_leaf2 - 0.00291975122 t190_leaf3
   - 0.00489435671 t190_leaf5 + 0.001766845 t191_leaf0
   - 2.4675388699999998e-04 t191_leaf1 - 0.00258733705 t191_leaf2
   + 0.00393714756 t191_leaf3 - 0.00103988894 t191_leaf4
   + 0.00810004584 t191_leaf5 + 0.00707049156 t192_leaf0
   + 3.9305555400000002e-04 t192_leaf1 + 3.47784e-05 t192_leaf2
   - 0.00192749954 t192_leaf3 - 0.00313206157 t192_leaf4
   + 1.19908866e-04 t192_leaf5 + 0.00115089642 t193_leaf0
   - 3.28976632e-04 t193_leaf1 + 0.00403570384 t193_leaf2
   - 0.000482483 t193_leaf3 - 0.00449705077 t193_leaf4
   + 0.00110108755 t193_leaf5 + 8.9709187200000002e-04 t194_leaf0
   - 3.0653239800000002e-04 t194_leaf1 - 3.95706273e-04 t194_leaf2
   - 0.00205318257 t194_leaf3 - 0.00637000613 t194_leaf5
   - 0.000149087 t195_leaf0 - 0.00222246209 t195_leaf1
   + 0.00113943289 t195_leaf2 - 0.00153163483 t195_leaf3
   - 0.00684904354 t195_leaf4 + 0.00281639118 t196_leaf0
   - 0.000126895 t196_leaf1 - 0.00377637986 t196_leaf2
   - 0.00469454518 t196_leaf4 + 0.0011784439 t196_leaf5
   - 5.19238e-05 t197_leaf0 - 0.00520558935 t197_leaf1
   + 0.00850885641 t197_leaf2 - 8.4202672599999998e-04 t197_leaf3
   - 0.00257989042 t197_leaf4 + 0.000809497 t197_leaf5
   - 8.2912319299999996e-04 t198_leaf0 + 0.00633360865 t198_leaf1
   + 0.00209274702 t198_leaf2 - 1.52186694e-04 t198_leaf3
   - 0.00127076614 t198_leaf4 - 0.00494745 t198_leaf5
   + 0.0021195889 t198_leaf6 + 9.5552858000000004e-04 t199_leaf0
   + 0.00451930752 t199_leaf1 - 0.00139960495 t199_leaf2
   + 7.80879e-05 t199_leaf3 + 0.00143579836 t199_leaf4
   - 0.00120715122 t199_leaf5 + 0.00159203063 t199_leaf6
   - 0.00141158071 t199_leaf7 - 5.9920334000000003e-04 t200_leaf0
   + 0.00148592982 t200_leaf1 - 2.9029659300000002e-04 t200_leaf2
   - 0.00220656046 t200_leaf3 - 0.00611735275 t200_leaf4
   + 0.00132421264 t200_leaf5 - 1.5815960100000001e-04 t200_leaf6
   + 2.4878917600000002e-04 t201_leaf0 - 9.5949682900000005e-04 t201_leaf1
   - 0.00137314154 t201_leaf2 - 0.00293048169 t201_leaf3
   - 0.00148315914 t201_leaf4 + 0.00235811807 t201_leaf5
   + 0.00188683684 t202_leaf0 - 0.00120626204 t202_leaf1
   - 0.00146893808 t202_leaf2 + 0.000498484 t202_leaf3
   + 0.000201563 t202_leaf4 + 0.00265713665 t202_leaf5
   + 0.00309325196 t202_leaf6 + 0.00168144016 t203_leaf0
   + 2.7770473399999998e-04 t203_leaf1 + 2.8690722000000002e-04 t203_leaf2
   - 0.00104580517 t203_leaf3 - 0.00208795466 t203_leaf4
   - 0.0019358797 t204_leaf0 - 0.000466451 t204_leaf1
   + 0.00341286534 t204_leaf2 + 1.4187149600000001e-04 t204_leaf3
   - 0.0037308156 t204_leaf4 + 0.00208267872 t205_leaf0
   - 0.00110858853 t205_leaf1 - 1.5371915699999999e-04 t205_leaf2
   - 0.00323804119 t205_leaf3 - 0.00270723528 t205_leaf4
   + 0.00232329709 t206_leaf0 - 0.00157900387 t206_leaf1
   + 4.08742e-05 t206_leaf2 - 0.00299712177 t206_leaf3
   - 0.00458605122 t206_leaf4 + 7.0449634200000003e-04 t207_leaf0
   - 4.4845481199999999e-04 t207_leaf1 - 0.00310187368 t207_leaf2
   - 8.2245480700000005e-04 t207_leaf3 - 0.00407332554 t207_leaf4
   - 6.2502024200000001e-04 t207_leaf5 + 0.000344178 t208_leaf0
   + 0.00274625886 t208_leaf1 - 0.00302002765 t208_leaf2
   + 0.00210733595 t208_leaf3 - 0.00122400548 t208_leaf4
   + 7.0025626300000005e-04 t208_leaf5 + 0.000416688 t209_leaf0
   - 0.00129165116 t209_leaf1 - 0.00280827167 t209_leaf2
   + 0.00154348277 t209_leaf3 - 0.00309192971 t209_leaf4
   + 0.0033226558 t209_leaf5 + 5.4746324899999997e-04 t209_leaf6
   - 0.00175111857 t209_leaf7 + 7.7185686700000002e-04 t210_leaf0
   + 0.00393193169 t210_leaf1 + 0.00182028196 t210_leaf2
   - 4.5044146800000002e-04 t210_leaf3 - 2.5080732299999998e-04 t210_leaf4
   - 0.00197042758 t210_leaf5 + 0.00501973834 t210_leaf6
   - 0.00199075532 t211_leaf0 + 4.41577e-05 t211_leaf1
   - 0.00258736685 t211_leaf2 + 4.8552674700000001e-04 t211_leaf3
   - 0.00224837614 t211_leaf4 - 1.6865070199999999e-04 t211_leaf5
   - 0.00358850858 t211_leaf6 + 3.9824334099999999e-04 t212_leaf0
   - 0.00130204263 t212_leaf1 + 2.6935740600000002e-04 t212_leaf2
   + 0.00423344458 t212_leaf3 - 0.00264734239 t212_leaf4
   + 4.35688e-05 t213_leaf0 - 0.00146575435 t213_leaf1
   - 0.0020575251 t213_leaf2 - 6.2164344100000004e-04 t213_leaf3
   + 0.00396510866 t213_leaf4 - 0.00104045565 t213_leaf5
   + 1.3136088099999999e-04 t214_leaf0 + 0.00397721538 t214_leaf1
   + 0.000204474 t214_leaf2 - 0.00166797172 t214_leaf3
   - 0.00281167193 t214_leaf4 + 0.00174609944 t214_leaf5
   + 0.00181351509 t215_leaf0 - 0.00118113961 t215_leaf1
   - 1.4781074399999999e-04 t215_leaf2 + 0.00529925805 t215_leaf3
   + 3.95965384e-04 t215_leaf4 + 0.00312076556 t215_leaf5
   - 6.8092474200000003e-04 t215_leaf6 + 4.5064865800000003e-04 t215_leaf7
   - 3.2232073100000002e-04 t216_leaf0 + 5.4401281499999996e-04 t216_leaf1
   - 3.4638852200000002e-04 t216_leaf2 - 0.00571958395 t216_leaf3
   + 0.00249898317 t216_leaf4 - 0.00182865444 t216_leaf5
   + 0.0080809053 t216_leaf6 + 0.00143572979 t217_leaf0
   - 7.3735229700000004e-04 t217_leaf1 - 4.2575944099999997e-04 t217_leaf2
   - 0.00230577542 t217_leaf3 + 9.6347846600000002e-04 t217_leaf4
   - 0.00139187265 t217_leaf5 + 0.00701378658 t217_leaf6
   + 3.2639712999999999e-04 t218_leaf0 - 0.00343602058 t218_leaf1
   - 0.00116288359 t218_leaf2 + 7.2020856899999996e-04 t218_leaf3
   - 0.0017107795 t218_leaf4 + 2.8282185699999999e-04 t219_leaf0
   - 4.8923958099999999e-04 t219_leaf1 + 0.00316418475 t219_leaf2
   - 0.00253826054 t219_leaf3 - 0.000861425 t219_leaf4
   - 0.00412042346 t219_leaf5 + 0.00285098981 t219_leaf6
   + 0.00223290524 t220_leaf0 - 0.00185173831 t220_leaf1
   + 6.5922085200000003e-04 t220_leaf2 - 8.7794649899999996e-04 t220_leaf3
   - 0.000204789 t220_leaf4 + 0.00675925519 t220_leaf5
   - 0.00413662288 t220_leaf6 + 0.000691826 t220_leaf7
   - 0.00298803067 t221_leaf0 - 2.95399368e-04 t221_leaf1
   - 0.00305462419 t221_leaf2 - 2.8720847300000001e-04 t221_leaf3
   + 0.0030669393 t221_leaf4 + 0.0018934086 t221_leaf5
   + 9.91057e-05 t221_leaf6 + 0.00033978 t222_leaf0
   - 7.0456281500000004e-04 t222_leaf1 - 6.6629482900000005e-04 t222_leaf2
   - 0.00367263 t222_leaf3 - 0.0052651586 t222_leaf4
   - 8.28746e-05 t223_leaf0 - 0.00207345723 t223_leaf1
   + 3.78479715e-04 t223_leaf2 + 0.00436127791 t223_leaf3
   - 0.00253852177 t223_leaf4 + 0.00242685736 t224_leaf0
   + 1.12128e-05 t224_leaf1 + 0.00741442433 t224_leaf2
   + 7.0430443200000001e-04 t224_leaf3 - 6.9450639399999995e-04 t224_leaf4
   - 0.00333853252 t224_leaf5 - 0.00111937313 t225_leaf0
   + 1.7554829499999999e-04 t225_leaf1 + 8.8634726100000004e-04 t225_leaf2
   - 6.3224323100000005e-04 t225_leaf3 - 0.00569347944 t225_leaf5
   - 0.00308629707 t226_leaf0 + 0.00116551388 t226_leaf1
   + 0.00638457388 t226_leaf2 + 0.00278561539 t226_leaf3
   - 8.4219733300000001e-04 t226_leaf4 + 4.7463850900000002e-04 t226_leaf5
   - 0.0020496794 t226_leaf6 + 0.00182561646 t227_leaf0
   - 5.12045e-05 t227_leaf1 - 0.00545603363 t227_leaf2
   - 0.00258025201 t227_leaf3 + 0.00139692798 t227_leaf4
   - 0.00157125026 t228_leaf0 + 0.00217600097 t228_leaf1
   - 4.30023e-05 t228_leaf2 - 0.00321861 t228_leaf3
   + 0.00474104146 t228_leaf4 - 0.000530945 t228_leaf5
   - 1.3118552999999999e-04 t228_leaf6 + 0.00123206724 t228_leaf7
   + 1.7211161300000001e-04 t229_leaf0 + 0.00564642763 t229_leaf1
   + 0.00305572734 t229_leaf2 - 1.04565821e-04 t229_leaf3
   - 0.00560040306 t229_leaf4 - 0.00180907873 t230_leaf0
   + 0.000414783 t230_leaf1 + 0.00380019681 t230_leaf2
   - 0.00238212594 t230_leaf3 + 0.00263332902 t230_leaf4
   - 2.4337044999999999e-04 t230_leaf5 - 0.00139296253 t230_leaf6
   + 0.00253706891 t230_leaf7 + 6.6854519499999997e-04 t231_leaf0
   - 0.000187392 t231_leaf1 - 0.00154365727 t231_leaf2
   + 0.00114845135 t231_leaf3 - 7.1448419500000002e-04 t231_leaf4
   - 0.00384321017 t231_leaf6 - 1.5302877e-04 t232_leaf0
   + 0.00199258584 t232_leaf1 - 0.00518827187 t232_leaf2
   + 0.00805672165 t232_leaf3 - 6.7486660599999995e-04 t232_leaf4
   + 0.0001699 t233_leaf0 - 0.00143543351 t233_leaf1
   + 0.00317255803 t233_leaf2 + 4.1707514899999999e-04 t233_leaf3
   - 0.00271438062 t233_leaf4 + 0.00078528 t234_leaf0
   + 0.00543473708 t234_leaf1 - 5.62754e-05 t234_leaf2
   - 0.00211698841 t234_leaf3 + 0.0010860851 t234_leaf4
   - 0.00323869823 t234_leaf5 + 5.10413e-05 t235_leaf0
   + 0.0114278495 t235_leaf1 + 0.00157070847 t235_leaf2
   - 1.69379229e-04 t235_leaf3 - 0.00333955302 t235_leaf4
   + 0.00424171612 t235_leaf5 - 3.4061039299999998e-04 t236_leaf0
   + 5.3114263600000002e-04 t236_leaf1 + 0.00333588896 t236_leaf2
   - 0.00213530171 t236_leaf3 + 0.00329956645 t236_leaf4
   + 0.0037253676 t236_leaf5 + 7.5750111100000004e-04 t237_leaf0
   - 1.9236579799999999e-04 t237_leaf1 - 0.00273611164 t237_leaf2
   - 0.000519108 t237_leaf3 - 0.00205322867 t237_leaf4
   + 0.00270742713 t237_leaf5 - 0.00101863162 t237_leaf6
   + 0.00146927778 t238_leaf0 + 0.00104896259 t238_leaf1
   - 0.00230523665 t238_leaf2 - 0.00163850118 t238_leaf3
   - 0.000222041 t238_leaf4 + 8.2029687499999995e-04 t239_leaf0
   - 1.45177677e-04 t239_leaf1 - 0.000589217 t239_leaf2
   + 0.00330967898 t239_leaf3 - 0.00136547349 t239_leaf4
   - 0.00111974508 t240_leaf0 + 0.00315636047 t240_leaf1
   + 0.0079483632 t240_leaf2 - 1.79277151e-04 t240_leaf3
   - 0.00171055645 t240_leaf4 + 0.00253709848 t240_leaf5
   + 9.74806e-05 t240_leaf6 + 5.4062966999999997e-04 t241_leaf0
   + 0.00224333489 t241_leaf1 - 8.0562458600000005e-04 t241_leaf2
   + 2.41762e-05 t241_leaf3 - 0.00302054198 t241_leaf4
   - 0.00161260355 t241_leaf5 - 0.00233164825 t242_leaf0
   - 5.2006164299999997e-04 t242_leaf1 + 8.77002e-05 t242_leaf2
   + 0.00219372404 t242_leaf3 - 0.00269450387 t242_leaf4
   - 0.00464616437 t243_leaf0 - 0.000129712 t243_leaf1
   + 0.00453364057 t243_leaf2 + 0.00699627819 t243_leaf3
   + 2.5317183499999998e-04 t243_leaf4 + 1.86803e-06 t244_leaf0
   - 0.0042326767 t244_leaf1 + 0.00608589081 t244_leaf2
   - 5.51345e-05 t244_leaf3 - 0.00426112441 t244_leaf4
   + 0.00362295983 t245_leaf0 - 1.3680750299999999e-04 t245_leaf1
   - 0.00235953066 t245_leaf2 + 0.00181187037 t245_leaf3
   + 1.2482686700000001e-04 t245_leaf4 - 0.00197997247 t245_leaf5
   - 0.00161404128 t245_leaf6 + 0.00155731896 t246_leaf0
   - 0.00488573406 t246_leaf1 - 3.0680032800000002e-04 t246_leaf2
   + 0.00908793136 t246_leaf3 + 0.000606353 t246_leaf4
   + 0.000739019 t247_leaf0 + 0.00752654672 t247_leaf1
   + 0.00106601033 t247_leaf2 - 7.70303e-05 t247_leaf3
   - 0.00299660093 t247_leaf4 - 4.4154742500000002e-04 t247_leaf5
   - 9.1593875499999995e-04 t248_leaf0 + 0.00116635521 t248_leaf1
   - 0.00154890155 t248_leaf2 + 5.85856e-05 t248_leaf3
   - 0.00156539341 t248_leaf4 - 0.00105998071 t249_leaf0
   + 0.00410862733 t249_leaf1 + 0.000641568 t249_leaf2
   - 9.46784e-05 t249_leaf3 - 0.00339379325 t249_leaf4
   + 0.00394555414 t249_leaf5 - 1.41084456e-04 t250_leaf0
   + 0.00217523822 t250_leaf1 + 0.00177554414 t250_leaf2
   + 3.7590041799999999e-04 t250_leaf3 - 0.00248919521 t250_leaf4
   - 4.7191331400000001e-04 t250_leaf5 + 0.00104853895 t251_leaf0
   - 0.00134483026 t251_leaf1 - 2.40856418e-04 t251_leaf2
   + 0.00334266876 t251_leaf3 - 0.00212663901 t251_leaf4
   + 4.01722442e-04 t252_leaf0 + 0.0063863392 t252_leaf1
   - 0.00508262217 t252_leaf2 + 4.5748820399999998e-04 t252_leaf3
   - 2.19219757e-04 t252_leaf4 - 0.00190283894 t252_leaf5
   - 0.00102101814 t252_leaf6 + 7.7271269399999999e-04 t252_leaf7
   + 0.000123208 t253_leaf0 - 0.00126933 t253_leaf1
   - 0.00115346862 t253_leaf2 + 0.00368501944 t253_leaf3
   - 4.1151241700000001e-04 t253_leaf4 - 0.00248370715 t253_leaf5
   + 0.00259288074 t253_leaf6 - 7.9249276299999999e-04 t253_leaf7
   - 0.00359928422 t254_leaf1 + 0.00669441419 t254_leaf2
   - 4.0927831999999998e-04 t254_leaf3 - 0.00204644096 t254_leaf4
   + 2.62715e-05 t255_leaf0 - 0.00244060787 t255_leaf1
   + 0.00213455362 t255_leaf2 - 0.0011784886 t255_leaf3
   - 0.0011513104 t255_leaf4 + 0.00173618656 t255_leaf5
   + 0.0103540346 t255_leaf6 + 0.00422447035 t256_leaf0
   + 0.000352885 t256_leaf1 + 6.3060346299999997e-04 t256_leaf2
   - 0.00235657231 t256_leaf3 - 0.00330485078 t256_leaf5
   + 0.000522125 t256_leaf6 + 1.05983338e-04 t257_leaf0
   - 0.00107024 t257_leaf1 - 0.00445156312 t257_leaf2
   + 0.00136041362 t257_leaf3 - 0.00275390572 t257_leaf4
   + 0.00730741676 t257_leaf5 + 0.00163998653 t257_leaf6
   - 4.4317383400000002e-04 t258_leaf0 + 0.0031824396 t258_leaf1
   - 0.00200600806 t258_leaf2 + 9.4765366499999996e-04 t258_leaf3
   + 5.76618e-05 t258_leaf4 - 0.00184365513 t258_leaf5
   + 1.4916376700000001e-04 t259_leaf0 - 0.00121816911 t259_leaf1
   + 7.3839357400000004e-04 t259_leaf2 - 0.00247351103 t259_leaf3
   - 0.00148742343 t259_leaf4 + 0.00271147466 t259_leaf5
   - 0.00367884431 t259_leaf6 - 7.5488572499999998e-04 t260_leaf0
   + 0.00103477784 t260_leaf1 + 0.0015401385 t260_leaf2
   - 8.4440654600000001e-04 t260_leaf3 - 7.2589e-05 t260_leaf4
   + 0.00116600527 t260_leaf5 + 0.00824452285 t260_leaf6
   + 0.00110527594 t260_leaf7 + 6.98944e-05 t261_leaf0
   - 0.00118711079 t261_leaf1 - 0.00391004886 t261_leaf2
   + 0.00680468418 t261_leaf3 + 3.49231e-05 t261_leaf4
   - 0.00245717401 t262_leaf0 + 0.00891338848 t262_leaf1
   + 0.00148248312 t262_leaf2 - 8.65194e-05 t262_leaf3
   - 0.00215559849 t262_leaf4 - 3.91628e-06 t263_leaf0
   - 0.00203209231 t263_leaf1 - 0.00212156074 t263_leaf2
   + 0.00885794312 t263_leaf3 - 9.5448875799999998e-04 t263_leaf4
   + 0.00169393944 t263_leaf5 - 7.8735174599999999e-04 t263_leaf6
   + 0.00208257325 t264_leaf0 - 0.00141002284 t264_leaf1
   - 3.02452e-06 t264_leaf2 - 1.3735210800000001e-04 t264_leaf3
   + 0.00156066974 t264_leaf4 + 0.00220912346 t264_leaf5
   - 0.00164436805 t264_leaf6 - 0.00571784982 t265_leaf1
   + 0.00338470563 t265_leaf2 + 7.1774638499999999e-04 t265_leaf3
   + 2.7796e-05 t265_leaf4 - 0.00275227614 t265_leaf5
   + 1.32863235e-04 t266_leaf0 - 0.00245565944 t266_leaf1
   - 0.00464952365 t266_leaf2 - 0.00127745781 t266_leaf4
   + 0.00217112899 t266_leaf5 - 0.00461134361 t266_leaf6
   - 3.71628e-05 t267_leaf0 - 0.00354600861 t267_leaf1
   + 0.000234388 t267_leaf2 + 0.00113110698 t267_leaf3
   + 5.47729e-05 t267_leaf4 - 4.9328751600000002e-04 t267_leaf5
   + 0.00182080816 t267_leaf6 + 2.5815717500000002e-04 t268_leaf0
   + 0.00486604357 t268_leaf1 - 0.00136273087 t268_leaf2
   - 1.7245329200000001e-04 t268_leaf3 - 0.00199497235 t268_leaf4
   + 4.6567118299999999e-04 t268_leaf5 + 2.84199e-05 t269_leaf0
   - 0.00436507538 t269_leaf1 + 0.00071612 t269_leaf2
   - 0.00581615372 t269_leaf3 - 0.00366124883 t269_leaf4
   - 2.04797441e-04 t270_leaf0 - 6.27212e-05 t270_leaf1
   + 0.00991662126 t270_leaf2 + 2.8695084599999999e-04 t270_leaf3
   - 0.000387209 t270_leaf4 + 0.00685864175 t270_leaf5
   - 5.1479856500000003e-04 t270_leaf6 - 0.00399347395 t271_leaf0
   + 9.28839e-05 t271_leaf1 - 0.00124206464 t271_leaf2
   + 0.00671022292 t271_leaf3 - 4.01018e-05 t272_leaf0
   + 0.00174628594 t272_leaf1 + 2.90811e-05 t272_leaf2
   - 0.00680971192 t272_leaf3 + 0.00669199 t272_leaf4
   + 0.000877012 t273_leaf0 - 0.00102181674 t273_leaf1
   + 0.000703549 t273_leaf2 - 0.00132204569 t273_leaf3
   + 3.1101520299999999e-04 t273_leaf4 + 0.00413743081 t273_leaf5
   + 0.00547910156 t273_leaf6 + 0.00116314425 t274_leaf0
   - 0.00155050226 t274_leaf1 + 0.00411159638 t274_leaf2
   + 8.07816e-05 t274_leaf3 + 0.00411405182 t274_leaf4
   - 5.4286565899999998e-04 t274_leaf5 - 0.00353524764 t274_leaf6
   + 0.00494418154 t275_leaf0 - 5.90266893e-04 t275_leaf1
   + 0.00020774 t275_leaf2 - 8.2461681400000002e-04 t275_leaf3
   - 0.00310945185 t275_leaf4 - 1.5073586799999999e-04 t276_leaf0
   + 0.00143977604 t276_leaf1 - 1.18665695e-04 t276_leaf2
   - 0.00208752137 t276_leaf3 + 0.00187863444 t276_leaf4
   - 0.00166140741 t276_leaf5 + 0.00419626292 t277_leaf1
   - 0.00197753124 t277_leaf2 + 2.22772e-05 t277_leaf3
   - 0.00248774048 t277_leaf4 + 0.00429556705 t277_leaf5
   - 0.00404966855 t278_leaf0 + 4.3994761700000002e-04 t278_leaf1
   - 6.7100254799999995e-04 t278_leaf2 - 0.00260122912 t278_leaf3
   - 0.00450765947 t279_leaf0 - 1.50021e-05 t279_leaf1
   + 4.50328e-05 t279_leaf2 - 0.00114241708 t279_leaf3
   - 6.3868425799999998e-04 t279_leaf4 + 0.0072959708 t279_leaf5
   + 0.00092536 t280_leaf0 + 0.00325983 t280_leaf1
   - 0.00134842528 t280_leaf2 - 0.00179400097 t280_leaf3
   + 0.00237884256 t280_leaf4 + 0.00427480042 t280_leaf5
   - 8.7362e-05 t280_leaf6 - 0.00245328923 t281_leaf0
   + 8.3064951500000004e-04 t281_leaf1 - 6.6471932200000004e-04 t281_leaf2
   - 0.00345302117 t281_leaf3 + 0.000904373 t281_leaf4
   - 0.00143102929 t281_leaf5 + 0.00529265404 t281_leaf6
   - 8.4289343800000003e-04 t282_leaf0 + 1.33530149e-04 t282_leaf1
   - 0.00170033937 t282_leaf2 + 0.0083962176 t282_leaf3
   + 0.00144849194 t282_leaf4 + 0.0050717569 t283_leaf0
   + 8.15135e-05 t283_leaf1 - 0.00166475493 t283_leaf2
   + 4.7393571000000002e-04 t283_leaf3 + 0.00550966756 t283_leaf4
   + 1.02184356e-04 t284_leaf0 - 0.00418398716 t284_leaf1
   + 0.00587698072 t284_leaf3 - 0.00350028253 t284_leaf4
   - 1.5294614399999999e-04 t285_leaf0 + 0.00345792552 t285_leaf1
   - 0.00288682943 t285_leaf2 + 7.23522e-05 t285_leaf3
   - 5.9577915799999997e-04 t285_leaf4 + 0.00267944857 t285_leaf5
   + 0.00409819745 t285_leaf6 - 0.00102997245 t286_leaf0
   + 0.00733971875 t286_leaf1 - 4.14926e-05 t286_leaf2
   - 0.00115444313 t286_leaf3 + 0.00170490844 t286_leaf4
   + 0.000201732 t286_leaf5 - 6.6219631100000005e-04 t287_leaf0
   + 0.00839482527 t287_leaf1 + 1.8132438700000001e-04 t287_leaf2
   - 0.00104120956 t287_leaf3 + 0.00258273282 t287_leaf4
   + 4.2147413500000003e-04 t287_leaf5 - 7.90792e-05 t288_leaf0
   + 8.2357571199999998e-04 t288_leaf1 - 0.00149949966 t288_leaf2
   + 0.00505592441 t288_leaf3 - 0.00163665588 t288_leaf4
   + 1.17124451e-04 t289_leaf0 - 0.0013715548 t289_leaf1
   - 4.63612669e-04 t289_leaf2 + 0.00255519757 t289_leaf3
   - 0.00184373499 t289_leaf4 + 0.000111657 t290_leaf0
   + 0.00556873577 t290_leaf1 - 0.00075217 t290_leaf3
   + 0.00137335644 t290_leaf4 - 0.00373153272 t290_leaf5
   + 7.81075e-05 t291_leaf0 - 0.00132018351 t291_leaf1
   - 0.00439068256 t291_leaf2 + 0.00448417338 t291_leaf3
   - 0.00381942675 t292_leaf0 + 8.3659053800000003e-04 t292_leaf1
   + 4.9593741999999995e-04 t292_leaf2 - 0.00346720591 t292_leaf3
   + 0.000189999 t292_leaf4 + 0.00361497025 t292_leaf5
   - 0.000619329 t292_leaf6 - 2.0773333400000001e-04 t293_leaf0
   + 0.000494535 t293_leaf1 - 0.00160486752 t293_leaf2
   - 0.00185986375 t293_leaf3 + 0.000425891 t294_leaf0
   - 0.00163819222 t294_leaf1 - 8.7871332700000004e-04 t294_leaf2
   + 3.4802375000000002e-04 t294_leaf3 + 0.00280797645 t294_leaf4
   - 0.00269161072 t294_leaf5 + 3.5310318299999998e-04 t295_leaf0
   + 0.00217480911 t295_leaf1 - 0.00148125843 t295_leaf2
   - 2.74402e-05 t295_leaf3 - 0.00193362485 t295_leaf4
   + 8.22354399e-04 t296_leaf0 + 0.00315972324 t296_leaf1
   - 9.16882e-06 t296_leaf3 - 0.00402660575 t296_leaf4
   - 0.00223640329 t296_leaf5 + 1.9720335e-04 t297_leaf0
   - 0.00131427601 t297_leaf1 + 0.0011063749 t297_leaf2
   - 2.8671623999999997e-04 t297_leaf3 - 3.4311218800000003e-04 t297_leaf4
   - 0.00272509782 t297_leaf5 + 1.8100324099999999e-04 t298_leaf0
   - 0.00190339785 t298_leaf1 - 0.00159103598 t298_leaf2
   + 0.00497262506 t298_leaf3 + 6.85671e-05 t299_leaf0
   - 0.00245168828 t299_leaf1 - 0.00172633305 t299_leaf2
   + 0.00138051203 t299_leaf3 + 5.9716211400000002e-04 t300_leaf0
   + 0.00297746528 t300_leaf1 + 0.00195837137 t300_leaf2
   - 5.60414e-05 t300_leaf3 - 1.2733331800000001e-04 t300_leaf4
   - 0.00144996529 t300_leaf5 - 6.3408527100000005e-04 t301_leaf0
   + 0.00190512114 t301_leaf1 + 0.000179465 t301_leaf2
   + 0.00582458405 t301_leaf3 + 1.3174639000000001e-04 t302_leaf0
   - 0.00151354261 t302_leaf1 - 0.00318878843 t302_leaf2
   - 0.00148522411 t302_leaf3 - 2.34053819e-04 t303_leaf0
   + 0.000501033 t303_leaf1 - 0.00341795362 t303_leaf2
   + 0.00684920745 t303_leaf3 - 0.00197475567 t303_leaf4
   + 0.0030579695 t304_leaf0 - 0.00162403181 t304_leaf1
   + 0.000884202 t304_leaf2 - 6.7994825099999996e-04 t304_leaf3
   - 6.46987872e-04 t304_leaf4 + 0.00271239341 t304_leaf5
   + 3.6990552299999999e-04 t304_leaf6 - 0.00181388867 t304_leaf7
   + 0.00332105602 t305_leaf0 - 0.00103451 t305_leaf1
   - 0.00287802028 t305_leaf2 + 8.1583659600000002e-04 t305_leaf3
   - 1.106e-05 t305_leaf4 - 0.00325063406 t305_leaf5
   + 0.00385841448 t305_leaf6 + 2.9208176400000001e-04 t306_leaf0
   + 0.00154723565 t306_leaf1 - 0.00178391277 t306_leaf2
   - 0.00149962131 t306_leaf3 - 0.000373013 t306_leaf4
   + 5.0050282199999997e-04 t306_leaf5 - 2.8269420699999999e-04 t307_leaf0
   + 0.00320024835 t307_leaf1 - 0.00339808501 t307_leaf2
   + 0.00118954049 t307_leaf3 + 0.000157691 t307_leaf4
   - 8.93439108e-04 t307_leaf5 + 0.00258525694 t308_leaf0
   + 2.1449459e-04 t308_leaf1 - 0.0015174849 t308_leaf2
   - 0.00133417628 t308_leaf3 + 0.00356578059 t309_leaf1
   + 0.001733226 t309_leaf2 - 0.00161464664 t309_leaf3
   + 0.0058611678 t309_leaf4 + 0.00151542341 t309_leaf5
   - 5.93818e-05 t309_leaf6 + 0.00525320088 t309_leaf7
   + 0.00421303371 t310_leaf0 + 9.09807e-05 t310_leaf1
   - 0.00169161393 t310_leaf2 - 0.00290432549 t310_leaf3
   + 0.00100526586 t311_leaf0 - 3.0417e-05 t311_leaf1
   - 0.00154232141 t311_leaf2 - 0.00406045699 t311_leaf3
   + 0.00134723051 t311_leaf4 + 0.00126092322 t312_leaf0
   - 0.00207259902 t312_leaf1 - 2.54378e-05 t312_leaf2
   + 0.00311550545 t312_leaf3 - 1.90298451e-04 t312_leaf4
   - 0.00185268081 t312_leaf5 + 0.00550360605 t312_leaf6
   + 1.8404860700000001e-04 t313_leaf0 - 5.0170620700000002e-04 t313_leaf1
   - 0.0036345392 t313_leaf2 + 0.00506505091 t313_leaf3
   - 5.0664175099999998e-04 t314_leaf0 - 0.00414649677 t314_leaf1
   + 9.0128777100000003e-04 t314_leaf2 - 7.16914e-05 t314_leaf3
   + 0.00153901963 t314_leaf4 - 0.0016285521 t315_leaf0
   + 0.000184527 t315_leaf1 - 0.00366796879 t315_leaf2
   - 8.9817162400000002e-04 t315_leaf3 + 0.00568616297 t316_leaf0
   + 0.00154966 t316_leaf1 + 6.2755e-05 t316_leaf2
   - 0.0010637684 t316_leaf3 - 7.8023568499999997e-04 t317_leaf0
   + 1.9996751500000001e-04 t317_leaf1 - 0.0012194569 t317_leaf2
   + 0.00136812695 t317_leaf3 - 1.72585e-05 t317_leaf4
   - 0.00102306937 t317_leaf5 + 0.00111449114 t317_leaf6
   + 6.9911289000000002e-04 t318_leaf0 - 1.74267509e-04 t318_leaf1
   + 0.00335674244 t318_leaf2 - 0.00207920303 t318_leaf3
   + 0.0021494606 t318_leaf4 - 0.0015890965 t319_leaf0
   - 3.3002041200000002e-04 t319_leaf1 + 4.39889584e-04 t319_leaf2
   - 0.00163130881 t319_leaf3 + 0.00284025539 t319_leaf4
   + 0.00134695927 t319_leaf5 + 5.22823e-05 t319_leaf6
   - 8.32058e-05 t320_leaf0 + 0.00151935569 t320_leaf1
   - 0.00222260901 t320_leaf2 + 6.3210248499999996e-04 t320_leaf3
   + 3.4834956799999997e-04 t320_leaf4 + 0.00700421259 t320_leaf5
   + 3.78647e-05 t321_leaf0 + 0.00225990196 t321_leaf1
   + 0.00430645747 t321_leaf2 - 6.9922226200000003e-04 t321_leaf3
   - 0.00426454982 t322_leaf0 + 2.8576556400000002e-04 t322_leaf1
   + 0.00261347182 t322_leaf2 - 9.69784e-05 t322_leaf3
   + 0.00239233254 t322_leaf4 - 0.00160980294 t322_leaf5
   + 8.6478132300000002e-04 t323_leaf0 - 0.000174195 t323_leaf1
   + 7.9884886500000002e-04 t323_leaf2 - 0.00205194764 t323_leaf3
   + 0.00363746844 t323_leaf4 - 7.4128276899999997e-04 t324_leaf0
   + 0.00139562157 t324_leaf1 + 0.00681125559 t324_leaf2
   - 3.90483e-05 t324_leaf4 - 0.00239868346 t324_leaf5
   - 0.00194562762 t324_leaf6 - 3.3706059900000002e-04 t325_leaf0
   + 3.4190321500000002e-04 t325_leaf1 - 0.000667316 t325_leaf2
   + 0.00247688638 t325_leaf3 - 0.00211542519 t325_leaf4
   + 9.1066461799999996e-04 t325_leaf5 - 0.00268216175 t326_leaf0
   + 0.00193321437 t326_leaf1 - 2.99728e-05 t326_leaf2
   - 0.0023362583 t326_leaf3 + 1.8587360699999999e-04 t326_leaf4
   - 0.00297425967 t326_leaf5 + 7.3531042999999999e-04 t327_leaf0
   - 3.0859388100000002e-04 t327_leaf1 - 9.3609839700000005e-04 t327_leaf2
   + 4.0260184299999998e-04 t327_leaf3 - 0.00117021753 t327_leaf4
   - 0.000763738 t328_leaf0 + 0.00399998855 t328_leaf1
   + 4.3438011199999998e-04 t328_leaf2 - 4.6e-05 t328_leaf3
   - 0.00225096731 t328_leaf4 - 0.000610289 t329_leaf0
   - 0.00366452895 t329_leaf1 + 0.000160351 t329_leaf2
   + 0.00353691122 t329_leaf3 - 0.00163994601 t329_leaf4
   + 8.6750485899999999e-04 t329_leaf5 + 1.4052630300000001e-04 t330_leaf0
   - 0.00130655267 t330_leaf1 + 0.00353588653 t330_leaf2
   - 0.00280773011 t330_leaf3 - 0.00193888915 t330_leaf4
   + 0.00270961947 t330_leaf5 - 2.06613127e-04 t331_leaf0
   + 0.000531083 t331_leaf1 - 0.00126649055 t331_leaf2
   - 6.0296017899999998e-04 t331_leaf3 + 0.00281963125 t331_leaf4
   - 0.00321199186 t331_leaf5 + 3.6507035799999998e-04 t332_leaf0
   - 0.00384833338 t332_leaf1 + 0.00600482104 t332_leaf2
   + 0.00135140168 t332_leaf3 + 7.57704e-05 t332_leaf4
   - 0.00116768479 t332_leaf5 - 0.00321780471 t333_leaf0
   + 1.33061e-05 t333_leaf1 - 0.00146600895 t333_leaf2
   + 0.00141434814 t333_leaf3 - 0.00295086112 t334_leaf0
   + 3.5311156499999999e-04 t334_leaf1 + 4.48668201e-04 t334_leaf2
   - 0.00071738 t334_leaf3 + 0.00737928133 t334_leaf5
   - 1.33405512e-04 t335_leaf0 - 0.00176598807 t335_leaf1
   + 6.7748327299999998e-04 t335_leaf2 - 3.9684854e-04 t335_leaf3
   + 0.00556668174 t335_leaf4 + 2.5806477100000002e-04 t335_leaf5
   - 0.00316905254 t335_leaf6 + 6.7264837000000001e-04 t335_leaf7
   + 0.00102067774 t336_leaf0 - 1.54703e-05 t336_leaf1
   - 0.00151818292 t336_leaf2 + 0.000388333 t336_leaf3
   + 0.00302762352 t336_leaf4 + 0.00286894338 t337_leaf1
   + 0.000282882 t337_leaf2 - 5.9308618099999997e-04 t337_leaf3
   - 0.00252212468 t337_leaf4 - 4.2945073799999998e-04 t338_leaf0
   + 7.6253217399999997e-04 t338_leaf1 - 0.00127339829 t338_leaf2
   + 3.2927928299999998e-04 t338_leaf3 - 0.00275615114 t338_leaf4
   + 0.00559530035 t338_leaf5 - 9.27401e-05 t339_leaf0
   + 0.00117317867 t339_leaf1 - 0.00221094885 t339_leaf2
   - 3.7942253400000001e-04 t339_leaf3 + 0.00497656222 t339_leaf4
   + 2.1108824899999999e-04 t340_leaf0 - 0.00301919808 t340_leaf1
   + 0.00134246808 t340_leaf2 + 0.00371244503 t340_leaf3
   + 2.1406155399999999e-04 t340_leaf4 + 7.0208520600000001e-04 t340_leaf5
   - 0.00140054198 t340_leaf6 - 0.00102917186 t341_leaf0
   + 0.00236521778 t341_leaf1 - 4.2851571900000003e-04 t341_leaf2
   + 2.87718314e-04 t341_leaf3 - 0.00297491346 t341_leaf4
   + 2.0693379299999999e-04 t342_leaf0 - 0.00231820345 t342_leaf1
   + 0.0026197643 t342_leaf2 - 1.2675329300000001e-04 t342_leaf3
   - 0.00150767807 t342_leaf4 + 0.00213134 t342_leaf5
   - 2.6546846500000003e-04 t343_leaf0 + 0.00302563771 t343_leaf1
   - 1.67639e-05 t343_leaf2 - 0.00156144914 t343_leaf3
   + 0.00028628 t343_leaf4 + 0.00516349077 t343_leaf5
   - 5.1792443300000001e-04 t344_leaf0 + 0.00183490315 t344_leaf1
   + 1.51073895e-04 t344_leaf2 + 0.00358616258 t344_leaf3
   + 3.8013028100000002e-04 t344_leaf4 - 0.00158885017 t344_leaf5
   + 0.00614696089 t344_leaf7 - 4.7181834800000001e-04 t345_leaf0
   + 5.9464469100000002e-04 t345_leaf1 + 0.0043575014 t345_leaf2
   - 8.8865269200000003e-04 t345_leaf3 + 0.00112315745 t345_leaf4
   + 5.39291e-05 t345_leaf5 + 0.00479067303 t345_leaf6
   + 0.00383726717 t346_leaf0 - 0.000345306 t346_leaf1
   - 0.00177628838 t346_leaf2 + 3.7695019300000002e-04 t346_leaf3
   - 0.00427357387 t347_leaf0 - 7.2790181699999997e-04 t347_leaf1
   + 0.00128162757 t347_leaf2 - 2.17684e-05 t347_leaf3
   - 0.00319140987 t347_leaf4 + 0.000317741 t348_leaf0
   - 2.9322470099999999e-04 t348_leaf1 + 0.00184905494 t348_leaf2
   - 0.00169216574 t348_leaf4 + 1.2092176600000001e-04 t349_leaf0
   - 0.00169801223 t349_leaf1 - 0.00249352539 t349_leaf2
   + 0.00219413475 t349_leaf3 - 0.00147343019 t349_leaf4
   + 0.00739351846 t349_leaf5 + 0.00880182255 t350_leaf0
   - 0.0010005607 t350_leaf1 + 8.1808934900000005e-04 t350_leaf2
   - 0.0017667989 t350_leaf3 - 0.000268504 t350_leaf4
   + 8.5087650200000003e-04 t350_leaf5 - 0.00490342127 t351_leaf0
   + 3.97192e-05 t351_leaf1 + 0.00265966379 t351_leaf2
   - 0.00135355012 t351_leaf3 + 0.00253330939 t351_leaf4
   + 0.00444150204 t351_leaf5 - 0.00134447822 t351_leaf6
   + 8.7825988899999997e-04 t352_leaf0 + 0.00444692653 t352_leaf1
   + 2.24714546e-04 t352_leaf2 - 4.7701355699999998e-04 t352_leaf3
   - 0.00270373328 t352_leaf4 - 0.000130108 t352_leaf5
   + 9.2490127999999997e-04 t352_leaf6 + 2.96556e-05 t353_leaf0
   - 0.00403136015 t353_leaf1 - 8.0591888400000002e-04 t353_leaf2
   - 0.00484293839 t353_leaf3 + 0.000231249 t353_leaf4
   + 0.00191740342 t353_leaf5 + 0.00547052082 t353_leaf6
   + 0.00300146267 t354_leaf0 - 2.5628521699999998e-04 t354_leaf1
   - 4.9347733099999996e-04 t354_leaf2 - 0.00597884785 t354_leaf3
   + 1.4578823199999999e-04 t354_leaf4 + 0.00222745724 t354_leaf5
   - 0.00143891014 t355_leaf0 + 0.00192193524 t355_leaf1
   - 0.000162245 t355_leaf2 + 0.0032075108 t355_leaf3
   - 0.00254548201 t355_leaf4 + 0.00116121501 t355_leaf5
   - 0.00319504086 t356_leaf0 - 9.81313e-05 t356_leaf1
   + 0.000535987 t356_leaf2 + 4.7598281500000001e-04 t356_leaf3
   - 0.00205418165 t356_leaf4 + 9.12841e-05 t357_leaf0
   - 0.00140382932 t357_leaf1 - 0.00303037209 t357_leaf2
   + 0.0011735328 t357_leaf3 - 0.00441189203 t358_leaf0
   - 5.8700027900000003e-04 t358_leaf1 + 0.000305358 t358_leaf2
   - 0.00111584237 t358_leaf3 + 3.86888e-05 t359_leaf0
   - 0.00297660893 t359_leaf1 + 0.00411702273 t359_leaf2
   + 0.00191692973 t360_leaf0 - 0.00259221066 t360_leaf1
   + 0.00521037402 t360_leaf2 - 3.7170734099999998e-04 t360_leaf3
   + 3.2890876199999998e-04 t360_leaf4 + 4.6910782099999999e-04 t360_leaf5
   + 0.00495221559 t360_leaf6 - 7.3252478600000003e-04 t361_leaf0
   + 0.00187805598 t361_leaf1 + 0.00124866574 t361_leaf2
   + 5.87894e-05 t361_leaf3 - 0.00178353023 t361_leaf4
   - 3.96836724e-04 t362_leaf0 + 0.000251373 t362_leaf1
   + 5.0938624200000003e-04 t362_leaf2 + 0.00485212309 t362_leaf3
   + 0.00472554192 t362_leaf4 + 6.5289338799999995e-04 t363_leaf0
   - 1.07048458e-04 t363_leaf1 - 0.00374010275 t363_leaf2
   + 0.00285631907 t363_leaf3 - 6.6349044199999995e-04 t363_leaf5
   + 0.00193918624 t364_leaf0 + 1.78589e-05 t364_leaf1
   - 0.00129580835 t364_leaf2 + 6.6726352099999995e-04 t364_leaf3
   - 0.00272558 t364_leaf4 + 0.00198523956 t364_leaf5
   + 0.00206955103 t365_leaf0 + 2.2362953999999999e-04 t365_leaf1
   - 0.000100196 t365_leaf2 - 0.00198981119 t365_leaf3
   + 0.00262786401 t365_leaf4 - 2.3966160400000001e-04 t365_leaf5
   - 0.00343757402 t366_leaf0 - 2.7635841999999999e-04 t366_leaf1
   - 0.00172770477 t366_leaf2 + 3.8943582300000001e-04 t366_leaf3
   - 0.00233713794 t366_leaf4 + 0.00258158753 t366_leaf5
   + 0.00314283324 t367_leaf0 - 4.9106788400000002e-04 t367_leaf1
   - 2.01939e-05 t367_leaf2 + 0.00263362098 t367_leaf3
   - 7.4679357900000001e-04 t367_leaf4 - 1.88173e-05 t368_leaf0
   - 0.00212145247 t368_leaf1 + 0.00184190727 t368_leaf2
   + 0.00196801801 t368_leaf3 - 5.52263e-05 t368_leaf4
   + 7.67118e-05 t369_leaf0 - 0.00230473862 t369_leaf1
   + 0.00153496931 t369_leaf2 + 2.34563e-06 t369_leaf3
   + 0.00363209215 t369_leaf4 + 7.5855664899999997e-04 t370_leaf0
   - 2.15797481e-04 t370_leaf1 - 1.93237764e-04 t370_leaf2
   + 0.00213971199 t370_leaf3 + 0.00315045263 t370_leaf4
   + 0.00125381968 t371_leaf0 - 0.00334834447 t371_leaf1
   + 1.10160137e-04 t371_leaf2 + 0.00219300739 t371_leaf3
   - 0.00206922437 t371_leaf4 - 0.00268805376 t372_leaf0
   + 3.42555e-06 t372_leaf1 + 0.00312292925 t372_leaf2
   - 8.48112337e-04 t372_leaf3 - 0.00150795269 t373_leaf0
   + 0.000113308 t373_leaf1 - 0.00120711327 t373_leaf2
   - 3.2024615199999999e-04 t373_leaf3 + 0.00372311706 t373_leaf4
   - 7.25643e-05 t374_leaf0 + 0.00521713449 t374_leaf1
   - 0.000585855 t374_leaf2 + 2.0936223199999999e-04 t374_leaf3
   - 0.00138570718 t374_leaf4 + 9.65937e-05 t375_leaf0
   - 0.00270271418 t375_leaf1 + 0.00125075353 t375_leaf2
   - 0.00259346841 t375_leaf3 - 0.00283166743 t375_leaf4
   + 0.00382765569 t375_leaf5 + 3.2700164600000002e-04 t376_leaf0
   + 7.0507352900000002e-04 t376_leaf1 - 4.87720332e-04 t376_leaf2
   + 0.002401829 t376_leaf3 - 2.3801048499999999e-04 t376_leaf4
   - 0.00333312666 t376_leaf5 - 0.00204584724 t377_leaf0
   + 8.52824e-05 t377_leaf1 + 0.00868170056 t377_leaf2
   - 9.5805112499999995e-04 t377_leaf3 - 3.96605e-05 t377_leaf4
   + 0.00214011804 t377_leaf5 - 7.2191853500000003e-04 t378_leaf0
   + 0.00644711545 t378_leaf1 - 2.2637427899999999e-04 t378_leaf2
   + 0.00139888201 t378_leaf3 + 0.00159564312 t378_leaf4
   - 3.36083e-05 t379_leaf0 + 0.0065628821 t379_leaf1
   - 3.4053952500000002e-04 t379_leaf2 - 0.0030074548 t379_leaf3
   - 0.00105417648 t379_leaf4 + 0.000606805 t379_leaf5
   + 0.00678287912 t379_leaf7 + 0.00160081522 t380_leaf0
   - 0.00158501382 t380_leaf1 + 3.61621e-05 t380_leaf2
   - 0.00150489155 t380_leaf3 - 0.00134174118 t381_leaf0
   + 0.00161517749 t381_leaf1 - 0.00377111556 t381_leaf2
   + 0.0032501393 t381_leaf3 - 0.000536943 t381_leaf4
   - 1.00527337e-04 t381_leaf5 + 0.00127836154 t381_leaf6
   + 0.00533724902 t382_leaf0 - 0.00157153816 t382_leaf1
   + 0.00264030928 t382_leaf2 - 8.5287366500000003e-04 t382_leaf3
   + 5.78954e-05 t382_leaf4 - 0.00139504392 t382_leaf5
   + 0.00496449368 t383_leaf0 + 0.000520435 t383_leaf1
   - 0.000113552 t383_leaf2 + 0.00385202933 t383_leaf3
   + 0.00109052146 t384_leaf0 + 0.00372175407 t384_leaf1
   + 1.7456586599999999e-04 t384_leaf2 - 2.78861145e-04 t384_leaf3
   - 0.00247275061 t385_leaf0 - 2.3706641499999999e-04 t385_leaf1
   + 0.000424927 t385_leaf2 + 0.00257273531 t385_leaf3
   - 0.00120794377 t386_leaf0 + 1.91622e-05 t386_leaf1
   - 0.00115328247 t386_leaf2 + 0.0050870236 t386_leaf3
   - 0.00120143488 t387_leaf0 - 6.75443e-05 t387_leaf1
   + 0.00656570401 t387_leaf2 + 4.4562984799999998e-04 t387_leaf3
   + 0.00317341206 t388_leaf0 - 3.1511843600000002e-04 t388_leaf1
   + 7.0254743299999999e-04 t388_leaf2 + 0.00130061083 t388_leaf3
   - 9.37318e-05 t389_leaf0 - 2.24493517e-04 t390_leaf0
   + 0.00149099913 t390_leaf1 + 3.8079210199999999e-04 t390_leaf2
   + 0.00477104681 t390_leaf3 - 0.0021002607 t390_leaf4
   - 0.00474556256 t391_leaf0 + 1.06397616e-04 t391_leaf1
   + 0.00128950155 t391_leaf2 - 0.00198024139 t391_leaf3
   - 0.00220551714 t392_leaf0 - 7.9159316400000004e-04 t392_leaf1
   + 0.00243655941 t392_leaf2 + 4.33368e-05 t392_leaf3
   + 9.30295e-05 t393_leaf0 + 0.000318971 t393_leaf1
   - 0.00181898812 t393_leaf2 + 8.9859648100000002e-04 t393_leaf3
   + 0.00438095629 t393_leaf4 - 0.00110415695 t393_leaf5
   - 0.00108143117 t394_leaf0 + 0.000240016 t394_leaf1
   - 0.00269112037 t394_leaf2 - 0.00127227942 t394_leaf3
   - 6.2700750999999998e-04 t395_leaf0 + 0.00256129354 t395_leaf1
   - 9.60188801e-04 t395_leaf2 - 8.2638970299999997e-04 t395_leaf3
   + 9.78179e-05 t395_leaf4 + 0.00547470525 t395_leaf5
   - 0.00144260796 t396_leaf0 + 0.00712255342 t396_leaf1
   - 0.00290469173 t396_leaf2 - 6.26566e-06 t396_leaf3
   - 0.00182439061 t396_leaf4 + 0.00426064339 t397_leaf0
   + 4.7887614300000001e-04 t397_leaf1 - 4.5006437e-04 t397_leaf2
   + 9.6324470400000004e-04 t397_leaf3 - 0.000299558 t397_leaf4
   - 1.42352554e-04 t398_leaf0 - 0.00262321695 t398_leaf1
   + 2.8447739900000003e-04 t398_leaf2 + 0.00263429014 t398_leaf3
   - 0.00121863186 t398_leaf4 + 0.00278328662 t399_leaf0
   - 0.00416475628 t399_leaf1 + 0.00105785485 t399_leaf2
   + 1.56203678e-04 t399_leaf3 - 0.0017345543 t399_leaf4
   + 1.80366143e-04 t400_leaf0 - 6.7530939099999998e-04 t400_leaf1
   - 0.00415685633 t401_leaf0 - 4.64326e-05 t401_leaf1
   - 0.000294873 t402_leaf0 + 3.24683497e-04 t402_leaf1
   + 0.00245647505 t402_leaf2 + 0.00145904918 t402_leaf3
   + 4.8904388700000002e-04 t403_leaf0 + 0.0045106574 t403_leaf1
   - 6.2082329499999996e-04 t403_leaf2 + 0.00023057 t403_leaf3
   - 0.00197943766 t403_leaf4 - 3.99703e-05 t403_leaf5
   - 0.00350812543 t404_leaf0 + 6.4876873499999997e-04 t404_leaf1
   - 0.00140824053 t404_leaf2 - 5.68754e-05 t404_leaf3
   + 4.99136e-05 t405_leaf0 + 0.00204165909 t405_leaf1
   - 3.8002262599999998e-04 t405_leaf2 - 0.00228364579 t405_leaf3
   + 0.00473556621 t405_leaf4 - 0.00199693535 t406_leaf0
   + 1.19875e-05 t406_leaf1 + 0.00363000715 t406_leaf2
   - 0.00246399362 t406_leaf3 + 0.00274907309 t407_leaf0
   - 1.4746371e-04 t407_leaf1 + 7.34858e-05 t407_leaf2
   - 0.00498379534 t407_leaf3 + 0.00110345078 t407_leaf4
   + 0.000189094 t408_leaf0 - 0.00168289838 t408_leaf1
   + 0.00551425526 t408_leaf2 - 0.00200066157 t408_leaf4
   - 2.5957028299999999e-04 t409_leaf0 + 0.00190517481 t409_leaf1
   - 0.00121517503 t409_leaf2 + 3.78309545e-04 t409_leaf3
   - 4.4606361200000001e-04 t409_leaf4 + 3.04573e-06 t409_leaf5
   + 0.00228598155 t409_leaf6 + 3.91616e-07 t410_leaf0
   - 0.00141981733 t410_leaf1 + 0.00150766619 t410_leaf2
   - 7.9127942400000002e-04 t410_leaf3 - 0.00320267351 t410_leaf4
   + 1.8805012199999999e-04 t411_leaf0 - 0.000512627 t411_leaf1
   + 7.7265437000000005e-04 t412_leaf0 - 7.7671126900000003e-04 t412_leaf1
   - 3.9654344399999998e-04 t412_leaf2 - 0.00231973804 t412_leaf3
   + 0.00494314358 t412_leaf4 + 7.7955337500000001e-04 t412_leaf5
   - 8.857e-07 t412_leaf6 + 0.00389747368 t412_leaf7
   + 0.00152070145 t413_leaf0 + 0.00172871 t413_leaf1
   - 0.00173669134 t413_leaf2 - 1.61141666e-04 t413_leaf3
   + 0.000646531 t414_leaf0 - 1.4146116200000001e-04 t414_leaf1
   - 0.00180197111 t414_leaf2 + 0.00139761146 t414_leaf3
   + 4.6959624199999998e-04 t415_leaf0 - 1.36403367e-04 t415_leaf1
   + 2.9055812000000001e-04 t415_leaf2 - 0.0025910174 t415_leaf3
   - 0.00157557649 t415_leaf4 - 2.21434137e-04 t416_leaf0
   + 6.9820653899999995e-04 t416_leaf1 + 0.00756653631 t416_leaf2
   + 0.00502013043 t416_leaf4 + 8.59507e-05 t417_leaf0
   - 0.00267804693 t417_leaf1 + 0.0044700494 t417_leaf2
   - 8.9800538299999999e-04 t417_leaf3 - 2.5458139099999999e-04 t418_leaf0
   + 4.57603746e-04 t418_leaf1 - 0.00320580602 t418_leaf2
   + 0.00795663614 t418_leaf3 + 5.5003969500000004e-04 t418_leaf4
   + 3.78560246e-04 t418_leaf5 + 8.59673e-05 t419_leaf0
   - 0.00177635369 t419_leaf1 + 0.00368267787 t419_leaf2
   - 9.1275555299999997e-04 t419_leaf3 - 0.00135646923 t419_leaf4
   + 3.2427353999999999e-04 t420_leaf0 - 0.00301549118 t420_leaf1
   - 1.2987822999999999e-04 t420_leaf2 + 0.00104806328 t420_leaf3
   + 3.1800829999999999e-04 t420_leaf4 + 0.0050214557 t420_leaf5
   - 0.000836184 t421_leaf0 + 2.88967e-05 t421_leaf1
   + 0.00344600109 t421_leaf2 - 0.000259823 t421_leaf3
   + 0.00632672245 t421_leaf4 + 0.00320825446 t422_leaf0
   - 4.64244e-05 t422_leaf1 - 0.00273592724 t422_leaf2
   + 0.00374342175 t422_leaf3 + 0.00186663563 t423_leaf0
   - 0.00103833876 t423_leaf1 - 0.0031411692 t423_leaf2
   - 0.00127586932 t423_leaf3 + 1.4547714100000001e-04 t423_leaf4
   - 0.00108162488 t423_leaf5 - 6.94545e-05 t424_leaf0
   + 0.00182509155 t424_leaf1 + 0.00301644742 t424_leaf2
   + 0.0036216774 t424_leaf3 - 3.48545e-05 t424_leaf4
   - 0.00219045556 t424_leaf5 - 3.43347e-05 t425_leaf0
   + 0.0032104617 t425_leaf1 + 5.22439e-05 t426_leaf0
   + 0.00329539971 t426_leaf1 + 3.0992666099999999e-04 t427_leaf0
   - 0.00275499886 t427_leaf1 + 8.6525111700000001e-04 t427_leaf2
   + 8.32785e-05 t427_leaf3 + 0.0027897926 t427_leaf4
   - 0.00266644289 t427_leaf5 + 0.00491890591 t428_leaf0
   + 3.44942e-05 t428_leaf2 - 0.00150969205 t428_leaf3
   - 0.00136693777 t428_leaf4 + 0.00437592762 t428_leaf5
   + 3.8764762600000002e-04 t428_leaf6 - 0.0025721821 t429_leaf0
   + 3.97592e-05 t429_leaf1 + 0.00432777498 t429_leaf2
   - 0.00187214371 t429_leaf3 - 0.00222212099 t430_leaf0
   + 5.2102387399999999e-04 t430_leaf1 - 7.3299137900000001e-04 t430_leaf2
   + 9.5856381800000004e-04 t430_leaf3 + 6.7072466499999998e-04 t430_leaf4
   + 0.00385455438 t430_leaf5 + 0.00654841587 t431_leaf1
   + 5.28761e-06 t431_leaf2 + 0.000247141 t432_leaf0
   - 5.1146052999999998e-04 t432_leaf1 - 0.00193588401 t432_leaf2
   - 0.00226737605 t432_leaf3 - 2.16459e-05 t433_leaf0
   - 0.00175803481 t433_leaf1 - 0.00634577405 t433_leaf2
   + 5.9201044499999998e-04 t433_leaf3 + 0.00232717418 t433_leaf4
   + 5.61708e-06 t434_leaf0 - 0.00131252338 t434_leaf1
   + 0.00315116975 t434_leaf2 - 0.00241471734 t434_leaf3
   + 0.00223937281 t435_leaf0 - 0.00118989742 t435_leaf1
   - 6.47069e-05 t435_leaf2 + 0.00282373372 t435_leaf3
   - 0.00168910832 t435_leaf4 - 0.002575533 t436_leaf0
   + 0.00133165135 t436_leaf1 + 2.40456e-05 t436_leaf2
   + 5.71794e-05 t437_leaf0 - 0.00538752181 t437_leaf1
   + 7.8945420700000002e-04 t437_leaf2 + 0.000693245 t438_leaf0
   + 6.4451037900000005e-04 t438_leaf1 - 0.00371983647 t438_leaf2
   + 4.79975e-05 t438_leaf3 - 7.2295783300000002e-04 t438_leaf4
   + 0.00199932046 t439_leaf0 - 6.7605083999999998e-04 t439_leaf1
   - 0.00367578329 t439_leaf2 + 8.47897e-05 t439_leaf4
   - 0.00281977304 t440_leaf0 + 1.5814960400000001e-04 t440_leaf1
   - 0.00131583842 t440_leaf2 - 0.00219845492 t440_leaf3
   + 0.00146280951 t441_leaf0 - 0.0034007614 t441_leaf1
   - 0.00434158184 t441_leaf2 + 9.19451e-05 t441_leaf3
   + 0.00197326066 t441_leaf4 - 1.66449e-05 t441_leaf5
   + 0.00688586431 t441_leaf7 + 3.7600382299999999e-04 t442_leaf0
   - 2.0844970999999999e-04 t442_leaf1 + 7.9791265300000001e-04 t442_leaf2
   - 0.00220656442 t442_leaf3 - 0.00307575078 t442_leaf4
   - 1.30492728e-04 t443_leaf0 + 0.00187512115 t443_leaf1
   - 0.00177436136 t443_leaf2 + 0.00298245926 t443_leaf3
   + 6.9552159399999995e-04 t444_leaf0 - 5.1517057000000002e-04 t444_leaf1
   + 0.00233350741 t444_leaf2 + 4.7444753000000002e-04 t444_leaf3
   + 0.000463418 t445_leaf0 - 0.000601694 t445_leaf1
   - 0.000777072 t445_leaf2 + 4.2710779200000001e-04 t445_leaf3
   + 0.00239563175 t445_leaf4 - 0.00136682158 t446_leaf0
   + 0.00126436027 t446_leaf1 - 1.80204821e-04 t446_leaf2
   - 0.00414080173 t446_leaf3 + 0.00171820261 t446_leaf4
   - 4.52323e-06 t447_leaf0 + 8.0886768399999998e-04 t447_leaf1
   - 0.00105428172 t447_leaf2 + 0.00263110059 t447_leaf3
   - 7.1662874000000005e-04 t447_leaf4 - 3.1418359100000002e-04 t448_leaf0
   + 0.000686638 t448_leaf1 - 8.1435777199999999e-04 t448_leaf2
   + 5.4975738699999997e-04 t448_leaf3 + 3.6176459999999998e-04 t448_leaf4
   - 0.0032484741 t448_leaf5 + 5.8426e-05 t449_leaf0
   - 4.13923e-05 t450_leaf0 + 2.97588e-05 t451_leaf0
   - 3.6922449400000001e-04 t451_leaf1 + 0.0080338018 t451_leaf2
   - 0.00217077858 t451_leaf3 - 0.00123852608 t452_leaf0
   + 6.49114e-05 t452_leaf1 - 0.00448611984 t452_leaf2
   + 0.00212780619 t452_leaf3 - 0.00132244348 t453_leaf0
   - 5.93736e-05 t453_leaf1 + 0.00237209839 t453_leaf2
   + 0.00567519572 t453_leaf3 + 2.5115342599999998e-04 t453_leaf4
   + 0.000287208 t454_leaf0 - 0.00163624738 t454_leaf1
   - 0.00580256525 t454_leaf2 + 4.18101e-05 t454_leaf3
   - 0.00229766546 t454_leaf4 - 2.6239032699999998e-04 t454_leaf5
   - 0.000358522 t455_leaf0 + 6.2749238e-04 t455_leaf1
   + 3.8592985900000001e-04 t455_leaf2 - 0.00233605481 t455_leaf3
   + 2.68741e-05 t456_leaf0 + 1.3751430399999999e-04 t457_leaf0
   - 0.00357986824 t457_leaf1 + 0.00100339076 t458_leaf0
   + 2.37025e-05 t458_leaf1 + 0.0031154172 t458_leaf2
   + 9.1684883199999998e-04 t458_leaf3 - 5.6032015700000003e-04 t458_leaf4
   - 8.1259087900000004e-04 t458_leaf5 + 0.00521456264 t458_leaf6
   - 0.000177668 t459_leaf0 - 0.00198494294 t459_leaf1
   + 4.7030529800000002e-04 t459_leaf2 - 0.00190337596 t459_leaf3
   + 0.00126841757 t459_leaf4 + 1.95602e-06 t460_leaf0
   - 0.0036002167 t461_leaf0 + 4.2357115300000001e-04 t461_leaf1
   + 1.19831e-05 t461_leaf2 + 0.0034093447 t461_leaf3
   - 0.00228496548 t461_leaf4 + 7.1057135899999999e-04 t461_leaf5
   + 2.97571e-05 t462_leaf0 - 0.00345921866 t462_leaf1
   + 0.00173886551 t463_leaf0 - 8.4039987999999998e-04 t463_leaf1
   - 2.65169e-07 t463_leaf2 - 0.00121164601 t463_leaf3
   + 0.0057831225 t463_leaf4 - 2.85875256e-04 t463_leaf6
   + 0.000243084 t464_leaf0 - 0.00163531234 t464_leaf1
   - 8.0429756800000005e-04 t464_leaf2 - 0.000274161 t465_leaf0
   - 0.00185714697 t465_leaf1 - 0.0013374798 t465_leaf2
   + 0.00269340235 t465_leaf3 + 1.6473446200000001e-04 t465_leaf4
   - 0.00110197626 t466_leaf0 - 0.00243919599 t466_leaf2
   - 0.00171319663 t466_leaf3 - 8.61652e-05 t467_leaf0
   - 0.00201584864 t468_leaf1 - 1.4623178899999999e-04 t469_leaf0
   - 0.00262473733 t469_leaf1 + 0.00411999319 t469_leaf3
   + 0.00104126 t469_leaf4 + 0.00165843463 t470_leaf0
   + 2.04971e-05 t470_leaf1 + 6.20711e-05 t470_leaf2
   - 0.00215242896 t470_leaf3 - 1.6205174300000001e-04 t471_leaf0
   + 0.00247954438 t471_leaf1 + 0.00107156264 t471_leaf2
   - 0.00225155405 t471_leaf3 + 3.9043166899999999e-04 t472_leaf0
   - 4.5918437499999998e-04 t472_leaf1 - 0.00173296977 t472_leaf2
   - 6.1874889100000004e-04 t472_leaf3 - 6.89865e-05 t473_leaf0
   + 0.00143817125 t473_leaf1 - 0.00106542837 t473_leaf2
   + 0.00325134071 t473_leaf3 + 6.66023e-07 t474_leaf0
   - 0.00204063929 t474_leaf1 + 0.00384825096 t474_leaf2
   + 8.0576277099999997e-04 t475_leaf0 - 0.000455842 t475_leaf1
   - 0.0022178581 t475_leaf2 + 0.00150208129 t475_leaf3
   + 0.000198023 t475_leaf4 + 3.46003e-07 t476_leaf0
   - 0.000172171 t477_leaf0 + 0.00207812968 t477_leaf1
   + 2.8911558899999998e-04 t477_leaf2 + 0.00457008276 t477_leaf3
   - 2.1925437599999999e-04 t477_leaf4 + 8.52036e-05 t478_leaf0
   - 0.0020142328 t478_leaf1 + 0.00318133878 t478_leaf2
   - 0.00203289371 t478_leaf3 - 0.000152586 t479_leaf0
   + 0.00106350111 t479_leaf1 - 0.00195198331 t479_leaf2
   + 0.00534718297 t479_leaf3 - 3.81939e-05 t480_leaf0
   - 0.00309209432 t480_leaf1 + 0.00538855186 t481_leaf0
   + 8.9415383999999997e-04 t481_leaf1 - 0.000220437 t481_leaf2
   + 0.000995662 t481_leaf3 - 1.9299924299999999e-04 t482_leaf0
   + 0.00452123862 t482_leaf2 - 5.3506e-05 t483_leaf0
   + 1.71639898e-04 t484_leaf0 - 0.00253849034 t484_leaf1
   - 1.3018069100000001e-04 t485_leaf0 + 7.2247558300000004e-04 t486_leaf0
   - 1.1292036699999999e-04 t486_leaf1 - 0.00198807125 t487_leaf1
   + 1.19147458e-04 t488_leaf0 - 0.00316321198 t488_leaf1
   + 0.00140254304 t488_leaf2 + 0.0014437536 t488_leaf3
   - 1.44225e-05 t489_leaf0 - 0.00228663697 t489_leaf1
   + 0.0031835807 t489_leaf2 - 0.00376671134 t489_leaf3
   + 6.57652e-05 t490_leaf0 - 0.00104805315 t490_leaf1
   - 0.000508158 t491_leaf0 + 0.00129454106 t491_leaf1
   - 0.00323612848 t491_leaf2 + 2.0250660599999999e-04 t491_leaf3
   - 0.00195706962 t491_leaf4 + 0.00316212093 t491_leaf5
   - 2.7138830000000001e-04 t491_leaf6 - 2.9544e-05 t492_leaf0
   - 0.00192276924 t492_leaf1 + 8.6255633499999996e-04 t492_leaf2
   - 4.47638769e-04 t493_leaf0 - 0.00270040706 t493_leaf1
   + 9.0458255699999996e-04 t493_leaf2 - 1.18871e-05 t493_leaf3
   - 0.000421003 t493_leaf4 + 0.00427162275 t493_leaf5
   + 1.8322572599999999e-04 t494_leaf0 - 6.0118344799999998e-04 t494_leaf1
   + 0.000480627 t494_leaf2 + 0.00508810813 t494_leaf3
   - 0.00232945476 t494_leaf4 - 0.00351812341 t495_leaf0
   - 7.52892e-05 t495_leaf1 - 0.00204704143 t496_leaf0
   - 1.01125501e-04 t496_leaf1 + 0.00132059422 t496_leaf2
   + 0.00391639443 t496_leaf3 + 0.00117375248 t497_leaf0
   - 2.1947448799999999e-04 t497_leaf1 + 6.4531661299999999e-04 t497_leaf2
   + 0.00379510014 t497_leaf3 + 2.7274276400000001e-04 t498_leaf0
   - 0.00162110582 t498_leaf1 - 5.3577014500000005e-04 t498_leaf2
   + 0.000771873 t499_leaf0 - 2.29674e-05 t499_leaf1
   + 0.00141431915 t499_leaf2 - 7.1096385399999995e-04 t499_leaf3
   - 6.28214e-05 t500_leaf0 - 0.00146643142 t500_leaf1
   + 1.21068108e-04 t500_leaf2 - 8.4656960100000002e-04 t500_leaf3
   + 0.00483638374 t500_leaf4 - 0.00120739697 t501_leaf0
   + 0.00682709878 t501_leaf1 + 2.47427e-05 t501_leaf2
   + 0.00339207076 t501_leaf3 - 0.000130209 t502_leaf0
   - 0.00387703255 t502_leaf1 - 0.000267761 t502_leaf2
   + 0.00165843207 t503_leaf0 - 9.6027134000000001e-04 t503_leaf1
   - 0.00325666019 t503_leaf2 + 2.4958e-05 t503_leaf3
   - 6.2784343000000002e-04 t503_leaf4 + 0.00721387658 t504_leaf1
   - 7.32871e-05 t504_leaf2 + 0.00310847163 t505_leaf0
   + 4.87285e-05 t505_leaf1 - 0.00224208971 t506_leaf0
   + 7.92494e-05 t506_leaf1 + 0.00264146156 t506_leaf2
   + 9.08444e-05 t507_leaf0 - 0.00192732 t507_leaf1
   - 3.00707e-06 t508_leaf0 - 1.0663058499999999e-04 t508_leaf1
   + 0.00512077287 t508_leaf2 + 0.00184420589 t509_leaf0
   - 0.00111641421 t509_leaf1 + 0.000386402 t509_leaf2
   - 5.33002429e-04 t509_leaf3 - 6.03219e-05 t510_leaf0
   - 1.21199e-05 t511_leaf0 - 2.35171e-05 t512_leaf0
   - 0.00186344166 t513_leaf0 + 4.04161e-05 t513_leaf1
   - 0.00247267936 t513_leaf2 + 0.0022229068 t513_leaf3
   - 0.00161224476 t513_leaf4 + 0.0061714137 t513_leaf5
   + 2.31862e-05 t514_leaf0 + 0.00170712452 t514_leaf1
   - 7.44405e-05 t515_leaf0 + 0.00113064644 t515_leaf1
   - 0.00139407802 t515_leaf2 + 0.00122039672 t515_leaf3
   - 1.4918310600000001e-04 t516_leaf0 + 5.9019151400000001e-04 t516_leaf1
   - 5.4082030000000001e-04 t516_leaf2 - 0.00326508237 t516_leaf3
   + 0.00254325708 t516_leaf4 - 0.00358489947 t517_leaf0
   + 5.07217e-05 t517_leaf1 - 0.00380229694 t518_leaf0
   - 6.86585e-05 t518_leaf1 - 0.00415793853 t518_leaf2
   + 5.8044283700000004e-04 t518_leaf3 - 0.000246338 t519_leaf0
   - 1.10018125e-04 t519_leaf1 + 8.0881488999999995e-04 t519_leaf2
   - 0.00197641645 t519_leaf3 + 0.000576888 t520_leaf0
   + 1.8700484399999999e-04 t520_leaf1 - 4.7090044199999998e-04 t520_leaf2
   + 1.8928712200000001e-04 t520_leaf3 - 0.000934626 t520_leaf4
   + 0.00559894601 t520_leaf5 + 0.00435374817 t521_leaf0
   + 0.00100400555 t521_leaf1 - 1.6483814399999999e-04 t521_leaf2
   + 0.00136562064 t521_leaf3 - 1.9500232899999999e-04 t521_leaf4
   - 9.1134762599999995e-04 t522_leaf0 + 0.00685840938 t522_leaf1
   + 8.19049e-05 t522_leaf2 - 6.8828463600000001e-04 t522_leaf3
   - 0.00226121605 t522_leaf4 - 3.0266810800000003e-04 t523_leaf0
   - 0.00227730023 t523_leaf1 - 1.1595439699999999e-04 t523_leaf2
   + 0.00147318421 t523_leaf3 + 6.1109068299999999e-04 t523_leaf4
   + 0.00543470029 t523_leaf5 + 0.00028569 t524_leaf0
   + 4.2371911699999999e-04 t524_leaf1 + 0.00544318045 t524_leaf2
   - 1.0008324200000001e-04 t524_leaf3 + 0.00254683103 t524_leaf4
   - 0.00109998893 t524_leaf5 + 9.90723e-05 t525_leaf0
   + 0.00384346186 t526_leaf0 - 0.00116946804 t526_leaf1
   + 5.96244e-05 t526_leaf2 - 0.00276997406 t527_leaf0
   + 1.8285043200000001e-04 t527_leaf1 - 1.11144669e-04 t528_leaf0
   + 4.4118432600000001e-04 t528_leaf1 + 0.00334722945 t528_leaf2
   - 8.4246130399999995e-04 t528_leaf3 - 0.00453419099 t528_leaf5
   + 1.37540832e-04 t529_leaf0 + 1.8989e-05 t530_leaf0
   + 0.00134173874 t531_leaf0 - 6.86817e-05 t531_leaf1
   + 0.000236721 t532_leaf0 + 0.00189757533 t532_leaf1
   - 0.00236287061 t532_leaf2 + 7.6756789300000004e-04 t532_leaf4
   - 5.6134210900000005e-04 t532_leaf5 + 2.68008e-05 t533_leaf0
   - 1.15725e-06 t534_leaf0 - 2.104218e-04 t535_leaf0
   - 0.00342705147 t535_leaf1 + 0.00107785314 t535_leaf2
   + 1.36097e-05 t535_leaf3 + 4.29978e-05 t535_leaf4
   + 0.00318933441 t535_leaf5 + 9.64673e-05 t536_leaf0
   - 2.69082346e-04 t537_leaf0 + 8.26378e-05 t537_leaf1
   + 0.0023555595 t537_leaf2 + 0.00445488654 t537_leaf3
   - 2.7167439100000002e-04 t538_leaf0 - 0.00265701581 t538_leaf1
   + 3.4636456999999999e-04 t538_leaf2 - 3.3081378200000002e-04 t539_leaf0
   - 0.00274898857 t539_leaf1 + 2.39361441e-04 t539_leaf2
   - 0.00272144191 t539_leaf3 + 0.00290202652 t539_leaf4
   - 0.00300089782 t540_leaf0 - 0.000108121 t540_leaf1
   + 0.0017879518 t540_leaf2 + 6.0514e-05 t540_leaf3
   - 8.8223343499999995e-04 t540_leaf4 + 5.69155e-05 t541_leaf0
   - 0.00339462096 t541_leaf1 - 6.42367e-05 t542_leaf0
   - 2.5288140799999999e-04 t543_leaf0 + 0.0010025769 t543_leaf1
   - 0.00283249561 t543_leaf2 + 3.3867624000000001e-04 t543_leaf3
   + 0.00306710764 t543_leaf4 - 7.65567e-05 t544_leaf0
   - 0.00184279948 t544_leaf1 + 0.00479606818 t544_leaf2
   + 8.9439464499999999e-04 t544_leaf3 + 7.24804e-05 t544_leaf4
   + 0.00395616749 t544_leaf5 + 3.17056e-05 t545_leaf0
   + 0.00360498647 t545_leaf1 + 6.5339944599999999e-04 t545_leaf2
   + 7.84719e-06 t546_leaf0 + 0.00170029944 t546_leaf1
   - 0.0032615955 t546_leaf2 + 0.00123939686 t547_leaf0
   - 7.4015767299999999e-04 t547_leaf1 + 0.00283605349 t547_leaf2
   - 0.0012357044 t547_leaf3 - 7.32245e-05 t547_leaf4
   - 5.70562e-05 t548_leaf0 - 1.7646524100000001e-04 t549_leaf0
   + 6.1304180399999997e-04 t549_leaf1 + 0.0015270228 t549_leaf2
   + 0.00141557353 t549_leaf3 + 0.00572434859 t550_leaf0
   + 1.40036864e-04 t550_leaf2 - 0.00061663 t550_leaf3
   + 2.04373e-05 t551_leaf0 + 2.22386734e-04 t552_leaf0
   - 0.00146948127 t552_leaf1 + 0.00258716452 t552_leaf2
   - 0.00152558158 t552_leaf3 + 7.28039e-06 t553_leaf0
   - 5.88867e-05 t554_leaf0 - 5.51483e-05 t555_leaf0
   + 8.74901e-05 t556_leaf0 - 4.65873e-05 t557_leaf0
   - 0.00167206477 t557_leaf1 + 8.72182893e-04 t558_leaf0
   - 1.8661520200000001e-04 t558_leaf1 + 0.00168651133 t558_leaf2
   - 0.00171373505 t558_leaf3 + 1.50501146e-04 t559_leaf0
   - 6.8871170499999995e-04 t559_leaf1 - 1.44046426e-04 t560_leaf0
   + 0.00230896519 t560_leaf1 + 0.00673072506 t560_leaf2
   + 3.90996e-05 t560_leaf3 + 0.000521895 t560_leaf4
   - 0.00224030809 t561_leaf0 + 0.00156506838 t561_leaf1
   - 1.11340523e-04 t561_leaf2 + 0.00263064099 t562_leaf0
   + 2.6523132699999999e-04 t562_leaf1 - 0.000171681 t562_leaf2
   - 9.80268e-05 t563_leaf0 + 0.000144216 t564_leaf0
   + 6.02915e-05 t565_leaf0 - 0.00103799859 t565_leaf1
   - 0.00168984651 t566_leaf0 - 2.29937e-05 t566_leaf1
   + 0.00126491638 t566_leaf2 + 8.60518e-05 t566_leaf3
   - 0.0013656615 t566_leaf4 - 0.00128633669 t567_leaf0
   + 0.00571589591 t567_leaf1 - 7.56625e-05 t567_leaf3
   - 0.00229918305 t568_leaf0 - 0.00123682828 t568_leaf1
   + 0.00138910685 t568_leaf2 - 2.1650672700000001e-04 t568_leaf3
   + 5.33979e-05 t569_leaf0 - 0.00281456253 t569_leaf1
   + 0.00248410949 t569_leaf2 - 5.70085e-05 t570_leaf0
   + 2.38797e-05 t571_leaf0 - 0.00219930476 t571_leaf1
   - 0.00192137 t571_leaf2 + 0.00638814084 t571_leaf4
   - 0.000136733 t572_leaf0 + 6.63496263e-04 t572_leaf1
   - 0.0022176723 t572_leaf2 - 0.00196523033 t572_leaf3
   - 0.00164257677 t573_leaf1 - 7.18801e-07 t574_leaf0
   - 0.00226514856 t575_leaf0 - 1.9549712300000001e-04 t575_leaf1
   + 7.06173e-05 t575_leaf2 + 0.0040844162 t575_leaf3
   + 0.00163173291 t575_leaf4 + 3.9542125900000002e-04 t576_leaf0
   - 0.00192595611 t576_leaf1 - 0.00221278309 t576_leaf2
   + 3.17879e-05 t576_leaf3 + 1.16207877e-04 t577_leaf0
   - 1.39604e-05 t578_leaf0 + 0.00480083562 t579_leaf0
   + 5.0738610999999996e-04 t579_leaf1 - 0.00149210892 t579_leaf2
   + 6.70867e-05 t579_leaf3 - 4.4882675899999999e-04 t580_leaf0
   + 0.00246788654 t580_leaf1 - 0.000194067 t580_leaf2
   + 0.000730997 t580_leaf3 - 0.00264843344 t580_leaf4
   - 1.83367e-05 t581_leaf0 + 0.00499327108 t582_leaf0
   + 3.0840374500000001e-04 t582_leaf1 - 6.97928e-05 t582_leaf2
   - 3.1702528900000001e-04 t582_leaf3 + 0.00601757225 t582_leaf4
   + 2.47505e-06 t583_leaf0 + 1.31194523e-04 t584_leaf0
   - 0.00239908532 t584_leaf1 - 0.000297674 t585_leaf0
   + 3.20981169e-04 t585_leaf1 - 0.00163751282 t585_leaf2
   - 9.3179865500000005e-04 t585_leaf3 + 0.0031396281 t585_leaf4
   + 4.8679e-05 t586_leaf0 - 0.00276522106 t586_leaf1
   + 9.97335e-05 t587_leaf0 - 2.89141753e-04 t588_leaf0
   + 0.00105247018 t588_leaf1 + 0.00409152312 t588_leaf2
   + 5.8972e-05 t588_leaf3 + 0.00326518342 t589_leaf0
   - 1.9943561299999999e-04 t589_leaf1 - 1.16814575e-04 t590_leaf0
   - 1.86245e-05 t591_leaf0 - 1.9356637499999999e-04 t592_leaf0
   + 0.00340114394 t592_leaf1 - 3.5039859399999999e-04 t592_leaf2
   - 4.32081e-06 t593_leaf0 - 8.1067154000000002e-04 t594_leaf0
   + 5.23258e-05 t594_leaf1 + 2.3907382300000001e-04 t595_leaf0
   + 5.36697e-05 t596_leaf0 - 3.33392789e-04 t597_leaf0
   + 3.0121378899999998e-04 t597_leaf1 + 0.00152121624 t598_leaf0
   - 1.48647567e-04 t598_leaf1 + 0.000492223 t599_leaf0
   - 4.1698446100000002e-04 t599_leaf1 + 6.30043854e-04 t599_leaf2
   - 0.00110274565 t599_leaf3 - 0.00145727198 t599_leaf4
   - 2.4491519399999999e-04 t600_leaf0 + 8.8153255600000003e-04 t600_leaf1
   - 8.12580576e-04 t600_leaf2 + 0.0010364214 t600_leaf3
   - 5.5764e-06 t601_leaf0 + 5.4165534699999996e-04 t602_leaf0
   - 4.7459945200000001e-04 t602_leaf1 + 2.5238166599999998e-04 t602_leaf2
   + 0.00296256668 t602_leaf3 - 0.00210072706 t602_leaf4
   + 7.0232554600000005e-04 t603_leaf0 + 0.00466659619 t603_leaf1
   + 6.18768e-05 t603_leaf2 - 0.000513092 t603_leaf3
   + 2.48768e-05 t604_leaf0 + 0.00275576697 t604_leaf1
   - 0.00182765094 t604_leaf2 + 5.07431e-05 t605_leaf0
   + 0.00347663462 t605_leaf1 + 1.95207e-05 t606_leaf0
   - 3.16054e-05 t607_leaf0 - 0.00208633114 t607_leaf1
   - 0.00296968501 t607_leaf2 - 0.00157013768 t608_leaf0
   + 6.3236383699999995e-04 t608_leaf1 - 4.73038e-05 t608_leaf2
   + 0.00203685788 t608_leaf3 + 8.02211e-05 t609_leaf0
   - 0.00258374447 t609_leaf1 - 3.9604611800000001e-04 t609_leaf2
   + 0.00411174446 t609_leaf3 - 9.82789e-06 t610_leaf0
   - 0.00258815894 t611_leaf0 + 9.13324e-05 t611_leaf1
   - 1.19106e-06 t612_leaf0 + 0.00355592556 t613_leaf0
   + 1.44619364e-04 t613_leaf1 + 0.00104066939 t613_leaf2
   - 2.79624684e-04 t613_leaf3 - 0.00199886458 t613_leaf4
   + 7.18229e-05 t614_leaf0 - 0.00207156152 t614_leaf1
   + 0.000228434 t615_leaf0 - 6.1250640999999995e-04 t615_leaf1
   - 0.00254307315 t616_leaf0 - 6.50036e-05 t616_leaf1
   - 0.00109667168 t617_leaf0 + 5.09612e-05 t617_leaf1
   + 0.00567051675 t617_leaf2 + 6.91641e-05 t617_leaf3
   - 1.34794e-05 t618_leaf0 + 0.00123309961 t618_leaf1
   + 7.7292328900000005e-04 t619_leaf0 + 5.12171e-06 t619_leaf1
   - 0.00241205841 t619_leaf2 + 3.6293879400000001e-04 t620_leaf0
   - 3.05491209e-04 t620_leaf1 + 0.00267561851 t620_leaf2
   - 0.00124470436 t620_leaf3 + 1.1678579199999999e-04 t621_leaf0
   - 0.000698038 t621_leaf1 - 2.8040839200000001e-04 t622_leaf0
   + 0.000231554 t622_leaf1 + 0.0022007986 t622_leaf2
   - 0.00189598114 t622_leaf3 - 6.91608e-05 t623_leaf0
   - 1.49176e-06 t623_leaf1 + 0.00448279502 t623_leaf2
   + 2.5581932300000002e-04 t623_leaf3 + 1.18184551e-04 t624_leaf0
   - 8.1098853800000003e-04 t624_leaf1 + 0.00143458752 t625_leaf0
   - 5.7857e-05 t625_leaf1 - 3.7946770299999999e-04 t626_leaf0
   + 0.00685564335 t626_leaf1 + 3.6072850299999999e-04 t626_leaf2
   + 0.00191672728 t627_leaf0 - 0.00207283162 t627_leaf1
   + 0.0013400109 t627_leaf2 - 1.8222240000000001e-04 t627_leaf3
   - 0.00215825764 t628_leaf0 - 1.01389356e-04 t628_leaf1
   + 0.00214692578 t628_leaf2 + 7.53688e-05 t630_leaf0
   + 3.4034432600000002e-04 t631_leaf0 - 0.00059231 t631_leaf1
   - 0.00086227 t631_leaf2 - 0.0013376131 t631_leaf3
   + 0.000656223 t632_leaf0 - 6.88595e-05 t632_leaf1
   - 0.00249048788 t632_leaf2 - 2.0374920900000001e-04 t632_leaf3
   + 0.00600839825 t632_leaf4 - 3.76092e-05 t633_leaf0
   - 1.6742061400000001e-04 t634_leaf0 + 0.00286975247 t634_leaf1
   + 3.2757531199999998e-04 t634_leaf2 + 3.55751166e-04 t635_leaf0
   - 0.00197711843 t635_leaf1 - 1.48862295e-04 t635_leaf2
   + 2.33965e-05 t636_leaf0 + 1.17520285e-04 t637_leaf0
   - 0.00148427067 t637_leaf1 + 2.6054386399999998e-04 t638_leaf0
   - 0.00073761 t638_leaf1 - 1.16725292e-04 t639_leaf0
   + 4.7127e-06 t640_leaf0 + 0.00618228829 t640_leaf1
   - 2.8493429999999997e-04 t640_leaf2 + 2.12241e-05 t641_leaf0
   + 2.58227e-05 t642_leaf0 - 0.00324216 t643_leaf0
   - 6.07342e-05 t643_leaf1 - 7.45173e-05 t644_leaf0
   - 1.08258551e-04 t645_leaf0 + 0.000239384 t646_leaf0
   - 0.000810464 t646_leaf1 + 2.9873111600000002e-04 t646_leaf2
   - 0.00112546072 t646_leaf3 - 0.00209141872 t647_leaf0
   + 0.000786845 t647_leaf1 + 0.00397948269 t647_leaf2
   + 1.85122e-06 t647_leaf3 - 0.0016922364 t647_leaf4
   + 1.55782749e-04 t647_leaf5 + 3.32832e-05 t648_leaf0
   - 1.89769e-05 t649_leaf0 + 0.00187498238 t649_leaf1
   - 0.00199656654 t650_leaf0 + 0.0012405063 t650_leaf1
   - 0.000228194 t650_leaf2 - 2.37596e-05 t651_leaf0
   + 2.96133e-07 t652_leaf0 + 1.88401e-05 t653_leaf0
   - 0.00314511731 t653_leaf1 + 1.05232e-05 t654_leaf0
   - 1.3843635700000001e-04 t655_leaf0 - 4.3569176299999997e-04 t655_leaf1
   + 0.00322243874 t655_leaf2 - 4.16574e-05 t656_leaf0
   + 0.00236192136 t656_leaf1 + 1.1096865600000001e-04 t657_leaf0
   - 0.0014238985 t657_leaf1 + 3.90781e-05 t658_leaf0
   + 0.00107935967 t659_leaf0 - 0.00114166876 t659_leaf1
   - 3.6961e-05 t659_leaf2 - 0.0019321919 t659_leaf3
   + 0.00213252311 t659_leaf4 - 4.61841e-06 t660_leaf0
   + 9.1342872500000001e-04 t660_leaf1 - 7.0124928599999998e-04 t660_leaf2
   - 0.00309028057 t660_leaf3 + 0.00173046964 t661_leaf0
   - 0.00117126293 t661_leaf1 - 5.23507e-05 t661_leaf2
   - 4.32005e-06 t662_leaf0 + 0.00100840733 t663_leaf0
   - 0.00166104874 t663_leaf1 + 2.76899e-05 t663_leaf2
   + 0.00329571404 t663_leaf3 + 8.45613e-05 t663_leaf4
   - 1.31438603e-04 t664_leaf0 - 4.49424e-05 t665_leaf0
   + 0.00125774904 t665_leaf1 + 0.00593773602 t665_leaf2
   - 0.00132237142 t665_leaf3 + 0.00200840365 t665_leaf4
   + 9.88056e-05 t666_leaf0 - 1.16225e-05 t667_leaf0
   - 2.09748556e-04 t668_leaf0 + 0.00147237105 t668_leaf1
   - 0.00199232087 t668_leaf2 + 1.3241e-05 t669_leaf0
   - 1.07493135e-04 t670_leaf0 + 7.8057532700000003e-04 t670_leaf1
   - 0.00331130391 t670_leaf2 + 0.00414482411 t670_leaf3
   - 2.7211679799999999e-04 t671_leaf0 + 3.8296234600000001e-04 t671_leaf1
   - 0.00172387273 t671_leaf2 + 6.55285e-05 t672_leaf0
   - 0.00197655638 t673_leaf0 + 0.00293301232 t673_leaf1
   + 0.0046979608 t673_leaf2 + 6.3549843599999996e-04 t673_leaf3
   - 6.35495e-05 t673_leaf4 - 1.3460926100000001e-04 t674_leaf0
   - 7.65269e-05 t675_leaf0 - 3.19515e-06 t676_leaf0
   + 4.24309e-05 t677_leaf0 + 5.13112e-05 t678_leaf0
   - 0.00256165862 t678_leaf1 + 1.77162e-05 t679_leaf0
   + 1.06689e-05 t680_leaf0 - 0.00222979253 t680_leaf1
   - 0.00438896 t680_leaf2 + 3.1391638899999998e-04 t680_leaf3
   + 8.71609e-05 t681_leaf0 + 0.00331523269 t681_leaf1
   + 6.59192e-05 t681_leaf2 + 0.00352146756 t681_leaf3
   - 0.000727462 t682_leaf0 + 0.0023326003 t682_leaf1
   - 6.68041e-05 t682_leaf2 + 1.2918119300000001e-04 t683_leaf0
   - 6.6607928600000003e-04 t683_leaf1 + 1.63953e-05 t684_leaf0
   + 5.26407e-05 t685_leaf0 - 2.37277e-05 t686_leaf0
   + 0.00325714215 t687_leaf0 - 9.3786430100000005e-04 t687_leaf1
   + 1.80925839e-04 t687_leaf2 - 0.00318124727 t687_leaf3
   + 9.6137617900000005e-04 t687_leaf4 - 3.4072369500000001e-04 t687_leaf5
   + 4.6610613900000003e-04 t688_leaf0 - 3.2380287300000002e-04 t688_leaf1
   + 7.2162511100000003e-04 t688_leaf2 - 0.00157196703 t688_leaf3
   + 0.0012902692 t689_leaf0 + 2.07928e-05 t689_leaf1
   + 7.1533e-05 t690_leaf0 - 3.03651e-05 t691_leaf0
   + 1.20355166e-04 t692_leaf0 + 0.00347242574 t692_leaf1
   - 6.9009e-05 t693_leaf0 - 0.00164817332 t693_leaf1
   - 4.82791e-05 t694_leaf0 + 5.01858e-05 t695_leaf0
   - 5.30503e-05 t696_leaf0 - 0.00160393969 t696_leaf1
   + 1.3520185900000001e-04 t697_leaf0 - 4.93679e-05 t698_leaf0
   + 0.00384521205 t698_leaf1 + 0.00138785841 t699_leaf0
   - 6.0224975500000003e-04 t699_leaf1 + 0.00339953206 t699_leaf2
   - 0.00218873657 t699_leaf3 + 5.67711e-06 t699_leaf4
   - 9.12805e-07 t700_leaf0 - 0.000101083 t700_leaf1
   + 0.00420768 t700_leaf2 - 0.0030518109 t701_leaf0
   + 9.29367e-05 t701_leaf1 + 0.00200143596 t702_leaf0
   + 0.000184476 t702_leaf1 + 8.37312e-06 t702_leaf2
   - 0.00282813725 t702_leaf3 - 0.00563832792 t702_leaf4
   - 3.18668579e-04 t703_leaf0 - 0.00317626307 t703_leaf1
   - 5.28868e-06 t703_leaf2 - 0.00166602887 t703_leaf3
   + 0.00114521571 t703_leaf4 - 3.93433e-05 t704_leaf0
   + 2.0071375199999999e-04 t705_leaf0 - 0.00260111177 t705_leaf1
   - 3.2445447899999999e-04 t705_leaf2 - 2.27927e-05 t706_leaf0
   - 0.00157483725 t706_leaf1 - 4.05859231e-04 t706_leaf2
   + 0.00197361 t706_leaf3 + 0.00154308113 t706_leaf4
   - 1.73737e-05 t707_leaf0 + 6.82426617e-04 t708_leaf0
   - 6.2814843800000002e-04 t708_leaf1 + 1.8393747500000001e-04 t708_leaf2
   + 0.00391700212 t708_leaf3 - 0.00236884644 t708_leaf4
   + 3.28598e-05 t709_leaf0 + 1.34034752e-04 t710_leaf0
   - 0.0022651623 t710_leaf1 - 3.8683923800000001e-04 t710_leaf2
   - 5.23341e-05 t711_leaf0 + 0.00160780037 t712_leaf0
   - 8.20677e-05 t712_leaf2 + 0.0037211806 t712_leaf3
   - 0.00163079263 t712_leaf4 + 8.1712735099999996e-04 t713_leaf0
   - 0.00294639869 t713_leaf1 + 3.0491553499999998e-04 t713_leaf2
   - 0.00181499461 t713_leaf3 - 0.00142552832 t713_leaf4
   - 9.6291e-06 t714_leaf0 - 0.00254862872 t714_leaf1
   - 5.84226e-05 t715_leaf0 + 0.00678135874 t715_leaf1
   - 4.6281615499999999e-04 t715_leaf2 - 7.71167572e-04 t715_leaf3
   + 8.59751832e-04 t715_leaf4 + 3.95054e-06 t716_leaf0
   + 1.8495054999999999e-04 t717_leaf0 + 4.96961e-05 t718_leaf0
   + 0.00282919756 t718_leaf1 + 0.00376139325 t718_leaf2
   + 2.84277e-05 t718_leaf3 - 3.4228281600000001e-04 t718_leaf4
   - 6.40364e-05 t719_leaf0 + 7.59166e-05 t720_leaf0
   - 2.11322666e-04 t721_leaf0 + 0.000363818 t721_leaf1
   - 0.00203367299 t721_leaf2 - 0.00156160293 t721_leaf3
   + 0.00126265851 t721_leaf4 - 4.81132e-05 t722_leaf0
   + 4.3332381799999998e-04 t723_leaf0 - 5.0009938400000003e-04 t723_leaf1
   - 5.51253e-05 t723_leaf2 - 0.00141284463 t723_leaf3
   + 8.7289587800000001e-04 t723_leaf4 - 0.00010224 t724_leaf0
   - 0.00185283623 t724_leaf1 + 0.000653619 t724_leaf2
   - 2.1474e-05 t725_leaf0 - 4.86898e-05 t726_leaf0
   + 0.00286344206 t726_leaf1 + 1.3003368800000001e-04 t727_leaf0
   - 0.00547064468 t727_leaf1 - 7.02483e-05 t728_leaf0
   - 1.6735613499999999e-04 t729_leaf0 + 1.16757728e-04 t730_leaf0
   - 4.90015e-05 t731_leaf0 - 1.21826597e-04 t732_leaf0
   + 2.35447e-05 t733_leaf0 + 8.39456e-05 t734_leaf0
   - 7.77149e-06 t735_leaf0 - 0.00171279814 t735_leaf1
   + 0.000346777 t735_leaf2 + 0.00259568845 t735_leaf3
   + 0.000257657 t736_leaf0 - 5.47681e-05 t737_leaf0
   - 0.00218474586 t738_leaf0 - 1.75844674e-04 t738_leaf1
   + 0.00111373467 t738_leaf2 + 0.00142412155 t738_leaf3
   - 6.98274e-05 t738_leaf4 - 6.27052e-05 t739_leaf0
   + 2.47805e-05 t740_leaf0 - 1.5115237299999999e-04 t741_leaf0
   + 6.8131141599999995e-04 t741_leaf1 + 0.00304961 t741_leaf2
   - 0.00182239106 t741_leaf3 + 0.000898821 t741_leaf4
   - 9.81361e-06 t742_leaf0 + 0.00446525635 t742_leaf1
   + 3.6954827400000002e-04 t742_leaf2 - 2.90928e-05 t743_leaf0
   - 3.61899e-05 t744_leaf0 - 2.79637e-05 t745_leaf0
   - 4.3299e-06 t746_leaf0 - 3.8382e-05 t747_leaf0 - 3.18818e-05 t748_leaf0
   + 0.000310677 t749_leaf0 + 0.00345476693 t749_leaf1
   - 1.6661541300000001e-04 t749_leaf2 + 6.54831e-05 t750_leaf0
   + 5.11851e-05 t751_leaf0 + 0.00374242174 t751_leaf1
   - 0.000206194 t751_leaf2 - 0.00159987912 t751_leaf3
   - 2.7457255100000002e-04 t751_leaf4 + 0.00278158044 t751_leaf5
   - 0.000162473 t752_leaf0 - 1.02084e-05 t753_leaf0
   + 1.80172e-05 t754_leaf0 + 7.64612e-06 t755_leaf0
   + 8.0213049700000004e-04 t756_leaf0 - 2.90423e-05 t756_leaf1
   - 6.8344583300000001e-04 t756_leaf2 - 3.5869365099999999e-04 t756_leaf3
   - 4.6468389299999998e-04 t757_leaf0 - 0.00298959576 t757_leaf1
   + 6.40539e-05 t757_leaf2 - 1.86083664e-04 t758_leaf0
   + 0.00501360931 t758_leaf1 + 0.00192486704 t758_leaf2
   + 9.67289e-05 t758_leaf3 - 6.68673718e-04 t758_leaf4
   + 1.9426747200000001e-04 t758_leaf5 + 1.2171358000000001e-04 t759_leaf0
   + 1.08694425e-04 t760_leaf0 - 4.66333e-05 t761_leaf0
   + 0.00550235808 t761_leaf2 + 3.81317e-05 t762_leaf0
   + 3.58505e-05 t763_leaf0 - 7.90440245e-04 t764_leaf0
   + 1.87937e-06 t764_leaf1 + 1.90579594e-04 t764_leaf2
   - 9.46197e-05 t765_leaf0 + 0.00394603424 t766_leaf0
   + 0.000885566 t766_leaf1 - 2.37633489e-04 t766_leaf2
   + 4.5056687699999997e-04 t766_leaf3 - 3.13575845e-04 t766_leaf4
   - 2.04208e-05 t767_leaf0 + 3.14728e-05 t768_leaf0
   - 0.00231542927 t768_leaf1 + 0.00229821238 t768_leaf2
   - 4.07087e-05 t769_leaf0 - 1.4455066399999999e-04 t770_leaf0
   - 3.04356e-05 t771_leaf0 - 2.03608e-05 t772_leaf0
   + 1.3746159699999999e-04 t773_leaf0 + 0.00409553014 t774_leaf0
   - 6.88269e-05 t774_leaf1 + 0.000122944 t775_leaf0
   + 0.00182314042 t775_leaf1 + 2.60373e-05 t776_leaf0
   - 2.29033e-05 t777_leaf0 + 0.0031635596 t777_leaf1
   - 1.53887e-05 t778_leaf0 + 0.00309331948 t778_leaf1
   - 0.00482132472 t778_leaf3 - 7.89722e-05 t779_leaf0
   - 2.48839e-05 t780_leaf0 + 0.00394469965 t781_leaf0
   + 2.1881029499999999e-04 t781_leaf1 - 4.7740558500000002e-04 t781_leaf2
   - 0.00345318462 t781_leaf3 - 3.74338e-05 t782_leaf0
   - 1.53212e-05 t783_leaf0 - 0.00144355558 t784_leaf0
   + 8.8853843000000004e-04 t784_leaf1 + 1.5132926600000001e-04 t784_leaf2
   - 0.00294112461 t784_leaf3 - 0.00310091884 t786_leaf0
   + 4.5502e-06 t786_leaf1 - 0.00205944432 t786_leaf2
   + 4.38985e-05 t787_leaf0 + 8.59327e-06 t788_leaf0
   - 0.00298585813 t788_leaf1 + 0.000150206 t789_leaf0
   - 3.14592e-05 t790_leaf0 + 0.00259288657 t790_leaf1
   + 1.33058827e-04 t791_leaf0 - 3.01922e-05 t792_leaf0
   + 2.94796e-05 t793_leaf0 + 5.9345253999999997e-04 t794_leaf0
   - 2.07620425e-04 t794_leaf1 + 0.0021461721 t795_leaf0
   - 9.06683e-05 t795_leaf1 - 2.5896017900000002e-04 t795_leaf2
   - 0.00224913121 t795_leaf3 + 8.6194917099999995e-04 t795_leaf4
   + 5.48694e-05 t795_leaf5 - 6.0417846500000001e-04 t796_leaf0
   + 0.0016287727 t796_leaf1 + 2.74165e-05 t796_leaf2
   + 0.00253701047 t796_leaf3 + 2.1029838499999999e-04 t796_leaf4
   + 7.79652e-05 t797_leaf0 + 0.00329153775 t798_leaf0
   - 5.76097e-05 t798_leaf1 + 6.20139e-05 t799_leaf0
   - 0.00123833586 t799_leaf1 + 0.00582942227 t799_leaf2
   + 4.3694383899999998e-04 t799_leaf3 - 1.5264634599999999e-04 t800_leaf0
   - 3.6985e-05 t801_leaf0 + 2.09693e-05 t802_leaf0
   - 5.72886e-05 t803_leaf0 + 7.83293e-05 t804_leaf0
   - 2.3707e-05 t805_leaf0 + 5.6567828900000005e-04 t806_leaf0
   - 1.9432646499999999e-04 t806_leaf1 + 0.00576540502 t806_leaf2
   - 5.1846e-05 t806_leaf3 + 7.7236e-05 t807_leaf0
   + 0.00380125619 t807_leaf1 - 0.0028027622 t808_leaf0
   + 0.00106997974 t808_leaf1 - 0.00237719179 t808_leaf2
   + 6.5095e-05 t808_leaf3 - 0.00276915 t809_leaf0 + 3.59455e-05 t809_leaf1
   - 7.20537e-05 t810_leaf0 - 2.02692e-05 t811_leaf0
   + 6.03158e-05 t812_leaf0 + 1.21758501e-04 t813_leaf0
   + 0.0047334847 t814_leaf0 - 2.26131364e-04 t814_leaf1
   + 6.7561434099999999e-04 t814_leaf2 + 0.00371122523 t814_leaf3
   - 2.0046620900000001e-04 t814_leaf4 + 1.4902e-05 t815_leaf0
   - 0.000997999 t815_leaf1 - 6.41048e-05 t816_leaf0
   - 3.50323e-05 t817_leaf0 - 2.72928e-05 t818_leaf0
   + 0.00118934154 t818_leaf1 + 4.92488e-05 t819_leaf0
   - 0.00129385677 t819_leaf1 + 0.00322944648 t819_leaf2
   + 0.00268852757 t819_leaf3 - 8.75346e-05 t820_leaf0
   - 1.79859e-05 t821_leaf0 + 0.00400362769 t822_leaf0
   - 0.00127576047 t822_leaf1 + 0.00134594622 t822_leaf2
   - 4.6724e-05 t822_leaf3 - 1.80833e-05 t823_leaf0
   - 1.4255914700000001e-04 t824_leaf0 - 0.00255027064 t824_leaf1
   + 9.0154825000000004e-04 t824_leaf2 - 7.70151e-05 t825_leaf0
   - 7.7476870500000001e-04 t826_leaf0 + 2.7226860400000002e-04 t826_leaf1
   - 0.0010397135 t826_leaf2 - 2.2461269699999999e-04 t827_leaf0
   - 0.0017864597 t827_leaf1 + 1.4601227299999999e-04 t827_leaf2
   - 9.61136e-05 t828_leaf0 - 0.0016138741 t828_leaf1
   - 0.00300114602 t828_leaf2 + 4.4218244099999998e-04 t828_leaf3
   + 0.00165108766 t829_leaf0 - 9.2976557800000001e-04 t829_leaf1
   - 0.00185920671 t829_leaf2 + 3.1272051300000001e-04 t829_leaf3
   - 5.0994771300000004e-04 t829_leaf4 + 0.00342979608 t829_leaf5
   - 3.0580977900000002e-04 t829_leaf6 + 0.00107628363 t829_leaf7
   + 4.17172e-05 t830_leaf0 - 6.24223e-08 t831_leaf0
   - 0.0038987794 t832_leaf0 - 3.3773e-05 t832_leaf1
   + 0.00344721135 t833_leaf0 - 3.576e-06 t833_leaf1
   - 0.0010301593 t834_leaf0 - 0.000181801 t834_leaf1
   + 0.00210401858 t834_leaf2 + 0.00131029671 t834_leaf3
   - 4.4480589000000002e-04 t835_leaf0 - 0.00266521238 t835_leaf1
   - 9.82975e-05 t835_leaf2 + 1.3030678399999999e-04 t836_leaf0
   - 0.000769194 t836_leaf1 + 5.86237e-06 t837_leaf0
   + 2.51247e-05 t838_leaf0 - 0.000130527 t839_leaf0
   + 0.00347493659 t839_leaf1 + 2.92206e-05 t840_leaf0
   - 1.06984138e-04 t841_leaf0 + 0.00122846919 t841_leaf1
   + 2.57865e-05 t842_leaf0 + 5.9929e-06 t843_leaf0
   + 4.79471e-05 t844_leaf0 - 0.00200248463 t844_leaf1
   - 5.01761e-05 t845_leaf0 - 7.38079e-07 t846_leaf0
   - 8.15937e-05 t847_leaf0 + 0.00135207339 t847_leaf1
   + 0.00181803247 t848_leaf0 - 4.7715843499999998e-04 t848_leaf1
   - 3.67114e-05 t848_leaf2 - 7.95345e-05 t849_leaf0
   - 7.7853e-05 t850_leaf0 - 1.38481e-05 t851_leaf0
   + 8.86039e-07 t853_leaf0 + 8.15873e-05 t854_leaf0
   - 4.31744e-05 t855_leaf0 - 0.00225561112 t855_leaf1
   - 4.00933e-05 t856_leaf0 + 6.89151e-05 t857_leaf0
   + 5.40282e-05 t858_leaf0 - 0.00267030555 t858_leaf1
   - 6.01056e-05 t859_leaf0 + 0.00136272283 t860_leaf0
   + 8.06908e-05 t860_leaf1 + 3.89235e-05 t861_leaf0
   - 0.00198826287 t862_leaf0 + 2.76129e-06 t862_leaf1
   - 0.00122085016 t862_leaf2 - 0.000513635 t863_leaf0
   + 0.0054920502 t863_leaf2 + 0.000223938 t863_leaf3
   + 6.38129e-05 t864_leaf0 - 0.00160765613 t865_leaf0
   + 7.9629523699999995e-04 t865_leaf1 + 0.00455872295 t865_leaf2
   - 1.3576126400000001e-04 t865_leaf3 + 2.02835246e-04 t866_leaf0
   + 0.00379992393 t866_leaf1 + 0.00258890167 t866_leaf2
   - 1.03952305e-04 t866_leaf3 + 7.59277e-05 t867_leaf0
   + 0.00265368237 t867_leaf1 - 0.00100777973 t867_leaf2
   + 3.83385e-05 t868_leaf0 - 4.71334e-05 t869_leaf0
   - 7.27103e-06 t870_leaf0 - 4.38161e-05 t871_leaf0
   + 3.39574e-05 t872_leaf0 - 0.0022973076 t873_leaf0
   + 1.67738865e-04 t873_leaf1 + 4.61047e-05 t874_leaf0
   + 9.10858e-05 t875_leaf0 - 0.00104393577 t876_leaf0
   + 1.04337e-05 t876_leaf1 - 0.00190702814 t876_leaf2
   + 3.48649861e-04 t876_leaf3 - 8.60385e-06 t877_leaf0
   + 7.7858712799999999e-04 t877_leaf1 - 0.0022482397 t877_leaf2
   - 8.39336e-05 t878_leaf0 - 0.00097534 t879_leaf0
   + 1.3216218199999999e-04 t879_leaf1 + 3.8955966000000001e-04 t880_leaf0
   - 3.0749876200000001e-04 t880_leaf1 - 6.3745869599999996e-04 t880_leaf2
   - 9.9778769100000004e-04 t880_leaf3 + 0.00344065181 t880_leaf4
   - 7.77916168e-04 t880_leaf5 + 0.00176870776 t881_leaf0
   - 1.9938575900000001e-04 t881_leaf1 - 7.95822e-05 t881_leaf2
   + 3.4702179300000001e-04 t881_leaf3 - 0.00439990684 t881_leaf4
   - 1.28748e-06 t882_leaf0 - 6.80383e-05 t883_leaf0
   + 0.000189462 t884_leaf0 - 8.6836650700000001e-04 t884_leaf1
   - 0.00214079628 t884_leaf2 + 0.000899012 t884_leaf3
   + 0.0028458687 t884_leaf4 + 3.37158e-05 t885_leaf0
   - 1.36790142e-04 t886_leaf0 + 0.00109359238 t886_leaf1
   - 0.00323858811 t886_leaf2 + 0.00368780876 t886_leaf3
   - 1.27755e-07 t886_leaf4 + 2.9817543700000001e-04 t886_leaf5
   - 0.000115995 t887_leaf0 + 0.00246949028 t887_leaf1
   - 9.28680645e-04 t888_leaf0 + 2.7653056899999999e-04 t888_leaf1
   - 0.00235303398 t888_leaf2 - 0.00364472088 t888_leaf3
   - 4.3452510700000001e-04 t888_leaf4 + 0.00299136527 t888_leaf5
   + 4.2735479699999997e-04 t889_leaf0 + 0.00461305492 t889_leaf1
   + 0.000689162 t889_leaf2 - 4.00476769e-04 t889_leaf3
   + 2.5524251399999999e-04 t889_leaf4 + 0.00359264924 t889_leaf5
   - 6.61603e-05 t890_leaf0 - 1.93659725e-04 t890_leaf1
   + 0.00214896654 t890_leaf2 + 6.63893e-06 t891_leaf0
   + 0.00299583212 t891_leaf1 + 1.8634e-05 t892_leaf0
   - 5.19944e-06 t893_leaf0 + 1.7578226099999999e-04 t894_leaf0
   - 0.0014008861 t894_leaf1 + 1.0988560800000001e-04 t895_leaf0
   + 8.4885361100000005e-04 t896_leaf0 - 1.07119471e-04 t896_leaf1
   - 2.26401e-05 t897_leaf0 - 6.56573e-05 t898_leaf0
   + 2.29431e-05 t899_leaf0 + 5.60668e-05 t900_leaf0
   + 4.07426e-05 t901_leaf0 - 7.6292e-08 t902_leaf0
   + 0.00313407392 t903_leaf0 - 6.54916e-05 t903_leaf1
   + 0.00299207447 t904_leaf0 - 1.4826527300000001e-04 t904_leaf1
   + 7.5813126800000001e-04 t904_leaf2 - 0.00121585093 t905_leaf0
   + 0.00274668075 t905_leaf1 + 8.95417e-06 t905_leaf2
   - 7.7310868099999999e-04 t906_leaf0 + 5.44148e-05 t906_leaf1
   - 7.8247865900000005e-04 t906_leaf2 + 9.6805510100000003e-04 t906_leaf3
   + 8.8648544599999996e-04 t906_leaf4 + 0.00604328606 t907_leaf0
   - 4.1564964300000001e-04 t907_leaf1 - 4.50115e-06 t907_leaf2
   - 1.5489454400000001e-04 t908_leaf0 - 5.0320022299999996e-04 t909_leaf0
   + 0.00203016354 t909_leaf1 + 0.00478181802 t909_leaf2
   + 2.09453e-06 t909_leaf3 + 1.78e-05 t910_leaf0
   + 6.1468349299999996e-04 t911_leaf0 - 4.0309523999999998e-04 t911_leaf1
   + 3.9739988299999999e-04 t911_leaf2 - 5.86481e-05 t912_leaf0
   + 1.05044703e-04 t913_leaf0 + 5.25074e-06 t914_leaf0
   + 1.3152173800000001e-04 t915_leaf0 - 0.00136213482 t915_leaf1
   - 0.00105628825 t915_leaf2 + 0.000490767 t916_leaf0
   + 0.00436181249 t916_leaf1 - 9.83813e-05 t916_leaf2
   + 1.42104e-05 t917_leaf0 + 5.13558e-05 t918_leaf0
   + 5.01269e-05 t919_leaf0 - 6.82613e-05 t920_leaf0
   - 4.40985e-05 t921_leaf0 - 2.41777e-05 t922_leaf0
   + 0.00316163129 t922_leaf1 + 9.20905e-06 t922_leaf2
   + 2.7434551200000002e-04 t923_leaf0 + 0.00597921433 t923_leaf1
   - 0.00048426 t923_leaf3 + 1.3381418800000001e-04 t923_leaf4
   - 0.00235924567 t923_leaf5 + 5.1548547300000005e-04 t924_leaf0
   + 0.00545417564 t924_leaf2 - 1.75155597e-04 t924_leaf3
   - 3.55719705e-04 t925_leaf0 + 2.1765432000000001e-04 t925_leaf1
   + 0.00145866384 t925_leaf2 - 4.73022e-05 t926_leaf0
   + 2.92503e-05 t927_leaf0 - 4.12997e-05 t928_leaf0
   - 3.41053e-05 t929_leaf0 + 2.52299e-05 t930_leaf0
   - 4.36073e-05 t931_leaf0 - 0.00141258352 t931_leaf1
   + 0.00215743668 t931_leaf2 - 6.87765e-05 t932_leaf0
   + 0.00315364916 t932_leaf1 + 9.98988e-05 t933_leaf0
   + 1.39463853e-04 t934_leaf0 - 0.00170511287 t935_leaf0
   + 7.6620379699999998e-04 t935_leaf1 - 8.80469e-05 t935_leaf2
   - 0.00366048818 t935_leaf3 + 4.36915e-05 t936_leaf0
   - 0.00202977634 t937_leaf0 + 1.4385410800000001e-04 t937_leaf1
   - 0.00293873553 t937_leaf2 + 1.30516157e-04 t938_leaf0
   + 0.00313225877 t939_leaf1 - 0.00105057447 t940_leaf0
   + 0.00122787256 t940_leaf1 + 1.08562897e-04 t940_leaf2
   + 1.9834528200000001e-04 t940_leaf3 + 6.95331e-05 t941_leaf0
   - 0.00126649276 t941_leaf1 + 4.28409e-05 t942_leaf0
   - 4.98851e-05 t943_leaf0 - 6.85729e-05 t944_leaf0
   + 9.64173e-05 t945_leaf0 - 6.44528e-05 t946_leaf0
   - 6.8265741000000002e-04 t947_leaf0 - 0.00347040733 t947_leaf1
   + 0.00259615248 t947_leaf2 - 8.7622669499999996e-04 t947_leaf3
   + 1.6277082599999999e-04 t947_leaf4 + 1.40076e-05 t948_leaf0
   - 3.83167e-05 t949_leaf0 - 1.17372e-05 t950_leaf0
   - 7.53246e-06 t951_leaf0 + 1.26238e-05 t951_leaf1
   + 0.00464316225 t951_leaf2 - 1.28036e-05 t952_leaf0
   - 5.3795e-05 t953_leaf0 - 2.9442e-05 t954_leaf0
   - 0.00349366083 t954_leaf1 - 6.2110315800000004e-04 t954_leaf2
   + 9.1989355900000001e-04 t954_leaf3 - 0.00234536035 t954_leaf4
   - 7.25053e-06 t955_leaf0 - 1.3063437700000001e-04 t956_leaf0
   + 6.0449825800000002e-04 t956_leaf1 + 0.00185802672 t957_leaf0
   - 4.4662627600000002e-04 t957_leaf1 + 0.00515933335 t957_leaf2
   - 1.5514979899999999e-04 t957_leaf3 + 0.00063023 t957_leaf4 = 0
 YLOG_CAL: - y_log + y_log_cal = 0.5000059085400519
 budget: cost_model <= 200000
 neigh_floor__1st_Flr_SF: x_1st_Flr_SF >= 788
 neigh_floor__2nd_Flr_SF: x_2nd_Flr_SF >= 0
 neigh_floor__Total_Bsmt_SF: x_Total_Bsmt_SF >= 728
 neigh_floor__Garage_Area: x_Garage_Area >= 484
 neigh_floor__Bedroom_AbvGr: x_Bedroom_AbvGr >= 1
 neigh_floor__Full_Bath: x_Full_Bath >= 1
 neigh_floor__Half_Bath: x_Half_Bath >= 0
 neigh_floor__Kitchen_AbvGr: x_Kitchen_AbvGr >= 1
 neigh_floor__Fireplaces: x_Fireplaces >= 1
 neigh_floor__Mas_Vnr_Area: x_Mas_Vnr_Area >= 0
 neigh_floor__Gr_Liv_Area: x_Gr_Liv_Area >= 788
 neigh_mode__Heating: Heating__GasA = 1
 neigh_mode__Electrical: Electrical__SBrkr = 1
 neigh_mode__Foundation: Foundation__CBlock = 1
 7.11.1__pool_space: x_Pool_Area - 7000 HasPool + [ x_1st_Flr_SF * HasPool
   + x_Garage_Area * HasPool + x_Wood_Deck_SF * HasPool
   + x_Open_Porch_SF * HasPool + x_Enclosed_Porch * HasPool
   + x_3Ssn_Porch * HasPool + x_Screen_Porch * HasPool ] <= 0
 7.16.1__mas_zero_if_none: x_Mas_Vnr_Area - 0.4 AreaExterior1st + [
   0.4 MasVnrType__None * AreaExterior1st ] <= 0
 7.16__mv_leq_u_type__BrkCmn: MvProd__BrkCmn + [
   - 0.4 MasVnrType__BrkCmn * AreaExterior1st ] <= 0
 7.16__mv_ge_area_u__BrkCmn: - x_Mas_Vnr_Area + 0.4 AreaExterior1st
   + MvProd__BrkCmn + [ - 0.4 MasVnrType__BrkCmn * AreaExterior1st ] >= 0
 7.16__mv_leq_u_type__BrkFace: MvProd__BrkFace + [
   - 0.4 MasVnrType__BrkFace * AreaExterior1st ] <= 0
 7.16__mv_ge_area_u__BrkFace: - x_Mas_Vnr_Area + 0.4 AreaExterior1st
   + MvProd__BrkFace + [ - 0.4 MasVnrType__BrkFace * AreaExterior1st ]
   >= 0
 7.16__mv_leq_u_type__CBlock: MvProd__CBlock + [
   - 0.4 MasVnrType__CBlock * AreaExterior1st ] <= 0
 7.16__mv_ge_area_u__CBlock: - x_Mas_Vnr_Area + 0.4 AreaExterior1st
   + MvProd__CBlock + [ - 0.4 MasVnrType__CBlock * AreaExterior1st ] >= 0
 7.16__mv_leq_u_type__None: MvProd__None + [
   - 0.4 MasVnrType__None * AreaExterior1st ] <= 0
 7.16__mv_ge_area_u__None: - x_Mas_Vnr_Area + 0.4 AreaExterior1st
   + MvProd__None + [ - 0.4 MasVnrType__None * AreaExterior1st ] >= 0
 7.16__mv_leq_u_type__Stone: MvProd__Stone + [
   - 0.4 MasVnrType__Stone * AreaExterior1st ] <= 0
 7.16__mv_ge_area_u__Stone: - x_Mas_Vnr_Area + 0.4 AreaExterior1st
   + MvProd__Stone + [ - 0.4 MasVnrType__Stone * AreaExterior1st ] >= 0
 11.xx__fireplace_qu_eq_na_if_none: HasFireplace = 0 -> x_Fireplace_Qu
   = -1
 link__fq_ord: HasFireplace = 1 -> x_Fireplace_Qu - 2 zFQ_TA - 3 zFQ_Gd
   - 4 zFQ_Ex = 0
 7.5.3__no_unf_if_has_gar: GarageType__NA = 0 -> GarageFinish__Unf = 0
 link__gq_ord_na: GarageType__NA = 1 -> x_Garage_Qual = -1
 link__gq_ord: GarageType__NA = 0 -> x_Garage_Qual - 2 zGQ_TA - 3 zGQ_Gd
   - 4 zGQ_Ex = 0
 11.xx__miscval_zero_if_na: MiscFeature__No_aplica = 1 -> x_Misc_Val = 0
 7.7.5__z_eq_plan_if_y__Flat_ClyTile: Y__Flat__ClyTile = 1 ->
   - PlanRoofArea + Z__Flat__ClyTile = 0
 7.7.5__z_eq_0_if_noty__Flat_ClyTile: Y__Flat__ClyTile = 0 ->
   Z__Flat__ClyTile = 0
 7.7.5__z_eq_plan_if_y__Flat_CompShg: Y__Flat__CompShg = 1 ->
   - PlanRoofArea + Z__Flat__CompShg = 0
 7.7.5__z_eq_0_if_noty__Flat_CompShg: Y__Flat__CompShg = 0 ->
   Z__Flat__CompShg = 0
 7.7.5__z_eq_plan_if_y__Flat_Membran: Y__Flat__Membran = 1 ->
   - PlanRoofArea + Z__Flat__Membran = 0
 7.7.5__z_eq_0_if_noty__Flat_Membran: Y__Flat__Membran = 0 ->
   Z__Flat__Membran = 0
 7.7.5__z_eq_plan_if_y__Flat_Metal: Y__Flat__Metal = 1 -> - PlanRoofArea
   + Z__Flat__Metal = 0
 7.7.5__z_eq_0_if_noty__Flat_Metal: Y__Flat__Metal = 0 -> Z__Flat__Metal
   = 0
 7.7.5__z_eq_plan_if_y__Flat_Roll: Y__Flat__Roll = 1 -> - PlanRoofArea
   + Z__Flat__Roll = 0
 7.7.5__z_eq_0_if_noty__Flat_Roll: Y__Flat__Roll = 0 -> Z__Flat__Roll = 0
 7.7.5__z_eq_plan_if_y__Flat_Tar&Grv: Y__Flat__Tar&Grv = 1 ->
   - PlanRoofArea + Z__Flat__Tar&Grv = 0
 7.7.5__z_eq_0_if_noty__Flat_Tar&Grv: Y__Flat__Tar&Grv = 0 ->
   Z__Flat__Tar&Grv = 0
 7.7.5__z_eq_plan_if_y__Flat_WdShake: Y__Flat__WdShake = 1 ->
   - PlanRoofArea + Z__Flat__WdShake = 0
 7.7.5__z_eq_0_if_noty__Flat_WdShake: Y__Flat__WdShake = 0 ->
   Z__Flat__WdShake = 0
 7.7.5__z_eq_plan_if_y__Flat_WdShngl: Y__Flat__WdShngl = 1 ->
   - PlanRoofArea + Z__Flat__WdShngl = 0
 7.7.5__z_eq_0_if_noty__Flat_WdShngl: Y__Flat__WdShngl = 0 ->
   Z__Flat__WdShngl = 0
 7.7.5__z_eq_plan_if_y__Gable_ClyTile: Y__Gable__ClyTile = 1 ->
   - PlanRoofArea + Z__Gable__ClyTile = 0
 7.7.5__z_eq_0_if_noty__Gable_ClyTile: Y__Gable__ClyTile = 0 ->
   Z__Gable__ClyTile = 0
 7.7.5__z_eq_plan_if_y__Gable_CompShg: Y__Gable__CompShg = 1 ->
   - PlanRoofArea + Z__Gable__CompShg = 0
 7.7.5__z_eq_0_if_noty__Gable_CompShg: Y__Gable__CompShg = 0 ->
   Z__Gable__CompShg = 0
 7.7.5__z_eq_plan_if_y__Gable_Membran: Y__Gable__Membran = 1 ->
   - PlanRoofArea + Z__Gable__Membran = 0
 7.7.5__z_eq_0_if_noty__Gable_Membran: Y__Gable__Membran = 0 ->
   Z__Gable__Membran = 0
 7.7.5__z_eq_plan_if_y__Gable_Metal: Y__Gable__Metal = 1 -> - PlanRoofArea
   + Z__Gable__Metal = 0
 7.7.5__z_eq_0_if_noty__Gable_Metal: Y__Gable__Metal = 0 -> Z__Gable__Metal
   = 0
 7.7.5__z_eq_plan_if_y__Gable_Roll: Y__Gable__Roll = 1 -> - PlanRoofArea
   + Z__Gable__Roll = 0
 7.7.5__z_eq_0_if_noty__Gable_Roll: Y__Gable__Roll = 0 -> Z__Gable__Roll
   = 0
 7.7.5__z_eq_plan_if_y__Gable_Tar&Grv: Y__Gable__Tar&Grv = 1 ->
   - PlanRoofArea + Z__Gable__Tar&Grv = 0
 7.7.5__z_eq_0_if_noty__Gable_Tar&Grv: Y__Gable__Tar&Grv = 0 ->
   Z__Gable__Tar&Grv = 0
 7.7.5__z_eq_plan_if_y__Gable_WdShake: Y__Gable__WdShake = 1 ->
   - PlanRoofArea + Z__Gable__WdShake = 0
 7.7.5__z_eq_0_if_noty__Gable_WdShake: Y__Gable__WdShake = 0 ->
   Z__Gable__WdShake = 0
 7.7.5__z_eq_plan_if_y__Gable_WdShngl: Y__Gable__WdShngl = 1 ->
   - PlanRoofArea + Z__Gable__WdShngl = 0
 7.7.5__z_eq_0_if_noty__Gable_WdShngl: Y__Gable__WdShngl = 0 ->
   Z__Gable__WdShngl = 0
 7.7.5__z_eq_plan_if_y__Gambrel_ClyTile: Y__Gambrel__ClyTile = 1 ->
   - PlanRoofArea + Z__Gambrel__ClyTile = 0
 7.7.5__z_eq_0_if_noty__Gambrel_ClyTile: Y__Gambrel__ClyTile = 0 ->
   Z__Gambrel__ClyTile = 0
 7.7.5__z_eq_plan_if_y__Gambrel_CompShg: Y__Gambrel__CompShg = 1 ->
   - PlanRoofArea + Z__Gambrel__CompShg = 0
 7.7.5__z_eq_0_if_noty__Gambrel_CompShg: Y__Gambrel__CompShg = 0 ->
   Z__Gambrel__CompShg = 0
 7.7.5__z_eq_plan_if_y__Gambrel_Membran: Y__Gambrel__Membran = 1 ->
   - PlanRoofArea + Z__Gambrel__Membran = 0
 7.7.5__z_eq_0_if_noty__Gambrel_Membran: Y__Gambrel__Membran = 0 ->
   Z__Gambrel__Membran = 0
 7.7.5__z_eq_plan_if_y__Gambrel_Metal: Y__Gambrel__Metal = 1 ->
   - PlanRoofArea + Z__Gambrel__Metal = 0
 7.7.5__z_eq_0_if_noty__Gambrel_Metal: Y__Gambrel__Metal = 0 ->
   Z__Gambrel__Metal = 0
 7.7.5__z_eq_plan_if_y__Gambrel_Roll: Y__Gambrel__Roll = 1 ->
   - PlanRoofArea + Z__Gambrel__Roll = 0
 7.7.5__z_eq_0_if_noty__Gambrel_Roll: Y__Gambrel__Roll = 0 ->
   Z__Gambrel__Roll = 0
 7.7.5__z_eq_plan_if_y__Gambrel_Tar&Grv: Y__Gambrel__Tar&Grv = 1 ->
   - PlanRoofArea + Z__Gambrel__Tar&Grv = 0
 7.7.5__z_eq_0_if_noty__Gambrel_Tar&Grv: Y__Gambrel__Tar&Grv = 0 ->
   Z__Gambrel__Tar&Grv = 0
 7.7.5__z_eq_plan_if_y__Gambrel_WdShake: Y__Gambrel__WdShake = 1 ->
   - PlanRoofArea + Z__Gambrel__WdShake = 0
 7.7.5__z_eq_0_if_noty__Gambrel_WdShake: Y__Gambrel__WdShake = 0 ->
   Z__Gambrel__WdShake = 0
 7.7.5__z_eq_plan_if_y__Gambrel_WdShngl: Y__Gambrel__WdShngl = 1 ->
   - PlanRoofArea + Z__Gambrel__WdShngl = 0
 7.7.5__z_eq_0_if_noty__Gambrel_WdShngl: Y__Gambrel__WdShngl = 0 ->
   Z__Gambrel__WdShngl = 0
 7.7.5__z_eq_plan_if_y__Hip_ClyTile: Y__Hip__ClyTile = 1 -> - PlanRoofArea
   + Z__Hip__ClyTile = 0
 7.7.5__z_eq_0_if_noty__Hip_ClyTile: Y__Hip__ClyTile = 0 -> Z__Hip__ClyTile
   = 0
 7.7.5__z_eq_plan_if_y__Hip_CompShg: Y__Hip__CompShg = 1 -> - PlanRoofArea
   + Z__Hip__CompShg = 0
 7.7.5__z_eq_0_if_noty__Hip_CompShg: Y__Hip__CompShg = 0 -> Z__Hip__CompShg
   = 0
 7.7.5__z_eq_plan_if_y__Hip_Membran: Y__Hip__Membran = 1 -> - PlanRoofArea
   + Z__Hip__Membran = 0
 7.7.5__z_eq_0_if_noty__Hip_Membran: Y__Hip__Membran = 0 -> Z__Hip__Membran
   = 0
 7.7.5__z_eq_plan_if_y__Hip_Metal: Y__Hip__Metal = 1 -> - PlanRoofArea
   + Z__Hip__Metal = 0
 7.7.5__z_eq_0_if_noty__Hip_Metal: Y__Hip__Metal = 0 -> Z__Hip__Metal = 0
 7.7.5__z_eq_plan_if_y__Hip_Roll: Y__Hip__Roll = 1 -> - PlanRoofArea
   + Z__Hip__Roll = 0
 7.7.5__z_eq_0_if_noty__Hip_Roll: Y__Hip__Roll = 0 -> Z__Hip__Roll = 0
 7.7.5__z_eq_plan_if_y__Hip_Tar&Grv: Y__Hip__Tar&Grv = 1 -> - PlanRoofArea
   + Z__Hip__Tar&Grv = 0
 7.7.5__z_eq_0_if_noty__Hip_Tar&Grv: Y__Hip__Tar&Grv = 0 -> Z__Hip__Tar&Grv
   = 0
 7.7.5__z_eq_plan_if_y__Hip_WdShake: Y__Hip__WdShake = 1 -> - PlanRoofArea
   + Z__Hip__WdShake = 0
 7.7.5__z_eq_0_if_noty__Hip_WdShake: Y__Hip__WdShake = 0 -> Z__Hip__WdShake
   = 0
 7.7.5__z_eq_plan_if_y__Hip_WdShngl: Y__Hip__WdShngl = 1 -> - PlanRoofArea
   + Z__Hip__WdShngl = 0
 7.7.5__z_eq_0_if_noty__Hip_WdShngl: Y__Hip__WdShngl = 0 -> Z__Hip__WdShngl
   = 0
 7.7.5__z_eq_plan_if_y__Mansard_ClyTile: Y__Mansard__ClyTile = 1 ->
   - PlanRoofArea + Z__Mansard__ClyTile = 0
 7.7.5__z_eq_0_if_noty__Mansard_ClyTile: Y__Mansard__ClyTile = 0 ->
   Z__Mansard__ClyTile = 0
 7.7.5__z_eq_plan_if_y__Mansard_CompShg: Y__Mansard__CompShg = 1 ->
   - PlanRoofArea + Z__Mansard__CompShg = 0
 7.7.5__z_eq_0_if_noty__Mansard_CompShg: Y__Mansard__CompShg = 0 ->
   Z__Mansard__CompShg = 0
 7.7.5__z_eq_plan_if_y__Mansard_Membran: Y__Mansard__Membran = 1 ->
   - PlanRoofArea + Z__Mansard__Membran = 0
 7.7.5__z_eq_0_if_noty__Mansard_Membran: Y__Mansard__Membran = 0 ->
   Z__Mansard__Membran = 0
 7.7.5__z_eq_plan_if_y__Mansard_Metal: Y__Mansard__Metal = 1 ->
   - PlanRoofArea + Z__Mansard__Metal = 0
 7.7.5__z_eq_0_if_noty__Mansard_Metal: Y__Mansard__Metal = 0 ->
   Z__Mansard__Metal = 0
 7.7.5__z_eq_plan_if_y__Mansard_Roll: Y__Mansard__Roll = 1 ->
   - PlanRoofArea + Z__Mansard__Roll = 0
 7.7.5__z_eq_0_if_noty__Mansard_Roll: Y__Mansard__Roll = 0 ->
   Z__Mansard__Roll = 0
 7.7.5__z_eq_plan_if_y__Mansard_Tar&Grv: Y__Mansard__Tar&Grv = 1 ->
   - PlanRoofArea + Z__Mansard__Tar&Grv = 0
 7.7.5__z_eq_0_if_noty__Mansard_Tar&Grv: Y__Mansard__Tar&Grv = 0 ->
   Z__Mansard__Tar&Grv = 0
 7.7.5__z_eq_plan_if_y__Mansard_WdShake: Y__Mansard__WdShake = 1 ->
   - PlanRoofArea + Z__Mansard__WdShake = 0
 7.7.5__z_eq_0_if_noty__Mansard_WdShake: Y__Mansard__WdShake = 0 ->
   Z__Mansard__WdShake = 0
 7.7.5__z_eq_plan_if_y__Mansard_WdShngl: Y__Mansard__WdShngl = 1 ->
   - PlanRoofArea + Z__Mansard__WdShngl = 0
 7.7.5__z_eq_0_if_noty__Mansard_WdShngl: Y__Mansard__WdShngl = 0 ->
   Z__Mansard__WdShngl = 0
 7.7.5__z_eq_plan_if_y__Shed_ClyTile: Y__Shed__ClyTile = 1 ->
   - PlanRoofArea + Z__Shed__ClyTile = 0
 7.7.5__z_eq_0_if_noty__Shed_ClyTile: Y__Shed__ClyTile = 0 ->
   Z__Shed__ClyTile = 0
 7.7.5__z_eq_plan_if_y__Shed_CompShg: Y__Shed__CompShg = 1 ->
   - PlanRoofArea + Z__Shed__CompShg = 0
 7.7.5__z_eq_0_if_noty__Shed_CompShg: Y__Shed__CompShg = 0 ->
   Z__Shed__CompShg = 0
 7.7.5__z_eq_plan_if_y__Shed_Membran: Y__Shed__Membran = 1 ->
   - PlanRoofArea + Z__Shed__Membran = 0
 7.7.5__z_eq_0_if_noty__Shed_Membran: Y__Shed__Membran = 0 ->
   Z__Shed__Membran = 0
 7.7.5__z_eq_plan_if_y__Shed_Metal: Y__Shed__Metal = 1 -> - PlanRoofArea
   + Z__Shed__Metal = 0
 7.7.5__z_eq_0_if_noty__Shed_Metal: Y__Shed__Metal = 0 -> Z__Shed__Metal
   = 0
 7.7.5__z_eq_plan_if_y__Shed_Roll: Y__Shed__Roll = 1 -> - PlanRoofArea
   + Z__Shed__Roll = 0
 7.7.5__z_eq_0_if_noty__Shed_Roll: Y__Shed__Roll = 0 -> Z__Shed__Roll = 0
 7.7.5__z_eq_plan_if_y__Shed_Tar&Grv: Y__Shed__Tar&Grv = 1 ->
   - PlanRoofArea + Z__Shed__Tar&Grv = 0
 7.7.5__z_eq_0_if_noty__Shed_Tar&Grv: Y__Shed__Tar&Grv = 0 ->
   Z__Shed__Tar&Grv = 0
 7.7.5__z_eq_plan_if_y__Shed_WdShake: Y__Shed__WdShake = 1 ->
   - PlanRoofArea + Z__Shed__WdShake = 0
 7.7.5__z_eq_0_if_noty__Shed_WdShake: Y__Shed__WdShake = 0 ->
   Z__Shed__WdShake = 0
 7.7.5__z_eq_plan_if_y__Shed_WdShngl: Y__Shed__WdShngl = 1 ->
   - PlanRoofArea + Z__Shed__WdShngl = 0
 7.7.5__z_eq_0_if_noty__Shed_WdShngl: Y__Shed__WdShngl = 0 ->
   Z__Shed__WdShngl = 0
 11.xx__poolqc_eq_na_if_no_pool: HasPool = 0 -> x_Pool_QC = -1
 link__pqc_ord: HasPool = 1 -> x_Pool_QC - 2 zPQC_TA - 3 zPQC_Gd
   - 4 zPQC_Ex = 0
 7.23__p2_ub_indicator: Floor2 = 1 -> - 0.1 x_2nd_Flr_SF + P2 <= 40
 7.23__p2_off: Floor2 = 0 -> P2 = 0
 7.23.1__w_eq_area_if_b__VinylSd: Exterior1st__VinylSd = 1 ->
   - AreaExterior1st + W__VinylSd = 0
 7.23.1__w_eq_0_if_notb__VinylSd: Exterior1st__VinylSd = 0 -> W__VinylSd
   = 0
 7.23.1__w_eq_area_if_b__MetalSd: Exterior1st__MetalSd = 1 ->
   - AreaExterior1st + W__MetalSd = 0
 7.23.1__w_eq_0_if_notb__MetalSd: Exterior1st__MetalSd = 0 -> W__MetalSd
   = 0
 7.23.1__w_eq_area_if_b__Wd_Sdng: Exterior1st__Wd_Sdng = 1 ->
   - AreaExterior1st + W__Wd_Sdng = 0
 7.23.1__w_eq_0_if_notb__Wd_Sdng: Exterior1st__Wd_Sdng = 0 -> W__Wd_Sdng
   = 0
 7.23.1__w_eq_area_if_b__HdBoard: Exterior1st__HdBoard = 1 ->
   - AreaExterior1st + W__HdBoard = 0
 7.23.1__w_eq_0_if_notb__HdBoard: Exterior1st__HdBoard = 0 -> W__HdBoard
   = 0
 7.23.1__w_eq_area_if_b__Stucco: Exterior1st__Stucco = 1 ->
   - AreaExterior1st + W__Stucco = 0
 7.23.1__w_eq_0_if_notb__Stucco: Exterior1st__Stucco = 0 -> W__Stucco = 0
 7.23.1__w_eq_area_if_b__Plywood: Exterior1st__Plywood = 1 ->
   - AreaExterior1st + W__Plywood = 0
 7.23.1__w_eq_0_if_notb__Plywood: Exterior1st__Plywood = 0 -> W__Plywood
   = 0
 7.23.1__w_eq_area_if_b__CemntBd: Exterior1st__CemntBd = 1 ->
   - AreaExterior1st + W__CemntBd = 0
 7.23.1__w_eq_0_if_notb__CemntBd: Exterior1st__CemntBd = 0 -> W__CemntBd
   = 0
 7.23.1__w_eq_area_if_b__BrkFace: Exterior1st__BrkFace = 1 ->
   - AreaExterior1st + W__BrkFace = 0
 7.23.1__w_eq_0_if_notb__BrkFace: Exterior1st__BrkFace = 0 -> W__BrkFace
   = 0
 7.23.1__w_eq_area_if_b__BrkComm: Exterior1st__BrkComm = 1 ->
   - AreaExterior1st + W__BrkComm = 0
 7.23.1__w_eq_0_if_notb__BrkComm: Exterior1st__BrkComm = 0 -> W__BrkComm
   = 0
 7.23.1__w_eq_area_if_b__WdShngl: Exterior1st__WdShngl = 1 ->
   - AreaExterior1st + W__WdShngl = 0
 7.23.1__w_eq_0_if_notb__WdShngl: Exterior1st__WdShngl = 0 -> W__WdShngl
   = 0
 7.23.1__w_eq_area_if_b__AsbShng: Exterior1st__AsbShng = 1 ->
   - AreaExterior1st + W__AsbShng = 0
 7.23.1__w_eq_0_if_notb__AsbShng: Exterior1st__AsbShng = 0 -> W__AsbShng
   = 0
 7.23.1__w_eq_area_if_b__Stone: Exterior1st__Stone = 1 -> - AreaExterior1st
   + W__Stone = 0
 7.23.1__w_eq_0_if_notb__Stone: Exterior1st__Stone = 0 -> W__Stone = 0
 7.23.1__w_eq_area_if_b__ImStucc: Exterior1st__ImStucc = 1 ->
   - AreaExterior1st + W__ImStucc = 0
 7.23.1__w_eq_0_if_notb__ImStucc: Exterior1st__ImStucc = 0 -> W__ImStucc
   = 0
 7.23.1__w_eq_area_if_b__AsphShn: Exterior1st__AsphShn = 1 ->
   - AreaExterior1st + W__AsphShn = 0
 7.23.1__w_eq_0_if_notb__AsphShn: Exterior1st__AsphShn = 0 -> W__AsphShn
   = 0
 7.23.1__w_eq_area_if_b__CBlock: Exterior1st__CBlock = 1 ->
   - AreaExterior1st + W__CBlock = 0
 7.23.1__w_eq_0_if_notb__CBlock: Exterior1st__CBlock = 0 -> W__CBlock = 0
Bounds
 x_Lot_Frontage <= 175.53
 x_Lot_Area = 7000
 -1 <= x_Utilities <= 4
 x_Mas_Vnr_Area <= 931.07
 2 <= x_Exter_Qual <= 4
 -1 <= x_Exter_Cond <= 4
 2 <= x_Bsmt_Qual <= 4
 -1 <= x_Bsmt_Cond <= 4
 x_BsmtFin_SF_1 <= 1984.5
 x_BsmtFin_SF_2 <= 757.05
 x_Bsmt_Unf_SF <= 1812.68
 x_Total_Bsmt_SF <= 2514.48
 2 <= x_Heating_QC <= 4
 x_1st_Flr_SF <= 2534.9
 x_2nd_Flr_SF <= 1641.54
 x_Gr_Liv_Area <= 3421.76
 x_Bsmt_Full_Bath <= 2.02
 x_Bsmt_Half_Bath <= 1.01
 x_Full_Bath <= 6
 x_Half_Bath <= 5
 x_Bedroom_AbvGr <= 7
 x_Kitchen_AbvGr <= 3
 2 <= x_Kitchen_Qual <= 4
 x_TotRms_AbvGrd <= 12.08
 x_Fireplaces <= 3
 2 <= x_Fireplace_Qu <= 4
 x_Garage_Cars <= 4
 x_Garage_Area <= 1163.75
 2 <= x_Garage_Qual <= 4
 -1 <= x_Garage_Cond <= 4
 x_Wood_Deck_SF <= 525.9
 x_Open_Porch_SF <= 488.81
 x_Enclosed_Porch <= 292.75
 x_3Ssn_Porch <= 1750
 x_Screen_Porch <= 268.27
 x_Pool_Area <= 444.77
 -1 <= x_Pool_QC <= 4
 x_Misc_Val <= 3029.47
 1 <= OverallQualInt <= 10
 CONST_ONE = 1
 10.3 <= y_log <= 13.8
 -1e+06 <= y_log_cal <= 1e+06
 y_price <= 984608.1112290357
Binaries
 x_MS_SubClass_20 x_MS_SubClass_30 x_MS_SubClass_40 x_MS_SubClass_45
 x_MS_SubClass_50 x_MS_SubClass_60 x_MS_SubClass_70 x_MS_SubClass_75
 x_MS_SubClass_80 x_MS_SubClass_85 x_MS_SubClass_90 x_MS_SubClass_120
 x_MS_SubClass_150 x_MS_SubClass_160 x_MS_SubClass_180 x_MS_SubClass_190
 x_MS_Zoning_A_(agr) x_MS_Zoning_C_(all) x_MS_Zoning_FV x_MS_Zoning_I_(all)
 x_MS_Zoning_RH x_MS_Zoning_RL x_MS_Zoning_RM x_Street_Grvl x_Street_Pave
 x_Alley_Grvl x_Alley_No_aplica x_Alley_Pave x_Lot_Shape_IR1
 x_Lot_Shape_IR2 x_Lot_Shape_IR3 x_Lot_Shape_Reg x_Land_Contour_Bnk
 x_Land_Contour_HLS x_Land_Contour_Low x_Land_Contour_Lvl
 x_Lot_Config_Corner x_Lot_Config_CulDSac x_Lot_Config_FR2 x_Lot_Config_FR3
 x_Lot_Config_Inside x_Land_Slope_Gtl x_Land_Slope_Mod x_Land_Slope_Sev
 x_Neighborhood_Blmngtn x_Neighborhood_Blueste x_Neighborhood_BrDale
 x_Neighborhood_BrkSide x_Neighborhood_ClearCr x_Neighborhood_CollgCr
 x_Neighborhood_Crawfor x_Neighborhood_Edwards x_Neighborhood_Gilbert
 x_Neighborhood_Greens x_Neighborhood_GrnHill x_Neighborhood_IDOTRR
 x_Neighborhood_Landmrk x_Neighborhood_MeadowV x_Neighborhood_Mitchel
 x_Neighborhood_NPkVill x_Neighborhood_NWAmes x_Neighborhood_No_aplicames
 x_Neighborhood_NoRidge x_Neighborhood_NridgHt x_Neighborhood_OldTown
 x_Neighborhood_SWISU x_Neighborhood_Sawyer x_Neighborhood_SawyerW
 x_Neighborhood_Somerst x_Neighborhood_StoneBr x_Neighborhood_Timber
 x_Neighborhood_Veenker x_Condition_1_Artery x_Condition_1_Feedr
 x_Condition_1_Norm x_Condition_1_PosA x_Condition_1_PosN
 x_Condition_1_RRAe x_Condition_1_RRAn x_Condition_1_RRNe
 x_Condition_1_RRNn x_Condition_2_Artery x_Condition_2_Feedr
 x_Condition_2_Norm x_Condition_2_PosA x_Condition_2_PosN
 x_Condition_2_RRAe x_Condition_2_RRAn x_Condition_2_RRNn x_Bldg_Type_1Fam
 x_Bldg_Type_2fmCon x_Bldg_Type_Duplex x_Bldg_Type_Twnhs x_Bldg_Type_TwnhsE
 x_House_Style_1.5Fin x_House_Style_1.5Unf x_House_Style_1Story
 x_House_Style_2.5Fin x_House_Style_2.5Unf x_House_Style_2Story
 x_House_Style_SFoyer x_House_Style_SLvl x_Roof_Style_Flat
 x_Roof_Style_Gable x_Roof_Style_Gambrel x_Roof_Style_Hip
 x_Roof_Style_Mansard x_Roof_Style_Shed x_Roof_Matl_ClyTile
 x_Roof_Matl_CompShg x_Roof_Matl_Membran x_Roof_Matl_Metal x_Roof_Matl_Roll
 x_Roof_Matl_Tar&Grv x_Roof_Matl_WdShake x_Roof_Matl_WdShngl
 x_Exterior_1st_AsbShng x_Exterior_1st_AsphShn x_Exterior_1st_BrkComm
 x_Exterior_1st_BrkFace x_Exterior_1st_CBlock x_Exterior_1st_CemntBd
 x_Exterior_1st_HdBoard x_Exterior_1st_ImStucc x_Exterior_1st_MetalSd
 x_Exterior_1st_Plywood x_Exterior_1st_PreCast x_Exterior_1st_Stone
 x_Exterior_1st_Stucco x_Exterior_1st_VinylSd x_Exterior_1st_Wd_Sdng
 x_Exterior_1st_WdShing x_Exterior_2nd_AsbShng x_Exterior_2nd_AsphShn
 x_Exterior_2nd_Brk_Cmn x_Exterior_2nd_BrkFace x_Exterior_2nd_CBlock
 x_Exterior_2nd_CmentBd x_Exterior_2nd_HdBoard x_Exterior_2nd_ImStucc
 x_Exterior_2nd_MetalSd x_Exterior_2nd_Other x_Exterior_2nd_Plywood
 x_Exterior_2nd_PreCast x_Exterior_2nd_Stone x_Exterior_2nd_Stucco
 x_Exterior_2nd_VinylSd x_Exterior_2nd_Wd_Sdng x_Exterior_2nd_Wd_Shng
 x_Mas_Vnr_Type_BrkCmn x_Mas_Vnr_Type_BrkFace x_Mas_Vnr_Type_CBlock
 x_Mas_Vnr_Type_No_aplica x_Mas_Vnr_Type_NoAplica x_Mas_Vnr_Type_Stone
 x_Foundation_BrkTil x_Foundation_CBlock x_Foundation_PConc
 x_Foundation_Slab x_Foundation_Stone x_Foundation_Wood x_Bsmt_Exposure_Av
 x_Bsmt_Exposure_Gd x_Bsmt_Exposure_Mn x_Bsmt_Exposure_No
 x_Bsmt_Exposure_No_aplica x_BsmtFin_Type_1_ALQ x_BsmtFin_Type_1_BLQ
 x_BsmtFin_Type_1_GLQ x_BsmtFin_Type_1_LwQ x_BsmtFin_Type_1_No_aplica
 x_BsmtFin_Type_1_Rec x_BsmtFin_Type_1_Unf x_BsmtFin_Type_2_ALQ
 x_BsmtFin_Type_2_BLQ x_BsmtFin_Type_2_GLQ x_BsmtFin_Type_2_LwQ
 x_BsmtFin_Type_2_No_aplica x_BsmtFin_Type_2_Rec x_BsmtFin_Type_2_Unf
 x_Heating_Floor x_Heating_GasA x_Heating_GasW x_Heating_Grav
 x_Heating_OthW x_Heating_Wall x_Central_Air_N x_Central_Air_Y
 x_Electrical_FuseA x_Electrical_FuseF x_Electrical_FuseP x_Electrical_Mix
 x_Electrical_SBrkr x_FunctioNo_aplical_Maj1 x_FunctioNo_aplical_Maj2
 x_FunctioNo_aplical_Min1 x_FunctioNo_aplical_Min2 x_FunctioNo_aplical_Mod
 x_FunctioNo_aplical_Sal x_FunctioNo_aplical_Sev x_FunctioNo_aplical_Typ
 x_Garage_Type_2Types x_Garage_Type_Attchd x_Garage_Type_Basment
 x_Garage_Type_BuiltIn x_Garage_Type_CarPort x_Garage_Type_Detchd
 x_Garage_Type_No_aplica x_Garage_Finish_Fin x_Garage_Finish_No_aplica
 x_Garage_Finish_RFn x_Garage_Finish_Unf x_Paved_Drive_N x_Paved_Drive_P
 x_Paved_Drive_Y x_Fence_GdPrv x_Fence_GdWo x_Fence_MnPrv x_Fence_MnWw
 x_Fence_No_aplica x_Misc_Feature_Elev x_Misc_Feature_Gar2
 x_Misc_Feature_No_aplica x_Misc_Feature_Othr x_Misc_Feature_Shed
 x_Misc_Feature_TenC x_Mo_Sold_1 x_Mo_Sold_2 x_Mo_Sold_3 x_Mo_Sold_4
 x_Mo_Sold_5 x_Mo_Sold_6 x_Mo_Sold_7 x_Mo_Sold_8 x_Mo_Sold_9 x_Mo_Sold_10
 x_Mo_Sold_11 x_Mo_Sold_12 x_Sale_Type_COD x_Sale_Type_CWD x_Sale_Type_Con
 x_Sale_Type_ConLD x_Sale_Type_ConLI x_Sale_Type_ConLw x_Sale_Type_New
 x_Sale_Type_Oth x_Sale_Type_VWD x_Sale_Type_WD_ x_Sale_Condition_Abnorml
 x_Sale_Condition_AdjLand x_Sale_Condition_Alloca x_Sale_Condition_Family
 x_Sale_Condition_Normal x_Sale_Condition_Partial Floor1 Floor2 IsOneStory
 IsTwoStory HasOpenPorch HasEnclosedPorch HasScreenPorch Has3SsnPorch
 HasWoodDeck HasPool HasReja HasKitchen BldgType__1Fam BldgType__TwnhsE
 BldgType__TwnhsI BldgType__Duplex BldgType__2FmCon zKitchen_2 zKitchen_3
 HasFireplace zFQ_TA zFQ_Gd zFQ_Ex GarageType__NA GarageType__Attchd
 GarageType__Detchd GarageType__BuiltIn GarageType__Basment
 GarageType__CarPort GarageType__2Types GarageFinish__NA GarageFinish__Fin
 GarageFinish__RFn GarageFinish__Unf zGQ_TA zGQ_Gd zGQ_Ex zHQ_TA zHQ_Gd
 zHQ_Ex zEQ_TA zEQ_Gd zEQ_Ex HouseStyle__1Story HouseStyle__2Story
 Heating__Floor Heating__GasA Heating__GasW Heating__Grav Heating__OthW
 Heating__Wall CentralAir__N CentralAir__Y Electrical__SBrkr
 Electrical__FuseA Electrical__FuseF Electrical__FuseP Electrical__Mix
 PavedDrive__Y PavedDrive__P PavedDrive__N MiscFeature__Elev
 MiscFeature__Gar2 MiscFeature__Othr MiscFeature__Shed MiscFeature__TenC
 MiscFeature__No_aplica RoofStyle__Flat RoofStyle__Gable RoofStyle__Gambrel
 RoofStyle__Hip RoofStyle__Mansard RoofStyle__Shed RoofMatl__ClyTile
 RoofMatl__CompShg RoofMatl__Membran RoofMatl__Metal RoofMatl__Roll
 RoofMatl__Tar&Grv RoofMatl__WdShake RoofMatl__WdShngl Y__Flat__ClyTile
 Y__Flat__CompShg Y__Flat__Membran Y__Flat__Metal Y__Flat__Roll
 Y__Flat__Tar&Grv Y__Flat__WdShake Y__Flat__WdShngl Y__Gable__ClyTile
 Y__Gable__CompShg Y__Gable__Membran Y__Gable__Metal Y__Gable__Roll
 Y__Gable__Tar&Grv Y__Gable__WdShake Y__Gable__WdShngl Y__Gambrel__ClyTile
 Y__Gambrel__CompShg Y__Gambrel__Membran Y__Gambrel__Metal Y__Gambrel__Roll
 Y__Gambrel__Tar&Grv Y__Gambrel__WdShake Y__Gambrel__WdShngl
 Y__Hip__ClyTile Y__Hip__CompShg Y__Hip__Membran Y__Hip__Metal Y__Hip__Roll
 Y__Hip__Tar&Grv Y__Hip__WdShake Y__Hip__WdShngl Y__Mansard__ClyTile
 Y__Mansard__CompShg Y__Mansard__Membran Y__Mansard__Metal Y__Mansard__Roll
 Y__Mansard__Tar&Grv Y__Mansard__WdShake Y__Mansard__WdShngl
 Y__Shed__ClyTile Y__Shed__CompShg Y__Shed__Membran Y__Shed__Metal
 Y__Shed__Roll Y__Shed__Tar&Grv Y__Shed__WdShake Y__Shed__WdShngl zPQC_TA
 zPQC_Gd zPQC_Ex BsmtExposure__Gd BsmtExposure__Av BsmtExposure__Mn
 BsmtExposure__No BsmtExposure__NA BsmtFinType1__GLQ BsmtFinType1__ALQ
 BsmtFinType1__BLQ BsmtFinType1__Rec BsmtFinType1__LwQ BsmtFinType1__Unf
 BsmtFinType1__NA BsmtFinType2__GLQ BsmtFinType2__ALQ BsmtFinType2__BLQ
 BsmtFinType2__Rec BsmtFinType2__LwQ BsmtFinType2__Unf BsmtFinType2__NA
 SameMaterial Exterior1st__VinylSd Exterior1st__MetalSd
 Exterior1st__Wd_Sdng Exterior1st__HdBoard Exterior1st__Stucco
 Exterior1st__Plywood Exterior1st__CemntBd Exterior1st__BrkFace
 Exterior1st__BrkComm Exterior1st__WdShngl Exterior1st__AsbShng
 Exterior1st__Stone Exterior1st__ImStucc Exterior1st__AsphShn
 Exterior1st__CBlock Exterior2nd__VinylSd Exterior2nd__MetalSd
 Exterior2nd__Wd_Sdng Exterior2nd__HdBoard Exterior2nd__Stucco
 Exterior2nd__Plywood Exterior2nd__CemntBd Exterior2nd__BrkFace
 Exterior2nd__BrkComm Exterior2nd__WdShngl Exterior2nd__AsbShng
 Exterior2nd__Stone Exterior2nd__ImStucc Exterior2nd__AsphShn
 Exterior2nd__CBlock MasVnrType__BrkCmn MasVnrType__BrkFace
 MasVnrType__CBlock MasVnrType__None MasVnrType__Stone Foundation__BrkTil
 Foundation__CBlock Foundation__PConc Foundation__Slab Foundation__Stone
 Foundation__Wood Fence__GdPrv Fence__MnPrv Fence__GdWo Fence__MnWw
 Fence__No_aplica MiscFeature__Elev MiscFeature__Gar2 MiscFeature__Othr
 MiscFeature__Shed MiscFeature__TenC MiscFeature__No_aplica t0_leaf0
 t1_leaf0 t2_leaf0 t3_leaf0 t4_leaf0 t5_leaf0 t6_leaf0 t7_leaf0 t8_leaf0
 t9_leaf0 t10_leaf0 t11_leaf0 t12_leaf0 t13_leaf0 t14_leaf0 t15_leaf0
 t16_leaf0 t17_leaf0 t18_leaf0 t19_leaf0 t20_leaf0 t21_leaf0 t22_leaf0
 t23_leaf0 t23_leaf1 t24_leaf0 t24_leaf1 t25_leaf0 t25_leaf1 t26_leaf0
 t26_leaf1 t27_leaf0 t27_leaf1 t28_leaf0 t28_leaf1 t29_leaf0 t29_leaf1
 t30_leaf0 t30_leaf1 t31_leaf0 t31_leaf1 t32_leaf0 t32_leaf1 t33_leaf0
 t33_leaf1 t34_leaf0 t34_leaf1 t34_leaf2 t35_leaf0 t35_leaf1 t36_leaf0
 t36_leaf1 t37_leaf0 t37_leaf1 t37_leaf2 t38_leaf0 t38_leaf1 t38_leaf2
 t39_leaf0 t39_leaf1 t39_leaf2 t40_leaf0 t40_leaf1 t40_leaf2 t41_leaf0
 t41_leaf1 t41_leaf2 t42_leaf0 t42_leaf1 t43_leaf0 t43_leaf1 t43_leaf2
 t44_leaf0 t44_leaf1 t44_leaf2 t45_leaf0 t45_leaf1 t45_leaf2 t46_leaf0
 t46_leaf1 t46_leaf2 t46_leaf3 t47_leaf0 t47_leaf1 t47_leaf2 t48_leaf0
 t48_leaf1 t48_leaf2 t48_leaf3 t48_leaf4 t48_leaf5 t49_leaf0 t49_leaf1
 t49_leaf2 t49_leaf3 t49_leaf4 t50_leaf0 t50_leaf1 t50_leaf2 t50_leaf3
 t51_leaf0 t51_leaf1 t51_leaf2 t51_leaf3 t51_leaf4 t52_leaf0 t52_leaf1
 t52_leaf2 t52_leaf3 t53_leaf0 t53_leaf1 t53_leaf2 t53_leaf3 t53_leaf4
 t54_leaf0 t54_leaf1 t54_leaf2 t54_leaf3 t54_leaf4 t55_leaf0 t55_leaf1
 t55_leaf2 t55_leaf3 t55_leaf4 t55_leaf5 t56_leaf0 t56_leaf1 t56_leaf2
 t56_leaf3 t57_leaf0 t57_leaf1 t57_leaf2 t57_leaf3 t58_leaf0 t58_leaf1
 t58_leaf2 t58_leaf3 t58_leaf4 t59_leaf0 t59_leaf1 t59_leaf2 t59_leaf3
 t59_leaf4 t60_leaf0 t60_leaf1 t60_leaf2 t60_leaf3 t60_leaf4 t61_leaf0
 t61_leaf1 t61_leaf2 t61_leaf3 t61_leaf4 t61_leaf5 t62_leaf0 t62_leaf1
 t62_leaf2 t62_leaf3 t62_leaf4 t63_leaf0 t63_leaf1 t63_leaf2 t63_leaf3
 t63_leaf4 t63_leaf5 t64_leaf0 t64_leaf1 t64_leaf2 t64_leaf3 t64_leaf4
 t64_leaf5 t64_leaf6 t65_leaf0 t65_leaf1 t65_leaf2 t65_leaf3 t65_leaf4
 t65_leaf5 t66_leaf0 t66_leaf1 t66_leaf2 t66_leaf3 t66_leaf4 t66_leaf5
 t66_leaf6 t67_leaf0 t67_leaf1 t67_leaf2 t67_leaf3 t67_leaf4 t67_leaf5
 t67_leaf6 t68_leaf0 t68_leaf1 t68_leaf2 t68_leaf3 t68_leaf4 t68_leaf5
 t69_leaf0 t69_leaf1 t69_leaf2 t69_leaf3 t69_leaf4 t69_leaf5 t70_leaf0
 t70_leaf1 t70_leaf2 t70_leaf3 t70_leaf4 t70_leaf5 t71_leaf0 t71_leaf1
 t71_leaf2 t71_leaf3 t71_leaf4 t71_leaf5 t71_leaf6 t72_leaf0 t72_leaf1
 t72_leaf2 t72_leaf3 t72_leaf4 t72_leaf5 t72_leaf6 t73_leaf0 t73_leaf1
 t73_leaf2 t73_leaf3 t73_leaf4 t73_leaf5 t74_leaf0 t74_leaf1 t74_leaf2
 t74_leaf3 t74_leaf4 t74_leaf5 t75_leaf0 t75_leaf1 t75_leaf2 t75_leaf3
 t75_leaf4 t75_leaf5 t75_leaf6 t76_leaf0 t76_leaf1 t76_leaf2 t76_leaf3
 t76_leaf4 t76_leaf5 t77_leaf0 t77_leaf1 t77_leaf2 t77_leaf3 t77_leaf4
 t77_leaf5 t77_leaf6 t77_leaf7 t78_leaf0 t78_leaf1 t78_leaf2 t78_leaf3
 t78_leaf4 t78_leaf5 t79_leaf0 t79_leaf1 t79_leaf2 t79_leaf3 t79_leaf4
 t79_leaf5 t79_leaf6 t80_leaf0 t80_leaf1 t80_leaf2 t80_leaf3 t80_leaf4
 t80_leaf5 t80_leaf6 t81_leaf0 t81_leaf1 t81_leaf2 t81_leaf3 t81_leaf4
 t81_leaf5 t81_leaf6 t82_leaf0 t82_leaf1 t82_leaf2 t82_leaf3 t82_leaf4
 t82_leaf5 t83_leaf0 t83_leaf1 t83_leaf2 t83_leaf3 t83_leaf4 t83_leaf5
 t83_leaf6 t84_leaf0 t84_leaf1 t84_leaf2 t84_leaf3 t84_leaf4 t84_leaf5
 t84_leaf6 t84_leaf7 t85_leaf0 t85_leaf1 t85_leaf2 t85_leaf3 t85_leaf4
 t85_leaf5 t85_leaf6 t85_leaf7 t86_leaf0 t86_leaf1 t86_leaf2 t86_leaf3
 t86_leaf4 t86_leaf5 t86_leaf6 t86_leaf7 t87_leaf0 t87_leaf1 t87_leaf2
 t87_leaf3 t87_leaf4 t87_leaf5 t87_leaf6 t87_leaf7 t88_leaf0 t88_leaf1
 t88_leaf2 t88_leaf3 t88_leaf4 t88_leaf5 t88_leaf6 t89_leaf0 t89_leaf1
 t89_leaf2 t89_leaf3 t89_leaf4 t89_leaf5 t89_leaf6 t89_leaf7 t90_leaf0
 t90_leaf1 t90_leaf2 t90_leaf3 t90_leaf4 t90_leaf5 t90_leaf6 t91_leaf0
 t91_leaf1 t91_leaf2 t91_leaf3 t91_leaf4 t91_leaf5 t91_leaf6 t91_leaf7
 t92_leaf0 t92_leaf1 t92_leaf2 t92_leaf3 t92_leaf4 t92_leaf5 t92_leaf6
 t93_leaf0 t93_leaf1 t93_leaf2 t93_leaf3 t93_leaf4 t93_leaf5 t93_leaf6
 t94_leaf0 t94_leaf1 t94_leaf2 t94_leaf3 t94_leaf4 t94_leaf5 t95_leaf0
 t95_leaf1 t95_leaf2 t95_leaf3 t95_leaf4 t95_leaf5 t95_leaf6 t95_leaf7
 t96_leaf0 t96_leaf1 t96_leaf2 t96_leaf3 t96_leaf4 t96_leaf5 t96_leaf6
 t96_leaf7 t97_leaf0 t97_leaf1 t97_leaf2 t97_leaf3 t97_leaf4 t97_leaf5
 t97_leaf6 t98_leaf0 t98_leaf1 t98_leaf2 t98_leaf3 t98_leaf4 t98_leaf5
 t98_leaf6 t99_leaf0 t99_leaf1 t99_leaf2 t99_leaf3 t99_leaf4 t99_leaf5
 t99_leaf6 t100_leaf0 t100_leaf1 t100_leaf2 t100_leaf3 t100_leaf4
 t100_leaf5 t100_leaf6 t101_leaf0 t101_leaf1 t101_leaf2 t101_leaf3
 t101_leaf4 t101_leaf5 t101_leaf6 t101_leaf7 t102_leaf0 t102_leaf1
 t102_leaf2 t102_leaf3 t102_leaf4 t102_leaf5 t102_leaf6 t102_leaf7
 t103_leaf0 t103_leaf1 t103_leaf2 t103_leaf3 t103_leaf4 t103_leaf5
 t103_leaf6 t103_leaf7 t104_leaf0 t104_leaf1 t104_leaf2 t104_leaf3
 t104_leaf4 t104_leaf5 t104_leaf6 t105_leaf0 t105_leaf1 t105_leaf2
 t105_leaf3 t105_leaf4 t105_leaf5 t106_leaf0 t106_leaf1 t106_leaf2
 t106_leaf3 t106_leaf4 t106_leaf5 t106_leaf6 t106_leaf7 t107_leaf0
 t107_leaf1 t107_leaf2 t107_leaf3 t107_leaf4 t107_leaf5 t107_leaf6
 t107_leaf7 t108_leaf0 t108_leaf1 t108_leaf2 t108_leaf3 t108_leaf4
 t108_leaf5 t108_leaf6 t109_leaf0 t109_leaf1 t109_leaf2 t109_leaf3
 t109_leaf4 t109_leaf5 t110_leaf0 t110_leaf1 t110_leaf2 t110_leaf3
 t110_leaf4 t110_leaf5 t111_leaf0 t111_leaf1 t111_leaf2 t111_leaf3
 t111_leaf4 t111_leaf5 t111_leaf6 t111_leaf7 t112_leaf0 t112_leaf1
 t112_leaf2 t112_leaf3 t112_leaf4 t112_leaf5 t113_leaf0 t113_leaf1
 t113_leaf2 t113_leaf3 t113_leaf4 t113_leaf5 t113_leaf6 t113_leaf7
 t114_leaf0 t114_leaf1 t114_leaf2 t114_leaf3 t114_leaf4 t114_leaf5
 t114_leaf6 t115_leaf0 t115_leaf1 t115_leaf2 t115_leaf3 t115_leaf4
 t116_leaf0 t116_leaf1 t116_leaf2 t116_leaf3 t116_leaf4 t116_leaf5
 t116_leaf6 t117_leaf0 t117_leaf1 t117_leaf2 t117_leaf3 t117_leaf4
 t118_leaf0 t118_leaf1 t118_leaf2 t118_leaf3 t118_leaf4 t118_leaf5
 t118_leaf6 t119_leaf0 t119_leaf1 t119_leaf2 t119_leaf3 t119_leaf4
 t119_leaf5 t119_leaf6 t120_leaf0 t120_leaf1 t120_leaf2 t120_leaf3
 t120_leaf4 t120_leaf5 t120_leaf6 t121_leaf0 t121_leaf1 t121_leaf2
 t121_leaf3 t121_leaf4 t121_leaf5 t121_leaf6 t122_leaf0 t122_leaf1
 t122_leaf2 t122_leaf3 t122_leaf4 t122_leaf5 t122_leaf6 t122_leaf7
 t123_leaf0 t123_leaf1 t123_leaf2 t123_leaf3 t123_leaf4 t124_leaf0
 t124_leaf1 t124_leaf2 t124_leaf3 t124_leaf4 t124_leaf5 t124_leaf6
 t125_leaf0 t125_leaf1 t125_leaf2 t125_leaf3 t125_leaf4 t125_leaf5
 t125_leaf6 t125_leaf7 t126_leaf0 t126_leaf1 t126_leaf2 t126_leaf3
 t126_leaf4 t126_leaf5 t126_leaf6 t127_leaf0 t127_leaf1 t127_leaf2
 t127_leaf3 t127_leaf4 t127_leaf5 t128_leaf0 t128_leaf1 t128_leaf2
 t128_leaf3 t128_leaf4 t128_leaf5 t128_leaf6 t128_leaf7 t129_leaf0
 t129_leaf1 t129_leaf2 t129_leaf3 t129_leaf4 t129_leaf5 t129_leaf6
 t130_leaf0 t130_leaf1 t130_leaf2 t130_leaf3 t130_leaf4 t130_leaf5
 t130_leaf6 t131_leaf0 t131_leaf1 t131_leaf2 t131_leaf3 t131_leaf4
 t132_leaf0 t132_leaf1 t132_leaf2 t132_leaf3 t132_leaf4 t132_leaf5
 t133_leaf0 t133_leaf1 t133_leaf2 t133_leaf3 t133_leaf4 t133_leaf5
 t133_leaf6 t134_leaf0 t134_leaf1 t134_leaf2 t134_leaf3 t134_leaf4
 t134_leaf5 t134_leaf6 t135_leaf0 t135_leaf1 t135_leaf2 t135_leaf3
 t135_leaf4 t135_leaf5 t135_leaf6 t136_leaf0 t136_leaf1 t136_leaf2
 t136_leaf3 t136_leaf4 t136_leaf5 t136_leaf6 t136_leaf7 t137_leaf0
 t137_leaf1 t137_leaf2 t137_leaf3 t137_leaf4 t137_leaf5 t137_leaf6
 t137_leaf7 t138_leaf0 t138_leaf1 t138_leaf2 t138_leaf3 t138_leaf4
 t138_leaf5 t138_leaf6 t138_leaf7 t139_leaf0 t139_leaf1 t139_leaf2
 t139_leaf3 t139_leaf4 t139_leaf5 t139_leaf6 t139_leaf7 t140_leaf0
 t140_leaf1 t140_leaf2 t140_leaf3 t140_leaf4 t140_leaf5 t140_leaf6
 t141_leaf0 t141_leaf1 t141_leaf2 t141_leaf3 t141_leaf4 t141_leaf5
 t141_leaf6 t142_leaf0 t142_leaf1 t142_leaf2 t142_leaf3 t142_leaf4
 t142_leaf5 t142_leaf6 t143_leaf0 t143_leaf1 t143_leaf2 t143_leaf3
 t143_leaf4 t143_leaf5 t143_leaf6 t143_leaf7 t144_leaf0 t144_leaf1
 t144_leaf2 t144_leaf3 t144_leaf4 t144_leaf5 t144_leaf6 t145_leaf0
 t145_leaf1 t145_leaf2 t145_leaf3 t145_leaf4 t145_leaf5 t146_leaf0
 t146_leaf1 t146_leaf2 t146_leaf3 t146_leaf4 t146_leaf5 t146_leaf6
 t146_leaf7 t147_leaf0 t147_leaf1 t147_leaf2 t147_leaf3 t147_leaf4
 t147_leaf5 t147_leaf6 t147_leaf7 t148_leaf0 t148_leaf1 t148_leaf2
 t148_leaf3 t148_leaf4 t148_leaf5 t148_leaf6 t148_leaf7 t149_leaf0
 t149_leaf1 t149_leaf2 t149_leaf3 t149_leaf4 t149_leaf5 t149_leaf6
 t149_leaf7 t150_leaf0 t150_leaf1 t150_leaf2 t150_leaf3 t150_leaf4
 t150_leaf5 t151_leaf0 t151_leaf1 t151_leaf2 t151_leaf3 t151_leaf4
 t151_leaf5 t151_leaf6 t152_leaf0 t152_leaf1 t152_leaf2 t152_leaf3
 t152_leaf4 t152_leaf5 t153_leaf0 t153_leaf1 t153_leaf2 t153_leaf3
 t153_leaf4 t153_leaf5 t153_leaf6 t153_leaf7 t154_leaf0 t154_leaf1
 t154_leaf2 t154_leaf3 t154_leaf4 t154_leaf5 t154_leaf6 t154_leaf7
 t155_leaf0 t155_leaf1 t155_leaf2 t155_leaf3 t155_leaf4 t155_leaf5
 t156_leaf0 t156_leaf1 t156_leaf2 t156_leaf3 t156_leaf4 t157_leaf0
 t157_leaf1 t157_leaf2 t157_leaf3 t157_leaf4 t157_leaf5 t157_leaf6
 t158_leaf0 t158_leaf1 t158_leaf2 t158_leaf3 t158_leaf4 t158_leaf5
 t159_leaf0 t159_leaf1 t159_leaf2 t159_leaf3 t159_leaf4 t159_leaf5
 t160_leaf0 t160_leaf1 t160_leaf2 t160_leaf3 t160_leaf4 t160_leaf5
 t160_leaf6 t161_leaf0 t161_leaf1 t161_leaf2 t161_leaf3 t161_leaf4
 t161_leaf5 t161_leaf6 t161_leaf7 t162_leaf0 t162_leaf1 t162_leaf2
 t162_leaf3 t162_leaf4 t162_leaf5 t162_leaf6 t162_leaf7 t163_leaf0
 t163_leaf1 t163_leaf2 t163_leaf3 t163_leaf4 t163_leaf5 t163_leaf6
 t163_leaf7 t164_leaf0 t164_leaf1 t164_leaf2 t164_leaf3 t164_leaf4
 t164_leaf5 t164_leaf6 t165_leaf0 t165_leaf1 t165_leaf2 t165_leaf3
 t165_leaf4 t165_leaf5 t166_leaf0 t166_leaf1 t166_leaf2 t166_leaf3
 t166_leaf4 t166_leaf5 t166_leaf6 t167_leaf0 t167_leaf1 t167_leaf2
 t167_leaf3 t167_leaf4 t167_leaf5 t167_leaf6 t168_leaf0 t168_leaf1
 t168_leaf2 t168_leaf3 t168_leaf4 t168_leaf5 t168_leaf6 t168_leaf7
 t169_leaf0 t169_leaf1 t169_leaf2 t169_leaf3 t169_leaf4 t169_leaf5
 t169_leaf6 t170_leaf0 t170_leaf1 t170_leaf2 t170_leaf3 t170_leaf4
 t170_leaf5 t171_leaf0 t171_leaf1 t171_leaf2 t171_leaf3 t171_leaf4
 t171_leaf5 t171_leaf6 t172_leaf0 t172_leaf1 t172_leaf2 t172_leaf3
 t172_leaf4 t172_leaf5 t173_leaf0 t173_leaf1 t173_leaf2 t173_leaf3
 t173_leaf4 t173_leaf5 t173_leaf6 t174_leaf0 t174_leaf1 t174_leaf2
 t174_leaf3 t174_leaf4 t174_leaf5 t174_leaf6 t175_leaf0 t175_leaf1
 t175_leaf2 t175_leaf3 t175_leaf4 t176_leaf0 t176_leaf1 t176_leaf2
 t176_leaf3 t176_leaf4 t177_leaf0 t177_leaf1 t177_leaf2 t177_leaf3
 t177_leaf4 t177_leaf5 t177_leaf6 t178_leaf0 t178_leaf1 t178_leaf2
 t178_leaf3 t178_leaf4 t178_leaf5 t178_leaf6 t179_leaf0 t179_leaf1
 t179_leaf2 t179_leaf3 t179_leaf4 t179_leaf5 t180_leaf0 t180_leaf1
 t180_leaf2 t180_leaf3 t180_leaf4 t181_leaf0 t181_leaf1 t181_leaf2
 t181_leaf3 t181_leaf4 t181_leaf5 t181_leaf6 t181_leaf7 t182_leaf0
 t182_leaf1 t182_leaf2 t182_leaf3 t182_leaf4 t182_leaf5 t182_leaf6
 t183_leaf0 t183_leaf1 t183_leaf2 t183_leaf3 t183_leaf4 t183_leaf5
 t183_leaf6 t184_leaf0 t184_leaf1 t184_leaf2 t184_leaf3 t184_leaf4
 t184_leaf5 t184_leaf6 t184_leaf7 t185_leaf0 t185_leaf1 t185_leaf2
 t185_leaf3 t185_leaf4 t185_leaf5 t186_leaf0 t186_leaf1 t186_leaf2
 t186_leaf3 t186_leaf4 t187_leaf0 t187_leaf1 t187_leaf2 t187_leaf3
 t187_leaf4 t188_leaf0 t188_leaf1 t188_leaf2 t188_leaf3 t188_leaf4
 t188_leaf5 t188_leaf6 t189_leaf0 t189_leaf1 t189_leaf2 t189_leaf3
 t189_leaf4 t189_leaf5 t190_leaf0 t190_leaf1 t190_leaf2 t190_leaf3
 t190_leaf4 t190_leaf5 t191_leaf0 t191_leaf1 t191_leaf2 t191_leaf3
 t191_leaf4 t191_leaf5 t192_leaf0 t192_leaf1 t192_leaf2 t192_leaf3
 t192_leaf4 t192_leaf5 t193_leaf0 t193_leaf1 t193_leaf2 t193_leaf3
 t193_leaf4 t193_leaf5 t194_leaf0 t194_leaf1 t194_leaf2 t194_leaf3
 t194_leaf4 t194_leaf5 t195_leaf0 t195_leaf1 t195_leaf2 t195_leaf3
 t195_leaf4 t196_leaf0 t196_leaf1 t196_leaf2 t196_leaf3 t196_leaf4
 t196_leaf5 t197_leaf0 t197_leaf1 t197_leaf2 t197_leaf3 t197_leaf4
 t197_leaf5 t198_leaf0 t198_leaf1 t198_leaf2 t198_leaf3 t198_leaf4
 t198_leaf5 t198_leaf6 t199_leaf0 t199_leaf1 t199_leaf2 t199_leaf3
 t199_leaf4 t199_leaf5 t199_leaf6 t199_leaf7 t200_leaf0 t200_leaf1
 t200_leaf2 t200_leaf3 t200_leaf4 t200_leaf5 t200_leaf6 t201_leaf0
 t201_leaf1 t201_leaf2 t201_leaf3 t201_leaf4 t201_leaf5 t202_leaf0
 t202_leaf1 t202_leaf2 t202_leaf3 t202_leaf4 t202_leaf5 t202_leaf6
 t203_leaf0 t203_leaf1 t203_leaf2 t203_leaf3 t203_leaf4 t204_leaf0
 t204_leaf1 t204_leaf2 t204_leaf3 t204_leaf4 t205_leaf0 t205_leaf1
 t205_leaf2 t205_leaf3 t205_leaf4 t206_leaf0 t206_leaf1 t206_leaf2
 t206_leaf3 t206_leaf4 t207_leaf0 t207_leaf1 t207_leaf2 t207_leaf3
 t207_leaf4 t207_leaf5 t208_leaf0 t208_leaf1 t208_leaf2 t208_leaf3
 t208_leaf4 t208_leaf5 t209_leaf0 t209_leaf1 t209_leaf2 t209_leaf3
 t209_leaf4 t209_leaf5 t209_leaf6 t209_leaf7 t210_leaf0 t210_leaf1
 t210_leaf2 t210_leaf3 t210_leaf4 t210_leaf5 t210_leaf6 t211_leaf0
 t211_leaf1 t211_leaf2 t211_leaf3 t211_leaf4 t211_leaf5 t211_leaf6
 t212_leaf0 t212_leaf1 t212_leaf2 t212_leaf3 t212_leaf4 t213_leaf0
 t213_leaf1 t213_leaf2 t213_leaf3 t213_leaf4 t213_leaf5 t214_leaf0
 t214_leaf1 t214_leaf2 t214_leaf3 t214_leaf4 t214_leaf5 t215_leaf0
 t215_leaf1 t215_leaf2 t215_leaf3 t215_leaf4 t215_leaf5 t215_leaf6
 t215_leaf7 t216_leaf0 t216_leaf1 t216_leaf2 t216_leaf3 t216_leaf4
 t216_leaf5 t216_leaf6 t217_leaf0 t217_leaf1 t217_leaf2 t217_leaf3
 t217_leaf4 t217_leaf5 t217_leaf6 t218_leaf0 t218_leaf1 t218_leaf2
 t218_leaf3 t218_leaf4 t219_leaf0 t219_leaf1 t219_leaf2 t219_leaf3
 t219_leaf4 t219_leaf5 t219_leaf6 t220_leaf0 t220_leaf1 t220_leaf2
 t220_leaf3 t220_leaf4 t220_leaf5 t220_leaf6 t220_leaf7 t221_leaf0
 t221_leaf1 t221_leaf2 t221_leaf3 t221_leaf4 t221_leaf5 t221_leaf6
 t222_leaf0 t222_leaf1 t222_leaf2 t222_leaf3 t222_leaf4 t223_leaf0
 t223_leaf1 t223_leaf2 t223_leaf3 t223_leaf4 t224_leaf0 t224_leaf1
 t224_leaf2 t224_leaf3 t224_leaf4 t224_leaf5 t225_leaf0 t225_leaf1
 t225_leaf2 t225_leaf3 t225_leaf4 t225_leaf5 t226_leaf0 t226_leaf1
 t226_leaf2 t226_leaf3 t226_leaf4 t226_leaf5 t226_leaf6 t227_leaf0
 t227_leaf1 t227_leaf2 t227_leaf3 t227_leaf4 t228_leaf0 t228_leaf1
 t228_leaf2 t228_leaf3 t228_leaf4 t228_leaf5 t228_leaf6 t228_leaf7
 t229_leaf0 t229_leaf1 t229_leaf2 t229_leaf3 t229_leaf4 t230_leaf0
 t230_leaf1 t230_leaf2 t230_leaf3 t230_leaf4 t230_leaf5 t230_leaf6
 t230_leaf7 t231_leaf0 t231_leaf1 t231_leaf2 t231_leaf3 t231_leaf4
 t231_leaf5 t231_leaf6 t232_leaf0 t232_leaf1 t232_leaf2 t232_leaf3
 t232_leaf4 t233_leaf0 t233_leaf1 t233_leaf2 t233_leaf3 t233_leaf4
 t234_leaf0 t234_leaf1 t234_leaf2 t234_leaf3 t234_leaf4 t234_leaf5
 t235_leaf0 t235_leaf1 t235_leaf2 t235_leaf3 t235_leaf4 t235_leaf5
 t236_leaf0 t236_leaf1 t236_leaf2 t236_leaf3 t236_leaf4 t236_leaf5
 t237_leaf0 t237_leaf1 t237_leaf2 t237_leaf3 t237_leaf4 t237_leaf5
 t237_leaf6 t238_leaf0 t238_leaf1 t238_leaf2 t238_leaf3 t238_leaf4
 t239_leaf0 t239_leaf1 t239_leaf2 t239_leaf3 t239_leaf4 t240_leaf0
 t240_leaf1 t240_leaf2 t240_leaf3 t240_leaf4 t240_leaf5 t240_leaf6
 t241_leaf0 t241_leaf1 t241_leaf2 t241_leaf3 t241_leaf4 t241_leaf5
 t242_leaf0 t242_leaf1 t242_leaf2 t242_leaf3 t242_leaf4 t243_leaf0
 t243_leaf1 t243_leaf2 t243_leaf3 t243_leaf4 t244_leaf0 t244_leaf1
 t244_leaf2 t244_leaf3 t244_leaf4 t245_leaf0 t245_leaf1 t245_leaf2
 t245_leaf3 t245_leaf4 t245_leaf5 t245_leaf6 t246_leaf0 t246_leaf1
 t246_leaf2 t246_leaf3 t246_leaf4 t247_leaf0 t247_leaf1 t247_leaf2
 t247_leaf3 t247_leaf4 t247_leaf5 t248_leaf0 t248_leaf1 t248_leaf2
 t248_leaf3 t248_leaf4 t249_leaf0 t249_leaf1 t249_leaf2 t249_leaf3
 t249_leaf4 t249_leaf5 t250_leaf0 t250_leaf1 t250_leaf2 t250_leaf3
 t250_leaf4 t250_leaf5 t251_leaf0 t251_leaf1 t251_leaf2 t251_leaf3
 t251_leaf4 t252_leaf0 t252_leaf1 t252_leaf2 t252_leaf3 t252_leaf4
 t252_leaf5 t252_leaf6 t252_leaf7 t253_leaf0 t253_leaf1 t253_leaf2
 t253_leaf3 t253_leaf4 t253_leaf5 t253_leaf6 t253_leaf7 t254_leaf0
 t254_leaf1 t254_leaf2 t254_leaf3 t254_leaf4 t255_leaf0 t255_leaf1
 t255_leaf2 t255_leaf3 t255_leaf4 t255_leaf5 t255_leaf6 t256_leaf0
 t256_leaf1 t256_leaf2 t256_leaf3 t256_leaf4 t256_leaf5 t256_leaf6
 t257_leaf0 t257_leaf1 t257_leaf2 t257_leaf3 t257_leaf4 t257_leaf5
 t257_leaf6 t258_leaf0 t258_leaf1 t258_leaf2 t258_leaf3 t258_leaf4
 t258_leaf5 t259_leaf0 t259_leaf1 t259_leaf2 t259_leaf3 t259_leaf4
 t259_leaf5 t259_leaf6 t260_leaf0 t260_leaf1 t260_leaf2 t260_leaf3
 t260_leaf4 t260_leaf5 t260_leaf6 t260_leaf7 t261_leaf0 t261_leaf1
 t261_leaf2 t261_leaf3 t261_leaf4 t262_leaf0 t262_leaf1 t262_leaf2
 t262_leaf3 t262_leaf4 t263_leaf0 t263_leaf1 t263_leaf2 t263_leaf3
 t263_leaf4 t263_leaf5 t263_leaf6 t264_leaf0 t264_leaf1 t264_leaf2
 t264_leaf3 t264_leaf4 t264_leaf5 t264_leaf6 t265_leaf0 t265_leaf1
 t265_leaf2 t265_leaf3 t265_leaf4 t265_leaf5 t266_leaf0 t266_leaf1
 t266_leaf2 t266_leaf3 t266_leaf4 t266_leaf5 t266_leaf6 t267_leaf0
 t267_leaf1 t267_leaf2 t267_leaf3 t267_leaf4 t267_leaf5 t267_leaf6
 t268_leaf0 t268_leaf1 t268_leaf2 t268_leaf3 t268_leaf4 t268_leaf5
 t269_leaf0 t269_leaf1 t269_leaf2 t269_leaf3 t269_leaf4 t270_leaf0
 t270_leaf1 t270_leaf2 t270_leaf3 t270_leaf4 t270_leaf5 t270_leaf6
 t271_leaf0 t271_leaf1 t271_leaf2 t271_leaf3 t271_leaf4 t272_leaf0
 t272_leaf1 t272_leaf2 t272_leaf3 t272_leaf4 t273_leaf0 t273_leaf1
 t273_leaf2 t273_leaf3 t273_leaf4 t273_leaf5 t273_leaf6 t274_leaf0
 t274_leaf1 t274_leaf2 t274_leaf3 t274_leaf4 t274_leaf5 t274_leaf6
 t275_leaf0 t275_leaf1 t275_leaf2 t275_leaf3 t275_leaf4 t276_leaf0
 t276_leaf1 t276_leaf2 t276_leaf3 t276_leaf4 t276_leaf5 t277_leaf0
 t277_leaf1 t277_leaf2 t277_leaf3 t277_leaf4 t277_leaf5 t278_leaf0
 t278_leaf1 t278_leaf2 t278_leaf3 t279_leaf0 t279_leaf1 t279_leaf2
 t279_leaf3 t279_leaf4 t279_leaf5 t280_leaf0 t280_leaf1 t280_leaf2
 t280_leaf3 t280_leaf4 t280_leaf5 t280_leaf6 t281_leaf0 t281_leaf1
 t281_leaf2 t281_leaf3 t281_leaf4 t281_leaf5 t281_leaf6 t282_leaf0
 t282_leaf1 t282_leaf2 t282_leaf3 t282_leaf4 t283_leaf0 t283_leaf1
 t283_leaf2 t283_leaf3 t283_leaf4 t284_leaf0 t284_leaf1 t284_leaf2
 t284_leaf3 t284_leaf4 t285_leaf0 t285_leaf1 t285_leaf2 t285_leaf3
 t285_leaf4 t285_leaf5 t285_leaf6 t286_leaf0 t286_leaf1 t286_leaf2
 t286_leaf3 t286_leaf4 t286_leaf5 t287_leaf0 t287_leaf1 t287_leaf2
 t287_leaf3 t287_leaf4 t287_leaf5 t288_leaf0 t288_leaf1 t288_leaf2
 t288_leaf3 t288_leaf4 t289_leaf0 t289_leaf1 t289_leaf2 t289_leaf3
 t289_leaf4 t290_leaf0 t290_leaf1 t290_leaf2 t290_leaf3 t290_leaf4
 t290_leaf5 t291_leaf0 t291_leaf1 t291_leaf2 t291_leaf3 t292_leaf0
 t292_leaf1 t292_leaf2 t292_leaf3 t292_leaf4 t292_leaf5 t292_leaf6
 t293_leaf0 t293_leaf1 t293_leaf2 t293_leaf3 t294_leaf0 t294_leaf1
 t294_leaf2 t294_leaf3 t294_leaf4 t294_leaf5 t295_leaf0 t295_leaf1
 t295_leaf2 t295_leaf3 t295_leaf4 t296_leaf0 t296_leaf1 t296_leaf2
 t296_leaf3 t296_leaf4 t296_leaf5 t297_leaf0 t297_leaf1 t297_leaf2
 t297_leaf3 t297_leaf4 t297_leaf5 t298_leaf0 t298_leaf1 t298_leaf2
 t298_leaf3 t299_leaf0 t299_leaf1 t299_leaf2 t299_leaf3 t300_leaf0
 t300_leaf1 t300_leaf2 t300_leaf3 t300_leaf4 t300_leaf5 t301_leaf0
 t301_leaf1 t301_leaf2 t301_leaf3 t302_leaf0 t302_leaf1 t302_leaf2
 t302_leaf3 t303_leaf0 t303_leaf1 t303_leaf2 t303_leaf3 t303_leaf4
 t304_leaf0 t304_leaf1 t304_leaf2 t304_leaf3 t304_leaf4 t304_leaf5
 t304_leaf6 t304_leaf7 t305_leaf0 t305_leaf1 t305_leaf2 t305_leaf3
 t305_leaf4 t305_leaf5 t305_leaf6 t306_leaf0 t306_leaf1 t306_leaf2
 t306_leaf3 t306_leaf4 t306_leaf5 t307_leaf0 t307_leaf1 t307_leaf2
 t307_leaf3 t307_leaf4 t307_leaf5 t308_leaf0 t308_leaf1 t308_leaf2
 t308_leaf3 t309_leaf0 t309_leaf1 t309_leaf2 t309_leaf3 t309_leaf4
 t309_leaf5 t309_leaf6 t309_leaf7 t310_leaf0 t310_leaf1 t310_leaf2
 t310_leaf3 t311_leaf0 t311_leaf1 t311_leaf2 t311_leaf3 t311_leaf4
 t312_leaf0 t312_leaf1 t312_leaf2 t312_leaf3 t312_leaf4 t312_leaf5
 t312_leaf6 t313_leaf0 t313_leaf1 t313_leaf2 t313_leaf3 t314_leaf0
 t314_leaf1 t314_leaf2 t314_leaf3 t314_leaf4 t315_leaf0 t315_leaf1
 t315_leaf2 t315_leaf3 t316_leaf0 t316_leaf1 t316_leaf2 t316_leaf3
 t317_leaf0 t317_leaf1 t317_leaf2 t317_leaf3 t317_leaf4 t317_leaf5
 t317_leaf6 t318_leaf0 t318_leaf1 t318_leaf2 t318_leaf3 t318_leaf4
 t319_leaf0 t319_leaf1 t319_leaf2 t319_leaf3 t319_leaf4 t319_leaf5
 t319_leaf6 t320_leaf0 t320_leaf1 t320_leaf2 t320_leaf3 t320_leaf4
 t320_leaf5 t321_leaf0 t321_leaf1 t321_leaf2 t321_leaf3 t322_leaf0
 t322_leaf1 t322_leaf2 t322_leaf3 t322_leaf4 t322_leaf5 t323_leaf0
 t323_leaf1 t323_leaf2 t323_leaf3 t323_leaf4 t324_leaf0 t324_leaf1
 t324_leaf2 t324_leaf3 t324_leaf4 t324_leaf5 t324_leaf6 t325_leaf0
 t325_leaf1 t325_leaf2 t325_leaf3 t325_leaf4 t325_leaf5 t326_leaf0
 t326_leaf1 t326_leaf2 t326_leaf3 t326_leaf4 t326_leaf5 t327_leaf0
 t327_leaf1 t327_leaf2 t327_leaf3 t327_leaf4 t328_leaf0 t328_leaf1
 t328_leaf2 t328_leaf3 t328_leaf4 t329_leaf0 t329_leaf1 t329_leaf2
 t329_leaf3 t329_leaf4 t329_leaf5 t330_leaf0 t330_leaf1 t330_leaf2
 t330_leaf3 t330_leaf4 t330_leaf5 t331_leaf0 t331_leaf1 t331_leaf2
 t331_leaf3 t331_leaf4 t331_leaf5 t332_leaf0 t332_leaf1 t332_leaf2
 t332_leaf3 t332_leaf4 t332_leaf5 t333_leaf0 t333_leaf1 t333_leaf2
 t333_leaf3 t334_leaf0 t334_leaf1 t334_leaf2 t334_leaf3 t334_leaf4
 t334_leaf5 t335_leaf0 t335_leaf1 t335_leaf2 t335_leaf3 t335_leaf4
 t335_leaf5 t335_leaf6 t335_leaf7 t336_leaf0 t336_leaf1 t336_leaf2
 t336_leaf3 t336_leaf4 t337_leaf0 t337_leaf1 t337_leaf2 t337_leaf3
 t337_leaf4 t338_leaf0 t338_leaf1 t338_leaf2 t338_leaf3 t338_leaf4
 t338_leaf5 t339_leaf0 t339_leaf1 t339_leaf2 t339_leaf3 t339_leaf4
 t340_leaf0 t340_leaf1 t340_leaf2 t340_leaf3 t340_leaf4 t340_leaf5
 t340_leaf6 t341_leaf0 t341_leaf1 t341_leaf2 t341_leaf3 t341_leaf4
 t342_leaf0 t342_leaf1 t342_leaf2 t342_leaf3 t342_leaf4 t342_leaf5
 t343_leaf0 t343_leaf1 t343_leaf2 t343_leaf3 t343_leaf4 t343_leaf5
 t344_leaf0 t344_leaf1 t344_leaf2 t344_leaf3 t344_leaf4 t344_leaf5
 t344_leaf6 t344_leaf7 t345_leaf0 t345_leaf1 t345_leaf2 t345_leaf3
 t345_leaf4 t345_leaf5 t345_leaf6 t346_leaf0 t346_leaf1 t346_leaf2
 t346_leaf3 t347_leaf0 t347_leaf1 t347_leaf2 t347_leaf3 t347_leaf4
 t348_leaf0 t348_leaf1 t348_leaf2 t348_leaf3 t348_leaf4 t349_leaf0
 t349_leaf1 t349_leaf2 t349_leaf3 t349_leaf4 t349_leaf5 t350_leaf0
 t350_leaf1 t350_leaf2 t350_leaf3 t350_leaf4 t350_leaf5 t351_leaf0
 t351_leaf1 t351_leaf2 t351_leaf3 t351_leaf4 t351_leaf5 t351_leaf6
 t352_leaf0 t352_leaf1 t352_leaf2 t352_leaf3 t352_leaf4 t352_leaf5
 t352_leaf6 t353_leaf0 t353_leaf1 t353_leaf2 t353_leaf3 t353_leaf4
 t353_leaf5 t353_leaf6 t354_leaf0 t354_leaf1 t354_leaf2 t354_leaf3
 t354_leaf4 t354_leaf5 t355_leaf0 t355_leaf1 t355_leaf2 t355_leaf3
 t355_leaf4 t355_leaf5 t356_leaf0 t356_leaf1 t356_leaf2 t356_leaf3
 t356_leaf4 t357_leaf0 t357_leaf1 t357_leaf2 t357_leaf3 t358_leaf0
 t358_leaf1 t358_leaf2 t358_leaf3 t359_leaf0 t359_leaf1 t359_leaf2
 t360_leaf0 t360_leaf1 t360_leaf2 t360_leaf3 t360_leaf4 t360_leaf5
 t360_leaf6 t361_leaf0 t361_leaf1 t361_leaf2 t361_leaf3 t361_leaf4
 t362_leaf0 t362_leaf1 t362_leaf2 t362_leaf3 t362_leaf4 t363_leaf0
 t363_leaf1 t363_leaf2 t363_leaf3 t363_leaf4 t363_leaf5 t364_leaf0
 t364_leaf1 t364_leaf2 t364_leaf3 t364_leaf4 t364_leaf5 t365_leaf0
 t365_leaf1 t365_leaf2 t365_leaf3 t365_leaf4 t365_leaf5 t366_leaf0
 t366_leaf1 t366_leaf2 t366_leaf3 t366_leaf4 t366_leaf5 t367_leaf0
 t367_leaf1 t367_leaf2 t367_leaf3 t367_leaf4 t368_leaf0 t368_leaf1
 t368_leaf2 t368_leaf3 t368_leaf4 t369_leaf0 t369_leaf1 t369_leaf2
 t369_leaf3 t369_leaf4 t370_leaf0 t370_leaf1 t370_leaf2 t370_leaf3
 t370_leaf4 t371_leaf0 t371_leaf1 t371_leaf2 t371_leaf3 t371_leaf4
 t372_leaf0 t372_leaf1 t372_leaf2 t372_leaf3 t373_leaf0 t373_leaf1
 t373_leaf2 t373_leaf3 t373_leaf4 t374_leaf0 t374_leaf1 t374_leaf2
 t374_leaf3 t374_leaf4 t375_leaf0 t375_leaf1 t375_leaf2 t375_leaf3
 t375_leaf4 t375_leaf5 t376_leaf0 t376_leaf1 t376_leaf2 t376_leaf3
 t376_leaf4 t376_leaf5 t377_leaf0 t377_leaf1 t377_leaf2 t377_leaf3
 t377_leaf4 t377_leaf5 t378_leaf0 t378_leaf1 t378_leaf2 t378_leaf3
 t378_leaf4 t379_leaf0 t379_leaf1 t379_leaf2 t379_leaf3 t379_leaf4
 t379_leaf5 t379_leaf6 t379_leaf7 t380_leaf0 t380_leaf1 t380_leaf2
 t380_leaf3 t381_leaf0 t381_leaf1 t381_leaf2 t381_leaf3 t381_leaf4
 t381_leaf5 t381_leaf6 t382_leaf0 t382_leaf1 t382_leaf2 t382_leaf3
 t382_leaf4 t382_leaf5 t383_leaf0 t383_leaf1 t383_leaf2 t383_leaf3
 t384_leaf0 t384_leaf1 t384_leaf2 t384_leaf3 t385_leaf0 t385_leaf1
 t385_leaf2 t385_leaf3 t386_leaf0 t386_leaf1 t386_leaf2 t386_leaf3
 t387_leaf0 t387_leaf1 t387_leaf2 t387_leaf3 t388_leaf0 t388_leaf1
 t388_leaf2 t388_leaf3 t389_leaf0 t390_leaf0 t390_leaf1 t390_leaf2
 t390_leaf3 t390_leaf4 t391_leaf0 t391_leaf1 t391_leaf2 t391_leaf3
 t392_leaf0 t392_leaf1 t392_leaf2 t392_leaf3 t393_leaf0 t393_leaf1
 t393_leaf2 t393_leaf3 t393_leaf4 t393_leaf5 t394_leaf0 t394_leaf1
 t394_leaf2 t394_leaf3 t395_leaf0 t395_leaf1 t395_leaf2 t395_leaf3
 t395_leaf4 t395_leaf5 t395_leaf6 t396_leaf0 t396_leaf1 t396_leaf2
 t396_leaf3 t396_leaf4 t397_leaf0 t397_leaf1 t397_leaf2 t397_leaf3
 t397_leaf4 t398_leaf0 t398_leaf1 t398_leaf2 t398_leaf3 t398_leaf4
 t399_leaf0 t399_leaf1 t399_leaf2 t399_leaf3 t399_leaf4 t400_leaf0
 t400_leaf1 t401_leaf0 t401_leaf1 t402_leaf0 t402_leaf1 t402_leaf2
 t402_leaf3 t403_leaf0 t403_leaf1 t403_leaf2 t403_leaf3 t403_leaf4
 t403_leaf5 t404_leaf0 t404_leaf1 t404_leaf2 t404_leaf3 t405_leaf0
 t405_leaf1 t405_leaf2 t405_leaf3 t405_leaf4 t406_leaf0 t406_leaf1
 t406_leaf2 t406_leaf3 t407_leaf0 t407_leaf1 t407_leaf2 t407_leaf3
 t407_leaf4 t408_leaf0 t408_leaf1 t408_leaf2 t408_leaf3 t408_leaf4
 t409_leaf0 t409_leaf1 t409_leaf2 t409_leaf3 t409_leaf4 t409_leaf5
 t409_leaf6 t410_leaf0 t410_leaf1 t410_leaf2 t410_leaf3 t410_leaf4
 t411_leaf0 t411_leaf1 t412_leaf0 t412_leaf1 t412_leaf2 t412_leaf3
 t412_leaf4 t412_leaf5 t412_leaf6 t412_leaf7 t413_leaf0 t413_leaf1
 t413_leaf2 t413_leaf3 t414_leaf0 t414_leaf1 t414_leaf2 t414_leaf3
 t415_leaf0 t415_leaf1 t415_leaf2 t415_leaf3 t415_leaf4 t416_leaf0
 t416_leaf1 t416_leaf2 t416_leaf3 t416_leaf4 t417_leaf0 t417_leaf1
 t417_leaf2 t417_leaf3 t418_leaf0 t418_leaf1 t418_leaf2 t418_leaf3
 t418_leaf4 t418_leaf5 t419_leaf0 t419_leaf1 t419_leaf2 t419_leaf3
 t419_leaf4 t420_leaf0 t420_leaf1 t420_leaf2 t420_leaf3 t420_leaf4
 t420_leaf5 t421_leaf0 t421_leaf1 t421_leaf2 t421_leaf3 t421_leaf4
 t422_leaf0 t422_leaf1 t422_leaf2 t422_leaf3 t423_leaf0 t423_leaf1
 t423_leaf2 t423_leaf3 t423_leaf4 t423_leaf5 t424_leaf0 t424_leaf1
 t424_leaf2 t424_leaf3 t424_leaf4 t424_leaf5 t425_leaf0 t425_leaf1
 t426_leaf0 t426_leaf1 t427_leaf0 t427_leaf1 t427_leaf2 t427_leaf3
 t427_leaf4 t427_leaf5 t428_leaf0 t428_leaf1 t428_leaf2 t428_leaf3
 t428_leaf4 t428_leaf5 t428_leaf6 t429_leaf0 t429_leaf1 t429_leaf2
 t429_leaf3 t430_leaf0 t430_leaf1 t430_leaf2 t430_leaf3 t430_leaf4
 t430_leaf5 t431_leaf0 t431_leaf1 t431_leaf2 t432_leaf0 t432_leaf1
 t432_leaf2 t432_leaf3 t433_leaf0 t433_leaf1 t433_leaf2 t433_leaf3
 t433_leaf4 t434_leaf0 t434_leaf1 t434_leaf2 t434_leaf3 t435_leaf0
 t435_leaf1 t435_leaf2 t435_leaf3 t435_leaf4 t436_leaf0 t436_leaf1
 t436_leaf2 t437_leaf0 t437_leaf1 t437_leaf2 t438_leaf0 t438_leaf1
 t438_leaf2 t438_leaf3 t438_leaf4 t439_leaf0 t439_leaf1 t439_leaf2
 t439_leaf3 t439_leaf4 t440_leaf0 t440_leaf1 t440_leaf2 t440_leaf3
 t441_leaf0 t441_leaf1 t441_leaf2 t441_leaf3 t441_leaf4 t441_leaf5
 t441_leaf6 t441_leaf7 t442_leaf0 t442_leaf1 t442_leaf2 t442_leaf3
 t442_leaf4 t443_leaf0 t443_leaf1 t443_leaf2 t443_leaf3 t444_leaf0
 t444_leaf1 t444_leaf2 t444_leaf3 t445_leaf0 t445_leaf1 t445_leaf2
 t445_leaf3 t445_leaf4 t446_leaf0 t446_leaf1 t446_leaf2 t446_leaf3
 t446_leaf4 t447_leaf0 t447_leaf1 t447_leaf2 t447_leaf3 t447_leaf4
 t448_leaf0 t448_leaf1 t448_leaf2 t448_leaf3 t448_leaf4 t448_leaf5
 t449_leaf0 t450_leaf0 t451_leaf0 t451_leaf1 t451_leaf2 t451_leaf3
 t452_leaf0 t452_leaf1 t452_leaf2 t452_leaf3 t453_leaf0 t453_leaf1
 t453_leaf2 t453_leaf3 t453_leaf4 t454_leaf0 t454_leaf1 t454_leaf2
 t454_leaf3 t454_leaf4 t454_leaf5 t455_leaf0 t455_leaf1 t455_leaf2
 t455_leaf3 t456_leaf0 t457_leaf0 t457_leaf1 t458_leaf0 t458_leaf1
 t458_leaf2 t458_leaf3 t458_leaf4 t458_leaf5 t458_leaf6 t459_leaf0
 t459_leaf1 t459_leaf2 t459_leaf3 t459_leaf4 t460_leaf0 t461_leaf0
 t461_leaf1 t461_leaf2 t461_leaf3 t461_leaf4 t461_leaf5 t462_leaf0
 t462_leaf1 t463_leaf0 t463_leaf1 t463_leaf2 t463_leaf3 t463_leaf4
 t463_leaf5 t463_leaf6 t464_leaf0 t464_leaf1 t464_leaf2 t465_leaf0
 t465_leaf1 t465_leaf2 t465_leaf3 t465_leaf4 t466_leaf0 t466_leaf1
 t466_leaf2 t466_leaf3 t467_leaf0 t468_leaf0 t468_leaf1 t469_leaf0
 t469_leaf1 t469_leaf2 t469_leaf3 t469_leaf4 t470_leaf0 t470_leaf1
 t470_leaf2 t470_leaf3 t471_leaf0 t471_leaf1 t471_leaf2 t471_leaf3
 t472_leaf0 t472_leaf1 t472_leaf2 t472_leaf3 t473_leaf0 t473_leaf1
 t473_leaf2 t473_leaf3 t474_leaf0 t474_leaf1 t474_leaf2 t475_leaf0
 t475_leaf1 t475_leaf2 t475_leaf3 t475_leaf4 t476_leaf0 t477_leaf0
 t477_leaf1 t477_leaf2 t477_leaf3 t477_leaf4 t478_leaf0 t478_leaf1
 t478_leaf2 t478_leaf3 t479_leaf0 t479_leaf1 t479_leaf2 t479_leaf3
 t480_leaf0 t480_leaf1 t481_leaf0 t481_leaf1 t481_leaf2 t481_leaf3
 t482_leaf0 t482_leaf1 t482_leaf2 t483_leaf0 t484_leaf0 t484_leaf1
 t485_leaf0 t486_leaf0 t486_leaf1 t487_leaf0 t487_leaf1 t488_leaf0
 t488_leaf1 t488_leaf2 t488_leaf3 t489_leaf0 t489_leaf1 t489_leaf2
 t489_leaf3 t490_leaf0 t490_leaf1 t491_leaf0 t491_leaf1 t491_leaf2
 t491_leaf3 t491_leaf4 t491_leaf5 t491_leaf6 t492_leaf0 t492_leaf1
 t492_leaf2 t493_leaf0 t493_leaf1 t493_leaf2 t493_leaf3 t493_leaf4
 t493_leaf5 t494_leaf0 t494_leaf1 t494_leaf2 t494_leaf3 t494_leaf4
 t495_leaf0 t495_leaf1 t496_leaf0 t496_leaf1 t496_leaf2 t496_leaf3
 t497_leaf0 t497_leaf1 t497_leaf2 t497_leaf3 t498_leaf0 t498_leaf1
 t498_leaf2 t499_leaf0 t499_leaf1 t499_leaf2 t499_leaf3 t500_leaf0
 t500_leaf1 t500_leaf2 t500_leaf3 t500_leaf4 t501_leaf0 t501_leaf1
 t501_leaf2 t501_leaf3 t502_leaf0 t502_leaf1 t502_leaf2 t503_leaf0
 t503_leaf1 t503_leaf2 t503_leaf3 t503_leaf4 t504_leaf0 t504_leaf1
 t504_leaf2 t505_leaf0 t505_leaf1 t506_leaf0 t506_leaf1 t506_leaf2
 t507_leaf0 t507_leaf1 t508_leaf0 t508_leaf1 t508_leaf2 t509_leaf0
 t509_leaf1 t509_leaf2 t509_leaf3 t510_leaf0 t511_leaf0 t512_leaf0
 t513_leaf0 t513_leaf1 t513_leaf2 t513_leaf3 t513_leaf4 t513_leaf5
 t514_leaf0 t514_leaf1 t515_leaf0 t515_leaf1 t515_leaf2 t515_leaf3
 t516_leaf0 t516_leaf1 t516_leaf2 t516_leaf3 t516_leaf4 t517_leaf0
 t517_leaf1 t518_leaf0 t518_leaf1 t518_leaf2 t518_leaf3 t519_leaf0
 t519_leaf1 t519_leaf2 t519_leaf3 t520_leaf0 t520_leaf1 t520_leaf2
 t520_leaf3 t520_leaf4 t520_leaf5 t521_leaf0 t521_leaf1 t521_leaf2
 t521_leaf3 t521_leaf4 t522_leaf0 t522_leaf1 t522_leaf2 t522_leaf3
 t522_leaf4 t523_leaf0 t523_leaf1 t523_leaf2 t523_leaf3 t523_leaf4
 t523_leaf5 t524_leaf0 t524_leaf1 t524_leaf2 t524_leaf3 t524_leaf4
 t524_leaf5 t525_leaf0 t526_leaf0 t526_leaf1 t526_leaf2 t527_leaf0
 t527_leaf1 t528_leaf0 t528_leaf1 t528_leaf2 t528_leaf3 t528_leaf4
 t528_leaf5 t529_leaf0 t530_leaf0 t531_leaf0 t531_leaf1 t532_leaf0
 t532_leaf1 t532_leaf2 t532_leaf3 t532_leaf4 t532_leaf5 t533_leaf0
 t534_leaf0 t535_leaf0 t535_leaf1 t535_leaf2 t535_leaf3 t535_leaf4
 t535_leaf5 t536_leaf0 t537_leaf0 t537_leaf1 t537_leaf2 t537_leaf3
 t538_leaf0 t538_leaf1 t538_leaf2 t539_leaf0 t539_leaf1 t539_leaf2
 t539_leaf3 t539_leaf4 t540_leaf0 t540_leaf1 t540_leaf2 t540_leaf3
 t540_leaf4 t541_leaf0 t541_leaf1 t542_leaf0 t543_leaf0 t543_leaf1
 t543_leaf2 t543_leaf3 t543_leaf4 t544_leaf0 t544_leaf1 t544_leaf2
 t544_leaf3 t544_leaf4 t544_leaf5 t545_leaf0 t545_leaf1 t545_leaf2
 t546_leaf0 t546_leaf1 t546_leaf2 t547_leaf0 t547_leaf1 t547_leaf2
 t547_leaf3 t547_leaf4 t548_leaf0 t549_leaf0 t549_leaf1 t549_leaf2
 t549_leaf3 t550_leaf0 t550_leaf1 t550_leaf2 t550_leaf3 t551_leaf0
 t552_leaf0 t552_leaf1 t552_leaf2 t552_leaf3 t553_leaf0 t554_leaf0
 t555_leaf0 t556_leaf0 t557_leaf0 t557_leaf1 t558_leaf0 t558_leaf1
 t558_leaf2 t558_leaf3 t559_leaf0 t559_leaf1 t560_leaf0 t560_leaf1
 t560_leaf2 t560_leaf3 t560_leaf4 t561_leaf0 t561_leaf1 t561_leaf2
 t562_leaf0 t562_leaf1 t562_leaf2 t563_leaf0 t564_leaf0 t565_leaf0
 t565_leaf1 t566_leaf0 t566_leaf1 t566_leaf2 t566_leaf3 t566_leaf4
 t567_leaf0 t567_leaf1 t567_leaf2 t567_leaf3 t568_leaf0 t568_leaf1
 t568_leaf2 t568_leaf3 t569_leaf0 t569_leaf1 t569_leaf2 t570_leaf0
 t571_leaf0 t571_leaf1 t571_leaf2 t571_leaf3 t571_leaf4 t572_leaf0
 t572_leaf1 t572_leaf2 t572_leaf3 t573_leaf0 t573_leaf1 t574_leaf0
 t575_leaf0 t575_leaf1 t575_leaf2 t575_leaf3 t575_leaf4 t576_leaf0
 t576_leaf1 t576_leaf2 t576_leaf3 t577_leaf0 t578_leaf0 t579_leaf0
 t579_leaf1 t579_leaf2 t579_leaf3 t580_leaf0 t580_leaf1 t580_leaf2
 t580_leaf3 t580_leaf4 t581_leaf0 t582_leaf0 t582_leaf1 t582_leaf2
 t582_leaf3 t582_leaf4 t583_leaf0 t584_leaf0 t584_leaf1 t585_leaf0
 t585_leaf1 t585_leaf2 t585_leaf3 t585_leaf4 t586_leaf0 t586_leaf1
 t587_leaf0 t588_leaf0 t588_leaf1 t588_leaf2 t588_leaf3 t589_leaf0
 t589_leaf1 t590_leaf0 t591_leaf0 t592_leaf0 t592_leaf1 t592_leaf2
 t593_leaf0 t594_leaf0 t594_leaf1 t595_leaf0 t596_leaf0 t597_leaf0
 t597_leaf1 t598_leaf0 t598_leaf1 t599_leaf0 t599_leaf1 t599_leaf2
 t599_leaf3 t599_leaf4 t600_leaf0 t600_leaf1 t600_leaf2 t600_leaf3
 t601_leaf0 t602_leaf0 t602_leaf1 t602_leaf2 t602_leaf3 t602_leaf4
 t603_leaf0 t603_leaf1 t603_leaf2 t603_leaf3 t604_leaf0 t604_leaf1
 t604_leaf2 t605_leaf0 t605_leaf1 t606_leaf0 t607_leaf0 t607_leaf1
 t607_leaf2 t608_leaf0 t608_leaf1 t608_leaf2 t608_leaf3 t609_leaf0
 t609_leaf1 t609_leaf2 t609_leaf3 t610_leaf0 t611_leaf0 t611_leaf1
 t612_leaf0 t613_leaf0 t613_leaf1 t613_leaf2 t613_leaf3 t613_leaf4
 t614_leaf0 t614_leaf1 t615_leaf0 t615_leaf1 t616_leaf0 t616_leaf1
 t617_leaf0 t617_leaf1 t617_leaf2 t617_leaf3 t618_leaf0 t618_leaf1
 t619_leaf0 t619_leaf1 t619_leaf2 t620_leaf0 t620_leaf1 t620_leaf2
 t620_leaf3 t621_leaf0 t621_leaf1 t622_leaf0 t622_leaf1 t622_leaf2
 t622_leaf3 t623_leaf0 t623_leaf1 t623_leaf2 t623_leaf3 t624_leaf0
 t624_leaf1 t625_leaf0 t625_leaf1 t626_leaf0 t626_leaf1 t626_leaf2
 t626_leaf3 t627_leaf0 t627_leaf1 t627_leaf2 t627_leaf3 t628_leaf0
 t628_leaf1 t628_leaf2 t628_leaf3 t629_leaf0 t630_leaf0 t631_leaf0
 t631_leaf1 t631_leaf2 t631_leaf3 t632_leaf0 t632_leaf1 t632_leaf2
 t632_leaf3 t632_leaf4 t633_leaf0 t634_leaf0 t634_leaf1 t634_leaf2
 t635_leaf0 t635_leaf1 t635_leaf2 t636_leaf0 t637_leaf0 t637_leaf1
 t638_leaf0 t638_leaf1 t639_leaf0 t640_leaf0 t640_leaf1 t640_leaf2
 t641_leaf0 t642_leaf0 t643_leaf0 t643_leaf1 t644_leaf0 t645_leaf0
 t646_leaf0 t646_leaf1 t646_leaf2 t646_leaf3 t647_leaf0 t647_leaf1
 t647_leaf2 t647_leaf3 t647_leaf4 t647_leaf5 t648_leaf0 t649_leaf0
 t649_leaf1 t650_leaf0 t650_leaf1 t650_leaf2 t651_leaf0 t652_leaf0
 t653_leaf0 t653_leaf1 t654_leaf0 t655_leaf0 t655_leaf1 t655_leaf2
 t656_leaf0 t656_leaf1 t657_leaf0 t657_leaf1 t658_leaf0 t659_leaf0
 t659_leaf1 t659_leaf2 t659_leaf3 t659_leaf4 t660_leaf0 t660_leaf1
 t660_leaf2 t660_leaf3 t661_leaf0 t661_leaf1 t661_leaf2 t662_leaf0
 t663_leaf0 t663_leaf1 t663_leaf2 t663_leaf3 t663_leaf4 t664_leaf0
 t665_leaf0 t665_leaf1 t665_leaf2 t665_leaf3 t665_leaf4 t666_leaf0
 t667_leaf0 t668_leaf0 t668_leaf1 t668_leaf2 t669_leaf0 t670_leaf0
 t670_leaf1 t670_leaf2 t670_leaf3 t671_leaf0 t671_leaf1 t671_leaf2
 t672_leaf0 t673_leaf0 t673_leaf1 t673_leaf2 t673_leaf3 t673_leaf4
 t674_leaf0 t675_leaf0 t676_leaf0 t677_leaf0 t678_leaf0 t678_leaf1
 t679_leaf0 t680_leaf0 t680_leaf1 t680_leaf2 t680_leaf3 t681_leaf0
 t681_leaf1 t681_leaf2 t681_leaf3 t682_leaf0 t682_leaf1 t682_leaf2
 t683_leaf0 t683_leaf1 t684_leaf0 t685_leaf0 t686_leaf0 t687_leaf0
 t687_leaf1 t687_leaf2 t687_leaf3 t687_leaf4 t687_leaf5 t688_leaf0
 t688_leaf1 t688_leaf2 t688_leaf3 t689_leaf0 t689_leaf1 t690_leaf0
 t691_leaf0 t692_leaf0 t692_leaf1 t693_leaf0 t693_leaf1 t694_leaf0
 t695_leaf0 t696_leaf0 t696_leaf1 t697_leaf0 t698_leaf0 t698_leaf1
 t699_leaf0 t699_leaf1 t699_leaf2 t699_leaf3 t699_leaf4 t700_leaf0
 t700_leaf1 t700_leaf2 t701_leaf0 t701_leaf1 t702_leaf0 t702_leaf1
 t702_leaf2 t702_leaf3 t702_leaf4 t702_leaf5 t703_leaf0 t703_leaf1
 t703_leaf2 t703_leaf3 t703_leaf4 t704_leaf0 t705_leaf0 t705_leaf1
 t705_leaf2 t706_leaf0 t706_leaf1 t706_leaf2 t706_leaf3 t706_leaf4
 t707_leaf0 t708_leaf0 t708_leaf1 t708_leaf2 t708_leaf3 t708_leaf4
 t709_leaf0 t710_leaf0 t710_leaf1 t710_leaf2 t711_leaf0 t712_leaf0
 t712_leaf1 t712_leaf2 t712_leaf3 t712_leaf4 t713_leaf0 t713_leaf1
 t713_leaf2 t713_leaf3 t713_leaf4 t714_leaf0 t714_leaf1 t715_leaf0
 t715_leaf1 t715_leaf2 t715_leaf3 t715_leaf4 t716_leaf0 t717_leaf0
 t718_leaf0 t718_leaf1 t718_leaf2 t718_leaf3 t718_leaf4 t719_leaf0
 t720_leaf0 t721_leaf0 t721_leaf1 t721_leaf2 t721_leaf3 t721_leaf4
 t722_leaf0 t723_leaf0 t723_leaf1 t723_leaf2 t723_leaf3 t723_leaf4
 t724_leaf0 t724_leaf1 t724_leaf2 t725_leaf0 t726_leaf0 t726_leaf1
 t727_leaf0 t727_leaf1 t727_leaf2 t728_leaf0 t729_leaf0 t730_leaf0
 t731_leaf0 t732_leaf0 t733_leaf0 t734_leaf0 t735_leaf0 t735_leaf1
 t735_leaf2 t735_leaf3 t736_leaf0 t737_leaf0 t738_leaf0 t738_leaf1
 t738_leaf2 t738_leaf3 t738_leaf4 t739_leaf0 t740_leaf0 t741_leaf0
 t741_leaf1 t741_leaf2 t741_leaf3 t741_leaf4 t742_leaf0 t742_leaf1
 t742_leaf2 t743_leaf0 t744_leaf0 t745_leaf0 t746_leaf0 t747_leaf0
 t748_leaf0 t749_leaf0 t749_leaf1 t749_leaf2 t750_leaf0 t751_leaf0
 t751_leaf1 t751_leaf2 t751_leaf3 t751_leaf4 t751_leaf5 t752_leaf0
 t753_leaf0 t754_leaf0 t755_leaf0 t756_leaf0 t756_leaf1 t756_leaf2
 t756_leaf3 t757_leaf0 t757_leaf1 t757_leaf2 t758_leaf0 t758_leaf1
 t758_leaf2 t758_leaf3 t758_leaf4 t758_leaf5 t759_leaf0 t760_leaf0
 t761_leaf0 t761_leaf1 t761_leaf2 t762_leaf0 t763_leaf0 t764_leaf0
 t764_leaf1 t764_leaf2 t765_leaf0 t766_leaf0 t766_leaf1 t766_leaf2
 t766_leaf3 t766_leaf4 t767_leaf0 t768_leaf0 t768_leaf1 t768_leaf2
 t769_leaf0 t770_leaf0 t771_leaf0 t772_leaf0 t773_leaf0 t774_leaf0
 t774_leaf1 t775_leaf0 t775_leaf1 t776_leaf0 t777_leaf0 t777_leaf1
 t778_leaf0 t778_leaf1 t778_leaf2 t778_leaf3 t779_leaf0 t780_leaf0
 t781_leaf0 t781_leaf1 t781_leaf2 t781_leaf3 t782_leaf0 t783_leaf0
 t784_leaf0 t784_leaf1 t784_leaf2 t784_leaf3 t785_leaf0 t786_leaf0
 t786_leaf1 t786_leaf2 t787_leaf0 t788_leaf0 t788_leaf1 t789_leaf0
 t790_leaf0 t790_leaf1 t791_leaf0 t792_leaf0 t793_leaf0 t794_leaf0
 t794_leaf1 t795_leaf0 t795_leaf1 t795_leaf2 t795_leaf3 t795_leaf4
 t795_leaf5 t796_leaf0 t796_leaf1 t796_leaf2 t796_leaf3 t796_leaf4
 t797_leaf0 t798_leaf0 t798_leaf1 t799_leaf0 t799_leaf1 t799_leaf2
 t799_leaf3 t800_leaf0 t801_leaf0 t802_leaf0 t803_leaf0 t804_leaf0
 t805_leaf0 t806_leaf0 t806_leaf1 t806_leaf2 t806_leaf3 t807_leaf0
 t807_leaf1 t808_leaf0 t808_leaf1 t808_leaf2 t808_leaf3 t809_leaf0
 t809_leaf1 t810_leaf0 t811_leaf0 t812_leaf0 t813_leaf0 t814_leaf0
 t814_leaf1 t814_leaf2 t814_leaf3 t814_leaf4 t815_leaf0 t815_leaf1
 t816_leaf0 t817_leaf0 t818_leaf0 t818_leaf1 t819_leaf0 t819_leaf1
 t819_leaf2 t819_leaf3 t820_leaf0 t821_leaf0 t822_leaf0 t822_leaf1
 t822_leaf2 t822_leaf3 t823_leaf0 t824_leaf0 t824_leaf1 t824_leaf2
 t825_leaf0 t826_leaf0 t826_leaf1 t826_leaf2 t827_leaf0 t827_leaf1
 t827_leaf2 t828_leaf0 t828_leaf1 t828_leaf2 t828_leaf3 t829_leaf0
 t829_leaf1 t829_leaf2 t829_leaf3 t829_leaf4 t829_leaf5 t829_leaf6
 t829_leaf7 t830_leaf0 t831_leaf0 t832_leaf0 t832_leaf1 t833_leaf0
 t833_leaf1 t834_leaf0 t834_leaf1 t834_leaf2 t834_leaf3 t835_leaf0
 t835_leaf1 t835_leaf2 t836_leaf0 t836_leaf1 t837_leaf0 t838_leaf0
 t839_leaf0 t839_leaf1 t840_leaf0 t841_leaf0 t841_leaf1 t842_leaf0
 t843_leaf0 t844_leaf0 t844_leaf1 t845_leaf0 t846_leaf0 t847_leaf0
 t847_leaf1 t848_leaf0 t848_leaf1 t848_leaf2 t849_leaf0 t850_leaf0
 t851_leaf0 t852_leaf0 t853_leaf0 t854_leaf0 t855_leaf0 t855_leaf1
 t856_leaf0 t857_leaf0 t858_leaf0 t858_leaf1 t859_leaf0 t860_leaf0
 t860_leaf1 t861_leaf0 t862_leaf0 t862_leaf1 t862_leaf2 t863_leaf0
 t863_leaf1 t863_leaf2 t863_leaf3 t864_leaf0 t865_leaf0 t865_leaf1
 t865_leaf2 t865_leaf3 t866_leaf0 t866_leaf1 t866_leaf2 t866_leaf3
 t867_leaf0 t867_leaf1 t867_leaf2 t868_leaf0 t869_leaf0 t870_leaf0
 t871_leaf0 t872_leaf0 t873_leaf0 t873_leaf1 t874_leaf0 t875_leaf0
 t876_leaf0 t876_leaf1 t876_leaf2 t876_leaf3 t877_leaf0 t877_leaf1
 t877_leaf2 t878_leaf0 t879_leaf0 t879_leaf1 t880_leaf0 t880_leaf1
 t880_leaf2 t880_leaf3 t880_leaf4 t880_leaf5 t881_leaf0 t881_leaf1
 t881_leaf2 t881_leaf3 t881_leaf4 t882_leaf0 t883_leaf0 t884_leaf0
 t884_leaf1 t884_leaf2 t884_leaf3 t884_leaf4 t885_leaf0 t886_leaf0
 t886_leaf1 t886_leaf2 t886_leaf3 t886_leaf4 t886_leaf5 t887_leaf0
 t887_leaf1 t888_leaf0 t888_leaf1 t888_leaf2 t888_leaf3 t888_leaf4
 t888_leaf5 t889_leaf0 t889_leaf1 t889_leaf2 t889_leaf3 t889_leaf4
 t889_leaf5 t890_leaf0 t890_leaf1 t890_leaf2 t891_leaf0 t891_leaf1
 t892_leaf0 t893_leaf0 t894_leaf0 t894_leaf1 t895_leaf0 t896_leaf0
 t896_leaf1 t897_leaf0 t898_leaf0 t899_leaf0 t900_leaf0 t901_leaf0
 t902_leaf0 t903_leaf0 t903_leaf1 t904_leaf0 t904_leaf1 t904_leaf2
 t905_leaf0 t905_leaf1 t905_leaf2 t906_leaf0 t906_leaf1 t906_leaf2
 t906_leaf3 t906_leaf4 t907_leaf0 t907_leaf1 t907_leaf2 t908_leaf0
 t909_leaf0 t909_leaf1 t909_leaf2 t909_leaf3 t910_leaf0 t911_leaf0
 t911_leaf1 t911_leaf2 t912_leaf0 t913_leaf0 t914_leaf0 t915_leaf0
 t915_leaf1 t915_leaf2 t916_leaf0 t916_leaf1 t916_leaf2 t917_leaf0
 t918_leaf0 t919_leaf0 t920_leaf0 t921_leaf0 t922_leaf0 t922_leaf1
 t922_leaf2 t923_leaf0 t923_leaf1 t923_leaf2 t923_leaf3 t923_leaf4
 t923_leaf5 t924_leaf0 t924_leaf1 t924_leaf2 t924_leaf3 t925_leaf0
 t925_leaf1 t925_leaf2 t926_leaf0 t927_leaf0 t928_leaf0 t929_leaf0
 t930_leaf0 t931_leaf0 t931_leaf1 t931_leaf2 t932_leaf0 t932_leaf1
 t933_leaf0 t934_leaf0 t935_leaf0 t935_leaf1 t935_leaf2 t935_leaf3
 t936_leaf0 t937_leaf0 t937_leaf1 t937_leaf2 t938_leaf0 t939_leaf0
 t939_leaf1 t940_leaf0 t940_leaf1 t940_leaf2 t940_leaf3 t941_leaf0
 t941_leaf1 t942_leaf0 t943_leaf0 t944_leaf0 t945_leaf0 t946_leaf0
 t947_leaf0 t947_leaf1 t947_leaf2 t947_leaf3 t947_leaf4 t948_leaf0
 t949_leaf0 t950_leaf0 t951_leaf0 t951_leaf1 t951_leaf2 t952_leaf0
 t953_leaf0 t954_leaf0 t954_leaf1 t954_leaf2 t954_leaf3 t954_leaf4
 t955_leaf0 t956_leaf0 t956_leaf1 t957_leaf0 t957_leaf1 t957_leaf2
 t957_leaf3 t957_leaf4
Generals
 x_Utilities x_Exter_Qual x_Exter_Cond x_Bsmt_Qual x_Bsmt_Cond x_Heating_QC
 x_Full_Bath x_Half_Bath x_Bedroom_AbvGr x_Kitchen_AbvGr x_Kitchen_Qual
 x_Fireplaces x_Fireplace_Qu x_Garage_Cars x_Garage_Qual x_Garage_Cond
 x_Pool_QC FullBath1 FullBath2 HalfBath1 HalfBath2 Kitchen1 Kitchen2
 Bedroom1 Bedroom2 OtherRooms1 OtherRooms2 OtherRooms FullBath HalfBath
 Kitchen Bedrooms OverallQualInt BsmtFullBath BsmtHalfBath
General Constraints
 PWL_expm1: y_price = PWL ( y_log_cal ) : (10.3, 29731.61885289144) 
  (10.30875, 29992.92079669966) (10.3175, 30256.51916472054) 
  (10.32625, 30522.43413883279) (10.335, 30790.68607828162) 
  (10.34375, 31061.29552123707) (10.3525, 31334.28318636691) 
  (10.36125, 31609.6699744225) (10.37, 31887.47696983943) 
  (10.37875, 32167.72544235138) (10.3875, 32450.43684861904) 
  (10.39625, 32735.63283387243) (10.405, 33023.33523356861) 
  (10.41375, 33313.56607506298) (10.4225, 33606.34757929621) 
  (10.43125, 33901.70216249515) (10.44, 34199.65243788947) 
  (10.44875, 34500.22121744253) (10.4575, 34803.43151359847) 
  (10.46625, 35109.3065410436) (10.475, 35417.86971848422) 
  (10.48375, 35729.14467043922) (10.4925, 36043.15522904929) 
  (10.50125, 36359.92543590108) (10.51, 36679.47954386842) 
  (10.51875, 37001.84201896868) (10.5275, 37327.03754223638) 
  (10.53625, 37655.0910116127) (10.545, 37986.02754385149) 
  (10.55375, 38319.87247644281) (10.5625, 38656.65136955225) 
  (10.57125, 38996.39000797846) (10.58, 39339.11440312676) 
  (10.58875, 39684.85079500121) (10.5975, 40033.62565421306) 
  (10.60625, 40385.46568400799) (10.615, 40740.39782231001) 
  (10.62375, 41098.44924378446) (10.6325, 41459.64736191807) 
  (10.64125, 41824.01983111834) (10.65, 42191.5945488303) 
  (10.65875, 42562.39965767296) (10.6675, 42936.46354759346) 
  (10.67625, 43313.81485804125) (10.685, 43694.48248016021) 
  (10.69375, 44078.49555900126) (10.7025, 44465.88349575319) 
  (10.71125, 44856.67594999429) (10.72, 45250.90284196256) 
  (10.72875, 45648.59435484718) (10.7375, 46049.78093709869) 
  (10.74625, 46454.49330476086) (10.755, 46862.76244382212) 
  (10.76375, 47274.61961258769) (10.7725, 47690.0963440734) 
  (10.78125, 48109.22444841931) (10.79, 48532.03601532579) 
  (10.79875, 48958.56341650982) (10.8075, 49388.83930818408) 
  (10.81625, 49822.89663355652) (10.825, 50260.76862535333) 
  (10.83375, 50702.4888083626) (10.8425, 51148.09100200179) 
  (10.85125, 51597.60932290634) (10.86, 52051.07818754247) 
  (10.86875, 52508.53231484144) (10.8775, 52970.00672885853) 
  (10.88625, 53435.5367614538) (10.895, 53905.15805499796) 
  (10.90375, 54378.90656510051) (10.9125, 54856.81856336336) 
  (10.92125, 55338.93064015716) (10.93, 55825.27970742346) 
  (10.93875, 56315.90300150012) (10.9475, 56810.83808597294) 
  (10.95625, 57310.12285455091) (10.965, 57813.79553396811) 
  (10.97375, 58321.89468691019) (10.9825, 58834.45921496666) 
  (10.99125, 59351.52836160945) (11, 59873.14171519782) 
  (11.00875, 60399.33921200904) (11.0175, 60930.16113929578) 
  (11.02625, 61465.64813837138) (11.035, 62005.84120772055) 
  (11.04375, 62550.78170613927) (11.0525, 63100.51135590042) 
  (11.06125, 63655.07224594908) (11.07, 64214.50683512404) 
  (11.07875, 64778.85795540956) (11.0875, 65348.16881521373) 
  (11.09625, 65922.48300267757) (11.105, 66501.84448901139) 
  (11.11375, 67086.29763186222) (11.1225, 67675.88717870908) 
  (11.13125, 68270.65827028987) (11.14, 68870.65644405661) 
  (11.14875, 69475.92763766281) (11.1575, 70086.51819247965) 
  (11.16625, 70702.474857145) (11.175, 71323.84479114162) 
  (11.18375, 71950.67556840874) (11.1925, 72583.01518098409) 
  (11.20125, 73220.912042678) (11.21, 73864.41499278042) 
  (11.21875, 74513.57329980044) (11.2275, 75168.43666523819) 
  (11.23625, 75829.05522738969) (11.245, 76495.4795651866) 
  (11.25375, 77167.76070206752) (11.2625, 77845.9501098857) 
  (11.27125, 78530.09971284868) (11.28, 79220.26189149488) 
  (11.28875, 79916.48948670299) (11.2975, 80618.83580373865) 
  (11.30625, 81327.35461633453) (11.315, 82042.10017080864) 
  (11.32375, 82763.12719021633) (11.3325, 83490.49087854127) 
  (11.34125, 84224.24692492088) (11.35, 84964.45150791117) 
  (11.35875, 85711.16129978682) (11.3675, 86464.43347088133) 
  (11.37625, 87224.32569396294) (11.385, 87990.89614865149) 
  (11.39375, 88764.20352587158) (11.4025, 89544.30703234716) 
  (11.41125, 90331.2663951342) (11.42, 91125.14186619317) 
  (11.42875, 91925.99422700243) (11.4375, 92733.88479321224) 
  (11.44625, 93548.87541933889) (11.455, 94371.02850349997) 
  (11.46375, 95200.40699219292) (11.4725, 96037.07438511308) 
  (11.48125, 96881.09474001679) (11.49, 97732.53267762439) 
  (11.49875, 98591.45338656926) (11.5075, 99457.92262838739) 
  (11.51625, 100332.0067425537) (11.525, 101213.7726515598) 
  (11.53375, 102103.2878660393) (11.5425, 103000.6204899348) 
  (11.55125, 103905.8392257141) (11.56, 104819.0133796285) 
  (11.56875, 105740.2128670205) (11.5775, 106669.5082176756) 
  (11.58625, 107606.9705812234) (11.595, 108552.6717325835) 
  (11.60375, 109506.6840774628) (11.6125, 110469.0806578971) 
  (11.62125, 111439.9351578448) (11.63, 112419.321908828) 
  (11.63875, 113407.3158956232) (11.6475, 114403.9927620023) 
  (11.65625, 115409.4288165249) (11.665, 116423.7010383804) 
  (11.67375, 117446.8870832808) (11.6825, 118479.0652894078) 
  (11.69125, 119520.3146834092) (11.7, 120570.7149864507) 
  (11.70875, 121630.3466203183) (11.7175, 122699.2907135769) 
  (11.72625, 123777.6291077804) (11.735, 124865.4443637393) 
  (11.74375, 125962.8197678401) (11.7525, 127069.8393384236) 
  (11.76125, 128186.5878322161) (11.77, 129313.1507508201) 
  (11.77875, 130449.6143472588) (11.7875, 131596.065632582) 
  (11.79625, 132752.5923825258) (11.805, 133919.2831442349) 
  (11.81375, 135096.2272430401) (11.8225, 136283.5147892995) 
  (11.83125, 137481.2366852953) (11.84, 138689.4846321954) 
  (11.84875, 139908.3511370738) (11.8575, 141137.9295199923) 
  (11.86625, 142378.3139211464) (11.875, 143629.5993080729) 
  (11.88375, 144891.8814829206) (11.8925, 146165.257089785) 
  (11.90125, 147449.8236221078) (11.91, 148745.6794301418) 
  (11.91875, 150052.9237284806) (11.9275, 151371.6566036538) 
  (11.93625, 152701.9790217919) (11.945, 154043.9928363544) 
  (11.95375, 155397.80079593) (11.9625, 156763.5065521013) 
  (11.97125, 158141.2146673831) (11.98, 159531.0306232254) 
  (11.98875, 160933.0608280917) (11.9975, 162347.4126256038) 
  (12.00625, 163774.1943027629) (12.015, 165213.5150982373) 
  (12.02375, 166665.4852107292) (12.0325, 168130.215807409) 
  (12.04125, 169607.819032429) (12.05, 171098.4080155075) 
  (12.05875, 172602.0968805921) (12.0675, 174119.0007545955) 
  (12.07625, 175649.2357762122) (12.085, 177192.9191048078) 
  (12.09375, 178750.1689293917) (12.1025, 180321.1044776644) 
  (12.11125, 181905.846025146) (12.12, 183504.5149043851) 
  (12.12875, 185117.2335142489) (12.1375, 186744.1253292941) 
  (12.14625, 188385.3149092193) (12.155, 190040.9279084042) 
  (12.16375, 191711.0910855271) (12.1725, 193395.9323132727) 
  (12.18125, 195095.5805881199) (12.19, 196810.1660402206) 
  (12.19875, 198539.8199433605) (12.2075, 200284.674725012) 
  (12.21625, 202044.863976471) (12.225, 203820.5224630876) 
  (12.23375, 205611.7861345814) (12.2425, 207418.792135453) 
  (12.25125, 209241.6788154812) (12.26, 211080.5857403187) 
  (12.26875, 212935.6537021745) (12.2775, 214807.0247305962) 
  (12.28625, 216694.8421033418) (12.295, 218599.2503573516) 
  (12.30375, 220520.3952998122) (12.3125, 222458.424019322) 
  (12.32125, 224413.484897152) (12.33, 226385.7276186052) 
  (12.33875, 228375.3031844781) (12.3475, 230382.3639226222) 
  (12.35625, 232407.063499606) (12.365, 234449.5569324796) 
  (12.37375, 236510.0006006437) (12.3825, 238588.5522578236) 
  (12.39125, 240685.3710441464) (12.4, 242800.6174983236) 
  (12.40875, 244934.4535699462) (12.4175, 247087.0426318799) 
  (12.42625, 249258.549492777) (12.435, 251449.1404096915) 
  (12.44375, 253658.9831008114) (12.4525, 255888.2467582964) 
  (12.46125, 258137.102061235) (12.47, 260405.7211887091) 
  (12.47875, 262694.2778329797) (12.4875, 265002.9472127818) 
  (12.49625, 267331.9060867432) (12.505, 269681.3327669156) 
  (12.51375, 272051.4071324266) (12.5225, 274442.3106432519) 
  (12.53125, 276854.2263541101) (12.54, 279287.3389284763) 
  (12.54875, 281741.8346527202) (12.5575, 284217.9014503691) 
  (12.56625, 286715.7288964973) (12.575, 289235.5082322392) 
  (12.58375, 291777.4323794306) (12.5925, 294341.6959553802) 
  (12.60125, 296928.495287771) (12.61, 299538.0284296914) 
  (12.61875, 302170.4951747959) (12.6275, 304826.0970726068) 
  (12.63625, 307505.0374439406) (12.645, 310207.5213964791) 
  (12.65375, 312933.755840469) (12.6625, 315683.9495045677) 
  (12.67125, 318458.3129518199) (12.68, 321257.0585957834) 
  (12.68875, 324080.4007167877) (12.6975, 326928.5554783442) 
  (12.70625, 329801.7409436919) (12.715, 332700.177092497) 
  (12.72375, 335624.0858376938) (12.7325, 338573.6910424731) 
  (12.74125, 341549.2185374238) (12.75, 344550.8961378237) 
  (12.75875, 347578.953661081) (12.7675, 350633.6229443285) 
  (12.77625, 353715.1378621745) (12.785, 356823.7343446111) 
  (12.79375, 359959.6503950753) (12.8025, 363123.1261086712) 
  (12.81125, 366314.4036905527) (12.82, 369533.7274744691) 
  (12.82875, 372781.343941471) (12.8375, 376057.5017387791) 
  (12.84625, 379362.4516988266) (12.855, 382696.4468584579) 
  (12.86375, 386059.7424783072) (12.8725, 389452.5960623371) 
  (12.88125, 392875.2673775589) (12.89, 396328.0184739155) 
  (12.89875, 399811.113704351) (12.9075, 403324.8197450438) 
  (12.91625, 406869.4056158305) (12.925, 410445.1427007969) 
  (12.93375, 414052.3047690611) (12.9425, 417691.1679957316) 
  (12.95125, 421362.010983051) (12.96, 425065.114781728) 
  (12.96875, 428800.7629124568) (12.9775, 432569.241387623) 
  (12.98625, 436370.8387331998) (12.995, 440205.8460108404) 
  (13.00375, 444074.5568401637) (13.0125, 447977.2674212329) 
  (13.02125, 451914.2765572326) (13.03, 455885.8856773473) 
  (13.03875, 459892.3988598409) (13.0475, 463934.122855337) 
  (13.05625, 468011.3671103017) (13.065, 472124.4437907416) 
  (13.07375, 476273.6678060975) (13.0825, 480459.3568333616) 
  (13.09125, 484681.8313413931) (13.1, 488941.4146154607) 
  (13.10875, 493238.4327819881) (13.1175, 497573.2148335291) 
  (13.12625, 501946.0926539505) (13.135, 506357.4010438482) 
  (13.14375, 510807.4777461739) (13.1525, 515296.6634721007) 
  (13.16125, 519825.3019271018) (13.17, 524393.739837273) 
  (13.17875, 529002.3269758715) (13.1875, 533651.4161901023) 
  (13.19625, 538341.3634281303) (13.205, 543072.527766331) 
  (13.21375, 547845.2714367841) (13.2225, 552659.9598550092) 
  (13.23125, 557516.9616479413) (13.24, 562416.6486821518) 
  (13.24875, 567359.3960923217) (13.2575, 572345.5823099649) 
  (13.26625, 577375.5890924006) (13.275, 582449.801551978) 
  (13.28375, 587568.6081855692) (13.2925, 592732.4009043056) 
  (13.30125, 597941.5750635917) (13.31, 603196.5294933666) 
  (13.31875, 608497.666528648) (13.3275, 613845.3920403274) 
  (13.33625, 619240.1154662538) (13.345, 624682.2498425726) 
  (13.35375, 630172.2118353576) (13.3625, 635710.4217725033) 
  (13.37125, 641297.3036759158) (13.38, 646933.2852939675) 
  (13.38875, 652618.7981342564) (13.3975, 658354.2774966345) 
  (13.40625, 664140.1625065432) (13.415, 669976.8961486311) 
  (13.42375, 675864.9253006676) (13.4325, 681804.7007677598) 
  (13.44125, 687796.6773168696) (13.45, 693841.3137116296) 
  (13.45875, 699939.0727474655) (13.4675, 706090.4212870307) 
  (13.47625, 712295.8302959539) (13.485, 718555.7748788948) 
  (13.49375, 724870.7343159161) (13.5025, 731241.1920991869) 
  (13.51125, 737667.6359699905) (13.52, 744150.5579560776) 
  (13.52875, 750690.454409327) (13.5375, 757287.8260437582) 
  (13.54625, 763943.1779738569) (13.555, 770657.0197532591) 
  (13.56375, 777429.8654137531) (13.5725, 784262.2335046455) 
  (13.58125, 791154.6471324534) (13.59, 798107.6340009647) 
  (13.59875, 805121.7264516301) (13.6075, 812197.461504332) 
  (13.61625, 819335.380898489) (13.625, 826536.0311345428) 
  (13.63375, 833799.963515796) (13.6425, 841127.7341906185) 
  (13.65125, 848519.9041950306) (13.66, 855977.0394956605) 
  (13.66875, 863499.7110330741) (13.6775, 871088.4947654842) 
  (13.68625, 878743.9717128506) (13.695, 886466.7280013685) 
  (13.70375, 894257.35490834) (13.7125, 902116.4489074418) 
  (13.72125, 910044.6117143957) (13.73, 918042.450333041) 
  (13.73875, 926110.5771018054) (13.7475, 934249.6097405824) 
  (13.75625, 942460.1713980377) (13.765, 950742.8906993055) 
  (13.77375, 959098.4017941316) (13.7825, 967527.3444054126) 
  (13.79125, 976030.3638781877) (13.8, 984608.1112290357)
End
