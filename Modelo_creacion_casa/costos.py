# costos.py  (añade esto dentro de COSTOS)

COSTOS = {
    "AMPLIACION_PSF": 106.49,
    "DEMOLICION_PSF": 1.65,

    "UTILITIES": {
        "AllPub": 31750, "NoSewr": 39500, "NoSeWa": 22000, "ELO": 20000,
    },

    "ROOFMATL_PSF": {
        "ClyTile": 8.00, "CompShg": 6.00, "Membran": 7.00, "Metal": 8.99,
        "Roll": 3.75, 'TarGrv': 5.5, "WdShake": 6.00, "WdShngl": 6.35,
    },

    # Revestimiento exterior (USD/ft²). Faltantes los agregamos luego.
        "EXTERIOR_PSF": {
        "AsbShng": 11.50,
        "AsphShn": 1.50,
        "BrkComm": 1.21,
        "BrkFace": 15.00,
        "CBlock": 0,      # Modificar
        "CemntBd": 0,     # Modificar
        "HdBoard": 0,     # Modificar
        "ImStucc": 0,     # Modificar
        "MetalSd": 5.48,
        "Plywood": 0,     # Modificar
        "PreCast": 0,     # Modificar
        "Stone": 27.50,
        "Stucco": 12.00,
        "VinylSd": 7.46,
        "WdSdng": 3.64,
        "WdShing": 0,     # Modificar
    },

        "MASVNRTYPE_PSF": {
        "BrkCmn": 1.21,
        "BrkFace": 15.00,
        "CBlock": 0,    # Modificar
        "None": 0,
        "Stone": 27.50,
    },

        "HEATING": {
        "Floor": 1773,
        "GasA": 5750,
        "GasW": 8500,
        "Grav": 6300,
        "OthW": 4900,
        "Wall": 3700,
    },

        "HEATING_QC": {
        "Ex": 0,   # Modificar
        "Gd": 0,   # Modificar
        "TA": 0,   # Modificar
        "Fa": 0,   # Modificar
        "Po": 0,   # Modificar
    },

    "CENTRALAIR": {
        "Yes": 5362,
        "No": 0,
    },

    "ELECTRICAL": {
        "SBrkr": 0,   # Modificar
        "FuseA": 0,   # Modificar
        "FuseF": 0,   # Modificar
        "FuseP": 0,   # Modificar
        "Mix": 0,     # Modificar
    },

        
    "MISCFEATURE": {
        "Elev": 48000,
        "Gar2": 32100,
        "Shed": 5631,
        "TenC": 15774,
        "Noaplica": 0,
    },


    "PAVEDDRIVE_PSF": {
        "Paved": 8.50,    # Pavement
        "PartialPavement": 5.25,    # Partial pavement
        "Dirt/Gravel": 2.00,    # Dirt/Gravel
    },

    "BASEMENT_PSF": {
        "Bsmt": 15.00,
    },

        "BASEMENTCOND": {
        "Ex": 0,   # Modificar
        "Gd": 0,   # Modificar
        "TA": 0,   # Modificar
        "Fa": 0,   # Modificar
        "Po": 0,   # Modificar
        "NA": 0,   # Modificar
    },

    "KITCHEN_PSF": {
        "Kitchen": 200,   # USD/ft²
    },

        "KITCHEN_REMODEL": {
        "Ex": 180000,
        "Gd": 111250,   # Interpolado entre Ex y TA (60k–300k aprox)
        "TA": 42500,
        "Fa": 27750,    # Interpolado entre TA y Po
        "Po": 13000,
    },

    "HALFBATH_CONSTR": {
        "HalfBath": 10000,
    },

    "FULLBATH_CONSTR": {
        "FullBath": 25000,
    },

        "BATH_REMODEL_PSF": {
        "BathRemodel": 650,   # USD/ft² promedio (Cedreo, 2025)
    },

    "FIREPLACE_QU": {
        "Ex": 0,      # Modificar
        "Gd": 4350,   # Promedio entre 3500–5600
        "TA": 2500,   # Promedio entre 2000–3000
        "Fa": 0,      # Modificar
        "Po": 3000,   # Promedio entre 1500–4000
        "NA": 0,
    },

        "BEDROOM_CONSTR": {
        "Bedroom": 100,   # Modificar: USD por dormitorio construido
    },

        "GARAGE_QU": {
        "Ex": 0,   # Modificar
        "Gd": 0,   # Modificar
        "TA": 0,   # Modificar
        "Fa": 0,   # Modificar
        "Po": 0,   # Modificar
        "NA": 0,
    },

    "GARAGE_FINISH": {
        "Fin": 0,   # Modificar
        "RFn": 0,   # Modificar
        "Unf": 0,   # Modificar
        "NA": 0, #Modificar
    },

        "GARAGE_CARS_AREA": {
        "1car": 18750,   # Promedio entre 10,500–27,000
        "2car": 0,       # Modificar (falta fuente)
        "3car": 0,       # Modificar (falta fuente)
        "4car": 0,       # Modificar (falta fuente)
    },

    "POOL_COSTS": {
        "PoolArea": 88,        # USD/ft² promedio (Loveland, 2025)
        "Ex": 135000,
        "Gd": 96333,           # Interpolada
        "TA": 57667,           # Interpolada
        "Fa": 19000,
    },

        "FENCE_COSTS": {
        "Fence": 40.00,      # Promedio general por pie lineal (Grupa, 2025)
        "GdPrv": 42.00,      # Buena privacidad, 8 pies (Moore, 2025)
        "MnPrv": 31.33,      # Privacidad menor, 6 pies (Moore, 2025)
        "GdWo": 12.00,       # Madera, 10–14 por pie (Grupa, 2025)
        "MnWw": 2.00,        # Alambre, 2 por pie (Grupa, 2025)
        "NA": 0.00,
    },















}
