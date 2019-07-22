BODIES = {
    'object': ['friction', 'rolling_friction', 'torsion_friction', 'mass'],
    'table': ['friction', 'mass'],
    'franka': ['stiffness', 'damping']
}

RANDOM_PARAMS = {
    "table": {
        "friction": {
            "active": False,
            "min_value": 0.0,
            "max_value": 1.0
        },
        "mass": {
            "active": False,
            "min_value": 0.0,
            "max_value": 1.0
        }
    },

    "object": {
        "friction": {
            "active": False,
            "min_value": 0.0,
            "max_value": 1.0
        },
        "rolling_friction": {
            "active": False,
            "min_value": 0.01,
            "max_value": 1.0
        },
        "torsion_friction": {
            "active": False,
            "min_value": 0.01,
            "max_value": 1.0
        },
        "mass": {
            "active": True,
            "min_value": 0.0,
            "max_value": 2.0
        }
    },

    "franka":{
        "damping": {
            "active": False,
            "min_value": 1000,
            "max_value": 5000
        },
        "stiffness": {
            "active": False,
            "min_value": 0,
            "max_value": 50000
        },
    }
}

