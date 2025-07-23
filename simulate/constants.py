RIDER_ARRIVAL = 'rider_arrival'
RIDER_LOST = 'rider_lost'
RIDE_COMPLETION = 'ride_completion'
RELOCATION_START = 'relocation_start'
RELOCATION_COMPLETION = 'relocation_completion'
RIDE_START = 'ride_start'
TIME_BLOCK_BOUNDARY = 'time_block_boundary'
TAXI_INIT = 'taxi_init'


### Vehicle status constants

IS_RELOCATING = 'IS_RELOCATING'
IS_ON_TRIP = 'IS_ON_TRIP'
IS_IDLE = 'IS_IDLE'

MAX_TAXI_ZONE_ID = 265
location_ids = range(1, MAX_TAXI_ZONE_ID+1)
excluded_location_ids = [
    # We exclude the following locations:
    # 1. Middle of nowheres
    # 2. EWR Airport
    # 3. Islands (except Roosevelt Island)
    
    # Staten Island
    5,
    6,
    23,
    44,
    84,
    99, 
    109,
    110,
    115,
    118,
    156,
    172,
    176,
    187,
    204,
    206,
    214,
    221,
    245,
    251,
    
    # Ellis Island
    103,
    104,
    105,
    46, # Bronx City Island
    1, # EWR Airport (Ridesharing app pickups and dropoffs at EWR are banned.)
    2, # Jamaica Bay
    194, # Randalls Island
    264, # Unknown
    265, # Outside NYC
    179, # Rikers Island
    199, # Rikers Island
    ]

location_ids = [id_ for id_ in location_ids if id_ not in excluded_location_ids]
# Create a mapping from location IDs to indices
location_id_to_index = {id_: i for i, id_ in enumerate(location_ids)}
num_locations = len(location_ids)

taxi_type = 'fhvhv' # 'green', 'fhv', 'fhvhv'