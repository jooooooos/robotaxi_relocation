from dataclasses import dataclass, field
import heapq
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from constants import (
    RIDER_ARRIVAL,
    RIDER_LOST,
    RIDE_COMPLETION,
    RELOCATION_START,
    RELOCATION_COMPLETION,
    RIDE_START,
    TIME_BLOCK_BOUNDARY,
    TAXI_INIT,
    
    IS_RELOCATING,
    IS_ON_TRIP,
    IS_IDLE,
)
from relocation_policies import (
    relocation_policy_blind_sampling,
    relocation_policy_jlcr_eta,
)

@dataclass(order=True)
class Event:
    """
    Represents a discrete event in the simulation with an associated time and type.
    Events are ordered by time and processed chronologically via a priority queue.
    """

    time: float
    priority: int
    event_type: str = field(compare=False)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)

# EventQueue manages the priority queue of events
class EventQueue:
    """
    A simple priority queue to manage simulation events using a min-heap.
    Events are executed in time-priority order.
    """
    def __init__(self):
        self.events: List[Event] = []

    def push(self, event: Event):
        heapq.heappush(self.events, event)

    def pop(self) -> Optional[Event]:
        if self.events:
            return heapq.heappop(self.events)
        return None

    def peek(self) -> Optional[Event]:
        return self.events[0] if self.events else None

    def is_empty(self) -> bool:
        return len(self.events) == 0


# Vehicle Class
@dataclass
class Vehicle:
    vehicle_id: int
    location: int
    status: str
    available_time: float = 0.0
    target_location: Optional[int] = None

# Simulator Core
class TaxiSimulator:
    def __init__(self, T: int, R: int, N: int, lambda_: np.ndarray,
                 mu_: np.ndarray, P: np.ndarray, Q: np.ndarray, 
                 relocation_policy: Callable,
                 relocation_kwargs: Optional[Dict[str, Any]] = None,
                 use_real_demand=False,
                 demand_events=None,
                 start_time=0.0,
                 ):
        """
        Initialize the Taxi Simulator with system parameters.

        Args:
            T (int): Number of time blocks in a day (e.g., 48 for 30-min blocks).
            R (int): Number of regions.
            N (int): Number of vehicles.
            lambda_ (np.ndarray): Rider arrival rates, shape (T, R).
            mu_ (np.ndarray): Travel time parameters, shape (T, R, R).
            P (np.ndarray): Rider destination probabilities, shape (T, R, R).
            Q (np.ndarray): Vehicle relocation probabilities, shape (T, R, R).
        """
        self.T = T
        self.R = R
        self.N = N
        self.lambda_ = lambda_
        self.mu_ = mu_
        self.P = P
        self.Q = Q

        self.vehicles: List[Vehicle] = []
        self.clock: float = start_time
        self.event_queue = EventQueue()
        self.logger: List[Dict[str, Any]] = []

        # Vehicle queues per region (FIFO for single-server queue)
        self.idle_queues: Dict[int, List[Vehicle]] = {i: [] for i in range(R)}
        self.next_arrival: Dict[int, float] = {}
        
        self.relocation_policy = relocation_policy
        self.relocation_kwargs = relocation_kwargs or {}
        
        self.use_real_demand = use_real_demand
        self.demand_events = demand_events
        
    def initialize(self, max_time: float):
        """
        Initialize simulation state:
        - Randomly distribute vehicles across regions.
        - Schedule first rider arrivals.
        - Schedule periodic time block boundary events for resampling arrivals.

        Args:
            max_time (float): The simulation horizon in hours.
        """
        
        # Randomly distribute vehicles initially
        for i in range(self.N):
            region = np.random.choice(self.R)
            vehicle = Vehicle(vehicle_id=i, location=region, status=IS_IDLE)
            self.vehicles.append(vehicle)
            self.idle_queues[region].append(vehicle)
        
        # Record initial vehicle distribution
        for region in range(self.R):
            self.event_queue.push(Event(
                time=0.0,
                priority=-1,
                event_type=TAXI_INIT,
                data={'num_vehicles': len(self.idle_queues[region]), 'region': region}
            ))

        # Queue all arrival events if real demand is used
        if self.use_real_demand and self.demand_events is not None:
            for event in self.demand_events:
                self.event_queue.push(Event(
                    time=event['t_sim'],
                    priority=0,
                    event_type=RIDER_ARRIVAL,
                    data={
                        'region': event['pu_idx'],
                        'destination': event['do_idx'],
                        'trip_time': event['trip_time_hr'],
                    }
                ))
        
        # Otherwise, schedule first rider arrivals for each region
        else:
            for i in range(self.R):
                self.schedule_next_rider_arrival(i)
                for block in range(1, self.T):
                    t = block * (24 / self.T)
                    while t < max_time:
                        self.event_queue.push(Event(
                            time=t,
                            priority=5,
                            event_type=TIME_BLOCK_BOUNDARY,
                            data={'region': i}
                        ))
                        t += 24

    def run(self, max_time: float):
        """
        Run the simulation until reaching max_time or until no more events.

        Args:
            max_time (float): The simulation horizon in hours.
        """
        self.initialize(max_time)
        while not self.event_queue.is_empty() and self.clock < max_time:
            event = self.event_queue.pop()
            if event:
                self.clock = event.time
                self.handle_event(event)

    def get_time_block(self, time: float) -> int:
        """
        Map simulation clock time to the corresponding time block index.

        Args:
            time (float): Current simulation time in hours.

        Returns:
            int: Time block index (0 to T-1).
        """
        return int((time % 24) / (24 / self.T))

    def schedule_next_rider_arrival(self, region: int):
        """
        Schedule the next rider arrival event for a given region,
        ensuring only the earliest arrival is kept in the system.

        Args:
            region (int): Region index where arrival is scheduled.
        """
        time_block = self.get_time_block(self.clock)
        rate = self.lambda_[time_block, region]
        if rate > 0:
            arrival_interval = np.random.exponential(1 / rate)
            event_time = self.clock + arrival_interval
            if (region not in self.next_arrival) or (event_time < self.next_arrival[region]):
                self.next_arrival[region] = event_time
                self.event_queue.push(Event(
                    time=event_time,
                    priority=0,
                    event_type=RIDER_ARRIVAL,
                    data={'region': region, 'origin_time_block': time_block}
                ))

    def handle_event(self, event: Event):
        """
        Dispatch incoming events to their appropriate handler.

        Args:
            event (Event): The event to process.
        """
        if event.event_type == RIDER_ARRIVAL:
            self.handle_rider_arrival(event)
        elif event.event_type == RIDE_START:
            self.handle_ride_start(event)
        elif event.event_type == RIDE_COMPLETION:
            self.handle_ride_completion(event)
        elif event.event_type == RELOCATION_START:
            self.handle_relocation_start(event)
        elif event.event_type == RELOCATION_COMPLETION:
            self.handle_relocation_completion(event)
        elif event.event_type == TIME_BLOCK_BOUNDARY:
            self.handle_time_block_boundary(event)
            
        self.logger.append({
            'time': event.time,
            'event_type': event.event_type,
            'data': event.data
        })

    def handle_rider_arrival(self, event: Event):
        """
        Process a rider arrival:
        - Assigns a vehicle if available.
        - Logs lost riders if no vehicle is idle.
        - Resets next arrival scheduling.

        Args:
            event (Event): The rider arrival event.
        """
        region = event.data['region']
        current_time = event.time
        if not self.use_real_demand and self.next_arrival.get(region, float('inf')) > current_time:
            # If the next arrival is scheduled in the future, ignore this event
            # Implemented to avoid double scheduling, when arrival rate changes abruptly across time blocks
            return

        time_block = self.get_time_block(current_time)
        if self.idle_queues[region]:
            vehicle = self.idle_queues[region].pop(0)
            
            if self.use_real_demand:
                destination = event.data['destination']
                travel_time = event.data['trip_time']
            else:
                destination = np.random.choice(self.R, p=self.P[time_block, region])
                travel_time = np.random.exponential(1 / self.mu_[time_block, region, destination])

            vehicle.status = IS_ON_TRIP
            vehicle.target_location = destination
            
            self.event_queue.push(Event(
                time=current_time,
                priority=1,
                event_type=RIDE_START,
                data={'vehicle_id': vehicle.vehicle_id, 'origin': region,
                      'destination': destination, 'travel_time': travel_time}
            ))
        else:
            self.logger.append({'time': current_time, 'event_type': RIDER_LOST, 'data': {'region': region}})

        if not self.use_real_demand:
            self.next_arrival.pop(region, None)
            self.schedule_next_rider_arrival(region)

    def handle_ride_start(self, event: Event):
        """
        Process the start of a ride by scheduling its completion.

        Args:
            event (Event): The ride start event.
        """
        vehicle = self.vehicles[event.data['vehicle_id']]
        vehicle.location = event.data['origin']
        vehicle.available_time = event.time + event.data['travel_time']

        self.event_queue.push(Event(
            time=vehicle.available_time,
            priority=2,
            event_type=RIDE_COMPLETION,
            data={'vehicle_id': vehicle.vehicle_id,
                  'origin': event.data['origin'],
                  'destination': event.data['destination']}
        ))

    def handle_ride_completion(self, event: Event):
        """
        Process ride completion:
        - Decide on vehicle relocation based on Q matrix.
        - If no relocation, return vehicle to idle queue.

        Args:
            event (Event): The ride completion event.
        """
        vehicle = self.vehicles[event.data['vehicle_id']]
        vehicle.location = event.data['destination']
        time_block = self.get_time_block(event.time)
        vehicle.status = IS_IDLE
        vehicle.target_location = None
        
        
        relocation_dest = self.relocation_policy(vehicle, event.time, self, **self.relocation_kwargs)
        if relocation_dest != vehicle.location:
            travel_time = np.random.exponential(1 / self.mu_[time_block, vehicle.location, relocation_dest])
            self.event_queue.push(Event(
                time=event.time,
                priority=3,
                event_type=RELOCATION_START,
                data={'vehicle_id': vehicle.vehicle_id, 'origin': vehicle.location,
                      'destination': relocation_dest, 'travel_time': travel_time}
            ))
        else:
            self.idle_queues[vehicle.location].append(vehicle)

    def handle_relocation_start(self, event: Event):
        """
        Start vehicle relocation by scheduling relocation completion.

        Args:
            event (Event): The relocation start event.
        """
        vehicle = self.vehicles[event.data['vehicle_id']]
        vehicle.available_time = event.time + event.data['travel_time']
        vehicle.status = IS_RELOCATING
        vehicle.target_location = event.data['destination']

        self.event_queue.push(Event(
            time=vehicle.available_time,
            priority=4,
            event_type=RELOCATION_COMPLETION,
            data={'vehicle_id': vehicle.vehicle_id,
                  'origin': event.data['origin'],
                  'destination': event.data['destination']}
        ))

    def handle_relocation_completion(self, event: Event):
        """
        Complete relocation and place vehicle back in the idle queue.

        Args:
            event (Event): The relocation completion event.
        """
        vehicle = self.vehicles[event.data['vehicle_id']]
        vehicle.location = event.data['destination']
        vehicle.status = IS_IDLE
        vehicle.target_location = None
        
        self.idle_queues[vehicle.location].append(vehicle)

    def handle_time_block_boundary(self, event: Event):
        """
        Resample the rider arrival time at the boundary of a time block
        to adapt to sudden changes in arrival rates (λ).
        Only schedules a new arrival if it's sooner than the existing one.

        Args:
            event (Event): Time block boundary event for a specific region.
        """
        region = event.data['region']
        current_time_block = self.get_time_block(event.time)
        rate = self.lambda_[current_time_block, region]
        if rate > 0:
            new_arrival_interval = np.random.exponential(1 / rate)
            new_arrival_time = self.clock + new_arrival_interval
            if region not in self.next_arrival or new_arrival_time < self.next_arrival[region]:
                self.next_arrival[region] = new_arrival_time
                self.event_queue.push(Event(
                    time=new_arrival_time,
                    priority=0,
                    event_type=RIDER_ARRIVAL,
                    data={'region': region, 'origin_time_block': current_time_block}
                ))


# import ace_tools as tools; tools.display_dataframe_to_user(name="Logger", dataframe=[])
