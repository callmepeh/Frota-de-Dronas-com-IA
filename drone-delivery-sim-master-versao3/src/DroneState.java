/** The single source of truth for a drone's operational status. */
enum DroneState {
    AVAILABLE,
    ACTIVE,
    RECHARGING,
    RENTING_TRANSIT,
    RETURNING_TRANSIT,
    RETURNED
}
